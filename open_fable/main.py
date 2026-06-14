"""
open_fable/main.py
==================
OpenFable — Recurrent-Depth Transformer for narrative reasoning.

Architecture summary
--------------------

                  ┌─────────────────────────────────────────────┐
  token ids ──►  │  Embedding + RoPE precompute                 │
                  └────────────────┬────────────────────────────┘
                                   │
                  ┌────────────────▼────────────────────────────┐
                  │  PRELUDE  (n_prelude standard xfmr layers)  │
                  └────────────────┬────────────────────────────┘
                                   │ e  (encoded input, fixed for all loops)
                  ┌────────────────▼────────────────────────────┐
                  │  RECURRENT BLOCK  ×T loops                  │
                  │                                             │
                  │  h_{t+1} = A·h_t  +  B·e  +  C·m           │
                  │          + Transformer(h_t, e, m)           │
                  │                                             │
                  │  where m = FableMemory.read(h_t)            │
                  │  CoherenceProbe.record_step() each loop     │
                  │  ACT halting gates early exit               │
                  │  LoRA depth adapters per loop index         │
                  └────────────────┬────────────────────────────┘
                                   │
                  ┌────────────────▼────────────────────────────┐
                  │  CODA  (n_coda standard xfmr layers)        │
                  └────────────────┬────────────────────────────┘
                                   │
                             lm_head  →  logits

FableConfig extends MythosConfig-equivalent fields with:
  - FableMemoryConfig  (memory_dim, max_characters, etc.)
  - NarrativeDepthController settings
  - CoherenceProbe settings

References
----------
- OpenMythos (MIT): recurrent-depth transformer base architecture
- arXiv:2603.21676  "Thinking Deeper, Not Longer": silent objective,
  LayerScale init, identity-biased recurrence
- Huginn-3.5B / latent-reasoning-interpretability: logit-lens + RDT
- LoopFormer (ICLR 2026): elastic-depth looped transformer
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .memory import FableMemory, FableMemoryConfig
from .depth import NarrativeDepthController, DepthTierConfig
from .probe import CoherenceProbe


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FableConfig:
    """
    Full configuration for an OpenFable model.

    Core architecture (mirrors OpenMythos MythosConfig)
    ---------------------------------------------------
    vocab_size      : int   = 32000
    dim             : int   = 2048     hidden dimension
    n_heads         : int   = 16       attention heads
    n_kv_heads      : int   = 8        GQA key/value heads (= n_heads → MHA)
    n_prelude       : int   = 4        standard transformer layers before recurrence
    n_coda          : int   = 4        standard transformer layers after recurrence
    n_loops         : int   = 8        default recurrence depth (overridden by NarrativeDC)
    ff_mult         : float = 4.0      FFN hidden = dim * ff_mult
    n_experts       : int   = 8        total MoE experts (1 = dense FFN)
    n_experts_used  : int   = 2        active routed experts per token
    n_shared_experts: int   = 1        always-on shared expert(s)
    max_seq_len     : int   = 4096
    rope_theta      : float = 500000.0
    layer_scale_init: float = 0.1      LayerScale init value (arXiv:2603.21676)
    use_act         : bool  = True     Adaptive Computation Time halting
    act_threshold   : float = 0.9      ACT halt threshold
    use_lora_adapters: bool = True     per-depth LoRA adapters
    lora_rank       : int   = 8

    FableMemory
    -----------
    memory : FableMemoryConfig

    NarrativeDepthController
    ------------------------
    narrative_mode  : str   = "dialogue"   default inference mode
    depth_jitter    : bool  = False        jitter loops during training

    CoherenceProbe
    --------------
    probe_top_k     : int   = 50
    probe_drift_threshold : float = 0.3
    """

    # ── Architecture ──────────────────────────────────────────────────────
    vocab_size:        int   = 32_000
    dim:               int   = 2048
    n_heads:           int   = 16
    n_kv_heads:        int   = 8
    n_prelude:         int   = 4
    n_coda:            int   = 4
    n_loops:           int   = 8
    ff_mult:           float = 4.0
    n_experts:         int   = 8
    n_experts_used:    int   = 2
    n_shared_experts:  int   = 1
    max_seq_len:       int   = 4096
    rope_theta:        float = 500_000.0
    layer_scale_init:  float = 0.1
    use_act:           bool  = True
    act_threshold:     float = 0.9
    use_lora_adapters: bool  = True
    lora_rank:         int   = 8

    # ── FableMemory ────────────────────────────────────────────────────────
    memory: FableMemoryConfig = field(default_factory=FableMemoryConfig)

    # ── NarrativeDepthController ───────────────────────────────────────────
    narrative_mode:  str  = "dialogue"
    depth_jitter:    bool = False

    # ── CoherenceProbe ─────────────────────────────────────────────────────
    probe_top_k:            int   = 50
    probe_drift_threshold:  float = 0.3

    # ── Behavioral alignment weights ──────────────────────────────────────────
    memory_scale_init:      float = 1.0   # Init scale for C mixing coefficient (FableMemory injection)
    loop_scale_init:        float = 1.0   # Init scale for LoRA depth adapter (encodes loop depth bias)
    default_narrative_mode: str   = ""    # If set, overrides narrative_mode for generate()

    @property
    def memory_dim(self) -> int:
        return self.memory.memory_dim

    @property
    def max_characters(self) -> int:
        return self.memory.max_characters

    @property
    def max_locations(self) -> int:
        return self.memory.max_locations

    @property
    def char_embed_dim(self) -> int:
        return self.memory.char_embed_dim

    @property
    def update_every_n_tokens(self) -> int:
        return self.memory.update_every_n_tokens


# ─────────────────────────────────────────────────────────────────────────────
# Primitives
# ─────────────────────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalisation (no bias, no mean subtraction)."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


def precompute_freqs_cis(
    dim: int, max_seq: int, theta: float = 500_000.0
) -> torch.Tensor:
    """Precompute RoPE rotation frequencies (complex exponentials)."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)   # [max_seq, dim/2]


def apply_rotary(
    x: torch.Tensor, freqs_cis: torch.Tensor
) -> torch.Tensor:
    """Apply RoPE to query or key tensor [batch, heads, seq, head_dim]."""
    xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs = freqs_cis[: x.shape[2]].unsqueeze(0).unsqueeze(0)
    xr = torch.view_as_real(xc * freqs).flatten(-2)
    return xr.type_as(x)


# ─────────────────────────────────────────────────────────────────────────────
# Attention  (GQA)
# ─────────────────────────────────────────────────────────────────────────────

class GQAttention(nn.Module):
    """Grouped-Query Attention with RoPE.

    n_kv_heads < n_heads → GQA;  n_kv_heads == n_heads → standard MHA.
    """

    def __init__(self, cfg: FableConfig) -> None:
        super().__init__()
        self.n_heads    = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim   = cfg.dim // cfg.n_heads
        self.n_rep      = cfg.n_heads // cfg.n_kv_heads

        d = cfg.dim
        h = self.head_dim
        self.q  = nn.Linear(d, cfg.n_heads    * h, bias=False)
        self.k  = nn.Linear(d, cfg.n_kv_heads * h, bias=False)
        self.v  = nn.Linear(d, cfg.n_kv_heads * h, bias=False)
        self.o  = nn.Linear(cfg.n_heads * h, d,   bias=False)
        self.scale = h ** -0.5

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, S, _ = x.shape
        h = self.head_dim

        q = self.q(x).view(B, S, self.n_heads,    h).transpose(1, 2)
        k = self.k(x).view(B, S, self.n_kv_heads, h).transpose(1, 2)
        v = self.v(x).view(B, S, self.n_kv_heads, h).transpose(1, 2)

        q = apply_rotary(q, freqs_cis)
        k = apply_rotary(k, freqs_cis)

        # Expand KV groups → heads
        if self.n_rep > 1:
            k = k.unsqueeze(2).expand(-1, -1, self.n_rep, -1, -1).reshape(B, self.n_heads, S, h)
            v = v.unsqueeze(2).expand(-1, -1, self.n_rep, -1, -1).reshape(B, self.n_heads, S, h)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            scores = scores + mask
        scores = F.softmax(scores.float(), dim=-1).type_as(x)
        out = torch.matmul(scores, v)                       # [B, H, S, h]
        out = out.transpose(1, 2).reshape(B, S, -1)
        return self.o(out)


# ─────────────────────────────────────────────────────────────────────────────
# Feed-forward  (sparse MoE)
# ─────────────────────────────────────────────────────────────────────────────

class SwiGLU(nn.Module):
    """SwiGLU feed-forward block (single expert)."""

    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up   = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class SparseMoE(nn.Module):
    """
    Sparse Mixture-of-Experts with shared + routed experts.

    Architecture (DeepSeek-V2 inspired, as in OpenMythos):
      - ``n_shared_experts`` always-active experts summed unconditionally
      - ``n_experts`` routed experts; top-``n_experts_used`` selected per token
      - Router: linear projection + softmax, top-k selection, load-balance loss
    """

    def __init__(self, cfg: FableConfig) -> None:
        super().__init__()
        hidden = int(cfg.dim * cfg.ff_mult)

        # Shared experts (always on)
        self.shared = nn.ModuleList(
            [SwiGLU(cfg.dim, hidden) for _ in range(cfg.n_shared_experts)]
        )

        # Routed experts
        self.experts = nn.ModuleList(
            [SwiGLU(cfg.dim, hidden) for _ in range(cfg.n_experts)]
        )

        # Router
        self.router  = nn.Linear(cfg.dim, cfg.n_experts, bias=False)
        self.top_k   = cfg.n_experts_used

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        x_flat = x.reshape(-1, D)                               # [B*S, D]

        # ── Shared experts ──────────────────────────────────────────────
        out = torch.zeros_like(x_flat)
        for exp in self.shared:
            out = out + exp(x_flat)

        # ── Routed experts ──────────────────────────────────────────────
        logits   = self.router(x_flat)                          # [B*S, E]
        weights, indices = torch.topk(logits, self.top_k, dim=-1)  # [B*S, k]
        weights  = F.softmax(weights, dim=-1)

        for k in range(self.top_k):
            idx = indices[:, k]           # [B*S]
            w   = weights[:, k].unsqueeze(-1)  # [B*S, 1]
            for e_idx in range(len(self.experts)):
                mask = (idx == e_idx)
                if mask.any():
                    out[mask] = out[mask] + w[mask] * self.experts[e_idx](x_flat[mask])

        return out.reshape(B, S, D)


# ─────────────────────────────────────────────────────────────────────────────
# Transformer layer  (shared by Prelude, Recurrent Block, Coda)
# ─────────────────────────────────────────────────────────────────────────────

class TransformerLayer(nn.Module):
    """Single transformer layer with pre-norm, GQA, MoE FFN, LayerScale."""

    def __init__(self, cfg: FableConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim)
        self.ffn_norm  = RMSNorm(cfg.dim)
        self.attn      = GQAttention(cfg)
        self.ffn       = SparseMoE(cfg)

        # LayerScale (arXiv:2603.21676) — scalar multiplier per residual branch
        ls = cfg.layer_scale_init
        self.ls_attn = nn.Parameter(torch.full((cfg.dim,), ls))
        self.ls_ffn  = nn.Parameter(torch.full((cfg.dim,), ls))

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.ls_attn * self.attn(self.attn_norm(x), freqs_cis, mask)
        x = x + self.ls_ffn  * self.ffn(self.ffn_norm(x))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# LoRA depth adapters
# ─────────────────────────────────────────────────────────────────────────────

class LoRAAdapter(nn.Module):
    """Per-loop-step LoRA adapter applied to the recurrent hidden state.

    Adds a low-rank residual:  h ← h + B·A·h  (A: [dim→r], B: [r→dim])
    Initialised with B=0 so identity at init.
    """

    def __init__(self, dim: int, rank: int) -> None:
        super().__init__()
        self.A = nn.Linear(dim, rank, bias=False)
        self.B = nn.Linear(rank, dim, bias=False)
        nn.init.zeros_(self.B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.B(self.A(x))


# ─────────────────────────────────────────────────────────────────────────────
# Recurrent Block
# ─────────────────────────────────────────────────────────────────────────────

class RecurrentBlock(nn.Module):
    """
    The core looped module.  Single weight set shared across all T loops.

    Update rule
    -----------
        h_{t+1} = A·h_t  +  B·e  +  C·m  +  Transformer(h_t, e, m)

    where:
        A, B, C  are learned scalar parameters (identity-biased init per 2603.21676)
        e        = encoded input (Prelude output, broadcast every loop)
        m        = FableMemory injection (optional, 0 if memory_dim=0)
        Transformer(·) is a standard TransformerLayer

    ACT halting
    -----------
    A linear head predicts a halt probability h_halt per token at each loop.
    When the cumulative remainder R drops below (1 - act_threshold), looping stops.
    The halted state is weighted by R for gradient continuity.
    """

    def __init__(self, cfg: FableConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.dim
        mem_d = cfg.memory.memory_dim

        # Shared transformer layer (same weights every loop)
        self.layer = TransformerLayer(cfg)

        # Identity-biased recurrence parameters (init A≈1, B≈0, C≈0)
        self.A = nn.Parameter(torch.ones(1))     # h_t coefficient
        self.B = nn.Parameter(torch.zeros(1))    # e coefficient
        self.C = nn.Parameter(torch.zeros(1)) if mem_d > 0 else None

        # Project memory vector m into model dim (optional)
        if mem_d > 0:
            self.mem_proj = nn.Linear(mem_d, d, bias=False)
        else:
            self.mem_proj = None

        # ACT halt head: per-token scalar halt probability
        if cfg.use_act:
            self.halt_head = nn.Linear(d, 1, bias=True)
            nn.init.constant_(self.halt_head.bias, -3.0)   # bias toward not halting
        else:
            self.halt_head = None

        # Per-loop LoRA adapters
        if cfg.use_lora_adapters:
            # Allocate up to max_loops adapters; index by loop step
            max_l = 32
            self.lora_adapters = nn.ModuleList(
                [LoRAAdapter(d, cfg.lora_rank) for _ in range(max_l)]
            )
        else:
            self.lora_adapters = None

    def forward(
        self,
        h: torch.Tensor,
        e: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
        n_loops: int,
        memory: Optional[FableMemory],
        probe: Optional[CoherenceProbe],
    ) -> Tuple[torch.Tensor, float]:
        """
        Parameters
        ----------
        h : [batch, seq, dim]   initial hidden state (= Prelude output)
        e : [batch, seq, dim]   encoded input (= Prelude output, kept fixed)
        freqs_cis, mask         standard attention args
        n_loops                 how many times to loop
        memory                  FableMemory instance or None
        probe                   CoherenceProbe instance or None

        Returns
        -------
        (h_final, act_halt_prob)
        """
        remainder   = torch.ones(h.shape[0], h.shape[1], 1, device=h.device)
        halt_accum  = torch.zeros_like(remainder)
        h_out       = torch.zeros_like(h)

        for t in range(n_loops):
            # ── Memory read ────────────────────────────────────────────────
            if memory is not None and self.mem_proj is not None:
                m_vec = memory.read(h[:, -1, :])          # [batch, mem_dim]
                m_seq = self.mem_proj(m_vec).unsqueeze(1)  # [batch, 1, dim]
                m_broadcast = m_seq.expand_as(h)           # [batch, seq, dim]
            else:
                m_broadcast = None

            # ── RDT update ─────────────────────────────────────────────────
            h_new = self.layer(h, freqs_cis, mask)
            h_new = self.A * h + self.B * e + h_new
            if m_broadcast is not None and self.C is not None:
                h_new = h_new + self.C * m_broadcast

            # ── LoRA depth adapter ─────────────────────────────────────────
            if self.lora_adapters is not None:
                idx = min(t, len(self.lora_adapters) - 1)
                h_new = self.lora_adapters[idx](h_new)

            h = h_new

            # ── CoherenceProbe ─────────────────────────────────────────────
            if probe is not None:
                probe.record_step(h)

            # ── ACT halting ────────────────────────────────────────────────
            if self.halt_head is not None and self.cfg.use_act:
                p_halt = torch.sigmoid(self.halt_head(h))  # [B, S, 1]

                # Weighted accumulation (Graves 2016 ACT)
                delta        = p_halt * remainder
                halt_accum   = halt_accum + delta
                h_out        = h_out + delta * h
                remainder    = remainder - delta

                # Check if all positions have halted
                if (halt_accum >= self.cfg.act_threshold).all():
                    break
            else:
                h_out = h

        if self.halt_head is not None and self.cfg.use_act:
            # Add remaining probability mass from last state
            h_out = h_out + remainder * h
            act_prob = float(halt_accum.detach().mean())
        else:
            h_out = h
            act_prob = 0.0

        return h_out, act_prob


# ─────────────────────────────────────────────────────────────────────────────
# OpenFable  (main model)
# ─────────────────────────────────────────────────────────────────────────────

class OpenFable(nn.Module):
    """
    OpenFable: Recurrent-Depth Transformer for narrative reasoning.

    Parameters
    ----------
    cfg : FableConfig

    Forward signature
    -----------------
    forward(
        token_ids:       Tensor [batch, seq],
        n_loops:         int | None        — overrides cfg.n_loops
        narrative_mode:  str | None        — passed to NarrativeDepthController
        memory:          FableMemory | None
        return_probes:   bool              — include CoherenceProbe report
    ) -> Tensor [batch, seq, vocab_size]   (logits)

    or if return_probes=True:
    -> (logits, probe_report: dict)
    """

    def __init__(self, cfg: FableConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # ── Embedding ──────────────────────────────────────────────────────
        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)

        # ── RoPE ──────────────────────────────────────────────────────────
        freqs = precompute_freqs_cis(
            cfg.dim // cfg.n_heads,
            cfg.max_seq_len,
            cfg.rope_theta,
        )
        self.register_buffer("freqs_cis", freqs)

        # ── Prelude ────────────────────────────────────────────────────────
        self.prelude = nn.ModuleList(
            [TransformerLayer(cfg) for _ in range(cfg.n_prelude)]
        )

        # ── Recurrent block ────────────────────────────────────────────────
        self.recurrent = RecurrentBlock(cfg)

        # ── Coda ───────────────────────────────────────────────────────────
        self.coda = nn.ModuleList(
            [TransformerLayer(cfg) for _ in range(cfg.n_coda)]
        )

        # ── Output ────────────────────────────────────────────────────────
        self.norm    = RMSNorm(cfg.dim)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)

        # Tie embeddings
        self.lm_head.weight = self.embed.weight

        # ── FableMemory ────────────────────────────────────────────────────
        if cfg.memory.memory_dim > 0:
            self.memory_module: Optional[FableMemory] = FableMemory(
                cfg.memory, cfg.dim
            )
        else:
            self.memory_module = None

        # ── NarrativeDepthController ───────────────────────────────────────
        self.depth_ctrl = NarrativeDepthController(
            default_mode=cfg.narrative_mode,
            use_act=cfg.use_act,
            act_threshold=cfg.act_threshold,
            jitter=cfg.depth_jitter,
        )

        # ── CoherenceProbe ─────────────────────────────────────────────────
        self.probe = CoherenceProbe(
            model_dim=cfg.dim,
            vocab_size=cfg.vocab_size,
            top_k=cfg.probe_top_k,
            drift_threshold=cfg.probe_drift_threshold,
        )
        self.probe.bind_lm_head(self.lm_head)

        # Weight init
        self._init_weights()

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        for name, p in self.named_parameters():
            if "weight" in name and p.dim() >= 2:
                nn.init.normal_(p, mean=0.0, std=0.02)
            elif "bias" in name:
                nn.init.zeros_(p)
        # Identity bias for recurrence A
        nn.init.ones_(self.recurrent.A)
        nn.init.zeros_(self.recurrent.B)
        if self.recurrent.C is not None:
            # Fable alignment: memory injection weight
            # memory_scale_init > 1.0 gives memory a stronger initial presence
            # (e.g. memory_scale_init=2.0 encodes Fable 5 +3x memory amplification)
            nn.init.constant_(self.recurrent.C, self.cfg.memory_scale_init * 0.01)
        # Fable alignment: LoRA depth adapter scale
        # loop_scale_init > 1.0 encodes "later loops do more work" (Fable 5 long-task scaling)
        if self.recurrent.lora_adapters is not None and self.cfg.loop_scale_init != 1.0:
            for adapter in self.recurrent.lora_adapters:
                # Scale A init std to encode loop depth bias
                nn.init.normal_(adapter.A.weight, mean=0.0, std=0.02 * self.cfg.loop_scale_init)

    # ------------------------------------------------------------------
    # Causal mask helper
    # ------------------------------------------------------------------

    def _causal_mask(self, seq: int, device: torch.device) -> torch.Tensor:
        mask = torch.full((1, 1, seq, seq), float("-inf"), device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        token_ids: torch.Tensor,
        n_loops: Optional[int] = None,
        narrative_mode: Optional[str] = None,
        memory: Optional[FableMemory] = None,
        return_probes: bool = False,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        token_ids : [batch, seq]
        n_loops   : explicit loop count; if None, uses NarrativeDepthController
        narrative_mode : "action" | "dialogue" | "exposition" | None
        memory    : FableMemory instance (overrides model's own if provided)
        return_probes : if True return (logits, probe_report)

        Returns
        -------
        logits : [batch, seq, vocab_size]
        or (logits, probe_report) if return_probes=True
        """
        B, S = token_ids.shape
        device = token_ids.device

        # ── Determine loop count ───────────────────────────────────────────
        if n_loops is None:
            n_loops = self.depth_ctrl.get_n_loops(narrative_mode)

        # ── Embedding ──────────────────────────────────────────────────────
        x = self.embed(token_ids)                           # [B, S, D]

        # ── Causal mask + RoPE ─────────────────────────────────────────────
        mask      = self._causal_mask(S, device)
        freqs_cis = self.freqs_cis[:S].to(device)

        # ── Prelude ────────────────────────────────────────────────────────
        for layer in self.prelude:
            x = layer(x, freqs_cis, mask)
        e = x                                               # frozen encoded input

        # ── Memory ────────────────────────────────────────────────────────
        mem = memory or self.memory_module

        # ── Probe reset ────────────────────────────────────────────────────
        self.probe.reset()

        # ── Recurrent block ────────────────────────────────────────────────
        x, act_prob = self.recurrent(
            h=x,
            e=e,
            freqs_cis=freqs_cis,
            mask=mask,
            n_loops=n_loops,
            memory=mem,
            probe=self.probe if return_probes else None,
        )

        # ── Coda ───────────────────────────────────────────────────────────
        for layer in self.coda:
            x = layer(x, freqs_cis, mask)

        # ── Output ────────────────────────────────────────────────────────
        logits = self.lm_head(self.norm(x))                 # [B, S, vocab]

        if return_probes:
            report = self.probe.report()
            report["act_halt_prob"] = act_prob
            return logits, report

        return logits

    # ------------------------------------------------------------------
    # Convenience: autoregressive generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        token_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_k: int = 50,
        narrative_mode: Optional[str] = None,
        memory: Optional[FableMemory] = None,
        update_memory_every: int = 0,
    ) -> torch.Tensor:
        """
        Simple autoregressive generation with optional FableMemory updates.

        Parameters
        ----------
        token_ids     : [1, prompt_len]  prompt tokens
        max_new_tokens: number of tokens to generate
        temperature   : sampling temperature (1.0 = unscaled)
        top_k         : top-k sampling (0 = greedy)
        narrative_mode: passed to NarrativeDepthController
        memory        : FableMemory instance for long-form generation
        update_memory_every : update memory every N tokens (0 = use cfg default)

        Returns
        -------
        Tensor [1, prompt_len + max_new_tokens]
        """
        cfg = self.cfg
        mem = memory or self.memory_module
        upd = update_memory_every or cfg.memory.update_every_n_tokens

        # Apply default_narrative_mode if set and caller did not provide one
        if narrative_mode is None and cfg.default_narrative_mode:
            narrative_mode = cfg.default_narrative_mode

        ids = token_ids
        for step in range(max_new_tokens):
            # Truncate to max_seq_len
            ids_ctx = ids[:, -cfg.max_seq_len:]

            # Forward
            logits = self(ids_ctx, narrative_mode=narrative_mode, memory=mem)
            next_logits = logits[:, -1, :] / max(temperature, 1e-6)

            # Top-k sampling
            if top_k > 0:
                v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < v[:, -1:]] = float("-inf")

            probs = F.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            ids = torch.cat([ids, next_id], dim=-1)

            # Memory write-back
            if mem is not None and upd > 0 and (step + 1) % upd == 0:
                h = self.embed(ids_ctx)
                for layer in self.prelude:
                    h = layer(h, self.freqs_cis[:ids_ctx.shape[1]])
                mem.write(h, ids_ctx, step=step)

        return ids

    # ------------------------------------------------------------------
    # Param count
    # ------------------------------------------------------------------

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def __repr__(self) -> str:
        M = self.param_count() / 1e6
        return (
            f"OpenFable("
            f"dim={self.cfg.dim}, "
            f"n_heads={self.cfg.n_heads}, "
            f"prelude={self.cfg.n_prelude}, "
            f"coda={self.cfg.n_coda}, "
            f"n_experts={self.cfg.n_experts}, "
            f"memory_dim={self.cfg.memory.memory_dim}, "
            f"params={M:.1f}M)"
        )
