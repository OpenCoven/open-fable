# OpenFable

**A Coven-flavored Recurrent-Depth Transformer for narrative reasoning, character consistency, and long-form story coherence.**

---

> **Disclaimer:** OpenFable is an independent, theoretical implementation. It is not affiliated with Anthropic, DeepMind, or any commercial AI lab. This is a research prototype and architectural exploration — not a trained model. No weights are distributed.

---

## Overview

OpenFable is a fork and extension of the OpenMythos Recurrent-Depth Transformer (RDT) architecture, adapted for narrative AI tasks. The core recurrence machinery is preserved faithfully; three new Coven-specific modules are added on top:

| Module | Purpose |
|---|---|
| **FableMemory** | Character and world-state persistence across generation windows |
| **NarrativeDepthController** | Loop-depth scheduling driven by narrative mode (action / dialogue / exposition) |
| **CoherenceProbe** | Per-loop logit-lens coherence scoring and character drift detection |

These additions are grounded in recent research on recurrent transformer interpretability and adaptive computation:

- [arXiv:2603.21676](https://arxiv.org/abs/2603.21676) *"Thinking Deeper, Not Longer"* — silent thinking objective, LayerScale init, identity-biased recurrence
- [Huginn-3.5B](https://arxiv.org/abs/2502.05171) / [latent-reasoning-interpretability](https://flowshu.github.io/latent-reasoning-interpretability/) — decoded latent states reveal progressive answer refinement across loops; logit lens works on recurrent models
- [LoopFormer (ICLR 2026)](https://loopformer.github.io/) — elastic-depth looped transformer, adaptive depth by task complexity

---

## Architecture

### Recurrent-Depth Transformer (base)

The pipeline follows OpenMythos exactly:

```
Token IDs
    │
    ▼
Embedding + RoPE
    │
    ▼
PRELUDE  (n_prelude standard transformer layers, run once)
    │  e = encoded input (frozen for all loops)
    ▼
RECURRENT BLOCK  (single weight set, looped T times)
    │
    ▼
CODA  (n_coda standard transformer layers, run once)
    │
    ▼
LM Head → logits
```

**Recurrent update rule (extended for FableMemory):**

```
h_{t+1} = A·h_t  +  B·e  +  C·m  +  Transformer(h_t, e, m)
```

| Symbol | Shape | Description |
|---|---|---|
| `h_t` | `[B, S, D]` | Hidden state at loop step t |
| `e` | `[B, S, D]` | Encoded input (Prelude output, fixed across all loops) |
| `m` | `[B, D]` | FableMemory injection (broadcast to sequence dim) |
| `A, B, C` | scalar | Learned mixing coefficients (identity-biased: A≈1, B≈0, C≈0 at init) |
| `Transformer(·)` | → `[B, S, D]` | Single transformer layer (shared weights every loop) |

When `memory_dim=0`, `C·m` is absent and the update reduces to the standard OpenMythos rule: `h_{t+1} = A·h_t + B·e + Transformer(h_t, e)`.

**Attention:** Grouped-Query Attention (GQA) with RoPE. `n_kv_heads < n_heads` enables GQA; `n_kv_heads == n_heads` gives standard MHA.

**FFN:** Sparse Mixture-of-Experts with:
- `n_shared_experts` always-active shared experts (summed unconditionally)
- `n_experts` routed experts, top-`n_experts_used` selected per token via linear router

**Stability:** LayerScale per residual branch (init = `layer_scale_init`, default 0.1), following [arXiv:2603.21676](https://arxiv.org/abs/2603.21676). Identity-biased recurrence (`A` init = 1) provides a gradient highway through depth.

**ACT halting:** Each loop step, a linear head predicts a per-token halt probability. Looping stops when cumulative halt probability exceeds `act_threshold`. Uses Graves (2016) ACT weighted accumulation for gradient continuity.

**LoRA depth adapters:** A separate low-rank adapter `h ← h + B·A·h` per loop index. B initialised to zero (identity at init). Allows the model to learn loop-specific transformations without blowing up parameter count.

---

## The Three Fable Modules

### 1. FableMemory

Character and world-state persistence across generation windows.

**Design:** The recurrence field naturally passes semantic structure forward loop-by-loop. But it resets between generation windows (independent forward passes). FableMemory bridges this gap by maintaining:

- `CharacterState`: named entity embedding + trait vector + relationship matrix
- `WorldState`: location embeddings (dict), time anchor, causality stack

**Injection:** Memory is read once per loop step as a flat vector `m ∈ ℝ^{memory_dim}`, produced by a learned aggregation network over the current state:

```
char_block   = [entity_emb_1 | entity_emb_2 | ... | 0 ... 0]   [max_characters × char_embed_dim]
time_anchor  = current narrative time vector                     [char_embed_dim]
loc_block    = [loc_emb_1 | ... | 0 ... 0]                      [max_locations × char_embed_dim]
caus_block   = [premise_1 | ... | 0 ... 0]                       [max_locations × char_embed_dim]

flat = concat(char_block, time_anchor, loc_block, caus_block)
m    = Linear(GELU(Linear(flat)))                               [memory_dim]
```

**Write-back:** Every `update_every_n_tokens` tokens (or at explicit scene boundaries), hidden states are projected into character embedding space via EMA:

```
entity_emb ← (1 - α)·entity_emb + α·Encoder(h_last)
```

Default `α = 0.1`. This keeps memory stable across short windows while allowing drift over longer arcs.

**Config (`FableMemoryConfig`):**

| Field | Default | Description |
|---|---|---|
| `memory_dim` | 256 | Flat injection vector dimension. 0 = disabled. |
| `max_characters` | 16 | Max tracked characters (FIFO eviction) |
| `max_locations` | 8 | Max tracked locations |
| `char_embed_dim` | 64 | Character embedding dimension (default: memory_dim // 4) |
| `update_every_n_tokens` | 128 | Write-back interval |

---

### 2. NarrativeDepthController

Loop-depth scheduling driven by narrative mode, extending [LoopFormer's](https://loopformer.github.io/) elastic-depth idea with narrative-domain priors.

**Depth tiers:**

| Mode | Loop range | Rationale |
|---|---|---|
| `action` | 4 – 8 | Fast surface prediction; plot momentum > compositional reasoning |
| `dialogue` | 8 – 16 | Character voice requires persona consistency across loops |
| `exposition` | 16 – 32 | World-building demands deep coherence across causal chains |

**ACT integration:** When `use_act=True`, the controller respects the model's ACT halt signal and stops early when the cumulative halt probability exceeds `act_threshold` — but never below `min_loops` for the current mode.

**Usage:**

```python
# At inference
logits = model(ids, narrative_mode="dialogue")     # 8–16 loops
logits = model(ids, narrative_mode="action")       # 4–8 loops
logits = model(ids, n_loops=12)                    # explicit override

# Custom modes
model.depth_ctrl.add_mode("dream", min_loops=12, max_loops=20)
logits = model(ids, narrative_mode="dream")
```

---

### 3. CoherenceProbe

Lightweight logit-lens interpretability hook. Inspired by the [Huginn-3.5B](https://arxiv.org/abs/2502.05171) finding that decoded latent representations reveal progressive answer refinement across loops.

**Metric — top-k entropy:**

At each loop step t, the probe projects `h_t` to vocabulary space (reusing lm_head weights, zero extra parameters), then computes:

```
p_k = softmax(top_k(h_t · W_vocab^T))
H_t = -Σ p_k · log(p_k)
coherence_t = 1 - H_t / log(k)      ∈ [0, 1]
```

Higher coherence score → more confident prediction → more focused recurrent computation.

**Character drift detection:**

Compares probability mass on a character's name tokens between early and late loops:

```
drift = |prob_mass(char_tokens, loop_early) - prob_mass(char_tokens, loop_late)|
```

`drift > drift_threshold` (default 0.3) flags a coherence failure.

**Usage:**

```python
logits, report = model(ids, n_loops=8, return_probes=True)
# report = {
#   "n_loops": 8,
#   "scores": [0.41, 0.48, 0.55, 0.61, 0.65, 0.68, 0.70, 0.71],
#   "mean": 0.6237,
#   "trend": "improving",
#   "act_halt_prob": 0.73
# }

# Character drift
elara_tokens = tokenizer.encode("Elara")
drift = model.probe.character_drift(elara_tokens)
if model.probe.is_drifting(elara_tokens):
    print("⚠️  Elara is drifting — increase loop depth or reset memory")
```

---

## Scale Presets

| Preset | dim | n_heads | n_experts | n_loops | memory_dim | max_chars | Notes |
|---|---|---|---|---|---|---|---|
| `fable_1b` | 2048 | 16 | 8 | 8 | 256 | 8 | Fast, short-to-medium |
| `fable_3b` | 3072 | 24 | 8 | 12 | 512 | 16 | Balanced quality/speed |
| `fable_10b` | 4096 | 32 | 16 | 16 | 1024 | 32 | High quality, long-form |
| `fable_50b` | 6144 | 48 | 32 | 24 | 2048 | 64 | Research-scale |
| `fable_100b` | 8192 | 64 | 64 | 32 | 4096 | 128 | Frontier-scale |
| `fable5` | 16384 | 128 | 128 | 32 | 16384 | 256 | ~9.47T total / ~878B active — Fable 5 alignment |

All presets use GQA (`n_kv_heads=8`), sparse MoE routing, ACT halting, and LoRA depth adapters.

---

## Fable 5 Alignment

The `fable5` preset encodes the behavioral signature of [Claude Fable 5](https://www.anthropic.com/news/claude-fable-5-mythos-5) (Anthropic, June 2026) -- the first Mythos-class model released for general use. Its name derives from the Latin *fabula*: "that which is told."

OpenFable shares the same architectural intuition: that narrative reasoning -- long-horizon coherence, character persistence, thematic throughlines -- requires more computational depth, not more parameters.

### Observed behavioral fingerprint -> architectural parameters

| Fable 5 observation | Source | OpenFable parameter |
|---|---|---|
| ~10T total / ~878B active params | Independent researcher analysis | `dim=16384`, `n_experts=128`, `n_experts_used=8` |
| "Longer task = larger lead" | Anthropic launch | `n_loops=32`, `loop_scale_init=1.5` |
| +3× memory amplification vs Opus 4.8 | Slay the Spire eval | `memory_dim=16384`, `memory_scale_init=2.0` |
| ~1/3 the tokens of GPT-5.5, equiv. results | Benchmark analysis | `use_act=True`, `act_threshold=0.92` |
| Stable across million-token contexts | Anthropic system card | `layer_scale_init=0.15`, `n_prelude=12` |
| Name: *fabula* -- "that which is told" | Etymology | `default_narrative_mode="exposition"` |

### Scale

```python
# Total parameters:    ~9.47 trillion
# Active per pass:     ~878 billion (top-8 of 128 experts + 4 shared)
# Sparsity ratio:      ~10.8×
# Recurrence depth:    32 loops × 1 shared weight set
# Unique weight layers: 25 (12 prelude + 1 recurrent + 12 coda)
# Memory injection:    16384-dim (full model width, no bottleneck)

from open_fable import OpenFable, fable5
model = OpenFable(fable5())
# ~9.47T parameter model — requires significant distributed infrastructure to train
# Forward pass: ~878B active params (comparable to a ~1T dense model)
```

> **Infrastructure note:** This configuration requires multi-node GPU/TPU infrastructure to instantiate, let alone train. For research and experimentation, use `fable_3b()` or `fable_10b()`. `fable5()` is provided as an architectural reference and for distributed training at scale.

### Usage

```python
from open_fable import OpenFable, fable5

model = OpenFable(fable5())
# ~9.47T parameter class, Fable 5 architectural alignment
# default narrative mode: exposition (32 loops)
```

**Important:** These are architectural calibration parameters, not learned weights. Claude Fable 5's actual weights are proprietary to Anthropic and are not distributed here. `fable5()` encodes behavioral alignment via initialization and hyperparameter choices -- the resulting model must be trained from scratch.

---

## Installation

```bash
pip install open-fable                      # PyPI (when available)
# or
git clone https://github.com/coven-ai/open-fable
cd open-fable && pip install -e .

# Optional: FlashAttention 2 (significant speedup on long sequences)
pip install open-fable[flash]
```

---

## Quick Start

```python
from open_fable import OpenFable, FableConfig, fable_1b
import torch

# Preset
cfg   = fable_1b()
model = OpenFable(cfg)

# Custom tiny config
cfg = FableConfig(
    vocab_size=32000,
    dim=2048,
    n_heads=16,
    n_kv_heads=8,
)
model = OpenFable(cfg)
print(model)
# OpenFable(dim=2048, n_heads=16, prelude=4, coda=4, n_experts=8, memory_dim=256, params=...)

# Forward pass
ids    = torch.randint(0, 32000, (1, 128))
logits = model(ids, narrative_mode="dialogue")    # [1, 128, 32000]

# With coherence probe
logits, report = model(ids, n_loops=8, return_probes=True)
print(report["trend"])   # "improving" / "stable" / "degrading"

# Generation
prompt    = torch.randint(0, 32000, (1, 32))
generated = model.generate(
    prompt,
    max_new_tokens=256,
    temperature=0.8,
    top_k=50,
    narrative_mode="dialogue",
)
```

---

## FableMemory Usage

```python
from open_fable import FableMemory, FableMemoryConfig

mem = FableMemory(FableMemoryConfig(memory_dim=256), model_dim=2048)

# Window 1 — introduce a character
h = ...  # [1, seq, 2048] hidden states from your own encoder
mem.write(h, step=0, character_names=["Elara"], location_names=["Thornwood"])

# Window 2 — memory is injected automatically during forward
logits = model(window2_ids, memory=mem, narrative_mode="dialogue")

# Push a causal premise (e.g. "the crown is hidden in the vault")
premise_vec = model.embed(torch.tensor([[crown_token_id]])).squeeze()
mem.push_causal_premise(premise_vec)

# Resolve it later
resolved = mem.pop_causal_premise()
```

---

## Project Structure

```
open-fable/
  open_fable/
    __init__.py                ← exports: OpenFable, FableConfig, fable_1b…fable_100b
    main.py                    ← OpenFable model class, FableConfig, core primitives
    memory.py                  ← FableMemory, CharacterState, WorldState
    depth.py                   ← NarrativeDepthController
    probe.py                   ← CoherenceProbe
    presets.py                 ← named scale presets
  examples/
    basic_generate.py
    character_consistency.py
    narrative_depth.py
  tests/
    test_forward.py
    test_memory.py
    test_probe.py
  pyproject.toml
  README.md
```

---

## Testing

```bash
pip install open-fable[dev]
pytest tests/ -v
```

---


---

## Training Data

OpenFable ships two complementary training pipelines. See [`datasets/README.md`](datasets/README.md) for full documentation.

### Stage 1 — MythosBridge

Bridges [`WithinUsAI/claude_mythos_distilled_25k`](https://huggingface.co/datasets/WithinUsAI/claude_mythos_distilled_25k) into OpenFable format.

25k synthetic examples across mathematical reasoning, advanced coding, cybersecurity, scientific analysis, agentic planning, and general expert QA — all re-annotated with `suggested_n_loops` and `narrative_mode` for RDT training.

```bash
python -m open_fable.data.mythos_bridge --output data/mythos_bridge.jsonl
```

### Stage 2 — FableForge

Generates synthetic narrative training examples where harder tasks explicitly require more recurrence loops. **The first dataset designed around recurrence depth requirements.**

```
character_trace:      loops = f(n_characters, n_scenes)      — FableMemory active
coherence_challenge:  loops = f(inconsistency_type)          — CoherenceProbe active
narrative_completion: loops = f(n_characters, n_constraints) — both active
```

| Inconsistency type | Loops | Reasoning depth |
|---|---|---|
| Name drift | 4 | Surface pattern |
| Location contradiction | 8 | Spatial reasoning |
| Object continuity | 8 | State tracking |
| Timeline error | 16 | Temporal ordering |
| Relationship error | 16 | Social graph recall |
| Trait reversal | 32 | Character psychology |

```bash
python -m open_fable.data.fable_forge --count 25000 --output data/fable_forge.jsonl
python -m open_fable.data.fable_forge --stats --count 1000  # distribution preview
```

```python
from open_fable.data import forge_dataset, bridge_dataset

stage2 = forge_dataset(count=25000, seed=42)
hard   = [e for e in stage2 if e["suggested_n_loops"] == 32]
print(f"{len(hard):,} examples require maximum recurrence depth")
```

## Credits and Lineage

OpenFable is a narrative-focused fork of the OpenMythos Recurrent-Depth Transformer.

**Core architecture credit:**
- **[OpenMythos](https://github.com/kyegomez/OpenMythos)** (MIT license) — Recurrent-Depth Transformer base: Prelude/Recurrent/Coda structure, GQA, sparse MoE, ACT halting, LoRA depth adapters

**Research that shaped the design:**
- Dong et al. (2025). *"Thinking Deeper, Not Longer: Recurrent Depth Transformers."* [arXiv:2603.21676](https://arxiv.org/abs/2603.21676) — LayerScale stability, identity-biased recurrence, silent thinking objective
- Geiping et al. — *[Huginn-3.5B](https://arxiv.org/abs/2502.05171) / [latent-reasoning-interpretability](https://flowshu.github.io/latent-reasoning-interpretability/)* — logit-lens on recurrent models, progressive answer refinement
- Liu et al. (ICLR 2026). *[LoopFormer](https://loopformer.github.io/)* — elastic depth by task complexity
- Su et al. (2023). *RoFormer* — RoPE rotary positional embeddings
- Dai et al. (2024). *DeepSeek-V2* — Multi-Latent Attention, shared+routed MoE experts
- Graves (2016). *Adaptive Computation Time for RNNs* — ACT halting mechanism
**Fable 5 behavioral alignment:**
- [Claude Fable 5 & Mythos 5](https://www.anthropic.com/news/claude-fable-5-mythos-5) (Anthropic, June 2026) -- behavioral fingerprint encoded as architectural calibration in `fable5()` preset. Claude Fable 5 is a product of Anthropic; this package is not affiliated with Anthropic.
- [Claude Fable 5 & Mythos 5 System Card](https://www-cdn.anthropic.com/d00db56fa754a1b115b6dd7cb2e3c342ee809620.pdf) -- capability profile and safety architecture

---

## Limitations and Honest Notes

1. **No trained weights.** OpenFable is an architectural prototype. The forward pass runs and gradients flow; the model is not trained on any narrative corpus.

2. **FableMemory is a latent mechanism, not retrieval.** It does not store verbatim text — only compressed latent vectors. It will not recall exact quotes; it shapes the model's latent trajectory.

3. **CoherenceProbe is a proxy metric.** Top-k entropy is a reasonable proxy for prediction confidence, not a ground-truth coherence score. Character drift detection is best used as a soft signal, not a hard gate.

4. **MoE routing is simplified.** The current implementation iterates over experts in a Python loop for clarity. Production use requires a fused CUDA kernel (e.g. Megablocks or grouped-GEMM).

5. **ACT is approximate.** The halting criterion is applied at the batch/sequence mean level, not per-token, for simplicity.

---

*OpenFable is part of the Coven research suite. Built with curiosity, not hype.*
