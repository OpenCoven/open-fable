"""
open_fable/memory.py
====================
FableMemory — Character and world-state persistence across generation windows.

Architecture note
-----------------
The core RDT update rule is:

    h_{t+1} = A·h_t  +  B·e  +  Transformer(h_t, e)

FableMemory extends this to:

    h_{t+1} = A·h_t  +  B·e  +  C·m  +  Transformer(h_t, e, m)

where `m ∈ ℝ^{batch × memory_dim}` is a single flat injection vector produced by
encoding the current CharacterState and WorldState into a shared latent space.

The memory vector is *read* every recurrent loop and *written* (updated) once per
generation window — at scene boundaries or every ``update_every_n_tokens`` tokens.
This asymmetry keeps the recurrence cheap while still allowing long-horizon coherence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class FableMemoryConfig:
    """Configuration for the FableMemory subsystem.

    Attributes
    ----------
    memory_dim:
        Dimensionality of the flat memory injection vector ``m``.
        Set to 0 to disable memory entirely (standard RDT behaviour).
    max_characters:
        Maximum number of tracked named characters.  Additional characters
        beyond this limit are silently dropped (FIFO).
    max_locations:
        Maximum number of tracked world locations / time anchors.
    char_embed_dim:
        Embedding dimension for character trait and relationship matrices.
        Defaults to ``memory_dim // 4``.
    update_every_n_tokens:
        How often (in generated tokens) the memory write-back runs.
        Set to 0 to update only at explicit scene-boundary calls.
    """

    memory_dim: int = 256
    max_characters: int = 16
    max_locations: int = 8
    char_embed_dim: int = 64       # overridden to memory_dim//4 if 0
    update_every_n_tokens: int = 128


# ---------------------------------------------------------------------------
# State containers
# ---------------------------------------------------------------------------

@dataclass
class CharacterState:
    """Latent representation of a single named character.

    Fields
    ------
    name:
        Display name (used as a dictionary key and for CoherenceProbe lookups).
    entity_embedding:
        Float tensor of shape ``[char_embed_dim]``.  Initialised from the
        token embedding of the character's name, then refined by the memory
        update rule.
    trait_vector:
        ``[char_embed_dim]`` vector encoding persistent personality traits
        (derived from a learnable linear head over the hidden states observed
        in the character's dialogue / action windows).
    relationship_matrix:
        ``[max_characters, char_embed_dim]`` matrix storing relational
        embeddings to other characters.  Row ``i`` is the directed relationship
        from *this* character to character ``i`` in the registry.
    last_seen_step:
        Token index of the last generation step that mentioned this character.
    """

    name: str
    entity_embedding: torch.Tensor          # [char_embed_dim]
    trait_vector: torch.Tensor              # [char_embed_dim]
    relationship_matrix: torch.Tensor       # [max_characters, char_embed_dim]
    last_seen_step: int = 0


@dataclass
class WorldState:
    """Latent anchors for the current narrative world.

    Fields
    ------
    location_embeddings:
        Dict mapping location name → ``[char_embed_dim]`` tensor.
        Maintained as a ring buffer capped at ``max_locations``.
    time_anchor:
        Single ``[char_embed_dim]`` vector encoding the current narrative
        "time" (chapter, era, etc.).  Updated at scene boundaries.
    causality_stack:
        Stack of ``[char_embed_dim]`` vectors representing causal premises
        that are currently "open" (events that have been set up but not yet
        resolved).  Max depth = ``max_locations``.
    """

    location_embeddings: Dict[str, torch.Tensor] = field(default_factory=dict)
    time_anchor: Optional[torch.Tensor] = None
    causality_stack: List[torch.Tensor] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core module
# ---------------------------------------------------------------------------

class FableMemory(nn.Module):
    """
    Character + world-state memory bank for OpenFable.

    Parameters
    ----------
    cfg : FableMemoryConfig
    model_dim : int
        Hidden dimension of the parent OpenFable model.  Used to size the
        projection that maps memory → model space.

    Forward contract
    ----------------
    ``read(hidden: Tensor) -> Tensor``
        Input:  hidden state of shape ``[batch, model_dim]``
        Output: memory injection ``m`` of shape ``[batch, memory_dim]``

    ``write(hidden: Tensor, token_ids: Tensor, step: int)``
        Updates CharacterState / WorldState from the current hidden state.
        Called by the model every ``update_every_n_tokens`` tokens.
    """

    def __init__(self, cfg: FableMemoryConfig, model_dim: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.model_dim = model_dim

        d = cfg.memory_dim
        ce = cfg.char_embed_dim or max(1, d // 4)
        self.char_embed_dim = ce

        # ── Character encoder ──────────────────────────────────────────────
        # Projects model hidden state to character embedding space
        self.char_encoder = nn.Linear(model_dim, ce, bias=False)

        # Character registry (not nn.Parameters — updated by write())
        self._char_registry: Dict[str, CharacterState] = {}
        self._char_order: List[str] = []          # FIFO eviction order

        # World state
        self._world: WorldState = WorldState()

        # ── Memory read-out ────────────────────────────────────────────────
        # Aggregates all character + world embeddings into a single m vector
        # Input size: max_characters * ce  (char)
        #           + max_locations  * ce  (location)
        #           + ce                   (time anchor)
        #           + max_locations  * ce  (causality stack)
        max_c = cfg.max_characters
        max_l = cfg.max_locations
        agg_in = (max_c + 1 + max_l + max_l) * ce
        self.memory_aggregator = nn.Sequential(
            nn.Linear(agg_in, d * 2, bias=True),
            nn.GELU(),
            nn.Linear(d * 2, d, bias=True),
        )

        # ── Memory write-back ──────────────────────────────────────────────
        # Given new hidden state, updates entity embedding via EMA
        self.trait_extractor = nn.Linear(model_dim, ce, bias=False)
        self.ema_alpha = 0.1          # exponential moving average weight

        self._step_counter: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def read(self, hidden: torch.Tensor) -> torch.Tensor:
        """Encode current memory state into injection vector m.

        Parameters
        ----------
        hidden : Tensor  [batch, model_dim]
            Current recurrent hidden state (used for dynamic gating; the
            memory content itself is state-independent, but future extensions
            may condition on hidden).

        Returns
        -------
        Tensor  [batch, memory_dim]
        """
        batch = hidden.shape[0]
        ce = self.char_embed_dim
        max_c = self.cfg.max_characters
        max_l = self.cfg.max_locations
        device = hidden.device
        dtype = hidden.dtype

        # ── Character block: [max_characters, ce] ─────────────────────
        char_block = torch.zeros(max_c, ce, device=device, dtype=dtype)
        for i, name in enumerate(self._char_order[:max_c]):
            cs = self._char_registry[name]
            char_block[i] = cs.entity_embedding.to(device=device, dtype=dtype)

        # ── Time anchor: [ce] ──────────────────────────────────────────
        if self._world.time_anchor is not None:
            time_vec = self._world.time_anchor.to(device=device, dtype=dtype)
        else:
            time_vec = torch.zeros(ce, device=device, dtype=dtype)

        # ── Location block: [max_locations, ce] ───────────────────────
        loc_block = torch.zeros(max_l, ce, device=device, dtype=dtype)
        for i, (_, v) in enumerate(list(self._world.location_embeddings.items())[:max_l]):
            loc_block[i] = v.to(device=device, dtype=dtype)

        # ── Causality stack: [max_locations, ce] ──────────────────────
        caus_block = torch.zeros(max_l, ce, device=device, dtype=dtype)
        for i, vec in enumerate(self._world.causality_stack[:max_l]):
            caus_block[i] = vec.to(device=device, dtype=dtype)

        # ── Aggregate ──────────────────────────────────────────────────
        flat = torch.cat([
            char_block.reshape(-1),
            time_vec,
            loc_block.reshape(-1),
            caus_block.reshape(-1),
        ], dim=0)                                         # [agg_in]
        flat = flat.unsqueeze(0).expand(batch, -1)        # [batch, agg_in]
        m = self.memory_aggregator(flat)                  # [batch, memory_dim]
        return m

    def write(
        self,
        hidden: torch.Tensor,
        token_ids: Optional[torch.Tensor] = None,
        step: int = 0,
        character_names: Optional[List[str]] = None,
        location_names: Optional[List[str]] = None,
    ) -> None:
        """Update character and world state from the current hidden state.

        Parameters
        ----------
        hidden : Tensor  [batch, seq, model_dim]  or  [batch, model_dim]
            Hidden states at the current generation step.
        token_ids : Tensor  [batch, seq]  optional
            Token IDs (reserved for future name-entity detection).
        step : int
            Current token generation step.
        character_names : list of str, optional
            Characters whose state should be updated.  If None, updates
            all registered characters using the mean hidden state.
        location_names : list of str, optional
            Locations to add/update.
        """
        if hidden.dim() == 3:
            h = hidden.mean(dim=1)          # [batch, model_dim]
        else:
            h = hidden                       # [batch, model_dim]

        # Take the first batch element for state updates (simplified)
        h0 = h[0].detach().float()

        new_entity = self.char_encoder(h0.unsqueeze(0)).squeeze(0).detach()
        new_trait  = self.trait_extractor(h0.unsqueeze(0)).squeeze(0).detach()

        # Update specified (or all) characters via EMA
        targets = character_names if character_names else list(self._char_order)
        for name in targets:
            if name in self._char_registry:
                cs = self._char_registry[name]
                cs.entity_embedding = (
                    (1 - self.ema_alpha) * cs.entity_embedding.float() +
                    self.ema_alpha * new_entity
                ).to(cs.entity_embedding.dtype)
                cs.trait_vector = (
                    (1 - self.ema_alpha) * cs.trait_vector.float() +
                    self.ema_alpha * new_trait
                ).to(cs.trait_vector.dtype)
                cs.last_seen_step = step

        # Register new characters
        for name in (character_names or []):
            if name not in self._char_registry:
                self._register_character(name, new_entity, new_trait)

        # Register new locations
        for loc in (location_names or []):
            self._register_location(loc, new_entity)

        # Update time anchor via EMA
        if self._world.time_anchor is None:
            self._world.time_anchor = new_entity.clone()
        else:
            self._world.time_anchor = (
                (1 - self.ema_alpha) * self._world.time_anchor.float() +
                self.ema_alpha * new_entity
            )

        self._step_counter = step

    # ------------------------------------------------------------------
    # Character / location helpers
    # ------------------------------------------------------------------

    def _register_character(
        self,
        name: str,
        entity_emb: torch.Tensor,
        trait_vec: torch.Tensor,
    ) -> None:
        if len(self._char_order) >= self.cfg.max_characters:
            evicted = self._char_order.pop(0)
            del self._char_registry[evicted]

        max_c = self.cfg.max_characters
        rel_matrix = torch.zeros(max_c, self.char_embed_dim)
        cs = CharacterState(
            name=name,
            entity_embedding=entity_emb.clone().cpu(),
            trait_vector=trait_vec.clone().cpu(),
            relationship_matrix=rel_matrix,
        )
        self._char_registry[name] = cs
        self._char_order.append(name)

    def _register_location(self, name: str, emb: torch.Tensor) -> None:
        locs = self._world.location_embeddings
        if len(locs) >= self.cfg.max_locations and name not in locs:
            oldest = next(iter(locs))
            del locs[oldest]
        locs[name] = emb.clone().cpu()

    def push_causal_premise(self, premise_vec: torch.Tensor) -> None:
        """Push a causal premise onto the stack (call at scene set-up)."""
        stack = self._world.causality_stack
        if len(stack) >= self.cfg.max_locations:
            stack.pop(0)
        stack.append(premise_vec.detach().cpu())

    def pop_causal_premise(self) -> Optional[torch.Tensor]:
        """Resolve and pop the most recent causal premise."""
        if self._world.causality_stack:
            return self._world.causality_stack.pop()
        return None

    # Convenience
    def reset(self) -> None:
        """Clear all character and world state."""
        self._char_registry.clear()
        self._char_order.clear()
        self._world = WorldState()
        self._step_counter = 0

    @property
    def character_names(self) -> List[str]:
        return list(self._char_order)

    @property
    def character_states(self) -> Dict[str, CharacterState]:
        return dict(self._char_registry)
