"""
open_fable/probe.py
===================
CoherenceProbe — logit-lens interpretability hook for recurrent narrative models.

Design basis
------------
The Huginn-3.5B / latent-reasoning-interpretability work showed that decoded
latent representations of a recurrent transformer reveal *progressive answer
refinement* across loop steps: early loops produce uncertain, scattered logit
distributions; later loops converge on a confident answer.

For narrative generation, we exploit this observation differently.  We track
not just whether the model is "confident" in general, but whether its
prediction of *named-entity tokens* (character names, locations) is *stable
across loops*.  A character drifting — being predicted with high probability
in loop 4 but suppressed in loop 12 — signals a coherence failure.

Metric: top-k entropy
---------------------
At each recurrence loop step t, the probe projects the hidden state h_t into
vocabulary space using the shared lm_head weights (weight tying, no extra
parameters), then computes the entropy of the top-k softmax distribution:

    p_k = softmax(top_k(h_t · W_vocab^T))
    H_t = -Σ p_k · log(p_k)

Lower H_t → more confident prediction → higher "coherence score" (we return
1 - H_t / log(k) so scores are in [0, 1] with 1 = maximally confident).

Entity-level drift detection
----------------------------
``character_drift(name, token_ids)`` compares the first-loop and last-loop
probability mass assigned to the token IDs of a character's name.  A drift
score > ``drift_threshold`` (default 0.3) suggests the model has lost track
of that character.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CoherenceProbe(nn.Module):
    """
    Lightweight interpretability hook attached to the recurrent block.

    Parameters
    ----------
    model_dim : int
        Hidden dimension of the OpenFable model.
    vocab_size : int
        Vocabulary size.  The probe reuses the model's lm_head weight.
    top_k : int
        Number of top logits to include in entropy computation.
    drift_threshold : float
        Absolute probability-mass difference above which a character is
        flagged as "drifting".
    """

    def __init__(
        self,
        model_dim: int,
        vocab_size: int,
        top_k: int = 50,
        drift_threshold: float = 0.3,
    ) -> None:
        super().__init__()
        self.model_dim = model_dim
        self.vocab_size = vocab_size
        self.top_k = top_k
        self.drift_threshold = drift_threshold

        # Projection from hidden → vocab space.
        # Weights are *shared* with lm_head at bind-time (no extra params).
        # Before bind_lm_head() is called we use a dedicated linear.
        self._proj = nn.Linear(model_dim, vocab_size, bias=False)
        self._bound = False

        # Per-forward loop cache (populated by record_step)
        self._loop_logits: List[torch.Tensor] = []   # list of [batch, vocab]

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def bind_lm_head(self, lm_head: nn.Linear) -> None:
        """Share lm_head weight with the probe (weight-tying, zero overhead).

        Call this after the parent model is constructed.
        """
        # Replace internal _proj weight with lm_head.weight (no copy)
        self._proj.weight = lm_head.weight
        self._bound = True

    # ------------------------------------------------------------------
    # Per-loop recording
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear per-forward loop cache.  Call before each new forward pass."""
        self._loop_logits.clear()

    def record_step(self, hidden: torch.Tensor) -> float:
        """
        Project hidden state to logits, compute coherence score, cache logits.

        Parameters
        ----------
        hidden : Tensor  [batch, seq, model_dim]  or  [batch, model_dim]
            Recurrent hidden state at the current loop step.

        Returns
        -------
        float
            Coherence score for this loop step in [0, 1].
            1.0 = maximally confident (low entropy); 0.0 = uniform.
        """
        if hidden.dim() == 3:
            h = hidden[:, -1, :]    # last token position: [batch, dim]
        else:
            h = hidden              # [batch, dim]

        with torch.no_grad():
            logits = self._proj(h.float())          # [batch, vocab]

        self._loop_logits.append(logits.detach())

        # Coherence score: mean across batch
        score = self._coherence_from_logits(logits)
        return score

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _coherence_from_logits(self, logits: torch.Tensor) -> float:
        """Top-k entropy → coherence score in [0, 1]."""
        k = min(self.top_k, logits.shape[-1])
        top_logits, _ = torch.topk(logits, k, dim=-1)          # [batch, k]
        probs = F.softmax(top_logits, dim=-1)                   # [batch, k]
        entropy = -(probs * (probs + 1e-9).log()).sum(dim=-1)   # [batch]
        max_entropy = math.log(k)
        # Normalise to [0,1], invert so higher = more confident
        score = float(1.0 - (entropy.mean() / max_entropy).clamp(0, 1))
        return score

    # ------------------------------------------------------------------
    # Full-forward summary
    # ------------------------------------------------------------------

    def coherence_scores(self) -> List[float]:
        """Return per-loop coherence scores collected since last reset().

        Returns
        -------
        List[float]
            One score per recorded loop step.
        """
        scores = []
        for logits in self._loop_logits:
            scores.append(self._coherence_from_logits(logits))
        return scores

    # ------------------------------------------------------------------
    # Character drift detection
    # ------------------------------------------------------------------

    def character_drift(
        self,
        char_token_ids: Sequence[int],
        early_loop: int = 0,
        late_loop: int = -1,
    ) -> float:
        """
        Measure how much the probability mass on a character's name tokens
        shifts between an early loop and a late loop.

        Parameters
        ----------
        char_token_ids : sequence of int
            Token IDs that spell out the character's name.
        early_loop : int
            Index of the "early" loop in the cached logits list.
        late_loop : int
            Index of the "late" loop (-1 = last recorded).

        Returns
        -------
        float
            Drift magnitude in [0, 1].  Values above ``drift_threshold``
            suggest a coherence failure for this character.
        """
        if len(self._loop_logits) < 2:
            return 0.0

        early_idx = early_loop
        late_idx  = late_loop if late_loop >= 0 else len(self._loop_logits) - 1
        early_idx = min(early_idx, len(self._loop_logits) - 1)
        late_idx  = min(late_idx,  len(self._loop_logits) - 1)

        ids = torch.tensor(char_token_ids, dtype=torch.long)

        def mass(logits: torch.Tensor) -> float:
            probs = F.softmax(logits[0], dim=-1)   # batch[0], [vocab]
            return float(probs[ids].sum())

        early_mass = mass(self._loop_logits[early_idx])
        late_mass  = mass(self._loop_logits[late_idx])
        drift = abs(late_mass - early_mass)
        return drift

    def is_drifting(self, char_token_ids: Sequence[int]) -> bool:
        """Return True if character drift exceeds the configured threshold."""
        return self.character_drift(char_token_ids) > self.drift_threshold

    # ------------------------------------------------------------------
    # Summary report
    # ------------------------------------------------------------------

    def report(self) -> Dict:
        """Return a diagnostic dict summarising the last forward pass."""
        scores = self.coherence_scores()
        if not scores:
            return {"n_loops": 0, "scores": [], "mean": 0.0, "trend": "none"}

        mean = sum(scores) / len(scores)
        # Trend: compare first vs last half
        mid = len(scores) // 2 or 1
        first_half = sum(scores[:mid]) / mid
        second_half = sum(scores[mid:]) / max(len(scores) - mid, 1)
        if second_half > first_half + 0.05:
            trend = "improving"
        elif second_half < first_half - 0.05:
            trend = "degrading"
        else:
            trend = "stable"

        return {
            "n_loops": len(scores),
            "scores": scores,
            "mean": round(mean, 4),
            "trend": trend,
        }
