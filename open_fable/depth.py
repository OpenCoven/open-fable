"""
open_fable/depth.py
===================
NarrativeDepthController — loop-depth scheduling driven by narrative mode.

Design basis
------------
LoopFormer (ICLR 2026) demonstrated that elastic depth — choosing how many
recurrence steps to run based on task complexity — yields better compute
efficiency than a fixed ``n_loops``.  OpenFable extends this idea with
*narrative-domain priors*: the required reasoning depth varies systematically
by discourse type.

Depth tiers
-----------
+---------------+------------+-----------------------------------------------+
| mode          | loop range | rationale                                     |
+===============+============+===============================================+
| ``action``    | 4–8        | Fast, surface-level prediction; plot momentum |
|               |            | matters more than deep compositional reasoning|
+---------------+------------+-----------------------------------------------+
| ``dialogue``  | 8–16       | Character voice requires persona consistency; |
|               |            | more loops let the recurrence field stabilise |
|               |            | the character's latent "voice" before decoding|
+---------------+------------+-----------------------------------------------+
| ``exposition``| 16–32      | World-building demands deep coherence across  |
|               |            | multiple causal chains; highest reasoning cost|
+---------------+------------+-----------------------------------------------+

ACT integration
---------------
When ``use_act=True`` (default), the controller respects the model's own
Adaptive Computation Time halting signal and will stop early if the ACT
halt probability crosses ``act_threshold`` — but *never below* ``min_loops``
for the current mode.  This prevents action passages from accidentally
triggering deep computation on trivial tokens.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DepthTierConfig:
    """Min/max loop range for a single narrative mode."""
    min_loops: int
    max_loops: int

    def __post_init__(self) -> None:
        if self.min_loops < 1:
            raise ValueError("min_loops must be ≥ 1")
        if self.max_loops < self.min_loops:
            raise ValueError("max_loops must be ≥ min_loops")


DEFAULT_TIERS: Dict[str, DepthTierConfig] = {
    "action":     DepthTierConfig(min_loops=4,  max_loops=8),
    "dialogue":   DepthTierConfig(min_loops=8,  max_loops=16),
    "exposition": DepthTierConfig(min_loops=16, max_loops=32),
}


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

class NarrativeDepthController:
    """
    Determines how many recurrence loops to run for each generation step.

    Parameters
    ----------
    tiers : dict, optional
        Mapping from mode name → DepthTierConfig.  Defaults to the three
        standard tiers above.  You may add custom modes (e.g. ``"flashback"``,
        ``"dream"``).
    default_mode : str
        Mode to use when no explicit mode is passed to ``get_n_loops``.
    use_act : bool
        If True, the controller accepts an ``act_halt_prob`` argument and
        can stop early when the model's ACT signal fires.
    act_threshold : float
        Halt when ACT cumulative probability exceeds this value (default 0.9).
    jitter : bool
        If True, samples uniformly from [min_loops, max_loops] rather than
        always returning max_loops.  Useful during training.
    """

    def __init__(
        self,
        tiers: Optional[Dict[str, DepthTierConfig]] = None,
        default_mode: str = "dialogue",
        use_act: bool = True,
        act_threshold: float = 0.9,
        jitter: bool = False,
    ) -> None:
        self.tiers = dict(DEFAULT_TIERS)
        if tiers:
            self.tiers.update(tiers)
        self.default_mode = default_mode
        self.use_act = use_act
        self.act_threshold = act_threshold
        self.jitter = jitter

    # ------------------------------------------------------------------
    # Core method
    # ------------------------------------------------------------------

    def get_n_loops(
        self,
        narrative_mode: Optional[str] = None,
        act_halt_prob: float = 0.0,
    ) -> int:
        """Return the number of recurrence loops for this generation step.

        Parameters
        ----------
        narrative_mode : str, optional
            One of ``"action"``, ``"dialogue"``, ``"exposition"`` (or any
            custom tier name added at construction).  Falls back to
            ``default_mode`` if None.
        act_halt_prob : float
            Cumulative ACT halt probability produced by the model.  Only
            consulted when ``use_act=True``.

        Returns
        -------
        int
            Number of loops to execute.  Always in [min_loops, max_loops]
            for the selected mode.
        """
        mode = narrative_mode or self.default_mode
        if mode not in self.tiers:
            raise ValueError(
                f"Unknown narrative mode '{mode}'. "
                f"Available: {list(self.tiers.keys())}"
            )
        tier = self.tiers[mode]

        if self.jitter:
            n = random.randint(tier.min_loops, tier.max_loops)
        else:
            n = tier.max_loops

        # ACT early stopping — never go below min_loops
        if self.use_act and act_halt_prob >= self.act_threshold:
            n = max(tier.min_loops, n // 2)

        return n

    def should_halt(self, loop_step: int, narrative_mode: Optional[str] = None) -> bool:
        """Return True if we have exceeded the max loops for this mode.

        Convenience wrapper so the recurrence loop can poll without holding
        a reference to the tier config.
        """
        mode = narrative_mode or self.default_mode
        return loop_step >= self.tiers[mode].max_loops

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def add_mode(self, name: str, min_loops: int, max_loops: int) -> None:
        """Register a custom narrative mode."""
        self.tiers[name] = DepthTierConfig(min_loops=min_loops, max_loops=max_loops)

    def mode_info(self, mode: Optional[str] = None) -> DepthTierConfig:
        """Return the DepthTierConfig for a given mode."""
        m = mode or self.default_mode
        if m not in self.tiers:
            raise ValueError(f"Unknown mode '{m}'")
        return self.tiers[m]

    def __repr__(self) -> str:
        tier_str = ", ".join(
            f"{k}=[{v.min_loops},{v.max_loops}]" for k, v in self.tiers.items()
        )
        return (
            f"NarrativeDepthController(default={self.default_mode!r}, "
            f"tiers={{{tier_str}}}, use_act={self.use_act})"
        )
