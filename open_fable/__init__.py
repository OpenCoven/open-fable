"""
open_fable
==========
OpenFable — a Coven-flavored Recurrent-Depth Transformer for narrative
reasoning, character consistency, and long-form story coherence.

Quick start
-----------
    from open_fable import OpenFable, FableConfig, fable_1b

    # From a named preset
    cfg = fable_1b()
    model = OpenFable(cfg)

    # From scratch
    cfg = FableConfig(vocab_size=32000, dim=2048, n_heads=16)
    model = OpenFable(cfg)

    import torch
    ids = torch.randint(0, cfg.vocab_size, (1, 128))
    logits = model(ids, narrative_mode="dialogue")       # [1, 128, vocab_size]

    # With CoherenceProbe
    logits, probe_report = model(ids, return_probes=True, n_loops=8)
    print(probe_report)   # {"n_loops": 8, "scores": [...], "mean": 0.72, "trend": "improving"}

Modules
-------
- OpenFable          — main model class
- FableConfig        — full configuration dataclass
- FableMemory        — character + world-state persistence
- FableMemoryConfig  — memory configuration
- NarrativeDepthController — narrative-mode loop scheduler
- CoherenceProbe     — per-loop logit-lens coherence scores
- fable_1b … fable_100b — named scale presets
"""

from .main import OpenFable, FableConfig
from .memory import FableMemory, FableMemoryConfig, CharacterState, WorldState
from .depth import NarrativeDepthController, DepthTierConfig
from .probe import CoherenceProbe
from .presets import fable_1b, fable_3b, fable_10b, fable_50b, fable_100b, fable5

__version__ = "0.1.0"
__all__ = [
    # Core
    "OpenFable",
    "FableConfig",
    # Memory
    "FableMemory",
    "FableMemoryConfig",
    "CharacterState",
    "WorldState",
    # Depth control
    "NarrativeDepthController",
    "DepthTierConfig",
    # Probe
    "CoherenceProbe",
    # Presets
    "fable_1b",
    "fable_3b",
    "fable_10b",
    "fable_50b",
    "fable_100b",
    "fable5",
]
