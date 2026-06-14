"""
open_fable.data — Training data pipelines for OpenFable.

MythosBridge: Ingests WithinUsAI/claude_mythos_distilled_25k and annotates
              examples with recurrence-depth metadata for RDT training.

FableForge:   Generates synthetic narrative training examples where harder
              tasks explicitly require more recurrence loops — the first
              dataset designed around recurrence depth requirements.

Two-stage training pipeline:
  Stage 1 (MythosBridge): general deep reasoning pretraining
  Stage 2 (FableForge):   narrative-specific fine-tuning
"""

from .mythos_bridge import MythosBridge, process_dataset as bridge_dataset
from .fable_forge import FableForge, generate as forge_dataset

__all__ = ["MythosBridge", "FableForge", "bridge_dataset", "forge_dataset"]
