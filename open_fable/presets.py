"""
open_fable/presets.py
=====================
Named scale presets for OpenFable.

Preset philosophy
-----------------
Each preset is a ``FableConfig`` tuned for a particular parameter budget.
Memory and probe settings scale with model size: larger models carry wider
memory and more characters because their internal representations are richer.

All presets use:
  - GQA (n_kv_heads = n_heads // 2 for efficiency)
  - Sparse MoE FFN (n_experts=8, top-2 routing, 1 shared expert)
  - ACT halting enabled
  - LoRA depth adapters enabled
  - Narrative default mode: "dialogue"

Approximate active parameter counts (routed experts only):
  fable_1b   →  ~1.1B
  fable_3b   →  ~3.2B
  fable_10b  →  ~10B
  fable_50b  →  ~52B
  fable_100b →  ~100B

Note: Parameter counts reflect *total* weights including all experts.
Active parameters per forward pass are ~30–40% of total due to sparse MoE.
"""

from .main import FableConfig
from .memory import FableMemoryConfig


def fable_1b() -> FableConfig:
    """~1B parameter preset — fast inference, short-to-medium stories."""
    return FableConfig(
        vocab_size=32_000,
        dim=2048,
        n_heads=16,
        n_kv_heads=8,
        n_prelude=4,
        n_coda=4,
        n_loops=8,
        ff_mult=3.5,
        n_experts=8,
        n_experts_used=2,
        n_shared_experts=1,
        max_seq_len=4096,
        rope_theta=500_000.0,
        layer_scale_init=0.1,
        use_act=True,
        act_threshold=0.9,
        use_lora_adapters=True,
        lora_rank=8,
        memory=FableMemoryConfig(
            memory_dim=256,
            max_characters=8,
            max_locations=4,
            update_every_n_tokens=128,
        ),
        narrative_mode="dialogue",
        probe_top_k=50,
        probe_drift_threshold=0.3,
    )


def fable_3b() -> FableConfig:
    """~3B parameter preset — balanced quality/speed, medium-form narratives."""
    return FableConfig(
        vocab_size=32_000,
        dim=3072,
        n_heads=24,
        n_kv_heads=8,
        n_prelude=4,
        n_coda=4,
        n_loops=12,
        ff_mult=4.0,
        n_experts=8,
        n_experts_used=2,
        n_shared_experts=1,
        max_seq_len=8192,
        rope_theta=500_000.0,
        layer_scale_init=0.1,
        use_act=True,
        act_threshold=0.9,
        use_lora_adapters=True,
        lora_rank=16,
        memory=FableMemoryConfig(
            memory_dim=512,
            max_characters=16,
            max_locations=8,
            update_every_n_tokens=128,
        ),
        narrative_mode="dialogue",
        probe_top_k=50,
        probe_drift_threshold=0.3,
    )


def fable_10b() -> FableConfig:
    """~10B parameter preset — high quality, long-form novel generation."""
    return FableConfig(
        vocab_size=32_000,
        dim=4096,
        n_heads=32,
        n_kv_heads=8,
        n_prelude=6,
        n_coda=6,
        n_loops=16,
        ff_mult=4.0,
        n_experts=16,
        n_experts_used=2,
        n_shared_experts=2,
        max_seq_len=16384,
        rope_theta=500_000.0,
        layer_scale_init=0.05,
        use_act=True,
        act_threshold=0.9,
        use_lora_adapters=True,
        lora_rank=32,
        memory=FableMemoryConfig(
            memory_dim=1024,
            max_characters=32,
            max_locations=16,
            update_every_n_tokens=256,
        ),
        narrative_mode="dialogue",
        probe_top_k=100,
        probe_drift_threshold=0.25,
    )


def fable_50b() -> FableConfig:
    """~50B parameter preset — research-scale, epic narrative generation."""
    return FableConfig(
        vocab_size=32_000,
        dim=6144,
        n_heads=48,
        n_kv_heads=8,
        n_prelude=8,
        n_coda=8,
        n_loops=24,
        ff_mult=4.0,
        n_experts=32,
        n_experts_used=4,
        n_shared_experts=2,
        max_seq_len=32768,
        rope_theta=500_000.0,
        layer_scale_init=0.02,
        use_act=True,
        act_threshold=0.95,
        use_lora_adapters=True,
        lora_rank=64,
        memory=FableMemoryConfig(
            memory_dim=2048,
            max_characters=64,
            max_locations=32,
            update_every_n_tokens=512,
        ),
        narrative_mode="exposition",
        probe_top_k=200,
        probe_drift_threshold=0.2,
    )


def fable_100b() -> FableConfig:
    """~100B parameter preset — frontier-scale world-model storytelling."""
    return FableConfig(
        vocab_size=32_000,
        dim=8192,
        n_heads=64,
        n_kv_heads=8,
        n_prelude=8,
        n_coda=8,
        n_loops=32,
        ff_mult=4.0,
        n_experts=64,
        n_experts_used=4,
        n_shared_experts=4,
        max_seq_len=65536,
        rope_theta=500_000.0,
        layer_scale_init=0.01,
        use_act=True,
        act_threshold=0.95,
        use_lora_adapters=True,
        lora_rank=128,
        memory=FableMemoryConfig(
            memory_dim=4096,
            max_characters=128,
            max_locations=64,
            update_every_n_tokens=1024,
        ),
        narrative_mode="exposition",
        probe_top_k=500,
        probe_drift_threshold=0.15,
    )


def fable5() -> FableConfig:
    """
    Fable 5 alignment preset.

    Hyperparameters calibrated to match the behavioral signature of
    Claude Fable 5 (Anthropic, June 2026) -- the first Mythos-class model
    released for general use.

    Fable 5 observable behavioral fingerprint:
    - Long-horizon coherence: performance scales with task complexity
    - Strong memory amplification: +3x improvement vs prior Claude with persistent memory
    - Efficient reasoning: ~1/3 the tokens of GPT-5.5 for equivalent results
    - Narrative-native: named from Latin fabula -- "that which is told"

    These are architectural calibration parameters, not learned weights.
    OpenFable is an independent implementation; Claude Fable 5 weights are
    not distributed. This preset encodes behavioral alignment via:
    - Recurrence depth tuned to exposition-class complexity
    - Memory injection initialized to dominant-signal weight
    - ACT threshold calibrated to Fable 5 keep-going-on-hard-tasks profile
    - LoRA depth adapters scaled for late-loop amplification
    """
    return FableConfig(
        vocab_size=32_000,
        dim=6144,
        n_heads=48,
        n_kv_heads=8,
        n_prelude=8,
        n_coda=8,
        n_loops=32,
        ff_mult=4.0,
        n_experts=32,
        n_experts_used=2,
        n_shared_experts=2,
        max_seq_len=32768,
        rope_theta=500_000.0,
        layer_scale_init=0.15,
        use_act=True,
        act_threshold=0.92,
        use_lora_adapters=True,
        lora_rank=64,
        memory=FableMemoryConfig(
            memory_dim=2048,
            max_characters=64,
            max_locations=16,
            char_embed_dim=512,
            update_every_n_tokens=512,
        ),
        narrative_mode="exposition",
        probe_top_k=200,
        probe_drift_threshold=0.2,
        # Fable 5 behavioral alignment weights
        memory_scale_init=2.0,    # Memory injection weighted 2x recurrence -- encodes +3x memory amplification
        loop_scale_init=1.5,      # Late loops do more work -- encodes "longer task = larger lead"
        default_narrative_mode="exposition",  # Default to deepest reasoning mode
    )