"""
examples/narrative_depth.py
============================
Demo: NarrativeDepthController depth scheduling.

Shows how loop count, latency proxy, and probe coherence scores differ
across narrative modes on the same sequence.
"""

import time
import torch
from open_fable import OpenFable, FableConfig, NarrativeDepthController

# ── Tiny model ──────────────────────────────────────────────────────────────
cfg = FableConfig(
    vocab_size=1000,
    dim=256,
    n_heads=4,
    n_kv_heads=2,
    n_prelude=2,
    n_coda=2,
    n_loops=8,
    ff_mult=2.0,
    n_experts=2,
    n_experts_used=1,
    n_shared_experts=1,
)
model = OpenFable(cfg)
ids   = torch.randint(0, cfg.vocab_size, (1, 32))

# ── NarrativeDepthController standalone ──────────────────────────────────────
print("=== NarrativeDepthController ===")
dc = NarrativeDepthController()
print(dc)
for mode in ("action", "dialogue", "exposition"):
    n     = dc.get_n_loops(mode)
    tier  = dc.mode_info(mode)
    print(f"  {mode:<12}  n_loops={n:2d}  range=[{tier.min_loops},{tier.max_loops}]")

# Custom mode
dc.add_mode("dream", min_loops=12, max_loops=20)
print(f"  {'dream':<12}  n_loops={dc.get_n_loops('dream'):2d}  range=[12,20]")

# ── Per-mode forward passes with timing ──────────────────────────────────────
print("\n=== Mode comparison (same prompt) ===")
print(f"  {'mode':<12}  {'loops':>6}  {'time_ms':>8}  {'probe_mean':>10}  {'trend':>10}")
print(f"  {'-'*54}")

for mode in ("action", "dialogue", "exposition"):
    n_loops = dc.get_n_loops(mode)
    t0 = time.perf_counter()
    logits, report = model(
        ids,
        n_loops=n_loops,
        narrative_mode=mode,
        return_probes=True,
    )
    ms = (time.perf_counter() - t0) * 1000
    print(
        f"  {mode:<12}  {n_loops:>6}  {ms:>8.1f}  "
        f"{report['mean']:>10.4f}  {report['trend']:>10}"
    )

# ── ACT early stopping demo ──────────────────────────────────────────────────
print("\n=== ACT halt integration ===")
dc_act = NarrativeDepthController(use_act=True, act_threshold=0.5)
# Simulate a high ACT signal
n_halted = dc_act.get_n_loops("dialogue", act_halt_prob=0.95)
n_full   = dc_act.get_n_loops("dialogue", act_halt_prob=0.0)
print(f"  dialogue, act=0.95 → {n_halted} loops")
print(f"  dialogue, act=0.00 → {n_full} loops")
