"""
examples/basic_generate.py
===========================
Minimal example: instantiate OpenFable and run a forward pass.
"""

import torch
from open_fable import OpenFable, FableConfig

# ── Tiny model for demonstration ──────────────────────────────────────────────
cfg = FableConfig(
    vocab_size=1000,
    dim=256,
    n_heads=4,
    n_kv_heads=2,
    n_prelude=2,
    n_coda=2,
    n_loops=4,
    ff_mult=2.0,
    n_experts=4,
    n_experts_used=2,
    n_shared_experts=1,
)

model = OpenFable(cfg)
print(model)

# ── Forward pass ──────────────────────────────────────────────────────────────
ids    = torch.randint(0, cfg.vocab_size, (1, 32))
logits = model(ids, n_loops=4)
print(f"Logits shape: {logits.shape}")   # [1, 32, 1000]

# ── With NarrativeDepthController ─────────────────────────────────────────────
for mode in ("action", "dialogue", "exposition"):
    n = model.depth_ctrl.get_n_loops(mode)
    print(f"  {mode:<12} → {n} loops")

# ── With CoherenceProbe ────────────────────────────────────────────────────────
logits, report = model(ids, n_loops=6, return_probes=True)
print("Probe report:", report)

# ── Greedy generation (short) ─────────────────────────────────────────────────
prompt = torch.randint(0, cfg.vocab_size, (1, 8))
generated = model.generate(prompt, max_new_tokens=20, temperature=1.0, top_k=10)
print(f"Generated sequence length: {generated.shape[1]}")
