"""
examples/character_consistency.py
===================================
Demo: using FableMemory to track a named character across generation windows.

Scenario
--------
We simulate a story that mentions character "Elara" across three windows.
After each window, we call memory.write() to update her state, then verify
that the memory injection vector changes between windows (showing that the
memory is being updated).
"""

import torch
from open_fable import OpenFable, FableConfig, FableMemory, FableMemoryConfig

# ── Setup ──────────────────────────────────────────────────────────────────────
cfg = FableConfig(
    vocab_size=1000,
    dim=256,
    n_heads=4,
    n_kv_heads=2,
    n_prelude=2,
    n_coda=2,
    n_loops=6,
    ff_mult=2.0,
    n_experts=2,
    n_experts_used=1,
    n_shared_experts=1,
    memory=FableMemoryConfig(
        memory_dim=128,
        max_characters=4,
        max_locations=2,
        update_every_n_tokens=16,
    ),
)
model = OpenFable(cfg)
memory = FableMemory(cfg.memory, cfg.dim)

# Simulate "Elara" as token IDs 42–44
elara_token_ids = [42, 43, 44]

# ── Window 1: Elara introduced ─────────────────────────────────────────────────
print("=== Window 1: Introduction ===")
window1 = torch.randint(0, cfg.vocab_size, (1, 24))
logits1 = model(window1, n_loops=6, memory=memory)
print(f"  Logits shape: {logits1.shape}")

# Extract hidden states for memory write (use embedding layer as proxy)
h1 = model.embed(window1)
for layer in model.prelude:
    h1 = layer(h1, model.freqs_cis[:24])

memory.write(h1, window1, step=24, character_names=["Elara"], location_names=["Thornwood"])

m_after_w1 = memory.read(h1[:, -1, :]).detach().clone()
print(f"  Characters registered: {memory.character_names}")
print(f"  Memory vector norm (after W1): {m_after_w1.norm():.4f}")

# ── Window 2: Elara in dialogue ─────────────────────────────────────────────────
print("\n=== Window 2: Dialogue ===")
window2 = torch.randint(0, cfg.vocab_size, (1, 24))
logits2 = model(window2, n_loops=10, narrative_mode="dialogue", memory=memory)

h2 = model.embed(window2)
for layer in model.prelude:
    h2 = layer(h2, model.freqs_cis[:24])

memory.write(h2, window2, step=48, character_names=["Elara"], location_names=["Thornwood"])

m_after_w2 = memory.read(h2[:, -1, :]).detach().clone()
print(f"  Logits shape: {logits2.shape}")
print(f"  Memory vector norm (after W2): {m_after_w2.norm():.4f}")

# Verify memory updated (norms should differ after EMA update)
diff = (m_after_w2 - m_after_w1).norm().item()
print(f"  Memory shift between windows: {diff:.6f}  (> 0 confirms update)")

# ── Window 3: CoherenceProbe drift check ──────────────────────────────────────
print("\n=== Window 3: Coherence check ===")
window3 = torch.randint(0, cfg.vocab_size, (1, 24))
logits3, probe_report = model(window3, n_loops=8, memory=memory, return_probes=True)

print(f"  Probe report: {probe_report}")
drift = model.probe.character_drift(elara_token_ids)
print(f"  Elara drift score: {drift:.4f}  (>{model.probe.drift_threshold} = flagged)")
if model.probe.is_drifting(elara_token_ids):
    print("  ⚠️  Character drift detected for Elara")
else:
    print("  ✓  Elara coherence within threshold")

print("\n=== Summary ===")
print(f"  Registered characters: {memory.character_names}")
print(f"  World locations: {list(memory._world.location_embeddings.keys())}")
