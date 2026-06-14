"""tests/test_memory.py — FableMemory inject/update cycle."""

import pytest
import torch
from open_fable import FableMemory, FableMemoryConfig


@pytest.fixture
def mem_cfg():
    return FableMemoryConfig(
        memory_dim=64,
        max_characters=4,
        max_locations=2,
        char_embed_dim=16,
        update_every_n_tokens=8,
    )


@pytest.fixture
def memory(mem_cfg):
    return FableMemory(mem_cfg, model_dim=128)


# ── Read before any writes ────────────────────────────────────────────────────

def test_read_initial_shape(memory, mem_cfg):
    hidden = torch.randn(2, 128)
    m = memory.read(hidden)
    assert m.shape == (2, mem_cfg.memory_dim), (
        f"Expected ({2}, {mem_cfg.memory_dim}), got {m.shape}"
    )


def test_read_returns_zero_norm_initially(memory):
    """With no characters registered, memory read should be non-NaN."""
    hidden = torch.zeros(1, 128)
    m = memory.read(hidden)
    assert not torch.isnan(m).any(), "Memory read contains NaN"


# ── Write / register characters ──────────────────────────────────────────────

def test_register_character(memory):
    hidden = torch.randn(1, 8, 128)
    memory.write(hidden, step=0, character_names=["Elara"])
    assert "Elara" in memory.character_names


def test_register_multiple_characters(memory):
    hidden = torch.randn(1, 4, 128)
    memory.write(hidden, step=0, character_names=["Elara", "Dorian"])
    assert set(["Elara", "Dorian"]).issubset(set(memory.character_names))


def test_fifo_eviction(memory, mem_cfg):
    hidden = torch.randn(1, 4, 128)
    names = [f"Char{i}" for i in range(mem_cfg.max_characters + 2)]
    for name in names:
        memory.write(hidden, step=0, character_names=[name])
    assert len(memory.character_names) <= mem_cfg.max_characters


# ── EMA update ────────────────────────────────────────────────────────────────

def test_ema_updates_entity_embedding(memory):
    hidden = torch.randn(1, 4, 128)
    memory.write(hidden, step=0, character_names=["Elara"])

    emb_before = memory.character_states["Elara"].entity_embedding.clone()

    hidden2 = torch.randn(1, 4, 128)
    memory.write(hidden2, step=16, character_names=["Elara"])

    emb_after = memory.character_states["Elara"].entity_embedding
    diff = (emb_after - emb_before).norm().item()
    assert diff > 0, "EMA did not update entity embedding"


# ── Memory vector changes after write ─────────────────────────────────────────

def test_memory_vector_changes_after_write(memory):
    hidden = torch.randn(1, 8, 128)
    h_last = hidden[:, -1, :]

    m_before = memory.read(h_last).detach().clone()
    memory.write(hidden, step=0, character_names=["Elara"])
    m_after = memory.read(h_last).detach().clone()

    diff = (m_after - m_before).norm().item()
    assert diff > 0, "Memory vector did not change after character registration"


# ── Location registration ────────────────────────────────────────────────────

def test_register_location(memory):
    hidden = torch.randn(1, 4, 128)
    memory.write(hidden, step=0, location_names=["Thornwood"])
    assert "Thornwood" in memory._world.location_embeddings


def test_location_fifo_eviction(memory, mem_cfg):
    hidden = torch.randn(1, 4, 128)
    locs = [f"Loc{i}" for i in range(mem_cfg.max_locations + 3)]
    for loc in locs:
        memory.write(hidden, step=0, location_names=[loc])
    assert len(memory._world.location_embeddings) <= mem_cfg.max_locations


# ── Causal stack ──────────────────────────────────────────────────────────────

def test_causal_premise_push_pop(memory):
    premise = torch.randn(16)
    memory.push_causal_premise(premise)
    assert len(memory._world.causality_stack) == 1

    popped = memory.pop_causal_premise()
    assert popped is not None
    assert len(memory._world.causality_stack) == 0


def test_causal_pop_empty_returns_none(memory):
    result = memory.pop_causal_premise()
    assert result is None


# ── Reset ────────────────────────────────────────────────────────────────────

def test_reset_clears_state(memory):
    hidden = torch.randn(1, 4, 128)
    memory.write(hidden, step=0, character_names=["Elara"], location_names=["Thornwood"])
    memory.reset()
    assert len(memory.character_names) == 0
    assert len(memory._world.location_embeddings) == 0


# ── Batch consistency ─────────────────────────────────────────────────────────

def test_read_consistent_across_batch(memory, mem_cfg):
    """Memory is broadcast to all batch elements."""
    hidden1 = torch.randn(1, 128)
    hidden3 = torch.randn(3, 128)
    m1 = memory.read(hidden1)
    m3 = memory.read(hidden3)
    assert m1.shape == (1, mem_cfg.memory_dim)
    assert m3.shape == (3, mem_cfg.memory_dim)
