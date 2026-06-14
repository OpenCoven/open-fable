"""tests/test_forward.py — basic forward pass and shape checks."""

import pytest
import torch
from open_fable import OpenFable, FableConfig
from open_fable.presets import fable_1b


# ── Fixture: tiny model for fast tests ──────────────────────────────────────
@pytest.fixture
def tiny_cfg():
    return FableConfig(
        vocab_size=512,
        dim=128,
        n_heads=4,
        n_kv_heads=2,
        n_prelude=2,
        n_coda=2,
        n_loops=4,
        ff_mult=2.0,
        n_experts=2,
        n_experts_used=1,
        n_shared_experts=1,
        use_act=False,         # disable ACT for determinism
        use_lora_adapters=True,
        lora_rank=4,
    )


@pytest.fixture
def tiny_model(tiny_cfg):
    return OpenFable(tiny_cfg)


# ── Forward shape ────────────────────────────────────────────────────────────

def test_forward_shape(tiny_model, tiny_cfg):
    ids    = torch.randint(0, tiny_cfg.vocab_size, (2, 16))
    logits = tiny_model(ids, n_loops=2)
    assert logits.shape == (2, 16, tiny_cfg.vocab_size), (
        f"Expected (2, 16, {tiny_cfg.vocab_size}), got {logits.shape}"
    )


def test_forward_batch_1(tiny_model, tiny_cfg):
    ids    = torch.randint(0, tiny_cfg.vocab_size, (1, 8))
    logits = tiny_model(ids, n_loops=3)
    assert logits.shape == (1, 8, tiny_cfg.vocab_size)


def test_forward_single_token(tiny_model, tiny_cfg):
    ids    = torch.randint(0, tiny_cfg.vocab_size, (1, 1))
    logits = tiny_model(ids, n_loops=2)
    assert logits.shape == (1, 1, tiny_cfg.vocab_size)


# ── Loop count variation ──────────────────────────────────────────────────────

@pytest.mark.parametrize("n_loops", [1, 2, 4, 8])
def test_forward_various_loops(tiny_model, tiny_cfg, n_loops):
    ids    = torch.randint(0, tiny_cfg.vocab_size, (1, 12))
    logits = tiny_model(ids, n_loops=n_loops)
    assert logits.shape == (1, 12, tiny_cfg.vocab_size)


# ── Narrative mode routing ────────────────────────────────────────────────────

@pytest.mark.parametrize("mode", ["action", "dialogue", "exposition"])
def test_narrative_mode_forward(tiny_model, tiny_cfg, mode):
    ids    = torch.randint(0, tiny_cfg.vocab_size, (1, 8))
    # n_loops=None → NarrativeDepthController decides
    logits = tiny_model(ids, narrative_mode=mode)
    assert logits.dim() == 3


# ── Return probes ─────────────────────────────────────────────────────────────

def test_return_probes(tiny_model, tiny_cfg):
    ids = torch.randint(0, tiny_cfg.vocab_size, (1, 8))
    result = tiny_model(ids, n_loops=4, return_probes=True)
    assert isinstance(result, tuple) and len(result) == 2
    logits, report = result
    assert logits.shape == (1, 8, tiny_cfg.vocab_size)
    assert "scores" in report
    assert "mean" in report
    assert "trend" in report
    # Should have 4 recorded steps
    assert len(report["scores"]) == 4


# ── Gradient flow ────────────────────────────────────────────────────────────

def test_gradient_flow(tiny_model, tiny_cfg):
    ids    = torch.randint(0, tiny_cfg.vocab_size, (1, 8))
    logits = tiny_model(ids, n_loops=2)
    loss   = logits.mean()
    loss.backward()
    # At least some parameters should have gradients
    has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in tiny_model.parameters()
    )
    assert has_grad, "No gradients flowed through the model"


# ── Param count sanity ────────────────────────────────────────────────────────

def test_param_count_positive(tiny_model):
    assert tiny_model.param_count() > 0


def test_fable_1b_config():
    """Smoke-test that fable_1b() returns a valid config."""
    cfg = fable_1b()
    assert cfg.dim == 2048
    assert cfg.n_heads == 16
    # Don't instantiate — too large for CI
