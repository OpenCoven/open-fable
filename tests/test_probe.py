"""tests/test_probe.py — CoherenceProbe output format and correctness."""

import math
import pytest
import torch
import torch.nn as nn
from open_fable import CoherenceProbe


@pytest.fixture
def probe():
    p = CoherenceProbe(model_dim=128, vocab_size=512, top_k=20, drift_threshold=0.25)
    return p


@pytest.fixture
def bound_probe():
    """Probe with a real lm_head bound."""
    p = CoherenceProbe(model_dim=128, vocab_size=512, top_k=20, drift_threshold=0.25)
    lm_head = nn.Linear(128, 512, bias=False)
    p.bind_lm_head(lm_head)
    return p, lm_head


# ── record_step return type ───────────────────────────────────────────────────

def test_record_step_returns_float(bound_probe):
    probe, _ = bound_probe
    hidden = torch.randn(1, 8, 128)
    score = probe.record_step(hidden)
    assert isinstance(score, float)


def test_record_step_score_in_range(bound_probe):
    probe, _ = bound_probe
    for _ in range(4):
        hidden = torch.randn(1, 4, 128)
        score = probe.record_step(hidden)
        assert 0.0 <= score <= 1.0, f"Score out of range: {score}"


# ── coherence_scores length ───────────────────────────────────────────────────

def test_coherence_scores_length(bound_probe):
    probe, _ = bound_probe
    n = 5
    for _ in range(n):
        hidden = torch.randn(1, 4, 128)
        probe.record_step(hidden)
    scores = probe.coherence_scores()
    assert len(scores) == n


def test_coherence_scores_empty_after_reset(bound_probe):
    probe, _ = bound_probe
    hidden = torch.randn(1, 4, 128)
    probe.record_step(hidden)
    probe.reset()
    assert probe.coherence_scores() == []


# ── report format ─────────────────────────────────────────────────────────────

def test_report_keys(bound_probe):
    probe, _ = bound_probe
    for _ in range(3):
        probe.record_step(torch.randn(1, 4, 128))
    report = probe.report()
    for key in ("n_loops", "scores", "mean", "trend"):
        assert key in report, f"Missing key '{key}' in probe report"


def test_report_n_loops_matches_records(bound_probe):
    probe, _ = bound_probe
    n = 6
    for _ in range(n):
        probe.record_step(torch.randn(1, 4, 128))
    report = probe.report()
    assert report["n_loops"] == n


def test_report_trend_valid_values(bound_probe):
    probe, _ = bound_probe
    for _ in range(4):
        probe.record_step(torch.randn(1, 4, 128))
    report = probe.report()
    assert report["trend"] in ("improving", "degrading", "stable")


def test_report_mean_is_average_of_scores(bound_probe):
    probe, _ = bound_probe
    for _ in range(4):
        probe.record_step(torch.randn(1, 4, 128))
    report  = probe.report()
    scores  = report["scores"]
    expected_mean = round(sum(scores) / len(scores), 4)
    assert abs(report["mean"] - expected_mean) < 1e-3


# ── character drift ───────────────────────────────────────────────────────────

def test_drift_no_records_returns_zero(bound_probe):
    probe, _ = bound_probe
    drift = probe.character_drift([42, 43, 44])
    assert drift == 0.0


def test_drift_returns_float_in_range(bound_probe):
    probe, _ = bound_probe
    for _ in range(4):
        probe.record_step(torch.randn(1, 4, 128))
    drift = probe.character_drift([42, 43, 44])
    assert isinstance(drift, float)
    assert 0.0 <= drift <= 1.0


def test_is_drifting_type(bound_probe):
    probe, _ = bound_probe
    for _ in range(3):
        probe.record_step(torch.randn(1, 4, 128))
    result = probe.is_drifting([10, 11, 12])
    assert isinstance(result, bool)


# ── weight binding ────────────────────────────────────────────────────────────

def test_bind_lm_head_shares_weights(bound_probe):
    probe, lm_head = bound_probe
    # After binding, _proj.weight should be the same object as lm_head.weight
    assert probe._proj.weight is lm_head.weight
    assert probe._bound is True


# ── 3D vs 2D hidden input ─────────────────────────────────────────────────────

def test_record_step_3d_input(bound_probe):
    probe, _ = bound_probe
    hidden = torch.randn(2, 12, 128)   # [batch, seq, dim]
    score  = probe.record_step(hidden)
    assert 0.0 <= score <= 1.0


def test_record_step_2d_input(bound_probe):
    probe, _ = bound_probe
    hidden = torch.randn(2, 128)       # [batch, dim]
    score  = probe.record_step(hidden)
    assert 0.0 <= score <= 1.0


# ── Determinism ───────────────────────────────────────────────────────────────

def test_record_step_deterministic(bound_probe):
    """Same hidden state should always yield the same score."""
    probe, _ = bound_probe
    hidden = torch.randn(1, 4, 128)
    s1 = probe.record_step(hidden)
    probe.reset()
    s2 = probe.record_step(hidden)
    assert abs(s1 - s2) < 1e-5
