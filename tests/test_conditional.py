"""Tests for ultratensor.conditional — ports of HyperTensor's conditional
runtime mechanisms (APC, thermal rank, qspec, frank, ONB, MCR, sinks,
jury)."""

import numpy as np
import pytest

from ultratensor.conditional import (
    OnlineBasis,
    NvmlCtypesSensor,
    ThermalRank,
    TpjTracker,
    apc_gate,
    apc_stats,
    confidence_to_jury,
    domain_route,
    frank_apply,
    frank_build,
    geodesic_confidence,
    jury_confidence,
    mcr_detect_phases,
    mcr_rank_budget,
    qspec_test_shared_basis,
    shannon_entropy,
    sink_check_basis_coverage,
    sink_detect,
)


class FakeSensor:
    def __init__(self, temp, power):
        self.temp = temp
        self.power = power
        self.calls = 0

    def read(self):
        self.calls += 1
        return (self.temp, self.power)


# ---------------------------------------------------------------------------
# APC — adaptive precision cascade
# ---------------------------------------------------------------------------

def test_shannon_entropy_extremes():
    n = 8
    assert shannon_entropy(np.full(n, 1.0 / n)) == pytest.approx(np.log2(n), abs=1e-9)
    onehot = np.zeros(n)
    onehot[0] = 1.0
    assert shannon_entropy(onehot) == pytest.approx(0.0, abs=1e-9)


def test_apc_gate_fast_vs_escalate():
    confident = np.array([10.0, -3.0, -3.0, -3.0])   # entropy ~ 0
    flat = np.array([1.0, 1.0, 1.0, 1.0])            # entropy = 2 bits
    stats = apc_stats()
    assert apc_gate(confident, stats=stats)[0] is True
    assert apc_gate(flat, stats=stats)[0] is False
    assert stats.total_inferences == 2
    assert stats.fast_hits == 1
    assert stats.escalations == 1
    assert stats.hit_rate() == 0.5


# ---------------------------------------------------------------------------
# Thermal rank + TPJ
# ---------------------------------------------------------------------------

def test_thermal_rank_clamp_curve():
    tr = ThermalRank(FakeSensor(40.0, 50.0), rank_min=8, rank_max=256)
    tr.sensor.temp = 40.0
    assert tr.get_rank(128) == 256          # below low -> max
    tr.sensor.temp = 95.0
    assert tr.get_rank(128) == 8            # above high -> min
    tr.sensor.temp = 75.0                   # midpoint -> ~132
    assert tr.get_rank(128) == pytest.approx(132, abs=2)


def test_thermal_power_budget_downscales():
    tr = ThermalRank(FakeSensor(50.0, 200.0), power_budget_w=100.0, rank_min=8, rank_max=256)
    tr.sensor.temp = 50.0                   # would give rank_max
    assert tr.get_rank(128) == pytest.approx(128, abs=2)  # 256 * 100/200


def test_thermal_no_sensor_returns_base():
    tr = ThermalRank(None)  # type: ignore[arg-type]  (read() returns None)
    assert tr.get_rank(77) == 77


def test_tpj_records_and_gradient():
    tr = ThermalRank(FakeSensor(50.0, 120.0))
    tpj = TpjTracker(tr)
    for tps in (0.5, 0.55, 0.6, 0.65, 0.7):
        jpt = tpj.record(tps)
        assert jpt > 0.0
    assert tpj.cumulative_tokens == 5
    assert tpj.rank_coeff > 0.0
    probs = np.full((2, 6), 1.0 / 6)
    rank_soft = np.array([100.0, 200.0])
    g = tpj.gradient(probs, rank_soft)
    assert g.shape == (2, 6)
    exp_rank = float(tpj.rank_levels.mean())          # E[R] under uniform p
    assert g[0].sum() == pytest.approx(tpj.lambda_ * tpj.rank_coeff * (exp_rank - 100.0))
    assert g[1].sum() < 0.0                            # above mean rank -> negative push


# ---------------------------------------------------------------------------
# qspec shared-basis feasibility + frank
# ---------------------------------------------------------------------------

def test_qspec_alignment_perfect_and_broken():
    rows = [
        (0, 1, 0.90, np.sqrt(0.1)),  # frob = sqrt(0.1) -> svd_expl 0.90 -> align 1.0
        (0, 2, 0.10, 0.5),           # svd_expl 0.75 vs 0.10 -> align 0.133
    ]
    r = qspec_test_shared_basis(rows, share_threshold=0.8)
    assert r.entries[0].alignment == pytest.approx(1.0)
    assert r.entries[0].shared_ok
    assert r.entries[1].alignment == pytest.approx(0.10 / 0.75, abs=1e-6)
    assert r.worst().slot == 2
    assert "recomputation" in r.verdict()


def test_frank_factual_dominant():
    err = np.array([0.20, 0.20, 0.20, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05])
    r = frank_build(err)
    assert r.valid and r.dominant_mode == "factual"
    ranks = frank_apply(r, np.full(9, 100), min_rank=8, max_rank=256)
    assert ranks[0] > ranks[5]              # early layers boosted
    assert ranks.min() >= 8 and ranks.max() <= 256


def test_frank_context_uniform():
    err = np.full(9, 0.2)
    r = frank_build(err)
    assert r.dominant_mode == "context"
    ranks = frank_apply(r, np.full(9, 100), min_rank=8, max_rank=256)
    assert np.all(ranks == pytest.approx(180, abs=1))  # uniform boost


# ---------------------------------------------------------------------------
# ONB — online basis via Oja's rule
# ---------------------------------------------------------------------------

def test_online_basis_converges_to_pc1():
    rng = np.random.default_rng(7)
    pc1 = rng.normal(size=16)
    pc1 /= np.linalg.norm(pc1)
    onb = OnlineBasis(dims=[16], ks=[4], eta0=0.1)
    for _ in range(400):
        x = 3.0 * pc1 + 0.05 * rng.normal(size=16)
        onb.record_residual(0, x)
        onb.apply_pending()
    best = max(abs(onb.layers[0].W @ pc1))
    assert best > 0.95


def test_online_basis_gate_and_reproject():
    onb = OnlineBasis(dims=[8], ks=[2], min_rejections_before_update=4)
    assert onb.apply_pending() == 0        # gate: fewer than 4 rejections
    for _ in range(3):
        onb.record_rejection(0, np.ones(8), np.zeros(8))
    assert onb.apply_pending() == 0        # still 3 rejections -> gated
    onb.record_rejection(0, np.ones(8), np.zeros(8))
    assert onb.apply_pending() == 4        # gate opens, all 4 processed
    W_orig = np.arange(24, dtype=float).reshape(3, 8)
    proj = onb.reproject_weight(W_orig, 0)
    assert proj.shape == (3, 2)
    assert np.allclose(proj, W_orig @ onb.layers[0].W.T)


# ---------------------------------------------------------------------------
# MCR phases + attention sinks
# ---------------------------------------------------------------------------

def test_mcr_detects_compress_zone():
    var = np.array([2.0, 2.0, 2.0, 0.5, 0.5, 0.5, 0.5, 2.0, 2.0])
    r = mcr_detect_phases(var, compress_thr=2.0)
    assert r.phases_valid
    assert r.phases[3:7] == ["compress"] * 4
    ranks = mcr_rank_budget(r, total_budget=900, min_rank=8, max_rank=256,
                            mix_scale=1.0, compress_scale=0.5, refine_scale=1.0)
    assert ranks.sum() == pytest.approx(900, abs=9)
    assert ranks[4] < ranks[0]


def test_mcr_flat_profile_invalid():
    r = mcr_detect_phases(np.full(12, 1.0))
    assert not r.phases_valid
    ranks = mcr_rank_budget(r, total_budget=240, min_rank=8, max_rank=256)
    assert np.all(ranks == 20)


def test_sink_detect_and_coverage():
    norms = np.array([5.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 6.0, 1.0])
    s = sink_detect(norms, sigma_threshold=2.0)
    assert s.valid
    assert set(s.indices.tolist()) == {0, 8}  # 0 = BOS rule, 8 = sigma outlier
    dim = 4
    sink_dir = np.array([1.0, 0.0, 0.0, 0.0])
    basis = np.eye(2, dim)                 # spans e0..e1, sink IS covered
    covered, extra = sink_check_basis_coverage(s, sink_dir, basis)
    assert covered and extra is None
    basis_missing = np.array([[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
    covered, extra = sink_check_basis_coverage(s, sink_dir, basis_missing)
    assert not covered
    assert np.allclose(extra, sink_dir)


# ---------------------------------------------------------------------------
# Jury aggregation
# ---------------------------------------------------------------------------

def test_jury_boundaries_and_monotonicity():
    assert jury_confidence([]) == pytest.approx(0.0)
    assert jury_confidence([1.0, 0.0]) == pytest.approx(1.0)
    a = jury_confidence([0.5, 0.3])
    b = jury_confidence([0.6, 0.3])
    assert b > a


def test_geodesic_confidence_and_domain_route():
    d = np.array([0.0, 1.0, 2.0])
    c = geodesic_confidence(d, coverage_radius=1.0)
    assert c[0] == pytest.approx(1.0)
    assert confidence_to_jury(d, 1.0) == pytest.approx(jury_confidence(c))
    sim = np.array([0.95, 0.93, 0.50, 0.48, 0.10, 0.09])
    ids = ["math", "math", "code", "code", "trivia", "trivia"]
    dom, transfer, tau = domain_route(sim, ids, {"math": 0.013, "code": 0.02, "trivia": 0.03})
    assert dom == "math" and transfer == "code"
    assert tau == pytest.approx(1.0 - 2 * 0.013)


def test_nvml_sensor_graceful_without_gpu_ok_attribute():
    # Must not raise at construction time, regardless of hardware.
    sensor = NvmlCtypesSensor()
    assert sensor.read() is None or isinstance(sensor.read(), tuple)
    sensor.close()
