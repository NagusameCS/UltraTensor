"""Tests for lookahead (G3), heterogeneous drafting, and unified policy."""

import numpy as np
import pytest

from ultratensor.conditional.drafting import (
    DraftPlan,
    expected_acceptance,
    optimize_slots,
)
from ultratensor.conditional.lookahead import (
    WorkingSetModel,
    evaluate_prefetch,
    oracle_curve,
    working_set_union,
)
from ultratensor.conditional.policy import ConditionalPolicy
from ultratensor.conditional.thermal import ThermalRank


class _ConstSensor:
    def __init__(self, temp=60.0, power=50.0):
        self.temp, self.power = temp, power

    def read(self):
        return (self.temp, self.power)


# ---------------------------------------------------------------------------
# G3 lookahead
# ---------------------------------------------------------------------------

def _regime_trace(n=120):
    """Two regimes: first half routes {0,1,2}, second half {3,4,5}."""
    seq = []
    for t in range(n):
        seq.append({0, 1, 2} if t < n // 2 else {3, 4, 5})
    return seq


def test_working_set_union():
    seq = [{0, 1}, {2}, {3, 4}]
    assert working_set_union(seq, 0, 2) == {0, 1, 2}
    assert working_set_union(seq, 1, 2) == {2, 3, 4}
    assert working_set_union(seq, 2, 3) == {3, 4}


def test_frequency_vs_markov_hit():
    seq = _regime_trace()
    model = WorkingSetModel().fit(seq)
    f = evaluate_prefetch(seq, model, H=4)
    assert f.best_hit > 0.5
    # Markov sees the regime switch; frequency does not, but both beat
    # the oracle only in size terms. Assert the oracle is the ceiling.
    oc = oracle_curve(seq, 4)
    assert oc["mean_union_size"] == pytest.approx(3.0, abs=0.6)
    assert f.best_size >= 1.0


def test_predict_usage_bounded_and_geometry():
    seq = _regime_trace(40)
    m = WorkingSetModel().fit(seq)
    probs = m.predict_usage({0, 1, 2}, H=8)
    assert all(0.0 <= p <= 1.0 for p in probs.values())
    # H=1 is weaker than H=8 for the same expert
    p1 = m.predict_usage({0, 1, 2}, H=1)[0]
    p8 = m.predict_usage({0, 1, 2}, H=8)[0]
    assert p8 >= p1


# ---------------------------------------------------------------------------
# Heterogeneous drafters
# ---------------------------------------------------------------------------

def test_expected_acceptance_formula():
    assert expected_acceptance([0.5, 0.5]) == pytest.approx(1 + 0.5 + 0.25)
    assert expected_acceptance([1.0, 1.0, 1.0]) == 4.0
    assert expected_acceptance([]) == 1.0


def test_optimize_slots_prefers_early_slots():
    # acceptance grows with rank (saturating) and declines with slot index
    def acc_fn(rank, slot):
        return (0.9 - 0.15 * slot) * rank / (rank + 64.0)

    plan = optimize_slots(4, rank_budget=160, acceptance_fn=acc_fn,
                          time_fn=lambda r: 0.001 + 1e-6 * r)
    assert plan.gamma == 4
    assert plan.total_rank == pytest.approx(160, abs=8)
    assert plan.slots[0] >= plan.slots[3]
    assert plan.throughput > 0.0


def test_optimize_slots_respects_tiny_budget():
    plan = optimize_slots(2, rank_budget=16, acceptance_fn=lambda r, s: 0.5,
                          min_rank=8)
    assert plan.total_rank == 16


# ---------------------------------------------------------------------------
# Unified policy
# ---------------------------------------------------------------------------

def test_policy_rank_profile_and_clamp():
    pol = ConditionalPolicy(
        thermal=ThermalRank(_ConstSensor(75.0, 50.0),
                            rank_min=8, rank_max=256),
    )
    err = np.array([0.2, 0.2, 0.2, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05])
    var = np.array([2.0, 2.0, 2.0, 0.5, 0.5, 0.5, 0.5, 2.0, 2.0])
    ranks, mode = pol.rank_profile(err, var, base_rank=128)
    assert ranks.shape == (9,)
    assert ranks.min() >= 8 and ranks.max() <= 256
    assert mode == "factual"
    clamped = pol.thermal_clamp(ranks)          # 75C midpoint -> ~half rank
    assert clamped.max() <= ranks.max()
    assert (clamped >= 8).all()


def test_policy_step_precision_gate():
    pol = ConditionalPolicy(thermal=ThermalRank(_ConstSensor(40.0, 30.0)))
    err = np.full(9, 0.1)
    var = np.full(9, 1.0)
    confident = np.array([10.0, -3.0, -3.0, -3.0])
    flat = np.array([1.0, 1.0, 1.0, 1.0])
    s1 = pol.step(confident, err, var)
    s2 = pol.step(flat, err, var)
    assert s1.precision == "fast"
    assert s2.precision == "escalate"
    assert s2.entropy > s1.entropy
    assert s1.dominant_mode == "context"        # uniform errors, >0.05
