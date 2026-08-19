"""Tests for G12 tiered residency simulation."""

import numpy as np
import pytest

from ultratensor.conditional.tiering import knee, simulate_tier, tier_sweep


def _trace(n=60, n_experts=32):
    # token sequence cycling; expert sets are deterministic per token
    tokens = [(t * 7 + 3) % n_experts for t in range(n)]
    return [frozenset((tok + 2 * i) % n_experts for i in range(6))
            for tok in tokens]


def _successor(seq):
    """Learned map: current set -> next set's experts (perfect predictor)."""
    succ = {}
    for a, b in zip(seq, seq[1:]):
        succ[a] = sorted(b)
    return lambda current_set: succ.get(frozenset(current_set), [])


def test_perfect_prefetch_zero_misses_at_cap():
    seq = _trace()
    r = simulate_tier(seq, _successor(seq), hot_cap=6)
    assert r.mean_miss_rate == 0.0
    assert r.mean_latency_ms == pytest.approx(0.0)
    assert r.resident_experts <= 32


def test_small_cap_has_misses():
    seq = _trace()

    def stub(current_set):
        return [0, 1, 2, 3, 4, 5]

    big = simulate_tier(seq, stub, hot_cap=64, miss_cost_ms=500.0)
    small = simulate_tier(seq, stub, hot_cap=4, miss_cost_ms=500.0)
    assert small.mean_miss_rate > 0.0
    assert small.mean_latency_ms > big.mean_latency_ms


def test_tier_sweep_and_knee():
    seq = _trace(n=100)
    curve = tier_sweep(seq, _successor(seq), hot_caps=(2, 4, 6, 8, 12))
    lats = [c.mean_latency_ms for c in curve]
    assert all(b <= a + 1e-9 for a, b in zip(lats, lats[1:]))
    k = knee(curve)
    assert k is not None and k.hot_cap == 6


def test_tail_p90_miss_recorded():
    seq = [{1, 2}, {3, 4}, {5, 6}, {7, 8}]

    def prefetch(prev):
        return sorted(prev)

    r = simulate_tier(seq, prefetch, hot_cap=2)
    assert r.step_misses == [2, 2, 2]
    assert r.tail_p90_miss == 2.0
