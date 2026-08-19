"""Tests for the curvature-weighted rank allocator (grcurv port)."""

import numpy as np
import pytest

from ultratensor.conditional.curvature import (
    curvature_correlation,
    grcurv_to_rank_budget,
)


def test_budget_respected_and_higher_curvature_gets_more():
    K = np.array([0.01, 0.05, 0.1, 0.5, 1.0, 0.2, 0.02, 0.03])
    ranks, weights = grcurv_to_rank_budget(K, min_rank=8, max_rank=256,
                                           total_rank_budget=800)
    assert ranks.sum() == pytest.approx(800, abs=8)
    assert (ranks >= 8).all() and (ranks <= 256).all()
    # monotone in |curvature| where the clamp does not bind
    assert ranks[4] >= ranks[2] >= ranks[0]


def test_uniform_curvature_gives_uniform_ranks():
    ranks, weights = grcurv_to_rank_budget(np.ones(6), 8, 256, 600)
    assert np.all(ranks == 100)


def test_correlation_honesty_check():
    rng = np.random.default_rng(0)
    K = rng.normal(size=64)
    proxy = 3.0 * K + 0.1 * rng.normal(size=64)
    assert curvature_correlation(K, proxy) > 0.9
    assert curvature_correlation(K, rng.normal(size=64)) == pytest.approx(0.0,
                                                                          abs=0.3)
    # degenerate inputs -> 0, never NaN
    assert curvature_correlation(np.zeros(8), np.zeros(8)) == 0.0


def test_single_layer_edge():
    ranks, weights = grcurv_to_rank_budget(np.array([0.5]), 8, 256, 128)
    assert ranks == np.array([128])
    assert weights == np.array([1.0])
