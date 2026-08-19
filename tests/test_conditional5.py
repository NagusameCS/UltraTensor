"""Tests for the G1/G8 tooling (bootstrap, sweeps, ablations, eviction)."""

import numpy as np
import pytest

from ultratensor.conditional.stats import (
    bootstrap_ci,
    eviction_ablation,
    fine_k_sweep,
    intrinsic_dim_compare,
    rank_ablation,
    sink_ablation,
)


def test_bootstrap_ci_contains_mean():
    rng = np.random.default_rng(0)
    data = rng.normal(3.0, 1.0, 200)
    r = bootstrap_ci(data, n_resamples=2000)
    assert r.ci_lower <= r.mean <= r.ci_upper
    assert r.ci_lower < r.ci_upper
    # known mean in interval
    assert r.ci_lower <= 3.0 <= r.ci_upper


def test_fine_k_sweep_detects_ramp():
    r = fine_k_sweep(k_star=1024, window=100, step=25)
    assert r["transition_type"] in ("cliff", "ramp")
    assert r["optimal_k"] >= 64


def test_rank_ablation_ranks_by_importance():
    matrices = {"A": 40, "B": 30, "C": 30}

    def quality_fn(alloc):
        # quality driven mainly by A's rank
        return 50.0 + 0.5 * alloc["A"] + 0.05 * alloc["B"] + 0.05 * alloc["C"]

    r = rank_ablation(100, matrices, quality_fn)
    assert r["ranking"][0] == "A"
    assert r["per_matrix"]["A"]["importance"] > r["per_matrix"]["C"]["importance"]


def test_intrinsic_dim_compare_on_gaussian_plane():
    rng = np.random.default_rng(5)
    basis = rng.normal(size=(20, 2))
    X = rng.normal(size=(300, 2)) @ basis.T
    r = intrinsic_dim_compare(X)
    assert r["pca_95"] == 2
    assert 1 <= r["twonn"] <= 4
    assert 1 <= r["mle_levina_bickel"] <= 5


def test_sink_ablation_monotone_benefit():
    r = sink_ablation(k=128)
    errs = [r["results"][f"T={t}"]["error"] for t in (0, 8, 16, 32, 64)]
    assert all(b <= a for a, b in zip(errs, errs[1:]))
    assert r["best_T"] == 64


def test_eviction_lru_beats_random_on_zipf():
    r = eviction_ablation(n_queries=2000, cache_size=50, seed=1)
    assert r["LRU"]["hit_rate"] > r["random"]["hit_rate"]
    assert 0.0 < r["jury_weighted"]["hit_rate"] <= 1.0
