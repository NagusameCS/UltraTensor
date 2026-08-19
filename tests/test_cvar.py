"""Tests for G7 CVaR tail-risk pruning (ultratensor.conditional.cvar)."""

import numpy as np
import pytest

from ultratensor.conditional.cvar import (
    cvar,
    cvar_ci,
    prune_mask,
    prune_report,
    tail_slice_size,
    tail_stats,
)


def test_cvar_is_tail_mean():
    rng = np.random.default_rng(0)
    D = rng.normal(size=(3, 200))
    a = cvar(D, alpha=0.9)
    k = tail_slice_size(200, 0.9)
    assert k == 20
    for i in range(3):
        assert a[i] == pytest.approx(np.sort(D[i])[-k:].mean())
        # CVaR >= mean for any distribution
        assert a[i] >= D[i].mean() - 1e-12


def test_prune_mask_requires_low_mean_and_low_tail():
    # candidate 0: low mean, low tail -> prune
    # candidate 1: low mean, high tail (rare-domain blowup) -> keep
    # candidate 2: high mean -> keep
    D = np.array([
        [0.01] * 100,
        [0.01] * 90 + [5.0] * 10,
        [3.0] * 100,
    ])
    mask = prune_mask(D, alpha=0.9)
    assert mask[0] and not mask[1] and not mask[2]


def test_prune_mask_explicit_thresholds():
    D = np.array([[0.0, 0.1], [1.0, 1.1], [0.0, 0.2]])
    # row 2 passes the mean gate (0.1 <= 0.15) but FAILS the tail gate
    # (CVaR 0.2 > 0.15): the tail must veto, not just the mean.
    mask = prune_mask(D, alpha=0.5, tau_mean=0.15, tau_cvar=0.15)
    assert mask[0] and not mask[1] and not mask[2]


def test_cvar_ci_contains_estimate():
    rng = np.random.default_rng(1)
    D = rng.normal(size=(2, 100))
    lo, hi = cvar_ci(D, alpha=0.9, n_resamples=500)
    a = cvar(D, alpha=0.9)
    assert np.all(lo <= a + 1e-9) and np.all(hi >= a - 1e-9)


def test_tail_stats_and_small_n():
    D = np.array([[0.0, 0.5, 2.0]])
    ts = tail_stats(D, alpha=2 / 3)
    assert ts.worst[0] == 2.0
    # k = ceil(1/3 * 3) = 1 -> CVaR is the single worst sample
    assert ts.cvar[0] == pytest.approx(2.0)
    assert ts.mean[0] == pytest.approx((0.0 + 0.5 + 2.0) / 3)


def test_prune_report_json_ready():
    D = np.array([[0.0, 0.1, 0.2], [5.0, 5.1, 5.2]])
    r = prune_report(D, alpha=2 / 3)
    assert r["n_candidates"] == 2 and r["n_samples"] == 3
    assert r["pruned"] == [0]
    assert r["per_candidate"][1]["safe_to_prune"] is False
    # CI bounds contain the CVaR point estimate
    c = r["per_candidate"][0]
    assert c["cvar_ci"][0] <= c["cvar"] <= c["cvar_ci"][1]
