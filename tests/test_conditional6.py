"""Tests for G6 shared-plus-private factorization."""

import numpy as np
import pytest

from ultratensor.conditional.actweight import activation_error, frob_error
from ultratensor.conditional.shared_factor import compare_budgets, fit_shared_dict


def _experts_with_shared_subspace(E=8, m=24, n=20, n_dir=6, seed=0):
    """Experts = common mean + per-expert combos of a SHARED 6-dim
    subspace: the structure a shared dictionary should exploit."""
    rng = np.random.default_rng(seed)
    W0 = rng.normal(size=(m, n)) * 0.1
    dirs = [rng.normal(size=(m, 1)) @ rng.normal(size=(1, n))
            for _ in range(n_dir)]
    experts = {}
    for e in range(E):
        coeffs = rng.normal(size=n_dir) * 0.5
        experts[e] = W0 + sum(c * d for c, d in zip(coeffs, dirs))
    X = rng.normal(size=(100, n))
    return experts, X


def test_identical_experts_collapse_to_shared_mean():
    rng = np.random.default_rng(1)
    W = rng.normal(size=(12, 10))
    experts = {e: W.copy() for e in range(4)}
    fit = fit_shared_dict(experts, r_shared=4)
    assert frob_error(W, fit.W_shared) < 1e-12
    rec = fit.reconstruct(0)
    assert frob_error(W, rec) < 1e-12


def test_shared_dict_beats_independent_at_equal_budget():
    experts, X = _experts_with_shared_subspace()
    # budget: shared r_sh^2 params/expert (~100 for r_sh=10) vs
    # independent r_ind*(m+n) params/expert (96 for r_ind=2)
    fit = fit_shared_dict(experts, r_shared=10, X=X)
    shared_errs, ind_errs = [], []
    for e, W in experts.items():
        W = np.asarray(W, dtype=np.float64)
        shared_errs.append(activation_error(W, fit.reconstruct(e), X))
        U, S, Vt = np.linalg.svd(W, full_matrices=False)
        W2 = (U[:, :2] * S[:2]) @ Vt[:2]
        ind_errs.append(activation_error(W, W2, X))
    assert np.mean(shared_errs) < 0.25 * np.mean(ind_errs)


def test_private_residual_reduces_error():
    experts, X = _experts_with_shared_subspace(seed=2)
    no_priv = fit_shared_dict(experts, r_shared=4, X=X)
    with_priv = fit_shared_dict(experts, r_shared=4, r_private=6, X=X,
                                keep_private=True)
    for e in experts:
        W = np.asarray(experts[e], dtype=np.float64)
        assert frob_error(W, with_priv.reconstruct(e)) < \
            frob_error(W, no_priv.reconstruct(e))


def test_compare_budgets_reports_both():
    experts, X = _experts_with_shared_subspace(E=4)
    out = compare_budgets(experts, X, r_shared=8)
    assert set(out) == {"shared", "independent"}
    e0 = next(iter(out["shared"]))
    assert "frob" in out["shared"][e0] and "act" in out["shared"][e0]
