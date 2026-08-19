"""Tests for activation-weighted reconstruction (G4) and ONB coverage."""

import numpy as np
import pytest

from ultratensor.conditional import OnlineBasis
from ultratensor.conditional.actweight import (
    activation_error,
    frob_error,
    heldout_rank_curves,
    projector,
    projector_error,
    rank_error_curve,
    subspace_basis,
    svd_truncate,
    weighted_pca_truncate,
)


def _synthetic_matrix_and_inputs(seed=0, m=24, n=16, t=200, rank=6):
    """W of effective rank r, inputs concentrated in a low-dim subspace."""
    rng = np.random.default_rng(seed)
    U = rng.normal(size=(m, rank))
    V = rng.normal(size=(n, rank))
    W = U @ V.T
    # inputs live mostly in a 4-dim subspace (misaligned with V's top dirs)
    basis = rng.normal(size=(n, 4))
    X = (rng.normal(size=(t, 4)) @ basis.T) + 0.05 * rng.normal(size=(t, n))
    return W, X


def test_svd_truncate_is_best_frob():
    W, _ = _synthetic_matrix_and_inputs()
    W3 = svd_truncate(W, 3)
    assert W3.shape == W.shape
    # rank-3 truncated svd beats a random rank-3 matrix in frob error
    rng = np.random.default_rng(1)
    random3 = (rng.normal(size=(W.shape[0], 3))
               @ rng.normal(size=(3, W.shape[1])))
    assert frob_error(W, W3) < frob_error(W, random3)


def test_frob_and_activation_errors_are_normalized():
    W, X = _synthetic_matrix_and_inputs()
    assert frob_error(W, W) == pytest.approx(0.0, abs=1e-12)
    assert activation_error(W, W, X) == pytest.approx(0.0, abs=1e-12)


def test_weighted_pca_beats_plain_svd_on_act_error():
    W, X = _synthetic_matrix_and_inputs()
    curve = rank_error_curve(W, X, ranks=[2, 4, 6])
    # In the lossy regime (ranks 2, 4), the activation-weighted
    # truncation beats plain SVD on activation error. At exact rank the
    # plain SVD is numerically perfect; weighted stays tiny too.
    assert np.all(curve.act_weighted[:2] <= curve.act[:2] + 1e-12)
    assert curve.act_weighted[0] < curve.act[0]
    assert curve.act_weighted[2] < 1e-3


def test_rank_curve_monotone_decreasing():
    W, X = _synthetic_matrix_and_inputs()
    c = rank_error_curve(W, X, ranks=[1, 2, 4, 8, 16])
    assert np.all(np.diff(c.frob) <= 0)
    assert np.all(np.diff(c.act) <= 0)
    assert c.k95_frob >= 1
    # effective rank 6 -> k95 around 6
    assert 3 <= c.k95_frob <= 8


def test_weighted_truncate_reconstruction():
    W, X = _synthetic_matrix_and_inputs()
    Ww = weighted_pca_truncate(W, X, 4)
    assert Ww.shape == W.shape
    assert activation_error(W, Ww, X) < 1.0


def test_subspace_projector_form():
    W, X = _synthetic_matrix_and_inputs()
    V = subspace_basis(X, 4)
    assert V.shape == (X.shape[1], 4)
    assert np.allclose(V.T @ V, np.eye(4), atol=1e-9)
    A = projector(W, V)
    assert A.shape == (W.shape[0], 4)
    # in-subspace inputs are reproduced exactly by the projector form
    x_in = V @ np.arange(4.0)
    assert np.allclose(W @ x_in, A @ (V.T @ x_in), atol=1e-6)


def test_projector_error_decreases_with_rank():
    W, X = _synthetic_matrix_and_inputs()
    e1 = projector_error(W, projector(W, subspace_basis(X, 1)),
                         subspace_basis(X, 1), X)
    e4 = projector_error(W, projector(W, subspace_basis(X, 4)),
                         subspace_basis(X, 4), X)
    assert e1 < 1.0
    assert e4 < e1


def test_heldout_curves_on_synthetic_low_rank():
    W, X = _synthetic_matrix_and_inputs()  # W rank 6, X in a 4-dim subspace
    c = heldout_rank_curves(W, X[:160], X[160:], ranks=[1, 2, 4, 8])
    assert np.all(np.diff(c["hold_pca"]) <= 0)
    # inputs live in a 4-dim subspace: the projector reaches it at r=4
    assert c["hold_pca"][2] < c["hold_pca"][0]
    assert c["hold_pca"][2] < 0.1
    assert c["hold_weighted"][2] < 0.1
    # weighted fitted on train must not blow up on held-out inputs
    assert np.all(c["hold_weighted"] <= 1.0)


# ---------------------------------------------------------------------------
# ONB coverage (rho-like readout)
# ---------------------------------------------------------------------------

def test_coverage_full_and_missing_directions():
    onb = OnlineBasis(dims=[8], ks=[2], min_rejections_before_update=4)
    e0 = np.zeros(8)
    e0[0] = 1.0
    e1 = np.zeros(8)
    e1[1] = 1.0
    assert onb.coverage(e0, 0) == pytest.approx(1.0)   # identity start
    assert onb.coverage(e1, 0) == pytest.approx(1.0)
    diag = np.full(8, 1.0 / np.sqrt(8))                # spread energy
    assert onb.coverage(diag, 0) == pytest.approx(2.0 / 8.0)


def test_coverage_grows_towards_learned_direction():
    rng = np.random.default_rng(2)
    pc1 = rng.normal(size=16)
    pc1 /= np.linalg.norm(pc1)
    onb = OnlineBasis(dims=[16], ks=[4], eta0=0.1)
    before = onb.coverage(pc1, 0)
    for _ in range(400):
        x = 3.0 * pc1 + 0.05 * rng.normal(size=16)
        onb.record_residual(0, x)
        onb.apply_pending()
    after = onb.coverage(pc1, 0)
    assert after > before
    assert after > 0.9


# ---------------------------------------------------------------------------
# Vision demo: closed-loop ONB x speculative decoding (GSD miniature)
# ---------------------------------------------------------------------------

def test_closed_loop_onb_raises_acceptance():
    """A drafter whose errors live in one direction; ONB learns it from
    rejections and the corrected drafter gets accepted."""
    rng = np.random.default_rng(11)
    dim, k = 8, 1
    d = rng.normal(size=dim)
    d /= np.linalg.norm(d)
    onb = OnlineBasis(dims=[dim], ks=[k], eta0=0.4,
                      min_rejections_before_update=4)

    accepted_early, accepted_late, total = 0, 0, 0
    for step in range(300):
        # "verifier" hidden state for this step
        h_true = rng.normal(size=dim)
        # "draft" is wrong by a fixed displacement along d
        residual = 0.3 * d
        h_draft = h_true + residual
        # correct using the current basis
        W = onb.layers[0].W
        h_corr = h_draft - W.T @ (W @ residual)
        ok = float(np.linalg.norm(h_corr - h_true)) < 0.05
        total += 1
        if step < 50:
            accepted_early += ok
        elif step >= 150:
            accepted_late += ok
        if not ok:
            onb.record_residual(0, residual)
            onb.apply_pending()

    assert onb.coverage(d, 0) > 0.95           # basis found the error direction
    assert accepted_late > accepted_early      # correction got learned
    assert accepted_late / 150 > 0.9
