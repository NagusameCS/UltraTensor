"""G4 — activation-weighted reconstruction (the review's E_{l,e,s}(r)).

The reviews' decisive metric for expert compression is NOT the
Frobenius norm ||W - What||_F but the activation-weighted distortion:

    E(r) = E_x || (W - W_r) x ||_2^2

measured on the ACTUAL routed inputs of that expert. Weight-energy
(Frobenius/SVD) is a weak proxy; this module computes both, plus the
activation-weighted rank profile and the weighted-PCA basis that is the
right starting point for any low-rank factorization.

    W_r = U_r S_r V_r^T     (plain SVD truncation)
    W_r^w = (whitened-PCA)  (activation-weighted; used for comparison)

The module is input-agnostic: feed synthetic or real hidden states.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RankErrorCurve:
    ranks: np.ndarray
    frob: np.ndarray          # ||W - W_r||_F^2 / ||W||_F^2
    act: np.ndarray           # E_x ||(W - W_r)x||^2 / E_x ||Wx||^2
    act_weighted: np.ndarray  # same, using the weighted-PCA truncation
    k95_frob: int
    k95_act: int


def svd_truncate(W: np.ndarray, r: int) -> np.ndarray:
    """Best rank-r Frobenius approximation of W."""
    W = np.asarray(W, dtype=np.float64)
    U, S, Vt = np.linalg.svd(W, full_matrices=False)
    if r >= len(S):
        return W.copy()
    return (U[:, :r] * S[:r]) @ Vt[:r]


def activation_error(W: np.ndarray, W_hat: np.ndarray, X: np.ndarray) -> float:
    """E_x ||(W - W_hat)x||^2, x rows of X."""
    d = W - np.asarray(W_hat, dtype=np.float64)
    e = np.einsum("ij,ij->i", X @ d.T, X @ d.T)
    denom = np.einsum("ij,ij->i", X @ W.T, X @ W.T)
    return float(e.sum() / max(denom.sum(), 1e-30))


def frob_error(W: np.ndarray, W_hat: np.ndarray) -> float:
    """||W - W_hat||_F^2 / ||W||_F^2."""
    W = np.asarray(W, dtype=np.float64)
    d = W - np.asarray(W_hat, dtype=np.float64)
    denom = float(np.sum(W * W))
    return float(np.sum(d * d) / max(denom, 1e-30))


def weighted_pca_truncate(W: np.ndarray, X: np.ndarray, r: int) -> np.ndarray:
    """Rank-r approximation aligned to the activation distribution.

    The activation-weighted operator is W C^{1/2} where C = E[xx^T]; its
    top-r right singular directions are the directions that matter for
    the inputs. Return W projected onto those directions.
    """
    W = np.asarray(W, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    C = (X.T @ X) / max(X.shape[0], 1)
    # whitening on the row-space of W via C^{1/2} (symmetrised, clipped)
    evals, evecs = np.linalg.eigh(C)
    evals = np.clip(evals, 1e-30, None)
    C_half = (evecs * np.sqrt(evals)) @ evecs.T
    U, S, Vt = np.linalg.svd(W @ C_half, full_matrices=False)
    if r >= len(S):
        return W.copy()
    P = C_half @ Vt[:r].T       # [n, r] input-side projector
    return W @ (P @ np.linalg.pinv(P))


def rank_error_curve(
    W: np.ndarray,
    X: np.ndarray,
    ranks=None,
) -> RankErrorCurve:
    """Frobenius vs activation-weighted error curves over rank r."""
    W = np.asarray(W, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    U, S, Vt = np.linalg.svd(W, full_matrices=False)
    s2 = S * S
    total = s2.sum()
    if ranks is None:
        m = min(W.shape)
        ranks = np.unique(np.concatenate([
            np.linspace(1, m, min(m, 20), dtype=int),
            np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]),
        ]))
        ranks = ranks[(ranks >= 1) & (ranks <= m)]

    frob, act, actw = [], [], []
    for r in ranks:
        W_r = (U[:, :r] * S[:r]) @ Vt[:r]
        frob.append(1.0 - s2[:r].sum() / total)
        act.append(activation_error(W, W_r, X))
        actw.append(activation_error(W, weighted_pca_truncate(W, X, int(r)), X))

    cum = np.cumsum(s2) / total
    k95_frob = int(np.searchsorted(cum, 0.95) + 1)
    return RankErrorCurve(
        ranks=np.asarray(ranks, dtype=int),
        frob=np.asarray(frob),
        act=np.asarray(act),
        act_weighted=np.asarray(actw),
        k95_frob=min(k95_frob, len(S)),
        k95_act=int(np.argmax(np.asarray(act) <= 0.05)) + 1
        if (np.asarray(act) <= 0.05).any() else int(np.argmin(np.asarray(act))) + 1,
    )


# ---------------------------------------------------------------------------
# Deployable projector form: W ~= W P_k with P_k = V_k V_k^T, where V_k
# are the top-k right singular directions of the routed inputs. Per expert
# the one-time precompute is A = W @ V_k [m, k]; per token the GEMM
# shrinks from m*d to k*(m+d) MACs:
#
#     y ~= A (V_k^T x)
#
# rank_error_curve above answers "what rank does the activation say";
# heldout_rank_curves below answers "does that rank survive unseen
# tokens" — the honest number before any kernel work.
# ---------------------------------------------------------------------------


def subspace_basis(X: np.ndarray, k: int) -> np.ndarray:
    """Top-k right singular vectors V_k [d, k] of the routed inputs."""
    X = np.asarray(X, dtype=np.float64)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    k = int(min(k, Vt.shape[0]))
    return np.ascontiguousarray(Vt[:k].T)


def projector(W: np.ndarray, V: np.ndarray) -> np.ndarray:
    """A = W @ V_k: the one-time [m, k] precompute for the projector form."""
    return np.asarray(W, dtype=np.float64) @ np.asarray(V, dtype=np.float64)


def projector_error(
    W: np.ndarray,
    A: np.ndarray,
    V: np.ndarray,
    X: np.ndarray,
) -> float:
    """E_x ||Wx - A V^T x||^2 / E_x ||Wx||^2, x rows of X."""
    Xf = np.asarray(X, dtype=np.float64)
    Wx = Xf @ np.asarray(W, dtype=np.float64).T
    yhat = (Xf @ np.asarray(V, dtype=np.float64)) @ np.asarray(
        A, dtype=np.float64).T
    denom = float(np.sum(Wx * Wx))
    return float(np.sum((Wx - yhat) ** 2) / max(denom, 1e-30))


def _weighted_svd(W: np.ndarray, X: np.ndarray):
    """Activation-weighted SVD precomputed once, shared across ranks."""
    W = np.asarray(W, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    C = (X.T @ X) / max(X.shape[0], 1)
    evals, evecs = np.linalg.eigh(C)
    evals = np.clip(evals, 1e-30, None)
    C_half = (evecs * np.sqrt(evals)) @ evecs.T
    U, S, Vt = np.linalg.svd(W @ C_half, full_matrices=False)
    return C_half, U, S, Vt


def weighted_truncate_from_svd(
    W: np.ndarray,
    C_half: np.ndarray,
    U: np.ndarray,
    S: np.ndarray,
    Vt: np.ndarray,
    r: int,
) -> np.ndarray:
    """Rank-r activation-weighted truncation from a precomputed _weighted_svd."""
    W = np.asarray(W, dtype=np.float64)
    if r >= len(S):
        return W.copy()
    P = C_half @ Vt[:r].T           # [d, r] oblique input-side basis
    return W @ (P @ np.linalg.pinv(P))


def heldout_rank_curves(
    W: np.ndarray,
    X_train: np.ndarray,
    X_hold: np.ndarray,
    ranks=None,
) -> dict:
    """Fit projector on train, evaluate on held-out tokens.

    Returns dict with keys:
      ranks          -> int ndarray
      train_pca      -> plain PCA projector, fit+eval on train
      hold_pca       -> plain PCA projector, fit on train / eval on hold
      train_weighted -> activation-weighted (whitened) truncation, train
      hold_weighted  -> activation-weighted truncation, fit train / hold

    All curves are the normalized E_x ||Wx - What x||^2 / E_x ||Wx||^2.
    Ranks are capped at the number of training samples for the PCA
    projector (it can only discover train-rank directions).
    """
    W = np.asarray(W, dtype=np.float64)
    X_train = np.asarray(X_train, dtype=np.float64)
    X_hold = np.asarray(X_hold, dtype=np.float64)
    m, d = W.shape
    cap = min(m, d, X_train.shape[0])
    if ranks is None:
        ranks = np.unique(np.concatenate([
            np.array([1, 2, 4, 8, 12, 16, 24, 32], dtype=int),
            np.linspace(1, cap, min(cap, 12), dtype=int),
        ]))
    ranks = np.asarray([r for r in ranks if 1 <= r <= cap], dtype=int)

    _, _, Vt_full = np.linalg.svd(X_train, full_matrices=False)
    C_half, Uw, Sw, Vtw = _weighted_svd(W, X_train)
    train_pca, hold_pca = [], []
    train_w, hold_w = [], []
    for r in ranks:
        V = np.ascontiguousarray(Vt_full[:r].T)
        A = W @ V
        train_pca.append(projector_error(W, A, V, X_train))
        hold_pca.append(projector_error(W, A, V, X_hold))
        Wh = weighted_truncate_from_svd(W, C_half, Uw, Sw, Vtw, int(r))
        train_w.append(activation_error(W, Wh, X_train))
        hold_w.append(activation_error(W, Wh, X_hold))
    return {
        "ranks": ranks,
        "train_pca": np.asarray(train_pca),
        "hold_pca": np.asarray(hold_pca),
        "train_weighted": np.asarray(train_w),
        "hold_weighted": np.asarray(hold_w),
    }
