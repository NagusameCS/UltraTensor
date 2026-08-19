"""G6 — shared-plus-private expert factorization.

The reviews' Phase 3 for Tier B experts:

    W_e ~= W_shared + U A_e V^T + R_e

where U, V are layer-shared input/output dictionaries, A_e is a small
expert-specific core, and R_e is an optional private residual.
Parameter cost changes from E * r * (m + n) (independent factorization)
towards r * (m + n) + E * r^2 — eliminating duplicated bases is the
bigger win than shaving bits.

This module fits the decomposition from a set of expert matrices and
(recommended) their activation inputs, and benchmarks it against
independent per-expert SVD at an equal total parameter budget.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .actweight import activation_error, frob_error


@dataclass
class SharedFactorFit:
    W_shared: np.ndarray
    U: np.ndarray            # [m, r_shared]
    V: np.ndarray            # [n, r_shared] (reconstruction uses V.T)
    A: dict                 # expert -> core [r_shared, r_shared]
    residuals: dict         # expert -> private residual R_e (None if skipped)
    params_per_expert: float
    independent_params_per_expert: float

    def reconstruct(self, e) -> np.ndarray:
        A = self.A[e]
        res = self.residuals.get(e)
        return self.W_shared + (self.U @ A) @ self.V.T + (
            0.0 if res is None else res
        )


def fit_shared_dict(
    experts: dict,
    r_shared: int,
    r_private: int = 0,
    X: np.ndarray | None = None,
    keep_private: bool = False,
) -> SharedFactorFit:
    """Fit W_shared + U A_e V^T (+ optional private rank-r_private R_e).

    Parameters
    ----------
    experts:
        {e: W_e} matrices, all [m, n].
    r_shared:
        Rank of the shared dictionary (columns of U, rows of V).
    r_private:
        Rank of the private residual per expert (0 = skip).
    X:
        Optional [t, n] activation inputs for activation-weighted
        dictionary fitting. When None, plain weight-space SVD is used.
    keep_private:
        Return the fitted private residuals in the result (reconstruction
        includes them only when True).

    Note: U and V are taken from two separate SVDs (rows and columns of
    the stacked residuals) — a cheap one-shot joint fit, not an
    alternating minimisation. It is the right first tool, not the last.
    """
    es = sorted(experts)
    Ws = [np.asarray(experts[e], dtype=np.float64) for e in es]
    m, n = Ws[0].shape
    W_stack = np.stack(Ws)                       # [E, m, n]
    W_shared = W_stack.mean(axis=0)
    R = W_stack - W_shared                       # [E, m, n]

    if r_shared >= min(m, n):
        U, V = np.eye(m), np.eye(n)
    else:
        if X is not None:
            # activation-weighted: fit on the whitened residuals
            Xa = np.asarray(X, dtype=np.float64)
            C = (Xa.T @ Xa) / max(Xa.shape[0], 1)
            evals, evecs = np.linalg.eigh(C)
            evals = np.clip(evals, 1e-30, None)
            C_half = (evecs * np.sqrt(evals)) @ evecs.T
            RW = R @ C_half                       # [E, m, n]
        else:
            RW = R
        # V: top right-singular directions of the stacked rows
        _, _, Vt = np.linalg.svd(RW.reshape(-1, n), full_matrices=False)
        V = Vt[:r_shared].T                       # [n, r_shared]
        # U: top right-singular directions of the stacked columns
        _, _, Vh = np.linalg.svd(
            RW.transpose(0, 2, 1).reshape(-1, m), full_matrices=False)
        U = Vh[:r_shared].T                       # [m, r_shared]

    cores: dict = {}
    residuals: dict = {}
    for e, W in zip(es, Ws):
        R_e = W - W_shared
        cores[e] = U.T @ R_e @ V
        if r_private > 0 and keep_private:
            leftover = R_e - (U @ cores[e]) @ V.T
            Ue, Se, Vte = np.linalg.svd(leftover, full_matrices=False)
            k = min(r_private, len(Se))
            residuals[e] = (Ue[:, :k] * Se[:k]) @ Vte[:k]
        else:
            residuals[e] = None

    return SharedFactorFit(
        W_shared=W_shared,
        U=U,
        V=V,
        A=cores,
        residuals=residuals,
        params_per_expert=r_shared * r_shared + r_private * (m + n),
        independent_params_per_expert=r_shared * (m + n),
    )


def compare_budgets(
    experts: dict,
    X: np.ndarray,
    r_shared: int,
    n_private_grid=(0, 4, 8, 16, 32),
) -> dict:
    """Shared-dict vs independent per-expert SVD at equal-ish budgets."""
    fit = fit_shared_dict(experts, r_shared, keep_private=True, X=X)
    out = {"shared": {}, "independent": {}}
    for e, W in experts.items():
        W = np.asarray(W, dtype=np.float64)
        rec = fit.reconstruct(e)
        out["shared"][e] = {
            "frob": frob_error(W, rec),
            "act": activation_error(W, rec, X),
        }
        # independent baseline at the same shared-rank r
        U, S, Vt = np.linalg.svd(W, full_matrices=False)
        r = min(r_shared, len(S))
        rec_i = (U[:, :r] * S[:r]) @ Vt[:r]
        out["independent"][e] = {
            "frob": frob_error(W, rec_i),
            "act": activation_error(W, rec_i, X),
        }
    return out
