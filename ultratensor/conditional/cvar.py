"""G7 — CVaR tail-risk pruning for expert / latent-direction removal.

The review demand: prune ONLY candidates whose mean damage is low AND
whose tail damage is low, where the tail is the CVaR_alpha — the mean
over the worst (1 - alpha) slice of per-sample damage. Rare-domain
slices (code, math, multilingual, long context) live in the tail, so
the tail must gate removal: "low mean AND low tail", never silent
wrong output on rare traffic.

Damage matrix D [n_candidates, n_samples] holds per-sample conditional
damage, e.g. KL(p_base || p_ablated | e in top-k) for an expert
removal, or reconstruction error per latent direction. Utilities are
input-agnostic: synthetic or real damages.

Real-data note (2026-08-16): no per-expert ablation damages have been
measured on V4 yet. This module is the machinery; the cluster ablation
run (exp96 trace as calibration base) will feed it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def tail_slice_size(n: int, alpha: float) -> int:
    """Number of worst samples entering the CVaR at level alpha."""
    return max(1, int(np.ceil((1.0 - alpha) * n)))


def cvar(D: np.ndarray, alpha: float = 0.9) -> np.ndarray:
    """Per-candidate CVaR_alpha: mean of the worst (1-alpha) samples."""
    D = np.asarray(D, dtype=np.float64)
    k = tail_slice_size(D.shape[-1], alpha)
    top = np.sort(D, axis=-1)[..., -k:]
    return top.mean(axis=-1)


@dataclass
class TailStats:
    mean: np.ndarray
    cvar: np.ndarray
    worst: np.ndarray
    alpha: float


def tail_stats(D: np.ndarray, alpha: float = 0.9) -> TailStats:
    """Mean / CVaR_alpha / worst per candidate (last axis = samples)."""
    D = np.asarray(D, dtype=np.float64)
    return TailStats(mean=D.mean(axis=-1), cvar=cvar(D, alpha),
                     worst=D.max(axis=-1), alpha=alpha)


def prune_mask(
    D: np.ndarray,
    alpha: float = 0.9,
    tau_mean: float | None = None,
    tau_cvar: float | None = None,
) -> np.ndarray:
    """Safe-to-prune mask: low mean AND low tail.

    Defaults: tau_mean / tau_cvar are the medians across candidates of
    the respective statistic (the bottom half on both gates). D must be
    finite — mask unobserved samples (NaN) before calling.
    """
    D = np.asarray(D, dtype=np.float64)
    means = D.mean(axis=-1)
    tails = cvar(D, alpha)
    if tau_mean is None:
        tau_mean = float(np.median(means))
    if tau_cvar is None:
        tau_cvar = float(np.median(tails))
    return (means <= tau_mean) & (tails <= tau_cvar)


def cvar_ci(
    D: np.ndarray,
    alpha: float = 0.9,
    ci_level: float = 0.95,
    n_resamples: int = 4000,
    seed: int = 42,
):
    """Bootstrap CI for per-candidate CVaR_alpha (percentile method).

    Returns (lo, hi) ndarrays over candidates.
    """
    rng = np.random.default_rng(seed)
    D = np.asarray(D, dtype=np.float64)
    n = D.shape[-1]
    k = tail_slice_size(n, alpha)
    idx = rng.integers(0, n, size=(n_resamples, n))
    top = np.sort(D[:, idx], axis=-1)[..., -k:]
    cv = top.mean(axis=-1)                       # [cand, n_resamples]
    a = (1.0 - ci_level) / 2.0
    lo = np.percentile(cv, 100 * a, axis=-1)
    hi = np.percentile(cv, 100 * (1.0 - a), axis=-1)
    return lo, hi


def prune_report(
    D: np.ndarray,
    alpha: float = 0.9,
    tau_mean: float | None = None,
    tau_cvar: float | None = None,
) -> dict:
    """JSON-ready pruning decision with per-candidate statistics."""
    D = np.asarray(D, dtype=np.float64)
    ts = tail_stats(D, alpha)
    lo, hi = cvar_ci(D, alpha)
    mask = prune_mask(D, alpha, tau_mean, tau_cvar)
    return {
        "alpha": alpha,
        "n_candidates": int(D.shape[0]),
        "n_samples": int(D.shape[1]),
        "tau_mean": float(tau_mean) if tau_mean is not None
        else float(np.median(ts.mean)),
        "tau_cvar": float(tau_cvar) if tau_cvar is not None
        else float(np.median(ts.cvar)),
        "pruned": [int(i) for i in np.flatnonzero(mask)],
        "per_candidate": [
            {
                "mean": round(float(ts.mean[i]), 6),
                "cvar": round(float(ts.cvar[i]), 6),
                "worst": round(float(ts.worst[i]), 6),
                "cvar_ci": [round(float(lo[i]), 6), round(float(hi[i]), 6)],
                "safe_to_prune": bool(mask[i]),
            }
            for i in range(D.shape[0])
        ],
    }
