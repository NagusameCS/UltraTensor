"""Curvature-weighted rank allocation (port of grcurv_to_rank_budget).

The policy part of HyperTensor's geo_research.c: per-layer sectional
curvature magnitudes weight a rank budget (raw = base * (1 + alpha * w),
normalised, last layer absorbs the remainder), plus the honesty check
the C runtime reports: Pearson correlation between the curvature signal
and the old Frobenius proxy. If that correlation is near zero, the
curvature allocator is not adding information over the cheap proxy and
should not be trusted.

The curvature ESTIMATOR itself stays in HyperTensor; this module takes
per-layer curvature values from any source and applies the policy.
"""

from __future__ import annotations

import numpy as np


def grcurv_to_rank_budget(
    sectional,
    min_rank: int,
    max_rank: int,
    total_rank_budget: int,
    alpha: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Allocate a total rank budget proportional to |sectional curvature|.

    Returns (ranks, weights); weights = |K_l| / max(|K|).
    """
    K = np.abs(np.asarray(sectional, dtype=np.float64))
    n = K.size
    if n == 0:
        return np.zeros(0, dtype=int), np.zeros(0)
    if n == 1:
        return (np.array([total_rank_budget], dtype=int),
                np.array([1.0]))
    K_max = K.max()
    weights = K / K_max if K_max > 1e-20 else np.zeros(n)

    base = total_rank_budget / n
    raw = np.clip(base * (1.0 + alpha * weights), min_rank, max_rank)
    total = raw.sum()
    scale = (total_rank_budget / total) if total > 0.0 else 1.0
    ranks = np.clip(np.round(raw * scale).astype(int), min_rank, max_rank)
    # last layer absorbs the rounding remainder (C semantics)
    ranks[-1] = int(np.clip(total_rank_budget - int(ranks[:-1].sum()),
                            min_rank, max_rank))
    return ranks, weights


def curvature_correlation(sectional, proxy) -> float:
    """Pearson r between curvature and the old Frobenius proxy."""
    a = np.asarray(sectional, dtype=np.float64)
    b = np.asarray(proxy, dtype=np.float64)
    if a.size < 2 or a.std() < 1e-20 or b.std() < 1e-20:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])
