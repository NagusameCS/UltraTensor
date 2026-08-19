"""G5 — conditional-rank policy comparison.

Four allocation policies over a per-layer importance signal:
  fixed  : uniform budget
  frank  : early/mid/late failure-mode scaling (basis.frank)
  mcr    : Mix/Compress/Refine phase scaling (sinks.mcr)
  grcurv : |curvature|-weighted (curvature.grcuv)

Reports per-policy rank vectors plus concentration (Gini) so the
question "is conditional rank buying us anything on this signal?" is
answered numerically before any experiment. Signals on real bytes are
staged behind phase B; this module is the policy layer.
"""

from __future__ import annotations

import numpy as np

from .basis import frank_build
from .curvature import grcurv_to_rank_budget
from .sinks import mcr_detect_phases, mcr_rank_budget


def gini(x: np.ndarray) -> float:
    """Gini concentration in [0, 1]: 0 uniform, 1 single layer."""
    x = np.sort(np.asarray(x, dtype=np.float64))
    n = x.size
    if n == 0 or x.sum() <= 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * idx - n - 1) @ x / (n * x.sum()))


def fixed_allocation(n: int, total: int, min_rank: int,
                     max_rank: int) -> np.ndarray:
    base = int(np.clip(total // n, min_rank, max_rank))
    return np.full(n, base, dtype=int)


def compare_policies(
    n_layers: int,
    total_budget: int,
    min_rank: int,
    max_rank: int,
    frob_err: np.ndarray | None = None,
    act_variance: np.ndarray | None = None,
    sectional: np.ndarray | None = None,
) -> dict:
    """All four policies at the same budget; returns ranks + gini each."""
    if frob_err is None:
        frob_err = np.full(n_layers, 0.1)
    if act_variance is None:
        act_variance = np.full(n_layers, 1.0)
    if sectional is None:
        sectional = np.linspace(0.01, 1.0, n_layers)

    out = {}
    out["fixed"] = fixed_allocation(n_layers, total_budget, min_rank, max_rank)

    f = frank_build(frob_err)
    base = np.full(n_layers, total_budget // n_layers)
    out["frank"] = np.clip(np.round(base * f.rank_scale).astype(int),
                           min_rank, max_rank)
    out["frank_dominant_mode"] = f.dominant_mode

    m = mcr_detect_phases(act_variance)
    out["mcr"] = mcr_rank_budget(m, total_budget, min_rank, max_rank)
    out["mcr_phases_valid"] = m.phases_valid

    out["grcurv"], _ = grcurv_to_rank_budget(sectional, min_rank, max_rank,
                                             total_budget)

    summary = {
        name: {"gini": round(gini(r), 4),
               "min": int(r.min()), "max": int(r.max()),
               "sum": int(r.sum())}
        for name, r in out.items() if isinstance(r, np.ndarray)
    }
    summary["frank_dominant_mode"] = out.pop("frank_dominant_mode")
    summary["mcr_phases_valid"] = out.pop("mcr_phases_valid")
    return {"ranks": {k: v.tolist() for k, v in out.items()},
            "summary": summary}
