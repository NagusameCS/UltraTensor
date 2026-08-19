"""MCR phase detection + attention-sink protection (mcr_compress.c port).

Two independent pieces:

1. MCR: the per-layer activation-variance profile splits naturally into
   Mix / Compress / Refine phases; rank budgets are allocated per phase
   instead of uniformly. A flat profile (ratio < 1.15) invalidates the
   phases and recommends uniform treatment — exactly what our V4 expert
   spectra showed.

2. Sinks: attention-sink positions are outliers in activation norm
   (>= mean + sigma_threshold * std, with a 1.5-sigma special case for
   position 0). Before compressing context, check the sink direction is
   covered by the compressed basis; if not, the normalised mean sink
   hidden state is returned as an ``extra_dir`` to graft in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class McrPhase(str, Enum):
    MIX = "mix"
    COMPRESS = "compress"
    REFINE = "refine"


@dataclass
class MCRResult:
    n_layers: int = 0
    phases_valid: bool = False
    phases: list[str] = field(default_factory=list)
    smoothed_var: np.ndarray | None = None
    var_min: float = 0.0
    var_max: float = 0.0
    mix_end: int = -1
    compress_start: int = 0
    compress_end: int = -1
    refine_start: int = 0

    @property
    def var_ratio(self) -> float:
        return (self.var_max / self.var_min) if self.var_min > 1e-15 else 1.0


def mcr_detect_phases(
    act_variance,
    compress_thr: float = 1.0,
) -> MCRResult:
    """Detect Mix/Compress/Refine zones from per-layer activation variance."""
    var = np.asarray(act_variance, dtype=np.float64)
    n = var.size
    if n < 2:
        return MCRResult(n_layers=n)
    if compress_thr < 1.0:
        compress_thr = 1.0

    # 3-tap moving average
    prev = np.concatenate((var[:1], var[:-1]))
    nxt = np.concatenate((var[1:], var[-1:]))
    smoothed = (prev + var + nxt) / 3.0

    r = MCRResult(
        n_layers=n,
        smoothed_var=smoothed,
        var_min=float(smoothed.min()),
        var_max=float(smoothed.max()),
    )

    if r.var_ratio < 1.15:
        r.phases = [McrPhase.MIX.value] * n
        r.mix_end = n - 1
        r.compress_start = n
        r.compress_end = n - 1
        r.refine_start = n
        r.phases_valid = False
        return r

    r.phases_valid = True
    l_min = int(np.argmin(smoothed))
    threshold = r.var_min * compress_thr

    cs = ce = l_min
    while cs > 0 and smoothed[cs - 1] <= threshold:
        cs -= 1
    while ce < n - 1 and smoothed[ce + 1] <= threshold:
        ce += 1

    r.compress_start, r.compress_end = cs, ce
    r.mix_end = cs - 1
    r.refine_start = ce + 1
    r.phases = [
        McrPhase.MIX.value if l < cs else
        McrPhase.COMPRESS.value if l <= ce else
        McrPhase.REFINE.value
        for l in range(n)
    ]
    return r


def mcr_rank_budget(
    result: MCRResult,
    total_budget: int,
    min_rank: int,
    max_rank: int,
    mix_scale: float = 1.0,
    compress_scale: float = 0.7,
    refine_scale: float = 1.0,
) -> np.ndarray:
    """Per-layer rank budget from phases, normalised to total_budget."""
    n = result.n_layers
    if n < 1:
        return np.zeros(0, dtype=int)
    if not result.phases_valid or total_budget <= 0:
        base = max(min_rank, min(max_rank, total_budget // n if total_budget else min_rank))
        return np.full(n, base, dtype=int)

    base = total_budget / n
    scales = {
        McrPhase.MIX.value: mix_scale,
        McrPhase.COMPRESS.value: compress_scale,
        McrPhase.REFINE.value: refine_scale,
    }
    raw = np.array([base * scales[p] for p in result.phases])
    norm = total_budget / raw.sum() if raw.sum() > 0.0 else 1.0
    return np.clip(np.round(raw * norm).astype(int), min_rank, max_rank)


@dataclass
class SinkResult:
    valid: bool = False
    indices: np.ndarray | None = None
    norms: np.ndarray | None = None
    norm_mean: float = 0.0
    norm_std: float = 0.0
    sigma_threshold: float = 3.0


def sink_detect(norms, sigma_threshold: float = 3.0) -> SinkResult:
    """Flag attention-sink positions (sink_detect logic)."""
    nv = np.asarray(norms, dtype=np.float64)
    n = nv.size
    if n < 2:
        return SinkResult()
    mean = float(nv.mean())
    var = float((nv * nv).mean() - mean * mean)
    std = np.sqrt(var) if var > 0.0 else 0.0
    hi_thr = mean + sigma_threshold * std
    lo_thr = mean + 1.5 * std

    hit = nv >= hi_thr
    if nv[0] >= lo_thr:
        hit[0] = True
    idx = np.nonzero(hit)[0]
    return SinkResult(
        valid=True,
        indices=idx,
        norms=nv[idx],
        norm_mean=mean,
        norm_std=std,
        sigma_threshold=sigma_threshold,
    )


def sink_check_basis_coverage(
    sinks: SinkResult,
    mean_sink_hs,
    basis: np.ndarray,
    cos_threshold: float = 0.9,
):
    """(covered: bool, extra_dir or None) — sink_check_basis_coverage."""
    if not sinks.valid or sinks.indices is None or len(sinks.indices) == 0:
        return True, None
    d = np.asarray(mean_sink_hs, dtype=np.float64)
    norm_sq = float(d @ d)
    if norm_sq < 1e-20:
        return True, None
    d_hat = d / np.sqrt(norm_sq)
    basis = np.asarray(basis, dtype=np.float64)
    max_cos = float(np.max(np.abs(basis @ d_hat))) if basis.size else 0.0
    if max_cos >= cos_threshold:
        return True, None
    return False, d_hat
