"""qspec shared-basis feasibility + frank rank profiles (qspec_basis.c port).

qspec answers "should layer X share a compressed basis with layer Y?"
with one bounded number per (layer, slot): alignment = min/max ratio of
the manifold projection energy fraction and the SVD explained-energy
fraction. 1.0 = identical energy capture; 0.0 = one path captures
nothing. This is the cheap pre-check for G6 shared dictionaries — run it
before spending any GPU time on a shared basis.

frank converts per-layer reconstruction errors into layer-position rank
profiles: identify the dominant failure mode (factual/early,
reasoning/mid, coherence/late, context/uniform) and boost rank in that
zone with geometric decay into the neighbouring zones.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Slot numbering convention (from axiom_exploit.h):
#   0 = FFN_down, 1 = Q, 2 = K, 3 = V, 4 = O, 5 = FFN_up, 6 = FFN_gate
QSPEC_MAX_SLOTS = 8


@dataclass
class QspecEntry:
    layer: int
    slot: int
    proj_energy: float
    svd_explained: float
    alignment: float
    shared_ok: bool


@dataclass
class QspecResult:
    entries: list[QspecEntry] = field(default_factory=list)
    share_threshold: float = 0.8
    mean_alignment: float = 0.0
    min_alignment: float = 0.0
    n_shared_ok: int = 0

    def worst(self) -> QspecEntry | None:
        if not self.entries:
            return None
        return min(self.entries, key=lambda e: e.alignment)

    def verdict(self) -> str:
        if not self.entries:
            return "no data"
        frac = self.n_shared_ok / len(self.entries)
        if frac >= 0.8:
            return "basis likely transferable across quant levels"
        return "basis may need recomputation per quant level"


def _alignment(proj_energy: float, frobenius_err: float) -> tuple[float, float]:
    """min/max ratio of proj_energy and svd_explained, bounded [0, 1]."""
    frob = float(np.clip(frobenius_err, 0.0, 1.0))
    svd_expl = 1.0 - frob * frob
    if svd_expl < 1e-6:
        svd_expl = 1e-6
    lo, hi = (proj_energy, svd_expl) if proj_energy < svd_expl else (svd_expl, proj_energy)
    align = (lo / hi) if hi > 1e-6 else 0.0
    return float(align), float(svd_expl)


def qspec_test_shared_basis(
    rows,
    share_threshold: float = 0.8,
) -> QspecResult:
    """Evaluate shared-basis feasibility for a set of (layer, slot) rows.

    ``rows``: iterable of (layer, slot, proj_energy, frobenius_err)
    tuples or objects with those attributes. Slots are taken as given
    (the C skips slot 0; pass pre-filtered rows if you want that).
    """
    result = QspecResult(share_threshold=share_threshold)
    align_sum = 0.0
    for row in rows:
        try:
            layer, slot, pe, frob = (
                row.layer,
                row.slot,
                row.proj_energy,
                row.frobenius_err,
            )
        except AttributeError:
            layer, slot, pe, frob = row
        align, svd_expl = _alignment(float(pe), float(frob))
        entry = QspecEntry(
            layer=int(layer),
            slot=int(slot),
            proj_energy=float(pe),
            svd_explained=svd_expl,
            alignment=align,
            shared_ok=align >= share_threshold,
        )
        result.entries.append(entry)
        align_sum += align
        if entry.shared_ok:
            result.n_shared_ok += 1
    if result.entries:
        result.mean_alignment = align_sum / len(result.entries)
        result.min_alignment = min(e.alignment for e in result.entries)
    return result


@dataclass
class FrankResult:
    n_layers: int = 0
    early_err: float = 0.0
    mid_err: float = 0.0
    late_err: float = 0.0
    global_err: float = 0.0
    dominant_mode: str = "none"  # factual | reasoning | coherence | context
    rank_scale: np.ndarray | None = None
    valid: bool = False


def frank_build(
    frob_err,
    dominant_boost: float = 1.8,
    decay: float = 0.6,
) -> FrankResult:
    """frank_build: zone-averaged errors -> dominant mode + rank scales."""
    err = np.asarray(frob_err, dtype=np.float64)
    err = np.clip(err, 0.0, None)
    n = err.size
    if n < 2:
        r = FrankResult(n_layers=n, rank_scale=np.ones(n) if n else None, valid=False)
        return r
    if dominant_boost < 1.0:
        dominant_boost = 1.8
    if not 0.0 <= decay <= 1.0:
        decay = 0.6

    b1, b2 = n // 3, 2 * n // 3
    r = FrankResult(
        n_layers=n,
        early_err=float(err[:b1].mean()),
        mid_err=float(err[b1:b2].mean()),
        late_err=float(err[b2:].mean()),
        global_err=float(err.mean()),
    )

    tol = r.global_err * 0.20
    uniform = (
        abs(r.early_err - r.global_err) < tol
        and abs(r.mid_err - r.global_err) < tol
        and abs(r.late_err - r.global_err) < tol
    )
    if uniform and r.global_err > 0.05:
        r.dominant_mode = "context"
        dom_zone = -1
    elif r.early_err >= r.mid_err and r.early_err >= r.late_err:
        r.dominant_mode = "factual"
        dom_zone = 0
    elif r.mid_err >= r.early_err and r.mid_err >= r.late_err:
        r.dominant_mode = "reasoning"
        dom_zone = 1
    else:
        r.dominant_mode = "coherence"
        dom_zone = 2

    scale = np.ones(n, dtype=np.float64)
    for l in range(n):
        if r.dominant_mode == "context":
            scale[l] = dominant_boost
            continue
        zone = 0 if l < b1 else 1 if l < b2 else 2
        if zone == dom_zone:
            scale[l] = dominant_boost
        else:
            s = dominant_boost
            for _ in range(abs(zone - dom_zone)):
                s = 1.0 + (s - 1.0) * decay
            scale[l] = s if s >= 1.0 else 1.0
    r.rank_scale = scale
    r.valid = True
    return r


def frank_apply(
    result: FrankResult,
    ranks,
    min_rank: int = 8,
    max_rank: int = 256,
) -> np.ndarray:
    """frank_apply: scale base ranks per layer and clamp (returns new array)."""
    ranks = np.asarray(ranks, dtype=np.float64)
    if not result.valid or result.rank_scale is None:
        return np.clip(ranks, min_rank, max_rank).astype(int)
    n = min(ranks.size, result.n_layers)
    scaled = ranks[:n] * result.rank_scale[:n]
    return np.clip(np.round(scaled).astype(int), min_rank, max_rank)
