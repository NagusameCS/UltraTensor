"""Unified conditional policy — composes the ported mechanisms.

This is the "vision" layer: instead of using APC / thermal / frank /
MCR in isolation, ``ConditionalPolicy`` produces one per-token and
per-layer decision tuple:

- per-layer rank profile   (frank failure-mode scales x MCR phase scales)
- global thermal clamp     (hardware budget scales the whole profile)
- per-token precision tier (APC entropy gate: fast quantized vs escalate)
- energy nudge             (TPJ gradient available for softmax policies)

Everything stays cheap and deterministic: the inputs are per-layer
frobenius errors, per-layer activation variances, the thermal sensor,
and the current output logits.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .basis import frank_build
from .precision import apc_gate
from .sinks import McrPhase, mcr_detect_phases
from .thermal import ThermalRank, TpjTracker

_MCR_SCALE = {McrPhase.MIX.value: 1.0, McrPhase.COMPRESS.value: 0.7,
              McrPhase.REFINE.value: 1.0}


@dataclass
class PolicyState:
    ranks: np.ndarray                      # per-layer rank
    precision: str = "fast"                # fast | escalate
    entropy: float = 0.0
    thermal_rank: int = 0
    dominant_mode: str = "none"


@dataclass
class ConditionalPolicy:
    thermal: ThermalRank | None = None
    tpj: TpjTracker | None = None
    apc_threshold: float = 0.5
    min_rank: int = 8
    max_rank: int = 256
    frank_boost: float = 1.8
    frank_decay: float = 0.6

    def rank_profile(
        self,
        frob_err,
        act_variance,
        base_rank: int = 128,
    ) -> tuple[np.ndarray, str]:
        """Per-layer rank profile: frank scales x MCR phase scales.

        Returns (ranks, dominant_failure_mode).
        """
        frank = frank_build(frob_err, self.frank_boost, self.frank_decay)
        mcr = mcr_detect_phases(act_variance)
        err = np.asarray(frob_err, dtype=np.float64)
        n = err.size
        base = np.full(n, float(base_rank))
        if frank.valid and frank.rank_scale is not None:
            base = base * frank.rank_scale[:n]
        if mcr.phases_valid:
            phase_scale = np.array([_MCR_SCALE[p] for p in mcr.phases[:n]])
            base = base * phase_scale
        # Keep the total budget constant (scales are relative allocations)
        if base.sum() > 0:
            base = base * (base_rank * n / base.sum())
        ranks = np.clip(np.round(base).astype(int), self.min_rank, self.max_rank)
        return ranks, frank.dominant_mode

    def thermal_clamp(self, ranks: np.ndarray) -> np.ndarray:
        """Global thermal scale; no-op when no sensor."""
        if self.thermal is None:
            return ranks
        top = int(ranks.max()) if ranks.size else self.max_rank
        tr = self.thermal.get_rank(top)
        if tr >= top:
            return ranks
        scale = tr / top
        return np.clip(np.round(ranks * scale).astype(int),
                       self.min_rank, self.max_rank)

    def precision_decision(self, logits) -> tuple[str, float]:
        """APC gate on the current output distribution."""
        fast, entropy = apc_gate(logits, threshold=self.apc_threshold)
        return ("fast" if fast else "escalate"), entropy

    def step(self, logits, frob_err, act_variance, base_rank: int = 128):
        """One decision tuple for the current token."""
        ranks, mode = self.rank_profile(frob_err, act_variance, base_rank)
        ranks = self.thermal_clamp(ranks)
        precision, entropy = self.precision_decision(logits)
        return PolicyState(
            ranks=ranks,
            precision=precision,
            entropy=entropy,
            thermal_rank=int(self.thermal.get_rank(self.max_rank))
            if self.thermal else 0,
            dominant_mode=mode,
        )
