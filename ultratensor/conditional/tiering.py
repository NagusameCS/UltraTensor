"""G12 — tiered expert residency simulation.

The reviews' deployment stack: hot (always-resident), warm
(predicted/prefetched), cold (on-demand, slow). This module simulates
the tier policy against a route trace + prefetch plan and sweeps the hot
cap to find the latency-vs-resident-bytes knee.

Cost constants are parameters so the caller can plug in measured
numbers (ours: 18.9 ms/tensor expert decode, ~2-3 GB/s SSD reads,
~21 GiB/token active working set at Q3_K).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TierResult:
    hot_cap: int
    mean_miss_rate: float        # steps needing a cold load
    mean_latency_ms: float
    resident_experts: int
    resident_bytes: float        # bytes if per_expert_bytes given
    total_steps: int = 0
    step_misses: list = field(default_factory=list)  # missing experts/step

    @property
    def tail_p90_miss(self) -> float:
        """P90 of per-step missing-expert counts (tail-latency proxy)."""
        if not self.step_misses:
            return 0.0
        return float(np.percentile(self.step_misses, 90))


def simulate_tier(
    route_seq,
    prefetch_fn,
    hot_cap: int,
    miss_cost_ms: float = 500.0,     # cold SSD read + decode stall
    hit_cost_ms: float = 0.0,        # resident expert cost (amortised)
    prefetch_cost_ms: float = 20.0,  # warm-pool fetch per extra expert
    per_expert_bytes: float = 0.0,
    seed: int = 0,
) -> TierResult:
    """Simulate one tier policy over a route trace.

    prefetch_fn(current_set) -> ordered list of predicted experts for
    the next step. Everything in the prefetch list within hot_cap is
    resident; the rest are warm/cold misses charged miss_cost.
    """
    rng = np.random.default_rng(seed)
    sets = [frozenset(s) for s in route_seq]
    misses, lat_ms, res = 0, 0.0, set()
    step_misses = []
    for t in range(1, len(sets)):
        needed = sets[t]
        plan = list(prefetch_fn(sets[t - 1]))[:hot_cap]
        hit = {e for e in needed if e in plan}
        missing = len(needed) - len(hit)
        step_misses.append(missing)
        misses += 1 if missing else 0
        lat_ms += hit_cost_ms * len(hit) + miss_cost_ms * missing \
            + prefetch_cost_ms * max(0, len(plan) - len(hit))
        res |= set(plan)
    n = max(1, len(sets) - 1)
    return TierResult(
        hot_cap=hot_cap,
        mean_miss_rate=misses / n,
        mean_latency_ms=lat_ms / n,
        resident_experts=len(res),
        resident_bytes=len(res) * per_expert_bytes,
        total_steps=len(sets) - 1,
        step_misses=step_misses,
    )


def tier_sweep(
    route_seq,
    prefetch_fn,
    hot_caps=(4, 6, 8, 12, 16, 24, 32, 48, 64),
    **kw,
) -> list[TierResult]:
    """Sweep hot_cap and return the full curve (pick the knee by eye or
    by the largest latency drop per resident expert)."""
    return [simulate_tier(route_seq, prefetch_fn, c, **kw) for c in hot_caps]


def knee(curve: list[TierResult]) -> TierResult | None:
    """Largest latency improvement per additional resident expert;
    ties break towards lower latency."""
    if len(curve) < 2:
        return curve[0] if curve else None
    best, best_gain = curve[0], -np.inf
    for a, b in zip(curve[:-1], curve[1:]):
        gain = (a.mean_latency_ms - b.mean_latency_ms) / max(
            b.resident_experts - a.resident_experts, 1)
        if gain >= best_gain:
            best_gain, best = gain, b
    return best
