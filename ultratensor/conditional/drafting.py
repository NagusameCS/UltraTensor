"""Heterogeneous drafter rank allocation (port of
HyperTensor scripts/heterogeneous_drafters.py semantics).

In speculative decoding with gamma draft slots, early slots accept more
often than late slots (each slot must survive all previous
verifications). Spending rank uniformly is wasteful: early slots deserve
a more accurate (higher-rank) drafter, late slots tolerate aggressive
compression. Expected throughput for slot ranks k_1..k_gamma:

    T(k_1:k_gamma) = E[accepted] / (sum_i t_D(k_i) + t_V)

with E[accepted] = 1 + a1 + a1*a2 + ... + prod(a_1..a_gamma).

This module simulates and greedily allocates a total rank budget across
slots to maximise T.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Default drafter-time model: per-token drafter cost scales with rank
# (a compressed-attention drafter costs roughly linearly in k for the
# projections it keeps).
DEFAULT_TIME_FN = lambda rank: 0.001 + 1e-6 * rank  # noqa: E731


def expected_acceptance(acceptances) -> float:
    """E[accepted] for gamma slots with per-slot acceptance a_i."""
    a = np.asarray(acceptances, dtype=np.float64)
    e = 1.0
    prod = 1.0
    for ai in a:
        prod *= ai
        e += prod
    return float(e)


@dataclass
class DraftPlan:
    slots: list[int] = field(default_factory=list)      # rank per slot
    acceptances: list[float] = field(default_factory=list)
    throughput: float = 0.0

    @property
    def gamma(self) -> int:
        return len(self.slots)

    @property
    def total_rank(self) -> int:
        return int(sum(self.slots))


def optimize_slots(
    gamma: int,
    rank_budget: int,
    acceptance_fn,
    time_fn=DEFAULT_TIME_FN,
    verifier_time: float = 0.05,
    min_rank: int = 8,
    step: int = 8,
    max_steps: int = 10000,
) -> DraftPlan:
    """Greedy rank allocation maximising expected throughput.

    acceptance_fn(rank, slot_index) -> acceptance in (0, 1).
    Starts every slot at min_rank and repeatedly adds ``step`` rank to
    the slot with the largest throughput gain, until the budget is spent
    (or no gain remains).
    """
    slots = [min_rank] * gamma
    used = min_rank * gamma

    def throughput(ranks):
        acc = [acceptance_fn(r, i) for i, r in enumerate(ranks)]
        t = sum(time_fn(r) for r in ranks) + verifier_time
        return expected_acceptance(acc) / t, acc

    t0, acc = throughput(slots)
    for _ in range(max_steps):
        if used + step > rank_budget:
            break
        best_i, best_t = -1, t0
        for i in range(gamma):
            trial = slots.copy()
            trial[i] += step
            t, _ = throughput(trial)
            if t > best_t:
                best_i, best_t = i, t
        if best_i < 0:
            break
        slots[best_i] += step
        used += step
        t0, acc = throughput(slots)

    return DraftPlan(slots=slots, acceptances=acc, throughput=t0)
