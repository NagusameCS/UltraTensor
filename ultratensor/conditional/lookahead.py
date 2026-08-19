"""G3 — lookahead expert working-set prediction (the review's E_{t:t+H}).

A deployed MoE should prefetch the union of expert sets for the next H
tokens, not just the current top-k. This module turns a route trace
(a sequence of per-step expert sets) into:

- ``WorkingSetModel``  : predicts Pr(e used in next H steps) from
  frequency or from a Markov bigram over successor sets (with a
  frequency fallback for unseen sets);
- ``PrefetchCurve``    : sweeps the prefetch threshold tau and reports
  hit rate (|P ∩ E| / |E|) vs resident-set size, so tau can be chosen
  for tail latency rather than raw hit rate.

The oracle (union of the true future sets) is included as the ceiling.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def working_set_union(route_seq, t: int, H: int) -> frozenset:
    """Union of expert sets for steps t..t+H-1 (the review's E_{t:t+H})."""
    u: set = set()
    for tau in range(t, min(t + H, len(route_seq))):
        u |= set(route_seq[tau])
    return frozenset(u)


def _to_sets(route_seq):
    return [frozenset(s) for s in route_seq]


@dataclass
class WorkingSetModel:
    """Predicts per-expert usage probability over the next H steps."""

    usage: dict[int, float] = field(default_factory=dict)
    successor_counts: dict[frozenset, dict[int, int]] = field(
        default_factory=dict
    )
    alpha: float = 2.0  # pseudo-count controlling frequency fallback blend
    n_experts: int = 0

    def fit(self, route_seq) -> "WorkingSetModel":
        sets = _to_sets(route_seq)
        n_steps = len(sets)
        total_slots = sum(len(s) for s in sets)
        usage: dict[int, float] = {}
        succ: dict[frozenset, dict[int, int]] = {}
        for t, s in enumerate(sets):
            for e in s:
                usage[e] = usage.get(e, 0) + 1.0
            if t + 1 < n_steps:
                nxt = succ.setdefault(s, {})
                for e in sets[t + 1]:
                    nxt[e] = nxt.get(e, 0) + 1
        if total_slots > 0:
            usage = {e: c / n_steps for e, c in usage.items()}
        self.usage = usage
        self.successor_counts = succ
        self.n_experts = max(usage, default=-1) + 1 if usage else 0
        return self

    def predict_usage(self, current_set, H: int) -> dict[int, float]:
        """Pr(e used in next H steps | current set).

        Frequency model: 1 - (1 - u_e)^H.
        Markov model: empirical successor frequency, blended with the
        frequency model by successor-sample strength.
        """
        current_set = frozenset(current_set)
        succ = self.successor_counts.get(current_set, {})
        n_succ = sum(succ.values())
        out: dict[int, float] = {}
        for e, u in self.usage.items():
            p1 = u
            if n_succ > 0 and e in succ:
                p_emp = succ[e] / n_succ
                w = n_succ / (n_succ + self.alpha)
                p1 = w * p_emp + (1.0 - w) * u
            out[e] = 1.0 - (1.0 - p1) ** H
        return out


@dataclass
class PrefetchCurve:
    """tau sweep result for one (model, H): hit rate vs set size."""

    H: int
    taus: np.ndarray
    hits: np.ndarray      # mean |P ∩ E| / |E|
    sizes: np.ndarray     # mean |P|
    best_tau: float = 0.0
    best_hit: float = 0.0
    best_size: float = 0.0

    @property
    def mean_target_size(self) -> float:
        return 0.0


def evaluate_prefetch(
    route_seq,
    model,
    H: int,
    taus=(0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
    objective_lambda: float = 0.1,
) -> PrefetchCurve:
    """Sweep tau over lookback steps and pick the best by
    hit - lambda * size (tail-latency-aware, not raw hit rate)."""
    sets = _to_sets(route_seq)
    n = len(sets)
    rows = []
    for t in range(1, n):  # need a current set to condition on
        target = working_set_union(sets, t, H)
        if not target:
            continue
        probs = model.predict_usage(sets[t - 1], H)
        rows.append((target, probs))
    if not rows:
        return PrefetchCurve(H=H, taus=np.asarray([]), hits=np.asarray([]),
                             sizes=np.asarray([]))

    hits, sizes = [], []
    for tau in taus:
        hs, ss = [], []
        for target, probs in rows:
            P = {e for e, p in probs.items() if p > tau}
            hs.append(len(P & target) / len(target))
            ss.append(len(P))
        hits.append(float(np.mean(hs)))
        sizes.append(float(np.mean(ss)))
    hits_a = np.asarray(hits)
    sizes_a = np.asarray(sizes)
    taus_a = np.asarray(taus, dtype=float)
    obj = hits_a - objective_lambda * sizes_a / max(1, model.n_experts)
    i = int(np.argmax(obj))
    return PrefetchCurve(
        H=H, taus=taus_a, hits=hits_a, sizes=sizes_a,
        best_tau=float(taus_a[i]), best_hit=float(hits_a[i]),
        best_size=float(sizes_a[i]),
    )


def oracle_curve(route_seq, H: int) -> dict:
    """The ceiling: prefetching the exact future union."""
    sets = _to_sets(route_seq)
    sizes, n_distinct = [], set()
    for t in range(1, len(sets)):
        u = working_set_union(sets, t, H)
        if u:
            sizes.append(len(u))
            n_distinct |= u
    return {
        "mean_union_size": float(np.mean(sizes)) if sizes else 0.0,
        "total_distinct_experts": len(n_distinct),
    }


@dataclass
class PrefetchController:
    """Draft-driven working-set prefetch for hash-routed layers.

    G3 wiring result: the token drafter IS the lookahead predictor for
    hash layers. Given the next H draft tokens and the deterministic
    token->expert table, this controller emits an ordered prefetch list
    (earliest-needed experts first) capped at ``size_cap``, and tracks
    hit rate against the actual tokens as they arrive.

    ``table_fn(tok)``: returns the expert ids the hash router selects
    for that token (top_k of them).
    """

    table_fn: object
    top_k: int = 6
    H: int = 4
    size_cap: int = 48
    requests: int = 0
    hits: int = 0
    _total_prefetch: int = 0

    def _set_of(self, tok: int) -> list:
        return [int(e) for e in self.table_fn(tok)][: self.top_k]

    def plan(self, draft_tokens) -> list:
        """Ordered expert list for the next H draft tokens, capped."""
        ordered: list = []
        for tok in draft_tokens[: self.H]:
            for e in self._set_of(tok):
                if e not in ordered:
                    ordered.append(e)
                    if len(ordered) >= self.size_cap:
                        return ordered
        return ordered

    def observe(self, draft_tokens, actual_tokens) -> float:
        """Score one step: were all actually-needed experts prefetched?"""
        prefetched = set(self.plan(draft_tokens))
        needed = {e for t in actual_tokens[: self.H] for e in self._set_of(t)}
        self.requests += 1
        self._total_prefetch += len(prefetched)
        if needed <= prefetched:
            self.hits += 1
            return 1.0
        return len(prefetched & needed) / max(1, len(needed))

    @property
    def hit_rate(self) -> float:
        return self.hits / self.requests if self.requests else 0.0

    @property
    def mean_prefetch_size(self) -> float:
        return self._total_prefetch / self.requests if self.requests else 0.0
