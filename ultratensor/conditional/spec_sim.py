"""Greedy speculative-decoding simulator (port of HyperTensor
hyperretro/bench/spec_decode_sim.py, Leviathan et al. 2023 §2 core).

The verifier scores prefix + all gamma draft tokens in ONE forward
pass; the longest accepted prefix is kept. The simulation layer is
model-agnostic: drafter_fn(prefix) -> gamma tokens, verifier_fn(tokens)
-> per-position argmax for the last gamma positions. Speedup is
computed against a verifier-only baseline.
"""

from __future__ import annotations

from dataclasses import dataclass, field


def accept_prefix(draft_tokens, verifier_argmax) -> tuple[list, int]:
    """Greedy acceptance: accept the longest prefix where draft == verifier.

    Returns (accepted_tokens, first_rejected_index).
    """
    accepted = []
    n = min(len(draft_tokens), len(verifier_argmax))
    for i in range(n):
        if draft_tokens[i] == verifier_argmax[i]:
            accepted.append(draft_tokens[i])
        else:
            break
    return accepted, len(accepted)


@dataclass
class SpecResult:
    tokens: list = field(default_factory=list)
    cycles: int = 0
    drafts_issued: int = 0
    drafts_accepted: int = 0
    verifier_calls: int = 0
    draft_cost: float = 0.0
    verify_cost: float = 0.0

    @property
    def mean_acceptance(self) -> float:
        if not self.cycles:
            return 0.0
        return self.drafts_accepted / self.cycles

    @property
    def total_cost(self) -> float:
        return self.draft_cost + self.verify_cost

    def speedup(self, verifier_only_cost: float) -> float:
        if self.total_cost <= 0.0:
            return 0.0
        return verifier_only_cost / self.total_cost


def simulate(
    verifier_fn,
    drafter_fn,
    prefix,
    gamma: int,
    max_cycles: int,
    max_tokens: int,
    draft_cost: float = 1.0,
    verify_cost: float = 10.0,
    seed: int = 0,
):
    """Simulate spec decoding. verifier_fn(token_list) -> list of argmax
    token ids for each of the LAST gamma positions of the input sequence
    (the prompt positions are ignored). drafter_fn(token_list) -> list of
    exactly gamma drafted tokens."""
    import random

    rng = random.Random(seed)
    tokens = list(prefix)
    r = SpecResult()
    while r.cycles < max_cycles and len(tokens) < max_tokens:
        drafts = list(drafter_fn(tokens)[:gamma])
        r.drafts_issued += len(drafts)
        r.draft_cost += draft_cost * len(drafts)
        if not drafts:
            break
        v_argmax = verifier_fn(tokens + drafts)
        accepted, n_acc = accept_prefix(drafts, v_argmax)
        tokens.extend(accepted)
        r.verify_cost += verify_cost
        r.verifier_calls += 1
        r.drafts_accepted += n_acc
        r.cycles += 1
        if len(tokens) >= max_tokens:
            tokens = tokens[:max_tokens]
            break
        if n_acc < len(drafts):   # rejected -> verifier's own token
            if n_acc < len(v_argmax):
                tokens.append(v_argmax[n_acc])
                if len(tokens) >= max_tokens:
                    tokens = tokens[:max_tokens]
                    break
    r.tokens = tokens
    return r


def expected_acceptance_geometric(q: float, gamma: int) -> float:
    """E[accepted drafts per cycle] for i.i.d. per-token acceptance q.

    Sum over i of P(first i accepted) = q(1-q^gamma)/(1-q) for q<1,
    gamma for q=1.
    """
    if gamma <= 0:
        return 0.0
    if abs(q - 1.0) < 1e-12:
        return float(gamma)
    return q * (1.0 - q ** gamma) / (1.0 - q)
