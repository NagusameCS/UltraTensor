"""Tests for spec-decode simulation and the prefetch controller."""

import random

import numpy as np
import pytest

from ultratensor.conditional.lookahead import PrefetchController
from ultratensor.conditional.spec_sim import (
    SpecResult,
    accept_prefix,
    expected_acceptance_geometric,
    simulate,
)


# ---------------------------------------------------------------------------
# spec-decode simulation
# ---------------------------------------------------------------------------

def test_accept_prefix():
    drafts = [1, 2, 3, 4]
    vm = [1, 2, 9, 4]
    acc, n = accept_prefix(drafts, vm)
    assert acc == [1, 2] and n == 2
    assert accept_prefix([5], [5]) == ([5], 1)
    assert accept_prefix([5], [6]) == ([], 0)


def test_geometric_acceptance_formula():
    assert expected_acceptance_geometric(1.0, 4) == 4.0
    # q=0.5, gamma=2: 0.5 + 0.25 = 0.75
    assert expected_acceptance_geometric(0.5, 2) == pytest.approx(0.75)
    assert expected_acceptance_geometric(0.0, 4) == 0.0


def _make_toy(q=0.6, vocab=8):
    """Deterministic drafter/verifier pair with i.i.d. acceptance q."""
    rng = random.Random(3)
    gamma = 4

    def drafter(tokens):
        nxt = rng.randrange(vocab)
        return [nxt] * gamma

    def verifier(tokens):
        # returns argmax for the LAST gamma positions only (the drafts)
        argmax = []
        for t in tokens[-gamma:]:
            argmax.append(t if rng.random() < q else (t + 1) % vocab)
        return argmax

    return drafter, verifier


def test_simulate_mean_acceptance_matches_theory():
    drafter, verifier = _make_toy(q=0.6)
    r = simulate(verifier, drafter, [0], gamma=4, max_cycles=3000,
                 max_tokens=10 ** 9, draft_cost=1.0, verify_cost=10.0)
    assert r.mean_acceptance == pytest.approx(
        expected_acceptance_geometric(0.6, 4), abs=0.05)
    assert r.speedup(10.0 * (len(r.tokens) - 1)) > 0.0


def test_simulate_respects_max_tokens():
    drafter, verifier = _make_toy(q=0.9)
    r = simulate(verifier, drafter, [0], gamma=4, max_cycles=100,
                 max_tokens=12)
    assert len(r.tokens) == 12


# ---------------------------------------------------------------------------
# Prefetch controller
# ---------------------------------------------------------------------------

def _table(tok):
    # deterministic: experts (tok % 16 + 3*i) % 64, i in 0..5
    return [(tok % 16 + 3 * i) % 64 for i in range(6)]


def test_prefetch_perfect_drafts_full_hit():
    ctl = PrefetchController(table_fn=_table, top_k=6, H=4, size_cap=24)
    drafts = [10, 11, 12, 13]
    plan = ctl.plan(drafts)
    needed = {e for t in drafts for e in _table(t)}
    assert needed <= set(plan)
    assert len(plan) == len(needed)       # deduped union, no overfetch
    score = ctl.observe(drafts, drafts)   # perfect draft -> full hit
    assert score == 1.0 and ctl.hit_rate == 1.0


def test_prefetch_cap_degrades_hit():
    ctl = PrefetchController(table_fn=_table, top_k=6, H=4, size_cap=8)
    drafts = [10, 11, 12, 13]
    plan = ctl.plan(drafts)
    assert len(plan) == 8
    needed = {e for t in drafts for e in _table(t)}
    assert len(needed) > 8
    assert not needed <= set(plan)
    score = ctl.observe(drafts, drafts)
    assert 0.0 < score < 1.0


def test_prefetch_noisy_drafts_penalize():
    good = PrefetchController(table_fn=_table, H=4, size_cap=24)
    bad = PrefetchController(table_fn=_table, H=4, size_cap=24)
    actual = [10, 11, 12, 13]
    for _ in range(20):
        good.observe(actual, actual)
        bad.observe([x + 1 for x in actual], actual)
    assert good.hit_rate > bad.hit_rate
