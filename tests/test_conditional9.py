"""Tests for the composed ServeController."""

import numpy as np
import pytest

from ultratensor.conditional import ConditionalPolicy, PrefetchController
from ultratensor.conditional.controller import ServeController


class _ConstSensor:
    def read(self):
        return (40.0, 30.0)


def _table(tok):
    return [(tok % 16 + 3 * i) % 64 for i in range(6)]


def test_serve_controller_composes_all_signals():
    ctl = PrefetchController(table_fn=_table, H=2, size_cap=24)
    pol = ConditionalPolicy()
    sc = ServeController(prefetch=ctl, policy=pol)

    err = np.full(9, 0.1)
    var = np.full(9, 1.0)
    drafts = [10, 11]
    actual = [10, 12]
    d = sc.step(drafts, actual,
                logits=np.array([8.0, -1.0, -1.0, -1.0]),
                frob_err=err, act_variance=var)
    assert d.prefetch_size > 0
    assert 0.0 <= d.prefetch_hit <= 1.0
    assert d.ranks is not None and len(d.ranks) == 9
    assert d.precision == "fast"
    assert d.entropy < 0.5


def test_serve_controller_escalates_on_flat_logits():
    ctl = PrefetchController(table_fn=_table)
    sc = ServeController(prefetch=ctl, policy=ConditionalPolicy())
    d = sc.step([0], [0], logits=np.ones(4),
                frob_err=np.full(9, 0.1), act_variance=np.full(9, 1.0))
    assert d.precision == "escalate"


def test_serve_controller_thermal_clamps_ranks():
    ctl = PrefetchController(table_fn=_table)
    pol = ConditionalPolicy(thermal=None)  # no sensor -> no clamp
    sc = ServeController(prefetch=ctl, policy=pol)
    d = sc.step([0], [0], logits=np.array([1.0, 0.0]),
                frob_err=np.full(9, 0.1), act_variance=np.full(9, 1.0))
    assert (d.ranks >= 8).all() and (d.ranks <= 256).all()


def test_serve_controller_rho_escalation():
    """rho predictor + escalation ladder: low coverage -> elevated/full."""
    ctl = PrefetchController(table_fn=_table)
    pol = ConditionalPolicy(thermal=None)
    sc = ServeController(prefetch=ctl, policy=pol,
                         rho=lambda h: float(h.mean() / 100.0))
    d = sc.step([0], [0], logits=np.array([1.0, 0.0]),
                frob_err=np.full(9, 0.1), act_variance=np.full(9, 1.0),
                hidden=np.full(7168, 5.0))       # coverage 0.05 -> bad
    assert d.coverage == pytest.approx(0.05)
    assert d.tier in ("elevated", "full")
    # high coverage -> routine
    d2 = sc.step([0], [0], logits=np.array([1.0, 0.0]),
                 frob_err=np.full(9, 0.1), act_variance=np.full(9, 1.0),
                 hidden=np.full(7168, 90.0))     # coverage 0.9 -> good
    assert d2.coverage == pytest.approx(0.9)
    assert d2.tier == "routine"
