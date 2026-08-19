"""Tests for G5 rank-policy comparison."""

import numpy as np
import pytest

from ultratensor.conditional.rank_policies import (
    compare_policies,
    fixed_allocation,
    gini,
)


def test_gini_extremes():
    assert gini(np.ones(8)) == pytest.approx(0.0, abs=1e-9)
    x = np.zeros(8)
    x[0] = 1.0
    assert gini(x) == pytest.approx(1.0 - 1 / 8)


def test_fixed_allocation_respects_budget():
    r = fixed_allocation(9, 900, 8, 256)
    assert np.all(r == 100)


def test_compare_policies_flat_signal_favors_nothing():
    out = compare_policies(9, 900, 8, 256,
                           frob_err=np.full(9, 0.1),
                           act_variance=np.full(9, 1.0))
    s = out["summary"]
    assert s["frank_dominant_mode"] == "context"
    assert s["mcr_phases_valid"] is False
    # uniform signal -> every policy concentrates weakly
    assert s["fixed"]["gini"] == pytest.approx(0.0, abs=1e-9)


def test_compare_policies_signal_concentrates():
    err = np.array([0.3, 0.3, 0.3, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05])
    out = compare_policies(9, 900, 8, 256, frob_err=err,
                           act_variance=np.full(9, 1.0))
    assert out["summary"]["frank_dominant_mode"] == "factual"
    ranks = out["ranks"]["frank"]
    assert ranks[0] > ranks[8]
