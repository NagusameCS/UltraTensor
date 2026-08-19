"""Tests for the G10 escalation ladder."""

from ultratensor.conditional.escalation import EscalationPolicy


def test_routine_when_all_signals_clean():
    p = EscalationPolicy()
    assert p.decide(coverage=0.95, entropy=0.5) == "routine"


def test_elevated_on_low_coverage_only():
    p = EscalationPolicy()
    assert p.decide(coverage=0.5, entropy=0.5) == "elevated"


def test_full_on_high_entropy():
    p = EscalationPolicy()
    assert p.decide(coverage=0.95, entropy=3.0) == "full"


def test_full_on_hot_plus_bad_coverage():
    p = EscalationPolicy(thermal_hot=True)
    assert p.decide(coverage=0.5, entropy=0.5) == "full"
    assert p.decide(coverage=0.95, entropy=0.5) == "elevated"


def test_recovery_actions_are_progressive():
    p = EscalationPolicy()
    r = p.recovery_action("routine")
    e = p.recovery_action("elevated")
    f = p.recovery_action("full")
    assert r["precision"] == "fast"
    assert e["rank"] == "raise"
    assert f["experts"] == "load_cold"
