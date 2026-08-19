"""G10 — escalation ladder (compression failure => progressive recovery).

Composes the reconstruction-risk signals into a three-tier decision:

    routine   : coverage high AND entropy low
    elevated  : one signal degraded
    full      : coverage low OR entropy high (recover capacity)

Inputs are the cheap readouts the controller already has: the ONB
coverage of the draft residual (rho proxy), the APC output entropy, and
optionally a thermal flag. The contract, per the reviews: compression
failure escalates recovery — it never silently degrades output.
"""

from __future__ import annotations

from dataclasses import dataclass

COVERAGE_LOW = 0.7
ENTROPY_HIGH = 2.0  # bits


@dataclass
class EscalationPolicy:
    coverage_low: float = COVERAGE_LOW
    entropy_high: float = ENTROPY_HIGH
    thermal_hot: bool = False

    def decide(self, coverage: float, entropy: float) -> str:
        """Tier for the current token's signals."""
        cov_bad = coverage < self.coverage_low
        ent_bad = entropy > self.entropy_high
        if self.thermal_hot or cov_bad or ent_bad:
            if (self.thermal_hot and cov_bad) or ent_bad:
                return "full"
            return "elevated"
        return "routine"

    def recovery_action(self, tier: str) -> dict:
        """Concrete recovery step per tier (wired by the caller)."""
        if tier == "full":
            return {"precision": "full", "rank": "max", "experts": "load_cold"}
        if tier == "elevated":
            return {"precision": "current", "rank": "raise", "experts": "prefetch"}
        return {"precision": "fast", "rank": "current", "experts": "resident"}
