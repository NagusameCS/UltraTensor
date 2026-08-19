"""G9 — ServeController: the composed deployment loop.

One headless step of the "tiny controller" vision, wiring the ported
mechanisms together:

    draft tokens -> prefetch plan (hash layers: exact via token->expert
    table) -> tier-residency accounting -> per-layer rank profile
    (frank x MCR) -> thermal clamp -> APC precision gate.

The controller stays model-agnostic: every input is either a cheap
signal (logits, per-layer errors, activation variance) or a table.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from .escalation import EscalationPolicy
from .lookahead import PrefetchController
from .policy import ConditionalPolicy


@dataclass
class ServeDecision:
    prefetch: list = field(default_factory=list)
    prefetch_hit: float = 0.0
    prefetch_size: int = 0
    ranks: np.ndarray | None = None
    precision: str = "fast"
    entropy: float = 0.0
    dominant_mode: str = "none"
    coverage: float = 0.0        # rho(h): predicted reconstruction risk
    tier: str = "routine"        # escalation decision from rho


@dataclass
class ServeController:
    prefetch: PrefetchController
    policy: ConditionalPolicy
    top_k: int = 6
    rho: Optional[Callable[[np.ndarray], float]] = None
    escalation: Optional[EscalationPolicy] = None

    def step(
        self,
        draft_tokens,
        actual_tokens,
        logits,
        frob_err,
        act_variance,
        base_rank: int = 128,
        hidden=None,
    ) -> ServeDecision:
        """One decode step of the composed conditional stack.

        When rho (a per-token coverage predictor, see the G10 measured
        ridge rho) and hidden state are supplied, the escalation ladder
        runs: coverage -> routine/elevated/full tier, which the caller
        wires to rank/precision/cold-expert recovery actions.
        """
        plan = self.prefetch.plan(draft_tokens)
        hit = self.prefetch.observe(draft_tokens, actual_tokens)
        ranks, mode = self.policy.rank_profile(frob_err, act_variance,
                                               base_rank)
        ranks = self.policy.thermal_clamp(ranks)
        precision, entropy = self.policy.precision_decision(logits)
        coverage, tier = 0.0, "routine"
        if self.rho is not None and hidden is not None:
            coverage = float(self.rho(np.asarray(hidden, dtype=np.float64)))
            if self.escalation is None:
                self.escalation = EscalationPolicy()
            tier = self.escalation.decide(coverage, entropy)
            if tier == "full":
                precision = "full"
        return ServeDecision(
            prefetch=plan,
            prefetch_hit=hit,
            prefetch_size=len(plan),
            ranks=ranks,
            precision=precision,
            entropy=entropy,
            dominant_mode=mode,
            coverage=coverage,
            tier=tier,
        )
