"""APC — Adaptive Precision Cascade (port of HyperTensor speculative.c).

Most inputs are "easy": they produce high-confidence outputs even at
reduced precision. Only ambiguous inputs need the full dynamic range.
The gate runs the cheap path first, computes the Shannon entropy of the
output distribution, and escalates only when entropy crosses a
threshold (default 0.5 bits, as tuned in the C runtime).

This is the review-demanded "conditional precision" primitive: wire the
gate in front of a quantized decoder and feed the escalation decision to
a higher-precision path.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

APC_ENTROPY_THRESHOLD: float = 0.5


def apc_softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax (max-subtracted), same as soft_inplace()."""
    x = np.asarray(x, dtype=np.float64)
    maxv = float(np.max(x)) if x.size else 0.0
    e = np.exp(x - maxv)
    return e / np.sum(e)


def shannon_entropy(probs: np.ndarray) -> float:
    """Shannon entropy in bits; probabilities <= 1e-7 contribute zero.

    Matches the C shannon_entropy(): 0 = perfectly certain,
    log2(N) = uniform.
    """
    p = np.asarray(probs, dtype=np.float64)
    h = 0.0
    for pi in p.flat:
        if pi > 1e-7:
            h -= pi * np.log2(pi)
    return float(h)


@dataclass
class apc_stats:
    """Mirrors apc_stats_t from speculative.h."""

    total_inferences: int = 0
    fast_hits: int = 0
    escalations: int = 0
    cycles_saved: int = 0

    def hit_rate(self) -> float:
        if self.total_inferences == 0:
            return 0.0
        return self.fast_hits / self.total_inferences


def apc_gate(
    outputs: np.ndarray,
    threshold: float = APC_ENTROPY_THRESHOLD,
    stats: apc_stats | None = None,
) -> tuple[bool, float]:
    """Decide fast-path vs escalation for a raw output vector.

    Parameters
    ----------
    outputs:
        Raw output of the cheap (quantized) path, before softmax.
    threshold:
        Entropy threshold in bits; entropy >= threshold escalates.
    stats:
        Optional stats object updated in place.

    Returns
    -------
    (fast_path, entropy)
        ``fast_path=True`` means the quantized output is accepted;
        ``False`` means escalate to full precision.
    """
    outputs = np.asarray(outputs, dtype=np.float64)
    probs = apc_softmax(outputs)
    entropy = shannon_entropy(probs)
    fast = entropy < threshold

    if stats is not None:
        stats.total_inferences += 1
        if fast:
            stats.fast_hits += 1
        else:
            stats.escalations += 1
    return bool(fast), entropy


def apc_demo_stats() -> apc_stats:
    return apc_stats()
