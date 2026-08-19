"""Geometric jury aggregation (port of jury_gtc_kernel.c / Paper XV sec. 12).

The jury is the confidence formula to reuse for G9/G10 controller
decisions (expert cache plan, rank, precision, fallback) — NOT the GTC
cache itself. Given per-candidate confidences ``c_i``, the aggregate is:

    J = 1 - prod(1 - c_i)

where ``c_i = exp(-d_i / R)`` for geodesic distance ``d_i`` and coverage
radius ``R``. Zero confidences give J = 0; any single certainty gives
J = 1; J is monotone in every c_i. Also included: the two-stage domain
routing used by Jury-GTC (sample S candidates, softmax-weighted vote,
dominant + transfer domain, adaptive threshold from the dominant
domain's coverage radius).
"""

from __future__ import annotations

import numpy as np


def jury_confidence(confidences) -> float:
    """J = 1 - prod(1 - c_i). Inputs clipped to [0, 1]."""
    c = np.clip(np.asarray(confidences, dtype=np.float64), 0.0, 1.0)
    return float(1.0 - np.prod(1.0 - c))


def geodesic_confidence(distances, coverage_radius: float) -> np.ndarray:
    """c_i = exp(-d_i / R), the single-trial confidence of one juror."""
    d = np.asarray(distances, dtype=np.float64)
    if coverage_radius <= 0.0:
        raise ValueError("coverage_radius must be > 0")
    return np.exp(-d / coverage_radius)


def confidence_to_jury(distances, coverage_radius: float) -> float:
    """Shortcut: jury confidence straight from geodesic distances."""
    return jury_confidence(geodesic_confidence(distances, coverage_radius))


def domain_route(
    similarities: np.ndarray,
    domain_ids,
    coverage_radii: dict,
    n_sample: int = 20,
    temperature: float = 8.0,
) -> tuple[str, str | None, float]:
    """Two-stage jury routing (stage 1 of Jury-GTC).

    Parameters
    ----------
    similarities:
        cos-similarities of the query to each cached trajectory.
    domain_ids:
        domain label per trajectory (parallel to ``similarities``).
    coverage_radii:
        per-domain coverage radius R_d (used for the adaptive threshold
        tau = 1 - 2 * R_dominant).
    n_sample, temperature:
        stage-1 sample size and softmax temperature.

    Returns
    -------
    (dominant_domain, transfer_domain_or_None, tau)
    """
    sim = np.asarray(similarities, dtype=np.float64)
    ids = np.asarray(domain_ids)
    if sim.size == 0:
        raise ValueError("empty similarity vector")
    if n_sample >= sim.size:
        sample_idx = np.arange(sim.size)
    else:
        rng = np.random.default_rng(0)
        sample_idx = rng.choice(sim.size, size=n_sample, replace=False)

    w = np.exp(temperature * sim[sample_idx])
    w /= w.sum()
    vote: dict[str, float] = {}
    for wi, di in zip(w, ids[sample_idx]):
        vote[di] = vote.get(di, 0.0) + float(wi)

    ranked = sorted(vote.items(), key=lambda kv: -kv[1])
    dominant = ranked[0][0]
    transfer = ranked[1][0] if len(ranked) > 1 else None
    r_dom = coverage_radii.get(dominant, 0.02)
    tau = 1.0 - 2.0 * r_dom
    return dominant, transfer, float(tau)
