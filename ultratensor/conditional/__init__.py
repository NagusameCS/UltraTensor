"""Conditional-resource toolkit for UltraTensor (successor to HyperTensor).

Faithful Python ports of the HyperTensor runtime mechanisms that make
inference conditional on precision, temperature, layer position,
draft-rejection feedback, activation phases, and cache confidence:

- ``precision``  : APC adaptive precision cascade (runtime/nn/speculative.c)
- ``thermal``    : thermal-adaptive rank + tokens-per-joule (thermal_rank.c)
- ``basis``      : qspec shared-basis feasibility + frank rank profiles
                   (qspec_basis.c)
- ``online_basis``: Oja-rule online PCA from draft rejections (online_basis.c)
- ``sinks``      : MCR phase detection + attention-sink protection
                   (mcr_compress.c)
- ``jury``       : geometric jury confidence aggregation (jury_gtc_kernel.c)

Provenance: same author, MIT. Original C sources live in the HyperTensor
repository; these ports keep the semantics, the names, and the default
constants, and drop the ASCII banners.
"""

from .precision import apc_gate, apc_softmax, apc_stats, shannon_entropy
from .thermal import (
    NullSensor,
    NvmlCtypesSensor,
    ThermalRank,
    TpjTracker,
)
from .basis import (
    FrankResult,
    QspecEntry,
    QspecResult,
    frank_apply,
    frank_build,
    qspec_test_shared_basis,
)
from .online_basis import OnlineBasis
from .sinks import (
    MCRResult,
    SinkResult,
    mcr_detect_phases,
    mcr_rank_budget,
    sink_check_basis_coverage,
    sink_detect,
)
from .jury import (
    confidence_to_jury,
    domain_route,
    geodesic_confidence,
    jury_confidence,
)
from .lookahead import (
    PrefetchController,
    PrefetchCurve,
    WorkingSetModel,
    evaluate_prefetch,
    oracle_curve,
    working_set_union,
)
from .drafting import DraftPlan, expected_acceptance, optimize_slots
from .policy import ConditionalPolicy, PolicyState
from .actweight import (
    RankErrorCurve,
    activation_error,
    frob_error,
    rank_error_curve,
    svd_truncate,
    weighted_pca_truncate,
)
from .spec_sim import (
    SpecResult,
    accept_prefix,
    expected_acceptance_geometric,
    simulate,
)
from .stats import (
    BootstrapResult,
    bootstrap_ci,
    eviction_ablation,
    fine_k_sweep,
    intrinsic_dim_compare,
    rank_ablation,
    sink_ablation,
)
from .shared_factor import (
    SharedFactorFit,
    compare_budgets,
    fit_shared_dict,
)
from .tiering import TierResult, knee, simulate_tier, tier_sweep
from .vq import PQResult, pq_bits_vs_error, pq_reconstruct, product_quantize
from .controller import ServeController, ServeDecision
from .curvature import curvature_correlation, grcurv_to_rank_budget
from .escalation import EscalationPolicy
from .rank_policies import compare_policies, fixed_allocation, gini

__all__ = [
    "apc_gate",
    "apc_softmax",
    "apc_stats",
    "shannon_entropy",
    "NullSensor",
    "NvmlCtypesSensor",
    "ThermalRank",
    "TpjTracker",
    "FrankResult",
    "QspecEntry",
    "QspecResult",
    "frank_apply",
    "frank_build",
    "qspec_test_shared_basis",
    "OnlineBasis",
    "MCRResult",
    "SinkResult",
    "mcr_detect_phases",
    "mcr_rank_budget",
    "sink_check_basis_coverage",
    "sink_detect",
    "confidence_to_jury",
    "domain_route",
    "geodesic_confidence",
    "jury_confidence",
    "PrefetchCurve",
    "PrefetchController",
    "WorkingSetModel",
    "evaluate_prefetch",
    "oracle_curve",
    "working_set_union",
    "DraftPlan",
    "expected_acceptance",
    "optimize_slots",
    "ConditionalPolicy",
    "PolicyState",
    "RankErrorCurve",
    "activation_error",
    "frob_error",
    "rank_error_curve",
    "svd_truncate",
    "weighted_pca_truncate",
    "SpecResult",
    "accept_prefix",
    "expected_acceptance_geometric",
    "simulate",
    "BootstrapResult",
    "bootstrap_ci",
    "eviction_ablation",
    "fine_k_sweep",
    "intrinsic_dim_compare",
    "rank_ablation",
    "sink_ablation",
    "SharedFactorFit",
    "compare_budgets",
    "fit_shared_dict",
    "TierResult",
    "knee",
    "simulate_tier",
    "tier_sweep",
    "PQResult",
    "pq_bits_vs_error",
    "pq_reconstruct",
    "product_quantize",
    "ServeController",
    "ServeDecision",
    "curvature_correlation",
    "grcurv_to_rank_budget",
    "EscalationPolicy",
    "compare_policies",
    "fixed_allocation",
    "gini",
]
