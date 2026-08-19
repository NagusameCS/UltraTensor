# CODER-SERVE — composed serving loop for the extracted V4-Coder

Design (2026-08-16), grounded exclusively in measured mechanisms. This
is the target runtime shape for Phase 2+ of docs/ROADMAP_V4CODER.md;
every component below carries a measured number from REVIEW_GAPS.md.

## Loop (one decode step)

    next_ids <- drafter (V4-Flash / lightweight)
    for layer L in 0..60:
        if L in 0..2 (hash):
            experts = HASH_TABLE[L][next_ids]     # exact, deterministic
        else (dense):
            scores  = CONTROLLER(h)               # factored ridge, 649k
            experts = top-6(scores over kept set)
            if margin(6th/7th) < 1.01: experts = top-8   # boundary guard
        rho = RHO(h)                              # ridge, Spearman 0.91
        tier = LADDER.decide(rho, entropy)        # routine/elevated/full
        prefetch(next experts)                    # during current GEMVs
        y = MoE(experts)                          # IQ2_XS kernels
        full path on tier=full                    # cold experts + F32

## Component map (measured)

| Component | Mechanism | Measured number |
|---|---|---|
| Hash-layer routing | token->expert table | `hash_route_tables.npz` (129280x6x3, 9.3 MB) — exact, zero GEMV |
| Dense-layer routing | factored ridge score regressor | rel-L1 0.086/0.269, 649k params (4.2x smaller than router) |
| Boundary guard | margin-aware top-8 | margin 1.003 mean; 100% of tokens <1.05 |
| Risk gate | ridge rho + escalation ladder | Spearman 0.91, tier agreement 1.0 (3-way split) |
| Prefetch | drafter + hash table (hash), controller scores (dense) | oracle cap-6 = 0 misses (G12) |
| Residency | hot = kept code experts (IQ2_XS) | keep64 = 74.5 GB IQ2_XS; veto list stays resident (G7) |
| Tail protection | CVaR veto on rare-code damages | tail gate (v4_tail_gate.py) |

## Budget example (laptop, 32 GB RAM)

- Hot tier: top-32 code experts x 61 layers, IQ2_XS ~ 37 GB (keep32)
  — plus hash tables + router + controller + rho (<< 1 GB).
- Warm: prefetched next-token experts (6 x 3 tensors ~ 90 MB/token).
- Cold: remaining kept experts on NVMe, paged by the tier policy.

## Honest limits

- Dense-layer routing is only as good as the controller (0.27 rel-L1
  worst split) and the boundary guard costs 2 extra experts on ~95%
  of tokens — budget accordingly.
- The extraction keeps only CENSUS-verified experts; quality on
  out-of-distribution code is gated by the rho ladder and the CVaR
  veto, not guaranteed.
- This loop is the SERVING shape; the extraction itself (router
  refit, IQ2_XS requant) is Phase 2, distillation is Phase 4.
