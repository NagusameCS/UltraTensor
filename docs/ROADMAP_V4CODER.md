# ROADMAP: V4-Coder — a domain-extracted coder model from V4-Pro

Status: program started 2026-08-16. Builds on the measured verdicts in
docs/REVIEW_GAPS.md. Goal: a laptop-runnable, V4-quality coding model
extracted from the 697 GB V4-Pro, using the conditional toolkit.

## Why this is the right target

The verdict board says the per-expert latent space of V4 gate experts
cannot be cut globally (G4/G5/G6/G11). But V4 concentrates intelligence
in ROUTING + expert composition, and MoE was designed so domains can
occupy different experts. The coder play is therefore at the model
level, not the weight level:

  1. CODE CENSUS       — which experts carry code traffic
  2. SUBNET EXTRACTION — router + code-active experts only
  3. DOMAIN SUBSPACE   — low-rank projection fit on code inputs only
  4. DISTILLATION      — dense/MLA coder from V4 code outputs
  5. TAIL GATE         — CVaR-validate every cut on code traffic

## Phase 1 — Census (in flight)

- scripts/v4_make_code_battery.py: 40-prompt cross-language battery,
  tokenized with the V4 tokenizer -> outputs/code_prompts.json.
- Cluster: 96-token dense trace of the code battery
  (cluster_dense_trace.py, /mnt/nas20/exp_code, auto-chains after
  exp96) -> per-expert routing mass on code traffic.
- scripts/cluster_code_census.py: top-64 code experts per layer,
  overlap with general traffic (exp96), code-exclusive experts,
  distinct-expert count, and the code-only domain-subspace projector
  test (heldout_rank_curves, train 64).
- Decision gates:
  * code mass concentrated? (top-64 share, distinct count vs 384)
  * code-exclusive experts exist? (subnetwork extraction viable)
  * code subspace holds rank better than mixed? (domain projection
    viable — the G4 correction retested per domain)

## Phase 2 — Subnetwork extraction (after census)

CENSUS VERDICT (2026-08-16): viable — keep64 per dense layer. Build
artifacts landed: scripts/v4_coder_manifest.py (build spec:
outputs/coder_manifest.json, 74.5 GB IQ2_XS extrapolated),
scripts/v4_hot_tier_smoke.py (real loader: 4 experts = 1.06 GB fp64,
2.2 s/token numpy reference). Cluster scripts parameterized for the
L4-L10 census (exp_mid) to verify the L3 assumption layer-wise.

BUILT (2026-08-16): Y:/models/coder/DeepSeek-V4-Coder-keep64.gguf —
156.1 GB, 1894 tensors, version 3, manifest KV embedded. Dense layers
3-60 keep 64 code experts (gate/up Q3_K, down Q4_K); hash layers 0-2
keep all 384 (Q3_K/Q5_K) with the I32 tid2eid tables; router + all
attention tensors copied verbatim. 4.3x smaller than the 677 GB full
model at Q3_K. Produced by ultratensor/gguf_keep.py
(write_keep_gguf, round-trip tested) + scripts/v4_coder_keep.py.
NEXT: IQ2_XS requant (~74.5 GB) and a llama.cpp serve smoke.

SERVE STATUS (2026-08-17, updated):
- UNIFORM KEEP16 LOADS NATIVELY: DeepSeek-V4-Coder-keep16u.gguf
  (42.2GB, 16 experts on EVERY layer, expert_count=16 metadata,
  router sliced to 16 cols, hash tables remapped) passes the fork's
  check_tensor_dims with NO patch. Prompt processing observed at
  0.3 t/s CPU under imatrix contention (llama-cli, local disk).
  This is the serve path for low-end machines: uniform-E extraction
  is the fork-patch killer.
- Next quality lever: per-specialist router refit (v4_train_router
  scaffold) to lift keep16's ~35% naive routing coverage.
- QUANTIZE PATH WORKS: llama-quantize loads keep64 fine (no graph dim
  check) and accepts the existing Flash imatrix (725 entries, names
  match). IQ2_XS requant RUNNING: 156.1GB -> 95.3GB (2.36 BPW).
- Serving paths: (a) our runtime (hash tables + factored controller +
  ExpertStore per-expert streaming — node2/laptop have only 30/32GB
  RAM, so 384-expert layers MUST stream per routed expert), (b)
  Phase-5 llama.cpp fork patch (per-layer expert counts).

RARE-TAIL CENSUS (exp_rare, 96 rare/adversarial tokens): rare code
SPREADS — 124 distinct experts vs 77, top-64 mass 0.875 vs 0.977,
subspace hold@8 0.579 vs 0.365, controller rel-L1 0.107/0.255 vs
0.056/0.188. The tail lives OUTSIDE the code subnetwork.

TAIL-GATE VETO (scripts/v4_tail_gate.py, outputs/tail_gate_L3.json):
veto list = [88, 244, 79] — experts cheap on ordinary code but
catastrophic on rare code (e244: code CVaR 0.222, rare CVaR 0.812).
All three must stay resident in the extracted coder; the gate is the
Phase-2 quality contract enforced on real bytes.

- If code-exclusive experts exist: keep router + code-active experts
  per layer; renormalize/re-fit router on code traffic; IQ2_XS
  quantization of the kept experts. Target: 4-6x smaller before
  quantization, 50-100 GB after — consumer hardware.
- Quality contract: CVaR tail gate on the extraction damage per expert
  (cvar.prune_report with real ablation damages on code prompts);
  escalation ladder refuses degraded tokens (routine/elevated/full).

## Phase 3 — Domain subspace (after census)

- Fit the activation subspace on code inputs only; deploy A = W @ V_k
  projectors at the largest k the held-out code curve supports. The
  hypothesis: code is a tamer distribution than mixed traffic, so the
  effective rank is far above the mixed-traffic result.
- Only pursue if the code held-out curve shows a knee (mixed traffic
  had none below r=16 — see G4 CORRECTED).

## Phase 4 — Distillation (needs V100/Flash)

- Generate a code corpus with the full model (V100/Flash at 25-60
  tok/s makes this days, not months); distill into a 7-14B dense/MLA
  coder; quantize to IQ2/IQ3. This is the classic 90% compression,
  paid upfront in generation compute.
- scripts/v4_eval_battery.py is the acceptance test (code PPL +
  top-logprob agreement vs the full model).

## Phase 5 — Serving

- V100 rebuild (docs/CLUSTER_V100_PLAN.md) serves the full model for
  corpus generation; the extracted coder then runs on the laptop.
- Wire PrefetchController + tiering with the drafter for the full
  model's decode loop.

## Measured inputs so far

CENSUS VERDICTS (2026-08-16, exp_code, 87 code tokens, outputs/code_census.json):
  * concentration: dense L3 top-64 = 97.7% of code routing mass, 77
distinct experts; hash layers 0-2 SPREAD (top-64 only 0.50-0.53 —
token-determined, handled by the 9.3MB table instead).
  * exclusivity: 32 of L3's top-64 code experts are code-exclusive
(not in the general top-64) — real domain subnetwork exists.
  * code subspace: hold_pca@8 = 0.3645 (L3) vs 0.50 general — gain
0.155 at r=16, but still no knee to r=64 (0.288): rank stays weak
even on code.
  * G9-on-code: factored controller rel-L1 0.056/0.188, agreement
0.68/0.59 at 483k params — better than general traffic.
  * ROUTER COVERAGE (scripts/v4_router_refit.py): keep64 covers
97.5% mean / 88.5% full of code top-6; keep96 adds nothing; keep32
only 50.6% full. DECISION: keep64, cold fallback for the rest via
the rho ladder.
- exp96 verdicts (86 tokens, train 64 / hold 22, committed 2617e78):
  * G4: held-out projector error has NO KNEE up to r=64 on mixed
    traffic (L0 keeps 38-49% at r=64) — the global rank lever is
    dead at any deployable rank. Phase 3's domain-subspace retest on
    code traffic is the only remaining rank hope.
  * G9: factored score-regression controller rel-L1 0.086/0.269 at
    649k params (4.2x smaller than the router) — the controller
    technology for the extracted coder's router replacement.
  * G10: KNN rho flips to Spearman +0.54 at n=64 — risk prediction
    works with calibration; the ladder can gate Phase 2 extraction.
- PQ-the-router test (scripts/v4_pq_router.py, outputs/pq_router_L3.json):
  4-bit agreement 0.03, 6-bit 0.24, 8-bit 0.72 at 67% params — PQ is
  NOT a cheap-router lever either (flat router spectrum). The factored
  ridge score regressor (181k params) stays the controller candidate.
- PPL battery live: code prompt PPL 1.15 on the :8774 server (first
  datum of the acceptance battery).
- exp_code census chain RUNNING on node2: 87-token domain-spanning
  code trace -> code census + code-domain subspace test + G9-on-code.
- exp_rare chain STAGED on node2 (auto-starts after exp_code): rare/
adversarial code trace (14 prompts) + rare-vs-code census — the
CVaR tail-gate calibration set for Phase 2 extraction.
- OVERNIGHT 2026-08-17 (exp256 harvest, watchdog):
  * exp256 (256-token general trace, L0-3) DONE; subspace_proj +
    controller_shrink + ffn_inputs harvested to outputs/exp256_*.
  * RHO@192 — best predictor result yet: 2-way ridge rel-L1 0.0415
    Spearman 0.971 tier-agreement 0.91; 3-way (proj 128 / fit 64 /
    hold 64) rel-L1 0.0281, Spearman 0.98, tier-agreement 1.0.
    The ladder's score regressor generalizes at calibration scale.
  * exp_mid crashed at L8: NAS /mnt/nas20/models/v4pro had only 2
    of 17 shards (previous jobs never touched layers >= 8). Shard
    00003 (L8-10) being copied; exp_mid -> exp_pl chain re-staged
    (launch_mid_pl.sh); bulk copy of remaining 14 shards scheduled
    04:00 via UltraTensorCopy task.
  * exp_mid round 2: passed L8-9, crashed at blk.10 (hc_attn_fn in
    shard 00004). Shard 00004 (L10-14) copied; chain relaunched.
    Lesson: blk.10 tensors span shards 2 AND 3.
- Requant pipeline gotchas (2026-08-17):
  * this fork's llama-imatrix: --chunk = from-chunk (start offset),
    NOT chunk size; there is no chunk-size flag. Use --chunks N.
  * Flash imatrix (ds4-0731) has only 725 sparse entries -> useless
    for keep64 IQ2_XS; computing full Pro imatrix instead
    (scripts/imatrix_diff.py diffs coverage).
  * IQ2_XS dry-run: 156.1GB -> 95.3GB (2.36 BPW); quantize path has
    no graph dim-check, so it works on mixed-E keep64.

## Open items

- exp_code census results (auto-chained on node2).
- exp_rare tail census (auto-chained after exp_code).
- Phase-2 gate tooling landed (scripts/v4_tail_gate.py code-vs-rare
  CVaR veto list; scripts/v4_coder_extract.py decision gates); they
  consume the census outputs at harvest.
- Battery retry: math prompt timed out on :8774 (retry via
  scripts/v4_eval_battery.py --only math --timeout 1800).
