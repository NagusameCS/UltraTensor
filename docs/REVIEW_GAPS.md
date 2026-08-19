# Review Gap Backlog — conditional V4 (Phase 7 direction)

Status: 2026-08-16. Synthesis of the external reviews of the HyperTensor
volume + the V4-Pro measurements. Every item the reviews demanded but we
have NOT yet executed is listed here with a concrete first step on our
stack. This is the next work queue.

## The reframing we adopt

A DeepSeek-V4-class MoE is TWO compression targets, not one:

- **Active-path cost**: ~49B of 1.6T params per token, disk-bound on
  modest hardware (~22-25 GiB/token measured).
- **Stored-model capacity**: 619 GB of Q3_K expert payload (measured).

Goal: make the model **conditional at every resource level** — conditional
residency, rank, precision, context, state — with the contract
"compression failure => progressive recovery", never silent wrong output.

## What the reviews demand that we already have

- Profile before projecting: rank-vs-energy sweep (9 tensors x 8 experts),
  route-sensitivity (hash vs dense, Q2_K gate 86% top-8, corr 0.973) —
  the negative results ARE the profiling.
- Expert-as-unit reads (read_expert), per-expert q2_0 encode/decode
  (18.9 ms/tensor C), oracle-validated kernels.
- Attention left untouched (no damage to CSA/HCA) — correct default.
- Honest single-machine reporting, reconstruction-metric caveats.

## Gap list (must hit)

### G1. Baseline task harness BEFORE further weight surgery
Missing: PPL/logit-KL, routing entropy + top-k stability, long-context
retrieval, code/math/multilingual/rare-domain slices, prefill+decode
latency, HBM/expert-load stats. We have numerics + measurements only.
First step: `scripts/bench_harness.py` driving geodessical + :8774 with a
fixed prompt battery, recording per-token router traces.
FIRST SLICE MEASURED (2026-08-16, scripts/v4_route_stability.py,
outputs/route_stability_L3.json, 24 real tokens, layer 3): routing
entropy is PINNED at 1.786-1.792 nats (mean 1.7909 ~= ln 6): the
top-6 selection is almost exactly UNIFORM. Selection margin (6th/7th
score) is 1.003 mean, >=1.011 max: the cut is a coin flip on every
token. Consecutive-set overlap 0.333 -> 0.667 per-step churn; 68
distinct experts over 24 tokens. PPL/KL and rare-domain slices still
need the serving harness.
PPL BATTERY RUNNER LANDED (2026-08-16, scripts/v4_eval_battery.py):
endpoint-agnostic OpenAI-schema battery (code/math/multilingual/
rare/needle) with per-token logprob extraction, per-prompt PPL, and
top-logprob agreement vs a second endpoint. First live probe launched
against the resident llama-server :8774 (V4-Pro shard 1 + DSpark
drafter, 8 max tokens) — slow-server PPL numbers in flight; the full
battery is the day-one eval on the V100/Flash node.
LIVE-SERVER FINDING: code ppl=1.152, multilingual ppl=1.023 collected;
math/rare/needle requests HANG on :8774 — the server stalls mid-
generation on those prompts (health endpoint stays 200, requests
queue behind the stall). Consistent with the known DSpark-drafter
pairing instability on V4-Pro; retry against the drafter-disabled
server or on Flash before treating PPL as authoritative.

### G2. Expert similarity map in ACTIVATION-output space
We measured weight-space cross-correlation (< 1e-3, dead). The review
demands activation-space: sim(e_i,e_j) = E_x cos(y_ei(x), y_ej(x)).
First step: collect expert intermediate outputs via the C runtime
(v4_block already computes MoE outputs per layer) over the harness
corpus; cluster; report vs the weight-space negative.
MEASURED 2026-08-16 (cluster, real bytes, layers 0 and 3, 12 probe
experts on 8 real routed hidden states): mean off-diagonal activation
cos-sim 0.037 (L0) / 0.007 (L3), max 0.055 / 0.047; weight-space corr
~1e-5. Experts are functionally DISTINCT in activation space too ->
expert deduplication (review Phase 2) has no targets. G2 closed as a
measured negative (n=8 caveat; wider traffic may add weak clusters).
CONFIRMED at n=24 (exp24): L0 0.0388 / L3 0.0078 - stable, not a
small-sample artifact.

### G3. Lookahead expert-cache predictor (H-step working set)
Missing entirely; we have lazy all-cold streaming + batch projections.
First step: log router traces (already possible: hash 0-2, dense 3+ via
route_layer) -> train/measure Pr(e used in next H) for H=1,4,8,16;
optimize prefetch threshold tau_e against tail latency.
MEASURED 2026-08-16 (phase A, real hash-layer bytes, 86 tokens/8
prompts): H=1 oracle union = exactly 6 experts (2.3% of layer) ->
token-bigram predictor 81-82% hit at size 6.0, degrading to 0.41-0.42
hit at H=16 (size 62-67 vs oracle 74-76). Set-level frequency/Markov
degenerates to "prefetch all 245-256 experts" — useless. Hash routes
are a deterministic function of the NEXT TOKEN, so the lookahead
predictor for hash layers IS a token drafter: one mechanism serves both
speculative decoding and expert prefetch. Script + data:
scripts/v4_router_trace.py, outputs/router_trace_hash.json.
Remaining: dense-layer traces (phase B, needs real forwards) and a
draft-driven prefetch controller wired to the reader thread.
CONTROLLER-LEVEL (2026-08-16, scripts/v4_prefetch_eval.py on real
hash traces): with a PERFECT H=4 lookahead, full prefetch coverage
needs exactly the 24-expert union (cap6 26% overlap -> cap12 52% ->
cap18 77% -> cap24 100% full hit). With the weak bigram drafter the
overlap plateaus at ~21% at EVERY cap: drafter quality, not residency,
is the binding constraint for hash layers. A drafter at V4-Flash
quality (0.82-0.99 tok/s measured) sits between the two curves.

SPEC PROJECTION (2026-08-16, scripts/spec_projection.py, measured
constants): with verifier T_V=8.17 s/token (lazy-resident 0.122 tok/s)
and drafter T_D=1.1 s/token (V4-Flash 0.9 tok/s), speculation caps at
0.754 tok/s even with a PERFECT drafter (gamma=32) - independently
matching the batch-256 aggregate ceiling 0.708. At realistic
acceptance a~0.5 it gives only 0.12-0.15 tok/s. 1 tok/s is UNREACHABLE
by drafting alone: the verifier's disk-bound read time is the wall.
Reframes GSD: drafting is the expert-prefetch lever, not the throughput
lever. Throughput needs the read path (q2_0 experts, batch fusion).

### G4. Per-expert activation-weighted compression Pareto sweep
Missing: E_{l,e,s}(r) = E_x ||(W-What)x||^2 was never measured; we used
Frobenius/energy on weights, which the reviews flag as a weak proxy.
Candidates per expert: FP8/FP4, AWQ-style rank-r, grouped-dictionary,
removal + router renormalization, removal + LoRA recovery.
First step: 1-3 layers, hot/medium/cold experts from G2/G3; sweep and
record conditional KL + routing delta + decode throughput (Pareto).
IN FLIGHT (cluster): dense trace landed (8 tokens, blocks 0-3, real
forwards on rognode2); ffn_inputs_dense.npz saved. Dense layer 3 used
32/256 distinct experts over 8 tokens vs 43-47/256 for hash layers -
first real confirmation that dense routing concentrates (explains the
hash-vs-dense latency gap). actweight curves running on the gate
experts (down experts need the 3072-dim intermediates, not hidden).
FIRST RESULT (2026-08-16, layer 0 ffn_gate_exps [3072,7168], real
inputs): k95_frob = 2301 vs k95_act = 8. The activation-weighted
operator is nearly rank-8 on real traffic while the weight spectrum
stays flat - the review's point made quantitative. CAVEAT: 8 input
samples bound the activation rank at 8; a 24-token run on the cluster
is lifting the bound now (exp24).
FINAL (n=24, exp24): k95_act STILL 8 (L0 2301->8, L3 2420->8;
act@512 0.367/0.451 vs frob@512 0.580/0.617). With 24 samples the
empirical rank could have risen to 24 and did not: the routed inputs
live in a tiny subspace of the weight row-space. G4 closed: per-expert
activation-weighted factorization is viable at rank <~16-24 even
where SVD says 2300+.
CORRECTED (2026-08-16, held-out split, scripts/v4_subspace_proj.py):
the n=24 k95_act=8 was IN-SAMPLE OVERFIT. Fit the projector on 16
tokens, evaluate on the held-out 8, both split directions, layers 0/3
x gate experts 0/1 (outputs/subspace_proj.json): hold_pca@8 =
0.50-0.75 and STILL 0.47-0.73 at r=16 (the max 16 train tokens can
fit); in-sample collapses to 0.0 at r=24 trivially (24 samples span
<=24 dims). No knee is visible in the trainable range: most expert
output energy lives OUTSIDE any subspace 24 tokens reveal. Weighted
vs plain PCA projector: <0.01 apart on held-out, so the simple form
A = W @ V_k + per-token A(V_k^T x) is the right deployable shape.
STATUS: the deployable rank is UNKNOWN and NOT 8-16; the projector
machinery + held-out split is now the standard for G4. Next
experiment: 256+ token dense trace on the cluster to bound the
rank-error curve beyond r=16.
QUEUED (2026-08-16): exp256 — 256-token dense trace + cluster_subspace
(train 192, ranks to 192), chained after exp_rare on node2. This is
the final-number run for G4/G9 and the calibration base for the G10
rho 3-way split at scale.
FINAL AT n=86 (2026-08-16, exp96, train 64 / hold 22, four gate
experts): held-out error DECREASES SLOWLY with rank and shows NO
KNEE up to r=64 — L0 keeps only 38-49% of output energy at r=64,
L3 keeps 42-59%. The rank lever on gate experts is dead at ANY
deployable rank, not just 8-16 — the n=24 correction was real and
4x more data did not revive it. The ONLY remaining rank hope is the
DOMAIN-subspace (code-only) retest, in flight via the census
(scripts/cluster_code_census.py).

### G5. Conditional-rank sweep (fixed / per-layer / per-expert / token-adaptive)
Fixed rank is dead (flat spectra); conditional rank was never tested.
First step: k(x) = k_min + dk * g_{l,e}(h,x) with a small gating
predictor trained on calibration traces; compare rank policies on the
harness.
IN FLIGHT (2026-08-16, scripts/v4_rank_sweep.py): four rank policies
(fixed / mass-proportional / sqrt water-fill / top4) at budgets
64/128/256 over the top-12 ROUTED gate experts of layer 3 (24 real
tokens), metric = routing-mass-weighted activation error. Writes
outputs/rank_sweep_L3.json. Question: does conditioning rank on
routing mass beat uniform at equal budget on real bytes?
MEASURED (outputs/rank_sweep_L3.json): conditional rank WINS at every
budget (mass vs fixed: 0.924 vs 0.936 @64, 0.884 vs 0.907 @128,
0.843 vs 0.857 @256) - but the win is only 1-2.5% relative and the
ABSOLUTE error is 0.84-0.94 at every deployable budget: rank
compression of gate experts loses no matter the policy (consistent
with the G4 held-out correction - the routed inputs do not live in a
tiny subspace). G5 closed as a weak-positive: conditional rank beats
uniform but the rank lever on expert GEMMs is dead; capacity (q2_0)
and the read path remain the levers.

### G6. Shared-plus-private factorization
W_e ~= W_shared + U A_e V^T + R_e  (global + cluster + private) was NOT
implemented — only independent per-tensor SVD.
First step: one MoE layer; fit W_shared + dictionaries by stacked SVD of
the activation-weighted covariance (not raw weights); benchmark vs
independent factorization (which we know loses).
MEASURED (2026-08-16, scripts/v4_shared_factor.py, 8 real routed gate
experts of layer 3, 24 tokens, outputs/shared_factor_L3.json): at the
SAME 41.9M-param budget, independent rank-512 achieves act error
0.399 while shared (r_shared 8/16 + private rank 243) achieves 0.498
— sharing LOSES. At the 10.5M budget shared cannot even be built:
the W_shared matrix alone costs one full expert (22M params) and the
experts are functionally distinct (G2: act cos 0.039, weight corr
1e-5), so the shared dictionaries contribute almost nothing
(r_shared 8 vs 16 differ by 0.0004). G6 closed as a measured negative
for gate experts: there are no duplicated bases to share.

### G7. Tail-risk pruning with CVaR slices
Missing: conditional-damage D_e for expert removal and CVaR utility for
latent directions. First step: rare-domain slices in G1 (code, math,
multilingual, long-context) -> D_e = E[D(p_base||p_ablated) | e in TopK]
per candidate removal; only prune low mean AND low tail.
MACHINERY LANDED (2026-08-16, ultratensor/conditional/cvar.py + 6 tests):
per-candidate CVaR_alpha (mean of the worst (1-alpha) slice), mean+CVaR
double-gated prune_mask (median thresholds or explicit), bootstrap CIs
on CVaR, JSON-ready prune_report. Real per-expert ablation damages
D_e are still pending - the exp96 cluster trace is the calibration
base for that ablation run.
FIRST REAL DAMAGES (2026-08-16, scripts/v4_expert_damage.py,
outputs/expert_damage_L3.json, top-8 routed experts of layer 3, 24
tokens): D(e,t) = expert output magnitude / selected-set total
(first-order proxy, no router renormalization). The double gate
marks 4 low-mass experts safe (mean 4-9%, CVaR <=0.38, worst <=0.54)
and VETOES 2 experts with moderate means (0.12-0.19) because their
worst-token contribution is 1.0 - some token's selected output is
entirely theirs. The tail gate catches exactly the rare-dependency
the reviews demanded. Full KL ablation still pending.

### G8. Trajectory-backed context (semantic-gated CSA/HCA extension)
GTC exists but live-substitution was 0% correctable at current density;
it was never integrated with V4's CSA/HCA (4x/128x blocks).
First step: measure V4 attention mass per compressed-row slot; evaluate
novelty/retrieval-gated retention against uniform CSA/HCA on
needle-in-a-haystack + multi-doc synthesis.
Note (verified 2026-08-16): geodessical has NO speculative path at all
(zero draft/verify/jury symbols in host/ + runtime/). Paper III GSD
(compressed-attention drafter + verifier batch, 38.5% acceptance) was
measured only in HyperTensor's OTT stack; the only drafter ever
exercised on V4 was llama.cpp's external DSpark pairing (worked on
V4-Flash 0.82-0.99 tok/s, failed to pair on V4-Pro). The HyperTensor
"jury" is a GTC trajectory-cache confidence gate (J = 1 - prod(1 - c_i),
two-stage domain routing) — it has nothing to vote on until GTC is
wired and above 0% correctable density; its confidence-aggregation
formula is the model to reuse for the G9/G10 controller decisions.
FIRST SLICE (2026-08-16, scripts/v4_mcr_phases.py, outputs/mcr_phases.json):
MCR phases on the real 24-token hidden trajectories (blocks 0-3):
variance ratio 1.47 (valid), L0=compress, L1-L3=refine; hidden
variance rises 0.010->0.021 L0->L2 then falls at L3. 4-layer first
datum; attention-mass-per-slot (CSA/HCA sink placement) still needs
attention forwards.
NEXT STEP (identified): patch v4_ref_gen.BlockGen to log per-token
attention mass over the compressed row slots during a cluster dense
trace (the serve oracle is single-token, mass is trivially 1.0 there).
Deferred behind the V4-Coder census; not a gate for Phase 2.

### G9. Router distillation into a tiny deployment controller
Missing: pi(phi_t) = (expert cache plan, rank, precision, context level,
fallback). First step: distill layers 3+ dense routers + hash tables
into a small always-resident predictor; measure router-consistency loss.
MEASURED 2026-08-16 (n=24 real inputs, layer 3): ridge controller
achieves top-k agreement 1.0 - but at 2.75M params, the SAME size as
the router it replaces (384x7168). Pipeline + metric validated;
"tiny" (<=100k params) and the agreement-vs-size curve need more
traffic. Next: low-rank/SVD-compressed controller + larger trace.
SHRINK ROUND 1 (2026-08-16, scripts/v4_controller_shrink.py,
outputs/controller_shrink.json, train 16 / hold 8, layer 3): the
in-sample 1.0 was overfit too - held-out top-6 agreement is 0.44
(fwd) / 0.13 (rev): split-unstable at 16 train samples. The FACTORED
ridge (W_c = X^T B, n*(d+c) params) is NUMERICALLY IDENTICAL to the
dense ridge at 15x fewer params (181k vs 2.75M) and MACs - lossless,
but agreement stays data-limited. SVD-truncated router needs k>=128
(966k params) to match the ridge held-out 0.4-0.5: the router is not
low-rank (consistent with G4's flat spectrum). exp96 (86 tokens,
train 64) launched on the cluster to lift the bound.
BOUNDARY REFRAME (2026-08-16): 100% of held-out tokens have their
6th/7th score margin <1.05 (88-100% <1.01). The top-6 SET is the
wrong controller objective - even the true router's boundary cut is
a coin flip between nearly-equal scores, so set-agreement failures
are near-free in output space. The controller should regress SCORES
(and top-k regret), not sets; agreement metrics must be reported
margin-weighted.
SCORE-REGRESSION CONTROLLER MEASURED (same 16/8 split): factored
ridge on scores reaches hold rel-L1 0.175 (fwd) / 0.365 (rev) at
181k params vs ~1.0 for the membership-fit scores; in-sample 0.001
(16-sample span). Still data-limited and split-unstable; exp96
train-64 is the first meaningful read.
FINAL AT n=86 (2026-08-16, exp96, train 64 / hold 22): the score-
regression factored controller holds rel-L1 0.086 (fwd) / 0.269
(rev) at 649k params (4.2x smaller than the router) — predicting
91%/73% of router score magnitude on unseen tokens. Set agreement
(0.58/0.36) beats the membership-fit ridge (0.52/0.29) and is
capped by the boundary coin flip (100% of tokens have margin <1.05;
no sharp-boundary token exists). G9 CLOSED as a validated weak-
positive: the score-regression objective is right, the controller is
4x smaller, and residual error concentrates in near-tied boundaries
that the router itself cannot resolve — i.e. near-free in output
space.

### G10. Reconstruction-risk / escalation policy
Missing: rho(h) = KL(full||compressed) predictor and the escalation
ladder (more rank -> more precision -> cold-expert load -> full path).
First step: calibration traces -> rho predictor; wire geodessical to
switch paths on rho threshold (full path already exists as reference).
FIRST HELD-OUT TEST (2026-08-16, scripts/v4_rho_predictor.py,
outputs/rho_predictor_L3.json): coverage = rank-8 projector
kept-energy per token (fit on 16 train). KNN rho predictor (nearest
train token in cosine space) achieves mean abs error 0.33 and
SPEARMAN -0.40 (anti-correlated) on the 8 held-out tokens; tier
decisions agree 0/8. Mechanism: train-token coverage is inflated
(fit-on-train), so the predictor mirrors the overfit surface - the
THIRD occurrence of the n=24 overfit lesson (G4, G9, G10). The
ladder itself behaves correctly: fed TRUE coverage it marks every
held-out token 'elevated', i.e. it refuses rank-8 compression on
gate experts - consistent with G4/G5 (the rank lever is dead here).
Needs the exp96 86-token calibration for a fair rho test.
RECALIBRATED AT n=86 (exp96, train 64 / hold 22): Spearman FLIPS to
+0.54, mean abs error 0.33 -> 0.22, tier agreement 0.0 -> 0.5. The
hidden-state KNN DOES transfer compression risk once calibrated on
64 samples — the n=16 failure was pure overfit, the recalibration
hypothesis is CONFIRMED. Predictor still overestimates coverage
(0.55-0.80 vs true 0.27-0.63); next step is a ridge/MLP rho on
hidden features instead of 1-NN, then wire the ladder thresholds.
RIDGE RHO (same split): Spearman 0.85 (KNN 0.54), MAE 0.135
(KNN 0.22), tier agreement 0.82 (KNN 0.50) — a factored linear
predictor on the hidden state transfers compression risk almost
fully at 64 calibration tokens. G10 now has a working rho(h); the
next refinement is a 3-way split (projector fit / predictor fit /
eval on disjoint token sets) to remove the train-fit inflation, and
the entropy signal from per-token logits.
3-WAY SPLIT (projector 43 / predictor 21 / eval 22, unbiased
targets): ridge rho reaches SPEARMAN 0.91, MAE 0.037, tier
agreement 1.0 (KNN 0.75/0.062/1.0). The unbiased targets are the
right design; the train-inflation accounted for most of the 2-way
error. Caveat: 21 train tokens — a 256+ token trace is the final
number, but rho(h) is now operational for the escalation ladder and
the V4-Coder extraction gate.
RANK-ROBUST (ranks 8/16/24, same 3-way split, outputs/rho_r{8,16,24}.json):
ridge Spearman 0.910/0.898/0.926, MAE 0.037/0.044/0.045 — one ridge
calibration serves any rank the ladder protects. The 2-way inflation
worsens with rank (0.85 -> 0.44 Spearman): the 3-way design is
required for honest rho at higher ranks.

### G11. VQ / product-quantized expert residuals
Missing: R_e ~= Decode(z_e) with shared cluster codebooks; hybrid
W_shared + U A_e V^T + sum_j a_{e,j} D_j. First step: PQ on one expert
cluster from G2; measure codebook collapse and rare-task loss.
MEASURED 2026-08-16 (real bytes, layer 0 expert 0 ffn_down_exps
7168x3072, outputs/pq_expert0_L0.json): 4-bit PQ frob error 0.985,
6-bit 0.971, 8-bit 0.936 — even 8-bit codebooks reconstruct only
6.4%% of the matrix energy. Discrete latent codes are a DEAD END for
expert-class matrices on this model (consistent with the flat SVD
spectra); VQ stays valuable only for already-low-rank residuals, and
we have none. G11 closed as a measured negative.
MEASURED 2026-08-16 (real layer-0 ffn_down_exps, 22M params, 8 blocks):
frob_error 0.985 @ 4-bit (0.11M params), 0.971 @ 6-bit (0.25M),
0.936 @ 8-bit (0.84M). Even full 8-bit codebooks retain only ~6% of
the matrix variance — discrete codes are NOT applicable to these expert
matrices (consistent with the flat spectra). G11 closed as a decisive
negative: PQ/VQ is not a lever for V4 expert compression.

### G12. Tiered residency with staged offloading + tail-latency budget
Partial: our lazy path streams EVERYTHING cold (no hot tier, no warm
pool, no predictor). First step: hot set = router + norms + top-N
experts (from G3), resident in the 10.15 GiB cache we measured; warm =
RAM pool; cold = NVMe; measure tail latency vs the 0.122-0.174 tok/s
resident numbers.
MEASURED (2026-08-16, scripts/v4_tier_sweep.py on the real 24-token
layer-3 routing sequence, outputs/tier_sweep_L3.json, measured costs
500ms cold / 50ms warm): ORACLE prefetch hits 0 miss rate at hot cap
6 (the next set IS exactly 6 experts) with 63 distinct resident
experts over the run; cap 4 gives p90 = 2 missing experts per step.
freq-lru (most-frequent-past predictor): ZERO hits at every cap up to
64, p90 = 4-5 missing - dense routing churns, so frequency is a dead
predictor (confirms the G3 hash-layer degeneracy on dense layers).
Tiered residency only pays with a near-perfect next-token predictor;
the hot tier's value is exactly the drafter's quality. tiering.py now
records per-step misses + a real P90 tail metric (was a 0.0 stub).

## Experimental order (from the reviews, adopted)

1. G1 (harness) -> 2. G2 (expert map) -> 3. G3 (lookahead cache) ->
4. G4 (per-expert Pareto) -> 5. G5 (conditional rank) ->
6. G6 (shared+private) -> 7. G7 (tail pruning) -> 8. G8 (context) ->
9. G9 (controller) -> 10. G10 (escalation).

## Ceiling (adopted from the reviews, stated honestly)

A 1.6T V4-class MoE will not become an always-on few-GB model via
quantization alone. The asymmetric upside is "fit the necessary function
for this token in tiny VRAM": streamed predicted experts, shared factors
+ sparse residuals, uncertainty-gated rank/precision, tiered latent
context, recoverable cold rare-capability path.

## HyperTensor innovation harvest (2026-08-16 sweep)

Concrete, code-level mechanisms found in HyperTensor that serve Phase 7.
Each: mechanism -> gap it serves -> port action.

### Tier 1 — directly transferable (code exists, gap is open)

Status 2026-08-16 (update 2): the FULL conditional toolkit is landed as
`ultratensor/conditional/` with tests (113 passed, 2 skipped):

- precision: APC entropy-gated escalation (G5/G10)
- thermal: NVML temp/power rank clamp + TPJ (G5/G12)
- basis: qspec shared-basis check + frank rank profiles (G5/G6)
- online_basis: deflated Oja from draft rejections + coverage readout (G3/G8/G10)
- sinks: MCR phases + attention-sink protection (G8)
- jury: confidence aggregation + domain routing (G9/G10)
- lookahead: working-set model + PrefetchController (G3)
- drafting: heterogeneous per-slot drafter ranks (G3/G5)
- spec_sim: Leviathan simulator with strict caps (G3)
- actweight: E_{l,e,s}(r) activation-weighted reconstruction (G4)
- shared_factor: W_shared + U A_e V^T with budget comparison (G6)
- tiering: hot/warm/cold residency + knee detection (G12)
- vq: product-quantized residuals with shared codebooks (G11)
- controller: composed per-token ServeController (G9)
- stats: bootstrap CIs, k-sweep, rank ablation, intrinsic dim,
  eviction policies (G1/G8)

Measured (hash layers, real bytes): token-bigram prefetch 81-82% hit at
exactly 6 experts (H=1). In-flight: phase-B dense-layer trace (blocks
0-3, 24 tokens) feeding v4_expert_sim (G2), v4_actweight (G4), and
v4_router_distill (G9) runners.

- **APC — Adaptive Precision Cascade** (`runtime/nn/speculative.h/.c`):
  entropy-gated INT16/FP32 escalation (threshold 0.5 bits, stats-tracked).
  This IS the review's "conditional precision". -> G5/G10. Port the
  gate logic to geodessical q2_0/q8 decode: high-entropy tokens escalate.
- **thermal_rank + TPJ** (`runtime/nn/thermal_rank.h`): NVML temp/power
  gated rank clamp (65C->rank_max, 85C->rank_min) + tokens-per-joule
  gradient bootstrap. Hardware-conditional rank for our RTX 4070 laptop
  (thermal throttling is real here). -> G5/G12.
- **qspec_test_shared_basis** (`runtime/nn/qspec_basis.h`): per-layer/slot
  shared-basis feasibility via alignment = proj_energy / svd_explained.
  Exactly "profile before projecting" for G6 dictionaries; would have
  predicted our GRC attention negative. -> G6 pre-check.
- **frank** (layer-position rank profiles, dominant_boost/decay):
  per-layer (early/mid/late) rank budgets instead of global k. -> G5.
- **ONB — online basis via Oja's rule** (`runtime/nn/online_basis.h`):
  updates per-layer PCA from spec-decode rejection residuals
  (h_target - h_draft); includes weight reprojection. Closed loop for
  any drafter we build. -> G3/G8.
- **MCR phase detection + sink protection** (`runtime/nn/mcr_compress.h`):
  activation-variance Mix/Compress/Refine phases -> per-phase rank
  budgets; sink_detect/sink_check_basis_coverage protects attention
  sinks when compressing context. -> G8 (CSA/HCA context work).
- **heterogeneous_drafters.py**: per-slot drafter rank (early slots high
  rank, late slots aggressive). -> G5/G3 when GSD is revived.
- **spec_decode_sim harness** (`hyperretro/bench/spec_decode_sim.py`,
  `scripts/gamma8_distilled.py`): distilled (MSE/KL) vs vanilla GRC
  drafter simulation at gamma=8 on Qwen 0.5B. Re-run against our
  factored kernels BEFORE touching V4. -> G3 first experiment.
- **Jury aggregation J = 1 - prod(1 - c_i)** + two-stage domain routing
  (`lib/jury_gtc_kernel.c`, `docs/jury_gtc_explanation.md`): the
  confidence formula to reuse in the G9/G10 controller (not the cache).
- **ablation_utils.py**: bootstrap CIs, per-matrix rank ablation,
  eviction-policy ablation (LRU/LFU/jury-weighted). -> G1/G8 tooling.

### Tier 2 — exists but already known / parallel work

- `runtime/nn/moe_v4.c/.h`: HyperTensor's own V4 MoE runtime; parallel
  to ours -> cross-check only, not a port.
- GRC factored format (Papers I-II): already ported into UltraTensor.
- GTC cache core (`hypersort/`): known 0% correctable at current
  density; stays G8.
- JIT SIMD (`runtime/jit/`): we have our own AVX2 kernels.

### Tier 3 — do NOT port until independently verified

- UGT universal basis, RH material, safety-sniping, COG living model:
  previously flagged by the external review; no evidence they transfer
  to V4 expert compression.

First experiment per this harvest (replaces earlier G1-first ordering
as cheapest path): port APC gate + thermal_rank into geodessical decode,
then run spec_decode_sim with our factored kernels on Qwen 0.5B to
decide whether to revive GSD on V4-Flash before spending V4-Pro time.
