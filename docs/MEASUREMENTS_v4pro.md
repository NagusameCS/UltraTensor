# V4-Pro Measurements — authoritative numbers (2026-08-14)

All numbers measured on this box (Windows 11, 32 GB RAM + 22 GB pagefile,
RTX 4070 Laptop 8 GB, D: SSD) against
`D:\hyperv4\models\pro\deepseek-ai-DeepSeek-V4-Pro-Q3_K_M-*.gguf`
(batiai, 17 shards, 698 GB). Raw artifacts: `D:\hyperv4\models\pro_factored\*`
(rank_sweep.json, route_sensitivity.txt, oracle files) and
`C:\Users\legom\hyperv4flash\bench_v4_pro.log`.

## Model facts (verified vs deepseek-ai/DeepSeek-V4-Pro + bati.cpp)

- 1.6T total / 49B active MoE, 61 layers, hidden 7168, 384 routed
  experts, **top-6**, 1 shared expert, `moe_intermediate_size` 3072.
- Routing: layers 0-2 **hash** (`ffn_gate_tid2eid`, zero GEMV, zero
  per-token IO; weights still from the dense gate); layers 3+ dense
  `ffn_gate_inp` (7168,384) F32 + `exp_probs_b` bias, **sqrt-softplus**,
  `route_scale` 2.5, `norm_topk_prob`, weight-before-down.
- `ffn_gate_exps` is the per-expert SwiGLU gate, NOT the router.
- Expert tensors: gate/up Q3_K (9.46 MB/expert), down Q5_K
  (15.1 MB/expert); 3.38 + 3.38 + 5.41 GiB per layer for the full
  384-expert stacks.

## Decisive negative results (why compression does not help V4-Pro)

| Question | Result | Method |
|---|---|---|
| Low-rank (SVD) factorization | **DEAD**: 50% energy needs rank ~654-770, 95% needs ~2300-2440 of 3072; spectra essentially flat | CUDA SVD sweep, 9 tensors x 8 experts (rank_sweep.json) |
| Expert cross-correlation | **NONE**: max off-diag corr < 1e-3 (shared subspace also dead) | same sweep + stacked-SVD |
| Q2_K requant of experts | **UNUSABLE**: 0.274 rel err (our exact port of quantize_row_q2_K_ref) vs 0.289 for llama-quantize itself on the same data | oracle mini-GGUF + llama-quantize requant |
| IQ2_XS/XXS | dead without imatrix (llama.cpp asserts quant_weights); imatrix at 0.05 tok/s impractical | code inspection |
| uq4/Q4_0 | larger than Q3_K (3.44 bpw) | arithmetic |

=> on this hardware, compression cannot bridge 698 GB vs 32 GB RAM.

## The lazy top-k serving path (Phase 3b, all real shard-1 measurements)

Per-token-layer cost (6 selected experts x 3 GEMVs + shared expert +
routing), layer 0, shard 1:

| Version | s/token-layer | projected tok/s (61 layers) |
|---|---|---|
| scalar C decode | 0.60 | 0.027 |
| + AVX2 fused decode+dot (Q3_K/Q5_K) + OpenMP rows | 0.169 | 0.097 |
| + prefetch read/decode overlap | **0.107** | **0.153** |

- llama.cpp baseline on the same box: **0.05 tok/s prefill** (25-token
  prompt = 550 s, disk-paging-bound). REAL DECODE NUMBERS (full
  overnight cycle 2026-08-14/15, llama.cpp + DSpark drafter):

  | prompt | time | tok/s | draft |
  |---|---|---|---|
  | greet | 2020.2 s | 0.01 | 6/26 |
  | math | 1368.7 s | 0.01 | 9/17 |
  | code | 1183.2 s | 0.02 | 10/14 |
  | logic | 1884.8 s | 0.01 | 8/18 |

  => llama.cpp V4-Pro decode = 0.01-0.02 tok/s; the lazy executor's
  0.153 tok/s projection is ~10-15x llama.cpp decode on this box.
- Batch-4: 0.74 s/layer (5.4 tok/s equivalent).
- Disk reads measured 1.24 GB/s scattered / 1.35 GB/s sequential.

### Complete per-token IO budget (full model, 61 layers)

| term | GiB/token |
|---|---|
| routed experts (top-6) | 13.0 |
| dense/attention/indexer/compressor/norms | 10.1 |
| shared expert | 2.2 |
| **total** | **25.3** => **0.079 tok/s at 2 GB/s** |

Attention (~166 MiB/layer) is the second read-bound; overlaps via the
same prefetch pipeline.

## Q2_K for routing only (gate compression hypothesis)

- Q2_K gates keep top-8 routes at 6.88/8 (86%), full-match 18.4%,
  score correlation 0.973 (32 experts, 256 random hidden states).
  Promising but NOT production-safe without a perplexity check.

## Container / kernel / executor verification

- All k-quant decoders (Q2_K..Q6_K, Q8_0) match llama.cpp within the
  Q4_0 requant noise floor (rel 0.055-0.08) — oracle-verified.
- Factored container round-trips; C executor C-vs-numpy < 1e-3 on real
  shard experts (Q3_K gate, Q5_K down, Q4_K down), both transpose modes.
- CUDA factored GEMV 30.8x vs two-pass fp32 (measured earlier).
- Suite 59/59 tests.

## Correctness gotchas recorded (for the next marathon)

- MSVC classic OpenMP: no `int i` in parallel-for init (C3015), no
  `#pragma omp atomic write` (C7660); **check DLL timestamps after
  builds** — silent build failures burned an hour.
- Windows `fseek` is 32-bit; use `_fseeki64` past 4 GB.
- GGUF offsets are relative to the data section; absolute =
  data_start + tensor_off (an earlier Q3_K "layout mismatch" was a
  5 MB probe offset error, not a real format difference).
- llama-quantize: `--allow-requantize` must precede positionals; mini
  GGUF inputs must drop `split.*` KVs or it errors "invalid split file
  name".

## Lazy serving end-state (2026-08-15, full 61-layer measurements)

Per-layer lazy MoE timing (real bytes, 300 tokens/layer, all layers):
mean 0.1339 s/token-layer, hash (0-2) 0.1535, dense (3-60) 0.1329,
slowest L1 0.1544, fastest L60 0.1188 -> FFN-only floor 0.122 tok/s.

Full-model projections (21.66 GiB/token; 2 GB/s disk):
| strategy                      | tok/s   |
|-------------------------------|---------|
| serial                        | 0.053   |
| pipelined                     | 0.092   |
| resident (10.15 GiB RAM cache)| 0.122   |
| resident ceiling (reads only) | 0.174   |
| batch-256 aggregate ceiling   | 0.708   |

llama.cpp on the same box: 0.01-0.02 tok/s. Lazy resident = ~6-12x;
batch-dedup serving design = ~35-70x aggregate.

Decisive negatives that bound these numbers:
- expert SVD spectra flat (k=128 -> 12% energy): per-expert factoring
  cannot shrink the 11.5 GiB/token routed read mass below Q3_K cost;
- Q2_K expert error 0.274-0.289 rel: the next-lower quant trades too
  much quality for a 1.31x read cut.

## Confidence intervals (2026-08-16, bootstrap over 61 layer means)

scripts/ci_report.py (outputs/ci_report.json). Raw per-token samples
were not saved, so per-layer CIs are analytic; the cross-layer bootstrap
is exact (20k resamples, percentile):

- projected tok/s (61 layers): 0.1229  [0.1210, 0.1246]
- dense layers (3-60):            0.1329  [0.1311, 0.1347] s/layer
- hash layers (0-2):              0.1535  [0.1521, 0.1544] s/layer
- difference (dense faster):     -0.0206 s/layer, CIs non-overlapping

The hash/dense gap is significant and was not explained by the earlier
measurements - worth one profiling run before further router work.

### Why dense > hash per layer (churn, measured 2026-08-16)

scripts/v4_churn_analysis.py: hash-layer routes change almost
completely between consecutive tokens (churn 0.98-0.99; overlap 0.06-0.12
of 6 experts), so every token streams ~200 MB of nearly-all-new experts
with no page-cache reuse. Dense top-6 over hidden-state scores should
concentrate (low churn) - to be verified when phase-B dense traces land.
This is the mechanism behind the -0.0206 s/layer hash-vs-dense gap, and
it is exactly why the drafter doubles as the hash-layer prefetcher
(81-82% working-set hit at 6 experts, H=1).

VERIFIED (exp96, 86 real dense tokens, 2026-08-16): dense layer 3
consecutive overlap 0.333 -> churn 0.667/step; 68 distinct experts
over 24 tokens (hash: 104-112). Dense concentrates but still churns.

## Phase 7 verdicts (2026-08-16, real bytes, held-out standard)

All numbers below are in the outputs/ artifacts and REVIEW_GAPS.md;
this is the condensed authoritative record.

| Gap | Verdict | Number |
|---|---|---|
| G1 routing structure | top-6 entropy PINNED at 1.7909 nats ~= ln 6; 6th/7th margin 1.003 (max 1.011) | route_stability_L3.json |
| G2 expert similarity | NEGATIVE | act cos 0.0388/0.0078 (L0/L3), weight corr ~1e-5 | expert_sim_activation.json |
| G3 prefetch/drafting | spec ceiling 0.754 tok/s (perfect drafter) == batch-256 0.708; drafter quality binds | spec_projection.json |
| G4 activation rank | CORRECTED: k95_act=8 was n=24 overfit; held-out (train 64) has NO KNEE to r=64 (L0 38-49% kept, L3 42-59%) | exp96_subspace_proj.json |
| G5 conditional rank | weak +: mass beats fixed 1-2.5% but abs error 0.84-0.94 | rank_sweep_L3.json |
| G6 shared bases | NEGATIVE: shared 0.498 vs independent 0.399 at equal 41.9M budget | shared_factor_L3.json |
| G7 CVaR pruning | tail gate vetoes worst-token-1.0 experts; 4 low-mass pass | expert_damage_L3.json |
| G8 MCR phases | L0 compress, L1-3 refine, var ratio 1.47 (4-layer datum) | mcr_phases.json |
| G9 tiny controller | factored ridge = dense EXACTLY (15x fewer params); score-reg hold rel-L1 0.086/0.269 @ 649k (4.2x smaller than router); set agreement capped by 1.003 margin | controller_shrink_86tok.json |
| G10 rho(h) | 3-way split: ridge Spearman 0.91, MAE 0.037, tier 1.0; KNN 0.75 | rho_predictor_L3.json |
| G11 PQ/VQ | NEGATIVE: 8-bit codebooks keep 6% of expert energy | pq_expert0_L0.json |
| G12 tiered residency | oracle cap 6 = 0 misses; frequency predictor 0 hits | tier_sweep_L3.json |

Cross-cutting lesson (three independent confirmations): every
in-sample headline (G4 k95, G9 1.0, G10 KNN) FAILED its held-out
split at n=24 and only some recovered at n=64-86 (G9, G10 did; G4 did
not). Held-out is the standard for this repo; in-sample numbers are
not published without it.

## V4-Coder program (2026-08-16, running)

Code-domain census (exp_code, 87-token domain-spanning battery) and
rare/adversarial tail census (exp_rare) on the cluster feed the
subnetwork-extraction decision: concentration >=0.8 top-64 mass share
and exclusivity > 0 make extraction viable; size targets keep64 =
74.5 GB IQ2_XS vs 677 GB full. See docs/ROADMAP_V4CODER.md.
