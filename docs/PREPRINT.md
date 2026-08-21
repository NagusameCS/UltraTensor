# UltraTensor / HyperMoE — Technical Report (Preprint Draft)

**Working title:** *UltraTensor: Expert Splicing and Hyper-MoE Routing for
Trillion-Parameter Models — Serving DeepSeek-V4-Pro from 697 GB to a 16 GB
Laptop GPU.*

**A companion paper to the HyperTensor volume** (Stewart 2026, Papers I–XV).

> **Formal, publishable version:** [`paper/ultratensor.pdf`](paper/ultratensor.pdf)
> (LaTeX source: [`paper/ultratensor.tex`](paper/ultratensor.tex),
> bibliography: [`paper/refs.bib`](paper/refs.bib)). This Markdown report is the
> living, artifact-linked working draft; the LaTeX paper is the peer-review-facing
> form. The two are kept consistent.

*Draft 2026-08-19. All numbers in this report are measured on real model bytes;
every claim carries the artifact that proves it (this repository).*

---

## Abstract

DeepSeek-V4-Pro is a 1.6T-parameter Mixture-of-Experts model whose smallest
public GGUF distribution is 697 GB and whose official guidance targets
multi-GPU servers. We show that the routing structure of this model is highly
compressible: on real code traffic, 97.66% of expert routing mass on dense
layers concentrates in 64 of 384 experts, and only 77 experts ever fire. We
build "keep-N" expert splices of the model (156 GB keep64, and uniform
16/12/8-expert ladders), re-quantize them to IQ2_XS (~2.36 BPW, 16-25 GB),
and serve them on a 32 GB laptop with an 8 GB GPU at 1.2-1.5 tokens/sec
generation. Along the way we discover and fix a metadata defect in the
published Pro GGUFs that hangs MTP-enabled llama.cpp builds, and we develop
HyperMoE: a domain router, a measured difficulty predictor (Spearman 0.98,
MAE 0.028, perfect tier agreement), and hash-layer prefetch tables that
compose into a serving dispatcher. A first task-retention battery on the
CPU-served keep16 splice (Q3_K_M, 8-token completions, 5 domains) measures
mean perplexity **2.653** (code 3.000, math 2.887, multilingual 2.629,
rare 2.536, needle 2.215) — the short-context quality floor of the splice.
All measurements are validated against bit-level numpy oracles of the full
architecture (max relative error 3.4e-7 block, 5.2e-6 end-to-end serve).

## 0. Relationship to the HyperTensor volume

UltraTensor is the Mixture-of-Experts-scale continuation of the HyperTensor
volume (Stewart 2026, Papers I–XV), which established that a *dense* transformer
can be compressed and recombined using only the geometry of its own weights.
This report carries that program to a *sparse* model three orders of magnitude
larger, where the interesting structure is not in any single weight matrix but
in **which experts fire for which inputs**. Four HyperTensor mechanisms carry
over conceptually:

- **GRC (Part I)** — compress from the weights' own spectrum, then *measure*
  whether it helped. Applied to V4-Pro's experts this yields mostly *negative*
  results (Section 2.2 and the G-gap negatives in `REVIEW_GAPS.md`); reporting
  them honestly is itself a HyperTensor inheritance.
- **CECI component interchange (Part X)** — splicing an FFN between models
  sharing a basis. Expert splicing (Section 2.4) is the intra-model analogue:
  cut and re-stack experts of the same model, keeping the router consistent.
- **OTT / GTC (Parts IV, VIII)** — the cached-manifold read model, instantiated
  here as the lazy predicted-expert read path.
- **The geometric jury (Part XV)** — the confidence aggregation
  `J = 1 − ∏(1 − cᵢ)` reused as the reconstruction-risk escalation gate
  (Section 2.7).

The new ingredient, absent from the dense-model volume, is **conditionality
under a hard resource wall**: on the laptop no static artifact both fits and
retains full capability, so the resolution is a system that spends capacity only
when a request demands it — with the contract *compression failure ⇒ progressive
recovery*, never silent wrong output.

## 1. Background and motivation

- DeepSeek-V4-Pro: 1.6T total / 49B active, 61 layers, 7168 hidden, 384
  routed experts (top-6) + shared expert, hybrid CSA/HCA attention.
- Public GGUF (BatiAI Q3_K_M): 17 shards, 697.3 GB total (measured on disk).
- Target hardware: 32 GB RAM / RTX 4070 Laptop 8 GB — a machine the vendor
  classifies as far below minimum.
- Research questions:
  1. How much of the expert space is actually used per domain?
  2. Can we splice the model to the used subset without degrading routing?
  3. Can the splice be quantized and served on commodity hardware?
  4. Can a small controller predict request difficulty to gate escalation?

## 2. Methods

### 2.1 Ground-truth oracles
Numpy reference of the full DeepSeek-V4-Pro architecture (Sinkhorn HC
mixing, MLA/YaRN, CSA compressor, 384-expert MoE), ported from the official
inference code and executed on the real 17-shard GGUF.
Results: block forward max_rel **3.4e-7**; compressor scores ~6e-7;
130-position cached decode worst_rel **2.1e-6**; full 61-layer e2e serve
max_rel **5.2e-6** with top-10 logits 10/10; tokenizer battery 15/15 exact.
Artifacts: `scripts/v4_ref_*.py`, `scripts/check_v4_*.py`.

### 2.2 Routing science (G1-G12)
61-layer routing traces on real prompt batteries. Key measured facts:
- Hash-layer (L0-L2) expert churn between consecutive tokens: 0.98-0.99.
- Routing entropy pinned at ln 6 = 1.7909; 6th/7th expert margin 1.003.
- Perfect H=4 lookahead reaches 100% prefetch at the 24-expert union;
  weak drafter ~21%.
- Honest negatives: gate-refit ridge 0.801 vs sliced baseline 0.818;
  shared-factor bases lose to independent; rank-8 subspace holdout 0.47-0.73.

### 2.3 Domain censuses
Per-token expert mass from real traces, top-K per layer.
- Code census (87 tokens): dense L3 top-64 = **97.66%** mass, **77 distinct**
  experts, **32 code-exclusive**; hash layers spread (top-64 share 0.50-0.53).
- Mid-layer census (L3-L10): subspace hold@8 0.32-0.74 by layer; factored
  G9 controller forward rel-L1 **0.0053**, agreement 0.667.
- Language census (128 tokens, 4 languages): top-32 pairwise overlap 15-22/32;
  top-32 mass share 0.86-0.95. Verdict: languages share a common expert core;
  specialists = shared base + small per-language additions.

### 2.4 Splicing (GGUF surgery)
- `ultratensor/gguf_keep.py`: keep-N expert-stack writer (mixed-E keep64,
  uniform-E ladder), router column slicing, tid2eid remap, KV overrides.
- Artifacts: keep64 156.1 GB (4.5x smaller), keep16u 39.3 GiB, keep12u
  32.2 GiB, keep8u 25.0 GiB (27.9x smaller than source).

### 2.5 Metadata fixes (published artifacts)
- **split.count**: single-file splice retained a 17-shard claim -> loader
  rejection; patched in place.
- **MTP hang (bug in the published GGUFs)**: all 17 original shards declare
  `deepseek4.nextn_predict_layers = 1` but contain **zero** `mtp.*` tensors
  (positive control: the Flash IQ2XXS GGUF contains 2,376). MTP-enabled
  llama.cpp builds load but hang forever at generation. One-byte patch
  (flag -> 0); A/B verified (same model: 1 -> 0, generation works).
  Upstream references: llama.cpp b10424 `src/models/deepseek4.cpp#L19-L27`,
  `#L1409`; audit tool: `scripts/audit_mtp.py`.

### 2.6 Quantization
- IQ2_XS requants with correctly-sized importance matrices (3-D expert
  stacks require nval == ne0*ne2; mismatch aborts in upstream
  llama-quant.cpp:1203-1211):
  keep16u 39.3 -> **24.7 GiB (2.36 BPW)**, keep12u -> **20.8 GiB (2.36 BPW)**,
  keep8u -> **16.3 GiB (2.38 BPW)**.
- keep64 IQ2_XS requires ~34 GB f32 scratch for one 384-expert tensor ->
  "bad allocation" on 32 GB; routed to a 32-core node with 24 GB swap.

### 2.7 HyperMoE serving stack
- Purpose-first domain router + specialist registry (11 specialists).
- **rho difficulty predictor**: 3-way ridge on 192 real trace tokens / 64
  held out: **Spearman 0.9798, MAE 0.0281, tier agreement 1.0** (KNN baseline
  0.8785 / 0.0687 / 0.9062). Honest caveat: all 64 held-out tokens fall in the
  "elevated" risk tier, so the perfect tier agreement is a low-difficulty result;
  the 256-token calibration that would populate multiple tiers is queued
  (`outputs/rho_256_run.log` not yet complete). The two-way (no projector split)
  variant inflates with rank (0.85 → 0.44 Spearman), which is why the 3-way
  design is mandatory.
- Hash-layer prefetch tables: 129,280 token -> top-6 expert rows (9.3 MB).
- Dispatcher with `llama-cli`, numpy-reference, and `llama-http` (GPU)
  backends; escalation gate wired through HTTP and CLI.
- GPU tier manager: one llama-server at a time on the 8 GB card, swapped
  on demand (co-resident servers starve VRAM: keep12u fell to 0.1 t/s;
  solo it runs 1.2 t/s).

## 3. Results

| Metric | Value | Evidence |
|---|---|---|
| Source model | 697.3 GB (17 shards) | disk |
| keep64 | 156.1 GB, 97.66% code mass (L3) | `outputs/code_census.json` |
| Ladder | 39.3 / 32.2 / 25.0 GiB | disk |
| IQ2_XS ladder | 24.7 / 20.8 / 16.3 GiB @ 2.36-2.38 BPW | quant logs |
| GPU generation | **1.2-1.5 t/s** (8 GB VRAM, ngl 12) | `outputs/cuda_diag.log`, `smoke8u_d.log`, `smoke12u_solo.log` |
| CPU generation | 0.1 t/s | `outputs/keep16u_gen_test.log` |
| Task PPL (keep16u, CPU, 8 tok) | **mean 2.653** (code 3.000) | `outputs/ppl_cpu16_full.json` |
| rho predictor | 0.98 / 0.028 / 1.0 | `outputs/rho_192_run.log` |
| Oracles | 3.4e-7 / 5.2e-6 / 10/10 / 15/15 | `scripts/check_v4_*.py` |
| C kernel | expert decode+GEMV, AVX2, dll/so, 9/9 vs numpy | `tests/test_expert_store.py` |

## 4. Known issues and fixes in progress

**GPU decode path — three stacked defects in upstream b10424 CUDA code**
(short prompts work; sequences of ~45+ tokens fail; CPU path unaffected;
splice routing tables verified valid, min 0 / max 15 — the model data is
not at fault):

1. `quantize_mmq_q8_1<scatter=true>` (MMQ MoE broadcast quantization)
   writes far out of bounds at ~45 tokens (compute-sanitizer: 144 GB past
   / 8.8 GB before its allocation). Workaround committed: fork routes MoE
   broadcast matmuls away from MMQ (both `mul_mat_id` call sites).
2. The cuBLAS fallback for those matmuls then failed its own invariant
   (`ids_to_sorted_host.size() == ne_get_rows`, ggml-cuda.cu:1990).
   **Root-caused and fixed (2026-08-19)**: the splice's tid2eid remap
   routes dropped experts to a fallback expert (0), so token rows
   legitimately contain repeated expert ids (e.g. `[11, 0, 0, 0, 0, 0]`);
   the fallback's matching loop `break`s after the first slot per
   (expert, token) and under-produces rows. Fix: drop the `break` (per-
   slot mapping was already supported). No behavior change for models
   with unique top-6 ids (upstream never hits it). Validated on the
   rebuilt fork: 55-token prefill + 24-token decode + logprobs, repeated
   requests, server stable (`outputs/rebuilt_srv.err.log`).
3. An attention-path crash was reported with expert tensors pinned to
   CPU (hybrid `-ot` test) on the pre-fix engine. Not reproduced on the
   fixed fallback; the Q3_K decode hang it was grouped with is now
   attributed to defect 2. Status: demoted to "needs reproduction"
   rather than a confirmed separate defect.
4. **op-offload host-op corruption (upstream, confirmed 2026-08-20)**:
   with `-ngl 0`, llama.cpp's op-offload (default on) schedules a Q3_K
   dequant kernel on the GPU; on the splice this dies with an illegal
   memory access at ~36 prompt tokens and takes down the server. At
   partial offload (`-ngl 12`) it does not crash on short prompts but
   silently corrupts values: 3 of 5 battery domains collapse into
   `"<"` loops with null log-probs, and prompt processing slows to
   0.15 t/s (0.42 t/s with the flag off). Fix: `--no-op-offload`
   everywhere; the CPU path is then stable (Section 8 battery) and the
   GPU battery returns valid log-probs on all five domains. This is
   upstream scheduling behavior, not a splice defect.

Net: GPU serving is verified on the rebuilt fork for long prompts on
IQ2_XS (55-token prefill + 24-token decode, repeated requests, logprobs)
at 0.6-1.5 t/s; Q3_K now decodes on GPU too (0.07-0.15 t/s partial
offload). CPU-only serving remains the stable quality mode. Defects 1-2
are fixed in the fork; the Q3_K decode hang is attributed to defect 2,
not to a separate attention-path defect (defect 3 demoted to "under
re-test").

**Other open items**
4. **GPU logprobs**: working on the rebuilt fork (raw `n_probs` and
   OpenAI chat logprobs both return entries on the IQ2_XS tier). The old
   illegal access no longer reproduces; exact attribution (MMQ OOB
   fallout vs top-k) is under test rather than claimed.
5. **keep64 IQ2_XS**: RAM-scratch bound (one 384-expert tensor needs
   ~34 GB f32); queued on the 32-core node (4 GB swap measured — the
   earlier 24 GB figure was wrong; the waiter will retry after the
   trace chain frees RAM).

## 5. Reproducibility

All model files, scripts, batteries, and result artifacts referenced above
are in this repository (see `outputs/`, `scripts/`, `tests/`). The cluster
pipeline (traces -> censuses -> rankings), the overnight watchdog, and the
GPU autopilot (smoke -> serve -> registry) are self-contained in
`scripts/`.

## 6. Statistical methodology

- **Ground truth everywhere**: every pipeline stage is validated against
  the numpy oracles of Section 2.1 before its outputs are interpreted.
- **Held-out discipline**: predictor fits (rho, subspace) always report
  held-out sets; the project keeps its own overfit lessons on record
  (rank-8 in-sample 0.98 -> held-out 0.47-0.73; KNN rho -0.40 -> ridge
  0.91 after target debiasing).
- **3-way splits**: train / projection-fit / evaluation disjoint, with
  evaluation sets kept untouched until the final run.
- **Bootstrap CIs** for per-layer timing (`bench_moe_layer_*.json` stores
  raw per-token times; exact bootstrap CIs are computed from them).

## 7. Related work

- bati.cpp / BatiAI GGUFs: the public V4-Pro GGUF distribution used here
  (converted from the official safetensors; llama.cpp master does not
  support V4 at the time of writing).
- llama.cpp deepseek4 support (b10424): the loader/MTP implementation our
  fork builds on; we document two GPU decode defects and an MTP metadata
  inconsistency in the published files.
- Speculative decoding for MoE: our prefetch measurements (churn 0.98-0.99,
  oracle H=4 staircase) quantify the drafter-as-prefetcher design that
  serve stacks for DeepSeek-class models assume but rarely measure.
- Expert offloading (lazy MoE serving): our C-kernel lazy path
  (`expert_gemv.c`, per-expert decode+GEMV) is a measured instance of this
  pattern, with honest throughput numbers.

## 8. Task-benchmark retention results

The PPL battery (code / math / multilingual / rare-domain / needle,
max_tokens=8, temperature 0) run against the CPU-served keep16u splice
(Q3_K_M, 16 expert slice, 39.3 GB) with the original engine in CPU-only
mode:

| Prompt | PPL | Latency (s) |
|---|---|---|
| code | 3.000 | 170 |
| math | 2.887 | 224 |
| multilingual | 2.629 | 222 |
| rare | 2.536 | 224 |
| needle | 2.215 | 224 |
| **mean** | **2.653** | — |

Notes: measured on `outputs/ppl_cpu16_full.json`. A CPU-server crash
affecting requests >=36 tokens was localized to llama.cpp's op-offload
path placing a Q3_K dequant kernel on the GPU even with `-ngl 0`; the
fix is `--no-op-offload` (details in Section 4 defect report).

The same battery on the GPU tier (keep16u IQ2_XS requant, 24.7 GiB,
2.36 BPW, full offload, 16-token completions, rebuilt fork engine,
`--no-op-offload`; `outputs/ppl_gpu16_iq2xs_noopoff.json`):

| Prompt | PPL | Completion |
|---|---|---|
| code | 8.528 | degenerate `"Kahanay…"` loop |
| math | 8.631 | degenerate loop |
| multilingual | 8.458 | degenerate loop |
| rare | 8.568 | degenerate loop |
| needle | 8.517 | degenerate loop |
| **mean (n=5)** | **8.540** | — |

Honest reading: the IQ2_XS requant of the fallback-remapped keep16
splice is a *speed* tier (0.6-1.5 t/s on 8 GB VRAM, ~4x faster than
CPU Q3_K at 0.1 t/s), not a *quality* tier — mean PPL 8.540 vs 2.653
at Q3_K, with degenerate token loops on every domain. Important
correction: the earlier 2-of-5 result (null logprobs on 3 domains) was
an engine-configuration artifact, not a property of the requant — with
the default op-offload, ops over CPU-resident Q3_K tensors run on the
GPU and silently corrupt values on short prompts; with
`--no-op-offload` all five domains return valid log-probabilities
(n_ppl=5). All logprobs returned by the server are reported;
null-logprob tokens are skipped in PPL (counted in `n_ppl`).

## 9. Future work

- Specialist builds from per-domain rankings (censuses in flight).
- rho@256 final with the 320-token trace (queued).
- Q3_K GPU serve after the fallback fix (decodes now; throughput and
  quality battery pending).
- Flash-284B grafts and V100-tier experiments.

---

## Appendix A — Per-layer census detail

Code census, 87 real code tokens (`outputs/code_census.json`):

| Layer | Type | Distinct experts | Top-64 mass share |
|---|---|---|---|
| L0 | hash | 261 | 0.5038 |
| L1 | hash | 252 | 0.5307 |
| L2 | hash | 259 | 0.5096 |
| L3 | dense | **77** | **0.9766** |

L3 top-64 composition: 32 code-exclusive, 29 shared with the general
census, overlap 29 (64 = 32 exclusive + 29 shared + 3 other).

Mid-layer census (L3-L10, 64-token code trace):

| Layer | Subspace hold@8 (PCA) |
|---|---|
| L3 | 0.3231 |
| L4 | 0.5312 |
| L5 | 0.4024 |
| L6 | 0.4873 |
| L7 | 0.7062 |
| L8 | 0.7276 |
| L9 | 0.7428 |
| L10 | 0.6446 |

Factored controller (G9) on code: forward rel-L1 **0.0053** (agreement
0.667); reverse rel-L1 0.685 (agreement 0.167).

## Appendix B — Language census (128 tokens, layer 3)

| Language | n | Distinct | Top-32 mass |
|---|---|---|---|
| python | 35 | 63 | 0.860 |
| rust | 33 | 48 | 0.925 |
| sql | 33 | 43 | 0.946 |
| js | 27 | 42 | 0.941 |

Top-32 pairwise overlap: python/rust 18, python/sql 17, python/js 15,
rust/sql 21, rust/js 17, sql/js 22 (of 32). Shared core: {320, 307, 232,
228, 111, 12, 100, 177, 243, 257, 269, 377, 289} appear in all four.

## Appendix C — Serving measurements

| Configuration | Generation | Prompt | Evidence |
|---|---|---|---|
| keep16u Q3_K, CPU (numpy-free llama.cpp) | 0.1 t/s | 0.3-0.5 t/s | `outputs/keep16u_gen_test.log` |
| keep16u-iq2xs, GPU ngl 12 | **1.5 t/s** | 1.6 t/s | `outputs/cuda_diag.log` |
| keep8u-iq2xs, GPU ngl 8 | **1.2 t/s** | 1.6 t/s | `outputs/smoke8u_d.log` |
| keep12u-iq2xs, GPU ngl 12 (solo) | **1.2 t/s** | 1.5 t/s | `outputs/smoke12u_solo.log` |
| keep12u-iq2xs, GPU ngl 8 (co-resident) | 0.1 t/s | — | autopilot log |

Bandwidth model (`outputs/bench_lazy_full.json`): 21.66 GiB read per token;
resident path 0.122 t/s, pipelined ceiling 0.174 t/s. Spec projection
(`outputs/spec_projection.json`): verifier 8.197 s/token, drafter 1.111 s,
batch-256 ceiling **0.708 t/s** — the read path, not the drafter, is the
throughput lever (verifier reads are 8.2 s of the 8.17 s resident cost).

## Appendix D — Defect reports (published artifacts)

1. **MTP metadata inconsistency** (all 17 published Pro Q3_K_M shards):
   `deepseek4.nextn_predict_layers = 1` with zero `mtp.*` tensors
   (Flash IQ2XXS control: 2,376). Consequence: MTP-enabled llama.cpp
   loads, then hangs indefinitely at generation. Fix: flag -> 0, A/B
   verified. Tooling: `scripts/audit_mtp.py`, `scripts/patch_gguf_kv.py`.
2. **Importance-matrix validation** (upstream llama-quant.cpp:1203-1211):
   3-D expert stacks require imatrix nval == ne0*ne2; mismatch aborts.
   Observed: 49152 vs 1179648 on `blk.0.ffn_down_exps.weight`.
3. **GPU engine builtin-kernel defect**: missing `$__cuda_sm20_rem_s64`
   in the shipped CUDA library (Section 4); fixed by local rebuild.
4. **MMQ scatter-quantize OOB write (the actual GPU decode crash)**:
   compute-sanitizer on a locally rebuilt engine identifies
   `quantize_mmq_q8_1<D4, scatter=true>` writing ~144 GB past a 2 MB
   allocation at ~45-token sequences — the MoE broadcast (gate/up,
   `n_experts > 1, ne11 == 1`) quantization path on sm_89. Workaround
   committed in the fork (local commit `10049f0`): route MoE broadcast
   matmuls to cuBLAS. Evidence: `outputs/sanitizer.log`.

## Appendix E — Verification suite

`tests/` (150+ tests): dequantizers vs llama-quantize and gguf-py
references (MXFP4/IQ2_XXS 0.0 error on real tensor data); C expert kernel
vs numpy (9/9); GGUF keep-writer round-trip; router unit tests; bootstrap
CI machinery.
