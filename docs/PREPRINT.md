# UltraTensor / HyperMoE — Technical Report (Preprint Draft)

**Working title:** *Expert Splicing and Hyper-MoE Routing for DeepSeek-V4-Class
Models: From 697 GB to a 16 GB Laptop GPU*

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
compose into a serving dispatcher. All measurements are validated against
bit-level numpy oracles of the full architecture (max relative error
3.4e-7 block, 5.2e-6 end-to-end serve).

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
  held out: **Spearman 0.9798, MAE 0.0281, tier agreement 1.0**.
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
| rho predictor | 0.98 / 0.028 / 1.0 | `outputs/rho_192_run.log` |
| Oracles | 3.4e-7 / 5.2e-6 / 10/10 / 15/15 | `scripts/check_v4_*.py` |
| C kernel | expert decode+GEMV, AVX2, dll/so, 9/9 vs numpy | `tests/test_expert_store.py` |

## 4. Known issues and fixes in progress

1. **GPU decode crash on longer sequences** (both IQ2_XS and Q3_K splices):
   diagnosed via compute-sanitizer as a missing compiler-builtin kernel
   (`$__cuda_sm20_rem_s64`, 64-bit modulo) in the prebuilt engine's CUDA
   library — a build artifact, not a model bug. Fix in progress: local
   rebuild with CUDA 13.2 / arch 89 (toolchain now on-machine).
2. **GPU logprobs crash**: same root family (length-dependent decode path),
   reproduced and characterized; plain completions unaffected.
3. **keep64 IQ2_XS**: RAM-scratch bound; queued on the cluster node with
   24 GB swap after the trace chain completes.

## 5. Reproducibility

All model files, scripts, batteries, and result artifacts referenced above
are in this repository (see `outputs/`, `scripts/`, `tests/`). The cluster
pipeline (traces -> censuses -> rankings), the overnight watchdog, and the
GPU autopilot (smoke -> serve -> registry) are self-contained in
`scripts/`.

## 6. Future work

- Specialist builds from per-domain rankings (censuses in flight).
- rho@256 final with the 320-token trace (queued).
- Task-benchmark retention study of the splices (PPL battery in flight on
  the CPU server; GPU logprobs path pending the engine rebuild).
- Flash-284B grafts and V100-tier experiments.
