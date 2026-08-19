# UltraTensor → llama: E2E Production Path for HyperTensor on 1.2T-Class Models

> PUBLISHED 2026-08-15: Phases 1-4 complete and validated on the real
> DeepSeek-V4-Pro Q3_K_M GGUF (17 shards / 697 GB). Phase 5 (the
> llama.cpp fork) continues in its own repository.

Status: design v1 (2026-08-14). Grounded in the actual code in the sibling
HyperTensor repo (`C:\Users\legom\OneDrive\Documents\GitHub\HyperTensor`) and
this repo.

## Where we actually stand (verified inventory)

| Piece | Exists | File |
|---|---|---|
| Compression (streaming, 100B+/1.6T on 32 GB RAM) | ✅ | `ultratensor/stream.py`, `ultratensor/grc.py` |
| Factored on-disk format + manifest + certificate | ✅ | `hyperretro/hf/factored.py`, `hyperretro/certificates.py` |
| Functional loader of factored weights (HF transformers) | ✅ | `hyperretro/hf/factored.py` (`FactoredLinear`) |
| Independent inference runtime (C) | ✅ | `host/main.c` (`geodessical`), `runtime/nn/llm.c`, JIT `runtime/jit/*`, CUDA `runtime/nn/cuda_kernels.cu` |
| GGUF reader in the C runtime | ✅ | `runtime/nn/gguf.c` (v2/v3, llama.cpp-compatible enum) |
| GGUF export of compressed models | ⚠️ dense-materialized only | `hyperretro/hf/gguf_export.py` |
| vLLM drafter wiring | ⚠️ stubbed, not wired | `hyperretro/vllm_adapter.py`, `hyperretro/vllm/draft.py` |
| OpenAI-compatible serving | ⚠️ custom `/v1/*` schema only | `host/api_server.c` |
| Runtimes that can EXECUTE factored/uq4 tensors | ❌ none | — |

**The gap is not compression. It is execution + container:**
- `ultratensor compress` / `grc` emit factored safetensors + manifests that no
  runtime can execute. GGUF export materializes to dense, discarding the win.
- The C runtime executes only dense quant types (Q4_0/Q8_0/F16/F32); it has
  no factored-tensor op, and llama.cpp has none either.
- Therefore "connect it into llama" = define a factored GGUF extension and
  implement the decode kernels. UltraTensor is the natural connecting point:
  it owns the streaming encoder and can own the container writer.

## The path

### Phase 1 — Container: factored GGUF from UltraTensor
- New GGML tensor types (proposal): `GRC_FACTORED_UQ4` (per-tensor `U_k [rows,k]`
  fp16 + `C` codes [k,cols] uq4 + sink rows) and `FACTORED_QKV` (shared basis
  `A [k,d_in]` + `Bq/Bk/Bv [d_out,k]`), plus manifest KV metadata mirroring
  `hyperretro_factored.json` (rank, keys, biases, certificate fields).
- Writer: `ultratensor/export_factored_gguf.py` — stream dequant → factor →
  emit GGUF **without materializing dense**; header-only multi-shard mode
  (same trick as `tensor_inventory`, works on 17-shard 697 GB inputs).
- Acceptance: `export → load → reconstruct == numpy` round-trip test suite;
  `llama-gguf-dump`-style inspection shows the new types; byte-stable output.

### Phase 2 — Kernels (prove speed on the C runtime first, we own it)
- CPU: fused factored GEMV/GEMV-pair in `runtime/jit/x86_jit.c`
  (AVX2/AVX-512): stage-1 `C^T·x` into k-dim accumulator, stage-2 `U_k·acc`
  — one pass, no intermediate buffer in DRAM. Precedent: `gemv_dual_q8_0`
  already shows ~2.3× over two Q8 GEMVs.
- CUDA: same fused op in `runtime/nn/cuda_kernels.cu`; MoE variant
  (fused over the 384-expert batch — the V4 mass lives here).
- Acceptance: kernel benchmarks vs dense Q2_K/Q3_K on the 4070, published in
  the existing benchmark harness.

### Phase 3 — Runtime wiring (geodessical executes factored GGUF)
- `runtime/nn/gguf.c`: parse new types + manifest KV.
- `runtime/nn/llm.c`: dispatch factored tensors to fused kernels; per-layer
  VRAM-budgeted placement already exists (`Layer %d: VRAM tight → CPU offload`).
- Attention: GRC-factored MLA path (attention is already factored — GRC caps
  at ~10–15% there; effort goes to MoE experts instead).
- Speculative: wire HyperRetro drafter into the existing
  `llm_speculative_verify_*` API.
- Acceptance: end-to-end decode of a compressed model with
  certificate-bounded quality (trust tiers already defined in
  `hyperretro/certificates.py`).

### Phase 4 — Production serving (OpenAI parity)
- Extend `host/api_server.c` `/v1/chat/completions` to OpenAI schema
  (streaming SSE, `finish_reason`, usage) so SillyTavern/any client works
  unchanged — this is the "prod viable" bar.
- Paged attention (`runtime/nn/paged_attn.c`) for multi-request; keep the
  custom `/v1/generate` for the native path.

### Phase 5 — Connect into llama itself
- Port the factored GGUF extension + kernels as a llama.cpp fork (GGML op
  `mul_mat_factored`, quant registration), validated against llama's test
  suite; then propose upstream. The C runtime remains the proving ground so
  the fork lands with benchmark evidence, not speculation.

### Phase 6 — The 1.2T certification run
- UltraTensor streaming pipeline over V4-Pro Q3_K_M (17 shards): factor +
  certificate each tensor → factored GGUF shards.
- Gate: trust-tier thresholds; then geodessical serve + benchmark vs the
  llama.cpp Q3_K_M baseline currently being established on :8774.

## Non-goals (avoid wasting effort)
- vLLM monkey-patching: the adapter is stubbed for a reason — vLLM's
  draft-runner API is not stable. Serve via geodessical/llama instead.
- MLIR backend: staged; CPU JIT + CUDA cover the two real targets.
- Dense-materialized GGUF export: only as a compatibility fallback, never the
  main path.

## Current short-term status (why this is parallel work)
V4-Pro Q3_K_M serving is still fighting a known llama.cpp MSVC draft-pairing
bug (`invalid vector subscript`, upstream #26913/#24795); retry launcher is
cycling. Phase 1 can start immediately — it does not depend on the server.

## Progress log
- **V4-Pro live measurements (2026-08-14):** llama.cpp Q3_K_M on this box:
  prefill 0.05 tok/s (25-token prompt = 550 s, disk-paging-bound; 697 GB model
  vs 32 GB RAM). Decode projections with the factored path measured on REAL
  expert shapes (m=3072, n=7168, E=8 routed, rank k=128, uq4 codes):
  CPU (scalar C twin) = 8.2 ms/MoE-tensor → 1.49 s/token = **0.67 tok/s**
  (~2 tok/s with the DSpark drafter) at 1.99 GB/token disk traffic —
  ~13-40x over the llama.cpp baseline and Flash-comparable, at ~95 GB model
  size. GPU: PCIe-bound ~0.17 s/token ≈ 6 tok/s (≈15 with drafter). Compute
  is negligible either way (3.8 GFLOP/token factored vs 64.5 GFLOP dense).
  Rank-vs-error curves on real experts pending (SVD sweep).
- **Phase 1 DONE (2026-08-14):** `ultratensor/gguf_factored.py` — factored
  container (U fp16 basis + uq4 codes, custom GGML type 2048, manifest KV,
  streaming writer + reader + `reconstruct`). 4 tests, round-trip verified.
  Caught: GGUF ne0-first dims vs numpy row-major; uq4 noise floor
  (rel ≈ 0.095 on low-rank structured data).
- **Phase 2 first kernel DONE (2026-08-14):**
  `ultratensor/kernels/factored_gemv.cu` — fused `y = U @ (C @ x)` (no
  intermediate in DRAM), standalone nvcc build (sm_89, `--cudart static`),
  ctypes loader, numpy reference. Benchmarked on the RTX 4070 Laptop
  (4096×32×12288): **0.052 ms vs 1.612 ms two-pass fp32 = 30.8×**.
  2 tests; suite green (42/42). Next: MoE-batched variant + AVX2 CPU twin,
  then drop the same entry point into `runtime/nn/cuda_kernels.cu`.
- **Phase 2 complete (2026-08-14):**
  `factored_gemv_moe.cu` — MoE-batched fused GEMV (E×k grid, atomic
  accumulation, one launch per routed token's experts).
  `factored_gemv_ref.c` — portable C CPU twin with the identical entry-point
  shape (`factored_gemv_uq4_cpu`), MSVC-built DLL, ctypes loader, self-test
  + pytest. Suite green (45/45). Phase 2 = DONE.
- **Phase 3 first connector (2026-08-14):**
  Container now supports 3-D expert stacks (gguf `(n,m,E)` -> U `(E,m,k)` +
  uq4 codes); `factored_gemv_from_gguf()` wires container -> fused
  CUDA/MoE kernel with automatic dispatch. E2E test green; suite 47/47.
  Remaining Phase 3: `geodessical` gguf.c parse + llm.c dispatch using the
  three kernel entry points.
- **Q3_K layout resolved + oracle (2026-08-14):**
  V4-Pro shard-1 `blk.0.ffn_gate_exps.weight` is STANDARD b10424 Q3_K
  (110 B/block: hmask[32] qs[64] scales[12] d fp16-last). The lone bug was
  `_q3_k_scales` b1 unpack (missing `>> 2`). Method: built a mini-GGUF with
  8 real experts (drop `split.*` KVs so llama-quantize accepts it), let
  llama.cpp requantize Q3_K -> Q4_0 as the ground-truth oracle, compared our
  decode: 0 NaN, rel err 0.058 = Q4_0's own requant noise floor. Also fixed
  `export_factored_v4.py`: finalize now applies `--limit-experts`/`--only`
  (EOFError) and writes payloads in container order (all U, all scales, all
  packed — was per-expert interleaved, reader sliced mid-array). Committed
  bde48b7; suite 49/49.
- **DECISIVE measurement — MoE experts have no low-rank structure
  (2026-08-14):** CUDA SVD sweep on real shard-1 gate experts
  (m=3072, n=7168, E=4):
  * energy sweep: 50% -> rank 654, 80% -> 1444, 90% -> 1927,
    95% -> **2303** (75% of rows), 99% -> 2819. S[1024]/S[0] = 0.56 —
    the spectrum is essentially flat (weights are near-Gaussian noise).
  * experts pairwise uncorrelated: |cos| < 1e-3; shared-subspace SVD of a
    stacked 4-expert matrix needs rank 2016 for 80% energy — no better.
  => 2D/3D SVD factorization of V4-Pro experts is mathematically dead: any
  useful rank costs MORE storage than dense Q3_K (3.44 bpw). The earlier
  rank-128 projection (95 GB, 0.67 tok/s) is retracted — rank 128 keeps
  ~2% of energy (rel err 0.98, measured on the converter output).
  **Pivot for experts:** the win is IO, not math:
  1. *Quantization:* llama-quantize shards Q3_K -> Q2_K (2.56 bpw) =
     -25% paging traffic, -25% latency, no new runtime needed. Overnight
     job, `--allow-requantize --keep-split`, output `pro_q2k/`.
  2. *Lazy top-k expert loading:* per token only gate (full) + 8 routed
     experts/layer are touched (~10 GB/token vs ~40 GB observed) — needs a
     dispatch-aware loader; llama.cpp has none. This is the Phase-3b
     executor work (the MoE fused kernel from Phase 2 becomes the compute
     side of it).
  3. Factored path stays for tensors that DO have structure: attention /
     shared / sink rows (GRC). Re-scope Phase 1-2 acceptance tests to
     structured tensors, keep expert path as uq4/Q2 storage only.
- **Q2_K ruled out for V4-Pro experts (2026-08-14):**
  Oracle test on 8 real gate experts (llama-quantize requant as ground
  truth): llama.cpp's own Q2_K gives rel 0.289 vs the Q3_K source; our
  new exact port of `quantize_row_q2_K_ref` (committed 36339cb) gives
  0.274. Noise-like expert weights have no structure for Q2_K to exploit,
  and 27%+ per-weight error in the FFN is not usable. => the planned
  overnight Q3_K -> Q2_K requantize of all 17 shards is CANCELED on
  quality grounds (it would only buy 25% IO for a destroyed model).
  Exhaustive verdict for expert compression on this box:
  * SVD/low-rank: dead (flat spectra, rank 2303/3072 for 95% energy).
  * Q2_K: dead (inherent 27-29% error on this data).
  * IQ2_XS/XXS: dead without imatrix (llama.cpp asserts quant_weights),
    and generating an imatrix at 0.05 tok/s is impractical.
  * uq4/Q4_0: larger than Q3_K (3.44 bpw) — no win.
  => Compression cannot make 697 GB fit the 32 GB-RAM/IO budget of this
  box; the only paths to "usable" V4-Pro are (a) more RAM/faster disk,
  (b) the lazy top-k expert executor (~10 GB/token floor -> ~0.2 tok/s
  at 2 GB/s SSD, still marginal), (c) accept the llama.cpp baseline
  (0.05 tok/s prefill, benchmark in progress on :8774).
  Q2_K quantizer + streaming exporter + all k-quant decoders are kept:
  they are correct llama-compatible tooling for structured weights and
  the next architecture.
- **Overnight plan (2026-08-14):** (1) V4-Pro benchmark on :8774 keeps
  running (V4ProLauncher task) — first full decode numbers expected
  overnight. (2) rank_sweep.py: 64-expert (4 layers x gate/down/up x 16)
  CUDA spectrum + cross-correlation profile -> pro_factored/rank_sweep.json
  for the certificate math. (3) Repo is committed and green (49/49).
- **Phase 3b started — ExpertStore + the IO model (2026-08-14):**
  `ultratensor/expert_store.py` (committed): header-only expert inventory,
  `read_expert()` (one expert, bounded RAM, Q8_0/Q2_K..Q6_K),
  `route_layer()` (streams the gate), `io_model()` (bytes/token).
  MEASURED on V4-Pro shard 1 (Q3_K gates, Q5_K down):
  * gate pass = 3.38 GiB/layer (384 experts x 9.46 MB) — the routing floor
  * routed down/up = 0.60 GiB/token (16 experts)
  * per-token floor: batch 1 = 10.8 GiB, batch 4 = 3.2 GiB,
    batch 16 = 1.3 GiB; gates at Q2_K (router-only, argmax-robust
    hypothesis) = 8.4 GiB at batch 1.
  => the gate read, not the routed experts, dominates per-token IO.
  Batching amortizes it (B concurrent requests share one gate pass); that
  is the serving lever, quantified. 5 new tests (incl. real-shard checks);
  suite 54/54.
  Next: routing-sensitivity measurement (does Q2_K gate keep top-k
  routes?) to validate the gate_q2k lever; then the C executor
  counterpart (ut_expert_gemv with per-expert decode) in Phase 3.
- **Phase 3b continued — C executor + routing sensitivity (2026-08-14):**
  * `ultratensor/kernels/expert_gemv.c` (committed): `ut_expert_open/gemv/
    close` — opens a shard, locates a 3-D expert tensor, computes
    W_e @ x / W_e^T @ x with 16-row streaming decode. All k-quant
    decoders (Q2_K/Q3_K/Q4_K/Q5_K/Q6_K + Q8_0), portable fp16, and
    Windows `_fseeki64` (32-bit fseek overflows at 4 GB — caught by
    rc=-3 on the real shard). Self-test + ctypes `ExpertGEMV` binding.
    Validated against numpy on real shard 1: Q3_K gate, Q5_K down,
    Q4_K blk.3 down — max rel < 1e-3 both transpose modes. Suite 57/57.
  * Routing sensitivity measured (32 real gate experts, 256 random
    hidden states): Q2_K gates keep top-8 routes at **6.88/8 (86%)**,
    full-match 18.4%, gate-score correlation **0.973**. Verdict:
    gate_q2k is promising for router-only use (saves 25% of the
    3.38 GiB/layer gate read) but the 14% route divergence needs a
    perplexity check before production; not assumed safe.
  * Remaining Phase 3b: multi-request batching loop that shares one gate
    pass across B tokens (the quantified 10.8 -> 1.3 GiB/token lever),
    then wire the same three entry points into geodessical gguf.c/llm.c.
- **CORRECTION — the real V4 router architecture (2026-08-14):**
  Verified against deepseek-ai/DeepSeek-V4-Pro (config.json +
  inference/model.py) and bati.cpp (llama-graph.cpp / deepseek4.cpp):
  * routing is NOT done by ffn_gate_exps. Layers 0-2 (n_hash_layers=3)
    route via the deterministic `ffn_gate_tid2eid` table (token_id -> 6
    ids, zero GEMV, zero per-token IO). Layers 3+ route via the DENSE
    `ffn_gate_inp` (7168,384) F32 + `exp_probs_b` bias with
    sqrt-softplus gating, top-6, route_scale 2.5.
  * `ffn_gate_exps` is the per-expert SwiGLU gate (w1) — read only for
    the top-6 selected experts, exactly like up/down_exps.
  * The earlier "gates dominate the IO / batch-amortize the gate pass"
    analysis is RETRACTED. Corrected ExpertStore.io_model (committed):
    per-token IO = top-6 x (gate 9.46 + up 9.46 + down 15.1 MB) = 204
    MB/layer + shared expert ~34 MB/layer => **~15.2 GiB/token over 61
    layers => ~0.13 tok/s floor at 2 GB/s** (vs llama.cpp's observed
    ~0.05 tok/s => the lazy executor is worth ~3x; batching does not
    reduce expert IO, only amortizes fixed costs).
  * Router storage is ~641 MB total (one-time read, resident).
  * Hash-layer gates also simplify layers 0-2: their routing needs no
    hidden state at all.
  ExpertStore now implements the true routing (hash table + dense
  sqrt-softplus), reads router/shexp tensors, and the io_model reflects
  it. Suite 58/58. Remaining: C-side dense-router GEMV is trivial; the
  lazy executor target is the ~15 GiB/token floor.
- **Phase 3b — lazy MoE layer executor, measured (2026-08-14):**
  `ultratensor/moe_exec.py` `MoELayer`: full DeepSeek-V4 layer semantics
  (hash/dense routing with the real weights — hash layers still use the
  dense gate for weights, sqrt-softplus + bias, top-6, route_scale 2.5,
  swiglu clamps [-10, 10]/max 10, per-expert weights applied before w2,
  shared expert). Only selected experts are decoded (C executor).
  C executor hardened: whole-expert single read (scattered small freads
  were pathological), OpenMP row parallelism, fused Q3_K decode+dot.
  MEASURED on real shard 1, layer 0: **0.60 s/token-layer** (18 GEMVs,
  decode-bound at ~700M elems/s scalar C; warm-cache invariant) ->
  36 s/token = 0.027 tok/s full-model projection. llama.cpp's 0.05
  tok/s wins because it uses SIMD dequant kernels; the lazy path's
  disk floor is ~0.13 tok/s. => the remaining gap is SIMD: either
  AVX2 decode in expert_gemv.c or reusing ggml's dequantize_row_*
  kernels via the geodessical integration. All primitives verified
  (mini-shard MoE reference test bit-tight <2%, suite 59/59).
- **Phase 3b — AVX2 decoders land, lazy path beats llama.cpp (2026-08-14):**
  expert_gemv.c now has AVX2 fused decode+dot for Q3_K and Q5_K
  (the V4-Pro expert hot paths) + working OpenMP row parallelism.
  Gotcha: MSVC CLASSIC OpenMP rejects `int i` in the parallel-for
  init (C3015) and `#pragma omp atomic write` (C7660) — the earlier
  /openmp builds were SILENTLY FAILING (check the DLL timestamp
  after building!). Measured on real shard 1, layer 0:
  **0.169 s/token-layer (3.5x)** -> 10.3 s/token = **0.097 tok/s**
  full-model projection, vs llama.cpp 0.05 tok/s on this box; batch-4
  = 0.70 s (5.7 tok/s equivalent). Disk floor ~0.13 tok/s is now in
  reach (double-buffered reads + per-layer expert dedup across a
  batch are the next steps). Suite 59/59.
- **CRITICAL LAYOUT FIX + CUDA factored kernel (2026-08-15):**
  * Container layout bug: the writer stored all scales then all packed
    codes, but every kernel (CPU AVX2, CUDA, geodessical llm.c) reads
    per-row interleaved scales|codes. New files therefore decoded
    garbage for rank rows > 0 — masked because the stale fixtures were
    written by an older interleaved writer. Now unified row-interleaved
    (writer + stride-view reader + factored_exec.c loader).
  * AVX2 stage-1 dot lane bug: x[0..7] was multiplied by LO nibbles
    only; elements are lo/hi interleaved per byte — fixed with
    unpacklo/unpackhi_epi8 (200/200 micro-trials vs scalar oracle).
  * CUDA two-stage kernel `ck_gemv_factored` in geodessical
    (cuda_kernels.cu): stage-1 one block of k threads over uq4 rows,
    stage-2 fp16 basis GEMV. Verified vs numpy oracle (max_rel ~1e-6)
    via tests/runtime/test_cuda_factored.py (ctypes).
  * ACCEPTANCE RE-MEASURED ON CORRECT MATH: smollm2 rank-128 factored
    FFN decodes 173 tok/s vs 93 dense Q8_0 (1.86x) at 65% file size.
  * Remaining: llm.c GPU-dispatch wiring (upload C+U pairs, call
    ck_gemv_factored from llm_gemv) for the full CUDA path.
- **Factored expert stacks (E slices) through the CUDA pipeline
  (2026-08-15):**
  `ck_gemv_factored` now takes E: stage-1 launches E blocks of k
  threads (one per expert over the row-interleaved C [E,k,n]), stage-2
  indexes the accumulator per expert and the U basis over m*E rows.
  llm.c uploads 3-D factored tensors (E*k C rows + fp16 U [E,m,k]) and
  passes E in the dispatch. Expert-stack test cases (E=4, E=2) match
  the numpy oracle at ~5e-7 rel; smollm2 regression clean. This is the
  compute-side groundwork for the V4-Pro MoE path: remaining is the
  router + expert-selection semantics in the forward pass.
- **GPU dispatch wired — factored GEMVs run on CUDA (2026-08-15):**
  * `ck_gemv_factored` now dispatched from `llm_gemv`: C codes + the
    paired U basis are uploaded to VRAM at model load (k rows, not
    out_dim — fixed a 12x over-upload), and the two-stage kernel runs
    on device. The GPU-resident forward pass is disabled for factored
    FFN (falls back to per-GEMV dispatch).
  * Verified vs the numpy oracle ON THE REAL MODEL BYTES
    (smollm2_factored.gguf blk.0 ffn_gate): max_rel 4.2e-7.
  * smollm2 GPU run: 53 tok/s decode (PCIe-bound at this size; the win
    is for V4-Pro-class expert stacks where compute dominates).
  * Complete V4-Pro benchmark cycle (llama.cpp baseline): greet 0.01,
    math 0.01, code 0.02, logic 0.01 tok/s — lazy executor projection
    ~10-15x on all four prompts.
- **Phase 2 speed on the CPU factored kernel (2026-08-15):**
  `runtime/nn/factored_gemv.h` gains AVX2+FMA paths (stage-1 uq4 dot
  decodes 4x8 nibble lanes per 32-col block; stage-2 fp16 basis GEMV
  converts 8 fp16/step; scalar path stays as the test oracle).
  Micro-bench on smollm2 shapes: 0.015 ms vs 0.128 ms scalar (9-11x);
  smoke test remains bit-exact vs the numpy oracle.
  **Acceptance re-measured after a bind-table fix (64 -> 512 entries;
  >64 tensors used to fall into the slow generic path with wrong
  stage-1-only math): factored smollm2 decodes at 173 tok/s vs 93
  dense Q8_0 — 1.86x faster at 65% of the file size (see the layout
  fix entry above — numbers now on verified-correct math).** The
  factored FFN compute win is real: ~270K MACs/tensor vs 884K dense.
- **Phase 3 ACCEPTED — full factored model runs end-to-end
  (2026-08-15):**
  smollm2-135m q8_0 (30 layers, 576-dim): factored all 90 FFN tensors
  at rank 128 -> 94.7 MB (65.4% of dense 144.8 MB; rank for 97%
  energy is ~485/576 — smollm FFN spectra are as flat as V4-Pro's).
  geodessical loads the factored GGUF (362 tensors), binds every
  `.factored_C`/`.factored_U` pair, and generates end-to-end via the
  two-stage `U @ (C @ x)` GEMV: 8 tokens, rc=0, 3.9 tok/s decode vs
  8.9 dense (the factored kernel is still scalar — the Phase 2 AVX2
  fused kernel is the next speedup; compute-side it should beat dense
  Q8_0 at these ranks). Pipeline: `scripts/factor_model.py src out
  --rank N` -> `geodessical out.gguf`.
  Two real bugs fixed along the way:
  * UltraTensor: GGML_TYPE_Q8_0 was 7 (Q5_1's id) — real Q8_0 is 8;
    container tests passed only against their own type-7 fixtures.
    `_tensor_byte_size` now carries the full GGML type table so copy
    tensors of any quant survive.
  * HyperTensor host/main.c: parse_args never zeroed GD_args_t; the
    uninitialized `axex_attn_svd` read stack garbage, so every plain
    run silently SVD-compressed Q/O at random ranks and exited rc=-1.
    Fixed with `memset(args, 0, sizeof(*args))` before defaults.
- **Phase 3 — geodessical executes factored GGUFs (2026-08-14/15):**
  HyperTensor runtime now parses AND executes the UltraTensor container:
  * `gguf.h`: `GGML_TYPE_FACTORED_C = 2048` (sparse id above
    GGML_TYPE_COUNT; gguf.c returns a dedicated type-info entry:
    32-elem blocks, 20 B — fp32 scale + 16 B packed uq4 codes).
  * `llm.c`: `llm_row_bytes` + `llm_vec_dot` uq4 cases; a bind registry
    maps each `<name>.factored_C` to its fp16 `<name>.factored_U` basis
    during tensor mapping; `llm_gemv` dispatches `out = U @ (C @ x)`.
  * `runtime/nn/factored_gemv.h`: the shared two-stage kernel (stage-1
    uq4 dot into a k-wide accumulator, stage-2 fp16 basis GEMV — no
    dense intermediate in DRAM).
  * New smoke test `tests/runtime/test_factored_gguf.c`: real gguf.c
    parse + the shared kernel reconstruct a container produced by
    `ultratensor.gguf_factored` — **bit-exact vs the numpy oracle**
    (max_abs_err 0.0e+00). geodessical.exe builds clean (zig).
  * Committed in HyperTensor (bbef379). Remaining: a full factored
    model file to run end-to-end through geodessical (the pipeline
    exists: factor -> factored GGUF -> geodessical decode).
- **FIRST REAL V4-Pro DECODE NUMBER (2026-08-15 ~00:00):**
  Overnight benchmark on :8774: greet = **2020.2 s, 0.01 tok/s**
  generation (spec draft 6/26). Prefill 0.05 tok/s, decode 0.01 tok/s
  => the lazy executor projection (0.153 tok/s ffn-only / 0.079 floor)
  is **~15x llama.cpp decode** on this box. [math] in flight.
- **Phase 3b — read/decode overlap lands (2026-08-14):**
  ut_expert_gemv_mem decodes from a caller-owned buffer; MoELayer
  runs a prefetch thread that pre-reads the next selected expert's
  three tensors while the current one decodes.
  **0.107 s/token-layer -> 0.153 tok/s** full-model projection —
  above the conservative 0.13 disk-floor estimate (reads measured at
  1.24 GB/s and now fully overlapped) and **3x llama.cpp's 0.05 tok/s**
  on this box. The lazy MoE compute path is now disk-bound; further
  gains need attention-layer reads or multi-request expert dedup.
  Suite 59/59.
- **Phase 3b — complete IO budget (2026-08-14):**
  io_model now inventories all blk tensors (attention/MLA/indexer/
  compressor/norm) and counts the per-token dense reads (~166 MiB/
  layer on V4-Pro: attn ~126 + indexer/compressor/norms). Complete
  full-model floor: **25.3 GiB/token = 0.079 tok/s at 2 GB/s**
  (routed experts 13.0, dense/attn 10.1, shexp 2.2 GiB). The measured
  ffn-only lazy path (0.153 tok/s) plus serial attention reads lands
  at ~0.078 tok/s — 1.5x llama.cpp. Attention is now the second
  read-bound, after routed experts; both can overlap via the same
  prefetch pipeline. Suite 59/59.


- **Phase 3c — multi-shard ExpertStore + dense-router lazy timing (2026-08-15):**
  ExpertStore now inventories and reads across ALL 17 shards
  (per-tensor shard routing in read_tensor/read_expert; _ExpertReader
  and the C executors open the owning shard). First-ever dense-router
  lazy measurement: layer 3 = 0.226 s/token-layer (vs 0.150 hash,
  0.157 serial) -> ~0.072 tok/s over 61 layers with the single-thread
  build.
  CRITICAL BUG: the /openmp MSVC build of expert_gemv.dll corrupts
  the heap under Python-thread concurrency (native AVs in ~50% of
  MoELayer runs; reader thread + decode thread both fault). Fixed by
  replacing OpenMP with a self-contained Win32 row pool (up to 16
  threads, per-worker scratch; serial fallback on POSIX); 10/10
  crash-free, identical numerics, suite 59/59.
  scripts/bench_moe_layer.py + bench_moe_sweep.py: per-layer and
  all-61-layer lazy timing (subprocess per layer, incremental JSON).
  Overnight 300-token sweep running.

- **Phase 3d — expert-factoring verdict + full-model lazy projection (2026-08-15):**
  DECISIVE NEGATIVE: V4 expert SVD spectra are extremely flat
  (layer-3 gate expert: k=128 -> 12.3% energy / 0.94 rel err;
  k=1024 -> 62% / 0.61). Matching Q3_K's own ~0.27 rel error needs
  k~2000+, where factored bytes (0.90 B/elem) EXCEED dense Q3_K
  (0.43 B/elem). Per-expert SVD factoring is rejected on data; the
  routed-expert read mass (11.5 GiB/token) is irreducible on this
  quant. (Also verified: the lazy C decoder matches numpy on real
  V4 bytes, max_rel 5.4e-7.)
  scripts/bench_lazy_full.py: honest full-model projection from the
  measured per-layer FFN times + the complete byte inventory:
    serial 0.042 / pipelined 0.075 / resident 0.076 tok/s;
    disk ceiling 0.092 tok/s (21.66 GiB/token at 2 GB/s).
  NEW STRATEGY - resident dense+shexp: attention/indexer/compressor/
  norm + shared-expert tensors total only 10.15 GiB -> pin in RAM;
  per-token reads drop to routed experts alone (11.5 GiB) and the
  ceiling rises to 0.174 tok/s. FFN decode then dominates.
  (Note: precise io_model inventory = 21.66 GiB/token, correcting
  the earlier 25.3 GiB estimate.)
  (RAM check: 31.3 GiB total; the resident 10.15 GiB cache fits when
  the lazy server runs standalone, not alongside llama-server.)

- **Phase 3e — e2e correctness on REAL model bytes (2026-08-15):**
  scripts/check_moe_layer_e2e.py compares the lazy MoELayer (C
  executors + prefetch reader + router) against a pure-numpy reference
  implementing the official Gate/Expert semantics on actual V4 tensors:
  layer 0 (hash) max_rel 5.5e-7, layer 3 (dense) 1.5e-6, layer 30
  1.2e-6 — PASS at the 2e-3 threshold. The lazy executor reproduces
  the model's layer computation on real bytes across hash and dense
  routing and deep shards. scripts/check_moe_layer_all.py runs the
  same check over all 61 layers (incremental JSON; waits for the
  timing sweep so the two jobs never contend for the disk) — the
  overnight job.

- **Phase 3f — batch-dedup serving math + geodessical MoE design (2026-08-15):**
  bench_lazy_full now models multi-request batching: with the resident
  cache, reading each DISTINCT routed expert once per batch gives
  aggregate ceilings 0.178 (B=4) / 0.195 (B=16) / 0.274 (B=64) /
  **0.708 tok/s (B=256, 4.1x dedup)** vs 0.092 single-token - the
  strongest serving strategy found (batch-256 latency = 256/0.71 =
  ~360 s/batch, i.e. throughput-oriented, many concurrent users).
  GEODESSICAL V4 MoE FORWARD - scoped design (next major C work):
  gguf.c already parses 3-D tensors; needed: (1) multi-shard model
  open, (2) deepseek4 arch branch (router: hash tid2eid layers 0-2,
  dense sqrt-softplus + bias-selection top-6, weights from UNBIASED
  scores x2.5), (3) per-expert streaming reads + decode reusing the
  expert_gemv.c row pool design, (4) MLA attention path (attn is
  deepseek MLA, not MHA). Milestones: A loader inventory (no exec),
  B hash-layer forward, C dense-router + full model.

- **Milestone A DONE (2026-08-15):** geodessical --inventory mode lands
  (HyperTensor commit d3f47a2): multi-shard glob expansion + full
  header/KV/tensor dump with per-shard data-section offsets, early
  exit before any KV/GPU allocation. Verified on the REAL V4-Pro set:
  17 shards, arch=deepseek4, 61 layers, 128 heads / 1 kv head (MLA),
  yarn rope, expert_count=384, hash_layer_count=3, total 714,048 MB
  (697 GiB) / 1.573T params - matches the known model. Inventory
  confirms blk.<L>.ffn_gate_exps.weight Q3_K (7168x3072x384, 3.47 GB
  per layer), ffn_gate_inp F32 (7168x384), ffn_gate_tid2eid
  (6x129280), ffn_*_shexp, indexer/compressor/hc tensors.
  Note: ffn_gate_tid2eid is F32 in the GGUF (not I32 as documented).

- **Phase 3g — FINAL full-model lazy projection (2026-08-15, complete
  300-token x 61-layer sweep):**
  Per-layer lazy MoE timing (real bytes, all 61 layers): mean 0.1339
  s/token-layer (hash 0.1535, dense 0.1329; slowest L1 0.1544,
  fastest L60 0.1188) -> FFN-only 0.122 tok/s.
  Full-model projections (21.66 GiB/token at 2 GB/s):
    serial     0.053 tok/s
    pipelined  0.092 tok/s   (= disk ceiling)
    RESIDENT   0.122 tok/s   (10.15 GiB dense+shexp cache; FFN-bound)
    resident ceiling         0.174 tok/s
    batch-256 aggregate      0.708 tok/s
  vs llama.cpp measured 0.01-0.02 tok/s on the same box: the lazy
  resident strategy is ~6-12x, with the batch-dedup serving design
  reaching ~35-70x aggregate. THE END STATE of the lazy serving
  measurement track.

- **E2E record COMPLETE (2026-08-15):** the overnight all-layer check
  finished 61/61 PASS - the lazy executor reproduces the reference
  semantics on EVERY layer of the real V4-Pro model (hash 0-2 and
  dense 3-60, all 17 shards), worst max_rel 1.58e-6 (threshold 2e-3).
  Outputs: outputs/e2e_all_layers.json. The lazy serving track's
  correctness + performance evidence is complete.

- **Milestone B DONE (2026-08-15):** geodessical hash-layer MoE forward
  (runtime/nn/moe_v4.c + tests/runtime/test_moe_v4.c; HyperTensor commit
  above). Streaming Q3_K/Q5_K/Q6_K decode from the memory-mapped shard,
  Win32 row pool, hash router (I32 tid2eid read — the bati GGUF stores
  the table as I32 bit patterns under an F32 header type) + unbiased
  sqrt(softplus) weights + swiglu clamps + shexp add. Validated on REAL
  bytes vs the 61/61-trusted Python oracle: layer 0 1.0e-6, layer 1
  1.8e-6, layer 2 1.6e-6 max_rel. scripts/check_moe_c.py drives the
  cross-check. Milestone C next: dense-router layers 3-60 (biased
  top-k selection + the same expert stack).

- **Milestone C DONE + dense-router bug fixed (2026-08-15):**
  geodessical now executes BOTH V4 MoE router variants on real bytes:
  hash (tid2eid, I32-in-F32 read) and dense (top-k of sqrt(softplus) +
  bias selection, unbiased weights). moe_v4_open_layer spans shard
  boundaries (blk.3 straddles shards 1/2). Cross-validated vs the
  python oracle: layers 0/1/2/3/30/60 at 1.0-1.8e-6 max_rel.
  IMPORTANT: the independent C router exposed a python lazy-executor
  bug - route_layer added exp_probs_b INSIDE the softplus; the
  reference adds it AFTER sqrt(softplus) for selection (bias never
  enters the routing weights). Fixed + exhaustive-sort regression
  test. The 61-layer e2e record is being re-run with the corrected
  selection (outputs/e2e_all_layers.json will be regenerated).

- **Milestone D DONE (2026-08-15):** geodessical embedding/norm/output
  path on real bytes: moe_v4_embd (token_embd Q3_K row), RMSNorm
  (eps 1e-5), moe_v4_logits (output.weight Q6_K, 129280 rows).
  Validated vs numpy: embd EXACT, norm 9.1e-8, logits 3.4e-6 rel with
  10/10 top-10 token agreement. Also fixed the Q6_K decoder to the
  b10424 scale layout (4 pairs/128-block; the old port used the
  pre-2024 layout and only output.weight exercises Q6_K). Corrected
  61/61 e2e record (bias-placement fix) complete: outputs/e2e_all_
  layers.json regenerated, all PASS. Remaining Phase 3: MLA attention
  + hyper-connection/indexer gating + full multi-shard serve loop.

- **Milestone E DONE (2026-08-15):** geodessical executes the FULL
  V4 transformer block on real bytes (runtime/nn/v4_block.c): hc_pre/
  hc_post with the split-Sinkhorn mixing (hc=4), attn_norm, MLA
  sliding-window attention (wq_a/q_norm/wq_b, per-head q norm, wkv +
  kv_norm, fp8-e4m3 act-quant on non-rope dims, sink dilution, grouped
  wo_a/wo_b), hc_post, hc_pre(ffn), ffn_norm, hash/dense MoE, hc_post.
  Validated vs a numpy port of inference/model.py + kernel.py: 3.4e-7
  max_rel. Fixed: row-pool write overrun when threads > rows (heap
  corruption on 24-row hc tensors); the block reuses the MoE
  accumulator (zeroed). Rope is identity at position 0 - positions
  >0, the compressor, and the indexer (ratio-4 layers) remain.

- **Rope DONE (2026-08-15):** YaRN compress rope (base 160000, factor 16,
  beta_fast 32, beta_slow 1, original 65536) now applied in v4_block.c at
  any start_pos on the last 64 dims of q (per head) and kv, with the
  inverse applied on o; numpy twin in v4_ref_block.py. Validated vs the
  reference at pos 0/7/7000/131072: 3.2-3.4e-7. Remain: compressor,
  indexer (ratio-4 layers), final head, serve loop.

- **Milestone F DONE (2026-08-15):** learned KV compressor in geodessical
  (runtime/nn/v4_comp.c): ratio-4 layers with OVERLAP (coff=2), gated
  softmax pooling over 8 slots (prev-window first-half + current second-half
  dims), prefill windowing + decode incremental state, RMSNorm + YaRN rope
  at the window position + fp8 act-quant emit. Validated vs a numpy port of
  the official Compressor on real layer-2 bytes: prefill 7.3e-7, decode
  6.1e-7. Remain: indexer (top-k 1024 over fp4-rotated compressed KV),
  final head, serve loop.

- **Milestone G DONE (2026-08-15):** learned indexer in geodessical
  (runtime/nn/v4_index.c) for ratio-4 layers: q path (wq_b 1536->8192,
  YaRN rope on last 64, normalized Hadamard rotate 128, fp4-e2m1 block-32
  quant with power-of-2 scales), its own rotated compressor (head_dim 128,
  rope->Hadamard->fp4), weights_proj scoring (relu dot * head_dim^-0.5
  n_heads^-0.5), causal mask (prefill), topk min(1024, visible) with
  seqlen/win offsets. Validated vs numpy oracle on real layer-2 bytes:
  scores 6.2e-7 / 4.6e-7, topk exact. Remain: final head (output_hc_*),
  multi-shard serve loop.

- **Milestone I DONE (2026-08-15):** geodessical executes the ENTIRE
  V4-Pro model end-to-end on real bytes (runtime/nn/v4_serve.c): 17 mapped
  shards, token_embd -> 61 full-HC-state blocks (hc split-Sinkhorn + MLA +
  rope + MoE) -> final head (hc_head + output_norm) -> 129280 logits in
  14.5 s (0.24 s/layer) on this 32 GB laptop. Validated against a numpy
  61-layer oracle on the same token: max_rel 5.2e-6, top10 10/10.
  PHASE 3 COMPLETE. Remain: multi-token decode with per-layer caches
  (window 128 + compressor + indexer wiring), Phase 4 (OpenAI parity
  serving), Phase 5 (llama.cpp fork).

- **Milestone J DONE (2026-08-15):** cached multi-token decode in the block
  (v4_block.c): window ring 128 + compressor rows (ratio 4 overlap and
  ratio 128 plain, v4_comp.c generalized) + multi-entry sparse attention
  with the sink term. Validated vs a numpy sequence oracle on real bytes:
  L1 (ratio 128) 3.9e-7, L2 (ratio 4) 5.5e-7 over 12 tokens.
- **Milestone K DONE (2026-08-15):** v4_serve_step/generate: chained HC states
  + greedy argmax over the cached 61-layer loop; 3-token smoke generation
  runs (16.6 s/tok under oracle load). Remain: tokenizer + sampling +
  OpenAI-parity API, indexer topk wiring past 4096 tokens.

- **Milestone L DONE (2026-08-15):** standalone BPE tokenizer for the V4
  GGUF (runtime/nn/v4_tok.c): DeepSeek3-style pre-tokenizer (digits 1-3 /
  CJK isolation, punct+letters, optional-char+letters, punct/symbol runs,
  newline and whitespace rules incl. the last-space-joins-word behavior),
  GPT-2 byte-level encoding, rank-ordered BPE merges from
  tokenizer.ggml.merges with hashed pair ranks. Differential battery vs the
  llama.cpp server /tokenize: 15/15 exact. Remain: OpenAI-parity API
  wiring (chat turn + SSE), sampling (norm_topk_prob).

- **Milestone M DONE (2026-08-15):** chat-turn API on top of the V4 serve
  loop (runtime/nn/v4_api.c): tokenizer encode, bos/eos sentence-boundary
  template wrap, temperature + top-k sampling, xorshift RNG. Hooked into
  the geodessical API server (host/api_server.c) via GD_api_enable_v4()
  so the existing /v1/chat/completions path routes through V4 under a
  per-process lock. Remain: SSE streaming (finish_reason/usage),
  norm_topk_prob, serve-loop launch, Phase 5 llama.cpp fork.

- **Milestone J wrap DONE (2026-08-15):** 130-token cached-decode
  sequence on L1 including the window-128 ring wrap (p=127-129):
  worst_rel 2.1e-6 PASS. Oracle view-aliasing at the compressor slot
  reset fixed (.copy()).

- **Milestone N (2026-08-15):** Phase-4 close-out in the geodessical
  runtime: chat-turn prompt prefill + per-token embedding (token_id is
  hash-routing only), SSE token streaming via v4_set_stream_cb, the
  official full-vocab Gumbel-max sampler (top_k=0), OpenAI-parity
  choices/finish_reason/usage, --v4-shards serve wiring, and a GPT-2
  byte-decode passthrough fix (Latin-1/high chars). Chat smoke now
  produces coherent English on the 697 GB model. Remain: generation
  oracle comparison, then the singular public push.

- **Milestone N DONE (2026-08-15):** chained greedy generation validated
  end-to-end against the numpy oracle on real bytes: 61 layers x 3
  tokens with per-token embedding + hash routing — 3/3 exact
  ([5, 223, 643] both sides, oracle 2389 s). Phase 4 complete; repo
  published as a single commit.

- **Milestone O DONE (2026-08-15):** Phase 5 kernel landed in the
  llama.cpp fork (branch ultratensor-factored, sibling repo): GGML type
  GGML_TYPE_FACTORED_C (uq4 codes, fp32 scale per 32 cols, the
  UltraTensor container layout) with quantize/dequantize, a scalar
  vec_dot for the CPU mul_mat path, traits entries and the GGUF type-id
  2048 mapping; execution = U @ (C @ x) via two stock mul_mats.
  Validated vs the numpy oracle on real expert bytes
  (blk.0.ffn_gate_exps e=0, 3072x7168, k=128): stage1 rel 9.6e-5,
  stage2 abs 1.2e-6, 0.46 ms/decode (4 GFLOPS scalar). Honest finding:
  gate experts have a near-flat spectrum — rank-128 truncation alone
  keeps ~100% of the dense residual, so factored serving needs higher
  ranks or sink-row variants (see MEASUREMENTS_v4pro.md projections).

- **Milestone P (2026-08-15):** Phase-5 fork hardened + first Phase-6 data:
  regression-compiled all 30 src + 27 common files of the fork with the
  FACTORED_C changes (clean); container _dequant extended to every GGUF
  quant type and drop_unmatched mode added. Honest rank curve on real
  bytes: blk.2.attn_q_a (7168x1536 Q3_K) needs rank 1459/1536 at 99%
  energy — 3.4x BIGGER factored, 13.7% frob err. q2_0 direct expert
  compression (the measured real lever) running over
  blk.0.ffn_gate_exps (384 experts, 8.4B elements).

- **Milestone Q DONE (2026-08-15):** first Phase-6 certification data on
  real V4-Pro expert bytes: blk.0.ffn_gate_exps (Q3_K, 384x3072x7168,
  8.4B elements, 3.63 GB) streamed to q2_0 in 373 s -> 2.64 GB (27%
  cut), block-bounded error (amax/6 grid, exact zeros) — matching the
  projected Q3_K->Q2_K reduction. Overnight full-shard certification
  run launched (scripts/q2_certify_all.ps1 -> outputs/q2full.log).

- **Milestone R DONE (2026-08-16):** full Phase-6 certification run over
  all 17 V4-Pro shards: 183 expert tensors (ffn gate/up/down_exps)
  streamed Q3_K -> q2_0 in 17.4 h wall (08-15 15:18 -> 08-16 08:43), all
  rc=0. Expert payload: ~619 GB source -> 450.4 GB q2_0
  (27% cut, matching the Q3_K->Q2_K projection). Remain: geodessical
  q2_0 expert decode + serve benchmark vs the Q3_K_M baseline.

- **Milestone S DONE (2026-08-16):** q2_0 expert decode in the
  geodessical runtime (test_v4_q2.c): on-the-fly 2-bit decode (block 32,
  fp16 scale = amax/3, grid {-3,-1,1,3}) + gemv on a real expert
  (3072x7168) = 18.9 ms/tensor (2.3 GFLOPS scalar), max_abs 2.4e-5 vs
  the numpy reference; error block-bounded (max_abs 0.042, max_rel 0.33
  on real bytes). ROADMAP COMPLETE: Phases 1-6 landed and validated.

## Phase 7 — Conditional V4 (next, per external review synthesis)

The reviewers' core reframing is adopted: treat V4 as TWO compression
targets — active-path cost (~49B active/token) and stored capacity
(619 GB Q3_K) — and make the model conditional at every resource level
(residency, rank, precision, context, state), with the contract
"compression failure => progressive recovery".

Every item the reviews demanded that we have not executed (activation-
space expert similarity, lookahead expert-cache predictor, per-expert
activation-weighted Pareto sweep, conditional-rank sweep,
shared+private factorization, CVaR tail pruning, trajectory-backed
context, router distillation, escalation policy, VQ residuals, tiered
residency) is tracked with first steps in `docs/REVIEW_GAPS.md`
(G1-G12). Phase 7 = work that queue in the order listed there.

### Phase 7 status (2026-08-16)

Toolkit: ALL gap families now have code in `ultratensor.conditional/`
(140 tests green, 2 skipped) — see the README table for the module map.

Measured so far (real V4-Pro bytes):
- hash-layer expert churn 0.98-0.99 (~200 MB new experts/token) ->
  the token drafter IS the hash-layer prefetch predictor; perfect H=4
  lookahead reaches 100% coverage at the 24-expert union, a weak
  drafter plateaus at ~21% (drafter quality is the binding constraint).
- spec-decode what-if with measured costs: perfect-drafter ceiling
  0.754 tok/s == batch-256 aggregate ceiling 0.708; drafting is the
  prefetch lever, not the throughput lever. q2_0 read-cut: 0.123 ->
  0.143 tok/s (capacity win, not latency).
- exact per-layer bootstrap CIs: hash L0 0.1511 [0.1466, 0.1550],
  dense L10 0.1452 [0.1398, 0.1504]; churn explains the gap.
- NVML thermal chain validated on the real GPU; GPU is 2.0% of decode
  energy here (thermal-GPU rank targets CUDA-bound paths).

Gap verdicts (see docs/REVIEW_GAPS.md for the full record):
- G2 closed negative: experts distinct in activation space (0.0388/0.0078).
- G4 CORRECTED by held-out split: k95_act=8 was in-sample overfit;
  deployable rank unknown (hold 0.47-0.73 at r=16); projector form +
  held-out split is now the standard estimator.
- G5 weak positive: conditional rank beats fixed (1-2.5%) but absolute
  error 0.84-0.94 -> the rank lever on expert GEMMs is dead.
- G6 closed negative: shared bases LOSE to independent at equal budget
  (0.498 vs 0.399 act error) — no duplicated bases to share.
- G11 closed negative: PQ/VQ dead on expert matrices (8-bit keeps 6%).
- G12 measured: oracle hot cap 6 -> 0 misses on the real routing
  sequence; frequency predictor 0 hits — churn kills it; P90 tail
  metric landed in tiering.
- G3/G9: data-limited, exp96 (86-token dense trace, train 64) is
  closing both on the cluster; G7 first real damages measured (CVaR
  tail gate vetoes worst-token-1.0 experts); G8 first slice measured
  (MCR: L0 compress, L1-3 refine); G10 first held-out test negative
  (KNN rho anti-correlated at n=16 — third overfit lesson; ladder
  correctly refuses rank-8); G1 routing-stability slice measured
  (entropy pinned at ln6, margin 1.003); PPL/KL battery + attention-
  mass-per-slot remain unmeasured.

In flight: exp96 dense-router trace on node2 (86 tokens, blocks 0-3)
-> held-out projector curves (G4) + factored controller at 64 train
samples (G9) via scripts/cluster_subspace.py.
