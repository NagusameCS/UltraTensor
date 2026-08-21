# UltraTensor (Raphael)

**MoE surgery, expert splicing, and serving for DeepSeek-V4-class models.**
UltraTensor takes the 1.6T-parameter DeepSeek-V4-Pro and produces runnable,
purpose-built slices of it — down to a 25 GB coder model that serves on a
32 GB laptop, CPU-only. Every number below is measured on real model bytes
and backed by a committed artifact in this repo.

> **Papers.** The formal, publishable write-up is
> [`docs/paper/ultratensor.pdf`](docs/paper/ultratensor.pdf) — a companion to the
> [HyperTensor volume](https://github.com/NagusameCS/HyperTensor) (Papers I–XV).
> The living, artifact-linked working draft is [`docs/PREPRINT.md`](docs/PREPRINT.md);
> the theory roadmap is [`docs/BUILDING_ON_HYPERTENSOR.md`](docs/BUILDING_ON_HYPERTENSOR.md).

---

## Headline results (all verified, evidence in this repo)

| Accomplishment | Number | Evidence |
|---|---|---|
| Expert splice of V4-Pro | 697.3 GB source (17 shards, Q3_K_M) → **156.1 GB keep64**, **4.5× smaller** | `scripts/v4_coder_keep.py`, sizes measured on disk |
| keep64 keeps code mass | dense layer L3: top-64 experts carry **97.66%** of code routing mass, from only **77 distinct experts** (32 of them never in the general top-64) | `outputs/code_census.json` (87 real code tokens) |
| Uniform expert ladder | **keep8u 25.0 GiB / keep12u 32.2 GiB / keep16u 39.3 GiB** — 8/12/16 experts per layer, all native-loading and generating; keep8u is **27.9× smaller** than the source | model files on disk; `outputs/keep16u_gen_test.log`; live server on `:8780` |
| IQ2_XS requant of keep16u | **39.3 → 24.7 GiB (2.36 BPW)**, native-loading and generating (smoke: 0.5 t/s prompt, 0.1 t/s gen) | `outputs/keep16u_iq2xs_test.log` |
| IQ2_XS requant of keep12u | **32.2 → 20.8 GiB (2.36 BPW)** | quant log `outputs/quant_keep12u.err.log` |
| IQ2_XS requant of keep8u | **25.0 → 16.3 GiB (2.38 BPW)** — smallest GPU-capable splice | quant log `outputs/quant_keep8u.err.log` |
| **GPU serving (8 GB VRAM)** | keep16u-iq2xs on the RTX 4070 8 GB: **1.5 tok/s generation, 1.6 t/s prompt** (`-ngl 12`) — **15× the CPU speed**; live server on `:8788` | `outputs/cuda_diag.log`, `outputs/keep16u_iq2xs_server.log` |
| **Full GPU fleet (3 models, 8 GB VRAM)** | keep8u-iq2xs **1.2 t/s** on `:8789`, keep12u-iq2xs **0.1 t/s** on `:8790`, keep16u-iq2xs **1.5 t/s** on `:8788` — all smoke-verified, all registered in HyperMoE (`coder-gpu-8/12/16`) | `outputs/smoke8u_d.log`, autopilot logs |
| Difficulty predictor (escalation gate) | 3-way ridge: **Spearman 0.98, MAE 0.028, tier agreement 1.0** (192 train / 64 held-out tokens; preliminary, 256-token traces in progress) | `outputs/rho_192_run.log` |
| Prefetch tables | **129,280 token → top-6 expert rows** exported from real routing traces (9.3 MB) | `outputs/hash_route_tables.npz` |
| Full-arch numpy oracles | block forward max_rel **3.4e-7**; 61-layer e2e serve max_rel **5.2e-6**, top-10 logits **10/10**; tokenizer battery **15/15 exact** | `scripts/v4_ref_*.py`, `scripts/check_v4_*.py` |

The laptop is a 32 GB RAM / RTX 4070
8 GB machine. **Both serving modes are now verified:** CPU-only across the
keep-ladder (live server on `:8780`), and **GPU on the 8 GB VRAM card**
with the IQ2_XS splice (`-ngl 12`, live server on `:8788`) at 1.5 tok/s
generation — 15× the CPU rate. Q3_K now decodes on GPU too (0.07–0.15 t/s
partial offload; the blocking defect was fixed in the fork). The headline:
1.6T-parameter-class models spliced down to 25–156 GB, served from a
laptop — CPU and GPU.

---

## Downloading the spliced models

The keep-ladder GGUFs and their IQ2_XS requants are published on Hugging Face.
See [`docs/MODELS.md`](docs/MODELS.md) for the catalog, license caveats, and
serving flags. One command after `pip install huggingface_hub`:

```powershell
python scripts/download_models.py --dest models
```

Two serving rules that matter: `--no-op-offload` is mandatory on every Q3_K
tier (the engine default corrupts CPU-resident Q3_K dequants), and
long-sequence GPU decode needs the UltraTensor llama.cpp fork.

---

## Bug found & fixed: the MTP hang in the published GGUFs

The most widely used public V4-Pro GGUF distribution (BatiAI Q3_K_M,
[huggingface.co/batiai/DeepSeek-V4-Pro-GGUF](https://huggingface.co/batiai/DeepSeek-V4-Pro-GGUF))
declares `deepseek4.nextn_predict_layers = 1` but ships **zero `mtp.*`
tensors**. llama.cpp builds with DeepSeek4 MTP decoding then load fine
but hang forever at generation (the MTP loop waits for a draft that can
never exist).

- Audited all 17 original shards: nextn = 1, mtp tensors = 0; Flash
  control file: 2,376 mtp tensors → the detector is sound
  (`scripts/audit_mtp.py`).
- One-line fix: zero the flag (`scripts/patch_gguf_kv.py`). A/B verified:
  same model, flag 1→0, generation goes from infinite hang to working.
- Upstream references: llama.cpp b10424
  [`src/models/deepseek4.cpp#L19-L27`](https://github.com/ggml-org/llama.cpp/blob/b10424/src/models/deepseek4.cpp#L19-L27),
  [`#L1409`](https://github.com/ggml-org/llama.cpp/blob/b10424/src/models/deepseek4.cpp#L1409),
  [`src/llama-context.cpp#L3631-L3633`](https://github.com/ggml-org/llama.cpp/blob/b10424/src/llama-context.cpp#L3631-L3633).

## HyperMoE

A routing + escalation layer on top of the splices (`ultratensor/hypermoe/`,
`scripts/hypermoe_serve.py`, `scripts/v4_hypermoe_registry.py`):

- **Purpose-first domain router** with a specialist registry (11
  specialists) dispatching code/math/backend/frontend/data/devops queries
  to the right expert slice.
- **Escalation gate** driven by the measured rho predictor above: promotes
  a query to a larger model when difficulty exceeds the tier threshold.
- **Hash-layer prefetch tables** (above): hot experts are resident before
  the forward pass.
- HTTP dispatcher verified end-to-end against the keep-ladder
  (`docs/HYPERMOE.md`, `docs/CODER_SERVE.md`).

## Routing science (measured on real traces, G1–G12 in `docs/REVIEW_GAPS.md`)

- **Language-isolation verdict (128 real tokens, layer 3):** programming
  languages share a large common expert core — top-32 pairwise overlap
  15–22 of 32 (python/rust 18, rust/sql 21, sql/js 22) — while 31–53% of
  each language's top-32 is language-specific. Top-32 mass coverage:
  python 0.86, rust 0.92, sql 0.95, js 0.94 (43–63 distinct experts each).
  Design implication: language specialists = shared base + small
  per-language additions, not fully separate networks.
  (`outputs/pl_census.json`)

- Hash-layer expert churn between consecutive tokens: **0.98–0.99** —
  the token drafter *is* the expert prefetcher. A perfect H=4 lookahead
  reaches 100% coverage at the 24-expert union; a weak drafter plateaus
  at ~21%.
- Routing entropy pinned at **ln 6 = 1.7909** (top-6 near-uniform);
  6th/7th expert margin **1.003** — the keep boundary is a coin flip.
- Honest negatives, kept in the record: gate-refit ridge **0.801 vs 0.818**
  sliced baseline (refit loses); shared-factor expert bases lose to
  independent at equal budget; rank-8 subspace was in-sample overfit
  (held-out 0.47–0.73).
- Throughput on this box (CPU lazy path): ~0.12 tok/s; a q2_0 read-cut
  moved 0.123 → 0.143 tok/s. Modest, honestly reported.

## Kernels & quant support

- `ultratensor/kernels/expert_gemv.c`: dispatch-aware per-expert
  decode+GEMV (Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K), AVX2 fused dots.
  Builds to `.dll` (MSVC) and `.so` (gcc); verified against numpy in
  `tests/test_expert_store.py` (9/9 green).
- Dequantization covers every llama.cpp b10424 format, cross-checked
  against `llama-quantize` and gguf-py reference decoders (MXFP4/IQ2_XXS
  verified on real DeepSeek tensor data, 0.0 error).
- GGUF surgery: KV patcher, split-count repair, manifest-aware keep
  writer (`ultratensor/gguf_keep.py`).

## Status — 2026-08-17 (what is done vs in flight)

| Item | State |
|---|---|
| keep16u / keep12u / keep8u | built, patched, native-loading, serving |
| keep64 | built, native-loading after split.count + MTP patches |
| keep16u IQ2_XS requant | done — 24.7 GiB, 2.36 BPW, native smoke OK |
| keep8u IQ2_XS requant | done — 16.3 GiB, 2.38 BPW |
| keep12u IQ2_XS requant | done — 20.8 GiB, 2.36 BPW |
| keep8u/keep12u GPU serve | done — :8789 (1.2 t/s) + :8790 (0.1 t/s), registered `coder-gpu-8/12` |
| keep64 IQ2_XS requant | blocked: one 384-expert tensor needs ~34 GB f32 scratch → "bad allocation" on 32 GB; needs chunked quant or a big-RAM host |
| CPU serving :8780 | working — requires `--no-op-offload` (llama.cpp op-offload puts a Q3_K dequant on the GPU even at `-ngl 0` and crashes at >=36 prompt tokens); PPL battery complete, mean 2.653 |
| CUDA offload | **working on IQ2_XS splice** — 1.5 tok/s gen on 8 GB VRAM (`:8788`); long-prompt decode crash ROOT-CAUSED + FIXED in fork (`95fcdad`: repeated expert ids from fallback remap broke the cuBLAS fallback invariant) — 55-token prefill / 24-token gen / logprobs validated on rebuilt engine |
| GPU logprobs | **working on rebuilt engine** — `n_probs`/`top_logprobs` validated on :8791 (was: CUDA illegal access in vocab top-k on old engine) |
| node2 (32-core cluster box) | second experiment host; parallel batteries + C-kernel runs |

## Evidence index

| File | Proves |
|---|---|
| `outputs/code_census.json` | 97.66% top-64 mass, 77 distinct, 32 code-exclusive |
| `outputs/rho_192_run.log` | ridge 0.9798 / MAE 0.0281 / tier 1.0 |
| `outputs/hash_route_tables.npz` | 129,280 token→expert prefetch rows |
| `outputs/keep16u_gen_test.log` | native generation on the patched splice |
| `outputs/keep16u_iq2xs_test.log` | IQ2_XS splice (2.36 BPW) generates natively |
| `scripts/audit_mtp.py` | nextn=1 + zero mtp across all 17 published shards |
| `tests/test_expert_store.py` | C kernel == numpy ground truth (9/9) |
| `scripts/check_v4_block.py`, `check_v4_serve.py`, `check_v4_tok.py` | full-arch oracle fidelity |

## Install & test

```powershell
pip install -e .
python -m pytest tests -v
```

## Limitations (explicit, to avoid over-reading)

- Reconstruction metrics are numerical fidelity, not task quality. Task
  retention now measured: PPL battery on CPU-served keep16u (Q3_K_M,
  max_tokens=8) = **mean 2.653** (code 3.000, math 2.887, multilingual
  2.629, rare 2.536, needle 2.215; `outputs/ppl_cpu16_full.json`). The
  GPU IQ2_XS tier is a speed tier, not a quality tier: 16-token battery
  with `--no-op-offload` measures **mean 8.540** on all five domains
  (degenerate token loops; `outputs/ppl_gpu16_iq2xs_noopoff.json`).
  `--no-op-offload` is mandatory on every Q3_K tier: the default
  op-offload crashes at 36+ prompt tokens AND silently corrupts
  short-prompt outputs.
- Single-machine evidence (one Windows 11 laptop, CPU serving; one
  32-core Linux node for traces). No cross-hardware claims.
- The rho predictor numbers are preliminary (192 train / 64 held-out);
  larger 256-token traces are the final run.
- GPU serving is verified on the IQ2_XS splice only (1.5 tok/s, 8 GB
  VRAM, `-ngl 12`); the Q3_K expert stacks still stall on the fork's
  GPU path, so the other splices remain CPU-only until that kernel is
  repaired.
