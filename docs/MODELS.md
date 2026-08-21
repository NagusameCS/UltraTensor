# UltraTensor Models — Splices of DeepSeek-V4-Coder

This page describes the spliced GGUFs published alongside UltraTensor, how to
download them, and how to serve them correctly.

Model hub: <https://huggingface.co/NagusameCS/ultratensor-models>
(model card auto-synced from this page by
`.github/workflows/hf-modelcard-sync.yml`).

## What these models are

They are "keep-N" expert splices of the public BatiAI DeepSeek-V4-Pro Q3_K_M
GGUF (697.3 GB, 17 shards), built with `ultratensor/gguf_keep.py`. A splice
keeps only the expert subset a domain actually uses:

- **keep64** — 64 dense experts / 384 hash experts (156.1 GiB). Holds 97.66 %
  of code routing mass on the first dense layer.
- **keep16u / keep12u / keep8u** — uniform 16/12/8-expert ladder splices
  (39.3 / 32.2 / 25.0 GiB).
- **IQ2_XS requants** of the ladder (24.7 / 20.3 / 16.0 GiB, 2.36-2.38 BPW)
  for 8 GB GPUs.

All files are single-file GGUFs (split count patched), with the MTP metadata
defect fixed (`nextn_predict_layers = 0`). They load natively in llama.cpp
builds with DeepSeek-V4 support (b10424 line).

**Licensing**: these files are derived from DeepSeek-V4-Pro weights. Verify
the upstream DeepSeek model license and the BatiAI distribution terms before
redistributing further.

## Download

```powershell
pip install huggingface_hub
huggingface-cli login          # once

python scripts/download_models.py --dest models
python scripts/download_models.py --model keep16u-iq2xs --dest models
python scripts/download_models.py --include-keep64 --dest models   # 156 GiB
```

## Serving

**`--no-op-offload` is mandatory on every Q3_K tier.** The engine's default
op-offload places Q3_K dequant kernels of CPU-resident tensors on the GPU,
which crashes at >=36 prompt tokens and silently corrupts short-prompt
outputs. The IQ2_XS GPU tiers are also served with `--no-op-offload`.

CPU quality tier (Q3_K_M):

```powershell
llama-server -m models\DeepSeek-V4-Coder-keep16u.gguf --host 127.0.0.1 --port 8780 -ngl 0 -c 512 --no-op-offload
```

GPU speed tier (IQ2_XS, 8 GB VRAM):

```powershell
llama-server -m models\DeepSeek-V4-Coder-keep16u-iq2xs.gguf --host 127.0.0.1 --port 8791 -ngl 12 -c 512 --no-op-offload
```

Long-sequence GPU decode (>45 tokens) requires the UltraTensor llama.cpp fork
(the MMQ broadcast bypass + the cuBLAS fallback fix, commit `95fcdad`); stock
b10424 builds serve short prompts and CPU fine.

## Measured quality (PPL battery, temperature 0)

| Tier | Mean PPL | Notes |
|---|---|---|
| keep16u Q3_K_M, CPU | **2.653** | code 3.000, math 2.887, multilingual 2.629, rare 2.536, needle 2.215 (8 tokens) |
| keep16u IQ2_XS, GPU | **8.540** | code 8.528, math 8.631, multilingual 8.458, rare 8.568, needle 8.517 (16 tokens); degenerate token loops on every domain |

The IQ2_XS ladder is a *speed* tier (1.2-1.5 tok/s on 8 GB VRAM), not a
*quality* tier. CPU Q3_K remains the quality baseline. See the paper
(`docs/paper/ultratensor.tex`) and `docs/PREPRINT.md` for the full story.

## Catalog

The machine-readable catalog is `ultratensor/model_catalog.json`. The
`download_models.py` and `upload_models.py` scripts share it.
