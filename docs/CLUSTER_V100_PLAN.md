# node2 rebuild: dual V100 32GB + 64GB RAM — build spec & day-one plan

Status: planning (2026-08-16). Keeps the Ryzen 9 5900XT (AM4); replaces
motherboard/case/PSU; adds 2x Tesla V100 32GB (PCIe) and 2x32GB DDR4.

## Build spec

- **Motherboard (AM4/X570)**: two physical x16 slots running x8/x8.
  Candidates: ASUS Prime X570-Pro, Gigabyte X570 Aorus Elite,
  ASRock X570 Taichi, MSI X570 Tomahawk. (Avoid B550: second slot is
  chipset x4.)
- **PSU**: 850W quality minimum, 1000W Gold recommended
  (2x250W cards + 105W CPU + transients). Verify each V100's power
  connector (1x 8-pin typical, some need 2).
- **BIOS**: enable Above 4G Decoding; disable CSM. V100 = PCIe Gen3;
  in a Gen4 x8 slot it runs Gen3 x8 (~8GB/s per card) - ample for
  inference decode (weights loaded once, activations tiny).
- **Case/cooling**: V100s are 10.5" blowers exhausting rear; any long
  ATX case, keep slots spaced.
- **Storage**: 476GB NVMe fits Flash-284B (~90GB). For the full q2_0
  expert pool (450GB) local, add a 2TB NVMe.

## Model inventory (as of 2026-08-16)

| Model | On laptop D: | On NAS | Notes |
|---|---|---|---|
| V4-Pro Q3_K_M (17 shards, 697GB) | yes | shards 1-2 (90GB) | full transfer ~2h at 1GbE |
| V4-Pro q2_0 experts (450GB) | certification outputs | no | capacity win, not latency |
| DSpark drafter GGUFs (2x 24.8GB) | yes | Y:/models/drafter | spec decode day one |
| V4-Flash-284B-IQ2XXS GGUF | — | **Y:/models/flash (91GB, DONE 2026-08-16)** | mrtib imatrix mix, 2 parts |
| V4-Coder keep64 (156.1GB) | — | **Y:/models/coder (built 2026-08-16)** | 64 experts/dense layer |

## Day-one plan (after hardware boots with Ubuntu 26.04)

1. Install NVIDIA driver (550+), CUDA 12.x toolkit; verify
   `nvidia-smi` shows both V100s (sm_70, supported by CUDA 12).
2. `git clone` llama.cpp; build with CUDA; run 14B Qwen benchmark
   (expect 40-80 tok/s; sanity check for the platform).
3. Copy Flash-284B IQ2XXS GGUF to NVMe; benchmark:
   - no GPU offload (CPU baseline),
   - `-ngl 999` both cards (expect 25-60 tok/s),
   - add the DSpark drafter (speculative) — target >60.
4. Wire our prefetch controller + benchmark harness from
   `ultratensor.conditional` (GPU at last makes the thermal-rank
   toolkit live: V100s DO throttle).
5. Then Pro: NVMe-cached shards + GPU tile-decode; batch serving with
   the prefetch controller; measure the new aggregate ceiling vs the
   laptop's 0.708 tok/s.

## Expectations (honest, order-of-magnitude)

- Flash-284B: 25-60 tok/s -> V4-class reasoning interactive; agent
  cycles ~30-90s -> viable autonomous coding.
- Pro: still disk-bound; batch aggregate rises from 0.708.
- 14B Qwen: 40-80 tok/s Copilot-lite.

## Risks

- V100 = 2017 silicon: no FP8/BF16; CUDA-core-bound IQ kernels (fine).
- Two 250W blowers = noise; keep the box out of living space.
- SXM2 32GB cards are cheap but need server carriers — buy PCIe only.
