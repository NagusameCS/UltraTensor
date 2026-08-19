"""TPJ honesty demo: does thermal rank apply to OUR decode workload?

Runs the real MoE decode for a few tokens while polling NVML. Our
decode is CPU + SSD bound; the GPU idles at ~1-2 W. Expected honest
result: TPJ rank_coeff stays ~0 because GPU joules/token is negligible,
so the thermal policy correctly does nothing for THIS bottleneck. The
thermal mechanism is for GPU-bound paths (CUDA kernels), not this one.

Usage: python scripts/tpj_honesty_demo.py [n_tokens]
"""

import glob
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

import v4_ref_serve as vs  # noqa: E402
from ultratensor.conditional import NvmlCtypesSensor, ThermalRank  # noqa: E402
from ultratensor.conditional import TpjTracker  # noqa: E402


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    shards = sorted(glob.glob("D:/hyperv4/models/pro/"
                              "deepseek-ai-DeepSeek-V4-Pro-Q3_K_M-*.gguf"))
    st = vs.ExpertStore(shards[0], extra_shards=shards[1:])
    ml = vs.MoELayer(st, 0)
    rng = np.random.default_rng(0)
    hiddens = rng.standard_normal((n, 7168), np.float32)
    token_ids = rng.integers(0, 100000, n)

    sensor = NvmlCtypesSensor()
    thermal = ThermalRank(sensor)
    tpj = TpjTracker(thermal)
    times = []
    for i in range(n):
        t0 = time.perf_counter()
        ml(hiddens[i:i + 1], token_ids[i:i + 1])
        times.append(time.perf_counter() - t0)
        tps = 1.0 / max(times[-1], 1e-9)
        tpj.record(tps)
    sensor.close()

    gpu_jpt = tpj.cumulative_joules / max(tpj.cumulative_tokens, 1)
    cpu_jpt_est = 90.0 * float(np.mean(times))   # assume ~90 W CPU under load
    gpu_share = gpu_jpt / (gpu_jpt + cpu_jpt_est)
    out = {
        "n_tokens": n,
        "mean_tok_s": round(float(np.mean(times)), 3),
        "gpu_idle_power_W": round(tpj.thermal.current_power_w, 2),
        "rank_coeff_joules_per_rank": round(tpj.rank_coeff, 8),
        "gpu_joules_per_token": round(gpu_jpt, 3),
        "cpu_joules_per_token_est": round(cpu_jpt_est, 3),
        "gpu_energy_share": round(gpu_share, 4),
        "verdict": (
            f"GPU is {gpu_share * 100:.1f}% of decode energy on this "
            "CPU/SSD-bound path; GPU-thermal rank is the wrong knob here "
            "(it targets CUDA-bound paths)") if gpu_share < 0.1 else
            "GPU energy share unexpected: re-check assumptions",
    }
    print(json.dumps(out, indent=2))
    (ROOT / "outputs" / "tpj_honesty_demo.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
