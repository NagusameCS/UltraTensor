"""G12 — tiered-residency sweep on the real layer-3 routing sequence.

Reconstructs the per-token top-6 expert sequence for layer 3 (24 real
tokens, outputs/ffn_inputs_dense.npz + router on D:), then sweeps the
hot cap with prefetch policies:

  oracle   : the actual next set (upper bound)
  freq-lru : most-frequent experts seen so far (cheap predictor)

Cost constants are the measured laptop-path numbers: cold miss 500 ms
(Q3_K shard read + dequant stall), warm prefetch 50 ms. Reports miss
rate, P90 tail misses, mean latency and resident bytes per hot cap.

Usage:
    python scripts/v4_tier_sweep.py
"""

import argparse
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import v4_ref_serve as vs  # noqa: E402
from ultratensor.conditional.tiering import simulate_tier  # noqa: E402

INPUTS = ROOT / "outputs" / "ffn_inputs_dense.npz"
TOP_K = 6
MISS_COST_MS = 500.0        # cold Q3_K read + dequant stall (measured path)
PREFETCH_COST_MS = 50.0     # warm-pool fetch per extra expert
BYTES_PER_EXPERT = 9.6e6    # gate expert 3072x7168 @ ~3.5 bits/param


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=3)
    a = ap.parse_args()

    data = np.load(INPUTS)
    X = np.asarray(data[f"L{a.layer}"][:24], dtype=np.float64)
    shards = sorted(glob.glob("D:/hyperv4/models/pro/"
                              "deepseek-ai-DeepSeek-V4-Pro-Q3_K_M-*.gguf"))
    st = vs.ExpertStore(shards[0], extra_shards=shards[1:])
    Wr = st.read_tensor(a.layer, "ffn_gate_inp")
    bias = st.read_tensor(a.layer, "exp_probs_b")

    S_ = np.sqrt(np.log1p(np.exp(X @ Wr.T))) + bias
    ids = np.argpartition(-S_, TOP_K - 1, axis=-1)[:, :TOP_K]
    seq = [set(int(i) for i in ids[t]) for t in range(ids.shape[0])]

    # oracle: map previous set -> actual next set
    next_map = {}
    for t in range(1, len(seq)):
        next_map.setdefault(frozenset(seq[t - 1]), seq[t])

    def oracle(prev):
        return sorted(next_map.get(frozenset(prev), set()))

    # freq-lru: plan from cumulative frequency up to step t (in order)
    freq = np.zeros(384, dtype=int)
    plans = {}
    for t in range(1, len(seq)):
        for e in seq[t - 1]:
            freq[e] += 1
        plans[t] = [int(e) for e in np.argsort(-freq)]
    step = [0]

    def freq_lru(prev):
        t = step[0]
        step[0] += 1
        return plans.get(t + 1, [])

    hot_caps = (4, 6, 8, 12, 16, 24, 32, 48, 64)
    report = {"layer": a.layer, "n_tokens": len(seq),
              "top_k": TOP_K, "miss_cost_ms": MISS_COST_MS,
              "prefetch_cost_ms": PREFETCH_COST_MS, "caps": {}}
    for name, fn in (("oracle", oracle), ("freq_lru", freq_lru)):
        for cap in hot_caps:
            step[0] = 0
            r = simulate_tier(seq, fn, hot_cap=cap,
                              miss_cost_ms=MISS_COST_MS,
                              prefetch_cost_ms=PREFETCH_COST_MS,
                              per_expert_bytes=BYTES_PER_EXPERT)
            report["caps"][f"{name}/cap{cap}"] = {
                "mean_miss_rate": round(r.mean_miss_rate, 4),
                "p90_missing_experts": round(r.tail_p90_miss, 2),
                "mean_latency_ms": round(r.mean_latency_ms, 1),
                "resident_experts": r.resident_experts,
                "resident_mb": round(r.resident_bytes / 1e6, 1),
            }
        print(f"{name} swept", flush=True)

    out = ROOT / "outputs" / "tier_sweep_L3.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
