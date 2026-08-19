"""Lazy MoE-layer benchmark for one real V4-Pro layer (multi-shard).

Usage:
    python scripts/bench_moe_layer.py <model_glob> <layer> [n_tokens]

Opens header-only inventories for ALL shards, builds MoELayer for the
layer (hash or dense router), and measures the per-token-layer decode
time with the prefetch-overlapped C executor. Writes
outputs/bench_moe_layer_<layer>.json.
"""
import argparse
import glob
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from ultratensor.expert_store import ExpertStore
from ultratensor.moe_exec import MoELayer

DIM = 7168


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_glob")
    ap.add_argument("layer", type=int)
    ap.add_argument("n_tokens", type=int, default=20)
    a = ap.parse_args()

    shards = sorted(glob.glob(a.model_glob))
    if not shards:
        print(f"no shards match {a.model_glob}")
        return 2
    print(f"{len(shards)} shards; inventorying headers...")
    t0 = time.perf_counter()
    st = ExpertStore(shards[0], extra_shards=shards[1:])
    print(f"inventory done in {time.perf_counter() - t0:.2f}s; "
          f"layers={st.layers()[:6]}... (hash layers < {st.n_hash_layers})")

    name = f"blk.{a.layer}.ffn_gate_exps.weight"
    if name not in st.tensors:
        print(f"layer {a.layer}: no gate_exps in any shard")
        return 3
    t = st.tensors[name]
    print(f"layer {a.layer} gate_exps in shard {shards[t['shard']]}")
    print(f"  router: {'hash' if a.layer < st.n_hash_layers else 'dense'}")

    ml = MoELayer(st, a.layer)
    rng = np.random.default_rng(0)
    hiddens = rng.standard_normal((a.n_tokens, DIM), np.float32)
    token_ids = rng.integers(0, 100000, a.n_tokens)

    # warm-up (builds C executors, warms caches)
    ml(hiddens[0:1], token_ids[0:1])
    ml(hiddens[1:2], token_ids[1:2])

    times = []
    for i in range(a.n_tokens):
        t1 = time.perf_counter()
        ml(hiddens[i:i + 1], token_ids[i:i + 1])
        times.append(time.perf_counter() - t1)
    times = np.asarray(times)
    per_layer = float(times.mean())
    result = {
        "layer": a.layer,
        "router": "hash" if a.layer < st.n_hash_layers else "dense",
        "n_tokens": a.n_tokens,
        "s_per_token_layer": per_layer,
        "s_per_token_layer_std": float(times.std()),
        "projected_tok_s_61_layers": 1.0 / (per_layer * 61),
        "min": float(times.min()), "max": float(times.max()),
        "times": [float(x) for x in times],
    }
    print(json.dumps(result, indent=2))
    out = Path("outputs") / f"bench_moe_layer_{a.layer}.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
