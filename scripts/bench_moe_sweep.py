"""Overnight sweep: lazy MoE-layer timing for EVERY V4-Pro layer.

Runs scripts/bench_moe_layer.py for each layer with gate_exps in its own
subprocess (a per-layer crash costs one data point, not the whole sweep),
then aggregates into outputs/bench_moe_sweep.json. Progress is printed and
the aggregate is written incrementally so a partial run still yields data.

Usage:
    python scripts/bench_moe_sweep.py <model_glob> [n_tokens=100]
"""
import argparse
import glob
import json
import subprocess
import sys
import time
from pathlib import Path

PY = sys.executable
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / "outputs"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_glob")
    ap.add_argument("n_tokens", type=int, default=100)
    a = ap.parse_args()

    sys.path.insert(0, str(ROOT))
    from ultratensor.expert_store import ExpertStore

    shards = sorted(glob.glob(a.model_glob))
    st = ExpertStore(shards[0], extra_shards=shards[1:])
    layers = [L for L in st.layers()
              if f"blk.{L}.ffn_gate_exps.weight" in st.tensors]
    print(f"sweep: {len(layers)} layers x {a.n_tokens} tokens, "
          f"hash layers < {st.n_hash_layers}", flush=True)

    results = {}
    out_path = OUT / "bench_moe_sweep.json"
    t_start = time.time()
    for i, layer in enumerate(layers):
        t0 = time.time()
        rc = subprocess.run(
            [PY, str(HERE / "bench_moe_layer.py"), a.model_glob,
             str(layer), str(a.n_tokens)],
            capture_output=True, text=True, cwd=str(ROOT))
        per = OUT / f"bench_moe_layer_{layer}.json"
        entry = None
        if rc.returncode == 0 and per.exists():
            entry = json.loads(per.read_text())
            results[layer] = entry
            line = (f"[{i + 1}/{len(layers)}] layer {layer} "
                    f"{entry['router']:5s} {entry['s_per_token_layer']:.4f}s "
                    f"({time.time() - t0:.1f}s)")
        else:
            results[layer] = {"layer": layer, "error": rc.returncode,
                              "stderr": rc.stderr[-200:]}
            line = (f"[{i + 1}/{len(layers)}] layer {layer} FAILED rc="
                    f"{rc.returncode} ({time.time() - t0:.1f}s)")
        print(line, flush=True)
        good = [v for k, v in results.items() if "s_per_token_layer" in v]
        if good:
            mean = sum(v["s_per_token_layer"] for v in good) / len(good)
            n_hash = sum(1 for k, v in results.items()
                         if "s_per_token_layer" in v and k < st.n_hash_layers)
            n_dense = len(good) - n_hash
            # per-token cost over 61 layers: hash cost x n_hash_layers_actual
            # + dense cost x remaining (model has 61 layers, 3 hash)
            h_mean = (sum(v["s_per_token_layer"] for k, v in
                          results.items()
                          if "s_per_token_layer" in v and
                          k < st.n_hash_layers) / max(n_hash, 1))
            d_mean = (sum(v["s_per_token_layer"] for k, v in
                          results.items()
                          if "s_per_token_layer" in v and
                          k >= st.n_hash_layers) /
                      max(n_dense, 1))
            proj = 1.0 / (3 * h_mean + 58 * d_mean)
            out_path.write_text(json.dumps(
                {"layers_done": len(good), "layers_total": len(layers),
                 "n_hash_layers": st.n_hash_layers,
                 "mean_s_per_layer": mean,
                 "hash_mean_s": h_mean, "dense_mean_s": d_mean,
                 "projected_tok_s_61_layers": proj,
                 "elapsed_s": time.time() - t_start,
                 "per_layer": results}, indent=2))
    print(f"done in {time.time() - t_start:.0f}s -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
