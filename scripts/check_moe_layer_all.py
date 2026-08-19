"""Overnight follow-on: run the real-bytes e2e check over ALL 61 layers.

Waits for the timing sweep (outputs/bench_moe_sweep.json reaching
layers_done == layers_total) so the two jobs never contend for the disk,
then checks every layer with gate_exps against the numpy reference.
Incremental JSON (outputs/e2e_all_layers.json); a partial run resumes.

Usage:
    python scripts/check_moe_layer_all.py <model_glob>
"""
import argparse
import glob
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from ultratensor.expert_store import ExpertStore  # noqa: E402
from ultratensor.moe_exec import MoELayer, _swiglu  # noqa: E402

DIM = 7168
LIMIT = 10.0
OUT = ROOT / "outputs" / "e2e_all_layers.json"
SWEEP = ROOT / "outputs" / "bench_moe_sweep.json"


def reference_layer(st, layer, h, token_ids, ml):
    h = np.ascontiguousarray(h, np.float32)
    B = h.shape[0]
    ids, w = ml.route(h, token_ids=token_ids)
    y = np.zeros((B, DIM), np.float32)
    for b in range(B):
        for slot in range(ids.shape[1]):
            e = int(ids[b, slot])
            gate = st.read_expert(layer, "ffn_gate_exps", e)
            up = st.read_expert(layer, "ffn_up_exps", e)
            down = st.read_expert(layer, "ffn_down_exps", e)
            g = gate @ h[b]
            u = up @ h[b]
            g = np.minimum(g, LIMIT)
            u = np.clip(u, -LIMIT, LIMIT)
            s = g / (1.0 + np.exp(-g))
            y[b] += float(w[b, slot]) * (down @ (s * u))
    g = h @ st.read_tensor(layer, "ffn_gate_shexp").T
    u = h @ st.read_tensor(layer, "ffn_up_shexp").T
    y += _swiglu(g, u) @ st.read_tensor(layer, "ffn_down_shexp").T
    return y


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_glob")
    ap.add_argument("--max_wait_s", type=int, default=3 * 3600)
    a = ap.parse_args()

    # 1. wait for the timing sweep to finish
    t0 = time.time()
    while time.time() - t0 < a.max_wait_s:
        if SWEEP.exists():
            d = json.loads(SWEEP.read_text())
            if d.get("layers_done", 0) >= d.get("layers_total", 0) > 0:
                print(f"timing sweep complete ({d['layers_done']} layers); "
                      f"starting e2e checks", flush=True)
                break
        time.sleep(60)
    else:
        print("timing sweep did not complete in time; proceeding anyway",
              flush=True)

    results = {}
    if OUT.exists():
        results = json.loads(OUT.read_text()).get("per_layer", {})

    shards = sorted(glob.glob(a.model_glob))
    st = ExpertStore(shards[0], extra_shards=shards[1:])
    layers = [L for L in st.layers()
              if f"blk.{L}.ffn_gate_exps.weight" in st.tensors]
    rng = np.random.default_rng(0)
    h = rng.standard_normal((2, DIM), np.float32)
    tids = rng.integers(0, 100000, 2)

    for layer in layers:
        if str(layer) in results:
            continue
        ml = MoELayer(st, layer)
        try:
            y_lazy = ml(h, tids)
            y_ref = reference_layer(st, layer, h, tids, ml)
            rel = float(np.abs(y_lazy - y_ref).max() / np.abs(y_ref).max())
            results[str(layer)] = {
                "layer": layer,
                "router": "hash" if layer < st.n_hash_layers else "dense",
                "max_rel": rel,
                "pass": rel < 2e-3,
            }
        except Exception as e:  # keep the sweep alive on any layer error
            results[str(layer)] = {"layer": layer, "error": repr(e)}
        finally:
            ml.close()
        n_ok = sum(1 for v in results.values() if v.get("pass"))
        print(f"[{len(results)}/{len(layers)}] layer {layer} "
              f"rel={results[str(layer)].get('max_rel', 'ERR')} "
              f"({n_ok} pass)", flush=True)
        OUT.write_text(json.dumps(
            {"layers_done": len(results), "layers_total": len(layers),
             "pass": n_ok, "per_layer": results}, indent=2))
    print(f"done: {sum(1 for v in results.values() if v.get('pass'))}/"
          f"{len(layers)} pass -> {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
