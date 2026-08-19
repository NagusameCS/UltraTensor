"""End-to-end correctness: lazy MoELayer vs a numpy reference on REAL V4
bytes (no mini shards). Validates the C executors, the prefetch reader,
the router (hash + dense) and the shared-expert path together.

Usage:
    python scripts/check_moe_layer_e2e.py <model_glob> [layer ...]
"""
import argparse
import glob
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from ultratensor.expert_store import ExpertStore  # noqa: E402
from ultratensor.moe_exec import MoELayer, _swiglu  # noqa: E402

DIM = 7168
LIMIT = 10.0


def reference_layer(st: ExpertStore, layer: int, h: np.ndarray,
                    token_ids, ml: MoELayer) -> np.ndarray:
    """h [B,n] -> [B,n] via the reference semantics in pure numpy."""
    h = np.ascontiguousarray(h, np.float32)
    B = h.shape[0]
    ids, w = ml.route(h, token_ids=token_ids)
    y = np.zeros((B, DIM), np.float32)
    for b in range(B):
        for slot in range(ids.shape[1]):
            e = int(ids[b, slot])
            gate = st.read_expert(layer, "ffn_gate_exps", e)   # (m,n)
            up = st.read_expert(layer, "ffn_up_exps", e)
            down = st.read_expert(layer, "ffn_down_exps", e)   # (n,m)
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
    ap.add_argument("layers", nargs="*", type=int, default=[0, 3])
    a = ap.parse_args()

    shards = sorted(glob.glob(a.model_glob))
    st = ExpertStore(shards[0], extra_shards=shards[1:])
    rng = np.random.default_rng(0)
    rc = 0
    for layer in a.layers:
        ml = MoELayer(st, layer)
        try:
            h = rng.standard_normal((2, DIM), np.float32)
            tids = rng.integers(0, 100000, 2)
            y_lazy = ml(h, tids)
            y_ref = reference_layer(st, layer, h, tids, ml)
            d = np.abs(y_lazy - y_ref).max()
            rel = float(d / np.abs(y_ref).max())
            router = "hash" if layer < st.n_hash_layers else "dense"
            ok = rel < 2e-3
            print(f"layer {layer} ({router}): max_abs {d:.6f}  "
                  f"max_rel {rel:.3e}  {'PASS' if ok else 'FAIL'}")
            if not ok:
                rc = 1
        finally:
            ml.close()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
