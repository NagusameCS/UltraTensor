"""G2: activation-space expert similarity on REAL V4 bytes.

The reviews demand expert similarity in activation-output space:
sim(e_i, e_j) = E_x cos(y_ei(x), y_ej(x)), NOT weight-space
correlation (which we already measured at < 1e-3). This script uses the
real MoE input hidden states saved by v4_router_trace_dense.py and runs
a fixed probe set of experts on ALL of them, then reports the pairwise
cosine-similarity matrix of the outputs plus the weight-space
correlation of the same experts for contrast.

Usage (after v4_router_trace_dense.py has produced its npz):
    python scripts/v4_expert_sim.py --layers 0 3 --probe 16
"""

import argparse
import glob
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import v4_ref_serve as vs  # noqa: E402

INPUTS = ROOT / "outputs" / "ffn_inputs_dense.npz"
KINDS = ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps")


def expert_outputs(st, layer, e, inputs):
    """y_e(x) = silu(x @ gate^T) * (x @ up^T) @ down^T for all inputs."""
    gate = st.read_expert(layer, "ffn_gate_exps", e)
    up = st.read_expert(layer, "ffn_up_exps", e)
    down = st.read_expert(layer, "ffn_down_exps", e)
    outs = []
    for x in inputs:
        g = x @ gate.T
        silu = g / (1.0 + np.exp(-g))
        o = (silu * (x @ up.T)) @ down.T
        outs.append(o)
    return np.stack(outs)


def pairwise_cos(mats):
    """mean cosine sim between outputs of expert pairs, on shared inputs."""
    E = len(mats)
    sim = np.zeros((E, E), np.float64)
    for i in range(E):
        for j in range(E):
            a, b = mats[i], mats[j]
            na = np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12
            nb = np.linalg.norm(b, axis=-1, keepdims=True) + 1e-12
            sim[i, j] = float(np.mean(np.sum(a * b, axis=-1) / (na * nb).ravel()))
    return sim


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="+", default=[0, 3])
    ap.add_argument("--probe", type=int, default=16)
    a = ap.parse_args()

    if not INPUTS.exists():
        print("run scripts/v4_router_trace_dense.py first")
        return 2
    data = np.load(INPUTS)
    shards = sorted(glob.glob("D:/hyperv4/models/pro/"
                              "deepseek-ai-DeepSeek-V4-Pro-Q3_K_M-*.gguf"))
    st = vs.ExpertStore(shards[0], extra_shards=shards[1:])

    report = {}
    t0 = time.time()
    for layer in a.layers:
        key = f"L{layer}"
        if key not in data:
            print(f"{key} missing from npz; skip")
            continue
        inputs = [x.astype(np.float32) for x in data[key][:24]]
        experts = list(range(a.probe))
        mats, wc = [], []
        for e in experts:
            y = expert_outputs(st, layer, e, inputs)
            mats.append(y)
            W = st.read_expert(layer, "ffn_down_exps", e).astype(np.float64)
            wc.append((W / (np.linalg.norm(W) + 1e-12)).ravel())
            print(f"layer {layer} expert {e} done ({time.time() - t0:.0f}s)",
                  flush=True)
        sim = pairwise_cos(mats)
        wcorr = np.array([[float(a @ b) for b in wc] for a in wc])
        off = ~np.eye(len(experts), dtype=bool)
        report[str(layer)] = {
            "mean_offdiag_act_cos": float(sim[off].mean()),
            "max_offdiag_act_cos": float(sim[off].max()),
            "mean_offdiag_weight_corr": float(wcorr[off].mean()),
            "max_offdiag_weight_corr": float(wcorr[off].max()),
        }
        print(f"layer {layer}: act_cos mean={sim[off].mean():.4f} "
              f"max={sim[off].max():.4f} | weight_corr "
              f"mean={wcorr[off].mean():.2e}", flush=True)

    out = ROOT / "outputs" / "expert_sim_activation.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out} ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
