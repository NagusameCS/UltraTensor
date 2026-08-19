"""G4 exploitation: the deployable projector form, held-out validated.

The in-sample k95_act=8 finding says routed inputs live in a tiny
subspace of the expert row-space. Deployment form per expert:

    A = W @ V_k            [m, k]  one-time precompute
    y ~= A (V_k^T x)       k*(m+d) MACs instead of m*d

This script fits V_k on a training split of REAL routed hidden states
and evaluates on the held-out split — the number any kernel work must
beat. It compares the plain PCA projector against the activation-
weighted (whitened) oblique projector, both fitted on train only, and
reports the all-samples in-sample fit for continuity with k95_act.

Usage:
    python scripts/v4_subspace_proj.py --layers 0 3 --experts 0 1
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

from ultratensor.conditional.actweight import heldout_rank_curves  # noqa: E402
from ultratensor.expert_store import ExpertStore  # noqa: E402

INPUTS = ROOT / "outputs" / "ffn_inputs_dense.npz"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="+", default=[0, 3])
    ap.add_argument("--experts", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--train", type=int, default=16)
    ap.add_argument("--ranks", type=int, nargs="+",
                    default=[1, 2, 4, 8, 12, 16, 24])
    ap.add_argument("--kind", default="ffn_gate_exps",
                    help="ffn_gate_exps (hidden->intermediate) recommended; "
                         "down_exps needs the 3072-dim intermediate inputs")
    a = ap.parse_args()

    if not INPUTS.exists():
        print("run scripts/v4_router_trace_dense.py first")
        return 2
    data = np.load(INPUTS)
    shards = sorted(glob.glob("D:/hyperv4/models/pro/"
                              "deepseek-ai-DeepSeek-V4-Pro-Q3_K_M-*.gguf"))
    st = ExpertStore(shards[0], extra_shards=shards[1:])

    report = {}
    t0 = time.time()
    for layer in a.layers:
        key = f"L{layer}"
        if key not in data:
            print(f"{key} missing from npz; skip")
            continue
        X = np.asarray(data[key][:24], dtype=np.float64)
        n = min(a.train, X.shape[0] - 1)
        for e in a.experts:
            W = st.read_expert(layer, a.kind, e).astype(np.float64)
            fits = {
                "fwd": (X[:n], X[n:]),   # first n train, rest hold
                "rev": (X[-n:], X[:-n]),  # last n train, rest hold
                "all": (X, X),            # in-sample continuity w/ k95_act
            }
            entry = {"shape": list(W.shape)}
            for name, (tr, ho) in fits.items():
                ranks = [r for r in a.ranks if r <= tr.shape[0]]
                c = heldout_rank_curves(W, tr, ho, ranks=ranks)
                entry[name] = {
                    "ranks": c["ranks"].tolist(),
                    "hold_pca": [round(float(x), 6) for x in c["hold_pca"]],
                    "hold_weighted":
                        [round(float(x), 6) for x in c["hold_weighted"]],
                }
            report[f"{layer}/{e}"] = entry
            i8 = entry["fwd"]["ranks"].index(8)
            print(f"L{layer}/e{e}: fwd  hold_pca@8={entry['fwd']['hold_pca'][i8]:.4f}"
                  f" hold_w@8={entry['fwd']['hold_weighted'][i8]:.4f} | "
                  f"rev  hold_pca@8={entry['rev']['hold_pca'][i8]:.4f} | "
                  f"all-in-sample pca@8={entry['all']['hold_pca'][i8]:.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)

    out = ROOT / "outputs" / "subspace_proj.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
