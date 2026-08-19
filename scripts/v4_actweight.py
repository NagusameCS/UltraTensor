"""G4 on REAL bytes: activation-weighted vs Frobenius rank-error curves.

Uses the real MoE input hidden states from v4_router_trace_dense.py and
real expert tensors from the V4-Pro Q3_K_M shards. For each probed
expert it reports:

- frob curve:    ||W - W_r||_F^2 / ||W||_F^2
- act curve:     E_x ||(W - W_r)x||^2 / E_x ||Wx||^2   (plain SVD)
- act-weighted:  same with the activation-weighted truncation
- k95_frob vs k95_act

This is the review's E_{l,e,s}(r) measured on the actual routed
distribution — the number we never had before.

Usage (after v4_router_trace_dense.py):
    python scripts/v4_actweight.py --layers 0 3 --experts 0 1 --rank-max 1536
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
from ultratensor.conditional.actweight import rank_error_curve  # noqa: E402

INPUTS = ROOT / "outputs" / "ffn_inputs_dense.npz"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="+", default=[0, 3])
    ap.add_argument("--experts", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--rank-max", type=int, default=1536)
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
    st = vs.ExpertStore(shards[0], extra_shards=shards[1:])

    ranks = [128, 256, 384, 512, 768, 1024, 1280, 1536]
    report = {}
    t0 = time.time()
    for layer in a.layers:
        key = f"L{layer}"
        if key not in data:
            print(f"{key} missing from npz; skip")
            continue
        inputs = np.asarray(data[key][:24], dtype=np.float64)
        for e in a.experts:
            W = st.read_expert(layer, a.kind, e).astype(np.float64)
            ranks_l = [r for r in ranks if r <= min(W.shape) and r <= a.rank_max]
            c = rank_error_curve(W, inputs, ranks=ranks_l)
            report[f"{layer}/{e}"] = {
                "shape": list(W.shape),
                "ranks": c.ranks.tolist(),
                "frob": [round(float(x), 6) for x in c.frob],
                "act": [round(float(x), 6) for x in c.act],
                "act_weighted": [round(float(x), 6) for x in c.act_weighted],
                "k95_frob": c.k95_frob,
                "k95_act": c.k95_act,
            }
            print(f"layer {layer} expert {e}: k95_frob={c.k95_frob} "
                  f"k95_act={c.k95_act}  "
                  f"act@512={c.act[ranks_l.index(512)]:.4f} "
                  f"frob@512={c.frob[ranks_l.index(512)]:.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)

    out = ROOT / "outputs" / "actweight_curves.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
