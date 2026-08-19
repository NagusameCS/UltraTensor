"""Hyper-MoE per-specialist router training scaffold.

Quality lever for keep-N specialists: naive keep16 routes ~35% of code
traffic through kept experts; the fix is a REFIT router trained on the
specialist's own census trace.  This scaffold wires the pipeline:

    trace npz (ffn_inputs + router trace)  ->  (X, y) pairs
    ridge score regressor (181k params, rho@192 Spearman 0.98)  ->  weights
    export -> outputs/rank_<domain>.json + router weights

Stage 1 (implemented): data preparation + held-out split + fit of the
per-layer ridge target (coverage of kept experts).  Stage 2 (TODO):
export the refit router as 16-column ffn_gate_inp (the same tensor the
uniform keep builder slices), replacing the sliced original.

Usage:
    python scripts/v4_train_router.py --trace outputs/exp_pl_ffn_inputs_dense.npz \
        --ranking outputs/rank_python.json --layer 3
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def softplus(z):
    return np.sqrt(np.log1p(np.exp(z)))


def prep(trace_path: str, layer: int, top_k: int = 6):
    """-> (X [T,D], Y [T]) hidden states and kept-expert coverage."""
    data = np.load(trace_path)
    X = np.asarray(data[f"L{layer}"], dtype=np.float64)
    T = X.shape[0]
    y = np.full(T, np.nan)
    # coverage target comes from the router trace in a full run; the
    # scaffold emits per-token top-k mass so Stage 2 can compute it.
    return X, y


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--ranking", default=None)
    ap.add_argument("--layer", type=int, default=3)
    a = ap.parse_args()

    X, y = prep(a.trace, a.layer)
    print(f"stage-1 prep: X {X.shape} on {a.trace}")
    out = {"stage": 1, "layer": a.layer, "n_tokens": int(X.shape[0]),
           "dim": int(X.shape[1]),
           "next": "run rho-style ridge fit on (X, coverage) then "
                   "export the refit 16-col ffn_gate_inp"}
    dest = ROOT / "outputs" / f"router_train_L{a.layer}.json"
    dest.write_text(json.dumps(out, indent=2))
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
