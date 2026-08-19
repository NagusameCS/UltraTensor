"""G5 companion — the four-policy comparator on REAL layer signals.

rank_policies.compare_policies (fixed/frank/mcr/grcurv) has only run
on synthetic inputs. This feeds it the REAL per-layer activation
variance from the exp96 trace (blocks 0-3) and reports the rank
allocations + Gini concentration each policy would choose — the first
real-data exercise of the policy layer.

Writes outputs/rank_policies_real.json.

Usage:
    python scripts/v4_rank_policies_real.py \
        --inputs outputs/exp96_ffn_inputs_dense.npz
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from ultratensor.conditional.rank_policies import compare_policies  # noqa: E402

INPUTS = ROOT / "outputs" / "exp96_ffn_inputs_dense.npz"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default=str(INPUTS))
    ap.add_argument("--budget", type=int, default=1024)
    a = ap.parse_args()

    data = np.load(a.inputs)
    layers, var, norms = [], [], []
    for key in ("L0", "L1", "L2", "L3"):
        if key not in data:
            continue
        X = np.asarray(data[key], dtype=np.float64)
        layers.append(key)
        var.append(float(X.var(axis=0).mean()))
        norms.append(float(np.linalg.norm(X, axis=-1).mean()))

    frob_err = np.full(len(layers), 0.1)       # weight-spectra are flat (G4)
    sectional = (np.array(var) - min(var)) / max(max(var) - min(var), 1e-9)
    res = compare_policies(
        n_layers=len(layers), total_budget=a.budget, min_rank=16,
        max_rank=512, frob_err=frob_err,
        act_variance=np.array(var), sectional=sectional)
    res["inputs"] = a.inputs
    res["layers"] = layers
    res["act_variance"] = [round(v, 6) for v in var]
    res["hidden_norm"] = [round(n, 2) for n in norms]
    res["note"] = ("real exp96 per-layer activation variance; frob_err "
                   "flat 0.1 (G4: weight spectra flat); sectional from "
                   "normalized variance")
    out = ROOT / "outputs" / "rank_policies_real.json"
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
