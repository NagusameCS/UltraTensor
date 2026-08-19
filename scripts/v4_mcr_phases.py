"""G8 first slice — MCR phase detection on real hidden-state trajectories.

The review's G8 demands trajectory-backed context decisions. The cheap
first slice computable from the traces we already have: per-layer
hidden-state variance over the 24 real tokens (blocks 0-3) fed to the
MCR phase detector (Mix/Compress/Refine, ultratensor.conditional.sinks)
with its 1.15 variance-ratio validity gate.

Caveat (honest): only 4 layers are traced so far — a first datum, not
a phase map. The attention-mass-per-slot part of G8 (CSA/HCA sink
placement) still needs attention forwards; MCR is the hidden-state side.

Writes outputs/mcr_phases.json.

Usage:
    python scripts/v4_mcr_phases.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from ultratensor.conditional.sinks import mcr_detect_phases  # noqa: E402

INPUTS = ROOT / "outputs" / "ffn_inputs_dense.npz"


def main() -> int:
    data = np.load(INPUTS)
    per_layer = {}
    for key in ("L0", "L1", "L2", "L3"):
        if key not in data:
            continue
        X = np.asarray(data[key], dtype=np.float64)   # [24, 7168]
        per_layer[key] = {
            "mean_hidden_norm": round(float(
                np.linalg.norm(X, axis=-1).mean()), 2),
            "mean_activation_variance": round(float(
                X.var(axis=0).mean()), 6),
        }

    keys = sorted(per_layer)
    var = np.array([per_layer[k]["mean_activation_variance"]
                    for k in keys])
    m = mcr_detect_phases(var)
    report = {
        "layers": keys,
        "per_layer": per_layer,
        "variance_ratio": round(m.var_ratio, 4),
        "phases_valid": m.phases_valid,
        "phases": m.phases,
        "compress_zone": [m.compress_start, m.compress_end],
        "note": ("4-layer first datum; attention-mass-per-slot (CSA/HCA "
                 "sink placement) still needs attention forwards"),
    }
    out = ROOT / "outputs" / "mcr_phases.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
