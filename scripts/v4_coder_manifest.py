"""V4-Coder step 7 — extraction manifest from the census.

Turns code_census.json into the Phase-2 build spec: per traced layer
the keep64 expert ids, code-exclusive vs shared split, the cold
fallback list (routed experts outside top-64), and per-layer size
estimates. This manifest is the input to the requant/build step.

Writes outputs/coder_manifest.json.

Usage:
    python scripts/v4_coder_manifest.py --census outputs/code_census.json
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EXPERT_PARAMS = 3072 * 7168 * 3
IQ2_XS_BITS = 2.3125
Q3_K_BITS = 3.5
N_LAYERS = 61


def gb(params, bits):
    return round(params * bits / 8 / 1e9, 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--census", default=str(ROOT / "outputs" /
                                            "code_census.json"))
    a = ap.parse_args()

    census = json.load(open(a.census, encoding="utf-8"))
    manifest = {"source": a.census, "layers": {}, "totals": {}}
    keep_total = 0
    for key, L in sorted(census.get("layers", {}).items()):
        top64 = L.get("top64_ids", [])
        exclusive = set(L.get("code_exclusive_top64", []))
        manifest["layers"][key] = {
            "routing": "hash" if key in ("L0", "L1", "L2") else "dense",
            "keep64_ids": top64,
            "n_code_exclusive": sum(1 for e in top64 if e in exclusive),
            "cold_fallback_hint": (
                "hash: keep the full 9.3MB table"
                if key in ("L0", "L1", "L2") else
                f"routed-outside-top64 fallback via rho ladder"),
            "distinct_experts": L.get("distinct_experts"),
            "top64_mass_share": L.get("top64_mass_share"),
        }
        keep_total += len(top64)
    manifest["totals"] = {
        "keep64_experts_per_layer": keep_total // max(len(manifest["layers"]), 1),
        "extrapolated_61_layers_q3_k_gb": gb(
            keep_total // max(len(manifest["layers"]), 1)
            * N_LAYERS * EXPERT_PARAMS, Q3_K_BITS),
        "extrapolated_61_layers_iq2_xs_gb": gb(
            keep_total // max(len(manifest["layers"]), 1)
            * N_LAYERS * EXPERT_PARAMS, IQ2_XS_BITS),
        "plus_router_controller_gb": 0.02,
        "note": ("dense-layer census is from L3 only; L4-L60 assumed "
                 "census-like (exp_mid trace will verify)"),
    }
    out = ROOT / "outputs" / "coder_manifest.json"
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
