"""V4-Coder step 4 — subnetwork extraction planner (Phase 2 tool).

Consumes code_census.json (cluster output) and emits an EXTRACTION
PLAN for the coder subnetwork:

  - top-K code experts per layer (by code routing mass), K sweep;
  - code-exclusive vs shared experts vs the general-traffic set;
  - router projection: keep the router rows of selected experts and
    renormalize softplus scores over the kept set (first order);
  - IQ2_XS size estimate of the kept expert payload;
  - CVaR gate plan: which experts the tail gate marks risky to drop
    (requires the rare-code damage run; placeholder until then).

Writes outputs/coder_extract_plan.json. Pure planning — no model
weights are written; the actual GGUF surgery is a later phase.

Usage:
    python scripts/v4_coder_extract.py --census outputs/code_census.json
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EXPERTS_PER_LAYER = 384
N_LAYERS = 61
EXPERT_PARAMS = 3072 * 7168 * 3   # gate + up + down per expert
Q3_K_BITS = 3.5
IQ2_XS_BITS = 2.3125


def size_gb(params, bits):
    return params * bits / 8 / 1e9


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--census", default=str(ROOT / "outputs" /
                                            "code_census.json"))
    ap.add_argument("--keep", type=int, nargs="+", default=[32, 64, 96])
    ap.add_argument("--layers", type=int, nargs="+",
                    default=[0, 1, 2, 3])
    a = ap.parse_args()

    try:
        census = json.load(open(a.census, encoding="utf-8"))
    except FileNotFoundError:
        print("run the cluster census first (code_census.json)")
        return 2

    plan = {"census": a.census, "keep_options": a.keep, "layers": {}}
    for layer in a.layers:
        L = census.get("layers", {}).get(f"L{layer}")
        if not L:
            continue
        top_ids = L.get("top64_ids", [])
        shared = set(L.get("shared_top64", []))
        excl = L.get("code_exclusive_top64", [])
        row = {"n_code_tokens": L.get("n_tokens"),
               "distinct_experts": L.get("distinct_experts"),
               "top64_mass_share": L.get("top64_mass_share")}
        for K in a.keep:
            kept = top_ids[:K]
            row[f"keep{K}"] = {
                "experts": kept,
                "n_shared_with_general": sum(
                    1 for e in kept if e in shared),
                "n_code_exclusive": sum(
                    1 for e in kept if e in excl),
            }
        plan["layers"][f"L{layer}"] = row

    # size estimate, extrapolated to 61 layers (census layers x
    # EXPERTS_PER_LAYER / traced distinct), first order
    sizes = {}
    base_gb = size_gb(EXPERTS_PER_LAYER * N_LAYERS * EXPERT_PARAMS,
                      Q3_K_BITS)
    for K in a.keep:
        kept_params = K * N_LAYERS * EXPERT_PARAMS
        sizes[f"keep{K}"] = {
            "q3_k_gb": round(size_gb(kept_params, Q3_K_BITS), 1),
            "iq2_xs_gb": round(size_gb(kept_params, IQ2_XS_BITS), 1),
            "vs_full_q3_k": round(K / EXPERTS_PER_LAYER, 3),
        }
    plan["size_estimates_61_layers"] = sizes
    plan["full_model_q3_k_gb"] = round(base_gb, 1)
    plan["notes"] = [
        "router projection + renormalization is first-order; needs the "
        "router refit on code traffic for correctness",
        "CVaR tail gate on rare-code damages pending (Phase 2 gate)",
        "size estimate assumes all 61 layers have census-like routing; "
        "only layers 0-3 were traced",
    ]

    # decision gates from the census
    dec = {"concentration_ok": None, "exclusivity": None,
           "subspace_gain": None, "controller_ok": None,
           "verdict": "pending"}
    L3 = census.get("layers", {}).get("L3", {})
    if L3.get("top64_mass_share") is not None:
        dec["concentration_ok"] = bool(L3["top64_mass_share"] >= 0.8)
    dec["exclusivity"] = len(L3.get("code_exclusive_top64", []))
    proj = census.get("code_subspace_proj", {})
    if proj and "L3" in proj:
        rs = proj["L3"].get("ranks", [])
        h = proj["L3"].get("hold_pca", [])
        if 16 in rs:
            h16 = h[rs.index(16)]
            # mixed-traffic reference: exp96 L3 fwd hold_pca@16 mean 0.49
            dec["subspace_gain"] = round(0.49 - h16, 3)
    g9 = census.get("g9_code", {})
    if g9:
        dec["controller_ok"] = bool(g9.get("fwd", {}).get(
            "hold_rel_l1", 1.0) < 0.25)
    if dec["concentration_ok"] is True and dec["exclusivity"] > 0:
        dec["verdict"] = "subnetwork extraction viable"
    elif dec["concentration_ok"] is True:
        dec["verdict"] = "concentrated but not exclusive; distill instead"
    else:
        dec["verdict"] = "no concentration; distillation is the path"
    plan["decision"] = dec

    out = ROOT / "outputs" / "coder_extract_plan.json"
    out.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(json.dumps(plan, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
