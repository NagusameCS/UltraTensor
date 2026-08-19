"""V4-Coder step 6 — kept-set coverage of the true router (Phase 2).

Subnetwork extraction keeps K of 384 experts; the extraction risk is
how much of the TRUE router's top-6 (over all 384) falls inside the
kept set:

  mean_top6_kept_fraction : average fraction of the true top-6 kept
  full_coverage_rate       : tokens where ALL 6 true experts are kept

(The refit's own top-6 over kept rows equals the true scores by
construction — agreement there is trivially 1.0 and is NOT reported.)
Coverage is reported on CODE traffic and, when given, on general
traffic for contrast.

Usage (after the census lands):
    python scripts/v4_router_refit.py --layer 3 --census outputs/code_census.json \
        --code-npz outputs/exp_code_ffn_inputs_dense.npz \
        --general-npz outputs/exp96_ffn_inputs_dense.npz
"""

import argparse
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import v4_ref_serve as vs  # noqa: E402

TOP_K = 6


def softplus(z):
    return np.sqrt(np.log1p(np.exp(z)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=3)
    ap.add_argument("--census", default=str(ROOT / "outputs" /
                                            "code_census.json"))
    ap.add_argument("--code-npz", required=True)
    ap.add_argument("--general-npz", default=None)
    ap.add_argument("--keep", type=int, nargs="+", default=[32, 64, 96])
    a = ap.parse_args()

    census = json.load(open(a.census, encoding="utf-8"))
    L = census.get("layers", {}).get(f"L{a.layer}")
    if not L or "top64_ids" not in L:
        print("census missing layer entry; run the census first")
        return 2
    top64 = L["top64_ids"]

    shards = sorted(glob.glob("D:/hyperv4/models/pro/"
                              "deepseek-ai-DeepSeek-V4-Pro-Q3_K_M-*.gguf"))
    st = vs.ExpertStore(shards[0], extra_shards=shards[1:])
    Wr = st.read_tensor(a.layer, "ffn_gate_inp")
    bias = st.read_tensor(a.layer, "exp_probs_b")

    Xc = np.asarray(np.load(a.code_npz)[f"L{a.layer}"], dtype=np.float64)
    true_c = softplus(Xc @ Wr.T) + bias

    report = {"layer": a.layer, "top_k": TOP_K,
              "n_code_tokens": int(Xc.shape[0]), "keep": {}}
    if a.general_npz:
        Xg = np.asarray(np.load(a.general_npz)[f"L{a.layer}"],
                        dtype=np.float64)
        true_g = softplus(Xg @ Wr.T) + bias
        report["n_general_tokens"] = int(Xg.shape[0])
    else:
        true_g = None

    for K in a.keep:
        kept = [int(e) for e in top64[:K]]
        kept_set = set(kept)
        # coverage: how much of the TRUE top-6 (over all 384) lives in
        # the kept set — the actual extraction risk. (The refit scores
        # over kept rows are the true scores by construction, so
        # top-6-within-kept agreement is trivially 1.0 — NOT the metric.)
        full_ids = np.argpartition(-true_c, TOP_K - 1, axis=-1)[:, :TOP_K]
        fracs = [sum(1 for e in full_ids[t] if int(e) in kept_set)
                 / TOP_K for t in range(full_ids.shape[0])]
        full_cover = sum(1 for f in fracs if f >= 1.0) / len(fracs)
        entry = {"n_kept": K, "mean_top6_kept_fraction": round(
            float(np.mean(fracs)), 4),
            "full_coverage_rate": round(full_cover, 4)}
        if true_g is not None:
            g_ids = np.argpartition(-true_g, TOP_K - 1, axis=-1)[:, :TOP_K]
            gfracs = [sum(1 for e in g_ids[t] if int(e) in kept_set)
                      / TOP_K for t in range(g_ids.shape[0])]
            entry["general_full_coverage_rate"] = round(
                sum(1 for f in gfracs if f >= 1.0) / len(gfracs), 4)
        report["keep"][str(K)] = entry
        print(f"keep{K}: code top6 kept frac={entry['mean_top6_kept_fraction']:.4f} "
              f"full={entry['full_coverage_rate']:.4f}",
              flush=True)

    out = ROOT / "outputs" / f"router_refit_L{a.layer}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
