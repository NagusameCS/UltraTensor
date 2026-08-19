"""Honest confidence-interval report over existing measurements.

Reads outputs/bench_*.json and wraps the measured numbers in bootstrap
CIs (across-layer means are treated as the sample). No raw per-token
samples were saved, so per-layer CIs are analytic (mean +- std/sqrt(n));
the cross-layer bootstrap is exact.

Usage: python scripts/ci_report.py
Writes outputs/ci_report.json
"""

import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from ultratensor.conditional.stats import bootstrap_ci  # noqa: E402


def main() -> int:
    files = sorted(glob.glob(str(ROOT / "outputs" / "bench_moe_layer_*.json")))
    layers, stds, ns = [], [], []
    for f in files:
        d = json.load(open(f, encoding="utf-8"))
        layers.append(d["s_per_token_layer"])
        stds.append(d.get("s_per_token_layer_std", 0.0))
        ns.append(d.get("n_tokens", 300))
    layers = np.asarray(layers)
    stds = np.asarray(stds)
    ns = np.asarray(ns, dtype=np.float64)

    report = {"n_layers": int(layers.size), "exact_per_layer": {}}

    # exact raw-sample bootstrap where raw times were saved
    for f in files:
        d = json.load(open(f, encoding="utf-8"))
        if "times" in d and len(d["times"]) >= 10:
            ci = bootstrap_ci(d["times"], n_resamples=20000)
            report["exact_per_layer"][d["layer"]] = {
                "mean": round(ci.mean, 4),
                "ci95": [round(ci.ci_lower, 4), round(ci.ci_upper, 4)],
                "n": len(d["times"]),
            }

    # cross-layer bootstrap: mean layer latency and 61-layer projection
    ci_mean = bootstrap_ci(layers, n_resamples=20000)
    report["layer_latency_s"] = {
        "mean": round(ci_mean.mean, 4),
        "ci95": [round(ci_mean.ci_lower, 4), round(ci_mean.ci_upper, 4)],
        "analytic_pooled_ci95": [
            round(float(np.mean(layers) - 1.96 * np.sqrt((stds ** 2 / ns).mean())), 4),
            round(float(np.mean(layers) + 1.96 * np.sqrt((stds ** 2 / ns).mean())), 4),
        ],
    }

    proj = np.asarray([json.load(open(f, encoding="utf-8"))[
        "projected_tok_s_61_layers"] for f in files])
    ci_proj = bootstrap_ci(proj, n_resamples=20000)
    report["projected_tok_s_61_layers"] = {
        "mean": round(ci_proj.mean, 4),
        "ci95": [round(ci_proj.ci_lower, 4), round(ci_proj.ci_upper, 4)],
    }

    # hash vs dense comparison (cross-layer bootstrap of the difference)
    hash_idx = [i for i, f in enumerate(files)
                if json.load(open(f, encoding="utf-8"))["router"] == "hash"]
    dense_idx = [i for i in range(len(files)) if i not in hash_idx]
    if hash_idx and dense_idx:
        diff = layers[dense_idx].mean() - layers[hash_idx].mean()
        ci_d = bootstrap_ci(layers[dense_idx], n_resamples=20000)
        ci_h = bootstrap_ci(layers[hash_idx], n_resamples=20000)
        report["hash_vs_dense"] = {
            "dense_mean": round(ci_d.mean, 4),
            "hash_mean": round(ci_h.mean, 4),
            "difference": round(diff, 4),
            "dense_ci95": [round(ci_d.ci_lower, 4), round(ci_d.ci_upper, 4)],
            "hash_ci95": [round(ci_h.ci_lower, 4), round(ci_h.ci_upper, 4)],
        }

    lazy = ROOT / "outputs" / "bench_lazy_full.json"
    if lazy.exists():
        d = json.load(open(lazy, encoding="utf-8"))
        report["lazy_full"] = {k: v for k, v in d.items()
                               if isinstance(v, (int, float))}

    out = ROOT / "outputs" / "ci_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
