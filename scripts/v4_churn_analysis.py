"""Explain the hash-vs-dense per-layer latency gap (CI finding 2026-08-16).

Hypothesis: hash layers route by token id, so consecutive tokens stream
almost entirely NEW experts (high churn -> no OS page-cache reuse);
dense layers' top-6 concentrates on a few popular experts (low churn ->
cache hits). Quantify the hash side with the real traces, and state the
dense-side prediction for when phase-B traces land.

Usage: python scripts/v4_churn_analysis.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from ultratensor.conditional.stats import bootstrap_ci  # noqa: E402


def main() -> int:
    p = ROOT / "outputs" / "router_trace_hash.json"
    if not p.exists():
        print("run scripts/v4_router_trace.py first")
        return 2
    d = json.load(open(p, encoding="utf-8"))
    # per-expert bytes for one hash layer's routed IO (docstring constants)
    expert_bytes = (9.46 + 9.46 + 15.1)  # MB: gate + up + down, one expert
    print(f"{'layer':>5} {'churn':>7} {'new_MB/tok':>11} {'distinct':>8}")
    for layer, entry in d["layers"].items():
        churn = entry.get("mean_churn", 0.0)
        new_mb = churn * 6 * expert_bytes
        print(f"{layer:>5} {churn:>7.3f} {new_mb:>11.1f} {entry['distinct_experts_used']:>8}")
        if entry.get("sets"):
            sets = [set(s) for s in entry["sets"]]
            # consecutive-step overlap
            overlap = [len(sets[t] & sets[t - 1]) for t in range(1, len(sets))]
            print(f"  consecutive-step overlap: mean {np.mean(overlap):.2f} "
                  f"of 6 experts")
    print()
    print("Dense-side prediction: top-6 over hidden-state scores should")
    print("concentrate on a small expert set (low churn -> cache reuse);")
    print("this is testable once phase-B dense traces land.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
