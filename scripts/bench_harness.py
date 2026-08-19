"""G1 — fixed benchmark harness driver.

One command that runs the cheap real-bytes experiments in a fixed order
and prints a consolidated, CI-wrapped summary. Everything it invokes is
quick by design (hash traces, prefetch eval, churn, spec projection,
CI report); the multi-hour dense trace runs separately.

Usage:
    python scripts/bench_harness.py [--skip <name>]...
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

STAGES = [
    ("hash_trace", "v4_router_trace.py"),
    ("churn", "v4_churn_analysis.py"),
    ("route_stability", "v4_route_stability.py"),
    ("prefetch_eval", "v4_prefetch_eval.py"),
    ("spec_projection", "spec_projection.py"),
    ("ci_report", "ci_report.py"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip", action="append", default=[])
    a = ap.parse_args()

    results = {}
    for name, script in STAGES:
        if name in (a.skip or []):
            results[name] = "skipped"
            continue
        rc = subprocess.run([PY, str(ROOT / "scripts" / script)],
                            cwd=ROOT, capture_output=True, text=True)
        results[name] = "ok" if rc.returncode == 0 else f"rc={rc.returncode}"
        tail = (rc.stdout.strip().splitlines() or ["<no output>"])[-1]
        print(f"[{name}] {results[name]}  {tail}")

    import json
    ci = json.load(open(ROOT / "outputs" / "ci_report.json",
                        encoding="utf-8"))
    print("\n== consolidated ==")
    print(f"projected tok/s: {ci['projected_tok_s_61_layers']['mean']} "
          f"{ci['projected_tok_s_61_layers']['ci95']}")
    if "exact_per_layer" in ci:
        for layer, e in sorted(ci["exact_per_layer"].items(),
                               key=lambda kv: int(kv[0])):
            print(f"  layer {layer}: {e['mean']} {e['ci95']} (n={e['n']})")
    return 0 if all(v == "ok" for v in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
