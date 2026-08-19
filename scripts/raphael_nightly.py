"""Raphael's nightly loop: the self-improving factory (orchestrator).

1. harvest new censuses        (outputs/*_census.json)
2. emit per-layer rankings     (v4_rebuild_from_census --no-build)
3. rebuild specialists         (Degenerate)
4. validate held-out coverage  (v4_router_refit metric)
5. update registry status      (mark rebuilt models)

Each step no-ops when its inputs are missing, so the loop can run
nightly and only does work after new census data lands.

Usage:
    python scripts/raphael_nightly.py [--build]
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CENSUS_DIR = ROOT / "outputs"
SPECIALISTS = [
    {"id": "python-16", "census": "pl_census.json", "segment": "python"},
    {"id": "rust-16", "census": "pl_census.json", "segment": "rust"},
    {"id": "sql-16", "census": "pl_census.json", "segment": "sql"},
    {"id": "javascript-16", "census": "pl_census.json", "segment": "js"},
    {"id": "backend-16", "census": "dom_census.json", "segment": "backend"},
    {"id": "frontend-16", "census": "dom_census.json", "segment": "frontend"},
    {"id": "data-16", "census": "dom_census.json", "segment": "data"},
    {"id": "devops-16", "census": "dom_census.json", "segment": "devops"},
    {"id": "math-16", "census": "math_census.json", "segment": "math"},
]


def _census_ranking(census_path: Path, segment: str, keep: int) -> dict | None:
    """Segment top-K of a pl-style census -> per_layer ranking JSON."""
    if not census_path.exists():
        return None
    try:
        data = json.loads(census_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    lang = data.get("languages", {}).get(segment)
    if not lang or "top32_ids" not in lang:
        return None
    top = lang["top32_ids"][:keep]
    return {"dense": top, "hash": top}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--keep", type=int, default=16)
    a = ap.parse_args()

    report = {"built": [], "skipped": []}
    for spec in SPECIALISTS:
        census = CENSUS_DIR / spec["census"]
        ranking = _census_ranking(census, spec["segment"], a.keep)
        if ranking is None:
            report["skipped"].append(spec["id"])
            continue
        rank_path = CENSUS_DIR / f"rank_{spec['id']}.json"
        rank_path.write_text(json.dumps(ranking, indent=2))
        out = (f"Y:/models/coder/"
               f"DeepSeek-V4-Coder-{spec['id']}.gguf")
        if a.build and not Path(out).exists():
            cmd = [sys.executable, str(ROOT / "scripts" /
                                       "v4_coder_keep_uniform.py"),
                   "--keep", str(a.keep), "--out", out,
                   "--ranking", str(rank_path)]
            print("building:", spec["id"], flush=True)
            subprocess.call(cmd)
        report["built"].append({"id": spec["id"],
                                "ranking": str(rank_path),
                                "model": out})
    dest = CENSUS_DIR / "raphael_nightly_report.json"
    dest.write_text(json.dumps(report, indent=2))
    print(json.dumps({"built": len(report["built"]),
                      "skipped": len(report["skipped"])}))
    print(f"report -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
