"""Rebuild a keep-N specialist using PER-LAYER rankings from a census.

The L3-ranking-for-all-layers assumption is the biggest cheap quality
loss in the current keep builds.  A census with `layers.L{idx}.top64_ids`
provides per-layer code-mass rankings; this emits a per_layer ranking
JSON and rebuilds with them (falling back to the deepest available
ranking for layers the census didn't touch).

Usage:
    python scripts/v4_rebuild_from_census.py --census outputs/mid_census.json \
        --keep 16 --out Y:/models/coder/DeepSeek-V4-Coder-keep16u-v2.gguf \
        [--build]
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--census", required=True)
    ap.add_argument("--keep", type=int, default=16)
    ap.add_argument("--hash-ranking", default=None,
                    help="JSON with hash-layer ranking (defaults to "
                         "census L0 ranking or keep64 L0)")
    ap.add_argument("--out", default="Y:/models/coder/"
                                    "DeepSeek-V4-Coder-keep16u-v2.gguf")
    ap.add_argument("--build", action="store_true")
    a = ap.parse_args()

    census = json.load(open(a.census, encoding="utf-8"))
    layers = census.get("layers", {})
    per_layer = {}
    for key, val in layers.items():
        if key.startswith("L") and "top64_ids" in val:
            per_layer[str(int(key[1:]))] = val["top64_ids"]
    if not per_layer:
        print("census has no per-layer top64_ids")
        return 2
    fallback = per_layer.get("3") or max(per_layer.values(), key=len)
    hash_rank = fallback
    if a.hash_ranking:
        h = json.load(open(a.hash_ranking, encoding="utf-8"))
        hash_rank = h.get("hash") or h.get("L0") or fallback

    ranking = {"per_layer": per_layer, "dense": fallback,
               "hash": hash_rank}
    dest = ROOT / "outputs" / "rank_per_layer.json"
    dest.write_text(json.dumps(ranking, indent=2))
    print(f"wrote {dest}: {len(per_layer)} layers, fallback len "
          f"{len(fallback)}")
    if a.build:
        cmd = [sys.executable, str(ROOT / "scripts" /
                                   "v4_coder_keep_uniform.py"),
               "--keep", str(a.keep), "--out", a.out,
               "--ranking", str(dest)]
        print("building:", " ".join(cmd))
        return subprocess.call(cmd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
