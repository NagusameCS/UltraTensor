"""Raphael provenance chains: certify how every artifact was made.

Hashes the source shards, census, ranking, model, and the applied
patches into a signed-feeling manifest: each stage references the
previous stage's hash, so a model can prove its full lineage.

Usage:
    python scripts/v4_provenance.py --model D:/.../keep16u.gguf \
        --census outputs/code_census.json \
        --ranking outputs/rank_per_layer.json \
        --out outputs/provenance_keep16u.json
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha256(path: str, limit_mb: int = 0) -> str:
    h = hashlib.sha256()
    size = 0
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
            size += len(chunk)
            if limit_mb and size > limit_mb << 20:
                h.update(f"<truncated@{size}>".encode())
                break
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--census", default=None)
    ap.add_argument("--ranking", default=None)
    ap.add_argument("--patches", nargs="*", default=[],
                    help="patch ops applied, e.g. split.count=1")
    ap.add_argument("--shards", nargs="*", default=[],
                    help="source shards (first 4 hashed fully)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    stages = []
    for s in a.shards[:4]:
        stages.append({"stage": "source-shard", "path": s,
                       "sha256": sha256(s)})
    if a.census:
        stages.append({"stage": "census", "path": a.census,
                       "sha256": sha256(a.census)})
    if a.ranking:
        stages.append({"stage": "ranking", "path": a.ranking,
                       "sha256": sha256(a.ranking)})
    stages.append({"stage": "model", "path": a.model,
                   "sha256": sha256(a.model, limit_mb=4096)})
    manifest = {
        "provenance_version": 1,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "patches": a.patches,
        "chain": stages,
    }
    # link hashes into the chain
    prev = ""
    for st in manifest["chain"]:
        st["prev"] = prev
        prev = st["sha256"]
    out = Path(a.out)
    out.write_text(json.dumps(manifest, indent=2))
    print(f"provenance manifest -> {out} "
          f"({len(manifest['chain'])} stages)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
