"""Upload UltraTensor spliced DeepSeek-V4-Coder GGUFs to Hugging Face.

Usage:
    huggingface-cli login                       # once, with your token
    python scripts/upload_models.py --src D:\\hyperv4\\models\\coder
    python scripts/upload_models.py --src D:\\hyperv4\\models\\coder --only keep16u-iq2xs
    python scripts/upload_models.py --src Y:\\models\\coder --include-keep64

Uploads the catalog files from a local source directory into the HF repo
(default NagusameCS/ultratensor-models; override with --repo or
ULTRATENSOR_HF_REPO). Resumable and idempotent (skips already-uploaded files).

Model repo layout: each file is uploaded under <model-id>/<filename> so
downloading one splice does not fetch the others.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "ultratensor" / "model_catalog.json"


def repo_files(repo: str) -> set:
    """List files already present in the repo (empty set if not reachable)."""
    try:
        out = subprocess.run(
            ["huggingface-cli", "repo", "files", repo, "--format", "json"],
            capture_output=True, text=True, timeout=120)
        if out.returncode != 0:
            return set()
        import json as _json
        return {f["path"] for f in _json.loads(out.stdout)}
    except Exception:
        return set()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True,
                    help="local directory holding the .gguf files")
    ap.add_argument("--only", default=None, help="comma-separated model ids")
    ap.add_argument("--repo", default=None,
                    help="HF repo (default NagusameCS/ultratensor-models or "
                    "ULTRATENSOR_HF_REPO)")
    ap.add_argument("--include-keep64", action="store_true")
    ap.add_argument("--yes", action="store_true", help="skip confirmation")
    a = ap.parse_args()

    import os
    repo = a.repo or os.environ.get("ULTRATENSOR_HF_REPO",
                                    "NagusameCS/ultratensor-models")
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    models = catalog["models"]
    if a.only:
        ids = set(a.only.split(","))
        models = [m for m in models if m["id"] in ids]
    if not a.include_keep64:
        models = [m for m in models if m["id"] != "keep64"]

    src = Path(a.src)
    missing = [m for m in models if not (src / m["file"]).exists()]
    if missing:
        print("missing files in --src:", file=sys.stderr)
        for m in missing:
            print(f"  {m['file']}", file=sys.stderr)
        return 2

    total = sum((src / m["file"]).stat().st_size for m in models)
    print(f"repo:  {repo}")
    print(f"files: {len(models)}  total: {total / 2**30:.1f} GiB")
    print(f"NOTE:  {catalog['note']}")
    if not a.yes:
        ans = input("proceed? [y/N] ")
        if ans.strip().lower() != "y":
            print("aborted")
            return 0

    existing = repo_files(repo)
    for m in models:
        target = f"{m['id']}/{m['file']}"
        if target in existing:
            print(f"skip   {target} (already present)")
            continue
        local = src / m["file"]
        print(f"upload {target} ({local.stat().st_size / 2**30:.2f} GiB)")
        t0 = time.time()
        rc = subprocess.run(
            ["huggingface-cli", "upload", repo, str(local), target],
            check=False)
        if rc.returncode != 0:
            print(f"FAILED {target} (rc={rc.returncode}) — rerun this "
                  f"command to resume; already-uploaded files are skipped",
                  file=sys.stderr)
            return 3
        print(f"  done in {(time.time() - t0) / 60:.1f} min")

    print("all files uploaded")
    print("users download with:")
    print("  python scripts/download_models.py --dest models")
    return 0


if __name__ == "__main__":
    sys.exit(main())
