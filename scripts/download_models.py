"""Download UltraTensor spliced DeepSeek-V4-Coder GGUFs from Hugging Face.

Usage:
    python scripts/download_models.py                 # all ladder + IQ2_XS models
    python scripts/download_models.py --model keep16u-iq2xs
    python scripts/download_models.py --dest D:\\models\\ultratensor
    python scripts/download_models.py --include-keep64

The models are expected at a Hugging Face repo (default:
NagusameCS/ultratensor-models). Override with --repo or the
ULTRATENSOR_HF_REPO environment variable. Requires: pip install huggingface_hub
(and `huggingface-cli login` once).
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "ultratensor" / "model_catalog.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="model id to download "
                    "(default: all except keep64)")
    ap.add_argument("--dest", default=str(ROOT / "models"),
                    help="destination directory")
    ap.add_argument("--repo", default=None,
                    help="HF repo (default: NagusameCS/ultratensor-models or "
                    "ULTRATENSOR_HF_REPO)")
    ap.add_argument("--include-keep64", action="store_true",
                    help="also download the 156 GiB keep64 splice")
    ap.add_argument("--local-dir", default=None,
                    help="optional local copy of the catalog source for "
                    "verify-only mode")
    a = ap.parse_args()

    import os
    repo = a.repo or os.environ.get("ULTRATENSOR_HF_REPO",
                                    "NagusameCS/ultratensor-models")

    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    models = catalog["models"]
    if a.model:
        models = [m for m in models if m["id"] == a.model]
        if not models:
            print(f"unknown model id: {a.model}", file=sys.stderr)
            return 2
    elif not a.include_keep64:
        models = [m for m in models if m["id"] != "keep64"]

    dest = Path(a.dest)
    dest.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("huggingface_hub is not installed. Run: "
              "pip install huggingface_hub && huggingface-cli login",
              file=sys.stderr)
        return 1

    patterns = [m["file"] for m in models]
    total = sum(m["bytes"] for m in models)
    print(f"repo: {repo}")
    print(f"files: {len(patterns)}  total: {total / 2**30:.1f} GiB")
    print(f"dest: {dest}")
    path = snapshot_download(repo_id=repo, allow_patterns=patterns,
                             local_dir=str(dest))
    print(f"downloaded to {path}")
    print("Serve example (CPU quality tier):")
    print("  llama-server -m models/DeepSeek-V4-Coder-keep16u.gguf "
          "--host 127.0.0.1 --port 8780 -ngl 0 -c 512 --no-op-offload")
    print("Serve example (GPU speed tier):")
    print("  llama-server -m models/DeepSeek-V4-Coder-keep16u-iq2xs.gguf "
          "--host 127.0.0.1 --port 8791 -ngl 12 -c 512 --no-op-offload")
    return 0


if __name__ == "__main__":
    sys.exit(main())
