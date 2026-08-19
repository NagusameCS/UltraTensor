"""Hyper-MoE specialist registry.

The routing table for the hyper-MoE: every specialist (purpose or
language), its battery, ranking source, keep-N, target quantization,
model path and backend.  The dispatcher resolves a request against
this file; only specialists with an existing model file are eligible.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

REGISTRY = {
    "version": 2,
    "router": {"type": "heuristic-v1",
               "module": "ultratensor.hypermoe.router"},
    "fallbacks": ["general", "coder-general"],
    "specialists": [
        {"id": "coder-general-64", "domain": "coder-general",
         "battery": "code_prompts.json", "keep": 64,
         "model": "Y:/models/coder/"
                  "DeepSeek-V4-Coder-keep64-00001-of-00001.gguf",
         "backend": "numpy", "quant": "Q3_K_M", "status": "built"},
        {"id": "coder-uniform-16", "domain": "general",
         "battery": "code_prompts.json", "keep": 16,
         "model": "D:/hyperv4/models/coder/"
                  "DeepSeek-V4-Coder-keep16u.gguf",
         "backend": "llama-cli", "quant": "Q3_K", "status": "built"},
        {"id": "python-16", "domain": "python",
         "battery": "pl_prompts.json", "segment": "python",
         "keep": 16, "model": None, "backend": "llama-cli",
         "quant": "IQ2_XS", "status": "planned",
         "ranking": "outputs/rank_python.json"},
        {"id": "rust-16", "domain": "rust",
         "battery": "pl_prompts.json", "segment": "rust",
         "keep": 16, "model": None, "backend": "llama-cli",
         "quant": "IQ2_XS", "status": "planned",
         "ranking": "outputs/rank_rust.json"},
        {"id": "sql-16", "domain": "sql",
         "battery": "pl_prompts.json", "segment": "sql",
         "keep": 16, "model": None, "backend": "llama-cli",
         "quant": "IQ2_XS", "status": "planned",
         "ranking": "outputs/rank_sql.json"},
        {"id": "javascript-16", "domain": "javascript",
         "battery": "pl_prompts.json", "segment": "js",
         "keep": 16, "model": None, "backend": "llama-cli",
         "quant": "IQ2_XS", "status": "planned",
         "ranking": "outputs/rank_javascript.json"},
        {"id": "backend-16", "domain": "backend",
         "battery": "domain_prompts.json", "segment": "backend",
         "keep": 16, "model": None, "backend": "llama-cli",
         "quant": "IQ2_XS", "status": "planned",
         "ranking": "outputs/rank_backend.json"},
        {"id": "frontend-16", "domain": "frontend",
         "battery": "domain_prompts.json", "segment": "frontend",
         "keep": 16, "model": None, "backend": "llama-cli",
         "quant": "IQ2_XS", "status": "planned",
         "ranking": "outputs/rank_frontend.json"},
        {"id": "data-16", "domain": "data",
         "battery": "domain_prompts.json", "segment": "data",
         "keep": 16, "model": None, "backend": "llama-cli",
         "quant": "IQ2_XS", "status": "planned",
         "ranking": "outputs/rank_data.json"},
        {"id": "devops-16", "domain": "devops",
         "battery": "domain_prompts.json", "segment": "devops",
         "keep": 16, "model": None, "backend": "llama-cli",
         "quant": "IQ2_XS", "status": "planned",
         "ranking": "outputs/rank_devops.json"},
        {"id": "math-16", "domain": "math",
         "battery": "math_prompts.json", "keep": 16,
         "model": None, "backend": "llama-cli",
         "quant": "IQ2_XS", "status": "planned",
         "ranking": "outputs/rank_math.json"},
    ],
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "outputs" /
                                        "hypermoe_registry.json"))
    a = ap.parse_args()
    dest = Path(a.out)
    dest.write_text(json.dumps(REGISTRY, indent=2), encoding="utf-8")
    n = len(REGISTRY["specialists"])
    built = sum(1 for s in REGISTRY["specialists"] if s["model"])
    print(f"wrote {dest}: {n} specialists ({built} built, "
          f"{n - built} planned)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
