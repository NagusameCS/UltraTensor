"""Requant a GGUF with llama-quantize (IQ2_XS etc.).

Thin wrapper so Raphael UIs/automation have one argv surface; the
actual quantization runs through the fork's llama-quantize binary.

Usage:
    python scripts/v4_requant.py --model M.gguf --out O.gguf \
        --imatrix imatrix.dat [--type IQ2_XS] [--threads 4] \
        [--quant-bin path/to/llama-quantize.exe]
"""

import argparse
import subprocess
import sys
from pathlib import Path

DEFAULT_BIN = "C:/Users/legom/hyperv4flash/engine/llama-quantize.exe"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--imatrix", required=True)
    ap.add_argument("--type", default="IQ2_XS")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--quant-bin", default=DEFAULT_BIN)
    a = ap.parse_args()

    bin_path = a.quant_bin
    if not Path(bin_path).exists():
        print(f"quant bin not found: {bin_path}")
        return 1
    cmd = [bin_path, "--allow-requantize", "--imatrix", a.imatrix,
           a.model, a.out, a.type, str(a.threads)]
    print("running:", " ".join(cmd), flush=True)
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
