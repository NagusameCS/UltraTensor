"""Factor a single-file GGUF into the UltraTensor factored container.

Usage:
    python scripts/factor_model.py <src.gguf> <out.gguf> \
        [--pattern <substr> ...] [--rank N | --energy E]

Matches 2-D tensors whose name contains any pattern (default: FFN tensors)
and emits <name>.factored_U (fp16 basis) + <name>.factored_C (uq4 codes).
Everything else is copied byte-for-byte; metadata KVs are preserved.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ultratensor.gguf_factored import (  # noqa: E402
    read_factored_gguf,
    write_factored_gguf,
)

DEFAULT_PATTERNS = ["ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--pattern", action="append", default=None)
    ap.add_argument("--rank", type=int, default=None)
    ap.add_argument("--energy", type=float, default=None)
    a = ap.parse_args()

    patterns = a.pattern or DEFAULT_PATTERNS
    write_factored_gguf(a.src, a.out, patterns=patterns,
                        rank=a.rank, energy=a.energy or 0.99)

    manifest, tensors = read_factored_gguf(a.out)
    src_bytes = Path(a.src).stat().st_size
    out_bytes = Path(a.out).stat().st_size
    print(f"factored: {len(manifest['tensors'])} tensors")
    for t in manifest["tensors"]:
        print(f"  {t['name']}: {t['shape']} rank={t['rank']}")
    print(f"size: {src_bytes/1e6:.1f} MB -> {out_bytes/1e6:.1f} MB "
          f"({100*out_bytes/src_bytes:.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
