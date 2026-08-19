"""Formal G11 runner: product quantization of a real V4 expert tensor.

Usage:
    python scripts/v4_pq_expert.py [--layer 0 --expert 0 --bits 4,6,8]

Uses the optimised float32 k-means (subsampled init, early exit).
"""

import argparse
import glob
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

import v4_ref_serve as vs  # noqa: E402
from ultratensor.conditional.vq import product_quantize  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--expert", type=int, default=0)
    ap.add_argument("--kind", default="ffn_down_exps")
    ap.add_argument("--bits", type=str, default="4,6,8",
                    help="comma-separated codebook bits")
    a = ap.parse_args()
    bits_list = [int(b) for b in a.bits.split(",") if b.strip()]

    shards = sorted(glob.glob("D:/hyperv4/models/pro/"
                              "deepseek-ai-DeepSeek-V4-Pro-Q3_K_M-*.gguf"))
    st = vs.ExpertStore(shards[0], extra_shards=shards[1:])
    t0 = time.time()
    W = st.read_expert(a.layer, a.kind, a.expert).astype(np.float64)
    report = {"layer": a.layer, "expert": a.expert, "kind": a.kind,
              "shape": list(W.shape),
              "read_s": round(time.time() - t0, 1)}
    for bits in bits_list:
        t0 = time.time()
        r = product_quantize(W, n_sub=8, n_bits=bits, iters=15, seed=0)
        report[f"bits{bits}"] = {
            "frob_error": round(r.frob_error, 5),
            "params_M": round((r.codes.size + r.codebooks.size) / 1e6, 2),
            "orig_params_M": round(W.size / 1e6, 2),
            "seconds": round(time.time() - t0, 1),
        }
        print(f"bits{bits}: {report[f'bits{bits}']}", flush=True)

    out = ROOT / "outputs" / f"pq_expert{a.expert}_L{a.layer}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
