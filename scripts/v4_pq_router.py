"""V4-Coder step 2 — PQ the ROUTER (the hypothesis G11's failure pointed at).

G11 showed PQ is dead on expert matrices (flat spectra). The
hypothesis: PQ belongs on the small dense tensors — the router first.
This measures product quantization of the real layer-3 router
[384, 7168] at 4/6/8 bits and, critically, the ROUTING consequence:
top-6 set agreement of the PQ-reconstructed router vs the true router
on the 24 real hidden states — the only metric that matters for
routing.

Writes outputs/pq_router_L3.json.

Usage:
    python scripts/v4_pq_router.py --layer 3 --bits 4,6,8
"""

import argparse
import glob
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import v4_ref_serve as vs  # noqa: E402
from ultratensor.conditional.vq import pq_reconstruct, product_quantize  # noqa: E402

INPUTS = ROOT / "outputs" / "ffn_inputs_dense.npz"
TOP_K = 6


def topk_agreement(pred, true, k):
    pk = np.argpartition(-pred, k - 1, axis=-1)[:, :k]
    tk = np.argpartition(-true, k - 1, axis=-1)[:, :k]
    return float(np.mean([len(set(a) & set(b)) / k
                          for a, b in zip(pk, tk)]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=3)
    ap.add_argument("--bits", type=str, default="4,6,8")
    a = ap.parse_args()
    bits_list = [int(b) for b in a.bits.split(",") if b.strip()]

    data = np.load(INPUTS)
    X = np.asarray(data[f"L{a.layer}"][:24], dtype=np.float64)
    shards = sorted(glob.glob("D:/hyperv4/models/pro/"
                              "deepseek-ai-DeepSeek-V4-Pro-Q3_K_M-*.gguf"))
    st = vs.ExpertStore(shards[0], extra_shards=shards[1:])
    W = st.read_tensor(a.layer, "ffn_gate_inp")          # [384, 7168]
    bias = st.read_tensor(a.layer, "exp_probs_b")

    true = np.sqrt(np.log1p(np.exp(X @ W.T))) + bias

    report = {"layer": a.layer, "shape": list(W.shape), "top_k": TOP_K,
              "n_tokens": int(X.shape[0])}
    for bits in bits_list:
        t0 = time.time()
        r = product_quantize(W, n_sub=8, n_bits=bits, iters=15, seed=0)
        Wq = pq_reconstruct(r.codes, r.codebooks, W.shape[1])
        pred = np.sqrt(np.log1p(np.exp(X @ Wq.T))) + bias
        report[f"bits{bits}"] = {
            "frob_error": round(r.frob_error, 5),
            "topk_agreement": round(topk_agreement(pred, true, TOP_K), 4),
            "params_ratio": round((r.codes.size + r.codebooks.size)
                                  / W.size, 4),
            "seconds": round(time.time() - t0, 1),
        }
        print(f"bits{bits}: {report[f'bits{bits}']}", flush=True)

    out = ROOT / "outputs" / f"pq_router_L{a.layer}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
