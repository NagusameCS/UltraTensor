"""Great Sage gate refit: retrain the K-column router for a specialist.

Quality recovery for keep-N: the sliced original gate routes only to
experts that may have been dropped; the refit learns, from census
traces, to score the KEPT experts directly.  Targets are the full
model's own gate scores (router distillation), so no teacher
generation is needed.

    X      [T, 7168]   ffn inputs (cluster_dense_trace npz)
    G      [384, 7168] original gate, B bias (full model shards)
    kept   [K]         kept expert ids (keep64 manifest ranking)
    s      = sqrt(softplus(X G^T + B))[:, kept]        teacher scores
    W      = ridge(X -> s), b = mean(s)                refit gate

Reports held-out coverage of the refit vs the sliced baseline, and
exports outputs/refit_gate_L{L}.json.

Usage:
    python scripts/v4_gate_refit.py --trace outputs/exp256_ffn_inputs_dense.npz \
        --layer 3 --keep 16
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import v4_ref_serve as vs  # noqa: E402

TOP_K = 6


def softplus(z):
    return np.sqrt(np.log1p(np.exp(z)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--layer", type=int, default=3)
    ap.add_argument("--keep", type=int, default=16)
    ap.add_argument("--reg", type=float, default=1.0)
    ap.add_argument("--train", type=int, default=0,
                    help="train-token count (0 = 2/3 of trace)")
    a = ap.parse_args()

    data = np.load(a.trace)
    X = np.asarray(data[f"L{a.layer}"], dtype=np.float64)
    shards = sorted(glob.glob("D:/hyperv4/models/pro/"
                              "deepseek-ai-DeepSeek-V4-Pro-Q3_K_M-*.gguf"))
    st = vs.ExpertStore(shards[0], extra_shards=shards[1:])
    G = st.read_tensor(a.layer, "ffn_gate_inp").astype(np.float64)
    bias = st.read_tensor(a.layer, "exp_probs_b").astype(np.float64)

    from ultratensor.gguf_factored import read_gguf_header
    _, kvs, _, _ = read_gguf_header(
        "Y:/models/coder/DeepSeek-V4-Coder-keep64-00001-of-00001.gguf")
    manifest = None
    for k, t, r in kvs:
        if k == b"ultratensor.keep_manifest":
            manifest = json.loads(r[8:].decode())
    kept = manifest["kept"][f"blk.{a.layer}.ffn_gate_exps.weight"][:a.keep]

    n = a.train or int(X.shape[0] * 2 / 3)
    Xtr, Xho = X[:n], X[n:]
    Y = softplus(Xtr @ G.T + bias)[:, kept]
    s_ho = softplus(Xho @ G.T + bias)                # [T, 384]
    top384 = np.argsort(-s_ho, axis=1)[:, :TOP_K]
    rows = np.arange(Xho.shape[0])[:, None]
    denom = s_ho[rows, top384].sum()                 # true top-6 mass

    A = Xtr.T @ Xtr + a.reg * np.eye(Xtr.shape[1])
    W = np.linalg.solve(A, Xtr.T @ Y).T               # [K, 7168]
    b = Y.mean(axis=0)                                # [K]

    pred = softplus(Xho @ W.T + b)                    # refit scores [T,K]
    top_refit = np.argsort(-pred, axis=1)[:, :TOP_K]
    refit_ids = np.array(kept)[top_refit]
    cov_refit = float(s_ho[rows, refit_ids].sum() / denom)

    # sliced baseline: original scores, top-6 among the kept columns
    s_kept = s_ho[:, kept]
    top_sliced = np.argsort(-s_kept, axis=1)[:, :TOP_K]
    cov_base = float(s_kept[rows, top_sliced].sum() / denom)

    out = {"layer": a.layer, "keep": a.keep, "n_train": int(Xtr.shape[0]),
           "n_hold": int(Xho.shape[0]), "reg": a.reg,
           "kept": kept, "weights": W.tolist(), "bias": b.tolist(),
           "coverage_hold_refit": cov_refit,
           "coverage_hold_sliced_baseline": cov_base}
    dest = ROOT / "outputs" / f"refit_gate_L{a.layer}.json"
    dest.write_text(json.dumps(out))
    print(f"layer {a.layer}: refit coverage {cov_refit:.3f} vs "
          f"sliced baseline {cov_base:.3f} "
          f"({cov_refit / max(cov_base, 1e-9):.2f}x) -> {dest.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
