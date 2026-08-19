"""G9 — controller shrink: agreement-vs-params on real router bytes.

The smoke test (v4_router_distill.py) fit a ridge controller in-sample
and got top-k agreement 1.0 at the router's own size — meaningless for
deployment. This script makes G9 honest with:

  1. a TRAIN/HELD-OUT split (fit on 16, evaluate on 8, both directions);
  2. the FACTORED ridge: the ridge solution lives in the span of the
     train samples, so W_c = X^T B with B = solve(X X^T + l I, Y).
     Params n*(d + c) instead of d*c, exact same predictions;
  3. SVD-truncated router baselines (zero training data) at rank k,
     params k*(d + c) with d=7168, c=384.

Output: outputs/controller_shrink.json — params and top-k agreement per
method, train and held-out.

Usage:
    python scripts/v4_controller_shrink.py --layer 3 --top-k 6
"""

import argparse
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import v4_ref_serve as vs  # noqa: E402

INPUTS = ROOT / "outputs" / "ffn_inputs_dense.npz"


def softplus(z):
    return np.sqrt(np.log1p(np.exp(z)))


def topk_agreement(pred: np.ndarray, true: np.ndarray, k: int) -> float:
    pk = np.argpartition(-pred, k - 1, axis=-1)[:, :k]
    tk = np.argpartition(-true, k - 1, axis=-1)[:, :k]
    return float(np.mean([len(set(a) & set(b)) / k
                          for a, b in zip(pk, tk)]))


def labels(T, k):
    """One-hot membership of the router's top-k per token."""
    ids = np.argpartition(-T, k - 1, axis=-1)[:, :k]
    Y = np.zeros_like(T)
    Y[np.arange(T.shape[0])[:, None], ids] = 1.0
    return Y


def svd_truncate(W, k):
    U, S, Vt = np.linalg.svd(W, full_matrices=False)
    if k >= len(S):
        return W.copy()
    return (U[:, :k] * S[:k]) @ Vt[:k]


def method_dense_ridge(Xtr, Xho, Ttr, Tho, k):
    Y = labels(Ttr, k)
    Wc = np.linalg.solve(Xtr.T @ Xtr + np.eye(Xtr.shape[1]), Xtr.T @ Y)
    pred = lambda Xq: Xq @ Wc
    return (pred, Wc.size, Xtr.shape[1] * Ttr.shape[1])


def method_factored_ridge(Xtr, Xho, Ttr, Tho, k):
    """W_c = X^T B with B = (X X^T + l I)^-1 Y: exact same predictions."""
    Y = labels(Ttr, k)
    B = np.linalg.solve(Xtr @ Xtr.T + np.eye(Xtr.shape[0]), Y)
    pred = lambda Xq: (Xq @ Xtr.T) @ B
    return (pred, int(Xtr.shape[0] * (Xtr.shape[1] + Ttr.shape[1])),
            int(Xtr.shape[0] * (Xtr.shape[1] + Ttr.shape[1])))


def method_factored_ridge_reg(Xtr, Xho, Ttr, Tho, k):
    """Score regression (the boundary-correct objective): fit the
    factored ridge to the SCORES, not the one-hot membership. Predicts
    a continuous score per expert; agreement + relative L1 reported."""
    B = np.linalg.solve(Xtr @ Xtr.T + np.eye(Xtr.shape[0]), Ttr)
    pred = lambda Xq: (Xq @ Xtr.T) @ B
    return (pred, int(Xtr.shape[0] * (Xtr.shape[1] + Ttr.shape[1])),
            int(Xtr.shape[0] * (Xtr.shape[1] + Ttr.shape[1])))


def method_svd_router(W_router, k_rank):
    def build(Xtr, Xho, Ttr, Tho, k):
        Wk = svd_truncate(W_router, k_rank)
        pred = lambda Xq: Xq @ Wk.T
        return (pred, int(k_rank * (W_router.shape[0] + W_router.shape[1])),
                int(k_rank * (W_router.shape[0] + W_router.shape[1])))
    return build


def evaluate(build, splits, T, k):
    out = {}
    for sname, (itr, iho) in splits.items():
        Xtr, Xho = itr[0], iho[0]
        Ttr, Tho = T[itr[1]], T[iho[1]]
        pred, params, macs = build(Xtr, Xho, Ttr, Tho, k)
        out[sname] = {
            "train_agreement": round(topk_agreement(pred(Xtr), Ttr, k), 4),
            "hold_agreement": round(topk_agreement(pred(Xho), Tho, k), 4),
        }
        # boundary context: how razor-thin is the true 6th/7th cut on the
        # held-out tokens? Agreement on near-tied boundaries is a coin
        # flip the router itself cannot resolve (measured margin ~1.003).
        tk = np.argpartition(-Tho, k - 1, axis=-1)[:, :k]
        sel = np.take_along_axis(Tho, tk, axis=-1)
        mask = np.zeros_like(Tho, dtype=bool)
        mask[np.arange(Tho.shape[0])[:, None], tk] = True
        best_out = np.where(mask, -np.inf, Tho).max(axis=-1)
        margins = sel.min(axis=-1) / np.maximum(best_out, 1e-30)
        out[sname]["margin_mean"] = round(float(margins.mean()), 4)
        out[sname]["margin_lt_1p01_frac"] = round(
            float((margins < 1.01).mean()), 4)
        out[sname]["margin_lt_1p05_frac"] = round(
            float((margins < 1.05).mean()), 4)
        sharp = margins >= 1.05
        if sharp.any():
            out[sname]["hold_agreement_sharp_boundary"] = round(
                topk_agreement(pred(Xho)[sharp], Tho[sharp], k), 4)
        else:
            out[sname]["hold_agreement_sharp_boundary"] = None
        # score-regression quality on hold (boundary-correct metric)
        sh = pred(Xho)
        out[sname]["hold_score_rel_l1"] = round(float(
            np.abs(sh - Tho).sum() / max(np.abs(Tho).sum(), 1e-30)), 6)
    out["params"] = params
    out["macs_per_token"] = macs
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=3)
    ap.add_argument("--top-k", type=int, default=6)
    ap.add_argument("--train", type=int, default=16)
    ap.add_argument("--ranks", type=int, nargs="+",
                    default=[4, 8, 16, 32, 64, 128])
    ap.add_argument("--inputs", default=str(INPUTS))
    a = ap.parse_args()

    data = np.load(a.inputs)
    X = np.asarray(data[f"L{a.layer}"], dtype=np.float64)[:86]
    shards = sorted(glob.glob("D:/hyperv4/models/pro/"
                              "deepseek-ai-DeepSeek-V4-Pro-Q3_K_M-*.gguf"))
    st = vs.ExpertStore(shards[0], extra_shards=shards[1:])
    W = st.read_tensor(a.layer, "ffn_gate_inp")
    bias = st.read_tensor(a.layer, "exp_probs_b")

    T = softplus(X @ W.T) + bias                 # [n, c] router scores
    d, c = W.shape[1], W.shape[0]
    n = min(a.train, X.shape[0] - 1)

    # splits as (X, row-indices into T)
    splits = {
        "fwd": ((X[:n], np.arange(n)), (X[n:], np.arange(n, X.shape[0]))),
        "rev": ((X[-n:], np.arange(X.shape[0] - n, X.shape[0])),
                (X[:-n], np.arange(X.shape[0] - n))),
        "all": ((X, np.arange(X.shape[0])), (X, np.arange(X.shape[0]))),
    }

    report = {"layer": a.layer, "top_k": a.top_k, "train": n,
              "d": d, "c": c, "methods": {}}
    report["methods"]["ridge_dense"] = evaluate(
        method_dense_ridge, splits, T, a.top_k)
    report["methods"]["ridge_factored"] = evaluate(
        method_factored_ridge, splits, T, a.top_k)
    report["methods"]["ridge_factored_reg"] = evaluate(
        method_factored_ridge_reg, splits, T, a.top_k)
    for k in a.ranks:
        report["methods"][f"router_svd_k{k}"] = evaluate(
            method_svd_router(W, k), splits, T, a.top_k)

    out = ROOT / "outputs" / "controller_shrink.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
