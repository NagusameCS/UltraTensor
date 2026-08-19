"""G10 — rho(h) escalation predictor: KNN + ridge, 2-way and 3-way splits.

Predicts per-token compression risk (coverage = rank-k projector
kept-energy, layer-3 gate expert 0) from the hidden state.

  2-way (baseline): projector fit + predictor fit on the same n train
tokens — the coverage TARGETS are train-inflated (G4 lesson).
  3-way (refinement): disjoint sets A/B/C — projector fit only /
predictor fit / eval — so eval targets are unbiased.

Predictors: 1-NN (cosine) and factored ridge on the hidden state.
The escalation ladder decides tiers from coverage.

Writes outputs/rho_predictor_L3.json.

Usage:
    python scripts/v4_rho_predictor.py --rank 8 --train 64 \
        --inputs outputs/exp96_ffn_inputs_dense.npz
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
from ultratensor.conditional.actweight import (  # noqa: E402
    projector,
    subspace_basis,
)
from ultratensor.conditional.escalation import EscalationPolicy  # noqa: E402

INPUTS = ROOT / "outputs" / "ffn_inputs_dense.npz"


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    if ra.std() == 0 or rb.std() == 0:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def ridge_predict(Xtr, ytr, Xho, lam=1.0):
    """Factored ridge: B = (X X^T + lam I)^-1 y; predict Xho @ Xtr^T @ B."""
    B = np.linalg.solve(Xtr @ Xtr.T + lam * np.eye(Xtr.shape[0]), ytr)
    return (Xho @ Xtr.T) @ B


def run_split(W, X, proj_fit, trn, evl, rank):
    """Coverage targets from a projector fit on proj_fit; predictors
    fit on trn; evaluated on evl. All index arrays."""
    V = subspace_basis(X[proj_fit], rank)
    A = projector(W, V)
    Wx = X @ W.T
    yhat = (X @ V) @ A.T
    err = ((Wx - yhat) ** 2).sum(axis=-1) / np.maximum(
        (Wx ** 2).sum(axis=-1), 1e-30)
    cov = 1.0 - err

    Xn = X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-12)
    sim = Xn @ Xn.T
    knn = np.empty_like(cov)
    for t in evl:
        pos = int(sim[t, trn].argmax())
        knn[t] = cov[trn[pos]]
    ridge = ridge_predict(X[trn], cov[trn], X[evl])

    policy = EscalationPolicy()
    tiers_true = [policy.decide(float(c), 0.0) for c in cov[evl]]

    def metrics(pred_evl):
        tp = [policy.decide(float(p), 0.0) for p in pred_evl]
        return {
            "mean_abs_error": round(float(
                np.abs(pred_evl - cov[evl]).mean()), 4),
            "spearman": round(spearman(pred_evl, cov[evl]), 4),
            "tier_agreement": round(float(
                sum(t == p for t, p in zip(tiers_true, tp))
                / len(evl)), 4),
        }

    return {
        "coverage_true": [round(float(c), 4) for c in cov[evl]],
        "coverage_pred_knn": [round(float(knn[t]), 4) for t in evl],
        "coverage_pred_ridge": [round(float(ridge[i]), 4)
                                for i in range(len(evl))],
        "knn": metrics(knn[evl]),
        "ridge": metrics(ridge),
        "tiers_true": tiers_true,
        "n_proj_fit": int(len(proj_fit)),
        "n_train": int(len(trn)),
        "n_eval": int(len(evl)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--train", type=int, default=64)
    ap.add_argument("--inputs", default=str(INPUTS))
    ap.add_argument("--proj3", type=int, default=43,
                    help="3-way: projector-fit token count")
    ap.add_argument("--pred3", type=int, default=64,
                    help="3-way: predictor-fit token count")
    a = ap.parse_args()

    data = np.load(a.inputs)
    X = np.asarray(data["L3"], dtype=np.float64)[:256]
    shards = sorted(glob.glob("D:/hyperv4/models/pro/"
                              "deepseek-ai-DeepSeek-V4-Pro-Q3_K_M-*.gguf"))
    st = vs.ExpertStore(shards[0], extra_shards=shards[1:])
    W = st.read_expert(3, "ffn_gate_exps", 0).astype(np.float64)

    n = min(a.train, X.shape[0] - 1)
    p3 = min(a.proj3, X.shape[0] - 1)
    d3 = min(a.pred3, X.shape[0] - p3 - 1)
    report = {
        "layer": 3, "expert": 0, "rank": a.rank,
        "two_way": run_split(W, X, np.arange(n), np.arange(n),
                             np.arange(n, X.shape[0]), a.rank),
        "three_way": run_split(W, X, np.arange(p3),
                               np.arange(p3, p3 + d3),
                               np.arange(p3 + d3, X.shape[0]), a.rank),
    }
    out = ROOT / "outputs" / "rho_predictor_L3.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
