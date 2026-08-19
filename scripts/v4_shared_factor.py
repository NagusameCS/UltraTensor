"""G6 — shared+private factorization on REAL bytes (layer 3 gate experts).

Compares, at EQUAL total parameter budget, over the top-8 routed gate
experts of layer 3 (24 real tokens):

  independent : per-expert rank-r_ind SVD truncation
  shared      : W_shared + U A_e V^T + per-expert private rank-r_p
                residual (same one-shot joint fit as
                ultratensor.conditional.shared_factor.fit_shared_dict;
                stacked-SVD computed once and sliced per r_shared)

Budget rule: E*r_ind*(m+n) = m*n + E*r_shared^2 + E*r_p*(m+n).

Metrics: routing-mass-weighted activation error (same as G5) and
Frobenius error. Writes outputs/shared_factor_L3.json; progress in
outputs/shared_factor_L3.progress.

Usage:
    python scripts/v4_shared_factor.py --experts 8
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

INPUTS = ROOT / "outputs" / "ffn_inputs_dense.npz"
PROGRESS = ROOT / "outputs" / "shared_factor_L3.progress"
TOP_K = 6
R_SHARED = (8, 16)
R_INDEP = (128, 512)


def act_error_mass(W_hat, refs, weights, X):
    """Routing-mass-weighted activation error over routed pairs."""
    num = den = 0.0
    for e in refs:
        yhat = X @ np.asarray(W_hat[e], dtype=np.float64).T
        d = refs[e] - yhat
        w = weights[e]
        num += float((w[:, None] * d * d).sum())
        den += float((w[:, None] * refs[e] * refs[e]).sum())
    return num / max(den, 1e-30)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experts", type=int, default=8)
    a = ap.parse_args()

    data = np.load(INPUTS)
    X = np.asarray(data["L3"][:24], dtype=np.float64)     # [24, 7168]
    shards = sorted(glob.glob("D:/hyperv4/models/pro/"
                              "deepseek-ai-DeepSeek-V4-Pro-Q3_K_M-*.gguf"))
    st = vs.ExpertStore(shards[0], extra_shards=shards[1:])
    Wr = st.read_tensor(3, "ffn_gate_inp")
    bias = st.read_tensor(3, "exp_probs_b")

    S_ = np.sqrt(np.log1p(np.exp(X @ Wr.T))) + bias
    ids = np.argpartition(-S_, TOP_K - 1, axis=-1)[:, :TOP_K]
    Wmat = np.zeros_like(S_)
    for t in range(S_.shape[0]):
        Wmat[t, ids[t]] = S_[t, ids[t]]
    mass = Wmat.sum(axis=0)
    order = np.argsort(-mass)
    experts = [int(e) for e in order[: a.experts] if mass[e] > 0]

    t0 = time.time()
    Ws, refs, weights, svds = {}, {}, {}, {}
    for i, e in enumerate(experts):
        W = st.read_expert(3, "ffn_gate_exps", e).astype(np.float64)
        Ws[e] = W
        refs[e] = X @ W.T
        weights[e] = Wmat[:, e]
        svds[e] = np.linalg.svd(W, full_matrices=False)
        PROGRESS.write_text(f"loaded {i + 1}/{len(experts)} (id {e}) "
                            f"({time.time() - t0:.0f}s)\n")
        print(f"expert {e} loaded ({time.time() - t0:.0f}s)", flush=True)

    E = len(experts)
    m, n = next(iter(Ws.values())).shape
    W_stack = np.stack([Ws[e] for e in experts])
    W_shared = W_stack.mean(axis=0)
    R = W_stack - W_shared
    # one-shot stacked SVDs (same math as shared_factor.fit_shared_dict)
    _, _, Vt = np.linalg.svd(R.reshape(-1, n), full_matrices=False)
    _, _, Vh = np.linalg.svd(R.transpose(0, 2, 1).reshape(-1, m),
                             full_matrices=False)
    PROGRESS.write_text(f"stacked SVDs done ({time.time() - t0:.0f}s)\n")
    print(f"stacked SVDs done ({time.time() - t0:.0f}s)", flush=True)

    report = {"layer": 3, "n_tokens": S_.shape[0], "n_experts": E,
              "expert_ids": experts, "m": m, "n": n,
              "independent": {}, "shared": {}}
    for r_ind in R_INDEP:
        ind_params = E * r_ind * (m + n)
        Wh = {e: ((U[:, :r_ind] * S[:r_ind]) @ Vt_[:r_ind])
              for e, (U, S, Vt_) in svds.items()}
        report["independent"][str(r_ind)] = {
            "params": ind_params,
            "act_error": round(act_error_mass(Wh, refs, weights, X), 6),
            "frob_error": round(float(np.mean([
                frob_fraction(Ws[e], Wh[e]) for e in experts])), 6),
        }
        print(f"independent r={r_ind}: "
              f"act={report['independent'][str(r_ind)]['act_error']:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)

    for r_shared in R_SHARED:
        U = Vh[:r_shared].T                         # [m, r_shared]
        V = Vt[:r_shared].T                         # [n, r_shared]
        for r_ind in R_INDEP:
            ind_params = E * r_ind * (m + n)
            shared_base = m * n + E * r_shared * r_shared
            r_p = int((ind_params - shared_base) // (E * (m + n)))
            if r_p < 0:
                report["shared"][f"rs{r_shared}/ind{r_ind}"] = {
                    "params": ind_params, "r_private": None,
                    "note": "shared matrix alone exceeds the budget",
                }
                continue
            Wh = {}
            for j, e in enumerate(experts):
                R_e = Ws[e] - W_shared
                core = U.T @ R_e @ V
                rec = W_shared + (U @ core) @ V.T
                if r_p > 0:
                    leftover = R_e - (U @ core) @ V.T
                    Ue, Se, Vte = np.linalg.svd(leftover,
                                                full_matrices=False)
                    k = min(r_p, len(Se))
                    rec = rec + (Ue[:, :k] * Se[:k]) @ Vte[:k]
                Wh[e] = rec
            report["shared"][f"rs{r_shared}/ind{r_ind}"] = {
                "params": ind_params,
                "r_shared": r_shared,
                "r_private": r_p,
                "act_error": round(act_error_mass(Wh, refs, weights, X), 6),
                "frob_error": round(float(np.mean([
                    frob_fraction(Ws[e], Wh[e]) for e in experts])), 6),
            }
            print(f"shared rs={r_shared} rp={r_p} vs ind={r_ind}: "
                  f"act={report['shared'][f'rs{r_shared}/ind{r_ind}']['act_error']:.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)

    out = ROOT / "outputs" / "shared_factor_L3.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    PROGRESS.write_text(f"done in {time.time() - t0:.0f}s\n")
    print(json.dumps(report, indent=2))
    return 0


def frob_fraction(W, Wh):
    d = W - Wh
    return float(np.sum(d * d) / max(float(np.sum(W * W)), 1e-30))


if __name__ == "__main__":
    raise SystemExit(main())
