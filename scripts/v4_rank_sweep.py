"""G5 — conditional-rank sweep on real routing mass (layer 3 gate experts).

The review demand: is per-expert conditional rank better than uniform
rank at equal budget? Compared over the ROUTED expert set (only routed
experts can contribute activation error), four policies at equal total
rank budget:

  fixed  : uniform rank
  mass   : rank proportional to routing mass (score-weighted count)
  sqrt   : water-filling, rank proportional to sqrt(mass)
  top4   : whole budget over the top-4 experts, rank 1 elsewhere

Metric: routing-mass-weighted activation error over routed
(expert, token) pairs:

    E = sum_{e,t routed} w_{e,t} ||(W_e - W_e^r) x_t||^2
        / sum_{e,t} w_{e,t} ||W_e x_t||^2

Uses the 24 real layer-3 hidden states (outputs/ffn_inputs_dense.npz)
and the top-12 routed gate experts. Writes outputs/rank_sweep_L3.json;
progress in outputs/rank_sweep_L3.progress (long: ~3 min/expert load).

Usage:
    python scripts/v4_rank_sweep.py --experts 12
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
from ultratensor.conditional.rank_policies import gini  # noqa: E402

INPUTS = ROOT / "outputs" / "ffn_inputs_dense.npz"
PROGRESS = ROOT / "outputs" / "rank_sweep_L3.progress"
TOP_K = 6
BUDGETS = (64, 128, 256)


def svd_full(W):
    return np.linalg.svd(W, full_matrices=False)


def truncate(U, S, Vt, r):
    if r >= len(S):
        return None                      # None == full-rank reference
    return (U[:, :r] * S[:r]) @ Vt[:r]


def alloc_fixed(mass, B):
    E = len(mass)
    return np.full(E, max(1, B // E), dtype=int)


def alloc_mass(mass, B):
    r = np.round(B * mass / mass.sum()).astype(int)
    return np.clip(r, 1, B)


def alloc_sqrt(mass, B):
    s = np.sqrt(mass)
    r = np.round(B * s / s.sum()).astype(int)
    return np.clip(r, 1, B)


def alloc_top4(mass, B):
    order = np.argsort(-mass)
    r = np.ones(len(mass), dtype=int)
    r[order[:4]] = max(1, B // 4)
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experts", type=int, default=12)
    a = ap.parse_args()

    data = np.load(INPUTS)
    X = np.asarray(data["L3"][:24], dtype=np.float64)     # [24, 7168]
    shards = sorted(glob.glob("D:/hyperv4/models/pro/"
                              "deepseek-ai-DeepSeek-V4-Pro-Q3_K_M-*.gguf"))
    st = vs.ExpertStore(shards[0], extra_shards=shards[1:])
    Wr = st.read_tensor(3, "ffn_gate_inp")                # router [384,7168]
    bias = st.read_tensor(3, "exp_probs_b")

    S_ = np.sqrt(np.log1p(np.exp(X @ Wr.T))) + bias       # [24, 384] scores
    ids = np.argpartition(-S_, TOP_K - 1, axis=-1)[:, :TOP_K]
    Wmat = np.zeros_like(S_)                              # routing weights
    for t in range(S_.shape[0]):
        Wmat[t, ids[t]] = S_[t, ids[t]]
    mass = Wmat.sum(axis=0)                               # per-expert mass
    order = np.argsort(-mass)[: a.experts]                # top routed experts
    experts = [int(e) for e in order if mass[e] > 0]

    t0 = time.time()
    svds, refs, weights = {}, {}, {}
    for i, e in enumerate(experts):
        W = st.read_expert(3, "ffn_gate_exps", e).astype(np.float64)
        U, S, Vt = svd_full(W)
        svds[e] = (U, S, Vt)
        refs[e] = X @ W.T                                 # [24, 3072]
        weights[e] = Wmat[:, e]                           # [24]
        PROGRESS.write_text(
            f"expert {i + 1}/{len(experts)} (id {e}) loaded "
            f"({time.time() - t0:.0f}s)\n")
        print(f"expert {e} loaded ({time.time() - t0:.0f}s)", flush=True)

    # per-policy activation error at each budget
    policies = {"fixed": alloc_fixed, "mass": alloc_mass,
                "sqrt": alloc_sqrt, "top4": alloc_top4}
    report = {"layer": 3, "n_tokens": S_.shape[0],
              "n_experts": len(experts), "expert_ids": experts,
              "budgets": list(BUDGETS), "policies": {}}
    mass_top = mass[experts]
    for pname, alloc in policies.items():
        report["policies"][pname] = {}
        for B in BUDGETS:
            r = alloc(mass_top, B)
            num = den = 0.0
            for j, e in enumerate(experts):
                U, S, Vt = svds[e]
                Wr_hat = truncate(U, S, Vt, int(r[j]))
                yhat = refs[e] if Wr_hat is None else X @ Wr_hat.T
                d = refs[e] - yhat
                w = weights[e]
                num += float((w[:, None] * d * d).sum())
                den += float((w[:, None] * refs[e] * refs[e]).sum())
            err = num / max(den, 1e-30)
            report["policies"][pname][str(B)] = {
                "ranks": r.tolist(),
                "act_error": round(err, 6),
                "gini": round(gini(r.astype(float)), 4),
            }
            print(f"{pname} B={B}: act_error={err:.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)

    out = ROOT / "outputs" / "rank_sweep_L3.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    PROGRESS.write_text(f"done in {time.time() - t0:.0f}s\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
