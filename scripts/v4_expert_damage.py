"""G7 — first real expert-damage estimates for CVaR tail pruning.

Computes each routed expert's ACTUAL output contribution over the 24
real layer-3 tokens (gate+up+down forward per expert) and feeds the
per-(expert, token) damage matrix to ultratensor.conditional.cvar:

    D(e, t) = ||y_e(x_t)|| / sum_e' ||y_e'(x_t)||   over the selected 6
    (0 for tokens where e is not selected — removing an unselected
    expert cannot damage that token, first order)

Caveat (honest): this is the first-order magnitude proxy, not the KL
ablation the review ultimately wants — it ignores router
renormalization and the shared experts. It answers the first question:
does the CVaR double gate find any safe-to-prune experts on real
routed traffic?

Writes outputs/expert_damage_L3.json; progress in
outputs/expert_damage_L3.progress.

Usage:
    python scripts/v4_expert_damage.py --experts 8
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
from ultratensor.conditional.cvar import prune_report  # noqa: E402

INPUTS = ROOT / "outputs" / "ffn_inputs_dense.npz"
PROGRESS = ROOT / "outputs" / "expert_damage_L3.progress"
TOP_K = 6


def silu(g):
    return g / (1.0 + np.exp(-g))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=0.9)
    a = ap.parse_args()

    data = np.load(INPUTS)
    X = np.asarray(data["L3"][:24], dtype=np.float64)
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
    contribs = {}
    for i, e in enumerate(experts):
        gate = st.read_expert(3, "ffn_gate_exps", e)
        up = st.read_expert(3, "ffn_up_exps", e)
        down = st.read_expert(3, "ffn_down_exps", e)
        g = X @ gate.T
        h = silu(g) * (X @ up.T)          # [24, 3072] swiglu intermediate
        y = h @ down.T                    # [24, 7168] expert output
        contribs[e] = np.linalg.norm(y, axis=-1)     # [24]
        PROGRESS.write_text(f"expert {i + 1}/{len(experts)} (id {e}) "
                            f"({time.time() - t0:.0f}s)\n")
        print(f"expert {e} outputs computed ({time.time() - t0:.0f}s)",
              flush=True)

    # D(e, t): fractional contribution over the token's selected set
    D = np.zeros((len(experts), S_.shape[0]))
    for t in range(S_.shape[0]):
        sel = set(int(i) for i in ids[t])
        tot = sum(contribs[e][t] for e in experts if e in sel)
        for j, e in enumerate(experts):
            D[j, t] = contribs[e][t] / tot if (e in sel and tot > 0) else 0.0

    rep = prune_report(D, alpha=a.alpha)
    rep["expert_ids"] = experts
    rep["n_routed_tokens"] = {str(e): int((Wmat[:, e] > 0).sum())
                              for e in experts}
    rep["note"] = ("first-order output-magnitude proxy; ignores router "
                   "renormalization and shared experts")
    out = ROOT / "outputs" / "expert_damage_L3.json"
    out.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    PROGRESS.write_text(f"done in {time.time() - t0:.0f}s\n")
    print(json.dumps(rep, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
