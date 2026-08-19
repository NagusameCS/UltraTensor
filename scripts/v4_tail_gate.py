"""V4-Coder step 5 — CVaR tail gate: code vs rare-code damage veto.

Phase-2 quality gate for subnetwork extraction. For the top-K code
experts of layer 3 it computes per-(expert, token) output-magnitude
damages on CODE traffic and on RARE/ADVERSARIAL code traffic (both
traces from cluster runs), runs the CVaR double gate per domain, and
emits the VETO list:

  veto = safe-to-prune on code AND NOT safe-to-prune on rare

Those experts are cheap on ordinary code but critical on the tail —
dropping them would silently break rare code. The coder extraction
must keep the vetoed experts.

Writes outputs/tail_gate_L3.json. Needs the exp_code and exp_rare
npz files locally (outputs/exp_code_ffn_inputs_dense.npz,
outputs/exp_rare_ffn_inputs_dense.npz).

Usage:
    python scripts/v4_tail_gate.py --layer 3 --experts 8
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
from ultratensor.conditional.cvar import cvar, prune_mask  # noqa: E402

CODE_NPZ = ROOT / "outputs" / "exp_code_ffn_inputs_dense.npz"
RARE_NPZ = ROOT / "outputs" / "exp_rare_ffn_inputs_dense.npz"
TOP_K = 6


def silu(g):
    return g / (1.0 + np.exp(-g))


def expert_contribs(st, layer, X, experts):
    """Fractional output contribution D[e, t] for each expert."""
    D = np.zeros((len(experts), X.shape[0]))
    for i, e in enumerate(experts):
        gate = st.read_expert(layer, "ffn_gate_exps", e)
        up = st.read_expert(layer, "ffn_up_exps", e)
        down = st.read_expert(layer, "ffn_down_exps", e)
        g = X @ gate.T
        y = (silu(g) * (X @ up.T)) @ down.T
        D[i] = np.linalg.norm(y, axis=-1)
    # normalize over the token's selected set
    Wr = st.read_tensor(layer, "ffn_gate_inp")
    bias = st.read_tensor(layer, "exp_probs_b")
    S = np.sqrt(np.log1p(np.exp(X @ Wr.T))) + bias
    ids = np.argpartition(-S, TOP_K - 1, axis=-1)[:, :TOP_K]
    for t in range(X.shape[0]):
        sel = set(int(i) for i in ids[t])
        tot = sum(D[j, t] for j, e in enumerate(experts) if e in sel)
        for j, e in enumerate(experts):
            D[j, t] = D[j, t] / tot if (e in sel and tot > 0) else 0.0
    return D


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=3)
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=0.9)
    ap.add_argument("--code-npz", default=str(CODE_NPZ))
    ap.add_argument("--rare-npz", default=str(RARE_NPZ))
    a = ap.parse_args()

    for p in (a.code_npz, a.rare_npz):
        if not Path(p).exists():
            print(f"missing {p}; run the cluster code/rare traces first")
            return 2

    shards = sorted(glob.glob("D:/hyperv4/models/pro/"
                              "deepseek-ai-DeepSeek-V4-Pro-Q3_K_M-*.gguf"))
    st = vs.ExpertStore(shards[0], extra_shards=shards[1:])
    layer = a.layer
    t0 = time.time()

    Xc = np.asarray(np.load(a.code_npz)[f"L{layer}"], dtype=np.float64)
    Xr = np.asarray(np.load(a.rare_npz)[f"L{layer}"], dtype=np.float64)

    # top-K code experts by routing mass
    Wr = st.read_tensor(layer, "ffn_gate_inp")
    bias = st.read_tensor(layer, "exp_probs_b")
    S = np.sqrt(np.log1p(np.exp(Xc @ Wr.T))) + bias
    ids = np.argpartition(-S, TOP_K - 1, axis=-1)[:, :TOP_K]
    M = np.zeros_like(S)
    for t in range(S.shape[0]):
        M[t, ids[t]] = S[t, ids[t]]
    mass = M.sum(axis=0)
    experts = [int(e) for e in np.argsort(-mass)[: a.experts]
               if mass[e] > 0]

    Dc = expert_contribs(st, layer, Xc, experts)
    Dr = expert_contribs(st, layer, Xr, experts)

    report = {"layer": layer, "alpha": a.alpha,
              "expert_ids": experts,
              "n_code_tokens": int(Xc.shape[0]),
              "n_rare_tokens": int(Xr.shape[0]), "experts": {}}
    # cohort-level double gates per domain (median thresholds over ALL
    # candidates — a single-row prune_mask degenerates to self-compare)
    safe_code = np.asarray(prune_mask(Dc, a.alpha))
    safe_rare = np.asarray(prune_mask(Dr, a.alpha))
    for j, e in enumerate(experts):
        mc, cc = Dc[j].mean(), cvar(Dc[j], a.alpha)
        mr, cr = Dr[j].mean(), cvar(Dr[j], a.alpha)
        report["experts"][str(e)] = {
            "code_mean": round(float(mc), 4),
            "code_cvar": round(float(cc), 4),
            "code_worst": round(float(Dc[j].max()), 4),
            "rare_mean": round(float(mr), 4),
            "rare_cvar": round(float(cr), 4),
            "rare_worst": round(float(Dr[j].max()), 4),
            "rare_over_code_cvar": round(float(cr / max(cc, 1e-30)), 2),
            "safe_on_code": bool(safe_code[j]),
            "safe_on_rare": bool(safe_rare[j]),
            "VETOED": bool(safe_code[j] and not safe_rare[j]),
        }
        print(f"e{e}: code mean/cvar {mc:.3f}/{cc:.3f} | "
              f"rare mean/cvar {mr:.3f}/{cr:.3f} "
              f"veto={safe_code[j] and not safe_rare[j]} "
              f"({time.time() - t0:.0f}s)", flush=True)
    report["veto_list"] = [e for e in experts
                           if report["experts"][str(e)]["VETOED"]]

    out = ROOT / "outputs" / f"tail_gate_L{layer}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"veto_list": report["veto_list"]}, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
