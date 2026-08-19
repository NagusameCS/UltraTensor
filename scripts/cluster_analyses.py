"""Cluster step 2: run G2/G4/G9 analyses on the NAS-resident npz.

Chained after cluster_dense_trace.py. Self-contained (no tokenizers).
--shards '/mnt/nas20/models/v4pro/*.gguf'
--in  /mnt/nas20/exp
Writes expert_sim_activation.json, actweight_curves.json,
router_distill.json into the same dir.
"""

import argparse
import glob
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

import v4_ref_serve as vs  # noqa: E402
from ultratensor.conditional.actweight import rank_error_curve  # noqa: E402


def expert_outputs(st, layer, e, inputs):
    gate = st.read_expert(layer, "ffn_gate_exps", e)
    up = st.read_expert(layer, "ffn_up_exps", e)
    down = st.read_expert(layer, "ffn_down_exps", e)
    outs = []
    for x in inputs:
        g = x @ gate.T
        silu = g / (1.0 + np.exp(-g))
        outs.append((silu * (x @ up.T)) @ down.T)
    return np.stack(outs)


def pairwise_cos(mats):
    E = len(mats)
    sim = np.zeros((E, E), np.float64)
    for i in range(E):
        for j in range(E):
            a, b = mats[i], mats[j]
            na = np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12
            nb = np.linalg.norm(b, axis=-1, keepdims=True) + 1e-12
            sim[i, j] = float(np.mean(np.sum(a * b, axis=-1)
                                      / (na * nb).ravel()))
    return sim


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True)
    ap.add_argument("--in", dest="indir", required=True)
    a = ap.parse_args()

    indir = Path(a.indir)
    data = np.load(indir / "ffn_inputs_dense.npz")
    shards = sorted(glob.glob(a.shards))
    st = vs.ExpertStore(shards[0], extra_shards=shards[1:])
    t0 = time.time()

    # G2: activation-space expert similarity, layers 0 and 3, 12 probe
    g2 = {}
    for layer in (0, 3):
        key = f"L{layer}"
        if key not in data:
            continue
        inputs = [x.astype(np.float32) for x in data[key][:24]]
        mats, wc = [], []
        for e in range(12):
            mats.append(expert_outputs(st, layer, e, inputs))
            W = st.read_expert(layer, "ffn_down_exps", e).astype(np.float64)
            wc.append((W / (np.linalg.norm(W) + 1e-12)).ravel())
        sim = pairwise_cos(mats)
        wcorr = np.array([[float(a @ b) for b in wc] for a in wc])
        off = ~np.eye(12, dtype=bool)
        g2[layer] = {
            "mean_offdiag_act_cos": float(sim[off].mean()),
            "max_offdiag_act_cos": float(sim[off].max()),
            "mean_offdiag_weight_corr": float(wcorr[off].mean()),
            "max_offdiag_weight_corr": float(wcorr[off].max()),
        }
        print(f"G2 layer {layer}: act_cos mean={sim[off].mean():.4f} "
              f"weight_corr mean={wcorr[off].mean():.2e}", flush=True)
    (indir / "expert_sim_activation.json").write_text(
        json.dumps(g2, indent=2))

    # G4: activation-weighted rank-error curves on real inputs
    ranks = [128, 256, 384, 512, 768, 1024, 1280, 1536]
    g4 = {}
    for layer in (0, 3):
        key = f"L{layer}"
        if key not in data:
            continue
        inputs = np.asarray(data[key][:24], dtype=np.float64)
        # gate/up map hidden (7168) -> intermediate; down maps the
        # 3072-dim swiglu intermediate back, so hidden-state inputs only
        # match gate/up.
        W = st.read_expert(layer, "ffn_gate_exps", 0).astype(np.float64)
        ranks_l = [r for r in ranks if r <= min(W.shape)]
        c = rank_error_curve(W, inputs, ranks=ranks_l)
        g4[layer] = {
            "k95_frob": c.k95_frob, "k95_act": c.k95_act,
            "act_at_512": round(float(c.act[ranks_l.index(512)]), 4),
            "frob_at_512": round(float(c.frob[ranks_l.index(512)]), 4),
        }
        print(f"G4 layer {layer}: k95_frob={c.k95_frob} "
              f"k95_act={c.k95_act}", flush=True)
    (indir / "actweight_curves.json").write_text(json.dumps(g4, indent=2))

    # G9: ridge router distillation on layer 3
    key = "L3"
    if key in data:
        X = np.asarray(data[key], dtype=np.float64)
        W = st.read_tensor(3, "ffn_gate_inp")
        bias_t = st._tensor(3, "exp_probs_b")
        bias = st.read_tensor(3, "exp_probs_b") if bias_t is not None else 0.0
        z = X @ W.T
        s = np.sqrt(np.log1p(np.exp(z)))
        sel = s + bias
        ids = np.argpartition(-sel, 5, axis=-1)[:, :6]
        n_experts = sel.shape[1]
        Y = np.zeros((X.shape[0], n_experts))
        for t in range(X.shape[0]):
            Y[t, ids[t]] = 1.0
        lam = 1.0
        A = X.T @ X + lam * np.eye(X.shape[1])
        params = np.linalg.solve(A, X.T @ Y)
        scores = X @ params
        pred = np.argpartition(-scores, 5, axis=-1)[:, :6]
        agree = sum(len(set(pred[t]) & set(ids[t])) / 6
                    for t in range(X.shape[0])) / X.shape[0]
        g9 = {"n_samples": int(X.shape[0]),
              "controller_params": int(params.size),
              "router_params": int(W.size) + int(np.asarray(bias).size),
              "topk_agreement": round(float(agree), 4)}
        (indir / "router_distill.json").write_text(json.dumps(g9, indent=2))
        print(f"G9: agreement={g9['topk_agreement']} "
              f"params {g9['controller_params']/1e6:.1f}M vs "
              f"{g9['router_params']/1e6:.1f}M", flush=True)

    print(f"all analyses done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
