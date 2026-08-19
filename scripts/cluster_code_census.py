"""V4-Coder step 3 — code-domain routing census (cluster, after exp_code).

Given the code-domain trace npz (cluster_dense_trace.py run on the
code battery), computes:

  1. per-expert routing mass on CODE traffic (top-6 score-weighted);
  2. top-K code experts per layer + distinct-expert count;
  3. overlap with GENERAL-traffic experts (from exp96 npz when
     present): shared experts, code-exclusive experts;
  4. domain-subspace test: held-out projector curves fitted on code
     inputs only (train 64 / hold rest) for gate expert 0 of L0/L3,
     compared against the general-traffic curves.

Writes code_census.json into --in.

Usage (on node2):
    python3 scripts/cluster_code_census.py \
      --shards '/mnt/nas20/models/v4pro/*.gguf' \
      --in /mnt/nas20/exp_code \
      --general /mnt/nas20/exp96
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
from ultratensor.conditional.actweight import heldout_rank_curves  # noqa: E402

TOP_K = 6


def softplus(z):
    return np.sqrt(np.log1p(np.exp(z)))


def expert_mass(st, layer, X):
    """Score-weighted per-expert routing mass (DENSE layers only)."""
    W = st.read_tensor(layer, "ffn_gate_inp")
    bias = st.read_tensor(layer, "exp_probs_b")
    S = softplus(X @ W.T) + bias
    ids = np.argpartition(-S, TOP_K - 1, axis=-1)[:, :TOP_K]
    M = np.zeros_like(S)
    for t in range(S.shape[0]):
        M[t, ids[t]] = S[t, ids[t]]
    return M.sum(axis=0), ids


def hash_mass(st, layer, token_ids):
    """Per-expert routing mass for HASH layers: deterministic table."""
    table = st.read_tensor(layer, "ffn_gate_tid2eid")     # [vocab, 6]
    mass = np.zeros(384)
    sets = []
    for t in token_ids:
        if 0 <= t < table.shape[0]:
            row = table[t][:TOP_K]
            sets.append(set(int(e) for e in row))
            for e in row:
                mass[int(e)] += 1.0
    return mass, sets


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True)
    ap.add_argument("--in", dest="indir", required=True)
    ap.add_argument("--general", default=None)
    ap.add_argument("--prompts", default=None,
                    help="prompts json with token_ids (for hash-layer mass)")
    ap.add_argument("--layers", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--train", type=int, default=64)
    a = ap.parse_args()

    indir = Path(a.indir)
    data = np.load(indir / "ffn_inputs_dense.npz")
    token_ids = None
    if a.prompts:
        token_ids = [int(t) for t in json.load(
            open(a.prompts, encoding="utf-8"))["token_ids"]]
    shards = sorted(glob.glob(a.shards))
    st = vs.ExpertStore(shards[0], extra_shards=shards[1:])
    t0 = time.time()

    report = {"n_code_tokens": {}, "top_k": TOP_K, "layers": {}}
    for layer in a.layers:
        key = f"L{layer}"
        if layer < st.n_hash_layers:
            if not token_ids:
                print(f"hash layer {layer} needs --prompts; skip")
                continue
            mass, sets = hash_mass(st, layer, token_ids)
            distinct = int(len(set().union(*sets)) if sets else 0)
            entry = {"distinct_experts": distinct, "n_tokens": len(token_ids)}
        else:
            if key not in data:
                continue
            X = np.asarray(data[key], dtype=np.float64)
            mass, ids = expert_mass(st, layer, X)
            distinct = int(len(set().union(*(set(ids[t]) for t in range(ids.shape[0])))))
            entry = {"distinct_experts": distinct, "n_tokens": int(X.shape[0])}
            report["n_code_tokens"][key] = int(X.shape[0])
        order = np.argsort(-mass)
        top = [int(e) for e in order[:64] if mass[e] > 0]
        entry["top64_mass_share"] = round(float(
            mass[top].sum() / max(mass.sum(), 1e-30)), 4)
        entry["top64_ids"] = top[:64]
        # general-traffic overlap (dense layers only)
        if a.general and layer >= st.n_hash_layers:
            gdata = np.load(Path(a.general) / "ffn_inputs_dense.npz")
            if key in gdata:
                GX = np.asarray(gdata[key], dtype=np.float64)
                gmass, _ = expert_mass(st, layer, GX)
                gorder = np.argsort(-gmass)
                gtop = set(int(e) for e in gorder[:64] if gmass[e] > 0)
                ctop = set(top[:64])
                entry["overlap_with_general_top64"] = len(ctop & gtop)
                entry["code_exclusive_top64"] = sorted(ctop - gtop)[:32]
                entry["shared_top64"] = sorted(ctop & gtop)[:32]
        report["layers"][key] = entry
        print(f"layer {layer}: distinct={distinct} "
              f"top64_mass={entry['top64_mass_share']:.3f} "
              f"({time.time() - t0:.0f}s)", flush=True)

    # domain-subspace test: held-out projector on code inputs only
    proj = {}
    for layer in a.layers:
        key = f"L{layer}"
        if layer < st.n_hash_layers or key not in data:
            continue
        X = np.asarray(data[key], dtype=np.float64)
        n = min(a.train, X.shape[0] - 1)
        W = st.read_expert(layer, "ffn_gate_exps", 0).astype(np.float64)
        c = heldout_rank_curves(W, X[:n], X[n:],
                                ranks=[4, 8, 16, 24, 32, 48, 64])
        proj[key] = {
            "ranks": c["ranks"].tolist(),
            "hold_pca": [round(float(x), 4) for x in c["hold_pca"]],
            "hold_weighted": [round(float(x), 4) for x in c["hold_weighted"]],
        }
        i8 = proj[key]["ranks"].index(8)
        print(f"code subspace L{layer}: hold_pca@8="
              f"{proj[key]['hold_pca'][i8]:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)
    report["code_subspace_proj"] = proj

    # G9-on-code: factored score-regression controller, train/hold
    if "L3" in data:
        X = np.asarray(data["L3"], dtype=np.float64)
        W = st.read_tensor(3, "ffn_gate_inp")
        bias_t = st._tensor(3, "exp_probs_b")
        bias = st.read_tensor(3, "exp_probs_b") if bias_t is not None else 0.0
        T = np.sqrt(np.log1p(np.exp(X @ W.T))) + bias
        d, c = W.shape[1], W.shape[0]
        n = min(a.train, X.shape[0] - 1)

        def topk_agree(pred, true):
            pk = np.argpartition(-pred, TOP_K - 1, axis=-1)[:, :TOP_K]
            tk = np.argpartition(-true, TOP_K - 1, axis=-1)[:, :TOP_K]
            return float(np.mean([len(set(x) & set(y)) / TOP_K
                                  for x, y in zip(pk, tk)]))

        g9 = {}
        for sname, (itr, iho) in {
                "fwd": ((X[:n], np.arange(n)),
                        (X[n:], np.arange(n, X.shape[0]))),
                "rev": ((X[-n:], np.arange(X.shape[0] - n, X.shape[0])),
                        (X[:-n], np.arange(X.shape[0] - n)))}.items():
            Xtr, Xho = itr[0], iho[0]
            Ttr, Tho = T[itr[1]], T[iho[1]]
            B = np.linalg.solve(Xtr @ Xtr.T + np.eye(Xtr.shape[0]), Ttr)
            pred = (Xho @ Xtr.T) @ B
            g9[sname] = {
                "hold_rel_l1": round(float(np.abs(pred - Tho).sum()
                                           / max(np.abs(Tho).sum(), 1e-30)), 4),
                "hold_agreement": round(topk_agree(pred, Tho), 4),
                "params": int(Xtr.shape[0] * (d + c)),
            }
            print(f"g9-code {sname}: rel_l1={g9[sname]['hold_rel_l1']:.4f} "
                  f"agree={g9[sname]['hold_agreement']:.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        report["g9_code"] = g9

    (indir / "code_census.json").write_text(json.dumps(report, indent=2))
    print(f"done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
