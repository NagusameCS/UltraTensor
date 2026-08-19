"""Cluster step 2b: held-out projector curves + factored controller.

Chained after cluster_dense_trace.py on a >=32-token npz. Self-contained
(no tokenizers). Runs the NEW honest estimators:

  - G4 held-out projector curves (fit train / eval hold, both split
    directions) via ultratensor.conditional.actweight.heldout_rank_curves;
  - G9 factored-ridge controller (X^T B form, params n*(d+c)) with
    held-out top-k agreement on layer 3.

Usage (on node2):
    python3 scripts/cluster_subspace.py \
      --shards '/mnt/nas20/models/v4pro/*.gguf' \
      --in /mnt/nas20/exp96

Writes subspace_proj.json + controller_shrink.json into --in.
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


def topk_agreement(pred, true, k):
    pk = np.argpartition(-pred, k - 1, axis=-1)[:, :k]
    tk = np.argpartition(-true, k - 1, axis=-1)[:, :k]
    return float(np.mean([len(set(a) & set(b)) / k
                          for a, b in zip(pk, tk)]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True)
    ap.add_argument("--in", dest="indir", required=True)
    ap.add_argument("--train", type=int, default=64)
    ap.add_argument("--ranks", type=int, nargs="+",
                    default=[4, 8, 16, 24, 32, 48, 64])
    a = ap.parse_args()

    indir = Path(a.indir)
    data = np.load(indir / "ffn_inputs_dense.npz")
    shards = sorted(glob.glob(a.shards))
    st = vs.ExpertStore(shards[0], extra_shards=shards[1:])
    t0 = time.time()

    # G4 held-out projector curves, gate experts 0/1, layers 0/3
    report = {}
    for layer in (0, 3):
        key = f"L{layer}"
        if key not in data:
            continue
        X = np.asarray(data[key], dtype=np.float64)
        n = min(a.train, X.shape[0] - 1)
        for e in (0, 1):
            W = st.read_expert(layer, "ffn_gate_exps", e).astype(np.float64)
            entry = {"shape": list(W.shape), "n": int(X.shape[0])}
            for name, (tr, ho) in {
                    "fwd": (X[:n], X[n:]),
                    "rev": (X[-n:], X[:-n])}.items():
                ranks = [r for r in a.ranks if r <= tr.shape[0]]
                c = heldout_rank_curves(W, tr, ho, ranks=ranks)
                entry[name] = {
                    "ranks": c["ranks"].tolist(),
                    "hold_pca": [round(float(x), 6)
                                 for x in c["hold_pca"]],
                    "hold_weighted": [round(float(x), 6)
                                      for x in c["hold_weighted"]],
                }
            report[f"{layer}/{e}"] = entry
            rs = entry["fwd"]["ranks"]
            i8 = rs.index(8) if 8 in rs else len(rs) - 1
            print(f"proj L{layer}/e{e}: fwd hold_pca@8="
                  f"{entry['fwd']['hold_pca'][i8]:.4f} "
                  f"hold_w@8={entry['fwd']['hold_weighted'][i8]:.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    (indir / "subspace_proj.json").write_text(json.dumps(report, indent=2))

    # G9 factored-ridge controller, layer 3
    if "L3" in data:
        X = np.asarray(data["L3"], dtype=np.float64)
        W = st.read_tensor(3, "ffn_gate_inp")
        bias_t = st._tensor(3, "exp_probs_b")
        bias = st.read_tensor(3, "exp_probs_b") if bias_t is not None else 0.0
        T = np.sqrt(np.log1p(np.exp(X @ W.T))) + bias
        d, c = W.shape[1], W.shape[0]
        n = min(a.train, X.shape[0] - 1)

        def labels(Tt):
            ids = np.argpartition(-Tt, TOP_K - 1, axis=-1)[:, :TOP_K]
            Y = np.zeros_like(Tt)
            Y[np.arange(Tt.shape[0])[:, None], ids] = 1.0
            return Y

        methods = {}
        for sname, (itr, iho) in {
                "fwd": ((X[:n], np.arange(n)),
                        (X[n:], np.arange(n, X.shape[0]))),
                "rev": ((X[-n:], np.arange(X.shape[0] - n, X.shape[0])),
                        (X[:-n], np.arange(X.shape[0] - n)))}.items():
            Xtr, Xho = itr[0], iho[0]
            Ttr, Tho = T[itr[1]], T[iho[1]]
            B = np.linalg.solve(Xtr @ Xtr.T + np.eye(Xtr.shape[0]), labels(Ttr))
            pred = (Xho @ Xtr.T) @ B
            methods[sname] = {
                "factored_hold_agreement":
                    round(topk_agreement(pred, Tho, TOP_K), 4),
                "params": int(Xtr.shape[0] * (d + c)),
            }
            print(f"g9 {sname}: factored hold="
                  f"{methods[sname]['factored_hold_agreement']:.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        (indir / "controller_shrink.json").write_text(
            json.dumps(methods, indent=2))

    print(f"done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
