"""V4-Coder language isolation — per-language routing census (cluster).

Consumes the pl_prompts.json segment meta + the exp_pl trace npz:
per-segment (language) expert routing mass on layer 3, top-32 per
language, and the pairwise top-32 overlap matrix — the language-
isolation verdict: distinct subnetworks vs shared ones.

Writes pl_census.json into --in.

Usage (on node2):
    python3 scripts/cluster_pl_census.py \
      --shards '/mnt/nas20/models/v4pro/*.gguf' \
      --in /mnt/nas20/exp_pl \
      --prompts /mnt/nas20/exp_pl/pl_prompts.json
"""

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

import v4_ref_serve as vs  # noqa: E402

TOP_K = 6


def softplus(z):
    return np.sqrt(np.log1p(np.exp(z)))


def dense_mass_segment(st, layer, X_seg):
    W = st.read_tensor(layer, "ffn_gate_inp")
    bias = st.read_tensor(layer, "exp_probs_b")
    S = softplus(X_seg @ W.T) + bias
    ids = np.argpartition(-S, TOP_K - 1, axis=-1)[:, :TOP_K]
    M = np.zeros_like(S)
    for t in range(S.shape[0]):
        M[t, ids[t]] = S[t, ids[t]]
    return M.sum(axis=0), ids


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True)
    ap.add_argument("--in", dest="indir", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--layer", type=int, default=3)
    a = ap.parse_args()

    indir = Path(a.indir)
    data = np.load(indir / "ffn_inputs_dense.npz")
    X = np.asarray(data[f"L{a.layer}"], dtype=np.float64)
    meta = json.load(open(a.prompts, encoding="utf-8"))["segments"]
    shards = sorted(glob.glob(a.shards))
    st = vs.ExpertStore(shards[0], extra_shards=shards[1:])

    report = {"layer": a.layer, "top_k": TOP_K,
              "n_tokens": int(X.shape[0]), "languages": {}}
    langs = []
    for seg in meta:
        lang = seg["language"]
        lo, hi = seg["start"], seg["start"] + seg["n"]
        mass, ids = dense_mass_segment(st, a.layer, X[lo:hi])
        top32 = [int(e) for e in np.argsort(-mass)[:32] if mass[e] > 0]
        distinct = int(len(set().union(*(set(ids[t])
                                         for t in range(ids.shape[0])))))
        report["languages"][lang] = {
            "n_tokens": seg["n"], "distinct_experts": distinct,
            "top32_ids": top32,
            "top32_mass_share": round(float(
                mass[top32].sum() / max(mass.sum(), 1e-30)), 4),
        }
        langs.append(lang)
        print(f"{lang}: distinct={distinct} "
              f"top32_mass={report['languages'][lang]['top32_mass_share']:.3f}",
              flush=True)

    # pairwise top-32 overlap matrix
    overlap = {}
    for i, a_l in enumerate(langs):
        for b_l in langs[i + 1:]:
            sa = set(report["languages"][a_l]["top32_ids"])
            sb = set(report["languages"][b_l]["top32_ids"])
            overlap[f"{a_l}/{b_l}"] = len(sa & sb)
            print(f"overlap {a_l}/{b_l}: {len(sa & sb)}/32", flush=True)
    report["overlap_top32"] = overlap

    (indir / "pl_census.json").write_text(json.dumps(report, indent=2))
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
