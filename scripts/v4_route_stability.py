"""G1 slice — routing entropy + top-k stability on real bytes.

The review's G1 demands routing entropy and top-k stability measured
BEFORE any weight surgery. Computed here from the real layer-3 dense
router on the 24-token trace (outputs/ffn_inputs_dense.npz + router
tensor on D:):

  per-token routing entropy : H = -sum p_e log p_e over the top-6
      score-normalized selection (nats)
  selection margin         : score[6th] / score[7th] — decisiveness of
      the cut (large = sharp)
  consecutive-set overlap  : |S_t & S_{t-1}| / 6 — top-k stability
      (1 - overlap is the per-step churn rate)

Writes outputs/route_stability_L3.json.

Usage:
    python scripts/v4_route_stability.py --layer 3
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
TOP_K = 6


def entropy_nats(p):
    p = np.asarray(p, dtype=np.float64)
    p = p / p.sum()
    return float(-np.sum(p * np.log(np.clip(p, 1e-30, None))))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=3)
    a = ap.parse_args()

    data = np.load(INPUTS)
    X = np.asarray(data[f"L{a.layer}"][:24], dtype=np.float64)
    shards = sorted(glob.glob("D:/hyperv4/models/pro/"
                              "deepseek-ai-DeepSeek-V4-Pro-Q3_K_M-*.gguf"))
    st = vs.ExpertStore(shards[0], extra_shards=shards[1:])
    Wr = st.read_tensor(a.layer, "ffn_gate_inp")
    bias = st.read_tensor(a.layer, "exp_probs_b")

    S_ = np.sqrt(np.log1p(np.exp(X @ Wr.T))) + bias      # [t, 384]
    ids = np.argpartition(-S_, TOP_K - 1, axis=-1)[:, :TOP_K]
    sel = np.take_along_axis(S_, ids, axis=-1)

    ents, margins, overlaps = [], [], []
    for t in range(S_.shape[0]):
        p = sel[t] / sel[t].sum()
        ents.append(entropy_nats(p))
        if t > 0:
            overlap = len(set(ids[t]) & set(ids[t - 1])) / TOP_K
            overlaps.append(overlap)
    # margin: 6th selected score over the best NON-selected score
    for t in range(S_.shape[0]):
        mask = np.zeros(S_.shape[1], dtype=bool)
        mask[ids[t]] = True
        kth = sel[t].min()
        best_out = S_[t][~mask].max()
        margins.append(float(kth / max(best_out, 1e-30)))

    report = {
        "layer": a.layer,
        "n_tokens": int(S_.shape[0]),
        "top_k": TOP_K,
        "per_token_entropy_nats": [round(float(h), 4) for h in ents],
        "mean_entropy_nats": round(float(np.mean(ents)), 4),
        "per_token_margin": [round(float(m), 4) for m in margins],
        "mean_margin": round(float(np.mean(margins)), 4),
        "consecutive_overlap": [round(float(o), 4) for o in overlaps],
        "mean_overlap": round(float(np.mean(overlaps)), 4),
        "mean_churn_rate": round(float(1.0 - np.mean(overlaps)), 4),
        "distinct_experts": int(len(set().union(
            *(set(ids[t]) for t in range(ids.shape[0]))))),
    }
    out = ROOT / "outputs" / "route_stability_L3.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
