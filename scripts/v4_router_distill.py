"""G9 — tiny deployment controller: router distillation smoke test.

Distills a layer's dense router into a small ridge classifier over the
REAL MoE input hidden states saved by v4_router_trace_dense.py:
    y_e = 1[e in selected set]  <-  ridge(h)
then measures top-k agreement of the distilled controller against the
real router. With ~24 samples this is a smoke test, not a model; it
establishes the pipeline and the agreement metric.

Usage (after v4_router_trace_dense.py):
    python scripts/v4_router_distill.py --layer 3
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

INPUTS = ROOT / "outputs" / "ffn_inputs_dense.npz"


def ridge_classifier(X, Y, lam=1e-2):
    """Closed-form ridge: W = (X^T X + lam I)^-1 X^T Y."""
    n, d = X.shape
    A = X.T @ X + lam * np.eye(d)
    return np.linalg.solve(A, X.T @ Y)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=3)
    ap.add_argument("--top-k", type=int, default=6)
    a = ap.parse_args()

    if not INPUTS.exists():
        print("run scripts/v4_router_trace_dense.py first")
        return 2
    data = np.load(INPUTS)
    key = f"L{a.layer}"
    if key not in data:
        print(f"{key} missing from npz")
        return 2

    X = np.asarray(data[key], dtype=np.float64)          # [steps, 7168]
    log_path = ROOT / "outputs" / "router_trace_dense.json"
    trace = json.loads(log_path.read_text(encoding="utf-8"))
    layer_log = trace["layers"][str(a.layer)]
    n_steps = layer_log["n_steps"]
    X = X[:n_steps]

    # rebuild the selected-set membership from the logged ids
    # (re-read the trace entries from the npz-companion logs not stored;
    # instead re-run routes are logged inside the json only as counts, so
    # we re-derive labels from the stored trace file if present, else
    # fall back to deterministic reconstruction via the router itself.)
    import v4_ref_serve as vs  # noqa: E402
    shards = sorted(glob.glob("D:/hyperv4/models/pro/"
                              "deepseek-ai-DeepSeek-V4-Pro-Q3_K_M-*.gguf"))
    st = vs.ExpertStore(shards[0], extra_shards=shards[1:])

    W = st.read_tensor(a.layer, "ffn_gate_inp")
    bias_t = st._tensor(a.layer, "exp_probs_b")
    bias = st.read_tensor(a.layer, "exp_probs_b") if bias_t is not None else 0.0
    z = X @ W.T
    s = np.sqrt(np.log1p(np.exp(z)))
    sel = s + bias
    ids = np.argpartition(-sel, a.top_k - 1, axis=-1)[:, :a.top_k]

    n_experts = sel.shape[1]
    Y = np.zeros((n_steps, n_experts), dtype=np.float64)
    for t in range(n_steps):
        Y[t, ids[t]] = 1.0

    # tiny controller: one ridge weight per expert (n_experts x 7168)
    params = ridge_classifier(X, Y, lam=1.0)      # [7168, n_experts]
    scores = X @ params
    pred = np.argpartition(-scores, a.top_k - 1, axis=-1)[:, :a.top_k]

    agree = 0
    for t in range(n_steps):
        agree += len(set(pred[t]) & set(ids[t])) / a.top_k
    agreement = agree / n_steps

    # honest G9: the in-sample number overfits. Add the FACTORED ridge
    # (W_c = X^T B, params n*(d+c)) evaluated on a train/hold split.
    n_tr = max(2, n_steps * 2 // 3)
    held = {}
    for name, (itr, iho) in {
            "fwd": (np.arange(n_tr), np.arange(n_tr, n_steps)),
            "rev": (np.arange(n_steps - n_tr, n_steps),
                    np.arange(n_steps - n_tr))}.items():
        B = np.linalg.solve(X[itr] @ X[itr].T + np.eye(len(itr)),
                            Y[itr])
        sc = (X[iho] @ X[itr].T) @ B
        hp = np.argpartition(-sc, a.top_k - 1, axis=-1)[:, :a.top_k]
        ha = 0
        for j in range(len(iho)):
            ha += len(set(hp[j]) & set(ids[iho[j]])) / a.top_k
        held[name] = round(ha / max(len(iho), 1), 4)

    out = {
        "layer": a.layer,
        "n_samples": n_steps,
        "top_k": a.top_k,
        "controller_params": int(params.size),
        "factored_params": int(n_tr * (X.shape[1] + Y.shape[1])),
        "router_params": int(W.size) + int(np.asarray(bias).size),
        "topk_agreement_insample": round(agreement, 4),
        "factored_holdout_agreement": held,
    }
    out_path = ROOT / "outputs" / "router_distill.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
