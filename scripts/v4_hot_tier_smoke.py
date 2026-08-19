"""V4-Coder step 8 — hot-tier loader smoke test.

The serve loop keeps the top-N code experts RESIDENT. This smoke-tests
the tier loader shape against the real bytes: given the keep64 list
from the census, load the top-N gate/up/down stacks of layer 3 through
the existing ExpertStore, report bytes, load time, and verify a
per-token forward on the loaded tensors.

Usage:
    python scripts/v4_hot_tier_smoke.py --n 4
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


def silu(g):
    return g / (1.0 + np.exp(-g))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--census", default=str(ROOT / "outputs" /
                                            "code_census.json"))
    a = ap.parse_args()

    census = json.load(open(a.census, encoding="utf-8"))
    keep = census["layers"]["L3"]["top64_ids"][: a.n]

    shards = sorted(glob.glob("D:/hyperv4/models/pro/"
                              "deepseek-ai-DeepSeek-V4-Pro-Q3_K_M-*.gguf"))
    st = vs.ExpertStore(shards[0], extra_shards=shards[1:])

    t0 = time.time()
    resident, bytes_total = {}, 0
    for e in keep:
        stack = {}
        for kind in ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"):
            W = st.read_expert(3, kind, e)
            stack[kind] = W
            bytes_total += int(W.nbytes)
        resident[e] = stack
    load_s = time.time() - t0

    # per-token forward smoke on one real code hidden state
    X = np.asarray(np.load(ROOT / "outputs" /
                           "exp_code_ffn_inputs_dense.npz")["L3"][0],
                   dtype=np.float64)
    t0 = time.time()
    y = np.zeros(7168)
    for e in keep:
        g = X @ resident[e]["ffn_gate_exps"].T
        y += (silu(g) * (X @ resident[e]["ffn_up_exps"].T)) @ \
            resident[e]["ffn_down_exps"].T
    fwd_ms = (time.time() - t0) * 1000

    report = {
        "n_resident": a.n,
        "expert_ids": keep,
        "load_s": round(load_s, 1),
        "resident_mb": round(bytes_total / 1e6, 1),
        "forward_ms_per_token": round(fwd_ms, 1),
        "output_norm": round(float(np.linalg.norm(y)), 1),
    }
    out = ROOT / "outputs" / "hot_tier_smoke.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
