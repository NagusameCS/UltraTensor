"""Milestone J validation: geodessical multi-token block forward (cached
decode) vs the numpy oracle (scripts/v4_ref_seq.py) on REAL V4 bytes."""
import argparse
import glob
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from ultratensor.expert_store import ExpertStore  # noqa: E402

EXE = Path(r"C:\Users\legom\OneDrive\Documents\GitHub\HyperTensor")
EXE = EXE / "build_host" / "test_v4_seq.exe"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_glob")
    ap.add_argument("--layer", type=int, default=1)
    ap.add_argument("--n", type=int, default=12)
    a = ap.parse_args()
    shards = sorted(glob.glob(
        a.model_glob.replace("*", "deepseek-ai-DeepSeek-V4-Pro-Q3_K_M-*")))
    shards = [s for s in shards if "dflash" not in s]
    st = ExpertStore(shards[0], extra_shards=shards[1:])
    kinds = ["attn_norm", "attn_q_a", "attn_q_a_norm", "attn_q_b",
             "attn_kv", "attn_kv_a_norm", "attn_output_a", "attn_output_b",
             "attn_sinks", "hc_attn_fn", "hc_attn_scale", "hc_attn_base",
             "hc_ffn_fn", "hc_ffn_scale", "hc_ffn_base", "ffn_norm",
             "ffn_gate_exps", "ffn_up_exps", "ffn_down_exps",
             "ffn_gate_inp", "ffn_gate_tid2eid", "ffn_gate_shexp",
             "ffn_up_shexp", "ffn_down_shexp",
             "attn_compressor_kv", "attn_compressor_gate",
             "attn_compressor_ape", "attn_compressor_norm"]
    sidx = set()
    for k in kinds:
        t = st.tensors.get(f"blk.{a.layer}.{k}.weight")
        if t:
            sidx.add(t["shard"])
    shard_paths = [str(st.shards[i]) for i in sorted(sidx)]

    out = ROOT / "outputs"
    rb = out / f"v4seq_y_L{a.layer}.bin"
    if not rb.exists():
        rc0 = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "v4_ref_seq.py"),
             str(a.layer), str(a.n)],
            capture_output=True, text=True)
        if rc0.returncode != 0:
            print("oracle failed:", rc0.stderr[-300:])
            return 1
    c_b = out / f"v4seq_y_L{a.layer}_c.bin"
    rc = subprocess.run(
        [str(EXE), str(a.layer), str(a.n), str(out / f"v4seq_x_L{a.layer}"
                                                ".bin"),
         str(c_b)] + shard_paths, capture_output=True, text=True)
    if rc.returncode != 0:
        print("exe failed:", rc.stderr[-400:])
        return 1
    c = np.fromfile(c_b, np.float32).reshape(a.n, -1)
    r = np.fromfile(rb, np.float32).reshape(a.n, -1)
    worst = 0.0
    for p in range(a.n):
        d = np.abs(c[p] - r[p])
        rel = float(d.max() / np.abs(r[p]).max())
        worst = max(worst, rel)
        print(f"p={p}: max_abs {d.max():.6f} max_rel {rel:.3e}")
    print(f"sequence L{a.layer}: worst_rel {worst:.3e} "
          f"{'PASS' if worst < 1e-3 else 'FAIL'}")
    return 0 if worst < 1e-3 else 1


if __name__ == "__main__":
    raise SystemExit(main())
