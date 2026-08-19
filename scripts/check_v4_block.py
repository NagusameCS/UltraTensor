"""Milestone E validation: geodessical full block forward vs the numpy
reference (scripts/v4_ref_block.py) on REAL V4 bytes, layer 0."""
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
EXE = EXE / "build_host" / "test_v4_block.exe"
TOKEN = 4242
DIM = 7168


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_glob")
    ap.add_argument("--pos", type=int, default=0)
    a = ap.parse_args()
    shards = sorted(glob.glob(
        a.model_glob.replace("*", "deepseek-ai-DeepSeek-V4-Pro-Q3_K_M-*")))
    shards = [s for s in shards if "dflash" not in s]
    st = ExpertStore(shards[0], extra_shards=shards[1:])
    # layer 0 tensor shards
    kinds = ["attn_norm", "attn_q_a", "attn_q_a_norm", "attn_q_b",
             "attn_kv", "attn_kv_a_norm", "attn_output_a", "attn_output_b",
             "attn_sinks", "hc_attn_fn", "hc_attn_scale", "hc_attn_base",
             "hc_ffn_fn", "hc_ffn_scale", "hc_ffn_base", "ffn_norm",
             "ffn_gate_exps", "ffn_up_exps", "ffn_down_exps",
             "ffn_gate_inp", "ffn_gate_tid2eid", "ffn_gate_shexp",
             "ffn_up_shexp", "ffn_down_shexp"]
    sidx = set()
    for k in kinds:
        t = st.tensors.get(f"blk.0.{k}.weight")
        if t:
            sidx.add(t["shard"])
    shard_paths = [str(st.shards[i]) for i in sorted(sidx)]

    rng = np.random.default_rng(0)
    x = rng.standard_normal(DIM).astype(np.float32)
    x = np.tile(x, (4, 1)).astype(np.float32)     # HC state [4, DIM]
    xb = ROOT / "outputs" / "v4_block_x.bin"
    yb = ROOT / "outputs" / "v4_block_y.bin"
    xb.write_bytes(x.tobytes())
    rc = subprocess.run([str(EXE), str(xb), str(TOKEN), str(a.pos), str(yb)]
                        + shard_paths, capture_output=True, text=True)
    if rc.returncode != 0:
        print("exe failed:", rc.stderr[-400:])
        return 1
    y_c = np.fromfile(yb, np.float32)
    y_ref = np.fromfile(ROOT / "outputs" / f"v4_block_ref_p{a.pos}.bin",
                        np.float32)
    d = np.abs(y_c - y_ref)
    rel = float(d.max() / np.abs(y_ref).max())
    print(f"block layer 0 pos {a.pos}: max_abs {d.max():.6f} "
          f"max_rel {rel:.3e} {'PASS' if rel < 1e-3 else 'FAIL'}")
    return 0 if rel < 1e-3 else 1


if __name__ == "__main__":
    raise SystemExit(main())
