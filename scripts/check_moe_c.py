"""Milestone B validation: geodessical C hash-layer forward vs the
reference-trusted Python lazy executor, on REAL V4 bytes.

Usage:
    python scripts/check_moe_c.py <model_glob> <shard_gguf> <layer>
"""
import argparse
import glob
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from ultratensor.expert_store import ExpertStore  # noqa: E402
from ultratensor.moe_exec import MoELayer  # noqa: E402

DIM = 7168
EXE = Path(r"C:\Users\legom\OneDrive\Documents\GitHub\HyperTensor")
EXE = EXE / "build_host" / "test_moe_v4.exe"
TOKEN = 4242


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_glob")
    ap.add_argument("layer", type=int)
    a = ap.parse_args()

    shards = sorted(glob.glob(
        a.model_glob.replace("*", "deepseek-ai-DeepSeek-V4-Pro-Q3_K_M-*")))
    shards = [s for s in shards if "dflash" not in s]
    st = ExpertStore(shards[0], extra_shards=shards[1:])
    name = f"blk.{a.layer}.ffn_gate_exps.weight"
    if name not in st.tensors:
        print(f"layer {a.layer} not in inventory")
        return 2
    kinds = ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps",
             "ffn_gate_inp", "ffn_gate_shexp", "ffn_up_shexp",
             "ffn_down_shexp", "ffn_gate_tid2eid"]
    sidx = set()
    for k in kinds:
        t = st.tensors.get(f"blk.{a.layer}.{k}.weight")
        if t:
            sidx.add(t["shard"])
    t = st.tensors.get(f"blk.{a.layer}.exp_probs_b.bias")
    if t:
        sidx.add(t["shard"])
    shard_paths = [str(st.shards[i]) for i in sorted(sidx)]
    ml = MoELayer(st, a.layer)

    rng = np.random.default_rng(0)
    x = rng.standard_normal(DIM).astype(np.float32)
    y_ref = ml(x[None, :], [TOKEN])[0]      # [DIM]

    xb = ROOT / "outputs" / f"moe_c_x_{a.layer}.bin"
    yb = ROOT / "outputs" / f"moe_c_y_{a.layer}.bin"
    xb.write_bytes(x.tobytes())
    rc = subprocess.run([str(EXE), str(xb), str(a.layer), str(TOKEN),
                         str(yb)] + shard_paths,
                        capture_output=True, text=True)
    if rc.returncode != 0:
        print("exe failed:", rc.stderr[-500:])
        return 1
    y_c = np.frombuffer(yb.read_bytes(), np.float32)
    if y_c.shape[0] != DIM:
        print(f"bad y.bin size {y_c.shape}")
        return 1
    d = np.abs(y_ref - y_c)
    rel = float(d.max() / np.abs(y_ref).max())
    ok = rel < 2e-3
    print(f"layer {a.layer}: C vs python lazy  max_abs {d.max():.6f}  "
          f"max_rel {rel:.3e}  {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
