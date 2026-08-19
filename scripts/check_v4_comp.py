"""Milestone F validation: geodessical learned KV compressor (layer 2,
ratio 4, overlap) vs the numpy oracle (scripts/v4_ref_compress.py) on
REAL V4 bytes."""
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
EXE = EXE / "build_host" / "test_v4_comp.exe"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_glob")
    a = ap.parse_args()
    shards = sorted(glob.glob(
        a.model_glob.replace("*", "deepseek-ai-DeepSeek-V4-Pro-Q3_K_M-*")))
    shards = [s for s in shards if "dflash" not in s]
    st = ExpertStore(shards[0], extra_shards=shards[1:])
    kinds = ["attn_compressor_kv", "attn_compressor_gate",
             "attn_compressor_ape", "attn_compressor_norm"]
    sidx = set()
    for k in kinds:
        t = st.tensors.get(f"blk.2.{k}.weight")
        if t:
            sidx.add(t["shard"])
    shard_paths = [str(st.shards[i]) for i in sorted(sidx)]

    out = ROOT / "outputs"
    pre_rb = out / "comp_cache_pre.bin"
    if not pre_rb.exists():
        orc = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "v4_ref_compress.py")],
            capture_output=True, text=True)
        if orc.returncode != 0:
            print("oracle failed:", orc.stderr[-300:])
            return 1
    pre_b, dec_b = out / "comp_cache_pre_c.bin", out / "comp_cache_dec_c.bin"
    rc = subprocess.run(
        [str(EXE), str(out / "comp_x_pre.bin"), str(out / "comp_x_dec.bin"),
         "12", "8", str(pre_b), str(dec_b)] + shard_paths,
        capture_output=True, text=True)
    if rc.returncode != 0:
        print("exe failed:", rc.stderr[-400:])
        return 1
    pre_c = np.fromfile(pre_b, np.float32)
    dec_c = np.fromfile(dec_b, np.float32)
    pre_r = np.fromfile(out / "comp_cache_pre.bin", np.float32)
    dec_r = np.fromfile(out / "comp_cache_dec.bin", np.float32)
    for tag, c, r in (("prefill", pre_c, pre_r), ("decode", dec_c, dec_r)):
        d = np.abs(c - r)
        rel = float(d.max() / np.abs(r).max())
        print(f"compressor {tag}: rows {r.shape[0]} max_abs {d.max():.6f} "
              f"max_rel {rel:.3e} {'PASS' if rel < 1e-3 else 'FAIL'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
