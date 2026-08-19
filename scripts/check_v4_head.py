"""Milestone H validation: geodessical final head (hc_head + output_norm
+ logits) vs the numpy oracle (scripts/v4_ref_head.py) on REAL V4 bytes."""
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
EXE = EXE / "build_host" / "test_v4_head.exe"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_glob")
    a = ap.parse_args()
    shards = sorted(glob.glob(
        a.model_glob.replace("*", "deepseek-ai-DeepSeek-V4-Pro-Q3_K_M-*")))
    shards = [s for s in shards if "dflash" not in s]
    st = ExpertStore(shards[0], extra_shards=shards[1:])
    # final-head tensors live in shard 00001 (shard index 0)
    shard_paths = [str(st.shards[0])]

    out = ROOT / "outputs"
    if not (out / "v4head_logits.bin").exists():
        rc0 = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "v4_ref_head.py")],
            capture_output=True, text=True)
        if rc0.returncode != 0:
            print("oracle failed:", rc0.stderr[-300:])
            return 1
    c_b = out / "v4head_logits_c.bin"
    rc = subprocess.run(
        [str(EXE), str(out / "v4head_hc.bin"), str(c_b)] + shard_paths,
        capture_output=True, text=True)
    if rc.returncode != 0:
        print("exe failed:", rc.stderr[-400:])
        return 1
    c = np.fromfile(c_b, np.float32)
    r = np.fromfile(out / "v4head_logits.bin", np.float32)
    d = np.abs(c - r)
    rel = float(d.max() / np.abs(r).max())
    top_c = set(np.argsort(-c)[:10].tolist())
    top_r = set(np.argsort(-r)[:10].tolist())
    print(f"head logits: max_abs {d.max():.6f} max_rel {rel:.3e} "
          f"top10 overlap {len(top_c & top_r)}/10 "
          f"{'PASS' if rel < 1e-4 else 'FAIL'}")
    return 0 if rel < 1e-4 else 1


if __name__ == "__main__":
    raise SystemExit(main())
