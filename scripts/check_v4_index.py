"""Milestone G validation: geodessical learned indexer (layer 2, ratio 4)
vs the numpy oracle (scripts/v4_ref_index.py) on REAL V4 bytes."""
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
EXE = EXE / "build_host" / "test_v4_index.exe"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_glob")
    a = ap.parse_args()
    shards = sorted(glob.glob(
        a.model_glob.replace("*", "deepseek-ai-DeepSeek-V4-Pro-Q3_K_M-*")))
    shards = [s for s in shards if "dflash" not in s]
    st = ExpertStore(shards[0], extra_shards=shards[1:])
    kinds = ["indexer.attn_q_b", "indexer.proj",
             "indexer_compressor_kv", "indexer_compressor_gate",
             "indexer_compressor_ape", "indexer_compressor_norm"]
    sidx = set()
    for k in kinds:
        t = st.tensors.get(f"blk.2.{k}.weight")
        if t:
            sidx.add(t["shard"])
    shard_paths = [str(st.shards[i]) for i in sorted(sidx)]

    out = ROOT / "outputs"
    orc = out / "idx_score_pre.bin"
    if not orc.exists():
        rc0 = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "v4_ref_index.py")],
            capture_output=True, text=True)
        if rc0.returncode != 0:
            print("oracle failed:", rc0.stderr[-300:])
            return 1
    pre_b, dec_b = out / "idx_score_pre_c.bin", out / "idx_score_dec_c.bin"
    tp_b, td_b = out / "idx_topk_pre_c.bin", out / "idx_topk_dec_c.bin"
    rc = subprocess.run(
        [str(EXE), str(out / "idx_x_pre.bin"), str(out / "idx_x_dec.bin"),
         str(out / "idx_qr_pre.bin"), str(out / "idx_qr_dec.bin"),
         "8", "4", str(pre_b), str(tp_b), str(dec_b), str(td_b)]
        + shard_paths, capture_output=True, text=True)
    if rc.returncode != 0:
        print("exe failed:", rc.stderr[-400:])
        return 1
    ok = True
    for tag, cb, rb in (("prefill", pre_b, orc),
                        ("decode", dec_b, out / "idx_score_dec.bin")):
        c = np.fromfile(cb, np.float32)
        r = np.fromfile(rb, np.float32)
        fin = np.isfinite(c) & np.isfinite(r)
        both_masked = (c <= -1e30) & (r <= -1e30)
        assert (fin | both_masked).all(), tag
        d = np.abs(c[fin] - r[fin])
        denom = np.abs(r[fin]).max()
        rel = float(d.max() / denom) if d.size else 0.0
        print(f"indexer score {tag}: max_abs {d.max():.6f} "
              f"max_rel {rel:.3e} {'PASS' if rel < 1e-3 else 'FAIL'}")
        ok &= rel < 1e-3
    for tag, cb, rb in (("prefill", tp_b, out / "idx_topk_pre.bin"),
                        ("decode", td_b, out / "idx_topk_dec.bin")):
        c = np.fromfile(cb, np.int64)
        r = np.fromfile(rb, np.int64)
        same = (c == r).all()
        print(f"indexer topk {tag}: {'MATCH' if same else 'MISMATCH'}")
        ok &= same
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
