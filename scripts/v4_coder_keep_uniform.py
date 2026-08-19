"""V4-Coder uniform keep16: the resident-coder extraction.

The inverse splice for a laptop-usable coder: every layer keeps K=16
experts (uniform count so llama.cpp's check_tensor_dims passes when
deepseek4.expert_count is overridden to K), the F32 dense router is
sliced to the same K columns (routing restricted to kept experts),
and the I32 hash tables of layers 0-2 are remapped so tokens that
hashed to dropped experts fall back to the most-used kept expert.

Kept sets: dense layers use keep64's manifest ranking (top-16 of the
code-census order); hash layers use the L0 census hash ranking.

Output ~42 GB at Q3_K (resident-ish; IQ2_XS requant ~26 GB later).

Usage:
    python scripts/v4_coder_keep_uniform.py \
        --keep64 Y:/models/coder/DeepSeek-V4-Coder-keep64-00001-of-00001.gguf \
        --census outputs/code_census.json \
        --out Y:/models/coder/DeepSeek-V4-Coder-keep16u.gguf
"""

import argparse
import glob
import json
import struct
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultratensor.gguf_factored import read_gguf_header  # noqa: E402
from ultratensor.gguf_keep import write_uniform_keep_gguf  # noqa: E402

N_LAYERS = 61
HASH_LAYERS = 3
K = 16


def load_keep64_manifest(path):
    v, kvs, infos, h = read_gguf_header(path)
    for k, t, r in kvs:
        if k == b"ultratensor.keep_manifest":
            return json.loads(r[8:].decode())
    raise KeyError("keep64 manifest KV not found")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep64",
                    default="Y:/models/coder/"
                            "DeepSeek-V4-Coder-keep64-00001-of-00001.gguf")
    ap.add_argument("--census", default=str(ROOT / "outputs" /
                                            "code_census.json"))
    ap.add_argument("--out", default="Y:/models/coder/"
                                    "DeepSeek-V4-Coder-keep16u.gguf")
    ap.add_argument("--keep", type=int, default=16)
    ap.add_argument("--ranking", default=None,
                    help="JSON {dense: [ids], hash: [ids]} per-domain "
                         "expert ranking; overrides keep64/census")
    ap.add_argument("--shards", default="D:/hyperv4/models/pro/"
                                        "deepseek-ai-DeepSeek-V4-Pro-"
                                        "Q3_K_M-*.gguf")
    a = ap.parse_args()

    K = a.keep
    if a.ranking:
        rank = json.load(open(a.ranking, encoding="utf-8"))
        hash16 = rank["hash"][:K]
        if "per_layer" in rank:                 # per-layer dense rankings
            dense_by_layer = {int(k): v for k, v in
                              rank["per_layer"].items()}
            fallback = rank.get("dense", [])[:K]
            dense16 = None                     # resolved per layer below
        else:
            dense_by_layer = None
            dense16 = rank["dense"][:K]
    else:
        manifest = load_keep64_manifest(a.keep64)
        census = json.load(open(a.census, encoding="utf-8"))
        hash16 = census["layers"]["L0"]["top64_ids"][:K]
        dense_by_layer = None
        dense16 = None
        for name, kept in manifest["kept"].items():
            if name.startswith("blk.3."):
                dense16 = kept[:K]
                break
        assert dense16 and len(dense16) == K

    keep, col_keep, remap = {}, {}, {}
    for L in range(N_LAYERS):
        if dense_by_layer is not None:
            idx = (dense_by_layer.get(L) or fallback)[:K]
        else:
            idx = dense16
        idx = hash16 if L < HASH_LAYERS else idx
        for kind in ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"):
            keep[f"blk.{L}.{kind}.weight".encode()] = idx
        col_keep[f"blk.{L}.ffn_gate_inp.weight".encode()] = idx
        col_keep[f"blk.{L}.exp_probs_b.bias".encode()] = idx
        if L < HASH_LAYERS:
            rank = {e: i for i, e in enumerate(idx)}
            remap[f"blk.{L}.ffn_gate_tid2eid.weight".encode()] = rank

    # KV overrides: uniform expert count + single-file split metadata
    v, kvs, infos, h = read_gguf_header(glob.glob(a.shards)[0])
    overrides = {}
    for k, t, r in kvs:
        if k == b"deepseek4.expert_count":
            overrides[k] = (struct.pack("<H", K) if len(r) == 2
                            else struct.pack("<I", K))
        elif k == b"split.count":
            overrides[k] = (struct.pack("<H", 1) if len(r) == 2
                            else struct.pack("<I", 1))

    shards = sorted(glob.glob(a.shards))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"uniform keep{K}: {len(keep)} stacks, "
          f"{len(col_keep)} router cols, {len(remap)} remaps, "
          f"overrides={[k.decode() for k in overrides]}", flush=True)
    write_uniform_keep_gguf(shards, out, keep, col_keep, remap, overrides)
    print(f"done in {time.time() - t0:.0f}s -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
