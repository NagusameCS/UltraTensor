"""V4-Coder step 9 — build the keep64 coder GGUF from the real shards.

Applies the census verdict to the real bytes: every dense layer
(3-60) keeps the L3 keep64 expert set (top-64 by code routing mass,
which INCLUDES the tail-gate vetoed experts [88, 244, 79]); hash
layers (0-2) keep all 384 experts (their routing is token-determined,
handled by the 9.3 MB table). All other tensors are copied verbatim.

Output: a valid GGUF with 64-expert dense stacks (~161 GB Q3_K for
the expert payload + unchanged attention/router tensors). The IQ2_XS
requant to ~74.5 GB is the NEXT step (needs requant kernels).

ASSUMPTION (flagged in the manifest): L4-L60 are census-like to L3;
exp_mid (layers 3-10 trace) verifies this. Regenerate if it differs.

Usage:
    python scripts/v4_coder_keep.py \
        --census outputs/code_census.json \
        --out Y:/models/coder/DeepSeek-V4-Coder-keep64.gguf
"""

import argparse
import glob
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultratensor.gguf_keep import write_keep_gguf  # noqa: E402

N_LAYERS = 61
HASH_LAYERS = 3
KEEP = 64
EXPERTS = 384


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--census", default=str(ROOT / "outputs" /
                                            "code_census.json"))
    ap.add_argument("--out", default="Y:/models/coder/"
                                    "DeepSeek-V4-Coder-keep64.gguf")
    ap.add_argument("--shards", default="D:/hyperv4/models/pro/"
                                        "deepseek-ai-DeepSeek-V4-Pro-"
                                        "Q3_K_M-*.gguf")
    a = ap.parse_args()

    census = json.load(open(a.census, encoding="utf-8"))
    top64 = census["layers"]["L3"]["top64_ids"]
    assert len(top64) >= KEEP

    keep = {}
    for L in range(N_LAYERS):
        if L < HASH_LAYERS:
            idx = list(range(EXPERTS))       # table routes, keep all
        else:
            idx = list(top64[:KEEP])         # code subnetwork
        for kind in ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"):
            keep[f"blk.{L}.{kind}.weight".encode()] = idx

    shards = sorted(glob.glob(a.shards))
    assert shards, "no shards found"
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"writing {out} from {len(shards)} shards "
          f"({sum(1 for _ in keep)//3} layers, keep64 dense / 384 hash)",
          flush=True)
    write_keep_gguf(shards, out, keep)
    print(f"done in {time.time() - t0:.0f}s -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
