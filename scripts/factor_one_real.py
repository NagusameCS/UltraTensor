"""Phase-6 first milestone: factor a real attention projection from the
V4-Pro GGUF into the factored container (minimal GGUF, drop_unmatched),
then report size reduction and reconstruction error.

Usage: factor_one_real.py <tensor_name>
"""
import glob
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultratensor.expert_store import ExpertStore
from ultratensor.gguf_factored import (write_factored_gguf,
                                       read_factored_gguf, reconstruct)


def main() -> int:
    name = sys.argv[1]
    out_dir = ROOT / "outputs" / "factored"
    out_dir.mkdir(parents=True, exist_ok=True)
    shards = sorted(glob.glob("D:/hyperv4/models/pro/"
                              "deepseek-ai-DeepSeek-V4-Pro-Q3_K_M-*.gguf"))
    st = ExpertStore(shards[0], extra_shards=shards[1:])
    t = st.tensors.get(name)
    if t is None:
        print(f"tensor {name} not found")
        return 1
    src = str(st.shards[t["shard"]])
    out = str(out_dir / (name.replace(".", "_") + ".factored.gguf"))
    print(f"factoring {name} (shard {t['shard']}, dims {tuple(t['dims'])}) "
          f"energy 0.99")

    n = int(t["dims"][0])
    from ultratensor.gguf_factored import (read_gguf_header, _align,
                                           _tensor_byte_size)
    v, kvs, infos, hdr_end = read_gguf_header(src)
    src_bytes = 0
    for nm, dims, ttype, off in infos:
        if nm.decode() == name:
            src_bytes = _tensor_byte_size(dims, ttype)
            break
    if src_bytes == 0:
        print("byte size unknown")
        return 1

    write_factored_gguf(src, out, patterns=[name], energy=0.99,
                        drop_unmatched=True)
    out_size = Path(out).stat().st_size
    print(f"source tensor bytes {src_bytes/1e6:.1f} MB -> factored gguf "
          f"{out_size/1e6:.1f} MB (ratio {out_size/src_bytes:.3f})")

    manifest, tensors = read_factored_gguf(out)
    entry = manifest["tensors"][0]
    rank = entry["rank"]
    print(f"rank {rank} (energy 0.99), uq4 block {entry['uq4_block']}")

    # reconstruction error vs the true dense tensor
    sys.path.insert(0, str(ROOT / "scripts"))
    import v4_ref_serve as vs
    W = vs.load_any(st, name)
    Wr = reconstruct(manifest, tensors, name + ".factored_C")
    rel = float(np.abs(Wr - W).max() / np.abs(W).max())
    frob = float(np.linalg.norm(Wr - W) / np.linalg.norm(W))
    print(f"reconstruction: max_rel {rel:.3e} frob_rel {frob:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
