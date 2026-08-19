"""Final-head oracle (Milestone H): ParallelHead.hc_head + output_norm +
logits in numpy on REAL V4 bytes (chunked Q6_K dequant for output.weight).

Writes: outputs/v4head_hc.bin (input), outputs/v4head_logits.bin
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ultratensor.dequant import dequantize_rows  # noqa: E402
from ultratensor.gguf_factored import _align, read_gguf_header  # noqa: E402

SHARD0 = ("D:/hyperv4/models/pro/"
          "deepseek-ai-DeepSeek-V4-Pro-Q3_K_M-00001-of-00017.gguf")
DIM, HC = 7168, 4
EPS = 1e-6
_Q = {11: "Q3_K", 14: "Q6_K"}


def tinfo(path, name):
    v, kvs, infos, hdr = read_gguf_header(path)
    ds = _align(hdr, 32)
    for nm, dims, tt, off in infos:
        if nm.decode() == name:
            return dims, tt, ds + off
    raise KeyError(name)


def load_tensor(path, name):
    dims, tt, off = tinfo(path, name)
    if tt == 0:
        with open(path, "rb") as f:
            f.seek(off)
            return np.frombuffer(f.read(int(np.prod(dims)) * 4),
                                 np.float32).copy()
    qname = _Q[tt]
    n, m = int(dims[0]), int(dims[1])
    if tt == 11:
        rb = (n // 256) * 110
    elif tt == 14:
        rb = (n // 256) * 210
    else:
        raise ValueError(tt)
    out = np.empty((m, n), np.float32)
    with open(path, "rb") as f:
        for r0 in range(0, m, 8192):
            r1 = min(r0 + 8192, m)
            f.seek(off + r0 * rb)
            raw = np.frombuffer(f.read((r1 - r0) * rb), np.uint8)
            out[r0:r1] = dequantize_rows(qname, raw, (n, r1 - r0), 0,
                                         r1 - r0)
    return out


def main() -> int:
    fn = load_tensor(SHARD0, "output_hc_fn.weight")      # [4, 28672]
    scale = load_tensor(SHARD0, "output_hc_scale.weight")  # [1]
    base = load_tensor(SHARD0, "output_hc_base.weight")    # [4]
    nw = load_tensor(SHARD0, "output_norm.weight")         # [7168]

    rng = np.random.default_rng(42)
    hc = rng.standard_normal((HC, DIM)).astype(np.float32)
    (ROOT / "outputs" / "v4head_hc.bin").write_bytes(hc.tobytes())

    x = hc.reshape(-1).astype(np.float64)
    rsqrt = 1.0 / np.sqrt((x ** 2).mean() + EPS)
    mixes = fn.astype(np.float64) @ x * rsqrt
    pre = 1.0 / (1.0 + np.exp(-(mixes * scale + base))) + EPS
    h = np.sum(pre[:, None] * hc, axis=0).astype(np.float64)
    h = h * nw / np.sqrt((h ** 2).mean() + EPS)
    (ROOT / "outputs" / "v4head_h.bin").write_bytes(
        h.astype(np.float32).tobytes())

    dims, tt, off = tinfo(SHARD0, "output.weight")
    n, vocab = int(dims[0]), int(dims[1])
    assert tt == 14
    rb = (n // 256) * 210
    logits = np.empty(vocab, np.float32)
    B = 16384
    with open(SHARD0, "rb") as f:
        for r0 in range(0, vocab, B):
            r1 = min(r0 + B, vocab)
            f.seek(off + r0 * rb)
            raw = np.frombuffer(f.read((r1 - r0) * rb), np.uint8)
            W = dequantize_rows("Q6_K", raw, (n, r1 - r0), 0, r1 - r0)
            logits[r0:r1] = W @ h.astype(np.float32)
    (ROOT / "outputs" / "v4head_logits.bin").write_bytes(logits.tobytes())
    print("head h sum %.4f std %.4f | logits sum %.4f"
          % (h.sum(), h.std(), logits.sum()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
