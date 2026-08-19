"""Milestone D validation: geodessical embedding/norm/logits vs numpy
dequantization on REAL V4 bytes.

Usage:
    python scripts/check_v4_io.py <shard0.gguf> <mode> [token]
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from ultratensor.dequant import dequantize_rows  # noqa: E402
from ultratensor.gguf_factored import _align, read_gguf_header  # noqa: E402

DIM = 7168
EXE = Path(r"C:\Users\legom\OneDrive\Documents\GitHub\HyperTensor")
EXE = EXE / "build_host" / "test_v4_io.exe"
TOKEN = 4242


def tensor_info(path, name):
    v, kvs, infos, hdr_end = read_gguf_header(path)
    ds = _align(hdr_end, 32)
    for nm, dims, tt, off in infos:
        if nm.decode() == name:
            return dims, tt, ds + off
    raise KeyError(name)


def read_row(path, name, qname, dims, tt, off, row, n, row_bytes):
    with open(path, "rb") as f:
        f.seek(off + row * row_bytes)
        raw = np.frombuffer(f.read(row_bytes), np.uint8)
    return dequantize_rows(qname, raw, (n, 1), 0, 1)[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("shard0")
    ap.add_argument("mode", choices=["embd", "logits", "norm"])
    a = ap.parse_args()

    out = ROOT / "outputs" / f"v4io_{a.mode}.bin"
    if a.mode == "embd":
        dims, tt, off = tensor_info(a.shard0, "token_embd.weight")
        n, vocab = int(dims[0]), int(dims[1])
        assert tt == 11   # Q3_K
        row_bytes = (n // 256) * 110
        rc = subprocess.run([str(EXE), a.shard0, "embd", str(TOKEN),
                             str(out)], capture_output=True, text=True)
        assert rc.returncode == 0, rc.stderr
        y_c = np.fromfile(out, np.float32)
        y_ref = read_row(a.shard0, "token_embd.weight", "Q3_K", dims, tt,
                         off, TOKEN, n, row_bytes)
        d = np.abs(y_c - y_ref)
        print(f"embd token {TOKEN}: max_abs {d.max():.6f} "
              f"max_rel {d.max() / np.abs(y_ref).max():.3e}")
        return 0 if d.max() / np.abs(y_ref).max() < 1e-4 else 1
    if a.mode == "logits":
        dims, tt, off = tensor_info(a.shard0, "output.weight")
        n, vocab = int(dims[0]), int(dims[1])
        assert tt == 14   # Q6_K
        rng = np.random.default_rng(1)
        h = rng.standard_normal(n).astype(np.float32)
        (ROOT / "outputs" / "v4io_h.bin").write_bytes(h.tobytes())
        rc = subprocess.run([str(EXE), a.shard0, "logits",
                             str(ROOT / "outputs" / "v4io_h.bin"),
                             str(out)], capture_output=True, text=True)
        assert rc.returncode == 0, rc.stderr
        y_c = np.fromfile(out, np.float32)
        row_bytes = (n // 256) * 210
        # chunked numpy reference (bounded memory)
        y_ref = np.empty(vocab, np.float32)
        B = 16384
        with open(a.shard0, "rb") as f:
            for r0 in range(0, vocab, B):
                r1 = min(r0 + B, vocab)
                f.seek(off + r0 * row_bytes)
                raw = np.frombuffer(f.read((r1 - r0) * row_bytes), np.uint8)
                W = dequantize_rows("Q6_K", raw, (n, r1 - r0), 0, r1 - r0)
                y_ref[r0:r1] = W @ h
        d = np.abs(y_c - y_ref)
        rel = float(d.max() / np.abs(y_ref).max())
        top_c = set(np.argsort(-y_c)[:10].tolist())
        top_r = set(np.argsort(-y_ref)[:10].tolist())
        print(f"logits: max_abs {d.max():.6f} max_rel {rel:.3e} "
              f"top10 overlap {len(top_c & top_r)}/10")
        return 0 if rel < 1e-4 else 1
    # norm
    dims, tt, off = tensor_info(a.shard0, "output_norm.weight")
    n = int(dims[0])
    assert tt == 0
    with open(a.shard0, "rb") as f:
        f.seek(off)
        w = np.frombuffer(f.read(n * 4), np.float32)
    rng = np.random.default_rng(2)
    x = rng.standard_normal(n).astype(np.float32)
    xb = ROOT / "outputs" / "v4io_x.bin"
    xb.write_bytes(x.tobytes())
    rc = subprocess.run([str(EXE), a.shard0, "norm", str(xb), str(out),
                         "1e-5"], capture_output=True, text=True)
    assert rc.returncode == 0, rc.stderr
    y_c = np.fromfile(out, np.float32)
    y_ref = x * w / np.sqrt((x ** 2).mean() + 1e-5)
    d = np.abs(y_c - y_ref)
    print(f"norm: max_abs {d.max():.6f} "
          f"max_rel {d.max() / np.abs(y_ref).max():.3e}")
    return 0 if d.max() / np.abs(y_ref).max() < 1e-5 else 1


if __name__ == "__main__":
    raise SystemExit(main())
