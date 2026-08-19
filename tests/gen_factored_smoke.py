"""Generate the factored-GGUF smoke fixtures for the geodessical C test.

Produces into this directory:
  src.gguf       minimal single-file GGUF (t1 = Q8_0 128x256, t2 = F16 64x32)
  out.gguf       factored container (t1 -> t1.factored_U + t1.factored_C, rank 16)
  x.bin          input vector  (n float32 LE)
  expected.bin   W_hat @ x     (m float32 LE)  — the numpy oracle
"""
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ultratensor.gguf_factored import (  # noqa: E402
    GGML_TYPE_F16,
    GGML_TYPE_Q8_0,
    read_factored_gguf,
    reconstruct,
    write_factored_gguf,
)

HERE = Path(__file__).resolve().parent


def _kv(key: bytes, vtype: int, raw: bytes) -> bytes:
    return (struct.pack("<Q", len(key)) + key +
            struct.pack("<I", vtype) + raw)


def _str_kv(key: str, s: str) -> bytes:
    b = s.encode()
    return _kv(key.encode(), 8, struct.pack("<Q", len(b)) + b)


def _write_source(path: Path, W: np.ndarray, F: np.ndarray) -> None:
    """Minimal single-file GGUF: t1 = Q8_0, t2 = F16."""
    t1_rows, t1_cols = W.shape
    t1_bytes = bytearray()
    for r in range(t1_rows):
        for b0 in range(0, t1_cols, 32):
            block = W[r, b0:b0 + 32]
            d = np.float16(np.max(np.abs(block)) / 127.0)
            q = np.round(block.astype(np.float64) / float(d)).clip(-128, 127)
            t1_bytes += struct.pack("<e", d) + q.astype(np.int8).tobytes()
    t1_size = len(t1_bytes)
    t2 = F.astype(np.float16).tobytes()

    kvs = (_str_kv("general.architecture", "llama")
           + _str_kv("general.name", "test-src")
           + _kv(b"general.alignment", 4, struct.pack("<I", 32)))

    def info(name: bytes, dims, ttype: int, off: int) -> bytes:
        return (struct.pack("<Q", len(name)) + name +
                struct.pack("<I", len(dims)) +
                struct.pack("<" + "Q" * len(dims), *dims) +
                struct.pack("<I", ttype) + struct.pack("<Q", off))

    header = (b"GGUF" + struct.pack("<I", 3) +
              struct.pack("<Q", 2) + struct.pack("<Q", 3) + kvs)
    header += info(b"t1", (W.shape[1], W.shape[0]), GGML_TYPE_Q8_0, 0)
    header += info(b"t2", (F.shape[1], F.shape[0]), GGML_TYPE_F16, t1_size)
    header += b"\0" * ((32 - len(header) % 32) % 32)
    with open(path, "wb") as f:
        f.write(header + bytes(t1_bytes) + t2)


def main() -> None:
    rng = np.random.default_rng(2)
    A = rng.standard_normal((128, 16)).astype(np.float32)
    B = rng.standard_normal((16, 256)).astype(np.float32)
    W = (A @ B + 0.01 * rng.standard_normal((128, 256))).astype(np.float32)
    F = rng.standard_normal((64, 32)).astype(np.float32)

    src = HERE / "src.gguf"
    out = HERE / "out.gguf"
    _write_source(src, W, F)
    write_factored_gguf(src, out, patterns=["t1"], rank=16)

    manifest, tensors = read_factored_gguf(out)
    assert manifest["tensors"][0]["name"] == "t1"
    W_hat = reconstruct(manifest, tensors, "t1.factored_C")

    x = np.linspace(-0.5, 0.5, W_hat.shape[1], dtype=np.float32)
    expected = (W_hat.astype(np.float32) @ x).astype(np.float32)
    (HERE / "x.bin").write_bytes(x.tobytes())
    (HERE / "expected.bin").write_bytes(expected.tobytes())
    print(f"ok: out.gguf tensors={sorted(tensors)}")
    print(f"    t1 factored: rank={manifest['tensors'][0]['rank']} "
          f"shape={W_hat.shape}")
    print(f"    x.bin {x.nbytes} B, expected.bin {expected.nbytes} B")


if __name__ == "__main__":
    main()
