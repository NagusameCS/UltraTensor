"""Phase 1 container tests: factored GGUF round-trip."""
import struct
import tempfile
from pathlib import Path

import numpy as np
import pytest

from ultratensor.gguf_factored import (
    GGML_TYPE_F16,
    GGML_TYPE_Q8_0,
    decode_codes,
    encode_codes,
    factor_matrix,
    read_factored_gguf,
    read_gguf_header,
    reconstruct,
    write_factored_gguf,
)

F16 = GGML_TYPE_F16
Q8_0 = GGML_TYPE_Q8_0


def _kv(key: bytes, vtype: int, raw: bytes) -> bytes:
    return (struct.pack("<Q", len(key)) + key +
            struct.pack("<I", vtype) + raw)


def _str_kv(key: str, s: str) -> bytes:
    b = s.encode()
    return _kv(key.encode(), 8, struct.pack("<Q", len(b)) + b)


def _write_source(path: Path, W: np.ndarray, F: np.ndarray):
    """Minimal single-file GGUF: t1 = Q8_0, t2 = F16."""
    t1_rows, t1_cols = W.shape
    # Q8_0: per 32-col block: fp16 d + 32 int8
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
    header += info(b"t1", (W.shape[1], W.shape[0]), Q8_0, 0)
    header += info(b"t2", (F.shape[1], F.shape[0]), F16, t1_size)
    header += b"\0" * ((32 - len(header) % 32) % 32)
    with open(path, "wb") as f:
        f.write(header + bytes(t1_bytes) + t2)


def test_codes_roundtrip():
    rng = np.random.default_rng(0)
    C = rng.standard_normal((16, 256)).astype(np.float32)
    scales, packed = encode_codes(C)
    assert packed.shape == (16, 128)
    assert scales.shape == (16, 8)
    err = np.abs(decode_codes(scales, packed) - C).max()
    assert err < np.abs(C).max() / 8 + 1e-6


def test_factor_energy():
    rng = np.random.default_rng(1)
    W = rng.standard_normal((128, 256)).astype(np.float32)
    U, C = factor_matrix(W, rank=None, energy=0.99)
    assert U.shape[0] == 128 and U.shape[1] == C.shape[0]
    err = np.linalg.norm(U.astype(np.float32) @ C - W)
    rel = err / np.linalg.norm(W)
    assert rel < 0.15  # 99% energy + fp16 basis


def test_factored_gguf_roundtrip():
    rng = np.random.default_rng(2)
    # low-rank + noise: realistic weight matrix with decaying spectrum
    A = rng.standard_normal((128, 16)).astype(np.float32)
    B = rng.standard_normal((16, 256)).astype(np.float32)
    W = (A @ B + 0.01 * rng.standard_normal((128, 256))).astype(np.float32)
    F = rng.standard_normal((64, 32)).astype(np.float32)
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "src.gguf"
        out = Path(td) / "out.gguf"
        _write_source(src, W, F)

        write_factored_gguf(src, out, patterns=["t1"], rank=16)

        manifest, tensors = read_factored_gguf(out)
        assert manifest["version"] == 1
        assert len(manifest["tensors"]) == 1
        assert manifest["tensors"][0]["name"] == "t1"

        # dense tensor copied verbatim
        t2 = tensors["t2"]
        assert t2["kind"] == "dense"
        assert t2["data"] == F.astype(np.float16).tobytes()

        W_hat = reconstruct(manifest, tensors, "t1.factored_C")
        assert W_hat.shape == W.shape
        rel = np.linalg.norm(W_hat - W) / np.linalg.norm(W)
        # rank-16 SVD captures the signal; residual is the uq4 codes noise
        assert rel < 0.15


def test_container_header_is_valid_gguf():
    rng = np.random.default_rng(3)
    W = rng.standard_normal((32, 64)).astype(np.float32)
    F = np.zeros((8, 8), np.float32)
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "src.gguf"
        out = Path(td) / "out.gguf"
        _write_source(src, W, F)
        write_factored_gguf(src, out, patterns=["t1"], rank=8)
        version, kvs, infos, hdr_end = read_gguf_header(out)
        assert version == 3
        assert len(infos) == 3  # t1.factored_U, t1.factored_C, t2
        keys = [k for k, _, _ in kvs]
        assert b"ultratensor.factored_manifest" in keys


def _write_source3(path: Path, W3: np.ndarray):
    """Single 3-D Q8_0 tensor t1 with gguf dims (n, m, E)."""
    E, m, n = W3.shape
    blob = bytearray()
    for e in range(E):
        for r in range(m):
            for b0 in range(0, n, 32):
                block = W3[e, r, b0:b0 + 32]
                d = np.float16(np.max(np.abs(block)) / 127.0)
                q = np.round(block.astype(np.float64) / float(d)).clip(-128, 127)
                blob += struct.pack("<e", d) + q.astype(np.int8).tobytes()
    kvs = (_str_kv("general.architecture", "llama")
           + _str_kv("general.name", "test-src3")
           + _kv(b"general.alignment", 4, struct.pack("<I", 32)))
    info = (struct.pack("<Q", 2) + b"t1" +
            struct.pack("<I", 3) + struct.pack("<QQQ", n, m, E) +
            struct.pack("<I", Q8_0) + struct.pack("<Q", 0))
    header = (b"GGUF" + struct.pack("<I", 3) +
              struct.pack("<Q", 1) + struct.pack("<Q", 3) + kvs + info)
    header += b"\0" * ((32 - len(header) % 32) % 32)
    with open(path, "wb") as f:
        f.write(header + bytes(blob))


def test_factored_gguf_3d_expert_roundtrip():
    rng = np.random.default_rng(4)
    E, m, n = 4, 64, 256
    # low-rank + noise experts (realistic decaying spectrum)
    W3 = np.stack([
        rng.standard_normal((m, 8)).astype(np.float32)
        @ rng.standard_normal((8, n)).astype(np.float32)
        + 0.01 * rng.standard_normal((m, n)).astype(np.float32)
        for _ in range(E)])
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "src.gguf"
        out = Path(td) / "out.gguf"
        _write_source3(src, W3)
        write_factored_gguf(src, out, patterns=["t1"], rank=8)
        manifest, tensors = read_factored_gguf(out)
        assert manifest["tensors"][0]["shape"] == [E, m, n]
        W_hat = reconstruct(manifest, tensors, "t1.factored_C")
        assert W_hat.shape == (E, m, n)
        rel = np.linalg.norm(W_hat - W3) / np.linalg.norm(W3)
        assert rel < 0.15
