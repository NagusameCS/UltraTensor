"""Round-trip tests for GGUF expert-stack surgery (ultratensor.gguf_keep)."""

import struct

import numpy as np
import pytest

from ultratensor.gguf_factored import (
    _align,
    _kv,
    _tensor_byte_size,
    read_gguf_header,
)
from ultratensor.gguf_keep import write_keep_gguf

GGML_Q8_0 = 8
GGML_F32 = 0


def _q80_blocks(values):
    """Pack [blocks, 32] floats into Q8_0 blocks (scale fp16 + 32 int8)."""
    blocks, n = values.shape
    out = bytearray()
    for b in range(blocks):
        amax = float(np.abs(values[b]).max()) or 1.0
        d = amax / 127.0
        q = np.clip(np.round(values[b] / d), -127, 127).astype(np.int8)
        out += struct.pack("<e", d)
        out += q.tobytes()
    return bytes(out)


def _write_source(path, e0_vals, e1_vals, dense_vals):
    """Minimal GGUF with one Q8_0 expert stack (n, m, E=2) + F32 dense."""
    n, m, E = e0_vals.shape[0], e0_vals.shape[1], 2
    stack = np.stack([e0_vals, e1_vals]).reshape(E, -1)
    stack_bytes = b"".join(_q80_blocks(stack[e].reshape(-1, 32))
                           for e in range(E))
    dense_bytes = np.asarray(dense_vals, np.float32).tobytes()

    kvs = [(b"general.architecture", 8,
            struct.pack("<Q", 8) + b"test\x00\x00\x00\x00")]
    infos = [
        (b"blk.0.ffn_gate_exps.weight", (n, m, E), GGML_Q8_0, stack_bytes),
        (b"blk.0.dense.weight", (dense_vals.shape[0],
                                 dense_vals.shape[1]), GGML_F32,
         dense_bytes),
    ]
    alignment = 32
    hdr = (b"GGUF" + struct.pack("<I", 3) +
           struct.pack("<Q", len(infos)) + struct.pack("<Q", len(kvs)))
    hdr += b"".join(_kv(k, t, r) for k, t, r in kvs)
    rels = []
    rel = 0
    for name, dims, ttype, blob in infos:
        hdr += struct.pack("<Q", len(name)) + name
        hdr += struct.pack("<I", len(dims))
        hdr += struct.pack("<" + "Q" * len(dims), *dims)
        hdr += struct.pack("<I", ttype)
        hdr += struct.pack("<Q", rel)
        rels.append(rel)
        rel += _align(len(blob), alignment)
    hdr = bytearray(hdr)
    data_start = _align(len(hdr), alignment)
    if len(hdr) < data_start:
        hdr += b"\0" * (data_start - len(hdr))
    with open(path, "wb") as f:
        f.write(bytes(hdr))
        for blob in [b for _, _, _, b in infos]:
            f.write(blob)
            f.write(b"\0" * (_align(len(blob), alignment) - len(blob)))


def test_keep_gguf_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    n, m = 32, 64
    e0 = rng.normal(0, 2, (n, m)).astype(np.float32)
    e1 = rng.normal(0, 2, (n, m)).astype(np.float32)
    dense = rng.normal(size=(8, 8)).astype(np.float32)
    src = tmp_path / "src.gguf"
    _write_source(src, e0, e1, dense)

    out = write_keep_gguf([src], tmp_path / "keep.gguf",
                          {b"blk.0.ffn_gate_exps.weight": [1]})
    assert out.exists() and out.stat().st_size > 0

    v, kvs, infos, hdr_end = read_gguf_header(out)
    names = [nm for nm, *_ in infos]
    assert b"blk.0.ffn_gate_exps.weight" in names
    assert b"blk.0.dense.weight" in names
    stack_info = next(i for i in infos
                      if i[0] == b"blk.0.ffn_gate_exps.weight")
    assert stack_info[1] == (n, m, 1)      # one expert kept

    # dequantize the kept stack and compare against the SOURCE expert-1
    # slice's own dequant — byte-exact surgery, independent of quant
    # packing conventions.
    from ultratensor.gguf_factored import _dequant, _file_slice
    alignment = 32
    data_start = _align(hdr_end, alignment)
    raw = _file_slice(out, data_start + stack_info[3],
                      _tensor_byte_size((n, m, 1), GGML_Q8_0))
    kept = _dequant(raw, GGML_Q8_0, (n, m, 1))
    assert kept.shape == (1, m, n)
    # source slice: expert 1's bytes from the original file
    src_v, src_kvs, src_infos, src_end = read_gguf_header(src)
    src_ds = _align(src_end, alignment)
    src_info = next(i for i in src_infos
                    if i[0] == b"blk.0.ffn_gate_exps.weight")
    src_size = _tensor_byte_size(src_info[1], src_info[2])
    ref_raw = _file_slice(src, src_ds + src_info[3] + src_size // 2,
                          src_size // 2)
    ref = _dequant(ref_raw, GGML_Q8_0, (n, m, 1))
    assert np.array_equal(kept, ref)

    # dense tensor copied verbatim
    dense_info = next(i for i in infos if i[0] == b"blk.0.dense.weight")
    raw = _file_slice(out, data_start + dense_info[3], 8 * 8 * 4)
    assert np.allclose(np.frombuffer(raw, np.float32), dense.ravel())


def test_keep_plan_requires_3d(tmp_path):
    rng = np.random.default_rng(1)
    src = tmp_path / "x.gguf"
    _write_source(src, rng.normal(size=(32, 64)).astype(np.float32),
                  rng.normal(size=(32, 64)).astype(np.float32),
                  rng.normal(size=(8, 8)).astype(np.float32))
    with pytest.raises(ValueError):
        write_keep_gguf([src], tmp_path / "y.gguf",
                        {b"blk.0.dense.weight": [0]})
