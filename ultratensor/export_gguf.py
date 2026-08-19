"""GGUF exporter: requantize a GGUF (or shard) to llama.cpp Q2_K.

Streams tensor-by-tensor (one row of one tensor in RAM at a time),
writes a standard GGUF that llama.cpp can load directly — making the
shrunk model runnable, unlike the safetensors q2_0 archival format.

All KV metadata (tokenizer, arch, chat template) is copied verbatim;
only quantized tensors are converted (Q3_K/Q5_K/Q6_K/Q4_K/Q8_0/... →
Q2_K); F32/F16/BF16/I32 and other raw tensors are copied byte-for-byte.
"""
from __future__ import annotations

import struct
import time
from pathlib import Path
from typing import Optional

import numpy as np

from .dequant import BLOCK_ALIGN, BLOCK_BYTES, dequantize_rows
from .quant import q2_k_quantize
from .stream import tensor_inventory

QUANT_TYPES = set(BLOCK_BYTES)   # everything our dequantizers understand

_GGUF_VALUE_SIZE = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1,
                    10: 8, 11: 8, 12: 8}


def _read_kv_raw(f):
    """Return the verbatim KV-section bytes (for copying to the output)."""
    f.seek(4)  # skip magic
    _ver, n_tensors, n_kv = struct.unpack("<IQQ", f.read(4 + 8 + 8))
    start = f.tell()
    for _ in range(n_kv):
        klen = struct.unpack("<Q", f.read(8))[0]
        f.read(klen)
        vtype = struct.unpack("<I", f.read(4))[0]
        if vtype == 8:
            slen = struct.unpack("<Q", f.read(8))[0]
            f.read(slen)
        elif vtype == 9:
            etype = struct.unpack("<I", f.read(4))[0]
            n = struct.unpack("<Q", f.read(8))[0]
            if etype == 8:
                for _ in range(n):
                    slen = struct.unpack("<Q", f.read(8))[0]
                    f.read(slen)
            else:
                f.read(n * _GGUF_VALUE_SIZE.get(etype, 8))
        else:
            f.read(_GGUF_VALUE_SIZE.get(vtype, 8))
    return start, f.tell(), n_tensors, n_kv


def _quant_type_id(name: str) -> int:
    from gguf.constants import GGMLQuantizationType
    return GGMLQuantizationType[name].value


def export_q2k(src: Path, dst: Path, only: Optional[set] = None,
               max_tensors: Optional[int] = None, progress: bool = True,
               use_torch: bool = False, chunk_blocks: Optional[int] = None):
    """Convert every quantized tensor in src to Q2_K; copy the rest."""
    t0 = time.time()
    inv = tensor_inventory(src)          # [(name, qtype, dims, raw_offset)]
    with open(src, "rb") as f:
        kv_start, kv_end, _nt, n_kv = _read_kv_raw(f)
        f.seek(kv_start)
        kv_bytes = f.read(kv_end - kv_start)
    # GGUF offsets are relative to the (32-byte aligned) data section
    info_size = 0
    for name, qtype, dims, _off in inv:
        info_size += 8 + len(name.encode()) + 4 + 8 * len(dims) + 4 + 8
    src_data_start = ((kv_end + info_size + 31) // 32) * 32
    src_abs = [(n, q, d, src_data_start + o) for n, q, d, o in inv]

    with open(dst, "wb") as out:
        out.write(b"GGUF")
        out.write(struct.pack("<I", 3))
        out.write(struct.pack("<QQ", len(inv), n_kv))
        out.write(kv_bytes)
        info_size = 0
        for name, qtype, dims, _off in inv:
            info_size += 8 + len(name.encode()) + 4 + 8 * len(dims) + 4 + 8
        data_start = ((out.tell() + info_size + 31) // 32) * 32
        offset = 0  # GGUF offsets are relative to the data section start
        for name, qtype, dims, _off in inv:
            nb = name.encode()
            out.write(struct.pack("<Q", len(nb)))
            out.write(nb)
            out.write(struct.pack("<I", len(dims)))
            out.write(struct.pack(f"<{len(dims)}Q", *dims))
            if qtype in QUANT_TYPES:
                out.write(struct.pack("<I", 10))  # Q2_K
            else:
                out.write(struct.pack("<I", _quant_type_id(qtype)))
            out.write(struct.pack("<Q", offset))
            n_elems = int(np.prod(dims))
            if qtype in QUANT_TYPES:
                offset += (n_elems // 256) * 84
            else:
                offset += n_elems * _elem_bytes(qtype)
        # pad to alignment
        pad = data_start - out.tell()
        if pad:
            out.write(b"\x00" * pad)
        # --- tensors ---
        with open(src, "rb") as sf:
            n = 0
            for (name, qtype, dims, src_off) in src_abs:
                if only is not None and name not in only:
                    continue
                n_elems = int(np.prod(dims))
                if qtype in QUANT_TYPES and dims[0] % 256 == 0:
                    rows_total = int(np.prod(dims[1:])) if len(dims) > 1 else 1
                    cols = int(dims[0])
                    row_bytes = ((cols // BLOCK_ALIGN[qtype]) * BLOCK_BYTES[qtype])
                    # bound fp32 footprint per chunk (~128 MB)
                    chunk_rows = max(1, (1 << 27) // max(cols * 4, 1))
                    kw = {}
                    if chunk_blocks:
                        kw["chunk_blocks"] = chunk_blocks
                    for r0 in range(0, rows_total, chunk_rows):
                        r1 = min(rows_total, r0 + chunk_rows)
                        sf.seek(src_off + r0 * row_bytes)
                        data = np.frombuffer(sf.read((r1 - r0) * row_bytes),
                                             np.uint8)
                        W = dequantize_rows(
                            qtype, data,
                            tuple(int(x) for x in dims[:1]) + (r1 - r0,),
                            0, r1 - r0)
                        sc, qs, d, dmin = q2_k_quantize(W, use_torch=use_torch,
                                                        **kw)
                        data = np.concatenate(
                            [sc, qs,
                             d.view(np.uint8).reshape(-1, 2),
                             dmin.view(np.uint8).reshape(-1, 2)],
                            axis=1).astype(np.uint8)
                        out.write(data.tobytes())
                    note = "Q2_K"
                else:
                    nbytes = n_elems * _elem_bytes(qtype)
                    if qtype in QUANT_TYPES:
                        nbytes = (n_elems // BLOCK_ALIGN[qtype]) * BLOCK_BYTES[qtype]
                    sf.seek(src_off)
                    out.write(sf.read(nbytes))
                    note = "copy"
                n += 1
                if progress:
                    print(f"[{n}] {name}: {qtype} -> {note}")
                if max_tensors and n >= max_tensors:
                    break
    print(f"exported {dst} in {time.time()-t0:.0f}s")
    return dst


def _elem_bytes(qtype: str) -> int:
    if qtype in ("F32", "I32"):
        return 4
    if qtype in ("F16", "BF16"):
        return 2
    return 1


def _load_full(sf, offset: int, dims, qtype: str) -> np.ndarray:
    """Materialize one tensor fully (per-row chunked dequant)."""
    true_shape = tuple(reversed(dims))
    n = int(np.prod(true_shape))
    rows = true_shape[0] if len(true_shape) > 1 else 1
    cols = n // rows
    data = np.fromfile(sf, dtype=np.uint8, count=(n // BLOCK_ALIGN[qtype])
                       * BLOCK_BYTES[qtype], offset=offset)
    W = np.empty((rows, cols), np.float32)
    chunk = max(1, (1 << 26) // max(cols, 1))
    for r0 in range(0, rows, chunk):
        r1 = min(rows, r0 + chunk)
        W[r0:r1] = dequantize_rows(qtype, data,
                                   tuple(int(x) for x in dims), r0, r1 - r0)
    return W
