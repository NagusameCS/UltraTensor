"""Factored GGUF container (UltraTensor -> llama Phase 1).

A factored tensor stores W (m x n) as:

    W ~= U @ C      U: fp16 [m, k]   (the shared basis)
                    C: uq4 codes [k, n] (scales fp32 per 32-col block + 4-bit)

C uses the UltraTensor uq4 layout: value = scale * (code - 8), scale = amax/8,
packed 2 codes per byte (lo nibble = even index). Stored tensor types:

    <name>.factored_U   GGML_TYPE_F16
    <name>.factored_C   GGML_TYPE_FACTORED_C (= 2048, custom; kernel work is
                        Phase 2/3 — llama.cpp does not know this type yet)

Metadata: a `ultratensor.factored_manifest` KV (JSON string) documents every
factored tensor: original name/shape, rank k, uq4 block, certificate fields.
The container is self-describing; a reader (here) reconstructs W = U @ C.

Limits (v1): single-file GGUF sources, dense numpy SVD (fine up to ~100M
elements; streaming/randomized SVD for expert-class tensors comes with the
Phase 2 kernel work).
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

from .quant import uq4_dequantize, uq4_quantize

GGML_TYPE_F16 = 1
GGML_TYPE_Q8_0 = 8  # GGUF tensor type id (7 would be Q5_1)
GGML_TYPE_FACTORED_C = 2048  # UltraTensor custom: uq4 codes + fp32 scales

_QTYPE_NAMES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
    8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K",
    13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS",
    18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S",
    22: "IQ2_S", 23: "IQ4_XS", 29: "IQ1_M", 30: "BF16",
    34: "TQ1_0", 35: "TQ2_0",
}

MANIFEST_KEY = "ultratensor.factored_manifest"
F16_BLOCK = 32  # f16 elements per GGUF alignment block
UQ4_BLOCK = 32  # cols per code block


# ---------------------------------------------------------------------------
# raw GGUF header I/O (byte-level, like fix_drafter_vocab.py)
# ---------------------------------------------------------------------------

_VT = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def _read_exact(f, n):
    b = f.read(n)
    if len(b) != n:
        raise EOFError(f"wanted {n} bytes, got {len(b)}")
    return b


def read_gguf_header(path):
    """-> (version, kvs [(key,type,raw)], infos [(name,dims,type,offset)],
           data_start)"""
    with open(path, "rb") as f:
        assert f.read(4) == b"GGUF", f"{path}: not GGUF"
        version, = struct.unpack("<I", _read_exact(f, 4))
        n_tensors, = struct.unpack("<Q", _read_exact(f, 8))
        n_kv, = struct.unpack("<Q", _read_exact(f, 8))
        kvs = []
        for _ in range(n_kv):
            klen, = struct.unpack("<Q", _read_exact(f, 8))
            key = _read_exact(f, klen)
            vtype, = struct.unpack("<I", _read_exact(f, 4))
            if vtype == 8:
                slen, = struct.unpack("<Q", _read_exact(f, 8))
                val = _read_exact(f, slen)
                raw = struct.pack("<Q", slen) + val
            elif vtype == 9:
                etype, = struct.unpack("<I", _read_exact(f, 4))
                cnt, = struct.unpack("<Q", _read_exact(f, 8))
                chunks = [struct.pack("<IQ", etype, cnt)]
                for _ in range(cnt):
                    if etype == 8:
                        slen, = struct.unpack("<Q", _read_exact(f, 8))
                        val = _read_exact(f, slen)
                        chunks.append(struct.pack("<Q", slen) + val)
                    else:
                        chunks.append(_read_exact(f, _VT[etype]))
                raw = b"".join(chunks)
            else:
                raw = _read_exact(f, _VT[vtype])
            kvs.append((key, vtype, raw))
        infos = []
        for _ in range(n_tensors):
            nlen, = struct.unpack("<Q", _read_exact(f, 8))
            name = _read_exact(f, nlen)
            nd, = struct.unpack("<I", _read_exact(f, 4))
            dims = tuple(struct.unpack("<" + "Q" * nd, _read_exact(f, 8 * nd)))
            ttype, = struct.unpack("<I", _read_exact(f, 4))
            off, = struct.unpack("<Q", _read_exact(f, 8))
            infos.append((name, dims, ttype, off))
        pos = f.tell()
    return version, kvs, infos, pos


def _kv(k, t, raw):
    return struct.pack("<Q", len(k)) + k + struct.pack("<I", t) + raw


def _str_kv(k: bytes, s: str) -> bytes:
    b = s.encode("utf-8")
    return _kv(k, 8, struct.pack("<Q", len(b)) + b)


def _align(n, a=32):
    return (n + a - 1) // a * a


# ---------------------------------------------------------------------------
# factoring
# ---------------------------------------------------------------------------

def factor_matrix(W: np.ndarray, rank: int | None = None,
                  energy: float = 0.99) -> tuple[np.ndarray, np.ndarray]:
    """SVD-factor W [m,n] -> (U fp16 [m,k], C fp32 [k,n]).

    k = min(rank or inf, min(m,n), #components for `energy` of variance).
    """
    W = W.astype(np.float32)
    m, n = W.shape
    U, s, Vt = np.linalg.svd(W, full_matrices=False)
    if rank is not None:
        k = min(rank, m, n)
    else:
        s2 = s * s
        total = s2.sum()
        cum = np.cumsum(s2)
        k = int(np.searchsorted(cum, energy * total)) + 1
        k = max(1, min(k, m, n))
    U = U[:, :k]
    C = (s[:k, None] * Vt[:k, :]).astype(np.float32)
    return U.astype(np.float16), C


def encode_codes(C: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """uq4-encode C [k,n] -> (scales fp32 [k, n/32], packed uint8 [k, n/2])."""
    scales, packed = uq4_quantize(C, block=UQ4_BLOCK)
    return scales.astype(np.float32), packed.astype(np.uint8)


def decode_codes(scales: np.ndarray, packed: np.ndarray) -> np.ndarray:
    if scales.ndim == 3:
        E = scales.shape[0]
        return np.stack([uq4_dequantize(scales[e], packed[e])
                         for e in range(E)])
    return uq4_dequantize(scales, packed)


# GGML block sizes/bytes per block (llama.cpp ggml type traits) for byte-size math
_GGML_TYPE_BLOCK = {
    0: (1, 4),    # F32
    1: (1, 2),    # F16
    2: (32, 18),  # Q4_0
    3: (32, 20),  # Q4_1
    6: (32, 22),  # Q5_0
    7: (32, 24),  # Q5_1
    8: (32, 34),  # Q8_0
    9: (32, 36),  # Q8_1
    10: (256, 84),  # Q2_K
    11: (256, 110),  # Q3_K
    12: (256, 144),  # Q4_K
    13: (256, 176),  # Q5_K
    14: (256, 210),  # Q6_K
    16: (256, 66),   # IQ2_XXS
    17: (256, 74),   # IQ2_XS
    18: (256, 98),   # IQ3_XXS
    19: (256, 50),   # IQ1_S
    20: (32, 18),    # IQ4_NL
    21: (256, 82),   # IQ2_S
    22: (256, 110),  # IQ3_S
    23: (256, 56),   # IQ1_M
    24: (256, 136),  # IQ4_XS
    25: (1, 2),      # I16
    26: (1, 4),      # I32
    30: (1, 2),      # BF16
}


def _tensor_byte_size(dims, ttype):
    n = int(np.prod(dims)) if dims else 0
    if ttype == GGML_TYPE_FACTORED_C:
        # gguf dims: (cols, rank) or (cols, rank, E)
        cols, k = int(dims[0]), int(dims[1])
        size = k * (4 * (cols // UQ4_BLOCK) + cols // 2)
        return size * (int(dims[2]) if len(dims) == 3 else 1)
    if ttype in _GGML_TYPE_BLOCK:
        block, bsize = _GGML_TYPE_BLOCK[ttype]
        return ((n + block - 1) // block) * bsize
    raise NotImplementedError(f"type {ttype}")


def _dequant(src_data: bytes, ttype: int, dims) -> np.ndarray:
    from .dequant import dequantize

    name = _QTYPE_NAMES.get(ttype)
    if name is None:
        raise NotImplementedError(f"source quant type {ttype} not supported yet")
    buf = np.frombuffer(src_data, np.uint8)
    # dims are ne0-first; dequantize returns the true (reversed) order
    return dequantize(name, buf, dims)


# ---------------------------------------------------------------------------
# the container writer
# ---------------------------------------------------------------------------

def write_factored_gguf(src: str | Path, out: str | Path,
                        patterns: list[str] | None = None,
                        rank: int | None = None, energy: float = 0.99,
                        drop_unmatched: bool = False):
    """Factor matching 2-D tensors of a single-file GGUF into the factored
    container. Non-matching tensors are copied byte-for-byte, unless
    drop_unmatched is set (then only the factored tensors are written)."""
    src, out = Path(src), Path(out)
    version, kvs, infos, hdr_end = read_gguf_header(src)
    alignment = 32
    for k, t, raw in kvs:
        if k == b"general.alignment":
            alignment = struct.unpack("<I", raw)[0]
    old_data_start = _align(hdr_end, alignment)

    names = {b"ultratensor.factored_manifest"}

    plan = []  # (info, kind) kind: "copy" | "factored"
    manifest = {"version": 1, "tensors": []}
    for name, dims, ttype, off in infos:
        hit = (patterns is None) or any(p in name.decode() for p in patterns)
        if hit and len(dims) == 2 and min(dims) > 4:
            # GGUF dims are ne0-first; numpy W = [dims[1], dims[0]]
            m, n = int(dims[1]), int(dims[0])
            raw = np.frombuffer(_file_slice(src, old_data_start + off,
                                            _tensor_byte_size(dims, ttype)),
                                np.uint8)
            W = _dequant(raw.tobytes(), ttype, dims)
            U, C = factor_matrix(W, rank=rank, energy=energy)
            scales, packed = encode_codes(C)
            plan.append((name, ("factored", m, n, U, scales, packed)))
            manifest["tensors"].append({
                "name": name.decode(), "shape": [m, n], "rank": int(U.shape[1]),
                "uq4_block": UQ4_BLOCK, "source_type": ttype,
            })
        elif hit and len(dims) == 3 and min(dims[:2]) > 4:
            # expert stack: gguf (n, m, E) -> numpy W3 = (E, m, n)
            n, m, E = int(dims[0]), int(dims[1]), int(dims[2])
            raw = np.frombuffer(_file_slice(src, old_data_start + off,
                                            _tensor_byte_size(dims, ttype)),
                                np.uint8)
            W3 = _dequant(raw.tobytes(), ttype, dims)
            U0, C0 = factor_matrix(W3[0], rank=rank, energy=energy)
            k = U0.shape[1]
            Us = np.empty((E, m, k), np.float16)
            Cs = np.empty((E, k, n), np.float32)
            for e in range(E):
                Ue, Ce = factor_matrix(W3[e], rank=k)
                Us[e] = Ue
                Cs[e] = Ce
            scales3 = np.empty((E, k, n // UQ4_BLOCK), np.float32)
            packed3 = np.empty((E, k, n // 2), np.uint8)
            for e in range(E):
                scales3[e], packed3[e] = encode_codes(Cs[e])
            plan.append((name, ("factored3", E, m, n, Us, scales3, packed3)))
            manifest["tensors"].append({
                "name": name.decode(), "shape": [E, m, n], "rank": k,
                "uq4_block": UQ4_BLOCK, "source_type": ttype,
            })
        else:
            plan.append((name, ("copy",)))

    manifest_raw = json.dumps(manifest)
    new_kvs = [kv for kv in kvs if kv[0] not in names]
    new_kvs.append((MANIFEST_KEY.encode(), 8,
                    struct.pack("<Q", len(manifest_raw)) +
                    manifest_raw.encode()))

    # tensor infos: factored tensors expand to two
    infos_out = []  # (name, dims, ttype, kind, *payload)
    for (name, dims, ttype, off), kind in zip(infos, [p[1] for p in plan]):
        if kind[0] == "copy":
            if drop_unmatched:
                continue
            infos_out.append((name, dims, ttype, "copy",
                              off, _tensor_byte_size(dims, ttype)))
        elif kind[0] == "factored":
            _, m, n, U, scales, packed = kind
            k = U.shape[1]
            u_name = name + b".factored_U"
            c_name = name + b".factored_C"
            infos_out.append((u_name, (k, m), GGML_TYPE_F16, "u", U))
            infos_out.append((c_name, (n, k), GGML_TYPE_FACTORED_C, "c",
                              (scales, packed)))
        else:  # factored3
            _, E, m, n, Us, scales3, packed3 = kind
            k = Us.shape[2]
            u_name = name + b".factored_U"
            c_name = name + b".factored_C"
            infos_out.append((u_name, (k, m, E), GGML_TYPE_F16, "u3", Us))
            infos_out.append((c_name, (n, k, E), GGML_TYPE_FACTORED_C, "c3",
                              (scales3, packed3)))

    hdr = (b"GGUF" + struct.pack("<I", version) +
           struct.pack("<Q", len(infos_out)) +
           struct.pack("<Q", len(new_kvs)))
    hdr += b"".join(_kv(k, t, r) for k, t, r in new_kvs)
    for name, dims, ttype, kind, *_ in infos_out:
        hdr += struct.pack("<Q", len(name)) + name
        hdr += struct.pack("<I", len(dims))
        hdr += struct.pack("<" + "Q" * len(dims), *dims)
        hdr += struct.pack("<I", ttype)
        hdr += struct.pack("<Q", 0)  # offset patched below
    # patch offsets (relative to data start, like gguf-py)
    hdr = bytearray(hdr)
    data_start = _align(len(hdr), alignment)
    pos = 0
    # walk header again to locate offset fields
    magic_len = 4 + 4 + 8 + 8
    pos = magic_len
    for k, t, r in new_kvs:
        pos += 8 + len(k) + 4 + len(r)
    offs = []
    for name, dims, ttype, kind, *_ in infos_out:
        pos += 8 + len(name) + 4 + 8 * len(dims) + 4
        offs.append(pos)
        pos += 8
    offsets = []
    rel = 0
    for (name, dims, ttype, kind, *pay), offpos in zip(infos_out, offs):
        struct.pack_into("<Q", hdr, offpos, rel)
        offsets.append(rel)
        if kind == "copy":
            rel += _align(pay[1], alignment)
        elif kind == "u":
            m, k = dims
            rel += _align(m * k * 2, alignment)
        elif kind == "u3":
            _, m, E = dims  # gguf (k, m, E)
            rel += _align(E * m * int(dims[0]) * 2, alignment)
        elif kind == "c":
            n, k = dims  # gguf dims: (cols, rank)
            rel += _align(k * (4 * (n // UQ4_BLOCK) + n // 2), alignment)
        else:  # c3
            n, k, E = dims
            rel += _align(E * k * (4 * (n // UQ4_BLOCK) + n // 2), alignment)
    if len(hdr) < data_start:
        hdr += b"\0" * (data_start - len(hdr))

    with open(src, "rb") as s, open(out, "wb") as d:
        d.write(bytes(hdr))
        for (name, dims, ttype, kind, *pay), rel_off in zip(infos_out, offsets):
            if kind == "copy":
                off, size = pay
                s.seek(old_data_start + off)
                _copy(s, d, size)
                d.write(b"\0" * (_align(size, alignment) - size))
            elif kind == "u":
                U = pay[0]
                blob = np.ascontiguousarray(U.astype(np.float16)
                                            ).view(np.uint8).tobytes()
                d.write(blob)
                d.write(b"\0" * (_align(len(blob), alignment) - len(blob)))
            elif kind == "u3":
                Us = pay[0]
                blob = np.ascontiguousarray(Us.astype(np.float16)
                                            ).view(np.uint8).tobytes()
                d.write(blob)
                d.write(b"\0" * (_align(len(blob), alignment) - len(blob)))
            elif kind == "c":
                scales, packed = pay[0]
                # Row-interleaved: scales[r] (4B x nb) then packed[r] (n/2 B),
                # matching the runtime kernels' row stride (4*nb + n/2).
                for r in range(scales.shape[0]):
                    d.write(np.ascontiguousarray(scales[r]).tobytes())
                    d.write(np.ascontiguousarray(packed[r]).tobytes())
                n, k = dims  # gguf dims: (cols, rank)
                size = k * (4 * (n // UQ4_BLOCK) + n // 2)
                d.write(b"\0" * (_align(size, alignment) - size))
            else:  # c3
                scales3, packed3 = pay[0]
                # Per-expert, per-row interleaved (kernel stride 4*nb + n/2).
                for e in range(scales3.shape[0]):
                    for r in range(scales3.shape[1]):
                        d.write(np.ascontiguousarray(scales3[e, r]).tobytes())
                        d.write(np.ascontiguousarray(packed3[e, r]).tobytes())
                n, k, E = dims
                size = E * k * (4 * (n // UQ4_BLOCK) + n // 2)
                d.write(b"\0" * (_align(size, alignment) - size))
    return out


def _file_slice(path: Path, off: int, size: int) -> bytes:
    with open(path, "rb") as f:
        f.seek(off)
        return f.read(size)


def _copy(s, d, size, buf=64 * 1024 * 1024):
    while size > 0:
        chunk = s.read(min(buf, size))
        if not chunk:
            raise EOFError("source truncated")
        d.write(chunk)
        size -= len(chunk)


# ---------------------------------------------------------------------------
# the reader (round-trip / Phase 3 loader reference)
# ---------------------------------------------------------------------------

def read_factored_gguf(path: str | Path):
    """-> (kvs, tensors) where tensors maps name -> dict with kind:
    'dense'  {dtype, data}, 'basis' {U}, 'codes' {scales, packed, shape}."""
    version, kvs, infos, hdr_end = read_gguf_header(path)
    alignment = 32
    for k, t, raw in kvs:
        if k == b"general.alignment":
            alignment = struct.unpack("<I", raw)[0]
    data_start = _align(hdr_end, alignment)
    manifest = None
    for k, t, raw in kvs:
        if k == MANIFEST_KEY.encode():
            slen, = struct.unpack("<Q", raw[:8])
            manifest = json.loads(raw[8:8 + slen].decode())
    out = {}
    rel = 0
    with open(path, "rb") as f:
        for name, dims, ttype, off in infos:
            f.seek(data_start + rel)
            if ttype == GGML_TYPE_FACTORED_C:
                if len(dims) == 3:
                    n, k, E = dims
                    nb = n // UQ4_BLOCK
                    raw = f.read(E * k * (4 * nb + n // 2))
                    buf = np.frombuffer(raw, np.uint8)
                    scales = np.ndarray((E, k, nb), np.float32, buf,
                                        strides=(k * (4 * nb + n // 2),
                                                 4 * nb + n // 2, 4))
                    packed = np.ndarray((E, k, n // 2), np.uint8, buf,
                                        strides=(k * (4 * nb + n // 2),
                                                 4 * nb + n // 2, 1),
                                        offset=4 * nb)
                    out[name.decode()] = {"kind": "codes", "scales": scales,
                                          "packed": packed,
                                          "shape": (E, k, n)}
                    size = E * k * (4 * nb + n // 2)
                else:
                    n, k = dims
                    nb = n // UQ4_BLOCK
                    raw = f.read(k * (4 * nb + n // 2))
                    buf = np.frombuffer(raw, np.uint8)
                    scales = np.ndarray((k, nb), np.float32, buf,
                                        strides=(4 * nb + n // 2, 4))
                    packed = np.ndarray((k, n // 2), np.uint8, buf,
                                        strides=(4 * nb + n // 2, 1),
                                        offset=4 * nb)
                    out[name.decode()] = {"kind": "codes", "scales": scales,
                                          "packed": packed, "shape": (k, n)}
                    size = k * (4 * nb + n // 2)
            elif ttype == GGML_TYPE_F16 and name.decode().endswith(".factored_U"):
                if len(dims) == 3:
                    k, m, E = dims
                    U = np.frombuffer(f.read(E * m * k * 2),
                                      np.float16).reshape(E, m, k)
                    out[name.decode()] = {"kind": "basis", "U": U,
                                          "shape": (E, m, k)}
                    size = E * m * k * 2
                else:
                    k, m = dims
                    U = np.frombuffer(f.read(m * k * 2),
                                      np.float16).reshape(m, k)
                    out[name.decode()] = {"kind": "basis", "U": U}
                    size = m * k * 2
            else:
                size = _tensor_byte_size(dims, ttype)
                out[name.decode()] = {"kind": "dense", "ttype": ttype,
                                      "data": f.read(size)}
            rel += _align(size, alignment)
    return manifest, out


def reconstruct(manifest, tensors, name: str) -> np.ndarray:
    """W_hat = U @ dequant(C) for a factored tensor (2-D or 3-D)."""
    t = tensors.get(name)
    if t is None or t["kind"] != "codes":
        raise KeyError(name)
    base = name[:-len(".factored_C")]
    U = tensors[base + ".factored_U"]["U"].astype(np.float32)
    C = decode_codes(t["scales"], t["packed"])
    if C.ndim == 3:  # (E, k, n)
        return np.einsum("emk,ekn->emn", U, C)
    return U @ C
