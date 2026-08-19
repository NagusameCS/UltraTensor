"""GGUF subnetwork surgery — drop experts from 3-D expert stacks.

V4-Coder Phase-2 build: keep only the code-active experts of each
`blk.{L}.ffn_{gate,up,down}_exps.weight` stack. Expert stacks are
expert-major in the file (the ExpertStore reader slices per expert by
offset), so a kept stack is the concatenation of the kept expert
blocks; every other tensor is copied byte-for-byte.

Layout facts reused from ultratensor/gguf_factored.py:
  - header rebuild + offset patching identical to the factored writer
  - tensor byte sizes per type via _tensor_byte_size
  - per-expert bytes = stack bytes // E (expert-major)

Keep plan: {tensor_name: sorted expert indices}. Tensors not in the
plan are copied whole; plan entries must be 3-D stacks.

This module is INPUT-AGNOSTIC (synthetic or real shards). The real
build runs via scripts/v4_coder_keep.py.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

from .gguf_factored import (
    _align,
    _copy,
    _kv,
    _tensor_byte_size,
    read_gguf_header,
)

MANIFEST_KEY = "ultratensor.keep_manifest"


def write_keep_gguf(srcs, out, keep):
    """Drop experts from stacks per the keep plan; copy everything else.

    srcs: list of source GGUF paths (multi-shard; tensor order follows
        shard 0's order).
    keep: {tensor_name: [expert indices]} for 3-D expert stacks.
    out:  output path.
    Returns the output Path.
    """
    srcs = [Path(s) for s in srcs]
    headers = [read_gguf_header(s) for s in srcs]
    version = headers[0][0]
    alignment = 32
    for k, t, raw in headers[0][1]:
        if k == b"general.alignment":
            alignment = struct.unpack("<I", raw)[0]

    # global tensor map: name -> (shard_idx, dims, ttype, off, size)
    tensor_shard = {}
    for si, (v, kvs, infos, hdr_end) in enumerate(headers):
        data_start = _align(hdr_end, alignment)
        for name, dims, ttype, off in infos:
            tensor_shard[name] = (si, dims, ttype,
                                  data_start + off,
                                  _tensor_byte_size(dims, ttype))

    # order: all shards' infos in shard order, first occurrence wins
    ordered = []
    seen = set()
    for (v, kvs, infos, _) in headers:
        for name, *_ in infos:
            if name not in seen:
                seen.add(name)
                ordered.append(name)

    plan = []
    manifest = {"version": 1, "dropped": {}, "kept": {}}
    for name in ordered:
        si, dims, ttype, off, size = tensor_shard[name]
        if name in keep:
            if len(dims) != 3:
                raise ValueError(f"keep plan entry {name!r} is not a "
                                 f"3-D stack: dims={dims}")
            E = dims[2]
            exp_size = size // E
            idx = keep[name]
            blob_offsets = [(si, off + e * exp_size, exp_size)
                            for e in idx]
            plan.append((name, dims, ttype, ("stack", blob_offsets,
                                             len(idx))))
            manifest["kept"][name.decode()] = idx
            manifest["dropped"][name.decode()] = E - len(idx)
        else:
            plan.append((name, dims, ttype, ("copy", si, off, size)))

    new_kvs = [kv for kv in headers[0][1]
               if kv[0] != MANIFEST_KEY.encode()]
    manifest_raw = json.dumps(manifest)
    new_kvs.append((MANIFEST_KEY.encode(), 8,
                    struct.pack("<Q", len(manifest_raw)) +
                    manifest_raw.encode()))

    infos_out = []
    for name, dims, ttype, kind in plan:
        if kind[0] == "copy":
            _, si, off, size = kind
            infos_out.append((name, dims, ttype, "copy", si, off, size))
        else:  # stack
            _, blobs, K = kind
            n, m, _ = dims
            infos_out.append((name, (n, m, K), ttype, "stack", blobs))

    hdr = (b"GGUF" + struct.pack("<I", version) +
           struct.pack("<Q", len(infos_out)) +
           struct.pack("<Q", len(new_kvs)))
    hdr += b"".join(_kv(k, t, r) for k, t, r in new_kvs)
    for name, dims, ttype, *_ in infos_out:
        hdr += struct.pack("<Q", len(name)) + name
        hdr += struct.pack("<I", len(dims))
        hdr += struct.pack("<" + "Q" * len(dims), *dims)
        hdr += struct.pack("<I", ttype)
        hdr += struct.pack("<Q", 0)  # offset patched below

    hdr = bytearray(hdr)
    data_start = _align(len(hdr), alignment)
    pos = 4 + 4 + 8 + 8
    for k, t, r in new_kvs:
        pos += 8 + len(k) + 4 + len(r)
    offs = []
    for name, dims, ttype, *_ in infos_out:
        pos += 8 + len(name) + 4 + 8 * len(dims) + 4
        offs.append(pos)
        pos += 8
    rel = 0
    offsets = []
    for (name, dims, ttype, kind, *pay), offpos in zip(infos_out, offs):
        struct.pack_into("<Q", hdr, offpos, rel)
        offsets.append(rel)
        if kind == "copy":
            rel += _align(pay[2], alignment)
        else:
            blobs = pay[0]
            rel += _align(sum(b[2] for b in blobs), alignment)
    if len(hdr) < data_start:
        hdr += b"\0" * (data_start - len(hdr))

    with open(out, "wb") as d:
        d.write(bytes(hdr))
        for (name, dims, ttype, kind, *pay), rel_off in zip(infos_out,
                                                            offsets):
            if kind == "copy":
                si, off, size = pay
                with open(srcs[si], "rb") as s:
                    s.seek(off)
                    _copy(s, d, size)
                d.write(b"\0" * (_align(size, alignment) - size))
            else:
                blobs = pay[0]
                for si, off, size in blobs:
                    with open(srcs[si], "rb") as s:
                        s.seek(off)
                        _copy(s, d, size)
                total = sum(b[2] for b in blobs)
                d.write(b"\0" * (_align(total, alignment) - total))
    return Path(out)


def write_uniform_keep_gguf(srcs, out, keep, col_keep, remap,
                            kv_overrides=None):
    """Uniform-E extraction: the inverse splice for a small resident
    coder.  Besides 3-D expert stacks it can slice F32 router columns
    (so routing is restricted to the kept experts), remap I32 hash
    tables (dropped experts -> a kept fallback), and override KV
    metadata (e.g. deepseek4.expert_count -> K so llama.cpp's
    check_tensor_dims accepts the file).

    keep:        {name: [expert idx]}   for 3-D stacks
    col_keep:    {name: [col idx]}      for F32 2-D (n,m)->(n,K) or 1-D
    remap:       {name: {old: new}}     for I32 value tensors
    kv_overrides:{name: new_raw}        same type, new value bytes
    """
    import numpy as np

    srcs = [Path(s) for s in srcs]
    headers = [read_gguf_header(s) for s in srcs]
    version = headers[0][0]
    alignment = 32
    for k, t, raw in headers[0][1]:
        if k == b"general.alignment":
            alignment = struct.unpack("<I", raw)[0]

    tensor_shard = {}
    for si, (v, kvs, infos, hdr_end) in enumerate(headers):
        data_start = _align(hdr_end, alignment)
        for name, dims, ttype, off in infos:
            tensor_shard[name] = (si, dims, ttype, data_start + off)

    ordered = []
    seen = set()
    for (v, kvs, infos, _) in headers:
        for name, *_ in infos:
            if name not in seen:
                seen.add(name)
                ordered.append(name)

    kv_overrides = kv_overrides or {}
    plan = []
    manifest = {"version": 2, "uniform": True,
                "kept": {}, "dropped": {}, "col_kept": {}, "remapped": {}}
    for name in ordered:
        si, dims, ttype, off = tensor_shard[name]
        size = _tensor_byte_size(dims, ttype)
        if name in keep:
            if len(dims) != 3:
                raise ValueError(f"keep entry {name!r} not 3-D: {dims}")
            E = dims[2]
            exp_size = size // E
            idx = keep[name]
            blob_offsets = [(si, off + e * exp_size, exp_size)
                            for e in idx]
            plan.append((name, (dims[0], dims[1], len(idx)), ttype,
                         ("stack", blob_offsets)))
            manifest["kept"][name.decode()] = idx
            manifest["dropped"][name.decode()] = E - len(idx)
        elif name in col_keep:
            cols = col_keep[name]
            with open(srcs[si], "rb") as s:
                s.seek(off)
                raw = s.read(size)
            if len(dims) == 2:
                n, m = dims
                W = np.frombuffer(raw, np.float32).reshape(n, m)
                data = np.ascontiguousarray(W[:, cols]).tobytes()
                new_dims = (n, len(cols))
            elif len(dims) == 1:
                W = np.frombuffer(raw, np.float32)
                data = np.ascontiguousarray(W[cols]).tobytes()
                new_dims = (len(cols),)
            else:
                raise ValueError(f"col_keep entry {name!r} bad dims {dims}")
            plan.append((name, new_dims, ttype, ("bytes", data)))
            manifest["col_kept"][name.decode()] = cols
        elif name in remap:
            mp = remap[name]
            with open(srcs[si], "rb") as s:
                s.seek(off)
                raw = s.read(size)
            vals = np.frombuffer(raw, np.int32).copy()
            vals2 = np.array([mp.get(int(v), 0) for v in vals],
                             np.int32)
            plan.append((name, dims, ttype, ("bytes", vals2.tobytes())))
            manifest["remapped"][name.decode()] = len(mp)
        else:
            plan.append((name, dims, ttype, ("copy", si, off, size)))

    new_kvs = [(k, t, kv_overrides[k] if k in kv_overrides else r)
               for k, t, r in headers[0][1]
               if k != MANIFEST_KEY.encode()]
    manifest_raw = json.dumps(manifest)
    new_kvs.append((MANIFEST_KEY.encode(), 8,
                    struct.pack("<Q", len(manifest_raw)) +
                    manifest_raw.encode()))

    hdr = (b"GGUF" + struct.pack("<I", version) +
           struct.pack("<Q", len(plan)) +
           struct.pack("<Q", len(new_kvs)))
    hdr += b"".join(_kv(k, t, r) for k, t, r in new_kvs)
    for name, dims, ttype, *_ in plan:
        hdr += struct.pack("<Q", len(name)) + name
        hdr += struct.pack("<I", len(dims))
        hdr += struct.pack("<" + "Q" * len(dims), *dims)
        hdr += struct.pack("<I", ttype)
        hdr += struct.pack("<Q", 0)

    hdr = bytearray(hdr)
    data_start = _align(len(hdr), alignment)
    pos = 4 + 4 + 8 + 8
    for k, t, r in new_kvs:
        pos += 8 + len(k) + 4 + len(r)
    offs = []
    for name, dims, ttype, *_ in plan:
        pos += 8 + len(name) + 4 + 8 * len(dims) + 4
        offs.append(pos)
        pos += 8
    rel = 0
    offsets = []
    for (name, dims, ttype, kind), offpos in zip(plan, offs):
        struct.pack_into("<Q", hdr, offpos, rel)
        offsets.append(rel)
        if kind[0] == "copy":
            rel += _align(kind[3], alignment)
        elif kind[0] == "stack":
            rel += _align(sum(b[2] for b in kind[1]), alignment)
        else:  # bytes
            rel += _align(len(kind[1]), alignment)
    if len(hdr) < data_start:
        hdr += b"\0" * (data_start - len(hdr))

    with open(out, "wb") as d:
        d.write(bytes(hdr))
        for (name, dims, ttype, kind), rel_off in zip(plan, offsets):
            if kind[0] == "copy":
                _, si, off, size = kind
                with open(srcs[si], "rb") as s:
                    s.seek(off)
                    _copy(s, d, size)
                d.write(b"\0" * (_align(size, alignment) - size))
            elif kind[0] == "stack":
                for si, off, size in kind[1]:
                    with open(srcs[si], "rb") as s:
                        s.seek(off)
                        _copy(s, d, size)
                total = sum(b[2] for b in kind[1])
                d.write(b"\0" * (_align(total, alignment) - total))
            else:
                data = kind[1]
                d.write(data)
                d.write(b"\0" * (_align(len(data), alignment) - len(data)))
    return Path(out)
