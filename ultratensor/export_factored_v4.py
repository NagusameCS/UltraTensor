"""Streaming, resumable V4-Pro expert factorization -> factored GGUF shards.

This is the overnight job behind "usable V4-Pro": it converts each Q3_K_M
shard into the Phase-1 factored container, expert stack by expert stack
(streaming; RAM stays bounded by a few experts), with a state file so an
interrupted run resumes without re-factoring finished tensors.

Usage:
    python -m ultratensor export-factored-v4 src.gguf --out out.gguf \
        --rank 128 [--batch 4] [--device cuda|cpu] [--only name-substr] \
        [--limit-experts N]

Output: <out>.gguf (factored container), <out>.tmp (tensor payloads),
<out>.state.json (resume state). Delete state+tmp to restart a shard.
"""
from __future__ import annotations

import argparse
import json
import struct
import time
from pathlib import Path

import numpy as np

from .gguf_factored import (_align, _kv, MANIFEST_KEY, GGML_TYPE_F16,
                            GGML_TYPE_FACTORED_C, read_gguf_header)
from .quant import uq4_quantize

# gguf quant type id -> (name, bytes per 256-element block)
_QTYPE = {10: ("Q2_K", 84), 11: ("Q3_K", 110), 12: ("Q4_K", 144),
          13: ("Q5_K", 176), 14: ("Q6_K", 210)}

_EXPERT_SUFFIXES = (b"ffn_gate_exps", b"ffn_down_exps", b"ffn_up_exps")


def _expert_bytes(dims, ttype):
    n, m = int(dims[0]), int(dims[1])
    per_block = _QTYPE[ttype][1]
    return m * (n // 256) * per_block


def _factor_batch(dev, Ws, rank):
    """Ws: list of fp32 numpy (m, n). Returns lists of U fp16 and C fp32."""
    import torch
    Wt = torch.from_numpy(np.stack([w.astype(np.float32) for w in Ws])).to(dev)
    U, S, Vt = torch.linalg.svd(Wt, full_matrices=False)
    U = U[:, :, :rank]
    C = (S[:, :rank, None] * Vt[:, :rank, :])
    Us = U.cpu().numpy().astype(np.float16)
    Cs = C.cpu().numpy().astype(np.float32)
    errs = (S[:, rank:].norm(dim=1) / S.norm(dim=1)).cpu().numpy().tolist()
    del Wt, U, S, Vt, C
    if dev != "cpu":
        torch.cuda.empty_cache()
    return list(Us), list(Cs), errs


def _encode_expert(C):
    scales, packed = uq4_quantize(C, block=32)
    return scales.astype(np.float32), packed.astype(np.uint8)


def convert_shard(src: str | Path, out: str | Path, rank: int = 128,
                  batch: int = 4, device: str = "cuda",
                  only: str | None = None,
                  limit_experts: int | None = None):
    src, out = Path(src), Path(out)
    version, kvs, infos, hdr_end = read_gguf_header(src)
    alignment = 32
    for k, t, raw in kvs:
        if k == b"general.alignment":
            alignment = struct.unpack("<I", raw)[0]
    data_start = _align(hdr_end, alignment)

    state_path = Path(str(out) + ".state.json")
    tmp_path = Path(str(out) + ".tmp")
    state = {"done": [], "tmp_bytes": 0, "rank": rank}
    if state_path.exists():
        state = json.loads(state_path.read_text())
        assert state.get("rank") == rank, "rank changed; delete state to restart"
        assert state.get("only") == only, "--only changed; delete state to restart"
        assert state.get("limit_experts") == limit_experts, \
            "--limit-experts changed; delete state to restart"
    state["only"] = only
    state["limit_experts"] = limit_experts

    from .dequant import dequantize_rows

    with open(src, "rb") as fsrc, open(tmp_path, "ab") as ftmp:
        n_total = len(infos)
        for idx, (name, dims, ttype, off) in enumerate(infos):
            tname = name.decode()
            if tname in state["done"]:
                continue
            t0 = time.time()
            is_expert = (len(dims) == 3 and
                         any(s in name for s in _EXPERT_SUFFIXES))
            if only is not None and only not in tname:
                continue  # --only: skip non-matching tensors entirely
            if is_expert and ttype in _QTYPE:
                n, m, E = int(dims[0]), int(dims[1]), int(dims[2])
                if limit_experts is not None:
                    E = min(E, limit_experts)
                ebytes = _expert_bytes(dims, ttype)
                qname = _QTYPE[ttype][0]
                Us, Cs, errs = [], [], []
                for e0 in range(0, E, batch):
                    Ws = []
                    for e in range(e0, min(e0 + batch, E)):
                        fsrc.seek(data_start + off + e * ebytes)
                        data = fsrc.read(ebytes)
                        W = dequantize_rows(
                            qname, np.frombuffer(data, np.uint8),
                            (n, m), 0, m)
                        Ws.append(W)
                    u, c, er = _factor_batch(device, Ws, rank)
                    Us.extend(u)
                    Cs.extend(c)
                    errs.extend(er)
                for e in range(E):
                    ftmp.write(np.ascontiguousarray(Us[e]).view(np.uint8)
                               .tobytes())
                enc = [_encode_expert(Cs[e]) for e in range(E)]
                for sc, _pk in enc:
                    ftmp.write(np.ascontiguousarray(sc).view(np.uint8)
                               .tobytes())
                for _sc, pk in enc:
                    ftmp.write(pk.tobytes())
                state["tmp_bytes"] += E * (m * rank * 2 +
                                           rank * (n // 32) * 4 +
                                           rank * (n // 2))
                print(f"[{idx+1}/{n_total}] {tname} E={E} k={rank} "
                      f"err={np.mean(errs):.4f} "
                      f"in {time.time()-t0:.1f}s", flush=True)
            else:
                size = _source_size(dims, ttype)
                fsrc.seek(data_start + off)
                _copy_n(fsrc, ftmp, size)
                state["tmp_bytes"] += size
            state["done"].append(tname)
            state_path.write_text(json.dumps(state))
            ftmp.flush()

    # finalize: header + tmp data
    finalize(src, out, tmp_path, state, alignment, rank, only, limit_experts)


def _source_size(dims, ttype):
    n = int(np.prod(dims)) if dims else 0
    if ttype in _QTYPE:
        return n // 256 * _QTYPE[ttype][1]
    if ttype in (0, 32):  # F32
        return n * 4
    if ttype == 1:  # F16
        return n * 2
    if ttype == 26:  # I32
        return n * 4
    if ttype in (2, 3):  # Q4_0/Q4_1
        return n // 32 * 18
    if ttype in (8, 9):  # Q8_0/Q8_1
        return n // 32 * 34
    if ttype == 7:  # Q8_0 alt id? (older gguf)
        return n // 32 * 34
    raise NotImplementedError(f"source type {ttype}")


def finalize(src, out, tmp_path, state, alignment, rank, only=None,
             limit_experts=None):
    """Build the factored container from the source header + temp payloads."""
    limit_experts = limit_experts if limit_experts is not None \
        else state.get("limit_experts")
    version, kvs, infos, hdr_end = read_gguf_header(src)
    names = {MANIFEST_KEY.encode()}
    new_kvs = [kv for kv in kvs if kv[0] not in names]

    manifest = {"version": 1, "rank": rank, "tensors": []}
    infos_out = []  # (name, dims, ttype, size)
    for name, dims, ttype, off in infos:
        if only is not None and only not in name.decode():
            continue
        is_expert = (len(dims) == 3 and
                     any(s in name for s in _EXPERT_SUFFIXES) and
                     ttype in _QTYPE)
        if is_expert:
            n, m, E = int(dims[0]), int(dims[1]), int(dims[2])
            if limit_experts is not None:
                E = min(E, limit_experts)
            u_size = E * m * rank * 2
            c_size = E * rank * (4 * (n // 32) + n // 2)
            infos_out.append((name + b".factored_U", (rank, m, E),
                              GGML_TYPE_F16, u_size))
            infos_out.append((name + b".factored_C", (n, rank, E),
                              GGML_TYPE_FACTORED_C, c_size))
            manifest["tensors"].append({"name": name.decode(),
                                        "shape": [E, m, n], "rank": rank})
        else:
            size = _source_size(dims, ttype)
            infos_out.append((name, dims, ttype, size))

    manifest_raw = json.dumps(manifest)
    new_kvs.append((MANIFEST_KEY.encode(), 8,
                    struct.pack("<Q", len(manifest_raw)) +
                    manifest_raw.encode()))

    hdr = (b"GGUF" + struct.pack("<I", version) +
           struct.pack("<Q", len(infos_out)) +
           struct.pack("<Q", len(new_kvs)))
    hdr += b"".join(_kv(k, t, r) for k, t, r in new_kvs)
    for name, dims, ttype, size in infos_out:
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
    rel = 0
    for name, dims, ttype, size in infos_out:
        pos += 8 + len(name) + 4 + 8 * len(dims) + 4
        struct.pack_into("<Q", hdr, pos, rel)
        pos += 8
        rel += _align(size, alignment)
    if len(hdr) < data_start:
        hdr += b"\0" * (data_start - len(hdr))

    with open(out, "wb") as d, open(tmp_path, "rb") as s:
        d.write(bytes(hdr))
        for name, dims, ttype, size in infos_out:
            _copy_n(s, d, size)
            pad = _align(size, alignment) - size
            if pad:
                d.write(b"\0" * pad)
    # cleanup resume files on success
    tmp_path.unlink(missing_ok=True)
    state_path = Path(str(out) + ".state.json")
    state_path.unlink(missing_ok=True)
    print(f"finalized {out}", flush=True)


def _copy_n(fsrc, fdst, size, buf=64 * 1024 * 1024):
    while size > 0:
        chunk = fsrc.read(min(buf, size))
        if not chunk:
            raise EOFError("source truncated")
        fdst.write(chunk)
        size -= len(chunk)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="export-factored-v4",
                                 description="factor V4-Pro expert shards")
    ap.add_argument("src")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rank", type=int, default=128)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--only", default=None)
    ap.add_argument("--limit-experts", type=int, default=None)
    args = ap.parse_args(argv)
    convert_shard(args.src, args.out, rank=args.rank, batch=args.batch,
                  device=args.device, only=args.only,
                  limit_experts=args.limit_experts)


if __name__ == "__main__":
    main()
