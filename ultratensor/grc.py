"""Streaming Geodesic Runtime Compression (GRC) for GGUF models.

HyperTensor Paper I (GRC) adapted to the UltraTensor streaming layer:

* Per-attention-tensor row-basis low-rank factorization
  ``W ~= U_k @ C`` with ``C = U_k^T @ W`` (rank k over the OUTPUT dim).
* The Gram matrix ``W @ W^T`` is accumulated incrementally over
  row-chunks via `dequantize_rows`, so RAM stays bounded by the Gram
  (rows x rows fp32) plus one row-chunk — never the full tensor.
* Optional sink-channel exemption: the top-T rows by norm are kept
  dense (Sun et al. 2024 style), the rest are factorized.
* Output: HyperRetro-style factored safetensors shards + manifest
  (``<name>.u`` fp16 basis, ``<name>.c`` uq4 codes + scales, optional
  ``<name>.sink`` dense rows).

Runtime form: ``y = (x @ U_k) @ C`` — the two halves are fused into one
pass by the HyperTensor kernels (Paper I kernel fusion).

Reconstruction is exact modulo quantization: ``W_hat = U_k @ C`` with
sink rows replaced by their dense copies.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np

from .dequant import dequantize_rows
from .quant import uq4_dequantize, uq4_quantize
from .stream import open_gguf, iter_tensors

ATTENTION_PATTERNS = (
    "attn_q_a.weight", "attn_q_b.weight", "attn_kv.weight",
    "attn_compressor_kv.weight", "attn_compressor_gate.weight",
    "attn_output_a.weight", "attn_output_b.weight",
)


def is_attention_tensor(name: str) -> bool:
    return any(name.endswith(p) for p in ATTENTION_PATTERNS)


def grc_compress_gguf(gguf_path: Path, out_dir: Path,
                      energy: float = 0.98, max_rank: Optional[int] = None,
                      sink_T: int = 0, only: Optional[set] = None,
                      max_tensors: Optional[int] = None,
                      quant_block: int = 128, progress: bool = True,
                      force_streaming: bool = False):
    """Streaming GRC on the attention tensors of a GGUF.

    Args:
        energy: kept-energy fraction used to pick the rank per tensor.
        max_rank: hard cap on the rank (None = no cap).
        sink_T: top-T rows by norm kept dense per tensor.
    """
    t_start = time.time()
    reader = open_gguf(gguf_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"format": "ultratensor-grc-v1", "energy": energy,
                "max_rank": max_rank, "sink_T": sink_T, "tensors": []}
    buf = {}          # payload name -> array
    buf_bytes = 0
    shard_idx = 0
    SHARD_BYTES = 4 * 1024 * 1024 * 1024

    def flush(force=False):
        nonlocal buf, buf_bytes, shard_idx
        if not buf or (not force and buf_bytes < SHARD_BYTES):
            return
        from safetensors.numpy import save_file
        path = out_dir / f"grc-{shard_idx:05d}.safetensors"
        save_file(buf, str(path))
        buf, buf_bytes = {}, 0
        shard_idx += 1

    def put(name, arr):
        nonlocal buf_bytes
        if buf_bytes + arr.nbytes > SHARD_BYTES and buf:
            flush()
        buf[name] = arr
        buf_bytes += arr.nbytes

    n_done = 0
    for tname, raw, qtype, shape in iter_tensors(reader):
        if only is not None and tname not in only:
            continue
        if not is_attention_tensor(tname):
            continue
        true_shape = tuple(reversed(shape))
        n = int(np.prod(true_shape))
        rows = true_shape[0] if len(true_shape) > 1 else 1
        cols = n // rows
        if min(rows, cols) < 64:
            continue
        t0 = time.time()

        # ---- SVD fast path: materialize (fits RAM for all V4 attention) ----
        if (not force_streaming) and rows * cols * 4 <= (1 << 31):
            W = np.empty((rows, cols), np.float64)
            chunk_rows = max(1, (1 << 26) // max(cols, 1))
            for r0 in range(0, rows, chunk_rows):
                r1 = min(rows, r0 + chunk_rows)
                W[r0:r1] = dequantize_rows(qtype, raw, shape, r0, r1 - r0)
            U, sv, Vt = np.linalg.svd(W.astype(np.float32),
                                     full_matrices=False)
            kept = np.cumsum(sv * sv) / np.sum(sv * sv)
            rank = int(np.searchsorted(kept, energy) + 1)
            rank = min(rank, rows, cols)
            if max_rank:
                rank = min(rank, max_rank)
            rank = max(1, rank)
            U_k = U[:, :rank].astype(np.float32)
            C = (sv[:rank, None] * Vt[:rank]).astype(np.float64)
        else:
            # ---- streaming Gram path for tensors too big for RAM ----
            gram = np.zeros((rows, rows), np.float64)
            chunk_rows = max(1, (1 << 26) // max(cols, 1))
            chunk_rows = min(max(chunk_rows, 8), rows)
            for r0 in range(0, rows, chunk_rows):
                r1 = min(rows, r0 + chunk_rows)
                Wc = dequantize_rows(qtype, raw, shape, r0, r1 - r0).astype(np.float64)
                gram += Wc @ Wc.T
            gram = 0.5 * (gram + gram.T)
            evals, evecs = np.linalg.eigh(gram / np.trace(gram))
            order = np.argsort(evals)[::-1]
            evals, evecs = evals[order], evecs[:, order]
            kept = np.cumsum(evals) / np.sum(evals)
            rank = int(np.searchsorted(kept, energy) + 1)
            rank = min(rank, rows, cols)
            if max_rank:
                rank = min(rank, max_rank)
            rank = max(1, rank)
            U_k = evecs[:, :rank].astype(np.float32)
            C = np.zeros((rank, cols), np.float64)
            for r0 in range(0, rows, chunk_rows):
                r1 = min(rows, r0 + chunk_rows)
                Wc = dequantize_rows(qtype, raw, shape, r0, r1 - r0).astype(np.float64)
                C += U_k.T.astype(np.float64) @ Wc

        factored_elems = rank * (rows + cols)
        dense_elems = rows * cols
        ratio = factored_elems / dense_elems
        if ratio >= 0.98 and sink_T == 0:
            if progress:
                print(f"[skip] {tname}: rank {rank}/{min(rows,cols)} keeps "
                      f"{kept[rank-1]:.4f} energy, ratio {ratio:.2f} - not worth factoring")
            continue

        # ---- sink exemption (top-T rows by norm, if requested) ----
        sink_rows = []
        if sink_T > 0:
            norms = np.zeros(rows, np.float64)
            chunk_rows = max(1, (1 << 26) // max(cols, 1))
            for r0 in range(0, rows, chunk_rows):
                r1 = min(rows, r0 + chunk_rows)
                Wc = dequantize_rows(qtype, raw, shape, r0, r1 - r0)
                norms[r0:r1] = (Wc * Wc).sum(axis=1)
            sink_rows = np.argsort(norms)[::-1][:sink_T].tolist()

        # quantize C (uq4 per block over cols)
        Cp = np.ascontiguousarray(C.T).astype(np.float32).reshape(1, cols, rank) \
            .transpose(0, 2, 1).reshape(rank, cols)
        sc, codes = uq4_quantize(Cp, block=quant_block)
        if sink_T > 0:
            S = np.zeros((sink_T, cols), np.float32)
            for i, r in enumerate(sink_rows):
                S[i] = dequantize_rows(qtype, raw, shape, r, 1)[0].astype(np.float32)
            sc_s, codes_s = uq4_quantize(S, block=quant_block)
        # ---- store ----
        put(f"{tname}.u", U_k.astype(np.float16))
        put(f"{tname}.c_scales", sc.astype(np.float32))
        put(f"{tname}.c_codes", codes)
        entry = {"name": tname, "kind": "grc-row-basis", "source_quant": qtype,
                 "shape": list(true_shape), "rows": rows, "cols": cols,
                 "rank": rank, "energy_kept": float(kept[rank - 1]),
                 "size_ratio": float(ratio), "quant_block": quant_block}
        if sink_T > 0:
            put(f"{tname}.sink_scales", sc_s.astype(np.float32))
            put(f"{tname}.sink_codes", codes_s)
            entry["sink_rows"] = sink_rows
        manifest["tensors"].append(entry)
        n_done += 1
        if progress:
            el = time.time() - t_start
            print(f"[{n_done}] {tname} {rows}x{cols} rank={rank}/{min(rows,cols)} "
                  f"energy={kept[rank-1]:.4f} size={ratio:.2f}x "
                  f"{time.time()-t0:.0f}s | total {el:.0f}s")
        if max_tensors and n_done >= max_tensors:
            break
    flush(force=True)
    (out_dir / "ultratensor_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def reconstruct(entry: dict, sd: dict) -> np.ndarray:
    """Rebuild the full tensor from the factored store."""
    n = entry["name"]
    U = sd[f"{n}.u"].astype(np.float32)
    C = uq4_dequantize(sd[f"{n}.c_scales"], sd[f"{n}.c_codes"])
    W = U @ C
    if "sink_rows" in entry:
        S = uq4_dequantize(sd[f"{n}.sink_scales"], sd[f"{n}.sink_codes"])
        for i, r in enumerate(entry["sink_rows"]):
            W[r] = S[i]
    return W
