"""UltraTensor streaming compressor.

Processes a GGUF (including multi-shard) tensor-by-tensor so that models
of any size can be compressed on RAM-limited machines. This is the
streaming layer HyperRetro lacked: instead of holding the whole model in
fp16, one tensor is dequantized, re-quantized, and written out at a time.

Output format: a directory with `ultratensor_manifest.json` plus
safetensors shards (<= 4 GB each) named `model-0000N.safetensors`.
Each quantized tensor is stored as `<name>.codes` / `<name>.scales`
exactly like the HyperRetro factored layout, so hyperretro-style
loading is possible. uq4 is symmetric per-block int4 (exact zeros).
"""

from __future__ import annotations

import json
import struct
import time
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

from .dequant import BLOCK_ALIGN, dequantize
from .quant import q2_0_quantize, q4_0_quantize, q8_0_quantize, uq4_quantize

SHARD_BYTES = 4 * 1024 * 1024 * 1024  # 4 GB

# GGUF value types (header parsing)
_GGUF_VALUE_SIZE = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1,
                    10: 8, 11: 8, 12: 8}


def _quant_type_from_id(tid: int) -> str:
    try:
        from gguf.constants import GGMLQuantizationType
        return GGMLQuantizationType(tid).name
    except Exception:
        return f"TYPE{tid}"


def tensor_inventory(path: Path) -> list:
    """Parse only the GGUF header: [(name, quant_name, dims, offset)].

    Works on INCOMPLETE downloads (no tensor data is touched), so a
    partially downloaded shard can be inventoried immediately.
    """
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"GGUF":
            raise ValueError(f"not a GGUF file: {path}")
        _ver, n_tensors, n_kv = struct.unpack("<IQQ", f.read(4 + 8 + 8))
        for _ in range(n_kv):
            klen = struct.unpack("<Q", f.read(8))[0]
            f.read(klen)
            vtype = struct.unpack("<I", f.read(4))[0]
            if vtype == 8:  # string
                slen = struct.unpack("<Q", f.read(8))[0]
                f.read(slen)
            elif vtype == 9:  # array
                etype = struct.unpack("<I", f.read(4))[0]
                n = struct.unpack("<Q", f.read(8))[0]
                if etype == 8:  # array of strings (e.g. tokenizer tokens)
                    for _ in range(n):
                        slen = struct.unpack("<Q", f.read(8))[0]
                        f.read(slen)
                else:
                    f.read(n * _GGUF_VALUE_SIZE.get(etype, 8))
            else:
                f.read(_GGUF_VALUE_SIZE.get(vtype, 8))
        out = []
        for _ in range(n_tensors):
            nlen = struct.unpack("<Q", f.read(8))[0]
            name = f.read(nlen).decode("utf-8", errors="replace")
            ndim = struct.unpack("<I", f.read(4))[0]
            dims = struct.unpack(f"<{ndim}Q", f.read(8 * ndim))
            ttype = struct.unpack("<I", f.read(4))[0]
            off = struct.unpack("<Q", f.read(8))[0]
            out.append((name, _quant_type_from_id(ttype), dims, off))
    return out


# ---------------------------------------------------------------------------
# GGUF reading (streaming)
# ---------------------------------------------------------------------------

def open_gguf(path: Path):
    """Open a GGUF file (single or first shard) via the gguf package."""
    from gguf import GGUFReader

    reader = GGUFReader(str(path))
    return reader


def tensor_logical_shape(tensor) -> tuple:
    raw = tensor.shape
    if raw is None:
        return ()
    if hasattr(raw, "tolist"):
        return tuple(int(s) for s in raw.tolist())
    return tuple(int(s) for s in list(raw))


def quant_type_name(tensor) -> str:
    tt = tensor.tensor_type
    try:
        return tt.name
    except Exception:
        return str(tt)


def iter_tensors(reader):
    """Yield (name, raw_data, quant_name, logical_shape)."""
    for t in reader.tensors:
        name = t.name
        qtype = quant_type_name(t)
        yield name, t.data, qtype, tensor_logical_shape(t)


# ---------------------------------------------------------------------------
# Compression core
# ---------------------------------------------------------------------------

class TensorCompressor:
    """Dequantize -> requantize -> emit one tensor at a time."""

    def __init__(self, target: str = "uq4", block: int = 128):
        self.target = target
        self.block = block

    def compress(self, name: str, raw: np.ndarray, qtype: str,
                 shape: tuple) -> dict:
        """Dequantize -> requantize one tensor, ROW BY ROW.

        Only a single row is ever materialized in fp32, so tensors with
        billions of elements (e.g. MXFP4 expert tensors) compress with
        bounded memory.
        """
        from .dequant import dequantize_rows

        true_shape = tuple(reversed(shape))
        n = int(np.prod(true_shape))
        rows = true_shape[0] if len(true_shape) > 1 else 1
        cols = n // rows

        out = {"name": name, "source_quant": qtype, "shape": list(true_shape),
               "target": self.target, "rows": rows, "cols": cols}

        if self.target == "q8_0":
            cols_pad = ((cols + 31) // 32) * 32
            scales = np.empty((rows, cols_pad // 32), np.float32)
            codes = np.empty((rows, cols_pad), np.int8)
            if cols_pad != cols:
                out["cols_padded"] = cols_pad
            for r in range(rows):
                Wr = dequantize_rows(qtype, raw, shape, r, 1).reshape(-1)
                if cols_pad != cols:
                    Wp = np.zeros(cols_pad, np.float32)
                    Wp[:cols] = Wr
                    Wr = Wp
                sc, co = q8_0_quantize(Wr.reshape(1, -1))
                scales[r] = sc[0]
                codes[r] = co.reshape(-1)
            out["scales"] = scales
            out["codes"] = codes
        elif self.target == "q4_0":
            cols_pad = ((cols + 31) // 32) * 32
            scales = np.empty((rows, cols_pad // 32), np.float32)
            codes = np.empty((rows, cols_pad // 2), np.uint8)
            if cols_pad != cols:
                out["cols_padded"] = cols_pad
            for r in range(rows):
                Wr = dequantize_rows(qtype, raw, shape, r, 1).reshape(-1)
                if cols_pad != cols:
                    Wp = np.zeros(cols_pad, np.float32)
                    Wp[:cols] = Wr
                    Wr = Wp
                sc, q = q4_0_quantize(Wr.reshape(1, -1))
                scales[r] = sc[0]
                q = q.reshape(cols_pad // 32, 32)
                codes[r] = (q[:, :16] | (q[:, 16:] << 4)).reshape(-1)
            out["scales"] = scales
            out["codes"] = codes
        elif self.target == "q2_0":
            blk = 32
            cols_pad = ((cols + blk - 1) // blk) * blk
            scales = np.empty((rows, cols_pad // blk), np.float16)
            codes = np.empty((rows, cols_pad // 4), np.uint8)
            if cols_pad != cols:
                out["cols_padded"] = cols_pad
            for r in range(rows):
                Wr = dequantize_rows(qtype, raw, shape, r, 1).reshape(-1)
                if cols_pad != cols:
                    Wp = np.zeros(cols_pad, np.float32)
                    Wp[:cols] = Wr
                    Wr = Wp
                sc, packed = q2_0_quantize(Wr.reshape(1, -1))
                scales[r] = sc[0]
                codes[r] = packed[0]
            out["scales"] = scales
            out["codes"] = codes
        elif self.target == "uq4":
            blk = self.block
            cols_pad = ((cols + blk - 1) // blk) * blk
            scales = np.empty((rows, cols_pad // blk), np.float32)
            codes = np.empty((rows, cols_pad // 2), np.uint8)
            if cols_pad != cols:
                out["cols_padded"] = cols_pad
            for r in range(rows):
                Wr = dequantize_rows(qtype, raw, shape, r, 1).reshape(-1)
                if cols_pad != cols:
                    Wp = np.zeros(cols_pad, np.float32)
                    Wp[:cols] = Wr
                    Wr = Wp
                sc, packed = uq4_quantize(Wr.reshape(1, -1), block=blk)
                scales[r] = sc[0]
                codes[r] = packed[0]
            out["scales"] = scales
            out["codes"] = codes
        else:
            raise ValueError(f"unknown target: {self.target}")
        return out


# ---------------------------------------------------------------------------
# Safetensors sharded writer
# ---------------------------------------------------------------------------

class ShardedWriter:
    def __init__(self, out_dir: Path):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.manifest = {
            "format": "ultratensor-v1",
            "tensors": [],
            "shards": [],
        }
        self._buf = {}
        self._buf_bytes = 0
        self._shard_idx = 0
        self._current = {}

    def _flush(self):
        if not self._buf:
            return
        from safetensors.numpy import save_file

        path = self.out_dir / f"model-{self._shard_idx:05d}.safetensors"
        save_file({k: v for k, v in self._buf.items()}, str(path))
        self.manifest["shards"].append(path.name)
        self._shard_idx += 1
        self._buf = {}
        self._buf_bytes = 0

    def add(self, entry: dict):
        name = entry["name"]
        payload = {}
        if "mins" in entry:
            payload[f"{name}.mins"] = entry["mins"]
        payload[f"{name}.scales"] = entry["scales"]
        payload[f"{name}.codes"] = entry["codes"]
        for k, v in payload.items():
            nbytes = int(v.nbytes)
            if self._buf_bytes + nbytes > SHARD_BYTES and self._buf:
                self._flush()
            self._buf[k] = v
            self._buf_bytes += nbytes
        meta = {k: v for k, v in entry.items()
                if k not in ("scales", "codes", "mins")}
        self.manifest["tensors"].append(meta)

    def finish(self, manifest_name: str = "ultratensor_manifest.json"):
        self._flush()
        (self.out_dir / manifest_name).write_text(
            json.dumps(self.manifest, indent=2), encoding="utf-8"
        )


# ---------------------------------------------------------------------------
# Top-level API
# ---------------------------------------------------------------------------

def _out_bytes_per_elem(target: str, block: int) -> float:
    """Estimated output bytes per element for the chosen target."""
    if target == "q8_0":
        return 32 / 32 + 4 / 32          # 8-bit codes + fp32 scale/32
    if target == "q4_0":
        return 16 / 32 + 4 / 32          # 4-bit codes + fp32 scale/32
    if target == "q2_0":
        return 8 / 32 + 2 / 32           # 2-bit codes + fp16 scale/32
    return 0.5 + 4.0 / block             # uq4 codes + fp32 scale per block


def dry_run(gguf_path: Path, target: str = "uq4", block: int = 128,
            max_tensors: Optional[int] = None, quiet: bool = False,
            only: Optional[set] = None):
    """Report per-tensor compressibility without writing anything."""
    reader = open_gguf(gguf_path)
    comp = TensorCompressor(target=target, block=block)
    rows_out = []
    total_src_bytes = 0
    total_out_est = 0.0
    bpe = _out_bytes_per_elem(target, block)
    for name, raw, qtype, shape in iter_tensors(reader):
        if only is not None and name not in only:
            continue
        if qtype not in BLOCK_ALIGN:
            rows_out.append((name, qtype, shape, "SKIP-UNSUPPORTED"))
            continue
        src_bytes = int(raw.nbytes)
        n = int(np.prod(shape))
        out_est = n * bpe
        total_src_bytes += src_bytes
        total_out_est += out_est
        rows_out.append((name, qtype, shape, src_bytes / 1e6, out_est / 1e6))
        if max_tensors and len(rows_out) >= max_tensors:
            break
    if not quiet:
        for r in rows_out:
            print(f"  {r[0]:<55s} {r[1]:<8s} {str(r[2]):<20s} "
                  + ("" if len(r) < 5 else f"{r[3]:9.1f}MB -> ~{r[4]:9.1f}MB"))
    print(f"\nTOTAL: {total_src_bytes/1e9:.2f} GB -> est {total_out_est/1e9:.2f} GB "
          f"(~{total_out_est/max(total_src_bytes,1)*100:.0f}%)")
    return rows_out


def compress_gguf(gguf_path: Path, out_dir: Path, target: str = "uq4",
                  block: int = 128, max_tensors: Optional[int] = None,
                  progress_every: int = 10, only: Optional[set] = None,
                  manifest_name: str = "ultratensor_manifest.json"):
    """Streaming compress: one tensor row at a time in RAM.

    manifest_name: name of the manifest inside out_dir; use a unique name
    per shard when compressing a split model into one directory.
    """
    t_start = time.time()
    reader = open_gguf(gguf_path)
    comp = TensorCompressor(target=target, block=block)
    writer = ShardedWriter(out_dir)
    n = 0
    skipped = 0
    src_bytes = 0
    for name, raw, qtype, shape in iter_tensors(reader):
        if only is not None and name not in only:
            continue
        if qtype not in BLOCK_ALIGN:
            print(f"[skip] {name}: unsupported source quant {qtype}")
            skipped += 1
            continue
        src_bytes += int(raw.nbytes)
        print(f"[start] {name} ({qtype} {tuple(shape)})...")
        entry = comp.compress(name, raw, qtype, shape)
        writer.add(entry)
        n += 1
        el = time.time() - t_start
        print(f"[done ] {name} | {el:.0f}s total")
        if n % progress_every == 0:
            print(f"[{n} tensors | {src_bytes/1e9:.1f} GB read | {el:.0f}s]")
        if max_tensors and n >= max_tensors:
            break
    writer.finish(manifest_name=manifest_name)
    out_bytes = sum(f.stat().st_size for f in Path(out_dir).glob("*.safetensors"))
    print(f"\nDONE: {n} tensors compressed, {skipped} skipped")
    print(f"  source data: {src_bytes/1e9:.2f} GB -> output: {out_bytes/1e9:.2f} GB")
    print(f"  wall time: {time.time()-t_start:.0f}s")
    return writer.manifest
