"""ExpertStore: dispatch-aware reading of MoE tensors from a GGUF shard.

Phase 3b of the llama e2e roadmap. llama.cpp reads every expert tensor
fully per forward pass; a lazy top-k executor needs per-expert, streaming
reads plus a routing pass. This module provides:

- ExpertStore(shard): header-only open; locate expert + router tensors
- read_expert(layer, kind, e)  -> (m, n) fp32  (one expert, bounded RAM)
- route_layer(layer, hidden, token_ids) -> top-k ids (dense router for
  layers >= n_hash_layers, hash table below)
- io_model(...) -> bytes/token floor for a given serving strategy

Verified V4-Pro facts (deepseek-ai/DeepSeek-V4-Pro + bati.cpp):
  * routing: layers 0..2 hash (ffn_gate_tid2eid, zero GEMV);
    layers 3+ dense router ffn_gate_inp (7168,384) F32 + exp_probs_b,
    sqrt-softplus gating, top-6, route_scale 2.5.
  * ffn_gate_exps is the per-expert SwiGLU gate (w1) — read ONLY for
    the top-6 selected experts, like up/down. It is NOT the router.
  * per-token expert IO: 6 x (gate 9.46 + up 9.46 + down 15.1 MB)
    = 204 MB/layer + shared expert ~34 MB -> ~14.5 GB/token over 61
    layers (~0.14 tok/s at 2 GB/s). Router is ~641 MB total: resident.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .dequant import BLOCK_ALIGN, BLOCK_BYTES, dequantize_rows
from .gguf_factored import _align, read_gguf_header

_EXPERT_KINDS = ("ffn_gate_exps", "ffn_down_exps", "ffn_up_exps")
_SHEXP_KINDS = ("ffn_gate_shexp", "ffn_down_shexp", "ffn_up_shexp")
_QTYPE = {8: "Q8_0", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K",
          13: "Q5_K", 14: "Q6_K"}


def _name(layer: int, kind: str) -> str:
    if kind == "exp_probs_b":
        return f"blk.{layer}.exp_probs_b.bias"
    return f"blk.{layer}.{kind}.weight"


class ExpertStore:
    def __init__(self, shard: str | Path, extra_shards=()):
        """Header-only open of one or more GGUF shards of the same model.

        Tensor inventories are merged across shards; reads route to the
        shard that holds the tensor. extra_shards: additional shard paths
        (e.g. the remaining N-of-17 files).
        """
        self.shards = [Path(shard)] + [Path(s) for s in extra_shards]
        self.shard = self.shards[0]   # backward-compatible attribute
        self.version = 0
        self.kvs = []
        self.n_hash_layers = 0
        self.alignment = 32
        self._data_starts = []
        interesting = _EXPERT_KINDS + _SHEXP_KINDS + (
            "ffn_gate_inp", "exp_probs_b", "ffn_gate_tid2eid")
        self.tensors = {}
        for si, sh in enumerate(self.shards):
            version, kvs, infos, hdr_end = read_gguf_header(sh)
            self.version = version
            if si == 0:
                self.kvs = kvs
                for k, t, raw in kvs:
                    if k == b"general.alignment":
                        self.alignment = int(np.frombuffer(raw, np.uint32)[0])
                    elif k == b"deepseek4.hash_layer_count":
                        self.n_hash_layers = int(np.frombuffer(raw,
                                                               np.uint32)[0])
            ds = _align(hdr_end, self.alignment)
            self._data_starts.append(ds)
            for name, dims, ttype, off in infos:
                tname = name.decode()
                is_blk = tname.startswith("blk.")
                is_expert3d = (len(dims) == 3 and ttype in _QTYPE and
                               any(k in tname for k in _EXPERT_KINDS))
                is_small = any(k in tname for k in interesting)
                if is_blk and (is_expert3d or is_small or len(dims) <= 2):
                    self.tensors[tname] = {
                        "name": tname, "dims": tuple(int(d) for d in dims),
                        "ttype": ttype, "off": off, "shard": si,
                    }
        self._cache = {}
        self.data_start = self._data_starts[0]  # first shard (compat)

    def _open_tensor(self, name: str):
        """-> (open file, absolute offset) for a tensor by name."""
        t = self.tensors[name]
        f = open(self.shards[t["shard"]], "rb")
        return f, self._data_starts[t["shard"]] + t["off"]

    # -- inventory --------------------------------------------------------
    def layers(self) -> list[int]:
        layers = set()
        for t in self.tensors:
            # blk.<L>.ffn_...
            try:
                layers.add(int(t.split(".")[1]))
            except (IndexError, ValueError):
                pass
        return sorted(layers)

    def _tensor(self, layer: int, kind: str):
        """Inventory entry for blk.<layer>.<kind>.weight (or None)."""
        return self.tensors.get(_name(layer, kind))

    def _tensor_bytes(self, layer: int, kind: str) -> int:
        t = self._tensor(layer, kind)
        if t is None:
            return 0
        dims = t["dims"]
        n_elems = int(np.prod(dims))
        ttype = t["ttype"]
        if ttype in _QTYPE:
            qname = _QTYPE[ttype]
            return (n_elems // BLOCK_ALIGN[qname]) * BLOCK_BYTES[qname]
        if ttype in (0, 26):          # F32 / I32
            return n_elems * 4
        if ttype == 1:                # F16
            return n_elems * 2
        raise NotImplementedError(f"type {ttype} ({kind})")

    def read_tensor(self, layer: int, kind: str) -> np.ndarray:
        """Load a small dense tensor (router, bias, shared expert).

        Returns in numpy row-major with the GGUF dims reversed, i.e.
        dims (n, m) -> array [m, n]. Cached per store instance.
        """
        key = (layer, kind)
        if key in self._cache:
            return self._cache[key]
        t = self._tensor(layer, kind)
        if t is None:
            raise KeyError(_name(layer, kind))
        dims = t["dims"]
        n_elems = int(np.prod(dims))
        ttype = t["ttype"]
        f, off = self._open_tensor(t["name"])
        with f:
            f.seek(off)
            if ttype in (0, 26):
                raw = f.read(n_elems * 4)
                arr = np.frombuffer(raw, np.uint32 if ttype == 26
                                    else np.float32)
            elif ttype == 1:
                raw = f.read(n_elems * 2)
                arr = np.frombuffer(raw, np.float16).astype(np.float32)
            elif ttype in _QTYPE:
                qname = _QTYPE[ttype]
                rows = int(np.prod(dims[1:])) if len(dims) > 1 else 1
                cols = int(dims[0])
                nbytes = self._tensor_bytes(layer, kind)
                arr = dequantize_rows(qname,
                                      np.frombuffer(f.read(nbytes), np.uint8),
                                      (cols, rows), 0, rows)
                self._cache[key] = arr
                return arr
            else:
                raise NotImplementedError(f"type {ttype} ({kind})")
        arr = arr.reshape(tuple(int(d) for d in reversed(dims)))
        self._cache[key] = arr
        return arr

    def expert_shape(self, layer: int, kind: str) -> tuple[int, int, int]:
        """-> (n, m, E) for blk.<layer>.ffn_<kind>_exps.weight."""
        name = _name(layer, kind)
        if name not in self.tensors:
            raise KeyError(name)
        n, m, E = self.tensors[name]["dims"]
        return int(n), int(m), int(E)

    def _expert_bytes(self, layer: int, kind: str) -> int:
        n, m, E = self.expert_shape(layer, kind)
        t = self.tensors[_name(layer, kind)]
        qname = _QTYPE[t["ttype"]]
        return m * (n // BLOCK_ALIGN[qname]) * BLOCK_BYTES[qname]

    # -- expert reads -----------------------------------------------------
    def read_expert(self, layer: int, kind: str, e: int) -> np.ndarray:
        """Dequantize expert e of blk.<layer>.ffn_<kind>_exps -> (m, n)."""
        name = _name(layer, kind)
        t = self.tensors[name]
        n, m, E = self.expert_shape(layer, kind)
        if not (0 <= e < E):
            raise IndexError(f"expert {e} out of range 0..{E - 1}")
        ebytes = self._expert_bytes(layer, kind)
        qname = _QTYPE[t["ttype"]]
        f, off = self._open_tensor(name)
        with f:
            f.seek(off + e * ebytes)
            data = np.frombuffer(f.read(ebytes), np.uint8)
        return dequantize_rows(qname, data, (n, m), 0, m)

    def route_layer(self, layer: int, hidden: np.ndarray,
                    token_ids: np.ndarray | None = None,
                    top_k: int = 6) -> np.ndarray:
        """Top-k expert ids for one token, per the real V4 router.

        layer < n_hash_layers: deterministic hash table (token_id -> ids);
        token_ids must be given (one per hidden row).
        layer >= n_hash_layers: scores = sqrt(softplus(gate_inp @ h)) + bias,
        then top-k (matching inference/model.py Gate.forward).
        hidden: [n] fp32 (or [B, n]; returns [B, top_k]).
        """
        h = np.atleast_2d(hidden.astype(np.float32))
        B = h.shape[0]
        if layer < self.n_hash_layers:
            tid = self._tensor(layer, "ffn_gate_tid2eid")
            if token_ids is None:
                raise ValueError("token_ids required for hash-routed layers")
            token_ids = np.atleast_1d(np.asarray(token_ids))
            table = self.read_tensor(layer, "ffn_gate_tid2eid")
            ids = np.stack([table[t] for t in token_ids[:B]])
            return ids[:, :top_k]
        n, m = self.router_shape(layer)
        if h.shape[1] != n:
            raise ValueError(f"hidden must be ({n},) per token")
        W = self.read_tensor(layer, "ffn_gate_inp")        # [m, n]
        bias_t = self._tensor(layer, "exp_probs_b")
        bias = self.read_tensor(layer, "exp_probs_b") if bias_t is not None else 0.0
        z = h @ W.T
        s = np.sqrt(np.log1p(np.exp(z)))                   # sqrt(softplus)
        # reference (inference/model.py): the bias shifts the SELECTION
        # scores AFTER sqrt(softplus) - NOT inside the softplus.
        sel = s + bias
        # argpartition kth must be top_k - 1: positions 0..kth are <= the
        # element at kth, so [:, :top_k] after kth=top_k can displace the
        # true k-th expert with the (k+1)-th.
        kth = min(max(top_k - 1, 0), max(m - 1, 0))
        return np.argpartition(-sel, kth, axis=-1)[:, :top_k]

    def router_shape(self, layer: int) -> tuple[int, int]:
        """-> (n_hidden, n_experts) of the dense router (layers >= hash)."""
        t = self._tensor(layer, "ffn_gate_inp")
        return int(t["dims"][0]), int(t["dims"][1])

    # -- IO model ---------------------------------------------------------
    def io_model(self, top_k: int = 6, include_shexp: bool = True,
                 router_amortized: bool = True,
                 include_dense: bool = True) -> dict:
        """Bytes touched per token for a serving strategy.

        top_k            : routed experts per token (V4: 6)
        include_shexp    : shared expert (dense, read every token)
        router_amortized : router tables stay resident after first read
                           (641 MB total) -> not counted per token
        include_dense    : attention/compressor/indexer/norm tensors
                           (~126 MiB/layer on V4-Pro; read every token)
        """
        total = 0.0
        per_layer = {}
        router_total = 0.0
        for layer in self.layers():
            routed = 0.0
            shexp = 0.0
            router = 0.0
            dense = 0.0
            for kind in _EXPERT_KINDS:
                if _name(layer, kind) in self.tensors:
                    routed += self._expert_bytes(layer, kind) * top_k
            if include_shexp:
                for kind in _SHEXP_KINDS:
                    if _name(layer, kind) in self.tensors:
                        shexp += self._tensor_bytes(layer, kind)
            t = self._tensor(layer, "ffn_gate_inp")
            if t is not None:
                router += self._tensor_bytes(layer, "ffn_gate_inp")
            t = self._tensor(layer, "exp_probs_b")
            if t is not None:
                router += self._tensor_bytes(layer, "exp_probs_b")
            t = self._tensor(layer, "ffn_gate_tid2eid")
            if t is not None:
                router += self._tensor_bytes(layer, "ffn_gate_tid2eid")
            router_total += router
            if include_dense:
                for tname, t in self.tensors.items():
                    if not tname.startswith(f"blk.{layer}."):
                        continue
                    kind = tname.split(".")[2]
                    if kind in _EXPERT_KINDS + _SHEXP_KINDS + (
                            "ffn_gate_inp", "exp_probs_b",
                            "ffn_gate_tid2eid"):
                        continue
                    dense += self._tensor_bytes(layer, kind)
            per_layer[layer] = {"routed": routed, "shexp": shexp,
                               "dense": dense,
                               "router": 0.0 if router_amortized else router}
            total += routed + shexp + dense
            if not router_amortized:
                total += router
        return {"bytes_per_token": total,
                "routed_total": sum(v["routed"] for v in per_layer.values()),
                "shexp_total": sum(v["shexp"] for v in per_layer.values()),
                "dense_total": sum(v["dense"] for v in per_layer.values()),
                "router_total_once": router_total,
                "per_layer": per_layer}

    def summary(self) -> str:
        rows = []
        for layer in self.layers():
            for t in sorted(self.tensors.values(), key=lambda t: t["name"]):
                if not t["name"].startswith(f"blk.{layer}."):
                    continue
                dims = t["dims"]
                kind = t["name"].split(".")[2]
                size = self._tensor_bytes(layer, kind)
                rows.append((layer, kind, dims, size))
        lines = [f"ExpertStore({self.shard.name}): "
                 f"{len(self.tensors)} tensors, "
                 f"{len(self.layers())} layers, "
                 f"n_hash_layers={self.n_hash_layers}"]
        for layer, kind, dims, size in rows[:8]:
            lines.append(f"  blk.{layer}.{kind}: {list(dims)} "
                         f"{size / 2**20:.1f} MiB")
        if len(rows) > 8:
            lines.append("  ...")
        total = sum(r[3] for r in rows)
        lines.append(f"  total inventoried storage: {total / 2**30:.1f} GiB")
        return "\n".join(lines)


def save_inventory(store: ExpertStore, path: str | Path) -> None:
    """Write the expert inventory (for the lazy-executor runtime)."""
    inv = {
        "version": 1,
        "alignment": store.alignment,
        "data_start": store.data_start,
        "tensors": [
            {"name": t["name"], "dims": list(t["dims"]),
             "ttype": t["ttype"], "off": t["off"]}
            for t in sorted(store.tensors.values(), key=lambda t: t["name"])
        ],
    }
    Path(path).write_text(json.dumps(inv, indent=2))
