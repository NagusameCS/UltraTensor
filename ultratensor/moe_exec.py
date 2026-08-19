"""MoE layer executor: the Phase 3b compute path, end to end.

Routes a token batch through one V4-Pro MoE layer using ONLY the
selected experts (top-6 + shared expert), decoded on the fly by the C
executor (expert_gemv.c). This is the primitive a lazy top-k server
loops over; its per-token cost is the floor the lazy path can reach.

Reference semantics (deepseek-ai/DeepSeek-V4-Pro inference/model.py,
bati.cpp deepseek4.cpp weight_before_down=true):
    z        = sqrt(softplus(gate_inp @ x)) + bias   (or hash table)
    ids      = top-k(z)
    w        = z[ids] / sum(z[ids]) * route_scale   (2.5)
    per expert e with tokens t:
        u = clamp(up_e(x_t), -10, 10)
        g = clamp(gate_e(x_t), max=10)
        y_t += w_t * down_e(silu(g) * u)
    y += shared_expert(x)   (same swiglu limits)

Only the gate/up/down rows of the SELECTED experts are decoded; the
router + shared expert stay resident (~641 MB + ~2 GB for the shard's
shexp tensors).
"""
from __future__ import annotations

import collections
import threading
import time

import numpy as np

from .expert_store import ExpertStore, _SHEXP_KINDS
from .kernels import ExpertGEMV

_SWIGLU_LIMIT = 10.0
_ROUTE_SCALE = 2.5


def _swiglu(gate: np.ndarray, up: np.ndarray) -> np.ndarray:
    gate = np.minimum(gate, _SWIGLU_LIMIT)
    up = np.clip(up, -_SWIGLU_LIMIT, _SWIGLU_LIMIT)
    s = gate
    s = s / (1.0 + np.exp(-s))          # silu, overflow-safe form
    return s * up


class _ExpertReader:
    """Worker thread that pre-reads expert bytes ahead of the decode
    loop, so the next expert's disk read overlaps the current decode."""

    def __init__(self, store: ExpertStore, layer: int,
                 kinds=("ffn_gate_exps", "ffn_down_exps", "ffn_up_exps")):
        self.kinds = kinds
        self._info = {}
        self._fhs = []
        for kind in kinds:
            t = store._tensor(layer, kind)
            fh = open(str(store.shards[t["shard"]]), "rb")
            abs_off = store._data_starts[t["shard"]] + t["off"]
            self._info[kind] = (fh, abs_off,
                                store._expert_bytes(layer, kind))
            self._fhs.append(fh)
        self._cond = threading.Condition()
        self._queue = collections.deque()
        self._ready = {}
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def request(self, e: int):
        with self._cond:
            for kind in self.kinds:
                self._queue.append((e, kind))
            self._cond.notify()

    def get(self, e: int, kind: str) -> bytes:
        with self._cond:
            while (e, kind) not in self._ready:
                self._cond.wait()
            return self._ready.pop((e, kind))

    def _run(self):
        while True:
            with self._cond:
                while not self._queue and not self._stop:
                    self._cond.wait()
                if self._stop and not self._queue:
                    return
                e, kind = self._queue.popleft()
            fh, off, size = self._info[kind]
            fh.seek(off + e * size)
            data = fh.read(size)
            with self._cond:
                self._ready[(e, kind)] = data
                self._cond.notify()

    def close(self):
        with self._cond:
            self._stop = True
            self._cond.notify()
        self._thread.join(timeout=5)
        for fh in self._fhs:
            fh.close()


class MoELayer:
    """One DeepSeek-V4 MoE layer, executed lazily per selected expert."""

    def __init__(self, store: ExpertStore, layer: int, top_k: int = 6):
        self.store = store
        self.layer = layer
        self.top_k = top_k
        self._execs = {}
        for kind in ("ffn_gate_exps", "ffn_down_exps", "ffn_up_exps"):
            name = f"blk.{layer}.{kind}.weight"
            if name in store.tensors:
                sh = str(store.shards[store.tensors[name]["shard"]])
                cg = ExpertGEMV()
                cg.open(sh, name)
                self._execs[kind] = cg
        # shared expert (dense, resident)
        self.shexp = {}
        for kind in _SHEXP_KINDS:
            try:
                self.shexp[kind] = store.read_tensor(layer, kind)
            except KeyError:
                pass
        self.n, self.m, self.E = store.expert_shape(layer, "ffn_gate_exps")
        self._reader = _ExpertReader(store, layer)

    def route(self, h: np.ndarray, token_ids=None,
              with_weights: bool = True):
        """h [B,n] -> (ids [B,k], weights [B,k]) using the real router.

        Reference semantics (deepseek-ai/DeepSeek-V4-Pro inference/model.py
        Gate.forward):
        - hash layers (0..n_hash-1): indices = ffn_gate_tid2eid[token_id]
          (no bias; the dense gate is still computed to WEIGHT the
          hashed experts).
        - dense layers: indices = top-k of sqrt(softplus(z)) + bias;
          the bias shifts SELECTION only.
        - weights in both cases: unbiased sqrt(softplus(z)) gathered at
          the indices, normalized, x route_scale.
        Accepts a 1-D [n] vector too.
        """
        h = np.atleast_2d(np.ascontiguousarray(h, np.float32))
        ids = self.store.route_layer(self.layer, h, token_ids=token_ids,
                                     top_k=self.top_k)
        if not with_weights:
            return ids, None
        z = self.store.read_tensor(self.layer, "ffn_gate_inp") @ h.T  # [E,B]
        # Reference semantics (deepseek-ai/DeepSeek-V4-Pro inference/model.py):
        # the bias shifts the top-k SELECTION only; the routing weights are
        # gathered from the UNBIASED sqrt(softplus) scores, normalized,
        # then scaled by route_scale. Hash layers have no bias at all.
        s = np.sqrt(np.log1p(np.exp(z)))                       # [E, B]
        w = np.stack([s[ids[b], b] for b in range(ids.shape[0])])  # [B,k]
        w = w / w.sum(axis=-1, keepdims=True) * _ROUTE_SCALE
        return ids, w.astype(np.float32)

    def _expert(self, e: int, x: np.ndarray):
        """x [T, n] -> y [T, n] for one expert (unweighted). Expert bytes
        were pre-read by the reader thread; decode per token from memory."""
        x = np.ascontiguousarray(x, np.float32)
        raw_g = self._reader.get(e, "ffn_gate_exps")
        raw_u = self._reader.get(e, "ffn_up_exps")
        raw_d = self._reader.get(e, "ffn_down_exps")
        out = []
        for t in range(x.shape[0]):
            g = self._execs["ffn_gate_exps"].gemv_mem(raw_g, x[t])
            u = self._execs["ffn_up_exps"].gemv_mem(raw_u, x[t])
            h = _swiglu(g, u)
            out.append(self._execs["ffn_down_exps"].gemv_mem(raw_d, h))
        return np.stack(out).astype(np.float32)

    def __call__(self, h: np.ndarray, token_ids=None,
                 timings: bool = False) -> np.ndarray:
        """h [B,n] (or a single [n] vector) -> MoE layer output [B,n]."""
        h = np.atleast_2d(np.ascontiguousarray(h, np.float32))
        B = h.shape[0]
        ids, weights = self.route(h, token_ids=token_ids)
        t0 = time.time()
        y = np.zeros((B, self.n), np.float32)
        counts = np.bincount(ids.reshape(-1), minlength=self.E)
        selected = [e for e in range(self.E) if counts[e] > 0]
        if selected:
            self._reader.request(selected[0])
        for i, e in enumerate(selected):
            if i + 1 < len(selected):
                self._reader.request(selected[i + 1])   # overlap next read
            rows, cols = np.where(ids == e)          # (token, topk slot)
            xt = h[rows]
            ye = self._expert(e, xt)
            ye *= weights[rows, cols][:, None]
            y[rows] += ye
        if self.shexp:
            xs = h
            g = xs @ self.shexp["ffn_gate_shexp"].T
            u = xs @ self.shexp["ffn_up_shexp"].T
            y += _swiglu(g, u) @ self.shexp["ffn_down_shexp"].T
        dt = time.time() - t0
        if timings:
            return y, {"layer_s": dt, "tokens": B}
        return y

    def close(self):
        for cg in self._execs.values():
            cg.close()
        self._reader.close()
