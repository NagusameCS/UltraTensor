"""Cluster phase-B runner (rognode2): dense-router traces from REAL bytes.

Self-contained variant of v4_router_trace_dense.py for cluster nodes:
reads pre-tokenized prompt ids (no tokenizers dep), NAS-resident shards,
bounded weight cache (6 GB on node2's 30 GB). Writes npz + json +
progress directly to the NAS so the laptop sees results on Y:.

Usage (on node2):
    python3 cluster_dense_trace.py \
      --shards '/mnt/nas20/models/v4pro/*.gguf' \
      --prompts /mnt/nas20/exp/cluster_prompts.json \
      --out /mnt/nas20/exp --tokens 8
"""

import argparse
import glob
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

import v4_ref_serve as vs  # noqa: E402
from v4_cache import cached_load  # noqa: E402
from ultratensor.conditional.lookahead import (  # noqa: E402
    WorkingSetModel,
    evaluate_prefetch,
    oracle_curve,
)
from v4_ref_gen import BlockGen, HC  # noqa: E402

MAX_LAYER = 4          # blocks 0..3: three hash + layer 3 dense
_ROUTE_SCALE = 2.5
_SHEXP_KINDS = ("ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp")


class NumpyMoELayer:
    """Pure-numpy MoE layer (Linux-friendly twin of moe_exec.MoELayer).

    Same route/__call__ semantics, using store.read_expert dequant per
    expert (validated against the C executor on Windows)."""

    def __init__(self, store, layer, top_k=6):
        self.store, self.layer, self.top_k = store, layer, top_k
        self.n, self.m, self.E = store.expert_shape(layer, "ffn_gate_exps")
        self.shexp = {}
        for kind in _SHEXP_KINDS:
            try:
                self.shexp[kind] = store.read_tensor(layer, kind)
            except (KeyError, FileNotFoundError):
                pass

    def route(self, h, token_ids=None, with_weights=True):
        h2 = np.atleast_2d(np.ascontiguousarray(h, np.float32))
        ids = self.store.route_layer(self.layer, h2, token_ids=token_ids,
                                     top_k=self.top_k)
        if not with_weights:
            return ids, None
        z = self.store.read_tensor(self.layer, "ffn_gate_inp") @ h2.T
        s = np.sqrt(np.log1p(np.exp(z)))
        w = np.stack([s[ids[b], b] for b in range(ids.shape[0])])
        w = w / w.sum(axis=-1, keepdims=True) * _ROUTE_SCALE
        return ids, w.astype(np.float32)

    def _expert(self, e, x):
        gate = self.store.read_expert(self.layer, "ffn_gate_exps", e)
        up = self.store.read_expert(self.layer, "ffn_up_exps", e)
        down = self.store.read_expert(self.layer, "ffn_down_exps", e)
        g = gate @ x
        u = up @ x
        h = g / (1.0 + np.exp(-g)) * u
        return down @ h

    def __call__(self, h, token_ids=None):
        h2 = np.atleast_2d(np.ascontiguousarray(h, np.float32))
        B = h2.shape[0]
        ids, weights = self.route(h2, token_ids=token_ids)
        y = np.zeros((B, self.n), np.float32)
        counts = np.bincount(ids.reshape(-1), minlength=self.E)
        selected = [e for e in range(self.E) if counts[e] > 0]
        for e in selected:
            rows, cols = np.where(ids == e)
            ye = self._expert(e, h2[rows[0]])
            ye *= weights[rows[0], cols[0]]
            y[rows] += ye
        if self.shexp:
            g = h2 @ self.shexp["ffn_gate_shexp"].T
            u = h2 @ self.shexp["ffn_up_shexp"].T
            y += (g / (1.0 + np.exp(-g)) * u) @ self.shexp["ffn_down_shexp"].T
        return y

    def close(self):
        pass


class LoggingMoELayer(NumpyMoELayer):
    def __init__(self, store, layer, log, p, tok):
        super().__init__(store, layer)
        self._log, self._p, self._tok = log, p, tok

    def route(self, h, token_ids=None, **kw):
        ids, w = super().route(h, token_ids=token_ids, **kw)
        self._log.append({"p": self._p, "tok": int(self._tok),
                          "ids": [int(i) for i in ids[0]],
                          "w": [float(x) for x in w[0]],
                          "h": np.asarray(h[0], dtype=np.float32).copy()})
        return ids, w


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", type=int, default=8)
    ap.add_argument("--max-layer", type=int, default=4,
                    help="blocks 0..max_layer-1")
    a = ap.parse_args()

    tokens = json.load(open(a.prompts, encoding="utf-8"))["token_ids"]
    tokens = tokens[:a.tokens]
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    shards = sorted(glob.glob(a.shards))
    if not shards:
        print("no shards at", a.shards)
        return 2
    st = vs.ExpertStore(shards[0], extra_shards=shards[1:])
    vs.load_any = cached_load(vs.load_any, max_bytes=6 << 30)

    blocks = [BlockGen(st, L) for L in range(a.max_layer)]
    logs = {L: [] for L in range(a.max_layer)}
    t0 = time.time()
    for p, tok in enumerate(tokens):
        h = vs.load_any(st, "token_embd.weight", rows=[tok])[0]
        x = np.tile(h, (HC, 1)).astype(np.float32)
        for L in range(a.max_layer):
            ml = LoggingMoELayer(st, L, logs[L], p, tok)
            x = blocks[L].step(x, p, tok, ml)
            ml.close()
            del ml
        (out_dir / "progress.txt").write_text(
            f"p={p}/{len(tokens)} tok={tok} ({time.time() - t0:.0f}s)\n")
        print(f"p={p} tok={tok} done ({time.time() - t0:.1f}s)", flush=True)

    report = {"n_tokens": len(tokens), "layers": {}}
    hh = {}
    for L in range(a.max_layer):
        seq = [set(e["ids"]) for e in logs[L]]
        entry = {"n_steps": len(seq),
                 "distinct_experts": len(set().union(*seq)),
                 "H": {}}
        model = WorkingSetModel().fit(seq)
        for H in (1, 4, 8):
            curve = evaluate_prefetch(seq, model, H)
            entry["H"][H] = {
                "oracle_union": oracle_curve(seq, H)["mean_union_size"],
                "set_markov_hit": round(curve.best_hit, 4),
                "set_markov_size": round(curve.best_size, 2),
            }
        report["layers"][L] = entry
        hh[f"L{L}"] = np.stack([e["h"] for e in logs[L]])
        print(f"layer {L}: distinct={entry['distinct_experts']} "
              f"H=4 oracle={entry['H'][4]['oracle_union']:.1f}", flush=True)

    (out_dir / "router_trace_dense.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(out_dir / "ffn_inputs_dense.npz", **hh)
    print(f"done in {time.time() - t0:.0f}s -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
