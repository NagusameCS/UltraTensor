"""G3 phase B: dense-router traces from REAL sequential forwards.

Runs the real V4 block forward (BlockGen from v4_ref_gen.py) through
layers 0..3 for a short prompt sequence, logging every router decision
and every MoE input hidden state. Layer 3 is the first dense layer —
its routes depend on hidden state, unlike the hash layers. This is the
data that decides whether draft-driven prefetch generalises past hash
layers, and it doubles as G4 input (activation-weighted analysis).

Usage:
    python scripts/v4_router_trace_dense.py [--tokens 24]
Writes outputs/router_trace_dense.json and outputs/ffn_inputs_dense.npz.
"""

import argparse
import glob
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import v4_ref_serve as vs  # noqa: E402
from v4_cache import cached_load  # noqa: E402
from ultratensor.conditional.lookahead import (  # noqa: E402
    WorkingSetModel,
    evaluate_prefetch,
    oracle_curve,
)
from v4_ref_gen import BlockGen, HC  # noqa: E402

TOKENIZER_JSON = ROOT / "outputs" / "v4_tokenizer.json"
MAX_LAYER = 4          # blocks 0..3: three hash + layer 3 dense
PROMPTS = [
    "The quick brown fox jumps over the lazy dog. ",
    "Write a Python function to compute Fibonacci numbers. ",
    "The derivative of x squared is two x, because ",
]


class LoggingMoELayer(vs.MoELayer):
    """MoELayer that records routes and FFN inputs on every call."""

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


def tokenize(texts, cap):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(TOKENIZER_JSON))
    ids = []
    for t in texts:
        ids.extend(tok.encode(t).ids)
        if len(ids) >= cap:
            break
    return ids[:cap]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=24)
    a = ap.parse_args()

    shards = sorted(glob.glob("D:/hyperv4/models/pro/"
                              "deepseek-ai-DeepSeek-V4-Pro-Q3_K_M-*.gguf"))
    st = vs.ExpertStore(shards[0], extra_shards=shards[1:])
    # cache attention/compressor tensors across tokens (LRU, bounded)
    vs.load_any = cached_load(vs.load_any, max_bytes=4 << 30)
    tokens = tokenize(PROMPTS, a.tokens)
    print(f"trace: {len(tokens)} tokens, blocks 0..{MAX_LAYER - 1}",
          flush=True)

    blocks = [BlockGen(st, L) for L in range(MAX_LAYER)]
    logs = {L: [] for L in range(MAX_LAYER)}
    progress = ROOT / "outputs" / "router_trace_dense.progress.txt"
    t0 = time.time()
    for p, tok in enumerate(tokens):
        h = vs.load_any(st, "token_embd.weight", rows=[tok])[0]
        x = np.tile(h, (HC, 1)).astype(np.float32)
        for L in range(MAX_LAYER):
            ml = LoggingMoELayer(st, L, logs[L], p, tok)
            x = blocks[L].step(x, p, tok, ml)
            ml.close()
            del ml
        progress.write_text(f"p={p}/{len(tokens)} tok={tok} "
                            f"({time.time() - t0:.0f}s)\n", encoding="utf-8")
        print(f"p={p} tok={tok} done ({time.time() - t0:.1f}s)", flush=True)

    # ---- analysis: lookahead prefetch on the dense layer ----
    report = {"n_tokens": len(tokens), "layers": {}}
    hh = {}
    for L in range(MAX_LAYER):
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
        hh[f"L{L}"] = np.stack([e["h"] for e in logs[L]])  # [steps, 7168]
        print(f"layer {L}: distinct={entry['distinct_experts']}  "
              f"H=1 markov_hit={entry['H'][1]['set_markov_hit']}  "
              f"H=4 oracle={entry['H'][4]['oracle_union']:.1f}", flush=True)

    out = ROOT / "outputs" / "router_trace_dense.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(ROOT / "outputs" / "ffn_inputs_dense.npz", **hh)
    print(f"wrote {out} (wall {time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
