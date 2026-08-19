"""G3 experiment (phase A): real V4-Pro hash-router traces + lookahead
prefetch curves.

Hash layers (0..n_hash_layers-1) route deterministically by token id via
ffn_gate_tid2eid, so their traces need no forward pass. This quantifies,
on REAL model bytes, how well a working-set predictor can prefetch the
next-H-token expert union for the cheap layers — and what the oracle
ceiling is. Phase B (dense layers 3+, hidden-state dependent) needs real
forwards and runs separately.

Usage:
    python scripts/v4_router_trace.py --glob "D:\\hyperv4\\models\\pro\\*.gguf"
"""

import argparse
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from ultratensor.conditional.lookahead import (  # noqa: E402
    WorkingSetModel,
    evaluate_prefetch,
    oracle_curve,
    working_set_union,
)
from ultratensor.expert_store import ExpertStore  # noqa: E402

TAUS = (0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)

TOKENIZER_JSON = ROOT / "outputs" / "v4_tokenizer.json"

PROMPTS = [
    "The quick brown fox jumps over the lazy dog. ",
    "Write a Python function to compute Fibonacci numbers. ",
    "The derivative of x squared is two x, because ",
    "Bonjour, comment allez-vous aujourd'hui? ",
    "今天天气很好，我们一起去公园散步吧。",
    "In 1945 the war ended and Europe began to rebuild. ",
    "SELECT * FROM users WHERE id = ",
    "The mitochondria is the powerhouse of the cell, and ",
]


def tokenize(texts):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(TOKENIZER_JSON))
    encs = [tok.encode(t) for t in texts]
    return [np.asarray(e.ids, dtype=np.int64) for e in encs]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default=r"D:\hyperv4\models\pro\*.gguf")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "router_trace_hash.json"))
    ap.add_argument("--top-k", type=int, default=6)
    ap.add_argument("--H", type=int, nargs="+", default=[1, 4, 8, 16])
    a = ap.parse_args()

    shards = sorted(glob.glob(a.glob))
    shards = [s for s in shards if "deepseek-ai-DeepSeek-V4-Pro-Q3_K_M-" in s]
    if not shards:
        print("no shards found under", a.glob)
        return 2
    st = ExpertStore(shards[0], extra_shards=shards[1:])
    n_hash = st.n_hash_layers
    print(f"inventory: {len(shards)} shards, n_hash_layers={n_hash}")

    ids = tokenize(PROMPTS)
    all_tokens = np.concatenate(ids)
    print(f"prompts: {len(ids)}  tokens: {all_tokens.size}")

    report = {"n_hash_layers": n_hash, "n_tokens": int(all_tokens.size),
              "top_k": a.top_k, "layers": {}}

    for layer in range(n_hash):
        dummy_h = np.zeros((all_tokens.size, 1), dtype=np.float32)
        routes = st.route_layer(layer, dummy_h, token_ids=all_tokens,
                                top_k=a.top_k)
        route_seq = [set(row.tolist()) for row in routes]

        n_distinct = len(set().union(*route_seq))
        entry = {"n_tokens": len(route_seq),
                 "distinct_experts_used": n_distinct,
                 "token_ids": [int(t) for t in all_tokens],
                 "sets": [sorted(s) for s in route_seq],
                 "H": {}}
        model = WorkingSetModel().fit(route_seq)

        # Token-bigram predictor: the hash route is a deterministic
        # function of the NEXT token, so a token drafter IS the exact
        # lookahead predictor. Simulate: argmax-bigram token path,
        # union of the deterministic expert sets along the path.
        table = st.read_tensor(layer, "ffn_gate_tid2eid")

        def set_of(tok: int) -> set:
            return set(int(x) for x in table[int(tok)][: a.top_k])

        from collections import Counter, defaultdict
        nxt: dict[int, Counter] = defaultdict(Counter)
        for i in range(len(all_tokens) - 1):
            nxt[int(all_tokens[i])][int(all_tokens[i + 1])] += 1

        def predict_path(tok: int, H: int):
            u = set()
            cur = tok
            for _ in range(H):
                c = nxt.get(cur)
                if not c:
                    break
                cur = c.most_common(1)[0][0]
                u |= set_of(cur)
            return u

        for H in a.H:
            entry["H"][H] = {"oracle": oracle_curve(route_seq, H)}
            for name, m in (("set_markov", model),):
                curve = evaluate_prefetch(route_seq, m, H, taus=TAUS)
                entry["H"][H][name] = {
                    "best_tau": curve.best_tau,
                    "hit": round(curve.best_hit, 4),
                    "prefetch_size": round(curve.best_size, 2),
                }
            # bigram: hit and size over all start tokens
            bh, bs, n_ev = [], [], 0
            for i in range(1, len(all_tokens) - 1):
                pred = predict_path(int(all_tokens[i - 1]), H)
                target = working_set_union(route_seq, i, H)
                if target:
                    bh.append(len(pred & target) / len(target))
                    bs.append(len(pred))
                    n_ev += 1
            entry["H"][H]["token_bigram"] = {
                "hit": round(float(np.mean(bh)), 4) if n_ev else 0.0,
                "prefetch_size": round(float(np.mean(bs)), 2) if n_ev else 0.0,
            }
            u = working_set_union(route_seq, 0, H)
            entry["H"][H]["first_union_size"] = len(u)
        # churn: fraction of step t's experts absent from step t-1
        churn = [
            len(route_seq[t] - route_seq[t - 1]) / max(1, len(route_seq[t]))
            for t in range(1, len(route_seq))
        ]
        entry["mean_churn"] = round(float(np.mean(churn)), 4) if churn else 0.0
        report["layers"][layer] = entry

        print(f"layer {layer}: distinct experts = {n_distinct}")
        for H in a.H:
            e = entry["H"][H]
            print(f"  H={H:2d}  oracle_union={e['oracle']['mean_union_size']:.1f}"
                  f"  bigram_hit={e['token_bigram']['hit']:.2f}"
                  f"  bigram_size={e['token_bigram']['prefetch_size']:.1f}"
                  f"  set_markov_hit={e['set_markov']['hit']:.2f}"
                  f"  set_markov_size={e['set_markov']['prefetch_size']:.1f}")

    out = Path(a.out)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
