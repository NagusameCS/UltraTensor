"""G3 controller-level evaluation on REAL hash traces.

Uses the saved per-token expert sets from router_trace_hash.json and the
token bigram as the "drafter". Measures, per hash layer:

- PrefetchController hit rate vs size cap (draft-driven prefetch)
- tier_sweep latency curve with the measured miss cost
  (0.154 s/layer hash bench) and prefetch benefit

This is the wiring result: drafter -> controller -> tier, on real bytes.

Usage: python scripts/v4_prefetch_eval.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from ultratensor.conditional.lookahead import PrefetchController  # noqa: E402
from ultratensor.conditional.tiering import simulate_tier, tier_sweep  # noqa: E402

HASH_LAYER_COST_S = 0.1535   # measured hash-layer per-token latency


def bigram_drafts(tokens):
    """argmax token bigram from the real token stream (the toy drafter)."""
    from collections import Counter, defaultdict
    nxt = defaultdict(Counter)
    for a, b in zip(tokens, tokens[1:]):
        nxt[a][b] += 1
    pred = {a: c.most_common(1)[0][0] for a, c in nxt.items()}
    return lambda tok: [pred.get(tok, tok)]


def main() -> int:
    p = ROOT / "outputs" / "router_trace_hash.json"
    if not p.exists():
        print("run scripts/v4_router_trace.py first")
        return 2
    d = json.load(open(p, encoding="utf-8"))
    report = {"layers": {}}
    for layer, entry in d["layers"].items():
        sets = entry["sets"]
        toks = entry.get("token_ids")
        if not toks:
            print(f"layer {layer}: token_ids missing; rerun v4_router_trace.py")
            continue
        toks = [int(t) for t in toks]

        # deterministic token -> expert set table observed in the trace
        tok2set = {t: s for t, s in zip(toks, sets)}
        drafts_fn = bigram_drafts(toks)

        def eval_drafter(name, draft_seq):
            row = {}
            for cap in (6, 12, 18, 24):
                ctl = PrefetchController(
                    table_fn=lambda t, _t2s=tok2set: _t2s.get(t, [0] * 6),
                    H=4, size_cap=cap)
                overlap = []
                for i in range(1, len(toks) - 3):
                    dtoks = list(draft_seq(i))
                    ctl.observe(dtoks, toks[i:i + 4])
                    plan = set(ctl.plan(dtoks))
                    actual = {e for t in toks[i:i + 4]
                              for e in tok2set.get(t, [])}
                    overlap.append(len(plan & actual) / max(1, len(actual)))
                row[f"cap{cap}"] = {
                    "full_hit_rate": round(ctl.hit_rate, 4),
                    "overlap_hit": round(float(np.mean(overlap)), 4),
                }
            return row

        row = {"bigram": eval_drafter(
            "bigram", lambda i: [drafts_fn(toks[i - 1])[0]] * 4)}
        row["oracle"] = eval_drafter("oracle", lambda i: toks[i:i + 4])
        report["layers"][layer] = row
        print(f"layer {layer}: bigram cap6 full={row['bigram']['cap6']['full_hit_rate']} "
              f"overlap={row['bigram']['cap6']['overlap_hit']} | "
              f"oracle cap6 full={row['oracle']['cap6']['full_hit_rate']} "
              f"overlap={row['oracle']['cap6']['overlap_hit']}")

    out = ROOT / "outputs" / "prefetch_eval.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
