"""V4-Coder language isolation — per-programming-language battery.

The isolation hypothesis: different programming languages route to
different expert subsets. Battery: four ~24-token segments (Python,
Rust, SQL, JavaScript) in ONE prompt file with segment labels in the
meta, so one 96-token cluster trace yields per-language routing
censuses and the pairwise expert-overlap matrix.

Writes outputs/pl_prompts.json.

Usage:
    python scripts/v4_make_pl_batteries.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tokenizers import Tokenizer  # noqa: E402

TOKENIZER_JSON = ROOT / "outputs" / "v4_tokenizer.json"

SEGMENTS = [
    ("python", [
        "Write a Python list comprehension that filters even numbers.",
        "Python decorator to time a function.",
        "Use collections.Counter to count words.",
        "Python dataclass with a custom validator.",
    ]),
    ("rust", [
        "Rust function to reverse a string safely.",
        "Rust match statement over a Result.",
        "Implement a Rust trait for a struct.",
        "Rust closure that captures by reference.",
    ]),
    ("sql", [
        "SQL query to join orders and customers.",
        "SQL window function to rank by date.",
        "SQL GROUP BY with HAVING filter.",
        "SQL index design for a composite key.",
    ]),
    ("js", [
        "JavaScript array map and filter chain.",
        "JS async await error handling pattern.",
        "JavaScript closure inside a loop.",
        "JS destructuring with default values.",
    ]),
]


def main() -> int:
    tok = Tokenizer.from_file(str(TOKENIZER_JSON))
    ids, meta = [], []
    for lang, prompts in SEGMENTS:
        start = len(ids)
        seg = []
        for p in prompts:
            enc = tok.encode(p)
            if enc.ids:
                seg.extend(int(i) for i in enc.ids)
        ids.extend(seg)
        meta.append({"language": lang, "start": start,
                     "n": len(seg)})
    out = {"token_ids": ids, "segments": meta,
           "n_tokens": len(ids), "domain": "pl-isolation"}
    dest = ROOT / "outputs" / "pl_prompts.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {dest}: {len(ids)} tokens, "
          f"{[(s['language'], s['n']) for s in meta]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
