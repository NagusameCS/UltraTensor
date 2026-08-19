"""V4-Coder Phase-4 distillation corpus generator.

Builds a diverse, deterministic code-prompt corpus for teacher runs
(V4-Pro/Flash) that Phase-4 distillation will compress into the
extracted coder.  Categories x languages with segment meta so the
existing census machinery can slice it too.

Writes outputs/distill_prompts.json.

Usage:
    python scripts/v4_distill_corpus.py            # default 200 prompts
    python scripts/v4_distill_corpus.py --size 1000
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tokenizers import Tokenizer  # noqa: E402

TOKENIZER_JSON = ROOT / "outputs" / "v4_tokenizer.json"

SIGNATURES = {
    "python": [
        "def {name}({args}) -> {ret}:",
        "class {Name}:",
        "async def {name}({args}):",
    ],
    "rust": [
        "fn {name}({args}) -> {ret} {{",
        "impl {Name} {{",
        "pub fn {name}({args}) -> Result<{ret}, Error> {{",
    ],
    "sql": [
        "SELECT {cols} FROM {table} WHERE",
        "CREATE INDEX idx_{name} ON {table}",
        "WITH cte AS (SELECT {cols} FROM {table})",
    ],
    "js": [
        "function {name}({args}) {{",
        "const {name} = ({args}) => {{",
        "async function {name}({args}) {{",
    ],
    "go": [
        "func {name}({args}) {ret} {{",
        "func (s *{Name}) {name}({args}) error {{",
    ],
    "cpp": [
        "template<typename T> T {name}({args}) {{",
        "class {Name} {{",
    ],
}

TASKS = [
    "implement it",
    "implement it with error handling",
    "implement it and write unit tests",
    "implement it iteratively",
    "implement it recursively",
    "implement it with caching",
    "implement it memory-efficiently",
    "write a one-line version",
    "explain then implement it",
]

NAMES = [
    "merge_intervals", "top_k_frequent", "binary_search", "lru_cache",
    "rate_limiter", "batch_processor", "retry_wrapper", "slugify",
    "deep_copy", "flatten_tree", "find_cycle", "shortest_path",
    "streaming_dedup", "event_bus", "serialize", "validate_input",
    "chunked_reader", "backoff", "normalize_tokens", "shard_assign",
]
ARGS = ["items", "data, limit", "xs", "events", "tree", "key, value",
        "rows", "path", "stream", "n"]
RETS = ["int", "bool", "list", "str", "None", "float"]
COLS = ["id, name, created_at", "user_id, COUNT(*)", "a.id, b.total"]
TABLES = ["orders", "users", "events", "sessions", "payments"]


def build(size: int, seed: int = 7):
    import random
    rng = random.Random(seed)
    prompts, meta, ids = [], [], []
    tok = Tokenizer.from_file(str(TOKENIZER_JSON))
    langs = list(SIGNATURES)
    per_lang = max(1, size // len(langs))
    for lang in langs:
        start = len(ids)
        seen = set()
        for _ in range(per_lang):
            name = rng.choice(NAMES)
            args = rng.choice(ARGS)
            ret = rng.choice(RETS)
            tmpl = rng.choice(SIGNATURES[lang])
            if lang == "sql":
                line = tmpl.format(name=name, cols=rng.choice(COLS),
                                   table=rng.choice(TABLES))
                if line.endswith("WHERE"):
                    line += f" {name}.active = 1"
            else:
                line = tmpl.format(name=name, Name=name.title(),
                                   args=args, ret=ret)
            task = rng.choice(TASKS)
            text = f"{line}\n# {task}"
            if text in seen:
                continue
            seen.add(text)
            prompts.append(text)
            enc = tok.encode(text)
            ids.extend(int(i) for i in (enc.ids or []))
        meta.append({"language": lang, "task": "mixed",
                     "start": start, "n": len(ids) - start})
    out = {"prompts": prompts, "segments": meta,
           "token_ids": ids, "n_tokens": len(ids),
           "domain": "distill-corpus", "seed": seed}
    dest = ROOT / "outputs" / "distill_prompts.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {dest}: {len(prompts)} prompts, {len(ids)} tokens")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    return build(a.size, a.seed)


if __name__ == "__main__":
    raise SystemExit(main())
