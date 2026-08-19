"""V4-Coder step 1 — code-domain prompt battery for the routing census.

Tokenizes a cross-language code battery with the V4 tokenizer
(outputs/v4_tokenizer.json) and writes outputs/code_prompts.json in
the same {"token_ids": [...]} format cluster_dense_trace.py consumes.

Usage:
    python scripts/v4_make_code_battery.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tokenizers import Tokenizer  # noqa: E402

TOKENIZER_JSON = ROOT / "outputs" / "v4_tokenizer.json"

CODE_PROMPTS = [
    # algorithms / data structures
    "Implement a binary search over a sorted array returning the index or -1.",
    "Write a function that reverses a linked list in place.",
    "Implement Dijkstra's shortest path with a priority queue.",
    "Write a LRU cache class with get and put in O(1).",
    "Implement quicksort with in-place partitioning.",
    "Write a function to merge k sorted lists efficiently.",
    "Implement a trie with insert, search, and startsWith.",
    "Find the longest common subsequence of two strings using dynamic programming.",
    "Implement a thread-safe ring buffer.",
    "Write a Bloom filter with configurable size and hash count.",
    # python specifics
    "Write a Python decorator that caches function results with functools.lru_cache.",
    "Use asyncio to fetch 10 URLs concurrently and collect the results.",
    "Write a context manager that times a block of code.",
    "Implement a dataclass with a custom __post_init__ validation.",
    "Write a generator that yields the first n Fibonacci numbers.",
    "Use itertools.groupby to group a list of dicts by a key.",
    # systems / concurrency
    "Write a producer-consumer queue in C with pthreads and a mutex.",
    "Implement a simple HTTP server in Go with graceful shutdown.",
    "Write a memory allocator with malloc/free using a free list.",
    "Implement a copy-on-write fork in a toy operating system.",
    "Write a lock-free stack using compare-and-swap.",
    # rust / c++ / js
    "Write a Rust function that parses an integer from a string with error handling.",
    "Implement a smart pointer with reference counting in C++.",
    "Write a JavaScript debounce function for a search input.",
    "Implement a TypeScript generic that maps over a tuple.",
    "Write a C++ template metaprogram to compute factorial at compile time.",
    # databases / query
    "Write SQL to find duplicate rows in a users table by email.",
    "Implement a b-tree insertion algorithm.",
    "Write a query to compute a running total per day in PostgreSQL.",
    # security / cryptography
    "Implement AES key expansion in Python.",
    "Write a constant-time string comparison function.",
    "Implement HMAC-SHA256 from primitives.",
    # numerics
    "Implement gradient descent for linear regression with numpy.",
    "Write a function to compute the matrix pseudoinverse via SVD.",
    "Implement the softmax function with numerical stability.",
    "Write a Kalman filter update step.",
    # code understanding / edge cases
    "Explain what a dangling pointer is and give an example.",
    "What is the difference between stack and heap allocation?",
    "Fix this bug: for i in range(len(items)): if items[i] == 3: items.pop(i)",
    "What does tail recursion mean and why do compilers optimize it?",
]

# short, domain-spanning selection for a 96-token census trace
CENSUS_PROMPTS = [
    "Merge two sorted arrays in place.",
    "Python async fetch 10 urls concurrently.",
    "Rust parse integer with error handling.",
    "SQL find duplicate emails in users.",
    "Implement AES key expansion.",
    "C lock-free stack compare-and-swap.",
    "JavaScript debounce a search input.",
    "Gradient descent for linear regression numpy.",
    "Implement a Bloom filter.",
    "Tail recursion why compilers optimize it.",
    "Implement a b-tree insertion.",
    "TypeScript generic maps over a tuple.",
]

# rare / adversarial / obfuscated code — where the CVaR tail lives
RARE_CODE_PROMPTS = [
    "Decode this obfuscated Python: exec(''.join(chr(ord(c)-1) for c in 'ifmmp xpsme'))",
    "What does this regex do: ^(?=.*[a-z])(?=.*[A-Z])(?=.*\\d)(?=.*[!@#])[A-Za-z\\d!@#]{12,}$",
    "Fix undefined behavior in this C code: int a[5]; for(int i=0;i<=5;i++) a[i]=i;",
    "Explain integer overflow in this Rust release-mode snippet.",
    "Deobfuscate this JS: []['filter']['constructor']('return this')()",
    "What is the bug in this SQL: SELECT * FROM users WHERE id = 1 OR 1=1",
    "Explain the ABA problem in this lock-free stack code.",
    "What does this C macro expand to: #define SQ(x) x*x",
    "Explain NaN poisoning in this softmax implementation.",
    "What is wrong with this constant-time comparison function?",
    "Explain use-after-free in this code snippet.",
    "Explain pointer aliasing violation in this function.",
    "What does this Brainfuck program do: ++++++++++[>+++++++>++++++++++<<-]>++.",
    "Parse this deeply nested YAML and explain the bomb risk.",
]


def main() -> int:
    tok = Tokenizer.from_file(str(TOKENIZER_JSON))
    ids = []
    meta = []
    for p in CODE_PROMPTS:
        enc = tok.encode(p)
        if enc.ids:
            meta.append({"text": p, "n": len(enc.ids),
                         "start": len(ids)})
            ids.extend(enc.ids)
    out = {
        "token_ids": [int(i) for i in ids],
        "prompts": meta,
        "n_prompts": len(meta),
        "n_tokens": len(ids),
        "domain": "code",
    }
    dest = ROOT / "outputs" / "code_prompts.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {dest}: {len(meta)} prompts, {len(ids)} tokens")

    # short domain-spanning variant for a 96-token census trace
    short_ids = []
    short_meta = []
    for p in CENSUS_PROMPTS:
        enc = tok.encode(p)
        if enc.ids:
            short_meta.append({"text": p, "n": len(enc.ids),
                               "start": len(short_ids)})
            short_ids.extend(enc.ids)
    s_out = {
        "token_ids": [int(i) for i in short_ids],
        "prompts": short_meta,
        "n_prompts": len(short_meta),
        "n_tokens": len(short_ids),
        "domain": "code",
    }
    s_dest = ROOT / "outputs" / "code_census_prompts.json"
    s_dest.write_text(json.dumps(s_out, indent=2), encoding="utf-8")
    print(f"wrote {s_dest}: {len(short_meta)} prompts, "
          f"{len(short_ids)} tokens")

    # rare/adversarial variant for the CVaR tail gate
    rare_ids = []
    rare_meta = []
    for p in RARE_CODE_PROMPTS:
        enc = tok.encode(p)
        if enc.ids:
            rare_meta.append({"text": p, "n": len(enc.ids),
                               "start": len(rare_ids)})
            rare_ids.extend(enc.ids)
    r_out = {
        "token_ids": [int(i) for i in rare_ids],
        "prompts": rare_meta,
        "n_prompts": len(rare_meta),
        "n_tokens": len(rare_ids),
        "domain": "rare-code",
    }
    r_dest = ROOT / "outputs" / "rare_code_prompts.json"
    r_dest.write_text(json.dumps(r_out, indent=2), encoding="utf-8")
    print(f"wrote {r_dest}: {len(rare_meta)} prompts, "
          f"{len(rare_ids)} tokens")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
