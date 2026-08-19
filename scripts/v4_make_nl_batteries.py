"""V4-Coder language isolation — per-NATURAL-language battery.

Second half of the isolation question: does the router separate
Chinese / Japanese / English text (same coding-domain meaning)?  Each
language gets ~3 prompts; one prompt file, segment labels in meta,
same format as pl_prompts.json so cluster_pl_census.py works as-is.

Writes outputs/nl_prompts.json.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tokenizers import Tokenizer  # noqa: E402

TOKENIZER_JSON = ROOT / "outputs" / "v4_tokenizer.json"

SEGMENTS = [
    ("chinese", [
        "写一个Python函数来反转一个字符串。",
        "用Python的字典来计算单词出现次数。",
        "实现一个带缓存装饰器的Python函数。",
    ]),
    ("japanese", [
        "文字列を逆順にするPython関数を書いてください。",
        "Pythonの辞書で単語の出現回数を数える。",
        "キャッシュ付きデコレータをPythonで実装する。",
    ]),
    ("english", [
        "Write a Python function to reverse a string.",
        "Count word frequencies with a Python dict.",
        "Implement a caching decorator in Python.",
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
           "n_tokens": len(ids), "domain": "nl-isolation"}
    dest = ROOT / "outputs" / "nl_prompts.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {dest}: {len(ids)} tokens, "
          f"{[(s['language'], s['n']) for s in meta]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
