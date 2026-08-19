"""Milestone L validation: the C BPE tokenizer vs the llama.cpp server
(:8774 /tokenize) as ground truth on a battery of strings."""
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXE = Path(r"C:\Users\legom\OneDrive\Documents\GitHub\HyperTensor")
EXE = EXE / "build_host" / "test_v4_tok.exe"
GGUF = ("D:/hyperv4/models/pro/"
        "deepseek-ai-DeepSeek-V4-Pro-Q3_K_M-00001-of-00017.gguf")

TESTS = [
    "Hello world",
    "The quick brown fox jumps over the lazy dog.",
    "Hello, how are you today? I am doing well!",
    "def fib(n):\n    return n if n < 2 else fib(n-1) + fib(n-2)",
    "The caf\u00e9 has cr\u00e8mes br\u00fbl\u00e9es.",
    "I'm testing punctuation: commas, semicolons; and (parentheses)!",
    "1 + 2 = 3, but 12 + 34 = 46.",
    "OpenAI-parity serving with SSE streaming.",
    "x = [i for i in range(100) if i % 7 == 0]",
    "M\u00fcnchen \u00dcberraschung na\u00efve fa\u00e7ade",
    "Hello\nworld\ttabs",
    "a b",
    "...",
    "(x) (n) (ab) a(b",
    "12345abc123",
]


def server_tokens(s: str):
    req = urllib.request.Request(
        "http://127.0.0.1:8774/tokenize",
        data=json.dumps({"content": s}).encode(),
        headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=120).read()) \
        .get("tokens")


def c_tokens(s: str):
    rc = subprocess.run([str(EXE), GGUF, s], capture_output=True,
                        text=True)
    for line in rc.stdout.splitlines():
        if "[" in line and "]" in line:
            body = line.rpartition("[")[2].rpartition("]")[0]
            if body.strip():
                return [int(x) for x in body.split(",")]
    return None


def main() -> int:
    bad = 0
    for s in TESTS:
        ref = server_tokens(s)
        got = c_tokens(s)
        ok = ref == got
        bad += not ok
        print(f"{'OK ' if ok else 'BAD'} {s[:40]!r}")
        if not ok:
            print(f"    ref={ref}\n    got={got}")
    print(f"battery: {len(TESTS) - bad}/{len(TESTS)} exact")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
