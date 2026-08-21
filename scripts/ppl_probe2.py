"""Crash bisect probe round 2: length vs content."""
import json
import sys
import urllib.error
import urllib.request

ENDPOINT = "http://127.0.0.1:8780/v1/chat/completions"


def send(tag, prompt, max_tokens=2):
    payload = {
        "model": "v4",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "logprobs": True,
        "top_logprobs": 5,
    }
    req = urllib.request.Request(
        ENDPOINT, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=240) as r:
            r.read()
        print(f"[{tag}] OK", flush=True)
    except Exception as e:
        print(f"[{tag}] FAIL {type(e).__name__}: {str(e)[:120]}", flush=True)


VARIANTS = [
    ("c2_special_short", "O(n^2)"),
    ("c1_length16", "hi hi hi hi hi hi hi hi"),
    ("c3_word", "palindromic"),
    ("c4_code_full", "Write a Python function that finds the longest "
                     "palindromic substring in O(n^2)."),
    ("c5_math", "Solve: integrate x^2 * exp(-x) from 0 to infinity."),
]

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    for tag, text in VARIANTS:
        if which != "all" and not tag.startswith(which):
            continue
        send(tag, text)
