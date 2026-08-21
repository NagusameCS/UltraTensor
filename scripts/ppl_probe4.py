"""Crash bisect round 4: exact token counts + same-length controls."""
import json
import sys
import urllib.error
import urllib.request

ENDPOINT = "http://127.0.0.1:8780/v1/chat/completions"
CODE = ("Write a Python function that finds the longest palindromic "
        "substring in O(n^2).")
WORDS = CODE.split()


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
            body = json.loads(r.read())
        usage = body.get("usage", {})
        print(f"[{tag}] OK pt={usage.get('prompt_tokens')}",
              flush=True)
    except Exception as e:
        print(f"[{tag}] FAIL {type(e).__name__}: {str(e)[:120]}",
              flush=True)


VARIANTS = [
    ("c1_hix16", "hi hi hi hi hi hi hi hi"),
    ("code10w", " ".join(WORDS[:10])),
    ("code12w", " ".join(WORDS[:12])),
    ("ctrl_a", "the quick brown fox jumps over the lazy dog and "
               "then runs away home again"),
    ("ctrl_b", "aaaaaaaaaa bbbbbbbbbb cccccccccc dddddddddd "
               "eeeeeeeeee ffffffffff"),
]

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    for tag, text in VARIANTS:
        if which != "all" and not tag.startswith(which):
            continue
        send(tag, text)
