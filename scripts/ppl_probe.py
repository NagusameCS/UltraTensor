"""Crash bisect probe: which request shape kills the CPU server?"""
import json
import sys
import urllib.error
import urllib.request

ENDPOINT = "http://127.0.0.1:8780/v1/chat/completions"
CODE = ("Write a Python function that finds the longest palindromic "
        "substring in O(n^2).")


def send(tag, prompt, max_tokens, logprobs):
    payload = {
        "model": "v4",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "logprobs": logprobs,
        "top_logprobs": 5,
    }
    req = urllib.request.Request(
        ENDPOINT, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=240) as r:
            body = r.read().decode("utf-8")
        n = len(json.loads(body)["choices"][0].get("logprobs", {})
                .get("content", []))
        print(f"[{tag}] OK n_logprobs={n}", flush=True)
        return True
    except Exception as e:
        print(f"[{tag}] FAIL {type(e).__name__}: {str(e)[:150]}",
              flush=True)
        return False


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "a":
        # variant A: hi + max_tokens 8
        send("A: hi mt8", "hi", 8, True)
    elif len(sys.argv) > 1 and sys.argv[1] == "b":
        # variant B: code prompt + max_tokens 2
        send("B: code mt2", CODE, 2, True)
    else:
        send("A: hi mt8", "hi", 8, True)
        send("B: code mt2", CODE, 2, True)


if __name__ == "__main__":
    main()
