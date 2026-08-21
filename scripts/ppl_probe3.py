"""Crash bisect round 3: length cliff on the code prompt."""
import json
import sys
import urllib.error
import urllib.request

ENDPOINT = "http://127.0.0.1:8780/v1/chat/completions"
CODE = "Write a Python function that finds the longest palindromic " \
       "substring in O(n^2)."
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
        print(f"[{tag}] OK pt={usage.get('prompt_tokens')}", flush=True)
    except Exception as e:
        print(f"[{tag}] FAIL {type(e).__name__}: {str(e)[:120]}", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "len"
    if mode == "len":
        for k in (4, 6, 8, 10, 12, 14, 16, len(WORDS)):
            text = " ".join(WORDS[:k])
            send(f"code{k}w", text)
    elif mode == "tail":
        send("tail_O_n2", "find a function " + "O(n^2)")
    elif mode == "plain":
        # raw completion endpoint, no chat template
        import urllib.parse
        url = "http://127.0.0.1:8780/completion"
        payload = {"prompt": CODE, "n_predict": 2, "temperature": 0.0}
        req = urllib.request.Request(
            url, data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=240) as r:
                print("[plain] OK", flush=True)
        except Exception as e:
            print(f"[plain] FAIL {type(e).__name__}: {str(e)[:120]}",
                  flush=True)
