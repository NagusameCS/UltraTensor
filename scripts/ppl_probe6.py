"""Wait for the sanitizer server, then fire one code12w request."""
import json
import time
import urllib.error
import urllib.request

ENDPOINT = "http://127.0.0.1:8781/v1/chat/completions"
CODE = ("Write a Python function that finds the longest palindromic "
        "substring in O(n^2).")


def main():
    # wait up to 20 min for the server to listen (sanitizer slows load)
    for i in range(240):
        try:
            with urllib.request.urlopen(ENDPOINT.replace("/v1/chat/", "/health"),
                                        timeout=2) as r:
                if r.status == 200:
                    break
        except Exception:
            time.sleep(5)
        if i % 12 == 0:
            print(f"waiting... {i*5}s", flush=True)

    payload = {
        "model": "v4",
        "messages": [{"role": "user", "content": CODE}],
        "max_tokens": 8,
        "temperature": 0.0,
        "logprobs": True,
        "top_logprobs": 5,
    }
    req = urllib.request.Request(
        ENDPOINT, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            print("RESP", r.status, flush=True)
    except Exception as e:
        print(f"FAIL {type(e).__name__}: {str(e)[:200]}", flush=True)


if __name__ == "__main__":
    main()
