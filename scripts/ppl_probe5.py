"""Crash bisect round 5: raw completion endpoint, exact token counts."""
import json
import sys
import urllib.error
import urllib.request

ENDPOINT = "http://127.0.0.1:8780"


def send_raw(tag, n_words, max_tokens=2):
    words = ["apple", "banana", "cherry", "date", "elder", "fig",
             "grape", "honey", "ice", "jam", "kiwi", "lemon",
             "mango", "nect", "olive", "pear", "quince", "raisin",
             "straw", "tanger"]
    prompt = " ".join(words[:n_words])
    payload = {"prompt": prompt, "n_predict": max_tokens,
               "temperature": 0.0, "logprobs": 0}
    req = urllib.request.Request(
        ENDPOINT + "/completion", data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=240) as r:
            body = json.loads(r.read())
        usage = body.get("usage", {})
        print(f"[{tag}] OK pt={usage.get('prompt_tokens')}", flush=True)
    except Exception as e:
        print(f"[{tag}] FAIL {type(e).__name__}: {str(e)[:120]}",
              flush=True)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "series"
    if which == "series":
        for n in (13, 14, 15, 16, 20):
            send_raw(f"raw{n}w", n)
    else:
        n = int(which)
        send_raw(f"raw{n}w", n)
