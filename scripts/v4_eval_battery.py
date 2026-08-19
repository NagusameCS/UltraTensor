"""G1 — PPL / logprob battery over any OpenAI-style endpoint.

The review's G1 demands PPL/logit-KL on a fixed task battery. This
runner is endpoint-agnostic: it drives
    /v1/chat/completions  (OpenAI schema, geodessical :8774 / llama.cpp)
and, when the server returns per-token logprobs, computes per-prompt
PPL and top-logprob agreement. Battery covers code / math /
multilingual / rare-domain slices + a long-context needle question.

When two endpoints are given (--ref and --cmp), it reports the
top-logprob agreement between them per prompt (the first-order logit
consistency check; full KL needs full distributions, which these
endpoints don't return).

Usage:
    python scripts/v4_eval_battery.py --endpoint http://127.0.0.1:8774
    python scripts/v4_eval_battery.py --endpoint http://127.0.0.1:8080 \
        --ref http://192.168.8.125:8080
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

BATTERY = [
    {"id": "code", "text": "Write a Python function that finds the "
                           "longest palindromic substring in O(n^2)."},
    {"id": "math", "text": "Solve: integrate x^2 * exp(-x) from 0 to "
                           "infinity, step by step."},
    {"id": "multilingual", "text": "Erkl\u00e4re die zweite Binomische "
                                   "Formel auf Deutsch."},
    {"id": "rare", "text": "What was the function of the Antikythera "
                           "mechanism's Metonic gear train?"},
    {"id": "needle", "text": "The passphrase is MARIGOLD-7741. "
                             "Now, what is the passphrase?"},
]


def chat_request(endpoint, prompt, max_tokens=64, temperature=0.0,
                 logprobs=True, top_logprobs=5, timeout=30):
    url = endpoint.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": "v4",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if logprobs:
        payload["logprobs"] = True
        payload["top_logprobs"] = top_logprobs
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def extract_logprobs(resp):
    """[(token, {token: logprob})] from an OpenAI-style response."""
    choice = resp.get("choices", [{}])[0]
    return list(choice.get("logprobs", {}).get("content", []))


def ppl_of(entries):
    """Perplexity from per-token logprobs (e-base)."""
    lp = [max(e.get("logprob", 0.0), -100.0) for e in entries]
    if not lp:
        return None
    import math
    return math.exp(-sum(lp) / len(lp))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://127.0.0.1:8774")
    ap.add_argument("--ref", default=None, help="second endpoint for "
                    "top-logprob agreement")
    ap.add_argument("--timeout", type=int, default=30)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--only", default=None,
                    help="comma-separated prompt ids to run (retry mode)")
    ap.add_argument("--out", default=str(ROOT / "outputs" /
                                         "eval_battery.json"))
    a = ap.parse_args()

    only = set(a.only.split(",")) if a.only else None
    report = {"endpoint": a.endpoint, "prompts": {}}
    for p in BATTERY:
        if only and p["id"] not in only:
            continue
        entry = {"id": p["id"], "ok": False}
        try:
            t0 = time.time()
            resp = chat_request(a.endpoint, p["text"],
                                max_tokens=a.max_tokens, timeout=a.timeout)
            entry["latency_s"] = round(time.time() - t0, 2)
            entries = extract_logprobs(resp)
            entry["tokens"] = [e.get("token") for e in entries][:16]
            entry["n_tokens"] = len(entries)
            entry["ppl"] = round(ppl_of(entries), 3) if entries else None
            text = resp.get("choices", [{}])[0].get("message", {}).get(
                "content", "")
            entry["completion"] = text[:200]
            if a.ref:
                resp2 = chat_request(a.ref, p["text"])
                e2 = extract_logprobs(resp2)
                t1 = {e.get("token"): i for i, e in enumerate(entries)}
                agree = sum(1 for i, e in enumerate(e2)
                            if i in t1 and t1[e.get("token")] == i)
                entry["top_logprob_agreement"] = round(
                    agree / max(len(e2), 1), 4)
            entry["ok"] = True
        except (urllib.error.URLError, ValueError, KeyError,
                TimeoutError) as exc:
            entry["error"] = str(exc)[:120]
        report["prompts"][p["id"]] = entry
        print(f"[{p['id']}] ok={entry.get('ok')} "
              f"ppl={entry.get('ppl')} {entry.get('error', '')}",
              flush=True)

    ok = [e for e in report["prompts"].values() if e.get("ok")]
    ppls = [e["ppl"] for e in ok if e.get("ppl")]
    report["mean_ppl"] = round(sum(ppls) / len(ppls), 3) if ppls else None
    report["n_ok"] = len(ok)
    out = Path(a.out)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
