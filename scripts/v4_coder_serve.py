"""V4-Coder serve: keep64 -> tokens via OUR runtime (numpy oracle).

The designed serve path for the extracted coder: ExpertStore streams
per routed expert (max ~6 experts x 3 tensors resident, not whole
layers), so a 30-32GB machine can generate from the 156GB keep64 via
mmap.  This mirrors scripts/v4_ref_gen.py (the validated K/M oracle)
but runs against the keep64 single-file coder build.

Usage:
    python scripts/v4_coder_serve.py --prompt "def fib" --n 8

First token is the smoke test (61 layers); expect minutes/token on a
laptop.  Output goes to outputs/v4coder_tokens.txt.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from tokenizers import Tokenizer  # noqa: E402

import v4_ref_serve as vs  # noqa: E402
from v4_ref_gen import BlockGen, HC, NLAYERS, head_logits  # noqa: E402

MODEL = "Y:/models/coder/DeepSeek-V4-Coder-keep64-00001-of-00001.gguf"
TOK_JSON = str(ROOT / "outputs" / "v4_tokenizer.json")


def sample(logits: np.ndarray, temp: float) -> int:
    if temp <= 0:
        return int(np.argmax(logits))
    l = logits.astype(np.float64) / temp
    l = l - l.max()
    p = np.exp(l)
    p = p / p.sum()
    return int(np.random.choice(len(p), p=p))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--n", type=int, default=8,
                    help="tokens to generate")
    ap.add_argument("--temp", type=float, default=1.0,
                    help="0 = greedy")
    ap.add_argument("--tokenizer", default=TOK_JSON)
    ap.add_argument("--persistent-moe", action="store_true",
                    help="keep one MoELayer per layer open across tokens "
                         "(fast; costs several GB of resident shexp)")
    a = ap.parse_args()

    tok = Tokenizer.from_file(a.tokenizer)
    st = vs.ExpertStore(a.model)
    ids = tok.encode(a.prompt).ids
    if not ids:
        print("empty prompt encoding")
        return 1
    print(f"prompt tokens: {len(ids)}; generating {a.n} tokens from "
          f"{a.model}", flush=True)

    moe_cache = {}
    if a.persistent_moe:
        print("building persistent MoELayers (61)...", flush=True)
        for L in range(NLAYERS):
            moe_cache[L] = vs.MoELayer(st, L)

    tokens = list(ids)
    blocks = [BlockGen(st, L) for L in range(NLAYERS)]
    t0 = time.time()

    # Prefill: run every prompt token through all 61 layers so each
    # BlockGen's window ring + compressed rows are populated exactly as
    # in contiguous decode. Generation previously started at p=len-1,
    # which left cpr[] holes on ratio-4 layers (KeyError: 0).
    for pos, tid in enumerate(ids):
        h = vs.load_any(st, "token_embd.weight", rows=[tid])[0]
        x = np.tile(h, (HC, 1)).astype(np.float32)
        for L, bg in enumerate(blocks):
            if a.persistent_moe:
                ml = moe_cache[L]
            else:
                ml = vs.MoELayer(st, L)
            x = bg.step(x, pos, tid, ml)
            if not a.persistent_moe:
                ml.close()
                del ml
    print(f"prefill {len(ids)} tokens done ({time.time() - t0:.1f}s)",
          flush=True)

    for p in range(a.n):
        t_step = time.time()
        cur = tokens[-1]
        h = vs.load_any(st, "token_embd.weight", rows=[cur])[0]
        x = np.tile(h, (HC, 1)).astype(np.float32)
        for L, bg in enumerate(blocks):
            if a.persistent_moe:
                ml = moe_cache[L]
            else:
                ml = vs.MoELayer(st, L)
            x = bg.step(x, p + len(ids) - 1, cur, ml)
            if not a.persistent_moe:
                ml.close()
                del ml
        logits = head_logits(st, x)
        nxt = sample(logits, a.temp)
        tokens.append(nxt)
        dt = time.time() - t_step
        print(f"token {p+1}/{a.n}: {nxt} ({dt:.1f}s, "
              f"max logit {logits.max():.2f})", flush=True)

    elapsed = time.time() - t0
    gen = tokens[len(ids):]
    text = tok.decode(gen)
    print(f"generated {len(gen)} tokens in {elapsed:.1f}s "
          f"({elapsed / max(len(gen), 1):.1f} s/tok)")
    print(f"text: {text!r}")
    (ROOT / "outputs" / "v4coder_tokens.txt").write_text(
        " ".join(str(t) for t in gen) + "\n" + text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
