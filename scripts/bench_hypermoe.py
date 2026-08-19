"""Hyper-MoE benchmark harness: measure any built specialist.

Runs one generation through the dispatcher and reports wall time and
tokens/sec for the selected backend.  This is the acceptance battery
every specialist must pass before it enters the registry.

Usage:
    python scripts/bench_hypermoe.py --prompt "def fib" --n 4
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.hypermoe_serve import Dispatcher, REGISTRY  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", default=REGISTRY)
    ap.add_argument("--prompt", default="def fib")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--threads", type=int, default=6)
    a = ap.parse_args()

    d = Dispatcher(a.registry)
    spec = d.resolve(a.prompt)
    print(f"route: {a.prompt!r} -> {spec['id']} "
          f"(domain={spec['domain']}, backend={spec['backend']})",
          flush=True)
    t0 = time.time()
    result = d.generate(a.prompt, a.n, a.threads)
    dt = time.time() - t0
    ntok = max(1, a.n)
    print(f"wall: {dt:.1f}s | {dt / ntok:.1f} s/tok | "
          f"{ntok / dt:.2f} tok/s | rc={result.get('rc')}", flush=True)
    print(result.get("raw", "")[-400:])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
