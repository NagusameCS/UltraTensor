"""Memoized weight loading for the numpy reference pipeline.

vs.load_any re-reads and re-dequantizes big attention tensors on every
token; phase B spends most of its wall time there. This module wraps it
in a bounded cache so repeated loads within one process are free. Safe
on this machine: float64 upcasts of ~6 tensors x 4 layers fit in ~2 GB.

Usage (in scripts that run sequential forwards):
    import v4_ref_serve as vs
    from v4_cache import cached_load
    vs.load_any = cached_load(vs.load_any, max_bytes=4 << 30)
"""

from __future__ import annotations

import sys
from collections import OrderedDict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def cached_load(real_load, max_bytes: int = 4 << 30):
    """Return a load_any-compatible function with an LRU cache."""
    cache: OrderedDict = OrderedDict()
    budget = max_bytes
    misses = hits = 0

    def wrapper(st, name, rows=None):
        nonlocal budget, misses, hits
        key = name if rows is None else (name, tuple(rows))
        if key in cache:
            hits += 1
            cache.move_to_end(key)
            return cache[key]
        misses += 1
        value = real_load(st, name, rows=rows)
        size = getattr(value, "nbytes", 0)
        if 0 < size <= max_bytes:
            cache[key] = value
            budget -= size
            while budget < 0 and len(cache) > 1:
                _, dropped = cache.popitem(last=False)
                budget += getattr(dropped, "nbytes", 0)
        return value

    wrapper.cache = cache        # type: ignore[attr-defined]
    wrapper.stats = lambda: {"hits": hits, "misses": misses,  # type: ignore[attr-defined]
                             "entries": len(cache)}
    return wrapper
