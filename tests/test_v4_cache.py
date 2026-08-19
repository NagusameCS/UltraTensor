"""Tests for the cached loader used by the numpy reference pipeline."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import numpy as np  # noqa: E402

from v4_cache import cached_load  # noqa: E402


def test_cached_load_returns_same_and_counts():
    calls = []

    def real(st, name, rows=None):
        calls.append((name, rows))
        n = 8
        return np.full((n, 4), float(len(calls)))

    st = object()
    loader = cached_load(real, max_bytes=1 << 20)
    a = loader(st, "blk.0.attn_q_a.weight")
    b = loader(st, "blk.0.attn_q_a.weight")
    assert a is b or np.array_equal(a, b)
    c = loader(st, "blk.0.attn_q_a.weight", rows=[0, 1])
    assert not np.array_equal(c, a)
    stats = loader.stats()
    assert stats == {"hits": 1, "misses": 2, "entries": 2}


def test_cached_load_evicts_oldest():
    calls = {"n": 0}

    def real(st, name, rows=None):
        calls["n"] += 1
        return np.full((16, 16), float(calls["n"]))  # 2 KB each

    loader = cached_load(real, max_bytes=8 * 1024)  # holds ~4 entries
    for i in range(6):
        loader(object(), f"t{i}")
    assert len(loader.cache) <= 4
    # newest entries retained
    assert ("t5",) in loader.cache or "t5" in loader.cache
