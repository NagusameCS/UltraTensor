"""Tests for G11 product quantization."""

import numpy as np
import pytest

from ultratensor.conditional.vq import (
    pq_bits_vs_error,
    pq_reconstruct,
    product_quantize,
)


def _clustered_matrix(seed=0, m=256, n=64, n_clusters=4):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(n_clusters, n)) * 3.0
    W = np.empty((m, n))
    for i in range(m):
        W[i] = centers[i % n_clusters] + 0.05 * rng.normal(size=n)
    return W


def test_pq_exact_on_clustered_data():
    W = _clustered_matrix()
    r = product_quantize(W, n_sub=8, n_bits=4, iters=20, seed=0)
    assert r.frob_error < 1e-3
    assert r.codes.shape == (256, 8)
    rec = pq_reconstruct(r.codes, r.codebooks)
    assert rec.shape == W.shape


def test_pq_more_bits_less_error():
    W = _clustered_matrix(seed=1)
    curve = pq_bits_vs_error(W, n_sub=8, bits=(4, 6, 8))
    errs = [curve[b]["frob_error"] for b in (4, 6, 8)]
    assert errs[0] >= errs[1] >= errs[2]


def test_pq_validates_shapes():
    W = np.random.default_rng(0).normal(size=(32, 9))
    with pytest.raises(ValueError):
        product_quantize(W, n_sub=4)   # 9 not divisible by 4
    with pytest.raises(ValueError):
        product_quantize(W, n_sub=3, n_bits=8)  # K=256 > m=32


def test_pq_roundtrip_error_bounded():
    W = np.random.default_rng(2).normal(size=(64, 32))
    r = product_quantize(W, n_sub=4, n_bits=4, seed=0)   # K=16 < 64 rows
    assert 0.0 < r.frob_error < 1.0
    assert r.bits_per_row == 4 * 4
