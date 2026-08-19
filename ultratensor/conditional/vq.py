"""G11 — product-quantized expert residuals.

    W ~= Decode(codes)   with shared per-block codebooks

splits columns into blocks, runs Lloyd k-means per block, and stores
per-row block codes. This turns a dense residual matrix into shared
codebooks + a few bits per row — the review's "shared decoder + expert
codes" primitive, with the codebook-collapse and rare-row hazards made
measurable by the error reporting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PQResult:
    codes: np.ndarray          # [m, n_sub] uint8 code indices
    codebooks: np.ndarray      # [n_sub, K, block]
    bits_per_row: int
    frob_error: float


def _kmeans_lloyd(X: np.ndarray, K: int, iters: int, rng) -> np.ndarray:
    """Lloyd k-means; returns [K, d] centroids (k-means++ init).

    Optimised for large expert matrices: float32 distance math, k-means++
    init on a subsample, early exit on assignment stability.
    """
    n, d = X.shape
    X32 = X.astype(np.float32)
    # k-means++ init on a subsample (memory + speed)
    sub = X32 if n <= 2048 else X32[rng.choice(n, 2048, replace=False)]
    centroids = np.empty((K, d), dtype=np.float32)
    first = int(rng.integers(sub.shape[0]))
    centroids[0] = sub[first]
    for k in range(1, K):
        dist = np.linalg.norm(sub[:, None] - centroids[:k][None], axis=-1)
        dmin = dist.min(axis=1)
        dmin = np.clip(dmin, 1e-30, None)
        p = dmin * dmin
        p /= p.sum()
        centroids[k] = sub[int(rng.choice(sub.shape[0], p=p))]
    prev_assign = None
    for _ in range(iters):
        dist = np.linalg.norm(X32[:, None] - centroids[None], axis=-1)
        assign = dist.argmin(axis=1)
        if prev_assign is not None and np.array_equal(assign, prev_assign):
            break
        prev_assign = assign
        for k in range(K):
            members = X32[assign == k]
            if len(members):
                centroids[k] = members.mean(axis=0)
    return centroids.astype(np.float64)


def product_quantize(
    W: np.ndarray,
    n_sub: int,
    n_bits: int = 8,
    iters: int = 10,
    seed: int = 0,
) -> PQResult:
    """PQ-encode W [m, n]. n_sub blocks of n/n_sub columns each."""
    W = np.asarray(W, dtype=np.float64)
    m, n = W.shape
    if n % n_sub != 0:
        raise ValueError("n must be divisible by n_sub")
    block = n // n_sub
    K = 1 << n_bits
    if K > m:
        raise ValueError("more codebook entries than rows: reduce n_bits")
    rng = np.random.default_rng(seed)

    codes = np.empty((m, n_sub), dtype=np.uint8)
    codebooks = np.empty((n_sub, K, block), dtype=np.float64)
    for b in range(n_sub):
        Xb = W[:, b * block:(b + 1) * block]
        centroids = _kmeans_lloyd(Xb, K, iters, rng)
        dist = np.linalg.norm(Xb[:, None] - centroids[None], axis=-1)
        codes[:, b] = dist.argmin(axis=1)
        codebooks[b] = centroids

    W_hat = pq_reconstruct(codes, codebooks, m)
    err = float(np.sum((W - W_hat) ** 2) / max(np.sum(W * W), 1e-30))
    return PQResult(codes=codes, codebooks=codebooks,
                    bits_per_row=n_sub * n_bits, frob_error=err)


def pq_reconstruct(codes: np.ndarray, codebooks: np.ndarray,
                   m: int | None = None) -> np.ndarray:
    """Decode codes back to a dense matrix [m, n]."""
    codes = np.asarray(codes, dtype=np.int64)
    codebooks = np.asarray(codebooks, dtype=np.float64)
    m = codes.shape[0]
    n_sub, K, block = codebooks.shape
    out = np.empty((m, n_sub * block), dtype=np.float64)
    for b in range(n_sub):
        out[:, b * block:(b + 1) * block] = codebooks[b, codes[:, b]]
    return out


def pq_bits_vs_error(
    W: np.ndarray, n_sub: int, bits=(4, 6, 8), seed: int = 0
) -> dict:
    """Error curve over codebook bits at fixed block count."""
    out = {}
    for b in bits:
        r = product_quantize(W, n_sub, n_bits=b, seed=seed)
        out[b] = {"frob_error": r.frob_error,
                  "params": r.codebooks.size + r.codes.size}
    return out
