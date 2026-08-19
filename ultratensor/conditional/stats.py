"""G1/G8 evaluation tooling (port of HyperTensor scripts/ablation_utils.py).

- ``bootstrap_ci``          : percentile CIs for measured numbers
- ``fine_k_sweep``          : cliff-vs-ramp detection around a rank k*
- ``rank_ablation``         : per-matrix rank-importance ranking
- ``intrinsic_dim_compare`` : PCA-95 / TwoNN / Levina-Bickel MLE
                              (numpy-only, no sklearn)
- ``sink_ablation``         : sink-exemption sweep
- ``eviction_ablation``     : LRU / LFU / jury-weighted / random cache
                              eviction comparison

Everything is numpy-only and seeded, so measurements in the published
docs can carry honest confidence intervals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


@dataclass
class BootstrapResult:
    mean: float
    ci_lower: float
    ci_upper: float
    ci_level: float = 0.95
    n_resamples: int = 10000
    method: str = "percentile"


def bootstrap_ci(
    data,
    ci_level: float = 0.95,
    n_resamples: int = 10000,
    seed: int = 42,
) -> BootstrapResult:
    """Percentile bootstrap CI (vectorised resampling)."""
    rng = np.random.default_rng(seed)
    d = np.asarray(data, dtype=np.float64)
    n = d.size
    if n == 0:
        raise ValueError("empty data")
    idx = rng.integers(0, n, size=(n_resamples, n))
    means = d[idx].mean(axis=1)
    alpha = (1.0 - ci_level) / 2.0
    return BootstrapResult(
        mean=float(d.mean()),
        ci_lower=float(np.percentile(means, 100 * alpha)),
        ci_upper=float(np.percentile(means, 100 * (1 - alpha))),
        ci_level=ci_level,
        n_resamples=n_resamples,
    )


def fine_k_sweep(
    k_star: int = 1024,
    window: int = 200,
    step: int = 25,
    throughput_fn: Optional[Callable[[int], float]] = None,
) -> dict:
    """Sweep ranks near k* and classify the drop as cliff or ramp."""
    if throughput_fn is None:
        def throughput_fn(k):
            base, peak = 35.0, 38.0
            working_set_mb = 2 * 4096 * k / (1024 * 1024)
            saturation = 1.0 / (1.0 + np.exp((working_set_mb - 32.0 * 0.8) / 2.0))
            return base + (peak - base) * saturation

    ks = list(range(max(k_star - window, 64), k_star + window + 1, step))
    tps = np.asarray([throughput_fn(k) for k in ks], dtype=np.float64)
    grad = np.gradient(tps)
    is_cliff = bool(np.max(np.abs(grad)) > 3 * np.mean(np.abs(grad))) if grad.size > 1 else False
    return {
        "k_star": k_star,
        "k_values": ks,
        "throughput": tps.tolist(),
        "transition_type": "cliff" if is_cliff else "ramp",
        "max_gradient": float(np.max(np.abs(grad))) if grad.size > 1 else 0.0,
        "optimal_k": int(ks[int(np.argmax(tps))]) if tps.size else k_star,
    }


def rank_ablation(
    rank_budget: int,
    matrices: dict,
    quality_fn: Callable[[dict], float],
) -> dict:
    """Drop one matrix-class at a time and rank importance by quality loss."""
    baseline = quality_fn(matrices)
    results = {}
    for name in matrices:
        ablated = matrices.copy()
        freed = ablated[name]
        ablated[name] = 0
        remaining = [n for n in ablated if n != name]
        total = sum(ablated[n] for n in remaining)
        if total > 0:
            for n in remaining:
                ablated[n] += int(freed * ablated[n] / total)
        quality = quality_fn(ablated)
        results[name] = {
            "quality": quality,
            "delta": quality - baseline,
            "importance": abs(quality - baseline) / max(baseline, 1e-10),
        }
    return {
        "baseline_quality": baseline,
        "rank_budget": rank_budget,
        "per_matrix": results,
        "ranking": sorted(results, key=lambda n: results[n]["importance"],
                          reverse=True),
    }


def intrinsic_dim_compare(data: np.ndarray, max_dim: int = 100) -> dict:
    """PCA-95, TwoNN (Facco et al.), and Levina-Bickel MLE estimators."""
    X = np.asarray(data, dtype=np.float64)
    n, d = X.shape
    out: dict = {}

    # PCA 95%
    Xc = X - X.mean(axis=0)
    cov = Xc.T @ Xc / max(n, 1)
    evals = np.linalg.eigvalsh(cov)[::-1]
    evals = np.clip(evals, 0.0, None)
    total = evals.sum()
    cum = np.cumsum(evals) / max(total, 1e-30)
    out["pca_95"] = min(int(np.searchsorted(cum, 0.95)) + 1, max_dim)

    # pairwise distances (memory-light: chunked)
    D = np.zeros((n, n))
    for i in range(0, n, 256):
        j0 = i + 256
        D[i:j0] = np.linalg.norm(X[i:j0, None] - X[None, :], axis=-1)
    np.fill_diagonal(D, np.inf)
    d1 = np.min(D, axis=1)
    d2 = np.partition(D, 1, axis=1)[:, 1]   # second-nearest neighbour
    mu = d2 / (d1 + 1e-10)
    mu = mu[mu > 1.0 + 1e-10]
    # Facco et al.: d ~= n / sum(log(mu_i))
    out["twonn"] = min(
        int(round(mu.size / np.log(mu).sum())) if mu.size else 1, max_dim)

    # Levina-Bickel MLE with k=20
    k = min(20, n - 1)
    if k >= 2:
        Tk = np.sort(D, axis=1)[:, k - 1]
        d_local = np.zeros(n)
        for i in range(n):
            ratios = np.log(Tk[i] / (np.sort(D[i])[1:k] + 1e-10))
            ratios = ratios[ratios > 0]
            if ratios.size:
                d_local[i] = (ratios.size - 1) / ratios.sum()
        pos = d_local[d_local > 0]
        out["mle_levina_bickel"] = min(int(round(np.median(pos))) if pos.size else 1,
                                       max_dim)
    else:
        out["mle_levina_bickel"] = 1
    return out


def sink_ablation(
    k: int,
    T_values=(0, 8, 16, 32, 64),
    error_fn: Optional[Callable[[int, int], float]] = None,
) -> dict:
    """Reconstruction error with/without sink-channel exemption."""
    if error_fn is None:
        def error_fn(k, T):
            base = 0.5 * np.exp(-k / 256) + 0.05
            benefit = T / (T + 16) * 0.3 * np.exp(-k / 512)
            return base - benefit

    results = {}
    t0_err = error_fn(k, 0)
    for T in T_values:
        err = error_fn(k, T)
        results[f"T={T}"] = {
            "error": err,
            "improvement_vs_T0": t0_err - err if T > 0 else 0.0,
        }
    return {
        "k": k,
        "results": results,
        "best_T": min(T_values, key=lambda t: results[f"T={t}"]["error"]),
    }


def eviction_ablation(
    n_queries: int = 1000,
    cache_size: int = 100,
    policies=("LRU", "LFU", "jury_weighted", "random"),
    seed: int = 42,
) -> dict:
    """Cache eviction policy comparison on a Zipf query stream."""
    rng = np.random.default_rng(seed)
    n_unique = max(n_queries // 4, cache_size)
    popularity = 1.0 / np.arange(1, n_unique + 1)
    popularity /= popularity.sum()
    queries = rng.choice(n_unique, size=n_queries, p=popularity)
    jury_scores = rng.uniform(0.5, 1.0, n_unique)

    results = {}
    for policy in policies:
        cache: dict = {}   # item -> [last_access, count, jury]
        hits = 0
        for t, q in enumerate(queries):
            q = int(q)
            if q in cache:
                hits += 1
                e = cache[q]
                cache[q] = [t, e[1] + 1, e[2]]
            else:
                if len(cache) >= cache_size:
                    if policy == "LRU":
                        victim = min(cache, key=lambda x: cache[x][0])
                    elif policy == "LFU":
                        victim = min(cache, key=lambda x: cache[x][1])
                    elif policy == "jury_weighted":
                        # keep high-confidence items: evict lowest jury
                        victim = min(cache, key=lambda x: cache[x][2])
                    else:  # random
                        victim = int(rng.choice(list(cache)))
                    del cache[victim]
                cache[q] = [t, 1, float(jury_scores[q])]
        results[policy] = {"hits": hits, "hit_rate": hits / n_queries}
    return results
