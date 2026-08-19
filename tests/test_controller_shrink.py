"""Tests for the G9 shrink helpers in scripts/v4_controller_shrink.py."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import v4_controller_shrink as cs  # noqa: E402


def test_topk_agreement():
    pred = np.array([[0.9, 0.8, 0.1], [0.1, 0.2, 0.3]])
    true = np.array([[0.8, 0.9, 0.2], [0.3, 0.2, 0.1]])
    # row 0: top-2 {0,1} == {0,1} -> 1.0; row 1: {1,2} vs {0,1} -> 0.5
    assert cs.topk_agreement(pred, true, 2) == 0.75


def test_topk_agreement_partial():
    rng = np.random.default_rng(0)
    pred = rng.normal(size=(50, 8))
    true = rng.normal(size=(50, 8))
    a = cs.topk_agreement(pred, true, 3)
    assert 0.0 <= a <= 1.0


def test_labels_one_hot_top_k():
    T = np.array([[0.1, 0.5, 0.3, 0.2]])
    Y = cs.labels(T, 2)
    assert Y.shape == T.shape
    assert Y[0].sum() == 2
    assert Y[0, 1] == 1.0 and Y[0, 2] == 1.0


def test_svd_truncate_rank():
    rng = np.random.default_rng(1)
    W = rng.normal(size=(20, 30))
    W4 = cs.svd_truncate(W, 4)
    assert W4.shape == W.shape
    assert np.linalg.matrix_rank(W4) == 4
    # full rank returns a copy equal to W
    assert np.allclose(cs.svd_truncate(W, 20), W)


def test_factored_ridge_matches_dense():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(16, 64))
    T = rng.normal(size=(16, 10))
    Y = cs.labels(T, 3)
    # dense closed form
    Wc = np.linalg.solve(X.T @ X + np.eye(64), X.T @ Y)
    # factored closed form (same predictions by construction)
    B = np.linalg.solve(X @ X.T + np.eye(16), Y)
    assert np.allclose((X @ X.T) @ B, X @ Wc, atol=1e-8)
