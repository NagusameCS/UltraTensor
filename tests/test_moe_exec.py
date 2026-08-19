"""MoELayer: lazy top-k layer execution end to end (mini shard)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ultratensor.expert_store import ExpertStore  # noqa: E402
from ultratensor.moe_exec import MoELayer, _ROUTE_SCALE  # noqa: E402

from test_expert_store import _write_mini_moe  # noqa: E402


def test_dense_topk_matches_exhaustive_sort(tmp_path):
    """Regression: argpartition kth must be top_k-1, or the true k-th
    expert can be displaced by the (k+1)-th (found by the independent
    geodessical C router, 2026-08-15). Compare against a full sort."""
    src = tmp_path / "mini8.gguf"
    _write_mini_moe(src, layers=1, E=8, m=32, n=128, bias_scale=0.5)
    st = ExpertStore(src)
    W = st.read_tensor(0, "ffn_gate_inp")            # [E, n]
    bias = st.read_tensor(0, "exp_probs_b")          # [E]
    rng = np.random.default_rng(5)
    for _ in range(32):
        h = rng.standard_normal(128).astype(np.float32)
        ids = st.route_layer(0, h)                    # [1, top_k]
        z = W @ h
        s = np.sqrt(np.log1p(np.exp(z)))
        sel = s + bias                              # bias AFTER softplus
        expect = set(np.argsort(-sel)[:6].tolist())
        got = set(ids[0].tolist())
        assert got == expect, f"selection mismatch: {got} vs {expect}"


def test_route_weights_ignore_bias(tmp_path):
    """Reference semantics (inference/model.py): the expert-prob bias shifts
    the top-k SELECTION only; routing weights use the UNBIASED
    sqrt(softplus) scores. With a large bias, biased weights would be
    nearly uniform - pin the unbiased behavior."""
    src = tmp_path / "mini.gguf"
    _write_mini_moe(src, layers=1, E=4, m=32, n=128, bias_scale=5.0)
    st = ExpertStore(src)
    L = MoELayer(st, 0)
    try:
        h = np.random.default_rng(11).normal(0, 0.5, (1, 128)) \
            .astype(np.float32)
        ids, w = L.route(h)
        W = st.read_tensor(0, "ffn_gate_inp")            # [E, n]
        z = W @ h[0]
        s = np.sqrt(np.log1p(np.exp(z)))                 # unbiased scores
        expect = s[ids[0]]
        expect = expect / expect.sum() * _ROUTE_SCALE
        assert np.allclose(w[0], expect, atol=1e-5)
        # and explicitly NOT the biased version
        bias = st.read_tensor(0, "exp_probs_b")
        sb = np.sqrt(np.log1p(np.exp(z + bias)))
        wrong = sb[ids[0]]
        wrong = wrong / wrong.sum() * _ROUTE_SCALE
        assert not np.allclose(w[0], wrong, atol=0.05)
    finally:
        L.close()


def test_moe_layer_mini(tmp_path):
    src = tmp_path / "mini.gguf"
    _write_mini_moe(src, layers=1, E=4, m=32, n=128)
    st = ExpertStore(src)
    L = MoELayer(st, 0)
    try:
        h = np.random.default_rng(7).normal(0, 0.5, (2, 128)).astype(np.float32)
        ids, w = L.route(h)
        assert ids.shape == (2, 4)     # 4 experts in the mini shard
        assert w.shape == (2, 4)
        # normalized routing weights x route scale
        assert np.allclose(w.sum(axis=-1), _ROUTE_SCALE, atol=1e-4)
        y, info = L(h, timings=True)
        assert y.shape == (2, 128)
        assert np.isfinite(y).all()
        assert info["tokens"] == 2
        # reference: full dense computation over the selected experts
        # (the mini shard has no shared expert, matching MoELayer)
        yref = np.zeros((2, 128), np.float32)
        for b in range(2):
            xb = h[b]
            for slot, e in enumerate(ids[b]):
                gate = st.read_expert(0, "ffn_gate_exps", e)     # (32,128)
                up = st.read_expert(0, "ffn_up_exps", e)
                down = st.read_expert(0, "ffn_down_exps", e)     # (128,32)
                g = gate @ xb
                u = up @ xb
                g = np.minimum(g, 10.0)
                u = np.clip(u, -10.0, 10.0)
                s = g / (1.0 + np.exp(-g))
                yref[b] += w[b, slot] * (down @ (s * u))
        rel = np.abs(y - yref).max() / (np.abs(yref).max() + 1e-6)
        assert rel < 0.02, f"max rel diff vs reference: {rel}"
    finally:
        L.close()
