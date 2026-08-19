"""ExpertStore: dispatch-aware MoE reads + per-token IO model.

Synthetic mini-shard tests (Q8_0 experts) always run; the real V4-Pro
shard tests skip when the model is not present.
"""
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ultratensor.expert_store import ExpertStore  # noqa: E402

V4_SHARD1 = ("D:/hyperv4/models/pro/"
             "deepseek-ai-DeepSeek-V4-Pro-Q3_K_M-00001-of-00017.gguf")


def _str_kv(key: str, s: str) -> bytes:
    b = s.encode()
    return (struct.pack("<Q", len(key.encode())) + key.encode() +
            struct.pack("<I", 8) + struct.pack("<Q", len(b)) + b)


def _kv_i(key: str, v: int) -> bytes:
    return (struct.pack("<Q", len(key.encode())) + key.encode() +
            struct.pack("<I", 4) + struct.pack("<I", v))


def _write_mini_moe(path: Path, layers=2, E=4, m=32, n=128, bias_scale=0.1):
    """Mini shard: blk.<L>.ffn_{gate,down,up}_exps.weight as Q8_0 (n,m,E),
    plus the dense router blk.<L>.ffn_gate_inp.weight (n,E) F32 and
    blk.<L>.exp_probs_b.bias (E,) F32 (deepseek4-style, 0 hash layers)."""
    rng = np.random.default_rng(42)
    blob = bytearray()
    tensors = []          # (name, dims, ttype, data)
    for L in range(layers):
        router = rng.standard_normal((E, n)).astype(np.float32)
        tensors.append((f"blk.{L}.ffn_gate_inp.weight", (n, E), 0,
                        router.tobytes()))
        tensors.append((f"blk.{L}.exp_probs_b.bias", (E,), 0,
                        (rng.standard_normal(E) * bias_scale)
                        .astype(np.float32).tobytes()))
        for kind in ("gate", "down", "up"):
            if kind == "down":
                # FFN down: input = intermediate (32), output = hidden (128)
                W3 = rng.standard_normal((E, n, m)).astype(np.float32)
                dims3 = (m, n, E)
            else:
                W3 = rng.standard_normal((E, m, n)).astype(np.float32)
                dims3 = (n, m, E)
            data = bytearray()
            rows, cols = W3.shape[1], W3.shape[2]
            for e in range(E):
                for r in range(rows):
                    for b0 in range(0, cols, 32):
                        block = W3[e, r, b0:b0 + 32]
                        d = np.float16(np.max(np.abs(block)) / 127.0)
                        q = np.round(block.astype(np.float64) / float(d)) \
                            .clip(-128, 127)
                        data += struct.pack("<e", d) + q.astype(np.int8).tobytes()
            tensors.append((f"blk.{L}.ffn_{kind}_exps.weight",
                            dims3, 8, bytes(data)))
    kvs = (_str_kv("general.architecture", "llama")
           + _str_kv("general.name", "mini-moe")
           + _kv_i("general.alignment", 32))
    hdr = (b"GGUF" + struct.pack("<I", 3) +
           struct.pack("<Q", len(tensors)) + struct.pack("<Q", 3) + kvs)
    off = 0
    for name, dims, ttype, data in tensors:
        nb = name.encode()
        hdr += (struct.pack("<Q", len(nb)) + nb +
                struct.pack("<I", len(dims)) +
                struct.pack("<" + "Q" * len(dims), *dims) +
                struct.pack("<I", ttype) + struct.pack("<Q", off))
        off += len(data)
        blob += data
    hdr += b"\0" * ((32 - len(hdr) % 32) % 32)
    with open(path, "wb") as f:
        f.write(hdr + bytes(blob))
    return tensors


def test_mini_moe_inventory_and_reads(tmp_path):
    src = tmp_path / "mini.gguf"
    _write_mini_moe(src)
    st = ExpertStore(src)
    assert st.layers() == [0, 1]
    assert len(st.tensors) == 10   # 3 experts + router + bias per layer
    assert st.expert_shape(0, "ffn_gate_exps") == (128, 32, 4)
    assert st.router_shape(0) == (128, 4)
    W = st.read_expert(0, "ffn_gate_exps", 2)
    assert W.shape == (32, 128)
    assert np.isfinite(W).all()
    assert abs(float(W.std()) - 1.0) < 0.3
    R = st.read_tensor(0, "ffn_gate_inp")
    assert R.shape == (4, 128)
    s = st.summary()
    assert "10 tensors" in s


def test_mini_moe_route(tmp_path):
    src = tmp_path / "mini.gguf"
    _write_mini_moe(src)
    st = ExpertStore(src)
    hidden = np.random.default_rng(1).normal(0, 1, (128,)).astype(np.float32)
    ids = st.route_layer(0, hidden, top_k=2)
    assert ids.shape == (1, 2)
    assert set(ids[0].tolist()).issubset({0, 1, 2, 3})
    # route must match sqrt(softplus(z)) + bias top-k (real V4 semantics)
    W = st.read_tensor(0, "ffn_gate_inp")
    bias = st.read_tensor(0, "exp_probs_b")
    z = W @ hidden + bias
    scores = np.sqrt(np.log1p(np.exp(z)))
    assert set(ids[0].tolist()) == set(np.argsort(-scores)[:2].tolist())
    # batched routing
    H = np.random.default_rng(2).normal(0, 1, (5, 128)).astype(np.float32)
    idsB = st.route_layer(0, H, top_k=2)
    assert idsB.shape == (5, 2)


def test_io_model(tmp_path):
    src = tmp_path / "mini.gguf"
    _write_mini_moe(src)
    st = ExpertStore(src)
    m1 = st.io_model(top_k=2, router_amortized=True)
    m2 = st.io_model(top_k=2, router_amortized=False)
    routed = sum(st._expert_bytes(0, k) * 2 for k in
                 ("ffn_gate_exps", "ffn_down_exps", "ffn_up_exps"))
    assert m1["per_layer"][0]["routed"] == pytest.approx(routed)
    # amortized: router bytes not in per-token total; unamortized: they are
    assert m1["per_layer"][0]["router"] == 0.0
    assert m2["per_layer"][0]["router"] > 0.0
    assert m2["bytes_per_token"] > m1["bytes_per_token"]
    assert m1["router_total_once"] > 0


@pytest.mark.skipif(not Path(V4_SHARD1).exists(),
                    reason="V4-Pro shard 1 not present")
def test_v4_shard1_inventory():
    st = ExpertStore(V4_SHARD1)
    assert st.layers()[:3] == [0, 1, 2]
    assert st.n_hash_layers == 3
    assert st.expert_shape(0, "ffn_gate_exps") == (7168, 3072, 384)
    # hash layers still carry the dense router weights (unused for routing)
    assert st.router_shape(0) == (7168, 384)
    m = st.io_model(top_k=6)
    # ~0.6 GiB/token routed + shexp on this 3-layer shard slice
    assert 0.3e9 < m["routed_total"] < 3e9
    # the one-time router reads are tiny vs the per-token routed IO
    assert m["router_total_once"] < m["routed_total"] / 10
    # shexp present for full layers
    assert m["shexp_total"] > 0


@pytest.mark.skipif(not Path(V4_SHARD1).exists(),
                    reason="V4-Pro shard 1 not present")
def test_v4_shard1_routing():
    st = ExpertStore(V4_SHARD1)
    # hash-routed layer 0: deterministic table lookup
    ids = st.route_layer(0, np.zeros((2, 7168), np.float32),
                         token_ids=[0, 12345], top_k=6)
    assert ids.shape == (2, 6)
    table = st.read_tensor(0, "ffn_gate_tid2eid")
    assert table.shape == (129280, 6)
    assert ids[0].tolist() == table[0][:6].tolist()
    assert ids[1].tolist() == table[12345][:6].tolist()
    # dense router weights + bias: hash layers (0-2) carry no bias (matches
    # the HF source: Gate.bias = None for hash layers); blk.3 does
    W = st.read_tensor(0, "ffn_gate_inp")
    assert W.shape == (384, 7168)
    bias = st.read_tensor(3, "exp_probs_b")
    assert bias.shape == (384,)


@pytest.mark.skipif(not Path(V4_SHARD1).exists(),
                    reason="V4-Pro shard 1 not present")
def test_v4_shard1_read_expert():
    st = ExpertStore(V4_SHARD1)
    W = st.read_expert(0, "ffn_gate_exps", 0)
    assert W.shape == (3072, 7168)
    assert not np.isnan(W).any()
    assert 0.001 < W.std() < 0.2


@pytest.mark.skipif(not Path(V4_SHARD1).exists(),
                    reason="V4-Pro shard 1 not present")
def test_v4_expert_gemv_c_matches_numpy():
    """C ut_expert_gemv (Q3_K decode) must match numpy ground truth."""
    from ultratensor.kernels import ExpertGEMV
    from ultratensor.expert_store import ExpertStore

    st = ExpertStore(V4_SHARD1)
    e = 0
    W = st.read_expert(0, "ffn_gate_exps", e)          # (3072, 7168)
    x = np.random.default_rng(3).normal(0, 1, (7168,)).astype(np.float32)
    ref = W @ x

    cg = ExpertGEMV()
    try:
        cg.open(V4_SHARD1, "blk.0.ffn_gate_exps.weight")
        assert cg.shape == (7168, 3072, 384)
        got = cg.gemv(e, x)
        assert got.shape == (3072,)
        rel = np.abs(got - ref) / (np.abs(ref) + 1e-3)
        assert rel.max() < 1e-3, f"max rel err {rel.max()}"
        # transpose path: y[n] = W^T @ xm
        xm = np.random.default_rng(4).normal(0, 1, (3072,)).astype(np.float32)
        refT = W.T @ xm
        gotT = cg.gemv(e, xm, transpose=True)
        relT = np.abs(gotT - refT) / (np.abs(refT) + 1e-3)
        assert relT.max() < 1e-3, f"max rel err T {relT.max()}"
    finally:
        cg.close()


@pytest.mark.skipif(not Path(V4_SHARD1).exists(),
                    reason="V4-Pro shard 1 not present")
def test_v4_expert_gemv_down_q5k_matches_numpy():
    """C ut_expert_gemv Q5_K decode (down_exps) vs numpy ground truth."""
    from ultratensor.kernels import ExpertGEMV
    from ultratensor.expert_store import ExpertStore

    st = ExpertStore(V4_SHARD1)
    e = 0
    W = st.read_expert(0, "ffn_down_exps", e)          # (7168, 3072) Q5_K
    x = np.random.default_rng(5).normal(0, 1, (3072,)).astype(np.float32)
    ref = W @ x

    cg = ExpertGEMV()
    try:
        cg.open(V4_SHARD1, "blk.0.ffn_down_exps.weight")
        assert cg.shape == (3072, 7168, 384)
        got = cg.gemv(e, x)
        assert got.shape == (7168,)
        rel = np.abs(got - ref) / (np.abs(ref) + 1e-3)
        assert rel.max() < 1e-3, f"max rel err {rel.max()}"
    finally:
        cg.close()


@pytest.mark.skipif(not Path(V4_SHARD1).exists(),
                    reason="V4-Pro shard 1 not present")
def test_v4_expert_gemv_down_q4k_matches_numpy():
    """C ut_expert_gemv Q4_K decode (blk.3 down_exps) vs numpy."""
    from ultratensor.kernels import ExpertGEMV
    from ultratensor.expert_store import ExpertStore

    st = ExpertStore(V4_SHARD1)
    W = st.read_expert(3, "ffn_down_exps", 0)          # (7168, 3072) Q4_K
    x = np.random.default_rng(6).normal(0, 1, (3072,)).astype(np.float32)
    ref = W @ x

    cg = ExpertGEMV()
    try:
        cg.open(V4_SHARD1, "blk.3.ffn_down_exps.weight")
        assert cg.shape == (3072, 7168, 384)
        got = cg.gemv(0, x)
        rel = np.abs(got - ref) / (np.abs(ref) + 1e-3)
        assert rel.max() < 1e-3, f"max rel err {rel.max()}"
    finally:
        cg.close()