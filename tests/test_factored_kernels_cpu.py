"""Phase 2/3 CPU tests: portable C factored GEMV + container executor."""
from pathlib import Path

import numpy as np
import pytest

from ultratensor.kernels import (FactoredExec, factored_gemv_cpu,
                                 numpy_reference)
from ultratensor.quant import uq4_quantize


def test_cpu_twin_matches_numpy():
    rng = np.random.default_rng(7)
    m, k, n = 256, 16, 2048
    U = rng.standard_normal((m, k)).astype(np.float16)
    C = rng.standard_normal((k, n)).astype(np.float32)
    scales, packed = uq4_quantize(C, block=32)
    x = rng.standard_normal(n).astype(np.float32)
    y = factored_gemv_cpu(U, scales, packed, x)
    y_ref = numpy_reference(U, scales, packed, x)
    rel = np.abs(y - y_ref).max() / np.abs(y_ref).max()
    assert rel < 1e-4


def test_cpu_twin_hand_calc():
    # k=1, n=32: codes 10 -> value scale*2; scale=0.5; U=[2]
    U = np.array([[2.0]], np.float16)
    scales = np.array([[0.5]], np.float32)
    packed = np.full((1, 16), 10 | (10 << 4), np.uint8)
    x = np.ones(32, np.float32)
    y = factored_gemv_cpu(U, scales, packed, x)
    assert y[0] == pytest.approx(2.0 * (32 * 0.5 * 2.0), rel=1e-5)


def test_c_executor_loads_container(tmp_path):
    """Phase 3: the C loader parses the factored GGUF and executes it."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
    from ultratensor.gguf_factored import (read_factored_gguf, reconstruct,
                                           write_factored_gguf)
    from test_gguf_factored import _write_source

    rng = np.random.default_rng(21)
    A = rng.standard_normal((64, 12)).astype(np.float32)
    B = rng.standard_normal((12, 128)).astype(np.float32)
    W = (A @ B + 0.01 * rng.standard_normal((64, 128))).astype(np.float32)
    F = np.zeros((8, 8), np.float32)
    src = tmp_path / "src.gguf"
    out = tmp_path / "out.gguf"
    _write_source(src, W, F)
    write_factored_gguf(src, out, patterns=["t1"], rank=12)

    x = rng.standard_normal(128).astype(np.float32)
    exec = FactoredExec().load(str(out), "t1")
    try:
        y = exec.gemv(x)
    finally:
        exec.close()

    manifest, tensors = read_factored_gguf(out)
    W_hat = reconstruct(manifest, tensors, "t1.factored_C")
    y_ref = W_hat @ x
    rel = np.abs(y - y_ref).max() / np.abs(y_ref).max()
    assert rel < 1e-4


def test_c_executor_loads_3d_experts(tmp_path):
    """C executor handles expert stacks (MoE accumulation)."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
    from ultratensor.gguf_factored import (read_factored_gguf, reconstruct,
                                           write_factored_gguf)
    from test_gguf_factored import _write_source3

    rng = np.random.default_rng(22)
    E, m, n = 4, 32, 128
    W3 = np.stack([
        rng.standard_normal((m, 8)).astype(np.float32)
        @ rng.standard_normal((8, n)).astype(np.float32)
        + 0.01 * rng.standard_normal((m, n)).astype(np.float32)
        for _ in range(E)])
    src = tmp_path / "src.gguf"
    out = tmp_path / "out.gguf"
    _write_source3(src, W3)
    write_factored_gguf(src, out, patterns=["t1"], rank=8)

    x = rng.standard_normal(n).astype(np.float32)
    exec = FactoredExec().load(str(out), "t1")
    try:
        y = exec.gemv(x)
    finally:
        exec.close()

    manifest, tensors = read_factored_gguf(out)
    W_hat = reconstruct(manifest, tensors, "t1.factored_C")
    y_ref = W_hat.sum(axis=0) @ x  # sum over experts
    rel = np.abs(y - y_ref).max() / np.abs(y_ref).max()
    assert rel < 1e-4
