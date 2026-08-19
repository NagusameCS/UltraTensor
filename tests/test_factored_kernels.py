"""Phase 2 kernel tests: fused factored GEMV (CUDA; skipped without GPU)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ultratensor.kernels import (factored_gemv, factored_gemv_from_gguf,
                                 factored_gemv_moe, numpy_reference)  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available")


def _random_factored(m, k, n, seed=0):
    rng = np.random.default_rng(seed)
    U = torch.from_numpy(rng.standard_normal((m, k)).astype(np.float16)).cuda()
    C = torch.from_numpy(rng.standard_normal((k, n)).astype(np.float32)).cuda()
    Cb = C.reshape(k, n // 32, 32)
    sc = (Cb.abs().amax(-1) / 8.0).clamp_min(1e-6)
    q = (Cb / sc.unsqueeze(-1) + 8.0).round().clamp(0, 15).to(torch.uint8)
    q = q.reshape(k, n // 2, 2)
    packed = (q[..., 0] | (q[..., 1] << 4)).contiguous()
    return U, sc.contiguous(), packed


def test_factored_gemv_matches_numpy():
    m, k, n = 1024, 16, 2048
    U, scales, packed = _random_factored(m, k, n, seed=1)
    x = torch.randn(n, device="cuda")
    y = factored_gemv(U, scales, packed, x)
    y_ref = torch.from_numpy(numpy_reference(
        U.cpu().numpy(), scales.cpu().numpy(), packed.cpu().numpy(),
        x.cpu().numpy())).cuda()
    rel = (y - y_ref).abs().max() / y_ref.abs().max()
    assert rel.item() < 1e-3


def test_factored_gemv_shape_handling():
    m, k, n = 256, 32, 4096
    U, scales, packed = _random_factored(m, k, n, seed=2)
    x = torch.randn(n, device="cuda")
    y = factored_gemv(U, scales, packed, x)
    assert y.shape == (m,)
    assert y.dtype == torch.float32


def _random_moe(E, m, k, n, seed=0):
    rng = np.random.default_rng(seed)
    U = torch.from_numpy(rng.standard_normal((E, m, k)).astype(np.float16)).cuda()
    C = torch.from_numpy(rng.standard_normal((E, k, n)).astype(np.float32)).cuda()
    Cb = C.reshape(E, k, n // 32, 32)
    sc = (Cb.abs().amax(-1) / 8.0).clamp_min(1e-6)
    q = (Cb / sc.unsqueeze(-1) + 8.0).round().clamp(0, 15).to(torch.uint8)
    q = q.reshape(E, k, n // 2, 2)
    packed = (q[..., 0] | (q[..., 1] << 4)).contiguous()
    return U, sc.contiguous(), packed


def test_factored_gemv_moe_matches_numpy():
    E, m, k, n = 8, 512, 16, 2048
    U, scales, packed = _random_moe(E, m, k, n, seed=3)
    x = torch.randn(n, device="cuda")
    y = factored_gemv_moe(U, scales, packed, x)
    # numpy: sum over experts of U_e @ C_e @ x
    y_ref = torch.zeros(m, device="cuda")
    for e in range(E):
        y_ref += torch.from_numpy(numpy_reference(
            U[e].cpu().numpy(), scales[e].cpu().numpy(),
            packed[e].cpu().numpy(), x.cpu().numpy())).cuda()
    rel = (y - y_ref).abs().max() / y_ref.abs().max()
    assert rel.item() < 1e-3


def test_container_to_kernel_e2e(tmp_path):
    """Phase 3 connector: factored GGUF container -> fused CUDA kernel."""
    from ultratensor.gguf_factored import (read_factored_gguf, reconstruct,
                                           write_factored_gguf)
    from test_gguf_factored import _write_source

    rng = np.random.default_rng(11)
    A = rng.standard_normal((128, 16)).astype(np.float32)
    B = rng.standard_normal((16, 256)).astype(np.float32)
    W = (A @ B + 0.01 * rng.standard_normal((128, 256))).astype(np.float32)
    F = np.zeros((8, 8), np.float32)
    src = tmp_path / "src.gguf"
    out = tmp_path / "out.gguf"
    _write_source(src, W, F)
    write_factored_gguf(src, out, patterns=["t1"], rank=16)

    x = torch.randn(256, device="cuda")
    y = factored_gemv_from_gguf(out, "t1", x)

    manifest, tensors = read_factored_gguf(out)
    W_hat = reconstruct(manifest, tensors, "t1.factored_C")
    y_ref = torch.from_numpy(W_hat @ x.cpu().numpy()).cuda()
    rel = (y - y_ref).abs().max() / y_ref.abs().max()
    assert rel.item() < 1e-4
