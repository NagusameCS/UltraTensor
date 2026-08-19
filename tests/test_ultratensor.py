"""UltraTensor tests: dequant correctness, quant round-trips, and the
streaming compressor end-to-end on a synthetic GGUF.

Cross-check against llama.cpp: if `llama-quantize.exe` exists at the
standard path, an F16 GGUF is quantized by llama.cpp to Q2_K/Q3_K/Q4_K/
Q5_K/Q6_K/Q8_0 and our dequantizers are verified to reproduce the
original values within the expected per-format tolerance.
"""

import json
import shutil
import subprocess
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultratensor.dequant import dequantize, decode_fp16  # noqa: E402
from ultratensor.quant import (  # noqa: E402
    q2_0_quantize, q2_0_dequantize,
    q4_0_quantize, q4_0_dequantize, q8_0_quantize,
    uq4_quantize, uq4_dequantize,
)
from ultratensor.stream import compress_gguf, dry_run  # noqa: E402

LLAMA_QUANTIZE = Path(r"C:\Users\legom\hyperv4flash\engine\llama-quantize.exe")


def make_f16_gguf(path: Path, tensors: dict[str, np.ndarray]):
    from gguf import GGUFWriter

    writer = GGUFWriter(str(path), "llama")
    writer.add_architecture()
    writer.add_name("ut-test")
    writer.add_context_length(512)
    writer.add_embedding_length(256)
    writer.add_block_count(1)
    writer.add_feed_forward_length(256)
    writer.add_head_count(8)
    writer.add_head_count_kv(8)
    writer.add_layer_norm_rms_eps(1e-5)
    writer.add_rope_freq_base(10000.0)
    writer.add_rope_dimension_count(32)
    writer.add_vocab_size(32000)
    for name, arr in tensors.items():
        arr = arr.astype(np.float16)
        writer.add_tensor(name, arr, raw_shape=arr.shape)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def test_fp16_decode_known_values():
    # 1.0 -> 0x3C00, -2.0 -> 0xC000, 0.5 -> 0x3800, 1536 -> 0x6600
    u = np.array([0x3C00, 0xC000, 0x3800, 0x6600], dtype=np.uint16)
    out = decode_fp16(u)
    np.testing.assert_allclose(out, [1.0, -2.0, 0.5, 1536.0], rtol=1e-6)


def test_q8_0_roundtrip():
    rng = np.random.default_rng(0)
    W = rng.normal(0, 0.02, size=(3, 256)).astype(np.float32)
    scales, codes = q8_0_quantize(W)
    Wb = (codes.astype(np.float32) * scales[..., None]).reshape(3, 256)
    rel = np.abs(Wb - W) / (np.abs(W) + 0.02)
    assert rel.max() < 0.02


def test_q4_0_roundtrip():
    rng = np.random.default_rng(1)
    W = rng.normal(0, 0.02, size=(2, 64)).astype(np.float32)
    scales, nib = q4_0_quantize(W)
    Wb = q4_0_dequantize(scales, nib)
    rel = np.abs(Wb - W) / (np.abs(W) + 0.02)
    assert rel.max() < 0.15


def test_uq4_roundtrip():
    rng = np.random.default_rng(2)
    W = rng.normal(0, 0.02, size=(3, 512)).astype(np.float32)
    scales, packed = uq4_quantize(W, block=128)
    Wb = uq4_dequantize(scales, packed)
    assert Wb.shape == W.shape
    rel = np.abs(Wb - W) / (np.abs(W) + 0.02)
    assert rel.max() < 0.2


def test_uq4_exact_zeros():
    """Sparse weights must keep exact zeros (symmetric grid)."""
    rng = np.random.default_rng(5)
    W = rng.normal(0, 0.02, size=(1, 256)).astype(np.float32)
    W[:, ::3] = 0.0  # sparse: every 3rd weight is exactly zero
    scales, packed = uq4_quantize(W, block=128)
    Wb = uq4_dequantize(scales, packed)
    assert np.all(Wb[:, ::3] == 0.0), "zeros must survive uq4 exactly"


def test_q2_0_roundtrip():
    rng = np.random.default_rng(6)
    W = rng.normal(0, 0.02, size=(3, 512)).astype(np.float32)
    scales, packed = q2_0_quantize(W)
    Wb = q2_0_dequantize(scales, packed)
    assert Wb.shape == W.shape
    # 2-bit error is bounded by scale/2 = amax/6 per block
    Wblk = W.reshape(3, 16, 32)
    amax = np.abs(Wblk).max(axis=-1)
    err = np.abs(Wb - W).reshape(3, 16, 32).max(axis=-1)
    assert np.all(err <= amax / 3.0 + 1e-6)
    # symmetric grid: sign(Wb) == sign(W) whenever |W| >= amax/6
    big = np.abs(W) >= (np.abs(Wblk).max(axis=-1).repeat(32, axis=1) / 6.0)
    assert np.all(np.sign(Wb[big]) == np.sign(W[big]))


def _gaussian(n, seed):
    return np.random.default_rng(seed).normal(0, 0.02, size=(n, 256)).astype(np.float16)


@pytest.mark.parametrize("quant,rtol", [
    ("Q8_0", 0.01), ("Q4_0", 0.1), ("Q2_K", 0.35),
    ("Q3_K", 0.2), ("Q4_K", 0.08), ("Q5_K", 0.04), ("Q6_K", 0.02),
])
def test_dequant_crosscheck_llamacpp(tmp_path, quant, rtol):
    """Quantize with llama.cpp, dequantize with UltraTensor, compare."""
    if not LLAMA_QUANTIZE.exists():
        pytest.skip("llama-quantize.exe not available")
    W = _gaussian(4, zlib.crc32(quant.encode()) % 1000)
    f16 = tmp_path / "src.gguf"
    out = tmp_path / f"qt-{quant}.gguf"
    make_f16_gguf(f16, {"blk.0.weight": W})
    r = subprocess.run(
        [str(LLAMA_QUANTIZE), "--pure", str(f16), str(out), quant],
        capture_output=True, text=True, timeout=600,
    )
    assert r.returncode == 0, r.stderr[-2000:]

    from gguf import GGUFReader
    reader = GGUFReader(str(out))
    t = next(iter(reader.tensors))
    qtype = t.tensor_type.name
    data = np.ascontiguousarray(t.data)
    logical = tuple(int(s) for s in t.shape.tolist())
    Wd = dequantize(qtype, data, logical)
    assert Wd.shape == logical[::-1], f"{Wd.shape} != {logical[::-1]}"
    W0 = W.astype(np.float32).reshape(logical[::-1])

    denom = np.abs(W0) + 0.02
    rel = np.abs(Wd - W0) / denom
    assert rel.mean() < rtol, f"{quant}: mean rel err {rel.mean():.4f} > {rtol}"


def test_stream_compress_end_to_end(tmp_path):
    """Synthetic GGUF with f32 tensors goes through the pipeline.

    The source file is re-quantized to Q8_0 by llama.cpp when available so
    the pipeline is exercised on a real quantized GGUF; otherwise the f32
    file is used directly.
    """
    rng = np.random.default_rng(7)
    W = rng.normal(0, 0.01, size=(8, 256)).astype(np.float32)
    f16 = tmp_path / "tiny.gguf"
    make_f16_gguf(f16, {"W": W})

    src = f16
    if LLAMA_QUANTIZE.exists():
        q8 = tmp_path / "tiny_q8.gguf"
        r = subprocess.run(
            [str(LLAMA_QUANTIZE), "--pure", str(f16), str(q8), "Q8_0"],
            capture_output=True, text=True, timeout=600,
        )
        if r.returncode == 0:
            src = q8

    outdir = tmp_path / "out"
    manifest = compress_gguf(src, outdir, target="uq4", block=128)
    assert (outdir / "ultratensor_manifest.json").exists()
    assert manifest["tensors"], "no tensors compressed"
    assert any(outdir.glob("*.safetensors")), "no safetensors written"

    from safetensors.numpy import load_file
    sd = {}
    for shard in outdir.glob("*.safetensors"):
        sd.update(load_file(str(shard)))
    entry = manifest["tensors"][0]
    assert f"{entry['name']}.codes" in sd
    assert f"{entry['name']}.scales" in sd

    # q4_0 target: same pipeline, packed uint8 codes
    outdir2 = tmp_path / "out_q4"
    manifest2 = compress_gguf(src, outdir2, target="q4_0")
    sd2 = {}
    for shard in outdir2.glob("*.safetensors"):
        sd2.update(load_file(str(shard)))
    e2 = manifest2["tensors"][0]
    assert e2["target"] == "q4_0"
    assert sd2[f"{e2['name']}.codes"].dtype == np.uint8


def test_dry_run_reports(tmp_path, capsys):
    rng = np.random.default_rng(3)
    W = rng.normal(0, 0.01, size=(4, 256)).astype(np.float32)
    f16 = tmp_path / "tiny.gguf"
    make_f16_gguf(f16, {"W": W})
    rows = dry_run(f16, target="uq4")
    assert rows and rows[0][1] == "F16"


def test_grc_end_to_end(tmp_path):
    """GRC on a synthetic attention-named tensor: compress and rebuild."""
    from safetensors.numpy import load_file
    from ultratensor.grc import grc_compress_gguf, reconstruct

    rng = np.random.default_rng(9)
    # low-rank-ish attention matrix: 128x512 with 3 dominant directions
    B = rng.normal(0, 1.0, (3, 512)).astype(np.float32)
    W = rng.normal(0, 0.2, (128, 3)).astype(np.float32) @ B
    W += rng.normal(0, 0.01, W.shape).astype(np.float32)
    f16 = tmp_path / "attn.gguf"
    make_f16_gguf(f16, {"blk.0.attn_q_a.weight": W, "blk.0.attn_kv.weight": W.T.copy()})

    outdir = tmp_path / "grc_out"
    manifest = grc_compress_gguf(f16, outdir, energy=0.99, progress=False)
    assert len(manifest["tensors"]) == 2
    sd = {}
    for shard in outdir.glob("*.safetensors"):
        sd.update(load_file(str(shard)))
    for entry in manifest["tensors"]:
        Wr = reconstruct(entry, sd)
        assert Wr.shape == tuple(entry["shape"])
        W0 = (W if "q_a" in entry["name"] else W.T).astype(np.float32)
        rel = np.abs(Wr - W0) / (np.abs(W0) + 0.05)
        assert rel.mean() < 0.35, f"{entry['name']}: {rel.mean()}"
        assert entry["size_ratio"] < 1.0

    # streaming Gram path must give the same rank
    outdir2 = tmp_path / "grc_out2"
    m2 = grc_compress_gguf(f16, outdir2, energy=0.99, progress=False,
                           force_streaming=True)
    ranks1 = [e["rank"] for e in manifest["tensors"]]
    ranks2 = [e["rank"] for e in m2["tensors"]]
    assert ranks1 == ranks2


def test_export_q2k_valid(tmp_path):
    """Exported Q2_K GGUF must be loadable by llama.cpp (and gguf-py)."""
    from ultratensor.export_gguf import export_q2k
    if not LLAMA_QUANTIZE.exists():
        pytest.skip("llama-quantize.exe not available")
    W = _gaussian(4, 77)
    f16 = tmp_path / "src.gguf"
    make_f16_gguf(f16, {"blk.0.weight": W})
    q3 = tmp_path / "q3.gguf"
    r = subprocess.run([str(LLAMA_QUANTIZE), "--pure", str(f16), str(q3),
                        "Q3_K"], capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-500:]
    out = tmp_path / "out_q2k.gguf"
    export_q2k(q3, out, progress=False)
    r2 = subprocess.run([str(LLAMA_QUANTIZE), "--allow-requantize", "--pure",
                         str(out), str(tmp_path / "reload.gguf"), "Q4_K"],
                        capture_output=True, text=True, timeout=600)
    assert r2.returncode == 0, r2.stderr[-500:]
    # decode our exported tensor and compare against the source
    from gguf import GGUFReader
    from ultratensor.dequant import dequantize
    t = next(iter(GGUFReader(str(out)).tensors))
    Wd = dequantize("Q2_K", np.ascontiguousarray(t.data),
                    tuple(int(s) for s in t.shape.tolist()))
    W0 = W.astype(np.float32).reshape(Wd.shape)
    rel = np.abs(Wd - W0) / (np.abs(W0) + 0.02)
    assert rel.mean() < 0.3, f"export rel err {rel.mean()}"


def test_tensor_inventory_header_only(tmp_path):
    """Header-only inventory parses names/types/dims without touching data."""
    from ultratensor.stream import tensor_inventory
    rng = np.random.default_rng(8)
    W = rng.normal(0, 0.01, size=(4, 256)).astype(np.float32)
    f16 = tmp_path / "tiny.gguf"
    make_f16_gguf(f16, {"W": W})
    rows = tensor_inventory(f16)
    assert len(rows) == 1
    name, q, dims, off = rows[0]
    assert name == "W" and q == "F16"
    assert tuple(int(d) for d in dims) == (256, 4)  # GGUF NE order
    assert off >= 0


def test_q2k_torch_matches_numpy():
    """CUDA Q2_K search is byte-identical to numpy on tie-free data and
    reaches the same quality on random data."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    from ultratensor.quant import q2_k_quantize
    rng = np.random.default_rng(11)
    # tie-free: exact grid values, every (m,n) argmin is unambiguous
    W = rng.choice([-3, -1, 1, 3], size=(16, 1024)).astype(np.float32) * 0.5
    a = q2_k_quantize(W, use_torch=True)
    b = q2_k_quantize(W, use_torch=False)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    # random: same decoded quality (scale-choice ties may differ)
    W = rng.normal(0, 0.02, size=(16, 2048)).astype(np.float32)
    a = q2_k_quantize(W, use_torch=True)
    from ultratensor.dequant import dequant_q2_K
    data = np.concatenate([a[0], a[1],
                           a[2].view(np.uint8).reshape(-1, 2),
                           a[3].view(np.uint8).reshape(-1, 2)], axis=1)
    Wb = dequant_q2_K(data).reshape(W.shape)
    rel = np.abs(Wb - W) / (np.abs(W) + 0.02)
    assert rel.mean() < 0.3


def test_mxfp4_dequant_matches_ggufpy():
    """Our MXFP4 decoder must match gguf-py's reference implementation."""
    from gguf.quants import MXFP4 as Ref
    from ultratensor.dequant import dequant_mxfp4

    rng = np.random.default_rng(11)
    blocks = np.concatenate([
        rng.normal(0, 2.0, (4, 32)),
        rng.normal(0, 0.02, (4, 32)),
    ], axis=0).astype(np.float32)
    enc = Ref.quantize_blocks(blocks)      # [8, 17]
    ref = Ref.dequantize_blocks(enc)       # [8, 32]
    out = dequant_mxfp4(enc)
    assert out.shape == (8, 32)
    np.testing.assert_allclose(out, ref, rtol=1e-6)


def test_dequantize_mxfp4_rows():
    """Row-sliced streaming decode of a 2-D MXFP4 tensor."""
    from gguf.quants import MXFP4 as Ref
    from ultratensor.dequant import dequantize, dequantize_rows

    rng = np.random.default_rng(12)
    blocks = rng.normal(0, 0.02, (8, 32)).astype(np.float32)
    enc = Ref.quantize_blocks(blocks)      # [8, 17]
    data = enc.reshape(4, 34)              # 4 rows x 2 blocks
    logical = (64, 4)

    full = dequantize("MXFP4", data, logical)
    assert full.shape == (4, 64)
    ref = Ref.dequantize_blocks(enc).reshape(4, 64)
    np.testing.assert_allclose(full, ref, rtol=1e-6)

    r12 = dequantize_rows("MXFP4", data, logical, row_start=1, row_count=2)
    assert r12.shape == (2, 64)
    np.testing.assert_allclose(r12, full[1:3], rtol=1e-6)


def test_iq2_xxs_dequant_matches_ggufpy():
    """Our IQ2_XXS decoder must match gguf-py's reference implementation."""
    from gguf.quants import IQ2_XXS as Ref
    from ultratensor.dequant import dequant_iq2_xxs

    Ref.init_grid()
    rng = np.random.default_rng(13)
    # realistic blocks: fp16 d + 64 qs bytes
    blocks = np.zeros((16, 66), dtype=np.uint8)
    blocks[:, 0:2] = np.ascontiguousarray(
        rng.normal(0, 0.05, (16, 1)).astype(np.float16)
    ).view(np.uint8).reshape(16, 2)
    blocks[:, 2:66] = rng.integers(0, 256, (16, 64), dtype=np.uint8)

    ref = Ref.dequantize_blocks(blocks)    # [16, 256]
    out = dequant_iq2_xxs(blocks)
    assert out.shape == (16, 256)
    np.testing.assert_allclose(out, ref, rtol=1e-6)


def test_dequantize_iq2_xxs_rows():
    """Row-sliced streaming decode of a 2-D IQ2_XXS tensor."""
    from gguf.quants import IQ2_XXS as Ref
    from ultratensor.dequant import dequantize, dequantize_rows

    Ref.init_grid()
    rng = np.random.default_rng(14)
    blocks = np.zeros((8, 66), dtype=np.uint8)
    blocks[:, 0:2] = np.ascontiguousarray(
        rng.normal(0, 0.05, (8, 1)).astype(np.float16)
    ).view(np.uint8).reshape(8, 2)
    blocks[:, 2:66] = rng.integers(0, 256, (8, 64), dtype=np.uint8)
    data = blocks.reshape(4, 2, 66).reshape(4, 132)  # 4 rows x 2 blocks
    logical = (512, 4)

    full = dequantize("IQ2_XXS", data, logical)
    assert full.shape == (4, 512)
    ref = Ref.dequantize_blocks(blocks).reshape(4, 512)
    np.testing.assert_allclose(full, ref, rtol=1e-6)

    r12 = dequantize_rows("IQ2_XXS", data, logical, row_start=1, row_count=2)
    assert r12.shape == (2, 512)
    np.testing.assert_allclose(r12, full[1:3], rtol=1e-6)


def _random_blocks(n, type_size, seed):
    rng = np.random.default_rng(seed)
    blocks = np.zeros((n, type_size), dtype=np.uint8)
    blocks[:, 0:2] = np.ascontiguousarray(
        rng.normal(0, 0.05, (n, 1)).astype(np.float16)
    ).view(np.uint8).reshape(n, 2)
    blocks[:, 2:type_size] = rng.integers(0, 256, (n, type_size - 2), dtype=np.uint8)
    return blocks


@pytest.mark.parametrize("quant,type_size", [
    ("IQ2_XS", 74), ("IQ3_XXS", 98), ("Q4_1", 20), ("Q5_0", 22), ("Q5_1", 24),
    ("IQ4_NL", 18), ("IQ4_XS", 136), ("IQ3_S", 110), ("IQ2_S", 82),
    ("IQ1_S", 50), ("IQ1_M", 56), ("TQ1_0", 54), ("TQ2_0", 66),
])
def test_i_quant_dequant_matches_ggufpy(quant, type_size):
    """Every remaining format must match gguf-py's reference decoder."""
    import importlib
    from ultratensor.dequant import DEQUANT
    Ref = getattr(importlib.import_module("gguf.quants"), quant)
    Ref.init_grid()
    blocks = _random_blocks(16, type_size, seed=zlib.crc32(quant.encode()) % 10000 + 15)
    ref = Ref.dequantize_blocks(blocks)      # [16, 256] (or [16, 32])
    out = DEQUANT[quant](blocks)
    assert out.shape == ref.shape, f"{out.shape} != {ref.shape}"
    np.testing.assert_allclose(out, ref, rtol=1e-6)
