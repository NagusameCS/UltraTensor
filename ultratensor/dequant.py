"""UltraTensor: dequantization of GGUF quant formats, vectorized with numpy.

Every dequantizer here is a direct port of the canonical llama.cpp
implementation (ggml-quants.c, build b10424), operating on the raw
tensor buffers exposed by the `gguf` package. All functions are
chunked: a huge tensor is dequantized block-batch by block-batch so
memory stays bounded (streaming-friendly).

Supported types (every quant format gguf-py/llama.cpp b10424 knows):
    F32, F16, BF16,
    Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K,
    MXFP4,
    IQ2_XXS, IQ2_XS, IQ3_XXS, IQ4_NL, IQ4_XS, IQ3_S, IQ2_S,
    IQ1_S, IQ1_M, TQ1_0, TQ2_0
"""

from __future__ import annotations

import numpy as np

QK_K = 256
K_SCALE_SIZE = 12


# ---------------------------------------------------------------------------
# fp16 / bf16 helpers (torch when available, numpy fallback)
# ---------------------------------------------------------------------------

def _has_torch() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def decode_fp16(u16: np.ndarray) -> np.ndarray:
    """Decode uint16 little-endian half-precision values to float32."""
    u = np.ascontiguousarray(u16, dtype=np.uint16)
    if _has_torch():
        import torch
        return torch.frombuffer(
            u.tobytes(), dtype=torch.float16
        ).float().numpy().reshape(u.shape)
    sign = (u >> 15).astype(np.float32)
    exp = ((u >> 10) & 0x1F).astype(np.float32)
    man = (u & 0x3FF).astype(np.float32)
    val = np.where(
        exp == 0,
        np.float32(2.0) ** -14 * (man / 1024.0),
        np.where(
            exp < 31,
            np.float32(2.0) ** (exp - 15) * (1.0 + man / 1024.0),
            np.where(man == 0, np.inf, np.nan),
        ),
    )
    return np.where(sign == 1, -val, val).astype(np.float32)


def decode_bf16(u16: np.ndarray) -> np.ndarray:
    """Decode uint16 little-endian bfloat16 values to float32."""
    u = np.asarray(u16, dtype=np.uint16).astype(np.uint32)
    return (u << 16).view(np.float32)


# ---------------------------------------------------------------------------
# Simple formats
# ---------------------------------------------------------------------------

def dequant_q8_0(data: np.ndarray) -> np.ndarray:
    """data: [n_blocks, 34] -> [n_blocks, 32]"""
    d = decode_fp16(data[:, 0:2].view(np.uint16).reshape(-1, 1))
    qs = data[:, 2:34].astype(np.int8).astype(np.float32)
    return qs * d


def dequant_q4_0(data: np.ndarray) -> np.ndarray:
    """data: [n_blocks, 18] -> [n_blocks, 16]"""
    d = decode_fp16(data[:, 0:2].view(np.uint16).reshape(-1, 1))
    qs = data[:, 2:18]
    lo = (qs & 0x0F).astype(np.float32) - 8.0
    hi = (qs >> 4).astype(np.float32) - 8.0
    return np.concatenate([lo, hi], axis=1) * d


# ---------------------------------------------------------------------------
# K-quants
# ---------------------------------------------------------------------------

def dequant_q2_K(data: np.ndarray) -> np.ndarray:
    """data: [n, 84] -> [n, 256]  (port of dequantize_row_q2_K)

    Serialized layout: scales[0:16], qs[16:80], d[80:82], dmin[82:84].
    """
    scales = data[:, 0:16]                                  # [n, 16]
    qs = data[:, 16:80]                                     # [n, 64]
    d = decode_fp16(data[:, 80:82].view(np.uint16)).reshape(-1)      # [n]
    dmin = decode_fp16(data[:, 82:84].view(np.uint16)).reshape(-1)   # [n]

    sc = scales.astype(np.int16)
    scale4 = (sc & 0x0F).astype(np.float32) * d[:, None]    # [n, 16]
    min4 = (sc >> 4).astype(np.float32) * dmin[:, None]

    i = np.arange(QK_K)
    g = i // 128
    j = (i % 128) // 32
    half = (i % 32) // 16
    l = i % 16
    qix = g * 32 + half * 16 + l
    is_ = g * 8 + j * 2 + half
    shift = 2 * j

    codes = (qs[:, qix] >> shift[None, :]) & 3
    dl = scale4[:, is_]
    ml = min4[:, is_]
    return dl * codes.astype(np.float32) - ml


def _q3_k_scales(scales12: np.ndarray) -> np.ndarray:
    """Unpack 12 scale bytes -> 16 int8 scale values (per block)."""
    n = scales12.shape[0]
    a = scales12.reshape(n, 3, 4).view(np.uint32).reshape(n, 3)  # little-endian
    a0, a1, a2 = a[:, 0], a[:, 1], a[:, 2]
    kmask1 = np.uint32(0x03030303)
    kmask2 = np.uint32(0x0F0F0F0F)
    b0 = (a0 & kmask2) | ((a2 & kmask1) << 4)
    b1 = (a1 & kmask2) | (((a2 >> 2) & kmask1) << 4)
    b2 = ((a0 >> 4) & kmask2) | (((a2 >> 4) & kmask1) << 4)
    b3 = ((a1 >> 4) & kmask2) | (((a2 >> 6) & kmask1) << 4)
    stacked = np.stack([b0, b1, b2, b3], axis=1)          # [n, 4]
    return stacked.view(np.int8).reshape(n, 16).astype(np.float32)


def dequant_q3_K(data: np.ndarray) -> np.ndarray:
    """data: [n, 110] -> [n, 256]  (port of dequantize_row_q3_K)"""
    hmask = data[:, 0:32]
    qs = data[:, 32:96]
    s = _q3_k_scales(data[:, 96:108]) - 32.0
    d_all = decode_fp16(data[:, 108:110].view(np.uint16)).reshape(-1)  # [n]

    i = np.arange(QK_K)
    g = i // 128
    j = (i % 128) // 32
    half = (i % 32) // 16
    l = i % 16
    qix = g * 32 + half * 16 + l      # qs index (64 bytes per block)
    hmix = half * 16 + l               # hmask index (32 bytes per block)
    is_ = g * 8 + j * 2 + half
    shift = (2 * j).astype(np.uint8)
    mbit = (1 << (4 * g + j)).astype(np.uint8)

    low2 = (qs[:, qix] >> shift[None, :]) & 3
    signed_mask = (hmask[:, hmix] & mbit[None, :]) != 0
    codes = low2.astype(np.float32) - np.where(signed_mask, 0.0, 4.0)
    dl = d_all[:, None] * s[:, is_]
    return dl * codes


def _q45_k_scales(scales12: np.ndarray):
    """Unpack q4_K/q5_K 12-byte scales -> (sc[8], m[8]) per block."""
    q = scales12
    sc = np.empty((q.shape[0], 8), dtype=np.float32)
    m = np.empty((q.shape[0], 8), dtype=np.float32)
    sc[:, 0:4] = (q[:, 0:4] & 63).astype(np.float32)
    m[:, 0:4] = (q[:, 4:8] & 63).astype(np.float32)
    sc[:, 4:8] = ((q[:, 8:12] & 0x0F) | ((q[:, 0:4] >> 6) << 4)).astype(np.float32)
    m[:, 4:8] = ((q[:, 8:12] >> 4) | ((q[:, 4:8] >> 6) << 4)).astype(np.float32)
    return sc, m


def dequant_q4_K(data: np.ndarray) -> np.ndarray:
    """data: [n, 144] -> [n, 256]  (port of dequantize_row_q4_K)"""
    d = decode_fp16(data[:, 0:2].view(np.uint16)).reshape(-1)
    dmin = decode_fp16(data[:, 2:4].view(np.uint16)).reshape(-1)
    sc, m = _q45_k_scales(data[:, 4:16])
    qs = data[:, 16:144]  # [n, 128]

    i = np.arange(QK_K)
    g = i // 64
    l32 = i % 32
    half = (i % 64) // 32
    qix = g * 32 + l32
    is_ = g * 2 + half

    qb = qs[:, qix]
    codes = np.where(half[None, :] == 0, (qb & 0x0F), (qb >> 4)).astype(np.float32)
    return d[:, None] * sc[:, is_] * codes - dmin[:, None] * m[:, is_]


def dequant_q5_K(data: np.ndarray) -> np.ndarray:
    """data: [n, 176] -> [n, 256]  (port of dequantize_row_q5_K)"""
    d = decode_fp16(data[:, 0:2].view(np.uint16)).reshape(-1)
    dmin = decode_fp16(data[:, 2:4].view(np.uint16)).reshape(-1)
    sc, m = _q45_k_scales(data[:, 4:16])
    qh = data[:, 16:48]   # [n, 32]
    ql = data[:, 48:176]  # [n, 128]

    i = np.arange(QK_K)
    g = i // 64
    l32 = i % 32
    half = (i % 64) // 32
    qix = g * 32 + l32
    is_ = g * 2 + half
    u = (1 << (2 * g)) if False else None

    base = np.where(half[None, :] == 0, (ql[:, qix] & 0x0F), (ql[:, qix] >> 4))
    ubit = np.where(half[None, :] == 0, (2 ** (2 * g))[None, :], (2 ** (2 * g + 1))[None, :])
    high = np.where((qh[:, l32][:, :] & ubit) != 0, 16.0, 0.0)
    codes = (base + high).astype(np.float32)
    return d[:, None] * sc[:, is_] * codes - dmin[:, None] * m[:, is_]


def dequant_q6_K(data: np.ndarray) -> np.ndarray:
    """data: [n, 210] -> [n, 256]  (port of dequantize_row_q6_K)"""
    d = decode_fp16(data[:, 208:210].view(np.uint16)).reshape(-1)          # [n]
    ql = data[:, 0:128]    # [n, 128]
    qh = data[:, 128:192]  # [n, 64]
    sc = data[:, 192:208].astype(np.int8).astype(np.float32)  # [n, 16]

    i = np.arange(QK_K)
    g = i // 128
    h = (i % 128) // 32
    l = i % 32
    is_ = l // 16

    ql_src = ql[:, g * 64 + (h % 2) * 32 + l]   # [n, 256]
    qhx = qh[:, g * 32 + l]                      # [n, 256]
    nib = np.where(h >= 2, ql_src >> 4, ql_src & 0x0F)
    hib = (qhx >> (2 * h)) & 3
    qcode = (nib | (hib << 4)).astype(np.float32) - 32.0

    # llama.cpp b10424 layout: 4 scale pairs per 128-block
    # (verified against the llama-quantize oracle crosscheck).
    sidx = g * 8 + is_ + 2 * h
    return d[:, None] * sc[:, sidx] * qcode


# ---------------------------------------------------------------------------
# OCP Microscaling MXFP4 (E8M0 scale, E2M1-doubled 4-bit values)
# ---------------------------------------------------------------------------

# kvalues_mxfp4 from ggml-quants.c (E2M1 values doubled, bias 8)
MXFP4_KVALUES = np.array(
    [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12],
    dtype=np.int8,
)


def dequant_mxfp4(data: np.ndarray) -> np.ndarray:
    """data: [n, 17] -> [n, 32]  (port of dequantize_row_mxfp4)

    Serialized layout: e[0:1] (E8M0 scale byte), qs[1:17] (16 packed
    4-bit codes; low nibble = element j, high nibble = element j+16).
    """
    e = data[:, 0:1].astype(np.uint32)
    bits = np.where(
        e < 2, np.uint32(0x00200000) << e, (e - np.uint32(1)) << np.uint32(23)
    ).astype(np.uint32)
    d = bits.view(np.float32)                                   # [n, 1]
    qs = data[:, 1:17]
    idx = np.concatenate([qs & 0x0F, qs >> 4], axis=1)          # [n, 32]
    return d * MXFP4_KVALUES[idx].astype(np.float32)


# ---------------------------------------------------------------------------
# IQ2_XXS ("true" 2-bit, 2.0625 bpw) — the DeepSeek expert-tensor format
# ---------------------------------------------------------------------------

def _iq2xxs_tables():
    """Reference tables from gguf-py (identical to ggml-quants.c).

    Returns (grid [256, 8] int8 odd values, ksigns [128] uint8).
    """
    from gguf.quants import IQ2_XXS
    IQ2_XXS.init_grid()
    grid = np.ascontiguousarray(IQ2_XXS.grid).reshape(256, 8).astype(np.int8)
    ksigns = np.frombuffer(IQ2_XXS.ksigns, dtype=np.uint8)
    return grid, ksigns


def dequant_iq2_xxs(data: np.ndarray) -> np.ndarray:
    """data: [n, 66] -> [n, 256]  (port of dequantize_row_iq2_xxs)

    Serialized layout: d[0:2] fp16, then 8 groups of 8 bytes. Each group:
    bytes 0-3 = 4 grid indices (one per 8-element subgroup),
    bytes 4-7 = uint32 LE: top 4 bits = scale shift, then 7 sign bits
    per subgroup (subgroup l at bits 7*l..7*l+6).
    """
    grid, ksigns = _iq2xxs_tables()
    n = data.shape[0]
    d = decode_fp16(data[:, 0:2].view(np.uint16)).reshape(n, 1)

    # 64 qs bytes as 16 little-endian uint32: group g uses [2g] (grid
    # indices) and [2g+1] (scale + signs).
    qs32 = (np.ascontiguousarray(data[:, 2:66])
            .view(np.uint32).reshape(n, 16))
    aux0 = qs32[:, 0::2]                                   # [n, 8]
    aux1 = qs32[:, 1::2]                                   # [n, 8]

    idx = np.ascontiguousarray(aux0).view(np.uint8).reshape(n, 8, 4)  # [n,8,4]
    db = d * (0.5 + (aux1 >> 28).astype(np.float32)) * 0.25          # [n,8]

    sign_idx = ((aux1[:, :, None]
                 >> np.array([0, 7, 14, 21], np.uint32)[None, None, :])
                & 0x7F)                                    # [n,8,4]
    s = ksigns[sign_idx]                                   # [n,8,4]
    shift = np.arange(8, dtype=np.uint8)                   # bit positions
    signs = (s[:, :, :, None] >> shift[None, None, None, :]) & 1
    signs = np.where(signs == 0, np.float32(1.0), np.float32(-1.0))  # [n,8,4,8]

    g = grid[idx].astype(np.float32)                       # [n,8,4,8]
    return (db[:, :, None, None] * g * signs).reshape(n, 256)


# ---------------------------------------------------------------------------
# IQ2_XS (2.3125 bpw) and IQ3_XXS (3.0625 bpw)
# ---------------------------------------------------------------------------

def _iq2xs_tables():
    """IQ2_XS grid [512, 8] and the shared IQ2_XXS ksigns table."""
    from gguf.quants import IQ2_XS, IQ2_XXS
    IQ2_XS.init_grid()
    grid = np.ascontiguousarray(IQ2_XS.grid).reshape(512, 8).astype(np.int8)
    ksigns = np.frombuffer(IQ2_XXS.ksigns, dtype=np.uint8)
    return grid, ksigns


def _iq3xxs_tables():
    """IQ3_XXS grid [256, 4] and the shared ksigns table."""
    from gguf.quants import IQ3_XXS, IQ2_XXS
    IQ3_XXS.init_grid()
    grid = np.ascontiguousarray(IQ3_XXS.grid).reshape(256, 4).astype(np.int8)
    ksigns = np.frombuffer(IQ2_XXS.ksigns, dtype=np.uint8)
    return grid, ksigns


def dequant_iq2_xs(data: np.ndarray) -> np.ndarray:
    """data: [n, 74] -> [n, 256]  (port of dequantize_row_iq2_xs)

    Layout: d[0:2] fp16, qs[2:66] = 32 little-endian uint16 (9-bit grid
    index + 7-bit sign index), scales[66:74] = 8 bytes (2 nibbles each).
    """
    grid, ksigns = _iq2xs_tables()
    n = data.shape[0]
    d = decode_fp16(data[:, 0:2].view(np.uint16)).reshape(n, 1)
    qs = np.ascontiguousarray(data[:, 2:66]).view(np.uint16)   # [n, 32]
    scales = data[:, 66:74]                                    # [n, 8]

    s = np.arange(32)
    g = s // 4
    half = (s % 4) // 2
    nib = (scales[:, g] >> (4 * half)[None, :]) & 0x0F        # [n, 32]
    db = d * (0.5 + nib.astype(np.float32)) * 0.25            # [n, 32]

    idx = qs & 0x1FF
    sign_idx = qs >> 9
    sbyte = ksigns[sign_idx]                                   # [n, 32]
    shift = np.arange(8, dtype=np.uint8)
    signs = (sbyte[:, :, None] >> shift[None, None, :]) & 1
    signs = np.where(signs == 0, np.float32(1.0), np.float32(-1.0))  # [n,32,8]

    gv = grid[idx].astype(np.float32)                          # [n, 32, 8]
    return (db[:, :, None] * gv * signs).reshape(n, 256)


def dequant_iq3_xxs(data: np.ndarray) -> np.ndarray:
    """data: [n, 98] -> [n, 256]  (port of dequantize_row_iq3_xxs)

    Layout: d[0:2] fp16, qs[2:66] = 64 grid-index bytes (8 per 32-group),
    scales[66:98] = 8 little-endian uint32 (top nibble = scale shift,
    then 7 sign bits per 8-element subgroup at shifts 7*l).
    """
    grid, ksigns = _iq3xxs_tables()
    n = data.shape[0]
    d = decode_fp16(data[:, 0:2].view(np.uint16)).reshape(n, 1)
    qs = data[:, 2:66]                                         # [n, 64]
    scales = (np.ascontiguousarray(data[:, 66:98])
              .view(np.uint32).reshape(n, 8))                  # [n, 8]

    idx = qs.reshape(n, 8, 8)                                  # [n, 8, 8]
    db = d * (0.5 + (scales >> 28).astype(np.float32)) * 0.5   # [n, 8]

    sign_idx = ((scales[:, :, None]
                 >> np.array([0, 7, 14, 21], np.uint32)[None, None, :])
                & 0x7F)                                        # [n, 8, 4]
    sbyte = ksigns[sign_idx]                                   # [n, 8, 4]
    shift = np.arange(8, dtype=np.uint8)
    signs = (sbyte[:, :, :, None] >> shift[None, None, None, :]) & 1
    signs = np.where(signs == 0, np.float32(1.0), np.float32(-1.0))  # [n,8,4,8]

    # subgroup l takes its first 4 values from grid index 2l and its
    # last 4 from grid index 2l+1 (llama.cpp element order).
    gv = grid[idx]                                             # [n, 8, 8, 4]
    gv = gv.reshape(n, 8, 4, 2, 4).reshape(n, 8, 4, 8)
    return (db[:, :, None, None] * gv.astype(np.float32) * signs).reshape(n, 256)


# ---------------------------------------------------------------------------
# Legacy formats: Q4_1, Q5_0, Q5_1
# ---------------------------------------------------------------------------

def dequant_q4_1(data: np.ndarray) -> np.ndarray:
    """data: [n, 20] -> [n, 32]  y = d*q + m"""
    d = decode_fp16(data[:, 0:2].view(np.uint16).reshape(-1, 1))
    m = decode_fp16(data[:, 2:4].view(np.uint16).reshape(-1, 1))
    qs = data[:, 4:20]
    lo = (qs & 0x0F).astype(np.float32)
    hi = (qs >> 4).astype(np.float32)
    return np.concatenate([lo, hi], axis=1) * d + m


def _q5_codes(qs, qh_bits):
    """Combine 4-bit codes with the 5th bit: [n,16] + [n,32] -> [n,32]."""
    lo = (qs & 0x0F).astype(np.uint8)
    hi = (qs >> 4).astype(np.uint8)
    q = np.concatenate([lo, hi], axis=1)
    return (q | (qh_bits << 4)).astype(np.int16) - 16


def dequant_q5_0(data: np.ndarray) -> np.ndarray:
    """data: [n, 22] -> [n, 32]  y = d*(q - 16), q = 5 bits signed"""
    d = decode_fp16(data[:, 0:2].view(np.uint16).reshape(-1, 1))
    qh = np.ascontiguousarray(data[:, 2:6]).view(np.uint32).reshape(-1, 1)
    bits = (qh >> np.arange(32, dtype=np.uint32)[None, :]) & 1
    return d * _q5_codes(data[:, 6:22], bits).astype(np.float32)


def dequant_q5_1(data: np.ndarray) -> np.ndarray:
    """data: [n, 24] -> [n, 32]  y = d*q + m (q unsigned 5-bit)"""
    d = decode_fp16(data[:, 0:2].view(np.uint16).reshape(-1, 1))
    m = decode_fp16(data[:, 2:4].view(np.uint16).reshape(-1, 1))
    qh = np.ascontiguousarray(data[:, 4:8]).view(np.uint32).reshape(-1, 1)
    bits = (qh >> np.arange(32, dtype=np.uint32)[None, :]) & 1
    qs = data[:, 8:24]
    lo = (qs & 0x0F).astype(np.uint8)
    hi = (qs >> 4).astype(np.uint8)
    q = (np.concatenate([lo, hi], axis=1) | (bits << 4)).astype(np.float32)
    return d * q + m


# ---------------------------------------------------------------------------
# IQ4_NL / IQ4_XS (4-bit i-quants with shared kvalues table)
# ---------------------------------------------------------------------------

IQ4_NL_KVALUES = np.array(
    [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113],
    dtype=np.int8,
)


def dequant_iq4_nl(data: np.ndarray) -> np.ndarray:
    """data: [n, 18] -> [n, 32]  y = d * kvalues_iq4nl[q]"""
    d = decode_fp16(data[:, 0:2].view(np.uint16).reshape(-1, 1))
    qs = data[:, 2:18]
    idx = np.concatenate([qs & 0x0F, qs >> 4], axis=1)
    return d * IQ4_NL_KVALUES[idx].astype(np.float32)


def dequant_iq4_xs(data: np.ndarray) -> np.ndarray:
    """data: [n, 136] -> [n, 256]  y = d*(scale-32) * kvalues_iq4nl[q]

    Layout: d[0:2] fp16, scales_h[2:4] uint16 (8 x 2 bits),
    scales_l[4:8] (8 nibbles), qs[8:136] (16 nibble-bytes per 32-group;
    low nibbles = elements 0-15, high = elements 16-31).
    """
    n = data.shape[0]
    d = decode_fp16(data[:, 0:2].view(np.uint16)).reshape(n, 1)
    sh = np.ascontiguousarray(data[:, 2:4]).view(np.uint16)      # [n, 1]
    sh = ((sh.reshape(n, 1, 1) >> (2 * np.arange(8, dtype=np.uint16))[None, None, :])
          & 3).reshape(n, 8)
    sl = (data[:, 4:8].reshape(n, 4, 1) >> np.array([0, 4], np.uint8)[None, None, :])
    sl = (sl & 0x0F).reshape(n, 8)
    scale = ((sl | (sh << 4)).astype(np.int16) - 32).astype(np.float32)  # [n, 8]
    dl = d * scale                                               # [n, 8]

    qs = data[:, 8:136].reshape(n, 8, 16)
    idx = np.concatenate([qs & 0x0F, qs >> 4], axis=2)           # [n, 8, 32]
    q = IQ4_NL_KVALUES[idx].astype(np.float32)
    return (dl[:, :, None] * q).reshape(n, 256)


# ---------------------------------------------------------------------------
# IQ3_S (3.4375 bpw) and IQ2_S (2.5625 bpw)
# ---------------------------------------------------------------------------

def _iq3s_tables():
    from gguf.quants import IQ3_S, IQ2_XXS
    IQ3_S.init_grid()
    grid = np.ascontiguousarray(IQ3_S.grid).reshape(512, 4).astype(np.int8)
    ksigns = np.frombuffer(IQ2_XXS.ksigns, dtype=np.uint8)
    return grid, ksigns


def _iq2s_tables():
    from gguf.quants import IQ2_S, IQ2_XXS
    IQ2_S.init_grid()
    grid = np.ascontiguousarray(IQ2_S.grid).reshape(1024, 8).astype(np.int8)
    ksigns = np.frombuffer(IQ2_XXS.ksigns, dtype=np.uint8)
    return grid, ksigns


def dequant_iq3_s(data: np.ndarray) -> np.ndarray:
    """data: [n, 110] -> [n, 256]  (port of dequantize_row_iq3_s)"""
    grid, ksigns = _iq3s_tables()
    n = data.shape[0]
    d = decode_fp16(data[:, 0:2].view(np.uint16)).reshape(n, 1)
    qs = data[:, 2:66]                                          # [n, 64]
    qh = data[:, 66:74]                                         # [n, 8]
    signs_raw = data[:, 74:106]                                 # [n, 32]
    sc = data[:, 106:110]                                       # [n, 4]

    sc = (sc.reshape(n, 4, 1) >> np.array([0, 4], np.uint8)[None, None, :])
    sc = (sc & 0x0F).reshape(n, 8).astype(np.float32)
    db = d * (1.0 + 2.0 * sc)                                   # [n, 8]

    bits = ((qh.reshape(n, 8, 1) >> np.arange(8, dtype=np.uint8)[None, None, :])
            & 1).reshape(n, 64)
    idx = qs.astype(np.uint16) | (bits.astype(np.uint16) << 8)  # [n, 64]

    sbyte = signs_raw.reshape(n, 32)
    shift = np.arange(8, dtype=np.uint8)
    signs = (sbyte[:, :, None] >> shift[None, None, :]) & 1
    signs = np.where(signs == 0, np.float32(1.0), np.float32(-1.0))  # [n, 32, 8]

    # 64 subgroups of 4 -> 32 subgroups of 8: index 2l -> elements 0-3,
    # index 2l+1 -> elements 4-7, within 8 groups of 32.
    gv = grid[idx].astype(np.float32)                          # [n, 64, 4]
    gv = gv.reshape(n, 8, 4, 2, 4).reshape(n, 8, 4, 8)
    signs = signs.reshape(n, 8, 4, 8)
    return (db[:, :, None, None] * gv * signs).reshape(n, 256)


def dequant_iq2_s(data: np.ndarray) -> np.ndarray:
    """data: [n, 82] -> [n, 256]  (port of dequantize_row_iq2_s)"""
    grid, ksigns = _iq2s_tables()
    n = data.shape[0]
    d = decode_fp16(data[:, 0:2].view(np.uint16)).reshape(n, 1)
    qs = data[:, 2:34]                                          # [n, 32]
    signs_raw = data[:, 34:66]                                  # [n, 32]
    qh = data[:, 66:74]                                         # [n, 8]
    sc = data[:, 74:82]                                         # [n, 8]

    sc = (sc.reshape(n, 8, 1) >> np.array([0, 4], np.uint8)[None, None, :])
    sc = (sc & 0x0F).reshape(n, 16).astype(np.float32)
    db = d * (0.5 + sc) * 0.25                                  # [n, 16]

    bits = ((qh.reshape(n, 8, 1) >> np.array([0, 2, 4, 6], np.uint8)[None, None, :])
            & 3).reshape(n, 32)
    idx = qs.astype(np.uint16) | (bits.astype(np.uint16) << 8)  # [n, 32]

    sbyte = signs_raw.reshape(n, 32)
    shift = np.arange(8, dtype=np.uint8)
    signs = (sbyte[:, :, None] >> shift[None, None, :]) & 1
    signs = np.where(signs == 0, np.float32(1.0), np.float32(-1.0))  # [n, 32, 8]

    gv = grid[idx].astype(np.float32)                          # [n, 32, 8]
    gv = gv.reshape(n, 16, 2, 8)
    signs = signs.reshape(n, 16, 2, 8)
    return (db[:, :, None, None] * gv * signs).reshape(n, 256)


# ---------------------------------------------------------------------------
# IQ1_S (1.5625 bpw) and IQ1_M (1.75 bpw)
# ---------------------------------------------------------------------------

def _iq1_tables():
    from gguf.quants import IQ1_S, IQ1_M
    IQ1_S.init_grid()
    IQ1_M.init_grid()
    g1s = np.ascontiguousarray(IQ1_S.grid).reshape(2048, 8).astype(np.int8)
    g1m = np.ascontiguousarray(IQ1_M.grid).reshape(2048, 8).astype(np.int8)
    return g1s, g1m, float(IQ1_S.delta)


def dequant_iq1_s(data: np.ndarray) -> np.ndarray:
    """data: [n, 50] -> [n, 256]  (port of dequantize_row_iq1_s)"""
    grid, _, delta = _iq1_tables()
    n = data.shape[0]
    d = decode_fp16(data[:, 0:2].view(np.uint16)).reshape(n, 1)
    qs = data[:, 2:34]                                          # [n, 32]
    qh = np.ascontiguousarray(data[:, 34:50]).view(np.uint16)   # [n, 8]

    dl = d * (2.0 * ((qh >> 12) & 7).astype(np.float32) + 1.0)  # [n, 8]
    delta = np.where((qh & 0x8000) == 0, delta, -delta).astype(np.float32)

    bits = ((qh.reshape(n, 8, 1) >> np.array([0, 3, 6, 9], np.uint16)[None, None, :])
            & 7).reshape(n, 32)
    idx = qs.astype(np.uint16) | (bits.astype(np.uint16) << 8)  # [n, 32]

    gv = grid[idx].astype(np.float32) + np.repeat(delta, 4, axis=1)[:, :, None]  # [n, 32, 8]
    gv = gv.reshape(n, 8, 4, 8)
    return (dl[:, :, None, None] * gv).reshape(n, 256)


def dequant_iq1_m(data: np.ndarray) -> np.ndarray:
    """data: [n, 56] -> [n, 256]  (port of dequantize_row_iq1_m)

    Note: the fp16 scale is packed across 4 uint16 words (4 bits each).
    """
    _, grid, delta = _iq1_tables()
    n = data.shape[0]
    qs = data[:, 0:32]                                           # [n, 32]
    qh = data[:, 32:48]                                          # [n, 16]
    sc = np.ascontiguousarray(data[:, 48:56]).view(np.uint16)    # [n, 4]

    dw = ((sc.reshape(n, 4) & np.uint16(0xF000))
          >> np.array([12, 8, 4, 0], np.uint16)[None, :])
    dw = (dw[:, 0] | dw[:, 1] | dw[:, 2] | dw[:, 3])             # [n]
    d = decode_fp16(np.ascontiguousarray(dw.reshape(-1, 1))).reshape(n, 1)

    scv = ((sc.reshape(n, 4, 1) >> np.array([0, 3, 6, 9], np.uint16)[None, None, :])
           & 7).reshape(n, 16).astype(np.float32)
    dl = d * (2.0 * scv + 1.0)                                   # [n, 16]

    nib = ((qh.reshape(n, 16, 1) >> np.array([0, 4], np.uint8)[None, None, :])
           & 0x0F).reshape(n, 32)
    bits = nib & 7
    idx = qs.astype(np.uint16) | (bits.astype(np.uint16) << 8)   # [n, 32]
    sgn = np.where((nib & 8) == 0, delta, -delta).astype(np.float32)  # [n, 32]

    gv = (grid[idx].astype(np.float32) + sgn[:, :, None])        # [n, 32, 8]
    gv = gv.reshape(n, 8, 2, 2, 8)
    return (dl.reshape(n, 8, 2, 1, 1) * gv).reshape(n, 256)


# ---------------------------------------------------------------------------
# TQ1_0 / TQ2_0 (ternary quants)
# ---------------------------------------------------------------------------

def dequant_tq1_0(data: np.ndarray) -> np.ndarray:
    """data: [n, 54] -> [n, 256]  (port of dequantize_row_tq1_0)

    Exact transcription of gguf-py's unpacking: packed base-3 digits
    are expanded with powers of 3, flattened, then rounded to -1/0/1.
    """
    n = data.shape[0]
    qs = data[:, 0:48]
    qh = data[:, 48:52]
    d = decode_fp16(data[:, 52:54].view(np.uint16)).reshape(n, 1)

    qs0, qs1 = qs[:, :32], qs[:, 32:]
    m5 = np.array([1, 3, 9, 27, 81], np.uint8).reshape(1, 1, 5, 1)
    qs0 = ((qs0.reshape(n, -1, 1, 32) * m5)          # uint8 wraps mod 256
           .reshape(n, -1).astype(np.uint16))
    qs1 = ((qs1.reshape(n, -1, 1, 16) * m5)
           .reshape(n, -1).astype(np.uint16))
    m4 = np.array([1, 3, 9, 27], np.uint8).reshape(1, 1, 4, 1)
    qh = ((qh.reshape(n, -1, 1, 4) * m4)
          .reshape(n, -1).astype(np.uint16))
    q = np.concatenate([qs0, qs1, qh], axis=-1)          # [n, 256]
    q = ((q * 3) >> 8).astype(np.int8).astype(np.float32) - 1.0
    return d * q


def dequant_tq2_0(data: np.ndarray) -> np.ndarray:
    """data: [n, 66] -> [n, 256]  (port of dequantize_row_tq2_0)"""
    n = data.shape[0]
    qs = data[:, 0:64]
    d = decode_fp16(data[:, 64:66].view(np.uint16)).reshape(n, 1)
    q = ((qs.reshape(n, -1, 1, 32).astype(np.uint8)
          >> np.array([0, 2, 4, 6], np.uint8).reshape(1, 1, 4, 1)) & 3)
    q = q.reshape(n, -1).astype(np.int8).astype(np.float32) - 1.0
    return d * q


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

DEQUANT = {
    "F32": lambda d: d.view(np.float32),
    "F16": lambda d: decode_fp16(d.view(np.uint16)),
    "BF16": lambda d: decode_bf16(d.view(np.uint16)),
    "Q8_0": dequant_q8_0,
    "Q4_0": dequant_q4_0,
    "Q2_K": dequant_q2_K,
    "Q3_K": dequant_q3_K,
    "Q4_K": dequant_q4_K,
    "Q5_K": dequant_q5_K,
    "Q6_K": dequant_q6_K,
    "MXFP4": dequant_mxfp4,
    "IQ2_XXS": dequant_iq2_xxs,
    "IQ2_XS": dequant_iq2_xs,
    "IQ3_XXS": dequant_iq3_xxs,
    "Q4_1": dequant_q4_1,
    "Q5_0": dequant_q5_0,
    "Q5_1": dequant_q5_1,
    "IQ4_NL": dequant_iq4_nl,
    "IQ4_XS": dequant_iq4_xs,
    "IQ3_S": dequant_iq3_s,
    "IQ2_S": dequant_iq2_s,
    "IQ1_S": dequant_iq1_s,
    "IQ1_M": dequant_iq1_m,
    "TQ1_0": dequant_tq1_0,
    "TQ2_0": dequant_tq2_0,
}

BLOCK_ALIGN = {
    "F32": 1, "F16": 1, "BF16": 1,
    "Q8_0": 32, "Q4_0": 32, "Q4_1": 32, "Q5_0": 32, "Q5_1": 32,
    "Q2_K": 256, "Q3_K": 256, "Q4_K": 256, "Q5_K": 256, "Q6_K": 256,
    "MXFP4": 32,
    "IQ2_XXS": 256, "IQ2_XS": 256, "IQ3_XXS": 256,
    "IQ4_NL": 32, "IQ4_XS": 256,
    "IQ3_S": 256, "IQ2_S": 256, "IQ1_S": 256, "IQ1_M": 256,
    "TQ1_0": 256, "TQ2_0": 256,
}

# Serialized bytes per block (per GGUF spec)
BLOCK_BYTES = {
    "Q8_0": 34, "Q4_0": 18, "Q4_1": 20, "Q5_0": 22, "Q5_1": 24,
    "Q2_K": 84, "Q3_K": 110, "Q4_K": 144, "Q5_K": 176, "Q6_K": 210,
    "MXFP4": 17,
    "IQ2_XXS": 66, "IQ2_XS": 74, "IQ3_XXS": 98,
    "IQ4_NL": 18, "IQ4_XS": 136,
    "IQ3_S": 110, "IQ2_S": 82, "IQ1_S": 50, "IQ1_M": 56,
    "TQ1_0": 54, "TQ2_0": 66,
}


def dequantize(quant_type: str, data: np.ndarray, logical_shape: tuple, chunk_blocks: int = 65536):
    """Dequantize a raw GGUF tensor buffer to float32.

    Args:
        quant_type: GGUF type name (e.g. 'Q3_K').
        data: raw uint8 buffer from tensor.data. For quantized types the
              gguf package returns this as [rows, blocks_per_row * blocksz].
        logical_shape: the tensor's logical element shape as reported by the
              gguf package (reversed NE order; we return the true order).

    Returns:
        float32 array with the TRUE row-major shape (logical_shape reversed).
    """
    fn = DEQUANT.get(quant_type)
    if fn is None:
        raise ValueError(f"unsupported quant type: {quant_type}")
    true_shape = tuple(reversed(logical_shape))
    n = int(np.prod(true_shape))
    if quant_type in ("F32", "F16", "BF16"):
        return fn(np.ascontiguousarray(data.reshape(-1))).reshape(true_shape)

    blocksz = BLOCK_BYTES[quant_type]
    rows = true_shape[0] if len(true_shape) > 1 else 1
    cols = n // rows
    blocks_per_row = cols // BLOCK_ALIGN[quant_type]
    flat = np.ascontiguousarray(data).reshape(rows * blocks_per_row, blocksz)
    n_blocks = flat.shape[0]

    out = np.empty((n_blocks * BLOCK_ALIGN[quant_type],), dtype=np.float32)
    for s in range(0, n_blocks, chunk_blocks):
        e = min(n_blocks, s + chunk_blocks)
        out[s * BLOCK_ALIGN[quant_type]: e * BLOCK_ALIGN[quant_type]] = (
            fn(flat[s:e]).reshape(-1)
        )
    return out[:n].reshape(true_shape)


def dequantize_rows(quant_type: str, data: np.ndarray, logical_shape: tuple,
                    row_start: int = 0, row_count: int = 1,
                    chunk_blocks: int = 65536):
    """Dequantize a slice of rows -> [row_count, cols] float32.

    Streaming-friendly: only the requested rows are materialized in
    float32, so enormous tensors (e.g. MXFP4 expert tensors with
    billions of elements) can be processed one row at a time.
    """
    fn = DEQUANT.get(quant_type)
    if fn is None:
        raise ValueError(f"unsupported quant type: {quant_type}")
    true_shape = tuple(reversed(logical_shape))
    n = int(np.prod(true_shape))
    rows_total = true_shape[0] if len(true_shape) > 1 else 1
    cols = n // rows_total

    if quant_type in ("F32", "F16", "BF16"):
        dense = fn(np.ascontiguousarray(data.reshape(-1))).reshape(true_shape)
        return np.ascontiguousarray(
            dense[row_start:row_start + row_count]
        ).reshape(row_count, cols)

    blocksz = BLOCK_BYTES[quant_type]
    blocks_per_row = cols // BLOCK_ALIGN[quant_type]
    flat = np.ascontiguousarray(data).reshape(rows_total * blocks_per_row, blocksz)
    s0 = row_start * blocks_per_row
    s1 = (row_start + row_count) * blocks_per_row
    n_blocks = s1 - s0

    out = np.empty((n_blocks * BLOCK_ALIGN[quant_type],), dtype=np.float32)
    for s in range(s0, s1, chunk_blocks):
        e = min(s1, s + chunk_blocks)
        o0 = (s - s0) * BLOCK_ALIGN[quant_type]
        o1 = (e - s0) * BLOCK_ALIGN[quant_type]
        out[o0:o1] = fn(flat[s:e]).reshape(-1)
    return out.reshape(row_count, cols)
