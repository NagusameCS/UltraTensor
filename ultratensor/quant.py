"""UltraTensor quantizers: numpy block quantizers for the streaming pipeline.

q8_0 reuses HyperRetro's kernel implementation when the HyperTensor repo
is importable (HyperRetro integration), with a built-in numpy fallback.
q4_0 and uq4 (symmetric per-block int4) are native numpy implementations.
"""

from __future__ import annotations

import numpy as np


def q8_0_quantize(W: np.ndarray):
    """Quantize [rows, cols] to (scales[rows, cols/32], codes[rows, cols/32, 32])."""
    try:
        import sys
        sys.path.insert(0, r"C:\Users\legom\OneDrive\Documents\GitHub\HyperTensor")
        from hyperretro.kernels import q8_0_quantize as _impl  # HyperRetro integration
        return _impl(W)
    except Exception:
        pass
    W = np.ascontiguousarray(W, dtype=np.float32)
    rows, cols = W.shape
    assert cols % 32 == 0
    Wb = W.reshape(rows, cols // 32, 32)
    amax = np.max(np.abs(Wb), axis=-1)
    scale = amax / 127.0
    scale = np.where(scale == 0, 1.0, scale)
    codes = np.round(Wb / scale[..., None]).clip(-128, 127).astype(np.int8)
    return scale.astype(np.float32), codes


def q4_0_quantize(W: np.ndarray):
    """Quantize [rows, cols] to (scales[rows, cols/32], nibbles[rows, cols/32, 16])."""
    W = np.ascontiguousarray(W, dtype=np.float32)
    rows, cols = W.shape
    assert cols % 32 == 0
    Wb = W.reshape(rows, cols // 32, 32)
    amax = np.max(np.abs(Wb), axis=-1)
    scale = amax / 8.0
    scale = np.where(scale == 0, 1.0, scale)
    q = np.round(Wb / scale[..., None] + 8.0).clip(0, 15).astype(np.uint8)
    return scale.astype(np.float32), q


def q4_0_dequantize(scale: np.ndarray, nibbles: np.ndarray) -> np.ndarray:
    return ((nibbles.astype(np.float32) - 8.0) * scale[..., None]).reshape(scale.shape[0], -1)


def uq4_quantize(W: np.ndarray, block: int = 128):
    """Symmetric per-block int4: zero-centered grid, exact zeros.

    grid = scale * (q - 8) for q in 0..15, scale = amax/8 per block.
    Zero is always representable exactly (q = 8), which matters for
    sparse expert tensors.

    Returns (scales[rows, cols/block], packed[rows, cols/2]) where each
    byte holds two 4-bit codes (low = element 2k, high = element 2k+1).
    """
    W = np.ascontiguousarray(W, dtype=np.float32)
    rows, cols = W.shape
    assert cols % block == 0
    Wb = W.reshape(rows, cols // block, block)
    amax = np.abs(Wb).max(axis=-1, keepdims=True)
    scale = amax / 8.0
    scale = np.where(scale == 0, 1.0, scale)
    q = np.round(Wb / scale + 8.0).clip(0, 15).astype(np.uint8)
    q = q.reshape(rows, cols // 2, 2)
    packed = (q[..., 0] | (q[..., 1] << 4)).astype(np.uint8)
    return scale[..., 0].astype(np.float32), packed


def uq4_dequantize(scales: np.ndarray, packed: np.ndarray) -> np.ndarray:
    """Dequantize symmetric per-block int4: [rows, cols] float32."""
    rows, cols2 = packed.shape
    cols = cols2 * 2
    block = cols // scales.shape[1]
    lo = (packed & 0x0F).astype(np.float32)
    hi = (packed >> 4).astype(np.float32)
    vals = np.stack([lo, hi], axis=-1).reshape(rows, cols)
    return (vals - 8.0) * np.repeat(scales, block, axis=1)


def q2_0_quantize(W: np.ndarray, block: int = 32):
    """Symmetric 2-bit, block-of-32: grid = scale * {-3, -1, 1, 3}.

    Codes are packed 4 per byte (elements 0-3 in one byte).
    Returns (scales[rows, cols/block] fp16, packed[rows, cols/4] uint8).
    """
    W = np.ascontiguousarray(W, dtype=np.float32)
    rows, cols = W.shape
    assert cols % block == 0 and block % 4 == 0
    Wb = W.reshape(rows, cols // block, block)
    amax = np.abs(Wb).max(axis=-1, keepdims=True)
    scale = amax / 3.0
    scale = np.where(scale == 0, 1.0, scale)
    q = np.round((Wb / scale + 3.0) / 2.0).clip(0, 3).astype(np.uint8)  # 0..3
    q = q.reshape(rows, cols // 4, 4)
    packed = (q[..., 0] | (q[..., 1] << 2) | (q[..., 2] << 4)
              | (q[..., 3] << 6)).astype(np.uint8)
    return (scale[..., 0].astype(np.float16), packed)


def q2_0_dequantize(scales: np.ndarray, packed: np.ndarray) -> np.ndarray:
    """Dequantize symmetric 2-bit: [rows, cols] float32."""
    rows, cols4 = packed.shape
    cols = cols4 * 4
    block = cols // scales.shape[1]
    q = (packed[:, :, None] >> np.array([0, 2, 4, 6], np.uint8)[None, None, :]) & 3
    vals = q.reshape(rows, cols).astype(np.float32) * 2.0 - 3.0
    sc = scales.astype(np.float32)
    return vals * np.repeat(sc, block, axis=1)


# ---------------------------------------------------------------------------
# Q2_K (llama.cpp-compatible 2.5625 bpw) quantizer
# ---------------------------------------------------------------------------

def q2_k_quantize(W: np.ndarray, chunk_blocks: int = 256, use_torch: bool = False):
    """Quantize [rows, cols] (cols % 256 == 0) to llama.cpp Q2_K.

    Returns (scales[B, 16] uint8, qs[B, 64] uint8, d[B] fp16,
    dmin[B] fp16) — byte-exact layout of block_q2_K, decodable by
    dequant_q2_K AND by llama.cpp.

    Exact port of llama.cpp quantize_row_q2_K_ref (no imatrix): per
    16-element subgroup an affine fit (scale*code - min, codes 0..3) via
    make_qkx2_quants' 16-step grid search; d/dmin from the subgroup
    maxima; scales[j] nibbles = (scale quant, min quant).
    """
    W = np.ascontiguousarray(W, dtype=np.float32)
    rows, cols = W.shape
    assert cols % 256 == 0
    nb = cols // 256
    B = rows * nb
    xg = W.reshape(B, 16, 16)                      # [B, subgroup, elem]
    w = np.abs(xg)
    mn = xg.min(axis=-1)                           # [B, 16]
    mx = xg.max(axis=-1)
    mn = np.where(mn > 0, 0.0, mn)

    iscale = 3.0 / np.maximum(mx - mn, 1e-30)
    L = np.clip(np.round(iscale[..., None] * (xg - mn[..., None])), 0, 3) \
        .astype(np.float32)
    scale = 1.0 / iscale
    mad = (w * np.abs(scale[..., None] * L - xg)).sum(axis=-1)   # [B,16]
    best_scale = scale.copy()
    best_min = -mn
    best_L = L

    # make_qkx2_quants: nstep=15, rmin=-0.5, rdelta=0.1, nmax=3, use_mad
    sum_w = w.sum(axis=-1)                         # [B,16]
    sum_x = (w * xg).sum(axis=-1)
    for is_ in range(16):
        isc = (2.5 + 0.1 * is_) / np.maximum(mx - mn, 1e-30)
        Laux = np.clip(np.round(isc[..., None] * (xg - mn[..., None])), 0, 3) \
            .astype(np.float32)
        sum_l = (w * Laux).sum(axis=-1)
        sum_l2 = (w * Laux * Laux).sum(axis=-1)
        sum_xl = (w * xg * Laux).sum(axis=-1)
        D = sum_w * sum_l2 - sum_l * sum_l
        ok = D > 0
        this_scale = np.where(ok, (sum_w * sum_xl - sum_x * sum_l)
                              / np.where(ok, D, 1.0), 0.0)
        this_min = np.where(ok, (sum_l2 * sum_x - sum_l * sum_xl)
                            / np.where(ok, D, 1.0), 0.0)
        pos = this_min > 0
        this_min = np.where(pos, 0.0, this_min)
        this_scale = np.where(pos, sum_xl / np.where(sum_l2 > 0, sum_l2, 1.0),
                              this_scale)
        cur = (w * np.abs(this_scale[..., None] * Laux
                          + this_min[..., None] - xg)).sum(axis=-1)
        better = cur < mad
        mad = np.where(better, cur, mad)
        best_scale = np.where(better, this_scale, best_scale)
        best_min = np.where(better, -this_min, best_min)
        best_L = np.where(better[..., None], Laux, best_L)

    max_scale = np.maximum(best_scale.max(axis=-1), 1e-30)      # [B]
    max_min = np.maximum(best_min.max(axis=-1), 0.0)

    scales = np.empty((B, 16), np.uint8)
    s_l = np.clip(np.round(15.0 * best_scale / max_scale[:, None]), 0, 15)
    m_l = np.clip(np.round(15.0 * best_min
                           / np.maximum(max_min[:, None], 1e-30)), 0, 15)
    scales = (s_l.astype(np.uint8) | (m_l.astype(np.uint8) << 4))

    d = (max_scale / 15.0).astype(np.float16)
    dmin = (max_min / 15.0).astype(np.float16)

    # final codes: l = round((x + dm)/dl), dl = d*(sc&0xF), dm = dmin*(sc>>4)
    dl = d[:, None].astype(np.float32) * s_l          # [B,16]
    dm = dmin[:, None].astype(np.float32) * m_l
    Lf = np.clip(np.round((xg + dm[..., None]) / dl[..., None]), 0, 3) \
        .astype(np.uint8)
    # pack per 128-group: byte holds codes of j, j+32, j+64, j+96
    Lg = Lf.reshape(B, 2, 128)                       # [B, group, elem]
    qs = (Lg[..., 0:32]
          | (Lg[..., 32:64] << 2)
          | (Lg[..., 64:96] << 4)
          | (Lg[..., 96:128] << 6)).reshape(B, 64).astype(np.uint8)
    return scales, qs, d, dmin


def q2_k_quantize_np(W: np.ndarray, chunk_blocks: int = 256):
    """Quantize [rows, cols] (cols % 256 == 0) to llama.cpp Q2_K.

    Returns (scales[rows, nb, 16] uint8, qs[rows, nb, 64] uint8,
    d[rows, nb] fp16, dmin[rows, nb] fp16) — byte-exact layout of
    block_q2_K, decodable by dequant_q2_K AND by llama.cpp.

    Per 256-block: value = d*(m*code - n) with dmin=d, m/n nibbles per
    16-element subgroup, code in 0..3. (m, n) chosen per subgroup by an
    exhaustive 256-grid search minimizing squared error (vectorized over
    the n axis, looped over m).
    """
    W = np.ascontiguousarray(W, dtype=np.float32)
    rows, cols = W.shape
    assert cols % 256 == 0
    nb = cols // 256
    B = rows * nb
    Wb = W.reshape(B, 256)
    amax = np.abs(Wb).max(axis=-1)
    d = np.where(amax == 0, 1.0, amax / 3.0).astype(np.float32)

    # element -> subgroup mapping: s = g*8 + j*2 + half (g,j,half,l axes
    # flatten in C order to exactly s), so reshape gives [s, l] directly
    xs_all = Wb.reshape(B, 16, 16)                   # [B, subgroup, elem]

    n_grid = np.arange(16, dtype=np.float32)
    m_grid = np.arange(16, dtype=np.float32)
    out_scales = np.empty((B, 16), np.uint8)
    out_qs = np.empty((B, 64), np.uint8)
    for b0 in range(0, B, chunk_blocks):
        b1 = min(B, b0 + chunk_blocks)
        c = b1 - b0
        xs = xs_all[b0:b1]                           # [c, 16, 16]
        dd = d[b0:b1].reshape(c, 1, 1)               # [c, 1, 1]
        # full (m, n) grid in one shot: [c, m, n, s, e]
        mg = m_grid[None, :, None, None, None].astype(np.float32)
        ng = n_grid[None, None, :, None, None].astype(np.float32)
        denom = np.where(mg == 0, 1.0, mg * dd[:, None, None])   # [c,16,1,1,1]
        codes = np.clip(
            np.round((xs[:, None, None] + ng * dd[:, None, None])
                     / denom), 0, 3).astype(np.uint8)  # [c,16,16,16,16]
        pred = (mg * dd[:, None, None] * codes.astype(np.float32)
                - ng * dd[:, None, None]).astype(np.float32)
        err = ((pred - xs[:, None, None]) ** 2).sum(axis=-1)  # [c,16,16,16]
        flat = err.reshape(c, 256, 16)
        best = np.argmin(flat, axis=1)               # [c, 16(s)]
        best_m = (best // 16).astype(np.uint8)
        best_n = (best % 16).astype(np.uint8)
        idx = (np.arange(c)[:, None, None],          # [c,1,1]
               best_m[:, :, None],                    # [c,16,1]
               best_n[:, :, None],                    # [c,16,1]
               np.arange(16)[None, :, None],          # [1,16,1]
               np.arange(16)[None, None, :])          # [1,1,16]
        best_c = codes[idx]                          # [c, 16(s), 16(e)]
        out_scales[b0:b1] = (best_m | (best_n << 4)).astype(np.uint8)
        # pack: byte (g,half,l) gets codes of j=0..3 at bits 2j
        cp = best_c.reshape(c, 2, 4, 2, 16).transpose(0, 1, 3, 4, 2) \
            .astype(np.uint16)                       # [c, g, half, l, j]
        packed = (cp * np.array([1, 4, 16, 64], np.uint16)).sum(axis=-1)
        out_qs[b0:b1] = packed.reshape(c, 64).astype(np.uint8)
    dmin = d.copy()
    return (out_scales, out_qs, d.astype(np.float16),
            dmin.astype(np.float16))


def q2_k_quantize_torch(W: np.ndarray, chunk_blocks: int = 1024):
    """CUDA (or CPU) torch implementation of the Q2_K grid search.

    Uses the exact decomposition
        err[c,m,n,s] = (m*d)^2 * S1 - 2*(m*d) * Sx + T[n,s]
    with S1 = sum_e code, Sx = sum_e code*x, T = sum_e (x + n*d)^2
    (T independent of m, S1/Sx via one batched matmul) so the big
    [c,16,16,16,16] intermediate never materializes as fp32.
    """
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    W = np.ascontiguousarray(W, dtype=np.float32)
    rows, cols = W.shape
    assert cols % 256 == 0
    B = rows * (cols // 256)
    Wb = torch.from_numpy(W.reshape(B, 256)).to(dev)
    amax = Wb.abs().max(dim=-1).values
    d = torch.where(amax == 0, torch.ones_like(amax), amax / 3.0)
    xs_all = Wb.reshape(B, 16, 16)                      # [B, subgroup, elem]

    mg = torch.arange(16, device=dev, dtype=torch.float32).view(1, 16, 1, 1, 1)
    ng = torch.arange(16, device=dev, dtype=torch.float32).view(1, 1, 16, 1, 1)
    ng2 = torch.arange(16, device=dev, dtype=torch.float32).view(1, 16, 1)
    out_scales = torch.empty(B, 16, dtype=torch.uint8)
    out_qs = torch.empty(B, 64, dtype=torch.uint8)
    out_d = torch.empty(B, dtype=torch.float16)
    pack_vec = torch.tensor([1, 4, 16, 64], device=dev, dtype=torch.int32)
    for b0 in range(0, B, chunk_blocks):
        b1 = min(B, b0 + chunk_blocks)
        c = b1 - b0
        xs = xs_all[b0:b1].view(c, 1, 1, 16, 16)        # [c,1,1,16,16]
        dd = d[b0:b1].view(c, 1, 1, 1, 1)               # [c,1,1,1,1]
        denom = torch.where(mg == 0, torch.ones_like(mg), mg * dd)
        codes = torch.clip(torch.round((xs + ng * dd) / denom), 0, 3) \
            .to(torch.uint8)                            # [c,16,16,16,16]
        # exact decomposition of sum_e (m*d*code - n*d - x)^2:
        #   (m*d)^2*S2 - 2*m*d*(n*d*S1 + Sx) + T[n,s]
        # with S2=sum code^2, S1=sum code, Sx=sum code*x, T=sum (x+n*d)^2
        xr = xs_all[b0:b1]                              # [c, 16(s), 16(e)]
        codes5 = codes.permute(0, 1, 2, 4, 3) \
            .reshape(c, 256, 16, 16)                    # [c, mn, e, s]
        xrT = xr.transpose(1, 2).view(c, 1, 16, 16)     # [c, 1, e, s]
        S1 = codes5.sum(dim=2, dtype=torch.int32)       # [c,256,16(s)]
        S2 = (codes5 * codes5).sum(dim=2, dtype=torch.int32)  # [c,256,16(s)]
        Sx = (codes5.float() * xrT).sum(dim=2)          # [c,256,16(s)]
        # T[n,s] = sum_e (x + n*d)^2  (independent of m)
        X1 = xr.sum(dim=2)                              # [c,16(s)]
        X2 = (xr ** 2).sum(dim=2)                       # [c,16(s)]
        T = X2.view(c, 1, 16) + 2 * ng2 * dd.view(c, 1, 1) * X1.view(c, 1, 16) \
            + 16 * (ng2 * dd.view(c, 1, 1)) ** 2        # [c,16(n),16(s)]
        md = (mg * dd).view(c, 16, 1, 1)                # [c,m,1,1]
        ndg = (ng2 * dd.view(c, 1, 1)).view(c, 1, 16, 1)  # [c,1,n,1]
        err = md * md * S2.float().view(c, 16, 16, 16) \
            + T.view(c, 1, 16, 16) \
            - 2 * md * Sx.view(c, 16, 16, 16) \
            - 2 * (md * ndg) * S1.float().view(c, 16, 16, 16)  # [c,m,n,s]
        best = err.reshape(c, 256, 16).argmin(dim=1)    # [c,16]
        best_m = (best // 16).to(torch.int64)
        best_n = (best % 16).to(torch.int64)
        # gather chosen codes: codes2d[c, row=m*16+n, col=s*16+e]
        ci3 = torch.arange(c, device=dev).view(-1, 1, 1) \
            .expand(c, 16, 16)
        row3 = (best_m.view(c, 16, 1) * 16 + best_n.view(c, 16, 1)) \
            .expand(c, 16, 16)
        col3 = (torch.arange(16, device=dev).view(1, 16, 1) * 16
                + torch.arange(16, device=dev).view(1, 1, 16)).expand(c, 16, 16)
        best_c = codes.reshape(c, 256, 256)[ci3, row3, col3]  # [c,16,16]
        out_scales[b0:b1] = (best_m.to(torch.uint8)
                             | (best_n.to(torch.uint8) << 4))
        # pack: mirror the numpy layout exactly —
        #   s = g*8 + j*2 + h,  e = l*4 + j
        #   reshape(c,2,4,2,16).transpose(0,1,3,4,2) -> [c,g,h,e,j]
        cp = best_c.reshape(c, 2, 4, 2, 16).permute(0, 1, 3, 4, 2)
        packed = torch.matmul(cp.float(), pack_vec.float()).round()
        out_qs[b0:b1] = packed.reshape(c, 64).to(torch.uint8)
        out_d[b0:b1] = d[b0:b1].half()
    return (out_scales.cpu().numpy(), out_qs.cpu().numpy(),
            out_d.cpu().numpy(), out_d.cpu().numpy())
