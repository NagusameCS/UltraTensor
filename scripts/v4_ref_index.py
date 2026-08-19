"""Indexer oracle (Milestone G): official Indexer semantics for layer 2
(ratio 4) on REAL V4 bytes - q path (rope + Hadamard rotate + fp4),
indexer's own rotated compressor, weights_proj scoring, causal mask,
topk. 8-token prefill then 4 decode tokens (compress at p=11).

Writes:
  outputs/idx_score_pre.bin  [8][3]  prefill scores after causal mask
  outputs/idx_topk_pre.bin   [8][2]  topk idxs (after offset, -1 padded)
  outputs/idx_score_dec.bin  [4][3]  decode scores (p=8..11)
  outputs/idx_topk_dec.bin   [4][3]  decode topk idxs (offset 128 added)
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import v4_ref_block as vb  # noqa: E402

DIM, RATIO, COFF, HD, RD = 7168, 4, 2, 128, 64
EPS = 1e-6
NH = 64
TOP = 1024
WIN = 128
FREQS = vb.rope_freqs(RD, 160000.0, 16.0, 32.0, 1.0, 65536.0)
PRE, DEC = 8, 4
SCALE = HD ** -0.5 * NH ** -0.5

FP4 = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], np.float64)


def hadamard(x):
    """Natural-order normalized Hadamard over the last dim (pow2)."""
    n = x.shape[-1]
    y = x.astype(np.float64).copy()
    k = 1
    while k < n:
        y = y.reshape(-1, n // (2 * k), 2, k)
        a, b = y[..., 0, :].copy(), y[..., 1, :].copy()
        y[..., 0, :] = a + b
        y[..., 1, :] = a - b
        k *= 2
    return (y.reshape(x.shape) * (n ** -0.5)).astype(np.float32)


def fp4_quant(x, block=32):
    """Block-wise e2m1 fake quant, power-of-2 scale (kernel.py)."""
    y = x.astype(np.float64).reshape(-1)
    out = y.copy()
    for b0 in range(0, y.size, block):
        v = y[b0:b0 + block]
        amax = max(float(np.abs(v).max()), 6 * 2 ** -126)
        s = 2.0 ** np.ceil(np.log2(amax / 6.0))
        q = np.clip(v / s, -6.0, 6.0)
        a = np.abs(q)
        idx = np.abs(a[:, None] - FP4[None, :]).argmin(1)
        qr = FP4[idx]
        out[b0:b0 + block] = qr * s * np.sign(q)
    return out.astype(np.float32).reshape(x.shape)


def rope_rows(v, positions):
    ang = np.asarray(positions, np.float64)[:, None] * FREQS[None, :]
    c, s = np.cos(ang), np.sin(ang)
    a = v[:, 0::2].copy()
    b = v[:, 1::2].copy()
    v[:, 0::2] = a * c - b * s
    v[:, 1::2] = a * s + b * c
    return v


def stable_softmax(s, axis):
    s = s - s.max(axis, keepdims=True)
    e = np.exp(s)
    return e / e.sum(axis, keepdims=True)


def emit_rot(kv, norm, positions):
    out = kv * (1.0 / np.sqrt((kv * kv).mean(-1, keepdims=True) + EPS))
    out = out * norm
    rope_rows(out[:, -RD:], positions)
    out = hadamard(out)
    out = fp4_quant(out)
    return out


class RotComp:
    """Indexer's compressor: head_dim 128, rotate=True, overlap."""

    def __init__(self, wkv, wgate, ape, norm):
        self.wkv, self.wgate = wkv, wgate
        self.ape, self.norm = ape, norm
        self.ks = np.zeros((2 * RATIO, 2 * HD), np.float32)
        self.ss = np.full((2 * RATIO, 2 * HD), -np.inf, np.float32)

    def prefill(self, x):
        S = x.shape[0]
        kv = x @ self.wkv.T
        score = x @ self.wgate.T
        cutoff = S - S % RATIO
        if cutoff >= RATIO:
            self.ks[:RATIO] = kv[cutoff - RATIO:cutoff]
            self.ss[:RATIO] = score[cutoff - RATIO:cutoff] + self.ape
        kv = kv[:cutoff].reshape(-1, RATIO, 2 * HD)
        score = score[:cutoff].reshape(-1, RATIO, 2 * HD) + self.ape
        new = np.zeros((kv.shape[0], 2 * RATIO, HD), np.float32)
        ns = np.full((kv.shape[0], 2 * RATIO, HD), -np.inf, np.float32)
        new[:, RATIO:] = kv[:, :, HD:]
        new[1:, :RATIO] = kv[:-1, :, :HD]
        ns[:, RATIO:] = score[:, :, HD:]
        ns[1:, :RATIO] = score[:-1, :, :HD]
        w = stable_softmax(ns, axis=1)
        out = (new * w).sum(1)
        W = out.shape[0]
        return emit_rot(out, self.norm, np.arange(0, cutoff, RATIO)[:W])

    def decode(self, x, start_pos):
        kv = x @ self.wkv.T + 0.0
        score = x @ self.wgate.T + self.ape[start_pos % RATIO]
        slot = RATIO + start_pos % RATIO
        self.ks[slot] = kv
        self.ss[slot] = score
        if (start_pos + 1) % RATIO != 0:
            return None
        kvc = np.concatenate([self.ks[:RATIO, :HD], self.ks[RATIO:, HD:]],
                             axis=0)
        sc = np.concatenate([self.ss[:RATIO, :HD], self.ss[RATIO:, HD:]],
                            axis=0)
        w = stable_softmax(sc, axis=0)
        out = (kvc * w).sum(0, keepdims=True)
        self.ks[:RATIO] = self.ks[RATIO:]
        self.ss[:RATIO] = self.ss[RATIO:]
        self.ss[RATIO:] = -np.inf
        return emit_rot(out, self.norm, [start_pos + 1 - RATIO])


def q_path(qb, qr, pos):
    q = qr @ qb.T                      # [64, 128]
    q = q.reshape(NH, HD)
    rope_rows(q[:, -RD:], [pos])
    q = hadamard(q)
    q = fp4_quant(q)
    return q


def scores(q, cache, weights):
    """q [64,128], cache [T,128], weights [64] -> [T]"""
    return (np.clip(q @ cache.T, 0, None) * weights[:, None]).sum(0)


def main() -> int:
    t = vb.tensors()
    names = ["blk.2.indexer.attn_q_b.weight", "blk.2.indexer.proj.weight",
             "blk.2.indexer_compressor_kv.weight",
             "blk.2.indexer_compressor_gate.weight",
             "blk.2.indexer_compressor_ape.weight",
             "blk.2.indexer_compressor_norm.weight"]
    for n in names:
        assert n in t, n
    qb = vb.load(t, names[0])          # [8192, 1536]
    proj = vb.load(t, names[1])        # [64, 7168]
    rc = RotComp(vb.load(t, names[2]), vb.load(t, names[3]),
                 vb.load(t, names[4]), vb.load(t, names[5]).astype(np.float32))

    rng = np.random.default_rng(777)
    xp = rng.standard_normal((PRE, DIM)).astype(np.float32)
    xd = rng.standard_normal((DEC, DIM)).astype(np.float32)
    qrp = rng.standard_normal((PRE, 1536)).astype(np.float32)
    qrd = rng.standard_normal((DEC, 1536)).astype(np.float32)
    for f, v in (("idx_x_pre.bin", xp), ("idx_x_dec.bin", xd),
                 ("idx_qr_pre.bin", qrp), ("idx_qr_dec.bin", qrd)):
        (ROOT / "outputs" / f).write_bytes(v.tobytes())

    cache = np.zeros((8, HD), np.float32)
    cache[:PRE // RATIO] = rc.prefill(xp)
    T = PRE // RATIO
    sp = np.zeros((PRE, 8), np.float32)
    for s in range(PRE):
        w = proj @ xp[s] * SCALE
        sp[s, :T] = scores(q_path(qb, qrp[s], s), cache[:T], w)
        sp[s, (np.arange(8) >= (s + 1) // RATIO)] = -np.inf
    tp = np.full((PRE, 2), -1, np.int64)
    for s in range(PRE):
        k = min(TOP, (s + 1) // RATIO)
        order = np.argsort(-sp[s, :T])[:k]
        tp[s, :k] = order + PRE          # prefill offset = seqlen
    sd = np.zeros((DEC, 8), np.float32)
    td = np.full((DEC, 4), -1, np.int64)
    for i in range(DEC):
        p = PRE + i
        row = rc.decode(xd[i:i + 1], p)
        if row is not None:
            cache[p // RATIO] = row
        Td = (p + 1) // RATIO
        w = proj @ xd[i] * SCALE
        sd[i, :Td] = scores(q_path(qb, qrd[i], p), cache[:Td], w)
        k = min(TOP, Td)
        order = np.argsort(-sd[i, :Td])[:k]
        td[i, :k] = order + WIN
        sd[i, Td:] = -np.inf
    (ROOT / "outputs" / "idx_score_pre.bin").write_bytes(sp.tobytes())
    (ROOT / "outputs" / "idx_topk_pre.bin").write_bytes(tp.tobytes())
    (ROOT / "outputs" / "idx_score_dec.bin").write_bytes(sd.tobytes())
    (ROOT / "outputs" / "idx_topk_dec.bin").write_bytes(td.tobytes())
    print("pre scores", sp.shape, "sum %.4f" % sp.sum(),
          "| dec scores", sd.shape, "sum %.4f" % sd.sum())
    print("pre topk", tp.tolist()[:3])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
