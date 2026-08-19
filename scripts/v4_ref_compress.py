"""Compressor oracle (Milestone F): official Compressor semantics for a
ratio-4 layer (layer 2, overlap) on REAL V4 bytes, implemented in numpy
with the same dequant/act_quant/rope helpers already validated by
v4_ref_block.py (block reference, 3.4e-7 vs C).

Sequence: 12-token prefill (start_pos 0) -> compressed rows 0..2,
then 8 decode tokens (start_pos 12..19) -> compressions at p=15 (row 3)
and p=19 (row 4).

Writes:
  outputs/comp_cache_pre.bin  [3][512]  prefill compressed cache
  outputs/comp_cache_dec.bin  [2][512]  decode compressed rows
  outputs/comp_states.bin     [8+8][1024] final kv_state+score_state (diag)
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import v4_ref_block as vb  # noqa: E402

DIM, RATIO, COFF, HD, RD = 7168, 4, 2, 512, 64
EPS = 1e-6
LAYER = 2
FREQS = vb.rope_freqs(RD, 160000.0, 16.0, 32.0, 1.0, 65536.0)
PRE, DEC = 12, 8


def stable_softmax(s, axis):
    s = s - s.max(axis, keepdims=True)
    e = np.exp(s)
    return e / e.sum(axis, keepdims=True)


def overlap_transform(t, value):
    """t: [W, RATIO, 1024] -> [W, 2*RATIO, 512] (official semantics)."""
    W = t.shape[0]
    new = np.full((W, COFF * RATIO, HD), value, np.float32)
    new[:, RATIO:] = t[:, :, HD:]               # second half dims
    new[1:, :RATIO] = t[:-1, :, :HD]            # prev window first half
    return new


def rope_rows(v, positions):
    """v [W, RD]; rotate row i by positions[i] (like apply_rotary_emb)."""
    ang = np.asarray(positions, np.float64)[:, None] * FREQS[None, :]
    c, s = np.cos(ang), np.sin(ang)
    a = v[:, 0::2].copy()
    b = v[:, 1::2].copy()
    v[:, 0::2] = a * c - b * s
    v[:, 1::2] = a * s + b * c
    return v


def compress_emit(kv, norm, positions):
    """kv [W, 1024] -> normalized/roped/quantized [W, 512]."""
    out = kv * (1.0 / np.sqrt((kv * kv).mean(-1, keepdims=True) + EPS))
    out = out * norm
    rope_rows(out[:, -RD:], positions)
    q = out[:, :HD - RD]
    out[:, :HD - RD] = vb.act_quant(q.reshape(-1)).reshape(q.shape)
    return out


def prefill(wkv, wgate, ape, norm, x):
    S = x.shape[0]
    kv = x @ wkv.T                    # [S, 1024]
    score = x @ wgate.T               # [S, 1024]
    cutoff = S - S % RATIO
    rem = S % RATIO
    kv_state = np.zeros((COFF * RATIO, COFF * HD), np.float32)
    score_state = np.full((COFF * RATIO, COFF * HD), -np.inf, np.float32)
    if cutoff >= RATIO:
        kv_state[:RATIO] = kv[cutoff - RATIO:cutoff]
        score_state[:RATIO] = score[cutoff - RATIO:cutoff] + ape
    if rem > 0:
        kv_state[RATIO:RATIO + rem] = kv[cutoff:]
        score_state[RATIO:RATIO + rem] = score[cutoff:] + ape[:rem]
        kv, score = kv[:cutoff], score[:cutoff]
    kv = kv.reshape(-1, RATIO, COFF * HD)
    score = score.reshape(-1, RATIO, COFF * HD) + ape
    kv = overlap_transform(kv, 0.0)
    score = overlap_transform(score, -np.inf)
    w = stable_softmax(score, axis=1)
    out = (kv * w).sum(1)             # [W, 512]
    W = out.shape[0]
    out = compress_emit(out, norm, np.arange(0, cutoff, RATIO)[:W])
    return out, kv_state, score_state


def decode(wkv, wgate, ape, norm, x, start_pos, kv_state, score_state):
    kv = x @ wkv.T + 0.0              # [1, 1024]
    score = x @ wgate.T + ape[start_pos % RATIO]
    kv_state[RATIO + start_pos % RATIO] = kv
    score_state[RATIO + start_pos % RATIO] = score
    if (start_pos + 1) % RATIO != 0:
        return None, kv_state, score_state
    kvc = np.concatenate([kv_state[:RATIO, :HD],
                          kv_state[RATIO:, HD:]], 0)       # [8, 512]
    sc = np.concatenate([score_state[:RATIO, :HD],
                         score_state[RATIO:, HD:]], 0)
    w = stable_softmax(sc, axis=0)
    out = (kvc * w).sum(0, keepdims=True)
    kv_state[:RATIO] = kv_state[RATIO:]
    score_state[:RATIO] = score_state[RATIO:]
    out = compress_emit(out, norm, [start_pos + 1 - RATIO])
    return out, kv_state, score_state


def main() -> int:
    t = vb.tensors()
    names = ["blk.2.attn_compressor_kv.weight",
             "blk.2.attn_compressor_gate.weight",
             "blk.2.attn_compressor_ape.weight",
             "blk.2.attn_compressor_norm.weight"]
    for n in names:
        assert n in t, n
    wkv = vb.load(t, names[0])        # [1024, 7168] fp32
    wgate = vb.load(t, names[1])
    ape = vb.load(t, names[2])        # [4, 1024]
    norm = vb.load(t, names[3]).astype(np.float32)  # [512]

    rng = np.random.default_rng(12345)
    xp = rng.standard_normal((PRE, DIM)).astype(np.float32)
    xd = rng.standard_normal((DEC, DIM)).astype(np.float32)
    (ROOT / "outputs" / "comp_x_pre.bin").write_bytes(xp.tobytes())
    (ROOT / "outputs" / "comp_x_dec.bin").write_bytes(xd.tobytes())

    pre, ks, ss = prefill(wkv, wgate, ape, norm, xp)
    rows = [pre]
    for i in range(DEC):
        out, ks, ss = decode(wkv, wgate, ape, norm, xd[i:i + 1],
                             12 + i, ks, ss)
        if out is not None:
            rows.append(out)
    dec = np.concatenate(rows[1:], 0)
    (ROOT / "outputs" / "comp_cache_pre.bin").write_bytes(pre.tobytes())
    (ROOT / "outputs" / "comp_cache_dec.bin").write_bytes(dec.tobytes())
    states = np.concatenate([ks, ss], 0).astype(np.float32)
    (ROOT / "outputs" / "comp_states.bin").write_bytes(states.tobytes())
    print("pre rows", pre.shape, "sum %.6f" % pre.sum(),
          "dec rows", dec.shape, "sum %.6f" % dec.sum())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
