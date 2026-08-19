"""Numpy reference for the FULL V4 block forward (layer 0, one token),
implementing the official inference/model.py + kernel.py semantics:
hc_pre/hc_post + split-Sinkhorn, RMSNorm, MLA sliding-window attention
with sink, fp8-e4m3 act-quant, YaRN rope at an arbitrary position, and
the MoE FFN.

Writes y_ref.bin [4*7168]; used to validate the geodessical C block.
Usage: v4_ref_block.py [start_pos]
"""
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from ultratensor.dequant import dequantize_rows  # noqa: E402
from ultratensor.expert_store import ExpertStore  # noqa: E402
from ultratensor.gguf_factored import _align, read_gguf_header  # noqa: E402
from ultratensor.moe_exec import MoELayer, _swiglu  # noqa: E402

DIM = 7168
HEADS, HEAD_DIM, ROPE_DIM = 128, 512, 64
Q_LORA, O_LORA, O_GROUPS = 1536, 1024, 16
EPS = 1e-6
SCALE = HEAD_DIM ** -0.5
HC = 4
MIX = (2 + HC) * HC

SHARD0 = "D:/hyperv4/models/pro/deepseek-ai-DeepSeek-V4-Pro-Q3_K_M-00001-of-00017.gguf"
GLOB = "D:/hyperv4/models/pro/deepseek-ai-DeepSeek-V4-Pro-Q3_K_M-*-of-00017.gguf"


def tensors():
    v, kvs, infos, hdr = read_gguf_header(SHARD0)
    ds = _align(hdr, 32)
    return {(nm.decode()): (dims, tt, ds + off) for nm, dims, tt, off in infos}


def get(t, name):
    dims, tt, off = t[name]
    return dims, tt, off


def dequant_chunked(path, name, dims, tt, off, qname, elem, bbytes, B=8192):
    """n rows dequantized in batches -> [n, dim0]."""
    n = int(dims[1])
    d0 = int(dims[0])
    rb = (d0 // elem) * bbytes
    out = np.empty((n, d0), np.float32)
    with open(path, "rb") as f:
        for r0 in range(0, n, B):
            r1 = min(r0 + B, n)
            f.seek(off + r0 * rb)
            raw = np.frombuffer(f.read((r1 - r0) * rb), np.uint8)
            out[r0:r1] = dequantize_rows(qname, raw, (d0, r1 - r0), 0,
                                         r1 - r0)
    return out


_Q = {8: ("Q8_0", 32, 34), 10: ("Q2_K", 256, 84),
      11: ("Q3_K", 256, 110), 12: ("Q4_K", 256, 144),
      13: ("Q5_K", 256, 176), 14: ("Q6_K", 256, 210)}


def load(t, name):
    dims, tt, off = get(t, name)
    if tt == 0:
        with open(SHARD0, "rb") as f:
            f.seek(off)
            return np.frombuffer(f.read(int(np.prod(dims)) * 4),
                                 np.float32).copy()
    qname, elem, bbytes = _Q[tt]
    return dequant_chunked(SHARD0, name, dims, tt, off, qname, elem, bbytes)


def rmsnorm(x, w, eps=EPS):
    return x * w / np.sqrt((x * x).mean() + eps)


def act_quant(x, block=64):
    """Block-wise fp8-e4m3 fake quant (kernel.py act_quant, inplace)."""
    y = x.copy()
    for b0 in range(0, x.size, block):
        v = x[b0:b0 + block]
        amax = max(float(np.abs(v).max()), 1e-4)
        s = amax / 448.0
        q = np.clip(v / s, -448.0, 448.0)
        q = np.asarray(q, dtype=np.float64)
        a = np.abs(q)
        with np.errstate(over="ignore", under="ignore"):
            m, e = np.frexp(a)
        mq = np.rint(m * 16.0) / 16.0
        carry = mq >= 1.0
        mq = np.where(carry, 0.5, mq)
        e = e + carry.astype(np.int32)
        r = np.ldexp(mq, e)
        r = np.where(a < 2 ** -9, np.rint(a * 512.0) / 512.0, r)
        r = np.where(a == 0, 0.0, r)
        r = np.where(r > 448.0, 448.0, r)
        y[b0:b0 + block] = np.copysign(r, v) * s
    return y


def rope_freqs(dim, base, factor, beta_fast, beta_slow, original):
    """YaRN freqs per model.py precompute_freqs_cis (dim/2 entries)."""

    def corr_dim(num_rot, base, max_seq_len):
        return dim * math.log(max_seq_len / (num_rot * 2 * math.pi)) \
            / (2 * math.log(base))

    freqs = 1.0 / (base ** (np.arange(0, dim, 2, dtype=np.float64) / dim))
    low = max(0.0, math.floor(corr_dim(beta_fast, base, original)))
    high = min(dim - 1.0, math.ceil(corr_dim(beta_slow, base, original)))
    if low == high:
        high += 0.001
    ramp = np.clip((np.arange(dim // 2, dtype=np.float64) - low)
                   / (high - low), 0, 1)
    smooth = 1 - ramp
    freqs = freqs / factor * (1 - smooth) + freqs * smooth
    return freqs.astype(np.float64)


def rope_apply(v, pos, inverse=False):
    """v: [..., ROPE_DIM]; rotate the last ROPE_DIM as complex pairs."""
    freqs = rope_freqs(ROPE_DIM, 160000.0, 16.0, 32.0, 1.0, 65536.0)
    ang = pos * freqs
    c, s = np.cos(ang), np.sin(ang)
    if inverse:
        s = -s
    a = v[..., 0::2].copy()
    b = v[..., 1::2].copy()
    v[..., 0::2] = a * c - b * s
    v[..., 1::2] = a * s + b * c
    return v


def hc_split(mixes, scale, base):
    pre = 1.0 / (1.0 + np.exp(-(mixes[:HC] * scale[0] + base[:HC]))) + EPS
    post = 2.0 / (1.0 + np.exp(-(mixes[HC:2 * HC] * scale[1]
                                 + base[HC:2 * HC])))
    comb = (mixes[2 * HC:].reshape(HC, HC) * scale[2]
            + base[2 * HC:].reshape(HC, HC))
    comb = np.exp(comb - comb.max(axis=1, keepdims=True))
    comb = comb / comb.sum(axis=1, keepdims=True) + EPS
    comb = comb / (comb.sum(axis=0) + EPS)
    for _ in range(19):
        comb = comb / (comb.sum(axis=1, keepdims=True) + EPS)
        comb = comb / (comb.sum(axis=0) + EPS)
    return pre, post, comb


def main() -> int:
    start_pos = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    import glob
    st = ExpertStore(glob.glob(GLOB)[0],
                     extra_shards=glob.glob(GLOB)[1:])
    ml = MoELayer(st, 0)
    t = tensors()
    rng = np.random.default_rng(0)
    x0 = rng.standard_normal(DIM).astype(np.float32)     # block input [4,D]
    x = np.tile(x0, (HC, 1))                              # hc copies

    hc_fn_a = load(t, "blk.0.hc_attn_fn.weight")               # (24, 28672)
    hc_sc_a = load(t, "blk.0.hc_attn_scale.weight")
    hc_ba_a = load(t, "blk.0.hc_attn_base.weight")
    hc_fn_f = load(t, "blk.0.hc_ffn_fn.weight")
    hc_sc_f = load(t, "blk.0.hc_ffn_scale.weight")
    hc_ba_f = load(t, "blk.0.hc_ffn_base.weight")
    attn_norm = load(t, "blk.0.attn_norm.weight")
    ffn_norm = load(t, "blk.0.ffn_norm.weight")

    # ---- attention branch ----
    xf = x.reshape(-1)
    mixes = hc_fn_a @ xf * (1.0 / np.sqrt((xf * xf).mean() + EPS))
    pre, post, comb = hc_split(mixes, hc_sc_a, hc_ba_a)
    y = (pre[:, None] * x).sum(axis=0)
    y = rmsnorm(y, attn_norm)

    qr = rmsnorm(load(t, "blk.0.attn_q_a.weight") @ y,
                 load(t, "blk.0.attn_q_a_norm.weight"))
    q = (load(t, "blk.0.attn_q_b.weight") @ qr).reshape(HEADS, HEAD_DIM)
    q = q / np.sqrt((q * q).mean(axis=-1, keepdims=True) + EPS)
    rope_apply(q[..., -ROPE_DIM:], start_pos)
    kv = rmsnorm(load(t, "blk.0.attn_kv.weight") @ y,
                 load(t, "blk.0.attn_kv_a_norm.weight"))
    rope_apply(kv[-ROPE_DIM:], start_pos)
    kv = np.concatenate([act_quant(kv[:HEAD_DIM - ROPE_DIM]), kv[-ROPE_DIM:]])

    sink = load(t, "blk.0.attn_sinks.weight")               # [128]
    s = (q @ kv) * SCALE                                 # [128]
    # single position: per-head max is the score itself
    p = 1.0 / (1.0 + np.exp(sink - s))
    o = p[:, None] * kv                                  # [128, 512]
    rope_apply(o[..., -ROPE_DIM:], start_pos, inverse=True)
    print("ref q sum %.6f" % q.sum())
    print("ref kv sum %.6f" % kv.sum())
    print("ref o sum %.6f" % o.sum())
    o_g = o.reshape(O_GROUPS, -1)                        # [16, 4096]
    wo_a = load(t, "blk.0.attn_output_a.weight")          # (16384, 4096)
    wo_a = wo_a.reshape(O_GROUPS, O_LORA, -1)            # (16,1024,4096)
    mid = np.einsum("grk,gk->gr", wo_a, o_g)             # [16,1024]
    a_out = load(t, "blk.0.attn_output_b.weight") @ mid.reshape(-1)
    print("ref attn out sum %.6f" % a_out.sum())
    x_new = post[:, None] * a_out + comb @ x
    print("ref attn post sum %.6f" % x_new.sum())
    x = x_new

    # ---- FFN branch ----
    xf = x.reshape(-1)
    mixes = hc_fn_f @ xf * (1.0 / np.sqrt((xf * xf).mean() + EPS))
    pre, post, comb = hc_split(mixes, hc_sc_f, hc_ba_f)
    y = (pre[:, None] * x).sum(axis=0)
    print("ref ffn pre sum %.6f" % y.sum())
    y = rmsnorm(y, ffn_norm)
    ffn_out = ml(y[None, :], [4242])[0]
    print("ref ffn out sum %.6f" % ffn_out.sum())
    x_out = post[:, None] * ffn_out + comb @ x
    print("ref final sum %.6f" % x_out.sum())

    np.asarray(x_out, np.float32).tofile(
        ROOT / "outputs" / f"v4_block_ref_p{start_pos}.bin")
    print("y_ref(p=%d) sum %.6f std %.6f"
          % (start_pos, x_out.sum(), x_out.std()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
