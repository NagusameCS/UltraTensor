"""Full-model serve oracle (Milestone I): embed -> 61 blocks -> final head
-> logits in numpy on REAL V4 bytes, single token at start_pos 0.

Writes: outputs/v4serve_logits.bin; prints per-layer progress.
"""
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ultratensor.dequant import dequantize_rows  # noqa: E402
from ultratensor.expert_store import ExpertStore  # noqa: E402
from ultratensor.moe_exec import MoELayer  # noqa: E402

DIM, HC, HEADS, HEAD_DIM, ROPE_DIM = 7168, 4, 128, 512, 64
QL, OL, O_GROUPS = 1536, 1024, 16
EPS = 1e-6
SCALE = HEAD_DIM ** -0.5
TOKEN = 4242
NLAYERS = 61

_QT = {8: ("Q8_0", 32, 34), 10: ("Q2_K", 256, 84),
       11: ("Q3_K", 256, 110), 12: ("Q4_K", 256, 144),
       13: ("Q5_K", 256, 176), 14: ("Q6_K", 256, 210)}


_BASES = {}


def _data_base(st, sidx):
    if sidx not in _BASES:
        from ultratensor.gguf_factored import _align, read_gguf_header
        v, kvs, infos, hdr = read_gguf_header(str(st.shards[sidx]))
        _BASES[sidx] = _align(hdr, 32)
    return _BASES[sidx]


def load_any(st, name, rows=None):
    t = st.tensors.get(name)
    if t is None:                      # non-block tensor: raw header
        from ultratensor.gguf_factored import _align, read_gguf_header
        path = str(st.shards[0])
        v, kvs, infos, hdr = read_gguf_header(path)
        ds = _align(hdr, 32)
        for nm, dims, tt, off in infos:
            if nm.decode() == name:
                t = {"dims": dims, "ttype": tt, "off": ds + off,
                     "shard": 0, "based": True}
                break
        else:
            raise KeyError(name)
    dims, tt, off = t["dims"], t["ttype"], t["off"]
    if not t.get("based"):
        off += _data_base(st, t["shard"])
    shard = str(st.shards[t["shard"]])
    if tt == 0:
        with open(shard, "rb") as f:
            f.seek(off)
            return np.frombuffer(f.read(int(np.prod(dims)) * 4),
                                 np.float32).copy()
    qname, elem, bb = _QT[tt]
    d0, d1 = int(dims[0]), int(dims[1])
    rb = (d0 // elem) * bb
    rsel = rows if rows is not None else range(d1)
    out = np.empty((len(rsel), d0), np.float32)
    with open(shard, "rb") as f:
        for i, r in enumerate(rsel):
            f.seek(off + r * rb)
            raw = np.frombuffer(f.read(rb), np.uint8)
            out[i] = dequantize_rows(qname, raw, (d0, 1), 0, 1)[0]
    return out


def rope_freqs():
    dim = ROPE_DIM
    base, factor, bf, bs, orig = 160000.0, 16.0, 32.0, 1.0, 65536.0
    freqs = 1.0 / (base ** (np.arange(0, dim, 2, dtype=np.float64) / dim))
    low = max(0.0, np.floor(dim * np.log(orig / (bf * 2 * np.pi))
                            / (2 * np.log(base))))
    high = min(dim - 1.0, np.ceil(dim * np.log(orig / (bs * 2 * np.pi))
                                  / (2 * np.log(base))))
    ramp = np.clip((np.arange(dim // 2) - low) / (high - low), 0, 1)
    return (freqs / factor * ramp + freqs * (1 - ramp)).astype(np.float64)


FREQS = rope_freqs()


def rope(v, pos, inverse=False):
    ang = pos * FREQS
    c, s = np.cos(ang), np.sin(ang)
    if inverse:
        s = -s
    a, b = v[..., 0::2].copy(), v[..., 1::2].copy()
    v[..., 0::2] = a * c - b * s
    v[..., 1::2] = a * s + b * c
    return v


def act_quant(x, block=64):
    y = x.copy().reshape(-1)
    for b0 in range(0, y.size, block):
        v = y[b0:b0 + block]
        amax = max(float(np.abs(v).max()), 1e-4)
        s = amax / 448.0
        q = np.clip(v / s, -448.0, 448.0)
        a = np.abs(q)
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
    return y.reshape(x.shape)


def hc_split(mixes, scale, base):
    pre = 1.0 / (1.0 + np.exp(-(mixes[:HC] * scale[0] + base[:HC]))) + EPS
    post = 2.0 / (1.0 + np.exp(-(mixes[HC:2 * HC] * scale[1]
                                 + base[HC:2 * HC])))
    comb = mixes[2 * HC:].reshape(HC, HC) * scale[2] \
        + base[2 * HC:].reshape(HC, HC)
    comb = np.exp(comb - comb.max(axis=1, keepdims=True))
    comb = comb / comb.sum(axis=1, keepdims=True) + EPS
    comb = comb / (comb.sum(axis=0) + EPS)
    for _ in range(19):
        comb = comb / (comb.sum(axis=1, keepdims=True) + EPS)
        comb = comb / (comb.sum(axis=0) + EPS)
    return pre, post, comb


def rmsnorm(x, w):
    return x * w / np.sqrt((x * x).mean() + EPS)


def block_forward(st, ml, L, x, pos):
    """x [4, D] -> [4, D] (official Block.forward)."""
    def ld(kind):
        return load_any(st, f"blk.{L}.{kind}.weight")

    xf = x.reshape(-1).astype(np.float64)
    mixes = ld("hc_attn_fn").astype(np.float64) @ xf \
        * (1.0 / np.sqrt((xf * xf).mean() + EPS))
    pre, post, comb = hc_split(mixes, ld("hc_attn_scale"),
                               ld("hc_attn_base"))
    y = (pre[:, None] * x).sum(axis=0)
    y = rmsnorm(y, ld("attn_norm"))

    qr = rmsnorm(ld("attn_q_a") @ y, ld("attn_q_a_norm"))
    q = (ld("attn_q_b") @ qr).reshape(HEADS, HEAD_DIM)
    q = q / np.sqrt((q * q).mean(axis=-1, keepdims=True) + EPS)
    rope(q[..., -ROPE_DIM:], pos)
    kv = rmsnorm(ld("attn_kv") @ y, ld("attn_kv_a_norm"))
    rope(kv[-ROPE_DIM:], pos)
    kv = np.concatenate([act_quant(kv[:HEAD_DIM - ROPE_DIM]),
                         kv[-ROPE_DIM:]])
    sink = ld("attn_sinks")
    s = (q @ kv) * SCALE
    p = 1.0 / (1.0 + np.exp(sink - s))
    o = p[:, None] * kv
    rope(o[..., -ROPE_DIM:], pos, inverse=True)
    wo_a = ld("attn_output_a").reshape(O_GROUPS, OL, -1)
    mid = np.einsum("grk,gk->gr", wo_a, o.reshape(O_GROUPS, -1))
    a_out = ld("attn_output_b") @ mid.reshape(-1)
    x = post[:, None] * a_out + comb @ x

    xf = x.reshape(-1).astype(np.float64)
    mixes = ld("hc_ffn_fn").astype(np.float64) @ xf \
        * (1.0 / np.sqrt((xf * xf).mean() + EPS))
    pre, post, comb = hc_split(mixes, ld("hc_ffn_scale"),
                               ld("hc_ffn_base"))
    y = (pre[:, None] * x).sum(axis=0)
    y = rmsnorm(y, ld("ffn_norm"))
    ffn_out = ml(y[None, :], [TOKEN])[0]
    x = post[:, None] * ffn_out + comb @ x
    return x


def main() -> int:
    import glob
    shards = sorted(glob.glob("D:/hyperv4/models/pro/"
                              "deepseek-ai-DeepSeek-V4-Pro-Q3_K_M-*.gguf"))
    st = ExpertStore(shards[0], extra_shards=shards[1:])
    h = load_any(st, "token_embd.weight", rows=[TOKEN])[0]
    x = np.tile(h, (HC, 1)).astype(np.float32)
    t0 = time.time()
    for L in range(NLAYERS):
        ml = MoELayer(st, L)
        tl = time.time()
        x = block_forward(st, ml, L, x, 0)
        print(f"layer {L}: {time.time()-tl:.2f}s | state sum "
              f"{x.sum():.4f}", flush=True)
        del ml
    print(f"blocks done in {time.time()-t0:.1f}s")

    fn = load_any(st, "output_hc_fn.weight")            # [4, 28672]
    scale = load_any(st, "output_hc_scale.weight")
    base = load_any(st, "output_hc_base.weight")
    nw = load_any(st, "output_norm.weight")
    xf = x.reshape(-1).astype(np.float64)
    mixes = fn.astype(np.float64) @ xf \
        * (1.0 / np.sqrt((xf * xf).mean() + EPS))
    pre = 1.0 / (1.0 + np.exp(-(mixes * scale + base))) + EPS
    hh = (pre[:, None] * x).sum(axis=0).astype(np.float64)
    hh = hh * nw / np.sqrt((hh ** 2).mean() + EPS)
    hh = hh.astype(np.float32)

    from ultratensor.gguf_factored import _align, read_gguf_header
    path = str(st.shards[0])
    v, kvs, infos, hdr = read_gguf_header(path)
    ds = _align(hdr, 32)
    for nm, dims, tt, off in infos:
        if nm.decode() == "output.weight":
            vocab = int(dims[1])
            off = ds + off
            break
    else:
        raise KeyError("output.weight")
    rb = (DIM // 256) * 210
    shard = path
    logits = np.empty(vocab, np.float32)
    B = 16384
    with open(shard, "rb") as f:
        for r0 in range(0, vocab, B):
            r1 = min(r0 + B, vocab)
            f.seek(off + r0 * rb)
            raw = np.frombuffer(f.read((r1 - r0) * rb), np.uint8)
            W = dequantize_rows("Q6_K", raw, (DIM, r1 - r0), 0, r1 - r0)
            logits[r0:r1] = W @ hh
    (ROOT / "outputs" / "v4serve_logits.bin").write_bytes(logits.tobytes())
    print("logits sum %.4f std %.4f" % (logits.sum(), logits.std()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
