"""Multi-token block oracle (Milestone J): per-layer block forward with
the official decode caches (window ring 128 + compressed rows via the
layer compressor, overlap at ratio 4) on REAL V4 bytes.

Usage: v4_ref_seq.py <layer> <n_tokens>
Writes outputs/v4seq_x_L{n}.bin (input states) and
outputs/v4seq_y_L{n}.bin (per-token block outputs).
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import v4_ref_serve as vs  # noqa: E402

DIM, HC, HEADS, HD, RD = 7168, 4, 128, 512, 64
QL, OL, OG = 1536, 1024, 16
EPS = 1e-6
SCALE = HD ** -0.5
WIN = 128
TOKEN = 4242


def block_seq(st, ml, L, xs):
    """xs [n, DIM] -> [n][4][DIM] block outputs with persistent cache."""
    ratio = 4 if (L >= 2 and L % 2 == 0) else 128
    overlap = ratio == 4
    w = (1 + overlap) * 512
    ckv = vs.load_any(st, f"blk.{L}.attn_compressor_kv.weight")   # [w,D]
    cg = vs.load_any(st, f"blk.{L}.attn_compressor_gate.weight")
    ape = vs.load_any(st, f"blk.{L}.attn_compressor_ape.weight")  # [r,w]
    cnorm = vs.load_any(st, f"blk.{L}.attn_compressor_norm.weight")
    ks = np.zeros((2 * ratio, w), np.float32)
    ss = np.full((2 * ratio, w), -np.inf, np.float32)
    win = np.zeros((WIN, HD), np.float32)
    cpr = {}
    out = np.empty((xs.shape[0], HC, DIM), np.float32)
    for p in range(xs.shape[0]):
        x0 = np.tile(xs[p], (HC, 1)).astype(np.float32)
        # ---- attention branch (with cache) ----
        xf = x0.reshape(-1).astype(np.float64)
        mixes = vs.load_any(st, f"blk.{L}.hc_attn_fn.weight") \
            .astype(np.float64) @ xf * (1.0 / np.sqrt((xf * xf).mean()
                                                      + EPS))
        pre, post, comb = vs.hc_split(mixes, vs.load_any(
            st, f"blk.{L}.hc_attn_scale.weight"), vs.load_any(
            st, f"blk.{L}.hc_attn_base.weight"))
        y = (pre[:, None] * x0).sum(axis=0)
        y = vs.rmsnorm(y, vs.load_any(st, f"blk.{L}.attn_norm.weight"))

        qr = vs.rmsnorm(vs.load_any(st, f"blk.{L}.attn_q_a.weight") @ y,
                        vs.load_any(st, f"blk.{L}.attn_q_a_norm.weight"))
        q = (vs.load_any(st, f"blk.{L}.attn_q_b.weight") @ qr) \
            .reshape(HEADS, HD)
        q = q / np.sqrt((q * q).mean(axis=-1, keepdims=True) + EPS)
        vs.rope(q[..., -RD:], p)
        kv = vs.rmsnorm(vs.load_any(st, f"blk.{L}.attn_kv.weight") @ y,
                        vs.load_any(st, f"blk.{L}.attn_kv_a_norm.weight"))
        vs.rope(kv[-RD:], p)
        kv = np.concatenate([vs.act_quant(kv[:HD - RD]), kv[-RD:]])
        win[p % WIN] = kv

        kv2 = (ckv @ y).astype(np.float64)
        score = (cg @ y).astype(np.float64) + ape[p % ratio]
        slot = (ratio if overlap else 0) + p % ratio
        ks[slot] = kv2
        ss[slot] = score
        ncomp = (p + 1) // ratio
        if (p + 1) % ratio == 0:
            if overlap:
                kvc = np.concatenate([ks[:ratio, :HD], ks[ratio:, HD:]],
                                     axis=0)
                sc = np.concatenate([ss[:ratio, :HD], ss[ratio:, HD:]],
                                    axis=0)
                ks[:ratio] = ks[ratio:]
                ss[:ratio] = ss[ratio:]
                ss[ratio:] = -np.inf
            else:
                kvc = ks[:ratio].copy()
                sc = ss[:ratio].copy()
                ss[:ratio] = -np.inf
            wt = vs.stable_softmax if hasattr(vs, "stable_softmax") \
                else None
            wts = np.exp(sc - sc.max(axis=0, keepdims=True))
            wts = wts / wts.sum(axis=0, keepdims=True)
            row = (kvc * wts).sum(axis=0)
            row = row * (1.0 / np.sqrt((row * row).mean() + EPS)) * cnorm
            vs.rope(row[-RD:], p + 1 - ratio)
            row[:HD - RD] = vs.act_quant(row[:HD - RD])
            cpr[p // ratio] = row

        idxs = []
        pmod = p % WIN if p >= WIN - 1 else -1
        if pmod >= 0:
            idxs += list(range(pmod + 1, WIN)) + list(range(pmod + 1))
        else:
            idxs += list(range(p + 1))
        idxs += [WIN + t for t in range(ncomp)]
        gathered = np.stack(
            [win[i] if i < WIN else cpr[i - WIN] for i in idxs])
        sink = vs.load_any(st, f"blk.{L}.attn_sinks.weight")
        s = (q @ gathered.T) * SCALE              # [128, T]
        smax = s.max(axis=-1)                     # [128]
        s = s - smax[:, None]
        e = np.exp(s)                             # [128, T]
        e = e / (e.sum(axis=-1, keepdims=True)
                 + np.exp(sink - smax)[:, None])
        o = e @ gathered                          # [128, 512]
        vs.rope(o[..., -RD:], p, inverse=True)
        wo_a = vs.load_any(st, f"blk.{L}.attn_output_a.weight") \
            .reshape(OG, OL, -1)
        mid = np.einsum("grk,gk->gr", wo_a, o.reshape(OG, -1))
        a_out = vs.load_any(st, f"blk.{L}.attn_output_b.weight") \
            @ mid.reshape(-1)
        x1 = post[:, None] * a_out + comb @ x0

        # ---- FFN branch ----
        xf = x1.reshape(-1).astype(np.float64)
        mixes = vs.load_any(st, f"blk.{L}.hc_ffn_fn.weight") \
            .astype(np.float64) @ xf * (1.0 / np.sqrt((xf * xf).mean()
                                                      + EPS))
        pre, post, comb = vs.hc_split(mixes, vs.load_any(
            st, f"blk.{L}.hc_ffn_scale.weight"), vs.load_any(
            st, f"blk.{L}.hc_ffn_base.weight"))
        y = (pre[:, None] * x1).sum(axis=0)
        y = vs.rmsnorm(y, vs.load_any(st, f"blk.{L}.ffn_norm.weight"))
        ffn_out = ml(y[None, :], [TOKEN])[0]
        out[p] = post[:, None] * ffn_out + comb @ x1
        print(f"p={p} sum {out[p].sum():.4f}", flush=True)
    return out


def main() -> int:
    import glob
    L = int(sys.argv[1])
    n = int(sys.argv[2])
    shards = sorted(glob.glob("D:/hyperv4/models/pro/"
                              "deepseek-ai-DeepSeek-V4-Pro-Q3_K_M-*.gguf"))
    st = vs.ExpertStore(shards[0], extra_shards=shards[1:])
    ml = vs.MoELayer(st, L)
    rng = np.random.default_rng(1000 + L)
    xs = rng.standard_normal((n, DIM)).astype(np.float32)
    xt = np.repeat(xs, HC, axis=0).astype(np.float32)   # 4 copies/token
    (ROOT / "outputs" / f"v4seq_x_L{L}.bin").write_bytes(xt.tobytes())
    y = block_seq(st, ml, L, xs)
    (ROOT / "outputs" / f"v4seq_y_L{L}.bin").write_bytes(y.tobytes())
    print("done L%d sum %.4f" % (L, y.sum()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
