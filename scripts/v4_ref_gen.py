"""Full-model GENERATION oracle (Milestone K/M validation): token -> 61
cached blocks -> final head -> greedy argmax -> next token, in numpy on
REAL V4 bytes.  Mirrors runtime/nn/v4_serve.c v4_serve_generate: each
step re-embeds the current token, tiled 4x across the HC slots, at
successive positions.

Usage: v4_ref_gen.py <t0> <n_tokens>
Writes outputs/v4gen_tokens.txt (one token id per line).
"""
import sys
import time
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
NLAYERS = 61


class BlockGen:
    """One layer with persistent decode caches (window ring + compressor
    rows), weights loaded once."""

    def __init__(self, st, L):
        self.st = st
        self.L = L
        self.ratio = 4 if (L >= 2 and L % 2 == 0) else 128
        self.overlap = self.ratio == 4
        w = (1 + self.overlap) * 512
        self.ckv = vs.load_any(st, f"blk.{L}.attn_compressor_kv.weight")
        self.cg = vs.load_any(st, f"blk.{L}.attn_compressor_gate.weight")
        self.ape = vs.load_any(st, f"blk.{L}.attn_compressor_ape.weight")
        self.cnorm = vs.load_any(st, f"blk.{L}.attn_compressor_norm.weight")
        self.ks = np.zeros((2 * self.ratio, w), np.float32)
        self.ss = np.full((2 * self.ratio, w), -np.inf, np.float32)
        self.win = np.zeros((WIN, HD), np.float32)
        self.cpr = {}

    def step(self, x0, p, tok, ml):
        """x0 [4, D] -> [4, D] output at position p, routed by tok.
        Big weight matrices are loaded transiently (the dequantizers
        upcast to float64; keeping them memoized would exhaust RAM)."""
        st = self.st
        L = self.L
        ratio = self.ratio
        overlap = self.overlap
        ld = vs.load_any
        xf = x0.reshape(-1).astype(np.float64)
        mixes = ld(st, f"blk.{L}.hc_attn_fn.weight") \
            .astype(np.float64) @ xf * (1.0 / np.sqrt((xf * xf).mean()
                                                      + EPS))
        pre, post, comb = vs.hc_split(mixes, ld(
            st, f"blk.{L}.hc_attn_scale.weight"), ld(
            st, f"blk.{L}.hc_attn_base.weight"))
        y = (pre[:, None] * x0).sum(axis=0)
        y = vs.rmsnorm(y, ld(st, f"blk.{L}.attn_norm.weight"))

        qr = vs.rmsnorm(ld(st, f"blk.{L}.attn_q_a.weight") @ y,
                        ld(st, f"blk.{L}.attn_q_a_norm.weight"))
        q = (ld(st, f"blk.{L}.attn_q_b.weight") @ qr) \
            .reshape(HEADS, HD)
        q = q / np.sqrt((q * q).mean(axis=-1, keepdims=True) + EPS)
        vs.rope(q[..., -RD:], p)
        kv = vs.rmsnorm(ld(st, f"blk.{L}.attn_kv.weight") @ y,
                        ld(st, f"blk.{L}.attn_kv_a_norm.weight"))
        vs.rope(kv[-RD:], p)
        kv = np.concatenate([vs.act_quant(kv[:HD - RD]), kv[-RD:]])
        self.win[p % WIN] = kv

        kv2 = (self.ckv @ y).astype(np.float64)
        score = (self.cg @ y).astype(np.float64) + self.ape[p % ratio]
        slot = (ratio if overlap else 0) + p % ratio
        self.ks[slot] = kv2
        self.ss[slot] = score
        ncomp = (p + 1) // ratio
        if (p + 1) % ratio == 0:
            if overlap:
                kvc = np.concatenate([self.ks[:ratio, :HD],
                                      self.ks[ratio:, HD:]], axis=0)
                sc = np.concatenate([self.ss[:ratio, :HD],
                                     self.ss[ratio:, HD:]], axis=0)
                self.ks[:ratio] = self.ks[ratio:]
                self.ss[:ratio] = self.ss[ratio:]
                self.ss[ratio:] = -np.inf
            else:
                kvc = self.ks[:ratio].copy()
                sc = self.ss[:ratio].copy()
                self.ss[:ratio] = -np.inf
            wts = np.exp(sc - sc.max(axis=0, keepdims=True))
            wts = wts / wts.sum(axis=0, keepdims=True)
            row = (kvc * wts).sum(axis=0)
            row = row * (1.0 / np.sqrt((row * row).mean() + EPS)) * self.cnorm
            vs.rope(row[-RD:], p + 1 - ratio)
            row[:HD - RD] = vs.act_quant(row[:HD - RD])
            self.cpr[p // ratio] = row

        idxs = []
        pmod = p % WIN if p >= WIN - 1 else -1
        if pmod >= 0:
            idxs += list(range(pmod + 1, WIN)) + list(range(pmod + 1))
        else:
            idxs += list(range(p + 1))
        idxs += [WIN + t for t in range(ncomp)]
        gathered = np.stack(
            [self.win[i] if i < WIN else self.cpr[i - WIN] for i in idxs])
        sink = ld(st, f"blk.{L}.attn_sinks.weight")
        s = (q @ gathered.T) * SCALE              # [128, T]
        smax = s.max(axis=-1)
        s = s - smax[:, None]
        e = np.exp(s)
        e = e / (e.sum(axis=-1, keepdims=True)
                 + np.exp(sink - smax)[:, None])
        o = e @ gathered                          # [128, 512]
        vs.rope(o[..., -RD:], p, inverse=True)
        wo_a = ld(st, f"blk.{L}.attn_output_a.weight") \
            .reshape(OG, OL, -1)
        mid = np.einsum("grk,gk->gr", wo_a, o.reshape(OG, -1))
        a_out = ld(st, f"blk.{L}.attn_output_b.weight") \
            @ mid.reshape(-1)
        x1 = post[:, None] * a_out + comb @ x0

        # ---- FFN branch ----
        xf = x1.reshape(-1).astype(np.float64)
        mixes = ld(st, f"blk.{L}.hc_ffn_fn.weight") \
            .astype(np.float64) @ xf * (1.0 / np.sqrt((xf * xf).mean()
                                                      + EPS))
        pre, post, comb = vs.hc_split(mixes, ld(
            st, f"blk.{L}.hc_ffn_scale.weight"), ld(
            st, f"blk.{L}.hc_ffn_base.weight"))
        y = (pre[:, None] * x1).sum(axis=0)
        y = vs.rmsnorm(y, ld(st, f"blk.{L}.ffn_norm.weight"))
        ffn_out = ml(y[None, :], [tok])[0]
        return post[:, None] * ffn_out + comb @ x1


def head_logits(st, x):
    fn = vs.load_any(st, "output_hc_fn.weight")
    scale = vs.load_any(st, "output_hc_scale.weight")
    base = vs.load_any(st, "output_hc_base.weight")
    nw = vs.load_any(st, "output_norm.weight")
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
    logits = np.empty(vocab, np.float32)
    B = 16384
    with open(path, "rb") as f:
        for r0 in range(0, vocab, B):
            r1 = min(r0 + B, vocab)
            f.seek(off + r0 * rb)
            raw = np.frombuffer(f.read((r1 - r0) * rb), np.uint8)
            W = vs.dequantize_rows("Q6_K", raw, (DIM, r1 - r0), 0, r1 - r0)
            logits[r0:r1] = W @ hh
    return logits


def main() -> int:
    import glob
    t0 = int(sys.argv[1])
    n = int(sys.argv[2])
    shards = sorted(glob.glob("D:/hyperv4/models/pro/"
                              "deepseek-ai-DeepSeek-V4-Pro-Q3_K_M-*.gguf"))
    st = vs.ExpertStore(shards[0], extra_shards=shards[1:])

    tok = t0
    tokens = []
    blocks = [BlockGen(st, L) for L in range(NLAYERS)]
    t_start = time.time()
    for p in range(n):
        h = vs.load_any(st, "token_embd.weight", rows=[tok])[0]
        x = np.tile(h, (HC, 1)).astype(np.float32)
        tp = time.time()
        for L, bg in enumerate(blocks):
            ml = vs.MoELayer(st, L)
            x = bg.step(x, p, tok, ml)
            ml.close()
            del ml
        print(f"p={p} blocks {time.time()-tp:.1f}s sum {x.sum():.4f}",
              flush=True)
        logits = head_logits(st, x)
        tok = int(np.argmax(logits))
        tokens.append(tok)
        print(f"p={p} -> token {tok} (logits max {logits.max():.2f})",
              flush=True)
    (ROOT / "outputs" / "v4gen_tokens.txt").write_text(
        " ".join(str(t) for t in tokens) + "\n")
    print(f"done in {time.time()-t_start:.1f}s: {tokens}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
