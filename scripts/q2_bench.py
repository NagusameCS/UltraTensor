"""q2_0 expert benchmark oracle (Phase 6): real V4-Pro expert bytes.

- dequantizes one expert (blk.0.ffn_gate_exps e=0) from Q3_K,
- quantizes it to UltraTensor q2_0 (block 32, fp16 scale, amax/3),
- times Q3_K vs q2_0 dequant and reports error,
- writes outputs/q2bench/*.bin for the geodessical C harness
  (test_v4_q2.c): scales fp16 [rows, cols/32], codes u8 [rows, cols/4],
  x f32 [cols], ref f32 [rows] = Wq @ x.
"""
import glob
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultratensor.expert_store import ExpertStore
from ultratensor.quant import q2_0_quantize, q2_0_dequantize


def main() -> int:
    out = ROOT / "outputs" / "q2bench"
    out.mkdir(parents=True, exist_ok=True)
    shards = sorted(glob.glob("D:/hyperv4/models/pro/"
                              "deepseek-ai-DeepSeek-V4-Pro-Q3_K_M-*.gguf"))
    st = ExpertStore(shards[0], extra_shards=shards[1:])

    t0 = time.time()
    W = st.read_expert(0, "ffn_gate_exps", 0)          # fp32 [m, n]
    q3_t = time.time() - t0
    m, n = W.shape

    # time the raw Q3_K dequant (re-read bytes, no cache)
    from ultratensor.dequant import dequantize_rows
    ebytes = st._expert_bytes(0, "ffn_gate_exps")
    f, off = st._open_tensor("blk.0.ffn_gate_exps.weight")
    with f:
        f.seek(off)
        raw = np.frombuffer(f.read(ebytes), np.uint8)
    t1 = time.time()
    _ = dequantize_rows("Q3_K", raw, (n, m), 0, m)
    q3k_raw_t = time.time() - t1

    t1 = time.time()
    scales, codes = q2_0_quantize(W, block=32)
    q2q_t = time.time() - t1
    t1 = time.time()
    Wq = q2_0_dequantize(scales, codes)
    q2d_t = time.time() - t1

    err = np.abs(Wq - W)
    print(f"expert {m}x{n}: Q3_K dequant {q3_t:.2f}s (first, warm) + "
          f"{q3k_raw_t:.2f}s (raw) | q2_0 quant {q2q_t:.2f}s "
          f"dequant {q2d_t:.2f}s")
    print(f"q2_0 error: max_abs {err.max():.5f} mean_abs {err.mean():.6f} "
          f"max_rel {float(err.max()/np.abs(W).max()):.3e}")

    rng = np.random.default_rng(4242)
    x = rng.standard_normal(n).astype(np.float32)
    ref = Wq @ x
    (out / "scales.bin").write_bytes(scales.astype(np.float16).tobytes())
    (out / "codes.bin").write_bytes(codes.tobytes())
    (out / "x.bin").write_bytes(x.tobytes())
    (out / "ref.bin").write_bytes(ref.astype(np.float32).tobytes())
    print(f"wrote bins rows={m} cols={n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
