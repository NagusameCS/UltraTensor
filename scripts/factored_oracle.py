"""Factored-matmul oracle (Phase 5): real V4 expert bytes -> factored
bins for the llama.cpp fork test (ggml/tests/test-factored).

Writes outputs/factored/: factoredC.bin (row = [scales nb*4][codes nb*16]),
U.bin (fp16 [m,k]), x.bin, y_ref.bin (C@x), z_ref.bin (U@y). Also prints
the dense-reconstruction error for the benchmark note.
"""
import glob
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultratensor.expert_store import ExpertStore
from ultratensor.gguf_factored import (factor_matrix, encode_codes,
                                       decode_codes, UQ4_BLOCK)


def main() -> int:
    out = ROOT / "outputs" / "factored"
    out.mkdir(parents=True, exist_ok=True)
    shards = sorted(glob.glob("D:/hyperv4/models/pro/"
                              "deepseek-ai-DeepSeek-V4-Pro-Q3_K_M-*.gguf"))
    st = ExpertStore(shards[0], extra_shards=shards[1:])
    W = st.read_expert(0, "ffn_gate_exps", 0)          # [m, n] real bytes
    m, n = W.shape
    assert n % UQ4_BLOCK == 0, f"n={n} not divisible by {UQ4_BLOCK}"
    k = 128
    U, C = factor_matrix(W, rank=k)                    # U [m,k] fp16, C [k,n]
    scales, packed = encode_codes(C)                   # [k,n/32], [k,n/2]

    # C tensor bytes: per row [scales][codes] (matches gguf_factored writer)
    crows = b"".join(
        scales[i].astype(np.float32).tobytes() + packed[i].tobytes()
        for i in range(k))
    (out / "factoredC.bin").write_bytes(crows)
    (out / "U.bin").write_bytes(U.astype(np.float16).tobytes())

    rng = np.random.default_rng(1234)
    x = rng.standard_normal(n).astype(np.float32)
    Cq = decode_codes(scales, packed)                 # quantized reconstruction
    y_ref = Cq.astype(np.float32) @ x                 # [k] exact kernel target
    # ggml's F16 mul_mat converts activations to fp16; emulate that here
    y_ref16 = y_ref.astype(np.float16).astype(np.float32)
    z_ref = U.astype(np.float32) @ y_ref16            # [m]
    (out / "x.bin").write_bytes(x.tobytes())
    (out / "y_ref.bin").write_bytes(y_ref.astype(np.float32).tobytes())
    (out / "z_ref.bin").write_bytes(z_ref.astype(np.float32).tobytes())

    dense = W @ x
    rel = float(np.abs(dense - z_ref).max() / np.abs(dense).max())
    print(f"expert blk.0.ffn_gate_exps e=0: m={m} n={n} k={k}")
    print(f"dense-vs-factored max_rel {rel:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
