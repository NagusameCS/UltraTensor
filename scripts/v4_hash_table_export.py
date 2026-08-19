"""V4-Coder deployment artifact — hash-layer token->expert lookup tables.

Hash layers (0-2) route by a DETERMINISTIC token->top-6 table
(ffn_gate_tid2eid, zero GEMV, zero per-token IO — G3's finding). For
the coder serving path this table IS the prefetch predictor: given the
next token id, the experts to pre-read are known exactly.

Exports the real tables to outputs/hash_route_tables.npz (compact,
int32) plus a JSON manifest, and validates: table shape, id bounds,
coverage of the code-battery token ids.

Usage:
    python scripts/v4_hash_table_export.py
"""

import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import v4_ref_serve as vs  # noqa: E402

OUT = ROOT / "outputs" / "hash_route_tables.npz"


def main() -> int:
    shards = sorted(glob.glob("D:/hyperv4/models/pro/"
                              "deepseek-ai-DeepSeek-V4-Pro-Q3_K_M-*.gguf"))
    st = vs.ExpertStore(shards[0], extra_shards=shards[1:])

    manifest = {"n_hash_layers": st.n_hash_layers, "layers": {}}
    tables = {}
    for layer in range(st.n_hash_layers):
        t = st.read_tensor(layer, "ffn_gate_tid2eid")
        assert t.ndim == 2 and t.shape[1] == 6, t.shape
        assert int(t.min()) >= 0 and int(t.max()) < 384
        tables[f"L{layer}"] = t.astype(np.int32)
        manifest["layers"][f"L{layer}"] = {
            "shape": list(t.shape),
            "dtype": "int32",
            "bytes": int(t.nbytes),
        }
        print(f"L{layer}: table {t.shape}, ids [{t.min()},{t.max()}]")

    # coverage of the code battery
    code = json.load(open(ROOT / "outputs" / "code_census_prompts.json",
                          encoding="utf-8"))
    tok_ids = [int(i) for i in code["token_ids"] if 0 <= i < tables["L0"].shape[0]]
    manifest["code_battery_covered_tokens"] = len(tok_ids)
    manifest["code_battery_n_tokens"] = len(code["token_ids"])

    np.savez_compressed(OUT, **tables)
    (ROOT / "outputs" / "hash_route_tables.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
