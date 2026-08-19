"""In-place GGUF KV value patch (generic).

Usage:
    python scripts/patch_gguf_kv.py <model.gguf> <key> <new_value>

Patches the raw little-endian value of a KV in place, preserving its
type width (u16/u32 detected by value length).  Works with this fork's
non-standard KV enum via patch_gguf_split.iter_kvs.

Example (disable broken MTP in V4-Pro GGUFs):
    python scripts/patch_gguf_kv.py keep16u.gguf deepseek4.nextn_predict_layers 0
"""

import struct
import sys

from patch_gguf_split import iter_kvs


def main() -> int:
    path, key, val = sys.argv[1], sys.argv[2].encode(), int(sys.argv[3])
    for k, ttype, kpos, klen, vlen in iter_kvs(path):
        if k == key:
            voff = kpos + 8 + klen + 4
            if vlen not in (2, 4):
                raise SystemExit(f"unexpected value width {vlen}")
            with open(path, "r+b") as f:
                f.seek(voff)
                old = struct.unpack("<H" if vlen == 2 else "<I",
                                    f.read(vlen))[0]
                f.seek(voff)
                f.write(struct.pack("<H" if vlen == 2 else "<I", val))
            print(f"{key.decode()}: {old} -> {val}")
            return 0
    print(f"KV {key.decode()} not found")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
