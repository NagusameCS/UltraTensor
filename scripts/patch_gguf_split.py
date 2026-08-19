"""In-place patch of GGUF split.count metadata.

llama.cpp refuses single-file GGUFs whose header still claims
split.count > 1 ("invalid split file name"), which is exactly what
happened to the keep64 coder build (header KVs copied from shard 0
of the 17-shard original).  Setting split.count = 1 makes the loader
treat the file as a single file; all tensor data is already present.

Usage:
    python scripts/patch_gguf_split.py <model.gguf> [count=1]

Prints the old and new values; idempotent.
"""

import struct
import sys

# STANDARD GGUF KV type enum (verified against gguf-py
# gguf.constants.GGUFValueType): 0/1=1B, 2/3=2B, 4/5/6=4B, 7=bool,
# 8=STRING, 9=ARRAY, 10/11/12=8B.  (Earlier revisions of this file
# used a wrong table with string/array at 12/13; that was a local
# parser bug, NOT a deviation in the model files.)
_VT = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1,
       10: 8, 11: 8, 12: 8}


def iter_kvs(path):
    """Yield (key_bytes, gguf_type, value_offset, value_len) by walking
    the GGUF header without decoding values."""
    with open(path, "rb") as f:
        head = f.read(24)   # magic(4) + version(4) + n_tensors(8) + n_kv(8)
        if head[:4] != b"GGUF":
            raise SystemExit("not a GGUF file")
        version, n_tensors, n_kv = struct.unpack("<IQQ", head[4:24])
        for _ in range(n_kv):
            pos = f.tell()
            klen = struct.unpack("<Q", f.read(8))[0]
            if klen > 4096:
                raise SystemExit(f"bad klen {klen} at pos {pos}")
            key = f.read(klen)
            ttype = struct.unpack("<I", f.read(4))[0]
            vlen = _VT.get(ttype, 0)
            if ttype == 8:                       # string
                slen = struct.unpack("<Q", f.read(8))[0]
                vlen = 8 + slen
                f.seek(slen, 1)
            elif ttype == 9:                     # array
                atype, alen = struct.unpack("<IQ", f.read(12))
                vlen = 12
                if atype == 8:                   # array of strings
                    for _ in range(alen):
                        slen = struct.unpack("<Q", f.read(8))[0]
                        vlen += 8 + slen
                        f.seek(slen, 1)
                else:
                    esize = _VT.get(atype, 0)
                    vlen += esize * alen
                    f.seek(esize * alen, 1)
            else:
                f.seek(vlen, 1)
            yield key, ttype, pos, klen, vlen


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    path = sys.argv[1]
    new_count = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    found = None
    for key, ttype, kpos, klen, vlen in iter_kvs(path):
        if key == b"split.count":
            # value bytes live right after key_len+key+type
            voff = kpos + 8 + klen + 4
            found = (voff, vlen)
    if found is None:
        print("split.count KV not found; nothing to do")
        return 1
    voff, vlen = found
    with open(path, "r+b") as f:
        f.seek(voff)
        old = struct.unpack("<H" if vlen == 2 else "<I", f.read(vlen))[0]
        f.seek(voff)
        if vlen == 2:
            f.write(struct.pack("<H", new_count))
        elif vlen == 4:
            f.write(struct.pack("<I", new_count))
        else:
            raise SystemExit(f"unexpected split.count size {vlen}")
    print(f"split.count: {old} -> {new_count} (patched in place)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
