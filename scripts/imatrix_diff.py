"""Parse the fork's binary imatrix and diff its tensor names against
a GGUF's, so requantization can exclude exactly the uncovered tensors.

Format (empirically): u32 n_entries, then per entry:
    u32 name_len, name bytes, u32 ncall, u32 nval, nval x f32
Usage:
    python scripts/imatrix_diff.py <imatrix.dat> <model.gguf>
"""

import struct
import sys

sys.path.insert(0, ".")
from ultratensor.gguf_factored import read_gguf_header  # noqa: E402


def parse_imatrix_names(path):
    with open(path, "rb") as f:
        (n,) = struct.unpack("<I", f.read(4))
        names = []
        for _ in range(n):
            klen, = struct.unpack("<I", f.read(4))
            name = f.read(klen).decode("latin1")
            ncall, nval = struct.unpack("<II", f.read(8))
            f.seek(nval * 4, 1)
            names.append(name)
    return names


def main():
    imat, gguf = sys.argv[1], sys.argv[2]
    names = parse_imatrix_names(imat)
    _, _, infos, _ = read_gguf_header(gguf)
    gguf_names = {nm.decode() for nm, *_ in infos}
    missing = sorted(n for n in gguf_names - set(names)
                     if n.endswith(".weight"))
    print(f"imatrix entries: {len(names)}  gguf tensors: {len(gguf_names)}")
    print(f"missing .weight tensors: {len(missing)}")
    for n in missing:
        print(" ", n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
