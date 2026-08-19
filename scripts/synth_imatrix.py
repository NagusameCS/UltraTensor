"""Synthesize imatrix entries for tensors a fork imatrix skips.

The fork's imatrix only records big weight matrices; IQ2_XS requant
bails on ANY targeted tensor without an entry (norms, hc_* scalars,
etc.).  This merges synthesized uniform-importance entries for those
tensors so quantization can proceed.  Small tensors => uniform
importance is quality-neutral.

Format (fork, binary): u32 count, then per entry:
    u32 name_len, name bytes, u32 ncall, u32 nval, nval x f32

Usage:
    python scripts/synth_imatrix.py <imatrix.dat> <model.gguf> <out.dat>
"""

import struct
import sys

sys.path.insert(0, ".")
from ultratensor.gguf_factored import read_gguf_header  # noqa: E402


def parse_entries(path):
    with open(path, "rb") as f:
        (n,) = struct.unpack("<I", f.read(4))
        entries = []
        for _ in range(n):
            klen, = struct.unpack("<I", f.read(4))
            name = f.read(klen)
            ncall, nval = struct.unpack("<II", f.read(8))
            vals = f.read(nval * 4)
            entries.append((name, ncall, nval, vals))
    return entries


def main():
    imat, model, out = sys.argv[1], sys.argv[2], sys.argv[3]
    entries = parse_entries(imat)
    _, _, infos, _ = read_gguf_header(model)

    def want(name, dims):
        # fork rule: nval == ne0 for 2-D, ne0*ne2 for 3-D stacks
        return int(dims[0]) * (int(dims[2]) if len(dims) == 3 else 1)

    ne0_of = {name: want(name, dims) for name, dims, ttype, off in infos}
    fixed, synth = [], []
    for name, ncall, nval, vals in entries:
        if name in ne0_of and nval != ne0_of[name]:
            need = ne0_of[name]
            if nval >= need:
                vals = vals[:need * 4]
            else:
                vals = vals + struct.pack("<%df" % (need - nval),
                                          *([1.0] * (need - nval)))
            nval = need
        fixed.append((name, ncall, nval, vals))
    have = {e[0] for e in fixed}
    for name, dims, ttype, off in infos:
        if name in have or ttype == 26:      # i32 tables never quantized
            continue
        if not name.decode().endswith(".weight"):
            continue
        nval = ne0_of[name]
        vals = struct.pack("<%df" % nval, *([1.0] * nval))
        synth.append((name, 2, nval, vals))
    total = len(fixed) + len(synth)
    with open(out, "wb") as f:
        f.write(struct.pack("<I", total))
        for name, ncall, nval, vals in fixed + synth:
            f.write(struct.pack("<I", len(name)) + name +
                    struct.pack("<II", ncall, nval) + vals)
    print(f"merged {len(fixed)} normalized + {len(synth)} synthesized "
          f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
