"""Audit MTP tensors + nextn KV across models (read-only, ground truth).

Answers: do the ORIGINAL published Pro GGUFs declare
deepseek4.nextn_predict_layers=1 while containing zero MTP tensors?

Searches tensor names CONTAINING 'mtp' anywhere (not just the prefix),
and validates the detector against a known-native-MTP file (Flash).

Usage:
    python scripts/audit_mtp.py
"""
import glob
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ultratensor.gguf_factored import read_gguf_header  # noqa: E402


def scan(path: str) -> tuple[int, list[str], int | None]:
    v, kvs, infos, h = read_gguf_header(path)
    mtp_names = [n.decode("latin1") for n, d, t, o in infos
                 if b"mtp" in n.lower()]
    nextn = None
    for k, t, r in kvs:
        if k == b"deepseek4.nextn_predict_layers":
            nextn = struct.unpack("<I", r)[0] if len(r) == 4 else r
    return len(infos), mtp_names, nextn


def main() -> int:
    print("=== ORIGINAL published Pro Q3_K_M shards (D:) ===")
    shards = sorted(glob.glob(
        r"D:/hyperv4/models/pro/deepseek-ai-DeepSeek-V4-Pro-Q3_K_M-*.gguf"))
    print(f"shards found: {len(shards)}")
    total_t, total_mtp = 0, 0
    for s in shards:
        nt, names, _ = scan(s)
        total_t += nt
        total_mtp += len(names)
        for nm in names[:3]:
            print(f"  MTP in {Path(s).name}: {nm}")
    print(f"total tensors: {total_t}; total MTP tensors: {total_mtp}")
    nt, names, nextn = scan(shards[0])
    print(f"shard 0 nextn_predict_layers = {nextn}")

    print("\n=== NAS copy (Y:) ===")
    shards_y = sorted(glob.glob(r"Y:/models/v4pro/*.gguf"))
    print(f"shards found: {len(shards_y)}")
    nt, names, nextn = scan(shards_y[0])
    print(f"shard 0 nextn_predict_layers = {nextn}; "
          f"mtp in shard 0: {len(names)}")

    print("\n=== Positive control: Flash (nativeMTP) ===")
    for cand in (r"Y:/models/flash/flash-correct.gguf",
                 r"Y:/models/flash/DeepSeek-V4-Flash-IQ2XXS-merged.gguf"):
        if Path(cand).exists():
            try:
                nt, names, nextn = scan(cand)
                print(f"{cand}: tensors={nt}, MTP={len(names)}, "
                      f"nextn={nextn}")
                for nm in names[:5]:
                    print(f"  MTP: {nm}")
            except Exception as e:  # noqa: BLE001
                print(f"{cand}: scan failed: {e}")
    parts = glob.glob(r"Y:/models/flash/*.part00")
    if parts and not any("merged" in p for p in parts):
        try:
            nt, names, nextn = scan(parts[0])
            print(f"{parts[0]}: tensors={nt}, MTP={len(names)}, nextn={nextn}")
            for nm in names[:5]:
                print(f"  MTP: {nm}")
        except Exception as e:  # noqa: BLE001
            print(f"{parts[0]}: scan failed: {e}")

    print("\n=== Our patched models (should show nextn=0, MTP=0) ===")
    for p in (r"D:/hyperv4/models/coder/DeepSeek-V4-Coder-keep16u.gguf",
              r"Y:/models/coder/DeepSeek-V4-Coder-keep64-00001-of-00001.gguf",
              r"Y:/models/coder/DeepSeek-V4-Coder-keep8u.gguf",
              r"Y:/models/coder/DeepSeek-V4-Coder-keep12u.gguf"):
        if Path(p).exists():
            nt, names, nextn = scan(p)
            print(f"{p}: nextn={nextn}, MTP={len(names)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
