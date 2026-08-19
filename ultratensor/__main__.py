"""UltraTensor CLI.

    python -m ultratensor dry-run model.gguf [--target uq4]
    python -m ultratensor compress model.gguf --out dir [--target uq4] [--max-tensors N]
"""

import argparse
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(prog="ultratensor",
                                 description="Streaming compression for very large models")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("dry-run", help="analyze a GGUF without writing")
    p1.add_argument("model")
    p1.add_argument("--target", default="uq4", choices=["uq4", "q4_0", "q8_0", "q2_0"])
    p1.add_argument("--block", type=int, default=128)
    p1.add_argument("--max-tensors", type=int, default=None)
    p1.add_argument("--only", default=None,
                    help="comma-separated tensor names to include")

    p2 = sub.add_parser("compress", help="streaming compress a GGUF")
    p2.add_argument("model")
    p2.add_argument("--out", required=True)
    p2.add_argument("--target", default="uq4", choices=["uq4", "q4_0", "q8_0", "q2_0"])
    p2.add_argument("--block", type=int, default=128)
    p2.add_argument("--max-tensors", type=int, default=None)
    p2.add_argument("--only", default=None,
                    help="comma-separated tensor names to include")
    p2.add_argument("--manifest-name", default="ultratensor_manifest.json",
                    help="manifest file name in --out (unique per shard)")

    p3 = sub.add_parser("inspect", help="header-only tensor inventory (works on partial downloads)")
    p3.add_argument("model")

    p4 = sub.add_parser("grc", help="streaming Geodesic Runtime Compression of attention tensors")
    p4.add_argument("model")
    p4.add_argument("--out", required=True)
    p4.add_argument("--energy", type=float, default=0.98,
                    help="kept-energy fraction used to pick each rank")
    p4.add_argument("--max-rank", type=int, default=None)
    p4.add_argument("--sink", type=int, default=0,
                    help="top-T rows by norm kept dense per tensor")
    p4.add_argument("--max-tensors", type=int, default=None)
    p4.add_argument("--only", default=None,
                    help="comma-separated tensor names to include")

    p5 = sub.add_parser("export", help="requantize a GGUF to runnable Q2_K")
    p5.add_argument("model")
    p5.add_argument("--out", required=True)
    p5.add_argument("--max-tensors", type=int, default=None)
    p5.add_argument("--torch", action="store_true",
                    help="use CUDA for the Q2_K grid search (much faster)")
    p5.add_argument("--chunk", type=int, default=None,
                    help="blocks per chunk (torch path; bigger = faster, more VRAM)")
    p5.add_argument("--only", default=None,
                    help="comma-separated tensor names to include")

    p6 = sub.add_parser("export-factored-v4",
                        help="streaming V4-Pro expert factorization to a "
                             "factored GGUF shard (overnight job)")
    p6.add_argument("src")
    p6.add_argument("--out", required=True)
    p6.add_argument("--rank", type=int, default=128)
    p6.add_argument("--batch", type=int, default=4)
    p6.add_argument("--device", default="cuda")
    p6.add_argument("--only", default=None)
    p6.add_argument("--limit-experts", type=int, default=None)

    args = ap.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from ultratensor.stream import compress_gguf, dry_run, tensor_inventory

    only = None
    if getattr(args, "only", None):
        only = {s.strip() for s in args.only.split(",") if s.strip()}

    if args.cmd == "dry-run":
        dry_run(Path(args.model), target=args.target, block=args.block,
                max_tensors=args.max_tensors, only=only)
    elif args.cmd == "compress":
        compress_gguf(Path(args.model), Path(args.out), target=args.target,
                      block=args.block, max_tensors=args.max_tensors, only=only,
                      manifest_name=args.manifest_name)
    elif args.cmd == "grc":
        from ultratensor.grc import grc_compress_gguf
        grc_compress_gguf(Path(args.model), Path(args.out),
                          energy=args.energy, max_rank=args.max_rank,
                          sink_T=args.sink, only=only,
                          max_tensors=args.max_tensors)
    elif args.cmd == "export":
        from ultratensor.export_gguf import export_q2k
        export_q2k(Path(args.model), Path(args.out), only=only,
                   max_tensors=args.max_tensors, use_torch=args.torch,
                   chunk_blocks=args.chunk)
    elif args.cmd == "export-factored-v4":
        from ultratensor.export_factored_v4 import convert_shard
        convert_shard(Path(args.src), Path(args.out), rank=args.rank,
                      batch=args.batch, device=args.device, only=args.only,
                      limit_experts=args.limit_experts)
    elif args.cmd == "inspect":
        from collections import Counter
        rows = tensor_inventory(Path(args.model))
        counts = Counter(q for _, q, _, _ in rows)
        total_elems = sum(int(__import__("numpy").prod(d)) for _, _, d, _ in rows)
        print(f"{len(rows)} tensors | type distribution: "
              f"{dict(counts)}")
        print(f"total elements: {total_elems:,}")
        for name, q, dims, off in rows:
            print(f"  {name:<55s} {q:<8s} {list(dims)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
