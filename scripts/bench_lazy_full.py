"""Full-model lazy-serving projection for V4-Pro on this box.

Combines three measured facts:
  1. per-layer lazy FFN timing (outputs/bench_moe_sweep.json, or the
     layer-0/3 measurements as fallback),
  2. the complete per-layer byte budget from ExpertStore.io_model
     (routed experts + shared expert + attention/indexer/compressor/
     norm reads),
  3. the measured disk bandwidth (2 GB/s on D:).

and projects tok/s under four serving strategies:
  serial    : reads and decode serialized per layer (pessimistic)
  pipelined : attention/dense reads of layer L+1 overlap the FFN decode
              of layer L (attention tensors are data-independent and
              prefetchable); floor = max(total reads, total decode)
  resident  : shared-expert + dense (attention/indexer/compressor/norm)
              tensors pinned in RAM (~10.1 GiB total) - only the routed
              expert slices are read per token
  batch     : resident cache + concurrent requests batched so that
              distinct routed experts are read ONCE per batch (analytic
              dedup: E distinct experts drawn with top-k of E per token)
  aggregate : total-model throughput ceiling at the disk bandwidth
              (multi-request serving shares the disk)

Usage:
    python scripts/bench_lazy_full.py <model_glob> [--bandwidth GBs]
"""
import argparse
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from ultratensor.expert_store import ExpertStore  # noqa: E402

SWEEP = ROOT / "outputs" / "bench_moe_sweep.json"
FALLBACK = {"hash": 0.150, "dense": 0.226}   # s/token-layer (measured)


def load_ffn_times() -> dict[int, float]:
    if SWEEP.exists():
        d = json.loads(SWEEP.read_text())
        per = d.get("per_layer", {})
        out = {int(k): float(v["s_per_token_layer"])
               for k, v in per.items() if "s_per_token_layer" in v}
        if out:
            return out
    return {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_glob")
    ap.add_argument("--bandwidth", type=float, default=2.0,
                    help="GiB/s sustained read bandwidth")
    a = ap.parse_args()

    shards = sorted(glob.glob(a.model_glob))
    st = ExpertStore(shards[0], extra_shards=shards[1:])
    io = st.io_model(top_k=6, router_amortized=True)
    bw = a.bandwidth * 1e9

    ffn = load_ffn_times()
    measured = bool(ffn)
    layers = st.layers()
    n_hash = st.n_hash_layers

    rows = []
    for L in layers:
        pl = io["per_layer"].get(L, {})
        routed = pl.get("routed", 0.0)
        shexp = pl.get("shexp", 0.0)
        dense = pl.get("dense", 0.0)
        read_total = routed + shexp + dense
        ffn_t = ffn.get(L, FALLBACK["hash"] if L < n_hash else
                        FALLBACK["dense"])
        rows.append({
            "layer": L,
            "router": "hash" if L < n_hash else "dense",
            "ffn_s": ffn_t,
            "read_GiB": read_total / 1e9,
            "routed_GiB": routed / 1e9,
            "shexp_GiB": shexp / 1e9,
            "dense_GiB": dense / 1e9,
            "serial_s": ffn_t + read_total / bw,
            "pipelined_s": max(ffn_t, read_total / bw),
            "resident_s": ffn_t,
        })

    serial = sum(r["serial_s"] for r in rows)
    pipelined = sum(r["pipelined_s"] for r in rows)
    resident = sum(r["resident_s"] for r in rows)
    total_bytes = io["bytes_per_token"]
    agg = total_bytes / bw
    resident_bytes = io["routed_total"]
    resident_agg = resident_bytes / bw
    resident_ram = (io["shexp_total"] + io["dense_total"]) / 1e9

    # batch dedup: E experts, top-k=6 per token, B tokens per batch ->
    # expected distinct = E * (1 - (1 - k/E)^B); routed reads amortize.
    E = 384
    k = 6
    batch = {}
    for B in (4, 16, 64, 256):
        distinct = E * (1.0 - (1.0 - k / E) ** B)
        read_bytes = (io["routed_total"] * distinct / (k * B))
        batch[str(B)] = {
            "routed_GiB_per_token": read_bytes / 1e9,
            "distinct_experts": round(distinct, 1),
            "dedup_vs_single": (k * B) / distinct,
            "aggregate_tok_s_ceiling": bw / read_bytes,
        }

    result = {
        "measured_ffn": measured,
        "n_hash_layers": n_hash,
        "bandwidth_GiBs": a.bandwidth,
        "bytes_per_token_GiB": total_bytes / 1e9,
        "routed_GiB": io["routed_total"] / 1e9,
        "shexp_GiB": io["shexp_total"] / 1e9,
        "dense_GiB": io["dense_total"] / 1e9,
        "resident_ram_GiB": resident_ram,
        "projected_tok_s_serial": 1.0 / serial,
        "projected_tok_s_pipelined": 1.0 / pipelined,
        "projected_tok_s_resident": 1.0 / resident,
        "aggregate_tok_s_ceiling": 1.0 / agg,
        "aggregate_tok_s_resident_ceiling": 1.0 / resident_agg,
        "s_per_token_serial": serial,
        "s_per_token_pipelined": pipelined,
        "s_per_token_resident": resident,
        "s_per_token_reads": agg,
        "batch_dedup": batch,
        "per_layer": rows,
    }
    out = ROOT / "outputs" / "bench_lazy_full.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items()
                      if k != "per_layer"}, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
