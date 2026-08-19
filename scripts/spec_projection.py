"""What-if: speculative decoding on the V4-Pro with measured constants.

Measured on this box:
  - verifier: lazy-resident V4-Pro = 0.122 tok/s  -> T_V ~= 8.17 s/token
  - drafter : V4-Flash + DSpark = 0.82-0.99 tok/s -> T_D ~= 1.1 s/token
  - batch-256 aggregate ceiling: 0.708 tok/s (independent estimate)

Sweeps gamma and per-slot acceptance alpha with the spec_sim cost
model: T = E[accepted] / (gamma * T_D + T_V). Answers:
  - what acceptance does the drafter need for 1 tok/s?
  - what is the hard ceiling with a perfect drafter?
  - does the ceiling match the batch-256 estimate?

Usage: python scripts/spec_projection.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultratensor.conditional.drafting import expected_acceptance  # noqa: E402

T_V = 1.0 / 0.122      # s per verifier step (lazy-resident measured)
T_D = 1.0 / 0.90       # s per draft token (V4-Flash mid measured)
BATCH_CEILING = 0.708  # independent batch-256 aggregate estimate


def throughput(gamma, alpha):
    e = expected_acceptance([alpha] * gamma)
    return e / (gamma * T_D + T_V)


def main() -> int:
    report = {"T_V_s": round(T_V, 3), "T_D_s": round(T_D, 3),
              "batch256_ceiling_tok_s": BATCH_CEILING,
              "grid": [], "need_for_1_tok_s": None, "perfect_ceiling": {}}

    for gamma in (1, 4, 8, 16, 32):
        row = {"gamma": gamma}
        for alpha in (0.3, 0.5, 0.8, 1.0):
            row[f"a{alpha}"] = round(throughput(gamma, alpha), 3)
        report["grid"].append(row)

    # acceptance needed for 1 tok/s at the best gamma
    best = None
    for gamma in (4, 8, 16, 32, 64):
        for alpha in [round(0.3 + 0.05 * i, 2) for i in range(15)]:
            t = throughput(gamma, alpha)
            if t >= 1.0:
                best = {"gamma": gamma, "alpha": alpha, "tok_s": round(t, 3)}
                break
        if best:
            break
    report["need_for_1_tok_s"] = best

    for gamma in (4, 8, 16, 32, 64):
        report["perfect_ceiling"][gamma] = round(throughput(gamma, 1.0), 3)

    # q2_0 read-cut scenario: expert payload 619 -> 450.4 GB (0.727x) only
    # affects the routed 11.51 GiB of the 21.66 GiB/token active read.
    q2_ratio = (11.5089 * 0.727 + 1.9182 + 8.2360) / 21.6631
    t_v_q2 = T_V * q2_ratio
    def t_q2(gamma, alpha):
        return expected_acceptance([alpha] * gamma) / (gamma * T_D + t_v_q2)
    report["q2_0_readcut"] = {
        "read_ratio": round(q2_ratio, 4),
        "T_V_s": round(t_v_q2, 3),
        "baseline_tok_s": round(1.0 / t_v_q2, 4),
        "perfect_gamma32": round(t_q2(32, 1.0), 3),
    }

    out = ROOT / "outputs" / "spec_projection.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
