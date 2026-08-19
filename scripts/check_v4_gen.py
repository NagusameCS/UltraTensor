"""Compare C greedy-generation tokens (space-separated file from
test_v4_gen) against the numpy oracle outputs/v4gen_tokens.txt.

Usage: check_v4_gen.py <c_tokens.txt>
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    c_path = Path(sys.argv[1])
    oracle = Path(ROOT, "outputs", "v4gen_tokens.txt").read_text().split()
    got = c_path.read_text().split()
    if oracle != got:
        print(f"FAIL: oracle={oracle} c={got}")
        for i, (a, b) in enumerate(zip(oracle, got)):
            if a != b:
                print(f"  first mismatch at token {i}: oracle {a} vs C {b}")
                break
        return 1
    print(f"PASS: {len(oracle)} generated tokens match the oracle exactly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
