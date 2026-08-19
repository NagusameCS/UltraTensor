"""V4 math-domain battery: prompts for a math census.

Same pipeline as the code census: math prompts -> cluster_dense_trace
-> cluster_code_census -> per-layer expert mass -> keep-N extraction
-> a math-specialized small model.

Writes outputs/math_prompts.json.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tokenizers import Tokenizer  # noqa: E402

TOKENIZER_JSON = ROOT / "outputs" / "v4_tokenizer.json"

PROMPTS = [
    # algebra / number theory
    "Prove that there are infinitely many primes.",
    "Solve x^2 + 5x + 6 = 0 for integer x.",
    "Compute the gcd of 1071 and 462 using Euclid's algorithm.",
    "Prove that sqrt(2) is irrational.",
    "What is the remainder of 7^100 divided by 13?",
    # calculus
    "Find the derivative of f(x) = x^3 ln(x).",
    "Evaluate the integral of e^(-x^2) from 0 to infinity.",
    "Find the Taylor series of sin(x) about x = 0.",
    "Compute the limit of (1 + 1/n)^n as n approaches infinity.",
    "Solve the differential equation dy/dx = y with y(0) = 1.",
    # linear algebra
    "Compute the eigenvalues of the matrix [[2,1],[1,2]].",
    "Prove that the inverse of a product of matrices is (AB)^-1 = B^-1 A^-1.",
    "Find the rank of the matrix [[1,2,3],[4,5,6],[7,8,9]].",
    # combinatorics / probability
    "How many ways can 5 distinct books be arranged on a shelf?",
    "Compute the expected value of a fair six-sided die roll.",
    "In how many ways can 10 people be split into two teams of 5?",
    # proofs / misc
    "Prove by induction that the sum of the first n integers is n(n+1)/2.",
    "State and prove the pigeonhole principle.",
    "Prove that a triangle with integer side lengths 3, 4, 5 is right.",
    "What is the probability of rolling two dice and getting a sum of 7?",
]


def main() -> int:
    tok = Tokenizer.from_file(str(TOKENIZER_JSON))
    ids = []
    for p in PROMPTS:
        enc = tok.encode(p)
        if enc.ids:
            ids.extend(int(i) for i in enc.ids)
    out = {"prompts": PROMPTS, "token_ids": ids,
           "segments": [{"language": "math", "start": 0,
                          "n": len(ids)}],
           "n_tokens": len(ids), "domain": "math"}
    dest = ROOT / "outputs" / "math_prompts.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {dest}: {len(PROMPTS)} prompts, {len(ids)} tokens")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
