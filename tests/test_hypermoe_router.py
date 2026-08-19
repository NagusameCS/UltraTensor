"""Hyper-MoE router classification tests (heuristic v1)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultratensor.hypermoe.router import DomainRouter  # noqa: E402

ROUTER = DomainRouter()

CASES = [
    ("SELECT id, name FROM orders WHERE active = 1", "sql"),
    ("Implement a Rust trait for a struct.", "rust"),
    ("Write a Python function to reverse a string.", "python"),
    ("JavaScript array map and filter chain.", "javascript"),
    ("Prove that there are infinitely many primes.", "math"),
    ("Design a REST API endpoint with JWT auth.", "backend"),
    ("Build a React component that debounces input.", "frontend"),
    ("Write a pandas pipeline to clean a CSV.", "data"),
    ("Write a Dockerfile for a multi-stage build.", "devops"),
    ("fn merge(a: &[i32]) -> Vec<i32> {", "rust"),
]


def test_classify_cases():
    for text, expected in CASES:
        domain, score = ROUTER.classify(text)
        assert domain == expected, f"{text!r}: {domain} != {expected}"


def test_rank_returns_purpose_first():
    ranked = ROUTER.rank("Design a Python REST API endpoint.")
    assert ranked and ranked[0][0] == "backend"
