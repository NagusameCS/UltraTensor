"""V4 coding-SUBDOMAIN batteries: backend / frontend / data / devops.

Finer-grained splits than the 4-PL battery: same segment-meta format
so cluster_pl_census.py slices per subdomain and the keep-N builder
can rank per subdomain.

Writes outputs/domain_prompts.json.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tokenizers import Tokenizer  # noqa: E402

TOKENIZER_JSON = ROOT / "outputs" / "v4_tokenizer.json"

SEGMENTS = [
    ("backend", [
        "Design a REST API endpoint for user authentication with JWT.",
        "Implement a connection pool for a PostgreSQL database.",
        "Write a rate-limited message queue worker.",
        "Implement pagination for a list endpoint with cursor tokens.",
    ]),
    ("frontend", [
        "Build a React component that debounces an input field.",
        "Implement a CSS grid layout for a dashboard.",
        "Write a JavaScript drag and drop handler.",
        "Implement client-side form validation with error states.",
    ]),
    ("data", [
        "Write a pandas pipeline to clean and aggregate a CSV.",
        "Implement k-means clustering from scratch in numpy.",
        "Vectorize a feature engineering step over a dataframe.",
        "Implement gradient descent for logistic regression.",
    ]),
    ("devops", [
        "Write a Dockerfile for a multi-stage Python build.",
        "Write a Kubernetes deployment with rolling updates.",
        "Implement a health check endpoint with retry logic.",
        "Write a bash script to rotate logs and send them to S3.",
    ]),
]


def main() -> int:
    tok = Tokenizer.from_file(str(TOKENIZER_JSON))
    ids, meta = [], []
    for name, prompts in SEGMENTS:
        start = len(ids)
        seg = []
        for p in prompts:
            enc = tok.encode(p)
            if enc.ids:
                seg.extend(int(i) for i in enc.ids)
        ids.extend(seg)
        meta.append({"language": name, "start": start, "n": len(seg)})
    out = {"token_ids": ids, "segments": meta,
           "n_tokens": len(ids), "domain": "coding-subdomains"}
    dest = ROOT / "outputs" / "domain_prompts.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {dest}: {len(ids)} tokens, "
          f"{[(s['language'], s['n']) for s in meta]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
