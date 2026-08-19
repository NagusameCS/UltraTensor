"""Hyper-MoE dispatcher: classify -> specialist -> generate.

Usage:
    python scripts/hypermoe_serve.py --prompt "SELECT ..." --n 8
    python scripts/hypermoe_serve.py --serve 8790
"""

import argparse
import json
import subprocess
import sys
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultratensor.hypermoe.router import DomainRouter  # noqa: E402

LLAMA_CLI = "C:/Users/legom/hyperv4flash/engine/llama-cli.exe"
NUMPY_SERVE = str(ROOT / "scripts" / "v4_coder_serve.py")
REGISTRY = str(ROOT / "outputs" / "hypermoe_registry.json")


class Dispatcher:
    def __init__(self, registry_path=REGISTRY):
        self.reg = json.load(open(registry_path, encoding="utf-8-sig"))
        self.router = DomainRouter()
        self.fallbacks = self.reg.get("fallbacks", [])
        self.threshold = 0.0

    def resolve(self, text: str, threshold: float = 0.0):
        """-> specialist dict with an existing model file, or None.

        Escalation inference: if the best domain score is below
        `threshold`, the request skips specialists and goes straight
        to the fallback chain (the general model)."""
        ranked = self.router.rank(text)
        if ranked and ranked[0][1] >= threshold:
            for domain, score in ranked:
                for s in self.reg["specialists"]:
                    if s["domain"] == domain and s["model"] and \
                            Path(s["model"]).exists():
                        return s
        for fb in self.fallbacks:
            for s in self.reg["specialists"]:
                if s["domain"] == fb and s["model"] and \
                        Path(s["model"]).exists():
                    return s
        return None

    def generate(self, text: str, n: int = 8, threads: int = 6):
        """-> dict(domain, model, backend, text)."""
        spec = self.resolve(text, self.threshold)
        if spec is None:
            return {"error": "no eligible specialist (no model built)"}
        backend = spec["backend"]
        if backend == "llama-http":
            port = spec.get("port", 8788)
            if port == 8788:  # local GPU tier: one model at a time
                sys.path.insert(0, str(ROOT / "scripts"))
                import gpu_tier
                ensure = gpu_tier.ensure_loaded(spec["model"],
                                                spec.get("ngl", 12))
                if not ensure.get("ok"):
                    return {"error": ensure.get("error", "gpu tier"),
                            "model": spec["id"]}
            payload = json.dumps({"prompt": text, "n_predict": n,
                                  "temperature": 0.0}).encode()
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/completion", data=payload,
                headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=3600) as r:
                    body = json.loads(r.read())
                return {"domain": spec["domain"], "model": spec["id"],
                        "backend": backend,
                        "raw": body.get("content", ""), "rc": 0}
            except Exception as e:  # noqa: BLE001
                return {"error": str(e), "model": spec["id"]}
        if backend == "llama-cli":
            proc = subprocess.run(
                [LLAMA_CLI, "-m", spec["model"], "-p", text,
                 "-n", str(n), "-t", str(threads), "-ngl", "0",
                 "-c", "512", "--no-display-prompt"],
                capture_output=True, text=True, timeout=24 * 3600)
            out = (proc.stdout or "") + (proc.stderr or "")
            return {"domain": spec["domain"], "model": spec["id"],
                    "backend": backend, "raw": out[-2000:],
                    "rc": proc.returncode}
        if backend == "numpy":
            proc = subprocess.run(
                [sys.executable, NUMPY_SERVE, "--prompt", text,
                 "--n", str(n), "--temp", "0"],
                capture_output=True, text=True, timeout=24 * 3600)
            return {"domain": spec["domain"], "model": spec["id"],
                    "backend": backend, "raw": proc.stdout[-2000:],
                    "rc": proc.returncode}
        return {"error": f"unknown backend {backend}"}


class Handler(BaseHTTPRequestHandler):
    dispatcher = None

    def do_POST(self):
        if self.path != "/v1/generate":
            self.send_response(404)
            self.end_headers()
            return
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        result = self.dispatcher.generate(
            body.get("prompt", ""), int(body.get("n", 8)))
        payload = json.dumps(result).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *a):
        pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", default=REGISTRY)
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--serve", type=int, default=0)
    ap.add_argument("--resolve-only", action="store_true",
                    help="print the routing decision without generating")
    ap.add_argument("--escalation-threshold", type=float, default=0.0,
                    help="Uriel gate: requests below this router score go "
                         "straight to the general model")
    a = ap.parse_args()

    d = Dispatcher(a.registry)
    d.threshold = a.escalation_threshold
    if a.serve:
        Handler.dispatcher = d
        srv = HTTPServer(("127.0.0.1", a.serve), Handler)
        print(f"hyper-moe serving on http://127.0.0.1:{a.serve}"
              f"/v1/generate", flush=True)
        srv.serve_forever()
    spec = d.resolve(a.prompt)
    if spec is None:
        print("no eligible specialist built for this prompt")
        print("router ranked:", d.router.rank(a.prompt))
        return 1
    print(f"route: {a.prompt!r} -> {spec['id']} ({spec['backend']})"
          f" [threshold {a.escalation_threshold}]",
          flush=True)
    if a.resolve_only:
        return 0
    print(json.dumps(d.generate(a.prompt, a.n), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
