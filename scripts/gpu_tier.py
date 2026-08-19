"""GPU tier manager: one llama-server at a time on the 8 GB GPU.

The three IQ2_XS splices each run at 1.2-1.5 t/s SOLO, but co-resident
servers starve each other's VRAM (keep12u fell to 0.1 t/s). This module
ensures exactly one llama-server is running with the requested model,
swapping it when a different specialist is selected.

State lives in outputs/gpu_tier_state.json (idempotent, crash-safe).
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENGINE = Path("C:/Users/legom/hyperv4flash/engine")
LLAMA_SERVER = str(ENGINE / "llama-server.exe")
STATE = ROOT / "outputs" / "gpu_tier_state.json"


def _read() -> dict:
    try:
        return json.loads(STATE.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def _write(st: dict) -> None:
    STATE.write_text(json.dumps(st), encoding="utf-8")


def _server_running() -> bool:
    # llama-server listens on the tier port when healthy
    r = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "(Get-NetTCPConnection -LocalPort 8788 -State Listen "
         "-ErrorAction SilentlyContinue) -ne $null"],
        capture_output=True, text=True, timeout=30)
    return "True" in (r.stdout or "")


def _kill_servers() -> None:
    subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "Get-Process llama-server -ErrorAction SilentlyContinue | "
         "Stop-Process -Force -ErrorAction SilentlyContinue"],
        capture_output=True, text=True, timeout=60)


def _start(model: str, ngl: int) -> None:
    subprocess.Popen(
        [LLAMA_SERVER, "-m", model, "--host", "127.0.0.1", "--port",
         "8788", "-ngl", str(ngl), "-c", "512", "-t", "4",
         "--parallel", "1"],
        stdout=open(ROOT / "outputs" / "gpu_tier_server.log", "ab"),
        stderr=open(ROOT / "outputs" / "gpu_tier_server.err.log", "ab"))


def ensure_loaded(model: str, ngl: int = 12, warmup: int = 45) -> dict:
    """Make sure the GPU server is running `model`. Returns status."""
    st = _read()
    if st.get("model") == model and _server_running():
        return {"ok": True, "swapped": False, "model": model}

    _kill_servers()
    time.sleep(3)
    _start(model, ngl)
    t0 = time.time()
    while time.time() - t0 < 240:
        if _server_running():
            time.sleep(warmup)
            _write({"model": model, "ngl": ngl,
                    "ts": time.strftime("%Y-%m-%d %H:%M:%S")})
            return {"ok": True, "swapped": True, "model": model}
        time.sleep(5)
    _write({"model": None, "error": "start timeout",
            "ts": time.strftime("%Y-%m-%d %H:%M:%S")})
    return {"ok": False, "model": model, "error": "start timeout"}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: gpu_tier.py <model.gguf> [ngl]")
        sys.exit(1)
    print(ensure_loaded(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2
                        else 12))
