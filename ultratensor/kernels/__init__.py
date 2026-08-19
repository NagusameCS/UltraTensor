"""UltraTensor Phase 2 kernels: fused factored GEMV (CUDA).

y = U @ (C @ x) with U fp16 [m,k] and C uq4 codes [k,n]. The .cu compiles
standalone (no torch headers), loads via ctypes, and takes torch device
buffers so it coexists with any runtime.
"""
from __future__ import annotations

import ctypes
import subprocess
import sys
from pathlib import Path

import numpy as np

_IS_WIN = sys.platform == "win32"

_HERE = Path(__file__).resolve().parent
_CU = _HERE / "factored_gemv.cu"
_CU_MOE = _HERE / "factored_gemv_moe.cu"
_C_REF = _HERE / "factored_gemv_ref.c"
_C_EXEC = _HERE / "factored_exec.c"
_DLL = _HERE / "factored_gemv.dll"
_DLL_MOE = _HERE / "factored_gemv_moe.dll"
_DLL_CPU = _HERE / "factored_gemv_cpu.dll"
_DLL_EXEC = _HERE / "factored_exec.dll"

_VCVARS = (r"C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools"
           r"\VC\Auxiliary\Build\vcvars64.bat")


def build_factored_gemv(arch: str = "sm_89") -> Path:
    """Compile the .cu files with nvcc (MSVC host) into DLLs."""
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    if arch == "auto":
        cap = torch.cuda.get_device_capability(0)
        arch = f"sm_{cap[0]}{cap[1]}"
    cmd = (f'call "{_VCVARS}" >nul && nvcc -shared -O3 --use_fast_math '
           f'--cudart static -Xcompiler "/MD" -gencode=arch=compute_'
           f'{arch[3:]},code={arch} "{_CU}" -o "{_DLL}" '
           f'&& nvcc -shared -O3 --use_fast_math --cudart static '
           f'-Xcompiler "/MD" -gencode=arch=compute_{arch[3:]},code={arch} '
           f'"{_CU_MOE}" -o "{_DLL_MOE}"')
    # shell=True: pass the string verbatim (list form would escape quotes)
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                       cwd=str(_HERE))
    if r.returncode != 0 or not _DLL.exists() or not _DLL_MOE.exists():
        raise RuntimeError(f"nvcc failed:\n{r.stdout}\n{r.stderr}")
    return _DLL


def build_factored_gemv_cpu() -> Path:
    """Compile the portable C twin with MSVC into factored_gemv_cpu.dll."""
    cmd = (f'call "{_VCVARS}" >nul && cl /nologo /O2 /LD "{_C_REF}" '
           f'/Fe:"{_DLL_CPU}" >nul')
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                       cwd=str(_HERE))
    if r.returncode != 0 or not _DLL_CPU.exists():
        raise RuntimeError(f"cl failed:\n{r.stdout}\n{r.stderr}")
    return _DLL_CPU


def build_factored_exec() -> Path:
    """Compile the C container loader+executor into factored_exec.dll."""
    cmd = (f'call "{_VCVARS}" >nul && cl /nologo /O2 /LD "{_C_EXEC}" '
           f'/Fe:"{_DLL_EXEC}" >nul')
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                       cwd=str(_HERE))
    if r.returncode != 0 or not _DLL_EXEC.exists():
        raise RuntimeError(f"cl failed:\n{r.stdout}\n{r.stderr}")
    return _DLL_EXEC


class FactoredGEMV:
    """ctypes binding for factored_gemv_uq4."""

    def __init__(self):
        if not _DLL.exists():
            build_factored_gemv()
        self._lib = ctypes.CDLL(str(_DLL))
        fn = self._lib.factored_gemv_uq4
        fn.restype = ctypes.c_int
        fn.argtypes = ([ctypes.c_void_p] * 3 + [ctypes.c_int] * 3 +
                       [ctypes.c_void_p] * 4)

    def __call__(self, U, scales, packed, x):
        """torch tensors on CUDA. U fp16 [m,k], scales fp32 [k,n/32],
        packed uint8 [k,n/2], x fp32 [n]. Returns y fp32 [m]."""
        import torch
        m, k = U.shape
        k2, nb = scales.shape
        n = nb * 32
        assert packed.shape == (k, n // 2)
        assert x.numel() == n and x.dtype == torch.float32
        y = torch.empty(m, device=x.device, dtype=torch.float32)
        t = torch.zeros(k, device=x.device, dtype=torch.float32)
        stream = torch.cuda.current_stream(x.device.index or 0).cuda_stream
        rc = self._lib.factored_gemv_uq4(
            ctypes.c_void_p(U.data_ptr()),
            ctypes.c_void_p(scales.data_ptr()),
            ctypes.c_void_p(packed.data_ptr()),
            ctypes.c_int(m), ctypes.c_int(k), ctypes.c_int(n),
            ctypes.c_void_p(x.data_ptr()),
            ctypes.c_void_p(y.data_ptr()),
            ctypes.c_void_p(t.data_ptr()),
            ctypes.c_void_p(stream))
        if rc != 0:
            raise RuntimeError(f"factored_gemv_uq4 rc={rc}")
        return y


def numpy_reference(U: np.ndarray, scales: np.ndarray,
                    packed: np.ndarray, x: np.ndarray) -> np.ndarray:
    """CPU reference: y = U @ dequant(C) @ x."""
    from ..quant import uq4_dequantize
    C = uq4_dequantize(scales, packed)
    return (U.astype(np.float32) @ C) @ x


class FactoredGEMVCPU:
    """ctypes binding for the portable C twin (factored_gemv_uq4_cpu)."""

    def __init__(self):
        if not _DLL_CPU.exists():
            build_factored_gemv_cpu()
        self._lib = ctypes.CDLL(str(_DLL_CPU))
        fn = self._lib.factored_gemv_uq4_cpu
        fn.restype = ctypes.c_int
        fn.argtypes = ([ctypes.c_void_p] * 3 + [ctypes.c_int] * 3 +
                       [ctypes.c_void_p] * 3)

    def __call__(self, U, scales, packed, x):
        """numpy arrays (float32). Returns y [m]."""
        m, k = U.shape
        k2, nb = scales.shape
        n = nb * 32
        assert (k2,) == (k,) and packed.shape == (k, n // 2)
        y = np.empty(m, np.float32)
        t = np.empty(k, np.float32)
        rc = self._lib.factored_gemv_uq4_cpu(
            ctypes.c_void_p(U.ctypes.data),
            ctypes.c_void_p(scales.ctypes.data),
            ctypes.c_void_p(packed.ctypes.data),
            ctypes.c_int(m), ctypes.c_int(k), ctypes.c_int(n),
            ctypes.c_void_p(x.ctypes.data),
            ctypes.c_void_p(y.ctypes.data),
            ctypes.c_void_p(t.ctypes.data))
        if rc != 0:
            raise RuntimeError(f"factored_gemv_uq4_cpu rc={rc}")
        return y


_factored_gemv_cpu: FactoredGEMVCPU | None = None


def factored_gemv_cpu(U, scales, packed, x):
    global _factored_gemv_cpu
    if _factored_gemv_cpu is None:
        _factored_gemv_cpu = FactoredGEMVCPU()
    return _factored_gemv_cpu(U, scales, packed, x)


class UTFactoredTensor(ctypes.Structure):
    _fields_ = [
        ("U", ctypes.c_void_p),
        ("scales", ctypes.c_void_p),
        ("packed", ctypes.c_void_p),
        ("E", ctypes.c_int),
        ("m", ctypes.c_int),
        ("k", ctypes.c_int),
        ("n", ctypes.c_int),
    ]


class FactoredExec:
    """ctypes binding for the C container loader + executor
    (ut_factored_load / ut_factored_gemv / ut_factored_free)."""

    def __init__(self):
        if not _DLL_EXEC.exists():
            build_factored_exec()
        self._lib = ctypes.CDLL(str(_DLL_EXEC))
        self._lib.ut_factored_load.restype = ctypes.c_int
        self._lib.ut_factored_load.argtypes = [ctypes.c_char_p, ctypes.c_char_p,
                                               ctypes.POINTER(UTFactoredTensor)]
        self._lib.ut_factored_free.argtypes = [ctypes.POINTER(UTFactoredTensor)]
        self._lib.ut_factored_gemv.restype = ctypes.c_int
        self._lib.ut_factored_gemv.argtypes = [ctypes.POINTER(UTFactoredTensor),
                                               ctypes.c_void_p, ctypes.c_void_p]
        self._t = UTFactoredTensor()

    def load(self, path: str, name: str):
        rc = self._lib.ut_factored_load(path.encode(), name.encode(),
                                        ctypes.byref(self._t))
        if rc != 0:
            raise RuntimeError(f"ut_factored_load rc={rc}")
        return self

    def gemv(self, x: np.ndarray) -> np.ndarray:
        y = np.empty(self._t.m, np.float32)
        x32 = np.ascontiguousarray(x, np.float32)
        rc = self._lib.ut_factored_gemv(
            ctypes.byref(self._t), x32.ctypes.data_as(ctypes.c_void_p),
            y.ctypes.data_as(ctypes.c_void_p))
        if rc != 0:
            raise RuntimeError(f"ut_factored_gemv rc={rc}")
        return y

    def close(self):
        self._lib.ut_factored_free(ctypes.byref(self._t))

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


_factored_gemv: FactoredGEMV | None = None


def factored_gemv(U, scales, packed, x):
    global _factored_gemv
    if _factored_gemv is None:
        _factored_gemv = FactoredGEMV()
    return _factored_gemv(U, scales, packed, x)


_C_EXPERT = _HERE / "expert_gemv.c"
_DLL_EXPERT = _HERE / "expert_gemv.dll"
_SO_EXPERT = _HERE / "expert_gemv.so"


def _expert_lib_path() -> Path:
    return _DLL_EXPERT if _IS_WIN else _SO_EXPERT


def build_expert_gemv() -> Path:
    """Compile the dispatch-aware expert executor into expert_gemv.dll
    (MSVC on Windows) or expert_gemv.so (gcc on Linux).

    NOTE: built WITHOUT /openmp — the MSVC classic OpenMP build passes
    tests but corrupts the heap under Python-thread concurrency (native
    access violations in both the reader thread and the decode thread,
    ~50% of runs; see the 2026-08-15 v4pro session). Single-threaded
    decode is the supported configuration; the AVX2 fused dots keep it
    fast enough for the lazy path."""
    if _IS_WIN:
        cmd = (f'call "{_VCVARS}" >nul && cl /nologo /O2 /arch:AVX2 '
               f'/LD "{_C_EXPERT}" /Fe:"{_DLL_EXPERT}" >nul')
        out = _DLL_EXPERT
    else:
        cmd = (f'gcc -shared -O3 -fPIC -mavx2 -mfma "{_C_EXPERT}" '
               f'-o "{_SO_EXPERT}"')
        out = _SO_EXPERT
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                       cwd=str(_HERE))
    if r.returncode != 0 or not out.exists():
        raise RuntimeError(f"build failed:\n{r.stdout}\n{r.stderr}")
    return out


class UTExpertStore(ctypes.Structure):
    _fields_ = [
        ("f", ctypes.c_void_p),
        ("data_start", ctypes.c_uint64),
        ("tensor_off", ctypes.c_uint64),
        ("ttype", ctypes.c_uint32),
        ("n", ctypes.c_int),
        ("m", ctypes.c_int),
        ("E", ctypes.c_int),
        ("row_blocks", ctypes.c_int),
        ("block_align", ctypes.c_int),
        ("block_bytes", ctypes.c_int),
    ]


class ExpertGEMV:
    """ctypes binding for ut_expert_open / ut_expert_gemv / ut_expert_close:
    per-expert decode + GEMV straight from a GGUF shard (Phase 3b)."""

    def __init__(self):
        lib = _expert_lib_path()
        if not lib.exists():
            build_expert_gemv()
        self._lib = ctypes.CDLL(str(lib))
        self._lib.ut_expert_open.restype = ctypes.c_int
        self._lib.ut_expert_open.argtypes = [ctypes.c_char_p, ctypes.c_char_p,
                                             ctypes.POINTER(UTExpertStore)]
        self._lib.ut_expert_close.argtypes = [ctypes.POINTER(UTExpertStore)]
        self._lib.ut_expert_gemv.restype = ctypes.c_int
        self._lib.ut_expert_gemv.argtypes = [ctypes.POINTER(UTExpertStore),
                                             ctypes.c_int, ctypes.c_void_p,
                                             ctypes.c_void_p, ctypes.c_int]
        self._lib.ut_expert_gemv_batch.restype = ctypes.c_int
        self._lib.ut_expert_gemv_batch.argtypes = [
            ctypes.POINTER(UTExpertStore), ctypes.c_int, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
        self._lib.ut_expert_gemv_mem.restype = ctypes.c_int
        self._lib.ut_expert_gemv_mem.argtypes = [
            ctypes.POINTER(UTExpertStore), ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_int]
        self._t = UTExpertStore()

    def open(self, path: str, name: str):
        rc = self._lib.ut_expert_open(path.encode(), name.encode(),
                                      ctypes.byref(self._t))
        if rc != 0:
            raise RuntimeError(f"ut_expert_open({name}) rc={rc}")
        return self

    def gemv(self, e: int, x: np.ndarray, transpose: bool = False) -> np.ndarray:
        n = self._t.n
        m = self._t.m
        x32 = np.ascontiguousarray(x, np.float32)
        y = np.empty(n if transpose else m, np.float32)
        rc = self._lib.ut_expert_gemv(
            ctypes.byref(self._t), ctypes.c_int(e),
            x32.ctypes.data_as(ctypes.c_void_p),
            y.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(transpose))
        if rc != 0:
            raise RuntimeError(f"ut_expert_gemv rc={rc}")
        return y

    def gemv_batch(self, e: int, X: np.ndarray,
                   transpose: bool = False) -> np.ndarray:
        """X [B, n] -> [B, m] (or [B, n] with transpose). One expert read
        amortized over the batch."""
        n = self._t.n
        m = self._t.m
        X32 = np.ascontiguousarray(X, np.float32)
        B = X32.shape[0]
        assert X32.shape[1] == n
        out_elems = n if transpose else m
        y = np.empty((B, out_elems), np.float32)
        rc = self._lib.ut_expert_gemv_batch(
            ctypes.byref(self._t), ctypes.c_int(e),
            X32.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(B),
            y.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(transpose))
        if rc != 0:
            raise RuntimeError(f"ut_expert_gemv_batch rc={rc}")
        return y

    def gemv_mem(self, raw: bytes, x: np.ndarray,
                 transpose: bool = False) -> np.ndarray:
        """Decode from a pre-read expert buffer (read/decode overlap)."""
        n = self._t.n
        m = self._t.m
        buf = np.frombuffer(raw, np.uint8)
        x32 = np.ascontiguousarray(x, np.float32)
        y = np.empty(n if transpose else m, np.float32)
        rc = self._lib.ut_expert_gemv_mem(
            ctypes.byref(self._t),
            buf.ctypes.data_as(ctypes.c_void_p),
            x32.ctypes.data_as(ctypes.c_void_p),
            y.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(transpose))
        if rc != 0:
            raise RuntimeError(f"ut_expert_gemv_mem rc={rc}")
        return y

    @property
    def shape(self):
        return (self._t.n, self._t.m, self._t.E)

    def close(self):
        self._lib.ut_expert_close(ctypes.byref(self._t))

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class FactoredGEMVMoE:
    """ctypes binding for factored_gemv_moe_uq4."""

    def __init__(self):
        if not _DLL_MOE.exists():
            build_factored_gemv()
        self._lib = ctypes.CDLL(str(_DLL_MOE))
        fn = self._lib.factored_gemv_moe_uq4
        fn.restype = ctypes.c_int
        fn.argtypes = ([ctypes.c_void_p] * 3 + [ctypes.c_int] * 4 +
                       [ctypes.c_void_p] * 4)

    def __call__(self, U, scales, packed, x):
        """torch tensors. U fp16 [E,m,k], scales fp32 [E,k,n/32],
        packed uint8 [E,k,n/2], x fp32 [n]. Returns y fp32 [m]."""
        import torch
        E, m, k = U.shape
        E2, k2, nb = scales.shape
        n = nb * 32
        assert (E2, k2) == (E, k)
        assert packed.shape == (E, k, n // 2)
        assert x.numel() == n and x.dtype == torch.float32
        y = torch.zeros(m, device=x.device, dtype=torch.float32)
        t = torch.zeros(E * k, device=x.device, dtype=torch.float32)
        stream = torch.cuda.current_stream(x.device.index or 0).cuda_stream
        rc = self._lib.factored_gemv_moe_uq4(
            ctypes.c_void_p(U.data_ptr()),
            ctypes.c_void_p(scales.data_ptr()),
            ctypes.c_void_p(packed.data_ptr()),
            ctypes.c_int(E), ctypes.c_int(m), ctypes.c_int(k), ctypes.c_int(n),
            ctypes.c_void_p(x.data_ptr()),
            ctypes.c_void_p(y.data_ptr()),
            ctypes.c_void_p(t.data_ptr()),
            ctypes.c_void_p(stream))
        if rc != 0:
            raise RuntimeError(f"factored_gemv_moe_uq4 rc={rc}")
        return y


_factored_gemv_moe: FactoredGEMVMoE | None = None


def factored_gemv_moe(U, scales, packed, x):
    global _factored_gemv_moe
    if _factored_gemv_moe is None:
        _factored_gemv_moe = FactoredGEMVMoE()
    return _factored_gemv_moe(U, scales, packed, x)


def factored_gemv_from_gguf(path, name, x):
    """Phase 3 connector: load a factored container tensor and run the fused
    CUDA kernel on it. name = 't1.factored_C' (or 't1'). Returns y fp32 [m]."""
    import torch
    from ..gguf_factored import read_factored_gguf
    _, tensors = read_factored_gguf(path)
    cname = name if name.endswith(".factored_C") else name + ".factored_C"
    base = cname[:-len(".factored_C")]
    U = torch.from_numpy(tensors[base + ".factored_U"]["U"]).cuda()
    codes = tensors[cname]
    scales = torch.from_numpy(codes["scales"].copy()).cuda()
    packed = torch.from_numpy(codes["packed"].copy()).cuda()
    if codes["scales"].ndim == 3:  # expert stack -> MoE kernel
        return factored_gemv_moe(U, scales, packed, x)
    return factored_gemv(U, scales, packed, x)


def factored_gemv_moe_from_gguf(path, name, x):
    """Expert-stack connector: 3-D factored tensor -> MoE batched kernel."""
    return factored_gemv_from_gguf(path, name, x)


def bench(m=4096, k=32, n=12288, iters=200):
    """Speed test vs torch two-pass GEMV. Prints a table."""
    import time
    import torch
    dev = "cuda"
    rng = np.random.default_rng(0)
    U = torch.from_numpy(rng.standard_normal((m, k)).astype(np.float16)).to(dev)
    C = torch.from_numpy(rng.standard_normal((k, n)).astype(np.float32)).to(dev)
    # uq4-encode C
    Cb = C.reshape(k, n // 32, 32)
    sc = (Cb.abs().amax(dim=-1) / 8.0).clamp_min(1e-6)
    q = (Cb / sc.unsqueeze(-1) + 8.0).round().clamp(0, 15).to(torch.uint8)
    q = q.reshape(k, n // 2, 2)
    packed = (q[..., 0] | (q[..., 1] << 4)).contiguous()
    scales = sc.contiguous()
    x = torch.randn(n, device=dev, dtype=torch.float32)
    # correctness
    y = factored_gemv(U, scales, packed, x)
    y_ref = (U.float() @ C) @ x
    err = (y - y_ref).abs().max().item() / y_ref.abs().max().item()
    # warmup + timed
    for _ in range(10):
        factored_gemv(U, scales, packed, x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        factored_gemv(U, scales, packed, x)
    torch.cuda.synchronize()
    dt_fact = (time.perf_counter() - t0) / iters
    for _ in range(10):
        (U.float() @ C) @ x
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        (U.float() @ C) @ x
    torch.cuda.synchronize()
    dt_two = (time.perf_counter() - t0) / iters
    flops = 2.0 * m * k + 2.0 * k * n
    print(f"factored GEMV {m}x{k}x{n}: max rel err {err:.2e}")
    print(f"  fused  uq4: {dt_fact*1e3:8.3f} ms  ({flops/dt_fact/1e9:7.1f} GFLOP/s)")
    print(f"  two-pass fp32: {dt_two*1e3:8.3f} ms  ({flops/dt_two/1e9:7.1f} GFLOP/s)")
    print(f"  speedup: {dt_two/dt_fact:5.2f}x")
