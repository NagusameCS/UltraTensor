"""Thermal-adaptive rank + tokens-per-joule (port of thermal_rank.c).

Conditional rank driven by the HARDWARE budget, not model uncertainty:
- Below ``temp_low``   -> rank_max (fastest, coolest)
- Above ``temp_high``  -> rank_min (protects the machine)
- Between              -> linear interpolation
- Power over budget    -> further rank downscale by budget/power.

TPJ (tokens per joule) records joules/token and estimates ``rank_coeff``
(joules per unit rank), then exposes a policy-gradient energy term that
nudges a softmax rank policy towards cheaper ranks.

The NVML sensor is a direct ctypes port of the C loader
(nvml.dll, nvmlInit(_v2), nvmlDeviceGetHandleByIndex(_v2),
nvmlDeviceGetTemperature, nvmlDeviceGetPowerUsage) with graceful
fallback so the module works on any machine.
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
from dataclasses import dataclass, field
from typing import Optional, Protocol

import numpy as np

NVML_SUCCESS = 0
NVML_TEMPERATURE_GPU = 0
_DLL_PATHS = (
    "nvml.dll",
    r"C:\Program Files\NVIDIA Corporation\NVSMI\nvml.dll",
    r"C:\Windows\System32\nvml.dll",
)

DEFAULT_RANK_LEVELS = np.array([8, 16, 32, 64, 128, 256], dtype=np.float64)
TPJ_HISTORY_LEN = 256


class ThermalSensor(Protocol):
    def read(self) -> Optional[tuple[float, float]]:  # (temp_C, power_W)
        ...


class NullSensor:
    """Always-unavailable sensor; thermal adaptation stays disabled."""

    def read(self) -> Optional[tuple[float, float]]:
        return None


class NvmlCtypesSensor:
    """NVML via ctypes — the exact loader logic of nvml_load/nvml_read."""

    def __init__(self, index: int = 0) -> None:
        self._lib = None
        self._device = ctypes.c_void_p(0)
        self._load(index)

    # -- loading ----------------------------------------------------------
    def _load(self, index: int) -> None:
        for path in _DLL_PATHS:
            try:
                lib = ctypes.WinDLL(path)
            except OSError:
                continue
            fn_init = self._sym(lib, "nvmlInit_v2") or self._sym(lib, "nvmlInit")
            fn_shutdown = self._sym(lib, "nvmlShutdown")
            fn_handle = self._sym(lib, "nvmlDeviceGetHandleByIndex_v2") or self._sym(
                lib, "nvmlDeviceGetHandleByIndex"
            )
            fn_temp = self._sym(lib, "nvmlDeviceGetTemperature")
            fn_power = self._sym(lib, "nvmlDeviceGetPowerUsage")
            if not (fn_init and fn_handle and fn_temp and fn_power):
                continue
            if fn_init() != NVML_SUCCESS:
                continue
            dev = ctypes.c_void_p(0)
            if fn_handle(ctypes.c_uint(index), ctypes.byref(dev)) != NVML_SUCCESS:
                fn_shutdown()
                continue
            self._lib = lib
            self._device = dev
            self._fn_shutdown = fn_shutdown
            self._fn_temp = fn_temp
            self._fn_power = fn_power
            self._fn_handle = fn_handle
            return
        self._lib = None

    @staticmethod
    def _sym(lib: ctypes.WinDLL, name: str):
        try:
            fn = getattr(lib, name)
        except AttributeError:
            return None
        if name == "nvmlShutdown":
            fn.restype = wt.UINT
        elif name in ("nvmlDeviceGetHandleByIndex", "nvmlDeviceGetHandleByIndex_v2"):
            fn.restype = wt.UINT
            fn.argtypes = [wt.UINT, ctypes.POINTER(ctypes.c_void_p)]
        elif name in ("nvmlInit", "nvmlInit_v2"):
            fn.restype = wt.UINT
        elif name == "nvmlDeviceGetTemperature":
            fn.restype = wt.UINT
            fn.argtypes = [ctypes.c_void_p, wt.UINT, ctypes.POINTER(wt.UINT)]
        elif name == "nvmlDeviceGetPowerUsage":
            fn.restype = wt.UINT
            fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(wt.UINT)]
        return fn

    # -- reading ----------------------------------------------------------
    @property
    def ok(self) -> bool:
        return self._lib is not None

    def read(self) -> Optional[tuple[float, float]]:
        if not self.ok:
            return None
        temp = wt.UINT(0)
        mw = wt.UINT(0)
        r1 = self._fn_temp(self._device, NVML_TEMPERATURE_GPU, ctypes.byref(temp))
        r2 = self._fn_power(self._device, ctypes.byref(mw))
        if r1 != NVML_SUCCESS or r2 != NVML_SUCCESS:
            return None
        return float(temp.value), float(mw.value) * 0.001

    def close(self) -> None:
        if self._lib is not None:
            self._fn_shutdown()
            self._lib = None


@dataclass
class ThermalRank:
    """Port of thermal_ctx_t + thermal_get_rank()."""

    sensor: ThermalSensor
    temp_low_c: float = 65.0
    temp_high_c: float = 85.0
    power_budget_w: float = 0.0
    rank_min: int = 8
    rank_max: int = 256

    def __post_init__(self) -> None:
        if self.sensor is None:
            self.sensor = NullSensor()
        if self.temp_low_c <= 0.0:
            self.temp_low_c = 65.0
        if self.temp_high_c <= 0.0:
            self.temp_high_c = 85.0
        if self.rank_min <= 0:
            self.rank_min = 8
        if self.rank_max <= self.rank_min:
            self.rank_max = 256
        self.current_temp_c = self.temp_low_c
        self.current_power_w = 0.0
        self.nvml_ok = False

    def poll(self) -> bool:
        reading = self.sensor.read()
        if reading is None:
            return False
        self.current_temp_c, self.current_power_w = reading
        self.nvml_ok = True
        return True

    def get_rank(self, base_rank: int) -> int:
        """Rank for the current hardware state (thermal_get_rank logic).

        Returns ``base_rank`` untouched when the sensor is unavailable.
        """
        if not self.poll():
            return base_rank

        lo, hi = self.temp_low_c, self.temp_high_c
        t = self.current_temp_c
        span = hi - lo
        if span < 0.1 or t <= lo:
            scaled = self.rank_max
        elif t >= hi:
            scaled = self.rank_min
        else:
            frac = (t - lo) / span
            fval = self.rank_max - frac * (self.rank_max - self.rank_min)
            scaled = int(fval + 0.5)

        if self.power_budget_w > 0.0 and self.current_power_w > self.power_budget_w:
            pscale = self.power_budget_w / self.current_power_w
            prank = int(scaled * pscale + 0.5)
            if prank < scaled:
                scaled = prank

        return int(min(max(scaled, self.rank_min), self.rank_max))


@dataclass
class TpjTracker:
    """Port of tpj_ctx_t: joules-per-token history + rank_coeff estimate."""

    thermal: ThermalRank
    lambda_: float = 0.005
    rank_levels: np.ndarray = field(default_factory=lambda: DEFAULT_RANK_LEVELS.copy())
    history_len: int = TPJ_HISTORY_LEN

    def __post_init__(self) -> None:
        if self.lambda_ <= 0.0:
            self.lambda_ = 0.005
        self.rank_levels = np.asarray(self.rank_levels, dtype=np.float64)
        self.rank_coeff = 0.0
        self.joules_history: list[float] = []
        self.cumulative_joules = 0.0
        self.cumulative_tokens = 0

    @property
    def mid_rank(self) -> float:
        return float(self.rank_levels[len(self.rank_levels) // 2])

    def record(self, tokens_per_second: float) -> float:
        """Record a decode observation; returns joules per token (0 if unknown)."""
        if tokens_per_second <= 0.0:
            return 0.0
        power_w = 0.0
        if self.thermal.poll():
            power_w = self.thermal.current_power_w
        if power_w <= 0.0:
            return 0.0

        joules_per_token = power_w / tokens_per_second
        if len(self.joules_history) < self.history_len:
            self.joules_history.append(joules_per_token)
        else:
            self.joules_history = self.joules_history[1:] + [joules_per_token]
        self.cumulative_joules += joules_per_token
        self.cumulative_tokens += 1

        if len(self.joules_history) >= 4:
            mean_j = float(np.mean(self.joules_history))
            mid = self.mid_rank
            self.rank_coeff = mean_j / (mid if mid > 0.0 else 1.0)
        return joules_per_token

    def bootstrap(self, tps_estimate: float) -> None:
        """Seed rank_coeff without history (tpj_bootstrap)."""
        if tps_estimate <= 0.0:
            return
        if not self.thermal.poll():
            return
        power_w = self.thermal.current_power_w
        if power_w <= 0.0:
            return
        mid = self.mid_rank
        self.rank_coeff = (power_w / tps_estimate) / (mid if mid > 0.0 else 1.0)

    def gradient(self, probs: np.ndarray, rank_soft: np.ndarray) -> np.ndarray:
        """Policy-gradient energy term of tpj_gradient.

        dL/dtheta[l][r] += lambda * rank_coeff * p[l][r] * (R_r - rank_soft[l])
        """
        probs = np.asarray(probs, dtype=np.float64)
        rank_soft = np.asarray(rank_soft, dtype=np.float64)
        if self.rank_coeff <= 0.0 or self.lambda_ <= 0.0:
            return np.zeros_like(probs)
        lam_k = self.lambda_ * self.rank_coeff
        advantage = self.rank_levels[None, :] - rank_soft[:, None]
        return lam_k * probs * advantage
