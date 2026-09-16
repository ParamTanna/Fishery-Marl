"""Configuration, quadrature, seeding, and small numerical helpers."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


def deep_update(base, extra):
    out = copy.deepcopy(base)

    def rec(dst, src):
        for key, value in src.items():
            if isinstance(value, dict) and isinstance(dst.get(key), dict):
                rec(dst[key], value)
            else:
                dst[key] = copy.deepcopy(value)

    rec(out, extra)
    return out


def load_config(path=None, quick=False):
    path = Path(path) if path is not None else ROOT / "config.yaml"
    with open(path, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if quick:
        overlay = copy.deepcopy(cfg.get("quick", {}))
        cfg = deep_update(cfg, overlay)
    return cfg


def config_hash(cfg):
    payload = copy.deepcopy(cfg)
    payload.pop("quick", None)
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def git_commit():
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "nogit"


_GH_CACHE = {}


def gauss_hermite_normal(n):
    """Nodes and weights for E[g(Z)], Z ~ N(0, 1)."""
    n_int = int(n)
    cached = _GH_CACHE.get(n_int)
    if cached is not None:
        return cached
    x, w = np.polynomial.hermite.hermgauss(n_int)
    nodes = np.ascontiguousarray(np.sqrt(2.0) * x, dtype=np.float64)
    weights = np.ascontiguousarray(w / np.sqrt(np.pi), dtype=np.float64)
    _GH_CACHE[n_int] = (nodes, weights)
    return nodes, weights


def horizon_for_tail(gamma, tail):
    """Smallest L with gamma^L < tail. gamma in (0, 1), tail in (0, 1)."""
    if not 0.0 < gamma < 1.0:
        raise ValueError("gamma must lie in (0, 1)")
    if not 0.0 < tail < 1.0:
        raise ValueError("tail must lie in (0, 1)")
    return int(math.ceil(math.log(tail) / math.log(gamma)))


def tragedy_margin(cfg, r_values=None):
    """Return rows (r, n*p/c, r*K/4, slack) and raise if any r has no tragedy."""
    model = cfg["model"]
    n = int(model["n_fleets"])
    demand = n * float(model["p"]) / float(model["c"])
    K = float(model["K"])
    if r_values is None:
        r_values = list(model["r_values"])
    rows = []
    for r in r_values:
        msy = float(r) * K / 4.0
        slack = demand - msy
        rows.append(
            {
                "r": float(r),
                "n_times_p_over_c": demand,
                "rK_over_4": msy,
                "slack": slack,
            }
        )
        if slack <= 0.0:
            raise AssertionError(
                f"No tragedy at r={r}: |N|*p/c={demand:.6g} <= rK/4={msy:.6g}. "
                "Adjust p/c or r on the known-r benchmark and record the change."
            )
    return rows


def stock_catch_grids(cfg, n_stock=None, n_catch=None):
    model = cfg["model"]
    grids = cfg["grids"]
    K = float(model["K"])
    n_s = int(grids["n_stock"] if n_stock is None else n_stock)
    n_a = int(grids["n_catch"] if n_catch is None else n_catch)
    s_grid = np.linspace(0.0, K, n_s, dtype=np.float64)
    a_grid = np.linspace(float(grids["catch_min"]), float(grids["catch_max"]), n_a, dtype=np.float64)
    return s_grid, a_grid


def t_interval(samples, level=0.95):
    """Mean and half-width of a Student t interval across seeds (one sample per seed)."""
    x = np.asarray(samples, dtype=np.float64)
    n = x.size
    mean = float(np.mean(x)) if n else float("nan")
    if n < 2:
        return mean, float("nan")
    # scipy is available; keep the import local so utils stays light for numba-only scripts
    from scipy import stats

    se = float(np.std(x, ddof=1) / math.sqrt(n))
    half = float(stats.t.ppf(0.5 + level / 2.0, n - 1) * se)
    return mean, half


def machine_summary():
    import os
    import platform

    ram_gb = None
    try:
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        ram_gb = round(stat.ullTotalPhys / (1024**3), 2)
    except (AttributeError, OSError):
        ram_gb = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "logical_cpus": os.cpu_count(),
        "ram_gb": ram_gb,
        "gpu": "none detected (nvidia-smi unavailable)",
    }
