"""Planner dynamic program: maximise the sum of fleet returns on the catch grid."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from env import ration
from kernels import planner_sweep_fishery, planner_sweep_levhari
from utils import gauss_hermite_normal, stock_catch_grids


@dataclass
class PlannerResult:
    model: str
    r: float
    H: int
    converged: bool
    delta: float
    s_grid: np.ndarray
    a_grid: np.ndarray
    V: np.ndarray
    a1_idx: np.ndarray
    a2_idx: np.ndarray
    executed: np.ndarray  # (n_s, 2) expected executed catch (fractions for Levhari-Mirman)

    @property
    def total_executed(self):
        return self.executed.sum(axis=1)

    def value_at_s0(self):
        return float(self.V[-1])


def _iterate(sweep, V0, tol, max_H, min_H, verbose, label):
    V = V0
    delta = np.inf
    bi = np.zeros(V.shape[0], dtype=np.int64)
    bj = np.zeros(V.shape[0], dtype=np.int64)
    H = 0
    for h in range(1, max_H + 1):
        Vn, bi, bj = sweep(V)
        delta = float(np.max(np.abs(Vn - V)))
        V = Vn
        H = h
        if verbose and (h == 1 or h % 25 == 0 or delta < tol):
            print(f"  planner[{label}] H={h} sup|dV|={delta:.3e}", flush=True)
        if delta < tol and h >= min_H:
            break
    return V, bi, bj, H, delta < tol, delta


def solve_planner(cfg, r, verbose=False, n_stock=None, n_catch=None, n_quad=None, model="fishery"):
    """Value iteration from V=0. `H` is the iteration at which the sup-norm residual stops."""
    if model == "fishery":
        s_grid, a_grid = stock_catch_grids(cfg, n_stock=n_stock, n_catch=n_catch)
        m = cfg["model"]
        sol = cfg["solver"]
        nq = int(cfg["grids"]["n_quad"] if n_quad is None else n_quad)
        nodes, weights = gauss_hermite_normal(nq)
        K = float(m["K"])
        sigma = float(m["sigma"])
        gamma = float(m["gamma"])
        p = float(m["p"])
        c = float(m["c"])
        rr = float(r)

        def sweep(V):
            return planner_sweep_fishery(V, s_grid, a_grid, rr, K, sigma, gamma, p, c, nodes, weights)

        V, bi, bj, H, ok, delta = _iterate(
            sweep, np.zeros(s_grid.shape[0]), float(sol["value_tol"]), int(sol["max_H"]), int(sol["min_H"]), verbose, f"r={rr}"
        )
        executed = np.zeros((s_grid.shape[0], 2), dtype=np.float64)
        for si, s in enumerate(s_grid):
            executed[si] = ration(np.array([a_grid[bi[si]], a_grid[bj[si]]]), float(s))
        return PlannerResult("fishery", rr, H, ok, delta, s_grid, a_grid, V, bi, bj, executed)

    if model != "levhari":
        raise ValueError(f"unknown model {model}")

    L = cfg["levhari_mirman"]
    n_s = int(L["n_stock"] if n_stock is None else n_stock)
    n_a = int(L["n_fraction"] if n_catch is None else n_catch)
    s_max = float(L["s_max"])
    # Positive stocks only. s=0 makes log utility undefined and a sentinel there
    # leaks through linear interpolation. Escapement below the first node is clamped.
    s_grid = np.linspace(s_max / n_s, s_max, n_s, dtype=np.float64)
    a_grid = np.linspace(0.0, 1.0, n_a, dtype=np.float64)
    alpha = float(L["alpha"])
    gamma = float(L["gamma"])

    def sweep(V):
        return planner_sweep_levhari(V, s_grid, a_grid, alpha, s_max, gamma)

    V, bi, bj, H, ok, delta = _iterate(
        sweep,
        np.zeros(n_s),
        float(L["value_tol"]),
        int(L["max_H"]),
        2,
        verbose,
        f"alpha={alpha}",
    )
    executed = np.column_stack([a_grid[bi] * s_grid, a_grid[bj] * s_grid])
    return PlannerResult("levhari", alpha, H, ok, delta, s_grid, a_grid, V, bi, bj, executed)
