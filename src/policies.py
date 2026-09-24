"""Fixed policies used as cross-play opponents. Each returns a distribution on the catch grid."""

from __future__ import annotations

import numpy as np


def _one_hot(index, n_actions):
    policy = np.zeros((index.shape[0], n_actions), dtype=np.float64)
    policy[np.arange(index.shape[0]), np.asarray(index, dtype=np.int64)] = 1.0
    return policy


def nearest_action(values, a_grid):
    values = np.asarray(values, dtype=np.float64)
    grid = np.asarray(a_grid, dtype=np.float64)
    return np.abs(values.reshape(-1, 1) - grid.reshape(1, -1)).argmin(axis=1)


def overharvester(s_grid, a_grid, request):
    """Every stock requests the same catch, the myopic optimum p/c snapped to the grid."""
    index = np.full(len(s_grid), int(nearest_action([request], a_grid)[0]), dtype=np.int64)
    table = _one_hot(index, len(a_grid))
    return table, table.copy()


def planner_share(plan):
    """Each fleet requests half of the planner's total executed catch at that stock."""
    half = np.asarray(plan.total_executed, dtype=np.float64) / 2.0
    index = nearest_action(half, plan.a_grid)
    table = _one_hot(index, plan.a_grid.shape[0])
    return table, table.copy()


def equilibrium_pair(eq):
    return np.array(eq.p1, dtype=np.float64, copy=True), np.array(eq.p2, dtype=np.float64, copy=True)
