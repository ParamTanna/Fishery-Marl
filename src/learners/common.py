"""Shared indexing for tabular learners. The stock is snapped to the benchmark grid."""

from __future__ import annotations

import numpy as np


def stock_index(stock, s_grid):
    K = float(s_grid[-1])
    n_s = s_grid.shape[0]
    if K <= 0.0:
        return 0
    raw = int(np.rint(float(stock) / K * (n_s - 1)))
    return int(np.clip(raw, 0, n_s - 1))


def one_hot_greedy(q_row_argmax, n_s, n_a):
    table = np.zeros((n_s, n_a), dtype=np.float64)
    table[np.arange(n_s), q_row_argmax] = 1.0
    return table


def nearest_catch(request, a_grid):
    return int(np.abs(np.asarray(a_grid, dtype=np.float64) - float(request)).argmin())
