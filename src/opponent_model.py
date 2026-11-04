"""Online model of the other fleet's request, as a distribution on the catch grid."""

from __future__ import annotations

import numpy as np


def recover_request(stock, own_request, own_executed, other_executed):
    """Opponent request implied by proportional rationing.

    When this fleet's request was positive and was cut, the requests sum to
    stock * own_request / own_executed. Otherwise the executed catch is the request.
    """
    own_request = float(own_request)
    own_executed = float(own_executed)
    if own_request > 1.0e-12 and own_executed > 1.0e-12 and own_executed < own_request - 1.0e-8:
        total = float(stock) * own_request / own_executed
        return max(0.0, total - own_request)
    return float(other_executed)


class KernelOpponent:
    """Gaussian kernel regression of the opponent's request against stock."""

    def __init__(self, bandwidth, default_request):
        self.bandwidth = float(bandwidth)
        self.default_request = float(default_request)
        self.stocks = []
        self.requests = []
        self._cached_key = None
        self._cached_s = None
        self._sum_w = None
        self._sum_wa = None
        self._sum_wa2 = None

    def update(self, stock, request):
        s_val = float(stock)
        a_val = float(request)
        self.stocks.append(s_val)
        self.requests.append(a_val)
        if self._cached_s is not None:
            width = max(self.bandwidth, 1.0e-6)
            w = np.exp(-0.5 * np.square((self._cached_s - s_val) / width))
            self._sum_w += w
            wa = w * a_val
            self._sum_wa += wa
            self._sum_wa2 += wa * a_val

    def policy(self, s_grid, a_grid):
        s_grid = np.asarray(s_grid, dtype=np.float64)
        a_grid = np.asarray(a_grid, dtype=np.float64)
        n_s = s_grid.shape[0]
        n_a = a_grid.shape[0]
        if len(self.stocks) == 0:
            return _spike(s_grid, a_grid, self.default_request)
        key = (n_s, float(s_grid[0]), float(s_grid[-1]), s_grid.tobytes())
        if self._cached_key != key:
            obs_s = np.asarray(self.stocks, dtype=np.float64)
            obs_a = np.asarray(self.requests, dtype=np.float64)
            width = max(self.bandwidth, 1.0e-6)
            weight = np.exp(-0.5 * ((s_grid[:, None] - obs_s[None, :]) / width) ** 2)
            self._cached_key = key
            self._cached_s = s_grid.copy()
            self._sum_w = weight.sum(axis=1)
            wa = weight * obs_a[None, :]
            self._sum_wa = wa.sum(axis=1)
            self._sum_wa2 = (wa * obs_a[None, :]).sum(axis=1)
        weight_sum = np.maximum(self._sum_w, 1.0e-300)
        mean = self._sum_wa / weight_sum
        var = self._sum_wa2 / weight_sum - np.square(mean)
        step = float(a_grid[1] - a_grid[0]) if n_a > 1 else 1.0
        std = np.maximum(np.sqrt(np.maximum(var, 0.0)), step)
        log_p = -0.5 * ((a_grid[None, :] - mean[:, None]) / std[:, None]) ** 2
        log_p -= log_p.max(axis=1, keepdims=True)
        prob = np.exp(log_p)
        prob /= prob.sum(axis=1, keepdims=True)
        return prob


class FrequencyOpponent:
    """Laplace-smoothed counts on a stock-bin by catch-bin table."""

    def __init__(self, s_grid, a_grid, n_stock_bins, n_catch_bins, alpha, default_request):
        self.s_grid = np.asarray(s_grid, dtype=np.float64)
        self.a_grid = np.asarray(a_grid, dtype=np.float64)
        self.n_stock_bins = int(n_stock_bins)
        self.n_catch_bins = int(n_catch_bins)
        self.alpha = float(alpha)
        self.default_request = float(default_request)
        self.counts = np.full((self.n_stock_bins, self.n_catch_bins), self.alpha, dtype=np.float64)
        self.n_obs = 0

    def _stock_bin(self, stock):
        K = float(self.s_grid[-1])
        if K <= 0.0:
            return 0
        raw = int(np.floor(float(stock) / K * self.n_stock_bins))
        return int(np.clip(raw, 0, self.n_stock_bins - 1))

    def _catch_bin(self, request):
        # Equal-width bins over the catch grid's range.
        lo = float(self.a_grid[0])
        hi = float(self.a_grid[-1])
        if hi <= lo:
            return 0
        raw = int(np.floor((float(request) - lo) / (hi - lo) * self.n_catch_bins))
        return int(np.clip(raw, 0, self.n_catch_bins - 1))

    def update(self, stock, request):
        self.counts[self._stock_bin(stock), self._catch_bin(request)] += 1.0
        self.n_obs += 1

    def policy(self, s_grid, a_grid):
        s_grid = np.asarray(s_grid, dtype=np.float64)
        a_grid = np.asarray(a_grid, dtype=np.float64)
        if self.n_obs == 0:
            return _spike(s_grid, a_grid, self.default_request)
        lo = float(self.a_grid[0])
        hi = float(self.a_grid[-1])
        edges = np.linspace(lo, hi, self.n_catch_bins + 1)
        centres = 0.5 * (edges[:-1] + edges[1:])
        out = np.zeros((s_grid.shape[0], a_grid.shape[0]), dtype=np.float64)
        for s_index, stock in enumerate(s_grid):
            row = self.counts[self._stock_bin(stock)]
            row = row / row.sum()
            # Put each bin's mass on the nearest catch-grid point.
            nearest = np.abs(centres.reshape(-1, 1) - a_grid.reshape(1, -1)).argmin(axis=1)
            for bin_index, action in enumerate(nearest):
                out[s_index, action] += row[bin_index]
            total = out[s_index].sum()
            if total > 0.0:
                out[s_index] /= total
        return out


def _spike(s_grid, a_grid, request):
    n_s = len(s_grid)
    n_a = len(a_grid)
    index = int(np.abs(np.asarray(a_grid) - float(request)).argmin())
    table = np.zeros((n_s, n_a), dtype=np.float64)
    table[:, index] = 1.0
    return table
