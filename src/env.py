"""Fishery environment: proportional rationing and logistic regrowth with lognormal noise."""

from __future__ import annotations

import math

import numpy as np


def logistic_growth(x, r, K):
    x = np.asarray(x, dtype=np.float64)
    return x * (1.0 + r * (1.0 - x / K))


def ration(requests, stock):
    """Executed catches. Sum of outputs is at most `stock`, and own catch is monotone in own request.

    requests: (..., n_fleets)
    stock: broadcastable to requests.shape[:-1]
    """
    req = np.asarray(requests, dtype=np.float64)
    scalar_stock = np.isscalar(stock) or (isinstance(stock, np.ndarray) and stock.ndim == 0)
    stock_arr = np.asarray(stock, dtype=np.float64)
    single = req.ndim == 1
    if single:
        req = req.reshape(1, -1)
        stock_arr = np.reshape(stock_arr, (1,))
    total = req.sum(axis=-1)
    scale = np.zeros(total.shape, dtype=np.float64)
    np.divide(stock_arr, total, out=scale, where=total > 0.0)
    np.minimum(scale, 1.0, out=scale)
    # A zero stock yields zero catch even if the ratio above was formed.
    scale = np.where(stock_arr <= 0.0, 0.0, scale)
    out = req * scale[..., None]
    if single and scalar_stock:
        return out[0]
    if single:
        return out[0]
    return out


def omega_from_xi(xi, sigma):
    return np.exp(sigma * xi - 0.5 * sigma * sigma)


def next_stock(x, r, K, sigma, xi):
    """s' = min(K, omega * gr(x)). xi is standard normal (or 0 when sigma is 0)."""
    gr = logistic_growth(x, r, K)
    if sigma <= 0.0:
        omega = 1.0
    else:
        omega = omega_from_xi(xi, sigma)
    sp = omega * gr
    return np.minimum(K, np.maximum(sp, 0.0))


class FisheryEnv:
    """One lake, n identical fleets. Gym-like but local: reset, step.

    Observation is the scalar stock. `step` returns
    (stock, rewards, terminated, info) with executed catches in info.
    The episode does not terminate; callers truncate at T and bootstrap.
    """

    def __init__(self, K, p, c, r, sigma, gamma, n_fleets=2, seed=0, s0=None):
        self.K = float(K)
        self.p = float(p)
        self.c = float(c)
        self.r = float(r)
        self.sigma = float(sigma)
        self.gamma = float(gamma)
        self.n_fleets = int(n_fleets)
        self.s0 = self.K if s0 is None else float(s0)
        self.seed = int(seed)
        self.rng = np.random.Generator(np.random.PCG64(self.seed))
        self.s = None

    def reset(self, s0=None):
        self.s = self.s0 if s0 is None else float(s0)
        return self.s

    def step(self, requests):
        req = np.asarray(requests, dtype=np.float64).reshape(self.n_fleets)
        hats = ration(req, self.s)
        rewards = self.p * hats - 0.5 * self.c * hats * hats
        x = float(self.s - np.sum(hats))
        if self.sigma <= 0.0:
            xi = 0.0
        else:
            xi = float(self.rng.normal())
        sp = float(next_stock(x, self.r, self.K, self.sigma, xi))
        info = {"x": x, "executed": hats.copy(), "xi": xi, "r": self.r}
        self.s = sp
        return sp, rewards, False, info


def mean_path(s_grid, a_grid, p1, p2, r, K, s0, horizon, model="fishery", alpha=None):
    """Deterministic skeleton with omega = 1, requests interpolated in the stock.

    For a mixed grid policy the request is the mean of the mixture. Executed catch
    then uses the rationing rule (fishery) or the feasible LM split.
    """
    s_grid = np.asarray(s_grid, dtype=np.float64)
    a_grid = np.asarray(a_grid, dtype=np.float64)
    mean1 = p1 @ a_grid
    mean2 = p2 @ a_grid
    upper = float(s_grid[-1])
    s = float(s0)
    stocks = np.empty(horizon, dtype=np.float64)
    catch = np.empty((horizon, 2), dtype=np.float64)
    for t in range(horizon):
        stocks[t] = s
        a1 = float(np.interp(s, s_grid, mean1))
        a2 = float(np.interp(s, s_grid, mean2))
        if model == "levhari":
            # fractions of the current stock; infeasible mass is clipped back onto the simplex
            f1 = min(max(a1, 0.0), 1.0)
            f2 = min(max(a2, 0.0), 1.0)
            if f1 + f2 > 1.0:
                scale = 1.0 / (f1 + f2)
                f1 *= scale
                f2 *= scale
            h1 = f1 * s
            h2 = f2 * s
            x = s - h1 - h2
            sp = 0.0 if x <= 0.0 else x ** float(alpha)
            s = min(upper, sp)
        else:
            h1, h2 = ration(np.array([a1, a2]), s)
            x = s - (h1 + h2)
            gr = float(logistic_growth(x, r, K))
            s = min(K, max(gr, 0.0))
        catch[t, 0] = h1
        catch[t, 1] = h2
    return stocks, catch
