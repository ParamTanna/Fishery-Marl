"""Gauss-Hermite weights and the censored expectation of next stock."""

import numpy as np

from kernels import expected_V
from utils import gauss_hermite_normal


def test_weights_sum_to_one_and_mean_matches_standard_normal():
    nodes, weights = gauss_hermite_normal(21)
    assert abs(float(weights.sum()) - 1.0) < 1.0e-12
    # E[Z] = 0, E[Z^2] = 1 for Z ~ N(0, 1).
    assert abs(float(weights @ nodes)) < 1.0e-12
    assert abs(float(weights @ nodes**2) - 1.0) < 1.0e-10


def test_expected_stock_matches_monte_carlo():
    K = 100.0
    r = 0.8
    sigma = 0.1
    x = 45.0
    n_s = 201
    s_grid = np.linspace(0.0, K, n_s)
    # V(s) = s, so the kernel returns E[min(K, omega * gr(x))].
    V = s_grid.copy()
    nodes, weights = gauss_hermite_normal(31)
    estimate = float(expected_V(V, x, r, K, sigma, nodes, weights))
    rng = np.random.default_rng(0)
    xi = rng.normal(size=400_000)
    gr = x * (1.0 + r * (1.0 - x / K))
    omega = np.exp(sigma * xi - 0.5 * sigma * sigma)
    mc = float(np.mean(np.minimum(K, omega * gr)))
    assert abs(estimate - mc) < 0.05, (estimate, mc)
