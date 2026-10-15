"""Belief updates: censoring identity, skip at x=0, concentration in r and in sigma."""

import numpy as np

from belief import Belief, censored_probability, density_of_s_prime, r_grid
from env import next_stock


def test_censored_likelihood_integrates_to_one():
    sigma = 0.1
    K = 100.0
    for gr in (20.0, 60.0, 95.0, 130.0):
        p_cens = censored_probability(gr, K, sigma)
        lo = 1.0e-6
        s = np.linspace(lo, K, 400_000)
        ds = s[1] - s[0]
        mass = float(np.sum(density_of_s_prime(s, gr, sigma)) * ds)
        assert abs(mass + p_cens - 1.0) < 2.0e-3, (gr, mass, p_cens)


def test_update_skips_zero_escapement_and_renormalises():
    grid = r_grid(0.1, 2.0, 191)
    belief = Belief(grid, sigma=0.1, K=100.0)
    changed = belief.update(0.0, 40.0)
    assert changed is False
    assert belief.n_skipped == 1
    assert belief.n_updates == 0
    assert np.allclose(belief.probs().sum(), 1.0)
    assert np.allclose(belief.probs(), np.full(grid.shape, 1.0 / grid.size))


def test_posterior_concentrates_and_noise_slows_it():
    grid = r_grid(0.1, 2.0, 191)
    true_r = 1.0
    K = 100.0
    x = 40.0
    xi = np.random.default_rng(11).normal(size=80)

    def run(sigma, n_obs):
        belief = Belief(grid, sigma=sigma, K=K)
        for z in xi[:n_obs]:
            sp = float(next_stock(x, true_r, K, sigma, z))
            belief.update(x, sp)
        return belief

    short = run(0.1, 8)
    long = run(0.1, 80)
    assert long.variance() < short.variance()
    assert abs(long.mean() - true_r) < 0.15
    assert run(0.3, 80).variance() > run(0.05, 80).variance()
