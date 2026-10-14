"""Belief over the unknown regrowth rate, with the censored lognormal likelihood.

The transition is s' = min(K, omega * gr(x)), omega = exp(sigma * xi - sigma^2 / 2),
xi ~ N(0, 1). For s' < K the likelihood is the lognormal density of s'. The factor
1/s' does not depend on r, so it cancels in the normalised posterior; it is still
included so the density integrates to one against the censoring probability.
At s' = K the likelihood is the analytic censoring probability
1 - Phi((ln(K / gr(x)) + sigma^2 / 2) / sigma).
Updates are skipped when x = 0, which is uninformative.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def r_grid(r_min, r_max, n):
    return np.linspace(float(r_min), float(r_max), int(n), dtype=np.float64)


def growth(x, r, K):
    return x * (1.0 + r * (1.0 - x / K))


def censored_probability(gr, K, sigma):
    """P(omega * gr >= K). gr, K, sigma may be scalars. gr <= 0 returns 0."""
    gr = np.asarray(gr, dtype=np.float64)
    scalar = gr.ndim == 0
    gr = np.atleast_1d(gr)
    out = np.zeros(gr.shape, dtype=np.float64)
    positive = gr > 0.0
    if np.any(positive):
        z = (np.log(K / gr[positive]) + 0.5 * sigma * sigma) / sigma
        out[positive] = norm.sf(z)
    out = np.clip(out, 0.0, 1.0)
    return float(out[0]) if scalar else out


def log_likelihood(r, x, s_next, sigma, K):
    """Log density of one transition. Shape matches `r`. -inf where the transition is impossible."""
    r = np.asarray(r, dtype=np.float64)
    gr = growth(float(x), r, K)
    ll = np.full(r.shape, -np.inf, dtype=np.float64)
    feasible = gr > 0.0
    if s_next >= K * (1.0 - 1.0e-12):
        z = np.empty(r.shape, dtype=np.float64)
        z[feasible] = (np.log(K / gr[feasible]) + 0.5 * sigma * sigma) / sigma
        ll[feasible] = norm.logsf(z[feasible])
        return ll
    if s_next <= 0.0:
        return ll
    mu = np.empty(r.shape, dtype=np.float64)
    mu[feasible] = np.log(gr[feasible]) - 0.5 * sigma * sigma
    ll[feasible] = norm.logpdf(np.log(s_next), loc=mu[feasible], scale=sigma) - np.log(s_next)
    return ll


def density_of_s_prime(s_prime, gr, sigma):
    """Uncensored lognormal density of omega * gr (the draw before the cap at K)."""
    s_prime = np.asarray(s_prime, dtype=np.float64)
    mu = np.log(gr) - 0.5 * sigma * sigma
    return np.exp(norm.logpdf(np.log(s_prime), loc=mu, scale=sigma) - np.log(s_prime))


class Belief:
    """Discrete belief on a fixed grid. Probabilities live in log space."""

    def __init__(self, grid, sigma, K, prior=None):
        self.grid = np.asarray(grid, dtype=np.float64)
        self.sigma = float(sigma)
        self.K = float(K)
        self.n_updates = 0
        self.n_skipped = 0
        if prior is None:
            self.log_b = np.zeros(self.grid.shape[0], dtype=np.float64)
        else:
            prior = np.asarray(prior, dtype=np.float64)
            if prior.shape != self.grid.shape:
                raise ValueError("prior shape must match the r grid")
            if np.any(prior < 0.0) or prior.sum() <= 0.0:
                raise ValueError("prior must be non-negative and not identically zero")
            with np.errstate(divide="ignore"):
                self.log_b = np.log(prior / prior.sum())

    def probs(self):
        m = np.max(self.log_b)
        b = np.exp(self.log_b - m)
        return b / b.sum()

    def mean(self):
        b = self.probs()
        return float(b @ self.grid)

    def variance(self):
        b = self.probs()
        mu = float(b @ self.grid)
        return float(b @ (self.grid - mu) ** 2)

    def update(self, x, s_next):
        """Bayes update from escapement x and next stock. Returns True if the belief changed."""
        if x <= 0.0:
            self.n_skipped += 1
            return False
        ll = log_likelihood(self.grid, x, s_next, self.sigma, self.K)
        if not np.any(np.isfinite(ll)):
            self.n_skipped += 1
            return False
        self.log_b = self.log_b + ll
        # Renormalise in probability space so the log-belief stays on a finite scale.
        m = np.max(self.log_b)
        b = np.exp(self.log_b - m)
        total = b.sum()
        b /= total
        with np.errstate(divide="ignore"):
            self.log_b = np.log(b)
        self.n_updates += 1
        return True

    def copy(self):
        other = Belief(self.grid, self.sigma, self.K, prior=self.probs())
        other.n_updates = self.n_updates
        other.n_skipped = self.n_skipped
        return other
