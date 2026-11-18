"""Exact evaluation of stationary profiles, written directly from the model equations.

This module is deliberately independent of env.py, dp_planner.py and equilibrium.py. It is used to
certify equilibria (exact best responses at every stock), to compute the Price of Anarchy and Stability,
the grim-trigger condition and the capped (safe) game, and to cross-check the main solver.

Model: rationing a_hat_i = a_i * min(1, s / sum_j a_j); reward p * a_hat - c/2 * a_hat^2;
escapement x = s - sum_i a_hat_i; s' = min(K, omega * g_r(x)), g_r(x) = x + r x (1 - x/K),
omega = exp(sigma * xi - sigma^2 / 2). Values live on the stock grid; off-grid s' is interpolated linearly.
E[V(s')] uses Gauss-Hermite quadrature, with the probability mass above K placed on s' = K
analytically ("team" scheme) or by clipping each node ("plain" scheme).
"""
from math import erfc, sqrt

import numpy as np

K, P, C, SIGMA = 100.0, 10.0, 10.0 / 15.0, 0.1
GAMMA = 0.99          # module-level discount; callers may set exact_check.GAMMA


def grids(n_s=101, n_a=51):
    return np.linspace(0, K, n_s), np.linspace(0, K, n_a)


def gauss_hermite(n=21):
    x, w = np.polynomial.hermite.hermgauss(n)
    return np.sqrt(2) * x, w / np.sqrt(np.pi)


def interp_weights(sp, n_s):
    """Linear-interpolation weights of point(s) sp on the uniform grid over [0, K]."""
    pos = np.clip(sp, 0, K) / K * (n_s - 1)
    lo = np.minimum(np.floor(pos).astype(np.int64), n_s - 2)
    t = pos - lo
    return lo, 1 - t, t


def transition_rows(x, r, n_s, scheme="team", sigma=SIGMA, nq=21):
    """Dense rows T[k, :]: distribution over stock nodes of s' given escapement x[k]."""
    x = np.asarray(x, float)
    T = np.zeros((x.size, n_s))
    nodes, wts = gauss_hermite(nq)
    g = x * (1 + r * (1 - x / K))
    dead = (x <= 0) | (g <= 0)
    T[dead, 0] = 1.0
    live = np.flatnonzero(~dead)
    gl = g[live]
    om = np.exp(sigma * nodes - 0.5 * sigma ** 2)
    sp = gl[:, None] * om[None, :]
    if scheme == "plain":
        wq = np.broadcast_to(wts, sp.shape)
    else:
        z = (np.log(K / gl) + 0.5 * sigma ** 2) / sigma
        pc = 0.5 * np.array([erfc(v / sqrt(2)) for v in z])          # P(s' >= K)
        below = sp < K
        uw = (below * wts).sum(1)
        ok = (uw > 1e-15) & (pc < 1)
        wq = np.where(below, wts[None, :], 0.0) * np.where(ok, (1 - pc) / np.where(uw > 0, uw, 1), 0)[:, None]
        T[live, n_s - 1] += np.where(ok, pc, 1.0)
    lo, wl, wh = interp_weights(np.minimum(sp, K), n_s)
    rows = np.repeat(live[:, None], sp.shape[1], 1)
    np.add.at(T, (rows, lo), wq * wl)
    np.add.at(T, (rows, lo + 1), wq * wh)
    return T


def stage_tables(r, s_grid, a_grid, scheme="team"):
    """Rewards R1, R2 (S, A, A) and transitions T (S, A, A, S) for every stock and catch pair."""
    S, A = s_grid.size, a_grid.size
    s = s_grid[:, None, None]; a1 = a_grid[None, :, None]; a2 = a_grid[None, None, :]
    tot = a1 + a2
    scale = np.where(tot > 0, np.minimum(1.0, s / np.where(tot > 0, tot, 1)), 0.0)
    scale = np.where(s <= 0, 0.0, scale)
    h1, h2 = a1 * scale, a2 * scale
    R1 = P * h1 - 0.5 * C * h1 ** 2
    R2 = P * h2 - 0.5 * C * h2 ** 2
    T = transition_rows((s - h1 - h2).reshape(-1), r, S, scheme)
    return R1, R2, T.reshape(S, A, A, S)


def _howard(R, Tf, mask=None):
    """Howard policy iteration for max_a R[s, a] + GAMMA * Tf[s, a, :] @ V."""
    S = R.shape[0]
    if mask is not None:
        R = np.where(mask, R, -1e18)
    V = np.zeros(S); pol = None
    for _ in range(300):
        new = (R + GAMMA * Tf @ V).argmax(1)
        if pol is not None and np.array_equal(new, pol):
            break
        pol = new
        V = np.linalg.solve(np.eye(S) - GAMMA * Tf[np.arange(S), pol], R[np.arange(S), pol])
    return V, pol


def planner(R1, R2, T, joint_mask=None, symmetric=False):
    """Planner over joint catches (or symmetric catches a1 = a2). Returns V and the policy index."""
    S, A = T.shape[0], T.shape[1]
    if symmetric:
        idx = np.arange(A)
        return _howard((R1 + R2)[:, idx, idx], T[:, idx, idx, :])
    jm = None if joint_mask is None else joint_mask.reshape(S, A * A)
    return _howard((R1 + R2).reshape(S, A * A), T.reshape(S, A * A, S), jm)


def profile_values(p1, p2, R1, R2, T):
    """Exact values of the stationary (mixed) profile p1[s, a1], p2[s, a2]."""
    S = T.shape[0]
    w = p1[:, :, None] * p2[:, None, :]
    r1 = (w * R1).sum((1, 2)); r2 = (w * R2).sum((1, 2))
    M = np.eye(S) - GAMMA * np.einsum("sab,sabt->st", w, T)
    return np.linalg.solve(M, r1), np.linalg.solve(M, r2)


def best_response_1(p2, R1, T, mask=None):
    """Fleet 1's exact best response to the fixed profile p2 (optionally restricted to legal catches)."""
    Rb = np.einsum("sab,sb->sa", R1, p2)
    Tb = np.einsum("sabt,sb->sat", T, p2)
    return _howard(Rb, Tb, mask)


def deviation_value_1(p2, R1, T, V_after):
    """max over a1 of R1(s, a1, p2) + GAMMA * E[V_after(s')]: one deviation, then V_after."""
    Q = np.einsum("sab,sb->sa", R1, p2) + GAMMA * (np.einsum("sabt,sb->sat", T, p2) @ V_after)
    return Q.max(1)


def certified(E_sup, J_fleet, abs_tol=1.0, rel_tol=1e-3):
    """Certificate used throughout: max gain <= 1 at every stock, or <= 0.1% of the fleet's value."""
    return bool(E_sup <= abs_tol or E_sup / max(abs(J_fleet), 1e-12) <= rel_tol)


def reserve_mask(s_grid, a_grid, s_res):
    """Legal own catches under a_i <= max(0, 0.5 (s - s_res)); at least the zero catch."""
    cap = np.maximum(0.0, 0.5 * (s_grid - s_res))
    legal = np.array([max(1, int(np.sum(a_grid <= c + 1e-12))) for c in cap])
    return np.arange(a_grid.size)[None, :] < legal[:, None]
