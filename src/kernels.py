"""Numba kernels shared by the planner, the equilibrium solver, and best responses.

The next-stock expectation uses Gauss-Hermite nodes for N(0, 1). The cap
s' = min(K, omega * gr(x)) is split exactly: the analytic censoring probability
multiplies V(K), and nodes that fall below the cap are renormalised to
approximate the conditional expectation on {s' < K}.
"""

from __future__ import annotations

import math

import numba
import numpy as np


@numba.njit(cache=True)
def interp_uniform(V, s, upper):
    """Linear interpolation on a uniform grid from 0 to `upper`."""
    return interp_grid(V, s, 0.0, upper)


@numba.njit(cache=True)
def interp_grid(V, s, s_lo, s_hi):
    n = V.shape[0]
    if s <= s_lo:
        return V[0]
    if s >= s_hi:
        return V[n - 1]
    pos = (s - s_lo) / (s_hi - s_lo) * (n - 1)
    i = int(pos)
    if i >= n - 1:
        return V[n - 1]
    t = pos - i
    return V[i] * (1.0 - t) + V[i + 1] * t


@numba.njit(cache=True)
def logistic_gr(x, r, K):
    return x * (1.0 + r * (1.0 - x / K))


@numba.njit(cache=True)
def ration_pair(a1, a2, s):
    total = a1 + a2
    if total <= 0.0 or s <= 0.0:
        return 0.0, 0.0
    if total <= s:
        return a1, a2
    scale = s / total
    return a1 * scale, a2 * scale


@numba.njit(cache=True)
def expected_V(V, x, r, K, sigma, nodes, weights):
    """E[V(s')] for logistic growth and lognormal noise, cap at K."""
    if x <= 0.0:
        return interp_uniform(V, 0.0, K)
    gr = logistic_gr(x, r, K)
    if gr <= 0.0:
        return interp_uniform(V, 0.0, K)
    if sigma <= 0.0:
        sp = gr if gr < K else K
        return interp_uniform(V, sp, K)
    z = (math.log(K / gr) + 0.5 * sigma * sigma) / sigma
    p_cens = 0.5 * math.erfc(z / math.sqrt(2.0))
    half_sig2 = 0.5 * sigma * sigma
    unc_acc = 0.0
    unc_w = 0.0
    for q in range(nodes.shape[0]):
        sp = gr * math.exp(sigma * nodes[q] - half_sig2)
        if sp < K:
            w = weights[q]
            unc_acc += w * interp_uniform(V, sp, K)
            unc_w += w
    v_cap = interp_uniform(V, K, K)
    if unc_w > 1e-15 and p_cens < 1.0:
        e_unc = unc_acc / unc_w
    else:
        e_unc = v_cap
    return p_cens * v_cap + (1.0 - p_cens) * e_unc


@numba.njit(parallel=True, cache=True)
def build_U_fishery_sym(V, s_grid, a_grid, r, K, sigma, gamma, p, c, nodes, weights):
    """Player-1 payoffs of a symmetric fishery stage game. Player 2 is the transpose."""
    n_s = s_grid.shape[0]
    n_a = a_grid.shape[0]
    U = np.empty((n_s, n_a, n_a))
    half_c = 0.5 * c
    for si in numba.prange(n_s):
        s = s_grid[si]
        for i in range(n_a):
            ai = a_grid[i]
            for j in range(n_a):
                h1, h2 = ration_pair(ai, a_grid[j], s)
                x = s - (h1 + h2)
                ev = expected_V(V, x, r, K, sigma, nodes, weights)
                U[si, i, j] = p * h1 - half_c * h1 * h1 + gamma * ev
    return U


@numba.njit(parallel=True, cache=True)
def build_U_fishery_asym(V1, V2, s_grid, a_grid, r, K, sigma, gamma, p, c, nodes, weights):
    n_s = s_grid.shape[0]
    n_a = a_grid.shape[0]
    U1 = np.empty((n_s, n_a, n_a))
    U2 = np.empty((n_s, n_a, n_a))
    half_c = 0.5 * c
    for si in numba.prange(n_s):
        s = s_grid[si]
        for i in range(n_a):
            ai = a_grid[i]
            for j in range(n_a):
                h1, h2 = ration_pair(ai, a_grid[j], s)
                x = s - (h1 + h2)
                ev1 = expected_V(V1, x, r, K, sigma, nodes, weights)
                ev2 = expected_V(V2, x, r, K, sigma, nodes, weights)
                U1[si, i, j] = p * h1 - half_c * h1 * h1 + gamma * ev1
                U2[si, i, j] = p * h2 - half_c * h2 * h2 + gamma * ev2
    return U1, U2


@numba.njit(parallel=True, cache=True)
def planner_sweep_fishery(V, s_grid, a_grid, r, K, sigma, gamma, p, c, nodes, weights):
    """One Bellman sweep for the total-return planner. Tie-break: lowest (i, j)."""
    n_s = s_grid.shape[0]
    n_a = a_grid.shape[0]
    Vn = np.empty(n_s)
    bi = np.empty(n_s, dtype=np.int64)
    bj = np.empty(n_s, dtype=np.int64)
    half_c = 0.5 * c
    for si in numba.prange(n_s):
        s = s_grid[si]
        best = -1.0e300
        ib = 0
        jb = 0
        for i in range(n_a):
            ai = a_grid[i]
            for j in range(n_a):
                h1, h2 = ration_pair(ai, a_grid[j], s)
                x = s - (h1 + h2)
                ev = expected_V(V, x, r, K, sigma, nodes, weights)
                val = p * (h1 + h2) - half_c * (h1 * h1 + h2 * h2) + gamma * ev
                if val > best:
                    best = val
                    ib = i
                    jb = j
        Vn[si] = best
        bi[si] = ib
        bj[si] = jb
    return Vn, bi, bj


@numba.njit(cache=True)
def _safe_log(x):
    if x <= 1.0e-300:
        return -690.0
    return math.log(x)


@numba.njit(parallel=True, cache=True)
def build_U_levhari_sym(V, s_grid, frac_grid, alpha, s_max, gamma):
    """Levhari-Mirman stage payoffs. Actions are harvest fractions of current stock.

    A profile with fractions summing to more than 1 is infeasible (payoff sentinel),
    which is the constraint in Levhari and Mirman, not proportional rationing.
    """
    n_s = s_grid.shape[0]
    n_a = frac_grid.shape[0]
    U = np.empty((n_s, n_a, n_a))
    sentinel = -1.0e6
    s_lo = s_grid[0]
    s_hi = s_grid[-1]
    for si in numba.prange(n_s):
        s = s_grid[si]
        for i in range(n_a):
            for j in range(n_a):
                f1 = frac_grid[i]
                f2 = frac_grid[j]
                if f1 + f2 > 1.0 + 1.0e-12:
                    U[si, i, j] = sentinel
                    continue
                c1 = f1 * s
                x = s * (1.0 - f1 - f2)
                # Zero catch is feasible but terrible. It must stay strictly above the
                # infeasible sentinel, otherwise a flat -inf plateau invents fake equilibria.
                reward = _safe_log(c1) if c1 > 0.0 else -690.0
                if x <= 0.0:
                    sp = s_lo
                else:
                    sp = x ** alpha
                U[si, i, j] = reward + gamma * interp_grid(V, sp, s_lo, s_hi)
    return U


@numba.njit(parallel=True, cache=True)
def planner_sweep_levhari(V, s_grid, frac_grid, alpha, s_max, gamma):
    """Planner maximises the sum of log utilities. Tie-break: lowest (i, j)."""
    n_s = s_grid.shape[0]
    n_a = frac_grid.shape[0]
    Vn = np.empty(n_s)
    bi = np.empty(n_s, dtype=np.int64)
    bj = np.empty(n_s, dtype=np.int64)
    s_lo = s_grid[0]
    s_hi = s_grid[-1]
    for si in numba.prange(n_s):
        s = s_grid[si]
        best = -1.0e300
        ib = 0
        jb = 0
        for i in range(n_a):
            f1 = frac_grid[i]
            for j in range(n_a):
                f2 = frac_grid[j]
                if f1 + f2 > 1.0 + 1.0e-12:
                    continue
                c1 = f1 * s
                c2 = f2 * s
                if c1 <= 0.0 or c2 <= 0.0:
                    continue
                x = s * (1.0 - f1 - f2)
                sp = s_lo if x <= 0.0 else x ** alpha
                val = _safe_log(c1) + _safe_log(c2) + gamma * interp_grid(V, sp, s_lo, s_hi)
                if val > best:
                    best = val
                    ib = i
                    jb = j
        Vn[si] = best
        bi[si] = ib
        bj[si] = jb
    return Vn, bi, bj


@numba.njit(parallel=True, cache=True)
def enum_sym_mixed(U, tol, max_store):
    """Symmetric 2-support mixed equilibria of a symmetric bimatrix, per state.

    U[s, own, opp] is player 1's payoff. Returns supports (i, j), probability of i,
    equilibrium payoff to one player, counts, and overflow counts.
    """
    n_s = U.shape[0]
    n_a = U.shape[1]
    out_i = np.empty((n_s, max_store), dtype=np.int16)
    out_j = np.empty((n_s, max_store), dtype=np.int16)
    out_p = np.empty((n_s, max_store))
    out_u = np.empty((n_s, max_store))
    out_n = np.zeros(n_s, dtype=np.int16)
    overflow = np.zeros(n_s, dtype=np.int16)
    for s in numba.prange(n_s):
        nfound = 0
        extra = 0
        for i in range(n_a):
            Uii = U[s, i, i]
            for j in range(i + 1, n_a):
                Uij = U[s, i, j]
                Uji = U[s, j, i]
                Ujj = U[s, j, j]
                denom = Uii - Uij - Uji + Ujj
                numer = Ujj - Uij
                if abs(denom) < 1.0e-14:
                    continue
                p = numer / denom
                if p <= 1.0e-9 or p >= 1.0 - 1.0e-9:
                    continue
                eu = p * Uii + (1.0 - p) * Uij
                ok = True
                for a in range(n_a):
                    eua = p * U[s, a, i] + (1.0 - p) * U[s, a, j]
                    if eua > eu + tol:
                        ok = False
                        break
                if not ok:
                    continue
                if nfound < max_store:
                    out_i[s, nfound] = i
                    out_j[s, nfound] = j
                    out_p[s, nfound] = p
                    out_u[s, nfound] = eu
                    nfound += 1
                else:
                    extra += 1
        out_n[s] = nfound
        overflow[s] = extra
    return out_i, out_j, out_p, out_u, out_n, overflow


@numba.njit(parallel=True, cache=True)
def policy_value_fishery(V1, V2, p1, p2, s_grid, a_grid, r, K, sigma, gamma, price, c, nodes, weights):
    """Expected payoff of a (possibly mixed) profile. Policies are held fixed.

    One sweep of V |-> (E[u1], E[u2]). This map is a contraction of modulus gamma,
    unlike the equilibrium-selection operator.
    """
    n_s = s_grid.shape[0]
    n_a = a_grid.shape[0]
    out1 = np.empty(n_s)
    out2 = np.empty(n_s)
    half_c = 0.5 * c
    for si in numba.prange(n_s):
        s = s_grid[si]
        acc1 = 0.0
        acc2 = 0.0
        for i in range(n_a):
            wi = p1[si, i]
            if wi <= 0.0:
                continue
            ai = a_grid[i]
            for j in range(n_a):
                w = wi * p2[si, j]
                if w <= 0.0:
                    continue
                h1, h2 = ration_pair(ai, a_grid[j], s)
                x = s - (h1 + h2)
                ev1 = expected_V(V1, x, r, K, sigma, nodes, weights)
                ev2 = expected_V(V2, x, r, K, sigma, nodes, weights)
                acc1 += w * (price * h1 - half_c * h1 * h1 + gamma * ev1)
                acc2 += w * (price * h2 - half_c * h2 * h2 + gamma * ev2)
        out1[si] = acc1
        out2[si] = acc2
    return out1, out2


@numba.njit(parallel=True, cache=True)
def br_sweep_fishery(V, opp, s_grid, a_grid, r, K, sigma, gamma, p, c, nodes, weights):
    """Best-response Bellman sweep. opp[s, j] is the opponent's mixture at stock s."""
    n_s = s_grid.shape[0]
    n_a = a_grid.shape[0]
    Vn = np.empty(n_s)
    bi = np.empty(n_s, dtype=np.int64)
    half_c = 0.5 * c
    for si in numba.prange(n_s):
        s = s_grid[si]
        best = -1.0e300
        ib = 0
        for i in range(n_a):
            ai = a_grid[i]
            q = 0.0
            for j in range(n_a):
                pj = opp[si, j]
                if pj <= 0.0:
                    continue
                h1, h2 = ration_pair(ai, a_grid[j], s)
                x = s - (h1 + h2)
                ev = expected_V(V, x, r, K, sigma, nodes, weights)
                q += pj * (p * h1 - half_c * h1 * h1 + gamma * ev)
            if q > best:
                best = q
                ib = i
        Vn[si] = best
        bi[si] = ib
    return Vn, bi


@numba.njit(cache=True)
def _add_interp_weight(T_row, coeff, s, upper):
    """Exact linear-interpolation weight transpose of interp_uniform(V, s, upper)."""
    n = T_row.shape[0]
    if s <= 0.0:
        T_row[0] += coeff
        return
    if s >= upper:
        T_row[n - 1] += coeff
        return
    pos = s / upper * (n - 1)
    i = int(pos)
    if i >= n - 1:
        T_row[n - 1] += coeff
        return
    t = pos - i
    T_row[i] += coeff * (1.0 - t)
    T_row[i + 1] += coeff * t


@numba.njit(cache=True)
def _add_expected_weights(T_row, coeff, x, r, K, sigma, half_sig2, shocks, weights):
    """Add transition weights into T_row so dot(T_row, V) == expected_V(V, x, ...)."""
    n = T_row.shape[0]
    if x <= 0.0:
        T_row[0] += coeff
        return
    gr = logistic_gr(x, r, K)
    if gr <= 0.0:
        T_row[0] += coeff
        return
    if sigma <= 0.0:
        sp = gr if gr < K else K
        _add_interp_weight(T_row, coeff, sp, K)
        return
    z = (math.log(K / gr) + half_sig2) / sigma
    p_cens = 0.5 * math.erfc(z / math.sqrt(2.0))
    unc_w = 0.0
    n_q = shocks.shape[0]
    for q in range(n_q):
        sp = gr * shocks[q]
        if sp < K:
            unc_w += weights[q]
    if unc_w > 1e-15 and p_cens < 1.0:
        T_row[n - 1] += coeff * p_cens
        scale = coeff * (1.0 - p_cens) / unc_w
        for q in range(n_q):
            sp = gr * shocks[q]
            if sp < K:
                _add_interp_weight(T_row, scale * weights[q], sp, K)
    else:
        T_row[n - 1] += coeff


@numba.njit(parallel=True, cache=True)
def build_br_mdp(opp, s_grid, a_grid, r, K, sigma, p, c, nodes, weights):
    """Precompute exact single-agent MDP rewards R[s, i] and transitions T[s, i, s'] against opp."""
    n_s = s_grid.shape[0]
    n_a = a_grid.shape[0]
    R = np.zeros((n_s, n_a))
    T = np.zeros((n_s, n_a, n_s))
    half_c = 0.5 * c
    half_sig2 = 0.5 * sigma * sigma
    shocks = np.empty(nodes.shape[0])
    for q in range(nodes.shape[0]):
        shocks[q] = math.exp(sigma * nodes[q] - half_sig2)

    is_uniform = True
    if n_a >= 2:
        da = a_grid[1] - a_grid[0]
        tol_u = 1.0e-12 * max(1.0, abs(a_grid[-1]))
        for k in range(n_a):
            if abs((a_grid[k] - a_grid[0]) - k * da) > tol_u:
                is_uniform = False
                break
    else:
        da = 0.0

    for si in numba.prange(n_s):
        s = s_grid[si]
        if is_uniform and s > 0.0:
            n_sum = 2 * n_a - 1
            W = np.zeros((n_sum, n_s))
            w_computed = np.zeros(n_sum, dtype=np.uint8)
            for k in range(n_sum):
                a_sum = 2.0 * a_grid[0] + k * da
                if 0.0 < a_sum < s:
                    _add_expected_weights(
                        W[k], 1.0, s - a_sum, r, K, sigma, half_sig2, shocks, weights
                    )
                    w_computed[k] = 1
                elif a_sum <= 0.0:
                    _add_expected_weights(
                        W[k], 1.0, s, r, K, sigma, half_sig2, shocks, weights
                    )
                    w_computed[k] = 1
            for i in range(n_a):
                ai = a_grid[i]
                r_acc = 0.0
                t_row = T[si, i]
                for j in range(n_a):
                    pj = opp[si, j]
                    if pj <= 0.0:
                        continue
                    aj = a_grid[j]
                    h1, h2 = ration_pair(ai, aj, s)
                    r_acc += pj * (p * h1 - half_c * h1 * h1)
                    k = i + j
                    if w_computed[k]:
                        wk = W[k]
                        for m_s in range(n_s):
                            wval = wk[m_s]
                            if wval != 0.0:
                                t_row[m_s] += pj * wval
                    else:
                        t_row[0] += pj
                R[si, i] = r_acc
        else:
            for i in range(n_a):
                ai = a_grid[i]
                r_acc = 0.0
                t_row = T[si, i]
                for j in range(n_a):
                    pj = opp[si, j]
                    if pj <= 0.0:
                        continue
                    h1, h2 = ration_pair(ai, a_grid[j], s)
                    r_acc += pj * (p * h1 - half_c * h1 * h1)
                    x = s - (h1 + h2)
                    _add_expected_weights(
                        t_row, pj, x, r, K, sigma, half_sig2, shocks, weights
                    )
                R[si, i] = r_acc
    return R, T


@numba.njit(cache=True)
def vi_br_mdp(R, T_flat, V_init, gamma, tol, limit, min_H):
    """Run value iteration on a precomputed MDP (R, T_flat) with zero Python overhead."""
    n_s, n_a = R.shape
    V = V_init.copy()
    action = np.zeros(n_s, dtype=np.int64)
    delta = np.inf
    H = 0
    for h in range(1, limit + 1):
        EV = np.dot(T_flat, V)
        delta = 0.0
        for si in range(n_s):
            base = si * n_a
            best = -1.0e300
            ib = 0
            for i in range(n_a):
                q = R[si, i] + gamma * EV[base + i]
                if q > best:
                    best = q
                    ib = i
            diff = abs(best - V[si])
            if diff > delta:
                delta = diff
            V[si] = best
            action[si] = ib
        H = h
        if delta < tol and h >= min_H:
            break
    return V, action, H, delta


@numba.njit(parallel=True, cache=True)
def build_policy_mdp(p1, p2, s_grid, a_grid, r, K, sigma, price, c, nodes, weights):
    """Precompute exact rewards R1[s], R2[s] and transitions P[s, s'] for fixed profile (p1, p2)."""
    n_s = s_grid.shape[0]
    n_a = a_grid.shape[0]
    R1 = np.zeros(n_s)
    R2 = np.zeros(n_s)
    P = np.zeros((n_s, n_s))
    half_c = 0.5 * c
    half_sig2 = 0.5 * sigma * sigma
    shocks = np.empty(nodes.shape[0])
    for q in range(nodes.shape[0]):
        shocks[q] = math.exp(sigma * nodes[q] - half_sig2)
    for si in numba.prange(n_s):
        s = s_grid[si]
        acc1 = 0.0
        acc2 = 0.0
        p_row = P[si]
        for i in range(n_a):
            wi = p1[si, i]
            if wi <= 0.0:
                continue
            ai = a_grid[i]
            for j in range(n_a):
                w = wi * p2[si, j]
                if w <= 0.0:
                    continue
                h1, h2 = ration_pair(ai, a_grid[j], s)
                acc1 += w * (price * h1 - half_c * h1 * h1)
                acc2 += w * (price * h2 - half_c * h2 * h2)
                x = s - (h1 + h2)
                _add_expected_weights(
                    p_row, w, x, r, K, sigma, half_sig2, shocks, weights
                )
        R1[si] = acc1
        R2[si] = acc2
    return R1, R2, P


def hpi_br_mdp(R, T_flat, V_init, gamma, tol, limit, min_H, max_pi_iters=50):
    """Solve best-response MDP (R, T_flat) via exact Howard Policy Iteration with VI fallback."""
    n_s, n_a = R.shape
    T3 = T_flat.reshape(n_s, n_a, n_s)
    idx_s = np.arange(n_s)
    eye_s = np.eye(n_s, dtype=np.float64)
    if V_init is not None and np.any(V_init):
        Q0 = R + gamma * np.dot(T_flat, V_init).reshape(n_s, n_a)
        action = np.argmax(Q0, axis=1)
    else:
        action = np.argmax(R, axis=1)

    V = V_init.copy() if V_init is not None else np.zeros(n_s, dtype=np.float64)
    converged = False
    iters = 0
    delta = np.inf
    for it in range(1, max_pi_iters + 1):
        iters = it
        P_pi = T3[idx_s, action]
        r_pi = R[idx_s, action]
        try:
            V = np.linalg.solve(eye_s - gamma * P_pi, r_pi)
        except np.linalg.LinAlgError:
            break
        Q = R + gamma * np.dot(T_flat, V).reshape(n_s, n_a)
        new_action = np.argmax(Q, axis=1)
        max_q = Q[idx_s, new_action]
        cur_q = Q[idx_s, action]
        improve = max_q - cur_q
        delta = float(np.max(np.abs(max_q - V)))
        strict = improve > 1.0e-11
        if not np.any(strict):
            converged = True
            break
        action = np.where(strict, new_action, action)

    if not converged:
        return vi_br_mdp(R, T_flat, V, gamma, tol, limit, min_H)
    return V, action, max(iters, int(min_H)), delta
