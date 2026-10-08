"""Best-response distance, rollout estimates of J, collapse, and long-run stock."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from env import next_stock, ration
from kernels import build_br_mdp, hpi_br_mdp, vi_br_mdp
from utils import gauss_hermite_normal, horizon_for_tail, stock_catch_grids


@dataclass
class BestResponseResult:
    V: np.ndarray
    policy: np.ndarray
    action_idx: np.ndarray
    H: int
    converged: bool
    delta: float

    def value_at_s0(self):
        return float(self.V[-1])


def solve_best_response(
    cfg,
    r,
    opponent,
    s_grid=None,
    a_grid=None,
    nodes=None,
    weights=None,
    verbose=False,
    n_stock=None,
    n_catch=None,
    n_quad=None,
    gamma=None,
    sigma=None,
    max_H=None,
    value_tol=None,
    v_init=None,
):
    """Single-agent value iteration against a fixed opponent mixture `opponent[s, a]`.

    Parameters
    ----------
    v_init : array-like, optional
        Warm-start value function with shape ``(n_stock,)``.  When provided the
        iteration starts from ``v_init`` instead of zeros, which dramatically
        reduces the number of sweeps needed when the opponent has changed only
        slightly since the last call (e.g. during MB-OA replanning).

        Correctness guarantee: the fixed point is unique (Bellman contraction)
        so the warm start reaches the same value and greedy action as a cold
        start, within ``value_tol``.  A test in ``tests/test_learners.py``
        verifies this on a small deterministic game.
    """
    if s_grid is None or a_grid is None:
        s_grid, a_grid = stock_catch_grids(cfg, n_stock=n_stock, n_catch=n_catch)
    s_grid = np.asarray(s_grid, dtype=np.float64)
    a_grid = np.asarray(a_grid, dtype=np.float64)
    opponent = np.asarray(opponent, dtype=np.float64)
    if opponent.shape != (s_grid.shape[0], a_grid.shape[0]):
        raise ValueError("opponent policy must have shape (n_stock, n_catch)")
    m = cfg["model"]
    sol = cfg["solver"]
    if nodes is None or weights is None:
        nq = int(cfg["grids"]["n_quad"] if n_quad is None else n_quad)
        nodes, weights = gauss_hermite_normal(nq)
    K = float(m["K"])
    sigma = float(m["sigma"] if sigma is None else sigma)
    gamma = float(m["gamma"] if gamma is None else gamma)
    p = float(m["p"])
    c = float(m["c"])
    rr = float(r)
    tol = float(sol["value_tol"] if value_tol is None else value_tol)
    limit = int(sol["max_H"] if max_H is None else max_H)
    min_H = 1 if gamma <= 0.0 else int(sol["min_H"])
    # Warm start: begin from v_init when provided, otherwise from zeros.
    n_s, n_a = s_grid.shape[0], a_grid.shape[0]
    R, T = build_br_mdp(opponent, s_grid, a_grid, rr, K, sigma, p, c, nodes, weights)
    T_flat = T.reshape(n_s * n_a, n_s)
    if v_init is not None:
        V0 = np.asarray(v_init, dtype=np.float64).reshape(n_s).copy()
    else:
        V0 = np.zeros(n_s, dtype=np.float64)
    if verbose:
        V = V0
        action = np.zeros(n_s, dtype=np.int64)
        rows_idx = np.arange(n_s)
        delta = np.inf
        H = 0
        for h in range(1, limit + 1):
            Q = R + gamma * (T_flat @ V).reshape(n_s, n_a)
            action = np.argmax(Q, axis=1)
            Vn = Q[rows_idx, action]
            delta = float(np.max(np.abs(Vn - V)))
            V = Vn
            H = h
            if h == 1 or h % 25 == 0 or delta < tol:
                print(f"  br H={h} sup|dV|={delta:.3e}", flush=True)
            if delta < tol and h >= min_H:
                break
    else:
        if max_H is None and 0.0 <= gamma < 1.0:
            V, action, H, delta = hpi_br_mdp(R, T_flat, V0, gamma, tol, limit, min_H)
        else:
            V, action, H, delta = vi_br_mdp(R, T_flat, V0, gamma, tol, limit, min_H)
    policy = np.zeros_like(opponent)
    policy[np.arange(n_s), action] = 1.0
    return BestResponseResult(V, policy, action, int(H), bool(delta < tol), float(delta))


def equilibrium_gap(V_own, V_br):
    """Return (gap at s=K, sup-norm gap). Positive means a profitable deviation."""
    gap = np.asarray(V_br, dtype=np.float64) - np.asarray(V_own, dtype=np.float64)
    return float(gap[-1]), float(np.max(gap))


def _sample_actions(probs, idx, a_grid, rng):
    u = rng.random(idx.shape[0])
    cdf = np.cumsum(probs[idx], axis=1)
    # Guard a cdf that stops just short of 1 because of roundoff.
    cdf[:, -1] = 1.0
    k = (cdf >= u[:, None]).argmax(axis=1)
    return a_grid[k]


def _sample_from_cdf(cdf_table, idx, a_grid, rng):
    u = rng.random(idx.shape[0])
    k = (cdf_table[idx] >= u[:, None]).argmax(axis=1)
    return a_grid[k]


def rollout_returns(
    s_grid,
    a_grid,
    p1,
    p2,
    cfg,
    r,
    n_rollouts=None,
    length=None,
    seed=None,
    s0=None,
    xi=None,
):
    """Monte Carlo returns from s0. Noise `xi` can be reused across policies (common random numbers).

    Action sampling uses a separate stream so mixed policies do not shift the stock shocks.
    Stock is snapped to the nearest grid node, which is where the policy is defined.
    """
    m = cfg["model"]
    ev = cfg["evaluation"]
    K = float(m["K"])
    p = float(m["p"])
    c = float(m["c"])
    sigma = float(m["sigma"])
    gamma = float(m["gamma"])
    delta = float(m["delta"])
    if n_rollouts is None:
        n_rollouts = int(ev["n_rollouts"])
    if length is None:
        length = horizon_for_tail(gamma, float(ev["gamma_tail"]))
    if seed is None:
        seed = int(ev["mc_seed"])
    if s0 is None:
        s0 = float(m["s0"])
    last = int(ev["long_run_last"])
    s_grid = np.asarray(s_grid, dtype=np.float64)
    a_grid = np.asarray(a_grid, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    n_s = s_grid.shape[0]
    if xi is None:
        xi = np.random.default_rng(seed).normal(size=(length, n_rollouts))
    else:
        xi = np.asarray(xi, dtype=np.float64)
        if xi.shape != (length, n_rollouts):
            raise ValueError("xi must have shape (length, n_rollouts)")
    act_rng = np.random.default_rng(seed + 10_000_019)
    pure1 = bool(np.all(p1.max(axis=1) >= 1.0 - 1.0e-10))
    pure2 = bool(np.all(p2.max(axis=1) >= 1.0 - 1.0e-10))
    arg1 = np.argmax(p1, axis=1)
    arg2 = np.argmax(p2, axis=1)
    cdf1 = None if pure1 else np.cumsum(p1, axis=1)
    if cdf1 is not None:
        cdf1[:, -1] = 1.0
    cdf2 = None if pure2 else np.cumsum(p2, axis=1)
    if cdf2 is not None:
        cdf2[:, -1] = 1.0
    s = np.full(n_rollouts, s0, dtype=np.float64)
    ret = np.zeros((n_rollouts, 2), dtype=np.float64)
    # collapse_50 is the minimum over the first T seasons. collapse_long is the
    # minimum over the whole evaluation rollout. They are different events.
    t_short = int(m["T"])
    min_long = np.full(n_rollouts, s0, dtype=np.float64)
    min_50 = np.full(n_rollouts, s0, dtype=np.float64)
    discount = 1.0
    tail = []
    for t in range(length):
        idx = np.clip(np.rint(s / K * (n_s - 1)).astype(np.int64), 0, n_s - 1)
        a1 = a_grid[arg1[idx]] if pure1 else _sample_from_cdf(cdf1, idx, a_grid, act_rng)
        a2 = a_grid[arg2[idx]] if pure2 else _sample_from_cdf(cdf2, idx, a_grid, act_rng)
        hats = ration(np.column_stack([a1, a2]), s)
        rewards = p * hats - 0.5 * c * np.square(hats)
        ret += discount * rewards
        x = s - hats.sum(axis=1)
        s = next_stock(x, r, K, sigma, xi[t])
        min_long = np.minimum(min_long, s)
        if t < t_short:
            min_50 = np.minimum(min_50, s)
        discount *= gamma
        if t >= length - last:
            tail.append(s.copy())
    se = ret.std(axis=0, ddof=1) / math.sqrt(n_rollouts)
    total = ret.sum(axis=1)
    threshold = delta * K
    return {
        "J": ret.mean(axis=0),
        "J_se": se,
        "J_total": float(total.mean()),
        "J_total_se": float(total.std(ddof=1) / math.sqrt(n_rollouts)),
        "collapse_50": float(np.mean(min_50 < threshold)),
        "collapse_long": float(np.mean(min_long < threshold)),
        "long_run_stock": float(np.mean(np.stack(tail))) if tail else float("nan"),
        "length": length,
        "n_rollouts": n_rollouts,
        "truncation_discount": float(discount),
        "T_collapse": t_short,
    }


def steady_stock(s_grid, a_grid, p1, p2, r, K, s0, tol=1.0e-6, limit=500):
    """Mean-dynamics fixed point (omega = 1) using interpolated mean requests."""
    from env import mean_path

    stocks, _catch = mean_path(s_grid, a_grid, p1, p2, r, K, s0, limit)
    for t in range(1, limit):
        if abs(stocks[t] - stocks[t - 1]) < tol:
            return float(stocks[t]), int(t)
    return float(stocks[-1]), limit


def policy_values(cfg, r, p1, p2, s_grid, a_grid, nodes=None, weights=None,
                  v1_init=None, v2_init=None):
    """Discounted values of a fixed profile. The last transition bootstraps; nothing is terminal."""
    from kernels import build_policy_mdp

    s_grid = np.asarray(s_grid, dtype=np.float64)
    a_grid = np.asarray(a_grid, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    m = cfg["model"]
    sol = cfg["solver"]
    if nodes is None or weights is None:
        nodes, weights = gauss_hermite_normal(int(cfg["grids"]["n_quad"]))
    K = float(m["K"])
    sigma = float(m["sigma"])
    gamma = float(m["gamma"])
    price = float(m["p"])
    c = float(m["c"])
    tol = float(sol["value_tol"])
    limit = int(sol["max_H"])
    R1, R2, P = build_policy_mdp(p1, p2, s_grid, a_grid, float(r), K, sigma, price, c, nodes, weights)
    n_s = s_grid.shape[0]
    if 0.0 <= gamma < 1.0:
        try:
            rhs = np.column_stack((R1, R2))
            sol_v = np.linalg.solve(np.eye(n_s, dtype=np.float64) - gamma * P, rhs)
            return sol_v[:, 0].copy(), sol_v[:, 1].copy(), True, 0.0
        except np.linalg.LinAlgError:
            pass
    V1 = np.zeros(n_s, dtype=np.float64) if v1_init is None else np.asarray(v1_init, dtype=np.float64).reshape(n_s).copy()
    V2 = np.zeros(n_s, dtype=np.float64) if v2_init is None else np.asarray(v2_init, dtype=np.float64).reshape(n_s).copy()
    delta = np.inf
    for _h in range(1, limit + 1):
        N1 = R1 + gamma * (P @ V1)
        N2 = R2 + gamma * (P @ V2)
        delta = float(max(np.max(np.abs(N1 - V1)), np.max(np.abs(N2 - V2))))
        V1, V2 = N1, N2
        if delta < tol:
            break
    return V1, V2, bool(delta < tol), float(delta)


def one_hot_from_index(index, n_actions):
    policy = np.zeros((index.shape[0], n_actions), dtype=np.float64)
    policy[np.arange(index.shape[0]), index] = 1.0
    return policy


def evaluate(p1, p2, r, cfg, s_grid, a_grid, planner_p1, planner_p2, eq_p1=None, eq_p2=None,
             planner_J_dp=None, eq_J_dp=None, xi=None, seed=None, with_gap=True,
             br1_v_init=None, br2_v_init=None, v1_init=None, v2_init=None,
             j_plan_cached=None, j_eq_cached=None):
    """Greedy-policy evaluation. Collapse within T and over the long rollout are separate.

    G uses Monte Carlo totals with the noise in `xi` shared with the planner rollouts.
    E is the exact best-response gap, not a learned Q gap. Own value is the policy's
    dynamic-programming value at full stock.

    Parameters
    ----------
    br1_v_init, br2_v_init : array-like, optional
        Warm-start value functions for the two best-response solves (fleet 1 responds
        to p2; fleet 2 responds to p1). Pass the ``V`` from the previous evaluation
        call to cut the number of Bellman sweeps when the policy has not changed much.
    """
    mc = rollout_returns(s_grid, a_grid, p1, p2, cfg, r, seed=seed, xi=xi)
    if xi is None:
        # Rebuild the shocks this call just drew so the planner uses the same noise.
        # rollout_returns draws them internally when xi is None; draw explicitly instead.
        raise ValueError("pass xi so the planner and the equilibrium share the shocks")
    j = mc["J_total"]
    if j_plan_cached is None:
        mc_plan = rollout_returns(
            s_grid, a_grid, planner_p1, planner_p2, cfg, r, seed=seed, xi=xi
        )
        j_plan = mc_plan["J_total"]
    else:
        j_plan = float(j_plan_cached)
    out = {
        "J1": float(mc["J"][0]),
        "J2": float(mc["J"][1]),
        "J": j,
        "J_se": mc["J_total_se"],
        "J_plan": j_plan,
        "G": float(1.0 - j / j_plan) if j_plan != 0.0 else float("nan"),
        "J_over_plan": float(j / j_plan) if j_plan != 0.0 else float("nan"),
        "collapse_50": mc["collapse_50"],
        "collapse_long": mc["collapse_long"],
        "long_run_stock": mc["long_run_stock"],
        "mc_length": mc["length"],
        "mc_rollouts": mc["n_rollouts"],
        "planner_J_dp": None if planner_J_dp is None else float(planner_J_dp),
        "J_over_plan_dp": None if not planner_J_dp else float(j / planner_J_dp),
    }
    if eq_p1 is not None:
        if j_eq_cached is None:
            mc_eq = rollout_returns(s_grid, a_grid, eq_p1, eq_p2, cfg, r, seed=seed, xi=xi)
            j_eq = mc_eq["J_total"]
        else:
            j_eq = float(j_eq_cached)
        out["J_eq"] = j_eq
        out["J_over_eq"] = float(j / j_eq) if j_eq != 0.0 else float("nan")
        out["G_vs_eq"] = float(1.0 - j / j_eq) if j_eq != 0.0 else float("nan")
    else:
        out["J_eq"] = None
        out["J_over_eq"] = None
        out["G_vs_eq"] = None
    out["eq_J_dp"] = None if eq_J_dp is None else float(eq_J_dp)
    out["J_over_eq_dp"] = None if not eq_J_dp else float(j / eq_J_dp)
    if with_gap:
        V1, V2, ok, delta = policy_values(
            cfg, r, p1, p2, s_grid, a_grid, v1_init=v1_init, v2_init=v2_init
        )
        br1 = solve_best_response(cfg, r, p2, s_grid=s_grid, a_grid=a_grid,
                                  v_init=br1_v_init)
        if np.array_equal(p1, p2):
            br2 = br1
        else:
            br2 = solve_best_response(cfg, r, p1, s_grid=s_grid, a_grid=a_grid,
                                      v_init=br2_v_init)
        e1 = float(br1.value_at_s0() - V1[-1])
        e2 = float(br2.value_at_s0() - V2[-1])
        out["E"] = float(max(e1, e2))
        out["E1"] = e1
        out["E2"] = e2
        out["gap_converged"] = bool(ok and br1.converged and br2.converged)
        out["gap_delta"] = float(delta)
        # Return the converged V's so callers can warm-start the next call.
        out["_br1_V"] = br1.V
        out["_br2_V"] = br2.V
        out["_v1_V"] = V1
        out["_v2_V"] = V2
    else:
        out["E"] = None
        out["E1"] = None
        out["E2"] = None
        out["gap_converged"] = None
        out["gap_delta"] = None
        out["_br1_V"] = None
        out["_br2_V"] = None
        out["_v1_V"] = None
        out["_v2_V"] = None
    return out
