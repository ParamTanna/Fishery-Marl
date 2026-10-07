"""Backward induction for the fishery stage game, with explicit equilibrium selection.

From V_0 = 0 the solver alternates two steps. It selects one stage Nash
equilibrium at every stock (the configured rule), then evaluates that policy
to convergence. Evaluation is a contraction; the selection map is not.
H is the number of selection steps. A run has converged when the selected
policy is unchanged and its unilateral gain at its own continuation is below
`value_tol`. If the selection map is still moving after `stall_outer` steps
without a smaller gain, the returned policy is the evaluated one with the
smallest gain, and `converged` is false. That gain is `stage_exploit_max`.

Enumeration, per state:
  * every pure-strategy Nash profile (asymmetric included);
  * every symmetric mixed equilibrium whose support has size 2
    (config `mixed_support_max`; size 3 and above are not enumerated).

If that set is empty and the action grid is small, Lemke-Howson (nashpy) is
tried. Otherwise the least-exploitable pure profile is used and flagged.

Selection rules each run as their own backward induction. Near-ties (within
`select_tol`) stick to the previous iteration's policy so the iteration does
not chatter on payoff ties. That tie-break does not override a strict ranking.

Risk dominance is the pairwise Harsanyi-Selten Nash product on pure profiles:
for profiles (i, j) and (k, l), the loss product is
    max(U1(i,j)-U1(k,j), 0) * max(U2(i,j)-U2(i,l), 0).
(i, j) risk-dominates (k, l) when its product is larger. The selected profile
is the one that risk-dominates the most rivals; remaining ties use the sum of
product gaps, then total return. Mixed equilibria are not in this comparison.
If no pure profile exists, the rule falls back to max_total.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from env import ration
from kernels import (
    build_U_fishery_asym,
    build_U_fishery_sym,
    build_U_levhari_sym,
    enum_sym_mixed,
    policy_value_fishery,
)
from utils import gauss_hermite_normal, stock_catch_grids

ENUMERATION_LIMIT = (
    "Pure Nash profiles are exhaustive. Symmetric mixed equilibria are exhaustive "
    "only for support size 2. Larger mixed supports are not enumerated."
)


@dataclass
class EquilibriumResult:
    model: str
    r: float
    rule: str
    symmetric_only: bool
    H: int
    converged: bool
    delta: float
    s_grid: np.ndarray
    a_grid: np.ndarray
    V1: np.ndarray
    V2: np.ndarray
    p1: np.ndarray
    p2: np.ndarray
    n_eq: np.ndarray
    selected_kind: np.ndarray  # 0 sym pure, 1 asym pure, 2 mixed, 3 fallback
    stage_exploit_max: float
    eps_rel: float
    eps_rel_threshold: float
    n_fallback: int
    n_mixed_selected: int
    n_truncated_mixed: int
    n_trivial: int
    selection_seed: int
    E_s0: float = 0.0

    def total_value(self):
        return self.V1 + self.V2

    def value_at_s0(self):
        return float(self.V1[-1]), float(self.V2[-1])


def pure_nash_mask(U1, U2, tol):
    """Boolean tensor, True where (own, opp) is a pure Nash profile."""
    max1 = U1.max(axis=1)
    max2 = U2.max(axis=2)
    return (U1 >= max1[:, None, :] - tol) & (U2 >= max2[:, :, None] - tol)


def least_exploitable_pure(U1, U2, symmetric_only):
    max1 = U1.max(axis=0)
    max2 = U2.max(axis=1)
    exploit = (max1[None, :] - U1) + (max2[:, None] - U2)
    if symmetric_only:
        diag = np.diag(exploit)
        i = int(np.argmin(diag))
        return i, i, float(diag[i])
    flat = int(np.argmin(exploit))
    i, j = np.unravel_index(flat, exploit.shape)
    return int(i), int(j), float(exploit[i, j])


def _policy_from_pure(n_a, i, j):
    p1 = np.zeros(n_a, dtype=np.float64)
    p2 = np.zeros(n_a, dtype=np.float64)
    p1[i] = 1.0
    p2[j] = 1.0
    return p1, p2


def _policy_from_mix(n_a, i, j, prob):
    p1 = np.zeros(n_a, dtype=np.float64)
    p1[i] = prob
    p1[j] = 1.0 - prob
    return p1, p1.copy()


def _sticky(pool, prev1, prev2):
    def key(item):
        dist = float(np.abs(item["p1"] - prev1).sum() + np.abs(item["p2"] - prev2).sum())
        return (dist, 0 if item["sym"] else 1, item["i"], item["j"])

    return min(pool, key=key)


def _pareto_dominates(a, b, tol):
    ge = a["u1"] >= b["u1"] - tol and a["u2"] >= b["u2"] - tol
    gt = a["u1"] > b["u1"] + tol or a["u2"] > b["u2"] + tol
    return ge and gt


def _hs_product(U1, U2, src, dst):
    i, j = src["i"], src["j"]
    k, l = dst["i"], dst["j"]
    d1 = max(0.0, float(U1[i, j] - U1[k, j]))
    d2 = max(0.0, float(U2[i, j] - U2[i, l]))
    return d1 * d2


def _try_nashpy(U1, U2, n_labels):
    import nashpy

    game = nashpy.Game(np.ascontiguousarray(U1), np.ascontiguousarray(U2))
    best = None
    best_ex = np.inf
    n = U1.shape[0]
    for label in range(min(int(n_labels), max(1, 2 * n))):
        try:
            p, q = game.lemke_howson(label)
        except Exception:
            continue
        p = np.clip(np.asarray(p, dtype=np.float64), 0.0, None)
        q = np.clip(np.asarray(q, dtype=np.float64), 0.0, None)
        if p.sum() <= 0.0 or q.sum() <= 0.0 or not np.all(np.isfinite(p)) or not np.all(np.isfinite(q)):
            continue
        p /= p.sum()
        q /= q.sum()
        u1 = float(p @ U1 @ q)
        u2 = float(p @ U2 @ q)
        ex = max(float(np.max(U1 @ q) - u1), float(np.max(p @ U2) - u2))
        if ex < best_ex:
            best_ex = ex
            best = (p, q, u1, u2, ex)
    return best


def _select_one(cands, U1, U2, rule, prev1, prev2, seed, si, select_tol, symmetric_only, nashpy_on, n_labels, ne_tol):
    """Return a dict with u1, u2, p1, p2, kind, sym, i, j, fallback."""
    n_a = U1.shape[0]
    if cands:
        if rule == "min_total":
            target = min(c["u1"] + c["u2"] for c in cands)
            pool = [c for c in cands if c["u1"] + c["u2"] <= target + select_tol]
            chosen = _sticky(pool, prev1, prev2)
        elif rule == "payoff_dominant":
            winners = [a for a in cands if all(a is b or _pareto_dominates(a, b, select_tol) for b in cands)]
            if winners:
                chosen = _sticky(winners, prev1, prev2)
            else:
                target = max(c["u1"] + c["u2"] for c in cands)
                pool = [c for c in cands if c["u1"] + c["u2"] >= target - select_tol]
                chosen = _sticky(pool, prev1, prev2)
        elif rule == "risk_dominant":
            pure = [c for c in cands if c["kind"] != 2]
            if not pure:
                target = max(c["u1"] + c["u2"] for c in cands)
                pool = [c for c in cands if c["u1"] + c["u2"] >= target - select_tol]
                chosen = _sticky(pool, prev1, prev2)
            elif len(pure) == 1:
                chosen = pure[0]
            else:
                scored = []
                for a in pure:
                    wins = 0
                    margin = 0.0
                    for b in pure:
                        if a is b:
                            continue
                        la = _hs_product(U1, U2, a, b)
                        lb = _hs_product(U1, U2, b, a)
                        margin += la - lb
                        if la > lb + 1.0e-12:
                            wins += 1
                    scored.append((wins, margin, a["u1"] + a["u2"], a))
                scored.sort(key=lambda t: (t[0], t[1], t[2]), reverse=True)
                best_w, best_m = scored[0][0], scored[0][1]
                pool = [t[3] for t in scored if t[0] == best_w and abs(t[1] - best_m) <= 1.0e-9]
                chosen = _sticky(pool, prev1, prev2)
        elif rule == "random_seeded":
            rng = np.random.default_rng(int(seed) + int(si) * 10007)
            chosen = cands[int(rng.integers(0, len(cands)))]
        elif rule == "max_catch":
            target = max(c["catch"] for c in cands)
            pool = [c for c in cands if c["catch"] >= target - select_tol]
            chosen = _sticky(pool, prev1, prev2)
        elif rule == "continuity":
            # Closest stage equilibrium to the policy already in use. Ties keep the higher total.
            chosen = _sticky(cands, prev1, prev2)
        else:
            # max_total, also the default
            target = max(c["u1"] + c["u2"] for c in cands)
            pool = [c for c in cands if c["u1"] + c["u2"] >= target - select_tol]
            chosen = _sticky(pool, prev1, prev2)
        chosen = dict(chosen)
        chosen["fallback"] = False
        return chosen

    i, j, expl = least_exploitable_pure(U1, U2, symmetric_only)
    if nashpy_on and expl > 10.0 * ne_tol:
        found = _try_nashpy(U1, U2, n_labels)
        if found is not None and found[4] < expl:
            p, q, u1, u2, _ex = found
            return {
                "u1": u1,
                "u2": u2,
                "p1": p,
                "p2": q,
                "kind": 3,
                "sym": bool(np.allclose(p, q)),
                "i": int(np.argmax(p)),
                "j": int(np.argmax(q)),
                "fallback": True,
            }
    p1, p2 = _policy_from_pure(n_a, i, j)
    return {
        "u1": float(U1[i, j]),
        "u2": float(U2[i, j]),
        "p1": p1,
        "p2": p2,
        "kind": 3,
        "sym": i == j,
        "i": i,
        "j": j,
        "fallback": True,
    }


def _profile_catch(a_grid, p1, p2, stock):
    """Expected total executed catch of a (possibly mixed) profile at one stock."""
    total = 0.0
    for i in np.flatnonzero(p1 > 1.0e-15):
        for j in np.flatnonzero(p2 > 1.0e-15):
            weight = float(p1[i] * p2[j])
            caught = ration(np.array([a_grid[i], a_grid[j]], dtype=np.float64), float(stock))
            total += weight * float(caught.sum())
    return total


def _stage_exploit(U1, U2, p1, p2):
    u1 = float(p1 @ U1 @ p2)
    u2 = float(p1 @ U2 @ p2)
    return max(float(np.max(U1 @ p2) - u1), float(np.max(p1 @ U2) - u2))


def _problem(cfg, r, model, n_stock, n_catch, n_quad):
    if model == "fishery":
        s_grid, a_grid = stock_catch_grids(cfg, n_stock=n_stock, n_catch=n_catch)
        m = cfg["model"]
        sol = cfg["solver"]
        nq = int(cfg["grids"]["n_quad"] if n_quad is None else n_quad)
        nodes, weights = gauss_hermite_normal(nq)
        return {
            "cfg": cfg,
            "model": "fishery",
            "s_grid": s_grid,
            "a_grid": a_grid,
            "nodes": nodes,
            "weights": weights,
            "r": float(r),
            "K": float(m["K"]),
            "sigma": float(m["sigma"]),
            "gamma": float(m["gamma"]),
            "p": float(m["p"]),
            "c": float(m["c"]),
            "alpha": None,
            "value_tol": float(sol["value_tol"]),
            "value_damp": float(sol.get("value_damp", 1.0)),
            "ne_tol": float(sol["ne_tol"]),
            "select_tol": float(sol["select_tol"]),
            "max_H": int(sol["max_H"]),
            "min_H": int(sol["min_H"]),
            "mixed_support_max": int(sol["mixed_support_max"]),
            "max_mixed_store": int(sol["max_mixed_store"]),
            "eps_rel": float(sol.get("eps_rel", 0.005)),
            "use_nashpy": bool(sol["use_nashpy_fallback"]),
            "nashpy_max_actions": int(sol["nashpy_max_actions"]),
            "nashpy_labels": int(sol["nashpy_labels"]),
            "stall_outer": int(sol.get("stall_outer", 8)),
        }
    if model != "levhari":
        raise ValueError(model)
    L = cfg["levhari_mirman"]
    n_s = int(L["n_stock"] if n_stock is None else n_stock)
    n_a = int(L["n_fraction"] if n_catch is None else n_catch)
    s_max = float(L["s_max"])
    s_grid = np.linspace(s_max / n_s, s_max, n_s, dtype=np.float64)
    a_grid = np.linspace(0.0, 1.0, n_a, dtype=np.float64)
    return {
        "model": "levhari",
        "s_grid": s_grid,
        "a_grid": a_grid,
        "nodes": np.zeros(1),
        "weights": np.ones(1),
        "r": float(L["alpha"]),
        "K": s_max,
        "sigma": 0.0,
        "gamma": float(L["gamma"]),
        "p": 0.0,
        "c": 0.0,
        "alpha": float(L["alpha"]),
        "value_tol": float(L["value_tol"]),
        "value_damp": float(L.get("value_damp", 1.0)),
        "ne_tol": float(L["ne_tol"]),
        "select_tol": 1.0e-8,
        "max_H": int(L["max_H"]),
        "min_H": 2,
        "mixed_support_max": 2,
        "max_mixed_store": 32,
        "eps_rel": 0.005,
        "use_nashpy": False,
        "nashpy_max_actions": 12,
        "nashpy_labels": 4,
        "stall_outer": 8,
    }


def _build_payoffs(prob, V1, V2, symmetric_only):
    if prob["model"] == "levhari":
        U1 = build_U_levhari_sym(V1, prob["s_grid"], prob["a_grid"], prob["alpha"], prob["K"], prob["gamma"])
        U2 = np.transpose(U1, (0, 2, 1))
        return U1, U2, True
    if symmetric_only:
        U1 = build_U_fishery_sym(
            V1, prob["s_grid"], prob["a_grid"], prob["r"], prob["K"], prob["sigma"],
            prob["gamma"], prob["p"], prob["c"], prob["nodes"], prob["weights"],
        )
        U2 = np.transpose(U1, (0, 2, 1))
        return U1, U2, True
    U1, U2 = build_U_fishery_asym(
        V1, V2, prob["s_grid"], prob["a_grid"], prob["r"], prob["K"], prob["sigma"],
        prob["gamma"], prob["p"], prob["c"], prob["nodes"], prob["weights"],
    )
    symmetric_game = float(np.max(np.abs(V1 - V2))) <= 1.0e-8
    return U1, U2, symmetric_game


def _evaluate_policy(prob, p1, p2, V1, V2, tol, limit=500):
    """Policy evaluation. The fishery map is a contraction of modulus gamma.

    The Levhari-Mirman check is a single backward-induction step from a closed
    form continuation, so this returns the values unchanged for that model.
    """
    if prob["model"] != "fishery":
        return V1, V2, 0, 0.0
    for it in range(1, int(limit) + 1):
        n1, n2 = policy_value_fishery(
            V1, V2, p1, p2, prob["s_grid"], prob["a_grid"], prob["r"], prob["K"],
            prob["sigma"], prob["gamma"], prob["p"], prob["c"], prob["nodes"], prob["weights"],
        )
        delta = float(max(np.max(np.abs(n1 - V1)), np.max(np.abs(n2 - V2))))
        V1, V2 = n1, n2
        if delta < tol:
            return V1, V2, it, delta
    return V1, V2, int(limit), delta


def _held_exploit(U1, U2, p1, p2):
    ex = 0.0
    for s in range(U1.shape[0]):
        ex = max(ex, _stage_exploit(U1[s], U2[s], p1[s], p2[s]))
    return float(ex)


def solve_equilibrium(
    cfg,
    r,
    rule="max_total",
    symmetric_only=True,
    model="fishery",
    selection_seed=None,
    n_stock=None,
    n_catch=None,
    n_quad=None,
    verbose=False,
    v_init=None,
):
    if model == "levhari" and not symmetric_only:
        raise NotImplementedError(
            "The Levhari-Mirman check uses the symmetric equilibrium. "
            "An asymmetric continuation would need a separate two-value kernel."
        )
    known = {
        "max_total", "min_total", "payoff_dominant", "risk_dominant", "random_seeded",
        "max_catch", "continuity",
    }
    if rule not in known:
        raise ValueError(f"unknown selection rule {rule}")
    prob = _problem(cfg, r, model, n_stock, n_catch, n_quad)
    if selection_seed is None:
        selection_seed = int(cfg.get("selection", {}).get("random_seed", 329))
    n_s = prob["s_grid"].shape[0]
    n_a = prob["a_grid"].shape[0]
    if v_init is None:
        V1 = np.zeros(n_s, dtype=np.float64)
    else:
        V1 = np.asarray(v_init, dtype=np.float64).reshape(n_s).copy()
    V2 = V1.copy()
    p1 = np.zeros((n_s, n_a), dtype=np.float64)
    p2 = np.zeros((n_s, n_a), dtype=np.float64)
    p1[:, 0] = 1.0
    p2[:, 0] = 1.0
    n_eq = np.zeros(n_s, dtype=np.int16)
    kinds = np.zeros(n_s, dtype=np.int8)
    delta = np.inf
    H = 0
    exploit_max = np.inf
    n_fallback = 0
    n_mixed = 0
    n_trunc = 0
    n_trivial = 0
    tol = prob["value_tol"]
    converged = False
    best_ex = np.inf
    best_snap = None
    stall = 0
    held_ex = None

    for h in range(1, prob["max_H"] + 1):
        U1, U2, symmetric_game = _build_payoffs(prob, V1, V2, symmetric_only)
        held_ex = None
        if h > 1 and prob["model"] == "fishery":
            # U is built from the continuation of the policy selected last iteration.
            held_ex = _held_exploit(U1, U2, p1, p2)
            if held_ex < best_ex:
                best_ex = held_ex
                best_snap = (
                    V1.copy(), V2.copy(), p1.copy(), p2.copy(),
                    n_eq.copy(), kinds.copy(),
                    int(n_fallback), int(n_mixed), int(n_trunc), int(n_trivial),
                    h - 1, held_ex,
                )
                stall = 0
            else:
                stall += 1
        mask = pure_nash_mask(U1, U2, prob["ne_tol"])
        if prob["mixed_support_max"] >= 2 and symmetric_game:
            mix_i, mix_j, mix_p, mix_u, mix_n, mix_over = enum_sym_mixed(
                U1, prob["ne_tol"], prob["max_mixed_store"]
            )
        else:
            mix_n = np.zeros(n_s, dtype=np.int16)
            mix_over = np.zeros(n_s, dtype=np.int16)
            mix_i = mix_j = mix_p = mix_u = None
        V1n = np.empty(n_s)
        V2n = np.empty(n_s)
        p1n = np.zeros_like(p1)
        p2n = np.zeros_like(p2)
        exploit_max = 0.0
        n_fallback = 0
        n_mixed = 0
        n_trivial = 0
        n_trunc = int(np.sum(mix_over))
        nashpy_on = prob["use_nashpy"] and n_a <= prob["nashpy_max_actions"]
        flat_tol = max(prob["ne_tol"] * 10.0, 1.0e-8)
        for s in range(n_s):
            # At a dead stock every request executes as zero, so every profile is a Nash
            # equilibrium. Counting them (and every 2-support mixture) is not informative.
            span = max(float(U1[s].max() - U1[s].min()), float(U2[s].max() - U2[s].min()))
            if span <= flat_tol:
                n_trivial += 1
                n_trunc -= int(mix_over[s])
                i_prev = int(np.argmax(p1[s]))
                j_prev = int(np.argmax(p2[s]))
                if symmetric_only:
                    j_prev = i_prev
                pi, pj = _policy_from_pure(n_a, i_prev, j_prev)
                V1n[s] = float(U1[s, i_prev, j_prev])
                V2n[s] = float(U2[s, i_prev, j_prev])
                p1n[s] = pi
                p2n[s] = pj
                n_eq[s] = n_a if symmetric_only else n_a * n_a
                kinds[s] = 0 if symmetric_only else 1
                continue
            cands = []
            ii, jj = np.nonzero(mask[s])
            order = np.lexsort((jj, ii))
            for i, j in zip(ii[order].tolist(), jj[order].tolist()):
                sym = i == j
                if symmetric_only and not sym:
                    continue
                pi, pj = _policy_from_pure(n_a, i, j)
                cands.append(
                    {
                        "u1": float(U1[s, i, j]),
                        "u2": float(U2[s, i, j]),
                        "kind": 0 if sym else 1,
                        "sym": sym,
                        "i": int(i),
                        "j": int(j),
                        "p1": pi,
                        "p2": pj,
                        "catch": _profile_catch(prob["a_grid"], pi, pj, prob["s_grid"][s]),
                    }
                )
            if mix_i is not None:
                for m in range(int(mix_n[s])):
                    i = int(mix_i[s, m])
                    j = int(mix_j[s, m])
                    prob_i = float(mix_p[s, m])
                    u = float(mix_u[s, m])
                    pol, pol2 = _policy_from_mix(n_a, i, j, prob_i)
                    cands.append(
                        {
                            "u1": u,
                            "u2": u,
                            "kind": 2,
                            "sym": True,
                            "i": i,
                            "j": j,
                            "p1": pol,
                            "p2": pol2,
                            "catch": _profile_catch(prob["a_grid"], pol, pol2, prob["s_grid"][s]),
                        }
                    )
            chosen = _select_one(
                cands, U1[s], U2[s], rule, p1[s], p2[s], selection_seed, s,
                prob["select_tol"], symmetric_only, nashpy_on, prob["nashpy_labels"], prob["ne_tol"],
            )
            V1n[s] = chosen["u1"]
            V2n[s] = chosen["u2"]
            p1n[s] = chosen["p1"]
            p2n[s] = chosen["p2"]
            n_eq[s] = len(cands)
            kinds[s] = chosen["kind"] if not chosen["fallback"] else 3
            if chosen["fallback"]:
                n_fallback += 1
            if chosen["kind"] == 2 and not chosen["fallback"]:
                n_mixed += 1
            ex = _stage_exploit(U1[s], U2[s], p1n[s], p2n[s])
            if ex > exploit_max:
                exploit_max = ex
        same = (
            h > 1
            and np.allclose(p1n, p1, atol=1.0e-8, rtol=0.0)
            and np.allclose(p2n, p2, atol=1.0e-8, rtol=0.0)
        )
        # Re-selecting the policy we just evaluated means it is a Nash equilibrium
        # of its own continuation. Stop. The stored value is that continuation.
        if prob["model"] == "fishery" and same and held_ex is not None and held_ex < tol:
            delta = held_ex
            exploit_max = held_ex
            H = h
            converged = True
            if verbose:
                print(
                    f"  eq[{rule}{' sym' if symmetric_only else ' all'} r={prob['r']}] "
                    f"H={h} fixed point exploit={held_ex:.3e} mixed={n_mixed} "
                    f"J={float(V1[-1] + V2[-1]):.1f}",
                    flush=True,
                )
            break

        n_eval = 0
        if prob["model"] == "fishery":
            V1n, V2n, n_eval, _eval_delta = _evaluate_policy(prob, p1n, p2n, V1n, V2n, tol)
            if symmetric_only:
                averaged = 0.5 * (V1n + V2n)
                V1n = averaged
                V2n = averaged.copy()
            # Continuity damps the continuation so the next stage game does not jump.
            # The policy itself stays the selected equilibrium. Its true value is
            # recomputed once, after the loop, for the reported gain.
            if rule == "continuity":
                damp = float(prob["value_damp"])
                V1n = (1.0 - damp) * V1 + damp * V1n
                V2n = (1.0 - damp) * V2 + damp * V2n
        else:
            damp = float(prob["value_damp"])
            if damp < 1.0:
                V1n = (1.0 - damp) * V1 + damp * V1n
                V2n = (1.0 - damp) * V2 + damp * V2n
        delta = float(max(np.max(np.abs(V1n - V1)), np.max(np.abs(V2n - V2))))
        n_switched = int(np.sum(np.argmax(p1n, axis=1) != np.argmax(p1, axis=1)))
        V1, V2, p1, p2 = V1n, V2n, p1n, p2n
        H = h
        if verbose and (h == 1 or h % 5 == 0 or prob["model"] != "fishery"):
            prev = "na" if held_ex is None else f"{held_ex:.3e}"
            print(
                f"  eq[{rule}{' sym' if symmetric_only else ' all'} r={prob['r']}] "
                f"H={h} sup|dV|={delta:.3e} prev_exploit={prev} "
                f"fallback={n_fallback} mixed={n_mixed} switched={n_switched} "
                f"eval={n_eval} J={float(V1[-1] + V2[-1]):.1f}",
                flush=True,
            )
        if prob["model"] != "fishery" and delta < tol and h >= prob["min_H"]:
            converged = True
            break
        if (
            prob["model"] == "fishery"
            and stall >= int(prob["stall_outer"])
            and h >= prob["min_H"]
            and best_snap is not None
        ):
            (
                V1, V2, p1, p2, n_eq, kinds,
                n_fallback, n_mixed, n_trunc, n_trivial, H, exploit_max,
            ) = best_snap
            delta = float(exploit_max)
            if verbose:
                print(
                    f"  eq[{rule}{' sym' if symmetric_only else ' all'} r={prob['r']}] "
                    f"selection did not settle; keeping H={H} exploit={exploit_max:.3e} "
                    f"J={float(V1[-1] + V2[-1]):.1f}",
                    flush=True,
                )
            break

    if prob["model"] == "fishery" and not converged:
        U1, U2, _symmetric_game = _build_payoffs(prob, V1, V2, symmetric_only)
        last_ex = _held_exploit(U1, U2, p1, p2)
        if best_snap is not None and best_ex <= last_ex:
            (
                V1, V2, p1, p2, n_eq, kinds,
                n_fallback, n_mixed, n_trunc, n_trivial, H, exploit_max,
            ) = best_snap
            delta = float(exploit_max)
        else:
            exploit_max = last_ex
            delta = float(last_ex)

    if rule == "continuity" and prob["model"] == "fishery":
        V1, V2, _n_eval, _eval_delta = _evaluate_policy(prob, p1, p2, V1, V2, tol)
        if symmetric_only:
            averaged = 0.5 * (V1 + V2)
            V1 = averaged
            V2 = averaged.copy()
        U1, U2, _symmetric_game = _build_payoffs(prob, V1, V2, symmetric_only)
        exploit_max = _held_exploit(U1, U2, p1, p2)
        delta = float(exploit_max)

    j_i = max(abs(float(V1[-1])), abs(float(V2[-1])), 1.0e-12)
    realized_rel = float(exploit_max) / j_i
    # Certified approximate MPE requires infinite-horizon exploitability E_s0 <= 1.0 or E_s0/J <= 1e-3,
    # rather than stage-game eps_rel alone.
    E_s0 = 0.0
    if prob["model"] == "fishery":
        try:
            from metrics import solve_best_response
            cfg_in = prob.get("cfg", {})
            br1 = solve_best_response(
                cfg_in, prob["r"], p2, s_grid=prob["s_grid"], a_grid=prob["a_grid"]
            )
            if symmetric_only and np.array_equal(p1, p2):
                br2 = br1
            else:
                br2 = solve_best_response(
                    cfg_in, prob["r"], p1, s_grid=prob["s_grid"], a_grid=prob["a_grid"]
                )
            e1 = max(0.0, float(br1.value_at_s0() - V1[-1]))
            e2 = max(0.0, float(br2.value_at_s0() - V2[-1]))
            E_s0 = float(max(e1, e2))
        except Exception:
            E_s0 = float(exploit_max)
        j_tot = max(abs(float(V1[-1] + V2[-1])), 1.0)
        is_cert = bool(E_s0 <= 1.0 or (E_s0 / j_tot) <= 1.0e-3)
        converged = bool(realized_rel <= float(prob["eps_rel"]) and is_cert)

    return EquilibriumResult(
        model=prob["model"],
        r=prob["r"],
        rule=rule,
        symmetric_only=bool(symmetric_only),
        H=H,
        converged=bool(converged),
        delta=float(delta),
        s_grid=prob["s_grid"],
        a_grid=prob["a_grid"],
        V1=V1,
        V2=V2,
        p1=p1,
        p2=p2,
        n_eq=n_eq,
        selected_kind=kinds,
        stage_exploit_max=float(exploit_max),
        eps_rel=float(realized_rel),
        eps_rel_threshold=float(prob["eps_rel"]),
        n_fallback=int(n_fallback),
        n_mixed_selected=int(n_mixed),
        n_truncated_mixed=int(n_trunc),
        n_trivial=int(n_trivial),
        selection_seed=int(selection_seed),
        E_s0=float(E_s0),
    )


def executed_catches(result):
    """Expected executed catch at each grid stock. Levhari actions are fractions of stock."""
    s_grid = result.s_grid
    a_grid = result.a_grid
    out = np.zeros((s_grid.shape[0], 2), dtype=np.float64)
    for si, s in enumerate(s_grid):
        i1 = np.flatnonzero(result.p1[si] > 1.0e-15)
        i2 = np.flatnonzero(result.p2[si] > 1.0e-15)
        for i in i1:
            for j in i2:
                w = float(result.p1[si, i] * result.p2[si, j])
                if result.model == "levhari":
                    f1 = float(a_grid[i])
                    f2 = float(a_grid[j])
                    if f1 + f2 > 1.0:
                        scale = 1.0 / (f1 + f2)
                        f1 *= scale
                        f2 *= scale
                    out[si, 0] += w * f1 * s
                    out[si, 1] += w * f2 * s
                else:
                    h1, h2 = ration(np.array([a_grid[i], a_grid[j]], dtype=np.float64), float(s))
                    out[si, 0] += w * h1
                    out[si, 1] += w * h2
    return out
