"""Stage-game enumeration, planner dominance, E = 0, and a known best response."""

import numpy as np
import pytest

from dp_planner import solve_planner
from env import ration
from equilibrium import enum_sym_mixed, pure_nash_mask, solve_equilibrium
from kernels import planner_sweep_fishery
from metrics import equilibrium_gap, solve_best_response
from utils import gauss_hermite_normal, load_config


def _small_cfg():
    cfg = load_config()
    cfg["grids"]["n_stock"] = 15
    cfg["grids"]["n_catch"] = 9
    cfg["grids"]["n_quad"] = 11
    cfg["solver"]["max_H"] = 3000
    cfg["solver"]["min_H"] = 5
    cfg["solver"]["value_tol"] = 1.0e-3
    cfg["solver"]["value_damp"] = 1.0
    cfg["solver"]["use_nashpy_fallback"] = False
    return cfg


def test_coordination_pure_and_symmetric_mixed():
    # Symmetric coordination: (2, 2) and (1, 1) are pure; the mixture puts weight 1/3 on the first action.
    payoff = np.array([[2.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    U1 = payoff[None, :, :]
    U2 = np.transpose(U1, (0, 2, 1))
    mask = pure_nash_mask(U1, U2, 1.0e-9)
    found = set(zip(*np.nonzero(mask[0])))
    assert found == {(0, 0), (1, 1)}
    _i, _j, prob, value, count, overflow = enum_sym_mixed(U1, 1.0e-9, 8)
    assert int(count[0]) == 1
    assert int(overflow[0]) == 0
    assert abs(float(prob[0, 0]) - 1.0 / 3.0) < 1.0e-8
    assert abs(float(value[0, 0]) - 2.0 / 3.0) < 1.0e-8


def test_prisoners_dilemma_has_only_mutual_defect():
    payoff = np.array([[-1.0, -3.0], [0.0, -2.0]], dtype=np.float64)
    U1 = payoff[None, :, :]
    U2 = np.transpose(U1, (0, 2, 1))
    mask = pure_nash_mask(U1, U2, 1.0e-9)
    found = set(zip(*np.nonzero(mask[0])))
    assert found == {(1, 1)}


def test_selection_rules_on_coordination():
    from equilibrium import _select_one

    payoff = np.array([[2.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    prev = np.zeros(2)
    prev[0] = 1.0

    def cand(i, u):
        p = np.zeros(2)
        p[i] = 1.0
        return {"u1": u, "u2": u, "kind": 0, "sym": True, "i": i, "j": i, "p1": p, "p2": p.copy()}

    mix = np.array([1.0 / 3.0, 2.0 / 3.0])
    cands = [
        cand(0, 2.0),
        cand(1, 1.0),
        {"u1": 2.0 / 3.0, "u2": 2.0 / 3.0, "kind": 2, "sym": True, "i": 0, "j": 1, "p1": mix, "p2": mix.copy()},
    ]
    mx = _select_one(cands, payoff, payoff.T, "max_total", prev, prev, 1, 0, 1.0e-8, True, False, 0, 1.0e-9)
    mn = _select_one(cands, payoff, payoff.T, "min_total", prev, prev, 1, 0, 1.0e-8, True, False, 0, 1.0e-9)
    risk = _select_one(cands, payoff, payoff.T, "risk_dominant", prev, prev, 1, 0, 1.0e-8, True, False, 0, 1.0e-9)
    pareto = _select_one(cands, payoff, payoff.T, "payoff_dominant", prev, prev, 1, 0, 1.0e-8, True, False, 0, 1.0e-9)
    assert mx["i"] == 0 and mx["kind"] == 0
    assert mn["kind"] == 2
    assert risk["i"] == 0 and risk["kind"] == 0
    assert pareto["i"] == 0 and pareto["kind"] == 0
    for item, caught in zip(cands, (1.0, 5.0, 2.0)):
        item["catch"] = caught
    # Previous policy is action 0, so continuity stays there. max_catch takes the larger catch.
    held = _select_one(cands, payoff, payoff.T, "continuity", prev, prev, 1, 0, 1.0e-8, True, False, 0, 1.0e-9)
    caught = _select_one(cands, payoff, payoff.T, "max_catch", prev, prev, 1, 0, 1.0e-8, True, False, 0, 1.0e-9)
    assert held["i"] == 0
    assert caught["i"] == 1


@pytest.fixture(scope="module")
def small_solution():
    cfg = _small_cfg()
    plan = solve_planner(cfg, 0.6, verbose=False)
    eq = solve_equilibrium(cfg, 0.6, rule="max_total", symmetric_only=True, verbose=False)
    return cfg, plan, eq


def test_planner_total_return_beats_equilibrium(small_solution):
    _cfg, plan, eq = small_solution
    assert plan.converged and eq.converged
    assert eq.n_fallback == 0
    assert eq.stage_exploit_max < 1.0e-4
    gap = plan.V - (eq.V1 + eq.V2)
    assert float(gap.min()) >= -1.0e-2, float(gap.min())


def test_equilibrium_has_no_profitable_deviation(small_solution):
    cfg, _plan, eq = small_solution
    br = solve_best_response(cfg, 0.6, eq.p2, s_grid=eq.s_grid, a_grid=eq.a_grid)
    j_gap, sup_gap = equilibrium_gap(eq.V1, br.V)
    assert br.converged
    assert sup_gap < 1.0e-2, (j_gap, sup_gap, eq.stage_exploit_max)
    assert j_gap < 1.0e-2


def test_best_response_recovers_myopic_optimum_against_a_quiet_opponent():
    cfg = _small_cfg()
    cfg["grids"]["n_stock"] = 11
    cfg["grids"]["n_catch"] = 11
    s_grid = np.linspace(0.0, cfg["model"]["K"], cfg["grids"]["n_stock"])
    a_grid = np.linspace(0.0, cfg["grids"]["catch_max"], cfg["grids"]["n_catch"])
    opp = np.zeros((s_grid.size, a_grid.size))
    opp[:, 0] = 1.0
    br = solve_best_response(
        cfg, 0.6, opp, s_grid=s_grid, a_grid=a_grid, gamma=0.0, sigma=0.0, max_H=4, value_tol=1.0e-12
    )
    p = cfg["model"]["p"]
    c = cfg["model"]["c"]
    for s, value, idx in zip(s_grid, br.V, br.action_idx):
        best = -1.0e300
        for a in a_grid:
            hats = ration(np.array([a, 0.0]), float(s))
            reward = p * hats[0] - 0.5 * c * hats[0] ** 2
            best = max(best, float(reward))
        assert abs(float(value) - best) < 1.0e-8
        hats = ration(np.array([a_grid[idx], 0.0]), float(s))
        got = p * hats[0] - 0.5 * c * hats[0] ** 2
        assert abs(float(got) - best) < 1.0e-8


def test_planner_matches_an_independent_python_bellman_when_sigma_is_zero():
    cfg = load_config()
    cfg["model"]["sigma"] = 0.0
    cfg["grids"]["n_stock"] = 8
    cfg["grids"]["n_catch"] = 6
    cfg["grids"]["n_quad"] = 5
    cfg["solver"]["max_H"] = 20
    cfg["solver"]["min_H"] = 20
    cfg["solver"]["value_tol"] = 0.0
    plan = solve_planner(cfg, 0.5, verbose=False)
    s_grid, a_grid = plan.s_grid, plan.a_grid
    r = 0.5
    K = cfg["model"]["K"]
    gamma = cfg["model"]["gamma"]
    p = cfg["model"]["p"]
    c = cfg["model"]["c"]
    V = np.zeros_like(s_grid)
    for _ in range(20):
        Vn = np.empty_like(V)
        for si, s in enumerate(s_grid):
            best = -1.0e300
            for a1 in a_grid:
                for a2 in a_grid:
                    hats = ration(np.array([a1, a2]), float(s))
                    x = float(s - hats.sum())
                    gr = x * (1.0 + r * (1.0 - x / K))
                    sp = min(K, max(gr, 0.0))
                    ev = float(np.interp(sp, s_grid, V))
                    val = p * float(hats.sum()) - 0.5 * c * float(np.square(hats).sum()) + gamma * ev
                    if val > best:
                        best = val
            Vn[si] = best
        V = Vn
    assert np.allclose(plan.V, V, atol=1.0e-8)


def test_default_parameters_have_a_tragedy():
    from utils import tragedy_margin

    cfg = load_config()
    rows = tragedy_margin(cfg)
    assert [row["r"] for row in rows] == [0.5, 0.6, 0.7, 0.8]
    assert min(row["slack"] for row in rows) > 0.0
    # r = 1.5 is the no-tragedy sanity check and must not be in the asserted set.
    sanity = [float(r) for r in cfg["model"]["r_sanity"]]
    assert sanity == [1.5]
    try:
        tragedy_margin(cfg, sanity)
    except AssertionError:
        return
    raise AssertionError("r=1.5 should fail the tragedy assertion at the frozen p/c")


def test_numba_planner_one_sweep_is_finite():
    s_grid = np.linspace(0.0, 100.0, 5)
    a_grid = np.linspace(0.0, 100.0, 4)
    nodes, weights = gauss_hermite_normal(7)
    V, bi, bj = planner_sweep_fishery(
        np.zeros(5), s_grid, a_grid, 0.6, 100.0, 0.1, 0.95, 10.0, 0.2, nodes, weights
    )
    assert np.all(np.isfinite(V))
    assert bi.shape == (5,) and bj.shape == (5,)


def test_mpe_certificate_on_small_grid_and_non_equilibrium():
    from metrics import policy_values, solve_best_response
    from utils import stock_catch_grids

    cfg = _small_cfg()
    eq = solve_equilibrium(cfg, 0.6, rule="min_total", symmetric_only=True, verbose=False)
    assert eq.converged
    assert eq.E_s0 <= 1.0 or (eq.E_s0 / max(eq.V1[-1] + eq.V2[-1], 1.0)) <= 1.0e-3

    # A deliberately non-equilibrium profile (e.g. both fleets harvest maximum) has E > 0
    s_grid, a_grid = stock_catch_grids(cfg)
    p_bad = np.zeros((s_grid.shape[0], a_grid.shape[0]), dtype=np.float64)
    p_bad[:, -1] = 1.0
    br = solve_best_response(cfg, 0.6, p_bad, s_grid=s_grid, a_grid=a_grid)
    v1_bad, _v2_bad, _ok, _delta = policy_values(cfg, 0.6, p_bad, p_bad, s_grid, a_grid)
    e_bad = float(br.value_at_s0() - v1_bad[-1])
    assert e_bad > 5.0, f"Expected non-equilibrium profile to have large exploitability, got {e_bad}"

