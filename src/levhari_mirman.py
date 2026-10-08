"""Closed-form check on the Levhari-Mirman fish war.

Planner total harvest: (1 - alpha * gamma) * s
Symmetric equilibrium, each of n fleets:
    (1 - alpha * gamma) / (n - (n - 1) * alpha * gamma) * s

The same stage-game solver is used (pure Nash profiles, symmetric 2-support
mixtures, max_total, symmetric fleets only). A profile is feasible only when
the catches sum to at most the stock. That is the Levhari-Mirman constraint.
Proportional rationing is a different institution and is not used here.

For every stationary constant-fraction policy the value is affine in log(s),
with slope 1/(1 - alpha * gamma) per fleet. The level of that line does not
change the best reply. The equilibrium check therefore takes one backward-
induction step from the continuation V(s) = log(s) / (1 - alpha * gamma).
The planner is ordinary value iteration from V = 0.

The stock grid starts at s_max / n_stock rather than 0, because log utility is
undefined at an empty lake. States within `edge_skip` of either end are left
out of the reported error.
"""

from __future__ import annotations

import copy

import numpy as np

from dp_planner import solve_planner
from equilibrium import executed_catches, solve_equilibrium


def closed_forms(alpha, gamma, n_fleets):
    total_plan = 1.0 - alpha * gamma
    per_fleet = total_plan / (n_fleets - (n_fleets - 1) * alpha * gamma)
    return float(total_plan), float(per_fleet)


def validate_levhari(cfg, verbose=False):
    L = cfg["levhari_mirman"]
    alpha = float(L["alpha"])
    gamma = float(L["gamma"])
    n = int(L["n_fleets"])
    plan = solve_planner(cfg, alpha, verbose=verbose, model="levhari")
    # Slope of any stationary constant-fraction value. See the module docstring.
    slope = 1.0 / (1.0 - alpha * gamma)
    eq_cfg = copy.deepcopy(cfg)
    eq_cfg["levhari_mirman"]["max_H"] = 1
    eq_cfg["levhari_mirman"]["value_tol"] = 0.0
    eq_cfg["levhari_mirman"]["value_damp"] = 1.0
    eq = solve_equilibrium(
        eq_cfg,
        alpha,
        rule="max_total",
        symmetric_only=True,
        model="levhari",
        verbose=verbose,
        v_init=slope * np.log(plan.s_grid),
    )
    f_plan, f_eq = closed_forms(alpha, gamma, n)
    plan_frac = plan.total_executed / plan.s_grid
    catches = executed_catches(eq)
    eq_frac_1 = catches[:, 0] / eq.s_grid
    eq_frac_2 = catches[:, 1] / eq.s_grid
    edge = int(L["edge_skip"])
    sl = slice(edge, -edge if edge else None)
    rel_plan = np.abs(plan_frac[sl] - f_plan) / abs(f_plan)
    rel_1 = np.abs(eq_frac_1[sl] - f_eq) / abs(f_eq)
    rel_2 = np.abs(eq_frac_2[sl] - f_eq) / abs(f_eq)
    # Interior relative error, per state, is the worst of the planner total and each fleet.
    per_state = np.maximum(rel_plan, np.maximum(rel_1, rel_2))
    worst = int(np.argmax(per_state))
    s_interior = eq.s_grid[sl]
    return {
        "alpha": alpha,
        "gamma": gamma,
        "n_fleets": n,
        "n_stock": int(plan.s_grid.shape[0]),
        "n_fraction": int(plan.a_grid.shape[0]),
        "s_min": float(plan.s_grid[0]),
        "s_max": float(plan.s_grid[-1]),
        "edge_skip": edge,
        "f_plan": f_plan,
        "f_eq": f_eq,
        "max_rel_error": float(per_state.max()),
        "max_rel_plan": float(rel_plan.max()),
        "max_rel_eq": float(max(rel_1.max(), rel_2.max())),
        "worst_stock": float(s_interior[worst]),
        "worst_plan_fraction": float(plan_frac[sl][worst]),
        "worst_eq_fraction": float(eq_frac_1[sl][worst]),
        "planner_H": plan.H,
        "planner_converged": plan.converged,
        "eq_H": eq.H,
        "eq_converged": eq.converged,
        "eq_stage_exploit": eq.stage_exploit_max,
        "eq_n_fallback": eq.n_fallback,
        "passed": bool(
            per_state.max() < float(L["rel_error_tol"])
            and plan.converged
            and eq.n_fallback == 0
            and eq.stage_exploit_max < 1.0e-6
        ),
    }
