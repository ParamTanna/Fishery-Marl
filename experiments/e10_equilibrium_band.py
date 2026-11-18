"""E10: certified equilibrium band, Price of Anarchy / Stability, and the cost of the reserve cap.

For every tragedy rate r and the symmetric selection rules, the profile returned by the equilibrium solver is
evaluated exactly and certified with an exact best response at every stock (src/exact_check.py). The certified
profiles form the equilibrium band; PoA and PoS compare it with the planner. The same is done for the game with
the reserve cap a_i <= max(0, 0.5 (s - 25)), and for a finer catch grid (101 x 101) as a discretisation check.

Writes results/exact/e10_equilibria.json. Run: python experiments/e10_equilibrium_band.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]
import exact_check as ex  # noqa: E402
from dp_planner import solve_planner  # noqa: E402
from equilibrium import solve_equilibrium  # noqa: E402
from utils import load_config  # noqa: E402
from e7_safe import solve_safe_equilibrium  # noqa: E402

R_VALUES = [0.5, 0.6, 0.7, 0.8]
RULES = ["max_total", "min_total", "max_catch", "continuity"]
S_RES = 25.0
OUT = ROOT / "results" / "exact" / "e10_equilibria.json"


def certify(p1, p2, R1, R2, T, mask=None):
    V1, V2 = ex.profile_values(p1, p2, R1, R2, T)
    Vbr, _ = ex.best_response_1(p2, R1, T, mask)
    gain = Vbr - V1
    return {"J": float(V1[-1] + V2[-1]), "E_s0": float(gain[-1]), "E_sup": float(gain.max()),
            "n_states_gain_gt_1": int((gain > 1).sum()), "certified": ex.certified(gain.max(), V1[-1])}


def main():
    cfg = load_config()
    s_grid, a_grid = ex.grids(101, 51)
    mask = ex.reserve_mask(s_grid, a_grid, S_RES)
    out = {"planner_check": [], "profiles": [], "band": [], "grid_check": []}
    for r in R_VALUES + [1.5]:
        R1, R2, T = ex.stage_tables(r, s_grid, a_grid)
        Vp, _ = ex.planner(R1, R2, T)
        team = solve_planner(cfg, r)
        out["planner_check"].append({"r": r, "J_plan_exact": float(Vp[-1]), "J_plan_solver": team.value_at_s0(),
                                     "rel_diff": float(Vp[-1] / team.value_at_s0() - 1)})
        if r > 1.0:
            continue
        t0 = time.time()
        rows = []
        for rule in RULES:
            eq = solve_equilibrium(cfg, r, rule=rule, symmetric_only=True)
            row = {"r": r, "rule": rule, **certify(eq.p1, eq.p2, R1, R2, T), "solver_E_s0": float(eq.E_s0)}
            rows.append(row)
            print("profile", json.dumps(row), flush=True)
        out["profiles"] += rows
        # planner: split, exploitability, value under the reserve cap
        _, pol = ex.planner(R1, R2, T)
        a1, a2 = pol // a_grid.size, pol % a_grid.size
        p1, p2 = np.eye(a_grid.size)[a1], np.eye(a_grid.size)[a2]
        plan = certify(p1, p2, R1, R2, T)
        Vcap, _ = ex.planner(R1, R2, T, joint_mask=mask[:, :, None] & mask[:, None, :])
        # equilibrium of the capped game (solver), certified with best responses restricted to legal catches
        _, ps, *_ = solve_safe_equilibrium(cfg, r, S_RES, s_grid=s_grid, a_grid=a_grid)
        safe = certify(ps, ps, R1, R2, T, mask)
        cert = {x["rule"]: x["J"] for x in rows if x["certified"]}
        jw, jb = min(cert.values()), max(cert.values())
        band = {"r": r, "J_plan": float(Vp[-1]), "E_plan_s0": plan["E_s0"], "E_plan_sup": plan["E_sup"],
                "certified_rules": sorted(cert), "J_mpe_worst": jw, "J_mpe_best": jb,
                "PoA": float(Vp[-1] / jw), "PoS": float(Vp[-1] / jb),
                "G_worst": float(1 - jw / Vp[-1]), "G_best": float(1 - jb / Vp[-1]),
                "J_plan_capped": float(Vcap[-1]), "J_safe_mpe": safe["J"], "E_safe_mpe_s0": safe["E_s0"],
                "E_safe_mpe_sup": safe["E_sup"], "cap_cost_planner": float(Vp[-1] - Vcap[-1]),
                "cap_cost_equilibrium_vs_best": float(jb - safe["J"]), "seconds": round(time.time() - t0, 1)}
        out["band"].append(band)
        print("band", json.dumps(band), flush=True)
    # discretisation check at r = 0.5: catch grid 101 x 101
    s2, a2g = ex.grids(101, 101)
    R1, R2, T = ex.stage_tables(0.5, s2, a2g)
    Vp2, _ = ex.planner(R1, R2, T)
    for rule in ("max_total", "min_total", "max_catch"):
        eq = solve_equilibrium(cfg, 0.5, rule=rule, symmetric_only=True, n_catch=101)
        row = {"grid": "101x101", "r": 0.5, "rule": rule, "J_plan": float(Vp2[-1]), **certify(eq.p1, eq.p2, R1, R2, T)}
        out["grid_check"].append(row)
        print("grid", json.dumps(row), flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
