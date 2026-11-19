"""E11: can cooperation be sustained? Exact grim-trigger threshold gamma*(r).

Cooperative profile: the symmetric planner (both fleets take the same catch, chosen to maximise the total return).
Grim trigger: keep the cooperative catch while everyone has; after any deviation play a certified Markov perfect
equilibrium (minimum-total rule) forever. The punishment phase is itself an MPE, so by the one-shot deviation
principle the grim trigger is subgame perfect iff, at every stock and for each fleet,
    V_i^coop(s) >= max_{a_i} [ R_i(s, a_i, a_j^coop(s)) + gamma * E V_i^MPE(s') ].
The condition is checked exactly on a grid of gamma (the MPE is re-solved at each gamma) and the smallest gamma at
which it holds is refined by bisection.

Writes results/exact/e11_cooperation.json. Run: python experiments/e11_cooperation_threshold.py [r ...]
"""
import copy
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import exact_check as ex  # noqa: E402
from equilibrium import solve_equilibrium  # noqa: E402
from utils import load_config  # noqa: E402

GAMMA_GRID = [0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.90, 0.99]
OUT = ROOT / "results" / "exact" / "e11_cooperation.json"
CFG = load_config()
S_GRID, A_GRID = ex.grids(101, 51)


def check(r, g, R1, R2, T):
    ex.GAMMA = g
    Vc, pol = ex.planner(R1, R2, T, symmetric=True)
    pc = np.eye(A_GRID.size)[pol]
    V1c, V2c = ex.profile_values(pc, pc, R1, R2, T)
    cfg = copy.deepcopy(CFG)
    cfg["model"]["gamma"] = float(g)
    eq = solve_equilibrium(cfg, r, rule="min_total", symmetric_only=True)
    V1m, V2m = ex.profile_values(eq.p1, eq.p2, R1, R2, T)
    Vbr, _ = ex.best_response_1(eq.p2, R1, T)
    margin = V1c - ex.deviation_value_1(pc, R1, T, V1m)      # >= 0 everywhere <=> no profitable deviation
    return {"r": r, "gamma": g, "sustainable": bool(margin.min() >= -1e-6), "min_margin": float(margin.min()),
            "argmin_stock": float(S_GRID[int(margin.argmin())]), "n_states_violated": int((margin < -1e-6).sum()),
            "J_coop": float(V1c[-1] + V2c[-1]), "J_mpe": float(V1m[-1] + V2m[-1]),
            "E_sup_punishment": float(np.max(Vbr - V1m)),
            "punishment_certified": ex.certified(np.max(Vbr - V1m), V1m[-1]),
            "coop_catch_at_K": float(2 * A_GRID[pol[-1]])}


def solve_r(r):
    t0 = time.time()
    R1, R2, T = ex.stage_tables(r, S_GRID, A_GRID)
    rows = []
    for g in GAMMA_GRID:
        rows.append(check(r, g, R1, R2, T))
        print(json.dumps(rows[-1]), flush=True)
    ok = [x["gamma"] for x in rows if x["sustainable"]]
    bad = [x["gamma"] for x in rows if not x["sustainable"] and (not ok or x["gamma"] < min(ok))]
    lo, hi = (max(bad) if bad else None), (min(ok) if ok else None)
    if lo is not None and hi is not None:
        for _ in range(4):                            # bisection on gamma
            mid = round((lo + hi) / 2, 4)
            x = check(r, mid, R1, R2, T)
            rows.append(x)
            print(json.dumps(x), flush=True)
            lo, hi = (lo, mid) if x["sustainable"] else (mid, hi)
    print(f"r={r}: gamma* in ({lo}, {hi}]", flush=True)
    return {"r": r, "gamma_star_lower": lo, "gamma_star_upper": hi,
            "rows": sorted(rows, key=lambda x: x["gamma"]), "seconds": round(time.time() - t0, 1)}


def main(r_values):
    with Pool(len(r_values)) as pool:                 # one process per regrowth rate
        results = pool.map(solve_r, r_values)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(sorted(results, key=lambda x: x["r"]), open(OUT, "w"), indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main([float(x) for x in sys.argv[1:]] or [0.5, 0.6, 0.7, 0.8])
