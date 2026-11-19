"""E12: MB-EQ when the true regrowth rate was never solved in advance.

MB-EQ plays the equilibrium interpolated between the r values solved before training (src/cache.py,
interpolated_equilibrium). In E2-E9 the true r is always one of those values. Here, for r between them, we take the
policy MB-EQ converges to once its posterior concentrates (r_hat = r), evaluate it exactly at the true r, and compare
it with the certified equilibrium solved at that r.

Writes results/exact/e12_unsolved_r.json. Run: python experiments/e12_unsolved_r.py
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)
import exact_check as ex  # noqa: E402
from cache import BenchmarkCache  # noqa: E402
from equilibrium import solve_equilibrium  # noqa: E402
from utils import load_config  # noqa: E402

OUT = ROOT / "results" / "exact" / "e12_unsolved_r.json"


def main():
    cfg = load_config()
    s_grid, a_grid = ex.grids(101, 51)
    cache = BenchmarkCache(cfg)
    cache.ensure_rule("min_total")
    out = []
    for r in (0.55, 0.65, 0.75):
        R1, R2, T = ex.stage_tables(r, s_grid, a_grid)
        Vp, _ = ex.planner(R1, R2, T)
        p1, p2 = cache.interpolated_equilibrium(r, rule="min_total")
        V1, V2 = ex.profile_values(p1, p2, R1, R2, T)
        Vbr, _ = ex.best_response_1(p2, R1, T)
        eq = solve_equilibrium(cfg, r, rule="min_total", symmetric_only=True)
        W1, W2 = ex.profile_values(eq.p1, eq.p2, R1, R2, T)
        Wbr, _ = ex.best_response_1(eq.p2, R1, T)
        row = {"r": r, "J_plan": float(Vp[-1]),
               "J_interp": float(V1[-1] + V2[-1]), "E_interp_s0": float(Vbr[-1] - V1[-1]),
               "E_interp_sup": float(np.max(Vbr - V1)), "G_interp": float(1 - (V1[-1] + V2[-1]) / Vp[-1]),
               "J_mpe": float(W1[-1] + W2[-1]), "E_mpe_sup": float(np.max(Wbr - W1)),
               "G_mpe": float(1 - (W1[-1] + W2[-1]) / Vp[-1])}
        print(json.dumps(row), flush=True)
        out.append(row)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
