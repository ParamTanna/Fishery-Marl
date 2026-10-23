"""Compute greedy 50-step discounted return of the cached planner policy from s0=K.

Uses exact Gauss-Hermite quadrature expectations matching the planner dynamic program.
Outputs results/raw/j50_reference.csv used to trace J_50 in tables and claims logs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cache import BenchmarkCache
from kernels import expected_V, ration_pair
from utils import config_hash, gauss_hermite_normal, git_commit, load_config


def compute_j50_reference():
    cfg = load_config()
    cache = BenchmarkCache(cfg)
    m = cfg["model"]
    K = float(m["K"])
    sigma = float(m["sigma"])
    gamma = float(m["gamma"])
    p = float(m["p"])
    c = float(m["c"])
    half_c = 0.5 * c
    nq = int(cfg["solver"].get("n_quadrature", cfg["grids"].get("n_quad", 21)))
    nodes, weights = gauss_hermite_normal(nq)

    chash = config_hash(cfg)
    commit = git_commit()

    rows = []
    r_values = [0.5, 0.6, 0.7, 0.8, 1.5]
    for r in r_values:
        plan = cache.get_planner(r)
        s_grid = plan.s_grid
        a_grid = plan.a_grid
        n_s = len(s_grid)

        R = np.zeros(n_s, dtype=np.float64)
        x_post = np.zeros(n_s, dtype=np.float64)
        for si, s in enumerate(s_grid):
            ai = a_grid[plan.a1_idx[si]]
            aj = a_grid[plan.a2_idx[si]]
            h1, h2 = ration_pair(ai, aj, s)
            x = s - (h1 + h2)
            R[si] = p * (h1 + h2) - half_c * (h1 * h1 + h2 * h2)
            x_post[si] = x

        V = np.zeros(n_s, dtype=np.float64)
        for _step in range(50):
            V_prev = V.copy()
            for si in range(n_s):
                ev = expected_V(V_prev, x_post[si], float(r), K, sigma, nodes, weights)
                V[si] = R[si] + gamma * ev

        j_50 = float(V[-1])
        print(f"r={r}: J_50={j_50:.6f}", flush=True)
        rows.append({
            "r": float(r),
            "policy": "planner",
            "J_50": j_50,
            "config_hash": chash,
            "git_commit": commit,
        })

    out_df = pd.DataFrame(rows)
    out_path = ROOT / "results" / "raw" / "j50_reference.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(out_df)} rows)", flush=True)


if __name__ == "__main__":
    compute_j50_reference()
