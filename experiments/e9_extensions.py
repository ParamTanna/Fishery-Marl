"""E9 (Step 6 Extensions): Tabular CTDE & Memory-One cooperation/punishment probe.

Evaluates two tabular extensions tied directly to Research Questions 2 and 4:
1. CTDEFleet: Joint-action Q_i(s, a_1, a_2) centralised critic with decentralised
   greedy execution pi_i(a_i | s), tested both unconstrained and safe (S_res=25.0).
2. MemoryOneFleet: Finite-history Q_i(s_t, regime_t, a_i) conditioning on prior
   partner over-harvest, evaluated via forced one-season over-harvest shock probe.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from learn_common import (  # noqa: E402
    SMOKE_SEEDS,
    episodes_to_G_within,
    grids_of,
    prepare,
    reported_seeds,
    summarize_metrics,
    train_pair,
    worker_init,
)
from learners.ctde import ctde_pair  # noqa: E402
from learners.memory_one import forced_overharvest_probe, memory_one_pair  # noqa: E402
from training import make_env  # noqa: E402
from utils import load_config  # noqa: E402

RAW_DIR = ROOT / "results" / "raw"
TRAGEDY_R = (0.5, 0.6, 0.7, 0.8)


def _train_ctde_one(task):
    r, seed, safe, n_episodes, n_eval_rollouts, n_eval_points = task
    cfg = load_config()
    if n_eval_rollouts is not None:
        cfg["evaluation"]["n_rollouts"] = int(n_eval_rollouts)
    s_grid, a_grid = grids_of(cfg)
    s_res = 25.0 if safe else None
    a1, a2 = ctde_pair(s_grid, a_grid, cfg, seed=seed, s_res=s_res, epsilon_decay=0.995)
    eval_eps = sorted({0, int(n_episodes) - 1} | set(
        np.linspace(0, int(n_episodes) - 1, min(int(n_eval_points), int(n_episodes)), dtype=int).tolist()
    ))
    rows, _online, _elapsed, _cache, _plan, _eq = train_pair(
        a1, a2, cfg, r, seed, n_episodes=int(n_episodes), eval_every=1,
        with_gap=True, rule=cfg["model"]["reference_rule"], eval_episodes=eval_eps,
    )
    ep10 = episodes_to_G_within(rows, frac=0.10)
    for idx, row in enumerate(rows):
        row["learner"] = "ctde_safe" if safe else "ctde"
        row["safe"] = bool(safe)
        row["s_res"] = 25.0 if safe else np.nan
        row["is_final"] = (idx == len(rows) - 1)
        row["ep_to_G_10pct"] = ep10
    return rows


def _train_memory_one_one(task):
    r, seed, safe, n_episodes, n_eval_rollouts = task
    cfg = load_config()
    if n_eval_rollouts is not None:
        cfg["evaluation"]["n_rollouts"] = int(n_eval_rollouts)
    s_grid, a_grid = grids_of(cfg)
    s_res = 25.0 if safe else None
    a1, a2 = memory_one_pair(s_grid, a_grid, cfg, seed=seed, s_res=s_res, epsilon_decay=0.995)
    rows, _online, _elapsed, _cache, _plan, _eq = train_pair(
        a1, a2, cfg, r, seed, n_episodes=int(n_episodes), eval_every=int(n_episodes),
        with_gap=True, rule=cfg["model"]["reference_rule"], eval_episodes=[int(n_episodes) - 1],
    )
    env = make_env(cfg, r, seed)
    probe = forced_overharvest_probe(a1, a2, env, shock_t=10, horizon=25, shock_request=15.0)
    row = rows[-1]
    row["learner"] = "memory_one_safe" if safe else "memory_one"
    row["safe"] = bool(safe)
    row["s_res"] = 25.0 if safe else np.nan
    row["is_final"] = True
    row["reaction_gap_above_25"] = float(a2.policy_reaction_gap(min_stock=25.0))
    row.update(probe)
    return [row]


def main():
    parser = argparse.ArgumentParser(description="E9: Step 6 tabular CTDE & MemoryOne extensions.")
    parser.add_argument("--quick", action="store_true", help="Smoke run on seeds [100, 101].")
    parser.add_argument("--full", action="store_true", help="Reported 20-seed run on seeds 10..29.")
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    cfg = load_config()
    if not args.quick and not args.full:
        args.full = True

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if args.quick:
        seeds = list(SMOKE_SEEDS)
        r_vals = (0.5, 0.6)
        n_ep = 60
        n_rollouts = 40
        n_pts = 3
    else:
        seeds = reported_seeds(cfg)
        r_vals = TRAGEDY_R
        n_ep = 1000  # Step 1/6 cut from 5000 ep; epsilon_decay=0.995 reaches eps_min=0.05 by ep ~600
        n_rollouts = None
        n_pts = 11

    ctde_tasks = [
        (r, s, safe, n_ep, n_rollouts, n_pts)
        for safe in (False, True)
        for r in r_vals
        for s in seeds
    ]
    m1_tasks = [
        (r, s, safe, n_ep, n_rollouts)
        for safe in (False, True)
        for r in r_vals
        for s in seeds
    ]

    t_start = time.perf_counter()
    ctde_rows, m1_rows = [], []
    with ProcessPoolExecutor(max_workers=int(args.workers), initializer=worker_init) as ex:
        futs_c = [ex.submit(_train_ctde_one, t) for t in ctde_tasks]
        futs_m = [ex.submit(_train_memory_one_one, t) for t in m1_tasks]
        for f in as_completed(futs_c):
            ctde_rows.extend(f.result())
        for f in as_completed(futs_m):
            m1_rows.extend(f.result())

    df_c = pd.DataFrame(ctde_rows).sort_values(["learner", "r", "seed", "episode"]).reset_index(drop=True)
    df_m = pd.DataFrame(m1_rows).sort_values(["learner", "r", "seed"]).reset_index(drop=True)

    if args.quick:
        df_c.to_csv(RAW_DIR / "e9_ctde_quick.csv", index=False)
        df_m.to_csv(RAW_DIR / "e9_memory_one_quick.csv", index=False)
        print(f"e9 --quick done in {time.perf_counter() - t_start:.1f}s (ctde={len(df_c)}, m1={len(df_m)})", flush=True)
        return

    df_c.to_csv(RAW_DIR / "e9_ctde.csv", index=False)
    df_m.to_csv(RAW_DIR / "e9_memory_one.csv", index=False)

    fin_c = df_c[df_c["is_final"] == True].copy()  # noqa: E712
    sum_c = summarize_metrics(
        fin_c,
        ["learner", "safe", "s_res", "r", "budget_episodes"],
        [
            "J", "G", "E", "collapse_50", "collapse_long",
            "cum_collapse_50_train", "long_run_stock", "L_tau_plan", "L_tau_eq", "ep_to_G_10pct",
        ],
    )
    sum_c.to_csv(RAW_DIR / "e9_ctde_summary.csv", index=False)

    sum_m = summarize_metrics(
        df_m,
        ["learner", "safe", "s_res", "r", "budget_episodes"],
        [
            "J", "G", "E", "collapse_50", "cum_collapse_50_train", "long_run_stock",
            "reaction_gap_above_25", "regime2_post_shock",
            "catch2_pre_shock", "catch2_post_1_shock", "catch2_post_1_base",
            "delta_catch2_post_1", "delta_catch2_post_6",
        ],
    )
    sum_m.to_csv(RAW_DIR / "e9_memory_one_summary.csv", index=False)
    print(f"e9 --full done in {time.perf_counter() - t_start:.1f}s (ctde={len(df_c)}, m1={len(df_m)})", flush=True)


if __name__ == "__main__":
    main()
