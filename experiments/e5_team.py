"""E5: Team-reward IQL and model-based team planner vs selfish learners.

Evaluates team_iql_pair (5000 ep) and team_planner_pair (200 ep) across frozen r in
{0.5, 0.6, 0.7, 0.8} and 20 reported seeds (10..29) at gamma=0.99, sigma=0.1,
and pairs them with selfish IQL / MB-EQ / MB-OA baselines.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import concurrent.futures
import copy
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from cache import BenchmarkCache  # noqa: E402
from learners.team import team_iql_pair, team_planner_pair  # noqa: E402
from learn_common import (  # noqa: E402
    SMOKE_SEEDS,
    episodes_to_G_within,
    grids_of,
    reported_seeds,
    summarize_metrics,
    train_pair,
    worker_init,
    write_rows,
)
from utils import load_config  # noqa: E402


E5_BUDGETS = {
    "team_iql": (
        5000,
        [0, 99, 249, 499, 999, 1499, 1999, 2499, 2999, 3499, 3999, 4499, 4999],
    ),
    "team_planner": (
        200,
        [0, 1, 2, 4, 9, 14, 19, 49, 99, 149, 199],
    ),
}


def _run_one(task):
    learner_name, r, seed = task
    cfg = copy.deepcopy(load_config(quick=False))
    s_grid, a_grid = grids_of(cfg)
    cache = BenchmarkCache(cfg)
    cache.ensure_rule()

    if learner_name == "team_iql":
        a1, a2 = team_iql_pair(s_grid, a_grid, cfg, seed)
    elif learner_name == "team_planner":
        a1, a2 = team_planner_pair(s_grid, a_grid, cfg, cache, seed, true_r=r)
    else:
        raise ValueError(learner_name)

    n_episodes, checkpoints = E5_BUDGETS[learner_name]
    rows, _online, _elapsed, _c, _p, _eq = train_pair(
        a1, a2, cfg, r, seed, n_episodes=n_episodes, eval_every=n_episodes,
        with_gap=True, eval_episodes=checkpoints,
    )
    ep10 = episodes_to_G_within(rows, frac=0.10)
    last_idx = len(rows) - 1
    for i, row in enumerate(rows):
        row["learner"] = learner_name
        row["is_team"] = True
        row["gamma"] = float(cfg["model"]["gamma"])
        row["sigma"] = float(cfg["model"]["sigma"])
        row["is_final"] = (i == last_idx)
        row["ep_to_G_10pct"] = int(ep10)
    return rows


def _run_quick(cfg):
    cache = BenchmarkCache(cfg)
    cache.ensure_rule()
    s_grid, a_grid = grids_of(cfg)
    r = float(cfg["model"]["r_values"][0])
    rows = []
    for seed in SMOKE_SEEDS:
        for name, pair in (
            ("team_iql", team_iql_pair(s_grid, a_grid, cfg, seed)),
            ("team_planner", team_planner_pair(s_grid, a_grid, cfg, cache, seed, true_r=r)),
        ):
            trained, _on, _el, *_rest = train_pair(
                pair[0], pair[1], cfg, r, seed, n_episodes=10, eval_every=5, with_gap=True,
            )
            for i, row in enumerate(trained):
                row["learner"] = name
                row["is_team"] = True
                row["is_final"] = (i == len(trained) - 1)
            rows.extend(trained)
    frame = write_rows(ROOT / "results" / "raw" / "e5_quick.csv", rows)
    print("rows", len(frame), sorted(frame.learner.unique()), flush=True)


def _run_full(workers):
    cfg = load_config(quick=False)
    BenchmarkCache(cfg).ensure_rule()
    r_values = [float(v) for v in cfg["model"]["r_values"]]
    seeds = reported_seeds(cfg)

    tasks = [(ln, r, s) for ln in ("team_iql", "team_planner") for r in r_values for s in seeds]
    print(f"Launching E5 full: {len(tasks)} runs across {workers} workers...", flush=True)
    t0 = time.perf_counter()
    all_rows = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers, initializer=worker_init) as pool:
        for r_rows in pool.map(_run_one, tasks, chunksize=4):
            all_rows.extend(r_rows)

    df = write_rows(ROOT / "results" / "raw" / "e5_team.csv", all_rows)
    final_df = df[df["is_final"] == True].copy()  # noqa: E712
    metrics = [
        "J", "G", "E", "collapse_50", "collapse_long", "cum_collapse_50_train",
        "long_run_stock", "cum_online_return", "L_tau_plan", "L_tau_eq", "ep_to_G_10pct",
    ]
    summary = summarize_metrics(final_df, ["learner", "r", "budget_episodes"], metrics)
    write_rows(ROOT / "results" / "raw" / "e5_summary.csv", summary.to_dict("records"))
    print(f"Wrote {len(df)} rows -> e5_team.csv in {time.perf_counter()-t0:.1f}s", flush=True)


def main():
    parser = argparse.ArgumentParser(description="E5 team baselines")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    if args.full and not args.quick:
        _run_full(workers=args.workers)
    elif args.quick:
        _run_quick(load_config(quick=True))
    else:
        raise SystemExit("pass --quick or --full")


if __name__ == "__main__":
    main()
