"""E2: where learners settle across r, gamma, sigma, and whether the method matters.

--quick: smoke run on seeds (100, 101) writing results/raw/e2_quick.csv.
--full:  20 reported seeds (10..29), 4 r x 3 gamma x 3 sigma + sanity r=1.5 at (0.99, 0.1),
         distinct budgets (IQL=5000, MB-EQ=500, MB-OA=200), 18-20 eval checkpoints per run,
         writes results/raw/e2_learners.csv and results/raw/e2_summary.csv.
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

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from cache import BenchmarkCache  # noqa: E402
from learners.iql import iql_pair  # noqa: E402
from learners.mb_eq import mbeq_pair  # noqa: E402
from learners.mb_oa import mboa_pair  # noqa: E402
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
from metrics import one_hot_from_index, solve_best_response, steady_stock  # noqa: E402
from policies import overharvester  # noqa: E402
from utils import load_config  # noqa: E402

E2_BUDGETS = {
    "iql": (
        5000,
        [0, 9, 49, 99, 249, 499, 749, 999, 1249, 1499, 1749, 1999, 2499, 2999, 3499, 3999, 4499, 4999],
    ),
    "mb_eq": (
        500,
        [0, 1, 2, 4, 9, 14, 19, 24, 34, 49, 74, 99, 149, 199, 249, 299, 349, 399, 449, 499],
    ),
    "mb_oa": (
        200,
        [0, 1, 2, 4, 9, 14, 19, 29, 39, 49, 69, 89, 109, 129, 149, 169, 179, 189, 199],
    ),
}


def _probe_br(cfg):
    s_grid, a_grid = grids_of(cfg)
    p = float(cfg["model"]["p"])
    c = float(cfg["model"]["c"])
    r = float(cfg["model"]["r_values"][0])
    opp, _ = overharvester(s_grid, a_grid, p / c)
    t0 = time.perf_counter()
    solve_best_response(cfg, r, opp, s_grid=s_grid, a_grid=a_grid)
    return time.perf_counter() - t0


def _run_one(task):
    learner_name, r, gamma, sigma, seed, is_sanity = task
    cfg = copy.deepcopy(load_config(quick=False))
    cfg["model"]["gamma"] = float(gamma)
    cfg["model"]["sigma"] = float(sigma)
    s_grid, a_grid = grids_of(cfg)
    cache = BenchmarkCache(cfg)
    cache.ensure_rule()

    if learner_name == "iql":
        a1, a2 = iql_pair(s_grid, a_grid, cfg, seed)
    elif learner_name == "mb_eq":
        a1, a2 = mbeq_pair(s_grid, a_grid, cfg, cache, seed, true_r=r)
    elif learner_name == "mb_oa":
        a1, a2 = mboa_pair(s_grid, a_grid, cfg, cache, seed, true_r=r)
    else:
        raise ValueError(learner_name)

    n_episodes, checkpoints = E2_BUDGETS[learner_name]
    rows, _online, elapsed, cache, plan, eq = train_pair(
        a1, a2, cfg, r, seed, n_episodes=n_episodes, eval_every=n_episodes,
        with_gap=True, eval_episodes=checkpoints,
    )
    ep_to_10pct = episodes_to_G_within(rows, frac=0.10)
    plan_p1 = one_hot_from_index(plan.a1_idx, a_grid.shape[0])
    plan_p2 = one_hot_from_index(plan.a2_idx, a_grid.shape[0])
    steady_p, _ = steady_stock(s_grid, a_grid, plan_p1, plan_p2, r, cfg["model"]["K"], cfg["model"]["s0"])
    steady_e, _ = steady_stock(s_grid, a_grid, eq.p1, eq.p2, r, cfg["model"]["K"], cfg["model"]["s0"])

    last_idx = len(rows) - 1
    for i, row in enumerate(rows):
        row["learner"] = learner_name
        row["gamma"] = float(gamma)
        row["sigma"] = float(sigma)
        row["rule"] = str(cfg["model"].get("reference_rule", "max_total"))
        row["is_sanity"] = bool(is_sanity)
        row["is_final"] = (i == last_idx)
        row["ep_to_G_10pct"] = int(ep_to_10pct)
        row["plan_steady_stock"] = float(steady_p)
        row["eq_steady_stock"] = float(steady_e)
    return rows


def _run_quick(cfg):
    cache = BenchmarkCache(cfg)
    cache.ensure_rule()
    s_grid, a_grid = grids_of(cfg)
    n_episodes = 12
    eval_every = 4
    rows = []
    br_seconds = _probe_br(cfg)
    print(f"one best response on this grid: {br_seconds:.3f}s", flush=True)
    builders = [
        ("iql", lambda seed, r: iql_pair(s_grid, a_grid, cfg, seed)),
        ("mb_eq", lambda seed, r: mbeq_pair(s_grid, a_grid, cfg, cache, seed, true_r=r)),
        ("mb_oa", lambda seed, r: mboa_pair(s_grid, a_grid, cfg, cache, seed, true_r=r)),
    ]
    for name, build in builders:
        for seed in SMOKE_SEEDS:
            r = float(cfg["model"]["r_values"][0])
            pair = build(seed, r)
            trained, _online, elapsed, _cache, _plan, _eq = train_pair(
                pair[0], pair[1], cfg, r, seed, n_episodes, eval_every, with_gap=True,
            )
            ep10 = episodes_to_G_within(trained, frac=0.10)
            for i, row in enumerate(trained):
                row["learner"] = name
                row["gamma"] = float(cfg["model"]["gamma"])
                row["sigma"] = float(cfg["model"]["sigma"])
                row["is_final"] = (i == len(trained) - 1)
                row["ep_to_G_10pct"] = int(ep10)
            rows.extend(trained)
            print(
                f"  {name} seed={seed} {elapsed:.2f}s total, {elapsed / n_episodes:.3f}s/episode",
                flush=True,
            )
    frame = write_rows(ROOT / "results" / "raw" / "e2_quick.csv", rows)
    print("rows", len(frame), "learners", sorted(frame.learner.unique()), flush=True)


def _run_full(workers, gammas=None, sigmas=None, r_list=None, include_sanity=True):
    cfg = load_config(quick=False)
    gammas = [0.9, 0.95, 0.99] if gammas is None else [float(g) for g in gammas]
    sigmas = [0.05, 0.1, 0.2] if sigmas is None else [float(s) for s in sigmas]
    r_values = [float(v) for v in cfg["model"]["r_values"]] if r_list is None else [float(v) for v in r_list]
    r_sanity = [float(v) for v in cfg["model"].get("r_sanity", [])] if include_sanity else []
    seeds = reported_seeds(cfg)

    # Ensure cache is warmed up in parent before workers spawn.
    cells = []
    for gamma in gammas:
        for sigma in sigmas:
            for r in r_values:
                cells.append((r, gamma, sigma, False))
    for r in r_sanity:
        if 0.99 in gammas and 0.1 in sigmas:
            cells.append((r, 0.99, 0.1, True))

    for gamma, sigma in sorted({(c[1], c[2]) for c in cells}):
        sub_cfg = copy.deepcopy(cfg)
        sub_cfg["model"]["gamma"] = gamma
        sub_cfg["model"]["sigma"] = sigma
        BenchmarkCache(sub_cfg).ensure_rule()

    tasks = []
    # Run MB-OA -> MB-EQ -> IQL interleaved by cell for steady throughput.
    for learner_name in ("iql", "mb_eq", "mb_oa"):
        for r, gamma, sigma, is_san in cells:
            for seed in seeds:
                tasks.append((learner_name, r, gamma, sigma, seed, is_san))

    print(
        f"Launching E2 full: {len(tasks)} runs across {workers} workers "
        f"({len(cells)} cells x 3 learners x {len(seeds)} seeds)...",
        flush=True,
    )
    t0 = time.perf_counter()
    all_rows = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers, initializer=worker_init) as pool:
        for i, res_rows in enumerate(pool.map(_run_one, tasks, chunksize=4), 1):
            all_rows.extend(res_rows)
            if i % 100 == 0 or i == len(tasks):
                dt = time.perf_counter() - t0
                rate = i / max(dt, 1e-6)
                rem = (len(tasks) - i) / max(rate, 1e-6)
                print(
                    f"  E2 progress: {i}/{len(tasks)} runs ({dt / 60:.1f}m elapsed, ~{rem / 60:.1f}m left)",
                    flush=True,
                )

    frame = write_rows(ROOT / "results" / "raw" / "e2_learners.csv", all_rows)
    final_df = frame[frame["is_final"] == True].copy()  # noqa: E712
    metrics = [
        "J", "J1", "J2", "G", "E", "E1", "E2", "collapse_50", "collapse_long",
        "cum_collapse_50_train", "long_run_stock", "cum_online_return", "tail20_online_return",
        "L_tau_plan", "L_tau_eq", "ep_to_G_10pct",
    ]
    summary = summarize_metrics(final_df, ["learner", "is_sanity", "r", "gamma", "sigma", "budget_episodes"], metrics)
    write_rows(ROOT / "results" / "raw" / "e2_summary.csv", summary.to_dict("records"))
    print(f"Wrote {len(frame)} rows -> e2_learners.csv and {len(summary)} rows -> e2_summary.csv in {(time.perf_counter()-t0)/60:.2f}m", flush=True)


def main():
    parser = argparse.ArgumentParser(description="E2 learner comparison")
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
