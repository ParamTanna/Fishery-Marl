"""E6: Effect of data amount (N_warm, N_replan, prior width) on model-based learners.

Sweeps N_warm in {0, 5, 20, 50}, N_replan in {1, 5, 20}, and two prior widths
(wide [0.1, 2.0] vs narrow [0.4, 1.0]) on MBOAFleet and MBEqFleet across frozen
r in {0.5, 0.6, 0.7, 0.8} and 20 reported seeds (10..29).
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from cache import BenchmarkCache  # noqa: E402
from learners.mb_eq import mbeq_pair  # noqa: E402
from learners.mb_oa import mboa_pair  # noqa: E402
from learn_common import (  # noqa: E402
    SMOKE_SEEDS,
    grids_of,
    reported_seeds,
    summarize_metrics,
    train_pair,
    worker_init,
    write_rows,
)
from utils import load_config  # noqa: E402


PRIORS = {
    "wide": {"r_min": 0.1, "r_max": 2.0, "n_grid": 191},
    "narrow": {"r_min": 0.4, "r_max": 1.0, "n_grid": 61},
}
N_WARM_GRID = (0, 5, 20, 50)
N_REPLAN_GRID = (1, 5, 20)
E6_EPISODES = 100


def _apply_design(cfg, n_warm, n_replan, prior_name):
    out = copy.deepcopy(cfg)
    out["learners"]["n_warm"] = int(n_warm)
    out["learners"]["n_replan"] = int(n_replan)
    spec = PRIORS[prior_name]
    out["belief"]["r_min"] = float(spec["r_min"])
    out["belief"]["r_max"] = float(spec["r_max"])
    out["belief"]["n_grid"] = int(spec["n_grid"])
    return out


def _run_one(task):
    learner_name, r, seed, n_warm, n_replan, prior_name, n_episodes = task
    cfg = _apply_design(load_config(quick=False), n_warm, n_replan, prior_name)
    s_grid, a_grid = grids_of(cfg)
    cache = BenchmarkCache(cfg)
    cache.ensure_rule()

    if learner_name == "mb_eq":
        a1, a2 = mbeq_pair(s_grid, a_grid, cfg, cache, seed, true_r=r)
    elif learner_name == "mb_oa":
        a1, a2 = mboa_pair(s_grid, a_grid, cfg, cache, seed, true_r=r, recover_requests=True)
    else:
        raise ValueError(learner_name)

    rows, _online, _elapsed, _c, _p, _eq = train_pair(
        a1, a2, cfg, r, seed, n_episodes=n_episodes, eval_every=n_episodes,
        with_gap=True, eval_episodes=[n_episodes - 1],
    )
    T = int(cfg["model"]["T"])
    post_warm_per_ep = max(0, T - int(n_warm))
    replans_per_ep = 0 if post_warm_per_ep == 0 else (
        1 + max(0, (T - 1 - int(n_warm)) // max(1, int(n_replan)))
        if int(n_warm) > 0
        else 1 + max(0, (T - 1) // max(1, int(n_replan)))
    )
    for row in rows:
        row["learner"] = learner_name
        row["n_warm"] = int(n_warm)
        row["n_replan"] = int(n_replan)
        row["prior"] = prior_name
        row["prior_width"] = float(PRIORS[prior_name]["r_max"] - PRIORS[prior_name]["r_min"])
        row["seasons_total"] = int(n_episodes) * T
        row["warm_seasons_total"] = int(n_episodes) * min(int(n_warm), T)
        row["post_warm_seasons_total"] = int(n_episodes) * post_warm_per_ep
        row["replans_per_ep"] = int(replans_per_ep)
    return rows


def _run_quick(cfg):
    s_grid, a_grid = grids_of(cfg)
    cache = BenchmarkCache(cfg)
    cache.ensure_rule()
    r = float(cfg["model"]["r_values"][0])
    rows = []
    for nw, nr in ((0, 5), (5, 5)):
        for prior_name in ("wide", "narrow"):
            sc = _apply_design(cfg, nw, nr, prior_name)
            for seed in SMOKE_SEEDS:
                a1, a2 = mboa_pair(s_grid, a_grid, sc, cache, seed, true_r=r)
                trained, *_rest = train_pair(
                    a1, a2, sc, r, seed, n_episodes=8, eval_every=8, with_gap=True, eval_episodes=[7],
                )
                for row in trained:
                    row["learner"] = "mb_oa"
                    row["n_warm"] = nw
                    row["n_replan"] = nr
                    row["prior"] = prior_name
                rows.extend(trained)
    frame = write_rows(ROOT / "results" / "raw" / "e6_quick.csv", rows)
    print("rows", len(frame), flush=True)


def _run_full(workers):
    cfg = load_config(quick=False)
    BenchmarkCache(cfg).ensure_rule()
    r_values = [float(v) for v in cfg["model"]["r_values"]]
    seeds = reported_seeds(cfg)

    tasks = []
    for learner_name in ("mb_eq", "mb_oa"):
        for nw in N_WARM_GRID:
            for nr in N_REPLAN_GRID:
                for prior_name in ("wide", "narrow"):
                    for r in r_values:
                        for seed in seeds:
                            tasks.append((learner_name, r, seed, nw, nr, prior_name, E6_EPISODES))

    print(f"Launching E6 full: {len(tasks)} runs across {workers} workers...", flush=True)
    t0 = time.perf_counter()
    all_rows = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers, initializer=worker_init) as pool:
        for i, r_rows in enumerate(pool.map(_run_one, tasks, chunksize=4), 1):
            all_rows.extend(r_rows)
            if i % 240 == 0 or i == len(tasks):
                dt = time.perf_counter() - t0
                rem = (len(tasks) - i) / max(i / max(dt, 1e-6), 1e-6)
                print(f"  E6 progress: {i}/{len(tasks)} ({dt/60:.1f}m elapsed, ~{rem/60:.1f}m left)", flush=True)

    df = write_rows(ROOT / "results" / "raw" / "e6_data_amount.csv", all_rows)
    metrics = [
        "J", "G", "E", "collapse_50", "collapse_long", "cum_collapse_50_train",
        "long_run_stock", "cum_online_return", "L_tau_plan", "L_tau_eq",
    ]
    summary = summarize_metrics(df, ["learner", "n_warm", "n_replan", "prior", "prior_width", "r"], metrics)
    write_rows(ROOT / "results" / "raw" / "e6_summary.csv", summary.to_dict("records"))
    print(f"Wrote {len(df)} rows -> e6_data_amount.csv in {(time.perf_counter()-t0)/60:.2f}m", flush=True)


def main():
    parser = argparse.ArgumentParser(description="E6 data amount and prior sensitivity")
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
