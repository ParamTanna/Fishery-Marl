"""E8: Equilibrium selection sensitivity (TA feedback point 2).

Part A: Independent backward induction under all 7 selection rules (symmetric and
        unrestricted), with 20 seeds for random_seeded, reporting state-multiplicity
        fractions, G, E, J, and long-run stock -> results/raw/e8_selection_benchmarks.csv.
Part B: Repeat core E2 learner comparisons (IQL, MB-EQ, MB-OA) at frozen (gamma=0.99,
        sigma=0.1, r in {0.5, 0.6, 0.7, 0.8}) under both max_total and min_total
        -> results/raw/e8_learners_by_rule.csv.
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
from equilibrium import executed_catches, solve_equilibrium  # noqa: E402
from learners.iql import iql_pair  # noqa: E402
from learners.mb_eq import mbeq_pair  # noqa: E402
from learners.mb_oa import mboa_pair  # noqa: E402
from learn_common import (  # noqa: E402
    SMOKE_SEEDS,
    build_agent,
    episodes_to_G_within,
    evaluate_pair,
    grids_of,
    noise_matrix,
    prepare,
    reported_seeds,
    summarize_metrics,
    train_pair,
    worker_init,
    write_rows,
)
from metrics import evaluate, one_hot_from_index, steady_stock  # noqa: E402
from utils import config_hash, git_commit, horizon_for_tail, load_config  # noqa: E402


E8_LEARNER_BUDGETS = {
    "iql": (
        5000,
        [0, 49, 249, 499, 999, 1999, 2999, 3999, 4999],
    ),
    "mb_eq": (
        500,
        [0, 4, 9, 19, 49, 99, 199, 349, 499],
    ),
    "mb_oa": (
        200,
        [0, 4, 9, 19, 49, 99, 149, 199],
    ),
}


def _solve_benchmark_task(task):
    r, rule, sym, sel_seed, quick = task
    cfg = copy.deepcopy(load_config(quick=quick))
    cache = BenchmarkCache(cfg, rule=rule)
    plan = cache.get_planner(float(r))
    t0 = time.perf_counter()
    if rule == "random_seeded":
        eq = solve_equilibrium(
            cfg, float(r), rule=rule, symmetric_only=bool(sym),
            selection_seed=int(sel_seed), verbose=False,
        )
    else:
        eq = cache.get_equilibrium(float(r), rule=rule, symmetric_only=bool(sym))
    elapsed = time.perf_counter() - t0

    n_a = plan.a_grid.shape[0]
    p1_plan = one_hot_from_index(plan.a1_idx, n_a)
    p2_plan = one_hot_from_index(plan.a2_idx, n_a)
    xi = noise_matrix(cfg, int(cfg["evaluation"]["mc_seed"]))
    ev = evaluate(
        eq.p1, eq.p2, float(r), cfg, eq.s_grid, eq.a_grid, p1_plan, p2_plan,
        eq_p1=eq.p1, eq_p2=eq.p2,
        planner_J_dp=float(plan.V[-1]), eq_J_dp=float(eq.V1[-1] + eq.V2[-1]),
        xi=xi, seed=int(cfg["evaluation"]["mc_seed"]), with_gap=True,
    )
    for k in ("_br1_V", "_br2_V", "_v1_V", "_v2_V"):
        ev.pop(k, None)
    steady_p, _ = steady_stock(
        plan.s_grid, plan.a_grid, p1_plan, p2_plan, float(r), cfg["model"]["K"], cfg["model"]["s0"]
    )
    steady_e, _ = steady_stock(
        eq.s_grid, eq.a_grid, eq.p1, eq.p2, float(r), cfg["model"]["K"], cfg["model"]["s0"]
    )
    return {
        "r": float(r),
        "rule": rule,
        "symmetric_only": bool(sym),
        "selection_seed": int(sel_seed),
        "eq_converged": bool(eq.converged),
        "H_eq": int(eq.H),
        "eps_abs": float(eq.stage_exploit_max),
        "eps_rel": float(eq.eps_rel),
        "n_states_multi_eq": int(np.sum(eq.n_eq > 1)),
        "frac_states_multi_eq": float(np.mean(eq.n_eq > 1)),
        "mean_n_eq": float(np.mean(eq.n_eq)),
        "n_mixed_selected": int(eq.n_mixed_selected),
        "n_fallback": int(eq.n_fallback),
        "J_plan_dp": float(plan.V[-1]),
        "J_eq_dp": float(eq.V1[-1] + eq.V2[-1]),
        "G_dp": float(1.0 - (eq.V1[-1] + eq.V2[-1]) / plan.V[-1]) if plan.V[-1] != 0 else float("nan"),
        "J_eq_mc": float(ev["J"]),
        "G_mc": float(ev["G"]),
        "E": float(ev["E"]),
        "E_converged": bool(ev["gap_converged"]),
        "steady_stock_plan": float(steady_p),
        "steady_stock_eq": float(steady_e),
        "long_run_stock_mc": float(ev["long_run_stock"]),
        "collapse_50_eq": float(ev["collapse_50"]),
        "collapse_long_eq": float(ev["collapse_long"]),
        "seconds": float(elapsed),
        "config_hash": config_hash(cfg),
        "git_commit": git_commit(),
    }


def _run_learner_rule_task(task):
    learner_name, r, seed, rule, n_ep, checkpoints = task
    cfg = copy.deepcopy(load_config(quick=False))
    s_grid, a_grid = grids_of(cfg)
    cache, plan, p1, p2, eq = prepare(cfg, r, rule=rule)
    a1 = build_agent(learner_name, s_grid, a_grid, cfg, cache, fleet_index=0, seed=seed, true_r=r)
    a2 = build_agent(learner_name, s_grid, a_grid, cfg, cache, fleet_index=1, seed=seed, true_r=r)

    rows, _online, _elapsed, _c, _p, _eq = train_pair(
        a1, a2, cfg, r, seed, n_episodes=n_ep, eval_every=n_ep,
        with_gap=True, rule=rule, eval_episodes=checkpoints,
    )
    ep10 = episodes_to_G_within(rows, frac=0.10)
    last_idx = len(rows) - 1
    for i, row in enumerate(rows):
        row["learner"] = learner_name
        row["rule"] = rule
        row["is_final"] = (i == last_idx)
        row["ep_to_G_10pct"] = int(ep10)
    return rows


def _run_quick(cfg):
    rules = ["max_total", "min_total", "payoff_dominant"]
    r_values = [float(cfg["model"]["r_values"][0])]
    bench_rows = [_solve_benchmark_task((r_values[0], rule, True, 329, True)) for rule in rules]
    write_rows(ROOT / "results" / "raw" / "e8_quick_bench.csv", bench_rows)

    s_grid, a_grid = grids_of(cfg)
    learn_rows = []
    for rule in ("max_total", "min_total"):
        cache, _p, _p1, _p2, _eq = prepare(cfg, r_values[0], rule=rule)
        for seed in SMOKE_SEEDS:
            for lname in ("iql", "mb_eq", "mb_oa"):
                a1 = build_agent(lname, s_grid, a_grid, cfg, cache, 0, seed, true_r=r_values[0])
                a2 = build_agent(lname, s_grid, a_grid, cfg, cache, 1, seed, true_r=r_values[0])
                trained, *_ = train_pair(
                    a1, a2, cfg, r_values[0], seed, n_episodes=8, eval_every=8,
                    with_gap=True, rule=rule, eval_episodes=[7],
                )
                for row in trained:
                    row["learner"] = lname
                    row["rule"] = rule
                learn_rows.extend(trained)
    frame = write_rows(ROOT / "results" / "raw" / "e8_quick.csv", learn_rows)
    print("bench", len(bench_rows), "learn", len(frame), flush=True)


def _run_full(workers):
    cfg = load_config(quick=False)
    r_bench = [float(v) for v in cfg["model"]["r_values"]] + [float(v) for v in cfg["model"].get("r_sanity", [])]
    r_learn = [float(v) for v in cfg["model"]["r_values"]]
    rules = list(cfg["selection"]["rules"])
    base_sel_seed = int(cfg["selection"]["random_seed"])
    n_rand = int(cfg["selection"]["n_random_seeds"])
    seeds = reported_seeds(cfg)

    # Part A: benchmarks across all rules, symmetric and unrestricted, with n_random_seeds for random_seeded.
    bench_tasks = []
    for r in r_bench:
        for rule in rules:
            for sym in (True, False):
                if rule == "random_seeded":
                    for k in range(n_rand):
                        bench_tasks.append((r, rule, sym, base_sel_seed + k, False))
                else:
                    bench_tasks.append((r, rule, sym, base_sel_seed, False))

    print(f"Launching E8 Part A: {len(bench_tasks)} benchmark solves across {workers} workers...", flush=True)
    t0 = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers, initializer=worker_init) as pool:
        bench_rows = list(pool.map(_solve_benchmark_task, bench_tasks, chunksize=2))
    df_bench = write_rows(ROOT / "results" / "raw" / "e8_selection_benchmarks.csv", bench_rows)
    print(f"Wrote {len(df_bench)} rows -> e8_selection_benchmarks.csv in {time.perf_counter()-t0:.1f}s", flush=True)

    # Pre-warm max_total and min_total caches for Part B.
    for rule in ("max_total", "min_total"):
        BenchmarkCache(cfg, rule=rule).ensure_rule(rule)

    # Part B: learners under max_total and min_total.
    learn_tasks = []
    for rule in ("max_total", "min_total"):
        for lname, (n_ep, cks) in E8_LEARNER_BUDGETS.items():
            for r in r_learn:
                for seed in seeds:
                    learn_tasks.append((lname, r, seed, rule, n_ep, cks))

    print(f"Launching E8 Part B: {len(learn_tasks)} learner runs across {workers} workers...", flush=True)
    t1 = time.perf_counter()
    learn_rows = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers, initializer=worker_init) as pool:
        for i, r_rows in enumerate(pool.map(_run_learner_rule_task, learn_tasks, chunksize=4), 1):
            learn_rows.extend(r_rows)
            if i % 120 == 0 or i == len(learn_tasks):
                print(f"  E8 Part B progress: {i}/{len(learn_tasks)} ({(time.perf_counter()-t1)/60:.1f}m)", flush=True)

    df_learn = write_rows(ROOT / "results" / "raw" / "e8_learners_by_rule.csv", learn_rows)
    final_df = df_learn[df_learn["is_final"] == True].copy()  # noqa: E712
    metrics = [
        "J", "G", "E", "collapse_50", "collapse_long", "cum_collapse_50_train",
        "long_run_stock", "cum_online_return", "L_tau_plan", "L_tau_eq", "ep_to_G_10pct",
    ]
    sum_df = summarize_metrics(final_df, ["rule", "learner", "r", "budget_episodes"], metrics)
    write_rows(ROOT / "results" / "raw" / "e8_learners_summary.csv", sum_df.to_dict("records"))
    print(f"E8 completed in {(time.perf_counter()-t0)/60:.2f}m", flush=True)


def main():
    parser = argparse.ArgumentParser(description="E8 equilibrium selection sensitivity")
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
