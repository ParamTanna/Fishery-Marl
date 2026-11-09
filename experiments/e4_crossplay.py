"""E4: Cross-play (TA feedback point 1).

Tests whether MBOAFleet outperforms MBEqFleet when facing non-equilibrium opponents
and matches MBEqFleet against an equilibrium opponent, with request_recovery both on
and off. Reports co-adapted online cross-play (primary), frozen self-play cross-play
(secondary), and paired CRN differences (MB-OA minus MB-EQ) with 95% t-intervals.
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    FixedTableAgent,
    build_agent,
    evaluate_pair,
    fixed_tables,
    grids_of,
    noise_matrix,
    prepare,
    reported_seeds,
    summarize_metrics,
    train_pair,
    worker_init,
    write_rows,
)
from utils import load_config  # noqa: E402


ROW_LEARNERS = ("mb_eq", "mb_oa", "mb_oa_norec", "iql")
COL_OPPONENTS = ("mb_eq", "mb_oa", "iql", "planner_share", "equilibrium", "overharvester")


def _budget_and_evals(row_name, col_name):
    if "mb_oa" in row_name or "mb_oa" in col_name:
        return 200, [0, 49, 99, 199]
    if "mb_eq" in row_name or col_name == "mb_eq":
        return 500, [0, 49, 99, 199, 499]
    return 5000, [0, 199, 499, 999, 2499, 4999]


def _run_coadapt_task(task):
    row_name, col_name, r, seed = task
    cfg = copy.deepcopy(load_config(quick=False))
    s_grid, a_grid = grids_of(cfg)
    cache, plan, plan_p1, plan_p2, eq = prepare(cfg, r)
    fixed = fixed_tables(cfg, r, cache, plan, eq)

    a1 = build_agent(row_name, s_grid, a_grid, cfg, cache, fleet_index=0, seed=seed, true_r=r)
    if col_name in fixed:
        a2 = build_agent(
            "fixed", s_grid, a_grid, cfg, cache, fleet_index=1, seed=seed,
            fixed_table=fixed[col_name][1],
        )
    else:
        a2 = build_agent(col_name, s_grid, a_grid, cfg, cache, fleet_index=1, seed=seed, true_r=r)

    n_episodes, checkpoints = _budget_and_evals(row_name, col_name)
    rows, _online, _elapsed, _c, _p, _eq = train_pair(
        a1, a2, cfg, r, seed, n_episodes=n_episodes, eval_every=n_episodes,
        with_gap=True, eval_episodes=checkpoints,
    )
    rec_flag = True if row_name == "mb_oa" else (False if row_name == "mb_oa_norec" else None)
    last_idx = len(rows) - 1
    for i, row in enumerate(rows):
        row["row_learner"] = row_name
        row["column_opponent"] = col_name
        row["request_recovery"] = rec_flag
        row["is_final"] = (i == last_idx)
        row["is_ep200"] = (int(row["episode"]) == 199)
    return rows


def _run_frozen_seed(task):
    r, seed = task
    cfg = copy.deepcopy(load_config(quick=False))
    s_grid, a_grid = grids_of(cfg)
    cache, plan, plan_p1, plan_p2, eq = prepare(cfg, r)
    fixed = fixed_tables(cfg, r, cache, plan, eq)
    xi = noise_matrix(cfg, seed)

    trained = {}
    for name, n_ep in (("mb_eq", 500), ("mb_oa", 200), ("mb_oa_norec", 200), ("iql", 5000)):
        a1 = build_agent(name, s_grid, a_grid, cfg, cache, fleet_index=0, seed=seed, true_r=r)
        a2 = build_agent(name, s_grid, a_grid, cfg, cache, fleet_index=1, seed=seed, true_r=r)
        train_pair(a1, a2, cfg, r, seed, n_episodes=n_ep, eval_every=n_ep, with_gap=False)
        trained[name] = (
            FixedTableAgent(a1.greedy_table(), s_grid, a_grid, seed=seed),
            FixedTableAgent(a2.greedy_table(), s_grid, a_grid, seed=seed + 17),
        )

    out = []
    for row_name in ROW_LEARNERS:
        for col_name in COL_OPPONENTS:
            right_tbl = fixed[col_name][1] if col_name in fixed else trained[col_name][1].greedy_table()
            right = FixedTableAgent(right_tbl, s_grid, a_grid, seed=seed + 99)
            row = evaluate_pair(
                trained[row_name][0], right, cfg, r, cache, plan, plan_p1, plan_p2, eq, xi, seed,
                with_gap=True,
            )
            for k in ("_br1_V", "_br2_V", "_v1_V", "_v2_V"):
                row.pop(k, None)
            row["row_learner"] = row_name
            row["column_opponent"] = col_name
            row["request_recovery"] = (
                True if row_name == "mb_oa" else (False if row_name == "mb_oa_norec" else None)
            )
            out.append(row)
    return out


def _compute_paired_diffs(final_df, comparison_name, left_row, right_row):
    metrics = ["J1", "J2", "J", "G", "E", "E1", "collapse_50", "collapse_long", "long_run_stock"]
    left = final_df[final_df["row_learner"] == left_row].set_index(["r", "seed", "column_opponent"])
    right = final_df[final_df["row_learner"] == right_row].set_index(["r", "seed", "column_opponent"])
    common_idx = left.index.intersection(right.index)
    diffs = []
    for idx in common_idx:
        r, seed, col = idx
        item = {
            "comparison": comparison_name,
            "r": float(r),
            "seed": int(seed),
            "column_opponent": col,
        }
        for m in metrics:
            item[f"diff_{m}"] = float(left.loc[idx, m]) - float(right.loc[idx, m])
            item[f"left_{m}"] = float(left.loc[idx, m])
            item[f"right_{m}"] = float(right.loc[idx, m])
        diffs.append(item)
    diff_df = pd.DataFrame(diffs)
    diff_cols = [f"diff_{m}" for m in metrics]
    per_r = summarize_metrics(diff_df, ["comparison", "column_opponent", "r"], diff_cols)
    per_r["scope"] = "by_r"
    pooled = summarize_metrics(diff_df, ["comparison", "column_opponent"], diff_cols)
    pooled["r"] = np.nan
    pooled["scope"] = "pooled_r"
    return pd.concat([pooled, per_r], ignore_index=True)


def _plot(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(frame["row_learner"].unique())
    cols = list(frame["column_opponent"].unique())
    grid = np.full((len(rows), len(cols)), np.nan)
    for i, row_name in enumerate(rows):
        for j, col_name in enumerate(cols):
            hit = frame[(frame.row_learner == row_name) & (frame.column_opponent == col_name)]
            if not hit.empty:
                grid[i, j] = float(hit["G"].mean())
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    image = ax.imshow(grid, cmap="cividis", aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=25, ha="right", fontsize=8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=8)
    for i in range(len(rows)):
        for j in range(len(cols)):
            ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(image, ax=ax, label="efficiency gap G")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _run_quick(cfg, recover):
    s_grid, a_grid = grids_of(cfg)
    r = float(cfg["model"]["r_values"][0])
    cache, plan, plan_p1, plan_p2, eq = prepare(cfg, r)
    fixed = fixed_tables(cfg, r, cache, plan, eq)
    rows = []
    for seed in SMOKE_SEEDS:
        built = {
            "iql": iql_pair(s_grid, a_grid, cfg, seed),
            "mb_eq": mbeq_pair(s_grid, a_grid, cfg, cache, seed, true_r=r),
            "mb_oa": mboa_pair(s_grid, a_grid, cfg, cache, seed, true_r=r, recover_requests=recover),
        }
        for name, pair in built.items():
            train_pair(pair[0], pair[1], cfg, r, seed, n_episodes=8, eval_every=8, with_gap=False)
        xi = noise_matrix(cfg, seed)
        for row_name in ("iql", "mb_eq", "mb_oa"):
            left = built[row_name][0]
            for col_name in ("iql", "mb_eq", "mb_oa", "planner_share", "equilibrium", "overharvester"):
                right_tbl = fixed[col_name][1] if col_name in fixed else built[col_name][1].greedy_table()
                right = FixedTableAgent(right_tbl, s_grid, a_grid, seed=seed)
                stats = evaluate_pair(left, right, cfg, r, cache, plan, plan_p1, plan_p2, eq, xi, seed, with_gap=True)
                for k in ("_br1_V", "_br2_V", "_v1_V", "_v2_V"):
                    stats.pop(k, None)
                stats["row_learner"] = row_name
                stats["column_opponent"] = col_name
                stats["request_recovery"] = recover if row_name == "mb_oa" else None
                rows.append(stats)
    frame = write_rows(ROOT / "results" / "raw" / "e4_quick.csv", rows)
    _plot(frame, ROOT / "results" / "figures" / "e4_crossplay_quick.png")
    print("rows", len(frame), flush=True)


def _run_full(workers):
    cfg = load_config(quick=False)
    BenchmarkCache(cfg).ensure_rule()
    r_values = [float(v) for v in cfg["model"]["r_values"]]
    seeds = reported_seeds(cfg)

    tasks = [
        (rn, cn, r, s)
        for rn in ROW_LEARNERS
        for cn in COL_OPPONENTS
        for r in r_values
        for s in seeds
    ]
    print(f"Launching E4 online cross-play: {len(tasks)} runs across {workers} workers...", flush=True)
    t0 = time.perf_counter()
    all_rows = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers, initializer=worker_init) as pool:
        for i, r_rows in enumerate(pool.map(_run_coadapt_task, tasks, chunksize=4), 1):
            all_rows.extend(r_rows)
            if i % 160 == 0 or i == len(tasks):
                print(f"  E4 online progress: {i}/{len(tasks)} ({(time.perf_counter()-t0)/60:.1f}m)", flush=True)

    df = write_rows(ROOT / "results" / "raw" / "e4_crossplay.csv", all_rows)
    final_df = df[df["is_final"] == True].copy()  # noqa: E712
    paired_parts = [
        _compute_paired_diffs(final_df, "mboa_rec_minus_mbeq_final", "mb_oa", "mb_eq"),
        _compute_paired_diffs(final_df, "mboa_norec_minus_mbeq_final", "mb_oa_norec", "mb_eq"),
        _compute_paired_diffs(final_df, "mboa_rec_minus_mboa_norec_final", "mb_oa", "mb_oa_norec"),
    ]
    ep200_df = df[df["is_ep200"] == True].copy()  # noqa: E712
    paired_parts.extend([
        _compute_paired_diffs(ep200_df, "mboa_rec_minus_mbeq_ep200", "mb_oa", "mb_eq"),
        _compute_paired_diffs(ep200_df, "mboa_norec_minus_mbeq_ep200", "mb_oa_norec", "mb_eq"),
    ])
    paired = pd.concat(paired_parts, ignore_index=True)
    write_rows(ROOT / "results" / "raw" / "e4_crossplay_paired.csv", paired.to_dict("records"))

    # Secondary: frozen self-play cross-play on the exact same (r, seeds).
    f_tasks = [(r, s) for r in r_values for s in seeds]
    print(f"Launching E4 frozen cross-play: {len(f_tasks)} (r,seed) blocks across {workers} workers...", flush=True)
    frozen_rows = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers, initializer=worker_init) as pool:
        for res in pool.map(_run_frozen_seed, f_tasks, chunksize=2):
            frozen_rows.extend(res)
    write_rows(ROOT / "results" / "raw" / "e4_crossplay_frozen.csv", frozen_rows)
    print(f"E4 completed in {(time.perf_counter()-t0)/60:.2f}m", flush=True)


def main():
    parser = argparse.ArgumentParser(description="E4 cross-play")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--no-request-recovery", action="store_true")
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    if args.full and not args.quick:
        _run_full(workers=args.workers)
    elif args.quick:
        _run_quick(load_config(quick=True), recover=not args.no_request_recovery)
    else:
        raise SystemExit("pass --quick or --full")


if __name__ == "__main__":
    main()
