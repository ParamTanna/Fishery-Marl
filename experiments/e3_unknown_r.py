"""E3: Cost of not knowing the intrinsic growth rate r.

Compares MBEqFleet with Bayes belief updates against MBEqFleet given true_r across
frozen r in {0.5, 0.6, 0.7, 0.8} and sigma in {0.05, 0.1, 0.2}, separating warm-up
from post-warm-up collapse, tracking posterior mass on the true r by season, and
measuring harvest vs stock relative to known-r equilibrium.
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
from learners.mb_eq import mbeq_pair  # noqa: E402
from learn_common import (  # noqa: E402
    SMOKE_SEEDS,
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
from metrics import rollout_returns  # noqa: E402
from training import make_env, learning_cost  # noqa: E402
from utils import config_hash, git_commit, load_config  # noqa: E402


E3_EPISODES = 200
E3_CHECKPOINTS = {0, 1, 2, 4, 9, 14, 19, 29, 49, 74, 99, 149, 199}
TRACE_SEASONS = set(range(1, 101)) | set(range(105, 501, 5))


def _expected_total_harvest(p1, p2, s_grid, a_grid):
    """Exact expected executed total catch at every grid stock under (p1, p2)."""
    a_sum = a_grid[:, None] + a_grid[None, :]
    n_s = s_grid.shape[0]
    out = np.empty(n_s, dtype=np.float64)
    for si in range(n_s):
        h_tot = np.minimum(float(s_grid[si]), a_sum)
        out[si] = float(np.sum(np.outer(p1[si], p2[si]) * h_tot))
    return out


def _run_one(task):
    label, r, sigma, seed, n_episodes, checkpoints = task
    cfg = copy.deepcopy(load_config(quick=False))
    cfg["model"]["sigma"] = float(sigma)
    s_grid, a_grid = grids_of(cfg)
    cache, plan, plan_p1, plan_p2, eq = prepare(cfg, r)
    known_r = None if label == "belief" else float(r)
    a1, a2 = mbeq_pair(s_grid, a_grid, cfg, cache, seed, known_r=known_r, true_r=r)

    xi = noise_matrix(cfg, seed)
    env = make_env(cfg, r, seed)
    T = int(cfg["model"]["T"])
    delta = float(cfg["model"]["delta"])
    K = float(cfg["model"]["K"])
    n_warm = int(cfg["learners"]["n_warm"])

    j_plan_mc = rollout_returns(s_grid, a_grid, plan_p1, plan_p2, cfg, r, seed=seed, xi=xi)["J_total"]
    j_eq_mc = rollout_returns(s_grid, a_grid, eq.p1, eq.p2, cfg, r, seed=seed, xi=xi)["J_total"]
    _br1_v, _br2_v, _v1_v, _v2_v = [None], [None], [None], [None]
    gap_cache = {}

    rows = []
    trace_rows = []
    online = []
    warm_collapses = []
    post_collapses_cond = []
    post_collapses_all = []
    any_collapses = []
    total_seasons = 0
    ep0_warm_collapsed = False

    b_grid = a1.belief.grid
    mask_02 = np.abs(b_grid - float(r)) <= 0.02 + 1.0e-9
    mask_05 = np.abs(b_grid - float(r)) <= 0.05 + 1.0e-9
    mask_exact = np.abs(b_grid - float(r)) <= 0.005 + 1.0e-9

    t0 = time.perf_counter()
    for ep in range(int(n_episodes)):
        a1.begin_episode()
        a2.begin_episode()
        stock = env.reset()
        tot = np.zeros(2, dtype=np.float64)
        disc = 1.0
        min_warm = float(stock)
        min_post = np.inf
        for t in range(T):
            req1 = a1.act(stock, explore=True)
            req2 = a2.act(stock, explore=True)
            nxt, rew, _done, info = env.step([req1, req2])
            hats = info["executed"]
            a1.observe(stock, req1, hats, float(rew[0]), nxt)
            a2.observe(stock, req2, hats, float(rew[1]), nxt)
            tot += disc * rew
            disc *= env.gamma
            if t < n_warm:
                min_warm = min(min_warm, float(nxt))
            else:
                min_post = min(min_post, float(nxt))
            stock = nxt
            total_seasons += 1
            if label == "belief" and (total_seasons in TRACE_SEASONS or ep == int(n_episodes) - 1 and t == T - 1):
                probs = a1.belief.probs()
                trace_rows.append({
                    "r": float(r),
                    "sigma": float(sigma),
                    "seed": int(seed),
                    "episode": int(ep),
                    "season_in_ep": int(t + 1),
                    "total_season": int(total_seasons),
                    "belief_mean": float(a1.belief.mean()),
                    "belief_var": float(a1.belief.variance()),
                    "mass_exact": float(probs[mask_exact].sum()),
                    "mass_tol02": float(probs[mask_02].sum()),
                    "mass_tol05": float(probs[mask_05].sum()),
                })
        a1.finish_episode()
        a2.finish_episode()
        online.append(float(tot.sum()))

        w_col = bool(min_warm < delta * K) if n_warm > 0 else False
        p_col = bool(min_post < delta * K) if np.isfinite(min_post) else False
        if ep == 0:
            ep0_warm_collapsed = w_col
        warm_collapses.append(w_col)
        post_collapses_all.append(p_col)
        if not w_col:
            post_collapses_cond.append(p_col)
        any_collapses.append(bool(w_col or p_col))

        if ep in checkpoints or ep == int(n_episodes) - 1:
            row = evaluate_pair(
                a1, a2, cfg, r, cache, plan, plan_p1, plan_p2, eq, xi, seed,
                with_gap=True, br1_v_init=_br1_v[0], br2_v_init=_br2_v[0],
                v1_v_init=_v1_v[0], v2_v_init=_v2_v[0], gap_cache=gap_cache,
                j_plan_cached=j_plan_mc, j_eq_cached=j_eq_mc,
            )
            if row.get("_br1_V") is not None:
                _br1_v[0], _br2_v[0] = row["_br1_V"], row["_br2_V"]
                _v1_v[0], _v2_v[0] = row["_v1_V"], row["_v2_V"]
            for k in ("_br1_V", "_br2_V", "_v1_V", "_v2_V"):
                row.pop(k, None)
            row["mode"] = label
            row["sigma"] = float(sigma)
            row["episode"] = int(ep)
            row["budget_episodes"] = int(n_episodes)
            row["online_return"] = float(online[-1])
            row["cum_online_return"] = float(np.mean(online))
            row["belief_mean"] = float(a1.belief.mean())
            row["belief_var"] = float(a1.belief.variance())
            probs = a1.belief.probs()
            row["mass_tol02"] = float(probs[mask_02].sum())
            row["mass_tol05"] = float(probs[mask_05].sum())
            row["collapse_50_train"] = bool(w_col or p_col)
            row["cum_collapse_50_train"] = float(np.mean(any_collapses))
            row["collapse_50_train_warm"] = float(np.mean(warm_collapses))
            row["collapse_50_train_post_cond"] = (
                float(np.mean(post_collapses_cond)) if post_collapses_cond else float("nan")
            )
            row["collapse_50_train_post_all"] = float(np.mean(post_collapses_all))
            row["ep0_warm_collapsed"] = bool(ep0_warm_collapsed)
            row["is_final"] = (ep == int(n_episodes) - 1)
            rows.append(row)

    elapsed = time.perf_counter() - t0
    cost_plan = learning_cost(online, float(plan.V[-1]))
    cost_eq = learning_cost(online, float(eq.V1[-1] + eq.V2[-1]))
    for row in rows:
        ep = int(row["episode"])
        row["L_tau_plan"] = float(cost_plan[ep])
        row["L_tau_eq"] = float(cost_eq[ep])
        row["train_seconds"] = elapsed

    # Final harvest-by-stock comparison against known-r equilibrium & planner.
    p1_fin = a1.greedy_table()
    p2_fin = a2.greedy_table()
    h_learned = _expected_total_harvest(p1_fin, p2_fin, s_grid, a_grid)
    h_eq = _expected_total_harvest(eq.p1, eq.p2, s_grid, a_grid)
    h_plan = np.asarray(plan.total_executed, dtype=np.float64)
    harvest_rows = []
    for si, s in enumerate(s_grid):
        harvest_rows.append({
            "mode": label,
            "r": float(r),
            "sigma": float(sigma),
            "seed": int(seed),
            "s": float(s),
            "catch_learned": float(h_learned[si]),
            "catch_eq": float(h_eq[si]),
            "catch_plan": float(h_plan[si]),
            "diff_vs_eq": float(h_learned[si] - h_eq[si]),
            "diff_vs_plan": float(h_learned[si] - h_plan[si]),
        })

    return rows, trace_rows, harvest_rows


def _run_quick(cfg):
    cfg["learners"]["n_warm"] = 2
    cfg["learners"]["n_replan"] = 4
    cache = BenchmarkCache(cfg)
    cache.ensure_rule()
    s_grid, a_grid = grids_of(cfg)
    r = float(cfg["model"]["r_values"][0])
    rows = []
    for seed in SMOKE_SEEDS:
        for label, known_r in (("belief", None), ("known_r", r)):
            pair = mbeq_pair(s_grid, a_grid, cfg, cache, seed, known_r=known_r, true_r=r)
            trained, _on, elapsed, *_rest = train_pair(
                pair[0], pair[1], cfg, r, seed, n_episodes=12, eval_every=6, with_gap=True,
            )
            for row in trained:
                row["mode"] = label
                row["sigma"] = float(cfg["model"]["sigma"])
                row["belief_mean"] = float(pair[0].belief.mean())
            rows.extend(trained)
            print(f"  {label} seed={seed} {elapsed:.2f}s belief_mean={pair[0].belief.mean():.3f}", flush=True)
    frame = write_rows(ROOT / "results" / "raw" / "e3_quick.csv", rows)
    print("rows", len(frame), flush=True)


def _run_full(workers):
    cfg = load_config(quick=False)
    r_values = [float(v) for v in cfg["model"]["r_values"]]
    sigmas = [0.05, 0.1, 0.2]
    seeds = reported_seeds(cfg)

    for sigma in sigmas:
        sc = copy.deepcopy(cfg)
        sc["model"]["sigma"] = sigma
        BenchmarkCache(sc).ensure_rule()

    tasks = []
    for label in ("belief", "known_r"):
        for r in r_values:
            for sigma in sigmas:
                for seed in seeds:
                    tasks.append((label, r, sigma, seed, E3_EPISODES, E3_CHECKPOINTS))

    print(f"Launching E3 full: {len(tasks)} runs across {workers} workers...", flush=True)
    t0 = time.perf_counter()
    all_rows, all_trace, all_harvest = [], [], []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers, initializer=worker_init) as pool:
        for i, (r_rows, t_rows, h_rows) in enumerate(pool.map(_run_one, tasks, chunksize=4), 1):
            all_rows.extend(r_rows)
            all_trace.extend(t_rows)
            all_harvest.extend(h_rows)
            if i % 96 == 0 or i == len(tasks):
                print(f"  E3 progress: {i}/{len(tasks)} ({time.perf_counter()-t0:.1f}s)", flush=True)

    df = write_rows(ROOT / "results" / "raw" / "e3_unknown_r.csv", all_rows)
    df_tr = write_rows(ROOT / "results" / "raw" / "e3_posterior_trace.csv", all_trace)
    df_hv = write_rows(ROOT / "results" / "raw" / "e3_harvest_by_stock.csv", all_harvest)
    print(f"Wrote E3: {len(df)} main rows, {len(df_tr)} trace rows, {len(df_hv)} harvest rows in {time.perf_counter()-t0:.1f}s", flush=True)


def main():
    parser = argparse.ArgumentParser(description="E3 cost of not knowing r")
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
