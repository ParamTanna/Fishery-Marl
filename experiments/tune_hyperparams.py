"""Step 2 hyperparameter search on tuning seeds 0, 1, 2 ONLY (r = 0.5, 101x51).

Never evaluates or selects on reported seeds (10-29) or smoke seeds (100-101).
Selection rule: highest mean online return (across tuning seeds 0, 1, 2) on
frozen r = 0.5, 101x51, gamma = 0.99, sigma = 0.1.

Writes results/raw/tuning.csv with one row per (learner, config_id, seed, budget_checkpoint).
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

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
from learn_common import evaluate_pair, grids_of, noise_matrix, prepare, write_rows  # noqa: E402
from metrics import rollout_returns  # noqa: E402
from training import run_episodes, make_env  # noqa: E402
from utils import config_hash, git_commit, load_config  # noqa: E402


def _worker_init():
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[var] = "1"


def _build_iql_candidates():
    """Focused grid covering alpha and epsilon schedules around config defaults."""
    candidates = []
    # (alpha0, alpha_min, alpha_decay, epsilon0, epsilon_min, epsilon_decay)
    specs = [
        # Baseline config default
        (0.20, 0.02, 0.99950, 1.0, 0.05, 0.9990),
        # Vary epsilon decay / min (alpha fixed at slower per-step decay too)
        (0.20, 0.02, 0.99950, 1.0, 0.05, 0.9980),
        (0.20, 0.02, 0.99950, 1.0, 0.02, 0.9985),
        (0.20, 0.02, 0.99950, 1.0, 0.02, 0.9990),
        (0.20, 0.02, 0.99950, 0.5, 0.05, 0.9990),
        (0.20, 0.02, 0.99950, 0.5, 0.02, 0.9985),
        # Slower alpha decay (recall ~50 updates/ep -> 0.9999 decays ~500 ep, 0.99998 ~2300 ep)
        (0.20, 0.02, 0.99995, 1.0, 0.05, 0.9990),
        (0.20, 0.02, 0.99995, 1.0, 0.02, 0.9985),
        (0.20, 0.01, 0.99998, 1.0, 0.02, 0.9990),
        (0.20, 0.01, 0.99998, 0.5, 0.02, 0.9985),
        # Vary alpha0 & alpha_min
        (0.10, 0.01, 0.99995, 1.0, 0.02, 0.9985),
        (0.30, 0.02, 0.99995, 1.0, 0.02, 0.9985),
        (0.30, 0.05, 0.99950, 1.0, 0.05, 0.9990),
        (0.15, 0.02, 0.99990, 0.5, 0.05, 0.9985),
    ]
    for idx, (a0, amin, adec, e0, emin, edec) in enumerate(specs):
        candidates.append({
            "config_id": f"iql_{idx:02d}",
            "alpha0": a0,
            "alpha_min": amin,
            "alpha_decay": adec,
            "epsilon0": e0,
            "epsilon_min": emin,
            "epsilon_decay": edec,
            "n_episodes": 5000,
        })
    return candidates


def _build_mbeq_candidates():
    """Grid over N_warm, N_replan, epsilon_warm, safe_default_catch."""
    candidates = []
    specs = [
        # Baseline
        (25, 5, 0.15, 10.0),
        # Vary N_warm
        (5, 5, 0.15, 10.0),
        (10, 5, 0.15, 10.0),
        (15, 5, 0.15, 10.0),
        (40, 5, 0.15, 10.0),
        # Vary N_replan at promising N_warm values
        (5, 1, 0.15, 10.0),
        (5, 10, 0.15, 10.0),
        (10, 1, 0.15, 10.0),
        (10, 10, 0.15, 10.0),
        # Vary epsilon_warm
        (5, 5, 0.05, 10.0),
        (5, 5, 0.30, 10.0),
        (10, 5, 0.05, 10.0),
        # Vary safe_default_catch (below/at/above myopic optimum 15)
        (5, 5, 0.15, 5.0),
        (5, 5, 0.15, 15.0),
        (10, 5, 0.15, 5.0),
        (10, 5, 0.15, 15.0),
        (10, 5, 0.05, 5.0),
        (5, 5, 0.05, 5.0),
    ]
    for idx, (nw, nr, ew, sc) in enumerate(specs):
        candidates.append({
            "config_id": f"mbeq_{idx:02d}",
            "n_warm": nw,
            "n_replan": nr,
            "epsilon_warm": ew,
            "safe_default_catch": sc,
            "n_episodes": 200,
        })
    return candidates


def _build_mboa_candidates():
    """Grid over N_warm, N_replan, epsilon_warm, safe_default_catch, bandwidth, request_recovery."""
    candidates = []
    specs = [
        # (n_warm, n_replan, epsilon_warm, safe_default_catch, bandwidth, request_recovery)
        (25, 5, 0.15, 10.0, 5.0, True),   # config default
        (5, 5, 0.15, 10.0, 5.0, True),
        (10, 5, 0.15, 10.0, 5.0, True),
        (5, 10, 0.15, 10.0, 5.0, True),
        (10, 10, 0.15, 10.0, 5.0, True),
        (5, 5, 0.05, 10.0, 5.0, True),
        (5, 5, 0.05, 5.0, 5.0, True),
        (5, 5, 0.15, 5.0, 5.0, True),
        (5, 5, 0.15, 15.0, 5.0, True),
        # Vary kernel bandwidth
        (5, 5, 0.15, 10.0, 2.0, True),
        (5, 5, 0.15, 10.0, 10.0, True),
        (10, 5, 0.15, 10.0, 2.0, True),
        (10, 5, 0.15, 10.0, 10.0, True),
        # Compare request recovery off
        (5, 5, 0.15, 10.0, 5.0, False),
        (10, 5, 0.15, 10.0, 5.0, False),
        (25, 5, 0.15, 10.0, 5.0, False),
    ]
    for idx, (nw, nr, ew, sc, bw, rr) in enumerate(specs):
        candidates.append({
            "config_id": f"mboa_{idx:02d}",
            "n_warm": nw,
            "n_replan": nr,
            "epsilon_warm": ew,
            "safe_default_catch": sc,
            "bandwidth": bw,
            "request_recovery": rr,
            "n_episodes": 100,
        })
    return candidates


def _run_task(task):
    learner_name, cand, seed = task
    base_cfg = load_config(quick=False)
    cfg = copy.deepcopy(base_cfg)
    r = float(cfg["model"]["r_values"][0])  # frozen r = 0.5
    s_grid, a_grid = grids_of(cfg)

    if learner_name == "iql":
        cfg["learners"]["iql"]["alpha0"] = cand["alpha0"]
        cfg["learners"]["iql"]["alpha_min"] = cand["alpha_min"]
        cfg["learners"]["iql"]["alpha_decay"] = cand["alpha_decay"]
        cfg["learners"]["iql"]["epsilon0"] = cand["epsilon0"]
        cfg["learners"]["iql"]["epsilon_min"] = cand["epsilon_min"]
        cfg["learners"]["iql"]["epsilon_decay"] = cand["epsilon_decay"]
        n_ep = int(cand["n_episodes"])
        checkpoints = [1000, 2000, 3000, 5000]
        eval_every = 1000
    else:
        cfg["learners"]["n_warm"] = cand["n_warm"]
        cfg["learners"]["n_replan"] = cand["n_replan"]
        cfg["learners"]["epsilon_warm"] = cand["epsilon_warm"]
        cfg["learners"]["safe_default_catch"] = cand["safe_default_catch"]
        if learner_name == "mb_oa":
            cfg["opponent_model"]["bandwidth"] = cand["bandwidth"]
            cfg["opponent_model"]["request_recovery"] = cand["request_recovery"]
        n_ep = int(cand["n_episodes"])
        checkpoints = [50, 100, 200] if n_ep >= 200 else [50, 100]
        eval_every = 50

    cache, plan, plan_p1, plan_p2, eq = prepare(cfg, r)
    if learner_name == "iql":
        a1, a2 = iql_pair(s_grid, a_grid, cfg, seed)
    elif learner_name == "mb_eq":
        a1, a2 = mbeq_pair(s_grid, a_grid, cfg, cache, seed, true_r=r)
    else:
        a1, a2 = mboa_pair(
            s_grid, a_grid, cfg, cache, seed, true_r=r,
            recover_requests=cand["request_recovery"],
        )

    xi = noise_matrix(cfg, seed)
    env = make_env(cfg, r, seed)
    env.horizon = int(cfg["model"]["T"])
    env.delta = float(cfg["model"]["delta"])
    j_plan_mc = rollout_returns(s_grid, a_grid, plan_p1, plan_p2, cfg, r, seed=seed, xi=xi)["J_total"]
    j_eq_mc = rollout_returns(s_grid, a_grid, eq.p1, eq.p2, cfg, r, seed=seed, xi=xi)["J_total"]

    _br1_v = [None]
    _br2_v = [None]
    _v1_v = [None]
    _v2_v = [None]
    gap_cache = {}
    eval_indices = set(c - 1 for c in checkpoints if c <= n_ep)
    eval_indices.add(n_ep - 1)

    rows = []
    online = []
    T = int(cfg["model"]["T"])
    delta = float(cfg["model"]["delta"])
    t_start = time.perf_counter()
    for episode in range(n_ep):
        a1.begin_episode()
        a2.begin_episode()
        stock = env.reset()
        total = np.zeros(2, dtype=np.float64)
        discount = 1.0
        min_stock = float(stock)
        for _t in range(T):
            req1 = a1.act(stock, explore=True)
            req2 = a2.act(stock, explore=True)
            nxt, rewards, _done, info = env.step([req1, req2])
            hats = info["executed"]
            a1.observe(stock, req1, hats, float(rewards[0]), nxt)
            a2.observe(stock, req2, hats, float(rewards[1]), nxt)
            total += discount * rewards
            discount *= env.gamma
            min_stock = min(min_stock, float(nxt))
            stock = nxt
        a1.finish_episode()
        a2.finish_episode()
        online.append(float(total.sum()))

        if episode in eval_indices:
            row = evaluate_pair(
                a1, a2, cfg, r, cache, plan, plan_p1, plan_p2, eq, xi, seed,
                with_gap=True, br1_v_init=_br1_v[0], br2_v_init=_br2_v[0],
                v1_v_init=_v1_v[0], v2_v_init=_v2_v[0], gap_cache=gap_cache,
                j_plan_cached=j_plan_mc, j_eq_cached=j_eq_mc,
            )
            if row.get("_br1_V") is not None:
                _br1_v[0] = row["_br1_V"]
                _br2_v[0] = row["_br2_V"]
                _v1_v[0] = row["_v1_V"]
                _v2_v[0] = row["_v2_V"]
            row.pop("_br1_V", None)
            row.pop("_br2_V", None)
            row.pop("_v1_V", None)
            row.pop("_v2_V", None)
            ep_count = episode + 1
            on_arr = np.asarray(online, dtype=np.float64)
            row["learner"] = learner_name
            row["config_id"] = cand["config_id"]
            row["seed"] = int(seed)
            row["budget_episodes"] = int(ep_count)
            row["mean_online_return"] = float(on_arr.mean())
            row["tail20_online_return"] = float(on_arr[-min(20, len(on_arr)):].mean())
            row["last_online_return"] = float(on_arr[-1])
            row["collapse_50_train"] = bool(min_stock < delta * env.K)
            row["elapsed_seconds"] = time.perf_counter() - t_start
            for k, v in cand.items():
                if k not in ("config_id", "n_episodes"):
                    row[f"hp_{k}"] = v
            row["config_hash"] = config_hash(cfg)
            row["git_commit"] = git_commit()
            rows.append(row)
    return rows


def main():
    base_cfg = load_config(quick=False)
    tuning_seeds = [int(s) for s in base_cfg["seeds"]["tuning_seeds"]]
    assert tuning_seeds == [0, 1, 2], f"Expected tuning seeds [0, 1, 2], got {tuning_seeds}"

    # Pre-populate BenchmarkCache on disk so workers only load .npz
    BenchmarkCache(base_cfg).ensure_rule()

    tasks = []
    for cand in _build_iql_candidates():
        for s in tuning_seeds:
            tasks.append(("iql", cand, s))
    for cand in _build_mbeq_candidates():
        for s in tuning_seeds:
            tasks.append(("mb_eq", cand, s))
    for cand in _build_mboa_candidates():
        for s in tuning_seeds:
            tasks.append(("mb_oa", cand, s))

    print(f"Launching {len(tasks)} tuning runs across 12 workers on seeds {tuning_seeds}...", flush=True)
    started = time.perf_counter()
    all_rows = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=12, initializer=_worker_init) as pool:
        for i, res_rows in enumerate(pool.map(_run_task, tasks), 1):
            all_rows.extend(res_rows)
            if i % 18 == 0 or i == len(tasks):
                print(f"  completed {i}/{len(tasks)} runs ({time.perf_counter() - started:.1f}s)", flush=True)

    frame = write_rows(ROOT / "results" / "raw" / "tuning.csv", all_rows)
    print(f"Wrote {len(frame)} rows to results/raw/tuning.csv in {time.perf_counter() - started:.1f}s", flush=True)

    # Summarize winners per learner using selection rule: mean_online_return on tuning seeds.
    for learner_name in ("iql", "mb_eq", "mb_oa"):
        sub = frame[frame["learner"] == learner_name]
        grp = (
            sub.groupby(["config_id", "budget_episodes"], as_index=False)
            .agg(
                mean_online=("mean_online_return", "mean"),
                tail20_online=("tail20_online_return", "mean"),
                mean_J=("J", "mean"),
                mean_G=("G", "mean"),
                mean_E=("E", "mean"),
                c50=("collapse_50", "mean"),
            )
            .sort_values(["mean_online", "mean_J"], ascending=[False, False])
        )
        print(f"\nTop 5 for {learner_name} (by mean_online_return on seeds {tuning_seeds}):")
        print(grp.head(5).to_string(index=False), flush=True)
        best_cid = grp.iloc[0]["config_id"]
        hp_cols = [c for c in sub.columns if c.startswith("hp_")]
        best_row = sub[sub["config_id"] == best_cid].iloc[0]
        print("Best params:", {c.replace("hp_", ""): best_row[c] for c in hp_cols if pd.notna(best_row[c])}, flush=True)


if __name__ == "__main__":
    main()
