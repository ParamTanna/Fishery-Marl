"""E7: Safe reserve-threshold exploration & equilibrium cap.

Enforces a continuous/grid reserve threshold S_res: each fleet's action is capped at
a_i <= max(0, 0.5 * (s - S_res)), so joint harvest leaves escapement >= min(s, S_res).

--tune:  Screen candidate S_res in {15.0, 20.0, 22.0, 25.0} strictly on tuning seeds
         [0, 1, 2] across r in {0.5, 0.6, 0.7, 0.8}.
--quick: Smoke run on seeds (100, 101) writing results/raw/e7_quick.csv.
--full:  Run 20 reported seeds (10..29) at the tuning-locked S_res and compare against
         unconstrained baselines, writing results/raw/e7_safe.csv & e7_summary.csv.
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
from equilibrium import _evaluate_policy, _problem, _select_one, pure_nash_mask  # noqa: E402
from kernels import build_br_mdp, build_U_fishery_sym, enum_sym_mixed, hpi_br_mdp  # noqa: E402
from learners.common import one_hot_greedy, stock_index  # noqa: E402
from learners.iql import IQLFleet  # noqa: E402
from learners.mb_eq import MBEqFleet  # noqa: E402
from learners.mb_oa import MBOAFleet  # noqa: E402
from learn_common import (  # noqa: E402
    SMOKE_SEEDS,
    evaluate_pair,
    grids_of,
    noise_matrix,
    prepare,
    reported_seeds,
    summarize_metrics,
    train_pair,
    tuning_seeds,
    worker_init,
    write_rows,
)
from metrics import BestResponseResult, policy_values, rollout_returns  # noqa: E402
from utils import gauss_hermite_normal, load_config  # noqa: E402


DEFAULT_TUNE_CANDIDATES = (15.0, 20.0, 22.0, 25.0)
BANK_R_GRID = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5], dtype=np.float64)
E7_BUDGETS = {
    "iql": (5000, [0, 49, 499, 1999, 4999]),
    "mb_eq": (500, [0, 4, 19, 99, 499]),
    "mb_oa": (200, [0, 4, 19, 99, 199]),
}


def legal_count_for_stock(stock: float, a_grid: np.ndarray, s_res: float) -> int:
    cap = max(0.0, 0.5 * (float(stock) - float(s_res)))
    idx = np.where(a_grid <= cap + 1.0e-12)[0]
    return int(np.max(idx)) + 1 if idx.size else 1


def legal_counts_grid(s_grid: np.ndarray, a_grid: np.ndarray, s_res: float) -> np.ndarray:
    return np.array([legal_count_for_stock(float(s), a_grid, s_res) for s in s_grid], dtype=np.int64)


def solve_safe_best_response(cfg, r, opponent, legal_counts, s_grid, a_grid, v_init=None):
    m = cfg["model"]
    sol = cfg["solver"]
    nodes, weights = gauss_hermite_normal(int(cfg["grids"]["n_quad"]))
    K = float(m["K"])
    sigma = float(m["sigma"])
    gamma = float(m["gamma"])
    p = float(m["p"])
    c = float(m["c"])
    n_s, n_a = s_grid.shape[0], a_grid.shape[0]
    R, T = build_br_mdp(opponent, s_grid, a_grid, float(r), K, sigma, p, c, nodes, weights)
    for si in range(n_s):
        mc = int(legal_counts[si])
        if mc < n_a:
            R[si, mc:] = -1.0e100
    T_flat = T.reshape(n_s * n_a, n_s)
    V0 = np.zeros(n_s, dtype=np.float64) if v_init is None else np.asarray(v_init, dtype=np.float64).reshape(n_s).copy()
    V, action, H, delta = hpi_br_mdp(
        R, T_flat, V0, gamma, float(sol["value_tol"]), int(sol["max_H"]), int(sol["min_H"])
    )
    policy = np.zeros_like(opponent)
    policy[np.arange(n_s), action] = 1.0
    return BestResponseResult(V, policy, action, int(H), bool(delta < float(sol["value_tol"])), float(delta))


def solve_safe_equilibrium(cfg, r, s_res, s_grid=None, a_grid=None):
    if s_grid is None or a_grid is None:
        s_grid, a_grid = grids_of(cfg)
    prob = _problem(cfg, float(r), "fishery", None, None, None)
    legal = legal_counts_grid(s_grid, a_grid, s_res)
    n_s, n_a = s_grid.shape[0], a_grid.shape[0]
    V1 = np.zeros(n_s, dtype=np.float64)
    p1 = np.zeros((n_s, n_a), dtype=np.float64)
    p1[:, 0] = 1.0
    p1_prev = p1.copy()
    best_ex = np.inf
    best_snap = None
    ok = False
    H_used = 0
    ex = np.inf

    for h in range(1, 120):
        U1 = build_U_fishery_sym(
            V1, s_grid, a_grid, float(r), prob["K"], prob["sigma"], prob["gamma"],
            prob["p"], prob["c"], prob["nodes"], prob["weights"],
        )
        if h > 1:
            ex = 0.0
            for si in range(n_s):
                mc = int(legal[si])
                u1m = U1[si, :mc, :mc]
                pp = p1[si, :mc]
                u = float(pp @ u1m @ pp)
                ex = max(ex, float(np.max(u1m @ pp) - u))
            if ex < best_ex:
                best_ex = ex
                best_snap = (V1.copy(), p1.copy(), h - 1, ex)
            if np.allclose(p1, p1_prev, atol=1.0e-8) and ex / max(abs(float(V1[-1])), 1.0e-12) <= prob["eps_rel"]:
                ok = True
                H_used = h - 1
                break
        V1n = np.zeros(n_s, dtype=np.float64)
        p1n = np.zeros((n_s, n_a), dtype=np.float64)
        for si in range(n_s):
            mc = int(legal[si])
            u_sub = U1[si, :mc, :mc][None]
            mask = pure_nash_mask(u_sub, np.transpose(u_sub, (0, 2, 1)), prob["ne_tol"])[0]
            ii, jj = np.nonzero(mask)
            cands = [
                {
                    "u1": float(u_sub[0, i, j]),
                    "u2": float(u_sub[0, j, i]),
                    "p1": np.eye(mc)[i],
                    "p2": np.eye(mc)[j],
                    "kind": 0,
                    "sym": True,
                    "i": int(i),
                    "j": int(j),
                    "fallback": False,
                }
                for i, j in zip(ii.tolist(), jj.tolist())
                if i == j
            ]
            if mc > 1:
                mi, mj, mp, mu, mn, _ = enum_sym_mixed(u_sub, prob["ne_tol"], prob["max_mixed_store"])
                for k in range(int(mn[0])):
                    i, j, pr = int(mi[0, k]), int(mj[0, k]), float(mp[0, k])
                    vec = np.zeros(mc, dtype=np.float64)
                    vec[i] = pr
                    vec[j] = 1.0 - pr
                    cands.append({
                        "u1": float(mu[0, k]),
                        "u2": float(mu[0, k]),
                        "p1": vec,
                        "p2": vec.copy(),
                        "kind": 2,
                        "sym": True,
                        "i": i,
                        "j": j,
                        "fallback": False,
                    })
            ch = _select_one(
                cands, u_sub[0], u_sub[0].T, "max_total", p1[si, :mc], p1[si, :mc],
                329, si, prob["select_tol"], True, False, 6, prob["ne_tol"],
            )
            V1n[si] = ch["u1"]
            p1n[si, :mc] = ch["p1"]
        V1n, _, _, _ = _evaluate_policy(prob, p1n, p1n, V1n, V1n, prob["value_tol"])
        p1_prev = p1
        V1, p1 = V1n, p1n
        H_used = h

    if not ok and best_snap is not None:
        V1, p1, H_used, ex = best_snap
        ok = bool(ex / max(abs(float(V1[-1])), 1.0e-12) <= prob["eps_rel"])
    return V1, p1, int(H_used), bool(ok), float(ex)


def _safe_bank_path(cfg, s_res: float) -> Path:
    n_s = int(cfg["grids"]["n_stock"])
    n_a = int(cfg["grids"]["n_catch"])
    return ROOT / "results" / "cache" / f"safe_eq_{n_s}x{n_a}_Sres{s_res:.1f}.npz"


def ensure_safe_bank(cfg, s_res: float):
    path = _safe_bank_path(cfg, s_res)
    if path.exists():
        data = np.load(path)
        return data["r_grid"], data["policies"], data["V_total"]
    s_grid, a_grid = grids_of(cfg)
    policies = np.zeros((len(BANK_R_GRID), s_grid.shape[0], a_grid.shape[0]), dtype=np.float64)
    v_tot = np.zeros((len(BANK_R_GRID), s_grid.shape[0]), dtype=np.float64)
    for k, rv in enumerate(BANK_R_GRID):
        V1, p1, *_ = solve_safe_equilibrium(cfg, float(rv), s_res, s_grid=s_grid, a_grid=a_grid)
        policies[k] = p1
        v_tot[k] = 2.0 * V1
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, r_grid=BANK_R_GRID, policies=policies, V_total=v_tot)
    return BANK_R_GRID, policies, v_tot


def _interp_safe_policy(r_hat: float, r_grid_bank: np.ndarray, policies: np.ndarray, legal: np.ndarray) -> np.ndarray:
    rv = float(np.clip(r_hat, r_grid_bank[0], r_grid_bank[-1]))
    hi = int(np.searchsorted(r_grid_bank, rv, side="right"))
    if hi <= 0:
        pol = policies[0].copy()
    elif hi >= len(r_grid_bank):
        pol = policies[-1].copy()
    else:
        lo = hi - 1
        w = (rv - float(r_grid_bank[lo])) / max(float(r_grid_bank[hi] - r_grid_bank[lo]), 1.0e-12)
        pol = (1.0 - w) * policies[lo] + w * policies[hi]
    out = np.zeros_like(pol)
    for si, mc in enumerate(legal):
        row = pol[si, : int(mc)].copy()
        s = float(row.sum())
        if s > 1.0e-12:
            out[si, : int(mc)] = row / s
        else:
            out[si, 0] = 1.0
    return out


class SafeIQLFleet(IQLFleet):
    def __init__(self, *args, s_res: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.s_res = float(s_res)
        self.legal = legal_counts_grid(self.s_grid, self.a_grid, self.s_res)

    def act(self, stock, explore=True):
        idx = stock_index(stock, self.s_grid)
        mc = min(int(self.legal[idx]), legal_count_for_stock(stock, self.a_grid, self.s_res))
        if explore and self.rng.random() < self.epsilon():
            action = int(self.rng.integers(0, mc))
        else:
            action = int(np.argmax(self.Q[idx, :mc]))
        self._last_index = action
        return float(self.a_grid[action])

    def observe(self, stock, own_request, hats, reward, next_stock, learn_reward=None):
        del own_request, hats
        state = stock_index(stock, self.s_grid)
        nxt = stock_index(next_stock, self.s_grid)
        mc_nxt = int(self.legal[nxt])
        used = float(reward if learn_reward is None else learn_reward)
        target = used + self.gamma * float(np.max(self.Q[nxt, :mc_nxt]))
        self.Q[state, self._last_index] += self.alpha() * (target - self.Q[state, self._last_index])
        self.updates += 1

    def greedy_table(self):
        choices = np.array([int(np.argmax(self.Q[si, : int(self.legal[si])])) for si in range(len(self.s_grid))], dtype=np.int64)
        return one_hot_greedy(choices, self.s_grid.shape[0], self.a_grid.shape[0])


class SafeMBEqFleet(MBEqFleet):
    def __init__(self, *args, s_res: float, safe_bank, **kwargs):
        super().__init__(*args, **kwargs)
        self.s_res = float(s_res)
        self.legal = legal_counts_grid(self.s_grid, self.a_grid, self.s_res)
        self.bank_r, self.bank_pol, _ = safe_bank

    def replan(self):
        r_hat = self._r_hat()
        self.plan = _interp_safe_policy(r_hat, self.bank_r, self.bank_pol, self.legal)

    def act(self, stock, explore=True):
        if self.plan is None and self.season >= self.n_warm:
            self.replan()
        mc = legal_count_for_stock(stock, self.a_grid, self.s_res)
        if explore and self.season < self.n_warm:
            if self.rng.random() < self.epsilon_warm:
                action = int(self.rng.integers(0, mc))
            else:
                action = min(self.safe_index, mc - 1)
        elif self.plan is None:
            action = min(self.safe_index, mc - 1)
        else:
            idx = stock_index(stock, self.s_grid)
            mc = min(mc, int(self.legal[idx]))
            action = int(np.argmax(self.plan[idx, :mc]))
        self._last_request = float(self.a_grid[action])
        return self._last_request


class SafeMBOAFleet(MBOAFleet):
    def __init__(self, *args, s_res: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.s_res = float(s_res)
        self.legal = legal_counts_grid(self.s_grid, self.a_grid, self.s_res)

    def replan(self):
        r_hat = self._r_hat()
        opp = self.opponent.policy(self.s_grid, self.a_grid)
        br = solve_safe_best_response(self.cfg, r_hat, opp, self.legal, self.s_grid, self.a_grid, v_init=self._br_v_cache)
        self._br_v_cache = br.V.copy()
        self.plan = br.policy

    def act(self, stock, explore=True):
        if self.plan is None and self.season >= self.n_warm:
            self.replan()
        mc = legal_count_for_stock(stock, self.a_grid, self.s_res)
        if explore and self.season < self.n_warm:
            if self.rng.random() < self.epsilon_warm:
                action = int(self.rng.integers(0, mc))
            else:
                action = min(self.safe_index, mc - 1)
        elif self.plan is None:
            action = min(self.safe_index, mc - 1)
        else:
            idx = stock_index(stock, self.s_grid)
            mc = min(mc, int(self.legal[idx]))
            action = int(np.argmax(self.plan[idx, :mc]))
        self._last_request = float(self.a_grid[action])
        return self._last_request


def _build_safe_pair(kind, s_grid, a_grid, cfg, cache, seed, s_res, safe_bank, r):
    if kind == "iql":
        b = cfg["learners"]["iql"]
        kw = dict(
            s_grid=s_grid, a_grid=a_grid, gamma=float(cfg["model"]["gamma"]),
            alpha0=float(b["alpha0"]), alpha_min=float(b["alpha_min"]), alpha_decay=float(b["alpha_decay"]),
            epsilon0=float(b["epsilon0"]), epsilon_min=float(b["epsilon_min"]), epsilon_decay=float(b["epsilon_decay"]),
            s_res=s_res,
        )
        return SafeIQLFleet(seed=int(seed), **kw), SafeIQLFleet(seed=int(seed) + 17, **kw)
    if kind == "mb_eq":
        kw = dict(s_grid=s_grid, a_grid=a_grid, cfg=cfg, cache=cache, true_r=r, s_res=s_res, safe_bank=safe_bank)
        return SafeMBEqFleet(fleet_index=0, seed=int(seed), **kw), SafeMBEqFleet(fleet_index=1, seed=int(seed), **kw)
    if kind == "mb_oa":
        kw = dict(s_grid=s_grid, a_grid=a_grid, cfg=cfg, cache=cache, true_r=r, s_res=s_res, recover_requests=True)
        return SafeMBOAFleet(fleet_index=0, seed=int(seed), **kw), SafeMBOAFleet(fleet_index=1, seed=int(seed), **kw)
    raise ValueError(kind)


def _run_safe_task(task):
    kind, r, seed, s_res, n_ep, cks = task
    cfg = copy.deepcopy(load_config(quick=False))
    s_grid, a_grid = grids_of(cfg)
    cache, plan, p1_plan, p2_plan, eq_un = prepare(cfg, r)
    safe_bank = ensure_safe_bank(cfg, s_res)
    a1, a2 = _build_safe_pair(kind, s_grid, a_grid, cfg, cache, seed, s_res, safe_bank, r)

    rows, *_ = train_pair(
        a1, a2, cfg, r, seed, n_episodes=n_ep, eval_every=n_ep,
        with_gap=True, eval_episodes=cks,
    )
    legal = legal_counts_grid(s_grid, a_grid, s_res)
    p1_fin = a1.greedy_table()
    p2_fin = a2.greedy_table()
    br1_safe = solve_safe_best_response(cfg, r, p2_fin, legal, s_grid, a_grid)
    br2_safe = solve_safe_best_response(cfg, r, p1_fin, legal, s_grid, a_grid)
    v1_arr, v2_arr, *_ = policy_values(cfg, r, p1_fin, p2_fin, s_grid=s_grid, a_grid=a_grid)
    e_safe = float((br1_safe.V[-1] - v1_arr[-1]) + (br2_safe.V[-1] - v2_arr[-1]))

    r_idx = int(np.argmin(np.abs(safe_bank[0] - r)))
    j_safe_eq_dp = float(safe_bank[2][r_idx, -1])
    j_un_eq_dp = float(eq_un.V1[-1] + eq_un.V2[-1])

    last = len(rows) - 1
    for i, row in enumerate(rows):
        row["learner"] = kind
        row["safe"] = True
        row["s_res"] = float(s_res)
        row["is_final"] = (i == last)
        row["E_safe"] = e_safe if i == last else float("nan")
        row["J_eq_unsafe_dp"] = j_un_eq_dp
        row["J_eq_safe_dp"] = j_safe_eq_dp
        row["J_eq_lost"] = j_un_eq_dp - j_safe_eq_dp
    return rows


def _run_tune(workers):
    cfg = load_config(quick=False)
    r_vals = [float(v) for v in cfg["model"]["r_values"]]
    seeds = tuning_seeds(cfg)
    s_grid, a_grid = grids_of(cfg)
    summary_rows = []
    for s_res in DEFAULT_TUNE_CANDIDATES:
        safe_bank = ensure_safe_bank(cfg, s_res)
        # First check Safe Equilibrium rollout collapse on tuning seeds
        for r in r_vals:
            V1, p1, H_eq, ok, ex = solve_safe_equilibrium(cfg, r, s_res, s_grid, a_grid)
            c50s = [rollout_returns(s_grid, a_grid, p1, p1, cfg, r, seed=sd, xi=noise_matrix(cfg, sd))["collapse_50"] for sd in seeds]
            _, _, _, _, eq_un = prepare(cfg, r)
            j_un = float(eq_un.V1[-1] + eq_un.V2[-1])
            summary_rows.append({
                "kind": "safe_eq", "s_res": s_res, "r": r, "H_eq": H_eq, "eq_ok": ok,
                "J_safe_dp": 2.0 * float(V1[-1]), "J_lost_vs_eq_un": j_un - 2.0 * float(V1[-1]),
                "collapse_50": float(np.mean(c50s)), "cum_collapse_50_train": 0.0,
            })

        tasks = [
            (k, r, sd, s_res, E7_BUDGETS[k][0], [E7_BUDGETS[k][0] - 1])
            for k in ("iql", "mb_eq", "mb_oa")
            for r in r_vals
            for sd in seeds
        ]
        all_rows = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers, initializer=worker_init) as pool:
            for r_rows in pool.map(_run_safe_task, tasks, chunksize=2):
                all_rows.extend(r_rows)
        df = pd.DataFrame(all_rows)
        fin = df[df["is_final"] == True]  # noqa: E712
        for (k, r), grp in fin.groupby(["learner", "r"]):
            summary_rows.append({
                "kind": k, "s_res": s_res, "r": float(r), "H_eq": np.nan, "eq_ok": True,
                "J_safe_dp": float(grp["J"].mean()),
                "J_lost_vs_eq_un": float(grp["J_eq_unsafe_dp"].mean() - grp["J"].mean()),
                "collapse_50": float(grp["collapse_50"].mean()),
                "cum_collapse_50_train": float(grp["cum_collapse_50_train"].mean()),
            })
        tdf = pd.DataFrame([row for row in summary_rows if row["s_res"] == s_res])
        max_c50 = float(tdf["collapse_50"].max())
        max_ctr = float(tdf["cum_collapse_50_train"].max())
        print(f"Tune S_res={s_res:.1f}: max(c50)={max_c50:.4f}, max(c50_train)={max_ctr:.4f}, mean(J)={tdf['J_safe_dp'].mean():.1f}", flush=True)
    out = write_rows(ROOT / "results" / "raw" / "e7_tune.csv", summary_rows)
    print(out.to_string(index=False), flush=True)


def _run_quick(cfg, s_res):
    s_grid, a_grid = grids_of(cfg)
    cache = BenchmarkCache(cfg)
    cache.ensure_rule()
    safe_bank = ensure_safe_bank(cfg, s_res)
    r = float(cfg["model"]["r_values"][0])
    rows = []
    for seed in SMOKE_SEEDS:
        for k in ("iql", "mb_eq", "mb_oa"):
            a1, a2 = _build_safe_pair(k, s_grid, a_grid, cfg, cache, seed, s_res, safe_bank, r)
            tr, *_ = train_pair(a1, a2, cfg, r, seed, n_episodes=10, eval_every=10, with_gap=True, eval_episodes=[9])
            for row in tr:
                row["learner"] = k
                row["safe"] = True
                row["s_res"] = float(s_res)
                row["is_final"] = True
            rows.extend(tr)
    frame = write_rows(ROOT / "results" / "raw" / "e7_quick.csv", rows)
    print("rows", len(frame), flush=True)


def _run_full(workers, s_res):
    cfg = load_config(quick=False)
    BenchmarkCache(cfg).ensure_rule()
    ensure_safe_bank(cfg, s_res)
    r_vals = [float(v) for v in cfg["model"]["r_values"]]
    seeds = reported_seeds(cfg)

    tasks = [
        (k, r, sd, s_res, E7_BUDGETS[k][0], E7_BUDGETS[k][1])
        for k in ("iql", "mb_eq", "mb_oa")
        for r in r_vals
        for sd in seeds
    ]
    print(f"Launching E7 full (S_res={s_res}): {len(tasks)} runs across {workers} workers...", flush=True)
    t0 = time.perf_counter()
    safe_rows = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers, initializer=worker_init) as pool:
        for i, r_rows in enumerate(pool.map(_run_safe_task, tasks, chunksize=4), 1):
            safe_rows.extend(r_rows)
            if i % 80 == 0 or i == len(tasks):
                print(f"  E7 progress: {i}/{len(tasks)} ({(time.perf_counter()-t0)/60:.1f}m)", flush=True)

    # Also attach unconstrained baseline rows at gamma=0.99, sigma=0.1 if e2_learners.csv exists.
    e2_path = ROOT / "results" / "raw" / "e2_learners.csv"
    if e2_path.exists():
        e2 = pd.read_csv(e2_path)
        un = e2[(e2["is_final"] == True) & (e2["gamma"] == 0.99) & (e2["sigma"] == 0.1) & (e2["is_sanity"] == False)].copy()  # noqa: E712
        un["safe"] = False
        un["s_res"] = 0.0
        combined = pd.concat([pd.DataFrame(safe_rows), un], ignore_index=True)
        write_rows(ROOT / "results" / "raw" / "e7_safe.csv", combined.to_dict("records"))
    else:
        combined = write_rows(ROOT / "results" / "raw" / "e7_safe.csv", safe_rows)

    fin = combined[combined["is_final"] == True]  # noqa: E712
    metrics = ["J", "G", "E", "E_safe", "collapse_50", "collapse_long", "cum_collapse_50_train", "long_run_stock", "J_eq_lost"]
    summary = summarize_metrics(fin, ["learner", "safe", "s_res", "r"], metrics)
    write_rows(ROOT / "results" / "raw" / "e7_summary.csv", summary.to_dict("records"))
    print(f"E7 completed in {(time.perf_counter()-t0)/60:.2f}m", flush=True)


def main():
    parser = argparse.ArgumentParser(description="E7 safe exploration reserve-threshold cap")
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--s-res", type=float, default=25.0)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    if args.tune:
        _run_tune(workers=args.workers)
    elif args.full and not args.quick:
        _run_full(workers=args.workers, s_res=args.s_res)
    elif args.quick:
        _run_quick(load_config(quick=True), s_res=args.s_res)
    else:
        raise SystemExit("pass --tune, --quick, or --full")


if __name__ == "__main__":
    main()
