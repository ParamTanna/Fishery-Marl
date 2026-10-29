"""Shared pieces of the quick and full learner runs.

Smoke seeds are 100 and 101. They are neither tuning seeds (0, 1, 2) nor reported
seeds (10 to 29). Hyperparameters for these smoke runs are the config defaults.
A logged tuning search has not been run yet; reported runs must not use these rows.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

from cache import BenchmarkCache
from learners.common import stock_index
from learners.iql import IQLFleet
from learners.mb_eq import MBEqFleet
from learners.mb_oa import MBOAFleet
from learners.team import TeamPlannerFleet
from metrics import evaluate, rollout_returns
from policies import equilibrium_pair, overharvester, planner_share
from training import learning_cost, make_env, run_episodes
from utils import config_hash, git_commit, horizon_for_tail, stock_catch_grids, t_interval

SMOKE_SEEDS = (100, 101)


def worker_init():
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[var] = "1"


def reported_seeds(cfg):
    return [int(s) for s in cfg["seeds"]["reported_seeds"]]


def tuning_seeds(cfg):
    return [int(s) for s in cfg["seeds"]["tuning_seeds"]]


def grids_of(cfg):
    return stock_catch_grids(cfg)


def noise_matrix(cfg, seed):
    length = horizon_for_tail(float(cfg["model"]["gamma"]), float(cfg["evaluation"]["gamma_tail"]))
    n_paths = int(cfg["evaluation"]["n_rollouts"])
    return np.random.default_rng(int(seed) + 50_000).normal(size=(length, n_paths))


def prepare(cfg, r, rule=None):
    cache = BenchmarkCache(cfg, rule=rule)
    cache.ensure_rule(rule)
    plan, p1, p2 = cache.interpolated_planner(float(r))
    eq = cache.get_equilibrium(float(r), rule=rule)
    return cache, plan, p1, p2, eq


_GAP_KEYS = ("E", "E1", "E2", "gap_converged", "gap_delta", "_br1_V", "_br2_V", "_v1_V", "_v2_V")


def evaluate_pair(agent1, agent2, cfg, r, cache, plan, plan_p1, plan_p2, eq, xi, seed,
                  with_gap, br1_v_init=None, br2_v_init=None, v1_v_init=None, v2_v_init=None,
                  gap_cache=None, j_plan_cached=None, j_eq_cached=None):
    s_grid, a_grid = grids_of(cfg)
    p1 = agent1.greedy_table()
    p2 = agent2.greedy_table()
    started = time.perf_counter()
    cache_key = None
    if with_gap and gap_cache is not None:
        cache_key = (
            float(r),
            np.ascontiguousarray(p1, dtype=np.float64).tobytes(),
            np.ascontiguousarray(p2, dtype=np.float64).tobytes(),
        )
    if cache_key is not None and cache_key in gap_cache:
        row = evaluate(
            p1, p2, r, cfg, s_grid, a_grid,
            plan_p1, plan_p2, eq.p1, eq.p2,
            planner_J_dp=float(plan.V[-1]), eq_J_dp=float(eq.V1[-1] + eq.V2[-1]),
            xi=xi, seed=seed, with_gap=False,
            j_plan_cached=j_plan_cached, j_eq_cached=j_eq_cached,
        )
        row.update(gap_cache[cache_key])
    else:
        row = evaluate(
            p1, p2, r, cfg, s_grid, a_grid,
            plan_p1, plan_p2, eq.p1, eq.p2,
            planner_J_dp=float(plan.V[-1]), eq_J_dp=float(eq.V1[-1] + eq.V2[-1]),
            xi=xi, seed=seed, with_gap=with_gap,
            br1_v_init=br1_v_init, br2_v_init=br2_v_init,
            v1_init=v1_v_init, v2_init=v2_v_init,
            j_plan_cached=j_plan_cached, j_eq_cached=j_eq_cached,
        )
        if cache_key is not None and row.get("E") is not None:
            gap_cache[cache_key] = {k: row[k] for k in _GAP_KEYS}
    row["eval_seconds"] = time.perf_counter() - started
    row["seed"] = int(seed)
    row["r"] = float(r)
    row["config_hash"] = config_hash(cfg)
    row["git_commit"] = git_commit()
    return row


def train_pair(agent1, agent2, cfg, r, seed, n_episodes, eval_every, with_gap=True, rule=None, eval_episodes=None):
    cache, plan, plan_p1, plan_p2, eq = prepare(cfg, r, rule=rule)
    xi = noise_matrix(cfg, seed)
    env = make_env(cfg, r, seed)
    env.horizon = int(cfg["model"]["T"])
    env.delta = float(cfg["model"]["delta"])
    s_grid, a_grid = grids_of(cfg)

    # Precompute reference Monte Carlo totals once per (r, seed) since CRN noise xi is fixed.
    j_plan_mc = rollout_returns(s_grid, a_grid, plan_p1, plan_p2, cfg, r, seed=seed, xi=xi)["J_total"]
    j_eq_mc = rollout_returns(s_grid, a_grid, eq.p1, eq.p2, cfg, r, seed=seed, xi=xi)["J_total"]

    # Warm-start caches for best-response and fixed-profile value solves inside evaluate().
    # Also cache exact gap metrics per distinct (r, p1, p2) policy pair within a run.
    _br1_v = [None]
    _br2_v = [None]
    _v1_v = [None]
    _v2_v = [None]
    gap_cache = {}

    def at_eval(left, right):
        row = evaluate_pair(
            left, right, cfg, r, cache, plan, plan_p1, plan_p2, eq, xi, seed,
            with_gap, br1_v_init=_br1_v[0], br2_v_init=_br2_v[0],
            v1_v_init=_v1_v[0], v2_v_init=_v2_v[0], gap_cache=gap_cache,
            j_plan_cached=j_plan_mc, j_eq_cached=j_eq_mc,
        )
        # Persist the converged V's for the next evaluation call.
        if row.get("_br1_V") is not None:
            _br1_v[0] = row["_br1_V"]
            _br2_v[0] = row["_br2_V"]
        if row.get("_v1_V") is not None:
            _v1_v[0] = row["_v1_V"]
            _v2_v[0] = row["_v2_V"]
        return row

    started = time.perf_counter()
    rows, online = run_episodes(
        agent1, agent2, env, n_episodes, eval_every, at_eval, eval_episodes=eval_episodes
    )
    elapsed = time.perf_counter() - started
    j_plan = float(plan.V[-1])
    j_eq = float(eq.V1[-1] + eq.V2[-1])
    cost_plan = learning_cost(online, j_plan)
    cost_eq = learning_cost(online, j_eq)
    for row in rows:
        episode = int(row["episode"])
        row["L_tau_plan"] = float(cost_plan[episode])
        row["L_tau_eq"] = float(cost_eq[episode])
        row["cum_online_return"] = float(np.mean(online[:episode + 1]))
        row["tail20_online_return"] = float(np.mean(online[max(0, episode - 19):episode + 1]))
        row["budget_episodes"] = int(n_episodes)
        row["train_seconds"] = elapsed
        # Remove internal warm-start keys before writing to CSV.
        row.pop("_br1_V", None)
        row.pop("_br2_V", None)
        row.pop("_v1_V", None)
        row.pop("_v2_V", None)
    return rows, online, elapsed, cache, plan, eq


def fixed_tables(cfg, r, cache, plan, eq):
    s_grid, a_grid = grids_of(cfg)
    share = planner_share(plan)
    equil = equilibrium_pair(eq)
    harvest = overharvester(s_grid, a_grid, float(cfg["model"]["p"]) / float(cfg["model"]["c"]))
    return {
        "planner_share": share,
        "equilibrium": equil,
        "overharvester": harvest,
    }


def write_rows(path, rows):
    # Strip internal warm-start keys before writing.
    cleaned = [{k: v for k, v in row.items() if not k.startswith("_")} for row in rows]
    frame = pd.DataFrame(cleaned)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame


class FixedTableAgent:
    """Static opponent table wrapper implementing the learner step interface."""

    team = False

    def __init__(self, table, s_grid, a_grid, seed=0):
        self.table = np.asarray(table, dtype=np.float64)
        self.s_grid = np.asarray(s_grid, dtype=np.float64)
        self.a_grid = np.asarray(a_grid, dtype=np.float64)
        self.rng = np.random.default_rng(int(seed) + 911)
        self._cdf = np.cumsum(self.table, axis=1)
        self._cdf[:, -1] = 1.0
        self._det = np.all((self.table == 0.0) | (self.table == 1.0), axis=1)
        self._argmax = np.argmax(self.table, axis=1)

    def begin_episode(self):
        return None

    def finish_episode(self):
        return None

    def act(self, stock, explore=True):
        si = stock_index(stock, self.s_grid)
        if not explore or self._det[si]:
            return float(self.a_grid[self._argmax[si]])
        u = float(self.rng.random())
        ai = int(np.argmax(self._cdf[si] >= u))
        return float(self.a_grid[ai])

    def observe(self, stock, own_request, hats, reward, next_stock, learn_reward=None):
        return None

    def greedy_table(self):
        return self.table.copy()


def build_agent(kind, s_grid, a_grid, cfg, cache, fleet_index, seed, true_r=None, fixed_table=None):
    """Instantiate one adaptive or fixed fleet for a slot (0 or 1)."""
    if fixed_table is not None:
        return FixedTableAgent(fixed_table, s_grid, a_grid, seed=int(seed) + 53 * int(fleet_index))
    if kind in ("iql", "team_iql"):
        blk = cfg["learners"]["iql"]
        iql_seed = int(seed) if int(fleet_index) == 0 else int(seed) + 17
        return IQLFleet(
            s_grid=s_grid,
            a_grid=a_grid,
            gamma=float(cfg["model"]["gamma"]),
            alpha0=float(blk["alpha0"]),
            alpha_min=float(blk["alpha_min"]),
            alpha_decay=float(blk["alpha_decay"]),
            epsilon0=float(blk["epsilon0"]),
            epsilon_min=float(blk["epsilon_min"]),
            epsilon_decay=float(blk["epsilon_decay"]),
            seed=iql_seed,
            team=(kind == "team_iql"),
        )
    if kind == "mb_eq":
        return MBEqFleet(
            s_grid=s_grid, a_grid=a_grid, cfg=cfg, cache=cache,
            fleet_index=int(fleet_index), seed=int(seed), true_r=true_r,
            thompson=bool(cfg["learners"].get("thompson", False)),
        )
    if kind in ("mb_oa", "mb_oa_norec"):
        rec = (kind == "mb_oa")
        return MBOAFleet(
            s_grid=s_grid, a_grid=a_grid, cfg=cfg, cache=cache,
            fleet_index=int(fleet_index), seed=int(seed), true_r=true_r,
            recover_requests=rec,
        )
    if kind == "team_planner":
        return TeamPlannerFleet(
            s_grid=s_grid, a_grid=a_grid, cfg=cfg, cache=cache,
            fleet_index=int(fleet_index), seed=int(seed), true_r=true_r,
        )
    raise ValueError(f"Unknown agent kind {kind!r}")


def episodes_to_G_within(rows, frac=0.10):
    """Earliest 1-based checkpoint after which G stays within `frac` of that run's final G."""
    if not rows:
        return 0
    g_vals = [float(r["G"]) for r in rows]
    g_final = g_vals[-1]
    band = float(frac) * max(abs(g_final), 1.0e-8)
    ok = [abs(g - g_final) <= band for g in g_vals]
    # Suffix minimum: find earliest index whose suffix is all True.
    earliest = len(ok) - 1
    while earliest > 0 and ok[earliest - 1]:
        earliest -= 1
    return int(rows[earliest]["episode"]) + 1


def summarize_metrics(frame, group_cols, metric_cols, confidence=0.95):
    """Aggregate mean and 95% t-interval half-width per group."""
    rows = []
    for key, grp in frame.groupby(group_cols, sort=False, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        out = dict(zip(group_cols, key))
        out["n"] = int(len(grp))
        for col in metric_cols:
            if col not in grp.columns:
                continue
            vals = grp[col].dropna().astype(float).tolist()
            if not vals:
                continue
            mean, half = t_interval(vals, level=confidence)
            arr = np.asarray(vals, dtype=np.float64)
            se = float(np.std(arr, ddof=1) / np.sqrt(arr.size)) if arr.size >= 2 else float("nan")
            lo = mean - half if np.isfinite(half) else float("nan")
            hi = mean + half if np.isfinite(half) else float("nan")
            out[f"{col}_mean"] = mean
            out[f"{col}_lo"] = lo
            out[f"{col}_hi"] = hi
            out[f"{col}_se"] = se
            out[f"{col}_ci95"] = half
        rows.append(out)
    return pd.DataFrame(rows)
