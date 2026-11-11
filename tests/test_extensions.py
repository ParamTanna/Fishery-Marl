"""Unit tests for Step 6 tabular extensions (CTDE and MemoryOne)."""

from __future__ import annotations

import numpy as np

from env import FisheryEnv
from learners.ctde import CTDEFleet, ctde_pair
from learners.memory_one import (
    MemoryOneFleet,
    forced_overharvest_probe,
    memory_one_pair,
    restraint_threshold,
)
from utils import load_config, stock_catch_grids


def test_ctde_decentralised_table_and_joint_critic_update():
    cfg = load_config()
    s_grid, a_grid = stock_catch_grids(cfg)
    f1, f2 = ctde_pair(s_grid, a_grid, cfg, seed=100)
    env = FisheryEnv(K=100.0, p=10.0, c=10.0 / 15.0, r=0.5, sigma=0.0, gamma=0.99, seed=100)
    s = env.reset()
    f1.rng = np.random.default_rng(7)
    r1 = f1.act(s, explore=True)
    r2 = 5.0
    nxt, rewards, _done, info = env.step([r1, r2])
    f1.observe(s, r1, info["executed"], float(rewards[0]), nxt)
    # Deterministic nonzero catch update at full stock s=K
    f1._last_index = 5
    f1.observe(100.0, 10.0, [10.0, 5.0], 66.67, 85.0)

    tab1 = f1.greedy_table()
    assert tab1.shape == (s_grid.shape[0], a_grid.shape[0])
    assert np.allclose(tab1.sum(axis=1), 1.0)
    assert f1.Q.shape == (s_grid.shape[0], a_grid.shape[0], a_grid.shape[0])
    assert f1.opp_counts[-1].sum() > 0.0
    assert np.any(f1.Q[-1] != 0.0)


def test_ctde_reserve_cap_respects_legal_bound():
    cfg = load_config()
    s_grid, a_grid = stock_catch_grids(cfg)
    f1 = CTDEFleet(s_grid, a_grid, gamma=0.99, fleet_idx=0, seed=100, s_res=25.0)
    for stock in (0.0, 20.0, 25.0, 50.0, 100.0):
        cap = max(0.0, 0.5 * (stock - 25.0))
        assert f1.greedy_request(stock) <= cap + 1e-12
        assert f1.act(stock, explore=True) <= cap + 1e-12


def test_memory_one_regime_switch_and_shock_probe():
    cfg = load_config()
    s_grid, a_grid = stock_catch_grids(cfg)
    f1, f2 = memory_one_pair(s_grid, a_grid, cfg, seed=100, s_res=25.0)
    env = FisheryEnv(K=100.0, p=10.0, c=10.0 / 15.0, r=0.6, sigma=0.05, gamma=0.99, seed=100)

    # Prior high catch by fleet 1 at K=100 exceeds restraint_threshold(100)=10 -> regime 1
    assert 15.0 > restraint_threshold(100.0)
    f2.begin_episode()
    assert f2.regime == 0
    _ = f2.act(100.0, explore=False)
    f2.observe(100.0, 4.0, [15.0, 4.0], 20.0, 81.0)
    assert f2.regime == 1

    # Forced overharvest probe runs cleanly with paired CRN outputs
    res = forced_overharvest_probe(f1, f2, env, shock_t=5, horizon=15, shock_request=15.0)
    assert res["regime2_post_shock"] == 1
    assert np.isfinite(res["delta_catch2_post_1"])
    assert np.isfinite(res["delta_catch2_post_6"])
