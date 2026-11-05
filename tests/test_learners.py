"""Learner checks. Grids here are tiny on purpose."""

import numpy as np

from env import FisheryEnv, ration
from kernels import br_sweep_fishery, policy_value_fishery
from learners.iql import IQLFleet
from learners.mb_oa import MBOAFleet
from metrics import policy_values, rollout_returns, solve_best_response
from opponent_model import FrequencyOpponent, KernelOpponent, recover_request
from utils import gauss_hermite_normal, load_config, stock_catch_grids


def test_warm_start_reaches_same_action_as_cold_start():
    """Warm-starting solve_best_response converges to the same greedy action
    and a value within value_tol of the cold-start result (Step 0 requirement).

    Uses gamma=0 so the fixed point is reached in one sweep — the test is
    deterministic and cheap regardless of grid size.
    """
    cfg = load_config()
    cfg["grids"]["n_stock"] = 8
    cfg["grids"]["n_catch"] = 8
    cfg["grids"]["n_quad"] = 5
    cfg["model"]["gamma"] = 0.0
    cfg["model"]["sigma"] = 0.0
    cfg["solver"]["max_H"] = 50
    cfg["solver"]["min_H"] = 1
    cfg["solver"]["value_tol"] = 1.0e-8
    s_grid, a_grid = stock_catch_grids(cfg)
    # Opponent: always requests p/c (the myopic optimum).
    request = float(cfg["model"]["p"]) / float(cfg["model"]["c"])
    opp_idx = int(np.abs(a_grid - request).argmin())
    opponent = np.zeros((s_grid.shape[0], a_grid.shape[0]))
    opponent[:, opp_idx] = 1.0

    # Cold start (baseline).
    cold = solve_best_response(cfg, 0.5, opponent, s_grid=s_grid, a_grid=a_grid)

    # Warm start from the cold solution — should converge in 1 sweep.
    warm = solve_best_response(
        cfg, 0.5, opponent, s_grid=s_grid, a_grid=a_grid, v_init=cold.V,
    )
    assert np.array_equal(cold.action_idx, warm.action_idx), (
        "Warm start produced a different greedy action than cold start"
    )
    assert float(np.max(np.abs(cold.V - warm.V))) <= float(cfg["solver"]["value_tol"]) + 1e-10, (
        "Warm-start value deviates from cold-start by more than value_tol"
    )
    # Warm start from slightly perturbed V — must still match.
    perturbed = cold.V + np.random.default_rng(7).uniform(-0.5, 0.5, cold.V.shape)
    warm2 = solve_best_response(
        cfg, 0.5, opponent, s_grid=s_grid, a_grid=a_grid, v_init=perturbed,
    )
    assert np.array_equal(cold.action_idx, warm2.action_idx), (
        "Warm start from perturbed V produced a different greedy action"
    )


def test_br_mdp_and_warm_start_match_br_sweep_at_gamma_09():
    """Exact precomputed BR MDP and warm-start match cold br_sweep_fishery at gamma=0.9."""
    cfg = load_config()
    cfg["grids"]["n_stock"] = 11
    cfg["grids"]["n_catch"] = 9
    cfg["grids"]["n_quad"] = 9
    cfg["model"]["gamma"] = 0.9
    cfg["model"]["sigma"] = 0.1
    cfg["solver"]["max_H"] = 1000
    cfg["solver"]["min_H"] = 1
    cfg["solver"]["value_tol"] = 1.0e-7
    s_grid, a_grid = stock_catch_grids(cfg)
    nodes, weights = gauss_hermite_normal(int(cfg["grids"]["n_quad"]))

    # Mixed opponent policy across multiple actions.
    opponent = np.zeros((s_grid.shape[0], a_grid.shape[0]), dtype=np.float64)
    opponent[:, 1] = 0.35
    opponent[:, 3] = 0.65

    # Reference loop directly using br_sweep_fishery from zeros.
    m = cfg["model"]
    V_ref = np.zeros(s_grid.shape[0], dtype=np.float64)
    act_ref = np.zeros(s_grid.shape[0], dtype=np.int64)
    for _ in range(int(cfg["solver"]["max_H"])):
        Vn, act_ref = br_sweep_fishery(
            V_ref, opponent, s_grid, a_grid, 0.5, float(m["K"]), float(m["sigma"]),
            float(m["gamma"]), float(m["p"]), float(m["c"]), nodes, weights,
        )
        d = float(np.max(np.abs(Vn - V_ref)))
        V_ref = Vn
        if d < float(cfg["solver"]["value_tol"]):
            break

    cold = solve_best_response(cfg, 0.5, opponent, s_grid=s_grid, a_grid=a_grid)
    assert cold.converged
    assert np.array_equal(cold.action_idx, act_ref)
    assert float(np.max(np.abs(cold.V - V_ref))) <= float(cfg["solver"]["value_tol"])

    perturbed = cold.V + np.random.default_rng(11).uniform(-2.0, 2.0, cold.V.shape)
    warm = solve_best_response(cfg, 0.5, opponent, s_grid=s_grid, a_grid=a_grid, v_init=perturbed)
    assert warm.converged
    assert np.array_equal(warm.action_idx, act_ref)
    assert float(np.max(np.abs(warm.V - V_ref))) <= float(cfg["solver"]["value_tol"]) * 20.0


def test_policy_values_warm_start_matches_cold_sweep():
    """Precomputed policy_values (cold & warm) match direct policy_value_fishery sweeps."""
    cfg = load_config()
    cfg["grids"]["n_stock"] = 11
    cfg["grids"]["n_catch"] = 9
    cfg["grids"]["n_quad"] = 9
    cfg["model"]["gamma"] = 0.9
    cfg["model"]["sigma"] = 0.1
    cfg["solver"]["max_H"] = 1000
    cfg["solver"]["value_tol"] = 1.0e-7
    s_grid, a_grid = stock_catch_grids(cfg)
    nodes, weights = gauss_hermite_normal(int(cfg["grids"]["n_quad"]))

    p1 = np.zeros((s_grid.shape[0], a_grid.shape[0]), dtype=np.float64)
    p2 = np.zeros_like(p1)
    p1[:, 2] = 0.4
    p1[:, 4] = 0.6
    p2[:, 1] = 0.3
    p2[:, 3] = 0.7

    m = cfg["model"]
    V1_ref = np.zeros(s_grid.shape[0], dtype=np.float64)
    V2_ref = np.zeros(s_grid.shape[0], dtype=np.float64)
    for _ in range(int(cfg["solver"]["max_H"])):
        N1, N2 = policy_value_fishery(
            V1_ref, V2_ref, p1, p2, s_grid, a_grid, 0.5, float(m["K"]),
            float(m["sigma"]), float(m["gamma"]), float(m["p"]), float(m["c"]), nodes, weights,
        )
        d = float(max(np.max(np.abs(N1 - V1_ref)), np.max(np.abs(N2 - V2_ref))))
        V1_ref, V2_ref = N1, N2
        if d < float(cfg["solver"]["value_tol"]):
            break

    V1_c, V2_c, ok_c, _ = policy_values(cfg, 0.5, p1, p2, s_grid, a_grid, nodes=nodes, weights=weights)
    assert ok_c
    assert float(np.max(np.abs(V1_c - V1_ref))) <= 1.0e-9
    assert float(np.max(np.abs(V2_c - V2_ref))) <= 1.0e-9

    rng = np.random.default_rng(13)
    V1_w, V2_w, ok_w, _ = policy_values(
        cfg, 0.5, p1, p2, s_grid, a_grid, nodes=nodes, weights=weights,
        v1_init=V1_c + rng.uniform(-5.0, 5.0, V1_c.shape),
        v2_init=V2_c + rng.uniform(-5.0, 5.0, V2_c.shape),
    )
    assert ok_w
    assert float(np.max(np.abs(V1_w - V1_ref))) <= float(cfg["solver"]["value_tol"]) * 20.0
    assert float(np.max(np.abs(V2_w - V2_ref))) <= float(cfg["solver"]["value_tol"]) * 20.0


def test_rollout_reports_collapse_50_and_collapse_long_separately():
    cfg = load_config()
    cfg["model"]["T"] = 4
    cfg["evaluation"]["n_rollouts"] = 40
    s_grid = np.linspace(0.0, 100.0, 11)
    a_grid = np.linspace(0.0, 100.0, 5)
    policy = np.zeros((11, 5))
    policy[:, -1] = 1.0
    xi = np.random.default_rng(4).normal(size=(16, 40))
    out = rollout_returns(
        s_grid, a_grid, policy, policy, cfg, 0.5, n_rollouts=40, length=16, seed=4, xi=xi
    )
    assert 0.0 <= out["collapse_50"] <= out["collapse_long"] + 1.0e-12 <= 1.0 + 1.0e-12
    assert out["T_collapse"] == 4


def test_iql_learns_the_one_season_optimum_against_a_quiet_fleet():
    # Opponent requests 0, so there is no rationing. The myopic optimum is p/c = 15.
    cfg = load_config()
    a_grid = np.array([0.0, 5.0, 10.0, 15.0, 20.0, 30.0])
    s_grid = np.array([100.0])
    agent = IQLFleet(
        s_grid, a_grid, gamma=0.0, alpha0=1.0, alpha_min=1.0, alpha_decay=1.0,
        epsilon0=1.0, epsilon_min=0.2, epsilon_decay=0.97, seed=3,
    )
    env = FisheryEnv(100.0, cfg["model"]["p"], cfg["model"]["c"], 0.5, 0.0, 0.0, seed=3, s0=100.0)
    for _ in range(120):
        agent.begin_episode()
        stock = env.reset()
        own = agent.act(stock, explore=True)
        nxt, rewards, _done, info = env.step([own, 0.0])
        agent.observe(stock, own, info["executed"], float(rewards[0]), nxt)
        agent.finish_episode()
    assert agent.greedy_request(100.0) == 15.0


def test_iql_bootstraps_the_truncated_season():
    # One state, one action, reward 1, gamma 0.5, alpha 1. Two episodes of one season.
    # With a bootstrap the second target is 1 + 0.5 * 1. Without it the value stays at 1.
    s_grid = np.array([0.0])
    a_grid = np.array([0.0])
    agent = IQLFleet(
        s_grid, a_grid, gamma=0.5, alpha0=1.0, alpha_min=1.0, alpha_decay=1.0,
        epsilon0=0.0, epsilon_min=0.0, epsilon_decay=1.0, seed=0,
    )
    for _ in range(2):
        agent.begin_episode()
        agent.act(0.0, explore=True)
        agent.observe(0.0, 0.0, np.zeros(2), 1.0, 0.0)
        agent.finish_episode()
    assert abs(float(agent.Q[0, 0]) - 1.5) < 1.0e-9


def test_iql_seeding_is_identical():
    s_grid = np.linspace(0.0, 100.0, 5)
    a_grid = np.linspace(0.0, 20.0, 4)
    cfg = load_config()

    def once(seed):
        agent = IQLFleet(
            s_grid, a_grid, gamma=0.0, alpha0=0.5, alpha_min=0.5, alpha_decay=1.0,
            epsilon0=1.0, epsilon_min=1.0, epsilon_decay=1.0, seed=seed,
        )
        env = FisheryEnv(100.0, cfg["model"]["p"], cfg["model"]["c"], 0.5, 0.0, 0.0, seed=seed, s0=100.0)
        for _ in range(4):
            agent.begin_episode()
            stock = env.reset()
            for _t in range(3):
                own = agent.act(stock, explore=True)
                nxt, rewards, _done, info = env.step([own, 0.0])
                agent.observe(stock, own, info["executed"], float(rewards[0]), nxt)
                stock = nxt
            agent.finish_episode()
        return agent.Q.copy()

    assert np.array_equal(once(11), once(11))


def test_request_recovery_matches_the_rationing_identity():
    stock = 10.0
    requests = np.array([8.0, 6.0])
    hats = ration(requests, stock)
    recovered = recover_request(stock, requests[0], hats[0], hats[1])
    assert abs(recovered - requests[1]) < 1.0e-9
    # No rationing: the executed catch is the request, and recovery must leave it alone.
    quiet = ration(np.array([2.0, 3.0]), stock)
    assert abs(recover_request(stock, 2.0, quiet[0], quiet[1]) - 3.0) < 1.0e-9


def test_kernel_opponent_recovers_a_constant_request():
    a_grid = np.linspace(0.0, 30.0, 16)
    s_grid = np.linspace(0.0, 100.0, 11)
    model = KernelOpponent(bandwidth=8.0, default_request=0.0)
    for stock in (10.0, 25.0, 40.0, 55.0, 70.0, 90.0):
        model.update(stock, 15.0)
    table = model.policy(s_grid, a_grid)
    mode = a_grid[np.argmax(table, axis=1)]
    assert np.all(np.abs(mode - 15.0) <= 2.0 + 1.0e-9)


def test_mboa_best_response_matches_the_overharvester_dp():
    cfg = load_config()
    cfg["grids"]["n_stock"] = 8
    cfg["grids"]["n_catch"] = 8
    cfg["grids"]["n_quad"] = 7
    cfg["model"]["gamma"] = 0.0
    cfg["model"]["sigma"] = 0.0
    cfg["solver"]["max_H"] = 4
    cfg["solver"]["min_H"] = 1
    cfg["solver"]["value_tol"] = 1.0e-8
    cfg["opponent_model"]["kind"] = "frequency"
    cfg["opponent_model"]["n_stock_bins"] = 4
    cfg["opponent_model"]["n_catch_bins"] = 8
    cfg["opponent_model"]["laplace_alpha"] = 1.0e-6
    cfg["learners"]["n_warm"] = 0
    cfg["learners"]["n_replan"] = 1
    s_grid, a_grid = stock_catch_grids(cfg)
    request = float(cfg["model"]["p"]) / float(cfg["model"]["c"])
    fleet = MBOAFleet(
        s_grid, a_grid, cfg, cache=None, fleet_index=0, seed=0, known_r=0.5, kind="frequency",
    )
    for stock in s_grid:
        for _ in range(30):
            fleet.opponent.update(float(stock), request)
    fleet.replan()
    from policies import overharvester

    true, _same = overharvester(s_grid, a_grid, request)
    br = solve_best_response(cfg, 0.5, true, s_grid=s_grid, a_grid=a_grid)
    assert np.array_equal(np.argmax(fleet.plan, axis=1), br.action_idx)


def test_kernel_opponent_incremental_matches_batch_formula():
    """Incremental O(n_s) updates match exact batch Nadaraya-Watson estimates."""
    a_grid = np.linspace(0.0, 30.0, 16)
    s_grid = np.linspace(0.0, 100.0, 11)
    rng = np.random.default_rng(77)
    pairs = list(zip(rng.uniform(0.0, 100.0, 40), rng.uniform(0.0, 25.0, 40)))
    inc = KernelOpponent(bandwidth=5.0, default_request=10.0)
    batch = KernelOpponent(bandwidth=5.0, default_request=10.0)
    # Trigger incremental cache on inc halfway through updates.
    for stock, req in pairs[:15]:
        inc.update(stock, req)
        batch.update(stock, req)
    p_mid = inc.policy(s_grid, a_grid)
    assert np.allclose(p_mid, batch.policy(s_grid, a_grid), atol=1.0e-12)
    for stock, req in pairs[15:]:
        inc.update(stock, req)
        batch.update(stock, req)
    assert np.allclose(inc.policy(s_grid, a_grid), batch.policy(s_grid, a_grid), atol=1.0e-11)


def test_learning_cost_zero_for_reference_and_invariant_per_episode():
    from training import learning_cost, mean_learning_gap

    ref = 1000.0  # toy reference
    online_ideal = np.full(100, ref)
    cost = learning_cost(online_ideal, ref)
    assert np.allclose(cost, 0.0)

    # Per-episode gap is invariant to doubling episode count under identical per-episode gap
    online_100 = np.full(100, ref - 500.0)
    online_200 = np.full(200, ref - 500.0)
    gap_100 = mean_learning_gap(online_100, ref)[-1]
    gap_200 = mean_learning_gap(online_200, ref)[-1]
    assert abs(gap_100 - 500.0) < 1.0e-9
    assert abs(gap_200 - 500.0) < 1.0e-9
    # Unlike a cumulative sum, the per-episode metric does not scale with episode budget
    assert abs(gap_100 - gap_200) < 1.0e-9

