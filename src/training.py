"""Episode loop shared by the learner experiments.

Online return includes exploration. Greedy evaluation is separate and uses common
random numbers supplied by the caller. L_tau is the cumulative gap between a
reference value and that online return.
"""

from __future__ import annotations

import numpy as np

from env import FisheryEnv


def make_env(cfg, r, seed):
    m = cfg["model"]
    return FisheryEnv(
        K=m["K"], p=m["p"], c=m["c"], r=float(r), sigma=m["sigma"], gamma=m["gamma"],
        n_fleets=int(m["n_fleets"]), seed=int(seed), s0=m["s0"],
    )


def _learn_reward(agent, rewards):
    if getattr(agent, "team", False):
        return float(np.sum(rewards))
    return None


def run_episodes(agent1, agent2, env, n_episodes, eval_every, evaluate_at, eval_episodes=None):
    """Train for n_episodes. `evaluate_at(episode)` is called on the greedy policies."""
    T = 50 if not hasattr(env, "T") else int(env.T)
    # T lives on the config, not the env. The caller passes it via env.horizon if set.
    T = int(getattr(env, "horizon", T))
    delta = float(getattr(env, "delta", 0.2))
    eval_set = None if eval_episodes is None else {int(e) for e in eval_episodes}
    rows = []
    online = []
    train_collapsed = []
    for episode in range(int(n_episodes)):
        agent1.begin_episode()
        agent2.begin_episode()
        stock = env.reset()
        total = np.zeros(2, dtype=np.float64)
        discount = 1.0
        min_stock = float(stock)
        for _t in range(T):
            request_1 = agent1.act(stock, explore=True)
            request_2 = agent2.act(stock, explore=True)
            nxt, rewards, _done, info = env.step([request_1, request_2])
            hats = info["executed"]
            agent1.observe(
                stock, request_1, hats, float(rewards[0]), nxt, _learn_reward(agent1, rewards)
            )
            agent2.observe(
                stock, request_2, hats, float(rewards[1]), nxt, _learn_reward(agent2, rewards)
            )
            total += discount * rewards
            discount *= env.gamma
            min_stock = min(min_stock, float(nxt))
            stock = nxt
        agent1.finish_episode()
        agent2.finish_episode()
        online.append(float(total.sum()))
        ep_collapsed = bool(min_stock < delta * env.K)
        train_collapsed.append(ep_collapsed)
        do_eval = (
            (episode in eval_set or episode == int(n_episodes) - 1)
            if eval_set is not None
            else (episode % int(eval_every) == 0 or episode == int(n_episodes) - 1)
        )
        if do_eval:
            row = evaluate_at(agent1, agent2)
            row["episode"] = int(episode)
            row["online_return"] = online[-1]
            row["collapse_50_train"] = ep_collapsed
            row["cum_collapse_50_train"] = float(np.mean(train_collapsed))
            rows.append(row)
    return rows, np.asarray(online, dtype=np.float64)


def learning_cost(online, reference):
    """L_tau cumulative cost of learning at every episode.

    `reference` is the 50-step discounted return J_50(pi_ref) of the reference policy
    (same T=50 horizon, greedy execution, no exploration).
    """
    gap = float(reference) - np.asarray(online, dtype=np.float64)
    return np.cumsum(gap)


def mean_learning_gap(online, reference):
    """Mean cost of learning per episode: (1/tau) * sum_{t<=tau} (J_50(ref) - online_t)."""
    online = np.asarray(online, dtype=np.float64)
    cum_online = np.cumsum(online) / np.arange(1, len(online) + 1)
    return float(reference) - cum_online
