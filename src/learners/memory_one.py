"""Tabular memory-one Q-learning fleet and forced over-harvest shock probe.

Conditioning on previous stock and executed opponent catch (st, s_{t-1}, hat_a_{t-1})
is discretised via a finite-history regime bit: regime_t = 1 if the rival's
previous executed catch exceeded a restrained share at s_{t-1}, else 0.
"""

from __future__ import annotations

import numpy as np

from env import next_stock, ration
from learners.common import one_hot_greedy, stock_index


def restraint_threshold(prev_stock):
    """Restrained single-fleet catch threshold as a function of prior stock s_{t-1}."""
    return float(np.clip(0.10 * float(prev_stock), 2.0, 10.0))


class MemoryOneFleet:
    """Tabular Q(s_t, regime_t, a_i) learner over finite-history memory-one regimes."""

    def __init__(
        self,
        s_grid,
        a_grid,
        gamma,
        fleet_idx=0,
        alpha0=0.2,
        alpha_min=0.02,
        alpha_decay=0.9995,
        epsilon0=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.999,
        seed=0,
        s_res=None,
    ):
        self.s_grid = np.asarray(s_grid, dtype=np.float64)
        self.a_grid = np.asarray(a_grid, dtype=np.float64)
        self.gamma = float(gamma)
        self.fleet_idx = int(fleet_idx)
        self.alpha0 = float(alpha0)
        self.alpha_min = float(alpha_min)
        self.alpha_decay = float(alpha_decay)
        self.epsilon0 = float(epsilon0)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)
        self.s_res = None if s_res is None else float(s_res)
        self.rng = np.random.default_rng(int(seed))

        self.Q = np.zeros((self.s_grid.shape[0], 2, self.a_grid.shape[0]), dtype=np.float64)
        self.updates = 0
        self.episodes = 0
        self.regime = 0
        self._last_index = 0

    def alpha(self):
        return max(self.alpha_min, self.alpha0 * (self.alpha_decay ** self.updates))

    def epsilon(self):
        return max(self.epsilon_min, self.epsilon0 * (self.epsilon_decay ** self.episodes))

    def begin_episode(self):
        self.regime = 0

    def finish_episode(self):
        self.episodes += 1

    def _allowed_mask(self, stock):
        if self.s_res is None:
            return slice(None)
        cap = max(0.0, 0.5 * (float(stock) - self.s_res))
        mask = self.a_grid <= cap + 1e-12
        if not np.any(mask):
            mask[0] = True
        return mask

    def _greedy_action_index(self, state_idx, regime):
        q_row = self.Q[state_idx, int(regime)]
        mask = self._allowed_mask(self.s_grid[state_idx])
        if isinstance(mask, slice):
            return int(np.argmax(q_row))
        masked = np.where(mask, q_row, -np.inf)
        return int(np.argmax(masked))

    def act(self, stock, explore=True):
        index = stock_index(stock, self.s_grid)
        if explore and self.rng.random() < self.epsilon():
            mask = self._allowed_mask(stock)
            if isinstance(mask, slice):
                action = int(self.rng.integers(0, self.a_grid.shape[0]))
            else:
                legal = np.flatnonzero(mask)
                action = int(legal[int(self.rng.integers(0, legal.shape[0]))])
        else:
            action = self._greedy_action_index(index, self.regime)
        self._last_index = action
        return float(self.a_grid[action])

    def observe(self, stock, own_request, hats, reward, next_stock, learn_reward=None):
        del own_request
        state = stock_index(stock, self.s_grid)
        nxt = stock_index(next_stock, self.s_grid)
        other_hat = float(hats[1 - self.fleet_idx])
        next_regime = 1 if other_hat > restraint_threshold(stock) else 0

        used = float(reward if learn_reward is None else learn_reward)
        mask_next = self._allowed_mask(next_stock)
        q_next = self.Q[nxt, next_regime]
        if not isinstance(mask_next, slice):
            q_next = np.where(mask_next, q_next, -np.inf)
        target = used + self.gamma * float(np.max(q_next))

        step = self.alpha()
        self.Q[state, self.regime, self._last_index] += step * (
            target - self.Q[state, self.regime, self._last_index]
        )
        self.updates += 1
        self.regime = next_regime

    def greedy_table(self, regime=0):
        choice = np.array(
            [self._greedy_action_index(s, regime) for s in range(self.s_grid.shape[0])],
            dtype=np.int64,
        )
        return one_hot_greedy(choice, self.s_grid.shape[0], self.a_grid.shape[0])

    def greedy_request(self, stock, regime=None):
        reg = self.regime if regime is None else int(regime)
        return float(self.a_grid[self._greedy_action_index(stock_index(stock, self.s_grid), reg)])

    def policy_reaction_gap(self, min_stock=None):
        """Mean (a(s, regime=1) - a(s, regime=0)) across states >= min_stock.

        Positive = retaliatory catch increase after partner over-harvest;
        negative = catch restraint / cut after partner over-harvest.
        """
        mask = np.ones(self.s_grid.shape[0], dtype=bool)
        if min_stock is not None:
            mask &= self.s_grid >= float(min_stock)
        diffs = []
        for s_idx in np.flatnonzero(mask):
            a0 = self.a_grid[self._greedy_action_index(int(s_idx), 0)]
            a1 = self.a_grid[self._greedy_action_index(int(s_idx), 1)]
            diffs.append(float(a1 - a0))
        return float(np.mean(diffs)) if diffs else 0.0


def memory_one_pair(s_grid, a_grid, cfg, seed, s_res=None, epsilon_decay=None):
    """Build two MemoryOne fleets reusing IQL schedules from config.yaml."""
    block = cfg["learners"]["iql"]
    common = dict(
        s_grid=s_grid,
        a_grid=a_grid,
        gamma=float(cfg["model"]["gamma"]),
        alpha0=float(block["alpha0"]),
        alpha_min=float(block["alpha_min"]),
        alpha_decay=float(block["alpha_decay"]),
        epsilon0=float(block["epsilon0"]),
        epsilon_min=float(block["epsilon_min"]),
        epsilon_decay=float(block["epsilon_decay"] if epsilon_decay is None else epsilon_decay),
        s_res=s_res,
    )
    return (
        MemoryOneFleet(**common, fleet_idx=0, seed=int(seed)),
        MemoryOneFleet(**common, fleet_idx=1, seed=int(seed) + 10_000),
    )


def forced_overharvest_probe(agent1, agent2, env, shock_t=10, horizon=25, shock_request=15.0):
    """Compare Fleet 2's catch trajectory with vs without a forced t=shock_t over-harvest.

    Uses identical noise draws across shocked and unshocked paired rollouts so any
    divergence directly measures the memory-one behavioural response.
    """
    xi = env.rng.normal(size=int(horizon))

    def _rollout(apply_shock):
        agent1.begin_episode()
        agent2.begin_episode()
        s = float(env.K)
        catches2 = []
        stocks = []
        regimes2 = []
        for t in range(int(horizon)):
            stocks.append(s)
            regimes2.append(int(agent2.regime))
            req1 = float(shock_request) if (apply_shock and t == int(shock_t)) else agent1.act(s, explore=False)
            req2 = agent2.act(s, explore=False)
            h1, h2 = ration([req1, req2], s)
            x = max(0.0, s - float(h1 + h2))
            s_next = float(next_stock(x, env.r, env.K, env.sigma, float(xi[t])))
            agent1.regime = 1 if h2 > restraint_threshold(s) else 0
            agent2.regime = 1 if h1 > restraint_threshold(s) else 0
            catches2.append(float(h2))
            s = s_next
        return stocks, catches2, regimes2

    st_base, c2_base, _reg_base = _rollout(False)
    st_shk, c2_shk, reg_shk = _rollout(True)
    t_post = min(int(shock_t) + 1, int(horizon) - 1)
    t_late = min(int(shock_t) + 6, int(horizon) - 1)
    return {
        "stock_pre_shock": float(st_shk[int(shock_t)]),
        "regime2_post_shock": int(reg_shk[t_post]),
        "catch2_pre_shock": float(c2_shk[max(0, int(shock_t) - 1)]),
        "catch2_post_1_shock": float(c2_shk[t_post]),
        "catch2_post_1_base": float(c2_base[t_post]),
        "delta_catch2_post_1": float(c2_shk[t_post] - c2_base[t_post]),
        "catch2_post_6_shock": float(c2_shk[t_late]),
        "catch2_post_6_base": float(c2_base[t_late]),
        "delta_catch2_post_6": float(c2_shk[t_late] - c2_base[t_late]),
    }
