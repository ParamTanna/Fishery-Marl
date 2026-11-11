"""Tabular Centralised Training with Decentralised Execution (CTDE) learner.

During training each fleet maintains a joint-action Q-critic Q_i(s, a_i, a_j)
over the benchmark state and catch grids plus an empirical opponent action
model hat_pi_{-i}(a_j | s) recovered from executed catches. At execution time,
each fleet selects a decentralised action conditioned solely on local stock s by
maximizing E_{a_j ~ hat_pi_{-i}(.|s)}[Q_i(s, a_i, a_j)].
"""

from __future__ import annotations

import numpy as np

from learners.common import nearest_catch, one_hot_greedy, stock_index
from opponent_model import recover_request


class CTDEFleet:
    """Joint-action tabular Q-critic with marginalised decentralised execution."""

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
        smooth=1e-3,
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
        self.smooth = float(smooth)
        self.rng = np.random.default_rng(int(seed))

        n_s = self.s_grid.shape[0]
        n_a = self.a_grid.shape[0]
        self.Q = np.zeros((n_s, n_a, n_a), dtype=np.float64)
        self.opp_counts = np.full((n_s, n_a), self.smooth, dtype=np.float64)
        self.updates = 0
        self.episodes = 0
        self._last_index = 0

    def alpha(self):
        return max(self.alpha_min, self.alpha0 * (self.alpha_decay ** self.updates))

    def epsilon(self):
        return max(self.epsilon_min, self.epsilon0 * (self.epsilon_decay ** self.episodes))

    def begin_episode(self):
        return None

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

    def _opp_dist(self, state_idx):
        row = self.opp_counts[state_idx]
        total = float(np.sum(row))
        if total <= 0.0:
            return np.full(self.a_grid.shape[0], 1.0 / self.a_grid.shape[0], dtype=np.float64)
        return row / total

    def _marginal_q_row(self, state_idx):
        # Marginalize centralised critic over empirical rival catch distribution.
        return self.Q[state_idx] @ self._opp_dist(state_idx)

    def _greedy_action_index(self, state_idx):
        q_row = self._marginal_q_row(state_idx)
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
            action = self._greedy_action_index(index)
        self._last_index = action
        return float(self.a_grid[action])

    def observe(self, stock, own_request, hats, reward, next_stock, learn_reward=None):
        state = stock_index(stock, self.s_grid)
        nxt = stock_index(next_stock, self.s_grid)
        other_idx = 1 - self.fleet_idx
        opp_req = recover_request(stock, own_request, hats[self.fleet_idx], hats[other_idx])
        j_idx = nearest_catch(opp_req, self.a_grid)
        self.opp_counts[state, j_idx] += 1.0

        used = float(reward if learn_reward is None else learn_reward)
        # Expected best-response value at next_stock over marginalised joint critic.
        q_next_marg = self.Q[nxt] @ self._opp_dist(nxt)
        mask_next = self._allowed_mask(next_stock)
        if not isinstance(mask_next, slice):
            q_next_marg = np.where(mask_next, q_next_marg, -np.inf)
        v_next = float(np.max(q_next_marg))

        target = used + self.gamma * v_next
        step = self.alpha()
        a_idx = self._last_index
        self.Q[state, a_idx, j_idx] += step * (target - self.Q[state, a_idx, j_idx])
        self.updates += 1

    def greedy_table(self):
        choice = np.array([self._greedy_action_index(s) for s in range(self.s_grid.shape[0])], dtype=np.int64)
        return one_hot_greedy(choice, self.s_grid.shape[0], self.a_grid.shape[0])

    def greedy_request(self, stock):
        idx = self._greedy_action_index(stock_index(stock, self.s_grid))
        return float(self.a_grid[idx])


def ctde_pair(s_grid, a_grid, cfg, seed, s_res=None, epsilon_decay=None):
    """Build two CTDE fleets reusing IQL schedules from config.yaml."""
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
        CTDEFleet(**common, fleet_idx=0, seed=int(seed)),
        CTDEFleet(**common, fleet_idx=1, seed=int(seed) + 10_000),
    )
