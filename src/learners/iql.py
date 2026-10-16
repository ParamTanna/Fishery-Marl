"""Independent tabular Q-learning.

The episode is a truncation, not a terminal state. Every transition, including the
last of the T seasons, bootstraps with gamma * max_a Q(s', a). The continuation
value at the stock reached after season T is the same max-Q, not zero.
"""

from __future__ import annotations

import numpy as np

from learners.common import nearest_catch, one_hot_greedy, stock_index


class IQLFleet:
    def __init__(
        self,
        s_grid,
        a_grid,
        gamma,
        alpha0=0.2,
        alpha_min=0.02,
        alpha_decay=0.9995,
        epsilon0=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.999,
        seed=0,
        team=False,
    ):
        self.s_grid = np.asarray(s_grid, dtype=np.float64)
        self.a_grid = np.asarray(a_grid, dtype=np.float64)
        self.gamma = float(gamma)
        self.alpha0 = float(alpha0)
        self.alpha_min = float(alpha_min)
        self.alpha_decay = float(alpha_decay)
        self.epsilon0 = float(epsilon0)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)
        self.team = bool(team)
        self.rng = np.random.default_rng(int(seed))
        self.Q = np.zeros((self.s_grid.shape[0], self.a_grid.shape[0]), dtype=np.float64)
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

    def act(self, stock, explore=True):
        index = stock_index(stock, self.s_grid)
        if explore and self.rng.random() < self.epsilon():
            action = int(self.rng.integers(0, self.a_grid.shape[0]))
        else:
            action = int(np.argmax(self.Q[index]))
        self._last_index = action
        return float(self.a_grid[action])

    def observe(self, stock, own_request, hats, reward, next_stock, learn_reward=None):
        """Bootstrap target is reward + gamma * max Q(next). `learn_reward` overrides the target reward.

        The team baseline passes the sum of the two rewards as `learn_reward`.
        """
        del own_request, hats
        state = stock_index(stock, self.s_grid)
        nxt = stock_index(next_stock, self.s_grid)
        action = self._last_index
        used = float(reward if learn_reward is None else learn_reward)
        # Truncation bootstrap: the season after T is not a terminal zero.
        target = used + self.gamma * float(np.max(self.Q[nxt]))
        step = self.alpha()
        self.Q[state, action] += step * (target - self.Q[state, action])
        self.updates += 1

    def greedy_table(self):
        choice = np.argmax(self.Q, axis=1)
        return one_hot_greedy(choice, self.s_grid.shape[0], self.a_grid.shape[0])

    def greedy_request(self, stock):
        return float(self.a_grid[int(np.argmax(self.Q[stock_index(stock, self.s_grid)]))])

    def set_q(self, state, action, value):
        self.Q[state, action] = float(value)


def iql_pair(s_grid, a_grid, cfg, seed, team=False):
    """Two fleets. Seeds differ so their exploration draws are not identical."""
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
        epsilon_decay=float(block["epsilon_decay"]),
        team=team,
    )
    return IQLFleet(seed=int(seed), **common), IQLFleet(seed=int(seed) + 17, **common)
