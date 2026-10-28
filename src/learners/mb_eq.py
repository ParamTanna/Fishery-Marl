"""Model-based fleet that plays its side of the symmetric equilibrium at r_hat.

r_hat is the posterior mean, unless Thompson sampling draws from the belief at each replan.
A known_r fleet skips the belief and plans at the true r. The first N_warm seasons are
epsilon-greedy around a fixed safe request, so the belief has something to update on.

Design note (TA feedback, Nov 2026):
  This fleet plays the symmetric equilibrium of the estimated game. That is rational
  only when the other fleet also plays its equilibrium strategy. When facing a learner
  that behaves differently (IQL, overharvester, etc.) this assumption is wrong and the
  fleet may leave money on the table or over-fish relative to what a best-response
  would prescribe.

  The opponent-aware variant in mb_oa.py (MBOAFleet) addresses this by maintaining an
  online model of the other fleet's requests and replanning as a best response to that
  model — it does NOT assume the opponent plays an equilibrium strategy. E4 cross-play
  compares the two variants when facing non-equilibrium opponents.
"""

from __future__ import annotations

import numpy as np

from belief import Belief, r_grid
from learners.common import nearest_catch, stock_index


class MBEqFleet:
    def __init__(self, s_grid, a_grid, cfg, cache, fleet_index, seed=0, known_r=None,
                 thompson=False, true_r=None):
        self.s_grid = np.asarray(s_grid, dtype=np.float64)
        self.a_grid = np.asarray(a_grid, dtype=np.float64)
        self.cfg = cfg
        self.cache = cache
        self.fleet_index = int(fleet_index)
        self.known_r = None if known_r is None else float(known_r)
        self.thompson = bool(thompson)
        self.true_r = None if true_r is None else float(true_r)
        belief_cfg = cfg["belief"]
        grid = r_grid(belief_cfg["r_min"], belief_cfg["r_max"], belief_cfg["n_grid"])
        self.belief = Belief(grid, sigma=float(cfg["model"]["sigma"]), K=float(cfg["model"]["K"]))
        learn = cfg["learners"]
        self.n_warm = int(learn["n_warm"])
        self.n_replan = max(1, int(learn["n_replan"]))
        self.epsilon_warm = float(learn["epsilon_warm"])
        self.safe_request = float(learn["safe_default_catch"])
        self.safe_index = nearest_catch(self.safe_request, self.a_grid)
        self.rng = np.random.default_rng(int(seed) + 31 * (self.fleet_index + 1))
        self.season = 0
        self.plan = None
        self._last_request = self.safe_request
        self.collapse_warm = []

    def begin_episode(self):
        self.season = 0
        self.plan = None

    def finish_episode(self):
        return None

    def _r_hat(self):
        if self.known_r is not None:
            return self.known_r
        if self.thompson:
            probs = self.belief.probs()
            return float(self.rng.choice(self.belief.grid, p=probs))
        return float(self.belief.mean())

    def replan(self):
        r_hat = self._r_hat()
        self.cache.ensure_rule()
        p1, p2 = self.cache.interpolated_equilibrium(r_hat)
        self.plan = p1 if self.fleet_index == 0 else p2

    def act(self, stock, explore=True):
        if self.plan is None and self.season >= self.n_warm:
            self.replan()
        if explore and self.season < self.n_warm:
            if self.rng.random() < self.epsilon_warm:
                action = int(self.rng.integers(0, self.a_grid.shape[0]))
            else:
                action = self.safe_index
        elif self.plan is None:
            action = self.safe_index
        else:
            index = stock_index(stock, self.s_grid)
            action = int(np.argmax(self.plan[index]))
        self._last_request = float(self.a_grid[action])
        return self._last_request

    def observe(self, stock, own_request, hats, reward, next_stock, learn_reward=None):
        del reward, learn_reward
        escapement = float(stock) - float(np.sum(hats))
        if self.known_r is None:
            self.belief.update(escapement, float(next_stock))
        self.season += 1
        if self.season == self.n_warm or (self.season > self.n_warm and self.season % self.n_replan == 0):
            self.replan()

    def greedy_table(self):
        if self.plan is None:
            self.replan()
        return np.array(self.plan, dtype=np.float64, copy=True)


def mbeq_pair(s_grid, a_grid, cfg, cache, seed, known_r=None, thompson=False, true_r=None):
    common = dict(
        s_grid=s_grid, a_grid=a_grid, cfg=cfg, cache=cache, known_r=known_r,
        thompson=thompson, true_r=true_r,
    )
    return (
        MBEqFleet(fleet_index=0, seed=int(seed), **common),
        MBEqFleet(fleet_index=1, seed=int(seed), **common),
    )
