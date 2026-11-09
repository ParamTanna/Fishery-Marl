"""Team baselines.

Team IQL learns the sum of the two rewards. The model-based team fleet plays the
planner's action for its own index, from the cached planner at the true r when
known_r is set, otherwise at the posterior mean. There is no separate equilibrium.
"""

from __future__ import annotations

import numpy as np

from learners.iql import iql_pair
from learners.mb_eq import MBEqFleet


class TeamPlannerFleet(MBEqFleet):
    def replan(self):
        _plan, p1, p2 = self.cache.interpolated_planner(self._r_hat())
        self.plan = p1 if self.fleet_index == 0 else p2


def team_iql_pair(s_grid, a_grid, cfg, seed):
    return iql_pair(s_grid, a_grid, cfg, seed, team=True)


def team_planner_pair(s_grid, a_grid, cfg, cache, seed, known_r=None, true_r=None):
    common = dict(
        s_grid=s_grid, a_grid=a_grid, cfg=cfg, cache=cache, known_r=known_r, true_r=true_r,
    )
    return (
        TeamPlannerFleet(fleet_index=0, seed=int(seed), **common),
        TeamPlannerFleet(fleet_index=1, seed=int(seed), **common),
    )


def team_reward(rewards):
    total = float(np.sum(rewards))
    return total, total
