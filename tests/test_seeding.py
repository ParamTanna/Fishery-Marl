"""The environment is bitwise reproducible under a fixed seed."""

import numpy as np

from env import FisheryEnv


def _roll(seed):
    env = FisheryEnv(K=100.0, p=10.0, c=0.2, r=0.6, sigma=0.1, gamma=0.95, seed=seed, s0=100.0)
    env.reset()
    stocks = []
    rewards = []
    for t in range(12):
        stock, reward, _done, _info = env.step([12.0 + t, 18.0])
        stocks.append(stock)
        rewards.append(np.array(reward, dtype=np.float64, copy=True))
    return np.array(stocks), np.vstack(rewards)


def test_same_seed_is_bitwise_identical_and_other_seed_differs():
    a_s, a_r = _roll(7)
    b_s, b_r = _roll(7)
    c_s, _c_r = _roll(8)
    assert np.array_equal(a_s, b_s)
    assert np.array_equal(a_r, b_r)
    assert not np.array_equal(a_s, c_s)
