"""Proportional rationing: feasibility and monotonicity."""

import numpy as np

from env import ration
from kernels import ration_pair


def test_never_exceeds_stock_and_respects_slack():
    rng = np.random.default_rng(1)
    for _ in range(300):
        s = float(rng.uniform(0.0, 100.0))
        requests = rng.uniform(0.0, 150.0, size=2)
        hats = ration(requests, s)
        assert hats.shape == (2,)
        assert float(hats.sum()) <= s + 1.0e-8
        assert np.all(hats >= -1.0e-12)
        if requests.sum() <= s + 1.0e-12:
            assert np.allclose(hats, requests)
        if requests.sum() <= 1.0e-15 or s <= 0.0:
            assert np.allclose(hats, 0.0)


def test_own_catch_is_monotone_in_own_request():
    stock = 40.0
    other = 25.0
    previous = -1.0
    for own in np.linspace(0.0, 120.0, 80):
        catch = float(ration(np.array([own, other]), stock)[0])
        assert catch >= previous - 1.0e-9
        previous = catch


def test_kernel_matches_numpy_ration():
    rng = np.random.default_rng(2)
    for _ in range(50):
        s = float(rng.uniform(0.0, 80.0))
        a1 = float(rng.uniform(0.0, 100.0))
        a2 = float(rng.uniform(0.0, 100.0))
        h1, h2 = ration_pair(a1, a2, s)
        hats = ration(np.array([a1, a2]), s)
        assert np.allclose([h1, h2], hats)


def test_vectorized_batch():
    requests = np.array([[10.0, 10.0], [80.0, 80.0], [0.0, 0.0]])
    stock = np.array([50.0, 40.0, 10.0])
    hats = ration(requests, stock)
    assert np.allclose(hats[0], [10.0, 10.0])
    assert np.allclose(hats[1], [20.0, 20.0])
    assert np.allclose(hats[2], [0.0, 0.0])
    assert np.all(hats.sum(axis=1) <= stock + 1.0e-8)
