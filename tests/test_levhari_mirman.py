"""The discrete solver matches the Levhari-Mirman closed form away from the grid edges."""

from levhari_mirman import validate_levhari
from utils import load_config


def test_levhari_mirman_relative_error_below_one_percent():
    cfg = load_config()
    report = validate_levhari(cfg, verbose=True)
    print("Levhari-Mirman:", {k: report[k] for k in ("max_rel_error", "max_rel_plan", "max_rel_eq", "worst_stock", "n_stock", "n_fraction")})
    assert report["planner_converged"], report
    assert report["eq_n_fallback"] == 0
    assert report["max_rel_error"] < cfg["levhari_mirman"]["rel_error_tol"], report
