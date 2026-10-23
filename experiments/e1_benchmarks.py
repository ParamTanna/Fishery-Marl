"""E1: planner versus equilibrium across r and across selection rules.

Writes results/raw only, then draws the preview figure from those files.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dp_planner import solve_planner  # noqa: E402
from env import mean_path  # noqa: E402
from equilibrium import ENUMERATION_LIMIT, executed_catches, solve_equilibrium  # noqa: E402
from metrics import rollout_returns, steady_stock  # noqa: E402
from utils import config_hash, git_commit, load_config, machine_summary, tragedy_margin  # noqa: E402

RULE_COLOR = {
    "planner": "#009E73",
    "max_total": "#0072B2",
    "min_total": "#D55E00",
    "payoff_dominant": "#CC79A7",
    "risk_dominant": "#332288",
    "random_seeded": "#E69F00",
    "max_catch": "#882255",
    "continuity": "#117733",
}


def _one_hot(index, n_actions):
    policy = np.zeros((index.shape[0], n_actions), dtype=np.float64)
    policy[np.arange(index.shape[0]), index] = 1.0
    return policy


def _solve_pair(cfg, r, rule, symmetric_only, selection_seed=None):
    t0 = time.perf_counter()
    plan = solve_planner(cfg, r, verbose=True)
    eq = solve_equilibrium(
        cfg, r, rule=rule, symmetric_only=symmetric_only, selection_seed=selection_seed, verbose=True
    )
    elapsed = time.perf_counter() - t0
    return plan, eq, elapsed


def _policy_frame(plan, eq, r, rule, symmetric_only):
    catches = executed_catches(eq)
    rows = []
    for si, s in enumerate(plan.s_grid):
        rows.append(
            {
                "r": r,
                "rule": rule,
                "symmetric_only": bool(symmetric_only),
                "s": float(s),
                "catch_plan_1": float(plan.executed[si, 0]),
                "catch_plan_2": float(plan.executed[si, 1]),
                "catch_plan": float(plan.total_executed[si]),
                "catch_eq_1": float(catches[si, 0]),
                "catch_eq_2": float(catches[si, 1]),
                "catch_eq": float(catches[si].sum()),
                "V_plan": float(plan.V[si]),
                "V1": float(eq.V1[si]),
                "V2": float(eq.V2[si]),
                "n_eq": int(eq.n_eq[si]),
            }
        )
    return pd.DataFrame(rows)


def _summary_row(cfg, plan, eq, r, rule, symmetric_only, elapsed, do_mc, xi_len_seed):
    catches = executed_catches(eq)
    catch_gap = catches.sum(axis=1) - plan.total_executed
    below = int(np.sum(catches.sum(axis=1) + 1.0e-6 < plan.total_executed))
    steady_p, _ = steady_stock(plan.s_grid, plan.a_grid, _one_hot(plan.a1_idx, plan.a_grid.size), _one_hot(plan.a2_idx, plan.a_grid.size), r, cfg["model"]["K"], cfg["model"]["s0"])
    steady_e, _ = steady_stock(eq.s_grid, eq.a_grid, eq.p1, eq.p2, r, cfg["model"]["K"], cfg["model"]["s0"])
    nontrivial = eq.n_eq.copy()
    # Trivial multiplicity at a flat stock is real but separate from interior multiplicity.
    multi = eq.n_eq > 1
    row = {
        "r": r,
        "rule": rule,
        "symmetric_only": bool(symmetric_only),
        "H_plan": plan.H,
        "H_eq": eq.H,
        "plan_converged": plan.converged,
        "eq_converged": eq.converged,
        "plan_delta": plan.delta,
        "eq_delta": eq.delta,
        "J_plan": float(plan.V[-1]),
        "J1": float(eq.V1[-1]),
        "J2": float(eq.V2[-1]),
        "J_eq": float(eq.V1[-1] + eq.V2[-1]),
        "G_dp": float(1.0 - (eq.V1[-1] + eq.V2[-1]) / plan.V[-1]) if plan.V[-1] != 0 else np.nan,
        "catch_at_K_plan": float(plan.total_executed[-1]),
        "catch_at_K_eq": float(catches[-1].sum()),
        "steady_stock_plan": steady_p,
        "steady_stock_eq": steady_e,
        "mean_n_eq": float(np.mean(eq.n_eq)),
        "frac_states_multi_eq": float(np.mean(multi)),
        "n_states_eq_catch_below_plan": below,
        "min_catch_gap_eq_minus_plan": float(catch_gap.min()),
        "stage_exploit_max": eq.stage_exploit_max,
        "eps_abs": eq.stage_exploit_max,
        "eps_rel": eq.eps_rel,
        "eps_rel_threshold": eq.eps_rel_threshold,
        "n_fallback": eq.n_fallback,
        "n_mixed_selected": eq.n_mixed_selected,
        "n_truncated_mixed": eq.n_truncated_mixed,
        "n_trivial_states": eq.n_trivial,
        "seconds": elapsed,
        "config_hash": config_hash(cfg),
        "git_commit": git_commit(),
    }
    if do_mc:
        length = xi_len_seed[0]
        seed = xi_len_seed[1]
        xi = np.random.default_rng(seed).normal(size=(length, int(cfg["evaluation"]["n_rollouts"])))
        p1 = _one_hot(plan.a1_idx, plan.a_grid.size)
        p2 = _one_hot(plan.a2_idx, plan.a_grid.size)
        mc_p = rollout_returns(plan.s_grid, plan.a_grid, p1, p2, cfg, r, length=length, seed=seed, xi=xi)
        mc_e = rollout_returns(eq.s_grid, eq.a_grid, eq.p1, eq.p2, cfg, r, length=length, seed=seed, xi=xi)
        row.update(
            {
                "J_plan_mc": mc_p["J_total"],
                "J_plan_mc_se": mc_p["J_total_se"],
                "J_eq_mc": mc_e["J_total"],
                "J_eq_mc_se": mc_e["J_total_se"],
                "collapse_50_plan": mc_p["collapse_50"],
                "collapse_long_plan": mc_p["collapse_long"],
                "collapse_50_eq": mc_e["collapse_50"],
                "collapse_long_eq": mc_e["collapse_long"],
                "long_run_plan_mc": mc_p["long_run_stock"],
                "long_run_eq_mc": mc_e["long_run_stock"],
                "mc_length": mc_p["length"],
                "mc_rollouts": mc_p["n_rollouts"],
            }
        )
    return row


def _paths(plan, eq, r, K, s0, horizon):
    p1 = _one_hot(plan.a1_idx, plan.a_grid.size)
    p2 = _one_hot(plan.a2_idx, plan.a_grid.size)
    sp, _cp = mean_path(plan.s_grid, plan.a_grid, p1, p2, r, K, s0, horizon)
    se, _ce = mean_path(eq.s_grid, eq.a_grid, eq.p1, eq.p2, r, K, s0, horizon)
    return sp, se


def run(cfg, rules, r_values, unrestricted_rules, do_refinement, do_mc, quick=False):
    raw = ROOT / "results" / "raw"
    fig_dir = ROOT / "results" / "figures"
    raw.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    print("Compute:", machine_summary())
    print("Enumeration:", ENUMERATION_LIMIT)
    rows_t = tragedy_margin(cfg, r_values)
    sanity = [float(v) for v in cfg["model"].get("r_sanity", [])]
    r_values = list(r_values) + [r for r in sanity if r not in r_values]
    print("Tragedy check |N|*p/c > rK/4 (sanity r is solved but not required to be a tragedy):")
    for row in rows_t:
        print(
            f"  r={row['r']}: |N|*p/c={row['n_times_p_over_c']:.4g}  "
            f"rK/4={row['rK_over_4']:.4g}  slack={row['slack']:.4g}"
        )

    from utils import horizon_for_tail

    length = horizon_for_tail(float(cfg["model"]["gamma"]), float(cfg["evaluation"]["gamma_tail"]))
    seed = int(cfg["evaluation"]["mc_seed"])
    frames = []
    summaries = []
    path_rows = []
    K = float(cfg["model"]["K"])
    s0 = float(cfg["model"]["s0"])
    horizon = int(cfg["model"]["T"])

    # Planner depends only on r, but each rule re-solves it. Cache by r.
    planner_cache = {}
    for r in r_values:
        jobs = [(rule, True) for rule in rules] + [(rule, False) for rule in unrestricted_rules]
        for rule, symmetric_only in jobs:
            print(f"\n=== r={r} rule={rule} symmetric_only={symmetric_only} ===", flush=True)
            t0 = time.perf_counter()
            if r not in planner_cache:
                planner_cache[r] = solve_planner(cfg, r, verbose=True)
            plan = planner_cache[r]
            eq = solve_equilibrium(
                cfg, r, rule=rule, symmetric_only=symmetric_only, verbose=True
            )
            elapsed = time.perf_counter() - t0
            print(
                f"  J_plan={plan.V[-1]:.4f} J_eq={eq.V1[-1] + eq.V2[-1]:.4f} "
                f"H={eq.H} converged={eq.converged} exploit={eq.stage_exploit_max:.3e} "
                f"eps_rel={eq.eps_rel:.4e} "
                f"fallback={eq.n_fallback} seconds={elapsed:.1f}",
                flush=True,
            )
            frames.append(_policy_frame(plan, eq, r, rule, symmetric_only))
            summaries.append(
                _summary_row(cfg, plan, eq, r, rule, symmetric_only, elapsed, do_mc, (length, seed))
            )
            sp, se = _paths(plan, eq, r, K, s0, horizon)
            for t in range(horizon):
                path_rows.append(
                    {
                        "r": r,
                        "rule": rule,
                        "symmetric_only": bool(symmetric_only),
                        "t": t,
                        "stock_plan": float(sp[t]),
                        "stock_eq": float(se[t]),
                    }
                )

    policy = pd.concat(frames, ignore_index=True)
    summary = pd.DataFrame(summaries)
    paths = pd.DataFrame(path_rows)
    ref_path = None
    if do_refinement:
        ref_path = _refinement(cfg, raw, quick=quick)
    if not quick:
        summary = _attach_band(summary, raw / "e1_refinement.csv")
    tag = config_hash(cfg)
    if quick:
        policy_path = raw / "e1_quick_policies.csv"
        summary_path = raw / "e1_quick_summary.csv"
        path_path = raw / "e1_quick_paths.csv"
        policy.to_csv(policy_path, index=False)
        summary.to_csv(summary_path, index=False)
        paths.to_csv(path_path, index=False)
    else:
        policy_path = raw / f"e1_policies_{tag}.csv"
        summary_path = raw / f"e1_summary_{tag}.csv"
        path_path = raw / f"e1_paths_{tag}.csv"
        policy.to_csv(policy_path, index=False)
        summary.to_csv(summary_path, index=False)
        paths.to_csv(path_path, index=False)
        # Stable names the figure script and the report kit read.
        policy.to_csv(raw / "e1_policies.csv", index=False)
        summary.to_csv(raw / "e1_summary.csv", index=False)
        paths.to_csv(raw / "e1_paths.csv", index=False)

    _print_degeneracy(summary)
    if not quick:
        plot_e1(raw / "e1_policies.csv", raw / "e1_paths.csv", fig_dir / "e1_preview.pdf", fig_dir / "e1_preview.png")
    print(f"\nWrote {summary_path}")
    print(f"Wrote {policy_path}")
    if ref_path:
        print(f"Wrote {ref_path}")
    return summary


def _attach_band(summary, refinement_csv):
    """Copy the r=0.5 catch-grid band onto every benchmark row. Numbers come from the CSV."""
    from band import catch_grid_band

    if not refinement_csv.exists():
        return summary
    band = catch_grid_band(refinement_csv)
    for key, value in band.items():
        summary[key] = value
    return summary


def _print_degeneracy(summary):
    print("\nPlanner vs equilibrium total catch at s=K, and steady stocks:")
    for _, row in summary.iterrows():
        print(
            f"  r={row['r']} {row['rule']} sym={row['symmetric_only']}: "
            f"catchK plan/eq={row['catch_at_K_plan']:.3f}/{row['catch_at_K_eq']:.3f}  "
            f"steady plan/eq={row['steady_stock_plan']:.3f}/{row['steady_stock_eq']:.3f}  "
            f"G_dp={row['G_dp']:.4f}  states with eq catch < plan={row['n_states_eq_catch_below_plan']}"
        )
    gap = (summary["catch_at_K_plan"] - summary["catch_at_K_eq"]).abs()
    if float(gap.max()) < 1.0e-3 and float((summary["J_plan"] - summary["J_eq"]).abs().max()) < 1.0e-2:
        print("DEGENERACY: planner and equilibrium are effectively the same on this grid.")
    if float(summary["steady_stock_eq"].max()) < 1.0 and float(summary["steady_stock_plan"].max()) < 1.0:
        print("DEGENERACY: mean dynamics collapse for every solved benchmark.")


def _refinement(cfg, raw, quick=False):
    spec = cfg["refinement"]
    rows = []
    curves = []
    K = float(cfg["model"]["K"])
    r = float(spec["r"])
    rule = spec["rule"]
    for n_s in spec["n_stock_list"]:
        for n_a in spec["n_catch_list"]:
            print(f"\n=== refinement n_stock={n_s} n_catch={n_a} r={r} ===", flush=True)
            t0 = time.perf_counter()
            plan = solve_planner(cfg, r, verbose=True, n_stock=int(n_s), n_catch=int(n_a))
            eq = solve_equilibrium(
                cfg, r, rule=rule, symmetric_only=True, verbose=True, n_stock=int(n_s), n_catch=int(n_a)
            )
            catches = executed_catches(eq)
            elapsed = time.perf_counter() - t0
            common = np.linspace(0.0, K, 101)
            plan_on = np.interp(common, plan.s_grid, plan.total_executed)
            eq_on = np.interp(common, eq.s_grid, catches.sum(axis=1))
            curves.append((int(n_s), int(n_a), plan_on, eq_on))
            rows.append(
                {
                    "r": r,
                    "rule": rule,
                    "n_stock": int(n_s),
                    "n_catch": int(n_a),
                    "J_plan": float(plan.V[-1]),
                    "J_eq": float(eq.V1[-1] + eq.V2[-1]),
                    "H_plan": plan.H,
                    "H_eq": eq.H,
                    "plan_converged": plan.converged,
                    "eq_converged": eq.converged,
                    "seconds": elapsed,
                    "config_hash": config_hash(cfg),
                    "git_commit": git_commit(),
                }
            )
            print(f"  J_plan={plan.V[-1]:.4f} J_eq={eq.V1[-1]+eq.V2[-1]:.4f} seconds={elapsed:.1f}", flush=True)
    # Distance of each catch curve to the finest grid in the list.
    ref = curves[-1]
    for (n_s, n_a, plan_on, eq_on), row in zip(curves, rows):
        row["plan_catch_sup_vs_finest"] = float(np.max(np.abs(plan_on - ref[2])))
        row["eq_catch_sup_vs_finest"] = float(np.max(np.abs(eq_on - ref[3])))
    frame = pd.DataFrame(rows)
    path = raw / ("e1_quick_refinement.csv" if quick else "e1_refinement.csv")
    frame.to_csv(path, index=False)
    return path


def plot_e1(policy_csv, path_csv, pdf_path, png_path):
    policy = pd.read_csv(policy_csv)
    paths = pd.read_csv(path_csv)
    # Preview uses the symmetric benchmarks only.
    policy = policy[policy["symmetric_only"] == True]  # noqa: E712
    paths = paths[paths["symmetric_only"] == True]  # noqa: E712
    r_values = list(policy["r"].drop_duplicates())
    n = len(r_values)
    fig, axes = plt.subplots(2, n, figsize=(7.2, 5.4), sharex="row", squeeze=False)
    for col, r in enumerate(r_values):
        sub = policy[policy["r"] == r]
        ax = axes[0, col]
        one = sub[sub["rule"] == sub["rule"].iloc[0]]
        ax.plot(one["s"], one["catch_plan"], color=RULE_COLOR["planner"], lw=1.8, label="planner")
        for rule, grp in sub.groupby("rule", sort=False):
            ax.plot(grp["s"], grp["catch_eq"], color=RULE_COLOR.get(rule, "#333333"), lw=1.2, label=f"eq {rule}")
        ax.set_title(f"r = {r:g}", fontsize=9)
        ax.set_ylabel("total catch" if col == 0 else "")
        ax.tick_params(labelsize=8)
        ax = axes[1, col]
        psub = paths[paths["r"] == r]
        one_p = psub[psub["rule"] == psub["rule"].iloc[0]]
        ax.plot(one_p["t"], one_p["stock_plan"], color=RULE_COLOR["planner"], lw=1.8, label="planner")
        for rule, grp in psub.groupby("rule", sort=False):
            ax.plot(grp["t"], grp["stock_eq"], color=RULE_COLOR.get(rule, "#333333"), lw=1.2, label=f"eq {rule}")
        ax.set_xlabel("season", fontsize=8)
        ax.set_ylabel("stock" if col == 0 else "")
        ax.set_ylim(0.0, 100.0)
        ax.tick_params(labelsize=8)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.86))
    fig.legend(
        handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=7,
        bbox_to_anchor=(0.5, 0.98),
    )
    fig.suptitle("Planner and equilibrium: catch policies and mean stock paths", fontsize=11, y=1.02)
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {pdf_path}")
    print(f"Wrote {png_path}")


def main():
    parser = argparse.ArgumentParser(description="E1 benchmarks")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--rules", nargs="*")
    parser.add_argument("--r", nargs="*", type=float)
    parser.add_argument("--skip-refinement", action="store_true")
    parser.add_argument("--skip-mc", action="store_true")
    parser.add_argument("--skip-unrestricted", action="store_true")
    args = parser.parse_args()
    quick = args.quick and not args.full
    cfg = load_config(quick=quick)
    if args.rules:
        rules = args.rules
    elif quick:
        rules = ["max_total", "min_total"]
    else:
        rules = list(cfg["selection"]["rules"])
    if args.r:
        r_values = args.r
        cfg["model"]["r_sanity"] = []
    else:
        r_values = [float(v) for v in cfg["model"]["r_values"]]
    if args.skip_unrestricted or quick:
        unrestricted = []
    else:
        unrestricted = ["max_total", "min_total"]
    do_refinement = not args.skip_refinement
    do_mc = not args.skip_mc
    run(cfg, rules, r_values, unrestricted, do_refinement, do_mc, quick=quick)


if __name__ == "__main__":
    main()
