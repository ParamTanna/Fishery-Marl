"""Known-r regime sweep. Varies p/c, gamma and r. No learners.

Writes results/raw/e0_regime_sweep.csv and a heatmap drawn only from that file.
The earlier p/c-only sweep is kept as e0_regime_sweep_b0.csv.
"""

from __future__ import annotations

import copy
import os
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
from equilibrium import executed_catches, solve_equilibrium  # noqa: E402
from metrics import steady_stock  # noqa: E402
from utils import config_hash, git_commit, load_config, machine_summary  # noqa: E402

PC_VALUES = [12, 15, 20, 25]
GAMMAS = [0.95, 0.98, 0.99]
R_VALUES = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.5]
RULES = [
    "max_total", "min_total", "payoff_dominant", "risk_dominant",
    "random_seeded", "max_catch", "continuity",
]
# Mean-dynamics stock below this is the absorbing empty lake.
EXTINCTION_STOCK = 1.0


def _save_rows(rows, path):
    """Write via a temp file. Windows sometimes rejects a direct overwrite."""
    frame = pd.DataFrame(rows)
    tmp = path.with_suffix(".csv.tmp")
    last = None
    for attempt in range(8):
        try:
            frame.to_csv(tmp, index=False)
            os.replace(tmp, path)
            return
        except OSError as exc:
            last = exc
            time.sleep(0.4 * (attempt + 1))
    raise last


def _one_hot(index, n_actions):
    policy = np.zeros((index.shape[0], n_actions), dtype=np.float64)
    policy[np.arange(index.shape[0]), index] = 1.0
    return policy


def _steady_plan(plan, r, K, s0):
    n_a = plan.a_grid.size
    return steady_stock(
        plan.s_grid, plan.a_grid, _one_hot(plan.a1_idx, n_a), _one_hot(plan.a2_idx, n_a), r, K, s0
    )[0]


def sweep(cfg):
    raw = ROOT / "results" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    path = raw / "e0_regime_sweep.csv"
    previous = raw / "e0_regime_sweep_b0.csv"
    if path.exists() and not previous.exists():
        old = pd.read_csv(path)
        if "gamma" not in old.columns:
            path.replace(previous)
            print(f"archived the p/c-only sweep to {previous}", flush=True)
    K = float(cfg["model"]["K"])
    p = float(cfg["model"]["p"])
    s0 = float(cfg["model"]["s0"])
    rows = []
    if path.exists():
        done = pd.read_csv(path)
        rows = done.to_dict("records")
        print(f"resuming, {len(rows)} rows on disk", flush=True)
    seen = {
        (float(r["p_over_c"]), float(r["gamma"]), float(r["r"]), str(r["rule"]))
        for r in rows
    }

    print("Compute:", machine_summary(), flush=True)
    for pc in PC_VALUES:
        c = p / float(pc)
        for gamma in GAMMAS:
            cell = copy.deepcopy(cfg)
            cell["model"]["c"] = float(c)
            cell["model"]["p"] = p
            cell["model"]["gamma"] = float(gamma)
            # gamma = 0.99 needs about 1600 planner iterations to reach 1e-4.
            if gamma >= 0.98:
                cell["solver"]["max_H"] = 2500
            for r in R_VALUES:
                key_rules = [(float(pc), float(gamma), float(r), rule) for rule in RULES]
                if all(key in seen for key in key_rules):
                    print(f"skip p/c={pc} gamma={gamma} r={r}", flush=True)
                    continue
                margin = 2.0 * float(pc) - r * K / 4.0
                tragedy = bool(margin > 0.0)
                print(
                    f"\n=== p/c={pc} c={c:.4f} gamma={gamma} r={r} "
                    f"tragedy={tragedy} slack={margin:.3g} ===",
                    flush=True,
                )
                t0 = time.perf_counter()
                plan = solve_planner(cell, r, verbose=False)
                steady_plan = _steady_plan(plan, r, K, s0)
                j_plan = float(plan.V[-1])
                for rule in RULES:
                    key = (float(pc), float(gamma), float(r), rule)
                    if key in seen:
                        continue
                    eq = solve_equilibrium(cell, r, rule=rule, symmetric_only=True, verbose=False)
                    steady_eq, _t = steady_stock(eq.s_grid, eq.a_grid, eq.p1, eq.p2, r, K, s0)
                    j_eq = float(eq.V1[-1] + eq.V2[-1])
                    j_i = max(abs(float(eq.V1[-1])), abs(float(eq.V2[-1])), 1.0e-12)
                    catches = executed_catches(eq)
                    row = {
                        "p": p,
                        "c": float(c),
                        "p_over_c": float(pc),
                        "gamma": float(gamma),
                        "r": float(r),
                        "rule": rule,
                        "tragedy": tragedy,
                        "tragedy_slack": float(margin),
                        "plan_converged": bool(plan.converged),
                        "eq_converged": bool(eq.converged),
                        "stage_exploit_max": float(eq.stage_exploit_max),
                        "eps_abs": float(eq.stage_exploit_max),
                        "eps_rel": float(eq.eps_rel),
                        "eps_rel_threshold": float(eq.eps_rel_threshold),
                        "J_i": float(j_i),
                        "H_plan": int(plan.H),
                        "H_eq": int(eq.H),
                        "J_plan": j_plan,
                        "J_eq": j_eq,
                        "G": float(1.0 - j_eq / j_plan) if j_plan != 0.0 else np.nan,
                        "steady_stock_plan": float(steady_plan),
                        "steady_stock_eq": float(steady_eq),
                        "eq_extinct": bool(steady_eq < EXTINCTION_STOCK),
                        "plan_interior": bool(1.0 < steady_plan < K - 1.0),
                        "catch_at_K_plan": float(plan.total_executed[-1]),
                        "catch_at_K_eq": float(catches[-1].sum()),
                        "frac_states_multi_eq": float(np.mean(eq.n_eq > 1)),
                        "mean_n_eq": float(np.mean(eq.n_eq)),
                        "n_mixed_selected": int(eq.n_mixed_selected),
                        "seconds": time.perf_counter() - t0,
                        "config_hash": config_hash(cell),
                        "git_commit": git_commit(),
                    }
                    rows.append(row)
                    seen.add(key)
                    _save_rows(rows, path)
                    print(
                        f"  {rule}: conv={eq.converged} eps_rel={eq.eps_rel:.4e} "
                        f"J={j_eq:.1f} G={row['G']:.3f} "
                        f"steady plan/eq={steady_plan:.2f}/{steady_eq:.2f} "
                        f"extinct={row['eq_extinct']} exploit={eq.stage_exploit_max:.3e}",
                        flush=True,
                    )
    frame = pd.DataFrame(rows)
    _save_rows(rows, path)
    plot_heatmap(frame, ROOT / "results" / "figures")
    print(f"Wrote {path}", flush=True)
    return frame


def plot_heatmap(frame, fig_dir):
    """G and steady stock of the smallest-residual converged rule, one row per gamma."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    gammas = sorted(frame["gamma"].unique())
    pcs = sorted(frame["p_over_c"].unique())
    rs = sorted(frame["r"].unique())
    fig, axes = plt.subplots(len(gammas), 2, figsize=(7.2, 2.2 * len(gammas)), squeeze=False)
    for row, gamma in enumerate(gammas):
        sub = frame[np.isclose(frame["gamma"], gamma)]
        gap = np.full((len(rs), len(pcs)), np.nan)
        stock = np.full((len(rs), len(pcs)), np.nan)
        for i, r in enumerate(rs):
            for j, pc in enumerate(pcs):
                cell = sub[np.isclose(sub["r"], r) & np.isclose(sub["p_over_c"], pc)]
                good = cell[cell["eq_converged"] == True]  # noqa: E712
                if len(good) == 0:
                    continue
                best = good.sort_values("eps_rel").iloc[0]
                gap[i, j] = float(best["G"])
                stock[i, j] = float(best["steady_stock_eq"])
        for ax, data, title, cmap in (
            (axes[row, 0], stock, f"gamma={gamma:g} steady stock", "viridis"),
            (axes[row, 1], gap, f"gamma={gamma:g} gap G", "cividis"),
        ):
            im = ax.imshow(data, origin="lower", aspect="auto", cmap=cmap)
            ax.set_xticks(range(len(pcs)), [f"{v:g}" for v in pcs], fontsize=8)
            ax.set_yticks(range(len(rs)), [f"{v:g}" for v in rs], fontsize=8)
            ax.set_xlabel("p/c")
            ax.set_ylabel("r")
            ax.set_title(title, fontsize=9)
            for i in range(len(rs)):
                for j in range(len(pcs)):
                    value = data[i, j]
                    text = "nc" if not np.isfinite(value) else f"{value:.2f}"
                    ax.text(j, i, text, ha="center", va="center", fontsize=6)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Smallest-residual converged symmetric rule. nc = none converged", fontsize=11)
    fig.tight_layout()
    pdf = fig_dir / "e0_regime_sweep.pdf"
    png = fig_dir / "e0_regime_sweep.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {pdf}", flush=True)


if __name__ == "__main__":
    sweep(load_config())
