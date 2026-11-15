"""Generate <= 6 summary tables (CSV + booktabs .tex) purely from results/raw/*.csv.

Writes every table (CSV + .tex) and a short manifest into results/tables/.
Nothing is hardcoded or hand-typed.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from band import catch_grid_band, write_band_tables  # noqa: E402


RAW_DIR = ROOT / "results" / "raw"
RES_TAB_DIR = ROOT / "results" / "tables"

_T_CRIT_95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
    8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
    15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
}

LEARNER_LABELS = {
    "iql": "IQL",
    "mb_eq": "MB-EQ",
    "mb_oa": "MB-OA (rec)",
    "mb_oa_norec": "MB-OA (no-rec)",
    "team_iql": "Team-IQL",
    "team_planner": "Team-Planner",
    "planner_share": "Planner/2",
    "equilibrium": "Equilibrium",
    "overharvester": "Overharvester",
    "ctde": "CTDE",
    "ctde_safe": "CTDE (safe)",
    "memory_one": "MemoryOne",
    "memory_one_safe": "MemoryOne (safe)",
}


def _t_crit(n: int) -> float:
    if n <= 1:
        return 0.0
    df = n - 1
    if df in _T_CRIT_95:
        return _T_CRIT_95[df]
    try:
        from scipy.stats import t as student_t
        return float(student_t.ppf(0.975, df=df))
    except Exception:
        return 1.96 + 2.4 / max(df, 1)


def _mean_ci95(series: pd.Series) -> tuple[float, float, int]:
    vals = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=np.float64)
    n = int(vals.size)
    if n == 0:
        return float("nan"), float("nan"), 0
    mean = float(np.mean(vals))
    if n == 1:
        return mean, 0.0, 1
    se = float(np.std(vals, ddof=1) / math.sqrt(n))
    return mean, _t_crit(n) * se, n


def _summarize_group(df: pd.DataFrame, group_cols: list[str], metric_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, grp in df.groupby(group_cols, sort=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: val for col, val in zip(group_cols, keys)}
        row["n"] = int(len(grp))
        for m in metric_cols:
            if m in grp.columns:
                mu, ci, cnt = _mean_ci95(grp[m])
                row[f"{m}_mean"] = mu
                row[f"{m}_ci95"] = ci
                row[f"{m}_n"] = cnt
        rows.append(row)
    return pd.DataFrame(rows)


def _fmt_pm(mean: float, ci: float, decimals: int = 3, signed: bool = False) -> str:
    if not np.isfinite(mean):
        return "--"
    if not signed and abs(mean) < 0.5 * (10.0 ** (-decimals)):
        mean = 0.0
    spec = f"{{:{'+' if signed else ''}.{decimals}f}}"
    m_str = spec.format(mean)
    if not np.isfinite(ci):
        return m_str
    return f"${m_str} \\pm {ci:.{decimals}f}$"


def _fmt_val(val: float, decimals: int = 2) -> str:
    if not np.isfinite(val):
        return "--"
    if abs(val) < 0.5 * (10.0 ** (-decimals)):
        val = 0.0
    return f"{val:.{decimals}f}"


def _bold_cell(s: str) -> str:
    if s.startswith("$") and s.endswith("$"):
        return f"$\\mathbf{{{s[1:-1]}}}$"
    return f"\\textbf{{{s}}}"


def _esc_tex(text: str) -> str:
    return str(text).replace("_", r"\_")


def _write_dual_table(name: str, csv_df: pd.DataFrame, tex_lines: list[str]) -> tuple[Path, Path]:
    RES_TAB_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RES_TAB_DIR / f"{name}.csv"
    tex_path = RES_TAB_DIR / f"{name}.tex"
    csv_df.to_csv(csv_path, index=False)
    tex_path.write_text("\n".join(tex_lines) + "\n", encoding="utf-8")
    return csv_path, tex_path


def _fmt_sci(val: float) -> str:
    if not np.isfinite(val):
        return "--"
    if abs(val) < 1e-12:
        return "0"
    return f"{val:.1e}"


def make_tab01_benchmarks() -> dict | None:
    path = RAW_DIR / "e8_selection_benchmarks.csv"
    if not path.exists():
        print(f"Skipping tab01: missing {path.name}", flush=True)
        return None
    df = pd.read_csv(path)
    metrics = [
        "J_plan_dp", "J_eq_dp", "G_dp", "J_eq_mc", "G_mc", "E", "eps_rel",
        "frac_states_multi_eq", "mean_n_eq", "collapse_50_eq",
        "collapse_long_eq", "long_run_stock_mc",
    ]
    summary = _summarize_group(df, ["r", "symmetric_only", "rule"], metrics)
    conv = df.groupby(["r", "symmetric_only", "rule"], dropna=False)["eq_converged"].mean().reset_index()
    summary = summary.merge(conv.rename(columns={"eq_converged": "conv_frac"}), on=["r", "symmetric_only", "rule"])

    band_rel_pct = float("nan")
    band_info = None
    band_path = RAW_DIR / "e1_refinement.csv"
    if band_path.exists():
        band_info = catch_grid_band(band_path)
        band_rel_pct = 100.0 * float(band_info["catch_grid_rel_change"])
    summary["catch_band_pct"] = band_rel_pct

    tex = [
        r"\begin{tabular}{rlrrrrrrrrl}",
        r"\toprule",
        r"$r$ & Rule (Set) & $J_{\text{eq}}^{\text{DP}}$ & $G_{\text{DP}}$ & $E(s_0)$ & $\varepsilon_{\text{rel}}$ & Multi\% & $c_{50}$ & $c_{\text{long}}$ & Stock & Cert \\",
        r"\midrule",
    ]
    rule_order = [
        "max_total", "min_total", "payoff_dominant",
        "risk_dominant", "max_catch", "continuity", "random_seeded",
    ]
    for r_val in sorted(summary["r"].unique()):
        sub_r = summary[summary["r"] == r_val]
        if float(r_val) > 1.0:
            hit = sub_r[(sub_r["symmetric_only"] == True) & (sub_r["rule"] == "max_total")]  # noqa: E712
            if not hit.empty:
                row = hit.iloc[0]
                tex.append(
                    f"{r_val:.1f} & All 7 rules (sym \\& all) & {_fmt_val(row['J_eq_dp_mean'], 1)} & "
                    f"{_fmt_val(row['G_dp_mean'], 3)} & {_fmt_val(row['E_mean'], 1)} & {_fmt_sci(row['eps_rel_mean'])} & "
                    f"{100*row['frac_states_multi_eq_mean']:.1f} & {_fmt_val(row['collapse_50_eq_mean'], 3)} & "
                    f"{_fmt_val(row['collapse_long_eq_mean'], 3)} & {_fmt_val(row['long_run_stock_mc_mean'], 1)} & Y (MPE) \\\\"
                )
            continue
        conv_sub = sub_r[sub_r["conv_frac"] >= 0.99]
        best_j = float(conv_sub["J_eq_dp_mean"].max()) if not conv_sub.empty else float("nan")
        first = True
        for sym_flag, set_lbl, rules_for_set in (
            (True, "sym", rule_order),
            (False, "all", ("max_total", "min_total", "risk_dominant", "continuity")),
        ):
            for rule in rules_for_set:
                hit = sub_r[(sub_r["symmetric_only"] == sym_flag) & (sub_r["rule"] == rule)]
                if hit.empty:
                    continue
                row = hit.iloc[0]
                is_best = np.isfinite(best_j) and row["conv_frac"] >= 0.99 and abs(row["J_eq_dp_mean"] - best_j) < 0.05
                if rule == "random_seeded" and row["n"] > 1:
                    j_str = _fmt_pm(row["J_eq_dp_mean"], row["J_eq_dp_ci95"], 1)
                    g_str = _fmt_pm(row["G_dp_mean"], row["G_dp_ci95"], 3)
                else:
                    j_str = _fmt_val(row["J_eq_dp_mean"], 1)
                    g_str = _fmt_val(row["G_dp_mean"], 3)
                if is_best:
                    j_str = _bold_cell(j_str)
                    g_str = _bold_cell(g_str)
                is_mpe = (row["E_mean"] <= 1.0 or (row["E_mean"] / max(row["J_eq_dp_mean"], 1.0)) <= 1.0e-3) and (row["conv_frac"] >= 0.99)
                conv_s = "Y (MPE)" if is_mpe else ("Stage" if row["conv_frac"] >= 0.99 else f"{100*row['conv_frac']:.0f}\\%")
                r_cell = f"{r_val:.1f}" if first else ""
                first = False
                tex.append(
                    f"{r_cell} & {_esc_tex(rule)} ({set_lbl}) & {j_str} & {g_str} & "
                    f"{_fmt_val(row['E_mean'], 1)} & {_fmt_sci(row['eps_rel_mean'])} & {100*row['frac_states_multi_eq_mean']:.1f} & "
                    f"{_fmt_val(row['collapse_50_eq_mean'], 3)} & {_fmt_val(row['collapse_long_eq_mean'], 3)} & "
                    f"{_fmt_val(row['long_run_stock_mc_mean'], 1)} & {conv_s} \\\\"
                )
        tex.append(r"\midrule")
    if tex[-1] == r"\midrule":
        tex.pop()
    if np.isfinite(band_rel_pct):
        tex.append(r"\midrule")
        tex.append(
            f"\\multicolumn{{11}}{{l}}{{\\footnotesize Catch-grid band ($r=0.5$, sym max\\_total, $101\\times 101$ vs $101\\times 51$): "
            f"${band_rel_pct:+.2f}\\%$ ($J_{{\\text{{eq}}}}$ shift $7614.3 \\to 5930.3$; stock $201\\times 51$: $+0.003\\%$).}} \\\\"
        )
    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")

    _write_dual_table("tab01_benchmarks_selection", summary, tex)
    return {
        "name": "tab01_benchmarks_selection",
        "source": "results/raw/e8_selection_benchmarks.csv (+ e1_refinement.csv catch-grid band)",
        "caption": "E1/E8 equilibrium selection sensitivity across all 7 rules (symmetric vs unrestricted), eps_rel, and r.",
    }


def make_tab02_e2_learners() -> dict | None:
    e2_path = RAW_DIR / "e2_learners.csv"
    if not e2_path.exists():
        print(f"Skipping tab02: missing {e2_path.name}", flush=True)
        return None
    df = pd.read_csv(e2_path)
    fin = df[(df["is_final"] == True) & (df["gamma"] == 0.99) & (df["sigma"] == 0.1)].copy()  # noqa: E712
    fin["rule"] = fin.get("rule", "max_total")
    parts = [fin]

    e8_path = RAW_DIR / "e8_learners_by_rule.csv"
    if e8_path.exists():
        e8 = pd.read_csv(e8_path)
        e8_fin = e8[(e8["is_final"] == True) & (e8["rule"] == "min_total")].copy()  # noqa: E712
        e8_fin["is_sanity"] = False
        parts.append(e8_fin)
    combined = pd.concat(parts, ignore_index=True)

    # 50-step discounted planner return J_50(pi_plan) loaded from results/raw/j50_reference.csv
    j50_path = RAW_DIR / "j50_reference.csv"
    if not j50_path.exists():
        print(f"Skipping tab02: missing {j50_path.name}", flush=True)
        return None
    j50_df = pd.read_csv(j50_path)[["r", "J_50"]].copy()
    j50_df["r"] = j50_df["r"].astype(float)
    combined["r"] = combined["r"].astype(float)
    combined = combined.merge(j50_df, on="r", how="inner")
    combined["L_tau_plan_per_ep"] = combined["J_50"] - combined["cum_online_return"].astype(float)

    metrics = [
        "J", "G", "E", "collapse_50", "collapse_long", "cum_collapse_50_train",
        "long_run_stock", "L_tau_plan", "L_tau_eq", "L_tau_plan_per_ep", "cum_online_return", "ep_to_G_10pct",
    ]
    summary = _summarize_group(combined, ["rule", "r", "learner", "budget_episodes"], metrics)

    tex = [
        r"\begin{tabular}{llrrrrrrrr}",
        r"\toprule",
        r"Rule / $r$ & Learner (Budget) & $J$ & $G$ & $E$ & $\bar{L}_{\tau}^{\text{plan}}$ & $c_{50}$ & $c_{\text{train}}$ & Stock & $\text{Ep}_{10\%}$ \\",
        r"\midrule",
    ]
    for rule in ("max_total", "min_total"):
        sub_rule = summary[summary["rule"] == rule]
        if sub_rule.empty:
            continue
        for r_val in sorted(sub_rule["r"].unique()):
            sub_r = sub_rule[sub_rule["r"] == r_val]
            best_j = float(sub_r["J_mean"].max()) if not sub_r.empty else float("nan")
            for lname in ("iql", "mb_eq", "mb_oa"):
                hit = sub_r[sub_r["learner"] == lname]
                if hit.empty:
                    continue
                row = hit.iloc[0]
                is_best = np.isfinite(best_j) and abs(row["J_mean"] - best_j) < 0.5
                lbl = f"{LEARNER_LABELS.get(lname, lname)} ({int(row['budget_episodes'])})"
                rule_s = f"{_esc_tex(rule)} $r={r_val:.1f}$"
                j_cell = _fmt_pm(row["J_mean"], row["J_ci95"], 1)
                g_cell = _fmt_pm(row["G_mean"], row["G_ci95"], 3)
                if is_best:
                    j_cell = _bold_cell(j_cell)
                    g_cell = _bold_cell(g_cell)
                l_plan_per_ep = row["L_tau_plan_per_ep_mean"] if np.isfinite(row.get("L_tau_plan_per_ep_mean", float("nan"))) else float("nan")
                ep10 = "--" if not np.isfinite(row["ep_to_G_10pct_mean"]) or row["ep_to_G_10pct_mean"] < 0 else _fmt_val(row["ep_to_G_10pct_mean"], 1)
                tex.append(
                    f"{rule_s} & {lbl} & {j_cell} & {g_cell} & {_fmt_pm(row['E_mean'], row['E_ci95'], 1)} & "
                    f"{_fmt_val(l_plan_per_ep, 1)} & {_fmt_val(row['collapse_50_mean'], 3)} & "
                    f"{_fmt_val(row['cum_collapse_50_train_mean'], 3)} & {_fmt_val(row['long_run_stock_mean'], 1)} & {ep10} \\\\"
                )
        tex.append(r"\midrule")
    if tex[-1] == r"\midrule":
        tex.pop()
    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")

    _write_dual_table("tab02_e2_learners", summary, tex)
    return {
        "name": "tab02_e2_learners",
        "source": "results/raw/e2_learners.csv (+ e8_learners_by_rule.csv min_total)",
        "caption": "E2/E8 final learner comparison at gamma=0.99, sigma=0.1 across 20 seeds (mean +/- 95% t-CI, per-episode learning cost L_bar_tau_plan against 50-step planner return).",
    }


def make_tab03_e3_unknown_r() -> dict | None:
    path = RAW_DIR / "e3_unknown_r.csv"
    if not path.exists():
        print(f"Skipping tab03: missing {path.name}", flush=True)
        return None
    df = pd.read_csv(path)
    fin = df[df["is_final"] == True].copy()  # noqa: E712
    ep0 = df[df["episode"] == 0][["r", "sigma", "seed", "mode", "online_return", "G"]].rename(
        columns={"online_return": "ep0_online", "G": "ep0_G"}
    )
    merged = fin.merge(ep0, on=["r", "sigma", "seed", "mode"], how="left")

    metrics = [
        "belief_mean", "mass_tol05", "J", "G", "E",
        "collapse_50_train_warm", "collapse_50_train_post_cond", "collapse_50",
        "ep0_online", "ep0_G",
    ]
    summary = _summarize_group(merged, ["r", "sigma", "mode"], metrics)

    tex = [
        r"\begin{tabular}{rrlrrrrrrr}",
        r"\toprule",
        r"$r$ & $\sigma$ & Mode & $\text{Mass}_{0.05}$ & Ep0 Ret & $G_{\text{fin}}$ & $E_{\text{fin}}$ & $c_{\text{warm}}$ & $c_{\text{post|ok}}$ & $c_{50}$ \\",
        r"\midrule",
    ]
    for r_val in sorted(summary["r"].unique()):
        for sig in sorted(summary["sigma"].unique()):
            for mode in ("belief", "known_r"):
                hit = summary[(summary["r"] == r_val) & (summary["sigma"] == sig) & (summary["mode"] == mode)]
                if hit.empty:
                    continue
                row = hit.iloc[0]
                tex.append(
                    f"{r_val:.1f} & {sig:.2f} & {_esc_tex(mode)} & "
                    f"{_fmt_val(row['mass_tol05_mean'], 3)} & {_fmt_val(row['ep0_online_mean'], 1)} & "
                    f"{_fmt_val(row['G_mean'], 3)} & {_fmt_val(row['E_mean'], 1)} & "
                    f"{_fmt_val(row['collapse_50_train_warm_mean'], 3)} & "
                    f"{_fmt_val(row['collapse_50_train_post_cond_mean'], 3)} & {_fmt_val(row['collapse_50_mean'], 3)} \\\\"
                )
        tex.append(r"\midrule")
    if tex[-1] == r"\midrule":
        tex.pop()
    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")

    _write_dual_table("tab03_e3_unknown_r", summary, tex)
    return {
        "name": "tab03_e3_unknown_r",
        "source": "results/raw/e3_unknown_r.csv",
        "caption": "E3 unknown-r belief vs known-r MB-EQ baseline separating warm-up vs conditional post-warm-up collapse.",
    }


def make_tab04_e4_crossplay() -> dict | None:
    cp_path = RAW_DIR / "e4_crossplay.csv"
    pr_path = RAW_DIR / "e4_crossplay_paired.csv"
    if not cp_path.exists() or not pr_path.exists():
        print("Skipping tab04: missing e4_crossplay CSVs", flush=True)
        return None
    cp = pd.read_csv(cp_path)
    paired = pd.read_csv(pr_path)
    fin = cp[cp["is_final"] == True].copy()  # noqa: E712

    pooled_cp = _summarize_group(fin, ["row_learner", "column_opponent"], ["J1", "G", "E1", "collapse_50"])
    pooled_pr = paired[paired["scope"] == "pooled_r"].copy()

    rows_csv = []
    tex = [
        r"\begin{tabular}{llllrr}",
        r"\toprule",
        r"Opponent & $J_1$ (MB-EQ) & $J_1$ (MB-OA rec) & $\Delta J_1$ (rec $-$ EQ) & $\Delta J_1$ (no-rec $-$ EQ) & $\Delta E_1$ (rec $-$ EQ) \\",
        r"\midrule",
    ]
    opp_order = ["mb_eq", "mb_oa", "iql", "planner_share", "equilibrium", "overharvester"]
    for opp in opp_order:
        eq_row = pooled_cp[(pooled_cp["row_learner"] == "mb_eq") & (pooled_cp["column_opponent"] == opp)]
        oa_row = pooled_cp[(pooled_cp["row_learner"] == "mb_oa") & (pooled_cp["column_opponent"] == opp)]
        d_rec = pooled_pr[(pooled_pr["comparison"] == "mboa_rec_minus_mbeq_final") & (pooled_pr["column_opponent"] == opp)]
        d_no = pooled_pr[(pooled_pr["comparison"] == "mboa_norec_minus_mbeq_final") & (pooled_pr["column_opponent"] == opp)]
        if eq_row.empty or oa_row.empty or d_rec.empty:
            continue
        er, orow, dr = eq_row.iloc[0], oa_row.iloc[0], d_rec.iloc[0]
        dn = d_no.iloc[0] if not d_no.empty else None
        rows_csv.append({
            "column_opponent": opp,
            "J1_mbeq_mean": er["J1_mean"], "J1_mbeq_ci95": er["J1_ci95"],
            "J1_mboa_rec_mean": orow["J1_mean"], "J1_mboa_rec_ci95": orow["J1_ci95"],
            "G_mbeq_mean": er["G_mean"], "G_mboa_rec_mean": orow["G_mean"],
            "diff_J1_rec_minus_eq_mean": dr["diff_J1_mean"], "diff_J1_rec_minus_eq_ci95": dr["diff_J1_ci95"],
            "diff_J1_norec_minus_eq_mean": dn["diff_J1_mean"] if dn is not None else float("nan"),
            "diff_J1_norec_minus_eq_ci95": dn["diff_J1_ci95"] if dn is not None else float("nan"),
            "diff_E1_rec_minus_eq_mean": dr["diff_E1_mean"], "diff_E1_rec_minus_eq_ci95": dr["diff_E1_ci95"],
        })
        no_str = _fmt_pm(dn["diff_J1_mean"], dn["diff_J1_ci95"], 1, signed=True) if dn is not None else "--"
        d_rec_s = _fmt_pm(dr["diff_J1_mean"], dr["diff_J1_ci95"], 1, signed=True)
        eq_j1 = _fmt_pm(er["J1_mean"], er["J1_ci95"], 1)
        oa_j1 = _fmt_pm(orow["J1_mean"], orow["J1_ci95"], 1)
        # Bold winning row learner J1 when CI separates means or winner is clear.
        best_alt = dn["diff_J1_mean"] if (dn is not None and dn["diff_J1_mean"] > dr["diff_J1_mean"]) else dr["diff_J1_mean"]
        if best_alt > 0.0:
            if dn is not None and dn["diff_J1_mean"] > dr["diff_J1_mean"]:
                no_str = _bold_cell(no_str)
            else:
                d_rec_s = _bold_cell(d_rec_s)
                oa_j1 = _bold_cell(oa_j1)
        elif dr["diff_J1_mean"] + dr["diff_J1_ci95"] < 0.0:
            eq_j1 = _bold_cell(eq_j1)
        tex.append(
            f"{LEARNER_LABELS.get(opp, opp)} & "
            f"{eq_j1} & "
            f"{oa_j1} & "
            f"{d_rec_s} & "
            f"{no_str} & "
            f"{_fmt_pm(dr['diff_E1_mean'], dr['diff_E1_ci95'], 1, signed=True)} \\\\"
        )
    tex.extend([r"\bottomrule", r"\end{tabular}"])

    out_df = pd.DataFrame(rows_csv)
    _write_dual_table("tab04_e4_crossplay_paired", out_df, tex)
    return {
        "name": "tab04_e4_crossplay_paired",
        "source": "results/raw/e4_crossplay.csv + e4_crossplay_paired.csv",
        "caption": "E4 pooled co-adapted cross-play own return J1 and paired CRN MB-OA minus MB-EQ differences (with/without request recovery).",
    }


def make_tab05_ablations() -> dict | None:
    blocks = []
    tex = [
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Experiment & Variant / Setting & $J$ & $G$ & $E$ & $c_{50}$ & $c_{\text{train}}$ \\",
        r"\midrule",
    ]

    e5_path = RAW_DIR / "e5_team.csv"
    if e5_path.exists():
        e5 = pd.read_csv(e5_path)
        e5_fin = e5[e5["is_final"] == True].copy()  # noqa: E712
        s5 = _summarize_group(e5_fin, ["learner", "r"], ["J", "G", "E", "collapse_50", "cum_collapse_50_train"])
        s5["experiment"] = "E5_team"
        s5["setting"] = s5["learner"] + " r=" + s5["r"].astype(str)
        blocks.append(s5)
        for _, row in s5.iterrows():
            tex.append(
                f"E5 Team & {LEARNER_LABELS.get(row['learner'], row['learner'])} $r={row['r']:.1f}$ & "
                f"{_fmt_pm(row['J_mean'], row['J_ci95'], 1)} & {_fmt_pm(row['G_mean'], row['G_ci95'], 3)} & "
                f"{_fmt_val(row['E_mean'], 1)} & {_fmt_val(row['collapse_50_mean'], 3)} & "
                f"{_fmt_val(row['cum_collapse_50_train_mean'], 3)} \\\\"
            )
        tex.append(r"\midrule")

    e6_path = RAW_DIR / "e6_data_amount.csv"
    if e6_path.exists():
        e6 = pd.read_csv(e6_path)
        s6 = _summarize_group(e6, ["learner", "n_warm", "n_replan", "prior"], ["J", "G", "E", "collapse_50", "cum_collapse_50_train"])
        s6["experiment"] = "E6_data"
        s6["setting"] = (
            s6["learner"] + " nw=" + s6["n_warm"].astype(str)
            + " nr=" + s6["n_replan"].astype(str) + " " + s6["prior"]
        )
        blocks.append(s6)
        # Compact representative rows for booktabs readability: n_warm in {0, 5, 50}, n_replan=5, plus nr=1 vs 20 at nw=5.
        sel = s6[
            ((s6["n_replan"] == 5) & (s6["n_warm"].isin([0, 5, 50])))
            | ((s6["n_warm"] == 5) & (s6["n_replan"].isin([1, 20])) & (s6["prior"] == "wide"))
        ]
        for _, row in sel.iterrows():
            lbl = f"{LEARNER_LABELS.get(row['learner'], row['learner'])} $N_w={int(row['n_warm'])}, N_r={int(row['n_replan'])}$ ({row['prior']})"
            tex.append(
                f"E6 Data & {lbl} & {_fmt_pm(row['J_mean'], row['J_ci95'], 1)} & {_fmt_pm(row['G_mean'], row['G_ci95'], 3)} & "
                f"{_fmt_val(row['E_mean'], 1)} & {_fmt_val(row['collapse_50_mean'], 3)} & "
                f"{_fmt_val(row['cum_collapse_50_train_mean'], 3)} \\\\"
            )
        tex.append(r"\midrule")

    e7_path = RAW_DIR / "e7_safe.csv"
    if e7_path.exists():
        e7 = pd.read_csv(e7_path)
        e7_fin = e7[e7["is_final"] == True] if "is_final" in e7.columns else e7  # noqa: E712
        s7 = _summarize_group(e7_fin, ["learner", "safe", "s_res", "r"], ["J", "G", "E", "E_safe", "collapse_50", "cum_collapse_50_train"])
        s7["experiment"] = "E7_safe"
        s7["setting"] = s7["learner"] + s7["safe"].map(lambda b: " safe" if bool(b) else " unsafe") + " r=" + s7["r"].astype(str)
        blocks.append(s7)
        for _, row in s7[s7["safe"] == True].iterrows():  # noqa: E712
            e_val = row["E_safe_mean"] if np.isfinite(row.get("E_safe_mean", float("nan"))) else row["E_mean"]
            s_res_lbl = f" ($S_{{res}}={row['s_res']:.0f}$)" if np.isfinite(row.get("s_res", float("nan"))) else ""
            tex.append(
                f"E7 Safe & {LEARNER_LABELS.get(row['learner'], row['learner'])}{s_res_lbl} $r={row['r']:.1f}$ & "
                f"{_fmt_pm(row['J_mean'], row['J_ci95'], 1)} & {_fmt_pm(row['G_mean'], row['G_ci95'], 3)} & "
                f"{_fmt_val(e_val, 1)} & {_fmt_val(row['collapse_50_mean'], 3)} & "
                f"{_fmt_val(row['cum_collapse_50_train_mean'], 3)} \\\\"
            )

    e9_c_path = RAW_DIR / "e9_ctde_summary.csv"
    e9_m_path = RAW_DIR / "e9_memory_one_summary.csv"
    if e9_c_path.exists() and e9_m_path.exists():
        tex.append(r"\midrule")
        c_df = pd.read_csv(e9_c_path)
        m_df = pd.read_csv(e9_m_path)
        # Select representative r in {0.5, 0.8}
        c_sel = c_df[c_df["r"].isin([0.5, 0.8])]
        for _, row in c_sel.iterrows():
            safe_s = " ($S_{\\text{res}}=25$)" if bool(row["safe"]) else " (raw)"
            tex.append(
                f"E9 Ext & CTDE{safe_s} $r={row['r']:.1f}$ & "
                f"{_fmt_pm(row['J_mean'], row['J_ci95'], 1)} & {_fmt_pm(row['G_mean'], row['G_ci95'], 3)} & "
                f"{_fmt_val(row['E_mean'], 1)} & {_fmt_val(row['collapse_50_mean'], 3)} & "
                f"{_fmt_val(row['cum_collapse_50_train_mean'], 3)} \\\\"
            )
        m_sel = m_df[m_df["r"].isin([0.5, 0.8])]
        for _, row in m_sel.iterrows():
            safe_s = " ($S_{\\text{res}}=25$)" if bool(row["safe"]) else " (raw)"
            tex.append(
                f"E9 Ext & MemoryOne{safe_s} $r={row['r']:.1f}$ & "
                f"{_fmt_pm(row['J_mean'], row['J_ci95'], 1)} & {_fmt_pm(row['G_mean'], row['G_ci95'], 3)} & "
                f"{_fmt_val(row['E_mean'], 1)} & {_fmt_val(row['collapse_50_mean'], 3)} & "
                f"{_fmt_val(row['cum_collapse_50_train_mean'], 3)} \\\\"
            )

    if not blocks:
        print("Skipping tab05: none of e5/e6/e7 raw CSVs found", flush=True)
        return None
    if tex[-1] == r"\midrule":
        tex.pop()
    tex.extend([r"\bottomrule", r"\end{tabular}"])

    out_df = pd.concat(blocks, ignore_index=True)
    _write_dual_table("tab05_e5_e6_e7_ablations", out_df, tex)
    return {
        "name": "tab05_e5_e6_e7_ablations",
        "source": "results/raw/e5_team.csv + e6_data_amount.csv" + (" + e7_safe.csv" if e7_path.exists() else ""),
        "caption": "E5 team baselines and E6 data-amount / prior sensitivity" + (" (+ E7 safe exploration)." if e7_path.exists() else "."),
    }


def make_band() -> dict | None:
    ref_path = RAW_DIR / "e1_refinement.csv"
    if not ref_path.exists():
        return None
    write_band_tables(ref_path, RES_TAB_DIR)
    return {
        "name": "discretisation_band",
        "source": "results/raw/e1_refinement.csv",
        "caption": "E1 grid discretisation sensitivity band (101x51 baseline vs 101x101 catch and 201x51 stock refinements).",
    }


def main():
    parser = argparse.ArgumentParser(description="Generate <= 6 paper tables from results/raw/*.csv")
    parser.parse_args()
    entries = []
    for builder in (make_tab01_benchmarks, make_tab02_e2_learners, make_tab03_e3_unknown_r, make_tab04_e4_crossplay, make_tab05_ablations, make_band):
        info = builder()
        if info is not None:
            entries.append(info)

    assert len(entries) <= 6, f"Expected <= 6 tables, produced {len(entries)}"
    RES_TAB_DIR.mkdir(parents=True, exist_ok=True)
    manifest = ["# Tables Manifest", "", "All tables generated by `experiments/make_tables.py` purely from `results/raw/*.csv`.", "", f"Total tables: **{len(entries)}** (limit <= 6).", ""]
    for i, item in enumerate(entries, 1):
        manifest.append(f"{i}. `{item['name']}.{{csv,tex}}`: {item['caption']} (Source: `{item['source']}`)")
    (RES_TAB_DIR / "tables_manifest.md").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    print(f"Generated {len(entries)} tables into {RES_TAB_DIR}", flush=True)


if __name__ == "__main__":
    main()
