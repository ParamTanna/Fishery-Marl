"""Build every results table (report/tables/*.tex) and figure (report/media/*.pdf, docs/images/*.png) of the report
from results/raw, results/tables (experiments/make_tables.py) and results/exact (E10-E12).
Nothing in the Results section of the report is typed by hand.

Run: python experiments/make_report_assets.py"""
import argparse, json, math, os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--raw", default=str(ROOT / "results" / "raw"))
ap.add_argument("--tables", default=str(ROOT / "results" / "tables"))
ap.add_argument("--exact", default=str(ROOT / "results" / "exact"))
ap.add_argument("--out", default=str(ROOT / "report"))
ap.add_argument("--images", default=str(ROOT / "docs" / "images"))
A = ap.parse_args()
os.makedirs(A.images, exist_ok=True)
os.makedirs(os.path.join(A.out, "tables"), exist_ok=True)
os.makedirs(os.path.join(A.out, "media"), exist_ok=True)
R4 = [0.5, 0.6, 0.7, 0.8]
raw = lambda n: pd.read_csv(os.path.join(A.raw, n))
tab = lambda n: pd.read_csv(os.path.join(A.tables, n))
exact = lambda n: json.load(open(os.path.join(A.exact, n), encoding="utf-8"))
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
                     "figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False})
MK = {"planner": ("k", "*"), "mpe": ("k", "o"), "greedy": ("0.45", "x"), "iql": ("0.25", "v"), "mbeq": ("0.25", "s"),
      "mboa": ("k", "D"), "ctde": ("0.45", "^"), "team": ("0.45", "P"), "safe": ("0.25", "h")}


def fmt(x, nd=1, sign=False):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    s = f"{x:+,.{nd}f}" if sign else f"{x:,.{nd}f}"
    return s.replace(",", "{,}")


def write(name, body):
    open(os.path.join(A.out, "tables", name), "w", encoding="utf-8").write(body)
    print("wrote tables/" + name)


# ------------------------------------------------------------------ data
E10 = exact("e10_equilibria.json")
band = {round(x["r"], 2): x for x in E10["band"]}
greedy = {round(e["r"], 2): e for e in E10["profiles"] if e["rule"] == "max_total"}
t2 = tab("tab02_e2_learners.csv")
e2 = raw("e2_summary.csv"); e2 = e2[(e2.gamma == 0.99) & (e2.sigma == 0.1) & (e2.is_sanity == False)]
pick = lambda df, **kw: df[np.logical_and.reduce([np.isclose(df[k], v) if isinstance(v, float) else (df[k] == v) for k, v in kw.items()])]

# ------------------------------------------------------------------ Table: benchmarks (TA feedback 2)
rows = []
for r in R4:
    b, g = band[r], greedy[r]
    rows.append(f"{r:.1f} & {fmt(b['J_plan'])} & {fmt(b['J_mpe_worst'])}--{fmt(b['J_mpe_best'])} & {b['PoA']:.3f} & {b['PoS']:.3f} & "
                f"{fmt(b['E_plan_s0'])} & {fmt(g['J'])} & {fmt(g['E_s0'])} \\\\")
write("tab_bench.tex", "\\begin{tabular}{cccccccc}\n\\toprule\n"
      " & Planner & Certified MPE band & & & Planner & \\multicolumn{2}{c}{Stage-greedy (max-total)} \\\\\n"
      "$r$ & $J^{\\text{plan}}$ & $[\\min,\\max]\\,J^{*}$ & PoA & PoS & exploitability $E$ & $J$ & $E$ (not an MPE) \\\\\n\\midrule\n"
      + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

# ------------------------------------------------------------------ Table: learners in self-play
def lrow(name, rule, lr, budget):
    out = []
    for r in R4:
        x = t2[(t2.rule == rule) & (t2.learner == lr) & np.isclose(t2.r, r)]
        if x.empty:
            out.append("--")
            continue
        x = x.iloc[0]
        out.append(f"{x.G_mean:.3f} / {x.collapse_50_mean:.2f} / {fmt(x.E_mean, 0)}")
    return f"{name} ({budget}) & " + " & ".join(out) + " \\\\"


e5 = raw("e5_summary.csv"); c9 = raw("e9_ctde_summary.csv")
def srow(name, df, lr, extra=""):
    out = []
    for r in R4:
        x = df[(df.learner == lr) & np.isclose(df.r, r)]
        out.append("--" if x.empty else f"{x.iloc[0].G_mean:.3f} / {x.iloc[0].collapse_50_mean:.2f} / {fmt(x.iloc[0].E_mean, 0)}")
    return f"{name}{extra} & " + " & ".join(out) + " \\\\"


L = [lrow("IQL", "max_total", "iql", 5000),
     lrow("MB-EQ, certified MPE", "min_total", "mb_eq", 500),
     lrow("MB-EQ, stage-greedy profile", "max_total", "mb_eq", 500),
     lrow("MB-OA", "max_total", "mb_oa", 200),
     srow("Joint-action critic", c9[c9.safe == False], "ctde", " (5000)"),
     srow("Team-IQL", e5, "team_iql", " (5000)")]
write("tab_learners.tex", "\\begin{tabular}{lcccc}\n\\toprule\n"
      "Learner (episodes) & $r=0.5$ & $r=0.6$ & $r=0.7$ & $r=0.8$ \\\\\n\\midrule\n" + "\n".join(L) +
      "\n\\bottomrule\n\\end{tabular}\n")

# cost of learning (per-episode, vs planner J50)
lt = {lr: [t2[(t2.rule == "max_total") & (t2.learner == lr) & np.isclose(t2.r, r)].iloc[0].L_tau_plan_per_ep_mean for r in R4]
      for lr in ("iql", "mb_eq", "mb_oa")}
json.dump(lt, open(os.path.join(A.out, "tables", "ltau.json"), "w"), indent=1)

# ------------------------------------------------------------------ Table: cross-play (TA feedback 1), per r, Holm
p4 = raw("e4_crossplay_paired.csv")
fam = p4[(p4.scope == "by_r") & (p4.comparison == "mboa_rec_minus_mbeq_final")].copy()
fam["t"] = fam.diff_J1_mean / fam.diff_J1_se
fam["p"] = 2 * stats.t.sf(np.abs(fam.t), df=fam.n - 1)
order = np.argsort(fam.p.values); m = len(fam); adj = np.empty(m); run = 0.0
for k, i in enumerate(order):
    run = max(run, min(1.0, (m - k) * fam.p.values[i])); adj[i] = run
fam["p_holm"] = adj
opp_names = {"mb_eq": "MB-EQ", "mb_oa": "MB-OA", "iql": "IQL$^{\\dagger}$", "planner_share": "Planner share",
             "equilibrium": "Stage-greedy profile", "overharvester": "Over-harvester ($a=14$)"}
rows = []
for o, nm in opp_names.items():
    cells = []
    for r in R4:
        x = fam[(fam.column_opponent == o) & np.isclose(fam.r, r)].iloc[0]
        star = "$^{*}$" if x.p_holm < 0.05 else ""
        cells.append(f"{fmt(x.diff_J1_mean, 1, True)}{star}")
    pooled = p4[(p4.scope == "pooled_r") & (p4.comparison == "mboa_rec_minus_mbeq_final") & (p4.column_opponent == o)].iloc[0]
    rows.append(f"{nm} & " + " & ".join(cells) + f" & {fmt(pooled.diff_J1_mean, 1, True)} $\\pm$ {fmt(pooled.diff_J1_ci95, 1)} \\\\")
write("tab_crossplay.tex", "\\begin{tabular}{lccccc}\n\\toprule\n"
      "Opponent & $r=0.5$ & $r=0.6$ & $r=0.7$ & $r=0.8$ & pooled (95\\% CI) \\\\\n\\midrule\n" + "\n".join(rows) +
      "\n\\bottomrule\n\\end{tabular}\n")
fam.to_csv(os.path.join(A.out, "tables", "crossplay_holm.csv"), index=False)

# ------------------------------------------------------------------ Table: safety decomposition
e7 = raw("e7_summary.csv"); bm = raw("e8_selection_benchmarks.csv")
rows = []
for r in R4:
    b = band[r]
    c_mpe = bm[bm.symmetric_only & (bm.rule == "min_total") & np.isclose(bm.r, r)].iloc[0].collapse_50_eq
    c_safe = e7[(e7.safe == True) & (e7.learner == "mb_eq") & np.isclose(e7.r, r)].iloc[0].collapse_50_mean
    rows.append(f"{r:.1f} & {fmt(b['J_plan'])} & {fmt(b['J_plan_capped'])} & {fmt(b['J_mpe_best'])} & {fmt(b['J_safe_mpe'])} & "
                f"{fmt(b['cap_cost_equilibrium_vs_best'], 1, True)} & {c_mpe:.3f} & {c_safe:.3f} \\\\")
write("tab_safety.tex", "\\begin{tabular}{cccccccc}\n\\toprule\n"
      " & \\multicolumn{2}{c}{Planner} & \\multicolumn{2}{c}{Best certified MPE} & Cost of cap & \\multicolumn{2}{c}{$C_{50}$ of MPE} \\\\\n"
      "$r$ & uncapped & capped & uncapped & capped & (equilibrium) & uncapped & capped \\\\\n\\midrule\n" + "\n".join(rows) +
      "\n\\bottomrule\n\\end{tabular}\n")

# ------------------------------------------------------------------ Table: cooperation threshold + memory-one probe
coop = [{"r": c["r"], "rows": c["rows"], "lo": c["gamma_star_lower"], "hi": c["gamma_star_upper"]}
        for c in exact("e11_cooperation.json")] if os.path.exists(os.path.join(A.exact, "e11_cooperation.json")) else []
m1 = raw("e9_memory_one_summary.csv")
rows = []
for c in coop:
    r = c["r"]
    if r > 1.0:
        continue
    g99 = [x for x in c["rows"] if abs(x["gamma"] - 0.99) < 1e-9]
    gs = (f"{(c['lo'] + c['hi']) / 2:.3f} $\\pm$ {(c['hi'] - c['lo']) / 2:.3f}" if c["lo"] is not None and c["hi"] is not None
          else ("$>0.99$" if c["hi"] is None else f"$\\le${c['hi']}"))
    gain = fmt(g99[0]["J_coop"] - g99[0]["J_mpe"], 1, True) if g99 else "--"
    mm = m1[(m1.learner == "memory_one") & np.isclose(m1.r, r)]
    gap = f"{mm.iloc[0].reaction_gap_above_25_mean:+.2f} $\\pm$ {mm.iloc[0].reaction_gap_above_25_ci95:.2f}" if not mm.empty else "--"
    rows.append(f"{r:.1f} & {gs} & {gain} & {gap} \\\\")
write("tab_coop.tex", "\\begin{tabular}{cccc}\n\\toprule\n"
      "$r$ & $\\gamma^*$ (SPE iff $\\gamma\\ge\\gamma^*$) & $J^{\\text{coop}}-J^{*}$ ($\\gamma=0.99$) & memory-one gap \\\\\n\\midrule\n"
      + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

# ------------------------------------------------------------------ Table: MB-EQ at regrowth rates it was not solved for
og = os.path.join(A.exact, "e12_unsolved_r.json")
if os.path.exists(og):
    rows = [f"{x['r']:.2f} & {fmt(x['J_mpe'])} & {fmt(x['J_interp'])} & {100 * (1 - x['J_interp'] / x['J_mpe']):.1f}\\% & "
            f"{fmt(x['E_interp_sup'])} ({100 * x['E_interp_sup'] / (x['J_interp'] / 2):.1f}\\%) \\\\" for x in json.load(open(og))]
    write("tab_offgrid.tex", "\\begin{tabular}{ccccc}\n\\toprule\n"
          "true $r$ & certified MPE $J^*$ & MB-EQ policy $J$ & welfare loss & exploitability $E$ (\\% of value) \\\\\n\\midrule\n"
          + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

# ------------------------------------------------------------------ Figure: benchmarks vs r + exploitability/efficiency frontier
fig, axs = plt.subplots(2, 2, figsize=(7.0, 3.9))
ax = [axs[0, 0], axs[1, 1]]
rs = np.array(R4)
# (b) stationary catch policies and (c) expected stock paths at r = 0.5, from E1
pol = raw("e1_policies.csv"); pth = raw("e1_paths.csv")
pol = pol[np.isclose(pol.r, 0.5) & (pol.symmetric_only == True)]
pth = pth[np.isclose(pth.r, 0.5) & (pth.symmetric_only == True)]
pm, pg = pol[pol.rule == "min_total"].sort_values("s"), pol[pol.rule == "max_total"].sort_values("s")
axs[0, 1].plot(pm.s, pm.catch_plan, "k-", lw=1.2, label="planner")
axs[0, 1].plot(pm.s, pm.catch_eq, "-", color="0.35", lw=1.0, label="certified MPE")
axs[0, 1].plot(pg.s, pg.catch_eq, "--", color="0.55", lw=1.0, label="stage-greedy")
axs[0, 1].set_xlabel("stock $s$"); axs[0, 1].set_ylabel("total executed catch")
axs[0, 1].set_title("(b) Catch policies, $r=0.5$"); axs[0, 1].legend(frameon=False, loc="lower right", fontsize=6.5)
qm, qg = pth[pth.rule == "min_total"].sort_values("t"), pth[pth.rule == "max_total"].sort_values("t")
axs[1, 0].plot(qm.t, qm.stock_plan, "k-", lw=1.2, label="planner")
axs[1, 0].plot(qm.t, qm.stock_eq, "-", color="0.35", lw=1.0, label="certified MPE")
axs[1, 0].plot(qg.t, qg.stock_eq, "--", color="0.55", lw=1.0, label="stage-greedy")
axs[1, 0].axhline(20, color="0.6", ls=":", lw=0.8)
axs[1, 0].set_xlabel("season $t$"); axs[1, 0].set_ylabel("expected stock")
axs[1, 0].set_title("(c) Stock paths from $s_0=K$, $r=0.5$"); axs[1, 0].legend(frameon=False, loc="upper right")
ax[0].plot(rs, [band[r]["J_plan"] for r in R4], "k*-", label="planner")
ax[0].fill_between(rs, [band[r]["J_mpe_worst"] for r in R4], [band[r]["J_mpe_best"] for r in R4], color="0.6", alpha=0.6, label="certified MPE band")
ax[0].plot(rs, [band[r]["J_mpe_best"] for r in R4], "ko-", ms=3, lw=0.8)
ax[0].plot(rs, [greedy[r]["J"] for r in R4], "x--", color="0.45", label="stage-greedy (not an MPE)")
ax[0].plot(rs, [band[r]["J_safe_mpe"] for r in R4], "h:", color="0.25", ms=4, label="MPE with reserve cap")
ax[0].set_xlabel("regrowth rate $r$"); ax[0].set_ylabel("total discounted return $J$"); ax[0].set_title("(a) Exact benchmarks")
ax[0].set_ylim(5600, 13600); ax[0].legend(frameon=False, loc="lower right", fontsize=6.5)
r0 = 0.5
pts = [("planner", 0.0, band[r0]["E_plan_s0"]), ("certified MPE", band[r0]["G_best"], 0.0),
       ("stage-greedy", 1 - greedy[r0]["J"] / band[r0]["J_plan"], greedy[r0]["E_s0"])]
for key, lr, rule in (("iql", "iql", "max_total"), ("mbeq", "mb_eq", "min_total"), ("mboa", "mb_oa", "max_total")):
    x = t2[(t2.rule == rule) & (t2.learner == lr) & np.isclose(t2.r, r0)]
    if not x.empty:
        pts.append(({"iql": "IQL", "mbeq": "MB-EQ (cert.)", "mboa": "MB-OA"}[key], x.iloc[0].G_mean, x.iloc[0].E_mean))
x = c9[(c9.learner == "ctde") & np.isclose(c9.r, r0)]
if not x.empty:
    pts.append(("joint-action critic", x.iloc[0].G_mean, x.iloc[0].E_mean))
keymap = {"planner": "planner", "certified MPE": "mpe", "stage-greedy": "greedy", "IQL": "iql", "MB-EQ (cert.)": "mbeq",
          "MB-OA": "mboa", "joint-action critic": "ctde"}
OFF = {"planner": (4, 6), "certified MPE": (6, 6), "stage-greedy": (5, -9), "IQL": (-14, -11), "MB-EQ (cert.)": (6, -9),
       "MB-OA": (6, -9), "joint-action critic": (-38, 7)}
for nm, gx, ex in pts:
    col, mk = MK[keymap[nm]]
    ax[1].scatter(gx, max(ex, 0.0), c=col, marker=mk, s=28, zorder=3)
    ax[1].annotate(nm, (gx, max(ex, 0.0)), textcoords="offset points", xytext=OFF.get(nm, (4, 3)), fontsize=6.5)
ax[1].set_xlim(-0.05, 1.0); ax[1].set_ylim(-1, 2e4)
ax[1].set_yscale("symlog", linthresh=10); ax[1].set_xlabel("efficiency gap $G$ (0 = planner)")
ax[1].set_ylabel("exploitability $E$"); ax[1].set_title(f"(d) Efficiency vs exploitability, $r={r0}$")
fig.tight_layout(); fig.savefig(os.path.join(A.out, "media", "fig_bench_frontier.pdf")); fig.savefig(os.path.join(A.images, "fig_bench_frontier.png"), dpi=200); plt.close(fig)
print("wrote media/fig_bench_frontier.pdf")

# ------------------------------------------------------------------ Figure: learning curves (G vs episode)
lb = raw("e8_learners_by_rule.csv"); l2 = raw("e2_learners.csv")
l2 = l2[(l2.gamma == 0.99) & (l2.sigma == 0.1) & (l2.is_sanity == False)]
fig, axg = plt.subplots(2, 4, figsize=(7.2, 2.75), sharey="row", sharex="col")
for k, r in enumerate(R4):
    for lab, df, rule, lr, key in (("IQL", l2, None, "iql", "iql"), ("MB-EQ (cert.)", lb, "min_total", "mb_eq", "mbeq"),
                                   ("MB-OA", l2, None, "mb_oa", "mboa")):
        d = df[(df.learner == lr) & np.isclose(df.r, r)]
        if rule is not None:
            d = d[d.rule == rule]
        col, mk = MK[key]
        for row, metric in ((0, "G"), (1, "collapse_50")):
            g = d.groupby("episode")[metric].agg(["mean", "std", "count"]).reset_index()
            hw = stats.t.ppf(0.975, g["count"] - 1) * g["std"] / np.sqrt(g["count"])
            axg[row, k].plot(g.episode + 1, g["mean"], color=col, marker=mk, ms=2.5, lw=0.9, label=lab)
            axg[row, k].fill_between(g.episode + 1, g["mean"] - hw, g["mean"] + hw, color=col, alpha=0.15, lw=0)
    axg[0, k].axhline(band[r]["G_best"], color="0.4", ls=":", lw=0.8)
    axg[0, k].set_title(f"$r={r}$"); axg[1, k].set_xscale("log"); axg[1, k].set_xlabel("episode")
axg[0, 0].set_ylabel("efficiency gap $G$"); axg[1, 0].set_ylabel("collapse prob. $C_{50}$")
axg[0, 0].legend(frameon=False, loc="center left", fontsize=6.5)
ax = axg[0]
fig.tight_layout(); fig.savefig(os.path.join(A.out, "media", "fig_curves.pdf")); fig.savefig(os.path.join(A.images, "fig_curves.png"), dpi=200); plt.close(fig)
print("wrote media/fig_curves.pdf")

# ------------------------------------------------------------------ Figure: cooperation margin vs gamma
if coop:
    fig, ax = plt.subplots(figsize=(3.6, 2.2))
    for c, mk, ls in zip([c for c in coop if c["r"] < 1.0], "osD^", ["-", "--", "-.", ":"]):
        g = [x["gamma"] for x in c["rows"]]; fv = [x["n_states_violated"] / 101 for x in c["rows"]]
        ax.plot(g, fv, marker=mk, ms=2.5, lw=0.9, ls=ls, color="k", label=f"$r={c['r']}$")
    ax.set_xlabel("discount $\\gamma$"); ax.set_ylabel("share of stocks where\ndeviating pays")
    ax.set_xlim(0.28, 1.0); ax.legend(frameon=False, fontsize=6.5); fig.tight_layout()
    fig.savefig(os.path.join(A.out, "media", "fig_coop.pdf")); fig.savefig(os.path.join(A.images, "fig_coop.png"), dpi=200); plt.close(fig)
    print("wrote media/fig_coop.pdf")

# ------------------------------------------------------------------ numbers used in the text
e3 = raw("e3_unknown_r.csv")
e3 = e3[(e3.episode == 0) & np.isclose(e3.sigma, 0.1)]
nums = {}
for r in R4:
    a = e3[(e3["mode"] == "belief") & np.isclose(e3.r, r)].set_index("seed").online_return
    b = e3[(e3["mode"] == "known_r") & np.isclose(e3.r, r)].set_index("seed").online_return
    d = (a - b).dropna()
    nums[f"ep0_paired_diff_r{r}"] = [float(d.mean()), float(stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d)))]
nums["ltau"] = lt
e6s = raw("e6_summary.csv"); oa6 = e6s[e6s.learner == "mb_oa"]
nums["e6_mboa_train_collapse_by_nwarm"] = {int(k): [float(v.min()), float(v.mean()), float(v.max())]
                                           for k, v in oa6.groupby("n_warm").cum_collapse_50_train_mean}
nums["e6_max_G_spread_replan_prior"] = e6s.groupby(["learner", "n_warm", "r"]).G_mean.agg(lambda s: s.max() - s.min()).groupby(level=0).max().to_dict()
nums["e3_warmup_collapse_range"] = [float(tab("tab03_e3_unknown_r.csv").collapse_50_train_warm_mean.min()),
                                    float(tab("tab03_e3_unknown_r.csv").collapse_50_train_warm_mean.max())]
e2all = raw("e2_summary.csv"); e2all = e2all[e2all.r.isin(R4) & (e2all.is_sanity == False)]
nums["e2_collapse_by_gamma_sigma_r_pooled"] = {
    f"{lr} gamma={g} sigma={s}": float(v) for (lr, g, s), v in
    e2all[e2all.learner.isin(["mb_eq", "mb_oa"])].groupby(["learner", "gamma", "sigma"]).collapse_50_mean.mean().items()}
json.dump(nums, open(os.path.join(A.out, "tables", "numbers.json"), "w"), indent=1)
print("wrote tables/numbers.json")
