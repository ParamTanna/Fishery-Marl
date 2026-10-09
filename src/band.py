"""Catch-grid band read from the refinement CSV. Nothing here is typed by hand."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def catch_grid_band(refinement_csv):
    """Relative change at r=0.5, symmetric max_total, from the refinement file.

    The working grid is 101 x 51. The catch-grid comparison is 101 x 101.
    The stock-grid comparison is 201 x 51.
    """
    frame = pd.read_csv(refinement_csv)
    rule = frame[frame["rule"] == "max_total"] if "rule" in frame.columns else frame

    def value(n_stock, n_catch):
        hit = rule[(rule["n_stock"] == n_stock) & (rule["n_catch"] == n_catch)]
        if len(hit) != 1:
            raise ValueError(f"expected one refinement row for {n_stock} x {n_catch}, found {len(hit)}")
        row = hit.iloc[0]
        return float(row["J_eq"]), bool(row["eq_converged"])

    base, base_ok = value(101, 51)
    finer_catch, catch_ok = value(101, 101)
    finer_stock, stock_ok = value(201, 51)
    return {
        "band_r": 0.5,
        "band_rule": "max_total",
        "J_eq_101x51": base,
        "J_eq_101x101": finer_catch,
        "J_eq_201x51": finer_stock,
        "catch_grid_rel_change": (finer_catch - base) / base,
        "stock_grid_rel_change": (finer_stock - base) / base,
        "band_catch_grids_converged": bool(base_ok and catch_ok),
        "band_stock_grids_converged": bool(base_ok and stock_ok),
    }


def write_band_tables(refinement_csv, table_dir):
    band = catch_grid_band(refinement_csv)
    table_dir = Path(table_dir)
    table_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([band])
    csv_path = table_dir / "discretisation_band.csv"
    frame.to_csv(csv_path, index=False)
    tex_path = table_dir / "discretisation_band.tex"
    rel_catch = 100.0 * band["catch_grid_rel_change"]
    rel_stock = 100.0 * band["stock_grid_rel_change"]
    tex = "\n".join(
        [
            r"\begin{tabular}{lrr}",
            r"\toprule",
            r"grid & $J_{eq}$ & change vs 101$\times$51 \\",
            r"\midrule",
            f"101 $\\times$ 51 & {band['J_eq_101x51']:.2f} & 0 \\\\",
            f"101 $\\times$ 101 & {band['J_eq_101x101']:.2f} & {rel_catch:.2f}\\% \\\\",
            f"201 $\\times$ 51 & {band['J_eq_201x51']:.2f} & {rel_stock:.3f}\\% \\\\",
            r"\bottomrule",
            r"\end{tabular}",
            "",
        ]
    )
    tex_path.write_text(tex, encoding="utf-8")
    return csv_path, tex_path, band
