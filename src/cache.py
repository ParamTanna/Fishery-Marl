"""On-demand planner and symmetric-equilibrium cache. Solved values are not recomputed."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path

import numpy as np

from dp_planner import PlannerResult, solve_planner
from equilibrium import EquilibriumResult, solve_equilibrium
from utils import ROOT


def _bench_hash(cfg):
    payload = {
        "model": {k: v for k, v in cfg.get("model", {}).items() if k not in ("r_values", "r_sanity", "reference_rule")},
        "grids": cfg.get("grids", {}),
        "solver": cfg.get("solver", {}),
        "selection": cfg.get("selection", {}),
    }
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


class BenchmarkCache:
    def __init__(self, cfg, rule=None, cache_dir=None, use_disk=True):
        self.cfg = cfg
        self.rule = rule or str(cfg["model"].get("reference_rule", "max_total"))
        self.planner = {}
        self.equilibrium = {}
        self.use_disk = bool(use_disk)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else (ROOT / "results" / "cache")
        self._bhash = _bench_hash(cfg)

    def _planner_path(self, r):
        return self.cache_dir / f"plan_{self._bhash}_r{float(r):.6g}.npz"

    def _eq_path(self, r, rule, symmetric_only=True):
        sym_tag = "sym" if symmetric_only else "full"
        return self.cache_dir / f"eq_{self._bhash}_r{float(r):.6g}_{rule}_{sym_tag}.npz"

    def get_planner(self, r):
        key = float(r)
        if key not in self.planner:
            path = self._planner_path(key)
            if self.use_disk and path.exists():
                with np.load(path, allow_pickle=False) as d:
                    self.planner[key] = PlannerResult(
                        model=str(d["model"].item()),
                        r=float(d["r"].item()),
                        H=int(d["H"].item()),
                        converged=bool(d["converged"].item()),
                        delta=float(d["delta"].item()),
                        s_grid=np.array(d["s_grid"]),
                        a_grid=np.array(d["a_grid"]),
                        V=np.array(d["V"]),
                        a1_idx=np.array(d["a1_idx"]),
                        a2_idx=np.array(d["a2_idx"]),
                        executed=np.array(d["executed"]),
                    )
                return self.planner[key]
            res = solve_planner(self.cfg, key, verbose=False)
            self.planner[key] = res
            if self.use_disk:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                tmp = self.cache_dir / f"{path.name}.{os.getpid()}.{uuid.uuid4().hex[:6]}.npz.tmp"
                with open(tmp, "wb") as fh:
                    np.savez(
                        fh,
                        model=np.array(res.model),
                        r=np.array(res.r),
                        H=np.array(res.H),
                        converged=np.array(res.converged),
                        delta=np.array(res.delta),
                        s_grid=res.s_grid,
                        a_grid=res.a_grid,
                        V=res.V,
                        a1_idx=res.a1_idx,
                        a2_idx=res.a2_idx,
                        executed=res.executed,
                    )
                os.replace(tmp, path)
        return self.planner[key]

    def get_equilibrium(self, r, rule=None, symmetric_only=True):
        used = self.rule if rule is None else str(rule)
        key = (float(r), used)
        if not symmetric_only:
            key_full = (float(r), used, False)
            path = self._eq_path(float(r), used, symmetric_only=False)
        else:
            key_full = key
            path = self._eq_path(float(r), used, symmetric_only=True)
        store = self.equilibrium if symmetric_only else getattr(self, "_eq_asym", {})
        if not symmetric_only and not hasattr(self, "_eq_asym"):
            self._eq_asym = store
        if key not in store:
            if self.use_disk and path.exists():
                with np.load(path, allow_pickle=False) as d:
                    store[key] = EquilibriumResult(
                        model=str(d["model"].item()),
                        r=float(d["r"].item()),
                        rule=str(d["rule"].item()),
                        symmetric_only=bool(d["symmetric_only"].item()),
                        H=int(d["H"].item()),
                        converged=bool(d["converged"].item()),
                        delta=float(d["delta"].item()),
                        s_grid=np.array(d["s_grid"]),
                        a_grid=np.array(d["a_grid"]),
                        V1=np.array(d["V1"]),
                        V2=np.array(d["V2"]),
                        p1=np.array(d["p1"]),
                        p2=np.array(d["p2"]),
                        n_eq=np.array(d["n_eq"]),
                        selected_kind=np.array(d["selected_kind"]),
                        stage_exploit_max=float(d["stage_exploit_max"].item()),
                        eps_rel=float(d["eps_rel"].item()),
                        eps_rel_threshold=float(d["eps_rel_threshold"].item()),
                        n_fallback=int(d["n_fallback"].item()),
                        n_mixed_selected=int(d["n_mixed_selected"].item()),
                        n_truncated_mixed=int(d["n_truncated_mixed"].item()),
                        n_trivial=int(d["n_trivial"].item()),
                        selection_seed=int(d["selection_seed"].item()),
                        E_s0=float(d["E_s0"].item()) if "E_s0" in d else 0.0,
                    )
                return store[key]
            res = solve_equilibrium(
                self.cfg, float(r), rule=used, symmetric_only=symmetric_only, verbose=False
            )
            store[key] = res
            if self.use_disk:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                tmp = self.cache_dir / f"{path.name}.{os.getpid()}.{uuid.uuid4().hex[:6]}.npz.tmp"
                with open(tmp, "wb") as fh:
                    np.savez(
                        fh,
                        model=np.array(res.model),
                        r=np.array(res.r),
                        rule=np.array(res.rule),
                        symmetric_only=np.array(res.symmetric_only),
                        H=np.array(res.H),
                        converged=np.array(res.converged),
                        delta=np.array(res.delta),
                        s_grid=res.s_grid,
                        a_grid=res.a_grid,
                        V1=res.V1,
                        V2=res.V2,
                        p1=res.p1,
                        p2=res.p2,
                        n_eq=res.n_eq,
                        selected_kind=res.selected_kind,
                        stage_exploit_max=np.array(res.stage_exploit_max),
                        eps_rel=np.array(res.eps_rel),
                        eps_rel_threshold=np.array(res.eps_rel_threshold),
                        n_fallback=np.array(res.n_fallback),
                        n_mixed_selected=np.array(res.n_mixed_selected),
                        n_truncated_mixed=np.array(res.n_truncated_mixed),
                        n_trivial=np.array(res.n_trivial),
                        selection_seed=np.array(res.selection_seed),
                        E_s0=np.array(res.E_s0),
                    )
                os.replace(tmp, path)
        return store[key]

    def ensure_rule(self, rule=None):
        """Solve the configured r values once. r_hat is then interpolated, not re-solved.

        The belief grid has 191 nodes. Solving an equilibrium at each of them does not
        fit the machine. The cache is the frozen r list (and the sanity r).
        """
        used = self.rule if rule is None else str(rule)
        if any(name == used for (_r, name) in self.equilibrium):
            return
        values = [float(r) for r in self.cfg["model"]["r_values"]]
        values += [float(r) for r in self.cfg["model"].get("r_sanity", [])]
        if not values:
            values = [0.5]
        for r in values:
            self.get_equilibrium(r, used)

    def interpolated_equilibrium(self, r_hat, rule=None):
        """Mix the solved policies on either side of r_hat. Probabilities are renormalised."""
        used = self.rule if rule is None else str(rule)
        solved = sorted(r for (r, rule_name) in self.equilibrium if rule_name == used)
        if not solved:
            eq = self.get_equilibrium(r_hat, used)
            return np.array(eq.p1, copy=True), np.array(eq.p2, copy=True)
        if r_hat <= solved[0] or r_hat >= solved[-1] or len(solved) == 1:
            anchor = solved[0] if r_hat <= solved[0] else solved[-1]
            if r_hat >= solved[-1]:
                anchor = solved[-1]
            if len(solved) == 1:
                anchor = solved[0]
            eq = self.equilibrium[(anchor, used)]
            return np.array(eq.p1, copy=True), np.array(eq.p2, copy=True)
        hi = next(r for r in solved if r >= r_hat)
        lo = max(r for r in solved if r <= r_hat)
        if hi == lo:
            eq = self.equilibrium[(lo, used)]
            return np.array(eq.p1, copy=True), np.array(eq.p2, copy=True)
        weight = (float(r_hat) - lo) / (hi - lo)
        left = self.equilibrium[(lo, used)]
        right = self.equilibrium[(hi, used)]
        p1 = (1.0 - weight) * left.p1 + weight * right.p1
        p2 = (1.0 - weight) * left.p2 + weight * right.p2
        p1 = p1 / np.maximum(p1.sum(axis=1, keepdims=True), 1.0e-300)
        p2 = p2 / np.maximum(p2.sum(axis=1, keepdims=True), 1.0e-300)
        return p1, p2

    def interpolated_planner(self, r_hat):
        """Planner policies at the configured r values, mixed when r_hat falls between them."""
        if not self.planner:
            values = [float(r) for r in self.cfg["model"]["r_values"]]
            values += [float(r) for r in self.cfg["model"].get("r_sanity", [])]
            for r in values or [float(r_hat)]:
                self.get_planner(r)
        solved = sorted(self.planner)
        if r_hat <= solved[0]:
            anchor = solved[0]
            weight = 0.0
            other = solved[0]
        elif r_hat >= solved[-1]:
            anchor = solved[-1]
            weight = 0.0
            other = solved[-1]
        else:
            other = next(r for r in solved if r >= r_hat)
            anchor = max(r for r in solved if r <= r_hat)
            weight = 0.0 if other == anchor else (float(r_hat) - anchor) / (other - anchor)
        left_plan, left_p1, left_p2 = self.planner_tables(anchor)
        if weight == 0.0:
            return left_plan, left_p1, left_p2
        _right, right_p1, right_p2 = self.planner_tables(other)
        p1 = (1.0 - weight) * left_p1 + weight * right_p1
        p2 = (1.0 - weight) * left_p2 + weight * right_p2
        p1 = p1 / np.maximum(p1.sum(axis=1, keepdims=True), 1.0e-300)
        p2 = p2 / np.maximum(p2.sum(axis=1, keepdims=True), 1.0e-300)
        return left_plan, p1, p2

    def planner_tables(self, r):
        plan = self.get_planner(r)
        n_a = plan.a_grid.shape[0]
        p1 = np.zeros((plan.s_grid.shape[0], n_a), dtype=np.float64)
        p2 = np.zeros_like(p1)
        p1[np.arange(p1.shape[0]), plan.a1_idx] = 1.0
        p2[np.arange(p2.shape[0]), plan.a2_idx] = 1.0
        return plan, p1, p2
