"""One command to test, reproduce or re-run the whole project.

  python run_all.py --reproduce   tests, then every table and figure from the saved results (a few minutes)
  python run_all.py --quick       tests plus small smoke runs of the main experiments (about 15 minutes)
  python run_all.py --full        every experiment from scratch, then tables and figures (many hours)

Use --workers N to set the number of parallel processes for the learning experiments.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXP = ROOT / "experiments"


def run(args: list[str]) -> None:
    cmd = [sys.executable, *args]
    print("\n$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--reproduce", action="store_true")
    mode.add_argument("--quick", action="store_true")
    mode.add_argument("--full", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    w = ["--workers", str(args.workers)]

    run(["-m", "pytest", "tests", "-q"])
    if args.quick:
        run([str(EXP / "e1_benchmarks.py"), "--quick"])
        run([str(EXP / "e2_learners.py"), "--quick"])
        run([str(EXP / "e3_unknown_r.py"), "--quick"])
        run([str(EXP / "e4_crossplay.py"), "--quick"])
        run([str(EXP / "e7_safe.py"), "--quick"])
        run([str(EXP / "e9_extensions.py"), "--quick", "--workers", "4"])
        return
    if args.full:
        run([str(EXP / "e0_regime_sweep.py")])
        run([str(EXP / "e1_benchmarks.py"), "--full"])
        run([str(EXP / "compute_j50_reference.py")])
        for name in ("e2_learners", "e3_unknown_r", "e4_crossplay", "e5_team", "e6_data_amount", "e8_selection",
                     "e9_extensions"):
            run([str(EXP / f"{name}.py"), "--full", *w])
        run([str(EXP / "e7_safe.py"), "--full", "--s-res", "25.0", *w])
        run([str(EXP / "e10_equilibrium_band.py")])
        run([str(EXP / "e11_cooperation_threshold.py")])
        run([str(EXP / "e12_unsolved_r.py")])
    run([str(EXP / "make_tables.py")])
    run([str(EXP / "make_report_assets.py")])


if __name__ == "__main__":
    main()
