"""Write the catch-grid band from the refinement CSV. No numbers are typed here."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from band import write_band_tables  # noqa: E402


def main():
    csv_path, tex_path, band = write_band_tables(
        ROOT / "results" / "raw" / "e1_refinement.csv",
        ROOT / "results" / "tables",
    )
    print(f"Wrote {csv_path}")
    print(f"Wrote {tex_path}")
    for key, value in band.items():
        print(f"  {key}={value}")


if __name__ == "__main__":
    main()
