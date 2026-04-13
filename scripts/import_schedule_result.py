import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "src"))

from aps_cp_sat.persistence.service import persist_run_analysis_from_excel  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Import APS analysis Excel into local MySQL")
    parser.add_argument("xlsx", help="Path to APS exported analysis xlsx")
    parser.add_argument("--run-code", default=None, help="Override run_code (default: file stem)")
    args = parser.parse_args()

    xlsx_path = Path(args.xlsx).resolve()
    if not xlsx_path.exists():
        print(f"[analysis][import] file not found: {xlsx_path}")
        return 2

    run_id = persist_run_analysis_from_excel(xlsx_path, run_code=args.run_code)
    print(f"[analysis][import] ok run_id={run_id} xlsx={xlsx_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

