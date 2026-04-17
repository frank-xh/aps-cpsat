import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
WEBAPP_DIR = REPO_ROOT / "webapp"


def main() -> int:
    if not WEBAPP_DIR.exists():
        print(f"[analysis][frontend] webapp directory not found: {WEBAPP_DIR}")
        return 2

    # Install once (safe if already installed)
    subprocess.check_call(["npm", "install"], cwd=str(WEBAPP_DIR))
    subprocess.check_call(["npm", "run", "dev"], cwd=str(WEBAPP_DIR))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

