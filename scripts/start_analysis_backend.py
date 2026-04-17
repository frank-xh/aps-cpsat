import os
import sys
from pathlib import Path

import uvicorn

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "src"))

from aps_cp_sat.persistence.db import load_env  # noqa: E402


def main() -> int:
    load_env()
    host = os.getenv("BACKEND_HOST", "127.0.0.1")
    port = int(os.getenv("BACKEND_PORT", "18080"))
    print(f"[analysis][backend] http://{host}:{port}")
    uvicorn.run("aps_cp_sat.api.main:app", host=host, port=port, reload=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

