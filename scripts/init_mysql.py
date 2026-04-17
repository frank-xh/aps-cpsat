import os
import sys
from pathlib import Path

import pymysql

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "src"))

from aps_cp_sat.persistence.db import MysqlConfig  # noqa: E402


def main() -> int:
    cfg = MysqlConfig.from_env()

    conn = pymysql.connect(
        host=cfg.host,
        port=cfg.port,
        user=cfg.user,
        password=cfg.password,
        charset="utf8mb4",
        autocommit=True,
    )
    try:
        with conn.cursor() as cur:
            cur.execute(f"CREATE DATABASE IF NOT EXISTS `{cfg.database}` DEFAULT CHARACTER SET utf8mb4")
            cur.execute(f"USE `{cfg.database}`")

            ddl_path = REPO_ROOT / "src" / "aps_cp_sat" / "persistence" / "ddl.sql"
            sql = ddl_path.read_text(encoding="utf-8")
            # Simple splitter (no procedures in ddl.sql)
            stmts = [s.strip() for s in sql.split(";") if s.strip()]
            for stmt in stmts:
                cur.execute(stmt)

        print(f"[analysis][mysql] initialized database={cfg.database}")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())

