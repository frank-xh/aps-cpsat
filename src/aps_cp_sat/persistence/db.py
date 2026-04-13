from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker


def _repo_root() -> Path:
    # .../src/aps_cp_sat/persistence/db.py -> parents[3] is repo root
    return Path(__file__).resolve().parents[3]


def load_env() -> None:
    """Load `.env` from repo root if present (no override)."""
    load_dotenv(dotenv_path=_repo_root() / ".env", override=False)


@dataclass(frozen=True)
class MysqlConfig:
    host: str
    port: int
    user: str
    password: str
    database: str

    @staticmethod
    def from_env() -> "MysqlConfig":
        load_env()
        host = os.getenv("MYSQL_HOST", "127.0.0.1")
        port = int(os.getenv("MYSQL_PORT", "3306"))
        user = os.getenv("MYSQL_USER", "root")
        password = os.getenv("MYSQL_PASSWORD", "")
        database = os.getenv("MYSQL_DATABASE", "aps_schedule_analysis")
        return MysqlConfig(host=host, port=port, user=user, password=password, database=database)

    def sqlalchemy_url(self) -> str:
        # Password must not be hard-coded; sourced from env/.env.
        return (
            f"mysql+pymysql://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}?charset=utf8mb4"
        )


_ENGINE: Engine | None = None
_SessionLocal: sessionmaker[Session] | None = None


def get_engine() -> Engine:
    global _ENGINE, _SessionLocal
    if _ENGINE is None:
        cfg = MysqlConfig.from_env()
        _ENGINE = create_engine(cfg.sqlalchemy_url(), pool_pre_ping=True, future=True)
        _SessionLocal = sessionmaker(bind=_ENGINE, autocommit=False, autoflush=False, future=True)
    return _ENGINE


def get_sessionmaker() -> sessionmaker[Session]:
    global _SessionLocal
    if _SessionLocal is None:
        get_engine()
    assert _SessionLocal is not None
    return _SessionLocal


def session_scope() -> Iterator[Session]:
    """Context manager style generator for a transactional session."""
    SessionLocal = get_sessionmaker()
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

