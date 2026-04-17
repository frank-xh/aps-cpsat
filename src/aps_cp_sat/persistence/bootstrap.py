from __future__ import annotations

from aps_cp_sat.persistence.db import get_engine, load_env
from aps_cp_sat.persistence.models import Base


def main() -> None:
    print("[APS][DB] bootstrap start")
    load_env()
    Base.metadata.create_all(get_engine())
    print("[APS][DB] bootstrap done")


if __name__ == "__main__":
    main()
