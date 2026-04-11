from __future__ import annotations

from pathlib import Path

import pandas as pd

from aps_cp_sat.config import PlannerConfig
from aps_cp_sat.io.readers import load_grade_catalog, load_orders


def prepare_orders_for_model(orders_path: Path, steel_info_path: Path, cfg: PlannerConfig) -> pd.DataFrame:
    catalog = load_grade_catalog(steel_info_path)
    return load_orders(orders_path, steel_info_path, cfg, catalog)
