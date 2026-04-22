from __future__ import annotations

from pathlib import Path

import pandas as pd

from aps_cp_sat.config import PlannerConfig
from aps_cp_sat.preprocess.grade_catalog import MergedGradeRuleCatalog
from aps_cp_sat.preprocess.order_preparation import prepare_orders


def load_grade_catalog(steel_info_path: Path | None) -> MergedGradeRuleCatalog:
    if steel_info_path is None:
        return MergedGradeRuleCatalog(dict(MergedGradeRuleCatalog.RULES))
    return MergedGradeRuleCatalog.build(steel_info_path)


def load_orders(
    orders_path: Path,
    steel_info_path: Path | None,
    cfg: PlannerConfig,
    catalog: MergedGradeRuleCatalog,
) -> pd.DataFrame:
    return prepare_orders(orders_path, steel_info_path, cfg, grade_catalog=catalog)