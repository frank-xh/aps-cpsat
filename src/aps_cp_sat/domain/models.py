from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from aps_cp_sat.config import PlannerConfig


@dataclass(frozen=True)
class ColdRollingRequest:
    orders_path: Path
    steel_info_path: Path | None
    output_path: Path
    config: PlannerConfig


@dataclass(frozen=True)
class ColdRollingResult:
    schedule_df: pd.DataFrame
    rounds_df: pd.DataFrame
    output_path: Path
    dropped_df: pd.DataFrame | None = None
    engine_meta: dict[str, Any] | None = None
    config: PlannerConfig | None = None