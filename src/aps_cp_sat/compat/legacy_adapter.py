from __future__ import annotations

import warnings
from typing import Any

import pandas as pd


def run_legacy_schedule(*, orders_path: str, steel_info_path: str, output_path: str, config: Any, prepared_orders: pd.DataFrame | None = None):
    from aps_cp_sat.cold_rolling_scheduler import _build_schedule_legacy_impl

    warnings.warn(
        "legacy cold_rolling_scheduler.build_schedule is deprecated; use ColdRollingPipeline joint master path",
        DeprecationWarning,
        stacklevel=2,
    )
    return _build_schedule_legacy_impl(
        orders_path=orders_path,
        steel_info_path=steel_info_path,
        output_path=output_path,
        config=config,
        prepared_orders=prepared_orders,
        prepared_dropped=None,
        use_dual_master=True,
        enable_rebalance=True,
        engine_used="legacy_fallback",
    )
