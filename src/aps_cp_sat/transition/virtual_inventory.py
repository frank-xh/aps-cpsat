from __future__ import annotations

from collections import Counter
from typing import Any

import pandas as pd


def _mode_enabled(cfg: Any) -> bool:
    model = getattr(cfg, "model", cfg)
    mode = str(getattr(model, "virtual_bridge_mode", "template_bridge") or "template_bridge")
    return bool(
        mode == "prebuilt_virtual_inventory"
        and getattr(model, "prebuilt_virtual_inventory_enabled", False)
    )


def _target_capabilities(model: Any) -> list[str]:
    caps: list[str] = []
    if bool(getattr(model, "prebuilt_virtual_generate_for_big_roll", True)):
        caps.append("big_only")
    if bool(getattr(model, "prebuilt_virtual_generate_for_small_roll", True)):
        caps.append("small_only")
    return caps or ["dual"]


def _sanitize_widths(model: Any) -> list[float]:
    raw = list(getattr(model, "prebuilt_virtual_widths", [1000, 1250, 1500]) or [])
    cleaned: list[float] = []
    seen: set[float] = set()
    for value in raw:
        try:
            width = float(value)
        except Exception:
            continue
        if width <= 0:
            continue
        if width in seen:
            continue
        seen.add(width)
        cleaned.append(width)
    return cleaned or [1000.0, 1250.0, 1500.0]


def _sanitize_thicknesses(model: Any) -> list[float]:
    raw = list(getattr(model, "prebuilt_virtual_thicknesses", [0.6, 0.8, 1.0, 1.2, 1.5, 2.0]) or [])
    cleaned: list[float] = []
    seen: set[float] = set()
    for value in raw:
        try:
            thk = float(value)
        except Exception:
            continue
        if thk <= 0:
            continue
        if thk in seen:
            continue
        seen.add(thk)
        cleaned.append(thk)
    return cleaned or [0.6, 0.8, 1.0, 1.2, 1.5, 2.0]


def build_prebuilt_virtual_inventory(cfg: Any, orders_df: pd.DataFrame) -> pd.DataFrame:
    """Build pre-generated virtual bridge inventory for graph construction only.

    IMPORTANT:
    - Width / thickness specs are kept exactly as configured.
    - No dynamic width re-anchoring is performed.
    - Both production lines receive their own finite inventory pool.
    """
    model = getattr(cfg, "model", cfg)
    if not _mode_enabled(cfg):
        return pd.DataFrame()

    widths = _sanitize_widths(model)
    thicknesses = _sanitize_thicknesses(model)
    count_per_spec = max(0, int(getattr(model, "prebuilt_virtual_count_per_spec", 5) or 0))
    temp_min = float(getattr(model, "prebuilt_virtual_temp_min", 600.0) or 600.0)
    temp_max = float(getattr(model, "prebuilt_virtual_temp_max", 900.0) or 900.0)
    if temp_min > temp_max:
        temp_min, temp_max = temp_max, temp_min
    steel_group = str(getattr(model, "prebuilt_virtual_group", "普碳") or "普碳")
    tons = float(
        getattr(model, "prebuilt_virtual_default_tons", None)
        or getattr(getattr(cfg, "rule", None), "virtual_tons", 20.0)
        or 20.0
    )
    target_caps = _target_capabilities(model)

    rows: list[dict[str, Any]] = []
    for target_cap in target_caps:
        line = {"big_only": "big_roll", "small_only": "small_roll", "dual": ""}.get(target_cap, "")
        for width in widths:
            for thickness in thicknesses:
                spec_key = f"{line or 'dual'}|W{int(width)}|T{float(thickness):.2f}"
                for idx in range(1, count_per_spec + 1):
                    virtual_id = (
                        f"VIRTUAL_PREBUILT__{line or 'dual'}__W{int(width)}"
                        f"__T{float(thickness):.2f}__{idx:02d}"
                    )
                    rows.append(
                        {
                            "order_id": virtual_id,
                            "source_order_id": virtual_id,
                            "parent_order_id": virtual_id,
                            "lot_id": virtual_id,
                            "virtual_id": virtual_id,
                            "grade": "VIRTUAL_PREBUILT",
                            "steel_group": steel_group,
                            "steel_group_raw": steel_group,
                            "width": float(width),
                            "thickness": float(thickness),
                            "temp_min": float(temp_min),
                            "temp_max": float(temp_max),
                            "temp_mean": (float(temp_min) + float(temp_max)) / 2.0,
                            "tons": float(tons),
                            "backlog": float(tons),
                            "due_date": pd.NaT,
                            "roll_type": "virtual",
                            "line_capability": target_cap,
                            "priority": 0,
                            "due_bucket": "virtual",
                            "due_rank": 99,
                            "line": line,
                            "line_source": target_cap,
                            "proc_hours_big": 0.0,
                            "proc_hours_small": 0.0,
                            "proc_hours": 0.0,
                            "is_virtual": True,
                            "virtual_origin": "prebuilt_inventory",
                            "virtual_inventory_mode": True,
                            "virtual_usage_type": "bridge",
                            "virtual_spec_key": spec_key,
                            "prebuilt_spec_key": spec_key,
                            "virtual_inventory_count_index": int(idx),
                            "inventory_count_index": int(idx),
                            "inventory_index": int(idx),
                            "can_be_campaign_seed": False,
                            "bridge_resource_only": True,
                        }
                    )
    return pd.DataFrame(rows)


def prebuilt_virtual_inventory_diagnostics(inventory_df: pd.DataFrame, cfg: Any) -> dict[str, Any]:
    model = getattr(cfg, "model", cfg)
    if not isinstance(inventory_df, pd.DataFrame) or inventory_df.empty:
        return {
            "virtual_bridge_mode": str(getattr(model, "virtual_bridge_mode", "template_bridge") or "template_bridge"),
            "prebuilt_virtual_inventory_enabled": bool(_mode_enabled(cfg)),
            "prebuilt_virtual_inventory_count": 0,
            "virtual_inventory_count_total": 0,
            "prebuilt_virtual_specs_count": 0,
            "prebuilt_virtual_big_roll_count": 0,
            "prebuilt_virtual_small_roll_count": 0,
            "virtual_inventory_big_roll_count": 0,
            "virtual_inventory_small_roll_count": 0,
            "virtual_inventory_remaining_count": 0,
            "virtual_inventory_consumed_count": 0,
            "prebuilt_virtual_line_capability_breakdown": {},
        }
    big_count = int((inventory_df["line_capability"] == "big_only").sum()) if "line_capability" in inventory_df.columns else 0
    small_count = int((inventory_df["line_capability"] == "small_only").sum()) if "line_capability" in inventory_df.columns else 0
    return {
        "virtual_bridge_mode": str(getattr(model, "virtual_bridge_mode", "template_bridge") or "template_bridge"),
        "prebuilt_virtual_inventory_enabled": bool(_mode_enabled(cfg)),
        "prebuilt_virtual_inventory_count": int(len(inventory_df)),
        "virtual_inventory_count_total": int(len(inventory_df)),
        "prebuilt_virtual_specs_count": int(inventory_df["virtual_spec_key"].nunique()) if "virtual_spec_key" in inventory_df.columns else 0,
        "prebuilt_virtual_big_roll_count": int(big_count),
        "prebuilt_virtual_small_roll_count": int(small_count),
        "virtual_inventory_big_roll_count": int(big_count),
        "virtual_inventory_small_roll_count": int(small_count),
        "virtual_inventory_remaining_count": int(len(inventory_df)),
        "virtual_inventory_consumed_count": 0,
        "prebuilt_virtual_line_capability_breakdown": inventory_df["line_capability"].value_counts(dropna=False).to_dict() if "line_capability" in inventory_df.columns else {},
    }
