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


def _infer_capability_for_width(orders_df: pd.DataFrame, width: float) -> str:
    """Infer capability from nearby real orders instead of duplicating line rules."""
    if not isinstance(orders_df, pd.DataFrame) or orders_df.empty or "line_capability" not in orders_df.columns:
        return "dual"
    work = orders_df.copy()
    if "is_virtual" in work.columns:
        work = work[~work["is_virtual"].fillna(False).astype(bool)]
    if work.empty or "width" not in work.columns:
        return "dual"
    width_num = pd.to_numeric(work["width"], errors="coerce")
    nearby = work[(width_num - float(width)).abs() <= 100.0]
    source = nearby if not nearby.empty else work
    caps = [str(v or "dual") for v in source["line_capability"].tolist()]
    if not caps:
        return "dual"
    return str(Counter(caps).most_common(1)[0][0] or "dual")


def _target_capabilities(model: Any) -> list[str]:
    caps: list[str] = []
    if bool(getattr(model, "prebuilt_virtual_generate_for_big_roll", True)):
        caps.append("big_only")
    if bool(getattr(model, "prebuilt_virtual_generate_for_small_roll", True)):
        caps.append("small_only")
    return caps or ["dual"]


def build_prebuilt_virtual_inventory(cfg: Any, orders_df: pd.DataFrame) -> pd.DataFrame:
    """Build pre-generated virtual bridge inventory for graph construction only."""
    model = getattr(cfg, "model", cfg)
    if not _mode_enabled(cfg):
        return pd.DataFrame()

    widths = list(getattr(model, "prebuilt_virtual_widths", [1000, 1250, 1500]) or [])
    thicknesses = list(getattr(model, "prebuilt_virtual_thicknesses", [0.6, 0.8, 1.0, 1.2, 1.5, 2.0]) or [])
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
    use_cap_rules = bool(getattr(model, "prebuilt_virtual_use_same_line_capability_rules", True))
    target_caps = _target_capabilities(model)

    rows: list[dict[str, Any]] = []
    for width in widths:
        for thickness in thicknesses:
            for target_cap in target_caps:
                # Keep the same capability labels as real orders; the target
                # line switches decide which finite inventory pool receives the spec.
                inferred_cap = _infer_capability_for_width(orders_df, float(width)) if use_cap_rules else target_cap
                cap = target_cap if target_cap in {"big_only", "small_only"} else inferred_cap
                line = {"big_only": "big_roll", "small_only": "small_roll", "dual": ""}.get(cap, "")
                spec_key = f"{line or 'dual'}|W{int(float(width))}|T{float(thickness):.2f}"
                for idx in range(1, count_per_spec + 1):
                    virtual_id = f"VIRTUAL_PREBUILT__{line or 'dual'}__W{int(float(width))}__T{float(thickness):.2f}__{idx:02d}"
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
                            "line_capability": cap,
                            "priority": 0,
                            "due_bucket": "virtual",
                            "due_rank": 99,
                            "line": line,
                            "line_source": cap,
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
        "prebuilt_virtual_line_capability_breakdown": inventory_df["line_capability"].value_counts(dropna=False).to_dict()
        if "line_capability" in inventory_df.columns
        else {},
    }
