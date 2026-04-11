from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

from aps_cp_sat.config import PlannerConfig
from aps_cp_sat.preprocess.grade_catalog import MergedGradeRuleCatalog
from aps_cp_sat.transition.bridge_rules import DUE_RANK


def _txt(v, default: str = "") -> str:
    if v is None:
        return default
    if isinstance(v, float) and pd.isna(v):
        return default
    return str(v).strip()


def _col_by_name_or_index(df: pd.DataFrame, names: List[str], idx: int) -> str:
    cols = [str(c) for c in df.columns]
    for n in names:
        for c in cols:
            if n.lower() in c.lower():
                return c
    if idx < len(cols):
        return cols[idx]
    raise ValueError(f"Cannot resolve column: {names} / {idx}")


def _col_exact_or_fallback(df: pd.DataFrame, exact_names: List[str], fuzzy_names: List[str], idx: int) -> str:
    cols = [str(c) for c in df.columns]
    for n in exact_names:
        if n in cols:
            return n
    return _col_by_name_or_index(df, fuzzy_names, idx)


def _normalize_order_id(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if float(v).is_integer():
            return str(int(v))
        return str(v)
    s = str(v).strip()
    if "." in s:
        left, right = s.split(".", 1)
        if left.isdigit() and right.isdigit() and set(right) <= {"0"}:
            return left
    return s


def _roll_capability(raw: str) -> str:
    t = _txt(raw).lower()
    has_big = ("大" in t) or ("big" in t)
    has_small = ("小" in t) or ("small" in t)
    if has_big and has_small:
        return "dual"
    if has_big:
        return "big_only"
    if has_small:
        return "small_only"
    return "dual"


def _merge_roll_capability(source_cap: str, spec_cap: str) -> str:
    if source_cap in {"big_only", "small_only", "dual"}:
        return source_cap
    return spec_cap if spec_cap in {"big_only", "small_only", "dual"} else "dual"


def _due_bucket(v) -> str:
    d = pd.to_datetime(v, errors="coerce")
    if pd.isna(d):
        return "slack"
    delta = (d.normalize() - pd.Timestamp.now().normalize()).days
    if delta < 0:
        return "overdue"
    if delta <= 3:
        return "urgent"
    if delta <= 14:
        return "normal"
    return "slack"


def _proc_hours(tons: float, thickness: float, line: str) -> float:
    base_tph = 95.0 if line == "small_roll" else 120.0
    if thickness < 0.7:
        base_tph *= 0.82
    elif thickness < 1.0:
        base_tph *= 0.90
    return float(tons / max(30.0, base_tph))


def prepare_orders(
    orders_path: Path,
    steel_info_path: Path,
    cfg: PlannerConfig,
    grade_catalog: MergedGradeRuleCatalog | None = None,
) -> pd.DataFrame:
    df = pd.read_excel(orders_path, sheet_name=0).copy()
    grade_catalog = grade_catalog or MergedGradeRuleCatalog.build(steel_info_path)

    col_id = _col_exact_or_fallback(df, ["合同完全号", "物料号", "订单号"], ["合同", "物料", "订单号"], 1)
    col_due = _col_by_name_or_index(df, ["交货", "due"], 4)
    col_grade = _col_by_name_or_index(df, ["牌号", "grade"], 5)
    col_backlog = _col_by_name_or_index(df, ["欠交", "backlog"], 9)
    col_group = _col_by_name_or_index(df, ["Steel Grade Class", "钢种", "group"], 11)
    col_width = _col_exact_or_fallback(df, ["宽度"], ["宽度", "width"], 15)
    col_thick = _col_exact_or_fallback(df, ["厚度"], ["厚度", "thickness"], 16)
    col_roll = _col_by_name_or_index(df, ["辊", "roll"], 30)
    col_temp_max = _col_exact_or_fallback(df, ["均热段温度最大值"], ["温度最大", "temp max"], 28)
    col_temp_min = _col_exact_or_fallback(df, ["均热段温度最小值"], ["温度最小", "temp min"], 29)
    col_tons = df.columns[33] if len(df.columns) > 33 else _col_by_name_or_index(df, ["吨", "重量", "weight"], 33)

    base = pd.DataFrame(
        {
            "source_order_id": df[col_id].map(_normalize_order_id),
            "grade": df[col_grade].astype(str),
            "steel_group_raw": df[col_group].astype(str),
            "width": pd.to_numeric(df[col_width], errors="coerce"),
            "thickness": pd.to_numeric(df[col_thick], errors="coerce"),
            "temp_min": pd.to_numeric(df[col_temp_min], errors="coerce"),
            "temp_max": pd.to_numeric(df[col_temp_max], errors="coerce"),
            "tons": pd.to_numeric(df[col_tons], errors="coerce"),
            "backlog": pd.to_numeric(df[col_backlog], errors="coerce"),
            "due_date": pd.to_datetime(df[col_due], errors="coerce"),
            "roll_type": df[col_roll].astype(str),
        }
    )
    base = base.dropna(subset=["width", "thickness", "temp_min", "temp_max", "tons", "backlog"]).copy()
    base["temp_min"], base["temp_max"] = (
        base[["temp_min", "temp_max"]].min(axis=1),
        base[["temp_min", "temp_max"]].max(axis=1),
    )
    base = base[base["backlog"] > 0].copy()

    rows: List[dict] = []
    for _, row in base.iterrows():
        src_id = _normalize_order_id(row["source_order_id"])
        grade = _txt(row["grade"]).upper()
        source_cap = _roll_capability(_txt(row["roll_type"]))
        rule = grade_catalog.get(grade)
        steel_group = _txt(rule.get("steel_group", "UNKNOWN")).upper()
        line_capability = _merge_roll_capability(source_cap, _txt(rule.get("roll_capability", "dual")))
        priority = int(rule.get("priority", 1))
        rows.append(
            {
                "order_id": src_id,
                "source_order_id": src_id,
                "parent_order_id": src_id,
                "lot_id": src_id,
                "grade": grade,
                "steel_group": steel_group,
                "steel_group_raw": _txt(row.get("steel_group_raw", "")),
                "width": float(row["width"]),
                "thickness": float(row["thickness"]),
                "temp_min": float(row["temp_min"]),
                "temp_max": float(row["temp_max"]),
                "temp_mean": (float(row["temp_min"]) + float(row["temp_max"])) / 2.0,
                "tons": float(row["tons"]),
                "backlog": float(row["backlog"]),
                "due_date": row["due_date"],
                "roll_type": _txt(row["roll_type"]),
                "line_capability": line_capability,
                "priority": priority,
            }
        )

    out = pd.DataFrame(rows)
    out["due_bucket"] = out["due_date"].map(_due_bucket)
    out["due_rank"] = out["due_bucket"].map(DUE_RANK).fillna(3).astype(int)
    out["line"] = out["line_capability"].map({"big_only": "big_roll", "small_only": "small_roll", "dual": ""}).fillna("")
    out["line_source"] = out["line"].map({"big_roll": "big_only", "small_roll": "small_only", "": "dual"}).fillna("dual")
    out = out.sort_values(["priority", "due_rank", "backlog", "tons"], ascending=[False, True, False, False]).head(cfg.max_orders).reset_index(drop=True)
    out["proc_hours_big"] = out.apply(lambda r: _proc_hours(float(r["tons"]), float(r["thickness"]), "big_roll"), axis=1)
    out["proc_hours_small"] = out.apply(lambda r: _proc_hours(float(r["tons"]), float(r["thickness"]), "small_roll"), axis=1)
    out["proc_hours"] = 0.0
    return out


__all__ = ["MergedGradeRuleCatalog", "prepare_orders"]
