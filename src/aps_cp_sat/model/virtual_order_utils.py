from __future__ import annotations

from typing import Any

import pandas as pd


VIRTUAL_PREBUILT_PREFIX = "VIRTUAL_PREBUILT__"


def _safe_str(value: object) -> str:
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value or "")


def is_effective_virtual_order(order_row_or_id: Any) -> bool:
    if isinstance(order_row_or_id, str):
        return order_row_or_id.startswith(VIRTUAL_PREBUILT_PREFIX)

    if isinstance(order_row_or_id, pd.Series):
        rec = order_row_or_id.to_dict()
    elif isinstance(order_row_or_id, dict):
        rec = order_row_or_id
    else:
        rec = {}

    raw = rec.get("is_virtual", False)
    try:
        if pd.isna(raw):
            raw = False
    except Exception:
        pass
    if bool(raw):
        return True
    if _safe_str(rec.get("virtual_origin")) == "prebuilt_inventory":
        return True
    oid = _safe_str(rec.get("order_id") or rec.get("source_order_id") or rec.get("parent_order_id"))
    return oid.startswith(VIRTUAL_PREBUILT_PREFIX)


def effective_virtual_mask(df: pd.DataFrame) -> pd.Series:
    if df is None or df.empty:
        return pd.Series([], dtype=bool)
    if "is_virtual" in df.columns and "virtual_origin" not in df.columns and "order_id" not in df.columns:
        return df["is_virtual"].fillna(False).astype(bool)
    return df.apply(is_effective_virtual_order, axis=1)


def normalize_effective_virtual_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        if "is_virtual" not in out.columns:
            out["is_virtual"] = pd.Series(dtype=bool)
        return out
    out["is_virtual"] = effective_virtual_mask(out).fillna(False).astype(bool)
    return out


def count_effective_virtual_rows(df: pd.DataFrame) -> int:
    if df is None or df.empty:
        return 0
    return int(effective_virtual_mask(df).fillna(False).astype(bool).sum())
