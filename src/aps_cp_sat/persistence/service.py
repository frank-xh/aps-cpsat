from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from aps_cp_sat.persistence.db import session_scope
from aps_cp_sat.persistence.repository import (
    replace_bridge_drop_summary,
    replace_order_results,
    replace_slot_summaries,
    replace_transition_metrics,
    replace_violation_summary,
    upsert_run,
)


def _read_sheet(xlsx_path: Path, candidates: list[str]) -> pd.DataFrame | None:
    for name in candidates:
        try:
            return pd.read_excel(xlsx_path, sheet_name=name, engine="openpyxl")
        except Exception:
            continue
    return None


def _kv_to_dict(df: pd.DataFrame | None) -> dict[str, Any]:
    if df is None or df.empty:
        return {}
    cols = list(df.columns)
    if len(cols) < 2:
        return {}
    kcol, vcol = cols[0], cols[1]
    out: dict[str, Any] = {}
    for _, row in df.iterrows():
        k = row.get(kcol)
        if pd.isna(k):
            continue
        out[str(k)] = row.get(vcol)
    return out


def _isna(v: Any) -> bool:
    return v is None or (isinstance(v, float) and pd.isna(v))


def _coerce_int(v: Any) -> int | None:
    if _isna(v):
        return None
    try:
        return int(v)
    except Exception:
        return None


def _coerce_float(v: Any) -> float | None:
    if _isna(v):
        return None
    try:
        return float(v)
    except Exception:
        return None


def _coerce_bool(v: Any) -> bool | None:
    if _isna(v):
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(int(v))
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "y"}:
        return True
    if s in {"0", "false", "no", "n"}:
        return False
    return None


def _build_transition_metrics(schedule_df: pd.DataFrame | None) -> list[dict[str, Any]]:
    # Uses candidate_position/global_seq as a stable seq for charts.
    if schedule_df is None or schedule_df.empty:
        return []
    if not {"line", "slot_no", "order_id"}.issubset(set(schedule_df.columns)):
        return []

    df = schedule_df.copy()
    if "candidate_position" in df.columns:
        df["__seq"] = pd.to_numeric(df["candidate_position"], errors="coerce")
    elif "global_seq" in df.columns:
        df["__seq"] = pd.to_numeric(df["global_seq"], errors="coerce")
    else:
        df["__seq"] = range(1, len(df) + 1)

    rows: list[dict[str, Any]] = []
    for (line, slot_no), g in df.sort_values(["line", "slot_no", "__seq"]).groupby(["line", "slot_no"], sort=False):
        g = g.reset_index(drop=True)
        for i in range(len(g) - 1):
            a = g.iloc[i]
            b = g.iloc[i + 1]
            from_w = _coerce_int(a.get("width"))
            to_w = _coerce_int(b.get("width"))
            from_t = _coerce_float(a.get("thickness"))
            to_t = _coerce_float(b.get("thickness"))

            a_tmin = _coerce_float(a.get("temp_min"))
            a_tmax = _coerce_float(a.get("temp_max"))
            b_tmin = _coerce_float(b.get("temp_min"))
            b_tmax = _coerce_float(b.get("temp_max"))
            if a_tmin is not None and a_tmax is not None:
                from_temp_mid = 0.5 * (a_tmin + a_tmax)
            else:
                from_temp_mid = _coerce_float(a.get("temp_mean") or a.get("temp_mid"))
            if b_tmin is not None and b_tmax is not None:
                to_temp_mid = 0.5 * (b_tmin + b_tmax)
            else:
                to_temp_mid = _coerce_float(b.get("temp_mean") or b.get("temp_mid"))

            # overlap of two temp ranges (if available)
            temp_overlap = None
            if a_tmin is not None and a_tmax is not None and b_tmin is not None and b_tmax is not None:
                temp_overlap = max(0.0, min(a_tmax, b_tmax) - max(a_tmin, b_tmin))

            # simple proxy: absolute diff of mid points (if available)
            temp_jump = None
            if from_temp_mid is not None and to_temp_mid is not None:
                temp_jump = abs(to_temp_mid - from_temp_mid)

            rows.append(
                {
                    "line": str(line),
                    "slot_no": _coerce_int(slot_no) or 0,
                    "seq_no": i + 1,
                    "from_order_id": None if _isna(a.get("order_id")) else str(a.get("order_id")),
                    "to_order_id": None if _isna(b.get("order_id")) else str(b.get("order_id")),
                    "from_temp_mid": from_temp_mid,
                    "to_temp_mid": to_temp_mid,
                    "temp_overlap": temp_overlap,
                    "temp_jump": temp_jump,
                    "from_thickness": from_t,
                    "to_thickness": to_t,
                    "thickness_jump": None if from_t is None or to_t is None else abs(to_t - from_t),
                    "from_width": from_w,
                    "to_width": to_w,
                    "width_jump": None if from_w is None or to_w is None else abs(to_w - from_w),
                    "reverse_flag": None if from_w is None or to_w is None else (to_w < from_w),
                    "bridge_type": b.get("bridge_type") if "bridge_type" in g.columns else None,
                }
            )
    return rows


def persist_run_analysis_from_excel(xlsx_path: str | Path, run_code: str | None = None) -> int:
    """Import APS exported analysis Excel into local MySQL.

    - Missing sheets are tolerated (partial import).
    - Idempotent by run_code (default: file stem) or result_file_path.
    """
    xlsx_path = Path(xlsx_path).resolve()
    if run_code is None:
        run_code = xlsx_path.stem

    run_kv = _kv_to_dict(_read_sheet(xlsx_path, ["RUN_SUMMARY", "运行汇总"]))
    progress_df = _read_sheet(xlsx_path, ["PROGRESS_METRICS", "进展指标"])
    progress = {}
    if progress_df is not None and not progress_df.empty:
        progress = {k: progress_df.iloc[0].get(k) for k in progress_df.columns}

    is_analysis_excel = bool(run_kv) or (progress_df is not None and not progress_df.empty)

    run_fields: dict[str, Any] = {
        "profile_name": progress.get("profile") or run_kv.get("profile"),
        "acceptance": progress.get("acceptance") or run_kv.get("acceptance"),
        "failure_mode": progress.get("failure_mode") or run_kv.get("failure_mode"),
        "routing_feasible": _coerce_bool(progress.get("routing_feasible") or run_kv.get("routing_feasible")),
        "analysis_only": bool(is_analysis_excel),
        "official_exported": bool(not is_analysis_excel),
        "analysis_exported": bool(is_analysis_excel),
        "total_orders": _coerce_int(run_kv.get("total_orders")),
        "scheduled_orders": _coerce_int(progress.get("scheduled_orders") or run_kv.get("scheduled_orders")),
        "unscheduled_orders": _coerce_int(progress.get("unscheduled_orders") or run_kv.get("unscheduled_orders")),
        "dropped_orders": _coerce_int(progress.get("dropped_order_count") or run_kv.get("dropped_order_count")),
        "scheduled_tons": _coerce_float(progress.get("scheduled_tons") or run_kv.get("scheduled_tons")),
        "unscheduled_tons": _coerce_float(progress.get("unscheduled_tons") or run_kv.get("unscheduled_tons")),
        "template_build_seconds": _coerce_float(progress.get("template_build_seconds") or run_kv.get("template_build_seconds")),
        "joint_master_seconds": _coerce_float(progress.get("joint_master_seconds") or run_kv.get("joint_master_seconds")),
        "fallback_total_seconds": _coerce_float(progress.get("fallback_total_seconds") or run_kv.get("fallback_total_seconds")),
        "total_run_seconds": _coerce_float(progress.get("total_run_seconds") or run_kv.get("total_run_seconds")),
        "evidence_level": run_kv.get("evidence_level"),
        "best_candidate_available": _coerce_bool(run_kv.get("best_candidate_available") or progress.get("best_candidate_available")),
        "best_candidate_type": run_kv.get("best_candidate_type") or progress.get("best_candidate_type"),
        "best_candidate_objective": _coerce_float(run_kv.get("best_candidate_objective")),
        "best_candidate_unroutable_slot_count": _coerce_int(run_kv.get("best_candidate_unroutable_slot_count")),
    }

    # candidate schedules (preferred). Fall back to official schedules.
    big_df = _read_sheet(xlsx_path, ["BIG_ROLL_CANDIDATE", "大辊线候选排程", "大辊线排程"])
    small_df = _read_sheet(xlsx_path, ["SMALL_ROLL_CANDIDATE", "小辊线候选排程", "小辊线排程"])
    schedule_df = None
    if big_df is not None or small_df is not None:
        schedule_df = pd.concat([df for df in [big_df, small_df] if df is not None], ignore_index=True)

    # Normalize older official schedule export (Chinese columns) into analysis columns.
    if schedule_df is not None and not schedule_df.empty and "order_id" not in schedule_df.columns and "物料号" in schedule_df.columns:
        schedule_df = schedule_df.rename(
            columns={
                "物料号": "order_id",
                "产线": "line",
                "轧期编号": "slot_no",
                "轧期内序号": "candidate_position",
                "吨位": "tons",
                "宽度": "width",
                "厚度": "thickness",
                "温度下限": "temp_min",
                "温度上限": "temp_max",
                "钢种组": "steel_group",
            }
        )
        if "line" in schedule_df.columns:
            schedule_df["line"] = schedule_df["line"].map({"大辊线": "big_roll", "小辊线": "small_roll"}).fillna(schedule_df["line"])
        schedule_df["drop_flag"] = False
        schedule_df["slot_unroutable_flag"] = False
        schedule_df["analysis_only"] = False
        schedule_df["candidate_status"] = "OFFICIAL_SCHEDULED"

    order_rows: list[dict[str, Any]] = []
    if schedule_df is not None and not schedule_df.empty:
        # Fill run-level scheduled metrics if missing.
        if run_fields.get("scheduled_orders") is None:
            run_fields["scheduled_orders"] = int(len(schedule_df))
        if run_fields.get("total_orders") is None:
            run_fields["total_orders"] = int(len(schedule_df))
        if run_fields.get("scheduled_tons") is None and "tons" in schedule_df.columns:
            run_fields["scheduled_tons"] = float(pd.to_numeric(schedule_df["tons"], errors="coerce").fillna(0.0).sum())

        for _, r in schedule_df.iterrows():
            order_rows.append(
                {
                    "order_id": None if _isna(r.get("order_id")) else str(r.get("order_id")),
                    "line": r.get("line"),
                    "slot_no": _coerce_int(r.get("slot_no")),
                    "candidate_position": _coerce_int(r.get("candidate_position") or r.get("candidate_slot_member_index")),
                    "tons": _coerce_float(r.get("tons")),
                    "width": _coerce_int(r.get("width")),
                    "thickness": _coerce_float(r.get("thickness")),
                    "temp_min": _coerce_float(r.get("temp_min")),
                    "temp_max": _coerce_float(r.get("temp_max")),
                    "steel_group": r.get("steel_group"),
                    "line_capability": r.get("line_capability"),
                    "candidate_status": r.get("candidate_status"),
                    "drop_flag": _coerce_bool(r.get("drop_flag") or r.get("dropped")),
                    "dominant_drop_reason": r.get("dominant_drop_reason"),
                    "secondary_reasons": r.get("secondary_reasons"),
                    "slot_unroutable_flag": _coerce_bool(r.get("slot_unroutable_flag")),
                    "slot_route_risk_score": _coerce_float(r.get("slot_route_risk_score")),
                    "analysis_only": _coerce_bool(r.get("analysis_only")) if "analysis_only" in schedule_df.columns else bool(is_analysis_excel),
                }
            )

    # slot summaries
    slot_df = _read_sheet(xlsx_path, ["SLOT_SUMMARY", "槽位汇总"])
    unroutable_df = _read_sheet(xlsx_path, ["UNROUTABLE_SLOTS", "不可路由槽位"])
    slot_rows: dict[tuple[str, int], dict[str, Any]] = {}

    if slot_df is not None and not slot_df.empty:
        for _, r in slot_df.iterrows():
            line = str(r.get("line"))
            slot_no = _coerce_int(r.get("slot_no")) or 0
            slot_rows[(line, slot_no)] = {
                "line": line,
                "slot_no": slot_no,
                "slot_order_count": _coerce_int(r.get("slot_order_count")),
                "slot_tons": _coerce_float(r.get("slot_tons")),
                "order_count_over_cap": _coerce_int(r.get("order_count_over_cap")),
                "slot_route_risk_score": _coerce_float(r.get("slot_route_risk_score")),
                "pair_gap_proxy": _coerce_float(r.get("pair_gap_proxy")),
                "span_risk": _coerce_float(r.get("span_risk")),
                "degree_risk": _coerce_float(r.get("degree_risk")),
                "isolated_order_penalty": _coerce_float(r.get("isolated_order_penalty")),
                "dominant_unroutable_reason": r.get("dominant_unroutable_reason"),
            }

    if unroutable_df is not None and not unroutable_df.empty:
        for _, r in unroutable_df.iterrows():
            line = str(r.get("line"))
            slot_no = _coerce_int(r.get("slot_no") or r.get("slot")) or 0
            key = (line, slot_no)
            base = slot_rows.get(key, {"line": line, "slot_no": slot_no})
            base.update(
                {
                    "template_coverage_ratio": _coerce_float(r.get("template_coverage_ratio")),
                    "missing_template_edge_count": _coerce_int(r.get("missing_template_edge_count")),
                    "zero_in_orders": _coerce_int(r.get("zero_in_orders")),
                    "zero_out_orders": _coerce_int(r.get("zero_out_orders")),
                    "width_span": _coerce_int(r.get("width_span")),
                    "thickness_span": _coerce_float(r.get("thickness_span")),
                    "steel_group_count": _coerce_int(r.get("steel_group_count")),
                    "slot_route_risk_score": _coerce_float(r.get("slot_route_risk_score")) or base.get("slot_route_risk_score"),
                    "dominant_unroutable_reason": r.get("dominant_unroutable_reason") or base.get("dominant_unroutable_reason"),
                }
            )
            slot_rows[key] = base

    # If SLOT_SUMMARY/UNROUTABLE_SLOTS sheets are missing (older official exports),
    # fall back to a minimal slot summary computed from schedule_df.
    if not slot_rows and schedule_df is not None and not schedule_df.empty and {"line", "slot_no"}.issubset(set(schedule_df.columns)):
        df2 = schedule_df.copy()
        df2["tons"] = pd.to_numeric(df2.get("tons"), errors="coerce")
        for (line, slot_no), g in df2.groupby(["line", "slot_no"], sort=False):
            widths = pd.to_numeric(g.get("width"), errors="coerce") if "width" in g.columns else None
            thks = pd.to_numeric(g.get("thickness"), errors="coerce") if "thickness" in g.columns else None
            slot_rows[(str(line), int(slot_no))] = {
                "line": str(line),
                "slot_no": int(slot_no),
                "slot_order_count": int(len(g)),
                "slot_tons": float(g["tons"].fillna(0.0).sum()) if "tons" in g.columns else None,
                "order_count_over_cap": None,
                "width_span": None if widths is None else _coerce_int(widths.max() - widths.min()),
                "thickness_span": None if thks is None else _coerce_float(thks.max() - thks.min()),
                "steel_group_count": int(g["steel_group"].nunique()) if "steel_group" in g.columns else None,
                "dominant_unroutable_reason": None,
            }

    # candidate violation summary (metric, candidate_value)
    vio_df = _read_sheet(xlsx_path, ["VIOLATION_SUMMARY_CANDIDATE", "VIOLATION_SUMMARY", "违规汇总_候选", "违规汇总"])
    vio_fields: dict[str, Any] = {}
    if vio_df is not None and not vio_df.empty and "metric" in vio_df.columns:
        if "candidate_value" in vio_df.columns:
            m = {str(r["metric"]): r.get("candidate_value") for _, r in vio_df.iterrows() if not _isna(r.get("metric"))}
            vio_fields = {
                "candidate_unroutable_slot_count": _coerce_int(m.get("candidate_unroutable_slot_count")),
                "candidate_bad_slot_count": _coerce_int(m.get("candidate_bad_slot_count")),
                "candidate_zero_in_order_count": _coerce_int(m.get("candidate_zero_in_order_count")),
                "candidate_zero_out_order_count": _coerce_int(m.get("candidate_zero_out_order_count")),
                "direct_reverse_step_violation_count": _coerce_int(m.get("candidate_direct_reverse_step_violation_count")),
                "virtual_attach_reverse_violation_count": _coerce_int(m.get("candidate_virtual_attach_reverse_violation_count")),
                "period_reverse_count_violation_count": _coerce_int(m.get("candidate_period_reverse_count_violation_count")),
                "bridge_count_violation_count": _coerce_int(m.get("candidate_bridge_count_violation_count")),
                "invalid_virtual_spec_count": _coerce_int(m.get("candidate_invalid_virtual_spec_count")),
            }
        elif "value" in vio_df.columns:
            vio_kv = _kv_to_dict(vio_df)
            vio_fields = {
                "direct_reverse_step_violation_count": _coerce_int(vio_kv.get("direct_reverse_step_violation_count")),
                "virtual_attach_reverse_violation_count": _coerce_int(vio_kv.get("virtual_attach_reverse_violation_count")),
                "period_reverse_count_violation_count": _coerce_int(vio_kv.get("period_reverse_count_violation_count")),
                "bridge_count_violation_count": _coerce_int(vio_kv.get("bridge_count_violation_count")),
                "invalid_virtual_spec_count": _coerce_int(vio_kv.get("invalid_virtual_spec_count")),
                "candidate_unroutable_slot_count": _coerce_int(vio_kv.get("candidate_unroutable_slot_count")),
                "candidate_bad_slot_count": _coerce_int(vio_kv.get("candidate_bad_slot_count")),
                "candidate_zero_in_order_count": _coerce_int(vio_kv.get("candidate_zero_in_order_count")),
                "candidate_zero_out_order_count": _coerce_int(vio_kv.get("candidate_zero_out_order_count")),
            }

    bridge_kv = _kv_to_dict(_read_sheet(xlsx_path, ["BRIDGE_SUMMARY", "桥接汇总"]))
    drop_bridge_kv = _kv_to_dict(_read_sheet(xlsx_path, ["DROP_AND_BRIDGE_SUMMARY", "剔除与桥接汇总"]))
    unscheduled_kv = _kv_to_dict(_read_sheet(xlsx_path, ["UNSCHEDULED_SUMMARY", "未排汇总"]))
    hist = drop_bridge_kv.get("dropped_reason_histogram") or unscheduled_kv.get("dropped_reason_histogram")
    hist_json = None
    if isinstance(hist, str) and hist.strip():
        try:
            hist_json = json.loads(hist)
        except Exception:
            hist_json = {"raw": hist}
    elif isinstance(hist, dict):
        hist_json = hist

    bridge_drop_fields: dict[str, Any] = {
        "selected_direct_edge_count": _coerce_int(bridge_kv.get("selected_direct_edge_count") or drop_bridge_kv.get("selected_direct_edge_count")),
        "selected_real_bridge_edge_count": _coerce_int(bridge_kv.get("selected_real_bridge_edge_count") or drop_bridge_kv.get("selected_real_bridge_edge_count")),
        "selected_virtual_bridge_edge_count": _coerce_int(
            bridge_kv.get("selected_virtual_bridge_edge_count") or drop_bridge_kv.get("selected_virtual_bridge_edge_count")
        ),
        "direct_edge_ratio": _coerce_float(bridge_kv.get("direct_edge_ratio")),
        "real_bridge_ratio": _coerce_float(bridge_kv.get("real_bridge_ratio")),
        "virtual_bridge_ratio": _coerce_float(bridge_kv.get("virtual_bridge_ratio")),
        "max_bridge_count_used": _coerce_int(bridge_kv.get("max_bridge_count_used") or drop_bridge_kv.get("max_bridge_count_used")),
        "avg_virtual_count": _coerce_float(bridge_kv.get("avg_virtual_count")),
        "dropped_reason_histogram_json": hist_json,
        "dropped_order_count": _coerce_int(drop_bridge_kv.get("dropped_order_count") or progress.get("dropped_order_count")),
        "dropped_tons": _coerce_float(progress.get("dropped_tons")),
    }

    transition_rows = _build_transition_metrics(schedule_df)

    with session_scope() as session:
        run = upsert_run(session, run_code=run_code, result_file_path=str(xlsx_path), fields=run_fields)

        if order_rows:
            replace_order_results(session, run_id=run.id, rows=order_rows)
        if slot_rows:
            replace_slot_summaries(session, run_id=run.id, rows=slot_rows.values())
        if vio_fields:
            replace_violation_summary(session, run_id=run.id, fields=vio_fields)
        if bridge_drop_fields:
            replace_bridge_drop_summary(session, run_id=run.id, fields=bridge_drop_fields)
        if transition_rows:
            replace_transition_metrics(session, run_id=run.id, rows=transition_rows)

        return int(run.id)
