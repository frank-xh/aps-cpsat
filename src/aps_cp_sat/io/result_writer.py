from __future__ import annotations

"""
Export layer contract.

This module only renders already-decoded schedules into files. It must not
change plan semantics, add bridge rows, or repair infeasibility.
"""

from pathlib import Path
from time import perf_counter
from typing import Dict, Iterable, Tuple

import pandas as pd
from openpyxl.utils import get_column_letter

from aps_cp_sat.config.rule_config import RuleConfig
from aps_cp_sat.rules import RULE_REGISTRY, RuleKey


EXPORT_RULE_KEYS = [
    RuleKey.LINE_COMPATIBILITY,
    RuleKey.TEMPERATURE_OVERLAP,
    RuleKey.WIDTH_TRANSITION,
    RuleKey.THICKNESS_TRANSITION,
    RuleKey.CROSS_GROUP_BRIDGE,
    RuleKey.CAMPAIGN_TON_MAX,
    RuleKey.CAMPAIGN_TON_MIN,
    RuleKey.REVERSE_WIDTH_COUNT,
    RuleKey.REVERSE_WIDTH_TOTAL,
    RuleKey.VIRTUAL_USAGE,
    RuleKey.VIRTUAL_RATIO,
]


_LINE_ZH = {"big_roll": "大辊线", "small_roll": "小辊线"}
_EDGE_TYPE_ZH = {"DIRECT_EDGE": "直接相邻", "REAL_BRIDGE_EDGE": "实物桥接", "VIRTUAL_BRIDGE_EDGE": "虚拟桥接"}
_CANDIDATE_STATUS_ZH = {
    "ASSIGNED_CANDIDATE": "候选已分配",
    "DROPPED_CANDIDATE": "候选已剔除",
    "UNROUTABLE_SLOT_MEMBER": "不可路由槽成员",
}
_SUMMARY_MODE_ZH = {"OFFICIAL": "正式口径", "CANDIDATE_ANALYSIS": "候选口径"}
_RESULT_USAGE_ZH = {"OFFICIAL": "正式", "PARTIAL_OFFICIAL": "部分正式", "ANALYSIS_ONLY": "仅分析", "NOT_EXPORTED": "未导出"}
_ACCEPTANCE_ZH = {
    "OFFICIAL_FULL_SCHEDULE": "正式全排结果",
    "PARTIAL_SCHEDULE_WITH_DROPS": "部分可接受(含剔除)",
    "BEST_SEARCH_CANDIDATE_ANALYSIS": "最佳搜索候选(仅分析)",
}
_FAILURE_MODE_ZH = {
    "FAILED_ROUTING_SEARCH": "路由搜索失败",
    "FAILED_STRONG_INFEASIBILITY_SIGNAL": "强不可行信号",
    "FAILED_TIME_BUDGET": "时间预算超限",
    "FAILED_NO_CANDIDATE": "无候选",
    "FAILED_IMPLEMENTATION_ERROR": "实现错误",
}
_VIOLATION_SRC_ZH = {
    "slot_router_diagnostics": "槽位路由诊断",
    "candidate_allocation_only": "候选分配推导",
    "candidate_validate": "候选校验",
    "UNAVAILABLE_FOR_CANDIDATE": "候选不可得",
}
_CONFIDENCE_ZH = {"high": "高", "medium": "中", "low": "低"}


def _zh_enum(value: object, mapping: Dict[str, str]) -> object:
    if value is None or value is pd.NA:
        return value
    s = str(value)
    return mapping.get(s, value)


def _zh_line(value: object) -> object:
    return _zh_enum(value, _LINE_ZH)


def _kv_to_zh(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if list(out.columns) == ["metric", "value"]:
        out = out.rename(columns={"metric": "指标", "value": "值"})
    elif list(out.columns) == ["指标", "值"]:
        return out
    return out


def _campaign_ton_penalty(df: pd.DataFrame, rule: RuleConfig) -> Tuple[int, float]:
    if df.empty or "is_virtual" not in df.columns:
        return 0, 0.0
    real = df[~df["is_virtual"]].copy()
    if real.empty:
        return 0, 0.0
    tons_by = real.groupby(["line", "campaign_id"])["tons"].sum().reset_index(name="campaign_tons")
    low = tons_by[tons_by["campaign_tons"] < float(rule.campaign_ton_min)].copy()
    if low.empty:
        return 0, 0.0
    low["gap_tons"] = float(rule.campaign_ton_min) - low["campaign_tons"]
    return int(len(low)), float(low["gap_tons"].sum())


def _autosize_excel(writer: pd.ExcelWriter) -> None:
    for ws in writer.book.worksheets:
        for idx, col in enumerate(
            ws.iter_cols(min_row=1, max_row=ws.max_row, min_col=1, max_col=ws.max_column),
            start=1,
        ):
            max_len = 0
            for cell in col:
                val = "" if cell.value is None else str(cell.value)
                max_len = max(max_len, len(val))
            ws.column_dimensions[get_column_letter(idx)].width = min(60, max(10, max_len + 2))


def _flatten_dict(data: Dict[str, object], prefix: str = "") -> Iterable[tuple[str, object]]:
    for key, value in data.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            yield from _flatten_dict(value, full_key)
        else:
            yield full_key, value


def _num_int(value: object) -> int:
    if value is None or value == "":
        return 0
    if isinstance(value, bool):
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return 0


def _num_float(value: object) -> float:
    if value is None or value == "":
        return 0.0
    if isinstance(value, bool):
        return float(int(value))
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _build_diagnostics_report(diagnostics: Dict[str, object] | None) -> pd.DataFrame:
    if not isinstance(diagnostics, dict) or not diagnostics:
        return pd.DataFrame(columns=["section", "metric", "value"])
    rows = []
    for section, value in diagnostics.items():
        if isinstance(value, dict):
            for item, item_value in _flatten_dict(value):
                rows.append((str(section), str(item), item_value))
        else:
            rows.append(("general", str(section), value))
    return pd.DataFrame(rows, columns=["section", "metric", "value"])


def _candidate_line_membership_matches(row: pd.Series, line: str) -> bool:
    if str(row.get("line", "")) == str(line):
        return True
    candidate_lines = [v.strip() for v in str(row.get("candidate_lines", "") or "").split(",") if v.strip()]
    return str(line) in candidate_lines


def _build_candidate_line_summary(candidate_schedule_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    candidate_df = candidate_schedule_df.copy() if isinstance(candidate_schedule_df, pd.DataFrame) else pd.DataFrame()
    for line in ["big_roll", "small_roll"]:
        assigned = candidate_df[
            (candidate_df.get("line", pd.Series(dtype=str)) == line)
            & (~candidate_df.get("drop_flag", pd.Series(dtype=bool)).fillna(False))
        ].copy() if not candidate_df.empty else pd.DataFrame()
        dropped = candidate_df[
            candidate_df.apply(lambda r: _candidate_line_membership_matches(r, line), axis=1)
            & candidate_df.get("drop_flag", pd.Series(dtype=bool)).fillna(False)
        ].copy() if not candidate_df.empty else pd.DataFrame()
        slot_sizes = assigned.groupby("slot_no").size() if ("slot_no" in assigned.columns and not assigned.empty) else pd.Series(dtype=int)
        edge_series = assigned.get("selected_edge_type", pd.Series(dtype=str)).astype(str) if not assigned.empty else pd.Series(dtype=str)
        rows.append(
            {
                "line": line,
                "candidate_assigned_orders": int(len(assigned)),
                "candidate_assigned_tons": round(float(assigned["tons"].sum()) if "tons" in assigned.columns else 0.0, 1),
                "candidate_slot_count": int(assigned["slot_no"].nunique()) if "slot_no" in assigned.columns and not assigned.empty else 0,
                "candidate_avg_slot_order_count": round(float(slot_sizes.mean()) if not slot_sizes.empty else 0.0, 2),
                "candidate_max_slot_order_count": int(slot_sizes.max()) if not slot_sizes.empty else 0,
                "candidate_unroutable_slot_count": int(assigned.loc[assigned.get("slot_unroutable_flag", pd.Series(dtype=bool)).fillna(False), "slot_no"].nunique()) if "slot_no" in assigned.columns and not assigned.empty else 0,
                "candidate_dropped_order_count": int(len(dropped)),
                "candidate_direct_edge_count": int((edge_series == "DIRECT_EDGE").sum()) if not edge_series.empty else 0,
                "candidate_real_bridge_edge_count": int((edge_series == "REAL_BRIDGE_EDGE").sum()) if not edge_series.empty else 0,
                "candidate_virtual_bridge_edge_count": int((edge_series == "VIRTUAL_BRIDGE_EDGE").sum()) if not edge_series.empty else 0,
            }
        )
    return pd.DataFrame(rows)


def _extract_raw_slots(
    failure_diagnostics: Dict[str, object] | None,
    engine_meta: Dict[str, object] | None,
) -> list[dict]:
    for source in (failure_diagnostics, engine_meta):
        if not isinstance(source, dict):
            continue
        raw_slots = source.get("slot_route_details")
        if isinstance(raw_slots, dict):
            raw_slots = list(raw_slots.values())
        if isinstance(raw_slots, list) and raw_slots:
            return [dict(item) for item in raw_slots if isinstance(item, dict)]
    return []


def _build_candidate_schedule_sheet(candidate_schedule_df: pd.DataFrame, line: str) -> pd.DataFrame:
    candidate_df = candidate_schedule_df.copy() if isinstance(candidate_schedule_df, pd.DataFrame) else pd.DataFrame()
    if candidate_df.empty:
        return pd.DataFrame(
            columns=[
                "order_id",
                "slot_no",
                "candidate_position",
                "candidate_slot_member_index",
                "tons",
                "width",
                "thickness",
                "steel_group",
                "line_capability",
                "candidate_status",
                "slot_unroutable_flag",
                "slot_route_risk_score",
                "selected_edge_type",
                "analysis_only",
                "official_usable",
                "result_usage",
            ]
        )
    out = candidate_df[
        (candidate_df.get("line", pd.Series(dtype=str)) == line)
        & (~candidate_df.get("drop_flag", pd.Series(dtype=bool)).fillna(False))
    ].copy()
    if out.empty:
        return pd.DataFrame(
            columns=[
                "order_id",
                "slot_no",
                "candidate_position",
                "candidate_slot_member_index",
                "tons",
                "width",
                "thickness",
                "steel_group",
                "line_capability",
                "candidate_status",
                "slot_unroutable_flag",
                "slot_route_risk_score",
                "selected_edge_type",
                "analysis_only",
                "official_usable",
                "result_usage",
            ]
        )
    if "candidate_position" not in out.columns:
        out["candidate_position"] = 0
    if "candidate_slot_member_index" not in out.columns:
        out["candidate_slot_member_index"] = out.get("candidate_position", 0)
    out["analysis_only"] = True
    out["official_usable"] = False
    out["result_usage"] = "ANALYSIS_ONLY"
    out = out.sort_values(
        ["slot_no", "candidate_position", "candidate_slot_member_index", "order_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    return out[
        [
            "order_id",
            "slot_no",
            "candidate_position",
            "candidate_slot_member_index",
            "tons",
            "width",
            "thickness",
            "steel_group",
            "line_capability",
            "candidate_status",
            "slot_unroutable_flag",
            "slot_route_risk_score",
            "selected_edge_type",
            "analysis_only",
            "official_usable",
            "result_usage",
        ]
    ]


def _build_slot_frames(raw_slots: list[dict], candidate_schedule_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    slot_rows: list[dict] = []
    unroutable_rows: list[dict] = []
    if raw_slots:
        for item in raw_slots:
            min_width = item.get("min_width", 0) or 0
            max_width = item.get("max_width", 0) or 0
            min_thickness = item.get("min_thickness", 0) or 0
            max_thickness = item.get("max_thickness", 0) or 0
            width_span = item.get("width_span", (float(max_width) - float(min_width)) if max_width or min_width else 0)
            thickness_span = item.get("thickness_span", (float(max_thickness) - float(min_thickness)) if max_thickness or min_thickness else 0)
            row = {
                "line": item.get("line", ""),
                "slot_no": item.get("slot_no", 0),
                "slot_order_count": item.get("order_count", 0),
                "slot_tons": item.get("slot_tons", 0),
                "order_count_over_cap": item.get("order_count_over_cap", 0),
                "slot_route_risk_score": item.get("slot_route_risk_score", 0),
                "pair_gap_proxy": item.get("pair_gap_proxy", 0),
                "span_risk": item.get("span_risk", 0),
                "degree_risk": item.get("degree_risk", 0),
                "isolated_order_penalty": item.get("isolated_order_penalty", 0),
                "dominant_unroutable_reason": item.get("dominant_unroutable_reason", ""),
                "template_coverage_ratio": item.get("template_coverage_ratio", 0),
                "missing_template_edge_count": item.get("missing_template_edge_count", item.get("missing_pair_count", 0)),
                "zero_in_orders": item.get("zero_in_orders", 0),
                "zero_out_orders": item.get("zero_out_orders", 0),
                "width_span": width_span,
                "thickness_span": thickness_span,
                "steel_group_count": item.get("steel_group_count", 0),
                "top_isolated_orders": item.get("top_isolated_orders", []),
                "status": item.get("status", ""),
            }
            slot_rows.append(row)
            if str(item.get("status", "")) == "UNROUTABLE_SLOT":
                unroutable_rows.append(
                    {
                        "line": row["line"],
                        "slot_no": row["slot_no"],
                        "order_count": row["slot_order_count"],
                        "slot_route_risk_score": row["slot_route_risk_score"],
                        "template_coverage_ratio": row["template_coverage_ratio"],
                        "missing_template_edge_count": row["missing_template_edge_count"],
                        "zero_in_orders": row["zero_in_orders"],
                        "zero_out_orders": row["zero_out_orders"],
                        "width_span": row["width_span"],
                        "thickness_span": row["thickness_span"],
                        "steel_group_count": row["steel_group_count"],
                        "dominant_unroutable_reason": row["dominant_unroutable_reason"],
                        "top_isolated_orders": row["top_isolated_orders"],
                    }
                )
    elif isinstance(candidate_schedule_df, pd.DataFrame) and not candidate_schedule_df.empty:
        assigned = candidate_schedule_df[
            (~candidate_schedule_df.get("drop_flag", pd.Series(dtype=bool)).fillna(False))
            & candidate_schedule_df.get("line", pd.Series(dtype=str)).astype(str).ne("")
        ].copy()
        if not assigned.empty:
            for (line, slot_no), grp in assigned.groupby(["line", "slot_no"], dropna=False):
                slot_unroutable = bool(grp.get("slot_unroutable_flag", pd.Series(dtype=bool)).fillna(False).any())
                row = {
                    "line": line,
                    "slot_no": slot_no,
                    "slot_order_count": int(len(grp)),
                    "slot_tons": round(float(grp["tons"].sum()) if "tons" in grp.columns else 0.0, 1),
                    "order_count_over_cap": 0,
                    "slot_route_risk_score": int(grp.get("slot_route_risk_score", pd.Series(dtype=float)).fillna(0).max()) if "slot_route_risk_score" in grp.columns else 0,
                    "pair_gap_proxy": pd.NA,
                    "span_risk": pd.NA,
                    "degree_risk": pd.NA,
                    "isolated_order_penalty": pd.NA,
                    "dominant_unroutable_reason": "UNROUTABLE_SLOT_MEMBER" if slot_unroutable else "",
                    "template_coverage_ratio": pd.NA,
                    "missing_template_edge_count": pd.NA,
                    "zero_in_orders": pd.NA,
                    "zero_out_orders": pd.NA,
                    "width_span": round(float(grp["width"].max() - grp["width"].min()), 1) if "width" in grp.columns else pd.NA,
                    "thickness_span": round(float(grp["thickness"].max() - grp["thickness"].min()), 3) if "thickness" in grp.columns else pd.NA,
                    "steel_group_count": int(grp["steel_group"].astype(str).nunique()) if "steel_group" in grp.columns else pd.NA,
                    "top_isolated_orders": [],
                    "status": "UNROUTABLE_SLOT" if slot_unroutable else "ASSIGNED_CANDIDATE",
                }
                slot_rows.append(row)
                if slot_unroutable:
                    unroutable_rows.append(
                        {
                            "line": row["line"],
                            "slot_no": row["slot_no"],
                            "order_count": row["slot_order_count"],
                            "slot_route_risk_score": row["slot_route_risk_score"],
                            "template_coverage_ratio": row["template_coverage_ratio"],
                            "missing_template_edge_count": row["missing_template_edge_count"],
                            "zero_in_orders": row["zero_in_orders"],
                            "zero_out_orders": row["zero_out_orders"],
                            "width_span": row["width_span"],
                            "thickness_span": row["thickness_span"],
                            "steel_group_count": row["steel_group_count"],
                            "dominant_unroutable_reason": row["dominant_unroutable_reason"],
                            "top_isolated_orders": row["top_isolated_orders"],
                        }
                    )
    slot_summary_df = pd.DataFrame(
        slot_rows,
        columns=[
            "line",
            "slot_no",
            "slot_order_count",
            "slot_tons",
            "order_count_over_cap",
            "slot_route_risk_score",
            "pair_gap_proxy",
            "span_risk",
            "degree_risk",
            "isolated_order_penalty",
            "dominant_unroutable_reason",
        ],
    )
    unroutable_slots_df = pd.DataFrame(
        unroutable_rows,
        columns=[
            "line",
            "slot_no",
            "order_count",
            "template_coverage_ratio",
            "missing_template_edge_count",
            "zero_in_orders",
            "zero_out_orders",
            "width_span",
            "thickness_span",
            "steel_group_count",
            "top_isolated_orders",
            "slot_route_risk_score",
            "dominant_unroutable_reason",
        ],
    )
    return slot_summary_df, unroutable_slots_df


def _build_candidate_violation_summary(
    candidate_schedule_df: pd.DataFrame,
    raw_slots: list[dict],
    validation_summary: dict,
    best_candidate_routing_feasible: bool,
) -> pd.DataFrame:
    candidate_df = candidate_schedule_df.copy() if isinstance(candidate_schedule_df, pd.DataFrame) else pd.DataFrame()
    if raw_slots:
        candidate_unroutable_slot_count = int(len([r for r in raw_slots if str(r.get("status", "")) == "UNROUTABLE_SLOT"]))
        candidate_bad_slot_count = candidate_unroutable_slot_count
        candidate_zero_in = int(sum(int(r.get("zero_in_orders", 0) or 0) for r in raw_slots))
        candidate_zero_out = int(sum(int(r.get("zero_out_orders", 0) or 0) for r in raw_slots))
        candidate_period_reverse_viol = int(sum(int(r.get("period_reverse_count_violation_count", 0) or 0) for r in raw_slots))
        slot_source = "slot_router_diagnostics"
        slot_conf = "high"
    elif not candidate_df.empty:
        assigned = candidate_df[
            (~candidate_df.get("drop_flag", pd.Series(dtype=bool)).fillna(False))
            & candidate_df.get("line", pd.Series(dtype=str)).astype(str).ne("")
        ].copy()
        if not assigned.empty and "slot_no" in assigned.columns:
            slot_unroutable = assigned.get("slot_unroutable_flag", pd.Series(dtype=bool)).fillna(False)
            candidate_unroutable_slot_count = int(assigned.loc[slot_unroutable, "slot_no"].nunique())
            candidate_bad_slot_count = candidate_unroutable_slot_count
        else:
            candidate_unroutable_slot_count = 0
            candidate_bad_slot_count = 0
        # Without per-slot diagnostics we cannot confidently infer zero-in/out counts.
        candidate_zero_in = pd.NA
        candidate_zero_out = pd.NA
        candidate_period_reverse_viol = pd.NA
        slot_source = "candidate_allocation_only"
        slot_conf = "medium"
    else:
        candidate_unroutable_slot_count = 0
        candidate_bad_slot_count = 0
        candidate_zero_in = pd.NA
        candidate_zero_out = pd.NA
        candidate_period_reverse_viol = pd.NA
        slot_source = "UNAVAILABLE_FOR_CANDIDATE"
        slot_conf = "low"
    unavailable = pd.NA
    candidate_direct_reverse = validation_summary.get("direct_reverse_step_violation_count", unavailable) if best_candidate_routing_feasible else unavailable
    candidate_virtual_attach_reverse = validation_summary.get("virtual_attach_reverse_violation_count", unavailable) if best_candidate_routing_feasible else unavailable
    candidate_bridge_count_violation = validation_summary.get("bridge_count_violation_count", unavailable) if best_candidate_routing_feasible else unavailable
    candidate_invalid_virtual_spec = validation_summary.get("invalid_virtual_spec_count", unavailable) if best_candidate_routing_feasible else unavailable
    rows = [
        ("candidate_unroutable_slot_count", _num_int(validation_summary.get("unroutable_slot_count", 0)), candidate_unroutable_slot_count, slot_source, slot_conf),
        ("candidate_bad_slot_count", 0, candidate_bad_slot_count, slot_source, slot_conf),
        ("candidate_zero_in_order_count", 0, candidate_zero_in, slot_source if raw_slots else "UNAVAILABLE_FOR_CANDIDATE", "high" if raw_slots else "low"),
        ("candidate_zero_out_order_count", 0, candidate_zero_out, slot_source if raw_slots else "UNAVAILABLE_FOR_CANDIDATE", "high" if raw_slots else "low"),
        ("candidate_direct_reverse_step_violation_count", _num_int(validation_summary.get("direct_reverse_step_violation_count", 0)), candidate_direct_reverse, "candidate_validate" if best_candidate_routing_feasible else "UNAVAILABLE_FOR_CANDIDATE", "high" if best_candidate_routing_feasible else "low"),
        ("candidate_virtual_attach_reverse_violation_count", _num_int(validation_summary.get("virtual_attach_reverse_violation_count", 0)), candidate_virtual_attach_reverse, "candidate_validate" if best_candidate_routing_feasible else "UNAVAILABLE_FOR_CANDIDATE", "high" if best_candidate_routing_feasible else "low"),
        ("candidate_period_reverse_count_violation_count", _num_int(validation_summary.get("period_reverse_count_violation_count", 0)), candidate_period_reverse_viol, slot_source, slot_conf),
        ("candidate_bridge_count_violation_count", _num_int(validation_summary.get("bridge_count_violation_count", 0)), candidate_bridge_count_violation, "candidate_validate" if best_candidate_routing_feasible else "UNAVAILABLE_FOR_CANDIDATE", "high" if best_candidate_routing_feasible else "low"),
        ("candidate_invalid_virtual_spec_count", _num_int(validation_summary.get("invalid_virtual_spec_count", 0)), candidate_invalid_virtual_spec, "candidate_validate" if best_candidate_routing_feasible else "UNAVAILABLE_FOR_CANDIDATE", "high" if best_candidate_routing_feasible else "low"),
    ]
    return pd.DataFrame(rows, columns=["metric", "official_value", "candidate_value", "violation_count_source", "violation_count_confidence"])


def export_schedule_results(
    final_df: pd.DataFrame,
    rounds_df: pd.DataFrame,
    dropped_df: pd.DataFrame,
    output_path: str,
    input_order_count: int,
    rule: RuleConfig,
    t_start: float | None = None,
    engine_used: str = "unknown",
    fallback_used: bool = False,
    fallback_type: str = "",
    fallback_reason: str = "",
    equivalence_summary: Dict[str, object] | None = None,
    failure_diagnostics: Dict[str, object] | None = None,
    engine_meta: Dict[str, object] | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    final_df = final_df.copy()
    rounds_df = rounds_df.copy()
    dropped_df = dropped_df.copy()
    final_df["engine_used"] = str(engine_used)
    rounds_df["engine_used"] = str(engine_used)

    low_cnt, low_gap = _campaign_ton_penalty(final_df, rule)
    diagnostics_report = _build_diagnostics_report(failure_diagnostics)
    fallback_diag = failure_diagnostics.get("fallback", {}) if isinstance(failure_diagnostics, dict) else {}
    em = engine_meta if isinstance(engine_meta, dict) else {}
    main_path = str(fallback_diag.get("main_path", engine_used))
    used_local_routing = bool(fallback_diag.get("used_local_routing", False))
    local_routing_role = str(fallback_diag.get("local_routing_role", "not_used"))
    bridge_modeling = str(fallback_diag.get("bridge_modeling", "template_based"))
    joint_estimates = failure_diagnostics.get("joint_estimates", {}) if isinstance(failure_diagnostics, dict) else {}
    routing_feasible = bool(fallback_diag.get("routing_feasible", False))
    template_pair_ok = bool(fallback_diag.get("template_pair_ok", False))
    adjacency_rule_ok = bool(fallback_diag.get("adjacency_rule_ok", False))
    bridge_expand_ok = bool(fallback_diag.get("bridge_expand_ok", False))
    unroutable_slot_count = int(fallback_diag.get("unroutable_slot_count", 0))
    strict_template_edges_enabled = bool(fallback_diag.get("strict_template_edges_enabled", True))
    result_acceptance_status = str(fallback_diag.get("result_acceptance_status", "UNKNOWN"))
    failure_mode = str(fallback_diag.get("failure_mode", ""))
    evidence_level = str(fallback_diag.get("evidence_level", "OK"))
    top_infeasibility_signals = fallback_diag.get("top_infeasibility_signals", [])
    template_graph_health = str(fallback_diag.get("template_graph_health", "UNKNOWN"))
    precheck_autorelax_applied = bool(fallback_diag.get("precheck_autorelax_applied", False))
    solve_attempt_count = int(fallback_diag.get("solve_attempt_count", 0))
    fallback_attempt_count = int(fallback_diag.get("fallback_attempt_count", 0))
    early_stop_reason = str(fallback_diag.get("early_stop_reason", ""))
    export_failed_result_for_debug = bool(fallback_diag.get("export_failed_result_for_debug", False))
    export_analysis_on_failure = bool(fallback_diag.get("export_analysis_on_failure", False))
    export_best_candidate_analysis = bool(fallback_diag.get("export_best_candidate_analysis", False))
    final_export_performed = bool(fallback_diag.get("final_export_performed", True))
    official_exported = bool(fallback_diag.get("official_exported", routing_feasible))
    analysis_exported = bool(fallback_diag.get("analysis_exported", False))
    result_usage = str(fallback_diag.get("result_usage", "OFFICIAL" if routing_feasible else "UNKNOWN"))
    best_candidate_available = bool(fallback_diag.get("best_candidate_available", False))
    best_candidate_type = str(fallback_diag.get("best_candidate_type", ""))
    best_candidate_objective = float(fallback_diag.get("best_candidate_objective", 0.0) or 0.0)
    best_candidate_search_status = str(fallback_diag.get("best_candidate_search_status", ""))
    best_candidate_routing_feasible = bool(fallback_diag.get("best_candidate_routing_feasible", False))
    best_candidate_unroutable_slot_count = int(fallback_diag.get("best_candidate_unroutable_slot_count", 0) or 0)
    export_consistency_ok = bool(fallback_diag.get("export_consistency_ok", True))
    # Keep console / engine_meta / export consistent: engine_meta is treated as the source of truth when present.
    if isinstance(em, dict) and em:
        result_acceptance_status = str(em.get("result_acceptance_status", result_acceptance_status))
        failure_mode = str(em.get("failure_mode", failure_mode))
        result_usage = str(em.get("result_usage", result_usage))
        best_candidate_available = bool(em.get("best_candidate_available", best_candidate_available))
        best_candidate_type = str(em.get("best_candidate_type", best_candidate_type))
        best_candidate_search_status = str(em.get("best_candidate_search_status", best_candidate_search_status))
        export_consistency_ok = bool(
            export_consistency_ok
            and (str(fallback_diag.get("result_acceptance_status", "")) in {"", str(result_acceptance_status)})
            and (str(fallback_diag.get("failure_mode", "")) in {"", str(failure_mode)})
            and (str(fallback_diag.get("result_usage", "")) in {"", str(result_usage)})
        )
    assignment_pressure_mode = str(fallback_diag.get("assignment_pressure_mode", "normal"))
    template_build_seconds = float(fallback_diag.get("template_build_seconds", 0.0))
    joint_master_seconds = float(fallback_diag.get("joint_master_seconds", 0.0))
    local_router_seconds = float(fallback_diag.get("local_router_seconds", 0.0))
    fallback_total_seconds = float(fallback_diag.get("fallback_total_seconds", 0.0))
    total_run_seconds = float(fallback_diag.get("total_run_seconds", 0.0))
    candidate_schedule_df = em.get("candidate_schedule_df") if isinstance(em.get("candidate_schedule_df"), pd.DataFrame) else pd.DataFrame()
    candidate_big_roll_df = em.get("candidate_big_roll_df") if isinstance(em.get("candidate_big_roll_df"), pd.DataFrame) else pd.DataFrame()
    candidate_small_roll_df = em.get("candidate_small_roll_df") if isinstance(em.get("candidate_small_roll_df"), pd.DataFrame) else pd.DataFrame()
    candidate_line_summary_df = _build_candidate_line_summary(candidate_schedule_df)
    raw_slots = _extract_raw_slots(failure_diagnostics if isinstance(failure_diagnostics, dict) else None, em)
    candidate_backfill_mode = (
        final_df.empty
        and (
            str(result_acceptance_status) == "BEST_SEARCH_CANDIDATE_ANALYSIS"
            or (bool(best_candidate_available) and not bool(official_exported))
        )
        and isinstance(candidate_schedule_df, pd.DataFrame)
        and not candidate_schedule_df.empty
    )
    if final_df.empty and raw_slots:
        computed_unroutable = int(len([r for r in raw_slots if str(r.get("status", "")) == "UNROUTABLE_SLOT"]))
        if computed_unroutable > 0 and unroutable_slot_count <= 0:
            unroutable_slot_count = computed_unroutable

    summary_df = pd.DataFrame(
        [
            ("结果引擎路径", str(engine_used)),
            ("主路径", main_path),
            ("是否触发 fallback", "是" if bool(fallback_used) else "否"),
            ("fallback 类型", str(fallback_type)),
            ("fallback 原因", str(fallback_reason)),
            ("是否使用局部路由", "是" if used_local_routing else "否"),
            ("局部路由角色", local_routing_role),
            ("bridge modeling", bridge_modeling),
            ("routing_feasible", routing_feasible),
            ("template_pair_ok", template_pair_ok),
            ("adjacency_rule_ok", adjacency_rule_ok),
            ("bridge_expand_ok", bridge_expand_ok),
            ("unroutable_slot_count", unroutable_slot_count),
            ("strict_template_edges_enabled", strict_template_edges_enabled),
            ("result_acceptance_status", result_acceptance_status),
            ("failure_mode", failure_mode),
            ("evidence_level", evidence_level),
            ("export_failed_result_for_debug", export_failed_result_for_debug),
            ("export_analysis_on_failure", export_analysis_on_failure),
            ("export_best_candidate_analysis", export_best_candidate_analysis),
            ("final_export_performed", final_export_performed),
            ("official_exported", official_exported),
            ("analysis_exported", analysis_exported),
            ("result_usage", result_usage),
            ("best_candidate_available", best_candidate_available),
            ("best_candidate_type", best_candidate_type),
            ("best_candidate_objective", best_candidate_objective),
            ("best_candidate_search_status", best_candidate_search_status),
            ("best_candidate_routing_feasible", best_candidate_routing_feasible),
            ("best_candidate_unroutable_slot_count", best_candidate_unroutable_slot_count),
            ("template_graph_health", template_graph_health),
            ("precheck_autorelax_applied", precheck_autorelax_applied),
            ("solve_attempt_count", solve_attempt_count),
            ("fallback_attempt_count", fallback_attempt_count),
            ("early_stop_reason", early_stop_reason),
            ("template_build_seconds", template_build_seconds),
            ("joint_master_seconds", joint_master_seconds),
            ("local_router_seconds", local_router_seconds),
            ("fallback_total_seconds", fallback_total_seconds),
            ("total_run_seconds", total_run_seconds),
            ("吨位上限语义", "HARD"),
            ("吨位下限语义", "STRONG_SOFT"),
            ("虚拟占比语义", "STRONG_SOFT"),
            ("逆宽次数/总量语义", "HARD"),
            ("真实订单数", int((~final_df["is_virtual"]).sum()) if "is_virtual" in final_df.columns else int(len(final_df))),
            (f"低吨位轧期数(<{int(rule.campaign_ton_min)})", int(low_cnt)),
            ("低吨位缺口吨位", round(float(low_gap), 1)),
            ("剔除订单数", int(len(dropped_df))),
        ],
        columns=["指标", "值"],
    )
    if isinstance(equivalence_summary, dict) and equivalence_summary:
        for k, v in equivalence_summary.items():
            summary_df.loc[len(summary_df)] = (f"等价性.{k}", v)
    if isinstance(top_infeasibility_signals, list):
        for idx, item in enumerate(top_infeasibility_signals[:5], start=1):
            summary_df.loc[len(summary_df)] = (f"不可行信号.top{idx}", item)
    if isinstance(joint_estimates, dict):
        for k, v in joint_estimates.items():
            summary_df.loc[len(summary_df)] = (f"主模型估计.{k}", v)

    runtime_rows = [
        ("结果引擎路径", str(engine_used)),
        ("主路径", main_path),
        ("是否触发 fallback", "是" if bool(fallback_used) else "否"),
        ("fallback 类型", str(fallback_type)),
        ("fallback 原因", str(fallback_reason)),
        ("是否使用局部路由", "是" if used_local_routing else "否"),
        ("局部路由角色", local_routing_role),
        ("bridge modeling", bridge_modeling),
        ("routing_feasible", routing_feasible),
        ("template_pair_ok", template_pair_ok),
        ("adjacency_rule_ok", adjacency_rule_ok),
        ("bridge_expand_ok", bridge_expand_ok),
        ("unroutable_slot_count", unroutable_slot_count),
        ("strict_template_edges_enabled", strict_template_edges_enabled),
        ("result_acceptance_status", result_acceptance_status),
        ("failure_mode", failure_mode),
        ("evidence_level", evidence_level),
        ("export_failed_result_for_debug", export_failed_result_for_debug),
        ("export_analysis_on_failure", export_analysis_on_failure),
        ("export_best_candidate_analysis", export_best_candidate_analysis),
        ("final_export_performed", final_export_performed),
        ("official_exported", official_exported),
        ("analysis_exported", analysis_exported),
        ("result_usage", result_usage),
        ("best_candidate_available", best_candidate_available),
        ("best_candidate_type", best_candidate_type),
        ("best_candidate_objective", best_candidate_objective),
        ("best_candidate_search_status", best_candidate_search_status),
        ("best_candidate_routing_feasible", best_candidate_routing_feasible),
        ("best_candidate_unroutable_slot_count", best_candidate_unroutable_slot_count),
        ("export_consistency_ok", export_consistency_ok),
            ("template_build_seconds", template_build_seconds),
            ("joint_master_seconds", joint_master_seconds),
            ("local_router_seconds", local_router_seconds),
            ("fallback_total_seconds", fallback_total_seconds),
            ("total_run_seconds", total_run_seconds),
            ("输入订单数", int(input_order_count)),
        ("输出真实订单数", int((~final_df["is_virtual"]).sum()) if "is_virtual" in final_df.columns else int(len(final_df))),
        ("输出虚拟板坯数", int(final_df["is_virtual"].sum()) if "is_virtual" in final_df.columns else 0),
        ("低吨位轧期数", int(low_cnt)),
        ("低吨位缺口吨位", round(float(low_gap), 1)),
        ("剔除订单数", int(len(dropped_df))),
    ]
    if isinstance(failure_diagnostics, dict):
        runtime_rows.extend((f"诊断.{k}", v) for k, v in _flatten_dict(failure_diagnostics))
    runtime_report = pd.DataFrame(runtime_rows, columns=["指标", "值"])

    equivalence_report = pd.DataFrame(columns=["检查项", "值"])
    if isinstance(equivalence_summary, dict) and equivalence_summary:
        equivalence_report = pd.DataFrame([(str(k), v) for k, v in equivalence_summary.items()], columns=["检查项", "值"])

    diagnostics_snapshot = pd.DataFrame(
        [
            ("main_path", main_path),
            ("engine_used", str(engine_used)),
            ("fallback_used", bool(fallback_used)),
            ("fallback_type", str(fallback_type)),
            ("fallback_reason", str(fallback_reason)),
            ("assignment_pressure_mode", assignment_pressure_mode),
            ("used_local_routing", bool(used_local_routing)),
            ("local_routing_role", str(local_routing_role)),
            ("bridge_modeling", str(bridge_modeling)),
            ("result_acceptance_status", result_acceptance_status),
            ("failure_mode", failure_mode),
            ("evidence_level", evidence_level),
            ("export_failed_result_for_debug", export_failed_result_for_debug),
            ("export_analysis_on_failure", export_analysis_on_failure),
            ("export_best_candidate_analysis", export_best_candidate_analysis),
            ("final_export_performed", final_export_performed),
            ("official_exported", official_exported),
            ("analysis_exported", analysis_exported),
            ("result_usage", result_usage),
            ("best_candidate_available", best_candidate_available),
            ("best_candidate_type", best_candidate_type),
            ("best_candidate_objective", best_candidate_objective),
            ("best_candidate_search_status", best_candidate_search_status),
            ("best_candidate_routing_feasible", best_candidate_routing_feasible),
            ("best_candidate_unroutable_slot_count", best_candidate_unroutable_slot_count),
            ("export_consistency_ok", export_consistency_ok),
            ("template_graph_health", template_graph_health),
            ("precheck_autorelax_applied", precheck_autorelax_applied),
            ("solve_attempt_count", solve_attempt_count),
            ("fallback_attempt_count", fallback_attempt_count),
            ("early_stop_reason", early_stop_reason),
            ("template_build_seconds", template_build_seconds),
            ("joint_master_seconds", joint_master_seconds),
            ("local_router_seconds", local_router_seconds),
            ("fallback_total_seconds", fallback_total_seconds),
            ("total_run_seconds", total_run_seconds),
        ],
        columns=["key", "value"],
    )

    precheck_summary_df = pd.DataFrame(
        [
            ("total_orders", int(input_order_count)),
            ("total_tons", round(float(final_df["tons"].sum() + dropped_df["tons"].sum()) if "tons" in final_df.columns and "tons" in dropped_df.columns else 0.0, 1)),
            ("line_capability_histogram", str((dropped_df.get("line_capability", pd.Series(dtype=str)).value_counts().to_dict() if "line_capability" in dropped_df.columns else {}))),
            ("suspicious_data_count", int(fallback_diag.get("suspicious_data_count", 0) if isinstance(fallback_diag, dict) else 0)),
        ],
        columns=["metric", "value"],
    )

    feasibility_evidence_df = pd.DataFrame(columns=["metric", "value"])
    if isinstance(failure_diagnostics, dict) and isinstance(failure_diagnostics.get("feasibility_evidence"), dict):
        fe = failure_diagnostics["feasibility_evidence"]
        rows = [("evidence_level", failure_diagnostics.get("fallback", {}).get("evidence_level", "OK"))]
        for key, value in _flatten_dict(fe):
            rows.append((str(key), value))
        feasibility_evidence_df = pd.DataFrame(rows, columns=["metric", "value"])

    template_summary_df = pd.DataFrame(columns=["line", "candidate_pairs", "templates_built", "template_coverage_ratio", "avg_virtual_count", "max_virtual_count", "rejected_by_chain_limit", "direct_edge_count", "real_bridge_edge_count", "virtual_bridge_edge_count"])
    if isinstance(failure_diagnostics, dict):
        template_section = failure_diagnostics.get("template", {})
        rows = []
        for line in ["big_roll", "small_roll"]:
            rows.append(
                {
                    "line": line,
                    "candidate_pairs": template_section.get(f"build_debug.{line}.candidate_pairs", 0),
                    "templates_built": template_section.get(f"{line}.kept_templates", template_section.get(f"build_debug.{line}.kept_templates", 0)),
                    "template_coverage_ratio": template_section.get(f"{line}.coverage_ratio", 0.0),
                    "avg_virtual_count": template_section.get(f"build_debug.{line}.avg_virtual_count", 0.0),
                    "max_virtual_count": template_section.get(f"build_debug.{line}.max_virtual_count", 0),
                    "rejected_by_chain_limit": template_section.get(f"build_debug.{line}.rejected_by_chain_limit", 0),
                    "direct_edge_count": template_section.get(f"build_debug.{line}.direct_edge_count", 0),
                    "real_bridge_edge_count": template_section.get(f"build_debug.{line}.real_bridge_edge_count", 0),
                    "virtual_bridge_edge_count": template_section.get(f"build_debug.{line}.virtual_bridge_edge_count", 0),
                }
            )
        template_summary_df = pd.DataFrame(rows)

    master_allocation_summary_df = pd.DataFrame(columns=["line", "slot_no", "slot_order_count", "slot_tons", "order_count_over_cap", "slot_route_risk_score", "pair_gap_proxy", "span_risk", "degree_risk", "isolated_order_penalty", "selected_bridge_mix"])
    if raw_slots:
        master_allocation_summary_df = pd.DataFrame(
            [
                {
                    "line": r.get("line", ""),
                    "slot_no": r.get("slot_no", 0),
                    "slot_order_count": r.get("order_count", 0),
                    "slot_tons": r.get("slot_tons", 0),
                    "order_count_over_cap": r.get("order_count_over_cap", 0),
                    "slot_route_risk_score": r.get("slot_route_risk_score", 0),
                    "pair_gap_proxy": r.get("pair_gap_proxy", 0),
                    "span_risk": r.get("span_risk", 0),
                    "degree_risk": r.get("degree_risk", 0),
                    "isolated_order_penalty": r.get("isolated_order_penalty", 0),
                    "selected_bridge_mix": r.get("selected_bridge_mix", ""),
                }
                for r in raw_slots
            ]
        )

    unroutable_slots_df = pd.DataFrame(
        columns=[
            "line", "slot_no", "order_count", "template_coverage_ratio", "missing_pair_count",
            "line_slot_order_cap", "order_count_over_cap",
            "zero_in_orders", "zero_out_orders", "min_width", "max_width",
            "min_thickness", "max_thickness", "steel_group_count",
            "estimated_reverse_count", "estimated_virtual_blocks",
            "isolated_order_penalty", "period_reverse_count",
            "period_reverse_count_violation_count", "reverse_count_definition",
            "top_isolated_orders", "slot_route_risk_score", "status",
        ]
    )
    if raw_slots:
        unroutable_slots_df = pd.DataFrame([r for r in raw_slots if str(r.get("status", "")) == "UNROUTABLE_SLOT"])

    rule_snapshot = pd.DataFrame(
        [
            (spec.key.value, spec.zh_name, spec.en_name, spec.solver_active, spec.validation_active, spec.export_visible)
            for spec in RULE_REGISTRY.export_visible_specs()
        ],
        columns=["rule_key", "zh_name", "en_name", "solver_active", "validation_active", "export_visible"],
    )

    flow_df = pd.DataFrame(
        [
            ("总订单数(输入-规划粒度)", int(input_order_count)),
            ("真实订单数(输出)", int((~final_df["is_virtual"]).sum()) if "is_virtual" in final_df.columns else int(len(final_df))),
            ("虚拟板坯数(输出)", int(final_df["is_virtual"].sum()) if "is_virtual" in final_df.columns else 0),
            ("剔除订单数", int(len(dropped_df))),
        ],
        columns=["指标", "值"],
    )

    campaign_tons_df = pd.DataFrame(columns=["产线", "产线代码", "轧期序号", "实物吨位", "虚拟吨位", "总吨位", "实物卷数", "虚拟卷数"])
    if not final_df.empty and {"line", "tons", "is_virtual"}.issubset(set(final_df.columns)):
        c = final_df.copy()
        c["实物吨位"] = c.apply(lambda r: float(r["tons"]) if not bool(r["is_virtual"]) else 0.0, axis=1)
        c["虚拟吨位"] = c.apply(lambda r: float(r["tons"]) if bool(r["is_virtual"]) else 0.0, axis=1)
        c["实物卷数"] = c["is_virtual"].map(lambda v: 0 if bool(v) else 1)
        c["虚拟卷数"] = c["is_virtual"].map(lambda v: 1 if bool(v) else 0)
        campaign_tons_df = (
            c.groupby(["line", "campaign_id"], as_index=False)[["实物吨位", "虚拟吨位", "实物卷数", "虚拟卷数"]]
            .sum()
            .rename(columns={"line": "产线代码", "campaign_id": "轧期序号"})
        )
        campaign_tons_df["总吨位"] = campaign_tons_df["实物吨位"] + campaign_tons_df["虚拟吨位"]
        campaign_tons_df["产线"] = campaign_tons_df["产线代码"].map({"big_roll": "大辊线", "small_roll": "小辊线"}).fillna(campaign_tons_df["产线代码"])
        campaign_tons_df = campaign_tons_df[["产线", "产线代码", "轧期序号", "实物吨位", "虚拟吨位", "总吨位", "实物卷数", "虚拟卷数"]]
        for col in ["实物吨位", "虚拟吨位", "总吨位"]:
            campaign_tons_df[col] = campaign_tons_df[col].round(1)

    show_cols = [
        "global_seq", "line_name", "line", "line_seq", "campaign_id", "campaign_seq", "campaign_real_seq",
        "order_id", "source_order_id", "parent_order_id", "lot_id", "is_virtual", "grade", "steel_group",
        "width", "thickness", "temp_min", "temp_max", "temp_mean", "tons", "backlog", "due_date", "due_bucket",
    ]
    schedule = final_df[[c for c in show_cols if c in final_df.columns]].copy()
    if "campaign_id" in schedule.columns:
        schedule["campaign_id"] = pd.to_numeric(schedule["campaign_id"], errors="coerce").fillna(0).astype(int)
    if "is_virtual" in schedule.columns:
        schedule["是否虚拟"] = schedule["is_virtual"].map(lambda x: "是" if bool(x) else "否")
        schedule = schedule.drop(columns=["is_virtual"])
    schedule = schedule.rename(
        columns={
            "global_seq": "全局序号",
            "line_name": "产线",
            "line": "产线代码",
            "line_seq": "产线内序号",
            "campaign_id": "轧期编号",
            "campaign_seq": "轧期内序号",
            "campaign_real_seq": "轧期内真实序号",
            "order_id": "物料号",
            "source_order_id": "来源订单号",
            "parent_order_id": "父订单号",
            "lot_id": "规划批号",
            "grade": "牌号",
            "steel_group": "钢种组",
            "width": "宽度",
            "thickness": "厚度",
            "temp_min": "温度下限",
            "temp_max": "温度上限",
            "temp_mean": "均热温度",
            "tons": "吨位",
            "backlog": "欠交量",
            "due_date": "交期",
            "due_bucket": "交期桶",
        }
    )
    if {"产线代码", "轧期编号", "产线内序号"}.issubset(set(schedule.columns)):
        schedule = schedule.sort_values(["产线代码", "轧期编号", "产线内序号"], kind="mergesort").reset_index(drop=True)
        schedule["轧期内序号"] = schedule.groupby(["产线代码", "轧期编号"]).cumcount() + 1
        schedule["轧期内真实序号"] = schedule["轧期内序号"]

    big_sheet = schedule[schedule["产线代码"] == "big_roll"].drop(columns=["产线代码"], errors="ignore").reset_index(drop=True) if "产线代码" in schedule.columns else schedule.copy()
    small_sheet = schedule[schedule["产线代码"] == "small_roll"].drop(columns=["产线代码"], errors="ignore").reset_index(drop=True) if "产线代码" in schedule.columns else schedule.iloc[0:0].copy()
    if candidate_backfill_mode:
        big_sheet = _build_candidate_schedule_sheet(candidate_schedule_df, "big_roll")
        small_sheet = _build_candidate_schedule_sheet(candidate_schedule_df, "small_roll")

    rounds_out = rounds_df.copy()
    if "line" in rounds_out.columns:
        rounds_out["产线"] = rounds_out["line"].map({"big_roll": "大辊线", "small_roll": "小辊线"}).fillna(rounds_out["line"])

    dropped_reason_stats = pd.DataFrame(columns=["剔除原因", "数量", "占比"])
    dropped_out = dropped_df.copy()
    if not dropped_out.empty and "dominant_drop_reason" in dropped_out.columns:
        stats = dropped_out["dominant_drop_reason"].fillna("UNKNOWN").value_counts().reset_index()
        stats.columns = ["剔除原因", "数量"]
        stats["占比"] = (stats["数量"] / max(1, int(stats["数量"].sum()))).round(4)
        dropped_reason_stats = stats

    scheduled_orders = _num_int((~final_df["is_virtual"]).sum()) if "is_virtual" in final_df.columns else _num_int(len(final_df))
    unscheduled_orders = _num_int(len(dropped_out))
    scheduled_tons = round(_num_float(final_df["tons"].sum()) if "tons" in final_df.columns else 0.0, 1)
    unscheduled_tons = round(_num_float(dropped_out["tons"].sum()) if "tons" in dropped_out.columns else 0.0, 1)
    selected_direct_edge_count = _num_int(fallback_diag.get("bridge_mix_summary", {}).get("selected_direct_edge_count", 0) if isinstance(fallback_diag, dict) else 0)
    selected_real_bridge_edge_count = _num_int(fallback_diag.get("bridge_mix_summary", {}).get("selected_real_bridge_edge_count", 0) if isinstance(fallback_diag, dict) else 0)
    selected_virtual_bridge_edge_count = _num_int(fallback_diag.get("bridge_mix_summary", {}).get("selected_virtual_bridge_edge_count", 0) if isinstance(fallback_diag, dict) else 0)
    if not final_df.empty and "line" in final_df.columns:
        real_mask = (~final_df["is_virtual"]) if "is_virtual" in final_df.columns else pd.Series([True] * len(final_df), index=final_df.index)
        big_roll_scheduled_orders = _num_int(((final_df["line"] == "big_roll") & real_mask).sum())
        small_roll_scheduled_orders = _num_int(((final_df["line"] == "small_roll") & real_mask).sum())
    else:
        big_roll_scheduled_orders = 0
        small_roll_scheduled_orders = 0
    if candidate_backfill_mode:
        candidate_assigned = candidate_schedule_df[
            (~candidate_schedule_df.get("drop_flag", pd.Series(dtype=bool)).fillna(False))
            & candidate_schedule_df.get("line", pd.Series(dtype=str)).astype(str).ne("")
        ].copy()
        scheduled_orders = int(len(candidate_assigned))
        scheduled_tons = round(float(candidate_assigned["tons"].sum()) if "tons" in candidate_assigned.columns else 0.0, 1)
        if not candidate_assigned.empty:
            big_roll_scheduled_orders = int((candidate_assigned.get("line", pd.Series(dtype=str)) == "big_roll").sum())
            small_roll_scheduled_orders = int((candidate_assigned.get("line", pd.Series(dtype=str)) == "small_roll").sum())
            edge_series = candidate_assigned.get("selected_edge_type", pd.Series(dtype=str)).astype(str)
            selected_direct_edge_count = int((edge_series == "DIRECT_EDGE").sum())
            selected_real_bridge_edge_count = int((edge_series == "REAL_BRIDGE_EDGE").sum())
            selected_virtual_bridge_edge_count = int((edge_series == "VIRTUAL_BRIDGE_EDGE").sum())
    max_big_roll_slot_order_count = _num_int(fallback_diag.get("big_roll_max_slot_order_count", joint_estimates.get("big_roll_max_slot_order_count", 0) if isinstance(joint_estimates, dict) else 0))
    max_small_roll_slot_order_count = _num_int(fallback_diag.get("small_roll_max_slot_order_count", joint_estimates.get("small_roll_max_slot_order_count", 0) if isinstance(joint_estimates, dict) else 0))

    run_summary_df = pd.DataFrame(
        [
            ("profile", str(fallback_diag.get("profile_name", "default"))),
            ("acceptance", result_acceptance_status),
            ("failure_mode", failure_mode),
            ("best_candidate_available", int(best_candidate_available)),
            ("best_candidate_type", best_candidate_type),
            ("best_candidate_objective", best_candidate_objective),
            ("best_candidate_search_status", best_candidate_search_status),
            ("best_candidate_routing_feasible", int(best_candidate_routing_feasible)),
            ("best_candidate_unroutable_slot_count", best_candidate_unroutable_slot_count),
            ("export_consistency_ok", int(export_consistency_ok)),
            ("routing_feasible", int(routing_feasible)),
            ("evidence_level", evidence_level),
            ("total_orders", int(input_order_count)),
            ("scheduled_orders", scheduled_orders),
            ("unscheduled_orders", unscheduled_orders),
            ("scheduled_tons", scheduled_tons),
            ("unscheduled_tons", unscheduled_tons),
            ("total_run_seconds", total_run_seconds),
            ("template_build_seconds", template_build_seconds),
            ("joint_master_seconds", joint_master_seconds),
            ("fallback_total_seconds", fallback_total_seconds),
        ],
        columns=["metric", "value"],
    )

    line_summary_rows = []
    slot_detail_rows = []
    # raw_slots already extracted from failure_diagnostics/engine_meta to support candidate backfill
    for line in ["big_roll", "small_roll"]:
        line_sched = final_df[final_df["line"] == line].copy() if (not final_df.empty and "line" in final_df.columns) else final_df.iloc[0:0].copy()
        real_orders = line_sched[(~line_sched["is_virtual"]) if "is_virtual" in line_sched.columns else pd.Series([True] * len(line_sched), index=line_sched.index)]
        slot_count = int(line_sched["campaign_id"].nunique()) if ("campaign_id" in line_sched.columns and not line_sched.empty) else 0
        avg_slot_order_count = float(real_orders.groupby("campaign_id").size().mean()) if ("campaign_id" in real_orders.columns and not real_orders.empty) else 0.0
        max_slot_order_count = int(real_orders.groupby("campaign_id").size().max()) if ("campaign_id" in real_orders.columns and not real_orders.empty) else 0
        unr_line = [r for r in raw_slots if str(r.get("line", "")) == line and str(r.get("status", "")) == "UNROUTABLE_SLOT"]
        edge_series = line_sched["selected_edge_type"].astype(str) if "selected_edge_type" in line_sched.columns else pd.Series(dtype=str)
        line_summary_rows.append(
            {
                "line": line,
                "scheduled_orders": int(len(real_orders)),
                "scheduled_tons": round(float(line_sched["tons"].sum()) if "tons" in line_sched.columns else 0.0, 1),
                "slot_count": slot_count,
                "avg_slot_order_count": round(avg_slot_order_count, 2),
                "max_slot_order_count": max_slot_order_count,
                "unroutable_slot_count": int(len(unr_line)),
                "direct_edge_count": int((edge_series == "DIRECT_EDGE").sum()) if not edge_series.empty else 0,
                "real_bridge_edge_count": int((edge_series == "REAL_BRIDGE_EDGE").sum()) if not edge_series.empty else 0,
                "virtual_bridge_edge_count": int((edge_series == "VIRTUAL_BRIDGE_EDGE").sum()) if not edge_series.empty else 0,
            }
        )
    line_summary_df = pd.DataFrame(
        line_summary_rows,
        columns=[
            "line", "scheduled_orders", "scheduled_tons", "slot_count", "avg_slot_order_count", "max_slot_order_count",
            "unroutable_slot_count", "direct_edge_count", "real_bridge_edge_count", "virtual_bridge_edge_count",
        ],
    )
    if final_df.empty and not candidate_line_summary_df.empty:
        line_summary_df = candidate_line_summary_df.rename(
            columns={
                "candidate_assigned_orders": "scheduled_orders",
                "candidate_assigned_tons": "scheduled_tons",
                "candidate_slot_count": "slot_count",
                "candidate_avg_slot_order_count": "avg_slot_order_count",
                "candidate_max_slot_order_count": "max_slot_order_count",
                "candidate_unroutable_slot_count": "unroutable_slot_count",
                "candidate_direct_edge_count": "direct_edge_count",
                "candidate_real_bridge_edge_count": "real_bridge_edge_count",
                "candidate_virtual_bridge_edge_count": "virtual_bridge_edge_count",
            }
        )[
            [
                "line", "scheduled_orders", "scheduled_tons", "slot_count", "avg_slot_order_count", "max_slot_order_count",
                "unroutable_slot_count", "direct_edge_count", "real_bridge_edge_count", "virtual_bridge_edge_count",
            ]
        ]
        line_summary_df["summary_mode"] = "CANDIDATE_ANALYSIS"
    else:
        line_summary_df["summary_mode"] = "OFFICIAL"

    slot_summary_df, unroutable_slots_df = _build_slot_frames(raw_slots, candidate_schedule_df)

    validation_summary = failure_diagnostics.get("validation_summary", {}) if isinstance(failure_diagnostics, dict) else {}
    hard_violation_total_official = 0
    for k in [
        "width_jump_violation_cnt",
        "thickness_violation_cnt",
        "temp_conflict_cnt",
        "non_pc_direct_switch_cnt",
        "direct_reverse_step_violation_count",
        "virtual_attach_reverse_violation_count",
        "period_reverse_count_violation_count",
        "bridge_count_violation_count",
        "invalid_virtual_spec_count",
    ]:
        try:
            hard_violation_total_official += int(validation_summary.get(k, 0) or 0)
        except Exception:
            continue
    candidate_violation_summary_df = _build_candidate_violation_summary(
        candidate_schedule_df,
        raw_slots,
        validation_summary if isinstance(validation_summary, dict) else {},
        bool(best_candidate_routing_feasible),
    )
    violation_summary_df = pd.DataFrame(
        [
            ("hard_constraint_violation_count", int(hard_violation_total_official)),
            ("direct_reverse_step_violation_count", int(validation_summary.get("direct_reverse_step_violation_count", 0))),
            ("virtual_attach_reverse_violation_count", int(validation_summary.get("virtual_attach_reverse_violation_count", 0))),
            ("period_reverse_count_violation_count", int(validation_summary.get("period_reverse_count_violation_count", 0))),
            ("bridge_count_violation_count", int(validation_summary.get("bridge_count_violation_count", 0))),
            ("invalid_virtual_spec_count", int(validation_summary.get("invalid_virtual_spec_count", 0))),
            ("unroutable_slot_count", unroutable_slot_count),
            ("hard_cap_not_enforced", bool(fallback_diag.get("hard_cap_not_enforced", False) or joint_estimates.get("hard_cap_not_enforced", False))),
            ("isolation_risk_not_effective", any(bool(r.get("isolation_risk_not_effective", False)) for r in raw_slots)),
        ],
        columns=["metric", "value"],
    )
    if final_df.empty and not candidate_violation_summary_df.empty:
        violation_summary_df = candidate_violation_summary_df.copy()

    reason_hist = dropped_reason_stats.set_index("剔除原因")["数量"].to_dict() if not dropped_reason_stats.empty else {}
    unscheduled_summary_df = pd.DataFrame(
        [
            ("unscheduled_order_count", int(len(dropped_out))),
            ("unscheduled_tons", round(float(dropped_out["tons"].sum()) if "tons" in dropped_out.columns else 0.0, 1)),
            ("dropped_reason_histogram", str(reason_hist)),
            ("globally_isolated_order_count", int(dropped_out["globally_isolated"].fillna(False).sum()) if "globally_isolated" in dropped_out.columns else 0),
            ("low_priority_drop_count", int((dropped_out.get("dominant_drop_reason", pd.Series(dtype=str)) == "LOW_PRIORITY_DROP").sum())),
            ("no_feasible_line_count", int((dropped_out.get("dominant_drop_reason", pd.Series(dtype=str)) == "NO_FEASIBLE_LINE").sum())),
            ("slot_routing_risk_too_high_count", int((dropped_out.get("dominant_drop_reason", pd.Series(dtype=str)) == "SLOT_ROUTING_RISK_TOO_HIGH").sum())),
        ],
        columns=["metric", "value"],
    )

    bridge_summary_df = pd.DataFrame(
        [
            ("selected_direct_edge_count", selected_direct_edge_count),
            ("selected_real_bridge_edge_count", selected_real_bridge_edge_count),
            ("selected_virtual_bridge_edge_count", selected_virtual_bridge_edge_count),
            ("direct_edge_ratio", float(fallback_diag.get("bridge_mix_summary", {}).get("direct_edge_ratio", 0.0) if isinstance(fallback_diag, dict) else 0.0)),
            ("real_bridge_ratio", float(fallback_diag.get("bridge_mix_summary", {}).get("real_bridge_ratio", 0.0) if isinstance(fallback_diag, dict) else 0.0)),
            ("virtual_bridge_ratio", float(fallback_diag.get("bridge_mix_summary", {}).get("virtual_bridge_ratio", 0.0) if isinstance(fallback_diag, dict) else 0.0)),
            ("max_bridge_count_used", int(validation_summary.get("max_bridge_count_used", 0))),
            ("avg_virtual_count", float(validation_summary.get("averageTemplateVirtualCount", joint_estimates.get("averageTemplateVirtualCount", 0.0) if isinstance(joint_estimates, dict) else 0.0) or 0.0)),
        ],
        columns=["metric", "value"],
    )

    progress_metrics_df = pd.DataFrame(
        [[
            str(fallback_diag.get("profile_name", "default")),
            result_acceptance_status,
            failure_mode,
            int(best_candidate_available),
            best_candidate_type,
            int(export_consistency_ok),
            int(routing_feasible),
            scheduled_orders,
            unscheduled_orders,
            _num_int(len(dropped_out)),
            unscheduled_tons,
            big_roll_scheduled_orders,
            small_roll_scheduled_orders,
            unroutable_slot_count,
            max_big_roll_slot_order_count,
            max_small_roll_slot_order_count,
            selected_direct_edge_count,
            selected_real_bridge_edge_count,
            selected_virtual_bridge_edge_count,
            int(validation_summary.get("direct_reverse_step_violation_count", 0)),
            int(validation_summary.get("virtual_attach_reverse_violation_count", 0)),
            int(validation_summary.get("period_reverse_count_violation_count", 0)),
            int(validation_summary.get("invalid_virtual_spec_count", 0)),
            (pd.NA if candidate_backfill_mode else int(hard_violation_total_official)),
            template_build_seconds,
            joint_master_seconds,
            fallback_total_seconds,
            total_run_seconds,
        ]],
        columns=[
            "profile", "acceptance", "failure_mode", "best_candidate_available", "best_candidate_type",
            "export_consistency_ok",
            "routing_feasible", "scheduled_orders", "unscheduled_orders", "dropped_order_count", "dropped_tons",
            "big_roll_scheduled_orders", "small_roll_scheduled_orders", "unroutable_slot_count",
            "max_big_roll_slot_order_count", "max_small_roll_slot_order_count",
            "selected_direct_edge_count", "selected_real_bridge_edge_count", "selected_virtual_bridge_edge_count",
            "direct_reverse_step_violation_count", "virtual_attach_reverse_violation_count",
            "period_reverse_count_violation_count", "invalid_virtual_spec_count",
            "hard_constraint_violation_count",
            "template_build_seconds", "joint_master_seconds", "fallback_total_seconds", "total_run_seconds",
        ],
    )

    drop_and_bridge_details_df = pd.DataFrame(columns=[
        "order_id", "dropped", "dominant_drop_reason", "secondary_reasons", "globally_isolated",
        "candidate_lines", "risk_summary", "selected_edge_type", "bridge_count", "bridge_type",
        "virtual_widths_used", "virtual_thicknesses_used",
    ])
    detail_rows = []
    if not final_df.empty:
        for _, row in final_df.iterrows():
            is_virtual = bool(row.get("is_virtual", False))
            detail_rows.append(
                {
                    "order_id": row.get("order_id", ""),
                    "dropped": False,
                    "dominant_drop_reason": "",
                    "secondary_reasons": "",
                    "globally_isolated": False,
                    "candidate_lines": "",
                    "risk_summary": "",
                    "selected_edge_type": row.get("selected_edge_type", ""),
                    "bridge_count": int(row.get("bridge_count", 0) or 0),
                    "bridge_type": row.get("selected_edge_type", ""),
                    "virtual_widths_used": row.get("width", "") if is_virtual else "",
                    "virtual_thicknesses_used": row.get("thickness", "") if is_virtual else "",
                }
            )
    if not dropped_out.empty:
        for _, row in dropped_out.iterrows():
            detail_rows.append(
                {
                    "order_id": row.get("order_id", ""),
                    "dropped": True,
                    "dominant_drop_reason": row.get("dominant_drop_reason", row.get("drop_reason", "")),
                    "secondary_reasons": row.get("secondary_reasons", ""),
                    "globally_isolated": bool(row.get("globally_isolated", False)),
                    "candidate_lines": row.get("candidate_lines", ""),
                    "risk_summary": row.get("risk_summary", ""),
                    "selected_edge_type": "",
                    "bridge_count": 0,
                    "bridge_type": "",
                    "virtual_widths_used": "",
                    "virtual_thicknesses_used": "",
                }
            )
    if detail_rows:
        drop_and_bridge_details_df = pd.DataFrame(detail_rows)

    drop_and_bridge_summary_df = pd.DataFrame(
        [
            ("dropped_order_count", int(len(dropped_out))),
            ("selected_direct_edge_count", int(fallback_diag.get("bridge_mix_summary", {}).get("selected_direct_edge_count", 0) if isinstance(fallback_diag, dict) else 0)),
            ("selected_real_bridge_edge_count", int(fallback_diag.get("bridge_mix_summary", {}).get("selected_real_bridge_edge_count", 0) if isinstance(fallback_diag, dict) else 0)),
            ("selected_virtual_bridge_edge_count", int(fallback_diag.get("bridge_mix_summary", {}).get("selected_virtual_bridge_edge_count", 0) if isinstance(fallback_diag, dict) else 0)),
            ("max_bridge_count_used", int((failure_diagnostics.get("validation_summary", {}) if isinstance(failure_diagnostics, dict) else {}).get("max_bridge_count_used", 0))),
            ("bridge_count_violation_count", int((failure_diagnostics.get("validation_summary", {}) if isinstance(failure_diagnostics, dict) else {}).get("bridge_count_violation_count", 0))),
            ("invalid_virtual_spec_count", int((failure_diagnostics.get("validation_summary", {}) if isinstance(failure_diagnostics, dict) else {}).get("invalid_virtual_spec_count", 0))),
            ("dropped_reason_histogram", str((dropped_reason_stats.set_index("剔除原因")["数量"].to_dict() if not dropped_reason_stats.empty else {}))),
        ],
        columns=["metric", "value"],
    )
    candidate_big_roll_out = candidate_big_roll_df.copy() if isinstance(candidate_big_roll_df, pd.DataFrame) else pd.DataFrame()
    candidate_small_roll_out = candidate_small_roll_df.copy() if isinstance(candidate_small_roll_df, pd.DataFrame) else pd.DataFrame()
    if candidate_big_roll_out.empty:
        candidate_big_roll_out = candidate_schedule_df[candidate_schedule_df.get("line", pd.Series(dtype=str)) == "big_roll"].copy() if not candidate_schedule_df.empty else pd.DataFrame()
    if candidate_small_roll_out.empty:
        candidate_small_roll_out = candidate_schedule_df[candidate_schedule_df.get("line", pd.Series(dtype=str)) == "small_roll"].copy() if not candidate_schedule_df.empty else pd.DataFrame()

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        # Main schedule sheets: keep official naming, but translate candidate drafts when backfilled.
        if candidate_backfill_mode:
            big_sheet_out = big_sheet.rename(
                columns={
                    "order_id": "物料号",
                    "slot_no": "槽位号",
                    "candidate_position": "槽位内序号",
                    "candidate_slot_member_index": "槽位内成员序号",
                    "tons": "吨位",
                    "width": "宽度",
                    "thickness": "厚度",
                    "steel_group": "钢种组",
                    "line_capability": "产线能力",
                    "candidate_status": "候选状态",
                    "slot_unroutable_flag": "槽不可路由标记",
                    "slot_route_risk_score": "槽路由风险",
                    "selected_edge_type": "选中边类型",
                    "analysis_only": "仅分析",
                    "official_usable": "可下发",
                    "result_usage": "结果用途",
                }
            )
            if "候选状态" in big_sheet_out.columns:
                big_sheet_out["候选状态"] = big_sheet_out["候选状态"].map(lambda v: _zh_enum(v, _CANDIDATE_STATUS_ZH))
            if "选中边类型" in big_sheet_out.columns:
                big_sheet_out["选中边类型"] = big_sheet_out["选中边类型"].map(lambda v: _zh_enum(v, _EDGE_TYPE_ZH))
            small_sheet_out = small_sheet.rename(
                columns={
                    "order_id": "物料号",
                    "slot_no": "槽位号",
                    "candidate_position": "槽位内序号",
                    "candidate_slot_member_index": "槽位内成员序号",
                    "tons": "吨位",
                    "width": "宽度",
                    "thickness": "厚度",
                    "steel_group": "钢种组",
                    "line_capability": "产线能力",
                    "candidate_status": "候选状态",
                    "slot_unroutable_flag": "槽不可路由标记",
                    "slot_route_risk_score": "槽路由风险",
                    "selected_edge_type": "选中边类型",
                    "analysis_only": "仅分析",
                    "official_usable": "可下发",
                    "result_usage": "结果用途",
                }
            )
            if "候选状态" in small_sheet_out.columns:
                small_sheet_out["候选状态"] = small_sheet_out["候选状态"].map(lambda v: _zh_enum(v, _CANDIDATE_STATUS_ZH))
            if "选中边类型" in small_sheet_out.columns:
                small_sheet_out["选中边类型"] = small_sheet_out["选中边类型"].map(lambda v: _zh_enum(v, _EDGE_TYPE_ZH))
        else:
            big_sheet_out = big_sheet
            small_sheet_out = small_sheet
        big_sheet_out.to_excel(writer, sheet_name="大辊线排程", index=False)
        small_sheet_out.to_excel(writer, sheet_name="小辊线排程", index=False)
        campaign_tons_df.to_excel(writer, sheet_name="轧期吨位统计", index=False)
        rounds_out.rename(columns={"round": "轮次", "line": "产线代码"}).to_excel(writer, sheet_name="轮次统计", index=False)
        summary_df.to_excel(writer, sheet_name="总览指标", index=False)
        flow_df.to_excel(writer, sheet_name="订单去向", index=False)
        dropped_out.to_excel(writer, sheet_name="剔除订单", index=False)
        dropped_reason_stats.to_excel(writer, sheet_name="剔除原因统计", index=False)
        runtime_report.to_excel(writer, sheet_name="标准化运行报表", index=False)
        diagnostics_report.rename(columns={"section": "分区", "metric": "指标", "value": "值"}).to_excel(writer, sheet_name="诊断摘要", index=False)
        equivalence_report.to_excel(writer, sheet_name="等价性检查", index=False)
        diagnostics_snapshot.rename(columns={"key": "字段", "value": "值"}).to_excel(writer, sheet_name="求解路径快照", index=False)
        rule_snapshot.rename(
            columns={
                "rule_key": "规则键",
                "zh_name": "中文名",
                "en_name": "英文名",
                "solver_active": "求解启用",
                "validation_active": "校验启用",
                "export_visible": "导出可见",
            }
        ).to_excel(writer, sheet_name="规则快照", index=False)
        _kv_to_zh(precheck_summary_df).to_excel(writer, sheet_name="数据预检汇总", index=False)
        _kv_to_zh(feasibility_evidence_df).to_excel(writer, sheet_name="不可行证据", index=False)
        template_summary_df.assign(产线=template_summary_df.get("line", "").map(_zh_line)).rename(
            columns={
                "line": "产线代码",
                "candidate_pairs": "候选pair数",
                "templates_built": "模板数",
                "template_coverage_ratio": "模板覆盖率",
                "avg_virtual_count": "平均虚拟段数",
                "max_virtual_count": "最大虚拟段数",
                "rejected_by_chain_limit": "链长拒绝数",
                "direct_edge_count": "直接边数",
                "real_bridge_edge_count": "实物桥接边数",
                "virtual_bridge_edge_count": "虚拟桥接边数",
            }
        )[[
            "产线", "产线代码", "候选pair数", "模板数", "模板覆盖率", "平均虚拟段数", "最大虚拟段数",
            "链长拒绝数", "直接边数", "实物桥接边数", "虚拟桥接边数",
        ]].to_excel(writer, sheet_name="模板汇总", index=False)
        master_allocation_summary_df.assign(产线=master_allocation_summary_df.get("line", "").map(_zh_line)).rename(
            columns={
                "line": "产线代码",
                "slot_no": "槽位号",
                "slot_order_count": "槽内订单数",
                "slot_tons": "槽内吨位",
                "order_count_over_cap": "超cap订单数",
                "slot_route_risk_score": "槽路由风险",
                "pair_gap_proxy": "pair缺口风险",
                "span_risk": "跨度风险",
                "degree_risk": "低度数风险",
                "isolated_order_penalty": "孤点惩罚",
                "selected_bridge_mix": "桥接mix",
            }
        )[[
            "产线", "产线代码", "槽位号", "槽内订单数", "槽内吨位", "超cap订单数",
            "槽路由风险", "pair缺口风险", "跨度风险", "低度数风险", "孤点惩罚", "桥接mix",
        ]].to_excel(writer, sheet_name="主模型分配汇总", index=False)
        unroutable_slots_df.assign(产线=unroutable_slots_df.get("line", "").map(_zh_line)).rename(
            columns={
                "line": "产线代码",
                "slot_no": "槽位号",
                "order_count": "槽内订单数",
                "template_coverage_ratio": "模板覆盖率",
                "missing_template_edge_count": "缺失模板边数",
                "zero_in_orders": "零入度订单数",
                "zero_out_orders": "零出度订单数",
                "width_span": "宽度跨度",
                "thickness_span": "厚度跨度",
                "steel_group_count": "钢种组数量",
                "slot_route_risk_score": "槽路由风险",
                "dominant_unroutable_reason": "主不可路由原因",
                "top_isolated_orders": "孤点订单Top",
            }
        )[[
            "产线", "产线代码", "槽位号", "槽内订单数", "模板覆盖率", "缺失模板边数",
            "零入度订单数", "零出度订单数", "宽度跨度", "厚度跨度", "钢种组数量",
            "槽路由风险", "主不可路由原因", "孤点订单Top",
        ]].to_excel(writer, sheet_name="不可路由槽位", index=False)
        _kv_to_zh(drop_and_bridge_summary_df).to_excel(writer, sheet_name="剔除与桥接汇总", index=False)
        run_summary_zh = _kv_to_zh(run_summary_df)
        if "指标" in run_summary_zh.columns and "值" in run_summary_zh.columns:
            run_summary_zh["指标"] = run_summary_zh["指标"].map(
                {
                    "profile": "Profile",
                    "acceptance": "结果类型",
                    "failure_mode": "失败模式",
                    "best_candidate_available": "是否有最佳候选",
                    "best_candidate_type": "最佳候选类型",
                    "best_candidate_objective": "最佳候选目标值",
                    "best_candidate_search_status": "最佳候选状态",
                    "best_candidate_routing_feasible": "最佳候选路由可行",
                    "best_candidate_unroutable_slot_count": "最佳候选不可路由槽位数",
                    "export_consistency_ok": "导出口径一致",
                    "routing_feasible": "路由可行",
                    "evidence_level": "证据等级",
                    "total_orders": "总订单数",
                    "scheduled_orders": "已排订单数",
                    "unscheduled_orders": "未排订单数",
                    "scheduled_tons": "已排吨位",
                    "unscheduled_tons": "未排吨位",
                    "total_run_seconds": "总耗时(s)",
                    "template_build_seconds": "模板耗时(s)",
                    "joint_master_seconds": "主模型耗时(s)",
                    "fallback_total_seconds": "fallback耗时(s)",
                }
            ).fillna(run_summary_zh["指标"])
            def _zh_run_value(k: str, v: object) -> object:
                if k in {"结果类型", "acceptance"}:
                    return _zh_enum(v, _ACCEPTANCE_ZH)
                if k in {"失败模式", "failure_mode"}:
                    return _zh_enum(v, _FAILURE_MODE_ZH)
                return v
            run_summary_zh["值"] = [
                _zh_run_value(str(k), v) for k, v in zip(run_summary_zh["指标"].tolist(), run_summary_zh["值"].tolist())
            ]
        run_summary_zh.to_excel(writer, sheet_name="运行汇总", index=False)
        line_summary_df.assign(产线=line_summary_df.get("line", "").map(_zh_line)).rename(
            columns={
                "line": "产线代码",
                "scheduled_orders": "已排订单数",
                "scheduled_tons": "已排吨位",
                "slot_count": "槽位数",
                "avg_slot_order_count": "平均槽位订单数",
                "max_slot_order_count": "最大槽位订单数",
                "unroutable_slot_count": "不可路由槽位数",
                "direct_edge_count": "直接边数",
                "real_bridge_edge_count": "实物桥接边数",
                "virtual_bridge_edge_count": "虚拟桥接边数",
                "summary_mode": "统计口径",
            }
        ).assign(**{"统计口径": line_summary_df.get("summary_mode", "").map(lambda v: _zh_enum(v, _SUMMARY_MODE_ZH))})[[
            "产线", "产线代码", "已排订单数", "已排吨位", "槽位数", "平均槽位订单数", "最大槽位订单数",
            "不可路由槽位数", "直接边数", "实物桥接边数", "虚拟桥接边数", "统计口径",
        ]].to_excel(writer, sheet_name="产线汇总", index=False)
        violation_out = _kv_to_zh(violation_summary_df) if list(violation_summary_df.columns) == ["metric", "value"] else violation_summary_df.copy()
        if list(violation_out.columns) == ["metric", "official_value", "candidate_value", "violation_count_source", "violation_count_confidence"]:
            violation_out = violation_out.rename(
                columns={
                    "metric": "指标",
                    "official_value": "正式值",
                    "candidate_value": "候选值",
                    "violation_count_source": "统计来源",
                    "violation_count_confidence": "置信度",
                }
            )
            violation_out["统计来源"] = violation_out["统计来源"].map(lambda v: _zh_enum(v, _VIOLATION_SRC_ZH))
            violation_out["置信度"] = violation_out["置信度"].map(lambda v: _zh_enum(v, _CONFIDENCE_ZH))
        else:
            violation_out = _kv_to_zh(violation_out).rename(columns={"指标": "指标", "值": "值"})
            if "指标" in violation_out.columns:
                violation_out["指标"] = violation_out["指标"].map(
                    {
                        "hard_constraint_violation_count": "硬约束违例总数",
                        "direct_reverse_step_violation_count": "直接逆宽违例数(>20)",
                        "virtual_attach_reverse_violation_count": "虚拟接入逆宽违例数(>250)",
                        "period_reverse_count_violation_count": "辊期逆宽次数违例数(>5)",
                        "bridge_count_violation_count": "桥链长度违例数(>5)",
                        "invalid_virtual_spec_count": "虚拟规格违例数",
                        "unroutable_slot_count": "不可路由槽位数",
                        "hard_cap_not_enforced": "hard cap 未生效",
                        "isolation_risk_not_effective": "孤点风险未生效",
                    }
                ).fillna(violation_out["指标"])
        violation_out.to_excel(writer, sheet_name="违规汇总", index=False)
        _kv_to_zh(unscheduled_summary_df).to_excel(writer, sheet_name="未排汇总", index=False)
        _kv_to_zh(bridge_summary_df).to_excel(writer, sheet_name="桥接汇总", index=False)
        slot_summary_df.assign(产线=slot_summary_df.get("line", "").map(_zh_line)).rename(
            columns={
                "line": "产线代码",
                "slot_no": "槽位号",
                "slot_order_count": "槽内订单数",
                "slot_tons": "槽内吨位",
                "order_count_over_cap": "超cap订单数",
                "slot_route_risk_score": "槽路由风险",
                "pair_gap_proxy": "pair缺口风险",
                "span_risk": "跨度风险",
                "degree_risk": "低度数风险",
                "isolated_order_penalty": "孤点惩罚",
                "dominant_unroutable_reason": "主不可路由原因",
            }
        )[[
            "产线", "产线代码", "槽位号", "槽内订单数", "槽内吨位", "超cap订单数",
            "槽路由风险", "pair缺口风险", "跨度风险", "低度数风险", "孤点惩罚", "主不可路由原因",
        ]].to_excel(writer, sheet_name="槽位汇总", index=False)
        drop_and_bridge_details_df.rename(
            columns={
                "order_id": "物料号",
                "dropped": "是否剔除",
                "dominant_drop_reason": "主剔除原因",
                "secondary_reasons": "次要原因",
                "globally_isolated": "全局孤点",
                "candidate_lines": "候选产线",
                "risk_summary": "风险摘要",
                "selected_edge_type": "选中边类型",
                "bridge_count": "桥段数",
                "bridge_type": "桥接类型",
                "virtual_widths_used": "虚拟宽度使用",
                "virtual_thicknesses_used": "虚拟厚度使用",
            }
        ).to_excel(writer, sheet_name="剔除与桥接明细", index=False)
        progress_out = progress_metrics_df.copy()
        progress_out = progress_out.rename(
            columns={
                "profile": "Profile",
                "acceptance": "结果类型",
                "failure_mode": "失败模式",
                "best_candidate_available": "是否有最佳候选",
                "best_candidate_type": "最佳候选类型",
                "export_consistency_ok": "导出口径一致",
                "routing_feasible": "路由可行",
                "scheduled_orders": "已排订单数",
                "unscheduled_orders": "未排订单数",
                "dropped_order_count": "剔除订单数",
                "dropped_tons": "剔除吨位",
                "big_roll_scheduled_orders": "大辊已排订单数",
                "small_roll_scheduled_orders": "小辊已排订单数",
                "unroutable_slot_count": "不可路由槽位数",
                "max_big_roll_slot_order_count": "大辊最大槽订单数",
                "max_small_roll_slot_order_count": "小辊最大槽订单数",
                "selected_direct_edge_count": "直接边数(选中)",
                "selected_real_bridge_edge_count": "实物桥接边数(选中)",
                "selected_virtual_bridge_edge_count": "虚拟桥接边数(选中)",
                "direct_reverse_step_violation_count": "直接逆宽违例数(>20)",
                "virtual_attach_reverse_violation_count": "虚拟接入逆宽违例数(>250)",
                "period_reverse_count_violation_count": "辊期逆宽次数违例数(>5)",
                "invalid_virtual_spec_count": "虚拟规格违例数",
                "hard_constraint_violation_count": "硬约束违例总数",
                "template_build_seconds": "模板耗时(s)",
                "joint_master_seconds": "主模型耗时(s)",
                "fallback_total_seconds": "fallback耗时(s)",
                "total_run_seconds": "总耗时(s)",
            }
        )
        if "结果类型" in progress_out.columns:
            progress_out["结果类型"] = progress_out["结果类型"].map(lambda v: _zh_enum(v, _ACCEPTANCE_ZH))
        if "失败模式" in progress_out.columns:
            progress_out["失败模式"] = progress_out["失败模式"].map(lambda v: _zh_enum(v, _FAILURE_MODE_ZH))
        progress_out.to_excel(writer, sheet_name="进展指标", index=False)
        if best_candidate_available:
            candidate_big_roll_out.to_excel(writer, sheet_name="大辊线候选排程", index=False)
            candidate_small_roll_out.to_excel(writer, sheet_name="小辊线候选排程", index=False)
            candidate_line_summary_df.to_excel(writer, sheet_name="产线汇总_候选", index=False)
            candidate_violation_summary_df.to_excel(writer, sheet_name="违规汇总_候选", index=False)
        _autosize_excel(writer)

    final_df.to_csv(out_path.with_suffix(".csv"), index=False, encoding="utf-8-sig")
    elapsed = (perf_counter() - t_start) if t_start is not None else None
    if elapsed is None:
        print(f"[APS] 排程完成, 引擎={engine_used}, 输出={out_path}, result_usage={result_usage}, official_exported={official_exported}, analysis_exported={analysis_exported}")
    else:
        print(f"[APS] 排程完成, 引擎={engine_used}, 总耗时={elapsed:.1f}s, 输出={out_path}, result_usage={result_usage}, official_exported={official_exported}, analysis_exported={analysis_exported}")
    if best_candidate_available:
        print(
            f"[APS][导出一致性] acceptance={result_acceptance_status}, failure_mode={failure_mode}, "
            f"result_usage={result_usage}, best_candidate_available={int(best_candidate_available)}, "
            f"best_candidate_type={best_candidate_type}, best_candidate_search_status={best_candidate_search_status}, "
            f"export_consistency_ok={export_consistency_ok}"
        )
    if str(result_usage) == "ANALYSIS_ONLY" and best_candidate_available:
        print(
            f"[APS][候选导出] candidate_schedule_exported={not candidate_schedule_df.empty}, "
            f"candidate_line_summary_exported={not candidate_line_summary_df.empty}, "
            f"candidate_violation_summary_exported={not candidate_violation_summary_df.empty}, "
            "ANALYSIS_ONLY, NOT_ROUTING_FEASIBLE, 不可下发"
        )
    return final_df, rounds_df
