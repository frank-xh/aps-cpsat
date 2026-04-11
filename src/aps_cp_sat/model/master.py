from __future__ import annotations

"""
CP-SAT master orchestration layer.

Production architecture contract:
- `_run_global_joint_model(...)` is the only production master entry.
- `_solve_slot_route_with_templates(...)` is only a transitional slot-local
  component that may refine sequence inside a slot after the joint master has
  already fixed line/slot assignment.
- Virtual slabs remain template-arc attributes and must not be expanded into
  explicit synthetic order nodes inside the master model.
- `decode / validate / export` only interpret solved results.
- `repair` remains no-op / recording and never changes feasibility semantics.

Rule semantics snapshot:
- line compatibility: hard constraint
- adjacent transition feasibility: hard semantics via template filtering
- campaign ton upper bound: hard constraint
- campaign ton lower bound: strong soft constraint
- unassigned real orders: strong soft constraint
- virtual slab ratio / quantity: strong soft constraint
- reverse-width count / total rise: hard semantics in template rules
"""

import pandas as pd
from time import perf_counter
from dataclasses import replace

from aps_cp_sat.compat import run_legacy_schedule
from aps_cp_sat.domain.models import ColdRollingRequest
from aps_cp_sat.model.diagnostics import _assess_template_graph_health, _build_joint_failure_diagnostics
from aps_cp_sat.model.feasibility_evidence import build_feasibility_evidence
from aps_cp_sat.model.fallback_policy import (
    _effective_global_prune_cap,
    _joint_profiles,
    _joint_seeds,
    _legacy_fallback_enabled,
    _precheck_relaxed_config,
    _scale_down_fallback_enabled,
    _semantic_fallback_configs,
    _semantic_fallback_enabled,
)
from aps_cp_sat.model.joint_master import _run_global_joint_model, _run_unified_master_skeleton
from aps_cp_sat.model.local_router import _solve_slot_route_with_templates
from aps_cp_sat.transition import build_transition_templates


def _is_diagnostic_profile(profile_name: str) -> bool:
    profile = str(profile_name or "").lower()
    return profile in {"feasibility_slot_diagnostic", "feasibility_fast_slot_safe"}


def _should_escalate_evidence_failure(profile_name: str, evidence_level: str) -> bool:
    level = str(evidence_level or "OK")
    if level == "STRONG_INFEASIBLE_SIGNAL":
        return True
    return _is_diagnostic_profile(profile_name) and level == "WARNING"


def _should_fast_fail_after_routing_infeasible(joint: dict, cfg) -> tuple[bool, str]:
    if not bool(getattr(cfg.model, "fast_fail_on_bad_slots", False)):
        return False, ""
    unroutable_cnt = int(joint.get("unroutable_slot_count", 0) or 0)
    if unroutable_cnt >= int(cfg.model.fast_fail_unroutable_slot_threshold):
        return True, f"unroutable_slot_count>={int(cfg.model.fast_fail_unroutable_slot_threshold)}"
    raw_slots = joint.get("slot_route_details", [])
    if not isinstance(raw_slots, list) or not raw_slots:
        return False, ""
    topn = int(max(1, cfg.model.fast_fail_topn_slots))
    bad_slots = [s for s in raw_slots if str(s.get("status", "")) == "UNROUTABLE_SLOT"][:topn]
    if not bad_slots:
        return False, ""
    for slot in bad_slots:
        order_count = int(slot.get("order_count", 0) or 0)
        coverage = float(slot.get("template_coverage_ratio", 1.0) or 1.0)
        zero_degree = int(slot.get("zero_in_orders", 0) or 0) + int(slot.get("zero_out_orders", 0) or 0)
        if (
            order_count >= int(cfg.model.fast_fail_slot_order_count_threshold)
            and coverage <= float(cfg.model.fast_fail_slot_coverage_threshold)
        ):
            return True, "top_unroutable_slot_overpacked_and_low_coverage"
        if zero_degree >= int(cfg.model.fast_fail_slot_zero_degree_threshold):
            return True, "top_unroutable_slot_high_zero_degree"
    return False, ""


def _structure_fallback_enabled(cfg) -> bool:
    return bool(getattr(cfg.model, "enableStructureFallback", False))


def _pick_structure_drop_candidates(
    orders_df: pd.DataFrame,
    joint: dict,
    feasibility_evidence: dict,
    cfg,
) -> pd.DataFrame:
    if orders_df is None or orders_df.empty:
        return pd.DataFrame()
    max_drop_count = max(0, int(getattr(cfg.model, "max_drop_count_for_partial", 0) or 0))
    max_drop_ratio = max(0.0, float(getattr(cfg.model, "max_drop_ratio_for_partial", 0.0) or 0.0))
    budget = min(max_drop_count if max_drop_count > 0 else len(orders_df), max(1, int(len(orders_df) * max_drop_ratio))) if max_drop_ratio > 0 else max_drop_count
    if budget <= 0:
        return pd.DataFrame()

    orders = orders_df.copy()
    by_id = {str(r["order_id"]): r for r in orders.to_dict("records")}
    scored: dict[str, dict] = {}

    def _allowed_lines(row: dict) -> list[str]:
        cap = str(row.get("line_capability", "dual") or "dual").lower()
        if cap in {"big_only", "large"}:
            return ["big_roll"]
        if cap in {"small_only", "small"}:
            return ["small_roll"]
        return ["big_roll", "small_roll"]

    def add_candidate(
        order_id: str,
        score: int,
        reason: str,
        secondary: str = "",
        *,
        slot: dict | None = None,
        risk_summary: str = "",
        would_break_slot_if_kept: bool = False,
        globally_isolated: bool = False,
    ) -> None:
        oid = str(order_id or "")
        if not oid or oid not in by_id:
            return
        row = scored.setdefault(
            oid,
            {
                "score": -1,
                "reasons": [],
                "secondary": [],
                "order": by_id[oid],
                "slot": {},
                "risk_summary": "",
                "would_break_slot_if_kept": False,
                "globally_isolated": False,
            },
        )
        row["score"] = max(int(score), int(row["score"]))
        if reason and reason not in row["reasons"]:
            row["reasons"].append(reason)
        if secondary and secondary not in row["secondary"]:
            row["secondary"].append(secondary)
        if isinstance(slot, dict) and slot:
            row["slot"] = {
                "line": str(slot.get("line", "")),
                "slot_no": int(slot.get("slot_no", 0) or 0),
                "order_count": int(slot.get("order_count", 0) or 0),
                "order_count_over_cap": int(slot.get("order_count_over_cap", 0) or 0),
                "slot_route_risk_score": int(slot.get("slot_route_risk_score", 0) or 0),
                "dominant_unroutable_reason": str(slot.get("dominant_unroutable_reason", "")),
            }
        if risk_summary:
            row["risk_summary"] = str(risk_summary)
        row["would_break_slot_if_kept"] = bool(row["would_break_slot_if_kept"] or would_break_slot_if_kept)
        row["globally_isolated"] = bool(row["globally_isolated"] or globally_isolated)

    isolated_top = feasibility_evidence.get("isolated_orders_topn", []) if isinstance(feasibility_evidence, dict) else []
    isolated_lookup = {str(item.get("order_id", "")): item for item in isolated_top}
    for item in isolated_top:
        add_candidate(
            str(item.get("order_id", "")),
            140,
            "GLOBAL_ISOLATED_ORDER",
            "LOW_DEGREE_ORDER",
            risk_summary="global_isolated",
            would_break_slot_if_kept=True,
            globally_isolated=True,
        )

    raw_slots = joint.get("slot_route_details", []) if isinstance(joint.get("slot_route_details", []), list) else []
    bad_slots = [slot for slot in raw_slots if str(slot.get("status", "")) == "UNROUTABLE_SLOT"]
    bad_slots.sort(
        key=lambda slot: (
            0 if str(slot.get("line", "")) == "big_roll" else 1,
            -int(slot.get("order_count_over_cap", 0) or 0),
            -int(slot.get("slot_route_risk_score", 0) or 0),
            float(slot.get("template_coverage_ratio", 1.0) or 1.0),
            -int(slot.get("zero_in_orders", 0) or 0) - int(slot.get("zero_out_orders", 0) or 0),
        )
    )
    focus_bad_slots = bad_slots[: max(1, min(len(bad_slots), max(3, budget // 4 or 1)))]
    for slot_idx, slot in enumerate(focus_bad_slots):
        slot_ids = [str(v) for v in (slot.get("order_ids", []) or []) if str(v or "") in by_id]
        if not slot_ids:
            continue
        slot_rows = [by_id[oid] for oid in slot_ids]
        slot_df = pd.DataFrame(slot_rows)
        width_median = float(slot_df["width"].median()) if "width" in slot_df.columns and not slot_df.empty else 0.0
        thick_median = float(slot_df["thickness"].median()) if "thickness" in slot_df.columns and not slot_df.empty else 0.0
        over_cap = int(slot.get("order_count_over_cap", 0) or 0)
        zero_degree = int(slot.get("zero_in_orders", 0) or 0) + int(slot.get("zero_out_orders", 0) or 0)
        target_remove = max(1, min(max(1, over_cap), max(1, budget // max(1, len(focus_bad_slots)))))
        if str(slot.get("line", "")) == "big_roll":
            target_remove = max(target_remove, 2)
        slot_risk_hint = (
            f"slot={slot.get('line', '')}:{int(slot.get('slot_no', 0) or 0)}|"
            f"coverage={float(slot.get('template_coverage_ratio', 0.0) or 0.0):.3f}|"
            f"pair_gap={int(slot.get('pair_gap_proxy', 0) or 0)}|"
            f"span={int(slot.get('span_risk', 0) or 0)}|"
            f"zero_degree={zero_degree}"
        )

        for oid in list(dict.fromkeys(slot.get("top_isolated_orders", []) or [])):
            add_candidate(
                str(oid),
                130 + max(0, 12 - slot_idx),
                "SLOT_ROUTING_RISK_TOO_HIGH",
                "LOW_DEGREE_ORDER",
                slot=slot,
                risk_summary=slot_risk_hint,
                would_break_slot_if_kept=True,
                globally_isolated=str(oid) in isolated_lookup,
            )

        ranked_rows = []
        for rec in slot_rows:
            oid = str(rec.get("order_id", ""))
            width_outlier = abs(float(rec.get("width", width_median) or width_median) - width_median)
            thick_outlier = abs(float(rec.get("thickness", thick_median) or thick_median) - thick_median)
            low_priority = int(rec.get("priority", 0) or 0) <= 0
            due_rank = int(rec.get("due_rank", 9) or 9)
            iso = oid in set(str(v) for v in (slot.get("top_isolated_orders", []) or []))
            global_iso = oid in isolated_lookup
            ranked_rows.append(
                (
                    oid,
                    {
                        "iso": iso,
                        "global_iso": global_iso,
                        "low_priority": low_priority,
                        "due_rank": due_rank,
                        "width_outlier": float(width_outlier),
                        "thick_outlier": float(thick_outlier),
                        "tons": float(rec.get("tons", 0.0) or 0.0),
                        "line_capability": str(rec.get("line_capability", "dual") or "dual"),
                    },
                )
            )
        ranked_rows.sort(
            key=lambda item: (
                0 if item[1]["iso"] else 1,
                0 if item[1]["global_iso"] else 1,
                0 if item[1]["low_priority"] else 1,
                -item[1]["width_outlier"],
                -item[1]["thick_outlier"],
                -item[1]["due_rank"],
                item[1]["tons"],
            )
        )
        for oid, feats in ranked_rows[: max(target_remove, 1) * 2]:
            secondary = []
            if feats["iso"] or feats["global_iso"]:
                secondary.append("LOW_DEGREE_ORDER")
            if over_cap > 0:
                secondary.append("OVERPACKED_SLOT")
            if float(slot.get("pair_gap_proxy", 0) or 0) > 0:
                secondary.append("PAIR_GAP_CLUSTER")
            if float(slot.get("span_risk", 0) or 0) > 0:
                secondary.append("SPAN_OUTLIER")
            dominant = "SLOT_ROUTING_RISK_TOO_HIGH" if (feats["iso"] or feats["global_iso"]) else "CAPACITY_PRESSURE"
            score = 95
            if feats["global_iso"]:
                score += 20
            if feats["iso"]:
                score += 15
            if feats["low_priority"]:
                score += 8
            score += min(20, int(float(slot.get("order_count_over_cap", 0) or 0)))
            score += min(15, int(float(slot.get("pair_gap_proxy", 0) or 0) // 10))
            score += min(10, int(float(slot.get("span_risk", 0) or 0) // 10))
            add_candidate(
                oid,
                score,
                dominant,
                ",".join(secondary),
                slot=slot,
                risk_summary=slot_risk_hint,
                would_break_slot_if_kept=True,
                globally_isolated=bool(feats["global_iso"]),
            )

    for oid, row in by_id.items():
        allowed = _allowed_lines(row)
        if not allowed:
            add_candidate(oid, 120, "NO_FEASIBLE_LINE", risk_summary="no_feasible_line", would_break_slot_if_kept=False)

    if not scored:
        return pd.DataFrame()

    picked = sorted(
        scored.values(),
        key=lambda item: (
            -int(item["score"]),
            int(item["order"].get("priority", 0) or 0),
            -int(item["order"].get("due_rank", 9) or 9),
            float(item["order"].get("tons", 0.0) or 0.0),
        ),
    )[:budget]
    rows = []
    for item in picked:
        rec = dict(item["order"])
        rec["drop_reason"] = item["reasons"][0] if item["reasons"] else "OTHER"
        rec["secondary_reasons"] = ",".join(item["secondary"])
        rec["drop_candidate_score"] = int(item["score"])
        slot = item.get("slot", {}) if isinstance(item.get("slot"), dict) else {}
        rec["source_bad_slot_line"] = str(slot.get("line", ""))
        rec["source_bad_slot_no"] = int(slot.get("slot_no", 0) or 0)
        rec["source_bad_slot_order_count"] = int(slot.get("order_count", 0) or 0)
        rec["source_bad_slot_over_cap"] = int(slot.get("order_count_over_cap", 0) or 0)
        rec["source_bad_slot_risk"] = int(slot.get("slot_route_risk_score", 0) or 0)
        rec["source_bad_slot_reason"] = str(slot.get("dominant_unroutable_reason", ""))
        rec["globally_isolated"] = bool(item.get("globally_isolated", False))
        rec["risk_summary"] = str(item.get("risk_summary", ""))
        rec["would_break_slot_if_kept"] = bool(item.get("would_break_slot_if_kept", False))
        rec["candidate_lines"] = ",".join(_allowed_lines(rec))
        rows.append(rec)
    return pd.DataFrame(rows)


def _structure_fallback_config(cfg, feasibility_evidence: dict, bad_slots: list[dict] | None = None):
    summary = feasibility_evidence.get("feasibility_evidence_summary", {}) if isinstance(feasibility_evidence, dict) else {}
    slot_safe_lb = int(summary.get("slot_safe_lower_bound", cfg.model.max_campaign_slots) or cfg.model.max_campaign_slots)
    raw_bad_slots = bad_slots or []
    big_bad_slots = [slot for slot in raw_bad_slots if str(slot.get("line", "")) == "big_roll"]
    slot_buffer = int(getattr(cfg.model, "structure_fallback_slot_buffer", 4) or 4)
    slot_buffer = max(slot_buffer, min(12, 4 + len(big_bad_slots)))
    risk_boost = float(getattr(cfg.model, "structure_fallback_risk_boost", 1.5) or 1.5)
    if big_bad_slots:
        risk_boost = max(risk_boost, 1.75)
    min_ratio = float(getattr(cfg.model, "structure_fallback_min_real_schedule_ratio", cfg.model.min_real_schedule_ratio) or cfg.model.min_real_schedule_ratio)
    new_score = replace(
        cfg.score,
        slot_isolation_risk_penalty=max(cfg.score.slot_isolation_risk_penalty, int(cfg.score.slot_isolation_risk_penalty * risk_boost)),
        slot_pair_gap_risk_penalty=max(cfg.score.slot_pair_gap_risk_penalty, int(cfg.score.slot_pair_gap_risk_penalty * risk_boost)),
        slot_span_risk_penalty=max(cfg.score.slot_span_risk_penalty, int(cfg.score.slot_span_risk_penalty * risk_boost)),
        slot_order_count_penalty=max(cfg.score.slot_order_count_penalty, int(cfg.score.slot_order_count_penalty * risk_boost)),
        unassigned_real=max(20, int(cfg.score.unassigned_real * 0.8)),
    )
    new_model = replace(
        cfg.model,
        max_campaign_slots=max(int(cfg.model.max_campaign_slots), slot_safe_lb + slot_buffer),
        min_real_schedule_ratio=min(float(cfg.model.min_real_schedule_ratio), min_ratio),
        big_roll_slot_soft_order_cap=max(12, int(cfg.model.big_roll_slot_soft_order_cap) - (4 if big_bad_slots else 2)),
        small_roll_slot_soft_order_cap=max(12, int(cfg.model.small_roll_slot_soft_order_cap) - 1),
    )
    return replace(cfg, model=new_model, score=new_score)


def _summarize_bad_slots(slot_details) -> list[dict]:
    raw = slot_details if isinstance(slot_details, list) else []
    return [
        {
            "line": str(slot.get("line", "")),
            "slot_no": int(slot.get("slot_no", 0) or 0),
            "order_count": int(slot.get("order_count", 0) or 0),
            "order_count_over_cap": int(slot.get("order_count_over_cap", 0) or 0),
            "slot_route_risk_score": int(slot.get("slot_route_risk_score", 0) or 0),
            "template_coverage_ratio": float(slot.get("template_coverage_ratio", 0.0) or 0.0),
            "zero_in_orders": int(slot.get("zero_in_orders", 0) or 0),
            "zero_out_orders": int(slot.get("zero_out_orders", 0) or 0),
            "pair_gap_proxy": int(slot.get("pair_gap_proxy", 0) or 0),
            "span_risk": int(slot.get("span_risk", 0) or 0),
            "dominant_unroutable_reason": str(slot.get("dominant_unroutable_reason", "")),
        }
        for slot in raw
        if str(slot.get("status", "")) == "UNROUTABLE_SLOT"
    ]


def _candidate_summary(
    joint: dict,
    *,
    extra_drop_count: int = 0,
    source_orders_df: pd.DataFrame | None = None,
    extra_dropped_df: pd.DataFrame | None = None,
) -> dict | None:
    status = str(joint.get("status", ""))
    if status not in {"FEASIBLE", "ROUTING_INFEASIBLE"}:
        return None
    unr = int(joint.get("unroutable_slot_count", 0) or 0)
    unassigned = int(joint.get("unassigned_count", 0) or 0) + int(extra_drop_count)
    hard_cap_viol = int(joint.get("big_roll_order_cap_violations", 0) or 0)
    risk = int(joint.get("slot_route_risk_score", 0) or 0)
    objective = float(joint.get("objective", 1e18) or 1e18)
    routing_feasible = status == "FEASIBLE"
    candidate_type = "BEST_SEARCH_CANDIDATE_ANALYSIS"
    if routing_feasible:
        candidate_type = "PARTIAL_SCHEDULE_WITH_DROPS" if unassigned > 0 else "OFFICIAL_FULL_SCHEDULE"
    return {
        "routing_feasible": routing_feasible,
        "candidate_type": candidate_type,
        "search_status": status,
        "unassigned_count": unassigned,
        "dropped_order_count": int(extra_drop_count),
        "unroutable_slot_count": unr,
        "hard_cap_violations": hard_cap_viol,
        "slot_route_risk_score": risk,
        "objective": objective,
        "joint": joint,
        "source_orders_df": source_orders_df.copy() if isinstance(source_orders_df, pd.DataFrame) else pd.DataFrame(),
        "extra_dropped_df": extra_dropped_df.copy() if isinstance(extra_dropped_df, pd.DataFrame) else pd.DataFrame(),
    }


def _candidate_better(lhs: dict | None, rhs: dict | None, *, mode: str) -> bool:
    if rhs is None:
        return False
    if lhs is None:
        return True
    if mode == "official":
        lkey = (
            0 if lhs["routing_feasible"] else 1,
            int(lhs["unassigned_count"]),
            int(lhs["unroutable_slot_count"]),
            int(lhs["hard_cap_violations"]),
            int(lhs["slot_route_risk_score"]),
            float(lhs["objective"]),
        )
        rkey = (
            0 if rhs["routing_feasible"] else 1,
            int(rhs["unassigned_count"]),
            int(rhs["unroutable_slot_count"]),
            int(rhs["hard_cap_violations"]),
            int(rhs["slot_route_risk_score"]),
            float(rhs["objective"]),
        )
        return rkey < lkey
    lkey = (
        int(lhs["unroutable_slot_count"]),
        int(lhs["unassigned_count"]),
        int(lhs["dropped_order_count"]),
        int(lhs["slot_route_risk_score"]),
        float(lhs["objective"]),
    )
    rkey = (
        int(rhs["unroutable_slot_count"]),
        int(rhs["unassigned_count"]),
        int(rhs["dropped_order_count"]),
        int(rhs["slot_route_risk_score"]),
        float(rhs["objective"]),
    )
    return rkey < lkey


def _planned_from_joint(
    joint_pack: dict,
    orders_df: pd.DataFrame | None,
    extra_dropped: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    if orders_df is None:
        return None
    jplan = joint_pack.get("plan_df")
    if not isinstance(jplan, pd.DataFrame) or jplan.empty or "row_idx" not in jplan.columns:
        return None
    merged = jplan.copy().sort_values("row_idx", kind="mergesort").reset_index(drop=True)
    planned = orders_df.iloc[merged["row_idx"].astype(int).tolist()].copy().reset_index(drop=True)
    planned["line"] = merged["assigned_line"].astype(str)
    planned["master_slot"] = merged["assigned_slot"].astype(int)
    planned["master_seq"] = merged["master_seq"].astype(int)
    planned["campaign_id_hint"] = merged.get("campaign_id_hint", pd.Series([0] * len(merged))).astype(int)
    planned["campaign_seq_hint"] = merged.get("campaign_seq_hint", pd.Series([0] * len(merged))).astype(int)
    planned["selected_template_id"] = merged.get("selected_template_id", pd.Series([""] * len(merged))).astype(str)
    planned["selected_bridge_path"] = merged.get("selected_bridge_path", pd.Series([""] * len(merged))).astype(str)
    planned["force_break_before"] = merged.get("force_break_before", pd.Series([0] * len(merged))).astype(int)
    if "line_source" in planned.columns:
        planned["line_source"] = planned["line_source"].where(planned["line_source"] != "dual", "master_dual")
    pre_dropped = joint_pack.get("dropped_df")
    if isinstance(extra_dropped, pd.DataFrame) and not extra_dropped.empty:
        if isinstance(pre_dropped, pd.DataFrame) and not pre_dropped.empty:
            pre_dropped = pd.concat([pre_dropped, extra_dropped], ignore_index=True)
        else:
            pre_dropped = extra_dropped.copy()
    return planned, (pre_dropped if isinstance(pre_dropped, pd.DataFrame) else pd.DataFrame())


def _build_meta(
    req: ColdRollingRequest,
    *,
    engine_used: str,
    main_path: str,
    fallback_used: bool,
    fallback_type: str,
    fallback_reason: str,
    fallback_trace: list[dict],
    used_local_routing: bool,
    local_routing_role: str,
    input_order_count: int,
    failure_diagnostics: dict,
    joint_estimates: dict | None = None,
    strict_template_edges_enabled: bool = True,
    unroutable_slot_count: int = 0,
    slot_route_details: list[dict] | None = None,
    template_graph_health: str = "UNKNOWN",
    precheck_autorelax_applied: bool = False,
    solve_attempt_count: int = 0,
    fallback_attempt_count: int = 0,
    early_stop_reason: str = "",
    early_stop_deferred_for_semantic_fallback: bool = False,
    semantic_fallback_first_attempt_status: str = "",
    assignment_pressure_mode: str = "normal",
    effective_config=None,
    template_build_seconds: float = 0.0,
    joint_master_seconds: float = 0.0,
    local_router_seconds: float = 0.0,
    fallback_total_seconds: float = 0.0,
    feasibility_evidence: dict | None = None,
    failure_mode: str = "",
    evidence_level: str = "OK",
    top_infeasibility_signals: list[str] | None = None,
    drop_strategy_applied: bool = False,
    dropped_candidates_considered: int = 0,
    dropped_candidates_selected: int = 0,
    dominant_drop_reason_histogram: dict | None = None,
    structure_fallback_applied: bool = False,
    fallback_mode: str = "",
    slot_count_after_fallback: int = 0,
    drop_budget_after_fallback: int = 0,
    best_candidate_available: bool = False,
    best_candidate_type: str = "",
    best_candidate_objective: float = 0.0,
    best_candidate_search_status: str = "",
    best_candidate_routing_feasible: bool = False,
    best_candidate_unroutable_slot_count: int = 0,
    drop_stage: str = "",
    drop_budget_used: int = 0,
    drop_budget_remaining: int = 0,
    structure_first_applied: bool = False,
    time_expansion_applied: bool = False,
    slot_count_before_fallback: int = 0,
    restructured_slot_count: int = 0,
    bad_slots_before_restructure: list[dict] | None = None,
    bad_slots_after_restructure: list[dict] | None = None,
    orders_removed_from_bad_slots: list[dict] | None = None,
    partial_result_available: bool = False,
    partial_acceptance_passed: bool = False,
    partial_drop_ratio: float = 0.0,
    partial_drop_tons_ratio: float = 0.0,
) -> dict:
    joint_estimates = joint_estimates or {}
    slot_route_details = slot_route_details or []
    return {
        "engine_used": engine_used,
        "main_path": main_path,
        "fallback_used": bool(fallback_used),
        "fallback_type": str(fallback_type),
        "fallback_reason": str(fallback_reason),
        "fallback_trace": list(fallback_trace),
        "master_entry": "_run_global_joint_model",
        "used_local_routing": bool(used_local_routing),
        "local_routing_role": str(local_routing_role),
        "bridge_modeling": "template_based",
        "profile_name": str(req.config.model.profile_name),
        "input_order_count": int(input_order_count),
        "failure_diagnostics": failure_diagnostics if isinstance(failure_diagnostics, dict) else {},
        "joint_estimates": joint_estimates,
        "strict_template_edges_enabled": bool(strict_template_edges_enabled),
        "unroutable_slot_count": int(unroutable_slot_count),
        "slot_route_details": slot_route_details,
        "template_graph_health": str(template_graph_health),
        "precheck_autorelax_applied": bool(precheck_autorelax_applied),
        "solve_attempt_count": int(solve_attempt_count),
        "fallback_attempt_count": int(fallback_attempt_count),
        "early_stop_reason": str(early_stop_reason),
        "early_stop_deferred_for_semantic_fallback": bool(early_stop_deferred_for_semantic_fallback),
        "semantic_fallback_first_attempt_status": str(semantic_fallback_first_attempt_status),
        "assignment_pressure_mode": str(assignment_pressure_mode),
        "effective_config": effective_config,
        "template_build_seconds": float(template_build_seconds),
        "joint_master_seconds": float(joint_master_seconds),
        "local_router_seconds": float(local_router_seconds),
        "fallback_total_seconds": float(fallback_total_seconds),
        "feasibility_evidence": feasibility_evidence if isinstance(feasibility_evidence, dict) else {},
        "failure_mode": str(failure_mode),
        "evidence_level": str(evidence_level),
        "top_infeasibility_signals": list(top_infeasibility_signals or []),
        "drop_strategy_applied": bool(drop_strategy_applied),
        "dropped_candidates_considered": int(dropped_candidates_considered),
        "dropped_candidates_selected": int(dropped_candidates_selected),
        "dominant_drop_reason_histogram": dominant_drop_reason_histogram or {},
        "structure_fallback_applied": bool(structure_fallback_applied),
        "fallback_mode": str(fallback_mode),
        "slot_count_after_fallback": int(slot_count_after_fallback),
        "drop_budget_after_fallback": int(drop_budget_after_fallback),
        "best_candidate_available": bool(best_candidate_available),
        "best_candidate_type": str(best_candidate_type),
        "best_candidate_objective": float(best_candidate_objective),
        "best_candidate_search_status": str(best_candidate_search_status),
        "best_candidate_routing_feasible": bool(best_candidate_routing_feasible),
        "best_candidate_unroutable_slot_count": int(best_candidate_unroutable_slot_count),
        "drop_stage": str(drop_stage),
        "drop_budget_used": int(drop_budget_used),
        "drop_budget_remaining": int(drop_budget_remaining),
        "structure_first_applied": bool(structure_first_applied),
        "time_expansion_applied": bool(time_expansion_applied),
        "slot_count_before_fallback": int(slot_count_before_fallback),
        "restructured_slot_count": int(restructured_slot_count),
        "bad_slots_before_restructure": list(bad_slots_before_restructure or []),
        "bad_slots_after_restructure": list(bad_slots_after_restructure or []),
        "orders_removed_from_bad_slots": list(orders_removed_from_bad_slots or []),
        "partial_result_available": bool(partial_result_available),
        "partial_acceptance_passed": bool(partial_acceptance_passed),
        "partial_drop_ratio": float(partial_drop_ratio),
        "partial_drop_tons_ratio": float(partial_drop_tons_ratio),
    }


def solve_master_model(
    req: ColdRollingRequest,
    transition_pack: dict | None = None,
    orders_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Production master solve entry; returns structured solver output plus engine metadata only."""
    if orders_df is not None:
        current_cfg = req.config
        current_transition_pack = transition_pack
        template_build_seconds = 0.0
        joint_master_seconds = 0.0
        local_router_seconds = 0.0
        fallback_total_seconds = 0.0
        print(f"[APS][joint_master] min_real_schedule_ratio={current_cfg.model.min_real_schedule_ratio}")
        health = _assess_template_graph_health(current_transition_pack, current_cfg)
        precheck_autorelax_applied = False
        if health.get("template_graph_health") in {"SPARSE", "DISCONNECTED"}:
            current_cfg = _precheck_relaxed_config(current_cfg)
            rebuild_t0 = perf_counter()
            current_transition_pack = build_transition_templates(orders_df, current_cfg)
            template_build_seconds += perf_counter() - rebuild_t0
            health = _assess_template_graph_health(current_transition_pack, current_cfg)
            precheck_autorelax_applied = True
            print(
                f"[APS][template_precheck] autorelax=true, health={health.get('template_graph_health')}, "
                f"prune={current_cfg.model.global_prune_max_pairs_per_from}, top_k={current_cfg.model.template_top_k}, "
                f"routes_per_slot={current_cfg.model.max_routes_per_slot}"
            )
        else:
            print(f"[APS][template_precheck] autorelax=false, health={health.get('template_graph_health')}")

        feasibility_evidence = build_feasibility_evidence(orders_df, current_transition_pack, current_cfg)
        evidence_level = str(feasibility_evidence.get("evidence_level", "OK"))
        top_infeasibility_signals = list(feasibility_evidence.get("top_infeasibility_signals", []))
        print(
            f"[APS][feasibility_evidence] level={evidence_level}, "
            f"signals={top_infeasibility_signals[:3]}, "
            f"isolated_orders={int(feasibility_evidence.get('isolated_order_count', 0))}"
        )
        if _should_escalate_evidence_failure(current_cfg.model.profile_name, evidence_level):
            print("[APS][feasibility_evidence] 当前数据在现有硬语义下存在强不可行信号，不建议继续深层 fallback 搜索")

        joint = {"status": "NOT_RUN"}
        best_key = None
        best_joint = None
        status_hist: list[str] = []
        fallback_trace: list[dict] = []
        attempts = _joint_profiles(current_cfg)
        seeds = _joint_seeds(current_cfg)
        solve_attempt_count = 0
        fallback_attempt_count = 0
        early_stop_reason = ""
        early_stop_deferred_for_semantic_fallback = False
        semantic_fallback_first_attempt_status = ""
        consecutive_routing_infeasible = 0
        drop_strategy_applied = False
        dropped_candidates_considered = 0
        dropped_candidates_selected = 0
        dominant_drop_reason_histogram: dict = {}
        structure_fallback_applied = False
        fallback_mode = ""
        drop_stage = ""
        drop_budget_used = 0
        drop_budget_remaining = 0
        structure_first_applied = False
        time_expansion_applied = False
        slot_count_before_fallback = int(current_cfg.model.max_campaign_slots)
        slot_count_after_fallback = 0
        drop_budget_after_fallback = 0
        restructured_slot_count = 0
        bad_slots_before_restructure: list[dict] = []
        bad_slots_after_restructure: list[dict] = []
        orders_removed_from_bad_slots: list[dict] = []
        best_official_candidate = None
        best_partial_candidate = None
        best_any_candidate = None

        for aidx, ap in enumerate(attempts, start=1):
            for seed in seeds:
                solve_attempt_count += 1
                solve_t0 = perf_counter()
                joint = _run_global_joint_model(
                    orders_df,
                    current_transition_pack,
                    current_cfg,
                    start_penalty=int(ap["start_penalty"]),
                    low_slot_penalty=int(ap["low_slot_penalty"]),
                    ultra_low_slot_penalty=int(ap["ultra_low_slot_penalty"]),
                    time_scale=float(ap["time_scale"]),
                    random_seed=int(seed),
                )
                solve_elapsed = perf_counter() - solve_t0
                joint_master_seconds += solve_elapsed
                local_router_seconds += float(joint.get("local_router_seconds", 0.0) or 0.0)
                st = str(joint.get("status"))
                cand = _candidate_summary(
                    joint,
                    extra_drop_count=0,
                    source_orders_df=orders_df,
                    extra_dropped_df=pd.DataFrame(),
                )
                if _candidate_better(best_any_candidate, cand, mode="search"):
                    best_any_candidate = cand
                if cand is not None and cand["routing_feasible"]:
                    if cand["candidate_type"] == "OFFICIAL_FULL_SCHEDULE":
                        if _candidate_better(best_official_candidate, cand, mode="official"):
                            best_official_candidate = cand
                    else:
                        if _candidate_better(best_partial_candidate, cand, mode="official"):
                            best_partial_candidate = cand
                status_hist.append(st)
                print(f"[APS][joint_master] attempt={aidx}, seed={seed}, status={st}, params={ap}")
                fast_fail, fast_fail_reason = _should_fast_fail_after_routing_infeasible(joint, current_cfg)
                if st == "ROUTING_INFEASIBLE" and fast_fail:
                    early_stop_reason = fast_fail_reason
                    print(f"[APS][fast_fail] reason={early_stop_reason}")
                    break
                if st == "ROUTING_INFEASIBLE":
                    consecutive_routing_infeasible += 1
                else:
                    consecutive_routing_infeasible = 0
                if consecutive_routing_infeasible >= 2 and health.get("template_graph_health") in {"SPARSE", "DISCONNECTED"}:
                    if _semantic_fallback_enabled(current_cfg):
                        early_stop_deferred_for_semantic_fallback = True
                        print(
                            f"[APS][early_stop] deferred_for_semantic_fallback=true, "
                            f"health={health.get('template_graph_health')}, consecutive_routing_infeasible=2"
                        )
                    else:
                        early_stop_reason = f"template_graph_{health.get('template_graph_health')}_routing_infeasible_twice"
                        print(f"[APS][early_stop] reason={early_stop_reason}")
                    break
                if st != "FEASIBLE":
                    if st not in ("TIMEOUT_NO_FEASIBLE", "INFEASIBLE"):
                        break
                    continue
                key = (
                    int(joint.get("unassigned_count", 10**9)),
                    int(joint.get("low_slot_count", 10**9)),
                    int(joint.get("ultra_low_slot_count", 10**9)),
                    int(joint.get("global_ratio_over", 10**9)),
                    int(joint.get("total_virtual_blocks", 10**9)),
                    float(joint.get("objective", 10**18)),
                )
                if best_key is None or key < best_key:
                    best_key = key
                    best_joint = joint
            if early_stop_reason or best_joint is not None:
                break

        if best_joint is not None:
            mat = _planned_from_joint(best_joint, orders_df)
            if mat is not None:
                planned_df, dropped_df = mat
                rounds_df = pd.DataFrame([{"round": 1, "line": "all", "engine_used": "joint_master"}])
                return planned_df, rounds_df, dropped_df, _build_meta(
                    req,
                    engine_used="joint_master",
                    main_path="joint_master",
                    fallback_used=False,
                    fallback_type="",
                    fallback_reason="",
                    fallback_trace=fallback_trace,
                    used_local_routing=bool(best_joint.get("used_local_routing", False)),
                    local_routing_role=str(best_joint.get("local_routing_role", "not_used")),
                    input_order_count=len(orders_df),
                    failure_diagnostics={},
                    joint_estimates={
                        "estimated_virtual_blocks": int(best_joint.get("estimated_virtual_blocks", 0)),
                        "estimated_virtual_ton10": int(best_joint.get("estimated_virtual_ton10", 0)),
                        "estimated_global_ratio_over": int(best_joint.get("estimated_global_ratio_over", 0)),
                        "estimated_reverse_count": int(best_joint.get("estimated_reverse_count", 0)),
                        "estimated_reverse_rise": int(best_joint.get("estimated_reverse_rise", 0)),
                        "actual_virtual_blocks": int(best_joint.get("total_virtual_blocks", 0)),
                        "actual_virtual_ton10": int(best_joint.get("total_virtual_ton10", 0)),
                        "actual_global_ratio_over": int(best_joint.get("global_ratio_over", 0)),
                        "actual_reverse_count": int(best_joint.get("logical_reverse_count", 0)),
                        "actual_reverse_rise": int(best_joint.get("logical_reverse_total_rise", 0)),
                        "slot_route_risk_score": int(best_joint.get("slot_route_risk_score", 0)),
                    },
                    strict_template_edges_enabled=bool(best_joint.get("strict_template_edges_enabled", current_cfg.model.strict_template_edges)),
                    unroutable_slot_count=int(best_joint.get("unroutable_slot_count", 0)),
                    slot_route_details=list(best_joint.get("slot_route_details", [])),
                    template_graph_health=str(health.get("template_graph_health", "UNKNOWN")),
                    precheck_autorelax_applied=precheck_autorelax_applied,
                    solve_attempt_count=solve_attempt_count,
                    fallback_attempt_count=fallback_attempt_count,
                    early_stop_reason=early_stop_reason,
                    early_stop_deferred_for_semantic_fallback=early_stop_deferred_for_semantic_fallback,
                    semantic_fallback_first_attempt_status=semantic_fallback_first_attempt_status,
                    assignment_pressure_mode="relaxed" if "relaxed" in str(current_cfg.model.profile_name) else "normal",
                    effective_config=current_cfg,
                    template_build_seconds=template_build_seconds,
                    joint_master_seconds=joint_master_seconds,
                    local_router_seconds=local_router_seconds,
                    fallback_total_seconds=fallback_total_seconds,
                    drop_strategy_applied=drop_strategy_applied,
                    dropped_candidates_considered=dropped_candidates_considered,
                    dropped_candidates_selected=dropped_candidates_selected,
                    dominant_drop_reason_histogram=dominant_drop_reason_histogram,
                    structure_fallback_applied=structure_fallback_applied,
                    fallback_mode=fallback_mode,
                    slot_count_after_fallback=slot_count_after_fallback,
                    drop_budget_after_fallback=drop_budget_after_fallback,
                    best_candidate_available=True,
                    best_candidate_type="OFFICIAL_FULL_SCHEDULE" if dropped_df.empty else "PARTIAL_SCHEDULE_WITH_DROPS",
                    best_candidate_objective=float(best_joint.get("objective", 0.0) or 0.0),
                    best_candidate_search_status=str(best_joint.get("status", "")),
                    best_candidate_routing_feasible=True,
                    best_candidate_unroutable_slot_count=int(best_joint.get("unroutable_slot_count", 0) or 0),
                )

        structure_keep_df = None
        structure_drop_df = pd.DataFrame()
        structure_cfg = current_cfg
        if best_joint is None and _structure_fallback_enabled(current_cfg):
            candidate_df = _pick_structure_drop_candidates(orders_df, joint, feasibility_evidence, current_cfg)
            dropped_candidates_considered = int(len(candidate_df))
            if not candidate_df.empty:
                structure_fallback_applied = True
                drop_strategy_applied = True
                drop_stage = "STRUCTURE_FALLBACK"
                dropped_candidates_selected = int(len(candidate_df))
                drop_budget_used = int(len(candidate_df))
                drop_budget_remaining = max(0, int(getattr(current_cfg.model, "max_drop_count_for_partial", 0) or 0) - drop_budget_used)
                dominant_drop_reason_histogram = (
                    candidate_df["drop_reason"].fillna("OTHER").value_counts().to_dict()
                    if "drop_reason" in candidate_df.columns else {}
                )
                structure_first_applied = True
                bad_slots_before_restructure = _summarize_bad_slots(joint.get("slot_route_details", []))
                restructured_slot_count = int(len(bad_slots_before_restructure))
                keep_ids = set(str(v) for v in candidate_df["order_id"].tolist())
                structure_keep_df = orders_df[~orders_df["order_id"].astype(str).isin(keep_ids)].copy().reset_index(drop=True)
                structure_drop_df = candidate_df.copy().reset_index(drop=True)
                drop_budget_after_fallback = int(len(candidate_df))
                structure_cfg = _structure_fallback_config(current_cfg, feasibility_evidence, bad_slots_before_restructure)
                slot_count_after_fallback = int(structure_cfg.model.max_campaign_slots)
                fallback_mode = "STRUCTURE_FIRST"
                orders_removed_from_bad_slots = (
                    candidate_df[
                        [
                            c
                            for c in [
                                "order_id",
                                "drop_reason",
                                "secondary_reasons",
                                "source_bad_slot_line",
                                "source_bad_slot_no",
                                "source_bad_slot_order_count",
                                "source_bad_slot_over_cap",
                                "source_bad_slot_risk",
                                "source_bad_slot_reason",
                                "risk_summary",
                                "would_break_slot_if_kept",
                            ]
                            if c in candidate_df.columns
                        ]
                    ].assign(stage="STRUCTURE_FALLBACK").to_dict("records")
                    if {"order_id", "drop_reason"}.issubset(candidate_df.columns)
                    else []
                )
                fallback_attempt_count += 1
                fallback_trace.append(
                    {
                        "mode": "STRUCTURE_FIRST",
                        "drop_count": int(len(candidate_df)),
                        "drop_reasons": dominant_drop_reason_histogram,
                        "max_campaign_slots": int(structure_cfg.model.max_campaign_slots),
                        "bad_slots_before_restructure": int(len(bad_slots_before_restructure)),
                        "drop_budget_used": int(drop_budget_used),
                    }
                )
                print(
                    f"[APS][structure_fallback] applied=true, drop_count={len(candidate_df)}, "
                    f"slot_cap={structure_cfg.model.max_campaign_slots}, "
                    f"restructured_slots={len(bad_slots_before_restructure)}, reasons={dominant_drop_reason_histogram}"
                )
                fallback_t0 = perf_counter()
                structure_joint = _run_global_joint_model(
                    structure_keep_df,
                    current_transition_pack,
                    structure_cfg,
                    start_penalty=int(attempts[0]["start_penalty"]) if attempts else 120000,
                    low_slot_penalty=int(attempts[0]["low_slot_penalty"]) if attempts else 120000,
                    ultra_low_slot_penalty=int(attempts[0]["ultra_low_slot_penalty"]) if attempts else 300000,
                    time_scale=max(0.8, float(attempts[0]["time_scale"])) if attempts else 0.8,
                    random_seed=int((seeds[0] if seeds else 2027) + 501),
                )
                elapsed = perf_counter() - fallback_t0
                fallback_total_seconds += elapsed
                joint_master_seconds += elapsed
                local_router_seconds += float(structure_joint.get("local_router_seconds", 0.0) or 0.0)
                cand = _candidate_summary(
                    structure_joint,
                    extra_drop_count=len(structure_drop_df),
                    source_orders_df=structure_keep_df,
                    extra_dropped_df=structure_drop_df,
                )
                if _candidate_better(best_any_candidate, cand, mode="search"):
                    best_any_candidate = cand
                if cand is not None and cand["routing_feasible"]:
                    if cand["candidate_type"] == "OFFICIAL_FULL_SCHEDULE":
                        if _candidate_better(best_official_candidate, cand, mode="official"):
                            best_official_candidate = cand
                    else:
                        if _candidate_better(best_partial_candidate, cand, mode="official"):
                            best_partial_candidate = cand
                fallback_trace.append({"mode": "STRUCTURE_FIRST_RESULT", "status": str(structure_joint.get("status", "UNKNOWN"))})
                bad_slots_after_restructure = _summarize_bad_slots(structure_joint.get("slot_route_details", []))
                fallback_trace.append(
                    {
                        "mode": "STRUCTURE_FIRST_RESTRUCTURE",
                        "bad_slots_before": int(len(bad_slots_before_restructure)),
                        "bad_slots_after": int(len(bad_slots_after_restructure)),
                        "orders_removed_from_bad_slots": int(len(orders_removed_from_bad_slots)),
                    }
                )
                if str(structure_joint.get("status")) == "FEASIBLE":
                    mat = _planned_from_joint(structure_joint, structure_keep_df, extra_dropped=structure_drop_df)
                    if mat is not None:
                        planned_df, dropped_df = mat
                        rounds_df = pd.DataFrame([{"round": 1, "line": "all", "engine_used": "structure_fallback"}])
                        return planned_df, rounds_df, dropped_df, _build_meta(
                            req,
                            engine_used="semantic_fallback",
                            main_path="fallback",
                            fallback_used=True,
                            fallback_type="structure_fallback",
                            fallback_reason="drop_high_risk_orders",
                            fallback_trace=fallback_trace,
                            used_local_routing=bool(structure_joint.get("used_local_routing", False)),
                            local_routing_role=str(structure_joint.get("local_routing_role", "not_used")),
                            input_order_count=len(orders_df),
                            failure_diagnostics={},
                            joint_estimates={
                                "estimated_virtual_blocks": int(structure_joint.get("estimated_virtual_blocks", 0)),
                                "estimated_virtual_ton10": int(structure_joint.get("estimated_virtual_ton10", 0)),
                                "estimated_global_ratio_over": int(structure_joint.get("estimated_global_ratio_over", 0)),
                                "estimated_reverse_count": int(structure_joint.get("estimated_reverse_count", 0)),
                                "estimated_reverse_rise": int(structure_joint.get("estimated_reverse_rise", 0)),
                                "slot_route_risk_score": int(structure_joint.get("slot_route_risk_score", 0)),
                            },
                            strict_template_edges_enabled=bool(structure_joint.get("strict_template_edges_enabled", structure_cfg.model.strict_template_edges)),
                            unroutable_slot_count=int(structure_joint.get("unroutable_slot_count", 0)),
                            slot_route_details=list(structure_joint.get("slot_route_details", [])),
                            template_graph_health=str(health.get("template_graph_health", "UNKNOWN")),
                            precheck_autorelax_applied=precheck_autorelax_applied,
                            solve_attempt_count=solve_attempt_count,
                            fallback_attempt_count=fallback_attempt_count,
                            early_stop_reason=early_stop_reason,
                            early_stop_deferred_for_semantic_fallback=early_stop_deferred_for_semantic_fallback,
                            semantic_fallback_first_attempt_status=semantic_fallback_first_attempt_status,
                            assignment_pressure_mode="relaxed",
                            effective_config=structure_cfg,
                            template_build_seconds=template_build_seconds,
                            joint_master_seconds=joint_master_seconds,
                            local_router_seconds=local_router_seconds,
                            fallback_total_seconds=fallback_total_seconds,
                            feasibility_evidence=feasibility_evidence,
                            failure_mode="",
                            evidence_level=evidence_level,
                            top_infeasibility_signals=top_infeasibility_signals,
                            drop_strategy_applied=drop_strategy_applied,
                            drop_stage=drop_stage,
                            drop_budget_used=drop_budget_used,
                            drop_budget_remaining=drop_budget_remaining,
                            dropped_candidates_considered=dropped_candidates_considered,
                            dropped_candidates_selected=dropped_candidates_selected,
                            dominant_drop_reason_histogram=dominant_drop_reason_histogram,
                            structure_fallback_applied=structure_fallback_applied,
                            fallback_mode=fallback_mode,
                            structure_first_applied=structure_first_applied,
                            time_expansion_applied=time_expansion_applied,
                            slot_count_before_fallback=slot_count_before_fallback,
                            slot_count_after_fallback=slot_count_after_fallback,
                            drop_budget_after_fallback=drop_budget_after_fallback,
                            restructured_slot_count=restructured_slot_count,
                            bad_slots_before_restructure=bad_slots_before_restructure,
                            bad_slots_after_restructure=bad_slots_after_restructure,
                            orders_removed_from_bad_slots=orders_removed_from_bad_slots,
                            partial_result_available=not dropped_df.empty,
                            partial_acceptance_passed=not dropped_df.empty,
                            partial_drop_ratio=float(len(dropped_df) / max(1, len(orders_df))),
                            partial_drop_tons_ratio=float(dropped_df["tons"].sum() / max(1e-9, float(orders_df["tons"].sum()))) if "tons" in dropped_df.columns and "tons" in orders_df.columns else 0.0,
                            best_candidate_available=True,
                            best_candidate_type="OFFICIAL_FULL_SCHEDULE" if dropped_df.empty else "PARTIAL_SCHEDULE_WITH_DROPS",
                            best_candidate_objective=float(structure_joint.get("objective", 0.0) or 0.0),
                            best_candidate_search_status=str(structure_joint.get("status", "")),
                            best_candidate_routing_feasible=True,
                            best_candidate_unroutable_slot_count=int(structure_joint.get("unroutable_slot_count", 0) or 0),
                        )

        if _semantic_fallback_enabled(current_cfg) or _scale_down_fallback_enabled(current_cfg):
            timeout_or_infeasible = bool(status_hist) and all(
                s in ("TIMEOUT_NO_FEASIBLE", "INFEASIBLE", "ROUTING_INFEASIBLE") for s in status_hist
            )
            if (timeout_or_infeasible or early_stop_deferred_for_semantic_fallback) and not early_stop_reason:
                working_orders_df = structure_keep_df if isinstance(structure_keep_df, pd.DataFrame) and not structure_keep_df.empty else orders_df
                carried_drop_df = structure_drop_df if isinstance(structure_drop_df, pd.DataFrame) and not structure_drop_df.empty else pd.DataFrame()
                working_cfg = structure_cfg if structure_fallback_applied else current_cfg
                fallback_cfgs = _semantic_fallback_configs(working_cfg) if _semantic_fallback_enabled(working_cfg) else []
                retry = attempts[0] if attempts else {"start_penalty": 90000, "low_slot_penalty": 100000, "ultra_low_slot_penalty": 400000, "time_scale": 0.45}
                retry_seed = seeds[0] if seeds else 2027
                for fidx, fcfg in enumerate(fallback_cfgs, start=1):
                    fallback_attempt_count += 1
                    fallback_mode = "TIME_EXPANSION" if structure_fallback_applied else "SEMANTIC_FALLBACK"
                    time_expansion_applied = True
                    trace_item = {
                        "idx": int(fidx),
                        "prune": fcfg.model.global_prune_max_pairs_per_from,
                        "template_top_k": int(fcfg.model.template_top_k),
                        "max_routes_per_slot": int(fcfg.model.max_routes_per_slot),
                        "time_limit_seconds": float(fcfg.model.time_limit_seconds),
                    }
                    print(
                        f"[APS][semantic_fallback] #{fidx}: prune={fcfg.model.global_prune_max_pairs_per_from}, "
                        f"template_top_k={fcfg.model.template_top_k}, max_routes_per_slot={fcfg.model.max_routes_per_slot}, "
                        f"time_limit_seconds={fcfg.model.time_limit_seconds}"
                    )
                    fallback_t0 = perf_counter()
                    joint = _run_global_joint_model(
                        working_orders_df,
                        current_transition_pack,
                        fcfg,
                        start_penalty=int(retry["start_penalty"]),
                        low_slot_penalty=int(retry["low_slot_penalty"]),
                        ultra_low_slot_penalty=int(retry["ultra_low_slot_penalty"]),
                        time_scale=max(1.1, float(retry["time_scale"]) * (1.4 + 0.2 * fidx)),
                        random_seed=int(retry_seed + fidx),
                    )
                    elapsed = perf_counter() - fallback_t0
                    fallback_total_seconds += elapsed
                    joint_master_seconds += elapsed
                    local_router_seconds += float(joint.get("local_router_seconds", 0.0) or 0.0)
                    cand = _candidate_summary(
                        joint,
                        extra_drop_count=len(carried_drop_df),
                        source_orders_df=working_orders_df,
                        extra_dropped_df=carried_drop_df,
                    )
                    if _candidate_better(best_any_candidate, cand, mode="search"):
                        best_any_candidate = cand
                    if cand is not None and cand["routing_feasible"]:
                        if cand["candidate_type"] == "OFFICIAL_FULL_SCHEDULE":
                            if _candidate_better(best_official_candidate, cand, mode="official"):
                                best_official_candidate = cand
                        else:
                            if _candidate_better(best_partial_candidate, cand, mode="official"):
                                best_partial_candidate = cand
                    trace_item["status"] = str(joint.get("status"))
                    fallback_trace.append(trace_item)
                    if fidx == 1:
                        semantic_fallback_first_attempt_status = str(joint.get("status"))
                        print(
                            f"[APS][semantic_fallback] first_attempt_status={semantic_fallback_first_attempt_status}, "
                            f"early_stop_deferred_for_semantic_fallback={early_stop_deferred_for_semantic_fallback}"
                        )
                        if (
                            str(current_cfg.model.profile_name) == "feasibility_fast_slot_safe"
                            and semantic_fallback_first_attempt_status == "ROUTING_INFEASIBLE"
                        ):
                            early_stop_reason = "fast_slot_safe_first_semantic_fallback_routing_infeasible"
                            break
                        if early_stop_deferred_for_semantic_fallback and semantic_fallback_first_attempt_status == "ROUTING_INFEASIBLE":
                            early_stop_reason = (
                                f"template_graph_{health.get('template_graph_health')}_"
                                f"routing_infeasible_after_first_semantic_fallback"
                            )
                            break
                    fast_fail, fast_fail_reason = _should_fast_fail_after_routing_infeasible(joint, fcfg)
                    if str(joint.get("status")) == "ROUTING_INFEASIBLE" and fast_fail:
                        early_stop_reason = fast_fail_reason
                        break
                    if str(joint.get("status")) != "FEASIBLE":
                        continue
                    mat = _planned_from_joint(joint, working_orders_df, extra_dropped=carried_drop_df)
                    if mat is None:
                        continue
                    planned_df, dropped_df = mat
                    rounds_df = pd.DataFrame([{"round": 1, "line": "all", "engine_used": "semantic_fallback"}])
                    return planned_df, rounds_df, dropped_df, _build_meta(
                        req,
                        engine_used="semantic_fallback",
                        main_path="fallback",
                        fallback_used=True,
                        fallback_type="semantic_fallback",
                        fallback_reason="joint_timeout_or_infeasible",
                        fallback_trace=fallback_trace,
                        used_local_routing=bool(joint.get("used_local_routing", False)),
                        local_routing_role=str(joint.get("local_routing_role", "not_used")),
                        input_order_count=len(orders_df),
                        failure_diagnostics={},
                        joint_estimates={
                            "estimated_virtual_blocks": int(joint.get("estimated_virtual_blocks", 0)),
                            "estimated_virtual_ton10": int(joint.get("estimated_virtual_ton10", 0)),
                            "estimated_global_ratio_over": int(joint.get("estimated_global_ratio_over", 0)),
                            "estimated_reverse_count": int(joint.get("estimated_reverse_count", 0)),
                            "estimated_reverse_rise": int(joint.get("estimated_reverse_rise", 0)),
                            "actual_virtual_blocks": int(joint.get("total_virtual_blocks", 0)),
                            "actual_virtual_ton10": int(joint.get("total_virtual_ton10", 0)),
                            "actual_global_ratio_over": int(joint.get("global_ratio_over", 0)),
                            "actual_reverse_count": int(joint.get("logical_reverse_count", 0)),
                            "actual_reverse_rise": int(joint.get("logical_reverse_total_rise", 0)),
                            "slot_route_risk_score": int(joint.get("slot_route_risk_score", 0)),
                            "hard_cap_not_enforced": bool(joint.get("hard_cap_not_enforced", False)),
                        },
                        strict_template_edges_enabled=bool(joint.get("strict_template_edges_enabled", fcfg.model.strict_template_edges)),
                        unroutable_slot_count=int(joint.get("unroutable_slot_count", 0)),
                        slot_route_details=list(joint.get("slot_route_details", [])),
                        template_graph_health=str(health.get("template_graph_health", "UNKNOWN")),
                        precheck_autorelax_applied=precheck_autorelax_applied,
                        solve_attempt_count=solve_attempt_count,
                        fallback_attempt_count=fallback_attempt_count,
                        early_stop_reason=early_stop_reason,
                        early_stop_deferred_for_semantic_fallback=early_stop_deferred_for_semantic_fallback,
                        semantic_fallback_first_attempt_status=semantic_fallback_first_attempt_status,
                        assignment_pressure_mode="relaxed" if "relaxed" in str(fcfg.model.profile_name) else "normal",
                        effective_config=fcfg,
                        template_build_seconds=template_build_seconds,
                        joint_master_seconds=joint_master_seconds,
                        local_router_seconds=local_router_seconds,
                        fallback_total_seconds=fallback_total_seconds,
                        feasibility_evidence=feasibility_evidence,
                        failure_mode="",
                        evidence_level=evidence_level,
                            top_infeasibility_signals=top_infeasibility_signals,
                            drop_strategy_applied=drop_strategy_applied,
                            drop_stage=drop_stage,
                            drop_budget_used=drop_budget_used,
                            drop_budget_remaining=drop_budget_remaining,
                            dropped_candidates_considered=dropped_candidates_considered,
                            dropped_candidates_selected=dropped_candidates_selected,
                            dominant_drop_reason_histogram=dominant_drop_reason_histogram,
                            structure_fallback_applied=structure_fallback_applied,
                            fallback_mode=fallback_mode,
                            structure_first_applied=structure_first_applied,
                            time_expansion_applied=time_expansion_applied,
                            slot_count_before_fallback=slot_count_before_fallback,
                            slot_count_after_fallback=slot_count_after_fallback,
                            drop_budget_after_fallback=drop_budget_after_fallback,
                            restructured_slot_count=restructured_slot_count,
                            bad_slots_before_restructure=bad_slots_before_restructure,
                            bad_slots_after_restructure=bad_slots_after_restructure,
                            orders_removed_from_bad_slots=orders_removed_from_bad_slots,
                            partial_result_available=not dropped_df.empty,
                            partial_acceptance_passed=not dropped_df.empty,
                            partial_drop_ratio=float(len(dropped_df) / max(1, len(orders_df))),
                            partial_drop_tons_ratio=float(dropped_df["tons"].sum() / max(1e-9, float(orders_df["tons"].sum()))) if "tons" in dropped_df.columns and "tons" in orders_df.columns else 0.0,
                            best_candidate_available=True,
                        best_candidate_type="OFFICIAL_FULL_SCHEDULE" if dropped_df.empty else "PARTIAL_SCHEDULE_WITH_DROPS",
                        best_candidate_objective=float(joint.get("objective", 0.0) or 0.0),
                        best_candidate_search_status=str(joint.get("status", "")),
                        best_candidate_routing_feasible=True,
                        best_candidate_unroutable_slot_count=int(joint.get("unroutable_slot_count", 0) or 0),
                    )

                if _scale_down_fallback_enabled(working_cfg) and len(working_orders_df) > 120 and not early_stop_reason:
                    scale_cfg = fallback_cfgs[-1] if fallback_cfgs else working_cfg
                    for n_keep in list(working_cfg.model.scale_down_keep_steps):
                        if len(working_orders_df) <= n_keep:
                            continue
                        fallback_attempt_count += 1
                        print(f"[APS][semantic_fallback] scale-down keep={n_keep}")
                        od = working_orders_df.copy().sort_values(["due_rank", "priority", "tons"], ascending=[True, False, False], kind="mergesort")
                        keep_df = od.head(n_keep).copy().reset_index(drop=True)
                        drop_df = od.iloc[n_keep:].copy().reset_index(drop=True)
                        drop_df["drop_reason"] = "FALLBACK_SCALE_UNSCHEDULED"
                        if not carried_drop_df.empty:
                            drop_df = pd.concat([carried_drop_df, drop_df], ignore_index=True)
                        fallback_t0 = perf_counter()
                        scale_joint = _run_global_joint_model(
                            keep_df,
                            current_transition_pack,
                            scale_cfg,
                            start_penalty=int(retry["start_penalty"]),
                            low_slot_penalty=int(retry["low_slot_penalty"]),
                            ultra_low_slot_penalty=int(retry["ultra_low_slot_penalty"]),
                            time_scale=max(1.1, float(retry["time_scale"]) * 2.0),
                            random_seed=int(retry_seed + 100 + n_keep),
                        )
                        elapsed = perf_counter() - fallback_t0
                        fallback_total_seconds += elapsed
                        joint_master_seconds += elapsed
                        local_router_seconds += float(scale_joint.get("local_router_seconds", 0.0) or 0.0)
                        cand = _candidate_summary(
                            scale_joint,
                            extra_drop_count=len(drop_df),
                            source_orders_df=keep_df,
                            extra_dropped_df=drop_df,
                        )
                        if _candidate_better(best_any_candidate, cand, mode="search"):
                            best_any_candidate = cand
                        if cand is not None and cand["routing_feasible"]:
                            if cand["candidate_type"] == "OFFICIAL_FULL_SCHEDULE":
                                if _candidate_better(best_official_candidate, cand, mode="official"):
                                    best_official_candidate = cand
                            else:
                                if _candidate_better(best_partial_candidate, cand, mode="official"):
                                    best_partial_candidate = cand
                        fallback_trace.append({"scale_keep": int(n_keep), "status": str(scale_joint.get("status"))})
                        fast_fail, fast_fail_reason = _should_fast_fail_after_routing_infeasible(scale_joint, scale_cfg)
                        if str(scale_joint.get("status")) == "ROUTING_INFEASIBLE" and fast_fail:
                            early_stop_reason = fast_fail_reason
                            break
                        if str(scale_joint.get("status")) != "FEASIBLE":
                            continue
                        mat = _planned_from_joint(scale_joint, keep_df, extra_dropped=drop_df)
                        if mat is None:
                            continue
                        planned_df, dropped_df = mat
                        rounds_df = pd.DataFrame([{"round": 1, "line": "all", "engine_used": "semantic_fallback"}])
                        return planned_df, rounds_df, dropped_df, _build_meta(
                            req,
                            engine_used="semantic_fallback",
                            main_path="fallback",
                            fallback_used=True,
                            fallback_type="semantic_fallback",
                            fallback_reason="scale_down_keep_top_priority",
                            fallback_trace=fallback_trace,
                            used_local_routing=bool(scale_joint.get("used_local_routing", False)),
                            local_routing_role=str(scale_joint.get("local_routing_role", "not_used")),
                            input_order_count=len(orders_df),
                            failure_diagnostics={},
                            joint_estimates={
                                "estimated_virtual_blocks": int(scale_joint.get("estimated_virtual_blocks", 0)),
                                "estimated_virtual_ton10": int(scale_joint.get("estimated_virtual_ton10", 0)),
                                "estimated_global_ratio_over": int(scale_joint.get("estimated_global_ratio_over", 0)),
                                "estimated_reverse_count": int(scale_joint.get("estimated_reverse_count", 0)),
                                "estimated_reverse_rise": int(scale_joint.get("estimated_reverse_rise", 0)),
                                "actual_virtual_blocks": int(scale_joint.get("total_virtual_blocks", 0)),
                                "actual_virtual_ton10": int(scale_joint.get("total_virtual_ton10", 0)),
                                "actual_global_ratio_over": int(scale_joint.get("global_ratio_over", 0)),
                                "actual_reverse_count": int(scale_joint.get("logical_reverse_count", 0)),
                                "actual_reverse_rise": int(scale_joint.get("logical_reverse_total_rise", 0)),
                                "slot_route_risk_score": int(scale_joint.get("slot_route_risk_score", 0)),
                                "hard_cap_not_enforced": bool(scale_joint.get("hard_cap_not_enforced", False)),
                            },
                            strict_template_edges_enabled=bool(scale_joint.get("strict_template_edges_enabled", scale_cfg.model.strict_template_edges)),
                            unroutable_slot_count=int(scale_joint.get("unroutable_slot_count", 0)),
                            slot_route_details=list(scale_joint.get("slot_route_details", [])),
                            template_graph_health=str(health.get("template_graph_health", "UNKNOWN")),
                            precheck_autorelax_applied=precheck_autorelax_applied,
                            solve_attempt_count=solve_attempt_count,
                            fallback_attempt_count=fallback_attempt_count,
                            early_stop_reason=early_stop_reason,
                            early_stop_deferred_for_semantic_fallback=early_stop_deferred_for_semantic_fallback,
                            semantic_fallback_first_attempt_status=semantic_fallback_first_attempt_status,
                            assignment_pressure_mode="relaxed" if "relaxed" in str(scale_cfg.model.profile_name) else "normal",
                            effective_config=scale_cfg,
                            template_build_seconds=template_build_seconds,
                            joint_master_seconds=joint_master_seconds,
                            local_router_seconds=local_router_seconds,
                            fallback_total_seconds=fallback_total_seconds,
                            feasibility_evidence=feasibility_evidence,
                            failure_mode="",
                            evidence_level=evidence_level,
                            top_infeasibility_signals=top_infeasibility_signals,
                            drop_strategy_applied=drop_strategy_applied,
                            drop_stage=drop_stage,
                            drop_budget_used=drop_budget_used,
                            drop_budget_remaining=drop_budget_remaining,
                            dropped_candidates_considered=dropped_candidates_considered,
                            dropped_candidates_selected=dropped_candidates_selected,
                            dominant_drop_reason_histogram=dominant_drop_reason_histogram,
                            structure_fallback_applied=structure_fallback_applied,
                            fallback_mode="TIME_EXPANSION",
                            structure_first_applied=structure_first_applied,
                            time_expansion_applied=time_expansion_applied,
                            slot_count_before_fallback=slot_count_before_fallback,
                            slot_count_after_fallback=slot_count_after_fallback,
                            drop_budget_after_fallback=drop_budget_after_fallback,
                            restructured_slot_count=restructured_slot_count,
                            bad_slots_before_restructure=bad_slots_before_restructure,
                            bad_slots_after_restructure=bad_slots_after_restructure,
                            orders_removed_from_bad_slots=orders_removed_from_bad_slots,
                            partial_result_available=not dropped_df.empty,
                            partial_acceptance_passed=not dropped_df.empty,
                            partial_drop_ratio=float(len(dropped_df) / max(1, len(orders_df))),
                            partial_drop_tons_ratio=float(dropped_df["tons"].sum() / max(1e-9, float(orders_df["tons"].sum()))) if "tons" in dropped_df.columns and "tons" in orders_df.columns else 0.0,
                            best_candidate_available=True,
                            best_candidate_type="OFFICIAL_FULL_SCHEDULE" if dropped_df.empty else "PARTIAL_SCHEDULE_WITH_DROPS",
                            best_candidate_objective=float(scale_joint.get("objective", 0.0) or 0.0),
                            best_candidate_search_status=str(scale_joint.get("status", "")),
                            best_candidate_routing_feasible=True,
                            best_candidate_unroutable_slot_count=int(scale_joint.get("unroutable_slot_count", 0) or 0),
                        )

        diagnostics = _build_joint_failure_diagnostics(current_cfg, orders_df, current_transition_pack, joint)
        diagnostics["precheck_autorelax_applied"] = bool(precheck_autorelax_applied)
        diagnostics["solve_attempt_count"] = int(solve_attempt_count)
        diagnostics["fallback_attempt_count"] = int(fallback_attempt_count)
        diagnostics["early_stop_reason"] = str(early_stop_reason)
        diagnostics["early_stop_deferred_for_semantic_fallback"] = bool(early_stop_deferred_for_semantic_fallback)
        diagnostics["semantic_fallback_first_attempt_status"] = str(semantic_fallback_first_attempt_status)
        diagnostics["unroutable_slot_count"] = int(joint.get("unroutable_slot_count", 0))
        diagnostics["unroutable_slots_topn"] = list(joint.get("slot_route_details", []))[:5]
        diagnostics["template_build_seconds"] = round(float(template_build_seconds), 6)
        diagnostics["joint_master_seconds"] = round(float(joint_master_seconds), 6)
        diagnostics["local_router_seconds"] = round(float(local_router_seconds), 6)
        diagnostics["fallback_total_seconds"] = round(float(fallback_total_seconds), 6)
        diagnostics["feasibility_evidence"] = feasibility_evidence
        diagnostics["evidence_level"] = evidence_level
        diagnostics["top_infeasibility_signals"] = top_infeasibility_signals
        diagnostics["drop_strategy_applied"] = bool(drop_strategy_applied)
        diagnostics["drop_stage"] = str(drop_stage)
        diagnostics["drop_budget_used"] = int(drop_budget_used)
        diagnostics["drop_budget_remaining"] = int(drop_budget_remaining)
        diagnostics["dropped_candidates_considered"] = int(dropped_candidates_considered)
        diagnostics["dropped_candidates_selected"] = int(dropped_candidates_selected)
        diagnostics["dominant_drop_reason_histogram"] = dict(dominant_drop_reason_histogram)
        diagnostics["structure_fallback_applied"] = bool(structure_fallback_applied)
        diagnostics["fallback_mode"] = str(fallback_mode)
        diagnostics["structure_first_applied"] = bool(structure_first_applied)
        diagnostics["time_expansion_applied"] = bool(time_expansion_applied)
        diagnostics["slot_count_before_fallback"] = int(slot_count_before_fallback)
        diagnostics["slot_count_after_fallback"] = int(slot_count_after_fallback)
        diagnostics["drop_budget_after_fallback"] = int(drop_budget_after_fallback)
        diagnostics["restructured_slot_count"] = int(restructured_slot_count)
        diagnostics["bad_slots_before_restructure"] = list(bad_slots_before_restructure)
        diagnostics["bad_slots_after_restructure"] = list(bad_slots_after_restructure)
        diagnostics["orders_removed_from_bad_slots"] = list(orders_removed_from_bad_slots)
        last_status = str(joint.get("status", "UNKNOWN"))
        if bool(joint.get("hard_cap_not_enforced", False)):
            failure_mode = "FAILED_IMPLEMENTATION_ERROR"
        elif last_status == "TIMEOUT_NO_FEASIBLE":
            failure_mode = "FAILED_TIME_BUDGET"
        else:
            failure_mode = (
                "FAILED_STRONG_INFEASIBILITY_SIGNAL"
                if _should_escalate_evidence_failure(current_cfg.model.profile_name, evidence_level)
                else "FAILED_ROUTING_SEARCH"
            )
        if not _legacy_fallback_enabled(current_cfg):
            best_candidate = best_partial_candidate or best_official_candidate or best_any_candidate
            meta = _build_meta(
                req,
                engine_used="joint_master_failed",
                main_path="joint_master",
                fallback_used=bool(fallback_attempt_count > 0),
                fallback_type="semantic_fallback" if fallback_attempt_count > 0 else "",
                fallback_reason="routing_infeasible",
                fallback_trace=fallback_trace,
                used_local_routing=bool(joint.get("used_local_routing", False)),
                local_routing_role=str(joint.get("local_routing_role", "not_used")),
                input_order_count=len(orders_df),
                failure_diagnostics=diagnostics,
                joint_estimates={
                    "estimated_virtual_blocks": int(joint.get("estimated_virtual_blocks", 0)),
                    "estimated_virtual_ton10": int(joint.get("estimated_virtual_ton10", 0)),
                    "estimated_global_ratio_over": int(joint.get("estimated_global_ratio_over", 0)),
                    "estimated_reverse_count": int(joint.get("estimated_reverse_count", 0)),
                    "estimated_reverse_rise": int(joint.get("estimated_reverse_rise", 0)),
                    "estimated_bridge_cost": int(joint.get("estimated_bridge_cost", 0)),
                    "estimated_isolation_risk": int(joint.get("estimated_isolation_risk", 0)),
                    "estimated_pair_gap_risk": int(joint.get("estimated_pair_gap_risk", 0)),
                    "estimated_span_risk": int(joint.get("estimated_span_risk", 0)),
                    "slot_route_risk_score": int(joint.get("slot_route_risk_score", 0)),
                    "max_slot_order_count": int(joint.get("max_slot_order_count", 0)),
                    "big_roll_max_slot_order_count": int(joint.get("big_roll_max_slot_order_count", 0)),
                    "small_roll_max_slot_order_count": int(joint.get("small_roll_max_slot_order_count", 0)),
                    "big_roll_slot_order_hard_cap": int(joint.get("big_roll_slot_order_hard_cap", 0)),
                    "big_roll_order_cap_violations": int(joint.get("big_roll_order_cap_violations", 0)),
                    "avg_slot_order_count": float(joint.get("avg_slot_order_count", 0.0)),
                },
                strict_template_edges_enabled=bool(joint.get("strict_template_edges_enabled", current_cfg.model.strict_template_edges)),
                unroutable_slot_count=int(joint.get("unroutable_slot_count", 0)),
                slot_route_details=list(joint.get("slot_route_details", [])),
                template_graph_health=str(health.get("template_graph_health", "UNKNOWN")),
                precheck_autorelax_applied=precheck_autorelax_applied,
                solve_attempt_count=solve_attempt_count,
                fallback_attempt_count=fallback_attempt_count,
                early_stop_reason=early_stop_reason,
                early_stop_deferred_for_semantic_fallback=early_stop_deferred_for_semantic_fallback,
                semantic_fallback_first_attempt_status=semantic_fallback_first_attempt_status,
                assignment_pressure_mode="relaxed" if "relaxed" in str(current_cfg.model.profile_name) else "normal",
                effective_config=current_cfg,
                template_build_seconds=template_build_seconds,
                joint_master_seconds=joint_master_seconds,
                local_router_seconds=local_router_seconds,
                fallback_total_seconds=fallback_total_seconds,
                feasibility_evidence=feasibility_evidence,
                failure_mode=failure_mode,
                evidence_level=evidence_level,
                top_infeasibility_signals=top_infeasibility_signals,
                drop_strategy_applied=drop_strategy_applied,
                drop_stage=drop_stage,
                drop_budget_used=drop_budget_used,
                drop_budget_remaining=drop_budget_remaining,
                dropped_candidates_considered=dropped_candidates_considered,
                dropped_candidates_selected=dropped_candidates_selected,
                dominant_drop_reason_histogram=dominant_drop_reason_histogram,
                structure_fallback_applied=structure_fallback_applied,
                fallback_mode=fallback_mode,
                structure_first_applied=structure_first_applied,
                time_expansion_applied=time_expansion_applied,
                slot_count_before_fallback=slot_count_before_fallback,
                slot_count_after_fallback=slot_count_after_fallback,
                drop_budget_after_fallback=drop_budget_after_fallback,
                restructured_slot_count=restructured_slot_count,
                bad_slots_before_restructure=bad_slots_before_restructure,
                bad_slots_after_restructure=bad_slots_after_restructure,
                orders_removed_from_bad_slots=orders_removed_from_bad_slots,
                partial_result_available=bool(best_partial_candidate is not None),
                partial_acceptance_passed=False,
                partial_drop_ratio=float(best_candidate.get("dropped_order_count", 0) / max(1, len(orders_df))) if best_candidate else 0.0,
                partial_drop_tons_ratio=0.0,
                best_candidate_available=best_candidate is not None,
                best_candidate_type=str(best_candidate.get("candidate_type", "")) if best_candidate else "",
                best_candidate_objective=float(best_candidate.get("objective", 0.0) or 0.0) if best_candidate else 0.0,
                best_candidate_search_status=str(best_candidate.get("search_status", "")) if best_candidate else "",
                best_candidate_routing_feasible=bool(best_candidate.get("routing_feasible", False)) if best_candidate else False,
                best_candidate_unroutable_slot_count=int(best_candidate.get("unroutable_slot_count", 0) or 0) if best_candidate else 0,
            )
            meta["routing_feasible"] = False
            meta["routing_status"] = "ROUTING_INFEASIBLE"
            meta["template_pair_ok"] = False
            meta["adjacency_rule_ok"] = False
            meta["bridge_expand_ok"] = False
            meta["result_acceptance_status"] = "BEST_SEARCH_CANDIDATE_ANALYSIS" if best_candidate is not None else failure_mode
            meta["export_failed_result_for_debug"] = bool(current_cfg.model.export_failed_result_for_debug)
            meta["export_best_candidate_analysis"] = bool(current_cfg.model.export_best_candidate_analysis)
            meta["final_export_performed"] = False
            meta["result_usage"] = "ANALYSIS_ONLY"
            failed_drop_df = pd.DataFrame()
            if best_candidate is not None:
                cand_joint = best_candidate.get("joint", {})
                cand_source_orders = best_candidate.get("source_orders_df")
                cand_extra_drop = best_candidate.get("extra_dropped_df")
                cand_drop = cand_joint.get("dropped_df")
                frames = []
                if isinstance(cand_drop, pd.DataFrame) and not cand_drop.empty:
                    frames.append(cand_drop.copy())
                if isinstance(cand_extra_drop, pd.DataFrame) and not cand_extra_drop.empty:
                    frames.append(cand_extra_drop.copy())
                if frames:
                    failed_drop_df = pd.concat(frames, ignore_index=True).drop_duplicates(
                        subset=["order_id"], keep="first"
                    ) if "order_id" in frames[0].columns else pd.concat(frames, ignore_index=True)
                meta["best_candidate_joint"] = cand_joint if isinstance(cand_joint, dict) else {}
                meta["best_candidate_source_orders_df"] = cand_source_orders.copy() if isinstance(cand_source_orders, pd.DataFrame) else pd.DataFrame()
                meta["best_candidate_extra_dropped_df"] = cand_extra_drop.copy() if isinstance(cand_extra_drop, pd.DataFrame) else pd.DataFrame()
            return pd.DataFrame(), pd.DataFrame(), failed_drop_df, meta
        print(f"[APS][legacy_fallback] diagnostics={diagnostics}")
        print("[APS][FALLBACK] type=legacy_fallback, reason=joint_and_semantic_failed")
        final_df, rounds_df = run_legacy_schedule(
            orders_path=str(req.orders_path),
            steel_info_path=str(req.steel_info_path),
            output_path=str(req.output_path),
            config=current_cfg,
            prepared_orders=orders_df.copy(),
        )
        return final_df, rounds_df, pd.DataFrame(), _build_meta(
            req,
            engine_used="legacy_fallback",
            main_path="legacy",
            fallback_used=True,
            fallback_type="legacy_fallback",
            fallback_reason="joint_and_semantic_failed",
            fallback_trace=fallback_trace,
            used_local_routing=False,
            local_routing_role="not_used",
            input_order_count=len(orders_df),
            failure_diagnostics=diagnostics,
            joint_estimates={},
            strict_template_edges_enabled=bool(current_cfg.model.strict_template_edges),
            unroutable_slot_count=0,
            slot_route_details=[],
            template_graph_health=str(health.get("template_graph_health", "UNKNOWN")),
            precheck_autorelax_applied=precheck_autorelax_applied,
            solve_attempt_count=solve_attempt_count,
            fallback_attempt_count=fallback_attempt_count,
            early_stop_reason=early_stop_reason,
            early_stop_deferred_for_semantic_fallback=early_stop_deferred_for_semantic_fallback,
            semantic_fallback_first_attempt_status=semantic_fallback_first_attempt_status,
            assignment_pressure_mode="relaxed" if "relaxed" in str(current_cfg.model.profile_name) else "normal",
            effective_config=current_cfg,
            template_build_seconds=template_build_seconds,
            joint_master_seconds=joint_master_seconds,
            local_router_seconds=local_router_seconds,
            fallback_total_seconds=fallback_total_seconds,
            feasibility_evidence=feasibility_evidence,
            failure_mode=failure_mode,
            evidence_level=evidence_level,
            top_infeasibility_signals=top_infeasibility_signals,
        )


    print("[APS][FALLBACK] type=legacy_fallback, reason=no_preprocessed_orders")
    final_df, rounds_df = run_legacy_schedule(
        orders_path=str(req.orders_path),
        steel_info_path=str(req.steel_info_path),
        output_path=str(req.output_path),
        config=req.config,
    )
    return final_df, rounds_df, pd.DataFrame(), _build_meta(
        req,
        engine_used="legacy_fallback",
        main_path="legacy",
        fallback_used=True,
        fallback_type="legacy_fallback",
        fallback_reason="no_preprocessed_orders",
        fallback_trace=[],
        used_local_routing=False,
        local_routing_role="not_used",
        input_order_count=0,
        failure_diagnostics={},
        joint_estimates={},
        strict_template_edges_enabled=bool(req.config.model.strict_template_edges),
        unroutable_slot_count=0,
        slot_route_details=[],
        assignment_pressure_mode="relaxed" if "relaxed" in str(req.config.model.profile_name) else "normal",
        template_build_seconds=0.0,
        joint_master_seconds=0.0,
        local_router_seconds=0.0,
        fallback_total_seconds=0.0,
    )


__all__ = [
    "_run_global_joint_model",
    "_run_unified_master_skeleton",
    "_solve_slot_route_with_templates",
    "_effective_global_prune_cap",
    "_semantic_fallback_configs",
    "solve_master_model",
]
