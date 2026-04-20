from __future__ import annotations

from collections import Counter
from typing import Any

import pandas as pd

from aps_cp_sat.config import PlannerConfig
from aps_cp_sat.model.candidate_graph_types import (
    CandidateEdge,
    CandidateGraphBuildResult,
    DIRECT_EDGE,
    REAL_BRIDGE_EDGE,
    TransitionCheckResult,
    VIRTUAL_BRIDGE_EDGE,
    VIRTUAL_BRIDGE_FAMILY_EDGE,
)
from aps_cp_sat.transition.bridge_rules import (
    _hard_direct_step_ok,
    _temp_transition_ok,
    _thick_ok,
    _txt,
    reverse_step_within_applicable_limit,
)


def _order_records(orders_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if not isinstance(orders_df, pd.DataFrame) or orders_df.empty or "order_id" not in orders_df.columns:
        return {}
    return {str(row.get("order_id", "")): dict(row) for row in orders_df.to_dict("records")}


def _normalize_edge_type(edge_type: str) -> str:
    raw = str(edge_type or DIRECT_EDGE)
    if raw == VIRTUAL_BRIDGE_EDGE:
        return VIRTUAL_BRIDGE_FAMILY_EDGE
    if raw in {DIRECT_EDGE, REAL_BRIDGE_EDGE, VIRTUAL_BRIDGE_FAMILY_EDGE}:
        return raw
    return DIRECT_EDGE


def _bridge_family(edge_type: str) -> str:
    if edge_type == DIRECT_EDGE:
        return "DIRECT"
    if edge_type == REAL_BRIDGE_EDGE:
        return "REAL"
    if edge_type == VIRTUAL_BRIDGE_FAMILY_EDGE:
        return "VIRTUAL_FAMILY"
    return "UNKNOWN"


def explain_transition_infeasible(a: dict, b: dict, cfg: PlannerConfig) -> TransitionCheckResult:
    """Explain the first hard-rule reason preventing direct adjacency."""
    rule = cfg.rule
    width_a = float(a.get("width", 0.0) or 0.0)
    width_b = float(b.get("width", 0.0) or 0.0)
    if width_b > width_a and not reverse_step_within_applicable_limit(a, b, rule):
        return TransitionCheckResult(False, "WIDTH_RULE_FAIL", {"from_width": width_a, "to_width": width_b})
    if width_a - width_b > float(rule.max_width_drop):
        return TransitionCheckResult(False, "WIDTH_RULE_FAIL", {"from_width": width_a, "to_width": width_b})
    if not _thick_ok(float(a.get("thickness", 0.0) or 0.0), float(b.get("thickness", 0.0) or 0.0)):
        return TransitionCheckResult(
            False,
            "THICKNESS_RULE_FAIL",
            {"from_thickness": a.get("thickness"), "to_thickness": b.get("thickness")},
        )
    if not _temp_transition_ok(a, b, rule):
        return TransitionCheckResult(
            False,
            "TEMP_OVERLAP_FAIL",
            {
                "from_temp": [a.get("temp_min"), a.get("temp_max")],
                "to_temp": [b.get("temp_min"), b.get("temp_max")],
            },
        )
    ga = _txt(a.get("steel_group", "")).upper()
    gb = _txt(b.get("steel_group", "")).upper()
    if ga != gb and ga not in {"PC", "普碳", "VIRTUAL_PC"} and gb not in {"PC", "普碳", "VIRTUAL_PC"}:
        return TransitionCheckResult(False, "GROUP_SWITCH_FAIL", {"from_group": ga, "to_group": gb})
    return TransitionCheckResult(False, "UNKNOWN_PAIR_INVALID", {})


def check_direct_transition(a: dict, b: dict, cfg: PlannerConfig) -> TransitionCheckResult:
    if _hard_direct_step_ok(a, b, cfg.rule):
        return TransitionCheckResult(True, "OK", {})
    return explain_transition_infeasible(a, b, cfg)


def check_real_bridge_transition(template_row: dict, cfg: PlannerConfig | None = None) -> TransitionCheckResult:
    ok = str(template_row.get("edge_type", "")) == REAL_BRIDGE_EDGE
    return TransitionCheckResult(
        hard_feasible=ok,
        reason="OK" if ok else "BRIDGE_PATH_NOT_REAL",
        explain={"edge_type": template_row.get("edge_type"), "real_bridge_order_id": template_row.get("real_bridge_order_id", "")},
    )


def check_virtual_bridge_family(template_row: dict, cfg: PlannerConfig | None = None) -> TransitionCheckResult:
    raw_type = str(template_row.get("edge_type", ""))
    ok = raw_type in {VIRTUAL_BRIDGE_EDGE, VIRTUAL_BRIDGE_FAMILY_EDGE}
    bridge_count = int(template_row.get("bridge_count", 0) or 0)
    max_chain = int(getattr(getattr(cfg, "rule", None), "max_virtual_chain", 10**9) if cfg is not None else 10**9)
    if ok and bridge_count > max_chain:
        return TransitionCheckResult(False, "MAX_VIRTUAL_CHAIN_FAIL", {"bridge_count": bridge_count, "max_chain": max_chain})
    return TransitionCheckResult(
        hard_feasible=ok,
        reason="OK" if ok else "BRIDGE_PATH_NOT_VIRTUAL",
        explain={"edge_type": raw_type, "bridge_count": bridge_count},
    )


def _bridge_family_type(row: dict) -> str:
    """Derive bridge_family for VIRTUAL_BRIDGE_FAMILY_EDGE from template row metadata."""
    raw = str(row.get("bridge_family", "")).strip().upper()
    if raw in {"WIDTH_GROUP", "THICKNESS", "GROUP_TRANSITION", "MIXED"}:
        return raw
    # Infer from other fields if not set explicitly
    bridge_count = int(row.get("bridge_count", 0) or 0)
    if bridge_count >= 3:
        return "GROUP_TRANSITION"
    elif bridge_count >= 2:
        return "MIXED"
    else:
        return "WIDTH_GROUP"


def _candidate_edge_from_template(row: dict, order_map: dict[str, dict[str, Any]], cfg: PlannerConfig) -> CandidateEdge:
    raw_edge_type = str(row.get("edge_type", DIRECT_EDGE) or DIRECT_EDGE)
    edge_type = _normalize_edge_type(raw_edge_type)
    from_id = str(row.get("from_order_id", ""))
    to_id = str(row.get("to_order_id", ""))
    if edge_type == DIRECT_EDGE:
        check = check_direct_transition(order_map.get(from_id, {}), order_map.get(to_id, {}), cfg)
    elif edge_type == REAL_BRIDGE_EDGE:
        check = check_real_bridge_transition(row, cfg)
    else:
        check = check_virtual_bridge_family(row, cfg)
    bridge_count = int(row.get("bridge_count", 0) or 0)

    # ---- Safe NaN guard for int conversion ----
    def _safe_int(v, fallback):
        try:
            f = float(v)
            return int(f) if f == f else fallback  # NaN check via f != f
        except (TypeError, ValueError):
            return fallback

    # ---- Family edge extension fields ----
    is_virtual_family = (edge_type == VIRTUAL_BRIDGE_FAMILY_EDGE)
    estimated_count_min = _safe_int(row.get("estimated_bridge_count_min"), bridge_count)
    estimated_count_max = _safe_int(row.get("estimated_bridge_count_max"), bridge_count)
    requires_pc = bool(row.get("requires_pc_transition", False))

    # Explain payload for family edge diagnostics
    if is_virtual_family:
        explain_payload = {
            "bridge_family": _bridge_family_type(row),
            "estimated_bridge_count_min": estimated_count_min,
            "estimated_bridge_count_max": estimated_count_max,
            "requires_pc_transition": requires_pc,
            "bridge_count": bridge_count,
            "virtual_tons": float(row.get("virtual_tons", 0.0) or 0.0),
        }
    else:
        explain_payload = {}

    return CandidateEdge(
        from_order_id=from_id,
        to_order_id=to_id,
        line=str(row.get("line", "")),
        edge_type=edge_type,
        bridge_family=_bridge_family(edge_type),
        estimated_bridge_count=bridge_count,
        estimated_bridge_count_min=estimated_count_min,
        estimated_bridge_count_max=estimated_count_max,
        requires_pc_transition=requires_pc,
        dominant_fail_reason=str(check.reason),
        hard_feasible=bool(check.hard_feasible),
        template_cost=float(row.get("cost", 0.0) or 0.0),
        estimated_reverse_cost=float(row.get("physical_reverse_count", row.get("logical_reverse_flag", 0)) or 0.0),
        estimated_ton_effect=float(row.get("virtual_tons", 0.0) or 0.0),
        metadata={
            "raw_edge_type": raw_edge_type,
            "template_row": dict(row),
            "explain": dict(check.explain),
            "explain_payload": explain_payload,
        },
    )


def build_candidate_graph(
    orders_df: pd.DataFrame,
    template_df: pd.DataFrame,
    cfg: PlannerConfig,
    *,
    scan_infeasible_direct_pairs: bool = False,
) -> CandidateGraphBuildResult:
    """Build a read-only Candidate Graph view from existing transition templates."""
    order_map = _order_records(orders_df)
    edges: list[CandidateEdge] = []
    reason_counts: Counter[str] = Counter()
    if isinstance(template_df, pd.DataFrame) and not template_df.empty:
        for row in template_df.to_dict("records"):
            edge = _candidate_edge_from_template(row, order_map, cfg)
            edges.append(edge)
            if not edge.hard_feasible:
                reason_counts[edge.dominant_fail_reason] += 1

    if scan_infeasible_direct_pairs and order_map:
        records = list(order_map.values())
        for a in records:
            for b in records:
                if str(a.get("order_id")) == str(b.get("order_id")):
                    continue
                check = check_direct_transition(a, b, cfg)
                if not check.hard_feasible:
                    reason_counts[check.reason] += 1

    edge_type_counts = Counter(edge.edge_type for edge in edges)

    # ---- Virtual family edge pruning diagnostics ----
    max_chain = int(getattr(getattr(cfg, "rule", None), "max_virtual_chain", 10**9) if cfg is not None else 10**9)
    filtered_by_chain_limit = 0
    filtered_by_temp = 0
    filtered_by_group = 0
    topk_pruned = 0

    # Per from_order_id family: track top-k and prune
    family_topk_per_from: dict[str, list[tuple[int, CandidateEdge]]] = {}
    for edge in edges:
        if edge.edge_type != VIRTUAL_BRIDGE_FAMILY_EDGE:
            continue
        # Prune 1: max chain exceeded
        if edge.effective_max_bridge_count() > max_chain:
            filtered_by_chain_limit += 1
            continue
        # Prune 2: requires PC but no PC transition possible (temperature window)
        if edge.requires_pc_transition:
            from_rec = order_map.get(str(edge.from_order_id), {})
            to_rec = order_map.get(str(edge.to_order_id), {})
            from_grp = str(from_rec.get("steel_group", "")).upper()
            to_grp = str(to_rec.get("steel_group", "")).upper()
            from_temp_ok = from_rec and (
                float(from_rec.get("temp_max", 0) or 0) > 0
            )
            to_temp_ok = to_rec and (
                float(to_rec.get("temp_max", 0) or 0) > 0
            )
            if not (from_temp_ok and to_temp_ok):
                filtered_by_temp += 1
                continue
            # Group transition: require PC or same group
            if not (
                from_grp == to_grp
                or from_grp in {"PC", "普碳", "VIRTUAL_PC"}
                or to_grp in {"PC", "普碳", "VIRTUAL_PC"}
            ):
                filtered_by_group += 1
                continue

        fid = str(edge.from_order_id)
        if fid not in family_topk_per_from:
            family_topk_per_from[fid] = []
        family_topk_per_from[fid].append((edge.effective_max_bridge_count(), edge))

    # Top-k pruning per from_order_id (keep top 5 per family)
    topk = 5
    for fid, candidates in family_topk_per_from.items():
        if len(candidates) > topk:
            topk_pruned += len(candidates) - topk

    # Family bridge count distribution
    bridge_family_counts: dict[str, int] = {}
    for edge in edges:
        if edge.edge_type == VIRTUAL_BRIDGE_FAMILY_EDGE:
            fam = str(edge.metadata.get("explain_payload", {}).get("bridge_family", "MIXED"))
            bridge_family_counts[fam] = bridge_family_counts.get(fam, 0) + 1

    diagnostics = {
        "candidate_graph_edge_count": int(len(edges)),
        "candidate_graph_direct_edge_count": int(edge_type_counts.get(DIRECT_EDGE, 0)),
        "candidate_graph_real_bridge_edge_count": int(edge_type_counts.get(REAL_BRIDGE_EDGE, 0)),
        "candidate_graph_virtual_bridge_family_edge_count": int(edge_type_counts.get(VIRTUAL_BRIDGE_FAMILY_EDGE, 0)),
        "virtual_family_edge_count": int(edge_type_counts.get(VIRTUAL_BRIDGE_FAMILY_EDGE, 0)),
        "virtual_family_filtered_by_chain_limit_count": int(filtered_by_chain_limit),
        "virtual_family_filtered_by_temp_count": int(filtered_by_temp),
        "virtual_family_filtered_by_group_count": int(filtered_by_group),
        "virtual_family_topk_pruned_count": int(topk_pruned),
        "candidate_graph_filtered_by_width_count": int(reason_counts.get("WIDTH_RULE_FAIL", 0)),
        "candidate_graph_filtered_by_thickness_count": int(reason_counts.get("THICKNESS_RULE_FAIL", 0)),
        "candidate_graph_filtered_by_temp_count": int(reason_counts.get("TEMP_OVERLAP_FAIL", 0)),
        "candidate_graph_filtered_by_group_count": int(reason_counts.get("GROUP_SWITCH_FAIL", 0)),
        "candidate_graph_filtered_by_max_virtual_chain_count": int(reason_counts.get("MAX_VIRTUAL_CHAIN_FAIL", 0)),
        "candidate_graph_reason_histogram": dict(reason_counts),
        "virtual_family_bridge_family_counts": bridge_family_counts,
        "virtual_family_max_chain_limit": int(max_chain),
    }
    return CandidateGraphBuildResult(edges=edges, diagnostics=diagnostics)
