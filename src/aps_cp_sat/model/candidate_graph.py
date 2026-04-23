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
    is_virtual_family_frontload_eligible,
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

    # ---- Virtual family edge pruning pipeline ----
    # Raw family edge count (before any pruning)
    virtual_family_raw_count = int(edge_type_counts.get(VIRTUAL_BRIDGE_FAMILY_EDGE, 0))

    max_chain = int(getattr(getattr(cfg, "rule", None), "max_virtual_chain", 10**9) if cfg is not None else 10**9)
    filtered_by_chain_limit = 0
    filtered_by_temp = 0
    filtered_by_group = 0
    filtered_by_family_allowlist = 0
    filtered_by_frontload_eligibility = 0
    topk_pruned = 0
    global_cap_pruned = 0

    # Per from_order_id + bridge_family: track candidates for top-k
    family_topk_key: dict[tuple[str, str, str], list[tuple[int, float, CandidateEdge]]] = {}
    # family_topk_key = (from_order_id, line, bridge_family) -> [(bridge_count, cost, edge), ...]

    # ---- Phase 1: basic feasibility pruning ----
    eligible_family_edges: list[CandidateEdge] = []
    # Extract model config once for reuse in later phases
    model = getattr(cfg, "model", None) if cfg else None
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

        # ---- Phase 2: family allowlist pruning ----
        if model is not None and getattr(model, "virtual_family_frontload_enabled", False):
            allowed_families = getattr(model, "virtual_family_frontload_allowed_families", [])
            if allowed_families:
                bridge_family = str(edge.metadata.get("explain_payload", {}).get("bridge_family", ""))
                if bridge_family not in allowed_families:
                    filtered_by_family_allowlist += 1
                    continue

        # ---- Phase 3: frontload eligibility gating ----
        eligible, _ = is_virtual_family_frontload_eligible(edge, cfg, None)
        if not eligible:
            filtered_by_frontload_eligibility += 1
            continue

        eligible_family_edges.append(edge)

    # ---- Phase 4: top-k per (from_order_id, line, bridge_family) ----
    topk_per = int(getattr(model, "virtual_family_frontload_global_topk_per_from", 2) if model else 2)
    for edge in eligible_family_edges:
        fam = str(edge.metadata.get("explain_payload", {}).get("bridge_family", "MIXED"))
        key = (str(edge.from_order_id), str(edge.line), fam)
        if key not in family_topk_key:
            family_topk_key[key] = []
        family_topk_key[key].append((edge.effective_max_bridge_count(), edge.template_cost, edge))

    topk_kept_edges: set[CandidateEdge] = set()
    for key, candidates in family_topk_key.items():
        # Sort by bridge_count asc, then cost asc
        candidates.sort(key=lambda x: (x[0], x[1]))
        kept = candidates[:topk_per]
        pruned = len(candidates) - len(kept)
        topk_pruned += pruned
        topk_kept_edges.update(e[2] for e in kept)

    # ---- Phase 5: global total cap (strict pool for greedy constructive) ----
    global_max = int(getattr(model, "virtual_family_frontload_global_max_edges_total", 360) if model else 360)
    if len(topk_kept_edges) > global_max:
        # Sort by priority: bridge_count asc, cost asc, reverse_cost asc, expected_underfill_gain desc
        def _global_cap_priority(e: CandidateEdge) -> tuple:
            payload = e.metadata.get("explain_payload", {})
            return (
                e.effective_max_bridge_count(),
                e.template_cost,
                e.estimated_reverse_cost,
                -float(payload.get("expected_underfill_gain", 0.0) or 0.0),
                e.from_order_id,
            )

        sorted_edges = sorted(topk_kept_edges, key=_global_cap_priority)
        topk_kept_edges = set(sorted_edges[:global_max])
        global_cap_pruned = len(sorted_edges) - global_max

    # ---- Build dual pool: strict global pool + wider repair pool ----
    # Global pool: already in topk_kept_edges (strict: topk_per + global_max)
    # Repair pool: wider (topk_per + alns_extra_topk) + repair_max_edges_total
    # The repair pool includes ALL family edges from the wider top-k that are not already in global pool.
    repair_family_edges_list: list[CandidateEdge] = []
    repair_topk_pruned = 0
    repair_cap_pruned = 0

    # Apply wider top-k: (global topk_per + alns_extra_topk)
    alns_extra_topk = int(getattr(model, "virtual_family_frontload_alns_only_extra_topk", 4) if model else 4)
    wider_topk_per = topk_per + alns_extra_topk  # e.g., 3 + 4 = 7
    wider_topk_kept: set[CandidateEdge] = set()
    wider_family_topk_key: dict[tuple[str, str, str], list[tuple[int, float, CandidateEdge]]] = {}
    for edge in eligible_family_edges:
        fam = str(edge.metadata.get("explain_payload", {}).get("bridge_family", "MIXED"))
        key = (str(edge.from_order_id), str(edge.line), fam)
        if key not in wider_family_topk_key:
            wider_family_topk_key[key] = []
        wider_family_topk_key[key].append((edge.effective_max_bridge_count(), edge.template_cost, edge))

    for key, candidates in wider_family_topk_key.items():
        candidates.sort(key=lambda x: (x[0], x[1]))
        kept = candidates[:wider_topk_per]
        pruned = len(candidates) - len(kept)
        repair_topk_pruned += pruned
        wider_topk_kept.update(e[2] for e in kept)

    # Apply repair pool cap
    repair_max = int(getattr(model, "virtual_family_frontload_repair_max_edges_total", 900) if model else 900)
    # wider_topk_kept includes edges from global top-k + extra ALNS edges
    # We want: ALL wider_topk_kept edges, then cap
    # But deduplicate: global pool already has topk_kept_edges ⊂ wider_topk_kept
    # repair pool = wider_topk_kept edges (including global ones, dedup by set)
    # Then cap
    all_repair_candidates = set(wider_topk_kept)  # wider top-k set
    if len(all_repair_candidates) > repair_max:
        def _repair_cap_priority(e: CandidateEdge) -> tuple:
            payload = e.metadata.get("explain_payload", {})
            return (
                e.effective_max_bridge_count(),
                e.template_cost,
                e.estimated_reverse_cost,
                -float(payload.get("expected_underfill_gain", 0.0) or 0.0),
                e.from_order_id,
            )
        sorted_repair = sorted(all_repair_candidates, key=_repair_cap_priority)
        all_repair_candidates = set(sorted_repair[:repair_max])
        repair_cap_pruned = len(sorted_repair) - repair_max

    # Build repair_family_edges: wider top-k kept, capped, as list
    repair_family_edges_list = list(all_repair_candidates)
    repair_family_edges_list.sort(key=lambda e: (
        e.effective_max_bridge_count(), e.template_cost, e.from_order_id, e.to_order_id
    ))

    # Diagnostics for repair pool
    repair_family_edge_pool_diagnostics = {
        "virtual_family_repair_pool_wider_topk_per": int(wider_topk_per),
        "virtual_family_repair_pool_repair_max_edges_total": int(repair_max),
        "virtual_family_repair_pool_wider_topk_candidates": int(len(wider_topk_kept)),
        "virtual_family_repair_pool_topk_pruned_count": int(repair_topk_pruned),
        "virtual_family_repair_pool_cap_pruned_count": int(repair_cap_pruned),
        "virtual_family_repair_pool_kept_count": int(len(repair_family_edges_list)),
        # Overlap with global pool
        "virtual_family_repair_pool_global_overlap_count": int(len(wider_topk_kept & topk_kept_edges)),
        "virtual_family_repair_pool_extra_count": int(len(wider_topk_kept) - len(wider_topk_kept & topk_kept_edges)),
    }
    repair_family_edge_count = len(repair_family_edges_list)

    # Rebuild final edges list: keep all non-family edges + topk-kept family edges
    final_edges: list[CandidateEdge] = []
    for edge in edges:
        if edge.edge_type != VIRTUAL_BRIDGE_FAMILY_EDGE:
            final_edges.append(edge)
        elif edge in topk_kept_edges:
            final_edges.append(edge)

    # Family bridge count distribution (for diagnostics)
    bridge_family_counts: dict[str, int] = {}
    for edge in final_edges:
        if edge.edge_type == VIRTUAL_BRIDGE_FAMILY_EDGE:
            fam = str(edge.metadata.get("explain_payload", {}).get("bridge_family", "MIXED"))
            bridge_family_counts[fam] = bridge_family_counts.get(fam, 0) + 1

    final_type_counts = Counter(edge.edge_type for edge in final_edges)
    def _is_virtual_record(rec: dict[str, Any]) -> bool:
        raw = rec.get("is_virtual", False)
        if pd.isna(raw):
            raw = False
        return bool(raw) or str(rec.get("virtual_origin", "")) == "prebuilt_inventory"

    virtual_node_ids = {
        str(rec.get("order_id", ""))
        for rec in order_map.values()
        if _is_virtual_record(rec)
    }
    virtual_edge_count = 0
    virtual_virtual_edge_count = 0
    real_virtual_edge_count = 0
    virtual_real_edge_count = 0
    for edge in final_edges:
        from_virtual = str(edge.from_order_id) in virtual_node_ids
        to_virtual = str(edge.to_order_id) in virtual_node_ids
        if from_virtual or to_virtual:
            virtual_edge_count += 1
        if from_virtual and to_virtual:
            virtual_virtual_edge_count += 1
        elif (not from_virtual) and to_virtual:
            real_virtual_edge_count += 1
        elif from_virtual and (not to_virtual):
            virtual_real_edge_count += 1
    real_bridge_capable_orders = set()
    for edge in final_edges:
        if edge.edge_type != REAL_BRIDGE_EDGE:
            continue
        if str(edge.from_order_id) not in virtual_node_ids:
            real_bridge_capable_orders.add(str(edge.from_order_id))
        if str(edge.to_order_id) not in virtual_node_ids:
            real_bridge_capable_orders.add(str(edge.to_order_id))

    diagnostics = {
        "candidate_graph_edge_count": int(len(final_edges)),
        "candidate_graph_virtual_node_count": int(len(virtual_node_ids)),
        "candidate_graph_virtual_edge_count": int(virtual_edge_count),
        "candidate_graph_virtual_virtual_edge_count": int(virtual_virtual_edge_count),
        "candidate_graph_real_virtual_edge_count": int(real_virtual_edge_count),
        "candidate_graph_virtual_real_edge_count": int(virtual_real_edge_count),
        "candidate_graph_real_bridge_capable_order_count": int(len(real_bridge_capable_orders)),
        "candidate_graph_direct_edge_count": int(final_type_counts.get(DIRECT_EDGE, 0)),
        "candidate_graph_real_bridge_edge_count": int(final_type_counts.get(REAL_BRIDGE_EDGE, 0)),
        "candidate_graph_virtual_bridge_family_edge_count": int(final_type_counts.get(VIRTUAL_BRIDGE_FAMILY_EDGE, 0)),
        "virtual_family_edge_count": int(final_type_counts.get(VIRTUAL_BRIDGE_FAMILY_EDGE, 0)),
        # Raw counts (before pruning)
        "candidate_graph_virtual_bridge_family_edge_raw_count": int(virtual_family_raw_count),
        "candidate_graph_virtual_bridge_family_edge_kept_count": int(final_type_counts.get(VIRTUAL_BRIDGE_FAMILY_EDGE, 0)),
        # Dual-pool: global (strict) vs repair (wider)
        "candidate_graph_virtual_bridge_family_edge_global_kept_count": int(final_type_counts.get(VIRTUAL_BRIDGE_FAMILY_EDGE, 0)),
        "candidate_graph_virtual_bridge_family_edge_repair_kept_count": int(repair_family_edge_count),
        # Pruning diagnostics
        "virtual_family_filtered_by_chain_limit_count": int(filtered_by_chain_limit),
        "virtual_family_filtered_by_temp_count": int(filtered_by_temp),
        "virtual_family_filtered_by_group_count": int(filtered_by_group),
        "virtual_family_frontload_filtered_by_family_count": int(filtered_by_family_allowlist),
        "virtual_family_frontload_filtered_by_context_count": int(filtered_by_frontload_eligibility),
        "virtual_family_frontload_topk_pruned_count": int(topk_pruned),
        "virtual_family_frontload_global_cap_pruned_count": int(global_cap_pruned),
        "virtual_family_repair_pool_topk_pruned_count": int(repair_topk_pruned),
        "virtual_family_repair_pool_cap_pruned_count": int(repair_cap_pruned),
        # Legacy diagnostics (for compatibility)
        "virtual_family_topk_pruned_count": int(topk_pruned),
        "candidate_graph_filtered_by_width_count": int(reason_counts.get("WIDTH_RULE_FAIL", 0)),
        "candidate_graph_filtered_by_thickness_count": int(reason_counts.get("THICKNESS_RULE_FAIL", 0)),
        "candidate_graph_filtered_by_temp_count": int(reason_counts.get("TEMP_OVERLAP_FAIL", 0)),
        "candidate_graph_filtered_by_group_count": int(reason_counts.get("GROUP_SWITCH_FAIL", 0)),
        "candidate_graph_filtered_by_max_virtual_chain_count": int(reason_counts.get("MAX_VIRTUAL_CHAIN_FAIL", 0)),
        "candidate_graph_reason_histogram": dict(reason_counts),
        "virtual_family_bridge_family_counts": bridge_family_counts,
        "virtual_family_max_chain_limit": int(max_chain),
        "virtual_family_frontload_global_topk_per_from": int(topk_per),
        "virtual_family_frontload_global_max_edges_total": int(global_max),
    }
    return CandidateGraphBuildResult(
        edges=final_edges,
        diagnostics=diagnostics,
        repair_family_edges=repair_family_edges_list,
        repair_family_edge_count=repair_family_edge_count,
        repair_family_edge_pool_diagnostics=repair_family_edge_pool_diagnostics,
    )
