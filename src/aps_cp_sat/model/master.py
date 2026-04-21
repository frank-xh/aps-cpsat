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

Rule semantics snapshot (updated):
- line compatibility: HARD
- adjacent transition feasibility: HARD via template filtering
- campaign ton upper bound: HARD
- campaign ton lower bound: HARD (changed from STRONG_SOFT)
- unassigned real orders: STRONG_SOFT
- virtual slab ratio / quantity: STRONG_SOFT
- reverse-width count / total rise: HARD in template rules

Two-phase solving:
- Phase "feasibility": minimizes unassigned count and slot count; hard constraints only
- Phase "optimize": full soft optimization on top of feasible solution
"""

import pandas as pd


# =============================================================================
# UNIFIED DROP PRIORITY HELPER
# Priority order (MUST be used by ALL drop sorting logic):
# 1. global_iso (GLOBAL_ISOLATED_ORDER)
# 2. would_break_ton_window_if_kept (TON_WINDOW_INFEASIBLE)
# 3. no_feasible_line (NO_FEASIBLE_LINE)
# 4. bridge_required_but_not_supported (BRIDGE_REQUIRED_BUT_NOT_SUPPORTED)
# 5. adjacency_risk (ADJACENCY_VIOLATION_RISK) [Priority 2: pre-strict to structure_fallback]
# 6. iso (SLOT_ROUTING_RISK_TOO_HIGH)
# 7. low_priority (LOW_PRIORITY_DROP)
# 8-11. width_outlier, thick_outlier, due_rank, tons (tiebreakers)
# =============================================================================
DROP_PRIORITY_ORDER = "GLOBAL_ISO > TON_WINDOW > NO_FEASIBLE_LINE > BRIDGE_NOT_SUPPORTED > ADJACENCY_RISK > ISO > LOW_PRIORITY"


def _drop_priority_key(item: dict) -> tuple:
    """Unified drop priority key for sorting.

    Supports two dict structures:
    - item[1] structure: {"global_iso": bool, "would_break_ton_window_if_kept": bool, ...}
    - item structure (scored): {"globally_isolated": bool, "would_break_ton_window_if_kept": bool, ...}

    Returns a tuple where lower values = higher priority (drop first).
    """
    # Handle nested dict structure (item[1])
    if isinstance(item, tuple) and len(item) == 2:
        d = item[1]
    else:
        d = item

    # Extract fields with defaults
    global_iso = bool(d.get("global_iso", False) or d.get("globally_isolated", False))
    ton_window = bool(d.get("would_break_ton_window_if_kept", False))
    no_feasible = bool(d.get("no_feasible_line", False))
    bridge_not_supported = bool(d.get("bridge_required_but_not_supported", False))
    # Priority 5: adjacency_risk (NEW) - orders that would cause adjacency violations if kept
    # Adjacency violations are HARD constraints (width_jump, thickness_violation, temp_conflict)
    # Priority order: global_iso > ton_window > no_feasible > bridge > adjacency_risk > iso > low_priority
    adjacency_risk = bool(d.get("adjacency_risk", False))
    iso = bool(d.get("iso", False) or d.get("isolated", False))
    
    # low_priority: check order dict or direct priority field
    if "order" in d and isinstance(d["order"], dict):
        priority = int(d["order"].get("priority", 0) or 0)
    else:
        priority = int(d.get("priority", 0) or 0)
    low_priority = priority <= 0

    # Outlier and tiebreaker fields
    width_outlier = float(d.get("width_outlier", 0) or 0)
    thick_outlier = float(d.get("thick_outlier", 0) or 0)
    
    if "order" in d and isinstance(d["order"], dict):
        due_rank = int(d["order"].get("due_rank", 9) or 9)
        tons = float(d["order"].get("tons", 0.0) or 0.0)
    else:
        due_rank = int(d.get("due_rank", 9) or 9)
        tons = float(d.get("tons", 0.0) or 0.0)

    return (
        0 if global_iso else 1,  # 1. GLOBAL_ISOLATED first
        0 if ton_window else 1,  # 2. TON_WINDOW second
        0 if no_feasible else 1,  # 3. NO_FEASIBLE_LINE third
        0 if bridge_not_supported else 1,  # 4. BRIDGE_NOT_SUPPORTED fourth
        0 if adjacency_risk else 1,  # 5. ADJACENCY_RISK fifth (Priority 2: pre-strict adjacency to structure_fallback)
        0 if iso else 1,  # 6. ISO sixth
        0 if low_priority else 1,  # 7. LOW_PRIORITY seventh
        -width_outlier,  # 8. width outlier (larger first)
        -thick_outlier,  # 9. thickness outlier (larger first)
        -due_rank,  # 10. due date (earlier first)
        tons,  # 11. tons (larger first)
    )


def _dominant_drop_reason(item: dict) -> str:
    """Determine dominant drop reason based on fixed priority order.

    Must match the sort priority order in _drop_priority_key.
    Supports two dict structures:
    - item[1] structure: (oid, {"global_iso": bool, ...}) from ranked_rows
    - item structure (scored): {"globally_isolated": bool, ...} from scored dict
    """
    # Handle nested dict structure (item[1])
    if isinstance(item, tuple) and len(item) == 2:
        d = item[1]
    else:
        d = item

    global_iso = bool(d.get("global_iso", False) or d.get("globally_isolated", False))
    ton_window = bool(d.get("would_break_ton_window_if_kept", False))
    no_feasible = bool(d.get("no_feasible_line", False))
    bridge_not_supported = bool(d.get("bridge_required_but_not_supported", False))
    # Priority 5: adjacency_risk (Priority 2: pre-strict to structure_fallback)
    adjacency_risk = bool(d.get("adjacency_risk", False))
    iso = bool(d.get("iso", False) or d.get("isolated", False))
    
    if "order" in d and isinstance(d["order"], dict):
        priority = int(d["order"].get("priority", 0) or 0)
    else:
        priority = int(d.get("priority", 0) or 0)
    low_priority = priority <= 0

    # Apply fixed priority order for dominant reason
    if global_iso:
        return "GLOBAL_ISOLATED_ORDER"
    elif ton_window:
        return "TON_WINDOW_INFEASIBLE"
    elif no_feasible:
        return "NO_FEASIBLE_LINE"
    elif bridge_not_supported:
        return "BRIDGE_REQUIRED_BUT_NOT_SUPPORTED"
    elif adjacency_risk:
        return "ADJACENCY_VIOLATION_RISK"
    elif iso:
        return "SLOT_ROUTING_RISK_TOO_HIGH"
    elif low_priority:
        return "LOW_PRIORITY_DROP"
    else:
        reasons = d.get("reasons", []) or []
        return str(reasons[0]) if reasons else "OTHER"
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
from aps_cp_sat.model.constructive_lns_master import run_constructive_lns_master
from aps_cp_sat.model.local_router import _solve_slot_route_with_templates
from aps_cp_sat.transition import build_transition_templates
from aps_cp_sat.model.candidate_graph import build_candidate_graph


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
        would_break_ton_window_if_kept: bool = False,
        globally_isolated: bool = False,
        no_feasible_line: bool = False,
        bridge_required_but_not_supported: bool = False,
        # Priority 2: adjacency_risk parameter for pre-strict to structure_fallback
        adjacency_risk: bool = False,
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
                "would_break_ton_window_if_kept": False,
                "globally_isolated": False,
                "no_feasible_line": False,
                "bridge_required_but_not_supported": False,
                # Priority 2: adjacency_risk for pre-strict to structure_fallback
                "adjacency_risk": False,
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
        row["would_break_ton_window_if_kept"] = bool(row["would_break_ton_window_if_kept"] or would_break_ton_window_if_kept)
        row["globally_isolated"] = bool(row["globally_isolated"] or globally_isolated)
        row["no_feasible_line"] = bool(row["no_feasible_line"] or no_feasible_line)
        row["bridge_required_but_not_supported"] = bool(row["bridge_required_but_not_supported"] or bridge_required_but_not_supported)
        # Priority 2: adjacency_risk update
        row["adjacency_risk"] = bool(row["adjacency_risk"] or adjacency_risk)

    isolated_top = feasibility_evidence.get("isolated_orders_topn", []) if isinstance(feasibility_evidence, dict) else []
    isolated_lookup = {str(item.get("order_id", "")): item for item in isolated_top}

    # Analyze ton window feasibility for each order
    # If an order's tons would force a slot to be < 700 or > 2000, it breaks the ton window
    ton_min = float(getattr(cfg.rule, "campaign_ton_min", 700.0) or 700.0)
    ton_max = float(getattr(cfg.rule, "campaign_ton_max", 2000.0) or 2000.0)
    slot_upper_bound = int(joint.get("slot_upper_bound_by_line", {}).get("big_roll", 10))
    ton_window_breakers: dict[str, dict] = {}
    
    # FIX: raw_slots must be defined before use (moved from below)
    raw_slots = joint.get("slot_route_details", []) if isinstance(joint.get("slot_route_details", []), list) else []

    # Analyze bad slots for ton window issues
    for slot in raw_slots:
        line = str(slot.get("line", ""))
        slot_tons = float(slot.get("total_tons", 0) or 0)
        slot_orders = [str(v) for v in (slot.get("order_ids", []) or [])]

        for oid in slot_orders:
            if oid not in by_id:
                continue
            order_tons = float(by_id[oid].get("tons", 0) or 0)
            # If removing this order would reduce slot below min, it's a ton window breaker
            if slot_tons - order_tons < ton_min and slot_tons >= ton_min:
                ton_window_breakers[oid] = {
                    "would_break": True,
                    "reason": "REMOVAL_WOULD_DROP_BELOW_MIN",
                    "slot_tons": slot_tons,
                    "order_tons": order_tons,
                    "line": line,
                }
            # If order is too large (> 2000), it can't fit in any slot
            if order_tons > ton_max:
                ton_window_breakers[oid] = {
                    "would_break": True,
                    "reason": "ORDER_EXCEEDS_MAX",
                    "slot_tons": slot_tons,
                    "order_tons": order_tons,
                    "line": line,
                }
            # If keeping this order would require enabling too many slots (> upper bound)
            if slot_tons + order_tons > slot_upper_bound * ton_max:
                ton_window_breakers[oid] = {
                    "would_break": True,
                    "reason": "WOULD_EXCEED_SLOT_UPPER_BOUND",
                    "slot_tons": slot_tons,
                    "order_tons": order_tons,
                    "line": line,
                }

    for item in isolated_top:
        oid = str(item.get("order_id", ""))
        break_ton = oid in ton_window_breakers
        add_candidate(
            oid,
            140,
            "GLOBAL_ISOLATED_ORDER",
            "LOW_DEGREE_ORDER",
            risk_summary="global_isolated",
            would_break_slot_if_kept=True,
            would_break_ton_window_if_kept=break_ton,
            globally_isolated=True,
        )

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
            soid = str(oid)
            break_ton = soid in ton_window_breakers
            add_candidate(
                soid,
                130 + max(0, 12 - slot_idx),
                "SLOT_ROUTING_RISK_TOO_HIGH",
                "LOW_DEGREE_ORDER",
                slot=slot,
                risk_summary=slot_risk_hint,
                would_break_slot_if_kept=True,
                would_break_ton_window_if_kept=break_ton,
                globally_isolated=soid in isolated_lookup,
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
            # Check ton window breaker status
            break_ton = oid in ton_window_breakers
            no_feasible_line = bool(not _allowed_lines(rec))
            bridge_required_but_not_supported = False  # Not directly tracked here
            
            # Priority 2: Pre-strict adjacency risk
            # An order has adjacency risk if keeping it would cause adjacency violations:
            # - The slot has low template coverage (many orders can't transition)
            # - The slot has high pair gaps (many potential adjacency breaks)
            # - The order is in top_isolated_orders (limited outgoing edges)
            slot_coverage = float(slot.get("template_coverage_ratio", 1.0) or 1.0)
            pair_gap = int(slot.get("pair_gap_proxy", 0) or 0)
            zero_out = int(slot.get("zero_out_orders", 0) or 0)
            # adjacency_risk is HIGH if:
            # 1. The slot itself has poor template coverage (< 0.7) OR
            # 2. The slot has high pair gaps (> 5) OR
            # 3. The order is in the top_isolated_orders list (has zero outgoing edges)
            # These are signals that the order would cause adjacency violations if kept
            adjacency_risk = bool(
                slot_coverage < 0.7
                or pair_gap > 5
                or oid in set(str(v) for v in (slot.get("top_isolated_orders", []) or []))
            )
            
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
                        # NEW: ton window and line capability flags for sorting
                        "would_break_ton_window_if_kept": break_ton,
                        "no_feasible_line": no_feasible_line,
                        "bridge_required_but_not_supported": bridge_required_but_not_supported,
                        # Priority 2: adjacency_risk pre-strict to structure_fallback
                        "adjacency_risk": adjacency_risk,
                    },
                )
            )
        # FIXED priority order using unified helper (see DROP_PRIORITY_ORDER)
        ranked_rows.sort(key=_drop_priority_key)
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
            # Priority 2: adjacency_risk adds ADJACENCY_VIOLATION_RISK to dominant reasons
            # Dominant reason must match sort priority order:
            # 1. global_iso -> GLOBAL_ISOLATED_ORDER
            # 2. ton_window -> TON_WINDOW_INFEASIBLE
            # 3. no_feasible_line -> NO_FEASIBLE_LINE
            # 4. bridge -> BRIDGE_REQUIRED_BUT_NOT_SUPPORTED
            # 5. adjacency_risk -> ADJACENCY_VIOLATION_RISK [Priority 2: pre-strict]
            # 6. iso -> SLOT_ROUTING_RISK_TOO_HIGH
            # 7. low_priority -> LOW_PRIORITY_DROP
            # else -> CAPACITY_PRESSURE
            soid = str(oid)
            break_ton = soid in ton_window_breakers
            if feats["global_iso"]:
                dominant = "GLOBAL_ISOLATED_ORDER"
            elif break_ton:
                dominant = "TON_WINDOW_INFEASIBLE"
            elif feats["no_feasible_line"]:
                dominant = "NO_FEASIBLE_LINE"
            elif feats["bridge_required_but_not_supported"]:
                dominant = "BRIDGE_REQUIRED_BUT_NOT_SUPPORTED"
            elif feats.get("adjacency_risk", False):
                dominant = "ADJACENCY_VIOLATION_RISK"
            elif feats["iso"]:
                dominant = "SLOT_ROUTING_RISK_TOO_HIGH"
            elif feats["low_priority"]:
                dominant = "LOW_PRIORITY_DROP"
            else:
                dominant = "CAPACITY_PRESSURE"
            score = 95
            if feats["global_iso"]:
                score += 20
            if break_ton:
                score += 15
            if feats.get("adjacency_risk", False):
                # Priority 2: adjacency risk is HIGH priority - boost score significantly
                score += 18
            if feats["iso"]:
                score += 15
            if feats["low_priority"]:
                score += 8
            score += min(20, int(float(slot.get("order_count_over_cap", 0) or 0)))
            score += min(15, int(float(slot.get("pair_gap_proxy", 0) or 0) // 10))
            score += min(10, int(float(slot.get("span_risk", 0) or 0) // 10))
            if break_ton:
                secondary.append(f"TON_WINDOW:{ton_window_breakers[soid].get('reason', 'UNKNOWN')}")
            add_candidate(
                soid,
                score,
                dominant,
                ",".join(secondary),
                slot=slot,
                risk_summary=slot_risk_hint,
                would_break_slot_if_kept=True,
                would_break_ton_window_if_kept=break_ton,
                globally_isolated=bool(feats["global_iso"]),
                # Priority 2: pass adjacency_risk to add_candidate
                adjacency_risk=bool(feats.get("adjacency_risk", False)),
            )

    # Add explicit ton window breakers first (highest priority)
    for oid, info in ton_window_breakers.items():
        if oid not in scored:
            add_candidate(
                oid,
                145,  # Highest score for ton window breakers
                "TON_WINDOW_INFEASIBLE",
                f"TON_WINDOW:{info.get('reason', 'UNKNOWN')}",
                risk_summary=f"ton_window:{info.get('reason', 'UNKNOWN')}",
                would_break_slot_if_kept=True,
                would_break_ton_window_if_kept=True,
                globally_isolated=oid in isolated_lookup,
            )

    for oid, row in by_id.items():
        allowed = _allowed_lines(row)
        soid = str(oid)
        break_ton = soid in ton_window_breakers
        if not allowed:
            add_candidate(soid, 120, "NO_FEASIBLE_LINE", risk_summary="no_feasible_line", would_break_slot_if_kept=False, would_break_ton_window_if_kept=break_ton, no_feasible_line=True)

    if not scored:
        return pd.DataFrame()

    # FIXED priority order (MUST match drop_priority_order diagnostics):
    # FIXED priority order using unified helper (see DROP_PRIORITY_ORDER)
    # score is still used as a tiebreaker but not as primary sort key
    picked = sorted(
        scored.values(),
        key=lambda item: (
            _drop_priority_key(item),
            -int(item["score"]),  # Higher score first (secondary)
        ),
    )[:budget]
    rows = []
    for item in picked:
        rec = dict(item["order"])
        # Use unified helper for dominant reason
        dominant = _dominant_drop_reason(item)
        rec["drop_reason"] = dominant
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
        rec["would_break_ton_window_if_kept"] = bool(item.get("would_break_ton_window_if_kept", False))
        rec["adjacency_risk"] = bool(item.get("adjacency_risk", False))  # Priority 2
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
    # Internal classification for candidate ranking (NOT final acceptance)
    internal_classification = "BEST_SEARCH_CANDIDATE_ANALYSIS"
    if routing_feasible:
        internal_classification = "HAS_DROPPED_ORDERS" if unassigned > 0 else "FULL_SCHEDULE_ATTEMPT"
    return {
        "routing_feasible": routing_feasible,
        "internal_classification": internal_classification,  # NOT final acceptance status
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
    # Intermediate candidate fields (not final acceptance)
    # FINAL acceptance is ONLY set by cold_rolling_pipeline after validation
    candidate_has_drop: bool = False,
    candidate_drop_count: int = 0,
    candidate_drop_tons: float = 0.0,
    candidate_routing_feasible: bool = False,
    candidate_validation_pending: bool = True,  # True until pipeline validates
    candidate_phase: str = "",  # "feasibility" or "optimize"
    candidate_reason: str = "",  # Why this candidate was selected
    # Two-phase solving diagnostics
    feasibility_phase_executed: bool = False,
    feasibility_phase_found_solution: bool = False,
    optimize_phase_executed: bool = False,
    optimize_phase_improved_solution: bool = False,
    official_solution_source: str = "NONE",  # FEASIBILITY_PHASE, OPTIMIZE_PHASE, NONE
    # Drop statistics by reason
    drop_due_to_ton_window_count: int = 0,
    drop_due_to_global_isolation_count: int = 0,
    drop_due_to_no_feasible_line_count: int = 0,
    drop_due_to_bridge_not_supported_count: int = 0,
    drop_due_to_routing_risk_count: int = 0,
    drop_due_to_capacity_count: int = 0,
    drop_due_to_low_priority_count: int = 0,
    drop_due_to_master_unassigned_count: int = 0,
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
        # Intermediate candidate fields - FINAL acceptance only set by pipeline after validation
        "candidate_has_drop": bool(candidate_has_drop),
        "candidate_drop_count": int(candidate_drop_count),
        "candidate_drop_tons": float(candidate_drop_tons),
        "candidate_routing_feasible": bool(candidate_routing_feasible),
        "candidate_validation_pending": bool(candidate_validation_pending),
        "candidate_phase": str(candidate_phase),
        "candidate_reason": str(candidate_reason),
        # Two-phase solving diagnostics
        "feasibility_phase_executed": bool(feasibility_phase_executed),
        "feasibility_phase_found_solution": bool(feasibility_phase_found_solution),
        "optimize_phase_executed": bool(optimize_phase_executed),
        "optimize_phase_improved_solution": bool(optimize_phase_improved_solution),
        "official_solution_source": str(official_solution_source),
        # Drop statistics by reason
        "drop_due_to_ton_window_count": int(drop_due_to_ton_window_count),
        "drop_due_to_global_isolation_count": int(drop_due_to_global_isolation_count),
        "drop_due_to_no_feasible_line_count": int(drop_due_to_no_feasible_line_count),
        "drop_due_to_bridge_not_supported_count": int(drop_due_to_bridge_not_supported_count),
        "drop_due_to_routing_risk_count": int(drop_due_to_routing_risk_count),
        "drop_due_to_capacity_count": int(drop_due_to_capacity_count),
        "drop_due_to_low_priority_count": int(drop_due_to_low_priority_count),
        "drop_due_to_master_unassigned_count": int(drop_due_to_master_unassigned_count),
        # Fixed drop priority order (do not change)
        "drop_priority_order": "GLOBAL_ISO > TON_WINDOW > NO_FEASIBLE_LINE > BRIDGE_NOT_SUPPORTED > ISO > LOW_PRIORITY > OUTLIER > OTHER",
    }


def solve_master_model(
    req: ColdRollingRequest,
    transition_pack: dict | None = None,
    orders_df: pd.DataFrame | None = None,
    phase_mode: str = "feasibility_only",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Production master solve entry; returns structured solver output plus engine metadata only.

    phase_mode:
        "feasibility_only" - only run phase="feasibility"; no optimize allowed (default)
        "optimize_only"    - only run phase="optimize"; only valid after pipeline validates feasibility
    """
    if orders_df is not None:
        current_cfg = req.config
        current_transition_pack = transition_pack
        template_build_seconds = 0.0
        joint_master_seconds = 0.0
        local_router_seconds = 0.0
        fallback_total_seconds = 0.0
        solver_strategy = str(getattr(current_cfg.model, "main_solver_strategy", "joint_master") or "joint_master")
        profile_name = str(getattr(current_cfg.model, "profile_name", "") or "")

        # ---- Profile guard: hard enforcement ----
        # 当前生产主路径允许以下 profile：
        #   - constructive_lns_search: 主线 (Route RB / real bridge mainline)
        #   - constructive_lns_real_bridge_frontload: 主线别名
        #   - constructive_lns_direct_only_baseline: 回归基线 (Route C / direct_only)
        #   - constructive_lns_virtual_guarded_frontload: 受控 virtual family 实验线
        #   - block_first_guarded_search: block-first 主路径 (block_generator -> block_master -> block_realizer -> block_alns)
        # 其他 alias / 实验线 / debug profile 已从主路径移除。
        allowed_engineering_profiles = {
            "constructive_lns_search",
            "constructive_lns_direct_only_baseline",
            "constructive_lns_real_bridge_frontload",
            "constructive_lns_virtual_guarded_frontload",
            "block_first_guarded_search",
        }
        if profile_name not in allowed_engineering_profiles:
            raise RuntimeError(
                f"[APS][PROFILE_GUARD] illegal profile: {profile_name}; "
                f"Only {sorted(allowed_engineering_profiles)} are allowed in current engineering mode"
            )
        if solver_strategy not in ("constructive_lns", "block_first"):
            raise RuntimeError(
                f"[APS][PROFILE_GUARD] illegal main_solver_strategy: {solver_strategy}; "
                "Only constructive_lns and block_first are allowed in current engineering mode"
            )

        # =====================================================================
        # Branch: block_first vs. constructive_lns
        # =====================================================================
        if solver_strategy == "block_first":
            # =================================================================
            # BLOCK-FIRST MASTER PATH (New Skeleton - Production Path)
            # =================================================================
            print(f"[APS][master] requested_profile={profile_name}")
            print(f"[APS][master] requested_main_solver_strategy={solver_strategy}")
            print(f"[APS][block_first] Entering block-first master path")
            print(f"[APS][block_first] main_path=block_first")
            print(f"[APS][block_first] profile_name={profile_name}")

            # Lazy imports for new skeleton (avoid circular imports)
            try:
                from aps_cp_sat.model.block_generator import generate_candidate_blocks
                from aps_cp_sat.model.block_master import solve_block_master
                from aps_cp_sat.model.block_realizer import realize_selected_blocks
                from aps_cp_sat.model.block_alns import run_block_alns
            except ImportError as e:
                raise ImportError(f"[APS][block_first] Failed to import block_first modules: {e}")

            # Log block generator config
            print(
                f"[APS][block_first] block_generator_max_blocks_total="
                f"{int(getattr(current_cfg.model, 'block_generator_max_blocks_total', 80))}"
            )
            print(
                f"[APS][block_first] block_alns_rounds="
                f"{int(getattr(current_cfg.model, 'block_alns_rounds', 10))}"
            )

            t0 = perf_counter()

            # Step 1: Candidate Block Generation
            gen_t0 = perf_counter()
            print(f"[APS][block_first] Generating candidate blocks...")
            block_pool = generate_candidate_blocks(
                orders_df=orders_df,
                transition_pack=current_transition_pack,
                cfg=current_cfg,
                constructive_result=None,
                cut_result=None,
                dropped_orders=None,
                random_seed=int(getattr(current_cfg.model, "random_seed", 42)),
            )
            gen_seconds = perf_counter() - gen_t0
            print(
                f"[APS][block_first] Generated {len(block_pool.blocks)} candidate blocks "
                f"in {gen_seconds:.2f}s"
            )

            # Step 2: Block Master Selection
            master_t0 = perf_counter()
            print(f"[APS][block_first] Running block master selection...")
            master_result = solve_block_master(
                pool=block_pool,
                orders_df=orders_df,
                cfg=current_cfg,
                random_seed=int(getattr(current_cfg.model, "random_seed", 42)),
            )
            master_seconds = perf_counter() - master_t0
            print(
                f"[APS][block_first] Selected {len(master_result.selected_blocks)} blocks "
                f"covering {len(master_result.selected_order_ids)} orders "
                f"in {master_seconds:.2f}s"
            )

            # Step 3: Block Realization
            realizer_t0 = perf_counter()
            print(f"[APS][block_first] Realizing selected blocks...")
            realization_result = realize_selected_blocks(
                master_result=master_result,
                orders_df=orders_df,
                transition_pack=current_transition_pack,
                cfg=current_cfg,
                random_seed=int(getattr(current_cfg.model, "random_seed", 42)),
            )
            realizer_seconds = perf_counter() - realizer_t0
            print(
                f"[APS][block_first] Realized {len(realization_result.realized_blocks)} blocks "
                f"in {realizer_seconds:.2f}s"
            )

            # Step 4: Block ALNS (lightweight)
            alns_t0 = perf_counter()
            alns_result = run_block_alns(
                initial_pool=block_pool,
                initial_master_result=master_result,
                initial_realization_result=realization_result,
                orders_df=orders_df,
                transition_pack=current_transition_pack,
                cfg=current_cfg,
                random_seed=int(getattr(current_cfg.model, "random_seed", 42)),
            )
            alns_seconds = perf_counter() - alns_t0
            print(
                f"[APS][block_first] Block ALNS: {alns_result.iterations_attempted} rounds attempted, "
                f"{alns_result.iterations_accepted} accepted, {alns_seconds:.2f}s"
            )

            total_seconds = perf_counter() - t0

            # Build schedule_df from realization
            schedule_df = pd.DataFrame()
            if realization_result.realized_schedule_df is not None and not realization_result.realized_schedule_df.empty:
                schedule_df = realization_result.realized_schedule_df.copy()
                schedule_df = schedule_df.rename(columns={
                    "order_id": "order_id",
                    "sequence_in_block": "master_seq",
                    "block_id": "block_id",
                    "block_line": "assigned_line",
                    "edge_type": "selected_edge_type",
                    "tons": "tons",
                })
                # Add required columns
                schedule_df["is_unassigned"] = 0
                schedule_df["assigned_slot"] = 1  # Block realization uses slot=1 per block
                schedule_df["force_break_before"] = 0
                schedule_df["selected_template_id"] = ""
                schedule_df["selected_bridge_path"] = ""

            # Build dropped_df
            dropped_df = pd.DataFrame()
            if master_result.dropped_order_ids:
                all_order_ids = set(str(oid) for oid in orders_df["order_id"])
                dropped_order_ids_set = set(master_result.dropped_order_ids)
                dropped_records = [
                    dict(row) for _, row in orders_df.iterrows()
                    if str(row["order_id"]) in dropped_order_ids_set
                ]
                for rec in dropped_records:
                    rec["drop_reason"] = "NOT_SELECTED_IN_BLOCKS"
                    rec["dominant_drop_reason"] = "BLOCK_COMPETITION_LOSS"
                    rec["secondary_reasons"] = "BLOCK_FIRST_MASTER"
                dropped_df = pd.DataFrame(dropped_records)

            # Build engine_meta
            diag = master_result.diagnostics
            block_real_diag = realization_result.block_realization_diag
            block_alns_diag = alns_result.neighborhood_diag

            engine_meta = {
                "solver_path": "block_first",
                "main_path": "block_first",
                "profile_name": profile_name,
                "engine_used": "block_first_new_skeleton",
                "assigned_count": len(master_result.selected_order_ids),
                "unassigned_count": len(master_result.dropped_order_ids),
                # Block generation metrics
                "generated_blocks_total": block_pool.diagnostics.get("generated_blocks_total", len(block_pool.blocks)),
                "generated_blocks_by_line": block_pool.diagnostics.get("generated_blocks_by_line", {}),
                "generated_blocks_by_mode": block_pool.diagnostics.get("generated_blocks_by_mode", {}),
                # Block master metrics
                "selected_blocks_count": len(master_result.selected_blocks),
                "selected_order_coverage": diag.get("selected_order_coverage", 0),
                "block_master_dropped_count": diag.get("block_master_dropped_count", 0),
                "avg_block_quality_in_selected": diag.get("avg_block_quality_in_selected", 0.0),
                "avg_block_tons_in_selected": diag.get("avg_block_tons_in_selected", 0.0),
                # Block realization metrics
                "block_realized_count": block_real_diag.get("block_realized_count", 0),
                "orders_in_realized_blocks": block_real_diag.get("orders_in_realized_blocks", 0),
                "mixed_bridge_attempt_count": block_real_diag.get("mixed_bridge_attempt_count", 0),
                "mixed_bridge_success_count": block_real_diag.get("mixed_bridge_success_count", 0),
                # Block ALNS metrics
                "block_alns_rounds_attempted": block_alns_diag.get("block_alns_rounds_attempted", 0),
                "block_alns_rounds_accepted": block_alns_diag.get("block_alns_rounds_accepted", 0),
                # Timing
                "block_generation_seconds": gen_seconds,
                "block_master_seconds": master_seconds,
                "block_realization_seconds": realizer_seconds,
                "block_alns_seconds": alns_seconds,
                "total_block_first_seconds": total_seconds,
                # Architecture markers
                "used_local_routing": False,
                "local_routing_role": "block_first_new_skeleton",
                "strict_template_edges_enabled": True,
                "master_architecture": "block_first_new_skeleton",
                "enforced_profile": profile_name,
                "enforced_main_solver_strategy": "block_first",
                # Legacy compatibility
                "plan_df": schedule_df,
                "dropped_df": dropped_df,
            }

            print(
                f"[APS][block_first] Complete: {engine_meta['assigned_count']} orders assigned, "
                f"{engine_meta['unassigned_count']} dropped, "
                f"total_time={total_seconds:.2f}s"
            )

            return schedule_df, pd.DataFrame(), dropped_df, engine_meta

        # =====================================================================
        # CONSTRUCTIVE LNS MASTER PATH (Legacy Main Path)
        # =====================================================================
        print(f"[APS][master] requested_profile={profile_name}")
        print(f"[APS][master] requested_main_solver_strategy={solver_strategy}")
        print(f"[APS][master] joint_master_branch_enabled=false")
        print(f"[APS][constructive_lns] ENTER_NEW_MAIN_PATH=true")
        print(f"[APS][constructive_lns] Entering constructive LNS master path")
        # Route C: explicit edge policy logging to prevent future misdiagnosis
        bridge_expansion_mode = str(getattr(current_cfg.model, "bridge_expansion_mode", "disabled"))
        allow_virtual_bridge = bool(getattr(current_cfg.model, "allow_virtual_bridge_edge_in_constructive", False))
        allow_real_bridge = bool(getattr(current_cfg.model, "allow_real_bridge_edge_in_constructive", False))
        constructive_edge_policy = "direct_only" if (not allow_virtual_bridge and not allow_real_bridge) else ("direct_plus_real_bridge" if not allow_virtual_bridge else "all_edges_allowed")
        print(f"[APS][constructive_lns] bridge_expansion_mode={bridge_expansion_mode}")
        print(f"[APS][constructive_lns] allow_virtual_bridge_edge_in_constructive={allow_virtual_bridge}")
        print(f"[APS][constructive_lns] allow_real_bridge_edge_in_constructive={allow_real_bridge}")
        print(f"[APS][constructive_lns] constructive_edge_policy={constructive_edge_policy}")
        if constructive_edge_policy == "direct_only":
            print(f"[APS][constructive_lns] Route C (baseline): only DIRECT_EDGE is allowed")
        elif constructive_edge_policy == "direct_plus_real_bridge":
            print(f"[APS][constructive_lns] Route RB (mainline): DIRECT_EDGE + REAL_BRIDGE_EDGE allowed, VIRTUAL_BRIDGE_EDGE blocked, bridge_expansion_mode={bridge_expansion_mode}")
        else:
            print(f"[APS][constructive_lns] edge_policy={constructive_edge_policy}, bridge_expansion_mode={bridge_expansion_mode}")
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

        # =====================================================================
        # Main solver: constructive_lns only (joint_master blocked by guard)
        # =====================================================================
        lns_t0 = perf_counter()
        lns_result = run_constructive_lns_master(
                orders_df=orders_df,
                transition_pack=current_transition_pack,
                cfg=current_cfg,
                random_seed=int(current_cfg.model.rounds or 42),
            )
        lns_elapsed = perf_counter() - lns_t0
        # Build engine_meta in joint_master style for compatibility
        lns_engine_meta = _build_meta(
            req,
            engine_used="constructive_lns",
            main_path="constructive_lns",
            fallback_used=False,
            fallback_type="none",
            fallback_reason="",
            fallback_trace=[],
            used_local_routing=False,
            local_routing_role="constructive_lns_local_inserter",
            input_order_count=len(orders_df),
            failure_diagnostics={},
            joint_estimates={},
            strict_template_edges_enabled=bool(current_cfg.model.strict_template_edges),
            unroutable_slot_count=0,
            slot_route_details=[],
            template_graph_health=str(health.get("template_graph_health", "UNKNOWN")),
            precheck_autorelax_applied=precheck_autorelax_applied,
            solve_attempt_count=1,
            fallback_attempt_count=0,
            early_stop_reason="",
            early_stop_deferred_for_semantic_fallback=False,
            semantic_fallback_first_attempt_status="",
            assignment_pressure_mode="normal",
            effective_config=current_cfg,
            template_build_seconds=template_build_seconds,
            joint_master_seconds=lns_elapsed,
            local_router_seconds=0.0,
            fallback_total_seconds=0.0,
            feasibility_evidence=feasibility_evidence,
            failure_mode="",
            evidence_level=evidence_level,
            top_infeasibility_signals=top_infeasibility_signals,
            drop_strategy_applied=False,
            dropped_candidates_considered=0,
            dropped_candidates_selected=0,
            dominant_drop_reason_histogram={},
            structure_fallback_applied=False,
            fallback_mode="",
            slot_count_after_fallback=0,
            drop_budget_after_fallback=0,
            best_candidate_available=False,
            best_candidate_type="",
            best_candidate_objective=0.0,
            best_candidate_search_status=str(lns_result.status.value),
            best_candidate_routing_feasible=(lns_result.status in {
                "OPTIMAL", "FEASIBLE", "IMPROVED"}),
            best_candidate_unroutable_slot_count=0,
            drop_stage="",
            drop_budget_used=0,
            drop_budget_remaining=0,
            structure_first_applied=False,
            time_expansion_applied=False,
            slot_count_before_fallback=int(current_cfg.model.max_campaign_slots),
            restructured_slot_count=0,
            bad_slots_before_restructure=[],
            bad_slots_after_restructure=[],
            orders_removed_from_bad_slots=[],
            drop_due_to_ton_window_count=0,
            drop_due_to_global_isolation_count=0,
            drop_due_to_no_feasible_line_count=0,
            drop_due_to_bridge_not_supported_count=0,
            drop_due_to_routing_risk_count=0,
            drop_due_to_capacity_count=0,
            drop_due_to_low_priority_count=0,
            drop_due_to_master_unassigned_count=0,
            feasibility_phase_executed=True,
            feasibility_phase_found_solution=(lns_result.status in {
                "OPTIMAL", "FEASIBLE", "IMPROVED", "INITIAL_FEASIBLE"}),
            optimize_phase_executed=False,
            optimize_phase_improved_solution=False,
            official_solution_source="CONSTRUCTIVE_LNS",
            candidate_has_drop=(not lns_result.dropped_df.empty) if lns_result.dropped_df is not None else False,
            candidate_drop_count=int(lns_result.dropped_df.shape[0]) if lns_result.dropped_df is not None and not lns_result.dropped_df.empty else 0,
            candidate_drop_tons=float(lns_result.dropped_df["tons"].sum()) if lns_result.dropped_df is not None and not lns_result.dropped_df.empty and "tons" in lns_result.dropped_df.columns else 0.0,
            candidate_routing_feasible=(lns_result.status in {
                "OPTIMAL", "FEASIBLE", "IMPROVED", "INITIAL_FEASIBLE"}),
            candidate_validation_pending=True,
            candidate_phase="feasibility",
            candidate_reason="constructive_lns_master",
        )
        # Merge LNS engine_meta into result
        lns_engine_meta.update({
            "lns_diagnostics": lns_result.diagnostics,
            "lns_engine_meta": lns_result.engine_meta,
            "lns_status": lns_result.status.value,
            "candidate_graph_diagnostics": (
                current_transition_pack.get("candidate_graph_diagnostics", {})
                if isinstance(current_transition_pack, dict) else {}
            ),
            # Enforced path enforcement metadata
            "enforced_profile": profile_name,
            "enforced_main_solver_strategy": "constructive_lns",
            "joint_master_branch_enabled": False,
            "old_master_blocked": True,
            # Run path fingerprints
            "run_path_fingerprint_pipeline": "PIPELINE_V2_20260416A",
            "run_path_fingerprint_constructive_builder": "CONSTRUCTIVE_SEQUENCE_BUILDER_V2_20260416A",
            "run_path_fingerprint_campaign_cutter": str(
                lns_result.diagnostics.get("campaign_cut_diags", {}).get("run_path_fingerprint_campaign_cutter", "")
                if isinstance(lns_result.diagnostics, dict) else ""
            ),
            "run_path_fingerprint_constructive_lns_master": str(
                lns_result.engine_meta.get("run_path_fingerprint_constructive_lns_master", "")
                if isinstance(lns_result.engine_meta, dict) else ""
            ),
        })
        # Ensure rounds_df is present (may be empty)
        _rounds_df = lns_result.rounds_df if lns_result.rounds_df is not None else pd.DataFrame()
        return (
            lns_result.planned_df if lns_result.planned_df is not None else pd.DataFrame(),
            _rounds_df,
            lns_result.dropped_df if lns_result.dropped_df is not None else pd.DataFrame(),
            lns_engine_meta,
        )

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
        # Two-phase solving tracking
        feasibility_phase_executed = False
        feasibility_phase_found_solution = False
        optimize_phase_executed = False
        optimize_phase_improved_solution = False
        official_solution_source = "NONE"
        # Candidate intermediate fields
        candidate_has_drop = False
        candidate_drop_count = 0
        candidate_drop_tons = 0.0
        candidate_routing_feasible = False
        candidate_validation_pending = True
        candidate_phase = ""
        candidate_reason = ""
        # Drop reason statistics
        drop_due_to_ton_window_count = 0
        drop_due_to_global_isolation_count = 0
        drop_due_to_no_feasible_line_count = 0
        drop_due_to_bridge_not_supported_count = 0
        drop_due_to_routing_risk_count = 0
        drop_due_to_capacity_count = 0
        drop_due_to_low_priority_count = 0
        drop_due_to_master_unassigned_count = 0

        for aidx, ap in enumerate(attempts, start=1):
            for seed in seeds:
                solve_attempt_count += 1
                solve_t0 = perf_counter()
                joint = _run_global_joint_model(
                    orders_df,
                    current_transition_pack,
                    current_cfg,
                    start_penalty=int(ap["start_penalty"]),
                    time_scale=float(ap["time_scale"]),
                    random_seed=int(seed),
                    phase="feasibility",
                )
                solve_elapsed = perf_counter() - solve_t0
                joint_master_seconds += solve_elapsed
                local_router_seconds += float(joint.get("local_router_seconds", 0.0) or 0.0)
                # Track feasibility phase
                feasibility_phase_executed = True
                # feasibility_phase_found_solution is set by pipeline AFTER validation, not here
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
                    if cand["internal_classification"] == "FULL_SCHEDULE_ATTEMPT":
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
                    best_candidate_type="",  # Set by pipeline after validation
                    best_candidate_objective=float(best_joint.get("objective", 0.0) or 0.0),
                    best_candidate_search_status=str(best_joint.get("status", "")),
                    best_candidate_routing_feasible=True,
                    best_candidate_unroutable_slot_count=int(best_joint.get("unroutable_slot_count", 0) or 0),
                    # Intermediate fields - FINAL acceptance by pipeline
                    candidate_has_drop=bool(not dropped_df.empty),
                    candidate_drop_count=int(len(dropped_df)),
                    candidate_drop_tons=float(dropped_df["tons"].sum()) if "tons" in dropped_df.columns and not dropped_df.empty else 0.0,
                    candidate_routing_feasible=True,
                    candidate_validation_pending=True,
                    candidate_phase="feasibility",
                    candidate_reason="MAIN_LOOP_FEASIBILITY",
                    # Two-phase tracking
                    feasibility_phase_executed=feasibility_phase_executed,
                    feasibility_phase_found_solution=feasibility_phase_found_solution,
                    official_solution_source="FEASIBILITY_PHASE",  # Pipeline sets final source after validation
                    # Drop reason statistics
                    drop_due_to_ton_window_count=int(len(dropped_df[dropped_df["dominant_drop_reason"].str.contains("TON_WINDOW", na=False)])) if not dropped_df.empty and "dominant_drop_reason" in dropped_df.columns else 0,
                    drop_due_to_global_isolation_count=int(len(dropped_df[dropped_df["dominant_drop_reason"].str.contains("GLOBAL_ISOLATED", na=False)])) if not dropped_df.empty and "dominant_drop_reason" in dropped_df.columns else 0,
                    drop_due_to_no_feasible_line_count=int(len(dropped_df[dropped_df["dominant_drop_reason"].str.contains("NO_FEASIBLE_LINE", na=False)])) if not dropped_df.empty and "dominant_drop_reason" in dropped_df.columns else 0,
                    drop_due_to_routing_risk_count=int(len(dropped_df[dropped_df["dominant_drop_reason"].str.contains("ROUTING_RISK|LOCAL_ROUTER", na=False)])) if not dropped_df.empty and "dominant_drop_reason" in dropped_df.columns else 0,
                    drop_due_to_master_unassigned_count=int(len(dropped_df[dropped_df["dominant_drop_reason"].str.contains("MASTER_UNASSIGNED", na=False)])) if not dropped_df.empty and "dominant_drop_reason" in dropped_df.columns else 0,
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
                # Priority 3: Track adjacency-related drops for shift from routing risk to strict validation repair
                adjacency_drop_count = int(
                    candidate_df["drop_reason"].fillna("").str.contains("ADJACENCY_VIOLATION_RISK", na=False).sum()
                ) if "drop_reason" in candidate_df.columns else 0
                ton_window_drop_count = int(
                    candidate_df["drop_reason"].fillna("").str.contains("TON_WINDOW", na=False).sum()
                ) if "drop_reason" in candidate_df.columns else 0
                global_iso_drop_count = int(
                    candidate_df["drop_reason"].fillna("").str.contains("GLOBAL_ISOLATED", na=False).sum()
                ) if "drop_reason" in candidate_df.columns else 0
                no_feasible_line_drop_count = int(
                    candidate_df["drop_reason"].fillna("").str.contains("NO_FEASIBLE_LINE", na=False).sum()
                ) if "drop_reason" in candidate_df.columns else 0
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
                        # Priority 3: Drop reason breakdown for strict validation repair shift
                        "adjacency_drop_count": int(adjacency_drop_count),
                        "ton_window_drop_count": int(ton_window_drop_count),
                        "global_iso_drop_count": int(global_iso_drop_count),
                        "no_feasible_line_drop_count": int(no_feasible_line_drop_count),
                        "drop_reason_breakdown": {
                            "ADJACENCY_VIOLATION_RISK": adjacency_drop_count,
                            "TON_WINDOW_INFEASIBLE": ton_window_drop_count,
                            "GLOBAL_ISOLATED_ORDER": global_iso_drop_count,
                            "NO_FEASIBLE_LINE": no_feasible_line_drop_count,
                        },
                    }
                )
                print(
                    f"[APS][structure_fallback] applied=true, drop_count={len(candidate_df)}, "
                    f"slot_cap={structure_cfg.model.max_campaign_slots}, "
                    f"restructured_slots={len(bad_slots_before_restructure)}, "
                    f"adjacency_drops={adjacency_drop_count}, "
                    f"ton_window_drops={ton_window_drop_count}, "
                    f"global_iso_drops={global_iso_drop_count}, "
                    f"reasons={dominant_drop_reason_histogram}"
                )
                fallback_t0 = perf_counter()
                structure_joint = _run_global_joint_model(
                    structure_keep_df,
                    current_transition_pack,
                    structure_cfg,
                    start_penalty=int(attempts[0]["start_penalty"]) if attempts else 120000,
                    time_scale=max(0.8, float(attempts[0]["time_scale"])) if attempts else 0.8,
                    random_seed=int((seeds[0] if seeds else 2027) + 501),
                    phase="feasibility",
                )
                elapsed = perf_counter() - fallback_t0
                fallback_total_seconds += elapsed
                joint_master_seconds += elapsed
                local_router_seconds += float(structure_joint.get("local_router_seconds", 0.0) or 0.0)
                # Track feasibility phase from structure fallback
                feasibility_phase_executed = True
                # feasibility_phase_found_solution is set by pipeline AFTER validation, not here
                cand = _candidate_summary(
                    structure_joint,
                    extra_drop_count=len(structure_drop_df),
                    source_orders_df=structure_keep_df,
                    extra_dropped_df=structure_drop_df,
                )
                if _candidate_better(best_any_candidate, cand, mode="search"):
                    best_any_candidate = cand
                if cand is not None and cand["routing_feasible"]:
                    if cand["internal_classification"] == "FULL_SCHEDULE_ATTEMPT":
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
                            # Intermediate fields - FINAL acceptance by pipeline
                            candidate_has_drop=bool(not dropped_df.empty),
                            candidate_drop_count=int(len(dropped_df)),
                            candidate_drop_tons=float(dropped_df["tons"].sum()) if "tons" in dropped_df.columns and not dropped_df.empty else 0.0,
                            candidate_routing_feasible=True,
                            candidate_validation_pending=True,
                            candidate_phase="feasibility",
                            candidate_reason="STRUCTURE_FALLBACK_FEASIBILITY",
                            # Two-phase tracking
                            feasibility_phase_executed=feasibility_phase_executed,
                            feasibility_phase_found_solution=feasibility_phase_found_solution,
                            official_solution_source="FEASIBILITY_PHASE",  # Pipeline sets final source after validation
                            # Drop reason statistics
                            drop_due_to_ton_window_count=drop_due_to_ton_window_count,
                            drop_due_to_global_isolation_count=drop_due_to_global_isolation_count,
                            drop_due_to_no_feasible_line_count=drop_due_to_no_feasible_line_count,
                            drop_due_to_routing_risk_count=drop_due_to_routing_risk_count,
                            drop_due_to_master_unassigned_count=drop_due_to_master_unassigned_count,
                            best_candidate_available=True,
                            best_candidate_type="",  # Set by pipeline after validation
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
                retry = attempts[0] if attempts else {"start_penalty": 90000, "time_scale": 0.45}
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
                    # Two-phase solving: phase is controlled by pipeline gate, not internal heuristic
                    # master.py always runs feasibility unless pipeline explicitly sets phase_mode="optimize_only"
                    actual_phase = "optimize" if phase_mode == "optimize_only" else "feasibility"
                    print(f"[APS][two_phase] semantic_fallback #{fidx}: phase_mode={phase_mode}, actual_phase={actual_phase}")
                    fallback_t0 = perf_counter()
                    joint = _run_global_joint_model(
                        working_orders_df,
                        current_transition_pack,
                        fcfg,
                        start_penalty=int(retry["start_penalty"]),
                        time_scale=max(1.1, float(retry["time_scale"]) * (1.4 + 0.2 * fidx)),
                        random_seed=int(retry_seed + fidx),
                        phase=actual_phase,
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
                        if cand["internal_classification"] == "FULL_SCHEDULE_ATTEMPT":
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
                            # Intermediate fields - FINAL acceptance by pipeline
                            candidate_has_drop=bool(not dropped_df.empty),
                            candidate_drop_count=int(len(dropped_df)),
                            candidate_drop_tons=float(dropped_df["tons"].sum()) if "tons" in dropped_df.columns and not dropped_df.empty else 0.0,
                            candidate_routing_feasible=True,
                            candidate_validation_pending=True,
                            candidate_phase=use_phase,
                            candidate_reason=f"SEMANTIC_FALLBACK_{actual_phase.upper()}",
                            # Two-phase tracking
                            feasibility_phase_executed=True,
                            feasibility_phase_found_solution=False,  # Set by pipeline after validation
                            optimize_phase_executed=(phase_mode == "optimize_only"),
                            official_solution_source="FEASIBILITY_PHASE" if phase_mode == "feasibility_only" else "OPTIMIZE_PHASE",
                            # Drop reason statistics
                            drop_due_to_ton_window_count=drop_due_to_ton_window_count,
                            drop_due_to_global_isolation_count=drop_due_to_global_isolation_count,
                            drop_due_to_no_feasible_line_count=drop_due_to_no_feasible_line_count,
                            drop_due_to_routing_risk_count=drop_due_to_routing_risk_count,
                            drop_due_to_master_unassigned_count=drop_due_to_master_unassigned_count,
                            best_candidate_available=True,
                            best_candidate_type="",  # Set by pipeline after validation
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
                        # Two-phase: phase controlled by pipeline gate, not internal heuristic
                        actual_phase = "optimize" if phase_mode == "optimize_only" else "feasibility"
                        print(f"[APS][two_phase] scale-down: phase_mode={phase_mode}, actual_phase={actual_phase}")
                        scale_joint = _run_global_joint_model(
                            keep_df,
                            current_transition_pack,
                            scale_cfg,
                            start_penalty=int(retry["start_penalty"]),
                            time_scale=max(1.1, float(retry["time_scale"]) * 2.0),
                            random_seed=int(retry_seed + 100 + n_keep),
                            phase=actual_phase,
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
                            if cand["internal_classification"] == "FULL_SCHEDULE_ATTEMPT":
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
                            # Intermediate fields - FINAL acceptance by pipeline
                            candidate_has_drop=bool(not dropped_df.empty),
                            candidate_drop_count=int(len(dropped_df)),
                            candidate_drop_tons=float(dropped_df["tons"].sum()) if "tons" in dropped_df.columns and not dropped_df.empty else 0.0,
                            candidate_routing_feasible=True,
                            candidate_validation_pending=True,
                            candidate_phase=actual_phase,
                            candidate_reason=f"SCALE_DOWN_{actual_phase.upper()}",
                            # Two-phase tracking
                            feasibility_phase_executed=True,
                            feasibility_phase_found_solution=False,  # Set by pipeline after validation
                            optimize_phase_executed=(phase_mode == "optimize_only"),
                            official_solution_source="FEASIBILITY_PHASE" if phase_mode == "feasibility_only" else "OPTIMIZE_PHASE",
                            # Drop reason statistics
                            drop_due_to_ton_window_count=drop_due_to_ton_window_count,
                            drop_due_to_global_isolation_count=drop_due_to_global_isolation_count,
                            drop_due_to_no_feasible_line_count=drop_due_to_no_feasible_line_count,
                            drop_due_to_routing_risk_count=drop_due_to_routing_risk_count,
                            drop_due_to_capacity_count=int(len(drop_df)) if "drop_reason" in drop_df.columns else 0,
                            drop_due_to_master_unassigned_count=drop_due_to_master_unassigned_count,
                            best_candidate_available=True,
                            best_candidate_type="",  # Set by pipeline after validation
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
                # Intermediate fields - FINAL acceptance by pipeline
                candidate_has_drop=bool(best_candidate is not None and best_candidate.get("dropped_order_count", 0) > 0) if best_candidate else False,
                candidate_drop_count=int(best_candidate.get("dropped_order_count", 0)) if best_candidate else 0,
                candidate_drop_tons=0.0,  # Not calculated for failed runs
                candidate_routing_feasible=bool(best_candidate.get("routing_feasible", False)) if best_candidate else False,
                candidate_validation_pending=True,
                candidate_phase="",
                candidate_reason="NO_FEASIBLE_SOLUTION",
                # Two-phase tracking
                feasibility_phase_executed=feasibility_phase_executed,
                feasibility_phase_found_solution=feasibility_phase_found_solution,
                optimize_phase_executed=optimize_phase_executed,
                official_solution_source="NONE",
                # Drop reason statistics
                drop_due_to_ton_window_count=drop_due_to_ton_window_count,
                drop_due_to_global_isolation_count=drop_due_to_global_isolation_count,
                drop_due_to_no_feasible_line_count=drop_due_to_no_feasible_line_count,
                drop_due_to_routing_risk_count=drop_due_to_routing_risk_count,
                drop_due_to_master_unassigned_count=drop_due_to_master_unassigned_count,
                best_candidate_available=best_candidate is not None,
                best_candidate_type="",  # Set by pipeline after validation
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
