"""
Block Realizer for Block-First Architecture.

After block master selects blocks, block_realizer:
1. Orders orders within each block
2. Realizes direct / real bridge / guarded family edges
3. Generates mixed bridge candidates only inside blocks (not global graph)
4. Produces the realized schedule DataFrame

Key constraint: mixed bridge is LOCAL to each block, not global Candidate Graph.
Only two forms are supported in this phase:
- REAL_BRIDGE_EDGE -> VIRTUAL_BRIDGE_FAMILY_EDGE (REAL_TO_GUARDED)
- VIRTUAL_BRIDGE_FAMILY_EDGE -> REAL_BRIDGE_EDGE (GUARDED_TO_REAL)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import random

import pandas as pd

from aps_cp_sat.config import PlannerConfig
from aps_cp_sat.model.block_types import CandidateBlock, CandidateBlockPool, BlockCampaignSlot
from aps_cp_sat.model.block_master import BlockMasterResult
from aps_cp_sat.rules.shared_checks import width_transition_metrics, thickness_transition_metrics, temperature_overlap_metrics


@dataclass
class BlockRealizationResult:
    """Result of realizing selected blocks into a schedule."""
    realized_blocks: List[CandidateBlock] = field(default_factory=list)
    realized_schedule_df: Optional[pd.DataFrame] = None
    realized_dropped_df: Optional[pd.DataFrame] = None

    # Per-block diagnostics
    block_realization_diag: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "realized_blocks": [b.to_dict() for b in self.realized_blocks],
            "block_realization_diag": dict(self.block_realization_diag),
        }


def _obj_get(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _edge_to_record(edge) -> Dict[str, Any]:
    """Normalize CandidateEdge / dict edge into a plain dict record."""
    if isinstance(edge, dict):
        return dict(edge)

    return {
        "from_order_id": str(_obj_get(edge, "from_order_id", "")),
        "to_order_id": str(_obj_get(edge, "to_order_id", "")),
        "edge_type": str(_obj_get(edge, "edge_type", "")),
        "score": float(_obj_get(edge, "score", 0.0) or 0.0),
        "bridge_path_payload": _obj_get(edge, "bridge_path_payload", None),
        "metadata": _obj_get(edge, "metadata", None),
    }


def _normalize_edges(edges: List[Any]) -> List[Dict[str, Any]]:
    normalized = [_edge_to_record(e) for e in (edges or [])]
    return [
        e for e in normalized
        if str(e.get("from_order_id", "")) and str(e.get("to_order_id", ""))
    ]

def _edge_priority(edge_type: str) -> int:
    if edge_type == "DIRECT_EDGE":
        return 3
    if edge_type == "REAL_BRIDGE_EDGE":
        return 2
    if edge_type == "VIRTUAL_BRIDGE_FAMILY_EDGE":
        return 1
    return 0


def _build_edge_indexes(graph_edges: List[Dict[str, Any]]) -> tuple[Dict[tuple[str, str], Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    """Build exact-pair lookup and successor buckets using a stable edge priority."""
    edges = _normalize_edges(graph_edges)
    edge_lookup: Dict[tuple[str, str], Dict[str, Any]] = {}
    edges_by_from: Dict[str, List[Dict[str, Any]]] = {}

    for e in edges:
        key = (str(e.get("from_order_id", "")), str(e.get("to_order_id", "")))
        if not key[0] or not key[1]:
            continue
        prev = edge_lookup.get(key)
        if prev is None:
            edge_lookup[key] = e
        else:
            prev_rank = (_edge_priority(str(prev.get("edge_type", ""))), float(prev.get("score", 0) or 0))
            new_rank = (_edge_priority(str(e.get("edge_type", ""))), float(e.get("score", 0) or 0))
            if new_rank > prev_rank:
                edge_lookup[key] = e

    for e in edge_lookup.values():
        edges_by_from.setdefault(str(e.get("from_order_id", "")), []).append(e)

    for frm, arr in edges_by_from.items():
        arr.sort(key=lambda x: (_edge_priority(str(x.get("edge_type", ""))), float(x.get("score", 0) or 0)), reverse=True)

    return edge_lookup, edges_by_from


def _line_name(line: str) -> str:
    if str(line) == "big_roll":
        return "大辊线"
    if str(line) == "small_roll":
        return "小辊线"
    return str(line)



def _dedupe_preserve(seq: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in seq:
        s = str(item)
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _group_transition_ok(prev_order: Dict[str, Any], cur_order: Dict[str, Any], edge_type: str) -> bool:
    prev_group = str(prev_order.get("steel_group", "") or "")
    cur_group = str(cur_order.get("steel_group", "") or "")
    if not prev_group or not cur_group or prev_group == cur_group:
        return True
    # In production audit, UNKNOWN_EVIDENCE is treated as a problem. Only explicit bridge edges
    # or explicit flags may justify a cross-group transition.
    if edge_type in {"REAL_BRIDGE_EDGE", "VIRTUAL_BRIDGE_EDGE", "VIRTUAL_BRIDGE_FAMILY_EDGE"}:
        return True
    for key in ("group_transition_ok", "cross_group_transition_ok", "is_pc_transition"):
        if key in cur_order:
            try:
                return bool(cur_order.get(key))
            except Exception:
                pass
    return False


def _pair_hard_feasible(prev_order: Dict[str, Any], cur_order: Dict[str, Any], cfg: PlannerConfig, edge_type: str) -> tuple[bool, List[str]]:
    reasons: List[str] = []
    rule = cfg.rule

    prev_w = float(prev_order.get("width", 0.0) or 0.0)
    cur_w = float(cur_order.get("width", 0.0) or 0.0)
    # Align with final schedule audit: no width increase inside a realized campaign.
    if cur_w > prev_w:
        reasons.append("WIDTH_INCREASE")
    elif (prev_w - cur_w) > float(rule.max_width_drop):
        reasons.append("WIDTH_OVERDROP")

    thk_bad, _ = thickness_transition_metrics(float(prev_order.get("thickness", 0.0) or 0.0), float(cur_order.get("thickness", 0.0) or 0.0))
    if thk_bad:
        reasons.append("THICKNESS")

    overlap, temp_bad, _ = temperature_overlap_metrics(
        float(prev_order.get("temp_min", 0.0) or 0.0),
        float(prev_order.get("temp_max", 0.0) or 0.0),
        float(cur_order.get("temp_min", 0.0) or 0.0),
        float(cur_order.get("temp_max", 0.0) or 0.0),
        10.0,
    )
    if temp_bad:
        reasons.append("TEMP")

    if not _group_transition_ok(prev_order, cur_order, edge_type):
        reasons.append("GROUP")

    return len(reasons) == 0, reasons


def _candidate_out_strength(oid: str, edge_lookup: Dict[tuple[str, str], Dict[str, Any]], orders_by_id: Dict[str, Dict[str, Any]], cfg: PlannerConfig) -> float:
    score = 0.0
    for (frm, to), edge in edge_lookup.items():
        if frm != oid or to not in orders_by_id:
            continue
        ok, _ = _pair_hard_feasible(orders_by_id.get(frm, {}), orders_by_id.get(to, {}), cfg, str(edge.get("edge_type", "")))
        if ok:
            score += 1.0 + max(0.0, float(edge.get("score", 0.0) or 0.0)) * 0.001
    return score

def _order_within_block(
    block: CandidateBlock,
    edge_lookup: Dict[tuple[str, str], Dict[str, Any]],
    edges_by_from: Dict[str, List[Dict[str, Any]]],
    orders_by_id: Dict[str, Dict[str, Any]],
    cfg: PlannerConfig,
    rng: random.Random,
    diag_counters: Dict[str, Any] | None = None,
) -> List[str]:
    """
    Build a single feasible chain inside the block.

    Unlike the previous implementation, this function no longer appends arbitrary
    leftovers. Orders that cannot be connected by a template-backed and hard-feasible
    pair are intentionally left out and later treated as dropped. This is stricter, but
    it avoids pushing illegal pairs into the final realized campaign.
    """
    internal_edges = _normalize_edges(block.internal_edges or [])
    order_pool = _dedupe_preserve(list(block.order_ids))
    if not order_pool:
        return []

    def _bump(key: str, delta: int = 1) -> None:
        if diag_counters is not None:
            diag_counters[key] = int(diag_counters.get(key, 0) or 0) + delta

    internal_by_from: Dict[str, List[Dict[str, Any]]] = {}
    for e in internal_edges:
        internal_by_from.setdefault(str(e.get("from_order_id", "")), []).append(e)

    candidate_starts = []
    if str(block.head_order_id or "") in order_pool:
        candidate_starts.append(str(block.head_order_id))
    ranked_by_strength = sorted(order_pool, key=lambda oid: _candidate_out_strength(oid, edge_lookup, orders_by_id, cfg), reverse=True)
    for oid in ranked_by_strength[:5]:
        if oid not in candidate_starts:
            candidate_starts.append(oid)
    if not candidate_starts:
        candidate_starts = [order_pool[0]]

    best_chain: List[str] = []
    for start_oid in candidate_starts:
        ordered = [start_oid]
        remaining = set(order_pool)
        remaining.discard(start_oid)
        current = start_oid

        while remaining:
            candidates = []
            for e in sorted(internal_by_from.get(current, []), key=lambda x: float(x.get("score", 0) or 0), reverse=True):
                cand = str(e.get("to_order_id", ""))
                if cand not in remaining:
                    continue
                ok, _ = _pair_hard_feasible(orders_by_id.get(current, {}), orders_by_id.get(cand, {}), cfg, str(e.get("edge_type", "")))
                if ok:
                    candidates.append((_edge_priority(str(e.get("edge_type", ""))), float(e.get("score", 0) or 0), cand))
            if not candidates:
                for cand in remaining:
                    edge = edge_lookup.get((current, cand))
                    if edge is None:
                        continue
                    ok, _ = _pair_hard_feasible(orders_by_id.get(current, {}), orders_by_id.get(cand, {}), cfg, str(edge.get("edge_type", "")))
                    if ok:
                        candidates.append((_edge_priority(str(edge.get("edge_type", ""))), float(edge.get("score", 0) or 0), cand))
                if candidates:
                    _bump("block_ordering_template_lookup_hit_count")
                else:
                    _bump("block_ordering_template_lookup_miss_count")

            if not candidates:
                break

            candidates.sort(reverse=True)
            next_oid = candidates[0][2]
            ordered.append(next_oid)
            remaining.discard(next_oid)
            current = next_oid

        if len(ordered) > len(best_chain):
            best_chain = ordered

    omitted = max(0, len(order_pool) - len(best_chain))
    if omitted:
        _bump("block_ordering_random_fallback_count", omitted)
        _bump("block_orders_omitted_due_to_infeasible_chain", omitted)
    return best_chain


def _generate_mixed_bridge_candidates(
    block: CandidateBlock,
    ordered_ids: List[str],
    edge_lookup: Dict[tuple[str, str], Dict[str, Any]],
    edges_by_from: Dict[str, List[Dict[str, Any]]],
    orders_by_id: Dict[str, Dict[str, Any]],
    cfg: PlannerConfig,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    """Generate mixed bridge candidates inside a block using the unified edge indexes."""
    if not getattr(cfg.model, "mixed_bridge_in_block_enabled", True):
        return []

    allowed_forms = getattr(cfg.model, "mixed_bridge_allowed_forms", ["REAL_TO_GUARDED", "GUARDED_TO_REAL"])
    allowed_hotspots = getattr(cfg.model, "mixed_bridge_allowed_hotspots", [])
    max_attempts = int(getattr(cfg.model, "mixed_bridge_max_attempts_per_block", 10))

    has_hotspot = False
    if block.has_underfill_hotspot and "underfill" in allowed_hotspots:
        has_hotspot = True
    if block.has_group_switch and "group_switch" in allowed_hotspots:
        has_hotspot = True
    if block.has_bridge_dependency_hotspot and "bridge_dependency" in allowed_hotspots:
        has_hotspot = True
    if block.has_width_tension and "width_tension" in allowed_hotspots:
        has_hotspot = True
    if not has_hotspot:
        return []

    candidates: List[Dict[str, Any]] = []
    attempts = 0

    for i, from_oid in enumerate(ordered_ids[:-1]):
        if attempts >= max_attempts:
            break
        to_oid = ordered_ids[i + 1]
        from_order = orders_by_id.get(from_oid, {})
        to_order = orders_by_id.get(to_oid, {})
        if not from_order or not to_order:
            continue

        pair_edge = edge_lookup.get((from_oid, to_oid))
        existing_edges = edges_by_from.get(from_oid, [])
        has_real = any(str(e.get("edge_type", "")) == "REAL_BRIDGE_EDGE" for e in existing_edges)
        has_guarded = any(str(e.get("edge_type", "")) == "VIRTUAL_BRIDGE_FAMILY_EDGE" for e in existing_edges)

        from_sg = str(from_order.get("steel_group", ""))
        to_sg = str(to_order.get("steel_group", ""))
        group_switch = (from_sg != to_sg and from_sg and to_sg)

        if has_real and "REAL_TO_GUARDED" in allowed_forms and (group_switch or block.has_underfill_hotspot):
            guarded_edges = [e for e in existing_edges if str(e.get("edge_type", "")) == "VIRTUAL_BRIDGE_FAMILY_EDGE"]
            if guarded_edges:
                g = guarded_edges[0]
                candidates.append({
                    "from_order_id": from_oid,
                    "to_order_id": to_oid,
                    "mixed_form": "REAL_TO_GUARDED",
                    "edge_type": "VIRTUAL_BRIDGE_FAMILY_EDGE",
                    "score": float(g.get("score", 0) or 0) * 0.9,
                    "reason": "group_switch_or_underfill",
                })
                attempts += 1

        if has_guarded and "GUARDED_TO_REAL" in allowed_forms and (group_switch or block.has_width_tension):
            real_edges = [e for e in existing_edges if str(e.get("edge_type", "")) == "REAL_BRIDGE_EDGE"]
            direct_edges = [e for e in existing_edges if str(e.get("edge_type", "")) == "DIRECT_EDGE"]
            if real_edges or direct_edges or pair_edge is not None:
                preferred_type = "REAL_BRIDGE_EDGE" if real_edges else (str(pair_edge.get("edge_type", "")) if pair_edge is not None else "DIRECT_EDGE")
                score = 0.0
                if real_edges:
                    score = max(score, float(real_edges[0].get("score", 0) or 0))
                if direct_edges:
                    score = max(score, float(direct_edges[0].get("score", 0) or 0))
                if pair_edge is not None:
                    score = max(score, float(pair_edge.get("score", 0) or 0))
                candidates.append({
                    "from_order_id": from_oid,
                    "to_order_id": to_oid,
                    "mixed_form": "GUARDED_TO_REAL",
                    "edge_type": preferred_type or "DIRECT_EDGE",
                    "score": score * 0.85,
                    "reason": "group_switch_or_width_tension",
                })
                attempts += 1

    return candidates[:max_attempts]


def _check_inter_block_boundary(
    prev_tail_id: str,
    cur_head_id: str,
    edge_lookup: Dict[tuple[str, str], Dict[str, Any]],
) -> str:
    """Check boundary between two blocks using the exact-pair edge lookup."""
    edge = edge_lookup.get((prev_tail_id, cur_head_id))
    if edge is None:
        return "UNRESOLVED"
    return str(edge.get("edge_type", "UNRESOLVED")) or "UNRESOLVED"


def _append_limited_example(diag: Dict[str, Any], key: str, item: Dict[str, Any], limit: int = 50) -> None:
    arr = diag.setdefault(key, [])
    if isinstance(arr, list) and len(arr) < limit:
        arr.append(item)


def _segment_real_tons(rows: List[Dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    total = 0.0
    for r in rows:
        if bool(r.get("is_virtual", False)):
            continue
        total += float(r.get("tons", 0.0) or 0.0)
    return float(total)


def _segment_viability_band(total_tons: float, cfg: PlannerConfig) -> str:
    min_ton = float(cfg.rule.campaign_ton_min)
    gap = max(0.0, min_ton - float(total_tons))
    near_gap = float(getattr(cfg.model, "near_viable_gap_tons", 80.0) or 80.0)
    merge_gap = float(getattr(cfg.model, "merge_candidate_gap_tons", 250.0) or 250.0)
    if gap <= 0.0:
        return "ALREADY_VIABLE"
    if gap <= near_gap:
        return "NEAR_VIABLE"
    if gap <= merge_gap:
        return "MERGE_CANDIDATE"
    return "HOPELESS_UNDERFILLED"


def _segment_viability_decision(total_tons: float, cfg: PlannerConfig) -> Dict[str, Any]:
    min_ton = float(cfg.rule.campaign_ton_min)
    gap = max(0.0, min_ton - float(total_tons))
    band = _segment_viability_band(total_tons, cfg)
    return {
        "total_tons": float(total_tons),
        "gap_tons": float(gap),
        "viability_band": band,
        "should_protect": band in {"ALREADY_VIABLE", "NEAR_VIABLE"},
        "should_defer": band in {"MERGE_CANDIDATE", "HOPELESS_UNDERFILLED"},
        "should_drop_last": band == "HOPELESS_UNDERFILLED",
    }


def _estimate_virtual_fill_count(needed_tons: float, cfg: PlannerConfig) -> tuple[int, float, str]:
    unit_tons = float(getattr(cfg.model, "virtual_fill_unit_tons_assumption", 0.0) or 0.0)
    if unit_tons <= 0.0:
        unit_tons = float(getattr(cfg.rule, "virtual_tons", 20.0) or 20.0)
    if unit_tons <= 0.0:
        return 0, 0.0, "INVALID_UNIT"
    count = int((max(0.0, float(needed_tons)) + unit_tons - 1e-9) // unit_tons)
    if float(count) * unit_tons + 1e-9 < float(needed_tons):
        count += 1
    return max(0, int(count)), float(unit_tons), "FIXED_UNIT_TONS"


def _evaluate_shadow_fill_for_campaign(
    *,
    line: str,
    steel_group: str,
    current_tons: float,
    cfg: PlannerConfig,
    remaining_budget_tons: Optional[float] = None,
) -> Dict[str, Any]:
    min_ton = float(cfg.rule.campaign_ton_min)
    needed_tons = max(0.0, min_ton - float(current_tons))
    needed_count, unit_tons, assumption = _estimate_virtual_fill_count(needed_tons, cfg)
    max_count = int(getattr(cfg.model, "virtual_max_count_per_campaign", 0) or 0)
    per_campaign_budget = float(getattr(cfg.model, "virtual_budget_per_campaign_tons", 0.0) or 0.0)
    allowed_lines = [str(x) for x in (getattr(cfg.model, "virtual_allowed_lines", []) or [])]
    allowed_groups = [str(x) for x in (getattr(cfg.model, "virtual_allowed_steel_groups", []) or [])]
    enabled = bool(getattr(cfg.model, "virtual_enabled", True)) and bool(getattr(cfg.model, "virtual_shadow_mode_enabled", True))
    fill_enabled = bool(getattr(cfg.model, "virtual_shadow_fill_enabled", True))
    if not enabled or not fill_enabled:
        return {
            "needed_tons": float(needed_tons),
            "needed_count": int(needed_count),
            "unit_tons": float(unit_tons),
            "assumption": str(assumption),
            "budget_ok": False,
            "viable": False,
            "reason_code": "RULE_NOT_CONFIGURED",
            "reason_detail": "virtual shadow fill disabled",
        }
    if float(needed_tons) <= 0.0:
        return {
            "needed_tons": 0.0,
            "needed_count": 0,
            "unit_tons": float(unit_tons),
            "assumption": str(assumption),
            "budget_ok": True,
            "viable": True,
            "reason_code": "VIABLE",
            "reason_detail": "already above minimum ton",
        }
    if allowed_lines and str(line) not in allowed_lines:
        return {
            "needed_tons": float(needed_tons),
            "needed_count": int(needed_count),
            "unit_tons": float(unit_tons),
            "assumption": str(assumption),
            "budget_ok": False,
            "viable": False,
            "reason_code": "LINE_NOT_ALLOWED",
            "reason_detail": f"line={line} not in virtual_allowed_lines",
        }
    if allowed_groups and str(steel_group or "") and str(steel_group) not in allowed_groups:
        return {
            "needed_tons": float(needed_tons),
            "needed_count": int(needed_count),
            "unit_tons": float(unit_tons),
            "assumption": str(assumption),
            "budget_ok": False,
            "viable": False,
            "reason_code": "STEEL_GROUP_NOT_ALLOWED",
            "reason_detail": f"steel_group={steel_group} not in virtual_allowed_steel_groups",
        }
    if not list(getattr(cfg.rule, "virtual_width_levels", ()) or ()) or not list(getattr(cfg.rule, "virtual_thickness_levels", ()) or ()):
        return {
            "needed_tons": float(needed_tons),
            "needed_count": int(needed_count),
            "unit_tons": float(unit_tons),
            "assumption": str(assumption),
            "budget_ok": False,
            "viable": False,
            "reason_code": "NO_COMPATIBLE_VIRTUAL_SPEC",
            "reason_detail": "virtual width/thickness spec levels are empty",
        }
    if unit_tons <= 0.0:
        return {
            "needed_tons": float(needed_tons),
            "needed_count": int(needed_count),
            "unit_tons": 0.0,
            "assumption": str(assumption),
            "budget_ok": False,
            "viable": False,
            "reason_code": "RULE_NOT_CONFIGURED",
            "reason_detail": "virtual fill unit tons not configured",
        }
    if needed_count > max_count:
        return {
            "needed_tons": float(needed_tons),
            "needed_count": int(needed_count),
            "unit_tons": float(unit_tons),
            "assumption": str(assumption),
            "budget_ok": False,
            "viable": False,
            "reason_code": "COUNT_LIMIT_EXCEEDED",
            "reason_detail": f"needed_count={needed_count} exceeds max_count={max_count}",
        }
    budget_limit = float(per_campaign_budget)
    if remaining_budget_tons is not None:
        budget_limit = min(budget_limit, float(remaining_budget_tons))
    budget_ok = bool(needed_tons <= budget_limit)
    if not budget_ok:
        return {
            "needed_tons": float(needed_tons),
            "needed_count": int(needed_count),
            "unit_tons": float(unit_tons),
            "assumption": str(assumption),
            "budget_ok": False,
            "viable": False,
            "reason_code": "BUDGET_EXCEEDED",
            "reason_detail": f"needed_tons={needed_tons:.1f} exceeds budget_limit={budget_limit:.1f}",
        }
    return {
        "needed_tons": float(needed_tons),
        "needed_count": int(needed_count),
        "unit_tons": float(unit_tons),
        "assumption": str(assumption),
        "budget_ok": True,
        "viable": True,
        "reason_code": "VIABLE",
        "reason_detail": "within line/group/budget/count constraints",
    }


def _evaluate_segment_min_ton_viability(
    segment_rows: List[Dict[str, Any]],
    cfg: PlannerConfig,
    line_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    _ctx = line_context or {}
    segment_total_tons = _segment_real_tons(segment_rows)
    min_ton = float(getattr(cfg.rule, "campaign_ton_min", 700.0) or 700.0)
    decision = _segment_viability_decision(segment_total_tons, cfg)
    below_min_gap = float(decision["gap_tons"])
    viability_band = str(decision["viability_band"])
    is_already_viable = bool(segment_total_tons >= min_ton)
    can_standalone_campaign = bool(is_already_viable)
    should_prevent_split = bool(not can_standalone_campaign and viability_band != "NEAR_VIABLE")
    reason_code = "OK_STANDALONE" if can_standalone_campaign else "SEGMENT_BELOW_MIN_TON"
    return {
        "line": str(_ctx.get("line", "")),
        "original_campaign_id": str(_ctx.get("original_campaign_id", "")),
        "segment_total_tons": float(segment_total_tons),
        "below_min_gap": float(below_min_gap),
        "viability_band": str(viability_band),
        "is_already_viable": bool(is_already_viable),
        "can_standalone_campaign": bool(can_standalone_campaign),
        "should_prevent_split": bool(should_prevent_split),
        "should_protect": bool(decision["should_protect"]),
        "should_defer": bool(decision["should_defer"]),
        "should_drop_last": bool(decision["should_drop_last"]),
        "reason_code": str(reason_code),
    }


def _defer_underfilled_segment(
    deferred_segments: List[Dict[str, Any]],
    *,
    line: str,
    original_campaign_id: str,
    deferred_campaign_id: str,
    slot_no: int,
    segment_total_tons: float,
    reason_code: str,
    viability_band: str,
) -> None:
    deferred_segments.append(
        {
            "line": str(line),
            "original_campaign_id": str(original_campaign_id),
            "deferred_campaign_id": str(deferred_campaign_id),
            "slot_no": int(slot_no),
            "segment_total_tons": float(segment_total_tons),
            "reason_code": str(reason_code),
            "viability_band": str(viability_band),
        }
    )


def _recompute_campaign_sequences(schedule_df: pd.DataFrame) -> pd.DataFrame:
    if schedule_df.empty:
        return schedule_df
    out = schedule_df.copy()
    if not {"line", "campaign_id"}.issubset(out.columns):
        return out
    if "master_slot" not in out.columns:
        out["master_slot"] = pd.to_numeric(out.get("slot_no", 0), errors="coerce").fillna(0).astype(int)
    sort_cols = [c for c in ["line", "master_slot", "global_sequence_on_line", "campaign_seq", "sequence_in_block"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    out["campaign_seq"] = out.groupby(["line", "campaign_id"], dropna=False).cumcount() + 1
    out["campaign_real_seq"] = out["campaign_seq"]
    return out


def _campaign_real_tons_map(schedule_df: pd.DataFrame) -> Dict[tuple[str, str], float]:
    if schedule_df.empty or not {"line", "campaign_id", "tons"}.issubset(schedule_df.columns):
        return {}
    work = schedule_df.copy()
    if "is_virtual" not in work.columns:
        work["is_virtual"] = False
    real = work[~work["is_virtual"].fillna(False).astype(bool)].copy()
    if real.empty:
        return {}
    grp = real.groupby(["line", "campaign_id"], dropna=False)["tons"].sum()
    return {(str(k[0]), str(k[1])): float(v) for k, v in grp.items()}


def _merge_adjacent_underfilled_segments(
    schedule_df: pd.DataFrame,
    cfg: PlannerConfig,
    orders_by_id: Dict[str, Dict[str, Any]],
    edge_lookup: Dict[tuple[str, str], Dict[str, Any]],
    diag_counters: Dict[str, Any],
) -> pd.DataFrame:
    if schedule_df.empty or not {"line", "campaign_id", "order_id"}.issubset(schedule_df.columns):
        return schedule_df
    out = _recompute_campaign_sequences(schedule_df)
    if "master_slot" not in out.columns:
        out["master_slot"] = pd.to_numeric(out.get("slot_no", 0), errors="coerce").fillna(0).astype(int)
    out["_slot_no_norm"] = pd.to_numeric(out["master_slot"], errors="coerce").fillna(0).astype(int)
    min_ton = float(cfg.rule.campaign_ton_min)
    max_ton = float(cfg.rule.campaign_ton_max)
    target_ton = float(getattr(cfg.rule, "campaign_ton_target", min_ton) or min_ton)

    merged = True
    while merged:
        merged = False
        tons_map = _campaign_real_tons_map(out)
        cmeta = (
            out.groupby(["line", "_slot_no_norm", "campaign_id"], as_index=False, dropna=False)
            .agg(
                first_pos=("campaign_seq", "min"),
                first_global=("global_sequence_on_line", "min"),
                last_global=("global_sequence_on_line", "max"),
            )
        )
        if cmeta.empty:
            break
        cmeta["real_tons"] = cmeta.apply(lambda r: tons_map.get((str(r["line"]), str(r["campaign_id"])), 0.0), axis=1)

        for line, lmeta in cmeta.groupby("line", dropna=False):
            lmeta = lmeta.sort_values(["_slot_no_norm", "first_global", "first_pos"], kind="mergesort").reset_index(drop=True)
            for i in range(len(lmeta) - 1):
                left_id = str(lmeta.iloc[i]["campaign_id"])
                right_id = str(lmeta.iloc[i + 1]["campaign_id"])
                left_slot = int(lmeta.iloc[i]["_slot_no_norm"] or 0)
                right_slot = int(lmeta.iloc[i + 1]["_slot_no_norm"] or 0)
                if left_slot != right_slot:
                    continue
                left_last = int(lmeta.iloc[i]["last_global"] or 0)
                right_first = int(lmeta.iloc[i + 1]["first_global"] or 0)
                if left_last >= right_first:
                    # Not strictly non-overlapping contiguous campaigns; skip to avoid hidden pair breaks.
                    continue
                left_tons = float(lmeta.iloc[i]["real_tons"] or 0.0)
                right_tons = float(lmeta.iloc[i + 1]["real_tons"] or 0.0)
                if not (left_tons < min_ton and right_tons < min_ton):
                    continue
                diag_counters["segment_remerge_attempt_count"] = int(diag_counters.get("segment_remerge_attempt_count", 0) or 0) + 1
                left_rows = out[
                    (out["line"] == line)
                    & (out["_slot_no_norm"] == left_slot)
                    & (out["campaign_id"].astype(str) == left_id)
                ].sort_values("campaign_seq", kind="mergesort")
                right_rows = out[
                    (out["line"] == line)
                    & (out["_slot_no_norm"] == right_slot)
                    & (out["campaign_id"].astype(str) == right_id)
                ].sort_values("campaign_seq", kind="mergesort")
                if left_rows.empty or right_rows.empty:
                    continue
                if set(left_rows["order_id"].astype(str)) & set(right_rows["order_id"].astype(str)):
                    diag_counters["segment_remerge_rejected_due_to_duplicate"] = int(diag_counters.get("segment_remerge_rejected_due_to_duplicate", 0) or 0) + 1
                    continue
                lt = str(left_rows.iloc[-1]["order_id"])
                rh = str(right_rows.iloc[0]["order_id"])
                edge = edge_lookup.get((lt, rh))
                edge_type = str(edge.get("edge_type", "UNRESOLVED")) if edge is not None else "UNRESOLVED"
                ok, _ = _pair_hard_feasible(orders_by_id.get(lt, {}), orders_by_id.get(rh, {}), cfg, edge_type)
                if not ok:
                    diag_counters["segment_remerge_rejected_due_to_pair_invalid"] = int(diag_counters.get("segment_remerge_rejected_due_to_pair_invalid", 0) or 0) + 1
                    continue
                merged_tons = left_tons + right_tons
                before_distance = abs(left_tons - target_ton) + abs(right_tons - target_ton)
                after_distance = abs(merged_tons - target_ton)
                ton_gain = (merged_tons >= min_ton and merged_tons <= max_ton) or (after_distance < before_distance)
                if not ton_gain:
                    diag_counters["segment_remerge_rejected_due_to_ton_no_gain"] = int(diag_counters.get("segment_remerge_rejected_due_to_ton_no_gain", 0) or 0) + 1
                    continue
                out.loc[
                    (out["line"] == line)
                    & (out["_slot_no_norm"] == right_slot)
                    & (out["campaign_id"].astype(str) == right_id),
                    "campaign_id",
                ] = left_id
                if "campaign_id_hint" in out.columns:
                    out.loc[
                        (out["line"] == line)
                        & (out["_slot_no_norm"] == right_slot)
                        & (out["campaign_id"].astype(str) == right_id),
                        "campaign_id_hint",
                    ] = left_id
                diag_counters["segment_remerge_success_count"] = int(diag_counters.get("segment_remerge_success_count", 0) or 0) + 1
                merged = True
                break
            if merged:
                break
    out = out.drop(columns=["_slot_no_norm"], errors="ignore")
    return _recompute_campaign_sequences(out)


def _finalize_deferred_segments(
    schedule_df: pd.DataFrame,
    deferred_segments: List[Dict[str, Any]],
    cfg: PlannerConfig,
    orders_by_id: Dict[str, Dict[str, Any]],
    edge_lookup: Dict[tuple[str, str], Dict[str, Any]],
    diag_counters: Dict[str, Any],
) -> pd.DataFrame:
    if schedule_df.empty or not deferred_segments:
        return schedule_df
    out = _recompute_campaign_sequences(schedule_df)
    if "master_slot" not in out.columns:
        out["master_slot"] = pd.to_numeric(out.get("slot_no", 0), errors="coerce").fillna(0).astype(int)
    out["_slot_no_norm"] = pd.to_numeric(out["master_slot"], errors="coerce").fillna(0).astype(int)
    deferred_ids = [str(x.get("deferred_campaign_id", "")) for x in deferred_segments if str(x.get("deferred_campaign_id", ""))]
    deferred_ids = [x for x in deferred_ids if x]
    if not deferred_ids:
        return out
    diag_counters["deferred_segment_count"] = int(len(deferred_ids))

    for d_idx, deferred_id in enumerate(deferred_ids, start=1):
        mask = out["campaign_id"].astype(str) == deferred_id
        if not bool(mask.any()):
            continue
        drows = out[mask].sort_values("campaign_seq", kind="mergesort")
        if drows.empty:
            continue
        dline = str(drows.iloc[0]["line"])
        dslot = int(pd.to_numeric(drows.iloc[0].get("_slot_no_norm", 0), errors="coerce") or 0)
        dtons = _segment_real_tons(drows.to_dict("records"))
        dband = _segment_viability_band(dtons, cfg)
        dids = set(drows["order_id"].astype(str))
        dfirst = int(pd.to_numeric(drows["global_sequence_on_line"], errors="coerce").min() or 0)
        dlast = int(pd.to_numeric(drows["global_sequence_on_line"], errors="coerce").max() or 0)
        if dband == "NEAR_VIABLE":
            diag_counters["deferred_near_viable_count"] = int(diag_counters.get("deferred_near_viable_count", 0) or 0) + 1
        if dband == "MERGE_CANDIDATE":
            diag_counters["deferred_merge_candidate_count"] = int(diag_counters.get("deferred_merge_candidate_count", 0) or 0) + 1
        if dband == "HOPELESS_UNDERFILLED":
            diag_counters["deferred_hopeless_count"] = int(diag_counters.get("deferred_hopeless_count", 0) or 0) + 1
        candidates = (
            out[
                (out["line"].astype(str) == dline)
                & (out["_slot_no_norm"] == dslot)
                & (out["campaign_id"].astype(str) != deferred_id)
            ]
            .groupby(["campaign_id", "_slot_no_norm"], as_index=False, dropna=False)
            .agg(
                first_global=("global_sequence_on_line", "min"),
                last_global=("global_sequence_on_line", "max"),
            )
        )
        if not candidates.empty:
            candidates = candidates.sort_values(["first_global"], kind="mergesort")

        merged = False
        for _, crow in candidates.iterrows():
            cid = str(crow["campaign_id"])
            cslot = int(pd.to_numeric(crow.get("_slot_no_norm", 0), errors="coerce") or 0)
            if cslot != dslot:
                continue
            cfirst = int(pd.to_numeric(crow.get("first_global", 0), errors="coerce") or 0)
            clast = int(pd.to_numeric(crow.get("last_global", 0), errors="coerce") or 0)
            relation = ""
            if clast < dfirst:
                relation = "CANDIDATE_BEFORE_DEFERRED"
            elif dlast < cfirst:
                relation = "DEFERRED_BEFORE_CANDIDATE"
            else:
                # overlapping/interleaving campaigns are not safe to merge
                continue
            crows = out[
                (out["line"].astype(str) == dline)
                & (out["_slot_no_norm"] == dslot)
                & (out["campaign_id"].astype(str) == cid)
            ].sort_values("campaign_seq", kind="mergesort")
            if crows.empty:
                continue
            if dids & set(crows["order_id"].astype(str)):
                continue

            if relation == "CANDIDATE_BEFORE_DEFERRED":
                left_oid = str(crows.iloc[-1]["order_id"])
                right_oid = str(drows.iloc[0]["order_id"])
            else:
                left_oid = str(drows.iloc[-1]["order_id"])
                right_oid = str(crows.iloc[0]["order_id"])

            edge = edge_lookup.get((left_oid, right_oid))
            edge_type = str(edge.get("edge_type", "UNRESOLVED")) if edge is not None else "UNRESOLVED"
            ok, _ = _pair_hard_feasible(orders_by_id.get(left_oid, {}), orders_by_id.get(right_oid, {}), cfg, edge_type)
            if ok:
                out.loc[mask & (out["_slot_no_norm"] == dslot), "campaign_id"] = cid
                if "campaign_id_hint" in out.columns:
                    out.loc[mask & (out["_slot_no_norm"] == dslot), "campaign_id_hint"] = cid
                diag_counters["deferred_segment_finalized_count"] = int(diag_counters.get("deferred_segment_finalized_count", 0) or 0) + 1
                merged = True
                break
        if not merged:
            if dband == "NEAR_VIABLE":
                # Keep near-viable deferred campaign alive for downstream formal fill trial.
                fallback_id = deferred_id.replace("__deferred_", "__seg_")
                if fallback_id == deferred_id:
                    fallback_id = f"{deferred_id}__seg_keep"
                out.loc[mask & (out["_slot_no_norm"] == dslot), "campaign_id"] = fallback_id
                if "campaign_id_hint" in out.columns:
                    out.loc[mask & (out["_slot_no_norm"] == dslot), "campaign_id_hint"] = fallback_id
                diag_counters["near_viable_preserved_count"] = int(diag_counters.get("near_viable_preserved_count", 0) or 0) + 1
            else:
                # Keep hopeless deferred segments non-materialized in final schedule.
                out = out.loc[~(mask & (out["_slot_no_norm"] == dslot))].copy()
            diag_counters["deferred_segment_finalized_count"] = int(diag_counters.get("deferred_segment_finalized_count", 0) or 0) + 1

    out = out.drop(columns=["_slot_no_norm"], errors="ignore")
    out = _recompute_campaign_sequences(out)
    out = _merge_adjacent_underfilled_segments(
        schedule_df=out,
        cfg=cfg,
        orders_by_id=orders_by_id,
        edge_lookup=edge_lookup,
        diag_counters=diag_counters,
    )
    return out


def _pick_virtual_width(prev_width: float, cfg: PlannerConfig) -> float:
    levels = sorted([float(v) for v in (getattr(cfg.rule, "virtual_width_levels", ()) or ())])
    if not levels:
        return float(prev_width)
    # Prefer non-increase candidate closest to previous width.
    lower = [v for v in levels if v <= float(prev_width)]
    if lower:
        picked = max(lower)
        if float(prev_width) - picked <= float(cfg.rule.max_width_drop):
            return float(picked)
    for v in sorted(levels, reverse=True):
        if v <= float(prev_width) + 1e-6 and float(prev_width) - v <= float(cfg.rule.max_width_drop):
            return float(v)
    return float(levels[0])


def _pick_virtual_thickness(prev_thickness: float, cfg: PlannerConfig) -> float:
    levels = [float(v) for v in (getattr(cfg.rule, "virtual_thickness_levels", ()) or ())]
    if not levels:
        return float(prev_thickness)
    return float(min(levels, key=lambda v: abs(v - float(prev_thickness))))


def _build_virtual_fill_rows_for_campaign(
    campaign_rows: pd.DataFrame,
    *,
    line: str,
    campaign_id: str,
    needed_tons: float,
    needed_count: int,
    unit_tons: float,
    cfg: PlannerConfig,
) -> List[Dict[str, Any]]:
    if campaign_rows.empty or needed_count <= 0 or needed_tons <= 0.0:
        return []
    last_real = campaign_rows.sort_values("campaign_seq", kind="mergesort").iloc[-1]
    base_seq = int(pd.to_numeric(last_real.get("campaign_seq", 0), errors="coerce") or 0)
    base_slot_seq = int(pd.to_numeric(last_real.get("sequence_in_slot", base_seq), errors="coerce") or base_seq)
    base_global = float(pd.to_numeric(last_real.get("global_sequence_on_line", 0), errors="coerce") or 0.0)
    master_slot = int(pd.to_numeric(last_real.get("master_slot", last_real.get("slot_no", 0)), errors="coerce") or 0)
    prev_width = float(last_real.get("width", 0.0) or 0.0)
    prev_thickness = float(last_real.get("thickness", 0.0) or 0.0)
    prev_temp_min = float(last_real.get("temp_min", 0.0) or 0.0)
    prev_temp_max = float(last_real.get("temp_max", 0.0) or 0.0)
    prev_group = str(last_real.get("steel_group", "") or "")
    prev_grade = str(last_real.get("grade", "") or "")

    v_width = _pick_virtual_width(prev_width, cfg)
    v_thk = _pick_virtual_thickness(prev_thickness, cfg)
    v_temp_min = max(float(cfg.rule.virtual_temp_min), prev_temp_min)
    v_temp_max = min(float(cfg.rule.virtual_temp_max), prev_temp_max)
    if v_temp_max - v_temp_min < 10.0:
        v_temp_min = float(cfg.rule.virtual_temp_min)
        v_temp_max = float(cfg.rule.virtual_temp_max)
    v_temp_mean = (v_temp_min + v_temp_max) / 2.0

    rows: List[Dict[str, Any]] = []
    remaining = float(needed_tons)
    for i in range(int(needed_count)):
        tons_i = float(min(unit_tons, remaining))
        if i == int(needed_count) - 1:
            tons_i = float(max(0.0, remaining))
        if tons_i <= 0.0:
            continue
        rows.append(
            {
                "line": str(line),
                "assigned_line": str(line),
                "campaign_id": str(campaign_id),
                "campaign_id_hint": str(campaign_id),
                "master_slot": int(master_slot),
                "assigned_slot": int(master_slot),
                "slot_no": int(master_slot),
                "campaign_no": int(master_slot),
                "campaign_seq": int(base_seq + len(rows) + 1),
                "campaign_real_seq": int(base_seq + len(rows) + 1),
                "sequence_in_slot": int(base_slot_seq + len(rows) + 1),
                "sequence_in_block": int(base_slot_seq + len(rows) + 1),
                "global_sequence_on_line": float(base_global + 0.001 * (len(rows) + 1)),
                "global_seq": int(last_real.get("global_seq", 0) or 0),
                "line_seq": int(last_real.get("line_seq", 0) or 0),
                "order_id": f"VIRTUAL_FILL::{line}::{campaign_id}::{i + 1}",
                "source_order_id": "",
                "selected_edge_type": "VIRTUAL_FILL_EDGE",
                "selected_bridge_path": "",
                "bridge_count": 0,
                "is_virtual": True,
                "virtual_usage_type": "fill",
                "virtual_fill_trial": True,
                "tons": float(round(tons_i, 3)),
                "width": float(v_width),
                "thickness": float(v_thk),
                "grade": str(prev_grade),
                "steel_group": str(prev_group),
                "temp_min": float(v_temp_min),
                "temp_max": float(v_temp_max),
                "temp_mean": float(v_temp_mean),
                "backlog": 0.0,
                "due_date": last_real.get("due_date"),
                "due_bucket": last_real.get("due_bucket", ""),
                "mixed_bridge_applied": False,
                "inter_block_boundary_before": False,
                "boundary_from_block_id": None,
                "boundary_to_block_id": None,
                "boundary_edge_type": None,
            }
        )
        remaining = max(0.0, remaining - tons_i)
    return rows


def _run_formal_virtual_fill_trial(schedule_df: pd.DataFrame, cfg: PlannerConfig, diag_counters: Dict[str, Any]) -> pd.DataFrame:
    if schedule_df.empty:
        return schedule_df
    if not bool(getattr(cfg.model, "virtual_formal_fill_enabled", False)):
        return schedule_df
    if not bool(getattr(cfg.model, "virtual_enabled", True)):
        return schedule_df
    if not bool(getattr(cfg.model, "virtual_formal_fill_tail_only", True)):
        return schedule_df
    if not {"line", "campaign_id", "tons", "order_id"}.issubset(schedule_df.columns):
        return schedule_df

    out = _recompute_campaign_sequences(schedule_df)
    if "is_virtual" not in out.columns:
        out["is_virtual"] = False
    if "master_slot" not in out.columns:
        out["master_slot"] = pd.to_numeric(out.get("slot_no", 0), errors="coerce").fillna(0).astype(int)

    max_gap = float(getattr(cfg.model, "virtual_formal_fill_max_gap_tons", 80.0) or 80.0)
    max_count = int(getattr(cfg.model, "virtual_formal_fill_max_count_per_campaign", 3) or 3)
    total_budget = float(getattr(cfg.model, "virtual_budget_total_tons", 0.0) or 0.0)
    remaining_budget = max(0.0, total_budget)
    min_ton = float(cfg.rule.campaign_ton_min)

    trial_rows: List[Dict[str, Any]] = []
    success_campaigns: List[str] = []
    rollback_reasons: List[Dict[str, Any]] = []
    trial_count = 0
    success_count = 0
    rollback_count = 0

    real = out[~out["is_virtual"].fillna(False).astype(bool)].copy()
    if real.empty:
        diag_counters["formal_virtual_fill_trial_rows"] = []
        return out

    csum = (
        real.groupby(["line", "campaign_id"], as_index=False, dropna=False)
        .agg(current_tons=("tons", "sum"))
        .sort_values(["line", "campaign_id"], kind="mergesort")
    )
    for _, crow in csum.iterrows():
        line = str(crow["line"])
        campaign_id = str(crow["campaign_id"])
        current_tons = float(crow["current_tons"] or 0.0)
        gap = max(0.0, min_ton - current_tons)
        if gap <= 0.0:
            continue
        if _segment_viability_band(current_tons, cfg) != "NEAR_VIABLE":
            continue
        if gap > max_gap:
            continue
        trial_count += 1

        camp_rows = real[(real["line"].astype(str) == line) & (real["campaign_id"].astype(str) == campaign_id)].sort_values("campaign_seq", kind="mergesort")
        if camp_rows.empty:
            rollback_count += 1
            rollback_reasons.append({"line": line, "campaign_id": campaign_id, "reason": "EMPTY_CAMPAIGN"})
            continue
        tail = camp_rows.iloc[-1]
        steel_group = str(tail.get("steel_group", "") or "")
        shadow_eval = _evaluate_shadow_fill_for_campaign(
            line=line,
            steel_group=steel_group,
            current_tons=current_tons,
            cfg=cfg,
            remaining_budget_tons=remaining_budget,
        )
        if str(shadow_eval.get("reason_code", "")) != "VIABLE":
            rollback_count += 1
            rollback_reasons.append({"line": line, "campaign_id": campaign_id, "reason": f"SHADOW_{shadow_eval.get('reason_code', 'UNKNOWN')}"})
            continue
        needed_count = int(shadow_eval.get("needed_count", 0) or 0)
        needed_tons = float(shadow_eval.get("needed_tons", 0.0) or 0.0)
        unit_tons = float(shadow_eval.get("unit_tons", 0.0) or 0.0)
        if needed_count <= 0 or needed_count > max_count:
            rollback_count += 1
            rollback_reasons.append({"line": line, "campaign_id": campaign_id, "reason": "COUNT_LIMIT_EXCEEDED"})
            continue

        fill_rows = _build_virtual_fill_rows_for_campaign(
            camp_rows,
            line=line,
            campaign_id=campaign_id,
            needed_tons=needed_tons,
            needed_count=needed_count,
            unit_tons=unit_tons,
            cfg=cfg,
        )
        if not fill_rows:
            rollback_count += 1
            rollback_reasons.append({"line": line, "campaign_id": campaign_id, "reason": "NO_COMPATIBLE_VIRTUAL_SPEC"})
            continue

        prev_order = dict(tail)
        pair_ok = True
        pair_reason = ""
        for fr in fill_rows:
            ok, reasons = _pair_hard_feasible(prev_order, fr, cfg, "DIRECT_EDGE")
            if not ok:
                pair_ok = False
                pair_reason = ",".join(reasons[:4]) if reasons else "PAIR_INVALID"
                break
            prev_order = fr
        if not pair_ok:
            rollback_count += 1
            rollback_reasons.append({"line": line, "campaign_id": campaign_id, "reason": f"PAIR_INVALID:{pair_reason}"})
            continue

        after_tons = current_tons + sum(float(r.get("tons", 0.0) or 0.0) for r in fill_rows)
        if after_tons + 1e-6 < min_ton:
            rollback_count += 1
            rollback_reasons.append({"line": line, "campaign_id": campaign_id, "reason": "TON_NO_IMPROVEMENT"})
            continue

        out = pd.concat([out, pd.DataFrame(fill_rows)], ignore_index=True)
        remaining_budget = max(0.0, remaining_budget - needed_tons)
        success_count += 1
        success_campaigns.append(f"{line}#{campaign_id}")
        trial_rows.append(
            {
                "line": line,
                "campaign_id": campaign_id,
                "current_tons_before": float(round(current_tons, 3)),
                "current_tons_after": float(round(after_tons, 3)),
                "gap_before": float(round(gap, 3)),
                "virtual_fill_count": int(needed_count),
                "virtual_fill_tons": float(round(after_tons - current_tons, 3)),
                "trial_result": "SUCCESS",
                "rollback_reason": "",
            }
        )

    for item in rollback_reasons:
        trial_rows.append(
            {
                "line": str(item.get("line", "")),
                "campaign_id": str(item.get("campaign_id", "")),
                "current_tons_before": None,
                "current_tons_after": None,
                "gap_before": None,
                "virtual_fill_count": 0,
                "virtual_fill_tons": 0.0,
                "trial_result": "ROLLED_BACK",
                "rollback_reason": str(item.get("reason", "")),
            }
        )

    diag_counters["formal_virtual_fill_trial_count"] = int(trial_count)
    diag_counters["formal_virtual_fill_success_count"] = int(success_count)
    diag_counters["formal_virtual_fill_rollback_count"] = int(rollback_count)
    diag_counters["formal_virtual_fill_success_campaigns"] = list(success_campaigns)
    diag_counters["formal_virtual_fill_rollback_reasons"] = list(rollback_reasons[:50])
    diag_counters["formal_virtual_fill_trial_rows"] = trial_rows

    out = _recompute_campaign_sequences(out)
    return out


def _local_reassemble_for_near_viable(
    schedule_df: pd.DataFrame,
    cfg: PlannerConfig,
    orders_by_id: Dict[str, Dict[str, Any]],
    edge_lookup: Dict[tuple[str, str], Dict[str, Any]],
    diag_counters: Dict[str, Any],
) -> pd.DataFrame:
    if schedule_df.empty or not {"line", "campaign_id", "order_id", "tons"}.issubset(schedule_df.columns):
        return schedule_df

    out = _recompute_campaign_sequences(schedule_df)
    if "is_virtual" not in out.columns:
        out["is_virtual"] = False
    if "master_slot" not in out.columns:
        out["master_slot"] = pd.to_numeric(out.get("slot_no", 0), errors="coerce").fillna(0).astype(int)
    for col in ["promoted_by_local_reassemble", "hopeless_dropped_for_near_viable"]:
        if col not in out.columns:
            out[col] = False

    promoted_rows: List[Dict[str, Any]] = []
    min_ton = float(cfg.rule.campaign_ton_min)
    max_ton = float(cfg.rule.campaign_ton_max)
    changed = True
    while changed:
        changed = False
        real = out[~out["is_virtual"].fillna(False).astype(bool)].copy()
        if real.empty:
            break
        cmeta = (
            real.groupby(["line", "campaign_id"], as_index=False, dropna=False)
            .agg(
                tons=("tons", "sum"),
                first_global=("global_sequence_on_line", "min"),
                last_global=("global_sequence_on_line", "max"),
                master_slot=("master_slot", "first"),
            )
            .sort_values(["line", "first_global"], kind="mergesort")
        )
        cmeta["viability_band"] = cmeta["tons"].map(lambda v: _segment_viability_band(float(v or 0.0), cfg))

        for line, lmeta in cmeta.groupby("line", dropna=False, sort=False):
            lmeta = lmeta.sort_values("first_global", kind="mergesort").reset_index(drop=True)
            for ridx, receiver in lmeta.iterrows():
                if str(receiver["viability_band"]) != "NEAR_VIABLE":
                    continue
                receiver_id = str(receiver["campaign_id"])
                receiver_tons = float(receiver["tons"] or 0.0)
                receiver_gap = max(0.0, min_ton - receiver_tons)
                if receiver_gap <= 0.0:
                    continue

                neighbor_indexes = [ridx - 1, ridx + 1, ridx - 2, ridx + 2]
                for didx in neighbor_indexes:
                    if didx < 0 or didx >= len(lmeta):
                        continue
                    donor = lmeta.iloc[didx]
                    donor_id = str(donor["campaign_id"])
                    donor_band = str(donor["viability_band"])
                    if donor_id == receiver_id or donor_band not in {"MERGE_CANDIDATE", "HOPELESS_UNDERFILLED"}:
                        continue

                    diag_counters["near_viable_promote_attempt_count"] = int(diag_counters.get("near_viable_promote_attempt_count", 0) or 0) + 1
                    receiver_rows = out[
                        (out["line"].astype(str) == str(line))
                        & (out["campaign_id"].astype(str) == receiver_id)
                        & (~out["is_virtual"].fillna(False).astype(bool))
                    ].sort_values("global_sequence_on_line", kind="mergesort")
                    donor_rows = out[
                        (out["line"].astype(str) == str(line))
                        & (out["campaign_id"].astype(str) == donor_id)
                        & (~out["is_virtual"].fillna(False).astype(bool))
                    ].sort_values("global_sequence_on_line", kind="mergesort")
                    if receiver_rows.empty or donor_rows.empty:
                        continue
                    if set(receiver_rows["order_id"].astype(str)) & set(donor_rows["order_id"].astype(str)):
                        diag_counters["near_viable_promote_rejected_due_to_duplicate"] = int(diag_counters.get("near_viable_promote_rejected_due_to_duplicate", 0) or 0) + 1
                        continue

                    donor_before = float(donor["last_global"] or 0.0) < float(receiver["first_global"] or 0.0)
                    if donor_before:
                        left_oid = str(donor_rows.iloc[-1]["order_id"])
                        right_oid = str(receiver_rows.iloc[0]["order_id"])
                    else:
                        left_oid = str(receiver_rows.iloc[-1]["order_id"])
                        right_oid = str(donor_rows.iloc[0]["order_id"])
                    edge = edge_lookup.get((left_oid, right_oid))
                    edge_type = str(edge.get("edge_type", "UNRESOLVED")) if edge is not None else "UNRESOLVED"
                    ok, _ = _pair_hard_feasible(orders_by_id.get(left_oid, {}), orders_by_id.get(right_oid, {}), cfg, edge_type)
                    if not ok:
                        diag_counters["near_viable_promote_rejected_due_to_pair_invalid"] = int(diag_counters.get("near_viable_promote_rejected_due_to_pair_invalid", 0) or 0) + 1
                        continue

                    donor_tons = float(donor["tons"] or 0.0)
                    merged_tons = receiver_tons + donor_tons
                    if merged_tons <= receiver_tons or merged_tons > max_ton:
                        diag_counters["near_viable_promote_rejected_due_to_no_ton_gain"] = int(diag_counters.get("near_viable_promote_rejected_due_to_no_ton_gain", 0) or 0) + 1
                        continue

                    donor_mask = (
                        (out["line"].astype(str) == str(line))
                        & (out["campaign_id"].astype(str) == donor_id)
                    )
                    receiver_slot = int(pd.to_numeric(receiver.get("master_slot", 0), errors="coerce") or 0)
                    out.loc[donor_mask, "campaign_id"] = receiver_id
                    if "campaign_id_hint" in out.columns:
                        out.loc[donor_mask, "campaign_id_hint"] = receiver_id
                    for slot_col in ["master_slot", "assigned_slot", "slot_no", "campaign_no"]:
                        if slot_col in out.columns:
                            out.loc[donor_mask, slot_col] = receiver_slot
                    out.loc[donor_mask, "promoted_by_local_reassemble"] = True
                    if donor_band == "HOPELESS_UNDERFILLED":
                        out.loc[donor_mask, "hopeless_dropped_for_near_viable"] = True
                        diag_counters["hopeless_segments_dropped_for_near_viable_count"] = int(diag_counters.get("hopeless_segments_dropped_for_near_viable_count", 0) or 0) + 1
                    diag_counters["near_viable_promote_success_count"] = int(diag_counters.get("near_viable_promote_success_count", 0) or 0) + 1
                    promoted_rows.append(
                        {
                            "line": str(line),
                            "receiver_campaign_id": receiver_id,
                            "donor_campaign_id": donor_id,
                            "receiver_tons_before": float(round(receiver_tons, 3)),
                            "donor_tons": float(round(donor_tons, 3)),
                            "receiver_tons_after": float(round(merged_tons, 3)),
                            "donor_viability_band": donor_band,
                            "receiver_gap_before": float(round(receiver_gap, 3)),
                        }
                    )
                    changed = True
                    break
                if changed:
                    break
            if changed:
                out = _recompute_campaign_sequences(out)
                break

    diag_counters["near_viable_promote_rows"] = promoted_rows
    return _recompute_campaign_sequences(out)


def _enumerate_local_donor_candidates_for_second_receiver(
    *,
    schedule_df: pd.DataFrame,
    line_meta: pd.DataFrame,
    receiver_idx: int,
    receiver: pd.Series,
    receiver_rows: pd.DataFrame,
    cfg: PlannerConfig,
    orders_by_id: Dict[str, Dict[str, Any]],
    edge_lookup: Dict[tuple[str, str], Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Enumerate local donor proof rows; rejected rows are as important as accepted ones."""
    candidates: List[Dict[str, Any]] = []
    if line_meta.empty or receiver_rows.empty:
        return candidates
    min_ton = float(cfg.rule.campaign_ton_min)
    max_ton = float(cfg.rule.campaign_ton_max)
    receiver_line = str(receiver["line"])
    receiver_id = str(receiver["campaign_id"])
    before_tons = float(receiver["tons"] or 0.0)
    before_gap = max(0.0, min_ton - before_tons)
    receiver_first = float(receiver["first_global"] or 0.0)
    receiver_last = float(receiver["last_global"] or 0.0)
    # Keep the pass local: immediate neighbors first, then one-hop near neighbors.
    donor_indexes = [receiver_idx - 1, receiver_idx + 1, receiver_idx - 2, receiver_idx + 2]
    seen: set[str] = set()
    for didx in donor_indexes:
        if didx < 0 or didx >= len(line_meta):
            continue
        donor = line_meta.iloc[didx]
        donor_id = str(donor["campaign_id"])
        if donor_id in seen or donor_id == receiver_id:
            continue
        seen.add(donor_id)
        donor_tons = float(donor["tons"] or 0.0)
        donor_band = str(donor["viability_band"])
        donor_mask = (schedule_df["line"].astype(str) == receiver_line) & (schedule_df["campaign_id"].astype(str) == donor_id)
        donor_rows = schedule_df[
            donor_mask & (~schedule_df["is_virtual"].fillna(False).astype(bool))
        ].sort_values("global_sequence_on_line", kind="mergesort")
        donor_before = float(donor["last_global"] or 0.0) < receiver_first
        donor_after = float(donor["first_global"] or 0.0) > receiver_last
        adjacent_side = "before" if donor_before else "after" if donor_after else "overlap"
        pair_feasible = False
        duplicate_safe = False
        rejection_reason = ""
        left_oid = ""
        right_oid = ""
        if donor_rows.empty:
            rejection_reason = "EMPTY_DONOR_ROWS"
        elif donor_tons >= min_ton:
            rejection_reason = "HEALTHY_CAMPAIGN_PROTECTED"
        elif donor_band not in {"HOPELESS_UNDERFILLED", "MERGE_CANDIDATE"}:
            rejection_reason = "DONOR_BAND_NOT_ALLOWED"
        else:
            receiver_oids = set(receiver_rows["order_id"].astype(str))
            donor_oids = set(donor_rows["order_id"].astype(str))
            duplicate_safe = not bool(receiver_oids & donor_oids)
            if not duplicate_safe:
                rejection_reason = "DUPLICATE_ORDER"
            else:
                if donor_before:
                    left_oid = str(donor_rows.iloc[-1]["order_id"])
                    right_oid = str(receiver_rows.iloc[0]["order_id"])
                else:
                    left_oid = str(receiver_rows.iloc[-1]["order_id"])
                    right_oid = str(donor_rows.iloc[0]["order_id"])
                edge = edge_lookup.get((left_oid, right_oid))
                edge_type = str(edge.get("edge_type", "UNRESOLVED")) if edge is not None else "UNRESOLVED"
                pair_feasible, _ = _pair_hard_feasible(orders_by_id.get(left_oid, {}), orders_by_id.get(right_oid, {}), cfg, edge_type)
                if not pair_feasible:
                    rejection_reason = "PAIR_INVALID"
                elif before_tons + donor_tons <= before_tons or before_tons + donor_tons > max_ton:
                    rejection_reason = "NO_TON_GAIN"
                else:
                    after_tons = before_tons + donor_tons
                    after_gap = max(0.0, min_ton - after_tons)
                    after_band = _segment_viability_band(after_tons, cfg)
                    if after_band == str(receiver["viability_band"]) and after_gap >= before_gap:
                        rejection_reason = "NO_BAND_OR_GAP_PROGRESS"
                    else:
                        rejection_reason = "ACCEPTABLE"
        after_estimate = before_tons + max(0.0, donor_tons)
        candidates.append(
            {
                "line": receiver_line,
                "second_receiver_campaign_id": receiver_id,
                "donor_campaign_id": donor_id,
                "donor_tons": float(round(donor_tons, 3)),
                "donor_viability_band": donor_band,
                "transferable_rows": int(len(donor_rows)),
                "transferable_tons": float(round(donor_tons, 3)),
                "receiver_gap_before": float(round(before_gap, 3)),
                "receiver_gap_after_estimate": float(round(max(0.0, min_ton - after_estimate), 3)),
                "receiver_viability_band_before": str(receiver["viability_band"]),
                "receiver_viability_band_after_estimate": _segment_viability_band(after_estimate, cfg),
                "pair_feasible": bool(pair_feasible),
                "duplicate_safe": bool(duplicate_safe),
                "rejection_reason": rejection_reason,
                "selected_for_reallocation": False,
                "adjacent_side": adjacent_side,
                "left_order_id": left_oid,
                "right_order_id": right_oid,
                "neighbor_rank": int(abs(didx - receiver_idx)),
            }
        )
    return candidates


def _summarize_second_receiver_failure(candidates: List[Dict[str, Any]]) -> str:
    if not candidates:
        return "NO_LOCAL_DONOR"
    reasons = [str(c.get("rejection_reason", "")) for c in candidates]
    if reasons and all(r == "HEALTHY_CAMPAIGN_PROTECTED" for r in reasons):
        return "ONLY_HEALTHY_DONORS_AVAILABLE"
    non_healthy = [r for r in reasons if r != "HEALTHY_CAMPAIGN_PROTECTED"]
    if non_healthy and all(r == "PAIR_INVALID" for r in non_healthy):
        return "ALL_DONORS_PAIR_INVALID"
    if non_healthy and all(r == "DUPLICATE_ORDER" for r in non_healthy):
        return "ALL_DONORS_DUPLICATE_RISK"
    if non_healthy and all(r in {"NO_TON_GAIN", "NO_BAND_OR_GAP_PROGRESS"} for r in non_healthy):
        return "ALL_DONORS_NO_TON_GAIN"
    if non_healthy and all(r in {"DONOR_BAND_NOT_ALLOWED", "EMPTY_DONOR_ROWS"} for r in non_healthy):
        return "SECOND_RECEIVER_NOT_WORTH_PROMOTING"
    return "NO_LOCAL_DONOR"


def _shadow_bridge_analysis_for_underfilled_campaigns(
    schedule_df: pd.DataFrame,
    cfg: PlannerConfig,
    diag_counters: Dict[str, Any],
) -> None:
    if schedule_df.empty or not {"line", "campaign_id", "tons"}.issubset(schedule_df.columns):
        diag_counters["second_receiver_shadow_bridge_reason_code"] = "BRIDGE_NOT_ALLOWED_BY_CONFIG"
        return
    enabled = bool(getattr(cfg.model, "virtual_enabled", True)) and bool(getattr(cfg.model, "virtual_shadow_mode_enabled", True))
    bridge_enabled = bool(getattr(cfg.model, "virtual_shadow_bridge_enabled", True))
    formal_bridge_enabled = bool(getattr(cfg.model, "virtual_formal_bridge_enabled", False))
    if not enabled or not bridge_enabled or formal_bridge_enabled:
        diag_counters["second_receiver_shadow_bridge_reason_code"] = "BRIDGE_NOT_ALLOWED_BY_CONFIG"
        return

    df = schedule_df.copy()
    if "is_virtual" not in df.columns:
        df["is_virtual"] = False
    real = df[~df["is_virtual"].fillna(False).astype(bool)].copy()
    if real.empty:
        diag_counters["second_receiver_shadow_bridge_reason_code"] = "NO_BRIDGE_CANDIDATE"
        return

    min_ton = float(cfg.rule.campaign_ton_min)
    near_gap = float(getattr(cfg.model, "near_viable_gap_tons", 80.0) or 80.0)
    merge_gap = float(getattr(cfg.model, "merge_candidate_gap_tons", 250.0) or 250.0)
    max_virtual_tons = float(getattr(cfg.model, "shadow_bridge_max_total_virtual_tons_per_campaign", 100.0) or 100.0)
    max_virtual_count = int(getattr(cfg.model, "shadow_bridge_max_virtual_count_per_gap", 5) or 5)
    unit_tons = float(getattr(cfg.model, "virtual_fill_unit_tons_assumption", 20.0) or 20.0)
    max_bridge_tons = max_virtual_tons if max_virtual_tons > 0 else float(max_virtual_count) * max(unit_tons, 0.0)
    selected_key = str(diag_counters.get("second_receiver_selected_campaign", ""))
    selected_line = ""
    selected_campaign = ""
    if "#" in selected_key:
        selected_line, selected_campaign = selected_key.split("#", 1)

    cmeta = (
        real.groupby(["line", "campaign_id"], as_index=False, dropna=False)
        .agg(
            tons=("tons", "sum"),
            first_global=("global_sequence_on_line", "min"),
            last_global=("global_sequence_on_line", "max"),
        )
        .sort_values(["line", "first_global"], kind="mergesort")
        .reset_index(drop=True)
    )
    cmeta["gap"] = cmeta["tons"].map(lambda v: max(0.0, min_ton - float(v or 0.0)))
    cmeta["viability_band"] = cmeta["tons"].map(lambda v: _segment_viability_band(float(v or 0.0), cfg))
    underfilled = cmeta[cmeta["tons"] < min_ton].copy()

    rows: List[Dict[str, Any]] = []
    attempt_count = 0
    success_count = 0
    for _, camp in underfilled.iterrows():
        line = str(camp["line"])
        campaign_id = str(camp["campaign_id"])
        current_tons = float(camp["tons"] or 0.0)
        gap = float(camp["gap"] or 0.0)
        before_band = str(camp["viability_band"])
        line_meta = cmeta[cmeta["line"].astype(str) == line].sort_values("first_global", kind="mergesort").reset_index(drop=True)
        idxs = line_meta.index[line_meta["campaign_id"].astype(str) == campaign_id].tolist()
        is_second = bool(line == selected_line and campaign_id == selected_campaign)
        candidate_count = 0
        best_after_tons = current_tons
        reason = "NO_BRIDGE_CANDIDATE"
        recommended = "NOT_WORTH_RESCUING"
        if not idxs:
            reason = "NO_BRIDGE_CANDIDATE"
        else:
            ridx = int(idxs[0])
            neighbor_idxs = [ridx - 1, ridx + 1, ridx - 2, ridx + 2]
            saw_healthy = False
            saw_nonhealthy_no_gain = False
            for nidx in neighbor_idxs:
                if nidx < 0 or nidx >= len(line_meta):
                    continue
                neighbor = line_meta.iloc[nidx]
                donor_id = str(neighbor["campaign_id"])
                donor_tons = float(neighbor["tons"] or 0.0)
                donor_band = str(neighbor["viability_band"])
                if donor_id == campaign_id:
                    continue
                attempt_count += 1
                if donor_tons >= min_ton:
                    saw_healthy = True
                    continue
                after_tons = current_tons + donor_tons
                after_gap = max(0.0, min_ton - after_tons)
                if after_tons <= current_tons:
                    saw_nonhealthy_no_gain = True
                    continue
                candidate_count += 1
                # Shadow bridge assumes one simple virtual gap can connect this local donor.
                if max_bridge_tons <= 0.0 or max_virtual_count <= 0:
                    reason = "BRIDGE_EXCEEDS_VIRTUAL_CHAIN_LIMIT"
                    continue
                best_after_tons = max(best_after_tons, after_tons)
                after_band = _segment_viability_band(after_tons, cfg)
                if after_gap < gap or after_band != before_band:
                    success_count += 1
                    reason = "BRIDGE_COULD_OPEN_DONOR_PATH"
                    recommended = "TRY_SHADOW_BRIDGE_NEXT"
                    break
                saw_nonhealthy_no_gain = True
            if candidate_count <= 0:
                if saw_healthy:
                    reason = "BRIDGE_ONLY_WITH_MIXED_COMPLEX_PATH"
                    recommended = "REQUIRES_COMPLEX_MIXED_BRIDGE"
                elif saw_nonhealthy_no_gain:
                    reason = "BRIDGE_COULD_NOT_IMPROVE_TON"
                    recommended = "KEEP_AS_IS"
                else:
                    reason = "NO_BRIDGE_CANDIDATE"
                    recommended = "NOT_WORTH_RESCUING"
        gap_after = max(0.0, min_ton - best_after_tons)
        after_band = _segment_viability_band(best_after_tons, cfg)
        if gap_after <= near_gap and reason == "BRIDGE_COULD_OPEN_DONOR_PATH":
            recommended = "TRY_FORMAL_FILL_NEXT"
        elif gap_after <= merge_gap and reason == "BRIDGE_COULD_OPEN_DONOR_PATH":
            recommended = "TRY_SHADOW_BRIDGE_NEXT"
        rows.append(
            {
                "line": line,
                "campaign_id": campaign_id,
                "current_tons": float(round(current_tons, 3)),
                "gap_tons": float(round(gap, 3)),
                "bridge_candidate_count": int(candidate_count),
                "bridge_analysis_attempt_count": int(attempt_count),
                "gap_after_estimate": float(round(gap_after, 3)),
                "viability_band_before": before_band,
                "viability_band_after_estimate": after_band,
                "shadow_bridge_reason_code": reason,
                "recommended_next_action": recommended,
                "is_second_receiver": bool(is_second),
                "second_receiver_selected": bool(is_second),
                "second_receiver_gap_before": float(round(gap, 3)) if is_second else 0.0,
                "second_receiver_gap_after_estimate": float(round(gap_after, 3)) if is_second else 0.0,
            }
        )

    second_rows = [r for r in rows if bool(r.get("is_second_receiver", False))]
    second = second_rows[0] if second_rows else {}
    diag_counters["shadow_bridge_analysis_rows"] = rows
    diag_counters["shadow_bridge_candidate_count"] = int(sum(int(r.get("bridge_candidate_count", 0) or 0) for r in rows))
    diag_counters["shadow_bridge_analysis_attempt_count"] = int(attempt_count)
    diag_counters["shadow_bridge_analysis_success_count"] = int(success_count)
    diag_counters["shadow_bridge_possible_reduced_underfilled_campaigns"] = int(
        sum(1 for r in rows if str(r.get("shadow_bridge_reason_code", "")) == "BRIDGE_COULD_OPEN_DONOR_PATH")
    )
    diag_counters["shadow_bridge_possible_reduced_hard_violations"] = int(diag_counters["shadow_bridge_possible_reduced_underfilled_campaigns"])
    diag_counters["second_receiver_shadow_bridge_gap_before"] = float(second.get("second_receiver_gap_before", 0.0) or 0.0)
    diag_counters["second_receiver_shadow_bridge_gap_after_estimate"] = float(second.get("second_receiver_gap_after_estimate", 0.0) or 0.0)
    diag_counters["second_receiver_shadow_bridge_viability_band_before"] = str(second.get("viability_band_before", ""))
    diag_counters["second_receiver_shadow_bridge_viability_band_after_estimate"] = str(second.get("viability_band_after_estimate", ""))
    diag_counters["second_receiver_shadow_bridge_reason_code"] = str(second.get("shadow_bridge_reason_code", "NO_BRIDGE_CANDIDATE"))


def _local_donor_reallocation_for_second_receiver(
    schedule_df: pd.DataFrame,
    cfg: PlannerConfig,
    orders_by_id: Dict[str, Dict[str, Any]],
    edge_lookup: Dict[tuple[str, str], Dict[str, Any]],
    diag_counters: Dict[str, Any],
) -> pd.DataFrame:
    if schedule_df.empty or not {"line", "campaign_id", "order_id", "tons"}.issubset(schedule_df.columns):
        return schedule_df

    out = _recompute_campaign_sequences(schedule_df)
    if "is_virtual" not in out.columns:
        out["is_virtual"] = False
    if "master_slot" not in out.columns:
        out["master_slot"] = pd.to_numeric(out.get("slot_no", 0), errors="coerce").fillna(0).astype(int)
    for col in [
        "second_receiver_role",
        "donor_reallocated_to_second_receiver",
        "second_receiver_selected",
        "second_receiver_tons_before",
        "second_receiver_tons_after",
        "second_receiver_gap_before",
        "second_receiver_gap_after",
        "second_receiver_viability_band_before",
        "second_receiver_viability_band_after",
        "second_receiver_failure_reason",
        "second_receiver_ready_for_formal_fill",
        "second_receiver_donor_campaign_id",
        "second_receiver_donor_viability_band",
        "second_receiver_donor_rejection_reason",
        "second_receiver_donor_pair_feasible",
        "second_receiver_donor_duplicate_safe",
        "second_receiver_donor_selected",
        "second_receiver_donor_tons",
        "second_receiver_donor_gap_after_estimate",
        "second_receiver_search_space_size",
        "second_receiver_donor_considered_count",
    ]:
        if col not in out.columns:
            if col in {
                "second_receiver_role",
                "second_receiver_viability_band_before",
                "second_receiver_viability_band_after",
                "second_receiver_failure_reason",
                "second_receiver_donor_campaign_id",
                "second_receiver_donor_viability_band",
                "second_receiver_donor_rejection_reason",
            }:
                out[col] = ""
            elif col in {
                "donor_reallocated_to_second_receiver",
                "second_receiver_selected",
                "second_receiver_ready_for_formal_fill",
                "second_receiver_donor_pair_feasible",
                "second_receiver_donor_duplicate_safe",
                "second_receiver_donor_selected",
            }:
                out[col] = False
            else:
                out[col] = 0.0

    real = out[~out["is_virtual"].fillna(False).astype(bool)].copy()
    if real.empty:
        return out

    min_ton = float(cfg.rule.campaign_ton_min)
    max_ton = float(cfg.rule.campaign_ton_max)
    cmeta = (
        real.groupby(["line", "campaign_id"], as_index=False, dropna=False)
        .agg(
            tons=("tons", "sum"),
            first_global=("global_sequence_on_line", "min"),
            last_global=("global_sequence_on_line", "max"),
            master_slot=("master_slot", "first"),
        )
        .sort_values(["line", "first_global"], kind="mergesort")
    )
    cmeta["gap"] = cmeta["tons"].map(lambda v: max(0.0, min_ton - float(v or 0.0)))
    cmeta["viability_band"] = cmeta["tons"].map(lambda v: _segment_viability_band(float(v or 0.0), cfg))
    healthy = cmeta[cmeta["tons"] >= min_ton].copy()
    underfilled = cmeta[cmeta["tons"] < min_ton].copy()

    diag_counters["frozen_healthy_campaign_count"] = int(len(healthy))
    diag_counters["second_receiver_candidate_count"] = int(len(underfilled))
    if healthy.empty:
        diag_counters["second_receiver_failure_reason"] = "NO_HEALTHY_CAMPAIGN_TO_FREEZE"
        return out
    if underfilled.empty:
        diag_counters["second_receiver_failure_reason"] = "NO_UNDERFILLED_CANDIDATE"
        return out

    # Prefer the underfilled campaign that is already closest to being useful.
    candidates = underfilled.sort_values(["tons", "gap"], ascending=[False, True], kind="mergesort").reset_index(drop=True)
    receiver = candidates.iloc[0]
    receiver_line = str(receiver["line"])
    receiver_id = str(receiver["campaign_id"])
    before_tons = float(receiver["tons"] or 0.0)
    before_gap = float(receiver["gap"] or 0.0)
    before_band = str(receiver["viability_band"])
    diag_counters["second_receiver_selected_campaign"] = f"{receiver_line}#{receiver_id}"
    diag_counters["second_receiver_gap_before"] = float(round(before_gap, 3))
    diag_counters["second_receiver_viability_band_before"] = before_band

    receiver_mask = (out["line"].astype(str) == receiver_line) & (out["campaign_id"].astype(str) == receiver_id)
    out.loc[receiver_mask, "second_receiver_role"] = "second_receiver"
    out.loc[receiver_mask, "second_receiver_selected"] = True
    out.loc[receiver_mask, "second_receiver_tons_before"] = before_tons
    out.loc[receiver_mask, "second_receiver_gap_before"] = before_gap
    out.loc[receiver_mask, "second_receiver_viability_band_before"] = before_band

    for _, row in healthy.iterrows():
        healthy_mask = (out["line"].astype(str) == str(row["line"])) & (out["campaign_id"].astype(str) == str(row["campaign_id"]))
        out.loc[healthy_mask, "second_receiver_role"] = "healthy_receiver"

    line_meta = cmeta[cmeta["line"].astype(str) == receiver_line].sort_values("first_global", kind="mergesort").reset_index(drop=True)
    ridx_list = line_meta.index[line_meta["campaign_id"].astype(str) == receiver_id].tolist()
    if not ridx_list:
        diag_counters["second_receiver_failure_reason"] = "SELECTED_RECEIVER_NOT_FOUND_IN_LINE"
        return out
    ridx = int(ridx_list[0])
    receiver_rows_initial = out[
        receiver_mask & (~out["is_virtual"].fillna(False).astype(bool))
    ].sort_values("global_sequence_on_line", kind="mergesort")
    donor_candidates = _enumerate_local_donor_candidates_for_second_receiver(
        schedule_df=out,
        line_meta=line_meta,
        receiver_idx=ridx,
        receiver=receiver,
        receiver_rows=receiver_rows_initial,
        cfg=cfg,
        orders_by_id=orders_by_id,
        edge_lookup=edge_lookup,
    )
    diag_counters["second_receiver_search_space_size"] = int(len(donor_candidates))
    diag_counters["donor_candidates_considered_count"] = int(len(donor_candidates))
    diag_counters["donor_reallocation_attempt_count"] = int(len(donor_candidates))
    diag_counters["donor_candidates_rejected_due_to_line_mismatch"] = 0
    diag_counters["donor_candidates_rejected_due_to_pair_invalid"] = int(sum(1 for c in donor_candidates if c.get("rejection_reason") == "PAIR_INVALID"))
    diag_counters["donor_candidates_rejected_due_to_duplicate"] = int(sum(1 for c in donor_candidates if c.get("rejection_reason") == "DUPLICATE_ORDER"))
    diag_counters["donor_candidates_rejected_due_to_no_ton_gain"] = int(sum(1 for c in donor_candidates if c.get("rejection_reason") in {"NO_TON_GAIN", "NO_BAND_OR_GAP_PROGRESS"}))
    diag_counters["donor_candidates_rejected_due_to_healthy_campaign_protection"] = int(
        sum(1 for c in donor_candidates if c.get("rejection_reason") == "HEALTHY_CAMPAIGN_PROTECTED")
    )
    diag_counters["second_receiver_donor_candidate_rows"] = donor_candidates
    out.loc[receiver_mask, "second_receiver_search_space_size"] = int(len(donor_candidates))
    out.loc[receiver_mask, "second_receiver_donor_considered_count"] = int(len(donor_candidates))
    for cand in donor_candidates:
        donor_id_for_mark = str(cand.get("donor_campaign_id", ""))
        donor_mask_for_mark = (out["line"].astype(str) == receiver_line) & (out["campaign_id"].astype(str) == donor_id_for_mark)
        if not donor_mask_for_mark.any():
            continue
        if str(cand.get("rejection_reason", "")) != "HEALTHY_CAMPAIGN_PROTECTED":
            out.loc[donor_mask_for_mark, "second_receiver_role"] = "donor"
        out.loc[donor_mask_for_mark, "second_receiver_donor_campaign_id"] = donor_id_for_mark
        out.loc[donor_mask_for_mark, "second_receiver_donor_viability_band"] = str(cand.get("donor_viability_band", ""))
        out.loc[donor_mask_for_mark, "second_receiver_donor_rejection_reason"] = str(cand.get("rejection_reason", ""))
        out.loc[donor_mask_for_mark, "second_receiver_donor_pair_feasible"] = bool(cand.get("pair_feasible", False))
        out.loc[donor_mask_for_mark, "second_receiver_donor_duplicate_safe"] = bool(cand.get("duplicate_safe", False))
        out.loc[donor_mask_for_mark, "second_receiver_donor_tons"] = float(cand.get("donor_tons", 0.0) or 0.0)
        out.loc[donor_mask_for_mark, "second_receiver_donor_gap_after_estimate"] = float(cand.get("receiver_gap_after_estimate", 0.0) or 0.0)
        out.loc[donor_mask_for_mark, "second_receiver_search_space_size"] = int(len(donor_candidates))
        out.loc[donor_mask_for_mark, "second_receiver_donor_considered_count"] = int(len(donor_candidates))
    best_attempt_reason = _summarize_second_receiver_failure(donor_candidates)
    changed = False
    for cand in donor_candidates:
        if str(cand.get("rejection_reason", "")) != "ACCEPTABLE":
            continue
        donor_id = str(cand.get("donor_campaign_id", ""))
        donor_rows_meta = line_meta[line_meta["campaign_id"].astype(str) == donor_id]
        if donor_rows_meta.empty:
            continue
        donor = donor_rows_meta.iloc[0]
        donor_tons = float(donor["tons"] or 0.0)
        donor_mask = (out["line"].astype(str) == receiver_line) & (out["campaign_id"].astype(str) == donor_id)
        out.loc[donor_mask, "second_receiver_role"] = "donor"
        receiver_rows = out[
            receiver_mask & (~out["is_virtual"].fillna(False).astype(bool))
        ].sort_values("global_sequence_on_line", kind="mergesort")
        donor_rows = out[
            donor_mask & (~out["is_virtual"].fillna(False).astype(bool))
        ].sort_values("global_sequence_on_line", kind="mergesort")
        if receiver_rows.empty or donor_rows.empty:
            best_attempt_reason = "EMPTY_RECEIVER_OR_DONOR_ROWS"
            continue
        if set(receiver_rows["order_id"].astype(str)) & set(donor_rows["order_id"].astype(str)):
            diag_counters["donor_reallocation_rejected_due_to_duplicate"] = int(diag_counters.get("donor_reallocation_rejected_due_to_duplicate", 0) or 0) + 1
            best_attempt_reason = "DUPLICATE_ORDER"
            continue

        donor_before = float(donor["last_global"] or 0.0) < float(receiver["first_global"] or 0.0)
        if donor_before:
            left_oid = str(donor_rows.iloc[-1]["order_id"])
            right_oid = str(receiver_rows.iloc[0]["order_id"])
        else:
            left_oid = str(receiver_rows.iloc[-1]["order_id"])
            right_oid = str(donor_rows.iloc[0]["order_id"])
        edge = edge_lookup.get((left_oid, right_oid))
        edge_type = str(edge.get("edge_type", "UNRESOLVED")) if edge is not None else "UNRESOLVED"
        ok, _ = _pair_hard_feasible(orders_by_id.get(left_oid, {}), orders_by_id.get(right_oid, {}), cfg, edge_type)
        if not ok:
            diag_counters["donor_reallocation_rejected_due_to_pair_invalid"] = int(diag_counters.get("donor_reallocation_rejected_due_to_pair_invalid", 0) or 0) + 1
            best_attempt_reason = "PAIR_INVALID"
            continue

        after_tons = before_tons + donor_tons
        if after_tons <= before_tons or after_tons > max_ton:
            diag_counters["donor_reallocation_rejected_due_to_no_ton_gain"] = int(diag_counters.get("donor_reallocation_rejected_due_to_no_ton_gain", 0) or 0) + 1
            best_attempt_reason = "NO_TON_GAIN"
            continue

        after_band = _segment_viability_band(after_tons, cfg)
        if after_band == before_band and max(0.0, min_ton - after_tons) >= before_gap:
            diag_counters["donor_reallocation_rejected_due_to_no_ton_gain"] = int(diag_counters.get("donor_reallocation_rejected_due_to_no_ton_gain", 0) or 0) + 1
            best_attempt_reason = "NO_BAND_OR_GAP_PROGRESS"
            continue

        receiver_slot = int(pd.to_numeric(receiver.get("master_slot", 0), errors="coerce") or 0)
        out.loc[donor_mask, "campaign_id"] = receiver_id
        if "campaign_id_hint" in out.columns:
            out.loc[donor_mask, "campaign_id_hint"] = receiver_id
        for slot_col in ["master_slot", "assigned_slot", "slot_no", "campaign_no"]:
            if slot_col in out.columns:
                out.loc[donor_mask, slot_col] = receiver_slot
        out.loc[donor_mask, "donor_reallocated_to_second_receiver"] = True
        out.loc[donor_mask, "second_receiver_role"] = "donor"
        out.loc[donor_mask, "second_receiver_donor_selected"] = True
        cand["selected_for_reallocation"] = True
        receiver_after_mask = (out["line"].astype(str) == receiver_line) & (out["campaign_id"].astype(str) == receiver_id)
        out.loc[receiver_after_mask, "second_receiver_tons_before"] = before_tons
        out.loc[receiver_after_mask, "second_receiver_tons_after"] = after_tons
        out.loc[receiver_after_mask, "second_receiver_gap_before"] = before_gap
        out.loc[receiver_after_mask, "second_receiver_gap_after"] = max(0.0, min_ton - after_tons)
        out.loc[receiver_after_mask, "second_receiver_viability_band_before"] = before_band
        out.loc[receiver_after_mask, "second_receiver_viability_band_after"] = after_band
        out.loc[receiver_after_mask, "second_receiver_ready_for_formal_fill"] = bool(after_band == "NEAR_VIABLE")
        out.loc[receiver_after_mask, "second_receiver_failure_reason"] = ""
        diag_counters["donor_reallocation_success_count"] = int(diag_counters.get("donor_reallocation_success_count", 0) or 0) + 1
        if after_band != before_band:
            diag_counters["second_receiver_promote_success_count"] = int(diag_counters.get("second_receiver_promote_success_count", 0) or 0) + 1
        diag_counters["second_receiver_gap_after"] = float(round(max(0.0, min_ton - after_tons), 3))
        diag_counters["second_receiver_viability_band_after"] = after_band
        diag_counters["second_receiver_ready_for_formal_fill"] = bool(after_band == "NEAR_VIABLE")
        diag_counters["second_receiver_gap_after_reallocation"] = float(round(max(0.0, min_ton - after_tons), 3))
        diag_counters["second_receiver_reallocation_rows"] = [
            {
                "line": receiver_line,
                "campaign_id": receiver_id,
                "donor_campaign_id": donor_id,
                "current_tons_before_reallocation": float(round(before_tons, 3)),
                "current_tons_after_reallocation": float(round(after_tons, 3)),
                "gap_before": float(round(before_gap, 3)),
                "gap_after": float(round(max(0.0, min_ton - after_tons), 3)),
                "viability_band_before": before_band,
                "viability_band_after": after_band,
                "shadow_fill_reason_code_after": "PENDING_VALIDATION",
                "shadow_fill_priority_band_after": "HIGH_PRIORITY_FILL" if after_band == "NEAR_VIABLE" else "NOT_RECOMMENDED",
                "local_reallocation_reason": "REALLOCATED",
            }
        ]
        diag_counters["second_receiver_donor_candidate_rows"] = donor_candidates
        changed = True
        break

    if not changed:
        out.loc[receiver_mask, "second_receiver_tons_after"] = before_tons
        out.loc[receiver_mask, "second_receiver_gap_after"] = before_gap
        out.loc[receiver_mask, "second_receiver_viability_band_after"] = before_band
        out.loc[receiver_mask, "second_receiver_ready_for_formal_fill"] = False
        out.loc[receiver_mask, "second_receiver_failure_reason"] = best_attempt_reason
        diag_counters["second_receiver_gap_after"] = float(round(before_gap, 3))
        diag_counters["second_receiver_viability_band_after"] = before_band
        diag_counters["second_receiver_ready_for_formal_fill"] = False
        diag_counters["second_receiver_gap_after_reallocation"] = float(round(before_gap, 3))
        diag_counters["second_receiver_failure_reason"] = best_attempt_reason
        diag_counters["second_receiver_reallocation_rows"] = [
            {
                "line": receiver_line,
                "campaign_id": receiver_id,
                "donor_campaign_id": "",
                "current_tons_before_reallocation": float(round(before_tons, 3)),
                "current_tons_after_reallocation": float(round(before_tons, 3)),
                "gap_before": float(round(before_gap, 3)),
                "gap_after": float(round(before_gap, 3)),
                "viability_band_before": before_band,
                "viability_band_after": before_band,
                "shadow_fill_reason_code_after": best_attempt_reason,
                "shadow_fill_priority_band_after": "NOT_RECOMMENDED",
            }
        ]

    return _recompute_campaign_sequences(out)


def _append_campaign_viability_diagnostics(schedule_df: pd.DataFrame, cfg: PlannerConfig, diag_counters: Dict[str, Any]) -> None:
    diag_counters.setdefault("near_viable_campaign_count", 0)
    diag_counters.setdefault("merge_candidate_campaign_count", 0)
    diag_counters.setdefault("hopeless_underfilled_campaign_count", 0)
    if schedule_df.empty or not {"line", "campaign_id", "tons"}.issubset(schedule_df.columns):
        return
    work = schedule_df.copy()
    if "is_virtual" not in work.columns:
        work["is_virtual"] = False
    real = work[~work["is_virtual"].fillna(False).astype(bool)].copy()
    if real.empty:
        return
    csum = real.groupby(["line", "campaign_id"], as_index=False, dropna=False)["tons"].sum()
    near = 0
    merge = 0
    hopeless = 0
    for _, row in csum.iterrows():
        band = _segment_viability_band(float(row["tons"] or 0.0), cfg)
        if band == "NEAR_VIABLE":
            near += 1
        elif band == "MERGE_CANDIDATE":
            merge += 1
        elif band == "HOPELESS_UNDERFILLED":
            hopeless += 1
    diag_counters["near_viable_campaign_count"] = int(near)
    diag_counters["merge_candidate_campaign_count"] = int(merge)
    diag_counters["hopeless_underfilled_campaign_count"] = int(hopeless)


def realize_selected_blocks(
    master_result: BlockMasterResult,
    orders_df: pd.DataFrame,
    transition_pack: dict,
    cfg: PlannerConfig,
    random_seed: int = 42,
    assembly_plan: Any | None = None,
) -> BlockRealizationResult:
    """
    Realize selected blocks into a schedule DataFrame.

    Uses campaign_slots from master_result to output real slot semantics.
    """
    rng = random.Random(random_seed)

    tpl_df = transition_pack.get("templates")
    if isinstance(tpl_df, pd.DataFrame) and not tpl_df.empty:
        graph_edges = tpl_df.to_dict("records")
    else:
        graph_edges = []

    cand_graph = transition_pack.get("candidate_graph")
    if cand_graph is not None and hasattr(cand_graph, "edges"):
        cand_edges = cand_graph.edges
        if isinstance(cand_edges, list) and cand_edges:
            graph_edges = cand_edges

    graph_edges = _normalize_edges(graph_edges)
    edge_lookup, edges_by_from = _build_edge_indexes(graph_edges)

    orders_by_id = {str(row.get("order_id", "")): dict(row) for _, row in orders_df.iterrows()}

    schedule_rows = []
    realized_blocks: List[CandidateBlock] = []

    diag_counters = {
        "mixed_bridge_attempt_count": 0,
        "mixed_bridge_success_count": 0,
        "mixed_bridge_reject_count": 0,
        "mixed_bridge_gain_dropped": 0,
        "mixed_bridge_gain_underfilled": 0,
        "block_realized_count": 0,
        "block_realization_failed": 0,
        "orders_in_realized_blocks": 0,
        "failed_block_boundary_count": 0,
        "failed_boundary_pairs": [],
        "realization_failed_slot_ids": [],
        "used_campaign_slots_fallback": False,
        "unresolved_internal_pair_count": 0,
        "unresolved_boundary_pair_count": 0,
        "resolved_internal_pair_count": 0,
        "resolved_boundary_pair_count": 0,
        "unresolved_pair_examples": [],
        "block_ordering_template_lookup_hit_count": 0,
        "block_ordering_template_lookup_miss_count": 0,
        "block_ordering_random_fallback_count": 0,
        "template_lookup_hit_count": 0,
        "template_lookup_miss_count": 0,
        "segment_split_attempt_count": 0,
        "segment_split_kept_count": 0,
        "segment_split_prevented_due_to_min_ton_count": 0,
        "segment_split_prevented_tons_examples": [],
        "deferred_segment_count": 0,
        "deferred_segment_finalized_count": 0,
        "segment_remerge_attempt_count": 0,
        "segment_remerge_success_count": 0,
        "segment_remerge_rejected_due_to_pair_invalid": 0,
        "segment_remerge_rejected_due_to_duplicate": 0,
        "segment_remerge_rejected_due_to_ton_no_gain": 0,
        "near_viable_campaign_count": 0,
        "merge_candidate_campaign_count": 0,
        "hopeless_underfilled_campaign_count": 0,
        "deferred_near_viable_count": 0,
        "deferred_hopeless_count": 0,
        "near_viable_preserved_count": 0,
        "near_viable_lost_count": 0,
        "near_viable_promote_attempt_count": 0,
        "near_viable_promote_success_count": 0,
        "near_viable_promote_rejected_due_to_pair_invalid": 0,
        "near_viable_promote_rejected_due_to_duplicate": 0,
        "near_viable_promote_rejected_due_to_no_ton_gain": 0,
        "hopeless_segments_dropped_for_near_viable_count": 0,
        "near_viable_promote_rows": [],
        "formal_virtual_fill_trial_count": 0,
        "formal_virtual_fill_success_count": 0,
        "formal_virtual_fill_rollback_count": 0,
        "formal_virtual_fill_success_campaigns": [],
        "formal_virtual_fill_rollback_reasons": [],
        "formal_virtual_fill_trial_rows": [],
        "second_receiver_candidate_count": 0,
        "second_receiver_selected_campaign": "",
        "second_receiver_gap_before": 0.0,
        "second_receiver_gap_after": 0.0,
        "second_receiver_viability_band_before": "",
        "second_receiver_viability_band_after": "",
        "second_receiver_promote_success_count": 0,
        "donor_reallocation_attempt_count": 0,
        "donor_reallocation_success_count": 0,
        "donor_reallocation_rejected_due_to_pair_invalid": 0,
        "donor_reallocation_rejected_due_to_duplicate": 0,
        "donor_reallocation_rejected_due_to_no_ton_gain": 0,
        "second_receiver_search_space_size": 0,
        "donor_candidates_considered_count": 0,
        "donor_candidates_rejected_due_to_line_mismatch": 0,
        "donor_candidates_rejected_due_to_pair_invalid": 0,
        "donor_candidates_rejected_due_to_duplicate": 0,
        "donor_candidates_rejected_due_to_no_ton_gain": 0,
        "donor_candidates_rejected_due_to_healthy_campaign_protection": 0,
        "frozen_healthy_campaign_count": 0,
        "second_receiver_ready_for_formal_fill": False,
        "second_receiver_gap_after_reallocation": 0.0,
        "second_receiver_failure_reason": "",
        "second_receiver_reallocation_rows": [],
        "second_receiver_donor_candidate_rows": [],
        "shadow_bridge_candidate_count": 0,
        "shadow_bridge_analysis_attempt_count": 0,
        "shadow_bridge_analysis_success_count": 0,
        "shadow_bridge_possible_reduced_underfilled_campaigns": 0,
        "shadow_bridge_possible_reduced_hard_violations": 0,
        "second_receiver_shadow_bridge_gap_before": 0.0,
        "second_receiver_shadow_bridge_gap_after_estimate": 0.0,
        "second_receiver_shadow_bridge_viability_band_before": "",
        "second_receiver_shadow_bridge_viability_band_after_estimate": "",
        "second_receiver_shadow_bridge_reason_code": "",
        "shadow_bridge_analysis_rows": [],
    }

    if assembly_plan is not None:
        planner_summary = dict(getattr(assembly_plan, "summary", {}) or {})
        planner_diag = dict(getattr(assembly_plan, "diagnostics", {}) or {})
        for key, value in planner_summary.items():
            if key not in diag_counters:
                diag_counters[key] = value
        if planner_diag.get("campaign_assembly_plan_rows"):
            diag_counters["campaign_assembly_plan_rows"] = list(planner_diag.get("campaign_assembly_plan_rows", []))
        if planner_diag.get("bridge_requirement_rows"):
            diag_counters["bridge_requirement_rows"] = list(planner_diag.get("bridge_requirement_rows", []))
        if planner_diag.get("donor_candidate_rows"):
            diag_counters["donor_candidate_rows"] = list(planner_diag.get("donor_candidate_rows", []))

    slots_to_process = master_result.campaign_slots
    if not slots_to_process:
        diag_counters["used_campaign_slots_fallback"] = True
        slots_to_process = []
        for line, b_ids in master_result.block_order_by_line.items():
            slots_to_process.append(
                BlockCampaignSlot(
                    line=line,
                    slot_no=1,
                    campaign_id=f"{line}__slot_1_fb",
                    block_ids=b_ids,
                    total_tons=0,
                    gap_to_min_tons=0,
                    remaining_to_max_tons=0,
                    is_underfilled=False,
                    head_block_id=b_ids[0] if b_ids else "",
                    tail_block_id=b_ids[-1] if b_ids else "",
                )
            )

    block_obj_by_id = {b.block_id: b for b in master_result.selected_blocks}

    global_sequence_on_line = {"big_roll": 0, "small_roll": 0}
    seen_global_orders: set[str] = set()
    deferred_segments: List[Dict[str, Any]] = []

    for slot in slots_to_process:
        sequence_in_slot = 0
        slot_failed = False
        prev_block_tail_id = None
        campaign_segment_no = 1
        deferred_segment_no = 0
        active_campaign_id = slot.campaign_id
        current_segment_rows: List[Dict[str, Any]] = []
        slot_segment_tons: List[float] = []

        for block_in_slot_index, block_id in enumerate(slot.block_ids, start=1):
            if block_id not in block_obj_by_id:
                continue
            block = block_obj_by_id[block_id]
            ordered_ids = _order_within_block(block, edge_lookup, edges_by_from, orders_by_id, cfg, rng, diag_counters)
            ordered_ids = [oid for oid in _dedupe_preserve(ordered_ids) if oid not in seen_global_orders]

            if not ordered_ids:
                diag_counters["block_realization_failed"] += 1
                slot_failed = True
                continue

            boundary_edge_type = None
            inter_block_boundary_before = False
            boundary_from_block_id = None
            boundary_to_block_id = None

            if prev_block_tail_id is not None:
                # Try to attach this block to the previous tail; if not possible, start a new segment
                # inside the same slot instead of forcing an illegal boundary.
                attach_index = None
                attach_edge_type = None
                for idx_attach, cand_head in enumerate(ordered_ids):
                    edge_rec = edge_lookup.get((prev_block_tail_id, cand_head))
                    edge_type_cand = str(edge_rec.get("edge_type", "UNRESOLVED")) if edge_rec is not None else "UNRESOLVED"
                    ok, _ = _pair_hard_feasible(orders_by_id.get(prev_block_tail_id, {}), orders_by_id.get(cand_head, {}), cfg, edge_type_cand)
                    if edge_rec is not None and ok:
                        attach_index = idx_attach
                        attach_edge_type = edge_type_cand
                        break
                if attach_index is not None:
                    ordered_ids = ordered_ids[attach_index:]
                    boundary_edge_type = attach_edge_type
                    inter_block_boundary_before = True
                    boundary_from_block_id = slot.block_ids[block_in_slot_index - 2]
                    boundary_to_block_id = block.block_id
                    diag_counters["resolved_boundary_pair_count"] += 1
                    diag_counters["template_lookup_hit_count"] += 1
                else:
                    diag_counters["failed_block_boundary_count"] += 1
                    diag_counters["unresolved_boundary_pair_count"] += 1
                    diag_counters["template_lookup_miss_count"] += 1
                    diag_counters["failed_boundary_pairs"].append(f"{prev_block_tail_id}->{ordered_ids[0]}")
                    _append_limited_example(
                        diag_counters,
                        "unresolved_pair_examples",
                        {
                            "scope": "boundary",
                            "line": slot.line,
                            "campaign_id": active_campaign_id,
                            "from_order_id": prev_block_tail_id,
                            "to_order_id": ordered_ids[0],
                            "from_block_id": slot.block_ids[block_in_slot_index - 2],
                            "to_block_id": block.block_id,
                        },
                    )
                    diag_counters["segment_split_attempt_count"] = int(diag_counters.get("segment_split_attempt_count", 0) or 0) + 1

                    viability = _evaluate_segment_min_ton_viability(
                        current_segment_rows,
                        cfg,
                        {
                            "line": slot.line,
                            "original_campaign_id": slot.campaign_id,
                        },
                    )
                    segment_total_tons = float(viability.get("segment_total_tons", 0.0) or 0.0)
                    viability_band = str(viability.get("viability_band", "HOPELESS_UNDERFILLED"))
                    allow_split = bool(viability.get("can_standalone_campaign", False)) and not bool(
                        viability.get("should_prevent_split", False)
                    )

                    if allow_split:
                        diag_counters["segment_split_kept_count"] = int(diag_counters.get("segment_split_kept_count", 0) or 0) + 1
                        slot_segment_tons.append(segment_total_tons)
                        campaign_segment_no += 1
                        active_campaign_id = f"{slot.campaign_id}__seg_{campaign_segment_no}"
                        sequence_in_slot = 0
                        prev_block_tail_id = None
                        boundary_edge_type = None
                        inter_block_boundary_before = False
                        current_segment_rows = []
                    else:
                        diag_counters["segment_split_prevented_due_to_min_ton_count"] = int(
                            diag_counters.get("segment_split_prevented_due_to_min_ton_count", 0) or 0
                        ) + 1
                        _append_limited_example(
                            diag_counters,
                            "segment_split_prevented_tons_examples",
                            {
                                "line": slot.line,
                                "original_campaign_id": slot.campaign_id,
                                "active_campaign_id": active_campaign_id,
                                "segment_total_tons": float(segment_total_tons),
                                "campaign_ton_min": float(cfg.rule.campaign_ton_min),
                                "reason_code": str(viability.get("reason_code", "SEGMENT_BELOW_MIN_TON")),
                                "viability_band": viability_band,
                            },
                            limit=10,
                        )
                        if viability_band == "NEAR_VIABLE":
                            diag_counters["near_viable_campaign_count"] = int(diag_counters.get("near_viable_campaign_count", 0) or 0) + 1
                        if viability_band == "HOPELESS_UNDERFILLED":
                            diag_counters["hopeless_underfilled_campaign_count"] = int(diag_counters.get("hopeless_underfilled_campaign_count", 0) or 0) + 1
                        # Defer this underfilled split candidate: keep orders for later finalization
                        # instead of dropping them immediately.
                        deferred_segment_no += 1
                        active_campaign_id = f"{slot.campaign_id}__deferred_{deferred_segment_no}"
                        _defer_underfilled_segment(
                            deferred_segments,
                            line=slot.line,
                            original_campaign_id=slot.campaign_id,
                            deferred_campaign_id=active_campaign_id,
                            slot_no=int(slot.slot_no),
                            segment_total_tons=segment_total_tons,
                            reason_code=str(viability.get("reason_code", "SEGMENT_BELOW_MIN_TON")),
                            viability_band=viability_band,
                        )
                        sequence_in_slot = 0
                        prev_block_tail_id = None
                        boundary_edge_type = None
                        inter_block_boundary_before = False
                        current_segment_rows = []

            mixed_candidates = _generate_mixed_bridge_candidates(
                block=block,
                ordered_ids=ordered_ids,
                edge_lookup=edge_lookup,
                edges_by_from=edges_by_from,
                orders_by_id=orders_by_id,
                cfg=cfg,
                rng=rng,
            )

            diag_counters["mixed_bridge_attempt_count"] += len(mixed_candidates)

            applied_mixed: Dict[str, str] = {}
            for mc in mixed_candidates:
                score = float(mc.get("score", 0))
                if score > 0:
                    applied_mixed[f"{mc['from_order_id']}->{mc['to_order_id']}"] = str(mc.get("mixed_form", ""))

            for seq_pos, oid in enumerate(ordered_ids):
                order = orders_by_id.get(oid)
                if order is None:
                    continue

                edge_type = "DIRECT_EDGE"
                mixed_applied = False

                if oid in seen_global_orders:
                    diag_counters["duplicate_order_skip_count"] = int(diag_counters.get("duplicate_order_skip_count", 0) or 0) + 1
                    continue

                if seq_pos == 0 and inter_block_boundary_before:
                    edge_type = boundary_edge_type
                    prev_for_check = prev_block_tail_id
                elif seq_pos > 0:
                    # Use the last *scheduled* order in this campaign rather than the raw previous entry.
                    prev_for_check = prev_block_tail_id
                    if prev_for_check is not None:
                        edge_key = f"{prev_for_check}->{oid}"
                        if edge_key in applied_mixed:
                            edge_type = applied_mixed[edge_key]
                            mixed_applied = True
                        else:
                            edge_rec = edge_lookup.get((prev_for_check, oid))
                            if edge_rec is not None:
                                edge_type = str(edge_rec.get("edge_type", "UNRESOLVED"))
                                diag_counters["resolved_internal_pair_count"] += 1
                                diag_counters["template_lookup_hit_count"] += 1
                            else:
                                edge_type = "UNRESOLVED"
                    else:
                        prev_for_check = None
                else:
                    prev_for_check = None

                if prev_for_check is not None:
                    ok, reasons = _pair_hard_feasible(orders_by_id.get(prev_for_check, {}), order, cfg, edge_type)
                    if edge_type == "UNRESOLVED":
                        ok = False
                        reasons = list(reasons) + ["UNRESOLVED"]
                    if not ok:
                        diag_counters["invalid_pair_drop_count"] = int(diag_counters.get("invalid_pair_drop_count", 0) or 0) + 1
                        for r in reasons:
                            key = f"invalid_pair_reason__{r}"
                            diag_counters[key] = int(diag_counters.get(key, 0) or 0) + 1
                        _append_limited_example(
                            diag_counters,
                            "unresolved_pair_examples",
                            {
                                "scope": "pair_rejected",
                                "line": slot.line,
                                "campaign_id": active_campaign_id,
                                "block_id": block.block_id,
                                "from_order_id": prev_for_check,
                                "to_order_id": oid,
                                "edge_type": edge_type,
                                "reasons": reasons,
                            },
                        )
                        continue

                line = slot.line
                global_sequence_on_line[line] += 1
                sequence_in_slot += 1

                row = {
                    "order_id": oid,
                    "source_order_id": str(order.get("source_order_id", oid)),
                    "parent_order_id": str(order.get("parent_order_id", oid)),
                    "lot_id": str(order.get("lot_id", oid)),

                    # block-first 原始语义列
                    "assigned_line": slot.line,
                    "assigned_slot": slot.slot_no,
                    "slot_no": slot.slot_no,
                    "master_slot": slot.slot_no,
                    "campaign_no": slot.slot_no,
                    "campaign_id": active_campaign_id,
                    "campaign_id_hint": active_campaign_id,
                    "block_id": block.block_id,
                    "block_in_slot_index": block_in_slot_index,
                    "sequence_in_block": seq_pos + 1,
                    "sequence_in_slot": sequence_in_slot,
                    "global_sequence_on_line": global_sequence_on_line[line],
                    "selected_edge_type": edge_type,

                    # validator / exporter 兼容别名
                    "line": slot.line,
                    "line_name": _line_name(slot.line),
                    "campaign_seq": sequence_in_slot,
                    "campaign_real_seq": sequence_in_slot,

                    "tons": float(order.get("tons", 0)),
                    "width": float(order.get("width", 0)),
                    "thickness": float(order.get("thickness", 0)),
                    "grade": str(order.get("grade", "")),
                    "steel_group": str(order.get("steel_group", "")),
                    "temp_min": float(order.get("temp_min", 0)),
                    "temp_max": float(order.get("temp_max", 0)),
                    "temp_mean": float(order.get("temp_mean", (float(order.get("temp_min", 0)) + float(order.get("temp_max", 0))) / 2.0)),
                    "tons": float(order.get("tons", 0)),
                    "backlog": float(order.get("backlog", 0) or 0),
                    "due_date": order.get("due_date"),
                    "due_bucket": order.get("due_bucket", ""),
                    "mixed_bridge_applied": mixed_applied,
                    "is_virtual": False,
                    "inter_block_boundary_before": inter_block_boundary_before if seq_pos == 0 else False,
                    "boundary_from_block_id": boundary_from_block_id if seq_pos == 0 else None,
                    "boundary_to_block_id": boundary_to_block_id if seq_pos == 0 else None,
                    "boundary_edge_type": boundary_edge_type if seq_pos == 0 else None,
                }
                schedule_rows.append(row)
                current_segment_rows.append(row)
                seen_global_orders.add(oid)
                prev_block_tail_id = oid

            kept_ids = [oid for oid in ordered_ids if oid in seen_global_orders]
            if kept_ids:
                diag_counters["mixed_bridge_success_count"] += len(applied_mixed)
                diag_counters["block_realized_count"] += 1
                diag_counters["orders_in_realized_blocks"] += len(kept_ids)
                block.is_realized = True
                block.scheduled_order_ids = list(kept_ids)
                block.real_order_ids = list(kept_ids)
                realized_blocks.append(block)
                prev_block_tail_id = kept_ids[-1]
            else:
                diag_counters["block_realization_failed"] += 1

        if slot_failed:
            diag_counters["realization_failed_slot_ids"].append(slot.campaign_id)
        if current_segment_rows:
            slot_segment_tons.append(_segment_real_tons(current_segment_rows))

    schedule_df = pd.DataFrame(schedule_rows) if schedule_rows else pd.DataFrame()
    diag_counters["deferred_segment_count"] = int(len(deferred_segments))
    if not schedule_df.empty:
        if "is_virtual" not in schedule_df.columns:
            schedule_df["is_virtual"] = False
        if "line" not in schedule_df.columns and "assigned_line" in schedule_df.columns:
            schedule_df["line"] = schedule_df["assigned_line"]
        if "master_slot" not in schedule_df.columns and "assigned_slot" in schedule_df.columns:
            schedule_df["master_slot"] = pd.to_numeric(schedule_df["assigned_slot"], errors="coerce").fillna(0).astype(int)
        if "slot_no" not in schedule_df.columns and "master_slot" in schedule_df.columns:
            schedule_df["slot_no"] = pd.to_numeric(schedule_df["master_slot"], errors="coerce").fillna(0).astype(int)
        if "campaign_no" not in schedule_df.columns and "master_slot" in schedule_df.columns:
            schedule_df["campaign_no"] = pd.to_numeric(schedule_df["master_slot"], errors="coerce").fillna(0).astype(int)
        if "campaign_seq" not in schedule_df.columns:
            if "sequence_in_slot" in schedule_df.columns:
                schedule_df["campaign_seq"] = schedule_df["sequence_in_slot"]
            else:
                schedule_df["campaign_seq"] = range(1, len(schedule_df) + 1)
        if "campaign_real_seq" not in schedule_df.columns:
            schedule_df["campaign_real_seq"] = schedule_df["campaign_seq"]

        sort_cols = [c for c in ["line", "master_slot", "campaign_seq", "sequence_in_block"] if c in schedule_df.columns]
        if sort_cols:
            schedule_df = schedule_df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
        schedule_df["global_seq"] = range(1, len(schedule_df) + 1)
        if "line" in schedule_df.columns:
            schedule_df["line_seq"] = schedule_df.groupby("line", sort=False).cumcount() + 1
        else:
            schedule_df["line_seq"] = range(1, len(schedule_df) + 1)
        schedule_df = _finalize_deferred_segments(
            schedule_df=schedule_df,
            deferred_segments=deferred_segments,
            cfg=cfg,
            orders_by_id=orders_by_id,
            edge_lookup=edge_lookup,
            diag_counters=diag_counters,
        )
        schedule_df = _local_reassemble_for_near_viable(
            schedule_df=schedule_df,
            cfg=cfg,
            orders_by_id=orders_by_id,
            edge_lookup=edge_lookup,
            diag_counters=diag_counters,
        )
        schedule_df = _local_donor_reallocation_for_second_receiver(
            schedule_df=schedule_df,
            cfg=cfg,
            orders_by_id=orders_by_id,
            edge_lookup=edge_lookup,
            diag_counters=diag_counters,
        )
        _shadow_bridge_analysis_for_underfilled_campaigns(schedule_df, cfg, diag_counters)
        schedule_df = _run_formal_virtual_fill_trial(schedule_df, cfg, diag_counters)
        _append_campaign_viability_diagnostics(schedule_df, cfg, diag_counters)

    if not schedule_df.empty and assembly_plan is not None:
        skeleton_rows = list(getattr(assembly_plan, "skeleton_rows", lambda: [])())
        if skeleton_rows:
            sk_df = pd.DataFrame(skeleton_rows)
            if not sk_df.empty:
                merge_cols = [c for c in ["line", "campaign_id", "viability_band", "assembly_status", "bridge_need_level", "bridge_type_needed", "donor_need_level"] if c in sk_df.columns or c == "campaign_id"]
                if "campaign_id" not in sk_df.columns and "skeleton_id" in sk_df.columns:
                    sk_df = sk_df.rename(columns={"skeleton_id": "campaign_id"})
                keep_cols = [c for c in ["line", "campaign_id", "viability_band", "assembly_status", "bridge_need_level", "bridge_type_needed", "donor_need_level"] if c in sk_df.columns]
                if {"line", "campaign_id"}.issubset(sk_df.columns):
                    schedule_df = schedule_df.merge(sk_df[keep_cols].drop_duplicates(), on=["line", "campaign_id"], how="left")

    dropped_order_ids_set = set(master_result.dropped_order_ids)

    if not schedule_df.empty:
        scheduled_set = set(schedule_df["order_id"])
        additional_drops = [oid for oid in master_result.selected_order_ids if oid not in scheduled_set]
        dropped_order_ids_set.update(additional_drops)

    dropped = [
        orders_by_id[oid]
        for oid in dropped_order_ids_set
        if oid in orders_by_id
    ]
    dropped_df = pd.DataFrame(dropped) if dropped else pd.DataFrame()

    if realized_blocks:
        avg_quality = sum(b.block_quality_score for b in realized_blocks) / len(realized_blocks)
        diag_counters["avg_realized_block_quality"] = avg_quality
        diag_counters["avg_realized_block_tons"] = (
            sum(b.total_tons for b in realized_blocks) / len(realized_blocks)
        )
    diag_counters["unresolved_pair_count_total"] = int(diag_counters.get("unresolved_internal_pair_count", 0) or 0) + int(diag_counters.get("unresolved_boundary_pair_count", 0) or 0)
    diag_counters["resolved_pair_count_total"] = int(diag_counters.get("resolved_internal_pair_count", 0) or 0) + int(diag_counters.get("resolved_boundary_pair_count", 0) or 0)
    deferred_near = int(diag_counters.get("deferred_near_viable_count", 0) or 0)
    preserved_near = int(diag_counters.get("near_viable_preserved_count", 0) or 0)
    diag_counters["near_viable_lost_count"] = int(max(0, deferred_near - preserved_near))

    return BlockRealizationResult(
        realized_blocks=realized_blocks,
        realized_schedule_df=schedule_df,
        realized_dropped_df=dropped_df,
        block_realization_diag=diag_counters,
    )
