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


def realize_selected_blocks(
    master_result: BlockMasterResult,
    orders_df: pd.DataFrame,
    transition_pack: dict,
    cfg: PlannerConfig,
    random_seed: int = 42,
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
    }

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

    for slot in slots_to_process:
        sequence_in_slot = 0
        slot_failed = False
        prev_block_tail_id = None
        campaign_segment_no = 1
        active_campaign_id = slot.campaign_id

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
                    campaign_segment_no += 1
                    active_campaign_id = f"{slot.campaign_id}__seg_{campaign_segment_no}"
                    sequence_in_slot = 0
                    prev_block_tail_id = None
                    boundary_edge_type = None
                    inter_block_boundary_before = False

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

    schedule_df = pd.DataFrame(schedule_rows) if schedule_rows else pd.DataFrame()
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

    return BlockRealizationResult(
        realized_blocks=realized_blocks,
        realized_schedule_df=schedule_df,
        realized_dropped_df=dropped_df,
        block_realization_diag=diag_counters,
    )
