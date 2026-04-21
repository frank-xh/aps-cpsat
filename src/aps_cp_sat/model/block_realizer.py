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
from aps_cp_sat.model.block_types import CandidateBlock, CandidateBlockPool
from aps_cp_sat.model.block_master import BlockMasterResult


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


def _order_within_block(
    block: CandidateBlock,
    graph_edges: List[Dict[str, Any]],
    orders_by_id: Dict[str, Dict[str, Any]],
    rng: random.Random,
) -> List[str]:
    """
    Order the orders within a block using internal edge constraints.
    If block already has an internal_edges list, follow that order.
    Otherwise, find the best topological ordering.
    """
    if block.internal_edges and len(block.internal_edges) >= block.order_count - 1:
        # Block was built with internal edges — use that order
        ordered = [block.order_ids[0]]
        remaining = set(block.order_ids[1:])
        edges_by_from: Dict[str, List[Dict]] = {}
        for e in block.internal_edges:
            frm = str(e.get("from_order_id", ""))
            edges_by_from.setdefault(frm, []).append(e)

        current = ordered[0]
        for _ in range(len(block.order_ids) - 1):
            next_edges = edges_by_from.get(current, [])
            found = False
            for ne in sorted(next_edges, key=lambda x: float(x.get("score", 0)), reverse=True):
                nxt = str(ne.get("to_order_id", ""))
                if nxt in remaining:
                    ordered.append(nxt)
                    remaining.discard(nxt)
                    current = nxt
                    found = True
                    break
            if not found:
                # Fall back: pick any remaining
                if remaining:
                    nxt = remaining.pop()
                    ordered.append(nxt)
                    current = nxt
                else:
                    break
        return ordered

    # No internal edges — fall back to original order_ids
    return list(block.order_ids)


def _generate_mixed_bridge_candidates(
    block: CandidateBlock,
    ordered_ids: List[str],
    graph_edges: List[Dict[str, Any]],
    orders_by_id: Dict[str, Dict[str, Any]],
    cfg: PlannerConfig,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    """
    Generate mixed bridge candidates INSIDE a block.

    Mixed bridge is only attempted at hotspots:
    - underfill_hotspot
    - group_switch_hotspot
    - bridge_dependency_hotspot
    - width_tension_hotspot

    Returns list of (from_oid, to_oid, mixed_form, score) tuples.
    Only two forms are supported:
    - REAL_TO_GUARDED: from has REAL_BRIDGE_EDGE, to needs GUARDED to connect
    - GUARDED_TO_REAL: from has GUARDED_FAMILY, to transitions to REAL
    """
    if not getattr(cfg.model, "mixed_bridge_in_block_enabled", True):
        return []

    allowed_forms = getattr(cfg.model, "mixed_bridge_allowed_forms", ["REAL_TO_GUARDED", "GUARDED_TO_REAL"])
    allowed_hotspots = getattr(cfg.model, "mixed_bridge_allowed_hotspots", [])
    max_attempts = int(getattr(cfg.model, "mixed_bridge_max_attempts_per_block", 10))

    # Only attempt if block has a hotspot
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

    candidates = []

    # Build edge lookup
    edges_by_from: Dict[str, List[Dict]] = {}
    for e in graph_edges:
        edges_by_from.setdefault(str(e.get("from_order_id", "")), []).append(e)

    # Try each pair of consecutive orders in block
    attempts = 0
    for i, from_oid in enumerate(ordered_ids):
        if attempts >= max_attempts:
            break
        to_oid = ordered_ids[i + 1] if i + 1 < len(ordered_ids) else None
        if to_oid is None:
            break

        from_order = orders_by_id.get(from_oid, {})
        to_order = orders_by_id.get(to_oid, {})

        if not from_order or not to_order:
            continue

        # Check if there's already a direct or real bridge edge
        existing_edges = edges_by_from.get(from_oid, [])
        has_direct = any(str(e.get("edge_type", "")) in {"DIRECT_EDGE"} for e in existing_edges)
        has_real = any(str(e.get("edge_type", "")) in {"REAL_BRIDGE_EDGE"} for e in existing_edges)
        has_guarded = any(str(e.get("edge_type", "")) in {"VIRTUAL_BRIDGE_FAMILY_EDGE"} for e in existing_edges)

        from_sg = str(from_order.get("steel_group", ""))
        to_sg = str(to_order.get("steel_group", ""))
        group_switch = (from_sg != to_sg and from_sg and to_sg)

        # REAL_TO_GUARDED: existing is REAL but to_oid doesn't have good transition
        if has_real and "REAL_TO_GUARDED" in allowed_forms:
            if group_switch or block.has_underfill_hotspot:
                # Check if there's a guarded family path
                guarded_edges = [
                    e for e in existing_edges
                    if str(e.get("edge_type", "")) == "VIRTUAL_BRIDGE_FAMILY_EDGE"
                ]
                if guarded_edges:
                    # Use guarded edge instead of real bridge
                    candidates.append({
                        "from_order_id": from_oid,
                        "to_order_id": to_oid,
                        "mixed_form": "REAL_TO_GUARDED",
                        "edge_type": "VIRTUAL_BRIDGE_FAMILY_EDGE",
                        "score": float(guarded_edges[0].get("score", 0)) * 0.9,
                        "reason": "group_switch_or_underfill",
                    })
                    attempts += 1

        # GUARDED_TO_REAL: existing is GUARDED but transitioning to REAL
        if has_guarded and "GUARDED_TO_REAL" in allowed_forms:
            if group_switch or block.has_width_tension:
                real_edges = [
                    e for e in existing_edges
                    if str(e.get("edge_type", "")) == "REAL_BRIDGE_EDGE"
                ]
                direct_edges = [
                    e for e in existing_edges
                    if str(e.get("edge_type", "")) == "DIRECT_EDGE"
                ]
                if real_edges or direct_edges:
                    candidates.append({
                        "from_order_id": from_oid,
                        "to_order_id": to_oid,
                        "mixed_form": "GUARDED_TO_REAL",
                        "edge_type": "REAL_BRIDGE_EDGE" if real_edges else "DIRECT_EDGE",
                        "score": max(
                            float(real_edges[0].get("score", 0)) if real_edges else 0,
                            float(direct_edges[0].get("score", 0)) if direct_edges else 0,
                        ) * 0.85,
                        "reason": "group_switch_or_width_tension",
                    })
                    attempts += 1

    return candidates[:max_attempts]


def realize_selected_blocks(
    master_result: BlockMasterResult,
    orders_df: pd.DataFrame,
    transition_pack: dict,
    cfg: PlannerConfig,
    random_seed: int = 42,
) -> BlockRealizationResult:
    """
    Realize selected blocks into a schedule DataFrame.

    For each selected block:
    1. Order orders within the block using internal edges
    2. Generate mixed bridge candidates at hotspots (block-internal only)
    3. Choose the best realization
    4. Build schedule rows

    Parameters
    ----------
    master_result: Result from solve_block_master
    orders_df: Full orders DataFrame
    transition_pack: Template transition pack (for graph edges)
    cfg: PlannerConfig
    random_seed: Random seed

    Returns
    -------
    BlockRealizationResult with realized schedule DataFrame
    """
    rng = random.Random(random_seed)

    # Get graph edges
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

    orders_by_id = {str(row.get("order_id", "")): dict(row) for _, row in orders_df.iterrows()}

    schedule_rows = []
    realized_blocks: List[CandidateBlock] = []

    # Counters for diagnostics
    diag_counters = {
        "mixed_bridge_attempt_count": 0,
        "mixed_bridge_success_count": 0,
        "mixed_bridge_reject_count": 0,
        "mixed_bridge_gain_dropped": 0,
        "mixed_bridge_gain_underfilled": 0,
        "block_realized_count": 0,
        "block_realization_failed": 0,
        "orders_in_realized_blocks": 0,
    }

    # Orders already in selected blocks
    selected_order_set = set(master_result.selected_order_ids)

    # Process each selected block
    for block in master_result.selected_blocks:
        ordered_ids = _order_within_block(block, graph_edges, orders_by_id, rng)

        # Generate mixed bridge candidates
        mixed_candidates = _generate_mixed_bridge_candidates(
            block=block,
            ordered_ids=ordered_ids,
            graph_edges=graph_edges,
            orders_by_id=orders_by_id,
            cfg=cfg,
            rng=rng,
        )

        diag_counters["mixed_bridge_attempt_count"] += len(mixed_candidates)

        # Apply mixed bridge candidates (accept all with positive score)
        applied_mixed: Dict[str, str] = {}  # from -> to with mixed type
        for mc in mixed_candidates:
            score = float(mc.get("score", 0))
            if score > 0:
                applied_mixed[f"{mc['from_order_id']}->{mc['to_order_id']}"] = str(mc.get("mixed_form", ""))

        # Build block-level schedule rows
        block_seq = 0
        block_dropped = []

        for seq_pos, oid in enumerate(ordered_ids):
            order = orders_by_id.get(oid)
            if order is None:
                block_dropped.append(oid)
                continue

            # Determine edge type for this position
            edge_type = "DIRECT_EDGE"
            edge_key = ""
            mixed_applied = False
            if seq_pos > 0:
                edge_key = f"{ordered_ids[seq_pos - 1]}->{oid}"
                if edge_key in applied_mixed:
                    edge_type = applied_mixed[edge_key]
                    mixed_applied = True
                else:
                    # Check original edges
                    for e in graph_edges:
                        if (str(e.get("from_order_id", "")) == ordered_ids[seq_pos - 1]
                           and str(e.get("to_order_id", "")) == oid):
                            edge_type = str(e.get("edge_type", "DIRECT_EDGE"))
                            break

            row = {
                "order_id": oid,
                "sequence_in_block": seq_pos,
                "block_id": block.block_id,
                "block_line": block.line,
                "block_seq": block_seq,
                "tons": float(order.get("tons", 0)),
                "width": float(order.get("width", 0)),
                "thickness": float(order.get("thickness", 0)),
                "steel_group": str(order.get("steel_group", "")),
                "temp_min": float(order.get("temp_min", 0)),
                "temp_max": float(order.get("temp_max", 0)),
                "edge_type": edge_type,
                "mixed_bridge_applied": mixed_applied,
            }
            schedule_rows.append(row)
            block_seq += 1

        if len(ordered_ids) > 0:
            diag_counters["mixed_bridge_success_count"] += len(applied_mixed)
            diag_counters["block_realized_count"] += 1
            diag_counters["orders_in_realized_blocks"] += len(ordered_ids)
            block.is_realized = True
            block.scheduled_order_ids = list(ordered_ids)
            realized_blocks.append(block)
        else:
            diag_counters["block_realization_failed"] += 1

    # Build DataFrames
    schedule_df = pd.DataFrame(schedule_rows) if schedule_rows else pd.DataFrame()

    # Dropped orders: orders in selected blocks but not in schedule
    scheduled_set = set(schedule_df["order_id"]) if not schedule_df.empty else set()
    dropped = [
        orders_by_id[oid]
        for oid in selected_order_set
        if oid not in scheduled_set and oid in orders_by_id
    ]
    dropped_df = pd.DataFrame(dropped) if dropped else pd.DataFrame()

    diag_counters["mixed_bridge_gain_dropped"] = 0  # Can't recover from block realization
    diag_counters["mixed_bridge_gain_underfilled"] = 0

    # Block quality in realized
    if realized_blocks:
        avg_quality = sum(b.block_quality_score for b in realized_blocks) / len(realized_blocks)
        diag_counters["avg_realized_block_quality"] = avg_quality
        diag_counters["avg_realized_block_tons"] = (
            sum(b.total_tons for b in realized_blocks) / len(realized_blocks)
        )

    return BlockRealizationResult(
        realized_blocks=realized_blocks,
        realized_schedule_df=schedule_df,
        realized_dropped_df=dropped_df,
        block_realization_diag=diag_counters,
    )
