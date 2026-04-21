"""
Block Master: select and order candidate blocks for Block-First Architecture.

Goal: NOT to solve global order-level chains, but to:
1. Select which candidate blocks to use
2. Ensure order coverage without conflicts
3. Order blocks on each line for campaign feasibility

This is a lightweight greedy block master, NOT an exact MILP master.

Key principles:
- Each order appears in at most one selected block
- Each line selects only its own blocks
- Prioritize: coverage > quality > bridge dependency > underfill risk
- Block-to-block ordering based on head/tail compatibility
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import random

import pandas as pd

from aps_cp_sat.config import PlannerConfig
from aps_cp_sat.model.block_types import CandidateBlock, CandidateBlockPool


@dataclass
class BlockMasterResult:
    """Result of block master selection and ordering."""
    selected_blocks: List[CandidateBlock] = field(default_factory=list)
    rejected_blocks: List[CandidateBlock] = field(default_factory=list)
    dropped_order_ids: List[str] = field(default_factory=list)
    selected_order_ids: List[str] = field(default_factory=list)
    block_order_by_line: Dict[str, List[str]] = field(default_factory=dict)

    # Diagnostics
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "selected_blocks": [b.to_dict() for b in self.selected_blocks],
            "rejected_blocks": [b.to_dict() for b in self.rejected_blocks],
            "dropped_order_ids": list(self.dropped_order_ids),
            "selected_order_ids": list(self.selected_order_ids),
            "block_order_by_line": {k: list(v) for k, v in self.block_order_by_line.items()},
            "diagnostics": dict(self.diagnostics),
        }


def _compute_transition_score(block_i: CandidateBlock, block_j: CandidateBlock) -> float:
    """
    Compute block-to-block transition score (higher = smoother transition).
    Used for ordering blocks on each line.
    """
    if block_i.line != block_j.line:
        return -999.0  # Different lines cannot transition

    h_sig = block_i.tail_signature
    t_sig = block_j.head_signature

    w_i = float(h_sig.get("width", 0))
    w_j = float(t_sig.get("width", 0))
    width_gap = abs(w_i - w_j)

    t_i = float(h_sig.get("thickness", 0))
    t_j = float(t_sig.get("thickness", 0))
    thick_gap = abs(t_i - t_j)

    g_i = str(h_sig.get("steel_group", ""))
    g_j = str(t_sig.get("steel_group", ""))
    group_switch = 0 if g_i == g_j else -5.0

    score = -width_gap * 0.01 - thick_gap * 2.0 + group_switch
    if width_gap <= 200 and thick_gap <= 0.3:
        score += 10.0

    return score


def _block_conflict_count(a: CandidateBlock, b: CandidateBlock) -> int:
    """Return number of overlapping order IDs between two blocks."""
    return len(set(a.order_ids) & set(b.order_ids))


def solve_block_master(
    pool: CandidateBlockPool,
    orders_df: pd.DataFrame,
    cfg: PlannerConfig,
    random_seed: int = 42,
) -> BlockMasterResult:
    """
    Lightweight greedy block master.

    Selection priority:
    1. Higher block_quality_score
    2. Lower underfill_risk_score
    3. Lower bridge_dependency_score
    4. Max dropped_recovery_potential

    Conflicts: skip blocks with too many overlapping orders.

    Ordering: greedy chain ordering on each line using transition scores.
    """
    rng = random.Random(random_seed)
    blocks = list(pool.blocks)
    max_skip = int(getattr(cfg.model, "block_master_max_conflict_skip", 5))
    prefer_quality = bool(getattr(cfg.model, "block_master_prefer_quality_score", True))

    # Sort blocks by composite score
    # NOTE: small_candidate blocks (candidate_tons_min <= total_tons < target_tons_min)
    # are allowed into the pool but receive a quality penalty here to discourage
    # them from being selected as final campaigns. This is intentional:
    # they exist for merge / boundary rebalance / block internal rebuild purposes.
    SMALL_CANDIDATE_PENALTY = 20.0  # subtracted from quality score

    def composite_score(b: CandidateBlock) -> float:
        quality = b.block_quality_score
        # Apply disadvantage to small candidates (below target_tons_min but above candidate_tons_min)
        if b.is_under_target_block:
            quality -= SMALL_CANDIDATE_PENALTY
        underfill_risk = 1.0 - b.underfill_risk_score
        bridge_dep = 1.0 - b.bridge_dependency_score
        recovery = b.dropped_recovery_potential * 10.0
        tons_score = min(1.0, b.total_tons / 400.0) * 5.0
        return quality * 0.4 + underfill_risk * 10.0 * 0.2 + bridge_dep * 10.0 * 0.2 + recovery * 0.1 + tons_score * 0.1

    blocks_sorted = sorted(blocks, key=composite_score, reverse=True)

    selected: List[CandidateBlock] = []
    rejected: List[CandidateBlock] = []
    order_to_block: Dict[str, str] = {}

    for b in blocks_sorted:
        conflicts = 0
        for oid in b.order_ids:
            if oid in order_to_block:
                conflicts += 1

        conflict_ratio = conflicts / max(1, b.order_count)
        if conflicts > max_skip or conflict_ratio > 0.5:
            rejected.append(b)
            continue

        for oid in b.order_ids:
            order_to_block[oid] = b.block_id
        selected.append(b)

    # All orders not in any selected block = dropped
    all_order_ids = set(str(oid) for oid in orders_df["order_id"])
    dropped_order_ids = [str(oid) for oid in all_order_ids if oid not in order_to_block]
    selected_order_ids = list(order_to_block.keys())

    # ---- Order blocks on each line ----
    block_order_by_line: Dict[str, List[str]] = {}
    ordered_blocks_global: List[CandidateBlock] = []

    for line in sorted(set(b.line for b in selected)):
        line_blocks = [b for b in selected if b.line == line]

        if not line_blocks:
            continue

        # Greedy ordering by transition score
        ordered: List[CandidateBlock] = []
        remaining = list(line_blocks)

        # Start with best quality block
        current = remaining.pop(
            rng.randint(0, min(2, len(remaining) - 1))
            if rng.random() < 0.5
            else 0
        )
        ordered.append(current)

        while remaining:
            best_next = None
            best_score = -9999.0
            for nxt in remaining:
                score = _compute_transition_score(current, nxt)
                if score > best_score:
                    best_score = score
                    best_next = nxt
            if best_next is None:
                break
            ordered.append(best_next)
            remaining.remove(best_next)
            current = best_next

        block_order_by_line[line] = [b.block_id for b in ordered]
        ordered_blocks_global.extend(ordered)

    # Mark selected blocks
    selected_ids = {b.block_id for b in selected}
    for b in pool.blocks:
        b.is_selected = (b.block_id in selected_ids)

    # ---- Diagnostics ----
    selected_by_line: Dict[str, int] = {}
    for b in selected:
        selected_by_line[b.line] = selected_by_line.get(b.line, 0) + 1

    diag = {
        "selected_blocks_count": len(selected),
        "rejected_blocks_count": len(rejected),
        "selected_blocks_by_line": selected_by_line,
        "selected_order_coverage": len(selected_order_ids),
        "total_input_orders": len(orders_df),
        "block_conflict_rejections": len(rejected),
        "dropped_order_count": len(dropped_order_ids),
        "block_master_dropped_count": len(dropped_order_ids),
        "avg_block_quality_in_selected": (
            sum(b.block_quality_score for b in selected) / max(1, len(selected))
        ),
        "avg_block_tons_in_selected": (
            sum(b.total_tons for b in selected) / max(1, len(selected))
        ),
        # ---- Candidate size class in selected/rejected ----
        # small_candidate blocks in selected are allowed (for merge/rebuild) but tracked
        "selected_small_candidate_blocks": sum(1 for b in selected if b.candidate_size_class == "small_candidate"),
        "selected_target_candidate_blocks": sum(1 for b in selected if b.candidate_size_class == "target_candidate"),
        "rejected_small_candidate_blocks": sum(1 for b in rejected if b.candidate_size_class == "small_candidate"),
    }

    if ordered_blocks_global and len(ordered_blocks_global) >= 2:
        trans_costs = []
        for i in range(len(ordered_blocks_global) - 1):
            tc = _compute_transition_score(ordered_blocks_global[i], ordered_blocks_global[i + 1])
            trans_costs.append(tc)
        diag["block_transition_avg_cost"] = sum(trans_costs) / max(1, len(trans_costs))

    return BlockMasterResult(
        selected_blocks=selected,
        rejected_blocks=rejected,
        dropped_order_ids=dropped_order_ids,
        selected_order_ids=selected_order_ids,
        block_order_by_line=block_order_by_line,
        diagnostics=diag,
    )
