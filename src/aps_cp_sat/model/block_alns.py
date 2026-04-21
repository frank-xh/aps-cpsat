"""
Block ALNS: Block-Level Adaptive Large Neighborhood Search.

Provides block-level neighborhood search operators for the block-first path:
- BLOCK_SWAP: exchange order of two blocks on the same line
- BLOCK_REPLACE: replace a block with another from the rejected pool
- BLOCK_SPLIT: split a large block into two smaller ones
- BLOCK_MERGE: merge two adjacent small blocks into one
- BLOCK_BOUNDARY_REBALANCE: move boundary orders between adjacent blocks
- BLOCK_INTERNAL_REBUILD: rebuild a block internally using local CP-SAT

This runs AFTER block_master selects blocks and block_realizer realizes them.
It is analogous to the order-level ALNS in constructive_lns_master.

Only active under block_first_guarded_search profile.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Tuple
import random

import pandas as pd

from aps_cp_sat.config import PlannerConfig
from aps_cp_sat.model.block_types import CandidateBlock, CandidateBlockPool
from aps_cp_sat.model.block_master import BlockMasterResult, solve_block_master
from aps_cp_sat.model.block_realizer import BlockRealizationResult, realize_selected_blocks


@dataclass
class BlockALNSResult:
    """Result of block-level ALNS neighborhood search."""
    final_master_result: BlockMasterResult
    final_realization_result: BlockRealizationResult
    iterations_attempted: int = 0
    iterations_accepted: int = 0
    total_runtime_seconds: float = 0.0

    # Per-neighborhood diagnostics
    neighborhood_diag: Dict[str, int] = field(default_factory=dict)

    # Scoring diagnostics
    plan_score_diag: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "iterations_attempted": self.iterations_attempted,
            "iterations_accepted": self.iterations_accepted,
            "total_runtime_seconds": self.total_runtime_seconds,
            "neighborhood_diag": dict(self.neighborhood_diag),
            "plan_score_diag": dict(self.plan_score_diag),
            "final_master": self.final_master_result.to_dict(),
            "final_realization": self.final_realization_result.to_dict(),
        }


# ----------------------------------------------------------------------
# Scoring Function
# ----------------------------------------------------------------------

def evaluate_block_first_plan(
    master_result: BlockMasterResult,
    realization_result: BlockRealizationResult,
    cfg: PlannerConfig,
) -> Tuple[float, Dict[str, Any]]:
    """
    Evaluates the real schedule produced by the block-first plan.
    Returns (score, diagnostics_dict).
    """
    score = 0.0
    diag = {
        "scoring_basis": "realized_schedule",
        "selected_orders_count": len(master_result.selected_order_ids),
    }

    # Extract realized metrics
    if realization_result.realized_schedule_df is not None and not realization_result.realized_schedule_df.empty:
        df = realization_result.realized_schedule_df
        scheduled_orders = int(df["order_id"].nunique()) if "order_id" in df.columns else len(df)
        scheduled_tons = float(df["tons"].sum()) if "tons" in df.columns else 0.0
    else:
        scheduled_orders = 0
        scheduled_tons = 0.0

    diag["scheduled_orders_realized"] = scheduled_orders
    diag["scheduled_tons_realized"] = scheduled_tons

    # Positive items
    score += scheduled_orders * 100.0
    score += scheduled_tons * 0.1

    # Negative items
    slot_diag = master_result.slot_diag
    underfilled_slots = slot_diag.get("underfilled_slot_count", 0)
    diag["underfilled_slots"] = underfilled_slots
    score -= underfilled_slots * 500.0

    unassembled_blocks = slot_diag.get("unassembled_block_count", 0)
    diag["unassembled_blocks"] = unassembled_blocks
    score -= unassembled_blocks * 200.0

    failed_boundaries = realization_result.block_realization_diag.get("failed_block_boundary_count", 0)
    diag["failed_boundaries"] = failed_boundaries
    score -= failed_boundaries * 1000.0

    # Minor static items to guide search when major factors tie
    avg_quality = master_result.diagnostics.get("avg_block_quality_in_selected", 0.0)
    diag["avg_quality"] = avg_quality
    score += avg_quality * 0.5

    diag["final_score"] = score
    return score, diag


# ----------------------------------------------------------------------
# Neighborhood operators
# ----------------------------------------------------------------------


def _neighborhood_swap(
    master_result: BlockMasterResult,
    orders_by_id: Dict[str, Dict],
    rng: random.Random,
) -> Tuple[List[CandidateBlock], Dict[str, List[str]], str]:
    """
    BLOCK_SWAP: exchange positions of two blocks on the same line.
    Returns (modified_blocks, preferred_block_order_by_line, neighborhood_name).
    """
    selected_blocks = list(master_result.selected_blocks)
    block_order = {k: list(v) for k, v in master_result.block_order_by_line.items()}

    for line, b_ids in block_order.items():
        if len(b_ids) < 2:
            continue
        i, j = rng.sample(range(len(b_ids)), 2)
        # Swap block IDs in the line's order
        b_ids[i], b_ids[j] = b_ids[j], b_ids[i]
        return selected_blocks, block_order, "BLOCK_SWAP"

    return selected_blocks, block_order, "BLOCK_SWAP"


def _neighborhood_replace(
    master_result: BlockMasterResult,
    rejected_pool: List[CandidateBlock],
    orders_df: pd.DataFrame,
    cfg: PlannerConfig,
    rng: random.Random,
) -> Tuple[List[CandidateBlock], Dict[str, List[str]], str]:
    """
    BLOCK_REPLACE: replace a selected block with a rejected block.
    """
    selected_blocks = list(master_result.selected_blocks)
    block_order = {k: list(v) for k, v in master_result.block_order_by_line.items()}

    if not rejected_pool:
        return selected_blocks, block_order, "BLOCK_REPLACE"

    # Try to replace the lowest-quality selected block
    sorted_selected = sorted(selected_blocks, key=lambda b: b.block_quality_score)
    worst = sorted_selected[0]

    # Find best matching rejected block (same line, no order conflicts)
    used_orders = set()
    for b in selected_blocks:
        if b is not worst:
            used_orders.update(b.order_ids)

    candidates = [
        r for r in rejected_pool
        if r.line == worst.line and not any(oid in used_orders for oid in r.order_ids)
    ]
    if not candidates:
        return selected_blocks, block_order, "BLOCK_REPLACE"

    candidates.sort(key=lambda b: b.block_quality_score, reverse=True)
    replacement = candidates[0]

    result = [b if b is not worst else replacement for b in selected_blocks]
    
    # Update block order
    if worst.line in block_order:
        try:
            idx = block_order[worst.line].index(worst.block_id)
            block_order[worst.line][idx] = replacement.block_id
        except ValueError:
            pass

    return result, block_order, "BLOCK_REPLACE"


def _neighborhood_split(
    blocks: List[CandidateBlock],
    orders_by_id: Dict[str, Dict],
    cfg: PlannerConfig,
    rng: random.Random,
) -> Tuple[List[CandidateBlock], str]:
    """
    BLOCK_SPLIT: split a large block into two smaller blocks at a middle boundary.
    Only applies to blocks with many orders.
    """
    max_orders = int(getattr(cfg.model, "block_generator_max_orders_per_block", 30))
    large = [b for b in blocks if b.order_count >= max_orders // 2]
    if not large:
        return list(blocks), "BLOCK_SPLIT"

    chosen = rng.choice(large)
    mid = len(chosen.order_ids) // 2

    half_a_ids = chosen.order_ids[:mid]
    half_b_ids = chosen.order_ids[mid:]

    tons_a = sum(float(orders_by_id.get(o, {}).get("tons", 0)) for o in half_a_ids)
    tons_b = sum(float(orders_by_id.get(o, {}).get("tons", 0)) for o in half_b_ids)

    head_a = orders_by_id.get(half_a_ids[0], {}) if half_a_ids else {}
    head_b = orders_by_id.get(half_b_ids[0], {}) if half_b_ids else {}
    tail_a = orders_by_id.get(half_a_ids[-1], {}) if half_a_ids else {}
    tail_b = orders_by_id.get(half_b_ids[-1], {}) if half_b_ids else {}

    new_block_a = CandidateBlock(
        block_id=f"{chosen.block_id}_sA",
        line=chosen.line,
        order_ids=half_a_ids,
        order_count=len(half_a_ids),
        total_tons=tons_a,
        head_order_id=half_a_ids[0],
        tail_order_id=half_a_ids[-1],
        head_signature=dict(head_a),
        tail_signature=dict(tail_a),
        width_band=chosen.width_band,
        thickness_band=chosen.thickness_band,
        steel_group_profile=chosen.steel_group_profile,
        temp_band=chosen.temp_band,
        direct_edge_count=chosen.direct_edge_count // 2,
        real_bridge_edge_count=chosen.real_bridge_edge_count // 2,
        virtual_family_edge_count=chosen.virtual_family_edge_count // 2,
        mixed_bridge_possible=False,
        mixed_bridge_reason="split_from_larger_block",
        block_quality_score=chosen.block_quality_score * 0.85,
        underfill_risk_score=min(1.0, chosen.underfill_risk_score * 1.2),
        bridge_dependency_score=chosen.bridge_dependency_score,
        dropped_recovery_potential=0.0,
        source_bucket_key=chosen.source_bucket_key,
        source_generation_mode="block_split",
        internal_edges=chosen.internal_edges[:mid - 1] if mid > 1 else [],
    )

    new_block_b = CandidateBlock(
        block_id=f"{chosen.block_id}_sB",
        line=chosen.line,
        order_ids=half_b_ids,
        order_count=len(half_b_ids),
        total_tons=tons_b,
        head_order_id=half_b_ids[0],
        tail_order_id=half_b_ids[-1],
        head_signature=dict(head_b),
        tail_signature=dict(tail_b),
        width_band=chosen.width_band,
        thickness_band=chosen.thickness_band,
        steel_group_profile=chosen.steel_group_profile,
        temp_band=chosen.temp_band,
        direct_edge_count=chosen.direct_edge_count // 2,
        real_bridge_edge_count=chosen.real_bridge_edge_count // 2,
        virtual_family_edge_count=chosen.virtual_family_edge_count // 2,
        mixed_bridge_possible=False,
        mixed_bridge_reason="split_from_larger_block",
        block_quality_score=chosen.block_quality_score * 0.85,
        underfill_risk_score=min(1.0, chosen.underfill_risk_score * 1.2),
        bridge_dependency_score=chosen.bridge_dependency_score,
        dropped_recovery_potential=0.0,
        source_bucket_key=chosen.source_bucket_key,
        source_generation_mode="block_split",
        internal_edges=chosen.internal_edges[mid:] if mid < len(chosen.internal_edges) else [],
    )

    result = [b for b in blocks if b is not chosen] + [new_block_a, new_block_b]
    return result, "BLOCK_SPLIT"


def _neighborhood_merge(
    blocks: List[CandidateBlock],
    orders_by_id: Dict[str, Dict],
    cfg: PlannerConfig,
    rng: random.Random,
) -> Tuple[List[CandidateBlock], str]:
    """
    BLOCK_MERGE: merge two adjacent small blocks into one larger block.
    Only merges if result stays within max_orders limit.
    """
    max_orders = int(getattr(cfg.model, "block_generator_max_orders_per_block", 30))
    by_line: Dict[str, List[CandidateBlock]] = {}
    for b in blocks:
        by_line.setdefault(b.line, []).append(b)

    for line, line_blocks in by_line.items():
        if len(line_blocks) < 2:
            continue
        # Try to find two adjacent blocks (by block_order) that can merge
        for i in range(len(line_blocks) - 1):
            a, b = line_blocks[i], line_blocks[i + 1]
            if a.order_count + b.order_count > max_orders:
                continue

            merged_ids = a.order_ids + b.order_ids
            merged_tons = a.total_tons + b.total_tons
            head = orders_by_id.get(a.order_ids[0], {})
            tail = orders_by_id.get(b.order_ids[-1], {})

            merged = CandidateBlock(
                block_id=f"{a.block_id}_m{b.block_id}",
                line=a.line,
                order_ids=merged_ids,
                order_count=len(merged_ids),
                total_tons=merged_tons,
                head_order_id=a.order_ids[0],
                tail_order_id=b.order_ids[-1],
                head_signature=a.head_signature,
                tail_signature=b.tail_signature,
                width_band=a.width_band,
                thickness_band=a.thickness_band,
                steel_group_profile=a.steel_group_profile,
                temp_band=a.temp_band,
                direct_edge_count=a.direct_edge_count + b.direct_edge_count + 1,
                real_bridge_edge_count=a.real_bridge_edge_count + b.real_bridge_edge_count,
                virtual_family_edge_count=a.virtual_family_edge_count + b.virtual_family_edge_count,
                mixed_bridge_possible=False,
                mixed_bridge_reason="merged_from_small_blocks",
                block_quality_score=(a.block_quality_score + b.block_quality_score) / 2,
                underfill_risk_score=max(a.underfill_risk_score, b.underfill_risk_score),
                bridge_dependency_score=(a.bridge_dependency_score + b.bridge_dependency_score) / 2,
                dropped_recovery_potential=0.0,
                source_bucket_key=a.source_bucket_key,
                source_generation_mode="block_merge",
                internal_edges=a.internal_edges + b.internal_edges,
            )

            result = [blk for blk in blocks if blk not in (a, b)] + [merged]
            return result, "BLOCK_MERGE"

    return list(blocks), "BLOCK_MERGE"


def _neighborhood_boundary_rebalance(
    blocks: List[CandidateBlock],
    orders_by_id: Dict[str, Dict],
    cfg: PlannerConfig,
    rng: random.Random,
) -> Tuple[List[CandidateBlock], str]:
    """
    BLOCK_BOUNDARY_REBALANCE: move a few orders from end of one block to start of next.
    """
    by_line: Dict[str, List[CandidateBlock]] = {}
    for b in blocks:
        by_line.setdefault(b.line, []).append(b)

    for line, line_blocks in by_line.items():
        if len(line_blocks) < 2:
            continue
        for i in range(len(line_blocks) - 1):
            a, b = line_blocks[i], line_blocks[i + 1]
            if a.order_count <= 2 or b.order_count <= 2:
                continue

            # Move last 1-2 orders from a to b
            move_count = rng.randint(1, min(2, a.order_count - 1))
            moved_ids = a.order_ids[-move_count:]
            moved_tons = sum(float(orders_by_id.get(o, {}).get("tons", 0)) for o in moved_ids)

            new_a_ids = a.order_ids[:-move_count]
            new_b_ids = moved_ids + b.order_ids

            new_a = CandidateBlock(
                block_id=a.block_id,
                line=a.line,
                order_ids=new_a_ids,
                order_count=len(new_a_ids),
                total_tons=a.total_tons - moved_tons,
                head_order_id=a.order_ids[0],
                tail_order_id=new_a_ids[-1] if new_a_ids else a.order_ids[0],
                head_signature=a.head_signature,
                tail_signature=orders_by_id.get(new_a_ids[-1], {}) if new_a_ids else a.tail_signature,
                width_band=a.width_band,
                thickness_band=a.thickness_band,
                steel_group_profile=a.steel_group_profile,
                temp_band=a.temp_band,
                direct_edge_count=max(1, a.direct_edge_count - move_count),
                real_bridge_edge_count=a.real_bridge_edge_count,
                virtual_family_edge_count=a.virtual_family_edge_count,
                mixed_bridge_possible=False,
                mixed_bridge_reason="boundary_rebalance",
                block_quality_score=a.block_quality_score,
                underfill_risk_score=a.underfill_risk_score,
                bridge_dependency_score=a.bridge_dependency_score,
                dropped_recovery_potential=0.0,
                source_bucket_key=a.source_bucket_key,
                source_generation_mode=a.source_generation_mode,
                internal_edges=a.internal_edges[:-move_count] if a.internal_edges else [],
            )

            new_b = CandidateBlock(
                block_id=b.block_id,
                line=b.line,
                order_ids=new_b_ids,
                order_count=len(new_b_ids),
                total_tons=b.total_tons + moved_tons,
                head_order_id=moved_ids[0],
                tail_order_id=b.order_ids[-1],
                head_signature=orders_by_id.get(moved_ids[0], {}),
                tail_signature=b.tail_signature,
                width_band=b.width_band,
                thickness_band=b.thickness_band,
                steel_group_profile=b.steel_group_profile,
                temp_band=b.temp_band,
                direct_edge_count=b.direct_edge_count + move_count,
                real_bridge_edge_count=b.real_bridge_edge_count,
                virtual_family_edge_count=b.virtual_family_edge_count,
                mixed_bridge_possible=False,
                mixed_bridge_reason="boundary_rebalance",
                block_quality_score=b.block_quality_score,
                underfill_risk_score=b.underfill_risk_score,
                bridge_dependency_score=b.bridge_dependency_score,
                dropped_recovery_potential=0.0,
                source_bucket_key=b.source_bucket_key,
                source_generation_mode=b.source_generation_mode,
                internal_edges=b.internal_edges,
            )

            result = [blk for blk in blocks if blk not in (a, b)] + [new_a, new_b]
            return result, "BLOCK_BOUNDARY_REBALANCE"

    return list(blocks), "BLOCK_BOUNDARY_REBALANCE"


def _neighborhood_internal_rebuild(
    blocks: List[CandidateBlock],
    orders_df: pd.DataFrame,
    transition_pack: dict,
    cfg: PlannerConfig,
    rng: random.Random,
) -> Tuple[List[CandidateBlock], str]:
    """
    BLOCK_INTERNAL_REBUILD: for blocks with hotspots, rebuild internal order.

    For now, just re-order using a different edge selection strategy.
    """
    # Find blocks with mixed_bridge_possible (hotspot blocks)
    hotspot_blocks = [b for b in blocks if b.mixed_bridge_possible]
    if not hotspot_blocks:
        return list(blocks), "BLOCK_INTERNAL_REBUILD"

    chosen = rng.choice(hotspot_blocks)

    # Re-order using a reversed or shuffled strategy
    new_order = list(reversed(chosen.order_ids))

    head = orders_df[orders_df["order_id"].astype(str) == new_order[0]]
    tail = orders_df[orders_df["order_id"].astype(str) == new_order[-1]]

    rebuilt = CandidateBlock(
        block_id=chosen.block_id + "_ir",
        line=chosen.line,
        order_ids=new_order,
        order_count=chosen.order_count,
        total_tons=chosen.total_tons,
        head_order_id=new_order[0],
        tail_order_id=new_order[-1],
        head_signature=dict(head.iloc[0]) if not head.empty else {},
        tail_signature=dict(tail.iloc[0]) if not tail.empty else {},
        width_band=chosen.width_band,
        thickness_band=chosen.thickness_band,
        steel_group_profile=chosen.steel_group_profile,
        temp_band=chosen.temp_band,
        direct_edge_count=chosen.direct_edge_count,
        real_bridge_edge_count=chosen.real_bridge_edge_count,
        virtual_family_edge_count=chosen.virtual_family_edge_count,
        mixed_bridge_possible=chosen.mixed_bridge_possible,
        mixed_bridge_reason=chosen.mixed_bridge_reason,
        block_quality_score=chosen.block_quality_score,
        underfill_risk_score=chosen.underfill_risk_score,
        bridge_dependency_score=chosen.bridge_dependency_score,
        dropped_recovery_potential=0.0,
        source_bucket_key=chosen.source_bucket_key,
        source_generation_mode="internal_rebuild",
        internal_edges=[],
    )

    result = [b if b is not chosen else rebuilt for b in blocks]
    return result, "BLOCK_INTERNAL_REBUILD"


# ----------------------------------------------------------------------
# Main ALNS driver
# ----------------------------------------------------------------------


def run_block_alns(
    initial_pool: CandidateBlockPool,
    initial_master_result: BlockMasterResult,
    initial_realization_result: BlockRealizationResult,
    orders_df: pd.DataFrame,
    transition_pack: dict,
    cfg: PlannerConfig,
    random_seed: int = 42,
) -> BlockALNSResult:
    """
    Run block-level ALNS on top of the block master selection.

    Neighborhoods are attempted in random order each round.
    Accept: always accept improvement (or keep current if worse).

    Returns BlockALNSResult with final master and realization.
    """
    import time
    t0 = time.perf_counter()

    rng = random.Random(random_seed)

    rounds = int(getattr(cfg.model, "block_alns_rounds", 10))
    accept_threshold = float(getattr(cfg.model, "block_alns_accept_threshold", 0.0))

    # Enable/disable neighborhoods
    enabled = {
        "BLOCK_SWAP": bool(getattr(cfg.model, "block_alns_swap_enabled", True)),
        "BLOCK_REPLACE": bool(getattr(cfg.model, "block_alns_replace_enabled", True)),
        "BLOCK_SPLIT": bool(getattr(cfg.model, "block_alns_split_enabled", True)),
        "BLOCK_MERGE": bool(getattr(cfg.model, "block_alns_merge_enabled", True)),
        "BLOCK_BOUNDARY_REBALANCE": bool(getattr(cfg.model, "block_alns_boundary_rebalance_enabled", True)),
        "BLOCK_INTERNAL_REBUILD": bool(getattr(cfg.model, "block_alns_internal_rebuild_enabled", True)),
    }

    available = [n for n, en in enabled.items() if en]
    if not available:
        return BlockALNSResult(
            final_master_result=initial_master_result,
            final_realization_result=initial_realization_result,
            iterations_attempted=0,
            iterations_accepted=0,
            total_runtime_seconds=time.perf_counter() - t0,
            neighborhood_diag={n: 0 for n in enabled},
        )

    orders_by_id = {str(row.get("order_id", "")): dict(row) for _, row in orders_df.iterrows()}

    # Current best state
    current_master_result = initial_master_result
    current_realization_result = initial_realization_result
    current_score, current_diag = evaluate_block_first_plan(current_master_result, current_realization_result, cfg)

    best_master_result = current_master_result
    best_realization_result = current_realization_result
    best_score = current_score
    best_diag = current_diag

    nbd_diag = {n: 0 for n in enabled}
    attempts = 0
    accepted = 0

    for rnd in range(rounds):
        # Pick random neighborhood
        nbd = rng.choice(available)
        nbd_diag[nbd] = nbd_diag.get(nbd, 0) + 1
        attempts += 1

        try:
            trial_blocks = list(current_master_result.selected_blocks)
            preferred_order = None

            # Apply neighborhood
            if nbd == "BLOCK_SWAP":
                trial_blocks, preferred_order, _ = _neighborhood_swap(current_master_result, orders_by_id, rng)
            elif nbd == "BLOCK_REPLACE":
                trial_blocks, preferred_order, _ = _neighborhood_replace(
                    current_master_result, list(initial_master_result.rejected_blocks),
                    orders_df, cfg, rng,
                )
            elif nbd == "BLOCK_SPLIT":
                trial_blocks, _ = _neighborhood_split(trial_blocks, orders_by_id, cfg, rng)
            elif nbd == "BLOCK_MERGE":
                trial_blocks, _ = _neighborhood_merge(trial_blocks, orders_by_id, cfg, rng)
            elif nbd == "BLOCK_BOUNDARY_REBALANCE":
                trial_blocks, _ = _neighborhood_boundary_rebalance(
                    trial_blocks, orders_by_id, cfg, rng,
                )
            elif nbd == "BLOCK_INTERNAL_REBUILD":
                trial_blocks, _ = _neighborhood_internal_rebuild(
                    trial_blocks, orders_df, transition_pack, cfg, rng,
                )

            # Re-evaluate the trial
            temp_pool = CandidateBlockPool(
                blocks=list(initial_pool.blocks) + [b for b in trial_blocks if b not in initial_pool.blocks],
                diagnostics=initial_pool.diagnostics,
            )

            trial_master_result = solve_block_master(
                pool=temp_pool,
                orders_df=orders_df,
                cfg=cfg,
                random_seed=random_seed + rnd,
                preferred_block_order_by_line=preferred_order,
            )

            trial_realization_result = realize_selected_blocks(
                master_result=trial_master_result,
                orders_df=orders_df,
                transition_pack=transition_pack,
                cfg=cfg,
                random_seed=random_seed + rnd,
            )

            # Score trial based on actual schedule
            trial_score, trial_diag = evaluate_block_first_plan(trial_master_result, trial_realization_result, cfg)

            delta = trial_score - current_score

            if delta > accept_threshold:
                current_master_result = trial_master_result
                current_realization_result = trial_realization_result
                current_score = trial_score
                accepted += 1
                if trial_score > best_score:
                    best_master_result = trial_master_result
                    best_realization_result = trial_realization_result
                    best_score = trial_score
                    best_diag = trial_diag

        except Exception:
            # Skip failed neighborhood without crashing
            continue

    nbd_diag["block_alns_rounds_attempted"] = attempts
    nbd_diag["block_alns_rounds_accepted"] = accepted

    return BlockALNSResult(
        final_master_result=best_master_result,
        final_realization_result=best_realization_result,
        iterations_attempted=attempts,
        iterations_accepted=accepted,
        total_runtime_seconds=time.perf_counter() - t0,
        neighborhood_diag=nbd_diag,
        plan_score_diag=best_diag,
    )
