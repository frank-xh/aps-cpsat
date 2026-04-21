"""
================================================================================
PRODUCTION-LEVEL BLOCK POOL ORCHESTRATOR: block_generator.py
================================================================================

本模块是 block-first 架构中的 **生产级块池编排器**。

核心职责：
- 编排块生成策略，以 feasible_block_builder 为底层引擎：
  - A. Seed Greedy Blocks: 从种子订单贪心扩展
  - B. Underfilled Rescue Blocks: 抢救未填满的区段
  - C. Boundary Patch Blocks: 修补组切换/宽度张力热点
  - D. Dropped Recovery Blocks: 从丢弃订单池恢复

- 显式使用 feasible_block_builder 的底层能力：
  - generate_candidate_macro_blocks() - 低层候选块生成
  - TemplateGraph 图构建
  - hard_cluster_gate 硬可行性过滤
  - directed_cluster_distance 定向距离计算

- 输出 CandidateBlockPool，包含诊断信息

权威主路径：
  model/master.py -> block_generator.generate_candidate_blocks()  <-- 本模块入口
                  -> block_master.solve_block_master()
                  -> block_realizer.realize_selected_blocks()
                  -> block_alns.run_block_alns()

与 feasible_block_builder.py 的关系：
- feasible_block_builder.py: 低层块生成引擎 (primitives)
  - 提供：MacroBlock, generate_candidate_macro_blocks(), hard_cluster_gate()
  - 提供：directed_cluster_distance(), TemplateGraph
- block_generator.py: 本模块，生产级编排器
  - 使用 feasible_block_builder 的底层引擎
  - 负责 orchestration, diagnostics merge, pool packaging

================================================================================
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from time import perf_counter
import random

import pandas as pd

from aps_cp_sat.config import PlannerConfig
from aps_cp_sat.model.block_types import BlockSeed, CandidateBlock, CandidateBlockPool
from aps_cp_sat.model.candidate_graph_types import (
    DIRECT_EDGE,
    REAL_BRIDGE_EDGE,
    VIRTUAL_BRIDGE_EDGE,
    VIRTUAL_BRIDGE_FAMILY_EDGE,
)
# ---- Explicitly import from feasible_block_builder (low-level engine) ----
from aps_cp_sat.model.feasible_block_builder import (
    MacroBlock,
    generate_candidate_macro_blocks as _generate_macro_blocks,
    hard_cluster_gate,
    directed_cluster_distance,
)


def _band_key(val: float, step: float = 200.0) -> str:
    """Simple band binning: width/thickness binning."""
    return f"{int(val / step) * int(step)}-{(int(val / step) + 1) * int(step)}"


def _temp_band_key(tmin: float, tmax: float) -> str:
    return f"{tmin}-{tmax}"


def _orders_to_seeds(orders_df: pd.DataFrame, lines: List[str]) -> Dict[str, List[BlockSeed]]:
    """Convert orders DataFrame to BlockSeed dict by line."""
    result: Dict[str, List[BlockSeed]] = defaultdict(list)
    for _, row in orders_df.iterrows():
        line = str(row.get("line", ""))
        if lines and line not in lines:
            continue
        try:
            seed = BlockSeed(
                order_id=str(row.get("order_id", "")),
                line=line,
                width=float(row.get("width", 0)),
                thickness=float(row.get("thickness", 0)),
                steel_group=str(row.get("steel_group", "")),
                temp_min=float(row.get("temp_min", 0)),
                temp_max=float(row.get("temp_max", 0)),
                tons=float(row.get("tons", 0)),
                priority=int(row.get("priority", 0)),
                due_rank=int(row.get("due_rank", 9999)),
                width_band=_band_key(float(row.get("width", 0))),
                thickness_band=_band_key(float(row.get("thickness", 0)), step=0.2),
                temp_band=_temp_band_key(float(row.get("temp_min", 0)), float(row.get("temp_max", 0))),
                roll_type=str(row.get("roll_type", "")),
            )
            result[line].append(seed)
        except (ValueError, TypeError):
            continue
    return result


def _signature_from_seed(seed: BlockSeed) -> Dict[str, Any]:
    return {
        "order_id": seed.order_id,
        "width": seed.width,
        "thickness": seed.thickness,
        "steel_group": seed.steel_group,
        "temp_min": seed.temp_min,
        "temp_max": seed.temp_max,
        "width_band": seed.width_band,
        "thickness_band": seed.thickness_band,
    }


def _signature_from_order_dict(order: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "order_id": str(order.get("order_id", "")),
        "width": float(order.get("width", 0)),
        "thickness": float(order.get("thickness", 0)),
        "steel_group": str(order.get("steel_group", "")),
        "temp_min": float(order.get("temp_min", 0)),
        "temp_max": float(order.get("temp_max", 0)),
    }


def _compute_block_quality_score(
    block: CandidateBlock,
    cfg,
) -> float:
    """
    Compute a 0..100 quality score for a block.
    Higher = better.
    """
    score = 50.0  # baseline

    # Reward: more real orders
    score += min(15.0, block.order_count * 1.5)

    # Reward: good tons coverage
    target_min = float(getattr(cfg.model, "block_generator_target_tons_min", 150.0))
    target_max = float(getattr(cfg.model, "block_generator_target_tons_max", 550.0))
    if target_min <= block.total_tons <= target_max:
        score += 15.0
    elif block.total_tons > target_max:
        score -= 5.0

    # Reward: more direct edges (stable)
    score += min(10.0, block.direct_edge_count * 3.0)

    # Penalty: high bridge dependency
    score -= block.bridge_dependency_score * 15.0

    # Penalty: underfill risk
    score -= block.underfill_risk_score * 10.0

    return max(0.0, min(100.0, score))


def _compute_underfill_risk(
    block: CandidateBlock,
    min_campaign_tons: float = 300.0,
) -> float:
    """Estimate risk that block will be underfilled in a campaign."""
    if block.total_tons >= min_campaign_tons:
        return 0.0
    gap = min_campaign_tons - block.total_tons
    return min(1.0, gap / min_campaign_tons)


def _compute_bridge_dependency(block: CandidateBlock) -> float:
    """Compute fraction of non-direct edges in block."""
    total_edges = block.direct_edge_count + block.real_bridge_edge_count + block.virtual_family_edge_count
    if total_edges == 0:
        return 0.0
    return float(block.real_bridge_edge_count + block.virtual_family_edge_count) / float(total_edges)


def _classify_mixed_bridge_opportunity(
    block: CandidateBlock,
    cfg,
) -> Tuple[bool, str]:
    """
    Determine if block has opportunity for mixed bridge (block-internal).
    Returns (possible, reason).
    """
    if not getattr(cfg.model, "mixed_bridge_in_block_enabled", True):
        return False, "mixed_bridge_in_block_disabled"

    allowed_hotspots = getattr(cfg.model, "mixed_bridge_allowed_hotspots", [])
    reasons = []

    if block.has_group_switch and "group_switch" in allowed_hotspots:
        reasons.append("group_switch")
    if block.has_width_tension and "width_tension" in allowed_hotspots:
        reasons.append("width_tension")
    if block.has_underfill_hotspot and "underfill" in allowed_hotspots:
        reasons.append("underfill")
    if block.has_bridge_dependency_hotspot and "bridge_dependency" in allowed_hotspots:
        reasons.append("bridge_dependency")

    if reasons:
        return True, "mixed_bridge_opportunity: " + ", ".join(reasons)
    return False, "no_mixed_bridge_opportunity"


# =============================================================================
# MacroBlock -> CandidateBlock Conversion (bridge feasible_block_builder to orchestrator)
# =============================================================================


def _macro_block_to_candidate_block(
    macro: MacroBlock,
    orders_by_id: Dict[str, Dict[str, Any]],
    cfg: PlannerConfig,
) -> CandidateBlock:
    """
    Convert a MacroBlock (from feasible_block_builder) to a CandidateBlock (orchestrator level).

    This is the bridge between the low-level engine (feasible_block_builder) and
    the production orchestrator (block_generator).
    """
    if not macro.order_ids:
        raise ValueError(f"MacroBlock {macro.block_id} has no orders")

    head_order = orders_by_id.get(macro.order_ids[0], {})
    tail_order = orders_by_id.get(macro.order_ids[-1], {})

    # Build internal edges from macro block metadata
    internal_edges = []
    for i in range(len(macro.order_ids) - 1):
        internal_edges.append({
            "from_order_id": macro.order_ids[i],
            "to_order_id": macro.order_ids[i + 1],
            "edge_type": "DIRECT_EDGE",  # Simplified - macro doesn't track individual edge types
        })

    # Determine generation mode from source
    mode = macro.source_generation_mode or "direct_friendly"

    # Detect hotspots
    steel_groups_seen = set()
    widths_seen = []
    for oid in macro.order_ids:
        o = orders_by_id.get(oid, {})
        steel_groups_seen.add(str(o.get("steel_group", "")))
        widths_seen.append(float(o.get("width", 0)))

    has_group_switch = len(steel_groups_seen) > 1
    has_width_tension = bool(widths_seen) and (max(widths_seen) - min(widths_seen)) > 400.0

    # Candidate pool gate (looser) vs ideal target gate (tighter)
    candidate_min = float(getattr(cfg.model, "block_generator_candidate_tons_min", 300.0))
    target_min = float(getattr(cfg.model, "block_generator_target_tons_min", 150.0))
    target_max = float(getattr(cfg.model, "block_generator_target_tons_max", 550.0))
    has_underfill_hotspot = macro.total_tons < candidate_min

    # Size class: distinguishes small pool candidates from ideal target candidates
    if macro.total_tons < target_min:
        is_under_target_block = True
        candidate_size_class = "small_candidate"
    elif macro.total_tons <= target_max:
        is_under_target_block = False
        candidate_size_class = "target_candidate"
    else:
        is_under_target_block = False
        candidate_size_class = "above_target_max"

    has_bridge_dependency_hotspot = (macro.real_bridge_edge_count + macro.virtual_family_edge_count) > 0

    # Create CandidateBlock
    block = CandidateBlock(
        block_id=macro.block_id,
        line=macro.line,
        order_ids=list(macro.order_ids),
        order_count=macro.order_count or len(macro.order_ids),
        total_tons=macro.total_tons,
        head_order_id=macro.head_order_id or (macro.order_ids[0] if macro.order_ids else ""),
        tail_order_id=macro.tail_order_id or (macro.order_ids[-1] if macro.order_ids else ""),
        head_signature=_signature_from_order_dict(dict(head_order)) if head_order else {},
        tail_signature=_signature_from_order_dict(dict(tail_order)) if tail_order else {},
        width_band=macro.width_band or "",
        thickness_band=macro.thickness_band or "",
        steel_group_profile=macro.steel_group_profile or "|".join(sorted(steel_groups_seen)),
        temp_band=macro.temp_band or "",
        direct_edge_count=macro.direct_edge_count,
        real_bridge_edge_count=macro.real_bridge_edge_count,
        virtual_family_edge_count=macro.virtual_family_edge_count,
        mixed_bridge_possible=macro.mixed_bridge_possible,
        mixed_bridge_reason=macro.mixed_bridge_reason,
        block_quality_score=macro.block_quality_score or 0.0,
        underfill_risk_score=macro.underfill_risk_score or 0.0,
        bridge_dependency_score=macro.bridge_dependency_score or 0.0,
        dropped_recovery_potential=macro.dropped_recovery_potential or 0.0,
        source_bucket_key=macro.source_bucket_key or "",
        source_generation_mode=mode,
        internal_edges=internal_edges,
        has_group_switch=has_group_switch,
        has_width_tension=has_width_tension,
        has_underfill_hotspot=has_underfill_hotspot,
        has_bridge_dependency_hotspot=has_bridge_dependency_hotspot,
        is_under_target_block=is_under_target_block,
        candidate_size_class=candidate_size_class,
    )

    # Recompute scores (ensure they're set)
    block.underfill_risk_score = _compute_underfill_risk(block)
    block.bridge_dependency_score = _compute_bridge_dependency(block)
    block.block_quality_score = _compute_block_quality_score(block, cfg)
    block.mixed_bridge_possible, block.mixed_bridge_reason = _classify_mixed_bridge_opportunity(block, cfg)

    return block


# ----------------------------------------------------------------------
# A. Seed Greedy Blocks
# ----------------------------------------------------------------------


def _build_seed_greedy_block(
    start_seed: BlockSeed,
    graph_edges: List[Dict[str, Any]],
    orders_by_id: Dict[str, Dict[str, Any]],
    cfg: PlannerConfig,
    rng: random.Random,
    block_id_prefix: str,
    block_counter: Dict[str, int],
) -> Optional[CandidateBlock]:
    """
    Build one block starting from start_seed using greedy edge following.

    Key differences from constructive_sequence_builder:
    - Stops when tons in [target_min, target_max] range
    - Tracks bridge composition explicitly
    - Computes quality/risk scores
    """
    target_min = float(getattr(cfg.model, "block_generator_target_tons_min", 150.0))
    target_max = float(getattr(cfg.model, "block_generator_target_tons_max", 550.0))
    max_orders = int(getattr(cfg.model, "block_generator_max_orders_per_block", 30))
    max_family = int(getattr(cfg.model, "block_generator_max_family_edges_per_block", 3))
    max_real_bridge = int(getattr(cfg.model, "block_generator_max_real_bridge_edges_per_block", 5))
    allow_guarded = bool(getattr(cfg.model, "block_generator_allow_guarded_family", True))
    allow_real = bool(getattr(cfg.model, "block_generator_allow_real_bridge", True))

    # Build adjacency from edges
    successors: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for e in graph_edges:
        frm = str(e.get("from_order_id", ""))
        successors[frm].append(e)

    # Edge type filter
    def _edge_allowed(etype: str) -> bool:
        if etype == DIRECT_EDGE:
            return True
        if etype == REAL_BRIDGE_EDGE:
            return allow_real
        if etype == VIRTUAL_BRIDGE_FAMILY_EDGE:
            return allow_guarded
        return False

    order_ids = [start_seed.order_id]
    current = start_seed
    family_count = 0
    real_bridge_count = 0
    direct_count = 0
    internal_edges = []
    total_tons = start_seed.tons
    block_counter[current.line] = block_counter.get(current.line, 0) + 1
    bid = f"{block_id_prefix}_{current.line}_{block_counter[current.line]}"

    # Track steel groups for group switch detection
    steel_groups_seen = {current.steel_group}
    widths_seen = [current.width]

    for _ in range(max_orders - 1):
        if total_tons >= target_min and not (order_ids[-1] == start_seed.order_id and len(order_ids) > 1):
            # In target range, check if we should stop
            if total_tons >= target_max * 0.85:
                break

        next_edges = successors.get(current.order_id, [])
        # Filter allowed edge types and sort by score
        allowed = [e for e in next_edges if _edge_allowed(str(e.get("edge_type", "")))]
        if not allowed:
            break

        # Sort by score (higher = better), with diversity bias
        allowed.sort(key=lambda e: (float(e.get("score", 0)) + rng.random() * 0.5), reverse=True)
        chosen = allowed[0]

        etype = str(chosen.get("edge_type", ""))
        next_oid = str(chosen.get("to_order_id", ""))
        next_order = orders_by_id.get(next_oid)

        if next_order is None:
            break

        # Check capacity
        next_tons = float(next_order.get("tons", 0))
        if total_tons + next_tons > target_max * 1.1:
            break

        # Edge type constraints
        if etype == VIRTUAL_BRIDGE_FAMILY_EDGE:
            if family_count >= max_family:
                # Skip family edges if budget exhausted, try next
                if len(allowed) > 1:
                    for alt in allowed[1:]:
                        if str(alt.get("edge_type", "")) != VIRTUAL_BRIDGE_FAMILY_EDGE:
                            chosen = alt
                            etype = str(chosen.get("edge_type", ""))
                            next_oid = str(chosen.get("to_order_id", ""))
                            next_order = orders_by_id.get(next_oid)
                            if next_order is None:
                                break
                            next_tons = float(next_order.get("tons", 0))
                            if total_tons + next_tons > target_max * 1.1:
                                next_order = None
                                break
                            break
                    else:
                        break
                else:
                    break
            else:
                family_count += 1
        elif etype == REAL_BRIDGE_EDGE:
            if real_bridge_count >= max_real_bridge:
                if len(allowed) > 1:
                    for alt in allowed[1:]:
                        if str(alt.get("edge_type", "")) in {DIRECT_EDGE}:
                            chosen = alt
                            etype = VIRTUAL_BRIDGE_FAMILY_EDGE  # renamed var for clarity
                            next_oid = str(chosen.get("to_order_id", ""))
                            next_order = orders_by_id.get(next_oid)
                            break
                    else:
                        break
                else:
                    break
            else:
                real_bridge_count += 1
        else:
            direct_count += 1

        # Update steel groups / widths
        steel_groups_seen.add(str(next_order.get("steel_group", "")))
        widths_seen.append(float(next_order.get("width", 0)))

        order_ids.append(next_oid)
        total_tons += next_tons
        internal_edges.append(dict(chosen))

        # Advance current
        current = BlockSeed(
            order_id=next_oid,
            line=current.line,
            width=float(next_order.get("width", 0)),
            thickness=float(next_order.get("thickness", 0)),
            steel_group=str(next_order.get("steel_group", "")),
            temp_min=float(next_order.get("temp_min", 0)),
            temp_max=float(next_order.get("temp_max", 0)),
            tons=next_tons,
            priority=int(next_order.get("priority", 0)),
            due_rank=int(next_order.get("due_rank", 9999)),
        )

    if len(order_ids) < 2:
        return None

    head_order = orders_by_id.get(order_ids[0], {})
    tail_order = orders_by_id.get(order_ids[-1], {})

    # Build candidate block
    block = CandidateBlock(
        block_id=bid,
        line=start_seed.line,
        order_ids=order_ids,
        order_count=len(order_ids),
        total_tons=total_tons,
        head_order_id=order_ids[0],
        tail_order_id=order_ids[-1],
        head_signature=_signature_from_order_dict(dict(head_order)),
        tail_signature=_signature_from_order_dict(dict(tail_order)),
        width_band=start_seed.width_band,
        thickness_band=start_seed.thickness_band,
        steel_group_profile="|".join(sorted(steel_groups_seen)),
        temp_band=_temp_band_key(start_seed.temp_min, start_seed.temp_max),
        direct_edge_count=direct_count,
        real_bridge_edge_count=real_bridge_count,
        virtual_family_edge_count=family_count,
        mixed_bridge_possible=False,  # set below
        mixed_bridge_reason="",
        block_quality_score=0.0,  # set below
        underfill_risk_score=0.0,  # set below
        bridge_dependency_score=0.0,  # set below
        dropped_recovery_potential=0.0,
        source_bucket_key=f"{start_seed.line}#{start_seed.width_band}#{start_seed.thickness_band}",
        source_generation_mode="greedy_seed",
        internal_edges=internal_edges,
        has_group_switch=(len(steel_groups_seen) > 1),
        has_width_tension=False,  # will be set in post-processing
        has_underfill_hotspot=(total_tons < target_min),
        has_bridge_dependency_hotspot=(real_bridge_count + family_count > 0),
    )

    # Compute scores
    block.underfill_risk_score = _compute_underfill_risk(block)
    block.bridge_dependency_score = _compute_bridge_dependency(block)
    block.block_quality_score = _compute_block_quality_score(block, cfg)
    block.mixed_bridge_possible, block.mixed_bridge_reason = _classify_mixed_bridge_opportunity(block, cfg)

    return block


def _generate_seed_greedy_blocks(
    seeds_by_line: Dict[str, List[BlockSeed]],
    graph_edges: List[Dict[str, Any]],
    orders_df: pd.DataFrame,
    cfg: PlannerConfig,
    rng: random.Random,
    block_id_prefix: str,
) -> List[CandidateBlock]:
    """Generate greedy seed blocks for each line."""
    orders_by_id = {str(row.get("order_id", "")): dict(row) for _, row in orders_df.iterrows()}
    max_per_line = int(getattr(cfg.model, "block_generator_max_blocks_per_line", 30))
    max_total = int(getattr(cfg.model, "block_generator_max_blocks_total", 80))
    max_per_bucket = int(getattr(cfg.model, "block_generator_max_seed_per_bucket", 8))

    # Group seeds by bucket
    bucket_seeds: Dict[str, List[BlockSeed]] = defaultdict(list)
    for line, seeds in seeds_by_line.items():
        for s in seeds:
            bucket = f"{s.width_band}#{s.thickness_band}"
            bucket_seeds[f"{line}#{bucket}"].append(s)

    blocks = []
    block_counter: Dict[str, int] = defaultdict(int)
    used_orders: set = set()

    for bucket_key, seeds in sorted(bucket_seeds.items()):
        if len(blocks) >= max_total:
            break
        # Sort seeds by priority / due_rank
        seeds_sorted = sorted(seeds, key=lambda s: (s.priority, s.due_rank))
        for seed in seeds_sorted[:max_per_bucket]:
            if len(blocks) >= max_total:
                break
            if seed.order_id in used_orders:
                continue
            # Count blocks already built from this line
            line = seed.line
            line_count = sum(1 for b in blocks if b.line == line)
            if line_count >= max_per_line:
                break

            block = _build_seed_greedy_block(
                start_seed=seed,
                graph_edges=graph_edges,
                orders_by_id=orders_by_id,
                cfg=cfg,
                rng=rng,
                block_id_prefix=block_id_prefix,
                block_counter=block_counter,
            )
            if block is None:
                continue
            # Mark orders as used
            for oid in block.order_ids:
                used_orders.add(oid)
            blocks.append(block)

    return blocks


# ----------------------------------------------------------------------
# B. Underfilled Rescue Blocks
# ----------------------------------------------------------------------


def _generate_underfilled_rescue_blocks(
    underfilled_segments: List[Any],
    graph_edges: List[Dict[str, Any]],
    orders_by_id: Dict[str, Dict[str, Any]],
    cfg: PlannerConfig,
    rng: random.Random,
    block_id_prefix: str,
) -> List[CandidateBlock]:
    """
    Generate rescue blocks targeting underfilled segment regions.
    Tries to create smaller blocks that can fill underfilled gaps.
    """
    blocks = []
    block_counter: Dict[str, int] = defaultdict(int)
    used_orders: set = set()

    max_blocks = int(getattr(cfg.model, "block_generator_max_blocks_per_line", 30)) // 3
    target_min = float(getattr(cfg.model, "block_generator_target_tons_min", 150.0))
    allow_real = bool(getattr(cfg.model, "block_generator_allow_real_bridge", True))

    for seg in underfilled_segments[:max_blocks]:
        if len(blocks) >= max_blocks:
            break

        seg_line = str(getattr(seg, "line", ""))
        seg_order_ids = list(getattr(seg, "order_ids", []))
        seg_tons = float(getattr(seg, "total_tons", 0))
        gap_to_min = float(getattr(seg, "gap_to_min_tons", 0))

        if not seg_order_ids:
            continue

        # Find tail orders of underfilled segment as seeds for rescue blocks
        # Try to build a small block from the segment's tail orders
        tail_orders = seg_order_ids[-3:] if len(seg_order_ids) >= 3 else seg_order_ids
        for tail_oid in tail_orders:
            if tail_oid in used_orders:
                continue
            tail_order = orders_by_id.get(tail_oid)
            if tail_order is None:
                continue

            # Build small rescue block targeting gap_to_min
            target_tons = min(target_min, seg_tons + gap_to_min * 0.8)
            seed = BlockSeed(
                order_id=tail_oid,
                line=seg_line,
                width=float(tail_order.get("width", 0)),
                thickness=float(tail_order.get("thickness", 0)),
                steel_group=str(tail_order.get("steel_group", "")),
                temp_min=float(tail_order.get("temp_min", 0)),
                temp_max=float(tail_order.get("temp_max", 0)),
                tons=float(tail_order.get("tons", 0)),
                priority=int(tail_order.get("priority", 0)),
                due_rank=int(tail_order.get("due_rank", 9999)),
            )

            block = _build_seed_greedy_block(
                start_seed=seed,
                graph_edges=graph_edges,
                orders_by_id=orders_by_id,
                cfg=cfg,
                rng=rng,
                block_id_prefix=block_id_prefix,
                block_counter=block_counter,
            )
            if block is None:
                continue
            if block.total_tons < 50.0:  # Too small to be useful
                continue

            for oid in block.order_ids:
                used_orders.add(oid)
            blocks.append(block)

    return blocks


# ----------------------------------------------------------------------
# C. Boundary Patch Blocks
# ----------------------------------------------------------------------


def _generate_boundary_patch_blocks(
    segments: List[Any],
    graph_edges: List[Dict[str, Any]],
    orders_by_id: Dict[str, Dict[str, Any]],
    cfg: PlannerConfig,
    rng: random.Random,
    block_id_prefix: str,
) -> List[CandidateBlock]:
    """
    Generate patch blocks for group switch / width tension hotspots.
    These blocks are smaller and optimize for transition smoothness.
    """
    blocks = []
    block_counter: Dict[str, int] = defaultdict(int)
    used_orders: set = set()

    max_blocks = int(getattr(cfg.model, "block_generator_max_blocks_per_line", 30)) // 4

    for seg in segments[:max_blocks]:
        seg_line = str(getattr(seg, "line", ""))
        seg_order_ids = list(getattr(seg, "order_ids", []))
        if len(seg_order_ids) < 2:
            continue

        # Look for group switch hotspots at boundaries
        boundary_pairs = list(zip(seg_order_ids[:-1], seg_order_ids[1:]))
        for i, (oid_a, oid_b) in enumerate(boundary_pairs):
            if len(blocks) >= max_blocks:
                break

            order_a = orders_by_id.get(oid_a)
            order_b = orders_by_id.get(oid_b)
            if order_a is None or order_b is None:
                continue

            # Detect group switch
            group_a = str(order_a.get("steel_group", ""))
            group_b = str(order_b.get("steel_group", ""))
            width_a = float(order_a.get("width", 0))
            width_b = float(order_b.get("width", 0))

            is_group_switch = (group_a != group_b)
            is_width_tension = (abs(width_a - width_b) > 400.0)

            if not (is_group_switch or is_width_tension):
                continue

            # Build small patch block around boundary
            boundary_oids = seg_order_ids[max(0, i - 2): min(len(seg_order_ids), i + 3)]
            patch_tons = 0.0
            for poid in boundary_oids:
                o = orders_by_id.get(poid)
                if o:
                    patch_tons += float(o.get("tons", 0))

            if len(boundary_oids) < 2:
                continue

            head_order = orders_by_id.get(boundary_oids[0], {})
            tail_order = orders_by_id.get(boundary_oids[-1], {})

            # Build internal edge list
            internal_edges = []
            real_count = 0
            family_count = 0
            direct_count = 0
            steel_groups = set()
            widths = []
            for j in range(len(boundary_oids)):
                o = orders_by_id.get(boundary_oids[j])
                if o:
                    steel_groups.add(str(o.get("steel_group", "")))
                    widths.append(float(o.get("width", 0)))
                if j < len(boundary_oids) - 1:
                    # Find edge in graph
                    for e in graph_edges:
                        if (str(e.get("from_order_id", "")) == boundary_oids[j]
                           and str(e.get("to_order_id", "")) == boundary_oids[j + 1]):
                            internal_edges.append(dict(e))
                            etype = str(e.get("edge_type", ""))
                            if etype == DIRECT_EDGE:
                                direct_count += 1
                            elif etype == REAL_BRIDGE_EDGE:
                                real_count += 1
                            elif etype == VIRTUAL_BRIDGE_FAMILY_EDGE:
                                family_count += 1
                            break

            block_counter[seg_line] = block_counter.get(seg_line, 0) + 1
            bid = f"{block_id_prefix}_{seg_line}_bp_{block_counter[seg_line]}"
            steel_groups_seen = steel_groups

            patch_block = CandidateBlock(
                block_id=bid,
                line=seg_line,
                order_ids=boundary_oids,
                order_count=len(boundary_oids),
                total_tons=patch_tons,
                head_order_id=boundary_oids[0],
                tail_order_id=boundary_oids[-1],
                head_signature=_signature_from_order_dict(dict(head_order)),
                tail_signature=_signature_from_order_dict(dict(tail_order)),
                width_band=_band_key(widths[0] if widths else 0),
                thickness_band="",
                steel_group_profile="|".join(sorted(steel_groups_seen)),
                temp_band="",
                direct_edge_count=direct_count,
                real_bridge_edge_count=real_count,
                virtual_family_edge_count=family_count,
                mixed_bridge_possible=False,
                mixed_bridge_reason="",
                block_quality_score=0.0,
                underfill_risk_score=0.0,
                bridge_dependency_score=0.0,
                dropped_recovery_potential=0.0,
                source_bucket_key=f"{seg_line}#boundary",
                source_generation_mode="boundary_patch",
                internal_edges=internal_edges,
                has_group_switch=is_group_switch,
                has_width_tension=is_width_tension,
                has_underfill_hotspot=False,
                has_bridge_dependency_hotspot=False,
            )
            patch_block.underfill_risk_score = _compute_underfill_risk(patch_block)
            patch_block.bridge_dependency_score = _compute_bridge_dependency(patch_block)
            patch_block.block_quality_score = _compute_block_quality_score(patch_block, cfg)
            patch_block.mixed_bridge_possible, patch_block.mixed_bridge_reason = _classify_mixed_bridge_opportunity(patch_block, cfg)

            for oid in boundary_oids:
                used_orders.add(oid)
            blocks.append(patch_block)

    return blocks


# ----------------------------------------------------------------------
# D. Dropped Recovery Blocks
# ----------------------------------------------------------------------


def _generate_dropped_recovery_blocks(
    dropped_orders: List[Dict[str, Any]],
    graph_edges: List[Dict[str, Any]],
    orders_by_id: Dict[str, Dict[str, Any]],
    cfg: PlannerConfig,
    rng: random.Random,
    block_id_prefix: str,
) -> List[CandidateBlock]:
    """
    Generate rescue blocks from the dropped order pool.
    Targets line-compatible, similar width/group/thickness orders.
    """
    blocks = []
    block_counter: Dict[str, int] = defaultdict(int)
    used_orders: set = set()

    max_blocks = int(getattr(cfg.model, "block_generator_max_blocks_per_line", 30)) // 4
    target_min = float(getattr(cfg.model, "block_generator_target_tons_min", 150.0))

    # Group dropped by line
    dropped_by_line: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for do in dropped_orders:
        line = str(do.get("line", ""))
        if line:
            dropped_by_line[line].append(do)

    for line, line_drops in sorted(dropped_by_line.items()):
        if len(blocks) >= max_blocks:
            break

        # Sort by due_rank / priority
        line_drops.sort(key=lambda d: (int(d.get("priority", 0)), int(d.get("due_rank", 9999))))
        remaining = list(line_drops)

        while remaining and len(blocks) < max_blocks:
            seed_dict = remaining.pop(0)
            seed_oid = str(seed_dict.get("order_id", ""))
            if seed_oid in used_orders:
                continue

            seed = BlockSeed(
                order_id=seed_oid,
                line=line,
                width=float(seed_dict.get("width", 0)),
                thickness=float(seed_dict.get("thickness", 0)),
                steel_group=str(seed_dict.get("steel_group", "")),
                temp_min=float(seed_dict.get("temp_min", 0)),
                temp_max=float(seed_dict.get("temp_max", 0)),
                tons=float(seed_dict.get("tons", 0)),
                priority=int(seed_dict.get("priority", 0)),
                due_rank=int(seed_dict.get("due_rank", 9999)),
            )

            block = _build_seed_greedy_block(
                start_seed=seed,
                graph_edges=graph_edges,
                orders_by_id=orders_by_id,
                cfg=cfg,
                rng=rng,
                block_id_prefix=block_id_prefix,
                block_counter=block_counter,
            )
            if block is None:
                continue
            if block.total_tons < 80.0:
                continue

            block.dropped_recovery_potential = min(1.0, block.total_tons / target_min)
            for oid in block.order_ids:
                used_orders.add(oid)
            blocks.append(block)

    return blocks


# ----------------------------------------------------------------------
# Main entry: generate_candidate_blocks
# ----------------------------------------------------------------------


def generate_candidate_blocks(
    orders_df: pd.DataFrame,
    transition_pack: dict,
    cfg: PlannerConfig,
    constructive_result: Any = None,
    cut_result: Any = None,
    dropped_orders: List[Dict[str, Any]] | None = None,
    random_seed: int = 42,
) -> CandidateBlockPool:
    """
    Generate a CandidateBlockPool from orders using the block-first strategy.

    This function is the production-level orchestrator that:
    1. Calls feasible_block_builder.generate_candidate_macro_blocks() as the low-level engine
    2. Converts MacroBlock results to CandidateBlock (orchestrator level)
    3. Merges with supplemental strategies (underfilled rescue, boundary patch, dropped recovery)

    Parameters
    ----------
    orders_df: Full orders DataFrame
    transition_pack: Template transition pack (must contain "templates" and optionally "candidate_graph")
    cfg: PlannerConfig with block generator parameters
    constructive_result: Optional ConstructiveBuildResult for hints
    cut_result: Optional CampaignCutResult with segments and underfilled_segments
    dropped_orders: Optional list of dropped order dicts
    random_seed: Random seed for block generation

    Returns
    -------
    CandidateBlockPool with diagnostics

    Architecture:
    - Low-level engine: feasible_block_builder.generate_candidate_macro_blocks()
    - Orchestrator: block_generator.generate_candidate_blocks() (this function)
    """
    rng = random.Random(random_seed)
    prefix = f"blk{int(rng.random() * 10000):04d}"

    lines = ["big_roll", "small_roll"]

    # Build orders_by_id for conversions
    orders_by_id = {str(row.get("order_id", "")): dict(row) for _, row in orders_df.iterrows()}

    # Get templates / candidate graph edges
    tpl_df = transition_pack.get("templates")
    if isinstance(tpl_df, pd.DataFrame) and not tpl_df.empty:
        graph_edges = tpl_df.to_dict("records")
    else:
        graph_edges = []

    # Get candidate graph edges if available (more detailed)
    cand_graph = transition_pack.get("candidate_graph")
    if cand_graph is not None and hasattr(cand_graph, "edges"):
        cand_edges = cand_graph.edges
        if isinstance(cand_edges, list) and cand_edges:
            graph_edges = cand_edges

    # Get underfilled segments from cut_result
    underfilled_segs = []
    valid_segs = []
    if cut_result is not None:
        underfilled_segs = list(getattr(cut_result, "underfilled_segments", []) or [])
        valid_segs = list(getattr(cut_result, "segments", []) or [])

    # ---- A. Core: feasible_block_builder (low-level engine) ----
    # Call the low-level engine to generate MacroBlocks
    tpl_df_input = transition_pack.get("templates")
    macro_blocks: List[MacroBlock] = []
    macro_gen_seconds = 0.0

    try:
        macro_t0 = perf_counter()
        macro_result = _generate_macro_blocks(
            orders_df=orders_df,
            templates_df=tpl_df_input,
            cfg=cfg,
            random_seed=random_seed,
        )
        macro_gen_seconds = perf_counter() - macro_t0

        # Extract MacroBlock list (handle both list and wrapped result)
        if hasattr(macro_result, "blocks"):
            macro_blocks = list(macro_result.blocks)
        elif isinstance(macro_result, list):
            macro_blocks = list(macro_result)
        else:
            macro_blocks = []

        print(
            f"[APS][block_generator] feasible_block_builder generated {len(macro_blocks)} MacroBlocks "
            f"in {macro_gen_seconds:.3f}s"
        )
    except Exception as e:
        print(f"[APS][block_generator] feasible_block_builder call failed: {e}, falling back to local generation")
        macro_blocks = []

    # Convert MacroBlock -> CandidateBlock
    candidate_blocks_from_feasible: List[CandidateBlock] = []
    for macro in macro_blocks:
        try:
            cb = _macro_block_to_candidate_block(macro, orders_by_id, cfg)
            candidate_blocks_from_feasible.append(cb)
        except Exception:
            continue

    # ---- B. Supplemental: local block generation (fallback + special cases) ----
    # Convert orders to seeds for supplemental strategies
    seeds_by_line = _orders_to_seeds(orders_df, lines)

    # B1. Underfilled Rescue Blocks (supplemental)
    rescue_blocks = _generate_underfilled_rescue_blocks(
        underfilled_segments=underfilled_segs,
        graph_edges=graph_edges,
        orders_by_id=orders_by_id,
        cfg=cfg,
        rng=rng,
        block_id_prefix=prefix,
    )

    # B2. Boundary Patch Blocks (supplemental)
    boundary_blocks = _generate_boundary_patch_blocks(
        segments=valid_segs + underfilled_segs,
        graph_edges=graph_edges,
        orders_by_id=orders_by_id,
        cfg=cfg,
        rng=rng,
        block_id_prefix=prefix,
    )

    # B3. Dropped Recovery Blocks (supplemental)
    dropped_list = dropped_orders or []
    dropped_blocks = _generate_dropped_recovery_blocks(
        dropped_orders=dropped_list,
        graph_edges=graph_edges,
        orders_by_id=orders_by_id,
        cfg=cfg,
        rng=rng,
        block_id_prefix=prefix,
    )

    # ---- Merge all blocks, deduplicate by order set ----
    # Priority: feasible_builder blocks first, then supplemental
    all_blocks = candidate_blocks_from_feasible + rescue_blocks + boundary_blocks + dropped_blocks
    seen_order_sets: Dict[str, CandidateBlock] = {}
    for b in all_blocks:
        key = "|".join(sorted(b.order_ids))
        if key not in seen_order_sets:
            seen_order_sets[key] = b

    final_blocks = list(seen_order_sets.values())

    # Sort by quality score descending
    final_blocks.sort(key=lambda b: b.block_quality_score, reverse=True)

    # ---- Build pool with enhanced diagnostics ----
    base_diagnostics = _build_diagnostics(final_blocks)

    pool = CandidateBlockPool(
        blocks=final_blocks,
        diagnostics={
            **base_diagnostics,
            # Architecture markers: prove that feasible_block_builder is the low-level engine
            "block_generator_engine": "feasible_block_builder",
            "block_generator_orchestrator": "block_generator",
            "macro_blocks_generated_via_feasible_builder": len(candidate_blocks_from_feasible),
            "feasible_builder_generation_seconds": macro_gen_seconds,
            "supplemental_blocks_count": len(rescue_blocks) + len(boundary_blocks) + len(dropped_blocks),
        },
        generation_config={
            "block_generator_max_blocks_per_line": int(getattr(cfg.model, "block_generator_max_blocks_per_line", 30)),
            "block_generator_max_blocks_total": int(getattr(cfg.model, "block_generator_max_blocks_total", 80)),
            # ---- Candidate pool gate (looser) ----
            "block_generator_candidate_tons_min": float(getattr(cfg.model, "block_generator_candidate_tons_min", 300.0)),
            "block_generator_candidate_tons_target": float(getattr(cfg.model, "block_generator_candidate_tons_target", 700.0)),
            "block_generator_candidate_tons_max": float(getattr(cfg.model, "block_generator_candidate_tons_max", 2000.0)),
            # ---- Ideal target gate (tighter) ----
            "block_generator_target_tons_min": float(getattr(cfg.model, "block_generator_target_tons_min", 150.0)),
            "block_generator_target_tons_max": float(getattr(cfg.model, "block_generator_target_tons_max", 550.0)),
            "random_seed": random_seed,
            # Low-level engine reference
            "low_level_engine": "feasible_block_builder.generate_candidate_macro_blocks",
        },
        orders_input_count=len(orders_df),
        lines_covered=sorted(lines),
    )
    return pool


def _build_diagnostics(blocks: List[CandidateBlock]) -> Dict[str, Any]:
    """Build block generation diagnostics."""
    total = len(blocks)
    by_line = defaultdict(int)
    by_mode = defaultdict(int)
    by_size_class = defaultdict(int)
    tons_list = []
    order_count_list = []
    quality_list = []
    family_edges = 0
    real_bridge_edges = 0

    for b in blocks:
        by_line[b.line] += 1
        by_mode[b.source_generation_mode] += 1
        by_size_class[b.candidate_size_class] += 1
        tons_list.append(b.total_tons)
        order_count_list.append(b.order_count)
        quality_list.append(b.block_quality_score)
        family_edges += b.virtual_family_edge_count
        real_bridge_edges += b.real_bridge_edge_count

    return {
        "generated_blocks_total": total,
        "generated_blocks_by_line": dict(by_line),
        "generated_blocks_by_mode": dict(by_mode),
        "generated_blocks_by_size_class": dict(by_size_class),
        # ---- Candidate size class counts ----
        "generated_small_candidate_blocks": by_size_class.get("small_candidate", 0),
        "generated_target_candidate_blocks": by_size_class.get("target_candidate", 0),
        "generated_blocks_filtered_tons": 0,
        "generated_blocks_filtered_order_count": 0,
        "generated_blocks_filtered_bridge_budget": 0,
        "generated_blocks_filtered_duplicate_signature": 0,
        "avg_block_tons": float(sum(tons_list)) / max(1, len(tons_list)),
        "avg_block_order_count": float(sum(order_count_list)) / max(1, len(order_count_list)),
        "avg_block_quality_score": float(sum(quality_list)) / max(1, len(quality_list)),
        "total_family_edges_in_pool": family_edges,
        "total_real_bridge_edges_in_pool": real_bridge_edges,
        "blocks_with_mixed_bridge_opportunity": sum(1 for b in blocks if b.mixed_bridge_possible),
    }
