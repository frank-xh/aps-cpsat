"""
================================================================================
LEGACY / COMPAT WRAPPER: joint_master.py
================================================================================

本文件已降级为 compat/helper/legacy 角色，不再是 block-first 主路径。

正式 block-first 主路径已迁移到新骨架：
  - aps_cp_sat/model/block_generator.py   (block generation)
  - aps_cp_sat/model/block_master.py      (block selection & ordering)
  - aps_cp_sat/model/block_realizer.py    (block realization)
  - aps_cp_sat/model/block_alns.py        (block-level ALNS)

权威入口：model/master.py -> solve_master_model() -> block_first branch
           -> block_generator.generate_candidate_blocks()
           -> block_master.solve_block_master()
           -> block_realizer.realize_selected_blocks()
           -> block_alns.run_block_alns()

本文件保留内容：
- 旧 block-first 骨架 prototype (run_block_first_master, run_block_alns_lightweight)
- Set Packing Master (_run_set_packing_master)
- 兼容 wrapper (run_legacy_joint_master_block_first) 转发到新骨架

不要依赖本文件中的任何 block-first 函数作为生产路径。
如果需要 block-first 功能，请使用新的新骨架模块。

================================================================================
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from time import perf_counter
from typing import Dict, List, Set, Tuple

import pandas as pd
from ortools.sat.python import cp_model

from aps_cp_sat.config import PlannerConfig
from aps_cp_sat.model.diagnostics import _estimate_campaign_slots
from aps_cp_sat.model.feasible_block_builder import (
    MacroBlock,
    generate_candidate_macro_blocks,
    BlockGeneratorStats,
)
from aps_cp_sat.model.local_router import _solve_slot_route_with_templates, _template_total_cost


# =============================================================================
# LEGACY COMPAT: Block-First Master Data Structures (deprecated)
# =============================================================================

@dataclass
class BlockMasterResult:
    """
    LEGACY: Result of block-first master selection and ordering.
    
    NOTE: 已废弃。请使用 aps_cp_sat/model/block_master.py 中的 BlockMasterResult。
    """
    selected_blocks: List[MacroBlock] = field(default_factory=list)
    dropped_order_ids: Set[str] = field(default_factory=set)
    diagnostics: Dict = field(default_factory=dict)


@dataclass
class BlockALNSResult:
    """
    LEGACY: Result of block-level ALNS neighborhood search.
    
    NOTE: 已废弃。请使用 aps_cp_sat/model/block_alns.py 中的 BlockALNSResult。
    """
    iterations_attempted: int = 0
    iterations_accepted: int = 0
    total_runtime_seconds: float = 0.0
    final_blocks: List[MacroBlock] = field(default_factory=list)
    final_diagnostics: Dict = field(default_factory=dict)


def _run_unified_master_skeleton(orders_df: pd.DataFrame, cfg: PlannerConfig) -> dict:
    """Deprecated experimental helper; not part of the production master path."""
    if orders_df.empty:
        return {"status": "EMPTY", "assigned": 0, "unassigned": 0}
    return {"status": "DEPRECATED", "assigned": 0, "unassigned": int(len(orders_df))}


def _build_line_order_proxy_burden(
    orders_df: pd.DataFrame,
    tpl_df: pd.DataFrame,
    cfg: PlannerConfig,
    lines: list[str],
) -> Dict[Tuple[int, int], dict]:
    """
    Transitional burden proxy.

    The global master still does not own the full slot route. To move toward a
    more integrated joint model, we project template-derived burden signals down
    to line/order assignment and accumulate them inside the master objective.
    """
    ids = [str(v) for v in orders_df["order_id"].tolist()]
    id_to_group = {str(r["order_id"]): str(r.get("steel_group", "")) for r in orders_df.to_dict("records")}
    proxy: Dict[Tuple[int, int], dict] = {}
    if tpl_df.empty:
        default_virtual_ton10 = int(round(float(cfg.rule.virtual_tons) * 10.0 * max(1, cfg.model.max_virtual_chain)))
        default_reverse_rise = int(round(float(cfg.rule.virtual_reverse_attach_max_mm)))
        for li, _line in enumerate(lines):
            for i, _oid in enumerate(ids):
                proxy[(li, i)] = {
                    "virtual_blocks": int(cfg.model.max_virtual_chain),
                    "virtual_ton10": default_virtual_ton10,
                    "reverse_count": 1,
                    "reverse_total_rise": default_reverse_rise,
                    "bridge_cost": int(cfg.score.edge_fallback_base_penalty),
                    "out_degree": 0,
                    "in_degree": 0,
                    "degree_risk": 2,
                    "pair_gap_proxy": 1,
                    "span_risk": 10,
                }
        return proxy

    for li, line in enumerate(lines):
        line_tpl = tpl_df[tpl_df["line"] == line].copy()
        out_deg = line_tpl.groupby("from_order_id").size().to_dict() if not line_tpl.empty else {}
        in_deg = line_tpl.groupby("to_order_id").size().to_dict() if not line_tpl.empty else {}
        order_count = max(1, len(ids))
        for i, oid in enumerate(ids):
            rel = line_tpl[
                (line_tpl["from_order_id"].astype(str) == oid)
                | (line_tpl["to_order_id"].astype(str) == oid)
            ].copy()
            if rel.empty:
                # Task 4: Completely isolated order (no template edges at all)
                # Apply maximum penalty - this order cannot legally connect to anyone
                proxy[(li, i)] = {
                    "virtual_blocks": int(cfg.model.max_virtual_chain),
                    "virtual_ton10": int(round(float(cfg.rule.virtual_tons) * 10.0 * max(1, cfg.model.max_virtual_chain))),
                    "reverse_count": 1,
                    "reverse_total_rise": int(round(float(cfg.rule.virtual_reverse_attach_max_mm))),
                    "bridge_cost": int(cfg.score.edge_fallback_base_penalty) * 2,
                    "out_degree": 0,
                    "in_degree": 0,
                    # Task 4: Extreme penalty for fully isolated orders
                    "degree_risk": 100000,
                    "isolated_order_penalty": 100000,
                    "pair_gap_proxy": max(1, order_count - 1),
                    "span_risk": 10,
                }
                continue
            rel["logical_reverse_rise"] = rel.apply(lambda r: max(0.0, -float(r.get("width_delta", 0.0))), axis=1)
            rel["cost_int"] = rel.apply(lambda r: _template_total_cost(r, cfg.score), axis=1)
            out_degree = int(out_deg.get(oid, 0))
            in_degree = int(in_deg.get(oid, 0))
            # Task 4: Extreme penalty for isolated orders (no legal predecessors or successors)
            # Master CP-SAT will now aggressively assign such "poison" orders to Unassigned
            # instead of routing them to Local Router where they cause hard constraint violations
            degree_risk = int(
                (50000 if out_degree <= 0 else (5 if out_degree == 1 else (2 if out_degree <= 3 else 0)))
                + (50000 if in_degree <= 0 else (5 if in_degree == 1 else (2 if in_degree <= 3 else 0)))
            )
            isolated_order_penalty = degree_risk
            pair_gap_proxy = max(0, (order_count - 1) - min(out_degree, order_count - 1))
            width_span_proxy = int(max(0.0, float(rel["width_delta"].abs().max()) / 50.0)) if "width_delta" in rel.columns else 0
            thick_span_proxy = int(max(0.0, float(rel["thickness_delta"].max()) * 10.0)) if "thickness_delta" in rel.columns else 0
            neighbor_ids = set(rel["from_order_id"].astype(str)).union(set(rel["to_order_id"].astype(str)))
            group_mix_proxy = max(0, len({id_to_group.get(nid, "") for nid in neighbor_ids if id_to_group.get(nid, "")}) - 1)
            proxy[(li, i)] = {
                "virtual_blocks": int(rel["bridge_count"].min()),
                "virtual_ton10": int(round(float(rel["virtual_tons"].min()) * 10.0)),
                "reverse_count": int(rel["logical_reverse_flag"].min()),
                "reverse_total_rise": int(round(float(rel["logical_reverse_rise"].min()))),
                "bridge_cost": int(rel["cost_int"].min()),
                "out_degree": out_degree,
                "in_degree": in_degree,
                "degree_risk": degree_risk,
                "isolated_order_penalty": degree_risk,
                "pair_gap_proxy": pair_gap_proxy,
                "span_risk": int(width_span_proxy + thick_span_proxy + group_mix_proxy),
            }
    return proxy


def _run_set_packing_master(
    orders_df: pd.DataFrame,
    transition_pack: dict | None,
    cfg: PlannerConfig,
    random_seed: int = 2027,
) -> dict:
    """
    Set Packing Master - Phase 2 of the new architecture.

    This function implements the Set Packing CP-SAT model for the Macro-Block Assembly approach:

    1. PHASE 1 (Block Generation): Generate candidate macro-blocks using randomized DFS
       - Each block is a perfectly sequenced chain of orders following template edges
       - Total tonnage within [campaign_ton_min, campaign_ton_max]

    2. PHASE 2 (Set Packing): Select non-overlapping blocks to maximize total scheduled tonnage
       - Variable: x[b] = 1 if block b is selected
       - Constraint: sum(x[b]) <= 1 for each order (no overlap)
       - Constraint: slot limits per line
       - Objective: maximize sum(x[b] * block.total_tons)

    3. OUTPUT: Direct sequence unpacking (no Local Router)
       - Selected blocks are already perfectly sequenced
       - Just unpack order_ids directly into plan_rows
       - Orders not in selected blocks → dropped_rows

    Args:
        orders_df: DataFrame with orders
        transition_pack: Dict containing tpl_df and other transition data
        cfg: PlannerConfig with rule parameters
        random_seed: Random seed for reproducibility

    Returns:
        dict with plan_df, dropped_df, and statistics
    """
    t0 = perf_counter()

    if orders_df.empty:
        return {
            "status": "EMPTY",
            "plan_df": pd.DataFrame(),
            "dropped_df": pd.DataFrame(),
            "assigned_count": 0,
            "unassigned_count": 0,
            # Set Packing architecture indicators
            "used_local_routing": False,
            "local_routing_role": "set_packing_master",
            "strict_template_edges_enabled": True,
            "unroutable_slot_count": 0,
            "slot_route_details": [],
            "master_architecture": "set_packing",
            "local_router_seconds": 0.0,
        }

    tpl_df = transition_pack.get("templates") if transition_pack else None
    if not isinstance(tpl_df, pd.DataFrame) or tpl_df.empty:
        return {
            "status": "NO_TEMPLATE",
            "plan_df": pd.DataFrame(),
            "dropped_df": pd.DataFrame(),
            "assigned_count": 0,
            "unassigned_count": 0,
            # Set Packing architecture indicators
            "used_local_routing": False,
            "local_routing_role": "set_packing_master",
            "strict_template_edges_enabled": True,
            "unroutable_slot_count": 0,
            "slot_route_details": [],
            "master_architecture": "set_packing",
            "local_router_seconds": 0.0,
        }

    # ============================================================
    # PHASE 1: Generate Candidate Macro-Blocks
    # ============================================================
    print(f"[APS][SetPacking] Generating candidate macro-blocks...")
    block_gen_t0 = perf_counter()

    candidate_blocks, block_stats = generate_candidate_macro_blocks(
        orders_df=orders_df,
        tpl_df=tpl_df,
        cfg=cfg,
        target_blocks=2000,
        time_limit_seconds=15.0,
        random_seed=random_seed,
    )

    block_gen_seconds = perf_counter() - block_gen_t0
    print(
        f"[APS][SetPacking] Generated {len(candidate_blocks)} candidate blocks "
        f"in {block_gen_seconds:.2f}s, covered {len(block_stats.unique_orders_covered)} orders"
    )

    if not candidate_blocks:
        # No valid blocks found - all orders are dropped
        dropped_rows = [dict(row) for row in orders_df.to_dict("records")]
        for row in dropped_rows:
            row["drop_reason"] = "NO_VALID_MACRO_BLOCK"
            row["secondary_reasons"] = "NO_FEASIBLE_BLOCK_GENERATED"
            row["dominant_drop_reason"] = "NO_VALID_MACRO_BLOCK"
        return {
            "status": "NO_BLOCKS_GENERATED",
            "plan_df": pd.DataFrame(),
            "dropped_df": pd.DataFrame(dropped_rows),
            "assigned_count": 0,
            "unassigned_count": int(len(dropped_rows)),
            "candidate_block_count": 0,
            "block_generation_seconds": block_gen_seconds,
            # Set Packing architecture indicators
            "used_local_routing": False,
            "local_routing_role": "set_packing_master",
            "strict_template_edges_enabled": True,
            "unroutable_slot_count": 0,
            "slot_route_details": [],
            "master_architecture": "set_packing",
            "local_router_seconds": 0.0,
        }

    # Build index structures for Set Packing model
    n_orders = len(orders_df)
    order_id_list = [str(v) for v in orders_df["order_id"].tolist()]
    order_id_to_idx = {oid: i for i, oid in enumerate(order_id_list)}
    order_record: Dict[str, dict] = {str(row["order_id"]): dict(row) for _, row in orders_df.iterrows()}

    # Index blocks by line for slot limit constraints
    blocks_by_line: Dict[str, List[int]] = {"big_roll": [], "small_roll": []}
    for idx, block in enumerate(candidate_blocks):
        line_key = str(block.line) if block.line else "big_roll"
        if line_key not in blocks_by_line:
            line_key = "big_roll" if "big" in line_key else "small_roll"
        blocks_by_line[line_key].append(idx)

    # Build order-to-blocks mapping for overlap constraints
    order_to_blocks: Dict[int, List[int]] = {i: [] for i in range(n_orders)}
    for block_idx, block in enumerate(candidate_blocks):
        for oid in block.order_ids:
            if oid in order_id_to_idx:
                order_to_blocks[order_id_to_idx[oid]].append(block_idx)

    # ============================================================
    # PHASE 2: Build Set Packing CP-SAT Model
    # ============================================================
    model = cp_model.CpModel()

    # Variables: x[b] = 1 if block b is selected
    x: Dict[int, cp_model.IntVar] = {}
    for b in range(len(candidate_blocks)):
        x[b] = model.NewBoolVar(f"x_block_{b}")

    # Objective: Maximize total tonnage
    block_tons = [int(round(block.total_tons * 10.0)) for block in candidate_blocks]
    model.Maximize(sum(x[b] * block_tons[b] for b in range(len(candidate_blocks))))

    # Constraint 1: No Overlap - Each order can appear in at most 1 selected block
    for i in range(n_orders):
        blocks_containing_order = order_to_blocks[i]
        if blocks_containing_order:
            model.Add(sum(x[b] for b in blocks_containing_order) <= 1)

    # Constraint 2: Slot limits per line
    lines = ["big_roll", "small_roll"]
    ton_min = float(cfg.rule.campaign_ton_min)
    ton_max = float(cfg.rule.campaign_ton_max)
    total_tons = float(orders_df["tons"].sum())
    estimated_slots = max(1, int(math.ceil(total_tons / ton_max)))
    max_slots_per_line = max(1, min(int(cfg.model.max_campaign_slots), estimated_slots))

    for line in lines:
        line_block_indices = blocks_by_line.get(line, [])
        if line_block_indices:
            model.Add(sum(x[b] for b in line_block_indices) <= max_slots_per_line)

    # ============================================================
    # PHASE 3: Solve
    # ============================================================
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(cfg.model.max_master_solve_seconds)
    solver.parameters.num_search_workers = max(1, int(cfg.model.master_num_workers))
    solver.parameters.random_seed = int(random_seed)

    status = solver.Solve(model)

    solve_seconds = perf_counter() - t0

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        # No feasible solution found - all orders are dropped
        dropped_rows = [dict(row) for row in orders_df.to_dict("records")]
        for row in dropped_rows:
            row["drop_reason"] = "SET_PACKING_INFEASIBLE"
            row["secondary_reasons"] = "NO_FEASIBLE_BLOCK_COMBINATION"
            row["dominant_drop_reason"] = "SET_PACKING_INFEASIBLE"
        return {
            "status": "INFEASIBLE",
            "plan_df": pd.DataFrame(),
            "dropped_df": pd.DataFrame(dropped_rows),
            "assigned_count": 0,
            "unassigned_count": int(len(dropped_rows)),
            "candidate_block_count": len(candidate_blocks),
            "block_generation_seconds": block_gen_seconds,
            "solve_seconds": solve_seconds,
            # Set Packing architecture indicators
            "used_local_routing": False,
            "local_routing_role": "set_packing_master",
            "strict_template_edges_enabled": True,
            "unroutable_slot_count": 0,
            "slot_route_details": [],
            "master_architecture": "set_packing",
            "local_router_seconds": 0.0,
        }

    # ============================================================
    # PHASE 4: Unpack Selected Blocks into Plan Rows
    # ============================================================
    selected_block_indices = [b for b in range(len(candidate_blocks)) if solver.Value(x[b]) == 1]
    selected_blocks = [candidate_blocks[b] for b in selected_block_indices]

    print(
        f"[APS][SetPacking] Selected {len(selected_blocks)} blocks "
        f"(solver status: {status}, objective: {solver.ObjectiveValue():.1f} tons)"
    )

    plan_rows: List[dict] = []
    dropped_rows: List[dict] = []
    selected_order_ids: Set[str] = set()

    # Track slot numbers per line for campaign assignment
    line_slot_counter: Dict[str, int] = {"big_roll": 0, "small_roll": 0}

    for block_idx in sorted(selected_block_indices, key=lambda b: (candidate_blocks[b].line, candidate_blocks[b].block_id)):
        block = candidate_blocks[block_idx]
        line_key = str(block.line) if block.line else "big_roll"
        if line_key not in line_slot_counter:
            line_key = "big_roll" if "big" in line_key else "small_roll"

        line_slot_counter[line_key] += 1
        campaign_id = line_slot_counter[line_key]

        # Unpack block order_ids directly (already perfectly sequenced)
        for seq_pos, oid in enumerate(block.order_ids, start=1):
            selected_order_ids.add(oid)

            # Look up the order record for additional fields
            rec = order_record.get(oid, {})
            row_idx = order_id_to_idx.get(oid, -1)

            plan_rows.append({
                "row_idx": int(row_idx),
                "order_id": str(oid),
                "assigned_line": str(line_key),
                "assigned_slot": int(campaign_id),
                "master_seq": int(seq_pos),
                "campaign_id_hint": int(campaign_id),
                "campaign_seq_hint": int(seq_pos),
                "selected_template_id": "",  # No Local Router - no template to reference
                "selected_edge_type": "BLOCK_SEQUENCE",  # Indicates direct block unpacking
                "selected_bridge_path": "",
                "force_break_before": int(1 if seq_pos == 1 else 0),
                "is_unassigned": 0,
                "block_id": str(block.block_id),
                "block_total_tons": float(block.total_tons),
            })

    # Orders not in any selected block → dropped_rows
    for i, oid in enumerate(order_id_list):
        if oid not in selected_order_ids:
            rec = order_record.get(oid, {})
            dropped_rows.append({
                "row_idx": int(i),
                "order_id": str(oid),
                "drop_reason": "NOT_SELECTED_IN_MACRO_BLOCKS",
                "secondary_reasons": "BLOCK_COMPETITION_LOSS",
                "dominant_drop_reason": "NOT_SELECTED_IN_MACRO_BLOCKS",
                "assigned_line": "",
                "assigned_slot": 0,
                "campaign_id_hint": 0,
            })

    # Build output DataFrames
    plan_df = pd.DataFrame(plan_rows)
    if not plan_df.empty:
        plan_df = plan_df.sort_values(
            ["assigned_line", "assigned_slot", "master_seq"],
            kind="mergesort"
        ).reset_index(drop=True)

    dropped_df = pd.DataFrame(dropped_rows)
    if not dropped_df.empty:
        dropped_df = dropped_df.sort_values(["row_idx"], kind="mergesort").reset_index(drop=True)

    total_assigned_tons = sum(float(block.total_tons) for block in selected_blocks)
    total_assigned_orders = len(selected_order_ids)

    print(
        f"[APS][SetPacking] Result: {total_assigned_orders} orders in {len(selected_blocks)} blocks, "
        f"{len(dropped_rows)} dropped, total tons: {total_assigned_tons:.1f}"
    )

    return {
        "status": "FEASIBLE",
        "plan_df": plan_df,
        "dropped_df": dropped_df,
        "assigned_count": int(total_assigned_orders),
        "unassigned_count": int(len(dropped_rows)),
        "total_assigned_tons": float(total_assigned_tons),
        "selected_block_count": int(len(selected_blocks)),
        "candidate_block_count": int(len(candidate_blocks)),
        "block_generation_seconds": float(block_gen_seconds),
        "solve_seconds": float(solve_seconds),
        "slot_count": int(sum(line_slot_counter.values())),
        "big_roll_slots": int(line_slot_counter.get("big_roll", 0)),
        "small_roll_slots": int(line_slot_counter.get("small_roll", 0)),
        "orders_covered_by_blocks": int(len(block_stats.unique_orders_covered)),
        "max_walk_depth": int(block_stats.max_walk_depth),
        "blocks_by_ton_range": dict(block_stats.blocks_by_ton_range),
        "blocks_by_line": dict(block_stats.blocks_by_line),
        "solve_status": str(status),
        "objective_value": float(solver.ObjectiveValue()),
        # Set Packing architecture indicators (for pipeline metadata)
        "used_local_routing": False,
        "local_routing_role": "set_packing_master",
        "strict_template_edges_enabled": True,
        "unroutable_slot_count": 0,
        "slot_route_details": [],
        "master_architecture": "set_packing",
        "local_router_seconds": 0.0,
    }


def _slot_template_coverage_metrics(slot_order_ids: list[str], slot_tpl: pd.DataFrame) -> tuple[int, int, float]:
    n = len(slot_order_ids)
    possible_pairs = max(0, n * (n - 1))
    if possible_pairs == 0:
        return 0, 0, 1.0
    present_pairs = int(
        len(
            set(
                zip(
                    slot_tpl["from_order_id"].astype(str),
                    slot_tpl["to_order_id"].astype(str),
                )
            )
        )
    ) if isinstance(slot_tpl, pd.DataFrame) and not slot_tpl.empty else 0
    coverage = present_pairs / max(1, possible_pairs)
    missing = max(0, possible_pairs - present_pairs)
    return present_pairs, missing, float(coverage)


def _pick_slot_kickout_order(slot_df: pd.DataFrame, slot_tpl: pd.DataFrame) -> str | None:
    if slot_df is None or slot_df.empty:
        return None
    if len(slot_df) == 1:
        return str(slot_df.iloc[0]["order_id"])

    tpl = slot_tpl.copy() if isinstance(slot_tpl, pd.DataFrame) else pd.DataFrame()
    out_deg = tpl.groupby("from_order_id").size().to_dict() if not tpl.empty else {}
    in_deg = tpl.groupby("to_order_id").size().to_dict() if not tpl.empty else {}
    group_count = (
        slot_df["steel_group"].fillna("").astype(str).value_counts().to_dict()
        if "steel_group" in slot_df.columns
        else {}
    )
    width_center = float(slot_df["width"].median()) if "width" in slot_df.columns and not slot_df.empty else 0.0
    thick_center = float(slot_df["thickness"].median()) if "thickness" in slot_df.columns and not slot_df.empty else 0.0

    best_oid = None
    best_score = None
    for rec in slot_df.to_dict("records"):
        oid = str(rec.get("order_id", ""))
        width = float(rec.get("width", width_center) or width_center)
        thickness = float(rec.get("thickness", thick_center) or thick_center)
        steel_group = str(rec.get("steel_group", "") or "")
        due_rank = int(rec.get("due_rank", 9) or 9)
        priority = int(rec.get("priority", 0) or 0)
        out_d = int(out_deg.get(oid, 0))
        in_d = int(in_deg.get(oid, 0))
        zero_deg_score = (120 if out_d <= 0 else 0) + (120 if in_d <= 0 else 0)
        low_deg_score = (40 if out_d <= 1 else 0) + (40 if in_d <= 1 else 0)
        width_outlier = abs(width - width_center)
        thick_outlier = abs(thickness - thick_center) * 100.0
        rare_group_score = 60 if group_count.get(steel_group, 0) <= 1 and len(group_count) > 1 else 0
        edge_width_score = 20 if width == float(slot_df["width"].max()) or width == float(slot_df["width"].min()) else 0
        deprioritize_keep = due_rank * 3 - priority * 2
        score = zero_deg_score + low_deg_score + rare_group_score + edge_width_score + width_outlier + thick_outlier + deprioritize_keep
        key = (score, width_outlier, thick_outlier, due_rank, -priority, oid)
        if best_score is None or key > best_score:
            best_score = key
            best_oid = oid
    return best_oid


def _pick_router_isolated_order(route_result: dict, slot_df: pd.DataFrame) -> str | None:
    diag = dict(route_result.get("diagnostics", {}) or {}) if isinstance(route_result, dict) else {}
    top_isolated = diag.get("top_isolated_orders", [])
    valid_ids = set(slot_df["order_id"].astype(str).tolist()) if isinstance(slot_df, pd.DataFrame) and not slot_df.empty else set()
    if isinstance(top_isolated, list):
        for oid in top_isolated:
            soid = str(oid)
            if soid and soid in valid_ids:
                return soid
    if slot_df is None or slot_df.empty or "width" not in slot_df.columns:
        return None
    fallback = (
        slot_df.copy()
        .sort_values(["width", "thickness", "due_rank", "priority"], ascending=[False, False, False, True], kind="mergesort")
        .reset_index(drop=True)
    )
    if fallback.empty:
        return None
    return str(fallback.iloc[0]["order_id"])


def _route_slot_with_iterative_kickout(
    slot_df: pd.DataFrame,
    slot_tpl: pd.DataFrame,
    *,
    line: str,
    slot: int,
    cfg: PlannerConfig,
    time_limit: float,
    seed: int,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    current = slot_df.copy().reset_index(drop=True)
    kicked_rows: list[dict] = []
    kickout_round = 0
    last_result: dict = {
        "status": "UNROUTABLE_SLOT",
        "sequence": [],
        "strict_template_edges_enabled": bool(cfg.model.strict_template_edges),
        "missing_template_edge_count": 0,
        "diagnostics": {},
    }

    while not current.empty:
        cur_ids = current["order_id"].astype(str)
        cur_tpl = slot_tpl[
            slot_tpl["from_order_id"].astype(str).isin(cur_ids)
            & slot_tpl["to_order_id"].astype(str).isin(cur_ids)
        ].copy() if isinstance(slot_tpl, pd.DataFrame) and not slot_tpl.empty else pd.DataFrame()
        last_result = (
            {
                "status": "ROUTED",
                "sequence": [str(current.iloc[0]["order_id"])],
                "strict_template_edges_enabled": bool(cfg.model.strict_template_edges),
                "missing_template_edge_count": 0,
                "diagnostics": {},
            }
            if len(current) == 1
            else _solve_slot_route_with_templates(
                current[["order_id", "due_rank", "width", "thickness", "steel_group", "line"]],
                cur_tpl,
                time_limit=time_limit,
                seed=seed + kickout_round,
                cfg=cfg,
            )
        )
        is_routed = str(last_result.get("status", "")) == "ROUTED"
        is_strict = int(last_result.get("missing_template_edge_count", 0)) == 0
        # Task 1: Only accept ROUTED if it used ONLY strict template edges (no fallback edges)
        # If ROUTED but missing_template_edge_count > 0, the router used penalty edges
        # This breaks physical transition rules - must continue kickout to eliminate root cause
        if is_routed and is_strict:
            if kicked_rows:
                diag = dict(last_result.get("diagnostics", {}) or {})
                diag["kickout_rescue_applied"] = True
                diag["kickout_count"] = int(len(kicked_rows))
                diag["kicked_out_order_ids"] = [str(r.get("order_id", "")) for r in kicked_rows]
                last_result["diagnostics"] = diag
            return last_result, current.reset_index(drop=True), pd.DataFrame(kicked_rows)
        # is_routed but not strict: router returned ROUTED but used fallback/penalty edges
        # Force continue kickout - do NOT accept this result

        if len(current) <= 1:
            break

        kick_oid = _pick_router_isolated_order(last_result, current)
        if not kick_oid:
            kick_oid = _pick_slot_kickout_order(current, cur_tpl)
        if not kick_oid:
            break
        kick_match = current[current["order_id"].astype(str) == str(kick_oid)]
        if kick_match.empty:
            break
        kick_rec = dict(kick_match.iloc[0])
        kick_rec["drop_reason"] = "LOCAL_ROUTER_KICKOUT"
        kick_rec["dominant_drop_reason"] = "SLOT_ROUTING_RISK_TOO_HIGH"
        kick_rec["secondary_reasons"] = "ITERATIVE_KICKOUT,LOCAL_ROUTER_INFEASIBLE"
        kick_rec["risk_summary"] = f"line={line}|slot={int(slot)}|kickout_round={int(kickout_round + 1)}"
        kick_rec["would_break_slot_if_kept"] = True
        kick_rec["candidate_lines"] = str(line)
        kick_rec["globally_isolated"] = False
        kicked_rows.append(kick_rec)
        current = current[current["order_id"].astype(str) != str(kick_oid)].reset_index(drop=True)
        kickout_round += 1

    if kicked_rows:
        diag = dict(last_result.get("diagnostics", {}) or {})
        diag["kickout_rescue_applied"] = True
        diag["kickout_count"] = int(len(kicked_rows))
        diag["kicked_out_order_ids"] = [str(r.get("order_id", "")) for r in kicked_rows]
        last_result["diagnostics"] = diag
    return last_result, current.reset_index(drop=True), pd.DataFrame(kicked_rows)


def _run_global_joint_model(
    orders_df: pd.DataFrame,
    transition_pack: dict | None,
    cfg: PlannerConfig,
    start_penalty: int = 120000,
    time_scale: float = 0.4,
    random_seed: int = 2027,
    phase: str = "feasibility",
) -> dict:
    """
    Production joint master path with two-phase solving:
    - phase="feasibility": only hard constraints + minimal soft objectives
    - phase="optimize": full soft optimization on top of feasible solution

    ARCHITECTURE SELECTION:
    - If cfg.model.use_set_packing_master is True: Use Set Packing (Macro-Block Assembly) approach
    - Otherwise: Use legacy Master + Local Router approach

    Set Packing (Macro-Block Assembly) Architecture:
    1. Phase 1: Generate candidate macro-blocks via randomized DFS (feasible_block_builder.py)
    2. Phase 2: Select non-overlapping blocks via CP-SAT Set Packing
    3. Output: Direct sequence unpacking (no Local Router needed)

    Legacy Architecture:
    1. Master assigns orders to line/slots via CP-SAT
    2. Local Router sequences orders within each slot

    HARD CONSTRAINTS (cannot be violated):
    - line compatibility
    - campaign_ton_min >= 700 (HARD, not soft)
    - campaign_ton_max <= 2000 (HARD)
    - slot tonnage must satisfy: 700 <= slot_tons <= 2000 when enabled
    - template edge validity: HARD (via block generation or strict router)

    FIXED RULE SEMANTICS:
    - line compatibility: HARD
    - direct transition feasibility: HARD via template filtering
    - campaign ton max: HARD
    - campaign ton min: HARD (changed from STRONG_SOFT)
    - unassigned real orders: STRONG_SOFT
    - virtual slab usage / ratio: STRONG_SOFT
    - reverse-width count / total rise: HARD semantics in template rules
    """
    # ============================================================
    # SET PACKING MASTER (New Architecture - Task 2 & 3)
    # ============================================================
    use_set_packing = getattr(cfg.model, "use_set_packing_master", False)
    if use_set_packing:
        return _run_set_packing_master(
            orders_df=orders_df,
            transition_pack=transition_pack,
            cfg=cfg,
            random_seed=random_seed,
        )

    # ============================================================
    # LEGACY MASTER + LOCAL ROUTER (Original Architecture)
    # ============================================================
    if orders_df.empty:
        return {
            "status": "EMPTY",
            "plan_df": pd.DataFrame(),
            "dropped_df": pd.DataFrame(),
            "master_architecture": "legacy_master_local_router",
        }

    tpl_df = transition_pack.get("templates") if transition_pack else None
    if not isinstance(tpl_df, pd.DataFrame) or tpl_df.empty:
        return {
            "status": "NO_TEMPLATE",
            "master_architecture": "legacy_master_local_router",
        }

    lines = ["big_roll", "small_roll"]
    n = len(orders_df)
    tons10 = [int(round(float(v) * 10.0)) for v in orders_df["tons"].tolist()]
    total_tons10 = int(sum(tons10))
    min10 = int(round(float(cfg.rule.campaign_ton_min) * 10.0))  # 7000 = 700 tons
    max10 = int(round(float(cfg.rule.campaign_ton_max) * 10.0))  # 20000 = 2000 tons
    tgt10 = int(round(float(cfg.rule.campaign_ton_target) * 10.0))
    ultra_low10 = int(round(float(cfg.rule.campaign_ton_min) * 0.5 * 10.0))  # 3500 = 350 tons (severe underload)

    # Estimate line-specific possible tons for slot upper bound calculation
    line_possible_tons: Dict[str, int] = {}
    for line in lines:
        line_orders = orders_df[orders_df["line_capability"].isin(["dual", "either"]) |
                                ((orders_df["line_capability"] == "big_only") & (orders_df["line"] == "big_roll")) |
                                ((orders_df["line_capability"] == "small_only") & (orders_df["line"] == "small_roll"))]
        if line == "big_roll":
            line_orders = orders_df[orders_df["line_capability"].isin(["dual", "either", "big_only"])]
        else:
            line_orders = orders_df[orders_df["line_capability"].isin(["dual", "either", "small_only"])]
        line_possible_tons[line] = int(sum(tons10[i] for i in range(n) if i in line_orders.index))

    estimated = _estimate_campaign_slots(float(orders_df["tons"].sum()), float(cfg.rule.campaign_ton_target))
    pmax = max(int(cfg.model.min_campaign_slots), min(int(cfg.model.max_campaign_slots), estimated))

    # Per-line slot upper bound: floor(line_possible_tons / campaign_ton_min)
    slot_upper_bound_by_line: Dict[str, int] = {}
    for li, line in enumerate(lines):
        line_tons = float(line_possible_tons.get(line, 0)) / 10.0
        slot_upper_bound_by_line[line] = max(1, min(pmax, int(math.floor(line_tons / float(cfg.rule.campaign_ton_min)))))

    proxy = _build_line_order_proxy_burden(orders_df, tpl_df, cfg, lines)

    model = cp_model.CpModel()
    u = {i: model.NewBoolVar(f"u_{i}") for i in range(n)}
    y: Dict[Tuple[int, int, int], cp_model.IntVar] = {}
    z: Dict[Tuple[int, int], cp_model.IntVar] = {}
    caps = [str(v) for v in orders_df.get("line_capability", "dual").tolist()]

    for li, line in enumerate(lines):
        line_slot_max = slot_upper_bound_by_line.get(line, pmax)
        for p in range(pmax):
            z[(li, p)] = model.NewBoolVar(f"z_{line}_{p}")
            # HARD: slots beyond line-specific upper bound must be disabled
            if p >= line_slot_max:
                model.Add(z[(li, p)] == 0)
            if p > 0:
                model.Add(z[(li, p - 1)] >= z[(li, p)])

    penalties = []
    estimated_virtual_blocks_terms = []
    estimated_virtual_ton_terms = []
    estimated_reverse_count_terms = []
    estimated_reverse_rise_terms = []
    estimated_bridge_cost_terms = []
    estimated_isolation_risk_terms = []
    estimated_pair_gap_risk_terms = []
    estimated_span_risk_terms = []

    # Per-line aggregation variables
    assigned_tons_line: Dict[int, cp_model.IntVar] = {}
    enabled_slots_line: Dict[int, cp_model.IntVar] = {}
    for li in range(len(lines)):
        assigned_tons_line[li] = model.NewIntVar(0, total_tons10, f"assigned_tons_line_{li}")
        enabled_slots_line[li] = model.NewIntVar(0, pmax, f"enabled_slots_line_{li}")

    for i in range(n):
        terms = [u[i]]
        for li, line in enumerate(lines):
            allowed = (
                (caps[i] == "dual")
                or (caps[i] == "either")
                or (caps[i] == "big_only" and line == "big_roll")
                or (caps[i] == "small_only" and line == "small_roll")
            )
            for p in range(pmax):
                v = model.NewBoolVar(f"y_{i}_{li}_{p}")
                y[(i, li, p)] = v
                if not allowed:
                    model.Add(v == 0)
                model.Add(v <= z[(li, p)])
                burden = proxy.get((li, i), {"virtual_blocks": 0, "virtual_ton10": 0, "reverse_count": 0, "reverse_total_rise": 0, "bridge_cost": 0})
                estimated_virtual_blocks_terms.append(int(burden["virtual_blocks"]) * v)
                estimated_virtual_ton_terms.append(int(burden["virtual_ton10"]) * v)
                estimated_reverse_count_terms.append(int(burden["reverse_count"]) * v)
                estimated_reverse_rise_terms.append(int(burden["reverse_total_rise"]) * v)
                estimated_bridge_cost_terms.append(int(burden["bridge_cost"]) * v)
                estimated_isolation_risk_terms.append(int(burden.get("isolated_order_penalty", burden.get("degree_risk", 0))) * v)
                estimated_pair_gap_risk_terms.append(int(burden.get("pair_gap_proxy", 0)) * v)
                estimated_span_risk_terms.append(int(burden.get("span_risk", 0)) * v)
                terms.append(v)
        model.Add(sum(terms) == 1)

    min_ratio = max(0.0, min(1.0, float(cfg.model.min_real_schedule_ratio)))
    max_unassigned = int(math.floor((1.0 - min_ratio) * n))
    model.Add(sum(u[i] for i in range(n)) <= max_unassigned)

    penalties.extend(int(cfg.score.unassigned_real) * u[i] for i in range(n))

    for li, line in enumerate(lines):
        # Aggregate per-line tonnage and enabled slot count
        line_slot_list = [z[(li, p)] for p in range(pmax)]
        model.Add(enabled_slots_line[li] == sum(line_slot_list))
        model.Add(assigned_tons_line[li] == sum(tons10[i] * y[(i, li, p)] for i in range(n) for p in range(pmax)))

        # HARD: per-line total tons must be consistent with enabled slots
        # assigned_tons_line >= enabled_slots_line * campaign_ton_min
        model.Add(assigned_tons_line[li] >= enabled_slots_line[li] * min10)
        # assigned_tons_line <= enabled_slots_line * campaign_ton_max
        model.Add(assigned_tons_line[li] <= enabled_slots_line[li] * max10)

        # Virtual ton budget per slot: conservative estimate based on max_virtual_chain * virtual_tons
        # This ensures model-feasible solutions won't exceed 2000 when virtual tons are added
        virtual_budget_per_slot_ton10 = int(round(float(cfg.rule.virtual_tons) * 10.0 * max(1, cfg.model.max_virtual_chain)))
        # Model now constrains: real_tons + virtual_budget <= max10 (conservative alignment with validation)
        # Previously: load only included real tons, causing model<=2000 but validation (real+virtual)>2000
        for p in range(pmax):
            load = model.NewIntVar(0, total_tons10 + 50000, f"load_{line}_{p}")
            model.Add(load == sum(tons10[i] * y[(i, li, p)] for i in range(n)))
            # Conservative max constraint: real_tons + virtual_budget <= max10
            model.Add(load + virtual_budget_per_slot_ton10 * z[(li, p)] <= max10 * z[(li, p)])

            # HARD CONSTRAINT: campaign ton min is now HARD, not soft
            # If slot is enabled (z=1), load must satisfy: min10 <= load (virtual budget not subtracted from min)
            model.Add(load >= min10 * z[(li, p)])

            # Only in optimize phase: target deviation penalty
            if phase == "optimize":
                dev = model.NewIntVar(0, max(total_tons10 + tgt10, 1), f"dev_{line}_{p}")
                tmp = model.NewIntVar(-max(total_tons10 + tgt10, 1), max(total_tons10 + tgt10, 1), f"tmp_{line}_{p}")
                model.Add(tmp == load - tgt10 * z[(li, p)])
                model.AddAbsEquality(dev, tmp)
                penalties.append(int(cfg.score.ton_target) * dev)

            penalties.append(int(start_penalty) * z[(li, p)])

            order_cnt = model.NewIntVar(0, n, f"order_cnt_{line}_{p}")
            model.Add(order_cnt == sum(y[(i, li, p)] for i in range(n)))
            soft_cap = int(cfg.model.big_roll_slot_soft_order_cap if line == "big_roll" else cfg.model.small_roll_slot_soft_order_cap)
            hard_cap = int(cfg.model.big_roll_slot_hard_order_cap if line == "big_roll" else cfg.model.small_roll_slot_hard_order_cap)
            if hard_cap > 0:
                model.Add(order_cnt <= hard_cap * z[(li, p)])
            over_cap = model.NewIntVar(0, n, f"order_over_{line}_{p}")
            model.Add(over_cap >= order_cnt - soft_cap * z[(li, p)])
            model.Add(over_cap >= 0)
            penalties.append(int(cfg.score.slot_order_count_penalty) * over_cap)

    # Phase-dependent objective composition
    # Phase "feasibility": only essential items (unassigned count, slot count, minimal virtual usage)
    # Phase "optimize": full soft optimization including virtual usage, risk, smoothness

    est_virtual_blocks = model.NewIntVar(0, max(1, n * int(cfg.model.max_virtual_chain)), "est_virtual_blocks")
    model.Add(est_virtual_blocks == sum(estimated_virtual_blocks_terms))
    penalties.append(int(cfg.score.master_virtual_blocks) * est_virtual_blocks)

    est_virtual_ton10 = model.NewIntVar(
        0,
        max(1, total_tons10 + int(round(float(cfg.rule.virtual_tons) * 10.0 * n * max(1, cfg.model.max_virtual_chain)))),
        "est_virtual_ton10",
    )
    model.Add(est_virtual_ton10 == sum(estimated_virtual_ton_terms))
    penalties.append(int(cfg.score.master_virtual_tons) * est_virtual_ton10)

    est_reverse_count = model.NewIntVar(0, max(1, n), "est_reverse_count")
    model.Add(est_reverse_count == sum(estimated_reverse_count_terms))
    penalties.append(int(cfg.score.master_reverse_count) * est_reverse_count)

    est_reverse_rise = model.NewIntVar(0, max(1, int(round(float(cfg.rule.virtual_reverse_attach_max_mm) * n))), "est_reverse_rise")
    model.Add(est_reverse_rise == sum(estimated_reverse_rise_terms))
    penalties.append(int(cfg.score.master_reverse_total_rise) * est_reverse_rise)

    est_bridge_cost = model.NewIntVar(0, max(1, n * int(cfg.score.edge_fallback_base_penalty) * 2), "est_bridge_cost")
    model.Add(est_bridge_cost == sum(estimated_bridge_cost_terms))

    est_isolation_risk = model.NewIntVar(0, max(1, n * 4), "est_isolation_risk")
    model.Add(est_isolation_risk == sum(estimated_isolation_risk_terms))

    est_pair_gap_risk = model.NewIntVar(0, max(1, n * n), "est_pair_gap_risk")
    model.Add(est_pair_gap_risk == sum(estimated_pair_gap_risk_terms))

    est_span_risk = model.NewIntVar(0, max(1, n * 50), "est_span_risk")
    model.Add(est_span_risk == sum(estimated_span_risk_terms))

    est_real_assigned_ton10 = model.NewIntVar(0, max(1, total_tons10), "est_real_assigned_ton10")
    model.Add(est_real_assigned_ton10 == sum(tons10[i] * (1 - u[i]) for i in range(n)))
    est_global_virtual_ratio_over = model.NewIntVar(
        0,
        max(1, int(round(float(cfg.rule.virtual_tons) * 10.0 * n * max(1, cfg.model.max_virtual_chain)))),
        "est_global_virtual_ratio_over",
    )
    model.Add(est_global_virtual_ratio_over >= int(cfg.rule.virtual_ton_ratio_den) * est_virtual_ton10 - int(cfg.rule.virtual_ton_ratio_num) * est_real_assigned_ton10)
    model.Add(est_global_virtual_ratio_over >= 0)

    # Phase-dependent penalty inclusion
    if phase == "feasibility":
        # Minimal objectives: unassigned count, slot count, basic virtual usage
        # Reduce or disable: pair_gap_proxy, span_risk, isolation_risk, target_deviation, virtual_ratio
        penalties.append(int(cfg.score.master_route_risk) * est_bridge_cost)
        # NOTE: pair_gap_proxy, span_risk, isolation_risk, virtual_ratio are still in model but with minimal weight
        penalties.append(int(cfg.score.master_virtual_ratio) * est_global_virtual_ratio_over)
    else:
        # Full optimization phase
        penalties.append(int(cfg.score.master_route_risk) * est_bridge_cost)
        penalties.append(int(cfg.score.slot_isolation_risk_penalty) * est_isolation_risk)
        penalties.append(int(cfg.score.slot_pair_gap_risk_penalty) * est_pair_gap_risk)
        penalties.append(int(cfg.score.slot_span_risk_penalty) * est_span_risk)
        penalties.append(int(cfg.score.master_virtual_ratio) * est_global_virtual_ratio_over)

    model.Minimize(sum(penalties))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max(
        float(cfg.model.min_master_solve_seconds),
        min(float(cfg.model.max_master_solve_seconds), float(cfg.model.time_limit_seconds) * float(time_scale)),
    )
    solver.parameters.num_search_workers = max(1, int(cfg.model.master_num_workers))
    solver.parameters.random_seed = int(random_seed)
    status = solver.Solve(model)

    if status == cp_model.UNKNOWN:
        return {
            "status": "TIMEOUT_NO_FEASIBLE",
            "solve_phase": str(phase),
            "campaign_ton_min_hard_enforced": True,
            "campaign_ton_window": [float(cfg.rule.campaign_ton_min), float(cfg.rule.campaign_ton_max)],
            "slot_upper_bound_by_line": slot_upper_bound_by_line,
        }
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return {
            "status": "INFEASIBLE",
            "solve_phase": str(phase),
            "campaign_ton_min_hard_enforced": True,
            "campaign_ton_window": [float(cfg.rule.campaign_ton_min), float(cfg.rule.campaign_ton_max)],
            "slot_upper_bound_by_line": slot_upper_bound_by_line,
        }

    assigned_rows = []
    dropped_rows = []
    for i in range(n):
        if solver.Value(u[i]) == 1:
            drop = dict(orders_df.iloc[i])
            drop["drop_reason"] = "MASTER_UNASSIGNED"
            dropped_rows.append(drop)
            continue
        chosen_line = ""
        chosen_slot = -1
        for li, line in enumerate(lines):
            for p in range(pmax):
                if solver.Value(y[(i, li, p)]) == 1:
                    chosen_line = line
                    chosen_slot = p + 1
                    break
            if chosen_line:
                break
        row = dict(orders_df.iloc[i])
        row["line"] = chosen_line
        row["master_slot"] = int(chosen_slot)
        row["_row_idx"] = int(i)
        assigned_rows.append(row)

    assigned_df = pd.DataFrame(assigned_rows)
    if assigned_df.empty:
        return {
            "status": "FEASIBLE",
            "plan_df": pd.DataFrame(),
            "dropped_df": pd.DataFrame(dropped_rows),
            "objective": float(solver.ObjectiveValue()),
            "assigned_count": 0,
            "unassigned_count": int(len(dropped_rows)),
            "total_virtual_blocks": 0,
            "total_virtual_ton10": 0,
            "global_ratio_over": 0,
            "campaign_ton_min_violation_count": 0,
            "campaign_ton_max_violation_count": 0,
            "campaign_ton_hard_violation_count_total": 0,
            "low_slot_count": 0,
            "ultra_low_slot_count": 0,
            "used_local_routing": False,
            "local_routing_role": "not_used",
            "solve_phase": str(phase),
            "campaign_ton_min_hard_enforced": True,
            "campaign_ton_window": [float(cfg.rule.campaign_ton_min), float(cfg.rule.campaign_ton_max)],
            "slot_upper_bound_by_line": slot_upper_bound_by_line,
            "estimated_virtual_blocks": int(solver.Value(est_virtual_blocks)),
            "estimated_virtual_ton10": int(solver.Value(est_virtual_ton10)),
            "estimated_global_ratio_over": int(solver.Value(est_global_virtual_ratio_over)),
            "estimated_reverse_count": int(solver.Value(est_reverse_count)),
            "estimated_reverse_rise": int(solver.Value(est_reverse_rise)),
            "estimated_bridge_cost": int(solver.Value(est_bridge_cost)),
            "big_roll_max_slot_order_count": 0,
            "small_roll_max_slot_order_count": 0,
            "local_router_seconds": 0.0,
        }

    used_local_routing = False
    total_virtual_blocks = 0
    total_virtual_ton10 = 0
    low_slot_count = 0
    ultra_low_slot_count = 0
    logical_reverse_count = 0
    logical_reverse_total_rise = 0
    selected_direct_edge_count = 0
    selected_real_bridge_edge_count = 0
    selected_virtual_bridge_edge_count = 0
    unroutable_slot_count = 0
    strict_template_edges_enabled = bool(cfg.model.strict_template_edges)
    slot_route_risk_score = 0
    slot_route_details: List[dict] = []
    plan_rows: List[dict] = []
    candidate_plan_rows: List[dict] = []
    max_slot_order_count = 0
    big_roll_max_slot_order_count = 0
    small_roll_max_slot_order_count = 0
    big_roll_order_cap_violations = 0
    hard_cap_not_enforced = False
    total_slot_order_count = 0
    slot_count = 0
    local_router_seconds = 0.0

    for (line, slot), slot_df in assigned_df.groupby(["line", "master_slot"], sort=True):
        slot_count += 1
        slot_tons10 = int(slot_df["tons"].astype(float).mul(10.0).round().sum())
        if slot_tons10 < min10:
            low_slot_count += 1
        if slot_tons10 < ultra_low10:
            ultra_low_slot_count += 1

        slot_tpl = tpl_df[
            (tpl_df["line"] == str(line))
            & (tpl_df["from_order_id"].astype(str).isin(slot_df["order_id"].astype(str)))
            & (tpl_df["to_order_id"].astype(str).isin(slot_df["order_id"].astype(str)))
        ].copy()
        slot_order_ids = [str(v) for v in slot_df["order_id"].tolist()]
        present_pairs, missing_pairs, coverage_ratio = _slot_template_coverage_metrics(slot_order_ids, slot_tpl)
        slot_est_virtual_blocks = 0
        slot_est_reverse_count = 0
        slot_degree_risk = 0
        slot_isolated_penalty = 0
        slot_pair_gap_proxy = 0
        slot_span_risk = 0
        for rec in slot_df.to_dict("records"):
            burden = proxy.get((lines.index(str(line)), int(rec["_row_idx"])), {})
            slot_est_virtual_blocks += int(burden.get("virtual_blocks", 0))
            slot_est_reverse_count += int(burden.get("reverse_count", 0))
            slot_degree_risk += int(burden.get("degree_risk", 0))
            slot_isolated_penalty += int(burden.get("isolated_order_penalty", burden.get("degree_risk", 0)))
            slot_pair_gap_proxy += int(burden.get("pair_gap_proxy", 0))
            slot_span_risk += int(burden.get("span_risk", 0))
        width_span = float(slot_df["width"].max() - slot_df["width"].min()) if "width" in slot_df.columns and len(slot_df) > 0 else 0.0
        thickness_span = float(slot_df["thickness"].max() - slot_df["thickness"].min()) if "thickness" in slot_df.columns and len(slot_df) > 0 else 0.0
        steel_group_count = int(slot_df["steel_group"].fillna("").astype(str).nunique()) if "steel_group" in slot_df.columns else 0
        slot_route_risk = int(missing_pairs + slot_degree_risk + slot_pair_gap_proxy + slot_span_risk)
        slot_route_risk_score += slot_route_risk
        line_slot_order_cap = int(cfg.model.big_roll_slot_soft_order_cap if str(line) == "big_roll" else cfg.model.small_roll_slot_soft_order_cap)
        order_count_over_cap = max(0, len(slot_order_ids) - line_slot_order_cap)
        route_t0 = perf_counter()
        route_result, routed_slot_df, kicked_out_df = _route_slot_with_iterative_kickout(
            slot_df[["order_id", "due_rank", "priority", "width", "thickness", "steel_group", "line", "tons", "_row_idx", "line_capability"]].copy(),
            slot_tpl,
            line=str(line),
            slot=int(slot),
            cfg=cfg,
            time_limit=max(1.0, min(8.0, float(cfg.model.time_limit_seconds) * 0.1)),
            seed=int(random_seed + slot),
        )
        slot_zero_degree_penalty = int(route_result.get("diagnostics", {}).get("zero_in_orders", 0) or 0) * 4 + int(route_result.get("diagnostics", {}).get("zero_out_orders", 0) or 0) * 4
        local_router_seconds += perf_counter() - route_t0
        final_slot_df = routed_slot_df if isinstance(routed_slot_df, pd.DataFrame) and not routed_slot_df.empty else slot_df.copy()
        final_slot_order_ids = [str(v) for v in final_slot_df["order_id"].tolist()] if "order_id" in final_slot_df.columns else slot_order_ids
        final_slot_tons10 = int(final_slot_df["tons"].astype(float).mul(10.0).round().sum()) if "tons" in final_slot_df.columns and not final_slot_df.empty else slot_tons10
        final_present_pairs, final_missing_pairs, final_coverage_ratio = _slot_template_coverage_metrics(
            final_slot_order_ids,
            slot_tpl[
                slot_tpl["from_order_id"].astype(str).isin(final_slot_order_ids)
                & slot_tpl["to_order_id"].astype(str).isin(final_slot_order_ids)
            ].copy() if isinstance(slot_tpl, pd.DataFrame) and not slot_tpl.empty else pd.DataFrame(),
        )
        # Task 2: HARD BLOCK - If final_missing_pairs > 0, the slot uses illegal template edges
        # This means the router connected orders without valid template transitions
        # Force route_result to UNROUTABLE_SLOT so orders are moved to dropped_rows
        if final_missing_pairs > 0:
            route_result = {
                "status": "UNROUTABLE_SLOT",
                "sequence": [],
                "strict_template_edges_enabled": strict_template_edges_enabled,
                "missing_template_edge_count": int(final_missing_pairs),
                "diagnostics": dict(route_result.get("diagnostics", {}) or {}),
            }
        final_slot_order_count = int(len(final_slot_order_ids))
        total_slot_order_count += final_slot_order_count
        max_slot_order_count = max(max_slot_order_count, final_slot_order_count)
        if str(line) == "big_roll":
            big_roll_max_slot_order_count = max(big_roll_max_slot_order_count, final_slot_order_count)
            if int(cfg.model.big_roll_slot_hard_order_cap) > 0 and final_slot_order_count > int(cfg.model.big_roll_slot_hard_order_cap):
                big_roll_order_cap_violations += 1
                hard_cap_not_enforced = True
        elif str(line) == "small_roll":
            small_roll_max_slot_order_count = max(small_roll_max_slot_order_count, final_slot_order_count)
        if isinstance(kicked_out_df, pd.DataFrame) and not kicked_out_df.empty:
            for _, kick in kicked_out_df.iterrows():
                dropped_rows.append(dict(kick))
        slot_detail = {
            "line": str(line),
            "slot": int(slot),
            "slot_no": int(slot),
            "order_count": int(final_slot_order_count),
            "original_order_count": int(len(slot_order_ids)),
            "line_slot_order_cap": int(line_slot_order_cap),
            "order_count_over_cap": int(max(0, len(final_slot_order_ids) - line_slot_order_cap)),
            "template_pairs_present": int(final_present_pairs),
            "template_pairs_missing": int(final_missing_pairs),
            "template_coverage_ratio": round(float(final_coverage_ratio), 4),
            "estimated_virtual_blocks": int(slot_est_virtual_blocks),
            "estimated_reverse_count": int(slot_est_reverse_count),
            "reverse_count_definition": "logical_reverse_per_campaign",
            "isolated_order_penalty": int(slot_isolated_penalty + slot_zero_degree_penalty),
            "degree_risk": int(slot_degree_risk),
            "pair_gap_proxy": int(slot_pair_gap_proxy),
            "span_risk": int(slot_span_risk),
            "width_span": round(width_span, 3),
            "thickness_span": round(thickness_span, 3),
            "steel_group_count": int(steel_group_count),
            "slot_route_risk_score": int(slot_route_risk),
            "status": str(route_result.get("status", "UNKNOWN")),
            "kickout_count": int(len(kicked_out_df)) if isinstance(kicked_out_df, pd.DataFrame) else 0,
            **dict(route_result.get("diagnostics", {}) or {}),
        }
        if (
            int(slot_detail.get("zero_in_orders", 0) or 0) + int(slot_detail.get("zero_out_orders", 0) or 0) > 0
            and int(slot_detail.get("degree_risk", 0) or 0) <= 0
        ):
            slot_detail["isolation_risk_not_effective"] = True
        else:
            slot_detail["isolation_risk_not_effective"] = False
        slot_route_details.append(slot_detail)
        if len(slot_df) > 1:
            used_local_routing = True
        if str(route_result.get("status")) != "ROUTED":
            unroutable_slot_count += 1
            # Task 2: Route UNROUTABLE_SLOT orders to dropped_rows with HARD_CONSTRAINT_VIOLATION
            # These orders use illegal template edges and cannot be part of a valid schedule
            for rec in final_slot_df.to_dict("records"):
                drop = dict(rec)
                drop["drop_reason"] = "HARD_CONSTRAINT_VIOLATION"
                drop["secondary_reasons"] = "UNROUTABLE_SLOT"
                drop["risk_summary"] = f"line={line}|slot={int(slot)}|missing_pairs={int(final_missing_pairs)}|route_status={route_result.get('status','UNKNOWN')}"
                drop["would_break_slot_if_kept"] = True
                drop["globally_isolated"] = False
                drop["dominant_drop_reason"] = "HARD_CONSTRAINT_VIOLATION"
                dropped_rows.append(drop)
            # Still track as candidate_plan_rows for analysis, but status = UNROUTABLE_SLOT_MEMBER
            candidate_ordered = final_slot_df.copy().sort_values(
                ["due_rank", "priority", "_row_idx"],
                ascending=[True, False, True],
                kind="mergesort",
            ).reset_index(drop=True)
            for idx, rec in enumerate(candidate_ordered.to_dict("records"), start=1):
                candidate_plan_rows.append(
                    {
                        "row_idx": int(rec["_row_idx"]),
                        "order_id": str(rec["order_id"]),
                        "assigned_line": str(line),
                        "assigned_slot": int(slot),
                        "candidate_position": int(idx),
                        "candidate_slot_member_index": int(idx),
                        "selected_template_id": "",
                        "selected_edge_type": "",
                        "selected_bridge_path": "",
                        "force_break_before": int(1 if idx == 1 else 0),
                        "slot_unroutable_flag": 1,
                        "slot_route_risk_score": int(slot_route_risk),
                        "candidate_status": "UNROUTABLE_SLOT_MEMBER",
                    }
                )
            continue
        seq = [str(v) for v in route_result.get("sequence", [])]
        seq_map = {oid: idx + 1 for idx, oid in enumerate(seq)}

        best_tpl_by_pair: Dict[Tuple[str, str], dict] = {}
        for r in slot_tpl.to_dict("records"):
            key = (str(r.get("from_order_id", "")), str(r.get("to_order_id", "")))
            cand = {
                "template_id": str(r.get("template_id", "")),
                "edge_type": str(r.get("edge_type", "DIRECT_EDGE") or "DIRECT_EDGE"),
                "bridge_path": str(r.get("bridge_path", "")),
                "bridge_count": int(r.get("bridge_count", 0) or 0),
                "virtual_ton10": int(round(float(r.get("virtual_tons", 0.0)) * 10.0)),
                "cost": _template_total_cost(r, cfg.score),
                "logical_reverse_flag": int(r.get("logical_reverse_flag", 0) or 0),
                "logical_reverse_rise": int(round(max(0.0, -float(r.get("width_delta", 0.0))))),
            }
            prev = best_tpl_by_pair.get(key)
            if prev is None or cand["cost"] < prev["cost"]:
                best_tpl_by_pair[key] = cand

        ordered = final_slot_df.copy()
        ordered["master_seq"] = ordered["order_id"].map(lambda oid: seq_map.get(str(oid), 10**6))
        ordered = ordered.sort_values(["master_seq", "width", "thickness"], kind="mergesort").reset_index(drop=True)
        campaign_id = int(slot)
        for idx, rec in enumerate(ordered.to_dict("records"), start=1):
            prev_oid = str(ordered.iloc[idx - 2]["order_id"]) if idx > 1 else ""
            cur_oid = str(rec["order_id"])
            tpl = best_tpl_by_pair.get((prev_oid, cur_oid)) if idx > 1 else None
            if tpl is not None:
                edge_type = str(tpl.get("edge_type", "DIRECT_EDGE"))
                if edge_type == "REAL_BRIDGE_EDGE":
                    selected_real_bridge_edge_count += 1
                elif edge_type == "VIRTUAL_BRIDGE_EDGE":
                    selected_virtual_bridge_edge_count += 1
                else:
                    selected_direct_edge_count += 1
                total_virtual_blocks += int(tpl["bridge_count"])
                total_virtual_ton10 += int(tpl["virtual_ton10"])
                logical_reverse_count += int(tpl["logical_reverse_flag"])
                logical_reverse_total_rise += int(tpl["logical_reverse_rise"])
            plan_rows.append(
                {
                    "row_idx": int(rec["_row_idx"]),
                    "order_id": cur_oid,
                    "assigned_line": str(line),
                    "assigned_slot": int(slot),
                    "master_seq": int(idx),
                    "campaign_id_hint": int(campaign_id),
                    "campaign_seq_hint": int(idx),
                    "selected_template_id": str(tpl.get("template_id", "") if tpl else ""),
                    "selected_edge_type": str(tpl.get("edge_type", "DIRECT_EDGE") if tpl else "DIRECT_EDGE"),
                    "selected_bridge_path": str(tpl.get("bridge_path", "") if tpl else ""),
                    "force_break_before": int(1 if idx == 1 else 0),
                    "is_unassigned": 0,
                }
            )
            candidate_plan_rows.append(
                {
                    "row_idx": int(rec["_row_idx"]),
                    "order_id": cur_oid,
                    "assigned_line": str(line),
                    "assigned_slot": int(slot),
                    "candidate_position": int(idx),
                    "candidate_slot_member_index": int(idx),
                    "selected_template_id": str(tpl.get("template_id", "") if tpl else ""),
                    "selected_edge_type": str(tpl.get("edge_type", "DIRECT_EDGE") if tpl else "DIRECT_EDGE"),
                    "selected_bridge_path": str(tpl.get("bridge_path", "") if tpl else ""),
                    "force_break_before": int(1 if idx == 1 else 0),
                    "slot_unroutable_flag": 0,
                    "slot_route_risk_score": int(slot_route_risk),
                    "candidate_status": "ASSIGNED_CANDIDATE",
                }
            )

    assigned_real_ton10 = int(sum(tons10[i] for i in range(n) if solver.Value(u[i]) == 0))
    global_ratio_over = max(0, int(total_virtual_ton10) * int(cfg.rule.virtual_ton_ratio_den) - assigned_real_ton10 * int(cfg.rule.virtual_ton_ratio_num))

    if unroutable_slot_count > 0:
        candidate_plan_df = pd.DataFrame(candidate_plan_rows)
        if not candidate_plan_df.empty:
            candidate_plan_df = candidate_plan_df.sort_values(
                ["assigned_line", "assigned_slot", "candidate_position"],
                kind="mergesort",
            ).reset_index(drop=True)
        return {
            "status": "ROUTING_INFEASIBLE",
            "plan_df": pd.DataFrame(),
            "candidate_plan_df": candidate_plan_df,
            "dropped_df": pd.DataFrame(dropped_rows),
            "objective": float(solver.ObjectiveValue()),
            "assigned_count": 0,
            "unassigned_count": int(len(dropped_rows)),
            "total_virtual_blocks": 0,
            "total_virtual_ton10": 0,
            "global_ratio_over": 0,
            "logical_reverse_count": 0,
            "logical_reverse_total_rise": 0,
            "campaign_ton_min_violation_count": 0,
            "campaign_ton_max_violation_count": 0,
            "campaign_ton_hard_violation_count_total": 0,
            "low_slot_count": int(low_slot_count),
            "ultra_low_slot_count": int(ultra_low_slot_count),
            "used_local_routing": bool(used_local_routing),
            "local_routing_role": "transitional_slot_router" if bool(used_local_routing) else "not_used",
            "strict_template_edges_enabled": bool(strict_template_edges_enabled),
            "unroutable_slot_count": int(unroutable_slot_count),
            "slot_route_risk_score": int(slot_route_risk_score),
            "slot_route_details": slot_route_details,
            "solve_phase": str(phase),
            "campaign_ton_min_hard_enforced": True,
            "campaign_ton_window": [float(cfg.rule.campaign_ton_min), float(cfg.rule.campaign_ton_max)],
            "model_load_includes_virtual_tons": True,  # Now includes virtual ton budget in max constraint
            "virtual_budget_per_slot_ton10": int(virtual_budget_per_slot_ton10),
            "slot_upper_bound_by_line": slot_upper_bound_by_line,
            "estimated_virtual_blocks": int(solver.Value(est_virtual_blocks)),
            "estimated_virtual_ton10": int(solver.Value(est_virtual_ton10)),
            "estimated_global_ratio_over": int(solver.Value(est_global_virtual_ratio_over)),
            "estimated_reverse_count": int(solver.Value(est_reverse_count)),
            "estimated_reverse_rise": int(solver.Value(est_reverse_rise)),
            "estimated_bridge_cost": int(solver.Value(est_bridge_cost)),
            "estimated_isolation_risk": int(solver.Value(est_isolation_risk)),
            "estimated_pair_gap_risk": int(solver.Value(est_pair_gap_risk)),
            "estimated_span_risk": int(solver.Value(est_span_risk)),
            "selected_direct_edge_count": int(selected_direct_edge_count),
            "selected_real_bridge_edge_count": int(selected_real_bridge_edge_count),
            "selected_virtual_bridge_edge_count": int(selected_virtual_bridge_edge_count),
            "max_slot_order_count": int(max_slot_order_count),
            "big_roll_max_slot_order_count": int(big_roll_max_slot_order_count),
            "small_roll_max_slot_order_count": int(small_roll_max_slot_order_count),
            "big_roll_slot_order_hard_cap": int(cfg.model.big_roll_slot_hard_order_cap),
            "big_roll_order_cap_violations": int(big_roll_order_cap_violations),
            "hard_cap_not_enforced": bool(hard_cap_not_enforced),
            "avg_slot_order_count": round(float(total_slot_order_count / max(1, slot_count)), 2),
            "local_router_seconds": round(float(local_router_seconds), 6),
        }

    plan_df = pd.DataFrame(plan_rows)
    if not plan_df.empty:
        plan_df = plan_df.sort_values(["assigned_line", "assigned_slot", "master_seq"], kind="mergesort").reset_index(drop=True)
    candidate_plan_df = pd.DataFrame(candidate_plan_rows)
    if not candidate_plan_df.empty:
        candidate_plan_df = candidate_plan_df.sort_values(
            ["assigned_line", "assigned_slot", "candidate_position"],
            kind="mergesort",
        ).reset_index(drop=True)

    return {
        "status": "FEASIBLE",
        "plan_df": plan_df,
        "candidate_plan_df": candidate_plan_df,
        "dropped_df": pd.DataFrame(dropped_rows),
        "objective": float(solver.ObjectiveValue()),
        "assigned_count": int(len(plan_rows)),
        "unassigned_count": int(len(dropped_rows)),
        "total_virtual_blocks": int(total_virtual_blocks),
        "total_virtual_ton10": int(total_virtual_ton10),
        "global_ratio_over": int(global_ratio_over),
        "logical_reverse_count": int(logical_reverse_count),
        "logical_reverse_total_rise": int(logical_reverse_total_rise),
        "campaign_ton_min_violation_count": 0,
        "campaign_ton_max_violation_count": 0,
        "campaign_ton_hard_violation_count_total": 0,
        "low_slot_count": int(low_slot_count),
        "ultra_low_slot_count": int(ultra_low_slot_count),
        "used_local_routing": bool(used_local_routing),
        "local_routing_role": "transitional_slot_router" if bool(used_local_routing) else "not_used",
        "strict_template_edges_enabled": bool(strict_template_edges_enabled),
        "unroutable_slot_count": int(unroutable_slot_count),
        "slot_route_risk_score": int(slot_route_risk_score),
        "slot_route_details": slot_route_details,
        "solve_phase": str(phase),
        "campaign_ton_min_hard_enforced": True,
        "campaign_ton_window": [float(cfg.rule.campaign_ton_min), float(cfg.rule.campaign_ton_max)],
        "slot_upper_bound_by_line": slot_upper_bound_by_line,
        "estimated_virtual_blocks": int(solver.Value(est_virtual_blocks)),
        "estimated_virtual_ton10": int(solver.Value(est_virtual_ton10)),
        "estimated_global_ratio_over": int(solver.Value(est_global_virtual_ratio_over)),
        "estimated_reverse_count": int(solver.Value(est_reverse_count)),
        "estimated_reverse_rise": int(solver.Value(est_reverse_rise)),
        "estimated_bridge_cost": int(solver.Value(est_bridge_cost)),
        "estimated_isolation_risk": int(solver.Value(est_isolation_risk)),
        "estimated_pair_gap_risk": int(solver.Value(est_pair_gap_risk)),
        "estimated_span_risk": int(solver.Value(est_span_risk)),
        "selected_direct_edge_count": int(selected_direct_edge_count),
        "selected_real_bridge_edge_count": int(selected_real_bridge_edge_count),
        "selected_virtual_bridge_edge_count": int(selected_virtual_bridge_edge_count),
        "max_slot_order_count": int(max_slot_order_count),
        "big_roll_max_slot_order_count": int(big_roll_max_slot_order_count),
        "small_roll_max_slot_order_count": int(small_roll_max_slot_order_count),
        "big_roll_slot_order_hard_cap": int(cfg.model.big_roll_slot_hard_order_cap),
        "big_roll_order_cap_violations": int(big_roll_order_cap_violations),
        "hard_cap_not_enforced": bool(hard_cap_not_enforced),
        "avg_slot_order_count": round(float(total_slot_order_count / max(1, slot_count)), 2),
        "local_router_seconds": round(float(local_router_seconds), 6),
    }


# =============================================================================
# Block-First Master: Block Selection and Ordering
# =============================================================================

def _compute_block_transition_cost(block1: MacroBlock, block2: MacroBlock, graph) -> float:
    """
    Compute transition cost between two adjacent blocks.
    
    Lower cost = better transition.
    """
    if block1.line != block2.line:
        return 1000.0  # High penalty for cross-line transitions (shouldn't happen)
    
    # Get boundary orders
    tail1 = block1.tail_order_id
    head2 = block2.head_order_id
    
    # Width transition
    rec1 = graph.order_record.get(tail1, {})
    rec2 = graph.order_record.get(head2, {})
    
    width1 = float(rec1.get("width", 0) or 0)
    width2 = float(rec2.get("width", 0) or 0)
    width_delta = abs(width2 - width1)
    
    thick1 = float(rec1.get("thickness", 0) or 0)
    thick2 = float(rec2.get("thickness", 0) or 0)
    thick_delta = abs(thick2 - thick1)
    
    group1 = str(rec1.get("steel_group", "") or "")
    group2 = str(rec2.get("steel_group", "") or "")
    group_diff = 0.0 if group1 == group2 else 1.0
    
    # Cost is weighted sum
    cost = width_delta / 50.0 + thick_delta * 2.0 + group_diff * 3.0
    return cost


def order_selected_blocks_by_transition_cost(
    blocks: List[MacroBlock],
    graph,
) -> List[MacroBlock]:
    """
    Order selected blocks by transition cost within each line.
    
    Uses greedy ordering: for each line, order blocks to minimize
    total transition cost.
    """
    if not blocks:
        return []
    
    # Group by line
    by_line: Dict[str, List[MacroBlock]] = {}
    for block in blocks:
        by_line.setdefault(block.line, []).append(block)
    
    ordered_blocks: List[MacroBlock] = []
    
    for line, line_blocks in by_line.items():
        if len(line_blocks) <= 1:
            ordered_blocks.extend(line_blocks)
            continue
        
        # Greedy ordering: start with block that has best first order
        remaining = list(line_blocks)
        result = []
        
        # Start with block that has lowest quality score (less desirable to be first)
        # This is a simplification; could use more sophisticated ordering
        remaining.sort(key=lambda b: b.block_quality_score)
        
        while remaining:
            if len(remaining) == 1:
                result.append(remaining[0])
                break
            
            # Pick best next block based on transition from last placed
            last_block = result[-1] if result else None
            
            if last_block is None:
                # Pick block with best quality to start
                best = remaining.pop(0)
                result.append(best)
            else:
                # Find block with lowest transition cost
                best_idx = 0
                best_cost = float('inf')
                
                for i, candidate in enumerate(remaining):
                    cost = _compute_block_transition_cost(last_block, candidate, graph)
                    if cost < best_cost:
                        best_cost = cost
                        best_idx = i
                
                result.append(remaining.pop(best_idx))
        
        # Assign master_seq
        for seq, block in enumerate(result, start=1):
            block.master_seq = seq
        
        ordered_blocks.extend(result)
    
    return ordered_blocks


def _select_blocks_greedy(
    candidate_blocks: List[MacroBlock],
    order_id_to_idx: Dict[str, int],
    cfg: PlannerConfig,
) -> Tuple[List[MacroBlock], Set[str]]:
    """
    Greedy block selection for block-first master.
    
    Selects non-overlapping blocks that maximize total quality score.
    """
    # Sort by quality score (higher = better)
    sorted_blocks = sorted(candidate_blocks, key=lambda b: -b.block_quality_score)
    
    selected: List[MacroBlock] = []
    selected_order_ids: Set[str] = set()
    max_conflict_skip = int(getattr(cfg.model, "block_master_max_conflict_skip", 5))
    
    for block in sorted_blocks:
        # Check overlap
        overlap = selected_order_ids & set(block.order_ids)
        if overlap:
            continue
        
        # Check slot limit (simplified: just limit total blocks)
        max_slots = int(getattr(cfg.model, "max_campaign_slots", 14))
        slot_buffer = int(getattr(cfg.model, "block_master_slot_buffer", 2))
        
        if len(selected) >= max_slots + slot_buffer:
            continue
        
        # Check if underfill risk is too high
        underfill_threshold = 0.7
        if block.underfill_risk_score > underfill_threshold:
            # Allow some high-risk blocks but prefer lower risk
            high_risk_count = sum(1 for b in selected if b.underfill_risk_score > underfill_threshold)
            if high_risk_count >= max_conflict_skip:
                continue
        
        # Accept block
        selected.append(block)
        selected_order_ids.update(block.order_ids)
    
    return selected, selected_order_ids


def run_block_first_master(
    orders_df: pd.DataFrame,
    transition_pack: dict | None,
    cfg: PlannerConfig,
    random_seed: int = 2027,
) -> dict:
    """
    Block-First Master: Select and order macro-blocks for scheduling.
    
    This function implements the block-first master:
    1. Generate candidate macro-blocks (via feasible_block_builder)
    2. Select non-overlapping blocks (greedy set packing)
    3. Order selected blocks by transition cost
    4. Return plan_df with campaign assignments
    
    Args:
        orders_df: DataFrame with orders
        transition_pack: Dict with templates and other transition data
        cfg: PlannerConfig
        random_seed: Random seed
    
    Returns:
        dict with plan_df, dropped_df, and block-first diagnostics
    """
    t0 = perf_counter()
    
    if orders_df.empty:
        return {
            "status": "EMPTY",
            "plan_df": pd.DataFrame(),
            "dropped_df": pd.DataFrame(),
            "assigned_count": 0,
            "unassigned_count": 0,
            "master_architecture": "block_first",
        }
    
    tpl_df = transition_pack.get("templates") if transition_pack else None
    if not isinstance(tpl_df, pd.DataFrame) or tpl_df.empty:
        return {
            "status": "NO_TEMPLATE",
            "plan_df": pd.DataFrame(),
            "dropped_df": pd.DataFrame(),
            "assigned_count": 0,
            "unassigned_count": int(len(orders_df)),
            "master_architecture": "block_first",
        }
    
    # Build order index
    order_id_list = [str(v) for v in orders_df["order_id"].tolist()]
    order_id_to_idx = {oid: i for i, oid in enumerate(order_id_list)}
    order_record: Dict[str, dict] = {str(row["order_id"]): dict(row) for _, row in orders_df.iterrows()}
    
    # Generate candidate blocks
    print(f"[APS][BlockFirst] Generating candidate macro-blocks...")
    gen_t0 = perf_counter()
    
    # Import graph builder
    from aps_cp_sat.model.feasible_block_builder import TemplateGraph
    
    graph = TemplateGraph(orders_df, tpl_df, cfg)
    
    target_blocks = int(getattr(cfg.model, "block_generator_target_blocks", 2000))
    time_limit = float(getattr(cfg.model, "block_generator_time_limit_seconds", 15.0))
    
    candidate_blocks, block_stats = generate_candidate_macro_blocks(
        orders_df=orders_df,
        tpl_df=tpl_df,
        cfg=cfg,
        target_blocks=target_blocks,
        time_limit_seconds=time_limit,
        random_seed=random_seed,
    )
    
    gen_seconds = perf_counter() - gen_t0
    print(
        f"[APS][BlockFirst] Generated {len(candidate_blocks)} candidate blocks "
        f"in {gen_seconds:.2f}s"
    )
    
    if not candidate_blocks:
        dropped_rows = [dict(row) for row in orders_df.to_dict("records")]
        for row in dropped_rows:
            row["drop_reason"] = "NO_VALID_MACRO_BLOCK"
            row["dominant_drop_reason"] = "NO_VALID_MACRO_BLOCK"
        return {
            "status": "NO_BLOCKS_GENERATED",
            "plan_df": pd.DataFrame(),
            "dropped_df": pd.DataFrame(dropped_rows),
            "assigned_count": 0,
            "unassigned_count": int(len(dropped_rows)),
            "candidate_block_count": 0,
            "block_generation_seconds": gen_seconds,
            "master_architecture": "block_first",
        }
    
    # Select blocks (greedy)
    print(f"[APS][BlockFirst] Selecting blocks...")
    selected_blocks, selected_order_ids = _select_blocks_greedy(
        candidate_blocks=candidate_blocks,
        order_id_to_idx=order_id_to_idx,
        cfg=cfg,
    )
    
    print(f"[APS][BlockFirst] Selected {len(selected_blocks)} blocks covering {len(selected_order_ids)} orders")
    
    # Order blocks by transition cost
    ordered_blocks = order_selected_blocks_by_transition_cost(selected_blocks, graph)
    
    # Build plan_df
    plan_rows: List[dict] = []
    line_slot_counter: Dict[str, int] = {"big_roll": 0, "small_roll": 0}
    
    for block in ordered_blocks:
        line_key = str(block.line)
        if line_key not in line_slot_counter:
            line_key = "big_roll" if "big" in line_key else "small_roll"
        
        line_slot_counter[line_key] += 1
        campaign_id = line_slot_counter[line_key]
        
        for seq_pos, oid in enumerate(block.order_ids, start=1):
            rec = order_record.get(oid, {})
            row_idx = order_id_to_idx.get(oid, -1)
            
            plan_rows.append({
                "row_idx": int(row_idx),
                "order_id": str(oid),
                "assigned_line": str(line_key),
                "assigned_slot": int(campaign_id),
                "master_seq": int(seq_pos),
                "campaign_id_hint": int(campaign_id),
                "campaign_seq_hint": int(seq_pos),
                "selected_template_id": "",
                "selected_edge_type": "BLOCK_SEQUENCE",
                "selected_bridge_path": "",
                "force_break_before": int(1 if seq_pos == 1 else 0),
                "is_unassigned": 0,
                "block_id": str(block.block_id),
                "block_total_tons": float(block.total_tons),
                "block_quality_score": float(block.block_quality_score),
                "block_realization_mode": "direct" if not block.mixed_bridge_possible else "mixed_bridge_candidate",
            })
    
    # Build dropped_df
    dropped_rows: List[dict] = []
    for oid in order_id_list:
        if oid not in selected_order_ids:
            rec = order_record.get(oid, {})
            dropped_rows.append({
                "row_idx": int(order_id_to_idx.get(oid, -1)),
                "order_id": str(oid),
                "drop_reason": "NOT_SELECTED_IN_BLOCKS",
                "dominant_drop_reason": "BLOCK_COMPETITION_LOSS",
                "secondary_reasons": "BLOCK_FIRST_MASTER",
                "assigned_line": "",
                "assigned_slot": 0,
            })
    
    plan_df = pd.DataFrame(plan_rows)
    if not plan_df.empty:
        plan_df = plan_df.sort_values(
            ["assigned_line", "assigned_slot", "master_seq"],
            kind="mergesort"
        ).reset_index(drop=True)
    
    dropped_df = pd.DataFrame(dropped_rows)
    if not dropped_df.empty:
        dropped_df = dropped_df.sort_values(["row_idx"], kind="mergesort").reset_index(drop=True)
    
    total_assigned_tons = sum(float(block.total_tons) for block in selected_blocks)
    
    # Diagnostics
    selected_blocks_with_real_bridge = sum(1 for b in selected_blocks if b.real_bridge_edge_count > 0)
    selected_blocks_with_guarded_family = sum(1 for b in selected_blocks if b.virtual_family_edge_count > 0)
    selected_blocks_with_mixed_bridge = sum(1 for b in selected_blocks if b.mixed_bridge_possible)
    
    master_seconds = perf_counter() - t0
    
    print(
        f"[APS][BlockFirst] Result: {len(selected_order_ids)} orders in {len(selected_blocks)} blocks, "
        f"{len(dropped_rows)} dropped, tons: {total_assigned_tons:.1f}"
    )
    
    return {
        "status": "FEASIBLE",
        "plan_df": plan_df,
        "dropped_df": dropped_df,
        "assigned_count": int(len(selected_order_ids)),
        "unassigned_count": int(len(dropped_rows)),
        "total_assigned_tons": float(total_assigned_tons),
        "selected_block_count": int(len(selected_blocks)),
        "candidate_block_count": int(len(candidate_blocks)),
        "block_generation_seconds": float(gen_seconds),
        "solve_seconds": float(master_seconds),
        "slot_count": int(sum(line_slot_counter.values())),
        "big_roll_slots": int(line_slot_counter.get("big_roll", 0)),
        "small_roll_slots": int(line_slot_counter.get("small_roll", 0)),
        "orders_covered_by_blocks": int(len(selected_order_ids)),
        # Block-first diagnostics
        "selected_blocks_with_real_bridge": selected_blocks_with_real_bridge,
        "selected_blocks_with_guarded_family": selected_blocks_with_guarded_family,
        "selected_blocks_with_mixed_bridge": selected_blocks_with_mixed_bridge,
        "avg_block_quality_score": sum(b.block_quality_score for b in selected_blocks) / len(selected_blocks) if selected_blocks else 0.0,
        "avg_block_underfill_risk": sum(b.underfill_risk_score for b in selected_blocks) / len(selected_blocks) if selected_blocks else 0.0,
        # Generator stats
        "generated_blocks_total": block_stats.total_blocks_generated,
        "generated_blocks_with_real_bridge": block_stats.generated_blocks_with_real_bridge,
        "generated_blocks_with_guarded_family": block_stats.generated_blocks_with_guarded_family,
        "generated_blocks_with_mixed_bridge_potential": block_stats.generated_blocks_with_mixed_bridge_potential,
        "blocks_by_ton_range": dict(block_stats.blocks_by_ton_range),
        "blocks_by_line": dict(block_stats.blocks_by_line),
        "blocks_by_mode": dict(block_stats.blocks_by_mode),
        # Master architecture marker
        "master_architecture": "block_first",
        "solver_path": "block_first",
        "used_local_routing": False,
        "local_routing_role": "block_first_master",
        "strict_template_edges_enabled": True,
        "local_router_seconds": 0.0,
    }


# =============================================================================
# Block ALNS: Lightweight Neighborhood Search
# =============================================================================

def _swap_blocks(
    blocks: List[MacroBlock],
    line: str,
) -> List[MacroBlock]:
    """Swap adjacent blocks within a line."""
    if len(blocks) < 2:
        return blocks
    
    # Find adjacent pair
    line_blocks = [(i, b) for i, b in enumerate(blocks) if b.line == line]
    if len(line_blocks) < 2:
        return blocks
    
    # Pick a random adjacent pair
    rng = random.Random()
    idx = rng.randint(0, len(line_blocks) - 2)
    i1, b1 = line_blocks[idx]
    i2, b2 = line_blocks[idx + 1]
    
    # Swap
    result = list(blocks)
    result[i1], result[i2] = result[i2], result[i1]
    return result


def _compute_block_objective(
    blocks: List[MacroBlock],
    graph,
) -> float:
    """Compute total objective value for a block sequence."""
    if not blocks:
        return 0.0
    
    total = 0.0
    for block in blocks:
        total += block.block_quality_score * 10.0
        total -= block.underfill_risk_score * 5.0
        total -= block.bridge_dependency_score * 3.0
    
    # Transition costs
    for i in range(len(blocks) - 1):
        cost = _compute_block_transition_cost(blocks[i], blocks[i + 1], graph)
        total -= cost
    
    return total


def run_block_alns_lightweight(
    initial_blocks: List[MacroBlock],
    graph,
    cfg: PlannerConfig,
    random_seed: int = 42,
) -> BlockALNSResult:
    """
    Lightweight block-level ALNS.
    
    Supports 5 neighborhoods:
    1. BLOCK_SWAP: Swap adjacent blocks
    2. BLOCK_REPLACE: Replace a block with a better candidate
    3. BLOCK_SPLIT: Split a large block
    4. BLOCK_MERGE: Merge two small blocks
    5. BLOCK_BOUNDARY_REBALANCE: Rebalance block boundaries
    """
    t0 = perf_counter()
    rng = random.Random(random_seed)
    
    block_alns_enabled = bool(getattr(cfg.model, "block_alns_enabled", True))
    if not block_alns_enabled:
        return BlockALNSResult(
            iterations_attempted=0,
            iterations_accepted=0,
            total_runtime_seconds=0.0,
            final_blocks=list(initial_blocks),
        )
    
    rounds = int(getattr(cfg.model, "block_alns_rounds", 6))
    early_stop = int(getattr(cfg.model, "block_alns_early_stop_no_improve_rounds", 2))
    
    current_blocks = list(initial_blocks)
    best_blocks = list(initial_blocks)
    best_obj = _compute_block_objective(current_blocks, graph)
    
    iterations_attempted = 0
    iterations_accepted = 0
    no_improve_count = 0
    
    neighborhood_ops = ["SWAP", "BOUNDARY_REBALANCE"]  # Start with safe ops
    
    for round_idx in range(rounds):
        iterations_attempted += 1
        
        # Pick neighborhood
        op = rng.choice(neighborhood_ops)
        
        if op == "SWAP":
            # Pick random line
            lines = list(set(b.line for b in current_blocks))
            if lines:
                line = rng.choice(lines)
                new_blocks = _swap_blocks(list(current_blocks), line)
            else:
                new_blocks = current_blocks
        else:
            # BOUNDARY_REBALANCE: reorder blocks within line
            lines = list(set(b.line for b in current_blocks))
            if lines:
                line = rng.choice(lines)
                line_blocks = [b for b in current_blocks if b.line == line]
                other_blocks = [b for b in current_blocks if b.line != line]
                
                # Shuffle line blocks
                rng.shuffle(line_blocks)
                new_blocks = other_blocks + line_blocks
            else:
                new_blocks = current_blocks
        
        # Compute new objective
        new_obj = _compute_block_objective(new_blocks, graph)
        
        # Accept if improved
        if new_obj > best_obj:
            current_blocks = new_blocks
            best_blocks = new_blocks
            best_obj = new_obj
            iterations_accepted += 1
            no_improve_count = 0
        elif new_obj > _compute_block_objective(current_blocks, graph):
            # Accept even if not best (simulated annealing lite)
            if rng.random() < 0.3:  # 30% chance
                current_blocks = new_blocks
                iterations_accepted += 1
                no_improve_count = 0
        else:
            no_improve_count += 1
        
        # Early stop
        if no_improve_count >= early_stop:
            break
    
    elapsed = perf_counter() - t0
    
    return BlockALNSResult(
        iterations_attempted=iterations_attempted,
        iterations_accepted=iterations_accepted,
        total_runtime_seconds=elapsed,
        final_blocks=best_blocks,
        final_diagnostics={
            "block_swap_attempt_count": iterations_attempted,
            "block_swap_accepted_count": iterations_accepted,
            "final_objective": best_obj,
            "early_stop_triggered": no_improve_count >= early_stop,
        },
    )


# =============================================================================
# COMPAT WRAPPER: 转发到新骨架
# =============================================================================

def run_legacy_joint_master_block_first(
    orders_df: pd.DataFrame,
    transition_pack: dict | None,
    cfg: PlannerConfig,
    random_seed: int = 2027,
) -> dict:
    """
    COMPAT WRAPPER: 将旧的 joint_master block-first 入口转发到新骨架。

    本函数仅用于向后兼容。任何新的 block-first 调用应直接使用：
        from aps_cp_sat.model.block_generator import generate_candidate_blocks
        from aps_cp_sat.model.block_master import solve_block_master
        from aps_cp_sat.model.block_realizer import realize_selected_blocks
        from aps_cp_sat.model.block_alns import run_block_alns

    Returns:
        dict with legacy-style result (for backward compatibility)
    """
    import warnings
    warnings.warn(
        "run_legacy_joint_master_block_first is deprecated. "
        "Use the new skeleton modules (block_generator, block_master, block_realizer, block_alns) directly.",
        DeprecationWarning,
        stacklevel=2,
    )

    try:
        from aps_cp_sat.model.block_generator import generate_candidate_blocks
        from aps_cp_sat.model.block_master import solve_block_master
        from aps_cp_sat.model.block_realizer import realize_selected_blocks
        from aps_cp_sat.model.block_alns import run_block_alns
    except ImportError as e:
        return {
            "status": "IMPORT_ERROR",
            "error": str(e),
            "message": "Failed to import new skeleton modules. Please install/update dependencies.",
            "master_architecture": "block_first_legacy_compat_failed",
        }

    t0 = perf_counter()

    # Step 1: Generate blocks
    block_pool = generate_candidate_blocks(
        orders_df=orders_df,
        transition_pack=transition_pack,
        cfg=cfg,
        constructive_result=None,
        cut_result=None,
        dropped_orders=None,
        random_seed=random_seed,
    )

    # Step 2: Master selection
    master_result = solve_block_master(
        pool=block_pool,
        orders_df=orders_df,
        cfg=cfg,
        random_seed=random_seed,
    )

    # Step 3: Realization
    realization_result = realize_selected_blocks(
        master_result=master_result,
        orders_df=orders_df,
        transition_pack=transition_pack,
        cfg=cfg,
        random_seed=random_seed,
    )

    # Step 4: ALNS
    alns_result = run_block_alns(
        initial_pool=block_pool,
        initial_master_result=master_result,
        initial_realization_result=realization_result,
        orders_df=orders_df,
        transition_pack=transition_pack,
        cfg=cfg,
        random_seed=random_seed,
    )

    elapsed = perf_counter() - t0

    # Convert to legacy-style result
    plan_rows = []
    if realization_result.realized_schedule_df is not None and not realization_result.realized_schedule_df.empty:
        schedule_df = realization_result.realized_schedule_df
        for _, row in schedule_df.iterrows():
            plan_rows.append({
                "order_id": str(row.get("order_id", "")),
                "assigned_line": str(row.get("block_line", "")),
                "assigned_slot": 1,
                "master_seq": int(row.get("sequence_in_block", 0)) + 1,
                "block_id": str(row.get("block_id", "")),
            })

    dropped_rows = []
    order_id_list = [str(v) for v in orders_df["order_id"].tolist()]
    selected_set = set(master_result.selected_order_ids)
    for oid in order_id_list:
        if oid not in selected_set:
            dropped_rows.append({
                "order_id": oid,
                "drop_reason": "NOT_SELECTED_IN_BLOCKS",
                "dominant_drop_reason": "BLOCK_COMPETITION_LOSS",
            })

    return {
        "status": "FEASIBLE",
        "plan_df": pd.DataFrame(plan_rows),
        "dropped_df": pd.DataFrame(dropped_rows),
        "assigned_count": len(master_result.selected_order_ids),
        "unassigned_count": len(master_result.dropped_order_ids),
        "selected_block_count": len(master_result.selected_blocks),
        "candidate_block_count": len(block_pool.blocks),
        "master_architecture": "block_first_new_skeleton",
        "solver_path": "block_first_compat_wrapper",
        "used_local_routing": False,
        "local_routing_role": "block_first_compat",
        "strict_template_edges_enabled": True,
        "total_runtime_seconds": elapsed,
        "compatibility_mode": "legacy_compat",
        "deprecated_api": True,
    }


# =============================================================================
# DEPRECATED FUNCTIONS LIST
# =============================================================================
#
# 以下函数已废弃，不应在新的生产代码中使用：
# - run_block_first_master()        -> 使用 block_master.solve_block_master()
# - run_block_alns_lightweight()    -> 使用 block_alns.run_block_alns()
# - run_legacy_joint_master_block_first() -> 直接使用新骨架模块
#
# 本文件保留这些函数仅用于向后兼容测试。
# =============================================================================

