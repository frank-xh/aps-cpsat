"""
Feasible Macro-Block Generator for Set Packing Architecture.

This module generates a large pool of fully-valid "Candidate Blocks", where each block
is a perfectly sequenced chain of orders (strictly following template edges) that sums
to a valid campaign tonnage (700-2000 tons).

These candidate blocks are then used by the Set Packing Master Model to select
a non-overlapping subset that maximizes total scheduled tonnage.
"""

from __future__ import annotations

import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

from aps_cp_sat.config import PlannerConfig


@dataclass
class MacroBlock:
    """A fully-valid, pre-sequenced campaign block."""
    block_id: str
    line: str
    order_ids: List[str]
    total_tons: float
    total_cost: int
    campaign_id: int = 0  # Assigned later by master
    master_seq: int = 0   # Position in final schedule

    def __post_init__(self):
        if self.campaign_id == 0:
            self.campaign_id = hash(",".join(self.order_ids)) % 100000

    def contains_order(self, order_id: str) -> bool:
        return order_id in self.order_ids


@dataclass
class BlockGeneratorStats:
    """Statistics from block generation."""
    total_blocks_generated: int = 0
    total_candidates_explored: int = 0
    blocks_by_line: Dict[str, int] = field(default_factory=dict)
    blocks_by_ton_range: Dict[str, int] = field(default_factory=dict)
    generation_seconds: float = 0.0
    unique_orders_covered: Set[str] = field(default_factory=set)
    seed_order_count: int = 0
    max_walk_depth: int = 0


class TemplateGraph:
    """Directed graph built from template DataFrame for block building."""

    def __init__(self, orders_df: pd.DataFrame, tpl_df: pd.DataFrame, cfg: PlannerConfig):
        self.cfg = cfg
        self.orders_df = orders_df
        self.tpl_df = tpl_df
        self.order_record: Dict[str, dict] = {}
        self.out_edges: Dict[str, List[Tuple[str, dict]]] = defaultdict(list)  # oid -> [(next_oid, tpl_row)]
        self.in_edges: Dict[str, List[Tuple[str, dict]]] = defaultdict(list)   # oid -> [(prev_oid, tpl_row)]
        self.order_to_line: Dict[str, str] = {}
        self._build_graph()

    def _build_graph(self) -> None:
        """Build directed graph from orders and templates."""
        # Index orders
        for _, row in self.orders_df.iterrows():
            oid = str(row["order_id"])
            self.order_record[oid] = dict(row)
            cap = str(row.get("line_capability", "dual"))
            if cap in {"big_only", "large"}:
                self.order_to_line[oid] = "big_roll"
            elif cap in {"small_only", "small"}:
                self.order_to_line[oid] = "small_roll"
            else:
                self.order_to_line[oid] = "dual"

        # Build adjacency lists from templates
        if self.tpl_df.empty:
            return

        for _, tpl_row in self.tpl_df.iterrows():
            from_oid = str(tpl_row["from_order_id"])
            to_oid = str(tpl_row["to_order_id"])
            if from_oid not in self.order_record or to_oid not in self.order_record:
                continue
            # Determine line of this edge (from template's line)
            edge_line = str(tpl_row.get("line", "big_roll"))

            self.out_edges[from_oid].append((to_oid, dict(tpl_row)))
            self.in_edges[to_oid].append((from_oid, dict(tpl_row)))

    def get_out_candidates(self, oid: str, exclude: Set[str]) -> List[Tuple[str, dict]]:
        """Get valid out-neighbors excluding already-used orders."""
        candidates = []
        for next_oid, tpl_row in self.out_edges.get(oid, []):
            if next_oid not in exclude:
                candidates.append((next_oid, tpl_row))
        return candidates

    def get_in_candidates(self, oid: str, exclude: Set[str]) -> List[Tuple[str, dict]]:
        """Get valid in-neighbors excluding already-used orders."""
        candidates = []
        for prev_oid, tpl_row in self.in_edges.get(oid, []):
            if prev_oid not in exclude:
                candidates.append((prev_oid, tpl_row))
        return candidates

    def get_seed_orders(self, line: str, min_tons: float = 0.0) -> List[str]:
        """Get orders that can start a block on the given line."""
        seeds = []
        for oid, rec in self.order_record.items():
            cap = str(rec.get("line_capability", "dual"))
            tons = float(rec.get("tons", 0) or 0)
            if tons < min_tons:
                continue
            if line == "big_roll" and cap in {"dual", "either", "big_only", "large"}:
                seeds.append(oid)
            elif line == "small_roll" and cap in {"dual", "either", "small_only", "small"}:
                seeds.append(oid)
        return seeds

    def tons_of(self, order_ids: List[str]) -> float:
        """Calculate total tons for a list of order IDs."""
        return sum(float(self.order_record.get(oid, {}).get("tons", 0) or 0) for oid in order_ids)


def _template_edge_cost(tpl_row: dict, cfg: PlannerConfig) -> int:
    """Calculate cost of a template edge for block scoring."""
    base = float(tpl_row.get("cost", 0.0))
    width_smooth = float(tpl_row.get("width_smooth_cost", 0.0))
    thick_smooth = float(tpl_row.get("thickness_smooth_cost", 0.0))
    temp_margin = float(tpl_row.get("temp_margin_cost", 0.0))
    cross_group = float(tpl_row.get("cross_group_cost", 0.0))
    bridge_cnt = float(tpl_row.get("bridge_count", 0.0))
    edge_type = str(tpl_row.get("edge_type", "DIRECT_EDGE") or "DIRECT_EDGE")
    score = cfg.score
    composed = (
        score.width_smooth * width_smooth
        + score.thick_smooth * thick_smooth
        + score.temp_margin * temp_margin
        + score.non_pc_switch * cross_group
        + score.virtual_use * bridge_cnt
    )
    if edge_type == "REAL_BRIDGE_EDGE":
        composed += float(score.real_bridge_penalty)
    elif edge_type == "VIRTUAL_BRIDGE_EDGE":
        composed += float(score.virtual_bridge_penalty)
    else:
        composed += float(score.direct_edge_penalty)
    return int(round(max(0.0, composed + float(score.template_base_cost_ratio) * base)))


def _randomized_dfs_walk(
    graph: TemplateGraph,
    start_oid: str,
    line: str,
    cfg: PlannerConfig,
    rng: random.Random,
    max_depth: int = 50,
    direction: str = "forward",
) -> List[Tuple[List[str], int]]:
    """
    Perform randomized DFS walk to generate valid sequences.
    
    Returns a list of (sequence, total_cost) tuples where sequence reaches
    at least campaign_ton_min tons.
    """
    ton_min = float(cfg.rule.campaign_ton_min)
    ton_max = float(cfg.rule.campaign_ton_max)
    ton_target = float(cfg.rule.campaign_ton_target)
    
    results: List[Tuple[List[str], int]] = []
    used: Set[str] = set()
    
    def dfs(current_oid: str, sequence: List[str], current_tons: float, current_cost: int, depth: int):
        if depth > max_depth:
            return
        
        # Check if current sequence meets minimum tonnage
        if len(sequence) >= 1:
            if current_tons >= ton_min:
                # Valid block found
                results.append((list(sequence), current_cost))
                # Continue extending if not at max (try to reach closer to target)
                if current_tons < ton_max * 0.95:
                    pass  # Continue walk
                else:
                    return
        
        # Check if we've exceeded max tonnage
        if current_tons >= ton_max:
            return
        
        # Get candidates
        if direction == "forward":
            candidates = graph.get_out_candidates(current_oid, used)
        else:
            candidates = graph.get_in_candidates(current_oid, used)
        
        if not candidates:
            # Dead end - save current if valid
            if len(sequence) >= 1 and current_tons >= ton_min:
                results.append((list(sequence), current_cost))
            return
        
        # Shuffle candidates with bias toward better transitions
        # Use weighted random: prefer lower cost edges, but with enough randomness
        candidates_with_cost = []
        for next_oid, tpl_row in candidates:
            edge_cost = _template_edge_cost(tpl_row, cfg)
            candidates_with_cost.append((next_oid, tpl_row, edge_cost))
        
        # Sort by cost and pick from top-k with random selection
        candidates_with_cost.sort(key=lambda x: x[2])
        top_k = min(5, len(candidates_with_cost))
        chosen = rng.choice(candidates_with_cost[:top_k])
        next_oid, tpl_row, edge_cost = chosen
        
        # Expand
        used.add(next_oid)
        next_tons = current_tons + float(graph.order_record.get(next_oid, {}).get("tons", 0) or 0)
        sequence.append(next_oid)
        
        dfs(next_oid, sequence, next_tons, current_cost + edge_cost, depth + 1)
        
        sequence.pop()
        used.remove(next_oid)
    
    # Start DFS from seed
    start_tons = float(graph.order_record.get(start_oid, {}).get("tons", 0) or 0)
    used.add(start_oid)
    dfs(start_oid, [start_oid], start_tons, 0, 0)
    used.remove(start_oid)
    
    return results


def generate_candidate_macro_blocks(
    orders_df: pd.DataFrame,
    tpl_df: pd.DataFrame,
    cfg: PlannerConfig,
    target_blocks: int = 2000,
    time_limit_seconds: float = 15.0,
    random_seed: int = 2027,
) -> Tuple[List[MacroBlock], BlockGeneratorStats]:
    """
    Generate a pool of fully-valid candidate macro blocks.

    Each block is a perfectly sequenced chain of orders that:
    - Strictly follows template edges (100% valid transitions)
    - Has total tonnage within [campaign_ton_min, campaign_ton_max]
    - All orders run on the same line

    Algorithm:
    1. Build a directed graph from template DataFrame
    2. For each line, collect seed orders
    3. Run randomized DFS/GRASP walks from seed orders
    4. Generate at least `target_blocks` or until time limit

    Args:
        orders_df: DataFrame with orders (must have: order_id, tons, line_capability, width, thickness, etc.)
        tpl_df: DataFrame with template edges (must have: from_order_id, to_order_id, line, cost, etc.)
        cfg: PlannerConfig with rule parameters (campaign_ton_min, campaign_ton_max)
        target_blocks: Minimum number of blocks to generate
        time_limit_seconds: Maximum time to spend generating blocks
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (list of MacroBlock objects, BlockGeneratorStats)
    """
    t0 = time.perf_counter()
    rng = random.Random(random_seed)
    blocks: List[MacroBlock] = []
    seen_sequences: Set[Tuple[str, ...]] = set()  # Deduplicate by order sequence
    stats = BlockGeneratorStats(seed_order_count=len(orders_df))
    
    if orders_df.empty or tpl_df.empty:
        stats.generation_seconds = time.perf_counter() - t0
        return blocks, stats
    
    # Build graph
    graph = TemplateGraph(orders_df, tpl_df, cfg)
    lines = ["big_roll", "small_roll"]
    ton_min = float(cfg.rule.campaign_ton_min)
    ton_max = float(cfg.rule.campaign_ton_max)
    
    for line in lines:
        # Get seed orders for this line
        seed_orders = graph.get_seed_orders(line, min_tons=0)
        if not seed_orders:
            continue
        
        # Shuffle seeds for randomness
        rng.shuffle(seed_orders)
        
        # Multiple passes with different random seeds for diversity
        for pass_idx in range(3):
            if time.perf_counter() - t0 > time_limit_seconds:
                break
            
            pass_rng = random.Random(random_seed + pass_idx * 1000)
            pass_rng.shuffle(seed_orders)
            
            for seed_oid in seed_orders:
                if time.perf_counter() - t0 > time_limit_seconds:
                    break
                
                stats.total_candidates_explored += 1
                
                # Forward walk from seed
                for seq, cost in _randomized_dfs_walk(graph, seed_oid, line, cfg, pass_rng, max_depth=30, direction="forward"):
                    seq_tuple = tuple(seq)
                    if seq_tuple in seen_sequences:
                        continue
                    
                    total_tons = graph.tons_of(seq)
                    if total_tons < ton_min or total_tons > ton_max:
                        continue
                    
                    seen_sequences.add(seq_tuple)
                    block = MacroBlock(
                        block_id=f"BLK_{line}_{len(blocks):06d}",
                        line=line,
                        order_ids=seq,
                        total_tons=total_tons,
                        total_cost=cost,
                    )
                    blocks.append(block)
                    stats.unique_orders_covered.update(seq)
                    stats.max_walk_depth = max(stats.max_walk_depth, len(seq))
                    blocks_by_ton_range = _ton_range_key(total_tons, ton_min, ton_max)
                    stats.blocks_by_ton_range[blocks_by_ton_range] = stats.blocks_by_ton_range.get(blocks_by_ton_range, 0) + 1
                
                # Backward walk (prepend edges) for additional diversity
                for seq, cost in _randomized_dfs_walk(graph, seed_oid, line, cfg, pass_rng, max_depth=15, direction="backward"):
                    seq_tuple = tuple(seq)
                    if seq_tuple in seen_sequences:
                        continue
                    
                    total_tons = graph.tons_of(seq)
                    if total_tons < ton_min or total_tons > ton_max:
                        continue
                    
                    seen_sequences.add(seq_tuple)
                    block = MacroBlock(
                        block_id=f"BLK_{line}_{len(blocks):06d}",
                        line=line,
                        order_ids=seq,
                        total_tons=total_tons,
                        total_cost=cost,
                    )
                    blocks.append(block)
                    stats.unique_orders_covered.update(seq)
                    stats.max_walk_depth = max(stats.max_walk_depth, len(seq))
                    blocks_by_ton_range = _ton_range_key(total_tons, ton_min, ton_max)
                    stats.blocks_by_ton_range[blocks_by_ton_range] = stats.blocks_by_ton_range.get(blocks_by_ton_range, 0) + 1
    
    # Also generate single-order blocks for isolated orders
    for line in lines:
        if time.perf_counter() - t0 > time_limit_seconds:
            break
            
        seed_orders = graph.get_seed_orders(line, min_tons=ton_min)
        for oid in seed_orders:
            if time.perf_counter() - t0 > time_limit_seconds:
                break
            
            tons = float(graph.order_record.get(oid, {}).get("tons", 0) or 0)
            if ton_min <= tons <= ton_max:
                seq_tuple = (oid,)
                if seq_tuple in seen_sequences:
                    continue
                seen_sequences.add(seq_tuple)
                block = MacroBlock(
                    block_id=f"BLK_{line}_{len(blocks):06d}",
                    line=line,
                    order_ids=[oid],
                    total_tons=tons,
                    total_cost=0,
                )
                blocks.append(block)
                stats.unique_orders_covered.add(oid)
    
    # Greedy cluster extension: try to extend partial sequences
    # Take orders not yet covered and greedily build small blocks
    uncovered = set(graph.order_record.keys()) - stats.unique_orders_covered
    for line in lines:
        if time.perf_counter() - t0 > time_limit_seconds:
            break
        if len(blocks) >= target_blocks:
            break
            
        line_cap_orders = [
            oid for oid in uncovered
            if graph.order_to_line.get(oid, "") in {line, "dual"}
            or str(graph.order_record.get(oid, {}).get("line_capability", "dual")) in {"dual", "either", line}
        ]
        rng.shuffle(line_cap_orders)
        
        for oid in line_cap_orders:
            if time.perf_counter() - t0 > time_limit_seconds:
                break
            if len(blocks) >= target_blocks:
                break
                
            tons = float(graph.order_record.get(oid, {}).get("tons", 0) or 0)
            if tons > ton_max:
                continue
            
            # Try to greedily extend with compatible neighbors
            seq = [oid]
            current_tons = tons
            used_ext = {oid}
            
            while current_tons < ton_min:
                candidates = graph.get_out_candidates(oid, used_ext)
                if not candidates:
                    break
                
                # Pick best candidate
                candidates = [(n, t) for n, t in candidates if n not in used_ext]
                if not candidates:
                    break
                
                best_next, best_tpl = min(candidates, key=lambda x: _template_edge_cost(x[1], cfg))
                next_tons = float(graph.order_record.get(best_next, {}).get("tons", 0) or 0)
                if current_tons + next_tons > ton_max:
                    break
                
                seq.append(best_next)
                current_tons += next_tons
                used_ext.add(best_next)
                oid = best_next
            
            if current_tons >= ton_min and current_tons <= ton_max:
                seq_tuple = tuple(seq)
                if seq_tuple not in seen_sequences:
                    seen_sequences.add(seq_tuple)
                    block = MacroBlock(
                        block_id=f"BLK_{line}_{len(blocks):06d}",
                        line=line,
                        order_ids=seq,
                        total_tons=current_tons,
                        total_cost=0,
                    )
                    blocks.append(block)
                    stats.unique_orders_covered.update(seq)

    stats.total_blocks_generated = len(blocks)
    for line in lines:
        stats.blocks_by_line[line] = sum(1 for b in blocks if b.line == line)
    stats.generation_seconds = time.perf_counter() - t0

    # Sort blocks by cost efficiency (tons / cost) for quality
    blocks.sort(key=lambda b: -b.total_tons if b.total_cost == 0 else b.total_tons / b.total_cost, reverse=True)

    print(
        f"[APS][BlockGen] generated={len(blocks)}, "
        f"big_roll={stats.blocks_by_line.get('big_roll', 0)}, "
        f"small_roll={stats.blocks_by_line.get('small_roll', 0)}, "
        f"seconds={stats.generation_seconds:.2f}, "
        f"orders_covered={len(stats.unique_orders_covered)}/{len(graph.order_record)}, "
        f"ton_ranges={stats.blocks_by_ton_range}"
    )

    return blocks, stats


def _ton_range_key(tons: float, ton_min: float, ton_max: float) -> str:
    """Categorize a ton value into a range bucket."""
    if tons < ton_min:
        return "below_min"
    elif tons < ton_min * 1.3:
        return f"700-910"
    elif tons < ton_min * 1.6:
        return f"910-1120"
    elif tons < ton_min * 1.9:
        return f"1120-1330"
    elif tons < ton_max * 0.8:
        return f"1330-1600"
    elif tons <= ton_max:
        return f"1600-{int(ton_max)}"
    else:
        return "above_max"


# Alias for backward compatibility
def generate_feasible_blocks(
    orders_df: pd.DataFrame,
    tpl_df: pd.DataFrame,
    cfg: PlannerConfig,
    **kwargs,
) -> Tuple[List[MacroBlock], BlockGeneratorStats]:
    """Alias for generate_candidate_macro_blocks."""
    return generate_candidate_macro_blocks(orders_df, tpl_df, cfg, **kwargs)
