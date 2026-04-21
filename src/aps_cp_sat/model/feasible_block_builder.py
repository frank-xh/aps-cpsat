"""
================================================================================
LOW-LEVEL BLOCK GENERATOR ENGINE: feasible_block_builder.py
================================================================================

本文件是 block-first 架构中的 **低层块生成引擎**。

角色划分：
- feasible_block_builder.py: 低层块生成引擎（directional clustering primitives）
  - 提供：MacroBlock 数据结构
  - 提供：hard_cluster_gate(), directed_cluster_distance(), build_seed_buckets()
  - 提供：generate_candidate_macro_blocks() - 底层候选块生成
  - 提供：TemplateGraph - 模板图构建

- block_generator.py: 生产级块池编排器（production-facing orchestrator）
  - 使用 feasible_block_builder 的 primitives
  - 编排：A. Seed Greedy / B. Underfilled Rescue / C. Boundary Patch / D. Dropped Recovery
  - 提供：CandidateBlock, CandidateBlockPool 数据结构
  - 提供：CandidateBlockPool 诊断合并

权威主路径：
  model/master.py -> block_generator.generate_candidate_blocks()
                  -> block_master.solve_block_master()
                  -> block_realizer.realize_selected_blocks()
                  -> block_alns.run_block_alns()

本文件直接给 block_generator.py 提供底层能力，不应被 pipeline 或 master 直接调用。

================================================================================
"""

from __future__ import annotations

import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

from aps_cp_sat.config import PlannerConfig


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class MacroBlock:
    """
    A fully-valid, pre-sequenced campaign block with directional clustering metadata.
    
    Enhanced fields for block-first architecture:
    - head_order_id, tail_order_id: Block boundaries for ordering
    - order_count: Number of orders in block
    - width_band, thickness_band, steel_group_profile, temp_band: Block profile
    - direct_edge_count, real_bridge_edge_count, virtual_family_edge_count: Edge composition
    - mixed_bridge_possible, mixed_bridge_reason: Mixed bridge potential
    - block_quality_score: Overall quality score (higher = better)
    - underfill_risk_score: Risk of being underfilled (higher = riskier)
    - bridge_dependency_score: Dependence on bridges (higher = more dependent)
    - dropped_recovery_potential: Potential to recover dropped orders
    - source_bucket_key: Original bucket for diagnostics
    - source_generation_mode: How this block was generated
    """
    block_id: str
    line: str
    order_ids: List[str]
    total_tons: float
    total_cost: int
    campaign_id: int = 0  # Assigned later by master
    master_seq: int = 0   # Position in final schedule
    # ---- Directional clustering metadata ----
    head_order_id: str = ""
    tail_order_id: str = ""
    order_count: int = 0
    width_band: str = ""
    thickness_band: str = ""
    steel_group_profile: str = ""
    temp_band: str = ""
    direct_edge_count: int = 0
    real_bridge_edge_count: int = 0
    virtual_family_edge_count: int = 0
    mixed_bridge_possible: bool = False
    mixed_bridge_reason: str = ""
    block_quality_score: float = 0.0
    underfill_risk_score: float = 0.0
    bridge_dependency_score: float = 0.0
    dropped_recovery_potential: float = 0.0
    source_bucket_key: str = ""
    source_generation_mode: str = "direct_friendly"  # direct_friendly | real_bridge_friendly | guarded_family_friendly | dropped_recovery_friendly

    def __post_init__(self):
        if self.campaign_id == 0:
            self.campaign_id = hash(",".join(self.order_ids)) % 100000
        if self.order_count == 0:
            self.order_count = len(self.order_ids)
        if not self.head_order_id and self.order_ids:
            self.head_order_id = self.order_ids[0]
        if not self.tail_order_id and self.order_ids:
            self.tail_order_id = self.order_ids[-1]

    def contains_order(self, order_id: str) -> bool:
        return order_id in self.order_ids

    def signature(self) -> str:
        """Block signature for deduplication."""
        return f"{self.line}:{self.head_order_id}:{self.tail_order_id}:{self.width_band}:{self.order_count}"


@dataclass
class BlockGeneratorStats:
    """Statistics from block generation."""
    total_blocks_generated: int = 0
    total_candidates_explored: int = 0
    blocks_by_line: Dict[str, int] = field(default_factory=dict)
    blocks_by_ton_range: Dict[str, int] = field(default_factory=dict)
    blocks_by_mode: Dict[str, int] = field(default_factory=dict)
    generation_seconds: float = 0.0
    unique_orders_covered: Set[str] = field(default_factory=set)
    seed_order_count: int = 0
    seed_buckets_count: int = 0
    max_walk_depth: int = 0
    # ---- Directional clustering diagnostics ----
    generated_blocks_filtered_by_hard_gate: int = 0
    generated_blocks_filtered_by_bridge_budget: int = 0
    generated_blocks_filtered_by_duplicate_signature: int = 0
    generated_blocks_with_real_bridge: int = 0
    generated_blocks_with_guarded_family: int = 0
    generated_blocks_with_mixed_bridge_potential: int = 0
    avg_block_quality_score: float = 0.0
    avg_underfill_risk_score: float = 0.0
    avg_bridge_dependency_score: float = 0.0


# =============================================================================
# Order Banding Utilities
# =============================================================================

def _compute_width_band(width: float) -> str:
    """Compute width band for bucketing."""
    if width < 1000:
        return "W_<1000"
    elif width < 1200:
        return "W_1000-1200"
    elif width < 1400:
        return "W_1200-1400"
    elif width < 1600:
        return "W_1400-1600"
    elif width < 1800:
        return "W_1600-1800"
    elif width < 2000:
        return "W_1800-2000"
    else:
        return "W_>=2000"


def _compute_thickness_band(thickness: float) -> str:
    """Compute thickness band for bucketing."""
    if thickness < 1.5:
        return "T_<1.5"
    elif thickness < 2.0:
        return "T_1.5-2.0"
    elif thickness < 2.5:
        return "T_2.0-2.5"
    elif thickness < 3.0:
        return "T_2.5-3.0"
    elif thickness < 3.5:
        return "T_3.0-3.5"
    else:
        return "T_>=3.5"


def _compute_temp_band(temperature: float) -> str:
    """Compute temperature band for bucketing."""
    if temperature < 200:
        return "TMP_<200"
    elif temperature < 300:
        return "TMP_200-300"
    else:
        return "TMP_>=300"


def _compute_steel_group_coarse(steel_group: str) -> str:
    """Compute coarse steel group profile."""
    group = str(steel_group or "").upper()
    if not group or group == "NAN":
        return "GRP_UNKNOWN"
    if "Q235" in group or "Q345" in group:
        return "GRP_CARBON"
    if "304" in group or "316" in group or "201" in group:
        return "GRP_STAINLESS"
    if "DX51" in group or "DC51" in group:
        return "GRP_COATED"
    return "GRP_OTHER"


# =============================================================================
# Template Graph
# =============================================================================

class TemplateGraph:
    """Directed graph built from template DataFrame for block building."""

    def __init__(self, orders_df: pd.DataFrame, tpl_df: pd.DataFrame, cfg: PlannerConfig):
        self.cfg = cfg
        self.orders_df = orders_df
        self.tpl_df = tpl_df
        self.order_record: Dict[str, dict] = {}
        self.out_edges: Dict[str, List[Tuple[str, dict]]] = defaultdict(list)
        self.in_edges: Dict[str, List[Tuple[str, dict]]] = defaultdict(list)
        self.order_to_line: Dict[str, str] = {}
        self._build_graph()

    def _build_graph(self) -> None:
        """Build directed graph from orders and templates."""
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

        if self.tpl_df.empty:
            return

        for _, tpl_row in self.tpl_df.iterrows():
            from_oid = str(tpl_row["from_order_id"])
            to_oid = str(tpl_row["to_order_id"])
            if from_oid not in self.order_record or to_oid not in self.order_record:
                continue
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

    def edge_type(self, from_oid: str, to_oid: str) -> str:
        """Get edge type between two orders."""
        for _, tpl_row in self.out_edges.get(from_oid, []):
            if str(tpl_row.get("to_order_id", "")) == to_oid:
                return str(tpl_row.get("edge_type", "DIRECT_EDGE") or "DIRECT_EDGE")
        return "DIRECT_EDGE"

    def has_real_bridge(self, from_oid: str, to_oid: str) -> bool:
        """Check if there's a real bridge edge between two orders."""
        return self.edge_type(from_oid, to_oid) == "REAL_BRIDGE_EDGE"

    def has_guarded_family(self, from_oid: str, to_oid: str) -> bool:
        """Check if there's a guarded family edge between two orders."""
        return self.edge_type(from_oid, to_oid) == "VIRTUAL_BRIDGE_FAMILY_EDGE"

    def check_mixed_bridge_potential(self, from_oid: str, to_oid: str) -> Tuple[bool, str]:
        """
        Check if there's mixed bridge potential between two orders.
        
        Returns:
            Tuple of (has_potential, reason)
            reason: "REAL_THEN_FAMILY" | "FAMILY_THEN_REAL" | ""
        """
        # Check direct real bridge
        if self.has_real_bridge(from_oid, to_oid):
            # Check if from_oid has family edge to another order that to_oid could bridge to
            for _, tpl_row in self.out_edges.get(from_oid, []):
                if str(tpl_row.get("edge_type", "")) == "VIRTUAL_BRIDGE_FAMILY_EDGE":
                    return True, "REAL_THEN_FAMILY"
        # Check direct family edge
        if self.has_guarded_family(from_oid, to_oid):
            # Check if from_oid has real bridge to another order
            for _, tpl_row in self.out_edges.get(from_oid, []):
                if str(tpl_row.get("edge_type", "")) == "REAL_BRIDGE_EDGE":
                    return True, "FAMILY_THEN_REAL"
        return False, ""


# =============================================================================
# Hard Feasibility Gate
# =============================================================================

def hard_cluster_gate(
    seed_rec: dict,
    cand_rec: dict,
    line: str,
    cfg: PlannerConfig,
    graph: TemplateGraph,
) -> Tuple[bool, List[str]]:
    """
    Hard feasibility gate for adding a candidate order to a growing block.
    
    Returns:
        Tuple of (passes_gate, failure_reasons)
    """
    failures = []
    
    # 1. Line capability check
    seed_cap = str(seed_rec.get("line_capability", "dual"))
    cand_cap = str(cand_rec.get("line_capability", "dual"))
    if line == "big_roll" and cand_cap in {"small_only", "small"}:
        failures.append("LINE_CAP_INCOMPATIBLE")
    elif line == "small_roll" and cand_cap in {"big_only", "large"}:
        failures.append("LINE_CAP_INCOMPATIBLE")
    
    # 2. Width delta check (allow reasonable width transitions)
    seed_width = float(seed_rec.get("width", 0) or 0)
    cand_width = float(cand_rec.get("width", 0) or 0)
    width_delta = abs(cand_width - seed_width)
    # Allow up to 200mm width transition (more lenient than hard rule)
    if width_delta > 200:
        failures.append("WIDTH_DELTA_TOO_LARGE")
    
    # 3. Thickness delta check
    seed_thick = float(seed_rec.get("thickness", 0) or 0)
    cand_thick = float(cand_rec.get("thickness", 0) or 0)
    thick_delta = abs(cand_thick - seed_thick)
    # Allow up to 1.0mm thickness transition
    if thick_delta > 1.0:
        failures.append("THICKNESS_DELTA_TOO_LARGE")
    
    # 4. Temperature band check (must be same or adjacent)
    seed_temp = float(seed_rec.get("temperature", 200) or 200)
    cand_temp = float(cand_rec.get("temperature", 200) or 200)
    if abs(seed_temp - cand_temp) > 150:
        failures.append("TEMP_BAND_DISCONNECTED")
    
    # 5. Steel group compatibility (no hard block, but track)
    seed_group = str(seed_rec.get("steel_group", "") or "")
    cand_group = str(cand_rec.get("steel_group", "") or "")
    # Cross-group is okay but will penalize quality score
    
    # 6. Tonnage budget check (estimate if adding this order exceeds max)
    seed_tons = float(seed_rec.get("tons", 0) or 0)
    cand_tons = float(cand_rec.get("tons", 0) or 0)
    ton_max = float(getattr(cfg.model, "block_generator_target_tons_max", 2000.0))
    # Approximate: don't add if single order already exceeds max
    if cand_tons > ton_max:
        failures.append("ORDER_EXCEEDS_MAX_TONS")
    
    return len(failures) == 0, failures


# =============================================================================
# Directed Distance Scoring
# =============================================================================

@dataclass
class DirectedDistanceResult:
    """Result of directional distance computation."""
    total_score: float = 0.0
    width_component: float = 0.0
    thickness_component: float = 0.0
    temp_component: float = 0.0
    group_component: float = 0.0
    due_component: float = 0.0
    tons_fill_component: float = 0.0
    real_bridge_bonus: float = 0.0
    guarded_family_bonus: float = 0.0
    mixed_bridge_potential_bonus: float = 0.0
    reasons: List[str] = field(default_factory=list)


def directed_cluster_distance(
    prev_rec: dict,
    cand_rec: dict,
    block_state: dict,
    cfg: PlannerConfig,
    graph: TemplateGraph,
) -> DirectedDistanceResult:
    """
    Compute directional distance from prev_rec to cand_rec for block growing.
    
    This distance is DIRECTED: dist(A -> B) may not equal dist(B -> A).
    
    Key features:
    - Width: prefers decreasing width (reasonable rolling direction)
    - Thickness: prefers stable thickness transitions
    - Group: prefers same group transitions
    - Due: prefers orders with similar due dates
    - Tonnage fill: rewards candidates that help reach target tonnage
    - Real bridge bonus: rewards if there's a real bridge edge
    - Guarded family bonus: rewards if there's a guarded family edge
    - Mixed bridge potential: rewards if there's mixed bridge potential
    """
    result = DirectedDistanceResult()
    
    # Get weights from config
    w_width = float(getattr(cfg.model, "directional_cluster_width_weight", 1.0))
    w_thick = float(getattr(cfg.model, "directional_cluster_thickness_weight", 1.0))
    w_temp = float(getattr(cfg.model, "directional_cluster_temp_weight", 0.8))
    w_group = float(getattr(cfg.model, "directional_cluster_group_weight", 1.2))
    w_due = float(getattr(cfg.model, "directional_cluster_due_weight", 0.6))
    w_tons = float(getattr(cfg.model, "directional_cluster_tons_fill_weight", 1.0))
    b_real = float(getattr(cfg.model, "directional_cluster_real_bridge_bonus", 0.8))
    b_family = float(getattr(cfg.model, "directional_cluster_guarded_family_bonus", 0.5))
    b_mixed = float(getattr(cfg.model, "directional_cluster_mixed_bridge_potential_bonus", 0.3))
    
    # Width component: prefer decreasing width (rolling direction optimization)
    prev_width = float(prev_rec.get("width", 0) or 0)
    cand_width = float(cand_rec.get("width", 0) or 0)
    width_delta = cand_width - prev_width  # Negative is good (decreasing width)
    # Penalize width increase, reward width decrease
    if width_delta > 0:
        result.width_component = -min(width_delta / 50.0, 2.0) * w_width
        result.reasons.append(f"width_increase:{width_delta:.0f}")
    else:
        result.width_component = -min(width_delta / 50.0, 2.0) * w_width
        if width_delta < 0:
            result.reasons.append(f"width_decrease:{abs(width_delta):.0f}")
    
    # Thickness component: prefer stable transitions
    prev_thick = float(prev_rec.get("thickness", 0) or 0)
    cand_thick = float(cand_rec.get("thickness", 0) or 0)
    thick_delta = abs(cand_thick - prev_thick)
    result.thickness_component = -min(thick_delta * 2.0, 2.0) * w_thick
    if thick_delta > 0.5:
        result.reasons.append(f"thick_transition:{thick_delta:.2f}")
    
    # Temperature component: prefer same temperature band
    prev_temp = float(prev_rec.get("temperature", 200) or 200)
    cand_temp = float(cand_rec.get("temperature", 200) or 200)
    temp_delta = abs(cand_temp - prev_temp)
    result.temp_component = -min(temp_delta / 50.0, 1.0) * w_temp
    if temp_delta > 100:
        result.reasons.append(f"temp_transition:{temp_delta:.0f}")
    
    # Group component: prefer same group
    prev_group = str(prev_rec.get("steel_group", "") or "")
    cand_group = str(cand_rec.get("steel_group", "") or "")
    if prev_group and cand_group and prev_group == cand_group:
        result.group_component = 1.0 * w_group
        result.reasons.append("same_group")
    elif prev_group and cand_group:
        result.group_component = -0.3 * w_group
        result.reasons.append("cross_group")
    
    # Due date component: prefer orders with similar urgency
    prev_due = int(prev_rec.get("due_rank", 5) or 5)
    cand_due = int(cand_rec.get("due_rank", 5) or 5)
    due_delta = abs(cand_due - prev_due)
    result.due_component = -min(due_delta / 2.0, 1.0) * w_due
    
    # Tonnage fill component: reward if helps reach target
    current_tons = block_state.get("total_tons", 0.0)
    target_tons = float(getattr(cfg.model, "block_generator_target_tons_target", 1200.0))
    max_tons = float(getattr(cfg.model, "block_generator_target_tons_max", 2000.0))
    cand_tons = float(cand_rec.get("tons", 0) or 0)
    
    gap_to_target = target_tons - current_tons
    if gap_to_target > 0:
        # We're under target, reward bringing us closer
        result.tons_fill_component = min(cand_tons / gap_to_target, 1.0) * w_tons
        result.reasons.append(f"tons_fill:{cand_tons:.0f}")
    elif current_tons + cand_tons <= max_tons:
        # We're at/above target but still have room
        result.tons_fill_component = 0.3 * w_tons
        result.reasons.append("tons_still_valid")
    else:
        # Would exceed max
        result.tons_fill_component = -1.0 * w_tons
        result.reasons.append("tons_would_exceed_max")
    
    # Bridge bonuses
    prev_oid = str(prev_rec.get("order_id", ""))
    cand_oid = str(cand_rec.get("order_id", ""))
    
    if graph.has_real_bridge(prev_oid, cand_oid):
        result.real_bridge_bonus = b_real
        result.reasons.append("real_bridge_available")
    
    if graph.has_guarded_family(prev_oid, cand_oid):
        result.guarded_family_bonus = b_family
        result.reasons.append("guarded_family_available")
    
    mixed_possible, mixed_reason = graph.check_mixed_bridge_potential(prev_oid, cand_oid)
    if mixed_possible:
        result.mixed_bridge_potential_bonus = b_mixed
        result.reasons.append(f"mixed_bridge:{mixed_reason}")
    
    # Total score
    result.total_score = (
        result.width_component
        + result.thickness_component
        + result.temp_component
        + result.group_component
        + result.due_component
        + result.tons_fill_component
        + result.real_bridge_bonus
        + result.guarded_family_bonus
        + result.mixed_bridge_potential_bonus
    )
    
    return result


# =============================================================================
# Bucket Building
# =============================================================================

def build_seed_buckets(
    orders_df: pd.DataFrame,
    graph: TemplateGraph,
    cfg: PlannerConfig,
) -> Dict[str, List[str]]:
    """
    Partition orders into semantic buckets for directional clustering.
    
    Bucket key: line + width_band + thickness_band + steel_group_profile + temp_band
    """
    buckets: Dict[str, List[str]] = defaultdict(list)
    
    for _, row in orders_df.iterrows():
        oid = str(row["order_id"])
        cap = str(row.get("line_capability", "dual"))
        
        # Determine line
        line = "dual"
        if cap in {"big_only", "large"}:
            line = "big_roll"
        elif cap in {"small_only", "small"}:
            line = "small_roll"
        
        # Compute band keys
        width = float(row.get("width", 0) or 0)
        thickness = float(row.get("thickness", 0) or 0)
        temperature = float(row.get("temperature", 200) or 200)
        steel_group = str(row.get("steel_group", "") or "")
        
        w_band = _compute_width_band(width)
        t_band = _compute_thickness_band(thickness)
        tmp_band = _compute_temp_band(temperature)
        grp_profile = _compute_steel_group_coarse(steel_group)
        
        bucket_key = f"{line}|{w_band}|{t_band}|{grp_profile}|{tmp_band}"
        buckets[bucket_key].append(oid)
    
    # Remove empty buckets
    return {k: v for k, v in buckets.items() if v}


# =============================================================================
# Block Growing with Directional Clustering
# =============================================================================

def _grow_block_from_seed_directional(
    seed_oid: str,
    line: str,
    graph: TemplateGraph,
    cfg: PlannerConfig,
    rng: random.Random,
    mode: str = "direct_friendly",
) -> List[List[str]]:
    """
    Grow blocks from a seed order using directional clustering.
    
    Args:
        seed_oid: Starting order ID
        line: Production line
        graph: Template graph
        cfg: Planner config
        rng: Random generator
        mode: Block generation mode
            - direct_friendly: prefer direct transitions
            - real_bridge_friendly: prefer real bridge edges
            - guarded_family_friendly: prefer guarded family edges
            - dropped_recovery_friendly: prefer orders that could recover dropped
    
    Returns:
        List of order sequences (each is a valid block)

    Layer A — candidate acceptance gate (loose):
        A block is accepted into the candidate pool once total_tons >= candidate_tons_min.
        This is deliberately looser than target_tons_min to allow small blocks
        to enter the pool for subsequent merge / boundary rebalance / block rebuild.

    Layer B — ideal stop zone (tight):
        Once total_tons >= target_tons_target * 0.95, growth stops because the block
        is considered "good enough" by ideal-target standards.
        Blocks in [candidate_tons_min, target_tons_target*0.95) continue growing.
    """
    # Layer A: looser candidate gate
    candidate_ton_min = float(getattr(cfg.model, "block_generator_candidate_tons_min", 300.0))
    candidate_ton_max = float(getattr(cfg.model, "block_generator_candidate_tons_max", 2000.0))
    # Layer B: ideal target (unchanged — still used for stop decision)
    ton_target = float(getattr(cfg.model, "block_generator_target_tons_target", 1200.0))
    ton_max = float(getattr(cfg.model, "block_generator_target_tons_max", 2000.0))
    max_orders = int(getattr(cfg.model, "block_generator_max_orders_per_block", 20))
    
    max_real_bridge = int(getattr(cfg.model, "block_generator_max_real_bridge_edges_per_block", 2))
    max_family = int(getattr(cfg.model, "block_generator_max_family_edges_per_block", 2))
    
    results: List[List[str]] = []
    used: Set[str] = set()
    
    def grow(
        current_oid: str,
        sequence: List[str],
        current_tons: float,
        real_bridge_count: int,
        family_count: int,
        depth: int,
    ):
        if depth > max_orders:
            return
        if current_tons >= candidate_ton_min:
            results.append(list(sequence))
            # Layer B: stop when we reach ideal target zone
            if current_tons >= ton_target * 0.95 and current_tons <= ton_max:
                return
        if current_tons >= candidate_ton_max:
            return

        # Get candidates
        candidates = graph.get_out_candidates(current_oid, used)
        if not candidates:
            if current_tons >= candidate_ton_min:
                results.append(list(sequence))
            return
        
        # Score candidates using directional distance
        current_rec = graph.order_record.get(current_oid, {})
        block_state = {"total_tons": current_tons, "order_count": depth}
        
        scored_candidates = []
        for next_oid, tpl_row in candidates:
            cand_rec = graph.order_record.get(next_oid, {})
            
            # Hard gate check
            passes, failures = hard_cluster_gate(current_rec, cand_rec, line, cfg, graph)
            if not passes:
                continue
            
            # Directional distance
            dist = directed_cluster_distance(current_rec, cand_rec, block_state, cfg, graph)
            
            # Mode-specific adjustments
            edge_type = str(tpl_row.get("edge_type", "DIRECT_EDGE") or "DIRECT_EDGE")
            mode_bonus = 0.0
            if mode == "real_bridge_friendly" and edge_type == "REAL_BRIDGE_EDGE":
                mode_bonus = 1.5
            elif mode == "guarded_family_friendly" and edge_type == "VIRTUAL_BRIDGE_FAMILY_EDGE":
                mode_bonus = 1.5
            elif mode == "direct_friendly" and edge_type == "DIRECT_EDGE":
                mode_bonus = 0.5
            
            # Bridge budget check
            bridge_count = int(tpl_row.get("bridge_count", 0) or 0)
            if edge_type == "REAL_BRIDGE_EDGE" and real_bridge_count >= max_real_bridge:
                continue
            if edge_type == "VIRTUAL_BRIDGE_FAMILY_EDGE" and family_count >= max_family:
                continue
            
            total_score = dist.total_score + mode_bonus
            scored_candidates.append((next_oid, tpl_row, dist, total_score))
        
        if not scored_candidates:
            if current_tons >= candidate_ton_min:
                results.append(list(sequence))
            return
        
        # Sort by score and try top candidates
        scored_candidates.sort(key=lambda x: -x[3])
        top_k = min(3, len(scored_candidates))
        
        for i in range(top_k):
            next_oid, tpl_row, dist, score = scored_candidates[i]
            
            # Only try multiple variants for first few additions
            if i > 0 and depth > 2:
                break
            
            edge_type = str(tpl_row.get("edge_type", "DIRECT_EDGE") or "DIRECT_EDGE")
            bridge_count = int(tpl_row.get("bridge_count", 0) or 0)
            
            new_real_bridge = real_bridge_count + (1 if edge_type == "REAL_BRIDGE_EDGE" else 0)
            new_family = family_count + (1 if edge_type == "VIRTUAL_BRIDGE_FAMILY_EDGE" else 0)
            
            next_tons = current_tons + float(graph.order_record.get(next_oid, {}).get("tons", 0) or 0)
            
            used.add(next_oid)
            sequence.append(next_oid)
            
            grow(next_oid, sequence, next_tons, new_real_bridge, new_family, depth + 1)
            
            sequence.pop()
            used.remove(next_oid)
    
    # Start growing
    seed_tons = float(graph.order_record.get(seed_oid, {}).get("tons", 0) or 0)
    used.add(seed_oid)
    grow(seed_oid, [seed_oid], seed_tons, 0, 0, 1)
    used.remove(seed_oid)
    
    return results


# =============================================================================
# Main Block Generator Functions
# =============================================================================

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
        
        if len(sequence) >= 1:
            if current_tons >= ton_min:
                results.append((list(sequence), current_cost))
                if current_tons < ton_max * 0.95:
                    pass  # Continue walk
                else:
                    return
        
        if current_tons >= ton_max:
            return
        
        if direction == "forward":
            candidates = graph.get_out_candidates(current_oid, used)
        else:
            candidates = graph.get_in_candidates(current_oid, used)
        
        if not candidates:
            if len(sequence) >= 1 and current_tons >= ton_min:
                results.append((list(sequence), current_cost))
            return
        
        candidates_with_cost = []
        for next_oid, tpl_row in candidates:
            edge_cost = _template_edge_cost(tpl_row, cfg)
            candidates_with_cost.append((next_oid, tpl_row, edge_cost))
        
        candidates_with_cost.sort(key=lambda x: x[2])
        top_k = min(5, len(candidates_with_cost))
        chosen = rng.choice(candidates_with_cost[:top_k])
        next_oid, tpl_row, edge_cost = chosen
        
        used.add(next_oid)
        next_tons = current_tons + float(graph.order_record.get(next_oid, {}).get("tons", 0) or 0)
        sequence.append(next_oid)
        
        dfs(next_oid, sequence, next_tons, current_cost + edge_cost, depth + 1)
        
        sequence.pop()
        used.remove(next_oid)
    
    start_tons = float(graph.order_record.get(start_oid, {}).get("tons", 0) or 0)
    used.add(start_oid)
    dfs(start_oid, [start_oid], start_tons, 0, 0)
    used.remove(start_oid)
    
    return results


def _compute_block_profile(
    order_ids: List[str],
    graph: TemplateGraph,
) -> dict:
    """Compute block profile metadata from order sequence."""
    if not order_ids:
        return {}
    
    records = [graph.order_record.get(oid, {}) for oid in order_ids]
    
    head_rec = records[0]
    tail_rec = records[-1]
    
    widths = [float(r.get("width", 0) or 0) for r in records if r]
    thicknesses = [float(r.get("thickness", 0) or 0) for r in records if r]
    temps = [float(r.get("temperature", 200) or 200) for r in records if r]
    groups = [str(r.get("steel_group", "") or "") for r in records if r]
    
    return {
        "head_order_id": order_ids[0],
        "tail_order_id": order_ids[-1],
        "order_count": len(order_ids),
        "width_band": _compute_width_band(sum(widths) / len(widths) if widths else 0),
        "thickness_band": _compute_thickness_band(sum(thicknesses) / len(thicknesses) if thicknesses else 0),
        "temp_band": _compute_temp_band(sum(temps) / len(temps) if temps else 200),
        "steel_group_profile": max(set(groups), key=groups.count) if groups else "",
        "direct_edge_count": sum(1 for i in range(len(order_ids) - 1) if graph.edge_type(order_ids[i], order_ids[i + 1]) == "DIRECT_EDGE"),
        "real_bridge_edge_count": sum(1 for i in range(len(order_ids) - 1) if graph.has_real_bridge(order_ids[i], order_ids[i + 1])),
        "virtual_family_edge_count": sum(1 for i in range(len(order_ids) - 1) if graph.has_guarded_family(order_ids[i], order_ids[i + 1])),
    }


def _compute_block_quality_score(
    order_ids: List[str],
    graph: TemplateGraph,
    cfg: PlannerConfig,
) -> Tuple[float, float, float]:
    """
    Compute block quality metrics.
    
    Returns:
        Tuple of (block_quality_score, underfill_risk_score, bridge_dependency_score)
    """
    if not order_ids:
        return 0.0, 1.0, 0.0
    
    ton_target = float(getattr(cfg.model, "block_generator_target_tons_target", 1200.0))
    ton_min = float(getattr(cfg.model, "block_generator_target_tons_min", 700.0))
    total_tons = graph.tons_of(order_ids)
    
    # Quality score: higher is better
    # Base on how close to target
    ton_ratio = total_tons / ton_target if ton_target > 0 else 0
    quality = min(1.0, ton_ratio) if ton_ratio >= 0.8 else ton_ratio * 0.8
    
    # Width smoothness
    widths = []
    for oid in order_ids:
        rec = graph.order_record.get(oid, {})
        w = float(rec.get("width", 0) or 0)
        if w > 0:
            widths.append(w)
    
    width_smoothness = 1.0
    if len(widths) > 1:
        width_changes = sum(abs(widths[i] - widths[i + 1]) for i in range(len(widths) - 1))
        avg_width = sum(widths) / len(widths)
        width_smoothness = 1.0 - min(1.0, width_changes / (avg_width * 2))
    
    # Group continuity
    groups = [str(graph.order_record.get(oid, {}).get("steel_group", "") or "") for oid in order_ids]
    group_score = 1.0 if len(set(groups)) <= 2 else 0.6
    
    block_quality = quality * 0.4 + width_smoothness * 0.3 + group_score * 0.3
    
    # Underfill risk: higher is riskier
    if total_tons >= ton_target:
        underfill_risk = 0.1
    elif total_tons >= ton_min:
        underfill_risk = 0.3 + 0.4 * (1 - (total_tons - ton_min) / (ton_target - ton_min))
    else:
        underfill_risk = 0.9
    
    # Bridge dependency: higher = more dependent on bridges
    bridge_edges = 0
    total_edges = len(order_ids) - 1
    for i in range(len(order_ids) - 1):
        edge_type = graph.edge_type(order_ids[i], order_ids[i + 1])
        if edge_type in ("REAL_BRIDGE_EDGE", "VIRTUAL_BRIDGE_FAMILY_EDGE"):
            bridge_edges += 1
    
    bridge_dependency = bridge_edges / max(1, total_edges)
    
    return block_quality, underfill_risk, bridge_dependency


def _ton_range_key(tons: float, ton_min: float, ton_max: float) -> str:
    """
    Categorize a ton value into a range bucket based on target_ton_min.
    
    NOTE: This uses target_ton_min (the ideal minimum), NOT candidate_ton_min.
    The "below_min" range captures blocks in [candidate_ton_min, target_ton_min)
    which are accepted as pool candidates but are sub-ideal.
    """
    if tons < ton_min:
        return "below_target_min"
    t1 = ton_min * 1.3
    t2 = ton_min * 1.6
    t3 = ton_min * 1.9
    if tons < t1:
        return f"{int(ton_min)}-{int(t1)}"
    elif tons < t2:
        return f"{int(t1)}-{int(t2)}"
    elif tons < t3:
        return f"{int(t2)}-{int(t3)}"
    elif tons < ton_max * 0.8:
        return f"{int(t3)}-{int(ton_max * 0.8)}"
    elif tons <= ton_max:
        return f"{int(ton_max * 0.8)}-{int(ton_max)}"
    else:
        return "above_target_max"


def generate_candidate_macro_blocks_directional(
    orders_df: pd.DataFrame,
    tpl_df: pd.DataFrame,
    cfg: PlannerConfig,
    target_blocks: int = 2000,
    time_limit_seconds: float = 15.0,
    random_seed: int = 2027,
) -> Tuple[List[MacroBlock], BlockGeneratorStats]:
    """
    Generate candidate macro-blocks using directional clustering.
    
    This is the primary block generation method for block_first_guarded_search.
    
    Algorithm:
    1. Build seed buckets based on width/thickness/group/temp bands
    2. For each seed, generate blocks in multiple modes:
       - direct_friendly: prefer direct transitions
       - real_bridge_friendly: prefer real bridge edges
       - guarded_family_friendly: prefer guarded family edges
       - dropped_recovery_friendly: prefer recovery candidates
    3. Apply directional distance scoring for block growing
    4. Apply hard feasibility gates
    5. Deduplicate by signature
    6. Return sorted blocks by quality score
    """
    t0 = time.perf_counter()
    rng = random.Random(random_seed)
    blocks: List[MacroBlock] = []
    seen_signatures: Set[str] = set()
    stats = BlockGeneratorStats()
    
    if orders_df.empty:
        stats.generation_seconds = time.perf_counter() - t0
        return blocks, stats
    
    # Build graph and buckets
    graph = TemplateGraph(orders_df, tpl_df, cfg)
    buckets = build_seed_buckets(orders_df, graph, cfg)

    stats.seed_buckets_count = len(buckets)
    stats.seed_order_count = len(orders_df)

    # Candidate block (pool) threshold — looser gate for pool entry
    candidate_ton_min = float(getattr(cfg.model, "block_generator_candidate_tons_min", 300.0))
    candidate_ton_max = float(getattr(cfg.model, "block_generator_candidate_tons_max", 2000.0))
    # Ideal target threshold — kept for diagnostics and final campaign validation
    target_ton_min = float(getattr(cfg.model, "block_generator_target_tons_min", 700.0))
    target_ton_max = float(getattr(cfg.model, "block_generator_target_tons_max", 2000.0))
    max_seed_per_bucket = int(getattr(cfg.model, "block_generator_max_seed_per_bucket", 25))
    
    generation_modes = ["direct_friendly", "real_bridge_friendly", "guarded_family_friendly"]
    
    for bucket_key, seed_oids in buckets.items():
        if time.perf_counter() - t0 > time_limit_seconds:
            break
        if len(blocks) >= target_blocks:
            break
        
        # Parse bucket key
        parts = bucket_key.split("|")
        line = parts[0] if len(parts) > 0 else "big_roll"
        
        # Sample seeds from bucket
        rng.shuffle(seed_oids)
        seeds_to_use = seed_oids[:max_seed_per_bucket]
        
        for seed_oid in seeds_to_use:
            if time.perf_counter() - t0 > time_limit_seconds:
                break
            if len(blocks) >= target_blocks:
                break
            
            stats.total_candidates_explored += 1
            
            # Generate blocks in each mode
            for mode in generation_modes:
                if time.perf_counter() - t0 > time_limit_seconds:
                    break
                if len(blocks) >= target_blocks:
                    break
                
                sequences = _grow_block_from_seed_directional(
                    seed_oid=seed_oid,
                    line=line,
                    graph=graph,
                    cfg=cfg,
                    rng=rng,
                    mode=mode,
                )
                
                for seq in sequences:
                    if len(blocks) >= target_blocks:
                        break
                    
                    seq_tuple = tuple(seq)
                    
                    # Compute block profile
                    profile = _compute_block_profile(seq, graph)
                    total_tons = graph.tons_of(seq)
                    
                    # Check tonnage validity (candidate pool gate — looser than final campaign)
                    if total_tons < candidate_ton_min or total_tons > candidate_ton_max:
                        stats.generated_blocks_filtered_by_hard_gate += 1
                        continue
                    
                    # Compute signature for deduplication
                    sig = f"{line}:{profile.get('head_order_id', '')}:{profile.get('tail_order_id', '')}:{profile.get('width_band', '')}:{len(seq)}"
                    if sig in seen_signatures:
                        stats.generated_blocks_filtered_by_duplicate_signature += 1
                        continue
                    seen_signatures.add(sig)
                    
                    # Compute quality metrics
                    quality, underfill_risk, bridge_dep = _compute_block_quality_score(seq, graph, cfg)
                    
                    # Check for mixed bridge potential
                    mixed_possible = False
                    mixed_reason = ""
                    for i in range(len(seq) - 1):
                        possible, reason = graph.check_mixed_bridge_potential(seq[i], seq[i + 1])
                        if possible:
                            mixed_possible = True
                            mixed_reason = reason
                            break
                    
                    # Count bridge types
                    real_bridge_count = profile.get("real_bridge_edge_count", 0)
                    family_count = profile.get("virtual_family_edge_count", 0)
                    
                    if real_bridge_count > 0:
                        stats.generated_blocks_with_real_bridge += 1
                    if family_count > 0:
                        stats.generated_blocks_with_guarded_family += 1
                    if mixed_possible:
                        stats.generated_blocks_with_mixed_bridge_potential += 1
                    
                    # Compute total edge cost
                    total_cost = 0
                    for i in range(len(seq) - 1):
                        for _, tpl_row in graph.out_edges.get(seq[i], []):
                            if str(tpl_row.get("to_order_id", "")) == seq[i + 1]:
                                total_cost += _template_edge_cost(tpl_row, cfg)
                                break
                    
                    # Create MacroBlock
                    block = MacroBlock(
                        block_id=f"BLK_{line}_{len(blocks):06d}",
                        line=line,
                        order_ids=seq,
                        total_tons=total_tons,
                        total_cost=total_cost,
                        head_order_id=profile.get("head_order_id", ""),
                        tail_order_id=profile.get("tail_order_id", ""),
                        order_count=len(seq),
                        width_band=profile.get("width_band", ""),
                        thickness_band=profile.get("thickness_band", ""),
                        steel_group_profile=profile.get("steel_group_profile", ""),
                        temp_band=profile.get("temp_band", ""),
                        direct_edge_count=profile.get("direct_edge_count", 0),
                        real_bridge_edge_count=real_bridge_count,
                        virtual_family_edge_count=family_count,
                        mixed_bridge_possible=mixed_possible,
                        mixed_bridge_reason=mixed_reason,
                        block_quality_score=quality,
                        underfill_risk_score=underfill_risk,
                        bridge_dependency_score=bridge_dep,
                        source_bucket_key=bucket_key,
                        source_generation_mode=mode,
                    )
                    
                    blocks.append(block)
                    stats.unique_orders_covered.update(seq)
                    stats.max_walk_depth = max(stats.max_walk_depth, len(seq))
                    stats.blocks_by_mode[mode] = stats.blocks_by_mode.get(mode, 0) + 1
    
    # Also generate single-order blocks for isolated orders
    isolated_oids = [oid for oid in graph.order_record.keys() if oid not in stats.unique_orders_covered]
    for oid in isolated_oids[:100]:  # Limit to avoid explosion
        if time.perf_counter() - t0 > time_limit_seconds:
            break
        if len(blocks) >= target_blocks:
            break
        
        tons = float(graph.order_record.get(oid, {}).get("tons", 0) or 0)
        if target_ton_min <= tons <= target_ton_max:
            sig = f"single:{oid}"
            if sig in seen_signatures:
                continue
            seen_signatures.add(sig)
            
            block = MacroBlock(
                block_id=f"BLK_single_{len(blocks):06d}",
                line=graph.order_to_line.get(oid, "big_roll"),
                order_ids=[oid],
                total_tons=tons,
                total_cost=0,
                head_order_id=oid,
                tail_order_id=oid,
                order_count=1,
                source_generation_mode="single_order",
            )
            blocks.append(block)
            stats.unique_orders_covered.add(oid)
    
    # Compute stats
    stats.total_blocks_generated = len(blocks)
    for line in ["big_roll", "small_roll"]:
        stats.blocks_by_line[line] = sum(1 for b in blocks if b.line == line)
    
    for block in blocks:
        rng_key = _ton_range_key(block.total_tons, target_ton_min, target_ton_max)
        stats.blocks_by_ton_range[rng_key] = stats.blocks_by_ton_range.get(rng_key, 0) + 1
    
    # Compute averages
    if blocks:
        stats.avg_block_quality_score = sum(b.block_quality_score for b in blocks) / len(blocks)
        stats.avg_underfill_risk_score = sum(b.underfill_risk_score for b in blocks) / len(blocks)
        stats.avg_bridge_dependency_score = sum(b.bridge_dependency_score for b in blocks) / len(blocks)
    
    stats.generation_seconds = time.perf_counter() - t0
    
    # Sort blocks by quality score (higher = better)
    blocks.sort(key=lambda b: -b.block_quality_score)
    
    print(
        f"[APS][DirectionalBlockGen] generated={len(blocks)}, "
        f"big_roll={stats.blocks_by_line.get('big_roll', 0)}, "
        f"small_roll={stats.blocks_by_line.get('small_roll', 0)}, "
        f"seconds={stats.generation_seconds:.2f}, "
        f"orders_covered={len(stats.unique_orders_covered)}/{len(graph.order_record)}, "
        f"buckets={stats.seed_buckets_count}, "
        f"by_mode={stats.blocks_by_mode}, "
        f"avg_quality={stats.avg_block_quality_score:.3f}, "
        f"with_mixed_bridge={stats.generated_blocks_with_mixed_bridge_potential}"
    )
    
    return blocks, stats


def generate_candidate_macro_blocks(
    orders_df: pd.DataFrame,
    tpl_df: pd.DataFrame,
    cfg: PlannerConfig,
    target_blocks: int = 2000,
    time_limit_seconds: float = 15.0,
    random_seed: int = 2027,
    use_directional_clustering: bool = True,
) -> Tuple[List[MacroBlock], BlockGeneratorStats]:
    """
    Generate a pool of candidate macro blocks.
    
    This function dispatches to directional clustering when:
    - use_directional_clustering=True (default)
    - block_first_guarded_search profile is detected
    
    Falls back to randomized DFS for backward compatibility.
    """
    # Check if we should use directional clustering
    profile_name = str(getattr(cfg.model, "profile_name", "") or "")
    main_strategy = str(getattr(cfg.model, "main_solver_strategy", "") or "")
    
    use_directional = (
        use_directional_clustering
        and profile_name == "block_first_guarded_search"
    ) or main_strategy == "block_first"
    
    if use_directional:
        return generate_candidate_macro_blocks_directional(
            orders_df=orders_df,
            tpl_df=tpl_df,
            cfg=cfg,
            target_blocks=target_blocks,
            time_limit_seconds=time_limit_seconds,
            random_seed=random_seed,
        )
    
    # Fallback to randomized DFS (backward compatibility)
    return _generate_candidate_macro_blocks_legacy(
        orders_df=orders_df,
        tpl_df=tpl_df,
        cfg=cfg,
        target_blocks=target_blocks,
        time_limit_seconds=time_limit_seconds,
        random_seed=random_seed,
    )


def _generate_candidate_macro_blocks_legacy(
    orders_df: pd.DataFrame,
    tpl_df: pd.DataFrame,
    cfg: PlannerConfig,
    target_blocks: int = 2000,
    time_limit_seconds: float = 15.0,
    random_seed: int = 2027,
) -> Tuple[List[MacroBlock], BlockGeneratorStats]:
    """
    Legacy randomized DFS block generator (for backward compatibility).
    """
    t0 = time.perf_counter()
    rng = random.Random(random_seed)
    blocks: List[MacroBlock] = []
    seen_sequences: Set[Tuple[str, ...]] = set()
    stats = BlockGeneratorStats(seed_order_count=len(orders_df))
    
    if orders_df.empty or tpl_df.empty:
        stats.generation_seconds = time.perf_counter() - t0
        return blocks, stats
    
    graph = TemplateGraph(orders_df, tpl_df, cfg)
    lines = ["big_roll", "small_roll"]
    ton_min = float(cfg.rule.campaign_ton_min)
    ton_max = float(cfg.rule.campaign_ton_max)
    
    for line in lines:
        seed_orders = graph.get_seed_orders(line, min_tons=0)
        if not seed_orders:
            continue
        
        rng.shuffle(seed_orders)
        
        for pass_idx in range(3):
            if time.perf_counter() - t0 > time_limit_seconds:
                break
            
            pass_rng = random.Random(random_seed + pass_idx * 1000)
            pass_rng.shuffle(seed_orders)
            
            for seed_oid in seed_orders:
                if time.perf_counter() - t0 > time_limit_seconds:
                    break
                
                stats.total_candidates_explored += 1
                
                for seq, cost in _randomized_dfs_walk(graph, seed_oid, line, cfg, pass_rng, max_depth=30, direction="forward"):
                    seq_tuple = tuple(seq)
                    if seq_tuple in seen_sequences:
                        continue
                    
                    total_tons = graph.tons_of(seq)
                    if total_tons < ton_min or total_tons > ton_max:
                        continue
                    
                    seen_sequences.add(seq_tuple)
                    profile = _compute_block_profile(seq, graph)
                    quality, underfill_risk, bridge_dep = _compute_block_quality_score(seq, graph, cfg)
                    
                    block = MacroBlock(
                        block_id=f"BLK_{line}_{len(blocks):06d}",
                        line=line,
                        order_ids=seq,
                        total_tons=total_tons,
                        total_cost=cost,
                        head_order_id=profile.get("head_order_id", ""),
                        tail_order_id=profile.get("tail_order_id", ""),
                        order_count=len(seq),
                        width_band=profile.get("width_band", ""),
                        thickness_band=profile.get("thickness_band", ""),
                        steel_group_profile=profile.get("steel_group_profile", ""),
                        temp_band=profile.get("temp_band", ""),
                        direct_edge_count=profile.get("direct_edge_count", 0),
                        real_bridge_edge_count=profile.get("real_bridge_edge_count", 0),
                        virtual_family_edge_count=profile.get("virtual_family_edge_count", 0),
                        block_quality_score=quality,
                        underfill_risk_score=underfill_risk,
                        bridge_dependency_score=bridge_dep,
                        source_generation_mode="legacy_dfs",
                    )
                    blocks.append(block)
                    stats.unique_orders_covered.update(seq)
                    stats.max_walk_depth = max(stats.max_walk_depth, len(seq))
                    blocks_by_ton_range = _ton_range_key(total_tons, target_ton_min, target_ton_max)
                    stats.blocks_by_ton_range[blocks_by_ton_range] = stats.blocks_by_ton_range.get(blocks_by_ton_range, 0) + 1
    
    stats.total_blocks_generated = len(blocks)
    for line in lines:
        stats.blocks_by_line[line] = sum(1 for b in blocks if b.line == line)
    stats.generation_seconds = time.perf_counter() - t0
    
    blocks.sort(key=lambda b: -b.block_quality_score)
    
    print(
        f"[APS][LegacyBlockGen] generated={len(blocks)}, "
        f"big_roll={stats.blocks_by_line.get('big_roll', 0)}, "
        f"small_roll={stats.blocks_by_line.get('small_roll', 0)}, "
        f"seconds={stats.generation_seconds:.2f}"
    )
    
    return blocks, stats


# Alias for backward compatibility
def generate_feasible_blocks(
    orders_df: pd.DataFrame,
    tpl_df: pd.DataFrame,
    cfg: PlannerConfig,
    **kwargs,
) -> Tuple[List[MacroBlock], BlockGeneratorStats]:
    """Alias for generate_candidate_macro_blocks."""
    return generate_candidate_macro_blocks(orders_df, tpl_df, cfg, **kwargs)
