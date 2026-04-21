"""
Block Types for Block-First Architecture.

Core abstractions:
- BlockSeed: a single order ready for block building
- CandidateBlock: a candidate block formed from multiple orders
- CandidateBlockPool: the collection of all candidate blocks

This module is the foundation for the block-first experiment line:
block_first_guarded_search.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class BlockSeed:
    """
    A single order ready for block construction.
    Mirrors the key fields of an order row for block building.
    """
    order_id: str
    line: str  # "big_roll" or "small_roll"
    width: float
    thickness: float
    steel_group: str
    temp_min: float
    temp_max: float
    tons: float
    priority: int
    due_rank: int  # lower = earlier due date
    width_band: str = ""
    thickness_band: str = ""
    temp_band: str = ""
    roll_type: str = ""  # "大辊", "小辊", "大辊/小辊"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "order_id": self.order_id,
            "line": self.line,
            "width": self.width,
            "thickness": self.thickness,
            "steel_group": self.steel_group,
            "temp_min": self.temp_min,
            "temp_max": self.temp_max,
            "tons": self.tons,
            "priority": self.priority,
            "due_rank": self.due_rank,
            "width_band": self.width_band,
            "thickness_band": self.thickness_band,
            "temp_band": self.temp_band,
            "roll_type": self.roll_type,
        }


@dataclass
class CandidateBlock:
    """
    A candidate block: the core unit of the block-first architecture.

    Unlike a full order chain (order-first), a block has:
    - Controlled size (tons, order count)
    - Known bridge composition (direct, real bridge, guarded family)
    - Quality scores for block selection

    Attributes:
        block_id: Unique identifier
        line: Production line
        order_ids: Ordered list of order IDs in this block
        order_count: Number of orders
        total_tons: Sum of tons

        head_order_id / tail_order_id: endpoints for block-to-block transition
        head_signature: structured signature of head order (width, thickness, steel_group)
        tail_signature: structured signature of tail order

        width_band / thickness_band / steel_group_profile / temp_band:
            Aggregate bands for the block

        Bridge composition:
        - direct_edge_count: number of direct (real-to-real) edges
        - real_bridge_edge_count: number of REAL_BRIDGE_EDGE connections
        - virtual_family_edge_count: number of VIRTUAL_BRIDGE_FAMILY_EDGE connections
        - mixed_bridge_possible: whether block has opportunity for mixed bridge
        - mixed_bridge_reason: human-readable reason if mixed_bridge_possible

        Quality / risk scores:
        - block_quality_score: overall quality (higher = better), 0..100
        - underfill_risk_score: risk that block will be underfilled in a campaign (0..1)
        - bridge_dependency_score: how much this block depends on bridge edges (0..1)
        - dropped_recovery_potential: how much this block can recover dropped orders

        Source tracking:
        - source_bucket_key: identifies which bucket this block came from
        - source_generation_mode: greedy_seed | underfilled_rescue | boundary_patch | dropped_recovery
    """
    block_id: str
    line: str

    order_ids: List[str]
    order_count: int
    total_tons: float

    head_order_id: str
    tail_order_id: str
    head_signature: Dict[str, Any]
    tail_signature: Dict[str, Any]

    width_band: str
    thickness_band: str
    steel_group_profile: str
    temp_band: str

    direct_edge_count: int
    real_bridge_edge_count: int
    virtual_family_edge_count: int
    mixed_bridge_possible: bool
    mixed_bridge_reason: str

    block_quality_score: float
    underfill_risk_score: float
    bridge_dependency_score: float
    dropped_recovery_potential: float

    source_bucket_key: str
    source_generation_mode: str  # greedy_seed | underfilled_rescue | boundary_patch | dropped_recovery

    # Optional: explicit edge list for block internal realization
    internal_edges: List[Dict[str, Any]] = field(default_factory=list)

    # Optional: which orders this block conflicts with (order_id -> conflict reason)
    conflicts: Dict[str, str] = field(default_factory=dict)

    # Block-internal hotspot flags
    has_group_switch: bool = False
    has_width_tension: bool = False
    has_underfill_hotspot: bool = False
    has_bridge_dependency_hotspot: bool = False

    # ---- Candidate size classification ----
    # Introduced to decouple "candidate pool gate" from "ideal target gate".
    # A block can enter the candidate pool (candidate_tons_min) even if it is
    # below the ideal target minimum (target_tons_min).
    #
    # is_under_target_block: True if total_tons < target_tons_min
    #     Such blocks are accepted into the pool for merge/rebuild purposes
    #     but block_master applies disadvantages to discourage them as final campaigns.
    # candidate_size_class: one of
    #     "small_candidate" : candidate_tons_min <= total_tons < target_tons_min
    #     "target_candidate" : target_tons_min <= total_tons <= target_tons_max
    #     "above_target_max" : total_tons > target_tons_max (should not enter pool)
    is_under_target_block: bool = False
    candidate_size_class: str = "target_candidate"  # default for blocks already above target

    # Block internal state (filled after realization)
    is_selected: bool = False
    is_realized: bool = False
    scheduled_order_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "block_id": self.block_id,
            "line": self.line,
            "order_ids": list(self.order_ids),
            "order_count": self.order_count,
            "total_tons": self.total_tons,
            "head_order_id": self.head_order_id,
            "tail_order_id": self.tail_order_id,
            "head_signature": dict(self.head_signature),
            "tail_signature": dict(self.tail_signature),
            "width_band": self.width_band,
            "thickness_band": self.thickness_band,
            "steel_group_profile": self.steel_group_profile,
            "temp_band": self.temp_band,
            "direct_edge_count": self.direct_edge_count,
            "real_bridge_edge_count": self.real_bridge_edge_count,
            "virtual_family_edge_count": self.virtual_family_edge_count,
            "mixed_bridge_possible": self.mixed_bridge_possible,
            "mixed_bridge_reason": self.mixed_bridge_reason,
            "block_quality_score": self.block_quality_score,
            "underfill_risk_score": self.underfill_risk_score,
            "bridge_dependency_score": self.bridge_dependency_score,
            "dropped_recovery_potential": self.dropped_recovery_potential,
            "source_bucket_key": self.source_bucket_key,
            "source_generation_mode": self.source_generation_mode,
            "has_group_switch": self.has_group_switch,
            "has_width_tension": self.has_width_tension,
            "has_underfill_hotspot": self.has_underfill_hotspot,
            "has_bridge_dependency_hotspot": self.has_bridge_dependency_hotspot,
            "is_under_target_block": self.is_under_target_block,
            "candidate_size_class": self.candidate_size_class,
            "is_selected": self.is_selected,
            "is_realized": self.is_realized,
            "scheduled_order_ids": list(self.scheduled_order_ids),
        }


@dataclass
class CandidateBlockPool:
    """
    Collection of all candidate blocks for block-first solver.
    """
    blocks: List[CandidateBlock] = field(default_factory=list)

    # Diagnostics
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    # Block generation metadata
    generation_config: Dict[str, Any] = field(default_factory=dict)
    orders_input_count: int = 0
    lines_covered: List[str] = field(default_factory=list)

    def add_block(self, block: CandidateBlock) -> None:
        self.blocks.append(block)

    def blocks_by_line(self, line: str) -> List[CandidateBlock]:
        return [b for b in self.blocks if b.line == line]

    def blocks_by_mode(self, mode: str) -> List[CandidateBlock]:
        return [b for b in self.blocks if b.source_generation_mode == mode]

    def total_blocks(self) -> int:
        return len(self.blocks)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "blocks": [b.to_dict() for b in self.blocks],
            "diagnostics": dict(self.diagnostics),
            "generation_config": dict(self.generation_config),
            "orders_input_count": self.orders_input_count,
            "lines_covered": list(self.lines_covered),
        }
