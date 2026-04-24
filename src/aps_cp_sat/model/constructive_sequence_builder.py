"""
Constructive Sequence Builder for ALNS Architecture.

This module implements a greedy constructive heuristic for building valid order chains
based on template edges. It does NOT use slot-first logic, but generates order chains
for each production line independently.

The constructive phase is the first layer of the Constructive + ALNS path:
- Phase 1: Construct valid order chains using greedy template-edge following
- Phase 2 (ALNS): Optimize/perturb chains using Large Neighborhood Search

Key principles:
- Only follow edges that EXIST in the template DataFrame
- No penalty fallback edges
- Each order used at most once
- Dead island orders (isolated nodes) are dropped
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import pandas as pd

from aps_cp_sat.config import PlannerConfig
from aps_cp_sat.model.candidate_graph_types import DIRECT_EDGE, REAL_BRIDGE_EDGE, VIRTUAL_BRIDGE_EDGE, VIRTUAL_BRIDGE_FAMILY_EDGE
from aps_cp_sat.model.business_scoring_common import compute_extendability_metrics
from aps_cp_sat.model.edge_hard_filter import (
    accumulate_adj_hard_filter_rejection,
    log_adj_hard_filter,
    sequence_pair_passes_final_hard_rules,
)
from aps_cp_sat.model.reverse_width_budget import (
    CampaignBuildState,
    can_accept_reverse_width_pair,
)
from aps_cp_sat.model.seed_scoring import rank_seed_candidates
from aps_cp_sat.model.local_router import _template_total_cost


def _safe_int_value(value, default: int = 0) -> int:
    try:
        if pd.isna(value):
            return int(default)
    except Exception:
        pass
    try:
        return int(value)
    except Exception:
        return int(default)


@dataclass
class ConstructiveChain:
    """
    A single order chain built by following valid template edges.

    Attributes:
    chain_id: Unique identifier for this chain
    line: Production line (big_roll or small_roll)
    order_ids: List of order IDs in sequence
    total_tons: Sum of tons for all orders
    edge_count: Number of template edges used
    start_order_id: First order in chain
    end_order_id: Last order in chain
    steel_groups: Set of steel groups covered
    """
    chain_id: str
    line: str
    order_ids: List[str]
    total_tons: float
    edge_count: int
    start_order_id: str
    end_order_id: str
    steel_groups: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.steel_groups and self.order_ids:
            self.steel_groups = []

    def to_dict(self) -> dict:
        """Convert chain to dictionary for serialization."""
        return {
            "chain_id": self.chain_id,
            "line": self.line,
            "order_ids": list(self.order_ids),
            "total_tons": self.total_tons,
            "edge_count": self.edge_count,
            "start_order_id": self.start_order_id,
            "end_order_id": self.end_order_id,
            "steel_groups": list(self.steel_groups),
            "order_count": len(self.order_ids),
        }


@dataclass
class ConstructiveBuildResult:
    """
    Result of constructive chain building.

    Attributes:
    chains_by_line: Dictionary mapping line name to list of ConstructiveChain
    dropped_seed_orders: List of orders that could not be placed in any chain
    diagnostics: Build statistics and diagnostics
    repair_family_edges: Aggregated repair pool of family edges from all lines,
        for use by ALNS / local rebuild in guarded family mode.
    """
    chains_by_line: Dict[str, List[ConstructiveChain]] = field(default_factory=dict)
    dropped_seed_orders: List[dict] = field(default_factory=list)
    diagnostics: Dict = field(default_factory=dict)
    repair_family_edges: List = field(default_factory=list)  # CandidateEdge list

    def get_all_chains(self) -> List[ConstructiveChain]:
        """Get all chains across all lines."""
        all_chains = []
        for chains in self.chains_by_line.values():
            all_chains.extend(chains)
        return all_chains

    def get_total_orders_placed(self) -> int:
        """Get total number of orders placed in chains."""
        return sum(len(chain.order_ids) for chain in self.get_all_chains())

    def get_total_tons_placed(self) -> float:
        """Get total tons placed in chains."""
        return sum(chain.total_tons for chain in self.get_all_chains())


class TemplateEdgeGraph:
    """
    Directed graph representation of template edges for constructive building.

    This graph is built from the template DataFrame and only contains
    edges that actually exist in the template. No penalty fallbacks.

    Bridge edge filtering:
    - If allow_virtual_bridge_edge_in_constructive is False, VIRTUAL_BRIDGE_EDGE is excluded
    - If allow_real_bridge_edge_in_constructive is False, REAL_BRIDGE_EDGE is excluded
    - DIRECT_EDGE is always allowed
    """

    def __init__(
        self,
        orders_df: pd.DataFrame,
        tpl_df: pd.DataFrame,
        cfg: PlannerConfig,
        candidate_graph=None,
    ):
        self.cfg = cfg
        self.orders_df = orders_df
        self.tpl_df = tpl_df
        self.order_record: Dict[str, dict] = {}
        self.out_edges: Dict[str, List[Tuple[str, dict]]] = defaultdict(list)
        self.in_edges: Dict[str, List[Tuple[str, dict]]] = defaultdict(list)
        self.out_degree: Dict[str, int] = {}
        self.in_degree: Dict[str, int] = {}
        self.edge_cost: Dict[Tuple[str, str], int] = {}
        self.order_to_lines: Dict[str, List[str]] = defaultdict(list)

        # Bridge edge filtering config
        self.allow_virtual = getattr(cfg.model, "allow_virtual_bridge_edge_in_constructive", False)
        self.allow_real = getattr(cfg.model, "allow_real_bridge_edge_in_constructive", True)

        # Edge filtering diagnostics
        self.filtered_virtual_bridge_edge_count: int = 0
        self.filtered_real_bridge_edge_count: int = 0
        self.candidate_graph_diagnostics: Dict = {}

        # ---- Single graph build tracking ----
        # candidate_graph_source: "pipeline" = reused from pipeline. Local graph
        # rebuilding was removed so augmented graph and solve graph cannot drift.
        self.candidate_graph_source: str = "pipeline"
        # Full CandidateGraphBuildResult (for accessing repair_family_edges)
        self.candidate_graph_result = None

        # ---- Small roll dual-order reserve: per-line degree tracking ----
        self.big_deg: Dict[str, int] = {}
        self.small_deg: Dict[str, int] = {}

        self.accepted_direct_edge_count: int = 0
        self.accepted_real_bridge_edge_count: int = 0
        self.accepted_virtual_bridge_family_edge_count: int = 0
        self.accepted_virtual_bridge_edge_count: int = 0
        self.accepted_virtual_to_real_edge_count: int = 0
        self.accepted_virtual_to_virtual_edge_count: int = 0
        self.rejected_edges_by_hard_filter_total: int = 0
        self.skipped_by_final_width_rule: int = 0
        self.skipped_by_final_thickness_rule: int = 0
        self.skipped_by_final_temp_rule: int = 0
        self.skipped_by_final_group_rule: int = 0
        self.skipped_by_virtual_chain_rule: int = 0

        # Determine edge policy string
        cfg_model = getattr(cfg, "model", None)
        guarded_enabled = (
            getattr(cfg_model, "virtual_family_frontload_enabled", False)
            if cfg_model else False
        )
        if not self.allow_virtual and self.allow_real:
            self.edge_policy: str = "direct_plus_real_bridge"
        elif not self.allow_virtual and not self.allow_real:
            self.edge_policy = "direct_only"  # Route C: strictest mode
        elif self.allow_virtual and self.allow_real and guarded_enabled:
            self.edge_policy = "direct_plus_real_plus_guarded_family"  # Guarded virtual family frontload
        elif self.allow_virtual and self.allow_real:
            self.edge_policy = "all_edges_allowed"
        elif self.allow_real:
            self.edge_policy = "family_frontload"
        else:
            self.edge_policy = "virtual_only"

        # Family edge budget tracking for greedy phase
        self.family_edge_used_per_line: dict[str, int] = defaultdict(int)
        self.family_edge_used_per_segment: dict[str, int] = defaultdict(int)
        self.greedy_virtual_family_edge_uses: int = 0
        self.greedy_virtual_family_edge_rejects: int = 0
        self.greedy_virtual_family_budget_blocked_count: int = 0
        # ---- Future-aware penalty tracking (guarded profile only) ----
        self.greedy_future_bridgeability_penalty_hits: int = 0
        self.greedy_tail_underfill_risk_penalty_hits: int = 0
        self.greedy_bridge_scarcity_penalty_hits: int = 0
        # Reverse-width budget diagnostics (campaign-state constraint)
        self.skipped_by_reverse_width_budget: int = 0
        self.accepted_with_reverse_width_budget: int = 0
        self.reverse_width_count_used_max: int = 0
        self.adj_hard_filter_rejections: dict[str, int] = {}
        # Continuation bias diagnostics
        self.continuation_bonus_total: float = 0.0
        self.continuation_cross_min_hits: int = 0
        self.chains_crossed_ton_min_during_constructive: int = 0
        self.virtual_rescue_attempt_count: int = 0
        self.virtual_rescue_eligible_count: int = 0
        self.virtual_rescue_selected_count: int = 0
        self.virtual_rescue_selected_big_roll_count: int = 0
        self.virtual_rescue_selected_small_roll_count: int = 0
        self.virtual_rescue_cross_min_count: int = 0
        self.virtual_rescue_rejected_by_hard: int = 0
        self.virtual_rescue_rejected_by_budget: int = 0
        self.virtual_rescue_reason_counts_by_line: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.virtual_rescue_line_stats: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.successor_cross_min_selected: int = 0
        self.successor_near_min_selected: int = 0
        self.successor_deadend_rejected_under_min: int = 0

        self._build_graph(candidate_graph=candidate_graph)

    def _build_graph(self, candidate_graph=None) -> None:
        """Build directed graph from orders and templates.

        Args:
            candidate_graph: Optional pre-built candidate graph from pipeline.
                              If provided, reuses it instead of building locally.
                              This ensures Candidate Graph is built exactly once.
        """
        # Index orders
        for _, row in self.orders_df.iterrows():
            oid = str(row["order_id"])
            self.order_record[oid] = dict(row)

            # Determine which lines this order can run on
            cap = str(row.get("line_capability", "dual"))
            if cap in {"dual", "either"}:
                self.order_to_lines[oid] = ["big_roll", "small_roll"]
            elif cap in {"big_only", "large", "big"}:
                self.order_to_lines[oid] = ["big_roll"]
            elif cap in {"small_only", "small"}:
                self.order_to_lines[oid] = ["small_roll"]
            else:
                self.order_to_lines[oid] = ["big_roll", "small_roll"]

        if candidate_graph is None:
            raise RuntimeError(
                "[APS][CandidateGraph][CANDIDATE_GRAPH_REQUIRED_FROM_PIPELINE] "
                "constructive_sequence_builder requires the augmented CandidateGraph from pipeline"
            )
        self.candidate_graph_source = "pipeline"
        self.candidate_graph_result = candidate_graph
        self.candidate_graph_diagnostics = dict(getattr(candidate_graph, "diagnostics", {}) or {})

        # Build adjacency lists from normalized Candidate Graph edges
        if self.tpl_df.empty:
            return

        for candidate_edge in candidate_graph.edges:
            tpl_row = candidate_edge.to_template_row()
            from_oid = str(candidate_edge.from_order_id)
            to_oid = str(candidate_edge.to_order_id)
            if from_oid not in self.order_record or to_oid not in self.order_record:
                continue

            edge_line = str(candidate_edge.line or tpl_row.get("line", "big_roll"))
            if edge_line not in {"big_roll", "small_roll"}:
                edge_line = "big_roll"

            # ---- Bridge edge type filtering ----
            edge_type = str(candidate_edge.edge_type or DIRECT_EDGE)
            is_virtual_family = (edge_type == VIRTUAL_BRIDGE_FAMILY_EDGE)
            is_virtual_legacy = (edge_type == VIRTUAL_BRIDGE_EDGE)
            from_is_virtual = self._is_virtual_order(from_oid)
            to_is_virtual = self._is_virtual_order(to_oid)

            # Guarded family policy: legacy virtual is ALWAYS blocked
            if is_virtual_legacy:
                self.filtered_virtual_bridge_edge_count += 1
                continue

            if is_virtual_family and not self.allow_virtual:
                self.filtered_virtual_bridge_edge_count += 1
                continue  # Skip family edge if allow_virtual=False

            if edge_type == REAL_BRIDGE_EDGE and not self.allow_real:
                self.filtered_real_bridge_edge_count += 1
                continue  # Skip real bridge edge

            # Guarded family: apply frontload eligibility check
            if is_virtual_family and self.edge_policy == "direct_plus_real_plus_guarded_family":
                from aps_cp_sat.model.candidate_graph_types import is_virtual_family_frontload_eligible
                eligible, _ = is_virtual_family_frontload_eligible(candidate_edge, self.cfg, None)
                if not eligible:
                    self.filtered_virtual_bridge_edge_count += 1
                    continue  # Skip ineligible family edge

            hard_ok, hard_reason = sequence_pair_passes_final_hard_rules(
                self.order_record.get(from_oid, {}),
                self.order_record.get(to_oid, {}),
                self.cfg,
                {
                    "line": edge_line,
                    "edge_type": edge_type,
                    "template_row": tpl_row,
                    "bridge_count": int(candidate_edge.effective_max_bridge_count() or 0),
                },
            )
            if not hard_ok:
                self.rejected_edges_by_hard_filter_total += 1
                self.adj_hard_filter_rejections = accumulate_adj_hard_filter_rejection(
                    self.adj_hard_filter_rejections, hard_reason
                )
                if hard_reason == "FINAL_WIDTH_RULE":
                    self.skipped_by_final_width_rule += 1
                elif hard_reason == "FINAL_WIDTH_DROP_RULE":
                    self.skipped_by_final_width_rule += 1
                elif hard_reason == "FINAL_THICKNESS_RULE":
                    self.skipped_by_final_thickness_rule += 1
                elif hard_reason == "FINAL_TEMP_RULE":
                    self.skipped_by_final_temp_rule += 1
                elif hard_reason == "FINAL_GROUP_RULE":
                    self.skipped_by_final_group_rule += 1
                elif hard_reason in {"FINAL_VIRTUAL_CHAIN_RULE", "FINAL_VIRTUAL_RATIO_RULE"}:
                    self.skipped_by_virtual_chain_rule += 1
                continue

            # Track accepted edge counts for diagnostics
            if edge_type == DIRECT_EDGE:
                self.accepted_direct_edge_count += 1
            elif edge_type == REAL_BRIDGE_EDGE:
                self.accepted_real_bridge_edge_count += 1
            elif edge_type == VIRTUAL_BRIDGE_FAMILY_EDGE:
                self.accepted_virtual_bridge_family_edge_count += 1
            if from_is_virtual or to_is_virtual or is_virtual_family:
                self.accepted_virtual_bridge_edge_count += 1
            if from_is_virtual and not to_is_virtual:
                self.accepted_virtual_to_real_edge_count += 1
            if from_is_virtual and to_is_virtual:
                self.accepted_virtual_to_virtual_edge_count += 1

            # Only add edge if from_oid can run on this line
            if edge_line in self.order_to_lines.get(from_oid, []):
                self.out_edges[from_oid].append((to_oid, dict(tpl_row)))
                self.in_edges[to_oid].append((from_oid, dict(tpl_row)))

                # Calculate edge cost
                cost = _template_total_cost(dict(tpl_row), self.cfg.score)
                key = (from_oid, to_oid)
                if key not in self.edge_cost or cost < self.edge_cost[key]:
                    self.edge_cost[key] = cost

        # Compute degrees
        for oid in self.order_record:
            self.out_degree[oid] = len(self.out_edges.get(oid, []))
            self.in_degree[oid] = len(self.in_edges.get(oid, []))

        # ---- Compute per-line degree for dual-order reserve ----
        for oid in self.order_record:
            big_out = sum(
                1 for (to_oid, tpl) in self.out_edges.get(oid, [])
                if str(tpl.get("line", "big_roll")) == "big_roll"
            )
            big_in = sum(
                1 for (from_oid, tpl) in self.in_edges.get(oid, [])
                if str(tpl.get("line", "big_roll")) == "big_roll"
            )
            small_out = sum(
                1 for (to_oid, tpl) in self.out_edges.get(oid, [])
                if str(tpl.get("line", "big_roll")) == "small_roll"
            )
            small_in = sum(
                1 for (from_oid, tpl) in self.in_edges.get(oid, [])
                if str(tpl.get("line", "big_roll")) == "small_roll"
            )
            self.big_deg[oid] = big_out + big_in
            self.small_deg[oid] = small_out + small_in
        log_adj_hard_filter("transition_edge_build", self.adj_hard_filter_rejections)

    def _is_virtual_order(self, oid: str) -> bool:
        rec = self.order_record.get(str(oid), {})
        raw = rec.get("is_virtual", False)
        try:
            if pd.isna(raw):
                raw = False
        except Exception:
            pass
        return bool(raw) or str(rec.get("virtual_origin", "")) == "prebuilt_inventory"

    def get_valid_successors(
        self,
        current_oid: str,
        used_orders: set[str],
        line: str,
    ) -> List[Tuple[str, dict, int]]:
        """
        Get valid successor orders for extending a chain.

        Returns list of (successor_oid, template_row, edge_cost) tuples
        that are not yet used and can run on the specified line.

        Budget Gate (family edge):
        - If edge_type == VIRTUAL_BRIDGE_FAMILY_EDGE and guarded mode,
          skip if line/segment budget is already exhausted.
        """
        cfg_model = getattr(self.cfg, "model", None) if self.cfg else None
        family_frontload = (
            getattr(cfg_model, "virtual_family_frontload_enabled", False)
            if cfg_model else False
        )
        budget_per_line = int(getattr(cfg_model, "virtual_family_budget_per_line", 3) if cfg_model else 3)

        candidates = []
        for next_oid, tpl_row in self.out_edges.get(current_oid, []):
            if next_oid in used_orders:
                continue
            edge_line = str(tpl_row.get("line", "big_roll"))
            if edge_line != line:
                continue
            edge_type = str(tpl_row.get("edge_type", "") or "")
            is_family = (edge_type == VIRTUAL_BRIDGE_FAMILY_EDGE)

            # Budget Gate 1: per-line family budget hard block
            if is_family and family_frontload:
                current_used = self.family_edge_used_per_line.get(line, 0)
                if current_used >= budget_per_line:
                    self.greedy_virtual_family_edge_rejects += 1
                    self.greedy_virtual_family_budget_blocked_count += 1
                    continue  # Budget exhausted: hard block before candidate list

            cost = self.edge_cost.get((current_oid, next_oid), 0)
            candidates.append((next_oid, tpl_row, cost))
        return candidates

    def is_dead_island(self, oid: str, line: str) -> bool:
        """Check if order is isolated on the specified line (no in/out edges)."""
        has_out = any(
            str(tpl.get("line", "big_roll")) == line
            for _, tpl in self.out_edges.get(oid, [])
        )
        has_in = any(
            str(tpl.get("line", "big_roll")) == line
            for _, tpl in self.in_edges.get(oid, [])
        )
        return not has_out and not has_in

    def get_seed_orders(self, line: str) -> List[Tuple[str, dict]]:
        """
        Get orders that can start a new chain on the specified line.

        Orders are sorted by: priority, due_rank, tons, connectivity
        """
        candidates = []
        for oid, rec in self.order_record.items():
            if oid in self._used_orders:
                continue
            cap = str(rec.get("line_capability", "dual"))
            if cap in {"small_only", "small"} and line == "big_roll":
                continue
            if cap in {"big_only", "large", "big"} and line == "small_roll":
                continue
            # Must have at least one outgoing edge on this line to start
            has_out = any(
                str(tpl.get("line", "big_roll")) == line
                for _, tpl in self.out_edges.get(oid, [])
            )
            if has_out:
                candidates.append((oid, rec))
        return candidates

    def set_used_orders(self, used: set[str]) -> None:
        """Set the global used orders tracker."""
        self._used_orders = used

    def _init_used_orders(self) -> None:
        """Initialize used orders tracking."""
        if not hasattr(self, "_used_orders"):
            self._used_orders = set()


def _compute_chain_score(
    order_rec: dict,
    edge_cost: int,
    edge_type: str = DIRECT_EDGE,
    bridge_count: int = 0,
    bridge_family: str = "",
    cfg: PlannerConfig | None = None,
    due_rank_weight: float = 1000.0,
    priority_weight: float = 500.0,
    tons_weight: float = 1.0,
    cost_weight: float = 0.1,
    # ---- Future-aware context (guarded profile only) ----
    future_successor_count: int = 0,
    remaining_chain_capacity: int = 999,
    bridge_scarcity_score: float = 0.0,
) -> tuple[float, dict[str, int]]:
    """
    Compute composite score for chain extension decision.

    Higher score = better candidate.

    Returns (score, penalty_hits) where penalty_hits tracks which future-aware
    penalties were applied (for diagnostics, guarded profile only).

    Family edge penalty (guarded profile only):
    - DIRECT_EDGE: no penalty
    - REAL_BRIDGE_EDGE: small penalty (bridge_count * 5)
    - VIRTUAL_BRIDGE_FAMILY_EDGE: significant penalty (global_penalty +
      bridge_count * extra_penalty + family_type_penalty +
      future-aware penalties)

    Future-aware penalties (guarded profile only):
      future_bridgeability_penalty: penalize if few future successors (hard to bridge further)
      tail_underfill_risk_penalty: penalize if near tail and may underfill
      bridge_scarcity_preservation_penalty: penalize if using scarce bridge type
    """
    penalty_hits: dict[str, int] = {}

    due_rank = _safe_int_value(order_rec.get("due_rank", 999), 999)
    priority = _safe_int_value(order_rec.get("priority", 0), 0)
    tons = float(order_rec.get("tons", 0) or 0)

    # Due rank: smaller is better (more urgent)
    due_score = due_rank_weight / max(1, due_rank)

    # Priority: higher is better
    priority_score = priority_weight * priority

    # Tons: higher is better (prefer larger orders)
    tons_score = tons_weight * tons

    # Edge cost: lower is better
    cost_score = cost_weight * max(0, 1000 - edge_cost)

    base_score = due_score + priority_score + tons_score + cost_score

    # ---- Family edge penalty: make family edges less preferred unless necessary ----
    is_guarded_profile = False
    if edge_type == VIRTUAL_BRIDGE_FAMILY_EDGE and cfg is not None:
        model = getattr(cfg, "model", None)
        if model is not None and getattr(model, "virtual_family_frontload_enabled", False):
            is_guarded_profile = True
            global_penalty = float(getattr(model, "virtual_family_frontload_global_penalty", 100.0))
            extra_penalty = float(getattr(model, "virtual_family_frontload_local_penalty", 70.0))
            # Family type penalty: GROUP_TRANSITION > MIXED > THICKNESS > WIDTH_GROUP
            family_type_penalty = {"GROUP_TRANSITION": 40, "MIXED": 20, "THICKNESS": 10, "WIDTH_GROUP": 5}.get(
                bridge_family, 10
            )
            base_score -= global_penalty + bridge_count * extra_penalty + family_type_penalty

    # ---- Future-aware penalties (guarded profile only) ----
    if is_guarded_profile:
        # 1. future_bridgeability_penalty: penalize if few future successors
        #    (hard to bridge further after this edge → don't consume scarce budget)
        if future_successor_count <= 1:
            future_bridge_penalty = 60.0
            base_score -= future_bridge_penalty
            penalty_hits["future_bridgeability_penalty"] = 1
        elif future_successor_count <= 3:
            future_bridge_penalty = 25.0
            base_score -= future_bridge_penalty
            penalty_hits["future_bridgeability_penalty"] = 1

        # 2. tail_underfill_risk_penalty: penalize if near chain tail and may underfill
        #    Remaining capacity ≤ 3 slots AND edge_type is family → higher underfill risk
        if remaining_chain_capacity <= 3 and edge_type == VIRTUAL_BRIDGE_FAMILY_EDGE:
            tail_penalty = 35.0
            base_score -= tail_penalty
            penalty_hits["tail_underfill_risk_penalty"] = 1

        # 3. bridge_scarcity_preservation_penalty: penalize if using scarce bridge type
        #    bridge_scarcity_score: higher = scarcer (inverted from count)
        if bridge_scarcity_score > 0.5:
            scarcity_penalty = bridge_scarcity_score * 30.0
            base_score -= scarcity_penalty
            penalty_hits["bridge_scarcity_preservation_penalty"] = 1

    return base_score, penalty_hits


def continuation_bias_score(
    current_state: CampaignBuildState,
    candidate_order: dict,
    cfg: PlannerConfig | None,
) -> tuple[float, bool]:
    """Bias legal successors toward forming campaign-shape chains near ton_min."""
    rule = getattr(cfg, "rule", None) if cfg is not None else None
    model = getattr(cfg, "model", None) if cfg is not None else None
    ton_min = float(getattr(rule, "campaign_ton_min", 700.0) or 700.0)
    ton_max = float(getattr(rule, "campaign_ton_max", 2000.0) or 2000.0)

    w_gain = float(getattr(model, "continuation_bias_gain_weight", 0.35) if model else 0.35)
    bonus_cross = float(getattr(model, "continuation_bias_cross_min_bonus", 20.0) if model else 20.0)
    penalty_near_max = float(getattr(model, "continuation_bias_near_max_penalty", 25.0) if model else 25.0)
    near_max_ratio = float(getattr(model, "continuation_bias_near_max_ratio", 0.9) if model else 0.9)
    short_chain_hold = float(getattr(model, "continuation_bias_short_chain_hold_bonus", 8.0) if model else 8.0)

    curr_tons = float(current_state.current_tons)
    cand_tons = float(candidate_order.get("tons", 0.0) or 0.0)
    next_tons = curr_tons + cand_tons

    gap_before = max(0.0, ton_min - curr_tons)
    gap_after = max(0.0, ton_min - next_tons)
    gain = gap_before - gap_after
    score = float(w_gain) * float(gain)
    crossed = curr_tons < ton_min - 1e-6 and next_tons >= ton_min - 1e-6
    if crossed:
        score += bonus_cross

    if next_tons >= ton_max * near_max_ratio:
        score -= penalty_near_max

    if curr_tons < 200.0:
        score += short_chain_hold

    if ton_min <= curr_tons <= min(1000.0, ton_max):
        score *= 0.5
    return score, crossed


def _estimate_extendable_successors(
    *,
    graph: "TemplateEdgeGraph",
    current_oid: str,
    line: str,
    used_orders: set[str],
    current_state: CampaignBuildState,
    max_reverse_count: int,
) -> tuple[int, int]:
    valid_total = 0
    real_total = 0
    current_rec = graph.order_record.get(str(current_oid), {})
    current_virtual_chain_len = int(getattr(current_state, "virtual_chain_length", 0) or 0)
    campaign_order_count = int(getattr(current_state, "order_count", 0) or 0)
    for succ_oid, tpl_row, _cost in graph.get_valid_successors(str(current_oid), used_orders, str(line)):
        succ_rec = graph.order_record.get(str(succ_oid), {})
        ok, _ = sequence_pair_passes_final_hard_rules(
            current_rec,
            succ_rec,
            graph.cfg,
            {
                "line": line,
                "edge_type": str(tpl_row.get("edge_type", DIRECT_EDGE) or DIRECT_EDGE),
                "template_row": tpl_row,
                "bridge_count": int(tpl_row.get("bridge_count", 0) or 0),
                "current_virtual_chain_length": current_virtual_chain_len,
                "campaign_order_count": campaign_order_count,
                "campaign_virtual_count": current_virtual_chain_len,
            },
        )
        if not ok:
            continue
        ok_budget, _, _ = can_accept_reverse_width_pair(
            current_rec,
            succ_rec,
            current_state,
            max_reverse_count=max_reverse_count,
        )
        if not ok_budget:
            continue
        valid_total += 1
        if not graph._is_virtual_order(str(succ_oid)):
            real_total += 1
    return int(valid_total), int(real_total)


def _virtual_rescue_bonus(
    *,
    graph: "TemplateEdgeGraph",
    line: str,
    current_state: CampaignBuildState,
    candidate_order: dict[str, Any],
    candidate_state: CampaignBuildState,
    candidate_extendable_degree: float,
    real_successor_count: int,
    real_successor_dead_end_count: int,
    used_orders_for_candidate: set[str],
    campaign_ton_min: float,
    max_reverse_count: int,
) -> dict[str, Any]:
    model = getattr(graph.cfg, "model", None) if graph.cfg is not None else None
    line = str(line)
    line_reason_counts = graph.virtual_rescue_reason_counts_by_line[line]
    line_stats = graph.virtual_rescue_line_stats[line]
    result = {
        "bonus": 0.0,
        "eligible": False,
        "cross_min": False,
        "reason": "not_virtual",
        "dead_end_help": False,
    }
    candidate_oid = str(candidate_order.get("order_id", "") or "")
    if not graph._is_virtual_order(candidate_oid):
        return result

    graph.virtual_rescue_attempt_count += 1
    line_stats["attempt"] += 1

    if not bool(getattr(model, "constructive_virtual_rescue_enabled", True) if model else True):
        line_reason_counts["disabled"] += 1
        line_stats["reject_disabled"] += 1
        result["reason"] = "disabled"
        return result

    curr_tons = float(current_state.current_tons)
    min_current_tons = float(getattr(model, "constructive_virtual_rescue_min_current_tons", 0.0) if model else 0.0)
    if curr_tons < min_current_tons - 1e-6:
        line_reason_counts["below_min_current_tons"] += 1
        line_stats["reject_below_min_current_tons"] += 1
        result["reason"] = "below_min_current_tons"
        return result

    if bool(getattr(model, "constructive_virtual_rescue_under_min_only", True) if model else True) and curr_tons >= campaign_ton_min - 1e-6:
        line_reason_counts["not_under_min"] += 1
        line_stats["reject_not_under_min"] += 1
        result["reason"] = "not_under_min"
        return result

    if line == "big_roll" and bool(getattr(model, "constructive_virtual_rescue_big_roll_enabled", True) if model else True):
        rich_threshold = int(getattr(model, "constructive_virtual_rescue_big_roll_real_successor_threshold", 8) if model else 8)
        ignore_rich_when_deadend = bool(getattr(model, "constructive_virtual_rescue_ignore_rich_real_when_under_min_deadend", True) if model else True)
    else:
        rich_threshold = int(getattr(model, "constructive_virtual_rescue_real_successor_threshold", 2) if model else 2)
        ignore_rich_when_deadend = False
    real_rich = real_successor_count > rich_threshold
    real_mostly_dead = real_successor_count > 0 and real_successor_dead_end_count >= max(1, int(real_successor_count * 0.6))
    if real_rich and not (ignore_rich_when_deadend and real_mostly_dead):
        line_reason_counts["real_successor_rich"] += 1
        line_stats["reject_real_successor_rich"] += 1
        result["reason"] = "real_successor_rich"
        return result

    max_virtual_chain = int(getattr(model, "constructive_virtual_rescue_max_virtual_chain", 5) if model else 5)
    if int(getattr(candidate_state, "virtual_chain_length", 0) or 0) > max_virtual_chain:
        graph.virtual_rescue_rejected_by_budget += 1
        line_reason_counts["virtual_chain_budget"] += 1
        line_stats["reject_virtual_chain_budget"] += 1
        result["reason"] = "virtual_chain_budget"
        return result

    future_total, _future_real = _estimate_extendable_successors(
        graph=graph,
        current_oid=candidate_oid,
        line=line,
        used_orders=used_orders_for_candidate,
        current_state=candidate_state,
        max_reverse_count=max_reverse_count,
    )
    next_tons = float(curr_tons + float(candidate_order.get("tons", 0.0) or 0.0))
    gap_before = abs(campaign_ton_min - curr_tons)
    gap_after = abs(campaign_ton_min - next_tons)
    crosses_min = bool(curr_tons < campaign_ton_min - 1e-6 and next_tons >= campaign_ton_min - 1e-6)

    if line == "big_roll":
        near_min_ratio = float(getattr(model, "constructive_virtual_rescue_big_roll_near_min_ratio", 0.65) if model else 0.65)
    else:
        near_min_ratio = float(getattr(model, "constructive_virtual_rescue_near_min_ratio", 0.75) if model else 0.75)
    helps_min = bool(gap_after < gap_before - 1e-6 or next_tons >= campaign_ton_min * near_min_ratio)
    if not helps_min and not crosses_min and future_total <= 0:
        line_reason_counts["not_closer_to_min"] += 1
        line_stats["reject_not_closer_to_min"] += 1
        result["reason"] = "not_closer_to_min"
        return result

    graph.virtual_rescue_eligible_count += 1
    line_reason_counts["eligible"] += 1
    line_stats["eligible"] += 1
    result["eligible"] = True
    result["cross_min"] = crosses_min
    result["reason"] = "eligible"

    if line == "big_roll":
        bonus = float(getattr(model, "constructive_virtual_rescue_big_roll_bonus", 70.0) if model else 70.0)
        cross_bonus = float(getattr(model, "constructive_virtual_rescue_big_roll_cross_min_bonus", 110.0) if model else 110.0)
    else:
        bonus = float(getattr(model, "constructive_virtual_rescue_bonus", 45.0) if model else 45.0)
        cross_bonus = float(getattr(model, "constructive_virtual_rescue_cross_min_bonus", 85.0) if model else 85.0)
    if crosses_min:
        bonus += cross_bonus
    if real_successor_count <= 0 or (real_successor_dead_end_count >= max(1, real_successor_count) and candidate_extendable_degree > 0 and future_total > 0):
        bonus += float(getattr(model, "constructive_virtual_rescue_dead_end_bonus", 40.0) if model else 40.0)
        result["dead_end_help"] = True
    penalty_floor = float(getattr(model, "constructive_virtual_rescue_penalty_floor", 0.0) if model else 0.0)
    scale = float(getattr(model, "constructive_virtual_rescue_score_scale", 1.0) if model else 1.0)
    result["bonus"] = float(max(penalty_floor, bonus) * max(0.0, scale))
    return result


def _log_seed_scoring(line: str, seed_df: pd.DataFrame, selected_seed_rows: list[dict[str, Any]]) -> None:
    if seed_df is None or seed_df.empty:
        print(
            f"[APS][SeedScoring] line={line}, candidates=0, low_formability=0, "
            f"avg_projected_tons=0.0, selected_avg_projected_tons=0.0, selected_width_percentile_avg=0.0"
        )
        return
    selected_df = pd.DataFrame(selected_seed_rows) if selected_seed_rows else pd.DataFrame()
    selected_avg_projected_tons = float(selected_df["seed_projected_tons_5step"].fillna(0.0).mean()) if not selected_df.empty else 0.0
    selected_width_percentile_avg = float(selected_df["seed_width_percentile"].fillna(0.0).mean()) if not selected_df.empty else 0.0
    print(
        f"[APS][SeedScoring] line={line}, "
        f"candidates={len(seed_df)}, "
        f"low_formability={int(seed_df['seed_low_formability'].fillna(False).astype(bool).sum())}, "
        f"avg_projected_tons={float(seed_df['seed_projected_tons_5step'].fillna(0.0).mean()):.1f}, "
        f"selected_avg_projected_tons={selected_avg_projected_tons:.1f}, "
        f"selected_width_percentile_avg={selected_width_percentile_avg:.3f}"
    )


def compute_successor_business_score(
    current_state: CampaignBuildState,
    candidate_order: dict,
    candidate_state: CampaignBuildState,
    edge_meta: dict,
    graph: "TemplateEdgeGraph",
    cfg: PlannerConfig | None,
    line: str,
    used_orders_for_candidate: set[str],
    real_successor_count: int,
    real_successor_dead_end_count: int,
    campaign_ton_min: float,
    campaign_ton_max: float,
    max_reverse_count: int,
) -> dict[str, float]:
    model = getattr(cfg, "model", None) if cfg is not None else None
    current_rec = graph.order_record.get(str(current_state.last_order_id), {}) if current_state.last_order_id else {}
    current_width = float(current_rec.get("width", 0.0) or 0.0)
    cand_width = float(candidate_order.get("width", 0.0) or 0.0)
    current_thickness = float(current_rec.get("thickness", 0.0) or 0.0)
    cand_thickness = float(candidate_order.get("thickness", 0.0) or 0.0)
    current_group = str(current_rec.get("steel_group", "") or "")
    cand_group = str(candidate_order.get("steel_group", "") or "")
    current_temp_center = (
        float(current_rec.get("temp_min", 0.0) or 0.0) + float(current_rec.get("temp_max", 0.0) or 0.0)
    ) * 0.5
    cand_temp_center = (
        float(candidate_order.get("temp_min", 0.0) or 0.0) + float(candidate_order.get("temp_max", 0.0) or 0.0)
    ) * 0.5
    width_drop = max(0.0, current_width - cand_width)
    width_rise = max(0.0, cand_width - current_width)
    width_desc_score = 1.0
    if width_rise > 0:
        width_desc_score = max(0.0, 0.25 - min(0.25, width_rise / 200.0))
    elif width_drop > 250.0:
        width_desc_score = max(0.0, 1.0 - min(1.0, (width_drop - 250.0) / 250.0))
    elif width_drop > 0:
        width_desc_score = max(0.55, 1.0 - (width_drop / 500.0))

    continuation_bonus, crossed = continuation_bias_score(current_state=current_state, candidate_order=candidate_order, cfg=cfg)

    next_oid = str(candidate_order.get("order_id", "") or "")
    future_total_count, future_real_count = _estimate_extendable_successors(
        graph=graph,
        current_oid=next_oid,
        line=str(line),
        used_orders=used_orders_for_candidate,
        current_state=candidate_state,
        max_reverse_count=max_reverse_count,
    )
    future_successors = graph.get_valid_successors(next_oid, used_orders_for_candidate, str(line))
    extendability_metrics = compute_extendability_metrics(
        current_order=current_rec,
        candidate_order=candidate_order,
        candidate_successor_ids=[str(sid) for sid, _tpl, _cost in future_successors],
        order_record_by_oid=graph.order_record,
        cfg=cfg,
        line=str(line),
        current_state=current_state,
        edge_meta=edge_meta,
    )
    extendable_degree = max(float(extendability_metrics["successor_extendable_out_degree"]), float(future_total_count))
    same_group_degree = float(extendability_metrics["successor_same_group_extendable_degree"])
    width_desc_degree = float(extendability_metrics["successor_width_desc_extendable_degree"])
    dead_end_risk = float(extendability_metrics["successor_dead_end_risk_score"])
    reverse_tight_degree = float(extendability_metrics["successor_reverse_budget_tight_degree"])
    extendability_score = max(
        0.0,
        min(
            1.0,
            (0.45 * min(1.0, extendable_degree / 6.0))
            + (0.30 * (same_group_degree / max(1.0, extendable_degree)))
            + (0.25 * (width_desc_degree / max(1.0, extendable_degree))),
        ),
    )
    group_continuity_score = 1.0 if current_group and current_group == cand_group else 0.4
    smoothness_score = max(
        0.0,
        1.0 - ((abs(cand_thickness - current_thickness) * 1.5) + (abs(cand_temp_center - current_temp_center) * 0.005)),
    )
    reverse_budget_cost = 1.0 if cand_width > current_width else 0.0
    if reverse_tight_degree > 0:
        reverse_budget_cost += min(1.0, reverse_tight_degree / max(1.0, extendable_degree + reverse_tight_degree))

    curr_tons = float(current_state.current_tons)
    cand_tons = float(candidate_order.get("tons", 0.0) or 0.0)
    next_tons = curr_tons + cand_tons
    ton_gap_before = max(0.0, campaign_ton_min - curr_tons)
    ton_gap_after = max(0.0, campaign_ton_min - next_tons)
    ton_gain = max(0.0, ton_gap_before - ton_gap_after)
    under_min_gain_weight = float(getattr(model, "successor_under_min_gain_weight", 0.70) if model else 0.70)
    cross_min_bonus = float(getattr(model, "successor_cross_min_bonus", 95.0) if model else 95.0)
    near_min_bonus = float(getattr(model, "successor_near_min_bonus", 35.0) if model else 35.0)
    near_max_penalty = float(getattr(model, "successor_near_max_penalty", 60.0) if model else 60.0)
    near_max_ratio = float(getattr(model, "successor_near_max_ratio", 0.92) if model else 0.92)
    short_chain_deadend_penalty = float(getattr(model, "successor_short_chain_deadend_penalty", 55.0) if model else 55.0)
    crossed_direct = bool(curr_tons < campaign_ton_min - 1e-6 and next_tons >= campaign_ton_min - 1e-6)
    near_min_selected = bool(
        curr_tons < campaign_ton_min - 1e-6
        and next_tons >= campaign_ton_min * 0.75
        and next_tons < campaign_ton_min - 1e-6
    )
    tonnage_bonus = 0.0
    if curr_tons < campaign_ton_min - 1e-6:
        tonnage_bonus += under_min_gain_weight * ton_gain
        if crossed_direct:
            tonnage_bonus += cross_min_bonus
        elif near_min_selected:
            tonnage_bonus += near_min_bonus
        if extendable_degree > 0:
            tonnage_bonus += min(near_min_bonus, float(extendable_degree) * 6.0)
    elif next_tons >= campaign_ton_max * near_max_ratio:
        tonnage_bonus -= near_max_penalty
    tonnage_formability_score = max(0.0, min(1.0, (continuation_bonus + tonnage_bonus) / 120.0))

    dead_end_penalty = 0.0
    if extendable_degree <= 0:
        dead_end_penalty = 1.0
    elif extendable_degree <= 1:
        dead_end_penalty = 0.65
    elif extendable_degree <= 2:
        dead_end_penalty = 0.3
    dead_end_penalty = max(dead_end_penalty, dead_end_risk)
    if current_group and extendable_degree > 0 and (same_group_degree / max(1.0, extendable_degree)) < 0.25:
        dead_end_penalty = min(1.0, dead_end_penalty + 0.2)
    if extendable_degree > 0 and (width_desc_degree / max(1.0, extendable_degree)) < 0.35:
        dead_end_penalty = min(1.0, dead_end_penalty + 0.15)
    if curr_tons < campaign_ton_min - 1e-6 and extendable_degree <= 0:
        dead_end_penalty = min(1.5, dead_end_penalty + (short_chain_deadend_penalty / 100.0))

    w_width = float(getattr(model, "successor_weight_width_desc", 0.22) if model else 0.22)
    w_cont = float(getattr(model, "successor_weight_continuation", 0.24) if model else 0.24)
    w_ext = float(getattr(model, "successor_weight_extendability", 0.30) if model else 0.30)
    w_group = float(getattr(model, "successor_weight_group_continuity", 0.14) if model else 0.14)
    w_reverse = float(getattr(model, "successor_weight_reverse_budget_cost", 0.14) if model else 0.14)
    w_dead = float(getattr(model, "successor_weight_dead_end_penalty", 0.32) if model else 0.32)

    total_score = (
        w_width * width_desc_score
        + w_cont * tonnage_formability_score
        + w_ext * extendability_score
        + w_group * group_continuity_score
        + 0.10 * smoothness_score
        - w_reverse * reverse_budget_cost
        - w_dead * dead_end_penalty
    )
    virtual_rescue = _virtual_rescue_bonus(
        graph=graph,
        line=str(line),
        current_state=current_state,
        candidate_order=candidate_order,
        candidate_state=candidate_state,
        candidate_extendable_degree=extendable_degree,
        real_successor_count=real_successor_count,
        real_successor_dead_end_count=real_successor_dead_end_count,
        used_orders_for_candidate=used_orders_for_candidate,
        campaign_ton_min=campaign_ton_min,
        max_reverse_count=max_reverse_count,
    )
    total_score += float(virtual_rescue["bonus"] / 100.0)
    return {
        "successor_business_score_total": float(total_score),
        "width_desc_score": float(width_desc_score),
        "tonnage_formability_score": float(tonnage_formability_score),
        "extendability_score": float(extendability_score),
        "group_continuity_score": float(group_continuity_score),
        "smoothness_score": float(smoothness_score),
        "reverse_budget_cost": float(reverse_budget_cost),
        "dead_end_penalty": float(dead_end_penalty),
        "successor_extendable_out_degree": float(extendable_degree),
        "successor_same_group_extendable_degree": float(same_group_degree),
        "successor_width_desc_extendable_degree": float(width_desc_degree),
        "successor_dead_end_risk_score": float(dead_end_risk),
        "continuation_bonus": float(continuation_bonus + tonnage_bonus),
        "crossed_ton_min": float(1 if crossed or crossed_direct else 0),
        "near_min_bonus_hit": float(1 if near_min_selected else 0),
        "near_max_penalty_hit": float(1 if next_tons >= campaign_ton_max * near_max_ratio else 0),
        "deadend_rejected_under_min": float(1 if curr_tons < campaign_ton_min - 1e-6 and extendable_degree <= 0 else 0),
        "virtual_rescue_bonus": float(virtual_rescue["bonus"]),
        "virtual_rescue_eligible": float(1 if virtual_rescue["eligible"] else 0),
        "virtual_rescue_cross_min": float(1 if virtual_rescue["cross_min"] else 0),
        "future_real_successor_count": float(future_real_count),
    }


def _bridge_family_type_from_row(tpl_row: dict) -> str:
    """Derive bridge_family from a template row for family edge scoring."""
    raw = str(tpl_row.get("bridge_family", "")).strip().upper()
    if raw in {"WIDTH_GROUP", "THICKNESS", "GROUP_TRANSITION", "MIXED"}:
        return raw
    bridge_count = int(tpl_row.get("bridge_count", 0) or 0)
    if bridge_count >= 3:
        return "GROUP_TRANSITION"
    elif bridge_count >= 2:
        return "MIXED"
    else:
        return "WIDTH_GROUP"


def _allowed_lines_for_order(raw_cap: str) -> set[str]:
    """
    Determine which lines an order is allowed to run on based on line_capability.

    Args:
        raw_cap: The line_capability field value from order record

    Returns:
        Set of allowed line names: {"big_roll"}, {"small_roll"}, or {"big_roll", "small_roll"}
    """
    cap = str(raw_cap or "dual")
    if cap in {"small_only", "small"}:
        return {"small_roll"}
    elif cap in {"big_only", "large", "big"}:
        return {"big_roll"}
    else:
        # Default: dual/both/either can run on either line
        return {"big_roll", "small_roll"}


def _build_single_chain(
    graph: TemplateEdgeGraph,
    start_oid: str,
    line: str,
    used_orders: set[str],
    campaign_ton_min: float,
    campaign_ton_max: float,
    chain_counter: Dict[str, int],
    blocked_successor_bucket: set[str] | None = None,
) -> Tuple[ConstructiveChain | None, List[str]]:
    """
    Build a single chain starting from start_oid.

    Args:
        blocked_successor_bucket: Orders that cannot be used as chain SUCCESSORS
            by this line. Only applies to big_roll. Orders in this set are
            completely forbidden as successors (but NOT as chain seeds).
            For small_roll, pass None (no blocking).
            IMPORTANT: This should only contain STILL-LOCKED orders.
            Orders already released to big_roll must NOT be in this set.

    Returns:
    Tuple of (ConstructiveChain or None if too short, list of order_ids used)
    """
    if start_oid in used_orders:
        return None, []

    order_ids = [start_oid]
    used_orders.add(start_oid)
    current_oid = start_oid
    current_tons = float(graph.order_record.get(start_oid, {}).get("tons", 0) or 0)
    steel_groups = [str(graph.order_record.get(start_oid, {}).get("steel_group", "") or "")]
    current_state = CampaignBuildState(
        current_tons=current_tons,
        reverse_width_count=0,
        reverse_width_total_mm=0.0,
        virtual_chain_length=1 if graph._is_virtual_order(start_oid) else 0,
        last_order_id=str(start_oid),
        line=str(line),
        order_count=1,
        last_width=float(graph.order_record.get(start_oid, {}).get("width", 0.0) or 0.0),
        episode_state={},
    )
    chain_crossed_ton_min = bool(current_tons >= campaign_ton_min - 1e-6)
    max_reverse_count = int(getattr(getattr(graph.cfg, "model", None), "constructive_reverse_width_max_count", 2) or 2)

    # Segment budget key: stable within the same greedy chain (use seed + line).
    # Each chain gets its own segment scope; family budget per segment = 1.
    segment_budget_key = f"{line}:{start_oid}"

    while True:
        # Get valid successors (budget gate is applied inside get_valid_successors)
        successors = graph.get_valid_successors(current_oid, used_orders, line)

        # ---- Successor blocking: only block STILL-LOCKED quota orders ----
        # CRITICAL: Only orders in blocked_successor_bucket (still locked) are blocked.
        # Released orders are NOT in this set and are fully available as successors.
        # This ensures big_roll can both SEE released orders as seeds AND use them as successors.
        if blocked_successor_bucket and line == "big_roll":
            successors = [
                (soid, tpl, c) for soid, tpl, c in successors
                if soid not in blocked_successor_bucket
            ]

        current_virtual_chain_len = 0
        for _oid in reversed(order_ids):
            if graph._is_virtual_order(_oid):
                current_virtual_chain_len += 1
            else:
                break
        current_virtual_count = sum(1 for _oid in order_ids if graph._is_virtual_order(_oid))
        hard_filtered_successors = []
        for succ_oid, tpl_row, cost in successors:
            ok, reason = sequence_pair_passes_final_hard_rules(
                graph.order_record.get(current_oid, {}),
                graph.order_record.get(succ_oid, {}),
                graph.cfg,
                {
                    "line": line,
                    "edge_type": str(tpl_row.get("edge_type", DIRECT_EDGE) or DIRECT_EDGE),
                    "template_row": tpl_row,
                    "bridge_count": int(tpl_row.get("bridge_count", 0) or 0),
                    "current_virtual_chain_length": current_virtual_chain_len,
                    "campaign_order_count": len(order_ids),
                    "campaign_virtual_count": current_virtual_count,
                },
            )
            if ok:
                ok_budget, _, budget_state = can_accept_reverse_width_pair(
                    graph.order_record.get(current_oid, {}),
                    graph.order_record.get(succ_oid, {}),
                    current_state,
                    max_reverse_count=max_reverse_count,
                )
                if ok_budget:
                    graph.accepted_with_reverse_width_budget += 1
                    graph.reverse_width_count_used_max = max(
                        graph.reverse_width_count_used_max, int(budget_state.reverse_width_count)
                    )
                    if graph._is_virtual_order(str(succ_oid)):
                        graph.virtual_rescue_line_stats[str(line)]["raw_virtual_seen"] += 1
                        graph.virtual_rescue_line_stats[str(line)]["hard_pass"] += 1
                    hard_filtered_successors.append((succ_oid, tpl_row, cost, budget_state))
                else:
                    graph.skipped_by_reverse_width_budget += 1
                    if graph._is_virtual_order(str(succ_oid)):
                        graph.virtual_rescue_line_stats[str(line)]["raw_virtual_seen"] += 1
                        graph.virtual_rescue_line_stats[str(line)]["reject_reverse_budget"] += 1
            else:
                graph.rejected_edges_by_hard_filter_total += 1
                if graph._is_virtual_order(str(succ_oid)):
                    _vr_stats = graph.virtual_rescue_line_stats[str(line)]
                    _vr_stats["raw_virtual_seen"] += 1
                    _vr_stats["hard_reject"] += 1
                    graph.virtual_rescue_rejected_by_hard += 1
                    graph.virtual_rescue_reason_counts_by_line[str(line)]["hard_filter"] += 1
                    _reason_key = str(reason or "")
                    if _reason_key in {"FINAL_WIDTH_RULE", "FINAL_WIDTH_DROP_RULE"}:
                        _vr_stats["reject_width"] += 1
                    elif _reason_key == "FINAL_THICKNESS_RULE":
                        _vr_stats["reject_thickness"] += 1
                    elif _reason_key == "FINAL_TEMP_RULE":
                        _vr_stats["reject_temp"] += 1
                    elif _reason_key == "FINAL_GROUP_RULE":
                        _vr_stats["reject_group"] += 1
                    elif _reason_key in {"FINAL_VIRTUAL_CHAIN_RULE", "FINAL_VIRTUAL_RATIO_RULE"}:
                        _vr_stats["reject_virtual_chain"] += 1
                    elif _reason_key == "FINAL_REVERSE_WIDTH_BUDGET_RULE":
                        _vr_stats["reject_reverse_budget"] += 1
                    else:
                        _vr_stats["reject_other"] += 1
                graph.adj_hard_filter_rejections = accumulate_adj_hard_filter_rejection(
                    graph.adj_hard_filter_rejections, reason
                )
                if reason == "FINAL_WIDTH_RULE":
                    graph.skipped_by_final_width_rule += 1
                elif reason == "FINAL_WIDTH_DROP_RULE":
                    graph.skipped_by_final_width_rule += 1
                elif reason == "FINAL_THICKNESS_RULE":
                    graph.skipped_by_final_thickness_rule += 1
                elif reason == "FINAL_TEMP_RULE":
                    graph.skipped_by_final_temp_rule += 1
                elif reason == "FINAL_GROUP_RULE":
                    graph.skipped_by_final_group_rule += 1
                elif reason in {"FINAL_VIRTUAL_CHAIN_RULE", "FINAL_VIRTUAL_RATIO_RULE"}:
                    graph.skipped_by_virtual_chain_rule += 1
        successors = hard_filtered_successors

        if not successors:
            graph.candidate_graph_diagnostics["dead_end_successor_count"] = int(
                graph.candidate_graph_diagnostics.get("dead_end_successor_count", 0) or 0
            ) + 1
            break

        # Score and sort successors
        scored_successors = []
        cfg_model = getattr(graph.cfg, "model", None) if graph.cfg else None
        dual_reserve_enabled = getattr(cfg_model, "small_roll_dual_reserve_enabled", True) if cfg_model else True
        dual_reserve_penalty = int(getattr(cfg_model, "small_roll_dual_reserve_penalty", 15) if cfg_model else 15)

        # ---- Compute future-aware context for scoring ----
        # remaining_chain_capacity: estimated remaining orders based on ton capacity
        avg_order_tons = float(getattr(cfg_model, "constructive_typical_order_tons", 25.0) if cfg_model else 25.0)
        avg_order_tons = max(1.0, avg_order_tons)
        remaining_orders_est = max(1, int((campaign_ton_max - current_tons) / avg_order_tons))
        remaining_chain_capacity = remaining_orders_est

        # bridge_scarcity_score: inverse of remaining per-line family budget
        # Higher score = scarcer = more penalty for using
        guarded_enabled = getattr(cfg_model, "virtual_family_frontload_enabled", False) if cfg_model else False
        if guarded_enabled:
            per_line_budget = int(getattr(cfg_model, "virtual_family_budget_per_line", 4) if cfg_model else 4)
            per_line_used = graph.family_edge_used_per_line.get(line, 0)
            remaining_budget = max(0, per_line_budget - per_line_used)
            # Score 0.0 = plenty left, score 1.0 = nearly exhausted
            bridge_scarcity_score = 1.0 - (remaining_budget / max(1, per_line_budget))
        else:
            bridge_scarcity_score = 0.0

        real_successor_count = 0
        real_successor_dead_end_count = 0
        successor_future_counts: dict[str, tuple[int, int]] = {}
        for succ_oid, _tpl_row, _cost, budget_state in successors:
            if graph._is_virtual_order(str(succ_oid)):
                continue
            real_successor_count += 1
            future_total, future_real = _estimate_extendable_successors(
                graph=graph,
                current_oid=str(succ_oid),
                line=str(line),
                used_orders=set(used_orders) | {str(succ_oid)},
                current_state=budget_state,
                max_reverse_count=max_reverse_count,
            )
            successor_future_counts[str(succ_oid)] = (future_total, future_real)
            if future_total <= 0:
                real_successor_dead_end_count += 1

        for succ_oid, tpl_row, cost, budget_state in successors:
            succ_rec = graph.order_record.get(succ_oid, {})
            edge_type = str(tpl_row.get("edge_type", DIRECT_EDGE))
            bridge_count = int(tpl_row.get("bridge_count", 0) or 0)
            bridge_family = _bridge_family_type_from_row(tpl_row)
            score, penalty_hits = _compute_chain_score(
                succ_rec, cost,
                edge_type=edge_type,
                bridge_count=bridge_count,
                bridge_family=bridge_family,
                cfg=graph.cfg,
                future_successor_count=len(successors),
                remaining_chain_capacity=remaining_chain_capacity,
                bridge_scarcity_score=bridge_scarcity_score,
            )

            # Accumulate future-aware penalty hits into graph diagnostics
            if penalty_hits.get("future_bridgeability_penalty"):
                graph.greedy_future_bridgeability_penalty_hits += 1
            if penalty_hits.get("tail_underfill_risk_penalty"):
                graph.greedy_tail_underfill_risk_penalty_hits += 1
            if penalty_hits.get("bridge_scarcity_preservation_penalty"):
                graph.greedy_bridge_scarcity_penalty_hits += 1

            # ---- Dual-order small-roll reserve: penalize big_roll taking dual orders ----
            # If order has edges on BOTH lines, add a penalty on big_roll to leave some for small_roll
            if dual_reserve_enabled and line == "big_roll":
                big_deg = graph.big_deg.get(succ_oid, 0)
                small_deg = graph.small_deg.get(succ_oid, 0)
                if small_deg > 0 and big_deg > 0:
                    # This order can run on both lines and has small_roll connectivity
                    dual_small_reserve_score = min(small_deg, 10)
                    # Penalize, not reward, big_roll taking dual orders with small_roll connectivity.
                    # The previous "+=" made the reserve signal work in the opposite direction.
                    score -= dual_small_reserve_score * dual_reserve_penalty

            business_score = compute_successor_business_score(
                current_state=current_state,
                candidate_order=succ_rec,
                candidate_state=budget_state,
                edge_meta=tpl_row,
                graph=graph,
                cfg=graph.cfg,
                line=str(line),
                used_orders_for_candidate=set(used_orders) | {str(succ_oid)},
                real_successor_count=real_successor_count,
                real_successor_dead_end_count=real_successor_dead_end_count,
                campaign_ton_min=campaign_ton_min,
                campaign_ton_max=campaign_ton_max,
                max_reverse_count=max_reverse_count,
            )
            score += float(business_score["successor_business_score_total"] * 100.0)
            graph.continuation_bonus_total += float(business_score["continuation_bonus"])
            graph.candidate_graph_diagnostics["successor_reverse_budget_cost_total"] = float(
                graph.candidate_graph_diagnostics.get("successor_reverse_budget_cost_total", 0.0) or 0.0
            ) + float(business_score["reverse_budget_cost"])
            graph.candidate_graph_diagnostics["dead_end_penalty_total"] = float(
                graph.candidate_graph_diagnostics.get("dead_end_penalty_total", 0.0) or 0.0
            ) + float(business_score["dead_end_penalty"])
            graph.candidate_graph_diagnostics["virtual_rescue_bonus_total"] = float(
                graph.candidate_graph_diagnostics.get("virtual_rescue_bonus_total", 0.0) or 0.0
            ) + float(business_score["virtual_rescue_bonus"])
            graph.candidate_graph_diagnostics["successor_score_component_sum"] = float(
                graph.candidate_graph_diagnostics.get("successor_score_component_sum", 0.0) or 0.0
            ) + float(business_score["successor_business_score_total"])
            graph.candidate_graph_diagnostics["successor_score_component_count"] = int(
                graph.candidate_graph_diagnostics.get("successor_score_component_count", 0) or 0
            ) + 1
            graph.candidate_graph_diagnostics["successor_extendability_component_sum"] = float(
                graph.candidate_graph_diagnostics.get("successor_extendability_component_sum", 0.0) or 0.0
            ) + float(business_score["extendability_score"])
            graph.candidate_graph_diagnostics["successor_dead_end_penalty_sum"] = float(
                graph.candidate_graph_diagnostics.get("successor_dead_end_penalty_sum", 0.0) or 0.0
            ) + float(business_score["dead_end_penalty"])
            if float(business_score["successor_extendable_out_degree"]) <= 0:
                graph.candidate_graph_diagnostics["successor_zero_extendable_degree_count"] = int(
                    graph.candidate_graph_diagnostics.get("successor_zero_extendable_degree_count", 0) or 0
                ) + 1
            elif float(business_score["successor_extendable_out_degree"]) <= 1:
                graph.candidate_graph_diagnostics["successor_low_extendable_degree_count"] = int(
                    graph.candidate_graph_diagnostics.get("successor_low_extendable_degree_count", 0) or 0
                ) + 1
            if business_score["crossed_ton_min"] > 0:
                graph.continuation_cross_min_hits += 1
            if business_score["deadend_rejected_under_min"] > 0:
                graph.successor_deadend_rejected_under_min += 1
            scored_successors.append(
                (
                    score,
                    succ_oid,
                    tpl_row,
                    cost,
                    budget_state,
                    business_score,
                )
            )

        # Sort by score descending (best first)
        scored_successors.sort(key=lambda x: -x[0])

        # Try best successor
        best_score, best_succ, best_tpl, best_cost, best_state, _best_business = scored_successors[0]
        succ_tons = float(graph.order_record.get(best_succ, {}).get("tons", 0) or 0)

        # Check if adding would exceed campaign ton max
        if current_tons + succ_tons > campaign_ton_max:
            # Try other successors if they fit
            fit_found = False
            for score, succ_oid, tpl_row, cost, budget_state, _biz in scored_successors[1:]:
                succ_tons = float(graph.order_record.get(succ_oid, {}).get("tons", 0) or 0)
                if current_tons + succ_tons <= campaign_ton_max:
                    best_succ, best_tpl, best_cost, succ_tons, best_state, _best_business = (
                        succ_oid,
                        tpl_row,
                        cost,
                        succ_tons,
                        budget_state,
                        _biz,
                    )
                    fit_found = True
                    break
            if not fit_found:
                break

        # ---- Budget Gate 2: segment-level family budget hard block ----
        # Check AFTER score selection but BEFORE extending.
        # This prevents a high-score family edge from consuming the segment budget.
        best_edge_type = str(best_tpl.get("edge_type", "") or "")
        best_is_family = (best_edge_type == VIRTUAL_BRIDGE_FAMILY_EDGE)
        if best_is_family and getattr(cfg_model, "virtual_family_frontload_enabled", False):
            seg_budget = int(getattr(cfg_model, "virtual_family_budget_per_segment", 1) if cfg_model else 1)
            seg_used = graph.family_edge_used_per_segment.get(segment_budget_key, 0)
            if seg_used >= seg_budget:
                # Segment budget exhausted: reject this family edge
                graph.greedy_virtual_family_edge_rejects += 1
                graph.greedy_virtual_family_budget_blocked_count += 1
                # Try next-best non-family successor
                non_family = [
                    (s, o, t, c, st, biz)
                    for s, o, t, c, st, biz in scored_successors[1:]
                    if str(t.get("edge_type", "")) != VIRTUAL_BRIDGE_FAMILY_EDGE
                ]
                if non_family:
                    best_score, best_succ, best_tpl, best_cost, best_state, _best_business = non_family[0]
                    succ_tons = float(graph.order_record.get(best_succ, {}).get("tons", 0) or 0)
                    best_is_family = False
                    if current_tons + succ_tons > campaign_ton_max:
                        break  # Can't fit non-family either
                else:
                    break  # No acceptable non-family successor

        if float(_best_business.get("crossed_ton_min", 0.0) or 0.0) > 0:
            graph.successor_cross_min_selected += 1
        if float(_best_business.get("near_min_bonus_hit", 0.0) or 0.0) > 0:
            graph.successor_near_min_selected += 1
        if float(_best_business.get("virtual_rescue_eligible", 0.0) or 0.0) > 0 and graph._is_virtual_order(str(best_succ)):
            graph.virtual_rescue_selected_count += 1
            if str(line) == "big_roll":
                graph.virtual_rescue_selected_big_roll_count += 1
            else:
                graph.virtual_rescue_selected_small_roll_count += 1
            if float(_best_business.get("virtual_rescue_cross_min", 0.0) or 0.0) > 0:
                graph.virtual_rescue_cross_min_count += 1

        # Extend chain
        order_ids.append(best_succ)
        used_orders.add(best_succ)
        current_oid = best_succ
        current_tons += succ_tons
        if not chain_crossed_ton_min and current_tons >= campaign_ton_min - 1e-6:
            chain_crossed_ton_min = True
        sg = str(graph.order_record.get(best_succ, {}).get("steel_group", "") or "")
        if sg and sg not in steel_groups:
            steel_groups.append(sg)
        current_state = best_state

        # ---- Count family edge usage AFTER successful selection ----
        if best_is_family:
            graph.family_edge_used_per_line[line] += 1
            graph.family_edge_used_per_segment[segment_budget_key] += 1
            graph.greedy_virtual_family_edge_uses += 1

    # Only return chain if it has at least 1 order
    if len(order_ids) >= 1:
        if chain_crossed_ton_min:
            graph.chains_crossed_ton_min_during_constructive += 1
        line_count = chain_counter.get(line, 0)
        chain_counter[line] = line_count + 1

        chain = ConstructiveChain(
            chain_id=f"CHAIN_{line}_{line_count + 1:04d}",
            line=line,
            order_ids=order_ids,
            total_tons=current_tons,
            edge_count=len(order_ids) - 1,
            start_order_id=order_ids[0],
            end_order_id=order_ids[-1],
            steel_groups=steel_groups,
        )
        return chain, order_ids

    return None, []


def _retry_line_after_quota_release(
    collapsed_line: str,
    graph: TemplateEdgeGraph,
    used_orders: set[str],
    released_orders: set[str],
    campaign_ton_min: float,
    campaign_ton_max: float,
    diagnostics: dict,
    existing_chains_by_line: dict[str, list[ConstructiveChain]],
) -> int:
    """
    Retry building chains for a collapsed line after quota release.

    This function is called when one line (big_roll or small_roll) got 0 orders
    despite having feasible nodes. The locked quota has been released, and now
    we retry building chains for the collapsed line.

    Args:
        collapsed_line: Which line collapsed ("big_roll" or "small_roll")
        graph: TemplateEdgeGraph with order and edge data
        used_orders: Global used orders set (will be updated)
        released_orders: Orders that were released from locked bucket and NOT yet used
        campaign_ton_max: Maximum tons per chain
        diagnostics: Diagnostics dict (will be updated with retry results)
        existing_chains_by_line: Current chains_by_line dict (new chains will be appended)

    Returns:
        Number of new orders added to chains during retry
    """
    if not released_orders:
        diagnostics["line_balance_retry_added_orders"] = 0
        diagnostics["line_balance_retry_added_chains"] = 0
        return 0

    # Collect orders that are actually available for retry
    # (released AND not yet used AND have feasible edges on collapsed_line)
    retry_available: list[tuple[str, dict]] = []
    for oid in released_orders:
        if oid in used_orders:
            continue
        rec = graph.order_record.get(oid, {})
        if not rec:
            continue
        # Check if order has feasible edges on collapsed line
        has_out = any(
            str(tpl.get("line", "big_roll")) == collapsed_line
            for _, tpl in graph.out_edges.get(oid, [])
        )
        if has_out:
            retry_available.append((oid, rec))
        else:
            # Order has no outgoing edges on this line, cannot be used
            diagnostics["line_balance_retry_no_edges_count"] = (
                diagnostics.get("line_balance_retry_no_edges_count", 0) + 1
            )

    if not retry_available:
        diagnostics["line_balance_retry_added_orders"] = 0
        diagnostics["line_balance_retry_added_chains"] = 0
        diagnostics["line_balance_retry_still_zero_after_retry"] = True
        print(
            f"[APS][BalanceProtection] RETRY {collapsed_line}: no feasible orders in released set "
            f"(released={len(released_orders)}, still_zero=True)"
        )
        return 0

    # Sort retry orders by seed quality
    retry_seed_scored = rank_seed_candidates(
        pd.DataFrame([dict(rec, order_id=oid, line=collapsed_line) for oid, rec in retry_available]),
        graph,
        cfg=graph.cfg,
    )
    retry_seed_rank = {str(row["order_id"]): dict(row) for row in retry_seed_scored.to_dict("records")}
    retry_available.sort(
        key=lambda item: (
            -float(retry_seed_rank.get(str(item[0]), {}).get("seed_score_total", 0.0)),
            str(item[0]),
        )
    )

    # Get or create chain counter for this line
    chain_counter: Dict[str, int] = {}
    for line_name, chains in existing_chains_by_line.items():
        chain_counter[line_name] = len(chains)

    # Build chains for collapsed line
    # NOTE: For collapsed_line, we do NOT pass blocked_successor_bucket.
    # All released orders are fully available (they were released specifically
    # to give this collapsed line a second chance).
    retry_chains: list[ConstructiveChain] = []
    retry_added_orders = 0

    for oid, rec in retry_available:
        if oid in used_orders:
            continue

        # Build chain without any successor blocking
        chain, chain_orders = _build_single_chain(
            graph=graph,
            start_oid=oid,
            line=collapsed_line,
            used_orders=used_orders,
            campaign_ton_min=campaign_ton_min,
            campaign_ton_max=campaign_ton_max,
            chain_counter=chain_counter,
            blocked_successor_bucket=None,  # No blocking for retry - all released orders available
        )

        if chain is not None and len(chain.order_ids) >= 1:
            retry_chains.append(chain)
            retry_added_orders += len(chain.order_ids)

    # Append new chains to existing chains_by_line
    if collapsed_line in existing_chains_by_line:
        existing_chains_by_line[collapsed_line].extend(retry_chains)
    else:
        existing_chains_by_line[collapsed_line] = retry_chains

    # Update diagnostics
    diagnostics["line_balance_retry_added_orders"] = retry_added_orders
    diagnostics["line_balance_retry_added_chains"] = len(retry_chains)
    diagnostics["line_balance_retry_still_zero_after_retry"] = (retry_added_orders == 0)

    print(
        f"[APS][BalanceProtection] RETRY collapsed_line={collapsed_line}, "
        f"released={len(released_orders)}, "
        f"feasible={len(retry_available)}, "
        f"added_orders={retry_added_orders}, "
        f"added_chains={len(retry_chains)}, "
        f"still_zero={retry_added_orders == 0}"
    )

    return retry_added_orders


def _finalize_unused_orders(
    graph: TemplateEdgeGraph,
    used_orders: set[str],
    diagnostics: dict,
) -> List[dict]:
    """
    Final settlement for orders that were not placed in any chain.

    This function is called AFTER all lines have been processed, so we can
    correctly determine which orders are truly unused vs. just not-yet-tried
    on their compatible lines.

    CRITICAL TIMING: This must NOT be called inside the per-line loop.
    Calling it there would cause big_only orders to be marked as used after
    small_roll processes, blocking big_roll from ever seeing them.

    Args:
        graph: TemplateEdgeGraph with order and edge data
        used_orders: Set of order IDs that have been placed in chains
        diagnostics: Diagnostics dict (will be updated)

    Returns:
        List of dropped order records with proper final rejection reasons
    """
    dropped_seed_orders: List[dict] = []
    final_unused_count = 0
    final_dead_island_count = 0
    final_rejected_count = 0

    for oid, rec in graph.order_record.items():
        # Skip orders that have been placed in chains
        if oid in used_orders:
            continue

        # Determine allowed lines for this order
        raw_cap = str(rec.get("line_capability", "dual"))
        allowed_lines = _allowed_lines_for_order(raw_cap)
        is_dual = "big_roll" in allowed_lines and "small_roll" in allowed_lines

        # Check connectivity on all ALLOWED lines (not just the lines we've tried)
        has_connectivity_on_any_allowed_line = False
        for line in allowed_lines:
            if line == "big_roll":
                if graph.big_deg.get(oid, 0) > 0:
                    has_connectivity_on_any_allowed_line = True
                    break
            else:  # small_roll
                if graph.small_deg.get(oid, 0) > 0:
                    has_connectivity_on_any_allowed_line = True
                    break

        # Create drop record with appropriate reason
        drop = dict(rec)
        if has_connectivity_on_any_allowed_line:
            # Order has template edges on at least one allowed line
            # but was not selected during constructive building
            drop["drop_reason"] = "CONSTRUCTIVE_REJECTED"
            drop["secondary_reasons"] = (
                f"not_selected_in_any_chain;"
                f"allowed_lines={list(allowed_lines)}"
            )
            drop["dominant_drop_reason"] = "CONSTRUCTIVE_REJECTED"
            drop["isolated_line"] = "none"
            final_rejected_count += 1
        else:
            # Order has NO template edges on any of its allowed lines
            # This is a true dead island
            drop["drop_reason"] = "DEAD_ISLAND_ORDER"
            drop["secondary_reasons"] = (
                f"no_template_edges_on_any_allowed_line;"
                f"allowed_lines={list(allowed_lines)}"
            )
            drop["dominant_drop_reason"] = "DEAD_ISLAND_ORDER"
            drop["isolated_line"] = ",".join(allowed_lines) if allowed_lines else "none"
            final_dead_island_count += 1

        dropped_seed_orders.append(drop)
        final_unused_count += 1

    # Update diagnostics
    diagnostics["final_unused_orders_count"] = final_unused_count
    diagnostics["final_constructive_rejected_count"] = final_rejected_count
    diagnostics["final_dead_island_count"] = final_dead_island_count
    diagnostics["unused_settlement_after_all_lines"] = True

    print(
        f"[APS][ConstructiveFinalSettlement] unused_after_all_lines={final_unused_count}, "
        f"rejected={final_rejected_count}, dead_islands={final_dead_island_count}"
    )

    return dropped_seed_orders


def build_constructive_sequences(
    orders_df: pd.DataFrame,
    transition_pack: dict | None,
    cfg: PlannerConfig,
) -> ConstructiveBuildResult:
    """
    Build valid order chains using greedy constructive heuristic.

    This function implements the first layer of the Constructive + ALNS path:
    - Processes big_roll and small_roll separately
    - Starts new chains from high-priority, high-tonnage, high-connectivity orders
    - Only follows edges that exist in the template graph
    - Each order used at most once
    - Dead island orders are dropped

    Args:
    orders_df: DataFrame with orders (must have: order_id, tons, width, thickness,
    steel_group, due_rank, priority, line_capability)
    transition_pack: Dict containing tpl_df (template DataFrame)
    cfg: PlannerConfig with rule parameters

    Returns:
    ConstructiveBuildResult with chains, dropped orders, and diagnostics
    """
    print(f"[APS][RUN_PATH_FINGERPRINT] CONSTRUCTIVE_SEQUENCE_BUILDER_V2_20260416A")
    # Initialize result structures
    chains_by_line: Dict[str, List[ConstructiveChain]] = {
        "big_roll": [],
        "small_roll": [],
    }
    dropped_seed_orders: List[dict] = []
    diagnostics: Dict = {
        "lines_processed": [],
        "total_chains": 0,
        "total_orders_placed": 0,
        "total_tons_placed": 0.0,
        "dead_island_count": 0,
        "unused_orders_count": 0,
        "chain_details": [],
        # Bridge edge filtering diagnostics
        "filtered_virtual_bridge_edge_count": 0,
        "filtered_real_bridge_edge_count": 0,
        "constructive_edge_policy": "unknown",
        # Guarded virtual family edge diagnostics
        "greedy_virtual_family_edge_uses": 0,
        "greedy_virtual_family_edge_rejects": 0,
        "greedy_virtual_family_budget_blocked_count": 0,
        # Future-aware penalty diagnostics (guarded profile)
        "greedy_future_bridgeability_penalty_hits": 0,
        "greedy_tail_underfill_risk_penalty_hits": 0,
        "greedy_bridge_scarcity_penalty_hits": 0,
        "skipped_by_reverse_width_budget": 0,
        "accepted_with_reverse_width_budget": 0,
        "reverse_width_count_used_max": 0,
        "seed_scoring_enabled": True,
        "width_desc_seed_bias_enabled": True,
        "selected_seed_score_avg": 0.0,
        "selected_seed_width_percentile_avg": 0.0,
        "top_seed_width_avg": 0.0,
        "top_seed_extendability_avg": 0.0,
        "top_seed_formability_avg": 0.0,
        "continuation_bias_used": True,
        "continuation_bonus_total": 0.0,
        "continuation_cross_min_hits": 0,
        "chains_crossed_ton_min_during_constructive": 0,
        "successor_business_scoring_enabled": True,
        "width_desc_successor_bias_enabled": True,
        "successor_extendability_bias_enabled": True,
        "successor_group_continuity_bias_enabled": True,
        "successor_dead_end_penalty_enabled": True,
        "successor_reverse_budget_cost_total": 0.0,
        "dead_end_penalty_total": 0.0,
        "successor_zero_extendable_degree_count": 0,
        "successor_low_extendable_degree_count": 0,
        "successor_score_component_sum": 0.0,
        "successor_score_component_count": 0,
        "successor_extendability_component_sum": 0.0,
        "successor_dead_end_penalty_sum": 0.0,
        "dead_end_successor_count": 0,
        "virtual_rescue_attempt_count": 0,
        "virtual_rescue_eligible_count": 0,
        "virtual_rescue_selected_count": 0,
        "virtual_rescue_selected_big_roll_count": 0,
        "virtual_rescue_selected_small_roll_count": 0,
        "virtual_rescue_cross_min_count": 0,
        "virtual_rescue_rejected_by_hard": 0,
        "virtual_rescue_rejected_by_budget": 0,
        "successor_cross_min_selected": 0,
        "successor_near_min_selected": 0,
        "successor_deadend_rejected_under_min": 0,
        # Dual-order small-roll reserve diagnostics
        "dual_orders_with_small_roll_option": 0,
        "dual_orders_reserved_from_big_roll": 0,
        "small_roll_seed_count": 0,
        "small_roll_chain_count": 0,
        # Reserve bucket diagnostics
        "small_roll_reserve_bucket_size": 0,
        "small_roll_reserve_bucket_used_count": 0,
        "small_roll_reserve_bucket_released_to_big_count": 0,
    }

    # Handle empty input
    if orders_df.empty:
        return ConstructiveBuildResult(
            chains_by_line=chains_by_line,
            dropped_seed_orders=dropped_seed_orders,
            diagnostics=diagnostics,
        )

    # Get template DataFrame
    tpl_df = None
    if isinstance(transition_pack, dict):
        tpl_df = transition_pack.get("templates")
        if not isinstance(tpl_df, pd.DataFrame):
            tpl_df = pd.DataFrame()

    # Build template edge graph
    # ---- Single Candidate Graph: reuse from pipeline if available ----
    pre_built_cg = None
    if isinstance(transition_pack, dict):
        pre_built_cg = transition_pack.get("candidate_graph")
    graph = TemplateEdgeGraph(orders_df, tpl_df, cfg, candidate_graph=pre_built_cg)
    graph._init_used_orders()
    seed_diag_samples: list[dict] = []
    line_seed_diag_samples: dict[str, list[dict[str, Any]]] = defaultdict(list)

    # Track single-build diagnostics in result
    diagnostics["candidate_graph_source"] = graph.candidate_graph_source
    used_orders: set[str] = set()
    graph.set_used_orders(used_orders)

    # Get campaign ton limits
    campaign_ton_min = float(cfg.rule.campaign_ton_min) if cfg.rule else 700.0
    campaign_ton_max = float(cfg.rule.campaign_ton_max) if cfg.rule else 2000.0

    # Process each line
    # ---- Phase 0: Compute small_roll reserve bucket (TWO-LAYER: candidates + quota) ----
    cfg_model = getattr(cfg, "model", None)
    reserve_bucket_enabled = getattr(cfg_model, "small_roll_dual_reserve_bucket_enabled", True) if cfg_model else True
    reserve_bucket_ratio = float(getattr(cfg_model, "small_roll_dual_reserve_bucket_ratio", 0.45) if cfg_model else 0.45)
    reserve_bucket_max = int(getattr(cfg_model, "small_roll_dual_reserve_bucket_max_orders", 120) if cfg_model else 120)
    seed_first_enabled = getattr(cfg_model, "small_roll_seed_first_enabled", True) if cfg_model else True
    seed_min_orders = int(getattr(cfg_model, "small_roll_seed_min_orders", 20) if cfg_model else 20)
    seed_min_tons10 = int(getattr(cfg_model, "small_roll_seed_min_tons10", 5000) if cfg_model else 5000)

    # ---- NEW: Quota-based balanced allocation parameters ----
    quota_enabled = getattr(cfg_model, "small_roll_dual_reserve_quota_enabled", True) if cfg_model else True
    quota_min_orders = int(getattr(cfg_model, "small_roll_dual_reserve_quota_min_orders", 25) if cfg_model else 25)
    quota_min_tons10 = int(getattr(cfg_model, "small_roll_dual_reserve_quota_min_tons10", 6000) if cfg_model else 6000)
    quota_max_orders = int(getattr(cfg_model, "small_roll_dual_reserve_quota_max_orders", 60) if cfg_model else 60)
    quota_max_tons10 = int(getattr(cfg_model, "small_roll_dual_reserve_quota_max_tons10", 14000) if cfg_model else 14000)
    big_roll_release_after_seed = getattr(cfg_model, "big_roll_dual_release_after_small_seed", True) if cfg_model else True

    reserve_bucket: set[str] = set()
    small_roll_reserve_candidates: list[tuple] = []

    if reserve_bucket_enabled:
        # Candidate: line_capability in {dual, both, either} AND has edges on BOTH lines
        for oid, rec in graph.order_record.items():
            cap = str(rec.get("line_capability", "dual"))
            if cap not in {"dual", "both", "either"}:
                continue
            big_d = graph.big_deg.get(oid, 0)
            small_d = graph.small_deg.get(oid, 0)
            if big_d > 0 and small_d > 0:
                # Score: small_deg higher = more worth preserving for small_roll
                due_rank = _safe_int_value(rec.get("due_rank", 999), 999)
                tons = float(rec.get("tons", 0) or 0)
                score = (small_d * 100) + (tons * 0.1) + max(0, 100 - due_rank)
                small_roll_reserve_candidates.append((-score, oid, rec))  # negative for ascending sort

        reserve_seed_df = rank_seed_candidates(
            pd.DataFrame([dict(rec, order_id=oid, line="small_roll") for _, oid, rec in small_roll_reserve_candidates]),
            graph,
            cfg,
        )
        reserve_rank = {str(row["order_id"]): dict(row) for row in reserve_seed_df.to_dict("records")}
        if not reserve_seed_df.empty:
            diagnostics["top_seed_width_avg"] = float(reserve_seed_df.head(10)["width_headroom_score"].mean())
            diagnostics["top_seed_extendability_avg"] = float(reserve_seed_df.head(10)["extendability_score"].mean())
            diagnostics["top_seed_formability_avg"] = float(reserve_seed_df.head(10)["tonnage_formability_score"].mean())
        small_roll_reserve_candidates.sort(
            key=lambda item: (
                -float(reserve_rank.get(str(item[1]), {}).get("seed_score_total", 0.0)),
                item[0],
                str(item[1]),
            )
        )
        k = min(int(len(small_roll_reserve_candidates) * reserve_bucket_ratio), reserve_bucket_max)

        # ---- TWO-LAYER RESERVE BUCKET (candidates + quota) ----
        # Layer 1: reserve_candidates_all = top K candidates from ratio/max
        reserve_candidates_all = {oid for _, oid, _ in small_roll_reserve_candidates[:k]}

        # Layer 2: reserve_bucket_quota = actually locked portion (capped by quota max)
        # quota_max is the HARD ceiling: small_roll cannot lock more than this
        if quota_enabled:
            # Sort by priority: high small_deg, high tons, tight due_rank
            quota_sorted = sorted(
                [(oid, rec) for _, oid, rec in small_roll_reserve_candidates if oid in reserve_candidates_all],
                key=lambda x: (
                    -graph.small_deg.get(x[0], 0),  # high small_deg first
                    -float(x[1].get("tons", 0) or 0),  # high tons first
                    _safe_int_value(x[1].get("due_rank", 999), 999),  # tight due_rank first
                ),
            )
            # Truncate by quota_max (both orders and tons)
            quota_oids: list[str] = []
            quota_tons10 = 0
            for oid, rec in quota_sorted:
                oid_tons10 = int((float(rec.get("tons", 0) or 0)) * 10)
                if len(quota_oids) < quota_max_orders and quota_tons10 + oid_tons10 <= quota_max_tons10:
                    quota_oids.append(oid)
                    quota_tons10 += oid_tons10
                else:
                    break  # quota ceiling reached
            reserve_bucket = set(quota_oids)
            diagnostics["small_roll_reserve_candidates_all_count"] = len(reserve_candidates_all)
            diagnostics["small_roll_reserve_quota_count"] = len(reserve_bucket)
            diagnostics["small_roll_reserve_quota_tons10"] = quota_tons10
        else:
            # Fallback to old behavior: lock all top K candidates
            reserve_bucket = reserve_candidates_all.copy()
            diagnostics["small_roll_reserve_candidates_all_count"] = len(reserve_candidates_all)
            diagnostics["small_roll_reserve_quota_count"] = len(reserve_bucket)
            diagnostics["small_roll_reserve_quota_tons10"] = sum(
                int(float(graph.order_record.get(oid, {}).get("tons", 0) or 0) * 10)
                for oid in reserve_bucket
            )

        diagnostics["small_roll_reserve_bucket_size"] = len(reserve_bucket)
        diagnostics["small_roll_seed_first_enabled"] = bool(seed_first_enabled)
        if reserve_bucket:
            print(
                f"[APS][ReserveBucket] candidates={len(reserve_candidates_all)}, "
                f"quota_size={len(reserve_bucket)} (K={k}, ratio={reserve_bucket_ratio}), "
                f"quota_max_orders={quota_max_orders}, quota_max_tons10={quota_max_tons10}, "
                f"quota_enabled={quota_enabled}, seed_first={seed_first_enabled}"
            )
    # else: reserve_bucket stays empty, no-op

    # ---- Phase 1: small_roll seed-first with BALANCED QUOTA RELEASE ----
    # Priority: small_roll gets MINIMUM guarantee first; once min is met,
    # remaining reserve bucket is released to big_roll immediately (no "lock all" behavior).
    # Key changes from old logic:
    #   - OLD: "until seed is fully done, lock all reserve from big_roll"
    #   - NEW: "once min threshold reached, release remaining to big_roll"
    released_from_seed_phase: set[str] = set()
    small_roll_seed_orders_placed = 0
    small_roll_seed_tons10_placed = 0
    small_roll_seed_chains: list[ConstructiveChain] = []
    small_roll_seed_chain_counter: Dict[str, int] = {"small_roll": 0}
    small_roll_seed_used_oids: set[str] = set()

    # ---- Quota release conditions tracking ----
    # These control when reserve bucket is released to big_roll
    quota_min_reached_by_orders = False
    quota_min_reached_by_tons = False
    small_stalled_no_successor = False
    reserve_bucket_released_early = False
    reserve_release_trigger: str = "not_released"  # "quota_min_orders", "quota_min_tons", "small_stalled", "quota_not_met", "not_applicable"

    if seed_first_enabled and reserve_bucket:
        # ---- Step 1: Build small_roll chains from reserve bucket first ----
        # Sort reserve seeds by: high small_deg, high tons, tight due_rank
        reserve_seeds = [(oid, rec) for _, oid, rec in small_roll_reserve_candidates if oid in reserve_bucket]
        if reserve_seeds:
            reserve_seed_scored = rank_seed_candidates(
                pd.DataFrame([dict(rec, order_id=oid, line="small_roll") for oid, rec in reserve_seeds]),
                graph,
                cfg,
            )
            reserve_seed_rank = {str(row["order_id"]): dict(row) for row in reserve_seed_scored.to_dict("records")}
            reserve_seeds.sort(
                key=lambda item: (
                    -float(reserve_seed_rank.get(str(item[0]), {}).get("seed_score_total", 0.0)),
                    str(item[0]),
                )
            )

            # ---- Build chains and track quota progress ----
            for oid, rec in reserve_seeds:
                if oid in used_orders:
                    continue
                # Only start a chain if the order has outgoing edges on small_roll.
                # Orders with no outgoing edges are dead islands; skip to avoid
                # consuming them before big_roll gets a chance.
                if not any(
                    str(tpl.get("line", "big_roll")) == "small_roll"
                    for _, tpl in graph.out_edges.get(oid, [])
                ):
                    continue

                chain, chain_orders = _build_single_chain(
                    graph=graph,
                    start_oid=oid,
                    line="small_roll",
                    used_orders=used_orders,
                    campaign_ton_min=campaign_ton_min,
                    campaign_ton_max=campaign_ton_max,
                    chain_counter=small_roll_seed_chain_counter,
                    blocked_successor_bucket=None,  # small_roll has full access to all orders
                )
                if chain is not None and len(chain.order_ids) >= 1:
                    small_roll_seed_chains.append(chain)
                    if oid in reserve_seed_rank:
                        seed_diag_samples.append(dict(reserve_seed_rank[oid]))
                        line_seed_diag_samples["small_roll"].append(dict(reserve_seed_rank[oid]))
                    small_roll_seed_orders_placed += len(chain.order_ids)
                    small_roll_seed_tons10_placed += int(chain.total_tons * 10)
                    small_roll_seed_used_oids.update(chain.order_ids)

                    # ---- Check quota release conditions in real-time ----
                    # If big_roll_release_after_seed is True: release as soon as quota_min threshold is reached
                    # If False: keep locking until seed phase completes
                    if big_roll_release_after_seed:
                        # Track if quota_min thresholds are reached
                        if small_roll_seed_orders_placed >= quota_min_orders:
                            quota_min_reached_by_orders = True
                        if small_roll_seed_tons10_placed >= quota_min_tons10:
                            quota_min_reached_by_tons = True

                        # Release condition A or B met via quota_min: release remaining reserve immediately
                        if quota_min_reached_by_orders or quota_min_reached_by_tons:
                            # Small_roll got its QUOTA MINIMUM guarantee; release rest to big_roll
                            released_from_seed_phase = reserve_bucket - small_roll_seed_used_oids
                            reserve_bucket_released_early = True
                            reserve_release_trigger = "quota_min_orders" if quota_min_reached_by_orders else "quota_min_tons"
                            diagnostics["reserve_bucket_released_early_to_big"] = True
                            diagnostics["reserve_bucket_released_count"] = len(released_from_seed_phase)
                            diagnostics["reserve_bucket_released_tons10"] = sum(
                                int(float(graph.order_record.get(oid, {}).get("tons", 0) or 0) * 10)
                                for oid in released_from_seed_phase
                            )
                            print(
                                f"[APS][SeedFirst] QUOTA MIN reached: orders={small_roll_seed_orders_placed}, "
                                f"tons10={small_roll_seed_tons10_placed} "
                                f"(quota_min_orders={quota_min_orders}, quota_min_tons10={quota_min_tons10}). "
                                f"Released {len(released_from_seed_phase)} remaining reserve to big_roll "
                                f"(trigger={reserve_release_trigger})."
                            )
                            # Stop seed phase early - no need to continue consuming reserve
                            break

                        # Check condition C: small_roll stalled (no more successors)
                        # If this seed has no valid successors, consider it stalled
                        if not any(
                            str(tpl.get("line", "big_roll")) == "small_roll"
                            for _, tpl in graph.out_edges.get(chain.end_order_id, [])
                        ):
                            small_stalled_no_successor = True
                            reserve_release_trigger = "small_stalled"
                            # Even if quota_min not reached, release remaining to give big_roll a chance
                            released_from_seed_phase = reserve_bucket - small_roll_seed_used_oids
                            reserve_bucket_released_early = True
                            diagnostics["reserve_bucket_released_early_to_big"] = True
                            diagnostics["reserve_bucket_released_count"] = len(released_from_seed_phase)
                            diagnostics["reserve_bucket_released_tons10"] = sum(
                                int(float(graph.order_record.get(oid, {}).get("tons", 0) or 0) * 10)
                                for oid in released_from_seed_phase
                            )
                            print(
                                f"[APS][SeedFirst] small_roll STALLED (no successors), "
                                f"orders={small_roll_seed_orders_placed}, tons10={small_roll_seed_tons10_placed}. "
                                f"Released {len(released_from_seed_phase)} reserve to big_roll "
                                f"(trigger=small_stalled)."
                            )
                            break

        # ---- Step 2: If not released early, finalize seed phase ----
        # This handles cases where:
        #   - big_roll_release_after_seed = False (legacy behavior)
        #   - Seed phase completed but quota_min threshold not met
        if not reserve_bucket_released_early:
            # Final quota_min threshold check
            quota_min_reached = (
                small_roll_seed_orders_placed >= quota_min_orders
                and small_roll_seed_tons10_placed >= quota_min_tons10
            )

            if quota_min_reached and big_roll_release_after_seed:
                # QUOTA MIN met: small_roll got its quota guarantee.
                # Release remaining reserve bucket to big_roll.
                released_from_seed_phase = reserve_bucket - small_roll_seed_used_oids
                reserve_bucket_released_early = True
                reserve_release_trigger = "quota_min_orders" if quota_min_reached_by_orders else "quota_min_tons"
                diagnostics["reserve_bucket_released_early_to_big"] = True
                diagnostics["reserve_bucket_released_count"] = len(released_from_seed_phase)
                diagnostics["reserve_bucket_released_tons10"] = sum(
                    int(float(graph.order_record.get(oid, {}).get("tons", 0) or 0) * 10)
                    for oid in released_from_seed_phase
                )
                print(
                    f"[APS][SeedFirst] QUOTA MIN reached (seed phase done): "
                    f"orders={small_roll_seed_orders_placed}, tons10={small_roll_seed_tons10_placed} "
                    f"(quota_min_orders={quota_min_orders}, quota_min_tons10={quota_min_tons10}). "
                    f"Released {len(released_from_seed_phase)} reserve to big_roll."
                )
            else:
                # Quota min NOT met: release remaining reserve bucket to big_roll.
                # This is the safety net: even if small_roll couldn't reach quota_min,
                # what wasn't used goes to big_roll.
                reserve_release_trigger = "quota_not_met"
                released_from_seed_phase = reserve_bucket - small_roll_seed_used_oids
                print(
                    f"[APS][SeedFirst] QUOTA MIN NOT MET: orders={small_roll_seed_orders_placed}, "
                    f"tons10={small_roll_seed_tons10_placed} "
                    f"(quota_min_orders={quota_min_orders}, quota_min_tons10={quota_min_tons10}). "
                    f"Released {len(released_from_seed_phase)} reserve bucket orders to big_roll."
                )

        # ---- Inject released orders into big_roll's drop pool ----
        for oid in released_from_seed_phase:
            if oid in used_orders:
                continue  # already consumed
            rec = graph.order_record.get(oid, {})
            drop = dict(rec)
            drop["drop_reason"] = "RESERVE_BUCKET_RELEASED_FROM_SEED_PHASE"
            drop["secondary_reasons"] = (
                f"small_roll_seeded={small_roll_seed_orders_placed}_orders_"
                f"{small_roll_seed_tons10_placed}_tons10;"
                f"early_release={reserve_bucket_released_early}"
            )
            drop["dominant_drop_reason"] = "RESERVE_BUCKET_RELEASED_FROM_SEED_PHASE"
            drop["isolated_line"] = "small_roll"
            dropped_seed_orders.append(drop)
            # used_orders.add(oid)  # DO NOT mark used — let big_roll pick these up

        diagnostics["small_roll_seed_orders"] = small_roll_seed_orders_placed
        diagnostics["small_roll_seed_tons10"] = small_roll_seed_tons10_placed
        diagnostics["small_roll_reserve_bucket_used_count"] = sum(
            len(c.order_ids) for c in small_roll_seed_chains
        )
        # Quota min tracking (controls release timing)
        diagnostics["quota_min_reached_by_orders"] = quota_min_reached_by_orders
        diagnostics["quota_min_reached_by_tons"] = quota_min_reached_by_tons
        diagnostics["small_stalled_no_successor"] = small_stalled_no_successor
        diagnostics["reserve_release_trigger"] = reserve_release_trigger
        if "small_roll_reserve_bucket_released_to_big_count" not in diagnostics:
            diagnostics["small_roll_reserve_bucket_released_to_big_count"] = len(released_from_seed_phase)

    # ---- Phase 2: Line processing ----
    # When seed_first is enabled, processing order is small_roll FIRST, then big_roll.
    # When seed_first is disabled, big_roll goes first (original order).
    if seed_first_enabled:
        line_processing_order = ["small_roll", "big_roll"]
    else:
        line_processing_order = ["big_roll", "small_roll"]

    for line in line_processing_order:
        line_chains: List[ConstructiveChain] = []
        chain_counter: Dict[str, int] = {line: 0}
        line_dead_islands = 0

        # ---- Small_roll seed phase: seed chains are already built; append them ----
        if line == "small_roll" and small_roll_seed_chains:
            line_chains.extend(small_roll_seed_chains)
            for c in small_roll_seed_chains:
                chain_counter[c.line] = chain_counter.get(c.line, 0) + 1

        # Get all orders that can run on this line
        line_orders = []
        for oid, rec in graph.order_record.items():
            cap = str(rec.get("line_capability", "dual"))
            if cap in {"small_only", "small"} and line == "big_roll":
                continue
            if cap in {"big_only", "large", "big"} and line == "small_roll":
                continue
            line_orders.append((oid, rec))

        # Check for dead islands first
        # IMPORTANT: dual orders (line_capability in {dual, both, either}) are NOT
        # consumed by the dead-island logic on a single line. They remain available
        # for the other line to use.
        # An order is a true dead island only if:
        #   - It is dead on ALL lines it can run on (dual orders are NOT dead islands here)
        #   - OR it is small_only/big_only and dead on its only compatible line
        remaining_orders = []
        for oid, rec in line_orders:
            cap = str(rec.get("line_capability", "dual"))
            is_dual = cap in {"dual", "both", "either"}
            if graph.is_dead_island(oid, line):
                if is_dual:
                    # Dual orders: skip dead-island classification here.
                    # They might have edges on the other line. Defer to the other line's pass.
                    # Keep them in remaining_orders so they can be chain-built on THIS line too.
                    remaining_orders.append((oid, rec))
                else:
                    # Non-dual (big_only / small_only): truly dead on this line
                    drop = dict(rec)
                    drop["drop_reason"] = "DEAD_ISLAND_ORDER"
                    drop["secondary_reasons"] = f"no_template_edges_on_{line}"
                    drop["dominant_drop_reason"] = "DEAD_ISLAND_ORDER"
                    drop["isolated_line"] = line
                    dropped_seed_orders.append(drop)
                    used_orders.add(oid)
                    line_dead_islands += 1
            else:
                remaining_orders.append((oid, rec))

        # ---- Successor blocking: compute locked vs released sets ----
        # Only orders STILL in locked_bucket are blocked from big_roll.
        # Released orders (released_from_seed_phase) are fully available:
        #   - As chain SEEDS (filtered in remaining_orders above)
        #   - As chain SUCCESSORS (filtered in _build_single_chain below)
        # This is the key fix: previously, released orders were blocked as successors
        # even though they were visible as seeds.
        locked_bucket: set[str] = set()
        if line == "big_roll" and reserve_bucket:
            locked_bucket = reserve_bucket - released_from_seed_phase
            remaining_orders = [(oid, rec) for oid, rec in remaining_orders if oid not in locked_bucket]
            diagnostics["big_roll_blocked_by_small_quota_count"] = len(locked_bucket)
            diagnostics["big_roll_blocked_successor_count"] = len(locked_bucket)
            # Count released orders that are now visible to big_roll
            released_visible = released_from_seed_phase & {oid for oid, _ in remaining_orders}
            diagnostics["big_roll_released_dual_visible_count"] = len(released_visible)
            diagnostics["big_roll_released_successor_visible_count"] = len(released_visible)

        diagnostics["dead_island_count"] += line_dead_islands

        remaining_seed_scored = rank_seed_candidates(
            pd.DataFrame([dict(rec, order_id=oid, line=line) for oid, rec in remaining_orders]),
            graph,
            cfg,
        )
        remaining_seed_rank = {str(row["order_id"]): dict(row) for row in remaining_seed_scored.to_dict("records")}
        if not remaining_seed_scored.empty and diagnostics.get("top_seed_width_avg", 0.0) == 0.0:
            diagnostics["top_seed_width_avg"] = float(remaining_seed_scored.head(10)["width_headroom_score"].mean())
            diagnostics["top_seed_extendability_avg"] = float(remaining_seed_scored.head(10)["extendability_score"].mean())
            diagnostics["top_seed_formability_avg"] = float(remaining_seed_scored.head(10)["tonnage_formability_score"].mean())
        remaining_orders.sort(
            key=lambda item: (
                -float(remaining_seed_rank.get(str(item[0]), {}).get("seed_score_total", 0.0)),
                str(item[0]),
            )
        )

        # Build chains greedily
        for oid, rec in remaining_orders:
            if oid in used_orders:
                continue

            # Skip orders with no outgoing edges on this line — they are dead islands.
            # They remain available for the other line to try.
            # (Dual orders were already excluded from dead_island classification above.)
            if not any(
                str(tpl.get("line", "big_roll")) == line
                for _, tpl in graph.out_edges.get(oid, [])
            ):
                continue  # dead island on this line; skip, do NOT consume

            # ---- Build chain with successor blocking ----
            # For big_roll: only block STILL-LOCKED orders as successors (locked_bucket).
            # Released orders (released_from_seed_phase) can be used as successors.
            # For small_roll: no successor blocking (small_roll has full access to all orders).
            chain, chain_orders = _build_single_chain(
                graph=graph,
                start_oid=oid,
                line=line,
                used_orders=used_orders,
                campaign_ton_min=campaign_ton_min,
                campaign_ton_max=campaign_ton_max,
                chain_counter=chain_counter,
                blocked_successor_bucket=locked_bucket if line == "big_roll" else None,
            )

            if chain is not None and len(chain.order_ids) >= 1:
                line_chains.append(chain)
                if oid in remaining_seed_rank:
                    seed_diag_samples.append(dict(remaining_seed_rank[oid]))
                    line_seed_diag_samples[str(line)].append(dict(remaining_seed_rank[oid]))

        chains_by_line[line] = line_chains

        # Line-level diagnostics
        line_order_count = len(line_orders)
        line_orders_placed = sum(len(c.order_ids) for c in line_chains)
        line_tons_placed = sum(c.total_tons for c in line_chains)
        line_avg_length = line_orders_placed / max(1, len(line_chains))
        line_avg_tons = line_tons_placed / max(1, len(line_chains))

        diagnostics["lines_processed"].append(line)
        diagnostics["total_chains"] += len(line_chains)
        diagnostics["total_orders_placed"] += line_orders_placed
        diagnostics["total_tons_placed"] += line_tons_placed

        for chain in line_chains:
            diagnostics["chain_details"].append({
                "chain_id": chain.chain_id,
                "line": chain.line,
                "start_order_id": chain.start_order_id,
                "end_order_id": chain.end_order_id,
                "order_count": len(chain.order_ids),
                "total_tons": chain.total_tons,
                "edge_count": chain.edge_count,
                "steel_groups": list(chain.steel_groups),
            })

        print(
            f"[APS][ConstructiveBuilder] line={line}, "
            f"chains={len(line_chains)}, "
            f"orders_placed={line_orders_placed}/{line_order_count}, "
            f"dead_islands={line_dead_islands}, "
            f"avg_length={line_avg_length:.1f}, "
            f"avg_tons={line_avg_tons:.0f}"
        )
        _log_seed_scoring(line, remaining_seed_scored, line_seed_diag_samples.get(str(line), []))

        # ---- Dual-order small-roll reserve diagnostics ----
        dual_with_small = 0
        reserved_from_big = 0
        for oid, rec in graph.order_record.items():
            cap = str(rec.get("line_capability", "dual"))
            if cap in {"dual", "both", "either"}:
                big_d = graph.big_deg.get(oid, 0)
                small_d = graph.small_deg.get(oid, 0)
                if small_d > 0:
                    dual_with_small += 1
                    # If it has big_roll connectivity but we prefer small_roll,
                    # it was "reserved from big_roll"
                    if big_d > 0:
                        reserved_from_big += 1

        if line == "big_roll":
            # Count how many dual orders went to big_roll vs small_roll
            dual_on_big = 0
            for chain in line_chains:
                for oid in chain.order_ids:
                    cap = str(graph.order_record.get(oid, {}).get("line_capability", "dual"))
                    if cap in {"dual", "both", "either"}:
                        dual_on_big += 1
            diagnostics["dual_orders_reserved_from_big_roll"] = dual_with_small - dual_on_big

        if line == "small_roll":
            diagnostics["dual_orders_with_small_roll_option"] = dual_with_small
            diagnostics["small_roll_seed_count"] = len(remaining_orders)
            diagnostics["small_roll_chain_count"] = len(line_chains)
            # Seed-first phase diagnostics (if seed_first_enabled, these were set above)
            diagnostics["small_roll_seed_orders"] = diagnostics.get(
                "small_roll_seed_orders", small_roll_seed_orders_placed
            )
            diagnostics["small_roll_seed_tons10"] = diagnostics.get(
                "small_roll_seed_tons10", small_roll_seed_tons10_placed
            )

            # ---- Release unused reserve bucket orders back to big_roll's drop pool ----
            # Only relevant when seed_first is DISABLED (seed-first already handled release above).
            # Orders in reserve_bucket that were NOT consumed by small_roll are added to
            # dropped_seed_orders (NOT used_orders) so big_roll can pick them up in its pass.
            if reserve_bucket and not seed_first_enabled:
                reserved_used_by_small = {
                    oid for chain in line_chains for oid in chain.order_ids
                } & reserve_bucket
                released_to_big = reserve_bucket - reserved_used_by_small
                diagnostics["small_roll_reserve_bucket_used_count"] = len(reserved_used_by_small)
                if "small_roll_reserve_bucket_released_to_big_count" not in diagnostics:
                    diagnostics["small_roll_reserve_bucket_released_to_big_count"] = len(released_to_big)
                for oid in released_to_big:
                    if oid in used_orders:
                        continue  # already used by something else
                    rec = graph.order_record.get(oid, {})
                    drop = dict(rec)
                    drop["drop_reason"] = "RESERVE_BUCKET_RELEASED"
                    drop["secondary_reasons"] = "small_roll_could_not_use_all_reserve_orders"
                    drop["dominant_drop_reason"] = "RESERVE_BUCKET_RELEASED"
                    drop["isolated_line"] = "small_roll"
                    dropped_seed_orders.append(drop)
                    # used_orders.add(oid)  # DO NOT mark used — let big_roll pick these up
                if released_to_big:
                    print(
                        f"[APS][ReserveBucket] released {len(released_to_big)} unused reserve orders "
                        f"to big_roll (used_by_small={len(reserved_used_by_small)}, "
                        f"total_bucket={len(reserve_bucket)})"
                    )

    # ---- Phase 3: Anti-collapse protection (balance check) ----
    # Detect if one line got 0 orders despite having feasible nodes.
    # If so, trigger a lightweight retry by releasing remaining locked quota and
    # actually rebuilding chains for the collapsed line.
    big_roll_placed = sum(len(c.order_ids) for c in chains_by_line.get("big_roll", []))
    small_roll_placed = sum(len(c.order_ids) for c in chains_by_line.get("small_roll", []))
    diagnostics["big_roll_final_orders_placed"] = big_roll_placed
    diagnostics["small_roll_final_orders_placed"] = small_roll_placed
    diagnostics["line_balance_collapse_detected"] = False
    diagnostics["line_balance_retry_triggered"] = False
    diagnostics["line_balance_retry_collapsed_line"] = None
    diagnostics["line_balance_retry_released_count"] = 0
    diagnostics["line_balance_retry_added_orders"] = 0
    diagnostics["line_balance_retry_added_chains"] = 0
    diagnostics["line_balance_retry_still_zero_after_retry"] = False

    # Check for collapse: one line got 0 despite having feasible nodes
    if (big_roll_placed == 0 or small_roll_placed == 0) and reserve_bucket:
        # Determine which line collapsed
        collapsed_line = "big_roll" if big_roll_placed == 0 else "small_roll"
        surviving_line = "small_roll" if collapsed_line == "big_roll" else "big_roll"

        # Check if the collapsed line has ANY feasible nodes (dual orders with edges)
        collapsed_has_feasible = False
        for oid, rec in graph.order_record.items():
            cap = str(rec.get("line_capability", "dual"))
            if cap in {"dual", "both", "either"}:
                if collapsed_line == "big_roll":
                    if graph.big_deg.get(oid, 0) > 0:
                        collapsed_has_feasible = True
                        break
                else:
                    if graph.small_deg.get(oid, 0) > 0:
                        collapsed_has_feasible = True
                        break

        if collapsed_has_feasible:
            # Collapse detected: surviving line consumed ALL reserve, blocking collapsed line
            diagnostics["line_balance_collapse_detected"] = True
            locked_bucket = reserve_bucket - released_from_seed_phase
            if locked_bucket:
                diagnostics["line_balance_retry_triggered"] = True
                retry_released = locked_bucket
                retry_released_tons10 = sum(
                    int(float(graph.order_record.get(oid, {}).get("tons", 0) or 0) * 10)
                    for oid in retry_released
                )
                diagnostics["line_balance_retry_collapsed_line"] = collapsed_line
                diagnostics["line_balance_retry_released_count"] = len(retry_released)
                diagnostics["reserve_bucket_released_count"] = (
                    diagnostics.get("reserve_bucket_released_count", 0) + len(retry_released)
                )
                diagnostics["reserve_bucket_released_tons10"] = (
                    diagnostics.get("reserve_bucket_released_tons10", 0) + retry_released_tons10
                )
                print(
                    f"[APS][BalanceProtection] COLLAPSE DETECTED: {collapsed_line}=0 orders despite feasible nodes. "
                    f"Releasing {len(retry_released)} locked quota to {collapsed_line}. "
                    f"({surviving_line} had {sum(len(c.order_ids) for c in chains_by_line.get(surviving_line, []))} orders)"
                )

                # ---- Actually retry building chains for the collapsed line ----
                # Filter out orders that were already used by surviving line
                retry_released_unused = {oid for oid in retry_released if oid not in used_orders}
                _retry_line_after_quota_release(
                    collapsed_line=collapsed_line,
                    graph=graph,
                    used_orders=used_orders,
                    released_orders=retry_released_unused,
                    campaign_ton_min=campaign_ton_min,
                    campaign_ton_max=campaign_ton_max,
                    diagnostics=diagnostics,
                    existing_chains_by_line=chains_by_line,
                )

                # Update final placed counts after retry
                if collapsed_line == "big_roll":
                    big_roll_placed = sum(len(c.order_ids) for c in chains_by_line.get("big_roll", []))
                    diagnostics["big_roll_final_orders_placed"] = big_roll_placed
                else:
                    small_roll_placed = sum(len(c.order_ids) for c in chains_by_line.get("small_roll", []))
                    diagnostics["small_roll_final_orders_placed"] = small_roll_placed

    # ---- Phase 4: Final settlement of unused orders ----
    # This must be called AFTER all lines have been processed.
    # Previously, unused orders were settled inside the per-line loop, which caused
    # big_only orders to be marked as used after small_roll processed, blocking big_roll.
    # Now we wait until all lines are done before finalizing unused orders.
    final_dropped = _finalize_unused_orders(
        graph=graph,
        used_orders=used_orders,
        diagnostics=diagnostics,
    )
    dropped_seed_orders.extend(final_dropped)

    # Final diagnostics summary
    diagnostics["total_orders_input"] = len(orders_df)
    diagnostics["total_orders_dropped"] = len(dropped_seed_orders)
    diagnostics["utilization_rate"] = (
        diagnostics["total_orders_placed"] / max(1, len(orders_df))
    )
    if seed_diag_samples:
        seed_diag_df = pd.DataFrame(seed_diag_samples)
        diagnostics["selected_seed_score_avg"] = float(seed_diag_df["seed_score_total"].mean())
        diagnostics["selected_seed_width_percentile_avg"] = float(seed_diag_df["seed_width_percentile"].mean())
    diagnostics["dead_end_successor_count"] = int(
        graph.candidate_graph_diagnostics.get("dead_end_successor_count", 0) or 0
    )
    diagnostics["successor_reverse_budget_cost_total"] = float(
        graph.candidate_graph_diagnostics.get("successor_reverse_budget_cost_total", 0.0) or 0.0
    )
    diagnostics["dead_end_penalty_total"] = float(
        graph.candidate_graph_diagnostics.get("dead_end_penalty_total", 0.0) or 0.0
    )
    diagnostics["successor_zero_extendable_degree_count"] = int(
        graph.candidate_graph_diagnostics.get("successor_zero_extendable_degree_count", 0) or 0
    )
    diagnostics["successor_low_extendable_degree_count"] = int(
        graph.candidate_graph_diagnostics.get("successor_low_extendable_degree_count", 0) or 0
    )
    _succ_score_count = int(graph.candidate_graph_diagnostics.get("successor_score_component_count", 0) or 0)
    diagnostics["successor_score_component_avg"] = float(
        (graph.candidate_graph_diagnostics.get("successor_score_component_sum", 0.0) or 0.0)
        / max(1, _succ_score_count)
    )
    diagnostics["successor_extendability_component_avg"] = float(
        (graph.candidate_graph_diagnostics.get("successor_extendability_component_sum", 0.0) or 0.0)
        / max(1, _succ_score_count)
    )
    diagnostics["successor_dead_end_penalty_avg"] = float(
        (graph.candidate_graph_diagnostics.get("successor_dead_end_penalty_sum", 0.0) or 0.0)
        / max(1, _succ_score_count)
    )
    diagnostics["virtual_rescue_attempt_count"] = int(graph.virtual_rescue_attempt_count)
    diagnostics["virtual_rescue_eligible_count"] = int(graph.virtual_rescue_eligible_count)
    diagnostics["virtual_rescue_selected_count"] = int(graph.virtual_rescue_selected_count)
    diagnostics["virtual_rescue_selected_big_roll_count"] = int(graph.virtual_rescue_selected_big_roll_count)
    diagnostics["virtual_rescue_selected_small_roll_count"] = int(graph.virtual_rescue_selected_small_roll_count)
    diagnostics["virtual_rescue_cross_min_count"] = int(graph.virtual_rescue_cross_min_count)
    diagnostics["virtual_rescue_rejected_by_hard"] = int(graph.virtual_rescue_rejected_by_hard)
    diagnostics["virtual_rescue_rejected_by_budget"] = int(graph.virtual_rescue_rejected_by_budget)
    diagnostics["successor_cross_min_selected"] = int(graph.successor_cross_min_selected)
    diagnostics["successor_near_min_selected"] = int(graph.successor_near_min_selected)
    diagnostics["successor_deadend_rejected_under_min"] = int(graph.successor_deadend_rejected_under_min)

    diagnostics["virtual_rescue_line_stats"] = {
        str(line): dict(stats) for line, stats in graph.virtual_rescue_line_stats.items()
    }
    for _line in diagnostics.get("lines_processed", []):
        _line = str(_line)
        line_reason_counts = dict(graph.virtual_rescue_reason_counts_by_line.get(_line, {}))
        line_stats = dict(graph.virtual_rescue_line_stats.get(_line, {}))
        line_selected = int(
            graph.virtual_rescue_selected_big_roll_count if _line == "big_roll"
            else graph.virtual_rescue_selected_small_roll_count
        )
        reason_summary = ",".join(f"{k}:{v}" for k, v in sorted(line_reason_counts.items())) if line_reason_counts else "none"
        print(
            f"[APS][VirtualRescueLine] "
            f"line={_line}, "
            f"raw_virtual_seen={int(line_stats.get('raw_virtual_seen', 0) or 0)}, "
            f"hard_pass={int(line_stats.get('hard_pass', 0) or 0)}, "
            f"hard_reject={int(line_stats.get('hard_reject', 0) or 0)}, "
            f"attempts={int(line_stats.get('attempt', 0) or 0)}, "
            f"eligible={int(line_stats.get('eligible', 0) or 0)}, "
            f"selected={line_selected}, "
            f"reject_width={int(line_stats.get('reject_width', 0) or 0)}, "
            f"reject_thickness={int(line_stats.get('reject_thickness', 0) or 0)}, "
            f"reject_temp={int(line_stats.get('reject_temp', 0) or 0)}, "
            f"reject_group={int(line_stats.get('reject_group', 0) or 0)}, "
            f"reject_virtual_chain={int(line_stats.get('reject_virtual_chain', 0) or 0)}, "
            f"reject_reverse_budget={int(line_stats.get('reject_reverse_budget', 0) or 0)}, "
            f"reject_real_successor_rich={int(line_stats.get('reject_real_successor_rich', 0) or 0)}, "
            f"reject_not_under_min={int(line_stats.get('reject_not_under_min', 0) or 0)}, "
            f"reject_not_closer_to_min={int(line_stats.get('reject_not_closer_to_min', 0) or 0)}, "
            f"reject_budget={int(line_stats.get('reject_virtual_chain_budget', 0) or 0)}, "
            f"reasons={reason_summary}"
        )

    print(
        f"[APS][SuccessorScoring] "
        f"cross_min_hits={graph.continuation_cross_min_hits}, "
        f"cross_min_selected={graph.successor_cross_min_selected}, "
        f"near_min_selected={graph.successor_near_min_selected}, "
        f"deadend_rejected_under_min={graph.successor_deadend_rejected_under_min}, "
        f"virtual_rescue_selected={graph.virtual_rescue_selected_count}"
    )

    print(
        f"[APS][ConstructiveBuilder] total: "
        f"chains={diagnostics['total_chains']}, "
        f"placed={diagnostics['total_orders_placed']}/{diagnostics['total_orders_input']}, "
        f"dropped={diagnostics['total_orders_dropped']}, "
        f"dead_islands={diagnostics['dead_island_count']}, "
        f"final_unused={diagnostics.get('final_unused_orders_count', 0)}, "
        f"final_rejected={diagnostics.get('final_constructive_rejected_count', 0)}, "
        f"final_dead_islands={diagnostics.get('final_dead_island_count', 0)}, "
        f"utilization={diagnostics['utilization_rate']:.2%}, "
        f"edge_policy={graph.edge_policy}, "
        f"greedy_family_uses={graph.greedy_virtual_family_edge_uses}, "
        f"greedy_family_rejects={graph.greedy_virtual_family_edge_rejects}, "
        f"greedy_family_budget_blocked={graph.greedy_virtual_family_budget_blocked_count}, "
        f"accepted_direct={graph.accepted_direct_edge_count}, "
        f"accepted_real_bridge={graph.accepted_real_bridge_edge_count}, "
        f"accepted_virtual_bridge={graph.accepted_virtual_bridge_edge_count}, "
        f"accepted_virtual_to_real={graph.accepted_virtual_to_real_edge_count}, "
        f"accepted_virtual_to_virtual={graph.accepted_virtual_to_virtual_edge_count}, "
        f"accepted_virtual_chain_paths={graph.candidate_graph_diagnostics.get('accepted_virtual_chain_paths', 0)}, "
        f"virtual_real_edge_count_effective={graph.candidate_graph_diagnostics.get('virtual_real_edge_count_effective', 0)}, "
        f"accepted_virtual_family={graph.accepted_virtual_bridge_family_edge_count}, "
        f"accepted_direct_after_hard_filter={graph.accepted_direct_edge_count}, "
        f"accepted_real_bridge_after_hard_filter={graph.accepted_real_bridge_edge_count}, "
        f"accepted_virtual_bridge_after_hard_filter={graph.accepted_virtual_bridge_edge_count}, "
        f"rejected_edges_by_hard_filter_total={graph.rejected_edges_by_hard_filter_total}, "
        f"skipped_by_final_width_rule={graph.skipped_by_final_width_rule}, "
        f"skipped_by_final_thickness_rule={graph.skipped_by_final_thickness_rule}, "
        f"skipped_by_final_temp_rule={graph.skipped_by_final_temp_rule}, "
        f"skipped_by_final_group_rule={graph.skipped_by_final_group_rule}, "
        f"skipped_by_virtual_chain_rule={graph.skipped_by_virtual_chain_rule}, "
        f"skipped_by_reverse_width_budget={graph.skipped_by_reverse_width_budget}, "
        f"accepted_with_reverse_width_budget={graph.accepted_with_reverse_width_budget}, "
        f"reverse_width_count_used_max={graph.reverse_width_count_used_max}, "
        f"seed_scoring_enabled={True}, "
        f"width_desc_seed_bias_enabled={True}, "
        f"selected_seed_score_avg={diagnostics.get('selected_seed_score_avg', 0.0):.3f}, "
        f"selected_seed_width_percentile_avg={diagnostics.get('selected_seed_width_percentile_avg', 0.0):.3f}, "
        f"top_seed_width_avg={diagnostics.get('top_seed_width_avg', 0.0):.3f}, "
        f"top_seed_extendability_avg={diagnostics.get('top_seed_extendability_avg', 0.0):.3f}, "
        f"top_seed_formability_avg={diagnostics.get('top_seed_formability_avg', 0.0):.3f}, "
        f"continuation_bias_used={True}, "
        f"continuation_bonus_total={graph.continuation_bonus_total:.3f}, "
        f"continuation_cross_min_hits={graph.continuation_cross_min_hits}, "
        f"chains_crossed_ton_min_during_constructive={graph.chains_crossed_ton_min_during_constructive}, "
        f"virtual_rescue_attempt_count={graph.virtual_rescue_attempt_count}, "
        f"virtual_rescue_eligible_count={graph.virtual_rescue_eligible_count}, "
        f"virtual_rescue_selected_count={graph.virtual_rescue_selected_count}, "
        f"virtual_rescue_cross_min_count={graph.virtual_rescue_cross_min_count}, "
        f"successor_business_scoring_enabled={True}, "
        f"width_desc_successor_bias_enabled={True}, "
        f"successor_extendability_bias_enabled={True}, "
        f"successor_dead_end_penalty_enabled={True}, "
        f"successor_group_continuity_bias_enabled={True}, "
        f"successor_reverse_budget_cost_total={diagnostics.get('successor_reverse_budget_cost_total', 0.0):.3f}, "
        f"successor_zero_extendable_degree_count={diagnostics.get('successor_zero_extendable_degree_count', 0)}, "
        f"successor_low_extendable_degree_count={diagnostics.get('successor_low_extendable_degree_count', 0)}, "
        f"successor_score_component_avg={diagnostics.get('successor_score_component_avg', 0.0):.3f}, "
        f"successor_extendability_component_avg={diagnostics.get('successor_extendability_component_avg', 0.0):.3f}, "
        f"successor_dead_end_penalty_avg={diagnostics.get('successor_dead_end_penalty_avg', 0.0):.3f}, "
        f"dead_end_successor_count={diagnostics.get('dead_end_successor_count', 0)}, "
        f"filtered_virtual={graph.filtered_virtual_bridge_edge_count}, "
        f"filtered_real={graph.filtered_real_bridge_edge_count}, "
        f"seed_first={diagnostics.get('small_roll_seed_first_enabled', False)}, "
        f"small_seed_orders={diagnostics.get('small_roll_seed_orders', 0)}, "
        f"small_seed_tons10={diagnostics.get('small_roll_seed_tons10', 0)}, "
        f"small_chains={diagnostics.get('small_roll_chain_count', 0)}, "
        f"reserve_candidates_all={diagnostics.get('small_roll_reserve_candidates_all_count', 0)}, "
        f"reserve_quota_count={diagnostics.get('small_roll_reserve_quota_count', 0)}, "
        f"reserve_quota_tons10={diagnostics.get('small_roll_reserve_quota_tons10', 0)}, "
        f"reserve_bucket_size={diagnostics.get('small_roll_reserve_bucket_size', 0)}, "
        f"reserve_bucket_used={diagnostics.get('small_roll_reserve_bucket_used_count', 0)}, "
        f"reserve_bucket_released={diagnostics.get('small_roll_reserve_bucket_released_to_big_count', 0)}, "
        f"early_release={diagnostics.get('reserve_bucket_released_early_to_big', False)}, "
        f"early_release_count={diagnostics.get('reserve_bucket_released_count', 0)}, "
        f"early_release_tons10={diagnostics.get('reserve_bucket_released_tons10', 0)}, "
        f"seed_target_orders={diagnostics.get('quota_min_reached_by_orders', False)}, "
        f"seed_target_tons={diagnostics.get('quota_min_reached_by_tons', False)}, "
        f"seed_stalled={diagnostics.get('small_stalled_no_successor', False)}, "
        f"release_trigger={diagnostics.get('reserve_release_trigger', 'not_applicable')}, "
        f"big_blocked_by_quota={diagnostics.get('big_roll_blocked_by_small_quota_count', 0)}, "
        f"big_blocked_successor={diagnostics.get('big_roll_blocked_successor_count', 0)}, "
        f"big_released_visible={diagnostics.get('big_roll_released_dual_visible_count', 0)}, "
        f"big_released_successor={diagnostics.get('big_roll_released_successor_visible_count', 0)}, "
        f"big_final_orders={diagnostics.get('big_roll_final_orders_placed', 0)}, "
        f"small_final_orders={diagnostics.get('small_roll_final_orders_placed', 0)}, "
        f"balance_collapse={diagnostics.get('line_balance_collapse_detected', False)}, "
        f"balance_retry={diagnostics.get('line_balance_retry_triggered', False)}"
    )

    # Populate bridge edge filtering diagnostics
    diagnostics["filtered_virtual_bridge_edge_count"] = graph.filtered_virtual_bridge_edge_count
    diagnostics["filtered_real_bridge_edge_count"] = graph.filtered_real_bridge_edge_count
    diagnostics["accepted_direct_edge_count"] = graph.accepted_direct_edge_count
    diagnostics["accepted_real_bridge_edge_count"] = graph.accepted_real_bridge_edge_count
    diagnostics["accepted_virtual_bridge_edge_count"] = graph.accepted_virtual_bridge_edge_count
    diagnostics["accepted_virtual_to_real_edge_count"] = graph.accepted_virtual_to_real_edge_count
    diagnostics["accepted_virtual_to_virtual_edge_count"] = graph.accepted_virtual_to_virtual_edge_count
    diagnostics["accepted_direct_after_hard_filter"] = graph.accepted_direct_edge_count
    diagnostics["accepted_real_bridge_after_hard_filter"] = graph.accepted_real_bridge_edge_count
    diagnostics["accepted_virtual_bridge_after_hard_filter"] = graph.accepted_virtual_bridge_edge_count
    diagnostics["rejected_edges_by_hard_filter_total"] = graph.rejected_edges_by_hard_filter_total
    diagnostics["skipped_by_final_width_rule"] = graph.skipped_by_final_width_rule
    diagnostics["skipped_by_final_thickness_rule"] = graph.skipped_by_final_thickness_rule
    diagnostics["skipped_by_final_temp_rule"] = graph.skipped_by_final_temp_rule
    diagnostics["skipped_by_final_group_rule"] = graph.skipped_by_final_group_rule
    diagnostics["skipped_by_virtual_chain_rule"] = graph.skipped_by_virtual_chain_rule
    diagnostics["skipped_by_reverse_width_budget"] = graph.skipped_by_reverse_width_budget
    diagnostics["accepted_with_reverse_width_budget"] = graph.accepted_with_reverse_width_budget
    diagnostics["reverse_width_count_used_max"] = graph.reverse_width_count_used_max
    diagnostics["seed_scoring_enabled"] = True
    diagnostics["width_desc_seed_bias_enabled"] = True
    diagnostics["continuation_bias_used"] = True
    diagnostics["continuation_bonus_total"] = float(graph.continuation_bonus_total)
    diagnostics["continuation_cross_min_hits"] = int(graph.continuation_cross_min_hits)
    diagnostics["chains_crossed_ton_min_during_constructive"] = int(
        graph.chains_crossed_ton_min_during_constructive
    )
    diagnostics["virtual_rescue_reason_counts_by_line"] = {
        str(line): dict(counts) for line, counts in graph.virtual_rescue_reason_counts_by_line.items()
    }
    diagnostics["successor_business_scoring_enabled"] = True
    diagnostics["width_desc_successor_bias_enabled"] = True
    diagnostics["successor_extendability_bias_enabled"] = True
    diagnostics["successor_dead_end_penalty_enabled"] = True
    diagnostics["successor_group_continuity_bias_enabled"] = True
    diagnostics["accepted_virtual_chain_paths"] = int(
        graph.candidate_graph_diagnostics.get("accepted_virtual_chain_paths", 0) or 0
    )
    diagnostics["virtual_real_edge_count_effective"] = int(
        graph.candidate_graph_diagnostics.get("virtual_real_edge_count_effective", 0) or 0
    )
    diagnostics["accepted_virtual_bridge_family_edge_count"] = graph.accepted_virtual_bridge_family_edge_count
    diagnostics["constructive_edge_policy"] = graph.edge_policy
    # Guarded family edge greedy diagnostics
    diagnostics["greedy_virtual_family_edge_uses"] = getattr(graph, "greedy_virtual_family_edge_uses", 0)
    diagnostics["greedy_virtual_family_edge_rejects"] = getattr(graph, "greedy_virtual_family_edge_rejects", 0)
    diagnostics["greedy_virtual_family_budget_blocked_count"] = getattr(graph, "greedy_virtual_family_budget_blocked_count", 0)
    diagnostics["greedy_virtual_family_edge_used_per_line"] = dict(getattr(graph, "family_edge_used_per_line", {}))
    diagnostics["greedy_virtual_family_edge_used_per_segment"] = dict(getattr(graph, "family_edge_used_per_segment", {}))
    # Future-aware penalty diagnostics (guarded profile only)
    diagnostics["greedy_future_bridgeability_penalty_hits"] = getattr(graph, "greedy_future_bridgeability_penalty_hits", 0)
    diagnostics["greedy_tail_underfill_risk_penalty_hits"] = getattr(graph, "greedy_tail_underfill_risk_penalty_hits", 0)
    diagnostics["greedy_bridge_scarcity_penalty_hits"] = getattr(graph, "greedy_bridge_scarcity_penalty_hits", 0)
    diagnostics.update(graph.candidate_graph_diagnostics)

    # Collect repair_family_edges from candidate_graph_result for ALNS/local rebuild
    _cgr = getattr(graph, "candidate_graph_result", None)
    _repair_family_edges: list = []
    if _cgr is not None:
        _repair_family_edges = list(getattr(_cgr, "repair_family_edges", []) or [])
    diagnostics["repair_family_edges_count"] = len(_repair_family_edges)

    return ConstructiveBuildResult(
        chains_by_line=chains_by_line,
        dropped_seed_orders=dropped_seed_orders,
        diagnostics=diagnostics,
        repair_family_edges=_repair_family_edges,
    )


# Compatibility alias while mainline wording shifts from "template" to
# "transition edge cache". Internal storage and behavior are unchanged.
TransitionEdgeGraph = TemplateEdgeGraph


__all__ = [
    "ConstructiveChain",
    "ConstructiveBuildResult",
    "TemplateEdgeGraph",
    "TransitionEdgeGraph",
    "build_constructive_sequences",
]
