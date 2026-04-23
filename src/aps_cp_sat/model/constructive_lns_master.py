"""
Constructive LNS Master: ALNS-driven optimization for cold rolling planning.

This module implements the FOURTH layer of the Constructive + ALNS path:
- Layer 1: build_constructive_sequences  (greedy chain building)
- Layer 2: cut_sequences_into_campaigns  (campaign segment cutting)
- Layer 3: solve_local_insertion_subproblem  (CP-SAT local repair)
- Layer 4: run_constructive_lns_master  (ALNS main loop — this module)

The ALNS master orchestrates the three underlying layers and adds:
- Destruction: random segment selection + order removal
- Repair: local insertion CP-SAT
- Acceptance: multi-criteria comparison with rollback
- Iteration: configurable rounds with diverse neighborhood selection

Key constraints this path enforces:
- NO _run_global_joint_model
- NO slot-first bucketing
- NO illegal edge penalization (strict template edges only)

Output format aligns with existing validate/result_writer contracts.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Set, Tuple

import pandas as pd

from aps_cp_sat.config import PlannerConfig
from aps_cp_sat.model.campaign_cutter import (
    CampaignCutResult,
    CampaignSegment,
    cut_sequences_into_campaigns,
    _validate_segment_template_pairs,
)
from aps_cp_sat.model.constructive_sequence_builder import (
    ConstructiveBuildResult,
    build_constructive_sequences,
)
from aps_cp_sat.model.graph_unused_pool import collect_graph_unused_candidates
from aps_cp_sat.model.local_inserter_cp_sat import (
    InsertStatus,
    LocalInsertRequest,
    solve_local_insertion_subproblem,
)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class DropReason(Enum):
    """Why an order ended up in the dropped pool."""
    DEAD_ISLAND_ORDER = "DEAD_ISLAND_ORDER"          # Isolated node, no template edge
    TAIL_UNDERFILLED = "TAIL_UNDERFILLED"            # Below campaign ton minimum
    LNS_REPAIR_REJECTED = "LNS_REPAIR_REJECTED"     # CP-SAT rejected during repair
    NO_FEASIBLE_LINE = "NO_FEASIBLE_LINE"            # No compatible production line
    HARD_CONSTRAINT_PROTECTED_DROP = "HARD_CONSTRAINT_PROTECTED_DROP"  # Hard constraint blocked
    CONSTRUCTIVE_REJECTED = "CONSTRUCTIVE_REJECTED"  # Layer 1 rejected
    CAMPAIGN_CUT_UNDERFILLED = "CAMPAIGN_CUT_UNDERFILLED"  # Cut as underfilled tail
    INITIAL_DROP = "INITIAL_DROP"                    # Dropped during initial formation
    FINAL_SEGMENT_TEMPLATE_MISS = "FINAL_SEGMENT_TEMPLATE_MISS"  # Segment had template pair miss at final output
    FINAL_SEGMENT_TEMPLATE_MISS_FRAGMENT = "FINAL_SEGMENT_TEMPLATE_MISS_FRAGMENT"  # Fragment from salvaged segment still has template miss
    FINAL_SEGMENT_TEMPLATE_MISS_TOO_SHORT = "FINAL_SEGMENT_TEMPLATE_MISS_TOO_SHORT"  # Salvaged fragment too short (single order or under tonnage)
    FINAL_SEGMENT_TEMPLATE_MISS_NO_VALID_SPLIT = "FINAL_SEGMENT_TEMPLATE_MISS_NO_VALID_SPLIT"  # No valid split found after max retries
    FINAL_SEGMENT_TEMPLATE_MISS_DIRTY_WINDOW = "FINAL_SEGMENT_TEMPLATE_MISS_DIRTY_WINDOW"  # Minimum dirty window around offending pair


class NeighborhoodType(Enum):
    """
    ALNS neighborhood destruction strategy.

    Current production neighborhoods (verified effective):
    - LOW_FILL_SEGMENT: Low fill % (near min) segments
    - HIGH_DROP_PRESSURE: Segments near many dropped orders
    - TAIL_REBALANCE: Low-fill tail segments eligible for rebalancing

    REMOVED neighborhoods (experimental, no proven gain in production):
    - HIGH_VIRTUAL_USAGE: Removed - no business meaning in current path
    - SMALL_ROLL_RESCUE: Removed - not part of constructive_lns main path
    """
    LOW_FILL_SEGMENT = "LOW_FILL_SEGMENT"            # Low fill % (near min) segments
    HIGH_DROP_PRESSURE = "HIGH_DROP_PRESSURE"         # Segments near many dropped orders
    TAIL_REBALANCE = "TAIL_REBALANCE"                # Low-fill tail segments eligible for rebalancing
    WIDTH_TENSION_HOTSPOT = "WIDTH_TENSION_HOTSPOT"   # Width tension / drop-pressure (family edge candidate)
    GROUP_SWITCH_HOTSPOT = "GROUP_SWITCH_HOTSPOT"      # Group switch pressure (family edge candidate)
    BRIDGE_DEPENDENT_SEGMENT = "BRIDGE_DEPENDENT_SEGMENT"  # Segments with many bridge edges


# ---------------------------------------------------------------------------
# Guarded virtual family ALNS helpers
# ---------------------------------------------------------------------------

def should_enable_virtual_family_for_neighborhood(
    neighborhood: NeighborhoodType,
    cfg: PlannerConfig,
    *,
    dropped_count: int = 0,
    underfill_detected: bool = False,
    drop_pressure_score: float = 0.0,
    group_switch_hotspot: bool = False,
    width_tension_hotspot: bool = False,
    bridge_dependent_segment: bool = False,
) -> tuple[bool, str]:
    """
    Determine if virtual family edges should be enabled for the given ALNS neighborhood.

    Returns (enabled, reason) tuple:
      - enabled: True only when profile enabled AND neighborhood qualifies AND pressure signal present
      - reason: Human-readable reason for diagnostics

    Priority order for family repair (guarded profile):
      PRIMARY (always considered first):
        - LOW_FILL_SEGMENT: most direct impact on fill quality
        - HIGH_DROP_PRESSURE: resolves drop pressure
        - GROUP_SWITCH_HOTSPOT: addresses group transition complexity
      SECONDARY (allowed but may be rate-limited):
        - WIDTH_TENSION_HOTSPOT: width tension resolution
        - BRIDGE_DEPENDENT_SEGMENT: bridge dependency resolution
    """
    model = getattr(cfg, "model", None)
    if model is None:
        return False, "cfg_none"
    if not getattr(model, "virtual_family_frontload_enabled", False):
        return False, "profile_disabled"

    primary_neighborhoods = {
        NeighborhoodType.LOW_FILL_SEGMENT,
        NeighborhoodType.HIGH_DROP_PRESSURE,
        NeighborhoodType.GROUP_SWITCH_HOTSPOT,
    }
    secondary_neighborhoods = {
        NeighborhoodType.WIDTH_TENSION_HOTSPOT,
        NeighborhoodType.BRIDGE_DEPENDENT_SEGMENT,
    }
    family_neighborhoods = primary_neighborhoods | secondary_neighborhoods

    if neighborhood not in family_neighborhoods:
        return False, f"neighborhood={neighborhood.value}_not_family_eligible"

    # Primary neighborhoods: require at least one pressure signal
    if neighborhood in primary_neighborhoods:
        if underfill_detected:
            return True, f"primary={neighborhood.value}_underfill_detected"
        if drop_pressure_score > 0 or dropped_count > 0:
            return True, f"primary={neighborhood.value}_drop_pressure"
        if group_switch_hotspot:
            return True, f"primary={neighborhood.value}_group_switch"
        # Primary neighborhoods can also enable with mild signal
        if drop_pressure_score >= 0:
            return True, f"primary={neighborhood.value}_mild_signal"
        return False, f"primary={neighborhood.value}_no_signal"

    # Secondary neighborhoods: require stronger signal
    if neighborhood in secondary_neighborhoods:
        if bridge_dependent_segment and underfill_detected:
            return True, f"secondary={neighborhood.value}_bridge_underfill"
        if width_tension_hotspot and dropped_count > 0:
            return True, f"secondary={neighborhood.value}_width_drop"
        return False, f"secondary={neighborhood.value}_insufficient_signal"


def should_run_local_cpsat(
    neighborhood: NeighborhoodType,
    cfg: PlannerConfig,
    candidate_count: int,
    *,
    round_num: int = 0,
    max_cpsat_rounds: int = 8,
) -> Tuple[bool, str]:
    """
    Unified gate for deciding whether to run expensive local CP-SAT subproblem.

    Returns:
        Tuple of (should_run: bool, reason: str)

    Gate conditions (skip if any matches):
        1. neighborhood not in ALNS-eligible set
        2. candidate count exceeds local_cpsat_max_orders
        3. already past max_cpsat_rounds (local CP-SAT not worthwhile after mid-run)
        4. neighborhood is TAIL_REBALANCE with no dropped orders to reinsert
        5. candidate count too small (< 2 orders)

    Gate conditions (allow if all pass):
        1. neighborhood in {LOW_FILL_SEGMENT, HIGH_DROP_PRESSURE,
                           GROUP_SWITCH_HOTSPOT, WIDTH_TENSION_HOTSPOT,
                           BRIDGE_DEPENDENT_SEGMENT}
        2. candidate_count <= local_cpsat_max_orders
        3. round_num < max_cpsat_rounds
        4. candidate_count >= 2
    """
    # ---- Condition 1: neighborhood eligibility ----
    cpsat_eligible_neighborhoods = {
        NeighborhoodType.LOW_FILL_SEGMENT,
        NeighborhoodType.HIGH_DROP_PRESSURE,
        NeighborhoodType.TAIL_REBALANCE,
        NeighborhoodType.GROUP_SWITCH_HOTSPOT,
        NeighborhoodType.WIDTH_TENSION_HOTSPOT,
        NeighborhoodType.BRIDGE_DEPENDENT_SEGMENT,
    }
    if neighborhood not in cpsat_eligible_neighborhoods:
        return False, f"NEIGHBORHOOD_NOT_ELIGIBLE({neighborhood.value})"

    # ---- Condition 2: candidate scale check ----
    max_orders = int(getattr(cfg.model, "local_cpsat_max_orders", 45) if cfg.model else 45)
    if candidate_count > max_orders:
        return False, f"CANDIDATE_COUNT_EXCEEDED({candidate_count}>{max_orders})"

    # ---- Condition 3: round budget check ----
    if round_num >= max_cpsat_rounds:
        return False, f"ROUND_EXCEEDED({round_num}>={max_cpsat_rounds})"

    # ---- Condition 4: minimum candidate check ----
    if candidate_count < 2:
        return False, f"TOO_FEW_CANDIDATES({candidate_count})"

    # ---- All checks passed ----
    return True, "GATE_PASSED"


# ---------------------------------------------------------------------------
# Final segment salvage helpers
# ---------------------------------------------------------------------------

def _salvage_segment_with_template_miss(
    seg: CampaignSegment,
    miss_examples: List[dict],
    tpl_df: pd.DataFrame,
    ton_min: float,
    ton_max: float,
    max_expand_steps: int = 3,
) -> Tuple[List[CampaignSegment], List[dict], dict]:
    """
    Fine-grained salvage: minimize dirty fragment, maximize clean piece retention.

    Strategy:
    1. Find first offending pair (a, b) at position i
    2. Center dirty window around (i, i+1), expand minimally to find valid clean sides
    3. Keep left/right clean subsegments if valid (pairs OK + tons OK)
    4. Only demote the minimal dirty window + any too-short fragments

    Args:
        seg: The CampaignSegment with template miss
        miss_examples: List of miss examples from _validate_segment_template_pairs
        tpl_df: Template DataFrame with from_order_id, to_order_id columns
        ton_min: Minimum tonnage for a valid segment
        ton_max: Maximum tonnage for a valid segment
        max_expand_steps: Maximum window expansion steps (default 3)

    Returns:
        Tuple of:
            - salvaged_clean_segments: List of CampaignSegment that passed validation
            - demoted_fragments: List of dicts with fragment info and drop reason
            - salvage_stats: Dict with statistics (dirty_window_size, etc.)
    """
    salvaged: List[CampaignSegment] = []
    demoted: List[dict] = []
    salvage_stats: dict = {
        "dirty_window_size": 0,
        "kept_left_orders": 0,
        "kept_right_orders": 0,
        "offending_pair_a": None,
        "offending_pair_b": None,
        "expand_steps_used": 0,
    }

    if not miss_examples:
        # No miss: segment is clean, return as-is
        salvaged.append(seg)
        return salvaged, demoted, salvage_stats

    # Build template keyset for fast lookup
    tpl_keys: set = set()
    for _, trow in tpl_df.iterrows():
        from_oid = str(trow.get("from_order_id", ""))
        to_oid = str(trow.get("to_order_id", ""))
        tpl_keys.add((from_oid, to_oid))

    order_ids = list(seg.order_ids)
    n = len(order_ids)

    def _validate_piece(piece_oids: List[str]) -> Tuple[bool, str]:
        """Check if a piece has all internal pairs legal (tonnage checked separately)."""
        if not piece_oids:
            return False, "EMPTY_PIECE"
        if len(piece_oids) == 1:
            # Single order: no adjacent pairs to check → trivially clean
            return True, ""
        # Check all internal adjacent pairs
        for i in range(len(piece_oids) - 1):
            oid_a = piece_oids[i]
            oid_b = piece_oids[i + 1]
            if (oid_a, oid_b) not in tpl_keys:
                return False, f"TEMPLATE_MISS_at_{i+1}_{i+2}"
        return True, ""

    def _check_tons(piece_oids: List[str], seg_total_tons: float) -> Tuple[bool, float, str]:
        """Check if piece tonnage is within [ton_min, ton_max]."""
        if not piece_oids or n == 0:
            return False, 0.0, "EMPTY_PIECE"
        ratio = len(piece_oids) / n
        piece_tons = seg_total_tons * ratio
        if len(piece_oids) == 1:
            # Single order: tonnage can be anything, but we still compute it
            return True, piece_tons, ""
        if piece_tons < ton_min:
            return False, piece_tons, f"below_min_{piece_tons:.1f}<{ton_min}"
        if piece_tons > ton_max:
            return False, piece_tons, f"above_max_{piece_tons:.1f}>{ton_max}"
        return True, piece_tons, ""

    def _is_single_order_too_short(piece_oids: List[str], seg_total_tons: float) -> Tuple[bool, float]:
        """Check if single order is too short to keep standalone."""
        if len(piece_oids) != 1:
            return False, 0.0
        ratio = 1.0 / n
        piece_tons = seg_total_tons * ratio
        # Single order kept only if it has valid pairs extension potential
        # For now, keep single orders if ton_min is very small, else demote
        return piece_tons < ton_min, piece_tons

    def _find_offending_idx(piece_oids: List[str]) -> int:
        """Find first offending pair index in piece."""
        for i in range(len(piece_oids) - 1):
            oid_a = piece_oids[i]
            oid_b = piece_oids[i + 1]
            if (oid_a, oid_b) not in tpl_keys:
                return i
        return -1

    # ---- Step 1: Find first offending pair in original segment ----
    first_miss = miss_examples[0]
    first_offending_idx = first_miss.get("seq_a", 1) - 1  # Convert to 0-based
    if first_offending_idx < 0 or first_offending_idx >= n - 1:
        first_offending_idx = _find_offending_idx(order_ids)
    if first_offending_idx < 0:
        # Fallback: scan for any miss
        for i in range(n - 1):
            oid_a = order_ids[i]
            oid_b = order_ids[i + 1]
            if (oid_a, oid_b) not in tpl_keys:
                first_offending_idx = i
                break

    if first_offending_idx < 0:
        # No miss found: treat as clean
        salvaged.append(seg)
        return salvaged, demoted, salvage_stats

    salvage_stats["offending_pair_a"] = order_ids[first_offending_idx]
    salvage_stats["offending_pair_b"] = order_ids[first_offending_idx + 1]

    # ---- Step 2: Fine-grained salvage with minimal dirty window ----
    # Strategy: center dirty window on (first_offending_idx, first_offending_idx+1)
    # Expand window minimally to find valid clean sides

    seg_total_tons = seg.total_tons
    kept_left: List[str] = []
    kept_right: List[str] = []
    dirty_start = first_offending_idx
    dirty_end = first_offending_idx + 1  # Inclusive end of dirty window

    # Try progressively larger dirty windows
    for expand_step in range(max_expand_steps + 1):
        salvage_stats["expand_steps_used"] = expand_step

        # Left clean region: order_ids[:dirty_start]
        left_candidates = order_ids[:dirty_start] if dirty_start > 0 else []

        # Right clean region: order_ids[dirty_end + 1:]
        right_candidates = order_ids[dirty_end + 1:] if dirty_end + 1 < n else []

        # Dirty window: order_ids[dirty_start:dirty_end + 1]
        dirty_window = order_ids[dirty_start:dirty_end + 1]

        # Check left side
        left_valid, left_reason = _validate_piece(left_candidates)
        left_tons_ok, left_tons, left_tons_reason = _check_tons(left_candidates, seg_total_tons)

        if left_candidates and left_valid and left_tons_ok:
            kept_left = left_candidates
            salvage_stats["kept_left_orders"] = len(kept_left)

        # Check right side
        right_valid, right_reason = _validate_piece(right_candidates)
        right_tons_ok, right_tons, right_tons_reason = _check_tons(right_candidates, seg_total_tons)

        if right_candidates and right_valid and right_tons_ok:
            kept_right = right_candidates
            salvage_stats["kept_right_orders"] = len(kept_right)

        # Check if we have at least one valid clean side
        # If yes, we can demote the dirty window and keep what we can
        if (kept_left or kept_right) or expand_step >= max_expand_steps:
            salvage_stats["dirty_window_size"] = len(dirty_window)
            break

        # Expand dirty window by 1 order on each side
        # But check if expansion would include another miss or break valid sides
        can_expand_left = dirty_start > 0
        can_expand_right = dirty_end < n - 1

        # Prefer expanding towards the side with more potential clean orders
        left_potential = dirty_start
        right_potential = n - dirty_end - 1

        if can_expand_left and left_potential >= right_potential:
            dirty_start -= 1
        elif can_expand_right:
            dirty_end += 1
        elif can_expand_left:
            dirty_start -= 1
        # else: stuck, will exit loop next iteration

    # ---- Step 3: Finalize kept pieces ----
    # Build kept_left segment
    if kept_left:
        tons_ok, tons, _ = _check_tons(kept_left, seg_total_tons)
        if tons_ok:
            new_seg = CampaignSegment(
                line=seg.line,
                campaign_local_id=seg.campaign_local_id + 1000,
                order_ids=list(kept_left),
                total_tons=tons,
                cut_reason=seg.cut_reason,
                start_order_id=kept_left[0],
                end_order_id=kept_left[-1],
                edge_count=len(kept_left) - 1,
                is_valid=True,
            )
            salvaged.append(new_seg)
        else:
            # Left side had tonnage issue
            demoted.append({
                "piece_oids": kept_left,
                "reason": DropReason.FINAL_SEGMENT_TEMPLATE_MISS_TOO_SHORT.value,
                "detail": f"left_clean_tons={tons:.1f}_{left_tons_reason}",
                "piece_type": "kept_left",
            })

    # Build kept_right segment
    if kept_right:
        tons_ok, tons, _ = _check_tons(kept_right, seg_total_tons)
        if tons_ok:
            new_seg = CampaignSegment(
                line=seg.line,
                campaign_local_id=seg.campaign_local_id + 2000,
                order_ids=list(kept_right),
                total_tons=tons,
                cut_reason=seg.cut_reason,
                start_order_id=kept_right[0],
                end_order_id=kept_right[-1],
                edge_count=len(kept_right) - 1,
                is_valid=True,
            )
            salvaged.append(new_seg)
        else:
            # Right side had tonnage issue
            demoted.append({
                "piece_oids": kept_right,
                "reason": DropReason.FINAL_SEGMENT_TEMPLATE_MISS_TOO_SHORT.value,
                "detail": f"right_clean_tons={tons:.1f}_{right_tons_reason}",
                "piece_type": "kept_right",
            })

    # ---- Step 4: Demote dirty window ----
    if dirty_window:
        demoted.append({
            "piece_oids": dirty_window,
            "reason": DropReason.FINAL_SEGMENT_TEMPLATE_MISS_DIRTY_WINDOW.value,
            "detail": f"dirty_window_orders={len(dirty_window)}_[{dirty_window[0]}..{dirty_window[-1]}]",
            "piece_type": "dirty_window",
            "offending_pair": f"{salvage_stats['offending_pair_a']}->{salvage_stats['offending_pair_b']}",
        })

    # ---- Step 5: Handle remaining orders outside kept_left/kept_right/dirty_window ----
    # These are orders that couldn't be kept due to expansion
    all_demoted_oids = set()
    for frag in demoted:
        all_demoted_oids.update(frag.get("piece_oids", []))

    kept_oids = set(kept_left) | set(kept_right)
    remaining = [oid for oid in order_ids if oid not in all_demoted_oids and oid not in kept_oids]

    for frag in demoted:
        remaining = [oid for oid in remaining if oid not in frag.get("piece_oids", [])]

    # Remaining orders: demote as fragment
    if remaining:
        demoted.append({
            "piece_oids": remaining,
            "reason": DropReason.FINAL_SEGMENT_TEMPLATE_MISS_FRAGMENT.value,
            "detail": f"remaining_orders={len(remaining)}",
            "piece_type": "remaining",
        })

    return salvaged, demoted, salvage_stats


class LnsStatus(Enum):
    """Overall ALNS solve status."""
    OPTIMAL = "OPTIMAL"          # Strictly improved over constructive baseline
    FEASIBLE = "FEASIBLE"        # Found feasible solution
    NO_IMPROVEMENT = "NO_IMPROVEMENT"  # Converged without improvement
    INFEASIBLE = "INFEASIBLE"     # Fundamental infeasibility detected


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class LnsRound:
    """Record of one ALNS iteration."""
    round: int
    neighborhood_type: NeighborhoodType
    accepted: bool
    scheduled_order_count: int
    scheduled_tons: float
    dropped_count: int
    hard_constraint_protected_drop_count: int = 0
    lns_repair_rejected_count: int = 0
    repair_segment_candidate_count: int = 0
    repair_pool_candidate_count: int = 0
    # Pool candidate filtering diagnostics for this round
    repair_pool_filtered_by_capability_count: int = 0
    repair_pool_filtered_by_connectivity_count: int = 0
    repair_pool_skipped_as_already_tried_count: int = 0
    virtual_bridge_count: int = 0
    template_cost: float = 0.0
    destroy_count: int = 0
    repair_count: int = 0
    elapsed_seconds: float = 0.0
    notes: str = ""
    # TAIL_REBALANCE neighborhood diagnostics
    target_fill_ratio: float = 0.0      # fill_ratio of the selected segment (closer to 1.0 = better)
    target_tail_tons: float = 0.0       # tons of the selected tail segment
    target_prev_segment_tons: float = 0.0  # tons of the preceding segment (available for pullback)
    # Tail repair LNS diagnostics
    tail_rebalance_lns_success_count: int = 0  # number of underfilled tails repaired in this round
    tail_rebalance_lns_shifted_orders: int = 0  # orders shifted from donor segment to tail
    tail_rebalance_lns_inserted_dropped_orders: int = 0  # dropped orders inserted into low-fill tail
    tail_rebalance_lns_merge_count: int = 0  # successful merge of tail into donor segment
    tail_rebalance_lns_recut_count: int = 0  # successful recut of two segments
    tail_rebalance_lns_fail_reasons: List[str] = field(default_factory=list)
    tail_rebalance_lns_dropped_reinsertion_attempts: int = 0  # attempts to insert dropped into underfilled tail
    tail_rebalance_lns_dropped_reinsertion_success: int = 0  # successful dropped reinsertion
    # LOW_FILL_SEGMENT neighborhood diagnostics
    low_fill_candidates: int = 0   # number of low-fill segment candidates considered
    low_fill_avg_gap_to_min: float = 0.0  # average tons below min across candidates
    low_fill_selected_gap: float = 0.0   # gap_to_min of the selected segment
    # Neighborhood selection counts across all rounds
    low_fill_neighborhood_selected_count: int = 0  # how many rounds used LOW_FILL_SEGMENT
    tail_rebalance_neighborhood_selected_count: int = 0  # how many rounds used TAIL_REBALANCE
    low_fill_neighborhood_success_count: int = 0  # accepted rounds for LOW_FILL_SEGMENT
    tail_rebalance_neighborhood_success_count: int = 0  # accepted rounds for TAIL_REBALANCE
    # NOTE: SMALL_ROLL_RESCUE and HIGH_VIRTUAL_USAGE removed (experimental, not in production)
    # ---- Guarded virtual family ALNS diagnostics ----
    virtual_family_frontload_enabled: bool = False  # Profile-level total switch
    virtual_family_enabled_for_this_round: bool = False  # Round-level actual enabled result
    virtual_family_enable_reason: str = ""  # Why family was enabled/disabled this round
    virtual_family_disable_reason: str = ""  # Why family was disabled this round (if applicable)
    alns_virtual_family_attempt_count: int = 0  # Rounds where family edge was attempted
    alns_virtual_family_accept_count: int = 0  # Rounds where family edge was accepted
    alns_virtual_family_reject_count: int = 0  # Rounds where family edge was rejected
    alns_virtual_family_budget_block_count: int = 0  # Rounds blocked by budget limit
    alns_virtual_family_no_gain_count: int = 0  # Rounds with no improvement from family edge
    alns_virtual_family_neighborhood_histogram: dict = field(default_factory=dict)  # Count by neighborhood type
    local_cpsat_virtual_family_selected_count: int = 0  # Family edges selected in local CP-SAT this round
    # ---- Family repair value metrics (guarded profile) ----
    # Count-based
    family_repair_attempt_count: int = 0  # Family repair subproblem attempts
    family_repair_accept_count: int = 0   # Family repair subproblem accepted
    family_repair_gain_dropped: int = 0   # Dropped orders resolved by family repair
    family_repair_gain_underfilled: int = 0  # Underfilled segments improved by family repair
    family_repair_gain_scheduled_orders: int = 0  # Orders scheduled via family repair
    family_repair_no_gain_count: int = 0   # Family repair attempts with no improvement
    # Average value metrics
    family_repair_avg_gain_per_attempt: float = 0.0  # avg(dropped + underfilled) per attempt
    family_repair_avg_seconds_per_success: float = 0.0  # avg seconds per successful attempt


@dataclass
class _SolutionSnapshot:
    """
    Internal snapshot of the current ALNS solution state.
    Immutable-ish copy used for rollback comparison.
    """
    planned_order_ids: Set[str]
    planned_by_line: Dict[str, List[str]]      # line -> ordered list
    dropped_by_reason: Dict[str, List[str]]    # reason -> order_ids
    total_tons: float
    virtual_bridge_count: int
    template_cost: float

    def order_count(self) -> int:
        return len(self.planned_order_ids)

    def _score_tuple(self) -> Tuple:
        """Five-level objective score (lower = better)."""
        return (
            0,                                              # always feasible (hard constraint)
            -self.order_count(),                            # more orders better
            -self.total_tons,                              # more tons better
            self.virtual_bridge_count,                     # fewer vb better
            self.template_cost,                            # lower cost better
        )


@dataclass
class ConstructiveLnsResult:
    """
    Final output of the Constructive + ALNS solver.

    Attributes:
        status: Overall solve status
        planned_df: Scheduled orders (aligns with existing validate schema)
        dropped_df: Unscheduled orders with drop reasons
        rounds_df: Per-round statistics
        engine_meta: Solver metadata (seed, rounds, time, etc.)
        diagnostics: Detailed solve diagnostics
    """
    status: LnsStatus
    planned_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    dropped_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    rounds_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    engine_meta: Dict = field(default_factory=dict)
    diagnostics: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helper: Build planned_df from segments
# ---------------------------------------------------------------------------

def _normalize_bridge_metadata(
    edge_type: str,
    bridge_path: str,
    tpl_row: dict | None,
    cfg: PlannerConfig | None,
) -> dict:
    """
    Normalize bridge metadata for a single edge.

    This function provides unified semantics for bridge metadata fields,
    ensuring consistent output format across constructive_lns path.

    Returns dict with keys:
        - selected_edge_type: str  ("DIRECT_EDGE", "REAL_BRIDGE_EDGE", "VIRTUAL_BRIDGE_EDGE")
        - selected_bridge_path: str  (bridge path string or "")
        - selected_bridge_expandable: bool  (True if edge type supports expansion)
        - selected_bridge_expand_mode: str  ("disabled", "virtual_expand", etc.)
        - selected_virtual_bridge_count: int  (number of virtual coils if expanded)
        - selected_real_bridge_order_id: str  (order_id of real bridge if applicable)

    Route RB (mainline, allow_virtual=False, allow_real=True):
        - DIRECT_EDGE: all expand fields = False/0/""
        - REAL_BRIDGE_EDGE: legitimate mainline edge, capture real_bridge_order_id
        - VIRTUAL_BRIDGE_EDGE: blocked (leak in mainline)

    Route C (baseline, allow_virtual=False, allow_real=False):
        - DIRECT_EDGE: all expand fields = False/0/""
        - REAL_BRIDGE_EDGE: blocked (leak in baseline)
        - VIRTUAL_BRIDGE_EDGE: blocked (leak in baseline)
    """
    # Get bridge expansion mode from config
    bridge_expand_mode = str(getattr(cfg.model, "bridge_expansion_mode", "disabled") if cfg else "disabled")
    allow_virtual = getattr(cfg.model, "allow_virtual_bridge_edge_in_constructive", False) if cfg else False
    allow_real = getattr(cfg.model, "allow_real_bridge_edge_in_constructive", False) if cfg else False

    # Route C (direct_only): both are False
    is_direct_only = not allow_virtual and not allow_real

    # Default values
    result = {
        "selected_edge_type": edge_type,
        "selected_bridge_path": bridge_path or "",
        "selected_bridge_expandable": False,
        "selected_bridge_expand_mode": bridge_expand_mode,
        "selected_virtual_bridge_count": 0,
        "selected_real_bridge_order_id": "",
    }

    if edge_type == "DIRECT_EDGE":
        result["selected_bridge_expandable"] = False
        result["selected_bridge_expand_mode"] = "disabled"
        result["selected_bridge_path"] = ""

    elif edge_type == "REAL_BRIDGE_EDGE":
        # Route RB: REAL_BRIDGE_EDGE is legitimate in mainline
        # Route C: REAL_BRIDGE_EDGE is blocked (leak in baseline)
        result["selected_bridge_expandable"] = False
        result["selected_bridge_expand_mode"] = "disabled"
        result["selected_virtual_bridge_count"] = 0
        if tpl_row:
            result["selected_real_bridge_order_id"] = str(
                tpl_row.get("bridge_order_id", tpl_row.get("real_bridge_order_id", ""))
            )
        if is_direct_only:
            # Mark as leak in baseline mode: force bridge_path to empty
            result["selected_bridge_path"] = ""

    elif edge_type == "VIRTUAL_BRIDGE_EDGE":
        # Route RB / Route C: VIRTUAL_BRIDGE_EDGE is blocked (leak in both modes)
        if is_direct_only:
            result["selected_bridge_expandable"] = False
            result["selected_bridge_expand_mode"] = "disabled"
            result["selected_virtual_bridge_count"] = 0
        elif bridge_expand_mode == "disabled":
            result["selected_bridge_expandable"] = True  # Could expand in future Route A
            result["selected_bridge_expand_mode"] = "disabled"
            result["selected_virtual_bridge_count"] = 0  # Not expanded yet
        elif bridge_expand_mode == "virtual_expand":
            result["selected_bridge_expandable"] = True
            result["selected_bridge_expand_mode"] = "virtual_expand"
            if tpl_row:
                result["selected_virtual_bridge_count"] = int(
                    tpl_row.get("bridge_count", tpl_row.get("virtual_bridge_count", 0) or 0)
                )
            else:
                result["selected_virtual_bridge_count"] = 0

    return result


def _segments_to_planned_df(
    segments: List[CampaignSegment],
    orders_df: pd.DataFrame,
    tpl_df: pd.DataFrame,
    rounds_offset: int = 0,
    cfg: Optional[PlannerConfig] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Convert a list of CampaignSegments into planned_df format.

    Produces columns: order_id, line, master_slot, master_seq,
    campaign_id_hint, campaign_seq_hint, selected_template_id,
    selected_bridge_path, selected_edge_type, selected_bridge_expandable,
    selected_bridge_expand_mode, selected_virtual_bridge_count,
    selected_real_bridge_order_id, force_break_before,
    plus tons/width/thickness.

    master_slot is assigned per-segment: all orders within the same
    CampaignSegment share the same master_slot (= campaign_local_id).
    master_seq is the 1-based position within the segment.

    Bridge edge handling (Route B):
    - If allow_virtual_bridge_edge_in_constructive is False, VIRTUAL_BRIDGE_EDGE
      edges will not appear in planned_df (they are already filtered upstream)
    - selected_edge_type is set based on the actual edge type used
    - All bridge metadata fields are normalized via _normalize_bridge_metadata()

    Returns:
        (planned_df, diagnostics_dict) — diagnostics includes
        slot_semantic_check_ok and any violations found.
    """
    # Get bridge edge policy from config
    allow_virtual = getattr(cfg.model, "allow_virtual_bridge_edge_in_constructive", False) if cfg else False
    allow_real = getattr(cfg.model, "allow_real_bridge_edge_in_constructive", True) if cfg else True
    rows: List[Dict] = []
    order_lookup: Dict = {}
    if not orders_df.empty:
        order_lookup = orders_df.set_index("order_id").to_dict("index")

    # Build template lookup: (from, to) -> row dict
    tpl_lookup: Dict[Tuple[str, str], dict] = {}
    if not tpl_df.empty:
        for _, trow in tpl_df.iterrows():
            key = (str(trow.get("from_order_id", "")), str(trow.get("to_order_id", "")))
            tpl_lookup[key] = dict(trow)

    seen_order_ids: Set[str] = set()
    slot_semantic_violations: List[str] = []

    for seg in segments:
        if not seg.is_valid:
            continue

        # master_slot = campaign_local_id: same for all orders in this segment
        slot_no = int(seg.campaign_local_id) if seg.campaign_local_id is not None else 0

        for seq_idx, oid in enumerate(seg.order_ids):
            # ---- Duplicate detection ----
            if oid in seen_order_ids:
                slot_semantic_violations.append(
                    f"DUPLICATE_ORDER_IN_PLANNED: order_id={oid} appears in "
                    f"multiple segments (first seen earlier, "
                    f"line={seg.line}, master_slot={slot_no})"
                )
            seen_order_ids.add(oid)

            rec = order_lookup.get(oid, {})
            # Find incoming edge template (if not first in segment)
            tpl_id: Optional[str] = None
            bridge_path: Optional[str] = None
            selected_edge_type: Optional[str] = None
            tpl_row: dict | None = None
            if seq_idx > 0:
                prev_oid = seg.order_ids[seq_idx - 1]
                tkey = (prev_oid, oid)
                if tkey in tpl_lookup:
                    tpl = tpl_lookup[tkey]
                    tpl_row = tpl
                    tpl_id = str(tpl.get("template_id", tpl.get("selected_template_id", "")))
                    edge_type = str(tpl.get("edge_type", "") or "DIRECT_EDGE")
                    selected_edge_type = edge_type

                    # Bridge path handling: only set for REAL_BRIDGE_EDGE (virtual is disabled)
                    if edge_type == "REAL_BRIDGE_EDGE":
                        bridge_path = f"{prev_oid}|{oid}"
                    elif edge_type == "DIRECT_EDGE":
                        bridge_path = ""  # No bridge path for direct edges
                    # VIRTUAL_BRIDGE_EDGE: should not reach here due to upstream filtering

            # If edge type was not determined (first in segment), default to DIRECT_EDGE
            if selected_edge_type is None:
                selected_edge_type = "DIRECT_EDGE"
            if bridge_path is None:
                bridge_path = ""

            # Normalize all bridge metadata fields via helper
            bridge_meta = _normalize_bridge_metadata(
                edge_type=selected_edge_type,
                bridge_path=bridge_path,
                tpl_row=tpl_row,
                cfg=cfg,
            )

            # Determine force_break_before: break if next order would exceed max
            force_break = False
            rows.append({
                "order_id": oid,
                "line": seg.line,
                "master_slot": slot_no,           # segment-level, NOT row-level cumcount
                "master_seq": seq_idx + 1,          # 1-based position within segment
                "campaign_id_hint": f"{seg.line}_{seg.campaign_local_id}",
                "campaign_seq_hint": seq_idx + 1,
                "selected_template_id": tpl_id,
                "selected_bridge_path": bridge_meta["selected_bridge_path"],
                "selected_edge_type": bridge_meta["selected_edge_type"],
                "selected_bridge_expandable": bridge_meta["selected_bridge_expandable"],
                "selected_bridge_expand_mode": bridge_meta["selected_bridge_expand_mode"],
                "selected_virtual_bridge_count": bridge_meta["selected_virtual_bridge_count"],
                "selected_real_bridge_order_id": bridge_meta["selected_real_bridge_order_id"],
                "force_break_before": force_break,
                "tons": rec.get("tons", 0.0),
                "width": rec.get("width", 0.0),
                "thickness": rec.get("thickness", 0.0),
                "steel_group": rec.get("steel_group", ""),
                "due_date": rec.get("due_date", ""),
            })

    df = pd.DataFrame(rows)

    # ---- Slot semantic self-check ----
    slot_check_ok = True
    slot_check_warnings: List[str] = []

    if not df.empty:
        # Check 1: each (line, master_slot) should have exactly one campaign_id_hint
        grp_key = df.groupby(["line", "master_slot"])["campaign_id_hint"].nunique()
        multi_camp = grp_key[grp_key > 1]
        if not multi_camp.empty:
            for (line_val, slot_val), n in multi_camp.items():
                slot_check_warnings.append(
                    f"MULTIPLE_CAMPAIGN_ID_FOR_SLOT: line={line_val}, master_slot={slot_val}, "
                    f"campaign_id_hint_count={n}"
                )
            slot_check_ok = False

        # Check 2: within each (line, master_slot), master_seq must be 1..N continuous
        for (line_val, slot_val), grp in df.groupby(["line", "master_slot"], dropna=False):
            grp_sorted = grp.sort_values("master_seq")
            seqs = grp_sorted["master_seq"].tolist()
            expected = list(range(1, len(seqs) + 1))
            if seqs != expected:
                slot_check_warnings.append(
                    f"NON_CONTINUOUS_master_seq: line={line_val}, master_slot={slot_val}, "
                    f"actual_seq={seqs}, expected={expected}"
                )
                slot_check_ok = False

    # Fail fast on duplicate orders
    if slot_semantic_violations:
        violation_msg = (
            "[constructive_lns_master] 计划 DataFrame 中存在同一订单被写入多个 slot！\n"
            + "\n".join(slot_semantic_violations)
        )
        raise RuntimeError(violation_msg)

    # ---- Order Integrity Self-Check ----
    # Verify that within each (line, master_slot) group, the three sequence
    # sources (input write order, master_seq sort, campaign_seq_hint sort) are
    # all consistent. This catches any accidental reordering during construction.
    order_integrity_ok = True
    order_integrity_violations: List[dict] = []

    if not df.empty:
        for (line_val, slot_val), grp in df.groupby(["line", "master_slot"], dropna=False):
            # 1. Input write order (df row order at this point = construction order)
            order_ids_by_input = grp["order_id"].tolist()

            # 2. master_seq sort order
            order_ids_by_master_seq = (
                grp.sort_values("master_seq")["order_id"].tolist()
            )

            # 3. campaign_seq_hint sort order
            order_ids_by_campaign_seq_hint = (
                grp.sort_values("campaign_seq_hint")["order_id"].tolist()
            )

            # master_seq and campaign_seq_hint should be identical
            seq_mismatch = order_ids_by_master_seq != order_ids_by_campaign_seq_hint

            # Input order must match master_seq order (segments iterate seg.order_ids)
            input_mismatch = order_ids_by_input != order_ids_by_master_seq

            if input_mismatch or seq_mismatch:
                order_integrity_ok = False
                order_integrity_violations.append({
                    "line": line_val,
                    "master_slot": slot_val,
                    "order_ids_by_input": order_ids_by_input,
                    "order_ids_by_master_seq": order_ids_by_master_seq,
                    "order_ids_by_campaign_seq_hint": order_ids_by_campaign_seq_hint,
                    "input_mismatch": input_mismatch,
                    "seq_mismatch": seq_mismatch,
                })

        if not order_integrity_ok:
            violation_msgs = []
            for v in order_integrity_violations:
                msgs = [
                    f"[APS][OrderIntegrity] line={v['line']}, master_slot={v['master_slot']}",
                    f"  input_order={v['order_ids_by_input']}",
                    f"  master_seq_order={v['order_ids_by_master_seq']}",
                    f"  campaign_seq_hint_order={v['order_ids_by_campaign_seq_hint']}",
                    f"  input_mismatch={v['input_mismatch']}, seq_mismatch={v['seq_mismatch']}",
                ]
                violation_msgs.append("\n".join(msgs))
            print("[APS][OrderIntegrity] FAILED — segment internal order is inconsistent!")
            for msg in violation_msgs:
                print(msg)
            # Fail fast: inconsistent ordering within a segment is a serious bug
            raise RuntimeError(
                "[constructive_lns_master] planned_df order integrity check FAILED.\n"
                "Within a (line, master_slot) group, input write order, master_seq sort, "
                "and campaign_seq_hint sort must all be identical.\n"
                + "\n\n".join(violation_msgs)
            )

        print(
            f"[APS][OrderIntegrity] OK, checked {df.groupby(['line','master_slot']).ngroups} slots"
        )

    # ---- Debug Diagnostics Sheet ----
    # Small table for quick manual verification of ordering consistency
    debug_diag_rows: List[Dict] = []
    if not df.empty:
        for (line_val, slot_val), grp in df.groupby(["line", "master_slot"], dropna=False):
            for input_pos, (_, row) in enumerate(grp.iterrows(), start=1):
                debug_diag_rows.append({
                    "line": row["line"],
                    "campaign_id_hint": row["campaign_id_hint"],
                    "master_slot": row["master_slot"],
                    "order_id": row["order_id"],
                    "input_position": input_pos,
                    "master_seq": row["master_seq"],
                    "campaign_seq_hint": row["campaign_seq_hint"],
                })
    debug_diag_df = pd.DataFrame(debug_diag_rows)

    diagnostics = {
        "slot_semantic_check_ok": slot_check_ok,
        "slot_semantic_violations": slot_check_warnings,
        "total_segments": sum(1 for s in segments if s.is_valid),
        "total_orders": len(seen_order_ids),
        "planned_df_order_integrity_ok": order_integrity_ok,
        "planned_df_order_integrity_violations": order_integrity_violations,
        "planned_df_debug_diag_df": debug_diag_df,
    }

    return df, diagnostics


# ---------------------------------------------------------------------------
# Helper: Infer single-line hint from line_capability value
# ---------------------------------------------------------------------------

def _infer_line_hint_from_capability(raw_cap: object) -> str:
    """
    Infer a single-line hint string from a raw line_capability value.

    Rules:
        - big_only / big / big_roll / large / large_only -> "big_roll"
        - small_only / small / small_roll                -> "small_roll"
        - dual / both / either / all / empty / NaN / unknown -> ""
          (dual-capability orders keep an empty hint — not constrained)
    """
    if raw_cap is None:
        return ""
    cap_str = str(raw_cap).strip().lower()
    if not cap_str or cap_str == "nan":
        return ""
    # big-family
    big_aliases = {"big_only", "big", "big_roll", "large", "large_only"}
    if cap_str in big_aliases:
        return "big_roll"
    # small-family
    small_aliases = {"small_only", "small", "small_roll"}
    if cap_str in small_aliases:
        return "small_roll"
    # dual / multi-line / unknown -> empty
    return ""


# ---------------------------------------------------------------------------
# Helper: dropped DataFrame builder
# ---------------------------------------------------------------------------

def _dropped_to_df(
    dropped_by_reason: Dict[str, List[str]],
    orders_df: pd.DataFrame,
    stage: Optional[str] = None,
    round_hint: Optional[int] = None,
    line_hint_map: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Convert dropped orders dict to dropped_df format.

    Args:
        dropped_by_reason: reason -> [order_ids]
        orders_df: Full orders DataFrame (used for metadata lookup)
        stage: "constructive", "lns_round_N", or None
        round_hint: Round number for LNS stages, or None
        line_hint_map: Optional map of order_id -> line_hint.
            For each order: use map value if non-empty, otherwise fall back to
            capability inference via _infer_line_hint_from_capability().
    """
    order_lookup: Dict = {}
    if not orders_df.empty:
        order_lookup = orders_df.set_index("order_id").to_dict("index")

    # Line-hint generation: per-order, always apply map-then-fallback.
    # We only pre-build the inferred map when line_hint_map is None (saves work).
    inferred_hint: Dict[str, str] = {}
    if line_hint_map is None:
        for oid, rec in order_lookup.items():
            inferred_hint[oid] = _infer_line_hint_from_capability(
                rec.get("line_capability", None)
            )

    # Diagnostics counters
    from_map_count = 0
    from_cap_count = 0

    rows = []
    for reason_str, oids in dropped_by_reason.items():
        for oid in oids:
            rec = order_lookup.get(oid, {})
            # A. map lookup
            map_hint = (line_hint_map.get(oid, "") if line_hint_map is not None else "")
            if map_hint:
                line_hint = map_hint
                from_map_count += 1
            else:
                # B. fallback to capability inference
                line_hint = (
                    inferred_hint.get(oid, "")
                    if line_hint_map is None
                    else _infer_line_hint_from_capability(rec.get("line_capability", None))
                )
                from_cap_count += 1
            rows.append({
                "order_id": oid,
                "drop_reason": reason_str,
                "stage": stage if stage is not None else "",
                "round": round_hint if round_hint is not None else -1,
                "line_hint": line_hint,
                "tons": rec.get("tons", 0.0),
                "width": rec.get("width", 0.0),
                "thickness": rec.get("thickness", 0.0),
                "steel_group": rec.get("steel_group", ""),
                "due_date": rec.get("due_date", ""),
            })

    df = pd.DataFrame(rows)
    # Attach diagnostics as DataFrame metadata (caller reads if needed)
    df.attrs["_hint_from_map_count"] = from_map_count
    df.attrs["_hint_from_capability_count"] = from_cap_count
    return df


# ---------------------------------------------------------------------------
# Helper: Score a solution snapshot
# ---------------------------------------------------------------------------

def _score_snapshot(snap: _SolutionSnapshot) -> Tuple:
    """Return the five-level comparison score for a snapshot."""
    return snap._score_tuple()


def _is_strictly_better(candidate: _SolutionSnapshot, incumbent: _SolutionSnapshot) -> bool:
    """
    Compare two solutions using the five-level objective.

    Returns True if candidate is strictly better than incumbent.
    Hard feasibility is always required (both must have 0 hard violations).
    """
    c_score = _score_snapshot(candidate)
    i_score = _score_snapshot(incumbent)
    return c_score < i_score


# ---------------------------------------------------------------------------
# Helper: Snapshot from current state
# ---------------------------------------------------------------------------

def _build_snapshot(
    planned_segs: List[CampaignSegment],
    dropped_by_reason: Dict[str, List[str]],
    tpl_df: pd.DataFrame,
) -> _SolutionSnapshot:
    """Build a _SolutionSnapshot from current state."""
    planned_by_line: Dict[str, List[str]] = {}
    planned_ids: Set[str] = set()
    total_tons = 0.0
    vb_count = 0
    tpl_cost = 0.0

    # Template cost lookup
    cost_lookup: Dict[Tuple[str, str], int] = {}
    if not tpl_df.empty:
        for _, trow in tpl_df.iterrows():
            key = (str(trow.get("from_order_id", "")), str(trow.get("to_order_id", "")))
            cost_lookup[key] = float(trow.get("cost", 0.0) or 0.0)
            if str(trow.get("edge_type", "")) == "VIRTUAL_BRIDGE_EDGE":
                cost_lookup[f"_vb_{key}"] = 1

    order_tons: Dict[str, float] = {}
    # We need tons from somewhere — estimate from segments
    for seg in planned_segs:
        if not seg.is_valid:
            continue
        planned_by_line.setdefault(seg.line, []).extend(seg.order_ids)
        for oid in seg.order_ids:
            planned_ids.add(oid)
            if oid not in order_tons:
                order_tons[oid] = 0.0  # Will be updated later

    # Count virtual bridges
    for seg in planned_segs:
        if not seg.is_valid:
            continue
        for i in range(len(seg.order_ids) - 1):
            prev, curr = seg.order_ids[i], seg.order_ids[i + 1]
            key = (prev, curr)
            if key in tpl_lookup_bridge(tpl_df) or _is_vb_edge(tpl_df, prev, curr):
                vb_count += 1
            tpl_cost += cost_lookup.get(key, 0.0)
            total_tons += 0.0  # placeholder

    return _SolutionSnapshot(
        planned_order_ids=planned_ids,
        planned_by_line=planned_by_line,
        dropped_by_reason=dropped_by_reason,
        total_tons=total_tons,
        virtual_bridge_count=vb_count,
        template_cost=tpl_cost,
    )


def _tpl_lookup_bridge(tpl_df: pd.DataFrame) -> Set[Tuple[str, str]]:
    s = set()
    if tpl_df.empty:
        return s
    for _, r in tpl_df.iterrows():
        if str(r.get("edge_type", "")) == "VIRTUAL_BRIDGE_EDGE":
            s.add((str(r.get("from_order_id", "")), str(r.get("to_order_id", ""))))
    return s


def _is_vb_edge(tpl_df: pd.DataFrame, from_oid: str, to_oid: str) -> bool:
    if tpl_df.empty:
        return False
    mask = (
        (tpl_df["from_order_id"].astype(str) == from_oid) &
        (tpl_df["to_order_id"].astype(str) == to_oid)
    )
    if mask.any():
        return str(tpl_df.loc[mask, "edge_type"].iloc[0]) == "VIRTUAL_BRIDGE_EDGE"
    return False


# ---------------------------------------------------------------------------
# Helper: Snapshot from planned_df + dropped_df
# ---------------------------------------------------------------------------

def _build_snapshot_from_dfs(
    planned_df: pd.DataFrame,
    dropped_df: pd.DataFrame,
    orders_df: pd.DataFrame,
    tpl_df: pd.DataFrame,
) -> _SolutionSnapshot:
    """Build a _SolutionSnapshot from output DataFrames."""
    planned_by_line: Dict[str, List[str]] = {}
    planned_ids: Set[str] = set()

    if not planned_df.empty:
        for _, row in planned_df.iterrows():
            oid = str(row["order_id"])
            line = str(row["line"])
            planned_ids.add(oid)
            planned_by_line.setdefault(line, []).append(oid)

    dropped_by_reason: Dict[str, List[str]] = {}
    if not dropped_df.empty:
        for _, row in dropped_df.iterrows():
            oid = str(row["order_id"])
            reason = str(row.get("drop_reason", "INITIAL_DROP"))
            dropped_by_reason.setdefault(reason, []).append(oid)

    total_tons = 0.0
    order_lookup = {}
    if not orders_df.empty:
        order_lookup = orders_df.set_index("order_id").to_dict("index")
    for oid in planned_ids:
        total_tons += float(order_lookup.get(oid, {}).get("tons", 0.0) or 0.0)

    # Count virtual bridges from planned_df
    vb_count = 0
    tpl_cost = 0.0
    if not planned_df.empty and not tpl_df.empty:
        # Group by campaign_id_hint to get sequence
        for _, grp in planned_df.groupby(["line", "campaign_id_hint"], dropna=False):
            seq = grp.sort_values("campaign_seq_hint")["order_id"].tolist()
            for i in range(len(seq) - 1):
                prev, curr = seq[i], seq[i + 1]
                if _is_vb_edge(tpl_df, prev, curr):
                    vb_count += 1
                mask = (
                    (tpl_df["from_order_id"].astype(str) == prev) &
                    (tpl_df["to_order_id"].astype(str) == curr)
                )
                if mask.any():
                    tpl_cost += float(tpl_df.loc[mask, "cost"].iloc[0] or 0.0)

    return _SolutionSnapshot(
        planned_order_ids=planned_ids,
        planned_by_line=planned_by_line,
        dropped_by_reason=dropped_by_reason,
        total_tons=total_tons,
        virtual_bridge_count=vb_count,
        template_cost=tpl_cost,
    )


# ---------------------------------------------------------------------------
# Helper: Neighborhood selection
# ---------------------------------------------------------------------------

def _select_neighborhood(
    strategy: NeighborhoodType,
    segments: List[CampaignSegment],
    dropped_by_reason: Dict[str, List[str]],
    orders_df: pd.DataFrame,
    rand: random.Random,
    cfg: PlannerConfig,
    underfilled_bias_segs: Optional[List[CampaignSegment]] = None,
) -> List[CampaignSegment]:
    """
    Select 1-2 campaign segments according to the given strategy.

    Current production strategies:
    - LOW_FILL_SEGMENT: segments with low fill % (near min) — enhanced multi-criteria
        scoring: gap_to_min, donor availability, dropped candidates, template integrity.
    - HIGH_DROP_PRESSURE: segments adjacent to many dropped orders
    - TAIL_REBALANCE: underfilled tails (or near-min valid segs) with donor candidates;
        when underfilled_bias_segs is provided, prioritize those.

    REMOVED strategies (experimental, no proven gain):
    - HIGH_VIRTUAL_USAGE: Removed - no business meaning in current path
    - SMALL_ROLL_RESCUE: Removed - not part of constructive_lns main path
    """
    if not segments:
        return []

    ton_min = float(cfg.rule.campaign_ton_min)
    ton_max = float(cfg.rule.campaign_ton_max)

    # ---- LOW_FILL_SEGMENT / TAIL_REBALANCE: use underfilled_bias_segs if available ----
    if strategy == NeighborhoodType.TAIL_REBALANCE and underfilled_bias_segs:
        # Prioritize underfilled segments, then fall back to valid segments near min
        scored: List[Tuple[float, CampaignSegment]] = []
        underfilled_ids = {s.campaign_local_id for s in underfilled_bias_segs if not s.is_valid}
        valid_segs = [s for s in segments if s.is_valid]

        # Score underfilled segments: smallest gap to min = highest priority
        for seg in valid_segs:
            if seg.campaign_local_id in underfilled_ids:
                gap = ton_min - seg.total_tons
                score = gap  # Smaller gap = lower score = higher priority
                scored.append((score, seg))
            else:
                # Also consider near-min valid segments
                if seg.total_tons >= ton_min:
                    fr = seg.total_tons / ton_min if ton_min > 0 else 0.0
                    score = abs(fr - 1.0) * 100  # Scale up so near-min still gets priority
                    scored.append((score, seg))

        if not scored:
            # Fall back to all valid segments sorted by fill ratio
            for seg in valid_segs:
                if seg.total_tons >= ton_min:
                    fr = seg.total_tons / ton_min if ton_min > 0 else 0.0
                    scored.append((abs(fr - 1.0), seg))
                else:
                    scored.append((0.0, seg))

        scored.sort(key=lambda x: x[0])
        return [scored[0][1]] if scored else []

    # ---- Standard selection for other strategies ----
    valid_segs = [s for s in segments if s.is_valid]
    if not valid_segs:
        return []

    scored = []
    for seg in valid_segs:
        if strategy == NeighborhoodType.LOW_FILL_SEGMENT:
            # Enhanced multi-criteria scoring:
            #   a) gap_to_min: how much below ton_min (0 if above min)
            gap_to_min = max(0.0, ton_min - seg.total_tons)
            #   b) fill ratio score: closer to 1.0 = lower score
            if seg.total_tons >= ton_min:
                fill_ratio = seg.total_tons / ton_min if ton_min > 0 else 0.0
                fill_score = abs(fill_ratio - 1.0) * 50
            else:
                fill_score = 0.0  # underfilled: highest priority
            # Combined: prioritize by gap (underfilled first), then by fill ratio
            # Scale gap to put underfilled segments clearly ahead
            score = gap_to_min + fill_score
            scored.append((score, seg))

        elif strategy == NeighborhoodType.HIGH_DROP_PRESSURE:
            # True local drop pressure: segment-specific based on nearby recoverable dropped orders
            pressure_info = _compute_segment_drop_pressure(seg, dropped_by_reason, orders_df, cfg)
            pressure = pressure_info["drop_pressure_score"]
            scored.append((pressure, seg))

        elif strategy == NeighborhoodType.WIDTH_TENSION_HOTSPOT:
            # Score by width range: larger range = higher hotspot tension
            if len(seg.order_ids) >= 2:
                widths = []
                for oid in seg.order_ids:
                    if oid in orders_df.set_index("order_id").index:
                        widths.append(float(orders_df.set_index("order_id").loc[oid].get("width", 0) or 0))
                width_range = max(widths) - min(widths) if widths else 0.0
                scored.append((width_range, seg))
            else:
                scored.append((0.0, seg))

        elif strategy == NeighborhoodType.GROUP_SWITCH_HOTSPOT:
            # Score by group switch count within segment
            if len(seg.order_ids) >= 2:
                groups = []
                for oid in seg.order_ids:
                    if oid in orders_df.set_index("order_id").index:
                        groups.append(str(orders_df.set_index("order_id").loc[oid].get("steel_group", "") or ""))
                switch_count = sum(1 for i in range(len(groups) - 1) if groups[i] != groups[i + 1])
                scored.append((switch_count, seg))
            else:
                scored.append((0.0, seg))

        elif strategy == NeighborhoodType.BRIDGE_DEPENDENT_SEGMENT:
            # Score by how underfilled + bridge edge dependent the segment is
            ton_min = float(cfg.rule.campaign_ton_min)
            gap = max(0.0, ton_min - seg.total_tons)
            # Also prefer segments with many orders (more candidates to disrupt)
            order_count_factor = len(seg.order_ids) / 10.0
            score = gap + order_count_factor
            scored.append((score, seg))

        elif strategy == NeighborhoodType.TAIL_REBALANCE:
            # Pick segments that are closest to ton_min (tail candidates).
            if seg.total_tons >= ton_min:
                fill_ratio = seg.total_tons / ton_min if ton_min > 0 else 0.0
                score = abs(fill_ratio - 1.0)
                scored.append((score, seg))
            else:
                scored.append((0.0, seg))
        else:
            scored.append((0.0, seg))

    # Shuffle ties and pick 1-2
    n_pick = rand.randint(1, min(2, len(scored)))
    if strategy in (NeighborhoodType.LOW_FILL_SEGMENT, NeighborhoodType.TAIL_REBALANCE):
        scored.sort(key=lambda x: x[0])  # Lower score = higher priority
    else:
        scored.sort(key=lambda x: x[0], reverse=True)

    # Add some randomness to avoid deterministic picks
    shuffle_range = min(3, len(scored))
    top_candidates = scored[:shuffle_range]
    rand.shuffle(top_candidates)
    return [seg for _, seg in top_candidates[:n_pick]]


# ---------------------------------------------------------------------------
# Helper: Destroy — remove orders from segments
# ---------------------------------------------------------------------------

def _destroy_orders_from_segments(
    segments: List[CampaignSegment],
    orders_to_remove: List[str],
    rand: random.Random,
) -> Tuple[List[CampaignSegment], List[str]]:
    """
    Remove orders from segments.

    Returns:
        - Updated segments (with removed orders; may become empty)
        - List of removed order IDs
    """
    remove_set = set(orders_to_remove)
    remaining_removed: List[str] = []

    new_segments: List[CampaignSegment] = []
    for seg in segments:
        if not seg.is_valid:
            new_segments.append(seg)
            continue

        kept = [oid for oid in seg.order_ids if oid not in remove_set]
        removed = [oid for oid in seg.order_ids if oid in remove_set]
        remaining_removed.extend(removed)

        if len(kept) == 0:
            # Segment is empty → mark invalid
            new_seg = CampaignSegment(
                line=seg.line,
                campaign_local_id=seg.campaign_local_id,
                order_ids=[],
                total_tons=0.0,
                cut_reason=seg.cut_reason,
                start_order_id="",
                end_order_id="",
                edge_count=0,
                is_valid=False,
            )
            new_segments.append(new_seg)
        elif len(kept) == len(seg.order_ids):
            # No removal from this segment
            new_segments.append(seg)
        else:
            # Partial removal — recalculate tons
            new_tons = 0.0
            ton_lookup = {}  # Would need orders_df; simplified here
            new_start = kept[0] if kept else ""
            new_end = kept[-1] if kept else ""
            new_seg = CampaignSegment(
                line=seg.line,
                campaign_local_id=seg.campaign_local_id,
                order_ids=kept,
                total_tons=new_tons,  # Caller should recalculate
                cut_reason=seg.cut_reason,
                start_order_id=new_start,
                end_order_id=new_end,
                edge_count=len(kept) - 1 if len(kept) > 1 else 0,
                is_valid=True,
            )
            new_segments.append(new_seg)

    return new_segments, remaining_removed


def _compute_destroy_count(
    segments: List[CampaignSegment],
    rand: random.Random,
    hotspot_strength: float = 0.0,
) -> int:
    """
    Compute how many orders to destroy (20%~35% of segment orders).

    Args:
        segments: Selected segments for destruction.
        rand: Random generator.
        hotspot_strength: [0,1] factor that biases destroy count upward when hotspots are strong.
            - hotspot_strength=0 → use lower end (20%)
            - hotspot_strength=1 → use upper end (35%)
    """
    total = sum(len(s.order_ids) for s in segments if s.is_valid)
    if total == 0:
        return 0
    base_frac = 0.20 + hotspot_strength * 0.15  # 0.20~0.35
    frac = rand.uniform(base_frac, min(0.35, base_frac + 0.05))
    return max(1, int(total * frac))


# ---------------------------------------------------------------------------
# Helper: Segment local drop-pressure scoring
# ---------------------------------------------------------------------------

def _compute_segment_drop_pressure(
    seg: CampaignSegment,
    dropped_by_reason: Dict[str, List[str]],
    orders_df: pd.DataFrame,
    cfg: PlannerConfig,
) -> dict:
    """
    Compute true local drop-pressure for a segment.

    Returns a dict with:
      - drop_pressure_score: combined score (higher = more pressure)
      - nearby_dropped_count: dropped orders with compatible line capability
      - same_line_capable_dropped_count: dropped orders on same line
      - width_compatible_dropped_count: width-delta within 250 of segment avg
      - group_compatible_or_bridgeable_count: same steel_group or bridgeable
      - tons_recoverable_estimate: estimated tonnage recoverable
      - segment_avg_width: average width of orders in segment
    """
    if orders_df.empty:
        return {
            "drop_pressure_score": 0.0,
            "nearby_dropped_count": 0,
            "same_line_capable_dropped_count": 0,
            "width_compatible_dropped_count": 0,
            "group_compatible_or_bridgeable_count": 0,
            "tons_recoverable_estimate": 0.0,
            "segment_avg_width": 0.0,
        }

    # Build order lookups
    order_lookup = orders_df.set_index("order_id")
    seg_oids = seg.order_ids

    # Segment properties
    seg_recs = {}
    for oid in seg_oids:
        if oid in order_lookup.index:
            seg_recs[oid] = dict(order_lookup.loc[oid])

    seg_avg_width = 0.0
    seg_widths = [float(r.get("width", 0) or 0) for r in seg_recs.values()]
    if seg_widths:
        seg_avg_width = sum(seg_widths) / len(seg_widths)

    seg_line_cap = set()
    for r in seg_recs.values():
        cap = str(r.get("line_capability", "dual") or "dual").lower()
        if cap in ("big_roll", "dual"):
            seg_line_cap.add("big_roll")
        if cap in ("small_roll", "dual"):
            seg_line_cap.add("small_roll")

    seg_groups = {str(r.get("steel_group", "") or "") for r in seg_recs.values()}

    # Collect all dropped orders
    all_dropped = []
    for oids in dropped_by_reason.values():
        all_dropped.extend(oids)

    nearby = 0
    same_line = 0
    width_compat = 0
    group_compat = 0
    tons_est = 0.0

    WIDTH_WINDOW = 250.0  # width delta tolerance
    for oid in all_dropped:
        if oid not in order_lookup.index:
            continue
        rec = dict(order_lookup.loc[oid])

        # 1. Line capability compatibility
        drop_cap = str(rec.get("line_capability", "dual") or "dual").lower()
        drop_line_ok = False
        if drop_cap in ("big_roll", "dual") and "big_roll" in seg_line_cap:
            drop_line_ok = True
        if drop_cap in ("small_roll", "dual") and "small_roll" in seg_line_cap:
            drop_line_ok = True

        if drop_line_ok:
            nearby += 1
        else:
            continue  # skip incompatible line

        # 2. Same line
        if seg.line == "big_roll" and drop_cap in ("big_roll", "dual"):
            same_line += 1
        elif seg.line == "small_roll" and drop_cap in ("small_roll", "dual"):
            same_line += 1

        # 3. Width compatibility (within window)
        drop_width = float(rec.get("width", 0) or 0)
        if abs(drop_width - seg_avg_width) <= WIDTH_WINDOW and seg_avg_width > 0:
            width_compat += 1

        # 4. Group compatibility or bridgeable
        drop_group = str(rec.get("steel_group", "") or "")
        if drop_group in seg_groups:
            group_compat += 1
        else:
            # Check if bridgeable (different groups but compatible thickness range)
            seg_thicks = [float(r.get("thickness", 0) or 0) for r in seg_recs.values() if float(r.get("thickness", 0) or 0) > 0]
            drop_thick = float(rec.get("thickness", 0) or 0)
            if seg_thicks and drop_thick > 0:
                thick_range = max(seg_thicks) - min(seg_thicks)
                if abs(drop_thick - (sum(seg_thicks) / len(seg_thicks))) <= thick_range * 0.5:
                    group_compat += 1  # bridgeable via thickness compatibility

        # 5. Tonnage estimate
        tons_est += float(rec.get("tons", 0) or 0)

    # Combined pressure score
    # Weight: nearby_dropped (0.4) + same_line (0.3) + width_compat (0.2) + group (0.1)
    score = (
        nearby * 0.4
        + same_line * 0.3
        + width_compat * 0.2
        + group_compat * 0.1
    )

    return {
        "drop_pressure_score": score,
        "nearby_dropped_count": nearby,
        "same_line_capable_dropped_count": same_line,
        "width_compatible_dropped_count": width_compat,
        "group_compatible_or_bridgeable_count": group_compat,
        "tons_recoverable_estimate": tons_est,
        "segment_avg_width": seg_avg_width,
    }


# ---------------------------------------------------------------------------
# Helper: Hotspot destroy candidates
# ---------------------------------------------------------------------------

def _collect_hotspot_destroy_candidates(
    seg: CampaignSegment,
    neighborhood: NeighborhoodType,
    orders_df: pd.DataFrame,
    orders_to_remove_count: int,
    rand: random.Random,
    drop_pressure_info: Optional[dict] = None,
) -> Tuple[List[str], dict]:
    """
    Collect hotspot-driven destroy candidates for a segment.

    Instead of uniform random sampling from middle orders, this focuses
    destruction around the business-defined "hotspot center" of each neighborhood.

    Returns:
        (candidate_order_ids, hotspot_diag)
    """
    hotspot_diag: dict = {
        "hotspot_type": neighborhood.value,
        "hotspot_center_order_id": "",
        "hotspot_window_size": 0,
        "hotspot_candidates": [],
        "hotspot_random_backfill_count": 0,
    }

    if orders_df.empty or not seg.order_ids or len(seg.order_ids) <= 2:
        return [], hotspot_diag

    order_lookup = orders_df.set_index("order_id")
    seg_oids = list(seg.order_ids)
    n = len(seg_oids)

    # Build per-order metadata
    def get_rec(oid: str) -> dict:
        if oid in order_lookup.index:
            return dict(order_lookup.loc[oid])
        return {}

    # ---- Identify hotspot center and window ----
    hotspot_center_idx = -1
    hotspot_window_start = 1  # default: middle orders only
    hotspot_window_end = n - 1

    if neighborhood == NeighborhoodType.LOW_FILL_SEGMENT:
        # Hotspot: near tail (end), where gap_to_min is largest
        # Prefer destroying last 2-4 middle orders (not anchors)
        hotspot_window_start = max(1, n - 4)
        hotspot_window_end = n - 1
        hotspot_center_idx = seg_oids[min(n - 2, n - 1)] if seg_oids else ""

    elif neighborhood == NeighborhoodType.HIGH_DROP_PRESSURE:
        # Hotspot: position where width/group most matches dropped pool
        dp_info = drop_pressure_info or {}
        seg_avg_w = dp_info.get("segment_avg_width", 0.0)
        # Find position where order width is closest to segment avg width
        best_idx = 1
        best_diff = float("inf")
        for i, oid in enumerate(seg_oids):
            rec = get_rec(oid)
            w = float(rec.get("width", 0) or 0)
            diff = abs(w - seg_avg_w) if seg_avg_w > 0 else float("inf")
            if diff < best_diff:
                best_diff = diff
                best_idx = i
        hotspot_center_idx = seg_oids[best_idx] if seg_oids else ""
        hotspot_window_start = max(1, best_idx - 1)
        hotspot_window_end = min(n - 1, best_idx + 2)

    elif neighborhood == NeighborhoodType.TAIL_REBALANCE:
        # Hotspot: last 2-3 middle orders (tail anchor stays, but near-tail is soft)
        hotspot_window_start = max(1, n - 3)
        hotspot_window_end = n - 1
        hotspot_center_idx = seg_oids[n - 2] if seg_oids and n > 1 else ""

    elif neighborhood == NeighborhoodType.GROUP_SWITCH_HOTSPOT:
        # Hotspot: group switch edge in segment
        # Find the position with most group transitions
        switch_positions = []
        for i in range(n - 1):
            rec_i = get_rec(seg_oids[i])
            rec_j = get_rec(seg_oids[i + 1])
            g_i = str(rec_i.get("steel_group", "") or "")
            g_j = str(rec_j.get("steel_group", "") or "")
            if g_i != g_j:
                switch_positions.append(i + 1)  # index after the switch
        if switch_positions:
            hotspot_center_idx = seg_oids[min(switch_positions, key=lambda x: abs(x - len(seg_oids) // 2))]
            # Window: ±1 around switch position
            center_pos = switch_positions[0]
            hotspot_window_start = max(1, center_pos - 1)
            hotspot_window_end = min(n - 1, center_pos + 2)
        else:
            # No switch found: use last third
            hotspot_window_start = max(1, n * 2 // 3)
            hotspot_window_end = n - 1
            hotspot_center_idx = seg_oids[hotspot_window_start]

    elif neighborhood == NeighborhoodType.WIDTH_TENSION_HOTSPOT:
        # Hotspot: largest width delta position
        max_delta = 0.0
        max_delta_pos = n // 2
        for i in range(n - 1):
            rec_i = get_rec(seg_oids[i])
            rec_j = get_rec(seg_oids[i + 1])
            w_i = float(rec_i.get("width", 0) or 0)
            w_j = float(rec_j.get("width", 0) or 0)
            delta = abs(w_i - w_j)
            if delta > max_delta:
                max_delta = delta
                max_delta_pos = i + 1
        hotspot_center_idx = seg_oids[max_delta_pos] if seg_oids else ""
        hotspot_window_start = max(1, max_delta_pos - 1)
        hotspot_window_end = min(n - 1, max_delta_pos + 2)

    elif neighborhood == NeighborhoodType.BRIDGE_DEPENDENT_SEGMENT:
        # Hotspot: positions adjacent to existing REAL_BRIDGE_EDGE in segment
        # For now, use the center of the segment as proxy
        center_pos = n // 2
        hotspot_center_idx = seg_oids[center_pos] if seg_oids else ""
        hotspot_window_start = max(1, center_pos - 1)
        hotspot_window_end = min(n - 1, center_pos + 2)

    else:
        # Fallback: use all middle orders
        hotspot_window_start = 1
        hotspot_window_end = n - 1
        hotspot_center_idx = seg_oids[n // 2] if seg_oids else ""

    # Clamp window
    hotspot_window_start = max(1, min(hotspot_window_start, n - 1))
    hotspot_window_end = max(hotspot_window_start, min(hotspot_window_end, n - 1))
    hotspot_window_size = max(0, hotspot_window_end - hotspot_window_start + 1)

    # Collect window candidates (exclude anchors at index 0 and n-1)
    window_candidates = seg_oids[hotspot_window_start:hotspot_window_end + 1]
    hotspot_diag["hotspot_window_size"] = hotspot_window_size
    hotspot_diag["hotspot_center_order_id"] = str(hotspot_center_idx) if hotspot_center_idx != -1 else ""

    # If window has enough candidates, prioritize them
    if len(window_candidates) >= orders_to_remove_count:
        selected = rand.sample(window_candidates, orders_to_remove_count)
        hotspot_diag["hotspot_candidates"] = selected
        hotspot_diag["hotspot_random_backfill_count"] = 0
        return selected, hotspot_diag

    # Otherwise: take all window candidates, then backfill from other middle orders
    selected = list(window_candidates)
    remaining = orders_to_remove_count - len(selected)

    # Backfill from other middle orders (not in window, not anchors)
    all_middle = set(seg_oids[1:n - 1]) - set(window_candidates)
    if remaining > 0 and all_middle:
        backfill_pool = list(all_middle)
        rand.shuffle(backfill_pool)
        backfill = backfill_pool[:remaining]
        selected.extend(backfill)
        hotspot_diag["hotspot_random_backfill_count"] = len(backfill)

    hotspot_diag["hotspot_candidates"] = selected[:orders_to_remove_count]
    return selected[:orders_to_remove_count], hotspot_diag


def _compute_hotspot_strength(
    neighborhood: NeighborhoodType,
    seg: Optional[CampaignSegment],
    orders_df: pd.DataFrame,
    cfg: PlannerConfig,
    drop_pressure_info: Optional[dict] = None,
) -> float:
    """
    Compute a [0,1] hotspot_strength score for the given neighborhood.

    Used to bias destroy_count upward (more aggressive destruction) when
    the segment has strong hotspot signals.
    """
    if seg is None or orders_df.empty:
        return 0.0

    order_lookup = orders_df.set_index("order_id")
    seg_oids = seg.order_ids

    if neighborhood == NeighborhoodType.HIGH_DROP_PRESSURE:
        dp_info = drop_pressure_info or _compute_segment_drop_pressure(seg, {}, orders_df, cfg)
        score = dp_info.get("drop_pressure_score", 0.0)
        # Normalize: score > 5 is strong hotspot
        return min(1.0, score / 5.0)

    elif neighborhood == NeighborhoodType.WIDTH_TENSION_HOTSPOT:
        widths = []
        for oid in seg_oids:
            if oid in order_lookup.index:
                widths.append(float(order_lookup.loc[oid].get("width", 0) or 0))
        if len(widths) < 2:
            return 0.0
        width_range = max(widths) - min(widths)
        # range > 300 is strong hotspot
        return min(1.0, width_range / 300.0)

    elif neighborhood == NeighborhoodType.GROUP_SWITCH_HOTSPOT:
        groups = []
        for oid in seg_oids:
            if oid in order_lookup.index:
                groups.append(str(order_lookup.loc[oid].get("steel_group", "") or ""))
        if len(groups) < 2:
            return 0.0
        # count switches
        switches = sum(1 for i in range(len(groups) - 1) if groups[i] != groups[i + 1])
        # 2+ switches = strong
        return min(1.0, switches / 2.0)

    elif neighborhood == NeighborhoodType.BRIDGE_DEPENDENT_SEGMENT:
        # Bridge-dependent means segment is underfilled with bridge edges used
        ton_min = float(cfg.rule.campaign_ton_min)
        if seg.total_tons < ton_min:
            gap_ratio = (ton_min - seg.total_tons) / ton_min
            return min(1.0, gap_ratio)
        return 0.0

    elif neighborhood == NeighborhoodType.LOW_FILL_SEGMENT:
        ton_min = float(cfg.rule.campaign_ton_min)
        gap_ratio = max(0.0, ton_min - seg.total_tons) / ton_min if ton_min > 0 else 0.0
        return min(1.0, gap_ratio * 2.0)  # amplify so 0.2 gap → 0.4 strength

    elif neighborhood == NeighborhoodType.TAIL_REBALANCE:
        ton_min = float(cfg.rule.campaign_ton_min)
        gap_ratio = max(0.0, ton_min - seg.total_tons) / ton_min if ton_min > 0 else 0.0
        return min(1.0, gap_ratio * 1.5)

    return 0.0


# ---------------------------------------------------------------------------
# Helper: Tail repair actions for TAIL_REBALANCE neighborhood
# ---------------------------------------------------------------------------

def _try_tail_donor_shift(
    tail_seg: CampaignSegment,
    segments: List[CampaignSegment],
    order_tons: Dict[str, float],
    cfg: PlannerConfig,
    tpl_df: pd.DataFrame | None,
    rand: random.Random,
) -> Tuple[bool, List[CampaignSegment], dict]:
    """
    Action A: Shift 1..K orders from donor (preceding valid segment on same line)
    to the underfilled tail.

    Returns: (success, updated_segments, action_diag)
    """
    action_diag: dict = {
        "action": "DONOR_SHIFT",
        "attempted": True,
        "success": False,
        "shift_k": 0,
        "failure_reason": "",
    }

    # Find donor segment: preceding segment on same line
    donor_seg: Optional[CampaignSegment] = None
    donor_idx = -1
    for i, seg in enumerate(segments):
        if seg.line == tail_seg.line and seg.is_valid:
            if seg.campaign_local_id < tail_seg.campaign_local_id:
                if donor_seg is None or seg.campaign_local_id > donor_seg.campaign_local_id:
                    donor_seg = seg
                    donor_idx = i

    if donor_seg is None or donor_idx < 0:
        action_diag["failure_reason"] = "NO_ADJACENT_DONOR"
        return False, segments, action_diag

    ton_min = float(cfg.rule.campaign_ton_min)
    ton_max = float(cfg.rule.campaign_ton_max)
    ton_target = float(cfg.rule.campaign_ton_target)

    donor_oids = list(donor_seg.order_ids)
    tail_oids = list(tail_seg.order_ids)

    if len(donor_oids) < 2:
        action_diag["failure_reason"] = "DONOR_WOULD_UNDERFILL"
        return False, segments, action_diag

    max_k = min(8, len(donor_oids) - 1)
    best_k = 0
    best_score = float("inf")
    best_donor_new_tons = 0.0
    best_tail_new_tons = 0.0

    def check_template_pairs(oids: List[str]) -> bool:
        if tpl_df is None or tpl_df.empty or len(oids) < 2:
            return True
        tpl_keys: set = set()
        for _, trow in tpl_df.iterrows():
            tpl_keys.add((str(trow.get("from_order_id", "")), str(trow.get("to_order_id", ""))))
        for i in range(len(oids) - 1):
            if (oids[i], oids[i + 1]) not in tpl_keys:
                return False
        return True

    for k in range(1, max_k + 1):
        pull_oids = donor_oids[-k:]
        pull_tons = sum(order_tons.get(oid, 0.0) for oid in pull_oids)

        donor_new_tons = donor_seg.total_tons - pull_tons
        tail_new_tons = tail_seg.total_tons + pull_tons

        if donor_new_tons < ton_min - 1e-6:
            continue
        if tail_new_tons < ton_min - 1e-6:
            continue
        if donor_new_tons > ton_max + 1e-6:
            continue
        if tail_new_tons > ton_max + 1e-6:
            continue

        # Validate template pairs for new donor tail
        new_donor_oids = donor_oids[:-k]
        if not check_template_pairs(new_donor_oids):
            continue
        if not check_template_pairs(pull_oids + tail_oids):
            continue

        donor_dev = abs(donor_new_tons - ton_target)
        tail_dev = abs(tail_new_tons - ton_target)
        score = (donor_dev + tail_dev) + k * ton_target * 0.02

        if score < best_score:
            best_score = score
            best_k = k
            best_donor_new_tons = donor_new_tons
            best_tail_new_tons = tail_new_tons

    if best_k == 0:
        action_diag["failure_reason"] = "DONOR_WOULD_UNDERFILL"
        return False, segments, action_diag

    new_donor_oids = donor_oids[:-best_k]
    new_tail_oids = donor_oids[-best_k:] + tail_oids

    new_donor_seg = CampaignSegment(
        line=donor_seg.line,
        campaign_local_id=donor_seg.campaign_local_id,
        order_ids=new_donor_oids,
        total_tons=best_donor_new_tons,
        cut_reason=donor_seg.cut_reason,
        start_order_id=new_donor_oids[0] if new_donor_oids else "",
        end_order_id=new_donor_oids[-1] if new_donor_oids else "",
        edge_count=max(0, len(new_donor_oids) - 1),
        is_valid=(best_donor_new_tons >= ton_min),
    )

    new_tail_seg = CampaignSegment(
        line=tail_seg.line,
        campaign_local_id=tail_seg.campaign_local_id,
        order_ids=new_tail_oids,
        total_tons=best_tail_new_tons,
        cut_reason=tail_seg.cut_reason,
        start_order_id=new_tail_oids[0],
        end_order_id=new_tail_oids[-1],
        edge_count=max(0, len(new_tail_oids) - 1),
        is_valid=(best_tail_new_tons >= ton_min),
    )

    updated = list(segments)
    updated[donor_idx] = new_donor_seg

    # Replace tail in the list
    tail_idx = next((i for i, s in enumerate(updated) if s.campaign_local_id == tail_seg.campaign_local_id and s.line == tail_seg.line), -1)
    if tail_idx >= 0:
        updated[tail_idx] = new_tail_seg
    else:
        updated.append(new_tail_seg)

    action_diag["success"] = True
    action_diag["shift_k"] = best_k

    print(
        f"[APS][TailRepairLNS] action=DONOR_SHIFT, tail_id={tail_seg.campaign_local_id}, "
        f"shift_orders={best_k}, donor_before={donor_seg.total_tons:.0f}, "
        f"donor_after={best_donor_new_tons:.0f}, tail_before={tail_seg.total_tons:.0f}, "
        f"tail_after={best_tail_new_tons:.0f}"
    )

    return True, updated, action_diag


def _try_tail_dropped_reinsertion(
    tail_seg: CampaignSegment,
    dropped_pool: List[str],
    order_tons: Dict[str, float],
    orders_df: pd.DataFrame,
    tpl_df: pd.DataFrame | None,
    cfg: PlannerConfig,
    rand: random.Random,
) -> Tuple[bool, CampaignSegment, dict]:
    """
    Action C: Insert compatible dropped orders into underfilled tail.

    Tries to fill the gap between current tail tons and campaign_ton_min
    by appending compatible dropped orders to the tail.

    Returns: (success, updated_tail_seg, action_diag)
    """
    action_diag: dict = {
        "action": "DROPPED_REINSERT",
        "attempted": True,
        "success": False,
        "inserted_count": 0,
        "inserted_orders": [],
        "failure_reason": "",
    }

    if not dropped_pool:
        action_diag["failure_reason"] = "NO_DROPPED_CANDIDATES"
        return False, tail_seg, action_diag

    ton_min = float(cfg.rule.campaign_ton_min)
    ton_max = float(cfg.rule.campaign_ton_max)
    gap = ton_min - tail_seg.total_tons
    if gap <= 0:
        action_diag["failure_reason"] = "NOT_UNDERFILLED"
        return False, tail_seg, action_diag

    # Build order lookup
    order_lookup: Dict = {}
    if not orders_df.empty:
        order_lookup = orders_df.set_index("order_id").to_dict("index")

    # Build template adjacency
    tpl_out: set = set()
    if tpl_df is not None and not tpl_df.empty:
        for _, row in tpl_df.iterrows():
            from_oid = str(row.get("from_order_id", ""))
            tpl_out.add(from_oid)

    # Score candidates: prefer those that bridge from tail's last order
    tail_last_oid = tail_seg.order_ids[-1] if tail_seg.order_ids else ""
    candidates: List[Tuple[float, str]] = []
    for oid in dropped_pool:
        rec = order_lookup.get(oid, {})
        raw_cap = rec.get("line_capability", "")
        cap_lines = _normalize_allowed_lines(raw_cap, [])
        if tail_seg.line not in cap_lines:
            continue
        oid_tons = order_tons.get(oid, 0.0)
        if oid_tons <= 0:
            continue
        # Score: smaller tons that fill the gap better = lower score
        gap_fill_score = abs(oid_tons - gap)
        # Bonus: if this oid is a template successor of tail's last order
        successor_bonus = 0.0
        if tail_last_oid and tpl_df is not None and not tpl_df.empty:
            for _, row in tpl_df.iterrows():
                if str(row.get("from_order_id", "")) == tail_last_oid and str(row.get("to_order_id", "")) == oid:
                    successor_bonus = -50.0  # strongly prefer direct successors
                    break
        score = gap_fill_score + successor_bonus
        candidates.append((score, oid))

    if not candidates:
        action_diag["failure_reason"] = "NO_VALID_DROPPED_CANDIDATE"
        return False, tail_seg, action_diag

    candidates.sort(key=lambda x: x[0])
    # Try to insert up to 5 candidates
    inserted: List[str] = []
    new_tail_tons = tail_seg.total_tons
    new_tail_oids = list(tail_seg.order_ids)

    for _, oid in candidates[:5]:
        oid_tons = order_tons.get(oid, 0.0)
        if new_tail_tons + oid_tons > ton_max:
            continue
        # Validate template pair
        if tpl_df is not None and not tpl_df.empty and new_tail_oids:
            last_oid = new_tail_oids[-1]
            valid_pair = False
            for _, row in tpl_df.iterrows():
                if str(row.get("from_order_id", "")) == last_oid and str(row.get("to_order_id", "")) == oid:
                    valid_pair = True
                    break
            if not valid_pair:
                continue
        inserted.append(oid)
        new_tail_oids.append(oid)
        new_tail_tons += oid_tons
        if new_tail_tons >= ton_min:
            break

    if not inserted:
        action_diag["failure_reason"] = "NO_VALID_INSERT_POSITION"
        return False, tail_seg, action_diag

    new_tail_seg = CampaignSegment(
        line=tail_seg.line,
        campaign_local_id=tail_seg.campaign_local_id,
        order_ids=new_tail_oids,
        total_tons=new_tail_tons,
        cut_reason=tail_seg.cut_reason,
        start_order_id=new_tail_oids[0],
        end_order_id=new_tail_oids[-1],
        edge_count=max(0, len(new_tail_oids) - 1),
        is_valid=(new_tail_tons >= ton_min),
    )

    action_diag["success"] = True
    action_diag["inserted_count"] = len(inserted)
    action_diag["inserted_orders"] = inserted

    print(
        f"[APS][TailRepairLNS] action=DROPPED_REINSERT, tail_id={tail_seg.campaign_local_id}, "
        f"inserted={inserted}, tail_before={tail_seg.total_tons:.0f}, "
        f"tail_after={new_tail_tons:.0f}, is_valid={new_tail_seg.is_valid}"
    )

    return True, new_tail_seg, action_diag


def _try_tail_recut_two(
    tail_seg: CampaignSegment,
    segments: List[CampaignSegment],
    order_tons: Dict[str, float],
    cfg: PlannerConfig,
    tpl_df: pd.DataFrame | None,
    rand: random.Random,
) -> Tuple[bool, List[CampaignSegment], dict]:
    """
    Action B: Recut donor segment + underfilled tail as a combined window.
    Find a new cut point where both parts meet ton_min/max.

    Returns: (success, updated_segments, action_diag)
    """
    action_diag: dict = {
        "action": "RECUT_TWO",
        "attempted": True,
        "success": False,
        "cut_index": -1,
        "failure_reason": "",
    }

    # Find donor
    donor_seg: Optional[CampaignSegment] = None
    donor_idx = -1
    for i, seg in enumerate(segments):
        if seg.line == tail_seg.line and seg.is_valid:
            if seg.campaign_local_id < tail_seg.campaign_local_id:
                if donor_seg is None or seg.campaign_local_id > donor_seg.campaign_local_id:
                    donor_seg = seg
                    donor_idx = i

    if donor_seg is None or donor_idx < 0:
        action_diag["failure_reason"] = "NO_ADJACENT_DONOR"
        return False, segments, action_diag

    ton_min = float(cfg.rule.campaign_ton_min)
    ton_max = float(cfg.rule.campaign_ton_max)
    ton_target = float(cfg.rule.campaign_ton_target)

    all_oids = list(donor_seg.order_ids) + list(tail_seg.order_ids)
    n = len(all_oids)
    if n < 2:
        action_diag["failure_reason"] = "NOT_ENOUGH_ORDERS"
        return False, segments, action_diag

    prefix_tons: List[float] = [0.0] * (n + 1)
    for i in range(n):
        prefix_tons[i + 1] = prefix_tons[i] + order_tons.get(all_oids[i], 0.0)

    def check_template(oids: List[str]) -> bool:
        if tpl_df is None or tpl_df.empty or len(oids) < 2:
            return True
        tpl_keys: set = set()
        for _, trow in tpl_df.iterrows():
            tpl_keys.add((str(trow.get("from_order_id", "")), str(trow.get("to_order_id", ""))))
        for i in range(len(oids) - 1):
            if (oids[i], oids[i + 1]) not in tpl_keys:
                return False
        return True

    best_k = -1
    best_score = float("inf")
    best_left_tons = 0.0
    best_right_tons = 0.0

    for k in range(1, n):
        left_tons = prefix_tons[k]
        right_tons = prefix_tons[n] - left_tons

        if left_tons < ton_min - 1e-6 or left_tons > ton_max + 1e-6:
            continue
        if right_tons < ton_min - 1e-6 or right_tons > ton_max + 1e-6:
            continue

        if not check_template(all_oids[:k]):
            continue
        if not check_template(all_oids[k:]):
            continue

        left_dev = abs(left_tons - ton_target)
        right_dev = abs(right_tons - ton_target)
        score = left_dev + right_dev

        if score < best_score:
            best_score = score
            best_k = k
            best_left_tons = left_tons
            best_right_tons = right_tons

    if best_k == -1:
        action_diag["failure_reason"] = "NO_VALID_RECUT_POINT"
        return False, segments, action_diag

    new_left_oids = all_oids[:best_k]
    new_right_oids = all_oids[best_k:]

    new_donor_seg = CampaignSegment(
        line=donor_seg.line,
        campaign_local_id=donor_seg.campaign_local_id,
        order_ids=new_left_oids,
        total_tons=best_left_tons,
        cut_reason=donor_seg.cut_reason,
        start_order_id=new_left_oids[0],
        end_order_id=new_left_oids[-1],
        edge_count=max(0, len(new_left_oids) - 1),
        is_valid=(best_left_tons >= ton_min),
    )

    new_tail_seg = CampaignSegment(
        line=tail_seg.line,
        campaign_local_id=tail_seg.campaign_local_id,
        order_ids=new_right_oids,
        total_tons=best_right_tons,
        cut_reason=tail_seg.cut_reason,
        start_order_id=new_right_oids[0],
        end_order_id=new_right_oids[-1],
        edge_count=max(0, len(new_right_oids) - 1),
        is_valid=(best_right_tons >= ton_min),
    )

    updated = list(segments)
    updated[donor_idx] = new_donor_seg
    tail_idx = next((i for i, s in enumerate(updated) if s.campaign_local_id == tail_seg.campaign_local_id and s.line == tail_seg.line), -1)
    if tail_idx >= 0:
        updated[tail_idx] = new_tail_seg
    else:
        updated.append(new_tail_seg)

    action_diag["success"] = True
    action_diag["cut_index"] = best_k

    print(
        f"[APS][TailRepairLNS] action=RECUT_TWO, tail_id={tail_seg.campaign_local_id}, "
        f"cut_index={best_k}, left_tons={best_left_tons:.0f}, right_tons={best_right_tons:.0f}"
    )

    return True, updated, action_diag


# ---------------------------------------------------------------------------
# Helper: Recalculate segment tons from orders_df
# ---------------------------------------------------------------------------

def _recalculate_segment_tons(
    segments: List[CampaignSegment],
    orders_df: pd.DataFrame,
) -> List[CampaignSegment]:
    """Recompute total_tons for segments based on orders_df."""
    if orders_df.empty:
        return segments
    ton_map: Dict[str, float] = {}
    for _, row in orders_df.iterrows():
        ton_map[str(row["order_id"])] = float(row.get("tons", 0.0) or 0.0)

    result = []
    for seg in segments:
        total = sum(ton_map.get(oid, 0.0) for oid in seg.order_ids)
        result.append(
            CampaignSegment(
                line=seg.line,
                campaign_local_id=seg.campaign_local_id,
                order_ids=list(seg.order_ids),
                total_tons=total,
                cut_reason=seg.cut_reason,
                start_order_id=seg.order_ids[0] if seg.order_ids else "",
                end_order_id=seg.order_ids[-1] if seg.order_ids else "",
                edge_count=seg.edge_count,
                is_valid=seg.is_valid,
            )
        )
    return result


# ---------------------------------------------------------------------------
# Helper: Filter dropped pool candidates for a specific line
# ---------------------------------------------------------------------------

def _pool_candidates_for_line(
    line: str,
    pool_order_ids: Sequence[str],
    orders_df: pd.DataFrame,
    transition_pack: dict,
    max_count: Optional[int] = None,
) -> Tuple[List[str], Dict[str, int]]:
    """
    Filter dropped pool candidates to those that are:
      A. Capability-compatible with the given line
      B. Have at least one template edge (incoming or outgoing) on that line's graph

    Args:
        line: "big_roll" or "small_roll"
        pool_order_ids: Sequence of order IDs from the dropped pool to consider
        orders_df: Full orders DataFrame (for line_capability lookup)
        transition_pack: Contains "templates" DataFrame keyed by line
        max_count: Optional cap on the number of returned candidates (random sample).

    Returns:
        (filtered_candidates, diagnostics) where diagnostics contains:
            - filtered_by_capability_count
            - filtered_by_connectivity_count
            - skipped_already_tried_count (always 0 at this level; tracked at caller)
            - total_input_count
    """
    diag: Dict[str, int] = {
        "filtered_by_capability_count": 0,
        "filtered_by_connectivity_count": 0,
        "skipped_already_tried_count": 0,
        "total_input_count": len(pool_order_ids),
    }

    if not pool_order_ids:
        return [], diag

    # ---- Build order capability lookup ----
    order_lookup: Dict = {}
    if not orders_df.empty:
        order_lookup = orders_df.set_index("order_id").to_dict("index")

    # ---- Build per-line template adjacency sets from transition_pack ----
    # templates DataFrame may have a "line" column; build sets per line
    out_by_line: Dict[str, Set[str]] = {"big_roll": set(), "small_roll": set()}
    in_by_line: Dict[str, Set[str]] = {"big_roll": set(), "small_roll": set()}

    tpl_df_raw = _get_templates(transition_pack)
    if not tpl_df_raw.empty:
        for _, trow in tpl_df_raw.iterrows():
            from_oid = str(trow.get("from_order_id", ""))
            to_oid = str(trow.get("to_order_id", ""))
            tpl_line = str(trow.get("line", "big_roll"))
            if tpl_line not in out_by_line:
                out_by_line[tpl_line] = set()
                in_by_line[tpl_line] = set()
            out_by_line[tpl_line].add(from_oid)
            in_by_line[tpl_line].add(to_oid)

    candidates: List[str] = []
    diag_warn_bucket: List[str] = []   # not used here but _normalize_allowed_lines expects it

    for oid in pool_order_ids:
        # ---- Filter A: capability ----
        rec = order_lookup.get(oid, {})
        raw_cap = rec.get("line_capability", None)
        allowed_lines = _normalize_allowed_lines(raw_cap, diag_warn_bucket)

        if line not in allowed_lines:
            diag["filtered_by_capability_count"] += 1
            continue

        # ---- Filter B: template connectivity on this line ----
        line_out = out_by_line.get(line, set())
        line_in = in_by_line.get(line, set())
        has_out = oid in line_out
        has_in = oid in line_in

        if not (has_out or has_in):
            diag["filtered_by_connectivity_count"] += 1
            continue

        candidates.append(oid)

    # Apply max_count cap (random shuffle to avoid ordering bias)
    if max_count is not None and len(candidates) > max_count:
        import random as _rand
        candidates = _rand.sample(candidates, max_count)

    return candidates, diag


# ---------------------------------------------------------------------------
# Main ALNS Engine
# ---------------------------------------------------------------------------

def _run_alns_iteration(
    current_segs: List[CampaignSegment],
    current_dropped: Dict[str, List[str]],
    orders_df: pd.DataFrame,
    tpl_df: pd.DataFrame,
    transition_pack: dict,
    cfg: PlannerConfig,
    neighborhood: NeighborhoodType,
    round_num: int,
    rand: random.Random,
    max_destroy_orders: int = 45,
    time_limit: float = 10.0,
    underfilled_segs: Optional[List[CampaignSegment]] = None,
    repair_family_edges: Optional[List] = None,
    family_repair_already_attempted_keys: Optional[set] = None,
) -> Tuple[
    List[CampaignSegment],
    Dict[str, List[str]],
    bool,
    int,
    int,
    int,
    int,
    int,
    int,
    Dict[str, str],
    Dict[str, int],
    dict,
]:
    """
    Run one ALNS iteration (destroy + repair).

    For TAIL_REBALANCE neighborhood: uses specialized multi-action repair
    targeting underfilled tails (donor shift, merge, recut, dropped reinsertion).

    For LOW_FILL_SEGMENT neighborhood: targets the most repairable low-fill segment
    with multi-criteria scoring.

    Returns:
        (new_segments, new_dropped, accepted, destroy_count, repair_count,
         hard_protected_drop_count, lns_repair_rejected_count,
         repair_segment_candidate_count, repair_pool_candidate_count,
         round_line_hint_map, pool_stats, tail_repair_diag)
        where tail_repair_diag contains TAIL_REBALANCE/LNS repair metrics.
    """
    tail_repair_diag: dict = {
        "tail_rebalance_lns_success_count": 0,
        "tail_rebalance_lns_shifted_orders": 0,
        "tail_rebalance_lns_inserted_dropped_orders": 0,
        "tail_rebalance_lns_merge_count": 0,
        "tail_rebalance_lns_recut_count": 0,
        "tail_rebalance_lns_fail_reasons": [],
        "tail_rebalance_lns_dropped_reinsertion_attempts": 0,
        "tail_rebalance_lns_dropped_reinsertion_success": 0,
        "local_inserter_direct_arcs_allowed": 0,
        "local_inserter_real_bridge_arcs_allowed": 0,
        "local_inserter_real_bridge_arcs_blocked": 0,
        "local_inserter_virtual_bridge_arcs_blocked": 0,
        "local_inserter_edge_policy_used": "",
        "low_fill_candidates": 0,
        "low_fill_avg_gap_to_min": 0.0,
        "low_fill_selected_gap": 0.0,
        # ---- Local CP-SAT gate diagnostics ----
        "local_cpsat_skipped_due_to_gate": 0,
        "local_cpsat_attempt_count": 0,
        "local_cpsat_success_count": 0,
        "local_cpsat_no_improve_count": 0,
        "local_cpsat_infeasible_count": 0,
        "local_cpsat_total_seconds": 0.0,
        "local_cpsat_skip_reason_histogram": {},
        # ---- Hotspot destroy diagnostics ----
        "destroy_hotspot_type": "",
        "destroy_hotspot_center": "",
        "destroy_window_size": 0,
        "destroy_hotspot_candidates_count": 0,
        "destroy_random_backfill_count": 0,
        "drop_pressure_score": 0.0,
        "drop_pressure_nearby_dropped_count": 0,
        "drop_pressure_recoverable_tons_estimate": 0.0,
        # ---- Family repair pool diagnostics ----
        "repair_family_edge_pool_seen_count": 0,
        "repair_family_edge_pool_filtered_count": 0,
        "repair_family_edge_pool_used_count": 0,
        "candidate_source": "graph_unused_nodes",
        "graph_unused_candidates_total": 0,
        "graph_unused_real_candidates": 0,
        "graph_unused_virtual_candidates": 0,
        "graph_unused_repair_attempts": 0,
        "graph_unused_repair_success": 0,
        "virtual_node_repair_attempts": 0,
        "virtual_node_repair_success": 0,
    }

    # ---- Step 1: Select neighborhood ----
    # For TAIL_REBALANCE, pass underfilled_segs to bias selection toward them
    if neighborhood == NeighborhoodType.TAIL_REBALANCE and underfilled_segs:
        # Use underfilled_segs as primary candidates, fall back to valid segs
        selected = _select_neighborhood(
            neighborhood, current_segs, current_dropped, orders_df, rand, cfg,
            underfilled_bias_segs=underfilled_segs,
        )
    else:
        selected = _select_neighborhood(
            neighborhood, current_segs, current_dropped, orders_df, rand, cfg,
        )

    # LOW_FILL diagnostics: how many candidates were considered
    ton_min = float(cfg.rule.campaign_ton_min)
    if neighborhood == NeighborhoodType.LOW_FILL_SEGMENT and current_segs:
        valid_segs = [s for s in current_segs if s.is_valid]
        tail_repair_diag["low_fill_candidates"] = len(valid_segs)
        if valid_segs:
            gaps = [max(0.0, ton_min - s.total_tons) for s in valid_segs]
            tail_repair_diag["low_fill_avg_gap_to_min"] = round(sum(gaps) / len(gaps), 2)
        if selected:
            tail_repair_diag["low_fill_selected_gap"] = round(
                max(0.0, ton_min - selected[0].total_tons), 2
            )

    if not selected:
        return current_segs, current_dropped, False, 0, 0, 0, 0, 0, 0, {}, {
            "filtered_by_capability_count": 0,
            "filtered_by_connectivity_count": 0,
            "skipped_already_tried_count": 0,
        }, tail_repair_diag

    # ---- Step 2: Compute destroy count with hotspot strength ----
    # Compute per-segment hotspot_strength for destroy count biasing
    primary_seg = selected[0]
    dp_info = None
    if neighborhood == NeighborhoodType.HIGH_DROP_PRESSURE:
        dp_info = _compute_segment_drop_pressure(primary_seg, current_dropped, orders_df, cfg)
    hotspot_strength = _compute_hotspot_strength(
        neighborhood, primary_seg, orders_df, cfg, drop_pressure_info=dp_info
    )
    destroy_count = _compute_destroy_count(selected, rand, hotspot_strength=hotspot_strength)
    if destroy_count == 0:
        return current_segs, current_dropped, False, 0, 0, 0, 0, 0, 0, {}, {
            "filtered_by_capability_count": 0,
            "filtered_by_connectivity_count": 0,
            "skipped_already_tried_count": 0,
        }, tail_repair_diag

    # ---- Step 2b: Specialized Tail Repair for TAIL_REBALANCE neighborhood ----
    # Apply multi-action repair to underfilled tails BEFORE standard destroy
    if neighborhood == NeighborhoodType.TAIL_REBALANCE and selected:
        target_seg = selected[0]
        # Build order tons lookup
        order_tons: Dict[str, float] = {}
        if not orders_df.empty:
            for _, row in orders_df.iterrows():
                order_tons[str(row["order_id"])] = float(row.get("tons", 0.0) or 0.0)

        # Collect all dropped orders across all reasons
        all_dropped: List[str] = []
        for oids in current_dropped.values():
            all_dropped.extend(oids)

        tail_seg_to_repair = target_seg
        repair_succeeded = False

        # Action A: Try donor shift
        ok_a, updated_a, diag_a = _try_tail_donor_shift(
            tail_seg_to_repair, current_segs, order_tons, cfg, tpl_df, rand
        )
        if ok_a:
            current_segs = updated_a
            tail_repair_diag["tail_rebalance_lns_success_count"] += 1
            tail_repair_diag["tail_rebalance_lns_shifted_orders"] += diag_a.get("shift_k", 0)
            repair_succeeded = True
            print(
                f"[APS][TailRepairLNS] round={round_num}, action=DONOR_SHIFT, "
                f"tail_id={tail_seg_to_repair.campaign_local_id}, "
                f"success=True, shifted={diag_a.get('shift_k', 0)}"
            )
        else:
            tail_repair_diag["tail_rebalance_lns_fail_reasons"].append(
                f"DONOR_SHIFT:{diag_a.get('failure_reason', 'unknown')}"
            )

        # Action B: Try recut two segments (if donor shift failed)
        if not repair_succeeded:
            ok_b, updated_b, diag_b = _try_tail_recut_two(
                tail_seg_to_repair, current_segs, order_tons, cfg, tpl_df, rand
            )
            if ok_b:
                current_segs = updated_b
                tail_repair_diag["tail_rebalance_lns_success_count"] += 1
                tail_repair_diag["tail_rebalance_lns_recut_count"] += 1
                repair_succeeded = True
                print(
                    f"[APS][TailRepairLNS] round={round_num}, action=RECUT_TWO, "
                    f"tail_id={tail_seg_to_repair.campaign_local_id}, "
                    f"success=True, cut_index={diag_b.get('cut_index', -1)}"
                )
            else:
                tail_repair_diag["tail_rebalance_lns_fail_reasons"].append(
                    f"RECUT_TWO:{diag_b.get('failure_reason', 'unknown')}"
                )

        # Action C: Try dropped order reinsertion (if both failed and tail still underfilled)
        if not repair_succeeded:
            # Re-find the tail segment (in case it was updated)
            tail_in_segs = next(
                (s for s in current_segs
                 if s.campaign_local_id == tail_seg_to_repair.campaign_local_id
                 and s.line == tail_seg_to_repair.line),
                None
            )
            if tail_in_segs is not None and not tail_in_segs.is_valid:
                tail_repair_diag["tail_rebalance_lns_dropped_reinsertion_attempts"] += 1
                ok_c, updated_tail_c, diag_c = _try_tail_dropped_reinsertion(
                    tail_in_segs, all_dropped, order_tons, orders_df, tpl_df, cfg, rand
                )
                if ok_c:
                    # Update the segment in current_segs
                    current_segs = [
                        updated_tail_c if (
                            s.campaign_local_id == tail_seg_to_repair.campaign_local_id
                            and s.line == tail_seg_to_repair.line
                        ) else s
                        for s in current_segs
                    ]
                    tail_repair_diag["tail_rebalance_lns_success_count"] += 1
                    tail_repair_diag["tail_rebalance_lns_inserted_dropped_orders"] += diag_c.get(
                        "inserted_count", 0
                    )
                    tail_repair_diag["tail_rebalance_lns_dropped_reinsertion_success"] += 1
                    repair_succeeded = True
                    print(
                        f"[APS][TailRepairLNS] round={round_num}, action=DROPPED_REINSERT, "
                        f"tail_id={tail_seg_to_repair.campaign_local_id}, "
                        f"success=True, inserted={diag_c.get('inserted_count', 0)}"
                    )
                else:
                    tail_repair_diag["tail_rebalance_lns_fail_reasons"].append(
                        f"DROPPED_REINSERT:{diag_c.get('failure_reason', 'unknown')}"
                    )

        if not repair_succeeded:
            print(
                f"[APS][TailRepairLNS] round={round_num}, tail_id={tail_seg_to_repair.campaign_local_id}, "
                f"all_actions_failed, reasons={tail_repair_diag['tail_rebalance_lns_fail_reasons'][-3:]}"
            )

    # ---- Step 2c: LOW_FILL_SEGMENT dropped reinsertion for underfilled segments ----
    if neighborhood == NeighborhoodType.LOW_FILL_SEGMENT and selected:
        target_seg = selected[0]
        if not target_seg.is_valid:
            # Try dropped reinsertion for underfilled selected segment
            order_tons: Dict[str, float] = {}
            if not orders_df.empty:
                for _, row in orders_df.iterrows():
                    order_tons[str(row["order_id"])] = float(row.get("tons", 0.0) or 0.0)
            all_dropped: List[str] = []
            for oids in current_dropped.values():
                all_dropped.extend(oids)
            tail_repair_diag["tail_rebalance_lns_dropped_reinsertion_attempts"] += 1
            ok_lf, updated_lf, diag_lf = _try_tail_dropped_reinsertion(
                target_seg, all_dropped, order_tons, orders_df, tpl_df, cfg, rand
            )
            if ok_lf:
                current_segs = [
                    updated_lf if (
                        s.campaign_local_id == target_seg.campaign_local_id
                        and s.line == target_seg.line
                    ) else s
                    for s in current_segs
                ]
                tail_repair_diag["tail_rebalance_lns_dropped_reinsertion_success"] += 1
                tail_repair_diag["tail_rebalance_lns_inserted_dropped_orders"] += diag_lf.get(
                    "inserted_count", 0
                )
                print(
                    f"[APS][TailRepairLNS] round={round_num}, action=LOW_FILL_DROPPED_REINSERT, "
                    f"tail_id={target_seg.campaign_local_id}, "
                    f"success=True, inserted={diag_lf.get('inserted_count', 0)}"
                )

    # ---- Step 3: Select orders to destroy (hotspot-driven) ----
    # Use hotspot-centered window instead of uniform random sampling
    orders_to_remove: List[str] = []
    _destroy_hotspot_diag: dict = {
        "destroy_hotspot_type": neighborhood.value,
        "destroy_hotspot_center": "",
        "destroy_window_size": 0,
        "destroy_hotspot_candidates": [],
        "destroy_random_backfill_count": 0,
    }
    _target_seg = selected[0]  # primary target segment
    if len(_target_seg.order_ids) > 2:
        _candidates, _hs_diag = _collect_hotspot_destroy_candidates(
            _target_seg,
            neighborhood,
            orders_df,
            destroy_count,
            rand,
            drop_pressure_info=dp_info,
        )
        orders_to_remove = _candidates
        _destroy_hotspot_diag = _hs_diag
    else:
        # Too short to apply hotspot: use fallback
        orders_to_remove = _target_seg.order_ids[1:destroy_count + 1]
        _destroy_hotspot_diag["hotspot_window_size"] = 0
        _destroy_hotspot_diag["hotspot_random_backfill_count"] = len(orders_to_remove)

    # Inject hotspot destroy diagnostics into tail_repair_diag
    tail_repair_diag["destroy_hotspot_type"] = str(neighborhood.value)
    tail_repair_diag["destroy_hotspot_center"] = str(_destroy_hotspot_diag.get("hotspot_center_order_id", ""))
    tail_repair_diag["destroy_window_size"] = int(_destroy_hotspot_diag.get("hotspot_window_size", 0))
    tail_repair_diag["destroy_hotspot_candidates_count"] = len(_destroy_hotspot_diag.get("hotspot_candidates", []))
    tail_repair_diag["destroy_random_backfill_count"] = int(_destroy_hotspot_diag.get("hotspot_random_backfill_count", 0))
    # Drop pressure diagnostics (available for HIGH_DROP_PRESSURE neighborhood)
    if dp_info is not None:
        tail_repair_diag["drop_pressure_score"] = float(dp_info.get("drop_pressure_score", 0.0))
        tail_repair_diag["drop_pressure_nearby_dropped_count"] = int(dp_info.get("nearby_dropped_count", 0))
        tail_repair_diag["drop_pressure_recoverable_tons_estimate"] = float(dp_info.get("tons_recoverable_estimate", 0.0))

    # ---- Step 4: Apply destruction ----
    new_segs, removed = _destroy_orders_from_segments(
        current_segs, orders_to_remove, rand,
    )

    # Recalculate tons
    new_segs = _recalculate_segment_tons(new_segs, orders_df)

    # Copy current dropped pool
    tentative_dropped: Dict[str, List[str]] = {
        k: list(v) for k, v in current_dropped.items()
    }
    # Round-level line_hint_map: tracks which line each order was attempted on this round
    round_line_hint_map: Dict[str, str] = {}

    # Place destroyed orders into LNS_REPAIR_REJECTED bucket initially;
    # they may be rescued if the repair subproblem accepts them.
    tentative_dropped.setdefault(DropReason.LNS_REPAIR_REJECTED.value, []).extend(removed)

    # ---- Step 5: Collect candidate pool from dropped ----
    all_dropped_pool: List[str] = []
    for oids in tentative_dropped.values():
        all_dropped_pool.extend(oids)

    pool_size = min(max_destroy_orders - len(orders_to_remove), len(all_dropped_pool))
    if pool_size > 0:
        candidates_from_pool = rand.sample(all_dropped_pool, pool_size)
    else:
        candidates_from_pool = []

    # ---- Step 6: Repair — call local inserter for each line ----
    repair_count = 0

    # Round-level stats (accumulated across all lines in this round)
    round_hard_protected_drop_count = 0
    round_lns_repair_rejected_count = 0
    round_repair_seg_candidates = 0
    round_repair_pool_candidates = 0

    # Per-line pool diagnostics (accumulated for the final tuple)
    total_pool_cap_filtered = 0
    total_pool_conn_filtered = 0
    total_pool_already_tried = 0

    # ---- ALNS family round-level stats (accumulated across lines in this round) ----
    _round_alns_family_attempt = 0
    _round_alns_family_accept = 0
    _round_alns_family_reject = 0
    _round_alns_family_budget_block = 0
    _round_total_local_family_selected = 0

    # ---- Compute pressure signals for family edge gating ----
    # These are used to decide if should_enable_virtual_family_for_neighborhood returns True.
    _dropped_count = sum(len(v) for v in current_dropped.values())
    _ton_min = float(cfg.rule.campaign_ton_min)
    _underfill_detected = any(
        s.total_tons < _ton_min for s in new_segs if s.is_valid
    )
    _drop_pressure_score = float(_dropped_count) / max(1, len(orders_df))
    _total_tons_in_segs = sum(s.total_tons for s in new_segs if s.is_valid)
    _width_tension_hotspot = False
    _group_switch_hotspot = False
    _bridge_dependent_segment = False
    # Detect hotspots from segments: wide-to-narrow transitions, group switches, low-fill bridge segments
    for seg in new_segs:
        if not seg.is_valid or not seg.order_ids:
            continue
        if seg.total_tons < _ton_min:
            # Underfilled segment may be bridge-dependent
            _bridge_dependent_segment = True
        # Check for group transitions / width tension within segment
        if len(seg.order_ids) >= 2:
            seg_recs = {oid: dict(orders_df.set_index("order_id").loc[oid]) for oid in seg.order_ids if oid in orders_df.set_index("order_id").index}
            widths = [float(r.get("width", 0) or 0) for r in seg_recs.values()]
            if len(widths) >= 2:
                width_range = max(widths) - min(widths)
                if width_range > 200:
                    _width_tension_hotspot = True
            groups = [str(r.get("steel_group", "") or "") for r in seg_recs.values()]
            if len(set(groups)) >= 3:
                _group_switch_hotspot = True

    # Decide round-level family enablement ONCE per ALNS round (before per-line loop)
    _profile_family_enabled = getattr(cfg.model, "virtual_family_frontload_enabled", False)
    if _profile_family_enabled:
        _allow_guarded_family_this_round, _family_enable_reason = should_enable_virtual_family_for_neighborhood(
            neighborhood,
            cfg,
            dropped_count=_dropped_count,
            underfill_detected=_underfill_detected,
            drop_pressure_score=_drop_pressure_score,
            group_switch_hotspot=_group_switch_hotspot,
            width_tension_hotspot=_width_tension_hotspot,
            bridge_dependent_segment=_bridge_dependent_segment,
        )
    else:
        _allow_guarded_family_this_round = False
        _family_enable_reason = "profile_disabled"

    # Track which pool candidates have already been assigned to a line this round
    # to avoid sending the same dropped order to multiple lines simultaneously
    already_assigned_pool_candidates: Set[str] = set()

    for line in ["big_roll", "small_roll"]:
        # Get segments for this line
        line_segs = [s for s in new_segs if s.line == line]
        if not line_segs:
            continue

        # Collect fixed + candidate orders for this line
        fixed_orders: List[str] = []
        seg_order_set: Set[str] = set()
        for seg in line_segs:
            for oid in seg.order_ids:
                seg_order_set.add(oid)

        # First + last orders in each segment are "anchors" (fixed-ish)
        for seg in line_segs:
            if seg.order_ids:
                fixed_orders.append(seg.order_ids[0])  # keep start
                if len(seg.order_ids) > 1:
                    fixed_orders.append(seg.order_ids[-1])  # keep end

        fixed_orders = list(dict.fromkeys(fixed_orders))  # dedupe
        seg_orders = list(seg_order_set)

        # Segment reinsert candidates: segment non-fixed orders (destroyed ones)
        segment_reinsert_candidates = [oid for oid in seg_orders if oid not in fixed_orders]

        # Track line_hint for segment candidates (which segment/line they came from)
        for oid in segment_reinsert_candidates:
            round_line_hint_map[oid] = line

        candidate_graph = transition_pack.get("candidate_graph") if isinstance(transition_pack, dict) else None
        graph_orders_df = transition_pack.get("graph_orders_df") if isinstance(transition_pack, dict) else orders_df
        used_now = set()
        for _seg in current_segs:
            used_now.update(str(_oid) for _oid in getattr(_seg, "order_ids", []) or [])
        used_now.difference_update(set(str(_oid) for _oid in orders_to_remove))
        boundary_nodes = list(dict.fromkeys(fixed_orders + [str(_oid) for _oid in seg_orders if str(_oid) not in orders_to_remove]))
        graph_pool = collect_graph_unused_candidates(
            campaigns=current_segs,
            used_node_ids=used_now | set(already_assigned_pool_candidates),
            candidate_graph=candidate_graph,
            graph_orders_df=graph_orders_df,
            line=line,
            boundary_nodes=boundary_nodes,
            cfg=cfg,
        )
        pool_candidates = list(dict.fromkeys(graph_pool.real_candidates + graph_pool.virtual_candidates))[:max_destroy_orders]
        pool_diag = dict(graph_pool.diagnostics or {})
        pool_diag.setdefault("filtered_by_capability_count", 0)
        pool_diag.setdefault("filtered_by_connectivity_count", 0)
        pool_diag.setdefault("skipped_already_tried_count", 0)
        tail_repair_diag["candidate_source"] = "graph_unused_nodes"
        tail_repair_diag["graph_unused_candidates_total"] = (
            tail_repair_diag.get("graph_unused_candidates_total", 0)
            + int(pool_diag.get("graph_unused_candidates_total", 0) or 0)
        )
        tail_repair_diag["graph_unused_real_candidates"] = (
            tail_repair_diag.get("graph_unused_real_candidates", 0)
            + int(pool_diag.get("graph_unused_real_candidates", 0) or 0)
        )
        tail_repair_diag["graph_unused_virtual_candidates"] = (
            tail_repair_diag.get("graph_unused_virtual_candidates", 0)
            + int(pool_diag.get("graph_unused_virtual_candidates", 0) or 0)
        )
        tail_repair_diag["graph_unused_repair_attempts"] = (
            tail_repair_diag.get("graph_unused_repair_attempts", 0) + len(pool_candidates)
        )
        tail_repair_diag["virtual_node_repair_attempts"] = (
            tail_repair_diag.get("virtual_node_repair_attempts", 0)
            + sum(1 for oid in pool_candidates if str(oid) in set(graph_pool.virtual_candidates))
        )

        # Per-round dedup: mark these as assigned so the other line won't retry them
        for oid in pool_candidates:
            already_assigned_pool_candidates.add(oid)

        dropped_pool_candidates = pool_candidates

        # Accumulate per-line diagnostics
        total_pool_cap_filtered += pool_diag["filtered_by_capability_count"]
        total_pool_conn_filtered += pool_diag["filtered_by_connectivity_count"]
        total_pool_already_tried += pool_diag["skipped_already_tried_count"]

        # Track line_hint for dropped pool candidates (which line is attempting repair)
        for oid in dropped_pool_candidates:
            round_line_hint_map[oid] = line

        # Total candidate list for local inserter
        candidate_orders = list(segment_reinsert_candidates) + list(dropped_pool_candidates)

        # Track stats
        round_repair_seg_candidates += len(segment_reinsert_candidates)
        round_repair_pool_candidates += len(dropped_pool_candidates)

        if not candidate_orders:
            continue

        # ---- Unified local CP-SAT gate ----
        max_cpsat_rounds = int(getattr(cfg.model, "lns_max_total_rounds", 10) if cfg.model else 10)
        _gate_ok, _gate_reason = should_run_local_cpsat(
            neighborhood,
            cfg,
            len(candidate_orders),
            round_num=round_num,
            max_cpsat_rounds=max_cpsat_rounds,
        )
        if not _gate_ok:
            tail_repair_diag["local_cpsat_skipped_due_to_gate"] += 1
            _hist = tail_repair_diag.get("local_cpsat_skip_reason_histogram", {})
            _hist[_gate_reason] = _hist.get(_gate_reason, 0) + 1
            tail_repair_diag["local_cpsat_skip_reason_histogram"] = _hist
            continue

        # Build subproblem with request-level family controls
        # prefer_guarded_virtual_family: when ALNS neighborhood is primary family neighborhood,
        # indicate that family edges should be moderately preferred in scoring
        _prefer_family = bool(_allow_guarded_family_this_round and (
            neighborhood in {
                NeighborhoodType.LOW_FILL_SEGMENT,
                NeighborhoodType.HIGH_DROP_PRESSURE,
                NeighborhoodType.GROUP_SWITCH_HOTSPOT,
            }
        ))
        _family_reason = (
            f"primary_neighborhood={neighborhood.value}"
            if _prefer_family
            else ("secondary_neighborhood" if _allow_guarded_family_this_round else "disabled")
        )

        # ---- Compute ALNS-round-level repair family edge keys ----
        # Only when guarded family is enabled this round, filter repair_family_edges
        # to form the round-local family repair pool for this line.
        _round_family_keys: set = set()
        _rfe_pool_seen = 0
        _rfe_pool_filtered = 0
        _round_family_keys: set = set()
        _req_family_allowed = list(getattr(cfg.model, "virtual_family_frontload_allowed_families", []) if cfg.model else [])
        _req_family_max_bridge = int(getattr(cfg.model, "virtual_family_frontload_max_bridge_count", 2) if cfg.model else 2)
        _local_subproblem_oids = set(fixed_orders) | set(candidate_orders)
        _already_attempted = family_repair_already_attempted_keys or set()

        if _allow_guarded_family_this_round and repair_family_edges:
            for _edge in repair_family_edges:
                _rfe_pool_seen += 1
                # Line filter
                if str(_edge.line) != line:
                    _rfe_pool_filtered += 1
                    continue
                # Edge type filter
                if str(_edge.edge_type) != "VIRTUAL_BRIDGE_FAMILY_EDGE":
                    _rfe_pool_filtered += 1
                    continue
                # Connectivity filter: at least one endpoint in local subproblem
                if str(_edge.from_order_id) not in _local_subproblem_oids and \
                   str(_edge.to_order_id) not in _local_subproblem_oids:
                    _rfe_pool_filtered += 1
                    continue
                # Family allowlist filter
                _bf = str(_edge.bridge_family or "MIXED")
                if _req_family_allowed and _bf not in _req_family_allowed:
                    _rfe_pool_filtered += 1
                    continue
                # Max bridge count filter (use effective_max_bridge_count)
                _ebc = int(getattr(_edge, "estimated_bridge_count_max", 0) or 0)
                if _ebc <= 0:
                    _ebc = int(getattr(_edge, "estimated_bridge_count", 0) or 0)
                if _req_family_max_bridge > 0 and _ebc > _req_family_max_bridge:
                    _rfe_pool_filtered += 1
                    continue
                # Already attempted dedup
                _key = (line, str(_edge.from_order_id), str(_edge.to_order_id), _bf)
                if _key in _already_attempted:
                    _rfe_pool_filtered += 1
                    continue
                _round_family_keys.add(_key)

        # Accumulate round-level diagnostics
        tail_repair_diag["repair_family_edge_pool_seen_count"] = _rfe_pool_seen
        tail_repair_diag["repair_family_edge_pool_filtered_count"] = _rfe_pool_filtered
        tail_repair_diag["repair_family_edge_pool_used_count"] = len(_round_family_keys)
        # Pass round keys back to caller for dedup tracking
        tail_repair_diag["_round_repair_family_edge_keys"] = set(_round_family_keys)

        # Hotspot context for local CP-SAT (available from destroy phase)
        _hs_center = str(_destroy_hotspot_diag.get("hotspot_center_order_id", "") or
                         _destroy_hotspot_diag.get("destroy_hotspot_center", ""))
        _hs_reason = (
            f"round={round_num}, neighborhood={neighborhood.value}, "
            f"hotspot_center={_hs_center}, "
            f"window_size={_destroy_hotspot_diag.get('hotspot_window_size', 0)}, "
            f"backfill={_destroy_hotspot_diag.get('hotspot_random_backfill_count', 0)}"
        )
        _order_context_by_id = {}
        if isinstance(orders_df, pd.DataFrame) and not orders_df.empty and "order_id" in orders_df.columns:
            _order_context_by_id = {
                str(row.get("order_id", "")): dict(row)
                for row in orders_df.to_dict("records")
            }

        req = LocalInsertRequest(
            line=line,
            fixed_order_ids=fixed_orders,
            candidate_insert_ids=candidate_orders,
            time_limit_seconds=time_limit,
            random_seed=rand.randint(1, 999999),
            max_orders_in_subproblem=max_destroy_orders,
            # Guarded virtual family request-level controls
            allow_guarded_virtual_family=_allow_guarded_family_this_round,
            virtual_family_budget_for_rebuild=int(getattr(cfg.model, "virtual_family_budget_per_line", 4) if cfg.model else 4),
            virtual_family_allowed_families=list(getattr(cfg.model, "virtual_family_frontload_allowed_families", []) if cfg.model else []),
            virtual_family_max_bridge_count=int(getattr(cfg.model, "virtual_family_frontload_max_bridge_count", 2) if cfg.model else 2),
            # Family repair scoring preference (guarded profile)
            prefer_guarded_virtual_family=_prefer_family,
            guarded_virtual_family_reason=_family_reason,
            # ALNS-round-level repair family edge keys
            repair_family_edge_keys=list(_round_family_keys),
            # Hotspot-driven destroy context
            hotspot_type=str(neighborhood.value),
            hotspot_center_order_id=_hs_center,
            hotspot_reason=_hs_reason,
            order_context_by_id=_order_context_by_id,
        )

        import time as _time_module
        _cpsat_t0 = _time_module.perf_counter()
        result = solve_local_insertion_subproblem(
            req, orders_df, transition_pack, cfg,
        )
        _cpsat_elapsed = _time_module.perf_counter() - _cpsat_t0

        # Record gate-aware CP-SAT diagnostics
        tail_repair_diag["local_cpsat_attempt_count"] += 1
        tail_repair_diag["local_cpsat_total_seconds"] += _cpsat_elapsed
        if result.status in (InsertStatus.OPTIMAL, InsertStatus.FEASIBLE):
            tail_repair_diag["local_cpsat_success_count"] += 1
        elif result.status == InsertStatus.INFEASIBLE:
            tail_repair_diag["local_cpsat_infeasible_count"] += 1
        else:
            tail_repair_diag["local_cpsat_no_improve_count"] += 1
        if isinstance(result.diagnostics, dict):
            tail_repair_diag["local_inserter_direct_arcs_allowed"] += int(result.diagnostics.get("direct_arcs_allowed", 0) or 0)
            tail_repair_diag["local_inserter_real_bridge_arcs_allowed"] += int(result.diagnostics.get("real_bridge_arcs_allowed", 0) or 0)
            tail_repair_diag["local_inserter_real_bridge_arcs_blocked"] += int(result.diagnostics.get("real_bridge_arcs_blocked", 0) or 0)
            tail_repair_diag["local_inserter_virtual_bridge_arcs_blocked"] += int(result.diagnostics.get("virtual_bridge_arcs_blocked", 0) or 0)
            tail_repair_diag["local_inserter_edge_policy_used"] = str(result.diagnostics.get("edge_policy_used", "") or tail_repair_diag.get("local_inserter_edge_policy_used", ""))

            # Track ALNS family stats from local inserter result
            if _allow_guarded_family_this_round:
                _local_family_selected = int(result.diagnostics.get("rebuild_virtual_family_selected_count", 0) or 0)
                _local_family_budget_blocked = int(result.diagnostics.get("rebuild_virtual_family_budget_blocked_count", 0) or 0)
                _local_prefer = bool(result.diagnostics.get("rebuild_guarded_virtual_family_preferred", 0))
                _local_reason = str(result.diagnostics.get("rebuild_guarded_virtual_family_reason", ""))
                # Attempt = this line's request allowed guarded family
                _round_alns_family_attempt += 1
                _round_total_local_family_selected += _local_family_selected
                if _local_family_selected > 0:
                    _round_alns_family_accept += 1
                else:
                    _round_alns_family_reject += 1
                if _local_family_budget_blocked > 0:
                    _round_alns_family_budget_block += 1
                # ---- Family repair value metrics (guarded profile) ----
                # These are tracked in round_diag, then aggregated to LnsEngineDiag
                _gain_dropped = (
                    len(set(getattr(result, "inserted_order_ids", []) or []) & set(dropped_pool_candidates))
                    if result.status in (InsertStatus.OPTIMAL, InsertStatus.FEASIBLE)
                    else 0
                )
                _gain_underfilled = 1 if (_local_family_selected > 0 and _underfill_detected) else 0
                _gain_scheduled = len(accepted_ids) if _local_family_selected > 0 else 0
                tail_repair_diag["family_repair_attempt_count"] += 1
                tail_repair_diag["family_repair_accept_count"] += (_local_family_selected > 0)
                tail_repair_diag["family_repair_gain_dropped"] += _gain_dropped
                tail_repair_diag["family_repair_gain_underfilled"] += _gain_underfilled
                tail_repair_diag["family_repair_gain_scheduled_orders"] += _gain_scheduled
                tail_repair_diag["family_repair_no_gain_count"] += (1 if (_local_family_selected == 0 and _gain_dropped == 0) else 0)
                tail_repair_diag["family_repair_prefer_count"] = tail_repair_diag.get("family_repair_prefer_count", 0) + int(_local_prefer)
                tail_repair_diag["family_repair_avg_seconds_per_success"] = (
                    (tail_repair_diag.get("family_repair_avg_seconds_per_success", 0.0) * max(0, tail_repair_diag.get("family_repair_accept_count", 0) - 1) + _cpsat_elapsed)
                    / max(1, tail_repair_diag.get("family_repair_accept_count", 0))
                )

        if result.status in (InsertStatus.OPTIMAL, InsertStatus.FEASIBLE):
            repair_count += 1
            accepted_ids = set(result.inserted_order_ids)
            tail_repair_diag["graph_unused_repair_success"] = (
                tail_repair_diag.get("graph_unused_repair_success", 0)
                + len(accepted_ids & set(dropped_pool_candidates))
            )
            tail_repair_diag["virtual_node_repair_success"] = (
                tail_repair_diag.get("virtual_node_repair_success", 0)
                + sum(
                    1
                    for oid in accepted_ids & set(dropped_pool_candidates)
                    if str(oid).startswith(("VIRTUAL_", "PREBUILT_VIRTUAL_"))
                )
            )

            # Rescue candidates that WERE in LNS_REPAIR_REJECTED and got accepted
            for oid in accepted_ids:
                reason_list = tentative_dropped.get(DropReason.LNS_REPAIR_REJECTED.value, [])
                if oid in reason_list:
                    reason_list.remove(oid)
                # Also remove from any other reason list (shouldn't happen, but be safe)
                for reason, oids_list in tentative_dropped.items():
                    if oid in oids_list:
                        oids_list.remove(oid)

            # ---- Determine fate per candidate source ----
            accepted_seg = accepted_ids & set(segment_reinsert_candidates)
            accepted_pool = accepted_ids & set(dropped_pool_candidates)
            rejected_seg = set(segment_reinsert_candidates) - accepted_ids
            rejected_pool = set(dropped_pool_candidates) - accepted_ids

            # Segment candidates NOT accepted → LNS_REPAIR_REJECTED (not hard-protected)
            for oid in rejected_seg:
                tentative_dropped.setdefault(
                    DropReason.LNS_REPAIR_REJECTED.value, []
                ).append(oid)
            round_lns_repair_rejected_count += len(rejected_seg)

            # Pool candidates NOT accepted → HARD_CONSTRAINT_PROTECTED_DROP
            if rejected_pool:
                unique_rejected_pool = list(dict.fromkeys(list(rejected_pool)))
                tentative_dropped.setdefault(
                    DropReason.HARD_CONSTRAINT_PROTECTED_DROP.value, []
                ).extend(unique_rejected_pool)
                round_hard_protected_drop_count += len(unique_rejected_pool)

        elif result.status == InsertStatus.INFEASIBLE:
            # Entire subproblem infeasible: segment candidates → LNS_REPAIR_REJECTED,
            # pool candidates → HARD_CONSTRAINT_PROTECTED_DROP
            for oid in segment_reinsert_candidates:
                tentative_dropped.setdefault(
                    DropReason.LNS_REPAIR_REJECTED.value, []
                ).append(oid)
            round_lns_repair_rejected_count += len(segment_reinsert_candidates)

            if dropped_pool_candidates:
                unique_pool = list(dict.fromkeys(dropped_pool_candidates))
                tentative_dropped.setdefault(
                    DropReason.HARD_CONSTRAINT_PROTECTED_DROP.value, []
                ).extend(unique_pool)
                round_hard_protected_drop_count += len(unique_pool)

        elif result.status == InsertStatus.NO_IMPROVEMENT:
            # Model found a solution but accepted zero candidates.
            # Segment candidates → LNS_REPAIR_REJECTED (just couldn't find a slot)
            for oid in segment_reinsert_candidates:
                tentative_dropped.setdefault(
                    DropReason.LNS_REPAIR_REJECTED.value, []
                ).append(oid)
            round_lns_repair_rejected_count += len(segment_reinsert_candidates)

            # Pool candidates → HARD_CONSTRAINT_PROTECTED_DROP (protected by hard constraints)
            if dropped_pool_candidates:
                unique_pool = list(dict.fromkeys(dropped_pool_candidates))
                tentative_dropped.setdefault(
                    DropReason.HARD_CONSTRAINT_PROTECTED_DROP.value, []
                ).extend(unique_pool)
                round_hard_protected_drop_count += len(unique_pool)

    # Dedupe the LNS_REPAIR_REJECTED bucket (may have duplicates from multiple lines)
    lns_rejected_list = tentative_dropped.get(DropReason.LNS_REPAIR_REJECTED.value, [])
    if lns_rejected_list:
        tentative_dropped[DropReason.LNS_REPAIR_REJECTED.value] = list(dict.fromkeys(lns_rejected_list))

    # Note: HARD_CONSTRAINT_PROTECTED_DROP is already deduped via dict.fromkeys above
    # before extending; but we also dedupe the bucket here for safety
    hard_list = tentative_dropped.get(DropReason.HARD_CONSTRAINT_PROTECTED_DROP.value, [])
    if hard_list:
        tentative_dropped[DropReason.HARD_CONSTRAINT_PROTECTED_DROP.value] = list(dict.fromkeys(hard_list))

    pool_stats: Dict[str, int] = {
        "filtered_by_capability_count": total_pool_cap_filtered,
        "filtered_by_connectivity_count": total_pool_conn_filtered,
        "skipped_already_tried_count": total_pool_already_tried,
        # ALNS round-level family stats
        "alns_virtual_family_attempt_count": _round_alns_family_attempt,
        "alns_virtual_family_accept_count": _round_alns_family_accept,
        "alns_virtual_family_reject_count": _round_alns_family_reject,
        "alns_virtual_family_budget_block_count": _round_alns_family_budget_block,
        "alns_virtual_family_no_gain_count": 0,
        "alns_virtual_family_neighborhood_histogram": {
            neighborhood.name: _round_alns_family_attempt,
        } if _allow_guarded_family_this_round else {},
        "local_cpsat_virtual_family_selected_count": _round_total_local_family_selected,
        "virtual_family_enabled_for_this_round": int(_allow_guarded_family_this_round),
        "virtual_family_enable_reason": str(_family_enable_reason),
        # ---- Family repair value metrics (guarded profile) ----
        "family_repair_attempt_count": int(tail_repair_diag.get("family_repair_attempt_count", 0)),
        "family_repair_accept_count": int(tail_repair_diag.get("family_repair_accept_count", 0)),
        "family_repair_gain_dropped": int(tail_repair_diag.get("family_repair_gain_dropped", 0)),
        "family_repair_gain_underfilled": int(tail_repair_diag.get("family_repair_gain_underfilled", 0)),
        "family_repair_gain_scheduled_orders": int(tail_repair_diag.get("family_repair_gain_scheduled_orders", 0)),
        "family_repair_no_gain_count": int(tail_repair_diag.get("family_repair_no_gain_count", 0)),
        "family_repair_avg_gain_per_attempt": float(
            tail_repair_diag.get("family_repair_gain_dropped", 0) + tail_repair_diag.get("family_repair_gain_underfilled", 0)
        ) / max(1, tail_repair_diag.get("family_repair_attempt_count", 1)),
        "family_repair_avg_seconds_per_success": float(tail_repair_diag.get("family_repair_avg_seconds_per_success", 0.0)),
    }

    # Propagate round-level family stats to tail_repair_diag for consistency
    tail_repair_diag["virtual_family_enabled_for_this_round"] = _allow_guarded_family_this_round
    tail_repair_diag["virtual_family_enable_reason"] = _family_enable_reason

    return (
        new_segs,
        tentative_dropped,
        True,
        destroy_count,
        repair_count,
        round_hard_protected_drop_count,
        round_lns_repair_rejected_count,
        round_repair_seg_candidates,
        round_repair_pool_candidates,
        round_line_hint_map,
        pool_stats,
        tail_repair_diag,
    )


# ---------------------------------------------------------------------------
# Helper: Normalize line_capability to allowed line set
# ---------------------------------------------------------------------------

def _normalize_allowed_lines(
    raw_cap: object,
    diag_warnings: List[str],
) -> Set[str]:
    """
    Normalize raw line_capability value to a set of allowed line names.

    Mapping (case-insensitive, stripped):
        "dual", "both", "either", "all", ""      -> {"big_roll", "small_roll"}
        "big_only", "big", "big_roll", "large",
        "large_only"                            -> {"big_roll"}
        "small_only", "small", "small_roll"      -> {"small_roll"}
        None / NaN / missing                      -> {"big_roll", "small_roll"}
        other unknown values                     -> {"big_roll", "small_roll"}
                                                  + diagnostics warning

    Args:
        raw_cap: raw value from orders_df["line_capability"]
        diag_warnings: list to append warnings to

    Returns:
        Set of allowed line names (subset of {"big_roll", "small_roll"})
    """
    import math

    # Handle missing / null
    if raw_cap is None:
        return {"big_roll", "small_roll"}
    if isinstance(raw_cap, float) and math.isnan(raw_cap):
        return {"big_roll", "small_roll"}
    if isinstance(raw_cap, str):
        normalized = raw_cap.lower().strip()
    else:
        # fallback: convert to string then normalize
        try:
            normalized = str(raw_cap).lower().strip()
        except Exception:
            diag_warnings.append(
                f"UNKNOWN_LINE_CAPABILITY_TYPE: type={type(raw_cap).__name__}, "
                f"value={raw_cap!r}, defaulting to both lines"
            )
            return {"big_roll", "small_roll"}

    # Empty string
    if normalized == "":
        return {"big_roll", "small_roll"}

    # Multi-capability: allowed on both lines
    if normalized in ("dual", "both", "either", "all"):
        return {"big_roll", "small_roll"}

    # Big-only group
    if normalized in ("big_only", "big", "big_roll", "large", "large_only"):
        return {"big_roll"}

    # Small-only group
    if normalized in ("small_only", "small", "small_roll"):
        return {"small_roll"}

    # Unknown value: be permissive but warn
    diag_warnings.append(
        f"UNKNOWN_LINE_CAPABILITY_VALUE: value={raw_cap!r}, "
        f"defaulting to both lines"
    )
    return {"big_roll", "small_roll"}


# ---------------------------------------------------------------------------
# Helper: NO_FEASIBLE_LINE pre-check
# ---------------------------------------------------------------------------

def _check_no_feasible_line(
    orders_df: pd.DataFrame,
    tpl_df: pd.DataFrame,
) -> Tuple[List[str], Dict]:
    """
    Identify orders that have no feasible production line.

    An order is marked NO_FEASIBLE_LINE when:
    - On EVERY line it is allowed to run on, it has neither any incoming template
      edge NOR any outgoing template edge in the template graph
      (meaning it can never be connected to any chain).
    - If at least one allowed line provides connectivity, it is NOT dropped.

    Returns:
        (no_feasible_oid_list, diagnostics_dict) where diagnostics contains
        line_capability_warnings and normalized_capability_counts.
    """
    if orders_df.empty or tpl_df.empty:
        return [], {
            "no_feasible_line_capability_warnings": [],
            "normalized_capability_counts": {},
            "no_feasible_line_count": 0,
        }

    # Build per-line adjacency sets from template
    out_by_line: Dict[str, Set[str]] = {"big_roll": set(), "small_roll": set()}
    in_by_line: Dict[str, Set[str]] = {"big_roll": set(), "small_roll": set()}

    for _, row in tpl_df.iterrows():
        from_oid = str(row.get("from_order_id", ""))
        to_oid = str(row.get("to_order_id", ""))
        line = str(row.get("line", "big_roll"))
        if line not in out_by_line:
            out_by_line[line] = set()
            in_by_line[line] = set()
        out_by_line[line].add(from_oid)
        in_by_line[line].add(to_oid)

    no_feasible: List[str] = []
    diag_warnings: List[str] = []
    cap_counts: Dict[str, int] = {}
    order_lookup = orders_df.set_index("order_id").to_dict("index")

    for oid, rec in order_lookup.items():
        raw_cap = rec.get("line_capability", None)

        # Normalize capability
        allowed_lines = _normalize_allowed_lines(raw_cap, diag_warnings)

        # Track normalized capability distribution
        cap_key = frozenset(allowed_lines)
        cap_counts[cap_key] = cap_counts.get(cap_key, 0) + 1

        # Check if at least one allowed line gives it template connectivity
        has_connectivity = False
        for line in allowed_lines:
            has_out = oid in out_by_line.get(line, set())
            has_in = oid in in_by_line.get(line, set())
            if has_out or has_in:
                has_connectivity = True
                break

        if not has_connectivity:
            no_feasible.append(oid)

    diagnostics = {
        "no_feasible_line_capability_warnings": list(diag_warnings),
        "normalized_capability_counts": {
            "_".join(sorted(k)): v for k, v in cap_counts.items()
        },
        "no_feasible_line_count": len(no_feasible),
    }

    return no_feasible, diagnostics


# ---------------------------------------------------------------------------
# Helper: Build segments from planned_df (for rolling state)
# ---------------------------------------------------------------------------

def _build_segments_from_planned_df(
    planned_df: pd.DataFrame,
) -> List[CampaignSegment]:
    """Reconstruct CampaignSegment list from planned_df (used for ALNS state)."""
    if planned_df.empty:
        return []

    segments: List[CampaignSegment] = []
    seg_id = 1
    for (line, camp_hint), grp in planned_df.groupby(
        ["line", "campaign_id_hint"], dropna=False
    ):
        grp_sorted = grp.sort_values("campaign_seq_hint")
        oids = grp_sorted["order_id"].tolist()
        total_tons = grp_sorted["tons"].sum()
        segments.append(CampaignSegment(
            line=str(line),
            campaign_local_id=seg_id,
            order_ids=oids,
            total_tons=float(total_tons),
            cut_reason=None,  # type: ignore[arg-type]
            start_order_id=oids[0] if oids else "",
            end_order_id=oids[-1] if oids else "",
            edge_count=len(oids) - 1 if len(oids) > 1 else 0,
            is_valid=True,
        ))
        seg_id += 1
    return segments


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_constructive_lns_master(
    orders_df: pd.DataFrame,
    transition_pack: dict,
    cfg: PlannerConfig,
    random_seed: int = 2027,
) -> ConstructiveLnsResult:
    """
    Run the Constructive + ALNS solver.

    This is the top-level entry point for the new LNS path.

    Flow:
    A. Build constructive sequences (Layer 1)
    B. Cut chains into campaign segments (Layer 2)
    C. Form initial planned / dropped
    D. ALNS iterations (Layer 3 repair + destroy/repair loop)
    E. Output final planned_df / dropped_df / rounds_df

    Args:
        orders_df: Input orders DataFrame
        transition_pack: Dict containing "templates" DataFrame and other metadata
        cfg: PlannerConfig with rule and score parameters
        random_seed: Random seed for reproducibility

    Returns:
        ConstructiveLnsResult with planned_df, dropped_df, rounds_df, diagnostics
    """
    import time
    print(f"[APS][RUN_PATH_FINGERPRINT] CONSTRUCTIVE_LNS_MASTER_V2_20260416A")
    t0 = time.perf_counter()

    rand = random.Random(random_seed)
    tpl_df = _get_templates(transition_pack)

    # ---- Bridge edge policy (unified, defined early, used throughout) ----
    allow_virtual_bridge = bool(
        getattr(cfg.model, "allow_virtual_bridge_edge_in_constructive", False)
    )
    allow_real_bridge = bool(
        getattr(cfg.model, "allow_real_bridge_edge_in_constructive", True)
    )
    bridge_expansion_mode = str(
        getattr(cfg.model, "bridge_expansion_mode", "disabled") or "disabled"
    )

    # ---- Unified result accumulators (initialized early, updated throughout) ----
    diagnostics: Dict[str, Any] = {}
    engine_meta: Dict[str, Any] = {
        "run_path_fingerprint_constructive_lns_master": "CONSTRUCTIVE_LNS_MASTER_V2_20260416A",
    }

    # ---- Step A0: NO_FEASIBLE_LINE pre-check ----
    # Identify orders with no compatible line or no template connectivity.
    # These are removed from the build pool before constructive building.
    no_feasible_oids, no_feas_line_diag = _check_no_feasible_line(orders_df, tpl_df)
    no_feasible_set = set(no_feasible_oids)

    # Filter orders_df to exclude NO_FEASIBLE_LINE orders for the build
    if no_feasible_set:
        orders_for_build = orders_df[
            ~orders_df["order_id"].astype(str).isin(no_feasible_set)
        ].copy()
    else:
        orders_for_build = orders_df

    # ---- Step A: Constructive building ----
    t_build_start = time.perf_counter()
    build_result: ConstructiveBuildResult = build_constructive_sequences(
        orders_for_build, transition_pack, cfg,
    )
    constructive_build_seconds = time.perf_counter() - t_build_start

    # Legacy family/bridge repair pools are removed from the mainline. ALNS
    # repair candidates now come from graph_unused_nodes only.
    repair_family_edges: List = []
    print("[APS][GRAPH_UNUSED_REPAIR] legacy_repair_family_pool_disabled=true, candidate_source=graph_unused_nodes")

    # Track family repair already attempted (for reconstruction dedup)
    family_repair_already_attempted_keys: set = set()

    # ---- Step B: Campaign cutting ----
    # cut_sequences_into_campaigns expects chains_by_line: Dict[str, List[ConstructiveChain]]
    t_cutter_start = time.perf_counter()
    chains_by_line: Dict[str, List] = build_result.chains_by_line
    candidate_graph = transition_pack.get("candidate_graph") if isinstance(transition_pack, dict) else None
    graph_orders_df = transition_pack.get("graph_orders_df") if isinstance(transition_pack, dict) else orders_for_build
    cut_result: CampaignCutResult = cut_sequences_into_campaigns(
        chains_by_line,
        orders_for_build,
        cfg,
        tpl_df,
        candidate_graph=candidate_graph,
        graph_orders_df=graph_orders_df,
    )
    campaign_cutter_seconds = time.perf_counter() - t_cutter_start

    # Collect initial segments
    initial_segments: List[CampaignSegment] = list(cut_result.segments)
    underfilled_segments: List[CampaignSegment] = list(cut_result.underfilled_segments)
    recon_diag = {
        "underfilled_reconstruction_enabled": False,
        "underfilled_reconstruction_not_entered_reason": "REMOVED_REPAIR_BRIDGE_POOL_MAINLINE",
        "reconstruction_no_gain": True,
        "candidate_source": "graph_unused_nodes",
    }
    if isinstance(cut_result.diagnostics, dict):
        cut_result.diagnostics.update({"underfilled_reconstruction": dict(recon_diag), **dict(recon_diag)})
    print("[APS][UNDERFILLED_RECON] removed_from_mainline=true, candidate_source=graph_unused_nodes")
    if isinstance(cut_result.diagnostics, dict):
        cut_result.diagnostics["total_valid_segments"] = int(len(initial_segments))
        cut_result.diagnostics["total_underfilled_segments"] = int(len(underfilled_segments))
        cut_result.diagnostics["underfilled_segments_after_reconstruction"] = int(len(underfilled_segments))

    # ---- Step C: Form initial planned / dropped ----
    # Planned = valid segments
    planned_segs: List[CampaignSegment] = list(initial_segments)

    # Dropped: underfilled + dead islands from build
    # NOTE: build_result.dropped_seed_orders may include orders that WERE placed
    # on the first processed line (big_roll) but appeared as dead islands on the
    # second line (small_roll). We must exclude orders that appear in any chain.
    all_placed_oids: Set[str] = set()
    for seg in initial_segments:
        all_placed_oids.update(seg.order_ids)
    # Also collect from underfilled (they are dropped, not placed)
    for seg in underfilled_segments:
        pass  # not placed

    # Build true dropped set
    dropped_by_reason: Dict[str, List[str]] = {}

    # NO_FEASIBLE_LINE
    for oid in no_feasible_oids:
        dropped_by_reason.setdefault(DropReason.NO_FEASIBLE_LINE.value, []).append(oid)

    # TAIL_UNDERFILLED
    for seg in underfilled_segments:
        dropped_by_reason.setdefault(DropReason.TAIL_UNDERFILLED.value, []).extend(
            seg.order_ids
        )

    # DEAD_ISLAND_ORDER: only orders that are truly not in any chain
    for dead_order in cut_result.dropped_orders:
        oid = str(dead_order.get("order_id", ""))
        if oid and oid not in all_placed_oids:
            dropped_by_reason.setdefault(DropReason.DEAD_ISLAND_ORDER.value, []).append(oid)

    # CONSTRUCTIVE_REJECTED: orders flagged as dead on secondary lines
    # (but NOT already placed on primary line — those are false positives from builder)
    for dead_order in build_result.dropped_seed_orders:
        oid = str(dead_order.get("order_id", ""))
        if oid and oid not in all_placed_oids:
            if oid not in dropped_by_reason.get(DropReason.DEAD_ISLAND_ORDER.value, []):
                dropped_by_reason.setdefault(DropReason.CONSTRUCTIVE_REJECTED.value, []).append(oid)

    # Recalculate tons for planned segments
    planned_segs = _recalculate_segment_tons(planned_segs, orders_df)

    # Build constructive phase line_hint_map:
    # - TAIL_UNDERFILLED orders: use the segment's line
    # - NO_FEASIBLE_LINE / DEAD_ISLAND / CONSTRUCTIVE_REJECTED: infer from line_capability
    constructive_line_hint_map: Dict[str, str] = {}
    order_lookup_all: Dict = {}
    if not orders_df.empty:
        order_lookup_all = orders_df.set_index("order_id").to_dict("index")
    for oid, rec in order_lookup_all.items():
        raw_cap = rec.get("line_capability", None)
        diag_bucket: List[str] = []
        allowed = _normalize_allowed_lines(raw_cap, diag_bucket)
        if allowed == {"big_roll"}:
            constructive_line_hint_map[oid] = "big_roll"
        elif allowed == {"small_roll"}:
            constructive_line_hint_map[oid] = "small_roll"
        else:
            constructive_line_hint_map[oid] = ""
    # Override with segment lines for TAIL_UNDERFILLED
    for seg in underfilled_segments:
        for oid in seg.order_ids:
            constructive_line_hint_map[oid] = seg.line

    # ---- Build initial planned_df for comparison ----
    initial_planned_df, init_slot_diag = _segments_to_planned_df(
        planned_segs, orders_for_build, tpl_df, cfg=cfg
    )
    initial_dropped_df = _dropped_to_df(
        dropped_by_reason,
        orders_df,          # Use FULL orders_df so NO_FEASIBLE_LINE orders have metadata
        stage="constructive",
        round_hint=0,
        line_hint_map=constructive_line_hint_map,
    )

    # Initial snapshot
    best_snap = _build_snapshot_from_dfs(
        initial_planned_df, initial_dropped_df, orders_df, tpl_df,
    )
    current_segs = list(planned_segs)
    current_dropped: Dict[str, List[str]] = {
        k: list(v) for k, v in dropped_by_reason.items()
    }

    # ---- Step D: ALNS iterations ----
    n_rounds = getattr(cfg.model, "rounds", 8) or 8
    # LNS Early Stop Configuration (runtime compression)
    lns_early_stop_no_improve_rounds = getattr(cfg.model, "lns_early_stop_no_improve_rounds", 3) or 3
    lns_max_total_rounds = getattr(cfg.model, "lns_max_total_rounds", 10) or 10
    lns_min_rounds_before_early_stop = getattr(cfg.model, "lns_min_rounds_before_early_stop", 4) or 4
    # Clamp n_rounds to lns_max_total_rounds
    n_rounds = min(n_rounds, lns_max_total_rounds)
    # Determine effective neighborhood pool based on edge policy
    # In direct_only mode (allow_virtual=False AND allow_real=False),
    # HIGH_VIRTUAL_USAGE has no business meaning and is excluded.
    is_direct_only = not allow_virtual_bridge and not allow_real_bridge
    base_neighborhoods: List[NeighborhoodType]
    # Current production neighborhoods only (HIGH_VIRTUAL_USAGE and SMALL_ROLL_RESCUE removed)
    base_neighborhoods = [
        NeighborhoodType.LOW_FILL_SEGMENT,
        NeighborhoodType.HIGH_DROP_PRESSURE,
        NeighborhoodType.TAIL_REBALANCE,
    ]
    print(
        f"[APS][constructive_lns] neighborhood_pool={[n.value for n in base_neighborhoods]}"
    )

    rounds_records: List[LnsRound] = []
    no_improve_streak = 0
    max_no_improve = lns_early_stop_no_improve_rounds
    # Early stop tracking
    lns_early_stop_triggered = False
    lns_early_stop_reason = ""
    lns_rounds_executed = 0
    # Track best values for improvement detection
    best_dropped_count_at_streak_start = None
    best_scheduled_tons_at_streak_start = None
    # Print LNS control configuration
    print(
        f"[APS][LNS_CONTROL] rounds_configured={n_rounds}, "
        f"time_limit_seconds={getattr(cfg.model, 'time_limit_seconds', 60.0)}, "
        f"early_stop_no_improve_rounds={lns_early_stop_no_improve_rounds}, "
        f"max_total_rounds={lns_max_total_rounds}, "
        f"min_rounds_before_early_stop={lns_min_rounds_before_early_stop}"
    )
    # Accumulates line_hint per order across all LNS rounds (later round wins)
    lns_line_hint_map: Dict[str, str] = {}
    # Accumulates pool diagnostics across all LNS rounds
    accum_pool_stats: Dict[str, int] = {
        "filtered_by_capability_count": 0,
        "filtered_by_connectivity_count": 0,
        "skipped_already_tried_count": 0,
    }
    # Neighborhood selection counters
    accum_tail_repair_diag: dict = {
        "tail_rebalance_lns_success_count": 0,
        "tail_rebalance_lns_shifted_orders": 0,
        "tail_rebalance_lns_inserted_dropped_orders": 0,
        "tail_rebalance_lns_merge_count": 0,
        "tail_rebalance_lns_recut_count": 0,
        "tail_rebalance_lns_dropped_reinsertion_attempts": 0,
        "tail_rebalance_lns_dropped_reinsertion_success": 0,
        "local_inserter_direct_arcs_allowed": 0,
        "local_inserter_real_bridge_arcs_allowed": 0,
        "local_inserter_real_bridge_arcs_blocked": 0,
        "local_inserter_virtual_bridge_arcs_blocked": 0,
        "local_inserter_edge_policy_used": "",
        "low_fill_candidates_total": 0,
        # ---- Local CP-SAT gate diagnostics (must be here for accumulation) ----
        "local_cpsat_skipped_due_to_gate": 0,
        "local_cpsat_attempt_count": 0,
        "local_cpsat_success_count": 0,
        "local_cpsat_no_improve_count": 0,
        "local_cpsat_infeasible_count": 0,
        "local_cpsat_total_seconds": 0.0,
        # ---- Family repair pool diagnostics ----
        "repair_family_edge_pool_seen_count": 0,
        "repair_family_edge_pool_filtered_count": 0,
        "repair_family_edge_pool_used_count": 0,
        # ---- Hotspot destroy diagnostics ----
        "destroy_hotspot_type": "",
        "destroy_hotspot_center": "",
        "destroy_window_size": 0,
        "destroy_hotspot_candidates_count": 0,
        "destroy_random_backfill_count": 0,
        "drop_pressure_score": 0.0,
        "drop_pressure_nearby_dropped_count": 0,
        "drop_pressure_recoverable_tons_estimate": 0.0,
        "active_neighborhoods_this_run": "",
        "hotspot_neighborhood_enable_reason": "",
    }

    # ---- Phase timers for LNS ----
    t_lns_start = time.perf_counter()

    for r in range(n_rounds):
        rt0 = time.perf_counter()

        # Dynamic neighborhood frequency: boost LOW_FILL / TAIL_REBALANCE when underfilled tails are many
        ton_min_local = float(cfg.rule.campaign_ton_min)
        underfilled_segs = [
            s for s in current_segs
            if not s.is_valid and s.total_tons < ton_min_local
        ]
        underfilled_count = len(underfilled_segs)

        if underfilled_count >= 10:
            # When many underfilled tails, give them 50% of slots
            # Insert extra LOW_FILL / TAIL_REBALANCE between base neighborhoods
            boost_pool: List[NeighborhoodType] = []
            for nb in base_neighborhoods:
                boost_pool.append(nb)
                if nb in (NeighborhoodType.LOW_FILL_SEGMENT, NeighborhoodType.TAIL_REBALANCE):
                    boost_pool.append(nb)  # Double weight
            neighborhoods = boost_pool
        elif underfilled_count >= 3:
            # Moderate: add 1 extra slot
            neighborhoods = []
            for nb in base_neighborhoods:
                neighborhoods.append(nb)
                if nb == NeighborhoodType.TAIL_REBALANCE:
                    neighborhoods.append(nb)
            # Add LOW_FILL if not already present
            if NeighborhoodType.LOW_FILL_SEGMENT not in neighborhoods:
                neighborhoods.insert(0, NeighborhoodType.LOW_FILL_SEGMENT)
        else:
            neighborhoods = list(base_neighborhoods)

        # ---- Hotspot-driven neighborhood activation ----
        # Detect business signals and conditionally add hotspot neighborhoods
        # to the effective pool for this round.
        _profile_name = str(getattr(cfg.model, "profile_name", "") or "")
        _is_guarded_profile = (
            getattr(cfg.model, "virtual_family_frontload_enabled", False) is True
            or "guarded" in _profile_name.lower()
        )
        _total_dropped = sum(len(v) for v in current_dropped.values())
        # Compute width_range and group_switch_count across all segments for this round
        _width_range_max = 0.0
        _group_switch_count = 0
        _bridge_seg_count = 0
        for _seg in current_segs:
            if not _seg.is_valid or len(_seg.order_ids) < 2:
                continue
            if _seg.total_tons < ton_min_local:
                _bridge_seg_count += 1
            _seg_recs = {}
            for _oid in _seg.order_ids:
                if _oid in orders_df.set_index("order_id").index:
                    _seg_recs[_oid] = dict(orders_df.set_index("order_id").loc[_oid])
            _widths = [float(_r.get("width", 0) or 0) for _r in _seg_recs.values()]
            if len(_widths) >= 2:
                _wr = max(_widths) - min(_widths)
                if _wr > _width_range_max:
                    _width_range_max = _wr
            _groups = [str(_r.get("steel_group", "") or "") for _r in _seg_recs.values()]
            for _i in range(len(_groups) - 1):
                if _groups[_i] != _groups[_i + 1]:
                    _group_switch_count += 1

        # Thresholds for enabling hotspot neighborhoods
        _DROP_THRESHOLD = 5
        _WIDTH_RANGE_THRESHOLD = 200.0
        _GROUP_SWITCH_THRESHOLD = 2
        _BRIDGE_THRESHOLD = 1
        _ENABLE_HOTSPOT = (
            _is_guarded_profile
            or _total_dropped > _DROP_THRESHOLD
            or underfilled_count >= 3
        )

        _hotspot_added = False
        if _ENABLE_HOTSPOT:
            # WIDTH_TENSION_HOTSPOT: width range signal
            if _width_range_max > _WIDTH_RANGE_THRESHOLD or _total_dropped > _DROP_THRESHOLD:
                if NeighborhoodType.WIDTH_TENSION_HOTSPOT not in neighborhoods:
                    neighborhoods.append(NeighborhoodType.WIDTH_TENSION_HOTSPOT)
                    _hotspot_added = True
            # GROUP_SWITCH_HOTSPOT: group transition signal
            if _group_switch_count >= _GROUP_SWITCH_THRESHOLD or _is_guarded_profile:
                if NeighborhoodType.GROUP_SWITCH_HOTSPOT not in neighborhoods:
                    neighborhoods.append(NeighborhoodType.GROUP_SWITCH_HOTSPOT)
                    _hotspot_added = True
            # BRIDGE_DEPENDENT_SEGMENT: underfilled segments with bridge usage
            if _bridge_seg_count >= _BRIDGE_THRESHOLD or _is_guarded_profile:
                if NeighborhoodType.BRIDGE_DEPENDENT_SEGMENT not in neighborhoods:
                    neighborhoods.append(NeighborhoodType.BRIDGE_DEPENDENT_SEGMENT)
                    _hotspot_added = True

        neighborhood = neighborhoods[r % len(neighborhoods)]

        # Record active neighborhoods and hotspot activation reason for diagnostics
        _nb_names = [str(nb.value) for nb in neighborhoods]
        _hotspot_reason = ""
        if _hotspot_added:
            _reasons = []
            if _is_guarded_profile:
                _reasons.append("guarded_profile")
            if _total_dropped > _DROP_THRESHOLD:
                _reasons.append(f"dropped>{_DROP_THRESHOLD}")
            if underfilled_count >= 3:
                _reasons.append("underfilled>=3")
            if _width_range_max > _WIDTH_RANGE_THRESHOLD:
                _reasons.append(f"width_range>{_WIDTH_RANGE_THRESHOLD}")
            if _group_switch_count >= _GROUP_SWITCH_THRESHOLD:
                _reasons.append(f"group_switch>={_GROUP_SWITCH_THRESHOLD}")
            if _bridge_seg_count >= _BRIDGE_THRESHOLD:
                _reasons.append(f"bridge_seg>={_BRIDGE_THRESHOLD}")
            _hotspot_reason = ",".join(_reasons)
        # Accumulate per-round diagnostics; the first round seeds the values
        if r == 0:
            accum_tail_repair_diag["active_neighborhoods_this_run"] = ",".join(_nb_names)
            accum_tail_repair_diag["hotspot_neighborhood_enable_reason"] = _hotspot_reason

        # Run destruction + repair
        (
            new_segs,
            new_dropped,
            did_work,
            destroy_count,
            repair_count,
            hard_prot_drop_cnt,
            lns_rej_cnt,
            seg_cand_cnt,
            pool_cand_cnt,
            round_line_hint_map,
            round_pool_stats,
            round_tail_diag,
        ) = _run_alns_iteration(
            current_segs=current_segs,
            current_dropped=current_dropped,
            orders_df=orders_for_build,
            tpl_df=tpl_df,
            transition_pack=transition_pack,
            cfg=cfg,
            neighborhood=neighborhood,
            round_num=r + 1,
            rand=rand,
            max_destroy_orders=45,
            time_limit=8.0,
            underfilled_segs=underfilled_segs,
            repair_family_edges=repair_family_edges,
            family_repair_already_attempted_keys=family_repair_already_attempted_keys,
        )

        # After iteration: add round's repair family edge keys to global dedup set
        # (applies only when family was enabled this round)
        _returned_family_keys = round_tail_diag.get("_round_repair_family_edge_keys", set())
        if _returned_family_keys:
            family_repair_already_attempted_keys.update(_returned_family_keys)

        # Accumulate tail repair diagnostics
        for k, v in round_tail_diag.items():
            if k in accum_tail_repair_diag and isinstance(v, (int, float)):
                accum_tail_repair_diag[k] += v
            elif k == "local_inserter_edge_policy_used" and v:
                accum_tail_repair_diag[k] = str(v)
            elif k == "low_fill_candidates":
                accum_tail_repair_diag["low_fill_candidates_total"] += v

        rt1 = time.perf_counter()
        elapsed = rt1 - rt0

        if not did_work:
            # NOTE: SMALL_ROLL_RESCUE removed (experimental, not in production)
            rounds_records.append(LnsRound(
                round=r + 1,
                neighborhood_type=neighborhood,
                accepted=False,
                scheduled_order_count=best_snap.order_count(),
                scheduled_tons=best_snap.total_tons,
                dropped_count=len(set(oid for v in current_dropped.values() for oid in v)),
                hard_constraint_protected_drop_count=0,
                lns_repair_rejected_count=0,
                repair_segment_candidate_count=0,
                repair_pool_candidate_count=0,
                repair_pool_filtered_by_capability_count=0,
                repair_pool_filtered_by_connectivity_count=0,
                repair_pool_skipped_as_already_tried_count=0,
                elapsed_seconds=elapsed,
                notes="No neighborhood available",
                target_fill_ratio=0.0,
                target_tail_tons=0.0,
                target_prev_segment_tons=0.0,
                tail_rebalance_lns_success_count=round_tail_diag.get("tail_rebalance_lns_success_count", 0),
                tail_rebalance_lns_shifted_orders=round_tail_diag.get("tail_rebalance_lns_shifted_orders", 0),
                tail_rebalance_lns_inserted_dropped_orders=round_tail_diag.get("tail_rebalance_lns_inserted_dropped_orders", 0),
                tail_rebalance_lns_merge_count=round_tail_diag.get("tail_rebalance_lns_merge_count", 0),
                tail_rebalance_lns_recut_count=round_tail_diag.get("tail_rebalance_lns_recut_count", 0),
                tail_rebalance_lns_fail_reasons=round_tail_diag.get("tail_rebalance_lns_fail_reasons", []),
                tail_rebalance_lns_dropped_reinsertion_attempts=round_tail_diag.get("tail_rebalance_lns_dropped_reinsertion_attempts", 0),
                tail_rebalance_lns_dropped_reinsertion_success=round_tail_diag.get("tail_rebalance_lns_dropped_reinsertion_success", 0),
                low_fill_candidates=round_tail_diag.get("low_fill_candidates", 0),
                low_fill_avg_gap_to_min=round_tail_diag.get("low_fill_avg_gap_to_min", 0.0),
                low_fill_selected_gap=round_tail_diag.get("low_fill_selected_gap", 0.0),
                low_fill_neighborhood_selected_count=0,
                tail_rebalance_neighborhood_selected_count=0,
                low_fill_neighborhood_success_count=0,
                tail_rebalance_neighborhood_success_count=0,
                virtual_family_frontload_enabled=getattr(cfg.model, "virtual_family_frontload_enabled", False),
                alns_virtual_family_attempt_count=round_pool_stats.get("alns_virtual_family_attempt_count", 0),
                alns_virtual_family_accept_count=round_pool_stats.get("alns_virtual_family_accept_count", 0),
                alns_virtual_family_reject_count=round_pool_stats.get("alns_virtual_family_reject_count", 0),
                alns_virtual_family_budget_block_count=round_pool_stats.get("alns_virtual_family_budget_block_count", 0),
                alns_virtual_family_no_gain_count=round_pool_stats.get("alns_virtual_family_no_gain_count", 0),
                alns_virtual_family_neighborhood_histogram=round_pool_stats.get("alns_virtual_family_neighborhood_histogram", {}),
                virtual_family_enabled_for_this_round=bool(round_pool_stats.get("virtual_family_enabled_for_this_round", 0)),
                virtual_family_enable_reason=str(round_pool_stats.get("virtual_family_enable_reason", "")),
                virtual_family_disable_reason="no_neighborhood_available" if not round_pool_stats.get("virtual_family_enabled_for_this_round", 0) else "",
                local_cpsat_virtual_family_selected_count=int(round_pool_stats.get("local_cpsat_virtual_family_selected_count", 0)),
                # ---- Family repair value metrics (guarded profile) ----
                family_repair_attempt_count=int(round_pool_stats.get("family_repair_attempt_count", 0)),
                family_repair_accept_count=int(round_pool_stats.get("family_repair_accept_count", 0)),
                family_repair_gain_dropped=int(round_pool_stats.get("family_repair_gain_dropped", 0)),
                family_repair_gain_underfilled=int(round_pool_stats.get("family_repair_gain_underfilled", 0)),
                family_repair_gain_scheduled_orders=int(round_pool_stats.get("family_repair_gain_scheduled_orders", 0)),
                family_repair_no_gain_count=int(round_pool_stats.get("family_repair_no_gain_count", 0)),
                family_repair_avg_gain_per_attempt=float(round_pool_stats.get("family_repair_avg_gain_per_attempt", 0.0)),
                family_repair_avg_seconds_per_success=float(round_pool_stats.get("family_repair_avg_seconds_per_success", 0.0)),
            ))
            no_improve_streak += 1
            continue

        # Build candidate snapshot
        cand_planned_df, _cand_slot_diag = _segments_to_planned_df(
            new_segs, orders_for_build, tpl_df, cfg=cfg
        )
        # Accumulate line hints: this round's map overrides previous rounds
        lns_line_hint_map.update(round_line_hint_map)
        # Accumulate pool filtering diagnostics
        accum_pool_stats["filtered_by_capability_count"] += round_pool_stats.get(
            "filtered_by_capability_count", 0)
        accum_pool_stats["filtered_by_connectivity_count"] += round_pool_stats.get(
            "filtered_by_connectivity_count", 0)
        accum_pool_stats["skipped_already_tried_count"] += round_pool_stats.get(
            "skipped_already_tried_count", 0)
        cand_dropped_df = _dropped_to_df(
            new_dropped,
            orders_for_build,
            stage=f"lns_round_{r + 1}",
            round_hint=r + 1,
            line_hint_map=round_line_hint_map,
        )
        cand_snap = _build_snapshot_from_dfs(
            cand_planned_df, cand_dropped_df, orders_df, tpl_df,
        )

        # Compare
        accepted = _is_strictly_better(cand_snap, best_snap)
        current_dropped_count = len(set(oid for v in new_dropped.values() for oid in v))
        current_scheduled_tons = cand_snap.total_tons
        if accepted:
            best_snap = cand_snap
            current_segs = new_segs
            current_dropped = new_dropped
            no_improve_streak = 0
            # Reset streak start tracking on successful improvement
            best_dropped_count_at_streak_start = None
            best_scheduled_tons_at_streak_start = None
        else:
            # Record best values at the START of the no-improve streak
            if no_improve_streak == 0:
                best_dropped_count_at_streak_start = len(set(oid for v in current_dropped.values() for oid in v))
                best_scheduled_tons_at_streak_start = best_snap.total_tons
            no_improve_streak += 1

        # ---- TAIL_REBALANCE diagnostics: compute fill_ratio for selected segments ----
        # Re-run selection logic to record what was targeted this round
        tail_target_fill = 0.0
        tail_target_tons = 0.0
        tail_target_prev_tons = 0.0
        # NOTE: SMALL_ROLL_RESCUE removed (experimental, not in production)
        if neighborhood == NeighborhoodType.TAIL_REBALANCE and current_segs:
            # Pick the same segment that _select_neighborhood would have picked
            # (ascending |fill_ratio-1|, underfilled segments score 0.0)
            valid_segs = [s for s in current_segs if s.is_valid]
            if valid_segs:
                ton_min_local = float(cfg.rule.campaign_ton_min)
                scored: List[Tuple[float, CampaignSegment]] = []
                for seg in valid_segs:
                    if seg.total_tons >= ton_min_local:
                        fr = seg.total_tons / ton_min_local if ton_min_local > 0 else 0.0
                        score = abs(fr - 1.0)
                    else:
                        score = 0.0  # underfilled: highest priority
                    scored.append((score, seg))
                scored.sort(key=lambda x: x[0])
                if scored:
                    top_seg = scored[0][1]
                    ton_min_local2 = float(cfg.rule.campaign_ton_min)
                    tail_target_tons = round(top_seg.total_tons / 10.0, 1)  # tons10
                    tail_target_fill = round(
                        top_seg.total_tons / ton_min_local2, 2
                    ) if ton_min_local2 > 0 else 0.0
                    # Find previous segment on same line for pullback context
                    seg_idx = current_segs.index(top_seg)
                    for prev_idx in range(seg_idx - 1, -1, -1):
                        if current_segs[prev_idx].line == top_seg.line:
                            tail_target_prev_tons = round(
                                current_segs[prev_idx].total_tons / 10.0, 1
                            )
                            break

        rounds_records.append(LnsRound(
            round=r + 1,
            neighborhood_type=neighborhood,
            accepted=accepted,
            scheduled_order_count=cand_snap.order_count(),
            scheduled_tons=cand_snap.total_tons,
            dropped_count=len(set(oid for v in new_dropped.values() for oid in v)),
            hard_constraint_protected_drop_count=hard_prot_drop_cnt,
            lns_repair_rejected_count=lns_rej_cnt,
            repair_segment_candidate_count=seg_cand_cnt,
            repair_pool_candidate_count=pool_cand_cnt,
            repair_pool_filtered_by_capability_count=round_pool_stats.get(
                "filtered_by_capability_count", 0),
            repair_pool_filtered_by_connectivity_count=round_pool_stats.get(
                "filtered_by_connectivity_count", 0),
            repair_pool_skipped_as_already_tried_count=round_pool_stats.get(
                "skipped_already_tried_count", 0),
            virtual_bridge_count=cand_snap.virtual_bridge_count,
            template_cost=cand_snap.template_cost,
            destroy_count=destroy_count,
            repair_count=repair_count,
            elapsed_seconds=elapsed,
            notes="Improved" if accepted else "No improvement",
            target_fill_ratio=tail_target_fill,
            target_tail_tons=tail_target_tons,
            target_prev_segment_tons=tail_target_prev_tons,
            tail_rebalance_lns_success_count=round_tail_diag.get("tail_rebalance_lns_success_count", 0),
            tail_rebalance_lns_shifted_orders=round_tail_diag.get("tail_rebalance_lns_shifted_orders", 0),
            tail_rebalance_lns_inserted_dropped_orders=round_tail_diag.get("tail_rebalance_lns_inserted_dropped_orders", 0),
            tail_rebalance_lns_merge_count=round_tail_diag.get("tail_rebalance_lns_merge_count", 0),
            tail_rebalance_lns_recut_count=round_tail_diag.get("tail_rebalance_lns_recut_count", 0),
            tail_rebalance_lns_fail_reasons=round_tail_diag.get("tail_rebalance_lns_fail_reasons", []),
            tail_rebalance_lns_dropped_reinsertion_attempts=round_tail_diag.get("tail_rebalance_lns_dropped_reinsertion_attempts", 0),
            tail_rebalance_lns_dropped_reinsertion_success=round_tail_diag.get("tail_rebalance_lns_dropped_reinsertion_success", 0),
            low_fill_candidates=round_tail_diag.get("low_fill_candidates", 0),
            low_fill_avg_gap_to_min=round_tail_diag.get("low_fill_avg_gap_to_min", 0.0),
            low_fill_selected_gap=round_tail_diag.get("low_fill_selected_gap", 0.0),
            low_fill_neighborhood_selected_count=(
                1 if neighborhood == NeighborhoodType.LOW_FILL_SEGMENT else 0
            ),
            tail_rebalance_neighborhood_selected_count=(
                1 if neighborhood == NeighborhoodType.TAIL_REBALANCE else 0
            ),
            low_fill_neighborhood_success_count=(
                1 if (neighborhood == NeighborhoodType.LOW_FILL_SEGMENT and accepted) else 0
            ),
            tail_rebalance_neighborhood_success_count=(
                1 if (neighborhood == NeighborhoodType.TAIL_REBALANCE and accepted) else 0
            ),
            virtual_family_frontload_enabled=getattr(cfg.model, "virtual_family_frontload_enabled", False),
            alns_virtual_family_attempt_count=round_pool_stats.get("alns_virtual_family_attempt_count", 0),
            alns_virtual_family_accept_count=round_pool_stats.get("alns_virtual_family_accept_count", 0),
            alns_virtual_family_reject_count=round_pool_stats.get("alns_virtual_family_reject_count", 0),
            alns_virtual_family_budget_block_count=round_pool_stats.get("alns_virtual_family_budget_block_count", 0),
            alns_virtual_family_no_gain_count=round_pool_stats.get("alns_virtual_family_no_gain_count", 0),
            alns_virtual_family_neighborhood_histogram=round_pool_stats.get("alns_virtual_family_neighborhood_histogram", {}),
            virtual_family_enabled_for_this_round=bool(round_pool_stats.get("virtual_family_enabled_for_this_round", 0)),
            virtual_family_enable_reason=str(round_pool_stats.get("virtual_family_enable_reason", "")),
            local_cpsat_virtual_family_selected_count=int(round_pool_stats.get("local_cpsat_virtual_family_selected_count", 0)),
            # ---- Family repair value metrics (guarded profile) ----
            family_repair_attempt_count=int(round_pool_stats.get("family_repair_attempt_count", 0)),
            family_repair_accept_count=int(round_pool_stats.get("family_repair_accept_count", 0)),
            family_repair_gain_dropped=int(round_pool_stats.get("family_repair_gain_dropped", 0)),
            family_repair_gain_underfilled=int(round_pool_stats.get("family_repair_gain_underfilled", 0)),
            family_repair_gain_scheduled_orders=int(round_pool_stats.get("family_repair_gain_scheduled_orders", 0)),
            family_repair_no_gain_count=int(round_pool_stats.get("family_repair_no_gain_count", 0)),
            family_repair_avg_gain_per_attempt=float(round_pool_stats.get("family_repair_avg_gain_per_attempt", 0.0)),
            family_repair_avg_seconds_per_success=float(round_pool_stats.get("family_repair_avg_seconds_per_success", 0.0)),
        ))

        if no_improve_streak >= max_no_improve:
            rounds_records.append(LnsRound(
                round=r + 2,
                neighborhood_type=neighborhood,
                accepted=False,
                scheduled_order_count=best_snap.order_count(),
                scheduled_tons=best_snap.total_tons,
                dropped_count=len(set(oid for v in current_dropped.values() for oid in v)),
                hard_constraint_protected_drop_count=hard_prot_drop_cnt,
                lns_repair_rejected_count=lns_rej_cnt,
                repair_segment_candidate_count=seg_cand_cnt,
                repair_pool_candidate_count=pool_cand_cnt,
                repair_pool_filtered_by_capability_count=round_pool_stats.get(
                    "filtered_by_capability_count", 0),
                repair_pool_filtered_by_connectivity_count=round_pool_stats.get(
                    "filtered_by_connectivity_count", 0),
                repair_pool_skipped_as_already_tried_count=round_pool_stats.get(
                    "skipped_already_tried_count", 0),
                elapsed_seconds=0.0,
                notes=f"Early stop: {max_no_improve} rounds without improvement",
                target_fill_ratio=0.0,
                target_tail_tons=0.0,
                target_prev_segment_tons=0.0,
                tail_rebalance_lns_success_count=round_tail_diag.get("tail_rebalance_lns_success_count", 0),
                tail_rebalance_lns_shifted_orders=round_tail_diag.get("tail_rebalance_lns_shifted_orders", 0),
                tail_rebalance_lns_inserted_dropped_orders=round_tail_diag.get("tail_rebalance_lns_inserted_dropped_orders", 0),
                tail_rebalance_lns_merge_count=round_tail_diag.get("tail_rebalance_lns_merge_count", 0),
                tail_rebalance_lns_recut_count=round_tail_diag.get("tail_rebalance_lns_recut_count", 0),
                tail_rebalance_lns_fail_reasons=round_tail_diag.get("tail_rebalance_lns_fail_reasons", []),
                tail_rebalance_lns_dropped_reinsertion_attempts=round_tail_diag.get("tail_rebalance_lns_dropped_reinsertion_attempts", 0),
                tail_rebalance_lns_dropped_reinsertion_success=round_tail_diag.get("tail_rebalance_lns_dropped_reinsertion_success", 0),
                low_fill_candidates=round_tail_diag.get("low_fill_candidates", 0),
                low_fill_avg_gap_to_min=round_tail_diag.get("low_fill_avg_gap_to_min", 0.0),
                low_fill_selected_gap=round_tail_diag.get("low_fill_selected_gap", 0.0),
                low_fill_neighborhood_selected_count=0,
                tail_rebalance_neighborhood_selected_count=0,
                low_fill_neighborhood_success_count=0,
                tail_rebalance_neighborhood_success_count=0,
                virtual_family_frontload_enabled=getattr(cfg.model, "virtual_family_frontload_enabled", False),
                alns_virtual_family_attempt_count=round_pool_stats.get("alns_virtual_family_attempt_count", 0),
                alns_virtual_family_accept_count=round_pool_stats.get("alns_virtual_family_accept_count", 0),
                alns_virtual_family_reject_count=round_pool_stats.get("alns_virtual_family_reject_count", 0),
                alns_virtual_family_budget_block_count=round_pool_stats.get("alns_virtual_family_budget_block_count", 0),
                alns_virtual_family_no_gain_count=round_pool_stats.get("alns_virtual_family_no_gain_count", 0),
                alns_virtual_family_neighborhood_histogram=round_pool_stats.get("alns_virtual_family_neighborhood_histogram", {}),
                virtual_family_enabled_for_this_round=bool(round_pool_stats.get("virtual_family_enabled_for_this_round", 0)),
                virtual_family_enable_reason=str(round_pool_stats.get("virtual_family_enable_reason", "")),
                local_cpsat_virtual_family_selected_count=int(round_pool_stats.get("local_cpsat_virtual_family_selected_count", 0)),
                # ---- Family repair value metrics (guarded profile) ----
                family_repair_attempt_count=int(round_pool_stats.get("family_repair_attempt_count", 0)),
                family_repair_accept_count=int(round_pool_stats.get("family_repair_accept_count", 0)),
                family_repair_gain_dropped=int(round_pool_stats.get("family_repair_gain_dropped", 0)),
                family_repair_gain_underfilled=int(round_pool_stats.get("family_repair_gain_underfilled", 0)),
                family_repair_gain_scheduled_orders=int(round_pool_stats.get("family_repair_gain_scheduled_orders", 0)),
                family_repair_no_gain_count=int(round_pool_stats.get("family_repair_no_gain_count", 0)),
                family_repair_avg_gain_per_attempt=float(round_pool_stats.get("family_repair_avg_gain_per_attempt", 0.0)),
                family_repair_avg_seconds_per_success=float(round_pool_stats.get("family_repair_avg_seconds_per_success", 0.0)),
            ))
            break

    # ---- Step E0: Final segment template pair validation + salvage demotion ----
    # Validate that all adjacent order pairs in final segments exist in the template.
    # For segments with template miss: try to salvage clean sub-segments by splitting at first offending pair.
    # Only truly invalid fragments are demoted to dropped pool.
    final_segment_template_miss_count = 0
    final_segment_template_miss_examples: List[dict] = []
    final_segment_template_miss_campaigns: List[str] = []
    final_segment_template_miss_orders: List[str] = []

    # Segments that passed the template check (clean output candidates)
    clean_final_segs: List[CampaignSegment] = []
    # Fragments that were demoted from miss segments (per-fragment, not whole segment)
    final_demoted_fragments: List[dict] = []

    # Salvage diagnostics
    final_segment_salvage_attempts = 0
    final_segment_salvaged_piece_count = 0
    final_segment_demoted_fragment_count = 0
    final_segment_salvage_success_count = 0
    final_segment_full_drop_count = 0
    final_segment_dirty_window_sizes: List[int] = []
    final_segment_kept_left_orders: List[int] = []
    final_segment_kept_right_orders: List[int] = []

    # Get campaign tonnage bounds from cfg
    ton_min = float(cfg.rule.campaign_ton_min)
    ton_max = float(cfg.rule.campaign_ton_max)

    if tpl_df is not None and not tpl_df.empty and current_segs:
        for seg in current_segs:
            if not seg.is_valid:
                # Already invalid segments: keep them out of planned output
                continue
            if len(seg.order_ids) < 2:
                # Single-order segment: no adjacent pair → trivially clean
                clean_final_segs.append(seg)
                continue

            all_ok, miss_examples = _validate_segment_template_pairs(seg, tpl_df)
            if not all_ok:
                # Try to salvage clean sub-segments from this miss segment
                final_segment_salvage_attempts += 1
                salvaged_segs, demoted_frags, salvage_stats = _salvage_segment_with_template_miss(
                    seg=seg,
                    miss_examples=miss_examples,
                    tpl_df=tpl_df,
                    ton_min=ton_min,
                    ton_max=ton_max,
                    max_expand_steps=3,
                )

                miss_campaign_id = f"{seg.line}_{seg.campaign_local_id}"
                final_segment_template_miss_campaigns.append(miss_campaign_id)
                final_segment_template_miss_count += len(miss_examples)
                final_segment_template_miss_examples.extend(miss_examples[:5])

                # Track salvage statistics
                final_segment_dirty_window_sizes.append(salvage_stats.get("dirty_window_size", 0))
                final_segment_kept_left_orders.append(salvage_stats.get("kept_left_orders", 0))
                final_segment_kept_right_orders.append(salvage_stats.get("kept_right_orders", 0))

                if salvaged_segs:
                    # At least some piece was salvaged
                    final_segment_salvage_success_count += 1
                    final_segment_salvaged_piece_count += len(salvaged_segs)
                    clean_final_segs.extend(salvaged_segs)
                    kept_clean = len(salvaged_segs)
                    demoted_frags_count = len(demoted_frags)

                    print(
                        f"[APS][FINAL_SALVAGE_HIT] campaign={miss_campaign_id}, "
                        f"line={seg.line}, "
                        f"offending_pair=({salvage_stats.get('offending_pair_a', '?')}->{salvage_stats.get('offending_pair_b', '?')}), "
                        f"kept_clean_pieces={kept_clean}, "
                        f"demoted_fragments={demoted_frags_count}, "
                        f"kept_left={salvage_stats.get('kept_left_orders', 0)}, "
                        f"kept_right={salvage_stats.get('kept_right_orders', 0)}"
                    )

                    print(
                        f"[APS][final_segment_salvage] campaign={miss_campaign_id}, "
                        f"offending_pair=({salvage_stats.get('offending_pair_a', '?')}->{salvage_stats.get('offending_pair_b', '?')}), "
                        f"kept_left={salvage_stats.get('kept_left_orders', 0)}, "
                        f"kept_right={salvage_stats.get('kept_right_orders', 0)}, "
                        f"dirty_window={salvage_stats.get('dirty_window_size', 0)}, "
                        f"kept_pieces={kept_clean}, "
                        f"dropped_fragments={demoted_frags_count}, "
                        f"expand_steps={salvage_stats.get('expand_steps_used', 0)}"
                    )
                else:
                    # Nothing could be salvaged: demote whole segment
                    final_segment_full_drop_count += 1
                    final_demoted_fragments.append({
                        "piece_oids": seg.order_ids,
                        "reason": DropReason.FINAL_SEGMENT_TEMPLATE_MISS_NO_VALID_SPLIT.value,
                        "detail": "entire_segment_salvage_failed",
                        "original_campaign": miss_campaign_id,
                    })
                    final_segment_template_miss_orders.extend(seg.order_ids)

                    print(
                        f"[APS][FINAL_SALVAGE_WARNING] campaign={miss_campaign_id}, "
                        f"full_drop_after_salvage=True, "
                        f"offending_pair=({miss_examples[0]['order_id_a']}->{miss_examples[0]['order_id_b']}), "
                        f"dropped_orders={seg.order_ids}, "
                        f"miss_edge_count={len(miss_examples)}"
                    )

                    print(
                        f"[APS][final_segment_drop] campaign={miss_campaign_id}, "
                        f"offending_pair=({miss_examples[0]['order_id_a']}->{miss_examples[0]['order_id_b']}), "
                        f"dropped_orders={seg.order_ids}, "
                        f"miss_edge_count={len(miss_examples)}, "
                        f"salvage_failed=True"
                    )

                # Track demoted fragments
                for frag in demoted_frags:
                    frag["original_campaign"] = miss_campaign_id
                    final_demoted_fragments.append(frag)
                    # Detailed log for each demoted fragment
                    print(
                        f"[APS][final_segment_demote_fragment] campaign={miss_campaign_id}, "
                        f"fragment_orders={frag.get('piece_oids', [])}, "
                        f"reason={frag.get('reason', '?')}, "
                        f"detail={frag.get('detail', '')}"
                    )
                final_segment_demoted_fragment_count += len(demoted_frags)

            else:
                clean_final_segs.append(seg)

        total_valid = sum(1 for s in current_segs if s.is_valid)
        if final_segment_template_miss_count > 0:
            salvage_rate = (
                final_segment_salvaged_piece_count / max(1, final_segment_salvaged_piece_count + final_segment_demoted_fragment_count) * 100
            )
            avg_dirty_window = sum(final_segment_dirty_window_sizes) / max(1, len(final_segment_dirty_window_sizes))
            avg_kept_left = sum(final_segment_kept_left_orders) / max(1, len(final_segment_kept_left_orders))
            avg_kept_right = sum(final_segment_kept_right_orders) / max(1, len(final_segment_kept_right_orders))
            total_kept_orders = sum(final_segment_kept_left_orders) + sum(final_segment_kept_right_orders) + final_segment_salvaged_piece_count

            print(
                f"[APS][FINAL_SALVAGE_SUMMARY] "
                f"attempts={final_segment_salvage_attempts}, "
                f"success={final_segment_salvage_success_count}, "
                f"salvaged_pieces={final_segment_salvaged_piece_count}, "
                f"demoted_fragments={final_segment_demoted_fragment_count}, "
                f"full_drops={final_segment_full_drop_count}"
            )
            if final_segment_salvage_attempts > 0 and final_segment_salvaged_piece_count == 0:
                print(
                    "[APS][FINAL_SALVAGE_WARNING] salvage logic entered but kept zero clean pieces"
                )

            print(
                f"[APS][final_segment_template_miss] "
                f"total_miss_count={final_segment_template_miss_count}, "
                f"miss_campaigns={final_segment_template_miss_campaigns}, "
                f"salvage_attempts={final_segment_salvage_attempts}, "
                f"salvage_success_count={final_segment_salvage_success_count}, "
                f"salvaged_pieces={final_segment_salvaged_piece_count}, "
                f"total_kept_orders={total_kept_orders}, "
                f"demoted_fragments={final_segment_demoted_fragment_count}, "
                f"salvage_rate={salvage_rate:.1f}%, "
                f"avg_dirty_window={avg_dirty_window:.1f}, "
                f"avg_kept_left={avg_kept_left:.1f}, "
                f"avg_kept_right={avg_kept_right:.1f}, "
                f"clean_segments={len(clean_final_segs)}, "
                f"total_valid={total_valid}"
            )
            for miss in final_segment_template_miss_examples[:5]:
                print(
                    f"  [FINAL_MISS] line={miss['line']}, "
                    f"campaign_local_id={miss['campaign_local_id']}, "
                    f"seq={miss['seq_a']}->{miss['seq_b']}, "
                    f"order_id_a={miss['order_id_a']}, "
                    f"order_id_b={miss['order_id_b']}"
                )
        else:
            print(
                f"[APS][final_segment_template_check] "
                f"all_ok=True, segments_checked={total_valid}"
            )

    else:
        # No tpl_df or no segments: everything is clean by default
        clean_final_segs = [s for s in current_segs if s.is_valid]

    # ---- Add demoted fragment orders to the dropped pool (per-fragment reason) ----
    for frag in final_demoted_fragments:
        frag_oids = frag.get("piece_oids", [])
        frag_reason = frag.get("reason", DropReason.FINAL_SEGMENT_TEMPLATE_MISS_FRAGMENT.value)
        current_dropped.setdefault(frag_reason, []).extend(frag_oids)

    # Also add orders from segments that couldn't be salvaged at all
    for miss_campaign_id in final_segment_template_miss_campaigns:
        # Find fragments for this campaign that represent whole-segment demotion
        for frag in final_demoted_fragments:
            if frag.get("original_campaign") == miss_campaign_id:
                break
        else:
            # If campaign was in miss list but no fragments tracked, it means salvage succeeded
            # So we don't need to add whole-segment drops here
            pass

    # Dedupe all drop reason entries
    for reason_key in list(current_dropped.keys()):
        if reason_key.startswith("FINAL_SEGMENT_TEMPLATE_MISS"):
            dropped_list = current_dropped.get(reason_key, [])
            if dropped_list:
                current_dropped[reason_key] = list(dict.fromkeys(dropped_list))

    # Build line_hint_map for demoted orders (use original segment's line)
    # Initialize from constructive_line_hint_map (defined earlier in the function)
    final_line_hint_map: Dict[str, str] = dict(constructive_line_hint_map)
    for frag in final_demoted_fragments:
        frag_oids = frag.get("piece_oids", [])
        # Use original segment's line from miss_campaigns tracking
        for miss_campaign_id in final_segment_template_miss_campaigns:
            # Parse line from campaign_id (format: "{line}_{local_id}")
            line_from_camp = miss_campaign_id.rsplit("_", 1)[0] if "_" in miss_campaign_id else miss_campaign_id
            # Only apply if this fragment belongs to this campaign
            # Since we don't have direct mapping, we'll apply to all demoted fragments
            for oid in frag_oids:
                if oid not in final_line_hint_map:
                    final_line_hint_map[oid] = line_from_camp

    # ---- Step E: Build final output from CLEAN segments only ----
    final_planned_df, final_slot_diag = _segments_to_planned_df(
        clean_final_segs, orders_for_build, tpl_df, cfg=cfg
    )


    edge_policy_label = (
        "direct_only"
        if not allow_virtual_bridge and not allow_real_bridge
        else "direct_plus_real_bridge"
        if not allow_virtual_bridge
        else "all_edges_allowed"
    )
    # ---- Constructive edge policy leak detection ----
    # Check only edge types forbidden by the currently selected policy.
    if not allow_virtual_bridge or not allow_real_bridge:
        real_bridge_leak = 0
        virtual_bridge_leak = 0
        if not final_planned_df.empty and "selected_edge_type" in final_planned_df.columns:
            if not allow_real_bridge:
                real_bridge_leak = int((final_planned_df["selected_edge_type"] == "REAL_BRIDGE_EDGE").sum())
            if not allow_virtual_bridge:
                virtual_bridge_leak = int((final_planned_df["selected_edge_type"] == "VIRTUAL_BRIDGE_EDGE").sum())
        if real_bridge_leak > 0 or virtual_bridge_leak > 0:
            print(
                f"[APS][WARNING] constructive edge policy leak detected: "
                f"edge_policy={edge_policy_label}, allow_virtual={allow_virtual_bridge}, "
                f"allow_real={allow_real_bridge}, forbidden_real_edges={real_bridge_leak}, "
                f"forbidden_virtual_edges={virtual_bridge_leak}"
            )
            diagnostics["bridge_edge_leak_detected"] = True
            diagnostics["bridge_edge_leak_real_count"] = real_bridge_leak
            diagnostics["bridge_edge_leak_virtual_count"] = virtual_bridge_leak
            engine_meta["bridge_edge_leak_detected"] = True
        else:
            print(
                f"[APS][constructive_edge_policy_confirm] constructive_edge_policy={edge_policy_label}, "
                f"forbidden edge leak count=0"
            )
            diagnostics["bridge_edge_leak_detected"] = False
            diagnostics["bridge_edge_leak_real_count"] = 0
            diagnostics["bridge_edge_leak_virtual_count"] = 0
            engine_meta["bridge_edge_leak_detected"] = False

    # Build a composite line_hint_map from all LNS rounds (later round wins per order)
    # final_line_hint_map was initialized in Step E0 from constructive_line_hint_map.
    # Merge LNS round hints on top: LNS round overrides for overlap.
    # (Note: demoted segment orders were already added to final_line_hint_map in Step E0)
    if lns_line_hint_map:
        final_line_hint_map.update(lns_line_hint_map)

    # Use full orders_df for dropped so NO_FEASIBLE_LINE orders are included
    final_dropped_df = _dropped_to_df(
        current_dropped, orders_df,
        stage="final", round_hint=-1,
        line_hint_map=final_line_hint_map,
    )

    # Determine final status
    if rounds_records and any(r.accepted for r in rounds_records):
        final_status = LnsStatus.OPTIMAL if any(
            r.accepted and r.round == r.round for r in rounds_records
        ) else LnsStatus.FEASIBLE
        if not any(r.accepted for r in rounds_records):
            final_status = LnsStatus.FEASIBLE
        else:
            # Check if any round improved
            any_improved = any(r.accepted for r in rounds_records)
            final_status = LnsStatus.OPTIMAL if any_improved else LnsStatus.FEASIBLE
    else:
        final_status = LnsStatus.NO_IMPROVEMENT

    # Build rounds_df
    rounds_rows = []
    for rec in rounds_records:
        rounds_rows.append({
            "round": rec.round,
            "neighborhood_type": rec.neighborhood_type.value,
            "accepted": rec.accepted,
            "scheduled_order_count": rec.scheduled_order_count,
            "scheduled_tons": rec.scheduled_tons,
            "dropped_count": rec.dropped_count,
            "hard_constraint_protected_drop_count": rec.hard_constraint_protected_drop_count,
            "lns_repair_rejected_count": rec.lns_repair_rejected_count,
            "repair_segment_candidate_count": rec.repair_segment_candidate_count,
            "repair_pool_candidate_count": rec.repair_pool_candidate_count,
            "repair_pool_filtered_by_capability_count": rec.repair_pool_filtered_by_capability_count,
            "repair_pool_filtered_by_connectivity_count": rec.repair_pool_filtered_by_connectivity_count,
            "repair_pool_skipped_as_already_tried_count": rec.repair_pool_skipped_as_already_tried_count,
            "virtual_bridge_count": rec.virtual_bridge_count,
            "template_cost": rec.template_cost,
            "destroy_count": rec.destroy_count,
            "repair_count": rec.repair_count,
            "elapsed_seconds": rec.elapsed_seconds,
            "notes": rec.notes,
            # TAIL_REBALANCE neighborhood diagnostics
            "target_fill_ratio": rec.target_fill_ratio,
            "target_tail_tons": rec.target_tail_tons,
            "target_prev_segment_tons": rec.target_prev_segment_tons,
            # Guarded virtual family ALNS diagnostics
            "virtual_family_frontload_enabled": rec.virtual_family_frontload_enabled,
            "alns_virtual_family_attempt_count": rec.alns_virtual_family_attempt_count,
            "alns_virtual_family_accept_count": rec.alns_virtual_family_accept_count,
            "alns_virtual_family_reject_count": rec.alns_virtual_family_reject_count,
            "alns_virtual_family_budget_block_count": rec.alns_virtual_family_budget_block_count,
            "alns_virtual_family_no_gain_count": rec.alns_virtual_family_no_gain_count,
            "local_cpsat_virtual_family_selected_count": rec.local_cpsat_virtual_family_selected_count,
        })
    rounds_df = pd.DataFrame(rounds_rows)

    # Engine metadata
    total_time = time.perf_counter() - t0
    lns_total_seconds = (time.perf_counter() - t_lns_start) if 't_lns_start' in dir() else 0.0

    # Bridge edge policy variables are already defined at the top of the function
    # (allow_virtual_bridge, allow_real_bridge, bridge_expansion_mode)

    cut_diag_for_meta = cut_result.diagnostics if cut_result and isinstance(cut_result.diagnostics, dict) else {}
    recon_observability_keys = [
        "underfilled_reconstruction_enabled",
        "underfilled_reconstruction_attempts",
        "underfilled_reconstruction_success",
        "underfilled_reconstruction_blocks_tested",
        "underfilled_reconstruction_blocks_skipped",
        "underfilled_reconstruction_valid_before",
        "underfilled_reconstruction_valid_after",
        "underfilled_reconstruction_underfilled_before",
        "underfilled_reconstruction_underfilled_after",
        "underfilled_reconstruction_valid_delta",
        "underfilled_reconstruction_underfilled_delta",
        "underfilled_reconstruction_segments_salvaged",
        "underfilled_reconstruction_orders_salvaged",
        "underfilled_reconstruction_not_entered_reason",
        "repair_only_real_bridge_enabled",
        "repair_only_real_bridge_attempts",
        "repair_only_real_bridge_success",
        "repair_only_real_bridge_candidates_total",
        "repair_only_real_bridge_candidates_kept",
        "repair_only_real_bridge_filtered_direct_feasible",
        "repair_only_real_bridge_filtered_pair_invalid",
        "repair_only_real_bridge_filtered_ton_invalid",
        "repair_only_real_bridge_filtered_score_worse",
        "repair_only_real_bridge_filtered_bridge_limit_exceeded",
        "repair_only_real_bridge_filtered_multiplicity_invalid",
        "repair_only_real_bridge_filtered_bridge_path_not_real",
        "repair_only_real_bridge_filtered_bridge_path_missing",
        "repair_only_real_bridge_filtered_block_order_mismatch",
        "repair_only_real_bridge_filtered_line_mismatch",
        "repair_only_real_bridge_filtered_block_membership_mismatch",
        "repair_only_real_bridge_filtered_bridge_path_payload_empty",
        "repair_bridge_pack_has_real_rows",
        "repair_bridge_pack_real_rows_total",
        "repair_bridge_pack_virtual_rows_total",
        "repair_bridge_raw_rows_total",
        "repair_bridge_matched_rows_total",
        "repair_bridge_kept_rows_total",
        "repair_bridge_endpoint_key_mismatch_count",
        "repair_bridge_field_name_mismatch_count",
        "repair_bridge_inconsistency_count",
        "repair_bridge_boundary_band_enabled",
        "repair_bridge_band_pairs_tested",
        "repair_bridge_band_hits",
        "repair_bridge_single_point_hits",
        "repair_bridge_band_only_hits",
        "repair_bridge_band_best_distance",
        "repair_bridge_endpoint_adjustment_enabled",
        "repair_bridge_adjustments_generated",
        "repair_bridge_adjustment_pairs_tested",
        "repair_bridge_adjustment_hits",
        "repair_bridge_adjustment_only_hits",
        "repair_bridge_best_adjustment_cost",
        "repair_bridge_candidates_matched",
        "repair_bridge_candidates_rejected_pair_invalid",
        "repair_bridge_candidates_rejected_ton_invalid",
        "repair_bridge_candidates_rejected_score_worse",
        "repair_bridge_candidates_accepted",
        "repair_bridge_exact_invalid_pair_count",
        "repair_bridge_frontier_mismatch_count",
        "repair_bridge_pair_invalid_width",
        "repair_bridge_pair_invalid_thickness",
        "repair_bridge_pair_invalid_temp",
        "repair_bridge_pair_invalid_group",
        "repair_bridge_pair_invalid_context",
        "repair_bridge_pair_invalid_unknown",
        "repair_bridge_pair_fail_thickness",
        "repair_bridge_prefilter_fail_thickness",
        "repair_bridge_rank_pass_thickness",
        "repair_bridge_rank_fail_thickness",
        "repair_bridge_template_no_edge_count",
        "repair_bridge_prefilter_all_fail_count",
        "repair_bridge_band_too_narrow_count",
        "repair_bridge_prefilter_reject_count",
        "repair_bridge_endpoint_early_stop_count",
        "repair_bridge_local_band_retry_count",
        "underfilled_reconstruction_improvement_recorded_count",
        "underfilled_reconstruction_improvement_applied_count",
        "underfilled_reconstruction_apply_reject_count",
        "bridgeability_route_suggestion",
        "bridgeability_census",
        "bridgeability_census_items",
        # ---- Legacy virtual pilot: 降级为单一总字段 ----
        # legacy virtual pilot 相关详细字段已从默认 engine_meta 移除；
        # 只保留一个总字段用于判断是否运行过 virtual pilot。
        # 旧有细粒度字段仍在 campaign_cutter.py 内部记录，兼容旧调用方。
        "virtual_pilot_skipped_due_to_disabled",
        "conservative_apply_attempt_count",
        "conservative_apply_success_count",
        "conservative_apply_reject_count",
        "repair_bridge_ton_rescue_attempts",
        "repair_bridge_ton_rescue_success",
        "repair_bridge_ton_rescue_windows_tested",
        "repair_bridge_ton_rescue_valid_delta",
        "repair_bridge_ton_rescue_underfilled_delta",
        "repair_bridge_ton_rescue_scheduled_orders_delta",
        "repair_bridge_filtered_ton_below_min_current_block",
        "repair_bridge_filtered_ton_below_min_even_after_neighbor_expansion",
        "repair_bridge_filtered_ton_above_max_after_expansion",
        "repair_bridge_filtered_ton_split_not_found",
        "repair_bridge_filtered_ton_rescue_no_gain",
        "repair_bridge_filtered_ton_rescue_impossible",
        "repair_bridge_ton_rescue_pair_fail_width",
        "repair_bridge_ton_rescue_pair_fail_thickness",
        "repair_bridge_ton_rescue_pair_fail_temp",
        "repair_bridge_ton_rescue_pair_fail_group",
        "repair_bridge_ton_rescue_pair_fail_context",
        "repair_bridge_ton_rescue_pair_fail_template",
        "repair_bridge_ton_rescue_pair_fail_multi",
        "repair_bridge_ton_rescue_pair_fail_unknown",
        "repair_bridge_pack_type",
        "repair_bridge_pack_keys",
        "repair_bridge_pack_line_keys",
        "repair_only_real_bridge_used_segments",
        "repair_only_real_bridge_used_orders",
        "repair_only_real_bridge_not_entered_reason",
        "underfilled_reconstruction_seconds",
        "repair_only_real_bridge_seconds",
    ]

    def _recon_default(key: str):
        if key == "underfilled_reconstruction_enabled":
            return True
        if key == "repair_only_real_bridge_enabled":
            return bool(getattr(cfg.model, "repair_only_real_bridge_enabled", True))
        if key == "repair_bridge_pack_has_real_rows":
            return False
        if key == "repair_bridge_boundary_band_enabled":
            return True
        if key == "repair_bridge_endpoint_adjustment_enabled":
            return True
        if key in {"repair_bridge_band_best_distance", "repair_bridge_best_adjustment_cost"}:
            return -1
        if key in {"repair_bridge_pack_keys", "repair_bridge_pack_line_keys", "bridgeability_census_items"}:
            return []
        if key == "bridgeability_census":
            return {}
        # ---- Legacy virtual pilot: 降级为单一布尔总字段 ----
        if key == "virtual_pilot_skipped_due_to_disabled":
            # 当 repair_only_virtual_bridge_pilot_enabled=False 时，virtual pilot 被禁用
            return bool(getattr(cfg.model, "repair_only_virtual_bridge_pilot_enabled", False) is False)
        if key in {"repair_bridge_pack_type", "bridgeability_route_suggestion"}:
            return ""
        if key.endswith("_reason"):
            return "NOT_REPORTED_BY_CUTTER"
        if key.endswith("_seconds"):
            return 0.0
        return 0

    recon_observability = {
        key: cut_diag_for_meta.get(key, _recon_default(key))
        for key in recon_observability_keys
    }

    # Update engine_meta (preserving any early-set values like bridge_edge_leak_detected)
    engine_meta.update({
        "random_seed": random_seed,
        "n_rounds_ran": len(rounds_records),
        "n_rounds_planned": n_rounds,
        "total_seconds": total_time,
        # ---- Phase timing ----
        "constructive_build_seconds": constructive_build_seconds if 'constructive_build_seconds' in dir() else 0.0,
        "campaign_cutter_seconds": campaign_cutter_seconds if 'campaign_cutter_seconds' in dir() else 0.0,
        "lns_total_seconds": lns_total_seconds,
        # ---- Tail repair sub-timing from cutter ----
        "tail_repair_seconds": cut_result.diagnostics.get("tail_rebalance_summary", {}).get("tail_repair_seconds", 0.0) if cut_result and cut_result.diagnostics else 0.0,
        "recut_seconds": cut_result.diagnostics.get("tail_rebalance_summary", {}).get("recut_seconds", 0.0) if cut_result and cut_result.diagnostics else 0.0,
        "shift_seconds": cut_result.diagnostics.get("tail_rebalance_summary", {}).get("shift_seconds", 0.0) if cut_result and cut_result.diagnostics else 0.0,
        "fill_seconds": cut_result.diagnostics.get("tail_rebalance_summary", {}).get("fill_seconds", 0.0) if cut_result and cut_result.diagnostics else 0.0,
        "merge_seconds": cut_result.diagnostics.get("tail_rebalance_summary", {}).get("merge_seconds", 0.0) if cut_result and cut_result.diagnostics else 0.0,
        "lns_path": "constructive_lns_master",
        "no_global_joint_model": True,
        "no_slot_bucket": True,
        "no_illegal_penalty": True,
        # Bridge edge policy (Route C = direct_only)
        "bridge_expansion_mode": bridge_expansion_mode,
        "virtual_bridge_edge_enabled_in_constructive": allow_virtual_bridge,
        "real_bridge_edge_enabled_in_constructive": allow_real_bridge,
        "constructive_edge_policy": (
            "direct_only"  # Route C: both virtual and real bridge edges are disabled
            if not allow_virtual_bridge and not allow_real_bridge
            else "direct_plus_real_bridge"  # Route RB: real allowed, virtual blocked
            if not allow_virtual_bridge
            else "direct_plus_real_plus_guarded_family"  # Guarded virtual family frontload
            if getattr(cfg.model, "virtual_family_frontload_enabled", False)
            else "all_edges_allowed"  # Legacy/experimental
        ),
        "accepted_direct_edge_count": int(build_result.diagnostics.get("accepted_direct_edge_count", 0) or 0),
        "accepted_real_bridge_edge_count": int(build_result.diagnostics.get("accepted_real_bridge_edge_count", 0) or 0),
        "filtered_virtual_bridge_edge_count": int(build_result.diagnostics.get("filtered_virtual_bridge_edge_count", 0) or 0),
        "filtered_real_bridge_edge_count": int(build_result.diagnostics.get("filtered_real_bridge_edge_count", 0) or 0),
        "joint_master_branch_enabled": False,
        "old_master_blocked": True,
        "campaign_id_string_preserved": True,
        # neighborhood pool diagnostics
        "neighborhood_pool_used": [n.value for n in neighborhoods],
        # bridge_edge_leak_detected is already set by the leak detection block above;
        # do NOT reset it here — preserve whatever was recorded.
    })
    engine_meta.update(recon_observability)

    # ---- Print PHASE_TIMING summary ----
    _tail_repair_sec = cut_result.diagnostics.get("tail_rebalance_summary", {}).get("tail_repair_seconds", 0.0) if cut_result and cut_result.diagnostics else 0.0
    _recut_sec = cut_result.diagnostics.get("tail_rebalance_summary", {}).get("recut_seconds", 0.0) if cut_result and cut_result.diagnostics else 0.0
    _shift_sec = cut_result.diagnostics.get("tail_rebalance_summary", {}).get("shift_seconds", 0.0) if cut_result and cut_result.diagnostics else 0.0
    _fill_sec = cut_result.diagnostics.get("tail_rebalance_summary", {}).get("fill_seconds", 0.0) if cut_result and cut_result.diagnostics else 0.0
    _merge_sec = cut_result.diagnostics.get("tail_rebalance_summary", {}).get("merge_seconds", 0.0) if cut_result and cut_result.diagnostics else 0.0
    print(
        f"[APS][PHASE_TIMING] constructive_build={constructive_build_seconds:.3f}s, "
        f"campaign_cutter={campaign_cutter_seconds:.3f}s, "
        f"lns={lns_total_seconds:.3f}s, total={total_time:.3f}s"
    )
    print(
        f"[APS][CUTTER_TIMING] tail_repair={_tail_repair_sec:.3f}s, "
        f"recut={_recut_sec:.3f}s, shift={_shift_sec:.3f}s, "
        f"fill={_fill_sec:.3f}s, merge={_merge_sec:.3f}s"
    )

    # Diagnostics
    # Build drop_reason_counts from final dropped_df
    drop_reason_counts: Dict[str, int] = {}
    if not final_dropped_df.empty:
        for _, row in final_dropped_df.iterrows():
            reason = str(row.get("drop_reason", ""))
            drop_reason_counts[reason] = drop_reason_counts.get(reason, 0) + 1

    # Update diagnostics (preserving early-set values like bridge_edge_leak_detected)
    diagnostics.update({
        "initial_planned_count": int(initial_planned_df.shape[0]) if not initial_planned_df.empty else 0,
        "final_planned_count": int(final_planned_df.shape[0]) if not final_planned_df.empty else 0,
        "initial_dropped_count": int(initial_dropped_df["order_id"].nunique()) if not initial_dropped_df.empty else 0,
        "final_dropped_count": int(final_dropped_df["order_id"].nunique()) if not final_dropped_df.empty else 0,
        # line_hint breakdown: source attribution
        "dropped_line_hint_from_map_count": int(
            final_dropped_df.attrs.get("_hint_from_map_count", 0)
            if not final_dropped_df.empty
            else 0
        ),
        "dropped_line_hint_from_capability_count": int(
            final_dropped_df.attrs.get("_hint_from_capability_count", 0)
            if not final_dropped_df.empty
            else 0
        ),
        "dropped_line_hint_empty_count": int(
            (final_dropped_df["line_hint"] == "").sum() if not final_dropped_df.empty else 0
        ),
        "no_feasible_line_count": drop_reason_counts.get(DropReason.NO_FEASIBLE_LINE.value, 0),
        "hard_constraint_protected_drop_count": drop_reason_counts.get(
            DropReason.HARD_CONSTRAINT_PROTECTED_DROP.value, 0
        ),
        "drop_reason_counts": drop_reason_counts,
        "improvement_delta_orders": (
            int(final_planned_df.shape[0]) - int(initial_planned_df.shape[0])
            if not final_planned_df.empty and not initial_planned_df.empty else 0
        ),
        # ---- Slot semantic self-check ----
        "slot_semantic_check_ok": bool(final_slot_diag.get("slot_semantic_check_ok", True)),
        "slot_semantic_violations": final_slot_diag.get("slot_semantic_violations", []),
        "slot_total_segments": int(final_slot_diag.get("total_segments", 0)),
        "slot_total_orders": int(final_slot_diag.get("total_orders", 0)),
        # Initial planned_df slot check (baseline comparison)
        "initial_slot_semantic_check_ok": bool(init_slot_diag.get("slot_semantic_check_ok", True)),
        # ---- NO_FEASIBLE_LINE pre-check diagnostics ----
        "no_feasible_line_capability_warnings": no_feas_line_diag.get(
            "no_feasible_line_capability_warnings", []
        ),
        "normalized_capability_counts": no_feas_line_diag.get(
            "normalized_capability_counts", {}
        ),
        "no_feasible_line_precheck_count": int(no_feas_line_diag.get("no_feasible_line_count", 0)),
        "constructive_build_diags": build_result.diagnostics,
        "campaign_cut_diags": cut_result.diagnostics if cut_result.diagnostics else {},
        **recon_observability,
        # ---- Final segment template pair validation + salvage diagnostics ----
        # template miss segments are now salvaged: clean pieces kept, only dirty fragments demoted
        "final_segment_template_miss_count": final_segment_template_miss_count,
        "final_segment_template_miss_campaigns": final_segment_template_miss_campaigns,
        "final_segment_template_miss_orders": final_segment_template_miss_orders,
        "final_segment_template_miss_examples": final_segment_template_miss_examples,
        "final_segment_demoted_count": len(final_demoted_fragments),
        "final_segment_clean_count": len(clean_final_segs),
        "final_segment_dropped_due_to_template_miss": len(final_demoted_fragments),
        # Salvage-specific diagnostics (fine-grained dirty-window strategy)
        "final_segment_salvage_attempts": final_segment_salvage_attempts,
        "final_segment_salvage_success_count": final_segment_salvage_success_count,
        "final_segment_salvaged_piece_count": final_segment_salvaged_piece_count,
        "final_segment_demoted_fragment_count": final_segment_demoted_fragment_count,
        "final_segment_full_drop_count": final_segment_full_drop_count,
        "final_segment_dirty_window_avg_size": round(sum(final_segment_dirty_window_sizes) / max(1, len(final_segment_dirty_window_sizes)), 2),
        "final_segment_dirty_window_sizes": final_segment_dirty_window_sizes,
        "final_segment_kept_left_orders_total": sum(final_segment_kept_left_orders),
        "final_segment_kept_right_orders_total": sum(final_segment_kept_right_orders),
        "final_segment_kept_orders_total": sum(final_segment_kept_left_orders) + sum(final_segment_kept_right_orders) + final_segment_salvaged_piece_count,
        "final_segment_demoted_fragment_reasons": [
            {"reason": f.get("reason", ""), "detail": f.get("detail", ""), "campaign": f.get("original_campaign", ""), "piece_type": f.get("piece_type", "")}
            for f in final_demoted_fragments[:50]  # Cap at 50 for diagnostics
        ],
        # Bridge edge policy diagnostics (Route C = direct_only)
        "bridge_expansion_mode": bridge_expansion_mode,
        "virtual_bridge_edge_enabled_in_constructive": allow_virtual_bridge,
        "real_bridge_edge_enabled_in_constructive": allow_real_bridge,
        "constructive_edge_policy": (
            "direct_only"  # Route C: both virtual and real bridge edges are disabled
            if not allow_virtual_bridge and not allow_real_bridge
            else "direct_plus_real_bridge"
            if not allow_virtual_bridge
            else "all_edges_allowed"
        ),
        "accepted_direct_edge_count": int(build_result.diagnostics.get("accepted_direct_edge_count", 0) or 0),
        "accepted_real_bridge_edge_count": int(build_result.diagnostics.get("accepted_real_bridge_edge_count", 0) or 0),
        "filtered_virtual_bridge_edge_count": int(build_result.diagnostics.get("filtered_virtual_bridge_edge_count", 0) or 0),
        "filtered_real_bridge_edge_count": int(build_result.diagnostics.get("filtered_real_bridge_edge_count", 0) or 0),
        # Leak detection: if direct_only mode but REAL/VIRTUAL_BRIDGE_EDGE appears in planned_df
        "bridge_edge_leak_detected": False,  # Will be set below after planned_df check
        # neighborhood pool diagnostics
        "neighborhood_pool_used": [n.value for n in neighborhoods],
        # ---- Tail Repair LNS Diagnostics ----
        "tail_rebalance_lns_success_count": accum_tail_repair_diag.get("tail_rebalance_lns_success_count", 0),
        "tail_rebalance_lns_shifted_orders": accum_tail_repair_diag.get("tail_rebalance_lns_shifted_orders", 0),
        "tail_rebalance_lns_inserted_dropped_orders": accum_tail_repair_diag.get("tail_rebalance_lns_inserted_dropped_orders", 0),
        "tail_rebalance_lns_merge_count": accum_tail_repair_diag.get("tail_rebalance_lns_merge_count", 0),
        "tail_rebalance_lns_recut_count": accum_tail_repair_diag.get("tail_rebalance_lns_recut_count", 0),
        "tail_rebalance_lns_dropped_reinsertion_attempts": accum_tail_repair_diag.get("tail_rebalance_lns_dropped_reinsertion_attempts", 0),
        "tail_rebalance_lns_dropped_reinsertion_success": accum_tail_repair_diag.get("tail_rebalance_lns_dropped_reinsertion_success", 0),
        "local_inserter_direct_arcs_allowed": accum_tail_repair_diag.get("local_inserter_direct_arcs_allowed", 0),
        "local_inserter_real_bridge_arcs_allowed": accum_tail_repair_diag.get("local_inserter_real_bridge_arcs_allowed", 0),
        "local_inserter_real_bridge_arcs_blocked": accum_tail_repair_diag.get("local_inserter_real_bridge_arcs_blocked", 0),
        "local_inserter_virtual_bridge_arcs_blocked": accum_tail_repair_diag.get("local_inserter_virtual_bridge_arcs_blocked", 0),
        "local_inserter_edge_policy_used": accum_tail_repair_diag.get("local_inserter_edge_policy_used", ""),
        "low_fill_neighborhood_total_count": sum(
            1 for r in rounds_records if r.neighborhood_type == NeighborhoodType.LOW_FILL_SEGMENT
        ),
        "tail_rebalance_neighborhood_total_count": sum(
            1 for r in rounds_records if r.neighborhood_type == NeighborhoodType.TAIL_REBALANCE
        ),
        "low_fill_neighborhood_success_count": sum(r.low_fill_neighborhood_success_count for r in rounds_records),
        "tail_rebalance_neighborhood_success_count": sum(r.tail_rebalance_neighborhood_success_count for r in rounds_records),
        "low_fill_candidates_total": accum_tail_repair_diag.get("low_fill_candidates_total", 0),
        # ---- Local CP-SAT gate stats (propagated from tail repair) ----
        "local_cpsat_skipped_due_to_gate": accum_tail_repair_diag.get("local_cpsat_skipped_due_to_gate", 0),
        "local_cpsat_attempt_count": accum_tail_repair_diag.get("local_cpsat_attempt_count", 0),
        "local_cpsat_success_count": accum_tail_repair_diag.get("local_cpsat_success_count", 0),
        "local_cpsat_no_improve_count": accum_tail_repair_diag.get("local_cpsat_no_improve_count", 0),
        "local_cpsat_infeasible_count": accum_tail_repair_diag.get("local_cpsat_infeasible_count", 0),
        "local_cpsat_total_seconds": accum_tail_repair_diag.get("local_cpsat_total_seconds", 0.0),
        "rounds_summary": {
            "accepted_count": sum(1 for r in rounds_records if r.accepted),
            "total_destroy_count": sum(r.destroy_count for r in rounds_records),
            "total_repair_count": sum(r.repair_count for r in rounds_records),
            "avg_round_seconds": (
                sum(r.elapsed_seconds for r in rounds_records) / max(1, len(rounds_records))
            ),
            # Pool candidate filtering diagnostics accumulated over all rounds
            "total_pool_filtered_by_capability": accum_pool_stats["filtered_by_capability_count"],
            "total_pool_filtered_by_connectivity": accum_pool_stats["filtered_by_connectivity_count"],
            "total_pool_skipped_already_tried": accum_pool_stats["skipped_already_tried_count"],
        },
    })

    return ConstructiveLnsResult(
        status=final_status,
        planned_df=final_planned_df,
        dropped_df=final_dropped_df,
        rounds_df=rounds_df,
        engine_meta=engine_meta,
        diagnostics=diagnostics,
    )


def _get_templates(transition_pack: dict) -> pd.DataFrame:
    """Safely extract templates DataFrame from transition_pack."""
    if not isinstance(transition_pack, dict):
        return pd.DataFrame()
    tpl = transition_pack.get("templates")
    if not isinstance(tpl, pd.DataFrame):
        return pd.DataFrame()
    return tpl


__all__ = [
    "ConstructiveLnsResult",
    "DropReason",
    "LnsRound",
    "LnsStatus",
    "NeighborhoodType",
    "run_constructive_lns_master",
    # Bridge expansion contract
    "_normalize_bridge_metadata",
]
