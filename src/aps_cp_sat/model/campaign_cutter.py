"""
Campaign Cutter: Split long constructive chains into legal campaign segments.

This module implements the SECOND layer of the Constructive + ALNS path:
- Input: Long order chains from constructive_sequence_builder
- Output: Campaign segments (draft proposals, NOT yet assigned campaign_id)

Key principles:
- Segments are cut from natural chain boundaries (tonnage window)
- No slot-first bucketing — segments emerge from sequence truncation
- No illegal edge insertion to patch underfilled tails
- Underfilled segments (< campaign_ton_min) are held separately
- Order order within segment is preserved from chain

Cut reasons:
- MAX_LIMIT: Adding next order would exceed campaign_ton_max
- TARGET_REACHED: Segment reached target window (min~max) and adding next would worsen deviation
- NO_SUCCESSOR: No more valid successors in chain
- TAIL_UNDERFILLED: Tail segment below min — placed in underfilled list
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from time import perf_counter
from typing import Dict, List, Optional, Tuple

import pandas as pd

from aps_cp_sat.config import PlannerConfig
from aps_cp_sat.model.constructive_sequence_builder import ConstructiveChain
from aps_cp_sat.transition.bridge_rules import _is_pc, _temp_overlap_len, _thick_ok, _txt


class CutReason(Enum):
    """Why a segment was cut at its end position."""
    MAX_LIMIT = "MAX_LIMIT"          # Adding next order exceeds hard max
    TARGET_REACHED = "TARGET_REACHED"  # Within window, adding next worsens deviation
    NO_SUCCESSOR = "NO_SUCCESSOR"    # Chain exhausted, natural end
    TAIL_UNDERFILLED = "TAIL_UNDERFILLED"  # Tail below min (underfilled output)


@dataclass
class CampaignSegment:
    """
    A campaign segment (draft proposal) cut from a constructive chain.

    This is NOT a final campaign with campaign_id — it is a draft
    segment waiting for ALNS optimization and campaign_id assignment.

    Attributes:
        line: Production line (big_roll / small_roll)
        campaign_local_id: Local segment index within this line (1-based)
        order_ids: Ordered list of order IDs (inherited from chain, never reordered)
        total_tons: Sum of tons for all orders in this segment
        cut_reason: Why this segment was cut at its end
        start_order_id: First order in segment
        end_order_id: Last order in segment
        edge_count: Number of template edges within this segment
        is_valid: Whether segment meets campaign ton minimum
    """
    line: str
    campaign_local_id: int
    order_ids: List[str]
    total_tons: float
    cut_reason: CutReason
    start_order_id: str
    end_order_id: str
    edge_count: int = 0
    is_valid: bool = True

    def __post_init__(self):
        if self.edge_count == 0 and len(self.order_ids) > 0:
            self.edge_count = len(self.order_ids) - 1

    def to_dict(self) -> dict:
        """Convert segment to dictionary for serialization."""
        return {
            "line": self.line,
            "campaign_local_id": self.campaign_local_id,
            "order_ids": list(self.order_ids),
            "total_tons": self.total_tons,
            "cut_reason": self.cut_reason.value,
            "start_order_id": self.start_order_id,
            "end_order_id": self.end_order_id,
            "edge_count": self.edge_count,
            "is_valid": self.is_valid,
            "order_count": len(self.order_ids),
        }


@dataclass
class PairEvalResult:
    template_ok: bool
    adjacency_ok: bool
    context_ok: bool
    reason: str
    sub_reasons: list[str] = field(default_factory=list)
    template_key: tuple[str, str, str] | tuple[str, str] | str = ""
    context_snapshot: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "template_ok": bool(self.template_ok),
            "adjacency_ok": bool(self.adjacency_ok),
            "context_ok": bool(self.context_ok),
            "reason": str(self.reason),
            "sub_reasons": list(self.sub_reasons),
            "template_key": self.template_key,
            "context_snapshot": dict(self.context_snapshot),
        }


@dataclass
class BridgeabilityCensus:
    items: list[dict] = field(default_factory=list)
    has_endpoint_edge_count: int = 0
    template_graph_no_edge_count: int = 0
    rule_prefilter_all_fail_count: int = 0
    band_too_narrow_count: int = 0
    prefilter_rejected_block_count: int = 0
    candidate_pool_empty_count: int = 0
    candidate_pool_nonempty_count: int = 0
    ton_rescue_entered_count: int = 0
    ton_rescue_success_count: int = 0
    improvement_recorded_count: int = 0
    improvement_applied_count: int = 0
    thickness_fail_dominant_count: int = 0
    width_fail_dominant_count: int = 0
    temp_fail_dominant_count: int = 0
    group_fail_dominant_count: int = 0
    multi_fail_dominant_count: int = 0

    def add_item(self, item: dict) -> None:
        endpoint_class = str(item.get("endpoint_class", "") or "")
        dominant = str(item.get("dominant_fail_reason", "") or "")
        candidate_count = int(item.get("candidate_count", 0) or 0)
        self.items.append(dict(item))
        if endpoint_class == "HAS_ENDPOINT_EDGE":
            self.has_endpoint_edge_count += 1
        elif endpoint_class == "TEMPLATE_GRAPH_NO_EDGE":
            self.template_graph_no_edge_count += 1
        elif endpoint_class == "RULE_PREFILTER_ALL_FAIL":
            self.rule_prefilter_all_fail_count += 1
        elif endpoint_class == "BAND_TOO_NARROW":
            self.band_too_narrow_count += 1
        if bool(item.get("prefilter_rejected", False)):
            self.prefilter_rejected_block_count += 1
        if candidate_count > 0:
            self.candidate_pool_nonempty_count += 1
        else:
            self.candidate_pool_empty_count += 1
        if bool(item.get("ton_rescue_entered", False)):
            self.ton_rescue_entered_count += 1
        if bool(item.get("ton_rescue_success", False)):
            self.ton_rescue_success_count += 1
        if str(item.get("final_decision", "")) == "IMPROVEMENT_RECORDED_BUT_NOT_APPLIED":
            self.improvement_recorded_count += 1
        if str(item.get("final_decision", "")) == "IMPROVEMENT_APPLIED":
            self.improvement_applied_count += 1
        if dominant == "THICKNESS_RULE_FAIL":
            self.thickness_fail_dominant_count += 1
        elif dominant == "WIDTH_RULE_FAIL":
            self.width_fail_dominant_count += 1
        elif dominant == "TEMP_OVERLAP_FAIL":
            self.temp_fail_dominant_count += 1
        elif dominant == "GROUP_SWITCH_FAIL":
            self.group_fail_dominant_count += 1
        elif dominant == "MULTI_RULE_FAIL":
            self.multi_fail_dominant_count += 1

    def to_dict(self) -> dict:
        return {
            "total_blocks": int(len(self.items)),
            "has_endpoint_edge_count": int(self.has_endpoint_edge_count),
            "template_graph_no_edge_count": int(self.template_graph_no_edge_count),
            "rule_prefilter_all_fail_count": int(self.rule_prefilter_all_fail_count),
            "band_too_narrow_count": int(self.band_too_narrow_count),
            "prefilter_rejected_block_count": int(self.prefilter_rejected_block_count),
            "candidate_pool_empty_count": int(self.candidate_pool_empty_count),
            "candidate_pool_nonempty_count": int(self.candidate_pool_nonempty_count),
            "ton_rescue_entered_count": int(self.ton_rescue_entered_count),
            "ton_rescue_success_count": int(self.ton_rescue_success_count),
            "improvement_recorded_count": int(self.improvement_recorded_count),
            "improvement_applied_count": int(self.improvement_applied_count),
            "thickness_fail_dominant_count": int(self.thickness_fail_dominant_count),
            "width_fail_dominant_count": int(self.width_fail_dominant_count),
            "temp_fail_dominant_count": int(self.temp_fail_dominant_count),
            "group_fail_dominant_count": int(self.group_fail_dominant_count),
            "multi_fail_dominant_count": int(self.multi_fail_dominant_count),
            "items": list(self.items),
        }


@dataclass
class VirtualPilotEligibilityResult:
    structural_eligible: bool
    runtime_enabled: bool
    final_eligible: bool
    endpoint_class: str
    candidate_count: int
    frontier_candidate_count: int
    pilot_target_candidates: int
    reject_reason: str
    reject_details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "structural_eligible": bool(self.structural_eligible),
            "runtime_enabled": bool(self.runtime_enabled),
            "final_eligible": bool(self.final_eligible),
            "endpoint_class": str(self.endpoint_class),
            "candidate_count": int(self.candidate_count),
            "frontier_candidate_count": int(self.frontier_candidate_count),
            "pilot_target_candidates": int(self.pilot_target_candidates),
            "reject_reason": str(self.reject_reason),
            "reject_details": dict(self.reject_details),
        }


@dataclass
class CampaignCutResult:
    """
    Result of cutting constructive chains into campaign segments.

    Attributes:
        segments: Valid segments meeting campaign ton minimum
        underfilled_segments: Segments below minimum — held separately, not final
        dropped_orders: Orders that could not form any segment
        diagnostics: Statistics and cut reason breakdowns
    """
    segments: List[CampaignSegment] = field(default_factory=list)
    underfilled_segments: List[CampaignSegment] = field(default_factory=list)
    dropped_orders: List[dict] = field(default_factory=list)
    diagnostics: Dict = field(default_factory=dict)

    def get_segments_by_line(self) -> Dict[str, List[CampaignSegment]]:
        """Group all segments (valid + underfilled) by line."""
        result: Dict[str, List[CampaignSegment]] = {"big_roll": [], "small_roll": []}
        for seg in self.segments + self.underfilled_segments:
            if seg.line in result:
                result[seg.line].append(seg)
            else:
                result[seg.line] = [seg]
        return result

    def get_total_orders_placed(self) -> int:
        """Total orders in valid segments."""
        return sum(len(seg.order_ids) for seg in self.segments)

    def get_total_tons_placed(self) -> float:
        """Total tons in valid segments."""
        return sum(seg.total_tons for seg in self.segments)


def _compute_target_deviation(
    current_tons: float,
    target: float,
    next_order_tons: float,
    max_: float,
) -> float:
    """
    Compute deviation from target after adding next order.

    Returns absolute deviation from target. Lower is better.
    We compute two scenarios:
    1. Stop now (current_tons)
    2. Continue (current_tons + next_order_tons)

    Returns the minimum deviation, preferring to stop if both are close.
    """
    dev_now = abs(current_tons - target)
    proposed = current_tons + next_order_tons
    if proposed <= max_:
        dev_cont = abs(proposed - target)
        # Prefer stopping if deviation difference is small (< 5% of target)
        if dev_now <= dev_cont + target * 0.05:
            return dev_now
        return dev_cont
    # Would exceed max — return large deviation
    return float("inf")


def _build_segment(
    chain: ConstructiveChain,
    start_idx: int,
    end_idx: int,
    cut_reason: CutReason,
    orders_df: pd.DataFrame,
    cfg: PlannerConfig,
    local_id: int,
) -> CampaignSegment:
    """Build a single CampaignSegment from a slice of a chain."""
    order_ids = chain.order_ids[start_idx : end_idx + 1]

    # Compute total tons
    total_tons = 0.0
    if not orders_df.empty:
        order_lookup = orders_df.set_index("order_id")
        for oid in order_ids:
            if oid in order_lookup.index:
                total_tons += float(order_lookup.loc[oid].get("tons", 0) or 0)
    else:
        # Fallback: use chain's ton record if orders_df unavailable
        for oid in order_ids:
            total_tons += 0.0

    return CampaignSegment(
        line=chain.line,
        campaign_local_id=local_id,
        order_ids=order_ids,
        total_tons=total_tons,
        cut_reason=cut_reason,
        start_order_id=order_ids[0],
        end_order_id=order_ids[-1],
        edge_count=max(0, len(order_ids) - 1),
    )


# ---------------------------------------------------------------------------
# Tail Rebalancing: rescue underfilled tails by pullback from previous segment
# ---------------------------------------------------------------------------

def _try_shift_from_prev(
    prev_seg: CampaignSegment,
    tail_seg: CampaignSegment,
    order_tons: Dict[str, float],
    cfg: PlannerConfig,
    tpl_df: pd.DataFrame | None = None,
) -> Tuple[bool, Optional[CampaignSegment], Optional[CampaignSegment], dict]:
    """
    Strategy B (SECONDARY): Shift K orders from the END of prev_seg to the START of tail_seg.

    Acceptance conditions (ALL must hold):
        a) After pullback, prev_seg.total_tons >= campaign_ton_min
        b) After pullback, tail_seg.total_tons >= campaign_ton_min
        c) Neither segment exceeds campaign_ton_max
        d) All adjacent pairs in both new segments exist in the template
           (including the new boundary: last(pulled) -> first(old_tail))

    Among multiple feasible K, prefer:
        A. Tail closest to target ton
        B. Prev segment not too far from target
        C. Fewer orders shifted (minimal disruption)

    Diagnostics tracked:
        - shift_k_tested_max: maximum K attempted
        - shift_valid_k_count: number of valid K found
        - shift_best_tail_gap_to_target: best tail deviation from target

    Returns:
        (success, new_prev_seg, new_tail_seg, diagnostics)
    """
    diagnostics: dict = {
        "method": "SHIFT_FROM_PREV",
        "attempted": True,
        "success": False,
        "shift_k": 0,
        "prev_tons_before": prev_seg.total_tons,
        "tail_tons_before": tail_seg.total_tons,
        "prev_tons_after": 0.0,
        "tail_tons_after": 0.0,
        "failure_reason": "",
        "shift_k_tested_max": 0,
        "shift_valid_k_count": 0,
        "shift_best_tail_gap_to_target": 0.0,
    }

    cfg_model = getattr(cfg, "model", None) if cfg else None
    if cfg_model is not None:
        if not getattr(cfg_model, "tail_rebalance_enabled", False):
            diagnostics["failure_reason"] = "disabled_in_config"
            return False, None, None, diagnostics

    ton_min = float(getattr(cfg.rule, "campaign_ton_min", 500.0) if cfg.rule else 500.0)
    ton_max = float(getattr(cfg.rule, "campaign_ton_max", 2000.0) if cfg.rule else 2000.0)
    ton_target = float(getattr(cfg.rule, "campaign_ton_target", 1500.0) if cfg.rule else 1500.0)

    max_pullback_orders = int(getattr(cfg_model, "tail_rebalance_max_pullback_orders", 8) if cfg_model else 8)
    max_pullback_tons10 = int(getattr(cfg_model, "tail_rebalance_max_pullback_tons10", 2500) if cfg_model else 2500)

    if prev_seg is None or tail_seg is None:
        diagnostics["failure_reason"] = "missing_segment"
        return False, None, None, diagnostics
    if prev_seg.line != tail_seg.line:
        diagnostics["failure_reason"] = "different_lines"
        return False, None, None, diagnostics
    if len(prev_seg.order_ids) < 2:
        diagnostics["failure_reason"] = "prev_too_short"
        return False, None, None, diagnostics

    prev_oids = list(prev_seg.order_ids)
    tail_oids = list(tail_seg.order_ids)

    # Build template keyset once
    tpl_keys: set = set()
    if tpl_df is not None and not tpl_df.empty:
        for _, trow in tpl_df.iterrows():
            tpl_keys.add((str(trow.get("from_order_id", "")), str(trow.get("to_order_id", ""))))

    def check_pairs(oids: List[str]) -> bool:
        """Check all adjacent pairs in oids are template-valid."""
        if not tpl_keys or len(oids) < 2:
            return True
        for i in range(len(oids) - 1):
            if (oids[i], oids[i + 1]) not in tpl_keys:
                return False
        return True

    best_k = 0
    best_score = float("inf")
    best_prev_new_tons = 0.0
    best_tail_new_tons = 0.0
    valid_k_count = 0

    max_k = min(max_pullback_orders, len(prev_oids) - 1)
    diagnostics["shift_k_tested_max"] = max_k

    for k in range(1, max_k + 1):
        pull_oids = prev_oids[-k:]
        pull_tons10 = sum(int(order_tons.get(oid, 0.0) * 10) for oid in pull_oids)

        if pull_tons10 > max_pullback_tons10:
            continue

        prev_new_tons = prev_seg.total_tons - (pull_tons10 / 10.0)
        tail_new_tons = tail_seg.total_tons + (pull_tons10 / 10.0)

        if prev_new_tons < ton_min - 1e-6:
            continue  # Don't record failure here, try other k
        if tail_new_tons < ton_min - 1e-6:
            continue
        if prev_new_tons > ton_max + 1e-6:
            continue
        if tail_new_tons > ton_max + 1e-6:
            continue

        new_prev_oids_k = prev_oids[:-k]
        new_tail_oids_k = pull_oids + tail_oids

        # Template pair validation for both new segments
        if not check_pairs(new_prev_oids_k):
            continue
        if not check_pairs(new_tail_oids_k):
            continue

        valid_k_count += 1

        # Multi-criteria scoring (lower is better):
        # A. Tail closest to target (primary)
        tail_dev = abs(tail_new_tons - ton_target) / ton_target

        # B. Prev not too far from target
        prev_dev = abs(prev_new_tons - ton_target) / ton_target

        # C. Fewer orders shifted (minimal disruption)
        shift_penalty = k / max_k * 0.1

        score = tail_dev + prev_dev * 0.5 + shift_penalty

        if score < best_score:
            best_score = score
            best_k = k
            best_prev_new_tons = prev_new_tons
            best_tail_new_tons = tail_new_tons

    diagnostics["shift_valid_k_count"] = valid_k_count
    diagnostics["shift_best_tail_gap_to_target"] = round(
        abs(best_tail_new_tons - ton_target) / ton_target, 4
    ) if best_k > 0 else 0.0

    if best_k == 0:
        if diagnostics["shift_k_tested_max"] > 0:
            diagnostics["failure_reason"] = "no_feasible_k_after_templates_check"
        else:
            diagnostics["failure_reason"] = "prev_too_short"
        return False, None, None, diagnostics

    pull_oids = prev_oids[-best_k:]
    new_prev_oids = prev_oids[:-best_k]
    new_tail_oids = pull_oids + tail_oids

    new_prev_seg = CampaignSegment(
        line=prev_seg.line,
        campaign_local_id=prev_seg.campaign_local_id,
        order_ids=new_prev_oids,
        total_tons=best_prev_new_tons,
        cut_reason=prev_seg.cut_reason,
        start_order_id=new_prev_oids[0] if new_prev_oids else "",
        end_order_id=new_prev_oids[-1] if new_prev_oids else "",
        edge_count=max(0, len(new_prev_oids) - 1),
        is_valid=(best_prev_new_tons >= ton_min),
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

    diagnostics["success"] = True
    diagnostics["shift_k"] = best_k
    diagnostics["prev_tons_after"] = best_prev_new_tons
    diagnostics["tail_tons_after"] = best_tail_new_tons

    print(
        f"[APS][TailRepair] line={tail_seg.line}, method=SHIFT_FROM_PREV, "
        f"shift_orders={best_k}, prev_before={prev_seg.total_tons:.0f}, "
        f"prev_after={best_prev_new_tons:.0f}, tail_before={tail_seg.total_tons:.0f}, "
        f"tail_after={best_tail_new_tons:.0f}"
    )

    return True, new_prev_seg, new_tail_seg, diagnostics


def _try_fill_from_dropped(
    tail_seg: CampaignSegment,
    order_tons: Dict[str, float],
    orders_df: pd.DataFrame,
    placed_oids: set,
    cfg: PlannerConfig,
    tpl_df: pd.DataFrame | None = None,
    max_candidates: int = 20,
) -> Tuple[bool, Optional[CampaignSegment], dict]:
    """
    Strategy D: Fill underfilled tail from dropped order candidates.

    Candidates are orders NOT in placed_oids, same line as tail, and
    (tail.last_order -> candidate) must be a valid direct template pair.

    Full success: candidate tons bring tail into [campaign_ton_min, campaign_ton_max].
    Partial progress (when enabled): candidate reduces gap_to_min even if still below min,
    provided it doesn't exceed ton_max and doesn't break template pair.

    Returns:
        (success, new_tail_seg, diagnostics)
    """
    diagnostics: dict = {
        "method": "TAIL_FILL_FROM_DROPPED",
        "attempted": True,
        "success": False,
        "partial_progress": False,
        "candidate_order_id": "",
        "fill_tons": 0.0,
        "tail_before": tail_seg.total_tons,
        "tail_after": 0.0,
        "failure_reason": "",
        "fill_candidates_tested": 0,
        "candidates_considered": 0,
        "best_gap_reduction": 0.0,
        # Candidate pool filtering diagnostics
        "pool_total_rows": len(orders_df),
        "pool_filtered_by_already_placed": 0,
        "pool_filtered_by_already_in_tail": 0,
        "pool_filtered_by_line": 0,
        "pool_filtered_by_template": 0,
    }

    cfg_model = getattr(cfg, "model", None) if cfg else None
    max_candidates = int(
        getattr(cfg_model, "max_fill_candidates_per_tail", 20) if cfg_model else 20
    )

    if tail_seg is None or orders_df.empty:
        diagnostics["failure_reason"] = "missing_segment_or_orders"
        return False, None, diagnostics

    fill_enabled = bool(getattr(cfg_model, "tail_fill_from_dropped_enabled", True) if cfg_model else True)
    if not fill_enabled:
        diagnostics["failure_reason"] = "disabled_in_config"
        return False, None, diagnostics

    accept_partial = bool(
        getattr(cfg_model, "tail_fill_accept_partial_progress", True) if cfg_model else True
    )
    fill_gap_limit = float(
        getattr(cfg_model, "tail_fill_gap_to_min_limit", 220.0) if cfg_model else 220.0
    )

    ton_min = float(getattr(cfg.rule, "campaign_ton_min", 500.0) if cfg.rule else 500.0)
    ton_max = float(getattr(cfg.rule, "campaign_ton_max", 2000.0) if cfg.rule else 2000.0)
    ton_target = float(getattr(cfg.rule, "campaign_ton_target", 1500.0) if cfg.rule else 1500.0)

    gap = ton_min - tail_seg.total_tons
    if gap <= 0:
        diagnostics["failure_reason"] = "tail_not_underfilled"
        return False, None, diagnostics
    if gap > fill_gap_limit:
        diagnostics["failure_reason"] = f"gap_too_large_{gap:.0f}_limit_{fill_gap_limit:.0f}"
        return False, None, diagnostics

    tail_last_oid = tail_seg.order_ids[-1]

    # Build template keyset (direct-only, no bridge)
    tpl_keys: set = set()
    if tpl_df is not None and not tpl_df.empty:
        for _, trow in tpl_df.iterrows():
            tpl_keys.add((str(trow.get("from_order_id", "")), str(trow.get("to_order_id", ""))))

    # Find candidate orders from orders_df.
    # Candidates must satisfy:
    #   1. NOT already in placed_oids (not in any segment at all)
    #   2. NOT already in tail_seg.order_ids
    #   3. Same line as tail_seg
    #   4. Template pair (tail_last_oid -> candidate) is valid (direct-only)
    line_col = "line" if "line" in orders_df.columns else None
    candidates = []
    for _, row in orders_df.iterrows():
        oid = str(row.get("order_id", ""))
        if oid in placed_oids:
            diagnostics["pool_filtered_by_already_placed"] += 1
            continue
        if oid in tail_seg.order_ids:
            diagnostics["pool_filtered_by_already_in_tail"] += 1
            continue
        if line_col and str(row.get(line_col, "")) != tail_seg.line:
            diagnostics["pool_filtered_by_line"] += 1
            continue
        candidates.append(oid)

    diagnostics["candidates_considered"] = len(candidates)

    # Print candidate pool summary BEFORE template filtering
    print(
        f"[APS][TAIL_FILL_POOL] "
        f"tail_seg={tail_seg.campaign_local_id}, "
        f"tail_tons={tail_seg.total_tons:.0f}, "
        f"gap={gap:.0f}, "
        f"pool_total_rows={diagnostics['pool_total_rows']}, "
        f"filtered_by_already_placed={diagnostics['pool_filtered_by_already_placed']}, "
        f"filtered_by_already_in_tail={diagnostics['pool_filtered_by_already_in_tail']}, "
        f"filtered_by_line={diagnostics['pool_filtered_by_line']}, "
        f"filtered_by_template=0 (pending), "
        f"remaining_for_test={len(candidates)}"
    )

    # Sort by tons ascending (smaller orders give partial progress without overshooting)
    candidates.sort(key=lambda o: order_tons.get(o, 0.0))
    candidates = candidates[:max_candidates]

    if not candidates:
        diagnostics["failure_reason"] = "NO_DROPPED_CANDIDATE_FOR_FILL"
        return False, None, diagnostics

    best_full_oid = ""
    best_full_score = float("inf")
    best_full_tail_tons = 0.0

    # Track best partial progress (reduces gap even if still underfilled)
    best_partial_oid = ""
    best_partial_gap_reduction = 0.0
    best_partial_tail_tons = 0.0

    for oid in candidates:
        diagnostics["fill_candidates_tested"] += 1
        cand_tons = order_tons.get(oid, 0.0)
        new_tons = tail_seg.total_tons + cand_tons

        # Template pair validation: tail_last_oid -> oid must be valid (direct-only)
        if tpl_keys and (tail_last_oid, oid) not in tpl_keys:
            diagnostics["pool_filtered_by_template"] += 1
            continue

        # Hard cap: never exceed ton_max
        if new_tons > ton_max + 1e-6:
            continue

        # Full success: brings tail into [ton_min, ton_max]
        if new_tons >= ton_min - 1e-6:
            dev = abs(new_tons - ton_target)
            if dev < best_full_score:
                best_full_score = dev
                best_full_oid = oid
                best_full_tail_tons = new_tons
        elif accept_partial:
            # Partial progress: reduces gap and doesn't overshoot max
            gap_reduction = cand_tons
            if gap_reduction > best_partial_gap_reduction:
                best_partial_gap_reduction = gap_reduction
                best_partial_oid = oid
                best_partial_tail_tons = new_tons

    diagnostics["best_gap_reduction"] = best_partial_gap_reduction

    # Prefer full success over partial
    best_oid = best_full_oid or best_partial_oid
    best_tail_after_tons = best_full_tail_tons or best_partial_tail_tons
    is_partial = bool(best_partial_oid and not best_full_oid)

    if not best_oid:
        diagnostics["failure_reason"] = "NO_DROPPED_CANDIDATE_FOR_FILL"
        return False, None, diagnostics

    new_tail_oids = list(tail_seg.order_ids) + [best_oid]
    new_tail_seg = CampaignSegment(
        line=tail_seg.line,
        campaign_local_id=tail_seg.campaign_local_id,
        order_ids=new_tail_oids,
        total_tons=best_tail_after_tons,
        cut_reason=tail_seg.cut_reason,
        start_order_id=new_tail_oids[0],
        end_order_id=new_tail_oids[-1],
        edge_count=max(0, len(new_tail_oids) - 1),
        is_valid=(best_tail_after_tons >= ton_min),
    )

    if is_partial:
        diagnostics["success"] = True
        diagnostics["partial_progress"] = True
        diagnostics["candidate_order_id"] = best_partial_oid
        diagnostics["fill_tons"] = order_tons.get(best_partial_oid, 0.0)
        diagnostics["tail_before"] = tail_seg.total_tons
        diagnostics["tail_after"] = best_partial_tail_tons
        print(
            f"[APS][TAIL_FILL_PARTIAL] line={tail_seg.line}, "
            f"seg_tail={tail_seg.campaign_local_id}, "
            f"inserted_order={best_partial_oid}, "
            f"gap_before={gap:.0f}, gap_after={ton_min - best_partial_tail_tons:.0f}, "
            f"fill_tons={order_tons.get(best_partial_oid, 0.0):.0f}"
        )
    else:
        diagnostics["success"] = True
        diagnostics["partial_progress"] = False
        diagnostics["candidate_order_id"] = best_full_oid
        diagnostics["fill_tons"] = order_tons.get(best_full_oid, 0.0)
        diagnostics["tail_before"] = tail_seg.total_tons
        diagnostics["tail_after"] = best_full_tail_tons
        print(
            f"[APS][TAIL_FILL_HIT] line={tail_seg.line}, "
            f"seg_tail={tail_seg.campaign_local_id}, "
            f"inserted_order={best_full_oid}, "
            f"gap_before={gap:.0f}, gap_after={ton_min - best_full_tail_tons:.0f}, "
            f"fill_tons={order_tons.get(best_full_oid, 0.0):.0f}"
        )

    return True, new_tail_seg, diagnostics


def _try_merge_with_prev(
    prev_seg: CampaignSegment,
    tail_seg: CampaignSegment,
    order_tons: Dict[str, float],
    cfg: PlannerConfig,
    tpl_df: pd.DataFrame | None = None,
) -> Tuple[bool, Optional[CampaignSegment], dict]:
    """
    Strategy C (fallback): Merge tail_seg into prev_seg (append all tail orders to prev).

    Acceptance conditions (ALL must hold):
        a) Combined tons <= campaign_ton_max
        b) All adjacent pairs within the merged segment exist in the template
           (including boundary: last(prev) -> first(tail))

    Returns:
        (success, merged_seg, diagnostics)
    """
    diagnostics: dict = {
        "method": "MERGE_WITH_PREV",
        "attempted": True,
        "success": False,
        "failure_reason": "",
    }

    if prev_seg is None or tail_seg is None:
        diagnostics["failure_reason"] = "missing_segment"
        return False, None, diagnostics
    if prev_seg.line != tail_seg.line:
        diagnostics["failure_reason"] = "different_lines"
        return False, None, diagnostics

    ton_min = float(getattr(cfg.rule, "campaign_ton_min", 500.0) if cfg.rule else 500.0)
    ton_max = float(getattr(cfg.rule, "campaign_ton_max", 2000.0) if cfg.rule else 2000.0)

    combined_tons = prev_seg.total_tons + tail_seg.total_tons
    if combined_tons > ton_max + 1e-6:
        diagnostics["failure_reason"] = "MERGED_EXCEEDS_MAX"
        return False, None, diagnostics

    merged_oids = list(prev_seg.order_ids) + list(tail_seg.order_ids)

    # Template pair validation
    if tpl_df is not None and not tpl_df.empty:
        tpl_keys: set = set()
        for _, trow in tpl_df.iterrows():
            tpl_keys.add((str(trow.get("from_order_id", "")), str(trow.get("to_order_id", ""))))
        for i in range(len(merged_oids) - 1):
            if (merged_oids[i], merged_oids[i + 1]) not in tpl_keys:
                diagnostics["failure_reason"] = "TEMPLATE_PAIR_INVALID_AFTER_REPAIR"
                return False, None, diagnostics

    merged_seg = CampaignSegment(
        line=prev_seg.line,
        campaign_local_id=prev_seg.campaign_local_id,
        order_ids=merged_oids,
        total_tons=combined_tons,
        cut_reason=prev_seg.cut_reason,
        start_order_id=merged_oids[0],
        end_order_id=merged_oids[-1],
        edge_count=max(0, len(merged_oids) - 1),
        is_valid=(combined_tons >= ton_min),
    )

    diagnostics["success"] = True
    print(
        f"[APS][TailRepair] line={tail_seg.line}, method=MERGE_WITH_PREV, "
        f"prev_orders={len(prev_seg.order_ids)}, tail_orders={len(tail_seg.order_ids)}, "
        f"merged_tons={combined_tons:.0f}"
    )

    return True, merged_seg, diagnostics


def _try_recut_two_segments(
    prev_seg: CampaignSegment,
    tail_seg: CampaignSegment,
    order_tons: Dict[str, float],
    cfg: PlannerConfig,
    tpl_df: pd.DataFrame | None = None,
    recut_disabled: bool = False,
) -> Tuple[bool, Optional[CampaignSegment], Optional[CampaignSegment], dict]:
    """
    Strategy A (PRIMARY): Recut prev_seg + tail_seg together to find optimal cut point.

    Concatenates all orders from prev + tail, then selects top max_cut_points candidates
    by deviation from target (Phase 1), then evaluates only those candidates (Phase 2).
    Candidates include:
        - Points closest to target ton on both sides
        - Original boundary point (always included if within budget)

    Evaluated positions must satisfy:
        - left part tons in [ton_min, ton_max]
        - right part tons in [ton_min, ton_max]
        - both parts have all adjacent pairs valid in template

    Scoring priority for valid cut points:
        A. Both segments closer to target ton
        B. Both segments more balanced (closer to 50/50 order split)
        C. Cut point closer to original boundary (preserves natural structure)
        D. Total orders kept maximized

    Returns:
        (success, new_left_seg, new_right_seg, diagnostics)
    """
    diagnostics: dict = {
        "method": "RECUT_TWO_SEGMENTS",
        "attempted": True,
        "success": False,
        "cut_index": -1,
        "failure_reason": "",
        "recut_candidate_points_tested": 0,
        "recut_valid_points_found": 0,
        "recut_best_gap_to_target": 0.0,
        "recut_original_boundary_idx": len(prev_seg.order_ids),  # Original cut point
        "recut_disabled": recut_disabled,
        "recut_score_tuple_samples": [],
    }

    if recut_disabled:
        diagnostics["failure_reason"] = "recut_disabled_after_failures"
        return False, None, None, diagnostics

    if prev_seg is None or tail_seg is None:
        diagnostics["failure_reason"] = "missing_segment"
        return False, None, None, diagnostics
    if prev_seg.line != tail_seg.line:
        diagnostics["failure_reason"] = "different_lines"
        return False, None, None, diagnostics

    ton_min = float(getattr(cfg.rule, "campaign_ton_min", 500.0) if cfg.rule else 500.0)
    ton_max = float(getattr(cfg.rule, "campaign_ton_max", 2000.0) if cfg.rule else 2000.0)
    ton_target = float(getattr(cfg.rule, "campaign_ton_target", 1500.0) if cfg.rule else 1500.0)

    all_oids = list(prev_seg.order_ids) + list(tail_seg.order_ids)
    n = len(all_oids)
    if n < 2:
        diagnostics["failure_reason"] = "not_enough_orders"
        return False, None, None, diagnostics

    original_boundary_idx = len(prev_seg.order_ids)

    # Pre-compute prefix tons for O(1) split evaluation
    prefix_tons: List[float] = [0.0] * (n + 1)
    for i in range(n):
        prefix_tons[i + 1] = prefix_tons[i] + order_tons.get(all_oids[i], 0.0)

    def make_seg(start: int, end: int, local_id: int) -> Tuple[CampaignSegment, float]:
        oids = all_oids[start:end]
        tons = prefix_tons[end] - prefix_tons[start]
        return CampaignSegment(
            line=prev_seg.line,
            campaign_local_id=local_id,
            order_ids=oids,
            total_tons=tons,
            cut_reason=CutReason.TAIL_UNDERFILLED,
            start_order_id=oids[0],
            end_order_id=oids[-1],
            edge_count=max(0, len(oids) - 1),
            is_valid=(tons >= ton_min),
        ), tons

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
    best_score = (float("inf"), float("inf"), float("inf"))
    best_left_tons = 0.0
    best_right_tons = 0.0
    valid_points_count = 0          # both segments fully valid
    partial_right_valid_count = 0    # right >= ton_min (left may be invalid)
    best_partial_right_tons = 0.0    # best right_tons among partial-valid points

    # ---- Smart cut point selection: pick at most max_cut_points candidates ----
    cfg_model = getattr(cfg, "model", None) if cfg else None
    max_cut_points = int(
        getattr(cfg_model, "max_recut_cutpoints_per_window", 12) if cfg_model else 12
    )

    # Phase 1: collect feasible cut points — two tiers
    #   Tier-1 (full):  left ∈ [ton_min, ton_max]  AND  right ∈ [ton_min, ton_max]
    #   Tier-2 (partial): right >= ton_min          AND  left may be < ton_min or > ton_max
    full_valid_points: List[Tuple[int, float, float]] = []
    partial_valid_points: List[Tuple[int, float, float]] = []

    for k in range(1, n):
        left_tons = prefix_tons[k]
        right_tons = prefix_tons[n] - left_tons

        # Tier-1: fully valid on both sides
        if (ton_min - 1e-6 <= left_tons <= ton_max + 1e-6 and
                ton_min - 1e-6 <= right_tons <= ton_max + 1e-6):
            full_valid_points.append((k, left_tons, right_tons))
        # Tier-2: right meets min threshold (left can be anything)
        elif right_tons >= ton_min - 1e-6:
            partial_valid_points.append((k, left_tons, right_tons))

    # Sort Tier-1 by deviation from target (ascending = closer is better)
    full_valid_points.sort(key=lambda x: abs(x[1] - ton_target) + abs(x[2] - ton_target))

    # Sort Tier-2 by: A. right_tons descending (bigger right is better),
    #                 B. left deviation ascending (left closer to target is better as tiebreaker)
    partial_valid_points.sort(key=lambda x: (-x[2], abs(x[1] - ton_target)))

    # Build candidate list: all Tier-1 first (up to max_cut_points), then Tier-2
    candidate_ks: List[int] = []
    for kp in full_valid_points[:max_cut_points]:
        candidate_ks.append(kp[0])
    for kp in partial_valid_points:
        if len(candidate_ks) >= max_cut_points:
            break
        candidate_ks.append(kp[0])

    # Always include original boundary if not already in candidates
    if original_boundary_idx not in candidate_ks and 0 < original_boundary_idx < n:
        if len(candidate_ks) < max_cut_points:
            candidate_ks.append(original_boundary_idx)
        else:
            # Replace the worst candidate with original boundary
            candidate_ks[-1] = original_boundary_idx

    # Phase 2: evaluate only candidate_ks (pre-filtered, bounded by max_cut_points)
    for k in candidate_ks:
        diagnostics["recut_candidate_points_tested"] += 1

        left_tons = prefix_tons[k]
        right_tons = prefix_tons[n] - left_tons

        left_oids = all_oids[:k]
        right_oids = all_oids[k:]

        # Template validation
        if not check_template(left_oids):
            continue
        if not check_template(right_oids):
            continue

        right_is_valid = right_tons >= ton_min - 1e-6
        left_is_valid = ton_min - 1e-6 <= left_tons <= ton_max + 1e-6

        if left_is_valid and right_is_valid:
            valid_points_count += 1
        elif right_is_valid:
            partial_right_valid_count += 1
            if right_tons > best_partial_right_tons:
                best_partial_right_tons = right_tons

        # Multi-tier scoring (lower is better):
        # Tier priority: RIGHT_VALID > left deviation > right deviation > boundary proximity
        tier_priority = 0 if (left_is_valid and right_is_valid) else 1  # 0 = full, 1 = partial

        if tier_priority == 0:
            # Full-valid: score by closeness to target
            left_dev = abs(left_tons - ton_target) / ton_target
            right_dev = abs(right_tons - ton_target) / ton_target
            target_score = left_dev + right_dev
            balance_score = 0.0
        else:
            # Partial-valid: prefer largest right_tons, then left closest to target
            # Normalize: smaller right_tons_deviation is better
            right_dev = abs(right_tons - ton_min) / ton_min  # how far right is above min
            left_dev = abs(left_tons - ton_target) / ton_target
            # Composite: prefer right-first, left-second
            target_score = -right_tons / ton_min + left_dev  # negative so larger right is lower score
            balance_score = 0.0

        # Boundary proximity (small weight, only tiebreaker)
        boundary_dev = abs(k - original_boundary_idx) / n
        boundary_score = boundary_dev * 0.01

        # Compose: (tier, target_score, boundary_score) — all entries are floats
        score = (float(tier_priority), float(target_score), float(boundary_score))
        score_samples = diagnostics.setdefault("recut_score_tuple_samples", [])
        if isinstance(score_samples, list) and len(score_samples) < 5:
            score_samples.append(
                {
                    "cut_index": int(k),
                    "recut_score_tuple": tuple(round(float(v), 6) for v in score),
                    "left_tons": round(float(left_tons), 4),
                    "right_tons": round(float(right_tons), 4),
                }
            )

        if score < best_score:
            best_score = score
            best_k = k
            best_left_tons = left_tons
            best_right_tons = right_tons
            # Record a sample score for diagnostics (first valid candidate)
            if "recut_score_sample" not in diagnostics:
                diagnostics["recut_score_sample"] = {
                    "tier": int(tier_priority),
                    "target_score": round(target_score, 6),
                    "boundary_score": round(boundary_score, 6),
                    "recut_score_tuple": tuple(round(float(v), 6) for v in score),
                }

    diagnostics["recut_valid_points_found"] = valid_points_count
    diagnostics["recut_partial_right_valid_count"] = partial_right_valid_count
    diagnostics["recut_best_gap_to_target"] = round(
        abs(best_left_tons - ton_target) + abs(best_right_tons - ton_target), 4
    ) if best_k != -1 else 0.0

    if best_k == -1:
        diagnostics["failure_reason"] = "NO_VALID_RECUT_POINT"
        diagnostics["recut_selected_solution_type"] = "NO_VALID_RECUT"
        return False, None, None, diagnostics

    new_prev_seg, _ = make_seg(0, best_k, prev_seg.campaign_local_id)
    new_tail_seg, _ = make_seg(best_k, n, tail_seg.campaign_local_id)

    # Determine solution type
    if new_prev_seg.is_valid and new_tail_seg.is_valid:
        sol_type = "BOTH_VALID"
    elif new_tail_seg.is_valid:
        sol_type = "RIGHT_VALID_LEFT_VALID"
    else:
        sol_type = "PARTIAL_RIGHT_UNDERFILLED"

    diagnostics["success"] = True
    diagnostics["cut_index"] = best_k
    diagnostics["recut_selected_solution_type"] = sol_type
    diagnostics["recut_valid_right_segment_count"] = valid_points_count
    diagnostics["recut_partial_right_underfilled_count"] = partial_right_valid_count

    print(
        f"[APS][TailRepair] line={tail_seg.line}, method=RECUT_TWO_SEGMENTS, "
        f"cut_index={best_k}, left_tons={best_left_tons:.0f}, right_tons={best_right_tons:.0f}, "
        f"sol_type={sol_type}, full_valid={valid_points_count}, partial_right={partial_right_valid_count}"
    )

    return True, new_prev_seg, new_tail_seg, diagnostics


# ---------------------------------------------------------------------------
# Order multiplicity conservation check
# ---------------------------------------------------------------------------

def _check_order_multiplicity_preserved(
    before_segs: List[CampaignSegment],
    after_segs: List[CampaignSegment],
    method: str,
) -> Tuple[bool, str, dict]:
    """
    Check that order multiplicity is strictly preserved across a repair operation.

    BEFORE = the list of segments involved in repair (e.g. [prev_seg, tail_seg])
    AFTER  = the list of segments replacing them (e.g. [new_prev, new_tail])

    We compute Counter(order_id) for both and require them to be equal.

    Returns:
        (is_preserved: bool, reason_if_failed: str, details: dict)
    """
    before_counter = Counter()
    for seg in before_segs:
        if seg is not None:
            before_counter.update(seg.order_ids)

    after_counter = Counter()
    for seg in after_segs:
        if seg is not None:
            after_counter.update(seg.order_ids)

    preserved = (before_counter == after_counter)

    details = {
        "before_count": sum(before_counter.values()),
        "after_count": sum(after_counter.values()),
        "before_unique": len(before_counter),
        "after_unique": len(after_counter),
        "multiplicity_match": preserved,
    }

    if not preserved:
        # Compute diff for diagnostics
        only_before = [oid for oid in before_counter if after_counter.get(oid, 0) != before_counter[oid]]
        only_after = [oid for oid in after_counter if before_counter.get(oid, 0) != after_counter[oid]]
        details["orders_missing_in_after"] = only_before[:10]
        details["orders_extra_in_after"] = only_after[:10]
        reason = f"ORDER_MULTIPLICITY_NOT_PRESERVED: before_uniq={len(before_counter)}, after_uniq={len(after_counter)}"
        return False, reason, details

    return True, "", details


def _transition_templates_df(transition_pack) -> pd.DataFrame:
    if isinstance(transition_pack, pd.DataFrame):
        return transition_pack
    if isinstance(transition_pack, dict):
        tpl = transition_pack.get("templates")
        if isinstance(tpl, pd.DataFrame):
            return tpl
    return pd.DataFrame()


def _as_float_or_none(value) -> float | None:
    try:
        if value is None:
            return None
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _build_order_spec_map_from_transition_context(
    transition_pack,
    tpl_df: pd.DataFrame,
    orders_df: pd.DataFrame | None = None,
) -> dict[str, dict]:
    specs: dict[str, dict] = {}
    seen_objects: set[int] = set()

    width_keys = ("width", "order_width", "strip_width", "from_width", "to_width", "left_width", "right_width")
    thickness_keys = ("thickness", "thick", "order_thickness", "strip_thickness", "from_thickness", "to_thickness", "left_thickness", "right_thickness")
    temp_min_keys = ("temp_min", "temp_low", "anneal_temp_min", "temperature_min", "from_temp_min", "to_temp_min", "left_temp_min", "right_temp_min")
    temp_max_keys = ("temp_max", "temp_high", "anneal_temp_max", "temperature_max", "from_temp_max", "to_temp_max", "left_temp_max", "right_temp_max")
    steel_group_keys = ("steel_group", "grade_group", "steel_grade_group", "from_steel_group", "to_steel_group", "left_steel_group", "right_steel_group")
    line_capability_keys = ("line_capability", "roll_capability", "capability")

    def first_present(row: dict, keys: tuple[str, ...]):
        for key in keys:
            if key in row:
                val = row.get(key)
                if val is not None and not (isinstance(val, str) and not val.strip()):
                    return val
        return None

    def add_spec(row: dict, oid_key: str = "order_id", prefix: str = "") -> None:
        oid = _txt(row.get(oid_key, ""))
        if not oid:
            return
        prefixed = {str(key)[len(prefix):]: val for key, val in row.items() if prefix and str(key).startswith(prefix)}
        merged = {**row, **prefixed}
        width = _as_float_or_none(first_present(merged, width_keys))
        thickness = _as_float_or_none(first_present(merged, thickness_keys))
        temp_min = _as_float_or_none(first_present(merged, temp_min_keys))
        temp_max = _as_float_or_none(first_present(merged, temp_max_keys))
        steel_group = _txt(first_present(merged, steel_group_keys) or "")
        line_capability = _txt(first_present(merged, line_capability_keys) or "")
        if width is None and thickness is None and temp_min is None and temp_max is None and not steel_group and not line_capability:
            return
        cur = specs.setdefault(oid, {"order_id": oid})
        if width is not None:
            cur["width"] = width
        if thickness is not None:
            cur["thickness"] = thickness
        if temp_min is not None:
            cur["temp_min"] = temp_min
        if temp_max is not None:
            cur["temp_max"] = temp_max
        if steel_group:
            cur["steel_group"] = steel_group
        if line_capability:
            cur["line_capability"] = line_capability

    def scan_df(df: pd.DataFrame) -> None:
        if not isinstance(df, pd.DataFrame) or df.empty:
            return
        obj_id = id(df)
        if obj_id in seen_objects:
            return
        seen_objects.add(obj_id)
        cols = {str(c) for c in df.columns}
        if {"order_id", "width", "thickness", "temp_min", "temp_max", "temp_low", "temp_high", "steel_group", "line_capability"}.intersection(cols):
            for row in df.to_dict("records"):
                add_spec(row)
        endpoint_specs = [
            ("from_order_id", "from_"),
            ("to_order_id", "to_"),
            ("source_order_id", "source_"),
            ("target_order_id", "target_"),
            ("left_order_id", "left_"),
            ("right_order_id", "right_"),
        ]
        for oid_key, prefix in endpoint_specs:
            if oid_key in cols:
                for row in df.to_dict("records"):
                    add_spec(row, oid_key=oid_key, prefix=prefix)

    def scan_value(value, depth: int = 0) -> None:
        if depth > 5 or value is None:
            return
        obj_id = id(value)
        if obj_id in seen_objects and not isinstance(value, pd.DataFrame):
            return
        if isinstance(value, pd.DataFrame):
            scan_df(value)
            return
        if isinstance(value, dict):
            seen_objects.add(obj_id)
            if any(key in value for key in ("order_id", "from_order_id", "to_order_id", "left_order_id", "right_order_id")):
                add_spec(value)
                for oid_key, prefix in (
                    ("from_order_id", "from_"),
                    ("to_order_id", "to_"),
                    ("left_order_id", "left_"),
                    ("right_order_id", "right_"),
                    ("source_order_id", "source_"),
                    ("target_order_id", "target_"),
                ):
                    if oid_key in value:
                        add_spec(value, oid_key=oid_key, prefix=prefix)
            for nested in value.values():
                scan_value(nested, depth + 1)
            return
        if isinstance(value, (list, tuple)):
            seen_objects.add(obj_id)
            for item in value:
                scan_value(item, depth + 1)

    scan_df(orders_df) if isinstance(orders_df, pd.DataFrame) else None
    scan_df(tpl_df)
    scan_value(transition_pack)
    return specs


def _build_template_pair_rows_by_key(tpl_df: pd.DataFrame, line: str) -> dict[tuple[str, str], list[dict]]:
    out: dict[tuple[str, str], list[dict]] = {}
    if not isinstance(tpl_df, pd.DataFrame) or tpl_df.empty:
        return out
    for row in tpl_df.to_dict("records"):
        if "line" in row and str(row.get("line", "")) != str(line):
            continue
        src, src_field = _extract_bridge_endpoint(row, "from")
        dst, dst_field = _extract_bridge_endpoint(row, "to")
        if not src_field or not dst_field:
            continue
        out.setdefault((str(src), str(dst)), []).append(row)
    return out


def _order_pair_meta(order_specs_by_id: dict[str, dict], oid: str) -> dict:
    spec = dict(order_specs_by_id.get(str(oid), {}) or {})
    width = _as_float_or_none(spec.get("width"))
    thickness = _as_float_or_none(spec.get("thickness"))
    temp_low = _as_float_or_none(spec.get("temp_min", spec.get("temp_low")))
    temp_high = _as_float_or_none(spec.get("temp_max", spec.get("temp_high")))
    steel_group = _txt(spec.get("steel_group", ""))
    context_present = bool(
        spec
        and width is not None
        and thickness is not None
        and temp_low is not None
        and temp_high is not None
        and bool(steel_group)
    )
    return {
        "order_id": str(oid),
        "width": width,
        "thickness": thickness,
        "temp_low": temp_low,
        "temp_high": temp_high,
        "steel_group": steel_group,
        "line_capability": _txt(spec.get("line_capability", "")),
        "context_present": bool(context_present),
    }


def _quick_thickness_feasible(left_ctx: dict, right_ctx: dict) -> tuple[bool, str, dict]:
    left_t = _as_float_or_none((left_ctx or {}).get("thickness"))
    right_t = _as_float_or_none((right_ctx or {}).get("thickness"))
    if left_t is None or right_t is None:
        return True, "ORDER_CONTEXT_MISSING", {
            "left_thickness": left_t,
            "right_thickness": right_t,
            "delta_mm": None,
            "delta_ratio": None,
        }
    delta_mm = abs(float(right_t) - float(left_t))
    delta_ratio = delta_mm / max(abs(float(left_t)), 1e-9)
    ok = bool(_thick_ok(float(left_t), float(right_t)))
    return ok, "OK" if ok else "THICKNESS_RULE_FAIL", {
        "left_thickness": float(left_t),
        "right_thickness": float(right_t),
        "delta_mm": float(delta_mm),
        "delta_ratio": float(delta_ratio),
    }


def _quick_pair_rank_metrics(order_context_by_id: dict[str, dict], left_oid: str, right_oid: str, rule=None) -> dict:
    left_meta = _order_pair_meta(order_context_by_id, str(left_oid))
    right_meta = _order_pair_meta(order_context_by_id, str(right_oid))
    prefilter = _quick_bridge_rule_prefilter(left_meta, right_meta, rule=rule)
    thick_detail = dict(prefilter.get("thickness_detail", {}) or {})
    lw = _as_float_or_none(left_meta.get("width"))
    rw = _as_float_or_none(right_meta.get("width"))
    width_distance = abs(float(lw) - float(rw)) if lw is not None and rw is not None else 999999.0
    thickness_distance = float(prefilter.get("thickness_gap") or 0.0) if prefilter.get("thickness_gap") is not None else 999999.0
    return {
        **prefilter,
        "thickness_prefilter_pass": bool(prefilter.get("thickness_pass", False)),
        "thickness_reason": "OK" if bool(prefilter.get("thickness_pass", False)) else "THICKNESS_RULE_FAIL",
        "thickness_distance_mm": float(thickness_distance),
        "thickness_delta_ratio": thick_detail.get("delta_ratio"),
        "width_distance": float(width_distance),
        "left_thickness": thick_detail.get("left_thickness"),
        "right_thickness": thick_detail.get("right_thickness"),
    }


def _quick_bridge_rule_prefilter(left_ctx: dict, right_ctx: dict, rule) -> dict:
    left = dict(left_ctx or {})
    right = dict(right_ctx or {})
    fail_reasons: list[str] = []

    thick_ok, thick_reason, thick_detail = _quick_thickness_feasible(left, right)
    thickness_pass = bool(thick_ok)
    if not thickness_pass and thick_reason != "ORDER_CONTEXT_MISSING":
        fail_reasons.append("THICKNESS_RULE_FAIL")

    lw = _as_float_or_none(left.get("width"))
    rw = _as_float_or_none(right.get("width"))
    width_delta = None
    width_pass = True
    if lw is None or rw is None:
        width_pass = False
        fail_reasons.append("ORDER_CONTEXT_MISSING")
    else:
        width_delta = float(rw) - float(lw)
        reverse_limit = float(getattr(rule, "real_reverse_step_max_mm", 0.0) or 0.0) if rule is not None else 0.0
        drop_limit = float(getattr(rule, "max_width_drop", 999999.0) or 999999.0) if rule is not None else 999999.0
        if width_delta > reverse_limit or -width_delta > drop_limit:
            width_pass = False
            fail_reasons.append("WIDTH_RULE_FAIL")

    lmin = _as_float_or_none(left.get("temp_min", left.get("temp_low")))
    lmax = _as_float_or_none(left.get("temp_max", left.get("temp_high")))
    rmin = _as_float_or_none(right.get("temp_min", right.get("temp_low")))
    rmax = _as_float_or_none(right.get("temp_max", right.get("temp_high")))
    temp_overlap = None
    temp_pass = True
    if None in (lmin, lmax, rmin, rmax):
        temp_pass = False
        fail_reasons.append("ORDER_CONTEXT_MISSING")
    else:
        temp_overlap = _temp_overlap_len(float(lmin), float(lmax), float(rmin), float(rmax))
        min_overlap = float(getattr(rule, "min_temp_overlap_real_real", 0.0) or 0.0) if rule is not None else 0.0
        if float(temp_overlap) < min_overlap:
            temp_pass = False
            fail_reasons.append("TEMP_OVERLAP_FAIL")

    lg = _txt(left.get("steel_group", "")).upper()
    rg = _txt(right.get("steel_group", "")).upper()
    group_transition_required = bool(lg and rg and lg != rg)
    group_pass = True
    if not lg or not rg:
        group_pass = False
        fail_reasons.append("ORDER_CONTEXT_MISSING")
    elif group_transition_required and (not _is_pc(lg)) and (not _is_pc(rg)):
        group_pass = False
        fail_reasons.append("GROUP_SWITCH_FAIL")

    fail_reasons = list(dict.fromkeys(fail_reasons))
    non_context_failures = [r for r in fail_reasons if r != "ORDER_CONTEXT_MISSING"]
    dominant_fail_reason = non_context_failures[0] if non_context_failures else (fail_reasons[0] if fail_reasons else "OK")
    thickness_gap = thick_detail.get("delta_mm")
    width_abs = abs(float(width_delta)) if width_delta is not None else None
    temp_penalty = 0.0
    if temp_overlap is None:
        temp_penalty = 5000.0
    elif not temp_pass:
        temp_penalty = 3000.0 + abs(float(temp_overlap)) * 10.0
    else:
        temp_penalty = max(0.0, 100.0 - float(temp_overlap))
    thickness_penalty = (10000.0 + float(thickness_gap or 0.0) * 1000.0) if not thickness_pass else float(thickness_gap or 0.0) * 100.0
    width_penalty = (5000.0 + float(width_abs or 0.0)) if not width_pass else float(width_abs or 0.0)
    group_penalty = 4000.0 if not group_pass else (50.0 if group_transition_required else 0.0)
    hard_prefilter_fail_penalty = 100000.0 * len(non_context_failures)
    return {
        "pass": bool(not fail_reasons),
        "thickness_pass": bool(thickness_pass),
        "width_pass": bool(width_pass),
        "temp_pass": bool(temp_pass),
        "group_pass": bool(group_pass),
        "dominant_fail_reason": str(dominant_fail_reason),
        "fail_reasons": fail_reasons,
        "thickness_gap": thickness_gap,
        "width_delta": width_delta,
        "temp_overlap": temp_overlap,
        "group_transition_required": bool(group_transition_required),
        "thickness_detail": thick_detail,
        "hard_prefilter_fail_penalty": float(hard_prefilter_fail_penalty),
        "thickness_penalty": float(thickness_penalty),
        "width_penalty": float(width_penalty),
        "temp_penalty": float(temp_penalty),
        "group_penalty": float(group_penalty),
    }


def _rank_pair_keys_by_rules(
    keys: list[tuple[str, str, str]],
    order_context_by_id: dict[str, dict],
    adjustment_cost_by_pair: dict[tuple[str, str], int] | None = None,
    rule=None,
) -> list[tuple[str, str, str]]:
    def sort_key(item: tuple[str, str, str]):
        _, left_oid, right_oid = item
        metrics = _quick_pair_rank_metrics(order_context_by_id, str(left_oid), str(right_oid), rule=rule)
        adjust_cost = int((adjustment_cost_by_pair or {}).get((str(left_oid), str(right_oid)), 0) or 0)
        rank_score = (
            float(metrics.get("hard_prefilter_fail_penalty", 0.0))
            + float(metrics.get("thickness_penalty", 0.0))
            + float(metrics.get("width_penalty", 0.0))
            + float(metrics.get("temp_penalty", 0.0))
            + float(metrics.get("group_penalty", 0.0))
            + int(adjust_cost)
        )
        return (
            0 if bool(metrics.get("pass", False)) else 1,
            len(list(metrics.get("fail_reasons", []) or [])),
            0 if bool(metrics.get("group_pass", False)) else 1,
            0 if bool(metrics.get("temp_pass", False)) else 1,
            0 if bool(metrics.get("width_pass", False)) else 1,
            0 if bool(metrics.get("thickness_pass", False)) else 1,
            float(rank_score),
        )

    return sorted(list(dict.fromkeys(keys)), key=sort_key)


def _rank_band_orders_for_thickness(
    *,
    order_context_by_id: dict[str, dict],
    left_band: list[str],
    right_band: list[str],
    line: str,
    block_id: int | None,
    split_id: int | None,
    rule=None,
) -> tuple[list[str], list[str]]:
    left_generation = list(reversed([str(v) for v in left_band]))
    right_generation = [str(v) for v in right_band]
    if not left_generation or not right_generation:
        return left_generation, right_generation

    left_tail = left_generation[0]
    right_generation = sorted(
        right_generation,
        key=lambda oid: (
            0 if _quick_pair_rank_metrics(order_context_by_id, left_tail, oid, rule=rule).get("pass") else 1,
            len(list(_quick_pair_rank_metrics(order_context_by_id, left_tail, oid, rule=rule).get("fail_reasons", []) or [])),
            -float(_quick_pair_rank_metrics(order_context_by_id, left_tail, oid, rule=rule).get("temp_overlap") or -999999.0),
            0 if _quick_pair_rank_metrics(order_context_by_id, left_tail, oid, rule=rule).get("group_pass") else 1,
            float(_quick_pair_rank_metrics(order_context_by_id, left_tail, oid, rule=rule).get("width_distance", 999999.0)),
            float(_quick_pair_rank_metrics(order_context_by_id, left_tail, oid, rule=rule).get("thickness_distance_mm", 999999.0)),
        ),
    )
    left_generation = sorted(
        left_generation,
        key=lambda oid: min(
            (
                0 if _quick_pair_rank_metrics(order_context_by_id, oid, right_oid, rule=rule).get("pass") else 1,
                len(list(_quick_pair_rank_metrics(order_context_by_id, oid, right_oid, rule=rule).get("fail_reasons", []) or [])),
                -float(_quick_pair_rank_metrics(order_context_by_id, oid, right_oid, rule=rule).get("temp_overlap") or -999999.0),
                0 if _quick_pair_rank_metrics(order_context_by_id, oid, right_oid, rule=rule).get("group_pass") else 1,
                float(_quick_pair_rank_metrics(order_context_by_id, oid, right_oid, rule=rule).get("width_distance", 999999.0)),
                float(_quick_pair_rank_metrics(order_context_by_id, oid, right_oid, rule=rule).get("thickness_distance_mm", 999999.0)),
            )
            for right_oid in right_generation
        ),
    )
    ranked_preview = []
    for right_oid in right_generation[:8]:
        meta = _order_pair_meta(order_context_by_id, right_oid)
        metrics = _quick_pair_rank_metrics(order_context_by_id, left_tail, right_oid, rule=rule)
        ranked_preview.append((
            right_oid,
            bool(metrics.get("pass", False)),
            list(metrics.get("fail_reasons", []) or []),
            metrics.get("thickness_gap"),
            metrics.get("width_delta"),
            metrics.get("temp_overlap"),
            bool(metrics.get("group_pass", False)),
        ))
    print(
        f"[APS][REPAIR_BRIDGE_BAND_ORDERING] line={line}, block_id={block_id}, split_id={split_id}, "
        f"left_tail={left_tail}, right_band_ranked={ranked_preview}"
    )
    return left_generation, right_generation


def _pair_context_changed_fields(pre_context: dict, extract_context: dict) -> list[str]:
    changed: list[str] = []
    keys = sorted(set(pre_context.keys()) | set(extract_context.keys()))
    for key in keys:
        if pre_context.get(key) != extract_context.get(key):
            changed.append(str(key))
    return changed


def _evaluate_pair_violation(
    *,
    left_order: str,
    right_order: str,
    line: str,
    template_lookup: dict[tuple[str, str], list[dict]],
    direct_edges: set[tuple[str, str]],
    real_edges: set[tuple[str, str]],
    adjacency_ok: bool,
    order_context_by_id: dict[str, dict],
    rule,
    lookup_key_mode: str = "(line,from_order_id,to_order_id)",
) -> PairEvalResult:
    left_oid = str(left_order)
    right_oid = str(right_order)
    pair = (left_oid, right_oid)
    line_key = (str(line), left_oid, right_oid)
    rows = list(template_lookup.get(pair, []) or [])
    template_found = bool(rows or pair in direct_edges or pair in real_edges)
    payload_present = any(not _bridge_payload_empty(row) for row in rows) if rows else bool(pair in direct_edges or pair in real_edges)
    template_ok = bool(template_found)
    left_meta = _order_pair_meta(order_context_by_id, left_oid)
    right_meta = _order_pair_meta(order_context_by_id, right_oid)
    context_ok = bool(left_meta.get("context_present") and right_meta.get("context_present"))
    effective_adjacency_ok = bool(adjacency_ok and context_ok)
    context_snapshot = {
        "line": str(line),
        "from_order_id": left_oid,
        "to_order_id": right_oid,
        "lookup_key_mode": str(lookup_key_mode),
        "template_key": line_key,
        "line_row_found": bool(rows),
        "template_found": bool(template_found),
        "payload_present": bool(payload_present),
        "context_ok": bool(context_ok),
        "left_meta": left_meta,
        "right_meta": right_meta,
    }

    sub_reasons: list[str] = []
    reason = "OK"
    if not context_ok:
        sub_reasons.append("ORDER_CONTEXT_MISSING")
    if not context_ok:
        reason = "ORDER_CONTEXT_MISSING"
    elif not template_found:
        reason = "TEMPLATE_KEY_MISSING"
    elif not rows and (pair in direct_edges or pair in real_edges):
        reason = "LINE_CONTEXT_MISMATCH"
    elif rows and not payload_present:
        reason = "TEMPLATE_PAYLOAD_EMPTY"
    elif not adjacency_ok:
        reason = "ANCHOR_PAIR_NOT_ADJACENT"
        sub_reasons.append("ANCHOR_NOT_ADJACENT")
    else:
        rule_reason, rule_sub_reasons = _classify_pair_rule_failure(
            left_oid,
            right_oid,
            template_known=template_found,
            order_specs_by_id=order_context_by_id,
            rule=rule,
        )
        if rule_reason != "UNKNOWN_PAIR_INVALID":
            reason = rule_reason
            sub_reasons.extend(rule_sub_reasons)
        elif rule_sub_reasons:
            reason = "MULTI_RULE_FAIL" if len(rule_sub_reasons) > 1 else rule_sub_reasons[0]
            sub_reasons.extend(rule_sub_reasons)

    sub_reasons = list(dict.fromkeys([str(v) for v in sub_reasons if str(v)]))
    if reason == "OK" and sub_reasons:
        reason = "MULTI_RULE_FAIL" if len(sub_reasons) > 1 else sub_reasons[0]
    if reason == "UNKNOWN_PAIR_INVALID" and sub_reasons:
        reason = "MULTI_RULE_FAIL" if len(sub_reasons) > 1 else sub_reasons[0]
    business_ok = reason == "OK"
    return PairEvalResult(
        template_ok=bool(template_ok),
        adjacency_ok=bool(effective_adjacency_ok and business_ok),
        context_ok=bool(context_ok),
        reason=str(reason),
        sub_reasons=sub_reasons,
        template_key=line_key,
        context_snapshot=context_snapshot,
    )


def _print_pair_audit(
    tag: str,
    *,
    candidate_id: str,
    pair: tuple[str, str],
    line: str,
    eval_result: PairEvalResult,
) -> None:
    print(
        f"[APS][{tag}] candidate_id={candidate_id}, pair=({pair[0]},{pair[1]}), "
        f"line={line}, template_ok={bool(eval_result.template_ok)}, "
        f"adjacency_ok={bool(eval_result.adjacency_ok)}, "
        f"context_ok={bool(eval_result.context_ok)}, "
        f"reason={eval_result.reason}, template_key={eval_result.template_key}, "
        f"context={eval_result.context_snapshot}"
    )


def _print_anchor_context(
    *,
    candidate_id: str,
    pair: tuple[str, str],
    line: str,
    eval_result: PairEvalResult,
) -> None:
    ctx = dict(eval_result.context_snapshot)
    print(
        f"[APS][REPAIR_BRIDGE_ANCHOR_CONTEXT] candidate_id={candidate_id}, "
        f"pair=({pair[0]},{pair[1]}), line={line}, "
        f"from_order_id={pair[0]}, to_order_id={pair[1]}, "
        f"lookup_key_mode={ctx.get('lookup_key_mode', '')}, "
        f"left_meta={ctx.get('left_meta', {})}, right_meta={ctx.get('right_meta', {})}, "
        f"template_key={ctx.get('template_key', '')}, "
        f"template_found={bool(ctx.get('template_found', False))}, "
        f"payload_present={bool(ctx.get('payload_present', False))}"
    )


def _print_order_context_miss(
    *,
    candidate_id: str,
    pair: tuple[str, str],
    eval_result: PairEvalResult,
) -> None:
    ctx = dict(eval_result.context_snapshot)
    left_ok = bool(dict(ctx.get("left_meta", {}) or {}).get("context_present", False))
    right_ok = bool(dict(ctx.get("right_meta", {}) or {}).get("context_present", False))
    if left_ok and right_ok:
        return
    missing_side = "both"
    if left_ok and not right_ok:
        missing_side = "right"
    elif right_ok and not left_ok:
        missing_side = "left"
    print(
        f"[APS][REPAIR_BRIDGE_ORDER_CONTEXT_MISS] candidate_id={candidate_id}, "
        f"pair=({pair[0]},{pair[1]}), missing_side={missing_side}"
    )


def _print_order_context_lookup_sample(order_context_by_id: dict[str, dict], order_ids: list[str], limit: int = 8) -> None:
    seen: set[str] = set()
    printed = 0
    for oid in order_ids:
        oid = str(oid)
        if oid in seen:
            continue
        seen.add(oid)
        meta = _order_pair_meta(order_context_by_id, oid)
        print(
            f"[APS][ORDER_CONTEXT_LOOKUP] order_id={oid}, found={bool(meta.get('context_present', False))}, "
            f"width={meta.get('width')}, thickness={meta.get('thickness')}, "
            f"temp_low={meta.get('temp_low')}, temp_high={meta.get('temp_high')}, "
            f"steel_group={meta.get('steel_group', '')}"
        )
        printed += 1
        if printed >= int(limit):
            break


def _safe_preview(value, max_len: int = 180):
    text = str(value)
    return text if len(text) <= max_len else text[:max_len] + "..."


def _pack_debug_meta(transition_pack) -> dict:
    keys: list[str] = []
    line_keys: list[str] = []
    if isinstance(transition_pack, dict):
        keys = [str(k) for k in transition_pack.keys()]
        for k, v in transition_pack.items():
            if isinstance(v, (dict, pd.DataFrame, list, tuple)):
                line_keys.append(str(k))
    return {
        "transition_pack_type": type(transition_pack).__name__,
        "keys": keys[:30],
        "line_keys": line_keys[:30],
        "has_transition_df": "transition_df" in keys,
        "has_template_df": "template_df" in keys or "templates" in keys,
        "has_bridge_paths": "bridge_paths" in keys,
        "has_real_bridge_edges": "real_bridge_edges" in keys,
        "has_virtual_bridge_edges": "virtual_bridge_edges" in keys,
        "has_transition_lookup": "transition_lookup" in keys,
        "has_edge_lookup": "edge_lookup" in keys,
        "has_path_lookup": "path_lookup" in keys,
    }


def _row_bridge_type(row: dict) -> str:
    for key in ("edge_type", "bridge_type", "path_kind"):
        val = str(row.get(key, "") or "").strip()
        if val:
            return val
    if bool(row.get("is_real_bridge", False)):
        return "REAL_BRIDGE_EDGE"
    if bool(row.get("is_virtual_bridge_family", False)):
        return "VIRTUAL_BRIDGE_FAMILY_EDGE"
    if bool(row.get("is_virtual_bridge", False)):
        return "VIRTUAL_BRIDGE_EDGE"
    return ""


def _is_real_bridge_row(row: dict) -> bool:
    kind = _row_bridge_type(row)
    return kind in {"REAL_BRIDGE_EDGE", "REAL_BRIDGE", "REAL"}


def _is_virtual_bridge_row(row: dict) -> bool:
    """Check if row is a legacy virtual bridge edge (EXCLUDES family edges)."""
    kind = _row_bridge_type(row)
    return kind in {"VIRTUAL_BRIDGE_EDGE", "VIRTUAL_BRIDGE", "VIRTUAL"}


def _is_virtual_bridge_family_row(row: dict) -> bool:
    """Check if row is a VIRTUAL_BRIDGE_FAMILY_EDGE edge."""
    kind = _row_bridge_type(row)
    return kind == "VIRTUAL_BRIDGE_FAMILY_EDGE"


def _extract_bridge_endpoint(row: dict, side: str) -> tuple[str, str]:
    if side == "from":
        candidates = ("from_order_id", "source_order_id", "from_id", "left_order_id", "src_order_id")
    else:
        candidates = ("to_order_id", "target_order_id", "to_id", "right_order_id", "dst_order_id")
    for key in candidates:
        val = row.get(key, None)
        if val is not None and str(val).strip() != "":
            return str(val), key
    return "", ""


def _bridge_payload_empty(row: dict) -> bool:
    for key in ("path", "bridge_path", "bridge_orders", "virtual_steps", "bridge_order_id", "real_bridge_order_id"):
        val = row.get(key, None)
        if val is None:
            continue
        if isinstance(val, (list, tuple, dict)) and len(val) > 0:
            return False
        if not isinstance(val, (list, tuple, dict)) and str(val).strip() not in {"", "[]", "{}"}:
            return False
    return True


def _bridge_rows_for_line(tpl_df: pd.DataFrame, line: str | None = None) -> list[dict]:
    if tpl_df.empty:
        return []
    rows = tpl_df.to_dict("records")
    out: list[dict] = []
    for row in rows:
        if line is not None and "line" in row and str(row.get("line", "")) != str(line):
            continue
        if str(row.get("edge_type", "") or "") == "REAL_BRIDGE_EDGE":
            out.append(row)
    return out


def _virtual_bridge_rows_for_line(tpl_df: pd.DataFrame, line: str | None = None) -> list[dict]:
    if tpl_df.empty:
        return []
    rows = tpl_df.to_dict("records")
    out: list[dict] = []
    for row in rows:
        if line is not None and "line" in row and str(row.get("line", "")) != str(line):
            continue
        if _is_virtual_bridge_row(row):
            out.append(row)
    return out


def _virtual_bridge_family_rows_for_line(tpl_df: pd.DataFrame, line: str | None = None) -> list[dict]:
    """Get VIRTUAL_BRIDGE_FAMILY_EDGE rows for a given line (or all lines if line is None)."""
    if tpl_df.empty:
        return []
    rows = tpl_df.to_dict("records")
    out: list[dict] = []
    for row in rows:
        if line is not None and "line" in row and str(row.get("line", "")) != str(line):
            continue
        if _is_virtual_bridge_family_row(row):
            out.append(row)
    return out


def _print_bridge_schema_samples(tpl_df: pd.DataFrame, line: str, max_samples: int = 3) -> dict:
    real_rows = _bridge_rows_for_line(tpl_df, line)
    virtual_rows = _virtual_bridge_rows_for_line(tpl_df, line)
    columns = [str(c) for c in tpl_df.columns] if not tpl_df.empty else []
    print(f"[APS][REPAIR_BRIDGE_SCHEMA] line={line}, columns={columns}")
    for idx, row in enumerate(real_rows[:max_samples]):
        sample = {
            "from_order_id": row.get("from_order_id", None),
            "source_order_id": row.get("source_order_id", None),
            "from_id": row.get("from_id", None),
            "to_order_id": row.get("to_order_id", None),
            "target_order_id": row.get("target_order_id", None),
            "to_id": row.get("to_id", None),
            "edge_type": row.get("edge_type", None),
            "bridge_type": row.get("bridge_type", None),
            "path_kind": row.get("path_kind", None),
            "is_real_bridge": row.get("is_real_bridge", None),
            "is_virtual_bridge": row.get("is_virtual_bridge", None),
            "bridge_count": row.get("bridge_count", None),
            "virtual_count": row.get("virtual_count", None),
            "line": row.get("line", None),
            "path": _safe_preview(row.get("path", "")),
            "bridge_path": _safe_preview(row.get("bridge_path", "")),
            "bridge_orders": _safe_preview(row.get("bridge_orders", "")),
        }
        print(f"[APS][REPAIR_BRIDGE_SAMPLE] line={line}, sample_{idx}={sample}")
    from_fields = {field for row in real_rows[:50] for _, field in [_extract_bridge_endpoint(row, "from")] if field}
    to_fields = {field for row in real_rows[:50] for _, field in [_extract_bridge_endpoint(row, "to")] if field}
    return {
        "raw_real_rows": int(len(real_rows)),
        "raw_virtual_rows": int(len(virtual_rows)),
        "columns": columns,
        "from_fields": sorted(from_fields),
        "to_fields": sorted(to_fields),
        "field_name_mismatch": bool(real_rows and (not from_fields or not to_fields)),
    }


def _count_real_bridge_matches(
    lookup: dict[tuple[str, str, str], list[dict]],
    boundary_keys: set[tuple[str, str, str]],
) -> dict:
    matched_rows: list[dict] = []
    payload_empty = 0
    endpoint_missing = 0
    for key in boundary_keys:
        rows = lookup.get(key, [])
        for row in rows:
            src, src_field = _extract_bridge_endpoint(row, "from")
            dst, dst_field = _extract_bridge_endpoint(row, "to")
            if not src_field or not dst_field:
                endpoint_missing += 1
                continue
            matched_rows.append(row)
            if _bridge_payload_empty(row):
                payload_empty += 1
    return {
        "matched_rows": int(len(matched_rows)),
        "line_mismatch": 0,
        "block_membership_mismatch": 0,
        "endpoint_missing": int(endpoint_missing),
        "payload_empty": int(payload_empty),
        "lookup_unique_keys": int(len(lookup)),
    }


def _build_real_bridge_lookup(real_rows: list[dict]) -> dict[tuple[str, str, str], list[dict]]:
    lookup: dict[tuple[str, str, str], list[dict]] = {}
    for row in real_rows:
        if str(row.get("edge_type", "") or "") != "REAL_BRIDGE_EDGE":
            continue
        line = str(row.get("line", ""))
        src, src_field = _extract_bridge_endpoint(row, "from")
        dst, dst_field = _extract_bridge_endpoint(row, "to")
        if not line or not src_field or not dst_field:
            continue
        lookup.setdefault((line, src, dst), []).append(row)
    return lookup


def _build_virtual_bridge_lookup(virtual_rows: list[dict]) -> dict[tuple[str, str, str], list[dict]]:
    lookup: dict[tuple[str, str, str], list[dict]] = {}
    for row in virtual_rows:
        if not _is_virtual_bridge_row(row):
            continue
        line = str(row.get("line", ""))
        src, src_field = _extract_bridge_endpoint(row, "from")
        dst, dst_field = _extract_bridge_endpoint(row, "to")
        if not line or not src_field or not dst_field:
            continue
        lookup.setdefault((line, src, dst), []).append(row)
    return lookup


def _virtual_bridge_tons_from_row(row: dict) -> float:
    for key in ("virtual_tons", "virtual_ton", "bridge_tons", "bridge_ton", "tons", "ton"):
        val = _as_float_or_none((row or {}).get(key))
        if val is not None:
            return float(val)
    tons10 = _as_float_or_none((row or {}).get("virtual_tons10"))
    if tons10 is not None:
        return float(tons10) / 10.0
    return 0.0


def _block_single_point_boundary_keys(block: list[CampaignSegment]) -> list[tuple[str, str, str]]:
    if not block:
        return []
    line = str(block[0].line)
    keys: list[tuple[str, str, str]] = []
    for left, right in zip(block, block[1:]):
        if not left.order_ids or not right.order_ids:
            continue
        keys.append((line, str(left.order_ids[-1]), str(right.order_ids[0])))
    return keys


def _block_boundary_band_keys(
    block: list[CampaignSegment],
    *,
    left_band_k: int,
    right_band_k: int,
    max_pairs_per_split: int,
    order_context_by_id: dict[str, dict] | None = None,
    block_id: int | None = None,
    rule=None,
) -> tuple[list[tuple[str, str, str]], list[dict]]:
    if not block:
        return [], []
    line = str(block[0].line)
    out: list[tuple[str, str, str]] = []
    logs: list[dict] = []
    for split_id, (left, right) in enumerate(zip(block, block[1:]), start=1):
        left_orders = [str(v) for v in left.order_ids]
        right_orders = [str(v) for v in right.order_ids]
        left_band = left_orders[-max(1, int(left_band_k)):]
        right_band = right_orders[: max(1, int(right_band_k))]
        left_generation = list(reversed(left_band))
        right_generation = list(right_band)
        if order_context_by_id:
            left_generation, right_generation = _rank_band_orders_for_thickness(
                order_context_by_id=order_context_by_id,
                left_band=left_band,
                right_band=right_band,
                line=line,
                block_id=block_id,
                split_id=split_id,
                rule=rule,
            )
        pairs: list[tuple[str, str, str]] = []
        for left_distance, left_oid in enumerate(left_generation):
            for right_distance, right_oid in enumerate(right_generation):
                if len(pairs) >= int(max_pairs_per_split):
                    break
                pairs.append((line, str(left_oid), str(right_oid)))
            if len(pairs) >= int(max_pairs_per_split):
                break
        dedup_pairs = list(dict.fromkeys(pairs))
        out.extend(dedup_pairs)
        logs.append(
            {
                "split_id": split_id,
                "left_band": left_band,
                "right_band": right_band,
                "left_generation_order": left_generation,
                "right_generation_order": right_generation,
                "band_pairs": dedup_pairs,
            }
        )
    return list(dict.fromkeys(out)), logs


def _generate_endpoint_adjustments(
    left_seg_orders: list[str],
    right_seg_orders: list[str],
    *,
    left_trim_max: int,
    right_trim_max: int,
    adjustment_limit: int,
    enable_left_trim: bool,
    enable_right_trim: bool,
    order_context_by_id: dict[str, dict] | None = None,
    rule=None,
) -> list[dict]:
    adjustments: list[dict] = []
    seen: set[tuple[tuple[str, ...], tuple[str, ...]]] = set()
    for left_trim in range(0, max(0, int(left_trim_max)) + 1):
        if left_trim > 0 and not enable_left_trim:
            continue
        for right_trim in range(0, max(0, int(right_trim_max)) + 1):
            if right_trim > 0 and not enable_right_trim:
                continue
            adjusted_left = list(left_seg_orders[:-left_trim]) if left_trim > 0 else list(left_seg_orders)
            adjusted_right = list(right_seg_orders[right_trim:]) if right_trim > 0 else list(right_seg_orders)
            if not adjusted_left or not adjusted_right:
                continue
            sig = (tuple(adjusted_left), tuple(adjusted_right))
            if sig in seen:
                continue
            seen.add(sig)
            adjustments.append(
                {
                    "left_trim": int(left_trim),
                    "right_trim": int(right_trim),
                    "adjusted_left_orders": adjusted_left,
                    "adjusted_right_orders": adjusted_right,
                }
            )
    if order_context_by_id:
        adjustments.sort(
            key=lambda item: (
                _quick_pair_rank_metrics(
                    order_context_by_id,
                    str(item.get("adjusted_left_orders", [""])[-1]),
                    str(item.get("adjusted_right_orders", [""])[0]),
                    rule=rule,
                ).get("hard_prefilter_fail_penalty", 0.0)
                + _quick_pair_rank_metrics(
                    order_context_by_id,
                    str(item.get("adjusted_left_orders", [""])[-1]),
                    str(item.get("adjusted_right_orders", [""])[0]),
                    rule=rule,
                ).get("thickness_penalty", 999999.0)
                + _quick_pair_rank_metrics(
                    order_context_by_id,
                    str(item.get("adjusted_left_orders", [""])[-1]),
                    str(item.get("adjusted_right_orders", [""])[0]),
                    rule=rule,
                ).get("width_penalty", 999999.0)
                + _quick_pair_rank_metrics(
                    order_context_by_id,
                    str(item.get("adjusted_left_orders", [""])[-1]),
                    str(item.get("adjusted_right_orders", [""])[0]),
                    rule=rule,
                ).get("temp_penalty", 999999.0)
                + _quick_pair_rank_metrics(
                    order_context_by_id,
                    str(item.get("adjusted_left_orders", [""])[-1]),
                    str(item.get("adjusted_right_orders", [""])[0]),
                    rule=rule,
                ).get("group_penalty", 999999.0),
                int(item.get("left_trim", 0) or 0) + int(item.get("right_trim", 0) or 0),
            )
        )
    return adjustments[: max(1, int(adjustment_limit))]


def _build_adjustment_band_pairs(
    *,
    line: str,
    adjusted_left_orders: list[str],
    adjusted_right_orders: list[str],
    left_band_k: int,
    right_band_k: int,
    max_pairs_per_split: int,
    order_context_by_id: dict[str, dict] | None = None,
    block_id: int | None = None,
    split_id: int | None = None,
    rule=None,
) -> tuple[list[tuple[str, str, str]], list[str], list[str], list[str], list[str]]:
    left_band = [str(v) for v in adjusted_left_orders[-max(1, int(left_band_k)):]]
    right_band = [str(v) for v in adjusted_right_orders[: max(1, int(right_band_k))]]
    left_generation = list(reversed(left_band))
    right_generation = list(right_band)
    if order_context_by_id:
        left_generation, right_generation = _rank_band_orders_for_thickness(
            order_context_by_id=order_context_by_id,
            left_band=left_band,
            right_band=right_band,
            line=str(line),
            block_id=block_id,
            split_id=split_id,
            rule=rule,
        )
    pairs: list[tuple[str, str, str]] = []
    for left_oid in left_generation:
        for right_oid in right_generation:
            if len(pairs) >= int(max_pairs_per_split):
                break
            pairs.append((str(line), str(left_oid), str(right_oid)))
        if len(pairs) >= int(max_pairs_per_split):
            break
    return list(dict.fromkeys(pairs)), left_band, right_band, left_generation, right_generation


def _boundary_band_distance_map(band_logs: list[dict]) -> dict[tuple[str, str], int]:
    distances: dict[tuple[str, str], int] = {}
    for item in band_logs:
        left_order = [str(v) for v in item.get("left_generation_order", [])] or list(reversed([str(v) for v in item.get("left_band", [])]))
        right_order = [str(v) for v in item.get("right_generation_order", [])] or [str(v) for v in item.get("right_band", [])]
        left_distance = {oid: idx for idx, oid in enumerate(left_order)}
        right_distance = {oid: idx for idx, oid in enumerate(right_order)}
        for _, left_oid, right_oid in item.get("band_pairs", []):
            dist = int(left_distance.get(str(left_oid), 999)) + int(right_distance.get(str(right_oid), 999))
            key = (str(left_oid), str(right_oid))
            distances[key] = min(distances.get(key, dist), dist)
    return distances


def _endpoint_adjustment_cost_map(adjustment_logs: list[dict]) -> dict[tuple[str, str], int]:
    costs: dict[tuple[str, str], int] = {}
    for item in adjustment_logs:
        cost = int(item.get("final_cost", item.get("left_trim", 0) + item.get("right_trim", 0)) or 0)
        for _, left_oid, right_oid in item.get("band_pairs", []):
            key = (str(left_oid), str(right_oid))
            costs[key] = min(costs.get(key, cost), cost)
    return costs


def _classify_endpoint_miss(
    *,
    line: str,
    block_id: int,
    boundary_keys: list[tuple[str, str, str]],
    real_bridge_lookup: dict[tuple[str, str, str], list[dict]],
    order_context_by_id: dict[str, dict],
    rule,
) -> dict:
    keys = list(dict.fromkeys(boundary_keys))
    exact_hits = [key for key in keys if key in real_bridge_lookup]
    same_line_graph_keys = [key for key in real_bridge_lookup if str(key[0]) == str(line)]
    boundary_pairs = {(str(k[1]), str(k[2])) for k in keys}
    same_left_or_right = [
        key for key in same_line_graph_keys
        if (str(key[1]), str(key[2])) not in boundary_pairs
        and (str(key[1]) in {p[0] for p in boundary_pairs} or str(key[2]) in {p[1] for p in boundary_pairs})
    ]
    prefilter_results = []
    for _, left_oid, right_oid in keys:
        left_meta = _order_pair_meta(order_context_by_id, str(left_oid))
        right_meta = _order_pair_meta(order_context_by_id, str(right_oid))
        prefilter_results.append(_quick_bridge_rule_prefilter(left_meta, right_meta, rule=rule))
    fail_counts = Counter()
    for item in prefilter_results:
        for reason in list(item.get("fail_reasons", []) or []):
            fail_counts[str(reason)] += 1

    if exact_hits:
        cls = "HAS_ENDPOINT_EDGE"
    elif prefilter_results and all(not bool(item.get("pass", False)) for item in prefilter_results):
        cls = "RULE_PREFILTER_ALL_FAIL"
    elif same_left_or_right:
        cls = "BAND_TOO_NARROW"
    elif not same_line_graph_keys or not exact_hits:
        cls = "TEMPLATE_GRAPH_NO_EDGE"
    elif same_line_graph_keys:
        cls = "LOOKUP_KEY_MISMATCH"
    else:
        cls = "UNKNOWN_ENDPOINT_MISS"

    details = {
        "boundary_key_count": int(len(keys)),
        "exact_hit_count": int(len(exact_hits)),
        "same_line_graph_key_count": int(len(same_line_graph_keys)),
        "near_graph_key_count": int(len(same_left_or_right)),
        "prefilter_fail_counts": dict(fail_counts),
        "sample_boundary_keys": keys[:5],
        "sample_near_graph_keys": same_left_or_right[:5],
    }
    print(
        f"[APS][REPAIR_BRIDGE_ENDPOINT_CLASSIFY] line={line}, block_id={block_id}, "
        f"class={cls}, details={details}"
    )
    return {"class": cls, "details": details}


def _segments_pair_integrity_ok(
    segments: list[CampaignSegment],
    *,
    direct_edges: set[tuple[str, str]],
    real_edges: set[tuple[str, str]],
    max_real_bridge_per_segment: int,
    virtual_edges: set[tuple[str, str]] | None = None,
    max_virtual_bridge_per_segment: int = 0,
) -> tuple[bool, str]:
    for seg in segments:
        real_used = 0
        virtual_used = 0
        for left_oid, right_oid in zip(seg.order_ids, seg.order_ids[1:]):
            pair = (str(left_oid), str(right_oid))
            if pair in direct_edges:
                continue
            if pair in real_edges:
                real_used += 1
                if real_used <= int(max_real_bridge_per_segment):
                    continue
                return False, "BRIDGE_LIMIT_EXCEEDED"
            if pair in (virtual_edges or set()):
                virtual_used += 1
                if virtual_used <= int(max_virtual_bridge_per_segment):
                    continue
                return False, "VIRTUAL_BRIDGE_LIMIT_EXCEEDED"
            return False, "PAIR_NOT_IN_TEMPLATE"
    return True, ""


def _should_apply_improvement(
    *,
    valid_before: int,
    valid_after: int,
    underfilled_before: int,
    underfilled_after: int,
    scheduled_orders_before: int,
    scheduled_orders_after: int,
    integrity_ok: bool,
    pair_integrity_ok: bool,
) -> tuple[bool, str]:
    if int(valid_after) < int(valid_before):
        return False, "VALID_COUNT_REGRESSION"
    if not (int(underfilled_after) < int(underfilled_before) or int(scheduled_orders_after) > int(scheduled_orders_before)):
        return False, "NO_APPLY_GAIN"
    if not integrity_ok:
        return False, "ORDER_INTEGRITY_FAIL"
    if not pair_integrity_ok:
        return False, "PAIR_INTEGRITY_FAIL"
    return True, ""


def _should_apply_virtual_pilot(
    *,
    gain: bool,
    multiplicity_ok: bool,
    pair_integrity_ok: bool,
    virtual_used: int,
    max_virtual_used: int,
    virtual_tons: float,
    max_virtual_tons: float,
) -> tuple[bool, str]:
    if not bool(gain):
        return False, "NO_APPLY_GAIN"
    if not bool(multiplicity_ok):
        return False, "ORDER_INTEGRITY_FAIL"
    if not bool(pair_integrity_ok):
        return False, "PAIR_INTEGRITY_FAIL"
    if int(virtual_used) <= 0:
        return False, "NO_VIRTUAL_BRIDGE_USED"
    if int(virtual_used) > int(max_virtual_used):
        return False, "VIRTUAL_BRIDGE_LIMIT_EXCEEDED"
    if float(virtual_tons) > float(max_virtual_tons) + 1e-6:
        return False, "VIRTUAL_TON_LIMIT_EXCEEDED"
    return True, ""


def _small_block_pilot_score(block_orders: int, block_tons: float, campaign_ton_min: float) -> tuple[bool, float]:
    if int(block_orders) <= 1 or float(block_tons) <= 1e-6:
        return True, 999999.0
    penalty = 0.0
    if int(block_orders) <= 2:
        penalty += 25.0
    soft_ton_floor = float(campaign_ton_min) * 0.5
    if float(block_tons) < soft_ton_floor:
        penalty += max(0.0, soft_ton_floor - float(block_tons))
    return False, float(penalty)


def _record_virtual_pilot_fail_stage(
    diag: dict,
    *,
    line: str,
    block_id: int,
    candidate_id: str,
    fail_stage: str,
    reason: str,
    details: dict,
) -> None:
    stage_counts = diag.setdefault("virtual_pilot_fail_stage_count", {})
    stage_counts[str(fail_stage)] = int(stage_counts.get(str(fail_stage), 0) or 0) + 1
    reject_by_reason = diag.setdefault("virtual_pilot_reject_by_reason_count", {})
    reject_by_reason[str(reason)] = int(reject_by_reason.get(str(reason), 0) or 0) + 1
    print(
        f"[APS][VIRTUAL_BRIDGE_PILOT_FAIL_STAGE] line={line}, block_id={block_id}, "
        f"candidate_id={candidate_id}, fail_stage={fail_stage}, reason={reason}, details={details}"
    )


def _record_virtual_pilot_execution_stage(
    diag: dict,
    *,
    line: str,
    block_id: int,
    candidate_id: str,
    family: str,
    stage: str,
    details: dict | None = None,
) -> None:
    stage_key = str(stage)
    field_by_stage = {
        "SELECTED": "virtual_pilot_selected_candidate_count",
        "DEDUP_KEPT": "virtual_pilot_dedup_kept_count",
        "DEDUP_SKIPPED": "virtual_pilot_dedup_skipped_count",
        "ATTEMPT_STARTED": "virtual_pilot_attempt_started_count",
        "SPEC_ENUM_DONE": "virtual_pilot_spec_enum_done_count",
        "RECUT_ENTERED": "virtual_pilot_recut_entered_count",
        "TON_FILL_ENTERED": "virtual_pilot_ton_fill_entered_count",
        "APPLY_CHECK_ENTERED": "virtual_pilot_apply_check_entered_count",
    }
    field = field_by_stage.get(stage_key)
    if field:
        diag[field] = int(diag.get(field, 0) or 0) + 1
    by_family = diag.setdefault("virtual_pilot_execution_stage_by_family", {})
    fam = str(family or "OTHER")
    fam_counts = by_family.setdefault(fam, {})
    fam_counts[stage_key] = int(fam_counts.get(stage_key, 0) or 0) + 1
    print(
        f"[APS][VIRTUAL_BRIDGE_EXECUTION_STAGE] line={line}, block_id={block_id}, "
        f"candidate_id={candidate_id}, family={fam}, stage={stage_key}, details={details or {}}"
    )


def _record_virtual_post_spec_fail(
    diag: dict,
    *,
    line: str,
    block_id: int,
    candidate_id: str,
    family: str,
    fail_stage: str,
    details: dict | None = None,
) -> None:
    counts = diag.setdefault("virtual_pilot_post_spec_fail_stage_count", {})
    counts[str(fail_stage)] = int(counts.get(str(fail_stage), 0) or 0) + 1
    print(
        f"[APS][VIRTUAL_BRIDGE_POST_SPEC_FAIL] line={line}, block_id={block_id}, "
        f"candidate_id={candidate_id}, family={family}, fail_stage={fail_stage}, details={details or {}}"
    )


def _classify_virtual_pilot_recut_fail(pilot_diag: dict, virtual_used: int, virtual_tons: float) -> tuple[str, str, dict]:
    details = {
        "candidate_cutpoints_total": int(pilot_diag.get("candidate_cutpoints_total", 0) or 0),
        "candidate_cutpoints_ton_window_valid": int(pilot_diag.get("candidate_cutpoints_ton_window_valid", 0) or 0),
        "candidate_cutpoints_pair_valid": int(pilot_diag.get("candidate_cutpoints_pair_valid", 0) or 0),
        "filtered_pair_invalid": int(pilot_diag.get("repair_only_real_bridge_filtered_pair_invalid", 0) or 0),
        "filtered_ton_invalid": int(pilot_diag.get("repair_only_real_bridge_filtered_ton_invalid", 0) or 0),
        "virtual_used": int(virtual_used),
        "virtual_tons": float(virtual_tons),
    }
    if int(details["candidate_cutpoints_total"]) <= 0:
        return "VIRTUAL_SPEC_BUILD_FAIL", "VIRTUAL_SPEC_BUILD_FAIL", details
    if int(details["candidate_cutpoints_ton_window_valid"]) <= 0:
        return "VIRTUAL_TON_BELOW_MIN", "VIRTUAL_TON_BELOW_MIN", details
    if int(details["candidate_cutpoints_pair_valid"]) <= 0:
        return "VIRTUAL_BOTH_TRANSITIONS_INVALID", "VIRTUAL_BOTH_TRANSITIONS_INVALID", details
    if int(virtual_used) <= 0:
        return "VIRTUAL_BOTH_TRANSITIONS_INVALID", "NO_VIRTUAL_BRIDGE_USED", details
    return "VIRTUAL_LOCAL_RECUT_FAIL", "VIRTUAL_LOCAL_RECUT_FAIL", details


def _virtual_pilot_failure_family(candidate: dict) -> str:
    fail_reasons = {str(v) for v in list((candidate or {}).get("fail_reasons", []) or []) if str(v)}
    if {"WIDTH_RULE_FAIL", "GROUP_SWITCH_FAIL"}.issubset(fail_reasons):
        return "WIDTH_GROUP"
    if "THICKNESS_RULE_FAIL" in fail_reasons and fail_reasons.issubset({"THICKNESS_RULE_FAIL"}):
        return "THICKNESS"
    return "OTHER"


def _virtual_pilot_key(line: str, candidate: dict) -> tuple[str, str, str]:
    return (
        str(line),
        str((candidate or {}).get("from_order_id", "") or ""),
        str((candidate or {}).get("to_order_id", "") or ""),
    )


def _virtual_pilot_family_prefilter(family: str, selected_spec: dict | None) -> tuple[bool, str]:
    if selected_spec is None:
        return False, "NO_BOTH_VALID_SPEC"
    left_fails = {str(v) for v in list(selected_spec.get("left_fail_reasons", []) or [])}
    right_fails = {str(v) for v in list(selected_spec.get("right_fail_reasons", []) or [])}
    all_fails = left_fails | right_fails
    if str(family) == "WIDTH_GROUP":
        if "WIDTH_RULE_FAIL" in all_fails or "GROUP_SWITCH_FAIL" in all_fails:
            return False, "WIDTH_GROUP_TARGET_NOT_FIXED"
    elif str(family) == "THICKNESS":
        if "THICKNESS_RULE_FAIL" in all_fails:
            return False, "THICKNESS_TARGET_NOT_FIXED"
    if not (bool(selected_spec.get("left_ok", False)) and bool(selected_spec.get("right_ok", False))):
        return False, "TRANSITION_NOT_BOTH_VALID"
    return True, "OK"


def _enumerate_virtual_pilot_specs(
    left_meta: dict,
    right_meta: dict,
    rule,
    max_specs: int = 5,
    family: str = "OTHER",
) -> tuple[list[dict], dict, dict | None]:
    left = dict(left_meta or {})
    right = dict(right_meta or {})
    lw = _as_float_or_none(left.get("width"))
    rw = _as_float_or_none(right.get("width"))
    lt = _as_float_or_none(left.get("thickness"))
    rt = _as_float_or_none(right.get("thickness"))
    llow = _as_float_or_none(left.get("temp_low", left.get("temp_min")))
    lhigh = _as_float_or_none(left.get("temp_high", left.get("temp_max")))
    rlow = _as_float_or_none(right.get("temp_low", right.get("temp_min")))
    rhigh = _as_float_or_none(right.get("temp_high", right.get("temp_max")))
    lg = _txt(left.get("steel_group", ""))
    rg = _txt(right.get("steel_group", ""))
    family = str(family or "OTHER")
    widths = []
    thicknesses = []
    if family == "WIDTH_GROUP":
        if lw is not None and rw is not None:
            widths.extend([min(float(lw), float(rw)), (float(lw) + float(rw)) / 2.0, float(rw), float(lw)])
        elif lw is not None:
            widths.append(float(lw))
        elif rw is not None:
            widths.append(float(rw))
        if lt is not None and rt is not None:
            thicknesses.extend([(float(lt) + float(rt)) / 2.0, float(lt), float(rt)])
        elif lt is not None:
            thicknesses.append(float(lt))
        elif rt is not None:
            thicknesses.append(float(rt))
    elif family == "THICKNESS":
        if lw is not None:
            widths.append(float(lw))
        if rw is not None:
            widths.append(float(rw))
        if lw is not None and rw is not None:
            widths.append((float(lw) + float(rw)) / 2.0)
        if lt is not None and rt is not None:
            thicknesses.extend([
                (float(lt) * 2.0 + float(rt)) / 3.0,
                (float(lt) + float(rt)) / 2.0,
                (float(lt) + float(rt) * 2.0) / 3.0,
                float(lt),
                float(rt),
            ])
        elif lt is not None:
            thicknesses.append(float(lt))
        elif rt is not None:
            thicknesses.append(float(rt))
    else:
        if lw is not None:
            widths.append(float(lw))
        if rw is not None:
            widths.append(float(rw))
        if lw is not None and rw is not None:
            widths.append((float(lw) + float(rw)) / 2.0)
        if lt is not None:
            thicknesses.append(float(lt))
        if rt is not None:
            thicknesses.append(float(rt))
        if lt is not None and rt is not None:
            thicknesses.extend([(float(lt) * 2.0 + float(rt)) / 3.0, (float(lt) + float(rt)) / 2.0, (float(lt) + float(rt) * 2.0) / 3.0])
    temp_windows: list[tuple[float | None, float | None]] = []
    if None not in (llow, lhigh, rlow, rhigh):
        overlap_low = max(float(llow), float(rlow))
        overlap_high = min(float(lhigh), float(rhigh))
        if overlap_high >= overlap_low:
            temp_windows.append((overlap_low, overlap_high))
            mid = (overlap_low + overlap_high) / 2.0
            temp_windows.append((max(overlap_low, mid - 5.0), min(overlap_high, mid + 5.0)))
        temp_windows.append(((float(llow) + float(rlow)) / 2.0, (float(lhigh) + float(rhigh)) / 2.0))
    else:
        temp_windows.append((llow if llow is not None else rlow, lhigh if lhigh is not None else rhigh))
    if family == "WIDTH_GROUP":
        groups = [g for g in dict.fromkeys([lg if _is_pc(lg) else "", rg if _is_pc(rg) else "", lg, rg]) if g]
    elif family == "THICKNESS":
        groups = [g for g in dict.fromkeys([lg if lg == rg else "", lg, rg]) if g]
    else:
        groups = [g for g in dict.fromkeys([lg, rg]) if g]
    if not groups:
        groups = [""]
    specs: list[dict] = []
    for width in dict.fromkeys([round(v, 6) for v in widths if v is not None]):
        for thickness in dict.fromkeys([round(v, 6) for v in thicknesses if v is not None]):
            for temp_low, temp_high in temp_windows:
                for group in groups:
                    spec = {
                        "width": float(width),
                        "thickness": float(thickness),
                        "temp_low": temp_low,
                        "temp_high": temp_high,
                        "temp_min": temp_low,
                        "temp_max": temp_high,
                        "steel_group": group,
                        "context_present": True,
                    }
                    left_eval = _quick_bridge_rule_prefilter(left, spec, rule=rule)
                    right_eval = _quick_bridge_rule_prefilter(spec, right, rule=rule)
                    spec["left_ok"] = bool(left_eval.get("pass", False))
                    spec["right_ok"] = bool(right_eval.get("pass", False))
                    spec["left_fail_reasons"] = list(left_eval.get("fail_reasons", []) or [])
                    spec["right_fail_reasons"] = list(right_eval.get("fail_reasons", []) or [])
                    specs.append(spec)
                    if len(specs) >= int(max_specs):
                        break
                if len(specs) >= int(max_specs):
                    break
            if len(specs) >= int(max_specs):
                break
        if len(specs) >= int(max_specs):
            break
    left_valid = sum(1 for spec in specs if bool(spec.get("left_ok", False)))
    right_valid = sum(1 for spec in specs if bool(spec.get("right_ok", False)))
    both_valid = [spec for spec in specs if bool(spec.get("left_ok", False)) and bool(spec.get("right_ok", False))]
    diag = {
        "specs_tested": int(len(specs)),
        "left_valid_count": int(left_valid),
        "right_valid_count": int(right_valid),
        "both_valid_count": int(len(both_valid)),
    }
    selected = both_valid[0] if both_valid else None
    return specs, diag, selected


def _classify_virtual_transition_spec_failure(specs: list[dict], enum_diag: dict) -> tuple[str, str, dict]:
    left_count = int(enum_diag.get("left_valid_count", 0) or 0)
    right_count = int(enum_diag.get("right_valid_count", 0) or 0)
    details = {
        "specs_tested": int(enum_diag.get("specs_tested", 0) or 0),
        "left_valid_count": left_count,
        "right_valid_count": right_count,
        "both_valid_count": int(enum_diag.get("both_valid_count", 0) or 0),
        "sample_specs": specs[:3],
    }
    if left_count <= 0 and right_count <= 0:
        return "VIRTUAL_BOTH_TRANSITIONS_INVALID", "VIRTUAL_BOTH_TRANSITIONS_INVALID", details
    if left_count <= 0:
        return "VIRTUAL_LEFT_TRANSITION_INVALID", "VIRTUAL_LEFT_TRANSITION_INVALID", details
    if right_count <= 0:
        return "VIRTUAL_RIGHT_TRANSITION_INVALID", "VIRTUAL_RIGHT_TRANSITION_INVALID", details
    return "VIRTUAL_BOTH_TRANSITIONS_INVALID", "VIRTUAL_BOTH_TRANSITIONS_INVALID", details


def _suggest_next_phase_from_census(census: dict) -> dict:
    total = max(1, int(census.get("total_blocks", 0) or 0))
    hard_no_edge = int(census.get("template_graph_no_edge_count", 0) or 0) + int(census.get("rule_prefilter_all_fail_count", 0) or 0)
    has_endpoint = int(census.get("has_endpoint_edge_count", 0) or 0)
    band_narrow = int(census.get("band_too_narrow_count", 0) or 0)
    ton_success = int(census.get("ton_rescue_success_count", 0) or 0)
    applied = int(census.get("improvement_applied_count", 0) or 0)
    nonempty = int(census.get("candidate_pool_nonempty_count", 0) or 0)
    reasons: list[str] = []
    if hard_no_edge / total >= 0.50:
        reasons.append("real_bridge_no_edge_or_prefilter_fail_dominates")
    if ton_success <= 0:
        reasons.append("ton_rescue_success_zero")
    if applied <= 0:
        reasons.append("improvement_applied_zero")
    if has_endpoint + band_narrow > 0:
        reasons.append("controlled_virtual_pilot_has_candidate_surface")
    if hard_no_edge / total >= 0.50 and ton_success <= 0 and applied <= 0:
        suggestion = "PREPARE_CONTROLLED_VIRTUAL_BRIDGE"
    elif (has_endpoint + band_narrow + nonempty) > 0 and applied <= 0:
        suggestion = "REAL_BRIDGE_ONLY_NEAR_LIMIT"
    else:
        suggestion = "CONTINUE_REAL_BRIDGE_ONLY"
        if not reasons:
            reasons.append("real_bridge_only_still_has_observed_gain")
    return {"suggestion": suggestion, "reasons": reasons}


def _dominant_bridge_fail_reason(bridge_candidate_audit: dict, reason_buckets: dict | None = None) -> str:
    rule_counts = {
        "THICKNESS_RULE_FAIL": int((bridge_candidate_audit or {}).get("pair_invalid_thickness", 0) or 0),
        "WIDTH_RULE_FAIL": int((bridge_candidate_audit or {}).get("pair_invalid_width", 0) or 0),
        "TEMP_OVERLAP_FAIL": int((bridge_candidate_audit or {}).get("pair_invalid_temp", 0) or 0),
        "GROUP_SWITCH_FAIL": int((bridge_candidate_audit or {}).get("pair_invalid_group", 0) or 0),
    }
    active = {key: val for key, val in rule_counts.items() if val > 0}
    if len(active) > 1:
        return "MULTI_RULE_FAIL"
    if len(active) == 1:
        return next(iter(active.keys()))
    if reason_buckets:
        nonzero = {str(k): int(v) for k, v in reason_buckets.items() if int(v) > 0}
        if nonzero:
            return max(nonzero.items(), key=lambda kv: kv[1])[0]
    return str((bridge_candidate_audit or {}).get("dominant_fail_reason", "") or "UNKNOWN")


def _candidate_is_pilotable_rule_fail(candidate: dict, allowed_fail_reasons: set[str] | None = None) -> bool:
    fail_reasons = [str(v) for v in list((candidate or {}).get("fail_reasons", []) or [])]
    dominant = str((candidate or {}).get("dominant_fail_reason", "") or "")
    allowed = set(allowed_fail_reasons or {"THICKNESS_RULE_FAIL", "WIDTH_RULE_FAIL", "GROUP_SWITCH_FAIL", "TEMP_OVERLAP_FAIL", "MULTI_RULE_FAIL"})
    base_allowed = {"THICKNESS_RULE_FAIL", "WIDTH_RULE_FAIL", "GROUP_SWITCH_FAIL", "TEMP_OVERLAP_FAIL"}
    if dominant in allowed:
        if dominant == "MULTI_RULE_FAIL":
            return 0 < len(fail_reasons) <= 2
        return True
    if "MULTI_RULE_FAIL" in allowed and dominant == "MULTI_RULE_FAIL" and 0 < len(fail_reasons) <= 2:
        return True
    explicit = [reason for reason in fail_reasons if reason in (allowed & base_allowed)]
    return bool(explicit) and len(fail_reasons) <= 2


def _select_virtual_pilot_targets(
    candidates: list[dict],
    *,
    endpoint_class: str,
    max_targets: int,
    allowed_fail_reasons: set[str] | None = None,
    campaign_ton_min: float = 0.0,
) -> list[dict]:
    pilotable = [
        dict(candidate)
        for candidate in candidates
        if bool(candidate.get("frontier_consistent", False)) and _candidate_is_pilotable_rule_fail(candidate, allowed_fail_reasons)
    ]
    pilotable.sort(
        key=lambda item: (
            len(list(item.get("fail_reasons", []) or [])),
            0 if str(endpoint_class) == "HAS_ENDPOINT_EDGE" else 1,
            _small_block_pilot_score(
                int(item.get("block_orders", 0) or 0),
                float(item.get("block_tons", 0.0) or 0.0),
                float(campaign_ton_min),
            )[1],
            float(item.get("rank_score", 999999999.0) or 999999999.0),
            -float(item.get("block_tons", 0.0) or 0.0),
            -int(item.get("block_orders", 0) or 0),
        )
    )
    return pilotable[: max(0, int(max_targets))]


def _evaluate_virtual_pilot_eligibility(
    *,
    endpoint_class: str,
    candidate_count: int,
    frontier_candidate_count: int,
    pilot_candidates: list[dict],
    block_orders_count: int,
    block_tons: float,
    campaign_ton_min: float,
    attempts_so_far: int,
    max_attempts: int,
    allowed_endpoint_classes: set[str],
    already_tried: bool,
    applied_improvement_count: int,
    enabled: bool,
) -> VirtualPilotEligibilityResult:
    details = {
        "runtime_enabled": bool(enabled),
        "pilotable_candidate_count": int(len(pilot_candidates)),
        "block_orders": int(block_orders_count),
        "block_tons": float(block_tons),
        "attempts_so_far": int(attempts_so_far),
        "max_attempts": int(max_attempts),
        "applied_improvement_count": int(applied_improvement_count),
    }
    structural_eligible = True
    structural_reason = "ELIGIBLE"
    if endpoint_class == "RULE_PREFILTER_ALL_FAIL":
        structural_eligible = False
        structural_reason = "RULE_PREFILTER_ALL_FAIL"
    elif endpoint_class == "TEMPLATE_GRAPH_NO_EDGE":
        structural_eligible = False
        structural_reason = "TEMPLATE_GRAPH_NO_EDGE"
    elif endpoint_class not in set(allowed_endpoint_classes or {"HAS_ENDPOINT_EDGE", "BAND_TOO_NARROW"}):
        structural_eligible = False
        structural_reason = "NO_ENDPOINT_EDGE"
    elif candidate_count <= 0:
        structural_eligible = False
        structural_reason = "NO_ENDPOINT_EDGE"
    elif frontier_candidate_count <= 0:
        structural_eligible = False
        structural_reason = "NO_FRONTIER_CANDIDATE"
    hard_small_reject, small_block_penalty = _small_block_pilot_score(block_orders_count, block_tons, campaign_ton_min)
    details["small_block_penalty"] = float(small_block_penalty)
    details["small_block_hard_reject"] = bool(hard_small_reject)
    if hard_small_reject:
        structural_eligible = False
        structural_reason = "BLOCK_TOO_SMALL"
    elif not pilot_candidates:
        structural_eligible = False
        structural_reason = "NO_PILOTABLE_CANDIDATE"
    runtime_enabled = bool(enabled)
    if not structural_eligible:
        return VirtualPilotEligibilityResult(False, runtime_enabled, False, endpoint_class, candidate_count, frontier_candidate_count, len(pilot_candidates), structural_reason, details)
    if not runtime_enabled:
        return VirtualPilotEligibilityResult(True, False, False, endpoint_class, candidate_count, frontier_candidate_count, len(pilot_candidates), "PILOT_DISABLED", details)
    if already_tried:
        return VirtualPilotEligibilityResult(True, True, False, endpoint_class, candidate_count, frontier_candidate_count, len(pilot_candidates), "BLOCK_ALREADY_TRIED", details)
    if attempts_so_far >= max_attempts:
        return VirtualPilotEligibilityResult(True, True, False, endpoint_class, candidate_count, frontier_candidate_count, len(pilot_candidates), "RUN_LIMIT_REACHED", details)
    if applied_improvement_count > 0:
        return VirtualPilotEligibilityResult(True, True, False, endpoint_class, candidate_count, frontier_candidate_count, len(pilot_candidates), "RUN_ALREADY_HAS_APPLIED_IMPROVEMENT", details)
    return VirtualPilotEligibilityResult(True, True, True, endpoint_class, candidate_count, frontier_candidate_count, len(pilot_candidates), "ELIGIBLE", details)


def _record_bridgeability_census_item(census: BridgeabilityCensus, item: dict) -> None:
    census.add_item(item)
    print(
        f"[APS][BRIDGEABILITY_CENSUS_ITEM] line={item.get('line')}, "
        f"block_id={item.get('block_id')}, block_size={item.get('block_size')}, "
        f"orders={item.get('orders')}, tons={float(item.get('tons', 0.0) or 0.0):.1f}, "
        f"endpoint_class={item.get('endpoint_class')}, "
        f"dominant_fail_reason={item.get('dominant_fail_reason')}, "
        f"candidate_count={item.get('candidate_count')}, "
        f"frontier_candidate_count={item.get('frontier_candidate_count')}, "
        f"final_decision={item.get('final_decision')}"
    )


def _update_bridgeability_census_final_decision(
    census: BridgeabilityCensus,
    *,
    line: str,
    block_id: int,
    final_decision: str,
) -> None:
    for item in reversed(census.items):
        if str(item.get("line", "")) == str(line) and int(item.get("block_id", -1) or -1) == int(block_id):
            previous = str(item.get("final_decision", "") or "")
            item["final_decision"] = str(final_decision)
            if previous != "IMPROVEMENT_RECORDED_BUT_NOT_APPLIED" and final_decision == "IMPROVEMENT_RECORDED_BUT_NOT_APPLIED":
                census.improvement_recorded_count += 1
            if previous != "IMPROVEMENT_APPLIED" and final_decision == "IMPROVEMENT_APPLIED":
                census.improvement_applied_count += 1
            return


def _repair_bridge_candidate_audit(
    *,
    line: str,
    block_id: int,
    block: list[CampaignSegment],
    real_bridge_lookup: dict[tuple[str, str, str], list[dict]],
    matched_keys: list[tuple[str, str, str]],
    frontier_contexts: dict[tuple[str, str, str], list[dict]],
    direct_edges: set[tuple[str, str]],
    real_edges: set[tuple[str, str]],
    pair_rows_by_key: dict[tuple[str, str], list[dict]],
    order_specs_by_id: dict[str, dict],
    rule,
    order_tons: dict[str, float],
    campaign_ton_min: float,
    campaign_ton_max: float,
    real_edge_adjustment_cost: dict[tuple[str, str], int] | None = None,
    endpoint_class: str = "",
) -> dict:
    combined: list[str] = []
    for seg in block:
        combined.extend([str(v) for v in seg.order_ids])
    pos_by_oid = {str(oid): idx for idx, oid in enumerate(combined)}
    prefix = [0.0] * (len(combined) + 1)
    for idx, oid in enumerate(combined):
        prefix[idx + 1] = prefix[idx] + float(order_tons.get(str(oid), 0.0) or 0.0)

    out = {
        "matched": 0,
        "rejected_pair_invalid": 0,
        "rejected_ton_invalid": 0,
        "rejected_score_worse": 0,
        "accepted": 0,
        "exact_invalid_pair_count": 0,
        "frontier_mismatch": 0,
        "pair_invalid_width": 0,
        "pair_invalid_thickness": 0,
        "pair_invalid_temp": 0,
        "pair_invalid_group": 0,
        "pair_invalid_context": 0,
        "pair_invalid_unknown": 0,
        "prefilter_fail_thickness": 0,
        "rank_pass_thickness": 0,
        "rank_fail_thickness": 0,
        "prefilter_reject_count": 0,
        "pilot_candidates": [],
        "ton_rescue_candidates": [],
        "filtered_ton_below_min_current_block": 0,
        "filtered_ton_below_min_even_after_neighbor_expansion": 0,
        "filtered_ton_above_max_after_expansion": 0,
        "filtered_ton_split_not_found": 0,
        "filtered_ton_rescue_no_gain": 0,
        "reject_buckets": Counter(),
    }
    seen: set[tuple[str, str, str]] = set()
    candidate_no = 0
    for key in matched_keys:
        if key in seen:
            continue
        seen.add(key)
        rows = real_bridge_lookup.get(key, [])
        if not rows:
            continue
        _, src, dst = key
        for row in rows:
            candidate_no += 1
            candidate_id = f"BRIDGE_CAND_{line}_{block_id}_{candidate_no}"
            bridge_count = int(row.get("bridge_count", row.get("virtual_count", 1)) or 1)
            rank_metrics = _quick_pair_rank_metrics(order_specs_by_id, str(src), str(dst), rule=rule)
            prefilter_pass = bool(rank_metrics.get("pass", False))
            thickness_prefilter_pass = bool(rank_metrics.get("thickness_prefilter_pass", False))
            thickness_distance = float(rank_metrics.get("thickness_distance_mm", 999999.0) or 999999.0)
            adjustment_cost = int((real_edge_adjustment_cost or {}).get((str(src), str(dst)), 0) or 0)
            candidate_rank_score = (
                int(round(float(rank_metrics.get("hard_prefilter_fail_penalty", 0.0))))
                + int(round(float(rank_metrics.get("thickness_penalty", 0.0))))
                + int(round(float(rank_metrics.get("width_penalty", 0.0))))
                + int(round(float(rank_metrics.get("temp_penalty", 0.0))))
                + int(round(float(rank_metrics.get("group_penalty", 0.0))))
                + int(adjustment_cost)
            )
            print(
                f"[APS][REPAIR_BRIDGE_MULTI_PREFILTER] candidate_id={candidate_id}, "
                f"pair=({src},{dst}), pass={bool(prefilter_pass)}, "
                f"thickness_pass={bool(rank_metrics.get('thickness_pass', False))}, "
                f"width_pass={bool(rank_metrics.get('width_pass', False))}, "
                f"temp_pass={bool(rank_metrics.get('temp_pass', False))}, "
                f"group_pass={bool(rank_metrics.get('group_pass', False))}, "
                f"dominant_fail_reason={rank_metrics.get('dominant_fail_reason', '')}, "
                f"fail_reasons={list(rank_metrics.get('fail_reasons', []) or [])}"
            )
            if thickness_prefilter_pass:
                out["rank_pass_thickness"] += 1
            else:
                out["rank_fail_thickness"] += 1
                out["prefilter_fail_thickness"] += 1
                print(
                    f"[APS][REPAIR_BRIDGE_THICKNESS_PREFILTER_FAIL] candidate_id={candidate_id}, "
                    f"pair=({src},{dst}), "
                    f"left_thickness={rank_metrics.get('left_thickness')}, "
                    f"right_thickness={rank_metrics.get('right_thickness')}, "
                    f"reason=THICKNESS_RULE_FAIL, "
                    f"delta_ratio={rank_metrics.get('thickness_delta_ratio')}, "
                    f"delta_mm={rank_metrics.get('thickness_distance_mm')}"
                )
            print(
                f"[APS][REPAIR_BRIDGE_CANDIDATE_RANK] candidate_id={candidate_id}, "
                f"pair=({src},{dst}), prefilter_pass={bool(prefilter_pass)}, "
                f"fail_reasons={list(rank_metrics.get('fail_reasons', []) or [])}, "
                f"hard_prefilter_fail_penalty={float(rank_metrics.get('hard_prefilter_fail_penalty', 0.0)):.3f}, "
                f"thickness_penalty={float(rank_metrics.get('thickness_penalty', 0.0)):.3f}, "
                f"width_penalty={float(rank_metrics.get('width_penalty', 0.0)):.3f}, "
                f"temp_penalty={float(rank_metrics.get('temp_penalty', 0.0)):.3f}, "
                f"group_penalty={float(rank_metrics.get('group_penalty', 0.0)):.3f}, "
                f"adjustment_cost={adjustment_cost}, "
                f"rank_score={candidate_rank_score}"
            )
            out["matched"] += 1
            print(
                f"[APS][REPAIR_BRIDGE_CANDIDATE] candidate_id={candidate_id}, line={line}, "
                f"block_id={block_id}, from_order_id={src}, to_order_id={dst}, "
                f"edge_type=REAL_BRIDGE_EDGE, bridge_count={bridge_count}"
            )
            print(f"[APS][REPAIR_BRIDGE_STAGE] candidate_id={candidate_id}, stage=MATCHED")
            pilot_candidate_base = {
                "candidate_id": candidate_id,
                "key": key,
                "pair": (str(src), str(dst)),
                "from_order_id": str(src),
                "to_order_id": str(dst),
                "frontier_consistent": False,
                "prefilter_pass": bool(prefilter_pass),
                "dominant_fail_reason": str(rank_metrics.get("dominant_fail_reason", "") or ""),
                "fail_reasons": list(rank_metrics.get("fail_reasons", []) or []),
                "fail_reason_count": len(list(rank_metrics.get("fail_reasons", []) or [])),
                "rank_score": float(candidate_rank_score),
                "adjustment_cost": int(adjustment_cost),
                "thickness_gap": rank_metrics.get("thickness_gap"),
                "width_delta": rank_metrics.get("width_delta"),
                "temp_overlap": rank_metrics.get("temp_overlap"),
                "group_pass": bool(rank_metrics.get("group_pass", False)),
                "block_orders": int(len(combined)),
                "block_tons": float(prefix[-1] if prefix else 0.0),
            }

            contexts = frontier_contexts.get(key, [])
            if not contexts:
                # A band hit is not necessarily a materializable repair edge. Only the
                # adjusted frontier pair is allowed to enter the landing candidate pool.
                actual_left_tail = ""
                actual_right_head = ""
                for ctx in frontier_contexts.values():
                    if ctx:
                        actual_left_tail = str(ctx[0].get("actual_left_tail", ""))
                        actual_right_head = str(ctx[0].get("actual_right_head", ""))
                        break
                out["frontier_mismatch"] += 1
                out["rejected_pair_invalid"] += 1
                out["exact_invalid_pair_count"] += 1
                out["reject_buckets"]["TEMPLATE_PAIR_INVALID"] += 1
                print(
                    f"[APS][REPAIR_BRIDGE_FRONTIER_MISMATCH] candidate_id={candidate_id}, "
                    f"bridge_from={src}, bridge_to={dst}, actual_left_tail={actual_left_tail}, "
                    f"actual_right_head={actual_right_head}"
                )
                print(f"[APS][REPAIR_BRIDGE_STAGE] candidate_id={candidate_id}, stage=FRONTIER_CHECKED")
                print(f"[APS][REPAIR_BRIDGE_STAGE] candidate_id={candidate_id}, stage=REJECTED, reason=TEMPLATE_PAIR_INVALID")
                continue

            ctx = contexts[0]
            adjusted_left_orders = [str(v) for v in ctx.get("adjusted_left_orders", [])]
            adjusted_right_orders = [str(v) for v in ctx.get("adjusted_right_orders", [])]
            if str(adjusted_left_orders[-1] if adjusted_left_orders else "") != str(src) or str(adjusted_right_orders[0] if adjusted_right_orders else "") != str(dst):
                out["frontier_mismatch"] += 1
                out["rejected_pair_invalid"] += 1
                out["exact_invalid_pair_count"] += 1
                out["reject_buckets"]["TEMPLATE_PAIR_INVALID"] += 1
                print(
                    f"[APS][REPAIR_BRIDGE_FRONTIER_MISMATCH] candidate_id={candidate_id}, "
                    f"bridge_from={src}, bridge_to={dst}, "
                    f"actual_left_tail={adjusted_left_orders[-1] if adjusted_left_orders else ''}, "
                    f"actual_right_head={adjusted_right_orders[0] if adjusted_right_orders else ''}"
                )
                print(f"[APS][REPAIR_BRIDGE_STAGE] candidate_id={candidate_id}, stage=FRONTIER_CHECKED")
                print(f"[APS][REPAIR_BRIDGE_STAGE] candidate_id={candidate_id}, stage=REJECTED, reason=TEMPLATE_PAIR_INVALID")
                continue
            print(f"[APS][REPAIR_BRIDGE_STAGE] candidate_id={candidate_id}, stage=FRONTIER_CHECKED")
            pilot_candidate = dict(pilot_candidate_base)
            pilot_candidate["frontier_consistent"] = True
            pilot_candidate["adjustment_id"] = int(ctx.get("adjustment_id", 0) or 0)
            out["pilot_candidates"].append(pilot_candidate)

            adjustment_id = int(ctx.get("adjustment_id", 0) or 0)
            allow_prefilter_bypass = (
                str(endpoint_class) == "BAND_TOO_NARROW"
                and 0 < int(adjustment_id) <= 2
            )
            if not prefilter_pass and not allow_prefilter_bypass:
                out["prefilter_reject_count"] += 1
                out["rejected_pair_invalid"] += 1
                out["exact_invalid_pair_count"] += 1
                dominant = str(rank_metrics.get("dominant_fail_reason", "") or "PREFILTER_REJECTED")
                out["reject_buckets"]["PREFILTER_REJECTED"] += 1
                out["reject_buckets"][dominant] += 1
                print(
                    f"[APS][REPAIR_BRIDGE_PREFILTER_REJECT] candidate_id={candidate_id}, "
                    f"pair=({src},{dst}), fail_reasons={list(rank_metrics.get('fail_reasons', []) or [])}, "
                    f"dominant_fail_reason={dominant}"
                )
                print(f"[APS][REPAIR_BRIDGE_STAGE] candidate_id={candidate_id}, stage=REJECTED, reason=PREFILTER_REJECTED")
                continue
            if not prefilter_pass and allow_prefilter_bypass:
                print(
                    f"[APS][REPAIR_BRIDGE_PREFILTER_BYPASS] candidate_id={candidate_id}, "
                    f"reason=BAND_TOO_NARROW_TOPK_EXCEPTION"
                )

            left_materialized = adjusted_left_orders[: adjusted_left_orders.index(str(src)) + 1]
            right_materialized = adjusted_right_orders[adjusted_right_orders.index(str(dst)) :]
            tentative_sequence = left_materialized + right_materialized
            frontier_idx = len(left_materialized) - 1
            if frontier_idx >= 0 and frontier_idx + 1 < len(tentative_sequence):
                tentative_frontier_pair = (str(tentative_sequence[frontier_idx]), str(tentative_sequence[frontier_idx + 1]))
            else:
                tentative_frontier_pair = ("", "")
            left_tail_window = left_materialized[-5:]
            right_head_window = right_materialized[:5]
            tentative_window = tentative_sequence[max(0, frontier_idx - 2) : min(len(tentative_sequence), frontier_idx + 4)]
            print(
                f"[APS][REPAIR_BRIDGE_MATERIALIZE] candidate_id={candidate_id}, bridge_from={src}, "
                f"bridge_to={dst}, left_materialized_tail={left_tail_window}, "
                f"right_materialized_head={right_head_window}, tentative_frontier_pair={tentative_frontier_pair}"
            )
            print(
                f"[APS][REPAIR_BRIDGE_TENTATIVE] candidate_id={candidate_id}, "
                f"left_tail_window={left_tail_window}, right_head_window={right_head_window}, "
                f"bridge_from={src}, bridge_to={dst}, tentative_window={tentative_window}"
            )
            print(f"[APS][REPAIR_BRIDGE_STAGE] candidate_id={candidate_id}, stage=MATERIALIZED")

            pair = (str(src), str(dst))
            original_pairs = set(zip(adjusted_left_orders, adjusted_left_orders[1:])) | set(zip(adjusted_right_orders, adjusted_right_orders[1:]))
            tentative_pairs = list(zip(tentative_sequence, tentative_sequence[1:]))
            new_pairs = [p for p in tentative_pairs if p not in original_pairs]
            print(f"[APS][REPAIR_BRIDGE_NEW_PAIRS] candidate_id={candidate_id}, new_pairs={new_pairs}")
            print(f"[APS][REPAIR_BRIDGE_STAGE] candidate_id={candidate_id}, stage=NEW_PAIRS_COMPUTED")
            if pair not in new_pairs or tentative_frontier_pair != pair:
                out["rejected_pair_invalid"] += 1
                out["exact_invalid_pair_count"] += 1
                out["pair_invalid_unknown"] += 1
                out["reject_buckets"]["TEMPLATE_PAIR_INVALID"] += 1
                print(
                    f"[APS][REPAIR_BRIDGE_MATERIALIZE_ERROR] candidate_id={candidate_id}, "
                    f"reason=BRIDGE_PAIR_NOT_FRONTIER_IN_TENTATIVE"
                )
                print(f"[APS][REPAIR_BRIDGE_STAGE] candidate_id={candidate_id}, stage=REJECTED, reason=TEMPLATE_PAIR_INVALID")
                continue
            adjacency_ok = tentative_frontier_pair == pair
            pair_eval = _evaluate_pair_violation(
                left_order=str(src),
                right_order=str(dst),
                line=str(line),
                template_lookup=pair_rows_by_key,
                direct_edges=direct_edges,
                real_edges=real_edges,
                adjacency_ok=bool(adjacency_ok),
                order_context_by_id=order_specs_by_id,
                rule=rule,
            )
            print(
                f"[APS][REPAIR_BRIDGE_PAIR_CHECK] candidate_id={candidate_id}, pair=({src},{dst}), "
                f"template_ok={bool(pair_eval.template_ok)}, adjacency_ok={bool(pair_eval.adjacency_ok)}, "
                f"context_ok={bool(pair_eval.context_ok)}, reason={pair_eval.reason}"
            )
            _print_pair_audit(
                "REPAIR_BRIDGE_PAIR_AUDIT_PRE",
                candidate_id=candidate_id,
                pair=pair,
                line=str(line),
                eval_result=pair_eval,
            )
            if not pair_eval.context_ok:
                _print_order_context_miss(candidate_id=candidate_id, pair=pair, eval_result=pair_eval)
            reason = "" if pair_eval.reason == "OK" else str(pair_eval.reason)
            if reason:
                out["rejected_pair_invalid"] += 1
                out["exact_invalid_pair_count"] += 1
                if reason == "WIDTH_RULE_FAIL":
                    out["pair_invalid_width"] += 1
                elif reason == "THICKNESS_RULE_FAIL":
                    out["pair_invalid_thickness"] += 1
                elif reason == "TEMP_OVERLAP_FAIL":
                    out["pair_invalid_temp"] += 1
                elif reason == "GROUP_SWITCH_FAIL":
                    out["pair_invalid_group"] += 1
                elif reason == "ORDER_CONTEXT_MISSING":
                    out["pair_invalid_context"] += 1
                else:
                    out["pair_invalid_unknown"] += 1
                out["reject_buckets"][reason] += 1
                out["reject_buckets"]["TEMPLATE_PAIR_INVALID"] += 1
                print(
                    f"[APS][REPAIR_BRIDGE_PAIR_INVALID] candidate_id={candidate_id}, "
                    f"pair=({src},{dst}), reason={reason}"
                )
                print(f"[APS][REPAIR_BRIDGE_STAGE] candidate_id={candidate_id}, stage=REJECTED, reason=TEMPLATE_PAIR_INVALID")
                continue
            print(f"[APS][REPAIR_BRIDGE_STAGE] candidate_id={candidate_id}, stage=PAIR_VALIDATED")

            ton_valid = False
            materialized_tons = sum(float(order_tons.get(str(oid), 0.0) or 0.0) for oid in tentative_sequence)
            if float(campaign_ton_min) - 1e-6 <= materialized_tons <= float(campaign_ton_max) + 1e-6:
                ton_valid = True
            if not ton_valid:
                if materialized_tons < float(campaign_ton_min) - 1e-6:
                    out["filtered_ton_below_min_current_block"] += 1
                    out["ton_rescue_candidates"].append(
                        {
                            "candidate_id": candidate_id,
                            "key": key,
                            "bridge_from": str(src),
                            "bridge_to": str(dst),
                            "context": dict(ctx),
                            "pre_pair_audit": pair_eval.to_dict(),
                            "materialized_tons": float(materialized_tons),
                            "tentative_sequence": list(tentative_sequence),
                        }
                    )
                    print(f"[APS][REPAIR_BRIDGE_STAGE] candidate_id={candidate_id}, stage=TON_RESCUE_ATTEMPT")
                else:
                    out["rejected_ton_invalid"] += 1
                    out["reject_buckets"]["TON_WINDOW_INVALID"] += 1
                    print(f"[APS][REPAIR_BRIDGE_STAGE] candidate_id={candidate_id}, stage=REJECTED, reason=TON_WINDOW_INVALID")
                continue
            print(f"[APS][REPAIR_BRIDGE_STAGE] candidate_id={candidate_id}, stage=TON_WINDOW_VALIDATED")
            print(f"[APS][REPAIR_BRIDGE_STAGE] candidate_id={candidate_id}, stage=SCORE_COMPARED")
            # This audit is observational; final ACCEPTED is decided by the existing reconstruction solver.
    return out


def _build_reconstruction_edge_maps(transition_pack) -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    tpl_df = _transition_templates_df(transition_pack)
    if tpl_df.empty:
        return set(), set()
    direct: set[tuple[str, str]] = set()
    real: set[tuple[str, str]] = set()
    for row in tpl_df.to_dict("records"):
        edge_type = str(row.get("edge_type", "DIRECT_EDGE") or "DIRECT_EDGE")
        src, src_field = _extract_bridge_endpoint(row, "from")
        dst, dst_field = _extract_bridge_endpoint(row, "to")
        if not src_field or not dst_field:
            continue
        key = (src, dst)
        if edge_type == "DIRECT_EDGE":
            direct.add(key)
        elif edge_type == "REAL_BRIDGE_EDGE":
            real.add(key)
    return direct, real


def _build_virtual_reconstruction_edges(transition_pack) -> tuple[set[tuple[str, str]], dict[tuple[str, str], float]]:
    tpl_df = _transition_templates_df(transition_pack)
    if tpl_df.empty:
        return set(), {}
    virtual: set[tuple[str, str]] = set()
    virtual_tons_by_edge: dict[tuple[str, str], float] = {}
    for row in tpl_df.to_dict("records"):
        if not _is_virtual_bridge_row(row):
            continue
        src, src_field = _extract_bridge_endpoint(row, "from")
        dst, dst_field = _extract_bridge_endpoint(row, "to")
        if not src_field or not dst_field:
            continue
        key = (str(src), str(dst))
        virtual.add(key)
        virtual_tons_by_edge[key] = max(float(virtual_tons_by_edge.get(key, 0.0) or 0.0), float(_virtual_bridge_tons_from_row(row)))
    return virtual, virtual_tons_by_edge


def _bridge_filter_diag_template() -> dict:
    return {
        "repair_only_real_bridge_candidates_total": 0,
        "repair_only_real_bridge_candidates_kept": 0,
        "repair_only_real_bridge_filtered_direct_feasible": 0,
        "repair_only_real_bridge_filtered_pair_invalid": 0,
        "repair_only_real_bridge_filtered_ton_invalid": 0,
        "repair_only_real_bridge_filtered_score_worse": 0,
        "repair_only_real_bridge_filtered_bridge_limit_exceeded": 0,
        "repair_only_real_bridge_filtered_multiplicity_invalid": 0,
        "repair_only_real_bridge_filtered_bridge_path_not_real": 0,
        "repair_only_real_bridge_filtered_bridge_path_missing": 0,
        "repair_only_real_bridge_filtered_block_order_mismatch": 0,
    }


def _segment_edge_valid_for_reconstruction(
    oids: list[str],
    direct_edges: set[tuple[str, str]],
    real_edges: set[tuple[str, str]],
    *,
    allow_real_bridge: bool,
    allow_virtual_bridge: bool,
    max_real_bridge_per_segment: int,
    virtual_edges: set[tuple[str, str]] | None = None,
    max_virtual_bridge_per_segment: int = 0,
) -> tuple[bool, int, int]:
    real_used = 0
    virtual_used = 0
    for i in range(len(oids) - 1):
        key = (str(oids[i]), str(oids[i + 1]))
        if key in direct_edges:
            continue
        if allow_real_bridge and key in real_edges and real_used < int(max_real_bridge_per_segment):
            real_used += 1
            continue
        if allow_virtual_bridge and key in (virtual_edges or set()) and virtual_used < int(max_virtual_bridge_per_segment):
            virtual_used += 1
            continue
        return False, real_used, virtual_used
    return True, real_used, virtual_used


def _segment_edge_stats_for_reconstruction(
    oids: list[str],
    direct_edges: set[tuple[str, str]],
    real_edges: set[tuple[str, str]],
    *,
    allow_real_bridge: bool,
    max_real_bridge_per_segment: int,
    real_edge_distance: dict[tuple[str, str], int] | None = None,
    real_edge_adjustment_cost: dict[tuple[str, str], int] | None = None,
    allow_virtual_bridge: bool = False,
    virtual_edges: set[tuple[str, str]] | None = None,
    max_virtual_bridge_per_segment: int = 0,
    virtual_tons_by_edge: dict[tuple[str, str], float] | None = None,
) -> dict:
    total_missing = 0
    real_candidate_edges = 0
    virtual_candidate_edges = 0
    bridge_limit_exceeded = 0
    virtual_limit_exceeded = 0
    real_used = 0
    virtual_used = 0
    virtual_tons = 0.0
    boundary_band_distance = 0
    endpoint_adjustment_cost = 0
    for i in range(len(oids) - 1):
        key = (str(oids[i]), str(oids[i + 1]))
        if key in direct_edges:
            continue
        total_missing += 1
        if key in real_edges:
            real_candidate_edges += 1
            if allow_real_bridge and real_used < int(max_real_bridge_per_segment):
                real_used += 1
                boundary_band_distance += int((real_edge_distance or {}).get(key, 0))
                endpoint_adjustment_cost += int((real_edge_adjustment_cost or {}).get(key, 0))
            elif allow_real_bridge:
                bridge_limit_exceeded += 1
        elif key in (virtual_edges or set()):
            virtual_candidate_edges += 1
            if allow_virtual_bridge and virtual_used < int(max_virtual_bridge_per_segment):
                virtual_used += 1
                virtual_tons += float((virtual_tons_by_edge or {}).get(key, 0.0) or 0.0)
            elif allow_virtual_bridge:
                virtual_limit_exceeded += 1
    return {
        "missing_edges": int(total_missing),
        "real_candidate_edges": int(real_candidate_edges),
        "virtual_candidate_edges": int(virtual_candidate_edges),
        "real_used": int(real_used),
        "virtual_used": int(virtual_used),
        "virtual_tons": float(virtual_tons),
        "bridge_limit_exceeded": int(bridge_limit_exceeded),
        "virtual_limit_exceeded": int(virtual_limit_exceeded),
        "boundary_band_distance": int(boundary_band_distance),
        "endpoint_adjustment_cost": int(endpoint_adjustment_cost),
    }


def _infer_order_tons_from_underfilled(
    underfilled_segments: list[CampaignSegment],
    order_tons: dict[str, float] | None,
) -> dict[str, float]:
    out = {str(k): float(v) for k, v in (order_tons or {}).items()}
    for seg in underfilled_segments:
        missing = [oid for oid in seg.order_ids if str(oid) not in out]
        if missing:
            each = float(seg.total_tons) / max(1, len(seg.order_ids))
            for oid in missing:
                out[str(oid)] = each
    return out


def _solve_underfilled_reconstruction_block(
    block: list[CampaignSegment],
    *,
    direct_edges: set[tuple[str, str]],
    real_edges: set[tuple[str, str]],
    order_tons: dict[str, float],
    campaign_ton_min: float,
    campaign_ton_max: float,
    campaign_ton_target: float,
    allow_real_bridge: bool,
    allow_virtual_bridge: bool,
    max_real_bridge_per_segment: int,
    bridge_cost_penalty: float,
    next_local_id: int,
    real_edge_distance: dict[tuple[str, str], int] | None = None,
    real_edge_adjustment_cost: dict[tuple[str, str], int] | None = None,
    virtual_edges: set[tuple[str, str]] | None = None,
    max_virtual_bridge_per_segment: int = 0,
    virtual_tons_by_edge: dict[tuple[str, str], float] | None = None,
    virtual_bridge_penalty: float = 0.0,
) -> tuple[list[CampaignSegment], list[CampaignSegment], dict]:
    combined: list[str] = []
    for seg in block:
        combined.extend([str(v) for v in seg.order_ids])
    n = len(combined)
    prefix = [0.0] * (n + 1)
    for i, oid in enumerate(combined):
        prefix[i + 1] = prefix[i] + float(order_tons.get(str(oid), 0.0) or 0.0)

    candidates: dict[int, list[dict]] = {i: [] for i in range(n)}
    obs = _bridge_filter_diag_template()
    obs["candidate_cutpoints_total"] = 0
    obs["candidate_cutpoints_ton_window_valid"] = 0
    obs["candidate_cutpoints_pair_valid"] = 0
    obs["direct_only_feasible"] = False
    for i in range(n):
        for j in range(i + 1, n + 1):
            obs["candidate_cutpoints_total"] += 1
            tons = prefix[j] - prefix[i]
            if tons > float(campaign_ton_max) + 1e-6:
                obs["repair_only_real_bridge_filtered_ton_invalid"] += 1
                break
            if tons < float(campaign_ton_min) - 1e-6:
                obs["repair_only_real_bridge_filtered_ton_invalid"] += 1
                continue
            obs["candidate_cutpoints_ton_window_valid"] += 1
            oids = combined[i:j]
            edge_stats = _segment_edge_stats_for_reconstruction(
                oids,
                direct_edges,
                real_edges,
                allow_real_bridge=allow_real_bridge,
                max_real_bridge_per_segment=max_real_bridge_per_segment,
                real_edge_distance=real_edge_distance,
                real_edge_adjustment_cost=real_edge_adjustment_cost,
                allow_virtual_bridge=allow_virtual_bridge,
                virtual_edges=virtual_edges,
                max_virtual_bridge_per_segment=max_virtual_bridge_per_segment,
                virtual_tons_by_edge=virtual_tons_by_edge,
            )
            obs["repair_only_real_bridge_candidates_total"] += int(edge_stats["real_candidate_edges"])
            obs["repair_only_real_bridge_filtered_bridge_limit_exceeded"] += int(edge_stats["bridge_limit_exceeded"])
            ok, real_used, virtual_used = _segment_edge_valid_for_reconstruction(
                oids,
                direct_edges,
                real_edges,
                allow_real_bridge=allow_real_bridge,
                allow_virtual_bridge=allow_virtual_bridge,
                max_real_bridge_per_segment=max_real_bridge_per_segment,
                virtual_edges=virtual_edges,
                max_virtual_bridge_per_segment=max_virtual_bridge_per_segment,
            )
            if not ok:
                obs["repair_only_real_bridge_filtered_pair_invalid"] += 1
                continue
            obs["candidate_cutpoints_pair_valid"] += 1
            if int(real_used) == 0 and int(virtual_used) == 0:
                obs["direct_only_feasible"] = True
            else:
                obs["repair_only_real_bridge_candidates_kept"] += int(real_used)
            candidates[i].append(
                {
                    "end": j,
                    "oids": oids,
                    "tons": tons,
                    "real_used": int(real_used),
                    "virtual_used": int(virtual_used),
                    "virtual_tons": float(edge_stats.get("virtual_tons", 0.0) or 0.0),
                    "boundary_band_distance": int(edge_stats.get("boundary_band_distance", 0)),
                    "endpoint_adjustment_cost": int(edge_stats.get("endpoint_adjustment_cost", 0)),
                    "target_gap": abs(tons - float(campaign_ton_target)),
                }
            )

    best_score = None
    best_path: list[dict] = []
    best_residual: list[str] = combined
    best_residual_tons = prefix[n]

    def dfs(pos: int, path: list[dict]) -> None:
        nonlocal best_score, best_path, best_residual, best_residual_tons
        residual = combined[pos:]
        residual_tons = prefix[n] - prefix[pos]
        valid_count = len(path)
        salvaged_orders = sum(len(item["oids"]) for item in path)
        real_used_total = sum(int(item.get("real_used", 0)) for item in path)
        virtual_used_total = sum(int(item.get("virtual_used", 0)) for item in path)
        boundary_band_distance = sum(int(item.get("boundary_band_distance", 0)) for item in path)
        endpoint_adjustment_cost = sum(int(item.get("endpoint_adjustment_cost", 0)) for item in path)
        target_gap = sum(float(item.get("target_gap", 0.0)) for item in path)
        residual_count = 1 if residual else 0
        score = (
            -valid_count,
            -salvaged_orders,
            residual_count,
            real_used_total * float(bridge_cost_penalty),
            virtual_used_total * float(virtual_bridge_penalty),
            boundary_band_distance,
            endpoint_adjustment_cost,
            target_gap,
        )
        if best_score is None or score < best_score:
            best_score = score
            best_path = [dict(item) for item in path]
            best_residual = list(residual)
            best_residual_tons = float(residual_tons)
        for cand in candidates.get(pos, []):
            dfs(int(cand["end"]), path + [cand])

    dfs(0, [])
    if not best_path:
        obs.update({"success": False, "valid_count": 0, "real_bridge_used": 0})
        return [], block, obs

    line = str(block[0].line)
    valid: list[CampaignSegment] = []
    local_id = int(next_local_id)
    for item in best_path:
        local_id += 1
        oids = [str(v) for v in item["oids"]]
        valid.append(
            CampaignSegment(
                line=line,
                campaign_local_id=local_id,
                order_ids=oids,
                total_tons=float(item["tons"]),
                cut_reason=CutReason.TARGET_REACHED,
                start_order_id=oids[0],
                end_order_id=oids[-1],
                edge_count=max(0, len(oids) - 1),
                is_valid=True,
            )
        )
    residual_under: list[CampaignSegment] = []
    if best_residual:
        local_id += 1
        residual_under.append(
            CampaignSegment(
                line=line,
                campaign_local_id=local_id,
                order_ids=list(best_residual),
                total_tons=float(best_residual_tons),
                cut_reason=CutReason.TAIL_UNDERFILLED,
                start_order_id=str(best_residual[0]),
                end_order_id=str(best_residual[-1]),
                edge_count=max(0, len(best_residual) - 1),
                is_valid=False,
            )
        )
    real_bridge_used = sum(int(item.get("real_used", 0)) for item in best_path)
    virtual_bridge_used = sum(int(item.get("virtual_used", 0)) for item in best_path)
    virtual_bridge_tons = sum(float(item.get("virtual_tons", 0.0) or 0.0) for item in best_path)
    obs.update({
        "success": True,
        "valid_count": int(len(valid)),
        "salvaged_orders": int(sum(len(s.order_ids) for s in valid)),
        "residual_underfilled": int(len(residual_under)),
        "real_bridge_used": int(real_bridge_used),
        "virtual_bridge_used": int(virtual_bridge_used),
        "virtual_bridge_tons": float(virtual_bridge_tons),
    })
    return valid, residual_under, obs


def _campaign_segment_from_orders(
    *,
    line: str,
    campaign_local_id: int,
    order_ids: list[str],
    order_tons: dict[str, float],
    is_valid: bool,
) -> CampaignSegment | None:
    oids = [str(v) for v in order_ids if str(v)]
    if not oids:
        return None
    total_tons = sum(float(order_tons.get(str(oid), 0.0) or 0.0) for oid in oids)
    return CampaignSegment(
        line=str(line),
        campaign_local_id=int(campaign_local_id),
        order_ids=oids,
        total_tons=float(total_tons),
        cut_reason=CutReason.TARGET_REACHED if is_valid else CutReason.TAIL_UNDERFILLED,
        start_order_id=oids[0],
        end_order_id=oids[-1],
        edge_count=max(0, len(oids) - 1),
        is_valid=bool(is_valid),
    )


def _generate_ton_rescue_windows(
    *,
    line_under: list[CampaignSegment],
    current_block_idx: int,
    current_block_size: int,
    order_tons: dict[str, float],
    campaign_ton_min: float,
    max_neighbor_blocks: int,
    max_orders_per_window: int,
    enable_backward: bool,
    enable_forward: bool,
    enable_bidirectional: bool,
) -> list[dict]:
    n = len(line_under)
    start0 = int(current_block_idx)
    end0 = min(n, int(current_block_idx) + int(current_block_size))
    if start0 < 0 or start0 >= n or end0 <= start0:
        return []

    windows: list[dict] = []
    seen: set[tuple[int, int]] = set()

    def add_window(direction: str, back: int, fwd: int) -> None:
        start = max(0, start0 - int(back))
        end = min(n, end0 + int(fwd))
        key = (start, end)
        if key in seen:
            return
        seen.add(key)
        block = line_under[start:end]
        orders = [str(oid) for seg in block for oid in seg.order_ids]
        if len(orders) > int(max_orders_per_window):
            return
        tons = sum(float(order_tons.get(str(oid), 0.0) or 0.0) for oid in orders)
        windows.append(
            {
                "direction": direction,
                "start": start,
                "end": end,
                "back": int(back),
                "fwd": int(fwd),
                "block": block,
                "order_count": len(orders),
                "tons": float(tons),
                "reaches_min": bool(tons >= float(campaign_ton_min) - 1e-6),
            }
        )

    max_neighbor_blocks = max(0, int(max_neighbor_blocks))
    if enable_forward:
        for fwd in range(0, max_neighbor_blocks + 1):
            add_window("forward", 0, fwd)
            if windows and windows[-1].get("direction") == "forward" and windows[-1].get("reaches_min"):
                break

    if enable_backward:
        for back in range(1, max_neighbor_blocks + 1):
            add_window("backward", back, 0)
            if windows and windows[-1].get("direction") == "backward" and windows[-1].get("reaches_min"):
                break

    if enable_bidirectional:
        for total in range(2, max_neighbor_blocks + 1):
            for back in range(1, total):
                fwd = total - back
                add_window("bidirectional", back, fwd)
                if windows and windows[-1].get("direction") == "bidirectional" and windows[-1].get("reaches_min"):
                    break
            if any(w.get("direction") == "bidirectional" and w.get("reaches_min") for w in windows):
                break

    windows.sort(
        key=lambda w: (
            0 if w.get("reaches_min") else 1,
            abs(float(w.get("tons", 0.0)) - float(campaign_ton_min)),
            int(w.get("back", 0)) + int(w.get("fwd", 0)),
            {"forward": 0, "backward": 1, "bidirectional": 2}.get(str(w.get("direction")), 9),
        )
    )
    return windows


def _enumerate_bridge_envelopes(
    *,
    left_anchor_idx: int,
    right_anchor_idx: int,
    order_count: int,
    max_left_expand: int,
    max_right_expand: int,
    max_windows: int,
) -> list[tuple[int, int, int, int]]:
    out: list[tuple[int, int, int, int]] = []
    max_windows = max(1, int(max_windows))
    max_left = min(max(0, int(max_left_expand)), max(0, int(left_anchor_idx)))
    max_right = min(max(0, int(max_right_expand)), max(0, int(order_count) - int(right_anchor_idx) - 1))
    max_span = max(max_left, max_right)
    for span in range(max_span + 1):
        pairs: list[tuple[int, int]]
        if span == 0:
            pairs = [(0, 0)]
        else:
            pairs = [(0, span), (span, 0)]
            pairs.extend((left, span) for left in range(1, span))
            pairs.extend((span, right) for right in range(1, span))
            pairs.append((span, span))
        for left_expand, right_expand in pairs:
            if left_expand > max_left or right_expand > max_right:
                continue
            start_idx = int(left_anchor_idx) - int(left_expand)
            end_idx = int(right_anchor_idx) + int(right_expand)
            if start_idx < 0 or end_idx >= int(order_count) or start_idx > end_idx:
                continue
            out.append((int(left_expand), int(right_expand), int(start_idx), int(end_idx)))
            if len(out) >= int(max_windows):
                return out
    return out


def _pair_fail_location(
    *,
    pair: tuple[str, str],
    pair_global_left_idx: int,
    start_idx: int,
    left_anchor_idx: int,
    right_anchor_idx: int,
    left_expand: int,
    right_expand: int,
    real_edge: tuple[str, str],
) -> str:
    if pair == real_edge:
        return "anchor_pair"
    if int(pair_global_left_idx) < int(left_anchor_idx):
        if int(left_expand) > 0 and int(pair_global_left_idx) == int(start_idx):
            return "left_expansion_boundary"
        return "left_internal"
    if int(pair_global_left_idx) >= int(right_anchor_idx):
        if int(right_expand) > 0 and int(pair_global_left_idx) == int(right_anchor_idx):
            return "right_expansion_boundary"
        return "right_internal"
    return "anchor_pair"


def _classify_pair_rule_failure(
    left_oid: str,
    right_oid: str,
    *,
    template_known: bool,
    order_specs_by_id: dict[str, dict],
    rule,
) -> tuple[str, list[str]]:
    sub_reasons: list[str] = []
    left = order_specs_by_id.get(str(left_oid), {})
    right = order_specs_by_id.get(str(right_oid), {})

    lw = _as_float_or_none(left.get("width"))
    rw = _as_float_or_none(right.get("width"))
    if lw is not None and rw is not None and rule is not None:
        if rw > lw and max(0.0, rw - lw) > float(getattr(rule, "real_reverse_step_max_mm", 0.0) or 0.0):
            sub_reasons.append("WIDTH_RULE_FAIL")
        elif lw - rw > float(getattr(rule, "max_width_drop", 0.0) or 0.0):
            sub_reasons.append("WIDTH_RULE_FAIL")

    lt = _as_float_or_none(left.get("thickness"))
    rt = _as_float_or_none(right.get("thickness"))
    if lt is not None and rt is not None and not _thick_ok(float(lt), float(rt)):
        sub_reasons.append("THICKNESS_RULE_FAIL")

    lmin = _as_float_or_none(left.get("temp_min"))
    lmax = _as_float_or_none(left.get("temp_max"))
    rmin = _as_float_or_none(right.get("temp_min"))
    rmax = _as_float_or_none(right.get("temp_max"))
    if None not in (lmin, lmax, rmin, rmax) and rule is not None:
        overlap = _temp_overlap_len(float(lmin), float(lmax), float(rmin), float(rmax))
        if overlap < float(getattr(rule, "min_temp_overlap_real_real", 0.0) or 0.0):
            sub_reasons.append("TEMP_OVERLAP_FAIL")

    lg = _txt(left.get("steel_group", "")).upper()
    rg = _txt(right.get("steel_group", "")).upper()
    if lg and rg and lg != rg and (not _is_pc(lg)) and (not _is_pc(rg)):
        sub_reasons.append("GROUP_SWITCH_FAIL")

    sub_reasons = list(dict.fromkeys(sub_reasons))
    if len(sub_reasons) > 1:
        return "MULTI_RULE_FAIL", sub_reasons
    if len(sub_reasons) == 1:
        return sub_reasons[0], sub_reasons
    if not template_known:
        return "TEMPLATE_KEY_MISSING", []
    return "UNKNOWN_PAIR_INVALID", []


def _validate_extracted_window_pairs(
    *,
    candidate_id: str,
    line: str,
    extracted_orders: list[str],
    start_idx: int,
    left_anchor_idx: int,
    right_anchor_idx: int,
    left_expand: int,
    right_expand: int,
    direct_edges: set[tuple[str, str]],
    real_edge: tuple[str, str],
    max_real_bridge_per_segment: int,
    pair_rows_by_key: dict[tuple[str, str], list[dict]],
    order_specs_by_id: dict[str, dict],
    rule,
    pre_pair_audit: dict | None = None,
) -> dict:
    failing_pairs: list[dict] = []
    fail_counts: Counter = Counter()
    real_used = 0
    anchor_adjacent = False
    anchor_eval: PairEvalResult | None = None

    for local_idx in range(len(extracted_orders) - 1):
        left_oid = str(extracted_orders[local_idx])
        right_oid = str(extracted_orders[local_idx + 1])
        pair = (left_oid, right_oid)
        eval_result = _evaluate_pair_violation(
            left_order=left_oid,
            right_order=right_oid,
            line=str(line),
            template_lookup=pair_rows_by_key,
            direct_edges=direct_edges,
            real_edges={real_edge},
            adjacency_ok=True,
            order_context_by_id=order_specs_by_id,
            rule=rule,
        )
        if pair == real_edge:
            real_used += 1
            anchor_adjacent = True
            anchor_eval = eval_result
        if eval_result.reason == "OK":
            continue
        reason = str(eval_result.reason or "UNKNOWN_PAIR_INVALID")
        sub_reasons = list(eval_result.sub_reasons)
        fail_counts[reason] += 1
        failing_pairs.append(
            {
                "left_order_id": left_oid,
                "right_order_id": right_oid,
                "template_ok": bool(eval_result.template_ok),
                "adjacency_ok": bool(eval_result.adjacency_ok),
                "reason": reason,
                "sub_reasons": list(sub_reasons),
                "template_key": eval_result.template_key,
                "context_snapshot": dict(eval_result.context_snapshot),
                "location": _pair_fail_location(
                    pair=pair,
                    pair_global_left_idx=int(start_idx) + int(local_idx),
                    start_idx=start_idx,
                    left_anchor_idx=left_anchor_idx,
                    right_anchor_idx=right_anchor_idx,
                    left_expand=left_expand,
                    right_expand=right_expand,
                    real_edge=real_edge,
                ),
            }
        )

    if not anchor_adjacent:
        anchor_eval = _evaluate_pair_violation(
            left_order=str(real_edge[0]),
            right_order=str(real_edge[1]),
            line=str(line),
            template_lookup=pair_rows_by_key,
            direct_edges=direct_edges,
            real_edges={real_edge},
            adjacency_ok=False,
            order_context_by_id=order_specs_by_id,
            rule=rule,
        )
        reason = str(anchor_eval.reason or "UNKNOWN_PAIR_INVALID")
        fail_counts[reason] += 1
        failing_pairs.insert(
            0,
            {
                "left_order_id": str(real_edge[0]),
                "right_order_id": str(real_edge[1]),
                "template_ok": bool(anchor_eval.template_ok),
                "adjacency_ok": bool(anchor_eval.adjacency_ok),
                "reason": reason,
                "sub_reasons": list(anchor_eval.sub_reasons),
                "template_key": anchor_eval.template_key,
                "context_snapshot": dict(anchor_eval.context_snapshot),
                "location": "anchor_pair",
            },
        )

    if anchor_eval is not None:
        _print_pair_audit(
            "REPAIR_BRIDGE_PAIR_AUDIT_EXTRACT",
            candidate_id=str(candidate_id),
            pair=(str(real_edge[0]), str(real_edge[1])),
            line=str(line),
            eval_result=anchor_eval,
        )
        _print_anchor_context(
            candidate_id=str(candidate_id),
            pair=(str(real_edge[0]), str(real_edge[1])),
            line=str(line),
            eval_result=anchor_eval,
        )
        if not anchor_eval.context_ok:
            _print_order_context_miss(
                candidate_id=str(candidate_id),
                pair=(str(real_edge[0]), str(real_edge[1])),
                eval_result=anchor_eval,
            )
        pre = dict(pre_pair_audit or {})
        pre_context = dict(pre.get("context_snapshot", {}) or {})
        extract_context = dict(anchor_eval.context_snapshot)
        changed_fields = _pair_context_changed_fields(pre_context, extract_context)
        for key in ("template_ok", "adjacency_ok", "context_ok", "reason", "template_key"):
            if pre.get(key) != anchor_eval.to_dict().get(key):
                changed_fields.append(key)
        changed_fields = list(dict.fromkeys(changed_fields))
        pre_reason = str(pre.get("reason", "") or "")
        extract_reason = str(anchor_eval.reason or "")
        print(
            f"[APS][REPAIR_BRIDGE_PAIR_AUDIT_DIFF] candidate_id={candidate_id}, "
            f"pair=({real_edge[0]},{real_edge[1]}), changed_fields={changed_fields}, "
            f"pre_reason={pre_reason}, extract_reason={extract_reason}"
        )
        if (
            bool(pre.get("template_ok", False))
            and bool(pre.get("adjacency_ok", False))
            and pre_reason == "OK"
            and extract_reason == "UNKNOWN_PAIR_INVALID"
        ):
            print(
                f"[APS][REPAIR_BRIDGE_ANCHOR_INCONSISTENCY] candidate_id={candidate_id}, "
                f"pair=({real_edge[0]},{real_edge[1]}), pre_reason=OK, "
                f"extract_reason=UNKNOWN_PAIR_INVALID, changed_fields={changed_fields}"
            )

    return {
        "pair_ok": bool(not failing_pairs and 0 < real_used <= int(max_real_bridge_per_segment)),
        "real_used": int(real_used),
        "anchor_adjacent": bool(anchor_adjacent),
        "failing_pairs": failing_pairs,
        "first_fail_reason": str(failing_pairs[0]["reason"]) if failing_pairs else "",
        "fail_counts_by_reason": dict(fail_counts),
    }


def _extract_min_valid_segment_around_bridge(
    *,
    candidate_id: str,
    line: str,
    combined_orders: list[str],
    bridge_from: str,
    bridge_to: str,
    direct_edges: set[tuple[str, str]],
    real_edge: tuple[str, str],
    pair_rows_by_key: dict[tuple[str, str], list[dict]],
    order_specs_by_id: dict[str, dict],
    rule,
    order_tons: dict[str, float],
    campaign_ton_min: float,
    campaign_ton_max: float,
    campaign_ton_target: float,
    max_real_bridge_per_segment: int,
    next_local_id: int,
    max_left_expand: int,
    max_right_expand: int,
    max_windows_per_candidate: int,
    pre_pair_audit: dict | None = None,
) -> tuple[CampaignSegment | None, CampaignSegment | None, dict]:
    diag = {
        "success": False,
        "left_expand": 0,
        "right_expand": 0,
        "extracted_orders": 0,
        "extracted_tons": 0.0,
        "pair_ok": False,
        "ton_ok": False,
        "first_fail_reason": "",
        "fail_pair_count": 0,
        "fail_counts_by_reason": {},
        "dominant_fail_reason": "",
        "first_success_window": "",
        "left_blocked_by": "",
        "right_blocked_by": "",
        "residual_underfilled": 0,
        "partial_success": False,
        "full_success": False,
        "real_bridge_used": 0,
        "envelopes_tested": 0,
    }
    oids = [str(v) for v in combined_orders]
    if not oids:
        return None, None, diag

    anchor_pairs = [
        (left_idx, right_idx)
        for left_idx, oid in enumerate(oids)
        if str(oid) == str(bridge_from)
        for right_idx, right_oid in enumerate(oids)
        if str(right_oid) == str(bridge_to) and int(right_idx) > int(left_idx)
    ]
    if not anchor_pairs:
        print(
            f"[APS][REPAIR_BRIDGE_TON_RESCUE_EXTRACT_ERROR] candidate_id={candidate_id}, "
            f"reason=ANCHOR_NOT_FOUND_OR_ORDER_INVALID"
        )
        return None, None, {**diag, "reject_reason": "TON_SPLIT_NOT_FOUND"}

    left_anchor_idx, right_anchor_idx = min(anchor_pairs, key=lambda item: (int(item[1]) - int(item[0]), int(item[0])))

    prefix = [0.0] * (len(oids) + 1)
    for idx, oid in enumerate(oids):
        prefix[idx + 1] = prefix[idx] + float(order_tons.get(str(oid), 0.0) or 0.0)

    candidates: list[dict] = []
    fail_counts_total: Counter = Counter()
    left_direction_fails: Counter = Counter()
    right_direction_fails: Counter = Counter()
    first_success_window = ""
    envelopes = _enumerate_bridge_envelopes(
        left_anchor_idx=int(left_anchor_idx),
        right_anchor_idx=int(right_anchor_idx),
        order_count=len(oids),
        max_left_expand=max_left_expand,
        max_right_expand=max_right_expand,
        max_windows=max_windows_per_candidate,
    )
    for left_expand, right_expand, start, end in envelopes:
        segment_orders = oids[start : end + 1]
        tons = prefix[end + 1] - prefix[start]
        ton_ok = float(campaign_ton_min) - 1e-6 <= float(tons) <= float(campaign_ton_max) + 1e-6
        if float(tons) > float(campaign_ton_max) + 1e-6:
            diag["envelopes_tested"] = int(diag.get("envelopes_tested", 0) or 0) + 1
            print(
                f"[APS][REPAIR_BRIDGE_TON_RESCUE_EXTRACT] candidate_id={candidate_id}, "
                f"left_expand={left_expand}, right_expand={right_expand}, "
                f"extracted_orders={len(segment_orders)}, extracted_tons={float(tons):.1f}, "
                f"pair_ok=False, ton_ok=False, success=False, "
                f"first_fail_reason=TON_ABOVE_MAX_AFTER_EXPANSION, fail_pair_count=0, "
                f"fail_counts_by_reason={{'TON_ABOVE_MAX_AFTER_EXPANSION': 1}}"
            )
            continue
        pair_diag = _validate_extracted_window_pairs(
            candidate_id=candidate_id,
            line=str(line),
            extracted_orders=segment_orders,
            start_idx=start,
            left_anchor_idx=int(left_anchor_idx),
            right_anchor_idx=int(right_anchor_idx),
            left_expand=left_expand,
            right_expand=right_expand,
            direct_edges=direct_edges,
            real_edge=real_edge,
            max_real_bridge_per_segment=max_real_bridge_per_segment,
            pair_rows_by_key=pair_rows_by_key,
            order_specs_by_id=order_specs_by_id,
            rule=rule,
            pre_pair_audit=pre_pair_audit,
        )
        pair_ok = bool(pair_diag.get("pair_ok", False))
        real_used = int(pair_diag.get("real_used", 0) or 0)
        success = bool(pair_ok and ton_ok)
        diag["envelopes_tested"] = int(diag.get("envelopes_tested", 0) or 0) + 1
        failing_pairs = list(pair_diag.get("failing_pairs", []) or [])
        first_fail = failing_pairs[0] if failing_pairs else {}
        fail_counts_by_reason = dict(pair_diag.get("fail_counts_by_reason", {}) or {})
        fail_counts_total.update(fail_counts_by_reason)
        if first_fail:
            fail_reason = str(first_fail.get("reason", "") or "UNKNOWN_PAIR_INVALID")
            location = str(first_fail.get("location", "") or "unknown")
            if location.startswith("left_"):
                left_direction_fails[fail_reason] += 1
            elif location.startswith("right_"):
                right_direction_fails[fail_reason] += 1
            print(
                f"[APS][REPAIR_BRIDGE_TON_RESCUE_PAIR_FAIL] candidate_id={candidate_id}, "
                f"left_expand={left_expand}, right_expand={right_expand}, "
                f"fail_location={location}, "
                f"fail_pair=({first_fail.get('left_order_id', '')},{first_fail.get('right_order_id', '')}), "
                f"reason={fail_reason}"
            )
            if str(first_fail.get("reason", "")) == "MULTI_RULE_FAIL":
                print(
                    f"[APS][REPAIR_BRIDGE_TON_RESCUE_PAIR_FAIL_DETAIL] candidate_id={candidate_id}, "
                    f"fail_pair=({first_fail.get('left_order_id', '')},{first_fail.get('right_order_id', '')}), "
                    f"sub_reasons={list(first_fail.get('sub_reasons', []) or [])}"
                )
        print(
            f"[APS][REPAIR_BRIDGE_TON_RESCUE_EXTRACT] candidate_id={candidate_id}, "
            f"left_expand={left_expand}, right_expand={right_expand}, "
            f"extracted_orders={len(segment_orders)}, extracted_tons={float(tons):.1f}, "
            f"pair_ok={bool(pair_ok)}, ton_ok={bool(ton_ok)}, success={bool(success)}, "
            f"first_fail_reason={str(pair_diag.get('first_fail_reason', '') or '')}, "
            f"fail_pair_count={len(failing_pairs)}, "
            f"fail_counts_by_reason={fail_counts_by_reason}"
        )
        if not success:
            continue
        if not first_success_window:
            first_success_window = f"left_expand={left_expand},right_expand={right_expand}"
        candidates.append(
            {
                "start": int(start),
                "end": int(end),
                "orders": segment_orders,
                "tons": float(tons),
                "real_used": int(real_used),
                "target_gap": abs(float(tons) - float(campaign_ton_target)),
                "order_count": len(segment_orders),
                "residual_count": max(0, len(oids) - len(segment_orders)),
                "left_expand": int(left_expand),
                "right_expand": int(right_expand),
                "expand_total": int(left_expand) + int(right_expand),
            }
        )

    if not candidates:
        dominant_fail_reason = fail_counts_total.most_common(1)[0][0] if fail_counts_total else ""
        left_blocked_by = left_direction_fails.most_common(1)[0][0] if left_direction_fails else ""
        right_blocked_by = right_direction_fails.most_common(1)[0][0] if right_direction_fails else ""
        if left_blocked_by or right_blocked_by:
            print(
                f"[APS][REPAIR_BRIDGE_TON_RESCUE_DIRECTION_HINT] candidate_id={candidate_id}, "
                f"left_blocked_by={left_blocked_by}, right_blocked_by={right_blocked_by}"
            )
        diag.update(
            {
                "first_fail_reason": str(dominant_fail_reason),
                "dominant_fail_reason": str(dominant_fail_reason),
                "fail_pair_count": int(sum(fail_counts_total.values())),
                "fail_counts_by_reason": dict(fail_counts_total),
                "left_blocked_by": str(left_blocked_by),
                "right_blocked_by": str(right_blocked_by),
                "first_success_window": "",
                "total_windows_tested": int(diag.get("envelopes_tested", 0) or 0),
            }
        )
        return None, None, diag

    best = min(
        candidates,
        key=lambda item: (
            float(item["target_gap"]),
            int(item["order_count"]),
            int(item["residual_count"]),
            int(item["expand_total"]),
        ),
    )
    extracted_orders = [str(v) for v in best["orders"]]
    residual_orders = oids[: int(best["start"])] + oids[int(best["end"]) + 1 :]

    valid_seg = _campaign_segment_from_orders(
        line=line,
        campaign_local_id=next_local_id + 1,
        order_ids=extracted_orders,
        order_tons=order_tons,
        is_valid=True,
    )
    if valid_seg is None:
        return None, None, diag

    residual_seg = _campaign_segment_from_orders(
        line=line,
        campaign_local_id=next_local_id + 2,
        order_ids=residual_orders,
        order_tons=order_tons,
        is_valid=False,
    )
    diag.update(
        {
            "success": True,
            "left_expand": int(best["left_expand"]),
            "right_expand": int(best["right_expand"]),
            "extracted_orders": int(len(extracted_orders)),
            "extracted_tons": float(best["tons"]),
            "pair_ok": True,
            "ton_ok": True,
            "first_fail_reason": "",
            "dominant_fail_reason": fail_counts_total.most_common(1)[0][0] if fail_counts_total else "",
            "fail_pair_count": int(sum(fail_counts_total.values())),
            "fail_counts_by_reason": dict(fail_counts_total),
            "first_success_window": first_success_window,
            "total_windows_tested": int(diag.get("envelopes_tested", 0) or 0),
            "residual_underfilled": 1 if residual_seg is not None else 0,
            "partial_success": residual_seg is not None,
            "full_success": residual_seg is None,
            "real_bridge_used": int(best["real_used"]),
        }
    )
    return valid_seg, residual_seg, diag


def _try_bridge_ton_rescue_recut(
    *,
    candidate: dict,
    line: str,
    block_id: int,
    window_id: int,
    direction: str,
    expansion_block: list[CampaignSegment],
    current_block_segment_count: int,
    current_block_segment_ids: list[int],
    direct_edges: set[tuple[str, str]],
    real_edge: tuple[str, str],
    order_tons: dict[str, float],
    campaign_ton_min: float,
    campaign_ton_max: float,
    campaign_ton_target: float,
    max_real_bridge_per_segment: int,
    bridge_cost_penalty: float,
    next_local_id: int,
    ton_rescue_max_left_expand: int,
    ton_rescue_max_right_expand: int,
    ton_rescue_max_windows_per_candidate: int,
    pair_rows_by_key: dict[tuple[str, str], list[dict]],
    order_specs_by_id: dict[str, dict],
    rule,
    real_edge_distance: dict[tuple[str, str], int] | None = None,
    real_edge_adjustment_cost: dict[tuple[str, str], int] | None = None,
) -> dict:
    candidate_id = str(candidate.get("candidate_id", "BRIDGE_CAND"))
    bridge_from = str(candidate.get("bridge_from", real_edge[0]))
    bridge_to = str(candidate.get("bridge_to", real_edge[1]))
    ctx = candidate.get("context", {}) if isinstance(candidate.get("context", {}), dict) else {}
    adjusted_left_orders = [str(v) for v in ctx.get("adjusted_left_orders", [])]
    adjusted_right_orders = [str(v) for v in ctx.get("adjusted_right_orders", [])]
    pre_pair_audit = dict(candidate.get("pre_pair_audit", {}) or {})
    combined_orders = [str(oid) for seg in expansion_block for oid in seg.order_ids]
    combined_tons = sum(float(order_tons.get(str(oid), 0.0) or 0.0) for oid in combined_orders)
    current_block_segment_ids_set = set(int(x) for x in current_block_segment_ids)
    neighbor_ids = [
        int(seg.campaign_local_id)
        for seg in expansion_block
        if int(seg.campaign_local_id) not in current_block_segment_ids_set
    ]
    diag = {}
    if not neighbor_ids:
        diag["neighbor_block_count"] = 0
    else:
        diag["neighbor_block_count"] = len(neighbor_ids)
    print(
        f"[APS][REPAIR_BRIDGE_TON_RESCUE_NEIGHBORS] candidate_id={candidate_id}, "
        f"neighbor_ids={neighbor_ids}"
    )
    print(
        f"[APS][REPAIR_BRIDGE_TON_RESCUE_WINDOW] candidate_id={candidate_id}, "
        f"window_id={window_id}, direction={direction}, neighbor_blocks={neighbor_ids}, "
        f"combined_orders={len(combined_orders)}, combined_tons={combined_tons:.1f}"
    )
    print(f"[APS][REPAIR_BRIDGE_STAGE] candidate_id={candidate_id}, stage=TON_RESCUE_RECUT")

    if bridge_from not in adjusted_left_orders or bridge_to not in adjusted_right_orders:
        print(
            f"[APS][REPAIR_BRIDGE_TON_RESCUE_FAIL] candidate_id={candidate_id}, "
            f"window_id={window_id}, reason=TEMPLATE_PAIR_INVALID"
        )
        print(f"[APS][REPAIR_BRIDGE_STAGE] candidate_id={candidate_id}, stage=TON_RESCUE_REJECTED")
        return {"success": False, "reject_reason": "TEMPLATE_PAIR_INVALID"}

    left_materialized = adjusted_left_orders[: adjusted_left_orders.index(bridge_from) + 1]
    right_materialized = adjusted_right_orders[adjusted_right_orders.index(bridge_to) :]
    anchor_orders = left_materialized + right_materialized
    used = Counter(anchor_orders)
    append_orders: list[str] = []
    for oid in combined_orders:
        if used[str(oid)] > 0:
            used[str(oid)] -= 1
        else:
            append_orders.append(str(oid))
    rescue_orders = anchor_orders + append_orders
    print(
        f"[APS][REPAIR_BRIDGE_TON_RESCUE_MATERIALIZE] candidate_id={candidate_id}, "
        f"window_id={window_id}, anchor_pair=({bridge_from},{bridge_to}), "
        f"left_anchor_tail={left_materialized[-5:]}, right_anchor_head={right_materialized[:5]}"
    )

    if not rescue_orders:
        print(
            f"[APS][REPAIR_BRIDGE_TON_RESCUE_FAIL] candidate_id={candidate_id}, "
            f"window_id={window_id}, reason=TON_SPLIT_NOT_FOUND"
        )
        print(f"[APS][REPAIR_BRIDGE_STAGE] candidate_id={candidate_id}, stage=TON_RESCUE_REJECTED")
        return {"success": False, "reject_reason": "TON_SPLIT_NOT_FOUND"}
    rescue_tons = sum(float(order_tons.get(str(oid), 0.0) or 0.0) for oid in rescue_orders)
    if rescue_tons < float(campaign_ton_min) - 1e-6:
        print(
            f"[APS][REPAIR_BRIDGE_TON_RESCUE_RECUT] candidate_id={candidate_id}, "
            f"window_id={window_id}, success=False, rescued_valid=0, residual_underfilled=1"
        )
        print(
            f"[APS][REPAIR_BRIDGE_TON_RESCUE_FAIL] candidate_id={candidate_id}, "
            f"window_id={window_id}, reason=TON_BELOW_MIN_EVEN_AFTER_NEIGHBOR_EXPANSION"
        )
        print(f"[APS][REPAIR_BRIDGE_STAGE] candidate_id={candidate_id}, stage=TON_RESCUE_REJECTED")
        return {"success": False, "reject_reason": "TON_BELOW_MIN_EVEN_AFTER_NEIGHBOR_EXPANSION"}

    extracted_valid, residual_seg, recut_diag = _extract_min_valid_segment_around_bridge(
        candidate_id=candidate_id,
        line=line,
        combined_orders=combined_orders,
        bridge_from=bridge_from,
        bridge_to=bridge_to,
        direct_edges=direct_edges,
        real_edge=real_edge,
        pair_rows_by_key=pair_rows_by_key,
        order_specs_by_id=order_specs_by_id,
        rule=rule,
        pre_pair_audit=pre_pair_audit,
        order_tons=order_tons,
        campaign_ton_min=campaign_ton_min,
        campaign_ton_max=campaign_ton_max,
        campaign_ton_target=campaign_ton_target,
        max_real_bridge_per_segment=max_real_bridge_per_segment,
        next_local_id=next_local_id,
        max_left_expand=ton_rescue_max_left_expand,
        max_right_expand=ton_rescue_max_right_expand,
        max_windows_per_candidate=ton_rescue_max_windows_per_candidate,
    )
    print(
        f"[APS][REPAIR_BRIDGE_TON_RESCUE_EXTRACT_SUMMARY] candidate_id={candidate_id}, "
        f"left_expand={int(recut_diag.get('left_expand', 0) or 0)}, "
        f"right_expand={int(recut_diag.get('right_expand', 0) or 0)}, "
        f"extracted_orders={int(recut_diag.get('extracted_orders', 0) or 0)}, "
        f"extracted_tons={float(recut_diag.get('extracted_tons', 0.0) or 0.0):.1f}, "
        f"pair_ok={bool(recut_diag.get('pair_ok', False))}, "
        f"ton_ok={bool(recut_diag.get('ton_ok', False))}, "
        f"dominant_fail_reason={str(recut_diag.get('dominant_fail_reason', '') or '')}, "
        f"first_success_window={str(recut_diag.get('first_success_window', '') or '')}, "
        f"total_windows_tested={int(recut_diag.get('total_windows_tested', recut_diag.get('envelopes_tested', 0)) or 0)}, "
        f"success={bool(recut_diag.get('success', False))}"
    )

    if extracted_valid is None:
        reason = str(recut_diag.get("reject_reason", "") or "TON_SPLIT_NOT_FOUND")
        print(
            f"[APS][REPAIR_BRIDGE_TON_RESCUE_RECUT] candidate_id={candidate_id}, "
            f"window_id={window_id}, success=False, rescued_valid=0, residual_underfilled=1"
        )
        print(f"[APS][REPAIR_BRIDGE_TON_RESCUE_FAIL] candidate_id={candidate_id}, window_id={window_id}, reason={reason}")
        print(f"[APS][REPAIR_BRIDGE_STAGE] candidate_id={candidate_id}, stage=TON_RESCUE_REJECTED")
        return {"success": False, "reject_reason": reason, "diag": recut_diag}

    valid = [extracted_valid]
    residual_all = [residual_seg] if residual_seg is not None else []

    before_counter = Counter(combined_orders)
    final_counter = Counter()
    for seg in valid + residual_all:
        final_counter.update([str(v) for v in seg.order_ids])
    if before_counter != final_counter:
        print(
            f"[APS][REPAIR_BRIDGE_TON_RESCUE_FAIL] candidate_id={candidate_id}, "
            f"window_id={window_id}, reason=MULTIPLICITY_INVALID"
        )
        print(f"[APS][REPAIR_BRIDGE_STAGE] candidate_id={candidate_id}, stage=TON_RESCUE_REJECTED")
        return {"success": False, "reject_reason": "MULTIPLICITY_INVALID"}

    bridge_pair_used = any(
        any((str(seg.order_ids[i]), str(seg.order_ids[i + 1])) == real_edge for i in range(len(seg.order_ids) - 1))
        for seg in valid
    )
    if not bridge_pair_used:
        print(
            f"[APS][REPAIR_BRIDGE_TON_RESCUE_RECUT] candidate_id={candidate_id}, "
            f"window_id={window_id}, success=False, rescued_valid={len(valid)}, residual_underfilled={len(residual_all)}"
        )
        print(
            f"[APS][REPAIR_BRIDGE_TON_RESCUE_FAIL] candidate_id={candidate_id}, "
            f"window_id={window_id}, reason=ANCHOR_PAIR_NOT_USED"
        )
        print(f"[APS][REPAIR_BRIDGE_STAGE] candidate_id={candidate_id}, stage=TON_RESCUE_REJECTED")
        return {"success": False, "reject_reason": "ANCHOR_PAIR_NOT_USED"}

    valid_delta = len(valid)
    underfilled_delta = len(residual_all) - len(expansion_block)
    scheduled_orders_delta = sum(len(seg.order_ids) for seg in valid)
    improved = valid_delta > 0 or underfilled_delta < 0 or scheduled_orders_delta > 0
    print(
        f"[APS][REPAIR_BRIDGE_TON_RESCUE_RECUT] candidate_id={candidate_id}, "
        f"window_id={window_id}, success={bool(improved)}, rescued_valid={len(valid)}, "
        f"residual_underfilled={len(residual_all)}, "
        f"partial_success={bool(recut_diag.get('partial_success', False))}"
    )
    if not improved:
        print(
            f"[APS][REPAIR_BRIDGE_TON_RESCUE_FAIL] candidate_id={candidate_id}, "
            f"window_id={window_id}, reason=TON_RESCUE_NO_GAIN"
        )
        print(f"[APS][REPAIR_BRIDGE_STAGE] candidate_id={candidate_id}, stage=TON_RESCUE_REJECTED")
        return {"success": False, "reject_reason": "TON_RESCUE_NO_GAIN"}

    print(
        f"[APS][REPAIR_BRIDGE_TON_RESCUE_ACCEPT] candidate_id={candidate_id}, "
        f"window_id={window_id}, left_expand={int(recut_diag.get('left_expand', 0) or 0)}, "
        f"right_expand={int(recut_diag.get('right_expand', 0) or 0)}, "
        f"extracted_tons={float(recut_diag.get('extracted_tons', 0.0) or 0.0):.1f}, "
        f"rescued_valid={len(valid)}, residual_underfilled={len(residual_all)}, "
        f"valid_delta={valid_delta}, underfilled_delta={underfilled_delta}, "
        f"scheduled_orders_delta={scheduled_orders_delta}, "
        f"partial_success={bool(recut_diag.get('partial_success', False))}"
    )
    print(f"[APS][REPAIR_BRIDGE_STAGE] candidate_id={candidate_id}, stage=TON_RESCUE_ACCEPTED")
    recut_diag.update(
        {
            "success": True,
            "neighbor_block_count": int(diag.get("neighbor_block_count", 0) or 0),
            "extract_success": True,
            "partial_success": bool(recut_diag.get("partial_success", False)),
            "full_success": bool(recut_diag.get("full_success", False)),
            "valid_delta": int(valid_delta),
            "underfilled_delta": int(underfilled_delta),
            "scheduled_orders_delta": int(scheduled_orders_delta),
            "salvaged_orders": int(scheduled_orders_delta),
            "real_bridge_used": max(1, int(recut_diag.get("real_bridge_used", 0) or 0)),
            "residual_underfilled_count": int(len(residual_all)),
        }
    )
    return {
        "success": True,
        "valid": valid,
        "underfilled": residual_all,
        "diag": recut_diag,
        "valid_delta": int(valid_delta),
        "underfilled_delta": int(underfilled_delta),
        "scheduled_orders_delta": int(scheduled_orders_delta),
    }


def _reconstruct_underfilled_segments(
    valid_segments: List[CampaignSegment],
    underfilled_segments: List[CampaignSegment],
    transition_pack,
    campaign_ton_min: float,
    campaign_ton_max: float,
    diagnostics: dict,
    allow_real_bridge: bool = False,
    allow_virtual_bridge: bool = False,
    order_tons: dict[str, float] | None = None,
    orders_df: pd.DataFrame | None = None,
    cfg: PlannerConfig | None = None,
    family_repair_already_attempted_keys: set | None = None,
) -> tuple[list[CampaignSegment], list[CampaignSegment], dict]:
    """
    Repair-only local reconstruction for underfilled segments.

    Initial constructive and initial cutter remain direct-only. This stage only
    rebuilds same-line underfilled blocks and optionally allows REAL_BRIDGE_EDGE
    as a repair candidate after direct-only reconstruction has no gain.
    """
    t0 = perf_counter()
    # Initialize family_repair_already_attempted_keys if None (for compatibility)
    _family_dedup_keys: set = set(family_repair_already_attempted_keys) if family_repair_already_attempted_keys else set()

    diag = {
        "underfilled_reconstruction_enabled": True,
        "underfilled_reconstruction_attempts": 0,
        "underfilled_reconstruction_success": 0,
        "underfilled_reconstruction_blocks_tested": 0,
        "underfilled_reconstruction_blocks_skipped": 0,
        "underfilled_reconstruction_valid_before": int(len(valid_segments)),
        "underfilled_reconstruction_valid_after": int(len(valid_segments)),
        "underfilled_reconstruction_underfilled_before": int(len(underfilled_segments)),
        "underfilled_reconstruction_underfilled_after": int(len(underfilled_segments)),
        "underfilled_reconstruction_segments_salvaged": 0,
        "underfilled_reconstruction_orders_salvaged": 0,
        "underfilled_reconstruction_valid_delta": 0,
        "underfilled_reconstruction_underfilled_delta": 0,
        "underfilled_reconstruction_not_entered_reason": "",
        "repair_only_real_bridge_enabled": bool(allow_real_bridge),
        "repair_only_real_bridge_attempts": 0,
        "repair_only_real_bridge_success": 0,
        "repair_only_real_bridge_candidates_total": 0,
        "repair_only_real_bridge_candidates_kept": 0,
        "repair_only_real_bridge_filtered_direct_feasible": 0,
        "repair_only_real_bridge_filtered_pair_invalid": 0,
        "repair_only_real_bridge_filtered_ton_invalid": 0,
        "repair_only_real_bridge_filtered_score_worse": 0,
        "repair_only_real_bridge_filtered_bridge_limit_exceeded": 0,
        "repair_only_real_bridge_filtered_multiplicity_invalid": 0,
        "repair_only_real_bridge_filtered_bridge_path_not_real": 0,
        "repair_only_real_bridge_filtered_bridge_path_missing": 0,
        "repair_only_real_bridge_filtered_block_order_mismatch": 0,
        "repair_only_real_bridge_filtered_line_mismatch": 0,
        "repair_only_real_bridge_filtered_block_membership_mismatch": 0,
        "repair_only_real_bridge_filtered_bridge_path_payload_empty": 0,
        "repair_bridge_pack_has_real_rows": False,
        "repair_bridge_raw_rows_total": 0,
        "repair_bridge_matched_rows_total": 0,
        "repair_bridge_kept_rows_total": 0,
        # Family repair dedup diagnostics (for ALNS vs reconstruction role clarity)
        "family_repair_already_attempted_key_count": len(_family_dedup_keys),
        "repair_virtual_family_skipped_due_to_existing_attempt_count": 0,
        "repair_bridge_endpoint_key_mismatch_count": 0,
        "repair_bridge_field_name_mismatch_count": 0,
        "repair_bridge_inconsistency_count": 0,
        "repair_bridge_boundary_band_enabled": True,
        "repair_bridge_band_pairs_tested": 0,
        "repair_bridge_band_hits": 0,
        "repair_bridge_single_point_hits": 0,
        "repair_bridge_band_only_hits": 0,
        "repair_bridge_band_best_distance": -1,
        "repair_bridge_endpoint_adjustment_enabled": True,
        "repair_bridge_adjustments_generated": 0,
        "repair_bridge_adjustment_pairs_tested": 0,
        "repair_bridge_adjustment_hits": 0,
        "repair_bridge_adjustment_only_hits": 0,
        "repair_bridge_best_adjustment_cost": -1,
        "repair_bridge_candidates_matched": 0,
        "repair_bridge_candidates_rejected_pair_invalid": 0,
        "repair_bridge_candidates_rejected_ton_invalid": 0,
        "repair_bridge_candidates_rejected_score_worse": 0,
        "repair_bridge_candidates_accepted": 0,
        "repair_bridge_exact_invalid_pair_count": 0,
        "repair_bridge_frontier_mismatch_count": 0,
        "repair_bridge_pair_invalid_width": 0,
        "repair_bridge_pair_invalid_thickness": 0,
        "repair_bridge_pair_invalid_temp": 0,
        "repair_bridge_pair_invalid_group": 0,
        "repair_bridge_pair_invalid_context": 0,
        "repair_bridge_pair_invalid_unknown": 0,
        "repair_bridge_pair_fail_thickness": 0,
        "repair_bridge_prefilter_fail_thickness": 0,
        "repair_bridge_rank_pass_thickness": 0,
        "repair_bridge_rank_fail_thickness": 0,
        "repair_bridge_template_no_edge_count": 0,
        "repair_bridge_prefilter_all_fail_count": 0,
        "repair_bridge_band_too_narrow_count": 0,
        "repair_bridge_prefilter_reject_count": 0,
        "repair_bridge_endpoint_early_stop_count": 0,
        "repair_bridge_local_band_retry_count": 0,
        "underfilled_reconstruction_improvement_recorded_count": 0,
        "underfilled_reconstruction_improvement_applied_count": 0,
        "underfilled_reconstruction_apply_reject_count": 0,
        "bridgeability_route_suggestion": "",
        "bridgeability_census": {},
        "bridgeability_census_items": [],
        # ---- Legacy virtual pilot: 降级为单一布尔字段 ----
        # 旧有细粒度 virtual_pilot 字段已从此处移除；
        # 只保留一个总布尔字段，表明 virtual_pilot 在本运行中被禁用。
        # 旧有字段仍在 internal virtual pilot 代码路径中兼容记录，
        # 但不在此处输出（避免大量空值污染 engine_meta）。
        "virtual_pilot_skipped_due_to_disabled": True,  # 主线始终 True（pilot disabled）
        # ---- Internal compat counters: 内部代码路径会写入这些字段；
        #     不暴露到 engine_meta/writer summary（避免大量空值污染）。
        #     初始化为 0 / {} 以保证 +=1 操作安全。
        "virtual_pilot_attempt_count": 0,
        "virtual_pilot_success_count": 0,
        "virtual_pilot_apply_count": 0,
        "virtual_pilot_reject_count": 0,
        "virtual_pilot_skipped_block_count": 0,
        "virtual_pilot_skipped_due_to_disabled_count": 0,
        "virtual_pilot_skipped_due_to_limit_count": 0,
        "virtual_pilot_skipped_due_to_no_pilotable_candidate_count": 0,
        "virtual_pilot_selected_block_count": 0,
        "virtual_pilot_selected_unique_pilot_key_count": 0,
        "virtual_pilot_structural_eligible_block_count": 0,
        "virtual_pilot_runtime_enabled_block_count": 0,
        "virtual_pilot_final_eligible_block_count": 0,
        "virtual_pilot_eligible_block_count": 0,
        "virtual_pilot_duplicate_candidate_skipped_count": 0,
        "virtual_pilot_dedup_group_count": 0,
        "virtual_pilot_small_block_soft_penalty_count": 0,
        "virtual_pilot_spec_enum_total": 0,
        "virtual_pilot_spec_enum_both_valid_count": 0,
        "virtual_pilot_spec_enum_done_count": 0,
        "virtual_pilot_family_prefilter_fail_count": 0,
        "virtual_pilot_width_group_family_attempt_count": 0,
        "virtual_pilot_thickness_family_attempt_count": 0,
        "virtual_pilot_ton_fill_attempt_count": 0,
        "virtual_pilot_ton_fill_success_count": 0,
        "virtual_pilot_selected_candidate_count": 0,
        "virtual_pilot_dedup_kept_count": 0,
        "virtual_pilot_dedup_skipped_count": 0,
        "virtual_pilot_attempt_started_count": 0,
        "virtual_pilot_recut_entered_count": 0,
        "virtual_pilot_segment_valid_count": 0,
        "virtual_pilot_ton_fill_entered_count": 0,
        "virtual_pilot_apply_check_entered_count": 0,
        "virtual_pilot_apply_success_count": 0,
        "virtual_pilot_selected_by_bucket_count": {},
        "virtual_pilot_selected_by_family_count": {},
        "virtual_pilot_fail_stage_count": {},
        "virtual_pilot_post_spec_fail_stage_count": {},
        "virtual_pilot_reject_by_reason_count": {},
        "conservative_apply_attempt_count": 0,
        "conservative_apply_success_count": 0,
        "conservative_apply_reject_count": 0,
        "repair_bridge_ton_rescue_attempts": 0,
        "repair_bridge_ton_rescue_success": 0,
        "repair_bridge_ton_rescue_windows_tested": 0,
        "repair_bridge_ton_rescue_valid_delta": 0,
        "repair_bridge_ton_rescue_underfilled_delta": 0,
        "repair_bridge_ton_rescue_scheduled_orders_delta": 0,
        "repair_bridge_ton_rescue_extract_attempts": 0,
        "repair_bridge_ton_rescue_extract_success": 0,
        "repair_bridge_ton_rescue_partial_success": 0,
        "repair_bridge_ton_rescue_full_success": 0,
        "repair_bridge_ton_rescue_residual_underfilled_count": 0,
        "repair_bridge_ton_rescue_pair_fail_width": 0,
        "repair_bridge_ton_rescue_pair_fail_thickness": 0,
        "repair_bridge_ton_rescue_pair_fail_temp": 0,
        "repair_bridge_ton_rescue_pair_fail_group": 0,
        "repair_bridge_ton_rescue_pair_fail_context": 0,
        "repair_bridge_ton_rescue_pair_fail_template": 0,
        "repair_bridge_ton_rescue_pair_fail_multi": 0,
        "repair_bridge_ton_rescue_pair_fail_unknown": 0,
        "repair_bridge_filtered_ton_below_min_current_block": 0,
        "repair_bridge_filtered_ton_below_min_even_after_neighbor_expansion": 0,
        "repair_bridge_filtered_ton_above_max_after_expansion": 0,
        "repair_bridge_filtered_ton_split_not_found": 0,
        "repair_bridge_filtered_ton_rescue_no_gain": 0,
        "repair_bridge_filtered_ton_rescue_impossible": 0,
        "repair_only_real_bridge_used_segments": 0,
        "repair_only_real_bridge_used_orders": 0,
        "repair_only_real_bridge_not_entered_reason": "",
        "reconstruction_no_gain": True,
        "underfilled_reconstruction_seconds": 0.0,
        "repair_only_real_bridge_seconds": 0.0,
        "reconstruction_seconds": 0.0,
        "repair_bridge_real_seconds": 0.0,
    }
    lines = sorted({str(s.line) for s in underfilled_segments})
    tpl_df = _transition_templates_df(transition_pack)
    order_specs_by_id = _build_order_spec_map_from_transition_context(transition_pack, tpl_df, orders_df=orders_df)
    bridgeability_census = BridgeabilityCensus()
    pack_meta = _pack_debug_meta(transition_pack)
    diag.update(
        {
            "repair_bridge_pack_type": pack_meta.get("transition_pack_type", ""),
            "repair_bridge_pack_keys": pack_meta.get("keys", []),
            "repair_bridge_pack_line_keys": pack_meta.get("line_keys", []),
            "repair_bridge_pack_has_real_rows": bool(len(_bridge_rows_for_line(tpl_df, None)) > 0),
            "repair_bridge_pack_real_rows_total": int(len(_bridge_rows_for_line(tpl_df, None))),
            "repair_bridge_pack_virtual_rows_total": int(len(_virtual_bridge_rows_for_line(tpl_df, None))),
        }
    )
    print(
        f"[APS][REPAIR_BRIDGE_PACK] transition_pack_type={pack_meta.get('transition_pack_type')}, "
        f"keys={pack_meta.get('keys')}, line_keys={pack_meta.get('line_keys')}, "
        f"has_transition_df={pack_meta.get('has_transition_df')}, "
        f"has_template_df={pack_meta.get('has_template_df')}, "
        f"has_bridge_paths={pack_meta.get('has_bridge_paths')}, "
        f"has_real_bridge_edges={pack_meta.get('has_real_bridge_edges')}, "
        f"has_virtual_bridge_edges={pack_meta.get('has_virtual_bridge_edges')}, "
        f"has_transition_lookup={pack_meta.get('has_transition_lookup')}, "
        f"has_edge_lookup={pack_meta.get('has_edge_lookup')}, "
        f"has_path_lookup={pack_meta.get('has_path_lookup')}"
    )
    for line in sorted(set(lines + [str(r.get("line")) for r in _bridge_rows_for_line(tpl_df, None) if r.get("line", None) is not None])):
        real_rows_line = _bridge_rows_for_line(tpl_df, line)
        virtual_rows_line = _virtual_bridge_rows_for_line(tpl_df, line)
        if line:
            print(
                f"[APS][REPAIR_BRIDGE_PACK_LINE] line={line}, "
                f"size={len(tpl_df) if not tpl_df.empty else 0}, "
                f"real_bridge_rows={len(real_rows_line)}, virtual_bridge_rows={len(virtual_rows_line)}"
            )
    print(
        f"[APS][UNDERFILLED_RECON_ENTER] enabled=True, valid_in={len(valid_segments)}, "
        f"underfilled_in={len(underfilled_segments)}, lines={lines}"
    )

    # =========================================================================
    # Gain precheck: only continue if there's a realistic improvement signal
    # =========================================================================
    # Add cutter optimization diagnostics
    diag.update({
        "cutter_blocks_touched": 0,
        "cutter_blocks_improved": 0,
        "cutter_blocks_skipped_by_precheck": 0,
        "cutter_blocks_skipped_by_no_gain_set": 0,
        "cutter_no_gain_streak_max": 0,
    })

    # Precheck: build useful signals
    _all_under = valid_segments + underfilled_segments
    _has_underfilled = bool(underfilled_segments)
    _gap_to_min_tons = max(0.0, campaign_ton_min - min((s.total_tons for s in underfilled_segments), default=0.0))
    _gap_threshold = campaign_ton_min * 0.15  # 15% of min = clear signal
    _has_significant_gap = _gap_to_min_tons > _gap_threshold
    _has_real_bridge = bool(allow_real_bridge)
    # Guarded family is profile-specific; check from config
    _has_guard_family = bool(getattr(cfg.model, "virtual_family_frontload_enabled", False)) if cfg and cfg.model else False
    _precheck_pass = (
        _has_underfilled
        or _has_significant_gap
        or _has_real_bridge
        or _has_guard_family
    )
    if not _precheck_pass:
        diag["cutter_blocks_skipped_by_precheck"] = len(underfilled_segments)
        diag["underfilled_reconstruction_not_entered_reason"] = "PRECHECK_FAILED_NO_GAIN_SIGNAL"
        diag["underfilled_reconstruction_seconds"] = round(perf_counter() - t0, 6)
        diag["reconstruction_seconds"] = diag["underfilled_reconstruction_seconds"]
        print(
            f"[APS][UNDERFILLED_RECON_PRECHECK] SKIP — no gain signal: "
            f"has_underfilled={_has_underfilled}, gap={_gap_to_min_tons:.0f}/{_gap_threshold:.0f}, "
            f"has_real_bridge={_has_real_bridge}, has_guard_family={_has_guard_family}"
        )
        return list(valid_segments), list(underfilled_segments), diag

    # =========================================================================
    # No-gain skip set: blocks that showed zero improvement twice → skip
    # =========================================================================
    # Note: no_gain_skip_set is initialized empty here. In the ALNS main loop,
    # it persists across round calls via a module-level variable.
    no_gain_skip_set: set[int] = set()

    # =========================================================================
    # Early stop: consecutive no-gain streak
    # =========================================================================
    no_gain_streak = 0
    early_stop_no_gain_limit = 5  # Stop after N consecutive blocks with no gain
    reconstruction_early_stopped = False

    if not underfilled_segments:
        census_dict = bridgeability_census.to_dict()
        suggestion = _suggest_next_phase_from_census(census_dict)
        diag["bridgeability_census"] = census_dict
        diag["bridgeability_census_items"] = list(census_dict.get("items", []))
        diag["bridgeability_route_suggestion"] = str(suggestion.get("suggestion", "CONTINUE_REAL_BRIDGE_ONLY"))
        diag["underfilled_reconstruction_not_entered_reason"] = "NO_UNDERFILLED_SEGMENTS"
        diag["repair_only_real_bridge_not_entered_reason"] = "NO_UNDERFILLED_SEGMENTS"
        diag["underfilled_reconstruction_seconds"] = round(perf_counter() - t0, 6)
        diag["reconstruction_seconds"] = diag["underfilled_reconstruction_seconds"]
        print("[APS][UNDERFILLED_RECON_SKIP] line=ALL, block_id=0, reason=NO_UNDERFILLED_SEGMENTS")
        print(
            "[APS][BRIDGEABILITY_CENSUS] total_blocks=0, has_endpoint_edge_count=0, "
            "template_graph_no_edge_count=0, rule_prefilter_all_fail_count=0, "
            "band_too_narrow_count=0, candidate_pool_empty_count=0, candidate_pool_nonempty_count=0, "
            "ton_rescue_entered_count=0, ton_rescue_success_count=0, improvement_recorded_count=0, "
            "improvement_applied_count=0, thickness_fail_dominant_count=0, width_fail_dominant_count=0, "
            "temp_fail_dominant_count=0, group_fail_dominant_count=0, multi_fail_dominant_count=0"
        )
        print(
            f"[APS][BRIDGEABILITY_ROUTE_SUGGESTION] suggestion={diag['bridgeability_route_suggestion']}, "
            f"reasons={list(suggestion.get('reasons', []))}"
        )
        print(
            f"[APS][UNDERFILLED_RECON_SUMMARY] attempts=0, success=0, blocks_tested=0, "
            f"valid_delta=0, underfilled_delta=0, salvaged_segments=0, salvaged_orders=0, "
            f"cutter_blocks_touched=0, cutter_blocks_improved=0, "
            f"cutter_blocks_skipped_by_precheck={diag.get('cutter_blocks_skipped_by_precheck', 0)}, "
            f"cutter_blocks_skipped_by_no_gain_set=0, cutter_no_gain_streak_max=0, "
            f"reconstruction_early_stopped=False, prefilter_reject_count=0, "
            f"endpoint_early_stop_count=0, local_band_retry_count=0, "
            f"improvement_recorded_count=0, improvement_applied_count=0, apply_reject_count=0, "
            f"bridgeability_route_suggestion={diag['bridgeability_route_suggestion']}, "
            f"virtual_pilot_skipped_due_to_disabled={bool(diag.get('virtual_pilot_skipped_due_to_disabled', True))}, "
            f"conservative_apply_attempt_count=0, conservative_apply_success_count=0, "
            f"conservative_apply_reject_count=0"
        )
        return list(valid_segments), [], diag

    direct_edges, real_edges = _build_reconstruction_edge_maps(transition_pack)
    tons_by_order = _infer_order_tons_from_underfilled(underfilled_segments, order_tons)
    ton_target = float(getattr(cfg.rule, "campaign_ton_target", (campaign_ton_min + campaign_ton_max) / 2.0) if cfg and cfg.rule else (campaign_ton_min + campaign_ton_max) / 2.0)
    max_real = int(getattr(cfg.model, "repair_only_bridge_max_per_segment", 1) if cfg and cfg.model else 1)
    bridge_penalty = float(getattr(cfg.model, "repair_only_bridge_cost_penalty", 100000.0) if cfg and cfg.model else 100000.0)
    left_band_k = int(getattr(cfg.model, "repair_bridge_left_band_k", 3) if cfg and cfg.model else 3)
    right_band_k = int(getattr(cfg.model, "repair_bridge_right_band_k", 3) if cfg and cfg.model else 3)
    max_band_pairs = int(getattr(cfg.model, "repair_bridge_band_max_pairs_per_split", 9) if cfg and cfg.model else 9)
    left_trim_max = int(getattr(cfg.model, "repair_bridge_left_trim_max", 2) if cfg and cfg.model else 2)
    right_trim_max = int(getattr(cfg.model, "repair_bridge_right_trim_max", 2) if cfg and cfg.model else 2)
    adjustment_limit = int(getattr(cfg.model, "repair_bridge_endpoint_adjustment_limit_per_split", 9) if cfg and cfg.model else 9)
    enable_left_trim = bool(getattr(cfg.model, "repair_bridge_adjustment_enable_left_trim", True) if cfg and cfg.model else True)
    enable_right_trim = bool(getattr(cfg.model, "repair_bridge_adjustment_enable_right_trim", True) if cfg and cfg.model else True)
    ton_rescue_max_neighbor_blocks = int(getattr(cfg.model, "repair_bridge_ton_rescue_max_neighbor_blocks", 2) if cfg and cfg.model else 2)
    ton_rescue_enable_backward = bool(getattr(cfg.model, "repair_bridge_ton_rescue_enable_backward", True) if cfg and cfg.model else True)
    ton_rescue_enable_forward = bool(getattr(cfg.model, "repair_bridge_ton_rescue_enable_forward", True) if cfg and cfg.model else True)
    ton_rescue_enable_bidirectional = bool(getattr(cfg.model, "repair_bridge_ton_rescue_enable_bidirectional", True) if cfg and cfg.model else True)
    ton_rescue_max_orders_per_window = int(getattr(cfg.model, "repair_bridge_ton_rescue_max_orders_per_window", 50) if cfg and cfg.model else 50)
    ton_rescue_max_failed_after_min = int(getattr(cfg.model, "repair_bridge_ton_rescue_max_failed_windows_after_min", 2) if cfg and cfg.model else 2)
    ton_rescue_max_left_expand = int(getattr(cfg.model, "repair_bridge_ton_rescue_max_left_expand", 12) if cfg and cfg.model else 12)
    ton_rescue_max_right_expand = int(getattr(cfg.model, "repair_bridge_ton_rescue_max_right_expand", 12) if cfg and cfg.model else 12)
    ton_rescue_max_extract_windows = int(getattr(cfg.model, "repair_bridge_ton_rescue_max_windows_per_candidate", 60) if cfg and cfg.model else 60)
    virtual_pilot_enabled = bool(getattr(cfg.model, "repair_only_virtual_bridge_pilot_enabled", False) if cfg and cfg.model else False)
    virtual_pilot_max_blocks = int(getattr(cfg.model, "virtual_bridge_pilot_max_blocks_per_run", 5) if cfg and cfg.model else 5)
    virtual_pilot_max_per_block = int(getattr(cfg.model, "virtual_bridge_pilot_max_per_block", 1) if cfg and cfg.model else 1)
    virtual_pilot_max_tons = float(getattr(cfg.model, "virtual_bridge_pilot_max_virtual_tons", 30.0) if cfg and cfg.model else 30.0)
    virtual_pilot_penalty = float(getattr(cfg.model, "virtual_bridge_pilot_penalty", 1000000.0) if cfg and cfg.model else 1000000.0)
    endpoint_class_cfg = getattr(cfg.model, "virtual_bridge_pilot_only_when_endpoint_class", None) if cfg and cfg.model else None
    fail_reason_cfg = getattr(cfg.model, "virtual_bridge_pilot_only_when_dominant_fail", None) if cfg and cfg.model else None
    virtual_pilot_endpoint_classes = set(endpoint_class_cfg or ["HAS_ENDPOINT_EDGE", "BAND_TOO_NARROW"])
    virtual_pilot_fail_reasons = set(fail_reason_cfg or ["THICKNESS_RULE_FAIL", "WIDTH_RULE_FAIL", "GROUP_SWITCH_FAIL", "MULTI_RULE_FAIL"])
    virtual_edges, virtual_tons_by_edge = _build_virtual_reconstruction_edges(transition_pack)
    diag["virtual_pilot_scheduler_budget"] = int(virtual_pilot_max_blocks)
    print(
        f"[APS][VIRTUAL_BRIDGE_PILOT_CONFIG] enabled={bool(virtual_pilot_enabled)}, "
        f"max_blocks_per_run={int(virtual_pilot_max_blocks)}, "
        f"max_per_block={int(virtual_pilot_max_per_block)}, "
        f"max_virtual_tons={float(virtual_pilot_max_tons):.1f}, "
        f"penalty={float(virtual_pilot_penalty):.1f}"
    )

    frozen_oids = Counter()
    for seg in valid_segments:
        frozen_oids.update([str(v) for v in seg.order_ids])
    under_oids = Counter()
    for seg in underfilled_segments:
        under_oids.update([str(v) for v in seg.order_ids])
    overlap = [oid for oid in under_oids if oid in frozen_oids]
    if overlap:
        diag["reconstruction_no_gain"] = True
        diag["frozen_underfilled_overlap_count"] = len(overlap)
        diag["frozen_underfilled_overlap_examples"] = overlap[:10]
        diag["underfilled_reconstruction_not_entered_reason"] = "FROZEN_UNDERFILLED_OVERLAP"
        diag["repair_only_real_bridge_not_entered_reason"] = "FROZEN_UNDERFILLED_OVERLAP"
        diag["underfilled_reconstruction_seconds"] = round(perf_counter() - t0, 6)
        diag["reconstruction_seconds"] = diag["underfilled_reconstruction_seconds"]
        print("[APS][UNDERFILLED_RECON_SKIP] line=ALL, block_id=0, reason=BLOCK_ORDER_MISMATCH")
        return list(valid_segments), list(underfilled_segments), diag

    new_valid = list(valid_segments)
    remaining_under: list[CampaignSegment] = []
    max_local_id = max([s.campaign_local_id for s in valid_segments + underfilled_segments] or [0])
    virtual_pilot_tried_blocks: set[tuple[str, int]] = set()
    virtual_pilot_seen_keys: set[tuple[str, str, str]] = set()
    virtual_pilot_selected_by_line: Counter = Counter()
    virtual_pilot_selected_by_bucket: Counter = Counter()
    virtual_pilot_selected_by_family: Counter = Counter()
    virtual_pilot_scheduler_selected_blocks: list[dict] = []
    virtual_pilot_scheduler_skipped_due_to_limit: list[dict] = []
    virtual_pilot_line_quota = max(2, int(virtual_pilot_max_blocks) // max(1, len(lines)))
    virtual_pilot_bucket_quota = max(3, int(virtual_pilot_max_blocks) // 3)
    virtual_pilot_family_quota = {
        "WIDTH_GROUP": min(5, int(virtual_pilot_max_blocks)),
        "THICKNESS": min(5, int(virtual_pilot_max_blocks)),
        "OTHER": max(0, int(virtual_pilot_max_blocks) - 10),
    }
    if int(virtual_pilot_max_blocks) > sum(virtual_pilot_family_quota.values()):
        virtual_pilot_family_quota["OTHER"] += int(virtual_pilot_max_blocks) - sum(virtual_pilot_family_quota.values())

    for line in sorted({s.line for s in underfilled_segments}):
        # ---- Early stop: consecutive no-gain streak limit ----
        if reconstruction_early_stopped:
            print(f"[APS][RECON_EARLY_STOP] line={line}, reason=NO_GAIN_STREAK_LIMIT({early_stop_no_gain_limit})")
            diag["reconstruction_stage_early_stopped"] = True
            break

        line_under = sorted([s for s in underfilled_segments if s.line == line], key=lambda s: s.campaign_local_id)
        line_pair_rows_by_key = _build_template_pair_rows_by_key(tpl_df, str(line))
        line_bridge_schema = _print_bridge_schema_samples(tpl_df, str(line))
        line_raw_real_rows = int(line_bridge_schema.get("raw_real_rows", 0) or 0)
        if bool(line_bridge_schema.get("field_name_mismatch", False)):
            diag["repair_bridge_field_name_mismatch_count"] += 1
        if not line_under:
            diag["underfilled_reconstruction_blocks_skipped"] += 1
            print(f"[APS][UNDERFILLED_RECON_SKIP] line={line}, block_id=0, reason=NO_UNDERFILLED_SEGMENTS")
            continue
        i = 0
        while i < len(line_under):
            # ---- No-gain skip set: block already tried with no gain twice ----
            block_uid = f"{line}:{i}"
            if block_uid in no_gain_skip_set:
                diag["cutter_blocks_skipped_by_no_gain_set"] = diag.get("cutter_blocks_skipped_by_no_gain_set", 0) + 1
                remaining_under.append(line_under[i])
                i += 1
                continue
            best = None
            best_size = 1
            block_id = i
            if len(line_under) - i < 2:
                diag["underfilled_reconstruction_blocks_skipped"] += 1
                print(f"[APS][UNDERFILLED_RECON_SKIP] line={line}, block_id={block_id}, reason=BLOCK_TOO_SMALL")
                remaining_under.append(line_under[i])
                i += 1
                continue
            # Block will be processed — count as touched
            diag["cutter_blocks_touched"] = diag.get("cutter_blocks_touched", 0) + 1
            for block_size in (4, 3, 2):
                if i + block_size > len(line_under):
                    continue
                block = line_under[i : i + block_size]
                block_orders = [oid for seg in block for oid in seg.order_ids]
                _print_order_context_lookup_sample(order_specs_by_id, [str(oid) for oid in block_orders[:8]], limit=4)
                block_tons = sum(float(tons_by_order.get(str(oid), 0.0) or 0.0) for oid in block_orders)
                block_real_rows = _bridge_rows_for_line(tpl_df, str(line))
                block_virtual_rows = _virtual_bridge_rows_for_line(tpl_df, str(line))
                real_bridge_lookup = _build_real_bridge_lookup(block_real_rows)
                virtual_bridge_lookup = _build_virtual_bridge_lookup(block_virtual_rows)
                single_point_keys = _block_single_point_boundary_keys(block)
                base_boundary_keys, base_band_logs = _block_boundary_band_keys(
                    block,
                    left_band_k=left_band_k,
                    right_band_k=right_band_k,
                    max_pairs_per_split=max_band_pairs,
                    order_context_by_id=order_specs_by_id,
                    block_id=int(block_id),
                    rule=cfg.rule if cfg and cfg.rule else None,
                )
                adjustment_boundary_keys: list[tuple[str, str, str]] = []
                adjustment_band_logs: list[dict] = []
                frontier_boundary_keys: list[tuple[str, str, str]] = []
                frontier_contexts: dict[tuple[str, str, str], list[dict]] = {}
                adjustment_generated_total = 0
                adjustment_pairs_tested_total = 0
                adjustment_hits_total = 0
                adjustment_only_hits_total = 0
                adjustment_best_cost: int | None = None
                endpoint_class = "UNKNOWN_ENDPOINT_MISS"
                dominant_fail_reason = "UNKNOWN"
                final_decision_for_census = "NO_DIRECT_OR_BRIDGE_CANDIDATE"
                ton_rescue_entered_for_census = False
                ton_rescue_success_for_census = False
                prefilter_rejected_for_census = False
                single_match_diag = _count_real_bridge_matches(real_bridge_lookup, set(single_point_keys))
                base_match_diag = _count_real_bridge_matches(real_bridge_lookup, set(base_boundary_keys))
                base_band_only_hit = int(base_match_diag.get("matched_rows", 0) or 0) > 0 and int(single_match_diag.get("matched_rows", 0) or 0) <= 0
                available_real_keys = list(real_bridge_lookup.keys())[:5]
                left_tail_ids = [str(seg.order_ids[-1]) for seg in block if seg.order_ids][:4]
                right_head_ids = [str(seg.order_ids[0]) for seg in block if seg.order_ids][:4]
                print(
                    f"[APS][UNDERFILLED_RECON_BLOCK] line={line}, block_id={block_id}, "
                    f"block_size={block_size}, orders={len(block_orders)}, tons={block_tons:.1f}, "
                    f"direct_only_enabled=True, real_bridge_enabled={bool(allow_real_bridge)}"
                )
                print(
                    f"[APS][REPAIR_BRIDGE_BOUNDARY_KEYS] line={line}, block_id={block_id}, "
                    f"split_count={len(base_boundary_keys)}, boundary_keys={base_boundary_keys[:8]}"
                )
                for band_item in base_band_logs:
                    split_id = int(band_item.get("split_id", 0) or 0)
                    band_pairs = list(band_item.get("band_pairs", []))
                    matched_pairs = [key for key in band_pairs if key in real_bridge_lookup]
                    matched_rows_count = sum(len(real_bridge_lookup.get(key, [])) for key in matched_pairs)
                    print(
                        f"[APS][REPAIR_BRIDGE_BAND] line={line}, block_id={block_id}, split_id={split_id}, "
                        f"left_band={band_item.get('left_band', [])}, right_band={band_item.get('right_band', [])}, "
                        f"band_pairs={band_pairs}"
                    )
                    print(
                        f"[APS][REPAIR_BRIDGE_BAND_MATCH] line={line}, block_id={block_id}, split_id={split_id}, "
                        f"matched_pairs={matched_pairs}, matched_rows={matched_rows_count}"
                    )
                    if matched_rows_count > 0:
                        single_key = single_point_keys[split_id - 1] if 0 <= split_id - 1 < len(single_point_keys) else None
                        single_hit = bool(single_key in real_bridge_lookup) if single_key else False
                        if not single_hit:
                            print(
                                f"[APS][REPAIR_BRIDGE_BAND_HIT] line={line}, block_id={block_id}, split_id={split_id}, "
                                f"single_point_miss=True, band_hit=True"
                            )
                endpoint_miss_result = _classify_endpoint_miss(
                    line=str(line),
                    block_id=int(block_id),
                    boundary_keys=base_boundary_keys,
                    real_bridge_lookup=real_bridge_lookup,
                    order_context_by_id=order_specs_by_id,
                    rule=cfg.rule if cfg and cfg.rule else None,
                )
                endpoint_class = str(endpoint_miss_result.get("class", "") or "")
                if endpoint_class == "TEMPLATE_GRAPH_NO_EDGE":
                    diag["repair_bridge_template_no_edge_count"] += 1
                elif endpoint_class == "RULE_PREFILTER_ALL_FAIL":
                    diag["repair_bridge_prefilter_all_fail_count"] += 1
                elif endpoint_class == "BAND_TOO_NARROW":
                    diag["repair_bridge_band_too_narrow_count"] += 1
                endpoint_early_stop = endpoint_class in {"TEMPLATE_GRAPH_NO_EDGE", "RULE_PREFILTER_ALL_FAIL"}
                if endpoint_early_stop:
                    diag["repair_bridge_endpoint_early_stop_count"] += 1
                    print(
                        f"[APS][REPAIR_BRIDGE_EARLY_STOP] line={line}, block_id={block_id}, "
                        f"endpoint_class={endpoint_class}, reason={endpoint_class}"
                    )
                elif endpoint_class == "BAND_TOO_NARROW":
                    diag["repair_bridge_local_band_retry_count"] += 1
                    print(
                        f"[APS][REPAIR_BRIDGE_LOCAL_BAND_RETRY] line={line}, block_id={block_id}, "
                        f"split_id=ALL, expand_side=right, extra=1"
                    )
                if not endpoint_early_stop:
                    for split_id, (left_seg, right_seg) in enumerate(zip(block, block[1:]), start=1):
                        left_orders = [str(v) for v in left_seg.order_ids]
                        right_orders = [str(v) for v in right_seg.order_ids]
                        adjustments = _generate_endpoint_adjustments(
                            left_orders,
                            right_orders,
                            left_trim_max=left_trim_max,
                            right_trim_max=right_trim_max,
                            adjustment_limit=adjustment_limit,
                            enable_left_trim=enable_left_trim,
                            enable_right_trim=enable_right_trim,
                            order_context_by_id=order_specs_by_id,
                            rule=cfg.rule if cfg and cfg.rule else None,
                        )
                        adjustment_generated_total += int(len(adjustments))
                        split_adjustment_hits = 0
                        split_adjustment_pairs = 0
                        split_best_adjust_cost: int | None = None
                        base_split_pairs = list(base_band_logs[split_id - 1].get("band_pairs", [])) if 0 <= split_id - 1 < len(base_band_logs) else []
                        base_split_hit = any(key in real_bridge_lookup for key in base_split_pairs)
                        for adjustment_id, adjustment in enumerate(adjustments, start=1):
                            adjusted_left_orders = [str(v) for v in adjustment.get("adjusted_left_orders", [])]
                            adjusted_right_orders = [str(v) for v in adjustment.get("adjusted_right_orders", [])]
                            band_pairs, adjusted_left_band, adjusted_right_band, adjusted_left_generation, adjusted_right_generation = _build_adjustment_band_pairs(
                                line=str(line),
                                adjusted_left_orders=adjusted_left_orders,
                                adjusted_right_orders=adjusted_right_orders,
                                left_band_k=left_band_k,
                                right_band_k=right_band_k,
                                max_pairs_per_split=max_band_pairs,
                                order_context_by_id=order_specs_by_id,
                                block_id=int(block_id),
                                split_id=int(split_id),
                                rule=cfg.rule if cfg and cfg.rule else None,
                            )
                            left_trim = int(adjustment.get("left_trim", 0) or 0)
                            right_trim = int(adjustment.get("right_trim", 0) or 0)
                            base_adjustment_cost = left_trim + right_trim
                            score_pair = (str(adjusted_left_orders[-1]), str(adjusted_right_orders[0])) if adjusted_left_orders and adjusted_right_orders else ("", "")
                            score_metrics = _quick_pair_rank_metrics(order_specs_by_id, score_pair[0], score_pair[1], rule=cfg.rule if cfg and cfg.rule else None) if score_pair[0] and score_pair[1] else {}
                            thickness_penalty = float(score_metrics.get("thickness_penalty", 0.0) or 0.0)
                            width_penalty = float(score_metrics.get("width_penalty", 0.0) or 0.0)
                            temp_penalty = float(score_metrics.get("temp_penalty", 0.0) or 0.0)
                            group_penalty = float(score_metrics.get("group_penalty", 0.0) or 0.0)
                            hard_penalty = float(score_metrics.get("hard_prefilter_fail_penalty", 0.0) or 0.0)
                            adjustment_cost = int(base_adjustment_cost + round(hard_penalty + thickness_penalty + width_penalty + temp_penalty + group_penalty))
                            print(
                                f"[APS][REPAIR_BRIDGE_ADJUST_SCORE] line={line}, block_id={block_id}, "
                                f"split_id={split_id}, adjustment_id={adjustment_id}, "
                                f"pair=({score_pair[0]},{score_pair[1]}), base_cost={base_adjustment_cost}, "
                                f"thickness_penalty={thickness_penalty:.3f}, width_penalty={width_penalty:.3f}, "
                                f"temp_penalty={temp_penalty:.3f}, group_penalty={group_penalty:.3f}, "
                                f"final_cost={adjustment_cost}"
                            )
                            matched_pairs = [key for key in band_pairs if key in real_bridge_lookup]
                            matched_rows_count = sum(len(real_bridge_lookup.get(key, [])) for key in matched_pairs)
                            adjustment_pairs_tested_total += int(len(band_pairs))
                            split_adjustment_pairs += int(len(band_pairs))
                            adjustment_boundary_keys.extend(band_pairs)
                            frontier_key: tuple[str, str, str] | None = None
                            if adjusted_left_orders and adjusted_right_orders:
                                frontier_key = (
                                    str(line),
                                    str(adjusted_left_orders[-1]),
                                    str(adjusted_right_orders[0]),
                                )
                                frontier_boundary_keys.append(frontier_key)
                                frontier_contexts.setdefault(frontier_key, []).append(
                                    {
                                        "split_id": split_id,
                                        "adjustment_id": adjustment_id,
                                        "left_trim": left_trim,
                                        "right_trim": right_trim,
                                        "adjusted_left_orders": list(adjusted_left_orders),
                                        "adjusted_right_orders": list(adjusted_right_orders),
                                        "actual_left_tail": str(adjusted_left_orders[-1]),
                                        "actual_right_head": str(adjusted_right_orders[0]),
                                    }
                                )
                            adjustment_band_logs.append(
                                {
                                    "split_id": split_id,
                                    "adjustment_id": adjustment_id,
                                    "left_trim": left_trim,
                                    "right_trim": right_trim,
                                    "base_cost": int(base_adjustment_cost),
                                    "thickness_penalty": float(thickness_penalty),
                                    "width_penalty": float(width_penalty),
                                    "temp_penalty": float(temp_penalty),
                                    "group_penalty": float(group_penalty),
                                    "final_cost": int(adjustment_cost),
                                    "left_band": adjusted_left_band,
                                    "right_band": adjusted_right_band,
                                    "left_generation_order": list(adjusted_left_generation),
                                    "right_generation_order": list(adjusted_right_generation),
                                    "band_pairs": band_pairs,
                                }
                            )
                            print(
                                f"[APS][REPAIR_BRIDGE_ADJUST] line={line}, block_id={block_id}, "
                                f"split_id={split_id}, adjustment_id={adjustment_id}, "
                                f"left_trim={left_trim}, right_trim={right_trim}"
                            )
                            print(
                                f"[APS][REPAIR_BRIDGE_ADJUST_BAND] line={line}, block_id={block_id}, "
                                f"split_id={split_id}, adjustment_id={adjustment_id}, "
                                f"left_band={adjusted_left_band}, right_band={adjusted_right_band}, "
                                f"band_pairs={band_pairs}"
                            )
                            print(
                                f"[APS][REPAIR_BRIDGE_ADJUST_MATCH] line={line}, block_id={block_id}, "
                                f"split_id={split_id}, adjustment_id={adjustment_id}, "
                                f"matched_pairs={matched_pairs}, matched_rows={matched_rows_count}"
                            )
                            if matched_rows_count > 0:
                                adjustment_hits_total += int(matched_rows_count)
                                split_adjustment_hits += int(matched_rows_count)
                                adjustment_best_cost = adjustment_cost if adjustment_best_cost is None else min(adjustment_best_cost, adjustment_cost)
                                split_best_adjust_cost = adjustment_cost if split_best_adjust_cost is None else min(split_best_adjust_cost, adjustment_cost)
                                if not base_split_hit and adjustment_cost > 0:
                                    adjustment_only_hits_total += int(matched_rows_count)
                                    print(
                                        f"[APS][REPAIR_BRIDGE_ADJUST_HIT] line={line}, block_id={block_id}, "
                                        f"split_id={split_id}, adjustment_id={adjustment_id}, "
                                        f"left_trim={left_trim}, right_trim={right_trim}, "
                                        f"base_hit=False, adjusted_hit=True"
                                    )
                        print(
                            f"[APS][REPAIR_BRIDGE_ADJUST_SUMMARY] line={line}, block_id={block_id}, "
                            f"split_id={split_id}, adjustments_generated={len(adjustments)}, "
                            f"pairs_tested={split_adjustment_pairs}, adjustment_hits={split_adjustment_hits}, "
                            f"best_adjustment_cost={-1 if split_best_adjust_cost is None else int(split_best_adjust_cost)}"
                        )
                boundary_keys = list(dict.fromkeys(base_boundary_keys + adjustment_boundary_keys))
                frontier_keys = list(dict.fromkeys(frontier_boundary_keys))
                band_logs = base_band_logs + adjustment_band_logs
                real_edge_distance = _boundary_band_distance_map(band_logs)
                real_edge_adjustment_cost = _endpoint_adjustment_cost_map(band_logs)
                bridge_match_diag = _count_real_bridge_matches(real_bridge_lookup, set(boundary_keys))
                frontier_match_diag = _count_real_bridge_matches(real_bridge_lookup, set(frontier_keys))
                adjustment_match_diag = _count_real_bridge_matches(real_bridge_lookup, set(adjustment_boundary_keys))
                band_only_hit = int(base_match_diag.get("matched_rows", 0) or 0) > 0 and int(single_match_diag.get("matched_rows", 0) or 0) <= 0
                matched_bridge_keys = _rank_pair_keys_by_rules(
                    [key for key in boundary_keys if key in real_bridge_lookup],
                    order_specs_by_id,
                    adjustment_cost_by_pair=real_edge_adjustment_cost,
                    rule=cfg.rule if cfg and cfg.rule else None,
                )
                frontier_matched_keys = _rank_pair_keys_by_rules(
                    [key for key in frontier_keys if key in real_bridge_lookup],
                    order_specs_by_id,
                    adjustment_cost_by_pair=real_edge_adjustment_cost,
                    rule=cfg.rule if cfg and cfg.rule else None,
                )
                virtual_frontier_keys = _rank_pair_keys_by_rules(
                    [key for key in frontier_keys if key in virtual_bridge_lookup],
                    order_specs_by_id,
                    adjustment_cost_by_pair=real_edge_adjustment_cost,
                    rule=cfg.rule if cfg and cfg.rule else None,
                )
                bridge_candidate_audit = _repair_bridge_candidate_audit(
                    line=str(line),
                    block_id=int(block_id),
                    block=block,
                    real_bridge_lookup=real_bridge_lookup,
                    matched_keys=matched_bridge_keys,
                    frontier_contexts=frontier_contexts,
                    direct_edges=direct_edges,
                    real_edges=real_edges,
                    pair_rows_by_key=line_pair_rows_by_key,
                    order_specs_by_id=order_specs_by_id,
                    rule=cfg.rule if cfg and cfg.rule else None,
                    real_edge_adjustment_cost=real_edge_adjustment_cost,
                    endpoint_class=endpoint_class,
                    order_tons=tons_by_order,
                    campaign_ton_min=campaign_ton_min,
                    campaign_ton_max=campaign_ton_max,
                ) if matched_bridge_keys else {
                    "matched": 0,
                    "rejected_pair_invalid": 0,
                    "rejected_ton_invalid": 0,
                    "rejected_score_worse": 0,
                    "accepted": 0,
                    "exact_invalid_pair_count": 0,
                    "frontier_mismatch": 0,
                    "pair_invalid_width": 0,
                    "pair_invalid_thickness": 0,
                    "pair_invalid_temp": 0,
                    "pair_invalid_group": 0,
                    "pair_invalid_context": 0,
                    "pair_invalid_unknown": 0,
                    "prefilter_fail_thickness": 0,
                    "rank_pass_thickness": 0,
                    "rank_fail_thickness": 0,
                    "prefilter_reject_count": 0,
                    "pilot_candidates": [],
                    "reject_buckets": Counter(),
                }
                print(
                    f"[APS][REPAIR_BRIDGE_LOOKUP] line={line}, "
                    f"total_real_rows={len(block_real_rows)}, unique_keys={len(real_bridge_lookup)}"
                )
                print(
                    f"[APS][REPAIR_BRIDGE_KEYS] line={line}, block_id={block_id}, "
                    f"left_tail_ids={left_tail_ids}, right_head_ids={right_head_ids}, "
                    f"lookup_key_mode=(line,from_order_id,to_order_id), expected_key_sample={boundary_keys[:5]}"
                )
                print(
                    f"[APS][REPAIR_BRIDGE_RAW] line={line}, block_id={block_id}, "
                    f"raw_rows={line_raw_real_rows}"
                )
                print(
                    f"[APS][REPAIR_BRIDGE_MATCH] line={line}, block_id={block_id}, "
                    f"matched_rows={int(bridge_match_diag.get('matched_rows', 0))}, "
                    f"frontier_matched_rows={int(frontier_match_diag.get('matched_rows', 0))}"
                )
                diag["repair_bridge_raw_rows_total"] += int(line_raw_real_rows)
                diag["repair_bridge_matched_rows_total"] += int(bridge_match_diag.get("matched_rows", 0))
                diag["repair_bridge_band_pairs_tested"] += int(len(base_boundary_keys))
                diag["repair_bridge_band_hits"] += int(base_match_diag.get("matched_rows", 0) or 0)
                diag["repair_bridge_single_point_hits"] += int(single_match_diag.get("matched_rows", 0) or 0)
                diag["repair_bridge_adjustments_generated"] += int(adjustment_generated_total)
                diag["repair_bridge_adjustment_pairs_tested"] += int(adjustment_pairs_tested_total)
                diag["repair_bridge_adjustment_hits"] += int(adjustment_hits_total)
                diag["repair_bridge_adjustment_only_hits"] += int(adjustment_only_hits_total)
                diag["repair_bridge_candidates_matched"] += int(bridge_candidate_audit.get("matched", 0) or 0)
                diag["repair_bridge_candidates_rejected_pair_invalid"] += int(bridge_candidate_audit.get("rejected_pair_invalid", 0) or 0)
                diag["repair_bridge_candidates_rejected_ton_invalid"] += int(bridge_candidate_audit.get("rejected_ton_invalid", 0) or 0)
                diag["repair_bridge_candidates_rejected_score_worse"] += int(bridge_candidate_audit.get("rejected_score_worse", 0) or 0)
                diag["repair_bridge_candidates_accepted"] += int(bridge_candidate_audit.get("accepted", 0) or 0)
                diag["repair_bridge_exact_invalid_pair_count"] += int(bridge_candidate_audit.get("exact_invalid_pair_count", 0) or 0)
                diag["repair_bridge_frontier_mismatch_count"] += int(bridge_candidate_audit.get("frontier_mismatch", 0) or 0)
                diag["repair_bridge_pair_invalid_width"] += int(bridge_candidate_audit.get("pair_invalid_width", 0) or 0)
                diag["repair_bridge_pair_invalid_thickness"] += int(bridge_candidate_audit.get("pair_invalid_thickness", 0) or 0)
                diag["repair_bridge_pair_invalid_temp"] += int(bridge_candidate_audit.get("pair_invalid_temp", 0) or 0)
                diag["repair_bridge_pair_invalid_group"] += int(bridge_candidate_audit.get("pair_invalid_group", 0) or 0)
                diag["repair_bridge_pair_invalid_context"] += int(bridge_candidate_audit.get("pair_invalid_context", 0) or 0)
                diag["repair_bridge_pair_invalid_unknown"] += int(bridge_candidate_audit.get("pair_invalid_unknown", 0) or 0)
                diag["repair_bridge_pair_fail_thickness"] += int(bridge_candidate_audit.get("pair_invalid_thickness", 0) or 0)
                diag["repair_bridge_prefilter_fail_thickness"] += int(bridge_candidate_audit.get("prefilter_fail_thickness", 0) or 0)
                diag["repair_bridge_rank_pass_thickness"] += int(bridge_candidate_audit.get("rank_pass_thickness", 0) or 0)
                diag["repair_bridge_rank_fail_thickness"] += int(bridge_candidate_audit.get("rank_fail_thickness", 0) or 0)
                diag["repair_bridge_prefilter_reject_count"] += int(bridge_candidate_audit.get("prefilter_reject_count", 0) or 0)
                diag["repair_bridge_filtered_ton_below_min_current_block"] += int(bridge_candidate_audit.get("filtered_ton_below_min_current_block", 0) or 0)
                if adjustment_best_cost is not None:
                    old_adjust_cost = int(diag.get("repair_bridge_best_adjustment_cost", -1) or -1)
                    diag["repair_bridge_best_adjustment_cost"] = int(adjustment_best_cost) if old_adjust_cost < 0 else min(old_adjust_cost, int(adjustment_best_cost))
                if base_band_only_hit:
                    diag["repair_bridge_band_only_hits"] += int(base_match_diag.get("matched_rows", 0) or 0)
                if int(base_match_diag.get("matched_rows", 0) or 0) > 0:
                    matched_distances = [
                        real_edge_distance.get((str(key[1]), str(key[2])), 999)
                        for key in base_boundary_keys
                        if key in real_bridge_lookup
                    ]
                    if matched_distances:
                        best_distance = int(min(matched_distances))
                        old_distance = int(diag.get("repair_bridge_band_best_distance", -1) or -1)
                        diag["repair_bridge_band_best_distance"] = best_distance if old_distance < 0 else min(old_distance, best_distance)
                diag["repair_only_real_bridge_filtered_line_mismatch"] += int(bridge_match_diag.get("line_mismatch", 0))
                diag["repair_only_real_bridge_filtered_block_membership_mismatch"] += int(bridge_match_diag.get("block_membership_mismatch", 0))
                diag["repair_only_real_bridge_filtered_bridge_path_payload_empty"] += int(bridge_match_diag.get("payload_empty", 0))
                if line_raw_real_rows > 0 and int(bridge_match_diag.get("matched_rows", 0)) <= 0:
                    diag["repair_bridge_endpoint_key_mismatch_count"] += 1
                    print(
                        f"[APS][REPAIR_BRIDGE_MATCH_FAIL] line={line}, block_id={block_id}, "
                        f"reason=ENDPOINT_KEY_MISMATCH"
                    )
                    print(
                        f"[APS][REPAIR_BRIDGE_ENDPOINT_FAIL] line={line}, block_id={block_id}, "
                        f"boundary_keys={boundary_keys[:8]}, sample_available_real_keys={available_real_keys}"
                    )
                print(
                    f"[APS][REPAIR_BRIDGE_BAND_SUMMARY] line={line}, block_id={block_id}, "
                    f"band_pairs_tested={len(base_boundary_keys)}, "
                    f"band_hits={int(base_match_diag.get('matched_rows', 0) or 0)}, "
                    f"single_point_hits={int(single_match_diag.get('matched_rows', 0) or 0)}, "
                    f"band_only_hits={int(base_match_diag.get('matched_rows', 0) or 0) if band_only_hit else 0}"
                )
                print(
                    f"[APS][REPAIR_BRIDGE_ADJUST_SUMMARY] line={line}, block_id={block_id}, "
                    f"adjustments_generated={adjustment_generated_total}, "
                    f"pairs_tested={adjustment_pairs_tested_total}, "
                    f"adjustment_hits={adjustment_hits_total}, "
                    f"adjustment_only_hits={adjustment_only_hits_total}"
                )
                if int(bridge_match_diag.get("payload_empty", 0) or 0) > 0:
                    for key in boundary_keys:
                        for row in real_bridge_lookup.get(key, [])[:2]:
                            if _bridge_payload_empty(row):
                                src, _ = _extract_bridge_endpoint(row, "from")
                                dst, _ = _extract_bridge_endpoint(row, "to")
                                print(
                                    f"[APS][REPAIR_BRIDGE_PAYLOAD_NOTICE] line={line}, block_id={block_id}, "
                                    f"from={src}, to={dst}, payload_empty=True, edge_kept=True"
                                )
                if line_raw_real_rows <= 0 and int(diag.get("repair_bridge_pack_real_rows_total", 0) or 0) > 0:
                    diag["repair_bridge_inconsistency_count"] += 1
                    print(
                        f"[APS][REPAIR_BRIDGE_INCONSISTENCY] line={line}, "
                        f"template_real_bridge_count={int(diag.get('repair_bridge_pack_real_rows_total', 0) or 0)}, "
                        f"reconstruction_raw_rows=0"
                    )
                diag["underfilled_reconstruction_attempts"] += 1
                diag["underfilled_reconstruction_blocks_tested"] += 1
                direct_valid, direct_under, direct_diag = _solve_underfilled_reconstruction_block(
                    block,
                    direct_edges=direct_edges,
                    real_edges=real_edges,
                    order_tons=tons_by_order,
                    campaign_ton_min=campaign_ton_min,
                    campaign_ton_max=campaign_ton_max,
                    campaign_ton_target=ton_target,
                    allow_real_bridge=False,
                    allow_virtual_bridge=False,
                    max_real_bridge_per_segment=0,
                    bridge_cost_penalty=bridge_penalty,
                    next_local_id=max_local_id,
                )
                if direct_valid:
                    best = (direct_valid, direct_under, direct_diag, "DIRECT", block)
                    best_size = block_size
                    _record_bridgeability_census_item(
                        bridgeability_census,
                        {
                            "line": str(line),
                            "block_id": int(block_id),
                            "block_size": int(block_size),
                            "orders": int(len(block_orders)),
                            "tons": float(block_tons),
                            "endpoint_class": str(endpoint_class),
                            "dominant_fail_reason": "DIRECT_ONLY_FEASIBLE",
                            "candidate_count": int(bridge_match_diag.get("matched_rows", 0) or 0),
                            "frontier_candidate_count": int(frontier_match_diag.get("matched_rows", 0) or 0),
                            "prefilter_rejected": False,
                            "ton_rescue_entered": False,
                            "ton_rescue_success": False,
                            "final_decision": "DIRECT_IMPROVEMENT_FOUND",
                        },
                    )
                    break
                if allow_real_bridge:
                    real_t0 = perf_counter()
                    diag["repair_only_real_bridge_attempts"] += 1
                    print(
                        f"[APS][REPAIR_BRIDGE_ENTER] line={line}, block_id={block_id}, "
                        f"repair_only_real_bridge_enabled={bool(allow_real_bridge)}, "
                        f"repair_only_virtual_bridge_enabled={bool(allow_virtual_bridge)}"
                    )
                    real_valid, real_under, real_diag = _solve_underfilled_reconstruction_block(
                        block,
                        direct_edges=direct_edges,
                        real_edges={(str(k[1]), str(k[2])) for k in frontier_matched_keys},
                        order_tons=tons_by_order,
                        campaign_ton_min=campaign_ton_min,
                        campaign_ton_max=campaign_ton_max,
                        campaign_ton_target=ton_target,
                        allow_real_bridge=True,
                        allow_virtual_bridge=allow_virtual_bridge,
                        max_real_bridge_per_segment=max_real,
                        bridge_cost_penalty=bridge_penalty,
                        next_local_id=max_local_id,
                        real_edge_distance=real_edge_distance,
                        real_edge_adjustment_cost=real_edge_adjustment_cost,
                    )
                    diag["repair_bridge_real_seconds"] += perf_counter() - real_t0
                    diag["repair_only_real_bridge_seconds"] = diag["repair_bridge_real_seconds"]
                    real_diag_report = dict(real_diag)
                    if int(bridge_match_diag.get("matched_rows", 0) or 0) > 0:
                        real_diag_report["repair_only_real_bridge_filtered_pair_invalid"] = max(
                            int(real_diag_report.get("repair_only_real_bridge_filtered_pair_invalid", 0) or 0),
                            int(bridge_candidate_audit.get("rejected_pair_invalid", 0) or 0),
                        )
                        real_diag_report["repair_only_real_bridge_filtered_ton_invalid"] = max(
                            int(real_diag_report.get("repair_only_real_bridge_filtered_ton_invalid", 0) or 0),
                            int(bridge_candidate_audit.get("rejected_ton_invalid", 0) or 0),
                        )
                    for key in _bridge_filter_diag_template():
                        diag[key] += int(real_diag_report.get(key, 0) or 0)
                    if bool(direct_diag.get("direct_only_feasible", False)):
                        diag["repair_only_real_bridge_filtered_direct_feasible"] += 1
                    print(
                        f"[APS][REPAIR_BRIDGE_POOL] line={line}, block_id={block_id}, "
                        f"total_candidates={int(bridge_match_diag.get('matched_rows', 0) or 0)}, "
                        f"real_candidates={int(frontier_match_diag.get('matched_rows', 0) or 0)}, "
                        f"virtual_candidates=0, "
                        f"kept_after_basic_filter={int(real_diag_report.get('repair_only_real_bridge_candidates_kept', 0) or 0)}"
                    )
                    diag["repair_bridge_kept_rows_total"] += int(real_diag_report.get("repair_only_real_bridge_candidates_kept", 0) or 0)
                    if int(bridge_match_diag.get("matched_rows", 0) or 0) > 0 and int(real_diag_report.get("repair_only_real_bridge_candidates_kept", 0) or 0) <= 0:
                        print(f"[APS][REPAIR_BRIDGE_FILTER_FAIL] line={line}, block_id={block_id}, reason=ALL_FILTERED")
                    print(
                        f"[APS][REPAIR_BRIDGE_FILTER_SUMMARY] line={line}, block_id={block_id}, "
                        f"attempts=1, raw={line_raw_real_rows}, "
                        f"matched={int(bridge_match_diag.get('matched_rows', 0) or 0)}, "
                        f"frontier_matched={int(frontier_match_diag.get('matched_rows', 0) or 0)}, "
                        f"kept={int(real_diag_report.get('repair_only_real_bridge_candidates_kept', 0) or 0)}, "
                        f"filtered_direct_feasible={int(real_diag_report.get('repair_only_real_bridge_filtered_direct_feasible', 0) or 0)}, "
                        f"filtered_pair_invalid={int(real_diag_report.get('repair_only_real_bridge_filtered_pair_invalid', 0) or 0)}, "
                        f"filtered_ton_invalid={int(real_diag_report.get('repair_only_real_bridge_filtered_ton_invalid', 0) or 0)}, "
                        f"filtered_score_worse={int(real_diag_report.get('repair_only_real_bridge_filtered_score_worse', 0) or 0)}, "
                        f"filtered_line_mismatch={int(bridge_match_diag.get('line_mismatch', 0) or 0)}, "
                        f"filtered_block_membership_mismatch={int(bridge_match_diag.get('block_membership_mismatch', 0) or 0)}, "
                        f"filtered_bridge_path_payload_empty={int(bridge_match_diag.get('payload_empty', 0) or 0)}, "
                        f"endpoint_missing={int(bridge_match_diag.get('endpoint_missing', 0) or 0)}"
                    )
                    if real_valid:
                        diag["repair_bridge_candidates_accepted"] += max(1, int(real_diag_report.get("repair_only_real_bridge_candidates_kept", 0) or 0))
                        best = (real_valid, real_under, real_diag, "REAL", block)
                        best_size = block_size
                        _record_bridgeability_census_item(
                            bridgeability_census,
                            {
                                "line": str(line),
                                "block_id": int(block_id),
                                "block_size": int(block_size),
                                "orders": int(len(block_orders)),
                                "tons": float(block_tons),
                                "endpoint_class": str(endpoint_class),
                                "dominant_fail_reason": "REAL_BRIDGE_FEASIBLE",
                                "candidate_count": int(bridge_match_diag.get("matched_rows", 0) or 0),
                                "frontier_candidate_count": int(frontier_match_diag.get("matched_rows", 0) or 0),
                                "prefilter_rejected": bool(int(bridge_candidate_audit.get("prefilter_reject_count", 0) or 0) > 0),
                                "ton_rescue_entered": False,
                                "ton_rescue_success": False,
                                "final_decision": "REAL_IMPROVEMENT_FOUND",
                            },
                        )
                        break
                    ton_rescue_candidates = list(bridge_candidate_audit.get("ton_rescue_candidates", []) or [])
                    rescue_result = None
                    rescue_reason_counts: Counter = Counter()
                    if ton_rescue_candidates and int(frontier_match_diag.get("matched_rows", 0) or 0) > 0:
                        for rescue_candidate in ton_rescue_candidates:
                            candidate_key = rescue_candidate.get("key")
                            if not candidate_key:
                                continue
                            rescue_edge = (str(candidate_key[1]), str(candidate_key[2]))
                            diag["repair_bridge_ton_rescue_attempts"] += 1
                            max_back = min(int(ton_rescue_max_neighbor_blocks), i)
                            max_fwd = min(int(ton_rescue_max_neighbor_blocks), max(0, len(line_under) - (i + block_size)))
                            theoretical_start = max(0, i - max_back)
                            theoretical_end = min(len(line_under), i + block_size + max_fwd)
                            theoretical_orders = [
                                str(oid)
                                for seg in line_under[theoretical_start:theoretical_end]
                                for oid in seg.order_ids
                            ]
                            theoretical_max_tons = sum(float(tons_by_order.get(str(oid), 0.0) or 0.0) for oid in theoretical_orders)
                            if theoretical_max_tons < float(campaign_ton_min) - 1e-6:
                                diag["repair_bridge_filtered_ton_rescue_impossible"] += 1
                                rescue_reason_counts["TON_RESCUE_IMPOSSIBLE"] += 1
                                print(
                                    f"[APS][REPAIR_BRIDGE_TON_RESCUE_IMPOSSIBLE] "
                                    f"candidate_id={rescue_candidate.get('candidate_id')}, "
                                    f"theoretical_max_tons={theoretical_max_tons:.1f}, "
                                    f"campaign_ton_min={float(campaign_ton_min):.1f}"
                                )
                                continue
                            rescue_windows = _generate_ton_rescue_windows(
                                line_under=line_under,
                                current_block_idx=i,
                                current_block_size=block_size,
                                order_tons=tons_by_order,
                                campaign_ton_min=campaign_ton_min,
                                max_neighbor_blocks=ton_rescue_max_neighbor_blocks,
                                max_orders_per_window=ton_rescue_max_orders_per_window,
                                enable_backward=ton_rescue_enable_backward,
                                enable_forward=ton_rescue_enable_forward,
                                enable_bidirectional=ton_rescue_enable_bidirectional,
                            )
                            failed_after_min = 0
                            last_tons: float | None = None
                            current_block_segment_ids = [int(seg.campaign_local_id) for seg in block]
                            current_block_segment_count = len(block)
                            for window_id, window in enumerate(rescue_windows, start=1):
                                combined_tons = float(window.get("tons", 0.0) or 0.0)
                                if last_tons is not None and abs(combined_tons - last_tons) <= 1e-6:
                                    continue
                                last_tons = combined_tons
                                expansion_block = list(window.get("block", []))
                                diag["repair_bridge_ton_rescue_windows_tested"] += 1
                                diag["repair_bridge_ton_rescue_extract_attempts"] += 1
                                result = _try_bridge_ton_rescue_recut(
                                    candidate=rescue_candidate,
                                    line=str(line),
                                    block_id=int(block_id),
                                    window_id=int(window_id),
                                    direction=str(window.get("direction", "")),
                                    expansion_block=expansion_block,
                                    current_block_segment_count=current_block_segment_count,
                                    current_block_segment_ids=current_block_segment_ids,
                                    direct_edges=direct_edges,
                                    real_edge=rescue_edge,
                                    order_tons=tons_by_order,
                                    campaign_ton_min=campaign_ton_min,
                                    campaign_ton_max=campaign_ton_max,
                                    campaign_ton_target=ton_target,
                                    max_real_bridge_per_segment=max_real,
                                    bridge_cost_penalty=bridge_penalty,
                                    next_local_id=max_local_id,
                                    ton_rescue_max_left_expand=ton_rescue_max_left_expand,
                                    ton_rescue_max_right_expand=ton_rescue_max_right_expand,
                                    ton_rescue_max_windows_per_candidate=ton_rescue_max_extract_windows,
                                    pair_rows_by_key=line_pair_rows_by_key,
                                    order_specs_by_id=order_specs_by_id,
                                    rule=cfg.rule if cfg and cfg.rule else None,
                                    real_edge_distance=real_edge_distance,
                                    real_edge_adjustment_cost=real_edge_adjustment_cost,
                                )
                                result_diag = dict(result.get("diag", {}) or {})
                                fail_counts = dict(result_diag.get("fail_counts_by_reason", {}) or {})
                                diag["repair_bridge_ton_rescue_pair_fail_width"] += int(fail_counts.get("WIDTH_RULE_FAIL", 0) or 0)
                                diag["repair_bridge_ton_rescue_pair_fail_thickness"] += int(fail_counts.get("THICKNESS_RULE_FAIL", 0) or 0)
                                diag["repair_bridge_ton_rescue_pair_fail_temp"] += int(fail_counts.get("TEMP_OVERLAP_FAIL", 0) or 0)
                                diag["repair_bridge_ton_rescue_pair_fail_group"] += int(fail_counts.get("GROUP_SWITCH_FAIL", 0) or 0)
                                diag["repair_bridge_ton_rescue_pair_fail_context"] += int(fail_counts.get("ORDER_CONTEXT_MISSING", 0) or 0)
                                diag["repair_bridge_ton_rescue_pair_fail_template"] += int(fail_counts.get("TEMPLATE_KEY_MISSING", 0) or 0)
                                diag["repair_bridge_ton_rescue_pair_fail_multi"] += int(fail_counts.get("MULTI_RULE_FAIL", 0) or 0)
                                diag["repair_bridge_ton_rescue_pair_fail_unknown"] += int(fail_counts.get("UNKNOWN_PAIR_INVALID", 0) or 0)
                                if result.get("success"):
                                    rescue_result = result
                                    best = (
                                        list(result.get("valid", [])),
                                        list(result.get("underfilled", [])),
                                        dict(result.get("diag", {})),
                                        "REAL",
                                        expansion_block,
                                    )
                                    best_size = int(window.get("end", i + block_size)) - i
                                    diag["repair_bridge_ton_rescue_success"] += 1
                                    diag["repair_bridge_ton_rescue_valid_delta"] += int(result.get("valid_delta", 0) or 0)
                                    diag["repair_bridge_ton_rescue_underfilled_delta"] += int(result.get("underfilled_delta", 0) or 0)
                                    diag["repair_bridge_ton_rescue_scheduled_orders_delta"] += int(result.get("scheduled_orders_delta", 0) or 0)
                                    if bool(result_diag.get("extract_success", False)):
                                        diag["repair_bridge_ton_rescue_extract_success"] += 1
                                    if bool(result_diag.get("partial_success", False)):
                                        diag["repair_bridge_ton_rescue_partial_success"] += 1
                                    if bool(result_diag.get("full_success", False)):
                                        diag["repair_bridge_ton_rescue_full_success"] += 1
                                    diag["repair_bridge_ton_rescue_residual_underfilled_count"] += int(result_diag.get("residual_underfilled_count", 0) or 0)
                                    diag["repair_bridge_candidates_accepted"] += 1
                                    break
                                reject_reason = str(result.get("reject_reason", "") or "")
                                if reject_reason:
                                    rescue_reason_counts[reject_reason] += 1
                                if reject_reason == "TON_BELOW_MIN_EVEN_AFTER_NEIGHBOR_EXPANSION":
                                    diag["repair_bridge_filtered_ton_below_min_even_after_neighbor_expansion"] += 1
                                elif reject_reason == "TON_SPLIT_NOT_FOUND":
                                    diag["repair_bridge_filtered_ton_split_not_found"] += 1
                                    if bool(window.get("reaches_min", False)):
                                        failed_after_min += 1
                                elif reject_reason == "TON_RESCUE_NO_GAIN":
                                    diag["repair_bridge_filtered_ton_rescue_no_gain"] += 1
                                    if bool(window.get("reaches_min", False)):
                                        failed_after_min += 1
                                elif reject_reason == "TON_ABOVE_MAX_AFTER_EXPANSION":
                                    diag["repair_bridge_filtered_ton_above_max_after_expansion"] += 1
                                if failed_after_min >= int(ton_rescue_max_failed_after_min):
                                    break
                            if rescue_result is not None:
                                break
                        if rescue_result is not None:
                            _record_bridgeability_census_item(
                                bridgeability_census,
                                {
                                    "line": str(line),
                                    "block_id": int(block_id),
                                    "block_size": int(block_size),
                                    "orders": int(len(block_orders)),
                                    "tons": float(block_tons),
                                    "endpoint_class": str(endpoint_class),
                                    "dominant_fail_reason": "TON_RESCUE_SUCCESS",
                                    "candidate_count": int(bridge_match_diag.get("matched_rows", 0) or 0),
                                    "frontier_candidate_count": int(frontier_match_diag.get("matched_rows", 0) or 0),
                                    "prefilter_rejected": bool(int(bridge_candidate_audit.get("prefilter_reject_count", 0) or 0) > 0),
                                    "ton_rescue_entered": True,
                                    "ton_rescue_success": True,
                                    "final_decision": "TON_RESCUE_IMPROVEMENT_FOUND",
                                },
                            )
                            break
                    ton_subtotal = (
                        int(bridge_candidate_audit.get("filtered_ton_below_min_current_block", 0) or 0)
                        + int(rescue_reason_counts.get("TON_RESCUE_IMPOSSIBLE", 0) or 0)
                        + int(rescue_reason_counts.get("TON_BELOW_MIN_EVEN_AFTER_NEIGHBOR_EXPANSION", 0) or 0)
                        + int(rescue_reason_counts.get("TON_SPLIT_NOT_FOUND", 0) or 0)
                        + int(rescue_reason_counts.get("TON_RESCUE_NO_GAIN", 0) or 0)
                        + int(rescue_reason_counts.get("TON_ABOVE_MAX_AFTER_EXPANSION", 0) or 0)
                    )
                    filtered_ton_invalid_effective = int(real_diag_report.get("repair_only_real_bridge_filtered_ton_invalid", 0) or 0)
                    if ton_subtotal <= 0:
                        filtered_ton_invalid_effective = 0
                    reason_buckets = {
                        "ORDER_CONTEXT_MISSING": int(bridge_candidate_audit.get("pair_invalid_context", 0) or 0),
                        "TEMPLATE_PAIR_INVALID": int(real_diag_report.get("repair_only_real_bridge_filtered_pair_invalid", 0) or 0),
                        "TON_BELOW_MIN_CURRENT_BLOCK": int(bridge_candidate_audit.get("filtered_ton_below_min_current_block", 0) or 0),
                        "TON_RESCUE_IMPOSSIBLE": int(rescue_reason_counts.get("TON_RESCUE_IMPOSSIBLE", 0) or 0),
                        "TON_BELOW_MIN_EVEN_AFTER_NEIGHBOR_EXPANSION": int(rescue_reason_counts.get("TON_BELOW_MIN_EVEN_AFTER_NEIGHBOR_EXPANSION", 0) or 0),
                        "TON_SPLIT_NOT_FOUND": int(rescue_reason_counts.get("TON_SPLIT_NOT_FOUND", 0) or 0),
                        "TON_RESCUE_NO_GAIN": int(rescue_reason_counts.get("TON_RESCUE_NO_GAIN", 0) or 0),
                        "TON_WINDOW_INVALID": int(filtered_ton_invalid_effective),
                        "SCORE_WORSE_THAN_DIRECT_ONLY": int(real_diag_report.get("repair_only_real_bridge_filtered_score_worse", 0) or 0),
                        "BRIDGE_LIMIT_EXCEEDED": int(real_diag_report.get("repair_only_real_bridge_filtered_bridge_limit_exceeded", 0) or 0),
                        "BLOCK_MEMBERSHIP_MISMATCH": int(bridge_match_diag.get("block_membership_mismatch", 0) or 0),
                        "LINE_MISMATCH": int(bridge_match_diag.get("line_mismatch", 0) or 0),
                    }
                    has_reject_bucket = any(v > 0 for v in reason_buckets.values())
                    dominant_bucket = max(reason_buckets.items(), key=lambda kv: kv[1])[0] if has_reject_bucket else "NO_DIRECT_OR_BRIDGE_CANDIDATE"
                    reason = dominant_bucket
                    if not has_reject_bucket and line_raw_real_rows <= 0:
                        reason = "PACK_HAS_NO_REAL_BRIDGE_ROWS"
                    elif not has_reject_bucket and bool(line_bridge_schema.get("field_name_mismatch", False)):
                        reason = "BRIDGE_FIELD_NAME_MISMATCH"
                    elif not has_reject_bucket and int(bridge_match_diag.get("matched_rows", 0) or 0) <= 0:
                        reason = endpoint_class if endpoint_class and endpoint_class != "HAS_ENDPOINT_EDGE" else "ENDPOINT_KEY_MISMATCH"
                    dominant_fail_reason = _dominant_bridge_fail_reason(bridge_candidate_audit, reason_buckets)
                    print(
                        f"[APS][REPAIR_BRIDGE_RULE_AUDIT] line={line}, block_id={block_id}, "
                        f"thickness_fail_count={int(bridge_candidate_audit.get('pair_invalid_thickness', 0) or 0)}, "
                        f"width_fail_count={int(bridge_candidate_audit.get('pair_invalid_width', 0) or 0)}, "
                        f"temp_fail_count={int(bridge_candidate_audit.get('pair_invalid_temp', 0) or 0)}, "
                        f"group_fail_count={int(bridge_candidate_audit.get('pair_invalid_group', 0) or 0)}, "
                        f"template_no_edge_count={1 if endpoint_class == 'TEMPLATE_GRAPH_NO_EDGE' else 0}, "
                        f"prefilter_all_fail_count={1 if endpoint_class == 'RULE_PREFILTER_ALL_FAIL' else 0}, "
                        f"band_too_narrow_count={1 if endpoint_class == 'BAND_TOO_NARROW' else 0}, "
                        f"thickness_prefilter_fail_count={int(bridge_candidate_audit.get('prefilter_fail_thickness', 0) or 0)}, "
                        f"thickness_rank_pass_count={int(bridge_candidate_audit.get('rank_pass_thickness', 0) or 0)}, "
                        f"thickness_rank_fail_count={int(bridge_candidate_audit.get('rank_fail_thickness', 0) or 0)}"
                    )
                    print(
                        f"[APS][REPAIR_BRIDGE_REASON_AUDIT] line={line}, block_id={block_id}, "
                        f"filtered_pair_invalid={int(real_diag_report.get('repair_only_real_bridge_filtered_pair_invalid', 0) or 0)}, "
                        f"filtered_context_missing={int(bridge_candidate_audit.get('pair_invalid_context', 0) or 0)}, "
                        f"filtered_ton_invalid={int(filtered_ton_invalid_effective)}, "
                        f"filtered_ton_below_min_current_block={int(bridge_candidate_audit.get('filtered_ton_below_min_current_block', 0) or 0)}, "
                        f"filtered_ton_rescue_impossible={int(rescue_reason_counts.get('TON_RESCUE_IMPOSSIBLE', 0) or 0)}, "
                        f"filtered_ton_below_min_even_after_neighbor_expansion={int(rescue_reason_counts.get('TON_BELOW_MIN_EVEN_AFTER_NEIGHBOR_EXPANSION', 0) or 0)}, "
                        f"filtered_ton_split_not_found={int(diag.get('repair_bridge_filtered_ton_split_not_found', 0) or 0)}, "
                        f"filtered_ton_rescue_no_gain={int(diag.get('repair_bridge_filtered_ton_rescue_no_gain', 0) or 0)}, "
                        f"filtered_score_worse={int(real_diag_report.get('repair_only_real_bridge_filtered_score_worse', 0) or 0)}, "
                        f"dominant_bucket={dominant_bucket}, "
                        f"final_reason={reason}"
                    )
                    if reason != dominant_bucket and int(bridge_match_diag.get("matched_rows", 0) or 0) > 0:
                        print(
                            f"[APS][REPAIR_BRIDGE_REASON_MISMATCH] line={line}, block_id={block_id}, "
                            f"final_reason={reason}, dominant_bucket={dominant_bucket}"
                        )
                    print(f"[APS][REPAIR_BRIDGE_NO_HIT] line={line}, block_id={block_id}, reason={reason}")
                    _record_bridgeability_census_item(
                        bridgeability_census,
                        {
                            "line": str(line),
                            "block_id": int(block_id),
                            "block_size": int(block_size),
                            "orders": int(len(block_orders)),
                            "tons": float(block_tons),
                            "endpoint_class": str(endpoint_class),
                            "dominant_fail_reason": str(dominant_fail_reason),
                            "candidate_count": int(bridge_match_diag.get("matched_rows", 0) or 0),
                            "frontier_candidate_count": int(frontier_match_diag.get("matched_rows", 0) or 0),
                            "prefilter_rejected": bool(int(bridge_candidate_audit.get("prefilter_reject_count", 0) or 0) > 0),
                            "ton_rescue_entered": bool(len(ton_rescue_candidates) > 0 and int(frontier_match_diag.get("matched_rows", 0) or 0) > 0),
                            "ton_rescue_success": False,
                            "final_decision": str(reason),
                        },
                    )
                    pilot_target_candidates = _select_virtual_pilot_targets(
                        list(bridge_candidate_audit.get("pilot_candidates", []) or []),
                        endpoint_class=str(endpoint_class),
                        max_targets=1,
                        allowed_fail_reasons=virtual_pilot_fail_reasons,
                        campaign_ton_min=float(campaign_ton_min),
                    )
                    small_hard_reject, small_block_penalty = _small_block_pilot_score(
                        int(len(block_orders)),
                        float(block_tons),
                        float(campaign_ton_min),
                    )
                    if float(small_block_penalty) > 0.0 and not bool(small_hard_reject):
                        diag["virtual_pilot_small_block_soft_penalty_count"] += 1
                    print(
                        f"[APS][VIRTUAL_BRIDGE_SMALL_BLOCK_SCORE] line={line}, block_id={block_id}, "
                        f"block_orders={int(len(block_orders))}, block_tons={float(block_tons):.1f}, "
                        f"small_block_penalty={float(small_block_penalty):.3f}, hard_reject={bool(small_hard_reject)}"
                    )
                    eligible_result = _evaluate_virtual_pilot_eligibility(
                        endpoint_class=str(endpoint_class),
                        candidate_count=int(bridge_match_diag.get("matched_rows", 0) or 0),
                        frontier_candidate_count=int(frontier_match_diag.get("matched_rows", 0) or 0),
                        pilot_candidates=pilot_target_candidates,
                        block_orders_count=int(len(block_orders)),
                        block_tons=float(block_tons),
                        campaign_ton_min=float(campaign_ton_min),
                        attempts_so_far=int(diag.get("virtual_pilot_attempt_count", 0) or 0),
                        max_attempts=int(virtual_pilot_max_blocks),
                        allowed_endpoint_classes=virtual_pilot_endpoint_classes,
                        already_tried=(str(line), int(block_id)) in virtual_pilot_tried_blocks,
                        applied_improvement_count=int(diag.get("underfilled_reconstruction_improvement_applied_count", 0) or 0),
                        enabled=bool(virtual_pilot_enabled),
                    )
                    pilot_eligible = bool(eligible_result.final_eligible)
                    reject_by_reason = diag.setdefault("virtual_pilot_reject_by_reason_count", {})
                    if bool(eligible_result.structural_eligible):
                        diag["virtual_pilot_structural_eligible_block_count"] += 1
                    if bool(eligible_result.runtime_enabled):
                        diag["virtual_pilot_runtime_enabled_block_count"] += 1
                    if bool(eligible_result.final_eligible):
                        diag["virtual_pilot_final_eligible_block_count"] += 1
                        diag["virtual_pilot_eligible_block_count"] += 1
                    else:
                        diag["virtual_pilot_skipped_block_count"] += 1
                        reject_by_reason[str(eligible_result.reject_reason)] = int(reject_by_reason.get(str(eligible_result.reject_reason), 0) or 0) + 1
                        if str(eligible_result.reject_reason) == "PILOT_DISABLED":
                            diag["virtual_pilot_skipped_due_to_disabled_count"] += 1
                        elif str(eligible_result.reject_reason) == "RUN_LIMIT_REACHED":
                            diag["virtual_pilot_skipped_due_to_limit_count"] += 1
                        elif str(eligible_result.reject_reason) in {"NO_PILOTABLE_CANDIDATE", "DOMINANT_FAIL_NOT_PILOTABLE"}:
                            diag["virtual_pilot_skipped_due_to_no_pilotable_candidate_count"] += 1
                    print(
                        f"[APS][VIRTUAL_BRIDGE_PILOT_ELIGIBLE] line={line}, block_id={block_id}, "
                        f"endpoint_class={endpoint_class}, dominant_fail_reason={dominant_fail_reason}, "
                        f"eligible={bool(pilot_eligible)}"
                    )
                    print(
                        f"[APS][VIRTUAL_BRIDGE_PILOT_ELIGIBILITY_AUDIT] line={line}, block_id={block_id}, "
                        f"endpoint_class={endpoint_class}, "
                        f"candidate_count={int(bridge_match_diag.get('matched_rows', 0) or 0)}, "
                        f"frontier_candidate_count={int(frontier_match_diag.get('matched_rows', 0) or 0)}, "
                        f"structural_eligible={bool(eligible_result.structural_eligible)}, "
                        f"runtime_enabled={bool(eligible_result.runtime_enabled)}, "
                        f"final_eligible={bool(eligible_result.final_eligible)}, "
                        f"reject_reason={eligible_result.reject_reason}, "
                        f"reject_details={eligible_result.reject_details}"
                    )
                    print(
                        f"[APS][VIRTUAL_BRIDGE_PILOT_TARGETS] line={line}, block_id={block_id}, "
                        f"selected_candidates={[{'candidate_id': c.get('candidate_id'), 'pair': c.get('pair'), 'fail_reasons': c.get('fail_reasons'), 'rank_score': c.get('rank_score')} for c in pilot_target_candidates]}, "
                        f"reason={eligible_result.reject_reason}"
                    )
                    if pilot_eligible:
                        target = dict(pilot_target_candidates[0])
                        scheduler_family = _virtual_pilot_failure_family(target)
                        scheduler_bucket = scheduler_family
                        pilot_key = _virtual_pilot_key(str(line), target)
                        _record_virtual_pilot_execution_stage(
                            diag,
                            line=str(line),
                            block_id=int(block_id),
                            candidate_id=str(target.get("candidate_id", "")),
                            family=scheduler_family,
                            stage="SELECTED",
                            details={"pilot_key": pilot_key},
                        )
                        if pilot_key in virtual_pilot_seen_keys:
                            diag["virtual_pilot_skipped_block_count"] += 1
                            diag["virtual_pilot_duplicate_candidate_skipped_count"] += 1
                            _record_virtual_pilot_execution_stage(
                                diag,
                                line=str(line),
                                block_id=int(block_id),
                                candidate_id=str(target.get("candidate_id", "")),
                                family=scheduler_family,
                                stage="DEDUP_SKIPPED",
                                details={"pilot_key": pilot_key, "reason": "DUPLICATE_PILOT_KEY"},
                            )
                            reject_by_reason = diag.setdefault("virtual_pilot_reject_by_reason_count", {})
                            reject_by_reason["DUPLICATE_PILOT_KEY"] = int(reject_by_reason.get("DUPLICATE_PILOT_KEY", 0) or 0) + 1
                            print(
                                f"[APS][VIRTUAL_BRIDGE_PILOT_DEDUP] candidate_id={target.get('candidate_id')}, "
                                f"pilot_key={pilot_key}, kept=False, reason=DUPLICATE_PILOT_KEY"
                            )
                            continue
                        virtual_pilot_seen_keys.add(pilot_key)
                        diag["virtual_pilot_dedup_group_count"] += 1
                        _record_virtual_pilot_execution_stage(
                            diag,
                            line=str(line),
                            block_id=int(block_id),
                            candidate_id=str(target.get("candidate_id", "")),
                            family=scheduler_family,
                            stage="DEDUP_KEPT",
                            details={"pilot_key": pilot_key},
                        )
                        print(
                            f"[APS][VIRTUAL_BRIDGE_PILOT_DEDUP] candidate_id={target.get('candidate_id')}, "
                            f"pilot_key={pilot_key}, kept=True, reason=BEST_OF_DUP_GROUP"
                        )
                        scheduler_allowed = (
                            int(diag.get("virtual_pilot_attempt_count", 0) or 0) < int(virtual_pilot_max_blocks)
                            and int(virtual_pilot_selected_by_line[str(line)]) < int(virtual_pilot_line_quota)
                            and int(virtual_pilot_selected_by_bucket[scheduler_bucket]) < int(virtual_pilot_family_quota.get(scheduler_bucket, virtual_pilot_bucket_quota))
                        )
                        if not scheduler_allowed:
                            diag["virtual_pilot_skipped_block_count"] += 1
                            diag["virtual_pilot_skipped_due_to_limit_count"] += 1
                            skip_item = {
                                "line": str(line),
                                "block_id": int(block_id),
                                "bucket": scheduler_bucket,
                                "family": scheduler_family,
                                "pilot_key": pilot_key,
                                "line_selected": int(virtual_pilot_selected_by_line[str(line)]),
                                "bucket_selected": int(virtual_pilot_selected_by_bucket[scheduler_bucket]),
                                "bucket_quota": int(virtual_pilot_family_quota.get(scheduler_bucket, virtual_pilot_bucket_quota)),
                            }
                            virtual_pilot_scheduler_skipped_due_to_limit.append(skip_item)
                            reject_by_reason = diag.setdefault("virtual_pilot_reject_by_reason_count", {})
                            reject_by_reason["RUN_LIMIT_REACHED"] = int(reject_by_reason.get("RUN_LIMIT_REACHED", 0) or 0) + 1
                            continue
                        diag["virtual_pilot_selected_block_count"] += 1
                        diag["virtual_pilot_selected_unique_pilot_key_count"] += 1
                        virtual_pilot_tried_blocks.add((str(line), int(block_id)))
                        virtual_pilot_selected_by_line[str(line)] += 1
                        virtual_pilot_selected_by_bucket[scheduler_bucket] += 1
                        virtual_pilot_selected_by_family[scheduler_family] += 1
                        selected_by_family_diag = diag.setdefault("virtual_pilot_selected_by_family_count", {})
                        selected_by_family_diag[scheduler_family] = int(selected_by_family_diag.get(scheduler_family, 0) or 0) + 1
                        pair = (str(target.get("from_order_id", "")), str(target.get("to_order_id", "")))
                        pilot_virtual_edges = {pair}
                        pilot_virtual_tons = max(float(virtual_tons_by_edge.get(pair, 0.0) or 0.0), 0.0)
                        virtual_pilot_scheduler_selected_blocks.append(
                            {"line": str(line), "block_id": int(block_id), "bucket": scheduler_bucket, "family": scheduler_family, "pilot_key": pilot_key, "pair": pair}
                        )
                        left_meta = _order_pair_meta(order_specs_by_id, pair[0])
                        right_meta = _order_pair_meta(order_specs_by_id, pair[1])
                        print(
                            f"[APS][VIRTUAL_BRIDGE_SPEC_FAMILY] candidate_id={target.get('candidate_id')}, "
                            f"family={scheduler_family}"
                        )
                        specs, spec_diag, selected_spec = _enumerate_virtual_pilot_specs(
                            left_meta,
                            right_meta,
                            cfg.rule if cfg and cfg.rule else None,
                            max_specs=5,
                            family=scheduler_family,
                        )
                        diag["virtual_pilot_spec_enum_total"] += int(spec_diag.get("specs_tested", 0) or 0)
                        diag["virtual_pilot_spec_enum_both_valid_count"] += int(spec_diag.get("both_valid_count", 0) or 0)
                        _record_virtual_pilot_execution_stage(
                            diag,
                            line=str(line),
                            block_id=int(block_id),
                            candidate_id=str(target.get("candidate_id", "")),
                            family=scheduler_family,
                            stage="SPEC_ENUM_DONE",
                            details=dict(spec_diag),
                        )
                        print(
                            f"[APS][VIRTUAL_BRIDGE_SPEC_ENUM] line={line}, block_id={block_id}, "
                            f"candidate_id={target.get('candidate_id')}, family={scheduler_family}, "
                            f"specs_tested={int(spec_diag.get('specs_tested', 0) or 0)}, "
                            f"left_valid_count={int(spec_diag.get('left_valid_count', 0) or 0)}, "
                            f"right_valid_count={int(spec_diag.get('right_valid_count', 0) or 0)}, "
                            f"both_valid_count={int(spec_diag.get('both_valid_count', 0) or 0)}"
                        )
                        family_prefilter_ok, family_prefilter_reason = _virtual_pilot_family_prefilter(scheduler_family, selected_spec)
                        width_group_guaranteed = False
                        if (
                            scheduler_family == "WIDTH_GROUP"
                            and selected_spec is not None
                            and not bool(family_prefilter_ok)
                        ):
                            family_prefilter_ok = True
                            width_group_guaranteed = True
                            diag["virtual_pilot_width_group_guarantee_attempted"] = True
                        print(
                            f"[APS][VIRTUAL_BRIDGE_FAMILY_PREFILTER] candidate_id={target.get('candidate_id')}, "
                            f"family={scheduler_family}, pass={bool(family_prefilter_ok)}, reason={family_prefilter_reason}"
                        )
                        if scheduler_family == "WIDTH_GROUP":
                            print(
                                f"[APS][VIRTUAL_BRIDGE_FAMILY_GUARANTEE] family=WIDTH_GROUP, "
                                f"guaranteed_attempt={bool(width_group_guaranteed)}, "
                                f"reason={'WIDTH_GROUP_PREFILTER_BYPASS' if width_group_guaranteed else family_prefilter_reason}"
                            )
                        if selected_spec is None:
                            diag["virtual_pilot_reject_count"] += 1
                            fail_stage, fail_reason, fail_details = _classify_virtual_transition_spec_failure(specs, spec_diag)
                            _record_virtual_pilot_fail_stage(
                                diag,
                                line=str(line),
                                block_id=int(block_id),
                                candidate_id=str(target.get("candidate_id", "")),
                                fail_stage=str(fail_stage),
                                reason=str(fail_reason),
                                details=dict(fail_details),
                            )
                            print(
                                f"[APS][VIRTUAL_BRIDGE_PILOT_FAIL] line={line}, block_id={block_id}, "
                                f"reason={fail_reason}"
                            )
                            _record_virtual_pilot_execution_stage(
                                diag,
                                line=str(line),
                                block_id=int(block_id),
                                candidate_id=str(target.get("candidate_id", "")),
                                family=scheduler_family,
                                stage="FAIL",
                                details={"reason": str(fail_reason), "fail_stage": str(fail_stage)},
                            )
                            continue
                        if not family_prefilter_ok:
                            diag["virtual_pilot_reject_count"] += 1
                            diag["virtual_pilot_family_prefilter_fail_count"] += 1
                            _record_virtual_pilot_fail_stage(
                                diag,
                                line=str(line),
                                block_id=int(block_id),
                                candidate_id=str(target.get("candidate_id", "")),
                                fail_stage="FAMILY_PREFILTER_FAIL",
                                reason=str(family_prefilter_reason),
                                details={"family": scheduler_family, "selected_spec": selected_spec},
                            )
                            print(
                                f"[APS][VIRTUAL_BRIDGE_PILOT_FAIL] line={line}, block_id={block_id}, "
                                f"reason=FAMILY_PREFILTER_FAIL"
                            )
                            _record_virtual_pilot_execution_stage(
                                diag,
                                line=str(line),
                                block_id=int(block_id),
                                candidate_id=str(target.get("candidate_id", "")),
                                family=scheduler_family,
                                stage="FAIL",
                                details={"reason": "FAMILY_PREFILTER_FAIL"},
                            )
                            continue
                        print(
                            f"[APS][VIRTUAL_BRIDGE_SPEC_SELECTED] line={line}, block_id={block_id}, "
                            f"candidate_id={target.get('candidate_id')}, family={scheduler_family}, "
                            f"selected_spec={{width:{selected_spec.get('width')}, thickness:{selected_spec.get('thickness')}, "
                            f"temp_low:{selected_spec.get('temp_low')}, temp_high:{selected_spec.get('temp_high')}}}"
                        )
                        diag["virtual_pilot_attempt_count"] += 1
                        _record_virtual_pilot_execution_stage(
                            diag,
                            line=str(line),
                            block_id=int(block_id),
                            candidate_id=str(target.get("candidate_id", "")),
                            family=scheduler_family,
                            stage="ATTEMPT_STARTED",
                            details={"pair": pair, "virtual_tons": float(pilot_virtual_tons)},
                        )
                        if scheduler_family == "WIDTH_GROUP":
                            diag["virtual_pilot_width_group_family_attempt_count"] += 1
                        elif scheduler_family == "THICKNESS":
                            diag["virtual_pilot_thickness_family_attempt_count"] += 1
                        print(
                            f"[APS][VIRTUAL_BRIDGE_PILOT_ATTEMPT] line={line}, block_id={block_id}, "
                            f"candidate_id={target.get('candidate_id')}, pair=({pair[0]},{pair[1]}), "
                            f"virtual_tons={pilot_virtual_tons:.1f}, "
                            f"penalty={float(virtual_pilot_penalty):.1f}"
                        )
                        if pilot_virtual_tons > float(virtual_pilot_max_tons) + 1e-6:
                            diag["virtual_pilot_reject_count"] += 1
                            _record_virtual_pilot_fail_stage(
                                diag,
                                line=str(line),
                                block_id=int(block_id),
                                candidate_id=str(target.get("candidate_id", "")),
                                fail_stage="VIRTUAL_TON_BELOW_MIN",
                                reason="VIRTUAL_TON_LIMIT_EXCEEDED",
                                details={"virtual_tons": float(pilot_virtual_tons), "max_virtual_tons": float(virtual_pilot_max_tons)},
                            )
                            print(
                                f"[APS][VIRTUAL_BRIDGE_PILOT_FAIL] line={line}, block_id={block_id}, "
                                f"reason=VIRTUAL_TON_LIMIT_EXCEEDED"
                            )
                            _record_virtual_pilot_execution_stage(
                                diag,
                                line=str(line),
                                block_id=int(block_id),
                                candidate_id=str(target.get("candidate_id", "")),
                                family=scheduler_family,
                                stage="FAIL",
                                details={"reason": "VIRTUAL_TON_LIMIT_EXCEEDED"},
                            )
                        else:
                            _record_virtual_pilot_execution_stage(
                                diag,
                                line=str(line),
                                block_id=int(block_id),
                                candidate_id=str(target.get("candidate_id", "")),
                                family=scheduler_family,
                                stage="RECUT_ENTERED",
                                details={"block_size": int(block_size)},
                            )
                            print(
                                f"[APS][VIRTUAL_BRIDGE_RECUT_ENTER] line={line}, block_id={block_id}, "
                                f"candidate_id={target.get('candidate_id')}, family={scheduler_family}"
                            )
                            pilot_valid, pilot_under, pilot_diag = _solve_underfilled_reconstruction_block(
                                block,
                                direct_edges=direct_edges,
                                real_edges={(str(k[1]), str(k[2])) for k in frontier_matched_keys},
                                order_tons=tons_by_order,
                                campaign_ton_min=campaign_ton_min,
                                campaign_ton_max=campaign_ton_max,
                                campaign_ton_target=ton_target,
                                allow_real_bridge=True,
                                allow_virtual_bridge=True,
                                max_real_bridge_per_segment=max_real,
                                bridge_cost_penalty=bridge_penalty,
                                next_local_id=max_local_id,
                                real_edge_distance=real_edge_distance,
                                real_edge_adjustment_cost=real_edge_adjustment_cost,
                                virtual_edges=pilot_virtual_edges,
                                max_virtual_bridge_per_segment=int(virtual_pilot_max_per_block),
                                virtual_tons_by_edge=virtual_tons_by_edge,
                                virtual_bridge_penalty=float(virtual_pilot_penalty),
                            )
                            pilot_virtual_used = int(pilot_diag.get("virtual_bridge_used", 0) or 0)
                            pilot_virtual_used_tons = float(pilot_diag.get("virtual_bridge_tons", 0.0) or 0.0)
                            print(
                                f"[APS][VIRTUAL_BRIDGE_SEGMENT_VALIDATE] line={line}, block_id={block_id}, "
                                f"candidate_id={target.get('candidate_id')}, family={scheduler_family}, "
                                f"pilot_valid={bool(pilot_valid)}, virtual_used={pilot_virtual_used}, "
                                f"virtual_tons={pilot_virtual_used_tons:.1f}"
                            )
                            if pilot_valid:
                                diag["virtual_pilot_segment_valid_count"] += 1
                            if (not pilot_valid) and int(pilot_diag.get("candidate_cutpoints_ton_window_valid", 0) or 0) <= 0:
                                diag["virtual_pilot_ton_fill_attempt_count"] += 1
                                _record_virtual_pilot_execution_stage(
                                    diag,
                                    line=str(line),
                                    block_id=int(block_id),
                                    candidate_id=str(target.get("candidate_id", "")),
                                    family=scheduler_family,
                                    stage="TON_FILL_ENTERED",
                                    details={"reason": "NO_TON_WINDOW_VALID_CUTPOINT"},
                                )
                                print(
                                    f"[APS][VIRTUAL_BRIDGE_TON_FILL_ENTER] line={line}, block_id={block_id}, "
                                    f"candidate_id={target.get('candidate_id')}, family={scheduler_family}"
                                )
                                fill_block = list(block)
                                fill_success = False
                                fill_after_tons = float(block_tons)
                                if i + block_size < len(line_under):
                                    fill_block = list(block) + [line_under[i + block_size]]
                                    fill_after_tons = sum(
                                        float(tons_by_order.get(str(oid), 0.0) or 0.0)
                                        for seg in fill_block
                                        for oid in seg.order_ids
                                    )
                                    fill_valid, fill_under, fill_diag = _solve_underfilled_reconstruction_block(
                                        fill_block,
                                        direct_edges=direct_edges,
                                        real_edges={(str(k[1]), str(k[2])) for k in frontier_matched_keys},
                                        order_tons=tons_by_order,
                                        campaign_ton_min=campaign_ton_min,
                                        campaign_ton_max=campaign_ton_max,
                                        campaign_ton_target=ton_target,
                                        allow_real_bridge=True,
                                        allow_virtual_bridge=True,
                                        max_real_bridge_per_segment=max_real,
                                        bridge_cost_penalty=bridge_penalty,
                                        next_local_id=max_local_id,
                                        real_edge_distance=real_edge_distance,
                                        real_edge_adjustment_cost=real_edge_adjustment_cost,
                                        virtual_edges=pilot_virtual_edges,
                                        max_virtual_bridge_per_segment=int(virtual_pilot_max_per_block),
                                        virtual_tons_by_edge=virtual_tons_by_edge,
                                        virtual_bridge_penalty=float(virtual_pilot_penalty),
                                    )
                                    fill_virtual_used = int(fill_diag.get("virtual_bridge_used", 0) or 0)
                                    fill_virtual_tons = float(fill_diag.get("virtual_bridge_tons", 0.0) or 0.0)
                                    fill_success = bool(
                                        fill_valid
                                        and 0 < fill_virtual_used <= int(virtual_pilot_max_per_block)
                                        and fill_virtual_tons <= float(virtual_pilot_max_tons) + 1e-6
                                    )
                                    if fill_success:
                                        diag["virtual_pilot_ton_fill_success_count"] += 1
                                        pilot_valid, pilot_under, pilot_diag = fill_valid, fill_under, fill_diag
                                        pilot_virtual_used = fill_virtual_used
                                        pilot_virtual_used_tons = fill_virtual_tons
                                        block = fill_block
                                        block_size += 1
                                print(
                                    f"[APS][VIRTUAL_BRIDGE_TON_FILL_ATTEMPT] line={line}, block_id={block_id}, "
                                    f"candidate_id={target.get('candidate_id')}, mode=EXPAND_ONE_NEIGHBOR, "
                                    f"before_tons={float(block_tons):.1f}, after_tons={float(fill_after_tons):.1f}, "
                                    f"success={bool(fill_success)}"
                                )
                            if pilot_valid and 0 < pilot_virtual_used <= int(virtual_pilot_max_per_block) and pilot_virtual_used_tons <= float(virtual_pilot_max_tons) + 1e-6:
                                diag["virtual_pilot_success_count"] += 1
                                pilot_diag["candidate_id"] = str(target.get("candidate_id", ""))
                                pilot_diag["family"] = str(scheduler_family)
                                best = (pilot_valid, pilot_under, pilot_diag, "VIRTUAL_PILOT", block)
                                best_size = block_size
                                print(
                                    f"[APS][VIRTUAL_BRIDGE_PILOT_SUCCESS] line={line}, block_id={block_id}, "
                                    f"salvaged_segments={len(pilot_valid)}, "
                                    f"salvaged_orders={int(pilot_diag.get('salvaged_orders', 0) or 0)}"
                                )
                                _record_virtual_pilot_execution_stage(
                                    diag,
                                    line=str(line),
                                    block_id=int(block_id),
                                    candidate_id=str(target.get("candidate_id", "")),
                                    family=scheduler_family,
                                    stage="SUCCESS",
                                    details={
                                        "salvaged_segments": int(len(pilot_valid)),
                                        "salvaged_orders": int(pilot_diag.get("salvaged_orders", 0) or 0),
                                    },
                                )
                                break
                            diag["virtual_pilot_reject_count"] += 1
                            if pilot_virtual_used > int(virtual_pilot_max_per_block):
                                fail_stage, fail_reason, fail_details = "VIRTUAL_TRANSITION_INVALID", "VIRTUAL_BRIDGE_LIMIT_EXCEEDED", {"virtual_used": int(pilot_virtual_used), "max_virtual_used": int(virtual_pilot_max_per_block)}
                            elif pilot_virtual_used_tons > float(virtual_pilot_max_tons) + 1e-6:
                                fail_stage, fail_reason, fail_details = "VIRTUAL_TON_BELOW_MIN", "VIRTUAL_TON_LIMIT_EXCEEDED", {"virtual_tons": float(pilot_virtual_used_tons), "max_virtual_tons": float(virtual_pilot_max_tons)}
                            else:
                                fail_stage, fail_reason, fail_details = _classify_virtual_pilot_recut_fail(pilot_diag, pilot_virtual_used, pilot_virtual_used_tons)
                            _record_virtual_pilot_fail_stage(
                                diag,
                                line=str(line),
                                block_id=int(block_id),
                                candidate_id=str(target.get("candidate_id", "")),
                                fail_stage=str(fail_stage),
                                reason=str(fail_reason),
                                details=dict(fail_details),
                            )
                            if int(spec_diag.get("both_valid_count", 0) or 0) > 0:
                                if fail_stage == "VIRTUAL_LOCAL_RECUT_FAIL":
                                    post_stage = "VIRTUAL_LOCAL_RECUT_FAIL"
                                elif fail_stage in {"VIRTUAL_BOTH_TRANSITIONS_INVALID", "VIRTUAL_TRANSITION_INVALID"}:
                                    post_stage = "VIRTUAL_SEGMENT_INVALID_AFTER_RECUT"
                                elif fail_stage == "VIRTUAL_TON_BELOW_MIN":
                                    post_stage = "VIRTUAL_TON_FILL_FAIL"
                                else:
                                    post_stage = "VIRTUAL_POST_SPEC_UNKNOWN_FAIL"
                                _record_virtual_post_spec_fail(
                                    diag,
                                    line=str(line),
                                    block_id=int(block_id),
                                    candidate_id=str(target.get("candidate_id", "")),
                                    family=scheduler_family,
                                    fail_stage=post_stage,
                                    details=dict(fail_details),
                                )
                            print(
                                f"[APS][VIRTUAL_BRIDGE_PILOT_FAIL] line={line}, block_id={block_id}, "
                                f"reason={fail_reason}"
                            )
                            _record_virtual_pilot_execution_stage(
                                diag,
                                line=str(line),
                                block_id=int(block_id),
                                candidate_id=str(target.get("candidate_id", "")),
                                family=scheduler_family,
                                stage="FAIL",
                                details={"reason": str(fail_reason), "fail_stage": str(fail_stage)},
                            )
            if best is None:
                diag["underfilled_reconstruction_blocks_skipped"] += 1
                valid_before = 0
                valid_after = 0
                underfilled_before = len(block) if "block" in locals() else 1
                underfilled_after = underfilled_before
                scheduled_orders_before = 0
                scheduled_orders_after = 0
                gain = False
                print(
                    f"[APS][UNDERFILLED_RECON_GAIN_AUDIT] line={line}, block_id={block_id}, "
                    f"valid_before={valid_before}, valid_after={valid_after}, "
                    f"underfilled_before={underfilled_before}, underfilled_after={underfilled_after}, "
                    f"scheduled_orders_before={scheduled_orders_before}, "
                    f"scheduled_orders_after={scheduled_orders_after}, gain={bool(gain)}"
                )
                if not gain:
                    print(
                        f"[APS][UNDERFILLED_RECON_NO_GAIN] line={line}, block_id={block_id}, "
                        f"valid_before={valid_before}, valid_after={valid_after}, "
                        f"underfilled_before={underfilled_before}, underfilled_after={underfilled_after}"
                    )
                final_decision = "PREFILTER_ALL_FAIL" if endpoint_class == "RULE_PREFILTER_ALL_FAIL" else "NO_DIRECT_OR_BRIDGE_CANDIDATE"
                _update_bridgeability_census_final_decision(
                    bridgeability_census,
                    line=str(line),
                    block_id=int(block_id),
                    final_decision=final_decision,
                )
                print(
                    f"[APS][UNDERFILLED_RECON_FINAL_DECISION] line={line}, block_id={block_id}, "
                    f"gain={bool(gain)}, final_decision={final_decision}, reason={final_decision}"
                )
                print(f"[APS][UNDERFILLED_RECON_SKIP] line={line}, block_id={block_id}, reason={final_decision}")
                remaining_under.append(line_under[i])
                i += 1
                continue

            valid_out, under_out, block_diag, mode, block = best
            before_counter = Counter()
            after_counter = Counter()
            for seg in block:
                before_counter.update([str(v) for v in seg.order_ids])
            for seg in valid_out + under_out:
                after_counter.update([str(v) for v in seg.order_ids])
            if before_counter != after_counter:
                diag["underfilled_reconstruction_blocks_skipped"] += 1
                diag["repair_only_real_bridge_filtered_multiplicity_invalid"] += 1
                diag["repair_only_real_bridge_filtered_block_order_mismatch"] += 1
                print(f"[APS][UNDERFILLED_RECON_SKIP] line={line}, block_id={block_id}, reason=BLOCK_ORDER_MISMATCH")
                remaining_under.extend(block)
                i += best_size
                continue

            valid_before = len(valid_segments)
            consumed_prev_ids = {int(seg.campaign_local_id) for seg in block if int(seg.campaign_local_id) < int(line_under[i].campaign_local_id)}
            remaining_after_apply = [seg for seg in remaining_under if int(seg.campaign_local_id) not in consumed_prev_ids] if consumed_prev_ids else list(remaining_under)
            valid_after = len(new_valid) + len(valid_out)
            underfilled_before = len(underfilled_segments)
            underfilled_after = len(remaining_after_apply) + len(under_out)
            scheduled_orders_before = sum(len(seg.order_ids) for seg in valid_segments)
            scheduled_orders_after = sum(len(seg.order_ids) for seg in new_valid + valid_out)
            recon_gain = (
                int(valid_after) > int(valid_before)
                or int(underfilled_after) < int(underfilled_before)
                or int(scheduled_orders_after) > int(scheduled_orders_before)
            )
            apply_real_edges = real_edges if mode in {"REAL", "VIRTUAL_PILOT"} else set()
            apply_virtual_edges = virtual_edges if mode == "VIRTUAL_PILOT" else set()
            pair_integrity_ok, pair_integrity_reason = _segments_pair_integrity_ok(
                list(valid_out),
                direct_edges=direct_edges,
                real_edges=apply_real_edges,
                max_real_bridge_per_segment=max_real,
                virtual_edges=apply_virtual_edges,
                max_virtual_bridge_per_segment=int(virtual_pilot_max_per_block) if mode == "VIRTUAL_PILOT" else 0,
            )
            virtual_pilot_limit_ok = True
            if mode == "VIRTUAL_PILOT":
                virtual_pilot_limit_ok = (
                    int(block_diag.get("virtual_bridge_used", 0) or 0) <= int(virtual_pilot_max_per_block)
                    and float(block_diag.get("virtual_bridge_tons", 0.0) or 0.0) <= float(virtual_pilot_max_tons) + 1e-6
                )
            apply_ok, apply_reject_reason = _should_apply_improvement(
                valid_before=valid_before,
                valid_after=valid_after,
                underfilled_before=underfilled_before,
                underfilled_after=underfilled_after,
                scheduled_orders_before=scheduled_orders_before,
                scheduled_orders_after=scheduled_orders_after,
                integrity_ok=True,
                pair_integrity_ok=pair_integrity_ok,
            )
            if not pair_integrity_ok and apply_reject_reason in {"", "PAIR_INTEGRITY_FAIL"}:
                apply_reject_reason = pair_integrity_reason or "PAIR_INTEGRITY_FAIL"
            if apply_ok and not virtual_pilot_limit_ok:
                apply_ok = False
                apply_reject_reason = "VIRTUAL_PILOT_LIMIT_FAIL"
            if mode == "VIRTUAL_PILOT":
                virtual_apply_ok, virtual_apply_reject = _should_apply_virtual_pilot(
                    gain=bool(recon_gain),
                    multiplicity_ok=True,
                    pair_integrity_ok=bool(pair_integrity_ok),
                    virtual_used=int(block_diag.get("virtual_bridge_used", 0) or 0),
                    max_virtual_used=int(virtual_pilot_max_per_block),
                    virtual_tons=float(block_diag.get("virtual_bridge_tons", 0.0) or 0.0),
                    max_virtual_tons=float(virtual_pilot_max_tons),
                )
                if not virtual_apply_ok:
                    apply_ok = False
                    apply_reject_reason = virtual_apply_reject
            print(
                f"[APS][UNDERFILLED_RECON_GAIN_AUDIT] line={line}, block_id={block_id}, "
                f"valid_before={valid_before}, valid_after={valid_after}, "
                f"underfilled_before={underfilled_before}, underfilled_after={underfilled_after}, "
                f"scheduled_orders_before={scheduled_orders_before}, "
                f"scheduled_orders_after={scheduled_orders_after}, gain={bool(recon_gain)}"
            )
            print(
                f"[APS][UNDERFILLED_RECON_APPLY_CHECK] line={line}, block_id={block_id}, "
                f"valid_before={valid_before}, valid_after={valid_after}, "
                f"underfilled_before={underfilled_before}, underfilled_after={underfilled_after}, "
                f"scheduled_orders_before={scheduled_orders_before}, scheduled_orders_after={scheduled_orders_after}, "
                f"integrity_ok={bool(pair_integrity_ok)}, apply={bool(apply_ok)}, reject_reason={apply_reject_reason}"
            )
            if mode == "VIRTUAL_PILOT":
                _record_virtual_pilot_execution_stage(
                    diag,
                    line=str(line),
                    block_id=int(block_id),
                    candidate_id=str(block_diag.get("candidate_id", "")),
                    family=str(block_diag.get("family", "")),
                    stage="APPLY_CHECK_ENTERED",
                    details={
                        "gain": bool(recon_gain),
                        "pair_integrity_ok": bool(pair_integrity_ok),
                        "virtual_pilot_limit_ok": bool(virtual_pilot_limit_ok),
                    },
                )
                print(
                    f"[APS][VIRTUAL_BRIDGE_APPLY_CHECK_ENTER] line={line}, block_id={block_id}, "
                    f"candidate_id={block_diag.get('candidate_id', '')}"
                )
                print(
                    f"[APS][VIRTUAL_BRIDGE_PILOT_APPLY_CHECK] line={line}, block_id={block_id}, "
                    f"gain={bool(recon_gain)}, integrity_ok={bool(pair_integrity_ok and virtual_pilot_limit_ok)}, "
                    f"apply={bool(apply_ok)}, reject_reason={apply_reject_reason}"
                )
            conservative_apply = False
            conservative_reject_reason = ""
            if not apply_ok and bool(recon_gain):
                diag["conservative_apply_attempt_count"] += 1
                template_ok = bool(pair_integrity_ok)
                hard_violation_delta = 0 if template_ok else 1
                conservative_apply = (
                    int(valid_after) >= int(valid_before)
                    and (
                        int(underfilled_after) < int(underfilled_before)
                        or int(scheduled_orders_after) > int(scheduled_orders_before)
                    )
                    and bool(pair_integrity_ok)
                    and template_ok
                    and int(hard_violation_delta) <= 0
                )
                if not conservative_apply:
                    diag["conservative_apply_reject_count"] += 1
                    conservative_reject_reason = "PAIR_INTEGRITY_FAIL" if not pair_integrity_ok else "NO_CONSERVATIVE_GAIN"
                else:
                    diag["conservative_apply_success_count"] += 1
                    apply_ok = True
                    apply_reject_reason = ""
                print(
                    f"[APS][UNDERFILLED_RECON_CONSERVATIVE_APPLY_CHECK] line={line}, block_id={block_id}, "
                    f"gain={bool(recon_gain)}, integrity_ok={bool(pair_integrity_ok)}, "
                    f"template_ok={bool(template_ok)}, hard_violation_delta={int(hard_violation_delta)}, "
                    f"apply={bool(conservative_apply)}, reject_reason={conservative_reject_reason}"
                )
                if conservative_apply:
                    print(
                        f"[APS][UNDERFILLED_RECON_CONSERVATIVE_APPLY] line={line}, block_id={block_id}, "
                        f"applied=True, salvaged_segments={len(valid_out)}, "
                        f"salvaged_orders={int(block_diag.get('salvaged_orders', 0))}"
                    )
            if not apply_ok:
                diag["underfilled_reconstruction_apply_reject_count"] += 1
                if mode == "VIRTUAL_PILOT":
                    diag["virtual_pilot_reject_count"] += 1
                    _record_virtual_pilot_fail_stage(
                        diag,
                        line=str(line),
                        block_id=int(block_id),
                        candidate_id=str(block_diag.get("candidate_id", "")),
                        fail_stage="VIRTUAL_APPLY_GATE_REJECT",
                        reason=str(apply_reject_reason or "VIRTUAL_APPLY_GATE_REJECT"),
                        details={
                            "gain": bool(recon_gain),
                            "pair_integrity_ok": bool(pair_integrity_ok),
                            "virtual_pilot_limit_ok": bool(virtual_pilot_limit_ok),
                        },
                    )
                    _record_virtual_post_spec_fail(
                        diag,
                        line=str(line),
                        block_id=int(block_id),
                        candidate_id=str(block_diag.get("candidate_id", "")),
                        family=str(block_diag.get("family", "")),
                        fail_stage="VIRTUAL_APPLY_GATE_REJECT",
                        details={
                            "gain": bool(recon_gain),
                            "pair_integrity_ok": bool(pair_integrity_ok),
                            "virtual_pilot_limit_ok": bool(virtual_pilot_limit_ok),
                            "apply_reject_reason": str(apply_reject_reason or ""),
                        },
                    )
                    _record_virtual_pilot_execution_stage(
                        diag,
                        line=str(line),
                        block_id=int(block_id),
                        candidate_id=str(block_diag.get("candidate_id", "")),
                        family=str(block_diag.get("family", "")),
                        stage="FAIL",
                        details={"reason": str(apply_reject_reason or "VIRTUAL_APPLY_GATE_REJECT")},
                    )
                if recon_gain:
                    diag["underfilled_reconstruction_improvement_recorded_count"] += 1
                final_decision = "IMPROVEMENT_RECORDED_BUT_NOT_APPLIED" if recon_gain else "NO_GAIN"
                if mode != "VIRTUAL_PILOT":
                    _update_bridgeability_census_final_decision(
                        bridgeability_census,
                        line=str(line),
                        block_id=int(block_id),
                        final_decision=final_decision,
                    )
                print(
                    f"[APS][UNDERFILLED_RECON_FINAL_DECISION] line={line}, block_id={block_id}, "
                    f"gain={bool(recon_gain)}, final_decision={final_decision}, "
                    f"reason={apply_reject_reason}"
                )
                remaining_under.extend(block)
                i += best_size
                continue
            diag["underfilled_reconstruction_improvement_applied_count"] += 1
            if mode == "VIRTUAL_PILOT":
                diag["virtual_pilot_apply_count"] += 1
                diag["virtual_pilot_apply_success_count"] += 1
                _record_virtual_pilot_execution_stage(
                    diag,
                    line=str(line),
                    block_id=int(block_id),
                    candidate_id=str(block_diag.get("candidate_id", "")),
                    family=str(block_diag.get("family", "")),
                    stage="SUCCESS",
                    details={"reason": "IMPROVEMENT_APPLIED"},
                )
            if mode != "VIRTUAL_PILOT":
                _update_bridgeability_census_final_decision(
                    bridgeability_census,
                    line=str(line),
                    block_id=int(block_id),
                    final_decision="IMPROVEMENT_APPLIED",
                )
            print(
                f"[APS][UNDERFILLED_RECON_FINAL_DECISION] line={line}, block_id={block_id}, "
                f"gain={bool(recon_gain)}, final_decision=IMPROVEMENT_APPLIED, reason=GAIN_ACCEPTED"
            )
            print(
                f"[APS][UNDERFILLED_RECON_APPLY] line={line}, block_id={block_id}, "
                f"applied=True, salvaged_segments={len(valid_out)}, "
                f"salvaged_orders={int(block_diag.get('salvaged_orders', 0))}"
            )
            remaining_under = remaining_after_apply
            max_local_id += len(valid_out) + len(under_out)
            new_valid.extend(valid_out)
            remaining_under.extend(under_out)
            diag["underfilled_reconstruction_success"] += 1
            diag["underfilled_reconstruction_segments_salvaged"] += len(valid_out)
            diag["underfilled_reconstruction_orders_salvaged"] += int(block_diag.get("salvaged_orders", 0))
            if mode == "REAL":
                diag["repair_only_real_bridge_success"] += 1
                diag["repair_only_real_bridge_used_segments"] += len(valid_out)
                diag["repair_only_real_bridge_used_orders"] += int(block_diag.get("salvaged_orders", 0))
                print(
                    f"[APS][REPAIR_BRIDGE_HIT] type=REAL, line={line}, block_id={block_id}, "
                    f"used_segments={len(valid_out)}, salvaged_orders={int(block_diag.get('salvaged_orders', 0))}, "
                    f"direct_only_feasible=False"
                )
                print(
                    f"[APS][REPAIR_BRIDGE] type=REAL, line={line}, block={i}-{i + best_size - 1}, "
                    f"used={int(block_diag.get('real_bridge_used', 0))}, direct_only_feasible=False"
                )
            print(
                f"[APS][UNDERFILLED_RECON_GAIN] line={line}, block_id={block_id}, "
                f"valid_before={len(valid_segments)}, valid_after={len(new_valid)}, "
                f"underfilled_before={len(underfilled_segments)}, underfilled_after={len(remaining_under)}, "
                f"salvaged_orders={int(block_diag.get('salvaged_orders', 0))}"
            )
            print(
                f"[APS][UNDERFILLED_RECON] line={line}, block_size={best_size}, "
                f"old_underfilled={len(block)}, new_valid={len(valid_out)}, "
                f"residual_underfilled={len(under_out)}"
            )
            diag["reconstruction_no_gain"] = False
            # Reset no-gain streak on improvement
            no_gain_streak = 0
            i += best_size
            # Track improved block
            diag["cutter_blocks_improved"] = diag.get("cutter_blocks_improved", 0) + 1
        else:
            # No improvement: increment streak
            no_gain_streak += 1
            if no_gain_streak > diag.get("cutter_no_gain_streak_max", 0):
                diag["cutter_no_gain_streak_max"] = no_gain_streak
            # Early stop if no-gain streak limit reached
            if no_gain_streak >= early_stop_no_gain_limit:
                reconstruction_early_stopped = True
                print(
                    f"[APS][RECON_EARLY_STOP] line={line}, block_id={block_id}, "
                    f"reason=NO_GAIN_STREAK({no_gain_streak}>={early_stop_no_gain_limit}), "
                    f"remaining_under={len(line_under) - i}"
                )
                # Count remaining blocks as skipped by precheck
                diag["cutter_blocks_skipped_by_precheck"] = (
                    diag.get("cutter_blocks_skipped_by_precheck", 0)
                    + len(line_under) - i
                )
                break

    diag["underfilled_reconstruction_valid_delta"] = int(len(new_valid) - len(valid_segments))
    diag["underfilled_reconstruction_underfilled_delta"] = int(len(underfilled_segments) - len(remaining_under))
    diag["underfilled_reconstruction_valid_after"] = int(len(new_valid))
    diag["underfilled_reconstruction_underfilled_after"] = int(len(remaining_under))
    diag["underfilled_reconstruction_seconds"] = round(perf_counter() - t0, 6)
    diag["reconstruction_seconds"] = diag["underfilled_reconstruction_seconds"]
    census_dict = bridgeability_census.to_dict()
    suggestion = _suggest_next_phase_from_census(census_dict)
    diag["bridgeability_census"] = census_dict
    diag["bridgeability_census_items"] = list(census_dict.get("items", []))
    diag["bridgeability_route_suggestion"] = str(suggestion.get("suggestion", "CONTINUE_REAL_BRIDGE_ONLY"))
    diag["virtual_pilot_selected_by_bucket_count"] = dict(virtual_pilot_selected_by_bucket)
    diag["virtual_pilot_selected_by_family_count"] = dict(virtual_pilot_selected_by_family)
    diag["virtual_pilot_scheduler_selected_blocks"] = list(virtual_pilot_scheduler_selected_blocks)
    diag["virtual_pilot_scheduler_skipped_due_to_limit"] = list(virtual_pilot_scheduler_skipped_due_to_limit)
    family_audit: dict[str, dict] = {}
    stage_by_family = dict(diag.get("virtual_pilot_execution_stage_by_family", {}) or {})
    for family in sorted(set(list(dict(virtual_pilot_selected_by_family).keys()) + list(stage_by_family.keys()) + ["WIDTH_GROUP", "THICKNESS", "OTHER"])):
        fam_counts = dict(stage_by_family.get(family, {}) or {})
        selected_count = int(dict(virtual_pilot_selected_by_family).get(family, 0) or 0)
        dedup_kept_count = int(fam_counts.get("DEDUP_KEPT", 0) or 0)
        attempt_started_count = int(fam_counts.get("ATTEMPT_STARTED", 0) or 0)
        if selected_count > 0 and dedup_kept_count <= 0:
            missing_reason = "ALL_DEDUP_SKIPPED"
        elif dedup_kept_count > 0 and attempt_started_count <= 0:
            missing_reason = "INTERNAL_FLOW_BROKEN"
        else:
            missing_reason = ""
        family_audit[family] = {
            "selected_count": selected_count,
            "dedup_kept_count": dedup_kept_count,
            "attempt_started_count": attempt_started_count,
            "missing_execution_reason": missing_reason,
        }
        print(
            f"[APS][VIRTUAL_BRIDGE_FAMILY_EXECUTION_AUDIT] family={family}, "
            f"selected_count={selected_count}, dedup_kept_count={dedup_kept_count}, "
            f"attempt_started_count={attempt_started_count}, "
            f"missing_execution_reason={missing_reason or 'OK'}"
        )
        if family == "WIDTH_GROUP":
            print(
                f"[APS][VIRTUAL_BRIDGE_FAMILY_GUARANTEE] family=WIDTH_GROUP, "
                f"guaranteed_attempt={bool(diag.get('virtual_pilot_width_group_guarantee_attempted', False))}, "
                f"reason={'WIDTH_GROUP_ATTEMPT_PATH_ENTERED' if attempt_started_count > 0 else (missing_reason or 'NO_WIDTH_GROUP_SELECTED')}"
            )
    diag["virtual_pilot_family_execution_audit"] = family_audit
    if int(diag.get("repair_only_real_bridge_attempts", 0) or 0) <= 0:
        diag["repair_only_real_bridge_not_entered_reason"] = "DIRECT_ONLY_OR_NO_BRIDGE_STAGE"
    print(
        f"[APS][VIRTUAL_BRIDGE_PILOT_SCHEDULER] total_budget={int(virtual_pilot_max_blocks)}, "
        f"bucket_quota={dict(virtual_pilot_family_quota)}, "
        f"selected_by_bucket={dict(virtual_pilot_selected_by_bucket)}, "
        f"selected_blocks={list(virtual_pilot_scheduler_selected_blocks)}, "
        f"skipped_due_to_limit={list(virtual_pilot_scheduler_skipped_due_to_limit)}"
    )
    print(
        f"[APS][BRIDGEABILITY_CENSUS] total_blocks={int(census_dict.get('total_blocks', 0) or 0)}, "
        f"has_endpoint_edge_count={int(census_dict.get('has_endpoint_edge_count', 0) or 0)}, "
        f"template_graph_no_edge_count={int(census_dict.get('template_graph_no_edge_count', 0) or 0)}, "
        f"rule_prefilter_all_fail_count={int(census_dict.get('rule_prefilter_all_fail_count', 0) or 0)}, "
        f"band_too_narrow_count={int(census_dict.get('band_too_narrow_count', 0) or 0)}, "
        f"prefilter_rejected_block_count={int(census_dict.get('prefilter_rejected_block_count', 0) or 0)}, "
        f"candidate_pool_empty_count={int(census_dict.get('candidate_pool_empty_count', 0) or 0)}, "
        f"candidate_pool_nonempty_count={int(census_dict.get('candidate_pool_nonempty_count', 0) or 0)}, "
        f"ton_rescue_entered_count={int(census_dict.get('ton_rescue_entered_count', 0) or 0)}, "
        f"ton_rescue_success_count={int(census_dict.get('ton_rescue_success_count', 0) or 0)}, "
        f"improvement_recorded_count={int(census_dict.get('improvement_recorded_count', 0) or 0)}, "
        f"improvement_applied_count={int(census_dict.get('improvement_applied_count', 0) or 0)}, "
        f"thickness_fail_dominant_count={int(census_dict.get('thickness_fail_dominant_count', 0) or 0)}, "
        f"width_fail_dominant_count={int(census_dict.get('width_fail_dominant_count', 0) or 0)}, "
        f"temp_fail_dominant_count={int(census_dict.get('temp_fail_dominant_count', 0) or 0)}, "
        f"group_fail_dominant_count={int(census_dict.get('group_fail_dominant_count', 0) or 0)}, "
        f"multi_fail_dominant_count={int(census_dict.get('multi_fail_dominant_count', 0) or 0)}"
    )
    print(
        f"[APS][BRIDGEABILITY_ROUTE_SUGGESTION] suggestion={diag['bridgeability_route_suggestion']}, "
        f"reasons={list(suggestion.get('reasons', []))}"
    )
    print(
        f"[APS][RECON_GAIN] valid_before={len(valid_segments)}, valid_after={len(new_valid)}, "
        f"underfilled_before={len(underfilled_segments)}, underfilled_after={len(remaining_under)}"
    )
    print(
        f"[APS][RECON_TIMING] reconstruction={float(diag['reconstruction_seconds']):.3f}, "
        f"repair_bridge_real={float(diag['repair_bridge_real_seconds']):.3f}"
    )
    print(
        f"[APS][VIRTUAL_BRIDGE_PILOT_SUMMARY] "
        f"scheduler_budget={int(diag.get('virtual_pilot_scheduler_budget', 0) or 0)}, "
        f"selected_by_bucket_count={dict(diag.get('virtual_pilot_selected_by_bucket_count', {}) or {})}, "
        f"dedup_group_count={int(diag.get('virtual_pilot_dedup_group_count', 0) or 0)}, "
        f"duplicate_candidate_skipped_count={int(diag.get('virtual_pilot_duplicate_candidate_skipped_count', 0) or 0)}, "
        f"selected_unique_pilot_key_count={int(diag.get('virtual_pilot_selected_unique_pilot_key_count', 0) or 0)}, "
        f"selected_by_family_count={dict(diag.get('virtual_pilot_selected_by_family_count', {}) or {})}, "
        f"family_prefilter_fail_count={int(diag.get('virtual_pilot_family_prefilter_fail_count', 0) or 0)}, "
        f"width_group_family_attempt_count={int(diag.get('virtual_pilot_width_group_family_attempt_count', 0) or 0)}, "
        f"thickness_family_attempt_count={int(diag.get('virtual_pilot_thickness_family_attempt_count', 0) or 0)}, "
        f"structural_eligible_block_count={int(diag.get('virtual_pilot_structural_eligible_block_count', 0) or 0)}, "
        f"runtime_enabled_block_count={int(diag.get('virtual_pilot_runtime_enabled_block_count', 0) or 0)}, "
        f"final_eligible_block_count={int(diag.get('virtual_pilot_final_eligible_block_count', 0) or 0)}, "
        f"eligible_block_count={int(diag.get('virtual_pilot_eligible_block_count', 0) or 0)}, "
        f"selected_block_count={int(diag.get('virtual_pilot_selected_block_count', 0) or 0)}, "
        f"attempt_count={int(diag.get('virtual_pilot_attempt_count', 0) or 0)}, "
        f"success_count={int(diag.get('virtual_pilot_success_count', 0) or 0)}, "
        f"apply_count={int(diag.get('virtual_pilot_apply_count', 0) or 0)}, "
        f"reject_count={int(diag.get('virtual_pilot_reject_count', 0) or 0)}, "
        f"skipped_count={int(diag.get('virtual_pilot_skipped_block_count', 0) or 0)}, "
        f"skipped_due_to_disabled_count={int(diag.get('virtual_pilot_skipped_due_to_disabled_count', 0) or 0)}, "
        f"skipped_due_to_limit_count={int(diag.get('virtual_pilot_skipped_due_to_limit_count', 0) or 0)}, "
        f"skipped_due_to_no_pilotable_candidate_count={int(diag.get('virtual_pilot_skipped_due_to_no_pilotable_candidate_count', 0) or 0)}, "
        f"small_block_soft_penalty_count={int(diag.get('virtual_pilot_small_block_soft_penalty_count', 0) or 0)}, "
        f"spec_enum_total={int(diag.get('virtual_pilot_spec_enum_total', 0) or 0)}, "
        f"spec_enum_both_valid_count={int(diag.get('virtual_pilot_spec_enum_both_valid_count', 0) or 0)}, "
        f"ton_fill_attempt_count={int(diag.get('virtual_pilot_ton_fill_attempt_count', 0) or 0)}, "
        f"ton_fill_success_count={int(diag.get('virtual_pilot_ton_fill_success_count', 0) or 0)}, "
        f"selected_candidate_count={int(diag.get('virtual_pilot_selected_candidate_count', 0) or 0)}, "
        f"dedup_kept_count={int(diag.get('virtual_pilot_dedup_kept_count', 0) or 0)}, "
        f"dedup_skipped_count={int(diag.get('virtual_pilot_dedup_skipped_count', 0) or 0)}, "
        f"attempt_started_count={int(diag.get('virtual_pilot_attempt_started_count', 0) or 0)}, "
        f"spec_enum_done_count={int(diag.get('virtual_pilot_spec_enum_done_count', 0) or 0)}, "
        f"recut_entered_count={int(diag.get('virtual_pilot_recut_entered_count', 0) or 0)}, "
        f"segment_valid_count={int(diag.get('virtual_pilot_segment_valid_count', 0) or 0)}, "
        f"ton_fill_entered_count={int(diag.get('virtual_pilot_ton_fill_entered_count', 0) or 0)}, "
        f"apply_check_entered_count={int(diag.get('virtual_pilot_apply_check_entered_count', 0) or 0)}, "
        f"apply_success_count={int(diag.get('virtual_pilot_apply_success_count', 0) or 0)}, "
        f"post_spec_fail_stage_count={dict(diag.get('virtual_pilot_post_spec_fail_stage_count', {}) or {})}, "
        f"fail_stage_count={dict(diag.get('virtual_pilot_fail_stage_count', {}) or {})}, "
        f"reject_by_reason={dict(diag.get('virtual_pilot_reject_by_reason_count', {}) or {})}"
    )
    print(
        f"[APS][UNDERFILLED_RECON_SUMMARY] attempts={int(diag.get('underfilled_reconstruction_attempts', 0))}, "
        f"success={int(diag.get('underfilled_reconstruction_success', 0))}, "
        f"blocks_tested={int(diag.get('underfilled_reconstruction_blocks_tested', 0))}, "
        f"valid_delta={int(diag.get('underfilled_reconstruction_valid_delta', 0))}, "
        f"underfilled_delta={int(diag.get('underfilled_reconstruction_underfilled_delta', 0))}, "
        f"salvaged_segments={int(diag.get('underfilled_reconstruction_segments_salvaged', 0))}, "
        f"salvaged_orders={int(diag.get('underfilled_reconstruction_orders_salvaged', 0))}, "
        # ---- Cutter optimization stats ----
        f"cutter_blocks_touched={int(diag.get('cutter_blocks_touched', 0))}, "
        f"cutter_blocks_improved={int(diag.get('cutter_blocks_improved', 0))}, "
        f"cutter_blocks_skipped_by_precheck={int(diag.get('cutter_blocks_skipped_by_precheck', 0))}, "
        f"cutter_blocks_skipped_by_no_gain_set={int(diag.get('cutter_blocks_skipped_by_no_gain_set', 0))}, "
        f"cutter_no_gain_streak_max={int(diag.get('cutter_no_gain_streak_max', 0))}, "
        f"reconstruction_early_stopped={bool(diag.get('reconstruction_stage_early_stopped', False))}, "
        f"ton_rescue_extract_success={int(diag.get('repair_bridge_ton_rescue_extract_success', 0))}, "
        f"ton_rescue_partial_success={int(diag.get('repair_bridge_ton_rescue_partial_success', 0))}, "
        f"ton_rescue_pair_fail_width={int(diag.get('repair_bridge_ton_rescue_pair_fail_width', 0))}, "
        f"ton_rescue_pair_fail_thickness={int(diag.get('repair_bridge_ton_rescue_pair_fail_thickness', 0))}, "
        f"ton_rescue_pair_fail_temp={int(diag.get('repair_bridge_ton_rescue_pair_fail_temp', 0))}, "
        f"ton_rescue_pair_fail_group={int(diag.get('repair_bridge_ton_rescue_pair_fail_group', 0))}, "
        f"ton_rescue_pair_fail_context={int(diag.get('repair_bridge_ton_rescue_pair_fail_context', 0))}, "
        f"ton_rescue_pair_fail_template={int(diag.get('repair_bridge_ton_rescue_pair_fail_template', 0))}, "
        f"ton_rescue_pair_fail_multi={int(diag.get('repair_bridge_ton_rescue_pair_fail_multi', 0))}, "
        f"ton_rescue_pair_fail_unknown={int(diag.get('repair_bridge_ton_rescue_pair_fail_unknown', 0))}, "
        f"prefilter_reject_count={int(diag.get('repair_bridge_prefilter_reject_count', 0))}, "
        f"endpoint_early_stop_count={int(diag.get('repair_bridge_endpoint_early_stop_count', 0))}, "
        f"local_band_retry_count={int(diag.get('repair_bridge_local_band_retry_count', 0))}, "
        f"improvement_recorded_count={int(diag.get('underfilled_reconstruction_improvement_recorded_count', 0))}, "
        f"improvement_applied_count={int(diag.get('underfilled_reconstruction_improvement_applied_count', 0))}, "
        f"apply_reject_count={int(diag.get('underfilled_reconstruction_apply_reject_count', 0))}, "
        f"bridgeability_route_suggestion={str(diag.get('bridgeability_route_suggestion', ''))}, "
        # ---- Legacy virtual pilot: 降级为单一布尔字段 ----
        f"virtual_pilot_skipped_due_to_disabled={bool(diag.get('virtual_pilot_skipped_due_to_disabled', True))}, "
        f"conservative_apply_attempt_count={int(diag.get('conservative_apply_attempt_count', 0) or 0)}, "
        f"conservative_apply_success_count={int(diag.get('conservative_apply_success_count', 0) or 0)}, "
        f"conservative_apply_reject_count={int(diag.get('conservative_apply_reject_count', 0) or 0)}"
    )
    diagnostics.update(diag)
    return new_valid, remaining_under, diag


def _validate_fill_multiplicity(
    tail_seg: CampaignSegment,
    new_tail_seg: CampaignSegment,
    inserted_oid: str,
) -> Tuple[bool, dict]:
    """
    Validate order multiplicity for TAIL_FILL_FROM_DROPPED.

    For FILL, the new tail contains ALL orders from the original tail PLUS
    exactly one inserted dropped order.  The check is:
        after_counter == before_counter + Counter([inserted_oid])

    Returns:
        (is_valid, details_dict)
    """
    before_counter = Counter(tail_seg.order_ids)
    after_counter = Counter(new_tail_seg.order_ids)

    before_count = sum(before_counter.values())
    # Expected: before + 1 (the inserted dropped order)
    expected_after_count = before_count + 1
    actual_after_count = sum(after_counter.values())

    # The inserted order must appear EXACTLY once in after
    inserted_count_in_after = after_counter.get(inserted_oid, 0)
    inserted_count_in_before = before_counter.get(inserted_oid, 0)

    is_valid = (
        actual_after_count == expected_after_count
        and inserted_count_in_after == 1
        and inserted_count_in_before == 0  # was NOT in the tail before
    )

    # All original tail orders must still be present (preserved)
    original_preserved = all(
        after_counter.get(oid, 0) == 1
        for oid in tail_seg.order_ids
    )

    details = {
        "before_count": before_count,
        "expected_after_count": expected_after_count,
        "actual_after_count": actual_after_count,
        "inserted_order_id": inserted_oid,
        "inserted_count_in_after": inserted_count_in_after,
        "inserted_count_in_before": inserted_count_in_before,
        "original_tail_preserved": original_preserved,
        "multiplicity_match": is_valid,
    }

    if not is_valid:
        reason = (
            f"FILL_MULTIPLICITY_ERROR: "
            f"inserted={inserted_oid}, "
            f"before={before_count}, "
            f"expected_after={expected_after_count}, "
            f"actual_after={actual_after_count}, "
            f"inserted_in_after={inserted_count_in_after}"
        )
        return False, details

    return True, details


def _check_global_order_uniqueness(
    segments: List[CampaignSegment],
) -> Tuple[int, List[dict]]:
    """
    Check that every order_id appears in at most one segment across ALL lines.

    Returns:
        (duplicate_count, duplicate_examples)
    """
    seen: Dict[str, dict] = {}
    duplicates: List[dict] = []

    for seg in segments:
        for oid in seg.order_ids:
            if oid in seen:
                duplicates.append({
                    "order_id": oid,
                    "segment_a_local_id": seen[oid]["local_id"],
                    "segment_a_line": seen[oid]["line"],
                    "segment_b_local_id": seg.campaign_local_id,
                    "segment_b_line": seg.line,
                })
            else:
                seen[oid] = {"line": seg.line, "local_id": seg.campaign_local_id}

    return len(duplicates), duplicates


# ---------------------------------------------------------------------------
# Rebalance orchestrator (window-based, no duplicates)
# ---------------------------------------------------------------------------

def _rebalance_underfilled_segments(
    segments: List[CampaignSegment],
    underfilled_segments: List[CampaignSegment],
    order_tons: Dict[str, float],
    cfg: PlannerConfig,
    tpl_df: pd.DataFrame | None = None,
    orders_df: pd.DataFrame | None = None,
    placed_oids: set | None = None,
) -> Tuple[List[CampaignSegment], List[CampaignSegment], dict]:
    """
    Orchestrator: apply multi-strategy rebalance to all underfilled segments
    within their true adjacent windows (by original line order).

    Key invariant: each order_id appears in at most ONE segment after repair.

    Strategies (in NEW priority order):
        A. RECUT_TWO_SEGMENTS: recut immediately preceding + tail with exhaustive cut point search
        B. SHIFT_FROM_PREV: pull back K orders from the immediately preceding segment
        C. TAIL_FILL_FROM_DROPPED: fill underfilled tail from dropped candidates (small gap only)
        D. MERGE_WITH_PREV: merge underfilled tail into the immediately preceding segment (LAST RESORT)

    Rationale for priority:
        - RECUT is preferred: can keep both segments valid, preserves structure
        - SHIFT is second: simple boundary adjustment, keeps both segments valid often
        - FILL is third: for gaps up to tail_fill_gap_to_min_limit (220 tons by default), adds orders from dropped pool
        - MERGE is last resort: destroys segment boundary, reduces total campaign count

    Window definition: an underfilled segment at index i in line_ordered_segments
    has ONLY one possible donor — the segment at index i-1. No other segment
    may be used as donor.

    Returns:
        (repaired_segments, still_underfilled_segments, diagnostics)
    """
    if placed_oids is None:
        placed_oids = set()

    diag: dict = {
        "total_underfilled_before": len(underfilled_segments),
        "tail_repair_attempts_total": 0,
        "tail_repair_recut_success": 0,
        "tail_repair_shift_success": 0,
        "tail_repair_fill_success": 0,
        "tail_repair_merge_success": 0,
        "underfilled_segments_before_repair": len(underfilled_segments),
        "underfilled_segments_after_repair": 0,
        "underfilled_segments_near_min_count": 0,
        "total_repaired": 0,
        "underfilled_after": 0,
        "fail_reasons": [],
        "repair_log": [],
        # Window-based fields
        "rebalance_window_repairs_attempted": 0,
        "rebalance_window_repairs_success": 0,
        "duplicate_orders_after_rebalance": 0,
        "duplicate_order_examples": [],
        "order_multiplicity_preserve_failures": 0,
        # Partial-success in-place replacement counters
        "partial_shift_replaced_in_place_count": 0,
        "partial_recut_replaced_in_place_count": 0,
        "final_underfilled_from_ordered_line_count": 0,
        # RECUT diagnostics
        "recut_candidate_points_tested": 0,
        "recut_valid_points_found": 0,
        "recut_best_gap_to_target": 0.0,
        # SHIFT diagnostics
        "shift_k_tested_max": 0,
        "shift_valid_k_count": 0,
        "shift_best_tail_gap_to_target": 0.0,
        # MERGE as last resort
        "merge_attempted_as_last_resort_count": 0,
        # Failure reason categories
        "fail_reason_PREV_WOULD_UNDERFILL": 0,
        "fail_reason_NO_VALID_RECUT_POINT": 0,
        "fail_reason_MERGED_EXCEEDS_MAX": 0,
        "fail_reason_TEMPLATE_PAIR_INVALID_AFTER_REPAIR": 0,
        "fail_reason_NO_DROPPED_CANDIDATE_FOR_FILL": 0,
        "fail_reason_other": 0,
        # Tail repair attempt counters
        "tail_repair_recut_attempts": 0,
        "tail_repair_shift_attempts": 0,
        "tail_repair_fill_attempts": 0,
        "tail_repair_merge_attempts": 0,
        # ---- Tail Fill From Dropped diagnostics ----
        "tail_fill_candidates_considered": 0,
        "tail_fill_partial_progress_count": 0,
        "tail_fill_best_gap_reduction": 0.0,
        "tail_fill_multiplicity_ok_count": 0,
        "tail_fill_multiplicity_fail_count": 0,
        "tail_fill_second_pass_attempts": 0,
        "tail_fill_second_pass_success": 0,
        "tail_fill_total_inserted_orders": 0,
        "tail_fill_stop_after_partial_count": 0,
        "tail_fill_stop_after_full_count": 0,
        "merge_skipped_due_to_fill_progress_count": 0,
        "tail_fill_candidates_rejected_already_in_segments": 0,
        "tail_fill_candidates_rejected_already_in_underfilled": 0,
        # Tail repair budget diagnostics
        "tail_repair_windows_attempted_total": 0,
        "tail_repair_windows_attempted_by_line": {"big_roll": 0, "small_roll": 0},
        "tail_repair_windows_skipped_by_budget": 0,
        "tail_repair_windows_skipped_by_gap": 0,
        "recut_disabled_after_failures": False,
        "recut_consecutive_failures": 0,
        "fill_candidates_tested_total": 0,
        # Recut failure de-weight tracking
        "recut_failures": 0,
        "recut_fail_streak": 0,
        # ---- Phase timing ----
        "tail_repair_seconds": 0.0,
        "recut_seconds": 0.0,
        "shift_seconds": 0.0,
        "fill_seconds": 0.0,
        "merge_seconds": 0.0,
    }

    # ---- Phase timers ----
    t0_rebalance = perf_counter()
    t0_recut = 0.0
    t0_shift = 0.0
    t0_fill = 0.0

    # ---- Track dropped orders consumed by fill (prevent reuse within same repair pass) ----
    fill_consumed_oids: set = set()
    t0_merge = 0.0

# ---- Tail Repair Budget: read configuration ----
    cfg_model = getattr(cfg, "model", None) if cfg else None
    gap_limit = float(
        getattr(cfg_model, "tail_repair_gap_to_min_limit", 220.0) if cfg_model else 220.0
    )
    max_per_line = int(
        getattr(cfg_model, "max_tail_repair_windows_per_line", 12) if cfg_model else 12
    )
    max_total = int(
        getattr(cfg_model, "max_tail_repair_windows_total", 24) if cfg_model else 24
    )
    windows_by_line: Dict[str, int] = {"big_roll": 0, "small_roll": 0}
    total_attempted = 0
    recut_fail_streak = 0
    recut_fail_threshold = 5
    recut_disabled_this_run = False

    if not underfilled_segments:
        diag["underfilled_after"] = 0
        return segments, [], diag

    # Count near-min underfilled segments for diagnostics
    ton_min = float(getattr(cfg.rule, "campaign_ton_min", 500.0) if cfg.rule else 500.0)
    diag["underfilled_segments_near_min_count"] = sum(
        1 for s in underfilled_segments if (ton_min - s.total_tons) <= 220.0
    )

    ton_target = float(getattr(cfg.rule, "campaign_ton_target", 1500.0) if cfg.rule else 1500.0)

    # Build per-line ordered list
    all_segs_by_line: Dict[str, List[CampaignSegment]] = {"big_roll": [], "small_roll": []}
    for seg in segments + underfilled_segments:
        if seg.line in all_segs_by_line:
            all_segs_by_line[seg.line].append(seg)
        else:
            all_segs_by_line[seg.line] = [seg]

    # Sort each line's segments by campaign_local_id (original cutting order)
    for line_key in all_segs_by_line:
        all_segs_by_line[line_key].sort(key=lambda s: s.campaign_local_id)

    # Build index map: (line, campaign_local_id) -> segment object
    # This lets us do O(1) lookup when updating
    seg_index_map: Dict[Tuple[str, int], CampaignSegment] = {}
    for line_key in all_segs_by_line:
        for seg in all_segs_by_line[line_key]:
            seg_index_map[(line_key, seg.campaign_local_id)] = seg

    # -------------------------------------------------------------------------
    # Step 2: Build set of underfilled segment identifiers for quick lookup
    # -------------------------------------------------------------------------
    underfilled_ids: set = set()
    for seg in underfilled_segments:
        underfilled_ids.add((seg.line, seg.campaign_local_id))

    # -------------------------------------------------------------------------
    # Step 3: Process underfilled segments IN LINE ORDER.
    # For each underfilled segment, find its TRUE predecessor in the sorted list.
    # Only the immediately preceding segment may be used as donor.
    # NOTE: we no longer maintain still_under in-loop.  unrepaired u_segs stay
    # in ordered_line as-is.  Final still_under is collected in Step 4 below.
    # -------------------------------------------------------------------------

    for u_seg in underfilled_segments:
        repaired = False
        line = u_seg.line
        ordered_line = all_segs_by_line.get(line, [])

        # Find the index of this underfilled segment in the ordered list
        try:
            u_idx = next(
                i for i, s in enumerate(ordered_line)
                if s.campaign_local_id == u_seg.campaign_local_id
            )
        except StopIteration:
            # Segment not found — shouldn't happen but guard against it
            diag["fail_reasons"].append({
                "line": line,
                "segment_id": u_seg.campaign_local_id,
                "method": "SHIFT_FROM_PREV",
                "reason": "segment_not_in_ordered_list",
            })
            # u_seg stays in ordered_line as invalid — picked up in Step 4
            continue

        # True predecessor: only the immediately preceding segment (index i-1)
        if u_idx == 0:
            # No predecessor on this line — cannot repair via SHIFT/MERGE/RECUT
            diag["fail_reasons"].append({
                "line": line,
                "segment_id": u_seg.campaign_local_id,
                "method": "SHIFT_FROM_PREV",
                "reason": "no_prev_segment",
            })
            # u_seg stays in ordered_line as invalid — picked up in Step 4
            continue

        prev_seg = ordered_line[u_idx - 1]

        # Sanity check: prev_seg must not also be underfilled (if it is,
        # it should have been repaired first; skip this underfilled seg)
        if (prev_seg.line, prev_seg.campaign_local_id) in underfilled_ids:
            # The predecessor is also underfilled — defer repair until prev is fixed
            diag["fail_reasons"].append({
                "line": line,
                "segment_id": u_seg.campaign_local_id,
                "method": "SHIFT_FROM_PREV",
                "reason": "prev_also_underfilled_defer",
            })
            # u_seg stays in ordered_line as invalid — picked up in Step 4
            continue

        diag["rebalance_window_repairs_attempted"] += 1
        diag["tail_repair_attempts_total"] += 1

        # ---- Budget check: enforce max windows limit ----
        gap_to_min = ton_min - u_seg.total_tons
        if gap_to_min > gap_limit:
            diag["tail_repair_windows_skipped_by_gap"] += 1
            diag["fail_reasons"].append({
                "line": line,
                "segment_id": u_seg.campaign_local_id,
                "method": "BUDGET_GAP_FILTER",
                "reason": f"gap_too_large_{gap_to_min:.0f}",
            })
            continue

        if total_attempted >= max_total:
            diag["tail_repair_windows_skipped_by_budget"] += 1
            diag["fail_reasons"].append({
                "line": line,
                "segment_id": u_seg.campaign_local_id,
                "method": "BUDGET_GAP_FILTER",
                "reason": "total_budget_exhausted",
            })
            continue

        if windows_by_line.get(line, 0) >= max_per_line:
            diag["tail_repair_windows_skipped_by_budget"] += 1
            diag["fail_reasons"].append({
                "line": line,
                "segment_id": u_seg.campaign_local_id,
                "method": "BUDGET_GAP_FILTER",
                "reason": "per_line_budget_exhausted",
            })
            continue

        # Budget consumed — count this window
        total_attempted += 1
        windows_by_line[line] = windows_by_line.get(line, 0) + 1
        diag["tail_repair_windows_attempted_total"] += 1

        print(
            f"[APS][TailRepairWindow] line={line}, "
            f"seg_prev={prev_seg.campaign_local_id}, "
            f"seg_tail={u_seg.campaign_local_id}, "
            f"prev_tons={prev_seg.total_tons:.0f}, tail_tons={u_seg.total_tons:.0f}"
        )

        # ---- Helper: record failure reason with categorization ----
        def _record_fail(method: str, reason: str):
            diag["fail_reasons"].append({
                "line": line,
                "segment_id": u_seg.campaign_local_id,
                "method": method,
                "reason": reason,
            })
            reason_key = f"fail_reason_{reason}"
            if reason_key in diag:
                diag[reason_key] += 1
            else:
                diag["fail_reason_other"] += 1

        # ---- Track which strategies have been tried (for MERGE as last resort) ----
        recut_tried = False
        shift_tried = False
        fill_tried = False

        # ---- Strategy A (PRIMARY): RECUT_TWO_SEGMENTS ----
        # Try smart cut point search to keep both segments valid
        # Skip recut if disabled by failure de-weight
        t0_recut = perf_counter()
        if recut_disabled_this_run:
            diag_c = {"method": "RECUT_TWO_SEGMENTS", "attempted": True, "success": False,
                      "failure_reason": "recut_disabled_after_failures",
                      "recut_candidate_points_tested": 0, "recut_valid_points_found": 0}
            ok_c, new_prev_c, new_tail_c = False, None, None
        else:
            ok_c, new_prev_c, new_tail_c, diag_c = _try_recut_two_segments(
                prev_seg, u_seg, order_tons, cfg, tpl_df, recut_disabled=recut_disabled_this_run
            )
        diag["recut_seconds"] = (diag.get("recut_seconds", 0.0) or 0.0) + (perf_counter() - t0_recut)
        recut_tried = True
        diag["tail_repair_recut_attempts"] += 1

        if ok_c and new_prev_c is not None and new_tail_c is not None:
            preserved, fail_reason, mult_details = _check_order_multiplicity_preserved(
                [prev_seg, u_seg],
                [new_prev_c, new_tail_c],
                "RECUT_TWO_SEGMENTS",
            )
            # Accumulate RECUT diagnostics
            diag["recut_candidate_points_tested"] += diag_c.get("recut_candidate_points_tested", 0)
            diag["recut_valid_points_found"] += diag_c.get("recut_valid_points_found", 0)
            if diag_c.get("recut_best_gap_to_target", 0) > 0:
                diag["recut_best_gap_to_target"] = min(
                    diag.get("recut_best_gap_to_target", float("inf")),
                    diag_c.get("recut_best_gap_to_target", float("inf"))
                )
            diag_c["multiplicity_check"] = "OK" if preserved else "FAIL"
            print(
                f"[APS][TailRepairWindow] method=RECUT_TWO_SEGMENTS, "
                f"multiplicity_check={'OK' if preserved else 'FAIL'}"
            )
            # Accumulate new-tier diagnostics
            diag["recut_partial_right_valid_count"] = (
                diag.get("recut_partial_right_valid_count", 0) +
                diag_c.get("recut_partial_right_valid_count", 0)
            )
            # Emit warning when only partial-valid recut found
            if not new_tail_c.is_valid and preserved:
                _ton_min = float(getattr(cfg.rule, "campaign_ton_min", 500.0) if cfg.rule else 500.0)
                sol_type = diag_c.get("recut_selected_solution_type", "UNKNOWN")
                print(
                    f"[APS][RECUT_WARNING] only partial-valid recut found, chaining fill, "
                    f"sol_type={sol_type}, "
                    f"right_tons={new_tail_c.total_tons:.0f}, "
                    f"gap_to_min={max(_ton_min - new_tail_c.total_tons, 0.0):.0f}"
                )
            if not preserved:
                diag["order_multiplicity_preserve_failures"] += 1
                _record_fail("RECUT_TWO_SEGMENTS", fail_reason)
            elif new_prev_c.is_valid and new_tail_c.is_valid:
                new_tail_c.campaign_local_id = prev_seg.campaign_local_id + 1
                ordered_line[u_idx - 1] = new_prev_c
                ordered_line[u_idx] = new_tail_c
                seg_index_map[(line, prev_seg.campaign_local_id)] = new_prev_c
                seg_index_map[(line, u_seg.campaign_local_id)] = new_tail_c
                diag["tail_repair_recut_success"] += 1
                diag["rebalance_window_repairs_success"] += 1
                diag["total_repaired"] += 1
                diag["repair_log"].append({
                    "method": "RECUT_TWO_SEGMENTS",
                    "segment_id": u_seg.campaign_local_id,
                    "cut_index": diag_c.get("cut_index", -1),
                    "multiplicity": "OK",
                })
                print(
                    f"[APS][TAIL_REPAIR_HIT] method=RECUT_TWO_SEGMENTS, line={line}, "
                    f"seg_prev={prev_seg.campaign_local_id}, seg_tail={u_seg.campaign_local_id}"
                )
                repaired = True
                recut_fail_streak = 0  # Reset on success
            elif new_prev_c.is_valid and not new_tail_c.is_valid:
                # Partial success: left is valid, right still underfilled.
                # Chain to FILL instead of stopping — do NOT set repaired=True.
                # This mirrors the SHIFT partial-success pattern.
                new_tail_c.campaign_local_id = u_seg.campaign_local_id
                ordered_line[u_idx - 1] = new_prev_c
                ordered_line[u_idx] = new_tail_c
                seg_index_map[(line, prev_seg.campaign_local_id)] = new_prev_c
                seg_index_map[(line, u_seg.campaign_local_id)] = new_tail_c
                u_seg = new_tail_c  # update ref so FILL operates on improved tail
                diag["tail_repair_recut_success"] += 1
                diag["partial_recut_replaced_in_place_count"] = (
                    diag.get("partial_recut_replaced_in_place_count", 0) + 1
                )
                diag["repair_log"].append({
                    "method": "RECUT_TWO_SEGMENTS",
                    "segment_id": u_seg.campaign_local_id,
                    "cut_index": diag_c.get("cut_index", -1),
                    "note": "partial_replaced_in_place_chain_to_fill",
                    "multiplicity": "OK",
                })
                gap_to_min = max(ton_min - new_tail_c.total_tons, 0.0)
                print(
                    f"[APS][RECUT_CHAIN_FILL] line={line}, "
                    f"seg_tail={u_seg.campaign_local_id}, "
                    f"right_tail_tons={new_tail_c.total_tons:.0f}, "
                    f"gap_to_min={gap_to_min:.0f}, "
                    f"sol_type={diag_c.get('recut_selected_solution_type', 'UNKNOWN')}"
                )
                # Partial success counts as improvement — reset fail streak so
                # recut stays available for subsequent windows.
                recut_fail_streak = 0
                # Skip streak increment; fall through to SHIFT and FILL on improved tail
            elif not preserved:
                _record_fail("RECUT_TWO_SEGMENTS", fail_reason)
                # Full failure — increment streak
                recut_fail_streak += 1
                if recut_fail_streak >= recut_fail_threshold:
                    recut_disabled_this_run = True
                    diag["recut_disabled_after_failures"] = True
                    print(
                        f"[APS][TAIL_REPAIR_BUDGET] recut_disabled_after_failures=True, "
                        f"consecutive_failures={recut_fail_streak}"
                    )
            # Note: partial success (new_prev valid, new_tail invalid) is handled above.
            # It chains to SHIFT and FILL without setting repaired=True.

        # ---- Strategy B (SECONDARY): SHIFT_FROM_PREV ----
        # Try pulling K orders from prev to tail (K up to 8)
        t0_shift = perf_counter()
        ok_a, new_prev_a, new_tail_a, diag_a = _try_shift_from_prev(
            prev_seg, u_seg, order_tons, cfg, tpl_df
        )
        diag["shift_seconds"] = (diag.get("shift_seconds", 0.0) or 0.0) + (perf_counter() - t0_shift)
        shift_tried = True
        diag["tail_repair_shift_attempts"] += 1

        if ok_a and new_prev_a is not None and new_tail_a is not None:
            preserved, fail_reason, mult_details = _check_order_multiplicity_preserved(
                [prev_seg, u_seg],
                [new_prev_a, new_tail_a],
                "SHIFT_FROM_PREV",
            )
            # Accumulate SHIFT diagnostics
            diag["shift_k_tested_max"] = max(diag.get("shift_k_tested_max", 0), diag_a.get("shift_k_tested_max", 0))
            diag["shift_valid_k_count"] += diag_a.get("shift_valid_k_count", 0)
            if diag_a.get("shift_best_tail_gap_to_target", 0) > 0:
                diag["shift_best_tail_gap_to_target"] = min(
                    diag.get("shift_best_tail_gap_to_target", float("inf")),
                    diag_a.get("shift_best_tail_gap_to_target", float("inf"))
                )
            diag_a["multiplicity_check"] = "OK" if preserved else "FAIL"
            print(
                f"[APS][TailRepairWindow] method=SHIFT_FROM_PREV, "
                f"multiplicity_check={'OK' if preserved else 'FAIL'}"
            )
            if not preserved:
                diag["order_multiplicity_preserve_failures"] += 1
                _record_fail("SHIFT_FROM_PREV", fail_reason)
            elif new_tail_a.is_valid:
                # Both parts valid: window in-place replacement
                new_tail_a.campaign_local_id = prev_seg.campaign_local_id + 1
                ordered_line[u_idx - 1] = new_prev_a
                ordered_line[u_idx] = new_tail_a
                seg_index_map[(line, prev_seg.campaign_local_id)] = new_prev_a
                seg_index_map[(line, u_seg.campaign_local_id)] = new_tail_a
                diag["tail_repair_shift_success"] += 1
                diag["rebalance_window_repairs_success"] += 1
                diag["total_repaired"] += 1
                diag["repair_log"].append({
                    "method": "SHIFT_FROM_PREV",
                    "segment_id": u_seg.campaign_local_id,
                    "shift_k": diag_a.get("shift_k", 0),
                    "multiplicity": "OK",
                })
                print(
                    f"[APS][TAIL_REPAIR_HIT] method=SHIFT_FROM_PREV, line={line}, "
                    f"seg_prev={prev_seg.campaign_local_id}, seg_tail={u_seg.campaign_local_id}, "
                    f"shift_k={diag_a.get('shift_k', 0)}"
                )
                repaired = True
            else:
                # Partial success: prev now valid but tail still underfilled.
                # Update segments in-place but do NOT set repaired=True
                # — this lets FILL attempt to rescue the underfilled tail.
                new_tail_a.campaign_local_id = u_seg.campaign_local_id
                ordered_line[u_idx - 1] = new_prev_a
                ordered_line[u_idx] = new_tail_a
                seg_index_map[(line, prev_seg.campaign_local_id)] = new_prev_a
                seg_index_map[(line, u_seg.campaign_local_id)] = new_tail_a
                u_seg = new_tail_a  # update reference so FILL operates on improved tail
                diag["tail_repair_shift_success"] += 1
                diag["partial_shift_replaced_in_place_count"] = (
                    diag.get("partial_shift_replaced_in_place_count", 0) + 1
                )
                diag["repair_log"].append({
                    "method": "SHIFT_FROM_PREV",
                    "segment_id": u_seg.campaign_local_id,
                    "shift_k": diag_a.get("shift_k", 0),
                    "note": "partial_replaced_in_place_tail_still_underfilled",
                    "multiplicity": "OK",
                })
                print(
                    f"[APS][TailRepairWindow] partial_underfilled_replaced_in_place=True, "
                    f"method=SHIFT_FROM_PREV, line={line}, "
                    f"new_tail_tons={new_tail_a.total_tons:.0f}, "
                    f"new_tail_is_valid={new_tail_a.is_valid}, "
                    f"continuing_to_fill=True"
                )
        elif not repaired:
            _record_fail("SHIFT_FROM_PREV", diag_a.get("failure_reason", "unknown"))

        # ---- Strategy C (TERTIARY): TAIL_FILL_FROM_DROPPED (multi-pass) ----
        # Try fill when prev strategies failed OR produced partial success (tail still underfilled).
        # Also try fill even when RECUT/SHIFT succeeded (full success) — fill is optional here.
        # Supports up to `tail_fill_max_inserts_per_tail` consecutive fill passes per tail segment.
        # After 1st fill, if tail still underfilled and gap <= `tail_fill_second_pass_gap_limit`,
        # attempt a 2nd fill to push the near-min tail over campaign_ton_min.
        fill_tried = True
        t0_fill = perf_counter()

        # Read fill configuration
        max_inserts = int(
            getattr(cfg_model, "tail_fill_max_inserts_per_tail", 2) if cfg_model else 2
        )
        second_pass_gap_limit = float(
            getattr(cfg_model, "tail_fill_second_pass_gap_limit", 30.0) if cfg_model else 30.0
        )

        inserts_used = 0
        pass_num = 0
        # fill_progress_result: FILL_FULL_VALID | FILL_PARTIAL_PROGRESS | FILL_NO_PROGRESS | FILL_FAILED
        fill_progress_result = "FILL_NO_PROGRESS"

        # Work on a local copy of the current tail segment reference
        tail_work = u_seg

        while inserts_used < max_inserts:
            pass_num += 1

            # Exclude orders already consumed by previous fill passes in this window
            ok_d, new_tail_d, diag_d = _try_fill_from_dropped(
                tail_work, order_tons,
                orders_df if orders_df is not None else pd.DataFrame(),
                placed_oids | fill_consumed_oids,
                cfg, tpl_df,
            )

            diag["tail_repair_fill_attempts"] += 1
            diag["fill_candidates_tested_total"] += diag_d.get("fill_candidates_tested", 0)
            diag["tail_fill_candidates_considered"] = (
                diag.get("tail_fill_candidates_considered", 0) + diag_d.get("candidates_considered", 0)
            )

            if not (ok_d and new_tail_d is not None):
                # No candidate found — stop filling
                if inserts_used == 0:
                    _record_fail("TAIL_FILL_FROM_DROPPED", diag_d.get("failure_reason", "unknown"))
                    fill_progress_result = "FILL_FAILED"
                else:
                    # We made progress but then ran out of candidates — already recorded
                    pass
                break

            # ---- Multiplicity check ----
            inserted_oid = diag_d.get("candidate_order_id", "")
            fill_ok, fill_mult_details = _validate_fill_multiplicity(
                tail_work, new_tail_d, inserted_oid
            )
            diag_d["multiplicity_check"] = "OK" if fill_ok else "FAIL"
            diag_d["multiplicity_details"] = fill_mult_details
            diag["tail_fill_multiplicity_ok_count"] = (
                diag.get("tail_fill_multiplicity_ok_count", 0) + (1 if fill_ok else 0)
            )
            diag["tail_fill_multiplicity_fail_count"] = (
                diag.get("tail_fill_multiplicity_fail_count", 0) + (0 if fill_ok else 1)
            )
            print(
                f"[APS][TAIL_FILL_MULTIPLICITY] "
                f"pass={pass_num}, "
                f"inserted={inserted_oid}, "
                f"before={fill_mult_details.get('before_count', 0)}, "
                f"expected_after={fill_mult_details.get('expected_after_count', 0)}, "
                f"actual_after={fill_mult_details.get('actual_after_count', 0)}, "
                f"ok={fill_ok}"
            )

            if not fill_ok:
                diag["order_multiplicity_preserve_failures"] += 1
                _record_fail("TAIL_FILL_FROM_DROPPED", "FILL_MULTIPLICITY_ERROR")
                fill_progress_result = "FILL_FAILED"
                break

            # Record this insert
            inserts_used += 1
            diag["tail_fill_total_inserted_orders"] = (
                diag.get("tail_fill_total_inserted_orders", 0) + 1
            )
            fill_consumed_oids.add(inserted_oid)

            # ---- Apply the fill result to the window ----
            new_tail_d.campaign_local_id = u_seg.campaign_local_id
            ordered_line[u_idx] = new_tail_d
            seg_index_map[(line, u_seg.campaign_local_id)] = new_tail_d
            tail_work = new_tail_d
            u_seg = new_tail_d  # keep u_seg in sync for diagnostics

            gap_after = max(ton_min - new_tail_d.total_tons, 0.0)

            if new_tail_d.is_valid:
                # ---- A. Full success: tail reaches campaign_ton_min ----
                print(
                    f"[APS][RECUT_CHAIN_FILL] pass={pass_num}, line={line}, "
                    f"seg_tail={u_seg.campaign_local_id}, "
                    f"tail_tons={new_tail_d.total_tons:.0f}, "
                    f"gap_after={gap_after:.0f}, "
                    f"inserts_used={inserts_used}, result=FULL_SUCCESS"
                )
                print(
                    f"[APS][TAIL_FILL_SUCCESS] line={line}, "
                    f"seg_tail={u_seg.campaign_local_id}, "
                    f"final_tail_tons={new_tail_d.total_tons:.0f}, "
                    f"inserts_used={inserts_used}, "
                    f"pass_1_ok={pass_num >= 1}"
                )
                print(
                    f"[APS][TAIL_FILL_CONTROL] result=FILL_FULL_VALID, "
                    f"stop_after_fill=True, "
                    f"inserts_used={inserts_used}"
                )
                diag["tail_repair_fill_success"] += 1
                diag["tail_fill_stop_after_full_count"] = (
                    diag.get("tail_fill_stop_after_full_count", 0) + 1
                )
                diag["rebalance_window_repairs_success"] += 1
                diag["total_repaired"] += 1
                diag["repair_log"].append({
                    "method": "TAIL_FILL_FROM_DROPPED",
                    "segment_id": u_seg.campaign_local_id,
                    "fill_oid": inserted_oid,
                    "multiplicity": "OK",
                    "result": "full_success",
                    "pass_num": pass_num,
                    "inserts_used": inserts_used,
                })
                fill_progress_result = "FILL_FULL_VALID"
                repaired = True
                break

            # Tail still underfilled — check whether to attempt another pass
            elif pass_num == 1 and gap_after <= second_pass_gap_limit + 1e-6:
                # ---- Partial: gap is small enough to warrant a 2nd fill pass ----
                diag["tail_fill_second_pass_attempts"] = (
                    diag.get("tail_fill_second_pass_attempts", 0) + 1
                )
                diag["tail_repair_fill_success"] += 1
                diag["tail_fill_partial_progress_count"] = (
                    diag.get("tail_fill_partial_progress_count", 0) + 1
                )
                print(
                    f"[APS][RECUT_CHAIN_FILL] pass={pass_num}, line={line}, "
                    f"seg_tail={u_seg.campaign_local_id}, "
                    f"tail_tons={new_tail_d.total_tons:.0f}, "
                    f"gap_after={gap_after:.0f}, "
                    f"inserts_used={inserts_used}, result=PARTIAL_CONTINUE_TO_PASS2"
                )
                diag["repair_log"].append({
                    "method": "TAIL_FILL_FROM_DROPPED",
                    "segment_id": u_seg.campaign_local_id,
                    "fill_oid": inserted_oid,
                    "multiplicity": "OK",
                    "result": "partial_continue_to_pass2",
                    "pass_num": pass_num,
                    "gap_after": gap_after,
                })
                # fill_progress_result stays FILL_PARTIAL_PROGRESS; continue to 2nd pass
                fill_progress_result = "FILL_PARTIAL_PROGRESS"
                # Loop continues to 2nd pass
                continue

            else:
                # ---- B. Partial progress, gap too large for 2nd pass (or pass 2 exhausted) ----
                print(
                    f"[APS][RECUT_CHAIN_FILL] pass={pass_num}, line={line}, "
                    f"seg_tail={u_seg.campaign_local_id}, "
                    f"tail_tons={new_tail_d.total_tons:.0f}, "
                    f"gap_after={gap_after:.0f}, "
                    f"inserts_used={inserts_used}, result=PARTIAL_NO_MORE_PASS"
                )
                print(
                    f"[APS][TAIL_FILL_SUCCESS] line={line}, "
                    f"seg_tail={u_seg.campaign_local_id}, "
                    f"final_tail_tons={new_tail_d.total_tons:.0f}, "
                    f"inserts_used={inserts_used}, "
                    f"pass_1_ok={pass_num >= 1}, "
                    f"pass_2_ok={pass_num >= 2}, "
                    f"result=PARTIAL"
                )
                print(
                    f"[APS][TAIL_FILL_CONTROL] result=FILL_PARTIAL_PROGRESS, "
                    f"stop_after_fill=True, "
                    f"inserts_used={inserts_used}, "
                    f"gap_after={gap_after:.0f}"
                )
                diag["tail_repair_fill_success"] += 1
                diag["tail_fill_partial_progress_count"] = (
                    diag.get("tail_fill_partial_progress_count", 0) + 1
                )
                if pass_num >= 2:
                    diag["tail_fill_second_pass_success"] = (
                        diag.get("tail_fill_second_pass_success", 0) + 1
                    )
                diag["tail_fill_stop_after_partial_count"] = (
                    diag.get("tail_fill_stop_after_partial_count", 0) + 1
                )
                diag["repair_log"].append({
                    "method": "TAIL_FILL_FROM_DROPPED",
                    "segment_id": u_seg.campaign_local_id,
                    "fill_oid": inserted_oid,
                    "multiplicity": "OK",
                    "result": "partial_no_more_pass",
                    "pass_num": pass_num,
                    "inserts_used": inserts_used,
                    "gap_after": gap_after,
                })
                # ACCEPT partial progress — mark repaired and stop.
                # Do NOT let MERGE run on the stale window (prev_seg may be out of sync).
                fill_progress_result = "FILL_PARTIAL_PROGRESS"
                repaired = True
                break

        diag["fill_seconds"] = (diag.get("fill_seconds", 0.0) or 0.0) + (perf_counter() - t0_fill)

        # ---- Strategy D (LAST RESORT): MERGE_WITH_PREV ----
        # Only attempt MERGE when fill produced NO progress.
        # If fill made any progress (full or partial), the window state is already updated;
        # running MERGE would use stale prev_seg (not updated by RECUT/FILL) and corrupt
        # the multiplicity check → duplicate orders.
        if not repaired:
            # SAFETY ASSERTION: prevent merge on a window that fill already modified.
            # prev_seg points to the ORIGINAL prev segment; the line has been updated by
            # RECUT (new_prev_c) and FILL (new_tail_d). Calling MERGE with the original
            # prev_seg + updated u_seg creates duplicate orders.
            assert not (inserts_used > 0), (
                "[campaign_cutter] merge attempted after fill-progress on stale window: "
                f"prev_seg={prev_seg.campaign_local_id} (original, not RECUT-updated), "
                f"u_seg={u_seg.campaign_local_id} (fill-updated), "
                f"inserts_used={inserts_used}. "
                "Set repaired=True after FILL_PARTIAL_PROGRESS to prevent this."
            )
            diag_b = {"method": "MERGE_WITH_PREV", "attempted": True}
            diag["tail_repair_merge_attempts"] += 1
            print(
                f"[APS][TailRepairWindow] method=MERGE_WITH_PREV, "
                f"attempted_as_last_resort=True, "
                f"prev_tried={recut_tried}, shift_tried={shift_tried}, fill_tried={fill_tried}, "
                f"fill_result={fill_progress_result}"
            )
            t0_merge = perf_counter()
            ok_b, merged_seg, diag_b = _try_merge_with_prev(
                prev_seg, u_seg, order_tons, cfg, tpl_df
            )
            diag_b = {"method": "MERGE_WITH_PREV", "attempted": True}
            diag["tail_repair_merge_attempts"] += 1
            print(
                f"[APS][TailRepairWindow] method=MERGE_WITH_PREV, "
                f"attempted_as_last_resort=True, "
                f"prev_tried={recut_tried}, shift_tried={shift_tried}, fill_tried={fill_tried}, "
                f"fill_result={fill_progress_result}"
            )
            t0_merge = perf_counter()
            ok_b, merged_seg, diag_b = _try_merge_with_prev(
                prev_seg, u_seg, order_tons, cfg, tpl_df
            )
            diag["merge_seconds"] = (diag.get("merge_seconds", 0.0) or 0.0) + (perf_counter() - t0_merge)
            diag_b["merge_attempted_as_last_resort"] = True
            if ok_b and merged_seg is not None:
                preserved, fail_reason, mult_details = _check_order_multiplicity_preserved(
                    [prev_seg, u_seg],
                    [merged_seg],
                    "MERGE_WITH_PREV",
                )
                diag_b["multiplicity_check"] = "OK" if preserved else "FAIL"
                print(
                    f"[APS][TailRepairWindow] method=MERGE_WITH_PREV, "
                    f"multiplicity_check={'OK' if preserved else 'FAIL'}"
                )
                if not preserved:
                    diag["order_multiplicity_preserve_failures"] += 1
                    _record_fail("MERGE_WITH_PREV", fail_reason)
                else:
                    merged_seg.campaign_local_id = prev_seg.campaign_local_id
                    ordered_line[u_idx - 1] = merged_seg
                    del ordered_line[u_idx]
                    seg_index_map[(line, prev_seg.campaign_local_id)] = merged_seg
                    del seg_index_map[(line, u_seg.campaign_local_id)]
                    diag["tail_repair_merge_success"] += 1
                    diag["merge_attempted_as_last_resort_count"] += 1
                    diag["rebalance_window_repairs_success"] += 1
                    diag["total_repaired"] += 1
                    diag["repair_log"].append({
                        "method": "MERGE_WITH_PREV",
                        "segment_id": u_seg.campaign_local_id,
                        "note": "last_resort_after_all_strategies_failed",
                        "multiplicity": "OK",
                    })
                    print(
                        f"[APS][TAIL_REPAIR_HIT] method=MERGE_WITH_PREV, line={line}, "
                        f"seg_prev={prev_seg.campaign_local_id}, seg_tail={u_seg.campaign_local_id}"
                    )
                    repaired = True
            elif not repaired:
                _record_fail("MERGE_WITH_PREV", diag_b.get("failure_reason", "unknown"))
                preserved, fail_reason, mult_details = _check_order_multiplicity_preserved(
                    [prev_seg, u_seg],
                    [merged_seg],
                    "MERGE_WITH_PREV",
                )
                diag_b["multiplicity_check"] = "OK" if preserved else "FAIL"
                print(
                    f"[APS][TailRepairWindow] method=MERGE_WITH_PREV, "
                    f"multiplicity_check={'OK' if preserved else 'FAIL'}"
                )
                if not preserved:
                    diag["order_multiplicity_preserve_failures"] += 1
                    _record_fail("MERGE_WITH_PREV", fail_reason)
                else:
                    merged_seg.campaign_local_id = prev_seg.campaign_local_id
                    ordered_line[u_idx - 1] = merged_seg
                    del ordered_line[u_idx]
                    seg_index_map[(line, prev_seg.campaign_local_id)] = merged_seg
                    del seg_index_map[(line, u_seg.campaign_local_id)]
                    diag["tail_repair_merge_success"] += 1
                    diag["rebalance_window_repairs_success"] += 1
                    diag["total_repaired"] += 1
                    diag["repair_log"].append({
                        "method": "MERGE_WITH_PREV",
                        "segment_id": u_seg.campaign_local_id,
                        "multiplicity": "OK",
                    })
                    print(
                        f"[APS][TAIL_REPAIR_HIT] method=MERGE_WITH_PREV, line={line}, "
                        f"seg_prev={prev_seg.campaign_local_id}, seg_tail={u_seg.campaign_local_id}"
                    )
                    repaired = True
            elif not repaired:
                _record_fail("MERGE_WITH_PREV", diag_b.get("failure_reason", "unknown"))

        # NOTE: if not repaired, u_seg remains in ordered_line as-is (invalid).
        # Final still_under is collected by scanning ordered_line below (Step 4).

    # -------------------------------------------------------------------------
    # Step 4: Rebuild final segment list by scanning the FINAL ordered_line.
    # Key invariant: after in-place replacement above, ordered_line contains the
    # authoritative state.  We NO LONGER trust still_under (which was populated
    # only with segments that were NEVER repaired — partial successes replaced
    # in-place).  By scanning ordered_line we guarantee:
    #     1. Every segment lives in exactly ONE of: repaired_segments / still_under
    #     2. No stale old-seg copies survive alongside their replacements
    # -------------------------------------------------------------------------
    repaired_segments: List[CampaignSegment] = []
    still_under: List[CampaignSegment] = []

    for line_key in ["big_roll", "small_roll"]:
        for seg in all_segs_by_line.get(line_key, []):
            if seg.is_valid:
                repaired_segments.append(seg)
            else:
                still_under.append(seg)

    diag["final_underfilled_from_ordered_line_count"] = len(still_under)

    # -------------------------------------------------------------------------
    # Step 5: Global uniqueness check — crash early if duplicates remain
    # -------------------------------------------------------------------------
    dup_count, dup_examples = _check_global_order_uniqueness(
        repaired_segments + still_under
    )
    diag["duplicate_orders_after_rebalance"] = dup_count
    diag["duplicate_order_examples"] = dup_examples[:20]  # cap at 20 examples

    if dup_count > 0:
        dup_ids = [e["order_id"] for e in dup_examples[:5]]
        raise RuntimeError(
            f"[campaign_cutter] duplicate orders across segments after rebalance: "
            f"count={dup_count}, examples={dup_ids}"
        )

    diag["underfilled_after"] = len(still_under)

    # ---- Compute total tail repair time ----
    diag["tail_repair_seconds"] = (diag.get("recut_seconds", 0.0) or 0.0) + \
                                  (diag.get("shift_seconds", 0.0) or 0.0) + \
                                  (diag.get("fill_seconds", 0.0) or 0.0) + \
                                  (diag.get("merge_seconds", 0.0) or 0.0)

    # ---- Print CUTTER_TIMING summary ----
    print(
        f"[APS][CUTTER_TIMING] tail_repair={diag.get('tail_repair_seconds', 0.0):.3f}s, "
        f"recut={diag.get('recut_seconds', 0.0):.3f}s, "
        f"shift={diag.get('shift_seconds', 0.0):.3f}s, "
        f"fill={diag.get('fill_seconds', 0.0):.3f}s, "
        f"merge={diag.get('merge_seconds', 0.0):.3f}s"
    )

    # ---- Print TAIL_REPAIR_BUDGET summary ----
    print(
        f"[APS][TAIL_REPAIR_BUDGET] total_attempted={diag.get('tail_repair_windows_attempted_total', 0)}, "
        f"skipped_by_gap={diag.get('tail_repair_windows_skipped_by_gap', 0)}, "
        f"skipped_by_budget={diag.get('tail_repair_windows_skipped_by_budget', 0)}, "
        f"per_line={windows_by_line}, "
        f"recut_disabled={diag.get('recut_disabled_after_failures', False)}, "
        f"recut_cutpoints_tested={diag.get('recut_candidate_points_tested', 0)}, "
        f"fill_candidates_tested={diag.get('fill_candidates_tested_total', 0)}"
    )

    # ---- Print TAIL_REPAIR summary ----
    recut_att = diag.get("tail_repair_recut_attempts", 0)
    recut_suc = diag.get("tail_repair_recut_success", 0)
    shift_att = diag.get("tail_repair_shift_attempts", 0)
    shift_suc = diag.get("tail_repair_shift_success", 0)
    fill_att = diag.get("tail_repair_fill_attempts", 0)
    fill_suc = diag.get("tail_repair_fill_success", 0)
    merge_att = diag.get("tail_repair_merge_attempts", 0)
    merge_suc = diag.get("tail_repair_merge_success", 0)
    print(
        f"[APS][TAIL_REPAIR_SUMMARY] recut_attempts={recut_att}, recut_success={recut_suc}, "
        f"shift_attempts={shift_att}, shift_success={shift_suc}, "
        f"fill_attempts={fill_att}, fill_success={fill_suc}, "
        f"merge_attempts={merge_att}, merge_success={merge_suc}"
    )
    print(
        f"[APS][TAIL_FILL_SUMMARY] attempts={fill_att}, success={fill_suc}, "
        f"partial_progress={diag.get('tail_fill_partial_progress_count', 0)}, "
        f"multiplicity_ok={diag.get('tail_fill_multiplicity_ok_count', 0)}, "
        f"multiplicity_fail={diag.get('tail_fill_multiplicity_fail_count', 0)}, "
        f"candidates_considered={diag.get('tail_fill_candidates_considered', 0)}"
    )
    if recut_att > 0 and recut_suc == 0:
        print("[APS][TAIL_REPAIR_WARNING] method=RECUT_TWO_SEGMENTS no success this run")
    if fill_att > 0 and fill_suc == 0:
        mult_fail = diag.get("tail_fill_multiplicity_fail_count", 0)
        print("[APS][TAIL_FILL_WARNING] entered fill branch but no successful rescue")
        print(
            f"[APS][TAIL_FILL_WARNING] candidates_considered={diag.get('tail_fill_candidates_considered', 0)}, "
            f"candidates_tested={diag.get('fill_candidates_tested_total', 0)}, "
            f"multiplicity_fail={mult_fail}, "
            f"failure_reasons={list(set(r.get('reason', '') for r in diag.get('fail_reasons', []) if r.get('method') == 'TAIL_FILL_FROM_DROPPED'))[:5]}"
        )

    return repaired_segments, still_under, diag


# ---------------------------------------------------------------------------
# Segment template pair validation
# ---------------------------------------------------------------------------

def _validate_segment_template_pairs(
    segment: CampaignSegment,
    tpl_df: pd.DataFrame,
) -> Tuple[bool, List[dict]]:
    """
    Validate that all adjacent order pairs within a segment exist in the template.

    This function checks if the segment's internal order sequence respects
    template edges. If any adjacent pair (a, b) is missing from the template,
    it records the miss with full context for debugging.

    Args:
        segment: The CampaignSegment to validate
        tpl_df: Template DataFrame with from_order_id, to_order_id columns

    Returns:
        Tuple of (all_ok: bool, miss_examples: list[dict]) where each miss example contains:
            - line: production line
            - campaign_local_id: segment identifier
            - order_id_a: first order in the missing pair
            - order_id_b: second order in the missing pair
            - seq_a: 1-based position of order_a in segment
            - seq_b: 1-based position of order_b in segment
    """
    miss_examples: List[dict] = []

    if tpl_df.empty or len(segment.order_ids) < 2:
        return True, miss_examples

    # Build template keyset: set of (from_oid, to_oid) tuples
    tpl_keys: set = set()
    for _, trow in tpl_df.iterrows():
        from_oid = str(trow.get("from_order_id", ""))
        to_oid = str(trow.get("to_order_id", ""))
        tpl_keys.add((from_oid, to_oid))

    # Check each adjacent pair
    for i in range(len(segment.order_ids) - 1):
        oid_a = segment.order_ids[i]
        oid_b = segment.order_ids[i + 1]
        key = (oid_a, oid_b)

        if key not in tpl_keys:
            miss_examples.append({
                "line": segment.line,
                "campaign_local_id": segment.campaign_local_id,
                "order_id_a": oid_a,
                "order_id_b": oid_b,
                "seq_a": i + 1,  # 1-based position
                "seq_b": i + 2,  # 1-based position
            })

    all_ok = len(miss_examples) == 0
    return all_ok, miss_examples


# ---------------------------------------------------------------------------
# Main cutting function
# ---------------------------------------------------------------------------

def cut_sequences_into_campaigns(
    chains_by_line: Dict[str, List[ConstructiveChain]],
    orders_df: pd.DataFrame,
    cfg: PlannerConfig,
    tpl_df: pd.DataFrame | None = None,
) -> CampaignCutResult:
    """
    Cut constructive chains into campaign segments.

    This function implements the second layer of the Constructive + ALNS path:
    - Takes long chains from constructive_sequence_builder
    - Cuts them at natural tonnage boundaries
    - Segments meeting min ton are output as valid
    - Underfilled tails are held separately (not discarded, not faked)
    - Validates segment internal template pair integrity

    Args:
        chains_by_line: Dict mapping line name to list of ConstructiveChain
                       e.g. {"big_roll": [chain1, chain2], "small_roll": [...]}
        orders_df: Full orders DataFrame (for ton lookup)
        cfg: PlannerConfig with campaign rule parameters
        tpl_df: Optional template DataFrame for segment validation.
                If provided, validates that all adjacent order pairs in segments
                exist in the template. Missing pairs are logged immediately.

    Returns:
        CampaignCutResult with segments, underfilled, dropped, diagnostics
    """
    print(f"[APS][RUN_PATH_FINGERPRINT] CAMPAIGN_CUTTER_V2_20260416A")
    # Load campaign parameters
    ton_min = float(cfg.rule.campaign_ton_min) if cfg.rule else 500.0
    ton_max = float(cfg.rule.campaign_ton_max) if cfg.rule else 2000.0
    ton_target = float(cfg.rule.campaign_ton_target) if cfg.rule else 1500.0

    segments: List[CampaignSegment] = []
    underfilled_segments: List[CampaignSegment] = []
    dropped_orders: List[dict] = []

    # Cut reason counters
    cut_reason_stats: Dict[str, int] = {
        "MAX_LIMIT": 0,
        "TARGET_REACHED": 0,
        "NO_SUCCESSOR": 0,
        "TAIL_UNDERFILLED": 0,
        "TAIL_REBALANCED": 0,
        "TAIL_UNDERFILLED_REBALANCE_FAILED": 0,
    }

    # Tail rebalancing diagnostics
    tail_rebalance_summary: Dict = {
        "total_attempted": 0,
        "total_success": 0,
        "total_failed": 0,
        "failed_tails": [],  # List of failed rebalance attempts with details
    }

    # Per-line segment counters for local_id assignment
    line_local_ids: Dict[str, int] = {"big_roll": 0, "small_roll": 0}

    # Build order ton lookup
    order_tons: Dict[str, float] = {}
    if not orders_df.empty:
        for _, row in orders_df.iterrows():
            oid = str(row["order_id"])
            order_tons[oid] = float(row.get("tons", 0) or 0)

    # Process each chain
    for line, chains in chains_by_line.items():
        for chain in chains:
            if len(chain.order_ids) == 0:
                continue

            # Slice pointer within this chain
            pos = 0
            while pos < len(chain.order_ids):
                # Start new segment from position pos
                seg_start = pos
                seg_tons = order_tons.get(chain.order_ids[pos], 0.0)

                # Advance through chain while within limits
                while True:
                    next_pos = pos + 1
                    if next_pos >= len(chain.order_ids):
                        # End of chain — no successor
                        cut_reason = CutReason.NO_SUCCESSOR
                        break

                    next_oid = chain.order_ids[next_pos]
                    next_tons = order_tons.get(next_oid, 0.0)
                    proposed_tons = seg_tons + next_tons

                    # HARD limit check: would exceed campaign_ton_max
                    if proposed_tons > ton_max:
                        cut_reason = CutReason.MAX_LIMIT
                        break

                    # SOFT target check: within window, adding next worsens deviation
                    if seg_tons >= ton_min:
                        deviation_now = abs(seg_tons - ton_target)
                        deviation_cont = abs(proposed_tons - ton_target)
                        # Cut if deviation would get noticeably worse
                        if deviation_cont > deviation_now + ton_target * 0.05:
                            cut_reason = CutReason.TARGET_REACHED
                            break

                    # Continue: add next order
                    pos = next_pos
                    seg_tons = proposed_tons

                # Build segment from chain[seg_start : pos+1]
                seg_end = pos
                order_ids = chain.order_ids[seg_start : seg_end + 1]

                if len(order_ids) == 0:
                    pos += 1
                    continue

                # Determine if valid
                is_valid = seg_tons >= ton_min

                # Assign local id
                line_local_ids[line] = line_local_ids.get(line, 0) + 1
                local_id = line_local_ids[line]

                segment = CampaignSegment(
                    line=line,
                    campaign_local_id=local_id,
                    order_ids=order_ids,
                    total_tons=seg_tons,
                    cut_reason=cut_reason,
                    start_order_id=order_ids[0],
                    end_order_id=order_ids[-1],
                    edge_count=max(0, len(order_ids) - 1),
                    is_valid=is_valid,
                )

                # ---- Tail rebalancing: collect underfilled tails for post-processing ----
                # Trigger conditions (ANY of):
                #   a) cut_reason == TAIL_UNDERFILLED  (explicitly underfilled)
                #   b) cut_reason == NO_SUCCESSOR and seg_tons < ton_min  (chain ended but tail too light)
                # Both cases mean the tail is underfilled and deserves a rebalance attempt.
                is_underfilled_tail = (
                    not is_valid
                    and (
                        cut_reason == CutReason.TAIL_UNDERFILLED
                        or (cut_reason == CutReason.NO_SUCCESSOR and seg_tons < ton_min)
                    )
                )

                if is_underfilled_tail:
                    # Collect underfilled tail for later multi-strategy repair
                    tail_rebalance_summary["total_attempted"] += 1
                    underfilled_segments.append(segment)
                    cut_reason_stats[cut_reason.value] = (
                        cut_reason_stats.get(cut_reason.value, 0) + 1
                    )
                    pos += 1
                    continue

                # ---- Normal segment append ----
                if is_valid:
                    segments.append(segment)
                    cut_reason_stats[cut_reason.value] = (
                        cut_reason_stats.get(cut_reason.value, 0) + 1
                    )
                else:
                    # Edge case: underfilled but NOT an underfilled tail
                    # (e.g., single-order segment at head of chain)
                    underfilled_segments.append(segment)
                    cut_reason_stats[cut_reason.value] = (
                        cut_reason_stats.get(cut_reason.value, 0) + 1
                    )

                # Advance to next segment start
                pos += 1

    # ---- Phase 2: Multi-strategy tail rebalance (window-based, no duplicates) ----
    rebal_diag: dict = {
        "total_underfilled_before": len(underfilled_segments),
        "tail_repair_attempts_total": 0,
        "tail_repair_shift_success": 0,
        "tail_repair_recut_success": 0,
        "tail_repair_merge_success": 0,
        "tail_repair_fill_success": 0,
        "underfilled_segments_before_repair": len(underfilled_segments),
        "underfilled_segments_after_repair": 0,
        "underfilled_segments_near_min_count": 0,
        "total_repaired": 0,
        "underfilled_after": len(underfilled_segments),
        "fail_reasons": [],
        "repair_log": [],
        "rebalance_window_repairs_attempted": 0,
        "rebalance_window_repairs_success": 0,
        "duplicate_orders_after_rebalance": 0,
        "duplicate_order_examples": [],
        "order_multiplicity_preserve_failures": 0,
        "partial_shift_replaced_in_place_count": 0,
        "partial_recut_replaced_in_place_count": 0,
        "fail_reason_PREV_WOULD_UNDERFILL": 0,
        "fail_reason_NO_VALID_RECUT_POINT": 0,
        "fail_reason_MERGED_EXCEEDS_MAX": 0,
        "fail_reason_TEMPLATE_PAIR_INVALID_AFTER_REPAIR": 0,
        "fail_reason_NO_DROPPED_CANDIDATE_FOR_FILL": 0,
        "fail_reason_other": 0,
    }
    all_underfilled_before = list(underfilled_segments)

    # Compute placed_oids from ALL segments already assigned to campaign structure.
    # placed_oids represents ALL orders already present in any current segment,
    # not only valid segments. Fill candidates must come only from orders NOT
    # already assigned to any segment (neither valid segments nor underfilled segments).
    placed_oids: set = set()
    for seg in segments + underfilled_segments:
        placed_oids.update(seg.order_ids)

    if underfilled_segments:
        segments, underfilled_segments, rebal_diag = _rebalance_underfilled_segments(
            segments, underfilled_segments, order_tons, cfg, tpl_df,
            orders_df, placed_oids,
        )
        repaired_count = rebal_diag.get("total_repaired", 0)
        repaired_by_shift = rebal_diag.get("tail_repair_shift_success", 0)
        repaired_by_recut = rebal_diag.get("tail_repair_recut_success", 0)
        repaired_by_merge = rebal_diag.get("tail_repair_merge_success", 0)
        repaired_by_fill = rebal_diag.get("tail_repair_fill_success", 0)
        still_under = len(underfilled_segments)
        if repaired_count > 0:
            print(
                f"[APS][TailRepair] repaired={repaired_count}, "
                f"shift_ok={repaired_by_shift}, "
                f"recut_ok={repaired_by_recut}, "
                f"merge_ok={repaired_by_merge}, "
                f"fill_ok={repaired_by_fill}, "
                f"still_underfilled={still_under}, "
                f"window_attempts={rebal_diag.get('rebalance_window_repairs_attempted', 0)}, "
                f"window_success={rebal_diag.get('rebalance_window_repairs_success', 0)}, "
                f"multiplicity_failures={rebal_diag.get('order_multiplicity_preserve_failures', 0)}"
            )
        if repaired_count > 0:
            cut_reason_stats["TAIL_REBALANCED"] = repaired_count

    # Update tail_rebalance_summary with full rebal diagnostics
    tail_rebalance_summary.update({
        "tail_repair_attempts_total": rebal_diag.get("tail_repair_attempts_total", 0),
        "tail_repair_recut_attempts": rebal_diag.get("tail_repair_recut_attempts", 0),
        "tail_repair_recut_success": rebal_diag.get("tail_repair_recut_success", 0),
        "tail_repair_shift_attempts": rebal_diag.get("tail_repair_shift_attempts", 0),
        "tail_repair_shift_success": rebal_diag.get("tail_repair_shift_success", 0),
        "tail_repair_fill_attempts": rebal_diag.get("tail_repair_fill_attempts", 0),
        "tail_repair_fill_success": rebal_diag.get("tail_repair_fill_success", 0),
        "tail_fill_candidates_considered": rebal_diag.get("tail_fill_candidates_considered", 0),
        "tail_fill_partial_progress_count": rebal_diag.get("tail_fill_partial_progress_count", 0),
        "tail_fill_multiplicity_ok_count": rebal_diag.get("tail_fill_multiplicity_ok_count", 0),
        "tail_fill_multiplicity_fail_count": rebal_diag.get("tail_fill_multiplicity_fail_count", 0),
        "tail_repair_merge_attempts": rebal_diag.get("tail_repair_merge_attempts", 0),
        "tail_repair_merge_success": rebal_diag.get("tail_repair_merge_success", 0),
        "total_repaired": rebal_diag.get("total_repaired", 0),
        "underfilled_segments_before_repair": rebal_diag.get("total_underfilled_before", 0),
        "underfilled_segments_after_repair": rebal_diag.get("underfilled_after", len(underfilled_segments)),
        "underfilled_segments_near_min_count": rebal_diag.get("underfilled_segments_near_min_count", 0),
        "fail_reasons": rebal_diag.get("fail_reasons", []),
        "repair_log": rebal_diag.get("repair_log", []),
        "partial_shift_replaced_in_place_count": rebal_diag.get("partial_shift_replaced_in_place_count", 0),
        "partial_recut_replaced_in_place_count": rebal_diag.get("partial_recut_replaced_in_place_count", 0),
        # Failure reason breakdown
        "fail_reason_PREV_WOULD_UNDERFILL": rebal_diag.get("fail_reason_PREV_WOULD_UNDERFILL", 0),
        "fail_reason_NO_VALID_RECUT_POINT": rebal_diag.get("fail_reason_NO_VALID_RECUT_POINT", 0),
        "fail_reason_MERGED_EXCEEDS_MAX": rebal_diag.get("fail_reason_MERGED_EXCEEDS_MAX", 0),
        "fail_reason_TEMPLATE_PAIR_INVALID_AFTER_REPAIR": rebal_diag.get("fail_reason_TEMPLATE_PAIR_INVALID_AFTER_REPAIR", 0),
        "fail_reason_NO_DROPPED_CANDIDATE_FOR_FILL": rebal_diag.get("fail_reason_NO_DROPPED_CANDIDATE_FOR_FILL", 0),
        "fail_reason_other": rebal_diag.get("fail_reason_other", 0),
        # Window-based diagnostics
        "rebalance_window_repairs_attempted": rebal_diag.get("rebalance_window_repairs_attempted", 0),
        "rebalance_window_repairs_success": rebal_diag.get("rebalance_window_repairs_success", 0),
        "duplicate_orders_after_rebalance": rebal_diag.get("duplicate_orders_after_rebalance", 0),
        "duplicate_order_examples": rebal_diag.get("duplicate_order_examples", []),
        "order_multiplicity_preserve_failures": rebal_diag.get("order_multiplicity_preserve_failures", 0),
        # Tail repair budget diagnostics
        "tail_repair_windows_attempted_total": rebal_diag.get("tail_repair_windows_attempted_total", 0),
        "tail_repair_windows_skipped_by_gap": rebal_diag.get("tail_repair_windows_skipped_by_gap", 0),
        "tail_repair_windows_skipped_by_budget": rebal_diag.get("tail_repair_windows_skipped_by_budget", 0),
        "recut_disabled_after_failures": rebal_diag.get("recut_disabled_after_failures", False),
        "fill_candidates_tested_total": rebal_diag.get("fill_candidates_tested_total", 0),
        # ---- Phase timing ----
        "tail_repair_seconds": rebal_diag.get("tail_repair_seconds", 0.0),
        "recut_seconds": rebal_diag.get("recut_seconds", 0.0),
        "shift_seconds": rebal_diag.get("shift_seconds", 0.0),
        "fill_seconds": rebal_diag.get("fill_seconds", 0.0),
        "merge_seconds": rebal_diag.get("merge_seconds", 0.0),
    })

    # Compute diagnostics
    all_segments = segments + underfilled_segments
    total_orders_in_segments = sum(len(s.order_ids) for s in all_segments)

    # ---- Segment template pair validation ----
    # Validate that all adjacent order pairs within segments exist in the template.
    # This catches template miss issues early, before they propagate to validator.
    segment_template_miss_count = 0
    segment_template_miss_examples: List[dict] = []
    segment_template_valid_miss_count = 0  # Misses found in valid segments
    segment_template_underfill_miss_count = 0  # Misses found in underfilled segments

    if tpl_df is not None and not tpl_df.empty:
        for seg in all_segments:
            if len(seg.order_ids) < 2:
                continue  # Single order segment has no adjacent pairs

            all_ok, miss_examples = _validate_segment_template_pairs(seg, tpl_df)
            if not all_ok:
                segment_template_miss_count += len(miss_examples)
                segment_template_miss_examples.extend(miss_examples[:20])  # Cap at 20 examples
                if seg.is_valid:
                    segment_template_valid_miss_count += len(miss_examples)
                else:
                    segment_template_underfill_miss_count += len(miss_examples)

                # Log miss immediately for each segment with misses
                if miss_examples:
                    print(
                        f"[APS][CampaignCutter] TEMPLATE_MISS in segment: "
                        f"line={seg.line}, campaign_local_id={seg.campaign_local_id}, "
                        f"is_valid={seg.is_valid}, miss_count={len(miss_examples)}"
                    )
                    # Print first 3 miss examples for quick debugging
                    for miss in miss_examples[:3]:
                        print(
                            f"  [MISS] seq={miss['seq_a']}->{miss['seq_b']}: "
                            f"{miss['order_id_a']} -> {miss['order_id_b']}"
                        )

    # Per-line breakdown
    line_stats: Dict[str, dict] = {}
    for line in ["big_roll", "small_roll"]:
        line_valid = [s for s in segments if s.line == line]
        line_under = [s for s in underfilled_segments if s.line == line]
        line_stats[line] = {
            "valid_segment_count": len(line_valid),
            "underfilled_segment_count": len(line_under),
            "total_tons": sum(s.total_tons for s in line_valid),
            "avg_tons": (
                sum(s.total_tons for s in line_valid) / max(1, len(line_valid))
            ),
            "tons_distribution": _compute_tons_distribution(line_valid, ton_min, ton_max, ton_target),
        }

    diagnostics = {
        "run_path_fingerprint_campaign_cutter": "CAMPAIGN_CUTTER_V2_20260416A",
        "total_valid_segments": len(segments),
        "total_underfilled_segments": len(underfilled_segments),
        "total_orders_in_segments": total_orders_in_segments,
        "cut_reason_stats": dict(cut_reason_stats),
        "line_stats": line_stats,
        "campaign_ton_min": ton_min,
        "campaign_ton_max": ton_max,
        "campaign_ton_target": ton_target,
        "tail_rebalance_summary": dict(tail_rebalance_summary),
        # Tail repair detailed diagnostics (Phase 2)
        "tail_rebalance_attempts": rebal_diag.get("shift_attempts", 0)
        + rebal_diag.get("merge_attempts", 0)
        + rebal_diag.get("recut_attempts", 0),
        "tail_rebalance_success_count": rebal_diag.get("total_repaired", 0),
        "tail_merge_success_count": rebal_diag.get("merge_success_count", 0),
        "tail_recut_success_count": rebal_diag.get("recut_success_count", 0),
        "underfilled_segments_total": rebal_diag.get("total_underfilled_before", 0),
        "underfilled_segments_near_min_count": tail_rebalance_summary.get(
            "underfilled_segments_near_min_count", 0
        ),
        "underfilled_segments_repaired_count": rebal_diag.get("total_repaired", 0),
        "underfilled_segments_before_repair": rebal_diag.get("total_underfilled_before", 0),
        "underfilled_segments_after_repair": rebal_diag.get("underfilled_after", len(underfilled_segments)),
        "tail_rebalance_fail_reasons": rebal_diag.get("fail_reasons", []),
        "tail_rebalance_repair_log": rebal_diag.get("repair_log", []),
        # Window-based + multiplicity diagnostics
        "rebalance_window_repairs_attempted": rebal_diag.get("rebalance_window_repairs_attempted", 0),
        "rebalance_window_repairs_success": rebal_diag.get("rebalance_window_repairs_success", 0),
        "duplicate_orders_after_rebalance": rebal_diag.get("duplicate_orders_after_rebalance", 0),
        "duplicate_order_examples": rebal_diag.get("duplicate_order_examples", []),
        "order_multiplicity_preserve_failures": rebal_diag.get("order_multiplicity_preserve_failures", 0),
        # Segment template pair validation diagnostics
        "segment_template_miss_count": segment_template_miss_count,
        "segment_template_miss_examples": segment_template_miss_examples,
        "segment_template_valid_miss_count": segment_template_valid_miss_count,
        "segment_template_underfill_miss_count": segment_template_underfill_miss_count,
        "underfilled_details": [
            {
                "line": s.line,
                "segment_id": s.campaign_local_id,
                "order_ids": s.order_ids,
                "total_tons": s.total_tons,
                "below_min_by": round(ton_min - s.total_tons, 2),
            }
            for s in underfilled_segments
        ],
    }

    # Print summary
    _print_cutting_summary(segments, underfilled_segments, cut_reason_stats, ton_min, ton_max)

    return CampaignCutResult(
        segments=segments,
        underfilled_segments=underfilled_segments,
        dropped_orders=dropped_orders,
        diagnostics=diagnostics,
    )


def _compute_tons_distribution(
    segments: List[CampaignSegment],
    ton_min: float,
    ton_max: float,
    ton_target: float,
) -> dict:
    """Compute tonnage distribution statistics for segments."""
    if not segments:
        return {
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "below_min": 0,
            "within_window": 0,
            "above_max": 0,
            "near_target": 0,
        }

    tons_list = sorted([s.total_tons for s in segments])
    n = len(tons_list)

    below_min = sum(1 for t in tons_list if t < ton_min)
    within_window = sum(1 for t in tons_list if ton_min <= t <= ton_max)
    above_max = sum(1 for t in tons_list if t > ton_max)
    near_target = sum(1 for t in tons_list if abs(t - ton_target) <= ton_target * 0.1)

    return {
        "min": round(min(tons_list), 2),
        "max": round(max(tons_list), 2),
        "mean": round(sum(tons_list) / n, 2),
        "median": round(tons_list[n // 2], 2),
        "below_min": below_min,
        "within_window": within_window,
        "above_max": above_max,
        "near_target": near_target,
    }


def _print_cutting_summary(
    segments: List[CampaignSegment],
    underfilled_segments: List[CampaignSegment],
    cut_reason_stats: Dict[str, int],
    ton_min: float,
    ton_max: float,
) -> None:
    """Print cutting summary to stdout."""
    total_orders = sum(len(s.order_ids) for s in segments + underfilled_segments)

    print(
        f"[APS][CampaignCutter] "
        f"valid={len(segments)}, "
        f"underfilled={len(underfilled_segments)}, "
        f"orders_in_segments={total_orders}"
    )

    reason_line = ", ".join(
        f"{k}={v}" for k, v in cut_reason_stats.items() if v > 0
    )
    print(f"[APS][CampaignCutter] cut_reasons: {reason_line}")

    # Per-line summary
    for line in ["big_roll", "small_roll"]:
        line_segs = [s for s in segments if s.line == line]
        line_under = [s for s in underfilled_segments if s.line == line]
        if line_segs or line_under:
            total_t = sum(s.total_tons for s in line_segs)
            avg_t = total_t / max(1, len(line_segs))
            print(
                f"[APS][CampaignCutter] {line}: "
                f"valid={len(line_segs)}, under={len(line_under)}, "
                f"avg_tons={avg_t:.0f}, min={ton_min}, max={ton_max}"
            )


# ---------------------------------------------------------------------------
# Family repair dedup helper for reconstruction
# ---------------------------------------------------------------------------

def _should_skip_family_repair(
    line: str,
    from_order_id: str,
    to_order_id: str,
    bridge_family: str,
    already_attempted_keys: set,
    recon_diag: dict,
) -> bool:
    """
    Check if a family repair candidate should be skipped because ALNS/local rebuild
    has already attempted it (dedup gate).

    Args:
        line: Production line
        from_order_id: Source order ID
        to_order_id: Target order ID
        bridge_family: Bridge family type
        already_attempted_keys: Set of keys already attempted by ALNS/local rebuild
        recon_diag: Reconstruction diagnostics dict (for incrementing skip counter)

    Returns:
        True if should skip (key already attempted), False if should proceed.
    """
    _key = (str(line), str(from_order_id), str(to_order_id), str(bridge_family))
    if _key in already_attempted_keys:
        if recon_diag is not None:
            recon_diag["repair_virtual_family_skipped_due_to_existing_attempt_count"] = (
                recon_diag.get("repair_virtual_family_skipped_due_to_existing_attempt_count", 0) + 1
            )
        return True
    return False


__all__ = [
    "CutReason",
    "CampaignSegment",
    "CampaignCutResult",
    "cut_sequences_into_campaigns",
    "_reconstruct_underfilled_segments",
]
