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
    if bool(row.get("is_virtual_bridge", False)):
        return "VIRTUAL_BRIDGE_EDGE"
    return ""


def _is_real_bridge_row(row: dict) -> bool:
    kind = _row_bridge_type(row)
    return kind in {"REAL_BRIDGE_EDGE", "REAL_BRIDGE", "REAL"}


def _is_virtual_bridge_row(row: dict) -> bool:
    kind = _row_bridge_type(row)
    return kind in {"VIRTUAL_BRIDGE_EDGE", "VIRTUAL_BRIDGE", "VIRTUAL"}


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
        pairs: list[tuple[str, str, str]] = []
        for left_distance, left_oid in enumerate(reversed(left_band)):
            for right_distance, right_oid in enumerate(right_band):
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
            if len(adjustments) >= int(adjustment_limit):
                return adjustments
    return adjustments


def _build_adjustment_band_pairs(
    *,
    line: str,
    adjusted_left_orders: list[str],
    adjusted_right_orders: list[str],
    left_band_k: int,
    right_band_k: int,
    max_pairs_per_split: int,
) -> tuple[list[tuple[str, str, str]], list[str], list[str]]:
    left_band = [str(v) for v in adjusted_left_orders[-max(1, int(left_band_k)):]]
    right_band = [str(v) for v in adjusted_right_orders[: max(1, int(right_band_k))]]
    pairs: list[tuple[str, str, str]] = []
    for left_oid in reversed(left_band):
        for right_oid in right_band:
            if len(pairs) >= int(max_pairs_per_split):
                break
            pairs.append((str(line), str(left_oid), str(right_oid)))
        if len(pairs) >= int(max_pairs_per_split):
            break
    return list(dict.fromkeys(pairs)), left_band, right_band


def _boundary_band_distance_map(band_logs: list[dict]) -> dict[tuple[str, str], int]:
    distances: dict[tuple[str, str], int] = {}
    for item in band_logs:
        left_band = [str(v) for v in item.get("left_band", [])]
        right_band = [str(v) for v in item.get("right_band", [])]
        left_distance = {oid: idx for idx, oid in enumerate(reversed(left_band))}
        right_distance = {oid: idx for idx, oid in enumerate(right_band)}
        for _, left_oid, right_oid in item.get("band_pairs", []):
            dist = int(left_distance.get(str(left_oid), 999)) + int(right_distance.get(str(right_oid), 999))
            key = (str(left_oid), str(right_oid))
            distances[key] = min(distances.get(key, dist), dist)
    return distances


def _endpoint_adjustment_cost_map(adjustment_logs: list[dict]) -> dict[tuple[str, str], int]:
    costs: dict[tuple[str, str], int] = {}
    for item in adjustment_logs:
        cost = int(item.get("left_trim", 0) or 0) + int(item.get("right_trim", 0) or 0)
        for _, left_oid, right_oid in item.get("band_pairs", []):
            key = (str(left_oid), str(right_oid))
            costs[key] = min(costs.get(key, cost), cost)
    return costs


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
    order_tons: dict[str, float],
    campaign_ton_min: float,
    campaign_ton_max: float,
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
        "pair_invalid_unknown": 0,
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
            out["matched"] += 1
            print(
                f"[APS][REPAIR_BRIDGE_CANDIDATE] candidate_id={candidate_id}, line={line}, "
                f"block_id={block_id}, from_order_id={src}, to_order_id={dst}, "
                f"edge_type=REAL_BRIDGE_EDGE, bridge_count={bridge_count}"
            )
            print(f"[APS][REPAIR_BRIDGE_STAGE] candidate_id={candidate_id}, stage=MATCHED")

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
            template_ok = pair in direct_edges or pair in real_edges
            adjacency_ok = tentative_frontier_pair == pair
            if not template_ok:
                reason = "TEMPLATE_KEY_MISSING"
            elif not adjacency_ok:
                reason = "UNKNOWN_PAIR_INVALID"
            else:
                reason = ""
            print(
                f"[APS][REPAIR_BRIDGE_PAIR_CHECK] candidate_id={candidate_id}, pair=({src},{dst}), "
                f"template_ok={bool(template_ok)}, adjacency_ok={bool(adjacency_ok)}, "
                f"reason={reason or 'OK'}"
            )
            if reason:
                out["rejected_pair_invalid"] += 1
                out["exact_invalid_pair_count"] += 1
                out["pair_invalid_unknown"] += 1
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
) -> tuple[bool, int]:
    # Virtual bridge is explicitly blocked in this repair-only reconstruction.
    if allow_virtual_bridge:
        return False, 0
    real_used = 0
    for i in range(len(oids) - 1):
        key = (str(oids[i]), str(oids[i + 1]))
        if key in direct_edges:
            continue
        if allow_real_bridge and key in real_edges and real_used < int(max_real_bridge_per_segment):
            real_used += 1
            continue
        return False, real_used
    return True, real_used


def _segment_edge_stats_for_reconstruction(
    oids: list[str],
    direct_edges: set[tuple[str, str]],
    real_edges: set[tuple[str, str]],
    *,
    allow_real_bridge: bool,
    max_real_bridge_per_segment: int,
    real_edge_distance: dict[tuple[str, str], int] | None = None,
    real_edge_adjustment_cost: dict[tuple[str, str], int] | None = None,
) -> dict:
    total_missing = 0
    real_candidate_edges = 0
    bridge_limit_exceeded = 0
    real_used = 0
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
    return {
        "missing_edges": int(total_missing),
        "real_candidate_edges": int(real_candidate_edges),
        "real_used": int(real_used),
        "bridge_limit_exceeded": int(bridge_limit_exceeded),
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
            )
            obs["repair_only_real_bridge_candidates_total"] += int(edge_stats["real_candidate_edges"])
            obs["repair_only_real_bridge_filtered_bridge_limit_exceeded"] += int(edge_stats["bridge_limit_exceeded"])
            ok, real_used = _segment_edge_valid_for_reconstruction(
                oids,
                direct_edges,
                real_edges,
                allow_real_bridge=allow_real_bridge,
                allow_virtual_bridge=allow_virtual_bridge,
                max_real_bridge_per_segment=max_real_bridge_per_segment,
            )
            if not ok:
                obs["repair_only_real_bridge_filtered_pair_invalid"] += 1
                continue
            obs["candidate_cutpoints_pair_valid"] += 1
            if int(real_used) == 0:
                obs["direct_only_feasible"] = True
            else:
                obs["repair_only_real_bridge_candidates_kept"] += int(real_used)
            candidates[i].append(
                {
                    "end": j,
                    "oids": oids,
                    "tons": tons,
                    "real_used": int(real_used),
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
        boundary_band_distance = sum(int(item.get("boundary_band_distance", 0)) for item in path)
        endpoint_adjustment_cost = sum(int(item.get("endpoint_adjustment_cost", 0)) for item in path)
        target_gap = sum(float(item.get("target_gap", 0.0)) for item in path)
        residual_count = 1 if residual else 0
        score = (
            -valid_count,
            -salvaged_orders,
            residual_count,
            real_used_total * float(bridge_cost_penalty),
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
    obs.update({
        "success": True,
        "valid_count": int(len(valid)),
        "salvaged_orders": int(sum(len(s.order_ids) for s in valid)),
        "residual_underfilled": int(len(residual_under)),
        "real_bridge_used": int(real_bridge_used),
    })
    return valid, residual_under, obs


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
    cfg: PlannerConfig | None = None,
) -> tuple[list[CampaignSegment], list[CampaignSegment], dict]:
    """
    Repair-only local reconstruction for underfilled segments.

    Initial constructive and initial cutter remain direct-only. This stage only
    rebuilds same-line underfilled blocks and optionally allows REAL_BRIDGE_EDGE
    as a repair candidate after direct-only reconstruction has no gain.
    """
    t0 = perf_counter()
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
        "repair_bridge_pair_invalid_unknown": 0,
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
    if not underfilled_segments:
        diag["underfilled_reconstruction_not_entered_reason"] = "NO_UNDERFILLED_SEGMENTS"
        diag["repair_only_real_bridge_not_entered_reason"] = "NO_UNDERFILLED_SEGMENTS"
        diag["underfilled_reconstruction_seconds"] = round(perf_counter() - t0, 6)
        diag["reconstruction_seconds"] = diag["underfilled_reconstruction_seconds"]
        print("[APS][UNDERFILLED_RECON_SKIP] line=ALL, block_id=0, reason=NO_UNDERFILLED_SEGMENTS")
        print(
            f"[APS][UNDERFILLED_RECON_SUMMARY] attempts=0, success=0, blocks_tested=0, "
            f"valid_delta=0, underfilled_delta=0, salvaged_segments=0, salvaged_orders=0"
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

    for line in sorted({s.line for s in underfilled_segments}):
        line_under = sorted([s for s in underfilled_segments if s.line == line], key=lambda s: s.campaign_local_id)
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
            best = None
            best_size = 1
            block_id = i
            if len(line_under) - i < 2:
                diag["underfilled_reconstruction_blocks_skipped"] += 1
                print(f"[APS][UNDERFILLED_RECON_SKIP] line={line}, block_id={block_id}, reason=BLOCK_TOO_SMALL")
                remaining_under.append(line_under[i])
                i += 1
                continue
            for block_size in (4, 3, 2):
                if i + block_size > len(line_under):
                    continue
                block = line_under[i : i + block_size]
                block_orders = [oid for seg in block for oid in seg.order_ids]
                block_tons = sum(float(tons_by_order.get(str(oid), 0.0) or 0.0) for oid in block_orders)
                block_real_rows = _bridge_rows_for_line(tpl_df, str(line))
                real_bridge_lookup = _build_real_bridge_lookup(block_real_rows)
                single_point_keys = _block_single_point_boundary_keys(block)
                base_boundary_keys, base_band_logs = _block_boundary_band_keys(
                    block,
                    left_band_k=left_band_k,
                    right_band_k=right_band_k,
                    max_pairs_per_split=max_band_pairs,
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
                        band_pairs, adjusted_left_band, adjusted_right_band = _build_adjustment_band_pairs(
                            line=str(line),
                            adjusted_left_orders=adjusted_left_orders,
                            adjusted_right_orders=adjusted_right_orders,
                            left_band_k=left_band_k,
                            right_band_k=right_band_k,
                            max_pairs_per_split=max_band_pairs,
                        )
                        left_trim = int(adjustment.get("left_trim", 0) or 0)
                        right_trim = int(adjustment.get("right_trim", 0) or 0)
                        adjustment_cost = left_trim + right_trim
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
                                "left_band": adjusted_left_band,
                                "right_band": adjusted_right_band,
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
                matched_bridge_keys = [key for key in boundary_keys if key in real_bridge_lookup]
                frontier_matched_keys = [key for key in frontier_keys if key in real_bridge_lookup]
                bridge_candidate_audit = _repair_bridge_candidate_audit(
                    line=str(line),
                    block_id=int(block_id),
                    block=block,
                    real_bridge_lookup=real_bridge_lookup,
                    matched_keys=matched_bridge_keys,
                    frontier_contexts=frontier_contexts,
                    direct_edges=direct_edges,
                    real_edges=real_edges,
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
                    "pair_invalid_unknown": 0,
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
                diag["repair_bridge_pair_invalid_unknown"] += int(bridge_candidate_audit.get("pair_invalid_unknown", 0) or 0)
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
                        break
                    reason_buckets = {
                        "TEMPLATE_PAIR_INVALID": int(real_diag_report.get("repair_only_real_bridge_filtered_pair_invalid", 0) or 0),
                        "TON_WINDOW_INVALID": int(real_diag_report.get("repair_only_real_bridge_filtered_ton_invalid", 0) or 0),
                        "SCORE_WORSE_THAN_DIRECT_ONLY": int(real_diag_report.get("repair_only_real_bridge_filtered_score_worse", 0) or 0),
                        "BRIDGE_LIMIT_EXCEEDED": int(real_diag_report.get("repair_only_real_bridge_filtered_bridge_limit_exceeded", 0) or 0),
                        "BLOCK_MEMBERSHIP_MISMATCH": int(bridge_match_diag.get("block_membership_mismatch", 0) or 0),
                        "LINE_MISMATCH": int(bridge_match_diag.get("line_mismatch", 0) or 0),
                    }
                    dominant_bucket = max(reason_buckets.items(), key=lambda kv: kv[1])[0] if any(v > 0 for v in reason_buckets.values()) else "NO_DIRECT_OR_BRIDGE_CANDIDATE"
                    reason = dominant_bucket
                    if line_raw_real_rows <= 0:
                        reason = "PACK_HAS_NO_REAL_BRIDGE_ROWS"
                    elif bool(line_bridge_schema.get("field_name_mismatch", False)):
                        reason = "BRIDGE_FIELD_NAME_MISMATCH"
                    elif int(bridge_match_diag.get("matched_rows", 0) or 0) <= 0:
                        reason = "ENDPOINT_KEY_MISMATCH"
                    print(
                        f"[APS][REPAIR_BRIDGE_REASON_AUDIT] line={line}, block_id={block_id}, "
                        f"filtered_pair_invalid={int(real_diag_report.get('repair_only_real_bridge_filtered_pair_invalid', 0) or 0)}, "
                        f"filtered_ton_invalid={int(real_diag_report.get('repair_only_real_bridge_filtered_ton_invalid', 0) or 0)}, "
                        f"filtered_score_worse={int(real_diag_report.get('repair_only_real_bridge_filtered_score_worse', 0) or 0)}, "
                        f"final_reason={reason}"
                    )
                    if reason != dominant_bucket and int(bridge_match_diag.get("matched_rows", 0) or 0) > 0:
                        print(
                            f"[APS][REPAIR_BRIDGE_REASON_MISMATCH] line={line}, block_id={block_id}, "
                            f"final_reason={reason}, dominant_bucket={dominant_bucket}"
                        )
                    print(f"[APS][REPAIR_BRIDGE_NO_HIT] line={line}, block_id={block_id}, reason={reason}")
            if best is None:
                diag["underfilled_reconstruction_blocks_skipped"] += 1
                print(
                    f"[APS][UNDERFILLED_RECON_NO_GAIN] line={line}, block_id={block_id}, "
                    f"valid_before={len(valid_segments)}, valid_after={len(new_valid)}, "
                    f"underfilled_before={len(underfilled_segments)}, underfilled_after={len(remaining_under) + 1}"
                )
                print(f"[APS][UNDERFILLED_RECON_SKIP] line={line}, block_id={block_id}, reason=NO_DIRECT_OR_BRIDGE_CANDIDATE")
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
            i += best_size

    diag["underfilled_reconstruction_valid_delta"] = int(len(new_valid) - len(valid_segments))
    diag["underfilled_reconstruction_underfilled_delta"] = int(len(underfilled_segments) - len(remaining_under))
    diag["underfilled_reconstruction_valid_after"] = int(len(new_valid))
    diag["underfilled_reconstruction_underfilled_after"] = int(len(remaining_under))
    diag["underfilled_reconstruction_seconds"] = round(perf_counter() - t0, 6)
    diag["reconstruction_seconds"] = diag["underfilled_reconstruction_seconds"]
    if int(diag.get("repair_only_real_bridge_attempts", 0) or 0) <= 0:
        diag["repair_only_real_bridge_not_entered_reason"] = "DIRECT_ONLY_OR_NO_BRIDGE_STAGE"
    print(
        f"[APS][RECON_GAIN] valid_before={len(valid_segments)}, valid_after={len(new_valid)}, "
        f"underfilled_before={len(underfilled_segments)}, underfilled_after={len(remaining_under)}"
    )
    print(
        f"[APS][RECON_TIMING] reconstruction={float(diag['reconstruction_seconds']):.3f}, "
        f"repair_bridge_real={float(diag['repair_bridge_real_seconds']):.3f}"
    )
    print(
        f"[APS][UNDERFILLED_RECON_SUMMARY] attempts={int(diag.get('underfilled_reconstruction_attempts', 0))}, "
        f"success={int(diag.get('underfilled_reconstruction_success', 0))}, "
        f"blocks_tested={int(diag.get('underfilled_reconstruction_blocks_tested', 0))}, "
        f"valid_delta={int(diag.get('underfilled_reconstruction_valid_delta', 0))}, "
        f"underfilled_delta={int(diag.get('underfilled_reconstruction_underfilled_delta', 0))}, "
        f"salvaged_segments={int(diag.get('underfilled_reconstruction_segments_salvaged', 0))}, "
        f"salvaged_orders={int(diag.get('underfilled_reconstruction_orders_salvaged', 0))}"
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


__all__ = [
    "CutReason",
    "CampaignSegment",
    "CampaignCutResult",
    "cut_sequences_into_campaigns",
    "_reconstruct_underfilled_segments",
]
