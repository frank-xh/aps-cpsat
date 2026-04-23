"""Campaign Assembly Planner for block-first final campaign shaping.

This module sits between block_master and block_realizer.  It does not produce
the final schedule.  Its first responsibility is to expose campaign skeletons,
donor space and bridge requirements before the realizer starts doing local
repair work.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Tuple
from uuid import uuid4

import pandas as pd

from aps_cp_sat.config import PlannerConfig
from aps_cp_sat.model.block_master import BlockMasterResult
from aps_cp_sat.model.block_types import CandidateBlock

ViabilityBand = Literal["ALREADY_VIABLE", "NEAR_VIABLE", "MERGE_CANDIDATE", "HOPELESS_UNDERFILLED"]
AssemblyStatus = Literal["STABLE", "NEEDS_DONOR", "NEEDS_BRIDGE", "NEEDS_MIXED_BRIDGE", "NOT_WORTH_RESCUING"]
BridgeTypeNeeded = Literal["NONE", "SIMPLE_REAL_BRIDGE", "SIMPLE_VIRTUAL_BRIDGE", "MIXED_COMPLEX_BRIDGE"]
TemplateSupportStatus = Literal["SUPPORTED", "UNSUPPORTED", "UNKNOWN"]
ConnectionType = Literal["DIRECT", "REAL_BRIDGE", "SIMPLE_VIRTUAL_BRIDGE", "MIXED_COMPLEX_BRIDGE", "INFEASIBLE"]


@dataclass
class AppendEvaluation:
    can_append: bool
    connection_type: ConnectionType
    pair_feasible: bool
    bridge_type_needed: BridgeTypeNeeded
    reason_code: str
    template_support_status: TemplateSupportStatus = "UNKNOWN"


@dataclass
class DonorCandidate:
    donor_block_id: str
    donor_campaign_skeleton_id: str
    receiver_campaign_skeleton_id: str
    line: str
    transferable_tons: float
    pair_feasible: bool
    duplicate_safe: bool
    rejection_reason: str

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class BridgeRequirement:
    line: str
    source_block_id: str
    target_block_id: str
    source_campaign_skeleton_id: str
    target_campaign_skeleton_id: str
    reason_code: str
    bridge_type_needed: BridgeTypeNeeded
    estimated_gap_closure_tons: float
    candidate_virtual_count: int
    mixed_bridge_required: bool
    template_support_status: TemplateSupportStatus

    def to_dict(self) -> Dict[str, Any]:
        out = dict(self.__dict__)
        out["source_skeleton_id"] = self.source_campaign_skeleton_id
        out["target_skeleton_id"] = self.target_campaign_skeleton_id
        return out


@dataclass
class BridgeMergeProposal:
    line: str
    proposal_id: str
    source_skeleton_id: str
    target_skeleton_id: str
    proposal_type: str
    bridge_type_needed: str
    projected_gap_before: float
    projected_gap_after: float
    projected_viability_before: str
    projected_viability_after: str
    confidence_band: str
    proposal_reason_code: str
    shadow_bridge_reason_code: str
    bridge_candidate_count: int
    source_block_ids: List[str] = field(default_factory=list)
    target_block_ids: List[str] = field(default_factory=list)
    selected_for_formal_trial: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class CampaignSkeleton:
    line: str
    skeleton_id: str
    block_ids: List[str]
    anchor_block_id: str
    planned_order_ids: List[str]
    current_real_tons: float
    target_ton_min: float
    target_ton_max: float
    viability_band: ViabilityBand
    assembly_status: AssemblyStatus
    bridge_need_level: str
    donor_need_level: str
    internal_bridge_needed: bool = False
    internal_bridge_type: str = "NONE"
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        out = dict(self.__dict__)
        out["block_ids"] = list(self.block_ids)
        out["planned_order_ids"] = list(self.planned_order_ids)
        return out


@dataclass
class LineAssemblyPlan:
    line: str
    campaign_skeletons: List[CampaignSkeleton] = field(default_factory=list)
    dropped_blocks: List[str] = field(default_factory=list)
    unresolved_blocks: List[str] = field(default_factory=list)
    donor_candidates: List[DonorCandidate] = field(default_factory=list)
    bridge_requirements: List[BridgeRequirement] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "line": self.line,
            "campaign_skeletons": [s.to_dict() for s in self.campaign_skeletons],
            "dropped_blocks": list(self.dropped_blocks),
            "unresolved_blocks": list(self.unresolved_blocks),
            "donor_candidates": [d.to_dict() for d in self.donor_candidates],
            "bridge_requirements": [b.to_dict() for b in self.bridge_requirements],
        }


@dataclass
class AssemblyPlan:
    plan_id: str
    line_plans: List[LineAssemblyPlan] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "line_plans": [lp.to_dict() for lp in self.line_plans],
            "summary": dict(self.summary),
            "diagnostics": dict(self.diagnostics),
        }

    def skeleton_rows(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        bridge_by_target: Dict[str, BridgeRequirement] = {}
        for lp in self.line_plans:
            for br in lp.bridge_requirements:
                bridge_by_target.setdefault(br.target_campaign_skeleton_id, br)
            for sk in lp.campaign_skeletons:
                br = bridge_by_target.get(sk.skeleton_id)
                rows.append(
                    {
                        "line": sk.line,
                        "skeleton_id": sk.skeleton_id,
                        "block_ids": ",".join(sk.block_ids),
                        "block_count": int(len(sk.block_ids)),
                        "anchor_block_id": sk.anchor_block_id,
                        "current_real_tons": float(round(sk.current_real_tons, 3)),
                        "viability_band": sk.viability_band,
                        "assembly_status": sk.assembly_status,
                        "internal_bridge_needed": bool(sk.internal_bridge_needed),
                        "internal_bridge_type": sk.internal_bridge_type,
                        "bridge_need_level": sk.bridge_need_level,
                        "bridge_type_needed": br.bridge_type_needed if br else "NONE",
                        "donor_need_level": sk.donor_need_level,
                        "notes": sk.notes,
                    }
                )
        return rows

    def bridge_rows(self) -> List[Dict[str, Any]]:
        return [br.to_dict() for lp in self.line_plans for br in lp.bridge_requirements]

    def donor_rows(self) -> List[Dict[str, Any]]:
        return [dc.to_dict() for lp in self.line_plans for dc in lp.donor_candidates]

    def bridge_merge_proposal_rows(self) -> List[Dict[str, Any]]:
        proposals = self.diagnostics.get("bridge_merge_proposal_rows", [])
        return [dict(r) for r in proposals if isinstance(r, dict)]


def _viability_band(tons: float, cfg: PlannerConfig) -> ViabilityBand:
    min_ton = float(cfg.rule.campaign_ton_min)
    gap = max(0.0, min_ton - float(tons))
    near_gap = float(getattr(cfg.model, "near_viable_gap_tons", 80.0) or 80.0)
    merge_gap = float(getattr(cfg.model, "merge_candidate_gap_tons", 250.0) or 250.0)
    if gap <= 0.0:
        return "ALREADY_VIABLE"
    if gap <= near_gap:
        return "NEAR_VIABLE"
    if gap <= merge_gap:
        return "MERGE_CANDIDATE"
    return "HOPELESS_UNDERFILLED"


def _block_boundary_pair_feasible(left: CandidateBlock, right: CandidateBlock, cfg: PlannerConfig) -> bool:
    left_sig = left.tail_signature or {}
    right_sig = right.head_signature or {}
    try:
        left_w = float(left_sig.get("width", 0.0) or 0.0)
        right_w = float(right_sig.get("width", 0.0) or 0.0)
        if right_w > left_w:
            return False
        if left_w - right_w > float(cfg.rule.max_width_drop):
            return False
        left_th = float(left_sig.get("thickness", 0.0) or 0.0)
        right_th = float(right_sig.get("thickness", 0.0) or 0.0)
        if right_th < left_th:
            return False
        left_group = str(left_sig.get("steel_group", "") or "")
        right_group = str(right_sig.get("steel_group", "") or "")
        if left_group and right_group and left_group != right_group:
            return False
    except Exception:
        return False
    return True


def _status_for_skeleton(sk: CampaignSkeleton) -> AssemblyStatus:
    if sk.viability_band == "ALREADY_VIABLE":
        return "STABLE"
    if sk.viability_band in {"NEAR_VIABLE", "MERGE_CANDIDATE"}:
        return "NEEDS_DONOR"
    return "NOT_WORTH_RESCUING"


def _iter_template_rows(transition_pack: dict) -> List[Dict[str, Any]]:
    """Return template edge rows from the known transition/template pack shapes."""
    if not isinstance(transition_pack, dict):
        return []
    candidates = []
    for key in ("templates", "template_df", "transition_df", "edges", "edge_df"):
        value = transition_pack.get(key)
        if value is None:
            continue
        candidates.append(value)
    for value in transition_pack.values():
        if isinstance(value, dict):
            for key in ("templates", "template_df", "transition_df", "edges", "edge_df"):
                nested = value.get(key)
                if nested is not None:
                    candidates.append(nested)

    rows: List[Dict[str, Any]] = []
    for value in candidates:
        if isinstance(value, pd.DataFrame):
            rows.extend(dict(r) for r in value.to_dict("records"))
        elif isinstance(value, list):
            rows.extend(dict(r) for r in value if isinstance(r, dict))
        elif isinstance(value, dict):
            rows.extend(dict(r) for r in value.values() if isinstance(r, dict))
    return rows


def _edge_lookup_from_transition_pack(transition_pack: dict) -> Dict[Tuple[str, str, str], set[str]]:
    lookup: Dict[Tuple[str, str, str], set[str]] = {}
    for row in _iter_template_rows(transition_pack):
        edge_type = str(row.get("edge_type", row.get("selected_edge_type", "")) or "")
        if not edge_type:
            continue
        line = str(row.get("line", row.get("assigned_line", "")) or "")
        left = str(row.get("from_order_id", row.get("source_order_id", row.get("from_id", ""))) or "")
        right = str(row.get("to_order_id", row.get("target_order_id", row.get("to_id", ""))) or "")
        if not line or not left or not right:
            continue
        lookup.setdefault((line, left, right), set()).add(edge_type)
    return lookup


def _evaluate_direct_append(
    left: CandidateBlock,
    right: CandidateBlock,
    edge_lookup: Dict[Tuple[str, str, str], set[str]],
    cfg: PlannerConfig,
) -> AppendEvaluation:
    line = str(right.line or left.line)
    key = (line, str(left.tail_order_id), str(right.head_order_id))
    edge_types = edge_lookup.get(key, set())
    hard_direct_ok = _block_boundary_pair_feasible(left, right, cfg)
    if hard_direct_ok:
        # Diagnostics-only planner grouping is allowed to recognize hard-feasible
        # direct adjacency even when template payload is incomplete. Final landing
        # remains protected by fallback until the planner maturity gate is opened.
        support = "SUPPORTED" if "DIRECT_EDGE" in edge_types or not edge_lookup else "UNKNOWN"
        return AppendEvaluation(True, "DIRECT", True, "NONE", "DIRECT_OK", support)
    if edge_lookup and "DIRECT_EDGE" not in edge_types:
        return AppendEvaluation(False, "INFEASIBLE", False, "NONE", "DIRECT_TEMPLATE_UNSUPPORTED", "UNSUPPORTED")
    return AppendEvaluation(False, "INFEASIBLE", False, "NONE", "DIRECT_PAIR_INVALID", "UNKNOWN")


def _evaluate_real_bridge_append(
    left: CandidateBlock,
    right: CandidateBlock,
    edge_lookup: Dict[Tuple[str, str, str], set[str]],
) -> AppendEvaluation:
    line = str(right.line or left.line)
    key = (line, str(left.tail_order_id), str(right.head_order_id))
    edge_types = edge_lookup.get(key, set())
    if "REAL_BRIDGE_EDGE" in edge_types:
        return AppendEvaluation(True, "REAL_BRIDGE", True, "SIMPLE_REAL_BRIDGE", "REAL_BRIDGE_OK", "SUPPORTED")
    if edge_lookup:
        return AppendEvaluation(False, "INFEASIBLE", False, "SIMPLE_REAL_BRIDGE", "REAL_BRIDGE_TEMPLATE_UNSUPPORTED", "UNSUPPORTED")
    return AppendEvaluation(False, "INFEASIBLE", False, "SIMPLE_REAL_BRIDGE", "REAL_BRIDGE_UNSUPPORTED", "UNKNOWN")


def _evaluate_block_to_skeleton_connection(
    skeleton_block_ids: List[str],
    next_block: CandidateBlock,
    block_by_id: Dict[str, CandidateBlock],
    edge_lookup: Dict[Tuple[str, str, str], set[str]],
    cfg: PlannerConfig,
) -> AppendEvaluation:
    if not skeleton_block_ids:
        return AppendEvaluation(True, "DIRECT", True, "NONE", "DIRECT_OK", "SUPPORTED")
    left = block_by_id.get(skeleton_block_ids[-1])
    right = next_block
    if left is None or right is None:
        return AppendEvaluation(False, "INFEASIBLE", False, "NONE", "PAIR_INFEASIBLE", "UNKNOWN")
    direct = _evaluate_direct_append(left, right, edge_lookup, cfg)
    if direct.can_append:
        return direct
    real_bridge = _evaluate_real_bridge_append(left, right, edge_lookup)
    if real_bridge.can_append:
        return real_bridge
    line = str(right.line or left.line)
    key = (line, str(left.tail_order_id), str(right.head_order_id))
    edge_types = edge_lookup.get(key, set())
    if "VIRTUAL_BRIDGE_EDGE" in edge_types or "VIRTUAL_BRIDGE_FAMILY_EDGE" in edge_types:
        return AppendEvaluation(
            False,
            "SIMPLE_VIRTUAL_BRIDGE",
            False,
            "SIMPLE_VIRTUAL_BRIDGE",
            "NEEDS_SIMPLE_VIRTUAL_BRIDGE",
            "SUPPORTED",
        )
    left_group = str((left.tail_signature or {}).get("steel_group", "") or "")
    right_group = str((right.head_signature or {}).get("steel_group", "") or "")
    if left_group and right_group and left_group != right_group:
        return AppendEvaluation(
            False,
            "MIXED_COMPLEX_BRIDGE",
            False,
            "MIXED_COMPLEX_BRIDGE",
            "NEEDS_MIXED_COMPLEX_BRIDGE",
            "UNSUPPORTED",
        )
    return AppendEvaluation(False, "INFEASIBLE", False, "NONE", "TEMPLATE_UNSUPPORTED", "UNSUPPORTED")



def _bridge_requirement_for_boundary(
    line: str,
    left: CandidateBlock,
    right: CandidateBlock,
    source_skeleton_id: str,
    target_skeleton_id: str,
    cfg: PlannerConfig,
) -> BridgeRequirement:
    left_sig = left.tail_signature or {}
    right_sig = right.head_signature or {}
    left_group = str(left_sig.get("steel_group", "") or "")
    right_group = str(right_sig.get("steel_group", "") or "")
    left_w = float(left_sig.get("width", 0.0) or 0.0)
    right_w = float(right_sig.get("width", 0.0) or 0.0)
    left_th = float(left_sig.get("thickness", 0.0) or 0.0)
    right_th = float(right_sig.get("thickness", 0.0) or 0.0)
    mixed = False
    bridge_type: BridgeTypeNeeded = "SIMPLE_VIRTUAL_BRIDGE"
    reason = "PAIR_INVALID_NEEDS_BRIDGE"
    if left_group and right_group and left_group != right_group:
        mixed = True
        bridge_type = "MIXED_COMPLEX_BRIDGE"
        reason = "GROUP_TRANSITION_REQUIRES_MIXED_BRIDGE"
    elif right_w > left_w or (left_w - right_w) > float(cfg.rule.max_width_drop):
        bridge_type = "SIMPLE_VIRTUAL_BRIDGE"
        reason = "WIDTH_BOUNDARY_REQUIRES_BRIDGE"
    elif right_th < left_th:
        bridge_type = "SIMPLE_VIRTUAL_BRIDGE"
        reason = "THICKNESS_BOUNDARY_REQUIRES_BRIDGE"
    else:
        bridge_type = "SIMPLE_REAL_BRIDGE"
        reason = "BOUNDARY_REQUIRES_REAL_BRIDGE"
    return BridgeRequirement(
        line=str(line),
        source_block_id=str(left.block_id),
        target_block_id=str(right.block_id),
        source_campaign_skeleton_id=source_skeleton_id,
        target_campaign_skeleton_id=target_skeleton_id,
        reason_code=reason,
        bridge_type_needed=bridge_type,
        estimated_gap_closure_tons=float(round(float(right.total_tons or 0.0), 3)),
        candidate_virtual_count=1 if bridge_type != "NONE" else 0,
        mixed_bridge_required=bool(mixed),
        template_support_status="UNKNOWN",
    )


def _build_skeleton_from_blocks(
    line: str,
    skeleton_id: str,
    block_ids: List[str],
    block_by_id: Dict[str, CandidateBlock],
    min_ton: float,
    max_ton: float,
    cfg: PlannerConfig,
    notes: str = "",
    internal_bridge_needed: bool = False,
    internal_bridge_type: str = "NONE",
) -> CampaignSkeleton:
    blocks = [block_by_id[bid] for bid in block_ids if bid in block_by_id]
    order_ids: List[str] = []
    seen_orders: set[str] = set()
    for block in blocks:
        for oid in block.order_ids:
            soid = str(oid)
            if soid and soid not in seen_orders:
                seen_orders.add(soid)
                order_ids.append(soid)
    tons = float(sum(float(b.total_tons or 0.0) for b in blocks))
    band = _viability_band(tons, cfg)
    sk = CampaignSkeleton(
        line=str(line),
        skeleton_id=skeleton_id,
        block_ids=list(block_ids),
        anchor_block_id=str(block_ids[0]) if block_ids else "",
        planned_order_ids=order_ids,
        current_real_tons=tons,
        target_ton_min=min_ton,
        target_ton_max=max_ton,
        viability_band=band,
        assembly_status="STABLE",
        bridge_need_level="NONE",
        donor_need_level="NONE" if band == "ALREADY_VIABLE" else "NEEDS_DONOR",
        internal_bridge_needed=bool(internal_bridge_needed),
        internal_bridge_type=str(internal_bridge_type or "NONE"),
        notes=notes,
    )
    sk.assembly_status = _status_for_skeleton(sk)
    return sk


def _segment_slot_into_skeletons(
    line: str,
    slot_campaign_id: str,
    block_ids: List[str],
    block_by_id: Dict[str, CandidateBlock],
    min_ton: float,
    max_ton: float,
    cfg: PlannerConfig,
) -> tuple[List[CampaignSkeleton], List[BridgeRequirement]]:
    if not block_ids:
        return [], []
    skeletons: List[CampaignSkeleton] = []
    bridges: List[BridgeRequirement] = []
    current_ids: List[str] = [block_ids[0]]
    seg_idx = 1
    for bid in block_ids[1:]:
        prev_block = block_by_id.get(current_ids[-1])
        next_block = block_by_id.get(bid)
        feasible = prev_block is not None and next_block is not None and _block_boundary_pair_feasible(prev_block, next_block, cfg)
        if feasible:
            current_ids.append(bid)
            continue
        skeleton_id = f"{slot_campaign_id}__sk{seg_idx}" if len(block_ids) > 1 else slot_campaign_id
        current_sk = _build_skeleton_from_blocks(line, skeleton_id, current_ids, block_by_id, min_ton, max_ton, cfg)
        skeletons.append(current_sk)
        seg_idx += 1
        next_skeleton_id = f"{slot_campaign_id}__sk{seg_idx}" if len(block_ids) > 1 else slot_campaign_id
        if prev_block is not None and next_block is not None:
            bridges.append(_bridge_requirement_for_boundary(line, prev_block, next_block, current_sk.skeleton_id, next_skeleton_id, cfg))
        current_ids = [bid]
    final_skeleton_id = f"{slot_campaign_id}__sk{seg_idx}" if len(block_ids) > 1 else slot_campaign_id
    skeletons.append(_build_skeleton_from_blocks(line, final_skeleton_id, current_ids, block_by_id, min_ton, max_ton, cfg))
    return skeletons, bridges


def _build_line_skeletons(
    line: str,
    block_ids: List[str],
    block_by_id: Dict[str, CandidateBlock],
    edge_lookup: Dict[Tuple[str, str, str], set[str]],
    min_ton: float,
    max_ton: float,
    cfg: PlannerConfig,
) -> tuple[List[CampaignSkeleton], List[BridgeRequirement], Dict[str, Any]]:
    """Group same-line blocks into campaign skeletons before viability tagging.

    DIRECT and REAL_BRIDGE connections are allowed to form one skeleton. Virtual
    and mixed-complex needs are emitted as structured bridge requirements.
    """
    skeletons: List[CampaignSkeleton] = []
    bridges: List[BridgeRequirement] = []
    diagnostics = {
        "grouping_direct_append_count": 0,
        "grouping_real_bridge_append_count": 0,
        "direct_group_merges": 0,
        "real_bridge_group_merges": 0,
        "direct_group_template_unknown_count": 0,
        "real_bridge_group_template_unsupported_count": 0,
        "grouping_virtual_boundary_count": 0,
        "grouping_mixed_boundary_count": 0,
        "grouping_infeasible_boundary_count": 0,
    }
    ids = [str(bid) for bid in block_ids if str(bid) in block_by_id]
    if not ids:
        return skeletons, bridges, diagnostics

    current_ids: List[str] = [ids[0]]
    current_internal_bridge_needed = False
    current_internal_bridge_types: List[str] = []
    seg_idx = 1

    def _flush(notes: str = "") -> CampaignSkeleton:
        sk = _build_skeleton_from_blocks(
            line=str(line),
            skeleton_id=f"{line}__sk{seg_idx}",
            block_ids=current_ids,
            block_by_id=block_by_id,
            min_ton=min_ton,
            max_ton=max_ton,
            cfg=cfg,
            notes=notes,
            internal_bridge_needed=current_internal_bridge_needed,
            internal_bridge_type=",".join(sorted(set(current_internal_bridge_types))) if current_internal_bridge_types else "NONE",
        )
        skeletons.append(sk)
        return sk

    for bid in ids[1:]:
        next_block = block_by_id[bid]
        eval_result = _evaluate_block_to_skeleton_connection(current_ids, next_block, block_by_id, edge_lookup, cfg)
        if eval_result.can_append:
            current_ids.append(bid)
            if eval_result.connection_type == "DIRECT":
                diagnostics["grouping_direct_append_count"] += 1
                diagnostics["direct_group_merges"] += 1
                if eval_result.template_support_status == "UNKNOWN":
                    diagnostics["direct_group_template_unknown_count"] += 1
            elif eval_result.connection_type == "REAL_BRIDGE":
                diagnostics["grouping_real_bridge_append_count"] += 1
                diagnostics["real_bridge_group_merges"] += 1
                current_internal_bridge_needed = True
                current_internal_bridge_types.append("REAL_BRIDGE")
            continue

        current_sk = _flush(notes=f"group_boundary={eval_result.reason_code}")
        seg_idx += 1
        next_skeleton_id = f"{line}__sk{seg_idx}"
        left_block = block_by_id.get(current_ids[-1])
        right_block = next_block
        if left_block is not None and right_block is not None:
            br = _bridge_requirement_for_boundary(
                line=str(line),
                left=left_block,
                right=right_block,
                source_skeleton_id=current_sk.skeleton_id,
                target_skeleton_id=next_skeleton_id,
                cfg=cfg,
            )
            if eval_result.bridge_type_needed != "NONE":
                br.bridge_type_needed = eval_result.bridge_type_needed
                br.mixed_bridge_required = eval_result.bridge_type_needed == "MIXED_COMPLEX_BRIDGE"
                br.reason_code = eval_result.reason_code
                br.template_support_status = eval_result.template_support_status
            bridges.append(br)
            if br.bridge_type_needed == "SIMPLE_VIRTUAL_BRIDGE":
                diagnostics["grouping_virtual_boundary_count"] += 1
            elif br.bridge_type_needed == "MIXED_COMPLEX_BRIDGE":
                diagnostics["grouping_mixed_boundary_count"] += 1
            elif br.reason_code == "REAL_BRIDGE_TEMPLATE_UNSUPPORTED":
                diagnostics["real_bridge_group_template_unsupported_count"] += 1
            elif br.bridge_type_needed == "NONE":
                diagnostics["grouping_infeasible_boundary_count"] += 1
        current_ids = [bid]
        current_internal_bridge_needed = False
        current_internal_bridge_types = []

    _flush(notes="grouped_by_line")
    return skeletons, bridges, diagnostics


def _judge_plan_maturity(skeletons: List[CampaignSkeleton], selected_block_count: int, cfg: PlannerConfig) -> Dict[str, Any]:
    """Conservative gate: immature planner output stays diagnostics-only."""
    skeleton_count = int(len(skeletons))
    selected_count = int(selected_block_count or 0)
    total_blocks = sum(len(s.block_ids) for s in skeletons)
    avg_blocks = float(round(total_blocks / max(1, skeleton_count), 3))
    hopeless_count = sum(1 for s in skeletons if s.viability_band == "HOPELESS_UNDERFILLED")
    near_merge_count = sum(1 for s in skeletons if s.viability_band in {"NEAR_VIABLE", "MERGE_CANDIDATE"})
    min_tons = float(getattr(cfg.rule, "campaign_ton_min", 700.0) or 700.0)
    single_under_min = sum(1 for s in skeletons if len(s.block_ids) == 1 and float(s.current_real_tons or 0.0) < min_tons)
    hopeless_ratio = float(round(hopeless_count / max(1, skeleton_count), 4))
    near_merge_ratio = float(round(near_merge_count / max(1, skeleton_count), 4))
    single_under_min_ratio = float(round(single_under_min / max(1, skeleton_count), 4))
    ready = True
    reason = "PLAN_READY"
    if skeleton_count <= 0 or selected_count <= 0:
        ready = False
        reason = "INSUFFICIENT_GROUPING_GAIN"
    elif single_under_min_ratio >= 0.7:
        ready = False
        reason = "MOSTLY_SINGLE_BLOCK_SKELETONS"
    elif avg_blocks <= 1.1:
        ready = False
        reason = "TOO_FRAGMENTED"
    elif skeleton_count >= 0.9 * max(1, selected_count):
        ready = False
        reason = "INSUFFICIENT_GROUPING_GAIN"
    elif hopeless_ratio >= 0.7 and near_merge_ratio <= 0.2:
        ready = False
        reason = "TOO_MANY_HOPELESS_SKELETONS"
    return {
        "assembly_plan_ready_for_finalization": bool(ready),
        "planner_ready_for_finalization": bool(ready),
        "assembly_plan_readiness_reason_code": reason,
        "readiness_reason_code": reason,
        "assembly_plan_avg_blocks_per_skeleton": avg_blocks,
        "assembly_plan_skeleton_count": skeleton_count,
        "assembly_plan_hopeless_ratio": hopeless_ratio,
        "assembly_plan_near_merge_ratio": near_merge_ratio,
        "assembly_plan_single_under_min_ratio": single_under_min_ratio,
        "assembly_plan_fallback_used": bool(not ready),
    }


def _band_from_gap(gap: float, cfg: PlannerConfig) -> ViabilityBand:
    min_ton = float(cfg.rule.campaign_ton_min)
    return _viability_band(max(0.0, min_ton - max(0.0, float(gap or 0.0))), cfg)


def _proposal_type_from_band(after_band: str, gain: float) -> str:
    if gain <= 0.0:
        return "NOT_PROPOSABLE"
    if after_band == "ALREADY_VIABLE":
        return "BRIDGE_TO_ALREADY_VIABLE"
    if after_band == "NEAR_VIABLE":
        return "BRIDGE_TO_NEAR_VIABLE"
    if after_band == "MERGE_CANDIDATE":
        return "BRIDGE_TO_MERGE_CANDIDATE"
    return "NOT_PROPOSABLE"


def _proposal_confidence(after_band: str, bridge_type: str, gain: float) -> str:
    if gain <= 0.0:
        return "LOW"
    if after_band == "ALREADY_VIABLE" and bridge_type != "MIXED_COMPLEX_BRIDGE":
        return "HIGH"
    if after_band in {"ALREADY_VIABLE", "NEAR_VIABLE", "MERGE_CANDIDATE"}:
        return "MEDIUM"
    return "LOW"


def _proposal_reason(after_band: str, bridge_type: str, gain: float) -> str:
    if gain <= 0.0:
        return "SHADOW_BRIDGE_NO_GAIN"
    if bridge_type == "MIXED_COMPLEX_BRIDGE":
        return "REQUIRES_COMPLEX_MIXED_BRIDGE"
    if after_band in {"ALREADY_VIABLE", "NEAR_VIABLE"}:
        return "SHADOW_BRIDGE_STRONG_GAIN"
    if after_band == "MERGE_CANDIDATE":
        return "SHADOW_BRIDGE_WEAK_GAIN"
    return "TEMPLATE_SUPPORT_UNCLEAR"


def _safe_id_part(value: Any) -> str:
    text = str(value or "").strip()
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in text) or "NA"


def _stable_bridge_merge_proposal_id(
    line: str,
    source_skeleton_id: str,
    target_skeleton_id: str,
    proposal_type: str,
    bridge_type_needed: str,
) -> str:
    return "__".join(
        [
            "bridge_merge",
            _safe_id_part(line),
            _safe_id_part(source_skeleton_id),
            _safe_id_part(target_skeleton_id),
            _safe_id_part(proposal_type),
            _safe_id_part(bridge_type_needed),
        ]
    )


def _viability_rank(band: str) -> int:
    ranks = {
        "HOPELESS_UNDERFILLED": 0,
        "MERGE_CANDIDATE": 1,
        "NEAR_VIABLE": 2,
        "ALREADY_VIABLE": 3,
    }
    return ranks.get(str(band), -1)


def _build_bridge_merge_proposals(
    skeletons: List[CampaignSkeleton],
    bridges: List[BridgeRequirement],
    cfg: PlannerConfig,
) -> List[BridgeMergeProposal]:
    sk_by_id = {s.skeleton_id: s for s in skeletons}
    min_ton = float(cfg.rule.campaign_ton_min)
    proposals: List[BridgeMergeProposal] = []
    id_seen: Dict[str, int] = {}
    for idx, br in enumerate(bridges, start=1):
        target = sk_by_id.get(br.target_campaign_skeleton_id)
        source = sk_by_id.get(br.source_campaign_skeleton_id)
        if target is None:
            continue
        gap_before = max(0.0, min_ton - float(target.current_real_tons or 0.0))
        source_tons = float(source.current_real_tons or 0.0) if source is not None else float(br.estimated_gap_closure_tons or 0.0)
        closure = max(float(br.estimated_gap_closure_tons or 0.0), source_tons)
        gap_after = max(0.0, gap_before - closure)
        before_band = target.viability_band
        after_band = _band_from_gap(gap_after, cfg)
        gain = gap_before - gap_after
        proposal_type = _proposal_type_from_band(after_band, gain)
        bridge_type = str(br.bridge_type_needed or "UNKNOWN")
        confidence = _proposal_confidence(after_band, bridge_type, gain)
        reason = _proposal_reason(after_band, bridge_type, gain)
        base_id = _stable_bridge_merge_proposal_id(
            str(br.line),
            str(br.source_campaign_skeleton_id),
            str(br.target_campaign_skeleton_id),
            proposal_type,
            bridge_type,
        )
        dup_no = id_seen.get(base_id, 0) + 1
        id_seen[base_id] = dup_no
        proposal_id = base_id if dup_no == 1 else f"{base_id}__{dup_no}"
        proposals.append(
            BridgeMergeProposal(
                line=str(br.line),
                proposal_id=proposal_id,
                source_skeleton_id=str(br.source_campaign_skeleton_id),
                target_skeleton_id=str(br.target_campaign_skeleton_id),
                proposal_type=proposal_type,
                bridge_type_needed=bridge_type,
                projected_gap_before=float(round(gap_before, 3)),
                projected_gap_after=float(round(gap_after, 3)),
                projected_viability_before=str(before_band),
                projected_viability_after=str(after_band),
                confidence_band=confidence,
                proposal_reason_code=reason,
                shadow_bridge_reason_code="PLANNER_SIDE_SHADOW_BRIDGE_HINT",
                bridge_candidate_count=int(max(1, br.candidate_virtual_count or 0)),
                source_block_ids=list(source.block_ids) if source is not None else [],
                target_block_ids=list(target.block_ids) if target is not None else [],
            )
        )
    return proposals


def _select_best_bridge_merge_proposal(proposals: List[BridgeMergeProposal]) -> BridgeMergeProposal | None:
    candidates = [p for p in proposals if p.proposal_type != "NOT_PROPOSABLE"]
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda p: (
            0 if p.confidence_band == "HIGH" else 1 if p.confidence_band == "MEDIUM" else 2,
            -_viability_rank(p.projected_viability_after),
            p.projected_gap_after,
            -(p.projected_gap_before - p.projected_gap_after),
        ),
    )[0]


def _assert_bridge_proposal_simulation_consistency(
    best_proposal: BridgeMergeProposal | None,
    simulation_row: Dict[str, Any] | None,
) -> tuple[bool, str]:
    if best_proposal is None or not simulation_row:
        return True, "NONE"
    if str(simulation_row.get("proposal_id", "")) != str(best_proposal.proposal_id):
        return False, "PROPOSAL_ID_MISMATCH"
    if (
        str(simulation_row.get("source_skeleton_id", "")) != str(best_proposal.source_skeleton_id)
        or str(simulation_row.get("target_skeleton_id", "")) != str(best_proposal.target_skeleton_id)
    ):
        return False, "SOURCE_TARGET_MISMATCH"
    try:
        before_a = float(simulation_row.get("projected_gap_before", 0.0) or 0.0)
        before_b = float(best_proposal.projected_gap_before)
        after_a = float(simulation_row.get("projected_gap_after", 0.0) or 0.0)
        after_b = float(best_proposal.projected_gap_after)
        if abs(before_a - before_b) > 1e-6 or abs(after_a - after_b) > 1e-6:
            return False, "PROJECTED_METRIC_MISMATCH"
    except Exception:
        return False, "PROJECTED_METRIC_MISMATCH"
    if str(simulation_row.get("projected_viability_after", "")) != str(best_proposal.projected_viability_after):
        return False, "PROJECTED_METRIC_MISMATCH"
    return True, "NONE"


def _simulate_best_bridge_merge(
    best: BridgeMergeProposal | None,
    skeletons: List[CampaignSkeleton],
    cfg: PlannerConfig,
) -> Dict[str, Any]:
    base_blocks = sum(len(s.block_ids) for s in skeletons)
    if best is None:
        return {
            "bridge_merge_simulation_enabled": True,
            "best_bridge_merge_proposal_selected": False,
            "best_bridge_merge_proposal_id": "",
            "best_bridge_merge_source": "",
            "best_bridge_merge_target": "",
            "best_bridge_merge_confidence_band": "",
            "best_bridge_merge_projected_gap_before": 0.0,
            "best_bridge_merge_projected_gap_after": 0.0,
            "best_bridge_merge_projected_viability_after": "",
            "simulation_proposal_id": "",
            "simulation_source": "",
            "simulation_target": "",
            "simulation_projected_gap_before": 0.0,
            "simulation_projected_gap_after": 0.0,
            "simulation_projected_viability_after": "",
            "proposal_simulation_consistent": True,
            "proposal_simulation_inconsistency_reason": "NONE",
            "simulated_merge_guard_passed": False,
            "simulated_merge_guard_reason": "NO_PROPOSAL",
            "simulated_merged_block_count": 0,
            "simulated_merged_real_tons_estimate": 0.0,
            "simulated_merged_gap_after": 0.0,
            "simulated_merged_viability_after": "",
            "simulated_skeleton_count": int(len(skeletons)),
            "simulated_avg_blocks_per_skeleton": float(round(base_blocks / max(1, len(skeletons)), 3)),
            "simulated_hopeless_count": int(sum(1 for s in skeletons if s.viability_band == "HOPELESS_UNDERFILLED")),
            "simulated_near_merge_count": int(sum(1 for s in skeletons if s.viability_band in {"NEAR_VIABLE", "MERGE_CANDIDATE"})),
            "bridge_merge_simulation_rows": [],
        }

    sk_by_id = {s.skeleton_id: s for s in skeletons}
    source = sk_by_id.get(best.source_skeleton_id)
    target = sk_by_id.get(best.target_skeleton_id)
    min_tons = float(getattr(cfg.rule, "campaign_ton_min", 700.0) or 700.0)
    guard_passed = True
    guard_reason = "SIMULATION_READY"
    if best.proposal_type == "NOT_PROPOSABLE":
        guard_passed = False
        guard_reason = "NO_VIABILITY_IMPROVEMENT"
    elif best.confidence_band == "LOW":
        guard_passed = False
        guard_reason = "LOW_CONFIDENCE"
    elif not (best.projected_gap_after < best.projected_gap_before):
        guard_passed = False
        guard_reason = "NO_GAP_IMPROVEMENT"
    elif _viability_rank(best.projected_viability_after) < _viability_rank(best.projected_viability_before):
        guard_passed = False
        guard_reason = "NO_VIABILITY_IMPROVEMENT"
    elif best.bridge_type_needed == "MIXED_COMPLEX_BRIDGE":
        guard_passed = False
        guard_reason = "REQUIRES_COMPLEX_MIXED_BRIDGE"
    elif best.proposal_reason_code == "TEMPLATE_SUPPORT_UNCLEAR" or best.bridge_type_needed == "UNKNOWN":
        guard_passed = False
        guard_reason = "TEMPLATE_SUPPORT_UNCLEAR"

    merged_block_count = int((len(source.block_ids) if source else 0) + (len(target.block_ids) if target else 0))
    merged_tons = float((source.current_real_tons if source else 0.0) + (target.current_real_tons if target else 0.0))
    merged_gap = max(0.0, min_tons - merged_tons)
    merged_band = _viability_band(merged_tons, cfg)
    touched = {best.source_skeleton_id, best.target_skeleton_id}
    remaining = [s for s in skeletons if s.skeleton_id not in touched]
    simulated_skeleton_count = int(len(remaining) + (1 if source is not None or target is not None else 0))
    simulated_blocks = int(sum(len(s.block_ids) for s in remaining) + merged_block_count)
    simulated_hopeless = int(sum(1 for s in remaining if s.viability_band == "HOPELESS_UNDERFILLED") + (1 if merged_band == "HOPELESS_UNDERFILLED" else 0))
    simulated_near_merge = int(sum(1 for s in remaining if s.viability_band in {"NEAR_VIABLE", "MERGE_CANDIDATE"}) + (1 if merged_band in {"NEAR_VIABLE", "MERGE_CANDIDATE"} else 0))
    row = {
        "proposal_id": best.proposal_id,
        "source_skeleton_id": best.source_skeleton_id,
        "target_skeleton_id": best.target_skeleton_id,
        "confidence_band": best.confidence_band,
        "projected_gap_before": float(best.projected_gap_before),
        "projected_gap_after": float(best.projected_gap_after),
        "projected_viability_after": best.projected_viability_after,
        "simulated_merge_guard_passed": bool(guard_passed),
        "simulated_merge_guard_reason": guard_reason,
        "simulated_skeleton_count": simulated_skeleton_count,
        "simulated_avg_blocks_per_skeleton": float(round(simulated_blocks / max(1, simulated_skeleton_count), 3)),
        "simulated_hopeless_count": simulated_hopeless,
        "simulated_near_merge_count": simulated_near_merge,
    }
    consistent, inconsistency_reason = _assert_bridge_proposal_simulation_consistency(best, row)
    if not consistent:
        guard_passed = False
        guard_reason = inconsistency_reason
        row["simulated_merge_guard_passed"] = False
        row["simulated_merge_guard_reason"] = guard_reason
    row["proposal_simulation_consistent"] = bool(consistent)
    row["proposal_simulation_inconsistency_reason"] = inconsistency_reason
    return {
        "bridge_merge_simulation_enabled": True,
        "best_bridge_merge_proposal_selected": True,
        "best_bridge_merge_proposal_id": best.proposal_id,
        "best_bridge_merge_source": best.source_skeleton_id,
        "best_bridge_merge_target": best.target_skeleton_id,
        "best_bridge_merge_confidence_band": best.confidence_band,
        "best_bridge_merge_projected_gap_before": float(best.projected_gap_before),
        "best_bridge_merge_projected_gap_after": float(best.projected_gap_after),
        "best_bridge_merge_projected_viability_after": best.projected_viability_after,
        "simulation_proposal_id": best.proposal_id,
        "simulation_source": best.source_skeleton_id,
        "simulation_target": best.target_skeleton_id,
        "simulation_projected_gap_before": float(best.projected_gap_before),
        "simulation_projected_gap_after": float(best.projected_gap_after),
        "simulation_projected_viability_after": best.projected_viability_after,
        "proposal_simulation_consistent": bool(consistent),
        "proposal_simulation_inconsistency_reason": inconsistency_reason,
        "simulated_merge_guard_passed": bool(guard_passed),
        "simulated_merge_guard_reason": guard_reason,
        "simulated_merged_block_count": merged_block_count,
        "simulated_merged_real_tons_estimate": float(round(merged_tons, 3)),
        "simulated_merged_gap_after": float(round(merged_gap, 3)),
        "simulated_merged_viability_after": str(merged_band),
        "simulated_skeleton_count": simulated_skeleton_count,
        "simulated_avg_blocks_per_skeleton": row["simulated_avg_blocks_per_skeleton"],
        "simulated_hopeless_count": simulated_hopeless,
        "simulated_near_merge_count": simulated_near_merge,
        "bridge_merge_simulation_rows": [row],
    }


def _summarize_bridge_merge_proposals(
    proposals: List[BridgeMergeProposal],
    best_p: BridgeMergeProposal | None,
) -> Dict[str, Any]:
    rows = [p.to_dict() for p in proposals]
    high = [p for p in proposals if p.confidence_band == "HIGH"]
    to_viable = [p for p in proposals if p.proposal_type == "BRIDGE_TO_ALREADY_VIABLE"]
    to_near = [p for p in proposals if p.proposal_type == "BRIDGE_TO_NEAR_VIABLE"]
    to_merge = [p for p in proposals if p.proposal_type == "BRIDGE_TO_MERGE_CANDIDATE"]
    mixed = [p for p in proposals if p.bridge_type_needed == "MIXED_COMPLEX_BRIDGE"]
    not_prop = [p for p in proposals if p.proposal_type == "NOT_PROPOSABLE"]
    return {
        "bridge_merge_proposal_count": int(len(proposals)),
        "bridge_merge_proposal_high_confidence_count": int(len(high)),
        "bridge_merge_to_already_viable_count": int(len(to_viable)),
        "bridge_merge_to_near_viable_count": int(len(to_near)),
        "bridge_merge_to_merge_candidate_count": int(len(to_merge)),
        "bridge_merge_requires_mixed_complex_count": int(len(mixed)),
        "bridge_merge_not_proposable_count": int(len(not_prop)),
        "best_bridge_merge_proposal_id": best_p.proposal_id if best_p else "",
        "best_bridge_merge_proposal_line": best_p.line if best_p else "",
        "best_bridge_merge_proposal_source": best_p.source_skeleton_id if best_p else "",
        "best_bridge_merge_proposal_target": best_p.target_skeleton_id if best_p else "",
        "best_bridge_merge_projected_gap_before": float(best_p.projected_gap_before) if best_p else 0.0,
        "best_bridge_merge_projected_gap_after": float(best_p.projected_gap_after) if best_p else 0.0,
        "best_bridge_merge_projected_viability_after": best_p.projected_viability_after if best_p else "",
        "bridge_merge_proposal_rows": rows,
    }


def build_campaign_assembly_plan(
    master_result: BlockMasterResult,
    orders_df: pd.DataFrame,
    transition_pack: dict,
    cfg: PlannerConfig,
) -> AssemblyPlan:
    """Build a conservative campaign assembly plan from selected blocks.

    The planner intentionally does not rewrite selected blocks.  It surfaces the
    current campaign skeleton topology and explains whether underfilled
    skeletons have donor or bridge routes before realization.
    """
    _ = orders_df
    block_by_id = {str(b.block_id): b for b in master_result.selected_blocks}
    edge_lookup = _edge_lookup_from_transition_pack(transition_pack)
    min_ton = float(cfg.rule.campaign_ton_min)
    max_ton = float(cfg.rule.campaign_ton_max)
    line_plans: List[LineAssemblyPlan] = []

    blocks_by_line: Dict[str, List[str]] = {}
    if master_result.block_order_by_line:
        for line, block_ids in master_result.block_order_by_line.items():
            clean_ids = [str(bid) for bid in block_ids if str(bid) in block_by_id]
            if clean_ids:
                blocks_by_line.setdefault(str(line), []).extend(clean_ids)
    if not blocks_by_line:
        for block in master_result.selected_blocks:
            blocks_by_line.setdefault(str(block.line), []).append(str(block.block_id))

    grouping_diag_total: Dict[str, int] = {
        "grouping_direct_append_count": 0,
        "grouping_real_bridge_append_count": 0,
        "direct_group_merges": 0,
        "real_bridge_group_merges": 0,
        "direct_group_template_unknown_count": 0,
        "real_bridge_group_template_unsupported_count": 0,
        "grouping_virtual_boundary_count": 0,
        "grouping_mixed_boundary_count": 0,
        "grouping_infeasible_boundary_count": 0,
    }

    for line, block_ids in blocks_by_line.items():
        lp = LineAssemblyPlan(line=str(line))
        grouped_skeletons, grouped_bridges, grouping_diag = _build_line_skeletons(
            line=str(line),
            block_ids=block_ids,
            block_by_id=block_by_id,
            edge_lookup=edge_lookup,
            min_ton=min_ton,
            max_ton=max_ton,
            cfg=cfg,
        )
        lp.campaign_skeletons.extend(grouped_skeletons)
        lp.bridge_requirements.extend(grouped_bridges)
        for key, value in grouping_diag.items():
            grouping_diag_total[key] = int(grouping_diag_total.get(key, 0) + int(value or 0))

        sk_by_id = {sk.skeleton_id: sk for sk in lp.campaign_skeletons}
        for ridx, receiver in enumerate(lp.campaign_skeletons):
            if receiver.viability_band == "ALREADY_VIABLE":
                continue
            neighbor_idxs = [ridx - 1, ridx + 1, ridx - 2, ridx + 2]
            donor_candidates_for_receiver: List[DonorCandidate] = []
            bridge_reqs_for_receiver: List[BridgeRequirement] = []
            for didx in neighbor_idxs:
                if didx < 0 or didx >= len(lp.campaign_skeletons):
                    continue
                donor = lp.campaign_skeletons[didx]
                if donor.skeleton_id == receiver.skeleton_id:
                    continue
                donor_block = block_by_id.get(donor.block_ids[-1] if didx < ridx else donor.block_ids[0])
                receiver_block = block_by_id.get(receiver.block_ids[0] if didx < ridx else receiver.block_ids[-1])
                pair_feasible = False
                if donor_block is not None and receiver_block is not None:
                    pair_feasible = _block_boundary_pair_feasible(donor_block, receiver_block, cfg)
                duplicate_safe = not bool(set(donor.planned_order_ids) & set(receiver.planned_order_ids))
                rejection = "ACCEPTABLE"
                if donor.viability_band == "ALREADY_VIABLE":
                    rejection = "HEALTHY_CAMPAIGN_PROTECTED"
                elif not duplicate_safe:
                    rejection = "DUPLICATE_ORDER"
                elif not pair_feasible:
                    rejection = "PAIR_INVALID_NEEDS_BRIDGE"
                elif receiver.current_real_tons + donor.current_real_tons <= receiver.current_real_tons:
                    rejection = "NO_TON_GAIN"
                dc = DonorCandidate(
                    donor_block_id=donor.anchor_block_id,
                    donor_campaign_skeleton_id=donor.skeleton_id,
                    receiver_campaign_skeleton_id=receiver.skeleton_id,
                    line=str(line),
                    transferable_tons=float(round(donor.current_real_tons, 3)),
                    pair_feasible=bool(pair_feasible),
                    duplicate_safe=bool(duplicate_safe),
                    rejection_reason=rejection,
                )
                lp.donor_candidates.append(dc)
                donor_candidates_for_receiver.append(dc)
                if rejection == "PAIR_INVALID_NEEDS_BRIDGE":
                    bridge_reqs_for_receiver.append(
                        BridgeRequirement(
                            line=str(line),
                            source_block_id=donor.anchor_block_id,
                            target_block_id=receiver.anchor_block_id,
                            source_campaign_skeleton_id=donor.skeleton_id,
                            target_campaign_skeleton_id=receiver.skeleton_id,
                            reason_code="PAIR_INVALID_NEEDS_BRIDGE",
                            bridge_type_needed="SIMPLE_VIRTUAL_BRIDGE",
                            estimated_gap_closure_tons=float(round(donor.current_real_tons, 3)),
                            candidate_virtual_count=1,
                            mixed_bridge_required=False,
                            template_support_status="UNKNOWN",
                        )
                    )
                elif rejection == "HEALTHY_CAMPAIGN_PROTECTED":
                    bridge_reqs_for_receiver.append(
                        BridgeRequirement(
                            line=str(line),
                            source_block_id=donor.anchor_block_id,
                            target_block_id=receiver.anchor_block_id,
                            source_campaign_skeleton_id=donor.skeleton_id,
                            target_campaign_skeleton_id=receiver.skeleton_id,
                            reason_code="ONLY_HEALTHY_DONORS_AVAILABLE",
                            bridge_type_needed="MIXED_COMPLEX_BRIDGE",
                            estimated_gap_closure_tons=float(round(donor.current_real_tons, 3)),
                            candidate_virtual_count=0,
                            mixed_bridge_required=True,
                            template_support_status="UNKNOWN",
                        )
                    )
            acceptable = [d for d in donor_candidates_for_receiver if d.rejection_reason == "ACCEPTABLE"]
            simple_bridge = [b for b in bridge_reqs_for_receiver if b.bridge_type_needed != "MIXED_COMPLEX_BRIDGE"]
            mixed_bridge = [b for b in bridge_reqs_for_receiver if b.bridge_type_needed == "MIXED_COMPLEX_BRIDGE"]
            if acceptable:
                receiver.assembly_status = "NEEDS_DONOR"
                receiver.bridge_need_level = "NONE"
                receiver.notes = "planner_found_real_donor_candidate"
            elif simple_bridge:
                receiver.assembly_status = "NEEDS_BRIDGE"
                receiver.bridge_need_level = "NEEDS_BRIDGE"
                receiver.notes = "planner_found_simple_bridge_requirement"
            elif mixed_bridge:
                receiver.assembly_status = "NEEDS_MIXED_BRIDGE"
                receiver.bridge_need_level = "NEEDS_MIXED_BRIDGE"
                receiver.notes = "only healthy donor or complex mixed path available"
            else:
                receiver.assembly_status = "NOT_WORTH_RESCUING"
                receiver.bridge_need_level = "NONE"
                receiver.notes = "no local donor or bridge candidate"
            for br in bridge_reqs_for_receiver:
                if not any((e.source_campaign_skeleton_id == br.source_campaign_skeleton_id and e.target_campaign_skeleton_id == br.target_campaign_skeleton_id and e.reason_code == br.reason_code) for e in lp.bridge_requirements):
                    lp.bridge_requirements.append(br)

        used_blocks = {bid for sk in lp.campaign_skeletons for bid in sk.block_ids}
        all_line_blocks = [str(b.block_id) for b in master_result.selected_blocks if str(b.line) == str(line)]
        lp.unresolved_blocks = [bid for bid in all_line_blocks if bid not in used_blocks]
        line_plans.append(lp)

    plan = AssemblyPlan(plan_id=f"assembly_plan_{uuid4().hex[:8]}", line_plans=line_plans)
    skeletons = [sk for lp in line_plans for sk in lp.campaign_skeletons]
    bridges = [br for lp in line_plans for br in lp.bridge_requirements]
    donors = [dc for lp in line_plans for dc in lp.donor_candidates]
    bridge_merge_proposals = _build_bridge_merge_proposals(skeletons, bridges, cfg)
    selected_bridge_merge_proposal = _select_best_bridge_merge_proposal(bridge_merge_proposals)
    bridge_proposal_registry = {p.proposal_id: p.to_dict() for p in bridge_merge_proposals}
    trial_enabled = bool(getattr(cfg.model, "formal_single_bridge_trial_enabled", False))
    trial_proposal_id = str(getattr(cfg.model, "formal_single_bridge_trial_proposal_id", "") or "")
    formal_trial_selected = False
    if trial_proposal_id in {"CURRENT_BEST", "AUTO_BEST", "__BEST__"} and selected_bridge_merge_proposal is not None:
        trial_proposal_id = selected_bridge_merge_proposal.proposal_id
    if trial_enabled and selected_bridge_merge_proposal is not None and trial_proposal_id == selected_bridge_merge_proposal.proposal_id:
        selected_bridge_merge_proposal.selected_for_formal_trial = True
        formal_trial_selected = True
    bridge_merge_summary = _summarize_bridge_merge_proposals(bridge_merge_proposals, selected_bridge_merge_proposal)
    bridge_merge_simulation = _simulate_best_bridge_merge(selected_bridge_merge_proposal, skeletons, cfg)
    readiness = _judge_plan_maturity(skeletons, len(master_result.selected_blocks), cfg)
    # This iteration is intentionally diagnostics-only.  The maturity signal is
    # still exported, but finalization stays disabled until grouping quality is
    # proven over multiple runs.
    maturity_ready = bool(readiness["assembly_plan_ready_for_finalization"])
    readiness["assembly_plan_maturity_ready"] = maturity_ready
    readiness["assembly_plan_ready_for_finalization"] = False
    readiness["planner_ready_for_finalization"] = False
    readiness["assembly_plan_fallback_used"] = True
    if maturity_ready:
        readiness["assembly_plan_readiness_reason_code"] = "DIAGNOSTICS_ONLY_MODE"
        readiness["readiness_reason_code"] = "DIAGNOSTICS_ONLY_MODE"
    plan.summary = {
        "assembly_plan_enabled": True,
        "assembly_plan_line_count": int(len(line_plans)),
        "assembly_plan_campaign_skeleton_count": int(len(skeletons)),
        "assembly_plan_avg_blocks_per_skeleton": float(
            round(sum(len(s.block_ids) for s in skeletons) / max(1, len(skeletons)), 3)
        ),
        "assembly_plan_already_viable_count": int(sum(1 for s in skeletons if s.viability_band == "ALREADY_VIABLE")),
        "assembly_plan_near_viable_count": int(sum(1 for s in skeletons if s.viability_band == "NEAR_VIABLE")),
        "assembly_plan_merge_candidate_count": int(sum(1 for s in skeletons if s.viability_band == "MERGE_CANDIDATE")),
        "assembly_plan_hopeless_count": int(sum(1 for s in skeletons if s.viability_band == "HOPELESS_UNDERFILLED")),
        "assembly_plan_bridge_requirement_count": int(len(bridges)),
        "assembly_plan_mixed_bridge_required_count": int(sum(1 for b in bridges if b.mixed_bridge_required)),
        "mixed_bridge_requirements": int(sum(1 for b in bridges if b.mixed_bridge_required)),
        "assembly_plan_simple_bridge_candidate_count": int(sum(1 for b in bridges if not b.mixed_bridge_required)),
        "assembly_plan_simple_real_bridge_count": int(sum(1 for b in bridges if b.bridge_type_needed == "SIMPLE_REAL_BRIDGE")),
        "assembly_plan_simple_virtual_bridge_count": int(sum(1 for b in bridges if b.bridge_type_needed == "SIMPLE_VIRTUAL_BRIDGE")),
        "assembly_plan_internal_real_bridge_count": int(sum(1 for s in skeletons if s.internal_bridge_needed and "REAL_BRIDGE" in s.internal_bridge_type)),
        "assembly_plan_not_worth_rescuing_count": int(sum(1 for s in skeletons if s.assembly_status == "NOT_WORTH_RESCUING")),
        "assembly_plan_donor_candidate_count": int(len(donors)),
        **grouping_diag_total,
        **{k: v for k, v in bridge_merge_summary.items() if k != "bridge_merge_proposal_rows"},
        **{k: v for k, v in bridge_merge_simulation.items() if k != "bridge_merge_simulation_rows"},
        "formal_single_bridge_trial_selected": bool(formal_trial_selected),
        "formal_single_bridge_trial_selected_proposal_id": selected_bridge_merge_proposal.proposal_id if formal_trial_selected else "",
        "formal_single_bridge_trial_selected_source": selected_bridge_merge_proposal.source_skeleton_id if formal_trial_selected else "",
        "formal_single_bridge_trial_selected_target": selected_bridge_merge_proposal.target_skeleton_id if formal_trial_selected else "",
        "formal_single_bridge_trial_selected_bridge_type": selected_bridge_merge_proposal.bridge_type_needed if formal_trial_selected else "",
        "bridge_proposal_registry_size": int(len(bridge_proposal_registry)),
        "bridge_proposal_registry_keys_sample": list(bridge_proposal_registry.keys())[:5],
        "best_proposal_registry_hit": bool(selected_bridge_merge_proposal and selected_bridge_merge_proposal.proposal_id in bridge_proposal_registry),
        "formal_trial_registry_hit": bool(formal_trial_selected),
        **readiness,
    }
    nonstable = [s for s in skeletons if s.viability_band != "ALREADY_VIABLE"]
    second = sorted(nonstable, key=lambda s: s.current_real_tons, reverse=True)[0] if nonstable else None
    if second is not None:
        br = next((b for b in bridges if b.target_campaign_skeleton_id == second.skeleton_id), None)
        plan.summary.update(
            {
                "second_receiver_preclassified_by_planner": second.skeleton_id,
                "second_receiver_bridge_need_level": second.bridge_need_level,
                "second_receiver_bridge_type_needed": br.bridge_type_needed if br else "NONE",
                "second_receiver_template_support_status": br.template_support_status if br else "UNKNOWN",
            }
        )
    else:
        plan.summary.update(
            {
                "second_receiver_preclassified_by_planner": "",
                "second_receiver_bridge_need_level": "",
                "second_receiver_bridge_type_needed": "NONE",
                "second_receiver_template_support_status": "UNKNOWN",
            }
        )
    skeleton_rows = plan.skeleton_rows()
    for row in skeleton_rows:
        row.update(
            {
                "planner_ready_for_finalization": readiness["planner_ready_for_finalization"],
                "readiness_reason_code": readiness["readiness_reason_code"],
                "fallback_used": readiness["assembly_plan_fallback_used"],
                "direct_group_merges": plan.summary["direct_group_merges"],
                "real_bridge_group_merges": plan.summary["real_bridge_group_merges"],
                "mixed_bridge_requirements": plan.summary["mixed_bridge_requirements"],
            }
        )
    plan.diagnostics = {
        "campaign_assembly_plan_rows": skeleton_rows,
        "assembly_plan_diagnostics_rows": skeleton_rows,
        "bridge_requirement_rows": plan.bridge_rows(),
        "donor_candidate_rows": plan.donor_rows(),
        "bridge_merge_proposal_rows": bridge_merge_summary["bridge_merge_proposal_rows"],
        "bridge_merge_simulation_rows": bridge_merge_simulation["bridge_merge_simulation_rows"],
        "bridge_proposal_registry": bridge_proposal_registry,
        "formal_single_bridge_trial_proposal": selected_bridge_merge_proposal.to_dict() if formal_trial_selected and selected_bridge_merge_proposal is not None else {},
    }
    return plan
