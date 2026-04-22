"""Campaign Assembly Planner for block-first final campaign shaping.

This module sits between block_master and block_realizer.  It does not produce
the final schedule.  Its first responsibility is to expose campaign skeletons,
donor space and bridge requirements before the realizer starts doing local
repair work.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal
from uuid import uuid4

import pandas as pd

from aps_cp_sat.config import PlannerConfig
from aps_cp_sat.model.block_master import BlockMasterResult
from aps_cp_sat.model.block_types import CandidateBlock

ViabilityBand = Literal["ALREADY_VIABLE", "NEAR_VIABLE", "MERGE_CANDIDATE", "HOPELESS_UNDERFILLED"]
AssemblyStatus = Literal["STABLE", "NEEDS_DONOR", "NEEDS_BRIDGE", "NEEDS_MIXED_BRIDGE", "NOT_WORTH_RESCUING"]
BridgeTypeNeeded = Literal["NONE", "SIMPLE_REAL_BRIDGE", "SIMPLE_VIRTUAL_BRIDGE", "MIXED_COMPLEX_BRIDGE"]
TemplateSupportStatus = Literal["SUPPORTED", "UNSUPPORTED", "UNKNOWN"]


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
                        "anchor_block_id": sk.anchor_block_id,
                        "current_real_tons": float(round(sk.current_real_tons, 3)),
                        "viability_band": sk.viability_band,
                        "assembly_status": sk.assembly_status,
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
    _ = orders_df, transition_pack
    block_by_id = {str(b.block_id): b for b in master_result.selected_blocks}
    min_ton = float(cfg.rule.campaign_ton_min)
    max_ton = float(cfg.rule.campaign_ton_max)
    line_plans: List[LineAssemblyPlan] = []

    slots_by_line: Dict[str, List[Any]] = {}
    for slot in master_result.campaign_slots:
        slots_by_line.setdefault(str(slot.line), []).append(slot)
    if not slots_by_line:
        for line, block_ids in master_result.block_order_by_line.items():
            fake_slot = type("_AssemblySlot", (), {})()
            fake_slot.line = line
            fake_slot.campaign_id = f"{line}__assembly_fallback"
            fake_slot.block_ids = list(block_ids)
            slots_by_line.setdefault(str(line), []).append(fake_slot)

    for line, slots in slots_by_line.items():
        lp = LineAssemblyPlan(line=str(line))
        for idx, slot in enumerate(slots, start=1):
            slot_campaign_id = str(getattr(slot, "campaign_id", "") or f"{line}__sk_{idx}")
            block_ids = [str(bid) for bid in getattr(slot, "block_ids", []) if str(bid) in block_by_id]
            if not block_ids:
                continue
            segmented_skeletons, segmented_bridges = _segment_slot_into_skeletons(
                line=str(line),
                slot_campaign_id=slot_campaign_id,
                block_ids=block_ids,
                block_by_id=block_by_id,
                min_ton=min_ton,
                max_ton=max_ton,
                cfg=cfg,
            )
            lp.campaign_skeletons.extend(segmented_skeletons)
            lp.bridge_requirements.extend(segmented_bridges)

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
    plan.summary = {
        "assembly_plan_enabled": True,
        "assembly_plan_line_count": int(len(line_plans)),
        "assembly_plan_campaign_skeleton_count": int(len(skeletons)),
        "assembly_plan_already_viable_count": int(sum(1 for s in skeletons if s.viability_band == "ALREADY_VIABLE")),
        "assembly_plan_near_viable_count": int(sum(1 for s in skeletons if s.viability_band == "NEAR_VIABLE")),
        "assembly_plan_merge_candidate_count": int(sum(1 for s in skeletons if s.viability_band == "MERGE_CANDIDATE")),
        "assembly_plan_hopeless_count": int(sum(1 for s in skeletons if s.viability_band == "HOPELESS_UNDERFILLED")),
        "assembly_plan_bridge_requirement_count": int(len(bridges)),
        "assembly_plan_mixed_bridge_required_count": int(sum(1 for b in bridges if b.mixed_bridge_required)),
        "assembly_plan_simple_bridge_candidate_count": int(sum(1 for b in bridges if not b.mixed_bridge_required)),
        "assembly_plan_not_worth_rescuing_count": int(sum(1 for s in skeletons if s.assembly_status == "NOT_WORTH_RESCUING")),
        "assembly_plan_donor_candidate_count": int(len(donors)),
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
    plan.diagnostics = {
        "campaign_assembly_plan_rows": plan.skeleton_rows(),
        "bridge_requirement_rows": plan.bridge_rows(),
        "donor_candidate_rows": plan.donor_rows(),
    }
    return plan
