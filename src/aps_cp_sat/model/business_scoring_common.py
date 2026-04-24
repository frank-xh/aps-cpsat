from __future__ import annotations

from typing import Any

from aps_cp_sat.model.edge_hard_filter import sequence_pair_passes_final_hard_rules
from aps_cp_sat.model.reverse_width_budget import CampaignBuildState, can_accept_reverse_width_pair


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def is_near_viable_tons(
    total_tons: float,
    ton_min: float,
    *,
    gap_limit: float,
    min_fill_ratio: float,
) -> bool:
    total = _safe_float(total_tons, 0.0)
    minimum = max(1.0, _safe_float(ton_min, 0.0))
    gap = max(0.0, minimum - total)
    ratio = total / minimum
    return gap <= gap_limit or ratio >= min_fill_ratio


def compute_extendability_metrics(
    *,
    current_order: dict[str, Any],
    candidate_order: dict[str, Any],
    candidate_successor_ids: list[str],
    order_record_by_oid: dict[str, dict[str, Any]],
    cfg,
    line: str,
    current_state: CampaignBuildState,
    edge_meta: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Count truly extendable successors after active hard filter + reverse budget."""
    candidate_oid = _safe_text(candidate_order.get("order_id"))
    candidate_group = _safe_text(candidate_order.get("steel_group"))
    candidate_width = _safe_float(candidate_order.get("width"), 0.0)
    edge_context = dict(edge_meta or {})
    edge_context.setdefault("line", line)

    extendable = 0
    same_group_extendable = 0
    width_desc_extendable = 0
    reverse_budget_tight = 0

    ok_into_candidate, _, candidate_state = can_accept_reverse_width_pair(
        current_order,
        candidate_order,
        current_state,
        max_reverse_count=int(
            getattr(getattr(cfg, "model", None), "constructive_reverse_width_max_count", 2) or 2
        ),
    )
    if not ok_into_candidate:
        return {
            "successor_extendable_out_degree": 0.0,
            "successor_same_group_extendable_degree": 0.0,
            "successor_width_desc_extendable_degree": 0.0,
            "successor_dead_end_risk_score": 1.0,
            "successor_reverse_budget_tight_degree": 1.0,
        }

    for succ_oid in candidate_successor_ids:
        succ_rec = dict(order_record_by_oid.get(_safe_text(succ_oid), {}) or {})
        if not succ_rec or _safe_text(succ_rec.get("order_id")) == candidate_oid:
            continue
        ok_pair, _reason = sequence_pair_passes_final_hard_rules(
            candidate_order,
            succ_rec,
            cfg,
            context=edge_context,
        )
        if not ok_pair:
            continue
        ok_budget, _budget_reason, next_state = can_accept_reverse_width_pair(
            candidate_order,
            succ_rec,
            candidate_state,
            max_reverse_count=int(
                getattr(getattr(cfg, "model", None), "constructive_reverse_width_max_count", 2) or 2
            ),
        )
        if not ok_budget:
            reverse_budget_tight += 1
            continue
        extendable += 1
        succ_group = _safe_text(succ_rec.get("steel_group"))
        succ_width = _safe_float(succ_rec.get("width"), 0.0)
        if candidate_group and succ_group == candidate_group:
            same_group_extendable += 1
        if succ_width <= candidate_width + 1e-6:
            width_desc_extendable += 1
        if int(next_state.reverse_width_count) >= int(
            getattr(getattr(cfg, "model", None), "constructive_reverse_width_max_count", 2) or 2
        ):
            reverse_budget_tight += 1

    dead_end_risk = 1.0
    if extendable >= 4:
        dead_end_risk = 0.0
    elif extendable == 3:
        dead_end_risk = 0.2
    elif extendable == 2:
        dead_end_risk = 0.45
    elif extendable == 1:
        dead_end_risk = 0.75

    return {
        "successor_extendable_out_degree": float(extendable),
        "successor_same_group_extendable_degree": float(same_group_extendable),
        "successor_width_desc_extendable_degree": float(width_desc_extendable),
        "successor_dead_end_risk_score": float(dead_end_risk),
        "successor_reverse_budget_tight_degree": float(reverse_budget_tight),
    }
