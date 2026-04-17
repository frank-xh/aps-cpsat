from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from aps_cp_sat.persistence.db import get_sessionmaker
from aps_cp_sat.persistence.models import ScheduleTransitionMetric, ScheduleViolationSummary


router = APIRouter()


def _get_db() -> Session:
    return get_sessionmaker()()


@router.get("/{run_id}/violations")
def get_violation_summary(run_id: int) -> dict[str, Any]:
    db = _get_db()
    try:
        vio = db.scalar(select(ScheduleViolationSummary).where(ScheduleViolationSummary.run_id == run_id))
        if vio is None:
            return {"run_id": run_id, "available": False}
        return {
            "run_id": run_id,
            "available": True,
            "direct_reverse_step_violation_count": vio.direct_reverse_step_violation_count,
            "virtual_attach_reverse_violation_count": vio.virtual_attach_reverse_violation_count,
            "period_reverse_count_violation_count": vio.period_reverse_count_violation_count,
            "bridge_count_violation_count": vio.bridge_count_violation_count,
            "invalid_virtual_spec_count": vio.invalid_virtual_spec_count,
            "candidate_unroutable_slot_count": vio.candidate_unroutable_slot_count,
            "candidate_bad_slot_count": vio.candidate_bad_slot_count,
            "candidate_zero_in_order_count": vio.candidate_zero_in_order_count,
            "candidate_zero_out_order_count": vio.candidate_zero_out_order_count,
            "hard_cap_not_enforced": vio.hard_cap_not_enforced,
        }
    finally:
        db.close()


@router.get("/{run_id}/transition-metrics")
def list_transition_metrics(
    run_id: int,
    line: str | None = None,
    slot_no: int | None = None,
    limit: int = Query(5000, ge=1, le=50000),
) -> dict[str, Any]:
    db = _get_db()
    try:
        q = select(ScheduleTransitionMetric).where(ScheduleTransitionMetric.run_id == run_id)
        if line:
            q = q.where(ScheduleTransitionMetric.line == line)
        if slot_no is not None:
            q = q.where(ScheduleTransitionMetric.slot_no == slot_no)

        total = db.scalar(select(func.count()).select_from(q.subquery()))
        q = q.order_by(ScheduleTransitionMetric.line.asc(), ScheduleTransitionMetric.slot_no.asc(), ScheduleTransitionMetric.seq_no.asc()).limit(limit)
        items = []
        for m in db.scalars(q).all():
            items.append(
                {
                    "line": m.line,
                    "slot_no": m.slot_no,
                    "seq_no": m.seq_no,
                    "from_order_id": m.from_order_id,
                    "to_order_id": m.to_order_id,
                    "from_temp_mid": m.from_temp_mid,
                    "to_temp_mid": m.to_temp_mid,
                    "temp_overlap": m.temp_overlap,
                    "temp_jump": m.temp_jump,
                    "from_thickness": m.from_thickness,
                    "to_thickness": m.to_thickness,
                    "thickness_jump": m.thickness_jump,
                    "from_width": m.from_width,
                    "to_width": m.to_width,
                    "width_jump": m.width_jump,
                    "reverse_flag": m.reverse_flag,
                    "bridge_type": m.bridge_type,
                }
            )
        return {"items": items, "total": int(total or 0), "limited_to": limit}
    finally:
        db.close()

