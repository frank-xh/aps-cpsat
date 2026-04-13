from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from aps_cp_sat.persistence.db import get_sessionmaker
from aps_cp_sat.persistence.models import ScheduleBridgeDropSummary, ScheduleRun, ScheduleViolationSummary


router = APIRouter()


def _get_db() -> Session:
    return get_sessionmaker()()


@router.get("/runs")
def compare_runs(run_id: list[int] = Query(..., description="Repeat run_id params, e.g. ?run_id=1&run_id=2")) -> dict[str, Any]:
    db = _get_db()
    try:
        runs = db.scalars(select(ScheduleRun).where(ScheduleRun.id.in_(run_id))).all()
        by_id = {r.id: r for r in runs}

        items = []
        for rid in run_id:
            r = by_id.get(rid)
            if r is None:
                continue
            vio = db.scalar(select(ScheduleViolationSummary).where(ScheduleViolationSummary.run_id == rid))
            bridge = db.scalar(select(ScheduleBridgeDropSummary).where(ScheduleBridgeDropSummary.run_id == rid))
            items.append(
                {
                    "id": r.id,
                    "run_code": r.run_code,
                    "profile_name": r.profile_name,
                    "acceptance": r.acceptance,
                    "failure_mode": r.failure_mode,
                    "routing_feasible": r.routing_feasible,
                    "scheduled_orders": r.scheduled_orders,
                    "unscheduled_orders": r.unscheduled_orders,
                    "dropped_orders": r.dropped_orders,
                    "total_run_seconds": r.total_run_seconds,
                    "template_build_seconds": r.template_build_seconds,
                    "joint_master_seconds": r.joint_master_seconds,
                    "fallback_total_seconds": r.fallback_total_seconds,
                    "violations": None
                    if vio is None
                    else {
                        "direct_reverse_step_violation_count": vio.direct_reverse_step_violation_count,
                        "virtual_attach_reverse_violation_count": vio.virtual_attach_reverse_violation_count,
                        "period_reverse_count_violation_count": vio.period_reverse_count_violation_count,
                        "bridge_count_violation_count": vio.bridge_count_violation_count,
                        "invalid_virtual_spec_count": vio.invalid_virtual_spec_count,
                        "candidate_unroutable_slot_count": vio.candidate_unroutable_slot_count,
                    },
                    "bridge": None
                    if bridge is None
                    else {
                        "selected_direct_edge_count": bridge.selected_direct_edge_count,
                        "selected_real_bridge_edge_count": bridge.selected_real_bridge_edge_count,
                        "selected_virtual_bridge_edge_count": bridge.selected_virtual_bridge_edge_count,
                        "max_bridge_count_used": bridge.max_bridge_count_used,
                        "dropped_order_count": bridge.dropped_order_count,
                    },
                }
            )
        return {"items": items}
    finally:
        db.close()

