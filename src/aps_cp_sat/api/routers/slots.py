from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from aps_cp_sat.persistence.db import get_sessionmaker
from aps_cp_sat.persistence.models import ScheduleSlotSummary


router = APIRouter()


def _get_db() -> Session:
    return get_sessionmaker()()


@router.get("/{run_id}/slots")
def list_slots(
    run_id: int,
    page: int = Query(1, ge=1),
    page_size: int = Query(100, ge=1, le=1000),
    line: str | None = None,
    slot_unroutable_only: bool = False,
) -> dict[str, Any]:
    db = _get_db()
    try:
        q = select(ScheduleSlotSummary).where(ScheduleSlotSummary.run_id == run_id)
        if line:
            q = q.where(ScheduleSlotSummary.line == line)
        if slot_unroutable_only:
            q = q.where(ScheduleSlotSummary.dominant_unroutable_reason.is_not(None))

        total = db.scalar(select(func.count()).select_from(q.subquery()))
        q = q.order_by(ScheduleSlotSummary.line.asc(), ScheduleSlotSummary.slot_no.asc())
        q = q.offset((page - 1) * page_size).limit(page_size)
        items = []
        for s in db.scalars(q).all():
            items.append(
                {
                    "line": s.line,
                    "slot_no": s.slot_no,
                    "slot_order_count": s.slot_order_count,
                    "slot_tons": s.slot_tons,
                    "order_count_over_cap": s.order_count_over_cap,
                    "template_coverage_ratio": s.template_coverage_ratio,
                    "missing_template_edge_count": s.missing_template_edge_count,
                    "zero_in_orders": s.zero_in_orders,
                    "zero_out_orders": s.zero_out_orders,
                    "width_span": s.width_span,
                    "thickness_span": s.thickness_span,
                    "steel_group_count": s.steel_group_count,
                    "pair_gap_proxy": s.pair_gap_proxy,
                    "span_risk": s.span_risk,
                    "degree_risk": s.degree_risk,
                    "isolated_order_penalty": s.isolated_order_penalty,
                    "slot_route_risk_score": s.slot_route_risk_score,
                    "dominant_unroutable_reason": s.dominant_unroutable_reason,
                }
            )
        return {"items": items, "total": int(total or 0), "page": page, "page_size": page_size}
    finally:
        db.close()

