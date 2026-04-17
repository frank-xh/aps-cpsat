from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Query
from sqlalchemy import and_, func, or_, select
from sqlalchemy.orm import Session

from aps_cp_sat.persistence.db import get_sessionmaker
from aps_cp_sat.persistence.models import ScheduleBridgeDropSummary, ScheduleOrderResult, ScheduleRun, ScheduleViolationSummary


router = APIRouter()


def _get_db() -> Session:
    return get_sessionmaker()()


def _run_brief(r: ScheduleRun) -> dict[str, Any]:
    return {
        "id": r.id,
        "run_code": r.run_code,
        "profile_name": r.profile_name,
        "result_file_path": r.result_file_path,
        "acceptance": r.acceptance,
        "failure_mode": r.failure_mode,
        "routing_feasible": r.routing_feasible,
        "analysis_only": r.analysis_only,
        "official_exported": r.official_exported,
        "analysis_exported": r.analysis_exported,
        "scheduled_orders": r.scheduled_orders,
        "dropped_orders": r.dropped_orders,
        "scheduled_tons": r.scheduled_tons,
        "total_run_seconds": r.total_run_seconds,
        "created_at": r.created_at,
    }


@router.get("")
def list_runs(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=200),
    profile: str | None = None,
    acceptance: str | None = None,
    failure_mode: str | None = None,
    created_from: datetime | None = None,
    created_to: datetime | None = None,
    sort_by: str = Query("created_at", pattern="^(created_at|scheduled_orders|total_run_seconds)$"),
    sort_order: str = Query("desc", pattern="^(asc|desc)$"),
) -> dict[str, Any]:
    db = _get_db()
    try:
        filters = []
        if profile:
            filters.append(ScheduleRun.profile_name == profile)
        if acceptance:
            filters.append(ScheduleRun.acceptance == acceptance)
        if failure_mode:
            filters.append(ScheduleRun.failure_mode == failure_mode)
        if created_from:
            filters.append(ScheduleRun.created_at >= created_from)
        if created_to:
            filters.append(ScheduleRun.created_at <= created_to)

        where = and_(*filters) if filters else None
        total = db.scalar(select(func.count()).select_from(ScheduleRun).where(where)) if where is not None else db.scalar(select(func.count()).select_from(ScheduleRun))

        q = select(ScheduleRun)
        if where is not None:
            q = q.where(where)

        col = getattr(ScheduleRun, sort_by)
        q = q.order_by(col.asc() if sort_order == "asc" else col.desc())
        q = q.offset((page - 1) * page_size).limit(page_size)

        items = []
        for r in db.scalars(q).all():
            items.append(
                {
                    "id": r.id,
                    "run_code": r.run_code,
                    "profile_name": r.profile_name,
                    "result_file_path": r.result_file_path,
                    "acceptance": r.acceptance,
                    "failure_mode": r.failure_mode,
                    "routing_feasible": r.routing_feasible,
                    "analysis_only": r.analysis_only,
                    "official_exported": r.official_exported,
                    "analysis_exported": r.analysis_exported,
                    "total_orders": r.total_orders,
                    "scheduled_orders": r.scheduled_orders,
                    "unscheduled_orders": r.unscheduled_orders,
                    "dropped_orders": r.dropped_orders,
                    "scheduled_tons": r.scheduled_tons,
                    "unscheduled_tons": r.unscheduled_tons,
                    "template_build_seconds": r.template_build_seconds,
                    "joint_master_seconds": r.joint_master_seconds,
                    "fallback_total_seconds": r.fallback_total_seconds,
                    "total_run_seconds": r.total_run_seconds,
                    "evidence_level": r.evidence_level,
                    "best_candidate_available": r.best_candidate_available,
                    "best_candidate_type": r.best_candidate_type,
                    "best_candidate_objective": r.best_candidate_objective,
                    "best_candidate_unroutable_slot_count": r.best_candidate_unroutable_slot_count,
                    "created_at": r.created_at,
                }
            )
        return {"items": items, "total": int(total or 0), "page": page, "page_size": page_size}
    finally:
        db.close()


@router.get("/latest")
def get_latest_run() -> dict[str, Any]:
    db = _get_db()
    try:
        run = db.scalar(select(ScheduleRun).order_by(ScheduleRun.created_at.desc(), ScheduleRun.id.desc()).limit(1))
        if run is None:
            return {"available": False}
        out = {"available": True}
        out.update(_run_brief(run))
        return out
    finally:
        db.close()


@router.get("/{run_id}")
def get_run_detail(run_id: int) -> dict[str, Any]:
    db = _get_db()
    try:
        run = db.get(ScheduleRun, run_id)
        if run is None:
            return {"error": "not_found"}

        vio = db.scalar(select(ScheduleViolationSummary).where(ScheduleViolationSummary.run_id == run_id))
        bridge = db.scalar(select(ScheduleBridgeDropSummary).where(ScheduleBridgeDropSummary.run_id == run_id))

        # line summary computed from order results (analysis or official)
        order_q = select(ScheduleOrderResult).where(ScheduleOrderResult.run_id == run_id)
        orders = db.scalars(order_q).all()
        lines = sorted({o.line for o in orders if o.line})
        line_summary = []
        for line in lines:
            members = [o for o in orders if o.line == line]
            assigned = [o for o in members if not (o.drop_flag is True) and o.slot_no is not None]
            slot_nos = sorted({o.slot_no for o in assigned if o.slot_no is not None})
            slot_counts = []
            for s in slot_nos:
                slot_counts.append(sum(1 for o in assigned if o.slot_no == s))
            line_summary.append(
                {
                    "line": line,
                    "assigned_orders": len(assigned),
                    "assigned_tons": float(sum((o.tons or 0.0) for o in assigned)),
                    "slot_count": len(slot_nos),
                    "avg_slot_order_count": float(sum(slot_counts) / len(slot_counts)) if slot_counts else 0.0,
                    "max_slot_order_count": max(slot_counts) if slot_counts else 0,
                    "unroutable_slot_count": len({o.slot_no for o in members if o.slot_unroutable_flag is True and o.slot_no is not None}),
                    "dropped_orders": sum(1 for o in members if o.drop_flag is True),
                }
            )

        return {
            "run": {
                "id": run.id,
                "run_code": run.run_code,
                "profile_name": run.profile_name,
                "input_order_file": run.input_order_file,
                "input_steel_file": run.input_steel_file,
                "result_file_path": run.result_file_path,
                "acceptance": run.acceptance,
                "failure_mode": run.failure_mode,
                "routing_feasible": run.routing_feasible,
                "analysis_only": run.analysis_only,
                "official_exported": run.official_exported,
                "analysis_exported": run.analysis_exported,
                "total_orders": run.total_orders,
                "scheduled_orders": run.scheduled_orders,
                "unscheduled_orders": run.unscheduled_orders,
                "dropped_orders": run.dropped_orders,
                "scheduled_tons": run.scheduled_tons,
                "unscheduled_tons": run.unscheduled_tons,
                "template_build_seconds": run.template_build_seconds,
                "joint_master_seconds": run.joint_master_seconds,
                "fallback_total_seconds": run.fallback_total_seconds,
                "total_run_seconds": run.total_run_seconds,
                "evidence_level": run.evidence_level,
                "best_candidate_available": run.best_candidate_available,
                "best_candidate_type": run.best_candidate_type,
                "best_candidate_objective": run.best_candidate_objective,
                "best_candidate_unroutable_slot_count": run.best_candidate_unroutable_slot_count,
                "created_at": run.created_at,
            },
            "violation_summary": None
            if vio is None
            else {
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
            },
            "bridge_drop_summary": None
            if bridge is None
            else {
                "selected_direct_edge_count": bridge.selected_direct_edge_count,
                "selected_real_bridge_edge_count": bridge.selected_real_bridge_edge_count,
                "selected_virtual_bridge_edge_count": bridge.selected_virtual_bridge_edge_count,
                "direct_edge_ratio": bridge.direct_edge_ratio,
                "real_bridge_ratio": bridge.real_bridge_ratio,
                "virtual_bridge_ratio": bridge.virtual_bridge_ratio,
                "max_bridge_count_used": bridge.max_bridge_count_used,
                "avg_virtual_count": bridge.avg_virtual_count,
                "dropped_reason_histogram_json": bridge.dropped_reason_histogram_json,
                "dropped_order_count": bridge.dropped_order_count,
                "dropped_tons": bridge.dropped_tons,
            },
            "line_summary": line_summary,
        }
    finally:
        db.close()


@router.get("/{run_id}/orders")
def list_orders(
    run_id: int,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
    line: str | None = None,
    slot_no: int | None = None,
    candidate_status: str | None = None,
    drop_flag: bool | None = None,
    order_id: str | None = None,
) -> dict[str, Any]:
    db = _get_db()
    try:
        q = select(ScheduleOrderResult).where(ScheduleOrderResult.run_id == run_id)
        if line:
            q = q.where(ScheduleOrderResult.line == line)
        if slot_no is not None:
            q = q.where(ScheduleOrderResult.slot_no == slot_no)
        if candidate_status:
            q = q.where(ScheduleOrderResult.candidate_status == candidate_status)
        if drop_flag is not None:
            q = q.where(ScheduleOrderResult.drop_flag == drop_flag)
        if order_id:
            q = q.where(ScheduleOrderResult.order_id.like(f"%{order_id}%"))

        total = db.scalar(select(func.count()).select_from(q.subquery()))
        q = q.order_by(ScheduleOrderResult.line.asc().nulls_last(), ScheduleOrderResult.slot_no.asc().nulls_last(), ScheduleOrderResult.candidate_position.asc().nulls_last())
        q = q.offset((page - 1) * page_size).limit(page_size)
        items = []
        for o in db.scalars(q).all():
            items.append(
                {
                    "order_id": o.order_id,
                    "line": o.line,
                    "slot_no": o.slot_no,
                    "candidate_position": o.candidate_position,
                    "tons": o.tons,
                    "width": o.width,
                    "thickness": o.thickness,
                    "temp_min": o.temp_min,
                    "temp_max": o.temp_max,
                    "steel_group": o.steel_group,
                    "line_capability": o.line_capability,
                    "candidate_status": o.candidate_status,
                    "drop_flag": o.drop_flag,
                    "dominant_drop_reason": o.dominant_drop_reason,
                    "secondary_reasons": o.secondary_reasons,
                    "slot_unroutable_flag": o.slot_unroutable_flag,
                    "slot_route_risk_score": o.slot_route_risk_score,
                    "analysis_only": o.analysis_only,
                }
            )
        return {"items": items, "total": int(total or 0), "page": page, "page_size": page_size}
    finally:
        db.close()
