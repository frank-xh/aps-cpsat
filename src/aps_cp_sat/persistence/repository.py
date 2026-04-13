from __future__ import annotations

from typing import Any, Iterable

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from aps_cp_sat.persistence.models import (
    ScheduleBridgeDropSummary,
    ScheduleOrderResult,
    ScheduleRun,
    ScheduleSlotSummary,
    ScheduleTransitionMetric,
    ScheduleViolationSummary,
)


def get_run_by_run_code(session: Session, run_code: str) -> ScheduleRun | None:
    return session.scalar(select(ScheduleRun).where(ScheduleRun.run_code == run_code))


def get_run_by_result_file_path(session: Session, result_file_path: str) -> ScheduleRun | None:
    if not result_file_path:
        return None
    return session.scalar(select(ScheduleRun).where(ScheduleRun.result_file_path == result_file_path))


def upsert_run(session: Session, run_code: str, result_file_path: str | None, fields: dict[str, Any]) -> ScheduleRun:
    run = get_run_by_run_code(session, run_code)
    if run is None and result_file_path:
        run = get_run_by_result_file_path(session, result_file_path)

    if run is None:
        run = ScheduleRun(run_code=run_code)
        session.add(run)

    if result_file_path:
        run.result_file_path = result_file_path

    for k, v in fields.items():
        if hasattr(run, k):
            setattr(run, k, v)
    session.flush()
    return run


def replace_order_results(session: Session, run_id: int, rows: Iterable[dict[str, Any]]) -> None:
    session.execute(delete(ScheduleOrderResult).where(ScheduleOrderResult.run_id == run_id))
    rows_list = list(rows)
    if not rows_list:
        return
    session.bulk_insert_mappings(ScheduleOrderResult, [dict(run_id=run_id, **r) for r in rows_list])


def replace_slot_summaries(session: Session, run_id: int, rows: Iterable[dict[str, Any]]) -> None:
    session.execute(delete(ScheduleSlotSummary).where(ScheduleSlotSummary.run_id == run_id))
    rows_list = list(rows)
    if not rows_list:
        return
    session.bulk_insert_mappings(ScheduleSlotSummary, [dict(run_id=run_id, **r) for r in rows_list])


def replace_violation_summary(session: Session, run_id: int, fields: dict[str, Any]) -> None:
    session.execute(delete(ScheduleViolationSummary).where(ScheduleViolationSummary.run_id == run_id))
    session.add(ScheduleViolationSummary(run_id=run_id, **fields))


def replace_bridge_drop_summary(session: Session, run_id: int, fields: dict[str, Any]) -> None:
    session.execute(delete(ScheduleBridgeDropSummary).where(ScheduleBridgeDropSummary.run_id == run_id))
    session.add(ScheduleBridgeDropSummary(run_id=run_id, **fields))


def replace_transition_metrics(session: Session, run_id: int, rows: Iterable[dict[str, Any]]) -> None:
    session.execute(delete(ScheduleTransitionMetric).where(ScheduleTransitionMetric.run_id == run_id))
    rows_list = list(rows)
    if not rows_list:
        return
    session.bulk_insert_mappings(ScheduleTransitionMetric, [dict(run_id=run_id, **r) for r in rows_list])

