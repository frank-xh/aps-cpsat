from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class ScheduleRun(Base):
    __tablename__ = "schedule_run"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    run_code: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)

    profile_name: Mapped[str | None] = mapped_column(String(64), nullable=True)
    input_order_file: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    input_steel_file: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    result_file_path: Mapped[str | None] = mapped_column(String(1024), unique=True, nullable=True)

    acceptance: Mapped[str | None] = mapped_column(String(64), nullable=True)
    failure_mode: Mapped[str | None] = mapped_column(String(64), nullable=True)
    routing_feasible: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    analysis_only: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    official_exported: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    analysis_exported: Mapped[bool | None] = mapped_column(Boolean, nullable=True)

    total_orders: Mapped[int | None] = mapped_column(Integer, nullable=True)
    scheduled_orders: Mapped[int | None] = mapped_column(Integer, nullable=True)
    unscheduled_orders: Mapped[int | None] = mapped_column(Integer, nullable=True)
    dropped_orders: Mapped[int | None] = mapped_column(Integer, nullable=True)
    scheduled_tons: Mapped[float | None] = mapped_column(Float, nullable=True)
    unscheduled_tons: Mapped[float | None] = mapped_column(Float, nullable=True)

    template_build_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    joint_master_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    fallback_total_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    total_run_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)

    evidence_level: Mapped[str | None] = mapped_column(String(64), nullable=True)

    best_candidate_available: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    best_candidate_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    best_candidate_objective: Mapped[float | None] = mapped_column(Float, nullable=True)
    best_candidate_unroutable_slot_count: Mapped[int | None] = mapped_column(Integer, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.current_timestamp(), nullable=False)


class ScheduleOrderResult(Base):
    __tablename__ = "schedule_order_result"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("schedule_run.id", ondelete="CASCADE"), nullable=False)

    order_id: Mapped[str] = mapped_column(String(128), nullable=False)
    line: Mapped[str | None] = mapped_column(String(32), nullable=True)
    slot_no: Mapped[int | None] = mapped_column(Integer, nullable=True)
    candidate_position: Mapped[int | None] = mapped_column(Integer, nullable=True)

    tons: Mapped[float | None] = mapped_column(Float, nullable=True)
    width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    thickness: Mapped[float | None] = mapped_column(Float, nullable=True)
    temp_min: Mapped[float | None] = mapped_column(Float, nullable=True)
    temp_max: Mapped[float | None] = mapped_column(Float, nullable=True)
    steel_group: Mapped[str | None] = mapped_column(String(256), nullable=True)
    line_capability: Mapped[str | None] = mapped_column(String(32), nullable=True)

    candidate_status: Mapped[str | None] = mapped_column(String(64), nullable=True)
    drop_flag: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    dominant_drop_reason: Mapped[str | None] = mapped_column(String(64), nullable=True)
    secondary_reasons: Mapped[str | None] = mapped_column(Text, nullable=True)
    slot_unroutable_flag: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    slot_route_risk_score: Mapped[float | None] = mapped_column(Float, nullable=True)

    analysis_only: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.current_timestamp(), nullable=False)


class ScheduleSlotSummary(Base):
    __tablename__ = "schedule_slot_summary"
    __table_args__ = (UniqueConstraint("run_id", "line", "slot_no", name="uk_slot_summary"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("schedule_run.id", ondelete="CASCADE"), nullable=False)

    line: Mapped[str] = mapped_column(String(32), nullable=False)
    slot_no: Mapped[int] = mapped_column(Integer, nullable=False)

    slot_order_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    slot_tons: Mapped[float | None] = mapped_column(Float, nullable=True)
    order_count_over_cap: Mapped[int | None] = mapped_column(Integer, nullable=True)

    template_coverage_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    missing_template_edge_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    zero_in_orders: Mapped[int | None] = mapped_column(Integer, nullable=True)
    zero_out_orders: Mapped[int | None] = mapped_column(Integer, nullable=True)

    width_span: Mapped[int | None] = mapped_column(Integer, nullable=True)
    thickness_span: Mapped[float | None] = mapped_column(Float, nullable=True)
    steel_group_count: Mapped[int | None] = mapped_column(Integer, nullable=True)

    pair_gap_proxy: Mapped[float | None] = mapped_column(Float, nullable=True)
    span_risk: Mapped[float | None] = mapped_column(Float, nullable=True)
    degree_risk: Mapped[float | None] = mapped_column(Float, nullable=True)
    isolated_order_penalty: Mapped[float | None] = mapped_column(Float, nullable=True)

    slot_route_risk_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    dominant_unroutable_reason: Mapped[str | None] = mapped_column(String(64), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.current_timestamp(), nullable=False)


class ScheduleViolationSummary(Base):
    __tablename__ = "schedule_violation_summary"
    __table_args__ = (UniqueConstraint("run_id", name="uk_violation_summary_run_id"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("schedule_run.id", ondelete="CASCADE"), nullable=False)

    direct_reverse_step_violation_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    virtual_attach_reverse_violation_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    period_reverse_count_violation_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    bridge_count_violation_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    invalid_virtual_spec_count: Mapped[int | None] = mapped_column(Integer, nullable=True)

    candidate_unroutable_slot_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    candidate_bad_slot_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    candidate_zero_in_order_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    candidate_zero_out_order_count: Mapped[int | None] = mapped_column(Integer, nullable=True)

    hard_cap_not_enforced: Mapped[bool | None] = mapped_column(Boolean, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.current_timestamp(), nullable=False)


class ScheduleBridgeDropSummary(Base):
    __tablename__ = "schedule_bridge_drop_summary"
    __table_args__ = (UniqueConstraint("run_id", name="uk_bridge_drop_summary_run_id"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("schedule_run.id", ondelete="CASCADE"), nullable=False)

    selected_direct_edge_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    selected_real_bridge_edge_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    selected_virtual_bridge_edge_count: Mapped[int | None] = mapped_column(Integer, nullable=True)

    direct_edge_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    real_bridge_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    virtual_bridge_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)

    max_bridge_count_used: Mapped[int | None] = mapped_column(Integer, nullable=True)
    avg_virtual_count: Mapped[float | None] = mapped_column(Float, nullable=True)

    dropped_reason_histogram_json: Mapped[Any | None] = mapped_column(JSON, nullable=True)
    dropped_order_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    dropped_tons: Mapped[float | None] = mapped_column(Float, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.current_timestamp(), nullable=False)


class ScheduleTransitionMetric(Base):
    __tablename__ = "schedule_transition_metric"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("schedule_run.id", ondelete="CASCADE"), nullable=False)

    line: Mapped[str] = mapped_column(String(32), nullable=False)
    slot_no: Mapped[int] = mapped_column(Integer, nullable=False)
    seq_no: Mapped[int] = mapped_column(Integer, nullable=False)

    from_order_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    to_order_id: Mapped[str | None] = mapped_column(String(128), nullable=True)

    from_temp_mid: Mapped[float | None] = mapped_column(Float, nullable=True)
    to_temp_mid: Mapped[float | None] = mapped_column(Float, nullable=True)
    temp_overlap: Mapped[float | None] = mapped_column(Float, nullable=True)
    temp_jump: Mapped[float | None] = mapped_column(Float, nullable=True)

    from_thickness: Mapped[float | None] = mapped_column(Float, nullable=True)
    to_thickness: Mapped[float | None] = mapped_column(Float, nullable=True)
    thickness_jump: Mapped[float | None] = mapped_column(Float, nullable=True)

    from_width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    to_width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    width_jump: Mapped[int | None] = mapped_column(Integer, nullable=True)

    reverse_flag: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    bridge_type: Mapped[str | None] = mapped_column(String(64), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.current_timestamp(), nullable=False)

