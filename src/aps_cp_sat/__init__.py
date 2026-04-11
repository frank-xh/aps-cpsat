from __future__ import annotations

from aps_cp_sat.domain.models import ColdRollingRequest, ColdRollingResult

__all__ = [
    "Job",
    "ScheduleResult",
    "solve_single_machine_schedule",
    "ColdRollingPipeline",
    "ColdRollingRequest",
    "ColdRollingResult",
]


def __getattr__(name: str):
    if name in {"Job", "ScheduleResult", "solve_single_machine_schedule"}:
        from aps_cp_sat.scheduler import Job, ScheduleResult, solve_single_machine_schedule

        return {"Job": Job, "ScheduleResult": ScheduleResult, "solve_single_machine_schedule": solve_single_machine_schedule}[name]
    if name == "ColdRollingPipeline":
        from aps_cp_sat.cold_rolling_pipeline import ColdRollingPipeline

        return ColdRollingPipeline
    raise AttributeError(name)
