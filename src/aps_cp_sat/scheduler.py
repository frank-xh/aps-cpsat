from dataclasses import dataclass
from typing import List, Optional

from ortools.sat.python import cp_model


@dataclass(frozen=True)
class Job:
    name: str
    duration: int
    due_date: Optional[int] = None


@dataclass(frozen=True)
class ScheduleResult:
    objective_value: int
    order: List[str]
    starts: List[int]
    ends: List[int]


def solve_single_machine_schedule(jobs: List[Job], time_limit_seconds: float = 10.0) -> ScheduleResult:
    if not jobs:
        raise ValueError("jobs must not be empty")
    if any(job.duration <= 0 for job in jobs):
        raise ValueError("all job durations must be positive")

    model = cp_model.CpModel()
    horizon = sum(job.duration for job in jobs)

    starts = []
    ends = []
    intervals = []
    tardiness_vars = []

    for i, job in enumerate(jobs):
        start = model.NewIntVar(0, horizon, f"start_{i}_{job.name}")
        end = model.NewIntVar(0, horizon, f"end_{i}_{job.name}")
        interval = model.NewIntervalVar(start, job.duration, end, f"interval_{i}_{job.name}")

        starts.append(start)
        ends.append(end)
        intervals.append(interval)

        if job.due_date is not None:
            t = model.NewIntVar(0, horizon, f"tardiness_{i}_{job.name}")
            model.Add(t >= end - job.due_date)
            tardiness_vars.append(t)

    model.AddNoOverlap(intervals)

    if tardiness_vars:
        model.Minimize(sum(tardiness_vars))
    else:
        makespan = model.NewIntVar(0, horizon, "makespan")
        model.AddMaxEquality(makespan, ends)
        model.Minimize(makespan)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_seconds
    solver.parameters.num_search_workers = 1
    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("No feasible schedule found")

    scheduled = []
    for i, job in enumerate(jobs):
        s = solver.Value(starts[i])
        e = solver.Value(ends[i])
        scheduled.append((s, e, job.name))

    scheduled.sort(key=lambda x: x[0])
    order = [name for _, _, name in scheduled]
    start_vals = [s for s, _, _ in scheduled]
    end_vals = [e for _, e, _ in scheduled]

    return ScheduleResult(
        objective_value=int(solver.ObjectiveValue()),
        order=order,
        starts=start_vals,
        ends=end_vals,
    )

