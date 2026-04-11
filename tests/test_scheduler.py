import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from aps_cp_sat import Job, solve_single_machine_schedule


def test_schedule_has_all_jobs():
    jobs = [
        Job(name="J1", duration=2, due_date=3),
        Job(name="J2", duration=1, due_date=4),
        Job(name="J3", duration=3, due_date=8),
    ]
    result = solve_single_machine_schedule(jobs)

    assert sorted(result.order) == ["J1", "J2", "J3"]
    assert len(result.starts) == 3
    assert len(result.ends) == 3

