import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from aps_cp_sat import Job, solve_single_machine_schedule


def main() -> None:
    jobs = [
        Job(name="A", duration=3, due_date=5),
        Job(name="B", duration=2, due_date=4),
        Job(name="C", duration=4, due_date=10),
    ]
    result = solve_single_machine_schedule(jobs)

    print(f"Objective: {result.objective_value}")
    for name, start, end in zip(result.order, result.starts, result.ends):
        print(f"{name}: [{start}, {end})")


if __name__ == "__main__":
    main()

