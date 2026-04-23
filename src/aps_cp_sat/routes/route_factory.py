from __future__ import annotations

from aps_cp_sat.routes.base_route import BaseRouteRunner
from aps_cp_sat.routes.constructive_lns_route import ConstructiveLnsRouteRunner


def _infer_strategy(profile_name: str) -> str:
    profile = str(profile_name or "")
    if profile == "constructive_lns_virtual_guarded_frontload":
        return "constructive_lns"
    raise ValueError(
        "[APS][ROUTE_FACTORY][ONLY_SINGLE_ROUTE_ALLOWED] "
        "expected strategy=constructive_lns, "
        "expected profile=constructive_lns_virtual_guarded_frontload, "
        f"got profile={profile!r}"
    )


def create_route_runner(main_solver_strategy: str | None, profile_name: str | None) -> BaseRouteRunner:
    strategy = str(main_solver_strategy or "").strip()
    profile = str(profile_name or "").strip()
    if not strategy:
        strategy = _infer_strategy(profile)

    if strategy != "constructive_lns":
        raise ValueError(
            "[APS][ROUTE_FACTORY][ONLY_SINGLE_ROUTE_ALLOWED] "
            "expected strategy=constructive_lns, "
            "expected profile=constructive_lns_virtual_guarded_frontload, "
            f"got strategy={strategy!r}, profile={profile!r}"
        )
    runner: BaseRouteRunner = ConstructiveLnsRouteRunner()

    if not runner.supports_profile(profile):
        raise ValueError(
            "[APS][ROUTE_FACTORY][ONLY_SINGLE_ROUTE_ALLOWED] "
            "expected strategy=constructive_lns, "
            "expected profile=constructive_lns_virtual_guarded_frontload, "
            f"got strategy={strategy!r}, profile={profile!r}"
        )
    return runner
