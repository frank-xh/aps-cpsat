from __future__ import annotations

from aps_cp_sat.domain.models import ColdRollingRequest
from aps_cp_sat.routes.base_route import BaseRouteRunner, PreparedWorld, RouteSolveResult


class BlockFirstRouteRunner(BaseRouteRunner):
    # Deprecated: removed_from_mainline. Kept only so stale imports fail with a
    # clear route-removal error instead of silently dispatching block-first.
    route_name = "block_first"
    supported_profiles = {"block_first_guarded_search"}

    def solve(self, req: ColdRollingRequest, world: PreparedWorld) -> RouteSolveResult:
        raise RuntimeError(
            "[APS][BLOCK_FIRST_ROUTE_REMOVED] "
            "Only constructive_lns_virtual_guarded_frontload is supported."
        )
