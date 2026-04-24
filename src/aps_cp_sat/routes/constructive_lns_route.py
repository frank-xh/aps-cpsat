from __future__ import annotations

from aps_cp_sat.domain.models import ColdRollingRequest
from aps_cp_sat.model import solve_master_model
from aps_cp_sat.routes.base_route import BaseRouteRunner, PreparedWorld, RouteSolveResult


class ConstructiveLnsRouteRunner(BaseRouteRunner):
    route_name = "constructive_lns"
    supported_profiles = {"constructive_lns_virtual_guarded_frontload"}

    def solve(self, req: ColdRollingRequest, world: PreparedWorld) -> RouteSolveResult:
        schedule_df, rounds_df, dropped_df, engine_meta = solve_master_model(
            req,
            transition_pack=world.transition_pack,
            orders_df=world.graph_orders_df,
        )
        meta = dict(engine_meta or {})
        meta["profile_name"] = "constructive_lns_virtual_guarded_frontload"
        meta["route_name"] = self.route_name
        meta["solver_path"] = self.route_name
        meta["main_path"] = self.route_name
        meta["strategy_name"] = self.route_name
        meta["template_build_seconds"] = float(world.template_build_seconds)
        meta.setdefault(
            "constructive_lns_alns_rounds",
            int(getattr(req.config.model, "constructive_lns_alns_rounds", 0) or 0),
        )
        meta.setdefault(
            "constructive_lns_alns_enable_tail_repair",
            bool(getattr(req.config.model, "constructive_lns_alns_enable_tail_repair", True)),
        )
        meta.setdefault(
            "constructive_lns_alns_enable_direct_insert",
            bool(getattr(req.config.model, "constructive_lns_alns_enable_direct_insert", True)),
        )
        meta.setdefault(
            "constructive_lns_alns_enable_campaign_merge",
            bool(getattr(req.config.model, "constructive_lns_alns_enable_campaign_merge", True)),
        )
        meta.setdefault(
            "constructive_lns_alns_enable_bridge_insertion",
            bool(getattr(req.config.model, "constructive_lns_alns_enable_bridge_insertion", True)),
        )
        meta.setdefault(
            "constructive_lns_alns_enable_virtual_inventory_moves",
            bool(getattr(req.config.model, "constructive_lns_alns_enable_virtual_inventory_moves", False)),
        )
        # Keep original order metadata available for decode/final audit column recovery.
        # The constructive_lns schedule rows are intentionally lightweight; final
        # validation columns such as temp_min/temp_max must be recovered from the
        # normalized graph orders rather than silently backfilled as missing.
        meta.setdefault("source_orders_df", world.graph_orders_df.copy())
        meta.setdefault("graph_orders_df", world.graph_orders_df.copy())
        meta.setdefault(
            "order_lookup",
            {
                str(row.get("order_id", "")): dict(row)
                for row in world.graph_orders_df.to_dict("records")
                if str(row.get("order_id", "") or "")
            },
        )
        for key, value in world.candidate_graph_diagnostics.items():
            meta.setdefault(key, value)
        for key, value in world.shared_diagnostics.items():
            meta.setdefault(key, value)
        return RouteSolveResult(
            schedule_df=schedule_df,
            rounds_df=rounds_df,
            dropped_df=dropped_df,
            engine_meta=meta,
        )
