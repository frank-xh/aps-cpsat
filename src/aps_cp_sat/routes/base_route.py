from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from aps_cp_sat.domain.models import ColdRollingRequest


@dataclass
class PreparedWorld:
    """Shared upstream artifacts used by all solver routes."""

    orders_df: pd.DataFrame
    normalized_orders_df: pd.DataFrame
    graph_orders_df: pd.DataFrame
    transition_pack: dict[str, Any]
    candidate_graph: Any | None = None
    prebuilt_virtual_inventory_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    shared_diagnostics: dict[str, Any] = field(default_factory=dict)
    candidate_graph_diagnostics: dict[str, Any] = field(default_factory=dict)
    template_build_seconds: float = 0.0
    data_diagnostics: dict[str, Any] = field(default_factory=dict)
    template_diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class RouteSolveResult:
    """Unified route output consumed by the pipeline orchestration layer."""

    schedule_df: pd.DataFrame
    rounds_df: pd.DataFrame
    dropped_df: pd.DataFrame
    engine_meta: dict[str, Any] = field(default_factory=dict)


class BaseRouteRunner(ABC):
    route_name: str = "unknown"
    supported_profiles: set[str] = set()

    def supports_profile(self, profile_name: str) -> bool:
        return str(profile_name or "") in self.supported_profiles

    @abstractmethod
    def solve(self, req: ColdRollingRequest, world: PreparedWorld) -> RouteSolveResult:
        raise NotImplementedError
