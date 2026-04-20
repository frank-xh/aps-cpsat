from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


DIRECT_EDGE = "DIRECT_EDGE"
REAL_BRIDGE_EDGE = "REAL_BRIDGE_EDGE"
VIRTUAL_BRIDGE_EDGE = "VIRTUAL_BRIDGE_EDGE"
VIRTUAL_BRIDGE_FAMILY_EDGE = "VIRTUAL_BRIDGE_FAMILY_EDGE"


@dataclass(frozen=True)
class TransitionCheckResult:
    """Rule check with an explain payload, not just a boolean."""

    hard_feasible: bool
    reason: str = "OK"
    explain: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CandidateEdge:
    """
    Unified edge representation consumed by graph/search layers.

    For VIRTUAL_BRIDGE_FAMILY_EDGE (family compressed edge):
    - bridge_family: one of "WIDTH_GROUP", "THICKNESS", "GROUP_TRANSITION", "MIXED"
    - estimated_bridge_count_min / _max: family-level estimate range
    - requires_pc_transition: True if family edge requires PC transition
    - metadata["explain_payload"]: family compression metadata

    Legacy VIRTUAL_BRIDGE_EDGE and DIRECT_EDGE / REAL_BRIDGE_EDGE work unchanged
    (estimated_bridge_count_min=max=estimated_bridge_count, requires_pc=False).
    """

    from_order_id: str
    to_order_id: str
    line: str
    edge_type: str
    bridge_family: str
    estimated_bridge_count: int = 0
    # ---- Family edge extension fields (unused for direct/real bridge) ----
    estimated_bridge_count_min: int = 0  # Family: minimum possible bridge count
    estimated_bridge_count_max: int = 0   # Family: maximum possible bridge count
    requires_pc_transition: bool = False  # Family: requires PC transition
    # ---- Standard fields ----
    dominant_fail_reason: str = "OK"
    hard_feasible: bool = True
    template_cost: float = 0.0
    estimated_reverse_cost: float = 0.0
    estimated_ton_effect: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_template_row(self) -> dict[str, Any]:
        return dict(self.metadata.get("template_row", {}))

    def effective_max_bridge_count(self) -> int:
        """Return the max bridge count for pruning (uses extended or legacy field)."""
        if self.estimated_bridge_count_max > 0:
            return self.estimated_bridge_count_max
        return self.estimated_bridge_count


@dataclass
class CandidateGraphBuildResult:
    """Candidate graph plus stable diagnostics for A/B comparisons."""

    edges: list[CandidateEdge] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def by_policy(self, *, allow_real: bool, allow_virtual: bool) -> list[CandidateEdge]:
        allowed: list[CandidateEdge] = []
        for edge in self.edges:
            if edge.edge_type == DIRECT_EDGE:
                allowed.append(edge)
            elif edge.edge_type == REAL_BRIDGE_EDGE and allow_real:
                allowed.append(edge)
            elif edge.edge_type in (VIRTUAL_BRIDGE_EDGE, VIRTUAL_BRIDGE_FAMILY_EDGE) and allow_virtual:
                # Both VIRTUAL_BRIDGE_EDGE and VIRTUAL_BRIDGE_FAMILY_EDGE filtered by same allow_virtual flag
                allowed.append(edge)
        return allowed
