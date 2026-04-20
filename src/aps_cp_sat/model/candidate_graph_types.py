from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from aps_cp_sat.config import PlannerConfig


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
    metadata: dict[str, Any] = field(default_factory=dict, hash=False, compare=False)

    # Custom hash: only use hashable fields to avoid dict hash issue
    def __hash__(self) -> int:
        return hash((
            self.from_order_id,
            self.to_order_id,
            self.line,
            self.edge_type,
            self.bridge_family,
            self.estimated_bridge_count,
            self.estimated_bridge_count_min,
            self.estimated_bridge_count_max,
            self.requires_pc_transition,
            self.dominant_fail_reason,
            self.hard_feasible,
            self.template_cost,
            self.estimated_reverse_cost,
            self.estimated_ton_effect,
        ))

    def to_template_row(self) -> dict[str, Any]:
        return dict(self.metadata.get("template_row", {}))

    def effective_max_bridge_count(self) -> int:
        """Return the max bridge count for pruning (uses extended or legacy field)."""
        if self.estimated_bridge_count_max > 0:
            return self.estimated_bridge_count_max
        return self.estimated_bridge_count


def is_virtual_family_frontload_eligible(
    edge: CandidateEdge,
    cfg: "PlannerConfig | None" = None,
    context: dict[str, Any] | None = None,
) -> tuple[bool, str]:
    """
    Check if a VIRTUAL_BRIDGE_FAMILY_EDGE is eligible for guarded frontload.

    Returns (eligible, reason) tuple.

    Eligibility gates (in order):
    1. Must be VIRTUAL_BRIDGE_FAMILY_EDGE, NOT legacy VIRTUAL_BRIDGE_EDGE
    2. bridge_family must be in virtual_family_frontload_allowed_families
    3. estimated_bridge_count_max <= virtual_family_frontload_max_bridge_count
    4. Block tons must be within [min_block_tons, max_block_tons]
    5. legacy VIRTUAL_BRIDGE_EDGE is NEVER eligible (always blocked)

    Context-aware gating (if virtual_family_frontload_only_when_underfill_or_drop_pressure):
    - Requires context["underfill_detected"] or context["drop_pressure_detected"]
    - This is optional; if context is None, skip this check (backwards compatible)
    """
    if edge.edge_type != VIRTUAL_BRIDGE_FAMILY_EDGE:
        return False, "NOT_VIRTUAL_BRIDGE_FAMILY_EDGE"

    if cfg is None:
        return False, "NO_CFG"

    model = getattr(cfg, "model", None)
    if model is None:
        return False, "NO_MODEL_CONFIG"

    if not getattr(model, "virtual_family_frontload_enabled", False):
        return False, "FRONTLOAD_DISABLED"

    # Gate 1: family allowlist
    allowed_families = getattr(model, "virtual_family_frontload_allowed_families", [])
    if allowed_families:
        bridge_family = str(edge.metadata.get("explain_payload", {}).get("bridge_family", ""))
        if bridge_family not in allowed_families:
            return False, f"FAMILY_NOT_ALLOWED:{bridge_family}"

    # Gate 2: max bridge count
    max_bridge_count = int(getattr(model, "virtual_family_frontload_max_bridge_count", 10))
    if edge.effective_max_bridge_count() > max_bridge_count:
        return False, f"BRIDGE_COUNT_EXCEEDED:{edge.effective_max_bridge_count()}>{max_bridge_count}"

    # Gate 3: block tons range
    explain_payload = edge.metadata.get("explain_payload", {})
    virtual_tons = float(explain_payload.get("virtual_tons", 0.0) or 0.0)
    min_tons = float(getattr(model, "virtual_family_frontload_min_block_tons", 0.0) or 0.0)
    max_tons = float(getattr(model, "virtual_family_frontload_max_block_tons", 999999.0) or 999999.0)
    if virtual_tons < min_tons or virtual_tons > max_tons:
        return False, f"BLOCK_TONS_OUT_OF_RANGE:{virtual_tons}<{min_tons}or>{max_tons}"

    # Gate 4: context-aware underfill / drop pressure (optional)
    if context is not None and getattr(model, "virtual_family_frontload_only_when_underfill_or_drop_pressure", False):
        underfill = bool(context.get("underfill_detected", False))
        drop_pressure = bool(context.get("drop_pressure_detected", False))
        if not (underfill or drop_pressure):
            return False, "NO_UNDERFILL_OR_DROP_PRESSURE"

    return True, "ELIGIBLE"


@dataclass
class CandidateGraphBuildResult:
    """
    Candidate graph plus stable diagnostics for A/B comparisons.

    Dual-pool design:
      - edges (global pool): strict family edges for greedy constructive.
        Strict: topk_per (3) + global_max (360).
      - repair_family_edges (repair pool): wider family edges for ALNS
        repair + local rebuild. Wider: (topk_per + alns_extra_topk=4) + repair_max (900).
    """

    edges: list[CandidateEdge] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    repair_family_edges: list[CandidateEdge] = field(default_factory=list)
    repair_family_edge_count: int = 0
    repair_family_edge_pool_diagnostics: dict[str, Any] = field(default_factory=dict)

    def by_policy(
        self,
        *,
        allow_real: bool,
        allow_virtual: bool,
        cfg: "PlannerConfig | None" = None,
        context: dict[str, Any] | None = None,
        apply_frontload_filter: bool = True,
    ) -> list[CandidateEdge]:
        """
        Filter edges by edge policy.

        Policy options (determined by allow_real + allow_virtual):
          direct_only              : DIRECT only
          direct_plus_real_bridge  : DIRECT + REAL
          direct_plus_real_plus_guarded_family : DIRECT + REAL + eligible VIRTUAL_BRIDGE_FAMILY_EDGE
          all_edges_allowed        : DIRECT + REAL + ALL virtual (legacy + family, experimental only)

        Args:
            allow_real: allow REAL_BRIDGE_EDGE
            allow_virtual: allow VIRTUAL_BRIDGE_EDGE / VIRTUAL_BRIDGE_FAMILY_EDGE
            cfg: PlannerConfig (needed for guarded family eligibility check)
            context: optional context dict for underfill/drop-pressure gating
            apply_frontload_filter: if True, apply frontload eligibility gating to family edges
        """
        allowed: list[CandidateEdge] = []
        for edge in self.edges:
            if edge.edge_type == DIRECT_EDGE:
                allowed.append(edge)
            elif edge.edge_type == REAL_BRIDGE_EDGE and allow_real:
                allowed.append(edge)
            elif edge.edge_type == VIRTUAL_BRIDGE_FAMILY_EDGE and allow_virtual:
                # Guarded family edge: apply frontload eligibility if cfg is available
                if apply_frontload_filter and cfg is not None:
                    eligible, _ = is_virtual_family_frontload_eligible(edge, cfg, context)
                    if not eligible:
                        continue
                allowed.append(edge)
            elif edge.edge_type == VIRTUAL_BRIDGE_EDGE and allow_virtual:
                # Legacy VIRTUAL_BRIDGE_EDGE: only allowed in all_edges_allowed mode
                # In guarded family mode, legacy virtual is ALWAYS blocked
                continue
        return allowed
