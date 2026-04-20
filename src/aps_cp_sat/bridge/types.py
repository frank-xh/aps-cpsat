"""
Bridge domain types for the Virtual Family Edge + Bridge Realization Oracle system.

Architecture intent:
- Legacy VIRTUAL_BRIDGE_EDGE stays in candidate_graph_types.py as-is.
- VIRTUAL_BRIDGE_FAMILY_EDGE is a compressed "family" representation that enters
  the Candidate Graph but does NOT carry exact spec details.
- BridgeRealizationOracle is the future resolution layer that turns family edges
  (and real bridge edges) into exact paths.
- Legacy virtual pilot repair remains functional but is marked as "legacy repair path".

Virtual family edge fields:
- from_order_id, to_order_id: endpoint order IDs
- line: production line (big_roll / small_roll)
- edge_type = VIRTUAL_BRIDGE_FAMILY_EDGE (constant in candidate_graph_types.py)
- bridge_family: one of WIDTH_GROUP | THICKNESS | GROUP_TRANSITION | MIXED
- estimated_bridge_count_min / _max: family-level estimate range
- estimated_cost: family-level cost estimate
- estimated_reverse_cost: reverse direction estimate
- estimated_ton_effect: estimated ton rescue effect
- requires_pc_transition: bool, True if family edge requires PC transition
- explain_payload: dict with family compression metadata
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


# -----------------------------------------------------------------------------
# Bridge Family Types
# -----------------------------------------------------------------------------

class BridgeFamily(Enum):
    """Compression family for virtual bridge edges."""
    WIDTH_GROUP = auto()      # Family by width proximity band
    THICKNESS = auto()        # Family by thickness compatibility
    GROUP_TRANSITION = auto() # Family for steel group transitions (non-PC)
    MIXED = auto()            # Mixed family (lower priority)


# -----------------------------------------------------------------------------
# Oracle Fail Reasons
# -----------------------------------------------------------------------------

class RealizationFailReason(Enum):
    """
    Canonical reasons why a bridge edge cannot be realized.

    These are the standardized failure codes returned by BridgeRealizationOracle.
    Legacy repair paths may use their own codes; oracle codes are for the new
    oracle-driven resolution system.
    """
    OK = "OK"                                    # Successfully realized
    NOT_IMPLEMENTED_YET = "NOT_IMPLEMENTED_YET"  # Oracle stub: feature not ready
    NO_REAL_DONOR = "NO_REAL_DONOR"              # No real bridge order available
    VIRTUAL_CHAIN_TOO_LONG = "VIRTUAL_CHAIN_TOO_LONG"  # Chain exceeds max_virtual_chain
    LEFT_TRANSITION_INVALID = "LEFT_TRANSITION_INVALID"  # from order transition impossible
    RIGHT_TRANSITION_INVALID = "RIGHT_TRANSITION_INVALID"  # to order transition impossible
    GROUP_TRANSITION_IMPOSSIBLE = "GROUP_TRANSITION_IMPOSSIBLE"  # group switch not allowed
    TON_RESCUE_IMPOSSIBLE = "TON_RESCUE_IMPOSSIBLE"  # ton rescue below minimum
    RESOURCE_CONFLICT = "RESOURCE_CONFLICT"      # Real bridge resource already used


# -----------------------------------------------------------------------------
# Realization Result
# -----------------------------------------------------------------------------

@dataclass
class RealizationResult:
    """
    Result of BridgeRealizationOracle.resolve(edge, context).

    This is the canonical output for all edge realization attempts:
    - feasible: True if the edge can be materialized
    - realization_type: "real" | "virtual_family" | "virtual_exact" | "failed"
    - exact_path: list of exact order specs for the bridge path
      - Each entry: dict with order_id, width, thickness, steel_group, tons, seq
    - realized_bridge_count: exact count of bridge coils
    - realized_cost: exact cost (may differ from estimated_* in edge)
    - fail_reason: canonical reason from RealizationFailReason
    - fail_detail: human-readable explanation
    - diagnostics: structured diagnostic payload
    """
    feasible: bool = False
    realization_type: str = "failed"  # "real" | "virtual_family" | "virtual_exact" | "failed"
    exact_path: list[dict] = field(default_factory=list)
    realized_bridge_count: int = 0
    realized_cost: float = 0.0
    fail_reason: RealizationFailReason = RealizationFailReason.OK
    fail_detail: str = ""
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def not_implemented(cls, edge_type: str, explain: str = "") -> RealizationResult:
        """Factory: oracle stub returns NOT_IMPLEMENTED_YET."""
        return cls(
            feasible=False,
            realization_type="failed",
            exact_path=[],
            realized_bridge_count=0,
            realized_cost=0.0,
            fail_reason=RealizationFailReason.NOT_IMPLEMENTED_YET,
            fail_detail=(
                f"{explain or 'Virtual family edge realization not yet implemented'}; "
                f"edge_type={edge_type}"
            ),
            diagnostics={"edge_type": edge_type, "stub": True},
        )

    @classmethod
    def ok_real(cls, path: list[dict], cost: float, diagnostics: dict[str, Any] | None = None) -> RealizationResult:
        """Factory: successful real bridge realization."""
        return cls(
            feasible=True,
            realization_type="real",
            exact_path=path,
            realized_bridge_count=len(path),
            realized_cost=cost,
            fail_reason=RealizationFailReason.OK,
            fail_detail="",
            diagnostics=diagnostics or {},
        )

    @classmethod
    def ok_virtual_exact(cls, path: list[dict], bridge_count: int, cost: float,
                         diagnostics: dict[str, Any] | None = None) -> RealizationResult:
        """Factory: successful virtual exact path realization."""
        return cls(
            feasible=True,
            realization_type="virtual_exact",
            exact_path=path,
            realized_bridge_count=bridge_count,
            realized_cost=cost,
            fail_reason=RealizationFailReason.OK,
            fail_detail="",
            diagnostics=diagnostics or {},
        )


# -----------------------------------------------------------------------------
# Oracle Context
# -----------------------------------------------------------------------------

@dataclass
class OracleContext:
    """
    Context available to BridgeRealizationOracle when resolving an edge.

    This provides the oracle with everything it needs to make a feasibility
    decision without requiring direct access to internal data structures.
    """
    orders_df: Any = None       # Full orders DataFrame
    tpl_df: Any = None          # Template DataFrame (for real bridge lookup)
    config: Any = None          # PlannerConfig
    line: str = "big_roll"      # Production line
    used_bridge_orders: set = field(default_factory=set)  # Real bridge orders already used
    available_donors: list = field(default_factory=list)   # Candidate real bridge donors
    max_virtual_chain: int = 5  # Config limit on virtual chain length

    @classmethod
    def from_config(
        cls,
        orders_df,
        tpl_df,
        config,
        line: str = "big_roll",
        used_bridge_orders: set | None = None,
    ) -> OracleContext:
        """Factory: build context from config, letting oracle access what's needed."""
        max_chain = 5
        if config and hasattr(config, "model"):
            max_chain = getattr(config.model, "max_virtual_chain", 5)
        return cls(
            orders_df=orders_df,
            tpl_df=tpl_df,
            config=config,
            line=line,
            used_bridge_orders=used_bridge_orders or set(),
            available_donors=[],
            max_virtual_chain=max_chain,
        )


# -----------------------------------------------------------------------------
# BridgeRealizationOracle Protocol
# -----------------------------------------------------------------------------

class BridgeRealizationOracle:
    """
    Unified interface for resolving bridge edges to exact paths.

    This is the future mainline interface. Legacy repair paths (virtual pilot,
    repair-only bridge) remain functional but are marked as legacy.

    realize() signature:
        realize(edge: CandidateEdge, context: OracleContext) -> RealizationResult

    Realization types:
    - "real": exact path through real bridge orders
    - "virtual_exact": exact path for virtual bridge (family -> exact spec)
    - "failed": could not resolve; fail_reason populated

    Example usage:
        oracle = BridgeRealizationOracle()
        result = oracle.realize(candidate_edge, context)
        if result.feasible:
            # materialize exact_path
        else:
            # handle fail_reason
    """

    def realize(self, edge: Any, context: OracleContext) -> RealizationResult:
        """
        Resolve a candidate edge to an exact path.

        This is the main entry point. Subclasses or strategies can override
        _realize_real_bridge and _realize_virtual_family for type-specific logic.

        Args:
            edge: CandidateEdge (or dict-like with edge_type, from_order_id, etc.)
            context: OracleContext with orders, templates, config, etc.

        Returns:
            RealizationResult with feasible=True/False and full path details.
        """
        edge_type = str(getattr(edge, "edge_type", "") or "")
        from_id = str(getattr(edge, "from_order_id", "") or "")
        to_id = str(getattr(edge, "to_order_id", "") or "")

        if edge_type == "REAL_BRIDGE_EDGE":
            return self._realize_real_bridge(edge, context)
        elif edge_type in ("VIRTUAL_BRIDGE_EDGE", "VIRTUAL_BRIDGE_FAMILY_EDGE"):
            return self._realize_virtual_family(edge, context)
        else:
            return RealizationResult.not_implemented(
                edge_type,
                explain=f"Unknown edge_type: {edge_type}",
            )

    def _realize_real_bridge(self, edge: Any, context: OracleContext) -> RealizationResult:
        """
        Resolve REAL_BRIDGE_EDGE to exact path.

        Base implementation: basic feasibility check.
        Looks for real_bridge_order_id in edge metadata.

        Override in subclasses for full realization logic.
        """
        metadata = dict(getattr(edge, "metadata", {}) or {})
        real_bridge_oid = str(metadata.get("real_bridge_order_id", "") or "")

        if not real_bridge_oid:
            return RealizationResult(
                feasible=False,
                realization_type="failed",
                fail_reason=RealizationFailReason.NO_REAL_DONOR,
                fail_detail="real_bridge_order_id not found in edge metadata",
                diagnostics={"edge_type": "REAL_BRIDGE_EDGE", "real_bridge_order_id": ""},
            )

        # Check if this real bridge order is already used
        if real_bridge_oid in context.used_bridge_orders:
            return RealizationResult(
                feasible=False,
                realization_type="failed",
                fail_reason=RealizationFailReason.RESOURCE_CONFLICT,
                fail_detail=f"real_bridge_order_id={real_bridge_oid} already used",
                diagnostics={
                    "edge_type": "REAL_BRIDGE_EDGE",
                    "real_bridge_order_id": real_bridge_oid,
                    "used_bridge_orders": list(context.used_bridge_orders),
                },
            )

        # Look up the real bridge order record
        orders_df = context.orders_df
        if hasattr(orders_df, "to_dict"):
            # DataFrame lookup
            row = next(
                (dict(r) for _, r in orders_df.iterrows() if str(r.get("order_id", "")) == real_bridge_oid),
                None,
            )
        else:
            row = None

        if row is None:
            return RealizationResult(
                feasible=False,
                realization_type="failed",
                fail_reason=RealizationFailReason.NO_REAL_DONOR,
                fail_detail=f"real_bridge_order_id={real_bridge_oid} not found in orders_df",
                diagnostics={"real_bridge_order_id": real_bridge_oid},
            )

        # Build exact path with single real bridge order
        exact_path = [dict(row)]
        template_cost = float(metadata.get("template_cost", 0) or 0)
        realized_cost = float(row.get("bridge_penalty", template_cost) if "bridge_penalty" in row else template_cost)

        return RealizationResult.ok_real(
            path=exact_path,
            cost=realized_cost,
            diagnostics={
                "edge_type": "REAL_BRIDGE_EDGE",
                "real_bridge_order_id": real_bridge_oid,
                "from_order_id": str(getattr(edge, "from_order_id", "")),
                "to_order_id": str(getattr(edge, "to_order_id", "")),
            },
        )

    def _realize_virtual_family(
        self, edge: Any, context: OracleContext
    ) -> RealizationResult:
        """
        Resolve VIRTUAL_BRIDGE_FAMILY_EDGE (or VIRTUAL_BRIDGE_EDGE) to exact path.

        Base implementation: returns NOT_IMPLEMENTED_YET.
        The oracle skeleton is in place; exact spec resolution is for next iteration.

        This is the stub that guarantees:
        1. Interface is fully wired
        2. Calls return properly structured RealizationResult
        3. Diagnostics are collected
        4. Fail reasons are standardized
        """
        metadata = dict(getattr(edge, "metadata", {}) or {})
        bridge_family = str(metadata.get("bridge_family", "MIXED"))
        bridge_count = int(getattr(edge, "estimated_bridge_count", 0) or 0)

        # Check chain length constraint
        if bridge_count > context.max_virtual_chain:
            return RealizationResult(
                feasible=False,
                realization_type="failed",
                fail_reason=RealizationFailReason.VIRTUAL_CHAIN_TOO_LONG,
                fail_detail=(
                    f"estimated_bridge_count={bridge_count} exceeds max_virtual_chain="
                    f"{context.max_virtual_chain}"
                ),
                diagnostics={
                    "edge_type": getattr(edge, "edge_type", "VIRTUAL_BRIDGE_FAMILY_EDGE"),
                    "bridge_family": bridge_family,
                    "estimated_bridge_count": bridge_count,
                    "max_virtual_chain": context.max_virtual_chain,
                },
            )

        # Stub: NOT_IMPLEMENTED_YET
        return RealizationResult.not_implemented(
            edge_type=getattr(edge, "edge_type", "VIRTUAL_BRIDGE_FAMILY_EDGE"),
            explain=(
                f"Virtual family edge realization not yet implemented; "
                f"bridge_family={bridge_family}, estimated_bridge_count={bridge_count}, "
                f"requires_pc_transition={metadata.get('requires_pc_transition', False)}. "
                f"Next iteration: implement exact path construction for family={bridge_family}."
            ),
        )


# -----------------------------------------------------------------------------
# Stub for legacy virtual pilot (marked as legacy)
# -----------------------------------------------------------------------------

def realize_virtual_bridge_legacy(
    from_order: dict,
    to_order: dict,
    context: OracleContext,
) -> RealizationResult:
    """
    Legacy stub for virtual bridge repair path.

    Marked as LEGACY: this is the old virtual bridge pilot path.
    Future work should replace this with oracle-driven realization.

    Note: This function is provided as a shim for legacy code that calls
    virtual bridge pilot logic. New code should use BridgeRealizationOracle.
    """
    return RealizationResult.not_implemented(
        edge_type="VIRTUAL_BRIDGE_EDGE",
        explain="Legacy virtual bridge realization; use BridgeRealizationOracle for new code",
    )