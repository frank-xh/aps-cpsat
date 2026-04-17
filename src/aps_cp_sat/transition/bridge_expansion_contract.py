"""
Bridge Expansion Contract.

This module defines the unified contract for bridge metadata handling across
the constructive_lns path, providing helper functions for:

1. Normalizing bridge metadata fields
2. Checking if an edge is expandable
3. Querying bridge expansion mode
4. Validating bridge metadata consistency

This contract is designed to support:
- Route B (bridge_expansion_mode="disabled"): No virtual expansion
- Route A (bridge_expansion_mode="virtual_expand"): Full virtual bridge expansion

Usage:
    from aps_cp_sat.transition.bridge_expansion_contract import (
        normalize_selected_bridge_metadata,
        edge_is_expandable,
        get_bridge_expansion_mode,
        validate_bridge_metadata_consistency,
    )
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class BridgeEdgeType:
    """Enum-like constants for edge types."""
    DIRECT_EDGE = "DIRECT_EDGE"
    REAL_BRIDGE_EDGE = "REAL_BRIDGE_EDGE"
    VIRTUAL_BRIDGE_EDGE = "VIRTUAL_BRIDGE_EDGE"

    @classmethod
    def all(cls) -> List[str]:
        return [cls.DIRECT_EDGE, cls.REAL_BRIDGE_EDGE, cls.VIRTUAL_BRIDGE_EDGE]


class BridgeExpansionMode:
    """Enum-like constants for bridge expansion modes."""
    DISABLED = "disabled"
    VIRTUAL_EXPAND = "virtual_expand"

    @classmethod
    def all(cls) -> List[str]:
        return [cls.DISABLED, cls.VIRTUAL_EXPAND]


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def get_bridge_expansion_mode(cfg_or_meta: Any) -> str:
    """
    Get bridge_expansion_mode from config or engine metadata.

    Args:
        cfg_or_meta: Either a PlannerConfig object or an engine_meta dict.

    Returns:
        Bridge expansion mode string: "disabled" (default) or "virtual_expand".
    """
    if isinstance(cfg_or_meta, dict):
        # Direct dict (engine_meta or similar)
        return str(getattr(cfg_or_meta, "bridge_expansion_mode", "disabled")
                   if hasattr(cfg_or_meta, "bridge_expansion_mode")
                   else cfg_or_meta.get("bridge_expansion_mode", "disabled"))

    # Assume it's a config object with model attribute
    return str(getattr(cfg_or_meta, "bridge_expansion_mode",
                       getattr(cfg_or_meta, "model", None) and
                       getattr(cfg_or_meta.model, "bridge_expansion_mode", "disabled")
                       or "disabled"))


def edge_is_expandable(edge_type: str, bridge_expand_mode: str) -> bool:
    """
    Check if an edge type is expandable under the given bridge expansion mode.

    Args:
        edge_type: One of DIRECT_EDGE, REAL_BRIDGE_EDGE, VIRTUAL_BRIDGE_EDGE
        bridge_expand_mode: One of "disabled", "virtual_expand"

    Returns:
        True if the edge can be expanded, False otherwise.

    Notes:
        - DIRECT_EDGE: Never expandable (it's already a direct edge)
        - REAL_BRIDGE_EDGE: Never expandable (real bridge is not virtual)
        - VIRTUAL_BRIDGE_EDGE: Expandable only when bridge_expand_mode == "virtual_expand"
    """
    if bridge_expand_mode == BridgeExpansionMode.VIRTUAL_EXPAND:
        return edge_type == BridgeEdgeType.VIRTUAL_BRIDGE_EDGE
    else:
        # DISABLED or unknown mode: nothing is expandable
        return False


def normalize_selected_bridge_metadata(
    edge_type: str,
    bridge_path: str,
    tpl_row: Optional[Dict] = None,
    cfg_or_meta: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Normalize bridge metadata for a single edge.

    This function provides unified semantics for bridge metadata fields,
    ensuring consistent output format across the constructive_lns path.

    Args:
        edge_type: Edge type string ("DIRECT_EDGE", "REAL_BRIDGE_EDGE", "VIRTUAL_BRIDGE_EDGE")
        bridge_path: Bridge path string (e.g., "prev_oid|curr_oid" or "")
        tpl_row: Optional template row dict containing bridge metadata
        cfg_or_meta: Optional config object or engine_meta dict for expansion mode

    Returns:
        Dict with keys:
            - selected_edge_type: str
            - selected_bridge_path: str
            - selected_bridge_expandable: bool
            - selected_bridge_expand_mode: str
            - selected_virtual_bridge_count: int
            - selected_real_bridge_order_id: str
    """
    # Get bridge expansion mode
    bridge_expand_mode = get_bridge_expansion_mode(cfg_or_meta) if cfg_or_meta else "disabled"

    # Default values
    result = {
        "selected_edge_type": edge_type,
        "selected_bridge_path": bridge_path or "",
        "selected_bridge_expandable": False,
        "selected_bridge_expand_mode": bridge_expand_mode,
        "selected_virtual_bridge_count": 0,
        "selected_real_bridge_order_id": "",
    }

    if edge_type == BridgeEdgeType.DIRECT_EDGE:
        result["selected_bridge_expandable"] = False
        result["selected_bridge_expand_mode"] = "disabled"
        result["selected_bridge_path"] = ""

    elif edge_type == BridgeEdgeType.REAL_BRIDGE_EDGE:
        result["selected_bridge_expandable"] = False  # Real bridge is not expandable
        result["selected_bridge_expand_mode"] = "disabled"
        result["selected_virtual_bridge_count"] = 0
        # Extract real bridge order_id from template row if available
        if tpl_row:
            result["selected_real_bridge_order_id"] = str(
                tpl_row.get("bridge_order_id",
                           tpl_row.get("real_bridge_order_id", ""))
            )

    elif edge_type == BridgeEdgeType.VIRTUAL_BRIDGE_EDGE:
        # Virtual bridge is expandable in principle
        if bridge_expand_mode == BridgeExpansionMode.DISABLED:
            # Route B: expandable=True (could expand in future), but not expanded
            result["selected_bridge_expandable"] = True
            result["selected_bridge_expand_mode"] = "disabled"
            result["selected_virtual_bridge_count"] = 0  # Not expanded yet
        elif bridge_expand_mode == BridgeExpansionMode.VIRTUAL_EXPAND:
            # Route A: expandable and expanded
            result["selected_bridge_expandable"] = True
            result["selected_bridge_expand_mode"] = "virtual_expand"
            if tpl_row:
                result["selected_virtual_bridge_count"] = int(
                    tpl_row.get("bridge_count",
                               tpl_row.get("virtual_bridge_count", 0) or 0)
                )

    return result


def validate_bridge_metadata_consistency(
    df_rows: List[Dict],
    bridge_expand_mode: str,
) -> Tuple[bool, List[str]]:
    """
    Validate consistency of bridge metadata across planned_df rows.

    Checks:
    1. In disabled mode: VIRTUAL_BRIDGE_EDGE should not appear (it should be filtered upstream)
    2. In disabled mode: selected_bridge_expandable should be False (nothing is expanded)
    3. selected_edge_type should be one of the known edge types
    4. Real bridge edges should have selected_real_bridge_order_id set

    Args:
        df_rows: List of row dicts from planned_df
        bridge_expand_mode: Current bridge expansion mode

    Returns:
        (is_valid, warnings): Tuple of validity flag and list of warning messages
    """
    warnings: List[str] = []
    is_valid = True

    virtual_bridge_in_disabled = 0

    for i, row in enumerate(df_rows):
        edge_type = str(row.get("selected_edge_type", "DIRECT_EDGE"))

        # Check 1: Edge type should be known
        if edge_type not in BridgeEdgeType.all():
            warnings.append(
                f"Row {i}: Unknown edge_type '{edge_type}', expected one of {BridgeEdgeType.all()}"
            )
            is_valid = False

        # Check 2: In disabled mode, VIRTUAL_BRIDGE_EDGE should not appear
        if bridge_expand_mode == BridgeExpansionMode.DISABLED:
            if edge_type == BridgeEdgeType.VIRTUAL_BRIDGE_EDGE:
                virtual_bridge_in_disabled += 1
                warnings.append(
                    f"Row {i}: VIRTUAL_BRIDGE_EDGE found in disabled mode! "
                    f"order_id={row.get('order_id', '?')}"
                )
                is_valid = False

            # Check 3: In disabled mode, nothing should be expandable
            expand_mode = str(row.get("selected_bridge_expand_mode", "disabled"))
            if expand_mode == BridgeExpansionMode.VIRTUAL_EXPAND:
                warnings.append(
                    f"Row {i}: Bridge expand_mode='virtual_expand' in disabled mode! "
                    f"order_id={row.get('order_id', '?')}"
                )
                is_valid = False

        # Check 4: Real bridge edges should have real_bridge_order_id
        if edge_type == BridgeEdgeType.REAL_BRIDGE_EDGE:
            real_bridge_oid = str(row.get("selected_real_bridge_order_id", ""))
            if not real_bridge_oid:
                warnings.append(
                    f"Row {i}: REAL_BRIDGE_EDGE missing selected_real_bridge_order_id! "
                    f"order_id={row.get('order_id', '?')}"
                )

    # Summary
    if virtual_bridge_in_disabled > 0:
        warnings.append(
            f"Total: {virtual_bridge_in_disabled} VIRTUAL_BRIDGE_EDGE rows found in disabled mode"
        )

    return is_valid, warnings


def get_bridge_statistics(df_rows: List[Dict]) -> Dict[str, int]:
    """
    Get bridge statistics from planned_df rows.

    Returns:
        Dict with:
            - total_rows: int
            - direct_edge_count: int
            - real_bridge_edge_count: int
            - virtual_bridge_edge_count: int
            - expandable_count: int (edges that are expandable but not yet expanded)
    """
    stats = {
        "total_rows": len(df_rows),
        "direct_edge_count": 0,
        "real_bridge_edge_count": 0,
        "virtual_bridge_edge_count": 0,
        "expandable_count": 0,
    }

    for row in df_rows:
        edge_type = str(row.get("selected_edge_type", "DIRECT_EDGE"))

        if edge_type == BridgeEdgeType.DIRECT_EDGE:
            stats["direct_edge_count"] += 1
        elif edge_type == BridgeEdgeType.REAL_BRIDGE_EDGE:
            stats["real_bridge_edge_count"] += 1
        elif edge_type == BridgeEdgeType.VIRTUAL_BRIDGE_EDGE:
            stats["virtual_bridge_edge_count"] += 1

        # Count expandable edges (virtual edges with expand_mode != disabled)
        if str(row.get("selected_bridge_expandable", False)) == "True":
            expand_mode = str(row.get("selected_bridge_expand_mode", "disabled"))
            if expand_mode != BridgeExpansionMode.DISABLED:
                stats["expandable_count"] += 1

    return stats


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    # Constants
    "BridgeEdgeType",
    "BridgeExpansionMode",
    # Functions
    "get_bridge_expansion_mode",
    "edge_is_expandable",
    "normalize_selected_bridge_metadata",
    "validate_bridge_metadata_consistency",
    "get_bridge_statistics",
]
