"""
Bridge subsystem: Virtual Family Edge + Bridge Realization Oracle.

Files:
- types.py: BridgeFamily, RealizationFailReason, RealizationResult,
           OracleContext, BridgeRealizationOracle
- __init__.py: public API re-exports

Architecture:
- VIRTUAL_BRIDGE_FAMILY_EDGE enters Candidate Graph as compressed family-level edge
- BridgeRealizationOracle.realize() resolves family edges to exact paths
- Legacy virtual pilot repair remains in model/ but is marked as legacy path
- Future: virtual pilot will be refactored as an oracle realization strategy

Current status (v1):
- Oracle interface fully wired
- Real bridge: basic feasibility check skeleton
- Virtual family: NOT_IMPLEMENTED_YET stub (exact path resolution for next iteration)
"""

from aps_cp_sat.bridge.types import (
    BridgeFamily,
    RealizationFailReason,
    RealizationResult,
    OracleContext,
    BridgeRealizationOracle,
    realize_virtual_bridge_legacy,
)

__all__ = [
    "BridgeFamily",
    "RealizationFailReason",
    "RealizationResult",
    "OracleContext",
    "BridgeRealizationOracle",
    "realize_virtual_bridge_legacy",
]