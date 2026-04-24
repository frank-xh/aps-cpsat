from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import pandas as pd

from aps_cp_sat.model.edge_hard_filter import (
    accumulate_adj_hard_filter_rejection,
    edge_passes_final_hard_rules,
    log_adj_hard_filter,
)


@dataclass
class GraphUnusedCandidates:
    real_candidates: list[str] = field(default_factory=list)
    virtual_candidates: list[str] = field(default_factory=list)
    candidate_edges: list[dict[str, Any]] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _order_map(graph_orders_df: pd.DataFrame | None) -> dict[str, dict[str, Any]]:
    if not isinstance(graph_orders_df, pd.DataFrame) or graph_orders_df.empty or "order_id" not in graph_orders_df.columns:
        return {}
    return {str(row.get("order_id", "")): dict(row) for row in graph_orders_df.to_dict("records")}


def _is_virtual(row: dict[str, Any]) -> bool:
    raw = row.get("is_virtual", False)
    try:
        if pd.isna(raw):
            raw = False
    except Exception:
        pass
    return bool(raw) or str(row.get("virtual_origin", "")) == "prebuilt_inventory"


def _edge_iter(candidate_graph) -> Iterable[Any]:
    if candidate_graph is None:
        return []
    return list(getattr(candidate_graph, "edges", []) or [])


def collect_graph_unused_candidates(
    campaigns,
    used_node_ids: set[str] | None,
    candidate_graph,
    graph_orders_df: pd.DataFrame | None,
    line: str,
    boundary_nodes: list[str] | tuple[str, ...] | set[str],
    cfg,
) -> GraphUnusedCandidates:
    """Collect repair candidates only from unused nodes adjacent in CandidateGraph."""
    used = {str(x) for x in (used_node_ids or set())}
    boundaries = {str(x) for x in (boundary_nodes or []) if str(x)}
    records = _order_map(graph_orders_df)
    real: list[str] = []
    virtual: list[str] = []
    candidate_edges: list[dict[str, Any]] = []
    rejected_by_hard: dict[str, int] = {}
    adj_rejections: dict[str, int] = {}
    scanned = 0
    line_filtered = 0
    used_filtered = 0
    boundary_filtered = 0

    for edge in _edge_iter(candidate_graph):
        scanned += 1
        edge_line = str(getattr(edge, "line", "") or "")
        if edge_line != str(line):
            line_filtered += 1
            continue
        from_oid = str(getattr(edge, "from_order_id", "") or "")
        to_oid = str(getattr(edge, "to_order_id", "") or "")
        if boundaries and from_oid not in boundaries:
            boundary_filtered += 1
            continue
        if to_oid in used:
            used_filtered += 1
            continue
        prev = records.get(from_oid, {})
        nxt = records.get(to_oid, {})
        tpl_row = edge.to_template_row() if hasattr(edge, "to_template_row") else {}
        ok, reason = edge_passes_final_hard_rules(
            prev,
            nxt,
            cfg,
            {
                "line": line,
                "edge_type": str(getattr(edge, "edge_type", "") or tpl_row.get("edge_type", "")),
                "template_row": tpl_row,
                "bridge_count": int(getattr(edge, "estimated_bridge_count", 0) or 0),
            },
        )
        if not ok:
            rejected_by_hard[reason] = rejected_by_hard.get(reason, 0) + 1
            adj_rejections = accumulate_adj_hard_filter_rejection(adj_rejections, reason)
            continue
        rec = records.get(to_oid, {})
        if _is_virtual(rec):
            virtual.append(to_oid)
        else:
            real.append(to_oid)
        candidate_edges.append(
            {
                "from_order_id": from_oid,
                "to_order_id": to_oid,
                "line": edge_line,
                "edge_type": str(getattr(edge, "edge_type", "") or ""),
                "template_row": tpl_row,
            }
        )

    total = len(real) + len(virtual)
    log_adj_hard_filter("graph_unused_pool", adj_rejections)
    return GraphUnusedCandidates(
        real_candidates=real,
        virtual_candidates=virtual,
        candidate_edges=candidate_edges,
        diagnostics={
            "candidate_source": "graph_unused_nodes",
            "graph_unused_candidates_total": int(total),
            "graph_unused_real_candidates": int(len(real)),
            "graph_unused_virtual_candidates": int(len(virtual)),
            "graph_unused_edges_total": int(len(candidate_edges)),
            "graph_unused_edges_scanned": int(scanned),
            "graph_unused_filtered_by_line": int(line_filtered),
            "graph_unused_filtered_by_boundary": int(boundary_filtered),
            "graph_unused_filtered_by_used": int(used_filtered),
            "graph_unused_rejected_by_hard_filter_total": int(sum(rejected_by_hard.values())),
            "graph_unused_rejected_by_hard_filter": rejected_by_hard,
        },
    )
