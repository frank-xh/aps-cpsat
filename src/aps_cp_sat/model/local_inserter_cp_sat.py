"""
Local Inserter CP-SAT: Local re-construction for ALNS.

This module implements the THIRD layer of the Constructive + ALNS path:
- Takes one or a few campaign segments (fixed + candidate orders)
- Tries to insert candidate orders into the fixed sequence using CP-SAT
- Strict mode: only edges that EXIST in template DataFrame are allowed
- No fallback to illegal edges — if no template edge exists, the order stays excluded

Key principles:
- Subproblem size is strictly bounded (default ≤ 45 orders)
- Only template-valid edges are modeled as variables
- Candidates may be partially accepted (not all-or-nothing)
- Fixed orders are preserved as much as possible

Reuses:
- local_router._template_total_cost for edge cost
- transition_pack["templates"] as the sole edge source
- OR-Tools CP-SAT AddCircuit and AddHint
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple
import sys

import pandas as pd
from ortools.sat.python import cp_model

from aps_cp_sat.config import PlannerConfig
from aps_cp_sat.model.local_router import _template_total_cost


class InsertStatus(Enum):
    """Status of the local insertion subproblem."""
    OPTIMAL = "OPTIMAL"
    FEASIBLE = "FEASIBLE"
    NO_IMPROVEMENT = "NO_IMPROVEMENT"
    INFEASIBLE = "INFEASIBLE"
    UNBOUNDED = "UNBOUNDED"
    UNKNOWN = "UNKNOWN"


@dataclass
class LocalInsertRequest:
    """
    Request to solve a local insertion subproblem.

    Attributes:
        line: Production line (big_roll or small_roll)
        fixed_order_ids: Orders that must appear in sequence (at least their order)
        candidate_insert_ids: Orders that may be inserted (may be rejected)
        time_limit_seconds: CP-SAT time limit for this subproblem
        random_seed: Random seed for CP-SAT
        max_orders_in_subproblem: Hard cap on total orders in subproblem
            (if fixed + candidates exceeds this, the largest candidates are dropped)
    """
    line: str
    fixed_order_ids: List[str]
    candidate_insert_ids: List[str]
    time_limit_seconds: float = 5.0
    random_seed: int = 42
    max_orders_in_subproblem: int = 45


@dataclass
class LocalInsertResult:
    """
    Result of solving a local insertion subproblem.

    Attributes:
        status: Solution status (OPTIMAL, FEASIBLE, INFEASIBLE, NO_IMPROVEMENT, ...)
        sequence: Final ordered list of order IDs (empty if infeasible)
        inserted_order_ids: Candidates that were accepted
        kept_order_ids: Fixed orders that remained in sequence
        dropped_candidate_ids: Candidates that were not accepted
        objective: Objective value (higher = better, lower = worse)
        accepted_candidate_count: Number of candidate orders accepted
        accepted_candidate_tons10: Accepted candidate tonnage * 10 (for diagnostics)
        diagnostics: Detailed statistics about the subproblem
    """
    status: InsertStatus
    sequence: List[str] = field(default_factory=list)
    inserted_order_ids: List[str] = field(default_factory=list)
    kept_order_ids: List[str] = field(default_factory=list)
    dropped_candidate_ids: List[str] = field(default_factory=list)
    objective: float = 0.0
    accepted_candidate_count: int = 0
    accepted_candidate_tons10: int = 0
    diagnostics: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Template edge graph builder (strict-only, no fallback)
# ---------------------------------------------------------------------------

@dataclass
class _SubproblemEdgeInfo:
    """Metadata for a single edge in the subproblem."""
    from_idx: int
    to_idx: int
    cost: int          # Template total cost (weighted)
    bridge_count: int
    is_virtual_bridge: bool
    tpl_row: dict      # Original template row for reference


class _StrictTemplateGraph:
    """
    Builds a strict subproblem graph from template DataFrame.

    Only edges that EXIST in the template DataFrame are included.
    No illegal edge creation, no fallback.

    Bridge edge filtering (Route B):
    - If allow_virtual_bridge_edge_in_constructive is False, VIRTUAL_BRIDGE_EDGE arcs are blocked
    - If allow_real_bridge_edge_in_constructive is False, REAL_BRIDGE_EDGE arcs are blocked
    - DIRECT_EDGE is always allowed
    """

    def __init__(
        self,
        order_ids: List[str],
        tpl_df: pd.DataFrame,
        cfg: PlannerConfig,
        line: str,
    ):
        self.order_ids = order_ids
        self.line = line
        self.cfg = cfg

        # Map order_id -> index in subproblem
        self.idx_map: Dict[str, int] = {oid: i for i, oid in enumerate(order_ids)}
        self.n = len(order_ids)

        # Build strict edge list
        self.edges: List[_SubproblemEdgeInfo] = []
        self.outgoing: Dict[int, List[_SubproblemEdgeInfo]] = {i: [] for i in range(self.n)}
        self.incoming: Dict[int, List[_SubproblemEdgeInfo]] = {i: [] for i in range(self.n)}

        # Bridge edge filtering config
        self.allow_virtual = getattr(cfg.model, "allow_virtual_bridge_edge_in_constructive", False)
        self.allow_real = getattr(cfg.model, "allow_real_bridge_edge_in_constructive", True)

        # Arc filtering diagnostics
        self.virtual_bridge_arcs_blocked: int = 0
        self.real_bridge_arcs_blocked: int = 0
        self.real_bridge_arcs_allowed: int = 0
        self.direct_arcs_allowed: int = 0

        # Determine edge policy string (Route C = direct_only)
        if not self.allow_virtual and self.allow_real:
            self.edge_policy_used: str = "direct_plus_real_bridge"
        elif not self.allow_virtual and not self.allow_real:
            self.edge_policy_used = "direct_only"  # Route C: strictest mode
        elif self.allow_virtual and self.allow_real:
            self.edge_policy_used = "all_edges_allowed"
        else:
            self.edge_policy_used = "virtual_only"

        if tpl_df is None or tpl_df.empty:
            return

        # Build from template DataFrame
        for _, row in tpl_df.iterrows():
            from_oid = str(row.get("from_order_id", ""))
            to_oid = str(row.get("to_order_id", ""))

            if from_oid not in self.idx_map or to_oid not in self.idx_map:
                continue

            # Filter by line
            tpl_line = str(row.get("line", "big_roll"))
            if tpl_line != self.line:
                continue

            from_idx = self.idx_map[from_oid]
            to_idx = self.idx_map[to_oid]

            # Only add if not already added (first occurrence wins)
            edge_key = (from_idx, to_idx)
            already_added = any(
                e.from_idx == from_idx and e.to_idx == to_idx
                for e in self.outgoing[from_idx]
            )
            if already_added:
                continue

            cost = _template_total_cost(dict(row), cfg.score)
            bridge_count = int(row.get("bridge_count", 0) or 0)
            edge_type = str(row.get("edge_type", "DIRECT_EDGE") or "DIRECT_EDGE")
            is_virtual = edge_type == "VIRTUAL_BRIDGE_EDGE"

            # ---- Bridge edge type filtering (Route B/C) ----
            if edge_type == "VIRTUAL_BRIDGE_EDGE" and not self.allow_virtual:
                self.virtual_bridge_arcs_blocked += 1
                continue  # Skip virtual bridge edge - do not create arc
            if edge_type == "REAL_BRIDGE_EDGE" and not self.allow_real:
                self.real_bridge_arcs_blocked += 1
                continue  # Skip real bridge edge - do not create arc
            # Track allowed edge counts for diagnostics
            if edge_type == "DIRECT_EDGE":
                self.direct_arcs_allowed += 1
            elif edge_type == "REAL_BRIDGE_EDGE" and self.allow_real:
                self.real_bridge_arcs_allowed += 1
            # DIRECT_EDGE is always allowed

            edge_info = _SubproblemEdgeInfo(
                from_idx=from_idx,
                to_idx=to_idx,
                cost=cost,
                bridge_count=bridge_count,
                is_virtual_bridge=is_virtual,
                tpl_row=dict(row),
            )
            self.edges.append(edge_info)
            self.outgoing[from_idx].append(edge_info)
            self.incoming[to_idx].append(edge_info)

    def has_outgoing(self, idx: int) -> bool:
        return len(self.outgoing.get(idx, [])) > 0

    def has_incoming(self, idx: int) -> bool:
        return len(self.incoming.get(idx, [])) > 0

    def has_edge(self, from_idx: int, to_idx: int) -> bool:
        return any(
            e.from_idx == from_idx and e.to_idx == to_idx
            for e in self.outgoing.get(from_idx, [])
        )

    def get_edge_info(self, from_idx: int, to_idx: int) -> Optional[_SubproblemEdgeInfo]:
        for e in self.outgoing.get(from_idx, []):
            if e.from_idx == from_idx and e.to_idx == to_idx:
                return e
        return None


# ---------------------------------------------------------------------------
# Hint builder: skeleton + candidates by width
# ---------------------------------------------------------------------------

def _build_hint_sequence(
    fixed_order_ids: List[str],
    candidate_ids: List[str],
    orders_df: pd.DataFrame,
    graph: _StrictTemplateGraph,
) -> Optional[List[int]]:
    """
    Build a CP-SAT hint sequence (list of indices).

    Strategy:
    1. Start with fixed_order_ids as backbone (preserving their relative order)
    2. Sort candidates by width descending
    3. Greedily insert each candidate at the best position
       (minimize template edge cost)
    4. Only use template-valid edges — skip insertion if no valid edge

    Returns:
        List of indices forming the hint path, or None if no valid hint exists.
    """
    if orders_df.empty:
        return None

    # Build order info lookup
    order_lookup = orders_df.set_index("order_id")

    # Map fixed orders to their indices in graph
    fixed_indices = []
    for oid in fixed_order_ids:
        if oid in graph.idx_map:
            fixed_indices.append(graph.idx_map[oid])

    # Map candidate orders to their indices
    cand_indices = []
    for oid in candidate_ids:
        if oid in graph.idx_map:
            cand_indices.append(graph.idx_map[oid])

    if not fixed_indices:
        return None

    # Sort candidates by width descending
    def get_width(idx: int) -> float:
        oid = graph.order_ids[idx]
        if oid in order_lookup.index:
            return float(order_lookup.loc[oid].get("width", 0) or 0)
        return 0.0

    cand_indices.sort(key=get_width, reverse=True)

    # Build hint sequence
    # Start with fixed backbone
    hint: List[int] = list(fixed_indices)
    used: Set[int] = set(fixed_indices)

    # Greedily insert candidates
    for ci in cand_indices:
        if ci in used:
            continue

        # Find best insertion point: between hint[i] and hint[i+1]
        best_pos = None
        best_cost = float("inf")

        # Try: insert at the beginning (before first fixed)
        if not hint:
            # No fixed orders in hint — skip
            continue

        # Positions to try: before each hint element (0..len(hint))
        for pos in range(len(hint) + 1):
            prev_idx = hint[pos - 1] if pos > 0 else None
            next_idx = hint[pos] if pos < len(hint) else None

            # Check if we can legally go prev -> ci -> next
            if prev_idx is not None and not graph.has_edge(prev_idx, ci):
                continue
            if next_idx is not None and not graph.has_edge(ci, next_idx):
                continue

            # Compute insertion cost
            cost = 0
            if prev_idx is not None:
                e = graph.get_edge_info(prev_idx, ci)
                if e:
                    cost += e.cost
            if next_idx is not None:
                e = graph.get_edge_info(ci, next_idx)
                if e:
                    cost += e.cost

            if cost < best_cost:
                best_cost = cost
                best_pos = pos

        if best_pos is not None:
            hint.insert(best_pos, ci)
            used.add(ci)

    # Add any remaining candidates that couldn't be inserted
    for ci in cand_indices:
        if ci not in used:
            # Append at end if last -> ci is valid
            if hint:
                last_idx = hint[-1]
                if graph.has_edge(last_idx, ci):
                    hint.append(ci)
                    used.add(ci)
            # Prepend at start if ci -> first is valid
            elif ci not in used:
                if graph.outgoing.get(ci) and not graph.incoming.get(ci):
                    hint.insert(0, ci)
                    used.add(ci)

    return hint if hint else None


# ---------------------------------------------------------------------------
# Path extractor: true single-circuit with depot
# ---------------------------------------------------------------------------

def _extract_path_from_selected_arcs(
    n: int,
    depot_idx: int,
    real_arc_vars: Dict[Tuple[int, int], cp_model.IntVar],
    start_vars: Dict[int, cp_model.IntVar],
    end_vars: Dict[int, cp_model.IntVar],
    self_loop_vars: Dict[int, cp_model.IntVar],
    solver: cp_model.CpSolver,
) -> Tuple[List[int], Dict]:
    """
    Extract the single depot path from solved AddCircuit variables.

    Returns:
        (path_indices, extraction_diag) where:
        - path_indices: ordered list of real-node indices (no depot)
        - extraction_diag: dict with debug info
    """
    diag: Dict = {
        "selected_start_node": None,
        "selected_end_node": None,
        "selected_real_arc_count": 0,
        "selected_self_loop_count": 0,
        "active_edges": [],
        "extracted_path_valid": False,
        "extraction_error": None,
    }

    try:
        # ---- 1. Find unique start node ----
        start_nodes = [
            i for i in range(n) if solver.Value(start_vars[i]) >= 1
        ]
        diag["selected_start_count"] = len(start_nodes)
        if len(start_nodes) != 1:
            diag["extraction_error"] = f"START_COUNT={len(start_nodes)}, expected 1"
            return [], diag
        start_node = start_nodes[0]
        diag["selected_start_node"] = start_node

        # ---- 2. Find unique end node ----
        end_nodes = [
            i for i in range(n) if solver.Value(end_vars[i]) >= 1
        ]
        diag["selected_end_count"] = len(end_nodes)
        if len(end_nodes) != 1:
            diag["extraction_error"] = f"END_COUNT={len(end_nodes)}, expected 1"
            return [], diag
        end_node = end_nodes[0]
        diag["selected_end_node"] = end_node

        if start_node == end_node and len(start_nodes) == 1:
            # Single-node circuit (only one order): path = [start_node]
            diag["selected_real_arc_count"] = 0
            diag["selected_self_loop_count"] = 0
            diag["active_edges"] = []
            diag["extracted_path_valid"] = True
            return [start_node], diag

        # ---- 3. Build active edge maps ----
        out_edges: Dict[int, List[int]] = {}
        in_edges: Dict[int, List[int]] = {}

        # Real arcs
        for (i, j), var in real_arc_vars.items():
            try:
                if solver.Value(var) >= 1:
                    out_edges.setdefault(i, []).append(j)
                    in_edges.setdefault(j, []).append(i)
                    diag["active_edges"].append((i, j))
                    diag["selected_real_arc_count"] += 1
            except Exception:
                pass

        # Self-loops
        for i in range(n):
            try:
                if solver.Value(self_loop_vars[i]) >= 1:
                    diag["selected_self_loop_count"] += 1
            except Exception:
                pass

        # ---- 4. Walk depot -> start -> ... -> end -> depot ----
        # Verify start has no incoming real arc
        if in_edges.get(start_node):
            diag["extraction_error"] = (
                f"START_HAS_INCOMING: start_node={start_node}, "
                f"incoming={in_edges[start_node]}"
            )
            return [], diag

        # Verify end has no outgoing real arc
        if out_edges.get(end_node):
            diag["extraction_error"] = (
                f"END_HAS_OUTGOING: end_node={end_node}, "
                f"outgoing={out_edges[end_node]}"
            )
            return [], diag

        # Walk from start to end following real arcs
        path: List[int] = []
        visited: Set[int] = set()
        current = start_node

        while current != end_node:
            if current in visited:
                diag["extraction_error"] = f"CYCLE_DETECTED: current={current}, path={path}"
                return [], diag
            if current not in out_edges or not out_edges[current]:
                diag["extraction_error"] = (
                    f"DEAD_END: current={current}, path={path}, "
                    f"outgoing={out_edges.get(current)}"
                )
                return [], diag

            successors = out_edges[current]
            if len(successors) > 1:
                diag["extraction_error"] = (
                    f"MULTIPLE_SUCCESSORS: current={current}, "
                    f"successors={successors}, path={path}"
                )
                return [], diag

            next_node = successors[0]
            path.append(current)
            visited.add(current)
            current = next_node

        # Add end node
        path.append(end_node)
        visited.add(end_node)

        # ---- 5. Verify no node has more than one incoming/outgoing ----
        for node in path:
            if node in out_edges and len(out_edges[node]) > 1:
                diag["extraction_error"] = f"MULTI_OUTGOING: node={node}"
                return [], diag
            if node in in_edges and len(in_edges[node]) > 1:
                diag["extraction_error"] = f"MULTI_INCOMING: node={node}"
                return [], diag

        diag["extracted_path_valid"] = True
        return path, diag

    except Exception as e:
        diag["extraction_error"] = f"EXCEPTION: {e}"
        return [], diag


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------

def solve_local_insertion_subproblem(
    req: LocalInsertRequest,
    orders_df: pd.DataFrame,
    transition_pack: dict,
    cfg: PlannerConfig,
) -> LocalInsertResult:
    """
    Solve a local insertion subproblem using CP-SAT.

    This function takes a small subproblem (fixed orders + candidates)
    and tries to insert as many candidates as possible using only
    template-valid edges.

    The model uses a depot node (index = n) to form a single circuit:
        depot -> start_node -> ... -> end_node -> depot

    Only edges that exist in the template DataFrame are allowed as real arcs.
    No fallback to illegal edges.

    Args:
        req: LocalInsertRequest with fixed/candidate orders and parameters
        orders_df: Full orders DataFrame
        transition_pack: Dict containing "templates" DataFrame
        cfg: PlannerConfig with rule and score parameters

    Returns:
        LocalInsertResult with status, sequence, diagnostics
    """
    # ---- 1. Prepare order lists ----
    fixed_ids = list(req.fixed_order_ids)
    candidate_ids = list(req.candidate_insert_ids)

    # Build order lookup
    order_lookup: Dict[str, dict] = {}
    if not orders_df.empty:
        for _, row in orders_df.iterrows():
            oid = str(row["order_id"])
            order_lookup[oid] = dict(row)

    # Filter: only orders that exist in orders_df
    all_fixed = [oid for oid in fixed_ids if oid in order_lookup]
    all_candidates = [oid for oid in candidate_ids if oid in order_lookup]

    # Cap subproblem size by dropping largest (tonnage) candidates first
    max_orders = req.max_orders_in_subproblem
    total_orders = all_fixed + all_candidates

    if len(total_orders) > max_orders:
        # Sort candidates by tons ascending (drop largest first to keep more orders)
        def get_tons(oid: str) -> float:
            return float(order_lookup.get(oid, {}).get("tons", 0) or 0)

        all_candidates.sort(key=get_tons, reverse=True)  # Largest first
        keep_candidates = all_candidates[: max_orders - len(all_fixed)]
        all_candidates = keep_candidates

    # Final subproblem order list
    subproblem_ids = list(all_fixed) + list(all_candidates)
    fixed_order_count = len(all_fixed)

    # ---- 2. Build strict template graph ----
    tpl_df = None
    if isinstance(transition_pack, dict):
        tpl_df = transition_pack.get("templates")
    if not isinstance(tpl_df, pd.DataFrame):
        tpl_df = pd.DataFrame()

    graph = _StrictTemplateGraph(subproblem_ids, tpl_df, cfg, req.line)

    # Check feasibility: at minimum, need at least one edge from fixed backbone
    has_any_edge = len(graph.edges) > 0
    if not has_any_edge or len(subproblem_ids) == 0:
        return LocalInsertResult(
            status=InsertStatus.INFEASIBLE,
            dropped_candidate_ids=list(all_candidates),
            diagnostics={
                "subproblem_order_count": len(subproblem_ids),
                "fixed_order_count": fixed_order_count,
                "fixed_order_pairs_count": 0,
                "fixed_order_precedence_enforced": False,
                "candidate_count": len(all_candidates),
                "edge_count_in_graph": len(graph.edges),
                "used_hint": False,
                "reason": "NO_VALID_TEMPLATE_EDGES",
                "objective_mode": "lexicographic_approx",
                "uses_true_addcircuit": True,
                "depot_idx": -1,
            },
        )

    # ---- 3. Build CP-SAT model with depot ----
    n = graph.n
    depot_idx = n  # depot is node n; total nodes = n + 1

    model = cp_model.CpModel()

    # Candidate indices
    fixed_indices_set = {graph.idx_map[oid] for oid in all_fixed if oid in graph.idx_map}
    candidate_indices = [graph.idx_map[oid] for oid in all_candidates if oid in graph.idx_map]
    real_nodes = list(range(n))

    # ---- 3a. Real arc variables: x[i][j] ----
    x: Dict[Tuple[int, int], cp_model.IntVar] = {}
    for e in graph.edges:
        x[(e.from_idx, e.to_idx)] = model.NewBoolVar(f"x_{e.from_idx}_{e.to_idx}")

    # ---- 3b. Depot arc variables ----
    # start[i] = 1 if depot -> i is in circuit
    start_vars: Dict[int, cp_model.IntVar] = {
        i: model.NewBoolVar(f"start_{i}") for i in real_nodes
    }
    # end[i] = 1 if i -> depot is in circuit
    end_vars: Dict[int, cp_model.IntVar] = {
        i: model.NewBoolVar(f"end_{i}") for i in real_nodes
    }

    # ---- 3c. Self-loop variables: s[i] = 1 means node i is NOT visited ----
    self_loop_vars: Dict[int, cp_model.IntVar] = {
        i: model.NewBoolVar(f"s_{i}") for i in real_nodes
    }

    # ---- 3d. Candidate acceptance: y[ci] = 1 - s[ci] ----
    y: Dict[int, cp_model.IntVar] = {
        ci: model.NewBoolVar(f"y_{ci}") for ci in candidate_indices
    }

    # ---- 3e. pos[i]: 0-based position in path for node i (0..n-1) ----
    pos: Dict[int, cp_model.IntVar] = {
        i: model.NewIntVar(0, max(1, n) - 1, f"pos_{i}") for i in real_nodes
    }

    # ---- 3f. Virtual bridge count var ----
    z_vb = model.NewIntVar(0, len(graph.edges), "z_vb")

    # ---- 4. Build arcs list for AddCircuit ----
    arcs: List[Tuple[int, int, cp_model.IntVar]] = []

    # A. Real arcs from template graph
    for e in graph.edges:
        arcs.append((e.from_idx, e.to_idx, x[(e.from_idx, e.to_idx)]))

    # B. Depot -> real node (start arcs)
    for i in real_nodes:
        arcs.append((depot_idx, i, start_vars[i]))

    # C. Real node -> depot (end arcs)
    for i in real_nodes:
        arcs.append((i, depot_idx, end_vars[i]))

    # D. Self-loop for each real node
    for i in real_nodes:
        arcs.append((i, i, self_loop_vars[i]))

    # E. NO depot self-loop

    # ---- 5. AddCircuit: enforces a single Hamiltonian circuit ----
    model.AddCircuit(arcs)

    # ---- 6. Exactly one start, exactly one end ----
    # This prevents the model from connecting multiple disjoint fragments to depot.
    model.Add(sum(start_vars.values()) == 1)
    model.Add(sum(end_vars.values()) == 1)

    # ---- 7. Candidate acceptance: y[ci] = 1 - self_loop[ci] ----
    for ci in candidate_indices:
        model.Add(y[ci] == 1 - self_loop_vars[ci])

    # ---- 8. Fixed nodes: must NOT self-loop (must be visited) ----
    for fi in fixed_indices_set:
        model.Add(self_loop_vars[fi] == 0)

    # ---- 9. pos propagation ----
    # If start[i] == 1, then pos[i] == 0 (first node in path)
    for i in real_nodes:
        model.Add(pos[i] == 0).OnlyEnforceIf(start_vars[i])

    # If real arc x[i,j] == 1, then pos[j] == pos[i] + 1
    for e in graph.edges:
        i, j = e.from_idx, e.to_idx
        model.Add(pos[j] == pos[i] + 1).OnlyEnforceIf(x[(i, j)])

    # ---- 10. Fixed-order precedence (relative order must be preserved) ----
    fixed_order_pairs: List[Tuple[int, int]] = []
    fixed_precedence_enforced = False
    failure_reason: Optional[str] = None

    for k in range(len(all_fixed) - 1):
        fk_oid = all_fixed[k]
        fn_oid = all_fixed[k + 1]
        if fk_oid in graph.idx_map and fn_oid in graph.idx_map:
            fixed_order_pairs.append((graph.idx_map[fk_oid], graph.idx_map[fn_oid]))

    fixed_order_pairs_count = len(fixed_order_pairs)

    if fixed_order_pairs:
        # BFS reachability check
        reachable_from: Dict[int, Set[int]] = {i: set() for i in range(n)}
        for src in range(n):
            visited_bfs: Set[int] = {src}
            queue = deque([src])
            while queue:
                cur = queue.popleft()
                for e in graph.outgoing.get(cur, []):
                    nxt = e.to_idx
                    if nxt not in visited_bfs:
                        visited_bfs.add(nxt)
                        queue.append(nxt)
            reachable_from[src] = visited_bfs

        # Check each pair
        precedence_feasible = True
        for (i, j) in fixed_order_pairs:
            if j not in reachable_from[i] or i in reachable_from[j]:
                precedence_feasible = False
                failure_reason = "FIXED_ORDER_PRECEDENCE_INFEASIBLE"
                break

        if not precedence_feasible:
            return LocalInsertResult(
                status=InsertStatus.INFEASIBLE,
                dropped_candidate_ids=list(all_candidates),
                diagnostics={
                    "subproblem_order_count": len(subproblem_ids),
                    "fixed_order_count": fixed_order_count,
                    "fixed_order_pairs_count": fixed_order_pairs_count,
                    "fixed_order_precedence_enforced": False,
                    "failure_reason": failure_reason,
                    "used_hint": False,
                    "objective_mode": "lexicographic_approx",
                    "uses_true_addcircuit": True,
                    "depot_idx": depot_idx,
                },
            )

        # Add hard precedence: pos[j] >= pos[i] + 1 for each fixed pair (i, j)
        for (i, j) in fixed_order_pairs:
            model.Add(pos[j] >= pos[i] + 1)

        fixed_precedence_enforced = True

    # Virtual bridge count
    vb_terms = []
    for e in graph.edges:
        if e.is_virtual_bridge:
            vb_terms.append(x[(e.from_idx, e.to_idx)])
    model.Add(z_vb == sum(vb_terms))

    # ---- 11. Objective (lexicographic) ----
    # Tier 1 (×10⁹): maximize accepted candidate COUNT
    # Tier 2 (×10⁵): maximize accepted candidate TONNAGE (×10 to preserve precision)
    # Tier 3 (×10¹): minimize total template edge COST
    # Tier 4 (×1):   minimize virtual bridge edge count
    # Coefficient gaps ensure no lower-tier improvement outweighs higher-tier.

    def get_tons(idx: int) -> float:
        oid = graph.order_ids[idx]
        return float(order_lookup.get(oid, {}).get("tons", 0) or 0)

    SCALE_COUNT = 10**9
    SCALE_TONS = 10**5
    SCALE_COST = 10**1
    OBJ_UB = 10**12

    objective = model.NewIntVar(-10**6, OBJ_UB, "objective")

    # Tier 1: accepted candidate count
    count_vars = [y[ci] for ci in candidate_indices]
    count_sum = model.NewIntVar(0, len(candidate_indices), "accept_count_sum")
    if count_vars:
        model.Add(count_sum == sum(count_vars))
    else:
        model.Add(count_sum == 0)

    # Tier 2: accepted candidate tonnage × 10
    tons10_vars = []
    for ci in candidate_indices:
        tons_i = get_tons(ci)
        tons10_i = int(tons_i * 10)
        t_var = model.NewIntVar(0, max(0, tons10_i), "tons10_%d" % ci)
        model.Add(t_var == tons10_i * y[ci])
        tons10_vars.append(t_var)

    tons10_sum = model.NewIntVar(0, 10**7, "accept_tons10_sum")
    if tons10_vars:
        model.Add(tons10_sum == sum(tons10_vars))
    else:
        model.Add(tons10_sum == 0)

    # Tier 3: template edge cost
    cost_vars = []
    for e in graph.edges:
        c_var = model.NewIntVar(0, max(1, e.cost), "cost_%d_%d" % (e.from_idx, e.to_idx))
        model.Add(c_var == e.cost * x[(e.from_idx, e.to_idx)])
        cost_vars.append(c_var)

    cost_sum = model.NewIntVar(0, 10**6, "template_cost_sum")
    if cost_vars:
        model.Add(cost_sum == sum(cost_vars))
    else:
        model.Add(cost_sum == 0)

    # Tier 4: virtual bridge count (z_vb already)
    tier1 = model.NewIntVar(0, OBJ_UB, "tier1_count")
    model.Add(tier1 == count_sum * SCALE_COUNT)

    tier2 = model.NewIntVar(0, OBJ_UB, "tier2_tons10")
    model.Add(tier2 == tons10_sum * SCALE_TONS)

    tier3 = model.NewIntVar(0, OBJ_UB, "tier3_cost")
    model.Add(tier3 == cost_sum * SCALE_COST)

    model.Add(objective == tier1 + tier2 - tier3 - z_vb)
    model.Maximize(objective)

    # ---- 12. AddHint ----
    hint = _build_hint_sequence(all_fixed, all_candidates, orders_df, graph)
    hint_used = False

    if hint is not None and len(hint) >= 1:
        hint_used = True
        # Build hint arcs: depot->hint[0], hint[i]->hint[i+1], hint[-1]->depot
        try:
            model.AddHint(objective, 0)
            # Depot -> first hint node
            if hint[0] in start_vars:
                model.AddHint(start_vars[hint[0]], 1)
            # Real arcs
            for k in range(len(hint) - 1):
                fi, ti = hint[k], hint[k + 1]
                if (fi, ti) in x:
                    model.AddHint(x[(fi, ti)], 1)
            # Last hint node -> depot
            if hint[-1] in end_vars:
                model.AddHint(end_vars[hint[-1]], 1)
            # Candidates in hint
            for ci in candidate_indices:
                if ci in y:
                    model.AddHint(y[ci], 1)
        except Exception:
            hint_used = False

    # ---- 13. Solve ----
    solver = cp_model.CpSolver()
    solver.parameters.log_search_progress = False
    solver.parameters.num_workers = 1
    solver.parameters.random_seed = req.random_seed
    solver.parameters.max_time_in_seconds = req.time_limit_seconds

    status = solver.Solve(model)

    # ---- 14. Extract and validate solution ----
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        path_indices, extract_diag = _extract_path_from_selected_arcs(
            n, depot_idx, x, start_vars, end_vars, self_loop_vars, solver
        )

        if not path_indices or not extract_diag.get("extracted_path_valid"):
            return LocalInsertResult(
                status=InsertStatus.INFEASIBLE,
                dropped_candidate_ids=list(all_candidates),
                diagnostics={
                    "subproblem_order_count": len(subproblem_ids),
                    "fixed_order_count": fixed_order_count,
                    "fixed_order_pairs_count": fixed_order_pairs_count,
                    "fixed_order_precedence_enforced": fixed_precedence_enforced,
                    "candidate_count": len(all_candidates),
                    "edge_count_in_graph": len(graph.edges),
                    "used_hint": hint_used,
                    "reason": "PATH_EXTRACTION_FAILED",
                    "extraction_diag": extract_diag,
                    "objective_mode": "lexicographic_approx",
                    "uses_true_addcircuit": True,
                    "depot_idx": depot_idx,
                },
            )

        # ---- Post-solution validation ----
        validation_errors: List[str] = []
        path_set = set(path_indices)

        # Check 1: all fixed nodes appear in path
        for fi in fixed_indices_set:
            if fi not in path_set:
                validation_errors.append(
                    f"FIXED_NODE_MISSING_IN_PATH: idx={fi}, "
                    f"oid={graph.order_ids[fi]}"
                )

        # Check 2: no duplicate nodes in path
        if len(path_indices) != len(path_set):
            validation_errors.append(
                f"DUPLICATE_NODES_IN_PATH: len={len(path_indices)}, "
                f"unique={len(path_set)}"
            )

        # Check 3: all adjacent pairs in path are valid template edges
        for k in range(len(path_indices) - 1):
            i, j = path_indices[k], path_indices[k + 1]
            if not graph.has_edge(i, j):
                validation_errors.append(
                    f"INVALID_EDGE_IN_PATH: {i}->{j} "
                    f"(oid {graph.order_ids[i]}->{graph.order_ids[j]})"
                )

        # Check 4: accepted candidates == path candidate nodes
        path_candidates = {
            ci for ci in candidate_indices if ci in path_set
        }
        accepted_set = set()
        for ci in candidate_indices:
            try:
                if solver.Value(y[ci]) >= 1:
                    accepted_set.add(ci)
            except Exception:
                pass
        if path_candidates != accepted_set:
            validation_errors.append(
                f"CANDIDATE_MISMATCH: path_candidates={sorted(path_candidates)}, "
                f"accepted={sorted(accepted_set)}"
            )

        if validation_errors:
            return LocalInsertResult(
                status=InsertStatus.INFEASIBLE,
                dropped_candidate_ids=list(all_candidates),
                diagnostics={
                    "subproblem_order_count": len(subproblem_ids),
                    "fixed_order_count": fixed_order_count,
                    "fixed_order_pairs_count": fixed_order_pairs_count,
                    "fixed_order_precedence_enforced": fixed_precedence_enforced,
                    "candidate_count": len(all_candidates),
                    "edge_count_in_graph": len(graph.edges),
                    "used_hint": hint_used,
                    "reason": "POST_SOLUTION_VALIDATION_FAILED",
                    "validation_errors": validation_errors,
                    "extraction_diag": extract_diag,
                    "objective_mode": "lexicographic_approx",
                    "uses_true_addcircuit": True,
                    "depot_idx": depot_idx,
                },
            )

        # ---- Build result ----
        sequence = [graph.order_ids[i] for i in path_indices]

        accepted: List[str] = []
        rejected: List[str] = []
        kept: List[str] = []

        for ci in candidate_indices:
            try:
                if solver.Value(y[ci]) >= 1:
                    accepted.append(graph.order_ids[ci])
                else:
                    rejected.append(graph.order_ids[ci])
            except Exception:
                rejected.append(graph.order_ids[ci])

        for fi in all_fixed:
            if fi in graph.idx_map:
                fi_idx = graph.idx_map[fi]
                if fi_idx in path_set:
                    kept.append(fi)

        # Objective components
        try:
            obj_val = float(solver.Value(objective))
        except Exception:
            obj_val = 0.0

        try:
            vb_count = int(solver.Value(z_vb))
        except Exception:
            vb_count = 0

        try:
            total_cost = int(solver.Value(cost_sum))
        except Exception:
            total_cost = 0

        try:
            accepted_count_sol = int(solver.Value(count_sum))
        except Exception:
            accepted_count_sol = len(accepted)

        try:
            accepted_tons10_sol = int(solver.Value(tons10_sum))
        except Exception:
            accepted_tons10_sol = 0

        if len(accepted) == 0:
            result_status = InsertStatus.NO_IMPROVEMENT
        elif status == cp_model.OPTIMAL:
            result_status = InsertStatus.OPTIMAL
        else:
            result_status = InsertStatus.FEASIBLE

        diagnostics = {
            # Standard diagnostics
            "subproblem_order_count": len(subproblem_ids),
            "fixed_order_count": fixed_order_count,
            "fixed_order_pairs_count": fixed_order_pairs_count,
            "fixed_order_precedence_enforced": fixed_precedence_enforced,
            "fixed_count": len(all_fixed),
            "candidate_count": len(all_candidates),
            "accepted_count": len(accepted),
            "rejected_count": len(rejected),
            "used_hint": hint_used,
            "selected_virtual_bridge_edge_count": vb_count,
            "total_template_cost": total_cost,
            "objective_value": obj_val,
            "objective_accept_count": accepted_count_sol,
            "objective_accept_tons10": accepted_tons10_sol,
            "objective_template_cost": total_cost,
            "objective_virtual_bridge_edges": vb_count,
            "objective_mode": "lexicographic_approx",
            "edge_count_in_graph": len(graph.edges),
            "path_length": len(path_indices),
            "solver_status": solver.StatusName(status),
            # New depot/AddCircuit diagnostics
            "uses_true_addcircuit": True,
            "depot_idx": depot_idx,
            "selected_start_count": int(extract_diag.get("selected_start_count", -1)),
            "selected_end_count": int(extract_diag.get("selected_end_count", -1)),
            "selected_real_arc_count": int(extract_diag.get("selected_real_arc_count", 0)),
            "selected_self_loop_count": int(extract_diag.get("selected_self_loop_count", 0)),
            "extracted_path_valid": bool(extract_diag.get("extracted_path_valid", False)),
            "extraction_error": extract_diag.get("extraction_error"),
            # Bridge edge filtering diagnostics (Route B/C)
            "virtual_bridge_arcs_blocked": graph.virtual_bridge_arcs_blocked,
            "real_bridge_arcs_blocked": graph.real_bridge_arcs_blocked,
            "real_bridge_arcs_allowed": graph.real_bridge_arcs_allowed,
            "direct_arcs_allowed": graph.direct_arcs_allowed,
            "edge_policy_used": graph.edge_policy_used,
        }

        return LocalInsertResult(
            status=result_status,
            sequence=sequence,
            inserted_order_ids=accepted,
            kept_order_ids=kept,
            dropped_candidate_ids=rejected,
            objective=obj_val,
            accepted_candidate_count=accepted_count_sol,
            accepted_candidate_tons10=accepted_tons10_sol,
            diagnostics=diagnostics,
        )

    elif status == cp_model.INFEASIBLE:
        return LocalInsertResult(
            status=InsertStatus.INFEASIBLE,
            dropped_candidate_ids=list(all_candidates),
            diagnostics={
                "subproblem_order_count": len(subproblem_ids),
                "fixed_order_count": fixed_order_count,
                "fixed_order_pairs_count": fixed_order_pairs_count,
                "fixed_order_precedence_enforced": fixed_precedence_enforced,
                "candidate_count": len(all_candidates),
                "edge_count_in_graph": len(graph.edges),
                "used_hint": hint_used,
                "reason": "MODEL_INFEASIBLE",
                "objective_mode": "lexicographic_approx",
                "uses_true_addcircuit": True,
                "depot_idx": depot_idx,
            },
        )

    else:
        status_map = {
            cp_model.UNBOUNDED: InsertStatus.UNBOUNDED,
            cp_model.MODEL_INVALID: InsertStatus.INFEASIBLE,
        }
        return LocalInsertResult(
            status=status_map.get(status, InsertStatus.UNKNOWN),
            dropped_candidate_ids=list(all_candidates),
            diagnostics={
                "subproblem_order_count": len(subproblem_ids),
                "fixed_order_count": fixed_order_count,
                "fixed_order_pairs_count": fixed_order_pairs_count,
                "fixed_order_precedence_enforced": fixed_precedence_enforced,
                "candidate_count": len(all_candidates),
                "edge_count_in_graph": len(graph.edges),
                "used_hint": hint_used,
                "solver_status": solver.StatusName(status),
                "objective_mode": "lexicographic_approx",
                "uses_true_addcircuit": True,
                "depot_idx": depot_idx,
            },
        )


__all__ = [
    "InsertStatus",
    "LocalInsertRequest",
    "LocalInsertResult",
    "solve_local_insertion_subproblem",
]
