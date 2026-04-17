from __future__ import annotations

"""
Transitional slot-local router.

This module is allowed in the production path as a transitional component after
the joint master has already fixed line/slot assignment. It is not an
independent master-model entry and must not replace the joint master.
"""

from typing import Dict, List, Tuple

import pandas as pd
from ortools.sat.python import cp_model

from aps_cp_sat.config import PlannerConfig


def _template_total_cost(row: dict, score) -> int:
    base = float(row.get("cost", 0.0))
    width_smooth = float(row.get("width_smooth_cost", 0.0))
    thick_smooth = float(row.get("thickness_smooth_cost", 0.0))
    temp_margin = float(row.get("temp_margin_cost", 0.0))
    cross_group = float(row.get("cross_group_cost", 0.0))
    bridge_cnt = float(row.get("bridge_count", 0.0))
    logical_reverse = int(row.get("logical_reverse_flag", 0) or 0)
    edge_type = str(row.get("edge_type", "DIRECT_EDGE") or "DIRECT_EDGE")
    composed = (
        score.width_smooth * width_smooth
        + score.thick_smooth * thick_smooth
        + score.temp_margin * temp_margin
        + score.non_pc_switch * cross_group
        + score.virtual_use * bridge_cnt
    )
    if edge_type == "REAL_BRIDGE_EDGE":
        composed += float(score.real_bridge_penalty)
    elif edge_type == "VIRTUAL_BRIDGE_EDGE":
        composed += float(score.virtual_bridge_penalty)
    else:
        composed += float(score.direct_edge_penalty)
    if logical_reverse > 0 and bridge_cnt > 0:
        composed += float(score.reverse_width_bridge_penalty) * bridge_cnt
    return int(round(max(0.0, composed + float(score.template_base_cost_ratio) * base)))


def _edge_cost_fallback(a: dict, b: dict, score, width_anchor: float) -> int:
    width_d = abs(float(a.get("width", 0.0)) - float(b.get("width", 0.0)))
    thick_d = abs(float(a.get("thickness", 0.0)) - float(b.get("thickness", 0.0)))
    due_inv = max(0, int(a.get("due_rank", 3)) - int(b.get("due_rank", 3)))
    anchor_bias = max(0.0, width_anchor - float(b.get("width", 0.0)))
    return int(
        int(score.edge_fallback_width_weight) * width_d
        + int(score.edge_fallback_thick_weight) * thick_d
        + int(score.edge_fallback_due_weight) * due_inv
        + int(score.edge_fallback_base_penalty)
        + int(anchor_bias * 0.1)
    )


def _build_unroutable_slot_diagnostics(slot_df: pd.DataFrame, tpl_df: pd.DataFrame) -> dict:
    orders = slot_df.copy()
    order_ids = [str(v) for v in orders.get("order_id", pd.Series(dtype=str)).tolist()]
    order_count = int(len(order_ids))
    possible_pairs = max(0, order_count * max(0, order_count - 1))
    tpl = tpl_df.copy() if isinstance(tpl_df, pd.DataFrame) else pd.DataFrame()
    tpl_pairs = (
        set(zip(tpl["from_order_id"].astype(str), tpl["to_order_id"].astype(str)))
        if not tpl.empty and {"from_order_id", "to_order_id"}.issubset(set(tpl.columns))
        else set()
    )
    present_pairs = int(len(tpl_pairs))
    missing_pair_count = int(max(0, possible_pairs - present_pairs))
    out_deg = tpl.groupby("from_order_id").size().to_dict() if not tpl.empty else {}
    in_deg = tpl.groupby("to_order_id").size().to_dict() if not tpl.empty else {}
    zero_out_orders = [oid for oid in order_ids if int(out_deg.get(oid, 0)) <= 0]
    zero_in_orders = [oid for oid in order_ids if int(in_deg.get(oid, 0)) <= 0]
    top_isolated_orders = list(dict.fromkeys((zero_in_orders + zero_out_orders)))[:5]
    steel_groups = sorted(
        {
            str(v)
            for v in orders.get("steel_group", pd.Series(dtype=str)).fillna("").astype(str).tolist()
            if str(v)
        }
    )
    return {
        "order_count": order_count,
        "template_coverage_ratio": round(present_pairs / max(1, possible_pairs), 4) if possible_pairs > 0 else 1.0,
        "missing_pair_count": missing_pair_count,
        "zero_in_orders": int(len(zero_in_orders)),
        "zero_out_orders": int(len(zero_out_orders)),
        "min_width": float(orders["width"].min()) if "width" in orders.columns and not orders.empty else 0.0,
        "max_width": float(orders["width"].max()) if "width" in orders.columns and not orders.empty else 0.0,
        "min_thickness": float(orders["thickness"].min()) if "thickness" in orders.columns and not orders.empty else 0.0,
        "max_thickness": float(orders["thickness"].max()) if "thickness" in orders.columns and not orders.empty else 0.0,
        "steel_group_count": int(len(steel_groups)),
        "estimated_reverse_count": int(tpl["logical_reverse_flag"].sum()) if "logical_reverse_flag" in tpl.columns and not tpl.empty else 0,
        "estimated_virtual_blocks": int(tpl["bridge_count"].sum()) if "bridge_count" in tpl.columns and not tpl.empty else 0,
        "top_isolated_orders": top_isolated_orders,
        "reverse_count_definition": "logical_reverse_per_campaign",
    }


def _solve_slot_route_with_templates(
    slot_df: pd.DataFrame,
    tpl_df: pd.DataFrame,
    time_limit: float,
    seed: int,
    cfg: PlannerConfig,
) -> dict:
    if slot_df.empty:
        return {
            "status": "EMPTY",
            "sequence": [],
            "strict_template_edges_enabled": bool(cfg.model.strict_template_edges),
            "missing_template_edge_count": 0,
            "diagnostics": {},
        }
    if len(slot_df) == 1:
        return {
            "status": "ROUTED",
            "sequence": [str(slot_df.iloc[0]["order_id"])],
            "strict_template_edges_enabled": bool(cfg.model.strict_template_edges),
            "missing_template_edge_count": 0,
            "diagnostics": _build_unroutable_slot_diagnostics(slot_df, tpl_df),
        }

    records = slot_df.to_dict("records")
    ids = [str(r["order_id"]) for r in records]
    rec_by = {str(r["order_id"]): r for r in records}
    idx_of = {oid: i + 1 for i, oid in enumerate(ids)}
    strict_template_edges = bool(cfg.model.strict_template_edges)

    tpl_best: Dict[Tuple[str, str], dict] = {}
    if isinstance(tpl_df, pd.DataFrame) and not tpl_df.empty:
        for row in tpl_df.to_dict("records"):
            oi = str(row.get("from_order_id", ""))
            oj = str(row.get("to_order_id", ""))
            if oi not in idx_of or oj not in idx_of or oi == oj:
                continue
            key = (oi, oj)
            cost = _template_total_cost(row, cfg.score)
            prev = tpl_best.get(key)
            if prev is None or cost < int(prev["cost"]):
                tpl_best[key] = {
                    "cost": cost,
                    "logical_reverse_flag": int(row.get("logical_reverse_flag", 0) or 0),
                }

    missing_template_edge_count = 0
    diagnostics = _build_unroutable_slot_diagnostics(slot_df, tpl_df)
    model = cp_model.CpModel()
    arcs = []
    var_map: Dict[Tuple[int, int], cp_model.IntVar] = {}
    obj_terms = []
    reverse_terms = []
    for oid in ids:
        i = idx_of[oid]
        s = model.NewBoolVar(f"start_{i}")
        e = model.NewBoolVar(f"end_{i}")
        var_map[(0, i)] = s
        var_map[(i, 0)] = e
        arcs.append((0, i, s))
        arcs.append((i, 0, e))
        rec = rec_by[oid]
        obj_terms.append(int(200 * int(rec.get("due_rank", 3))) * s)
    for oi in ids:
        for oj in ids:
            if oi == oj:
                continue
            i = idx_of[oi]
            j = idx_of[oj]
            tpl = tpl_best.get((oi, oj))
            if tpl is None:
                missing_template_edge_count += 1
                # Task 3: HARD CONSTRAINT - Prohibit this arc entirely if no valid template edge exists
                # No soft fallback: model cannot connect orders without a valid template transition
                v = model.NewBoolVar(f"x_{i}_{j}")
                var_map[(i, j)] = v
                arcs.append((i, j, v))
                # Hard prohibition: this arc MUST NOT be selected in any solution
                model.Add(v == 0)
                continue
            v = model.NewBoolVar(f"x_{i}_{j}")
            var_map[(i, j)] = v
            arcs.append((i, j, v))
            obj_terms.append(int(tpl["cost"]) * v)
            if tpl is not None:
                is_logical_reverse = bool(int(tpl.get("logical_reverse_flag", 0) or 0) > 0)
            else:
                is_logical_reverse = float(rec_by[oj].get("width", 0.0)) > float(rec_by[oi].get("width", 0.0))
            if is_logical_reverse:
                reverse_terms.append(v)
    diagnostics["missing_template_edge_count"] = int(missing_template_edge_count)
    if strict_template_edges:
        node_out = {idx_of[oid]: 0 for oid in ids}
        node_in = {idx_of[oid]: 0 for oid in ids}
        for i, j in var_map:
            if i == 0 or j == 0:
                continue
            node_out[i] += 1
            node_in[j] += 1
        if any(v == 0 for v in node_out.values()) or any(v == 0 for v in node_in.values()):
            return {
                "status": "UNROUTABLE_SLOT",
                "sequence": [],
                "strict_template_edges_enabled": True,
                "missing_template_edge_count": int(missing_template_edge_count),
                "diagnostics": diagnostics,
            }
    model.Add(sum(reverse_terms) <= int(cfg.rule.max_logical_reverse_per_campaign))
    model.AddCircuit(arcs)
    model.Minimize(sum(obj_terms))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max(1.0, float(time_limit))
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = int(seed)
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return {
            "status": "UNROUTABLE_SLOT" if strict_template_edges else "ROUTER_FALLBACK_SORT",
            "sequence": [] if strict_template_edges else sorted(
                ids,
                key=lambda oid: (-float(rec_by[oid].get("width", 0.0)), -float(rec_by[oid].get("thickness", 0.0))),
            ),
            "strict_template_edges_enabled": bool(strict_template_edges),
            "missing_template_edge_count": int(missing_template_edge_count),
            "diagnostics": diagnostics,
        }

    succ: Dict[str, str | None] = {oid: None for oid in ids}
    start_oid: str | None = None
    for (i, j), var in var_map.items():
        if solver.Value(var) != 1:
            continue
        if i == 0 and j != 0:
            start_oid = ids[j - 1]
        elif i != 0 and j != 0:
            succ[ids[i - 1]] = ids[j - 1]
    seq: List[str] = []
    cur = start_oid or ids[0]
    seen: set[str] = set()
    while cur and cur not in seen:
        seen.add(cur)
        seq.append(cur)
        cur = succ.get(cur)
    for oid in ids:
        if oid not in seen:
            if strict_template_edges:
                return {
                    "status": "UNROUTABLE_SLOT",
                    "sequence": [],
                    "strict_template_edges_enabled": True,
                    "missing_template_edge_count": int(missing_template_edge_count),
                    "diagnostics": diagnostics,
                }
            seq.append(oid)
    logical_reverse_count = 0
    for pos in range(len(seq) - 1):
        left = rec_by[seq[pos]]
        right = rec_by[seq[pos + 1]]
        tpl = tpl_best.get((seq[pos], seq[pos + 1]))
        if tpl is not None:
            logical_reverse_count += int(tpl.get("logical_reverse_flag", 0) or 0)
        elif float(right.get("width", 0.0)) < float(left.get("width", 0.0)):
            logical_reverse_count += 1
    diagnostics["period_reverse_count"] = int(logical_reverse_count)
    diagnostics["period_reverse_count_violation_count"] = max(0, logical_reverse_count - int(cfg.rule.max_logical_reverse_per_campaign))
    diagnostics["reverse_count_definition"] = "logical_reverse_per_campaign"
    return {
        "status": "ROUTED",
        "sequence": seq,
        "strict_template_edges_enabled": bool(strict_template_edges),
        "missing_template_edge_count": int(missing_template_edge_count),
        "diagnostics": diagnostics,
    }
