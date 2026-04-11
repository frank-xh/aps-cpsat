from __future__ import annotations

import math
from time import perf_counter
from typing import Dict, List, Tuple

import pandas as pd
from ortools.sat.python import cp_model

from aps_cp_sat.config import PlannerConfig
from aps_cp_sat.model.diagnostics import _estimate_campaign_slots
from aps_cp_sat.model.local_router import _solve_slot_route_with_templates, _template_total_cost


def _run_unified_master_skeleton(orders_df: pd.DataFrame, cfg: PlannerConfig) -> dict:
    """Deprecated experimental helper; not part of the production master path."""
    if orders_df.empty:
        return {"status": "EMPTY", "assigned": 0, "unassigned": 0}
    return {"status": "DEPRECATED", "assigned": 0, "unassigned": int(len(orders_df))}


def _build_line_order_proxy_burden(
    orders_df: pd.DataFrame,
    tpl_df: pd.DataFrame,
    cfg: PlannerConfig,
    lines: list[str],
) -> Dict[Tuple[int, int], dict]:
    """
    Transitional burden proxy.

    The global master still does not own the full slot route. To move toward a
    more integrated joint model, we project template-derived burden signals down
    to line/order assignment and accumulate them inside the master objective.
    """
    ids = [str(v) for v in orders_df["order_id"].tolist()]
    id_to_group = {str(r["order_id"]): str(r.get("steel_group", "")) for r in orders_df.to_dict("records")}
    proxy: Dict[Tuple[int, int], dict] = {}
    if tpl_df.empty:
        default_virtual_ton10 = int(round(float(cfg.rule.virtual_tons) * 10.0 * max(1, cfg.model.max_virtual_chain)))
        default_reverse_rise = int(round(float(cfg.rule.virtual_reverse_attach_max_mm)))
        for li, _line in enumerate(lines):
            for i, _oid in enumerate(ids):
                proxy[(li, i)] = {
                    "virtual_blocks": int(cfg.model.max_virtual_chain),
                    "virtual_ton10": default_virtual_ton10,
                    "reverse_count": 1,
                    "reverse_total_rise": default_reverse_rise,
                    "bridge_cost": int(cfg.score.edge_fallback_base_penalty),
                    "out_degree": 0,
                    "in_degree": 0,
                    "degree_risk": 2,
                    "pair_gap_proxy": 1,
                    "span_risk": 10,
                }
        return proxy

    for li, line in enumerate(lines):
        line_tpl = tpl_df[tpl_df["line"] == line].copy()
        out_deg = line_tpl.groupby("from_order_id").size().to_dict() if not line_tpl.empty else {}
        in_deg = line_tpl.groupby("to_order_id").size().to_dict() if not line_tpl.empty else {}
        order_count = max(1, len(ids))
        for i, oid in enumerate(ids):
            rel = line_tpl[
                (line_tpl["from_order_id"].astype(str) == oid)
                | (line_tpl["to_order_id"].astype(str) == oid)
            ].copy()
            if rel.empty:
                proxy[(li, i)] = {
                    "virtual_blocks": int(cfg.model.max_virtual_chain),
                    "virtual_ton10": int(round(float(cfg.rule.virtual_tons) * 10.0 * max(1, cfg.model.max_virtual_chain))),
                    "reverse_count": 1,
                    "reverse_total_rise": int(round(float(cfg.rule.virtual_reverse_attach_max_mm))),
                    "bridge_cost": int(cfg.score.edge_fallback_base_penalty) * 2,
                    "out_degree": 0,
                    "in_degree": 0,
                    "degree_risk": 12,
                    "isolated_order_penalty": 12,
                    "pair_gap_proxy": max(1, order_count - 1),
                    "span_risk": 10,
                }
                continue
            rel["logical_reverse_rise"] = rel.apply(lambda r: max(0.0, -float(r.get("width_delta", 0.0))), axis=1)
            rel["cost_int"] = rel.apply(lambda r: _template_total_cost(r, cfg.score), axis=1)
            out_degree = int(out_deg.get(oid, 0))
            in_degree = int(in_deg.get(oid, 0))
            degree_risk = int(
                (8 if out_degree <= 0 else (5 if out_degree == 1 else (2 if out_degree <= 3 else 0)))
                + (8 if in_degree <= 0 else (5 if in_degree == 1 else (2 if in_degree <= 3 else 0)))
            )
            pair_gap_proxy = max(0, (order_count - 1) - min(out_degree, order_count - 1))
            width_span_proxy = int(max(0.0, float(rel["width_delta"].abs().max()) / 50.0)) if "width_delta" in rel.columns else 0
            thick_span_proxy = int(max(0.0, float(rel["thickness_delta"].max()) * 10.0)) if "thickness_delta" in rel.columns else 0
            neighbor_ids = set(rel["from_order_id"].astype(str)).union(set(rel["to_order_id"].astype(str)))
            group_mix_proxy = max(0, len({id_to_group.get(nid, "") for nid in neighbor_ids if id_to_group.get(nid, "")}) - 1)
            proxy[(li, i)] = {
                "virtual_blocks": int(rel["bridge_count"].min()),
                "virtual_ton10": int(round(float(rel["virtual_tons"].min()) * 10.0)),
                "reverse_count": int(rel["logical_reverse_flag"].min()),
                "reverse_total_rise": int(round(float(rel["logical_reverse_rise"].min()))),
                "bridge_cost": int(rel["cost_int"].min()),
                "out_degree": out_degree,
                "in_degree": in_degree,
                "degree_risk": degree_risk,
                "isolated_order_penalty": degree_risk,
                "pair_gap_proxy": pair_gap_proxy,
                "span_risk": int(width_span_proxy + thick_span_proxy + group_mix_proxy),
            }
    return proxy


def _slot_template_coverage_metrics(slot_order_ids: list[str], slot_tpl: pd.DataFrame) -> tuple[int, int, float]:
    n = len(slot_order_ids)
    possible_pairs = max(0, n * (n - 1))
    if possible_pairs == 0:
        return 0, 0, 1.0
    present_pairs = int(
        len(
            set(
                zip(
                    slot_tpl["from_order_id"].astype(str),
                    slot_tpl["to_order_id"].astype(str),
                )
            )
        )
    ) if isinstance(slot_tpl, pd.DataFrame) and not slot_tpl.empty else 0
    coverage = present_pairs / max(1, possible_pairs)
    missing = max(0, possible_pairs - present_pairs)
    return present_pairs, missing, float(coverage)


def _run_global_joint_model(
    orders_df: pd.DataFrame,
    transition_pack: dict | None,
    cfg: PlannerConfig,
    start_penalty: int = 120000,
    low_slot_penalty: int = 120000,
    ultra_low_slot_penalty: int = 300000,
    time_scale: float = 0.4,
    random_seed: int = 2027,
) -> dict:
    """Production joint master path."""
    if orders_df.empty:
        return {"status": "EMPTY", "plan_df": pd.DataFrame(), "dropped_df": pd.DataFrame()}

    tpl_df = transition_pack.get("templates") if transition_pack else None
    if not isinstance(tpl_df, pd.DataFrame) or tpl_df.empty:
        return {"status": "NO_TEMPLATE"}

    lines = ["big_roll", "small_roll"]
    n = len(orders_df)
    tons10 = [int(round(float(v) * 10.0)) for v in orders_df["tons"].tolist()]
    total_tons10 = int(sum(tons10))
    min10 = int(round(float(cfg.rule.campaign_ton_min) * 10.0))
    max10 = int(round(float(cfg.rule.campaign_ton_max) * 10.0))
    tgt10 = int(round(float(cfg.rule.campaign_ton_target) * 10.0))
    ultra_low10 = 1000
    estimated = _estimate_campaign_slots(float(orders_df["tons"].sum()), float(cfg.rule.campaign_ton_target))
    pmax = max(int(cfg.model.min_campaign_slots), min(int(cfg.model.max_campaign_slots), estimated))

    # Fixed rule semantics snapshot:
    # - line compatibility: HARD
    # - direct transition feasibility: HARD via template filtering
    # - campaign ton max: HARD
    # - campaign ton min: STRONG_SOFT
    # - unassigned real orders: STRONG_SOFT
    # - virtual slab usage / ratio: STRONG_SOFT
    # - reverse-width count / total rise: HARD semantics in template rules, plus soft burden estimate here
    proxy = _build_line_order_proxy_burden(orders_df, tpl_df, cfg, lines)

    model = cp_model.CpModel()
    u = {i: model.NewBoolVar(f"u_{i}") for i in range(n)}
    y: Dict[Tuple[int, int, int], cp_model.IntVar] = {}
    z: Dict[Tuple[int, int], cp_model.IntVar] = {}
    caps = [str(v) for v in orders_df.get("line_capability", "dual").tolist()]

    for li, line in enumerate(lines):
        for p in range(pmax):
            z[(li, p)] = model.NewBoolVar(f"z_{line}_{p}")
            if p > 0:
                model.Add(z[(li, p - 1)] >= z[(li, p)])

    penalties = []
    estimated_virtual_blocks_terms = []
    estimated_virtual_ton_terms = []
    estimated_reverse_count_terms = []
    estimated_reverse_rise_terms = []
    estimated_bridge_cost_terms = []
    estimated_isolation_risk_terms = []
    estimated_pair_gap_risk_terms = []
    estimated_span_risk_terms = []

    for i in range(n):
        terms = [u[i]]
        for li, line in enumerate(lines):
            allowed = (
                (caps[i] == "dual")
                or (caps[i] == "either")
                or (caps[i] == "big_only" and line == "big_roll")
                or (caps[i] == "small_only" and line == "small_roll")
            )
            for p in range(pmax):
                v = model.NewBoolVar(f"y_{i}_{li}_{p}")
                y[(i, li, p)] = v
                if not allowed:
                    model.Add(v == 0)
                model.Add(v <= z[(li, p)])
                burden = proxy.get((li, i), {"virtual_blocks": 0, "virtual_ton10": 0, "reverse_count": 0, "reverse_total_rise": 0, "bridge_cost": 0})
                estimated_virtual_blocks_terms.append(int(burden["virtual_blocks"]) * v)
                estimated_virtual_ton_terms.append(int(burden["virtual_ton10"]) * v)
                estimated_reverse_count_terms.append(int(burden["reverse_count"]) * v)
                estimated_reverse_rise_terms.append(int(burden["reverse_total_rise"]) * v)
                estimated_bridge_cost_terms.append(int(burden["bridge_cost"]) * v)
                estimated_isolation_risk_terms.append(int(burden.get("isolated_order_penalty", burden.get("degree_risk", 0))) * v)
                estimated_pair_gap_risk_terms.append(int(burden.get("pair_gap_proxy", 0)) * v)
                estimated_span_risk_terms.append(int(burden.get("span_risk", 0)) * v)
                terms.append(v)
        model.Add(sum(terms) == 1)

    min_ratio = max(0.0, min(1.0, float(cfg.model.min_real_schedule_ratio)))
    max_unassigned = int(math.floor((1.0 - min_ratio) * n))
    model.Add(sum(u[i] for i in range(n)) <= max_unassigned)

    penalties.extend(int(cfg.score.unassigned_real) * u[i] for i in range(n))

    for li, line in enumerate(lines):
        for p in range(pmax):
            load = model.NewIntVar(0, total_tons10 + 50000, f"load_{line}_{p}")
            model.Add(load == sum(tons10[i] * y[(i, li, p)] for i in range(n)))
            model.Add(load <= max10 * z[(li, p)])

            # campaign ton min is intentionally modeled as strong-soft, not hard.
            under = model.NewIntVar(0, max10, f"under_{line}_{p}")
            model.Add(under >= min10 * z[(li, p)] - load)
            model.Add(under >= 0)
            penalties.append(int(cfg.score.ton_under) * under)

            dev = model.NewIntVar(0, max(total_tons10 + tgt10, 1), f"dev_{line}_{p}")
            tmp = model.NewIntVar(-max(total_tons10 + tgt10, 1), max(total_tons10 + tgt10, 1), f"tmp_{line}_{p}")
            model.Add(tmp == load - tgt10 * z[(li, p)])
            model.AddAbsEquality(dev, tmp)
            penalties.append(int(cfg.score.ton_target) * dev)

            is_low = model.NewBoolVar(f"low_{line}_{p}")
            model.Add(load <= min10 - 1).OnlyEnforceIf(is_low)
            model.Add(load >= min10).OnlyEnforceIf(is_low.Not())
            model.Add(is_low <= z[(li, p)])
            penalties.append(int(low_slot_penalty) * is_low)

            is_ultra = model.NewBoolVar(f"ultra_{line}_{p}")
            model.Add(load <= ultra_low10 - 1).OnlyEnforceIf(is_ultra)
            model.Add(load >= ultra_low10).OnlyEnforceIf(is_ultra.Not())
            model.Add(is_ultra <= z[(li, p)])
            penalties.append(int(ultra_low_slot_penalty) * is_ultra)

            penalties.append(int(start_penalty) * z[(li, p)])

            order_cnt = model.NewIntVar(0, n, f"order_cnt_{line}_{p}")
            model.Add(order_cnt == sum(y[(i, li, p)] for i in range(n)))
            soft_cap = int(cfg.model.big_roll_slot_soft_order_cap if line == "big_roll" else cfg.model.small_roll_slot_soft_order_cap)
            hard_cap = int(cfg.model.big_roll_slot_hard_order_cap if line == "big_roll" else cfg.model.small_roll_slot_hard_order_cap)
            if hard_cap > 0:
                model.Add(order_cnt <= hard_cap * z[(li, p)])
            over_cap = model.NewIntVar(0, n, f"order_over_{line}_{p}")
            model.Add(over_cap >= order_cnt - soft_cap * z[(li, p)])
            model.Add(over_cap >= 0)
            penalties.append(int(cfg.score.slot_order_count_penalty) * over_cap)

    est_virtual_blocks = model.NewIntVar(0, max(1, n * int(cfg.model.max_virtual_chain)), "est_virtual_blocks")
    model.Add(est_virtual_blocks == sum(estimated_virtual_blocks_terms))
    penalties.append(int(cfg.score.master_virtual_blocks) * est_virtual_blocks)

    est_virtual_ton10 = model.NewIntVar(
        0,
        max(1, total_tons10 + int(round(float(cfg.rule.virtual_tons) * 10.0 * n * max(1, cfg.model.max_virtual_chain)))),
        "est_virtual_ton10",
    )
    model.Add(est_virtual_ton10 == sum(estimated_virtual_ton_terms))
    penalties.append(int(cfg.score.master_virtual_tons) * est_virtual_ton10)

    est_reverse_count = model.NewIntVar(0, max(1, n), "est_reverse_count")
    model.Add(est_reverse_count == sum(estimated_reverse_count_terms))
    penalties.append(int(cfg.score.master_reverse_count) * est_reverse_count)

    est_reverse_rise = model.NewIntVar(0, max(1, int(round(float(cfg.rule.virtual_reverse_attach_max_mm) * n))), "est_reverse_rise")
    model.Add(est_reverse_rise == sum(estimated_reverse_rise_terms))
    penalties.append(int(cfg.score.master_reverse_total_rise) * est_reverse_rise)

    est_bridge_cost = model.NewIntVar(0, max(1, n * int(cfg.score.edge_fallback_base_penalty) * 2), "est_bridge_cost")
    model.Add(est_bridge_cost == sum(estimated_bridge_cost_terms))
    penalties.append(int(cfg.score.master_route_risk) * est_bridge_cost)

    est_isolation_risk = model.NewIntVar(0, max(1, n * 4), "est_isolation_risk")
    model.Add(est_isolation_risk == sum(estimated_isolation_risk_terms))
    penalties.append(int(cfg.score.slot_isolation_risk_penalty) * est_isolation_risk)

    est_pair_gap_risk = model.NewIntVar(0, max(1, n * n), "est_pair_gap_risk")
    model.Add(est_pair_gap_risk == sum(estimated_pair_gap_risk_terms))
    penalties.append(int(cfg.score.slot_pair_gap_risk_penalty) * est_pair_gap_risk)

    est_span_risk = model.NewIntVar(0, max(1, n * 50), "est_span_risk")
    model.Add(est_span_risk == sum(estimated_span_risk_terms))
    penalties.append(int(cfg.score.slot_span_risk_penalty) * est_span_risk)

    est_real_assigned_ton10 = model.NewIntVar(0, max(1, total_tons10), "est_real_assigned_ton10")
    model.Add(est_real_assigned_ton10 == sum(tons10[i] * (1 - u[i]) for i in range(n)))
    est_global_virtual_ratio_over = model.NewIntVar(
        0,
        max(1, int(round(float(cfg.rule.virtual_tons) * 10.0 * n * max(1, cfg.model.max_virtual_chain)))),
        "est_global_virtual_ratio_over",
    )
    model.Add(est_global_virtual_ratio_over >= int(cfg.rule.virtual_ton_ratio_den) * est_virtual_ton10 - int(cfg.rule.virtual_ton_ratio_num) * est_real_assigned_ton10)
    model.Add(est_global_virtual_ratio_over >= 0)
    penalties.append(int(cfg.score.master_virtual_ratio) * est_global_virtual_ratio_over)

    model.Minimize(sum(penalties))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max(
        float(cfg.model.min_master_solve_seconds),
        min(float(cfg.model.max_master_solve_seconds), float(cfg.model.time_limit_seconds) * float(time_scale)),
    )
    solver.parameters.num_search_workers = max(1, int(cfg.model.master_num_workers))
    solver.parameters.random_seed = int(random_seed)
    status = solver.Solve(model)

    if status == cp_model.UNKNOWN:
        return {"status": "TIMEOUT_NO_FEASIBLE"}
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return {"status": "INFEASIBLE"}

    assigned_rows = []
    dropped_rows = []
    for i in range(n):
        if solver.Value(u[i]) == 1:
            drop = dict(orders_df.iloc[i])
            drop["drop_reason"] = "MASTER_UNASSIGNED"
            dropped_rows.append(drop)
            continue
        chosen_line = ""
        chosen_slot = -1
        for li, line in enumerate(lines):
            for p in range(pmax):
                if solver.Value(y[(i, li, p)]) == 1:
                    chosen_line = line
                    chosen_slot = p + 1
                    break
            if chosen_line:
                break
        row = dict(orders_df.iloc[i])
        row["line"] = chosen_line
        row["master_slot"] = int(chosen_slot)
        row["_row_idx"] = int(i)
        assigned_rows.append(row)

    assigned_df = pd.DataFrame(assigned_rows)
    if assigned_df.empty:
        return {
            "status": "FEASIBLE",
            "plan_df": pd.DataFrame(),
            "dropped_df": pd.DataFrame(dropped_rows),
            "objective": float(solver.ObjectiveValue()),
            "assigned_count": 0,
            "unassigned_count": int(len(dropped_rows)),
            "total_virtual_blocks": 0,
            "total_virtual_ton10": 0,
            "global_ratio_over": 0,
            "low_slot_count": 0,
            "ultra_low_slot_count": 0,
            "used_local_routing": False,
            "local_routing_role": "not_used",
            "estimated_virtual_blocks": int(solver.Value(est_virtual_blocks)),
            "estimated_virtual_ton10": int(solver.Value(est_virtual_ton10)),
            "estimated_global_ratio_over": int(solver.Value(est_global_virtual_ratio_over)),
            "estimated_reverse_count": int(solver.Value(est_reverse_count)),
            "estimated_reverse_rise": int(solver.Value(est_reverse_rise)),
            "estimated_bridge_cost": int(solver.Value(est_bridge_cost)),
            "big_roll_max_slot_order_count": 0,
            "small_roll_max_slot_order_count": 0,
            "local_router_seconds": 0.0,
        }

    used_local_routing = False
    total_virtual_blocks = 0
    total_virtual_ton10 = 0
    low_slot_count = 0
    ultra_low_slot_count = 0
    logical_reverse_count = 0
    logical_reverse_total_rise = 0
    selected_direct_edge_count = 0
    selected_real_bridge_edge_count = 0
    selected_virtual_bridge_edge_count = 0
    unroutable_slot_count = 0
    strict_template_edges_enabled = bool(cfg.model.strict_template_edges)
    slot_route_risk_score = 0
    slot_route_details: List[dict] = []
    plan_rows: List[dict] = []
    candidate_plan_rows: List[dict] = []
    max_slot_order_count = 0
    big_roll_max_slot_order_count = 0
    small_roll_max_slot_order_count = 0
    big_roll_order_cap_violations = 0
    hard_cap_not_enforced = False
    total_slot_order_count = 0
    slot_count = 0
    local_router_seconds = 0.0

    for (line, slot), slot_df in assigned_df.groupby(["line", "master_slot"], sort=True):
        slot_count += 1
        slot_order_count = int(len(slot_df))
        total_slot_order_count += slot_order_count
        max_slot_order_count = max(max_slot_order_count, slot_order_count)
        if str(line) == "big_roll":
            big_roll_max_slot_order_count = max(big_roll_max_slot_order_count, slot_order_count)
            if int(cfg.model.big_roll_slot_hard_order_cap) > 0 and slot_order_count > int(cfg.model.big_roll_slot_hard_order_cap):
                big_roll_order_cap_violations += 1
                hard_cap_not_enforced = True
        elif str(line) == "small_roll":
            small_roll_max_slot_order_count = max(small_roll_max_slot_order_count, slot_order_count)
        slot_tons10 = int(slot_df["tons"].astype(float).mul(10.0).round().sum())
        if slot_tons10 < min10:
            low_slot_count += 1
        if slot_tons10 < ultra_low10:
            ultra_low_slot_count += 1

        slot_tpl = tpl_df[
            (tpl_df["line"] == str(line))
            & (tpl_df["from_order_id"].astype(str).isin(slot_df["order_id"].astype(str)))
            & (tpl_df["to_order_id"].astype(str).isin(slot_df["order_id"].astype(str)))
        ].copy()
        slot_order_ids = [str(v) for v in slot_df["order_id"].tolist()]
        present_pairs, missing_pairs, coverage_ratio = _slot_template_coverage_metrics(slot_order_ids, slot_tpl)
        slot_est_virtual_blocks = 0
        slot_est_reverse_count = 0
        slot_degree_risk = 0
        slot_isolated_penalty = 0
        slot_pair_gap_proxy = 0
        slot_span_risk = 0
        for rec in slot_df.to_dict("records"):
            burden = proxy.get((lines.index(str(line)), int(rec["_row_idx"])), {})
            slot_est_virtual_blocks += int(burden.get("virtual_blocks", 0))
            slot_est_reverse_count += int(burden.get("reverse_count", 0))
            slot_degree_risk += int(burden.get("degree_risk", 0))
            slot_isolated_penalty += int(burden.get("isolated_order_penalty", burden.get("degree_risk", 0)))
            slot_pair_gap_proxy += int(burden.get("pair_gap_proxy", 0))
            slot_span_risk += int(burden.get("span_risk", 0))
        width_span = float(slot_df["width"].max() - slot_df["width"].min()) if "width" in slot_df.columns and len(slot_df) > 0 else 0.0
        thickness_span = float(slot_df["thickness"].max() - slot_df["thickness"].min()) if "thickness" in slot_df.columns and len(slot_df) > 0 else 0.0
        steel_group_count = int(slot_df["steel_group"].fillna("").astype(str).nunique()) if "steel_group" in slot_df.columns else 0
        slot_route_risk = int(missing_pairs + slot_degree_risk + slot_pair_gap_proxy + slot_span_risk)
        slot_route_risk_score += slot_route_risk
        line_slot_order_cap = int(cfg.model.big_roll_slot_soft_order_cap if str(line) == "big_roll" else cfg.model.small_roll_slot_soft_order_cap)
        order_count_over_cap = max(0, slot_order_count - line_slot_order_cap)
        route_t0 = perf_counter()
        route_result = (
            {
                "status": "ROUTED",
                "sequence": slot_order_ids,
                "strict_template_edges_enabled": strict_template_edges_enabled,
                "missing_template_edge_count": 0,
                "diagnostics": {},
            }
            if len(slot_df) == 1
            else _solve_slot_route_with_templates(
                slot_df[["order_id", "due_rank", "width", "thickness", "steel_group", "line"]],
                slot_tpl,
                time_limit=max(1.0, min(8.0, float(cfg.model.time_limit_seconds) * 0.1)),
                seed=int(random_seed + slot),
                cfg=cfg,
            )
        )
        slot_zero_degree_penalty = int(route_result.get("diagnostics", {}).get("zero_in_orders", 0) or 0) * 4 + int(route_result.get("diagnostics", {}).get("zero_out_orders", 0) or 0) * 4
        local_router_seconds += perf_counter() - route_t0
        slot_detail = {
            "line": str(line),
            "slot": int(slot),
            "slot_no": int(slot),
            "order_count": int(len(slot_order_ids)),
            "line_slot_order_cap": int(line_slot_order_cap),
            "order_count_over_cap": int(order_count_over_cap),
            "template_pairs_present": int(present_pairs),
            "template_pairs_missing": int(missing_pairs),
            "template_coverage_ratio": round(float(coverage_ratio), 4),
            "estimated_virtual_blocks": int(slot_est_virtual_blocks),
            "estimated_reverse_count": int(slot_est_reverse_count),
            "reverse_count_definition": "logical_reverse_per_campaign",
            "isolated_order_penalty": int(slot_isolated_penalty + slot_zero_degree_penalty),
            "degree_risk": int(slot_degree_risk),
            "pair_gap_proxy": int(slot_pair_gap_proxy),
            "span_risk": int(slot_span_risk),
            "width_span": round(width_span, 3),
            "thickness_span": round(thickness_span, 3),
            "steel_group_count": int(steel_group_count),
            "slot_route_risk_score": int(slot_route_risk),
            "status": str(route_result.get("status", "UNKNOWN")),
            **dict(route_result.get("diagnostics", {}) or {}),
        }
        if (
            int(slot_detail.get("zero_in_orders", 0) or 0) + int(slot_detail.get("zero_out_orders", 0) or 0) > 0
            and int(slot_detail.get("degree_risk", 0) or 0) <= 0
        ):
            slot_detail["isolation_risk_not_effective"] = True
        else:
            slot_detail["isolation_risk_not_effective"] = False
        slot_route_details.append(slot_detail)
        if len(slot_df) > 1:
            used_local_routing = True
        if str(route_result.get("status")) != "ROUTED":
            unroutable_slot_count += 1
            candidate_ordered = slot_df.copy().sort_values(
                ["due_rank", "priority", "_row_idx"],
                ascending=[True, False, True],
                kind="mergesort",
            ).reset_index(drop=True)
            for idx, rec in enumerate(candidate_ordered.to_dict("records"), start=1):
                candidate_plan_rows.append(
                    {
                        "row_idx": int(rec["_row_idx"]),
                        "order_id": str(rec["order_id"]),
                        "assigned_line": str(line),
                        "assigned_slot": int(slot),
                        "candidate_position": int(idx),
                        "candidate_slot_member_index": int(idx),
                        "selected_template_id": "",
                        "selected_edge_type": "",
                        "selected_bridge_path": "",
                        "force_break_before": int(1 if idx == 1 else 0),
                        "slot_unroutable_flag": 1,
                        "slot_route_risk_score": int(slot_route_risk),
                        "candidate_status": "UNROUTABLE_SLOT_MEMBER",
                    }
                )
            continue
        seq = [str(v) for v in route_result.get("sequence", [])]
        seq_map = {oid: idx + 1 for idx, oid in enumerate(seq)}

        best_tpl_by_pair: Dict[Tuple[str, str], dict] = {}
        for r in slot_tpl.to_dict("records"):
            key = (str(r.get("from_order_id", "")), str(r.get("to_order_id", "")))
            cand = {
                "template_id": str(r.get("template_id", "")),
                "edge_type": str(r.get("edge_type", "DIRECT_EDGE") or "DIRECT_EDGE"),
                "bridge_path": str(r.get("bridge_path", "")),
                "bridge_count": int(r.get("bridge_count", 0) or 0),
                "virtual_ton10": int(round(float(r.get("virtual_tons", 0.0)) * 10.0)),
                "cost": _template_total_cost(r, cfg.score),
                "logical_reverse_flag": int(r.get("logical_reverse_flag", 0) or 0),
                "logical_reverse_rise": int(round(max(0.0, -float(r.get("width_delta", 0.0))))),
            }
            prev = best_tpl_by_pair.get(key)
            if prev is None or cand["cost"] < prev["cost"]:
                best_tpl_by_pair[key] = cand

        ordered = slot_df.copy()
        ordered["master_seq"] = ordered["order_id"].map(lambda oid: seq_map.get(str(oid), 10**6))
        ordered = ordered.sort_values(["master_seq", "width", "thickness"], kind="mergesort").reset_index(drop=True)
        campaign_id = int(slot)
        for idx, rec in enumerate(ordered.to_dict("records"), start=1):
            prev_oid = str(ordered.iloc[idx - 2]["order_id"]) if idx > 1 else ""
            cur_oid = str(rec["order_id"])
            tpl = best_tpl_by_pair.get((prev_oid, cur_oid)) if idx > 1 else None
            if tpl is not None:
                edge_type = str(tpl.get("edge_type", "DIRECT_EDGE"))
                if edge_type == "REAL_BRIDGE_EDGE":
                    selected_real_bridge_edge_count += 1
                elif edge_type == "VIRTUAL_BRIDGE_EDGE":
                    selected_virtual_bridge_edge_count += 1
                else:
                    selected_direct_edge_count += 1
                total_virtual_blocks += int(tpl["bridge_count"])
                total_virtual_ton10 += int(tpl["virtual_ton10"])
                logical_reverse_count += int(tpl["logical_reverse_flag"])
                logical_reverse_total_rise += int(tpl["logical_reverse_rise"])
            plan_rows.append(
                {
                    "row_idx": int(rec["_row_idx"]),
                    "order_id": cur_oid,
                    "assigned_line": str(line),
                    "assigned_slot": int(slot),
                    "master_seq": int(idx),
                    "campaign_id_hint": int(campaign_id),
                    "campaign_seq_hint": int(idx),
                    "selected_template_id": str(tpl.get("template_id", "") if tpl else ""),
                    "selected_edge_type": str(tpl.get("edge_type", "DIRECT_EDGE") if tpl else "DIRECT_EDGE"),
                    "selected_bridge_path": str(tpl.get("bridge_path", "") if tpl else ""),
                    "force_break_before": int(1 if idx == 1 else 0),
                    "is_unassigned": 0,
                }
            )
            candidate_plan_rows.append(
                {
                    "row_idx": int(rec["_row_idx"]),
                    "order_id": cur_oid,
                    "assigned_line": str(line),
                    "assigned_slot": int(slot),
                    "candidate_position": int(idx),
                    "candidate_slot_member_index": int(idx),
                    "selected_template_id": str(tpl.get("template_id", "") if tpl else ""),
                    "selected_edge_type": str(tpl.get("edge_type", "DIRECT_EDGE") if tpl else "DIRECT_EDGE"),
                    "selected_bridge_path": str(tpl.get("bridge_path", "") if tpl else ""),
                    "force_break_before": int(1 if idx == 1 else 0),
                    "slot_unroutable_flag": 0,
                    "slot_route_risk_score": int(slot_route_risk),
                    "candidate_status": "ASSIGNED_CANDIDATE",
                }
            )

    assigned_real_ton10 = int(sum(tons10[i] for i in range(n) if solver.Value(u[i]) == 0))
    global_ratio_over = max(0, int(total_virtual_ton10) * int(cfg.rule.virtual_ton_ratio_den) - assigned_real_ton10 * int(cfg.rule.virtual_ton_ratio_num))

    if unroutable_slot_count > 0:
        candidate_plan_df = pd.DataFrame(candidate_plan_rows)
        if not candidate_plan_df.empty:
            candidate_plan_df = candidate_plan_df.sort_values(
                ["assigned_line", "assigned_slot", "candidate_position"],
                kind="mergesort",
            ).reset_index(drop=True)
        return {
            "status": "ROUTING_INFEASIBLE",
            "plan_df": pd.DataFrame(),
            "candidate_plan_df": candidate_plan_df,
            "dropped_df": pd.DataFrame(dropped_rows),
            "objective": float(solver.ObjectiveValue()),
            "assigned_count": 0,
            "unassigned_count": int(len(dropped_rows)),
            "total_virtual_blocks": 0,
            "total_virtual_ton10": 0,
            "global_ratio_over": 0,
            "logical_reverse_count": 0,
            "logical_reverse_total_rise": 0,
            "low_slot_count": int(low_slot_count),
            "ultra_low_slot_count": int(ultra_low_slot_count),
            "used_local_routing": bool(used_local_routing),
            "local_routing_role": "transitional_slot_router" if bool(used_local_routing) else "not_used",
            "strict_template_edges_enabled": bool(strict_template_edges_enabled),
            "unroutable_slot_count": int(unroutable_slot_count),
            "slot_route_risk_score": int(slot_route_risk_score),
            "slot_route_details": slot_route_details,
            "estimated_virtual_blocks": int(solver.Value(est_virtual_blocks)),
            "estimated_virtual_ton10": int(solver.Value(est_virtual_ton10)),
            "estimated_global_ratio_over": int(solver.Value(est_global_virtual_ratio_over)),
            "estimated_reverse_count": int(solver.Value(est_reverse_count)),
            "estimated_reverse_rise": int(solver.Value(est_reverse_rise)),
            "estimated_bridge_cost": int(solver.Value(est_bridge_cost)),
            "estimated_isolation_risk": int(solver.Value(est_isolation_risk)),
            "estimated_pair_gap_risk": int(solver.Value(est_pair_gap_risk)),
            "estimated_span_risk": int(solver.Value(est_span_risk)),
            "selected_direct_edge_count": int(selected_direct_edge_count),
            "selected_real_bridge_edge_count": int(selected_real_bridge_edge_count),
            "selected_virtual_bridge_edge_count": int(selected_virtual_bridge_edge_count),
            "max_slot_order_count": int(max_slot_order_count),
            "big_roll_max_slot_order_count": int(big_roll_max_slot_order_count),
            "small_roll_max_slot_order_count": int(small_roll_max_slot_order_count),
            "big_roll_slot_order_hard_cap": int(cfg.model.big_roll_slot_hard_order_cap),
            "big_roll_order_cap_violations": int(big_roll_order_cap_violations),
            "hard_cap_not_enforced": bool(hard_cap_not_enforced),
            "avg_slot_order_count": round(float(total_slot_order_count / max(1, slot_count)), 2),
            "local_router_seconds": round(float(local_router_seconds), 6),
        }

    plan_df = pd.DataFrame(plan_rows)
    if not plan_df.empty:
        plan_df = plan_df.sort_values(["assigned_line", "assigned_slot", "master_seq"], kind="mergesort").reset_index(drop=True)
    candidate_plan_df = pd.DataFrame(candidate_plan_rows)
    if not candidate_plan_df.empty:
        candidate_plan_df = candidate_plan_df.sort_values(
            ["assigned_line", "assigned_slot", "candidate_position"],
            kind="mergesort",
        ).reset_index(drop=True)

    return {
        "status": "FEASIBLE",
        "plan_df": plan_df,
        "candidate_plan_df": candidate_plan_df,
        "dropped_df": pd.DataFrame(dropped_rows),
        "objective": float(solver.ObjectiveValue()),
        "assigned_count": int(len(plan_rows)),
        "unassigned_count": int(len(dropped_rows)),
        "total_virtual_blocks": int(total_virtual_blocks),
        "total_virtual_ton10": int(total_virtual_ton10),
        "global_ratio_over": int(global_ratio_over),
        "logical_reverse_count": int(logical_reverse_count),
        "logical_reverse_total_rise": int(logical_reverse_total_rise),
        "low_slot_count": int(low_slot_count),
        "ultra_low_slot_count": int(ultra_low_slot_count),
        "used_local_routing": bool(used_local_routing),
        "local_routing_role": "transitional_slot_router" if bool(used_local_routing) else "not_used",
        "strict_template_edges_enabled": bool(strict_template_edges_enabled),
        "unroutable_slot_count": int(unroutable_slot_count),
        "slot_route_risk_score": int(slot_route_risk_score),
        "slot_route_details": slot_route_details,
        "estimated_virtual_blocks": int(solver.Value(est_virtual_blocks)),
        "estimated_virtual_ton10": int(solver.Value(est_virtual_ton10)),
        "estimated_global_ratio_over": int(solver.Value(est_global_virtual_ratio_over)),
        "estimated_reverse_count": int(solver.Value(est_reverse_count)),
        "estimated_reverse_rise": int(solver.Value(est_reverse_rise)),
        "estimated_bridge_cost": int(solver.Value(est_bridge_cost)),
        "estimated_isolation_risk": int(solver.Value(est_isolation_risk)),
        "estimated_pair_gap_risk": int(solver.Value(est_pair_gap_risk)),
        "estimated_span_risk": int(solver.Value(est_span_risk)),
        "selected_direct_edge_count": int(selected_direct_edge_count),
        "selected_real_bridge_edge_count": int(selected_real_bridge_edge_count),
        "selected_virtual_bridge_edge_count": int(selected_virtual_bridge_edge_count),
        "max_slot_order_count": int(max_slot_order_count),
        "big_roll_max_slot_order_count": int(big_roll_max_slot_order_count),
        "small_roll_max_slot_order_count": int(small_roll_max_slot_order_count),
        "big_roll_slot_order_hard_cap": int(cfg.model.big_roll_slot_hard_order_cap),
        "big_roll_order_cap_violations": int(big_roll_order_cap_violations),
        "hard_cap_not_enforced": bool(hard_cap_not_enforced),
        "avg_slot_order_count": round(float(total_slot_order_count / max(1, slot_count)), 2),
        "local_router_seconds": round(float(local_router_seconds), 6),
    }
