from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

import pandas as pd

from aps_cp_sat.config import PlannerConfig


def _allowed_lines(line_capability: str) -> list[str]:
    cap = str(line_capability or "dual")
    if cap in {"big_only", "large", "LARGE"}:
        return ["big_roll"]
    if cap in {"small_only", "small", "SMALL"}:
        return ["small_roll"]
    return ["big_roll", "small_roll"]


def _weak_components(nodes: Iterable[str], edges: Iterable[Tuple[str, str]]) -> tuple[int, int, float]:
    node_list = list(dict.fromkeys(str(n) for n in nodes))
    if not node_list:
        return 0, 0, 0.0
    parent = {n: n for n in node_list}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b in edges:
        if a in parent and b in parent:
            union(a, b)
    comp_sizes: Dict[str, int] = defaultdict(int)
    for n in node_list:
        comp_sizes[find(n)] += 1
    largest = max(comp_sizes.values()) if comp_sizes else 0
    isolated = sum(1 for v in comp_sizes.values() if v == 1)
    return len(comp_sizes), isolated, float(largest / max(1, len(node_list)))


def build_feasibility_evidence(
    orders_df: pd.DataFrame,
    transition_pack: dict | None,
    cfg: PlannerConfig,
) -> dict:
    """
    Lightweight evidence layer.

    It does not relax any rule. It only surfaces whether the current data and
    current hard template semantics already show strong infeasibility signals.
    """
    tpl_df = transition_pack.get("templates") if isinstance(transition_pack, dict) else None
    if not isinstance(tpl_df, pd.DataFrame):
        tpl_df = pd.DataFrame()
    summaries = transition_pack.get("summaries", []) if isinstance(transition_pack, dict) else []

    order_ids = [str(v) for v in orders_df.get("order_id", pd.Series(dtype=str)).tolist()]
    line_caps = {
        str(r["order_id"]): _allowed_lines(str(r.get("line_capability", "dual")))
        for r in orders_df.to_dict("records")
    }
    tons_by = {
        str(r["order_id"]): float(r.get("tons", 0.0) or 0.0)
        for r in orders_df.to_dict("records")
    }

    line_evidence: Dict[str, dict] = {}
    pred_succ_by_line: Dict[str, dict] = {}
    total_zero_nodes = 0
    for line in ["big_roll", "small_roll"]:
        line_tpl = tpl_df[tpl_df.get("line", pd.Series(dtype=str)) == line].copy() if not tpl_df.empty else pd.DataFrame()
        line_nodes = set(line_tpl["from_order_id"].astype(str)).union(set(line_tpl["to_order_id"].astype(str))) if not line_tpl.empty else set()
        out_deg = line_tpl.groupby("from_order_id").size().to_dict() if not line_tpl.empty else {}
        in_deg = line_tpl.groupby("to_order_id").size().to_dict() if not line_tpl.empty else {}
        nodes_for_line = [oid for oid, allowed in line_caps.items() if line in allowed]
        component_count, isolated_node_count, largest_component_ratio = _weak_components(
            nodes_for_line,
            zip(
                line_tpl["from_order_id"].astype(str).tolist(),
                line_tpl["to_order_id"].astype(str).tolist(),
            ) if not line_tpl.empty else [],
        )
        summary = next((s for s in summaries if getattr(s, "line", "") == line), None)
        coverage_ratio = 0.0
        if summary is not None:
            coverage_ratio = float(int(summary.feasible_pairs) / max(1, int(summary.candidate_pairs)))
        zero_nodes = [oid for oid in nodes_for_line if int(out_deg.get(oid, 0)) <= 0 or int(in_deg.get(oid, 0)) <= 0]
        total_zero_nodes += len(zero_nodes)
        line_evidence[line] = {
            "component_count": int(component_count),
            "largest_component_ratio": round(float(largest_component_ratio), 4),
            "isolated_node_count": int(isolated_node_count),
            "template_coverage_ratio": round(float(coverage_ratio), 4),
            "zero_degree_node_count": int(len(zero_nodes)),
            "node_count": int(len(nodes_for_line)),
        }
        pred_succ_by_line[line] = {
            "in": {oid: int(in_deg.get(oid, 0)) for oid in nodes_for_line},
            "out": {oid: int(out_deg.get(oid, 0)) for oid in nodes_for_line},
        }

    isolated_orders = []
    for oid in order_ids:
        allowed = line_caps.get(oid, ["big_roll", "small_roll"])
        preds = [pred_succ_by_line[line]["in"].get(oid, 0) for line in allowed]
        succs = [pred_succ_by_line[line]["out"].get(oid, 0) for line in allowed]
        isolated_globally = all(p <= 0 or s <= 0 for p, s in zip(preds, succs))
        if isolated_globally:
            isolated_orders.append(
                {
                    "order_id": oid,
                    "allowed_lines": ",".join(allowed),
                    "pred_counts": preds,
                    "succ_counts": succs,
                    "tons": tons_by.get(oid, 0.0),
                    "isolated_globally": True,
                }
            )

    total_tons = float(orders_df["tons"].sum()) if "tons" in orders_df.columns and not orders_df.empty else 0.0
    ton_max = float(cfg.rule.campaign_ton_max)
    total_slot_lb = int(math.ceil(total_tons / max(1.0, ton_max)))
    line_cap_series = orders_df["line_capability"] if "line_capability" in orders_df.columns else pd.Series(["dual"] * len(orders_df))
    big_only_tons = float(orders_df.loc[line_cap_series == "big_only", "tons"].sum()) if "tons" in orders_df.columns else 0.0
    small_only_tons = float(orders_df.loc[line_cap_series == "small_only", "tons"].sum()) if "tons" in orders_df.columns else 0.0
    line_capacity_lb = {
        "big_roll": int(math.ceil(big_only_tons / max(1.0, ton_max))),
        "small_roll": int(math.ceil(small_only_tons / max(1.0, ton_max))),
    }

    big_slot_order_cap = int(cfg.model.big_roll_slot_hard_order_cap or cfg.model.big_roll_slot_soft_order_cap or 22)
    small_slot_order_cap = int(cfg.model.small_roll_slot_hard_order_cap or cfg.model.small_roll_slot_soft_order_cap or 22)
    slot_safe_lb = {
        "big_roll": int(math.ceil(max(0, int((line_cap_series == "big_only").sum() + (line_cap_series.isin(["dual", "either"]).sum() * 0.5))) / max(1, big_slot_order_cap))),
        "small_roll": int(math.ceil(max(0, int((line_cap_series == "small_only").sum() + (line_cap_series.isin(["dual", "either"]).sum() * 0.5))) / max(1, small_slot_order_cap))),
    }

    top_signals: List[str] = []
    evidence_level = "OK"
    isolated_ratio = float(len(isolated_orders) / max(1, len(order_ids)))
    largest_component_min = min(
        [line_evidence[line]["largest_component_ratio"] for line in line_evidence if line_evidence[line]["node_count"] > 0] or [1.0]
    )
    max_slots = int(cfg.model.max_campaign_slots)
    if isolated_ratio >= 0.08:
        top_signals.append(f"global_isolated_orders={len(isolated_orders)}")
    if largest_component_min < 0.45:
        top_signals.append(f"largest_component_ratio_min={largest_component_min:.3f}")
    if total_slot_lb > max_slots:
        top_signals.append(f"capacity_slot_lower_bound_exceeds_cap={total_slot_lb}>{max_slots}")
    if slot_safe_lb["big_roll"] + slot_safe_lb["small_roll"] > max_slots:
        top_signals.append(
            f"slot_safe_lower_bound_exceeds_cap={slot_safe_lb['big_roll'] + slot_safe_lb['small_roll']}>{max_slots}"
        )

    if len(top_signals) >= 2 or isolated_ratio >= 0.15:
        evidence_level = "STRONG_INFEASIBLE_SIGNAL"
    elif top_signals or total_zero_nodes > 0:
        evidence_level = "WARNING"

    return {
        "evidence_level": evidence_level,
        "top_infeasibility_signals": top_signals[:5],
        "isolated_order_count": int(len(isolated_orders)),
        "globally_isolated_orders": int(len(isolated_orders)),
        "isolated_orders_topn": isolated_orders[:10],
        "line_evidence": line_evidence,
        "capacity_lower_bound": {
            "total_slot_lower_bound": int(total_slot_lb),
            "line_slot_lower_bound": line_capacity_lb,
            "theoretical_min_slots_by_line": line_capacity_lb,
            "max_campaign_slots": int(max_slots),
        },
        "slot_safe_lower_bound": {
            "big_roll": int(slot_safe_lb["big_roll"]),
            "small_roll": int(slot_safe_lb["small_roll"]),
            "big_roll_order_cap": int(big_slot_order_cap),
            "small_roll_order_cap": int(small_slot_order_cap),
            "current_slot_cap": {
                "big_roll": int(big_slot_order_cap),
                "small_roll": int(small_slot_order_cap),
            },
        },
        "feasibility_evidence_summary": {
            "slot_safe_lower_bound": int(slot_safe_lb["big_roll"] + slot_safe_lb["small_roll"]),
            "current_slot_cap": int(max_slots),
            "globally_isolated_orders": int(len(isolated_orders)),
            "isolated_ratio": round(isolated_ratio, 4),
            "largest_component_ratio_min": round(float(largest_component_min), 4),
            "total_zero_degree_nodes": int(total_zero_nodes),
            "per_line_component_count": {line: int(line_evidence[line]["component_count"]) for line in line_evidence},
            "isolated_node_count": {line: int(line_evidence[line]["isolated_node_count"]) for line in line_evidence},
            "theoretical_min_slots_by_line": line_capacity_lb,
        },
    }
