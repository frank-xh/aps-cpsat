from __future__ import annotations

import math
from typing import Dict

import pandas as pd

from aps_cp_sat.config import PlannerConfig


def _estimate_campaign_slots(total_tons: float, target_tons: float) -> int:
    if total_tons <= 0:
        return 1
    return max(4, min(36, int(math.ceil(total_tons / max(1.0, target_tons) * 1.15))))


def _assess_template_graph_health(transition_pack: dict | None, cfg: PlannerConfig) -> Dict[str, object]:
    health = {
        "template_graph_health": "UNKNOWN",
        "min_line_coverage_ratio": 0.0,
        "zero_out_nodes": 0,
        "zero_in_nodes": 0,
        "disconnected_lines": 0,
        "diagnose_global_prune": "",
    }
    if not isinstance(transition_pack, dict):
        return health

    summaries = transition_pack.get("summaries") or []
    tpl = transition_pack.get("templates")
    line_covs: list[float] = []
    zero_out = 0
    zero_in = 0
    disconnected = 0

    if isinstance(tpl, pd.DataFrame) and not tpl.empty:
        for s in summaries:
            cand = max(1, int(getattr(s, "candidate_pairs", 0)))
            feasible = int(getattr(s, "feasible_pairs", 0))
            cov = feasible / cand
            line_covs.append(cov)
            line = str(getattr(s, "line", ""))
            lt = tpl[tpl["line"] == line]
            nodes = int(getattr(s, "nodes", 0))
            if nodes <= 1:
                continue

            out_deg = lt.groupby("from_order_id").size() if not lt.empty else pd.Series(dtype=int)
            in_deg = lt.groupby("to_order_id").size() if not lt.empty else pd.Series(dtype=int)
            node_set = set(lt["from_order_id"].astype(str)).union(set(lt["to_order_id"].astype(str))) if not lt.empty else set()
            missing_nodes = max(0, nodes - len(node_set))
            line_zero_out = int(missing_nodes) + max(0, len(node_set) - int((out_deg > 0).sum()))
            line_zero_in = int(missing_nodes) + max(0, len(node_set) - int((in_deg > 0).sum()))
            zero_out += line_zero_out
            zero_in += line_zero_in
            zero_degree_ratio = (line_zero_out + line_zero_in) / max(1, 2 * nodes)
            if cov < float(cfg.model.template_health_sparse_ratio) or zero_degree_ratio > float(cfg.model.template_health_zero_degree_ratio):
                disconnected += 1

    prune_summaries = transition_pack.get("prune_summaries") or []
    prune_too_strong = False
    if prune_summaries:
        pruned_topk = int(sum(int(getattr(p, "pruned_topk", 0)) for p in prune_summaries))
        kept = int(sum(int(getattr(p, "kept_templates", 0)) for p in prune_summaries))
        prune_too_strong = kept > 0 and pruned_topk > kept * 4

    min_cov = min(line_covs) if line_covs else 0.0
    status = "HEALTHY"
    if disconnected > 0 or (line_covs and min_cov < float(cfg.model.template_health_sparse_ratio) * 0.5):
        status = "DISCONNECTED"
    elif prune_too_strong or (line_covs and min_cov < float(cfg.model.template_health_sparse_ratio)):
        status = "SPARSE"

    health.update(
        {
            "template_graph_health": status,
            "min_line_coverage_ratio": round(float(min_cov), 4),
            "zero_out_nodes": int(zero_out),
            "zero_in_nodes": int(zero_in),
            "disconnected_lines": int(disconnected),
            "diagnose_global_prune": "全局剪枝过强" if prune_too_strong else "",
        }
    )
    return health


def _build_joint_failure_diagnostics(
    cfg: PlannerConfig,
    orders_df: pd.DataFrame | None,
    transition_pack: dict | None,
    last_joint: dict | None,
) -> Dict[str, object]:
    diag: Dict[str, object] = {}
    diag.update(_assess_template_graph_health(transition_pack, cfg))
    diag["last_status"] = str((last_joint or {}).get("status", "UNKNOWN"))
    diag["enable_semantic_fallback"] = bool(getattr(cfg.model, "enableSemanticFallback", False))
    diag["enable_scale_down_fallback"] = bool(getattr(cfg.model, "enableScaleDownFallback", False))
    diag["enable_legacy_fallback"] = bool(getattr(cfg.model, "enableLegacyFallback", False))
    diag["global_prune_max_pairs_per_from"] = int(cfg.model.global_prune_max_pairs_per_from or 0)
    diag["min_real_schedule_ratio"] = float(cfg.model.min_real_schedule_ratio)

    if isinstance(orders_df, pd.DataFrame) and not orders_df.empty:
        diag["orders"] = int(len(orders_df))
        total_tons = float(orders_df["tons"].sum()) if "tons" in orders_df.columns else 0.0
        diag["total_tons"] = round(total_tons, 1)
        est_slots = _estimate_campaign_slots(total_tons, float(cfg.rule.campaign_ton_target))
        diag["estimated_slots"] = int(est_slots)
        slot_cap = int(cfg.model.max_campaign_slots)
        diag["slot_cap"] = slot_cap
        if est_slots > slot_cap:
            diag["diagnose_slot_capacity"] = "槽位数量不足"

    if isinstance(transition_pack, dict):
        summaries = transition_pack.get("summaries") or []
        if summaries:
            feasible = int(sum(int(s.feasible_pairs) for s in summaries))
            cand = int(sum(int(s.candidate_pairs) for s in summaries))
            diag["template_coverage_ratio"] = round(feasible / max(1, cand), 4)
            diag["template_coverage_pairs"] = f"{feasible}/{cand}"
            if cand > 0 and feasible / cand < float(cfg.model.template_health_sparse_ratio):
                diag["diagnose_template_coverage"] = "模板覆盖不足"
        prune_summaries = transition_pack.get("prune_summaries") or []
        if prune_summaries:
            pruned_topk = int(sum(int(p.pruned_topk) for p in prune_summaries))
            kept = int(sum(int(p.kept_templates) for p in prune_summaries))
            diag["pruned_topk"] = pruned_topk
            diag["kept_templates"] = kept
            if kept > 0 and pruned_topk > kept * 4:
                diag["diagnose_global_prune"] = "全局剪枝过强"

    if float(cfg.model.min_real_schedule_ratio) >= 0.95:
        diag["diagnose_min_real_ratio"] = "min_real_schedule_ratio 可能过紧"
    return diag
