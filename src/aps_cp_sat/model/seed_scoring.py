from __future__ import annotations

from typing import Any

import pandas as pd

from aps_cp_sat.model.candidate_graph_types import DIRECT_EDGE
from aps_cp_sat.model.edge_hard_filter import sequence_pair_passes_final_hard_rules


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value: object, default: int = 0) -> int:
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass
    try:
        return int(value)
    except Exception:
        return default


def _normalize_score(value: float, scale: float) -> float:
    if scale <= 0:
        return 0.0
    return max(0.0, min(1.0, float(value) / float(scale)))


def _is_virtual_record(rec: dict[str, Any]) -> bool:
    return bool(rec.get("is_virtual", False)) or str(rec.get("virtual_origin", "") or "") == "prebuilt_inventory"


def _candidate_successors_for_seed(
    *,
    order_id: str,
    order_row: dict[str, Any],
    line: str,
    candidate_graph,
    cfg,
) -> list[dict[str, Any]]:
    out_edges = getattr(candidate_graph, "out_edges", {}) or {}
    edge_cost = getattr(candidate_graph, "edge_cost", {}) or {}
    order_record = getattr(candidate_graph, "order_record", {}) or {}
    rows: list[dict[str, Any]] = []
    for succ_oid, tpl_row in out_edges.get(str(order_id), []):
        if str(tpl_row.get("line", "big_roll")) != str(line):
            continue
        succ_rec = dict(order_record.get(str(succ_oid), {}) or {})
        if not succ_rec:
            continue
        ok, _ = sequence_pair_passes_final_hard_rules(
            order_row,
            succ_rec,
            cfg,
            {
                "line": line,
                "edge_type": str(tpl_row.get("edge_type", DIRECT_EDGE) or DIRECT_EDGE),
                "template_row": tpl_row,
                "bridge_count": int(tpl_row.get("bridge_count", 0) or 0),
            },
        )
        if not ok:
            continue
        rows.append(
            {
                "order_id": str(succ_oid),
                "record": succ_rec,
                "tpl_row": dict(tpl_row),
                "edge_cost": int(edge_cost.get((str(order_id), str(succ_oid)), 0) or 0),
            }
        )
    return rows


def _seed_lookahead_projected_tons(
    *,
    order_id: str,
    order_row: dict[str, Any],
    line: str,
    candidate_graph,
    cfg,
    lookahead_orders: int,
) -> tuple[float, int]:
    order_record = getattr(candidate_graph, "order_record", {}) or {}
    projected_tons = _safe_float(order_row.get("tons", 0.0))
    visited: set[str] = {str(order_id)}
    current_oid = str(order_id)
    current_row = dict(order_row)
    steps = 0
    while steps < max(1, int(lookahead_orders)):
        successors = [
            s for s in _candidate_successors_for_seed(
                order_id=current_oid,
                order_row=current_row,
                line=line,
                candidate_graph=candidate_graph,
                cfg=cfg,
            )
            if str(s.get("order_id", "")) not in visited
        ]
        if not successors:
            break
        successors.sort(
            key=lambda s: (
                1 if _is_virtual_record(dict(s.get("record", {}) or {})) else 0,
                -_safe_float(dict(s.get("record", {}) or {}).get("tons", 0.0)),
                int(s.get("edge_cost", 0) or 0),
                str(s.get("order_id", "")),
            )
        )
        best = successors[0]
        next_oid = str(best.get("order_id", "") or "")
        if not next_oid:
            break
        next_row = dict(order_record.get(next_oid, best.get("record", {}) or {}))
        projected_tons += _safe_float(next_row.get("tons", 0.0))
        visited.add(next_oid)
        current_oid = next_oid
        current_row = next_row
        steps += 1
    return float(projected_tons), int(max(0, len(visited) - 1))


def compute_seed_business_score(order_row, line_orders_df, candidate_graph, cfg) -> dict:
    rec = dict(order_row if isinstance(order_row, dict) else getattr(order_row, "to_dict", lambda: {})())
    oid = str(rec.get("order_id", "") or "")
    line = str(rec.get("line", "") or "")
    if not oid:
        return {
            "order_id": oid,
            "line": line,
            "seed_score_total": 0.0,
            "width_headroom_score": 0.0,
            "extendability_score": 0.0,
            "tonnage_formability_score": 0.0,
            "group_continuity_score": 0.0,
            "smoothness_seed_score": 0.0,
            "reverse_budget_friendliness_score": 0.0,
        }

    line_df = line_orders_df.copy() if isinstance(line_orders_df, pd.DataFrame) else pd.DataFrame()
    if not line_df.empty and "line" in line_df.columns:
        line_df = line_df[line_df["line"].astype(str) == line].copy()

    width = _safe_float(rec.get("width", 0.0))
    tons = _safe_float(rec.get("tons", 0.0))
    steel_group = str(rec.get("steel_group", "") or "")
    model = getattr(cfg, "model", None) if cfg is not None else None
    lookahead_orders = int(getattr(model, "seed_formability_lookahead_orders", 4) if model else 4)
    lookahead_orders = max(1, lookahead_orders)
    min_projected_tons = float(getattr(model, "seed_formability_min_projected_tons", 350.0) if model else 350.0)
    soft_gate_enabled = bool(getattr(model, "seed_formability_soft_gate_enabled", True) if model else True)
    hard_gate_enabled = bool(getattr(model, "seed_formability_hard_gate_enabled", False) if model else False)
    low_formability_penalty = float(getattr(model, "seed_penalty_low_formability", 80.0) if model else 80.0)
    width_head_bonus = float(getattr(model, "seed_bonus_width_head", 35.0) if model else 35.0)
    high_extendability_bonus = float(getattr(model, "seed_bonus_high_extendability", 25.0) if model else 25.0)

    width_headroom_score = 0.0
    if not line_df.empty and "width" in line_df.columns:
        widths = pd.to_numeric(line_df["width"], errors="coerce").dropna()
        if not widths.empty:
            width_headroom_score = float((widths <= width).mean())

    successors = _candidate_successors_for_seed(
        order_id=oid,
        order_row=rec,
        line=line,
        candidate_graph=candidate_graph,
        cfg=cfg,
    )
    extendability_score = _normalize_score(len(successors), 8.0)

    projected_tons, projected_successor_count = _seed_lookahead_projected_tons(
        order_id=oid,
        order_row=rec,
        line=line,
        candidate_graph=candidate_graph,
        cfg=cfg,
        lookahead_orders=lookahead_orders,
    )
    ton_min = float(getattr(getattr(cfg, "rule", None), "campaign_ton_min", 700.0) or 700.0)
    gap_after = max(0.0, ton_min - projected_tons)
    tonnage_formability_score = 1.0 - min(1.0, gap_after / max(ton_min, 1.0))
    low_formability = bool(projected_tons < min_projected_tons - 1e-6)

    if successors:
        same_group = sum(1 for s in successors if str(s["record"].get("steel_group", "") or "") == steel_group)
        group_continuity_score = float(same_group / max(1, len(successors)))
        smooth_samples: list[float] = []
        reverse_hits = 0
        for s in successors:
            succ_width = _safe_float(s["record"].get("width", 0.0))
            succ_thickness = _safe_float(s["record"].get("thickness", 0.0))
            succ_temp_min = _safe_float(s["record"].get("temp_min", rec.get("temp_min", 0.0)))
            succ_temp_max = _safe_float(s["record"].get("temp_max", rec.get("temp_max", 0.0)))
            width_drop = max(0.0, width - succ_width)
            width_rise = max(0.0, succ_width - width)
            thickness_gap = abs(succ_thickness - _safe_float(rec.get("thickness", 0.0)))
            temp_center_gap = abs(
                ((_safe_float(rec.get("temp_min", 0.0)) + _safe_float(rec.get("temp_max", 0.0))) * 0.5)
                - ((succ_temp_min + succ_temp_max) * 0.5)
            )
            smooth_samples.append(max(0.0, 1.0 - ((width_drop * 0.002) + (width_rise * 0.004) + (thickness_gap * 1.5) + (temp_center_gap * 0.005))))
            if succ_width > width:
                reverse_hits += 1
        smoothness_seed_score = float(sum(smooth_samples) / max(1, len(smooth_samples)))
        reverse_budget_friendliness_score = 1.0 - float(reverse_hits / max(1, len(successors)))
    else:
        group_continuity_score = 0.0
        smoothness_seed_score = 0.0
        reverse_budget_friendliness_score = 0.0

    w_width = float(getattr(model, "seed_weight_width_headroom", 0.26) if model else 0.26)
    w_extend = float(getattr(model, "seed_weight_extendability", 0.20) if model else 0.20)
    w_form = float(getattr(model, "seed_weight_tonnage_formability", 0.22) if model else 0.22)
    w_group = float(getattr(model, "seed_weight_group_continuity", 0.14) if model else 0.14)
    w_smooth = float(getattr(model, "seed_weight_smoothness", 0.10) if model else 0.10)
    w_reverse = float(getattr(model, "seed_weight_reverse_friendliness", 0.08) if model else 0.08)
    score_total = (
        w_width * width_headroom_score
        + w_extend * extendability_score
        + w_form * tonnage_formability_score
        + w_group * group_continuity_score
        + w_smooth * smoothness_seed_score
        + w_reverse * reverse_budget_friendliness_score
    )
    score_bonus = 0.0
    score_penalty = 0.0
    if width_headroom_score >= 0.75:
        score_bonus += width_head_bonus * width_headroom_score
    if extendability_score >= 0.5:
        score_bonus += high_extendability_bonus * extendability_score
    if low_formability:
        if hard_gate_enabled:
            score_penalty += 1_000_000.0
        elif soft_gate_enabled:
            score_penalty += low_formability_penalty
    score_total = float(score_total * 100.0 + score_bonus - score_penalty)

    return {
        "order_id": oid,
        "line": line,
        "seed_score_total": float(score_total),
        "width_headroom_score": float(width_headroom_score),
        "extendability_score": float(extendability_score),
        "tonnage_formability_score": float(tonnage_formability_score),
        "group_continuity_score": float(group_continuity_score),
        "smoothness_seed_score": float(smoothness_seed_score),
        "reverse_budget_friendliness_score": float(reverse_budget_friendliness_score),
        "seed_successor_count": int(len(successors)),
        "seed_projected_tons_5step": float(projected_tons),
        "seed_projected_successor_count": int(projected_successor_count),
        "seed_formability_lookahead_orders": int(lookahead_orders),
        "seed_width_percentile": float(width_headroom_score),
        "seed_low_formability": bool(low_formability),
        "seed_formability_min_projected_tons": float(min_projected_tons),
        "seed_score_bonus": float(score_bonus),
        "seed_score_penalty": float(score_penalty),
    }


def rank_seed_candidates(line_orders_df, candidate_graph, cfg) -> pd.DataFrame:
    line_df = line_orders_df.copy() if isinstance(line_orders_df, pd.DataFrame) else pd.DataFrame()
    if line_df.empty:
        return pd.DataFrame(
            columns=[
                "order_id",
                "line",
                "seed_score_total",
                "width_headroom_score",
                "extendability_score",
                "tonnage_formability_score",
                "group_continuity_score",
                "smoothness_seed_score",
                "reverse_budget_friendliness_score",
                "seed_successor_count",
                "seed_projected_tons_5step",
                "seed_projected_successor_count",
                "seed_formability_lookahead_orders",
                "seed_width_percentile",
                "seed_low_formability",
                "seed_formability_min_projected_tons",
                "seed_score_bonus",
                "seed_score_penalty",
            ]
        )

    rows = [
        compute_seed_business_score(rec, line_df, candidate_graph, cfg)
        for rec in line_df.to_dict("records")
    ]
    ranked = pd.DataFrame(rows)
    if ranked.empty:
        return ranked
    ranked = ranked.sort_values(
        ["seed_score_total", "width_headroom_score", "extendability_score", "tonnage_formability_score", "order_id"],
        ascending=[False, False, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    ranked["seed_rank"] = ranked.index + 1
    return ranked
