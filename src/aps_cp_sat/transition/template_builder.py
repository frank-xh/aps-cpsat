from __future__ import annotations

import math
from time import perf_counter
from typing import Dict, Optional, Tuple

import pandas as pd

from aps_cp_sat.config import PlannerConfig
from aps_cp_sat.transition.bridge_rules import build_virtual_spec_views, _bridge_need, _bridge_pair, _temp_overlap_len, _txt
from aps_cp_sat.transition.template_cost import _template_cost, _template_cost_parts


def _line_feasible(df: pd.DataFrame, line: str) -> pd.DataFrame:
    if "line_capability" not in df.columns:
        return df.copy()
    if line == "big_roll":
        mask = df["line_capability"].isin(["big_only", "dual"])
    else:
        mask = df["line_capability"].isin(["small_only", "dual"])
    return df[mask].copy()


def _encode_bridge_path(bridge_path: list[dict]) -> str:
    if not bridge_path:
        return ""
    parts = []
    for v in bridge_path:
        parts.append(
            f"W{float(v['width']):.0f}|T{float(v['thickness']):.1f}|TMP[{float(v['temp_min']):.0f},{float(v['temp_max']):.0f}]"
        )
    return " -> ".join(parts)


def _materialize_bridge_template(a: dict, b: dict, max_chain: int, cfg: PlannerConfig, spec_views: dict[str, list | tuple]) -> Optional[Tuple[list[dict], int]]:
    need = _bridge_need(a, b, max_chain, cfg.rule, strict_virtual_width_levels=cfg.model.strict_virtual_width_levels, spec_views=spec_views)
    if need < 0 or need > max_chain:
        return None
    if need == 0:
        return ([], 0)
    try:
        vs, _ = _bridge_pair(
            a,
            b,
            max_chain,
            vidx_start=1,
            campaign_id=0,
            rule=cfg.rule,
            strict_virtual_width_levels=cfg.model.strict_virtual_width_levels,
            physical_reverse_step_mode=cfg.model.physical_reverse_step_mode,
            spec_views=spec_views,
        )
    except Exception:
        return None
    return (vs, len(vs))


def _build_line_templates(df_line: pd.DataFrame, line: str, cfg: PlannerConfig) -> tuple[pd.DataFrame, dict]:
    if df_line.empty:
        return pd.DataFrame(
            columns=[
                "line",
                "from_order_id",
                "to_order_id",
                "bridge_count",
                "virtual_tons",
                "cost",
                "temp_overlap",
                "width_delta",
                "thickness_delta",
                "group_switch",
                "logical_reverse_flag",
                "physical_reverse_count",
            ]
        ), {
            "pruned_unbridgeable": 0,
            "pruned_topk": 0,
            "pruned_degree": 0,
            "kept_templates": 0,
        }
    recs = df_line.to_dict("records")
    # Read-only virtual spec snapshot for this line build only. This is not a
    # global cache; the lifetime is exactly one line-template construction pass.
    spec_views = build_virtual_spec_views(cfg.rule)
    pair_scan_seconds = 0.0
    bridge_check_seconds = 0.0
    template_prune_seconds = 0.0
    rows: list[dict] = []
    keep_k = max(
        int(cfg.model.template_top_k),
        cfg.model.sparse_k_same_group + cfg.model.sparse_k_same_thickness + cfg.model.sparse_k_cross_group + cfg.model.sparse_k_due_tight,
    )
    min_out = max(1, int(cfg.model.template_min_out_degree))
    min_in = max(0, int(cfg.model.template_min_in_degree))
    hard_max_bridge = min(int(cfg.model.max_virtual_chain), int(cfg.rule.max_virtual_chain))

    cands_cache: Dict[int, list[tuple[int, int, float, float]]] = {}
    incoming_cache: Dict[int, list[tuple[int, int, float, float]]] = {j: [] for j in range(len(recs))}
    pruned_unbridgeable = 0
    pruned_topk = 0
    pruned_degree = 0
    candidate_pairs = 0

    for i, a in enumerate(recs):
        cands: list[tuple[int, int, float, float]] = []
        scan_t0 = perf_counter()
        for j, b in enumerate(recs):
            if i == j:
                continue
            candidate_pairs += 1
            bridge_t0 = perf_counter()
            need = _bridge_need(a, b, hard_max_bridge, cfg.rule, strict_virtual_width_levels=cfg.model.strict_virtual_width_levels, spec_views=spec_views)
            bridge_check_seconds += perf_counter() - bridge_t0
            if need > hard_max_bridge:
                pruned_unbridgeable += 1
                continue
            temp_overlap = _temp_overlap_len(float(a["temp_min"]), float(a["temp_max"]), float(b["temp_min"]), float(b["temp_max"]))
            c = _template_cost(a, b, cfg, need)
            cands.append((j, need, c, temp_overlap))
        pair_scan_seconds += perf_counter() - scan_t0
        cands.sort(key=lambda x: (x[2], x[1]))
        cands_cache[i] = cands
        for j, need, c, temp_overlap in cands:
            incoming_cache.setdefault(j, []).append((i, need, c, temp_overlap))

    selected_pairs = set()

    def add_pair(i: int, j: int, need: int, c: float, temp_overlap: float) -> bool:
        if (i, j) in selected_pairs:
            return False
        a = recs[i]
        b = recs[j]
        mat = _materialize_bridge_template(a, b, hard_max_bridge, cfg, spec_views)
        if mat is None:
            return False
        bridge_path, bridge_count = mat
        if float(b["width"]) > float(a["width"]) and int(bridge_count) <= 0:
            return False
        parts = _template_cost_parts(a, b, bridge_path)
        tpl_id = f"{line}:{_txt(a.get('order_id'))}->{_txt(b.get('order_id'))}#k{bridge_count}"
        rows.append(
            {
                "template_id": tpl_id,
                "line": line,
                "from_order_id": _txt(a.get("order_id")),
                "to_order_id": _txt(b.get("order_id")),
                "edge_type": "VIRTUAL_BRIDGE_EDGE" if int(bridge_count) > 0 else "DIRECT_EDGE",
                "bridge_count": int(bridge_count),
                "virtual_tons": float(cfg.rule.virtual_tons * bridge_count),
                "cost": float(c),
                "width_smooth_cost": float(parts["width_smooth_cost"]),
                "thickness_smooth_cost": float(parts["thickness_smooth_cost"]),
                "temp_margin_cost": float(parts["temp_margin_cost"]),
                "cross_group_cost": float(parts["cross_group_cost"]),
                "bridge_path": _encode_bridge_path(bridge_path),
                "temp_overlap": float(temp_overlap),
                "width_delta": float(a["width"]) - float(b["width"]),
                "thickness_delta": abs(float(a["thickness"]) - float(b["thickness"])),
                "group_switch": int(_txt(a.get("steel_group")) != _txt(b.get("steel_group"))),
                "logical_reverse_flag": int(float(b["width"]) > float(a["width"])),
                "physical_reverse_count": int(
                    max(
                        0,
                        math.ceil(
                            max(0.0, float(b["width"]) - float(a["width"]))
                            / max(1.0, float(cfg.rule.max_width_rise_physical_step))
                        ),
                    )
                ),
            }
        )
        selected_pairs.add((i, j))
        return True

    for i in range(len(recs)):
        prune_t0 = perf_counter()
        cands = cands_cache.get(i, [])
        if not cands:
            template_prune_seconds += perf_counter() - prune_t0
            continue
        added = 0
        for j, need, c, temp_overlap in cands[:keep_k]:
            if add_pair(i, j, need, c, temp_overlap):
                added += 1
        pruned_topk += max(0, len(cands) - keep_k)
        if added >= min_out:
            continue
        for j, need, c, temp_overlap in cands[keep_k:]:
            if added >= min_out:
                break
            if add_pair(i, j, need, c, temp_overlap):
                added += 1
        if added < min_out:
            pruned_degree += 1
        template_prune_seconds += perf_counter() - prune_t0

    if min_in <= 0:
        avg_virtual_count = float(sum(int(r.get("bridge_count", 0) or 0) for r in rows) / max(1, len(rows))) if rows else 0.0
        max_virtual_count = int(max((int(r.get("bridge_count", 0) or 0) for r in rows), default=0))
        return pd.DataFrame(rows), {
            "candidate_pairs": int(candidate_pairs),
            "pruned_unbridgeable": int(pruned_unbridgeable),
            "pruned_topk": int(pruned_topk),
            "pruned_degree": int(pruned_degree),
            "kept_templates": int(len(rows)),
            "avg_virtual_count": round(avg_virtual_count, 4),
            "max_virtual_count": int(max_virtual_count),
            "direct_edge_count": int(sum(1 for r in rows if str(r.get("edge_type")) == "DIRECT_EDGE")),
            "real_bridge_edge_count": int(sum(1 for r in rows if str(r.get("edge_type")) == "REAL_BRIDGE_EDGE")),
            "virtual_bridge_edge_count": int(sum(1 for r in rows if str(r.get("edge_type")) == "VIRTUAL_BRIDGE_EDGE")),
            "rejected_by_chain_limit": int(pruned_unbridgeable),
            "rejected_by_no_legal_next_step": 0,
            "rejected_by_later_step_not_feasible": 0,
            "spec_views_reused": True,
            "spec_views_build_count": 1,
            "template_pair_scan_seconds": round(float(pair_scan_seconds), 6),
            "bridge_check_seconds": round(float(bridge_check_seconds), 6),
            "template_prune_seconds": round(float(template_prune_seconds), 6),
        }

    in_deg: Dict[int, int] = {j: 0 for j in range(len(recs))}
    for _, j in selected_pairs:
        in_deg[j] = in_deg.get(j, 0) + 1
    for j in range(len(recs)):
        prune_t0 = perf_counter()
        need_in = max(0, min_in - int(in_deg.get(j, 0)))
        if need_in <= 0:
            template_prune_seconds += perf_counter() - prune_t0
            continue
        cands = sorted(incoming_cache.get(j, []), key=lambda x: (x[2], x[1]))
        for i, need, c, temp_overlap in cands:
            if need_in <= 0:
                break
            if add_pair(i, j, need, c, temp_overlap):
                need_in -= 1
        if need_in > 0:
            pruned_degree += 1
        template_prune_seconds += perf_counter() - prune_t0

    out = pd.DataFrame(rows)
    avg_virtual_count = float(out["bridge_count"].mean()) if not out.empty and "bridge_count" in out.columns else 0.0
    max_virtual_count = int(out["bridge_count"].max()) if not out.empty and "bridge_count" in out.columns else 0
    return out, {
        "candidate_pairs": int(candidate_pairs),
        "pruned_unbridgeable": int(pruned_unbridgeable),
        "pruned_topk": int(pruned_topk),
        "pruned_degree": int(pruned_degree),
        "kept_templates": int(len(out)),
        "avg_virtual_count": round(avg_virtual_count, 4),
        "max_virtual_count": int(max_virtual_count),
        "direct_edge_count": int((out["edge_type"] == "DIRECT_EDGE").sum()) if not out.empty and "edge_type" in out.columns else 0,
        "real_bridge_edge_count": int((out["edge_type"] == "REAL_BRIDGE_EDGE").sum()) if not out.empty and "edge_type" in out.columns else 0,
        "virtual_bridge_edge_count": int((out["edge_type"] == "VIRTUAL_BRIDGE_EDGE").sum()) if not out.empty and "edge_type" in out.columns else 0,
        "rejected_by_chain_limit": int(pruned_unbridgeable),
        "rejected_by_no_legal_next_step": 0,
        "rejected_by_later_step_not_feasible": 0,
        "spec_views_reused": True,
        "spec_views_build_count": 1,
        "template_pair_scan_seconds": round(float(pair_scan_seconds), 6),
        "bridge_check_seconds": round(float(bridge_check_seconds), 6),
        "template_prune_seconds": round(float(template_prune_seconds), 6),
    }
