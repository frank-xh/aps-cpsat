from __future__ import annotations

from aps_cp_sat.config import PlannerConfig
from aps_cp_sat.transition.bridge_rules import _temp_overlap_len, _txt


def _template_cost(a: dict, b: dict, cfg: PlannerConfig, need: int, edge_type: str | None = None) -> float:
    width_d = abs(float(a["width"]) - float(b["width"]))
    thick_d = abs(float(a["thickness"]) - float(b["thickness"]))
    temp_overlap = _temp_overlap_len(float(a["temp_min"]), float(a["temp_max"]), float(b["temp_min"]), float(b["temp_max"]))
    group_switch = 1 if _txt(a.get("steel_group")) != _txt(b.get("steel_group")) else 0
    due_inv = max(0, int(a.get("due_rank", 3)) - int(b.get("due_rank", 3)))
    temp_gap = max(0.0, float(cfg.rule.min_temp_overlap_real_real) - temp_overlap)
    resolved_edge_type = str(edge_type or ("VIRTUAL_BRIDGE_EDGE" if int(need) > 0 else "DIRECT_EDGE"))
    if resolved_edge_type == "REAL_BRIDGE_EDGE":
        edge_penalty = float(cfg.score.real_bridge_penalty)
    elif resolved_edge_type == "VIRTUAL_BRIDGE_EDGE":
        edge_penalty = float(cfg.score.virtual_bridge_penalty)
    else:
        edge_penalty = float(cfg.score.direct_edge_penalty)
    return float(
        float(cfg.score.width_smooth) * width_d
        + float(max(1, cfg.score.thick_smooth // 10)) * thick_d
        + float(max(1, cfg.score.temp_shortage // 100)) * temp_gap
        + float(max(1, cfg.score.non_pc_switch // 100)) * group_switch
        + float(max(1, cfg.score.edge_fallback_due_weight // 10)) * due_inv
        + float(max(1, cfg.score.virtual_use * 40)) * need
        + edge_penalty
    )


def _template_cost_parts(a: dict, b: dict, bridge_path: list[dict]) -> dict[str, float]:
    seq = [a] + list(bridge_path) + [b]
    width_smooth = 0.0
    thick_smooth = 0.0
    temp_margin = 0.0
    cross_group = 0.0
    for i in range(len(seq) - 1):
        x, y = seq[i], seq[i + 1]
        wd = max(0.0, float(x["width"]) - float(y["width"]))
        td = abs(float(x["thickness"]) - float(y["thickness"]))
        ov = _temp_overlap_len(float(x["temp_min"]), float(x["temp_max"]), float(y["temp_min"]), float(y["temp_max"]))
        width_smooth += wd
        thick_smooth += td
        if 10.0 <= ov < 30.0:
            temp_margin += 30.0 - ov
        if _txt(x.get("steel_group")) != _txt(y.get("steel_group")):
            cross_group += 1.0
    return {
        "width_smooth_cost": float(width_smooth),
        "thickness_smooth_cost": float(thick_smooth),
        "temp_margin_cost": float(temp_margin),
        "cross_group_cost": float(cross_group),
    }
