from __future__ import annotations

from typing import Any

import pandas as pd

from aps_cp_sat.model.candidate_graph_types import (
    REAL_BRIDGE_EDGE,
    VIRTUAL_BRIDGE_EDGE,
    VIRTUAL_BRIDGE_FAMILY_EDGE,
)


BRIDGE_EDGE_TYPES = {REAL_BRIDGE_EDGE, VIRTUAL_BRIDGE_EDGE, VIRTUAL_BRIDGE_FAMILY_EDGE}


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _as_bool(value: Any) -> bool:
    try:
        if pd.isna(value):
            return False
    except Exception:
        pass
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _txt(value: Any) -> str:
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value or "").strip()


def _is_virtual_order(row: dict[str, Any] | None) -> bool:
    rec = dict(row or {})
    return _as_bool(rec.get("is_virtual", False)) or _txt(rec.get("virtual_origin")) == "prebuilt_inventory"


def _thickness_ok(prev_thk: float, next_thk: float) -> bool:
    delta = abs(float(next_thk) - float(prev_thk))
    if prev_thk >= 0.8:
        return (delta / max(prev_thk, 1e-9)) <= 0.30
    if prev_thk >= 0.6:
        return delta <= 0.2
    return delta <= 0.1


def _edge_type(context: dict[str, Any] | None) -> str:
    ctx = dict(context or {})
    row = dict(ctx.get("template_row") or ctx.get("edge_row") or {})
    return _txt(ctx.get("edge_type") or row.get("edge_type") or row.get("selected_edge_type"))


def _line_allowed(order: dict[str, Any], line: str | None) -> bool:
    if not line:
        return True
    cap = _txt(order.get("line_capability") or "dual").lower()
    if cap in {"dual", "both", "either", ""}:
        return True
    if cap in {"big_only", "large", "big"}:
        return line == "big_roll"
    if cap in {"small_only", "small"}:
        return line == "small_roll"
    return True


def edge_passes_final_hard_rules(
    prev_order: dict[str, Any] | None,
    next_order: dict[str, Any] | None,
    cfg,
    context: dict[str, Any] | None = None,
) -> tuple[bool, str]:
    """Pair-level hard-rule filter aligned with final schedule audit.

    This deliberately rejects edges before constructive/repair can consume them.
    It uses the realized final schedule audit semantics: no width rise, width
    drop <= max_width_drop, final thickness bands, temperature overlap, group
    transition evidence, virtual-chain limits, and line capability.
    """
    prev = dict(prev_order or {})
    nxt = dict(next_order or {})
    ctx = dict(context or {})
    rule = getattr(cfg, "rule", None)
    if not prev or not nxt:
        return False, "ORDER_CONTEXT_MISSING"

    line = _txt(ctx.get("line") or ctx.get("edge_line"))
    if not _line_allowed(prev, line) or not _line_allowed(nxt, line):
        return False, "FINAL_LINE_CAPABILITY_RULE"

    prev_width = _as_float(prev.get("width"))
    next_width = _as_float(nxt.get("width"))
    if next_width > prev_width:
        return False, "FINAL_WIDTH_RULE"
    max_drop = _as_float(getattr(rule, "max_width_drop", 250.0), 250.0)
    if prev_width - next_width > max_drop:
        return False, "FINAL_WIDTH_RULE"

    if not _thickness_ok(_as_float(prev.get("thickness")), _as_float(nxt.get("thickness"))):
        return False, "FINAL_THICKNESS_RULE"

    overlap_required = _as_float(getattr(rule, "min_temp_overlap_real_real", 10.0), 10.0)
    prev_min = _as_float(prev.get("temp_min"))
    prev_max = _as_float(prev.get("temp_max"))
    next_min = _as_float(nxt.get("temp_min"))
    next_max = _as_float(nxt.get("temp_max"))
    overlap = min(prev_max, next_max) - max(prev_min, next_min)
    if overlap < overlap_required:
        return False, "FINAL_TEMP_RULE"

    prev_group = _txt(prev.get("steel_group")).upper()
    next_group = _txt(nxt.get("steel_group")).upper()
    edge_type = _edge_type(ctx)
    if prev_group != next_group:
        row = dict(ctx.get("template_row") or ctx.get("edge_row") or {})
        explicit_ok = any(
            key in row and _as_bool(row.get(key))
            for key in ("group_transition_ok", "cross_group_transition_ok", "is_pc_transition")
        )
        pc_groups = {"PC", "VIRTUAL_PC", "鏅⒊"}
        has_bridge_evidence = edge_type in BRIDGE_EDGE_TYPES
        if not (has_bridge_evidence or explicit_ok or prev_group in pc_groups or next_group in pc_groups):
            return False, "FINAL_GROUP_RULE"

    return maybe_chain_passes_virtual_limits(prev, nxt, cfg, ctx)


def maybe_chain_passes_virtual_limits(
    prev_order: dict[str, Any] | None,
    next_order: dict[str, Any] | None,
    cfg,
    context: dict[str, Any] | None = None,
) -> tuple[bool, str]:
    ctx = dict(context or {})
    rule = getattr(cfg, "rule", None)
    max_chain = int(_as_float(getattr(rule, "max_virtual_chain", 5), 5))
    prev_run = int(_as_float(ctx.get("current_virtual_chain_length", 0), 0))
    bridge_count = int(_as_float(ctx.get("bridge_count", 0), 0))
    next_is_virtual = _is_virtual_order(next_order)
    next_run = prev_run + 1 if next_is_virtual else 0
    effective = max(next_run, bridge_count)
    if effective > max_chain:
        return False, "FINAL_VIRTUAL_CHAIN_RULE"

    campaign_count = int(_as_float(ctx.get("campaign_order_count", 0), 0))
    campaign_virtual = int(_as_float(ctx.get("campaign_virtual_count", 0), 0)) + (1 if next_is_virtual else 0)
    if campaign_count > 0:
        model = getattr(cfg, "model", None)
        ratio_num = int(_as_float(getattr(rule, "virtual_ton_ratio_num", 1), 1))
        ratio_den = int(_as_float(getattr(rule, "virtual_ton_ratio_den", 5), 5))
        max_ratio = _as_float(getattr(model, "max_virtual_node_ratio", ratio_num / max(1, ratio_den)), ratio_num / max(1, ratio_den))
        if max_ratio > 0 and campaign_virtual / max(1, campaign_count + 1) > max_ratio:
            return False, "FINAL_VIRTUAL_RATIO_RULE"

    return True, "OK"


def sequence_pair_passes_final_hard_rules(
    prev_order: dict[str, Any] | None,
    next_order: dict[str, Any] | None,
    cfg,
    context: dict[str, Any] | None = None,
) -> tuple[bool, str]:
    return edge_passes_final_hard_rules(prev_order, next_order, cfg, context)
