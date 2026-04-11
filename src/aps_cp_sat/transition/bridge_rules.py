from __future__ import annotations

import math
from typing import List, Tuple

import pandas as pd

from aps_cp_sat.config.rule_config import RuleConfig


DUE_RANK = {"overdue": 0, "urgent": 1, "normal": 2, "slack": 3}


def build_virtual_spec_views(rule: RuleConfig) -> dict[str, list | tuple]:
    widths = [float(v) for v in rule.virtual_width_levels]
    thicks = [float(v) for v in rule.virtual_thickness_levels]
    temp_range = (float(rule.virtual_temp_min), float(rule.virtual_temp_max))
    return {"widths": widths, "thicks": thicks, "temp_range": temp_range}


def _resolve_spec_views(rule: RuleConfig, spec_views: dict[str, list | tuple] | None) -> dict[str, list | tuple]:
    # Compatibility/test fallback only.
    # Production template building must construct spec_views once per build pass
    # and pass it explicitly through the high-frequency bridge evaluation chain.
    return spec_views if spec_views is not None else build_virtual_spec_views(rule)


def _txt(v, default: str = "") -> str:
    if v is None:
        return default
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if float(v).is_integer():
            return str(int(v))
        return str(v)
    s = str(v).strip()
    if "." in s:
        left, right = s.split(".", 1)
        if left.isdigit() and right.isdigit() and set(right) <= {"0"}:
            return left
    return s


def _is_pc(group: str) -> bool:
    return _txt(group).upper() in {"PC", "普碳", "VIRTUAL_PC"}


def _thick_ok(prev_t: float, cur_t: float) -> bool:
    diff = abs(cur_t - prev_t)
    if prev_t > 0.8:
        return (diff / prev_t) <= 0.30
    if prev_t >= 0.6:
        return diff <= 0.2
    return diff <= 0.1


def _temp_overlap_len(a_min: float, a_max: float, b_min: float, b_max: float) -> float:
    return max(0.0, min(float(a_max), float(b_max)) - max(float(a_min), float(b_min)))


def _required_temp_overlap(a: dict, b: dict, rule: RuleConfig) -> float:
    if bool(a.get("is_virtual", False)) or bool(b.get("is_virtual", False)):
        return 0.0
    return float(rule.min_temp_overlap_real_real)


def _temp_transition_ok(a: dict, b: dict, rule: RuleConfig) -> bool:
    overlap = _temp_overlap_len(a.get("temp_min", 0.0), a.get("temp_max", 0.0), b.get("temp_min", 0.0), b.get("temp_max", 0.0))
    return overlap >= _required_temp_overlap(a, b, rule)


def _temp_center(a: dict) -> float:
    lo = float(a.get("temp_min", 0.0))
    hi = float(a.get("temp_max", 0.0))
    return (lo + hi) / 2.0


def _nearest(vals: List[float], x: float) -> float:
    return min(vals, key=lambda v: abs(v - x))


def _nearest_n(vals: List[float], x: float, n: int) -> List[float]:
    return sorted(vals, key=lambda v: abs(v - x))[: max(1, int(n))]


def _virtual_temp_window(center: float, temp_range: Tuple[float, float]) -> Tuple[float, float]:
    # Virtual slab temperature is a continuous feasible interval [min, max].
    # No band slicing is applied in the production path.
    lo, hi = float(temp_range[0]), float(temp_range[1])
    return lo, hi


def real_reverse_step_within_limit(a: dict, b: dict, rule: RuleConfig) -> bool:
    return max(0.0, float(b.get("width", 0.0)) - float(a.get("width", 0.0))) <= float(rule.real_reverse_step_max_mm)


def bridge_step_within_limit(a: dict, b: dict, rule: RuleConfig) -> bool:
    return bool(b.get("is_virtual", False)) and (
        max(0.0, float(b.get("width", 0.0)) - float(a.get("width", 0.0))) <= float(rule.virtual_reverse_attach_max_mm)
    )


def reverse_step_within_applicable_limit(a: dict, b: dict, rule: RuleConfig) -> bool:
    if bool(b.get("is_virtual", False)):
        return bridge_step_within_limit(a, b, rule)
    return real_reverse_step_within_limit(a, b, rule)


def _hard_direct_step_ok(a: dict, b: dict, rule: RuleConfig) -> bool:
    if float(b["width"]) > float(a["width"]):
        if not reverse_step_within_applicable_limit(a, b, rule):
            return False
    elif (float(a["width"]) - float(b["width"])) > float(rule.max_width_drop):
        return False
    if not _thick_ok(a["thickness"], b["thickness"]):
        return False
    if not _temp_transition_ok(a, b, rule):
        return False
    ga = _txt(a.get("steel_group", "")).upper()
    gb = _txt(b.get("steel_group", "")).upper()
    if ga != gb and (not _is_pc(ga)) and (not _is_pc(gb)):
        return False
    return True


def _width_reverse_virtual_need(
    a_width: float,
    b_width: float,
    rule: RuleConfig,
    strict_virtual_width_levels: bool = False,
    spec_views: dict[str, list | tuple] | None = None,
) -> int:
    spec = _resolve_spec_views(rule, spec_views)
    widths = spec["widths"]
    delta = float(b_width) - float(a_width)
    if delta <= 0.0:
        return 0
    step = max(1.0, float(rule.virtual_reverse_attach_max_mm))
    if strict_virtual_width_levels:
        w0 = _nearest(widths, float(a_width))
        w1 = _nearest(widths, float(b_width))
        if w1 <= w0:
            return 0
        physical_steps = int(math.ceil((w1 - w0) / step))
    else:
        physical_steps = int(math.ceil(delta / step))
    return max(1, physical_steps - 1)


def _bridge_need(
    a: dict,
    b: dict,
    max_virtual_chain: int,
    rule: RuleConfig,
    strict_virtual_width_levels: bool = False,
    spec_views: dict[str, list | tuple] | None = None,
) -> int:
    spec = _resolve_spec_views(rule, spec_views)
    widths = spec["widths"]
    max_virtual_chain = int(max_virtual_chain)
    if _hard_direct_step_ok(a, b, rule):
        return 0
    if float(b["width"]) > float(a["width"]):
        need_reverse = _width_reverse_virtual_need(
            float(a["width"]),
            float(b["width"]),
            rule,
            strict_virtual_width_levels=bool(strict_virtual_width_levels),
            spec_views=spec,
        )
        if need_reverse >= 10**9:
            return max_virtual_chain + 1
    else:
        need_reverse = 0
    v_w_max = float(max(widths)) if widths else 0.0
    if float(b["width"]) > v_w_max:
        return max_virtual_chain + 1
    if float(a["width"]) - v_w_max > float(rule.max_width_drop):
        return max_virtual_chain + 1
    need_w = max(
        0,
        math.ceil(
            max(0.0, a["width"] - b["width"] - float(rule.max_width_drop))
            / max(1.0, float(rule.max_width_drop))
        ),
    )
    need_group = 1 if (_txt(a["steel_group"]).upper() != _txt(b["steel_group"]).upper() and (not _is_pc(a["steel_group"])) and (not _is_pc(b["steel_group"]))) else 0
    if _thick_ok(a["thickness"], b["thickness"]):
        need_t = 0
    else:
        need_t = max_virtual_chain + 1
        for n in range(1, max_virtual_chain + 1):
            prev = a["thickness"]
            ok = True
            for k in range(1, n + 1):
                t = a["thickness"] + (b["thickness"] - a["thickness"]) * k / (n + 1)
                t = min(2.0, max(0.5, round(t, 1)))
                if not _thick_ok(prev, t):
                    ok = False
                    break
                prev = t
            if ok and _thick_ok(prev, b["thickness"]):
                need_t = n
                break
    need_temp = 0
    if not _temp_transition_ok(a, b, rule):
        need_temp = 1
    need = max(need_w, need_group, need_t, need_temp, int(need_reverse))
    return need if need <= max_virtual_chain else max_virtual_chain + 1


def _make_virtual(
    idx: int,
    line: str,
    width: float,
    thickness: float,
    temp_min: float,
    temp_max: float,
    campaign_id: int,
    rule: RuleConfig,
    spec_views: dict[str, list | tuple] | None = None,
) -> dict:
    temp_min = max(float(rule.virtual_temp_min), float(temp_min))
    temp_max = min(float(rule.virtual_temp_max), float(temp_max))
    return {
        "order_id": f"VIRTUAL-{idx:05d}",
        "source_order_id": "",
        "parent_order_id": "",
        "lot_id": "",
        "grade": "VIRTUAL_PC",
        "steel_group": "PC",
        "width": float(width),
        "thickness": float(thickness),
        "temp_min": float(temp_min),
        "temp_max": float(temp_max),
        "temp_mean": (float(temp_min) + float(temp_max)) / 2.0,
        "tons": float(rule.virtual_tons),
        "backlog": 0.0,
        "line": line,
        "line_source": "virtual",
        "due_date": pd.NaT,
        "due_bucket": "slack",
        "due_rank": DUE_RANK["slack"],
        "proc_hours": 0.0,
        "priority": 0,
        "campaign_id": int(campaign_id),
        "is_virtual": True,
    }


def _bridge_pair(
    a: dict,
    b: dict,
    max_virtual_chain: int,
    vidx_start: int,
    campaign_id: int,
    rule: RuleConfig,
    strict_virtual_width_levels: bool = False,
    physical_reverse_step_mode: bool = True,
    spec_views: dict[str, list | tuple] | None = None,
) -> Tuple[List[dict], int]:
    spec = _resolve_spec_views(rule, spec_views)
    widths = spec["widths"]
    thicks = spec["thicks"]
    temp_range = spec["temp_range"]
    max_virtual_chain = int(max_virtual_chain)
    need = _bridge_need(
        a,
        b,
        max_virtual_chain,
        rule,
        strict_virtual_width_levels=bool(strict_virtual_width_levels),
        spec_views=spec,
    )
    if need == 0:
        return [], vidx_start
    if need > max_virtual_chain:
        raise RuntimeError("bridge unavailable")

    if bool(physical_reverse_step_mode) and float(b["width"]) > float(a["width"]):
        delta = float(b["width"]) - float(a["width"])
        physical_step = max(1.0, float(rule.virtual_reverse_attach_max_mm))
        physical_steps = int(math.ceil(delta / physical_step))
        n = max(int(need), max(1, physical_steps - 1))
        if n > max_virtual_chain:
            raise RuntimeError("reverse-width bridge exceeds max virtual chain")
        chain: List[dict] = []
        prev = a
        for k in range(n):
            frac = (k + 1) / (n + 1)
            raw_w = float(a["width"]) + delta * frac
            raw_t = float(a["thickness"] + (b["thickness"] - a["thickness"]) * frac)
            t = _nearest(thicks, raw_t)
            raw_temp = float(_temp_center(a) + (_temp_center(b) - _temp_center(a)) * frac)
            tb_min, tb_max = _virtual_temp_window(raw_temp, temp_range)
            v = None
            for w in _nearest_n(widths, raw_w, len(widths)):
                cand = _make_virtual(vidx_start + k, _txt(a["line"]), float(w), float(t), float(tb_min), float(tb_max), campaign_id, rule, spec_views=spec)
                if not _hard_direct_step_ok(prev, cand, rule):
                    continue
                if k == n - 1 and not _hard_direct_step_ok(cand, b, rule):
                    continue
                v = cand
                break
            if v is None:
                raise RuntimeError("reverse-width virtual step invalid")
            prev = v
            chain.append(v)
        if not _hard_direct_step_ok(prev, b, rule):
            raise RuntimeError("reverse-width tail step invalid")
        return chain, vidx_start + n

    def _try_with_n(n: int, start_vidx: int) -> Tuple[List[dict], int] | None:
        chain: List[dict] = []
        prev = a
        for k in range(n):
            frac = (k + 1) / (n + 1)
            raw_w = float(a["width"] + (b["width"] - a["width"]) * frac)
            raw_t = float(a["thickness"] + (b["thickness"] - a["thickness"]) * frac)
            raw_temp = float(_temp_center(a) + (_temp_center(b) - _temp_center(a)) * frac)
            w_cands = _nearest_n(widths, raw_w, 3)
            t_cands = _nearest_n(thicks, raw_t, 5)
            tb_cands = [_virtual_temp_window(raw_temp, temp_range)]
            rem = n - k - 1
            best_v = None
            best_score = None
            pools = [
                (w_cands, t_cands, tb_cands),
                (widths, thicks, [temp_range]),
            ]
            for ww, tt, bb in pools:
                for w in ww:
                    for t in tt:
                        for tb_min, tb_max in bb:
                            v = _make_virtual(start_vidx + k, _txt(a["line"]), float(w), float(t), float(tb_min), float(tb_max), campaign_id, rule, spec_views=spec)
                            if not _hard_direct_step_ok(prev, v, rule):
                                continue
                            if rem == 0:
                                if not _hard_direct_step_ok(v, b, rule):
                                    continue
                            else:
                                if _bridge_need(v, b, rem, rule, strict_virtual_width_levels=bool(strict_virtual_width_levels), spec_views=spec) > rem:
                                    continue
                            score = (
                                2.0 * abs(float(w) - raw_w)
                                + 300.0 * abs(float(t) - raw_t)
                                + abs(((tb_min + tb_max) / 2.0) - raw_temp)
                            )
                            if (best_score is None) or (score < best_score):
                                best_score = score
                                best_v = v
                if best_v is not None:
                    break
            if best_v is None:
                return None
            chain.append(best_v)
            prev = best_v
        if not _hard_direct_step_ok(prev, b, rule):
            return None
        return chain, start_vidx + n

    for n in range(need, max_virtual_chain + 1):
        res = _try_with_n(n, vidx_start)
        if res is not None:
            return res
    raise RuntimeError("virtual step invalid")


__all__ = [
    "DUE_RANK",
    "build_virtual_spec_views",
    "real_reverse_step_within_limit",
    "bridge_step_within_limit",
    "reverse_step_within_applicable_limit",
    "_bridge_need",
    "_bridge_pair",
    "_hard_direct_step_ok",
    "_required_temp_overlap",
    "_temp_overlap_len",
    "_temp_transition_ok",
    "_txt",
    "_width_reverse_virtual_need",
]
