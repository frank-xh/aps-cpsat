from __future__ import annotations

from typing import Tuple

from aps_cp_sat.config import RuleConfig
from aps_cp_sat.transition.bridge_rules import _temp_overlap_len


def width_transition_metrics(
    prev_width: float,
    cur_width: float,
    rule: RuleConfig,
    prev_is_virtual: bool = False,
    is_virtual: bool = False,
) -> Tuple[bool, float, bool, bool]:
    width_drop = float(prev_width) - float(cur_width)
    narrow_to_wide = float(cur_width) > float(prev_width)
    width_rise = max(0.0, float(cur_width) - float(prev_width))
    reverse_limit = float(rule.virtual_reverse_attach_max_mm) if bool(is_virtual) else float(rule.real_reverse_step_max_mm)
    physical_reverse_violation = bool(narrow_to_wide and (width_rise > reverse_limit))
    width_violation = bool((width_drop > float(rule.max_width_drop)) or physical_reverse_violation)
    width_over = max(0.0, width_drop - float(rule.max_width_drop)) + max(0.0, width_rise - reverse_limit)
    return width_violation, width_over, narrow_to_wide, physical_reverse_violation


def thickness_transition_metrics(prev_thickness: float, cur_thickness: float) -> Tuple[bool, float]:
    diff = abs(float(cur_thickness) - float(prev_thickness))
    if float(prev_thickness) > 0.8:
        limit = abs(float(prev_thickness)) * 0.30
    elif float(prev_thickness) >= 0.6:
        limit = 0.2
    else:
        limit = 0.1
    return bool(diff > limit), max(0.0, diff - limit)


def temperature_overlap_metrics(
    prev_temp_min: float,
    prev_temp_max: float,
    temp_min: float,
    temp_max: float,
    required_overlap: float,
) -> Tuple[float, bool, float]:
    overlap = _temp_overlap_len(float(prev_temp_min), float(prev_temp_max), float(temp_min), float(temp_max))
    shortage = max(0.0, float(required_overlap) - overlap)
    return overlap, bool(overlap < float(required_overlap)), shortage


def reverse_width_flags(narrow_to_wide: bool, prev_is_virtual: bool, is_virtual: bool) -> bool:
    # Count one logical reverse episode at the beginning of a reverse chain.
    return bool(narrow_to_wide and (not bool(prev_is_virtual)))


def max_virtual_chain_len(is_virtual_values: list[bool]) -> int:
    max_len = 0
    cur = 0
    for item in is_virtual_values:
        if bool(item):
            cur += 1
            max_len = max(max_len, cur)
        else:
            cur = 0
    return int(max_len)
