import os
import sys

import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from aps_cp_sat.config import PlannerConfig
from aps_cp_sat.config.rule_config import RuleConfig
from aps_cp_sat.transition import build_transition_templates
from aps_cp_sat.transition import template_builder as template_builder_module
from aps_cp_sat.transition.bridge_rules import (
    _bridge_need,
    _bridge_pair,
    bridge_step_within_limit,
    build_virtual_spec_views,
    real_reverse_step_within_limit,
    reverse_step_within_applicable_limit,
)


def test_transition_pack_contains_prune_summaries():
    df = pd.DataFrame(
        [
            {"order_id": "A", "line_capability": "dual", "width": 1200.0, "thickness": 1.0, "temp_min": 700.0, "temp_max": 760.0, "tons": 20.0, "steel_group": "PC", "due_rank": 2},
            {"order_id": "B", "line_capability": "dual", "width": 1100.0, "thickness": 1.0, "temp_min": 700.0, "temp_max": 760.0, "tons": 20.0, "steel_group": "PC", "due_rank": 2},
        ]
    )
    pack = build_transition_templates(df, PlannerConfig())
    assert "templates" in pack
    assert "summaries" in pack
    assert "prune_summaries" in pack
    assert "build_debug" in pack
    assert len(pack["prune_summaries"]) == 2


def test_spec_views_reused_in_template_builder(monkeypatch):
    cfg = PlannerConfig()
    df = pd.DataFrame(
        [
            {"order_id": "A", "line_capability": "big_only", "width": 1200.0, "thickness": 1.0, "temp_min": 700.0, "temp_max": 760.0, "tons": 20.0, "steel_group": "PC", "due_rank": 2},
            {"order_id": "B", "line_capability": "big_only", "width": 1100.0, "thickness": 1.0, "temp_min": 700.0, "temp_max": 760.0, "tons": 20.0, "steel_group": "PC", "due_rank": 2},
            {"order_id": "C", "line_capability": "big_only", "width": 1000.0, "thickness": 1.0, "temp_min": 700.0, "temp_max": 760.0, "tons": 20.0, "steel_group": "PC", "due_rank": 2},
        ]
    )
    call_count = {"n": 0}
    real_build = template_builder_module.build_virtual_spec_views

    def spy(rule):
        call_count["n"] += 1
        return real_build(rule)

    monkeypatch.setattr(template_builder_module, "build_virtual_spec_views", spy)
    pack = build_transition_templates(df, cfg)
    assert call_count["n"] == 1
    debug = {item["line"]: item for item in pack["build_debug"]}
    assert debug["big_roll"]["spec_views_reused"] is True
    assert debug["big_roll"]["spec_views_build_count"] == 1
    assert debug["small_roll"]["spec_views_build_count"] == 0
    assert "template_pair_scan_seconds" in debug["big_roll"]
    assert "__all__" in debug
    assert "bridge_check_seconds" in debug["__all__"]
    assert "template_prune_seconds" in debug["__all__"]


def test_template_build_debug_contains_full_timing_breakdown():
    df = pd.DataFrame(
        [
            {"order_id": "A", "line_capability": "big_only", "width": 1200.0, "thickness": 1.0, "temp_min": 700.0, "temp_max": 760.0, "tons": 20.0, "steel_group": "PC", "due_rank": 2},
            {"order_id": "B", "line_capability": "big_only", "width": 1100.0, "thickness": 1.0, "temp_min": 700.0, "temp_max": 760.0, "tons": 20.0, "steel_group": "PC", "due_rank": 2},
        ]
    )
    pack = build_transition_templates(df, PlannerConfig())
    debug = {item["line"]: item for item in pack["build_debug"]}
    total = debug["__all__"]
    assert "preprocess_seconds" in total
    assert "line_partition_seconds" in total
    assert "template_pair_scan_seconds" in total
    assert "bridge_check_seconds" in total
    assert "template_prune_seconds" in total
    assert "transition_pack_build_seconds" in total
    assert "diagnostics_build_seconds" in total
    assert "template_build_seconds" in total


def test_bridge_rules_use_20_for_real_and_250_for_virtual_steps():
    cfg = PlannerConfig()
    rule = cfg.rule
    spec = build_virtual_spec_views(rule)
    a = {"order_id": "A", "line": "big_roll", "width": 1000.0, "thickness": 1.0, "temp_min": 700.0, "temp_max": 760.0, "steel_group": "PC", "is_virtual": False}
    b = {"order_id": "B", "line": "big_roll", "width": 1100.0, "thickness": 1.0, "temp_min": 700.0, "temp_max": 760.0, "steel_group": "PC", "is_virtual": False}
    v = {"order_id": "V", "line": "big_roll", "width": 1250.0, "thickness": 1.0, "temp_min": 700.0, "temp_max": 760.0, "steel_group": "PC", "is_virtual": True}
    assert real_reverse_step_within_limit(a, b, rule) is False
    assert bridge_step_within_limit(a, v, rule) is True
    assert reverse_step_within_applicable_limit(a, b, rule) is False
    assert reverse_step_within_applicable_limit(a, v, rule) is True
    need = _bridge_need(a, b, cfg.model.max_virtual_chain, rule, spec_views=spec)
    assert 1 <= need <= cfg.model.max_virtual_chain


def test_virtual_bridge_only_uses_allowed_discrete_thickness_and_widths():
    cfg = PlannerConfig()
    rule = cfg.rule
    spec = build_virtual_spec_views(rule)
    a = {"order_id": "A", "line": "big_roll", "width": 1000.0, "thickness": 1.0, "temp_min": 700.0, "temp_max": 760.0, "steel_group": "PC", "is_virtual": False}
    b = {"order_id": "B", "line": "big_roll", "width": 1180.0, "thickness": 1.0, "temp_min": 700.0, "temp_max": 760.0, "steel_group": "PC", "is_virtual": False}
    bridge_path, _ = _bridge_pair(a, b, cfg.model.max_virtual_chain, 1, 1, rule, spec_views=spec)
    assert bridge_path
    assert all(float(v["width"]) in set(rule.virtual_width_levels) for v in bridge_path)
    assert all(float(v["thickness"]) in set(rule.virtual_thickness_levels) for v in bridge_path)
    assert all(float(v["temp_min"]) == float(rule.virtual_temp_min) for v in bridge_path)
    assert all(float(v["temp_max"]) == float(rule.virtual_temp_max) for v in bridge_path)


def test_bridge_chain_is_capped_at_five_virtual_steps():
    cfg = PlannerConfig()
    assert cfg.model.max_virtual_chain == 5
    assert cfg.rule.max_virtual_chain == 5


def test_virtual_attach_reverse_rule_applies_only_when_right_side_is_virtual():
    rule = RuleConfig()
    left = {"width": 1000.0, "is_virtual": False}
    right_virtual = {"width": 1240.0, "is_virtual": True}
    right_real = {"width": 1240.0, "is_virtual": False}
    assert reverse_step_within_applicable_limit(left, right_virtual, rule) is True
    assert reverse_step_within_applicable_limit(left, right_real, rule) is False
