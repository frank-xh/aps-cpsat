import os
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from aps_cp_sat.config import build_default_solve_config
from aps_cp_sat.config import ModelConfig, PlannerConfig, RuleConfig, ScoreConfig
from aps_cp_sat.decode import decode_solution
from aps_cp_sat.domain.models import ColdRollingRequest, ColdRollingResult
from aps_cp_sat.io import export_schedule_results
from aps_cp_sat.model.feasibility_evidence import build_feasibility_evidence
from aps_cp_sat.model import master as master_module
from aps_cp_sat.model.joint_master import _run_global_joint_model
from aps_cp_sat.transition.bridge_rules import _make_virtual, _width_reverse_virtual_need, build_virtual_spec_views
from aps_cp_sat.validate import validate_model_equivalence, validate_solution_summary


def _raw_master_plan():
    return pd.DataFrame(
        [
            {
                "order_id": "A",
                "source_order_id": "A",
                "parent_order_id": "A",
                "lot_id": "A-L01",
                "grade": "DC01",
                "steel_group": "PC",
                "line_capability": "dual",
                "width": 1200.0,
                "thickness": 1.0,
                "temp_min": 700.0,
                "temp_max": 760.0,
                "temp_mean": 730.0,
                "tons": 30.0,
                "line": "big_roll",
                "master_slot": 1,
                "master_seq": 1,
                "campaign_id_hint": 1,
                "campaign_seq_hint": 1,
                "selected_bridge_path": "W1000|T1.0|TMP[600,620]",
                "selected_template_id": "TPL-A-B",
                "force_break_before": 1,
                "line_source": "dual",
                "is_virtual": False,
            },
            {
                "order_id": "B",
                "source_order_id": "B",
                "parent_order_id": "B",
                "lot_id": "B-L01",
                "grade": "DC01",
                "steel_group": "PC",
                "line_capability": "dual",
                "width": 1000.0,
                "thickness": 1.0,
                "temp_min": 600.0,
                "temp_max": 640.0,
                "temp_mean": 620.0,
                "tons": 31.0,
                "line": "big_roll",
                "master_slot": 1,
                "master_seq": 2,
                "campaign_id_hint": 1,
                "campaign_seq_hint": 2,
                "selected_bridge_path": "",
                "selected_template_id": "",
                "force_break_before": 0,
                "line_source": "dual",
                "is_virtual": False,
            },
        ]
    )


def _final_schedule():
    return pd.DataFrame(
        [
            {
                "global_seq": 1,
                "line_name": "大辊线",
                "line": "big_roll",
                "line_seq": 1,
                "campaign_id": 1,
                "campaign_seq": 1,
                "campaign_real_seq": 1,
                "order_id": "A",
                "source_order_id": "A",
                "parent_order_id": "A",
                "lot_id": "A-L01",
                "is_virtual": False,
                "grade": "DC01",
                "steel_group": "PC",
                "width": 1200.0,
                "thickness": 1.0,
                "temp_min": 700.0,
                "temp_max": 760.0,
                "temp_mean": 730.0,
                "tons": 30.0,
                "width_jump_violation": False,
                "thickness_violation": False,
                "temp_conflict": False,
                "non_pc_direct_switch": False,
            }
        ]
    )


def test_master_path_selection_joint_master(monkeypatch):
    cfg = build_default_solve_config(validation_mode=False, production_compatibility_mode=False)
    req = ColdRollingRequest(
        orders_path=Path("orders.xlsx"),
        steel_info_path=Path("steel.xlsx"),
        output_path=Path("out.xlsx"),
        config=cfg,
    )
    orders = pd.DataFrame(
        [
            {"order_id": "A", "tons": 30.0, "line_source": "dual"},
            {"order_id": "B", "tons": 31.0, "line_source": "dual"},
        ]
    )

    def fake_joint(*args, **kwargs):
        return {
            "status": "FEASIBLE",
            "plan_df": pd.DataFrame(
                [
                    {"row_idx": 0, "assigned_line": "big_roll", "assigned_slot": 1, "master_seq": 1, "campaign_id_hint": 1, "campaign_seq_hint": 1, "selected_template_id": "", "selected_bridge_path": "", "force_break_before": 1, "is_unassigned": 0},
                    {"row_idx": 1, "assigned_line": "big_roll", "assigned_slot": 1, "master_seq": 2, "campaign_id_hint": 1, "campaign_seq_hint": 2, "selected_template_id": "", "selected_bridge_path": "", "force_break_before": 0, "is_unassigned": 0},
                ]
            ),
            "dropped_df": pd.DataFrame(),
            "objective": 1.0,
            "assigned_count": 2,
            "unassigned_count": 0,
            "total_virtual_blocks": 0,
            "global_ratio_over": 0,
            "low_slot_count": 0,
            "ultra_low_slot_count": 0,
            "used_local_routing": False,
        }

    monkeypatch.setattr(master_module, "_run_global_joint_model", fake_joint)
    monkeypatch.setattr(master_module, "run_legacy_schedule", lambda **kwargs: (_ for _ in ()).throw(AssertionError("legacy should not run")))
    planned_df, rounds_df, dropped_df, meta = master_module.solve_master_model(req, transition_pack={}, orders_df=orders)
    assert not planned_df.empty
    assert dropped_df.empty
    assert meta["main_path"] == "joint_master"
    assert meta["master_entry"] == "_run_global_joint_model"


def test_no_post_solve_mutation_decode_validate_export(tmp_path: Path):
    cfg = build_default_solve_config(validation_mode=False, production_compatibility_mode=False)

    raw_df = _raw_master_plan()
    raw_before = raw_df.copy(deep=True)
    raw = ColdRollingResult(schedule_df=raw_df, rounds_df=pd.DataFrame(), output_path=tmp_path / "a.xlsx", dropped_df=pd.DataFrame(), config=cfg)
    decoded = decode_solution(raw)
    assert raw.schedule_df.equals(raw_before)
    assert not decoded.schedule_df.empty

    final_df = decoded.schedule_df.copy(deep=True)
    final_before = final_df.copy(deep=True)
    result = ColdRollingResult(schedule_df=final_df, rounds_df=decoded.rounds_df, output_path=tmp_path / "b.xlsx", dropped_df=decoded.dropped_df, config=cfg)
    _ = validate_solution_summary(result, cfg.rule)
    _ = validate_model_equivalence(result.schedule_df, None)
    assert result.schedule_df.equals(final_before)

    export_df = _final_schedule()
    export_before = export_df.copy(deep=True)
    rounds_df = pd.DataFrame([{"round": 1, "line": "big_roll", "rows_total": 1, "virtual_cnt": 0, "dropped_cnt": 0}])
    rounds_before = rounds_df.copy(deep=True)
    export_schedule_results(
        final_df=export_df,
        rounds_df=rounds_df,
        dropped_df=pd.DataFrame(),
        output_path=str(tmp_path / "export.xlsx"),
        input_order_count=1,
        rule=cfg.rule,
        engine_used="joint_master",
        failure_diagnostics={"fallback": {"main_path": "joint_master", "used_local_routing": False, "bridge_modeling": "template_based"}},
    )
    assert export_df.equals(export_before)
    assert rounds_df.equals(rounds_before)


def test_template_bridge_modeling_guard():
    master_src = Path(__file__).resolve().parent.parent / "src" / "aps_cp_sat" / "model" / "master.py"
    decode_src = Path(__file__).resolve().parent.parent / "src" / "aps_cp_sat" / "decode" / "joint_solution_decoder.py"
    text_master = master_src.read_text(encoding="utf-8")
    text_decode = decode_src.read_text(encoding="utf-8")
    assert "template_based" in text_master
    assert "VIRTUAL-" not in text_master
    assert "VIRTUAL-" in text_decode


def test_bridge_rules_config_propagation():
    base_rule = RuleConfig()
    custom_rule = RuleConfig(
        virtual_width_levels=(1000.0, 1100.0, 1300.0),
        virtual_tons=33.0,
        virtual_reverse_attach_max_mm=120.0,
    )
    spec = build_virtual_spec_views(custom_rule)
    assert spec["widths"] == [1000.0, 1100.0, 1300.0]
    need_custom = _width_reverse_virtual_need(1000.0, 1100.0, custom_rule, strict_virtual_width_levels=False)
    assert need_custom == 1
    virtual = _make_virtual(1, "big_roll", 1100.0, 1.0, 700.0, 720.0, 1, custom_rule)
    assert float(virtual["tons"]) == 33.0
    narrow_rule = RuleConfig(
        virtual_width_levels=base_rule.virtual_width_levels,
        virtual_tons=base_rule.virtual_tons,
        virtual_reverse_attach_max_mm=40.0,
    )
    need_narrow = _width_reverse_virtual_need(1000.0, 1300.0, narrow_rule, strict_virtual_width_levels=False)
    assert need_narrow > base_rule.max_virtual_chain


def test_no_default_rule_config_in_production_path():
    root = Path(__file__).resolve().parent.parent / "src" / "aps_cp_sat"
    planner_text = (root / "config" / "planner_config.py").read_text(encoding="utf-8")
    validator_text = (root / "validate" / "solution_validator.py").read_text(encoding="utf-8")
    pipeline_text = (root / "cold_rolling_pipeline.py").read_text(encoding="utf-8")
    legacy_text = (root / "cold_rolling_scheduler.py").read_text(encoding="utf-8")

    assert "field(default_factory=RuleConfig)" in planner_text
    assert "rule = rule or _COMPAT_RULE" in validator_text
    assert "validate_solution_summary(result, result.config.rule)" in pipeline_text
    assert "LEGACY ONLY / NOT FOR PRODUCTION PATH" in legacy_text


def test_joint_master_objective_coverage():
    cfg = PlannerConfig(
        rule=RuleConfig(),
        score=ScoreConfig(),
        model=ModelConfig(min_campaign_slots=1, max_campaign_slots=2, min_real_schedule_ratio=0.0),
    )
    orders = pd.DataFrame(
        [
            {"order_id": "A", "tons": 30.0, "line_capability": "dual", "width": 1200.0, "thickness": 1.0, "temp_min": 700.0, "temp_max": 760.0, "steel_group": "PC", "due_rank": 2},
            {"order_id": "B", "tons": 31.0, "line_capability": "dual", "width": 1000.0, "thickness": 1.0, "temp_min": 700.0, "temp_max": 760.0, "steel_group": "PC", "due_rank": 2},
        ]
    )
    tpl = pd.DataFrame(
        [
            {
                "line": "big_roll",
                "from_order_id": "A",
                "to_order_id": "B",
                "template_id": "T1",
                "bridge_path": "W1100|T1.0|TMP[700,720]",
                "bridge_count": 1,
                "virtual_tons": 20.0,
                "width_delta": -100.0,
                "logical_reverse_flag": 1,
                "width_smooth_cost": 0.0,
                "thickness_smooth_cost": 0.0,
                "temp_margin_cost": 0.0,
                "cross_group_cost": 0.0,
                "cost": 10.0,
            },
            {
                "line": "small_roll",
                "from_order_id": "A",
                "to_order_id": "B",
                "template_id": "T2",
                "bridge_path": "W1100|T1.0|TMP[700,720]",
                "bridge_count": 2,
                "virtual_tons": 40.0,
                "width_delta": -100.0,
                "logical_reverse_flag": 1,
                "width_smooth_cost": 0.0,
                "thickness_smooth_cost": 0.0,
                "temp_margin_cost": 0.0,
                "cross_group_cost": 0.0,
                "cost": 20.0,
            },
        ]
    )
    joint = _run_global_joint_model(orders, {"templates": tpl}, cfg, time_scale=0.1, random_seed=2027)
    assert "estimated_virtual_blocks" in joint
    assert "estimated_virtual_ton10" in joint
    assert "estimated_global_ratio_over" in joint
    assert "estimated_reverse_count" in joint
    assert "estimated_reverse_rise" in joint


def test_feasibility_evidence_detects_strong_signal():
    cfg = PlannerConfig(
        rule=RuleConfig(campaign_ton_max=100.0),
        score=ScoreConfig(),
        model=ModelConfig(max_campaign_slots=1, big_roll_slot_hard_order_cap=1, small_roll_slot_hard_order_cap=1),
    )
    orders = pd.DataFrame(
        [
            {"order_id": "A", "line_capability": "big_only", "tons": 80.0},
            {"order_id": "B", "line_capability": "big_only", "tons": 80.0},
            {"order_id": "C", "line_capability": "small_only", "tons": 80.0},
        ]
    )
    tpl = pd.DataFrame(
        [
            {"line": "big_roll", "from_order_id": "A", "to_order_id": "B"},
        ]
    )
    pack = {
        "templates": tpl,
        "summaries": [],
    }
    evidence = build_feasibility_evidence(orders, pack, cfg)
    assert evidence["evidence_level"] == "STRONG_INFEASIBLE_SIGNAL"
    assert evidence["top_infeasibility_signals"]
    summary = evidence["feasibility_evidence_summary"]
    assert "slot_safe_lower_bound" in summary
    assert "current_slot_cap" in summary
    assert "globally_isolated_orders" in summary
    assert "theoretical_min_slots_by_line" in summary


def test_validate_summary_reports_virtual_and_reverse_audit_fields():
    cfg = PlannerConfig()
    df = pd.DataFrame(
        [
            {
                "line": "big_roll",
                "campaign_id": 1,
                "order_id": "A",
                "is_virtual": False,
                "tons": 20.0,
                "width": 1200.0,
                "thickness": 1.0,
                "temp_min": 700.0,
                "temp_max": 760.0,
                "direct_reverse_step_violation": False,
                "virtual_attach_reverse_violation": False,
                "bridge_reverse_step_flag": False,
                "logical_reverse_cnt_campaign": 6,
            },
            {
                "line": "big_roll",
                "campaign_id": 1,
                "order_id": "VIRTUAL-00001",
                "is_virtual": True,
                "tons": 20.0,
                "width": 1250.0,
                "thickness": 1.2,
                "temp_min": 600.0,
                "temp_max": 900.0,
                "direct_reverse_step_violation": False,
                "virtual_attach_reverse_violation": True,
                "bridge_reverse_step_flag": True,
                "logical_reverse_cnt_campaign": 6,
            },
            {
                "line": "big_roll",
                "campaign_id": 1,
                "order_id": "B",
                "is_virtual": False,
                "tons": 21.0,
                "width": 1300.0,
                "thickness": 1.0,
                "temp_min": 700.0,
                "temp_max": 760.0,
                "direct_reverse_step_violation": True,
                "virtual_attach_reverse_violation": False,
                "bridge_reverse_step_flag": False,
                "logical_reverse_cnt_campaign": 6,
            },
        ]
    )
    result = ColdRollingResult(schedule_df=df, rounds_df=pd.DataFrame(), output_path=Path("out.xlsx"), dropped_df=pd.DataFrame(), config=cfg)
    summary = validate_solution_summary(result, cfg.rule)
    assert summary["virtual_widths_used"] == [1250.0]
    assert summary["virtual_thicknesses_used"] == [1.2]
    assert summary["virtual_temp_min_used"] == 600.0
    assert summary["virtual_temp_max_used"] == 900.0
    assert summary["invalid_virtual_spec_count"] == 0
    assert summary["direct_reverse_step_violation_count"] == 1
    assert summary["virtual_attach_reverse_violation_count"] == 1
    assert summary["period_reverse_count"] == 6
    assert summary["period_reverse_count_violation_count"] == 1
    assert summary["reverse_count_definition"] == "logical_reverse_per_campaign"
