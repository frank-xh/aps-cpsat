import os
import sys
from pathlib import Path

import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from aps_cp_sat.cold_rolling_pipeline import ColdRollingPipeline
from aps_cp_sat.config import ModelConfig, PlannerConfig, RuleConfig, ScoreConfig
from aps_cp_sat.domain.models import ColdRollingRequest
from aps_cp_sat.model.joint_master import _build_line_order_proxy_burden
from aps_cp_sat.model.local_router import _solve_slot_route_with_templates


def _cfg(**model_overrides) -> PlannerConfig:
    return PlannerConfig(
        rule=RuleConfig(),
        score=ScoreConfig(),
        model=ModelConfig(**model_overrides),
    )


def test_strict_template_edge_router_blocks_missing_pairs():
    cfg = _cfg(strict_template_edges=True)
    slot_df = pd.DataFrame(
        [
            {"order_id": "A", "due_rank": 2, "width": 1200.0, "thickness": 1.0},
            {"order_id": "B", "due_rank": 2, "width": 1100.0, "thickness": 1.0},
            {"order_id": "C", "due_rank": 2, "width": 1000.0, "thickness": 1.0},
        ]
    )
    tpl_df = pd.DataFrame(
        [
            {"from_order_id": "A", "to_order_id": "B", "cost": 1.0, "width_smooth_cost": 0.0, "thickness_smooth_cost": 0.0, "temp_margin_cost": 0.0, "cross_group_cost": 0.0, "bridge_count": 0, "logical_reverse_flag": 0},
        ]
    )
    result = _solve_slot_route_with_templates(slot_df, tpl_df, time_limit=1.0, seed=1, cfg=cfg)
    assert result["status"] == "UNROUTABLE_SLOT"
    assert result["sequence"] == []
    assert result["strict_template_edges_enabled"] is True
    assert result["diagnostics"]["zero_in_orders"] >= 1
    assert result["diagnostics"]["zero_out_orders"] >= 1
    assert result["diagnostics"]["top_isolated_orders"]


def test_strict_template_edge_router_routes_only_template_pairs():
    cfg = _cfg(strict_template_edges=True)
    slot_df = pd.DataFrame(
        [
            {"order_id": "A", "due_rank": 2, "width": 1200.0, "thickness": 1.0},
            {"order_id": "B", "due_rank": 2, "width": 1100.0, "thickness": 1.0},
        ]
    )
    tpl_df = pd.DataFrame(
        [
            {"from_order_id": "A", "to_order_id": "B", "cost": 1.0, "width_smooth_cost": 0.0, "thickness_smooth_cost": 0.0, "temp_margin_cost": 0.0, "cross_group_cost": 0.0, "bridge_count": 0, "logical_reverse_flag": 0},
            {"from_order_id": "B", "to_order_id": "A", "cost": 2.0, "width_smooth_cost": 0.0, "thickness_smooth_cost": 0.0, "temp_margin_cost": 0.0, "cross_group_cost": 0.0, "bridge_count": 0, "logical_reverse_flag": 0},
        ]
    )
    result = _solve_slot_route_with_templates(slot_df, tpl_df, time_limit=1.0, seed=1, cfg=cfg)
    assert result["status"] == "ROUTED"
    assert sorted(result["sequence"]) == ["A", "B"]


def test_routing_feasibility_gate_marks_failed_result(monkeypatch, tmp_path: Path):
    cfg = _cfg()
    req = ColdRollingRequest(
        orders_path=Path("orders.xlsx"),
        steel_info_path=Path("steel.xlsx"),
        output_path=tmp_path / "out.xlsx",
        config=cfg,
    )
    raw_schedule = pd.DataFrame(
        [
            {
                "order_id": "A", "source_order_id": "A", "parent_order_id": "A", "lot_id": "A-L01",
                "grade": "DC01", "steel_group": "PC", "line_capability": "dual",
                "width": 1200.0, "thickness": 1.0, "temp_min": 700.0, "temp_max": 760.0, "temp_mean": 730.0,
                "tons": 30.0, "line": "big_roll", "master_slot": 1, "master_seq": 1, "campaign_id_hint": 1, "campaign_seq_hint": 1,
                "selected_bridge_path": "", "selected_template_id": "", "force_break_before": 1, "line_source": "dual", "is_virtual": False,
            }
        ]
    )

    monkeypatch.setattr("aps_cp_sat.cold_rolling_pipeline.prepare_orders_for_model", lambda *args, **kwargs: raw_schedule.copy())
    monkeypatch.setattr("aps_cp_sat.cold_rolling_pipeline.build_transition_templates", lambda *args, **kwargs: {"templates": pd.DataFrame(), "summaries": [], "prune_summaries": []})
    monkeypatch.setattr("aps_cp_sat.cold_rolling_pipeline.solve_master_model", lambda *args, **kwargs: (raw_schedule.copy(), pd.DataFrame(), pd.DataFrame(), {"engine_used": "joint_master", "main_path": "joint_master"}))
    monkeypatch.setattr("aps_cp_sat.cold_rolling_pipeline.decode_solution", lambda result: result)
    monkeypatch.setattr("aps_cp_sat.cold_rolling_pipeline.validate_solution_summary", lambda *args, **kwargs: {})
    monkeypatch.setattr("aps_cp_sat.cold_rolling_pipeline.validate_model_equivalence", lambda *args, **kwargs: {"template_pair_ok": False, "adjacency_rule_ok": False, "bridge_expand_ok": False})
    monkeypatch.setattr("aps_cp_sat.cold_rolling_pipeline.export_schedule_results", lambda **kwargs: None)

    result = ColdRollingPipeline().run(req)
    assert result.engine_meta["routing_feasible"] is False
    assert result.engine_meta["routing_status"] == "ROUTING_INFEASIBLE"
    assert result.engine_meta["result_acceptance_status"] == "FAILED_ROUTING_SEARCH"
    assert "ROUTING_INFEASIBLE_NOT_PRODUCTION_READY" in str(result.output_path)


def test_failed_routing_can_skip_debug_export(monkeypatch, tmp_path: Path):
    cfg = _cfg(export_failed_result_for_debug=False)
    req = ColdRollingRequest(
        orders_path=Path("orders.xlsx"),
        steel_info_path=Path("steel.xlsx"),
        output_path=tmp_path / "out.xlsx",
        config=cfg,
    )
    raw_schedule = pd.DataFrame(
        [
            {
                "order_id": "A", "source_order_id": "A", "parent_order_id": "A", "lot_id": "A-L01",
                "grade": "DC01", "steel_group": "PC", "line_capability": "dual",
                "width": 1200.0, "thickness": 1.0, "temp_min": 700.0, "temp_max": 760.0, "temp_mean": 730.0,
                "tons": 30.0, "line": "big_roll", "master_slot": 1, "master_seq": 1, "campaign_id_hint": 1, "campaign_seq_hint": 1,
                "selected_bridge_path": "", "selected_template_id": "", "force_break_before": 1, "line_source": "dual", "is_virtual": False,
            }
        ]
    )
    export_calls = []

    monkeypatch.setattr("aps_cp_sat.cold_rolling_pipeline.prepare_orders_for_model", lambda *args, **kwargs: raw_schedule.copy())
    monkeypatch.setattr("aps_cp_sat.cold_rolling_pipeline.build_transition_templates", lambda *args, **kwargs: {"templates": pd.DataFrame(), "summaries": [], "prune_summaries": []})
    monkeypatch.setattr("aps_cp_sat.cold_rolling_pipeline.solve_master_model", lambda *args, **kwargs: (raw_schedule.copy(), pd.DataFrame(), pd.DataFrame(), {"engine_used": "joint_master", "main_path": "joint_master"}))
    monkeypatch.setattr("aps_cp_sat.cold_rolling_pipeline.decode_solution", lambda result: result)
    monkeypatch.setattr("aps_cp_sat.cold_rolling_pipeline.validate_solution_summary", lambda *args, **kwargs: {})
    monkeypatch.setattr("aps_cp_sat.cold_rolling_pipeline.validate_model_equivalence", lambda *args, **kwargs: {"template_pair_ok": False, "adjacency_rule_ok": False, "bridge_expand_ok": False})
    monkeypatch.setattr("aps_cp_sat.cold_rolling_pipeline.export_schedule_results", lambda **kwargs: export_calls.append(kwargs))

    result = ColdRollingPipeline().run(req)
    assert result.engine_meta["routing_feasible"] is False
    assert result.engine_meta["result_acceptance_status"] == "FAILED_ROUTING_SEARCH"
    assert result.engine_meta["export_failed_result_for_debug"] is False
    assert result.engine_meta["final_export_performed"] is False
    assert export_calls == []


def test_failed_routing_can_export_analysis_only(monkeypatch, tmp_path: Path):
    cfg = _cfg(export_failed_result_for_debug=False, export_analysis_on_failure=True)
    req = ColdRollingRequest(
        orders_path=Path("orders.xlsx"),
        steel_info_path=Path("steel.xlsx"),
        output_path=tmp_path / "out.xlsx",
        config=cfg,
    )
    raw_schedule = pd.DataFrame(
        [
            {
                "order_id": "A", "source_order_id": "A", "parent_order_id": "A", "lot_id": "A-L01",
                "grade": "DC01", "steel_group": "PC", "line_capability": "dual",
                "width": 1200.0, "thickness": 1.0, "temp_min": 700.0, "temp_max": 760.0, "temp_mean": 730.0,
                "tons": 30.0, "line": "big_roll", "master_slot": 1, "master_seq": 1, "campaign_id_hint": 1, "campaign_seq_hint": 1,
                "selected_bridge_path": "", "selected_template_id": "", "force_break_before": 1, "line_source": "dual", "is_virtual": False,
            }
        ]
    )
    export_calls = []

    monkeypatch.setattr("aps_cp_sat.cold_rolling_pipeline.prepare_orders_for_model", lambda *args, **kwargs: raw_schedule.copy())
    monkeypatch.setattr("aps_cp_sat.cold_rolling_pipeline.build_transition_templates", lambda *args, **kwargs: {"templates": pd.DataFrame(), "summaries": [], "prune_summaries": []})
    monkeypatch.setattr("aps_cp_sat.cold_rolling_pipeline.solve_master_model", lambda *args, **kwargs: (raw_schedule.copy(), pd.DataFrame(), pd.DataFrame(), {"engine_used": "joint_master", "main_path": "joint_master"}))
    monkeypatch.setattr("aps_cp_sat.cold_rolling_pipeline.decode_solution", lambda result: result)
    monkeypatch.setattr("aps_cp_sat.cold_rolling_pipeline.validate_solution_summary", lambda *args, **kwargs: {})
    monkeypatch.setattr("aps_cp_sat.cold_rolling_pipeline.validate_model_equivalence", lambda *args, **kwargs: {"template_pair_ok": False, "adjacency_rule_ok": False, "bridge_expand_ok": False})
    monkeypatch.setattr("aps_cp_sat.cold_rolling_pipeline.export_schedule_results", lambda **kwargs: export_calls.append(kwargs))

    result = ColdRollingPipeline().run(req)
    assert result.engine_meta["routing_feasible"] is False
    assert result.engine_meta["analysis_exported"] is True
    assert result.engine_meta["official_exported"] is False
    assert result.engine_meta["result_usage"] == "ANALYSIS_ONLY"
    assert result.engine_meta["final_export_performed"] is True
    assert "FAILED_ROUTING_ANALYSIS" in str(result.output_path)
    assert len(export_calls) == 1


def test_partial_schedule_with_drops_is_official(monkeypatch, tmp_path: Path):
    cfg = _cfg(
        export_failed_result_for_debug=False,
        export_analysis_on_failure=True,
        max_drop_ratio_for_partial=0.8,
        max_drop_tons_ratio_for_partial=0.8,
        min_scheduled_orders_for_partial=1,
        min_scheduled_tons_for_partial=1.0,
    )
    req = ColdRollingRequest(
        orders_path=Path("orders.xlsx"),
        steel_info_path=Path("steel.xlsx"),
        output_path=tmp_path / "partial.xlsx",
        config=cfg,
    )
    input_orders = pd.DataFrame(
        [
            {
                "order_id": "A", "source_order_id": "A", "parent_order_id": "A", "lot_id": "A-L01",
                "grade": "DC01", "steel_group": "PC", "line_capability": "dual",
                "width": 1200.0, "thickness": 1.0, "temp_min": 700.0, "temp_max": 760.0, "temp_mean": 730.0,
                "tons": 30.0, "line": "big_roll", "master_slot": 1, "master_seq": 1, "campaign_id_hint": 1, "campaign_seq_hint": 1,
                "selected_bridge_path": "", "selected_template_id": "", "selected_edge_type": "DIRECT_EDGE", "force_break_before": 1, "line_source": "dual", "is_virtual": False,
            },
            {
                "order_id": "B", "source_order_id": "B", "parent_order_id": "B", "lot_id": "B-L01",
                "grade": "DC01", "steel_group": "PC", "line_capability": "dual",
                "width": 1180.0, "thickness": 1.0, "temp_min": 700.0, "temp_max": 760.0, "temp_mean": 730.0,
                "tons": 10.0, "line": "big_roll", "master_slot": 1, "master_seq": 2, "campaign_id_hint": 1, "campaign_seq_hint": 2,
                "selected_bridge_path": "", "selected_template_id": "", "selected_edge_type": "DIRECT_EDGE", "force_break_before": 0, "line_source": "dual", "is_virtual": False,
            },
        ]
    )
    raw_schedule = input_orders.iloc[[0]].copy()
    dropped = pd.DataFrame([{"order_id": "B", "tons": 10.0, "drop_reason": "GLOBAL_ISOLATED_ORDER", "line_capability": "dual"}])
    export_calls = []

    monkeypatch.setattr("aps_cp_sat.cold_rolling_pipeline.prepare_orders_for_model", lambda *args, **kwargs: input_orders.copy())
    monkeypatch.setattr("aps_cp_sat.cold_rolling_pipeline.build_transition_templates", lambda *args, **kwargs: {"templates": pd.DataFrame(), "summaries": [], "prune_summaries": []})
    monkeypatch.setattr("aps_cp_sat.cold_rolling_pipeline.solve_master_model", lambda *args, **kwargs: (raw_schedule.copy(), pd.DataFrame(), dropped.copy(), {"engine_used": "joint_master", "main_path": "joint_master"}))
    monkeypatch.setattr("aps_cp_sat.cold_rolling_pipeline.decode_solution", lambda result: result)
    monkeypatch.setattr("aps_cp_sat.cold_rolling_pipeline.validate_solution_summary", lambda *args, **kwargs: {})
    monkeypatch.setattr("aps_cp_sat.cold_rolling_pipeline.validate_model_equivalence", lambda *args, **kwargs: {"template_pair_ok": True, "adjacency_rule_ok": True, "bridge_expand_ok": True})
    monkeypatch.setattr("aps_cp_sat.cold_rolling_pipeline.export_schedule_results", lambda **kwargs: export_calls.append(kwargs))

    result = ColdRollingPipeline().run(req)
    assert result.engine_meta["routing_feasible"] is True
    assert result.engine_meta["result_acceptance_status"] == "PARTIAL_SCHEDULE_WITH_DROPS"
    assert result.engine_meta["result_usage"] == "PARTIAL_OFFICIAL"
    assert result.engine_meta["official_exported"] is True
    assert result.engine_meta["analysis_exported"] is False
    assert result.engine_meta["final_export_performed"] is True
    assert len(export_calls) == 1


def test_partial_schedule_threshold_failure_becomes_analysis_only(monkeypatch, tmp_path: Path):
    cfg = _cfg(
        export_failed_result_for_debug=False,
        export_best_candidate_analysis=True,
        max_drop_ratio_for_partial=0.05,
        max_drop_tons_ratio_for_partial=0.05,
        min_scheduled_orders_for_partial=10,
        min_scheduled_tons_for_partial=100.0,
    )
    req = ColdRollingRequest(
        orders_path=Path("orders.xlsx"),
        steel_info_path=Path("steel.xlsx"),
        output_path=tmp_path / "partial_threshold.xlsx",
        config=cfg,
    )
    input_orders = pd.DataFrame(
        [
            {
                "order_id": "A", "source_order_id": "A", "parent_order_id": "A", "lot_id": "A-L01",
                "grade": "DC01", "steel_group": "PC", "line_capability": "dual",
                "width": 1200.0, "thickness": 1.0, "temp_min": 700.0, "temp_max": 760.0, "temp_mean": 730.0,
                "tons": 30.0, "line": "big_roll", "master_slot": 1, "master_seq": 1, "campaign_id_hint": 1, "campaign_seq_hint": 1,
                "selected_bridge_path": "", "selected_template_id": "", "selected_edge_type": "DIRECT_EDGE", "force_break_before": 1, "line_source": "dual", "is_virtual": False,
            },
            {
                "order_id": "B", "source_order_id": "B", "parent_order_id": "B", "lot_id": "B-L01",
                "grade": "DC01", "steel_group": "PC", "line_capability": "dual",
                "width": 1180.0, "thickness": 1.0, "temp_min": 700.0, "temp_max": 760.0, "temp_mean": 730.0,
                "tons": 10.0, "line": "big_roll", "master_slot": 1, "master_seq": 2, "campaign_id_hint": 1, "campaign_seq_hint": 2,
                "selected_bridge_path": "", "selected_template_id": "", "selected_edge_type": "DIRECT_EDGE", "force_break_before": 0, "line_source": "dual", "is_virtual": False,
            },
        ]
    )
    raw_schedule = input_orders.iloc[[0]].copy()
    dropped = pd.DataFrame([{"order_id": "B", "tons": 10.0, "drop_reason": "GLOBAL_ISOLATED_ORDER", "line_capability": "dual"}])
    export_calls = []

    monkeypatch.setattr("aps_cp_sat.cold_rolling_pipeline.prepare_orders_for_model", lambda *args, **kwargs: input_orders.copy())
    monkeypatch.setattr("aps_cp_sat.cold_rolling_pipeline.build_transition_templates", lambda *args, **kwargs: {"templates": pd.DataFrame(), "summaries": [], "prune_summaries": []})
    monkeypatch.setattr("aps_cp_sat.cold_rolling_pipeline.solve_master_model", lambda *args, **kwargs: (raw_schedule.copy(), pd.DataFrame(), dropped.copy(), {"engine_used": "joint_master", "main_path": "joint_master"}))
    monkeypatch.setattr("aps_cp_sat.cold_rolling_pipeline.decode_solution", lambda result: result)
    monkeypatch.setattr("aps_cp_sat.cold_rolling_pipeline.validate_solution_summary", lambda *args, **kwargs: {})
    monkeypatch.setattr("aps_cp_sat.cold_rolling_pipeline.validate_model_equivalence", lambda *args, **kwargs: {"template_pair_ok": True, "adjacency_rule_ok": True, "bridge_expand_ok": True})
    monkeypatch.setattr("aps_cp_sat.cold_rolling_pipeline.export_schedule_results", lambda **kwargs: export_calls.append(kwargs))

    result = ColdRollingPipeline().run(req)
    assert result.engine_meta["routing_feasible"] is True
    assert result.engine_meta["partial_result_available"] is True
    assert result.engine_meta["partial_acceptance_passed"] is False
    assert result.engine_meta["result_acceptance_status"] == "BEST_SEARCH_CANDIDATE_ANALYSIS"
    assert result.engine_meta["result_usage"] == "ANALYSIS_ONLY"
    assert result.engine_meta["official_exported"] is False
    assert result.engine_meta["analysis_exported"] is True
    assert "BEST_SEARCH_CANDIDATE_ANALYSIS" in str(result.output_path)
    assert len(export_calls) == 1


def test_best_search_candidate_is_exported_as_analysis_only(monkeypatch, tmp_path: Path):
    cfg = _cfg(export_failed_result_for_debug=False, export_analysis_on_failure=False, export_best_candidate_analysis=True)
    req = ColdRollingRequest(
        orders_path=Path("orders.xlsx"),
        steel_info_path=Path("steel.xlsx"),
        output_path=tmp_path / "candidate.xlsx",
        config=cfg,
    )
    candidate_schedule = pd.DataFrame(
        [
            {
                "order_id": "A", "source_order_id": "A", "parent_order_id": "A", "lot_id": "A-L01",
                "grade": "DC01", "steel_group": "PC", "line_capability": "dual",
                "width": 1200.0, "thickness": 1.0, "temp_min": 700.0, "temp_max": 760.0, "temp_mean": 730.0,
                "tons": 30.0, "line": "big_roll", "master_slot": 1, "master_seq": 1, "campaign_id_hint": 1, "campaign_seq_hint": 1,
                "selected_bridge_path": "", "selected_template_id": "", "selected_edge_type": "DIRECT_EDGE", "force_break_before": 1, "line_source": "dual", "is_virtual": False,
            }
        ]
    )
    export_calls = []

    monkeypatch.setattr("aps_cp_sat.cold_rolling_pipeline.prepare_orders_for_model", lambda *args, **kwargs: candidate_schedule.copy())
    monkeypatch.setattr("aps_cp_sat.cold_rolling_pipeline.build_transition_templates", lambda *args, **kwargs: {"templates": pd.DataFrame(), "summaries": [], "prune_summaries": []})
    monkeypatch.setattr(
        "aps_cp_sat.cold_rolling_pipeline.solve_master_model",
        lambda *args, **kwargs: (
            candidate_schedule.copy(),
            pd.DataFrame(),
            pd.DataFrame(),
            {
                "engine_used": "joint_master",
                "main_path": "joint_master",
                "result_acceptance_status": "BEST_SEARCH_CANDIDATE_ANALYSIS",
                "failure_mode": "FAILED_ROUTING_SEARCH",
                "best_candidate_available": True,
                "best_candidate_type": "BEST_SEARCH_CANDIDATE_ANALYSIS",
                "best_candidate_objective": 123.0,
                "best_candidate_search_status": "ROUTING_INFEASIBLE",
                "best_candidate_routing_feasible": False,
                "best_candidate_unroutable_slot_count": 3,
            },
        ),
    )
    monkeypatch.setattr("aps_cp_sat.cold_rolling_pipeline.decode_solution", lambda result: result)
    monkeypatch.setattr("aps_cp_sat.cold_rolling_pipeline.validate_solution_summary", lambda *args, **kwargs: {})
    monkeypatch.setattr("aps_cp_sat.cold_rolling_pipeline.validate_model_equivalence", lambda *args, **kwargs: {"template_pair_ok": False, "adjacency_rule_ok": False, "bridge_expand_ok": False})
    monkeypatch.setattr("aps_cp_sat.cold_rolling_pipeline.export_schedule_results", lambda **kwargs: export_calls.append(kwargs))

    result = ColdRollingPipeline().run(req)
    assert result.engine_meta["result_acceptance_status"] == "BEST_SEARCH_CANDIDATE_ANALYSIS"
    assert result.engine_meta["result_usage"] == "ANALYSIS_ONLY"
    assert result.engine_meta["analysis_exported"] is True
    assert result.engine_meta["official_exported"] is False
    assert result.engine_meta["best_candidate_available"] is True
    assert "BEST_SEARCH_CANDIDATE_ANALYSIS" in str(result.output_path)
    assert len(export_calls) == 1


def test_drop_annotation_preserves_structure_fallback_reasoning():
    orders = pd.DataFrame(
        [
            {
                "order_id": "A",
                "line_capability": "dual",
                "priority": 0,
                "due_rank": 3,
                "tons": 12.0,
            }
        ]
    )
    dropped = pd.DataFrame(
        [
            {
                "order_id": "A",
                "drop_reason": "SLOT_ROUTING_RISK_TOO_HIGH",
                "dominant_drop_reason": "SLOT_ROUTING_RISK_TOO_HIGH",
                "secondary_reasons": "LOW_DEGREE_ORDER,OVERPACKED_SLOT",
                "risk_summary": "slot=big_roll:2|coverage=0.140|pair_gap=60",
                "would_break_slot_if_kept": True,
                "candidate_lines": "big_roll,small_roll",
                "globally_isolated": False,
            }
        ]
    )
    meta = {"unroutable_slot_count": 3, "feasibility_evidence": {"isolated_orders_topn": []}}

    annotated = ColdRollingPipeline._annotate_dropped_orders(orders, dropped, meta)

    assert annotated.iloc[0]["dominant_drop_reason"] == "SLOT_ROUTING_RISK_TOO_HIGH"
    assert "LOW_DEGREE_ORDER" in str(annotated.iloc[0]["secondary_reasons"])
    assert "slot=big_roll:2" in str(annotated.iloc[0]["risk_summary"])
    assert bool(annotated.iloc[0]["would_break_slot_if_kept"]) is True


def test_joint_master_route_risk_signal_changes_with_template_coverage():
    cfg = _cfg()
    orders = pd.DataFrame(
        [
            {"order_id": "A"},
            {"order_id": "B"},
            {"order_id": "C"},
        ]
    )
    full_tpl = pd.DataFrame(
        [
            {"line": "big_roll", "from_order_id": "A", "to_order_id": "B", "virtual_tons": 0.0, "bridge_count": 0, "logical_reverse_flag": 0, "width_delta": 10.0, "cost": 1.0, "width_smooth_cost": 0.0, "thickness_smooth_cost": 0.0, "temp_margin_cost": 0.0, "cross_group_cost": 0.0},
            {"line": "big_roll", "from_order_id": "B", "to_order_id": "C", "virtual_tons": 0.0, "bridge_count": 0, "logical_reverse_flag": 0, "width_delta": 10.0, "cost": 1.0, "width_smooth_cost": 0.0, "thickness_smooth_cost": 0.0, "temp_margin_cost": 0.0, "cross_group_cost": 0.0},
            {"line": "big_roll", "from_order_id": "A", "to_order_id": "C", "virtual_tons": 0.0, "bridge_count": 0, "logical_reverse_flag": 0, "width_delta": 10.0, "cost": 1.0, "width_smooth_cost": 0.0, "thickness_smooth_cost": 0.0, "temp_margin_cost": 0.0, "cross_group_cost": 0.0},
        ]
    )
    sparse_tpl = pd.DataFrame(
        [
            {"line": "big_roll", "from_order_id": "A", "to_order_id": "B", "virtual_tons": 20.0, "bridge_count": 2, "logical_reverse_flag": 1, "width_delta": -80.0, "cost": 50.0, "width_smooth_cost": 0.0, "thickness_smooth_cost": 0.0, "temp_margin_cost": 0.0, "cross_group_cost": 0.0},
        ]
    )
    full_proxy = _build_line_order_proxy_burden(orders, full_tpl, cfg, ["big_roll"])
    sparse_proxy = _build_line_order_proxy_burden(orders, sparse_tpl, cfg, ["big_roll"])
    full_score = sum(v["bridge_cost"] + v["virtual_blocks"] + v["reverse_count"] for v in full_proxy.values())
    sparse_score = sum(v["bridge_cost"] + v["virtual_blocks"] + v["reverse_count"] for v in sparse_proxy.values())
    assert sparse_score > full_score
    assert any(int(v.get("degree_risk", 0)) > 0 for v in sparse_proxy.values())
    assert any(int(v.get("isolated_order_penalty", 0)) > 0 for v in sparse_proxy.values())


def test_strict_template_edge_end_to_end_blocks_false_success(monkeypatch, tmp_path: Path):
    cfg = _cfg(strict_template_edges=True)
    req = ColdRollingRequest(
        orders_path=Path("orders.xlsx"),
        steel_info_path=Path("steel.xlsx"),
        output_path=tmp_path / "strict_out.xlsx",
        config=cfg,
    )
    raw_schedule = pd.DataFrame(
        [
            {
                "order_id": "A", "source_order_id": "A", "parent_order_id": "A", "lot_id": "A-L01",
                "grade": "DC01", "steel_group": "PC", "line_capability": "dual",
                "width": 1200.0, "thickness": 1.0, "temp_min": 700.0, "temp_max": 760.0, "temp_mean": 730.0,
                "tons": 30.0, "line": "big_roll", "master_slot": 1, "master_seq": 1, "campaign_id_hint": 1, "campaign_seq_hint": 1,
                "selected_bridge_path": "", "selected_template_id": "", "force_break_before": 1, "line_source": "dual", "is_virtual": False,
            },
            {
                "order_id": "B", "source_order_id": "B", "parent_order_id": "B", "lot_id": "B-L01",
                "grade": "DC01", "steel_group": "PC", "line_capability": "dual",
                "width": 1100.0, "thickness": 1.0, "temp_min": 700.0, "temp_max": 760.0, "temp_mean": 730.0,
                "tons": 32.0, "line": "big_roll", "master_slot": 1, "master_seq": 2, "campaign_id_hint": 1, "campaign_seq_hint": 2,
                "selected_bridge_path": "", "selected_template_id": "", "force_break_before": 0, "line_source": "dual", "is_virtual": False,
            },
        ]
    )
    monkeypatch.setattr("aps_cp_sat.cold_rolling_pipeline.prepare_orders_for_model", lambda *args, **kwargs: raw_schedule.copy())
    monkeypatch.setattr("aps_cp_sat.cold_rolling_pipeline.build_transition_templates", lambda *args, **kwargs: {"templates": pd.DataFrame(), "summaries": [], "prune_summaries": []})
    monkeypatch.setattr(
        "aps_cp_sat.cold_rolling_pipeline.solve_master_model",
        lambda *args, **kwargs: (
            raw_schedule.copy(),
            pd.DataFrame(),
            pd.DataFrame(),
            {"engine_used": "joint_master", "main_path": "joint_master", "strict_template_edges_enabled": True, "unroutable_slot_count": 1},
        ),
    )
    monkeypatch.setattr("aps_cp_sat.cold_rolling_pipeline.decode_solution", lambda result: result)
    monkeypatch.setattr("aps_cp_sat.cold_rolling_pipeline.validate_solution_summary", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        "aps_cp_sat.cold_rolling_pipeline.validate_model_equivalence",
        lambda *args, **kwargs: {"template_pair_ok": False, "adjacency_rule_ok": False, "bridge_expand_ok": False},
    )
    monkeypatch.setattr("aps_cp_sat.cold_rolling_pipeline.export_schedule_results", lambda **kwargs: None)

    result = ColdRollingPipeline().run(req)
    assert result.engine_meta["result_acceptance_status"] == "FAILED_ROUTING_SEARCH"
    assert result.engine_meta["routing_feasible"] is False


def test_pipeline_returns_failed_routing_result_without_exception(monkeypatch, tmp_path: Path):
    cfg = _cfg(export_failed_result_for_debug=False)
    req = ColdRollingRequest(
        orders_path=Path("orders.xlsx"),
        steel_info_path=Path("steel.xlsx"),
        output_path=tmp_path / "failed.xlsx",
        config=cfg,
    )
    monkeypatch.setattr("aps_cp_sat.cold_rolling_pipeline.prepare_orders_for_model", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr("aps_cp_sat.cold_rolling_pipeline.build_transition_templates", lambda *args, **kwargs: {"templates": pd.DataFrame(), "summaries": [], "prune_summaries": []})
    monkeypatch.setattr(
        "aps_cp_sat.cold_rolling_pipeline.solve_master_model",
        lambda *args, **kwargs: (
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            {
                "engine_used": "joint_master_failed",
                "main_path": "joint_master",
                "result_acceptance_status": "FAILED_ROUTING",
                "routing_feasible": False,
                "routing_status": "ROUTING_INFEASIBLE",
                "template_pair_ok": False,
                "adjacency_rule_ok": False,
                "bridge_expand_ok": False,
                "failure_diagnostics": {"unroutable_slot_count": 3},
                "export_failed_result_for_debug": False,
                "final_export_performed": False,
            },
        ),
    )
    monkeypatch.setattr("aps_cp_sat.cold_rolling_pipeline.decode_solution", lambda result: result)
    monkeypatch.setattr("aps_cp_sat.cold_rolling_pipeline.validate_solution_summary", lambda *args, **kwargs: {})
    monkeypatch.setattr("aps_cp_sat.cold_rolling_pipeline.export_schedule_results", lambda **kwargs: None)

    result = ColdRollingPipeline().run(req)
    assert result.engine_meta["result_acceptance_status"] == "FAILED_ROUTING"
    assert result.engine_meta["routing_feasible"] is False
    assert result.engine_meta["final_export_performed"] is False


def test_pipeline_diagnostics_contains_timing_accounted_ratio(monkeypatch, tmp_path: Path):
    cfg = _cfg(export_failed_result_for_debug=False)
    req = ColdRollingRequest(
        orders_path=Path("orders.xlsx"),
        steel_info_path=Path("steel.xlsx"),
        output_path=tmp_path / "failed.xlsx",
        config=cfg,
    )
    raw_schedule = pd.DataFrame()
    transition_pack = {
        "templates": pd.DataFrame(),
        "summaries": [],
        "prune_summaries": [],
        "build_debug": [
            {
                "line": "__all__",
                "preprocess_seconds": 1.0,
                "line_partition_seconds": 2.0,
                "template_pair_scan_seconds": 3.0,
                "bridge_check_seconds": 4.0,
                "template_prune_seconds": 5.0,
                "transition_pack_build_seconds": 6.0,
                "diagnostics_build_seconds": 1.0,
                "template_build_seconds": 22.0,
            }
        ],
    }

    monkeypatch.setattr("aps_cp_sat.cold_rolling_pipeline.prepare_orders_for_model", lambda *args, **kwargs: raw_schedule.copy())
    monkeypatch.setattr("aps_cp_sat.cold_rolling_pipeline.build_transition_templates", lambda *args, **kwargs: transition_pack)
    monkeypatch.setattr(
        "aps_cp_sat.cold_rolling_pipeline.solve_master_model",
        lambda *args, **kwargs: (
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            {
                "engine_used": "joint_master_failed",
                "main_path": "joint_master",
                "result_acceptance_status": "FAILED_ROUTING_SEARCH",
                "failure_mode": "FAILED_ROUTING_SEARCH",
                "routing_feasible": False,
                "routing_status": "ROUTING_INFEASIBLE",
                "template_pair_ok": False,
                "adjacency_rule_ok": False,
                "bridge_expand_ok": False,
                "failure_diagnostics": {"unroutable_slot_count": 3},
                "export_failed_result_for_debug": False,
                "final_export_performed": False,
                "template_build_seconds": 22.0,
                "joint_master_seconds": 10.0,
                "local_router_seconds": 5.0,
                "fallback_total_seconds": 0.0,
                "total_run_seconds": 40.0,
            },
        ),
    )
    monkeypatch.setattr("aps_cp_sat.cold_rolling_pipeline.decode_solution", lambda result: result)
    monkeypatch.setattr("aps_cp_sat.cold_rolling_pipeline.validate_solution_summary", lambda *args, **kwargs: {})
    monkeypatch.setattr("aps_cp_sat.cold_rolling_pipeline.export_schedule_results", lambda **kwargs: None)

    result = ColdRollingPipeline().run(req)
    diagnostics = ColdRollingPipeline()._build_run_diagnostics(raw_schedule, transition_pack, result)
    assert "timing_accounted_ratio" in diagnostics["fallback"]
    assert "timing_gap_detected" in diagnostics["fallback"]
