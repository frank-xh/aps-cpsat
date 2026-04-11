import importlib.util
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from aps_cp_sat.cold_rolling_pipeline import ColdRollingPipeline
from aps_cp_sat.config import PlannerConfig, build_default_solve_config, build_profile_config
from aps_cp_sat.decode import decode_solution
from aps_cp_sat.domain.models import ColdRollingResult
from aps_cp_sat.model.master import _effective_global_prune_cap, _semantic_fallback_configs
from aps_cp_sat.model import master as master_module
from aps_cp_sat.transition import build_transition_templates


def test_no_default_constraint_config_in_main_path():
    roots = [
        Path(__file__).resolve().parent.parent / "src" / "aps_cp_sat" / "transition" / "templates.py",
        Path(__file__).resolve().parent.parent / "src" / "aps_cp_sat" / "validate" / "solution_validator.py",
        Path(__file__).resolve().parent.parent / "src" / "aps_cp_sat" / "model" / "master.py",
        Path(__file__).resolve().parent.parent / "src" / "aps_cp_sat" / "decode" / "joint_solution_decoder.py",
    ]
    for p in roots:
        txt = p.read_text(encoding="utf-8")
        assert "DEFAULT_CONSTRAINT_CONFIG" not in txt


def test_transition_does_not_import_legacy_scheduler_for_bridge_rules():
    p = Path(__file__).resolve().parent.parent / "src" / "aps_cp_sat" / "transition" / "templates.py"
    txt = p.read_text(encoding="utf-8")
    assert "from aps_cp_sat.cold_rolling_scheduler import (" not in txt
    assert "from aps_cp_sat.transition.bridge_rules import (" in txt


def test_io_readers_do_not_import_legacy_scheduler():
    p = Path(__file__).resolve().parent.parent / "src" / "aps_cp_sat" / "io" / "readers.py"
    txt = p.read_text(encoding="utf-8")
    assert "from aps_cp_sat.cold_rolling_scheduler import" not in txt
    assert "from aps_cp_sat.preprocess.grade_catalog import MergedGradeRuleCatalog" in txt
    assert "from aps_cp_sat.preprocess.order_preparation import prepare_orders" in txt


def test_new_config_is_default_source():
    cfg = build_default_solve_config(validation_mode=False, production_compatibility_mode=False)
    assert hasattr(cfg, "rule")
    assert hasattr(cfg, "model")
    assert hasattr(cfg, "score")
    assert cfg.rule.campaign_ton_min == 700.0
    assert cfg.model.max_routes_per_slot == 5
    assert cfg.score.ton_under >= 1


def test_legacy_fallback_disabled_by_default():
    cfg_normal = build_default_solve_config(validation_mode=False, production_compatibility_mode=False)
    assert cfg_normal.allow_fallback is False
    assert cfg_normal.allow_legacy_fallback is False

    cfg_validation = build_default_solve_config(validation_mode=True, production_compatibility_mode=False)
    assert cfg_validation.allow_fallback is False
    assert cfg_validation.allow_legacy_fallback is False

    cfg_compat = build_default_solve_config(validation_mode=False, production_compatibility_mode=True)
    assert cfg_compat.allow_legacy_fallback is True


def test_decode_solution_is_single_materialization_entry():
    cfg = build_default_solve_config(validation_mode=True, production_compatibility_mode=False)
    planned = pd.DataFrame(
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
                "line_name": "大辊线",
                "master_slot": 1,
                "master_seq": 1,
                "campaign_id_hint": 1001,
                "campaign_seq_hint": 1,
                "selected_bridge_path": "W1000|T1.0|TMP[600,620]",
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
                "line_name": "大辊线",
                "master_slot": 1,
                "master_seq": 2,
                "campaign_id_hint": 1001,
                "campaign_seq_hint": 2,
                "selected_bridge_path": "",
                "line_source": "dual",
                "is_virtual": False,
            },
        ]
    )

    raw = ColdRollingResult(
        schedule_df=planned,
        rounds_df=pd.DataFrame(),
        output_path=Path("out.xlsx"),
        dropped_df=pd.DataFrame(),
        config=cfg,
    )
    decoded = decode_solution(raw)
    assert not decoded.schedule_df.empty
    assert "campaign_id" in decoded.schedule_df.columns
    assert "is_virtual" in decoded.schedule_df.columns
    assert int(decoded.schedule_df["is_virtual"].sum()) >= 1
    assert not decoded.rounds_df.empty


def test_semantic_fallback_relaxes_prune():
    base = build_default_solve_config(validation_mode=False, production_compatibility_mode=False)
    cfg = PlannerConfig(
        rule=base.rule,
        score=base.score,
        model=base.model.__class__(
            **{
                **base.model.__dict__,
                "global_prune_max_pairs_per_from": 4,
                "time_limit_seconds": 120.0,
                "template_top_k": 40,
                "max_routes_per_slot": 5,
            }
        ),
    )
    fallbacks = _semantic_fallback_configs(cfg)
    assert [fcfg.global_prune_max_pairs_per_from for fcfg in fallbacks[:2]] == [6, 8]
    assert len(fallbacks) == 2
    assert all(float(fcfg.time_limit_seconds) > float(cfg.time_limit_seconds) for fcfg in fallbacks)
    assert all(int(fcfg.max_routes_per_slot) > int(cfg.max_routes_per_slot) for fcfg in fallbacks)


def test_zero_or_none_disables_global_prune():
    base = build_default_solve_config(validation_mode=False, production_compatibility_mode=False)
    cfg_zero = PlannerConfig(
        rule=base.rule,
        score=base.score,
        model=base.model.__class__(**{**base.model.__dict__, "global_prune_max_pairs_per_from": 0}),
    )
    cfg_none = PlannerConfig(
        rule=base.rule,
        score=base.score,
        model=base.model.__class__(**{**base.model.__dict__, "global_prune_max_pairs_per_from": None}),
    )
    cfg_on = PlannerConfig(
        rule=base.rule,
        score=base.score,
        model=base.model.__class__(**{**base.model.__dict__, "global_prune_max_pairs_per_from": 6}),
    )
    assert _effective_global_prune_cap(cfg_zero) is None
    assert _effective_global_prune_cap(cfg_none) is None
    assert _effective_global_prune_cap(cfg_on) == 6


def test_example_runner_default_is_non_strict():
    example_path = Path(__file__).resolve().parent.parent / "examples" / "run_cold_rolling_schedule.py"
    spec = importlib.util.spec_from_file_location("example_runner", example_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg = module.build_example_config(strict=False)
    strict_cfg = module.build_example_config(strict=True)
    assert cfg.model.profile_name == "production_search"
    assert cfg.allow_fallback is True
    assert cfg.allow_legacy_fallback is False
    assert strict_cfg.allow_fallback is False
    assert strict_cfg.allow_legacy_fallback is False


def test_template_graph_health_precheck_autorelax(monkeypatch):
    cfg = build_profile_config("feasibility", validation_mode=False, production_compatibility_mode=False)
    req = type(
        "Req",
        (),
        {
            "orders_path": Path("orders.xlsx"),
            "steel_info_path": Path("steel.xlsx"),
            "output_path": Path("out.xlsx"),
            "config": cfg,
        },
    )()
    orders = pd.DataFrame(
        [
            {"order_id": "A", "tons": 30.0, "line_source": "dual", "line_capability": "dual"},
            {"order_id": "B", "tons": 31.0, "line_source": "dual", "line_capability": "dual"},
        ]
    )
    health_calls = [{"template_graph_health": "SPARSE"}, {"template_graph_health": "HEALTHY"}]
    rebuild_calls = []

    def fake_health(*args, **kwargs):
        return health_calls.pop(0) if health_calls else {"template_graph_health": "HEALTHY"}

    def fake_rebuild(orders_df, relaxed_cfg):
        rebuild_calls.append(
            (
                relaxed_cfg.model.global_prune_max_pairs_per_from,
                relaxed_cfg.model.template_top_k,
                relaxed_cfg.model.max_routes_per_slot,
            )
        )
        return {"templates": pd.DataFrame(), "summaries": [], "prune_summaries": []}

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
            "slot_route_risk_score": 0,
            "slot_route_details": [],
        }

    monkeypatch.setattr(master_module, "_assess_template_graph_health", fake_health)
    monkeypatch.setattr(master_module, "build_transition_templates", fake_rebuild)
    monkeypatch.setattr(master_module, "_run_global_joint_model", fake_joint)

    _, _, _, meta = master_module.solve_master_model(req, transition_pack={"templates": pd.DataFrame()}, orders_df=orders)
    assert meta["precheck_autorelax_applied"] is True
    assert meta["template_graph_health"] == "HEALTHY"
    assert len(rebuild_calls) == 1
    assert rebuild_calls[0][1] > cfg.model.template_top_k
    assert rebuild_calls[0][2] > cfg.model.max_routes_per_slot


def test_config_is_single_source_in_main_path():
    roots = [
        Path(__file__).resolve().parent.parent / "src" / "aps_cp_sat" / "transition" / "templates.py",
        Path(__file__).resolve().parent.parent / "src" / "aps_cp_sat" / "decode" / "joint_solution_decoder.py",
        Path(__file__).resolve().parent.parent / "src" / "aps_cp_sat" / "model" / "master.py",
    ]
    for p in roots:
        txt = p.read_text(encoding="utf-8")
        assert "20.0" not in txt
        assert "1800.0" not in txt

    cfg = PlannerConfig(
        rule=build_default_solve_config(validation_mode=False, production_compatibility_mode=False).rule.__class__(
            **{
                **build_default_solve_config(validation_mode=False, production_compatibility_mode=False).rule.__dict__,
                "virtual_tons": 33.0,
            }
        ),
        model=build_default_solve_config(validation_mode=False, production_compatibility_mode=False).model,
        score=build_default_solve_config(validation_mode=False, production_compatibility_mode=False).score,
    )
    orders = pd.DataFrame(
        [
            {"order_id": "A", "line_capability": "dual", "width": 1200.0, "thickness": 1.0, "temp_min": 700.0, "temp_max": 760.0, "tons": 20.0, "steel_group": "PC", "due_rank": 2},
            {"order_id": "B", "line_capability": "dual", "width": 1300.0, "thickness": 1.0, "temp_min": 700.0, "temp_max": 760.0, "tons": 20.0, "steel_group": "PC", "due_rank": 2},
        ]
    )
    transition_pack = build_transition_templates(orders, cfg)
    tpl = transition_pack["templates"]
    if not tpl.empty:
        assert (tpl["virtual_tons"] % 33.0 == 0).all()

    planned = pd.DataFrame(
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
                "tons": 20.0,
                "line": "big_roll",
                "master_slot": 1,
                "master_seq": 1,
                "campaign_id_hint": 1001,
                "campaign_seq_hint": 1,
                "selected_bridge_path": "W1000|T1.0|TMP[600,620]",
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
                "tons": 20.0,
                "line": "big_roll",
                "master_slot": 1,
                "master_seq": 2,
                "campaign_id_hint": 1001,
                "campaign_seq_hint": 2,
                "selected_bridge_path": "",
                "line_source": "dual",
                "is_virtual": False,
            },
        ]
    )
    decoded = decode_solution(
        ColdRollingResult(
            schedule_df=planned,
            rounds_df=pd.DataFrame(),
            output_path=Path("out.xlsx"),
            dropped_df=pd.DataFrame(),
            config=cfg,
        )
    )
    assert float(decoded.schedule_df.loc[decoded.schedule_df["is_virtual"], "tons"].iloc[0]) == 33.0


def test_feasibility_profile_relaxes_solver():
    default_cfg = build_profile_config("default", validation_mode=False, production_compatibility_mode=False)
    feasibility_cfg = build_profile_config("feasibility", validation_mode=False, production_compatibility_mode=False)
    quality_cfg = build_profile_config("quality", validation_mode=False, production_compatibility_mode=False)
    assert feasibility_cfg.model.profile_name == "feasibility"
    assert feasibility_cfg.allow_fallback is True
    assert feasibility_cfg.allow_legacy_fallback is False
    assert feasibility_cfg.time_limit_seconds > default_cfg.time_limit_seconds
    assert feasibility_cfg.template_top_k > default_cfg.template_top_k
    assert feasibility_cfg.max_routes_per_slot > default_cfg.max_routes_per_slot
    assert feasibility_cfg.min_real_schedule_ratio < default_cfg.min_real_schedule_ratio
    assert quality_cfg.model.profile_name == "quality"


def test_diagnostics_capture_template_and_fallback_signals():
    cfg = build_profile_config("feasibility", validation_mode=False, production_compatibility_mode=False)
    orders = pd.DataFrame(
        [
            {"order_id": "A", "tons": 30.0, "width": 1200.0, "thickness": 1.0, "temp_min": 700.0, "temp_max": 760.0, "steel_group": "PC", "line_capability": "dual"},
            {"order_id": "B", "tons": 31.0, "width": 1100.0, "thickness": 1.0, "temp_min": 700.0, "temp_max": 760.0, "steel_group": "PC", "line_capability": "dual"},
        ]
    )
    transition_pack = {
        "templates": pd.DataFrame(),
        "summaries": [],
        "prune_summaries": [],
    }
    result = ColdRollingResult(
        schedule_df=pd.DataFrame(
            [
                {"line": "big_roll", "campaign_id": 1, "tons": 50.0, "is_virtual": False},
            ]
        ),
        rounds_df=pd.DataFrame(),
        output_path=Path("out.xlsx"),
        dropped_df=pd.DataFrame([{"order_id": "X", "tons": 12.0, "drop_reason": "MASTER_UNASSIGNED"}]),
        engine_meta={
            "engine_used": "semantic_fallback",
            "fallback_used": True,
            "fallback_type": "semantic_fallback",
            "fallback_reason": "joint_timeout_or_infeasible",
            "fallback_trace": [{"idx": 1, "prune": 6, "status": "TIMEOUT_NO_FEASIBLE"}],
            "failure_diagnostics": {"diagnose_global_prune": "全局剪枝过强"},
            "profile_name": "feasibility",
        },
        config=cfg,
    )
    diagnostics = ColdRollingPipeline._build_run_diagnostics(orders, transition_pack, result)
    assert diagnostics["fallback"]["fallback_type"] == "semantic_fallback"
    assert diagnostics["fallback"]["trace_count"] == 1
    assert diagnostics["unassigned"]["top_reason"] == "MASTER_UNASSIGNED"
    assert diagnostics["failure"]["diagnose_global_prune"] == "全局剪枝过强"



def test_sparse_routing_infeasible_defers_early_stop_for_first_semantic_fallback(monkeypatch):
    cfg = build_profile_config("feasibility", validation_mode=False, production_compatibility_mode=False)
    req = type(
        "Req",
        (),
        {
            "orders_path": Path("orders.xlsx"),
            "steel_info_path": Path("steel.xlsx"),
            "output_path": Path("out.xlsx"),
            "config": cfg,
        },
    )()
    orders = pd.DataFrame(
        [
            {"order_id": "A", "tons": 30.0, "line_source": "dual", "line_capability": "dual"},
            {"order_id": "B", "tons": 31.0, "line_source": "dual", "line_capability": "dual"},
        ]
    )
    call_statuses = iter(["ROUTING_INFEASIBLE", "ROUTING_INFEASIBLE", "ROUTING_INFEASIBLE"])

    monkeypatch.setattr(master_module, "_assess_template_graph_health", lambda *args, **kwargs: {"template_graph_health": "SPARSE"})
    monkeypatch.setattr(master_module, "build_transition_templates", lambda *args, **kwargs: {"templates": pd.DataFrame(), "summaries": [], "prune_summaries": []})
    monkeypatch.setattr(master_module, "_run_global_joint_model", lambda *args, **kwargs: {"status": next(call_statuses)})
    monkeypatch.setattr(master_module, "_legacy_fallback_enabled", lambda cfg: True)
    monkeypatch.setattr(master_module, "run_legacy_schedule", lambda **kwargs: (pd.DataFrame(), pd.DataFrame()))

    _, _, _, meta = master_module.solve_master_model(req, transition_pack={"templates": pd.DataFrame()}, orders_df=orders)
    assert meta["engine_used"] == "legacy_fallback"
    assert meta["fallback_attempt_count"] >= 1
    assert meta["early_stop_deferred_for_semantic_fallback"] is True
    assert meta["semantic_fallback_first_attempt_status"] == "ROUTING_INFEASIBLE"
    assert "after_first_semantic_fallback" in meta["early_stop_reason"]
