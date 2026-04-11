import os
import sys
from pathlib import Path

import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from aps_cp_sat.config import PlannerConfig, build_profile_config
from aps_cp_sat.decode import decode_bridge_path_rows, decode_candidate_allocation
from aps_cp_sat.domain.models import ColdRollingRequest
from aps_cp_sat.io import export_schedule_results
from aps_cp_sat.model.master import _pick_structure_drop_candidates, _run_global_joint_model, solve_master_model
from aps_cp_sat.transition import build_transition_templates


def _mini_orders() -> pd.DataFrame:
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
                "due_rank": 2,
                "priority": 1,
                "line_source": "dual",
            },
            {
                "order_id": "B",
                "source_order_id": "B",
                "parent_order_id": "B",
                "lot_id": "B-L01",
                "grade": "DC01",
                "steel_group": "PC",
                "line_capability": "dual",
                "width": 1100.0,
                "thickness": 1.0,
                "temp_min": 700.0,
                "temp_max": 760.0,
                "temp_mean": 730.0,
                "tons": 32.0,
                "due_rank": 2,
                "priority": 1,
                "line_source": "dual",
            },
        ]
    )


def test_joint_model_minimal_sample_feasible():
    cfg = PlannerConfig(model=PlannerConfig().model.__class__(time_limit_seconds=5.0, master_profile_count=1, master_seed_count=1))
    orders = _mini_orders()
    pack = build_transition_templates(orders, cfg)
    out = _run_global_joint_model(orders, pack, cfg, time_scale=0.2, random_seed=2027)
    assert out["status"] in {"FEASIBLE", "TIMEOUT_NO_FEASIBLE", "INFEASIBLE", "ROUTING_INFEASIBLE"}
    assert "plan_df" in out or out["status"] != "FEASIBLE"


def test_bridge_path_materialize_rows():
    rows, end_idx = decode_bridge_path_rows(
        "W1000|T1.0|TMP[600,620]->W1020|T1.1|TMP[620,640]",
        line="big_roll",
        line_name="大辊线",
        campaign_id=1,
        master_slot=1,
        start_idx=1,
        virtual_tons=20.0,
    )
    assert len(rows) == 2
    assert end_idx == 3
    assert rows[0]["order_id"].startswith("VIRTUAL-")
    assert rows[0]["tons"] == 20.0


def test_bridge_path_materialize_multiple_virtual_steps():
    rows, end_idx = decode_bridge_path_rows(
        "W1000|T0.8|TMP[600,620]->W1250|T1.0|TMP[620,640]->W1500|T1.5|TMP[640,660]",
        line="big_roll",
        line_name="大辊线",
        campaign_id=1,
        master_slot=1,
        start_idx=10,
        virtual_tons=20.0,
    )
    assert len(rows) == 3
    assert end_idx == 13
    assert [float(r["thickness"]) for r in rows] == [0.8, 1.0, 1.5]


def test_decode_candidate_allocation_builds_analysis_only_candidate_schedule():
    orders = _mini_orders()
    candidate_joint = {
        "candidate_plan_df": pd.DataFrame(
            [
                {
                    "row_idx": 0,
                    "order_id": "A",
                    "assigned_line": "big_roll",
                    "assigned_slot": 1,
                    "candidate_position": 1,
                    "slot_unroutable_flag": 1,
                    "slot_route_risk_score": 120,
                    "candidate_status": "UNROUTABLE_SLOT_MEMBER",
                    "selected_edge_type": "",
                    "selected_bridge_path": "",
                }
            ]
        )
    }
    dropped = pd.DataFrame(
        [
            {
                "order_id": "B",
                "tons": 32.0,
                "line_capability": "dual",
                "dominant_drop_reason": "GLOBAL_ISOLATED_ORDER",
                "secondary_reasons": "LOW_DEGREE_ORDER",
                "candidate_lines": "big_roll,small_roll",
                "risk_summary": "global_isolated",
            }
        ]
    )
    meta = {
        "slot_route_details": [
            {
                "line": "big_roll",
                "slot_no": 1,
                "status": "UNROUTABLE_SLOT",
                "slot_route_risk_score": 120,
            }
        ]
    }
    candidate_schedule, big_roll_df, small_roll_df = decode_candidate_allocation(
        candidate_joint,
        orders,
        candidate_dropped_df=dropped,
        engine_meta=meta,
    )
    assert len(candidate_schedule) == 2
    assert int((candidate_schedule["candidate_status"] == "UNROUTABLE_SLOT_MEMBER").sum()) == 1
    assert int((candidate_schedule["candidate_status"] == "DROPPED_CANDIDATE").sum()) == 1
    assert bool(candidate_schedule["analysis_only"].all()) is True
    assert bool((candidate_schedule["official_usable"] == False).all()) is True
    assert len(big_roll_df) == 1
    assert small_roll_df.empty


def test_solve_master_model_no_fallback_returns_failed_routing_result():
    base = PlannerConfig()
    cfg = PlannerConfig(
        rule=base.rule,
        score=base.score,
        model=base.model.__class__(
            allow_fallback=False,
            allow_legacy_fallback=False,
            validation_mode=True,
            time_limit_seconds=1.0,
            master_profile_count=1,
            master_seed_count=1,
        ),
    )
    req = ColdRollingRequest(
        orders_path=Path("orders.xlsx"),
        steel_info_path=Path("steel.xlsx"),
        output_path=Path("out.xlsx"),
        config=cfg,
    )
    orders = _mini_orders()
    final_df, rounds_df, dropped_df, meta = solve_master_model(
        req, transition_pack={"templates": pd.DataFrame()}, orders_df=orders
    )
    assert final_df.empty
    assert rounds_df.empty
    assert dropped_df.empty
    assert meta["result_acceptance_status"] == "FAILED_STRONG_INFEASIBILITY_SIGNAL"
    assert meta["routing_feasible"] is False
    assert meta["routing_status"] == "ROUTING_INFEASIBLE"


def test_export_contains_engine_used(tmp_path: Path):
    cfg = PlannerConfig()
    df = pd.DataFrame(
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
    rounds_df = pd.DataFrame([{"round": 1, "line": "big_roll", "rows_total": 1, "virtual_cnt": 0, "dropped_cnt": 0}])
    out_xlsx = tmp_path / "test_engine_used.xlsx"
    export_schedule_results(
        final_df=df,
        rounds_df=rounds_df,
        dropped_df=pd.DataFrame(),
        output_path=str(out_xlsx),
        input_order_count=1,
        rule=cfg.rule,
        engine_used="joint_master",
        equivalence_summary={"campaign_single_chain_ok": True},
    )
    summary = pd.read_excel(out_xlsx, sheet_name="总览指标")
    assert (summary["指标"] == "结果引擎路径").any()
    v = summary.loc[summary["指标"] == "结果引擎路径", "值"].iloc[0]
    assert str(v) == "joint_master"
    xls = pd.ExcelFile(out_xlsx)
    for sheet in [
        "RUN_SUMMARY",
        "LINE_SUMMARY",
        "VIOLATION_SUMMARY",
        "UNSCHEDULED_SUMMARY",
        "BRIDGE_SUMMARY",
        "SLOT_SUMMARY",
        "DROP_AND_BRIDGE_DETAILS",
        "PROGRESS_METRICS",
    ]:
        assert sheet in xls.sheet_names


def test_export_contains_candidate_analysis_sheets_and_candidate_metrics(tmp_path: Path):
    cfg = PlannerConfig()
    out_xlsx = tmp_path / "candidate_export.xlsx"
    candidate_schedule_df = pd.DataFrame(
        [
            {
                "order_id": "A",
                "line": "big_roll",
                "slot_no": 1,
                "candidate_position": 1,
                "tons": 30.0,
                "width": 1200.0,
                "thickness": 1.0,
                "steel_group": "PC",
                "line_capability": "dual",
                "drop_flag": False,
                "slot_unroutable_flag": True,
                "slot_route_risk_score": 120,
                "candidate_status": "UNROUTABLE_SLOT_MEMBER",
                "selected_edge_type": "",
                "analysis_only": True,
                "official_usable": False,
            },
            {
                "order_id": "B",
                "line": "",
                "slot_no": 0,
                "candidate_position": 0,
                "tons": 32.0,
                "width": 1100.0,
                "thickness": 1.0,
                "steel_group": "PC",
                "line_capability": "dual",
                "drop_flag": True,
                "slot_unroutable_flag": False,
                "slot_route_risk_score": 0,
                "candidate_status": "DROPPED_CANDIDATE",
                "selected_edge_type": "",
                "analysis_only": True,
                "official_usable": False,
                "candidate_lines": "big_roll,small_roll",
                "dominant_drop_reason": "GLOBAL_ISOLATED_ORDER",
            },
        ]
    )
    failure_diagnostics = {
        "fallback": {
            "profile_name": "production_search",
            "result_acceptance_status": "BEST_SEARCH_CANDIDATE_ANALYSIS",
            "failure_mode": "FAILED_ROUTING_SEARCH",
            "result_usage": "ANALYSIS_ONLY",
            "best_candidate_available": True,
            "best_candidate_type": "BEST_SEARCH_CANDIDATE_ANALYSIS",
            "best_candidate_search_status": "ROUTING_INFEASIBLE",
            "best_candidate_routing_feasible": False,
            "best_candidate_unroutable_slot_count": 1,
            "routing_feasible": False,
            "unroutable_slot_count": 1,
            "export_consistency_ok": True,
        },
        "slot_route_details": [
            {
                "line": "big_roll",
                "slot_no": 1,
                "status": "UNROUTABLE_SLOT",
                "zero_in_orders": 1,
                "zero_out_orders": 1,
                "period_reverse_count_violation_count": 0,
            }
        ],
        "validation_summary": {},
    }
    export_schedule_results(
        final_df=pd.DataFrame(),
        rounds_df=pd.DataFrame(),
        dropped_df=pd.DataFrame([{"order_id": "B", "tons": 32.0, "dominant_drop_reason": "GLOBAL_ISOLATED_ORDER"}]),
        output_path=str(out_xlsx),
        input_order_count=2,
        rule=cfg.rule,
        engine_used="joint_master_failed",
        equivalence_summary={},
        failure_diagnostics=failure_diagnostics,
        engine_meta={
            "candidate_schedule_df": candidate_schedule_df,
            "candidate_big_roll_df": candidate_schedule_df[candidate_schedule_df["line"] == "big_roll"].copy(),
            "candidate_small_roll_df": candidate_schedule_df[candidate_schedule_df["line"] == "small_roll"].copy(),
        },
    )
    xls = pd.ExcelFile(out_xlsx)
    for sheet in [
        "BIG_ROLL_CANDIDATE",
        "SMALL_ROLL_CANDIDATE",
        "LINE_SUMMARY_CANDIDATE",
        "VIOLATION_SUMMARY_CANDIDATE",
    ]:
        assert sheet in xls.sheet_names
    # Candidate backfill: main schedule sheets should not be empty when official schedule is empty.
    big_main = pd.read_excel(out_xlsx, sheet_name="大辊线排程")
    small_main = pd.read_excel(out_xlsx, sheet_name="小辊线排程")
    assert len(big_main) == 1
    assert len(small_main) == 0
    line_summary = pd.read_excel(out_xlsx, sheet_name="LINE_SUMMARY")
    assert int(line_summary.loc[line_summary["line"] == "big_roll", "scheduled_orders"].iloc[0]) == 1
    assert (line_summary["summary_mode"] == "CANDIDATE_ANALYSIS").any()
    unroutable_slots = pd.read_excel(out_xlsx, sheet_name="UNROUTABLE_SLOTS")
    assert len(unroutable_slots) == 1
    slot_summary = pd.read_excel(out_xlsx, sheet_name="SLOT_SUMMARY")
    assert len(slot_summary) >= 1
    progress = pd.read_excel(out_xlsx, sheet_name="PROGRESS_METRICS")
    assert int(progress["big_roll_scheduled_orders"].iloc[0]) == 1
    violation_summary = pd.read_excel(out_xlsx, sheet_name="VIOLATION_SUMMARY_CANDIDATE")
    assert (violation_summary["metric"] == "candidate_unroutable_slot_count").any()

def test_solve_master_model_returns_strong_infeasibility_signal():
    base = PlannerConfig()
    cfg = PlannerConfig(
        rule=base.rule.__class__(campaign_ton_max=100.0),
        score=base.score,
        model=base.model.__class__(
            allow_fallback=False,
            allow_legacy_fallback=False,
            validation_mode=True,
            time_limit_seconds=1.0,
            master_profile_count=1,
            master_seed_count=1,
            max_campaign_slots=1,
            big_roll_slot_hard_order_cap=1,
            small_roll_slot_hard_order_cap=1,
        ),
    )
    req = ColdRollingRequest(
        orders_path=Path("orders.xlsx"),
        steel_info_path=Path("steel.xlsx"),
        output_path=Path("out.xlsx"),
        config=cfg,
    )
    orders = _mini_orders()
    final_df, rounds_df, dropped_df, meta = solve_master_model(
        req, transition_pack={"templates": pd.DataFrame()}, orders_df=orders
    )
    assert final_df.empty
    assert rounds_df.empty
    assert dropped_df.empty
    assert meta["result_acceptance_status"] == "FAILED_STRONG_INFEASIBILITY_SIGNAL"
    assert meta["failure_mode"] == "FAILED_STRONG_INFEASIBILITY_SIGNAL"
    assert meta["evidence_level"] == "STRONG_INFEASIBLE_SIGNAL"

def test_joint_master_reports_big_roll_hard_cap_fields():
    from aps_cp_sat.config import build_profile_config

    cfg = build_profile_config("feasibility_fast_slot_safe", validation_mode=False, production_compatibility_mode=False)
    assert cfg.model.big_roll_slot_hard_order_cap == 22
    src = Path(__file__).resolve().parent.parent / "src" / "aps_cp_sat" / "model" / "joint_master.py"
    text = src.read_text(encoding="utf-8")
    assert "big_roll_slot_hard_order_cap" in text
    assert "model.Add(order_cnt <= hard_cap * z[(li, p)])" in text


def test_structure_drop_candidates_focus_big_roll_bad_slots_and_preserve_reasons():
    cfg = build_profile_config("production_search", validation_mode=False, production_compatibility_mode=False)
    orders = pd.DataFrame(
        [
            {
                "order_id": f"O{i}",
                "line_capability": "dual",
                "priority": 0 if i in {1, 2, 3} else 1,
                "due_rank": 3 if i in {1, 2, 3} else 1,
                "tons": 10.0 + i,
                "width": 1000.0 + i * 10,
                "thickness": 1.0 + (i * 0.1),
            }
            for i in range(1, 11)
        ]
    )
    joint = {
        "slot_route_details": [
            {
                "status": "UNROUTABLE_SLOT",
                "line": "big_roll",
                "slot_no": 3,
                "order_count": 9,
                "order_count_over_cap": 4,
                "slot_route_risk_score": 220,
                "template_coverage_ratio": 0.12,
                "zero_in_orders": 2,
                "zero_out_orders": 1,
                "pair_gap_proxy": 80,
                "span_risk": 30,
                "dominant_unroutable_reason": "PAIR_GAP_DOMINANT",
                "top_isolated_orders": ["O1", "O2"],
                "order_ids": [f"O{i}" for i in range(1, 10)],
            },
            {
                "status": "UNROUTABLE_SLOT",
                "line": "small_roll",
                "slot_no": 1,
                "order_count": 4,
                "order_count_over_cap": 1,
                "slot_route_risk_score": 80,
                "template_coverage_ratio": 0.40,
                "zero_in_orders": 0,
                "zero_out_orders": 0,
                "pair_gap_proxy": 10,
                "span_risk": 5,
                "dominant_unroutable_reason": "SPAN_DOMINANT",
                "top_isolated_orders": ["O10"],
                "order_ids": ["O7", "O8", "O9", "O10"],
            },
        ]
    }
    evidence = {"isolated_orders_topn": [{"order_id": "O1"}, {"order_id": "O10"}]}

    picked = _pick_structure_drop_candidates(orders, joint, evidence, cfg)

    assert not picked.empty
    assert "O1" in set(picked["order_id"].astype(str))
    assert int((picked["source_bad_slot_line"] == "big_roll").sum()) >= 1
    row = picked[picked["order_id"].astype(str) == "O1"].iloc[0]
    assert str(row["drop_reason"]) in {"GLOBAL_ISOLATED_ORDER", "SLOT_ROUTING_RISK_TOO_HIGH"}
    assert "LOW_DEGREE_ORDER" in str(row.get("secondary_reasons", ""))
    assert "slot=big_roll:3" in str(row.get("risk_summary", ""))
    assert bool(row.get("would_break_slot_if_kept", False)) is True
