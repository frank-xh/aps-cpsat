import os
import sys
from pathlib import Path

import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from aps_cp_sat.config import ModelConfig, PlannerConfig, RuleConfig, ScoreConfig
from aps_cp_sat.domain.models import ColdRollingRequest
from aps_cp_sat.model import master as master_module


def _req(cfg: PlannerConfig) -> ColdRollingRequest:
    return ColdRollingRequest(
        orders_path=Path("orders.xlsx"),
        steel_info_path=Path("steel.xlsx"),
        output_path=Path("out.xlsx"),
        config=cfg,
    )


def _orders() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"order_id": "A", "tons": 30.0, "line_source": "dual", "due_rank": 1, "priority": 1},
            {"order_id": "B", "tons": 31.0, "line_source": "dual", "due_rank": 1, "priority": 1},
        ]
    )


def _feasible_joint(used_local_routing: bool = False) -> dict:
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
        "used_local_routing": used_local_routing,
        "local_routing_role": "transitional_slot_router" if used_local_routing else "not_used",
    }


def test_default_production_path_prefers_joint_master(monkeypatch):
    cfg = PlannerConfig(rule=RuleConfig(), score=ScoreConfig(), model=ModelConfig())
    monkeypatch.setattr(master_module, "_run_global_joint_model", lambda *args, **kwargs: _feasible_joint(False))
    monkeypatch.setattr(master_module, "run_legacy_schedule", lambda **kwargs: (_ for _ in ()).throw(AssertionError("legacy should not run")))
    planned_df, _, dropped_df, meta = master_module.solve_master_model(_req(cfg), transition_pack={}, orders_df=_orders())
    assert not planned_df.empty
    assert dropped_df.empty
    assert meta["main_path"] == "joint_master"
    assert meta["engine_used"] == "joint_master"
    assert meta["fallback_used"] is False


def test_fallback_meta_is_explicit(monkeypatch):
    cfg = PlannerConfig(
        rule=RuleConfig(),
        score=ScoreConfig(),
        model=ModelConfig(enableSemanticFallback=True, enableScaleDownFallback=False, allow_fallback=True),
    )
    calls = {"n": 0}

    def fake_joint(*args, **kwargs):
        calls["n"] += 1
        solve_cfg = args[2]
        if solve_cfg is cfg:
            return {"status": "TIMEOUT_NO_FEASIBLE"}
        return _feasible_joint(True)

    monkeypatch.setattr(master_module, "_run_global_joint_model", fake_joint)
    monkeypatch.setattr(master_module, "run_legacy_schedule", lambda **kwargs: (_ for _ in ()).throw(AssertionError("legacy should not run")))
    _, _, _, meta = master_module.solve_master_model(_req(cfg), transition_pack={}, orders_df=_orders())
    assert meta["main_path"] == "fallback"
    assert meta["fallback_used"] is True
    assert meta["fallback_type"] == "semantic_fallback"
    assert meta["master_entry"] == "_run_global_joint_model"
