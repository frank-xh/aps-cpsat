import os
import sys
from pathlib import Path

import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from aps_cp_sat.config import ModelConfig, PlannerConfig, RuleConfig, ScoreConfig
from aps_cp_sat.domain.models import ColdRollingRequest
from aps_cp_sat.model import master as master_module


def _req() -> ColdRollingRequest:
    return ColdRollingRequest(
        orders_path=Path("orders.xlsx"),
        steel_info_path=Path("steel.xlsx"),
        output_path=Path("out.xlsx"),
        config=PlannerConfig(rule=RuleConfig(), score=ScoreConfig(), model=ModelConfig()),
    )


def _orders() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"order_id": "A", "tons": 30.0, "line_source": "dual"},
            {"order_id": "B", "tons": 31.0, "line_source": "dual"},
        ]
    )


def _feasible_joint(used_local_routing: bool) -> dict:
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


def test_local_router_role_marked_when_used(monkeypatch):
    monkeypatch.setattr(master_module, "_run_global_joint_model", lambda *args, **kwargs: _feasible_joint(True))
    _, _, _, meta = master_module.solve_master_model(_req(), transition_pack={}, orders_df=_orders())
    assert meta["used_local_routing"] is True
    assert meta["local_routing_role"] == "transitional_slot_router"


def test_local_router_role_marked_when_not_used(monkeypatch):
    monkeypatch.setattr(master_module, "_run_global_joint_model", lambda *args, **kwargs: _feasible_joint(False))
    _, _, _, meta = master_module.solve_master_model(_req(), transition_pack={}, orders_df=_orders())
    assert meta["used_local_routing"] is False
    assert meta["local_routing_role"] == "not_used"
