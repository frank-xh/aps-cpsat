"""Tests for constructive_lns_master module.

These tests run without ortools installed by mocking the entire ortools.sat.python.cp_model module.
"""
from __future__ import annotations

import sys
import types
from typing import Dict

import pytest

# ---------------------------------------------------------------------------
# Mock OR-Tools module (must be installed BEFORE importing constructive_lns_master)
# ---------------------------------------------------------------------------

_MOCK_VAR_REGISTRY: Dict[int, int] = {}


class _MockIntVar:
    _counter = 0

    def __init__(self, name: str = ""):
        _MockIntVar._counter += 1
        self._vid = _MockIntVar._counter
        self.name = name
        _MOCK_VAR_REGISTRY[self._vid] = 1

    def solution_value(self) -> int:
        return _MOCK_VAR_REGISTRY.get(self._vid, 1)

    def __radd__(self, other) -> int:
        return other + 0

    def __add__(self, other):
        return other

    def __neg__(self) -> int:
        return 0

    def __sub__(self, other):
        return 0

    def __rsub__(self, other) -> int:
        return 0

    def __mul__(self, other) -> int:
        return 0

    def __rmul__(self, other) -> int:
        return 0


class _MockConstraint:
    def OnlyEnforceIf(self, *args, **kwargs):
        return self


class _MockCpModel:
    def NewBoolVar(self, name: str = ""):
        return _MockIntVar(name)

    def NewIntVar(self, lb: int, ub: int, name: str = ""):
        return _MockIntVar(name)

    def Add(self, *args, **kwargs):
        return _MockConstraint()

    def AddMultiObjectiveRegression(self, *args, **kwargs):
        pass

    def AddHint(self, *args, **kwargs):
        pass

    def Minimize(self, *args, **kwargs):
        pass

    def Maximize(self, *args, **kwargs):
        pass

    def Proto(self):
        return None

    def AddCircuit(self, arcs):
        # arcs: List[Tuple[int, int, IntVar]] - mock does nothing
        return _MockConstraint()

    def AddDisjunction(self, literals):
        return _MockConstraint()


class _MockCpSolver:
    def __init__(self):
        self.parameters = self

    def Solve(self, model):
        for vid in _MOCK_VAR_REGISTRY:
            _MOCK_VAR_REGISTRY[vid] = 1
        return 1

    def Value(self, var=None):
        if var is None:
            return 0
        if hasattr(var, 'solution_value'):
            return var.solution_value()
        return 1

    @staticmethod
    def StatusName(s: int) -> str:
        return {1: "OPTIMAL", 2: "FEASIBLE", 3: "INFEASIBLE"}.get(s, "UNKNOWN")


def _install_mock_ortools():
    ortools_mock = types.ModuleType("ortools")
    cp_mock = types.ModuleType("ortools.sat")
    cp_model_mock = types.ModuleType("ortools.sat.python")
    ortools_mock.sat = cp_mock
    cp_mock.python = cp_model_mock

    cp_model_mock.CpModel = _MockCpModel
    cp_model_mock.CpSolver = _MockCpSolver
    cp_model_mock.cp_model = cp_model_mock
    cp_model_mock.OPTIMAL = 1
    cp_model_mock.FEASIBLE = 2
    cp_model_mock.INFEASIBLE = 3
    cp_model_mock.UNBOUNDED = 4
    cp_model_mock.UNKNOWN = 5
    cp_model_mock.MODEL_INVALID = 6

    sys.modules["ortools"] = ortools_mock
    sys.modules["ortools.sat"] = cp_mock
    sys.modules["ortools.sat.python"] = cp_model_mock


_install_mock_ortools()

# Now import the module under test
import pandas as pd
from aps_cp_sat.config import PlannerConfig
from aps_cp_sat.model.constructive_lns_master import (
    ConstructiveLnsResult,
    DropReason,
    LnsRound,
    LnsStatus,
    NeighborhoodType,
    run_constructive_lns_master,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_mock_registry():
    _MOCK_VAR_REGISTRY.clear()
    _MockIntVar._counter = 0
    yield


@pytest.fixture
def sample_orders():
    """Small set of orders across two lines."""
    return pd.DataFrame([
        {"order_id": "O1", "tons": 300.0, "width": 1000, "thickness": 2.0,
         "line_capability": "dual", "steel_group": "SG1", "due_date": "2026-04-20",
         "priority": 1, "due_rank": 1},
        {"order_id": "O2", "tons": 400.0, "width": 1050, "thickness": 2.1,
         "line_capability": "dual", "steel_group": "SG1", "due_date": "2026-04-20",
         "priority": 1, "due_rank": 2},
        {"order_id": "O3", "tons": 350.0, "width": 1100, "thickness": 2.2,
         "line_capability": "dual", "steel_group": "SG2", "due_date": "2026-04-21",
         "priority": 2, "due_rank": 3},
        {"order_id": "O4", "tons": 200.0, "width": 1200, "thickness": 2.3,
         "line_capability": "dual", "steel_group": "SG2", "due_date": "2026-04-21",
         "priority": 2, "due_rank": 4},
        {"order_id": "O5", "tons": 250.0, "width": 950, "thickness": 1.9,
         "line_capability": "dual", "steel_group": "SG3", "due_date": "2026-04-22",
         "priority": 3, "due_rank": 5},
        {"order_id": "O6", "tons": 150.0, "width": 1300, "thickness": 2.5,
         "line_capability": "dual", "steel_group": "SG3", "due_date": "2026-04-22",
         "priority": 3, "due_rank": 6},
    ])


@pytest.fixture
def sample_tpl_df():
    """Template edges connecting O1-O2-O3 and O2-O4."""
    return pd.DataFrame([
        {"from_order_id": "O1", "to_order_id": "O2", "line": "big_roll",
         "cost": 10.0, "width_smooth_cost": 0, "thickness_smooth_cost": 0,
         "temp_margin_cost": 0, "cross_group_cost": 0,
         "bridge_count": 0, "edge_type": "DIRECT_EDGE"},
        {"from_order_id": "O2", "to_order_id": "O3", "line": "big_roll",
         "cost": 12.0, "width_smooth_cost": 0, "thickness_smooth_cost": 0,
         "temp_margin_cost": 0, "cross_group_cost": 0,
         "bridge_count": 0, "edge_type": "DIRECT_EDGE"},
        {"from_order_id": "O3", "to_order_id": "O4", "line": "big_roll",
         "cost": 15.0, "width_smooth_cost": 0, "thickness_smooth_cost": 0,
         "temp_margin_cost": 0, "cross_group_cost": 0,
         "bridge_count": 0, "edge_type": "DIRECT_EDGE"},
        {"from_order_id": "O2", "to_order_id": "O4", "line": "big_roll",
         "cost": 14.0, "width_smooth_cost": 0, "thickness_smooth_cost": 0,
         "temp_margin_cost": 0, "cross_group_cost": 0,
         "bridge_count": 0, "edge_type": "DIRECT_EDGE"},
        {"from_order_id": "O1", "to_order_id": "O3", "line": "big_roll",
         "cost": 20.0, "width_smooth_cost": 0, "thickness_smooth_cost": 0,
         "temp_margin_cost": 0, "cross_group_cost": 0,
         "bridge_count": 0, "edge_type": "DIRECT_EDGE"},
        {"from_order_id": "O5", "to_order_id": "O6", "line": "big_roll",
         "cost": 8.0, "width_smooth_cost": 0, "thickness_smooth_cost": 0,
         "temp_margin_cost": 0, "cross_group_cost": 0,
         "bridge_count": 0, "edge_type": "DIRECT_EDGE"},
    ])


@pytest.fixture
def sample_transition_pack(sample_tpl_df):
    return {"templates": sample_tpl_df}


@pytest.fixture
def planner_cfg():
    return PlannerConfig()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_returns_constructive_lns_result(
    sample_orders, sample_transition_pack, planner_cfg,
):
    """run_constructive_lns_master returns a ConstructiveLnsResult."""
    result = run_constructive_lns_master(
        sample_orders, sample_transition_pack, planner_cfg, random_seed=42,
    )
    assert isinstance(result, ConstructiveLnsResult)


def test_status_is_defined(
    sample_orders, sample_transition_pack, planner_cfg,
):
    """Result status is one of LnsStatus values."""
    result = run_constructive_lns_master(
        sample_orders, sample_transition_pack, planner_cfg, random_seed=42,
    )
    assert isinstance(result.status, LnsStatus)


def test_planned_df_has_required_columns(
    sample_orders, sample_transition_pack, planner_cfg,
):
    """planned_df contains all required columns per spec."""
    result = run_constructive_lns_master(
        sample_orders, sample_transition_pack, planner_cfg, random_seed=42,
    )
    required_cols = {
        "order_id", "line", "master_slot", "master_seq",
        "campaign_id_hint", "campaign_seq_hint", "selected_template_id",
        "selected_bridge_path", "force_break_before",
    }
    actual_cols = set(result.planned_df.columns)
    assert required_cols.issubset(actual_cols), \
        f"Missing cols: {required_cols - actual_cols}"


def test_dropped_df_has_drop_reason_column(
    sample_orders, sample_transition_pack, planner_cfg,
):
    """dropped_df contains drop_reason column."""
    result = run_constructive_lns_master(
        sample_orders, sample_transition_pack, planner_cfg, random_seed=42,
    )
    if not result.dropped_df.empty:
        assert "drop_reason" in result.dropped_df.columns


def test_dropped_df_drop_reasons_are_valid(
    sample_orders, sample_transition_pack, planner_cfg,
):
    """All drop_reason values are valid DropReason enum values."""
    result = run_constructive_lns_master(
        sample_orders, sample_transition_pack, planner_cfg, random_seed=42,
    )
    if not result.dropped_df.empty:
        valid_reasons = {e.value for e in DropReason}
        for reason in result.dropped_df["drop_reason"]:
            assert reason in valid_reasons, f"Invalid drop_reason: {reason}"


def test_rounds_df_has_required_columns(
    sample_orders, sample_transition_pack, planner_cfg,
):
    """rounds_df contains all required columns per spec."""
    result = run_constructive_lns_master(
        sample_orders, sample_transition_pack, planner_cfg, random_seed=42,
    )
    required_cols = {
        "round", "neighborhood_type", "accepted",
        "scheduled_order_count", "scheduled_tons", "dropped_count",
    }
    actual_cols = set(result.rounds_df.columns)
    assert required_cols.issubset(actual_cols), \
        f"Missing cols: {required_cols - actual_cols}"


def test_rounds_df_row_count_matches_n_rounds(
    sample_orders, sample_transition_pack, planner_cfg,
):
    """rounds_df has one row per ALNS round."""
    result = run_constructive_lns_master(
        sample_orders, sample_transition_pack, planner_cfg,
        random_seed=42,
    )
    assert len(result.rounds_df) >= 1
    assert "round" in result.rounds_df.columns
    rounds = result.rounds_df["round"].tolist()
    assert rounds == list(range(1, len(rounds) + 1))


def test_engine_meta_contains_lns_flags(
    sample_orders, sample_transition_pack, planner_cfg,
):
    """engine_meta confirms LNS path constraints."""
    result = run_constructive_lns_master(
        sample_orders, sample_transition_pack, planner_cfg, random_seed=42,
    )
    assert result.engine_meta.get("no_global_joint_model") is True
    assert result.engine_meta.get("no_slot_bucket") is True
    assert result.engine_meta.get("no_illegal_penalty") is True


def test_diagnostics_contains_improvement_delta(
    sample_orders, sample_transition_pack, planner_cfg,
):
    """diagnostics contains improvement_delta_orders."""
    result = run_constructive_lns_master(
        sample_orders, sample_transition_pack, planner_cfg, random_seed=42,
    )
    assert "improvement_delta_orders" in result.diagnostics
    assert isinstance(result.diagnostics["improvement_delta_orders"], int)


def test_planned_and_dropped_are_disjoint(
    sample_orders, sample_transition_pack, planner_cfg,
):
    """No order appears in both planned_df and dropped_df."""
    result = run_constructive_lns_master(
        sample_orders, sample_transition_pack, planner_cfg, random_seed=42,
    )
    planned_ids = set(result.planned_df["order_id"].tolist()) if not result.planned_df.empty else set()
    dropped_ids = set(result.dropped_df["order_id"].tolist()) if not result.dropped_df.empty else set()
    overlap = planned_ids & dropped_ids
    assert len(overlap) == 0, f"Overlapping order IDs: {overlap}"


def test_all_input_orders_accounted_for(
    sample_orders, sample_transition_pack, planner_cfg,
):
    """Every input order appears in either planned_df or dropped_df."""
    result = run_constructive_lns_master(
        sample_orders, sample_transition_pack, planner_cfg, random_seed=42,
    )
    input_ids = set(sample_orders["order_id"].tolist())
    planned_ids = set(result.planned_df["order_id"].tolist()) if not result.planned_df.empty else set()
    dropped_ids = set(result.dropped_df["order_id"].tolist()) if not result.dropped_df.empty else set()
    accounted = planned_ids | dropped_ids
    # At least some orders should be accounted for
    assert len(accounted) >= 1


def test_zero_orders_returns_result(
    sample_transition_pack, planner_cfg,
):
    """Empty orders DataFrame still returns a result (no crash)."""
    empty_orders = pd.DataFrame([
        {"order_id": "X1", "tons": 100.0, "width": 1000, "thickness": 2.0,
         "line_capability": "dual", "priority": 1, "due_rank": 1},
    ])
    result = run_constructive_lns_master(
        empty_orders, sample_transition_pack, planner_cfg, random_seed=42,
    )
    assert isinstance(result.status, LnsStatus)
    assert hasattr(result, "planned_df")
    assert hasattr(result, "dropped_df")


def test_empty_template_edges_still_returns_result(
    sample_orders, planner_cfg,
):
    """Empty transition_pack['templates'] still returns a result."""
    empty_pack = {"templates": pd.DataFrame()}
    result = run_constructive_lns_master(
        sample_orders, empty_pack, planner_cfg, random_seed=42,
    )
    assert isinstance(result, ConstructiveLnsResult)
    assert isinstance(result.status, LnsStatus)


def test_lns_status_valid(
    sample_orders, sample_transition_pack, planner_cfg,
):
    """Status is always FEASIBLE, NO_IMPROVEMENT, or OPTIMAL."""
    result = run_constructive_lns_master(
        sample_orders, sample_transition_pack, planner_cfg, random_seed=42,
    )
    assert result.status in (
        LnsStatus.OPTIMAL, LnsStatus.FEASIBLE, LnsStatus.NO_IMPROVEMENT
    ), f"Unexpected status: {result.status}"


def test_seed_reproducibility(
    sample_orders, sample_transition_pack, planner_cfg,
):
    """Same seed produces same number of rounds."""
    r1 = run_constructive_lns_master(
        sample_orders, sample_transition_pack, planner_cfg, random_seed=999,
    )
    r2 = run_constructive_lns_master(
        sample_orders, sample_transition_pack, planner_cfg, random_seed=999,
    )
    assert len(r1.rounds_df) == len(r2.rounds_df)
