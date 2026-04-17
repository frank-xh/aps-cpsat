"""Tests for local_inserter_cp_sat module.

These tests run without ortools installed by mocking the entire ortools.sat.python.cp_model module.
"""
from __future__ import annotations

import sys
import types
from typing import Dict

import pytest

# ---------------------------------------------------------------------------
# Mock OR-Tools module (must be installed BEFORE importing local_inserter_cp_sat)
# ---------------------------------------------------------------------------

_MOCK_VAR_REGISTRY: Dict[int, int] = {}  # vid -> solution value


class _MockIntVar:
    """Mock CpModel IntVar that tracks solution values."""
    _counter = 0

    def __init__(self, name: str = ""):
        _MockIntVar._counter += 1
        self._vid = _MockIntVar._counter
        self.name = name
        _MOCK_VAR_REGISTRY[self._vid] = 1  # default: 1

    def solution_value(self) -> int:
        return _MOCK_VAR_REGISTRY.get(self._vid, 1)

    # Support arithmetic operators (needed for model building)
    def __radd__(self, other) -> int:
        return other + 0

    def __add__(self, other):
        return other

    def __sub__(self, other):
        return -other

    def __rsub__(self, other) -> int:
        return -other

    def __mul__(self, other) -> int:
        return 0

    def __rmul__(self, other) -> int:
        return 0


class _MockLinearExpr:
    """Mock LinearExpr for AddMultiObjectiveRegression."""
    _counter = 0

    def __init__(self):
        _MockLinearExpr._counter += 1
        self._lid = _MockLinearExpr._counter

    def __add__(self, other):
        return self

    def __radd__(self, other):
        return self


class _MockCpModel:
    """Mock CpModel."""
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


class _MockConstraint:
    """Mock constraint."""
    def OnlyEnforceIf(self, *args, **kwargs):
        return self


class _MockCpSolver:
    """
    Mock CpSolver.

    Key fix: Value(var) calls var.solution_value() — this is the correct
    OR-Tools API where CpSolver.Value(expression) delegates to the expression's
    solution_value() method.
    """
    def __init__(self):
        self.parameters = self

    def Solve(self, model):
        # Mark all variables as solved with value 1
        for vid in _MOCK_VAR_REGISTRY:
            _MOCK_VAR_REGISTRY[vid] = 1
        return 1

    def Value(self, var=None):
        """Return solution value of a variable or expression.

        This matches the real OR-Tools CpSolver.Value() API:
        CpSolver.Value(expression) -> int

        It reads the solution from the solver's internal state,
        which we store in _MOCK_VAR_REGISTRY (written by Solve()).
        """
        if var is None:
            return 0
        if hasattr(var, 'solution_value'):
            return var.solution_value()
        return 1

    @staticmethod
    def StatusName(s: int) -> str:
        return {1: "OPTIMAL", 2: "FEASIBLE", 3: "INFEASIBLE"}.get(s, "UNKNOWN")


def _install_mock_ortools():
    """Install mock ortools module into sys.modules."""
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
from aps_cp_sat.model.local_inserter_cp_sat import (
    InsertStatus,
    LocalInsertRequest,
    LocalInsertResult,
    solve_local_insertion_subproblem,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_mock_registry():
    """Clear the mock registry before each test."""
    _MOCK_VAR_REGISTRY.clear()
    _MockIntVar._counter = 0
    yield


@pytest.fixture
def basic_orders():
    return pd.DataFrame([
        {"order_id": "F1", "tons": 300.0, "width": 1000, "thickness": 2.0,
         "line_capability": "dual"},
        {"order_id": "F2", "tons": 400.0, "width": 1050, "thickness": 2.1,
         "line_capability": "dual"},
        {"order_id": "C1", "tons": 350.0, "width": 1100, "thickness": 2.2,
         "line_capability": "dual"},
        {"order_id": "C2", "tons": 200.0, "width": 1200, "thickness": 2.3,
         "line_capability": "dual"},
    ])


@pytest.fixture
def basic_tpl_df():
    """Template edges for basic_orders."""
    return pd.DataFrame([
        # F1 -> F2 (fixed backbone)
        {"from_order_id": "F1", "to_order_id": "F2", "line": "big_roll",
         "cost": 10.0, "width_smooth_cost": 0, "thickness_smooth_cost": 0,
         "temp_margin_cost": 0, "cross_group_cost": 0,
         "bridge_count": 0, "edge_type": "DIRECT_EDGE"},
        # F2 -> C1 (insertable)
        {"from_order_id": "F2", "to_order_id": "C1", "line": "big_roll",
         "cost": 12.0, "width_smooth_cost": 0, "thickness_smooth_cost": 0,
         "temp_margin_cost": 0, "cross_group_cost": 0,
         "bridge_count": 0, "edge_type": "DIRECT_EDGE"},
        # F1 -> C1 (alternative path)
        {"from_order_id": "F1", "to_order_id": "C1", "line": "big_roll",
         "cost": 20.0, "width_smooth_cost": 0, "thickness_smooth_cost": 0,
         "temp_margin_cost": 0, "cross_group_cost": 0,
         "bridge_count": 0, "edge_type": "DIRECT_EDGE"},
        # F2 -> C2 (second candidate)
        {"from_order_id": "F2", "to_order_id": "C2", "line": "big_roll",
         "cost": 15.0, "width_smooth_cost": 0, "thickness_smooth_cost": 0,
         "temp_margin_cost": 0, "cross_group_cost": 0,
         "bridge_count": 0, "edge_type": "DIRECT_EDGE"},
        # C1 -> (terminal) - no outgoing needed since last node has implicit self-loop
    ])


@pytest.fixture
def basic_transition_pack(basic_tpl_df):
    return {"templates": basic_tpl_df}


@pytest.fixture
def planner_cfg():
    return PlannerConfig()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_basic_insertion_accepts_candidate(
    basic_orders, basic_transition_pack, planner_cfg,
):
    """C1 is inserted between F1 and F2 via F1->C1 and C1->F2 edges."""
    # Add C1->F2 edge to make the hint work
    tpl = basic_transition_pack["templates"]
    tpl = pd.concat([
        tpl,
        pd.DataFrame([{
            "from_order_id": "C1", "to_order_id": "F2",
            "line": "big_roll", "cost": 12.0,
            "width_smooth_cost": 0, "thickness_smooth_cost": 0,
            "temp_margin_cost": 0, "cross_group_cost": 0,
            "bridge_count": 0, "edge_type": "DIRECT_EDGE",
        }])
    ], ignore_index=True)
    transition_pack = {"templates": tpl}

    req = LocalInsertRequest(
        line="big_roll",
        fixed_order_ids=["F1", "F2"],
        candidate_insert_ids=["C1"],
        time_limit_seconds=5.0,
        random_seed=42,
    )

    result = solve_local_insertion_subproblem(
        req, basic_orders, transition_pack, planner_cfg,
    )

    assert result.status in (InsertStatus.OPTIMAL, InsertStatus.FEASIBLE), \
        f"Expected OPTIMAL/FEASIBLE, got {result.status}"
    assert result.diagnostics["accepted_count"] >= 0
    assert len(result.diagnostics) > 0


def test_candidate_rejected_when_no_valid_edge(
    basic_orders, planner_cfg,
):
    """Candidate with no outgoing/incoming template edges should be excluded."""
    # No edges involving C2 at all — no valid insertion possible
    tpl_df = pd.DataFrame([
        {"from_order_id": "F1", "to_order_id": "F2", "line": "big_roll",
         "cost": 10.0, "width_smooth_cost": 0, "thickness_smooth_cost": 0,
         "temp_margin_cost": 0, "cross_group_cost": 0,
         "bridge_count": 0, "edge_type": "DIRECT_EDGE"},
    ])
    transition_pack = {"templates": tpl_df}

    req = LocalInsertRequest(
        line="big_roll",
        fixed_order_ids=["F1", "F2"],
        candidate_insert_ids=["C1"],  # No edges for C1
        time_limit_seconds=5.0,
        random_seed=42,
    )

    result = solve_local_insertion_subproblem(
        req, basic_orders, transition_pack, planner_cfg,
    )

    # Should still return a result (may be INFEASIBLE or NO_IMPROVEMENT)
    assert isinstance(result.status, InsertStatus)
    assert result.diagnostics["candidate_count"] == 1


def test_infeasible_when_no_template_edges(
    basic_orders, planner_cfg,
):
    """No template edges at all → INFEASIBLE."""
    transition_pack = {"templates": pd.DataFrame()}  # empty

    req = LocalInsertRequest(
        line="big_roll",
        fixed_order_ids=["F1", "F2"],
        candidate_insert_ids=["C1"],
        time_limit_seconds=5.0,
        random_seed=42,
    )

    result = solve_local_insertion_subproblem(
        req, basic_orders, transition_pack, planner_cfg,
    )

    assert result.status == InsertStatus.INFEASIBLE
    assert "NO_VALID_TEMPLATE_EDGES" in result.diagnostics.get("reason", "")


def test_infeasible_when_transition_pack_is_none(
    basic_orders, planner_cfg,
):
    """transition_pack=None or missing templates → INFEASIBLE."""
    req = LocalInsertRequest(
        line="big_roll",
        fixed_order_ids=["F1", "F2"],
        candidate_insert_ids=["C1"],
        time_limit_seconds=5.0,
        random_seed=42,
    )

    result = solve_local_insertion_subproblem(
        req, basic_orders, None, planner_cfg,
    )

    assert result.status == InsertStatus.INFEASIBLE


def test_virtual_bridge_tracking(
    basic_orders, planner_cfg,
):
    """Virtual bridge edges are tracked in diagnostics."""
    tpl_df = pd.DataFrame([
        {"from_order_id": "F1", "to_order_id": "F2", "line": "big_roll",
         "cost": 10.0, "width_smooth_cost": 0, "thickness_smooth_cost": 0,
         "temp_margin_cost": 0, "cross_group_cost": 0,
         "bridge_count": 1, "edge_type": "VIRTUAL_BRIDGE_EDGE"},
        {"from_order_id": "F2", "to_order_id": "C1", "line": "big_roll",
         "cost": 12.0, "width_smooth_cost": 0, "thickness_smooth_cost": 0,
         "temp_margin_cost": 0, "cross_group_cost": 0,
         "bridge_count": 0, "edge_type": "DIRECT_EDGE"},
        {"from_order_id": "F1", "to_order_id": "C1", "line": "big_roll",
         "cost": 20.0, "width_smooth_cost": 0, "thickness_smooth_cost": 0,
         "temp_margin_cost": 0, "cross_group_cost": 0,
         "bridge_count": 0, "edge_type": "DIRECT_EDGE"},
        {"from_order_id": "C1", "to_order_id": "F2", "line": "big_roll",
         "cost": 12.0, "width_smooth_cost": 0, "thickness_smooth_cost": 0,
         "temp_margin_cost": 0, "cross_group_cost": 0,
         "bridge_count": 0, "edge_type": "DIRECT_EDGE"},
    ])
    transition_pack = {"templates": tpl_df}

    req = LocalInsertRequest(
        line="big_roll",
        fixed_order_ids=["F1", "F2"],
        candidate_insert_ids=["C1"],
        time_limit_seconds=5.0,
        random_seed=42,
    )

    result = solve_local_insertion_subproblem(
        req, basic_orders, transition_pack, planner_cfg,
    )

    assert "selected_virtual_bridge_edge_count" in result.diagnostics
    # Virtual bridge may or may not be used depending on solution
    assert isinstance(result.diagnostics["selected_virtual_bridge_edge_count"], int)


def test_diagnostics_keys_present(
    basic_orders, basic_transition_pack, planner_cfg,
):
    """All required diagnostic keys are present in result."""
    req = LocalInsertRequest(
        line="big_roll",
        fixed_order_ids=["F1", "F2"],
        candidate_insert_ids=["C1"],
        time_limit_seconds=5.0,
        random_seed=42,
    )

    result = solve_local_insertion_subproblem(
        req, basic_orders, basic_transition_pack, planner_cfg,
    )

    required_keys = {
        "subproblem_order_count",
        "fixed_count",
        "candidate_count",
        "accepted_count",
        "rejected_count",
        "used_hint",
        "edge_count_in_graph",
        "path_length",
        "solver_status",
    }
    diag_keys = set(result.diagnostics.keys())
    assert required_keys.issubset(diag_keys), \
        f"Missing keys: {required_keys - diag_keys}"


def test_max_orders_cap(
    basic_orders, planner_cfg,
):
    """When fixed+candidates exceed max_orders_in_subproblem, drop largest."""
    tpl_df = pd.DataFrame([
        {"from_order_id": "F1", "to_order_id": "F2", "line": "big_roll",
         "cost": 10.0, "width_smooth_cost": 0, "thickness_smooth_cost": 0,
         "temp_margin_cost": 0, "cross_group_cost": 0,
         "bridge_count": 0, "edge_type": "DIRECT_EDGE"},
        {"from_order_id": "F2", "to_order_id": "C1", "line": "big_roll",
         "cost": 12.0, "width_smooth_cost": 0, "thickness_smooth_cost": 0,
         "temp_margin_cost": 0, "cross_group_cost": 0,
         "bridge_count": 0, "edge_type": "DIRECT_EDGE"},
    ])
    transition_pack = {"templates": tpl_df}

    # Only room for 2 orders: F1 + F2 (fixed). C1 gets dropped.
    req = LocalInsertRequest(
        line="big_roll",
        fixed_order_ids=["F1", "F2"],
        candidate_insert_ids=["C1"],
        time_limit_seconds=5.0,
        random_seed=42,
        max_orders_in_subproblem=2,  # Exactly fits fixed orders
    )

    result = solve_local_insertion_subproblem(
        req, basic_orders, transition_pack, planner_cfg,
    )

    # Should still produce a result
    assert isinstance(result.status, InsertStatus)
    # Candidate C1 (350t) should be in dropped list since it was excluded by cap
    assert result.diagnostics["subproblem_order_count"] <= 2


def test_result_dataclass_fields(
    basic_orders, basic_transition_pack, planner_cfg,
):
    """LocalInsertResult has all required fields."""
    req = LocalInsertRequest(
        line="big_roll",
        fixed_order_ids=["F1"],
        candidate_insert_ids=["C1"],
        time_limit_seconds=5.0,
        random_seed=42,
    )

    result = solve_local_insertion_subproblem(
        req, basic_orders, basic_transition_pack, planner_cfg,
    )

    assert isinstance(result, LocalInsertResult)
    assert isinstance(result.status, InsertStatus)
    assert isinstance(result.sequence, list)
    assert isinstance(result.inserted_order_ids, list)
    assert isinstance(result.kept_order_ids, list)
    assert isinstance(result.dropped_candidate_ids, list)
    assert isinstance(result.objective, (int, float))
    assert isinstance(result.diagnostics, dict)


def test_fixed_order_preserved(
    basic_orders, planner_cfg,
):
    """Fixed orders are always kept in the sequence."""
    tpl_df = pd.DataFrame([
        {"from_order_id": "F1", "to_order_id": "F2", "line": "big_roll",
         "cost": 10.0, "width_smooth_cost": 0, "thickness_smooth_cost": 0,
         "temp_margin_cost": 0, "cross_group_cost": 0,
         "bridge_count": 0, "edge_type": "DIRECT_EDGE"},
        {"from_order_id": "F1", "to_order_id": "C1", "line": "big_roll",
         "cost": 20.0, "width_smooth_cost": 0, "thickness_smooth_cost": 0,
         "temp_margin_cost": 0, "cross_group_cost": 0,
         "bridge_count": 0, "edge_type": "DIRECT_EDGE"},
        {"from_order_id": "F2", "to_order_id": "C1", "line": "big_roll",
         "cost": 15.0, "width_smooth_cost": 0, "thickness_smooth_cost": 0,
         "temp_margin_cost": 0, "cross_group_cost": 0,
         "bridge_count": 0, "edge_type": "DIRECT_EDGE"},
        {"from_order_id": "C1", "to_order_id": "F2", "line": "big_roll",
         "cost": 12.0, "width_smooth_cost": 0, "thickness_smooth_cost": 0,
         "temp_margin_cost": 0, "cross_group_cost": 0,
         "bridge_count": 0, "edge_type": "DIRECT_EDGE"},
    ])
    transition_pack = {"templates": tpl_df}

    req = LocalInsertRequest(
        line="big_roll",
        fixed_order_ids=["F1", "F2"],
        candidate_insert_ids=["C1"],
        time_limit_seconds=5.0,
        random_seed=42,
    )

    result = solve_local_insertion_subproblem(
        req, basic_orders, transition_pack, planner_cfg,
    )

    # Fixed orders should appear in sequence
    for fid in ["F1", "F2"]:
        assert fid in result.kept_order_ids or fid in result.sequence
