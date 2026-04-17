# Feasible Block Architecture

## Overview

The Feasible Block Architecture is a "feasibility-first" approach to cold rolling production scheduling. It separates feasibility checking from optimization, making hard constraint violations visible at each step before any optimization attempt.

## Core Principles

1. **Hard constraints are checked BEFORE any optimization attempt**
   - campaign_ton_min (700 tons) is enforced as a HARD constraint, not a soft penalty
   - campaign_ton_max (2000 tons) is enforced as a HARD constraint
   - All violations are reported immediately, not hidden in objective function

2. **Each block (slot) is validated independently for hard constraints**
   - Slots are checked for tonnage window compliance
   - Line compatibility is verified
   - Template transition feasibility is validated

3. **Drop decisions are made based on feasibility analysis, not just soft penalties**
   - Priority-based drop reasons ensure optimal order removal
   - Drop impact is measured by feasibility improvement

4. **The architecture supports progressive relaxation of soft constraints while keeping hard constraints inviolable**

## Two-Phase Solving

### Phase 1: Feasibility
- Minimizes unassigned count and slot count
- Only hard constraints are enforced
- Soft penalties are minimized to essential items only
- Returns INFEASIBLE if any hard constraint is violated

### Phase 2: Optimize
- Full soft optimization on top of feasible solution
- All soft objectives are considered
- Virtual usage, risk scores, smoothness are optimized

## Key Components

### BlockFeasibilityAnalyzer
Analyzes individual blocks (slots) for hard constraint violations.

```python
from aps_cp_sat.model.feasible_block_builder import BlockFeasibilityAnalyzer

analyzer = BlockFeasibilityAnalyzer(campaign_ton_min=700.0, campaign_ton_max=2000.0)
report = analyzer.analyze_all_slots(assigned_df)
```

### DropDecisionEngine
Makes drop decisions based on feasibility impact analysis.

**Drop Priority (highest to lowest):**
1. `TON_WINDOW_INFEASIBLE`: Would break legal 700-2000 ton window
2. `GLOBAL_ISOLATED_ORDER`: No feasible line or transition path
3. `NO_FEASIBLE_LINE`: Line capability mismatch
4. `BRIDGE_REQUIRED_BUT_NOT_SUPPORTED`: Requires bridge exceeding limits
5. `SLOT_ROUTING_RISK_TOO_HIGH`: Local router cannot find valid sequence
6. `LOW_PRIORITY_DROP`: Low priority, loose due date
7. `CAPACITY_PRESSURE`: Forced drop due to capacity constraints

### build_feasible_blocks
Main entry point for the feasible block architecture.

```python
from aps_cp_sat.model.feasible_block_builder import build_feasible_blocks

feasible_orders, dropped_df, report = build_feasible_blocks(orders_df)
```

## Hard Constraint Types

| Constraint | Type | Description |
|------------|------|-------------|
| campaign_ton_min | HARD | Slot tonnage >= 700 tons |
| campaign_ton_max | HARD | Slot tonnage <= 2000 tons |
| line_compatibility | HARD | Order line_capability matches assigned line |
| transition_feasibility | HARD | Valid template exists for all order pairs |
| bridge_count | HARD | Bridge count per transition <= limit |
| reverse_step | HARD | Reverse width step not allowed |

## Hard Violation Types

```python
class HardViolationType(Enum):
    CAMPAIGN_TON_MIN = auto()      # slot < 700 tons
    CAMPAIGN_TON_MAX = auto()      # slot > 2000 tons
    LINE_INCOMPATIBLE = auto()     # order not compatible with slot line
    TRANSITION_INFEASIBLE = auto() # no valid template transition
    BRIDGE_COUNT_EXCEEDED = auto() # bridge count exceeds limit
    REVERSE_STEP_INVALID = auto()  # reverse width step not allowed
```

## Integration with Existing Architecture

The feasible block architecture is designed to integrate with the existing `joint_master.py`:

1. **Pre-processing**: Use `BlockFeasibilityAnalyzer` to identify infeasible orders before calling `_run_global_joint_model`
2. **Post-processing**: Use `DropDecisionEngine` to determine which orders to drop when partial schedules are returned
3. **Validation**: Use `BlockFeasibilityReport` to verify that returned solutions satisfy all hard constraints

## Future Work

- [ ] Implement `BlockFeasibilityAnalyzer` with template coverage checking
- [ ] Implement `GlobalFeasibilityChecker` for cross-block validation
- [ ] Integrate feasible block analysis as pre-processing step in `solve_master_model`
- [ ] Add hard violation tracking to solution validator
- [ ] Create visualization for feasibility analysis results
