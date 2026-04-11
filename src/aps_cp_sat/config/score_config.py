from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True)
class ScoreConfig:
    seq_unique: int = 500
    seq_contiguous: int = 300
    seq_start: int = 300
    roll_line_mismatch: int = 10000
    ton_over: int = 10000
    ton_under: int = 1000
    temp_shortage: int = 10000
    width_violation: int = 10000
    thick_violation: int = 10000
    non_pc_switch: int = 10000
    virtual_chain_excess: int = 10000
    unassigned_real: int = 200
    virtual_ratio: int = 8000
    pure_virtual_campaign: int = 300000
    virtual_use: int = 5
    master_virtual_blocks: int = 80
    master_virtual_tons: int = 4
    master_virtual_ratio: int = 12
    master_reverse_count: int = 400
    master_reverse_total_rise: int = 3
    master_route_risk: int = 20
    slot_isolation_risk_penalty: int = 30
    slot_pair_gap_risk_penalty: int = 20
    slot_span_risk_penalty: int = 10
    slot_order_count_penalty: int = 40
    width_smooth: int = 2
    thick_smooth: int = 2000
    temp_margin: int = 40
    ton_target: int = 20
    ton_target_value: float = 1800.0
    reverse_width_bridge_penalty: int = 600
    template_base_cost_ratio: float = 0.05
    direct_edge_penalty: int = 0
    real_bridge_penalty: int = 50
    virtual_bridge_penalty: int = 200
    edge_fallback_width_weight: int = 3
    edge_fallback_thick_weight: int = 250
    edge_fallback_due_weight: int = 500
    edge_fallback_base_penalty: int = 5000
