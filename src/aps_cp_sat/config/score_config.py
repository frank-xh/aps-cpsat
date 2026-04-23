from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScoreConfig:
    # ------------------------ 序列结构硬罚分 ------------------------
    seq_unique: int = 500  # 参数：seq_unique，用于控制相关算法行为。
    seq_contiguous: int = 300  # 参数：seq_contiguous，用于控制相关算法行为。
    seq_start: int = 300  # 参数：seq_start，用于控制相关算法行为。

    # ------------------------ 工艺硬罚分 ------------------------
    roll_line_mismatch: int = 10000  # 参数：roll_line_mismatch，用于控制相关算法行为。
    ton_over: int = 10000  # 吨位参数：ton_over。
    ton_under: int = 1000  # 吨位参数：ton_under。
    temp_shortage: int = 10000  # 参数：temp_shortage，用于控制相关算法行为。
    width_violation: int = 10000  # 参数：width_violation，用于控制相关算法行为。
    thick_violation: int = 10000  # 参数：thick_violation，用于控制相关算法行为。
    non_pc_switch: int = 10000  # 参数：non_pc_switch，用于控制相关算法行为。
    virtual_chain_excess: int = 10000  # 虚拟板坯相关参数：virtual_chain_excess。

    # ------------------------ 中强度业务罚分 ------------------------
    unassigned_real: int = 300  # 参数：unassigned_real，用于控制相关算法行为。
    virtual_ratio: int = 9000  # 比例参数：virtual_ratio。
    pure_virtual_campaign: int = 300000  # 虚拟板坯相关参数：pure_virtual_campaign。
    virtual_use: int = 8  # 虚拟板坯相关参数：virtual_use。

    # 主问题层面对虚拟与逆宽的整体控制。
    master_virtual_blocks: int = 80  # 虚拟板坯相关参数：master_virtual_blocks。
    master_virtual_tons: int = 4  # 吨位参数：master_virtual_tons。
    master_virtual_ratio: int = 12  # 比例参数：master_virtual_ratio。
    master_reverse_count: int = 400  # 数量参数：master_reverse_count。
    master_reverse_total_rise: int = 3  # 参数：master_reverse_total_rise，用于控制相关算法行为。
    master_route_risk: int = 20  # 参数：master_route_risk，用于控制相关算法行为。

    # ------------------------ 槽位 / campaign 风险罚分 ------------------------
    slot_isolation_risk_penalty: int = 40  # 惩罚权重：slot_isolation_risk_penalty。
    slot_pair_gap_risk_penalty: int = 30  # 惩罚权重：slot_pair_gap_risk_penalty。
    slot_span_risk_penalty: int = 15  # 惩罚权重：slot_span_risk_penalty。
    slot_order_count_penalty: int = 50  # 惩罚权重：slot_order_count_penalty。

    # ------------------------ 平滑与目标类软分 ------------------------
    width_smooth: int = 2  # 参数：width_smooth，用于控制相关算法行为。
    thick_smooth: int = 2000  # 参数：thick_smooth，用于控制相关算法行为。
    temp_margin: int = 40  # 参数：temp_margin，用于控制相关算法行为。
    ton_target: int = 20  # 吨位参数：ton_target。
    ton_target_value: float = 1800.0  # 吨位参数：ton_target_value。

    # ------------------------ 桥接边代价 ------------------------
    reverse_width_bridge_penalty: int = 600  # 惩罚权重：reverse_width_bridge_penalty。
    template_base_cost_ratio: float = 0.05  # 比例参数：template_base_cost_ratio。
    direct_edge_penalty: int = 0  # 惩罚权重：direct_edge_penalty。
    real_bridge_penalty: int = 40  # 惩罚权重：real_bridge_penalty。
    virtual_bridge_penalty: int = 200  # 惩罚权重：virtual_bridge_penalty。

    # 当模板缺失时的 fallback 边代价权重。
    edge_fallback_width_weight: int = 3  # 评分权重：edge_fallback_width_weight。
    edge_fallback_thick_weight: int = 250  # 评分权重：edge_fallback_thick_weight。
    edge_fallback_due_weight: int = 500  # 评分权重：edge_fallback_due_weight。
    edge_fallback_base_penalty: int = 5000  # 惩罚权重：edge_fallback_base_penalty。
