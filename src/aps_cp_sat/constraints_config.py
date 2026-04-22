from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConstraintThresholds:
    # 工艺阈值
    min_temp_overlap_real_real: float = 10.0
    max_width_drop: float = 250.0
    # 逆宽规则：逻辑逆宽量上限、单次物理逆宽步长、每轧期逻辑逆宽次数上限。
    real_reverse_step_max_mm: float = 50.0
    virtual_reverse_attach_max_mm: float = 50.0
    max_width_rise_physical_step: float = 50.0
    max_logical_reverse_per_campaign: int = 5

    # 轧期吨位阈值
    campaign_ton_min: float = 700.0
    campaign_ton_max: float = 2000.0
    campaign_ton_target: float = 1800.0

    # 虚拟桥接阈值
    virtual_chain_penalty_threshold: int = 5

    # 温度富余度阈值（overlap 在 [min_temp_overlap_real_real, temp_margin_target) 计罚）
    temp_margin_target: float = 30.0

    # 虚拟吨位占比阈值：virtualTon10 * den - totalTon10 * num > 0 触发
    # 默认 20% -> 1 / 5
    virtual_ton_ratio_num: int = 1
    virtual_ton_ratio_den: int = 5


@dataclass(frozen=True)
class ConstraintWeights:
    # 硬约束罚分（统一评分口径）
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

    # 中约束
    unassigned_real: int = 200
    # 保持20%阈值不变，仅降低超限惩罚强度，允许更积极地用虚拟板坯换低吨位改善
    virtual_ratio: int = 8000
    pure_virtual_campaign: int = 300000
    # 降低单块虚拟板坯使用惩罚，鼓励桥接补吨
    virtual_use: int = 5

    # 软约束
    width_smooth: int = 2
    thick_smooth: int = 2000
    temp_margin: int = 40
    ton_target: int = 20


@dataclass(frozen=True)
class ConstraintConfig:
    thresholds: ConstraintThresholds = ConstraintThresholds()
    weights: ConstraintWeights = ConstraintWeights()


# 统一默认配置：后续调参只需修改本文件
DEFAULT_CONSTRAINT_CONFIG = ConstraintConfig()
