from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConstraintThresholds:
    # ------------------------ 工艺阈值 ------------------------
    min_temp_overlap_real_real: float = 10.0  # 实物卷直接相邻时要求的最小温区重叠，单位为摄氏度。
    max_width_drop: float = 250.0  # 相邻实物卷允许的最大降宽幅度，单位为毫米。

    # 逆宽相关：真实卷与虚拟吸附的单步上升宽度上限。
    real_reverse_step_max_mm: float = 50.0  # 实物卷相邻时允许的单步逆宽上限，单位为毫米。
    virtual_reverse_attach_max_mm: float = 50.0  # 虚拟卷吸附或虚拟链中允许的单步逆宽上限，单位为毫米。
    max_width_rise_physical_step: float = 50.0  # 兼容旧代码的逆宽上限别名，语义同 real_reverse_step_max_mm。
    max_logical_reverse_per_campaign: int = 5  # 单个轧期允许的逻辑逆宽事件数量上限。

    # ------------------------ 轧期吨位窗口 ------------------------
    campaign_ton_min: float = 700.0  # 正式可交付轧期最低吨位硬约束。
    campaign_ton_max: float = 2000.0  # 正式可交付轧期最高吨位硬约束。
    campaign_ton_target: float = 1800.0  # 评分使用的理想轧期吨位，不是硬约束。

    # ------------------------ 虚拟桥接阈值 ------------------------
    virtual_chain_penalty_threshold: int = 5  # 超过该虚拟链长后加重惩罚。
    temp_margin_target: float = 30.0  # 温区富余度目标值。
    virtual_ton_ratio_num: int = 1  # 虚拟吨位占比上限的分子。
    virtual_ton_ratio_den: int = 5  # 虚拟吨位占比上限的分母。


@dataclass(frozen=True)
class ConstraintWeights:
    # ------------------------ 硬约束罚分 ------------------------
    seq_unique: int = 500  # 同一订单重复出现在多个位置时的惩罚。
    seq_contiguous: int = 300  # 序列索引不连续时的惩罚。
    seq_start: int = 300  # 序列起始位置异常时的惩罚。
    roll_line_mismatch: int = 10000  # 订单被分配到不兼容产线时的硬惩罚。
    ton_over: int = 10000  # 轧期总吨位超过上限时的硬惩罚。
    ton_under: int = 1000  # 轧期总吨位低于下限时的惩罚。
    temp_shortage: int = 10000  # 相邻温度区间重叠不足时的硬惩罚。
    width_violation: int = 10000  # 宽度规则违反时的硬惩罚。
    thick_violation: int = 10000  # 厚度规则违反时的硬惩罚。
    non_pc_switch: int = 10000  # 不合法跨钢种组切换时的硬惩罚。
    virtual_chain_excess: int = 10000  # 虚拟链长度超过上限时的硬惩罚。

    # ------------------------ 中约束 / 业务权衡 ------------------------
    unassigned_real: int = 300  # 实物订单未排入结果时的惩罚。
    virtual_ratio: int = 9000  # 虚拟吨位占比过高时的惩罚。
    pure_virtual_campaign: int = 300000  # 整段轧期全部由虚拟卷构成时的高额惩罚。
    virtual_use: int = 8  # 每使用一个虚拟卷的基础惩罚。

    # ------------------------ 软约束 ------------------------
    width_smooth: int = 2  # 宽度变化平滑性惩罚权重。
    thick_smooth: int = 2000  # 厚度变化平滑性惩罚权重。
    temp_margin: int = 40  # 温度富余度不足但未违规时的柔性惩罚。
    ton_target: int = 20  # 向目标吨位靠拢的引导权重。


@dataclass(frozen=True)
class ConstraintConfig:
    thresholds: ConstraintThresholds = ConstraintThresholds()  # 旧版兼容配置中的业务阈值集合。
    weights: ConstraintWeights = ConstraintWeights()  # 旧版兼容配置中的评分权重集合。


# 统一默认配置：后续调参优先修改本文件，再由兼容层映射到新 PlannerConfig。
DEFAULT_CONSTRAINT_CONFIG = ConstraintConfig()
