from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RuleConfig:
    # ------------------------ 温度 / 宽度 / 厚度类硬规则 ------------------------
    min_temp_overlap_real_real: float = 10.0  # 实物卷直接相邻时要求的最小温区重叠，单位为摄氏度。
    max_width_drop: float = 250.0  # 相邻实物卷允许的最大降宽幅度，单位为毫米。

    # 逆宽控制：真实卷之间保持严格；虚拟吸附场景允许同样的单步上限，避免前置层过松。
    real_reverse_step_max_mm: float = 50.0  # 实物卷相邻时允许的单步逆宽上限，单位为毫米。
    virtual_reverse_attach_max_mm: float = 250.0  # 虚拟卷吸附或虚拟链中允许的单步逆宽上限，单位为毫米。
    max_width_rise_physical_step: float = 50.0  # 兼容旧代码的逆宽上限别名，语义同 real_reverse_step_max_mm。

    # 每个轧期允许的“逻辑逆宽事件”个数，而不是把一条虚拟链中的每个虚拟卷都算一次。
    max_logical_reverse_per_campaign: int = 5  # 约束上限：max_logical_reverse_per_campaign。

    # ------------------------ 轧期吨位窗口 ------------------------
    campaign_ton_min: float = 700.0  # 正式可交付轧期最低吨位硬约束。
    campaign_ton_max: float = 2000.0  # 正式可交付轧期最高吨位硬约束。
    campaign_ton_target: float = 1800.0  # 评分使用的理想轧期吨位，不是硬约束。

    # ------------------------ 虚拟桥接 / 虚拟卷参数 ------------------------
    virtual_chain_penalty_threshold: int = 5  # 惩罚权重：virtual_chain_penalty_threshold。
    temp_margin_target: float = 30.0  # 参数：temp_margin_target，用于控制相关算法行为。

    # 虚拟吨位占比阈值：默认 20%（1/5）。
    virtual_ton_ratio_num: int = 1  # 比例参数：virtual_ton_ratio_num。
    virtual_ton_ratio_den: int = 5  # 比例参数：virtual_ton_ratio_den。

    # 当前业务允许的虚拟卷离散宽度与厚度规格。
    virtual_width_levels: tuple[float, ...] = (1000.0, 1250.0, 1500.0)  # 虚拟板坯相关参数：virtual_width_levels。
    virtual_thickness_levels: tuple[float, ...] = (0.6, 0.8, 1.0, 1.2, 1.5, 2.0)  # 虚拟板坯相关参数：virtual_thickness_levels。

    # 虚拟卷温区采用连续区间，不做离散温度档枚举。
    virtual_temp_min: float = 600.0  # 约束下限：virtual_temp_min。
    virtual_temp_max: float = 900.0  # 约束上限：virtual_temp_max。
    virtual_tons: float = 20.0  # 吨位参数：virtual_tons。

    # 模板层 / repair 层允许的最大虚拟链长度。
    max_virtual_chain: int = 5  # 约束上限：max_virtual_chain。
