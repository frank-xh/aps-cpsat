from __future__ import annotations

from dataclasses import dataclass, field

from aps_cp_sat.config.model_config import ModelConfig
from aps_cp_sat.config.rule_config import RuleConfig
from aps_cp_sat.config.score_config import ScoreConfig


@dataclass(frozen=True)
class PlannerConfig:
    rule: RuleConfig = field(default_factory=RuleConfig)  # 工艺硬规则参数，控制宽度、厚度、温度、吨位、虚拟卷规格等硬约束口径。
    model: ModelConfig = field(default_factory=ModelConfig)  # 求解流程参数，控制路线选择、模板图、LNS、block-first、桥接、虚拟影子模式等算法行为。
    score: ScoreConfig = field(default_factory=ScoreConfig)  # 目标函数和惩罚参数，控制掉单、欠吨、虚拟使用、桥接、平滑性等评分权重。

    def __getattr__(self, name: str):
        # 历史兼容入口：旧代码可能直接访问 cfg.xxx。
        # 新代码应显式使用 cfg.model.* / cfg.rule.* / cfg.score.*，避免参数来源不清。
        if hasattr(self.model, name):
            return getattr(self.model, name)
        if hasattr(self.rule, name):
            return getattr(self.rule, name)
        if hasattr(self.score, name):
            return getattr(self.score, name)
        raise AttributeError(name)
