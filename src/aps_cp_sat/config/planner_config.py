from __future__ import annotations

from dataclasses import dataclass, field

from aps_cp_sat.config.model_config import ModelConfig
from aps_cp_sat.config.rule_config import RuleConfig
from aps_cp_sat.config.score_config import ScoreConfig


@dataclass(frozen=True)
class PlannerConfig:
    rule: RuleConfig = field(default_factory=RuleConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    score: ScoreConfig = field(default_factory=ScoreConfig)

    def __getattr__(self, name: str):
        # Historical compatibility only. New code should use explicit access:
        # cfg.model.*, cfg.rule.*, cfg.score.*
        if hasattr(self.model, name):
            return getattr(self.model, name)
        if hasattr(self.rule, name):
            return getattr(self.rule, name)
        if hasattr(self.score, name):
            return getattr(self.score, name)
        raise AttributeError(name)
