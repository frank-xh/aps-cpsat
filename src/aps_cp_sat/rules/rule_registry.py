from __future__ import annotations

from aps_cp_sat.rules.rule_key import RuleKey
from aps_cp_sat.rules.rule_spec import RuleSpec


class RuleRegistry:
    def __init__(self) -> None:
        specs = [
            RuleSpec(RuleKey.LINE_COMPATIBILITY, "产线兼容", "Line compatibility"),
            RuleSpec(RuleKey.TEMPERATURE_OVERLAP, "温度重叠", "Temperature overlap"),
            RuleSpec(RuleKey.WIDTH_TRANSITION, "宽度跳变", "Width transition"),
            RuleSpec(RuleKey.THICKNESS_TRANSITION, "厚度跳变", "Thickness transition"),
            RuleSpec(RuleKey.CROSS_GROUP_BRIDGE, "跨组过渡", "Cross-group bridge"),
            RuleSpec(RuleKey.CAMPAIGN_TON_MAX, "轧期吨位上限", "Campaign ton max"),
            RuleSpec(RuleKey.CAMPAIGN_TON_MIN, "轧期吨位下限", "Campaign ton min"),
            RuleSpec(RuleKey.REVERSE_WIDTH_COUNT, "逆宽次数", "Reverse-width count"),
            RuleSpec(RuleKey.REVERSE_WIDTH_TOTAL, "逆宽总量", "Reverse-width total"),
            RuleSpec(RuleKey.VIRTUAL_USAGE, "虚拟板坯使用", "Virtual usage"),
            RuleSpec(RuleKey.VIRTUAL_RATIO, "虚拟板坯占比", "Virtual ratio"),
        ]
        self._specs = {spec.key: spec for spec in specs}

    def get(self, key: RuleKey) -> RuleSpec:
        return self._specs[key]

    def items(self):
        return self._specs.items()

    def export_visible_specs(self):
        return [spec for spec in self._specs.values() if spec.export_visible]


RULE_REGISTRY = RuleRegistry()
