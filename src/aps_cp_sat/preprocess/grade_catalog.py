from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict


def _txt(v, default: str = "") -> str:
    if v is None:
        return default
    s = str(v).strip()
    return s if s else default


@dataclass(frozen=True)
class SteelSpec:
    steel_group: str
    roll_capability: str
    priority: int


class SteelSpecCatalog:
    SPECS: Dict[str, SteelSpec] = {
        "DC01": SteelSpec("PC", "dual", 1),
        "DC01-D": SteelSpec("PC", "dual", 1),
        "SPCC": SteelSpec("PC", "dual", 1),
        "DC04": SteelSpec("IF", "dual", 2),
        "DC05": SteelSpec("IF", "dual", 2),
        "DC06": SteelSpec("IF", "dual", 2),
        "T1500HS": SteelSpec("HOTFORM", "small_only", 3),
        "T280VK": SteelSpec("DP", "small_only", 3),
        "HC180Y": SteelSpec("BH", "small_only", 3),
        "HC340/590DP": SteelSpec("DP", "small_only", 3),
    }

    @classmethod
    def get(cls, grade: str) -> SteelSpec | None:
        return cls.SPECS.get(_txt(grade).upper())


class MergedGradeRuleCatalog:
    """
    统一钢种规则目录（已固化快照）：
    本字典由“先读文件 + 合并硬编码参数”生成后写入代码，后续直接查字典。
    """

    RULES: Dict[str, dict] = {
        '180Y': {'steel_group': '加磷高强、烘烤硬化钢', 'roll_capability': 'dual', 'priority': 1, 'source': 'data'},
        '210P1': {'steel_group': '加磷高强、烘烤硬化钢', 'roll_capability': 'dual', 'priority': 1, 'source': 'data'},
        '280VK': {'steel_group': '双相钢', 'roll_capability': 'dual', 'priority': 1, 'source': 'data'},
        'CR300LA': {'steel_group': '低合金高强', 'roll_capability': 'dual', 'priority': 1, 'source': 'data'},
        'CR330Y590T-DP': {'steel_group': '双相钢', 'roll_capability': 'dual', 'priority': 1, 'source': 'data'},
        'CR340/590DP': {'steel_group': '双相钢', 'roll_capability': 'dual', 'priority': 1, 'source': 'data'},
        'CR420LA': {'steel_group': '低合金高强', 'roll_capability': 'dual', 'priority': 1, 'source': 'data'},
        'DC01': {'steel_group': '普碳', 'roll_capability': 'dual', 'priority': 1, 'source': 'data+catalog'},
        'DC01-D': {'steel_group': '普碳', 'roll_capability': 'dual', 'priority': 1, 'source': 'data+catalog'},
        'DC03': {'steel_group': 'IF钢', 'roll_capability': 'dual', 'priority': 1, 'source': 'data'},
        'DC03EK': {'steel_group': 'IF钢', 'roll_capability': 'dual', 'priority': 1, 'source': 'data'},
        'DC04': {'steel_group': 'IF钢', 'roll_capability': 'dual', 'priority': 2, 'source': 'data+catalog'},
        'DC05': {'steel_group': 'IF钢', 'roll_capability': 'dual', 'priority': 2, 'source': 'data+catalog'},
        'DC06': {'steel_group': 'IF钢', 'roll_capability': 'dual', 'priority': 2, 'source': 'data+catalog'},
        'HC180B': {'steel_group': '加磷高强、烘烤硬化钢', 'roll_capability': 'dual', 'priority': 1, 'source': 'data'},
        'HC180Y': {'steel_group': '加磷高强、烘烤硬化钢', 'roll_capability': 'small_only', 'priority': 3, 'source': 'data+catalog'},
        'HC220B': {'steel_group': '加磷高强、烘烤硬化钢', 'roll_capability': 'dual', 'priority': 1, 'source': 'data'},
        'HC220Y': {'steel_group': '加磷高强、烘烤硬化钢', 'roll_capability': 'dual', 'priority': 1, 'source': 'data'},
        'HC260/450DP': {'steel_group': '双相钢', 'roll_capability': 'dual', 'priority': 1, 'source': 'data'},
        'HC260LA': {'steel_group': '低合金高强', 'roll_capability': 'dual', 'priority': 1, 'source': 'data'},
        'HC260Y': {'steel_group': '加磷高强、烘烤硬化钢', 'roll_capability': 'dual', 'priority': 1, 'source': 'data'},
        'HC300/500DP': {'steel_group': '双相钢', 'roll_capability': 'dual', 'priority': 1, 'source': 'data'},
        'HC300LA': {'steel_group': '低合金高强', 'roll_capability': 'dual', 'priority': 1, 'source': 'data'},
        'HC340/590DP': {'steel_group': '双相钢', 'roll_capability': 'small_only', 'priority': 3, 'source': 'data+catalog'},
        'HC340LA': {'steel_group': '低合金高强', 'roll_capability': 'dual', 'priority': 1, 'source': 'data'},
        'HC380LA': {'steel_group': '低合金高强', 'roll_capability': 'dual', 'priority': 1, 'source': 'data'},
        'HC420/780DP': {'steel_group': '顶级牌号780以上', 'roll_capability': 'dual', 'priority': 1, 'source': 'data'},
        'HC420LA': {'steel_group': '低合金高强', 'roll_capability': 'dual', 'priority': 1, 'source': 'data'},
        'HC550/980DP': {'steel_group': '顶级牌号780以上', 'roll_capability': 'dual', 'priority': 1, 'source': 'data'},
        'HC600/980QP': {'steel_group': '顶级牌号780以上', 'roll_capability': 'dual', 'priority': 1, 'source': 'data'},
        'HC700/980DP': {'steel_group': '顶级牌号780以上', 'roll_capability': 'dual', 'priority': 1, 'source': 'data'},
        'HC820/1180DP': {'steel_group': '顶级牌号780以上', 'roll_capability': 'dual', 'priority': 1, 'source': 'data'},
        'Q235': {'steel_group': '低合金1', 'roll_capability': 'dual', 'priority': 1, 'source': 'data'},
        'Q235A': {'steel_group': '低合金高强', 'roll_capability': 'dual', 'priority': 1, 'source': 'data'},
        'Q235B': {'steel_group': '低合金1', 'roll_capability': 'dual', 'priority': 1, 'source': 'data'},
        'SPA-C': {'steel_group': '结构钢', 'roll_capability': 'dual', 'priority': 1, 'source': 'data'},
        'SPCC': {'steel_group': '普碳', 'roll_capability': 'dual', 'priority': 1, 'source': 'data+catalog'},
        'SPCC-SD': {'steel_group': '普碳', 'roll_capability': 'dual', 'priority': 1, 'source': 'data'},
        'ST13': {'steel_group': 'IF钢', 'roll_capability': 'dual', 'priority': 1, 'source': 'data'},
        'ST14': {'steel_group': 'IF钢', 'roll_capability': 'dual', 'priority': 1, 'source': 'data'},
        'ST16': {'steel_group': 'IF钢', 'roll_capability': 'dual', 'priority': 1, 'source': 'data'},
        'ST37-2G': {'steel_group': '低合金高强', 'roll_capability': 'dual', 'priority': 1, 'source': 'data'},
        'T1500HS': {'steel_group': '热冲压钢', 'roll_capability': 'small_only', 'priority': 3, 'source': 'data+catalog'},
        'T170P1': {'steel_group': '加磷高强、烘烤硬化钢', 'roll_capability': 'dual', 'priority': 1, 'source': 'data'},
        'T210P1': {'steel_group': '加磷高强、烘烤硬化钢', 'roll_capability': 'dual', 'priority': 1, 'source': 'data'},
        'T250P1': {'steel_group': '加磷高强、烘烤硬化钢', 'roll_capability': 'dual', 'priority': 1, 'source': 'data'},
        'T280VK': {'steel_group': '双相钢', 'roll_capability': 'small_only', 'priority': 3, 'source': 'data+catalog'},
        'THS2-TY': {'steel_group': '普碳', 'roll_capability': 'dual', 'priority': 1, 'source': 'data'},
        'THS3-TY': {'steel_group': '普碳', 'roll_capability': 'dual', 'priority': 1, 'source': 'data'},
        'TLA': {'steel_group': '普碳', 'roll_capability': 'dual', 'priority': 1, 'source': 'data'},
        'TTC1': {'steel_group': 'IF钢', 'roll_capability': 'dual', 'priority': 1, 'source': 'data'},
        'TYH': {'steel_group': '普碳', 'roll_capability': 'dual', 'priority': 1, 'source': 'data'},
        'TZT1': {'steel_group': '普碳', 'roll_capability': 'dual', 'priority': 1, 'source': 'data'},
        'TZT2': {'steel_group': '普碳', 'roll_capability': 'dual', 'priority': 1, 'source': 'data'},
    }

    def __init__(self, rules: Dict[str, dict] | None = None):
        self.rules = rules or dict(self.RULES)

    @classmethod
    def build(cls, steel_info_path: Path) -> "MergedGradeRuleCatalog":
        _ = steel_info_path
        return cls(dict(cls.RULES))

    def get(self, grade: str) -> dict:
        g = _txt(grade).upper()
        return self.rules.get(g, {"steel_group": "UNKNOWN", "roll_capability": "dual", "priority": 1, "source": "default"})


__all__ = ["MergedGradeRuleCatalog", "SteelSpec", "SteelSpecCatalog"]
