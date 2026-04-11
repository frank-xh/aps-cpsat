from __future__ import annotations

from dataclasses import dataclass

from aps_cp_sat.rules.rule_key import RuleKey


@dataclass(frozen=True)
class RuleSpec:
    key: RuleKey
    zh_name: str
    en_name: str
    solver_active: bool = True
    validation_active: bool = True
    export_visible: bool = True
