import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from aps_cp_sat.config import RuleConfig
from aps_cp_sat.transition.bridge_rules import _width_reverse_virtual_need


def test_reverse_width_need_has_minimum_one_bridge():
    rule = RuleConfig()
    need = _width_reverse_virtual_need(1000.0, 1100.0, rule)
    assert need >= 1
    assert need == 1


def test_reverse_width_need_supports_strict_level_mode():
    rule = RuleConfig()
    loose = _width_reverse_virtual_need(1000.0, 1180.0, rule, strict_virtual_width_levels=False)
    strict = _width_reverse_virtual_need(1000.0, 1180.0, rule, strict_virtual_width_levels=True)
    assert strict >= loose
