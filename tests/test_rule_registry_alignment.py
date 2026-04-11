import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from aps_cp_sat.io.result_writer import EXPORT_RULE_KEYS
from aps_cp_sat.rules import RULE_REGISTRY
from aps_cp_sat.validate.solution_validator import VALIDATION_RULE_KEYS


def test_validation_and_export_rule_keys_exist_in_registry():
    registry_keys = {key for key, _ in RULE_REGISTRY.items()}
    assert set(VALIDATION_RULE_KEYS.values()).issubset(registry_keys)
    assert set(EXPORT_RULE_KEYS).issubset(registry_keys)


def test_export_visible_rules_have_display_names():
    for spec in RULE_REGISTRY.export_visible_specs():
        assert spec.zh_name
        assert spec.en_name
