import os
import sys

import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from aps_cp_sat.validate import validate_model_equivalence


def test_equivalence_validator_detects_chain_break():
    df = pd.DataFrame(
        [
            {"line": "big_roll", "campaign_id": 1, "campaign_seq": 1, "order_id": "A", "is_virtual": False, "width_jump_violation": False, "thickness_violation": False, "non_pc_direct_switch": False, "temp_conflict": False},
            {"line": "big_roll", "campaign_id": 1, "campaign_seq": 3, "order_id": "B", "is_virtual": False, "width_jump_violation": False, "thickness_violation": False, "non_pc_direct_switch": False, "temp_conflict": False},
        ]
    )
    out = validate_model_equivalence(df, None)
    assert out["campaign_single_chain_ok"] is False
    assert out["chain_break_cnt"] >= 1
