from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd


def run_repair_pipeline(final_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Repair layer contract.

    This pipeline is recording-only in the current architecture:
    - it does not change master-model feasibility semantics
    - it does not mutate schedule structure
    - any future step must be explicit and logged here
    """
    logs: List[Dict[str, object]] = []
    if final_df is None or final_df.empty:
        logs.append({"step": "repair_pipeline", "action": "skip_empty", "changed_rows": 0})
        return final_df, pd.DataFrame(logs)

    logs.append({"step": "repair_pipeline", "action": "no_op", "changed_rows": 0})
    return final_df, pd.DataFrame(logs)
