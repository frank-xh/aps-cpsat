from __future__ import annotations

import pandas as pd

from aps_cp_sat.config import PlannerConfig
from aps_cp_sat.decode.joint_solution_decoder import materialize_master_plan
from aps_cp_sat.domain.models import ColdRollingResult


def _encoded_virtual_count(encoded: str) -> int:
    return len([p for p in str(encoded).split("->") if str(p).strip()])


def decode_solution(result: ColdRollingResult) -> ColdRollingResult:
    """
    Decode layer is the single materialization entry.

    Responsibilities:
    - materialize raw joint-master output into business schedule rows
    - expand template bridge paths into virtual slab rows
    - keep already-materialized legacy output as pass-through

    Non-responsibilities:
    - no validation
    - no export
    - no post-solve feasibility repair
    """
    df = result.schedule_df if isinstance(result.schedule_df, pd.DataFrame) else pd.DataFrame()
    if df.empty:
        return result
    if "campaign_id" in df.columns:
        return result
    cfg = result.config or PlannerConfig()
    meta = dict(result.engine_meta or {})
    selected_pair_count = 0
    selected_template_virtual_count_total = 0
    max_template_virtual_count = 0
    selected_direct_edge_count = 0
    selected_real_bridge_edge_count = 0
    selected_virtual_bridge_edge_count = 0
    if "selected_bridge_path" in df.columns:
        counts = df["selected_bridge_path"].map(_encoded_virtual_count)
        selected_pair_count = int((counts > 0).sum())
        selected_template_virtual_count_total = int(counts.sum())
        max_template_virtual_count = int(counts.max()) if not counts.empty else 0
    if "selected_edge_type" in df.columns:
        selected_direct_edge_count = int((df["selected_edge_type"].astype(str) == "DIRECT_EDGE").sum())
        selected_real_bridge_edge_count = int((df["selected_edge_type"].astype(str) == "REAL_BRIDGE_EDGE").sum())
        selected_virtual_bridge_edge_count = int((df["selected_edge_type"].astype(str) == "VIRTUAL_BRIDGE_EDGE").sum())
    final_df, rounds_df, dropped_df = materialize_master_plan(
        df,
        cfg,
        pre_dropped=result.dropped_df if isinstance(result.dropped_df, pd.DataFrame) else None,
    )
    decoded_virtual_cnt = int(final_df["is_virtual"].sum()) if not final_df.empty and "is_virtual" in final_df.columns else 0
    joint_estimates = dict(meta.get("joint_estimates", {}) or {})
    meta["bridgeTemplateSelectedPairCount"] = int(selected_pair_count)
    meta["selectedTemplateVirtualCountTotal"] = int(selected_template_virtual_count_total)
    meta["averageTemplateVirtualCount"] = float(selected_template_virtual_count_total / max(1, selected_pair_count)) if selected_pair_count > 0 else 0.0
    meta["maxTemplateVirtualCount"] = int(max_template_virtual_count)
    meta["decodedTemplatePairCount"] = int(selected_pair_count)
    meta["decodedTemplateVirtualCoilCount"] = int(decoded_virtual_cnt)
    meta["decodedAverageVirtualCountPerPair"] = float(decoded_virtual_cnt / max(1, selected_pair_count)) if selected_pair_count > 0 else 0.0
    meta["selected_direct_edge_count"] = int(selected_direct_edge_count)
    meta["selected_real_bridge_edge_count"] = int(selected_real_bridge_edge_count)
    meta["selected_virtual_bridge_edge_count"] = int(selected_virtual_bridge_edge_count)
    total_selected_edges = max(1, selected_direct_edge_count + selected_real_bridge_edge_count + selected_virtual_bridge_edge_count)
    meta["direct_edge_ratio"] = float(selected_direct_edge_count / total_selected_edges)
    meta["real_bridge_ratio"] = float(selected_real_bridge_edge_count / total_selected_edges)
    meta["virtual_bridge_ratio"] = float(selected_virtual_bridge_edge_count / total_selected_edges)
    joint_estimates.update(
        {
            "bridgeTemplateSelectedPairCount": int(selected_pair_count),
            "selectedTemplateVirtualCountTotal": int(selected_template_virtual_count_total),
            "averageTemplateVirtualCount": float(selected_template_virtual_count_total / max(1, selected_pair_count)) if selected_pair_count > 0 else 0.0,
            "maxTemplateVirtualCount": int(max_template_virtual_count),
            "decodedTemplatePairCount": int(selected_pair_count),
            "decodedTemplateVirtualCoilCount": int(decoded_virtual_cnt),
            "decodedAverageVirtualCountPerPair": float(decoded_virtual_cnt / max(1, selected_pair_count)) if selected_pair_count > 0 else 0.0,
            "selected_direct_edge_count": int(selected_direct_edge_count),
            "selected_real_bridge_edge_count": int(selected_real_bridge_edge_count),
            "selected_virtual_bridge_edge_count": int(selected_virtual_bridge_edge_count),
            "direct_edge_ratio": float(selected_direct_edge_count / total_selected_edges),
            "real_bridge_ratio": float(selected_real_bridge_edge_count / total_selected_edges),
            "virtual_bridge_ratio": float(selected_virtual_bridge_edge_count / total_selected_edges),
        }
    )
    meta["joint_estimates"] = joint_estimates
    return ColdRollingResult(
        schedule_df=final_df,
        rounds_df=rounds_df if not rounds_df.empty else result.rounds_df,
        output_path=result.output_path,
        dropped_df=dropped_df,
        engine_meta=meta,
        config=result.config,
    )
