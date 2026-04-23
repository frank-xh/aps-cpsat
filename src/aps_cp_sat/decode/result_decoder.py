from __future__ import annotations

from typing import List

import pandas as pd

from aps_cp_sat.config import PlannerConfig
from aps_cp_sat.decode.joint_solution_decoder import materialize_master_plan
from aps_cp_sat.domain.models import ColdRollingResult


def _encoded_virtual_count(encoded: str) -> int:
    return len([p for p in str(encoded).split("->") if str(p).strip()])


def _is_constructive_lns_path(df: pd.DataFrame, meta: dict) -> bool:
    """Detect if df comes from the constructive_lns or block_first path (not joint_master)."""
    if str(meta.get("engine_used", "")) == "constructive_lns":
        return True
    if str(meta.get("main_path", "")) == "constructive_lns":
        return True
    # block_first uses a lightweight path (no materialize_master_plan needed)
    if str(meta.get("solver_path", "")) == "block_first":
        return True
    if str(meta.get("main_solver_strategy", "")) == "block_first":
        return True
    # Fallback: constructive_lns planned_df always has campaign_id_hint but no campaign_id
    if "campaign_id_hint" in df.columns and "campaign_id" not in df.columns:
        return True
    return False


def _get_bridge_expansion_mode(meta: dict) -> str:
    """
    Get bridge_expansion_mode from engine metadata.

    Returns:
        "disabled" (default), "virtual_expand", or other future modes.
    """
    # Check multiple possible locations for bridge_expansion_mode. Top-level
    # engine_meta is the stable source of truth after pipeline normalization.
    if str(meta.get("bridge_expansion_mode", "") or "").strip():
        return str(meta["bridge_expansion_mode"])
    if "lns_engine_meta" in meta and isinstance(meta["lns_engine_meta"], dict):
        return str(meta["lns_engine_meta"].get("bridge_expansion_mode", "disabled"))
    if "lns_diagnostics" in meta and isinstance(meta["lns_diagnostics"], dict):
        return str(meta["lns_diagnostics"].get("bridge_expansion_mode", "disabled"))
    return "disabled"


def _read_bridge_metadata_fields(df: pd.DataFrame) -> dict:
    """
    Read unified bridge metadata fields from planned_df.

    Returns dict with:
        - has_edge_type: bool
        - has_bridge_expandable: bool
        - has_bridge_expand_mode: bool
        - has_virtual_bridge_count: bool
        - has_real_bridge_order_id: bool
        - virtual_bridge_count_in_df: int
        - real_bridge_edge_count: int
        - direct_edge_count: int
    """
    result = {
        "has_edge_type": "selected_edge_type" in df.columns,
        "has_bridge_expandable": "selected_bridge_expandable" in df.columns,
        "has_bridge_expand_mode": "selected_bridge_expand_mode" in df.columns,
        "has_virtual_bridge_count": "selected_virtual_bridge_count" in df.columns,
        "has_real_bridge_order_id": "selected_real_bridge_order_id" in df.columns,
        "virtual_bridge_count_in_df": 0,
        "real_bridge_edge_count": 0,
        "direct_edge_count": 0,
    }

    if df.empty:
        return result

    # Count by edge type
    if "selected_edge_type" in df.columns:
        edge_types = df["selected_edge_type"].astype(str)
        result["direct_edge_count"] = int((edge_types == "DIRECT_EDGE").sum())
        result["real_bridge_edge_count"] = int((edge_types == "REAL_BRIDGE_EDGE").sum())
        result["virtual_bridge_count_in_df"] = int((edge_types == "VIRTUAL_BRIDGE_EDGE").sum())

    return result


def _lightweight_decode_constructive_lns(
    df: pd.DataFrame,
    result: ColdRollingResult,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """
    Lightweight decode for constructive_lns path.

    The constructive_lns output already has:
    - line, master_slot, master_seq
    - campaign_id_hint, campaign_seq_hint
    - selected_template_id, selected_bridge_path
    - selected_edge_type, selected_bridge_expandable, selected_bridge_expand_mode
    - selected_virtual_bridge_count, selected_real_bridge_order_id

    We only need to:
    1. Read bridge_expansion_mode from meta
    2. Handle bridge expansion based on mode
    3. Rename campaign_id_hint -> campaign_id
    4. Add seq column (1-based within campaign) from campaign_seq_hint
    5. Pre-sort order consistency check: seq vs master_seq must agree (fail-fast)
    6. Demote corrupted campaigns to decode_demoted_df
    7. Sort clean campaigns into final order
    8. Add global sequence indices
    9. Map line to Chinese name

    Returns:
        final_df: clean schedule rows (corrupted campaigns removed)
        rounds_df: pass-through rounds
        dropped_df: pass-through dropped rows
        decode_meta: structured mismatch metadata
    """
    meta = dict(result.engine_meta or {})

    df = df.copy()

    # 1. Get bridge_expansion_mode
    bridge_expand_mode = _get_bridge_expansion_mode(meta)

    # 2. Handle bridge expansion based on mode
    if bridge_expand_mode == "disabled":
        # In constructive + prebuilt-virtual-inventory mode, rows may already
        # carry real/virtual identity from the graph. Preserve it instead of
        # forcibly demoting all rows to real.
        if "is_virtual" in df.columns:
            df["is_virtual"] = df["is_virtual"].fillna(False).astype(bool)
        else:
            df["is_virtual"] = False
        # Ensure bridge metadata fields exist (even if disabled)
        for col in [
            "selected_edge_type", "selected_bridge_expandable",
            "selected_bridge_expand_mode", "selected_virtual_bridge_count",
            "selected_real_bridge_order_id",
        ]:
            if col not in df.columns:
                if col == "selected_edge_type":
                    df[col] = "DIRECT_EDGE"
                elif col == "selected_bridge_expandable":
                    df[col] = False
                elif col == "selected_bridge_expand_mode":
                    df[col] = "disabled"
                elif col == "selected_virtual_bridge_count":
                    df[col] = 0
                elif col == "selected_real_bridge_order_id":
                    df[col] = ""

    elif bridge_expand_mode == "virtual_expand":
        # Route A: Virtual expansion (future implementation)
        # For now, fall back to disabled behavior
        df["is_virtual"] = False
        for col in ["selected_bridge_expand_mode"]:
            if col not in df.columns:
                df[col] = "virtual_expand"  # Mark as expandable but not expanded yet

    else:
        # Unknown mode: default to disabled behavior
        df["is_virtual"] = False
        if "selected_bridge_expand_mode" not in df.columns:
            df["selected_bridge_expand_mode"] = bridge_expand_mode

    # 3. Use campaign_id_hint as campaign_id
    if "campaign_id_hint" in df.columns:
        df["campaign_id"] = df["campaign_id_hint"]

    # 4. Add seq column (1-based within campaign)
    if "campaign_seq_hint" in df.columns:
        df["seq"] = df["campaign_seq_hint"]
    elif "master_seq" in df.columns:
        df["seq"] = df["master_seq"]

    # 4b. Pre-sort order consistency check: seq vs master_seq must agree
    # This catches data pipeline bugs where the two seq fields drift apart.
    decode_order_mismatch_examples: List[dict] = []
    # Store corrupted campaign info: each entry is {"camp_key": ..., "line": ..., "campaign_id": ...}
    decode_order_corrupted_campaigns: List[dict] = []
    if not df.empty and "master_seq" in df.columns:
        for (line_val, camp_val), grp in df.groupby(["line", "campaign_id"], dropna=False):
            seq_list = grp["seq"].tolist()
            master_seq_list = grp["master_seq"].tolist()
            if seq_list != master_seq_list:
                mismatch_count = sum(1 for s, m in zip(seq_list, master_seq_list) if s != m)
                camp_key = f"{line_val}_{camp_val}"
                decode_order_corrupted_campaigns.append({"camp_key": camp_key, "line": line_val, "campaign_id": camp_val})
                # Record up to 5 sample rows per corrupted campaign
                for i, (_, row) in enumerate(grp.iterrows()):
                    if i >= 5:
                        break
                    example = {
                        "line": line_val,
                        "campaign_id": camp_val,
                        "order_id": row.get("order_id", ""),
                        "seq": row.get("seq", 0),
                        "master_seq": row.get("master_seq", 0),
                        "campaign_seq_hint": row.get("campaign_seq_hint", 0),
                        "campaign_key": camp_key,
                        "mismatch_count": mismatch_count,
                    }
                    decode_order_mismatch_examples.append(example)
                print(
                    f"[APS][decode_order_mismatch] line={line_val}, campaign_id={camp_val}, "
                    f"mismatch_count={mismatch_count}, "
                    f"order_ids={grp['order_id'].tolist()}, "
                    f"seq_list={seq_list}, master_seq_list={master_seq_list}"
                )

    # 4c. Build decode_meta (returned for pipeline-level handling)
    # NOTE: decode_demoted_df must be initialized BEFORE this dict so len() is safe
    decode_demoted_df: pd.DataFrame = pd.DataFrame()
    # campaign keys as plain strings for mask filtering
    corrupted_camp_keys = [c["camp_key"] for c in decode_order_corrupted_campaigns]
    decode_meta = {
        "decode_order_mismatch_count": len(decode_order_corrupted_campaigns),
        "decode_order_mismatch_campaigns": corrupted_camp_keys,
        "decode_order_mismatch_examples": decode_order_mismatch_examples[:20],
        "decode_order_integrity_ok": len(decode_order_corrupted_campaigns) == 0,
        # demoted order count: 0 if clean, else the number of rows removed
        "decode_demoted_order_count": 0,
    }

    # 5. Demote corrupted campaigns: remove their rows from df, collect into demoted_df
    if corrupted_camp_keys:
        # Build campaign key for each row
        df["_campaign_key"] = df["line"].astype(str) + "_" + df["campaign_id"].astype(str)
        corrupted_mask = df["_campaign_key"].isin(corrupted_camp_keys)
        decode_demoted_df = df[corrupted_mask].copy()
        df = df[~corrupted_mask].copy()
        df.drop(columns=["_campaign_key"], inplace=True)
        decode_demoted_df.drop(columns=["_campaign_key"], inplace=True)

        # Add drop_reason to demoted rows
        decode_demoted_df["drop_reason"] = "DECODE_ORDER_MISMATCH"
        decode_demoted_df["decode_demotion_stage"] = "decode"

        # Update decode_meta with actual demoted count
        decode_meta["decode_demoted_order_count"] = len(decode_demoted_df)

        # Log per-campaign (not per-row) summary
        for camp_info in decode_order_corrupted_campaigns:
            camp_orders = decode_demoted_df[
                decode_demoted_df["campaign_id"] == camp_info["campaign_id"]
            ]["order_id"].tolist()
            print(
                f"[APS][decode_fail_fast] campaign={camp_info['camp_key']}, "
                f"mismatch_count={len(corrupted_camp_keys)}, "
                f"demoted_orders={camp_orders[:10]}"
            )

        print(
            f"[APS][decode_fail_fast] total demoted campaigns={len(decode_order_corrupted_campaigns)}, "
            f"total demoted orders={len(decode_demoted_df)}"
        )

    # 6. Sort clean df into final output order: line -> campaign_id -> seq
    # Priority order: seq (primary, from campaign_seq_hint) > master_slot > master_seq
    # Using kind="mergesort" ensures stable sort — ties are broken by input order,
    # which already preserves the original planned_df construction order.
    sort_cols = [c for c in ["line", "campaign_id", "seq", "master_slot", "master_seq"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    # 7. Add global sequence indices (derived from final sorted order, NOT sort keys)
    df["global_seq"] = df.index + 1
    df["line_seq"] = df.groupby("line", sort=False).cumcount() + 1
    df["campaign_seq"] = df.groupby(["line", "campaign_id"], sort=False).cumcount() + 1
    df["campaign_real_seq"] = df["campaign_seq"]

    # 8. Map line to Chinese name
    df["line_name"] = df["line"].map({"big_roll": "大辊线", "small_roll": "小辊线"}).fillna(df["line"])

    # 9. Ensure required bridge columns exist (even if empty)
    for col in ["selected_bridge_path", "selected_template_id", "force_break_before"]:
        if col not in df.columns:
            df[col] = ""

    # 10. dropped_df passes through unchanged
    dropped = result.dropped_df if isinstance(result.dropped_df, pd.DataFrame) else pd.DataFrame()

    # 11. rounds_df passes through
    rounds = result.rounds_df if isinstance(result.rounds_df, pd.DataFrame) else pd.DataFrame()

    return df, rounds, dropped, decode_meta


def decode_solution(result: ColdRollingResult) -> ColdRollingResult:
    """
    Decode layer is the single materialization entry.

    Responsibilities:
    - materialize raw joint-master output into business schedule rows
    - expand template bridge paths into virtual slab rows
    - keep already-materialized legacy output as pass-through
    - lightweight path for constructive_lns (no joint_master candidate_plan_df dependency)

    Non-responsibilities:
    - no validation
    - no export
    - no post-solve feasibility repair
    """
    df = result.schedule_df if isinstance(result.schedule_df, pd.DataFrame) else pd.DataFrame()
    meta = dict(result.engine_meta or {})

    if df.empty:
        return result

    # Already materialized (joint_master or prior decode pass)
    if "campaign_id" in df.columns:
        return result

    # -------------------------------------------------------------------------
    # Lightweight path: constructive_lns bypasses materialize_master_plan
    # because it has no candidate_plan_df / row_idx dependency.
    # -------------------------------------------------------------------------
    decode_meta: dict = {}
    if _is_constructive_lns_path(df, meta):
        final_df, rounds_df, dropped_df, decode_meta = _lightweight_decode_constructive_lns(df, result)

        # Merge decode-demoted rows into dropped_df with DECODE_ORDER_MISMATCH reason
        # Use campaign_id_hint (original column before lightweight decode renamed to campaign_id)
        if decode_meta.get("decode_order_mismatch_count", 0) > 0:
            corrupted_campaigns = decode_meta.get("decode_order_mismatch_campaigns", [])
            if corrupted_campaigns:
                # Re-identify demoted rows from the ORIGINAL df (pre-decode, has campaign_id_hint)
                df_src = result.schedule_df if isinstance(result.schedule_df, pd.DataFrame) else pd.DataFrame()
                if "campaign_id" in df_src.columns and "campaign_id_hint" not in df_src.columns:
                    df_src = df_src.rename(columns={"campaign_id": "campaign_id_hint"})
                df_src = df_src.copy()
                df_src["_campaign_key"] = df_src["line"].astype(str) + "_" + df_src["campaign_id_hint"].astype(str)
                demoted_mask = df_src["_campaign_key"].isin(corrupted_campaigns)
                decode_demoted_df = df_src[demoted_mask].copy().drop(columns=["_campaign_key"])
                decode_demoted_df["drop_reason"] = "DECODE_ORDER_MISMATCH"
                decode_demoted_df["decode_demotion_stage"] = "decode"
                decode_demoted_df["dominant_drop_reason"] = "DECODE_ORDER_MISMATCH"
                # Ensure dropped_df has drop_reason column
                if dropped_df.empty:
                    dropped_df = decode_demoted_df
                else:
                    if "drop_reason" not in dropped_df.columns:
                        dropped_df["drop_reason"] = ""
                    common_cols = [c for c in dropped_df.columns if c in decode_demoted_df.columns]
                    dropped_df = pd.concat([dropped_df[common_cols], decode_demoted_df[common_cols]], ignore_index=True)
            print(
                f"[APS][decode_gate] decode_order_integrity_ok=False, "
                f"decode_order_mismatch_count={decode_meta['decode_order_mismatch_count']}, "
                f"campaigns={decode_meta['decode_order_mismatch_campaigns']}"
            )
        else:
            # Clean decode: no order mismatch
            print(
                f"[APS][decode_gate] decode_order_integrity_ok=True, "
                f"decode_order_mismatch_count=0"
            )

        # Propagate decode mismatch metadata into engine_meta
        meta.update({
            "decode_order_mismatch_count": decode_meta.get("decode_order_mismatch_count", 0),
            "decode_order_mismatch_campaigns": decode_meta.get("decode_order_mismatch_campaigns", []),
            "decode_order_mismatch_examples": decode_meta.get("decode_order_mismatch_examples", []),
            "decode_order_integrity_ok": decode_meta.get("decode_order_integrity_ok", True),
            "decode_demoted_order_count": decode_meta.get("decode_demoted_order_count", 0),
        })

        # Get bridge_expansion_mode
        bridge_expand_mode = _get_bridge_expansion_mode(meta)

        # Read bridge metadata from unified fields
        bridge_meta = _read_bridge_metadata_fields(final_df)

        # Compute bridge statistics for meta
        selected_pair_count = 0
        if "selected_bridge_path" in final_df.columns:
            # Count non-empty bridge paths
            bridge_paths = final_df["selected_bridge_path"].astype(str)
            selected_pair_count = int((bridge_paths.str.len() > 0).sum())

        # Use unified fields if available, otherwise fallback to counting
        if bridge_meta["has_edge_type"]:
            selected_virtual_bridge_edge_count = bridge_meta["virtual_bridge_count_in_df"]
            selected_real_bridge_edge_count = bridge_meta["real_bridge_edge_count"]
            selected_direct_edge_count = bridge_meta["direct_edge_count"]
        else:
            # Fallback: count from selected_bridge_path
            counts = final_df["selected_bridge_path"].map(_encoded_virtual_count)
            selected_virtual_bridge_edge_count = int((counts > 0).sum())
            selected_real_bridge_edge_count = 0
            selected_direct_edge_count = int(len(final_df) - selected_virtual_bridge_edge_count)

        # decoded virtual coil count: depends on bridge_expansion_mode
        if bridge_expand_mode == "disabled":
            decoded_virtual_coil_count = 0
        else:
            # Future Route A: count from selected_virtual_bridge_count
            if bridge_meta["has_virtual_bridge_count"]:
                decoded_virtual_coil_count = int(final_df["selected_virtual_bridge_count"].sum())
            else:
                decoded_virtual_coil_count = 0

        joint_estimates = dict(meta.get("joint_estimates", {}) or {})
        joint_estimates.update({
            "bridgeTemplateSelectedPairCount": selected_pair_count,
            "decodedTemplatePairCount": selected_pair_count,
            "decodedTemplateVirtualCoilCount": decoded_virtual_coil_count,
            "selected_direct_edge_count": max(0, selected_direct_edge_count),
            "selected_virtual_bridge_edge_count": selected_virtual_bridge_edge_count,
            "selected_real_bridge_edge_count": selected_real_bridge_edge_count,
            "bridge_expansion_mode": bridge_expand_mode,
        })
        meta.update({
            "bridgeTemplateSelectedPairCount": selected_pair_count,
            "decodedTemplatePairCount": selected_pair_count,
            "decodedTemplateVirtualCoilCount": decoded_virtual_coil_count,
            "selected_direct_edge_count": max(0, selected_direct_edge_count),
            "selected_virtual_bridge_edge_count": selected_virtual_bridge_edge_count,
            "selected_real_bridge_edge_count": selected_real_bridge_edge_count,
            "direct_edge_ratio": float(max(0, selected_direct_edge_count) / max(1, len(final_df) - 1)),
            "virtual_bridge_ratio": float(selected_virtual_bridge_edge_count / max(1, len(final_df) - 1)),
            "real_bridge_ratio": float(selected_real_bridge_edge_count / max(1, len(final_df) - 1)),
            "bridge_expansion_mode": bridge_expand_mode,
            "lns_lightweight_decode": True,
            "decode_order_mismatch_count": decode_meta.get("decode_order_mismatch_count", 0),
            "decode_order_mismatch_campaigns": decode_meta.get("decode_order_mismatch_campaigns", []),
            "decode_order_mismatch_examples": decode_meta.get("decode_order_mismatch_examples", []),
            "decode_order_integrity_ok": decode_meta.get("decode_order_integrity_ok", True),
            "decode_demoted_order_count": decode_meta.get("decode_demoted_order_count", 0),
        })
        meta["joint_estimates"] = joint_estimates
        return ColdRollingResult(
            schedule_df=final_df,
            rounds_df=rounds_df if not rounds_df.empty else result.rounds_df,
            output_path=result.output_path,
            dropped_df=dropped_df,
            engine_meta=meta,
            config=result.config,
        )
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
        selected_virtual_bridge_family_edge_count = int((df["selected_edge_type"].astype(str) == "VIRTUAL_BRIDGE_FAMILY_EDGE").sum())
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
    total_selected_edges = max(1, selected_direct_edge_count + selected_real_bridge_edge_count + selected_virtual_bridge_edge_count + selected_virtual_bridge_family_edge_count)
    meta["direct_edge_ratio"] = float(selected_direct_edge_count / total_selected_edges)
    meta["real_bridge_ratio"] = float(selected_real_bridge_edge_count / total_selected_edges)
    meta["virtual_bridge_ratio"] = float(selected_virtual_bridge_edge_count / total_selected_edges)
    meta["virtual_bridge_family_ratio"] = float(selected_virtual_bridge_family_edge_count / total_selected_edges)
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
            "selected_virtual_bridge_family_edge_count": int(selected_virtual_bridge_family_edge_count),
            "direct_edge_ratio": float(selected_direct_edge_count / total_selected_edges),
            "real_bridge_ratio": float(selected_real_bridge_edge_count / total_selected_edges),
            "virtual_bridge_ratio": float(selected_virtual_bridge_edge_count / total_selected_edges),
            "virtual_bridge_family_ratio": float(selected_virtual_bridge_family_edge_count / total_selected_edges),
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
