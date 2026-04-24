from __future__ import annotations

"""
Validate layer contract.

This module only checks solved plans and produces summaries. It must not mutate
schedules, inject bridge rows, or repair feasibility.
"""

from typing import Dict
import math

import pandas as pd

from aps_cp_sat.config import RuleConfig
from aps_cp_sat.domain.models import ColdRollingResult
from aps_cp_sat.model.virtual_order_utils import count_effective_virtual_rows, normalize_effective_virtual_flags
from aps_cp_sat.rules import RULE_REGISTRY, RuleKey
from aps_cp_sat.validate.final_schedule_audit import run_final_schedule_audit

VALIDATION_RULE_KEYS = {
    "width_jump_violation": RuleKey.WIDTH_TRANSITION,
    "thickness_violation": RuleKey.THICKNESS_TRANSITION,
    "non_pc_direct_switch": RuleKey.CROSS_GROUP_BRIDGE,
    "temp_conflict": RuleKey.TEMPERATURE_OVERLAP,
    "low_ton_campaign_cnt": RuleKey.CAMPAIGN_TON_MIN,
}

# Compatibility-only fallback. Production path must pass cfg.rule explicitly.
_COMPAT_RULE = RuleConfig()


def _first_existing_col(df: pd.DataFrame, names: list[str]) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    return None


def _campaign_display_id(row) -> str:
    """Display-safe campaign id.

    block-first uses semantic campaign ids such as ``big_roll__slot_3``. The
    validator must treat campaign_id as a label, not a numeric slot.
    """
    try:
        if "campaign_id" in row.index:
            value = row.get("campaign_id")
            if not pd.isna(value) and str(value).strip():
                return str(value)
    except Exception:
        pass
    try:
        if "assigned_slot" in row.index:
            value = row.get("assigned_slot")
            if not pd.isna(value) and str(value).strip():
                return str(value)
    except Exception:
        pass
    return "NA"


def recompute_final_schedule_summary(schedule_df: pd.DataFrame, rule: RuleConfig) -> Dict[str, object]:
    """Recompute final schedule KPIs from the final schedule dataframe only."""
    out: Dict[str, object] = {
        "campaign_cnt": 0,
        "assembled_slot_count": 0,
        "underfilled_slot_count": 0,
        "campaign_ton_min_violation_count": 0,
        "campaign_ton_max_violation_count": 0,
        "final_realized_order_count": 0,
        "final_realized_tons": 0.0,
        "final_realized_total_rows": 0,
        "final_total_tons_including_virtual": 0.0,
        "campaign_total_tons_min": 0.0,
        "campaign_total_tons_max": 0.0,
        "campaign_total_tons_avg": 0.0,
    }
    if schedule_df is None or schedule_df.empty:
        return out

    df = normalize_effective_virtual_flags(schedule_df)
    if "line" not in df.columns:
        df["line"] = ""
    if "campaign_id" not in df.columns:
        if "master_slot" in df.columns:
            df["campaign_id"] = df["line"].astype(str) + "__slot_" + pd.to_numeric(
                df["master_slot"], errors="coerce"
            ).fillna(0).astype(int).astype(str)
        elif "slot_no" in df.columns:
            df["campaign_id"] = df["line"].astype(str) + "__slot_" + pd.to_numeric(
                df["slot_no"], errors="coerce"
            ).fillna(0).astype(int).astype(str)
        else:
            df["campaign_id"] = "campaign_1"

    real = df[~df["is_virtual"].fillna(False).astype(bool)].copy()
    out["final_realized_order_count"] = int(len(real))
    out["final_realized_tons"] = float(real["tons"].sum()) if "tons" in real.columns else 0.0
    out["final_realized_total_rows"] = int(len(df))
    out["final_total_tons_including_virtual"] = float(df["tons"].sum()) if "tons" in df.columns else 0.0

    if "tons" not in df.columns:
        return out

    csum = df.groupby(["line", "campaign_id"], as_index=False)["tons"].sum()
    min_ton = float(rule.campaign_ton_min)
    max_ton = float(rule.campaign_ton_max)
    low_mask = csum["tons"] < min_ton
    high_mask = csum["tons"] > max_ton
    out["campaign_cnt"] = int(len(csum))
    out["assembled_slot_count"] = int(len(csum))
    out["underfilled_slot_count"] = int(low_mask.sum())
    out["campaign_ton_min_violation_count"] = int(low_mask.sum())
    out["campaign_ton_max_violation_count"] = int(high_mask.sum())
    out["campaign_total_tons_min"] = float(csum["tons"].min()) if not csum.empty else 0.0
    out["campaign_total_tons_max"] = float(csum["tons"].max()) if not csum.empty else 0.0
    out["campaign_total_tons_avg"] = float(csum["tons"].mean()) if not csum.empty else 0.0
    return out


def _build_shadow_virtual_fill_analysis(
        schedule_df: pd.DataFrame,
        rule: RuleConfig,
        model_cfg,
) -> tuple[list[dict], Dict[str, object]]:
    def campaign_viability_band(current_tons: float) -> str:
        min_ton = float(rule.campaign_ton_min)
        gap = max(0.0, min_ton - float(current_tons))
        near_gap = float(getattr(model_cfg, "near_viable_gap_tons", 80.0) or 80.0)
        merge_gap = float(getattr(model_cfg, "merge_candidate_gap_tons", 250.0) or 250.0)
        if gap <= 0.0:
            return "ALREADY_VIABLE"
        if gap <= near_gap:
            return "NEAR_VIABLE"
        if gap <= merge_gap:
            return "MERGE_CANDIDATE"
        return "HOPELESS_UNDERFILLED"

    def estimate_shadow_virtual_fill_count(needed_tons: float, cfg_obj, rule_obj: RuleConfig) -> tuple[int, float, str]:
        unit_tons = float(getattr(cfg_obj, "virtual_fill_unit_tons_assumption", 0.0) or 0.0)
        if unit_tons <= 0.0:
            unit_tons = float(getattr(rule_obj, "virtual_tons", 20.0) or 20.0)
        if unit_tons <= 0.0:
            return 0, 0.0, "INVALID_UNIT"
        needed = max(0.0, float(needed_tons))
        cnt = int(math.ceil(needed / unit_tons)) if needed > 0 else 0
        return int(cnt), float(unit_tons), "FIXED_UNIT_TONS"

    rows: list[dict] = []
    metrics: Dict[str, object] = {
        "shadow_virtual_fill_candidate_count": 0,
        "shadow_virtual_fill_viable_count": 0,
        "shadow_virtual_fill_needed_tons_total": 0.0,
        "shadow_virtual_possible_reduced_underfilled_campaigns": 0,
        "shadow_virtual_budget_needed_estimate": 0.0,
        "shadow_virtual_possible_reduced_hard_violations": 0,
        "shadow_virtual_fill_unit_tons_assumption": "",
        "near_viable_promote_success_count": 0,
        "hopeless_segments_dropped_for_near_viable_count": 0,
        "second_receiver_candidate_count": 0,
        "second_receiver_selected_campaign": "",
        "second_receiver_gap_before": 0.0,
        "second_receiver_gap_after": 0.0,
        "second_receiver_viability_band_before": "",
        "second_receiver_viability_band_after": "",
        "second_receiver_promote_success_count": 0,
        "donor_reallocation_attempt_count": 0,
        "donor_reallocation_success_count": 0,
        "donor_reallocation_rejected_due_to_pair_invalid": 0,
        "donor_reallocation_rejected_due_to_duplicate": 0,
        "donor_reallocation_rejected_due_to_no_ton_gain": 0,
        "second_receiver_search_space_size": 0,
        "donor_candidates_considered_count": 0,
        "donor_candidates_rejected_due_to_line_mismatch": 0,
        "donor_candidates_rejected_due_to_pair_invalid": 0,
        "donor_candidates_rejected_due_to_duplicate": 0,
        "donor_candidates_rejected_due_to_no_ton_gain": 0,
        "donor_candidates_rejected_due_to_healthy_campaign_protection": 0,
        "frozen_healthy_campaign_count": 0,
        "second_receiver_ready_for_formal_fill": False,
        "second_receiver_gap_after_reallocation": 0.0,
        "second_receiver_failure_reason": "",
        "second_receiver_analysis_rows": [],
        "second_receiver_donor_candidate_rows": [],
    }
    if schedule_df is None or schedule_df.empty:
        return rows, metrics
    enabled = bool(getattr(model_cfg, "virtual_enabled", True)) and bool(
        getattr(model_cfg, "virtual_shadow_mode_enabled", True))
    fill_enabled = bool(getattr(model_cfg, "virtual_shadow_fill_enabled", True))
    formal_enabled = bool(getattr(model_cfg, "virtual_formal_enabled", False))
    if not enabled or not fill_enabled or formal_enabled:
        return rows, metrics

    df = schedule_df.copy()
    if "is_virtual" not in df.columns:
        df["is_virtual"] = False
    if "steel_group" not in df.columns:
        df["steel_group"] = ""
    if not {"line", "campaign_id", "tons"}.issubset(df.columns):
        return rows, metrics
    real = df[~df["is_virtual"].fillna(False).astype(bool)].copy()
    if real.empty:
        return rows, metrics

    min_ton = float(rule.campaign_ton_min)
    total_budget = float(getattr(model_cfg, "virtual_budget_total_tons", 0.0) or 0.0)
    per_campaign_budget = float(getattr(model_cfg, "virtual_budget_per_campaign_tons", 0.0) or 0.0)
    max_count = int(getattr(model_cfg, "virtual_max_count_per_campaign", 0) or 0)
    max_gap_for_trial = float(getattr(model_cfg, "virtual_formal_fill_max_gap_tons", per_campaign_budget or 0.0) or 0.0)
    allowed_lines = [str(x) for x in (getattr(model_cfg, "virtual_allowed_lines", []) or [])]
    allowed_groups = [str(x) for x in (getattr(model_cfg, "virtual_allowed_steel_groups", []) or [])]
    remaining_budget = max(0.0, total_budget)

    csum = real.groupby(["line", "campaign_id"], as_index=False).agg(
        tons=("tons", "sum"),
        steel_group=("steel_group", "last"),
    )
    csum = csum.sort_values(["line", "campaign_id"], kind="mergesort")
    for _, row in csum.iterrows():
        current_tons = float(row["tons"] or 0.0)
        if current_tons >= min_ton:
            continue
        needed_tons = max(0.0, min_ton - current_tons)
        needed_count, unit_tons, assumption = estimate_shadow_virtual_fill_count(needed_tons, model_cfg, rule)
        metrics["shadow_virtual_fill_unit_tons_assumption"] = str(assumption)
        viability_band = campaign_viability_band(current_tons)

        line = str(row["line"])
        steel_group = str(row.get("steel_group", "") or "")
        reason_code = "VIABLE"
        reason_detail = "within constraints"
        budget_ok = bool(needed_tons <= per_campaign_budget and needed_tons <= remaining_budget)
        viable = True
        if not list(getattr(rule, "virtual_width_levels", ()) or ()) or not list(
                getattr(rule, "virtual_thickness_levels", ()) or ()):
            viable = False
            reason_code = "NO_COMPATIBLE_VIRTUAL_SPEC"
            reason_detail = "virtual width/thickness levels not configured"
        elif allowed_lines and line not in allowed_lines:
            viable = False
            reason_code = "LINE_NOT_ALLOWED"
            reason_detail = f"line={line} not in virtual_allowed_lines"
        elif allowed_groups and steel_group and steel_group not in allowed_groups:
            viable = False
            reason_code = "STEEL_GROUP_NOT_ALLOWED"
            reason_detail = f"steel_group={steel_group} not in virtual_allowed_steel_groups"
        elif max_gap_for_trial > 0 and needed_tons > max_gap_for_trial:
            viable = False
            reason_code = "GAP_TOO_LARGE"
            reason_detail = f"needed_tons={needed_tons:.1f} > formal_gap_limit={max_gap_for_trial:.1f}"
        elif needed_count > max_count:
            viable = False
            reason_code = "COUNT_LIMIT_EXCEEDED"
            reason_detail = f"needed_count={needed_count} > max_count={max_count}"
        elif not budget_ok:
            viable = False
            reason_code = "BUDGET_EXCEEDED"
            reason_detail = (
                f"needed_tons={needed_tons:.1f} exceeds per_campaign_budget={per_campaign_budget:.1f}"
                if needed_tons > per_campaign_budget
                else f"needed_tons={needed_tons:.1f} exceeds remaining_budget={remaining_budget:.1f}"
            )
        elif unit_tons <= 0.0:
            viable = False
            reason_code = "RULE_NOT_CONFIGURED"
            reason_detail = "virtual fill unit tons assumption is invalid"
        if viability_band == "NEAR_VIABLE" and viable:
            priority_band = "HIGH_PRIORITY_FILL"
        elif viable:
            priority_band = "LOW_PRIORITY_FILL"
        else:
            priority_band = "NOT_RECOMMENDED"

        rows.append(
            {
                "line": line,
                "campaign_id": str(row["campaign_id"]),
                "current_tons": float(round(current_tons, 3)),
                "min_required_tons": float(min_ton),
                "gap_tons": float(round(needed_tons, 3)),
                "viability_band": str(viability_band),
                "shadow_fill_needed_tons": float(round(needed_tons, 3)),
                "shadow_fill_needed_count": int(needed_count),
                "shadow_fill_viable": bool(viable),
                "shadow_fill_budget_ok": bool(budget_ok),
                "shadow_fill_reason_code": str(reason_code),
                "shadow_fill_reason_detail": str(reason_detail),
                "shadow_fill_priority_band": str(priority_band),
                "shadow_virtual_fill_unit_tons_assumption": str(assumption),
                "expected_gain_type": "REDUCE_UNDERFILLED_CAMPAIGN",
                "expected_gain_value": int(1 if viable else 0),
            }
        )
        metrics["shadow_virtual_fill_needed_tons_total"] = float(
            metrics["shadow_virtual_fill_needed_tons_total"]) + needed_tons
        if viable:
            metrics["shadow_virtual_fill_viable_count"] = int(metrics["shadow_virtual_fill_viable_count"]) + 1
            remaining_budget -= needed_tons

    metrics["shadow_virtual_fill_candidate_count"] = int(len(rows))
    metrics["shadow_virtual_possible_reduced_underfilled_campaigns"] = int(metrics["shadow_virtual_fill_viable_count"])
    metrics["shadow_virtual_possible_reduced_hard_violations"] = int(metrics["shadow_virtual_fill_viable_count"])
    metrics["shadow_virtual_budget_needed_estimate"] = float(round(total_budget - remaining_budget, 3))
    if "promoted_by_local_reassemble" in df.columns:
        promoted = df[df["promoted_by_local_reassemble"].fillna(False).astype(bool)].copy()
        if not promoted.empty and {"line", "campaign_id"}.issubset(promoted.columns):
            metrics["near_viable_promote_success_count"] = int(
                len(promoted.groupby(["line", "campaign_id"], dropna=False)))
    if "hopeless_dropped_for_near_viable" in df.columns:
        metrics["hopeless_segments_dropped_for_near_viable_count"] = int(
            df["hopeless_dropped_for_near_viable"].fillna(False).astype(bool).sum())
    metrics["second_receiver_candidate_count"] = int((csum["tons"] < min_ton).sum()) if not csum.empty else 0
    metrics["frozen_healthy_campaign_count"] = int((csum["tons"] >= min_ton).sum()) if not csum.empty else 0
    if "donor_reallocated_to_second_receiver" in df.columns:
        donor_reallocated = df[df["donor_reallocated_to_second_receiver"].fillna(False).astype(bool)].copy()
        if not donor_reallocated.empty and {"line", "campaign_id"}.issubset(donor_reallocated.columns):
            metrics["donor_reallocation_success_count"] = int(
                len(donor_reallocated.groupby(["line", "campaign_id"], dropna=False)))
    if "second_receiver_donor_campaign_id" in df.columns:
        donor_marks = df[df["second_receiver_donor_campaign_id"].fillna("").astype(str) != ""].copy()
        donor_rows: list[dict] = []
        if not donor_marks.empty and {"line", "campaign_id"}.issubset(donor_marks.columns):
            for (line, campaign_id), cdf in donor_marks.groupby(["line", "campaign_id"], dropna=False):
                first = cdf.iloc[0]
                reason = str(first.get("second_receiver_donor_rejection_reason", "") or "")
                pair_ok = bool(first.get("second_receiver_donor_pair_feasible", False))
                dup_ok = bool(first.get("second_receiver_donor_duplicate_safe", False))
                donor_tons = float(
                    pd.to_numeric(cdf.get("second_receiver_donor_tons", pd.Series([0.0])), errors="coerce").fillna(
                        0.0).max())
                gap_after_est = float(
                    pd.to_numeric(cdf.get("second_receiver_donor_gap_after_estimate", pd.Series([0.0])),
                                  errors="coerce").fillna(0.0).max())
                selected_for_reallocation = bool(
                    cdf.get("second_receiver_donor_selected", pd.Series([False] * len(cdf))).fillna(False).astype(
                        bool).any())
                donor_rows.append(
                    {
                        "line": str(line),
                        "second_receiver_campaign_id": "",
                        "donor_campaign_id": str(campaign_id),
                        "donor_viability_band": str(first.get("second_receiver_donor_viability_band", "")),
                        "donor_tons": float(round(donor_tons, 3)),
                        "receiver_gap_before": 0.0,
                        "receiver_gap_after_estimate": float(round(gap_after_est, 3)),
                        "pair_feasible": bool(pair_ok),
                        "duplicate_safe": bool(dup_ok),
                        "rejection_reason": reason,
                        "selected_for_reallocation": bool(selected_for_reallocation),
                    }
                )
        metrics["second_receiver_donor_candidate_rows"] = donor_rows
        metrics["donor_candidates_considered_count"] = int(len(donor_rows))
        metrics["donor_reallocation_attempt_count"] = int(len(donor_rows))
        metrics["second_receiver_search_space_size"] = int(len(donor_rows))
        metrics["donor_candidates_rejected_due_to_pair_invalid"] = int(
            sum(1 for r in donor_rows if r.get("rejection_reason") == "PAIR_INVALID"))
        metrics["donor_candidates_rejected_due_to_duplicate"] = int(
            sum(1 for r in donor_rows if r.get("rejection_reason") == "DUPLICATE_ORDER"))
        metrics["donor_candidates_rejected_due_to_no_ton_gain"] = int(
            sum(1 for r in donor_rows if r.get("rejection_reason") in {"NO_TON_GAIN", "NO_BAND_OR_GAP_PROGRESS"})
        )
        metrics["donor_candidates_rejected_due_to_healthy_campaign_protection"] = int(
            sum(1 for r in donor_rows if r.get("rejection_reason") == "HEALTHY_CAMPAIGN_PROTECTED")
        )
        metrics["donor_reallocation_rejected_due_to_pair_invalid"] = int(
            metrics["donor_candidates_rejected_due_to_pair_invalid"])
        metrics["donor_reallocation_rejected_due_to_duplicate"] = int(
            metrics["donor_candidates_rejected_due_to_duplicate"])
        metrics["donor_reallocation_rejected_due_to_no_ton_gain"] = int(
            metrics["donor_candidates_rejected_due_to_no_ton_gain"])
    if "second_receiver_selected" in df.columns:
        selected = df[df["second_receiver_selected"].fillna(False).astype(bool)].copy()
        if not selected.empty and {"line", "campaign_id"}.issubset(selected.columns):
            first = selected.iloc[0]
            line = str(first.get("line", ""))
            campaign_id = str(first.get("campaign_id", ""))
            current = csum[(csum["line"].astype(str) == line) & (csum["campaign_id"].astype(str) == campaign_id)]
            tons_after = float(current.iloc[0]["tons"] or 0.0) if not current.empty else float(selected["tons"].sum())
            gap_after = max(0.0, min_ton - tons_after)
            band_after = campaign_viability_band(tons_after)

            def _first_existing(col: str, default):
                if col not in selected.columns:
                    return default
                vals = selected[col].dropna()
                if vals.empty:
                    return default
                val = vals.iloc[0]
                if isinstance(default, (float, int)):
                    try:
                        return float(val)
                    except Exception:
                        return default
                if isinstance(default, bool):
                    try:
                        return bool(val)
                    except Exception:
                        return default
                text = str(val)
                return text if text else default

            gap_before = float(_first_existing("second_receiver_gap_before", gap_after))
            band_before = str(_first_existing("second_receiver_viability_band_before", band_after))
            stored_gap_after = float(_first_existing("second_receiver_gap_after", gap_after))
            stored_band_after = str(_first_existing("second_receiver_viability_band_after", band_after))
            ready = bool(_first_existing("second_receiver_ready_for_formal_fill", stored_band_after == "NEAR_VIABLE"))
            failure_reason = str(_first_existing("second_receiver_failure_reason", ""))
            if stored_gap_after == 0.0 and gap_after > 0.0:
                stored_gap_after = gap_after
            if not stored_band_after:
                stored_band_after = band_after
            metrics["second_receiver_selected_campaign"] = f"{line}#{campaign_id}"
            metrics["second_receiver_gap_before"] = float(round(gap_before, 3))
            metrics["second_receiver_gap_after"] = float(round(stored_gap_after, 3))
            metrics["second_receiver_gap_after_reallocation"] = float(round(stored_gap_after, 3))
            metrics["second_receiver_viability_band_before"] = band_before
            metrics["second_receiver_viability_band_after"] = stored_band_after
            metrics["second_receiver_ready_for_formal_fill"] = bool(ready)
            metrics["second_receiver_failure_reason"] = failure_reason
            if stored_band_after != band_before or stored_gap_after + 1e-9 < gap_before:
                metrics["second_receiver_promote_success_count"] = 1
            shadow_by_key = {(str(r.get("line", "")), str(r.get("campaign_id", ""))): r for r in rows}
            shadow = shadow_by_key.get((line, campaign_id), {})
            for drow in metrics.get("second_receiver_donor_candidate_rows", []):
                drow["second_receiver_campaign_id"] = campaign_id
                drow["receiver_gap_before"] = float(round(gap_before, 3))
            metrics["second_receiver_analysis_rows"] = [
                {
                    "line": line,
                    "campaign_id": campaign_id,
                    "role": "second_receiver",
                    "tons_before": float(round(max(0.0, min_ton - gap_before), 3)),
                    "tons_after": float(round(tons_after, 3)),
                    "gap_before": float(round(gap_before, 3)),
                    "gap_after": float(round(stored_gap_after, 3)),
                    "viability_band_before": band_before,
                    "viability_band_after": stored_band_after,
                    "donor_reallocated": bool(metrics["donor_reallocation_success_count"]),
                    "shadow_fill_reason_code_after": str(shadow.get("shadow_fill_reason_code", failure_reason)),
                    "shadow_fill_priority_band_after": str(
                        shadow.get("shadow_fill_priority_band", "HIGH_PRIORITY_FILL" if ready else "NOT_RECOMMENDED")),
                    "local_reallocation_reason": str(failure_reason),
                    "ready_for_formal_fill_next_step": bool(ready),
                }
            ]
    return rows, metrics


def _build_formal_virtual_fill_trial_summary(schedule_df: pd.DataFrame) -> tuple[list[dict], Dict[str, object]]:
    rows: list[dict] = []
    metrics: Dict[str, object] = {
        "formal_virtual_fill_trial_count": 0,
        "formal_virtual_fill_success_count": 0,
        "formal_virtual_fill_rollback_count": 0,
        "formal_virtual_fill_success_campaigns": [],
        "formal_virtual_fill_rollback_reasons": [],
    }
    if schedule_df is None or schedule_df.empty:
        return rows, metrics
    if not {"line", "campaign_id", "tons", "is_virtual"}.issubset(schedule_df.columns):
        return rows, metrics
    work = schedule_df.copy()
    if "virtual_usage_type" not in work.columns:
        work["virtual_usage_type"] = ""
    virt = work[
        work.get("is_virtual", pd.Series([False] * len(work), index=work.index)).fillna(False).astype(bool)
        & (work["virtual_usage_type"].astype(str) == "fill")
        ].copy()
    if virt.empty:
        return rows, metrics

    real = work[
        ~work.get("is_virtual", pd.Series([False] * len(work), index=work.index)).fillna(False).astype(bool)].copy()
    real_tons = real.groupby(["line", "campaign_id"], as_index=False)["tons"].sum().rename(
        columns={"tons": "current_tons_before"})
    fill_tons = virt.groupby(["line", "campaign_id"], as_index=False).agg(
        virtual_fill_count=("tons", "count"),
        virtual_fill_tons=("tons", "sum"),
    )
    merged = pd.merge(real_tons, fill_tons, on=["line", "campaign_id"], how="inner")
    for _, row in merged.iterrows():
        before = float(row["current_tons_before"] or 0.0)
        fill_tons_val = float(row["virtual_fill_tons"] or 0.0)
        rows.append(
            {
                "line": str(row["line"]),
                "campaign_id": str(row["campaign_id"]),
                "current_tons_before": float(round(before, 3)),
                "current_tons_after": float(round(before + fill_tons_val, 3)),
                "gap_before": None,
                "virtual_fill_count": int(row["virtual_fill_count"] or 0),
                "virtual_fill_tons": float(round(fill_tons_val, 3)),
                "trial_result": "SUCCESS",
                "rollback_reason": "",
            }
        )
    metrics["formal_virtual_fill_trial_count"] = int(len(rows))
    metrics["formal_virtual_fill_success_count"] = int(len(rows))
    metrics["formal_virtual_fill_rollback_count"] = 0
    metrics["formal_virtual_fill_success_campaigns"] = [f"{r['line']}#{r['campaign_id']}" for r in rows]
    metrics["formal_virtual_fill_rollback_reasons"] = []
    return rows, metrics


def _build_shadow_bridge_analysis(schedule_df: pd.DataFrame, rule: RuleConfig, model_cfg) -> tuple[
    list[dict], Dict[str, object]]:
    def _band(tons: float) -> str:
        gap = max(0.0, float(rule.campaign_ton_min) - float(tons))
        near_gap = float(getattr(model_cfg, "near_viable_gap_tons", 80.0) or 80.0)
        merge_gap = float(getattr(model_cfg, "merge_candidate_gap_tons", 250.0) or 250.0)
        if gap <= 0.0:
            return "ALREADY_VIABLE"
        if gap <= near_gap:
            return "NEAR_VIABLE"
        if gap <= merge_gap:
            return "MERGE_CANDIDATE"
        return "HOPELESS_UNDERFILLED"

    metrics: Dict[str, object] = {
        "shadow_bridge_candidate_count": 0,
        "shadow_bridge_analysis_attempt_count": 0,
        "shadow_bridge_analysis_success_count": 0,
        "shadow_bridge_possible_reduced_underfilled_campaigns": 0,
        "shadow_bridge_possible_reduced_hard_violations": 0,
        "second_receiver_shadow_bridge_gap_before": 0.0,
        "second_receiver_shadow_bridge_gap_after_estimate": 0.0,
        "second_receiver_shadow_bridge_viability_band_before": "",
        "second_receiver_shadow_bridge_viability_band_after_estimate": "",
        "second_receiver_shadow_bridge_reason_code": "",
        "shadow_bridge_analysis_rows": [],
    }
    rows: list[dict] = []
    if schedule_df is None or schedule_df.empty or not {"line", "campaign_id", "tons"}.issubset(schedule_df.columns):
        metrics["second_receiver_shadow_bridge_reason_code"] = "NO_BRIDGE_CANDIDATE"
        return rows, metrics
    enabled = bool(getattr(model_cfg, "virtual_enabled", True)) and bool(
        getattr(model_cfg, "virtual_shadow_mode_enabled", True))
    bridge_enabled = bool(getattr(model_cfg, "virtual_shadow_bridge_enabled", True))
    formal_bridge = bool(getattr(model_cfg, "virtual_formal_bridge_enabled", False))
    if not enabled or not bridge_enabled or formal_bridge:
        metrics["second_receiver_shadow_bridge_reason_code"] = "BRIDGE_NOT_ALLOWED_BY_CONFIG"
        return rows, metrics

    df = schedule_df.copy()
    if "is_virtual" not in df.columns:
        df["is_virtual"] = False
    real = df[~df["is_virtual"].fillna(False).astype(bool)].copy()
    if real.empty:
        metrics["second_receiver_shadow_bridge_reason_code"] = "NO_BRIDGE_CANDIDATE"
        return rows, metrics
    if "global_sequence_on_line" not in real.columns:
        real["global_sequence_on_line"] = real.groupby("line", sort=False).cumcount() + 1

    min_ton = float(rule.campaign_ton_min)
    near_gap = float(getattr(model_cfg, "near_viable_gap_tons", 80.0) or 80.0)
    merge_gap = float(getattr(model_cfg, "merge_candidate_gap_tons", 250.0) or 250.0)
    max_virtual_tons = float(getattr(model_cfg, "shadow_bridge_max_total_virtual_tons_per_campaign", 100.0) or 100.0)
    max_virtual_count = int(getattr(model_cfg, "shadow_bridge_max_virtual_count_per_gap", 5) or 5)

    csum = (
        real.groupby(["line", "campaign_id"], as_index=False, dropna=False)
        .agg(
            tons=("tons", "sum"),
            first_global=("global_sequence_on_line", "min"),
            last_global=("global_sequence_on_line", "max"),
        )
        .sort_values(["line", "first_global"], kind="mergesort")
        .reset_index(drop=True)
    )
    csum["gap"] = csum["tons"].map(lambda v: max(0.0, min_ton - float(v or 0.0)))
    csum["viability_band"] = csum["tons"].map(lambda v: _band(float(v or 0.0)))

    selected_line = ""
    selected_campaign = ""
    if "second_receiver_selected" in df.columns:
        selected = df[df["second_receiver_selected"].fillna(False).astype(bool)]
        if not selected.empty:
            selected_line = str(selected.iloc[0].get("line", ""))
            selected_campaign = str(selected.iloc[0].get("campaign_id", ""))
    attempt_count = 0
    success_count = 0
    for _, camp in csum[csum["tons"] < min_ton].iterrows():
        line = str(camp["line"])
        campaign_id = str(camp["campaign_id"])
        current_tons = float(camp["tons"] or 0.0)
        gap = float(camp["gap"] or 0.0)
        before_band = str(camp["viability_band"])
        line_meta = csum[csum["line"].astype(str) == line].sort_values("first_global", kind="mergesort").reset_index(
            drop=True)
        idxs = line_meta.index[line_meta["campaign_id"].astype(str) == campaign_id].tolist()
        best_after = current_tons
        candidate_count = 0
        reason = "NO_BRIDGE_CANDIDATE"
        action = "NOT_WORTH_RESCUING"
        if idxs:
            idx = int(idxs[0])
            saw_healthy = False
            saw_nonhealthy_no_gain = False
            for nidx in [idx - 1, idx + 1, idx - 2, idx + 2]:
                if nidx < 0 or nidx >= len(line_meta):
                    continue
                neighbor = line_meta.iloc[nidx]
                donor_tons = float(neighbor["tons"] or 0.0)
                if donor_tons >= min_ton:
                    saw_healthy = True
                    attempt_count += 1
                    continue
                attempt_count += 1
                after = current_tons + donor_tons
                if after <= current_tons:
                    saw_nonhealthy_no_gain = True
                    continue
                candidate_count += 1
                if max_virtual_tons <= 0.0 or max_virtual_count <= 0:
                    reason = "BRIDGE_EXCEEDS_VIRTUAL_CHAIN_LIMIT"
                    continue
                best_after = max(best_after, after)
                after_gap = max(0.0, min_ton - after)
                after_band = _band(after)
                if after_gap < gap or after_band != before_band:
                    reason = "BRIDGE_COULD_OPEN_DONOR_PATH"
                    success_count += 1
                    action = "TRY_SHADOW_BRIDGE_NEXT"
                    break
                saw_nonhealthy_no_gain = True
            if candidate_count <= 0:
                if saw_healthy:
                    reason = "BRIDGE_ONLY_WITH_MIXED_COMPLEX_PATH"
                    action = "REQUIRES_COMPLEX_MIXED_BRIDGE"
                elif saw_nonhealthy_no_gain:
                    reason = "BRIDGE_COULD_NOT_IMPROVE_TON"
                    action = "KEEP_AS_IS"
        after_gap = max(0.0, min_ton - best_after)
        after_band = _band(best_after)
        if reason == "BRIDGE_COULD_OPEN_DONOR_PATH" and after_gap <= near_gap:
            action = "TRY_FORMAL_FILL_NEXT"
        elif reason == "BRIDGE_COULD_OPEN_DONOR_PATH" and after_gap <= merge_gap:
            action = "TRY_SHADOW_BRIDGE_NEXT"
        is_second = bool(line == selected_line and campaign_id == selected_campaign)
        rows.append(
            {
                "line": line,
                "campaign_id": campaign_id,
                "current_tons": float(round(current_tons, 3)),
                "gap_tons": float(round(gap, 3)),
                "bridge_candidate_count": int(candidate_count),
                "bridge_analysis_attempt_count": int(attempt_count),
                "gap_after_estimate": float(round(after_gap, 3)),
                "viability_band_before": before_band,
                "viability_band_after_estimate": after_band,
                "shadow_bridge_reason_code": reason,
                "recommended_next_action": action,
                "is_second_receiver": bool(is_second),
                "second_receiver_selected": bool(is_second),
                "second_receiver_gap_before": float(round(gap, 3)) if is_second else 0.0,
                "second_receiver_gap_after_estimate": float(round(after_gap, 3)) if is_second else 0.0,
            }
        )
    second = next((r for r in rows if bool(r.get("is_second_receiver", False))), {})
    metrics["shadow_bridge_analysis_rows"] = rows
    metrics["shadow_bridge_candidate_count"] = int(sum(int(r.get("bridge_candidate_count", 0) or 0) for r in rows))
    metrics["shadow_bridge_analysis_attempt_count"] = int(attempt_count)
    metrics["shadow_bridge_analysis_success_count"] = int(success_count)
    metrics["shadow_bridge_possible_reduced_underfilled_campaigns"] = int(
        sum(1 for r in rows if r.get("shadow_bridge_reason_code") == "BRIDGE_COULD_OPEN_DONOR_PATH"))
    metrics["shadow_bridge_possible_reduced_hard_violations"] = int(
        metrics["shadow_bridge_possible_reduced_underfilled_campaigns"])
    metrics["second_receiver_shadow_bridge_gap_before"] = float(second.get("second_receiver_gap_before", 0.0) or 0.0)
    metrics["second_receiver_shadow_bridge_gap_after_estimate"] = float(
        second.get("second_receiver_gap_after_estimate", 0.0) or 0.0)
    metrics["second_receiver_shadow_bridge_viability_band_before"] = str(second.get("viability_band_before", ""))
    metrics["second_receiver_shadow_bridge_viability_band_after_estimate"] = str(
        second.get("viability_band_after_estimate", ""))
    metrics["second_receiver_shadow_bridge_reason_code"] = str(
        second.get("shadow_bridge_reason_code", "NO_BRIDGE_CANDIDATE"))
    return rows, metrics


def validate_width_audit_by_campaign(df: pd.DataFrame) -> Dict[str, object]:
    """Independent display-level width audit.

    This audit only checks adjacent coils inside the same campaign. It does not
    use the template graph, so it can separate real in-campaign width issues
    from cross-campaign visual misreads in exported sheets.
    """
    out: Dict[str, object] = {
        "width_audit_pair_count": 0,
        "width_audit_increase_violation_count": 0,
        "width_audit_overdrop_violation_count": 0,
        "width_audit_total_violation_count": 0,
        "width_audit_examples": [],
    }
    if df.empty or "width" not in df.columns:
        return out

    line_col = _first_existing_col(df, ["line", "assigned_line"])
    campaign_col = _first_existing_col(df, ["campaign_id", "master_slot", "assigned_slot", "slot_no"])
    seq_col = _first_existing_col(df, ["campaign_seq", "sequence_in_slot"])
    order_col = _first_existing_col(df, ["order_id", "source_order_id"])
    if not line_col or not campaign_col:
        return out

    work = df.copy()
    work["_audit_line"] = work[line_col].astype(str)
    work["_audit_campaign"] = work[campaign_col].astype(str)
    if seq_col:
        work["_audit_seq"] = pd.to_numeric(work[seq_col], errors="coerce")
    else:
        work["_audit_seq"] = pd.NA
    if work["_audit_seq"].isna().any():
        fallback = work.groupby(["_audit_line", "_audit_campaign"], dropna=False).cumcount() + 1
        work["_audit_seq"] = work["_audit_seq"].fillna(fallback)
    work["_audit_seq"] = pd.to_numeric(work["_audit_seq"], errors="coerce").fillna(0).astype(int)
    work["_audit_width"] = pd.to_numeric(work["width"], errors="coerce")
    work["_audit_order_id"] = work[order_col].astype(str) if order_col else ""
    work = work.dropna(subset=["_audit_width"]).sort_values(
        ["_audit_line", "_audit_campaign", "_audit_seq"],
        kind="mergesort",
    )

    examples: list[dict[str, object]] = []
    pair_count = 0
    increase_count = 0
    overdrop_count = 0
    for (line, campaign), grp in work.groupby(["_audit_line", "_audit_campaign"], dropna=False, sort=False):
        prev = None
        for _, row in grp.iterrows():
            if prev is None:
                prev = row
                continue
            pair_count += 1
            width_a = float(prev["_audit_width"])
            width_b = float(row["_audit_width"])
            delta = width_b - width_a
            audit_result = "OK"
            if width_b > width_a:
                increase_count += 1
                audit_result = "WIDTH_INCREASE_VIOLATION"
            elif width_a - width_b > 250.0:
                overdrop_count += 1
                audit_result = "WIDTH_OVERDROP_VIOLATION"
            if audit_result != "OK" and len(examples) < 50:
                examples.append(
                    {
                        "line": line,
                        "campaign_id": campaign,
                        "seq_a": int(prev["_audit_seq"]),
                        "seq_b": int(row["_audit_seq"]),
                        "order_id_a": str(prev["_audit_order_id"]),
                        "order_id_b": str(row["_audit_order_id"]),
                        "width_a": width_a,
                        "width_b": width_b,
                        "width_delta": delta,
                        "same_campaign": True,
                        "audit_result": audit_result,
                    }
                )
            prev = row

    out["width_audit_pair_count"] = int(pair_count)
    out["width_audit_increase_violation_count"] = int(increase_count)
    out["width_audit_overdrop_violation_count"] = int(overdrop_count)
    out["width_audit_total_violation_count"] = int(increase_count + overdrop_count)
    out["width_audit_examples"] = examples
    return out


def _audit_virtual_and_reverse(df: pd.DataFrame, rule: RuleConfig) -> Dict[str, object]:
    def _safe_numeric_series(frame: pd.DataFrame, col_name: str) -> pd.Series:
        if col_name not in frame.columns:
            return pd.Series([pd.NA] * len(frame), index=frame.index, dtype="Float64")
        return pd.to_numeric(frame[col_name], errors="coerce")

    out: Dict[str, object] = {}
    if df.empty or "is_virtual" not in df.columns:
        return out

    virtual = df[df["is_virtual"]].copy()
    # Validation must be resilient to lightweight decode outputs that may omit
    # some physical columns. Missing columns are treated as unknown rather than
    # crashing the entire validation pipeline.
    for col in ["width", "thickness", "temp_min", "temp_max"]:
        if col not in virtual.columns:
            virtual[col] = pd.NA
    out["virtual_temp_columns_present"] = bool({"temp_min", "temp_max"}.issubset(set(df.columns)))
    out["virtual_spec_columns_present"] = bool({"width", "thickness", "temp_min", "temp_max"}.issubset(set(df.columns)))
    out["virtual_audit_missing_cols"] = [
        col for col in ["width", "thickness", "temp_min", "temp_max"] if col not in df.columns
    ]

    allowed_widths = {float(v) for v in rule.virtual_width_levels}
    allowed_thicknesses = {float(v) for v in rule.virtual_thickness_levels}
    invalid_virtual_spec_count = 0
    if not virtual.empty:
        width_s = _safe_numeric_series(virtual, "width")
        thick_s = _safe_numeric_series(virtual, "thickness")
        temp_min_s = _safe_numeric_series(virtual, "temp_min")
        temp_max_s = _safe_numeric_series(virtual, "temp_max")

        invalid_virtual_spec_count = int(
            ((~width_s.isin(list(allowed_widths))) & width_s.notna()).sum()
            + ((~thick_s.isin(list(allowed_thicknesses))) & thick_s.notna()).sum()
            + ((temp_min_s < float(rule.virtual_temp_min)) & temp_min_s.notna()).sum()
            + ((temp_max_s > float(rule.virtual_temp_max)) & temp_max_s.notna()).sum()
        )
        out["virtual_widths_used"] = sorted({float(v) for v in width_s.dropna().tolist()})
        out["virtual_thicknesses_used"] = sorted({float(v) for v in thick_s.dropna().tolist()})
        out["virtual_temp_min_used"] = float(temp_min_s.min()) if temp_min_s.notna().any() else 0.0
        out["virtual_temp_max_used"] = float(temp_max_s.max()) if temp_max_s.notna().any() else 0.0
    else:
        out["virtual_widths_used"] = []
        out["virtual_thicknesses_used"] = []
        out["virtual_temp_min_used"] = 0.0
        out["virtual_temp_max_used"] = 0.0
    out["invalid_virtual_spec_count"] = int(invalid_virtual_spec_count)

    max_bridge_count_used = 0
    bridge_count_violation_count = 0
    if {"line", "campaign_id"}.issubset(df.columns):
        for _, g in df.groupby(["line", "campaign_id"], dropna=False):
            chain = 0
            local_max = 0
            for flag in g["is_virtual"].tolist():
                if bool(flag):
                    chain += 1
                    local_max = max(local_max, chain)
                else:
                    chain = 0
            max_bridge_count_used = max(max_bridge_count_used, local_max)
            if local_max > int(rule.max_virtual_chain):
                bridge_count_violation_count += 1
    out["max_bridge_count_used"] = int(max_bridge_count_used)
    out["bridge_count_violation_count"] = int(bridge_count_violation_count)
    out["bridge_widths"] = out["virtual_widths_used"]
    out["bridge_thicknesses"] = out["virtual_thicknesses_used"]

    out["direct_reverse_step_violation_count"] = int(
        df.get("direct_reverse_step_violation", pd.Series(dtype=bool)).fillna(False).sum())
    out["virtual_attach_reverse_violation_count"] = int(
        df.get("virtual_attach_reverse_violation", pd.Series(dtype=bool)).fillna(False).sum())
    out["virtual_attach_reverse_span_max_used"] = float(
        df.loc[df.get("is_virtual", pd.Series(dtype=bool)).fillna(False), "width_rise"].max()
    ) if "width_rise" in df.columns and "is_virtual" in df.columns and bool(df["is_virtual"].any()) else 0.0
    out["bridge_reverse_step_count"] = int(
        df.get("bridge_reverse_step_flag", pd.Series(dtype=bool)).fillna(False).sum())

    if "logical_reverse_cnt_campaign" in df.columns and {"line", "campaign_id"}.issubset(df.columns):
        rev_by_campaign = df.groupby(["line", "campaign_id"], dropna=False)["logical_reverse_cnt_campaign"].max()
        out["period_reverse_count"] = int(rev_by_campaign.max()) if not rev_by_campaign.empty else 0
        out["period_reverse_count_violation_count"] = int(
            (rev_by_campaign > int(rule.max_logical_reverse_per_campaign)).sum()) if not rev_by_campaign.empty else 0
    else:
        out["period_reverse_count"] = 0
        out["period_reverse_count_violation_count"] = 0
    out["reverse_count_definition"] = "logical_reverse_per_campaign"
    return out


def validate_solution_summary(result: ColdRollingResult, rule: RuleConfig | None = None) -> Dict[str, object]:
    # Production path should pass cfg.rule explicitly. The default is kept only
    # for compatibility and lightweight tests.
    rule = rule or _COMPAT_RULE
    df = result.schedule_df
    model_cfg = getattr(getattr(result, "config", None), "model", None)
    if df.empty:
        empty_width_audit = validate_width_audit_by_campaign(df)
        final_audit = run_final_schedule_audit(df, rule, getattr(result, "engine_meta", None) or {})
        final_audit_summary = final_audit.get("summary", {}) if isinstance(final_audit, dict) else {}
        final_audit_details = final_audit.get("details", {}) if isinstance(final_audit, dict) else {}
        out = {"real_orders": 0, "virtual_orders": 0, **empty_width_audit}
        out.update(recompute_final_schedule_summary(df, rule))
        shadow_rows, shadow_metrics = _build_shadow_virtual_fill_analysis(df, rule, model_cfg)
        formal_rows, formal_metrics = _build_formal_virtual_fill_trial_summary(df)
        shadow_bridge_rows, shadow_bridge_metrics = _build_shadow_bridge_analysis(df, rule, model_cfg)
        out["shadow_virtual_fill_rows"] = shadow_rows
        out["formal_virtual_fill_trial_rows"] = formal_rows
        out["shadow_bridge_analysis_rows"] = shadow_bridge_rows
        out.update(shadow_metrics)
        out.update(formal_metrics)
        out.update(shadow_bridge_metrics)
        out["final_schedule_audit_summary"] = final_audit_summary
        out["final_schedule_audit_details"] = final_audit_details
        for key, value in final_audit_summary.items():
            out[key if str(key).startswith("final_") else f"final_{key}"] = value
        return out

    df = normalize_effective_virtual_flags(df)
    out: Dict[str, object] = {
        "real_orders": int(len(df) - count_effective_virtual_rows(df)),
        "virtual_orders": int(count_effective_virtual_rows(df)),
    }
    final_summary = recompute_final_schedule_summary(df, rule)
    out.update(final_summary)
    shadow_rows, shadow_metrics = _build_shadow_virtual_fill_analysis(df, rule, model_cfg)
    formal_rows, formal_metrics = _build_formal_virtual_fill_trial_summary(df)
    shadow_bridge_rows, shadow_bridge_metrics = _build_shadow_bridge_analysis(df, rule, model_cfg)
    out["shadow_virtual_fill_rows"] = shadow_rows
    out["formal_virtual_fill_trial_rows"] = formal_rows
    out["shadow_bridge_analysis_rows"] = shadow_bridge_rows
    out.update(shadow_metrics)
    out.update(formal_metrics)
    out.update(shadow_bridge_metrics)

    # HARD VIOLATION TRACKING: campaign_ton_min and campaign_ton_max are now HARD constraints
    campaign_ton_min_violation_count = int(final_summary.get("campaign_ton_min_violation_count", 0) or 0)
    campaign_ton_max_violation_count = int(final_summary.get("campaign_ton_max_violation_count", 0) or 0)
    campaign_ton_hard_violation_count_total = campaign_ton_min_violation_count + campaign_ton_max_violation_count
    hard_violation_count_total = 0

    if {"line", "campaign_id", "tons"}.issubset(set(df.columns)):
        real = df[~df["is_virtual"]].copy()
        if not real.empty:
            csum = real.groupby(["line", "campaign_id"], as_index=False)["tons"].sum()
            min_ton = float(rule.campaign_ton_min)
            low = csum["tons"] < min_ton
            out["campaign_ton_min_hard_enforced"] = True
            out["campaign_ton_window"] = [float(rule.campaign_ton_min), float(rule.campaign_ton_max)]
            out["campaign_ton_min_violation_count"] = campaign_ton_min_violation_count
            out["campaign_ton_max_violation_count"] = campaign_ton_max_violation_count
            out["campaign_ton_hard_violation_count_total"] = campaign_ton_hard_violation_count_total

            # Legacy support: still track low_ton_campaign_cnt as diagnostic
            low_ton_cnt = campaign_ton_min_violation_count
            out["low_ton_campaign_cnt"] = low_ton_cnt

            csum["gap"] = (min_ton - csum["tons"]).clip(lower=0.0)
            low_top = csum[csum["gap"] > 0.0].sort_values("gap", ascending=False).head(5)
            if not low_top.empty:
                explain = []
                for _, row in low_top.iterrows():
                    explain.append(f"{row['line']}#{_campaign_display_id(row)}:缺口{float(row['gap']):.1f}t")
                out["low_ton_top5"] = " | ".join(explain)

    # Count hard violations from validation checks
    direct_reverse_step_violation_count = 0
    virtual_attach_reverse_violation_count = 0
    period_reverse_count_violation_count = 0
    invalid_virtual_spec_count = 0
    bridge_count_violation_count = 0

    for key in ["width_jump_violation", "thickness_violation", "non_pc_direct_switch", "temp_conflict"]:
        if key in df.columns:
            cnt = int(df[key].fillna(False).sum())
            out[f"{key}_cnt"] = cnt
            hard_violation_count_total += cnt

    # Add virtual and reverse violations to hard count
    vkeys = ["width_jump_violation_cnt", "thickness_violation_cnt", "non_pc_direct_switch_cnt", "temp_conflict_cnt"]
    parts = []
    for key in vkeys:
        if key in out:
            parts.append((key, int(out[key])))
    if parts:
        parts = sorted(parts, key=lambda item: -item[1])
        out["violation_rank"] = " > ".join([f"{key}:{val}" for key, val in parts if val > 0]) or "none"
    out["visible_rule_labels"] = [
        RULE_REGISTRY.get(rule_key).zh_name
        for rule_key in VALIDATION_RULE_KEYS.values()
        if RULE_REGISTRY.get(rule_key).validation_active
    ]

    # Update with virtual and reverse audit data
    virtual_audit = _audit_virtual_and_reverse(df, rule)
    out.update(virtual_audit)
    width_audit = validate_width_audit_by_campaign(df)
    out.update(width_audit)
    final_audit = run_final_schedule_audit(df, rule, getattr(result, "engine_meta", None) or {})
    final_audit_summary = final_audit.get("summary", {}) if isinstance(final_audit, dict) else {}
    final_audit_details = final_audit.get("details", {}) if isinstance(final_audit, dict) else {}
    out["final_schedule_audit_summary"] = final_audit_summary
    out["final_schedule_audit_details"] = final_audit_details
    for key, value in final_audit_summary.items():
        out[key if str(key).startswith("final_") else f"final_{key}"] = value

    # Model vs Validation ton comparison (Priority 1C)
    # Separate real tons and virtual tons per campaign for alignment check
    if {"line", "campaign_id", "tons", "is_virtual"}.issubset(set(df.columns)):
        real_campaign_tons = df[~df["is_virtual"]].groupby(["line", "campaign_id"])["tons"].sum()
        virtual_campaign_tons = df[df["is_virtual"]].groupby(["line", "campaign_id"])["tons"].sum()
        total_campaign_tons = df.groupby(["line", "campaign_id"])["tons"].sum()

        # Calculate real, virtual, total tons per campaign
        campaign_real_tons_list = []
        campaign_virtual_tons_list = []
        campaign_total_tons_list = []
        for idx in total_campaign_tons.index:
            real_t = float(real_campaign_tons.get(idx, 0.0))
            virtual_t = float(virtual_campaign_tons.get(idx, 0.0))
            total_t = float(total_campaign_tons.get(idx, 0.0))
            campaign_real_tons_list.append(real_t)
            campaign_virtual_tons_list.append(virtual_t)
            campaign_total_tons_list.append(total_t)

        out["campaign_real_tons"] = campaign_real_tons_list
        out["campaign_virtual_tons"] = campaign_virtual_tons_list
        out["campaign_total_tons"] = campaign_total_tons_list
        out["campaign_total_tons_min"] = float(min(campaign_total_tons_list)) if campaign_total_tons_list else 0.0
        out["campaign_total_tons_max"] = float(max(campaign_total_tons_list)) if campaign_total_tons_list else 0.0
        out["campaign_total_tons_avg"] = float(
            sum(campaign_total_tons_list) / len(campaign_total_tons_list)) if campaign_total_tons_list else 0.0

        # Model ton vs validation ton gap (if model tons available in result)
        # Note: hasattr returns True even when engine_meta is None, so we must check for None explicitly
        _engine_meta = getattr(result, "engine_meta", None) or {}
        model_load_includes_virtual = bool(_engine_meta.get("model_load_includes_virtual_tons", False))
        out["model_load_includes_virtual_tons"] = model_load_includes_virtual

        # Calculate gap between validation total and expected model limit (2000)
        # If model includes virtual: gap = validation_total - (2000 - virtual_budget)
        # If model excludes virtual: gap = validation_total - 2000
        virtual_budget = float(_engine_meta.get("virtual_budget_per_slot_ton10", 0) or 0) / 10.0
        max_ton = float(rule.campaign_ton_max)

        gaps = []
        for i, total_t in enumerate(campaign_total_tons_list):
            if model_load_includes_virtual:
                # Model already accounts for virtual budget, so gap is against max_ton directly
                effective_limit = max_ton
            else:
                # Model only checks real tons, so validation might exceed due to virtual tons
                effective_limit = max_ton  # Validation compares against max_ton
            gap = max(0.0, total_t - effective_limit)
            gaps.append(gap)

        max_gap = float(max(gaps)) if gaps else 0.0
        gap_slot_count = int(sum(1 for g in gaps if g > 0))
        gap_total = float(sum(gaps))

        out["campaign_model_validation_ton_gap"] = gaps
        out["max_campaign_ton_gap"] = max_gap
        out["campaign_ton_gap_slot_count"] = gap_slot_count
        out["campaign_ton_gap_total"] = gap_total

    # Add virtual/reverse violations to hard count
    direct_reverse_step_violation_count = int(virtual_audit.get("direct_reverse_step_violation_count", 0))
    virtual_attach_reverse_violation_count = int(virtual_audit.get("virtual_attach_reverse_violation_count", 0))
    period_reverse_count_violation_count = int(virtual_audit.get("period_reverse_count_violation_count", 0))
    invalid_virtual_spec_count = int(virtual_audit.get("invalid_virtual_spec_count", 0))
    bridge_count_violation_count = int(virtual_audit.get("bridge_count_violation_count", 0))

    hard_violation_count_total += (
            direct_reverse_step_violation_count
            + virtual_attach_reverse_violation_count
            + period_reverse_count_violation_count
            + invalid_virtual_spec_count
            + bridge_count_violation_count
    )

    # TOTAL HARD VIOLATION COUNT (including campaign ton violations)
    hard_violation_count_total += campaign_ton_hard_violation_count_total
    out["hard_violation_count_total"] = int(hard_violation_count_total)

    # Hard violation breakdown completeness check (Priority 2)
    # Ensure total = sum of all breakdown items
    breakdown_items = [
        ("campaign_ton_min_violation_count", campaign_ton_min_violation_count),
        ("campaign_ton_max_violation_count", campaign_ton_max_violation_count),
        ("width_jump_violation_cnt", int(out.get("width_jump_violation_cnt", 0))),
        ("thickness_violation_cnt", int(out.get("thickness_violation_cnt", 0))),
        ("non_pc_direct_switch_cnt", int(out.get("non_pc_direct_switch_cnt", 0))),
        ("temp_conflict_cnt", int(out.get("temp_conflict_cnt", 0))),
        ("direct_reverse_step_violation_count", direct_reverse_step_violation_count),
        ("virtual_attach_reverse_violation_count", virtual_attach_reverse_violation_count),
        ("period_reverse_count_violation_count", period_reverse_count_violation_count),
        ("invalid_virtual_spec_count", invalid_virtual_spec_count),
        ("bridge_count_violation_count", bridge_count_violation_count),
        ("virtual_bridge_edge_leaked_into_disabled_mode",
         int(out.get("virtual_bridge_edge_leaked_into_disabled_mode", 0))),
    ]
    breakdown_sum = sum(v for _, v in breakdown_items)
    breakdown_gap = abs(int(hard_violation_count_total) - breakdown_sum)
    out["hard_violation_breakdown_complete"] = bool(breakdown_gap == 0)
    out["hard_violation_breakdown_gap"] = breakdown_gap
    out["hard_violation_breakdown_items"] = {k: v for k, v in breakdown_items if v > 0}

    # Add alias fields for clearer naming (Priority 2C)
    out["temperature_overlap_violation_count"] = int(out.get("temp_conflict_cnt", 0))
    out["width_transition_violation_count"] = int(out.get("width_jump_violation_cnt", 0))
    out["thickness_transition_violation_count"] = int(out.get("thickness_violation_cnt", 0))
    out["cross_group_bridge_violation_count"] = int(out.get("non_pc_direct_switch_cnt", 0))

    # Priority 1: Add bridge_expand_violation_cnt - matches validate_model_equivalence output
    # Bridge expand violations = adjacency violations after bridge expansion
    bridge_expand_violation_count = int(virtual_audit.get("bridge_expand_violation_count", 0))
    if "bridge_expand_violation_cnt" not in out:
        out["bridge_expand_violation_cnt"] = bridge_expand_violation_count
    out["bridge_expand_violation_count"] = bridge_expand_violation_count

    # chain_break_cnt: campaign 内序号链断裂次数（来自 validate_model_equivalence）
    # Note: this is tracked separately as it indicates structural issues, not rule violations
    if "chain_break_cnt" not in out:
        out["chain_break_cnt"] = 0
    out["chain_break_warning"] = bool(out["chain_break_cnt"] > 0)

    # Priority 4: Per-line violation breakdown for failure source identification
    # Group violations by line to identify which line causes routing failures
    if {"line", "width_jump_violation", "thickness_violation", "temp_conflict", "non_pc_direct_switch"}.issubset(
            df.columns):
        for line in sorted(df["line"].dropna().unique().tolist()):
            line_df = df[df["line"] == line]
            out[f"{line}_width_jump_violation_cnt"] = int(line_df["width_jump_violation"].fillna(False).sum())
            out[f"{line}_thickness_violation_cnt"] = int(line_df["thickness_violation"].fillna(False).sum())
            out[f"{line}_temp_conflict_cnt"] = int(line_df["temp_conflict"].fillna(False).sum())
            out[f"{line}_non_pc_direct_switch_cnt"] = int(line_df["non_pc_direct_switch"].fillna(False).sum())
            line_adj = int(line_df["width_jump_violation"].fillna(False).sum()) + \
                       int(line_df["thickness_violation"].fillna(False).sum()) + \
                       int(line_df["temp_conflict"].fillna(False).sum()) + \
                       int(line_df["non_pc_direct_switch"].fillna(False).sum())
            out[f"{line}_adjacency_violation_cnt"] = line_adj
            out[f"{line}_real_orders"] = int(
                (~line_df["is_virtual"]).sum()) if "is_virtual" in line_df.columns else int(len(line_df))
            out[f"{line}_campaign_count"] = int(
                line_df["campaign_id"].nunique()) if "campaign_id" in line_df.columns else 0
        # Failure source summary: which line has the most violations
        line_adj_viols = {k.replace("_adjacency_violation_cnt", ""): v for k, v in out.items() if
                          k.endswith("_adjacency_violation_cnt")}
        if line_adj_viols:
            max_line = max(line_adj_viols, key=line_adj_viols.get)
            max_viols = line_adj_viols[max_line]
            out["failure_source_line"] = str(max_line) if max_viols > 0 else ""
            out["failure_source_max_violations"] = int(max_viols) if max_viols > 0 else 0
            out["failure_source_summary"] = (
                f"line={max_line}, violations={max_viols}"
                if max_viols > 0
                else "no_adjacency_violations"
            )

    # -------------------------------------------------------------------------
    # Constructive LNS path diagnostics
    # Extract metrics from engine_meta / lns_diagnostics when using constructive_lns path.
    # -------------------------------------------------------------------------
    _engine_meta = getattr(result, "engine_meta", None) or {}
    if str(_engine_meta.get("engine_used", "")) == "constructive_lns" or str(
            _engine_meta.get("main_path", "")) == "constructive_lns":
        lns_diag = _engine_meta.get("lns_diagnostics", {}) or {}
        rounds_summary = lns_diag.get("rounds_summary", {}) or {}

        out["constructive_initial_chain_count"] = int(
            lns_diag.get("constructive_build_diags", {}).get("total_chains", 0)
            if isinstance(lns_diag.get("constructive_build_diags"), dict)
            else 0
        )
        out["constructive_initial_segment_count"] = int(
            lns_diag.get("initial_planned_count", 0)
        )
        out["constructive_final_segment_count"] = int(
            lns_diag.get("final_planned_count", 0)
        )
        out["lns_accepted_rounds"] = int(
            rounds_summary.get("accepted_count", 0)
        )
        out["lns_total_rounds"] = int(
            _engine_meta.get("lns_engine_meta", {}).get("n_rounds_ran", 0)
            if isinstance(_engine_meta.get("lns_engine_meta"), dict)
            else 0
        )
        out["lns_status"] = str(_engine_meta.get("lns_status", "UNKNOWN"))
        out["lns_improvement_delta_orders"] = int(
            lns_diag.get("improvement_delta_orders", 0)
        )
        out["lns_initial_dropped_count"] = int(
            lns_diag.get("initial_dropped_count", 0)
        )
        out["lns_final_dropped_count"] = int(
            lns_diag.get("final_dropped_count", 0)
        )
        out["lns_drop_delta"] = int(
            lns_diag.get("final_dropped_count", 0) - lns_diag.get("initial_dropped_count", 0)
        )
        out["solver_path"] = "constructive_lns"

        # ---- Bridge expansion mode guard ----
        # Top-level engine_meta is the stable source of truth after pipeline normalization.
        bridge_expand_mode = str(
            _engine_meta.get("bridge_expansion_mode", "disabled")
            or lns_diag.get("bridge_expansion_mode", "disabled")
            or "disabled"
        )
        out["bridge_expansion_mode"] = bridge_expand_mode

        # Check for VIRTUAL_BRIDGE_EDGE leak in disabled mode
        virtual_bridge_edge_leaked = 0
        if bridge_expand_mode == "disabled":
            if "selected_edge_type" in df.columns:
                virtual_bridge_edge_leaked = int(
                    (df["selected_edge_type"].astype(str) == "VIRTUAL_BRIDGE_EDGE").sum()
                )
            elif "selected_bridge_path" in df.columns:
                # Fallback: count non-empty bridge paths as potential leaks
                bridge_paths = df["selected_bridge_path"].astype(str)
                virtual_bridge_edge_leaked = int((bridge_paths.str.len() > 0).sum())

        out["virtual_bridge_edge_leaked_into_disabled_mode"] = virtual_bridge_edge_leaked

        # If VIRTUAL_BRIDGE_EDGE leaked, mark bridge_expand_ok = False
        if virtual_bridge_edge_leaked > 0:
            out["bridge_expand_ok"] = False
            hard_violation_count_total += virtual_bridge_edge_leaked

            # Also update breakdown items
            breakdown_items.append(
                ("virtual_bridge_edge_leaked_into_disabled_mode", virtual_bridge_edge_leaked)
            )
    else:
        out["solver_path"] = "joint_master"
        out["bridge_expansion_mode"] = "unknown"
        out["virtual_bridge_edge_leaked_into_disabled_mode"] = 0

    print(f"[APS][校验摘要] {out}")
    return out


def validate_model_equivalence(schedule_df: pd.DataFrame, templates_df: pd.DataFrame | None = None) -> Dict[
    str, object]:
    """
    独立等价校验器：
    - campaign内序号是否单链连续
    - 相邻对是否满足宽/厚/温/跨组规则（复用检查列）
    - 若给了模板表，检查相邻实物对是否在模板候选中
    """
    out: Dict[str, object] = {
        "campaign_single_chain_ok": True,
        "adjacency_rule_ok": True,
        "template_pair_ok": True,
        "bridge_expand_ok": True,
        "hint_consistency_ok": True,
        "chain_break_cnt": 0,
        "adjacency_violation_cnt": 0,
        "template_miss_cnt": 0,
        "bridge_expand_violation_cnt": 0,
        "hint_mismatch_cnt": 0,
    }
    if schedule_df is None or schedule_df.empty:
        return out

    df = schedule_df.copy().sort_values(["line", "campaign_id", "campaign_seq"], kind="mergesort")
    if {"line", "campaign_id", "campaign_seq"}.issubset(df.columns):
        for _, d in df.groupby(["line", "campaign_id"], dropna=False):
            seq = pd.to_numeric(d["campaign_seq"], errors="coerce").fillna(0).astype(int).tolist()
            if seq and seq != list(range(1, len(seq) + 1)):
                out["campaign_single_chain_ok"] = False
                out["chain_break_cnt"] = int(out["chain_break_cnt"]) + 1

    for col in ["width_jump_violation", "thickness_violation", "non_pc_direct_switch", "temp_conflict"]:
        if col in df.columns:
            out["adjacency_violation_cnt"] = int(out["adjacency_violation_cnt"]) + int(df[col].fillna(False).sum())
    out["adjacency_rule_ok"] = int(out["adjacency_violation_cnt"]) == 0

    # 桥接展开一致性：展开后序列不应再出现虚拟相关相邻违规。
    bridge_violation_cnt = 0
    for col in ["width_jump_violation", "thickness_violation", "temp_conflict", "non_pc_direct_switch"]:
        if col in df.columns:
            bridge_violation_cnt += int(df[col].fillna(False).sum())
    out["bridge_expand_violation_cnt"] = int(bridge_violation_cnt)
    out["bridge_expand_ok"] = int(bridge_violation_cnt) == 0
    if {"line", "campaign_id", "campaign_seq", "selected_bridge_path", "is_virtual"}.issubset(df.columns):
        bridge_path_break = 0
        for _, g in df.groupby(["line", "campaign_id"], dropna=False):
            rows = g.sort_values("campaign_seq", kind="mergesort").to_dict("records")
            for i in range(len(rows) - 1):
                cur = rows[i]
                nxt = rows[i + 1]
                has_path = str(cur.get("selected_bridge_path", "")).strip() != ""
                if has_path and not bool(nxt.get("is_virtual", False)):
                    bridge_path_break += 1
        out["bridge_path_expand_miss_cnt"] = int(bridge_path_break)
        out["bridge_expand_ok"] = bool(out["bridge_expand_ok"]) and int(bridge_path_break) == 0

    # hint一致性：campaign_id_hint / campaign_seq_hint / force_break_before 与导出序列一致。
    hint_mismatch = 0
    if {"campaign_id_hint", "campaign_id"}.issubset(df.columns):
        x = df["campaign_id_hint"].astype(str).fillna("").str.strip()
        y = df["campaign_id"].astype(str).fillna("").str.strip()
        hint_mismatch += int(((x != "") & (x != "nan") & (x != y)).sum())
    if {"campaign_seq_hint", "campaign_seq"}.issubset(df.columns):
        x = pd.to_numeric(df["campaign_seq_hint"], errors="coerce").fillna(0).astype(int)
        y = pd.to_numeric(df["campaign_seq"], errors="coerce").fillna(0).astype(int)
        hint_mismatch += int(((x > 0) & (x != y)).sum())
    if "force_break_before" in df.columns and {"line", "campaign_id", "campaign_seq"}.issubset(df.columns):
        d = df.sort_values(["line", "campaign_id", "campaign_seq"], kind="mergesort").copy()
        fb = pd.to_numeric(d["force_break_before"], errors="coerce").fillna(0).astype(int)
        cs = pd.to_numeric(d["campaign_seq"], errors="coerce").fillna(0).astype(int)
        hint_mismatch += int(((fb > 0) & (cs != 1)).sum())
    out["hint_mismatch_cnt"] = int(hint_mismatch)
    out["hint_consistency_ok"] = int(hint_mismatch) == 0

    if isinstance(templates_df, pd.DataFrame) and not templates_df.empty:
        keyset = set(
            zip(
                templates_df["line"].astype(str),
                templates_df["from_order_id"].astype(str),
                templates_df["to_order_id"].astype(str),
            )
        )
        miss = 0
        template_miss_examples: list[dict] = []
        for (_, _), d in df.groupby(["line", "campaign_id"], dropna=False):
            rows = d.sort_values("campaign_seq").to_dict("records")
            for i in range(len(rows) - 1):
                a = rows[i]
                b = rows[i + 1]
                if bool(a.get("is_virtual", False)) or bool(b.get("is_virtual", False)):
                    continue
                key = (str(a.get("line", "")), str(a.get("order_id", "")), str(b.get("order_id", "")))
                if key not in keyset:
                    miss += 1
                    if len(template_miss_examples) < 20:
                        sample = {
                            "line": str(a.get("line", "")),
                            "campaign_id": str(a.get("campaign_id", "")),
                            "campaign_seq_a": int(a.get("campaign_seq", 0) or 0),
                            "campaign_seq_b": int(b.get("campaign_seq", 0) or 0),
                            "order_id_a": str(a.get("order_id", "")),
                            "order_id_b": str(b.get("order_id", "")),
                        }
                        if "selected_edge_type" in a:
                            sample["selected_edge_type_a"] = str(a.get("selected_edge_type", ""))
                        if "selected_edge_type" in b:
                            sample["selected_edge_type_b"] = str(b.get("selected_edge_type", ""))
                        template_miss_examples.append(sample)
        out["template_miss_cnt"] = int(miss)
        out["template_pair_ok"] = miss == 0
        out["template_miss_examples"] = template_miss_examples
        if miss > 0:
            preview = template_miss_examples[:5]
            print(f"[APS][template_miss_examples] 共 {miss} 个 miss，前 {len(preview)} 条示例:")
            for ex in preview:
                print(f"  line={ex['line']}, campaign={ex['campaign_id']}, "
                      f"seq=({ex['campaign_seq_a']}->{ex['campaign_seq_b']}), "
                      f"order=({ex['order_id_a']}->{ex['order_id_b']})")

    print(f"[APS][等价校验] {out}")
    return out
