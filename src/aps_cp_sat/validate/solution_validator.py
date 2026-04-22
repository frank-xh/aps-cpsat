from __future__ import annotations

"""
Validate layer contract.

This module only checks solved plans and produces summaries. It must not mutate
schedules, inject bridge rows, or repair feasibility.
"""

from typing import Dict

import pandas as pd

from aps_cp_sat.config import RuleConfig
from aps_cp_sat.domain.models import ColdRollingResult
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
    out: Dict[str, object] = {}
    if df.empty or "is_virtual" not in df.columns:
        return out

    virtual = df[df["is_virtual"]].copy()
    allowed_widths = {float(v) for v in rule.virtual_width_levels}
    allowed_thicknesses = {float(v) for v in rule.virtual_thickness_levels}
    invalid_virtual_spec_count = 0
    if not virtual.empty:
        invalid_virtual_spec_count = int(
            (~virtual["width"].astype(float).isin(allowed_widths)).sum()
            + (~virtual["thickness"].astype(float).isin(allowed_thicknesses)).sum()
            + (virtual["temp_min"].astype(float) < float(rule.virtual_temp_min)).sum()
            + (virtual["temp_max"].astype(float) > float(rule.virtual_temp_max)).sum()
        )
        out["virtual_widths_used"] = sorted({float(v) for v in virtual["width"].astype(float).tolist()})
        out["virtual_thicknesses_used"] = sorted({float(v) for v in virtual["thickness"].astype(float).tolist()})
        out["virtual_temp_min_used"] = float(virtual["temp_min"].astype(float).min())
        out["virtual_temp_max_used"] = float(virtual["temp_max"].astype(float).max())
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

    out["direct_reverse_step_violation_count"] = int(df.get("direct_reverse_step_violation", pd.Series(dtype=bool)).fillna(False).sum())
    out["virtual_attach_reverse_violation_count"] = int(df.get("virtual_attach_reverse_violation", pd.Series(dtype=bool)).fillna(False).sum())
    out["virtual_attach_reverse_span_max_used"] = float(
        df.loc[df.get("is_virtual", pd.Series(dtype=bool)).fillna(False), "width_rise"].max()
    ) if "width_rise" in df.columns and "is_virtual" in df.columns and bool(df["is_virtual"].any()) else 0.0
    out["bridge_reverse_step_count"] = int(df.get("bridge_reverse_step_flag", pd.Series(dtype=bool)).fillna(False).sum())

    if "logical_reverse_cnt_campaign" in df.columns and {"line", "campaign_id"}.issubset(df.columns):
        rev_by_campaign = df.groupby(["line", "campaign_id"], dropna=False)["logical_reverse_cnt_campaign"].max()
        out["period_reverse_count"] = int(rev_by_campaign.max()) if not rev_by_campaign.empty else 0
        out["period_reverse_count_violation_count"] = int((rev_by_campaign > int(rule.max_logical_reverse_per_campaign)).sum()) if not rev_by_campaign.empty else 0
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
    if df.empty:
        empty_width_audit = validate_width_audit_by_campaign(df)
        final_audit = run_final_schedule_audit(df, rule, getattr(result, "engine_meta", None) or {})
        final_audit_summary = final_audit.get("summary", {}) if isinstance(final_audit, dict) else {}
        final_audit_details = final_audit.get("details", {}) if isinstance(final_audit, dict) else {}
        out = {"real_orders": 0, "virtual_orders": 0, **empty_width_audit}
        out["final_schedule_audit_summary"] = final_audit_summary
        out["final_schedule_audit_details"] = final_audit_details
        for key, value in final_audit_summary.items():
            out[key if str(key).startswith("final_") else f"final_{key}"] = value
        return out

    out: Dict[str, object] = {
        "real_orders": int((~df["is_virtual"]).sum()),
        "virtual_orders": int(df["is_virtual"].sum()),
    }

    # HARD VIOLATION TRACKING: campaign_ton_min and campaign_ton_max are now HARD constraints
    campaign_ton_min_violation_count = 0
    campaign_ton_max_violation_count = 0
    campaign_ton_hard_violation_count_total = 0
    hard_violation_count_total = 0

    if {"line", "campaign_id", "tons"}.issubset(set(df.columns)):
        real = df[~df["is_virtual"]].copy()
        if not real.empty:
            csum = real.groupby(["line", "campaign_id"], as_index=False)["tons"].sum()
            min_ton = float(rule.campaign_ton_min)
            max_ton = float(rule.campaign_ton_max)

            # HARD: campaign_ton_min violation (slot tons < 700)
            low = csum["tons"] < min_ton
            campaign_ton_min_violation_count = int(low.sum())

            # HARD: campaign_ton_max violation (slot tons > 2000)
            over = csum["tons"] > max_ton
            campaign_ton_max_violation_count = int(over.sum())

            # Total campaign ton hard violations
            campaign_ton_hard_violation_count_total = campaign_ton_min_violation_count + campaign_ton_max_violation_count

            out["campaign_cnt"] = int(len(csum))
            out["campaign_ton_min_hard_enforced"] = True
            out["campaign_ton_window"] = [min_ton, max_ton]
            out["campaign_ton_min_violation_count"] = campaign_ton_min_violation_count
            out["campaign_ton_max_violation_count"] = campaign_ton_max_violation_count
            out["campaign_ton_hard_violation_count_total"] = campaign_ton_hard_violation_count_total

            # Legacy support: still track low_ton_campaign_cnt as diagnostic
            low_ton_cnt = int(low.sum())
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
        out["campaign_total_tons_avg"] = float(sum(campaign_total_tons_list) / len(campaign_total_tons_list)) if campaign_total_tons_list else 0.0
        
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
        ("virtual_bridge_edge_leaked_into_disabled_mode", int(out.get("virtual_bridge_edge_leaked_into_disabled_mode", 0))),
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
    if {"line", "width_jump_violation", "thickness_violation", "temp_conflict", "non_pc_direct_switch"}.issubset(df.columns):
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
            out[f"{line}_real_orders"] = int((~line_df["is_virtual"]).sum()) if "is_virtual" in line_df.columns else int(len(line_df))
            out[f"{line}_campaign_count"] = int(line_df["campaign_id"].nunique()) if "campaign_id" in line_df.columns else 0
        # Failure source summary: which line has the most violations
        line_adj_viols = {k.replace("_adjacency_violation_cnt", ""): v for k, v in out.items() if k.endswith("_adjacency_violation_cnt")}
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
    if str(_engine_meta.get("engine_used", "")) == "constructive_lns" or str(_engine_meta.get("main_path", "")) == "constructive_lns":
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


def validate_model_equivalence(schedule_df: pd.DataFrame, templates_df: pd.DataFrame | None = None) -> Dict[str, object]:
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
