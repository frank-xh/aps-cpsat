from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd

from aps_cp_sat.config.rule_config import RuleConfig
from aps_cp_sat.config.score_config import ScoreConfig
from aps_cp_sat.rules.shared_checks import (
    max_virtual_chain_len,
    reverse_width_flags,
    temperature_overlap_metrics,
    thickness_transition_metrics,
    width_transition_metrics,
)


def apply_solution_checks(df: pd.DataFrame, rule: RuleConfig) -> pd.DataFrame:
    out = df.copy().reset_index(drop=True)
    if out.empty:
        return out
    group_cols = ["line", "campaign_id"]
    out["seq"] = out.index + 1
    out["prev_width"] = out.groupby(group_cols)["width"].shift(1)
    out["prev_thickness"] = out.groupby(group_cols)["thickness"].shift(1)
    out["prev_group"] = out.groupby(group_cols)["steel_group"].shift(1)
    out["prev_is_virtual"] = (
        out.groupby(group_cols)["is_virtual"].shift(1).astype("boolean").fillna(False).astype(bool)
    )
    out["prev_temp_min"] = out.groupby(group_cols)["temp_min"].shift(1)
    out["prev_temp_max"] = out.groupby(group_cols)["temp_max"].shift(1)
    out["width_drop"] = out["prev_width"] - out["width"]
    width_metrics = out.apply(
        lambda row: width_transition_metrics(
            row.get("prev_width", 0.0),
            row.get("width", 0.0),
            rule,
            bool(row.get("prev_is_virtual", False)),
            bool(row.get("is_virtual", False)),
        )
        if pd.notna(row.get("prev_width"))
        else (False, 0.0, False, False),
        axis=1,
        result_type="expand",
    )
    width_metrics.columns = ["width_jump_violation", "width_over_mm", "narrow_to_wide", "physical_reverse_violation"]
    out[["width_jump_violation", "width_over_mm", "narrow_to_wide", "physical_reverse_violation"]] = width_metrics
    out["width_rise"] = (out["width"] - out["prev_width"]).clip(lower=0.0).fillna(0.0)
    out["direct_reverse_step_violation"] = (
        out["narrow_to_wide"]
        & (~out["prev_is_virtual"])
        & (~out["is_virtual"])
        & (out["width_rise"] > float(rule.real_reverse_step_max_mm))
    ).fillna(False)
    out["virtual_attach_reverse_violation"] = (
        out["narrow_to_wide"]
        & out["is_virtual"]
        & (out["width_rise"] > float(rule.virtual_reverse_attach_max_mm))
    ).fillna(False)
    out["bridge_reverse_step_flag"] = (out["narrow_to_wide"] & out["is_virtual"]).fillna(False)
    out["logical_reverse_flag"] = out.apply(
        lambda row: reverse_width_flags(row.get("narrow_to_wide", False), row.get("prev_is_virtual", False), row.get("is_virtual", False)),
        axis=1,
    )
    thick_metrics = out.apply(
        lambda row: thickness_transition_metrics(row.get("prev_thickness", 0.0), row.get("thickness", 0.0))
        if pd.notna(row.get("prev_thickness"))
        else (False, 0.0),
        axis=1,
        result_type="expand",
    )
    thick_metrics.columns = ["thickness_violation", "thick_over_mm"]
    out[["thickness_violation", "thick_over_mm"]] = thick_metrics
    out["group_switch"] = (out["steel_group"] != out["prev_group"]).fillna(False)
    out["non_pc_direct_switch"] = (
        out["group_switch"]
        & ~out["steel_group"].astype(str).str.upper().isin(["PC", "VIRTUAL_PC", "普碳"])
        & ~out["prev_group"].astype(str).str.upper().isin(["PC", "VIRTUAL_PC", "普碳"])
    ).fillna(False)
    out["temp_required_overlap"] = ((~out["is_virtual"]) & (~out["prev_is_virtual"])).map(
        lambda x: float(rule.min_temp_overlap_real_real) if bool(x) else 0.0
    )
    temp_metrics = out.apply(
        lambda row: temperature_overlap_metrics(
            row.get("prev_temp_min", 0.0),
            row.get("prev_temp_max", 0.0),
            row.get("temp_min", 0.0),
            row.get("temp_max", 0.0),
            row.get("temp_required_overlap", 0.0),
        )
        if pd.notna(row.get("prev_temp_min")) and pd.notna(row.get("prev_temp_max"))
        else (0.0, False, 0.0),
        axis=1,
        result_type="expand",
    )
    temp_metrics.columns = ["temp_overlap", "temp_conflict", "temp_shortage"]
    out[["temp_overlap", "temp_conflict", "temp_shortage"]] = temp_metrics
    out["width_smooth_mm"] = out["width_drop"].clip(lower=0.0).fillna(0.0)
    out["thick_smooth_mm"] = (out["thickness"] - out["prev_thickness"]).abs().fillna(0.0)
    out["temp_margin_pen"] = (
        (float(rule.temp_margin_target) - out["temp_overlap"]).clip(lower=0.0)
        * ((out["temp_overlap"] >= float(rule.min_temp_overlap_real_real)) & (out["temp_overlap"] < float(rule.temp_margin_target))).astype(int)
    )
    rev_cnt = out.groupby(group_cols)["logical_reverse_flag"].transform("sum").fillna(0).astype(int)
    out["logical_reverse_cnt_campaign"] = rev_cnt
    out["period_reverse_count"] = rev_cnt
    out["logical_reverse_count_violation"] = (rev_cnt > int(rule.max_logical_reverse_per_campaign)).fillna(False)
    out["width_jump_violation"] = (out["width_jump_violation"] | out["logical_reverse_count_violation"]).fillna(False)
    out["period_reverse_count_violation_count"] = (
        rev_cnt - int(rule.max_logical_reverse_per_campaign)
    ).clip(lower=0).astype(int)
    out["reverse_count_definition"] = "logical_reverse_per_campaign"
    first_rows = out.groupby(group_cols).head(1).index
    out.loc[first_rows, ["narrow_to_wide", "width_jump_violation", "thickness_violation", "group_switch", "non_pc_direct_switch", "temp_conflict"]] = False
    out.loc[first_rows, ["width_over_mm", "thick_over_mm", "temp_shortage", "width_smooth_mm", "thick_smooth_mm", "temp_margin_pen"]] = 0.0
    out.loc[first_rows, ["physical_reverse_violation", "logical_reverse_flag", "logical_reverse_count_violation", "direct_reverse_step_violation", "virtual_attach_reverse_violation", "bridge_reverse_step_flag"]] = False
    return out


def _campaign_ton_penalty(df: pd.DataFrame, rule: RuleConfig) -> Tuple[int, float]:
    if df.empty:
        return 0, 0.0
    real = df[~df["is_virtual"]].copy()
    if real.empty:
        return 0, 0.0
    tons_by = real.groupby(["line", "campaign_id"])["tons"].sum().reset_index(name="campaign_tons")
    low = tons_by[tons_by["campaign_tons"] < float(rule.campaign_ton_min)].copy()
    if low.empty:
        return 0, 0.0
    low["gap_tons"] = float(rule.campaign_ton_min) - low["campaign_tons"]
    return int(len(low)), float(low["gap_tons"].sum())


def _campaign_ton_excess(df: pd.DataFrame, rule: RuleConfig) -> float:
    if df.empty:
        return 0.0
    real = df[~df["is_virtual"]].copy()
    if real.empty:
        return 0.0
    csum = real.groupby(["line", "campaign_id"])["tons"].sum().reset_index(name="campaign_tons")
    return float((csum["campaign_tons"] - float(rule.campaign_ton_max)).clip(lower=0.0).sum())


def _campaign_ton_target_deviation(df: pd.DataFrame, target: float) -> float:
    if df.empty:
        return 0.0
    real = df[~df["is_virtual"]].copy()
    if real.empty:
        return 0.0
    csum = real.groupby(["line", "campaign_id"])["tons"].sum().reset_index(name="campaign_tons")
    return float((csum["campaign_tons"] * 10.0 - target * 10.0).abs().sum())


def _max_virtual_chain_len(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    m = 0
    for _, g in df.groupby(["line", "campaign_id"], dropna=False):
        m = max(m, max_virtual_chain_len(g["is_virtual"].tolist()))
    return int(m)


def _pure_virtual_campaign_count(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    cnt = 0
    for _, g in df.groupby(["line", "campaign_id"], dropna=False):
        if bool(g["is_virtual"].all()):
            cnt += 1
    return int(cnt)


def weighted_penalties(df: pd.DataFrame, dropped_cnt: int, score: ScoreConfig, rule: RuleConfig) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if df.empty:
        out["total_penalty"] = float(score.unassigned_real * dropped_cnt)
        return out
    real = (~df["is_virtual"])
    roll_line_mismatch_v = int(
        (
            ((df["line_capability"] == "big_only") & (df["line"] != "big_roll"))
            | ((df["line_capability"] == "small_only") & (df["line"] != "small_roll"))
        ).sum()
    ) if "line_capability" in df.columns else 0
    ton_over = _campaign_ton_excess(df, rule)
    low_cnt, ton_under_gap = _campaign_ton_penalty(df, rule)
    temp_short = float(df.loc[real, "temp_shortage"].sum()) if "temp_shortage" in df.columns else 0.0
    width_over = float(df.loc[real, "width_over_mm"].sum()) if "width_over_mm" in df.columns else 0.0
    thick_over = float(df.loc[real, "thick_over_mm"].sum()) if "thick_over_mm" in df.columns else 0.0
    non_pc = int(df.loc[real, "non_pc_direct_switch"].sum()) if "non_pc_direct_switch" in df.columns else 0
    vchain_excess = max(0, _max_virtual_chain_len(df) - int(rule.virtual_chain_penalty_threshold))
    virtual_cnt = int(df["is_virtual"].sum())
    virtual_ton10 = int(round(float(df.loc[df["is_virtual"], "tons"].sum()) * 10.0))
    total_ton10 = int(round(float(df["tons"].sum()) * 10.0))
    virtual_ratio_pen_units = max(0, virtual_ton10 * int(rule.virtual_ton_ratio_den) - total_ton10 * int(rule.virtual_ton_ratio_num))
    width_smooth = float(df.loc[real, "width_smooth_mm"].sum()) if "width_smooth_mm" in df.columns else 0.0
    thick_smooth = float(df.loc[real, "thick_smooth_mm"].sum()) if "thick_smooth_mm" in df.columns else 0.0
    temp_margin = float(df.loc[real, "temp_margin_pen"].sum()) if "temp_margin_pen" in df.columns else 0.0
    ton_target_dev = _campaign_ton_target_deviation(df, target=score.ton_target_value)
    pure_virtual_campaign_cnt = _pure_virtual_campaign_count(df)
    out["roll_line_mismatch_pen"] = float(score.roll_line_mismatch * roll_line_mismatch_v)
    out["ton_over_pen"] = float(score.ton_over * ton_over)
    out["ton_under_pen"] = float(score.ton_under * ton_under_gap)
    out["temp_short_pen"] = float(score.temp_shortage * temp_short)
    out["width_over_pen"] = float(score.width_violation * width_over)
    out["thick_over_pen"] = float(score.thick_violation * thick_over)
    out["non_pc_pen"] = float(score.non_pc_switch * non_pc)
    out["vchain_excess_pen"] = float(score.virtual_chain_excess * vchain_excess)
    out["unassigned_pen"] = float(score.unassigned_real * int(dropped_cnt))
    out["virtual_ratio_pen"] = float(score.virtual_ratio * virtual_ratio_pen_units)
    out["pure_virtual_campaign_pen"] = float(score.pure_virtual_campaign * pure_virtual_campaign_cnt)
    out["virtual_use_pen"] = float(score.virtual_use * virtual_cnt)
    out["width_smooth_pen"] = float(score.width_smooth * width_smooth)
    out["thick_smooth_pen"] = float(score.thick_smooth * thick_smooth)
    out["temp_margin_pen"] = float(score.temp_margin * temp_margin)
    out["ton_target_pen"] = float(score.ton_target * ton_target_dev)
    out["low_ton_campaign_cnt"] = float(low_cnt)
    out["low_ton_gap_tons"] = float(ton_under_gap)
    out["pure_virtual_campaign_cnt"] = float(pure_virtual_campaign_cnt)
    out["total_penalty"] = float(sum(v for k, v in out.items() if k.endswith("_pen")))
    return out
