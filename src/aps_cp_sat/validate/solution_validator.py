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


VALIDATION_RULE_KEYS = {
    "width_jump_violation": RuleKey.WIDTH_TRANSITION,
    "thickness_violation": RuleKey.THICKNESS_TRANSITION,
    "non_pc_direct_switch": RuleKey.CROSS_GROUP_BRIDGE,
    "temp_conflict": RuleKey.TEMPERATURE_OVERLAP,
    "low_ton_campaign_cnt": RuleKey.CAMPAIGN_TON_MIN,
}

# Compatibility-only fallback. Production path must pass cfg.rule explicitly.
_COMPAT_RULE = RuleConfig()


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
        return {"real_orders": 0, "virtual_orders": 0}

    out: Dict[str, object] = {
        "real_orders": int((~df["is_virtual"]).sum()),
        "virtual_orders": int(df["is_virtual"].sum()),
    }

    if {"line", "campaign_id", "tons"}.issubset(set(df.columns)):
        real = df[~df["is_virtual"]].copy()
        if not real.empty:
            csum = real.groupby(["line", "campaign_id"], as_index=False)["tons"].sum()
            low = csum["tons"] < float(rule.campaign_ton_min)
            out["campaign_cnt"] = int(len(csum))
            out["low_ton_campaign_cnt"] = int(low.sum())

            csum["gap"] = (float(rule.campaign_ton_min) - csum["tons"]).clip(lower=0.0)
            low_top = csum[csum["gap"] > 0.0].sort_values("gap", ascending=False).head(5)
            if not low_top.empty:
                explain = []
                for _, row in low_top.iterrows():
                    explain.append(f"{row['line']}#{int(row['campaign_id'])}:缺口{float(row['gap']):.1f}t")
                out["low_ton_top5"] = " | ".join(explain)

    for key in ["width_jump_violation", "thickness_violation", "non_pc_direct_switch", "temp_conflict"]:
        if key in df.columns:
            out[f"{key}_cnt"] = int(df[key].fillna(False).sum())

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
    out.update(_audit_virtual_and_reverse(df, rule))

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
        x = pd.to_numeric(df["campaign_id_hint"], errors="coerce").fillna(0).astype(int)
        y = pd.to_numeric(df["campaign_id"], errors="coerce").fillna(0).astype(int)
        hint_mismatch += int(((x > 0) & (x != y)).sum())
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
        out["template_miss_cnt"] = int(miss)
        out["template_pair_ok"] = miss == 0

    print(f"[APS][等价校验] {out}")
    return out
