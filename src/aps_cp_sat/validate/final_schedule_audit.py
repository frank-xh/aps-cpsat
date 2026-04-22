from __future__ import annotations

"""Final schedule audit.

This module audits the realized/decode final schedule directly. It does not use
template graph completeness, so it can validate what operators actually see in
the exported schedule.
"""

from typing import Any, Dict

import pandas as pd

from aps_cp_sat.config.rule_config import RuleConfig


def _first_col(df: pd.DataFrame, names: list[str]) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    return None


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _append_limited(rows: list[dict[str, Any]], item: dict[str, Any], limit: int = 100) -> None:
    if len(rows) < limit:
        rows.append(item)


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        return out

    line_col = _first_col(out, ["line", "assigned_line"])
    campaign_col = _first_col(out, ["campaign_id", "master_slot", "assigned_slot", "slot_no"])
    seq_col = _first_col(out, ["campaign_seq", "sequence_in_slot", "global_sequence_on_line"])
    slot_col = _first_col(out, ["master_slot", "assigned_slot", "slot_no"])
    order_col = _first_col(out, ["order_id", "source_order_id"])

    out["_audit_line"] = out[line_col].astype(str) if line_col else ""
    out["_audit_campaign"] = out[campaign_col].astype(str) if campaign_col else ""
    out["_audit_slot"] = out[slot_col].astype(str) if slot_col else ""
    out["_audit_order_id"] = out[order_col].astype(str) if order_col else ""

    if seq_col:
        out["_audit_seq"] = pd.to_numeric(out[seq_col], errors="coerce")
    else:
        out["_audit_seq"] = pd.NA
    if out["_audit_seq"].isna().any():
        fallback = out.groupby(["_audit_line", "_audit_campaign"], dropna=False).cumcount() + 1
        out["_audit_seq"] = out["_audit_seq"].fillna(fallback)
    out["_audit_seq"] = pd.to_numeric(out["_audit_seq"], errors="coerce").fillna(0).astype(int)
    return out.sort_values(["_audit_line", "_audit_campaign", "_audit_seq"], kind="mergesort").reset_index(drop=True)


def _iter_adjacent(work: pd.DataFrame):
    for (line, campaign), grp in work.groupby(["_audit_line", "_audit_campaign"], dropna=False, sort=False):
        prev = None
        for _, row in grp.iterrows():
            if prev is not None:
                yield line, campaign, prev, row
            prev = row


def _thickness_rule(prev_thk: float, curr_thk: float) -> tuple[bool, str, float, float]:
    delta = abs(curr_thk - prev_thk)
    if prev_thk >= 0.8:
        ratio = delta / max(prev_thk, 1e-9)
        return ratio <= 0.30, "PREV_GE_0.8_RATIO_LE_30_PERCENT", delta, ratio
    if prev_thk >= 0.6:
        return delta <= 0.2, "PREV_0.6_TO_0.8_ABS_LE_0.2", delta, delta / max(prev_thk, 1e-9)
    return delta <= 0.1, "PREV_LT_0.6_ABS_LE_0.1", delta, delta / max(prev_thk, 1e-9)


def _group_transition_result(prev: pd.Series, cur: pd.Series) -> tuple[str, bool]:
    prev_group = str(prev.get("steel_group", "") or "")
    curr_group = str(cur.get("steel_group", "") or "")
    if prev_group == curr_group:
        return "OK", False

    edge_type = str(cur.get("selected_edge_type", cur.get("boundary_edge_type", "")) or "")
    if edge_type in {"REAL_BRIDGE_EDGE", "VIRTUAL_BRIDGE_EDGE", "VIRTUAL_BRIDGE_FAMILY_EDGE"}:
        return "OK_BRIDGE_EVIDENCE", False

    for col in ["group_transition_ok", "cross_group_transition_ok", "is_pc_transition"]:
        if col in cur.index:
            val = cur.get(col)
            if pd.isna(val):
                continue
            return ("OK_EXPLICIT_EVIDENCE", False) if bool(val) else ("GROUP_TRANSITION_VIOLATION", True)

    return "UNKNOWN_EVIDENCE", True


def run_final_schedule_audit(
    schedule_df: pd.DataFrame,
    rule: RuleConfig,
    engine_meta: dict | None = None,
) -> Dict[str, Any]:
    work = _prepare(schedule_df if isinstance(schedule_df, pd.DataFrame) else pd.DataFrame())
    summary: Dict[str, Any] = {}
    details: Dict[str, list[dict[str, Any]]] = {
        "width": [],
        "thickness": [],
        "temperature": [],
        "group_transition": [],
        "campaign_ton": [],
        "sequence": [],
    }

    width_pairs = width_increase = width_overdrop = 0
    thk_pairs = thk_viol = 0
    temp_pairs = temp_viol = 0
    group_pairs = group_viol = group_unknown = 0

    for line, campaign, prev, cur in _iter_adjacent(work):
        seq_a = int(prev.get("_audit_seq", 0) or 0)
        seq_b = int(cur.get("_audit_seq", 0) or 0)
        common = {
            "line": line,
            "campaign_id": campaign,
            "seq_a": seq_a,
            "seq_b": seq_b,
            "order_id_a": str(prev.get("_audit_order_id", "")),
            "order_id_b": str(cur.get("_audit_order_id", "")),
        }

        if "width" in work.columns:
            width_pairs += 1
            wa = _to_float(prev.get("width"))
            wb = _to_float(cur.get("width"))
            delta = wb - wa
            audit_result = "OK"
            if wb > wa:
                width_increase += 1
                audit_result = "WIDTH_INCREASE_VIOLATION"
            elif wa - wb > 250.0:
                width_overdrop += 1
                audit_result = "WIDTH_OVERDROP_VIOLATION"
            if audit_result != "OK":
                _append_limited(details["width"], {**common, "width_a": wa, "width_b": wb, "width_delta": delta, "audit_result": audit_result})

        if "thickness" in work.columns:
            thk_pairs += 1
            ta = _to_float(prev.get("thickness"))
            tb = _to_float(cur.get("thickness"))
            ok, band, delta, ratio = _thickness_rule(ta, tb)
            if not ok:
                thk_viol += 1
                _append_limited(
                    details["thickness"],
                    {
                        **common,
                        "thickness_a": ta,
                        "thickness_b": tb,
                        "thickness_rule_band": band,
                        "thickness_delta": delta,
                        "thickness_ratio_change": ratio,
                        "audit_result": "THICKNESS_VIOLATION",
                    },
                )

        if {"temp_min", "temp_max"}.issubset(work.columns):
            temp_pairs += 1
            amin = _to_float(prev.get("temp_min"))
            amax = _to_float(prev.get("temp_max"))
            bmin = _to_float(cur.get("temp_min"))
            bmax = _to_float(cur.get("temp_max"))
            overlap = min(amax, bmax) - max(amin, bmin)
            if overlap < 10.0:
                temp_viol += 1
                _append_limited(
                    details["temperature"],
                    {
                        **common,
                        "temp_min_a": amin,
                        "temp_max_a": amax,
                        "temp_min_b": bmin,
                        "temp_max_b": bmax,
                        "overlap_value": overlap,
                        "required_overlap": 10.0,
                        "audit_result": "TEMPERATURE_OVERLAP_VIOLATION",
                    },
                )

        if "steel_group" in work.columns:
            if str(prev.get("steel_group", "") or "") != str(cur.get("steel_group", "") or ""):
                group_pairs += 1
                result, is_problem = _group_transition_result(prev, cur)
                if is_problem:
                    group_viol += 1
                    if result == "UNKNOWN_EVIDENCE":
                        group_unknown += 1
                    _append_limited(
                        details["group_transition"],
                        {
                            **common,
                            "steel_group_a": str(prev.get("steel_group", "") or ""),
                            "steel_group_b": str(cur.get("steel_group", "") or ""),
                            "selected_edge_type": str(cur.get("selected_edge_type", cur.get("boundary_edge_type", "")) or ""),
                            "audit_result": result,
                        },
                    )

    campaign_ton_count = ton_min_viol = ton_max_viol = 0
    if not work.empty and "tons" in work.columns:
        real_mask = ~work.get("is_virtual", pd.Series([False] * len(work), index=work.index)).fillna(False).astype(bool)
        work["_real_tons"] = work["tons"].where(real_mask, 0.0).map(_to_float)
        work["_virtual_tons"] = work["tons"].where(~real_mask, 0.0).map(_to_float)
        tons = work.groupby(["_audit_line", "_audit_campaign"], dropna=False).agg(
            real_tons=("_real_tons", "sum"),
            virtual_tons=("_virtual_tons", "sum"),
        ).reset_index()
        min_limit = float(rule.campaign_ton_min)
        max_limit = float(rule.campaign_ton_max)
        for _, row in tons.iterrows():
            campaign_ton_count += 1
            total = float(row["real_tons"]) + float(row["virtual_tons"])
            audit_result = "OK"
            if total < min_limit:
                ton_min_viol += 1
                audit_result = "CAMPAIGN_TON_BELOW_MIN"
            elif total > max_limit:
                ton_max_viol += 1
                audit_result = "CAMPAIGN_TON_ABOVE_MAX"
            if audit_result != "OK":
                _append_limited(
                    details["campaign_ton"],
                    {
                        "line": str(row["_audit_line"]),
                        "campaign_id": str(row["_audit_campaign"]),
                        "real_tons": float(row["real_tons"]),
                        "virtual_tons": float(row["virtual_tons"]),
                        "total_tons": total,
                        "min_limit": min_limit,
                        "max_limit": max_limit,
                        "audit_result": audit_result,
                    },
                )

    grouping_viol = sequence_viol = duplicate_viol = 0
    if not work.empty:
        slot_counts = work.groupby(["_audit_line", "_audit_campaign"], dropna=False)["_audit_slot"].nunique(dropna=False)
        for (line, campaign), nslots in slot_counts.items():
            if int(nslots) > 1:
                grouping_viol += 1
                _append_limited(details["sequence"], {"line": line, "campaign_id": campaign, "audit_result": "CAMPAIGN_GROUPING_MULTIPLE_SLOTS", "slot_count": int(nslots)})

        for (line, campaign), grp in work.groupby(["_audit_line", "_audit_campaign"], dropna=False, sort=False):
            seqs = [int(v) for v in grp["_audit_seq"].tolist()]
            if len(seqs) != len(set(seqs)) or seqs != sorted(seqs) or (seqs and seqs != list(range(min(seqs), min(seqs) + len(seqs)))):
                sequence_viol += 1
                _append_limited(details["sequence"], {"line": line, "campaign_id": campaign, "sequence_values": seqs[:50], "audit_result": "CAMPAIGN_SEQUENCE_NOT_CONTINUOUS"})
            grp_orders = grp[grp["_audit_order_id"].astype(str).str.strip().ne("")]
            if "_audit_order_id" in grp_orders.columns and grp_orders["_audit_order_id"].duplicated().any():
                duplicate_viol += 1
                dupes = grp_orders.loc[grp_orders["_audit_order_id"].duplicated(keep=False), "_audit_order_id"].astype(str).unique().tolist()
                _append_limited(details["sequence"], {"line": line, "campaign_id": campaign, "duplicate_order_ids": dupes[:20], "audit_result": "DUPLICATE_ORDER_IN_CAMPAIGN"})
        nonblank_orders = work[work["_audit_order_id"].astype(str).str.strip().ne("")]
        if "_audit_order_id" in nonblank_orders.columns and nonblank_orders["_audit_order_id"].duplicated().any():
            duplicate_viol += 1
            dupes = nonblank_orders.loc[nonblank_orders["_audit_order_id"].duplicated(keep=False), "_audit_order_id"].astype(str).unique().tolist()
            _append_limited(details["sequence"], {"line": "ALL", "campaign_id": "ALL", "duplicate_order_ids": dupes[:20], "audit_result": "DUPLICATE_ORDER_GLOBAL"})

    final_hard_total = (
        width_increase + width_overdrop + thk_viol + temp_viol + group_viol
        + ton_min_viol + ton_max_viol + grouping_viol + sequence_viol + duplicate_viol
    )
    summary.update(
        {
            "width_audit_pair_count": int(width_pairs),
            "width_increase_violation_count": int(width_increase),
            "width_overdrop_violation_count": int(width_overdrop),
            "width_total_violation_count": int(width_increase + width_overdrop),
            "thickness_audit_pair_count": int(thk_pairs),
            "thickness_violation_count": int(thk_viol),
            "temperature_audit_pair_count": int(temp_pairs),
            "temperature_violation_count": int(temp_viol),
            "group_transition_pair_count": int(group_pairs),
            "group_transition_violation_count": int(group_viol),
            "group_transition_unknown_count": int(group_unknown),
            "campaign_ton_audit_count": int(campaign_ton_count),
            "campaign_ton_min_violation_count": int(ton_min_viol),
            "campaign_ton_max_violation_count": int(ton_max_viol),
            "campaign_grouping_violation_count": int(grouping_viol),
            "campaign_sequence_violation_count": int(sequence_viol),
            "duplicate_order_violation_count": int(duplicate_viol),
            "final_hard_violation_count_total": int(final_hard_total),
        }
    )
    return {"summary": summary, "details": details}
