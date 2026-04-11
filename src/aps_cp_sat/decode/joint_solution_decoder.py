from __future__ import annotations

"""
Template-based decode helpers.

Virtual slabs are modeled as template-arc attributes in the CP-SAT architecture.
They are materialized here only after solve. They must not be pushed back into
the master model as a large pool of explicit synthetic orders.
"""

from typing import List, Tuple

import pandas as pd

from aps_cp_sat.config.planner_config import PlannerConfig
from aps_cp_sat.repair import run_repair_pipeline
from aps_cp_sat.validate.checks import apply_solution_checks, weighted_penalties


def decode_candidate_allocation(
    candidate_joint: dict | None,
    orders_df: pd.DataFrame | None,
    *,
    candidate_dropped_df: pd.DataFrame | None = None,
    engine_meta: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Decode a best-so-far candidate into an analysis-only allocation draft.

    This does not assert routing feasibility and does not fabricate a legal
    final sequence. It only materializes the current line/slot membership.
    """
    joint = candidate_joint if isinstance(candidate_joint, dict) else {}
    source_orders = orders_df if isinstance(orders_df, pd.DataFrame) else pd.DataFrame()
    candidate_plan = joint.get("candidate_plan_df")
    if not isinstance(candidate_plan, pd.DataFrame) or candidate_plan.empty:
        candidate_plan = joint.get("plan_df") if isinstance(joint.get("plan_df"), pd.DataFrame) else pd.DataFrame()
    slot_details = (engine_meta or {}).get("slot_route_details", [])
    if not isinstance(slot_details, list):
        slot_details = []
    slot_map = {
        (str(item.get("line", "")), int(item.get("slot_no", 0) or 0)): dict(item)
        for item in slot_details
        if isinstance(item, dict)
    }

    rows: list[dict] = []
    if isinstance(candidate_plan, pd.DataFrame) and not candidate_plan.empty and not source_orders.empty:
        ordered = candidate_plan.copy()
        sort_cols = [c for c in ["assigned_line", "assigned_slot", "candidate_position", "master_seq", "row_idx"] if c in ordered.columns]
        if sort_cols:
            ordered = ordered.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
        for _, rec in ordered.iterrows():
            try:
                src = dict(source_orders.iloc[int(rec.get("row_idx", -1))])
            except Exception:
                src = {"order_id": rec.get("order_id", "")}
            line = str(rec.get("assigned_line", ""))
            slot_no = int(rec.get("assigned_slot", 0) or 0)
            slot_detail = slot_map.get((line, slot_no), {})
            slot_unroutable = bool(
                int(rec.get("slot_unroutable_flag", 0) or 0)
                or str(slot_detail.get("status", "")) == "UNROUTABLE_SLOT"
            )
            candidate_status = str(rec.get("candidate_status", "") or "")
            if not candidate_status:
                candidate_status = "UNROUTABLE_SLOT_MEMBER" if slot_unroutable else "ASSIGNED_CANDIDATE"
            rows.append(
                {
                    "order_id": str(src.get("order_id", rec.get("order_id", ""))),
                    "line": line,
                    "slot_no": slot_no,
                    "candidate_position": int(rec.get("candidate_position", rec.get("master_seq", 0) or 0)),
                    "candidate_slot_member_index": int(rec.get("candidate_slot_member_index", rec.get("candidate_position", rec.get("master_seq", 0) or 0))),
                    "tons": float(src.get("tons", 0.0) or 0.0),
                    "width": float(src.get("width", 0.0) or 0.0),
                    "thickness": float(src.get("thickness", 0.0) or 0.0),
                    "steel_group": str(src.get("steel_group", "")),
                    "line_capability": str(src.get("line_capability", "")),
                    "drop_flag": False,
                    "slot_unroutable_flag": slot_unroutable,
                    "slot_route_risk_score": int(slot_detail.get("slot_route_risk_score", rec.get("slot_route_risk_score", 0) or 0) or 0),
                    "candidate_status": candidate_status,
                    "selected_edge_type": str(rec.get("selected_edge_type", "")),
                    "selected_template_id": str(rec.get("selected_template_id", "")),
                    "selected_bridge_path": str(rec.get("selected_bridge_path", "")),
                    "bridge_count": len([p for p in str(rec.get("selected_bridge_path", "")).split("->") if str(p).strip()]),
                    "analysis_only": True,
                    "official_usable": False,
                }
            )

    dropped = candidate_dropped_df if isinstance(candidate_dropped_df, pd.DataFrame) else pd.DataFrame()
    if not dropped.empty:
        for _, row in dropped.iterrows():
            rows.append(
                {
                    "order_id": str(row.get("order_id", "")),
                    "line": "",
                    "slot_no": 0,
                    "candidate_position": 0,
                    "candidate_slot_member_index": 0,
                    "tons": float(row.get("tons", 0.0) or 0.0),
                    "width": float(row.get("width", 0.0) or 0.0),
                    "thickness": float(row.get("thickness", 0.0) or 0.0),
                    "steel_group": str(row.get("steel_group", "")),
                    "line_capability": str(row.get("line_capability", "")),
                    "drop_flag": True,
                    "slot_unroutable_flag": False,
                    "slot_route_risk_score": 0,
                    "candidate_status": "DROPPED_CANDIDATE",
                    "selected_edge_type": "",
                    "selected_template_id": "",
                    "selected_bridge_path": "",
                    "bridge_count": 0,
                    "dominant_drop_reason": str(row.get("dominant_drop_reason", row.get("drop_reason", ""))),
                    "secondary_reasons": str(row.get("secondary_reasons", "")),
                    "candidate_lines": str(row.get("candidate_lines", "")),
                    "risk_summary": str(row.get("risk_summary", "")),
                    "analysis_only": True,
                    "official_usable": False,
                }
            )

    candidate_schedule_df = pd.DataFrame(rows)
    if candidate_schedule_df.empty:
        cols = [
            "order_id", "line", "slot_no", "candidate_position", "tons", "width", "thickness", "steel_group",
            "line_capability", "drop_flag", "slot_unroutable_flag", "slot_route_risk_score", "candidate_status",
            "analysis_only", "official_usable",
        ]
        candidate_schedule_df = pd.DataFrame(columns=cols)
    else:
        candidate_schedule_df = candidate_schedule_df.sort_values(
            ["drop_flag", "line", "slot_no", "candidate_position", "order_id"],
            ascending=[True, True, True, True, True],
            kind="mergesort",
        ).reset_index(drop=True)
    big_roll_df = candidate_schedule_df[
        (candidate_schedule_df.get("line", pd.Series(dtype=str)) == "big_roll")
        & (~candidate_schedule_df.get("drop_flag", pd.Series(dtype=bool)).fillna(False))
    ].reset_index(drop=True)
    small_roll_df = candidate_schedule_df[
        (candidate_schedule_df.get("line", pd.Series(dtype=str)) == "small_roll")
        & (~candidate_schedule_df.get("drop_flag", pd.Series(dtype=bool)).fillna(False))
    ].reset_index(drop=True)
    return candidate_schedule_df, big_roll_df, small_roll_df


def decode_bridge_path_rows(
    encoded: str,
    line: str,
    line_name: str,
    campaign_id: int,
    master_slot: int,
    start_idx: int,
    virtual_tons: float,
) -> Tuple[List[dict], int]:
    if not encoded:
        return [], start_idx
    rows: List[dict] = []
    idx = start_idx
    vtons = float(virtual_tons)
    for part in [p.strip() for p in str(encoded).split("->") if p.strip()]:
        try:
            w_part, t_part, temp_part = part.split("|")
            width = float(w_part.replace("W", "").strip())
            thickness = float(t_part.replace("T", "").strip())
            temp_txt = temp_part.replace("TMP[", "").replace("]", "")
            temp_min_txt, temp_max_txt = [v.strip() for v in temp_txt.split(",")]
            temp_min = float(temp_min_txt)
            temp_max = float(temp_max_txt)
        except Exception:
            continue
        rows.append(
            {
                "order_id": f"VIRTUAL-{idx:05d}",
                "source_order_id": "",
                "parent_order_id": "",
                "lot_id": "",
                "grade": "VIRTUAL_PC",
                "steel_group": "PC",
                "line_capability": "dual",
                "width": width,
                "thickness": thickness,
                "temp_min": temp_min,
                "temp_max": temp_max,
                "temp_mean": (temp_min + temp_max) / 2.0,
                "tons": vtons,
                "is_virtual": True,
                "line": line,
                "line_name": line_name,
                "campaign_id": int(campaign_id),
                "master_slot": int(master_slot),
            }
        )
        idx += 1
    return rows, idx


def finalize_decoded_output(final_df: pd.DataFrame, cfg: PlannerConfig) -> pd.DataFrame:
    if final_df.empty:
        return final_df
    remapped = []
    for _, g in final_df.groupby("line", sort=False):
        if "line_order_local" in g.columns:
            g = g.sort_values("line_order_local", kind="mergesort")
        uniq = list(dict.fromkeys(g["campaign_id"].tolist()))
        mp = {old: i + 1 for i, old in enumerate(uniq)}
        gg = g.copy()
        gg["campaign_id"] = gg["campaign_id"].map(mp)
        remapped.append(gg)
    out = pd.concat(remapped, ignore_index=True)
    out["_line_order"] = out["line"].map(lambda x: {"big_roll": 0, "small_roll": 1}.get(str(x), 99))
    out = out.sort_values(["_line_order", "campaign_id", "seq"], kind="mergesort").reset_index(drop=True)
    out = out.drop(columns=["_line_order"])
    out["global_seq"] = out.index + 1
    out["line_seq"] = out.groupby("line").cumcount() + 1
    out["campaign_seq"] = out.groupby(["line", "campaign_id"]).cumcount() + 1
    out["campaign_real_seq"] = out.groupby(["line", "campaign_id"]).cumcount() + 1
    out["line_name"] = out["line"].map({"big_roll": "大辊线", "small_roll": "小辊线"}).fillna(out["line"])
    return apply_solution_checks(out, cfg.rule)


def materialize_master_plan(
    planned: pd.DataFrame,
    cfg: PlannerConfig,
    pre_dropped: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if planned.empty:
        dropped = pre_dropped.copy() if isinstance(pre_dropped, pd.DataFrame) else pd.DataFrame()
        return pd.DataFrame(), pd.DataFrame(), dropped
    all_parts: List[pd.DataFrame] = []
    dropped = pre_dropped.copy() if isinstance(pre_dropped, pd.DataFrame) else pd.DataFrame()
    for line, g in planned.groupby("line", sort=False):
        ordered = g.sort_values(["master_slot", "master_seq"], kind="mergesort").copy().reset_index(drop=True)
        recs = ordered.to_dict("records")
        rows: List[dict] = []
        vidx = 1
        for i, cur in enumerate(recs):
            slot = int(cur.get("master_slot", 0))
            cid = int(cur.get("campaign_id_hint", 0))
            if cid <= 0:
                cid = int(slot * 1000 + 1)
            cur_row = dict(cur)
            cur_row["is_virtual"] = bool(cur_row.get("is_virtual", False))
            cur_row["campaign_id"] = cid
            rows.append(cur_row)
            if i >= len(recs) - 1:
                continue
            nxt = recs[i + 1]
            if int(cur.get("campaign_id_hint", -1)) != int(nxt.get("campaign_id_hint", -1)):
                continue
            bridge_rows, vidx = decode_bridge_path_rows(
                str(cur.get("selected_bridge_path", "")),
                line=str(cur.get("line", line)),
                line_name=str(cur.get("line_name", "")),
                campaign_id=cid,
                master_slot=slot,
                start_idx=vidx,
                virtual_tons=float(cfg.rule.virtual_tons),
            )
            rows.extend(bridge_rows)
        line_df = pd.DataFrame(rows)
        if not line_df.empty:
            line_df["seq"] = range(1, len(line_df) + 1)
        all_parts.append(line_df)
    final_df = pd.concat(all_parts, ignore_index=True) if all_parts else planned.copy()
    final_df = finalize_decoded_output(final_df, cfg)
    final_df, repair_logs = run_repair_pipeline(final_df)
    round_rows: List[dict] = []
    for line, g in final_df.groupby("line", sort=False):
        pen = weighted_penalties(g, int(0), cfg.score, cfg.rule)
        round_rows.append(
            {
                "round": 1,
                "line": str(line),
                "rows_total": int(len(g)),
                "virtual_cnt": int(g["is_virtual"].sum()),
                "dropped_cnt": int(0),
                "width_jump_violation_cnt": int((g["width_jump_violation"] & (~g["is_virtual"])).sum()),
                "thickness_violation_cnt": int((g["thickness_violation"] & (~g["is_virtual"])).sum()),
                "non_pc_direct_switch_cnt": int((g["non_pc_direct_switch"] & (~g["is_virtual"])).sum()),
                "temp_conflict_cnt": int((g["temp_conflict"] & (~g["is_virtual"])).sum()),
                "model_seconds": 0.0,
                "solve_seconds": 0.0,
                "postprocess_seconds": 0.0,
                "elapsed_seconds": 0.0,
                "low_ton_campaign_cnt": int(pen.get("low_ton_campaign_cnt", 0)),
                "low_ton_gap_tons": float(pen.get("low_ton_gap_tons", 0.0)),
                "penalty_total": float(pen.get("total_penalty", 0.0)),
                "penalty_ton_under": float(pen.get("ton_under_pen", 0.0)),
                "penalty_unassigned": float(pen.get("unassigned_pen", 0.0)),
                "penalty_virtual_ratio": float(pen.get("virtual_ratio_pen", 0.0)),
                "penalty_pure_virtual_campaign": float(pen.get("pure_virtual_campaign_pen", 0.0)),
                "pure_virtual_campaign_cnt": int(pen.get("pure_virtual_campaign_cnt", 0.0)),
                "penalty_virtual_use": float(pen.get("virtual_use_pen", 0.0)),
                "rebalance_pass": 1,
                "repair_steps": int(len(repair_logs)),
            }
        )
    rounds_df = pd.DataFrame(round_rows)
    return final_df, rounds_df, dropped
