from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Dict
import pandas as pd

from aps_cp_sat.decode import decode_candidate_allocation, decode_solution
from aps_cp_sat.domain.models import ColdRollingRequest, ColdRollingResult
from aps_cp_sat.io import export_schedule_results
from aps_cp_sat.model import solve_master_model
from aps_cp_sat.preprocess import prepare_orders_for_model
from aps_cp_sat.rules import RULE_REGISTRY
from aps_cp_sat.transition import build_transition_templates
from aps_cp_sat.validate import validate_model_equivalence, validate_solution_summary


class ColdRollingPipeline:
    """
    冷轧排程分层门面：
    preprocess -> transition -> model -> decode -> validate
    """

    @staticmethod
    def _print_data_diagnostics(orders_df) -> None:
        if orders_df is None or orders_df.empty:
            print("[APS][数据诊断] 订单为空")
            return
        key_cols = ["width", "thickness", "temp_min", "temp_max", "tons", "steel_group", "line_capability"]
        missing: Dict[str, int] = {}
        for c in key_cols:
            if c in orders_df.columns:
                missing[c] = int(orders_df[c].isna().sum())
        bad_temp = 0
        if {"temp_min", "temp_max"}.issubset(set(orders_df.columns)):
            bad_temp = int((orders_df["temp_min"] > orders_df["temp_max"]).sum())
        cap_cnt = {}
        if "line_capability" in orders_df.columns:
            cap_cnt = orders_df["line_capability"].value_counts(dropna=False).to_dict()
        print(
            f"[APS][数据诊断] rows={len(orders_df)}, "
            f"missing={missing}, bad_temp_range={bad_temp}, line_capability={cap_cnt}"
        )

    @staticmethod
    def _print_template_diagnostics(transition_pack: dict) -> None:
        tpl = transition_pack.get("templates") if isinstance(transition_pack, dict) else None
        if tpl is None or tpl.empty:
            print("[APS][模板诊断] 无模板")
            return
        for line in sorted(tpl["line"].dropna().unique().tolist()):
            t = tpl[tpl["line"] == line]
            nodes = set(t["from_order_id"].astype(str)).union(set(t["to_order_id"].astype(str)))
            out_deg = t.groupby("from_order_id").size()
            in_deg = t.groupby("to_order_id").size()
            avg_out = float(out_deg.mean()) if len(out_deg) else 0.0
            avg_in = float(in_deg.mean()) if len(in_deg) else 0.0
            print(
                f"[APS][模板诊断] line={line}, templates={len(t)}, nodes={len(nodes)}, "
                f"avg_out={avg_out:.2f}, avg_in={avg_in:.2f}, "
                f"max_bridge={int(t['bridge_count'].max()) if not t.empty else 0}"
            )
        ps = transition_pack.get("prune_summaries", []) if isinstance(transition_pack, dict) else []
        for s in ps:
            print(
                f"[APS][模板剪枝] line={s.line}, unbridgeable={s.pruned_unbridgeable}, "
                f"topk={s.pruned_topk}, degree={s.pruned_degree}, kept={s.kept_templates}"
            )
        bd = transition_pack.get("build_debug", []) if isinstance(transition_pack, dict) else []
        if isinstance(bd, list):
            total = next((x for x in bd if str(x.get("line")) == "__all__"), None)
            for item in bd:
                line = str(item.get("line", ""))
                if line in {"big_roll", "small_roll"}:
                    print(
                        f"[APS][模板构建摘要] line={line}, candidatePairs={int(item.get('candidate_pairs', 0) or 0)}, "
                        f"templatesBuilt={int(item.get('kept_templates', 0) or 0)}, "
                        f"avgVirtualCount={float(item.get('avg_virtual_count', 0.0) or 0.0):.2f}, "
                        f"maxVirtualCount={int(item.get('max_virtual_count', 0) or 0)}, "
                        f"directEdgeCount={int(item.get('direct_edge_count', 0) or 0)}, "
                        f"realBridgeEdgeCount={int(item.get('real_bridge_edge_count', 0) or 0)}, "
                        f"virtualBridgeEdgeCount={int(item.get('virtual_bridge_edge_count', 0) or 0)}, "
                        f"rejectedByChainLimit={int(item.get('rejected_by_chain_limit', 0) or 0)}"
                    )
            if isinstance(total, dict):
                print(
                    f"[APS][模板耗时] preprocess_seconds={float(total.get('preprocess_seconds', 0.0)):.3f}, "
                    f"line_partition_seconds={float(total.get('line_partition_seconds', 0.0)):.3f}, "
                    f"template_pair_scan_seconds={float(total.get('template_pair_scan_seconds', 0.0)):.3f}, "
                    f"bridge_check_seconds={float(total.get('bridge_check_seconds', 0.0)):.3f}, "
                    f"template_prune_seconds={float(total.get('template_prune_seconds', 0.0)):.3f}, "
                    f"transition_pack_build_seconds={float(total.get('transition_pack_build_seconds', 0.0)):.3f}, "
                    f"diagnostics_build_seconds={float(total.get('diagnostics_build_seconds', 0.0)):.3f}, "
                    f"template_build_seconds={float(total.get('template_build_seconds', 0.0)):.3f}"
                )

    @staticmethod
    def _allowed_lines(line_capability: str) -> list[str]:
        cap = str(line_capability or "dual")
        if cap in {"big_only", "large", "LARGE"}:
            return ["big_roll"]
        if cap in {"small_only", "small", "SMALL"}:
            return ["small_roll"]
        return ["big_roll", "small_roll"]

    @staticmethod
    def _annotate_dropped_orders(orders_df: pd.DataFrame, dropped_df: pd.DataFrame, engine_meta: dict) -> pd.DataFrame:
        if dropped_df is None or dropped_df.empty:
            return dropped_df if isinstance(dropped_df, pd.DataFrame) else pd.DataFrame()
        out = dropped_df.copy()
        evidence = engine_meta.get("feasibility_evidence", {}) if isinstance(engine_meta, dict) else {}
        isolated_top = {
            str(item.get("order_id", "")): item
            for item in (evidence.get("isolated_orders_topn", []) if isinstance(evidence, dict) else [])
        }
        candidate_lines = []
        dominant_reasons = []
        secondary_reasons = []
        risk_summaries = []
        globally_isolated = []
        would_break_slot = []
        for _, row in out.iterrows():
            oid = str(row.get("order_id", ""))
            allowed = ColdRollingPipeline._allowed_lines(str(row.get("line_capability", "dual")))
            candidate_lines.append(",".join(allowed))
            iso = oid in isolated_top
            globally_isolated.append(bool(iso))
            reasons = []
            explicit_reason = str(row.get("drop_reason", "") or "").strip()
            if explicit_reason:
                reasons.append(explicit_reason)
            if iso:
                reasons.append("GLOBAL_ISOLATED_ORDER")
            if not allowed:
                reasons.append("NO_FEASIBLE_LINE")
            if explicit_reason == "FALLBACK_SCALE_UNSCHEDULED":
                reasons.append("CAPACITY_PRESSURE")
            if int(row.get("priority", 0) or 0) <= 0 or int(row.get("due_rank", 3) or 3) >= 2:
                reasons.append("LOW_PRIORITY_DROP")
            if engine_meta.get("unroutable_slot_count", 0):
                reasons.append("SLOT_ROUTING_RISK_TOO_HIGH")
            existing_dominant = str(row.get("dominant_drop_reason", "") or "").strip()
            dominant = existing_dominant or (reasons[0] if reasons else "OTHER")
            existing_secondary = [s for s in str(row.get("secondary_reasons", "") or "").split(",") if s]
            merged_secondary = list(dict.fromkeys(existing_secondary + reasons[1:]))
            existing_risk = str(row.get("risk_summary", "") or "").strip()
            dominant_reasons.append(dominant)
            secondary_reasons.append(",".join(merged_secondary))
            risk_summaries.append(existing_risk or ("|".join(reasons) if reasons else "OTHER"))
            existing_break = bool(row.get("would_break_slot_if_kept", False))
            would_break_slot.append(bool(existing_break or "SLOT_ROUTING_RISK_TOO_HIGH" in reasons or iso))
        if "globally_isolated" in out.columns:
            out["globally_isolated"] = out["globally_isolated"].fillna(pd.Series(globally_isolated)).astype(bool)
        else:
            out["globally_isolated"] = globally_isolated
        if "candidate_lines" in out.columns:
            out["candidate_lines"] = out["candidate_lines"].replace("", pd.NA).fillna(pd.Series(candidate_lines))
        else:
            out["candidate_lines"] = candidate_lines
        out["dominant_drop_reason"] = dominant_reasons
        out["secondary_reasons"] = secondary_reasons
        out["risk_summary"] = risk_summaries
        out["would_break_slot_if_kept"] = would_break_slot
        return out

    @staticmethod
    def _build_run_diagnostics(orders_df, transition_pack: dict, result: ColdRollingResult) -> dict:
        diagnostics: dict = {}
        em = result.engine_meta or {}

        tpl = transition_pack.get("templates") if isinstance(transition_pack, dict) else None
        summaries = transition_pack.get("summaries", []) if isinstance(transition_pack, dict) else []
        prune_summaries = transition_pack.get("prune_summaries", []) if isinstance(transition_pack, dict) else []
        template_section: dict = {}
        for s in summaries:
            template_section[f"{s.line}.coverage_pairs"] = f"{int(s.feasible_pairs)}/{int(s.candidate_pairs)}"
            template_section[f"{s.line}.coverage_ratio"] = round(int(s.feasible_pairs) / max(1, int(s.candidate_pairs)), 4)
            template_section[f"{s.line}.nodes"] = int(s.nodes)
            template_section[f"{s.line}.max_bridge_used"] = int(s.max_bridge_used)
        for p in prune_summaries:
            template_section[f"{p.line}.kept_templates"] = int(p.kept_templates)
            template_section[f"{p.line}.pruned_unbridgeable"] = int(p.pruned_unbridgeable)
            template_section[f"{p.line}.pruned_topk"] = int(p.pruned_topk)
            template_section[f"{p.line}.pruned_degree"] = int(p.pruned_degree)
        build_debug = transition_pack.get("build_debug", []) if isinstance(transition_pack, dict) else []
        if isinstance(build_debug, list):
            for item in build_debug:
                line = str(item.get("line", "unknown"))
                for key, value in item.items():
                    if key == "line":
                        continue
                    template_section[f"build_debug.{line}.{key}"] = value
        if isinstance(tpl, pd.DataFrame) and not tpl.empty:
            template_section["total_templates"] = int(len(tpl))
        diagnostics["template"] = template_section

        slot_section: dict = {}
        sched = result.schedule_df if isinstance(result.schedule_df, pd.DataFrame) else pd.DataFrame()
        if not sched.empty and {"line", "campaign_id", "tons", "is_virtual"}.issubset(set(sched.columns)):
            slot_loads = (
                sched.groupby(["line", "campaign_id"], dropna=False)["tons"]
                .sum()
                .reset_index(name="slot_tons")
            )
            low_slots = slot_loads[slot_loads["slot_tons"] < float(result.config.rule.campaign_ton_min)] if result.config else slot_loads.iloc[0:0]
            slot_section["slot_count"] = int(len(slot_loads))
            slot_section["low_slot_count"] = int(len(low_slots))
            slot_section["slot_tons_min"] = round(float(slot_loads["slot_tons"].min()), 1) if not slot_loads.empty else 0.0
            slot_section["slot_tons_max"] = round(float(slot_loads["slot_tons"].max()), 1) if not slot_loads.empty else 0.0
            slot_section["slot_tons_avg"] = round(float(slot_loads["slot_tons"].mean()), 1) if not slot_loads.empty else 0.0
        slot_section["unroutable_slot_count"] = int(em.get("unroutable_slot_count", 0))
        if isinstance(em.get("joint_estimates"), dict):
            slot_section["slot_route_risk_score"] = int(em["joint_estimates"].get("slot_route_risk_score", em.get("slot_route_risk_score", 0) or 0))
            slot_section["max_slot_order_count"] = int(em["joint_estimates"].get("max_slot_order_count", em.get("max_slot_order_count", 0) or 0))
            slot_section["avg_slot_order_count"] = float(em["joint_estimates"].get("avg_slot_order_count", em.get("avg_slot_order_count", 0.0) or 0.0))
            slot_section["big_roll_max_slot_order_count"] = int(em["joint_estimates"].get("big_roll_max_slot_order_count", em.get("big_roll_max_slot_order_count", 0) or 0))
            slot_section["small_roll_max_slot_order_count"] = int(em["joint_estimates"].get("small_roll_max_slot_order_count", em.get("small_roll_max_slot_order_count", 0) or 0))
            slot_section["big_roll_slot_order_hard_cap"] = int(em["joint_estimates"].get("big_roll_slot_order_hard_cap", em.get("big_roll_slot_order_hard_cap", 0) or 0))
            slot_section["big_roll_order_cap_violations"] = int(em["joint_estimates"].get("big_roll_order_cap_violations", em.get("big_roll_order_cap_violations", 0) or 0))
            slot_section["hard_cap_not_enforced"] = bool(em["joint_estimates"].get("hard_cap_not_enforced", em.get("hard_cap_not_enforced", False) or False))
        else:
            slot_section["slot_route_risk_score"] = int(em.get("slot_route_risk_score", 0) or 0)
            slot_section["max_slot_order_count"] = int(em.get("max_slot_order_count", 0) or 0)
            slot_section["avg_slot_order_count"] = float(em.get("avg_slot_order_count", 0.0) or 0.0)
            slot_section["big_roll_max_slot_order_count"] = int(em.get("big_roll_max_slot_order_count", 0) or 0)
            slot_section["small_roll_max_slot_order_count"] = int(em.get("small_roll_max_slot_order_count", 0) or 0)
            slot_section["big_roll_slot_order_hard_cap"] = int(em.get("big_roll_slot_order_hard_cap", 0) or 0)
            slot_section["big_roll_order_cap_violations"] = int(em.get("big_roll_order_cap_violations", 0) or 0)
            slot_section["hard_cap_not_enforced"] = bool(em.get("hard_cap_not_enforced", False) or False)
        if isinstance(em.get("slot_route_details"), list):
            for idx, item in enumerate(em["slot_route_details"], start=1):
                slot_section[f"slot{idx}"] = str(item)
            slot_section["unroutable_slots_topn"] = [item for item in em["slot_route_details"] if str(item.get("status", "")) == "UNROUTABLE_SLOT"][:5]
        if isinstance(orders_df, pd.DataFrame) and not orders_df.empty and result.config:
            total_tons = float(orders_df["tons"].sum()) if "tons" in orders_df.columns else 0.0
            target = float(result.config.rule.campaign_ton_target)
            est_slots = max(1, int(round(total_tons / max(1.0, target))))
            slot_section["estimated_slots"] = int(est_slots)
            slot_section["slot_cap"] = int(result.config.max_campaign_slots)
        diagnostics["slot"] = slot_section

        dropped = result.dropped_df if isinstance(result.dropped_df, pd.DataFrame) else pd.DataFrame()
        dropped = ColdRollingPipeline._annotate_dropped_orders(orders_df, dropped, em)
        unassigned_section: dict = {
            "count": int(len(dropped)),
            "tons": round(float(dropped["tons"].sum()), 1) if (not dropped.empty and "tons" in dropped.columns) else 0.0,
        }
        if not dropped.empty and "dominant_drop_reason" in dropped.columns:
            vc = dropped["dominant_drop_reason"].fillna("UNKNOWN").value_counts()
            topn = vc.head(5)
            for idx, (reason, cnt) in enumerate(topn.items(), start=1):
                unassigned_section[f"top{idx}_reason"] = str(reason)
                unassigned_section[f"top{idx}_count"] = int(cnt)
            if not vc.empty:
                unassigned_section["top_reason"] = str(vc.index[0])
            unassigned_section["dropped_reason_histogram"] = vc.to_dict()
        diagnostics["unassigned"] = unassigned_section

        diagnostics["fallback"] = {
            "profile_name": str(em.get("profile_name", getattr(result.config.model, "profile_name", "default")) if result.config else em.get("profile_name", "default")),
            "assignment_pressure_mode": str(em.get("assignment_pressure_mode", "normal")),
            "engine_used": str(em.get("engine_used", "unknown")),
            "main_path": str(em.get("main_path", em.get("engine_used", "unknown"))),
            "master_entry": str(em.get("master_entry", "_run_global_joint_model")),
            "used_local_routing": bool(em.get("used_local_routing", False)),
            "local_routing_role": str(em.get("local_routing_role", "not_used")),
            "bridge_modeling": str(em.get("bridge_modeling", "template_based")),
            "fallback_used": bool(em.get("fallback_used", False)),
            "fallback_type": str(em.get("fallback_type", "")),
            "fallback_reason": str(em.get("fallback_reason", "")),
            "trace_count": int(len(em.get("fallback_trace", []))) if isinstance(em.get("fallback_trace"), list) else 0,
            "strict_template_edges_enabled": bool(em.get("strict_template_edges_enabled", result.config.model.strict_template_edges)) if result.config else bool(em.get("strict_template_edges_enabled", True)),
            "routing_feasible": bool(em.get("routing_feasible", False)),
            "routing_status": str(em.get("routing_status", "UNKNOWN")),
            "template_pair_ok": bool(em.get("template_pair_ok", False)),
            "adjacency_rule_ok": bool(em.get("adjacency_rule_ok", False)),
            "bridge_expand_ok": bool(em.get("bridge_expand_ok", False)),
            "unroutable_slot_count": int(em.get("unroutable_slot_count", 0)),
            "result_acceptance_status": str(em.get("result_acceptance_status", "UNKNOWN")),
            "failure_mode": str(em.get("failure_mode", "")),
            "evidence_level": str(em.get("evidence_level", "OK")),
            "drop_strategy_applied": bool(em.get("drop_strategy_applied", False)),
            "drop_stage": str(em.get("drop_stage", "")),
            "drop_budget_used": int(em.get("drop_budget_used", 0)),
            "drop_budget_remaining": int(em.get("drop_budget_remaining", 0)),
            "dropped_candidates_considered": int(em.get("dropped_candidates_considered", 0)),
            "dropped_candidates_selected": int(em.get("dropped_candidates_selected", 0)),
            "dominant_drop_reason_histogram": em.get("dominant_drop_reason_histogram", {}),
            "structure_fallback_applied": bool(em.get("structure_fallback_applied", False)),
            "fallback_mode": str(em.get("fallback_mode", "")),
            "structure_first_applied": bool(em.get("structure_first_applied", False)),
            "time_expansion_applied": bool(em.get("time_expansion_applied", False)),
            "slot_count_before_fallback": int(em.get("slot_count_before_fallback", 0)),
            "slot_count_after_fallback": int(em.get("slot_count_after_fallback", 0)),
            "drop_budget_after_fallback": int(em.get("drop_budget_after_fallback", 0)),
            "restructured_slot_count": int(em.get("restructured_slot_count", 0)),
            "bad_slots_before_restructure": em.get("bad_slots_before_restructure", []),
            "bad_slots_after_restructure": em.get("bad_slots_after_restructure", []),
            "orders_removed_from_bad_slots": em.get("orders_removed_from_bad_slots", []),
            "partial_result_available": bool(em.get("partial_result_available", False)),
            "partial_acceptance_passed": bool(em.get("partial_acceptance_passed", False)),
            "partial_drop_ratio": float(em.get("partial_drop_ratio", 0.0) or 0.0),
            "partial_drop_tons_ratio": float(em.get("partial_drop_tons_ratio", 0.0) or 0.0),
            "template_graph_health": str(em.get("template_graph_health", "UNKNOWN")),
            "precheck_autorelax_applied": bool(em.get("precheck_autorelax_applied", False)),
            "solve_attempt_count": int(em.get("solve_attempt_count", 0)),
            "fallback_attempt_count": int(em.get("fallback_attempt_count", 0)),
            "early_stop_reason": str(em.get("early_stop_reason", "")),
            "early_stop_deferred_for_semantic_fallback": bool(em.get("early_stop_deferred_for_semantic_fallback", False)),
            "semantic_fallback_first_attempt_status": str(em.get("semantic_fallback_first_attempt_status", "")),
            "export_failed_result_for_debug": bool(em.get("export_failed_result_for_debug", False)),
            "export_analysis_on_failure": bool(em.get("export_analysis_on_failure", False)),
            "export_best_candidate_analysis": bool(em.get("export_best_candidate_analysis", False)),
            "final_export_performed": bool(em.get("final_export_performed", False)),
            "official_exported": bool(em.get("official_exported", False)),
            "analysis_exported": bool(em.get("analysis_exported", False)),
            "result_usage": str(em.get("result_usage", "UNKNOWN")),
            "best_candidate_available": bool(em.get("best_candidate_available", False)),
            "best_candidate_type": str(em.get("best_candidate_type", "")),
            "best_candidate_objective": float(em.get("best_candidate_objective", 0.0) or 0.0),
            "best_candidate_search_status": str(em.get("best_candidate_search_status", "")),
            "best_candidate_routing_feasible": bool(em.get("best_candidate_routing_feasible", False)),
            "best_candidate_unroutable_slot_count": int(em.get("best_candidate_unroutable_slot_count", 0)),
            "candidate_line_summary_available": bool(em.get("candidate_line_summary_available", False)),
            "candidate_schedule_rows": int(em.get("candidate_schedule_rows", 0)),
            "candidate_big_roll_rows": int(em.get("candidate_big_roll_rows", 0)),
            "candidate_small_roll_rows": int(em.get("candidate_small_roll_rows", 0)),
            "export_consistency_ok": bool(em.get("export_consistency_ok", True)),
            "template_build_seconds": float(em.get("template_build_seconds", 0.0)),
            "joint_master_seconds": float(em.get("joint_master_seconds", 0.0)),
            "local_router_seconds": float(em.get("local_router_seconds", 0.0)),
            "fallback_total_seconds": float(em.get("fallback_total_seconds", 0.0)),
            "total_run_seconds": float(em.get("total_run_seconds", 0.0)),
            "bridge_mix_summary": {
                "selected_direct_edge_count": int(em.get("selected_direct_edge_count", 0)),
                "selected_real_bridge_edge_count": int(em.get("selected_real_bridge_edge_count", 0)),
                "selected_virtual_bridge_edge_count": int(em.get("selected_virtual_bridge_edge_count", 0)),
                "direct_edge_ratio": float(em.get("direct_edge_ratio", 0.0) or 0.0),
                "real_bridge_ratio": float(em.get("real_bridge_ratio", 0.0) or 0.0),
                "virtual_bridge_ratio": float(em.get("virtual_bridge_ratio", 0.0) or 0.0),
            },
        }
        if isinstance(em.get("top_infeasibility_signals"), list):
            diagnostics["fallback"]["top_infeasibility_signals"] = list(em.get("top_infeasibility_signals", []))
        if isinstance(em.get("fallback_trace"), list):
            for idx, item in enumerate(em["fallback_trace"], start=1):
                diagnostics["fallback"][f"trace{idx}"] = str(item)

        diagnostics["rules"] = {
            spec.key.value: spec.zh_name
            for spec in RULE_REGISTRY.export_visible_specs()
        }
        diagnostics["rule_semantics"] = {
            "line_compatibility": "HARD",
            "direct_transition_feasibility": "HARD",
            "campaign_ton_max": "HARD",
            "campaign_ton_min": "STRONG_SOFT",
            "unassigned_real_orders": "STRONG_SOFT",
            "virtual_slab_usage_ratio": "STRONG_SOFT",
            "reverse_width_count_total": "HARD",
        }
        if isinstance(em.get("joint_estimates"), dict) and em.get("joint_estimates"):
            diagnostics["joint_estimates"] = dict(em["joint_estimates"])
        if isinstance(em.get("feasibility_evidence"), dict) and em.get("feasibility_evidence"):
            diagnostics["feasibility_evidence"] = dict(em["feasibility_evidence"])
        if isinstance(em.get("slot_route_details"), list) and em.get("slot_route_details"):
            diagnostics["slot_route_details"] = {f"slot_{idx}": item for idx, item in enumerate(em["slot_route_details"], start=1)}

        raw_fail = em.get("failure_diagnostics")
        if isinstance(raw_fail, dict) and raw_fail:
            diagnostics["failure"] = raw_fail
        template_debug = diagnostics.get("template", {})
        known_template_seconds = float(template_debug.get("build_debug.__all__.preprocess_seconds", 0.0) or 0.0)
        known_template_seconds += float(template_debug.get("build_debug.__all__.line_partition_seconds", 0.0) or 0.0)
        known_template_seconds += float(template_debug.get("build_debug.__all__.template_pair_scan_seconds", 0.0) or 0.0)
        known_template_seconds += float(template_debug.get("build_debug.__all__.bridge_check_seconds", 0.0) or 0.0)
        known_template_seconds += float(template_debug.get("build_debug.__all__.template_prune_seconds", 0.0) or 0.0)
        known_template_seconds += float(template_debug.get("build_debug.__all__.transition_pack_build_seconds", 0.0) or 0.0)
        known_template_seconds += float(template_debug.get("build_debug.__all__.diagnostics_build_seconds", 0.0) or 0.0)
        total_run_seconds = float(diagnostics.get("fallback", {}).get("total_run_seconds", 0.0) or 0.0)
        accounted = known_template_seconds
        accounted += float(diagnostics.get("fallback", {}).get("joint_master_seconds", 0.0) or 0.0)
        accounted += float(diagnostics.get("fallback", {}).get("local_router_seconds", 0.0) or 0.0)
        accounted += float(diagnostics.get("fallback", {}).get("fallback_total_seconds", 0.0) or 0.0)
        ratio = float(accounted / max(1e-9, total_run_seconds)) if total_run_seconds > 0 else 0.0
        diagnostics["fallback"]["timing_accounted_ratio"] = round(ratio, 4)
        diagnostics["fallback"]["timing_gap_detected"] = bool(total_run_seconds > 0 and ratio < 0.85)
        if diagnostics["fallback"]["timing_gap_detected"]:
            diagnostics["fallback"]["timing_gap_hint"] = "unaccounted_runtime_outside_recorded_template_build_and_solve_stages"
        return diagnostics

    @staticmethod
    def _evaluate_partial_acceptance(result: ColdRollingResult, orders_df: pd.DataFrame) -> dict:
        dropped_df = result.dropped_df if isinstance(result.dropped_df, pd.DataFrame) else pd.DataFrame()
        schedule_df = result.schedule_df if isinstance(result.schedule_df, pd.DataFrame) else pd.DataFrame()
        total_orders = int(len(orders_df)) if isinstance(orders_df, pd.DataFrame) else 0
        total_tons = float(orders_df["tons"].sum()) if isinstance(orders_df, pd.DataFrame) and "tons" in orders_df.columns else 0.0
        dropped_order_count = int(len(dropped_df))
        dropped_tons = float(dropped_df["tons"].sum()) if "tons" in dropped_df.columns and not dropped_df.empty else 0.0
        real_sched = schedule_df[~schedule_df["is_virtual"]] if isinstance(schedule_df, pd.DataFrame) and "is_virtual" in schedule_df.columns else schedule_df
        scheduled_orders = int(len(real_sched)) if isinstance(real_sched, pd.DataFrame) else 0
        scheduled_tons = float(real_sched["tons"].sum()) if isinstance(real_sched, pd.DataFrame) and "tons" in real_sched.columns else 0.0
        drop_ratio = float(dropped_order_count / max(1, total_orders))
        drop_tons_ratio = float(dropped_tons / max(1e-9, total_tons)) if total_tons > 0 else 0.0
        cfg = result.config.model
        passed = (
            dropped_order_count > 0
            and drop_ratio <= float(cfg.max_drop_ratio_for_partial)
            and drop_tons_ratio <= float(cfg.max_drop_tons_ratio_for_partial)
            and scheduled_orders >= int(cfg.min_scheduled_orders_for_partial)
            and scheduled_tons >= float(cfg.min_scheduled_tons_for_partial)
        )
        return {
            "partial_result_available": bool(dropped_order_count > 0),
            "partial_acceptance_passed": bool(passed),
            "partial_drop_ratio": drop_ratio,
            "partial_drop_tons_ratio": drop_tons_ratio,
            "scheduled_orders": scheduled_orders,
            "scheduled_tons": scheduled_tons,
        }

    def run(self, req: ColdRollingRequest) -> ColdRollingResult:
        run_t0 = perf_counter()
        print(f"[APS][Profile] name={req.config.model.profile_name}")
        print(
            f"[APS][Profile] unassigned_real={req.config.score.unassigned_real}, "
            f"route_risk=({req.config.score.slot_isolation_risk_penalty},"
            f"{req.config.score.slot_pair_gap_risk_penalty},{req.config.score.slot_span_risk_penalty}), "
            f"slot_order_cap=({req.config.model.big_roll_slot_soft_order_cap},{req.config.model.small_roll_slot_soft_order_cap}), "
            f"slot_order_hard_cap=({req.config.model.big_roll_slot_hard_order_cap},{req.config.model.small_roll_slot_hard_order_cap}), "
            f"slot_order_penalty={req.config.score.slot_order_count_penalty}"
        )
        orders_df = prepare_orders_for_model(req.orders_path, req.steel_info_path, req.config)
        self._print_data_diagnostics(orders_df)

        build_t0 = perf_counter()
        transition_pack = build_transition_templates(orders_df, req.config)
        template_build_seconds = perf_counter() - build_t0
        self._print_template_diagnostics(transition_pack)

        schedule_df, rounds_df, dropped_df, engine_meta = solve_master_model(req, transition_pack=transition_pack, orders_df=orders_df)
        effective_cfg = engine_meta.get("effective_config", req.config) if isinstance(engine_meta, dict) else req.config
        if isinstance(engine_meta, dict) and engine_meta.get("precheck_autorelax_applied") and effective_cfg is not req.config:
            transition_pack = build_transition_templates(orders_df, effective_cfg)
        result = ColdRollingResult(
            schedule_df=schedule_df,
            rounds_df=rounds_df,
            output_path=Path(req.output_path),
            dropped_df=dropped_df,
            engine_meta=engine_meta,
            config=effective_cfg,
        )

        result = decode_solution(result)
        annotated_dropped = ColdRollingPipeline._annotate_dropped_orders(
            orders_df,
            result.dropped_df if isinstance(result.dropped_df, pd.DataFrame) else pd.DataFrame(),
            result.engine_meta or {},
        )
        result = replace(result, dropped_df=annotated_dropped)

        updated_engine_meta = dict(result.engine_meta or {})
        candidate_schedule_df = pd.DataFrame()
        candidate_big_roll_df = pd.DataFrame()
        candidate_small_roll_df = pd.DataFrame()
        if bool(updated_engine_meta.get("best_candidate_available", False)):
            best_candidate_joint = updated_engine_meta.get("best_candidate_joint")
            best_candidate_source_orders = updated_engine_meta.get("best_candidate_source_orders_df")
            candidate_schedule_df, candidate_big_roll_df, candidate_small_roll_df = decode_candidate_allocation(
                best_candidate_joint if isinstance(best_candidate_joint, dict) else {},
                best_candidate_source_orders if isinstance(best_candidate_source_orders, pd.DataFrame) else orders_df,
                candidate_dropped_df=result.dropped_df if isinstance(result.dropped_df, pd.DataFrame) else pd.DataFrame(),
                engine_meta=updated_engine_meta,
            )
            updated_engine_meta["candidate_schedule_df"] = candidate_schedule_df
            updated_engine_meta["candidate_big_roll_df"] = candidate_big_roll_df
            updated_engine_meta["candidate_small_roll_df"] = candidate_small_roll_df
            updated_engine_meta["candidate_schedule_rows"] = int(len(candidate_schedule_df))
            updated_engine_meta["candidate_big_roll_rows"] = int(len(candidate_big_roll_df))
            updated_engine_meta["candidate_small_roll_rows"] = int(len(candidate_small_roll_df))
            updated_engine_meta["candidate_line_summary_available"] = bool(not candidate_schedule_df.empty)
        acceptance_before_gate = str(updated_engine_meta.get("result_acceptance_status", ""))
        if acceptance_before_gate in {"BEST_SEARCH_CANDIDATE_ANALYSIS"}:
            summary = validate_solution_summary(result, result.config.rule)
            if result.schedule_df is None or result.schedule_df.empty:
                eq = {
                    "template_pair_ok": bool(updated_engine_meta.get("template_pair_ok", False)),
                    "adjacency_rule_ok": bool(updated_engine_meta.get("adjacency_rule_ok", False)),
                    "bridge_expand_ok": bool(updated_engine_meta.get("bridge_expand_ok", False)),
                }
                routing_feasible = False
            else:
                eq = validate_model_equivalence(
                    result.schedule_df,
                    transition_pack.get("templates") if isinstance(transition_pack, dict) else None,
                )
                routing_feasible = bool(
                    eq.get("template_pair_ok", False)
                    and eq.get("adjacency_rule_ok", False)
                    and eq.get("bridge_expand_ok", False)
                )
            updated_engine_meta["routing_feasible"] = bool(routing_feasible)
            updated_engine_meta["routing_status"] = "OK" if routing_feasible else "ROUTING_INFEASIBLE"
            updated_engine_meta["template_pair_ok"] = bool(eq.get("template_pair_ok", False))
            updated_engine_meta["adjacency_rule_ok"] = bool(eq.get("adjacency_rule_ok", False))
            updated_engine_meta["bridge_expand_ok"] = bool(eq.get("bridge_expand_ok", False))
            updated_engine_meta["result_acceptance_status"] = "BEST_SEARCH_CANDIDATE_ANALYSIS"
            updated_engine_meta["result_usage"] = "ANALYSIS_ONLY"
            updated_engine_meta["failure_mode"] = str(updated_engine_meta.get("failure_mode", "FAILED_ROUTING_SEARCH") or "FAILED_ROUTING_SEARCH")
        elif str(updated_engine_meta.get("result_acceptance_status", "")).startswith("FAILED_") and (
            result.schedule_df is None or result.schedule_df.empty
        ):
            summary = validate_solution_summary(result, result.config.rule)
            eq = {
                "template_pair_ok": bool(updated_engine_meta.get("template_pair_ok", False)),
                "adjacency_rule_ok": bool(updated_engine_meta.get("adjacency_rule_ok", False)),
                "bridge_expand_ok": bool(updated_engine_meta.get("bridge_expand_ok", False)),
            }
            routing_feasible = False
        else:
            summary = validate_solution_summary(result, result.config.rule)
            eq = validate_model_equivalence(
                result.schedule_df,
                transition_pack.get("templates") if isinstance(transition_pack, dict) else None,
            )
            routing_feasible = bool(
                eq.get("template_pair_ok", False)
                and eq.get("adjacency_rule_ok", False)
                and eq.get("bridge_expand_ok", False)
            )
            updated_engine_meta["routing_feasible"] = bool(routing_feasible)
            updated_engine_meta["routing_status"] = "OK" if routing_feasible else "ROUTING_INFEASIBLE"
            updated_engine_meta["template_pair_ok"] = bool(eq.get("template_pair_ok", False))
            updated_engine_meta["adjacency_rule_ok"] = bool(eq.get("adjacency_rule_ok", False))
            updated_engine_meta["bridge_expand_ok"] = bool(eq.get("bridge_expand_ok", False))
            if routing_feasible:
                partial_eval = self._evaluate_partial_acceptance(result, orders_df)
                updated_engine_meta["partial_result_available"] = bool(partial_eval["partial_result_available"])
                updated_engine_meta["partial_acceptance_passed"] = bool(partial_eval["partial_acceptance_passed"])
                updated_engine_meta["partial_drop_ratio"] = float(partial_eval["partial_drop_ratio"])
                updated_engine_meta["partial_drop_tons_ratio"] = float(partial_eval["partial_drop_tons_ratio"])
                if isinstance(result.dropped_df, pd.DataFrame) and not result.dropped_df.empty and partial_eval["partial_acceptance_passed"]:
                    updated_engine_meta["result_acceptance_status"] = "PARTIAL_SCHEDULE_WITH_DROPS"
                    updated_engine_meta["failure_mode"] = "PARTIAL_SCHEDULE_WITH_DROPS"
                    updated_engine_meta["result_usage"] = "PARTIAL_OFFICIAL"
                elif isinstance(result.dropped_df, pd.DataFrame) and not result.dropped_df.empty:
                    updated_engine_meta["result_acceptance_status"] = "BEST_SEARCH_CANDIDATE_ANALYSIS"
                    updated_engine_meta["failure_mode"] = "FAILED_PARTIAL_ACCEPTANCE_THRESHOLD"
                    updated_engine_meta["result_usage"] = "ANALYSIS_ONLY"
                    print(
                        "[APS][结果门槛] routing 已合法，但 partial 阈值未通过，"
                        "结果降级为 BEST_SEARCH_CANDIDATE_ANALYSIS"
                    )
                else:
                    updated_engine_meta["result_acceptance_status"] = "OFFICIAL_FULL_SCHEDULE"
                    updated_engine_meta["failure_mode"] = ""
                    updated_engine_meta["result_usage"] = "OFFICIAL"
            else:
                updated_engine_meta["result_acceptance_status"] = "FAILED_ROUTING_SEARCH"
                updated_engine_meta["result_usage"] = "ANALYSIS_ONLY"
        if bool(updated_engine_meta.get("hard_cap_not_enforced", False)):
            updated_engine_meta["result_acceptance_status"] = "FAILED_IMPLEMENTATION_ERROR"
            updated_engine_meta["failure_mode"] = "FAILED_IMPLEMENTATION_ERROR"
        updated_engine_meta["export_failed_result_for_debug"] = bool(result.config.model.export_failed_result_for_debug)
        updated_engine_meta["export_analysis_on_failure"] = bool(result.config.model.export_analysis_on_failure)
        updated_engine_meta["export_best_candidate_analysis"] = bool(result.config.model.export_best_candidate_analysis)
        updated_engine_meta["template_build_seconds"] = float(template_build_seconds) + float(updated_engine_meta.get("template_build_seconds", 0.0) or 0.0)
        export_path = result.output_path
        final_export_performed = True
        acceptance_status = str(updated_engine_meta.get("result_acceptance_status", ""))
        official_exported = bool(routing_feasible and acceptance_status in {"OFFICIAL_FULL_SCHEDULE", "PARTIAL_SCHEDULE_WITH_DROPS"})
        analysis_exported = False
        result_usage = "OFFICIAL" if official_exported else "NOT_EXPORTED"
        if not official_exported:
            if acceptance_status == "BEST_SEARCH_CANDIDATE_ANALYSIS" and bool(result.config.model.export_best_candidate_analysis):
                export_path = result.output_path.with_name(
                    f"{result.output_path.stem}_BEST_SEARCH_CANDIDATE_ANALYSIS{result.output_path.suffix}"
                )
                print(f"[APS][结果门槛] best_candidate_exported=True, official_exported=False, result_usage=ANALYSIS_ONLY, 该文件仅用于调优分析，不可下发: {export_path}")
                analysis_exported = True
                official_exported = False
                result_usage = "ANALYSIS_ONLY"
            elif bool(result.config.model.export_failed_result_for_debug):
                export_path = result.output_path.with_name(
                    f"{result.output_path.stem}_ROUTING_INFEASIBLE_NOT_PRODUCTION_READY{result.output_path.suffix}"
                )
                print(f"[APS][结果门槛] routing 不合法，结果仅按调试输出导出: {export_path}")
                analysis_exported = True
                official_exported = False
                result_usage = "ANALYSIS_ONLY"
            elif bool(result.config.model.export_analysis_on_failure):
                export_path = result.output_path.with_name(
                    f"{result.output_path.stem}_FAILED_ROUTING_ANALYSIS{result.output_path.suffix}"
                )
                print(f"[APS][结果门槛] analysis_exported=True, 正式排程不可下发, 该文件仅用于排查与调优: {export_path}")
                analysis_exported = True
                official_exported = False
                result_usage = "ANALYSIS_ONLY"
            else:
                final_export_performed = False
                official_exported = False
                analysis_exported = False
                result_usage = "NOT_EXPORTED"
                print("[APS][结果门槛] routing 不合法，已禁止失败结果导出")
        else:
            official_exported = True
            analysis_exported = False
            result_usage = str(updated_engine_meta.get("result_usage", "OFFICIAL"))
        updated_engine_meta["final_export_performed"] = bool(final_export_performed)
        updated_engine_meta["official_exported"] = bool(official_exported)
        updated_engine_meta["analysis_exported"] = bool(analysis_exported)
        updated_engine_meta["result_usage"] = str(result_usage)
        updated_engine_meta["total_run_seconds"] = float(perf_counter() - run_t0)
        updated_engine_meta["export_consistency_ok"] = True
        result = replace(result, engine_meta=updated_engine_meta, output_path=export_path)
        em = result.engine_meta or {}
        diagnostics = self._build_run_diagnostics(orders_df, transition_pack, result)
        diagnostics["validation_summary"] = dict(summary)
        if bool(em.get("best_candidate_available", False)) and int(em.get("candidate_schedule_rows", 0) or 0) > 0:
            print(
                f"[APS][候选导出] candidate_schedule_available=True, rows={int(em.get('candidate_schedule_rows', 0))}, "
                f"big_roll_rows={int(em.get('candidate_big_roll_rows', 0))}, "
                f"small_roll_rows={int(em.get('candidate_small_roll_rows', 0))}"
            )
        print(
            f"[APS][运行诊断] profile={req.config.model.profile_name}, "
            f"engine={em.get('engine_used', 'unknown')}, main_path={em.get('main_path', 'unknown')}, "
            f"local_router={em.get('local_routing_role', 'not_used')}, "
            f"routing_feasible={routing_feasible}, "
            f"acceptance={em.get('result_acceptance_status', 'UNKNOWN')}, "
            f"failure_mode={em.get('failure_mode', '')}, "
            f"evidence_level={em.get('evidence_level', 'OK')}, "
            f"export_failed={em.get('export_failed_result_for_debug', False)}, "
            f"analysis_on_failure={em.get('export_analysis_on_failure', False)}, "
            f"exported={em.get('final_export_performed', False)}, "
            f"official_exported={em.get('official_exported', False)}, "
            f"analysis_exported={em.get('analysis_exported', False)}, "
            f"result_usage={em.get('result_usage', 'UNKNOWN')}, "
            f"dropped={diagnostics.get('unassigned', {}).get('count', 0)}, "
            f"low_slots={diagnostics.get('slot', {}).get('low_slot_count', 0)}"
        )
        if str(em.get("result_acceptance_status", "")) == "PARTIAL_SCHEDULE_WITH_DROPS":
            print("[APS][结果门槛] 本轮为部分可接受结果，已剔除部分高风险物料")
        print(
            f"[APS][耗时] template_build_seconds={em.get('template_build_seconds', 0.0):.3f}, "
            f"joint_master_seconds={em.get('joint_master_seconds', 0.0):.3f}, "
            f"local_router_seconds={em.get('local_router_seconds', 0.0):.3f}, "
            f"fallback_total_seconds={em.get('fallback_total_seconds', 0.0):.3f}, "
            f"total_run_seconds={em.get('total_run_seconds', 0.0):.3f}"
        )
        if bool(em.get("final_export_performed", False)):
            if str(em.get("result_usage", "")) == "ANALYSIS_ONLY" and bool(em.get("best_candidate_available", False)):
                print(
                    "[APS][候选导出] candidate_schedule_exported=True, "
                    "candidate_line_summary_exported=True, candidate_violation_summary_exported=True"
                )
            export_schedule_results(
                final_df=result.schedule_df,
                rounds_df=result.rounds_df,
                dropped_df=result.dropped_df if isinstance(result.dropped_df, pd.DataFrame) else result.schedule_df.iloc[0:0].copy(),
                output_path=str(result.output_path),
                input_order_count=int(em.get("input_order_count", len(orders_df))),
                rule=result.config.rule,
                engine_used=str(em.get("engine_used", "unknown")),
                fallback_used=bool(em.get("fallback_used", False)),
                fallback_type=str(em.get("fallback_type", "")),
                fallback_reason=str(em.get("fallback_reason", "")),
                equivalence_summary=eq,
                failure_diagnostics=diagnostics if diagnostics else (em.get("failure_diagnostics") if isinstance(em.get("failure_diagnostics"), dict) else summary),
                engine_meta=em,
            )
        return result
