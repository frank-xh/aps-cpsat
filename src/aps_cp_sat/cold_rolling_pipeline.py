from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Dict
import pandas as pd

from aps_cp_sat.decode import decode_candidate_allocation, decode_solution
from aps_cp_sat.domain.models import ColdRollingRequest, ColdRollingResult
from aps_cp_sat.io import export_schedule_results
from aps_cp_sat.model.candidate_graph import build_candidate_graph
from aps_cp_sat.model import solve_master_model
from aps_cp_sat.preprocess import prepare_orders_for_model
from aps_cp_sat.rules import RULE_REGISTRY
from aps_cp_sat.transition import build_transition_templates
from aps_cp_sat.config.parameters import build_profile_config, normalize_enforced_profile_name
from aps_cp_sat.validate import validate_model_equivalence, validate_solution_summary

try:
    from aps_cp_sat.persistence.service import persist_run_analysis_from_excel
except Exception:
    persist_run_analysis_from_excel = None


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
                print(
                    f"[APS][CandidateGraph] edges={int(total.get('candidate_graph_edge_count', 0) or 0)}, "
                    f"direct={int(total.get('candidate_graph_direct_edge_count', 0) or 0)}, "
                    f"real_bridge={int(total.get('candidate_graph_real_bridge_edge_count', 0) or 0)}, "
                    f"virtual_family={int(total.get('candidate_graph_virtual_bridge_family_edge_count', 0) or 0)}, "
                    f"filtered_width={int(total.get('candidate_graph_filtered_by_width_count', 0) or 0)}, "
                    f"filtered_thickness={int(total.get('candidate_graph_filtered_by_thickness_count', 0) or 0)}, "
                    f"filtered_temp={int(total.get('candidate_graph_filtered_by_temp_count', 0) or 0)}, "
                    f"filtered_group={int(total.get('candidate_graph_filtered_by_group_count', 0) or 0)}, "
                    f"filtered_chain={int(total.get('candidate_graph_filtered_by_max_virtual_chain_count', 0) or 0)}"
                )

    @staticmethod
    def _attach_candidate_graph(orders_df: pd.DataFrame, transition_pack: dict, cfg) -> dict:
        if not isinstance(transition_pack, dict):
            return transition_pack
        tpl_df = transition_pack.get("templates")
        candidate_graph = build_candidate_graph(
            orders_df,
            tpl_df if isinstance(tpl_df, pd.DataFrame) else pd.DataFrame(),
            cfg,
        )
        transition_pack["candidate_graph"] = candidate_graph
        transition_pack["candidate_graph_diagnostics"] = dict(candidate_graph.diagnostics)
        build_debug = transition_pack.get("build_debug")
        if isinstance(build_debug, list):
            for item in build_debug:
                if isinstance(item, dict) and str(item.get("line")) == "__all__":
                    item.update(candidate_graph.diagnostics)
                    break
        return transition_pack

    # ---------------------------------------------------------------------------
    # Profile guard: enforce constructive_lns_search as the only allowed path
    # ---------------------------------------------------------------------------

    def _enforce_constructive_lns_profile(self, req: ColdRollingRequest) -> ColdRollingRequest:
        """
        Guard that enforces constructive_lns_search (or debug_acceptance) as the only allowed
        profile/strategy.

        A. Reads requested profile and strategy.
        B. Auto-corrects empty / "default" -> constructive_lns_search (silent enforcement).
        C. Accepts constructive_lns_search and constructive_lns_debug_acceptance explicitly.
        D. Rejects any other profile explicitly.
        """
        requested_profile = str(req.config.model.profile_name or "").strip()

        # ---- Allowed ALNS profiles (current production paths only) ----
        # Profile semantics:
        #   - constructive_lns_search: Production mainline (Route RB / direct_plus_real_bridge)
        #   - constructive_lns_real_bridge_frontload: Alias of constructive_lns_search
        #   - constructive_lns_direct_only_baseline: Regression baseline (Route C / direct_only)
        #   - constructive_lns_virtual_guarded_frontload: Guarded virtual family experiment line
        # NOTE: constructive_lns_bridge_family_master / debug_acceptance 已移出默认主路径，
        # 保留于 compat/experimental_disabled/ 或 debug-only，不进入默认生产守卫。
        _ALLOWED_ALNS_PROFILES = {
            "constructive_lns_search",                      # Production mainline (Route RB)
            "constructive_lns_real_bridge_frontload",       # Alias of mainline (Route RB)
            "constructive_lns_direct_only_baseline",        # Diagnostic baseline (Route C)
            "constructive_lns_virtual_guarded_frontload",   # Guarded virtual family experiment
        }

        # Case B: auto-correct empty/default
        if requested_profile in ("", "default"):
            enforced_cfg = build_profile_config(
                "constructive_lns_search",
                validation_mode=bool(req.config.model.validation_mode),
                production_compatibility_mode=bool(req.config.model.production_compatibility_mode),
            )
            print(
                f"[APS][PROFILE_GUARD] requested_profile={requested_profile or '(empty)'} -> "
                f"enforced_profile=constructive_lns_search"
            )
            return replace(req, config=enforced_cfg)

        # Case C: allowed profiles — return as-is
        if requested_profile in _ALLOWED_ALNS_PROFILES:
            return req

        # Case D: illegal profile -> reject
        raise ValueError(
            f"[APS][PROFILE_GUARD] illegal profile: {requested_profile}; "
            f"Only {sorted(_ALLOWED_ALNS_PROFILES)} are allowed in current engineering mode"
        )

    # =========================================================================
    # UNIFIED_ENGINE_META_FIELDS: unified口径的元数据字段
    # 只保留当前真实主线使用的能力字段；旧 virtual pilot / oracle / family master 等已移除
    # =========================================================================
    _UNIFIED_ENGINE_META_FIELDS = (
        # ---- Configuration ----
        "profile_name",
        "solver_path",
        "constructive_edge_policy",
        "bridge_expansion_mode",
        "allow_virtual_bridge_edge_in_constructive",
        "allow_real_bridge_edge_in_constructive",
        # ---- Single Candidate Graph 追踪 ----
        "candidate_graph_source",
        # ---- Guarded virtual family (受控虚拟族桥接) ----
        "virtual_family_frontload_enabled",
        # Greedy/constructive phase family stats
        "greedy_virtual_family_edge_uses",
        "greedy_virtual_family_budget_blocked_count",
        # ---- Scheduling statistics ----
        "scheduled_real_orders",
        "scheduled_virtual_orders",
        "dropped_count",
        "campaign_count",
        "low_slots",
        "tail_underfilled_count",
        # ---- Edge type statistics ----
        "selected_real_bridge_edge_count",
        "selected_virtual_bridge_edge_count",
        "selected_virtual_bridge_family_edge_count",
        "selected_legacy_virtual_bridge_edge_count",
        "max_bridge_count_used",
        # ---- ALNS performance ----
        "lns_accepted_rounds",
        "lns_drop_delta",
        "reconstruction_no_gain",
        # ---- ALNS guarded virtual family stats ----
        "alns_virtual_family_attempt_count",
        "alns_virtual_family_accept_count",
        "local_cpsat_virtual_family_selected_count",
        # ---- Cutter optimization stats ----
        "cutter_blocks_touched",
        "cutter_blocks_improved",
        "cutter_blocks_skipped_by_precheck",
        "cutter_blocks_skipped_by_no_gain_set",
        "cutter_no_gain_streak_max",
        # ---- Local CP-SAT gate stats ----
        "local_cpsat_attempt_count",
        "local_cpsat_success_count",
        "local_cpsat_skipped_due_to_gate",
        "local_cpsat_total_seconds",
        # ---- Legacy virtual pilot 降级统计 ----
        "virtual_pilot_skipped_due_to_disabled",
        # ---- Acceptance status ----
        "acceptance",
        "acceptance_gate_reason",
        "validation_gate_reason",
    )

    @staticmethod
    def _constructive_edge_policy_from_flags(allow_virtual_bridge: bool, allow_real_bridge: bool) -> str:
        if allow_virtual_bridge:
            return "all_edges_allowed"
        if allow_real_bridge:
            return "direct_plus_real_bridge"
        return "direct_only"

    @staticmethod
    def _count_campaigns(df: pd.DataFrame | None) -> int:
        if df is None or df.empty:
            return 0
        for col in ("campaign_id", "campaign_id_hint", "slot_no", "round_id"):
            if col in df.columns:
                return int(df[col].dropna().nunique())
        return 0

    @staticmethod
    def _count_edge_type(df: pd.DataFrame | None, edge_type: str) -> int:
        if df is None or df.empty or "selected_edge_type" not in df.columns:
            return 0
        return int((df["selected_edge_type"].astype(str) == edge_type).sum())

    @classmethod
    def _ensure_unified_engine_meta(
        cls,
        engine_meta: dict,
        cfg,
        schedule_df: pd.DataFrame | None = None,
        dropped_df: pd.DataFrame | None = None,
        rounds_df: pd.DataFrame | None = None,
    ) -> dict:
        """Normalize top-level run metadata so logs, Excel and APIs share one key set."""
        em = dict(engine_meta or {})
        model_cfg = getattr(cfg, "model", cfg)
        lns_engine_meta = em.get("lns_engine_meta", {}) if isinstance(em.get("lns_engine_meta"), dict) else {}
        lns_diag = em.get("lns_diagnostics", {}) if isinstance(em.get("lns_diagnostics"), dict) else {}
        candidate_graph_diag = em.get("candidate_graph_diagnostics", {}) if isinstance(em.get("candidate_graph_diagnostics"), dict) else {}
        rounds_summary = lns_diag.get("rounds_summary", {}) if isinstance(lns_diag.get("rounds_summary"), dict) else {}
        cut_diags = lns_diag.get("campaign_cut_diags", {}) if isinstance(lns_diag.get("campaign_cut_diags"), dict) else {}

        profile_name = str(em.get("profile_name") or getattr(model_cfg, "profile_name", "") or "unknown")
        solver_path = str(
            em.get("solver_path")
            or em.get("main_path")
            or em.get("engine_used")
            or getattr(model_cfg, "main_solver_strategy", "")
            or "unknown"
        )
        allow_virtual = bool(
            em.get(
                "allow_virtual_bridge_edge_in_constructive",
                em.get(
                    "virtual_bridge_edge_enabled_in_constructive",
                    lns_engine_meta.get(
                        "allow_virtual_bridge_edge_in_constructive",
                        lns_engine_meta.get(
                            "virtual_bridge_edge_enabled_in_constructive",
                            getattr(model_cfg, "allow_virtual_bridge_edge_in_constructive", False),
                        ),
                    ),
                ),
            )
        )
        allow_real = bool(
            em.get(
                "allow_real_bridge_edge_in_constructive",
                em.get(
                    "real_bridge_edge_enabled_in_constructive",
                    lns_engine_meta.get(
                        "allow_real_bridge_edge_in_constructive",
                        lns_engine_meta.get(
                            "real_bridge_edge_enabled_in_constructive",
                            getattr(model_cfg, "allow_real_bridge_edge_in_constructive", False),
                        ),
                    ),
                ),
            )
        )
        bridge_expansion_mode = str(
            em.get("bridge_expansion_mode")
            or lns_engine_meta.get("bridge_expansion_mode")
            or getattr(model_cfg, "bridge_expansion_mode", "disabled")
            or "disabled"
        )
        constructive_edge_policy = str(
            em.get("constructive_edge_policy")
            or lns_engine_meta.get("constructive_edge_policy")
            or cls._constructive_edge_policy_from_flags(allow_virtual, allow_real)
        )
        # Only propagate candidate graph diagnostics that are actually used in production
        for key in (
            "candidate_graph_edge_count",
            "candidate_graph_direct_edge_count",
            "candidate_graph_real_bridge_edge_count",
            "candidate_graph_filtered_by_width_count",
            "candidate_graph_filtered_by_thickness_count",
            "candidate_graph_filtered_by_temp_count",
            "candidate_graph_filtered_by_group_count",
        ):
            if key in candidate_graph_diag:
                em[key] = candidate_graph_diag.get(key)
        # ---- Guarded virtual family (受控虚拟族桥接) diagnostics ----
        build_diags = lns_diag.get("constructive_build_diags", {}) if isinstance(lns_diag, dict) else {}

        if schedule_df is None:
            schedule_df = pd.DataFrame()
        if dropped_df is None:
            dropped_df = pd.DataFrame()
        if rounds_df is None:
            rounds_df = pd.DataFrame()

        if not schedule_df.empty and "is_virtual" in schedule_df.columns:
            scheduled_virtual = int(schedule_df["is_virtual"].fillna(False).astype(bool).sum())
            scheduled_real = int(len(schedule_df) - scheduled_virtual)
        else:
            scheduled_real = int(len(schedule_df)) if schedule_df is not None else 0
            scheduled_virtual = int(
                em.get(
                    "scheduled_virtual_orders",
                    em.get("decodedTemplateVirtualCoilCount", em.get("decodedTemplateVirtualCoilCountTotal", 0)),
                )
                or 0
            )
        dropped_count = (
            int(dropped_df["order_id"].nunique())
            if isinstance(dropped_df, pd.DataFrame) and not dropped_df.empty and "order_id" in dropped_df.columns
            else int(len(dropped_df)) if isinstance(dropped_df, pd.DataFrame) else int(em.get("dropped_count", 0) or 0)
        )
        lns_initial_drop = int(lns_diag.get("initial_dropped_count", em.get("lns_initial_dropped_count", 0)) or 0)
        lns_final_drop = int(lns_diag.get("final_dropped_count", em.get("lns_final_dropped_count", dropped_count)) or dropped_count)

        # ---- Compute ALNS family stats from rounds_df (guarded virtual family) ----
        _round_alns_attempt = 0
        _round_alns_accept = 0
        _round_local_family_selected = 0
        if isinstance(rounds_df, pd.DataFrame) and not rounds_df.empty:
            for col in ("alns_virtual_family_attempt_count", "alns_virtual_family_accept_count"):
                if col in rounds_df.columns:
                    _round_alns_attempt += int(rounds_df[col].sum())
            _round_local_family_selected = int(rounds_df["local_cpsat_virtual_family_selected_count"].sum()) if "local_cpsat_virtual_family_selected_count" in rounds_df.columns else 0

        em.update(
            {
                "profile_name": profile_name,
                "solver_path": solver_path,
                "constructive_edge_policy": constructive_edge_policy,
                "bridge_expansion_mode": bridge_expansion_mode,
                "allow_virtual_bridge_edge_in_constructive": allow_virtual,
                "allow_real_bridge_edge_in_constructive": allow_real,
                # Backward-compatible aliases used by older writer/decoder code.
                "virtual_bridge_edge_enabled_in_constructive": allow_virtual,
                "real_bridge_edge_enabled_in_constructive": allow_real,
                # ---- Single Candidate Graph source tracking ----
                "candidate_graph_source": str(
                    em.get("candidate_graph_source", build_diags.get("candidate_graph_source", "unknown")) or "unknown"
                ),
                # Guarded virtual family (受控虚拟族桥接) - profile-level switch
                "virtual_family_frontload_enabled": bool(
                    getattr(model_cfg, "virtual_family_frontload_enabled", False)
                ),
                # Greedy/constructive phase family stats (from constructive_build_diags)
                "greedy_virtual_family_edge_uses": int(
                    em.get("greedy_virtual_family_edge_uses", build_diags.get("greedy_virtual_family_edge_uses", 0)) or 0
                ),
                "greedy_virtual_family_budget_blocked_count": int(
                    em.get("greedy_virtual_family_budget_blocked_count", build_diags.get("greedy_virtual_family_budget_blocked_count", 0)) or 0
                ),
                "scheduled_real_orders": scheduled_real,
                "scheduled_virtual_orders": scheduled_virtual,
                "dropped_count": dropped_count,
                "campaign_count": int(em.get("campaign_count", cls._count_campaigns(schedule_df)) or cls._count_campaigns(schedule_df)),
                "low_slots": int(em.get("low_slots", em.get("ultra_low_slot_count", 0)) or 0),
                "tail_underfilled_count": int(
                    em.get(
                        "tail_underfilled_count",
                        cut_diags.get(
                            "underfilled_reconstruction_underfilled_after",
                            cut_diags.get("total_underfilled_segments", 0),
                        ),
                    )
                    or 0
                ),
                "selected_real_bridge_edge_count": int(
                    em.get("selected_real_bridge_edge_count", cls._count_edge_type(schedule_df, "REAL_BRIDGE_EDGE")) or 0
                ),
                "selected_virtual_bridge_edge_count": int(
                    em.get("selected_virtual_bridge_edge_count", cls._count_edge_type(schedule_df, "VIRTUAL_BRIDGE_EDGE")) or 0
                ),
                # Guarded virtual family edge counts (受控虚拟族桥接)
                "selected_virtual_bridge_family_edge_count": int(
                    em.get("selected_virtual_bridge_family_edge_count", cls._count_edge_type(schedule_df, "VIRTUAL_BRIDGE_FAMILY_EDGE")) or 0
                ),
                # Legacy virtual bridge edge count (旧式虚拟桥接, 始终为0因legacy被禁用)
                "selected_legacy_virtual_bridge_edge_count": int(0),
                "max_bridge_count_used": int(
                    em.get(
                        "max_bridge_count_used",
                        int(schedule_df["bridge_count"].max()) if not schedule_df.empty and "bridge_count" in schedule_df.columns else 0,
                    )
                    or 0
                ),
                "lns_accepted_rounds": int(
                    em.get("lns_accepted_rounds", rounds_summary.get("accepted_count", 0)) or 0
                ),
                "lns_drop_delta": int(em.get("lns_drop_delta", lns_final_drop - lns_initial_drop) or 0),
                "reconstruction_no_gain": bool(
                    em.get(
                        "reconstruction_no_gain",
                        int(em.get("underfilled_reconstruction_valid_delta", cut_diags.get("underfilled_reconstruction_valid_delta", 0)) or 0) == 0
                        and int(em.get("underfilled_reconstruction_underfilled_delta", cut_diags.get("underfilled_reconstruction_underfilled_delta", 0)) or 0) == 0,
                    )
                ),
                # ALNS guarded virtual family stats (受控虚拟族桥接)
                "alns_virtual_family_attempt_count": _round_alns_attempt,
                "alns_virtual_family_accept_count": _round_alns_accept,
                "local_cpsat_virtual_family_selected_count": _round_local_family_selected,
                # ---- Cutter optimization stats ----
                "cutter_blocks_touched": int(
                    em.get("cutter_blocks_touched", cut_diags.get("cutter_blocks_touched", 0)) or 0
                ),
                "cutter_blocks_improved": int(
                    em.get("cutter_blocks_improved", cut_diags.get("cutter_blocks_improved", 0)) or 0
                ),
                "cutter_blocks_skipped_by_precheck": int(
                    em.get("cutter_blocks_skipped_by_precheck", cut_diags.get("cutter_blocks_skipped_by_precheck", 0)) or 0
                ),
                "cutter_blocks_skipped_by_no_gain_set": int(
                    em.get("cutter_blocks_skipped_by_no_gain_set", cut_diags.get("cutter_blocks_skipped_by_no_gain_set", 0)) or 0
                ),
                "cutter_no_gain_streak_max": int(
                    em.get("cutter_no_gain_streak_max", cut_diags.get("cutter_no_gain_streak_max", 0)) or 0
                ),
                # ---- Local CP-SAT gate stats ----
                "local_cpsat_attempt_count": int(
                    em.get("local_cpsat_attempt_count", lns_diag.get("local_cpsat_attempt_count", 0)) or 0
                ),
                "local_cpsat_success_count": int(
                    em.get("local_cpsat_success_count", lns_diag.get("local_cpsat_success_count", 0)) or 0
                ),
                "local_cpsat_skipped_due_to_gate": int(
                    em.get("local_cpsat_skipped_due_to_gate", lns_diag.get("local_cpsat_skipped_due_to_gate", 0)) or 0
                ),
                "local_cpsat_total_seconds": float(
                    em.get("local_cpsat_total_seconds", lns_diag.get("local_cpsat_total_seconds", 0.0)) or 0.0
                ),
                # ---- Legacy virtual pilot 降级统计 ----
                # virtual_pilot 相关字段已从默认主线移除；此处只保留一个总字段
                "virtual_pilot_skipped_due_to_disabled": bool(
                    getattr(model_cfg, "repair_only_virtual_bridge_pilot_enabled", False) is False
                ),
                "acceptance": str(em.get("acceptance", em.get("result_acceptance_status", "unknown")) or "unknown"),
                "acceptance_gate_reason": str(em.get("acceptance_gate_reason", "unknown") or "unknown"),
                "validation_gate_reason": str(em.get("validation_gate_reason", "unknown") or "unknown"),
            }
        )
        for key in cls._UNIFIED_ENGINE_META_FIELDS:
            em.setdefault(key, 0 if key.endswith("_count") or key.endswith("_orders") else "unknown")
        return em

    @classmethod
    def _print_config_snapshot(cls, cfg) -> None:
        model_cfg = getattr(cfg, "model", cfg)
        allow_virtual = bool(getattr(model_cfg, "allow_virtual_bridge_edge_in_constructive", False))
        allow_real = bool(getattr(model_cfg, "allow_real_bridge_edge_in_constructive", False))
        print(
            "[APS][CONFIG_SNAPSHOT] "
            f"profile_name={getattr(model_cfg, 'profile_name', 'unknown')}, "
            f"solver_path={getattr(model_cfg, 'main_solver_strategy', 'unknown')}, "
            f"constructive_edge_policy={cls._constructive_edge_policy_from_flags(allow_virtual, allow_real)}, "
            f"allow_virtual_bridge_edge_in_constructive={allow_virtual}, "
            f"allow_real_bridge_edge_in_constructive={allow_real}, "
            f"bridge_expansion_mode={getattr(model_cfg, 'bridge_expansion_mode', 'disabled')}, "
            f"virtual_family_frontload_enabled={bool(getattr(model_cfg, 'virtual_family_frontload_enabled', False))}, "
            f"candidate_graph_source=from_pipeline"
        )

    @classmethod
    def _print_result_snapshot(cls, engine_meta: dict) -> None:
        em = engine_meta or {}
        fields = ", ".join(f"{key}={em.get(key, 'unknown')}" for key in cls._UNIFIED_ENGINE_META_FIELDS)
        print(f"[APS][RESULT_SNAPSHOT] {fields}")

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
        """
        Annotate dropped orders with structured drop reasons.

        Drop priority order (from highest to lowest priority for dropping):
        1. TON_WINDOW_INFEASIBLE: would break legal 700-2000 ton campaign window
        2. GLOBAL_ISOLATED_ORDER: no feasible line or transition path
        3. NO_FEASIBLE_LINE: line capability mismatch
        4. BRIDGE_REQUIRED_BUT_NOT_SUPPORTED: requires bridge that exceeds limits
        5. SLOT_ROUTING_RISK_TOO_HIGH: local router cannot find valid sequence
        6. LOW_PRIORITY_DROP: low priority, loose due date
        7. CAPACITY_PRESSURE: forced drop due to capacity constraints
        """
        if dropped_df is None or dropped_df.empty:
            return dropped_df if isinstance(dropped_df, pd.DataFrame) else pd.DataFrame()
        out = dropped_df.copy()
        evidence = engine_meta.get("feasibility_evidence", {}) if isinstance(engine_meta, dict) else {}
        isolated_top = {
            str(item.get("order_id", "")): item
            for item in (evidence.get("isolated_orders_topn", []) if isinstance(evidence, dict) else [])
        }

        # Drop priority: TON_WINDOW > GLOBAL_ISOLATED > NO_FEASIBLE_LINE > BRIDGE_REQUIRED > ROUTING_RISK > LOW_PRIORITY > CAPACITY
        DROP_PRIORITY = {
            "TON_WINDOW_INFEASIBLE": 1,
            "GLOBAL_ISOLATED_ORDER": 2,
            "NO_FEASIBLE_LINE": 3,
            "BRIDGE_REQUIRED_BUT_NOT_SUPPORTED": 4,
            "SLOT_ROUTING_RISK_TOO_HIGH": 5,
            "LOW_PRIORITY_DROP": 6,
            "CAPACITY_PRESSURE": 7,
            "MASTER_UNASSIGNED": 8,
            "LOCAL_ROUTER_KICKOUT": 9,
            "FALLBACK_SCALE_UNSCHEDULED": 10,
            "DECODE_ORDER_MISMATCH": 11,  # Demoted during decode due to seq/master_seq mismatch
        }

        candidate_lines = []
        dominant_reasons = []
        secondary_reasons = []
        risk_summaries = []
        globally_isolated = []
        would_break_slot = []
        would_break_ton_window = []
        drop_stages = []

        for _, row in out.iterrows():
            oid = str(row.get("order_id", ""))
            allowed = ColdRollingPipeline._allowed_lines(str(row.get("line_capability", "dual")))
            candidate_lines.append(",".join(allowed))
            iso = oid in isolated_top
            globally_isolated.append(bool(iso))
            reasons = []
            explicit_reason = str(row.get("drop_reason", "") or "").strip()

            # Classify drop reasons into priority tiers
            if explicit_reason:
                reasons.append(explicit_reason)

            # Tier 1: Global isolation
            if iso:
                reasons.append("GLOBAL_ISOLATED_ORDER")
            if not allowed:
                reasons.append("NO_FEASIBLE_LINE")

            # Tier 2: Ton window (check order tons vs campaign limits)
            order_tons = float(row.get("tons", 0) or 0)
            # If order is too large (> 2000) or too small but would force extra slot
            if order_tons > 2000:
                reasons.append("TON_WINDOW_INFEASIBLE")

            # Tier 3: Bridge required but not supported
            if explicit_reason == "NO_FEASIBLE_TRANSITION":
                reasons.append("BRIDGE_REQUIRED_BUT_NOT_SUPPORTED")

            # Tier 4: Routing risk
            if engine_meta.get("unroutable_slot_count", 0) > 0:
                reasons.append("SLOT_ROUTING_RISK_TOO_HIGH")

            # Tier 5: Capacity pressure
            if explicit_reason == "FALLBACK_SCALE_UNSCHEDULED":
                reasons.append("CAPACITY_PRESSURE")

            # Tier 6: Low priority (only if priority/due_rank fields actually exist in input data)
            # FIX: Only apply low_priority check when the fields are explicitly present
            # This prevents test mock data (without priority/due_rank) from incorrectly triggering LOW_PRIORITY_DROP
            if "priority" in row.index or "due_rank" in row.index:
                due_rank = int(row.get("due_rank", 3) or 3)
                priority = int(row.get("priority", 0) or 0)
                if priority <= 0 or due_rank >= 2:
                    reasons.append("LOW_PRIORITY_DROP")

            # Determine dominant reason based on priority
            existing_dominant = str(row.get("dominant_drop_reason", "") or "").strip()
            if existing_dominant:
                dominant = existing_dominant
            else:
                # Pick the highest priority (lowest number) reason
                reason_priority = 999
                dominant = "OTHER"
                for r in reasons:
                    p = DROP_PRIORITY.get(r, 99)
                    if p < reason_priority:
                        reason_priority = p
                        dominant = r
                if dominant == "OTHER" and reasons:
                    dominant = reasons[0]

            # Secondary reasons: all other reasons except dominant
            existing_secondary = [s for s in str(row.get("secondary_reasons", "") or "").split(",") if s]
            merged_secondary = list(dict.fromkeys(existing_secondary + [r for r in reasons if r != dominant]))

            existing_risk = str(row.get("risk_summary", "") or "").strip()

            # Determine drop stage
            if "MASTER_UNASSIGNED" in dominant:
                drop_stage = "MASTER_ASSIGNMENT"
            elif "LOCAL_ROUTER" in dominant:
                drop_stage = "LOCAL_ROUTING"
            elif "TON_WINDOW" in dominant:
                drop_stage = "TON_WINDOW_FORMATION"
            elif "GLOBAL_ISOLATED" in dominant:
                drop_stage = "GLOBAL_ISOLATION_CHECK"
            else:
                drop_stage = "FALLBACK_RECOVERY"

            dominant_reasons.append(dominant)
            secondary_reasons.append(",".join(merged_secondary))
            risk_summaries.append(existing_risk or ("|".join(reasons) if reasons else "OTHER"))

            # Determine would_break flags
            existing_break = bool(row.get("would_break_slot_if_kept", False))
            would_break_slot.append(bool(existing_break or "SLOT_ROUTING_RISK_TOO_HIGH" in reasons or iso))
            would_break_ton_window.append("TON_WINDOW_INFEASIBLE" in dominant or order_tons > 2000)
            drop_stages.append(drop_stage)

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
        out["would_break_ton_window_if_kept"] = would_break_ton_window
        out["drop_stage"] = drop_stages
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
        dropped_unique_count = int(dropped["order_id"].nunique()) if not dropped.empty and "order_id" in dropped.columns else int(len(dropped))
        dropped_row_count = int(len(dropped))
        duplicated_dropped_rows_count = max(0, dropped_row_count - dropped_unique_count)
        unassigned_section: dict = {
            "count": int(dropped_unique_count),
            "count_unique": int(dropped_unique_count),
            "count_rows": int(dropped_row_count),
            "duplicated_dropped_rows_count": int(duplicated_dropped_rows_count),
            "tons": round(float(dropped["tons"].sum()), 1) if (not dropped.empty and "tons" in dropped.columns) else 0.0,
        }

        # Extended drop reason statistics
        if not dropped.empty and "dominant_drop_reason" in dropped.columns:
            vc = dropped["dominant_drop_reason"].fillna("UNKNOWN").value_counts()
            topn = vc.head(5)
            for idx, (reason, cnt) in enumerate(topn.items(), start=1):
                unassigned_section[f"top{idx}_reason"] = str(reason)
                unassigned_section[f"top{idx}_count"] = int(cnt)
            if not vc.empty:
                unassigned_section["top_reason"] = str(vc.index[0])
            unassigned_section["dropped_reason_histogram"] = vc.to_dict()

            # Count drops by reason category
            unassigned_section["drop_due_to_ton_window_count"] = int(
                dropped["dominant_drop_reason"].fillna("").str.contains("TON_WINDOW").sum()
            )
            unassigned_section["drop_due_to_global_isolation_count"] = int(
                dropped["dominant_drop_reason"].fillna("").str.contains("GLOBAL_ISOLATED").sum()
            )
            unassigned_section["drop_due_to_no_feasible_line_count"] = int(
                dropped["dominant_drop_reason"].fillna("").str.contains("NO_FEASIBLE_LINE").sum()
            )
            unassigned_section["drop_due_to_routing_risk_count"] = int(
                dropped["dominant_drop_reason"].fillna("").str.contains("ROUTING_RISK|LOCAL_ROUTER").sum()
            )
            unassigned_section["drop_due_to_capacity_count"] = int(
                dropped["dominant_drop_reason"].fillna("").str.contains("CAPACITY_PRESSURE").sum()
            )
            unassigned_section["drop_due_to_low_priority_count"] = int(
                dropped["dominant_drop_reason"].fillna("").str.contains("LOW_PRIORITY").sum()
            )
            unassigned_section["drop_due_to_master_unassigned_count"] = int(
                dropped["dominant_drop_reason"].fillna("").str.contains("MASTER_UNASSIGNED").sum()
            )

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
            "export_block_stage": str(em.get("export_block_stage", "none")),  # routing / partial_acceptance / none
            "partial_acceptance_block_reason": str(em.get("partial_acceptance_block_reason", "")),
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
                # Priority 3: Extract adjacency/validation repair drop breakdown from trace
                if isinstance(item, dict):
                    if "adjacency_drop_count" in item:
                        diagnostics["fallback"]["structure_fallback_adjacency_drops"] = int(item.get("adjacency_drop_count", 0))
                    if "ton_window_drop_count" in item:
                        diagnostics["fallback"]["structure_fallback_ton_window_drops"] = int(item.get("ton_window_drop_count", 0))
                    if "global_iso_drop_count" in item:
                        diagnostics["fallback"]["structure_fallback_global_iso_drops"] = int(item.get("global_iso_drop_count", 0))
        # Priority 1/3: Hard violation breakdown from eq validation
        diagnostics["fallback"]["adjacency_violation_cnt"] = int(em.get("adjacency_violation_cnt", 0))
        diagnostics["fallback"]["bridge_expand_violation_cnt"] = int(em.get("bridge_expand_violation_cnt", 0))
        diagnostics["fallback"]["chain_break_cnt"] = int(em.get("chain_break_cnt", 0))
        diagnostics["fallback"]["template_miss_cnt"] = int(em.get("template_miss_cnt", 0))

        diagnostics["underfilled_reconstruction"] = {
            "underfilled_reconstruction_enabled": bool(em.get("underfilled_reconstruction_enabled", True)),
            "underfilled_reconstruction_attempts": int(em.get("underfilled_reconstruction_attempts", 0) or 0),
            "underfilled_reconstruction_success": int(em.get("underfilled_reconstruction_success", 0) or 0),
            "underfilled_reconstruction_blocks_tested": int(em.get("underfilled_reconstruction_blocks_tested", 0) or 0),
            "underfilled_reconstruction_blocks_skipped": int(em.get("underfilled_reconstruction_blocks_skipped", 0) or 0),
            "underfilled_reconstruction_valid_before": int(em.get("underfilled_reconstruction_valid_before", 0) or 0),
            "underfilled_reconstruction_valid_after": int(em.get("underfilled_reconstruction_valid_after", 0) or 0),
            "underfilled_reconstruction_underfilled_before": int(em.get("underfilled_reconstruction_underfilled_before", 0) or 0),
            "underfilled_reconstruction_underfilled_after": int(em.get("underfilled_reconstruction_underfilled_after", 0) or 0),
            "underfilled_reconstruction_valid_delta": int(em.get("underfilled_reconstruction_valid_delta", 0) or 0),
            "underfilled_reconstruction_underfilled_delta": int(em.get("underfilled_reconstruction_underfilled_delta", 0) or 0),
            "underfilled_reconstruction_segments_salvaged": int(em.get("underfilled_reconstruction_segments_salvaged", 0) or 0),
            "underfilled_reconstruction_orders_salvaged": int(em.get("underfilled_reconstruction_orders_salvaged", 0) or 0),
            "underfilled_reconstruction_not_entered_reason": str(em.get("underfilled_reconstruction_not_entered_reason", "")),
            "underfilled_reconstruction_seconds": float(em.get("underfilled_reconstruction_seconds", 0.0) or 0.0),
        }
        diagnostics["repair_only_real_bridge"] = {
            "repair_only_real_bridge_enabled": bool(em.get("repair_only_real_bridge_enabled", False)),
            "repair_only_real_bridge_attempts": int(em.get("repair_only_real_bridge_attempts", 0) or 0),
            "repair_only_real_bridge_success": int(em.get("repair_only_real_bridge_success", 0) or 0),
            "repair_only_real_bridge_candidates_total": int(em.get("repair_only_real_bridge_candidates_total", 0) or 0),
            "repair_only_real_bridge_candidates_kept": int(em.get("repair_only_real_bridge_candidates_kept", 0) or 0),
            "repair_only_real_bridge_filtered_direct_feasible": int(em.get("repair_only_real_bridge_filtered_direct_feasible", 0) or 0),
            "repair_only_real_bridge_filtered_pair_invalid": int(em.get("repair_only_real_bridge_filtered_pair_invalid", 0) or 0),
            "repair_only_real_bridge_filtered_ton_invalid": int(em.get("repair_only_real_bridge_filtered_ton_invalid", 0) or 0),
            "repair_only_real_bridge_filtered_score_worse": int(em.get("repair_only_real_bridge_filtered_score_worse", 0) or 0),
            "repair_only_real_bridge_filtered_bridge_limit_exceeded": int(em.get("repair_only_real_bridge_filtered_bridge_limit_exceeded", 0) or 0),
            "repair_only_real_bridge_filtered_multiplicity_invalid": int(em.get("repair_only_real_bridge_filtered_multiplicity_invalid", 0) or 0),
            "repair_only_real_bridge_filtered_bridge_path_not_real": int(em.get("repair_only_real_bridge_filtered_bridge_path_not_real", 0) or 0),
            "repair_only_real_bridge_filtered_bridge_path_missing": int(em.get("repair_only_real_bridge_filtered_bridge_path_missing", 0) or 0),
            "repair_only_real_bridge_filtered_block_order_mismatch": int(em.get("repair_only_real_bridge_filtered_block_order_mismatch", 0) or 0),
            "repair_only_real_bridge_filtered_line_mismatch": int(em.get("repair_only_real_bridge_filtered_line_mismatch", 0) or 0),
            "repair_only_real_bridge_filtered_block_membership_mismatch": int(em.get("repair_only_real_bridge_filtered_block_membership_mismatch", 0) or 0),
            "repair_only_real_bridge_filtered_bridge_path_payload_empty": int(em.get("repair_only_real_bridge_filtered_bridge_path_payload_empty", 0) or 0),
            "repair_bridge_pack_has_real_rows": bool(em.get("repair_bridge_pack_has_real_rows", False)),
            "repair_bridge_pack_type": str(em.get("repair_bridge_pack_type", "")),
            "repair_bridge_pack_keys": em.get("repair_bridge_pack_keys", []),
            "repair_bridge_pack_line_keys": em.get("repair_bridge_pack_line_keys", []),
            "repair_bridge_pack_real_rows_total": int(em.get("repair_bridge_pack_real_rows_total", 0) or 0),
            "repair_bridge_pack_virtual_rows_total": int(em.get("repair_bridge_pack_virtual_rows_total", 0) or 0),
            "repair_bridge_raw_rows_total": int(em.get("repair_bridge_raw_rows_total", 0) or 0),
            "repair_bridge_matched_rows_total": int(em.get("repair_bridge_matched_rows_total", 0) or 0),
            "repair_bridge_kept_rows_total": int(em.get("repair_bridge_kept_rows_total", 0) or 0),
            "repair_bridge_endpoint_key_mismatch_count": int(em.get("repair_bridge_endpoint_key_mismatch_count", 0) or 0),
            "repair_bridge_field_name_mismatch_count": int(em.get("repair_bridge_field_name_mismatch_count", 0) or 0),
            "repair_bridge_inconsistency_count": int(em.get("repair_bridge_inconsistency_count", 0) or 0),
            "repair_bridge_boundary_band_enabled": bool(em.get("repair_bridge_boundary_band_enabled", True)),
            "repair_bridge_band_pairs_tested": int(em.get("repair_bridge_band_pairs_tested", 0) or 0),
            "repair_bridge_band_hits": int(em.get("repair_bridge_band_hits", 0) or 0),
            "repair_bridge_single_point_hits": int(em.get("repair_bridge_single_point_hits", 0) or 0),
            "repair_bridge_band_only_hits": int(em.get("repair_bridge_band_only_hits", 0) or 0),
            "repair_bridge_band_best_distance": int(em.get("repair_bridge_band_best_distance", -1) or -1),
            "repair_bridge_endpoint_adjustment_enabled": bool(em.get("repair_bridge_endpoint_adjustment_enabled", True)),
            "repair_bridge_adjustments_generated": int(em.get("repair_bridge_adjustments_generated", 0) or 0),
            "repair_bridge_adjustment_pairs_tested": int(em.get("repair_bridge_adjustment_pairs_tested", 0) or 0),
            "repair_bridge_adjustment_hits": int(em.get("repair_bridge_adjustment_hits", 0) or 0),
            "repair_bridge_adjustment_only_hits": int(em.get("repair_bridge_adjustment_only_hits", 0) or 0),
            "repair_bridge_best_adjustment_cost": int(em.get("repair_bridge_best_adjustment_cost", -1) or -1),
            "repair_bridge_candidates_matched": int(em.get("repair_bridge_candidates_matched", 0) or 0),
            "repair_bridge_candidates_rejected_pair_invalid": int(em.get("repair_bridge_candidates_rejected_pair_invalid", 0) or 0),
            "repair_bridge_candidates_rejected_ton_invalid": int(em.get("repair_bridge_candidates_rejected_ton_invalid", 0) or 0),
            "repair_bridge_candidates_rejected_score_worse": int(em.get("repair_bridge_candidates_rejected_score_worse", 0) or 0),
            "repair_bridge_candidates_accepted": int(em.get("repair_bridge_candidates_accepted", 0) or 0),
            "repair_bridge_exact_invalid_pair_count": int(em.get("repair_bridge_exact_invalid_pair_count", 0) or 0),
            "repair_bridge_frontier_mismatch_count": int(em.get("repair_bridge_frontier_mismatch_count", 0) or 0),
            "repair_bridge_pair_invalid_width": int(em.get("repair_bridge_pair_invalid_width", 0) or 0),
            "repair_bridge_pair_invalid_thickness": int(em.get("repair_bridge_pair_invalid_thickness", 0) or 0),
            "repair_bridge_pair_invalid_temp": int(em.get("repair_bridge_pair_invalid_temp", 0) or 0),
            "repair_bridge_pair_invalid_group": int(em.get("repair_bridge_pair_invalid_group", 0) or 0),
            "repair_bridge_pair_invalid_unknown": int(em.get("repair_bridge_pair_invalid_unknown", 0) or 0),
            "repair_bridge_ton_rescue_attempts": int(em.get("repair_bridge_ton_rescue_attempts", 0) or 0),
            "repair_bridge_ton_rescue_success": int(em.get("repair_bridge_ton_rescue_success", 0) or 0),
            "repair_bridge_ton_rescue_windows_tested": int(em.get("repair_bridge_ton_rescue_windows_tested", 0) or 0),
            "repair_bridge_ton_rescue_valid_delta": int(em.get("repair_bridge_ton_rescue_valid_delta", 0) or 0),
            "repair_bridge_ton_rescue_underfilled_delta": int(em.get("repair_bridge_ton_rescue_underfilled_delta", 0) or 0),
            "repair_bridge_ton_rescue_scheduled_orders_delta": int(em.get("repair_bridge_ton_rescue_scheduled_orders_delta", 0) or 0),
            "repair_bridge_filtered_ton_below_min_current_block": int(em.get("repair_bridge_filtered_ton_below_min_current_block", 0) or 0),
            "repair_bridge_filtered_ton_below_min_even_after_neighbor_expansion": int(em.get("repair_bridge_filtered_ton_below_min_even_after_neighbor_expansion", 0) or 0),
            "repair_bridge_filtered_ton_above_max_after_expansion": int(em.get("repair_bridge_filtered_ton_above_max_after_expansion", 0) or 0),
            "repair_bridge_filtered_ton_split_not_found": int(em.get("repair_bridge_filtered_ton_split_not_found", 0) or 0),
            "repair_bridge_filtered_ton_rescue_no_gain": int(em.get("repair_bridge_filtered_ton_rescue_no_gain", 0) or 0),
            "repair_bridge_filtered_ton_rescue_impossible": int(em.get("repair_bridge_filtered_ton_rescue_impossible", 0) or 0),
            "bridgeability_route_suggestion": str(em.get("bridgeability_route_suggestion", "")),
            "bridgeability_census": em.get("bridgeability_census", {}),
            "bridgeability_census_items": em.get("bridgeability_census_items", []),
            # ---- Legacy virtual pilot: 已降级为单一总字段 ----
            # 旧有 ~45 个 virtual_pilot 细项字段已从此处移除；只保留总字段
            "conservative_apply_attempt_count": int(em.get("conservative_apply_attempt_count", 0) or 0),
            "conservative_apply_success_count": int(em.get("conservative_apply_success_count", 0) or 0),
            "conservative_apply_reject_count": int(em.get("conservative_apply_reject_count", 0) or 0),
            "repair_only_real_bridge_used_segments": int(em.get("repair_only_real_bridge_used_segments", 0) or 0),
            "repair_only_real_bridge_used_orders": int(em.get("repair_only_real_bridge_used_orders", 0) or 0),
            "repair_only_real_bridge_not_entered_reason": str(em.get("repair_only_real_bridge_not_entered_reason", "")),
            "repair_only_real_bridge_seconds": float(em.get("repair_only_real_bridge_seconds", 0.0) or 0.0),
        }

        diagnostics["rules"] = {
            spec.key.value: spec.zh_name
            for spec in RULE_REGISTRY.export_visible_specs()
        }
        diagnostics["rule_semantics"] = {
            "line_compatibility": "HARD",
            "direct_transition_feasibility": "HARD",
            "campaign_ton_max": "HARD",
            "campaign_ton_min": "HARD",  # Changed from STRONG_SOFT to HARD
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
    def _evaluate_partial_acceptance(result: ColdRollingResult, orders_df: pd.DataFrame, validation_summary: dict | None = None) -> dict:
        """
        Evaluate if PARTIAL_SCHEDULE_WITH_DROPS can be accepted.

        HARD RULE: Only accept partial results when ALL hard violations are ZERO.
        This includes:
        - campaign_ton_min_violation_count = 0 (slot tons >= 700)
        - campaign_ton_max_violation_count = 0 (slot tons <= 2000)
        - campaign_ton_hard_violation_count_total = 0
        - direct_reverse_step_violation_count = 0
        - virtual_attach_reverse_violation_count = 0
        - period_reverse_count_violation_count = 0
        - invalid_virtual_spec_count = 0
        - bridge_count_violation_count = 0
        - hard_violation_count_total = 0
        """
        dropped_df = result.dropped_df if isinstance(result.dropped_df, pd.DataFrame) else pd.DataFrame()
        schedule_df = result.schedule_df if isinstance(result.schedule_df, pd.DataFrame) else pd.DataFrame()
        # 按 unique order_id 统计 dropped/scheduled/total（dropped_df 中可能存在重复订单行）
        total_orders = int(orders_df["order_id"].nunique()) if isinstance(orders_df, pd.DataFrame) and "order_id" in orders_df.columns else int(len(orders_df))
        total_tons = float(orders_df["tons"].sum()) if isinstance(orders_df, pd.DataFrame) and "tons" in orders_df.columns else 0.0
        dropped_order_count = int(dropped_df["order_id"].nunique()) if not dropped_df.empty and "order_id" in dropped_df.columns else int(len(dropped_df))
        dropped_row_count = int(len(dropped_df))
        dropped_tons = float(dropped_df["tons"].sum()) if "tons" in dropped_df.columns and not dropped_df.empty else 0.0
        real_sched = schedule_df[~schedule_df["is_virtual"]] if isinstance(schedule_df, pd.DataFrame) and "is_virtual" in schedule_df.columns else schedule_df
        scheduled_orders = int(real_sched["order_id"].nunique()) if isinstance(real_sched, pd.DataFrame) and not real_sched.empty and "order_id" in real_sched.columns else int(len(real_sched))
        scheduled_tons = float(real_sched["tons"].sum()) if isinstance(real_sched, pd.DataFrame) and "tons" in real_sched.columns else 0.0
        drop_ratio = float(dropped_order_count / max(1, total_orders))
        drop_tons_ratio = float(dropped_tons / max(1e-9, total_tons)) if total_tons > 0 else 0.0
        cfg = result.config.model

        # Check hard violations from validation summary
        vs = validation_summary or {}
        hard_violation_count_total = int(vs.get("hard_violation_count_total", 0))
        campaign_ton_hard_violation_count_total = int(vs.get("campaign_ton_hard_violation_count_total", 0))
        campaign_ton_min_violation_count = int(vs.get("campaign_ton_min_violation_count", 0))
        campaign_ton_max_violation_count = int(vs.get("campaign_ton_max_violation_count", 0))
        direct_reverse_step_violation_count = int(vs.get("direct_reverse_step_violation_count", 0))
        virtual_attach_reverse_violation_count = int(vs.get("virtual_attach_reverse_violation_count", 0))
        period_reverse_count_violation_count = int(vs.get("period_reverse_count_violation_count", 0))
        invalid_virtual_spec_count = int(vs.get("invalid_virtual_spec_count", 0))
        bridge_count_violation_count = int(vs.get("bridge_count_violation_count", 0))

        # HARD RULE: all hard violations must be zero
        hard_violations_zero = (
            hard_violation_count_total == 0
            and campaign_ton_hard_violation_count_total == 0
            and campaign_ton_min_violation_count == 0
            and campaign_ton_max_violation_count == 0
            and direct_reverse_step_violation_count == 0
            and virtual_attach_reverse_violation_count == 0
            and period_reverse_count_violation_count == 0
            and invalid_virtual_spec_count == 0
            and bridge_count_violation_count == 0
        )

        # Soft threshold checks
        soft_threshold_passed = (
            drop_ratio <= float(cfg.max_drop_ratio_for_partial)
            and drop_tons_ratio <= float(cfg.max_drop_tons_ratio_for_partial)
            and scheduled_orders >= int(cfg.min_scheduled_orders_for_partial)
            and scheduled_tons >= float(cfg.min_scheduled_tons_for_partial)
        )

        # Combined: must have dropped orders, pass soft thresholds, AND have zero hard violations
        passed = (
            dropped_order_count > 0
            and soft_threshold_passed
            and hard_violations_zero
        )

        # Build block reason if not passed
        block_reason = ""
        if dropped_order_count == 0:
            block_reason = "NO_DROPPED_ORDERS"
        elif not soft_threshold_passed:
            block_reason = "SOFT_THRESHOLD_NOT_MET"
        elif not hard_violations_zero:
            if hard_violation_count_total > 0:
                block_reason = f"HARD_VIOLATIONS_EXIST:{hard_violation_count_total}"
            elif campaign_ton_hard_violation_count_total > 0:
                block_reason = f"CAMPAIGN_TON_VIOLATION:{campaign_ton_hard_violation_count_total}"
            elif direct_reverse_step_violation_count > 0:
                block_reason = f"DIRECT_REVERSE_VIOLATION:{direct_reverse_step_violation_count}"
            elif virtual_attach_reverse_violation_count > 0:
                block_reason = f"VIRTUAL_ATTACH_REVERSE_VIOLATION:{virtual_attach_reverse_violation_count}"
            elif period_reverse_count_violation_count > 0:
                block_reason = f"PERIOD_REVERSE_VIOLATION:{period_reverse_count_violation_count}"
            elif invalid_virtual_spec_count > 0:
                block_reason = f"INVALID_VIRTUAL_SPEC:{invalid_virtual_spec_count}"
            else:
                block_reason = "HARD_VIOLATIONS_UNKNOWN"

        return {
            "partial_result_available": bool(dropped_order_count > 0),
            "partial_acceptance_passed": bool(passed),
            "partial_drop_ratio": drop_ratio,
            "partial_drop_tons_ratio": drop_tons_ratio,
            "scheduled_orders": scheduled_orders,
            "scheduled_tons": scheduled_tons,
            "hard_violations_zero": bool(hard_violations_zero),
            "hard_violation_count_total": int(hard_violation_count_total),
            "campaign_ton_hard_violation_count_total": int(campaign_ton_hard_violation_count_total),
            "campaign_ton_min_violation_count": int(campaign_ton_min_violation_count),
            "campaign_ton_max_violation_count": int(campaign_ton_max_violation_count),
            "direct_reverse_step_violation_count": int(direct_reverse_step_violation_count),
            "virtual_attach_reverse_violation_count": int(virtual_attach_reverse_violation_count),
            "period_reverse_count_violation_count": int(period_reverse_count_violation_count),
            "invalid_virtual_spec_count": int(invalid_virtual_spec_count),
            "bridge_count_violation_count": int(bridge_count_violation_count),
            "partial_acceptance_block_reason": str(block_reason),
        }

    def run(self, req: ColdRollingRequest) -> ColdRollingResult:
            run_t0 = perf_counter()
            print(f"[APS][RUN_PATH_FINGERPRINT] PIPELINE_V2_20260416A")
            # ---- Profile guard: enforce constructive_lns_search ----
            req = self._enforce_constructive_lns_profile(req)
            print(f"[APS][PROFILE_GUARD] effective_profile={req.config.model.profile_name}")
            print(f"[APS][PROFILE_GUARD] effective_main_solver_strategy={req.config.model.main_solver_strategy}")
            print(f"[APS][PROFILE_GUARD] joint_master_disabled=true")
            print(f"[APS][constructive_lns] preparing transition templates for constructive graph search")
            print(f"[APS][Profile] name={req.config.model.profile_name}")
            if req.config.model.profile_name == "constructive_lns_debug_acceptance":
                print(
                    "[APS][debug_acceptance] using relaxed partial acceptance thresholds for validation only"
                )
            # Route B: Log bridge expansion mode to prevent future misdiagnosis
            bridge_expansion_mode = str(getattr(req.config.model, "bridge_expansion_mode", "disabled"))
            allow_virtual_bridge = bool(getattr(req.config.model, "allow_virtual_bridge_edge_in_constructive", False))
            allow_real_bridge = bool(getattr(req.config.model, "allow_real_bridge_edge_in_constructive", False))
            print(f"[APS][constructive_lns] bridge_expansion_mode={bridge_expansion_mode}")
            print(f"[APS][constructive_lns] allow_virtual_bridge_edge_in_constructive={allow_virtual_bridge}")
            print(f"[APS][constructive_lns] allow_real_bridge_edge_in_constructive={allow_real_bridge}")
            constructive_edge_policy = "direct_only" if (not allow_virtual_bridge and not allow_real_bridge) else ("direct_plus_real_bridge" if not allow_virtual_bridge else "all_edges_allowed")
            print(f"[APS][constructive_lns] constructive_edge_policy={constructive_edge_policy}")
            self._print_config_snapshot(req.config)
            if constructive_edge_policy == "direct_only":
                print(f"[APS][constructive_lns] 路线C(direct_only): 只允许 DIRECT_EDGE, 禁用所有桥接边, 快速验证桥接展开是否为 official_exported 唯一障碍")
            elif constructive_edge_policy == "direct_plus_real_bridge":
                print(f"[APS][constructive_lns] 路线RB(direct_plus_real_bridge): 允许 DIRECT_EDGE + REAL_BRIDGE_EDGE, 禁用 VIRTUAL_BRIDGE_EDGE, bridge_expansion_mode={bridge_expansion_mode}")
            else:
                print(f"[APS][constructive_lns] edge_policy={constructive_edge_policy}, bridge_expansion_mode={bridge_expansion_mode}")
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
            transition_pack = self._attach_candidate_graph(orders_df, transition_pack, req.config)
            template_build_seconds = perf_counter() - build_t0
            self._print_template_diagnostics(transition_pack)

            schedule_df, rounds_df, dropped_df, engine_meta = solve_master_model(req, transition_pack=transition_pack, orders_df=orders_df)
            effective_cfg = engine_meta.get("effective_config", req.config) if isinstance(engine_meta, dict) else req.config
            if isinstance(engine_meta, dict) and engine_meta.get("precheck_autorelax_applied") and effective_cfg is not req.config:
                transition_pack = build_transition_templates(orders_df, effective_cfg)
                transition_pack = self._attach_candidate_graph(orders_df, transition_pack, effective_cfg)
            result = ColdRollingResult(
                schedule_df=schedule_df,
                rounds_df=rounds_df,
                output_path=Path(req.output_path),
                dropped_df=dropped_df,
                engine_meta=engine_meta,
                config=effective_cfg,
            )

            # ---- Decode phase ----
            t_decode_start = perf_counter()
            result = decode_solution(result)
            decode_seconds = perf_counter() - t_decode_start
            annotated_dropped = ColdRollingPipeline._annotate_dropped_orders(
                orders_df,
                result.dropped_df if isinstance(result.dropped_df, pd.DataFrame) else pd.DataFrame(),
                result.engine_meta or {},
            )
            result = replace(result, dropped_df=annotated_dropped)

            updated_engine_meta = dict(result.engine_meta or {})

            # ---- Decode order mismatch acceptance gate (hard block) ----
            # If decode introduced order corruption (seq != master_seq), the offending
            # campaign has already been demoted in decode_solution. However, we must
            # also mark the result as NOT OFFICIAL_READY and update acceptance gate.
            decode_order_mismatch_count = int(updated_engine_meta.get("decode_order_mismatch_count", 0))
            decode_order_integrity_ok = bool(decode_order_mismatch_count == 0)
            updated_engine_meta["decode_order_integrity_ok"] = decode_order_integrity_ok
            updated_engine_meta["decode_order_mismatch_count"] = decode_order_mismatch_count
            if not decode_order_integrity_ok:
                print(
                    f"[APS][decode_gate] HARD_BLOCK: decode_order_mismatch detected, "
                    f"campaigns={updated_engine_meta.get('decode_order_mismatch_campaigns', [])}, "
                    f"demoted_orders={updated_engine_meta.get('decode_demoted_order_count', 0)}"
                )
                # Record into acceptance gate diagnostics
                updated_engine_meta["_decode_gate_block"] = True
                updated_engine_meta["_decode_gate_block_reason"] = "DECODE_ORDER_MISMATCH"
            else:
                print(
                    f"[APS][decode_gate] decode_order_integrity_ok=True, "
                    f"decode_order_mismatch_count=0"
                )
                updated_engine_meta["_decode_gate_block"] = False

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
                # Priority 1: capture full eq breakdown for diagnostics
                updated_engine_meta["adjacency_violation_cnt"] = int(eq.get("adjacency_violation_cnt", 0))
                updated_engine_meta["bridge_expand_violation_cnt"] = int(eq.get("bridge_expand_violation_cnt", 0))
                updated_engine_meta["chain_break_cnt"] = int(eq.get("chain_break_cnt", 0))
                updated_engine_meta["bridge_path_expand_miss_cnt"] = int(eq.get("bridge_path_expand_miss_cnt", 0))
                updated_engine_meta["template_miss_cnt"] = int(eq.get("template_miss_cnt", 0))
                updated_engine_meta["result_acceptance_status"] = "BEST_SEARCH_CANDIDATE_ANALYSIS"
                updated_engine_meta["result_usage"] = "ANALYSIS_ONLY"
                updated_engine_meta["failure_mode"] = str(updated_engine_meta.get("failure_mode", "FAILED_ROUTING_SEARCH") or "FAILED_ROUTING_SEARCH")
            elif str(updated_engine_meta.get("result_acceptance_status", "")).startswith("FAILED_") and (
                result.schedule_df is None or result.schedule_df.empty
            ):
                summary = validate_solution_summary(result, result.config.rule)
                # FAILED branch: schedule may be empty, but still call validate_model_equivalence
                # to get full adjacency/bridge_expand violation breakdown for diagnostics
                _failed_eq = validate_model_equivalence(
                    result.schedule_df if result.schedule_df is not None else pd.DataFrame(),
                    transition_pack.get("templates") if isinstance(transition_pack, dict) else None,
                )
                eq = {
                    "template_pair_ok": bool(_failed_eq.get("template_pair_ok", False)),
                    "adjacency_rule_ok": bool(_failed_eq.get("adjacency_rule_ok", False)),
                    "bridge_expand_ok": bool(_failed_eq.get("bridge_expand_ok", False)),
                    "adjacency_violation_cnt": int(_failed_eq.get("adjacency_violation_cnt", 0)),
                    "bridge_expand_violation_cnt": int(_failed_eq.get("bridge_expand_violation_cnt", 0)),
                    "chain_break_cnt": int(_failed_eq.get("chain_break_cnt", 0)),
                    "bridge_path_expand_miss_cnt": int(_failed_eq.get("bridge_path_expand_miss_cnt", 0)),
                    "template_miss_cnt": int(_failed_eq.get("template_miss_cnt", 0)),
                    "hint_mismatch_cnt": int(_failed_eq.get("hint_mismatch_cnt", 0)),
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
                    partial_eval = self._evaluate_partial_acceptance(result, orders_df, validation_summary=summary)
                    updated_engine_meta["partial_result_available"] = bool(partial_eval["partial_result_available"])
                    updated_engine_meta["partial_acceptance_passed"] = bool(partial_eval["partial_acceptance_passed"])
                    updated_engine_meta["partial_drop_ratio"] = float(partial_eval["partial_drop_ratio"])
                    updated_engine_meta["partial_drop_tons_ratio"] = float(partial_eval["partial_drop_tons_ratio"])
                    updated_engine_meta["hard_violation_count_total"] = int(partial_eval["hard_violation_count_total"])
                    updated_engine_meta["campaign_ton_hard_violation_count_total"] = int(partial_eval["campaign_ton_hard_violation_count_total"])
                    updated_engine_meta["partial_acceptance_block_reason"] = str(partial_eval.get("partial_acceptance_block_reason", ""))

                    # HARD RULE: Only accept PARTIAL_SCHEDULE_WITH_DROPS when hard violations = 0
                    # HARD RULE: decode_order_integrity_ok must be True for any OFFICIAL_* status
                    _decode_ok = bool(updated_engine_meta.get("decode_order_integrity_ok", True))
                    if isinstance(result.dropped_df, pd.DataFrame) and not result.dropped_df.empty and partial_eval["partial_acceptance_passed"]:
                        updated_engine_meta["result_acceptance_status"] = "PARTIAL_SCHEDULE_WITH_DROPS"
                        updated_engine_meta["failure_mode"] = "PARTIAL_SCHEDULE_WITH_DROPS"
                        updated_engine_meta["result_usage"] = "PARTIAL_OFFICIAL"
                    elif isinstance(result.dropped_df, pd.DataFrame) and not result.dropped_df.empty:
                        # Distinguish: hard violations vs soft threshold failure
                        hard_viols_zero = partial_eval.get("hard_violations_zero", True)
                        block_reason = partial_eval.get("partial_acceptance_block_reason", "")
                        if not hard_viols_zero:
                            # routing feasible but hard violations exist
                            updated_engine_meta["failure_mode"] = "FAILED_PARTIAL_ACCEPTANCE_HARD_VIOLATIONS"
                            print(
                                f"[APS][结果门槛] routing 已可行，但 hard violations 未通过 "
                                f"({block_reason}), hard_violations={partial_eval['hard_violation_count_total']}, "
                                f"结果降级为 BEST_SEARCH_CANDIDATE_ANALYSIS"
                            )
                        else:
                            # routing feasible, no hard violations, but soft threshold not met
                            updated_engine_meta["failure_mode"] = "FAILED_PARTIAL_ACCEPTANCE_SOFT_THRESHOLD_NOT_MET"
                            print(
                                f"[APS][结果门槛] routing 已可行，但 partial acceptance 软阈值未通过 "
                                f"({block_reason}), hard_violations={partial_eval['hard_violation_count_total']}, "
                                f"结果降级为 BEST_SEARCH_CANDIDATE_ANALYSIS"
                            )
                        updated_engine_meta["result_acceptance_status"] = "BEST_SEARCH_CANDIDATE_ANALYSIS"
                        updated_engine_meta["result_usage"] = "ANALYSIS_ONLY"
                    # HARD GATE: decode_order_integrity_ok must be True for OFFICIAL_FULL_SCHEDULE
                    elif _decode_ok:
                        updated_engine_meta["result_acceptance_status"] = "OFFICIAL_FULL_SCHEDULE"
                        updated_engine_meta["failure_mode"] = ""
                        updated_engine_meta["result_usage"] = "OFFICIAL"
                    else:
                        # decode mismatch but routing feasible: degrade to ANALYSIS_ONLY
                        updated_engine_meta["result_acceptance_status"] = "BEST_SEARCH_CANDIDATE_ANALYSIS"
                        updated_engine_meta["failure_mode"] = "FAILED_DECODE_ORDER_MISMATCH"
                        updated_engine_meta["result_usage"] = "ANALYSIS_ONLY"
                        print(
                            f"[APS][decode_gate][acceptance] routing_feasible=True but decode_order_integrity_ok=False, "
                            f"campaigns={updated_engine_meta.get('decode_order_mismatch_campaigns', [])}, "
                            f"result degraded to BEST_SEARCH_CANDIDATE_ANALYSIS"
                        )
                else:
                    updated_engine_meta["result_acceptance_status"] = "FAILED_ROUTING_SEARCH"
                    updated_engine_meta["result_usage"] = "ANALYSIS_ONLY"
            if bool(updated_engine_meta.get("hard_cap_not_enforced", False)):
                updated_engine_meta["result_acceptance_status"] = "FAILED_IMPLEMENTATION_ERROR"
                updated_engine_meta["failure_mode"] = "FAILED_IMPLEMENTATION_ERROR"
        
            # Priority 1: Always expose eq (validate_model_equivalence) violation breakdown to engine_meta
            # This ensures full adjacency/bridge_expand diagnostics are available regardless of acceptance path
            updated_engine_meta["adjacency_violation_cnt"] = int(eq.get("adjacency_violation_cnt", 0))
            updated_engine_meta["bridge_expand_violation_cnt"] = int(eq.get("bridge_expand_violation_cnt", 0))
            updated_engine_meta["chain_break_cnt"] = int(eq.get("chain_break_cnt", 0))
            updated_engine_meta["bridge_path_expand_miss_cnt"] = int(eq.get("bridge_path_expand_miss_cnt", 0))
            updated_engine_meta["template_miss_cnt"] = int(eq.get("template_miss_cnt", 0))
            updated_engine_meta["hint_mismatch_cnt"] = int(eq.get("hint_mismatch_cnt", 0))
            # bridge_expand_ok = (bridge_expand_violation_cnt == 0) and (bridge_path_expand_miss_cnt == 0)
            updated_engine_meta["bridge_expand_ok"] = bool(
                updated_engine_meta["bridge_expand_violation_cnt"] == 0
                and updated_engine_meta["bridge_path_expand_miss_cnt"] == 0
            )
            # adjacency_rule_ok = adjacency_violation_cnt == 0
            updated_engine_meta["adjacency_rule_ok"] = bool(updated_engine_meta["adjacency_violation_cnt"] == 0)
        
            # Priority 1: Merge eq violation breakdown into summary for unified gate checks
            # This ensures hard_violations used in acceptance gate includes bridge_expand violations
            summary["adjacency_violation_cnt"] = int(eq.get("adjacency_violation_cnt", 0))
            summary["bridge_expand_violation_cnt"] = int(eq.get("bridge_expand_violation_cnt", 0))
            summary["bridge_path_expand_miss_cnt"] = int(eq.get("bridge_path_expand_miss_cnt", 0))
            summary["template_miss_cnt"] = int(eq.get("template_miss_cnt", 0))
            # Priority 1: Merge decode order mismatch metadata into summary for gate diagnostics
            summary["decode_order_integrity_ok"] = bool(updated_engine_meta.get("decode_order_integrity_ok", True))
            summary["decode_order_mismatch_count"] = int(updated_engine_meta.get("decode_order_mismatch_count", 0))
            summary["decode_order_mismatch_campaigns"] = list(updated_engine_meta.get("decode_order_mismatch_campaigns", []))
            summary["decode_demoted_order_count"] = int(updated_engine_meta.get("decode_demoted_order_count", 0))
        
            # Priority 4: Failure source identification - propagate per-line violation counts from summary
            updated_engine_meta["failure_source_line"] = str(summary.get("failure_source_line", ""))
            updated_engine_meta["failure_source_max_violations"] = int(summary.get("failure_source_max_violations", 0))
            updated_engine_meta["failure_source_summary"] = str(summary.get("failure_source_summary", "no_adjacency_violations"))
            for k, v in summary.items():
                if (
                    k.endswith("_adjacency_violation_cnt")
                    or k.endswith("_width_jump_violation_cnt")
                    or k.endswith("_thickness_violation_cnt")
                    or k.endswith("_temp_conflict_cnt")
                    or k.endswith("_non_pc_direct_switch_cnt")
                    or k.endswith("_real_orders")
                    or k.endswith("_campaign_count")
                ):
                    updated_engine_meta[k] = v
            # template_pair_ok already set in each branch
        
            acceptance_status = str(updated_engine_meta.get("result_acceptance_status", ""))
        
            # Priority 4: Clarify acceptance status fields
            # master_candidate_status: status from master/feasibility solve
            master_candidate_status = str(updated_engine_meta.get("best_candidate_search_status", "UNKNOWN"))
            updated_engine_meta["master_candidate_status"] = master_candidate_status
        
            # Priority 1: Use summary hard_violation_count_total for all gate checks
            # FIX: hard_violations was undefined, now properly sourced from validation summary
            hard_violations = int(summary.get("hard_violation_count_total", 0))
        
            # validated_feasible: whether final validation passed (hard_violation_count_total = 0)
            validated_feasible = bool(routing_feasible and hard_violations == 0)
            updated_engine_meta["validated_feasible"] = validated_feasible
        
            # Priority 1: validated_feasible_candidate_available must be defined BEFORE acceptance_gate_reason
            # This determines if the result can be used for optimize phase
            validated_feasible_candidate_available = (
                routing_feasible
                and hard_violations == 0
                and acceptance_status in {"OFFICIAL_FULL_SCHEDULE", "PARTIAL_SCHEDULE_WITH_DROPS"}
            )
            can_enter_optimize = bool(validated_feasible_candidate_available)
            optimize_block_reason = ""
            if not routing_feasible:
                optimize_block_reason = "ROUTING_NOT_FEASIBLE"
            elif hard_violations > 0:
                optimize_block_reason = "HARD_VIOLATIONS_PRESENT"
            elif not updated_engine_meta.get("partial_acceptance_passed", False):
                optimize_block_reason = "PARTIAL_ACCEPTANCE_SOFT_THRESHOLD_NOT_MET"
            elif acceptance_status not in {"OFFICIAL_FULL_SCHEDULE", "PARTIAL_SCHEDULE_WITH_DROPS"}:
                optimize_block_reason = "PARTIAL_ACCEPTANCE_NOT_PASSED"
        
            # acceptance_gate_reason: why the result got its acceptance status
            if acceptance_status == "OFFICIAL_FULL_SCHEDULE":
                acceptance_gate_reason = "NO_DROP_HARD_VIOLATIONS_ZERO"
            elif acceptance_status == "PARTIAL_SCHEDULE_WITH_DROPS":
                acceptance_gate_reason = "HARD_VIOLATIONS_ZERO_DROP_WITHIN_THRESHOLD"
            elif acceptance_status == "BEST_SEARCH_CANDIDATE_ANALYSIS":
                if not routing_feasible:
                    acceptance_gate_reason = "ROUTING_INFEASIBLE"
                elif hard_violations > 0:
                    acceptance_gate_reason = f"HARD_VIOLATIONS_EXIST:{hard_violations}"
                elif not updated_engine_meta.get("partial_acceptance_passed", False):
                    # routing feasible, no hard violations, but soft threshold not met
                    acceptance_gate_reason = "PARTIAL_ACCEPTANCE_SOFT_THRESHOLD_NOT_MET"
                elif not validated_feasible_candidate_available:
                    acceptance_gate_reason = "PARTIAL_ACCEPTANCE_NOT_PASSED"
                else:
                    acceptance_gate_reason = "ANALYSIS_ONLY_MODE"
            elif acceptance_status.startswith("FAILED_"):
                acceptance_gate_reason = acceptance_status
            else:
                acceptance_gate_reason = "UNKNOWN"
            updated_engine_meta["acceptance_gate_reason"] = acceptance_gate_reason

            # validation_gate_reason: specific reason for validation pass/fail
            if validated_feasible:
                validation_gate_reason = "ALL_HARD_CONSTRAINTS_SATISFIED"
            elif not routing_feasible:
                validation_gate_reason = "ROUTING_FEASIBILITY_FAILED"
            elif hard_violations > 0:
                validation_gate_reason = f"HARD_VIOLATIONS:{hard_violations}"
            else:
                # routing feasible, no hard violations, but soft threshold failed
                validation_gate_reason = "PARTIAL_ACCEPTANCE_SOFT_THRESHOLD_NOT_MET"
            updated_engine_meta["validation_gate_reason"] = validation_gate_reason
        
            # Priority 1: print gate status AFTER all fields are set
            print(
                f"[APS][optimize_gate] validated_feasible={validated_feasible}, "
                f"validated_feasible_candidate={validated_feasible_candidate_available}, "
                f"can_enter_optimize={can_enter_optimize}, "
                f"routing_feasible={routing_feasible}, "
                f"hard_violations={hard_violations}, "
                f"acceptance={acceptance_status}, "
                f"master_candidate={master_candidate_status}, "
                f"acceptance_gate_reason={acceptance_gate_reason}, "
                f"validation_gate_reason={validation_gate_reason}, "
                f"block_reason={optimize_block_reason or 'NONE'}"
            )
            updated_engine_meta["validated_feasible_candidate_available"] = bool(validated_feasible_candidate_available)
            updated_engine_meta["can_enter_optimize"] = bool(can_enter_optimize)
            updated_engine_meta["optimize_block_reason"] = str(optimize_block_reason)
            # Phase tracking: feasibility always executed, optimize only if gate passes
            updated_engine_meta["feasibility_phase_executed"] = True
            updated_engine_meta["optimize_phase_executed"] = bool(can_enter_optimize)
            # Official solution source: set by pipeline after validation gate
            if acceptance_status in {"OFFICIAL_FULL_SCHEDULE", "PARTIAL_SCHEDULE_WITH_DROPS"}:
                updated_engine_meta["official_solution_source"] = "FEASIBILITY_PHASE"
            elif acceptance_status == "BEST_SEARCH_CANDIDATE_ANALYSIS":
                updated_engine_meta["official_solution_source"] = "NONE"
            if can_enter_optimize:
                # Second solve: run optimize phase on top of validated feasible result
                print(f"[APS][optimize_gate] Entering optimize phase (phase_mode=optimize_only)...")
                from aps_cp_sat.model import solve_master_model as _solve_master_model
                _build_t0 = perf_counter()
                _transition_pack = build_transition_templates(orders_df, result.config)
                _template_build_seconds = perf_counter() - _build_t0
                opt_sched_df, opt_rounds_df, opt_dropped_df, opt_engine_meta = _solve_master_model(
                    req,
                    transition_pack=_transition_pack,
                    orders_df=orders_df,
                    phase_mode="optimize_only",
                )
                _opt_elapsed = perf_counter() - _build_t0
                updated_engine_meta["optimize_phase_executed"] = True
                updated_engine_meta["optimize_phase_improved_solution"] = False
                updated_engine_meta["optimize_joint_master_seconds"] = float(opt_engine_meta.get("joint_master_seconds", 0.0) or 0.0)
                updated_engine_meta["optimize_fallback_seconds"] = float(opt_engine_meta.get("fallback_total_seconds", 0.0) or 0.0)
                updated_engine_meta["template_build_seconds"] = float(template_build_seconds) + _template_build_seconds + float(updated_engine_meta.get("template_build_seconds", 0.0) or 0.0)
                # Validate optimize result
                # ONLY accept optimize result when hard_violation_count_total == 0 (the single hard gate)
                if isinstance(opt_sched_df, pd.DataFrame) and not opt_sched_df.empty:
                    _opt_eq = validate_model_equivalence(opt_sched_df, _transition_pack.get("templates") if isinstance(_transition_pack, dict) else None)
                    _opt_routing_feasible = bool(_opt_eq.get("template_pair_ok", False) and _opt_eq.get("adjacency_rule_ok", False) and _opt_eq.get("bridge_expand_ok", False))
                    _opt_summary = validate_solution_summary(replace(result, schedule_df=opt_sched_df, dropped_df=opt_dropped_df), result.config.rule)
                    _opt_hard_viol_total = int(_opt_summary.get("hard_violation_count_total", 0) or 0)
                    _opt_campaign_hard_viol = int(_opt_summary.get("campaign_ton_hard_violation_count_total", 0) or 0)
                    # Diagnostics for optimize validation
                    updated_engine_meta["optimize_validation_hard_violation_count_total"] = _opt_hard_viol_total
                    updated_engine_meta["optimize_validation_campaign_ton_hard_violation_count_total"] = _opt_campaign_hard_viol
                    _opt_result_accepted = False
                    if _opt_routing_feasible and _opt_hard_viol_total == 0:
                        # Optimize succeeded: hard_violation_count_total == 0 is the ONLY acceptance gate
                        result = replace(result, schedule_df=opt_sched_df, dropped_df=opt_dropped_df)
                        updated_engine_meta["official_solution_source"] = "OPTIMIZE_PHASE"
                        updated_engine_meta["optimize_phase_improved_solution"] = True
                        _opt_result_accepted = True
                        print(f"[APS][optimize_gate] Optimize succeeded: hard_viol_total={_opt_hard_viol_total}, campaign_hard_viol={_opt_campaign_hard_viol}")
                    else:
                        # Reject optimize result: keep pre-opt feasible result
                        updated_engine_meta["official_solution_source"] = "FEASIBILITY_PHASE"
                        updated_engine_meta["optimize_phase_improved_solution"] = False
                        _reject_reason = "ROUTING_INFEASIBLE" if not _opt_routing_feasible else f"HARD_VIOLATIONS:{_opt_hard_viol_total}"
                        print(f"[APS][optimize_gate] Optimize rejected: reason={_reject_reason}, hard_viol_total={_opt_hard_viol_total}, campaign_hard_viol={_opt_campaign_hard_viol}, keeping feasibility result")
                    updated_engine_meta["optimize_result_accepted"] = _opt_result_accepted
                    updated_engine_meta["optimize_result_rejected_due_to_hard_violations"] = bool(not _opt_result_accepted)
                else:
                    updated_engine_meta["optimize_validation_hard_violation_count_total"] = -1
                    updated_engine_meta["optimize_validation_campaign_ton_hard_violation_count_total"] = -1
                    updated_engine_meta["optimize_result_accepted"] = False
                    updated_engine_meta["optimize_result_rejected_due_to_hard_violations"] = False
                    print(f"[APS][optimize_gate] Optimize phase produced no schedule, keeping feasibility result")
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
                # Class C: routing feasible but partial acceptance failed
                if not routing_feasible:
                    # Class A: routing infeasible
                    suffix = "_ROUTING_INFEASIBLE_NOT_PRODUCTION_READY"
                    if bool(result.config.model.export_failed_result_for_debug):
                        export_path = result.output_path.with_name(
                            f"{result.output_path.stem}{suffix}{result.output_path.suffix}"
                        )
                        print(f"[APS][结果门槛] routing 不可行，结果仅按调试输出导出: {export_path}")
                        analysis_exported = True
                        official_exported = False
                        result_usage = "ANALYSIS_ONLY"
                    elif bool(result.config.model.export_analysis_on_failure):
                        export_path = result.output_path.with_name(
                            f"{result.output_path.stem}{suffix}{result.output_path.suffix}"
                        )
                        print(f"[APS][结果门槛] routing 不可行，结果按分析用途导出: {export_path}")
                        analysis_exported = True
                        official_exported = False
                        result_usage = "ANALYSIS_ONLY"
                    else:
                        final_export_performed = False
                        analysis_exported = False
                        result_usage = "NOT_EXPORTED"
                        print("[APS][结果门槛] routing 不可行，已禁止失败结果导出")
                elif acceptance_status == "BEST_SEARCH_CANDIDATE_ANALYSIS":
                    # Class B: routing feasible but partial acceptance failed
                    suffix = "_PARTIAL_ACCEPTANCE_FAILED_ANALYSIS"
                    if bool(result.config.model.export_best_candidate_analysis):
                        export_path = result.output_path.with_name(
                            f"{result.output_path.stem}_BEST_SEARCH_CANDIDATE_ANALYSIS{result.output_path.suffix}"
                        )
                        print(f"[APS][结果门槛] routing 已可行，但 partial acceptance 未通过，"
                              f"结果降级为 BEST_SEARCH_CANDIDATE_ANALYSIS，该文件仅用于调优分析: {export_path}")
                        analysis_exported = True
                        official_exported = False
                        result_usage = "ANALYSIS_ONLY"
                    elif bool(result.config.model.export_analysis_on_failure):
                        export_path = result.output_path.with_name(
                            f"{result.output_path.stem}{suffix}{result.output_path.suffix}"
                        )
                        print(f"[APS][结果门槛] routing 已可行，但 partial acceptance 未通过，"
                              f"结果按分析用途导出: {export_path}")
                        analysis_exported = True
                        official_exported = False
                        result_usage = "ANALYSIS_ONLY"
                    elif bool(result.config.model.export_failed_result_for_debug):
                        export_path = result.output_path.with_name(
                            f"{result.output_path.stem}{suffix}{result.output_path.suffix}"
                        )
                        print(f"[APS][结果门槛] routing 已可行，但 partial acceptance 未通过，"
                              f"结果按调试用途导出: {export_path}")
                        analysis_exported = True
                        official_exported = False
                        result_usage = "ANALYSIS_ONLY"
                    else:
                        final_export_performed = False
                        analysis_exported = False
                        result_usage = "NOT_EXPORTED"
                        print("[APS][结果门槛] routing 已可行，但 partial acceptance 未通过，已禁止导出")
                else:
                    # routing_feasible=True, acceptance_status not official, and all export flags off
                    final_export_performed = False
                    official_exported = False
                    analysis_exported = False
                    result_usage = "NOT_EXPORTED"
                    print("[APS][结果门槛] routing 已可行，但 acceptance 状态异常，已禁止导出")
            else:
                official_exported = True
                analysis_exported = False
                result_usage = str(updated_engine_meta.get("result_usage", "OFFICIAL"))
            updated_engine_meta["final_export_performed"] = bool(final_export_performed)
            updated_engine_meta["official_exported"] = bool(official_exported)
            updated_engine_meta["analysis_exported"] = bool(analysis_exported)
            updated_engine_meta["result_usage"] = str(result_usage)
            # export_block_stage: which stage blocked the result from being official
            if not routing_feasible:
                updated_engine_meta["export_block_stage"] = "routing"
            elif acceptance_status == "BEST_SEARCH_CANDIDATE_ANALYSIS":
                updated_engine_meta["export_block_stage"] = "partial_acceptance"
            else:
                updated_engine_meta["export_block_stage"] = "none"
            updated_engine_meta["total_run_seconds"] = float(perf_counter() - run_t0)
            updated_engine_meta["decode_seconds"] = float(decode_seconds) if 'decode_seconds' in dir() else 0.0
            # ---- Expose LNS phase timing from engine_meta ----
            lns_engine_meta = updated_engine_meta.get("lns_engine_meta", {}) or {}
            lns_diag = updated_engine_meta.get("lns_diagnostics", {}) or {}
            cut_diags = lns_diag.get("campaign_cut_diags", {}) or {}
            tail_summary = cut_diags.get("tail_rebalance_summary", {}) or {}
            updated_engine_meta["constructive_build_seconds"] = float(lns_engine_meta.get("constructive_build_seconds", 0.0) or 0.0)
            updated_engine_meta["campaign_cutter_seconds"] = float(lns_engine_meta.get("campaign_cutter_seconds", 0.0) or 0.0)
            updated_engine_meta["lns_total_seconds"] = float(lns_engine_meta.get("lns_total_seconds", 0.0) or 0.0)
            updated_engine_meta["tail_repair_seconds"] = float(tail_summary.get("tail_repair_seconds", 0.0) or 0.0)
            updated_engine_meta["recut_seconds"] = float(tail_summary.get("recut_seconds", 0.0) or 0.0)
            updated_engine_meta["shift_seconds"] = float(tail_summary.get("shift_seconds", 0.0) or 0.0)
            updated_engine_meta["fill_seconds"] = float(tail_summary.get("fill_seconds", 0.0) or 0.0)
            updated_engine_meta["merge_seconds"] = float(tail_summary.get("merge_seconds", 0.0) or 0.0)
            updated_engine_meta["export_consistency_ok"] = True

            # ---- Expose run_path_fingerprint fields from nested engine_meta ----
            lns_engine_meta = updated_engine_meta.get("lns_engine_meta", {}) or {}
            lns_diag = updated_engine_meta.get("lns_diagnostics", {}) or {}
            cut_diags = lns_diag.get("campaign_cut_diags", {}) or {}
            build_diags = lns_diag.get("constructive_build_diags", {}) or {}

            def _int(v, default=0):
                try:
                    return int(v or default)
                except (ValueError, TypeError):
                    return default

            def _str(v, default=""):
                return str(v) if v is not None else default

            # Run path fingerprints
            updated_engine_meta["run_path_fingerprint_pipeline"] = _str(
                updated_engine_meta.get("run_path_fingerprint_pipeline", "PIPELINE_V2_20260416A")
            )
            updated_engine_meta["run_path_fingerprint_constructive_builder"] = _str(
                updated_engine_meta.get("run_path_fingerprint_constructive_builder", "CONSTRUCTIVE_SEQUENCE_BUILDER_V2_20260416A")
            )
            updated_engine_meta["run_path_fingerprint_campaign_cutter"] = _str(
                updated_engine_meta.get("run_path_fingerprint_campaign_cutter", "")
            )
            updated_engine_meta["run_path_fingerprint_constructive_lns_master"] = _str(
                lns_engine_meta.get("run_path_fingerprint_constructive_lns_master", "")
            )

            # Tail repair diagnostics (from campaign_cutter)
            tail_summary = cut_diags.get("tail_rebalance_summary", {}) or {}
            updated_engine_meta["tail_repair_recut_attempts"] = _int(tail_summary.get("tail_repair_recut_attempts"))
            updated_engine_meta["tail_repair_recut_success"] = _int(tail_summary.get("tail_repair_recut_success"))
            updated_engine_meta["tail_repair_shift_attempts"] = _int(tail_summary.get("tail_repair_shift_attempts"))
            updated_engine_meta["tail_repair_shift_success"] = _int(tail_summary.get("tail_repair_shift_success"))
            updated_engine_meta["tail_repair_fill_attempts"] = _int(tail_summary.get("tail_repair_fill_attempts"))
            updated_engine_meta["tail_repair_fill_success"] = _int(tail_summary.get("tail_repair_fill_success"))
            updated_engine_meta["tail_repair_merge_attempts"] = _int(tail_summary.get("tail_repair_merge_attempts"))
            updated_engine_meta["tail_repair_merge_success"] = _int(tail_summary.get("tail_repair_merge_success"))

            # Underfilled reconstruction / repair-only real bridge observability.
            # Source of truth is campaign_cutter diagnostics; expose selected fields at top-level engine_meta.
            _recon_keys = [
                "underfilled_reconstruction_enabled",
                "underfilled_reconstruction_attempts",
                "underfilled_reconstruction_success",
                "underfilled_reconstruction_blocks_tested",
                "underfilled_reconstruction_blocks_skipped",
                "underfilled_reconstruction_valid_before",
                "underfilled_reconstruction_valid_after",
                "underfilled_reconstruction_underfilled_before",
                "underfilled_reconstruction_underfilled_after",
                "underfilled_reconstruction_valid_delta",
                "underfilled_reconstruction_underfilled_delta",
                "underfilled_reconstruction_segments_salvaged",
                "underfilled_reconstruction_orders_salvaged",
                "underfilled_reconstruction_not_entered_reason",
                "repair_only_real_bridge_enabled",
                "repair_only_real_bridge_attempts",
                "repair_only_real_bridge_success",
                "repair_only_real_bridge_candidates_total",
                "repair_only_real_bridge_candidates_kept",
                "repair_only_real_bridge_filtered_direct_feasible",
                "repair_only_real_bridge_filtered_pair_invalid",
                "repair_only_real_bridge_filtered_ton_invalid",
                "repair_only_real_bridge_filtered_score_worse",
                "repair_only_real_bridge_filtered_bridge_limit_exceeded",
                "repair_only_real_bridge_filtered_multiplicity_invalid",
                "repair_only_real_bridge_filtered_bridge_path_not_real",
                "repair_only_real_bridge_filtered_bridge_path_missing",
                "repair_only_real_bridge_filtered_block_order_mismatch",
                "repair_only_real_bridge_filtered_line_mismatch",
                "repair_only_real_bridge_filtered_block_membership_mismatch",
                "repair_only_real_bridge_filtered_bridge_path_payload_empty",
                "repair_bridge_pack_has_real_rows",
                "repair_bridge_pack_real_rows_total",
                "repair_bridge_pack_virtual_rows_total",
                "repair_bridge_raw_rows_total",
                "repair_bridge_matched_rows_total",
                "repair_bridge_kept_rows_total",
                "repair_bridge_endpoint_key_mismatch_count",
                "repair_bridge_field_name_mismatch_count",
                "repair_bridge_inconsistency_count",
                "repair_bridge_boundary_band_enabled",
                "repair_bridge_band_pairs_tested",
                "repair_bridge_band_hits",
                "repair_bridge_single_point_hits",
                "repair_bridge_band_only_hits",
                "repair_bridge_band_best_distance",
                "repair_bridge_endpoint_adjustment_enabled",
                "repair_bridge_adjustments_generated",
                "repair_bridge_adjustment_pairs_tested",
                "repair_bridge_adjustment_hits",
                "repair_bridge_adjustment_only_hits",
                "repair_bridge_best_adjustment_cost",
                "repair_bridge_candidates_matched",
                "repair_bridge_candidates_rejected_pair_invalid",
                "repair_bridge_candidates_rejected_ton_invalid",
                "repair_bridge_candidates_rejected_score_worse",
                "repair_bridge_candidates_accepted",
                "repair_bridge_exact_invalid_pair_count",
                "repair_bridge_frontier_mismatch_count",
                "repair_bridge_pair_invalid_width",
                "repair_bridge_pair_invalid_thickness",
                "repair_bridge_pair_invalid_temp",
                "repair_bridge_pair_invalid_group",
                "repair_bridge_pair_invalid_unknown",
                "repair_bridge_ton_rescue_attempts",
                "repair_bridge_ton_rescue_success",
                "repair_bridge_ton_rescue_windows_tested",
                "repair_bridge_ton_rescue_valid_delta",
                "repair_bridge_ton_rescue_underfilled_delta",
                "repair_bridge_ton_rescue_scheduled_orders_delta",
                "repair_bridge_filtered_ton_below_min_current_block",
                "repair_bridge_filtered_ton_below_min_even_after_neighbor_expansion",
                "repair_bridge_filtered_ton_above_max_after_expansion",
                "repair_bridge_filtered_ton_split_not_found",
                "repair_bridge_filtered_ton_rescue_no_gain",
                "repair_bridge_filtered_ton_rescue_impossible",
                "repair_bridge_ton_rescue_pair_fail_width",
                "repair_bridge_ton_rescue_pair_fail_thickness",
                "repair_bridge_ton_rescue_pair_fail_temp",
                "repair_bridge_ton_rescue_pair_fail_group",
                "repair_bridge_ton_rescue_pair_fail_template",
                "repair_bridge_ton_rescue_pair_fail_multi",
                "repair_bridge_ton_rescue_pair_fail_unknown",
                "bridgeability_route_suggestion",
                "bridgeability_census",
                "bridgeability_census_items",
                # ---- Legacy virtual pilot: 已降级为单一总字段 ----
                # 旧有 ~45 个 virtual_pilot 细项字段已从此处移除
                "conservative_apply_attempt_count",
                "conservative_apply_success_count",
                "conservative_apply_reject_count",
                "repair_bridge_pack_type",
                "repair_bridge_pack_keys",
                "repair_bridge_pack_line_keys",
                "repair_only_real_bridge_used_segments",
                "repair_only_real_bridge_used_orders",
                "repair_only_real_bridge_not_entered_reason",
                "underfilled_reconstruction_seconds",
                "repair_only_real_bridge_seconds",
            ]
            for _k in _recon_keys:
                if isinstance(cut_diags, dict) and _k in cut_diags:
                    updated_engine_meta[_k] = cut_diags.get(_k)
                else:
                    if _k in {"repair_bridge_pack_keys", "repair_bridge_pack_line_keys", "bridgeability_census_items"}:
                        _default_val = []
                    elif _k == "bridgeability_census":
                        _default_val = {}
                    elif _k in {"repair_bridge_pack_type", "bridgeability_route_suggestion"} or _k.endswith("_reason"):
                        _default_val = ""
                    elif _k == "repair_bridge_pack_has_real_rows":
                        _default_val = False
                    elif _k == "repair_bridge_boundary_band_enabled":
                        _default_val = True
                    elif _k == "repair_bridge_endpoint_adjustment_enabled":
                        _default_val = True
                    elif _k in {"repair_bridge_band_best_distance", "repair_bridge_best_adjustment_cost"}:
                        _default_val = -1
                    elif _k.endswith("_seconds"):
                        _default_val = 0.0
                    else:
                        _default_val = 0
                    updated_engine_meta.setdefault(_k, _default_val)

            # Final segment salvage diagnostics (from constructive_lns_master)
            updated_engine_meta["final_segment_salvage_attempts"] = _int(lns_diag.get("final_segment_salvage_attempts"))
            updated_engine_meta["final_segment_salvage_success_count"] = _int(lns_diag.get("final_segment_salvage_success_count"))
            updated_engine_meta["final_segment_salvaged_piece_count"] = _int(lns_diag.get("final_segment_salvaged_piece_count"))
            updated_engine_meta["final_segment_demoted_fragment_count"] = _int(lns_diag.get("final_segment_demoted_fragment_count"))
            updated_engine_meta["final_segment_full_drop_count"] = _int(lns_diag.get("final_segment_full_drop_count"))

            updated_engine_meta = self._ensure_unified_engine_meta(
                updated_engine_meta,
                result.config,
                schedule_df=result.schedule_df if isinstance(result.schedule_df, pd.DataFrame) else pd.DataFrame(),
                dropped_df=result.dropped_df if isinstance(result.dropped_df, pd.DataFrame) else pd.DataFrame(),
                rounds_df=result.rounds_df if isinstance(result.rounds_df, pd.DataFrame) else pd.DataFrame(),
            )
            result = replace(result, engine_meta=updated_engine_meta, output_path=export_path)
            em = result.engine_meta or {}
            self._print_result_snapshot(em)
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
                f"validated_feasible={em.get('validated_feasible', False)}, "
                f"acceptance={em.get('result_acceptance_status', 'UNKNOWN')}, "
                f"master_candidate_status={em.get('master_candidate_status', 'UNKNOWN')}, "
                f"acceptance_gate_reason={em.get('acceptance_gate_reason', 'UNKNOWN')}, "
                f"validation_gate_reason={em.get('validation_gate_reason', 'UNKNOWN')}, "
                f"official_solution_source={em.get('official_solution_source', 'NONE')}, "
                f"failure_mode={em.get('failure_mode', '')}, "
                f"evidence_level={em.get('evidence_level', 'OK')}, "
                # Priority 4: Show failure source in diagnostics
                f"adj_viols={em.get('adjacency_violation_cnt', 0)}, "
                f"bridge_expand_viols={em.get('bridge_expand_violation_cnt', 0)}, "
                f"failure_source={em.get('failure_source_summary', 'none')}, "
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

            # ---- Final PHASE_TIMING summary ----
            _tpl_sec = float(em.get("template_build_seconds", 0.0) or 0.0)
            _cnstr_sec = float(em.get("constructive_build_seconds", 0.0) or 0.0)
            _cut_sec = float(em.get("campaign_cutter_seconds", 0.0) or 0.0)
            _lns_sec = float(em.get("lns_total_seconds", 0.0) or 0.0)
            _dec_sec = float(em.get("decode_seconds", 0.0) or 0.0)
            _tail_sec = float(em.get("tail_repair_seconds", 0.0) or 0.0)
            _recut_sec = float(em.get("recut_seconds", 0.0) or 0.0)
            _shift_sec = float(em.get("shift_seconds", 0.0) or 0.0)
            _fill_sec = float(em.get("fill_seconds", 0.0) or 0.0)
            _merge_sec = float(em.get("merge_seconds", 0.0) or 0.0)
            print(
                f"[APS][PHASE_TIMING] template={_tpl_sec:.3f}s, constructive={_cnstr_sec:.3f}s, "
                f"cutter={_cut_sec:.3f}s, lns={_lns_sec:.3f}s, decode={_dec_sec:.3f}s, "
                f"total={em.get('total_run_seconds', 0.0):.3f}s"
            )
            print(
                f"[APS][CUTTER_TIMING] tail_repair={_tail_sec:.3f}s, "
                f"recut={_recut_sec:.3f}s, shift={_shift_sec:.3f}s, "
                f"fill={_fill_sec:.3f}s, merge={_merge_sec:.3f}s"
            )

            if bool(em.get("final_export_performed", False)):
                if str(em.get("result_usage", "")) == "ANALYSIS_ONLY" and bool(em.get("best_candidate_available", False)):
                    print(
                        "[APS][候选导出] candidate_schedule_exported=True, "
                        "candidate_line_summary_exported=True, candidate_violation_summary_exported=True"
                    )
                t_export_start = perf_counter()
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
                persist_enabled = str(os.getenv("APS_PERSIST_AFTER_EXPORT", "")).strip().lower() in {"1", "true", "yes", "y", "on"}
                updated_engine_meta["persistence_enabled"] = bool(persist_enabled)
                if persist_enabled:
                    export_path_for_persist = Path(result.output_path)
                    run_code = str(em.get("run_code") or export_path_for_persist.stem)
                    updated_engine_meta["persisted_run_code"] = run_code
                    updated_engine_meta["persisted_result_file_path"] = str(export_path_for_persist)
                    print(f"[APS][PERSIST] enabled=True, run_code={run_code}, xlsx={export_path_for_persist}")
                    try:
                        if persist_run_analysis_from_excel is None:
                            raise RuntimeError("persist_run_analysis_from_excel unavailable; install analysis platform dependencies")
                        if not export_path_for_persist.exists():
                            raise FileNotFoundError(str(export_path_for_persist))
                        persisted_run_id = persist_run_analysis_from_excel(
                            xlsx_path=str(export_path_for_persist),
                            run_code=run_code,
                        )
                        updated_engine_meta["persistence_success"] = True
                        updated_engine_meta["persisted_run_id"] = int(persisted_run_id)
                        print(f"[APS][PERSIST] success=True, run_id={persisted_run_id}")
                    except Exception as ex:
                        updated_engine_meta["persistence_success"] = False
                        updated_engine_meta["persistence_error"] = str(ex)
                        print(f"[APS][PERSIST] success=False, error={ex}")
            result_writer_seconds = perf_counter() - t_export_start
            updated_engine_meta["result_writer_seconds"] = float(result_writer_seconds)
            print(f"[APS][PHASE_TIMING] result_writer={result_writer_seconds:.3f}s")
            result = replace(result, engine_meta=updated_engine_meta)
            return result

