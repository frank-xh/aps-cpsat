from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Dict, Any
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

# Block-first import check (lazy, used only for error reporting)
_block_first_import_error: str | None = None
try:
    from aps_cp_sat.model.block_generator import generate_candidate_blocks  # noqa: F401
    from aps_cp_sat.model.block_master import solve_block_master  # noqa: F401
    from aps_cp_sat.model.block_realizer import realize_selected_blocks  # noqa: F401
    from aps_cp_sat.model.block_alns import run_block_alns  # noqa: F401
except ImportError as e:
    _block_first_import_error = str(e)


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
        #   - constructive_lns_real_bridge_frontload: Alias of mainline (Route RB)
        #   - constructive_lns_direct_only_baseline: Diagnostic baseline (Route C)
        #   - constructive_lns_virtual_guarded_frontload: Guarded virtual family experiment
        #   - block_first_guarded_search: Block-first experiment line
        # NOTE: constructive_lns_bridge_family_master / debug_acceptance 已移出默认主路径，
        # 保留于 compat/experimental_disabled/ 或 debug-only，不进入默认生产守卫。
        _ALLOWED_ALNS_PROFILES = {
            "constructive_lns_search",                      # Production mainline (Route RB)
            "constructive_lns_real_bridge_frontload",      # Alias of mainline (Route RB)
            "constructive_lns_direct_only_baseline",        # Diagnostic baseline (Route C)
            "constructive_lns_virtual_guarded_frontload",    # Guarded virtual family experiment
            "block_first_guarded_search",                   # Block-first experiment line
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

    def _assert_block_first_result(self, result: ColdRollingResult) -> None:
        """
        Hard verification that we ACTUALLY ran block_first
        """
        cfg = result.config
        profile = getattr(cfg.model, "profile_name", "")
        if profile != "block_first_guarded_search":
            return

        meta = result.engine_meta or {}
        
        actual_profile = meta.get("profile_name", "")
        solver_path = meta.get("solver_path", "")
        main_path = meta.get("main_path", "")
        
        if actual_profile != "block_first_guarded_search":
            raise RuntimeError(f"[APS][block_first_verify] profile_name mismatch. Expected block_first_guarded_search, got {actual_profile}")
        if solver_path != "block_first":
            raise RuntimeError(f"[APS][block_first_verify] solver_path != block_first. Got {solver_path}")
        if main_path != "block_first":
            raise RuntimeError(f"[APS][block_first_verify] main_path != block_first. Got {main_path}")

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
        # ---- Block-first experiment line fields ----
        "generated_blocks_total",
        "selected_blocks_count",
        "selected_order_coverage",
        "block_master_dropped_count",
        "mixed_bridge_attempt_count",
        "mixed_bridge_success_count",
        "mixed_bridge_reject_count",
        "block_alns_rounds_attempted",
        "block_alns_rounds_accepted",
        "block_swap_attempt_count",
        "block_replace_attempt_count",
        "block_split_attempt_count",
        "block_merge_attempt_count",
        "block_boundary_rebalance_attempt_count",
        "block_internal_rebuild_attempt_count",
        "avg_block_quality_score",
        "avg_block_tons_in_selected",
        "block_transition_avg_cost",
        "orders_in_realized_blocks",
        "avg_realized_block_quality",
        "block_generation_seconds",
        "block_master_seconds",
        "block_realization_seconds",
        "block_alns_seconds",
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
            "repair_bridge_ton_rescue_pair_fail_width": int(em.get("repair_bridge_ton_rescue_pair_fail_width", 0) or 0),
            "repair_bridge_ton_rescue_pair_fail_thickness": int(em.get("repair_bridge_ton_rescue_pair_fail_thickness", 0) or 0),
            "repair_bridge_ton_rescue_pair_fail_temp": int(em.get("repair_bridge_ton_rescue_pair_fail_temp", 0) or 0),
            "repair_bridge_ton_rescue_pair_fail_group": int(em.get("repair_bridge_ton_rescue_pair_fail_group", 0) or 0),
            "repair_bridge_ton_rescue_pair_fail_template": int(em.get("repair_bridge_ton_rescue_pair_fail_template", 0) or 0),
            "repair_bridge_ton_rescue_pair_fail_multi": int(em.get("repair_bridge_ton_rescue_pair_fail_multi", 0) or 0),
            "repair_bridge_ton_rescue_pair_fail_unknown": int(em.get("repair_bridge_ton_rescue_pair_fail_unknown", 0) or 0),
            "bridgeability_route_suggestion": str(em.get("bridgeability_route_suggestion", "")),
            "bridgeability_census": em.get("bridgeability_census", {}),
            "bridgeability_census_items": em.get("bridgeability_census_items", []),
            # ---- Legacy virtual pilot: 已降级为单一总字段 ----
            # 旧有 ~45 个 virtual_pilot 细项字段已从此处移除
            "conservative_apply_attempt_count": int(em.get("conservative_apply_attempt_count", 0) or 0),
            "conservative_apply_success_count": int(em.get("conservative_apply_success_count", 0) or 0),
            "conservative_apply_reject_count": int(em.get("conservative_apply_reject_count", 0) or 0),
            "repair_bridge_pack_type": str(em.get("repair_bridge_pack_type", "")),
            "repair_bridge_pack_keys": em.get("repair_bridge_pack_keys", []),
            "repair_bridge_pack_line_keys": em.get("repair_bridge_pack_line_keys", []),
            "repair_only_real_bridge_used_segments": int(em.get("repair_only_real_bridge_used_segments", 0) or 0),
            "repair_only_real_bridge_used_orders": int(em.get("repair_only_real_bridge_used_orders", 0) or 0),
            "repair_only_real_bridge_not_entered_reason": str(em.get("repair_only_real_bridge_not_entered_reason", "")),
            "underfilled_reconstruction_seconds": float(em.get("underfilled_reconstruction_seconds", 0.0) or 0.0),
            "repair_only_real_bridge_seconds": float(em.get("repair_only_real_bridge_seconds", 0.0) or 0.0),
        }
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
        
        # FINAL BLOCK_FIRST ASSERTION GATE
        self._assert_block_first_result(result)
        
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
