from __future__ import annotations

from aps_cp_sat.config.model_config import ModelConfig
from aps_cp_sat.config.planner_config import PlannerConfig
from aps_cp_sat.config.rule_config import RuleConfig
from aps_cp_sat.config.score_config import ScoreConfig


def build_profile_config(
    profile_name: str = "default",
    *,
    validation_mode: bool = False,
    production_compatibility_mode: bool = False,
) -> PlannerConfig:
    profile = str(profile_name or "default").lower()
    base_rule = RuleConfig()
    base_score = ScoreConfig()
    base_model = ModelConfig(
        profile_name=profile,
        max_orders=10_000_000,
        rounds=5,
        time_limit_seconds=150.0,
        max_virtual_chain=5,
        global_prune_max_pairs_per_from=4,
        max_routes_per_slot=5,
        validation_mode=bool(validation_mode),
        production_compatibility_mode=bool(production_compatibility_mode),
        allow_fallback=False,
        allow_legacy_fallback=bool(production_compatibility_mode) and (not bool(validation_mode)),
        enableSemanticFallback=False,
        enableScaleDownFallback=False,
        enableLegacyFallback=bool(production_compatibility_mode) and (not bool(validation_mode)),
        productionAllowLegacyFallback=bool(production_compatibility_mode),
    )

    if profile == "feasibility":
        score = ScoreConfig(
            **{
                **base_score.__dict__,
                "slot_isolation_risk_penalty": 80,
                "slot_pair_gap_risk_penalty": 60,
                "slot_span_risk_penalty": 40,
                "slot_order_count_penalty": 60,
            }
        )
        model = ModelConfig(
            **{
                **base_model.__dict__,
                "profile_name": "feasibility",
                "allow_fallback": True,
                "allow_legacy_fallback": False,
                "enableSemanticFallback": True,
                "enableScaleDownFallback": True,
                "enableLegacyFallback": False,
                "productionAllowLegacyFallback": False,
                "validation_mode": False,
                "time_limit_seconds": 240.0,
                "global_prune_max_pairs_per_from": 6,
                "template_top_k": 60,
                "max_routes_per_slot": 7,
                "min_real_schedule_ratio": 0.70,
                "master_profile_count": 2,
                "master_seed_count": 1,
                "semantic_fallback_rounds": 2,
                "scale_down_keep_steps": (520, 320, 160),
            }
        )
        if bool(validation_mode):
            model = ModelConfig(
                **{
                    **model.__dict__,
                    "allow_fallback": False,
                    "allow_legacy_fallback": False,
                    "enableSemanticFallback": False,
                    "enableScaleDownFallback": False,
                    "enableLegacyFallback": False,
                }
            )
        return PlannerConfig(rule=base_rule, model=model, score=score)

    if profile in {"feasibility_slot_diagnostic", "feasibility_fast_slot_safe"}:
        score = ScoreConfig(
            **{
                **base_score.__dict__,
                "unassigned_real": max(100, int(base_score.unassigned_real * 0.7)),
                "slot_isolation_risk_penalty": 80,
                "slot_pair_gap_risk_penalty": 60,
                "slot_span_risk_penalty": 40,
                "slot_order_count_penalty": 120,
            }
        )
        model = ModelConfig(
            **{
                **base_model.__dict__,
                "profile_name": "feasibility_slot_diagnostic" if profile == "feasibility_slot_diagnostic" else "feasibility_fast_slot_safe",
                "allow_fallback": True,
                "allow_legacy_fallback": False,
                "enableSemanticFallback": True,
                "enableScaleDownFallback": True,
                "enableLegacyFallback": False,
                "productionAllowLegacyFallback": False,
                "validation_mode": False,
                "time_limit_seconds": 180.0,
                "global_prune_max_pairs_per_from": 6,
                "template_top_k": 60,
                "max_routes_per_slot": 7,
                "min_real_schedule_ratio": 0.70,
                "master_profile_count": 1,
                "master_seed_count": 1,
                "semantic_fallback_rounds": 1,
                "scale_down_keep_steps": (520, 320),
                "export_failed_result_for_debug": False,
                "fast_fail_on_bad_slots": True,
                "big_roll_slot_soft_order_cap": 20,
                "small_roll_slot_soft_order_cap": 18,
                "big_roll_slot_hard_order_cap": 22,
            }
        )
        if bool(validation_mode):
            model = ModelConfig(
                **{
                    **model.__dict__,
                    "allow_fallback": False,
                    "allow_legacy_fallback": False,
                    "enableSemanticFallback": False,
                    "enableScaleDownFallback": False,
                    "enableLegacyFallback": False,
                }
            )
        return PlannerConfig(rule=base_rule, model=model, score=score)

    if profile == "feasibility_search_relaxed_slots":
        score = ScoreConfig(
            **{
                **base_score.__dict__,
                "unassigned_real": max(80, int(base_score.unassigned_real * 0.6)),
                "slot_isolation_risk_penalty": 80,
                "slot_pair_gap_risk_penalty": 60,
                "slot_span_risk_penalty": 40,
                "slot_order_count_penalty": 80,
            }
        )
        model = ModelConfig(
            **{
                **base_model.__dict__,
                "profile_name": "feasibility_search_relaxed_slots",
                "allow_fallback": True,
                "allow_legacy_fallback": False,
                "enableSemanticFallback": True,
                "enableScaleDownFallback": True,
                "enableLegacyFallback": False,
                "productionAllowLegacyFallback": False,
                "validation_mode": False,
                "time_limit_seconds": 180.0,
                "global_prune_max_pairs_per_from": 6,
                "template_top_k": 60,
                "max_routes_per_slot": 7,
                "min_real_schedule_ratio": 0.55,
                "master_profile_count": 1,
                "master_seed_count": 1,
                "semantic_fallback_rounds": 1,
                "scale_down_keep_steps": (520,),
                "export_failed_result_for_debug": False,
                "export_analysis_on_failure": True,
                "export_best_candidate_analysis": True,
                "enableStructureFallback": True,
                "max_drop_ratio_for_partial": 0.05,
                "max_drop_tons_ratio_for_partial": 0.08,
                "max_drop_count_for_partial": 20,
                "min_scheduled_orders_for_partial": 100,
                "min_scheduled_tons_for_partial": 1000.0,
                "structure_fallback_slot_buffer": 4,
                "structure_fallback_risk_boost": 1.5,
                "structure_fallback_min_real_schedule_ratio": 0.55,
                "fast_fail_on_bad_slots": False,
                "big_roll_slot_soft_order_cap": 20,
                "small_roll_slot_soft_order_cap": 18,
                "big_roll_slot_hard_order_cap": 0,
                "small_roll_slot_hard_order_cap": 0,
                "max_campaign_slots": 40,
            }
        )
        if bool(validation_mode):
            model = ModelConfig(
                **{
                    **model.__dict__,
                    "allow_fallback": False,
                    "allow_legacy_fallback": False,
                    "enableSemanticFallback": False,
                    "enableScaleDownFallback": False,
                    "enableLegacyFallback": False,
                }
            )
        return PlannerConfig(rule=base_rule, model=model, score=score)

    if profile == "production_search":
        score = ScoreConfig(
            **{
                **base_score.__dict__,
                "unassigned_real": max(120, int(base_score.unassigned_real * 0.8)),
                "slot_isolation_risk_penalty": 80,
                "slot_pair_gap_risk_penalty": 60,
                "slot_span_risk_penalty": 40,
                "slot_order_count_penalty": 70,
                "virtual_bridge_penalty": 240,
                "real_bridge_penalty": 40,
                "direct_edge_penalty": 0,
            }
        )
        model = ModelConfig(
            **{
                **base_model.__dict__,
                "profile_name": "production_search",
                "allow_fallback": True,
                "allow_legacy_fallback": False,
                "enableSemanticFallback": True,
                "enableScaleDownFallback": True,
                "enableLegacyFallback": False,
                "productionAllowLegacyFallback": False,
                "validation_mode": False,
                "time_limit_seconds": 180.0,
                "global_prune_max_pairs_per_from": 6,
                "template_top_k": 60,
                "max_routes_per_slot": 7,
                "min_real_schedule_ratio": 0.6,
                "master_profile_count": 1,
                "master_seed_count": 1,
                "semantic_fallback_rounds": 1,
                "scale_down_keep_steps": (520,),
                "export_failed_result_for_debug": False,
                "export_analysis_on_failure": True,
                "enableStructureFallback": True,
                "max_drop_ratio_for_partial": 0.08,
                "max_drop_tons_ratio_for_partial": 0.10,
                "max_drop_count_for_partial": 30,
                "min_scheduled_orders_for_partial": 120,
                "min_scheduled_tons_for_partial": 1200.0,
                "structure_fallback_slot_buffer": 8,
                "structure_fallback_risk_boost": 2.0,
                "structure_fallback_min_real_schedule_ratio": 0.50,
                "fast_fail_on_bad_slots": False,
                "big_roll_slot_soft_order_cap": 20,
                "small_roll_slot_soft_order_cap": 18,
                "big_roll_slot_hard_order_cap": 0,
                "small_roll_slot_hard_order_cap": 0,
                "max_campaign_slots": 44,
            }
        )
        if bool(validation_mode):
            model = ModelConfig(
                **{
                    **model.__dict__,
                    "allow_fallback": False,
                    "allow_legacy_fallback": False,
                    "enableSemanticFallback": False,
                    "enableScaleDownFallback": False,
                    "enableLegacyFallback": False,
                }
            )
        return PlannerConfig(rule=base_rule, model=model, score=score)

    if profile == "quality":
        model = ModelConfig(
            **{
                **base_model.__dict__,
                "profile_name": "quality",
                "allow_fallback": True,
                "allow_legacy_fallback": False,
                "enableSemanticFallback": True,
                "enableScaleDownFallback": True,
                "enableLegacyFallback": False,
                "productionAllowLegacyFallback": False,
                "validation_mode": False,
                "time_limit_seconds": 210.0,
                "global_prune_max_pairs_per_from": 4,
                "template_top_k": 50,
                "max_routes_per_slot": 6,
                "min_real_schedule_ratio": 0.9,
                "master_profile_count": 2,
                "master_seed_count": 1,
                "semantic_fallback_rounds": 2,
                "scale_down_keep_steps": (520, 320, 160),
            }
        )
        if bool(validation_mode):
            model = ModelConfig(
                **{
                    **model.__dict__,
                    "allow_fallback": False,
                    "allow_legacy_fallback": False,
                    "enableSemanticFallback": False,
                    "enableScaleDownFallback": False,
                    "enableLegacyFallback": False,
                }
            )
        score = ScoreConfig(**{**base_score.__dict__, "virtual_use": max(10, int(base_score.virtual_use) * 2)})
        return PlannerConfig(rule=base_rule, model=model, score=score)

    if profile in {"feasibility_relaxed_routing", "feasibility_slot_safe"}:
        model = ModelConfig(
            **{
                **base_model.__dict__,
                "profile_name": "feasibility_slot_safe" if profile == "feasibility_slot_safe" else "feasibility_relaxed_routing",
                "allow_fallback": True,
                "allow_legacy_fallback": False,
                "enableSemanticFallback": True,
                "enableScaleDownFallback": True,
                "enableLegacyFallback": False,
                "productionAllowLegacyFallback": False,
                "validation_mode": False,
                "time_limit_seconds": 240.0,
                "global_prune_max_pairs_per_from": 6,
                "template_top_k": 60,
                "max_routes_per_slot": 7,
                "min_real_schedule_ratio": 0.70,
                "master_profile_count": 2,
                "master_seed_count": 1,
                "semantic_fallback_rounds": 2,
                "scale_down_keep_steps": (520, 320, 160),
                "big_roll_slot_soft_order_cap": 18,
                "small_roll_slot_soft_order_cap": 22,
            }
        )
        score = ScoreConfig(
            **{
                **base_score.__dict__,
                "unassigned_real": max(80, int(base_score.unassigned_real * 0.6)),
                "slot_isolation_risk_penalty": 120,
                "slot_pair_gap_risk_penalty": 90,
                "slot_span_risk_penalty": 60,
                "slot_order_count_penalty": 100,
            }
        )
        if bool(validation_mode):
            model = ModelConfig(
                **{
                    **model.__dict__,
                    "allow_fallback": False,
                    "allow_legacy_fallback": False,
                    "enableSemanticFallback": False,
                    "enableScaleDownFallback": False,
                    "enableLegacyFallback": False,
                }
            )
        return PlannerConfig(rule=base_rule, model=model, score=score)

    # -------------------------------------------------------------------------
    # Constructive LNS search profile
    # Uses the new ALNS-driven constructive path (constructive_lns_master)
    # instead of the joint_master model.
    # -------------------------------------------------------------------------
    if profile in {
        "constructive_lns_search",
        "constructive_lns_real_bridge_frontload",
        "constructive_lns_direct_only_baseline",
    }:
        real_bridge_frontload = profile == "constructive_lns_real_bridge_frontload"
        direct_only_baseline = profile == "constructive_lns_direct_only_baseline"
        profile_name = (
            "constructive_lns_real_bridge_frontload"
            if real_bridge_frontload
            else "constructive_lns_direct_only_baseline"
            if direct_only_baseline
            else "constructive_lns_search"
        )
        score = ScoreConfig(
            **{
                **base_score.__dict__,
                "slot_order_count_penalty": 0,
                "unassigned_real": 0,
                "virtual_bridge_penalty": 200,
                "real_bridge_penalty": 40,
            }
        )
        model = ModelConfig(
            **{
                **base_model.__dict__,
                "profile_name": profile_name,
                # Switch to new ALNS master path
                "main_solver_strategy": "constructive_lns",
                # LNS parameters
                "rounds": 10,
                "constructive_lns_rounds": 10,
                "lns_early_stop_no_improve_rounds": 3,
                "lns_max_total_rounds": 10,
                "lns_min_rounds_before_early_stop": 4,
                "constructive_destroy_ratio_min": 0.20,
                "constructive_destroy_ratio_max": 0.35,
                "constructive_subproblem_max_orders": 40,
                "constructive_enable_cp_sat_repair": True,
                # Isolated bridge-frontload experiment:
                # constructive_lns_search stays direct_only; the frontload
                # profile allows REAL_BRIDGE_EDGE in constructive/local inserter
                # while VIRTUAL_BRIDGE_EDGE and bridge expansion stay disabled.
                "allow_virtual_bridge_edge_in_constructive": False,
                "allow_real_bridge_edge_in_constructive": bool(real_bridge_frontload),
                "bridge_expansion_mode": "disabled",
                "repair_only_real_bridge_enabled": True,
                "repair_only_virtual_bridge_enabled": False,
                "repair_only_virtual_bridge_pilot_enabled": False if (real_bridge_frontload or direct_only_baseline) else True,
                "virtual_bridge_pilot_max_blocks_per_run": 15,
                "virtual_bridge_pilot_max_per_block": 1,
                "virtual_bridge_pilot_max_virtual_tons": 30.0,
                "virtual_bridge_pilot_penalty": 1000000.0,
                "virtual_bridge_pilot_only_when_endpoint_class": ["HAS_ENDPOINT_EDGE", "BAND_TOO_NARROW"],
                "virtual_bridge_pilot_only_when_dominant_fail": ["THICKNESS_RULE_FAIL", "WIDTH_RULE_FAIL", "GROUP_SWITCH_FAIL", "MULTI_RULE_FAIL"],
                "repair_only_bridge_max_per_segment": 1,
                "repair_only_bridge_cost_penalty": 100000.0,
                "repair_bridge_left_band_k": 3,
                "repair_bridge_right_band_k": 3,
                "repair_bridge_band_max_pairs_per_split": 9,
                "repair_bridge_left_trim_max": 2,
                "repair_bridge_right_trim_max": 2,
                "repair_bridge_endpoint_adjustment_limit_per_split": 9,
                "repair_bridge_adjustment_enable_left_trim": True,
                "repair_bridge_adjustment_enable_right_trim": True,
                "repair_bridge_adjustment_enable_swap": False,
                "repair_bridge_ton_rescue_max_neighbor_blocks": 6,
                "repair_bridge_ton_rescue_enable_backward": True,
                "repair_bridge_ton_rescue_enable_forward": True,
                "repair_bridge_ton_rescue_enable_bidirectional": True,
                "repair_bridge_ton_rescue_max_orders_per_window": 50,
                "repair_bridge_ton_rescue_max_failed_windows_after_min": 2,
                # Strict template enforcement (no illegal edge penalty)
                "strict_template_edges": True,
                # Disable fallbacks that conflict with LNS path
                "allow_fallback": False,
                "allow_legacy_fallback": False,
                "enableSemanticFallback": False,
                "enableScaleDownFallback": False,
                "enableStructureFallback": False,
                "enableLegacyFallback": False,
                # Production flags
                "validation_mode": bool(validation_mode),
                "time_limit_seconds": 60.0,
                "min_real_schedule_ratio": 0.90,
                "master_profile_count": 1,
                "master_seed_count": 1,
                "max_campaign_slots": 14,
                "min_campaign_slots": 6,
                "big_roll_slot_soft_order_cap": 20,
                "small_roll_slot_soft_order_cap": 24,
                "export_failed_result_for_debug": True,
                # ---- Tail rebalancing: rescue underfilled tail by pullback ----
                "tail_rebalance_enabled": True,
                "tail_rebalance_max_pullback_orders": 8,
                "tail_rebalance_max_pullback_tons10": 2500,
                "tail_rebalance_accept_if_prev_stays_above_min": True,
                # ---- Tail Repair Budget: limit search scope ----
                "max_tail_repair_windows_per_line": 12,
                "max_tail_repair_windows_total": 24,
                "max_recut_cutpoints_per_window": 12,
                "max_fill_candidates_per_tail": 20,
                "tail_repair_gap_to_min_limit": 220.0,
                # ---- Tail Fill From Dropped: make fill branch actually trigger ----
                "tail_fill_from_dropped_enabled": True,
                "tail_fill_gap_to_min_limit": 220.0,
                "tail_fill_accept_partial_progress": True,
                "tail_fill_max_inserts_per_tail": 2,
                "tail_fill_second_pass_gap_limit": 30.0,
                # ---- Small roll dual-order reserve: prevent big_roll monopoly ----
                "small_roll_dual_reserve_enabled": True,
                "small_roll_dual_reserve_penalty": 15,
                # ---- Small roll dual-order reserve bucket: hard-prevent big_roll from taking top dual candidates ----
                "small_roll_dual_reserve_bucket_enabled": True,
                "small_roll_dual_reserve_bucket_ratio": 0.45,
                "small_roll_dual_reserve_bucket_max_orders": 120,
                # ---- Small roll seed-first: prioritize small_roll chain building before big_roll consumes reserve ----
                "small_roll_seed_first_enabled": True,
                "small_roll_seed_min_orders": 20,
                "small_roll_seed_min_tons10": 5000,
                # ---- Small roll dual-order reserve QUOTA: balanced allocation instead of "lock all" ----
                # Min = guarantee, Max = ceiling; once max reached, remaining quota released to big_roll.
                "small_roll_dual_reserve_quota_enabled": True,
                "small_roll_dual_reserve_quota_min_orders": 25,
                "small_roll_dual_reserve_quota_min_tons10": 6000,
                "small_roll_dual_reserve_quota_max_orders": 60,
                "small_roll_dual_reserve_quota_max_tons10": 14000,
                # Release remaining quota to big_roll after small_roll seed phase completes
                "big_roll_dual_release_after_small_seed": True,
            }
        )
        return PlannerConfig(rule=base_rule, model=model, score=score)

    # -------------------------------------------------------------------------
    # Constructive LNS debug acceptance profile (validation-only)
    # Same as constructive_lns_search but with RELAXED partial acceptance
    # thresholds to confirm whether official_exported=True is blocked ONLY by
    # soft threshold (partial acceptance), not by routing/hard constraints.
    # This profile is NOT for production use.
    # -------------------------------------------------------------------------
    if profile == "constructive_lns_debug_acceptance":
        print(
            "[APS][debug_acceptance] using relaxed partial acceptance thresholds for validation only"
        )
        score = ScoreConfig(
            **{
                **base_score.__dict__,
                "slot_order_count_penalty": 0,
                "unassigned_real": 0,
                "virtual_bridge_penalty": 200,
                "real_bridge_penalty": 40,
            }
        )
        model = ModelConfig(
            **{
                **base_model.__dict__,
                "profile_name": "constructive_lns_debug_acceptance",
                # Same ALNS master path as constructive_lns_search
                "main_solver_strategy": "constructive_lns",
                # LNS parameters
                "rounds": 10,
                "constructive_lns_rounds": 10,
                "lns_early_stop_no_improve_rounds": 3,
                "lns_max_total_rounds": 10,
                "lns_min_rounds_before_early_stop": 4,
                "constructive_destroy_ratio_min": 0.20,
                "constructive_destroy_ratio_max": 0.35,
                "constructive_subproblem_max_orders": 40,
                "constructive_enable_cp_sat_repair": True,
                # Same direct_only mode as constructive_lns_search
                "allow_virtual_bridge_edge_in_constructive": False,
                "allow_real_bridge_edge_in_constructive": False,
                "bridge_expansion_mode": "disabled",
                "repair_only_real_bridge_enabled": True,
                "repair_only_virtual_bridge_enabled": False,
                "repair_only_virtual_bridge_pilot_enabled": True,
                "virtual_bridge_pilot_max_blocks_per_run": 15,
                "virtual_bridge_pilot_max_per_block": 1,
                "virtual_bridge_pilot_max_virtual_tons": 30.0,
                "virtual_bridge_pilot_penalty": 1000000.0,
                "virtual_bridge_pilot_only_when_endpoint_class": ["HAS_ENDPOINT_EDGE", "BAND_TOO_NARROW"],
                "virtual_bridge_pilot_only_when_dominant_fail": ["THICKNESS_RULE_FAIL", "WIDTH_RULE_FAIL", "GROUP_SWITCH_FAIL", "MULTI_RULE_FAIL"],
                "repair_only_bridge_max_per_segment": 1,
                "repair_only_bridge_cost_penalty": 100000.0,
                "repair_bridge_left_band_k": 3,
                "repair_bridge_right_band_k": 3,
                "repair_bridge_band_max_pairs_per_split": 9,
                "repair_bridge_left_trim_max": 2,
                "repair_bridge_right_trim_max": 2,
                "repair_bridge_endpoint_adjustment_limit_per_split": 9,
                "repair_bridge_adjustment_enable_left_trim": True,
                "repair_bridge_adjustment_enable_right_trim": True,
                "repair_bridge_adjustment_enable_swap": False,
                "repair_bridge_ton_rescue_max_neighbor_blocks": 6,
                "repair_bridge_ton_rescue_enable_backward": True,
                "repair_bridge_ton_rescue_enable_forward": True,
                "repair_bridge_ton_rescue_enable_bidirectional": True,
                "repair_bridge_ton_rescue_max_orders_per_window": 50,
                "repair_bridge_ton_rescue_max_failed_windows_after_min": 2,
                "strict_template_edges": True,
                "allow_fallback": False,
                "allow_legacy_fallback": False,
                "enableSemanticFallback": False,
                "enableScaleDownFallback": False,
                "enableStructureFallback": False,
                "enableLegacyFallback": False,
                # Production flags
                "validation_mode": bool(validation_mode),
                "time_limit_seconds": 60.0,
                "min_real_schedule_ratio": 0.90,
                "master_profile_count": 1,
                "master_seed_count": 1,
                "max_campaign_slots": 14,
                "min_campaign_slots": 6,
                "big_roll_slot_soft_order_cap": 20,
                "small_roll_slot_soft_order_cap": 24,
                "export_failed_result_for_debug": True,
                # Tail rebalancing (same as constructive_lns_search)
                "tail_rebalance_enabled": True,
                "tail_rebalance_max_pullback_orders": 8,
                "tail_rebalance_max_pullback_tons10": 2500,
                "tail_rebalance_accept_if_prev_stays_above_min": True,
                # ---- Tail Repair Budget: limit search scope ----
                "max_tail_repair_windows_per_line": 12,
                "max_tail_repair_windows_total": 24,
                "max_recut_cutpoints_per_window": 12,
                "max_fill_candidates_per_tail": 20,
                "tail_repair_gap_to_min_limit": 220.0,
                # ---- Tail Fill From Dropped: make fill branch actually trigger ----
                "tail_fill_from_dropped_enabled": True,
                "tail_fill_gap_to_min_limit": 220.0,
                "tail_fill_accept_partial_progress": True,
                "tail_fill_max_inserts_per_tail": 2,
                "tail_fill_second_pass_gap_limit": 30.0,
                # Small roll dual-order reserve (same as constructive_lns_search)
                "small_roll_dual_reserve_enabled": True,
                "small_roll_dual_reserve_penalty": 15,
                "small_roll_dual_reserve_bucket_enabled": True,
                "small_roll_dual_reserve_bucket_ratio": 0.45,
                "small_roll_dual_reserve_bucket_max_orders": 120,
                # ---- Small roll seed-first: prioritize small_roll chain building before big_roll consumes reserve ----
                "small_roll_seed_first_enabled": True,
                "small_roll_seed_min_orders": 20,
                "small_roll_seed_min_tons10": 5000,
                # ---- Small roll dual-order reserve QUOTA: balanced allocation (same as constructive_lns_search) ----
                "small_roll_dual_reserve_quota_enabled": True,
                "small_roll_dual_reserve_quota_min_orders": 25,
                "small_roll_dual_reserve_quota_min_tons10": 6000,
                "small_roll_dual_reserve_quota_max_orders": 60,
                "small_roll_dual_reserve_quota_max_tons10": 14000,
                "big_roll_dual_release_after_small_seed": True,
                # ---- RELAXED partial acceptance thresholds (validation only) ----
                # These thresholds are intentionally very loose to allow official_exported=True
                # when routing_feasible=True and hard violations=0.
                # The goal is to confirm the ONLY remaining blocker is the soft threshold itself.
                "max_drop_ratio_for_partial": 0.90,        # original: 0.05 (now: 90% drop allowed)
                "max_drop_tons_ratio_for_partial": 0.90,  # original: 0.08
                "max_drop_count_for_partial": 1000,       # original: 20
                "min_scheduled_orders_for_partial": 50,   # original: 100 (relaxed to 50)
                "min_scheduled_tons_for_partial": 500.0,  # original: 1000.0 (relaxed to 500)
            }
        )
        return PlannerConfig(rule=base_rule, model=model, score=score)

    # -------------------------------------------------------------------------
    # Constructive LNS bridge family master profile (future / oracle-ready)
    # -------------------------------------------------------------------------
    # This profile is a FUTURE placeholder that allows VIRTUAL_BRIDGE_FAMILY_EDGE
    # to enter the Candidate Graph / Master as a controlled candidate.
    #
    # Key differences from existing profiles:
    #   - allow_virtual_bridge_edge_in_constructive = True  (for family edge)
    #   - bridge_expansion_mode = "disabled"  (no exact path expansion yet)
    #   - Legacy virtual pilot remains available but marked as "legacy repair path"
    #
    # NOTE: This profile is NOT enabled by default. It is a占位 for the
    # full oracle-driven realization pipeline that comes in the next iteration.
    # -------------------------------------------------------------------------
    if profile == "constructive_lns_bridge_family_master":
        print(
            "[APS][bridge_family_master] CONSTRUCTIVE_LNS_BRIDGE_FAMILY_MASTER profile: "
            "VIRTUAL_BRIDGE_FAMILY_EDGE enters as controlled candidate; "
            "exact path realization via BridgeRealizationOracle is stub (NOT_IMPLEMENTED_YET). "
            "bridge_expansion_mode=disabled, legacy virtual pilot available."
        )
        score = ScoreConfig(
            **{
                **base_score.__dict__,
                "slot_order_count_penalty": 0,
                "unassigned_real": 0,
                "virtual_bridge_penalty": 200,
                "real_bridge_penalty": 40,
            }
        )
        model = ModelConfig(
            **{
                **base_model.__dict__,
                "profile_name": "constructive_lns_bridge_family_master",
                # Switch to new ALNS master path
                "main_solver_strategy": "constructive_lns",
                # LNS parameters
                "rounds": 10,
                "constructive_lns_rounds": 10,
                "lns_early_stop_no_improve_rounds": 3,
                "lns_max_total_rounds": 10,
                "lns_min_rounds_before_early_stop": 4,
                "constructive_destroy_ratio_min": 0.20,
                "constructive_destroy_ratio_max": 0.35,
                "constructive_subproblem_max_orders": 40,
                "constructive_enable_cp_sat_repair": True,
                # ---- Bridge Family Master: allow family edge, no expansion ----
                # allow_virtual = True → VIRTUAL_BRIDGE_FAMILY_EDGE enters graph
                # bridge_expansion disabled → no exact path expansion (oracle stub)
                "allow_virtual_bridge_edge_in_constructive": True,
                "allow_real_bridge_edge_in_constructive": True,
                "bridge_expansion_mode": "disabled",
                "repair_only_real_bridge_enabled": True,
                # Legacy virtual pilot stays available (marked as legacy repair path)
                "repair_only_virtual_bridge_enabled": True,
                "repair_only_virtual_bridge_pilot_enabled": True,
                "virtual_bridge_pilot_max_blocks_per_run": 15,
                "virtual_bridge_pilot_max_per_block": 1,
                "virtual_bridge_pilot_max_virtual_tons": 30.0,
                "virtual_bridge_pilot_penalty": 1000000.0,
                "virtual_bridge_pilot_only_when_endpoint_class": ["HAS_ENDPOINT_EDGE", "BAND_TOO_NARROW"],
                "virtual_bridge_pilot_only_when_dominant_fail": ["THICKNESS_RULE_FAIL", "WIDTH_RULE_FAIL", "GROUP_SWITCH_FAIL", "MULTI_RULE_FAIL"],
                "repair_only_bridge_max_per_segment": 1,
                "repair_only_bridge_cost_penalty": 100000.0,
                "repair_bridge_left_band_k": 3,
                "repair_bridge_right_band_k": 3,
                "repair_bridge_band_max_pairs_per_split": 9,
                "repair_bridge_left_trim_max": 2,
                "repair_bridge_right_trim_max": 2,
                "repair_bridge_endpoint_adjustment_limit_per_split": 9,
                "repair_bridge_adjustment_enable_left_trim": True,
                "repair_bridge_adjustment_enable_right_trim": True,
                "repair_bridge_adjustment_enable_swap": False,
                "repair_bridge_ton_rescue_max_neighbor_blocks": 6,
                "repair_bridge_ton_rescue_enable_backward": True,
                "repair_bridge_ton_rescue_enable_forward": True,
                "repair_bridge_ton_rescue_enable_bidirectional": True,
                "repair_bridge_ton_rescue_max_orders_per_window": 50,
                "repair_bridge_ton_rescue_max_failed_windows_after_min": 2,
                "strict_template_edges": True,
                "allow_fallback": False,
                "allow_legacy_fallback": False,
                "enableSemanticFallback": False,
                "enableScaleDownFallback": False,
                "enableStructureFallback": False,
                "enableLegacyFallback": False,
                "validation_mode": bool(validation_mode),
                "time_limit_seconds": 60.0,
                "min_real_schedule_ratio": 0.90,
                "master_profile_count": 1,
                "master_seed_count": 1,
                "max_campaign_slots": 14,
                "min_campaign_slots": 6,
                "big_roll_slot_soft_order_cap": 20,
                "small_roll_slot_soft_order_cap": 24,
                "export_failed_result_for_debug": True,
                # Tail rebalancing
                "tail_rebalance_enabled": True,
                "tail_rebalance_max_pullback_orders": 8,
                "tail_rebalance_max_pullback_tons10": 2500,
                "tail_rebalance_accept_if_prev_stays_above_min": True,
                # Tail Repair Budget
                "max_tail_repair_windows_per_line": 12,
                "max_tail_repair_windows_total": 24,
                "max_recut_cutpoints_per_window": 12,
                "max_fill_candidates_per_tail": 20,
                "tail_repair_gap_to_min_limit": 220.0,
                # Tail Fill From Dropped
                "tail_fill_from_dropped_enabled": True,
                "tail_fill_gap_to_min_limit": 220.0,
                "tail_fill_accept_partial_progress": True,
                "tail_fill_max_inserts_per_tail": 2,
                "tail_fill_second_pass_gap_limit": 30.0,
                # Small roll dual-order reserve
                "small_roll_dual_reserve_enabled": True,
                "small_roll_dual_reserve_penalty": 15,
                "small_roll_dual_reserve_bucket_enabled": True,
                "small_roll_dual_reserve_bucket_ratio": 0.45,
                "small_roll_dual_reserve_bucket_max_orders": 120,
                # Small roll seed-first
                "small_roll_seed_first_enabled": True,
                "small_roll_seed_min_orders": 20,
                "small_roll_seed_min_tons10": 5000,
                # Small roll dual-order reserve QUOTA
                "small_roll_dual_reserve_quota_enabled": True,
                "small_roll_dual_reserve_quota_min_orders": 25,
                "small_roll_dual_reserve_quota_min_tons10": 6000,
                "small_roll_dual_reserve_quota_max_orders": 60,
                "small_roll_dual_reserve_quota_max_tons10": 14000,
                "big_roll_dual_release_after_small_seed": True,
            }
        )
        return PlannerConfig(rule=base_rule, model=model, score=score)

    model = base_model
    if bool(validation_mode):
        model = ModelConfig(
            **{
                **model.__dict__,
                "allow_fallback": False,
                "allow_legacy_fallback": False,
                "enableLegacyFallback": False,
            }
        )
    return PlannerConfig(rule=base_rule, model=model, score=base_score)


def normalize_enforced_profile_name(profile_name: str | None) -> str:
    """
    Normalize profile name to the current enforced default.

    Rules:
        - None / "" / "default" -> "constructive_lns_search"
        - "constructive_lns_search" -> "constructive_lns_search"
        - Other values are returned as-is (to allow explicit runtime errors)
    """
    if profile_name is None:
        return "constructive_lns_search"
    name = str(profile_name).strip().lower()
    if name in ("", "default"):
        return "constructive_lns_search"
    return str(profile_name)


def build_default_solve_config(
    validation_mode: bool = True,
    production_compatibility_mode: bool = False,
) -> PlannerConfig:
    """
    统一参数入口：当前工程默认且唯一允许的主试验 profile 为 constructive_lns_search。
    其他 profile 值会在运行时被 master.py / cold_rolling_pipeline.py 的守卫拒绝。
    """
    return build_profile_config(
        "constructive_lns_search",
        validation_mode=validation_mode,
        production_compatibility_mode=production_compatibility_mode,
    )
