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


def build_default_solve_config(
    validation_mode: bool = True,
    production_compatibility_mode: bool = False,
) -> PlannerConfig:
    """
    统一参数入口：新架构默认从这里构造 default profile。
    """
    return build_profile_config(
        "default",
        validation_mode=validation_mode,
        production_compatibility_mode=production_compatibility_mode,
    )
