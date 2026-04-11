import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from aps_cp_sat.config import PlannerConfig, build_default_solve_config, build_profile_config


def test_default_tiered_objective_enabled():
    cfg = PlannerConfig()
    assert cfg.model.enable_tiered_objective is True
    assert cfg.model.allow_fallback is False
    assert cfg.model.enableSemanticFallback is False
    assert cfg.model.enableScaleDownFallback is False
    assert cfg.model.enableLegacyFallback is False
    assert cfg.model.max_virtual_chain == 5
    assert cfg.rule.max_virtual_chain == 5
    assert cfg.rule.max_logical_reverse_per_campaign == 5
    assert cfg.rule.virtual_temp_min == 600.0
    assert cfg.rule.virtual_temp_max == 900.0


def test_validation_mode_disables_fallback():
    cfg = build_default_solve_config(validation_mode=True, production_compatibility_mode=False)
    assert cfg.allow_fallback is False
    assert cfg.allow_legacy_fallback is False
    assert cfg.enableLegacyFallback is False


def test_feasibility_profile_shallower_fallback_waterfall():
    cfg = build_profile_config("feasibility", validation_mode=False, production_compatibility_mode=False)
    assert cfg.model.master_profile_count == 2
    assert cfg.model.master_seed_count == 1
    assert cfg.model.semantic_fallback_rounds == 2
    assert tuple(cfg.model.scale_down_keep_steps) == (520, 320, 160)
    assert cfg.model.min_real_schedule_ratio == 0.70


def test_feasibility_relaxed_routing_profile_reduces_assignment_pressure():
    base = build_profile_config("feasibility", validation_mode=False, production_compatibility_mode=False)
    relaxed = build_profile_config("feasibility_relaxed_routing", validation_mode=False, production_compatibility_mode=False)
    assert relaxed.model.profile_name == "feasibility_relaxed_routing"
    assert relaxed.score.unassigned_real < base.score.unassigned_real
    assert relaxed.score.slot_isolation_risk_penalty >= base.score.slot_isolation_risk_penalty
    assert relaxed.score.slot_pair_gap_risk_penalty >= base.score.slot_pair_gap_risk_penalty
    assert relaxed.score.slot_span_risk_penalty >= base.score.slot_span_risk_penalty


def test_feasibility_slot_safe_profile_strengthens_route_risk_and_slot_cap():
    base = build_profile_config("feasibility", validation_mode=False, production_compatibility_mode=False)
    safe = build_profile_config("feasibility_slot_safe", validation_mode=False, production_compatibility_mode=False)
    assert safe.model.profile_name == "feasibility_slot_safe"
    assert safe.score.unassigned_real < base.score.unassigned_real
    assert safe.score.slot_isolation_risk_penalty > base.score.slot_isolation_risk_penalty
    assert safe.score.slot_pair_gap_risk_penalty > base.score.slot_pair_gap_risk_penalty
    assert safe.score.slot_span_risk_penalty > base.score.slot_span_risk_penalty
    assert safe.score.slot_order_count_penalty > base.score.slot_order_count_penalty
    assert safe.model.big_roll_slot_soft_order_cap < base.model.big_roll_slot_soft_order_cap


def test_feasibility_fast_slot_safe_profile_limits_waterfall_and_export():
    fast = build_profile_config("feasibility_fast_slot_safe", validation_mode=False, production_compatibility_mode=False)
    assert fast.model.profile_name == "feasibility_fast_slot_safe"
    assert fast.model.master_profile_count == 1
    assert fast.model.master_seed_count == 1
    assert fast.model.semantic_fallback_rounds == 1
    assert tuple(fast.model.scale_down_keep_steps) == (520, 320)
    assert fast.model.export_failed_result_for_debug is False
    assert fast.model.fast_fail_on_bad_slots is True
    assert fast.model.big_roll_slot_soft_order_cap == 20
    assert fast.model.small_roll_slot_soft_order_cap == 18
    assert fast.model.big_roll_slot_hard_order_cap == 22
    assert fast.model.small_roll_slot_hard_order_cap == 0


def test_feasibility_slot_diagnostic_profile_keeps_tight_diagnostic_semantics():
    diagnostic = build_profile_config("feasibility_slot_diagnostic", validation_mode=False, production_compatibility_mode=False)
    assert diagnostic.model.profile_name == "feasibility_slot_diagnostic"
    assert diagnostic.model.fast_fail_on_bad_slots is True
    assert diagnostic.model.big_roll_slot_hard_order_cap == 22
    assert diagnostic.model.max_campaign_slots == 14


def test_feasibility_search_relaxed_slots_profile_opens_search_space():
    search = build_profile_config("feasibility_search_relaxed_slots", validation_mode=False, production_compatibility_mode=False)
    assert search.model.profile_name == "feasibility_search_relaxed_slots"
    assert search.model.fast_fail_on_bad_slots is False
    assert search.model.big_roll_slot_hard_order_cap == 0
    assert search.model.small_roll_slot_hard_order_cap == 0
    assert search.model.max_campaign_slots == 40
    assert search.model.min_real_schedule_ratio == 0.55
    assert search.model.semantic_fallback_rounds == 1
    assert tuple(search.model.scale_down_keep_steps) == (520,)


def test_production_search_profile_is_default_production_mode():
    prod = build_profile_config("production_search", validation_mode=False, production_compatibility_mode=False)
    assert prod.model.profile_name == "production_search"
    assert prod.model.allow_fallback is True
    assert prod.model.allow_legacy_fallback is False
    assert prod.model.fast_fail_on_bad_slots is False
    assert prod.model.export_failed_result_for_debug is False
    assert prod.model.export_analysis_on_failure is True
    assert prod.model.enableStructureFallback is True
    assert prod.model.max_drop_ratio_for_partial == 0.08
    assert prod.model.max_drop_tons_ratio_for_partial == 0.10
    assert prod.model.max_drop_count_for_partial == 30
    assert prod.model.min_scheduled_orders_for_partial == 120
    assert prod.model.min_scheduled_tons_for_partial == 1200.0
    assert prod.model.max_campaign_slots == 44
    assert prod.model.min_real_schedule_ratio == 0.6
    assert prod.model.structure_fallback_slot_buffer == 8
    assert prod.model.structure_fallback_risk_boost == 2.0
    assert prod.model.structure_fallback_min_real_schedule_ratio == 0.50
