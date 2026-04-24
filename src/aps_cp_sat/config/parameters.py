"""参数档位构建器。

本文件负责把基础 RuleConfig / ModelConfig / ScoreConfig 组合成具体 profile。

阅读和调参规则：
1. 所有算法参数的字段级中文注释在对应 dataclass 中维护：
   - RuleConfig：工艺硬规则和虚拟卷规格口径。
   - ModelConfig：求解路线、模板图、LNS、block-first、桥接、虚拟影子模式等流程参数。
   - ScoreConfig：目标函数、掉单、桥接、虚拟使用、平滑性等评分权重。
2. 本文件中的 profile 只负责覆盖这些基础参数，不重新定义参数语义。
3. 如果新增 profile 覆盖项，应先在对应 dataclass 中补字段和中文注释，再在这里覆盖取值。
4. 当前正式主路径只允许 constructive_lns / constructive_lns_virtual_guarded_frontload。
5. 不在 profile 层绕过硬约束；硬约束口径统一以 RuleConfig 和终态审计为准。
"""

from __future__ import annotations

from aps_cp_sat.config.model_config import ModelConfig
from aps_cp_sat.config.planner_config import PlannerConfig
from aps_cp_sat.config.rule_config import RuleConfig
from aps_cp_sat.config.score_config import ScoreConfig


def _log_model_overrides(base_model: ModelConfig, overrides: dict, source: str) -> None:
    default_model = ModelConfig()
    for key, profile_value in overrides.items():
        if not hasattr(base_model, key):
            continue
        base_value = getattr(base_model, key)
        if base_value == profile_value:
            continue
        default_value = getattr(default_model, key, base_value)
        print(
            f"[APS][CONFIG_OVERRIDE] key={key}, default={default_value!r}, "
            f"profile_value={profile_value!r}, source={source}"
        )


def build_profile_config(
    profile_name: str = "default",
    *,
    validation_mode: bool = False,
    production_compatibility_mode: bool = False,
) -> PlannerConfig:
    profile = str(profile_name or "default").lower()
    if profile in {"", "default"}:
        profile = "constructive_lns_virtual_guarded_frontload"
    if profile != "constructive_lns_virtual_guarded_frontload":
        raise ValueError(
            f"[APS][PROFILE_GUARD][ONLY_SINGLE_ROUTE_ALLOWED] "
            "expected strategy=constructive_lns, "
            "expected profile=constructive_lns_virtual_guarded_frontload, "
            f"got profile={profile_name!r}"
        )
    base_rule = RuleConfig()
    base_score = ScoreConfig()
    base_model = ModelConfig(
        profile_name=profile,
        max_orders=10_000_000,
        rounds=6,  # 通用默认轮数：比旧值略高，兼顾稳定性与调参空间
        time_limit_seconds=180.0,  # 通用默认时限：给模板/局部修复留足工程时间
        max_virtual_chain=5,
        global_prune_max_pairs_per_from=4,  # 默认仍保持保守剪枝，避免基础 profile 直接炸图
        max_routes_per_slot=6,  # 适度放宽槽位路由候选，减少过早无路可走
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
    # Constructive LNS search profile (mainline real-bridge-frontload)
    # Uses the new ALNS-driven constructive path (constructive_lns_master)
    # instead of the joint_master model.
    #
    # Profile semantics:
    #   - constructive_lns_search: mainline real-bridge-frontload (Route RB)
    #       allow_real_bridge_edge_in_constructive = True
    #       allow_virtual_bridge_edge_in_constructive = False
    #       bridge_expansion_mode = "disabled"
    #   - constructive_lns_real_bridge_frontload: alias of constructive_lns_search
    #   - constructive_lns_direct_only_baseline: regression baseline (Route C)
    #       allow_real_bridge_edge_in_constructive = False
    #       allow_virtual_bridge_edge_in_constructive = False
    # -------------------------------------------------------------------------
    if profile in {
        "constructive_lns_search",
        "constructive_lns_real_bridge_frontload",
        "constructive_lns_direct_only_baseline",
    }:
        is_direct_only_baseline = profile == "constructive_lns_direct_only_baseline"
        # Mainline uses direct_plus_real_bridge; baseline uses direct_only
        is_mainline = not is_direct_only_baseline
        profile_name = (
            "constructive_lns_search"
            if profile == "constructive_lns_search"
            else "constructive_lns_real_bridge_frontload"
            if profile == "constructive_lns_real_bridge_frontload"
            else "constructive_lns_direct_only_baseline"
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
                "rounds": 12,  # 主线默认 12 轮局部搜索，给 repair 更多机会
                "constructive_lns_rounds": 12,  # ALNS 总轮数：从 10 提到 12，仍属可控范围
                "lns_early_stop_no_improve_rounds": 4,  # 连续 4 轮无改进再停，减少过早停机
                "lns_max_total_rounds": 12,  # 与 constructive_lns_rounds 对齐
                "lns_min_rounds_before_early_stop": 5,  # 至少跑 5 轮再允许早停
                "constructive_destroy_ratio_min": 0.18,  # 下界略降，减小对好链的过度破坏
                "constructive_destroy_ratio_max": 0.30,  # 上界略降，控制修补子问题规模
                "constructive_subproblem_max_orders": 48,  # 本地 CP-SAT 子问题略放宽，增强局部修复能力
                "constructive_enable_cp_sat_repair": True,
                # ---- Mainline (Route RB): allow REAL_BRIDGE_EDGE in constructive ----
                # allow_real_bridge_edge_in_constructive = True for mainline
                # allow_virtual_bridge_edge_in_constructive = False (virtual family not in mainline)
                # bridge_expansion_mode = "disabled" (virtual expansion not in mainline)
                "allow_virtual_bridge_edge_in_constructive": False,
                "allow_real_bridge_edge_in_constructive": bool(is_mainline),  # True for mainline, False for baseline
                "bridge_expansion_mode": "disabled",
                "repair_only_real_bridge_enabled": True,
                "repair_only_virtual_bridge_enabled": False,
                "repair_only_virtual_bridge_pilot_enabled": False,
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
                "time_limit_seconds": 90.0,  # 主线默认 90 秒，更接近工程验证时长
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
                "max_tail_repair_windows_per_line": 16,
                "max_tail_repair_windows_total": 32,
                "max_recut_cutpoints_per_window": 12,
                "max_fill_candidates_per_tail": 30,
                "tail_repair_gap_to_min_limit": 320.0,
                # ---- Tail Fill From Dropped: make fill branch actually trigger ----
                "tail_fill_from_dropped_enabled": True,
                "tail_fill_gap_to_min_limit": 320.0,
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
    # Constructive LNS guarded virtual family frontload (受控 virtual 前移实验线)
    #
    # Profile semantics:
    #   constructive_lns_virtual_guarded_frontload:
    #       allow_real_bridge_edge_in_constructive = True
    #       allow_virtual_bridge_edge_in_constructive = True (only VIRTUAL_BRIDGE_FAMILY_EDGE)
    #       bridge_expansion_mode = "disabled"
    #       virtual_family_frontload_enabled = True
    #       virtual_family_budget_per_line = 3
    #       virtual_family_budget_per_segment = 1
    #
    # Key principle: "virtual as a frontload CAPABILITY, not exact path expansion"
    #   - Legacy VIRTUAL_BRIDGE_EDGE: STILL BLOCKED (bridge_expansion_mode = disabled)
    #   - VIRTUAL_BRIDGE_FAMILY_EDGE: ALLOWED but STRONGLY constrained
    #       - Only eligible families: WIDTH_GROUP, THICKNESS, GROUP_TRANSITION
    #       - Global top-k: 2 per (from_order, line, family)
    #       - Global cap: 300 total
    #       - Budget: 3 per line, 1 per segment in greedy
    #       - ALNS repair: extra_topk = 4 (more aggressive than global constructive)
    #       - bridge_expansion_mode: still disabled (no exact spec chain expansion)
    # -------------------------------------------------------------------------
    if profile == "constructive_lns_virtual_guarded_frontload":
        score = ScoreConfig(
            **{
                **base_score.__dict__,
                "slot_order_count_penalty": 0,
                "unassigned_real": 0,
                "virtual_bridge_penalty": 200,
                "real_bridge_penalty": 40,
            }
        )
        model_overrides = {
                "profile_name": "constructive_lns_virtual_guarded_frontload",
                "main_solver_strategy": "constructive_lns",
                # LNS parameters
                "rounds": 16,
                "constructive_lns_rounds": 16,
                "lns_early_stop_no_improve_rounds": 5,
                "lns_max_total_rounds": 16,
                "lns_min_rounds_before_early_stop": 6,
                "constructive_destroy_ratio_min": 0.20,
                "constructive_destroy_ratio_max": 0.35,
                "constructive_subproblem_max_orders": 56,
                "local_cpsat_max_orders": 56,
                "constructive_local_cpsat_time_limit_seconds": 8.0,
                "constructive_enable_cp_sat_repair": True,
                # ---- Guarded virtual family frontload (受控前移实验线) ----
                # allow_virtual = True means VIRTUAL_BRIDGE_FAMILY_EDGE is allowed (not legacy)
                # allow_real = True (same as mainline Route RB)
                "allow_virtual_bridge_edge_in_constructive": True,
                "allow_real_bridge_edge_in_constructive": True,
                "virtual_bridge_mode": "prebuilt_virtual_inventory",
                "prebuilt_virtual_inventory_enabled": True,
                "prebuilt_virtual_count_per_spec": 5,
                "bridge_expansion_mode": "disabled",  # Still disabled: no virtual exact expansion
                "repair_only_real_bridge_enabled": True,
                "repair_only_virtual_bridge_enabled": False,
                "repair_only_virtual_bridge_pilot_enabled": False,
                # ---- Virtual family frontload configuration ----
                # Dual-pool: global pool = strict (for greedy constructive),
                #            repair pool = slightly wider (for ALNS repair + local rebuild).
                "virtual_family_frontload_enabled": True,
                "virtual_family_frontload_global_topk_per_from": 3,     # 2 → 3: stronger frontload
                "virtual_family_frontload_global_max_edges_total": 360, # 300 → 360: more candidates
                "virtual_family_frontload_repair_max_edges_total": 900, # NEW: wider repair pool cap
                "virtual_family_frontload_allowed_families": ["WIDTH_GROUP", "THICKNESS", "GROUP_TRANSITION"],
                "virtual_family_frontload_max_bridge_count": 2,
                "virtual_family_frontload_only_when_underfill_or_drop_pressure": True,
                "virtual_family_frontload_min_block_tons": 80.0,
                "virtual_family_frontload_max_block_tons": 450.0,
                "virtual_family_frontload_alns_only_extra_topk": 4,
                "virtual_family_frontload_local_cpsat_only": False,
                "virtual_family_frontload_global_penalty": 100.0,  # 120 → 100: slightly more aggressive in greedy
                "virtual_family_frontload_local_penalty": 70.0,   # 80 → 70: slightly more favorable in CP-SAT
                "virtual_family_frontload_require_family_budget": True,
                "virtual_family_budget_per_line": 4,    # 3 → 4: more per-line budget
                "virtual_family_budget_per_segment": 2, # 1 → 2: more per-segment budget
                # Virtual bridge pilot (still disabled for this profile)
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
                "tail_rebalance_enabled": True,
                "tail_rebalance_max_pullback_orders": 8,
                "tail_rebalance_max_pullback_tons10": 2500,
                "tail_rebalance_accept_if_prev_stays_above_min": True,
                "max_tail_repair_windows_per_line": 6,
                "max_tail_repair_windows_total": 12,
                "max_recut_cutpoints_per_window": 6,
                "max_fill_candidates_per_tail": 30,
                "tail_repair_gap_to_min_limit": 320.0,
                "tail_repair_gap_limit_tons": 320.0,
                "tail_repair_min_fill_ratio": 0.55,
                "tail_repair_enable_near_viable_only": True,
                "tail_repair_max_windows_per_line": 6,
                "constructive_reverse_width_max_count": 3,
                "continuation_bias_gain_weight": 0.55,
                "continuation_bias_cross_min_bonus": 65.0,
                "continuation_bias_near_max_penalty": 45.0,
                "continuation_bias_near_max_ratio": 0.92,
                "continuation_bias_short_chain_hold_bonus": 18.0,
                "successor_weight_width_desc": 0.30,
                "successor_weight_continuation": 0.48,
                "successor_weight_extendability": 0.24,
                "successor_weight_group_continuity": 0.12,
                "successor_weight_reverse_budget_cost": 0.18,
                "successor_weight_dead_end_penalty": 0.20,
                "seed_formability_lookahead_orders": 28,
                "constructive_typical_order_tons": 25.0,
                "seed_weight_width_headroom": 0.38,
                "seed_weight_extendability": 0.16,
                "seed_weight_tonnage_formability": 0.28,
                "seed_weight_group_continuity": 0.08,
                "seed_weight_smoothness": 0.06,
                "seed_weight_reverse_friendliness": 0.04,
                "seed_formability_min_projected_tons": 350.0,
                "seed_formability_soft_gate_enabled": True,
                "seed_formability_hard_gate_enabled": False,
                "seed_penalty_low_formability": 80.0,
                "seed_bonus_width_head": 35.0,
                "seed_bonus_high_extendability": 25.0,
                "constructive_virtual_rescue_enabled": True,
                "constructive_virtual_rescue_under_min_only": True,
                "constructive_virtual_rescue_real_successor_threshold": 2,
                "constructive_virtual_rescue_bonus": 45.0,
                "constructive_virtual_rescue_cross_min_bonus": 85.0,
                "constructive_virtual_rescue_dead_end_bonus": 40.0,
                "constructive_virtual_rescue_near_min_ratio": 0.75,
                "constructive_virtual_rescue_max_virtual_chain": 5,
                "constructive_virtual_rescue_penalty_floor": 0.0,
                "constructive_virtual_rescue_big_roll_enabled": True,
                "constructive_virtual_rescue_big_roll_real_successor_threshold": 8,
                "constructive_virtual_rescue_ignore_rich_real_when_under_min_deadend": True,
                "constructive_virtual_rescue_big_roll_bonus": 70.0,
                "constructive_virtual_rescue_big_roll_cross_min_bonus": 110.0,
                "constructive_virtual_rescue_big_roll_near_min_ratio": 0.65,
                "constructive_virtual_rescue_score_scale": 1.0,
                "constructive_virtual_rescue_min_current_tons": 80.0,
                "constructive_virtual_rescue_try_even_when_real_successor_exists": True,
                "successor_cross_min_bonus": 95.0,
                "successor_near_min_bonus": 35.0,
                "successor_under_min_gain_weight": 0.70,
                "successor_near_max_penalty": 60.0,
                "successor_near_max_ratio": 0.92,
                "successor_short_chain_deadend_penalty": 55.0,
                "alns_include_underfilled_fragments": True,
                "alns_underfilled_fragment_max_count": 80,
                "alns_underfilled_fragment_min_tons": 80.0,
                "alns_underfilled_fragment_priority_near_min": True,
                "alns_underfilled_fragment_allow_virtual_rescue": True,
                "alns_underfilled_stitch_enabled": True,
                "alns_underfilled_stitch_max_pairs_per_round": 200,
                "alns_underfilled_stitch_max_bridge_trials": 3,
                "alns_underfilled_stitch_allow_virtual": True,
                "alns_underfilled_stitch_allow_real_bridge": True,
                "alns_underfilled_stitch_min_combined_tons": 650.0,
                "alns_underfilled_stitch_max_combined_tons": 2000.0,
                "alns_underfilled_stitch_prefer_same_line": True,
                "tail_borrow_from_adjacent_enabled": True,
                "tail_borrow_max_orders_per_attempt": 2,
                "tail_fill_from_dropped_enabled": True,
                "tail_fill_gap_to_min_limit": 320.0,
                "tail_fill_accept_partial_progress": True,
                "tail_fill_max_inserts_per_tail": 2,
                "tail_fill_second_pass_gap_limit": 30.0,
                "small_roll_dual_reserve_enabled": True,
                "small_roll_dual_reserve_penalty": 15,
                "small_roll_dual_reserve_bucket_enabled": True,
                "small_roll_dual_reserve_bucket_ratio": 0.38,
                "small_roll_dual_reserve_bucket_max_orders": 120,
                "small_roll_seed_first_enabled": True,
                "small_roll_seed_min_orders": 16,
                "small_roll_seed_min_tons10": 4500,
                "small_roll_dual_reserve_quota_enabled": True,
                "small_roll_dual_reserve_quota_min_orders": 25,
                "small_roll_dual_reserve_quota_min_tons10": 6000,
                "small_roll_dual_reserve_quota_max_orders": 50,
                "small_roll_dual_reserve_quota_max_tons10": 12000,
                "big_roll_dual_release_after_small_seed": True,
        }
        _log_model_overrides(
            base_model,
            model_overrides,
            source="aps_cp_sat.config.parameters:constructive_lns_virtual_guarded_frontload",
        )
        model = ModelConfig(
            **{
                **base_model.__dict__,
                **model_overrides,
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
                "rounds": 16,
                "constructive_lns_rounds": 16,
                "lns_early_stop_no_improve_rounds": 5,
                "lns_max_total_rounds": 16,
                "lns_min_rounds_before_early_stop": 6,
                "constructive_destroy_ratio_min": 0.20,
                "constructive_destroy_ratio_max": 0.35,
                "constructive_subproblem_max_orders": 56,
                "constructive_enable_cp_sat_repair": True,
                "constructive_lns_alns_enable_real_bridge_moves": True,
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
                "max_tail_repair_windows_per_line": 16,
                "max_tail_repair_windows_total": 32,
                "max_recut_cutpoints_per_window": 12,
                "max_fill_candidates_per_tail": 30,
                "tail_repair_gap_to_min_limit": 320.0,
                # ---- Tail Fill From Dropped: make fill branch actually trigger ----
                "tail_fill_from_dropped_enabled": True,
                "tail_fill_gap_to_min_limit": 320.0,
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
    # Block-First experiment line (block-first guard experiment)
    #
    # Profile semantics:
    #   block_first_guarded_search:
    #       main_solver_strategy = "block_first"
    #       allow_real_bridge_edge_in_constructive = True
    #       allow_virtual_bridge_edge_in_constructive = True (guarded family only)
    #       bridge_expansion_mode = "disabled"
    #       virtual_family_frontload_enabled = True
    #       repair_only_real_bridge_enabled = True
    #       repair_only_virtual_bridge_enabled = False
    #
    # Key difference from constructive_lns_virtual_guarded_frontload:
    #   - NOT order-first (chain → segment → repair)
    #   - INSTEAD block-first (candidate blocks → block master → block realize → block ALNS)
    #   - Bridge/mixed-bridge happen INSIDE each block, not in global graph
    # -------------------------------------------------------------------------
    if profile == "block_first_guarded_search":
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
                "profile_name": "block_first_guarded_search",
                # ---- Block-first master strategy ----
                "main_solver_strategy": "block_first",
                # LNS parameters (for block ALNS, not order ALNS)
                "rounds": 8,  # block-first 默认 8 轮块级 ALNS，先稳后强
                "constructive_lns_rounds": 8,  # 兼容字段，保持与块级 ALNS 一致
                "lns_early_stop_no_improve_rounds": 5,
                "lns_max_total_rounds": 16,
                "lns_min_rounds_before_early_stop": 6,
                "constructive_destroy_ratio_min": 0.20,
                "constructive_destroy_ratio_max": 0.35,
                "constructive_subproblem_max_orders": 56,
                "constructive_enable_cp_sat_repair": True,
                # ---- Bridge edge policy: same as guarded frontload ----
                "allow_virtual_bridge_edge_in_constructive": True,
                "allow_real_bridge_edge_in_constructive": True,
                "bridge_expansion_mode": "disabled",
                "repair_only_real_bridge_enabled": True,
                "repair_only_virtual_bridge_enabled": False,
                "repair_only_virtual_bridge_pilot_enabled": False,
                # ---- Guarded virtual family (block context) ----
                "virtual_family_frontload_enabled": True,
                "virtual_family_frontload_global_topk_per_from": 3,
                "virtual_family_frontload_global_max_edges_total": 360,
                "virtual_family_frontload_repair_max_edges_total": 900,
                "virtual_family_frontload_allowed_families": ["WIDTH_GROUP", "THICKNESS", "GROUP_TRANSITION"],
                "virtual_family_frontload_max_bridge_count": 2,
                "virtual_family_frontload_only_when_underfill_or_drop_pressure": True,
                "virtual_family_frontload_min_block_tons": 80.0,
                "virtual_family_frontload_max_block_tons": 450.0,
                "virtual_family_frontload_alns_only_extra_topk": 4,
                "virtual_family_frontload_local_cpsat_only": False,
                "virtual_family_frontload_global_penalty": 100.0,
                "virtual_family_frontload_local_penalty": 70.0,
                "virtual_family_frontload_require_family_budget": True,
                "virtual_family_budget_per_line": 4,
                "virtual_family_budget_per_segment": 2,
                # ---- Block generator configuration (Sub-block validation version) ----
                # 这里把 block 明确定义为“子块”，不是完整轧期；完整轧期由后续 slot 装配得到。
                "block_generator_target_blocks": 2000,
                "block_generator_time_limit_seconds": 18.0,  # 建块阶段多给少量时间，避免候选块供给过薄
                "block_generator_max_blocks_per_line": 50,  # 控制单线块池规模，避免噪声块太多
                "block_generator_max_blocks_total": 160,
                "block_generator_max_seed_per_bucket": 10,  # 每桶种子略收敛，减少重复风格块
                # ---- Block generator: candidate block (pool) threshold ----
                "block_generator_candidate_tons_min": 120.0,  # 子块候选最小吨位：允许小块先入池，后续再装槽
                "block_generator_candidate_tons_target": 220.0,  # 子块理想吨位：更贴近当前订单吨位规模
                "block_generator_candidate_tons_max": 480.0,  # 子块候选上限：避免把一个块长成半个轧期
                # ---- Block generator: ideal target block threshold ----
                "block_generator_target_tons_min": 180.0,  # 理想子块最低吨位
                "block_generator_target_tons_target": 320.0,  # 理想子块目标吨位：从 700 下调到更合理的子块语义
                "block_generator_target_tons_max": 520.0,  # 理想子块上限：为后续 slot 组装留空间
                "block_generator_max_orders_per_block": 12,  # 子块最多订单数：避免块内部复杂度过高
                "block_generator_allow_guarded_family": True,
                "block_generator_allow_real_bridge": True,
                "block_generator_allow_mixed_bridge_potential": True,
                "block_generator_max_family_edges_per_block": 1,
                "block_generator_max_real_bridge_edges_per_block": 1,
                "block_generator_max_bridge_count_per_block": 1,
                # ---- Directional clustering weights ----
                "directional_cluster_width_weight": 1.0,
                "directional_cluster_thickness_weight": 1.0,
                "directional_cluster_temp_weight": 0.8,
                "directional_cluster_group_weight": 1.2,
                "directional_cluster_due_weight": 0.6,
                "directional_cluster_tons_fill_weight": 1.0,
                "directional_cluster_real_bridge_bonus": 0.8,
                "directional_cluster_guarded_family_bonus": 0.5,
                "directional_cluster_mixed_bridge_potential_bonus": 0.3,
                # ---- Block ALNS configuration ----
                "block_alns_enabled": True,
                "block_alns_rounds": 8,
                "block_alns_early_stop_no_improve_rounds": 3,
                "block_alns_swap_enabled": True,
                "block_alns_replace_enabled": True,
                "block_alns_split_enabled": True,
                "block_alns_merge_enabled": True,
                "block_alns_boundary_rebalance_enabled": True,
                "block_alns_internal_rebuild_enabled": True,
                "block_alns_accept_threshold": 0.0,
                # ---- Block master configuration ----
                "block_master_slot_buffer": 3,
                "block_master_greedy": True,
                "block_master_max_conflict_skip": 5,
                "block_master_prefer_quality_score": True,
                # ---- Mixed bridge (block-internal only, NOT global) ----
                "mixed_bridge_in_block_enabled": True,
                "mixed_bridge_allowed_forms": ["REAL_TO_GUARDED", "GUARDED_TO_REAL"],
                "mixed_bridge_allowed_hotspots": ["underfill", "group_switch", "bridge_dependency", "width_tension"],
                "mixed_bridge_max_attempts_per_block": 8,  # 混合桥接尝试次数略收敛，避免 realizer 过度耗时
                # ---- Repair bridge (same as guarded frontload) ----
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
                "tail_rebalance_enabled": True,
                "tail_rebalance_max_pullback_orders": 8,
                "tail_rebalance_max_pullback_tons10": 2500,
                "tail_rebalance_accept_if_prev_stays_above_min": True,
                "max_tail_repair_windows_per_line": 16,
                "max_tail_repair_windows_total": 32,
                "max_recut_cutpoints_per_window": 12,
                "max_fill_candidates_per_tail": 30,
                "tail_repair_gap_to_min_limit": 320.0,
                "tail_fill_from_dropped_enabled": True,
                "tail_fill_gap_to_min_limit": 320.0,
                "tail_fill_accept_partial_progress": True,
                "tail_fill_max_inserts_per_tail": 2,
                "tail_fill_second_pass_gap_limit": 30.0,
                # ---- Segment shaping + minimal formal fill trial ----
                "near_viable_gap_tons": 80.0,
                "merge_candidate_gap_tons": 250.0,
                "virtual_fill_unit_tons_assumption": 20.0,
                "virtual_max_count_per_campaign": 3,
                "virtual_shadow_bridge_enabled": True,
                "virtual_formal_bridge_enabled": False,
                "shadow_bridge_max_gap_count": 1,
                "shadow_bridge_max_virtual_count_per_gap": 5,
                "shadow_bridge_max_total_virtual_tons_per_campaign": 100.0,
                "shadow_bridge_same_line_only": True,
                "virtual_formal_fill_enabled": True,
                "virtual_formal_fill_max_gap_tons": 80.0,
                "virtual_formal_fill_max_count_per_campaign": 3,
                "virtual_formal_fill_tail_only": True,
                "formal_single_bridge_trial_enabled": False,
                "formal_single_bridge_trial_proposal_id": "",
                "formal_single_bridge_trial_dry_run": False,
                "formal_single_bridge_trial_allow_virtual_bridge": True,
                "formal_single_bridge_trial_allow_mixed_bridge": False,
                "small_roll_dual_reserve_enabled": True,
                "small_roll_dual_reserve_penalty": 15,
                "small_roll_dual_reserve_bucket_enabled": True,
                "small_roll_dual_reserve_bucket_ratio": 0.45,
                "small_roll_dual_reserve_bucket_max_orders": 120,
                "small_roll_seed_first_enabled": True,
                "small_roll_seed_min_orders": 20,
                "small_roll_seed_min_tons10": 5000,
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
        - None / "" / "default" -> "constructive_lns_virtual_guarded_frontload"
        - Other values are returned as-is (to allow explicit runtime errors)
    """
    if profile_name is None:
        return "constructive_lns_virtual_guarded_frontload"
    name = str(profile_name).strip().lower()
    if name in ("", "default"):
        return "constructive_lns_virtual_guarded_frontload"
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
        "constructive_lns_virtual_guarded_frontload",
        validation_mode=validation_mode,
        production_compatibility_mode=production_compatibility_mode,
    )
