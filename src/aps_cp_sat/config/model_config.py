from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class ModelConfig:
    # ------------------------ 运行入口与全局预算 ------------------------
    profile_name: str = "constructive_lns_virtual_guarded_frontload"  # 当前使用的参数档位名称，用于区分路线和实验。
    max_orders: int = 720  # 单次运行允许读取的订单数量上限。
    rounds: int = 6  # 通用迭代轮数，具体 profile 可覆盖。
    time_limit_seconds: float = 30.0  # 单次求解总时间上限，单位为秒。
    random_seed: int = 42  # 随机种子，用于复现实验结果。
    master_profile_count: int = 2  # 数量参数：master_profile_count。
    master_seed_count: int = 1  # 数量参数：master_seed_count。
    master_num_workers: int = 4  # 参数：master_num_workers，用于控制相关算法行为。
    # Explicit fallback strategy switches for the production joint-master path.
    enableSemanticFallback: bool = False  # 算法开关：enableSemanticFallback，用于控制相关算法行为。
    enableScaleDownFallback: bool = False  # 算法开关：enableScaleDownFallback，用于控制相关算法行为。
    enableLegacyFallback: bool = False  # 算法开关：enableLegacyFallback，用于控制相关算法行为。
    productionAllowLegacyFallback: bool = False  # 参数：productionAllowLegacyFallback，用于控制相关算法行为。
    export_failed_result_for_debug: bool = True  # 参数：export_failed_result_for_debug，用于控制相关算法行为。
    export_analysis_on_failure: bool = False  # 参数：export_analysis_on_failure，用于控制相关算法行为。
    export_best_candidate_analysis: bool = False  # 参数：export_best_candidate_analysis，用于控制相关算法行为。
    enableStructureFallback: bool = False  # 算法开关：enableStructureFallback，用于控制相关算法行为。
    max_drop_ratio_for_partial: float = 0.05  # 约束上限：max_drop_ratio_for_partial。
    max_drop_tons_ratio_for_partial: float = 0.08  # 约束上限：max_drop_tons_ratio_for_partial。
    max_drop_count_for_partial: int = 20  # 约束上限：max_drop_count_for_partial。
    min_scheduled_orders_for_partial: int = 100  # 约束下限：min_scheduled_orders_for_partial。
    min_scheduled_tons_for_partial: float = 1000.0  # 约束下限：min_scheduled_tons_for_partial。
    structure_fallback_slot_buffer: int = 4  # 轧期或槽位相关参数：structure_fallback_slot_buffer。
    structure_fallback_risk_boost: float = 1.5  # 参数：structure_fallback_risk_boost，用于控制相关算法行为。
    structure_fallback_min_real_schedule_ratio: float = 0.55  # 约束下限：structure_fallback_min_real_schedule_ratio。
    fast_fail_on_bad_slots: bool = False  # 轧期或槽位相关参数：fast_fail_on_bad_slots。
    fast_fail_unroutable_slot_threshold: int = 10  # 轧期或槽位相关参数：fast_fail_unroutable_slot_threshold。
    fast_fail_slot_order_count_threshold: int = 28  # 数量参数：fast_fail_slot_order_count_threshold。
    fast_fail_slot_coverage_threshold: float = 0.15  # 轧期或槽位相关参数：fast_fail_slot_coverage_threshold。
    fast_fail_slot_zero_degree_threshold: int = 6  # 轧期或槽位相关参数：fast_fail_slot_zero_degree_threshold。
    fast_fail_topn_slots: int = 3  # 轧期或槽位相关参数：fast_fail_topn_slots。
    allow_fallback: bool = False  # 算法开关：allow_fallback，用于控制相关算法行为。
    allow_legacy_fallback: bool = False  # 算法开关：allow_legacy_fallback，用于控制相关算法行为。
    validation_mode: bool = False  # 参数：validation_mode，用于控制相关算法行为。
    production_compatibility_mode: bool = False  # 参数：production_compatibility_mode，用于控制相关算法行为。
    enable_tiered_objective: bool = True  # 算法开关：enable_tiered_objective，用于控制相关算法行为。
    min_real_schedule_ratio: float = 0.90  # 约束下限：min_real_schedule_ratio。
    max_virtual_chain: int = 5  # 约束上限：max_virtual_chain。
    max_unbridgeable_drop_ratio: float = 0.08  # 约束上限：max_unbridgeable_drop_ratio。
    lot_max_tons: float = 120.0  # 约束上限：lot_max_tons。
    sparse_k_same_group: int = 10  # 参数：sparse_k_same_group，用于控制相关算法行为。
    sparse_k_same_thickness: int = 8  # 参数：sparse_k_same_thickness，用于控制相关算法行为。
    sparse_k_cross_group: int = 6  # 参数：sparse_k_cross_group，用于控制相关算法行为。
    sparse_k_due_tight: int = 6  # 参数：sparse_k_due_tight，用于控制相关算法行为。
    template_min_out_degree: int = 1  # 约束下限：template_min_out_degree。
    template_min_in_degree: int = 1  # 约束下限：template_min_in_degree。
    template_top_k: int = 48  # 数量参数：template_top_k。
    global_prune_max_pairs_per_from: int = 0  # 约束上限：global_prune_max_pairs_per_from。
    max_routes_per_slot: int = 6  # 约束上限：max_routes_per_slot。
    template_health_sparse_ratio: float = 0.08  # 比例参数：template_health_sparse_ratio。
    template_health_zero_degree_ratio: float = 0.15  # 比例参数：template_health_zero_degree_ratio。
    semantic_fallback_rounds: int = 2  # 数量参数：semantic_fallback_rounds。
    scale_down_keep_steps: Tuple[int, ...] = (520, 320, 160)  # 参数：scale_down_keep_steps，用于控制相关算法行为。
    strict_template_edges: bool = True  # 参数：strict_template_edges，用于控制相关算法行为。
    min_campaign_slots: int = 6  # 约束下限：min_campaign_slots。
    max_campaign_slots: int = 14  # 约束上限：max_campaign_slots。
    big_roll_slot_soft_order_cap: int = 20  # 轧期或槽位相关参数：big_roll_slot_soft_order_cap。
    small_roll_slot_soft_order_cap: int = 24  # 轧期或槽位相关参数：small_roll_slot_soft_order_cap。
    big_roll_slot_hard_order_cap: int = 0  # 轧期或槽位相关参数：big_roll_slot_hard_order_cap。
    small_roll_slot_hard_order_cap: int = 0  # 轧期或槽位相关参数：small_roll_slot_hard_order_cap。
    strict_virtual_width_levels: bool = False  # 虚拟板坯相关参数：strict_virtual_width_levels。
    physical_reverse_step_mode: bool = True  # 参数：physical_reverse_step_mode，用于控制相关算法行为。
    min_master_solve_seconds: float = 20.0  # 约束下限：min_master_solve_seconds。
    max_master_solve_seconds: float = 180.0  # 约束上限：max_master_solve_seconds。
    min_skeleton_solve_seconds: float = 2.0  # 约束下限：min_skeleton_solve_seconds。
    max_skeleton_solve_seconds: float = 20.0  # 约束上限：max_skeleton_solve_seconds。
    dual_width_anchor: float = 1800.0  # 参数：dual_width_anchor，用于控制相关算法行为。
    # Set Packing Master (Macro-Block Assembly) architecture
    use_set_packing_master: bool = False  # 参数：use_set_packing_master，用于控制相关算法行为。
    # -------------------------------------------------------------------------
    # Constructive LNS path (new solver strategy)
    # main_solver_strategy controls which master path is used:
    #   "joint_master"        - original production joint model (default)
    #   "constructive_lns"    - new ALNS-driven constructive path
    # -------------------------------------------------------------------------
    main_solver_strategy: str = "constructive_lns"  # 主求解路线选择，例如 constructive_lns 或 block_first。
    constructive_lns_rounds: int = 12  # 数量参数：constructive_lns_rounds。
    constructive_destroy_ratio_min: float = 0.18  # 约束下限：constructive_destroy_ratio_min。
    constructive_destroy_ratio_max: float = 0.30  # 约束上限：constructive_destroy_ratio_max。
    constructive_subproblem_max_orders: int = 48  # 约束上限：constructive_subproblem_max_orders。
    constructive_enable_cp_sat_repair: bool = True  # 参数：constructive_enable_cp_sat_repair，用于控制相关算法行为。
    # ---- LNS Early Stop Configuration ----
    lns_early_stop_no_improve_rounds: int = 4  # 数量参数：lns_early_stop_no_improve_rounds。
    lns_max_total_rounds: int = 12  # 约束上限：lns_max_total_rounds。
    lns_min_rounds_before_early_stop: int = 5  # 约束下限：lns_min_rounds_before_early_stop。
    # ---- Bridge edge policy for constructive_lns path ----
    # Route RB (mainline): allow REAL_BRIDGE_EDGE, disable VIRTUAL_BRIDGE_EDGE
    #   - allow_real_bridge_edge_in_constructive = True  (mainline default)
    #   - allow_virtual_bridge_edge_in_constructive = False
    # Route C (baseline): disable ALL bridge edges in constructive
    #   - allow_real_bridge_edge_in_constructive = False
    #   - allow_virtual_bridge_edge_in_constructive = False
    allow_virtual_bridge_edge_in_constructive: bool = False  # constructive 主搜索中是否允许虚拟桥接边进入候选图。
    allow_real_bridge_edge_in_constructive: bool = False  # constructive 主搜索中是否允许实物桥接边进入候选图。
    bridge_expansion_mode: str = "disabled"  # 桥接展开模式，disabled 表示不展开虚拟桥。
    # ---- Virtual Bridge Family Edge: Guarded Frontload (受控前移实验线) ----
    # Only applicable when allow_virtual_bridge_edge_in_constructive = True AND
    # the edge type is VIRTUAL_BRIDGE_FAMILY_EDGE (NOT legacy VIRTUAL_BRIDGE_EDGE).
    # Legacy virtual exact expansion remains disabled.
    # Dual-pool: global pool = strict (greedy), repair pool = wider (ALNS + local rebuild).
    virtual_family_frontload_enabled: bool = False  # 算法开关：virtual_family_frontload_enabled，用于控制相关算法行为。
    virtual_family_frontload_global_topk_per_from: int = 3  # 数量参数：virtual_family_frontload_global_topk_per_from。
    virtual_family_frontload_global_max_edges_total: int = 360  # 约束上限：virtual_family_frontload_global_max_edges_total。
    virtual_family_frontload_repair_max_edges_total: int = 900  # 约束上限：virtual_family_frontload_repair_max_edges_total。
    virtual_family_frontload_allowed_families: list[str] = field(default_factory=lambda: ["WIDTH_GROUP", "THICKNESS", "GROUP_TRANSITION"])  # 虚拟板坯相关参数：virtual_family_frontload_allowed_families。
    virtual_family_frontload_max_bridge_count: int = 2  # 约束上限：virtual_family_frontload_max_bridge_count。
    virtual_family_frontload_only_when_underfill_or_drop_pressure: bool = True  # 虚拟板坯相关参数：virtual_family_frontload_only_when_underfill_or_drop_pressure。
    virtual_family_frontload_min_block_tons: float = 80.0  # 约束下限：virtual_family_frontload_min_block_tons。
    virtual_family_frontload_max_block_tons: float = 450.0  # 约束上限：virtual_family_frontload_max_block_tons。
    virtual_family_frontload_alns_only_extra_topk: int = 4  # 数量参数：virtual_family_frontload_alns_only_extra_topk。
    virtual_family_frontload_local_cpsat_only: bool = False  # 虚拟板坯相关参数：virtual_family_frontload_local_cpsat_only。
    virtual_family_frontload_global_penalty: float = 100.0  # 惩罚权重：virtual_family_frontload_global_penalty。
    virtual_family_frontload_local_penalty: float = 70.0  # 惩罚权重：virtual_family_frontload_local_penalty。
    virtual_family_frontload_require_family_budget: bool = True  # 虚拟板坯相关参数：virtual_family_frontload_require_family_budget。
    virtual_family_budget_per_line: int = 4  # 虚拟板坯相关参数：virtual_family_budget_per_line。
    virtual_family_budget_per_segment: int = 2  # 虚拟板坯相关参数：virtual_family_budget_per_segment。
    # Repair-only bridge policy. Initial constructive and initial cutter remain
    # direct-only; these switches apply only to underfilled reconstruction.
    repair_only_real_bridge_enabled: bool = True  # 算法开关：repair_only_real_bridge_enabled，用于控制相关算法行为。
    repair_only_virtual_bridge_enabled: bool = False  # 算法开关：repair_only_virtual_bridge_enabled，用于控制相关算法行为。
    repair_only_virtual_bridge_pilot_enabled: bool = False  # 算法开关：repair_only_virtual_bridge_pilot_enabled，用于控制相关算法行为。
    virtual_bridge_pilot_max_blocks_per_run: int = 15  # 约束上限：virtual_bridge_pilot_max_blocks_per_run。
    virtual_bridge_pilot_max_per_block: int = 1  # 约束上限：virtual_bridge_pilot_max_per_block。
    virtual_bridge_pilot_max_virtual_tons: float = 30.0  # 约束上限：virtual_bridge_pilot_max_virtual_tons。
    virtual_bridge_pilot_penalty: float = 1000000.0  # 惩罚权重：virtual_bridge_pilot_penalty。
    virtual_bridge_pilot_only_when_endpoint_class: list[str] = field(default_factory=lambda: ["HAS_ENDPOINT_EDGE", "BAND_TOO_NARROW"])  # 桥接相关参数：virtual_bridge_pilot_only_when_endpoint_class。
    virtual_bridge_pilot_only_when_dominant_fail: list[str] = field(default_factory=lambda: ["THICKNESS_RULE_FAIL", "WIDTH_RULE_FAIL", "GROUP_SWITCH_FAIL", "MULTI_RULE_FAIL"])  # 约束下限：virtual_bridge_pilot_only_when_dominant_fail。
    repair_only_bridge_max_per_segment: int = 1  # 约束上限：repair_only_bridge_max_per_segment。
    repair_only_bridge_cost_penalty: float = 100000.0  # 惩罚权重：repair_only_bridge_cost_penalty。
    repair_bridge_left_band_k: int = 3  # 桥接相关参数：repair_bridge_left_band_k。
    repair_bridge_right_band_k: int = 3  # 桥接相关参数：repair_bridge_right_band_k。
    repair_bridge_band_max_pairs_per_split: int = 9  # 约束上限：repair_bridge_band_max_pairs_per_split。
    repair_bridge_left_trim_max: int = 2  # 约束上限：repair_bridge_left_trim_max。
    repair_bridge_right_trim_max: int = 2  # 约束上限：repair_bridge_right_trim_max。
    repair_bridge_endpoint_adjustment_limit_per_split: int = 9  # 桥接相关参数：repair_bridge_endpoint_adjustment_limit_per_split。
    repair_bridge_adjustment_enable_left_trim: bool = True  # 桥接相关参数：repair_bridge_adjustment_enable_left_trim。
    repair_bridge_adjustment_enable_right_trim: bool = True  # 桥接相关参数：repair_bridge_adjustment_enable_right_trim。
    repair_bridge_adjustment_enable_swap: bool = False  # 桥接相关参数：repair_bridge_adjustment_enable_swap。
    repair_bridge_ton_rescue_max_neighbor_blocks: int = 6  # 约束上限：repair_bridge_ton_rescue_max_neighbor_blocks。
    repair_bridge_ton_rescue_enable_backward: bool = True  # 吨位参数：repair_bridge_ton_rescue_enable_backward。
    repair_bridge_ton_rescue_enable_forward: bool = True  # 吨位参数：repair_bridge_ton_rescue_enable_forward。
    repair_bridge_ton_rescue_enable_bidirectional: bool = True  # 吨位参数：repair_bridge_ton_rescue_enable_bidirectional。
    repair_bridge_ton_rescue_max_orders_per_window: int = 50  # 约束上限：repair_bridge_ton_rescue_max_orders_per_window。
    repair_bridge_ton_rescue_max_failed_windows_after_min: int = 2  # 约束上限：repair_bridge_ton_rescue_max_failed_windows_after_min。
    # ---- Tail rebalancing: rescue underfilled tail segments by pullback ----
    tail_rebalance_enabled: bool = True  # 算法开关：tail_rebalance_enabled，用于控制相关算法行为。
    tail_rebalance_max_pullback_orders: int = 8  # 约束上限：tail_rebalance_max_pullback_orders。
    tail_rebalance_max_pullback_tons10: int = 2500  # 约束上限：tail_rebalance_max_pullback_tons10。
    tail_rebalance_accept_if_prev_stays_above_min: bool = True  # 约束下限：tail_rebalance_accept_if_prev_stays_above_min。
    # ---- Small roll dual-order reserve: prevent big_roll from monopolizing dual orders ----
    small_roll_dual_reserve_enabled: bool = True  # 算法开关：small_roll_dual_reserve_enabled，用于控制相关算法行为。
    small_roll_dual_reserve_penalty: int = 15  # 惩罚权重：small_roll_dual_reserve_penalty。
    # ---- Small roll dual-order reserve bucket: hard-prevent big_roll from taking top dual candidates ----
    small_roll_dual_reserve_bucket_enabled: bool = True  # 算法开关：small_roll_dual_reserve_bucket_enabled，用于控制相关算法行为。
    small_roll_dual_reserve_bucket_ratio: float = 0.40  # 比例参数：small_roll_dual_reserve_bucket_ratio。
    small_roll_dual_reserve_bucket_max_orders: int = 100  # 约束上限：small_roll_dual_reserve_bucket_max_orders。
    # ---- Small roll seed-first: prioritize small_roll chain building before big_roll consumes reserve ----
    small_roll_seed_first_enabled: bool = True  # 算法开关：small_roll_seed_first_enabled，用于控制相关算法行为。
    small_roll_seed_min_orders: int = 24  # 约束下限：small_roll_seed_min_orders。
    small_roll_seed_min_tons10: int = 6000  # 约束下限：small_roll_seed_min_tons10。
    # ---- Small roll dual-order reserve QUOTA: balanced allocation instead of "lock all" ----
    # Quota enables a floor (min) AND a ceiling (max) for small_roll's priority on dual orders.
    # Once small_roll reaches max quota, remaining reserve bucket is released to big_roll.
    small_roll_dual_reserve_quota_enabled: bool = True  # 算法开关：small_roll_dual_reserve_quota_enabled，用于控制相关算法行为。
    small_roll_dual_reserve_quota_min_orders: int = 25  # 约束下限：small_roll_dual_reserve_quota_min_orders。
    small_roll_dual_reserve_quota_min_tons10: int = 6000  # 约束下限：small_roll_dual_reserve_quota_min_tons10。
    small_roll_dual_reserve_quota_max_orders: int = 60  # 约束上限：small_roll_dual_reserve_quota_max_orders。
    small_roll_dual_reserve_quota_max_tons10: int = 14000  # 约束上限：small_roll_dual_reserve_quota_max_tons10。
    # ---- Big roll release: after small_roll seed phase, release remaining quota to big_roll ----
    big_roll_dual_release_after_small_seed: bool = True  # 参数：big_roll_dual_release_after_small_seed，用于控制相关算法行为。
    # ---- Tail Repair Budget: limit search scope to reduce runtime ----
    max_tail_repair_windows_per_line: int = 12  # 约束上限：max_tail_repair_windows_per_line。
    max_tail_repair_windows_total: int = 24  # 约束上限：max_tail_repair_windows_total。
    max_recut_cutpoints_per_window: int = 12  # 约束上限：max_recut_cutpoints_per_window。
    max_fill_candidates_per_tail: int = 8  # 约束上限：max_fill_candidates_per_tail。
    tail_repair_gap_to_min_limit: float = 180.0  # 约束下限：tail_repair_gap_to_min_limit。
    tail_fill_from_dropped_enabled: bool = True  # 算法开关：tail_fill_from_dropped_enabled，用于控制相关算法行为。
    tail_fill_gap_to_min_limit: float = 220.0  # 约束下限：tail_fill_gap_to_min_limit。
    tail_fill_accept_partial_progress: bool = True  # 参数：tail_fill_accept_partial_progress，用于控制相关算法行为。
    tail_fill_max_inserts_per_tail: int = 2  # 约束上限：tail_fill_max_inserts_per_tail。
    tail_fill_second_pass_gap_limit: float = 30.0  # 参数：tail_fill_second_pass_gap_limit，用于控制相关算法行为。
    # ---- Block Generator Configuration (block_first_guarded_search) ----
    # 这里的 block 语义按“子块/sub-block”理解，而不是完整轧期。
    block_generator_target_blocks: int = 2000  # block-first 块级求解参数：block_generator_target_blocks。
    block_generator_time_limit_seconds: float = 15.0  # block-first 块级求解参数：block_generator_time_limit_seconds。
    block_generator_max_blocks_per_line: int = 50  # 约束上限：block_generator_max_blocks_per_line。
    block_generator_max_blocks_total: int = 160  # 约束上限：block_generator_max_blocks_total。
    block_generator_max_seed_per_bucket: int = 10  # 约束上限：block_generator_max_seed_per_bucket。
    block_generator_target_tons_min: float = 180.0  # 约束下限：block_generator_target_tons_min。
    block_generator_target_tons_target: float = 320.0  # 吨位参数：block_generator_target_tons_target。
    block_generator_target_tons_max: float = 520.0  # 约束上限：block_generator_target_tons_max。
    # ---- Candidate block (pool) threshold: looser than target_tons_* ----
    # Blocks between candidate_tons_min and target_tons_min are accepted
    # into the candidate pool as small_candidate blocks, but are penalized
    # by block_master to discourage them as final campaign blocks.
    block_generator_candidate_tons_min: float = 120.0  # 约束下限：block_generator_candidate_tons_min。
    block_generator_candidate_tons_target: float = 220.0  # 吨位参数：block_generator_candidate_tons_target。
    block_generator_candidate_tons_max: float = 480.0  # 约束上限：block_generator_candidate_tons_max。
    block_generator_max_orders_per_block: int = 12  # 约束上限：block_generator_max_orders_per_block。
    block_generator_allow_guarded_family: bool = True  # block-first 块级求解参数：block_generator_allow_guarded_family。
    block_generator_allow_real_bridge: bool = True  # 桥接相关参数：block_generator_allow_real_bridge。
    block_generator_allow_mixed_bridge_potential: bool = True  # 桥接相关参数：block_generator_allow_mixed_bridge_potential。
    block_generator_max_family_edges_per_block: int = 1  # 约束上限：block_generator_max_family_edges_per_block。
    block_generator_max_real_bridge_edges_per_block: int = 1  # 约束上限：block_generator_max_real_bridge_edges_per_block。
    block_generator_max_bridge_count_per_block: int = 1  # 约束上限：block_generator_max_bridge_count_per_block。
    # ---- Directional Clustering Weights ----
    directional_cluster_width_weight: float = 1.0  # 评分权重：directional_cluster_width_weight。
    directional_cluster_thickness_weight: float = 1.0  # 评分权重：directional_cluster_thickness_weight。
    directional_cluster_temp_weight: float = 0.8  # 评分权重：directional_cluster_temp_weight。
    directional_cluster_group_weight: float = 1.2  # 评分权重：directional_cluster_group_weight。
    directional_cluster_due_weight: float = 0.6  # 评分权重：directional_cluster_due_weight。
    directional_cluster_tons_fill_weight: float = 1.0  # 评分权重：directional_cluster_tons_fill_weight。
    directional_cluster_real_bridge_bonus: float = 0.8  # 桥接相关参数：directional_cluster_real_bridge_bonus。
    directional_cluster_guarded_family_bonus: float = 0.5  # 参数：directional_cluster_guarded_family_bonus，用于控制相关算法行为。
    directional_cluster_mixed_bridge_potential_bonus: float = 0.3  # 桥接相关参数：directional_cluster_mixed_bridge_potential_bonus。
    # ---- Block ALNS Configuration ----
    block_alns_enabled: bool = True  # 算法开关：block_alns_enabled，用于控制相关算法行为。
    block_alns_rounds: int = 8  # 数量参数：block_alns_rounds。
    block_alns_early_stop_no_improve_rounds: int = 3  # 数量参数：block_alns_early_stop_no_improve_rounds。
    block_alns_swap_enabled: bool = True  # 算法开关：block_alns_swap_enabled，用于控制相关算法行为。
    block_alns_replace_enabled: bool = True  # 算法开关：block_alns_replace_enabled，用于控制相关算法行为。
    block_alns_split_enabled: bool = True  # 算法开关：block_alns_split_enabled，用于控制相关算法行为。
    block_alns_merge_enabled: bool = True  # 算法开关：block_alns_merge_enabled，用于控制相关算法行为。
    block_alns_boundary_rebalance_enabled: bool = True  # 算法开关：block_alns_boundary_rebalance_enabled，用于控制相关算法行为。
    block_alns_internal_rebuild_enabled: bool = True  # 算法开关：block_alns_internal_rebuild_enabled，用于控制相关算法行为。
    block_alns_accept_threshold: float = 0.0  # block-first 块级求解参数：block_alns_accept_threshold。
    # ---- Block Master Configuration ----
    block_master_slot_buffer: int = 3  # block-first 块级求解参数：block_master_slot_buffer。
    block_master_greedy: bool = True  # block-first 块级求解参数：block_master_greedy。
    block_master_max_conflict_skip: int = 5  # 约束上限：block_master_max_conflict_skip。
    block_master_prefer_quality_score: bool = True  # block-first 块级求解参数：block_master_prefer_quality_score。
    # ---- Mixed Bridge (block-internal only) ----
    mixed_bridge_in_block_enabled: bool = True  # 算法开关：mixed_bridge_in_block_enabled，用于控制相关算法行为。
    mixed_bridge_allowed_forms: list[str] = field(default_factory=lambda: ["REAL_TO_GUARDED", "GUARDED_TO_REAL"])  # 桥接相关参数：mixed_bridge_allowed_forms。
    mixed_bridge_allowed_hotspots: list[str] = field(default_factory=lambda: ["underfill", "group_switch", "bridge_dependency", "width_tension"])  # 桥接相关参数：mixed_bridge_allowed_hotspots。
    mixed_bridge_max_attempts_per_block: int = 8  # 约束上限：mixed_bridge_max_attempts_per_block。
    # ---- Segment viability bands (final campaign shaping) ----
    near_viable_gap_tons: float = 80.0  # 吨位参数：near_viable_gap_tons。
    merge_candidate_gap_tons: float = 250.0  # 吨位参数：merge_candidate_gap_tons。
    # ---- Virtual slab shadow/formal contract ----
    # Shadow mode estimates virtual bridge/fill benefits without writing virtual
    # rows into official schedules. Formal mode is intentionally off by default.
    virtual_enabled: bool = True  # 算法开关：virtual_enabled，用于控制相关算法行为。
    virtual_shadow_mode_enabled: bool = True  # 算法开关：virtual_shadow_mode_enabled，用于控制相关算法行为。
    virtual_shadow_fill_enabled: bool = True  # 算法开关：virtual_shadow_fill_enabled，用于控制相关算法行为。
    virtual_shadow_bridge_enabled: bool = True  # 算法开关：virtual_shadow_bridge_enabled，用于控制相关算法行为。
    virtual_formal_enabled: bool = False  # 算法开关：virtual_formal_enabled，用于控制相关算法行为。
    virtual_budget_total_tons: float = 500.0  # 吨位参数：virtual_budget_total_tons。
    virtual_budget_per_campaign_tons: float = 80.0  # 吨位参数：virtual_budget_per_campaign_tons。
    virtual_max_count_per_campaign: int = 2  # 约束上限：virtual_max_count_per_campaign。
    virtual_max_consecutive_chain: int = 5  # 约束上限：virtual_max_consecutive_chain。
    virtual_penalty_per_piece: float = 1000.0  # 惩罚权重：virtual_penalty_per_piece。
    virtual_penalty_per_ton: float = 20.0  # 惩罚权重：virtual_penalty_per_ton。
    virtual_bridge_only_for_allowed_gaps: bool = True  # 桥接相关参数：virtual_bridge_only_for_allowed_gaps。
    virtual_formal_bridge_enabled: bool = False  # 算法开关：virtual_formal_bridge_enabled，用于控制相关算法行为。
    shadow_bridge_max_gap_count: int = 1  # 约束上限：shadow_bridge_max_gap_count。
    shadow_bridge_max_virtual_count_per_gap: int = 5  # 约束上限：shadow_bridge_max_virtual_count_per_gap。
    shadow_bridge_max_total_virtual_tons_per_campaign: float = 100.0  # 约束上限：shadow_bridge_max_total_virtual_tons_per_campaign。
    shadow_bridge_same_line_only: bool = True  # 桥接相关参数：shadow_bridge_same_line_only。
    virtual_fill_unit_tons_assumption: float = 20.0  # 吨位参数：virtual_fill_unit_tons_assumption。
    virtual_allowed_lines: list[str] = field(default_factory=lambda: ["big_roll", "small_roll"])  # 虚拟板坯相关参数：virtual_allowed_lines。
    virtual_allowed_steel_groups: list[str] = field(default_factory=list)  # 虚拟板坯相关参数：virtual_allowed_steel_groups。
    virtual_bridge_mode: str = "prebuilt_virtual_inventory"  # 虚拟桥接模式：template_bridge 使用原模板桥接，prebuilt_virtual_inventory 使用预生成虚拟库存。
    prebuilt_virtual_inventory_enabled: bool = True  # 是否启用图生成阶段预生成虚拟卷库存。
    prebuilt_virtual_count_per_spec: int = 5  # 每个宽度和厚度规格组合预生成的虚拟卷数量。
    prebuilt_virtual_widths: list[int] = field(default_factory=lambda: [1000, 1250, 1500])  # 预生成虚拟卷允许的宽度规格集合。
    prebuilt_virtual_thicknesses: list[float] = field(default_factory=lambda: [0.6, 0.8, 1.0, 1.2, 1.5, 2.0])  # 预生成虚拟卷允许的厚度规格集合。
    prebuilt_virtual_temp_min: float = 600.0  # 预生成虚拟卷默认温区下限。
    prebuilt_virtual_temp_max: float = 900.0  # 预生成虚拟卷默认温区上限。
    prebuilt_virtual_group: str = "普碳"  # 预生成虚拟卷默认钢种组。
    prebuilt_virtual_default_tons: float = 20.0  # 预生成虚拟卷默认吨位。
    prebuilt_virtual_generate_for_big_roll: bool = True  # 是否为大辊线生成预置虚拟卷库存。
    prebuilt_virtual_generate_for_small_roll: bool = True  # 是否为小辊线生成预置虚拟卷库存。
    prebuilt_virtual_use_same_line_capability_rules: bool = True  # 是否按现有真实订单能力分布推断虚拟卷 line_capability。
    # ---- Minimal formal virtual fill trial (tail-only, no bridge) ----
    virtual_formal_fill_enabled: bool = False  # 算法开关：virtual_formal_fill_enabled，用于控制相关算法行为。
    virtual_formal_fill_max_gap_tons: float = 80.0  # 约束上限：virtual_formal_fill_max_gap_tons。
    virtual_formal_fill_max_count_per_campaign: int = 3  # 约束上限：virtual_formal_fill_max_count_per_campaign。
    virtual_formal_fill_tail_only: bool = True  # 虚拟板坯相关参数：virtual_formal_fill_tail_only。
    # ---- Formal single bridge trial (one proposal only, sandbox guarded) ----
    formal_single_bridge_trial_enabled: bool = False  # 算法开关：formal_single_bridge_trial_enabled，用于控制相关算法行为。
    formal_single_bridge_trial_proposal_id: str = ""  # 桥接相关参数：formal_single_bridge_trial_proposal_id。
    formal_single_bridge_trial_dry_run: bool = False  # 桥接相关参数：formal_single_bridge_trial_dry_run。
    formal_single_bridge_trial_allow_virtual_bridge: bool = True  # 桥接相关参数：formal_single_bridge_trial_allow_virtual_bridge。
    formal_single_bridge_trial_allow_mixed_bridge: bool = False  # 桥接相关参数：formal_single_bridge_trial_allow_mixed_bridge。
    penalty_virtual_piece_count: float = 1000.0  # 惩罚权重：penalty_virtual_piece_count。
    penalty_virtual_tons: float = 20.0  # 惩罚权重：penalty_virtual_tons。
    penalty_virtual_bridge_segments: float = 500.0  # 惩罚权重：penalty_virtual_bridge_segments。
    penalty_consecutive_virtual_chain: float = 2000.0  # 惩罚权重：penalty_consecutive_virtual_chain。
    reward_drop_reduction_if_virtual: float = 800.0  # 奖励权重：reward_drop_reduction_if_virtual。
    reward_hard_violation_elimination_if_virtual: float = 1500.0  # 奖励权重：reward_hard_violation_elimination_if_virtual。
    # ---- Constructive-only ALNS controls ----
    constructive_lns_alns_rounds: int = 12  # constructive_lns 路线专用 ALNS 轮数。
    constructive_lns_alns_enable_tail_repair: bool = True  # constructive_lns ALNS 是否允许尾段修复。
    constructive_lns_alns_enable_direct_insert: bool = True  # constructive_lns ALNS 是否允许直接插入。
    constructive_lns_alns_enable_campaign_merge: bool = True  # constructive_lns ALNS 是否允许轧期合并。
    constructive_lns_alns_enable_bridge_insertion: bool = True  # constructive_lns ALNS 是否允许桥接插入。
    constructive_lns_alns_enable_virtual_inventory_moves: bool = False  # constructive_lns ALNS 是否允许虚拟库存感知移动。
    constructive_lns_alns_enable_real_bridge_moves: bool = True  # constructive_lns ALNS 是否允许真实桥接边重排移动。

