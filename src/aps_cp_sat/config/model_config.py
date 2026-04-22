from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class ModelConfig:
    profile_name: str = "default"
    max_orders: int = 720
    rounds: int = 4
    time_limit_seconds: float = 20.0
    random_seed: int = 42
    master_profile_count: int = 2
    master_seed_count: int = 1
    master_num_workers: int = 4
    # Explicit fallback strategy switches for the production joint-master path.
    enableSemanticFallback: bool = False
    enableScaleDownFallback: bool = False
    enableLegacyFallback: bool = False
    productionAllowLegacyFallback: bool = False
    export_failed_result_for_debug: bool = True
    export_analysis_on_failure: bool = False
    export_best_candidate_analysis: bool = False
    enableStructureFallback: bool = False
    max_drop_ratio_for_partial: float = 0.05
    max_drop_tons_ratio_for_partial: float = 0.08
    max_drop_count_for_partial: int = 20
    min_scheduled_orders_for_partial: int = 100
    min_scheduled_tons_for_partial: float = 1000.0
    structure_fallback_slot_buffer: int = 4
    structure_fallback_risk_boost: float = 1.5
    structure_fallback_min_real_schedule_ratio: float = 0.55
    fast_fail_on_bad_slots: bool = False
    fast_fail_unroutable_slot_threshold: int = 10
    fast_fail_slot_order_count_threshold: int = 28
    fast_fail_slot_coverage_threshold: float = 0.15
    fast_fail_slot_zero_degree_threshold: int = 6
    fast_fail_topn_slots: int = 3
    allow_fallback: bool = False
    allow_legacy_fallback: bool = False
    validation_mode: bool = False
    production_compatibility_mode: bool = False
    enable_tiered_objective: bool = True
    min_real_schedule_ratio: float = 0.90
    max_virtual_chain: int = 5
    max_unbridgeable_drop_ratio: float = 0.08
    lot_max_tons: float = 120.0
    sparse_k_same_group: int = 10
    sparse_k_same_thickness: int = 8
    sparse_k_cross_group: int = 6
    sparse_k_due_tight: int = 6
    template_min_out_degree: int = 1
    template_min_in_degree: int = 1
    template_top_k: int = 40
    global_prune_max_pairs_per_from: int = 0
    max_routes_per_slot: int = 6
    template_health_sparse_ratio: float = 0.08
    template_health_zero_degree_ratio: float = 0.15
    semantic_fallback_rounds: int = 2
    scale_down_keep_steps: Tuple[int, ...] = (520, 320, 160)
    strict_template_edges: bool = True
    min_campaign_slots: int = 6
    max_campaign_slots: int = 14
    big_roll_slot_soft_order_cap: int = 20
    small_roll_slot_soft_order_cap: int = 24
    big_roll_slot_hard_order_cap: int = 0
    small_roll_slot_hard_order_cap: int = 0
    strict_virtual_width_levels: bool = False
    physical_reverse_step_mode: bool = True
    min_master_solve_seconds: float = 20.0
    max_master_solve_seconds: float = 180.0
    min_skeleton_solve_seconds: float = 2.0
    max_skeleton_solve_seconds: float = 20.0
    dual_width_anchor: float = 1800.0
    # Set Packing Master (Macro-Block Assembly) architecture
    use_set_packing_master: bool = False  # Set to True to enable new Set Packing approach
    # -------------------------------------------------------------------------
    # Constructive LNS path (new solver strategy)
    # main_solver_strategy controls which master path is used:
    #   "joint_master"        - original production joint model (default)
    #   "constructive_lns"    - new ALNS-driven constructive path
    # -------------------------------------------------------------------------
    main_solver_strategy: str = "joint_master"
    constructive_lns_rounds: int = 60          # Number of ALNS destruction/repair rounds
    constructive_destroy_ratio_min: float = 0.20  # Min fraction of segment orders to destroy
    constructive_destroy_ratio_max: float = 0.35  # Max fraction of segment orders to destroy
    constructive_subproblem_max_orders: int = 40  # Max orders in a local CP-SAT subproblem
    constructive_enable_cp_sat_repair: bool = True  # Enable CP-SAT repair in LNS repair phase
    # ---- LNS Early Stop Configuration ----
    lns_early_stop_no_improve_rounds: int = 3      # Consecutive non-improving rounds to trigger early stop
    lns_max_total_rounds: int = 10                  # Maximum total rounds allowed
    lns_min_rounds_before_early_stop: int = 4      # Minimum rounds before early stop is allowed
    # ---- Bridge edge policy for constructive_lns path ----
    # Route RB (mainline): allow REAL_BRIDGE_EDGE, disable VIRTUAL_BRIDGE_EDGE
    #   - allow_real_bridge_edge_in_constructive = True  (mainline default)
    #   - allow_virtual_bridge_edge_in_constructive = False
    # Route C (baseline): disable ALL bridge edges in constructive
    #   - allow_real_bridge_edge_in_constructive = False
    #   - allow_virtual_bridge_edge_in_constructive = False
    allow_virtual_bridge_edge_in_constructive: bool = False  # False = no virtual bridge in mainline
    allow_real_bridge_edge_in_constructive: bool = False     # False = direct_only (Route C), True = real-bridge-frontload (Route RB)
    bridge_expansion_mode: str = "disabled"  # "disabled" = no virtual bridge expansion (mainline)
    # ---- Virtual Bridge Family Edge: Guarded Frontload (受控前移实验线) ----
    # Only applicable when allow_virtual_bridge_edge_in_constructive = True AND
    # the edge type is VIRTUAL_BRIDGE_FAMILY_EDGE (NOT legacy VIRTUAL_BRIDGE_EDGE).
    # Legacy virtual exact expansion remains disabled.
    # Dual-pool: global pool = strict (greedy), repair pool = wider (ALNS + local rebuild).
    virtual_family_frontload_enabled: bool = False  # Master switch for guarded family frontload
    virtual_family_frontload_global_topk_per_from: int = 3  # Top-K per (from_order_id, line, bridge_family)
    virtual_family_frontload_global_max_edges_total: int = 360  # Global cap on family edges for greedy
    virtual_family_frontload_repair_max_edges_total: int = 900  # Cap for repair pool (ALNS + local rebuild)
    virtual_family_frontload_allowed_families: list[str] = field(default_factory=lambda: ["WIDTH_GROUP", "THICKNESS", "GROUP_TRANSITION"])
    virtual_family_frontload_max_bridge_count: int = 2  # Max bridge_count for frontload eligibility
    virtual_family_frontload_only_when_underfill_or_drop_pressure: bool = True  # Context-aware gating
    virtual_family_frontload_min_block_tons: float = 80.0
    virtual_family_frontload_max_block_tons: float = 450.0
    virtual_family_frontload_alns_only_extra_topk: int = 4  # Extra top-k for ALNS repair neighborhood
    virtual_family_frontload_local_cpsat_only: bool = False  # If True, global constructive gets 0 family edges
    virtual_family_frontload_global_penalty: float = 100.0  # Penalty added to family edge score in greedy
    virtual_family_frontload_local_penalty: float = 70.0    # Penalty in local CP-SAT objective
    virtual_family_frontload_require_family_budget: bool = True  # Enforce budget limits in greedy
    virtual_family_budget_per_line: int = 4    # Max family edges per line in global constructive
    virtual_family_budget_per_segment: int = 2  # Max family edges per segment in global constructive
    # Repair-only bridge policy. Initial constructive and initial cutter remain
    # direct-only; these switches apply only to underfilled reconstruction.
    repair_only_real_bridge_enabled: bool = True
    repair_only_virtual_bridge_enabled: bool = False
    repair_only_virtual_bridge_pilot_enabled: bool = False
    virtual_bridge_pilot_max_blocks_per_run: int = 15
    virtual_bridge_pilot_max_per_block: int = 1
    virtual_bridge_pilot_max_virtual_tons: float = 30.0
    virtual_bridge_pilot_penalty: float = 1000000.0
    virtual_bridge_pilot_only_when_endpoint_class: list[str] = field(default_factory=lambda: ["HAS_ENDPOINT_EDGE", "BAND_TOO_NARROW"])
    virtual_bridge_pilot_only_when_dominant_fail: list[str] = field(default_factory=lambda: ["THICKNESS_RULE_FAIL", "WIDTH_RULE_FAIL", "GROUP_SWITCH_FAIL", "MULTI_RULE_FAIL"])
    repair_only_bridge_max_per_segment: int = 1
    repair_only_bridge_cost_penalty: float = 100000.0
    repair_bridge_left_band_k: int = 3
    repair_bridge_right_band_k: int = 3
    repair_bridge_band_max_pairs_per_split: int = 9
    repair_bridge_left_trim_max: int = 2
    repair_bridge_right_trim_max: int = 2
    repair_bridge_endpoint_adjustment_limit_per_split: int = 9
    repair_bridge_adjustment_enable_left_trim: bool = True
    repair_bridge_adjustment_enable_right_trim: bool = True
    repair_bridge_adjustment_enable_swap: bool = False
    repair_bridge_ton_rescue_max_neighbor_blocks: int = 6
    repair_bridge_ton_rescue_enable_backward: bool = True
    repair_bridge_ton_rescue_enable_forward: bool = True
    repair_bridge_ton_rescue_enable_bidirectional: bool = True
    repair_bridge_ton_rescue_max_orders_per_window: int = 50
    repair_bridge_ton_rescue_max_failed_windows_after_min: int = 2
    # ---- Tail rebalancing: rescue underfilled tail segments by pullback ----
    tail_rebalance_enabled: bool = True
    tail_rebalance_max_pullback_orders: int = 8
    tail_rebalance_max_pullback_tons10: int = 2500   # 250.0 tons (ton10 = tons * 10)
    tail_rebalance_accept_if_prev_stays_above_min: bool = True
    # ---- Small roll dual-order reserve: prevent big_roll from monopolizing dual orders ----
    small_roll_dual_reserve_enabled: bool = True
    small_roll_dual_reserve_penalty: int = 15
    # ---- Small roll dual-order reserve bucket: hard-prevent big_roll from taking top dual candidates ----
    small_roll_dual_reserve_bucket_enabled: bool = True
    small_roll_dual_reserve_bucket_ratio: float = 0.35   # 取 top 35% dual orders
    small_roll_dual_reserve_bucket_max_orders: int = 80  # 最多保留 80 个
    # ---- Small roll seed-first: prioritize small_roll chain building before big_roll consumes reserve ----
    small_roll_seed_first_enabled: bool = True            # True = small_roll goes first with reserve bucket
    small_roll_seed_min_orders: int = 20                  # Min orders small_roll must place in seed phase
    small_roll_seed_min_tons10: int = 5000               # Min tons (ton10=tons*10) small_roll must place
    # ---- Small roll dual-order reserve QUOTA: balanced allocation instead of "lock all" ----
    # Quota enables a floor (min) AND a ceiling (max) for small_roll's priority on dual orders.
    # Once small_roll reaches max quota, remaining reserve bucket is released to big_roll.
    small_roll_dual_reserve_quota_enabled: bool = True   # Enable quota-based balanced allocation
    small_roll_dual_reserve_quota_min_orders: int = 25    # Min orders small_roll gets priority on
    small_roll_dual_reserve_quota_min_tons10: int = 6000  # Min tons small_roll gets priority on
    small_roll_dual_reserve_quota_max_orders: int = 60    # Max orders small_roll can lock (ceiling)
    small_roll_dual_reserve_quota_max_tons10: int = 14000  # Max tons small_roll can lock (ceiling)
    # ---- Big roll release: after small_roll seed phase, release remaining quota to big_roll ----
    big_roll_dual_release_after_small_seed: bool = True   # Release remaining quota to big_roll after seed
    # ---- Tail Repair Budget: limit search scope to reduce runtime ----
    max_tail_repair_windows_per_line: int = 12          # Max repair windows per line (big_roll/small_roll)
    max_tail_repair_windows_total: int = 24             # Max repair windows across all lines
    max_recut_cutpoints_per_window: int = 12            # Max cut points to test per recut window
    max_fill_candidates_per_tail: int = 8               # Max dropped fill candidates per underfilled tail
    tail_repair_gap_to_min_limit: float = 180.0        # Only attempt repair when gap_to_min <= this value
    tail_fill_from_dropped_enabled: bool = True          # Enable TAIL_FILL_FROM_DROPPED chained after SHIFT/RECUT
    tail_fill_gap_to_min_limit: float = 220.0           # Max gap for fill attempt (partial progress allowed)
    tail_fill_accept_partial_progress: bool = True      # Accept fill if it narrows gap even if tail still underfilled
    tail_fill_max_inserts_per_tail: int = 2             # Max consecutive fill inserts per tail (2 = 2-pass fill)
    tail_fill_second_pass_gap_limit: float = 30.0       # Only do 2nd fill if gap_to_min <= this after 1st fill
    # ---- Block Generator Configuration (block_first_guarded_search) ----
    block_generator_target_blocks: int = 2000
    block_generator_time_limit_seconds: float = 15.0
    block_generator_max_blocks_per_line: int = 30
    block_generator_max_blocks_total: int = 80
    block_generator_max_seed_per_bucket: int = 8
    block_generator_target_tons_min: float = 200.0
    block_generator_target_tons_target: float = 700.0
    block_generator_target_tons_max: float = 1200.0
    # ---- Candidate block (pool) threshold: looser than target_tons_* ----
    # Blocks between candidate_tons_min and target_tons_min are accepted
    # into the candidate pool as small_candidate blocks, but are penalized
    # by block_master to discourage them as final campaign blocks.
    block_generator_candidate_tons_min: float = 200.0
    block_generator_candidate_tons_target: float = 700.0
    block_generator_candidate_tons_max: float = 2000.0
    block_generator_max_orders_per_block: int = 30
    block_generator_allow_guarded_family: bool = True
    block_generator_allow_real_bridge: bool = True
    block_generator_allow_mixed_bridge_potential: bool = True
    block_generator_max_family_edges_per_block: int = 3
    block_generator_max_real_bridge_edges_per_block: int = 5
    block_generator_max_bridge_count_per_block: int = 2
    # ---- Directional Clustering Weights ----
    directional_cluster_width_weight: float = 1.0
    directional_cluster_thickness_weight: float = 1.0
    directional_cluster_temp_weight: float = 0.8
    directional_cluster_group_weight: float = 1.2
    directional_cluster_due_weight: float = 0.6
    directional_cluster_tons_fill_weight: float = 1.0
    directional_cluster_real_bridge_bonus: float = 0.8
    directional_cluster_guarded_family_bonus: float = 0.5
    directional_cluster_mixed_bridge_potential_bonus: float = 0.3
    # ---- Block ALNS Configuration ----
    block_alns_enabled: bool = True
    block_alns_rounds: int = 10
    block_alns_early_stop_no_improve_rounds: int = 2
    block_alns_swap_enabled: bool = True
    block_alns_replace_enabled: bool = True
    block_alns_split_enabled: bool = True
    block_alns_merge_enabled: bool = True
    block_alns_boundary_rebalance_enabled: bool = True
    block_alns_internal_rebuild_enabled: bool = True
    block_alns_accept_threshold: float = 0.0
    # ---- Block Master Configuration ----
    block_master_slot_buffer: int = 2
    block_master_greedy: bool = True
    block_master_max_conflict_skip: int = 5
    block_master_prefer_quality_score: bool = True
    # ---- Mixed Bridge (block-internal only) ----
    mixed_bridge_in_block_enabled: bool = True
    mixed_bridge_allowed_forms: list[str] = field(default_factory=lambda: ["REAL_TO_GUARDED", "GUARDED_TO_REAL"])
    mixed_bridge_allowed_hotspots: list[str] = field(default_factory=lambda: ["underfill", "group_switch", "bridge_dependency", "width_tension"])
    mixed_bridge_max_attempts_per_block: int = 10
    # ---- Virtual slab shadow/formal contract ----
    # Shadow mode estimates virtual bridge/fill benefits without writing virtual
    # rows into official schedules. Formal mode is intentionally off by default.
    virtual_enabled: bool = True
    virtual_shadow_mode_enabled: bool = True
    virtual_formal_enabled: bool = False
    virtual_budget_total_tons: float = 500.0
    virtual_budget_per_campaign_tons: float = 80.0
    virtual_max_count_per_campaign: int = 2
    virtual_max_consecutive_chain: int = 5
    virtual_penalty_per_piece: float = 1000.0
    virtual_penalty_per_ton: float = 20.0
    virtual_bridge_only_for_allowed_gaps: bool = True
    virtual_allowed_lines: list[str] = field(default_factory=lambda: ["big_roll", "small_roll"])
    virtual_allowed_steel_groups: list[str] = field(default_factory=list)
    penalty_virtual_piece_count: float = 1000.0
    penalty_virtual_tons: float = 20.0
    penalty_virtual_bridge_segments: float = 500.0
    penalty_consecutive_virtual_chain: float = 2000.0
    reward_drop_reduction_if_virtual: float = 800.0
    reward_hard_violation_elimination_if_virtual: float = 1500.0

