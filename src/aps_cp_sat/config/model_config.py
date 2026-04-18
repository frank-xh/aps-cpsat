from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class ModelConfig:
    profile_name: str = "default"
    max_orders: int = 720
    rounds: int = 4
    time_limit_seconds: float = 20.0
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
    # Route B: disable virtual bridge edge to prioritize official_exported=True
    allow_virtual_bridge_edge_in_constructive: bool = False  # False = only DIRECT_EDGE allowed
    allow_real_bridge_edge_in_constructive: bool = False     # False = direct_only mode (Route C)
    bridge_expansion_mode: str = "disabled"  # "disabled" = no virtual bridge expansion
    # Repair-only bridge policy. Initial constructive and initial cutter remain
    # direct-only; these switches apply only to underfilled reconstruction.
    repair_only_real_bridge_enabled: bool = True
    repair_only_virtual_bridge_enabled: bool = False
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
