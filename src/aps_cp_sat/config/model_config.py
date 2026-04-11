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
