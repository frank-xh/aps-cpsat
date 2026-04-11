from __future__ import annotations

from aps_cp_sat.config.model_config import ModelConfig
from aps_cp_sat.config.planner_config import PlannerConfig
from aps_cp_sat.config.rule_config import RuleConfig
from aps_cp_sat.config.score_config import ScoreConfig


def legacy_to_planner_config(legacy_constraint_config, *, model: ModelConfig | None = None) -> PlannerConfig:
    """
    兼容层：将旧 ConstraintConfig 显式转换为新 PlannerConfig。
    新架构默认不依赖该函数，只有 legacy 适配场景使用。
    """
    th = legacy_constraint_config.thresholds
    w = legacy_constraint_config.weights
    rule = RuleConfig(
        min_temp_overlap_real_real=float(th.min_temp_overlap_real_real),
        max_width_drop=float(th.max_width_drop),
        max_width_rise_physical_step=float(th.max_width_rise_physical_step),
        virtual_reverse_attach_max_mm=float(getattr(th, "virtual_reverse_attach_max_mm", 250.0)),
        max_logical_reverse_per_campaign=int(th.max_logical_reverse_per_campaign),
        campaign_ton_min=float(th.campaign_ton_min),
        campaign_ton_max=float(th.campaign_ton_max),
        campaign_ton_target=float(th.campaign_ton_target),
        virtual_chain_penalty_threshold=int(th.virtual_chain_penalty_threshold),
        temp_margin_target=float(th.temp_margin_target),
        virtual_ton_ratio_num=int(th.virtual_ton_ratio_num),
        virtual_ton_ratio_den=int(th.virtual_ton_ratio_den),
    )
    score = ScoreConfig(
        seq_unique=int(w.seq_unique),
        seq_contiguous=int(w.seq_contiguous),
        seq_start=int(w.seq_start),
        roll_line_mismatch=int(w.roll_line_mismatch),
        ton_over=int(w.ton_over),
        ton_under=int(w.ton_under),
        temp_shortage=int(w.temp_shortage),
        width_violation=int(w.width_violation),
        thick_violation=int(w.thick_violation),
        non_pc_switch=int(w.non_pc_switch),
        virtual_chain_excess=int(w.virtual_chain_excess),
        unassigned_real=int(w.unassigned_real),
        virtual_ratio=int(w.virtual_ratio),
        pure_virtual_campaign=int(w.pure_virtual_campaign),
        virtual_use=int(w.virtual_use),
        width_smooth=int(w.width_smooth),
        thick_smooth=int(w.thick_smooth),
        temp_margin=int(w.temp_margin),
        ton_target=int(w.ton_target),
        ton_target_value=float(th.campaign_ton_target),
    )
    return PlannerConfig(rule=rule, score=score, model=model or ModelConfig())
