from __future__ import annotations

import math
from dataclasses import replace
from typing import List

from aps_cp_sat.config import ModelConfig, PlannerConfig
from aps_cp_sat.model.solve_policy import MasterSolvePolicy


def _joint_profiles(cfg: PlannerConfig) -> List[dict]:
    base = [
        {"start_penalty": 450000, "low_slot_penalty": 120000, "ultra_low_slot_penalty": 450000, "time_scale": 0.45},
        {"start_penalty": 250000, "low_slot_penalty": 80000, "ultra_low_slot_penalty": 300000, "time_scale": 0.7},
        {"start_penalty": 120000, "low_slot_penalty": 40000, "ultra_low_slot_penalty": 180000, "time_scale": 0.85},
    ]
    k = max(1, min(len(base), int(cfg.model.master_profile_count)))
    return base[:k]


def _joint_seeds(cfg: PlannerConfig) -> List[int]:
    seeds = [2027, 2081, 2137, 2213]
    k = max(1, min(len(seeds), int(cfg.model.master_seed_count)))
    return seeds[:k]


def _cfg_with_model_overrides(cfg: PlannerConfig, **overrides) -> PlannerConfig:
    return PlannerConfig(rule=cfg.rule, score=cfg.score, model=replace(cfg.model, **overrides))


def _effective_global_prune_cap(cfg: PlannerConfig) -> int | None:
    raw = cfg.model.global_prune_max_pairs_per_from
    if raw is None:
        return None
    raw_int = int(raw)
    if raw_int <= 0:
        return None
    return raw_int


def _semantic_fallback_configs(cfg: PlannerConfig) -> List[PlannerConfig]:
    base_prune = _effective_global_prune_cap(cfg)
    base_time = float(cfg.model.time_limit_seconds)
    base_top_k = int(cfg.model.template_top_k)
    base_routes = int(cfg.model.max_routes_per_slot)
    if base_prune is None:
        prune_steps: List[int | None] = [None, None, None]
    else:
        prune_steps = [max(6, base_prune + 2), max(8, base_prune + 4), None]
    cfgs: List[PlannerConfig] = []
    seen: set[tuple[int | None, int, int, float]] = set()
    max_rounds = max(1, int(cfg.model.semantic_fallback_rounds))
    for idx, prune_cap in enumerate(prune_steps[:max_rounds], start=1):
        time_factor = 1.0 + 0.5 * idx
        key = (prune_cap, base_top_k + 20 * idx, base_routes + idx, round(base_time * time_factor, 3))
        if key in seen:
            continue
        seen.add(key)
        cfgs.append(
            _cfg_with_model_overrides(
                cfg,
                global_prune_max_pairs_per_from=prune_cap,
                template_top_k=base_top_k + 20 * idx,
                max_routes_per_slot=base_routes + idx,
                time_limit_seconds=base_time * time_factor,
                master_num_workers=1,
            )
        )
    return cfgs


def _precheck_relaxed_config(cfg: PlannerConfig) -> PlannerConfig:
    base_prune = _effective_global_prune_cap(cfg)
    next_prune = None if base_prune is None else max(10, base_prune + 6)
    return _cfg_with_model_overrides(
        cfg,
        global_prune_max_pairs_per_from=next_prune,
        template_top_k=max(120, int(cfg.model.template_top_k) + 60),
        max_routes_per_slot=max(10, int(cfg.model.max_routes_per_slot) + 4),
    )


def _semantic_fallback_enabled(cfg: PlannerConfig) -> bool:
    return MasterSolvePolicy.from_config(cfg).semantic_enabled()


def _scale_down_fallback_enabled(cfg: PlannerConfig) -> bool:
    return MasterSolvePolicy.from_config(cfg).scale_down_enabled()


def _legacy_fallback_enabled(cfg: PlannerConfig) -> bool:
    return MasterSolvePolicy.from_config(cfg).legacy_enabled(validation_mode=bool(cfg.model.validation_mode))
