from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True)
class RuleConfig:
    min_temp_overlap_real_real: float = 10.0
    max_width_drop: float = 250.0
    # real_step_max_mm: only for real->real adjacency
    real_reverse_step_max_mm: float = 20.0
    # virtual_reverse_attach_max_mm: only when the right-side coil is virtual
    # (real->virtual or virtual->virtual). It does not relax real->real.
    virtual_reverse_attach_max_mm: float = 250.0
    # compatibility alias retained for older helper code
    max_width_rise_physical_step: float = 20.0
    # same campaign reverse-width episode limit, counted as logical reverse
    # events per campaign rather than every virtual slab inside a bridge chain.
    max_logical_reverse_per_campaign: int = 5
    campaign_ton_min: float = 700.0
    campaign_ton_max: float = 2000.0
    campaign_ton_target: float = 1800.0
    virtual_chain_penalty_threshold: int = 5
    temp_margin_target: float = 30.0
    virtual_ton_ratio_num: int = 1
    virtual_ton_ratio_den: int = 5
    virtual_width_levels: tuple[float, ...] = (1000.0, 1250.0, 1500.0)
    virtual_thickness_levels: tuple[float, ...] = (0.6, 0.8, 1.0, 1.2, 1.5, 2.0)
    virtual_temp_min: float = 600.0
    virtual_temp_max: float = 900.0
    # Continuous feasible interval for virtual slab temperatures. Production
    # path no longer discretizes virtual temperature into step bands.
    virtual_tons: float = 20.0
    # max_bridge_chain: maximum number of virtual slabs inside one bridge_path
    max_virtual_chain: int = 5
