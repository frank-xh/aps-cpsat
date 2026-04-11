from __future__ import annotations

from dataclasses import dataclass

from aps_cp_sat.config import PlannerConfig


@dataclass(frozen=True)
class MasterSolvePolicy:
    """Explicit fallback strategy for the production joint-master path."""

    enable_semantic_fallback: bool = True
    enable_scale_down_fallback: bool = True
    enable_legacy_fallback: bool = False
    production_allow_legacy_fallback: bool = False

    @classmethod
    def from_config(cls, cfg: PlannerConfig) -> "MasterSolvePolicy":
        return cls(
            enable_semantic_fallback=bool(
                getattr(cfg.model, "enableSemanticFallback", getattr(cfg.model, "allow_fallback", False))
            ),
            enable_scale_down_fallback=bool(
                getattr(cfg.model, "enableScaleDownFallback", getattr(cfg.model, "allow_fallback", False))
            ),
            enable_legacy_fallback=bool(
                getattr(cfg.model, "enableLegacyFallback", getattr(cfg.model, "allow_legacy_fallback", False))
            ),
            production_allow_legacy_fallback=bool(
                getattr(
                    cfg.model,
                    "productionAllowLegacyFallback",
                    getattr(cfg.model, "production_compatibility_mode", False),
                )
            ),
        )

    def semantic_enabled(self) -> bool:
        return bool(self.enable_semantic_fallback)

    def scale_down_enabled(self) -> bool:
        return bool(self.enable_scale_down_fallback)

    def legacy_enabled(self, *, validation_mode: bool = False) -> bool:
        if bool(validation_mode):
            return False
        return bool(self.enable_legacy_fallback and self.production_allow_legacy_fallback)
