from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TransitionPruneSummary:
    line: str
    pruned_unbridgeable: int
    pruned_topk: int
    pruned_degree: int
    kept_templates: int
