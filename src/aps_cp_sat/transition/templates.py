from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import List

import pandas as pd

from aps_cp_sat.config import PlannerConfig
from aps_cp_sat.transition.bridge_rules import (
    _txt,
)
from aps_cp_sat.transition.template_builder import _build_line_templates, _line_feasible
from aps_cp_sat.transition.template_pruning import TransitionPruneSummary


@dataclass(frozen=True)
class TransitionTemplateSummary:
    line: str
    nodes: int
    candidate_pairs: int
    feasible_pairs: int
    max_bridge_used: int


def build_transition_templates(orders_df: pd.DataFrame, cfg: PlannerConfig) -> dict[str, object]:
    """
    Transition template facade.

    This module keeps the historical external interface stable while delegating:
    - feasibility / materialization to `template_builder`
    - cost composition to `template_cost`
    - prune summary types to `template_pruning`
    """
    if orders_df.empty:
        return {"templates": pd.DataFrame(), "summaries": [], "prune_summaries": [], "build_debug": []}

    total_t0 = perf_counter()
    preprocess_seconds = 0.0
    line_partition_seconds = 0.0
    transition_pack_build_seconds = 0.0
    diagnostics_build_seconds = 0.0
    pair_scan_total = 0.0
    bridge_check_total = 0.0
    prune_total = 0.0
    all_tpl = []
    summaries: List[TransitionTemplateSummary] = []
    prune_summaries: List[TransitionPruneSummary] = []
    build_debug: list[dict[str, object]] = []
    for line in ["big_roll", "small_roll"]:
        line_t0 = perf_counter()
        partition_t0 = perf_counter()
        d = _line_feasible(orders_df, line)
        line_partition_seconds += perf_counter() - partition_t0
        build_t0 = perf_counter()
        tpl, stats = _build_line_templates(d, line, cfg)
        line_build_seconds = perf_counter() - build_t0
        all_tpl.append(tpl)
        pair_scan_total += float(stats.get("template_pair_scan_seconds", 0.0))
        bridge_check_total += float(stats.get("bridge_check_seconds", 0.0))
        prune_total += float(stats.get("template_prune_seconds", 0.0))
        build_debug.append(
            {
                "line": line,
                "spec_views_reused": bool(stats.get("spec_views_reused", False)),
                "spec_views_build_count": int(stats.get("spec_views_build_count", 0)),
                "candidate_pairs": int(stats.get("candidate_pairs", 0)),
                "kept_templates": int(stats.get("kept_templates", 0)),
                "avg_virtual_count": float(stats.get("avg_virtual_count", 0.0)),
                "max_virtual_count": int(stats.get("max_virtual_count", 0)),
                "direct_edge_count": int(stats.get("direct_edge_count", 0)),
                "real_bridge_edge_count": int(stats.get("real_bridge_edge_count", 0)),
                "virtual_bridge_edge_count": int(stats.get("virtual_bridge_edge_count", 0)),
                "rejected_by_chain_limit": int(stats.get("rejected_by_chain_limit", 0)),
                "rejected_by_no_legal_next_step": int(stats.get("rejected_by_no_legal_next_step", 0)),
                "rejected_by_later_step_not_feasible": int(stats.get("rejected_by_later_step_not_feasible", 0)),
                "template_pair_scan_seconds": float(stats.get("template_pair_scan_seconds", 0.0)),
                "bridge_check_seconds": float(stats.get("bridge_check_seconds", 0.0)),
                "template_prune_seconds": float(stats.get("template_prune_seconds", 0.0)),
                "line_template_build_seconds": round(float(line_build_seconds), 6),
            }
        )
        if line == "big_roll":
            build_debug[-1]["big_roll_template_build_seconds"] = round(float(line_build_seconds), 6)
        else:
            build_debug[-1]["small_roll_template_build_seconds"] = round(float(line_build_seconds), 6)
        summaries.append(
            TransitionTemplateSummary(
                line=line,
                nodes=int(len(d)),
                candidate_pairs=int(max(0, len(d) * max(0, len(d) - 1))),
                feasible_pairs=int(len(tpl)),
                max_bridge_used=int(tpl["bridge_count"].max()) if not tpl.empty else 0,
            )
        )
        prune_summaries.append(
            TransitionPruneSummary(
                line=line,
                pruned_unbridgeable=int(stats.get("pruned_unbridgeable", 0)),
                pruned_topk=int(stats.get("pruned_topk", 0)),
                pruned_degree=int(stats.get("pruned_degree", 0)),
                kept_templates=int(stats.get("kept_templates", 0)),
            )
        )
        transition_pack_build_seconds += perf_counter() - line_t0

    diag_t0 = perf_counter()
    out = pd.concat(all_tpl, ignore_index=True) if all_tpl else pd.DataFrame()
    diagnostics_build_seconds += perf_counter() - diag_t0
    total_seconds = perf_counter() - total_t0
    build_debug.append(
        {
            "line": "__all__",
            "preprocess_seconds": round(float(preprocess_seconds), 6),
            "line_partition_seconds": round(float(line_partition_seconds), 6),
            "template_pair_scan_seconds": round(float(pair_scan_total), 6),
            "bridge_check_seconds": round(float(bridge_check_total), 6),
            "template_prune_seconds": round(float(prune_total), 6),
            "transition_pack_build_seconds": round(float(transition_pack_build_seconds), 6),
            "diagnostics_build_seconds": round(float(diagnostics_build_seconds), 6),
            "template_build_seconds": round(float(total_seconds), 6),
        }
    )
    return {"templates": out, "summaries": summaries, "prune_summaries": prune_summaries, "build_debug": build_debug}
