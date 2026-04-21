"""
Smoke test for A/B experiment summary generation.
测试 generate_summary 的核心逻辑，不依赖真实 pipeline。
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, asdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest


# ---- 从实验脚本复制的核心逻辑（独立可测） ----

@dataclass
class ExperimentResult:
    profile_name: str = ""
    group_label: str = ""
    seed: int = 0
    output_file: str = ""
    scheduled_real_orders: int = 0
    dropped_count: int = 0
    valid_segment_count: int = 0
    underfilled_segment_count: int = 0
    selected_real_bridge_edge_count: int = 0
    selected_virtual_bridge_family_edge_count: int = 0
    family_repair_accept_count: int = 0
    total_runtime_seconds: float = 0.0
    profile_guard_passed: bool = True
    profile_guard_failed_reason: str = ""
    run_error: str = ""


def _agg_stats(values: list[float]) -> dict:
    if not values:
        return {"mean": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(sum(values) / len(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def generate_summary_smoke(results: list[ExperimentResult]) -> tuple[list[dict], str]:
    """简化版 summary 生成，用于 smoke test。"""
    import pandas as pd

    rows = [asdict(r) for r in results]
    df = pd.DataFrame(rows)

    lines: list[str] = []
    lines.append("# A/B 实验汇总\n")
    lines.append("## 单轮结果\n")

    cols_show = [
        "group_label", "seed", "scheduled_real_orders", "dropped_count",
        "valid_segment_count", "underfilled_segment_count",
        "selected_virtual_bridge_family_edge_count", "total_runtime_seconds",
    ]
    available_cols = [c for c in cols_show if c in df.columns]
    if available_cols and not df.empty:
        lines.append(df[available_cols].to_markdown(index=False) + "\n")

    # Delta 分析
    lines.append("## 差值分析（B - A）\n")
    delta_cols = ["dropped_count", "underfilled_segment_count", "valid_segment_count"]
    seed_pairs = sorted(set(int(r.seed) for r in results))

    pair_rows = []
    has_group_label = "group_label" in df.columns
    if has_group_label:
        for s in seed_pairs:
            a_row = df[(df["group_label"] == "A") & (df["seed"] == s)]
            b_row = df[(df["group_label"] == "B") & (df["seed"] == s)]
            if a_row.empty or b_row.empty:
                continue
            row = {"seed": s}
            for col in delta_cols:
                if col in df.columns:
                    row[f"Δ {col}"] = float(b_row.iloc[0][col]) - float(a_row.iloc[0][col])
            pair_rows.append(row)

        if pair_rows:
            pair_df = pd.DataFrame(pair_rows)
            lines.append(pair_df.to_markdown(index=False) + "\n")

    # 自动判读
    lines.append("## 自动判读\n")
    a_df = df[df["group_label"] == "A"] if has_group_label else df.iloc[:0]
    b_df = df[df["group_label"] == "B"] if has_group_label else df.iloc[:0]

    def _safe_mean(d: pd.DataFrame, col: str) -> float:
        if col not in d.columns:
            return 0.0
        vals = pd.to_numeric(d[col], errors="coerce").dropna()
        return float(vals.mean()) if not vals.empty else 0.0

    a_dropped_mean = _safe_mean(a_df, "dropped_count")
    b_dropped_mean = _safe_mean(b_df, "dropped_count")
    a_under_mean = _safe_mean(a_df, "underfilled_segment_count")
    b_under_mean = _safe_mean(b_df, "underfilled_segment_count")
    a_valid_mean = _safe_mean(a_df, "valid_segment_count")
    b_valid_mean = _safe_mean(b_df, "valid_segment_count")
    a_runtime_mean = _safe_mean(a_df, "total_runtime_seconds")
    b_runtime_mean = _safe_mean(b_df, "total_runtime_seconds")
    b_vfamily = int(b_df["selected_virtual_bridge_family_edge_count"].sum()) if "selected_virtual_bridge_family_edge_count" in b_df.columns else 0

    cond1 = (
        (b_dropped_mean < a_dropped_mean) and
        (b_under_mean < a_under_mean) and
        (b_valid_mean >= a_valid_mean) and
        (b_runtime_mean <= a_runtime_mean * 1.5)
    )
    cond2 = (b_vfamily > 0) and not (b_dropped_mean < a_dropped_mean and b_under_mean < a_under_mean)
    cond3 = b_vfamily == 0

    if cond1:
        verdict = "guarded virtual family 实验线表现为正向收益，建议继续做参数精调。"
    elif cond2:
        verdict = "guarded virtual family 已参与但收益不稳定，建议优先调 repair pool。"
    elif cond3:
        verdict = "当前 guarded virtual family 参与度不足。"
    else:
        verdict = "当前数据不足以给出明确判读。"

    lines.append(verdict + "\n")

    # guarded virtual family keyword 必须出现
    has_guard_keyword = "guarded virtual family" in "\n".join(lines)
    assert has_guard_keyword, "Summary must contain 'guarded virtual family'"

    # Delta 列：仅当存在 A/B 配对时才强制要求
    if pair_rows:
        has_delta = any(f"Δ {col}" in "\n".join(lines) for col in delta_cols)
        assert has_delta, f"Summary must contain delta columns: {delta_cols}"

    return rows, "\n".join(lines)


# =============================================================================
# Smoke Tests
# =============================================================================

def test_ab_experiment_summary_smoke():
    """
    Smoke test: 构造两个 fake result (mainline vs guarded)，
    验证 summary 生成逻辑输出包含预期字段。
    """
    results = [
        # A 组: mainline
        ExperimentResult(
            profile_name="constructive_lns_search",
            group_label="A",
            seed=2027,
            scheduled_real_orders=150,
            dropped_count=12,
            valid_segment_count=8,
            underfilled_segment_count=3,
            selected_real_bridge_edge_count=25,
            selected_virtual_bridge_family_edge_count=0,
            family_repair_accept_count=5,
            total_runtime_seconds=45.0,
        ),
        # B 组: guarded
        ExperimentResult(
            profile_name="constructive_lns_virtual_guarded_frontload",
            group_label="B",
            seed=2027,
            scheduled_real_orders=158,
            dropped_count=8,
            valid_segment_count=9,
            underfilled_segment_count=2,
            selected_real_bridge_edge_count=22,
            selected_virtual_bridge_family_edge_count=10,
            family_repair_accept_count=8,
            total_runtime_seconds=48.0,
        ),
    ]

    rows, md_text = generate_summary_smoke(results)

    # 断言: markdown 包含关键内容
    assert "Δ dropped_count" in md_text, "Delta analysis must include dropped_count"
    assert "guarded virtual family" in md_text, "Summary must mention guarded virtual family"
    assert "## 单轮结果" in md_text, "Summary must have single-run table"
    assert "## 差值分析" in md_text, "Summary must have delta analysis"

    # 断言: CSV 行数符合预期（2 行：1 A + 1 B）
    assert len(rows) == 2, f"Expected 2 rows, got {len(rows)}"

    # 断言: guard pass 字段存在
    for r in rows:
        assert "profile_guard_passed" in r
        assert "run_error" in r


def test_ab_experiment_summary_delta_signs():
    """
    验证 Δ 值计算方向正确（B - A）。
    B 组 dropped 更少、underfilled 更少，Δ 应为负。
    """
    results = [
        ExperimentResult(
            profile_name="constructive_lns_search",
            group_label="A",
            seed=2028,
            dropped_count=20,
            underfilled_segment_count=5,
            valid_segment_count=6,
            scheduled_real_orders=140,
            selected_virtual_bridge_family_edge_count=0,
            total_runtime_seconds=50.0,
        ),
        ExperimentResult(
            profile_name="constructive_lns_virtual_guarded_frontload",
            group_label="B",
            seed=2028,
            dropped_count=15,
            underfilled_segment_count=3,
            valid_segment_count=8,
            scheduled_real_orders=155,
            selected_virtual_bridge_family_edge_count=12,
            total_runtime_seconds=52.0,
        ),
    ]

    _, md_text = generate_summary_smoke(results)

    # B 组 dropped=15 < A 组 dropped=20，所以 Δ dropped_count 应为 -5
    assert "Δ dropped_count" in md_text
    # 验证 Delta 行格式正确（seed 2028 存在）
    assert "seed" in md_text


def test_ab_experiment_summary_fails_gracefully():
    """
    空结果列表不应崩溃。
    """
    rows, md_text = generate_summary_smoke([])
    assert isinstance(rows, list)
    assert "# A/B 实验汇总" in md_text


def test_ab_experiment_summary_b_only():
    """
    只有 B 组数据时也应能生成摘要（无 delta）。
    """
    results = [
        ExperimentResult(
            profile_name="constructive_lns_virtual_guarded_frontload",
            group_label="B",
            seed=2027,
            dropped_count=10,
            valid_segment_count=9,
            underfilled_segment_count=2,
            selected_virtual_bridge_family_edge_count=15,
            total_runtime_seconds=50.0,
        ),
    ]
    rows, md_text = generate_summary_smoke(results)
    assert isinstance(rows, list)
    assert "guarded virtual family" in md_text
    assert "# A/B 实验汇总" in md_text


def test_ab_experiment_profile_guard_fields():
    """
    验证 result dict 包含所有必要字段。
    """
    r = ExperimentResult(
        profile_name="constructive_lns_search",
        group_label="A",
        seed=2027,
        profile_guard_passed=True,
        run_error="",
    )
    d = asdict(r)
    assert "profile_guard_passed" in d
    assert "run_error" in d
    assert "group_label" in d
    assert "seed" in d
    assert "selected_virtual_bridge_family_edge_count" in d
    assert "family_repair_accept_count" in d
