"""
experiment_summary.py - Block-first 实验线结果汇总与自动判读

提供 A/B 对比实验与参数收敛实验的结果汇总逻辑，包括：
- Raw Run Table 生成
- Aggregated by Profile 统计
- Paired Deltas 成对比较
- Automatic Interpretation 自动判读规则
- Stage Summary 参数收敛推荐

典型用法:
    from aps_cp_sat.model.experiment_summary import (
        build_ab_summary_markdown,
        aggregate_by_profile,
        paired_deltas,
        automatic_interpretation,
        build_param_sweep_summary_markdown,
        recommend_stage_config,
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

# =============================================================================
# 辅助函数
# =============================================================================


def _safe_mean(series: pd.Series) -> float:
    """安全计算均值，忽略 NaN。"""
    return float(series.mean()) if len(series) > 0 else 0.0


def _safe_min(series: pd.Series) -> float:
    """安全计算最小值，忽略 NaN。"""
    return float(series.min()) if len(series) > 0 else 0.0


def _safe_max(series: pd.Series) -> float:
    """安全计算最大值，忽略 NaN。"""
    return float(series.max()) if len(series) > 0 else 0.0


def _fmt(v: float, decimals: int = 2) -> str:
    """格式化浮点数。"""
    if pd.isna(v) or v is None:
        return "N/A"
    if isinstance(v, (int, bool)):
        return str(int(v))
    return f"{float(v):.{decimals}f}"


def _select_numeric(df: pd.DataFrame, col: str) -> pd.Series:
    """安全选择数值列，失败时返回全零序列。"""
    if col not in df.columns:
        return pd.Series(dtype=float)
    s = pd.to_numeric(df[col], errors="coerce")
    return s.fillna(0.0)


# =============================================================================
# A/B 实验汇总
# =============================================================================


def aggregate_by_profile(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    按 profile 聚合实验结果。

    Parameters
    ----------
    df : pd.DataFrame
        实验结果 DataFrame。

    Returns
    -------
    dict[profile_name -> dict[field -> float]]
        每个 profile 的聚合统计。
    """
    if df.empty:
        return {}

    profiles = df["profile_name"].unique().tolist()
    result: Dict[str, Dict[str, float]] = {}

    for profile in profiles:
        sub = df[df["profile_name"] == profile]
        if sub.empty:
            continue

        agg: Dict[str, float] = {}

        # 基础指标
        for col in ["dropped_count", "tail_underfilled_count", "scheduled_real_orders",
                     "total_runtime_seconds"]:
            if col in sub.columns:
                agg[f"mean_{col}"] = _safe_mean(_select_numeric(sub, col))
                agg[f"min_{col}"] = _safe_min(_select_numeric(sub, col))
                agg[f"max_{col}"] = _safe_max(_select_numeric(sub, col))

        # 桥接相关
        for col in ["selected_real_bridge_edge_count",
                     "selected_virtual_bridge_family_edge_count"]:
            if col in sub.columns:
                agg[f"mean_{col}"] = _safe_mean(_select_numeric(sub, col))

        # Block-first 关键
        for col in ["selected_blocks_count", "mixed_bridge_success_count",
                     "block_alns_rounds_accepted", "generated_blocks_total"]:
            if col in sub.columns:
                agg[f"mean_{col}"] = _safe_mean(_select_numeric(sub, col))

        # Block ALNS 接受率
        if "block_alns_rounds_attempted" in sub.columns and "block_alns_rounds_accepted" in sub.columns:
            attempted = _select_numeric(sub, "block_alns_rounds_attempted")
            accepted = _select_numeric(sub, "block_alns_rounds_accepted")
            total_attempted = attempted.sum()
            total_accepted = accepted.sum()
            agg["block_alns_accept_rate"] = (
                float(total_accepted / total_attempted) if total_attempted > 0 else 0.0
            )

        result[str(profile)] = agg

    return result


def paired_deltas(
    df: pd.DataFrame,
    group_a: str = "constructive_lns_search",
    group_b: str = "block_first_guarded_search",
    group_c: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """
    按 seed 做 A/B 成对比较（及可选 C 组比较）。

    Parameters
    ----------
    df : pd.DataFrame
        实验结果 DataFrame。
    group_a : str
        A 组 profile 名称。
    group_b : str
        B 组 profile 名称。
    group_c : str, optional
        C 组 profile 名称。

    Returns
    -------
    dict
        包含 B-A、C-A、B-C delta 的字典。
    """
    seeds = sorted(df["seed"].unique().tolist())
    deltas: Dict[str, Dict[str, List[float]]] = {
        "B_minus_A": {
            "dropped_count": [], "tail_underfilled_count": [],
            "scheduled_real_orders": [], "total_runtime_seconds": [],
            "mixed_bridge_success_count": [], "selected_blocks_count": [],
        },
    }

    if group_c:
        deltas["C_minus_A"] = {
            "dropped_count": [], "tail_underfilled_count": [],
            "scheduled_real_orders": [], "total_runtime_seconds": [],
            "mixed_bridge_success_count": [], "selected_blocks_count": [],
        }
        deltas["B_minus_C"] = {
            "dropped_count": [], "tail_underfilled_count": [],
            "scheduled_real_orders": [], "total_runtime_seconds": [],
            "mixed_bridge_success_count": [], "selected_blocks_count": [],
        }

    compare_cols = [
        "dropped_count", "tail_underfilled_count", "scheduled_real_orders",
        "total_runtime_seconds", "mixed_bridge_success_count", "selected_blocks_count",
    ]

    for seed in seeds:
        seed_df = df[df["seed"] == seed]

        row_a = seed_df[seed_df["profile_name"] == group_a]
        row_b = seed_df[seed_df["profile_name"] == group_b]

        if row_a.empty or row_b.empty:
            continue

        for col in compare_cols:
            if col not in row_a.columns or col not in row_b.columns:
                continue
            val_a = float(row_a[col].iloc[0]) if not pd.isna(row_a[col].iloc[0]) else 0.0
            val_b = float(row_b[col].iloc[0]) if not pd.isna(row_b[col].iloc[0]) else 0.0
            deltas["B_minus_A"][col].append(val_b - val_a)

        if group_c:
            row_c = seed_df[seed_df["profile_name"] == group_c]
            if not row_c.empty:
                for col in compare_cols:
                    if col not in row_c.columns:
                        continue
                    val_c = float(row_c[col].iloc[0]) if not pd.isna(row_c[col].iloc[0]) else 0.0
                    deltas["C_minus_A"][col].append(val_c - val_a)
                    deltas["B_minus_C"][col].append(val_b - val_c)

    # 聚合
    result: Dict[str, Dict[str, float]] = {}
    for pair_key, cols_dict in deltas.items():
        result[pair_key] = {}
        for col, values in cols_dict.items():
            if values:
                result[pair_key][f"{col}_mean"] = _safe_mean(pd.Series(values))
                result[pair_key][f"{col}_min"] = _safe_min(pd.Series(values))
                result[pair_key][f"{col}_max"] = _safe_max(pd.Series(values))

    return result


def automatic_interpretation(
    profile_stats: Dict[str, Dict[str, float]],
    paired: Dict[str, Dict[str, float]],
) -> str:
    """
    根据聚合统计和成对比较结果，自动生成判读结论。

    Parameters
    ----------
    profile_stats : dict
        aggregate_by_profile 返回的统计数据。
    paired : dict
        paired_deltas 返回的 delta 数据。

    Returns
    -------
    str
        Markdown 格式的判读结论段落。
    """
    group_a = "constructive_lns_search"
    group_b = "block_first_guarded_search"

    if group_a not in profile_stats or group_b not in profile_stats:
        return "\n> **判读结论**: 数据不足，无法进行自动判读。\n"

    stats_a = profile_stats.get(group_a, {})
    stats_b = profile_stats.get(group_b, {})

    # 提取关键指标
    a_dropped = stats_a.get("mean_dropped_count", 0.0)
    b_dropped = stats_b.get("mean_dropped_count", 0.0)
    a_tail = stats_a.get("mean_tail_underfilled_count", 0.0)
    b_tail = stats_b.get("mean_tail_underfilled_count", 0.0)
    a_scheduled = stats_a.get("mean_scheduled_real_orders", 0.0)
    b_scheduled = stats_b.get("mean_scheduled_real_orders", 0.0)
    a_runtime = stats_a.get("mean_total_runtime_seconds", 0.0)
    b_runtime = stats_b.get("mean_total_runtime_seconds", 0.0)

    b_selected_blocks = stats_b.get("mean_selected_blocks_count", 0.0)
    b_generated = stats_b.get("mean_generated_blocks_total", 0.0)
    b_alns_accepted = stats_b.get("mean_block_alns_rounds_accepted", 0.0)
    b_mixed_attempt = stats_b.get("mean_mixed_bridge_attempt_count", 0.0)
    b_mixed_success = stats_b.get("mean_mixed_bridge_success_count", 0.0)

    # ---- 规则 1: Block-first 明显正向 ----
    cond1_dropped = b_dropped < a_dropped
    cond1_tail = b_tail <= a_tail
    cond1_scheduled = b_scheduled >= a_scheduled
    cond1_runtime = b_runtime <= a_runtime * 1.6

    if cond1_dropped and cond1_tail and cond1_scheduled and cond1_runtime:
        return (
            "\n## Automatic Interpretation\n\n"
            "### 规则 1: Block-first 明显正向\n\n"
            "> **判读结论**: block-first 当前版本表现为正向收益，建议进入参数收敛阶段。\n\n"
            f"- A 组 (constructive_lns_search) mean dropped_count = {_fmt(a_dropped)}\n"
            f"- B 组 (block_first_guarded_search) mean dropped_count = {_fmt(b_dropped)}\n"
            f"- B 组 tail_underfilled_count ({_fmt(b_tail)}) <= A 组 ({_fmt(a_tail)})\n"
            f"- B 组 scheduled_real_orders ({_fmt(b_scheduled)}) >= A 组 ({_fmt(a_scheduled)})\n"
            f"- B 组 runtime ({_fmt(b_runtime)}s) <= A 组 runtime ({_fmt(a_runtime)}s) * 1.6\n\n"
            f"**建议**: 继续参数收敛阶段（Stage A: 块供给 → Stage B: 聚类权重 → Stage C: ALNS 配置）。\n"
        )

    # ---- 规则 2: Block-first 已使用块/混合桥接，但结果不稳 ----
    cond2_blocks = b_selected_blocks > 0
    cond2_mixed = b_mixed_attempt > 0
    cond2_unstable = not (cond1_dropped and cond1_tail)

    if cond2_blocks and cond2_mixed and cond2_unstable:
        return (
            "\n## Automatic Interpretation\n\n"
            "### 规则 2: Block-first 已使用块级与混合桥接，但结果不稳\n\n"
            "> **判读结论**: block-first 已经在使用块级与混合桥接能力，但收益不稳定，"
            "建议优先收敛 block generator 与 block ALNS 参数，而不是继续扩展 mixed bridge。\n\n"
            f"- B 组 mean selected_blocks_count = {_fmt(b_selected_blocks)}\n"
            f"- B 组 mean mixed_bridge_attempt_count = {_fmt(b_mixed_attempt)}\n"
            f"- B 组 mean mixed_bridge_success_count = {_fmt(b_mixed_success)}\n"
            f"- B 组 mean dropped_count = {_fmt(b_dropped)}, A 组 = {_fmt(a_dropped)}\n\n"
            f"**建议**: 冻结混合桥接逻辑，优先做 Stage A（块供给）和 Stage B（聚类权重）收敛。\n"
        )

    # ---- 规则 3: Block-first 参与度不足 ----
    cond3_low_generated = b_generated < 20
    cond3_low_selected = b_selected_blocks < 5
    cond3_low_alns = b_alns_accepted < 1

    if cond3_low_generated or cond3_low_selected or cond3_low_alns:
        reasons: List[str] = []
        if cond3_low_generated:
            reasons.append(f"generated_blocks_total 过低 ({_fmt(b_generated)} < 20)")
        if cond3_low_selected:
            reasons.append(f"selected_blocks_count 过低 ({_fmt(b_selected_blocks)} < 5)")
        if cond3_low_alns:
            reasons.append(f"block_alns_rounds_accepted 过低 ({_fmt(b_alns_accepted)} < 1)")

        return (
            "\n## Automatic Interpretation\n\n"
            "### 规则 3: Block-first 参与度不足\n\n"
            "> **判读结论**: block-first 当前候选块供给不足或块级搜索接受率偏低，"
            "建议优先放宽 block generator 供给参数，再做质量收敛。\n\n"
            f"- 原因: {', '.join(reasons)}\n"
            f"- B 组 mean generated_blocks_total = {_fmt(b_generated)}\n"
            f"- B 组 mean selected_blocks_count = {_fmt(b_selected_blocks)}\n"
            f"- B 组 mean block_alns_rounds_accepted = {_fmt(b_alns_accepted)}\n\n"
            f"**建议**: 在 Stage A 中采用更宽的块供给配置（A3_wider），"
            "提高 block_generator_max_blocks_total 和 block_generator_max_seed_per_bucket。\n"
        )

    # ---- 默认判读 ----
    return (
        "\n## Automatic Interpretation\n\n"
        "### 默认判读: 收益不显著，需要更多数据\n\n"
        "> **判读结论**: block-first 与 constructive_lns_search 相比收益不显著，"
        "建议增加实验轮次（更多 seeds）或调整参数后再做判断。\n\n"
        f"- A 组 mean dropped_count = {_fmt(a_dropped)}\n"
        f"- B 组 mean dropped_count = {_fmt(b_dropped)}, Δ = {_fmt(b_dropped - a_dropped)}\n"
        f"- A 组 mean tail_underfilled_count = {_fmt(a_tail)}\n"
        f"- B 组 mean tail_underfilled_count = {_fmt(b_tail)}, Δ = {_fmt(b_tail - a_tail)}\n"
        f"- A 组 mean scheduled_real_orders = {_fmt(a_scheduled)}\n"
        f"- B 组 mean scheduled_real_orders = {_fmt(b_scheduled)}, Δ = {_fmt(b_scheduled - a_scheduled)}\n\n"
        f"**建议**: 收集更多 seeds（如 5-10 个）的数据后再做收敛决策。\n"
    )


def _build_raw_run_table(df: pd.DataFrame, include_c: bool = False) -> str:
    """构建 Raw Run Table markdown。"""
    cols = [
        "profile_name", "seed", "scheduled_real_orders", "dropped_count",
        "tail_underfilled_count", "selected_real_bridge_edge_count",
        "selected_virtual_bridge_family_edge_count", "selected_blocks_count",
        "mixed_bridge_success_count", "total_runtime_seconds", "run_failed",
    ]

    # 过滤列
    available_cols = [c for c in cols if c in df.columns]

    # 选择关键列并格式化
    table_df = df[available_cols].copy()
    for col in available_cols:
        if col in ("profile_name", "run_failed"):
            continue
        table_df[col] = table_df[col].apply(lambda x: _fmt(x) if pd.notna(x) else "N/A")

    # 按 profile, seed 排序
    table_df = table_df.sort_values(["profile_name", "seed"]).reset_index(drop=True)

    # 构建 markdown 表
    lines = ["| " + " | ".join(available_cols) + " |"]
    lines.append("| " + " | ".join(["---"] * len(available_cols)) + " |")
    for _, row in table_df.iterrows():
        vals = [str(row[c]) for c in available_cols]
        lines.append("| " + " | ".join(vals) + " |")

    return "\n".join(lines)


def _build_aggregated_table(profile_stats: Dict[str, Dict[str, float]]) -> str:
    """构建 Aggregated by Profile markdown。"""
    if not profile_stats:
        return "> 无聚合数据。\n"

    profiles = sorted(profile_stats.keys())
    metrics = [
        ("mean_dropped_count", "Mean Dropped"),
        ("min_dropped_count", "Min Dropped"),
        ("max_dropped_count", "Max Dropped"),
        ("mean_tail_underfilled_count", "Mean Tail"),
        ("min_tail_underfilled_count", "Min Tail"),
        ("max_tail_underfilled_count", "Max Tail"),
        ("mean_scheduled_real_orders", "Mean Scheduled"),
        ("min_scheduled_real_orders", "Min Scheduled"),
        ("max_scheduled_real_orders", "Max Scheduled"),
        ("mean_total_runtime_seconds", "Mean Runtime (s)"),
        ("mean_selected_real_bridge_edge_count", "Mean Real Bridge"),
        ("mean_selected_virtual_bridge_family_edge_count", "Mean Virtual Family"),
        ("mean_selected_blocks_count", "Mean Selected Blocks"),
        ("mean_mixed_bridge_success_count", "Mean Mixed Bridge"),
        ("mean_block_alns_rounds_accepted", "Mean ALNS Accepted"),
        ("block_alns_accept_rate", "ALNS Accept Rate"),
    ]

    # 只保留存在的 metric
    available_metrics = []
    for key, label in metrics:
        for profile in profiles:
            if key in profile_stats.get(profile, {}):
                available_metrics.append((key, label))
                break

    if not available_metrics:
        return "> 无有效聚合指标。\n"

    header = "| Profile | " + " | ".join([m[1] for m in available_metrics]) + " |"
    sep = "| --- | " + " | ".join(["---"] * len(available_metrics)) + " |"
    lines = [header, sep]

    for profile in profiles:
        stats = profile_stats.get(profile, {})
        vals = [_fmt(stats.get(key, 0.0)) for key, _ in available_metrics]
        profile_label = profile.split("_")[-1] if "_" in profile else profile
        lines.append(f"| {profile_label} | " + " | ".join(vals) + " |")

    return "\n".join(lines)


def _build_paired_deltas_table(deltas: Dict[str, Dict[str, float]], include_c: bool = False) -> str:
    """构建 Paired Deltas markdown。"""
    if not deltas:
        return "> 无成对比较数据。\n"

    pairs = [("B_minus_A", "B (block_first) - A (mainline)")]
    if "C_minus_A" in deltas:
        pairs.append(("C_minus_A", "C (guarded_order) - A (mainline)"))
    if "B_minus_C" in deltas:
        pairs.append(("B_minus_C", "B (block_first) - C (guarded_order)"))

    compare_cols = [
        ("dropped_count_mean", "Δ dropped_count"),
        ("tail_underfilled_count_mean", "Δ tail_underfilled"),
        ("scheduled_real_orders_mean", "Δ scheduled"),
        ("total_runtime_seconds_mean", "Δ runtime (s)"),
        ("mixed_bridge_success_count_mean", "Δ mixed_bridge_success"),
        ("selected_blocks_count_mean", "Δ selected_blocks"),
    ]

    lines = ["| Pair | " + " | ".join([c[1] for c in compare_cols]) + " |"]
    lines.append("| --- | " + " | ".join(["---"] * len(compare_cols)) + " |")

    for pair_key, pair_label in pairs:
        if pair_key not in deltas:
            continue
        d = deltas[pair_key]
        vals = [_fmt(d.get(col, 0.0)) for col, _ in compare_cols]
        lines.append(f"| {pair_label} | " + " | ".join(vals) + " |")

    return "\n".join(lines)


def build_ab_summary_markdown(
    df: pd.DataFrame,
    setup: Dict[str, Any],
    group_a: str = "constructive_lns_search",
    group_b: str = "block_first_guarded_search",
    group_c: Optional[str] = None,
) -> str:
    """
    构建 A/B 实验完整 Markdown 汇总报告。

    Parameters
    ----------
    df : pd.DataFrame
        实验结果 DataFrame。
    setup : dict
        实验配置信息，包含 keys: orders_file, steel_info_file, profiles,
        seeds, rounds, time_limit_seconds, max_virtual_chain。
    group_a : str
        A 组 profile。
    group_b : str
        B 组 profile。
    group_c : str, optional
        C 组 profile（可选）。

    Returns
    -------
    str
        Markdown 格式的汇总报告。
    """
    profile_stats = aggregate_by_profile(df)
    deltas = paired_deltas(df, group_a=group_a, group_b=group_b, group_c=group_c)
    interpretation = automatic_interpretation(profile_stats, deltas)

    include_c = group_c is not None

    lines = [
        "# Block-first A/B Experiment Summary",
        "",
        "## Experiment Setup",
        f"- **orders file**: {setup.get('orders_file', 'N/A')}",
        f"- **steel info file**: {setup.get('steel_info_file', 'N/A')}",
        f"- **profiles**: {', '.join(setup.get('profiles', []))}",
        f"- **seeds**: {setup.get('seeds', [])}",
        f"- **rounds**: {setup.get('rounds', 'N/A')}",
        f"- **time_limit_seconds**: {setup.get('time_limit_seconds', 'N/A')}",
        f"- **max_virtual_chain**: {setup.get('max_virtual_chain', 'N/A')}",
        "",
        "## Raw Run Table",
        "",
        _build_raw_run_table(df, include_c=include_c),
        "",
        "## Aggregated by Profile",
        "",
        _build_aggregated_table(profile_stats),
        "",
        "## Paired Deltas",
        "",
        _build_paired_deltas_table(deltas, include_c=include_c),
        "",
    ]

    # 合并 interpretation（它已包含标题）
    lines.append(interpretation)

    return "\n".join(lines)


# =============================================================================
# 参数收敛实验汇总
# =============================================================================


def recommend_stage_config(
    stage_rows: List[Dict[str, Any]],
    stage_name: str,
) -> Dict[str, Any]:
    """
    根据阶段实验结果，推荐最佳配置。

    选择规则：
    1. 先过滤掉失败的行
    2. 按 dropped_count 升序排序
    3. 相同 dropped_count 时选 tail_underfilled_count 更低的
    4. 相同 tail 时选 runtime 更短的

    Parameters
    ----------
    stage_rows : list[dict]
        该阶段所有实验行。
    stage_name : str
        阶段名称，如 "Stage_A"、"Stage_B"。

    Returns
    -------
    dict
        包含 recommended_config_name 和 reason 的推荐结果。
    """
    if not stage_rows:
        return {"recommended_config_name": "", "reason": "无实验数据"}

    # 过滤失败行
    valid = [r for r in stage_rows if not r.get("run_failed", False)]
    if not valid:
        return {
            "recommended_config_name": stage_rows[0].get("config_name", "unknown"),
            "reason": "所有配置均失败，保留首个配置",
        }

    # 排序
    def sort_key(r: Dict[str, Any]) -> tuple:
        dropped = float(r.get("dropped_count", 0))
        tail = float(r.get("tail_underfilled_count", 0))
        runtime = float(r.get("total_runtime_seconds", 999999))
        return (dropped, tail, runtime)

    valid.sort(key=sort_key)
    best = valid[0]

    return {
        "recommended_config_name": str(best.get("config_name", "")),
        "reason": (
            f"dropped={best.get('dropped_count')}, "
            f"tail={best.get('tail_underfilled_count')}, "
            f"runtime={best.get('total_runtime_seconds')}s"
        ),
    }


def build_param_sweep_summary_markdown(
    stage_a_rows: List[Dict[str, Any]],
    stage_b_rows: List[Dict[str, Any]],
    stage_c_rows: List[Dict[str, Any]],
    setup: Dict[str, Any],
) -> str:
    """
    构建参数收敛实验的 Markdown 汇总报告。

    Parameters
    ----------
    stage_a_rows, stage_b_rows, stage_c_rows : list[dict]
        各阶段实验结果行。
    setup : dict
        实验基础配置。

    Returns
    -------
    str
        Markdown 格式的汇总报告。
    """
    rec_a = recommend_stage_config(stage_a_rows, "Stage_A")
    rec_b = recommend_stage_config(stage_b_rows, "Stage_B")
    rec_c = recommend_stage_config(stage_c_rows, "Stage_C")

    def _build_stage_section(
        stage_name: str,
        stage_rows: List[Dict[str, Any]],
        recommendation: Dict[str, Any],
    ) -> str:
        if not stage_rows:
            return f"## {stage_name} Summary\n\n> 无实验数据。\n"

        df = pd.DataFrame(stage_rows)
        if df.empty:
            return f"## {stage_name} Summary\n\n> 无实验数据。\n"

        configs = df["config_name"].unique().tolist()

        lines = [f"## {stage_name} Summary", ""]

        # 配置对比表
        metrics = [
            ("dropped_count", "Dropped"),
            ("tail_underfilled_count", "Tail"),
            ("scheduled_real_orders", "Scheduled"),
            ("total_runtime_seconds", "Runtime(s)"),
            ("generated_blocks_total", "Gen Blocks"),
            ("selected_blocks_count", "Sel Blocks"),
            ("mixed_bridge_success_count", "Mixed Bridge"),
        ]

        available_metrics = [(m, l) for m, l in metrics if m in df.columns]

        header = "| Config | " + " | ".join([l for _, l in available_metrics]) + " |"
        sep = "| --- | " + " | ".join(["---"] * len(available_metrics)) + " |"
        lines.append(header)
        lines.append(sep)

        for config in sorted(configs):
            sub = df[df["config_name"] == config]
            vals = []
            for metric, _ in available_metrics:
                mean_val = _safe_mean(_select_numeric(sub, metric))
                vals.append(_fmt(mean_val))
            lines.append(f"| {config} | " + " | ".join(vals) + " |")

        lines.append("")
        lines.append(
            f"**推荐保留配置**: {recommendation.get('recommended_config_name', 'N/A')} "
            f"(原因: {recommendation.get('reason', 'N/A')})"
        )
        lines.append("")

        return "\n".join(lines)

    lines = [
        "# Block-first Parameter Sweep Summary",
        "",
        "## Experiment Setup",
        f"- **orders file**: {setup.get('orders_file', 'N/A')}",
        f"- **steel info file**: {setup.get('steel_info_file', 'N/A')}",
        f"- **base profile**: {setup.get('profile', 'block_first_guarded_search')}",
        f"- **seeds**: {setup.get('seeds', []) if isinstance(setup.get('seeds'), list) else str(setup.get('seeds', ''))}",
        f"- **time_limit_seconds**: {setup.get('time_limit_seconds', 'N/A')}",
        f"- **max_virtual_chain**: {setup.get('max_virtual_chain', 'N/A')}",
        "",
    ]

    lines.append(_build_stage_section("Stage A: Block Supply", stage_a_rows, rec_a))
    lines.append(_build_stage_section("Stage B: Directional Clustering Weights", stage_b_rows, rec_b))
    lines.append(_build_stage_section("Stage C: Block ALNS / Mixed Bridge", stage_c_rows, rec_c))

    # 最终推荐
    lines.extend([
        "## Final Recommended Block-first Config",
        "",
        "| Stage | Recommended Config | Reason |",
        "| --- | --- | --- |",
        f"| Stage_A | {rec_a.get('recommended_config_name', 'N/A')} | {rec_a.get('reason', '')} |",
        f"| Stage_B | {rec_b.get('recommended_config_name', 'N/A')} | {rec_b.get('reason', '')} |",
        f"| Stage_C | {rec_c.get('recommended_config_name', 'N/A')} | {rec_c.get('reason', '')} |",
        "",
        "**收敛流程**: 依次应用 Stage_A → Stage_B → Stage_C 的推荐配置，"
        "形成最终 block-first 参数组合。",
    ])

    return "\n".join(lines)


def write_param_sweep_outputs(
    stage_a_rows: List[Dict[str, Any]],
    stage_b_rows: List[Dict[str, Any]],
    stage_c_rows: List[Dict[str, Any]],
    output_dir: Union[str, Path],
    setup: Dict[str, Any],
) -> Tuple[Path, Path]:
    """
    写入参数收敛实验结果到 CSV 和 Markdown。

    Parameters
    ----------
    stage_a_rows, stage_b_rows, stage_c_rows : list[dict]
        各阶段实验结果。
    output_dir : str or Path
        输出目录。
    setup : dict
        实验基础配置。

    Returns
    -------
    (csv_path, md_path) : (Path, Path)
        输出的 CSV 和 Markdown 文件路径。
    """
    from aps_cp_sat.model.experiment_utils import (
        ALL_EXPERIMENT_FIELDS,
        write_results_csv,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # CSV: 合并所有阶段
    all_rows = []
    for row in stage_a_rows:
        all_rows.append({**row, "stage_name": "Stage_A"})
    for row in stage_b_rows:
        all_rows.append({**row, "stage_name": "Stage_B"})
    for row in stage_c_rows:
        all_rows.append({**row, "stage_name": "Stage_C"})

    csv_path = output_dir / "block_first_param_sweep_results.csv"
    write_results_csv(all_rows, csv_path)

    # Markdown
    md_path = output_dir / "block_first_param_sweep_summary.md"
    md_content = build_param_sweep_summary_markdown(
        stage_a_rows, stage_b_rows, stage_c_rows, setup
    )
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    return csv_path, md_path
