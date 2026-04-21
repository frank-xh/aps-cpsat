"""
A/B 实验：guard virtual family frontload vs mainline
对比 constructive_lns_search (A组) vs constructive_lns_virtual_guarded_frontload (B组)
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

# ---- 路径设置 ----
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from aps_cp_sat.cold_rolling_pipeline import ColdRollingPipeline
from aps_cp_sat.config import PlannerConfig, build_profile_config
from aps_cp_sat.domain.models import ColdRollingRequest


# =============================================================================
# 实验配置
# =============================================================================

@dataclass
class ExperimentConfig:
    orders_path: Path
    steel_info_path: Path
    output_dir: Path
    seeds: list[int]
    rounds: int
    time_limit_seconds: float
    max_virtual_chain: int
    max_orders: int = 10_000_000
    ab_profiles: tuple[str, str] = ("constructive_lns_search", "constructive_lns_virtual_guarded_frontload")

    def __post_init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 结果采集器
# =============================================================================

@dataclass
class ExperimentResult:
    # 基础标识
    profile_name: str = ""
    group_label: str = ""  # "A" or "B"
    seed: int = 0
    output_file: str = ""
    # 基础结果
    scheduled_real_orders: int = 0
    dropped_count: int = 0
    valid_segment_count: int = 0
    underfilled_segment_count: int = 0
    orders_in_segments: int = 0
    acceptance: str = ""
    acceptance_gate_reason: str = ""
    validation_gate_reason: str = ""
    # 桥接相关
    selected_direct_edge_count: int = 0
    selected_real_bridge_edge_count: int = 0
    selected_virtual_bridge_family_edge_count: int = 0
    selected_legacy_virtual_bridge_edge_count: int = 0
    max_bridge_count_used: int = 0
    # Guarded family
    greedy_virtual_family_edge_uses: int = 0
    greedy_virtual_family_budget_blocked_count: int = 0
    alns_virtual_family_attempt_count: int = 0
    alns_virtual_family_accept_count: int = 0
    local_cpsat_virtual_family_selected_count: int = 0
    family_repair_attempt_count: int = 0
    family_repair_accept_count: int = 0
    family_repair_gain_dropped: int = 0
    family_repair_gain_underfilled: int = 0
    family_repair_gain_scheduled_orders: int = 0
    family_repair_no_gain_count: int = 0
    # 耗时
    template_build_seconds: float = 0.0
    constructive_seconds: float = 0.0
    cutter_seconds: float = 0.0
    lns_seconds: float = 0.0
    local_cpsat_total_seconds: float = 0.0
    total_runtime_seconds: float = 0.0
    # Candidate graph / run path
    constructive_edge_policy: str = ""
    allow_real_bridge_edge_in_constructive: bool = False
    allow_virtual_bridge_edge_in_constructive: bool = False
    bridge_expansion_mode: str = ""
    candidate_graph_source: str = ""
    # Guard
    profile_guard_passed: bool = True
    profile_guard_failed_reason: str = ""
    # 异常
    run_error: str = ""


def _safe_int(val: Any, default: int = 0) -> int:
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _collect_result(
    profile_name: str,
    group_label: str,
    seed: int,
    output_file: str,
    result,
    t0: float,
) -> ExperimentResult:
    """从 ColdRollingResult / engine_meta 中采集所有指标。"""
    res = ExperimentResult(
        profile_name=profile_name,
        group_label=group_label,
        seed=seed,
        output_file=output_file,
        total_runtime_seconds=time.perf_counter() - t0,
    )

    try:
        em = result.engine_meta or {}
        cfg = result.config
        seq_df = result.schedule_df
        rounds_df = result.rounds_df

        # ---- 基础结果 ----
        res.scheduled_real_orders = _safe_int(em.get("scheduled_real_orders"))
        res.dropped_count = _safe_int(em.get("dropped_count"))
        res.acceptance = str(em.get("acceptance", ""))
        res.acceptance_gate_reason = str(em.get("acceptance_gate_reason", ""))
        res.validation_gate_reason = str(em.get("validation_gate_reason", ""))

        # ---- Segment 统计 ----
        if rounds_df is not None and hasattr(rounds_df, "shape"):
            # valid_segment_count: segments with is_valid=True
            if "is_valid" in rounds_df.columns:
                res.valid_segment_count = int(rounds_df["is_valid"].sum())
            # underfilled_segment_count: valid but below campaign ton min
            rule = getattr(cfg, "rule", None) if cfg else None
            ton_min = float(getattr(rule, "campaign_ton_min", 200.0) or 200.0) if rule else 200.0
            if "campaign_tons" in rounds_df.columns and "is_valid" in rounds_df.columns:
                underfilled = rounds_df[(rounds_df["is_valid"] == True) & (rounds_df["campaign_tons"] < ton_min)]
                res.underfilled_segment_count = int(len(underfilled))
            if "orders_in_segment" in rounds_df.columns:
                res.orders_in_segments = int(rounds_df["orders_in_segment"].sum())
            elif "order_count" in rounds_df.columns:
                res.orders_in_segments = int(rounds_df["order_count"].sum())

        # ---- 桥接统计 ----
        res.selected_direct_edge_count = _safe_int(em.get("selected_direct_edge_count"))
        res.selected_real_bridge_edge_count = _safe_int(em.get("selected_real_bridge_edge_count"))
        res.selected_virtual_bridge_family_edge_count = _safe_int(em.get("selected_virtual_bridge_family_edge_count"))
        res.selected_legacy_virtual_bridge_edge_count = _safe_int(em.get("selected_legacy_virtual_bridge_edge_count"))
        res.max_bridge_count_used = _safe_int(em.get("max_bridge_count_used"))

        # ---- Guarded family ----
        res.greedy_virtual_family_edge_uses = _safe_int(em.get("greedy_virtual_family_edge_uses"))
        res.greedy_virtual_family_budget_blocked_count = _safe_int(em.get("greedy_virtual_family_budget_blocked_count"))
        res.alns_virtual_family_attempt_count = _safe_int(em.get("alns_virtual_family_attempt_count"))
        res.alns_virtual_family_accept_count = _safe_int(em.get("alns_virtual_family_accept_count"))
        res.local_cpsat_virtual_family_selected_count = _safe_int(em.get("local_cpsat_virtual_family_selected_count"))
        res.family_repair_attempt_count = _safe_int(em.get("family_repair_attempt_count"))
        res.family_repair_accept_count = _safe_int(em.get("family_repair_accept_count"))
        res.family_repair_gain_dropped = _safe_int(em.get("family_repair_gain_dropped"))
        res.family_repair_gain_underfilled = _safe_int(em.get("family_repair_gain_underfilled"))
        res.family_repair_gain_scheduled_orders = _safe_int(em.get("family_repair_gain_scheduled_orders"))
        res.family_repair_no_gain_count = _safe_int(em.get("family_repair_no_gain_count"))

        # ---- 耗时 ----
        res.template_build_seconds = _safe_float(em.get("template_build_seconds"))
        res.constructive_seconds = _safe_float(em.get("constructive_seconds"))
        res.cutter_seconds = _safe_float(em.get("campaign_cutter_seconds"))
        res.lns_seconds = _safe_float(em.get("lns_seconds"))
        res.local_cpsat_total_seconds = _safe_float(em.get("local_cpsat_total_seconds"))

        # ---- Candidate graph / run path ----
        res.constructive_edge_policy = str(em.get("constructive_edge_policy", ""))
        res.allow_real_bridge_edge_in_constructive = bool(em.get("allow_real_bridge_edge_in_constructive", False))
        res.allow_virtual_bridge_edge_in_constructive = bool(em.get("allow_virtual_bridge_edge_in_constructive", False))
        res.bridge_expansion_mode = str(em.get("bridge_expansion_mode", ""))
        res.candidate_graph_source = str(em.get("candidate_graph_source", ""))

    except Exception as e:
        res.run_error = f"collection_error: {e}"

    return res


# =============================================================================
# Profile Guard 校验
# =============================================================================

def _validate_profile_guard(result: ExperimentResult, group_label: str) -> tuple[bool, str]:
    """
    校验实际生效的 profile 配置是否符合预期。

    A 组期望:
        constructive_edge_policy == "direct_plus_real_bridge"
        allow_real_bridge_edge_in_constructive == True
        allow_virtual_bridge_edge_in_constructive == False

    B 组期望:
        allow_real_bridge_edge_in_constructive == True
        allow_virtual_bridge_edge_in_constructive == True
        bridge_expansion_mode == "disabled"
    """
    reasons: list[str] = []

    if group_label == "A":
        if result.constructive_edge_policy not in {"", "direct_plus_real_bridge"}:
            reasons.append(f"constructive_edge_policy={result.constructive_edge_policy!r} != 'direct_plus_real_bridge'")
        if not result.allow_real_bridge_edge_in_constructive:
            reasons.append("allow_real_bridge_edge_in_constructive != True")
        if result.allow_virtual_bridge_edge_in_constructive:
            reasons.append("allow_virtual_bridge_edge_in_constructive != False")
    elif group_label == "B":
        if not result.allow_real_bridge_edge_in_constructive:
            reasons.append("allow_real_bridge_edge_in_constructive != True")
        if not result.allow_virtual_bridge_edge_in_constructive:
            reasons.append("allow_virtual_bridge_edge_in_constructive != True")
        if result.bridge_expansion_mode not in {"", "disabled"}:
            reasons.append(f"bridge_expansion_mode={result.bridge_expansion_mode!r} != 'disabled'")

    if reasons:
        return False, "; ".join(reasons)
    return True, ""


# =============================================================================
# 单轮实验执行
# =============================================================================

def _run_single_experiment(
    profile_name: str,
    group_label: str,
    seed: int,
    exp_cfg: ExperimentConfig,
    pipeline: ColdRollingPipeline,
) -> ExperimentResult:
    """执行单轮实验。失败时不崩溃，返回带错误标记的 result。"""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 输出文件命名: A_mainline_seed2027.xlsx / B_guarded_seed2027.xlsx
    label = "mainline" if group_label == "A" else "guarded"
    output_file = exp_cfg.output_dir / f"{group_label}_{label}_seed{seed}_{ts}.xlsx"

    print(f"\n{'='*60}")
    print(f"[{group_label}] profile={profile_name}  seed={seed}")
    print(f"[{group_label}] rounds={exp_cfg.rounds}  time_limit={exp_cfg.time_limit_seconds}s")
    print(f"[{group_label}] max_virtual_chain={exp_cfg.max_virtual_chain}")
    print(f"[{group_label}] output={output_file}")

    try:
        cfg = build_profile_config(profile_name, validation_mode=False)
        # 覆写时序和种子参数
        cfg.model.rounds = exp_cfg.rounds
        cfg.model.time_limit_seconds = exp_cfg.time_limit_seconds
        cfg.model.max_virtual_chain = exp_cfg.max_virtual_chain
        cfg.max_orders = exp_cfg.max_orders
        # 设置种子
        if hasattr(cfg.model, "seed"):
            cfg.model.seed = seed
        if hasattr(cfg, "seed"):
            cfg.seed = seed

        t0 = time.perf_counter()
        result = pipeline.run(
            ColdRollingRequest(
                orders_path=exp_cfg.orders_path,
                steel_info_path=exp_cfg.steel_info_path,
                output_path=output_file,
                config=cfg,
            )
        )
        elapsed = time.perf_counter() - t0

        # 采集结果
        exp_result = _collect_result(profile_name, group_label, seed, str(output_file), result, t0)

        # Profile guard 校验
        guard_passed, guard_reason = _validate_profile_guard(exp_result, group_label)
        exp_result.profile_guard_passed = guard_passed
        exp_result.profile_guard_failed_reason = guard_reason

        print(f"[{group_label}] done  elapsed={elapsed:.1f}s  dropped={exp_result.dropped_count}  scheduled={exp_result.scheduled_real_orders}")
        if not guard_passed:
            print(f"[{group_label}] [WARN] profile_guard FAILED: {guard_reason}")

        return exp_result

    except Exception as e:
        import traceback
        err_msg = f"{e}\n{traceback.format_exc()}"
        print(f"[{group_label}] [ERROR] {e}")
        res = ExperimentResult(
            profile_name=profile_name,
            group_label=group_label,
            seed=seed,
            output_file=str(output_file),
            run_error=err_msg,
            profile_guard_passed=False,
            profile_guard_failed_reason="run_failed_before_guard_check",
        )
        return res


# =============================================================================
# 实验汇总输出
# =============================================================================

def _agg_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(sum(values) / len(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def generate_summary(
    results: list[ExperimentResult],
    exp_cfg: ExperimentConfig,
) -> tuple[pd.DataFrame, str]:
    """
    生成 CSV DataFrame 和 Markdown 汇总文本。
    """
    rows = [asdict(r) for r in results]
    df = pd.DataFrame(rows)

    # ---- Markdown ----
    lines: list[str] = []
    lines.append("# A/B 实验汇总\n")
    lines.append("> 自动生成 by run_guarded_virtual_ab_experiment.py\n")
    lines.append(f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # 实验配置
    lines.append("## 实验配置\n")
    lines.append(f"- **orders file**: `{exp_cfg.orders_path}`")
    lines.append(f"- **steel info file**: `{exp_cfg.steel_info_path}`")
    lines.append(f"- **seeds**: `{exp_cfg.seeds}`")
    lines.append(f"- **rounds**: `{exp_cfg.rounds}`")
    lines.append(f"- **time_limit_seconds**: `{exp_cfg.time_limit_seconds}`")
    lines.append(f"- **max_virtual_chain**: `{exp_cfg.max_virtual_chain}`")
    lines.append(f"- **profiles**: A=`constructive_lns_search` / B=`constructive_lns_virtual_guarded_frontload`\n")

    # 单轮结果表
    lines.append("## 单轮结果\n")
    cols_show = [
        "group_label", "seed", "scheduled_real_orders", "dropped_count",
        "valid_segment_count", "underfilled_segment_count",
        "selected_real_bridge_edge_count", "selected_virtual_bridge_family_edge_count",
        "family_repair_accept_count", "total_runtime_seconds",
        "profile_guard_passed", "run_error",
    ]
    available_cols = [c for c in cols_show if c in df.columns]
    lines.append(df[available_cols].to_markdown(index=False) + "\n")

    # 失败标记
    failed = df[df["run_error"] != ""]
    if not failed.empty:
        lines.append(f"### 失败轮次 ({len(failed)} 轮)\n")
        lines.append(failed[["group_label", "seed", "run_error"]].to_markdown(index=False) + "\n")

    # ---- 按 profile 聚合统计 ----
    lines.append("## 按 profile 聚合统计\n")

    agg_cols = [
        "dropped_count", "underfilled_segment_count", "valid_segment_count",
        "scheduled_real_orders", "total_runtime_seconds",
        "selected_virtual_bridge_family_edge_count", "family_repair_accept_count",
    ]
    for col in agg_cols:
        if col not in df.columns:
            continue
        lines.append(f"### {col}\n")
        grp = df.groupby("group_label")[col].agg(["mean", "min", "max"]).round(3)
        lines.append(grp.to_markdown() + "\n")

    # ---- 差值分析（B - A） ----
    lines.append("## 差值分析（B - A）\n")
    delta_cols = ["dropped_count", "underfilled_segment_count", "valid_segment_count",
                  "scheduled_real_orders", "total_runtime_seconds"]
    seed_pairs = exp_cfg.seeds

    pair_rows = []
    for s in seed_pairs:
        a_row = df[(df["group_label"] == "A") & (df["seed"] == s)]
        b_row = df[(df["group_label"] == "B") & (df["seed"] == s)]
        if a_row.empty or b_row.empty:
            continue
        a = a_row.iloc[0]
        b = b_row.iloc[0]
        row = {"seed": s}
        for col in delta_cols:
            if col in df.columns:
                row[f"Δ {col}"] = float(b[col]) - float(a[col])
        pair_rows.append(row)

    if pair_rows:
        pair_df = pd.DataFrame(pair_rows)
        lines.append(pair_df.to_markdown(index=False) + "\n")

    # ---- 自动判读 ----
    lines.append("## 自动判读\n")
    a_df = df[df["group_label"] == "A"]
    b_df = df[df["group_label"] == "B"]

    def _safe_mean(d: pd.DataFrame, col: str) -> float:
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
    b_vfamily = int(b_df["selected_virtual_bridge_family_edge_count"].sum())

    # 规则1: 正向收益
    cond1 = (
        (b_dropped_mean < a_dropped_mean) and
        (b_under_mean < a_under_mean) and
        (b_valid_mean >= a_valid_mean) and
        (b_runtime_mean <= a_runtime_mean * 1.5)
    )
    # 规则2: 已使用 guarded family 但收益不稳定
    cond2 = (
        (b_vfamily > 0) and
        not (b_dropped_mean < a_dropped_mean and b_under_mean < a_under_mean)
    )
    # 规则3: guarded family 参与度不足
    cond3 = b_vfamily == 0

    if cond1:
        verdict = (
            "**guarded virtual family 实验线表现为正向收益，建议继续做参数精调。**\n\n"
            f"- B组 dropped 均值({b_dropped_mean:.1f}) < A组({a_dropped_mean:.1f})\n"
            f"- B组 underfilled 均值({b_under_mean:.1f}) < A组({a_under_mean:.1f})\n"
            f"- B组 valid 均值({b_valid_mean:.1f}) >= A组({a_valid_mean:.1f})\n"
            f"- B组耗时({b_runtime_mean:.1f}s) <= A组耗时({a_runtime_mean:.1f}s)*1.5"
        )
    elif cond2:
        verdict = (
            "**guarded virtual family 已参与但收益不稳定，建议优先调 repair pool / ALNS gate，"
            "而不是继续放宽 global constructive。**\n\n"
            f"- guarded family edges used = {b_vfamily}\n"
            f"- B组 dropped 均值({b_dropped_mean:.1f}) vs A组({a_dropped_mean:.1f})\n"
            f"- B组 underfilled 均值({b_under_mean:.1f}) vs A组({a_under_mean:.1f})"
        )
    elif cond3:
        verdict = (
            "**当前 guarded virtual family 参与度不足，建议优先放宽 repair pool，"
            "而不是放宽主线 global constructive。**\n\n"
            f"- guarded family edges used = {b_vfamily} (几乎未参与)"
        )
    else:
        verdict = "当前数据不足以给出明确判读，请检查实验结果。"

    lines.append(verdict + "\n")

    # 汇总表格
    lines.append("### 汇总对比\n")
    summary_rows = []
    for label, grp in [("A (mainline)", a_df), ("B (guarded)", b_df)]:
        summary_rows.append({
            "profile": label,
            "dropped (mean)": round(_safe_mean(grp, "dropped_count"), 2),
            "underfilled (mean)": round(_safe_mean(grp, "underfilled_segment_count"), 2),
            "valid_seg (mean)": round(_safe_mean(grp, "valid_segment_count"), 2),
            "scheduled (mean)": round(_safe_mean(grp, "scheduled_real_orders"), 2),
            "runtime_s (mean)": round(_safe_mean(grp, "total_runtime_seconds"), 2),
            "vfamily_edges (sum)": int(grp["selected_virtual_bridge_family_edge_count"].sum()),
            "family_repair_accept (sum)": int(grp["family_repair_accept_count"].sum()),
        })
    summary_df = pd.DataFrame(summary_rows)
    lines.append(summary_df.to_markdown(index=False) + "\n")

    md_text = "\n".join(lines)
    return df, md_text


# =============================================================================
# 主入口
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="A/B guarded virtual family experiment")
    parser.add_argument("--orders", type=str, default=str(_ROOT / "data_orders.xlsx"),
                        help="Orders Excel file path")
    parser.add_argument("--steel-info", type=str, default=str(_ROOT / "data_steel_info.xlsx"),
                        help="Steel info Excel file path")
    parser.add_argument("--output-dir", type=str, default=str(_ROOT / "outputs" / "ab_experiment"),
                        help="Output directory")
    parser.add_argument("--seeds", type=str, default="2027,2028,2029",
                        help="Comma-separated seed list, e.g. 2027,2028,2029")
    parser.add_argument("--rounds", type=int, default=10,
                        help="LNS rounds per run")
    parser.add_argument("--time-limit", type=float, default=60.0,
                        help="time_limit_seconds per run")
    parser.add_argument("--max-virtual-chain", type=int, default=5,
                        help="max_virtual_chain")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    exp_cfg = ExperimentConfig(
        orders_path=Path(args.orders),
        steel_info_path=Path(args.steel_info),
        output_dir=Path(args.output_dir),
        seeds=seeds,
        rounds=args.rounds,
        time_limit_seconds=args.time_limit,
        max_virtual_chain=args.max_virtual_chain,
    )

    profile_a, profile_b = exp_cfg.ab_profiles
    pipeline = ColdRollingPipeline()
    results: list[ExperimentResult] = []

    total_runs = len(exp_cfg.seeds) * 2
    run_idx = 0

    print(f"\n{'#'*60}")
    print(f"# A/B 实验开始")
    print(f"# 总轮次: {total_runs} ({len(exp_cfg.seeds)} seeds x 2 profiles)")
    print(f"# 输出目录: {exp_cfg.output_dir}")
    print(f"{'#'*60}")

    for seed in exp_cfg.seeds:
        # A 组
        run_idx += 1
        print(f"\n>>> [{run_idx}/{total_runs}] A 组...")
        r_a = _run_single_experiment(profile_a, "A", seed, exp_cfg, pipeline)
        results.append(r_a)

        # B 组
        run_idx += 1
        print(f"\n>>> [{run_idx}/{total_runs}] B 组...")
        r_b = _run_single_experiment(profile_b, "B", seed, exp_cfg, pipeline)
        results.append(r_b)

    # ---- 生成汇总 ----
    csv_path = exp_cfg.output_dir / "ab_experiment_results.csv"
    md_path = exp_cfg.output_dir / "ab_experiment_summary.md"

    df, md_text = generate_summary(results, exp_cfg)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    md_path.write_text(md_text, encoding="utf-8")

    print(f"\n{'#'*60}")
    print(f"# 实验完成")
    print(f"# CSV: {csv_path}")
    print(f"# Markdown: {md_path}")
    print(f"# 总实验轮次: {total_runs}")
    print(f"# 失败轮次: {sum(1 for r in results if r.run_error)}")
    print(f"{'#'*60}")
    print("\n=== Markdown 汇总预览 ===")
    print(md_text)


if __name__ == "__main__":
    main()
