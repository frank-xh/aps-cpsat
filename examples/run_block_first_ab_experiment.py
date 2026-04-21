#!/usr/bin/env python3
"""
run_block_first_ab_experiment.py - Block-first A/B 对比实验脚本

使用方法:
    python examples/run_block_first_ab_experiment.py \\
        --orders data_orders.xlsx \\
        --steel-info data_steel_info.xlsx \\
        --output-dir outputs/ab_experiment \\
        --seeds 2027 2028 2029 \\
        --rounds 4 \\
        --time-limit-seconds 60 \\
        --max-virtual-chain 5

可选参数:
    --include-guarded-order-first   添加 C 组 (constructive_lns_virtual_guarded_frontload)
    --profile-b                      指定 B 组 profile (默认 block_first_guarded_search)

输出文件:
    block_first_ab_results.csv       每轮实验结果
    block_first_ab_summary.md       汇总报告（含自动判读）

实验矩阵:
    默认 A/B: constructive_lns_search vs block_first_guarded_search
    seeds = [2027, 2028, 2029] → 6 轮
    启用 C 组后 → 9 轮
"""

from __future__ import annotations

import argparse
import sys
import traceback
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from time import perf_counter

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aps_cp_sat.cold_rolling_pipeline import ColdRollingPipeline
from aps_cp_sat.config.parameters import build_profile_config
from aps_cp_sat.domain.models import ColdRollingRequest
from aps_cp_sat.model.experiment_summary import build_ab_summary_markdown
from aps_cp_sat.model.experiment_utils import (
    apply_profile_guard,
    collect_experiment_row,
    write_results_csv,
)


# =============================================================================
# 默认配置
# =============================================================================

DEFAULT_PROFILES_A = "constructive_lns_search"
DEFAULT_PROFILES_B = "block_first_guarded_search"
DEFAULT_PROFILES_C = "constructive_lns_virtual_guarded_frontload"
DEFAULT_SEEDS = [2027, 2028, 2029]
DEFAULT_ROUNDS = 4
DEFAULT_TIME_LIMIT = 60.0
DEFAULT_MAX_VIRTUAL_CHAIN = 5


# =============================================================================
# 实验矩阵定义
# =============================================================================


def build_experiment_matrix(
    profiles: list[str],
    seeds: list[int],
    rounds: int,
    time_limit: float,
    max_virtual_chain: int,
    orders_path: Path,
    steel_info_path: Path,
    output_dir: Path,
) -> list[dict]:
    """
    构建实验矩阵。

    Returns
    -------
    list[dict]
        每个元素包含: profile_name, seed, req, output_path
    """
    matrix = []
    for profile in profiles:
        for seed in seeds:
            # 构造输出文件名
            profile_prefix = {
                DEFAULT_PROFILES_A: "A_mainline",
                DEFAULT_PROFILES_B: "B_block_first",
                DEFAULT_PROFILES_C: "C_guarded_order_first",
            }.get(profile, f"profile_{profile}")

            output_path = output_dir / f"{profile_prefix}_seed{seed}.xlsx"

            # 构建请求
            cfg = build_profile_config(
                profile,
                validation_mode=False,
                production_compatibility_mode=False,
            )
            cfg = replace(
                cfg,
                model=replace(
                    cfg.model,
                    rounds=rounds,
                    time_limit_seconds=time_limit,
                    max_virtual_chain=max_virtual_chain,
                    master_seed_count=1,
                ),
            )

            req = ColdRollingRequest(
                orders_path=orders_path,
                steel_info_path=steel_info_path,
                output_path=output_path,
                config=cfg,
            )

            matrix.append({
                "profile_name": profile,
                "seed": seed,
                "req": req,
                "output_path": output_path,
            })

    return matrix


# =============================================================================
# 单轮实验执行
# =============================================================================


def run_single_experiment(
    profile_name: str,
    seed: int,
    req: ColdRollingRequest,
    output_path: Path,
) -> dict:
    """
    执行单轮实验。

    Returns
    -------
    dict
        实验结果行。
    """
    print(f"\n{'='*60}")
    print(f"[Experiment] profile={profile_name}, seed={seed}")
    print(f"  rounds={req.config.model.rounds}")
    print(f"  time_limit={req.config.model.time_limit_seconds}s")
    print(f"  max_virtual_chain={req.config.model.max_virtual_chain}")
    print(f"  output={output_path}")
    print(f"{'='*60}")

    # 设置随机种子
    import random
    random.seed(seed)

    start_time = perf_counter()
    error_message = None
    result = None

    try:
        pipeline = ColdRollingPipeline()
        result = pipeline.run(req)
    except Exception as e:
        error_message = f"{type(e).__name__}: {str(e)}"
        print(f"[ERROR] {error_message}")
        traceback.print_exc()

    elapsed = perf_counter() - start_time
    print(f"[Experiment] completed in {elapsed:.1f}s")

    # 采集结果
    row = collect_experiment_row(
        result=result,
        profile_name=profile_name,
        seed=seed,
        output_path=output_path,
        elapsed_seconds=elapsed,
        run_error=error_message,
    )

    return row


# =============================================================================
# 主入口
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Block-first A/B 对比实验脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--orders",
        type=Path,
        required=True,
        help="订单数据文件路径",
    )
    parser.add_argument(
        "--steel-info",
        type=Path,
        required=True,
        help="钢材信息文件路径",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/ab_experiment"),
        help="输出目录 (默认: outputs/ab_experiment)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=DEFAULT_SEEDS,
        help=f"随机种子列表 (默认: {DEFAULT_SEEDS})",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=DEFAULT_ROUNDS,
        help=f"轮次数 (默认: {DEFAULT_ROUNDS})",
    )
    parser.add_argument(
        "--time-limit-seconds",
        type=float,
        default=DEFAULT_TIME_LIMIT,
        help=f"时间限制 (默认: {DEFAULT_TIME_LIMIT})",
    )
    parser.add_argument(
        "--max-virtual-chain",
        type=int,
        default=DEFAULT_MAX_VIRTUAL_CHAIN,
        help=f"最大虚拟链长度 (默认: {DEFAULT_MAX_VIRTUAL_CHAIN})",
    )
    parser.add_argument(
        "--include-guarded-order-first",
        action="store_true",
        help="包含 C 组 (constructive_lns_virtual_guarded_frontload)",
    )
    parser.add_argument(
        "--profile-a",
        type=str,
        default=DEFAULT_PROFILES_A,
        help=f"A 组 profile (默认: {DEFAULT_PROFILES_A})",
    )
    parser.add_argument(
        "--profile-b",
        type=str,
        default=DEFAULT_PROFILES_B,
        help=f"B 组 profile (默认: {DEFAULT_PROFILES_B})",
    )
    parser.add_argument(
        "--profile-c",
        type=str,
        default=DEFAULT_PROFILES_C,
        help=f"C 组 profile (默认: {DEFAULT_PROFILES_C})",
    )

    args = parser.parse_args()

    # 构建 profiles 列表
    profiles = [args.profile_a, args.profile_b]
    group_c = args.profile_c if args.include_guarded_order_first else None
    if group_c:
        profiles.append(group_c)

    print(f"\n{'#'*60}")
    print(f"# Block-first A/B Experiment")
    print(f"# Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"# {'#'*60}")
    print(f"# Orders: {args.orders}")
    print(f"# Steel Info: {args.steel_info}")
    print(f"# Profiles: {profiles}")
    print(f"# Seeds: {args.seeds}")
    print(f"# Rounds: {args.rounds}")
    print(f"# Time Limit: {args.time_limit_seconds}s")
    print(f"# Max Virtual Chain: {args.max_virtual_chain}")
    print(f"# Output Dir: {args.output_dir}")
    print(f"# {'#'*60}\n")

    # 创建输出目录
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 构建实验矩阵
    matrix = build_experiment_matrix(
        profiles=profiles,
        seeds=args.seeds,
        rounds=args.rounds,
        time_limit=args.time_limit_seconds,
        max_virtual_chain=args.max_virtual_chain,
        orders_path=args.orders,
        steel_info_path=args.steel_info,
        output_dir=args.output_dir,
    )

    total_runs = len(matrix)
    print(f"[Setup] Total experiment runs: {total_runs}")

    # 执行实验
    all_rows = []
    for idx, run_info in enumerate(matrix, 1):
        print(f"\n>>> Run {idx}/{total_runs}")
        row = run_single_experiment(
            profile_name=run_info["profile_name"],
            seed=run_info["seed"],
            req=run_info["req"],
            output_path=run_info["output_path"],
        )
        all_rows.append(row)

    # 应用 profile guard
    print("\n[Guard] Applying profile guard validation...")
    all_rows = apply_profile_guard(all_rows, args.profile_a)
    all_rows = apply_profile_guard(all_rows, args.profile_b)
    if group_c:
        all_rows = apply_profile_guard(all_rows, group_c)

    # 写入 CSV
    csv_path = args.output_dir / "block_first_ab_results.csv"
    write_results_csv(all_rows, csv_path)
    print(f"\n[Output] CSV: {csv_path}")

    # 生成 Markdown 汇总
    import pandas as pd
    df = pd.DataFrame(all_rows)

    setup = {
        "orders_file": str(args.orders),
        "steel_info_file": str(args.steel_info),
        "profiles": profiles,
        "seeds": args.seeds,
        "rounds": args.rounds,
        "time_limit_seconds": args.time_limit_seconds,
        "max_virtual_chain": args.max_virtual_chain,
    }

    md_content = build_ab_summary_markdown(
        df=df,
        setup=setup,
        group_a=args.profile_a,
        group_b=args.profile_b,
        group_c=group_c,
    )

    md_path = args.output_dir / "block_first_ab_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"[Output] Markdown: {md_path}")

    # 打印 guard 结果摘要
    failed_guards = [r for r in all_rows if r.get("profile_guard_failed", False)]
    if failed_guards:
        print(f"\n[Warning] {len(failed_guards)} runs with profile guard failure:")
        for r in failed_guards:
            print(f"  - profile={r['profile_name']}, seed={r['seed']}: {r.get('profile_guard_reason', '')}")
    else:
        print("\n[Guard] All runs passed profile validation.")

    print(f"\n{'#'*60}")
    print(f"# Block-first A/B Experiment Completed")
    print(f"# Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"# {'#'*60}")


if __name__ == "__main__":
    main()
