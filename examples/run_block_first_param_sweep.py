#!/usr/bin/env python3
"""
run_block_first_param_sweep.py - Block-first 参数收敛实验脚本

使用方法:
    python examples/run_block_first_param_sweep.py \\
        --orders data_orders.xlsx \\
        --steel-info data_steel_info.xlsx \\
        --output-dir outputs/param_sweep \\
        --seeds 2027 2028 2029 \\
        --rounds 4 \\
        --time-limit-seconds 60 \\
        --max-virtual-chain 5

参数收敛分 3 个阶段（不混跑）：

Stage A: 候选块供给参数收敛
    配置: A1_baseline, A2_relaxed, A3_wider
    参数: block_generator_max_blocks_*

Stage B: 有向聚类权重收敛
    配置: B1_current, B2_heavy_tons, B3_heavy_bridge
    参数: directional_cluster_*

Stage C: Block ALNS / Mixed Bridge 收敛
    配置: C1_current, C2_stronger, C3_flexible
    参数: block_alns_*, mixed_bridge_*, block_master_slot_buffer

收敛规则:
    1. 过滤失败行
    2. 按 dropped_count 升序
    3. 相同 dropped_count 按 tail_underfilled_count
    4. 相同 tail 按 runtime 升序
"""

from __future__ import annotations

import argparse
import sys
import traceback
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aps_cp_sat.cold_rolling_pipeline import ColdRollingPipeline
from aps_cp_sat.config.parameters import build_profile_config
from aps_cp_sat.domain.models import ColdRollingRequest
from aps_cp_sat.model.experiment_summary import (
    recommend_stage_config,
    write_param_sweep_outputs,
)
from aps_cp_sat.model.experiment_utils import (
    BLOCK_ALNS_CONFIGS,
    BLOCK_GENERATOR_SUPPLY_CONFIGS,
    DIRECTIONAL_CLUSTER_CONFIGS,
    apply_profile_guard,
    collect_experiment_row,
    extract_sweep_params,
    merge_row_with_sweep_params,
)


# =============================================================================
# 默认配置
# =============================================================================

DEFAULT_BASE_PROFILE = "block_first_guarded_search"
DEFAULT_SEEDS = [2027, 2028, 2029]
DEFAULT_ROUNDS = 4
DEFAULT_TIME_LIMIT = 60.0
DEFAULT_MAX_VIRTUAL_CHAIN = 5


# =============================================================================
# 参数更新工具
# =============================================================================


def update_model_params(cfg: Any, params: Dict[str, Any]) -> Any:
    """
    将参数字典应用到 ModelConfig。

    Parameters
    ----------
    cfg : PlannerConfig
        原始配置。
    params : dict
        要更新的参数字典。

    Returns
    -------
    PlannerConfig
        更新后的配置。
    """
    new_model = replace(cfg.model)
    for key, value in params.items():
        if hasattr(new_model, key):
            new_model = replace(new_model, **{key: value})
    return replace(cfg, model=new_model)


def build_base_request(
    orders_path: Path,
    steel_info_path: Path,
    output_path: Path,
    profile: str,
    rounds: int,
    time_limit: float,
    max_virtual_chain: int,
    extra_params: Optional[Dict[str, Any]] = None,
) -> ColdRollingRequest:
    """
    构建基础请求对象。

    Parameters
    ----------
    extra_params : dict, optional
        额外的模型参数。
    """
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

    if extra_params:
        cfg = update_model_params(cfg, extra_params)

    return ColdRollingRequest(
        orders_path=orders_path,
        steel_info_path=steel_info_path,
        output_path=output_path,
        config=cfg,
    )


# =============================================================================
# 单轮实验执行
# =============================================================================


def run_single_sweep_experiment(
    config_name: str,
    stage_name: str,
    profile: str,
    seed: int,
    req: ColdRollingRequest,
    output_path: Path,
) -> dict:
    """
    执行单轮 sweep 实验。

    Returns
    -------
    dict
        包含实验结果和 sweep 参数的行数据。
    """
    print(f"\n  [{stage_name}] {config_name} | seed={seed}")
    print(f"    output={output_path.name}")

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
        print(f"    [ERROR] {error_message}")
        traceback.print_exc()

    elapsed = perf_counter() - start_time
    print(f"    elapsed={elapsed:.1f}s {'✓' if error_message is None else '✗'}")

    # 采集结果
    row = collect_experiment_row(
        result=result,
        profile_name=profile,
        seed=seed,
        output_path=output_path,
        elapsed_seconds=elapsed,
        run_error=error_message,
    )

    # 提取并合并 sweep 参数
    sweep_params = extract_sweep_params(req.config)
    row = merge_row_with_sweep_params(
        row=row,
        sweep_params=sweep_params,
        stage_name=stage_name,
        config_name=config_name,
    )

    return row


# =============================================================================
# 阶段执行
# =============================================================================


def run_stage_a(
    orders_path: Path,
    steel_info_path: Path,
    output_dir: Path,
    seeds: List[int],
    rounds: int,
    time_limit: float,
    max_virtual_chain: int,
) -> List[dict]:
    """
    Stage A: 候选块供给参数收敛。
    """
    print(f"\n{'='*60}")
    print(f"Stage A: Candidate Block Supply Convergence")
    print(f"{'='*60}")

    stage_dir = output_dir / "stage_a"
    stage_dir.mkdir(parents=True, exist_ok=True)

    rows: List[dict] = []
    base_output_path = stage_dir / "output.xlsx"

    for config_name, supply_params in BLOCK_GENERATOR_SUPPLY_CONFIGS.items():
        print(f"\n>> Config: {config_name}")
        for seed in seeds:
            req = build_base_request(
                orders_path=orders_path,
                steel_info_path=steel_info_path,
                output_path=base_output_path,
                profile=DEFAULT_BASE_PROFILE,
                rounds=rounds,
                time_limit=time_limit,
                max_virtual_chain=max_virtual_chain,
                extra_params=supply_params,
            )
            row = run_single_sweep_experiment(
                config_name=config_name,
                stage_name="Stage_A",
                profile=DEFAULT_BASE_PROFILE,
                seed=seed,
                req=req,
                output_path=base_output_path,
            )
            rows.append(row)

    # 推荐
    rec = recommend_stage_config(rows, "Stage_A")
    print(f"\n[Stage_A] Recommended: {rec['recommended_config_name']} ({rec['reason']})")

    return rows


def run_stage_b(
    orders_path: Path,
    steel_info_path: Path,
    output_dir: Path,
    seeds: List[int],
    rounds: int,
    time_limit: float,
    max_virtual_chain: int,
    stage_a_best_config: str,
    stage_a_rows: List[dict],
) -> List[dict]:
    """
    Stage B: 有向聚类权重收敛。
    在 Stage A 最佳供给配置上继续。
    """
    print(f"\n{'='*60}")
    print(f"Stage B: Directional Clustering Weights Convergence")
    print(f"  Using Stage_A best config: {stage_a_best_config}")
    print(f"{'='*60}")

    # 获取 Stage A 最佳供给参数
    best_supply = BLOCK_GENERATOR_SUPPLY_CONFIGS.get(stage_a_best_config, {})
    if not best_supply:
        print(f"[Warning] Stage A best config '{stage_a_best_config}' not found, using A1_baseline")
        best_supply = BLOCK_GENERATOR_SUPPLY_CONFIGS.get("A1_baseline", {})

    stage_dir = output_dir / "stage_b"
    stage_dir.mkdir(parents=True, exist_ok=True)

    rows: List[dict] = []
    base_output_path = stage_dir / "output.xlsx"

    for config_name, cluster_params in DIRECTIONAL_CLUSTER_CONFIGS.items():
        print(f"\n>> Config: {config_name}")
        for seed in seeds:
            # 合并 Stage A 供给参数 + Stage B 聚类参数
            extra_params = {**best_supply, **cluster_params}

            req = build_base_request(
                orders_path=orders_path,
                steel_info_path=steel_info_path,
                output_path=base_output_path,
                profile=DEFAULT_BASE_PROFILE,
                rounds=rounds,
                time_limit=time_limit,
                max_virtual_chain=max_virtual_chain,
                extra_params=extra_params,
            )
            row = run_single_sweep_experiment(
                config_name=config_name,
                stage_name="Stage_B",
                profile=DEFAULT_BASE_PROFILE,
                seed=seed,
                req=req,
                output_path=base_output_path,
            )
            rows.append(row)

    # 推荐
    rec = recommend_stage_config(rows, "Stage_B")
    print(f"\n[Stage_B] Recommended: {rec['recommended_config_name']} ({rec['reason']})")

    return rows


def run_stage_c(
    orders_path: Path,
    steel_info_path: Path,
    output_dir: Path,
    seeds: List[int],
    rounds: int,
    time_limit: float,
    max_virtual_chain: int,
    stage_a_best_config: str,
    stage_b_best_config: str,
    stage_a_rows: List[dict],
    stage_b_rows: List[dict],
) -> List[dict]:
    """
    Stage C: Block ALNS / Mixed Bridge 收敛。
    在 Stage A + Stage B 最佳配置上继续。
    """
    print(f"\n{'='*60}")
    print(f"Stage C: Block ALNS / Mixed Bridge Convergence")
    print(f"  Using Stage_A best: {stage_a_best_config}")
    print(f"  Using Stage_B best: {stage_b_best_config}")
    print(f"{'='*60}")

    # 获取 Stage A 最佳供给参数
    best_supply = BLOCK_GENERATOR_SUPPLY_CONFIGS.get(stage_a_best_config, {})
    if not best_supply:
        best_supply = BLOCK_GENERATOR_SUPPLY_CONFIGS.get("A1_baseline", {})

    # 获取 Stage B 最佳聚类参数
    best_cluster = DIRECTIONAL_CLUSTER_CONFIGS.get(stage_b_best_config, {})
    if not best_cluster:
        best_cluster = DIRECTIONAL_CLUSTER_CONFIGS.get("B1_current", {})

    stage_dir = output_dir / "stage_c"
    stage_dir.mkdir(parents=True, exist_ok=True)

    rows: List[dict] = []
    base_output_path = stage_dir / "output.xlsx"

    for config_name, alns_params in BLOCK_ALNS_CONFIGS.items():
        print(f"\n>> Config: {config_name}")
        for seed in seeds:
            # 合并所有阶段参数
            extra_params = {**best_supply, **best_cluster, **alns_params}

            req = build_base_request(
                orders_path=orders_path,
                steel_info_path=steel_info_path,
                output_path=base_output_path,
                profile=DEFAULT_BASE_PROFILE,
                rounds=rounds,
                time_limit=time_limit,
                max_virtual_chain=max_virtual_chain,
                extra_params=extra_params,
            )
            row = run_single_sweep_experiment(
                config_name=config_name,
                stage_name="Stage_C",
                profile=DEFAULT_BASE_PROFILE,
                seed=seed,
                req=req,
                output_path=base_output_path,
            )
            rows.append(row)

    # 推荐
    rec = recommend_stage_config(rows, "Stage_C")
    print(f"\n[Stage_C] Recommended: {rec['recommended_config_name']} ({rec['reason']})")

    return rows


# =============================================================================
# 主入口
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Block-first 参数收敛实验脚本",
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
        default=Path("outputs/param_sweep"),
        help="输出目录 (默认: outputs/param_sweep)",
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
        "--profile",
        type=str,
        default=DEFAULT_BASE_PROFILE,
        help=f"基础 profile (默认: {DEFAULT_BASE_PROFILE})",
    )

    args = parser.parse_args()

    print(f"\n{'#'*60}")
    print(f"# Block-first Parameter Sweep Experiment")
    print(f"# Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"# {'#'*60}")
    print(f"# Orders: {args.orders}")
    print(f"# Steel Info: {args.steel_info}")
    print(f"# Profile: {args.profile}")
    print(f"# Seeds: {args.seeds}")
    print(f"# Rounds: {args.rounds}")
    print(f"# Time Limit: {args.time_limit_seconds}s")
    print(f"# Max Virtual Chain: {args.max_virtual_chain}")
    print(f"# Output Dir: {args.output_dir}")
    print(f"# {'#'*60}\n")

    # 创建输出目录
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Stage A ----
    stage_a_rows = run_stage_a(
        orders_path=args.orders,
        steel_info_path=args.steel_info,
        output_dir=args.output_dir,
        seeds=args.seeds,
        rounds=args.rounds,
        time_limit=args.time_limit_seconds,
        max_virtual_chain=args.max_virtual_chain,
    )

    rec_a = recommend_stage_config(stage_a_rows, "Stage_A")
    best_a = rec_a.get("recommended_config_name", "A1_baseline")

    # ---- Stage B ----
    stage_b_rows = run_stage_b(
        orders_path=args.orders,
        steel_info_path=args.steel_info,
        output_dir=args.output_dir,
        seeds=args.seeds,
        rounds=args.rounds,
        time_limit=args.time_limit_seconds,
        max_virtual_chain=args.max_virtual_chain,
        stage_a_best_config=best_a,
        stage_a_rows=stage_a_rows,
    )

    rec_b = recommend_stage_config(stage_b_rows, "Stage_B")
    best_b = rec_b.get("recommended_config_name", "B1_current")

    # ---- Stage C ----
    stage_c_rows = run_stage_c(
        orders_path=args.orders,
        steel_info_path=args.steel_info,
        output_dir=args.output_dir,
        seeds=args.seeds,
        rounds=args.rounds,
        time_limit=args.time_limit_seconds,
        max_virtual_chain=args.max_virtual_chain,
        stage_a_best_config=best_a,
        stage_b_best_config=best_b,
        stage_a_rows=stage_a_rows,
        stage_b_rows=stage_b_rows,
    )

    rec_c = recommend_stage_config(stage_c_rows, "Stage_C")

    # ---- 输出 ----
    setup = {
        "orders_file": str(args.orders),
        "steel_info_file": str(args.steel_info),
        "profile": args.profile,
        "seeds": args.seeds,
        "rounds": args.rounds,
        "time_limit_seconds": args.time_limit_seconds,
        "max_virtual_chain": args.max_virtual_chain,
    }

    csv_path, md_path = write_param_sweep_outputs(
        stage_a_rows=stage_a_rows,
        stage_b_rows=stage_b_rows,
        stage_c_rows=stage_c_rows,
        output_dir=args.output_dir,
        setup=setup,
    )

    print(f"\n[Output] CSV: {csv_path}")
    print(f"[Output] Markdown: {md_path}")

    # 打印最终推荐
    print(f"\n{'='*60}")
    print(f"# Final Recommended Config")
    print(f"{'='*60}")
    print(f"Stage A: {rec_a['recommended_config_name']} ({rec_a['reason']})")
    print(f"Stage B: {rec_b['recommended_config_name']} ({rec_b['reason']})")
    print(f"Stage C: {rec_c['recommended_config_name']} ({rec_c['reason']})")
    print(f"\n收敛流程: Stage_A ({rec_a['recommended_config_name']})")
    print(f"       → Stage_B ({rec_b['recommended_config_name']})")
    print(f"       → Stage_C ({rec_c['recommended_config_name']})")

    print(f"\n{'#'*60}")
    print(f"# Parameter Sweep Completed")
    print(f"# Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"# {'#'*60}")


if __name__ == "__main__":
    main()
