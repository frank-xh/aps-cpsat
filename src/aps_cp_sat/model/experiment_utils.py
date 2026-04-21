"""
experiment_utils.py - Block-first 实验线统一结果采集工具

提供统一的实验结果采集接口，从 ColdRollingResult.engine_meta 中提取关键字段，
支持 A/B 对比实验与参数收敛实验的数据采集需求。

典型用法:
    from aps_cp_sat.model.experiment_utils import collect_experiment_row, validate_profile_guard

    # 单轮采集
    row = collect_experiment_row(result, "block_first_guarded_search", seed=2027,
                                  output_path=Path("output.xlsx"), elapsed_seconds=45.3)

    # Profile 校验
    guard_ok, reason = validate_profile_guard(row, "block_first_guarded_search")
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

# =============================================================================
# 字段定义
# =============================================================================

# ---- 基础结果字段 ----
BASIC_FIELDS = [
    "profile_name",
    "seed",
    "output_file",
    "solver_path",
    "main_path",
    "acceptance",
    "acceptance_gate_reason",
    "validation_gate_reason",
    "scheduled_real_orders",
    "dropped_count",
    "campaign_count",
    "tail_underfilled_count",
    "total_runtime_seconds",
]

# ---- 桥接 / family / mixed bridge 字段 ----
BRIDGE_FIELDS = [
    "selected_real_bridge_edge_count",
    "selected_virtual_bridge_family_edge_count",
    "selected_legacy_virtual_bridge_edge_count",
    "max_bridge_count_used",
    "greedy_virtual_family_edge_uses",
    "greedy_virtual_family_budget_blocked_count",
    "alns_virtual_family_attempt_count",
    "alns_virtual_family_accept_count",
    "local_cpsat_virtual_family_selected_count",
    "mixed_bridge_attempt_count",
    "mixed_bridge_success_count",
    "mixed_bridge_reject_count",
]

# ---- Block-first 关键字段 ----
BLOCK_FIRST_FIELDS = [
    "generated_blocks_total",
    "selected_blocks_count",
    "selected_order_coverage",
    "block_master_dropped_count",
    "block_alns_rounds_attempted",
    "block_alns_rounds_accepted",
    "block_swap_attempt_count",
    "block_replace_attempt_count",
    "block_split_attempt_count",
    "block_merge_attempt_count",
    "block_boundary_rebalance_attempt_count",
    "block_internal_rebuild_attempt_count",
    "avg_block_quality_score",
    "avg_block_tons_in_selected",
    "block_transition_avg_cost",
    "orders_in_realized_blocks",
    "avg_realized_block_quality",
    "block_generation_seconds",
    "block_master_seconds",
    "block_realization_seconds",
    "block_alns_seconds",
]

# ---- Candidate graph / path 字段 ----
GRAPH_CONFIG_FIELDS = [
    "constructive_edge_policy",
    "allow_real_bridge_edge_in_constructive",
    "allow_virtual_bridge_edge_in_constructive",
    "bridge_expansion_mode",
    "candidate_graph_source",
]

# ---- 完整字段列表 ----
ALL_EXPERIMENT_FIELDS = (
    BASIC_FIELDS
    + BRIDGE_FIELDS
    + BLOCK_FIRST_FIELDS
    + GRAPH_CONFIG_FIELDS
    + [
        # 实验元数据
        "run_failed",
        "error_message",
        "profile_guard_failed",
        "profile_guard_reason",
    ]
)


def _safe_get(data: Any, key: str, default: Any = 0) -> Any:
    """安全地从 dict 或对象中获取字段值。"""
    if data is None:
        return default
    if isinstance(data, dict):
        return data.get(key, default)
    # 尝试作为对象属性获取
    return getattr(data, key, default)


def _get_nested(data: Any, *keys: str, default: Any = 0) -> Any:
    """安全地获取嵌套字段。"""
    result = data
    for k in keys:
        if result is None:
            return default
        if isinstance(result, dict):
            result = result.get(k, None)
        else:
            result = getattr(result, k, None)
    return result if result is not None else default


# =============================================================================
# 结果采集
# =============================================================================


def collect_experiment_row(
    result: Any,
    profile_name: str,
    *,
    seed: int,
    output_path: Union[str, Path],
    elapsed_seconds: float,
    run_error: Optional[str] = None,
) -> Dict[str, Any]:
    """
    从 ColdRollingResult 对象中采集实验结果行。

    Parameters
    ----------
    result : ColdRollingResult or None
        管道运行结果对象。
        如果为 None 或运行失败，row 中 run_failed=True。
    profile_name : str
        实验 profile 名称，如 "constructive_lns_search" 或 "block_first_guarded_search"。
    seed : int
        当前实验的随机种子。
    output_path : str or Path
        输出 Excel 文件路径。
    elapsed_seconds : float
        实际运行耗时（秒），从外部计时。
    run_error : str, optional
        如果运行失败，记录错误信息。

    Returns
    -------
    dict
        包含所有采集字段的字典。
    """
    row: Dict[str, Any] = {field: 0 for field in ALL_EXPERIMENT_FIELDS}

    # 实验元数据
    row["profile_name"] = profile_name
    row["seed"] = seed
    row["output_file"] = str(output_path)
    row["total_runtime_seconds"] = elapsed_seconds
    row["run_failed"] = bool(run_error is not None)
    row["error_message"] = run_error or ""

    if run_error is not None:
        return row

    # 提取 engine_meta
    engine_meta: Optional[Dict[str, Any]] = None
    if result is not None:
        engine_meta = getattr(result, "engine_meta", None)
        if engine_meta is None and hasattr(result, "__dict__"):
            engine_meta = result.__dict__.get("engine_meta", {})

    if engine_meta is None:
        engine_meta = {}

    # ---- 基础结果字段 ----
    row["solver_path"] = str(_safe_get(engine_meta, "solver_path", "unknown"))
    row["main_path"] = str(_safe_get(engine_meta, "main_path", "unknown"))
    row["acceptance"] = str(_safe_get(engine_meta, "acceptance", "unknown"))
    row["acceptance_gate_reason"] = str(_safe_get(engine_meta, "acceptance_gate_reason", ""))
    row["validation_gate_reason"] = str(_safe_get(engine_meta, "validation_gate_reason", ""))
    row["scheduled_real_orders"] = int(_safe_get(engine_meta, "scheduled_real_orders", 0))
    row["dropped_count"] = int(_safe_get(engine_meta, "dropped_count", 0))
    row["campaign_count"] = int(_safe_get(engine_meta, "campaign_count", 0))
    row["tail_underfilled_count"] = int(_safe_get(engine_meta, "tail_underfilled_count", 0))

    # ---- 桥接 / family / mixed bridge 字段 ----
    row["selected_real_bridge_edge_count"] = int(
        _safe_get(engine_meta, "selected_real_bridge_edge_count", 0)
    )
    row["selected_virtual_bridge_family_edge_count"] = int(
        _safe_get(engine_meta, "selected_virtual_bridge_family_edge_count", 0)
    )
    row["selected_legacy_virtual_bridge_edge_count"] = int(
        _safe_get(engine_meta, "selected_legacy_virtual_bridge_edge_count", 0)
    )
    row["max_bridge_count_used"] = int(_safe_get(engine_meta, "max_bridge_count_used", 0))
    row["greedy_virtual_family_edge_uses"] = int(
        _safe_get(engine_meta, "greedy_virtual_family_edge_uses", 0)
    )
    row["greedy_virtual_family_budget_blocked_count"] = int(
        _safe_get(engine_meta, "greedy_virtual_family_budget_blocked_count", 0)
    )
    row["alns_virtual_family_attempt_count"] = int(
        _safe_get(engine_meta, "alns_virtual_family_attempt_count", 0)
    )
    row["alns_virtual_family_accept_count"] = int(
        _safe_get(engine_meta, "alns_virtual_family_accept_count", 0)
    )
    row["local_cpsat_virtual_family_selected_count"] = int(
        _safe_get(engine_meta, "local_cpsat_virtual_family_selected_count", 0)
    )
    row["mixed_bridge_attempt_count"] = int(
        _safe_get(engine_meta, "mixed_bridge_attempt_count", 0)
    )
    row["mixed_bridge_success_count"] = int(
        _safe_get(engine_meta, "mixed_bridge_success_count", 0)
    )
    row["mixed_bridge_reject_count"] = int(
        _safe_get(engine_meta, "mixed_bridge_reject_count", 0)
    )

    # ---- Block-first 关键字段 ----
    row["generated_blocks_total"] = int(_safe_get(engine_meta, "generated_blocks_total", 0))
    row["selected_blocks_count"] = int(_safe_get(engine_meta, "selected_blocks_count", 0))
    row["selected_order_coverage"] = float(_safe_get(engine_meta, "selected_order_coverage", 0.0))
    row["block_master_dropped_count"] = int(
        _safe_get(engine_meta, "block_master_dropped_count", 0)
    )
    row["block_alns_rounds_attempted"] = int(
        _safe_get(engine_meta, "block_alns_rounds_attempted", 0)
    )
    row["block_alns_rounds_accepted"] = int(
        _safe_get(engine_meta, "block_alns_rounds_accepted", 0)
    )
    row["block_swap_attempt_count"] = int(_safe_get(engine_meta, "block_swap_attempt_count", 0))
    row["block_replace_attempt_count"] = int(
        _safe_get(engine_meta, "block_replace_attempt_count", 0)
    )
    row["block_split_attempt_count"] = int(_safe_get(engine_meta, "block_split_attempt_count", 0))
    row["block_merge_attempt_count"] = int(_safe_get(engine_meta, "block_merge_attempt_count", 0))
    row["block_boundary_rebalance_attempt_count"] = int(
        _safe_get(engine_meta, "block_boundary_rebalance_attempt_count", 0)
    )
    row["block_internal_rebuild_attempt_count"] = int(
        _safe_get(engine_meta, "block_internal_rebuild_attempt_count", 0)
    )
    row["avg_block_quality_score"] = float(
        _safe_get(engine_meta, "avg_block_quality_score", 0.0)
    )
    row["avg_block_tons_in_selected"] = float(
        _safe_get(engine_meta, "avg_block_tons_in_selected", 0.0)
    )
    row["block_transition_avg_cost"] = float(
        _safe_get(engine_meta, "block_transition_avg_cost", 0.0)
    )
    row["orders_in_realized_blocks"] = int(
        _safe_get(engine_meta, "orders_in_realized_blocks", 0)
    )
    row["avg_realized_block_quality"] = float(
        _safe_get(engine_meta, "avg_realized_block_quality", 0.0)
    )
    row["block_generation_seconds"] = float(
        _safe_get(engine_meta, "block_generation_seconds", 0.0)
    )
    row["block_master_seconds"] = float(_safe_get(engine_meta, "block_master_seconds", 0.0))
    row["block_realization_seconds"] = float(
        _safe_get(engine_meta, "block_realization_seconds", 0.0)
    )
    row["block_alns_seconds"] = float(_safe_get(engine_meta, "block_alns_seconds", 0.0))

    # ---- Candidate graph / path 字段 ----
    row["constructive_edge_policy"] = str(
        _safe_get(engine_meta, "constructive_edge_policy", "unknown")
    )
    row["allow_real_bridge_edge_in_constructive"] = bool(
        _safe_get(engine_meta, "allow_real_bridge_edge_in_constructive", False)
    )
    row["allow_virtual_bridge_edge_in_constructive"] = bool(
        _safe_get(engine_meta, "allow_virtual_bridge_edge_in_constructive", False)
    )
    row["bridge_expansion_mode"] = str(
        _safe_get(engine_meta, "bridge_expansion_mode", "disabled")
    )
    row["candidate_graph_source"] = str(
        _safe_get(engine_meta, "candidate_graph_source", "unknown")
    )

    return row


def collect_experiment_row_from_dict(
    meta_dict: Dict[str, Any],
    profile_name: str,
    *,
    seed: int,
    output_path: Union[str, Path],
    elapsed_seconds: float,
    run_error: Optional[str] = None,
) -> Dict[str, Any]:
    """
    从 engine_meta dict 直接采集实验结果行（不需要 ColdRollingResult 对象）。

    Parameters
    ----------
    meta_dict : dict
        engine_meta 字典。
    profile_name : str
        实验 profile 名称。
    seed : int
        当前实验的随机种子。
    output_path : str or Path
        输出 Excel 文件路径。
    elapsed_seconds : float
        实际运行耗时（秒）。
    run_error : str, optional
        如果运行失败，记录错误信息。

    Returns
    -------
    dict
        包含所有采集字段的字典。
    """
    row: Dict[str, Any] = {field: 0 for field in ALL_EXPERIMENT_FIELDS}

    row["profile_name"] = profile_name
    row["seed"] = seed
    row["output_file"] = str(output_path)
    row["total_runtime_seconds"] = elapsed_seconds
    row["run_failed"] = bool(run_error is not None)
    row["error_message"] = run_error or ""

    if run_error is not None:
        return row

    engine_meta = meta_dict or {}

    # 复用 collect_experiment_row 的逻辑，但传入已提取的 engine_meta
    row["solver_path"] = str(_safe_get(engine_meta, "solver_path", "unknown"))
    row["main_path"] = str(_safe_get(engine_meta, "main_path", "unknown"))
    row["acceptance"] = str(_safe_get(engine_meta, "acceptance", "unknown"))
    row["acceptance_gate_reason"] = str(_safe_get(engine_meta, "acceptance_gate_reason", ""))
    row["validation_gate_reason"] = str(_safe_get(engine_meta, "validation_gate_reason", ""))
    row["scheduled_real_orders"] = int(_safe_get(engine_meta, "scheduled_real_orders", 0))
    row["dropped_count"] = int(_safe_get(engine_meta, "dropped_count", 0))
    row["campaign_count"] = int(_safe_get(engine_meta, "campaign_count", 0))
    row["tail_underfilled_count"] = int(_safe_get(engine_meta, "tail_underfilled_count", 0))
    row["selected_real_bridge_edge_count"] = int(
        _safe_get(engine_meta, "selected_real_bridge_edge_count", 0)
    )
    row["selected_virtual_bridge_family_edge_count"] = int(
        _safe_get(engine_meta, "selected_virtual_bridge_family_edge_count", 0)
    )
    row["selected_legacy_virtual_bridge_edge_count"] = int(
        _safe_get(engine_meta, "selected_legacy_virtual_bridge_edge_count", 0)
    )
    row["max_bridge_count_used"] = int(_safe_get(engine_meta, "max_bridge_count_used", 0))
    row["greedy_virtual_family_edge_uses"] = int(
        _safe_get(engine_meta, "greedy_virtual_family_edge_uses", 0)
    )
    row["greedy_virtual_family_budget_blocked_count"] = int(
        _safe_get(engine_meta, "greedy_virtual_family_budget_blocked_count", 0)
    )
    row["alns_virtual_family_attempt_count"] = int(
        _safe_get(engine_meta, "alns_virtual_family_attempt_count", 0)
    )
    row["alns_virtual_family_accept_count"] = int(
        _safe_get(engine_meta, "alns_virtual_family_accept_count", 0)
    )
    row["local_cpsat_virtual_family_selected_count"] = int(
        _safe_get(engine_meta, "local_cpsat_virtual_family_selected_count", 0)
    )
    row["mixed_bridge_attempt_count"] = int(
        _safe_get(engine_meta, "mixed_bridge_attempt_count", 0)
    )
    row["mixed_bridge_success_count"] = int(
        _safe_get(engine_meta, "mixed_bridge_success_count", 0)
    )
    row["mixed_bridge_reject_count"] = int(
        _safe_get(engine_meta, "mixed_bridge_reject_count", 0)
    )
    row["generated_blocks_total"] = int(_safe_get(engine_meta, "generated_blocks_total", 0))
    row["selected_blocks_count"] = int(_safe_get(engine_meta, "selected_blocks_count", 0))
    row["selected_order_coverage"] = float(_safe_get(engine_meta, "selected_order_coverage", 0.0))
    row["block_master_dropped_count"] = int(
        _safe_get(engine_meta, "block_master_dropped_count", 0)
    )
    row["block_alns_rounds_attempted"] = int(
        _safe_get(engine_meta, "block_alns_rounds_attempted", 0)
    )
    row["block_alns_rounds_accepted"] = int(
        _safe_get(engine_meta, "block_alns_rounds_accepted", 0)
    )
    row["block_swap_attempt_count"] = int(_safe_get(engine_meta, "block_swap_attempt_count", 0))
    row["block_replace_attempt_count"] = int(
        _safe_get(engine_meta, "block_replace_attempt_count", 0)
    )
    row["block_split_attempt_count"] = int(_safe_get(engine_meta, "block_split_attempt_count", 0))
    row["block_merge_attempt_count"] = int(_safe_get(engine_meta, "block_merge_attempt_count", 0))
    row["block_boundary_rebalance_attempt_count"] = int(
        _safe_get(engine_meta, "block_boundary_rebalance_attempt_count", 0)
    )
    row["block_internal_rebuild_attempt_count"] = int(
        _safe_get(engine_meta, "block_internal_rebuild_attempt_count", 0)
    )
    row["avg_block_quality_score"] = float(
        _safe_get(engine_meta, "avg_block_quality_score", 0.0)
    )
    row["avg_block_tons_in_selected"] = float(
        _safe_get(engine_meta, "avg_block_tons_in_selected", 0.0)
    )
    row["block_transition_avg_cost"] = float(
        _safe_get(engine_meta, "block_transition_avg_cost", 0.0)
    )
    row["orders_in_realized_blocks"] = int(
        _safe_get(engine_meta, "orders_in_realized_blocks", 0)
    )
    row["avg_realized_block_quality"] = float(
        _safe_get(engine_meta, "avg_realized_block_quality", 0.0)
    )
    row["block_generation_seconds"] = float(
        _safe_get(engine_meta, "block_generation_seconds", 0.0)
    )
    row["block_master_seconds"] = float(_safe_get(engine_meta, "block_master_seconds", 0.0))
    row["block_realization_seconds"] = float(
        _safe_get(engine_meta, "block_realization_seconds", 0.0)
    )
    row["block_alns_seconds"] = float(_safe_get(engine_meta, "block_alns_seconds", 0.0))
    row["constructive_edge_policy"] = str(
        _safe_get(engine_meta, "constructive_edge_policy", "unknown")
    )
    row["allow_real_bridge_edge_in_constructive"] = bool(
        _safe_get(engine_meta, "allow_real_bridge_edge_in_constructive", False)
    )
    row["allow_virtual_bridge_edge_in_constructive"] = bool(
        _safe_get(engine_meta, "allow_virtual_bridge_edge_in_constructive", False)
    )
    row["bridge_expansion_mode"] = str(
        _safe_get(engine_meta, "bridge_expansion_mode", "disabled")
    )
    row["candidate_graph_source"] = str(
        _safe_get(engine_meta, "candidate_graph_source", "unknown")
    )

    return row


# =============================================================================
# Profile Guard 校验
# =============================================================================


def validate_profile_guard(
    row: Dict[str, Any],
    expected_profile: str,
) -> Tuple[bool, str]:
    """
    校验实验行是否符合预期 profile 的路径特征。

    Parameters
    ----------
    row : dict
        collect_experiment_row 采集的行数据。
    expected_profile : str
        预期 profile 名称。

    Returns
    -------
    (passed, reason) : (bool, str)
        passed=True 表示校验通过；passed=False 时 reason 说明原因。
    """
    profile = str(expected_profile).lower()
    solver_path = str(row.get("solver_path", "")).lower()
    main_path = str(row.get("main_path", "")).lower()
    allow_real = bool(row.get("allow_real_bridge_edge_in_constructive", False))
    allow_virtual = bool(row.get("allow_virtual_bridge_edge_in_constructive", False))

    if profile in {"constructive_lns_search", "constructive_lns_real_bridge_frontload"}:
        # A 组：constructive_lns_search
        # 要求：solver_path != "block_first"，allow_real=True，allow_virtual=False
        if solver_path == "block_first":
            return False, f"expected solver_path != 'block_first' for {expected_profile}, got '{solver_path}'"
        if not allow_real:
            return False, f"expected allow_real_bridge_edge_in_constructive=True for {expected_profile}"
        if allow_virtual:
            return False, f"expected allow_virtual_bridge_edge_in_constructive=False for {expected_profile}"
        return True, "ok"

    if profile == "constructive_lns_direct_only_baseline":
        # Baseline：不允许 real bridge 和 virtual bridge
        if solver_path == "block_first":
            return False, f"expected solver_path != 'block_first' for {expected_profile}"
        if allow_real:
            return False, f"expected allow_real_bridge_edge_in_constructive=False for {expected_profile}"
        if allow_virtual:
            return False, f"expected allow_virtual_bridge_edge_in_constructive=False for {expected_profile}"
        return True, "ok"

    if profile == "constructive_lns_virtual_guarded_frontload":
        # C 组：guarded virtual family
        # 要求：allow_real=True，allow_virtual=True，block_first 不参与
        if solver_path == "block_first":
            return False, f"expected solver_path != 'block_first' for {expected_profile}"
        if not allow_real:
            return False, f"expected allow_real_bridge_edge_in_constructive=True for {expected_profile}"
        if not allow_virtual:
            return False, f"expected allow_virtual_bridge_edge_in_constructive=True for {expected_profile}"
        return True, "ok"

    if profile == "block_first_guarded_search":
        # B 组：block-first
        # 要求：solver_path == "block_first"，且有 block-first 特有字段
        if solver_path != "block_first":
            return False, f"expected solver_path='block_first' for {expected_profile}, got '{solver_path}'"
        if main_path != "block_first":
            return False, f"expected main_path='block_first' for {expected_profile}, got '{main_path}'"
        # 检查 block-first 特有字段存在
        if row.get("mixed_bridge_attempt_count", -1) < 0:
            return False, f"expected mixed_bridge_attempt_count field for {expected_profile}"
        if row.get("selected_blocks_count", -1) < 0:
            return False, f"expected selected_blocks_count field for {expected_profile}"
        return True, "ok"

    # 未知 profile，不校验
    return True, "ok"


def apply_profile_guard(
    rows: List[Dict[str, Any]],
    profile_name: str,
) -> List[Dict[str, Any]]:
    """
    对一组实验行应用 profile guard 校验，在行中添加 guard 结果。

    Parameters
    ----------
    rows : list[dict]
        实验行列表。
    profile_name : str
        预期 profile 名称。

    Returns
    -------
    list[dict]
        每个行增加 profile_guard_failed 和 profile_guard_reason 字段。
    """
    result = []
    for row in rows:
        passed, reason = validate_profile_guard(row, profile_name)
        row_copy = dict(row)
        row_copy["profile_guard_failed"] = not passed
        row_copy["profile_guard_reason"] = reason
        result.append(row_copy)
    return result


# =============================================================================
# CSV I/O
# =============================================================================


def write_results_csv(
    rows: List[Dict[str, Any]],
    output_path: Union[str, Path],
) -> None:
    """
    将实验结果行写入 CSV 文件。

    Parameters
    ----------
    rows : list[dict]
        实验结果行列表。
    output_path : str or Path
        输出 CSV 路径。
    """
    if not rows:
        # 写入带表头的空文件
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=ALL_EXPERIMENT_FIELDS)
            writer.writeheader()
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 确保字段顺序一致
    ordered_rows = []
    for row in rows:
        ordered_row = {field: row.get(field, 0) for field in ALL_EXPERIMENT_FIELDS}
        ordered_rows.append(ordered_row)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ALL_EXPERIMENT_FIELDS)
        writer.writeheader()
        writer.writerows(ordered_rows)


def read_results_csv(
    csv_path: Union[str, Path],
) -> pd.DataFrame:
    """
    读取实验结果 CSV 文件。

    Parameters
    ----------
    csv_path : str or Path
        CSV 文件路径。

    Returns
    -------
    pd.DataFrame
        实验结果 DataFrame。
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return pd.DataFrame(columns=ALL_EXPERIMENT_FIELDS)
    return pd.read_csv(csv_path, encoding="utf-8")


# =============================================================================
# 配置参数辅助
# =============================================================================


def extract_sweep_params(
    cfg: Any,
) -> Dict[str, Any]:
    """
    从 PlannerConfig 中提取与 block-first sweep 相关的参数。

    Parameters
    ----------
    cfg : PlannerConfig
        配置对象。

    Returns
    -------
    dict
        当前配置的 sweep 参数快照。
    """
    model = getattr(cfg, "model", cfg) if hasattr(cfg, "model") else cfg

    return {
        # Stage A: 候选块供给参数
        "block_generator_max_blocks_per_line": int(getattr(model, "block_generator_max_blocks_per_line", 30)),
        "block_generator_max_blocks_total": int(getattr(model, "block_generator_max_blocks_total", 80)),
        "block_generator_max_seed_per_bucket": int(getattr(model, "block_generator_max_seed_per_bucket", 8)),
        "block_generator_max_orders_per_block": int(getattr(model, "block_generator_max_orders_per_block", 20)),
        # Stage B: 有向聚类权重参数
        "directional_cluster_group_weight": float(getattr(model, "directional_cluster_group_weight", 1.2)),
        "directional_cluster_tons_fill_weight": float(getattr(model, "directional_cluster_tons_fill_weight", 1.0)),
        "directional_cluster_real_bridge_bonus": float(getattr(model, "directional_cluster_real_bridge_bonus", 0.8)),
        "directional_cluster_guarded_family_bonus": float(getattr(model, "directional_cluster_guarded_family_bonus", 0.5)),
        "directional_cluster_mixed_bridge_potential_bonus": float(getattr(model, "directional_cluster_mixed_bridge_potential_bonus", 0.3)),
        # Stage C: ALNS / mixed bridge 参数
        "block_alns_rounds": int(getattr(model, "block_alns_rounds", 6)),
        "block_alns_early_stop_no_improve_rounds": int(getattr(model, "block_alns_early_stop_no_improve_rounds", 2)),
        "mixed_bridge_max_attempts_per_block": int(getattr(model, "mixed_bridge_max_attempts_per_block", 10)),
        "block_master_slot_buffer": int(getattr(model, "block_master_slot_buffer", 2)),
    }


def merge_row_with_sweep_params(
    row: Dict[str, Any],
    sweep_params: Dict[str, Any],
    stage_name: Optional[str] = None,
    config_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    将 sweep 参数合并到实验结果行中。

    Parameters
    ----------
    row : dict
        实验结果行。
    sweep_params : dict
        extract_sweep_params 提取的参数。
    stage_name : str, optional
        当前阶段名称，如 "Stage_A"。
    config_name : str, optional
        当前配置名称，如 "A2"。

    Returns
    -------
    dict
        合并后的行。
    """
    result = dict(row)
    result["stage_name"] = stage_name or ""
    result["config_name"] = config_name or ""
    for k, v in sweep_params.items():
        result[f"sweep_{k}"] = v
    return result


# =============================================================================
# Block-first 默认供给配置
# =============================================================================

# Stage A 供给配置
BLOCK_GENERATOR_SUPPLY_CONFIGS = {
    "A1_baseline": {
        "block_generator_max_blocks_per_line": 30,
        "block_generator_max_blocks_total": 80,
        "block_generator_max_seed_per_bucket": 8,
        "block_generator_max_orders_per_block": 20,
    },
    "A2_relaxed": {
        "block_generator_max_blocks_per_line": 45,
        "block_generator_max_blocks_total": 120,
        "block_generator_max_seed_per_bucket": 12,
        "block_generator_max_orders_per_block": 24,
    },
    "A3_wider": {
        "block_generator_max_blocks_per_line": 60,
        "block_generator_max_blocks_total": 160,
        "block_generator_max_seed_per_bucket": 16,
        "block_generator_max_orders_per_block": 24,
    },
}

# Stage B 有向聚类权重配置
DIRECTIONAL_CLUSTER_CONFIGS = {
    "B1_current": {
        "directional_cluster_group_weight": 1.2,
        "directional_cluster_tons_fill_weight": 1.0,
        "directional_cluster_real_bridge_bonus": 0.8,
        "directional_cluster_guarded_family_bonus": 0.5,
        "directional_cluster_mixed_bridge_potential_bonus": 0.3,
    },
    "B2_heavy_tons_and_stability": {
        "directional_cluster_group_weight": 1.4,
        "directional_cluster_tons_fill_weight": 1.2,
        "directional_cluster_real_bridge_bonus": 0.8,
        "directional_cluster_guarded_family_bonus": 0.5,
        "directional_cluster_mixed_bridge_potential_bonus": 0.3,
    },
    "B3_heavy_bridge_potential": {
        "directional_cluster_group_weight": 1.2,
        "directional_cluster_tons_fill_weight": 1.0,
        "directional_cluster_real_bridge_bonus": 1.0,
        "directional_cluster_guarded_family_bonus": 0.7,
        "directional_cluster_mixed_bridge_potential_bonus": 0.5,
    },
}

# Stage C ALNS / mixed bridge 配置
BLOCK_ALNS_CONFIGS = {
    "C1_current": {
        "block_alns_rounds": 6,
        "block_alns_early_stop_no_improve_rounds": 2,
        "mixed_bridge_max_attempts_per_block": 10,
        "block_master_slot_buffer": 2,
    },
    "C2_stronger": {
        "block_alns_rounds": 8,
        "block_alns_early_stop_no_improve_rounds": 3,
        "mixed_bridge_max_attempts_per_block": 12,
        "block_master_slot_buffer": 2,
    },
    "C3_flexible": {
        "block_alns_rounds": 8,
        "block_alns_early_stop_no_improve_rounds": 3,
        "mixed_bridge_max_attempts_per_block": 12,
        "block_master_slot_buffer": 3,
    },
}
