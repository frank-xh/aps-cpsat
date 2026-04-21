"""
tests_smoke_block_first_experiments.py - Block-first 实验线 smoke 测试

包含：
1. test_block_first_ab_summary_smoke        - A/B 汇总 smoke
2. test_block_first_param_stage_selection_smoke - 参数收敛 stage 推荐 smoke

这些测试验证实验基础设施的核心逻辑，不执行实际求解。
"""

from __future__ import annotations

import pytest


# =============================================================================
# Smoke Test 1: A/B 汇总
# =============================================================================


def test_block_first_ab_summary_smoke():
    """
    smoke: A/B 汇总逻辑生成预期的 markdown 结构。

    验证：
    - Markdown 中包含 "Paired Deltas"
    - Markdown 中包含 "block-first"
    - Markdown 中包含 "Δ dropped_count"
    """
    from aps_cp_sat.model.experiment_summary import (
        aggregate_by_profile,
        automatic_interpretation,
        build_ab_summary_markdown,
        paired_deltas,
    )

    # 构造 A/B 两个 fake result rows
    fake_rows = [
        # A 组 (mainline)
        {
            "profile_name": "constructive_lns_search",
            "seed": 2027,
            "dropped_count": 5,
            "tail_underfilled_count": 1,
            "scheduled_real_orders": 195,
            "total_runtime_seconds": 30.0,
            "mixed_bridge_success_count": 0,
            "selected_blocks_count": 0,
        },
        {
            "profile_name": "block_first_guarded_search",
            "seed": 2027,
            "dropped_count": 3,
            "tail_underfilled_count": 1,
            "scheduled_real_orders": 197,
            "total_runtime_seconds": 45.0,
            "mixed_bridge_success_count": 2,
            "selected_blocks_count": 8,
        },
        # A 组 seed 2
        {
            "profile_name": "constructive_lns_search",
            "seed": 2028,
            "dropped_count": 6,
            "tail_underfilled_count": 2,
            "scheduled_real_orders": 194,
            "total_runtime_seconds": 28.0,
            "mixed_bridge_success_count": 0,
            "selected_blocks_count": 0,
        },
        {
            "profile_name": "block_first_guarded_search",
            "seed": 2028,
            "dropped_count": 4,
            "tail_underfilled_count": 1,
            "scheduled_real_orders": 196,
            "total_runtime_seconds": 48.0,
            "mixed_bridge_success_count": 3,
            "selected_blocks_count": 10,
        },
    ]

    import pandas as pd

    df = pd.DataFrame(fake_rows)

    # 调用聚合
    profile_stats = aggregate_by_profile(df)
    assert "constructive_lns_search" in profile_stats
    assert "block_first_guarded_search" in profile_stats

    # 验证均值
    assert profile_stats["constructive_lns_search"]["mean_dropped_count"] == pytest.approx(5.5)
    assert profile_stats["block_first_guarded_search"]["mean_dropped_count"] == pytest.approx(3.5)

    # Paired deltas
    deltas = paired_deltas(df)
    assert "B_minus_A" in deltas
    assert "dropped_count_mean" in deltas["B_minus_A"]
    # (3+4)/2 - (5+6)/2 = 3.5 - 5.5 = -2.0
    assert deltas["B_minus_A"]["dropped_count_mean"] == pytest.approx(-2.0)

    # Automatic interpretation
    interpretation = automatic_interpretation(profile_stats, deltas)
    # B 组 dropped < A 组，tail 相同，scheduled >= A，runtime <= 1.6x
    # 应触发规则 1
    assert "block-first" in interpretation
    assert "明显正向" in interpretation

    # 生成完整 markdown
    setup = {
        "orders_file": "data_orders.xlsx",
        "steel_info_file": "data_steel_info.xlsx",
        "profiles": ["constructive_lns_search", "block_first_guarded_search"],
        "seeds": [2027, 2028],
        "rounds": 4,
        "time_limit_seconds": 60.0,
        "max_virtual_chain": 5,
    }

    md = build_ab_summary_markdown(df, setup)

    # 验证关键结构
    assert "Paired Deltas" in md
    assert "block-first" in md
    assert "Δ dropped_count" in md
    assert "Raw Run Table" in md
    assert "Aggregated by Profile" in md
    assert "Automatic Interpretation" in md


def test_block_first_ab_summary_with_c_group():
    """
    smoke: A/B/C 三组汇总。

    验证：
    - Markdown 中包含 C 组的 paired deltas
    """
    from aps_cp_sat.model.experiment_summary import (
        aggregate_by_profile,
        automatic_interpretation,
        build_ab_summary_markdown,
        paired_deltas,
    )

    import pandas as pd

    fake_rows = [
        {"profile_name": "constructive_lns_search", "seed": 2027,
         "dropped_count": 5, "tail_underfilled_count": 1,
         "scheduled_real_orders": 195, "total_runtime_seconds": 30.0,
         "mixed_bridge_success_count": 0, "selected_blocks_count": 0},
        {"profile_name": "block_first_guarded_search", "seed": 2027,
         "dropped_count": 3, "tail_underfilled_count": 1,
         "scheduled_real_orders": 197, "total_runtime_seconds": 45.0,
         "mixed_bridge_success_count": 2, "selected_blocks_count": 8},
        {"profile_name": "constructive_lns_virtual_guarded_frontload", "seed": 2027,
         "dropped_count": 4, "tail_underfilled_count": 1,
         "scheduled_real_orders": 196, "total_runtime_seconds": 35.0,
         "mixed_bridge_success_count": 1, "selected_blocks_count": 0},
    ]

    df = pd.DataFrame(fake_rows)
    deltas = paired_deltas(df, group_c="constructive_lns_virtual_guarded_frontload")

    assert "B_minus_A" in deltas
    assert "C_minus_A" in deltas
    assert "B_minus_C" in deltas

    setup = {
        "orders_file": "data_orders.xlsx",
        "steel_info_file": "data_steel_info.xlsx",
        "profiles": [
            "constructive_lns_search",
            "block_first_guarded_search",
            "constructive_lns_virtual_guarded_frontload",
        ],
        "seeds": [2027],
        "rounds": 4,
        "time_limit_seconds": 60.0,
        "max_virtual_chain": 5,
    }

    md = build_ab_summary_markdown(df, setup, group_c="constructive_lns_virtual_guarded_frontload")
    assert "C (guarded_order) - A (mainline)" in md
    assert "B (block_first) - C (guarded_order)" in md


# =============================================================================
# Smoke Test 2: 参数收敛 Stage 推荐
# =============================================================================


def test_block_first_param_stage_selection_smoke():
    """
    smoke: 参数阶段推荐逻辑输出正确的配置。

    验证：
    - 输出包含 "Final Recommended Block-first Config"
    - 各 stage 推荐正确的 config
    """
    from aps_cp_sat.model.experiment_summary import (
        build_param_sweep_summary_markdown,
        recommend_stage_config,
    )

    # ---- Stage A fake rows ----
    stage_a_rows = [
        {"config_name": "A1_baseline", "run_failed": False,
         "dropped_count": 5, "tail_underfilled_count": 2,
         "total_runtime_seconds": 40.0},
        {"config_name": "A2_relaxed", "run_failed": False,
         "dropped_count": 3, "tail_underfilled_count": 1,
         "total_runtime_seconds": 50.0},  # 最佳
        {"config_name": "A3_wider", "run_failed": True,
         "dropped_count": 0, "tail_underfilled_count": 0,
         "total_runtime_seconds": 0.0},   # 失败，应被排除
    ]

    # ---- Stage B fake rows ----
    stage_b_rows = [
        {"config_name": "B1_current", "run_failed": False,
         "dropped_count": 3, "tail_underfilled_count": 1,
         "total_runtime_seconds": 50.0},
        {"config_name": "B2_heavy_tons", "run_failed": False,
         "dropped_count": 4, "tail_underfilled_count": 1,
         "total_runtime_seconds": 48.0},
        {"config_name": "B3_heavy_bridge", "run_failed": False,
         "dropped_count": 3, "tail_underfilled_count": 2,
         "total_runtime_seconds": 52.0},  # dropped 相同，但 tail 更差
    ]

    # ---- Stage C fake rows ----
    stage_c_rows = [
        {"config_name": "C1_current", "run_failed": False,
         "dropped_count": 3, "tail_underfilled_count": 1,
         "total_runtime_seconds": 50.0},  # 最佳
        {"config_name": "C2_stronger", "run_failed": False,
         "dropped_count": 4, "tail_underfilled_count": 1,
         "total_runtime_seconds": 55.0},
    ]

    # ---- 推荐验证 ----
    rec_a = recommend_stage_config(stage_a_rows, "Stage_A")
    assert rec_a["recommended_config_name"] == "A2_relaxed"
    assert "所有配置均失败" not in rec_a["reason"]

    rec_b = recommend_stage_config(stage_b_rows, "Stage_B")
    assert rec_b["recommended_config_name"] == "B1_current"
    # B1_current 和 B3_heavy_bridge dropped 都是 3，但 B1_current tail=1 更优

    rec_c = recommend_stage_config(stage_c_rows, "Stage_C")
    assert rec_c["recommended_config_name"] == "C1_current"

    # ---- Markdown 生成验证 ----
    setup = {
        "orders_file": "data_orders.xlsx",
        "steel_info_file": "data_steel_info.xlsx",
        "profile": "block_first_guarded_search",
        "seeds": [2027, 2028],
        "time_limit_seconds": 60.0,
        "max_virtual_chain": 5,
    }

    md = build_param_sweep_summary_markdown(
        stage_a_rows, stage_b_rows, stage_c_rows, setup
    )

    assert "Final Recommended Block-first Config" in md
    assert "Stage A: Block Supply" in md
    assert "Stage B: Directional Clustering Weights" in md
    assert "Stage C: Block ALNS" in md
    assert "A2_relaxed" in md
    assert "B1_current" in md
    assert "C1_current" in md
    assert "收敛流程" in md


def test_block_first_param_stage_selection_all_failed():
    """
    smoke: 所有配置都失败时，推荐首个配置并注明原因。
    """
    from aps_cp_sat.model.experiment_summary import recommend_stage_config

    failed_rows = [
        {"config_name": "X1", "run_failed": True,
         "dropped_count": 0, "tail_underfilled_count": 0,
         "total_runtime_seconds": 0.0},
        {"config_name": "X2", "run_failed": True,
         "dropped_count": 0, "tail_underfilled_count": 0,
         "total_runtime_seconds": 0.0},
    ]

    rec = recommend_stage_config(failed_rows, "Stage_X")
    assert rec["recommended_config_name"] == "X1"
    assert "所有配置均失败" in rec["reason"]


def test_block_first_param_stage_selection_empty():
    """
    smoke: 无数据时返回空推荐。
    """
    from aps_cp_sat.model.experiment_summary import recommend_stage_config

    rec = recommend_stage_config([], "Stage_X")
    assert rec["recommended_config_name"] == ""
    assert "无实验数据" in rec["reason"]


# =============================================================================
# Smoke Test 3: Profile Guard
# =============================================================================


def test_profile_guard_constructive_lns():
    """
    smoke: constructive_lns_search 的 profile guard 校验。
    """
    from aps_cp_sat.model.experiment_utils import validate_profile_guard

    # A 组合法行
    a_row = {
        "solver_path": "constructive_lns",
        "main_path": "constructive_lns",
        "allow_real_bridge_edge_in_constructive": True,
        "allow_virtual_bridge_edge_in_constructive": False,
        "mixed_bridge_attempt_count": 0,
        "selected_blocks_count": 0,
    }
    passed, reason = validate_profile_guard(a_row, "constructive_lns_search")
    assert passed, reason

    # A 组非法: solver_path 为 block_first
    bad_row = dict(a_row)
    bad_row["solver_path"] = "block_first"
    passed, reason = validate_profile_guard(bad_row, "constructive_lns_search")
    assert not passed
    assert "block_first" in reason.lower()


def test_profile_guard_block_first():
    """
    smoke: block_first_guarded_search 的 profile guard 校验。
    """
    from aps_cp_sat.model.experiment_utils import validate_profile_guard

    # B 组合法行
    b_row = {
        "solver_path": "block_first",
        "main_path": "block_first",
        "allow_real_bridge_edge_in_constructive": True,
        "allow_virtual_bridge_edge_in_constructive": True,
        "mixed_bridge_attempt_count": 5,
        "selected_blocks_count": 8,
    }
    passed, reason = validate_profile_guard(b_row, "block_first_guarded_search")
    assert passed, reason

    # B 组非法: solver_path 不对
    bad_row = dict(b_row)
    bad_row["solver_path"] = "constructive_lns"
    passed, reason = validate_profile_guard(bad_row, "block_first_guarded_search")
    assert not passed
    assert "block_first" in reason.lower()

    # B 组非法: 缺少 block-first 字段
    bad_row2 = dict(b_row)
    bad_row2["mixed_bridge_attempt_count"] = -1
    passed, reason = validate_profile_guard(bad_row2, "block_first_guarded_search")
    assert not passed
    assert "mixed_bridge_attempt_count" in reason


# =============================================================================
# Smoke Test 4: 结果采集
# =============================================================================


def test_collect_experiment_row():
    """
    smoke: collect_experiment_row 正确处理 engine_meta。
    """
    from aps_cp_sat.model.experiment_utils import collect_experiment_row

    class FakeEngineMeta:
        def __init__(self):
            self.solver_path = "block_first"
            self.main_path = "block_first"
            self.dropped_count = 3
            self.scheduled_real_orders = 197
            self.tail_underfilled_count = 1
            self.campaign_count = 8
            self.selected_blocks_count = 8
            self.mixed_bridge_attempt_count = 5
            self.mixed_bridge_success_count = 3
            self.block_alns_rounds_accepted = 4
            self.generated_blocks_total = 50
            self.avg_block_quality_score = 0.85

    class FakeResult:
        def __init__(self):
            self.engine_meta = FakeEngineMeta()

    result = FakeResult()
    row = collect_experiment_row(
        result=result,
        profile_name="block_first_guarded_search",
        seed=2027,
        output_path="/tmp/test.xlsx",
        elapsed_seconds=45.0,
    )

    assert row["profile_name"] == "block_first_guarded_search"
    assert row["seed"] == 2027
    assert row["run_failed"] is False
    assert row["dropped_count"] == 3
    assert row["scheduled_real_orders"] == 197
    assert row["selected_blocks_count"] == 8
    assert row["mixed_bridge_success_count"] == 3
    assert row["block_alns_rounds_accepted"] == 4
    assert row["generated_blocks_total"] == 50
    assert row["avg_block_quality_score"] == 0.85
    assert row["total_runtime_seconds"] == 45.0


def test_collect_experiment_row_failure():
    """
    smoke: 运行失败时正确记录 error_message。
    """
    from aps_cp_sat.model.experiment_utils import collect_experiment_row

    row = collect_experiment_row(
        result=None,
        profile_name="block_first_guarded_search",
        seed=2027,
        output_path="/tmp/test.xlsx",
        elapsed_seconds=0.0,
        run_error="RuntimeError: solver crashed",
    )

    assert row["run_failed"] is True
    assert "RuntimeError" in row["error_message"]
    # 失败时其他数值应为默认值
    assert row["dropped_count"] == 0


def test_collect_experiment_row_from_dict():
    """
    smoke: collect_experiment_row_from_dict 直接从 dict 采集。
    """
    from aps_cp_sat.model.experiment_utils import collect_experiment_row_from_dict

    meta_dict = {
        "solver_path": "constructive_lns",
        "main_path": "constructive_lns",
        "dropped_count": 5,
        "scheduled_real_orders": 195,
        "tail_underfilled_count": 2,
        "campaign_count": 9,
        "allow_real_bridge_edge_in_constructive": True,
        "allow_virtual_bridge_edge_in_constructive": False,
    }

    row = collect_experiment_row_from_dict(
        meta_dict=meta_dict,
        profile_name="constructive_lns_search",
        seed=2028,
        output_path="/tmp/test2.xlsx",
        elapsed_seconds=32.0,
    )

    assert row["profile_name"] == "constructive_lns_search"
    assert row["seed"] == 2028
    assert row["dropped_count"] == 5
    assert row["scheduled_real_orders"] == 195
    assert row["allow_real_bridge_edge_in_constructive"] is True
    assert row["allow_virtual_bridge_edge_in_constructive"] is False


# =============================================================================
# Smoke Test 5: Sweep 配置
# =============================================================================


def test_sweep_configs_exist():
    """
    smoke: 所有预设 sweep 配置都存在且格式正确。
    """
    from aps_cp_sat.model.experiment_utils import (
        BLOCK_ALNS_CONFIGS,
        BLOCK_GENERATOR_SUPPLY_CONFIGS,
        DIRECTIONAL_CLUSTER_CONFIGS,
    )

    # Stage A
    assert "A1_baseline" in BLOCK_GENERATOR_SUPPLY_CONFIGS
    assert "A2_relaxed" in BLOCK_GENERATOR_SUPPLY_CONFIGS
    assert "A3_wider" in BLOCK_GENERATOR_SUPPLY_CONFIGS
    for cfg in BLOCK_GENERATOR_SUPPLY_CONFIGS.values():
        assert "block_generator_max_blocks_per_line" in cfg
        assert "block_generator_max_blocks_total" in cfg

    # Stage B
    assert "B1_current" in DIRECTIONAL_CLUSTER_CONFIGS
    assert "B2_heavy_tons_and_stability" in DIRECTIONAL_CLUSTER_CONFIGS
    assert "B3_heavy_bridge_potential" in DIRECTIONAL_CLUSTER_CONFIGS
    for cfg in DIRECTIONAL_CLUSTER_CONFIGS.values():
        assert "directional_cluster_group_weight" in cfg
        assert "directional_cluster_tons_fill_weight" in cfg

    # Stage C
    assert "C1_current" in BLOCK_ALNS_CONFIGS
    assert "C2_stronger" in BLOCK_ALNS_CONFIGS
    assert "C3_flexible" in BLOCK_ALNS_CONFIGS
    for cfg in BLOCK_ALNS_CONFIGS.values():
        assert "block_alns_rounds" in cfg
        assert "mixed_bridge_max_attempts_per_block" in cfg


def test_extract_sweep_params():
    """
    smoke: extract_sweep_params 正确从配置对象提取参数。
    """
    from aps_cp_sat.config.parameters import build_profile_config
    from aps_cp_sat.model.experiment_utils import extract_sweep_params

    cfg = build_profile_config("block_first_guarded_search")
    params = extract_sweep_params(cfg)

    assert params["block_generator_max_blocks_per_line"] == 30
    assert params["block_generator_max_blocks_total"] == 80
    assert params["directional_cluster_group_weight"] == 1.2
    assert params["block_alns_rounds"] == 6
    assert params["mixed_bridge_max_attempts_per_block"] == 10


# =============================================================================
# Smoke Test 6: CSV I/O
# =============================================================================


def test_write_and_read_csv(tmp_path):
    """
    smoke: 写入并读回 CSV。
    """
    from aps_cp_sat.model.experiment_utils import (
        read_results_csv,
        write_results_csv,
    )

    rows = [
        {
            "profile_name": "constructive_lns_search",
            "seed": 2027,
            "dropped_count": 5,
            "scheduled_real_orders": 195,
            "tail_underfilled_count": 1,
            "total_runtime_seconds": 30.0,
            "run_failed": False,
            "error_message": "",
        },
        {
            "profile_name": "block_first_guarded_search",
            "seed": 2027,
            "dropped_count": 3,
            "scheduled_real_orders": 197,
            "tail_underfilled_count": 1,
            "total_runtime_seconds": 45.0,
            "run_failed": False,
            "error_message": "",
        },
    ]

    csv_path = tmp_path / "test_results.csv"
    write_results_csv(rows, csv_path)
    assert csv_path.exists()

    df = read_results_csv(csv_path)
    assert len(df) == 2
    assert "profile_name" in df.columns
    assert "seed" in df.columns
    assert "dropped_count" in df.columns


def test_write_empty_csv(tmp_path):
    """
    smoke: 写入空结果也生成带表头的 CSV。
    """
    from aps_cp_sat.model.experiment_utils import read_results_csv, write_results_csv

    csv_path = tmp_path / "empty.csv"
    write_results_csv([], csv_path)
    assert csv_path.exists()

    df = read_results_csv(csv_path)
    assert len(df) == 0
