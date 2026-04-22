from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from time import perf_counter

from aps_cp_sat.cold_rolling_pipeline import ColdRollingPipeline
from aps_cp_sat.config.parameters import build_profile_config
from aps_cp_sat.domain.models import ColdRollingRequest


# =========================
# 写死的输入输出路径
# =========================
ORDERS_PATH = Path(r"D:\桌面\冷轧连退钢带-202511订单明细数据-原始数据.xlsx")
STEEL_INFO_PATH = None  # 没有单独钢种文件时保持为 None
OUTPUT_DIR = Path(r"D:\桌面\SAT排程结果")
OUTPUT_FILE_NAME = "block_first_verify.xlsx"

# =========================
# 写死的运行参数
# =========================
PROFILE_NAME = "block_first_guarded_search"
TIME_LIMIT_SECONDS = 60.0
RANDOM_SEED = 42


def main():
    print("============================================================")
    print("[APS][block_first_verify] Starting dedicated verification run")
    print("============================================================")
    print(f"[APS][block_first_verify] orders_path = {ORDERS_PATH}")
    print(f"[APS][block_first_verify] steel_info_path = {STEEL_INFO_PATH}")
    print(f"[APS][block_first_verify] output_dir = {OUTPUT_DIR}")
    print(f"[APS][block_first_verify] profile = {PROFILE_NAME}")
    print(f"[APS][block_first_verify] time_limit_seconds = {TIME_LIMIT_SECONDS}")
    print(f"[APS][block_first_verify] random_seed = {RANDOM_SEED}")

    # 1) 输入文件存在性检查
    if not ORDERS_PATH.exists():
        raise FileNotFoundError(
            f"[APS][block_first_verify] orders file not found: {ORDERS_PATH}"
        )

    if STEEL_INFO_PATH is None:
        print(
            "[APS][block_first_verify] steel_info not provided; "
            "using single-file order input mode"
        )
    elif not Path(STEEL_INFO_PATH).exists():
        raise FileNotFoundError(
            f"[APS][block_first_verify] steel_info file not found: {STEEL_INFO_PATH}"
        )

    # 2) 创建输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / OUTPUT_FILE_NAME

    # 3) 构建 block-first 配置
    cfg = build_profile_config(PROFILE_NAME)

    model_changes = {
        "time_limit_seconds": TIME_LIMIT_SECONDS,
    }

    # 如果当前 ModelConfig 已经支持 random_seed，就写进去；否则跳过
    if hasattr(cfg.model, "random_seed"):
        model_changes["random_seed"] = RANDOM_SEED
    else:
        print(
            "[APS][block_first_verify][WARN] "
            "ModelConfig has no 'random_seed' field; seed override skipped."
        )

    new_model = replace(cfg.model, **model_changes)
    cfg = replace(cfg, model=new_model)

    # 4) 构造请求
    req = ColdRollingRequest(
        orders_path=ORDERS_PATH,
        steel_info_path=STEEL_INFO_PATH,
        output_path=output_path,
        config=cfg,
    )

    # 5) 运行 pipeline
    t0 = perf_counter()
    pipeline = ColdRollingPipeline()
    result = pipeline.run(req)
    total_time = perf_counter() - t0

    # 6) block-first 路径断言
    meta = result.engine_meta or {}

    actual_profile = meta.get("profile_name", "UNKNOWN")
    actual_solver = meta.get("solver_path", "UNKNOWN")
    actual_main = meta.get("main_path", "UNKNOWN")

    print("--- Verification Assertions ---")
    print(f"Expected profile_name : {PROFILE_NAME} | Actual: {actual_profile}")
    print(f"Expected solver_path  : block_first | Actual: {actual_solver}")
    print(f"Expected main_path    : block_first | Actual: {actual_main}")

    if actual_profile != PROFILE_NAME:
        raise RuntimeError(
            f"[APS][block_first_verify] profile_name mismatch. "
            f"Expected {PROFILE_NAME}, got {actual_profile}"
        )
    if actual_solver != "block_first":
        raise RuntimeError(
            f"[APS][block_first_verify] solver_path != block_first. Got {actual_solver}"
        )
    if actual_main != "block_first":
        raise RuntimeError(
            f"[APS][block_first_verify] main_path != block_first. Got {actual_main}"
        )

    print("✅ Architecture Path Assertions Passed")

    # 7) 打印结果摘要
    print("\n--- Result Summary ---")
    print(f"profile_name                  : {actual_profile}")
    print(f"solver_path                   : {actual_solver}")
    print(f"main_path                     : {actual_main}")
    print(f"final_realized_order_count    : {meta.get('final_realized_order_count', 0)}")
    print(f"final_realized_tons           : {meta.get('final_realized_tons', 0.0)}")
    print(f"effective_dropped_order_count : {meta.get('effective_dropped_order_count', 0)}")
    print(f"assembled_slot_count          : {meta.get('assembled_slot_count', 0)}")
    print(f"underfilled_slot_count        : {meta.get('underfilled_slot_count', 0)}")
    print(f"unassembled_block_count       : {meta.get('unassembled_block_count', 0)}")
    print(f"block_alns_rounds_accepted    : {meta.get('block_alns_rounds_accepted', 0)}")
    print(f"exported_from_alns_final      : {meta.get('exported_from_alns_final', False)}")
    print(f"output_file                   : {output_path}")
    print(f"total_time_seconds            : {total_time:.2f}s")

    print("============================================================")
    print("[APS][block_first_verify] Run completed successfully.")
    print("============================================================")


if __name__ == "__main__":
    main()