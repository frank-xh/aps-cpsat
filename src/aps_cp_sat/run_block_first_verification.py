from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

from aps_cp_sat.cold_rolling_pipeline import ColdRollingPipeline
from aps_cp_sat.config.parameters import build_profile_config
from aps_cp_sat.domain.models import ColdRollingRequest


def main():
    parser = argparse.ArgumentParser(description="Dedicated verification runner for block-first architecture")
    parser.add_argument("--orders", type=str, required=True, help="Path to orders file")
    parser.add_argument("--steel_info", type=str, required=True, help="Path to steel info file")
    parser.add_argument("--output", type=str, required=True, help="Path for output schedule file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--time_limit", type=float, default=60.0, help="Time limit in seconds")
    
    args = parser.parse_args()
    
    print(f"============================================================")
    print(f"[APS][block_first_verify] Starting dedicated verification run")
    print(f"============================================================")
    
    # 1) 显式使用 build_profile_config("block_first_guarded_search")
    profile = "block_first_guarded_search"
    cfg = build_profile_config(profile)
    
    # Set explicitly requested runtime overrides
    cfg.model.time_limit_seconds = args.time_limit
    cfg.model.random_seed = args.seed
    
    req = ColdRollingRequest(
        orders_path=Path(args.orders),
        steel_info_path=Path(args.steel_info),
        output_path=Path(args.output),
        config=cfg,
    )
    
    t0 = perf_counter()
    pipeline = ColdRollingPipeline()
    result = pipeline.run(req)
    total_time = perf_counter() - t0
    
    # 3) 运行完成后，立刻对 result.engine_meta 做硬断言
    meta = result.engine_meta or {}
    
    actual_profile = meta.get("profile_name", "UNKNOWN")
    actual_solver = meta.get("solver_path", "UNKNOWN")
    actual_main = meta.get("main_path", "UNKNOWN")
    
    print(f"--- Verification Assertions ---")
    print(f"Expected profile_name : {profile} | Actual: {actual_profile}")
    print(f"Expected solver_path  : block_first | Actual: {actual_solver}")
    print(f"Expected main_path    : block_first | Actual: {actual_main}")
    
    if actual_profile != profile:
        raise RuntimeError(f"[APS][block_first_verify] profile_name mismatch. Expected {profile}, got {actual_profile}")
    if actual_solver != "block_first":
        raise RuntimeError(f"[APS][block_first_verify] solver_path != block_first. Got {actual_solver}")
    if actual_main != "block_first":
        raise RuntimeError(f"[APS][block_first_verify] main_path != block_first. Got {actual_main}")
        
    print(f"✅ Architecture Path Assertions Passed")
    
    # 5) 脚本输出应至少打印关键字段
    print(f"\n--- Result Summary ---")
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
    print(f"total_time_seconds            : {total_time:.2f}s")
    
    print(f"============================================================")
    print(f"[APS][block_first_verify] Run completed successfully.")
    print(f"============================================================")


if __name__ == "__main__":
    main()
