from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter
from typing import Dict, Any

from aps_cp_sat.cold_rolling_pipeline import ColdRollingPipeline
from aps_cp_sat.config.parameters import build_profile_config
from aps_cp_sat.domain.models import ColdRollingRequest


def main():
    parser = argparse.ArgumentParser(description="A/B Comparison Runner: constructive vs block-first")
    parser.add_argument("--orders", type=str, required=True, help="Path to orders file")
    parser.add_argument("--steel_info", type=str, required=True, help="Path to steel info file")
    parser.add_argument("--output_a", type=str, required=True, help="Path for A output schedule file")
    parser.add_argument("--output_b", type=str, required=True, help="Path for B output schedule file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--time_limit", type=float, default=60.0, help="Time limit in seconds")
    
    args = parser.parse_args()
    
    print(f"============================================================")
    print(f"[APS][ab_compare] Starting A/B Comparison")
    print(f"============================================================")
    
    pipeline = ColdRollingPipeline()
    
    def run_profile(profile: str, out_path: str) -> Dict[str, Any]:
        cfg = build_profile_config(profile)
        cfg.model.time_limit_seconds = args.time_limit
        cfg.model.random_seed = args.seed
        
        req = ColdRollingRequest(
            orders_path=Path(args.orders),
            steel_info_path=Path(args.steel_info),
            output_path=Path(out_path),
            config=cfg,
        )
        
        t0 = perf_counter()
        result = pipeline.run(req)
        total_time = perf_counter() - t0
        
        meta = result.engine_meta or {}
        actual_profile = meta.get("profile_name", "UNKNOWN")
        if actual_profile != profile:
            raise RuntimeError(f"[APS][ab_compare] Profile mismatch! Expected {profile}, got {actual_profile}")
            
        return {
            "profile_name": actual_profile,
            "solver_path": meta.get("solver_path", "UNKNOWN"),
            "main_path": meta.get("main_path", "UNKNOWN"),
            "final_realized_order_count": meta.get("final_realized_order_count", meta.get("assigned_count", 0)),
            "final_realized_tons": meta.get("final_realized_tons", meta.get("assigned_tons", 0.0)),
            "effective_dropped_order_count": meta.get("effective_dropped_order_count", meta.get("dropped_count", 0)),
            "assembled_slot_count": meta.get("assembled_slot_count", meta.get("campaign_count", 0)),
            "underfilled_slot_count": meta.get("underfilled_slot_count", meta.get("tail_underfilled_count", 0)),
            "unassembled_block_count": meta.get("unassembled_block_count", 0),
            "block_alns_rounds_accepted": meta.get("block_alns_rounds_accepted", 0),
            "failed_block_boundary_count": meta.get("failed_block_boundary_count", 0),
            "exported_from_alns_final": meta.get("exported_from_alns_final", False),
            "total_time_seconds": round(total_time, 2),
        }

    # 1) Run A: constructive_lns_search
    print(f"\n---> Starting Run A: constructive_lns_search")
    res_a = run_profile("constructive_lns_search", args.output_a)
    
    if res_a["solver_path"] != "constructive_lns":
        raise RuntimeError(f"Run A failed validation: solver_path is {res_a['solver_path']}, expected constructive_lns")

    # 2) Run B: block_first_guarded_search
    print(f"\n---> Starting Run B: block_first_guarded_search")
    res_b = run_profile("block_first_guarded_search", args.output_b)
    
    if res_b["solver_path"] != "block_first":
        raise RuntimeError(f"Run B failed validation: solver_path is {res_b['solver_path']}, expected block_first")
        
    print(f"\n============================================================")
    print(f"[APS][ab_compare] Summary Report")
    print(f"============================================================")
    
    keys = [
        "profile_name",
        "solver_path",
        "main_path",
        "final_realized_order_count",
        "final_realized_tons",
        "effective_dropped_order_count",
        "assembled_slot_count",
        "underfilled_slot_count",
        "unassembled_block_count",
        "block_alns_rounds_accepted",
        "failed_block_boundary_count",
        "exported_from_alns_final",
        "total_time_seconds",
    ]
    
    print(f"{'Metric':<35} | {'Run A':<30} | {'Run B':<30}")
    print("-" * 100)
    for k in keys:
        val_a = str(res_a.get(k, "N/A"))
        val_b = str(res_b.get(k, "N/A"))
        print(f"{k:<35} | {val_a:<30} | {val_b:<30}")


if __name__ == "__main__":
    main()
