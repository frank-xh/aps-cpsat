import argparse
import os
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

# Add src to path to ensure we are running the latest code
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from aps_cp_sat.cold_rolling_pipeline import ColdRollingPipeline
from aps_cp_sat.config import PlannerConfig, build_profile_config
from aps_cp_sat.domain.models import ColdRollingRequest


def build_example_config(profile: str = "constructive_lns_search", strict: bool = False) -> PlannerConfig:
    base = build_profile_config(
        profile,
        validation_mode=bool(strict),
        production_compatibility_mode=False,
    )
    if strict:
        model = replace(
            base.model,
            allow_fallback=False,
            allow_legacy_fallback=False,
        )
    else:
        model = replace(
            base.model,
            allow_legacy_fallback=False,
        )
    return PlannerConfig(rule=base.rule, score=base.score, model=model)


def main() -> None:
    t0 = time.perf_counter()
    parser = argparse.ArgumentParser(description="Run cold rolling schedule example")
    parser.add_argument("--strict", action="store_true", help="Disable semantic fallback and fail fast")
    parser.add_argument("--persist-mysql", action="store_true", help="Import analysis Excel into local MySQL (optional)")
    parser.add_argument(
        "--profile",
        choices=[
            "constructive_lns_search",
            "constructive_lns_direct_only_baseline",
            "constructive_lns_real_bridge_frontload",
        ],
        default="constructive_lns_search",
        help="Solver tuning profile",
    )
    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    orders = os.path.join(project_root, "data_orders.xlsx")
    steel_info = os.path.join(project_root, "data_steel_info.xlsx")

    output_dir = "D:\\Desktop\\SAT排程结果"
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = os.path.join(output_dir, f"SAT排程结果_{ts}.xlsx")

    cfg = build_example_config(profile=args.profile, strict=bool(args.strict))

    print("[APS] 冷轧排程任务开始")
    print(f"[APS] 订单文件: {orders}")
    print(f"[APS] 钢种文件: {steel_info}")
    print(f"[APS] 输出文件: {output}")
    print(f"[APS] Profile: {cfg.model.profile_name}")
    print(
        f"[APS] 参数: max_orders={cfg.max_orders}, rounds={cfg.rounds}, "
        f"time_limit_seconds={cfg.time_limit_seconds}, max_virtual_chain={cfg.max_virtual_chain}"
    )

    pipeline = ColdRollingPipeline()
    result = pipeline.run(
        ColdRollingRequest(
            orders_path=Path(os.path.abspath(orders)),
            steel_info_path=Path(os.path.abspath(steel_info)),
            output_path=Path(os.path.abspath(output)),
            config=cfg,
        )
    )
    seq_df, it_df = result.schedule_df, result.rounds_df

    print("完成。")
    print(f"[APS] 总耗时: {time.perf_counter() - t0:.1f}s")
    print(f"输出文件: {result.output_path}")
    print(f"排程订单数: {len(seq_df)}")
    print("轮次汇总:")
    print(it_df.to_string(index=False))
    print("\n前 10 行排程预览:")
    preview_cols = [c for c in ["global_seq", "order_id", "grade", "steel_group", "width", "thickness", "tons"] if c in seq_df.columns]
    print(seq_df[preview_cols].head(10).to_string(index=False))

    if bool(args.persist_mysql):
        try:
            from aps_cp_sat.persistence.service import persist_run_analysis_from_excel

            run_id = persist_run_analysis_from_excel(result.output_path)
            print(f"[analysis][persist] ok run_id={run_id} xlsx={result.output_path}")
        except Exception as e:
            print(f"[analysis][persist] failed: {e}")


if __name__ == "__main__":
    main()
