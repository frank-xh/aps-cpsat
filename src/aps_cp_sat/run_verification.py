from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from time import perf_counter

from aps_cp_sat.cold_rolling_pipeline import ColdRollingPipeline
from aps_cp_sat.config.parameters import build_profile_config
from aps_cp_sat.domain.models import ColdRollingRequest, ColdRollingResult
from aps_cp_sat.routes.route_factory import create_route_runner

DEFAULT_PROFILE = "constructive_lns_virtual_guarded_frontload"
DEFAULT_STRATEGY = "constructive_lns"
DEFAULT_OUTPUT_NAME = "verification.xlsx"


def _infer_strategy(profile_name: str) -> str:
    profile = str(profile_name or "")
    if profile == DEFAULT_PROFILE:
        return DEFAULT_STRATEGY
    raise ValueError(
        "[APS][verification][ONLY_SINGLE_PROFILE_ALLOWED] "
        f"expected strategy={DEFAULT_STRATEGY}, expected profile={DEFAULT_PROFILE}, got profile={profile!r}"
    )


def _candidate_order_files() -> list[Path]:
    candidates: list[Path] = []
    cwd = Path.cwd()
    roots = [cwd, cwd / 'data', Path(__file__).resolve().parent.parent, Path(__file__).resolve().parent.parent / 'data']
    # Common Windows locations. Using forward slashes avoids raw-string trailing-backslash issues.
    roots.extend([Path('D:/Desktop'), Path('D:/桌面'), Path('C:/Users/Public/Desktop')])
    home = Path.home()
    roots.extend([home, home / 'Desktop', home / '桌面', home / 'OneDrive', home / 'OneDrive' / 'Desktop'])

    patterns = [
        'data_orders.xlsx',
        '*订单*.xlsx',
        '*冷轧*订单*.xlsx',
        '*原始数据*.xlsx',
        '*.xlsx',
    ]
    seen: set[str] = set()
    for root in roots:
        try:
            if not root.exists():
                continue
            for pat in patterns:
                for path in root.glob(pat):
                    if not path.is_file():
                        continue
                    key = str(path.resolve()).lower()
                    if key not in seen:
                        seen.add(key)
                        candidates.append(path)
        except Exception:
            continue
    return candidates


def _resolve_orders_path(user_value: str | Path | None) -> Path:
    if user_value:
        path = Path(user_value)
        if path.exists():
            return path
    for path in _candidate_order_files():
        return path
    raise FileNotFoundError(
        '[APS][verification] orders file not found. '        'Please pass --orders explicitly, e.g. --orders "D:/Desktop/冷轧连退钢带-202511订单明细数据-原始数据.xlsx"'
    )


def _resolve_output_path(user_value: str | Path | None) -> Path:
    if user_value:
        return Path(user_value)
    preferred = [Path('D:/Desktop/SAT排程结果'), Path('D:/桌面/SAT排程结果'), Path.cwd() / 'out']
    for base in preferred:
        try:
            base.mkdir(parents=True, exist_ok=True)
            return base / DEFAULT_OUTPUT_NAME
        except Exception:
            continue
    return Path(DEFAULT_OUTPUT_NAME)


def _assert_route_result(
    result: ColdRollingResult,
    expected_profile: str,
    expected_solver_path: str,
    expected_main_path: str,
) -> None:
    meta = result.engine_meta or {}
    actual_profile = str(meta.get('profile_name', 'UNKNOWN'))
    actual_solver = str(meta.get('solver_path', 'UNKNOWN'))
    actual_main = str(meta.get('main_path', 'UNKNOWN'))
    print('--- Verification Assertions ---')
    print(f'Expected profile_name : {expected_profile} | Actual: {actual_profile}')
    print(f'Expected solver_path  : {expected_solver_path} | Actual: {actual_solver}')
    print(f'Expected main_path    : {expected_main_path} | Actual: {actual_main}')
    if actual_profile != expected_profile:
        raise RuntimeError(f'[APS][verification] profile_name mismatch. Expected {expected_profile}, got {actual_profile}')
    if actual_solver != expected_solver_path:
        raise RuntimeError(f'[APS][verification] solver_path mismatch. Expected {expected_solver_path}, got {actual_solver}')
    if actual_main != expected_main_path:
        raise RuntimeError(f'[APS][verification] main_path mismatch. Expected {expected_main_path}, got {actual_main}')
    print('[APS][verification] route assertions passed')


def run_verification(
    *,
    profile: str = DEFAULT_PROFILE,
    strategy: str | None = None,
    orders: Path | str | None = None,
    steel_info: Path | str | None = None,
    output: Path | str | None = None,
    seed: int = 42,
    time_limit: float = 60.0,
    virtual_bridge_mode: str | None = None,
    prebuilt_virtual_inventory_enabled: bool | None = None,
    prebuilt_virtual_count_per_spec: int | None = None,
) -> ColdRollingResult:
    strategy_name = str(strategy or '').strip() or _infer_strategy(profile)
    if strategy_name != DEFAULT_STRATEGY or str(profile) != DEFAULT_PROFILE:
        raise ValueError(
            f'[APS][verification][ONLY_SINGLE_ROUTE_ALLOWED] '
            f'expected strategy={DEFAULT_STRATEGY}, expected profile={DEFAULT_PROFILE}, '
            f'got strategy={strategy_name!r}, profile={profile!r}'
        )
    create_route_runner(strategy_name, profile)

    orders_path = _resolve_orders_path(orders)
    steel_info_path = Path(steel_info) if steel_info else None
    output_path = _resolve_output_path(output)

    print('============================================================')
    print('[APS][verification] Starting route verification run')
    print('============================================================')
    print(f'[APS][verification] profile={profile}')
    print(f'[APS][verification] strategy={strategy_name}')
    print(f'[APS][verification] orders_path={orders_path}')
    print(f'[APS][verification] steel_info_path={steel_info_path}')
    print(f'[APS][verification] output_path={output_path}')
    print(f'[APS][verification] time_limit_seconds={time_limit}')
    print(f'[APS][verification] random_seed={seed}')

    if steel_info_path is not None and not steel_info_path.exists():
        raise FileNotFoundError(f'[APS][verification] steel_info file not found: {steel_info_path}')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = build_profile_config(profile)
    model_changes = {
        'profile_name': profile,
        'main_solver_strategy': strategy_name,
        'time_limit_seconds': float(time_limit),
    }
    if virtual_bridge_mode:
        model_changes['virtual_bridge_mode'] = str(virtual_bridge_mode)
    if prebuilt_virtual_inventory_enabled is not None:
        model_changes['prebuilt_virtual_inventory_enabled'] = bool(prebuilt_virtual_inventory_enabled)
    if prebuilt_virtual_count_per_spec is not None:
        model_changes['prebuilt_virtual_count_per_spec'] = int(prebuilt_virtual_count_per_spec)
    if hasattr(cfg.model, 'random_seed'):
        model_changes['random_seed'] = int(seed)
    cfg = replace(cfg, model=replace(cfg.model, **model_changes))

    req = ColdRollingRequest(
        orders_path=orders_path,
        steel_info_path=steel_info_path,
        output_path=output_path,
        config=cfg,
    )

    t0 = perf_counter()
    result = ColdRollingPipeline().run(req)
    total_time = perf_counter() - t0

    _assert_route_result(result, expected_profile=profile, expected_solver_path=strategy_name, expected_main_path=strategy_name)

    meta = result.engine_meta or {}
    print('\n--- Result Summary ---')
    print(f"profile_name                  : {meta.get('profile_name', 'UNKNOWN')}")
    print(f"route_name                    : {meta.get('route_name', 'UNKNOWN')}")
    print(f"solver_path                   : {meta.get('solver_path', 'UNKNOWN')}")
    print(f"main_path                     : {meta.get('main_path', 'UNKNOWN')}")
    print(f"final_realized_order_count    : {meta.get('final_realized_order_count', 0)}")
    print(f"final_realized_tons           : {meta.get('final_realized_tons', 0.0)}")
    print(f"effective_dropped_order_count : {meta.get('effective_dropped_order_count', 0)}")
    print(f"campaign_cnt                  : {meta.get('campaign_cnt', 0)}")
    print(f"underfilled_slot_count        : {meta.get('underfilled_slot_count', 0)}")
    print(f"final_hard_violation_count_total : {meta.get('final_hard_violation_count_total', 0)}")
    print(f"output_file                   : {output_path}")
    print(f"total_time_seconds            : {total_time:.2f}s")
    print('============================================================')
    print('[APS][verification] Run completed')
    print('============================================================')
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run APS verification for the single constructive route.')
    parser.add_argument('--profile', default=DEFAULT_PROFILE)
    parser.add_argument('--strategy', default=DEFAULT_STRATEGY)
    parser.add_argument('--orders', default='')
    parser.add_argument('--steel_info', default='')
    parser.add_argument('--output', default='')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--time_limit', type=float, default=60.0)
    parser.add_argument('--virtual_bridge_mode', default='', choices=['', 'template_bridge', 'prebuilt_virtual_inventory'])
    parser.add_argument('--prebuilt_virtual_inventory_enabled', action='store_true')
    parser.add_argument('--prebuilt_virtual_count_per_spec', type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_verification(
        profile=args.profile,
        strategy=args.strategy or None,
        orders=Path(args.orders) if args.orders else None,
        steel_info=Path(args.steel_info) if args.steel_info else None,
        output=Path(args.output) if args.output else None,
        seed=int(args.seed),
        time_limit=float(args.time_limit),
        virtual_bridge_mode=args.virtual_bridge_mode or None,
        prebuilt_virtual_inventory_enabled=bool(args.prebuilt_virtual_inventory_enabled) if args.prebuilt_virtual_inventory_enabled else None,
        prebuilt_virtual_count_per_spec=int(args.prebuilt_virtual_count_per_spec) if int(args.prebuilt_virtual_count_per_spec or 0) > 0 else None,
    )


if __name__ == '__main__':
    main()
