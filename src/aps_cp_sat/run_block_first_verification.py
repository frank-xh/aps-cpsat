from __future__ import annotations


def main() -> None:
    raise RuntimeError(
        "[APS][BLOCK_FIRST_ROUTE_REMOVED] "
        "Only constructive_lns_virtual_guarded_frontload is supported. "
        "Use `python -m aps_cp_sat.run_verification --strategy constructive_lns "
        "--profile constructive_lns_virtual_guarded_frontload`."
    )


if __name__ == "__main__":
    main()
