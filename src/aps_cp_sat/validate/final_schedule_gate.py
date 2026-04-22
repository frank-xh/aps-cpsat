from __future__ import annotations

from typing import Any, Dict, List


FINAL_GATE_FIELDS = (
    ("width_total_violation_count", "WIDTH"),
    ("thickness_violation_count", "THICKNESS"),
    ("temperature_violation_count", "TEMPERATURE"),
    ("group_transition_violation_count", "GROUP_TRANSITION"),
    ("duplicate_order_violation_count", "DUPLICATE_ORDER"),
    ("campaign_ton_min_violation_count", "CAMPAIGN_TON_MIN"),
    ("campaign_ton_max_violation_count", "CAMPAIGN_TON_MAX"),
    ("campaign_grouping_violation_count", "CAMPAIGN_GROUPING"),
    ("campaign_sequence_violation_count", "CAMPAIGN_SEQUENCE"),
)


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except Exception:
        return 0


def evaluate_final_schedule_gate(
    engine_meta: Dict[str, Any] | None,
    audit_summary: Dict[str, Any] | None,
) -> Dict[str, Any]:
    """Final production gate based on final schedule audit counters.

    This gate is intentionally separate from the legacy validator because the
    legacy counters can pass while the realized exported sequence still
    contains same-campaign hard-rule violations.
    """
    audit = dict(audit_summary or {})
    reasons: List[str] = []
    total = 0
    for field, reason in FINAL_GATE_FIELDS:
        count = _as_int(audit.get(field, 0))
        total += count
        if count > 0:
            reasons.append(f"{reason}:{count}")

    passed = len(reasons) == 0
    return {
        "final_schedule_gate_passed": bool(passed),
        "final_schedule_gate_reason": "OK" if passed else "FINAL_SCHEDULE_HARD_VIOLATIONS",
        "final_schedule_rejection_reasons": reasons,
        "final_hard_violation_count_total": int(total),
    }
