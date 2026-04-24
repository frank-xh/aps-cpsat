from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)


@dataclass
class CampaignBuildState:
    """Incremental campaign build state used by constructive/cutter local checks."""

    current_tons: float = 0.0
    reverse_width_count: int = 0
    reverse_width_total_mm: float = 0.0
    virtual_chain_length: int = 0
    last_order_id: str = ""
    line: str = ""
    order_count: int = 0
    last_width: float = 0.0
    episode_state: dict[str, Any] = field(default_factory=dict)


def is_reverse_width_pair(prev_order: dict[str, Any] | None, next_order: dict[str, Any] | None) -> bool:
    """Treat width increase as reverse-width pair."""
    prev = dict(prev_order or {})
    nxt = dict(next_order or {})
    if not prev or not nxt:
        return False
    prev_w = prev.get("width")
    next_w = nxt.get("width")
    if prev_w is None or next_w is None:
        return False
    return _as_float(next_w) > _as_float(prev_w)


def compute_reverse_width_episode_delta(
    prev_order: dict[str, Any] | None,
    next_order: dict[str, Any] | None,
    current_state: CampaignBuildState,
) -> int:
    """V1 simplified: each reverse-width pair counts as one episode."""
    _ = current_state  # Reserved for future episode-level logic.
    return 1 if is_reverse_width_pair(prev_order, next_order) else 0


def can_accept_reverse_width_pair(
    prev_order: dict[str, Any] | None,
    next_order: dict[str, Any] | None,
    current_state: CampaignBuildState,
    max_reverse_count: int = 2,
) -> tuple[bool, str, CampaignBuildState]:
    """Apply reverse-width budget and return updated state."""
    prev = dict(prev_order or {})
    nxt = dict(next_order or {})

    delta = compute_reverse_width_episode_delta(prev, nxt, current_state)
    new_count = int(current_state.reverse_width_count) + int(delta)
    if new_count > int(max_reverse_count):
        return False, "REVERSE_WIDTH_BUDGET_EXCEEDED", current_state

    prev_w = _as_float(prev.get("width"), current_state.last_width)
    next_w = _as_float(nxt.get("width"), prev_w)
    rise = max(0.0, next_w - prev_w)
    next_tons = _as_float(nxt.get("tons"), 0.0)
    is_virtual = bool(nxt.get("is_virtual", False)) or str(nxt.get("virtual_origin", "")) == "prebuilt_inventory"

    new_state = CampaignBuildState(
        current_tons=float(current_state.current_tons) + float(next_tons),
        reverse_width_count=new_count,
        reverse_width_total_mm=float(current_state.reverse_width_total_mm) + float(rise),
        virtual_chain_length=(int(current_state.virtual_chain_length) + 1) if is_virtual else 0,
        last_order_id=str(nxt.get("order_id", current_state.last_order_id)),
        line=str(current_state.line),
        order_count=int(current_state.order_count) + 1,
        last_width=next_w,
        episode_state=dict(current_state.episode_state or {}),
    )
    return True, "OK", new_state


def evaluate_sequence_reverse_width_budget(
    order_ids: list[str],
    order_record_by_oid: dict[str, dict[str, Any]],
    line: str = "",
    max_reverse_count: int = 2,
) -> tuple[bool, CampaignBuildState]:
    """Check entire sequence against reverse-width budget."""
    if not order_ids:
        return True, CampaignBuildState(line=line)
    first = dict(order_record_by_oid.get(str(order_ids[0]), {}) or {})
    state = CampaignBuildState(
        current_tons=_as_float(first.get("tons"), 0.0),
        reverse_width_count=0,
        reverse_width_total_mm=0.0,
        virtual_chain_length=1
        if (bool(first.get("is_virtual", False)) or str(first.get("virtual_origin", "")) == "prebuilt_inventory")
        else 0,
        last_order_id=str(order_ids[0]),
        line=str(line),
        order_count=1,
        last_width=_as_float(first.get("width"), 0.0),
        episode_state={},
    )
    for i in range(len(order_ids) - 1):
        prev = dict(order_record_by_oid.get(str(order_ids[i]), {}) or {})
        nxt = dict(order_record_by_oid.get(str(order_ids[i + 1]), {}) or {})
        ok, _, state = can_accept_reverse_width_pair(prev, nxt, state, max_reverse_count=max_reverse_count)
        if not ok:
            return False, state
    return True, state

