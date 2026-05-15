"""Deterministic builders for extreme rename-chain (storm) tests."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

# Four-hop rename chain from ADVANCED_BENCHMARK_SUITE §2
RENAME_STORM_CHAIN: tuple[tuple[str, str], ...] = (
    ("svc-a", "svc-b"),
    ("svc-b", "svc-c"),
    ("svc-c", "svc-d"),
    ("svc-d", "svc-e"),
)

RENAME_STORM_CHAIN_5: tuple[tuple[str, str], ...] = RENAME_STORM_CHAIN + (
    ("svc-e", "svc-f"),
)


def incident_signal_event(
    *,
    service: str,
    incident_id: str,
    ts: datetime,
    trigger_type: str = "error_rate",
    upstream: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "kind": "incident_signal",
        "service": service,
        "occurred_at": ts,
        "payload": {
            "incident_id": incident_id,
            "trigger_type": trigger_type,
            "upstream": list(upstream or ["auth", "db"]),
        },
    }


def identity_drift_event(
    *,
    from_svc: str,
    to_svc: str,
    ts: datetime,
    anchor_service: str,
) -> dict[str, Any]:
    return {
        "kind": "identity.drift",
        "service": anchor_service,
        "occurred_at": ts,
        "payload": {"from_": from_svc, "to": to_svc},
    }


def drift_chain_events(
    base_ts: datetime,
    *,
    step_seconds: int = 60,
    chain: tuple[tuple[str, str], ...] = RENAME_STORM_CHAIN,
) -> list[dict[str, Any]]:
    """Ordered ``identity.drift`` events for a rename chain."""
    events: list[dict[str, Any]] = []
    t = base_ts
    for old, new in chain:
        events.append(
            identity_drift_event(
                from_svc=old,
                to_svc=new,
                ts=t,
                anchor_service=new,
            )
        )
        t += timedelta(seconds=step_seconds)
    return events
