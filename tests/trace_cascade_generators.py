"""Realistic distributed-trace payloads for multi-hop cascade tests."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

UTC = timezone.utc


def checkout_to_database_trace(
    *,
    trace_id: str = "trace-cascade-1",
    occurred_at: datetime | None = None,
    emitter_service: str = "frontend",
) -> dict[str, Any]:
    """Single trace: frontend → checkout-api → payments → database."""
    ts = occurred_at or datetime(2026, 5, 15, 18, 0, 0, tzinfo=UTC)
    return {
        "kind": "trace",
        "service": emitter_service,
        "occurred_at": ts,
        "payload": {
            "trace_id": trace_id,
            "spans": [
                {
                    "span_id": "s-fe",
                    "parent_span_id": None,
                    "service": "frontend",
                },
                {
                    "span_id": "s-co",
                    "parent_span_id": "s-fe",
                    "service": "checkout-api",
                },
                {
                    "span_id": "s-pay",
                    "parent_span_id": "s-co",
                    "service": "payments",
                },
                {
                    "span_id": "s-db",
                    "parent_span_id": "s-pay",
                    "service": "database",
                },
            ],
        },
    }


def degraded_latency_metric(
    service: str,
    *,
    occurred_at: datetime,
    name: str = "p99_latency_ms",
) -> dict[str, Any]:
    return {
        "kind": "metric",
        "service": service,
        "occurred_at": occurred_at,
        "payload": {
            "name": name,
            "value": 9000.0,
            "degraded": True,
        },
    }


def incident_signal(
    *,
    incident_id: str,
    service: str,
    occurred_at: datetime,
    trigger_type: str = "error_rate",
    upstream: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "kind": "incident_signal",
        "service": service,
        "occurred_at": occurred_at,
        "payload": {
            "incident_id": incident_id,
            "trigger_type": trigger_type,
            "upstream": upstream or ["database"],
        },
    }


def cascade_timeline(
    t0: datetime,
) -> list[dict[str, Any]]:
    """Ordered events: topology first, then root-outward latency spikes, then signal."""
    return [
        checkout_to_database_trace(occurred_at=t0),
        degraded_latency_metric("database", occurred_at=t0 + timedelta(seconds=5)),
        degraded_latency_metric("payments", occurred_at=t0 + timedelta(seconds=15)),
        degraded_latency_metric("checkout-api", occurred_at=t0 + timedelta(seconds=25)),
        degraded_latency_metric("frontend", occurred_at=t0 + timedelta(seconds=35)),
        incident_signal(
            incident_id="inc-frontend-cascade",
            service="frontend",
            occurred_at=t0 + timedelta(seconds=45),
        ),
    ]
