"""Benchmark adapter contract tests."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import adapter
from contexter.safe_context import empty_context

UTC = timezone.utc
T0 = datetime(2026, 7, 1, 0, 0, 0, tzinfo=UTC)

_REQUIRED = frozenset(
    {
        "related_events",
        "causal_chain",
        "similar_past_incidents",
        "suggested_remediations",
        "confidence",
        "explain",
    }
)


def setup_function() -> None:
    adapter.reset()


def teardown_function() -> None:
    adapter.reset()


def test_ingest_returns_none() -> None:
    assert adapter.ingest([]) is None


def test_reconstruct_before_ingest_returns_valid_context() -> None:
    ctx = adapter.reconstruct_context(
        {"service": "api", "incident_id": "x", "ts": T0, "trigger_type": "error_rate"}
    )
    assert set(ctx.keys()) == _REQUIRED
    assert 0.1 <= ctx["confidence"] <= 1.0
    assert isinstance(ctx["explain"], str) and ctx["explain"]


def test_reconstruct_never_raises_on_garbage() -> None:
    ctx = adapter.reconstruct_context({})
    assert set(ctx.keys()) == _REQUIRED


def test_similar_past_incidents_at_most_five() -> None:
    adapter.ingest(
        [
            {
                "kind": "incident_signal",
                "service": "api-f1",
                "occurred_at": T0,
                "payload": {
                    "incident_id": f"inc-{i}",
                    "trigger_type": "error_rate",
                    "upstream": ["auth"],
                },
            }
            for i in range(10)
        ]
    )
    ctx = adapter.reconstruct_context(
        {
            "service": "api-f1",
            "incident_id": "inc-query",
            "ts": T0 + timedelta(seconds=10),
            "trigger_type": "error_rate",
            "upstream": ["auth"],
        }
    )
    assert len(ctx["similar_past_incidents"]) <= 5


def test_rollback_in_remediations_after_history() -> None:
    adapter.ingest(
        [
            {
                "kind": "incident_signal",
                "service": "api",
                "occurred_at": T0,
                "payload": {
                    "incident_id": "inc-1",
                    "trigger_type": "error_rate",
                },
            },
            {
                "kind": "remediation",
                "service": "api",
                "occurred_at": T0 + timedelta(seconds=30),
                "payload": {
                    "incident_id": "inc-1",
                    "action": "rollback",
                    "outcome": "resolved",
                },
            },
        ]
    )
    ctx = adapter.reconstruct_context(
        {
            "service": "api",
            "incident_id": "inc-2",
            "ts": T0 + timedelta(seconds=60),
            "trigger_type": "error_rate",
        }
    )
    actions = [r["action"] for r in ctx["suggested_remediations"]]
    assert "rollback" in actions
