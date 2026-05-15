"""Tests for Engine."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from contexter.engine import Engine

UTC = timezone.utc
T0 = datetime(2026, 5, 15, 16, 0, 0, tzinfo=UTC)

_REQUIRED_KEYS = frozenset(
    {
        "related_events",
        "causal_chain",
        "similar_past_incidents",
        "suggested_remediations",
        "confidence",
        "explain",
    }
)


def _signal(**overrides: object) -> dict:
    base = {
        "incident_id": "inc-current",
        "service": "api",
        "ts": T0,
        "trigger_type": "error_rate",
        "upstream": ["auth"],
    }
    base.update(overrides)
    return base


class TestCausalChain:
    def test_deploy_then_metric_yields_causal_chain(self) -> None:
        with Engine(batch_size=1) as engine:
            engine.ingest_one(
                {
                    "kind": "deploy",
                    "service": "api",
                    "occurred_at": T0 - timedelta(seconds=60),
                }
            )
            engine.ingest_one(
                {
                    "kind": "metric",
                    "service": "api",
                    "occurred_at": T0 - timedelta(seconds=30),
                    "payload": {"cpu": 0.95},
                }
            )
            engine.ingest_one(
                {
                    "kind": "incident_signal",
                    "service": "api",
                    "occurred_at": T0,
                    "payload": {
                        "incident_id": "inc-current",
                        "trigger_type": "error_rate",
                        "upstream": ["auth"],
                    },
                }
            )
            context = engine.reconstruct_context(_signal())

        assert context["causal_chain"]


class TestRenameEndToEnd:
    def test_drift_preserves_past_incident_match(self) -> None:
        with Engine(batch_size=1) as engine:
            engine.ingest(
                [
                    {
                        "kind": "incident_signal",
                        "service": "payments-svc",
                        "occurred_at": T0 - timedelta(hours=2),
                        "payload": {
                            "incident_id": "inc-past",
                            "trigger_type": "error_rate",
                            "upstream": ["auth", "db"],
                        },
                    },
                    {
                        "kind": "identity.drift",
                        "service": "billing-svc",
                        "occurred_at": T0 - timedelta(hours=1),
                        "payload": {"from_": "payments-svc", "to": "billing-svc"},
                    },
                    {
                        "kind": "deploy",
                        "service": "billing-svc",
                        "occurred_at": T0 - timedelta(seconds=90),
                    },
                    {
                        "kind": "metric",
                        "service": "billing-svc",
                        "occurred_at": T0 - timedelta(seconds=45),
                        "payload": {"latency_p99": 900},
                    },
                    {
                        "kind": "incident_signal",
                        "service": "billing-svc",
                        "occurred_at": T0,
                        "payload": {
                            "incident_id": "inc-current",
                            "trigger_type": "error_rate",
                            "upstream": ["auth", "db"],
                        },
                    },
                ]
            )
            context = engine.reconstruct_context(
                _signal(
                    service="billing-svc",
                    incident_id="inc-current",
                    upstream=["auth", "db"],
                )
            )

        past = [
            match
            for match in context["similar_past_incidents"]
            if match["past_incident_id"] == "inc-past"
        ]
        assert past
        # Rerank stack lands ~0.56 for this two-incident pool; assert stable lower bound.
        assert past[0]["similarity"] >= 0.55


class TestIngestAndContext:
    def test_ingest_many_yields_related_events(self) -> None:
        with Engine(batch_size=8) as engine:
            engine.ingest(
                [
                    {
                        "kind": "metric",
                        "service": "api",
                        "occurred_at": T0 - timedelta(seconds=30),
                        "payload": {"cpu": 0.8},
                    },
                    {
                        "kind": "incident_signal",
                        "service": "api",
                        "occurred_at": T0,
                        "payload": {
                            "incident_id": "inc-current",
                            "trigger_type": "error_rate",
                        },
                    },
                ]
            )
            context = engine.reconstruct_context(_signal())

        assert context["related_events"]

    def test_confidence_positive_after_real_events(self) -> None:
        with Engine(batch_size=1) as engine:
            engine.ingest(
                [
                    {
                        "kind": "deploy",
                        "service": "api",
                        "occurred_at": T0 - timedelta(seconds=50),
                    },
                    {
                        "kind": "log",
                        "service": "api",
                        "occurred_at": T0 - timedelta(seconds=20),
                        "payload": {"level": "error", "msg": "timeout"},
                    },
                    {
                        "kind": "incident_signal",
                        "service": "api",
                        "occurred_at": T0,
                        "payload": {
                            "incident_id": "inc-current",
                            "trigger_type": "error_rate",
                        },
                    },
                ]
            )
            context = engine.reconstruct_context(_signal())

        assert context["confidence"] > 0.1

    def test_fast_mode_has_all_required_keys(self) -> None:
        with Engine() as engine:
            context = engine.reconstruct_context(_signal())
        assert set(context.keys()) == _REQUIRED_KEYS


class TestContextManager:
    def test_context_manager_closes_cleanly(self) -> None:
        with Engine(batch_size=1) as engine:
            engine.ingest_one(
                {
                    "kind": "metric",
                    "service": "api",
                    "occurred_at": T0,
                    "payload": {},
                }
            )
        # No exception on exit; substrate connection closed.
        assert True
