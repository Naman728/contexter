"""Tests for MemorySubstrate."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from contexter.events import Event
from contexter.identity_tracker import IdentityTracker
from contexter.memory_substrate import MemorySubstrate


@pytest.fixture
def substrate() -> MemorySubstrate:
    with MemorySubstrate(batch_size=2, identity=IdentityTracker()) as s:
        yield s


class TestSchemaAndBatching:
    def test_in_memory_table_exists(self, substrate: MemorySubstrate) -> None:
        rows = substrate.query(
            "SELECT table_name FROM information_schema.tables WHERE table_name = 'events'"
        )
        assert rows == [("events",)]

    def test_batched_flush_on_threshold(self, substrate: MemorySubstrate) -> None:
        substrate.ingest({"kind": "metric", "service": "a", "payload": {"v": 1}})
        assert substrate.query("SELECT COUNT(*) FROM events")[0][0] == 0
        substrate.ingest({"kind": "metric", "service": "b", "payload": {"v": 2}})
        assert substrate.query("SELECT COUNT(*) FROM events")[0][0] == 2

    def test_explicit_flush(self) -> None:
        with MemorySubstrate(batch_size=10_000) as s:
            s.ingest({"kind": "log", "service": "svc"})
            assert s.flush() == 1
            assert s.query("SELECT COUNT(*) FROM events")[0][0] == 1


class TestIngestPipeline:
    def test_ingest_many_flushes_remainder(self) -> None:
        with MemorySubstrate(batch_size=100) as s:
            ids = s.ingest_many(
                [
                    {"kind": "span", "service": "api"},
                    {"kind": "span", "service": "api"},
                ]
            )
            assert ids == [1, 2]
            assert s.query("SELECT COUNT(*) FROM events")[0][0] == 2

    def test_canonical_service_after_drift(self) -> None:
        identity = IdentityTracker()
        with MemorySubstrate(batch_size=4, identity=identity) as s:
            s.ingest(
                {
                    "kind": "identity.drift",
                    "service": "ignored",
                    "payload": {"old": "svc-v1", "new": "svc-v2"},
                }
            )
            s.ingest(
                {
                    "kind": "metric",
                    "service": "svc-v1",
                    "payload": {"cpu": 0.5},
                }
            )
            s.flush()
            row = s.query(
                "SELECT canonical_service, raw_service FROM events WHERE kind = 'metric'"
            )[0]
            assert row == ("svc-v2", "svc-v1")
        assert identity.resolve("svc-v1") == "svc-v2"

    def test_drift_accepts_from_and_to_keys(self) -> None:
        identity = IdentityTracker()
        with MemorySubstrate(batch_size=1, identity=identity) as s:
            s.ingest(
                {
                    "kind": "identity.drift",
                    "service": "x",
                    "payload": {"from_": "payments-svc", "to": "billing-svc"},
                }
            )
        assert identity.resolve("payments-svc") == "billing-svc"

    def test_drift_malformed_payload_does_not_raise(self) -> None:
        with MemorySubstrate(batch_size=1) as s:
            s.ingest(
                {
                    "kind": "identity.drift",
                    "service": "x",
                    "payload": {},
                }
            )


class TestRouting:
    def test_kind_router_invoked(self) -> None:
        seen: list[tuple[str, str]] = []

        def on_metric(event: Event, canonical: str) -> None:
            seen.append((event.kind, canonical))

        with MemorySubstrate(batch_size=1) as s:
            s.register_router("metric", on_metric)
            s.ingest({"kind": "metric", "service": "worker-1"})
            s.ingest({"kind": "log", "service": "worker-1"})

        assert seen == [("metric", "worker-1")]

    def test_wildcard_router(self) -> None:
        kinds: list[str] = []

        with MemorySubstrate(batch_size=1) as s:
            s.register_router("*", lambda e, _c: kinds.append(e.kind))
            s.ingest({"kind": "a", "service": "s"})
            s.ingest({"kind": "b", "service": "s"})

        assert kinds == ["a", "b"]


class TestQueries:
    def test_events_for_service_resolves_aliases(self) -> None:
        identity = IdentityTracker()
        with MemorySubstrate(batch_size=8, identity=identity) as s:
            s.ingest(
                {
                    "kind": "identity.drift",
                    "service": "x",
                    "payload": {"old": "alpha", "new": "beta"},
                }
            )
            s.ingest({"kind": "log", "service": "alpha", "payload": {"msg": "hi"}})
            s.flush()

            by_alias = s.events_for_service("alpha")
            by_canonical = s.events_for_service("beta")

        logs = [e for e in by_alias if e["kind"] == "log"]
        assert len(logs) == 1
        assert logs[0]["canonical_service"] == "beta"
        assert logs[0]["raw_service"] == "alpha"
        assert by_canonical == by_alias

    def test_events_by_kind(self) -> None:
        ts = datetime(2026, 5, 15, 12, 0, tzinfo=timezone.utc)
        with MemorySubstrate(batch_size=8) as s:
            s.ingest(Event("deploy", "api", {"rev": "abc"}, occurred_at=ts))
            s.flush()
            rows = s.events_by_kind("deploy")

        assert rows[0]["payload"] == {"rev": "abc"}
        assert rows[0]["occurred_at"] == ts
