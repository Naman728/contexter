"""Extreme rename-chain (storm) benchmark — ADVANCED_BENCHMARK_SUITE §2."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from contexter.engine import Engine
from tests.rename_storm_generators import (
    RENAME_STORM_CHAIN,
    RENAME_STORM_CHAIN_5,
    drift_chain_events,
    incident_signal_event,
)

UTC = timezone.utc
T0 = datetime(2026, 8, 1, 10, 0, 0, tzinfo=UTC)

PAST_INCIDENT_ID = "INC-STORM-PAST"
CURRENT_INCIDENT_ID = "INC-STORM-CURRENT"
SIMILARITY_FLOOR = 0.6


def _format_matches(context: dict) -> str:
    raw = context.get("similar_past_incidents") or []
    try:
        return json.dumps(list(raw), indent=2, default=str)
    except TypeError:
        return repr(raw)


def _assert_past_retrieved(context: dict, *, past_id: str) -> None:
    matches = context.get("similar_past_incidents") or []
    found = [m for m in matches if m.get("past_incident_id") == past_id]
    if not found:
        pytest.fail(
            "Expected past incident in similar_past_incidents after rename storm.\n"
            f"  looking_for: {past_id!r}\n"
            f"  similar_past_incidents ({len(matches)}):\n{_format_matches(context)}\n"
            f"  confidence: {context.get('confidence')!r}\n"
            f"  explain (prefix): {str(context.get('explain', ''))[:400]!r}\n"
        )
    best = max(found, key=lambda m: float(m.get("similarity", 0.0)))
    sim = float(best["similarity"])
    assert sim >= SIMILARITY_FLOOR, (
        f"Similarity too low after rename storm: {sim} < {SIMILARITY_FLOOR}\n"
        f"  match: {best!r}\n"
        f"  all_matches:\n{_format_matches(context)}"
    )


class TestRenameStormChain:
    """Union-find path across svc-a → … → svc-e preserves structural retrieval."""

    def test_four_hop_chain_retrieves_past_incident(self) -> None:
        with Engine(batch_size=1) as engine:
            engine.ingest_one(
                incident_signal_event(
                    service="svc-a",
                    incident_id=PAST_INCIDENT_ID,
                    ts=T0,
                    trigger_type="error_rate",
                    upstream=["auth", "db"],
                )
            )
            engine.ingest(drift_chain_events(T0 + timedelta(minutes=1), step_seconds=30))
            engine.ingest_one(
                incident_signal_event(
                    service="svc-e",
                    incident_id=CURRENT_INCIDENT_ID,
                    ts=T0 + timedelta(hours=2),
                    trigger_type="error_rate",
                    upstream=["auth", "db"],
                )
            )

            context = engine.reconstruct_context(
                {
                    "incident_id": CURRENT_INCIDENT_ID,
                    "service": "svc-e",
                    "ts": T0 + timedelta(hours=2, minutes=5),
                    "trigger_type": "error_rate",
                    "upstream": ["auth", "db"],
                },
                mode="fast",
            )

        _assert_past_retrieved(context, past_id=PAST_INCIDENT_ID)

    def test_query_from_original_alias_still_resolves(self) -> None:
        """Query using pre-chain name ``svc-a``; identity resolves to canonical ``svc-e``."""
        with Engine(batch_size=1) as engine:
            engine.ingest_one(
                incident_signal_event(
                    service="svc-a",
                    incident_id=PAST_INCIDENT_ID,
                    ts=T0,
                )
            )
            engine.ingest(drift_chain_events(T0 + timedelta(minutes=5)))
            engine.ingest_one(
                incident_signal_event(
                    service="svc-e",
                    incident_id=CURRENT_INCIDENT_ID,
                    ts=T0 + timedelta(hours=1),
                )
            )

            context = engine.reconstruct_context(
                {
                    "incident_id": CURRENT_INCIDENT_ID,
                    "service": "svc-a",
                    "ts": T0 + timedelta(hours=1, minutes=1),
                    "trigger_type": "error_rate",
                    "upstream": ["auth", "db"],
                },
                mode="fast",
            )

        _assert_past_retrieved(context, past_id=PAST_INCIDENT_ID)

    def test_identity_all_aliases_resolve_to_terminal(self) -> None:
        """Every name in the chain shares one canonical representative (union-find)."""
        with Engine(batch_size=1) as engine:
            engine.ingest(drift_chain_events(T0))
            identity = engine._substrate.identity
            expected = identity.resolve("svc-e")
            for old, _ in RENAME_STORM_CHAIN:
                got = identity.resolve(old)
                assert got == expected, (
                    f"resolve({old!r}) -> {got!r}, expected {expected!r}\n"
                    f"  chain: {RENAME_STORM_CHAIN!r}"
                )
            assert identity.resolve("svc-a") == expected

    def test_five_hop_chain_retrieves_past_incident(self) -> None:
        with Engine(batch_size=1) as engine:
            engine.ingest_one(
                incident_signal_event(
                    service="svc-a",
                    incident_id=PAST_INCIDENT_ID,
                    ts=T0,
                    trigger_type="error_rate",
                    upstream=["auth", "db"],
                )
            )
            engine.ingest(
                drift_chain_events(
                    T0 + timedelta(minutes=1),
                    step_seconds=30,
                    chain=RENAME_STORM_CHAIN_5,
                )
            )
            engine.ingest_one(
                incident_signal_event(
                    service="svc-f",
                    incident_id=CURRENT_INCIDENT_ID,
                    ts=T0 + timedelta(hours=2),
                    trigger_type="error_rate",
                    upstream=["auth", "db"],
                )
            )

            context = engine.reconstruct_context(
                {
                    "incident_id": CURRENT_INCIDENT_ID,
                    "service": "svc-f",
                    "ts": T0 + timedelta(hours=2, minutes=5),
                    "trigger_type": "error_rate",
                    "upstream": ["auth", "db"],
                },
                mode="fast",
            )

        _assert_past_retrieved(context, past_id=PAST_INCIDENT_ID)
