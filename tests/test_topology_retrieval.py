"""Topology mutation: cross-alias recall and historical neighborhood retrieval."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from contexter.engine import Engine

UTC = timezone.utc
T0 = datetime(2026, 9, 1, 12, 0, 0, tzinfo=UTC)

PAST_ID = "INC-TOPO-PAST"
CURR_ID = "INC-TOPO-CURR"
_SIM_FLOOR = 0.55


def _assert_past_retrieved(context: dict, *, past_id: str) -> None:
    matches = context.get("similar_past_incidents") or []
    found = [m for m in matches if m.get("past_incident_id") == past_id]
    if not found:
        pytest.fail(
            "Expected past incident in similar_past_incidents.\n"
            f"  looking_for: {past_id!r}\n"
            f"  similar_past_incidents ({len(matches)}):\n"
            f"{json.dumps(list(matches), indent=2, default=str)}\n"
        )
    best = max(found, key=lambda m: float(m.get("similarity", 0.0)))
    assert float(best["similarity"]) >= _SIM_FLOOR


def _incident_signal(
    *,
    service: str,
    incident_id: str,
    ts: datetime,
    upstream: list[str],
    trigger: str = "error_rate",
) -> dict[str, Any]:
    return {
        "kind": "incident_signal",
        "service": service,
        "occurred_at": ts,
        "payload": {
            "incident_id": incident_id,
            "trigger_type": trigger,
            "upstream": upstream,
        },
    }


def _drift(from_: str, to: str, ts: datetime) -> dict[str, Any]:
    return {
        "kind": "identity.drift",
        "service": to,
        "occurred_at": ts,
        "payload": {"from_": from_, "to": to},
    }


class TestCrossAliasRetrieval:
    """Past fingerprint keeps pre-rename label; resolve() still joins the recall pool."""

    def test_stale_corpus_label_still_matches_after_chain(self) -> None:
        with Engine(batch_size=1) as engine:
            engine.ingest_one(
                _incident_signal(
                    service="svc-a",
                    incident_id=PAST_ID,
                    ts=T0,
                    upstream=["auth", "db"],
                )
            )
            t = T0 + timedelta(minutes=2)
            for pair in (
                ("svc-a", "svc-b"),
                ("svc-b", "svc-c"),
                ("svc-c", "svc-d"),
            ):
                engine.ingest_one(_drift(pair[0], pair[1], t))
                t += timedelta(seconds=45)

            engine.ingest_one(
                _incident_signal(
                    service="svc-d",
                    incident_id=CURR_ID,
                    ts=T0 + timedelta(hours=1),
                    upstream=["auth", "db"],
                )
            )

            context = engine.reconstruct_context(
                {
                    "incident_id": CURR_ID,
                    "service": "svc-d",
                    "ts": T0 + timedelta(hours=1, minutes=2),
                    "trigger_type": "error_rate",
                    "upstream": ["auth", "db"],
                },
                mode="fast",
            )

        _assert_past_retrieved(context, past_id=PAST_ID)


class TestNeighborhoodContinuityRetrieval:
    """Historical co-involvement lifts recall across services without shared canonical."""

    def test_historical_pair_cross_matches(self) -> None:
        with Engine(batch_size=1) as engine:
            engine.ingest_one(
                _incident_signal(
                    service="frontend-api",
                    incident_id=PAST_ID,
                    ts=T0,
                    upstream=["checkout-api", "auth"],
                )
            )
            engine.ingest_one(
                _incident_signal(
                    service="checkout-api",
                    incident_id=CURR_ID,
                    ts=T0 + timedelta(hours=3),
                    upstream=["auth"],
                )
            )

            context = engine.reconstruct_context(
                {
                    "incident_id": CURR_ID,
                    "service": "checkout-api",
                    "ts": T0 + timedelta(hours=3, minutes=1),
                    "trigger_type": "error_rate",
                    "upstream": ["auth"],
                },
                mode="fast",
            )

        _assert_past_retrieved(context, past_id=PAST_ID)
