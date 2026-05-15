#!/usr/bin/env python3
"""Smoke test: topology rename preserves incident similarity and remediation."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Allow running as `python scripts/rename_smoke.py` from project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from contexter import Engine

UTC = timezone.utc
T0 = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)


def ts(seconds: float) -> datetime:
    return T0 + timedelta(seconds=seconds)


def check(label: str, condition: bool) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"{status}: {label}")
    return condition


def main() -> int:
    engine = Engine(claude_api_key=None)

    engine.ingest_one(
        {"kind": "deploy", "service": "payments-svc", "occurred_at": ts(0)}
    )
    engine.ingest_one(
        {
            "kind": "metric",
            "service": "payments-svc",
            "occurred_at": ts(30),
            "payload": {"latency_p99_ms": 4800},
        }
    )
    engine.ingest_one(
        {
            "kind": "incident_signal",
            "service": "payments-svc",
            "occurred_at": ts(35),
            "payload": {
                "incident_id": "INC-100",
                "trigger_type": "error_rate",
                "upstream": ["auth", "db"],
            },
        }
    )
    engine.ingest_one(
        {
            "kind": "remediation",
            "service": "payments-svc",
            "occurred_at": ts(60),
            "payload": {
                "incident_id": "INC-100",
                "action": "rollback",
                "outcome": "resolved",
            },
        }
    )
    engine.ingest_one(
        {
            "kind": "identity.drift",
            "service": "payments-svc",
            "occurred_at": ts(120),
            "payload": {"old": "payments-svc", "new": "billing-svc"},
        }
    )
    engine.ingest_one(
        {"kind": "deploy", "service": "billing-svc", "occurred_at": ts(300)}
    )
    engine.ingest_one(
        {
            "kind": "metric",
            "service": "billing-svc",
            "occurred_at": ts(330),
            "payload": {"latency_p99_ms": 5100},
        }
    )
    engine.ingest_one(
        {
            "kind": "incident_signal",
            "service": "billing-svc",
            "occurred_at": ts(335),
            "payload": {
                "incident_id": "INC-714",
                "trigger_type": "error_rate",
                "upstream": ["auth", "db"],
            },
        }
    )

    context = engine.reconstruct_context(
        {
            "service": "billing-svc",
            "incident_id": "INC-714",
            "ts": ts(335),
            "trigger_type": "error_rate",
            "upstream": ["auth", "db"],
        },
        mode="fast",
    )

    similar = context["similar_past_incidents"]
    inc_100 = [m for m in similar if m["past_incident_id"] == "INC-100"]
    remediations = context["suggested_remediations"]

    results = [
        check("similar_past_incidents is non-empty", bool(similar)),
        check("INC-100 appears in similar_past_incidents", bool(inc_100)),
        check(
            "similarity for INC-100 is >= 0.6",
            bool(inc_100) and inc_100[0]["similarity"] >= 0.6,
        ),
        check("suggested_remediations is non-empty", bool(remediations)),
        check(
            'suggested_remediations[0]["target"] == "billing-svc" (not payments-svc)',
            bool(remediations)
            and remediations[0]["target"] == "billing-svc"
            and remediations[0]["target"] != "payments-svc",
        ),
        check("confidence >= 0.1", context["confidence"] >= 0.1),
        check("explain is a non-empty string", bool(context["explain"])),
        check("causal_chain is non-empty", bool(context["causal_chain"])),
    ]

    engine.close()
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
