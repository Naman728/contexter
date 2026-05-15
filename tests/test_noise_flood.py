"""Advanced Benchmark Suite §3 — noise flood (retrieval + latency under DB bloat)."""

from __future__ import annotations

import json
import os
import random
import time
from datetime import datetime, timedelta, timezone

import pytest

from contexter.engine import Engine
from tests.noise_generators import flood_substrate_with_noise, noise_flood_log_count

UTC = timezone.utc
T0 = datetime(2026, 9, 1, 8, 0, 0, tzinfo=UTC)
TARGET_SVC = "noise-flood-target"
PAST_ID = "INC-NOISE-PAST"
CURRENT_ID = "INC-NOISE-CURRENT"
SIMILARITY_FLOOR = 0.6


def _latency_budget_ms() -> float:
    n = noise_flood_log_count()
    if n >= 80_000:
        return 25_000.0
    if n >= 40_000:
        return 12_000.0
    if n >= 15_000:
        return 4_000.0
    return 1_500.0


def _fail_debug(context: dict, *, past_id: str, latency_ms: float, noise_counts: dict[str, int]) -> str:
    matches = context.get("similar_past_incidents") or []
    try:
        dumped = json.dumps(list(matches), indent=2, default=str)
    except TypeError:
        dumped = repr(matches)
    return (
        f"expected_past_incident_id={past_id!r}\n"
        f"noise_counts={noise_counts!r}\n"
        f"NOISE_FLOOD_LOGS={os.environ.get('NOISE_FLOOD_LOGS', '(default)')!r}\n"
        f"reconstruct_latency_ms={latency_ms:.2f}\n"
        f"similar_past_incidents ({len(matches)}):\n{dumped}\n"
        f"confidence={context.get('confidence')!r}\n"
        f"related_events_len={len(context.get('related_events') or [])}\n"
    )


class TestNoiseFloodRetrieval:
    def test_past_incident_survives_massive_unrelated_telemetry(self) -> None:
        rng = random.Random(20260901)
        window_end = T0 + timedelta(hours=6)
        window_start = window_end - timedelta(seconds=280)

        with Engine(batch_size=2048) as engine:
            substrate = engine._substrate
            noise_counts = flood_substrate_with_noise(
                substrate,
                service=TARGET_SVC,
                window_start=window_start,
                window_end=window_end,
                rng=rng,
                batch_size=8000,
            )
            total_noise = sum(noise_counts.values())

            engine.ingest_one(
                {
                    "kind": "incident_signal",
                    "service": TARGET_SVC,
                    "occurred_at": T0 + timedelta(hours=1),
                    "payload": {
                        "incident_id": PAST_ID,
                        "trigger_type": "error_rate",
                        "upstream": ["auth", "db"],
                    },
                }
            )
            engine.ingest_one(
                {
                    "kind": "incident_signal",
                    "service": TARGET_SVC,
                    "occurred_at": window_end - timedelta(minutes=2),
                    "payload": {
                        "incident_id": CURRENT_ID,
                        "trigger_type": "error_rate",
                        "upstream": ["auth", "db"],
                    },
                }
            )

            signal_ts = window_end - timedelta(seconds=30)
            signal = {
                "incident_id": CURRENT_ID,
                "service": TARGET_SVC,
                "ts": signal_ts,
                "trigger_type": "error_rate",
                "upstream": ["auth", "db"],
            }

            t0 = time.perf_counter()
            context = engine.reconstruct_context(signal, mode="fast")
            latency_ms = (time.perf_counter() - t0) * 1000.0

        related = context.get("related_events") or []
        info_logs_in_related = sum(
            1
            for e in related
            if e.get("kind") == "log"
            and (e.get("payload") or {}).get("level") == "info"
        )
        assert info_logs_in_related == 0, (
            "Info-level noise logs must not appear in related_events (only error logs qualify).\n"
            f"info_logs_in_related={info_logs_in_related} related_len={len(related)}\n"
            + _fail_debug(context, past_id=PAST_ID, latency_ms=latency_ms, noise_counts=noise_counts)
        )

        assert total_noise >= min(5000, noise_flood_log_count()), (
            f"expected substantial noise volume, got total_noise={total_noise}\n"
            + _fail_debug(context, past_id=PAST_ID, latency_ms=latency_ms, noise_counts=noise_counts)
        )

        hits = [m for m in (context.get("similar_past_incidents") or []) if m.get("past_incident_id") == PAST_ID]
        if not hits:
            pytest.fail(
                "Past incident not in similar_past_incidents.\n"
                + _fail_debug(context, past_id=PAST_ID, latency_ms=latency_ms, noise_counts=noise_counts)
            )

        best = max(hits, key=lambda m: float(m.get("similarity", 0.0)))
        sim = float(best["similarity"])
        assert sim >= SIMILARITY_FLOOR, (
            f"similarity {sim} < {SIMILARITY_FLOOR}\n"
            + _fail_debug(context, past_id=PAST_ID, latency_ms=latency_ms, noise_counts=noise_counts)
        )

        budget = _latency_budget_ms()
        assert latency_ms < budget, (
            f"reconstruct too slow: {latency_ms:.1f}ms >= {budget}ms (noise log count={noise_flood_log_count()})\n"
            + _fail_debug(context, past_id=PAST_ID, latency_ms=latency_ms, noise_counts=noise_counts)
        )

    def test_reconstruct_latency_stable_across_warmup(self) -> None:
        """Several consecutive reconstructs stay under budget (no pathological growth)."""
        if int(os.environ.get("NOISE_FLOOD_LOGS", "9000")) > 25_000:
            pytest.skip("Skip deep latency sweep when NOISE_FLOOD_LOGS is very large")

        rng = random.Random(20260902)
        window_end = T0 + timedelta(hours=3)
        window_start = window_end - timedelta(seconds=250)

        with Engine(batch_size=2048) as engine:
            flood_substrate_with_noise(
                engine._substrate,
                service=TARGET_SVC,
                window_start=window_start,
                window_end=window_end,
                rng=rng,
            )
            engine.ingest_one(
                {
                    "kind": "incident_signal",
                    "service": TARGET_SVC,
                    "occurred_at": T0 + timedelta(minutes=30),
                    "payload": {
                        "incident_id": PAST_ID,
                        "trigger_type": "error_rate",
                        "upstream": ["auth"],
                    },
                }
            )
            engine.ingest_one(
                {
                    "kind": "incident_signal",
                    "service": TARGET_SVC,
                    "occurred_at": window_end - timedelta(minutes=1),
                    "payload": {
                        "incident_id": CURRENT_ID,
                        "trigger_type": "error_rate",
                        "upstream": ["auth"],
                    },
                }
            )
            signal = {
                "incident_id": CURRENT_ID,
                "service": TARGET_SVC,
                "ts": window_end - timedelta(seconds=10),
                "trigger_type": "error_rate",
                "upstream": ["auth"],
            }

            latencies: list[float] = []
            for _ in range(7):
                t0 = time.perf_counter()
                engine.reconstruct_context(signal, mode="fast")
                latencies.append((time.perf_counter() - t0) * 1000.0)

        worst = max(latencies)
        budget = _latency_budget_ms()
        assert worst < budget, (
            f"worst reconstruct {worst:.1f}ms across warmup runs (budget {budget}ms), latencies={latencies!r}"
        )
