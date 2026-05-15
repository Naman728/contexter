#!/usr/bin/env python3
"""Measure ingest throughput and reconstruct latency."""

from __future__ import annotations

import cProfile
import pstats
import sys
import time
from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from contexter import Engine

UTC = timezone.utc
T0 = datetime(2026, 6, 1, 12, 0, 0, tzinfo=UTC)
KINDS = ("deploy", "metric", "log", "trace", "metric")
N_EVENTS = 5000
N_RECONSTRUCT = 50
INGEST_TARGET_EPS = 1000
RECONSTRUCT_P95_MS = 2000
BATCH_CANDIDATES = (50, 256, 512)


def ts(offset_s: int) -> datetime:
    return T0 + timedelta(seconds=offset_s)


def generate_events(n: int) -> list[dict]:
    events: list[dict] = []
    for i in range(n):
        kind = KINDS[i % len(KINDS)]
        service = f"svc-{i % 20}"
        payload: dict | None = None
        if kind == "metric":
            payload = {"cpu": i % 100}
        elif kind == "log":
            payload = {"level": "error" if i % 3 == 0 else "info", "msg": f"e{i}"}
        elif kind == "trace":
            payload = {"trace_id": f"tr-{i}"}
        events.append(
            {
                "kind": kind,
                "service": service,
                "occurred_at": ts(i),
                "payload": payload,
            }
        )
    return events


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = (len(ordered) - 1) * pct
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def measure_ingest(batch_size: int, events: list[dict]) -> tuple[float, Engine]:
    engine = Engine(batch_size=batch_size, claude_api_key=None)
    start = time.perf_counter()
    engine.ingest(events)
    elapsed = time.perf_counter() - start
    return elapsed, engine


def measure_reconstruct(engine: Engine, signal: dict) -> list[float]:
    latencies: list[float] = []
    for _ in range(N_RECONSTRUCT):
        start = time.perf_counter()
        engine.reconstruct_context(signal, mode="fast")
        latencies.append(time.perf_counter() - start)
    return latencies


def profile_reconstruct(engine: Engine, signal: dict) -> None:
    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(10):
        engine.reconstruct_context(signal, mode="fast")
    profiler.disable()
    stream = StringIO()
    stats = pstats.Stats(profiler, stream=stream).sort_stats("cumulative")
    stats.print_stats(20)
    print("\n--- cProfile (reconstruct, top 20 cumulative) ---", file=sys.stderr)
    print(stream.getvalue(), file=sys.stderr)


def best_ingest_throughput(events: list[dict]) -> tuple[float, int, Engine]:
    best_eps = 0.0
    best_batch = BATCH_CANDIDATES[0]
    best_engine: Engine | None = None
    for batch_size in BATCH_CANDIDATES:
        elapsed, engine = measure_ingest(batch_size, events)
        eps = N_EVENTS / elapsed if elapsed > 0 else 0.0
        if eps > best_eps:
            if best_engine is not None:
                best_engine.close()
            best_eps = eps
            best_batch = batch_size
            best_engine = engine
        else:
            engine.close()
    assert best_engine is not None
    return best_eps, best_batch, best_engine


def main() -> int:
    events = generate_events(N_EVENTS)
    signal = {
        "service": "svc-0",
        "incident_id": "inc-perf",
        "ts": ts(N_EVENTS),
        "trigger_type": "error_rate",
        "upstream": ["auth"],
    }

    ingest_eps, batch_size, engine = best_ingest_throughput(events)
    ingest_elapsed = N_EVENTS / ingest_eps if ingest_eps > 0 else 0.0
    print(f"Ingest throughput: {ingest_eps:.0f} events/sec")

    latencies = measure_reconstruct(engine, signal)
    p50 = percentile(latencies, 0.50)
    p95 = percentile(latencies, 0.95)
    print(f"reconstruct p50: {p50 * 1000:.1f}ms  p95: {p95 * 1000:.1f}ms")

    ingest_ok = ingest_eps > INGEST_TARGET_EPS
    reconstruct_ok = (p95 * 1000) < RECONSTRUCT_P95_MS

    if not ingest_ok or not reconstruct_ok:
        print(
            f"\n(batch_size={batch_size} selected for ingest tuning)",
            file=sys.stderr,
        )
        if not reconstruct_ok:
            profile_reconstruct(engine, signal)

    engine.close()

    if not ingest_ok:
        print(f"FAIL: ingest below {INGEST_TARGET_EPS} events/sec", file=sys.stderr)
    if not reconstruct_ok:
        print(f"FAIL: reconstruct p95 above {RECONSTRUCT_P95_MS}ms", file=sys.stderr)
    return 0 if ingest_ok and reconstruct_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
