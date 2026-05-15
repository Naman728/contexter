"""Synthetic noise and bulk DuckDB helpers for noise-flood benchmark tests."""

from __future__ import annotations

import json
import os
import random
from datetime import datetime, timedelta, timezone
from typing import Any

from contexter.memory_substrate import MemorySubstrate

UTC = timezone.utc


def noise_flood_log_count() -> int:
    """Primary noise volume; set ``NOISE_FLOOD_LOGS=100000`` for full §3 scale."""
    return int(os.environ.get("NOISE_FLOOD_LOGS", "9000"))


def _payload_json(obj: dict[str, Any]) -> str:
    return json.dumps(obj, separators=(",", ":"), sort_keys=True)


def bulk_insert_event_rows(
    substrate: MemorySubstrate,
    rows: list[tuple[Any, ...]],
) -> None:
    """Append rows to ``events`` without running ingest routers (DuckDB only).

    Rows are ``(event_id, kind, canonical_service, raw_service, payload, occurred_at)``
    matching ``MemorySubstrate._enqueue`` column order.
    """
    substrate.flush()
    if not rows:
        return
    substrate._conn.executemany(
        """
        INSERT INTO events (event_id, kind, canonical_service, raw_service, payload, occurred_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    max_id = max(int(r[0]) for r in rows)
    substrate._next_id = max(substrate._next_id, max_id + 1)


def flood_substrate_with_noise(
    substrate: MemorySubstrate,
    *,
    service: str,
    window_start: datetime,
    window_end: datetime,
    rng: random.Random,
    batch_size: int = 8000,
) -> dict[str, int]:
    """Bulk-load unrelated logs, metrics, traces, and deploys on ``service`` in the window.

    Inserts **no** causal-graph or fingerprint side effects (DuckDB rows only).
    Returns counts by kind inserted.
    """
    substrate.identity.register(service)
    n_logs = noise_flood_log_count()
    n_metrics = max(1, n_logs // 6)
    n_traces = max(1, n_logs // 12)
    n_deploys = max(1, n_logs // 18)
    span_s = max(1.0, (window_end - window_start).total_seconds())

    def rand_ts() -> datetime:
        return window_start + timedelta(seconds=rng.random() * span_s)

    counts = {"log": 0, "metric": 0, "trace": 0, "deploy": 0}
    cur_id = substrate._next_id
    pending: list[tuple[Any, ...]] = []

    def flush_pending() -> None:
        nonlocal pending
        if len(pending) >= batch_size:
            bulk_insert_event_rows(substrate, pending)
            pending = []

    def push_row(row: tuple[Any, ...], kind_key: str) -> None:
        nonlocal cur_id, pending
        pending.append(row)
        counts[kind_key] += 1
        cur_id += 1
        flush_pending()

    for _ in range(n_logs):
        push_row(
            (
                cur_id,
                "log",
                service,
                service,
                _payload_json({"level": "info", "msg": f"noise-{cur_id}"}),
                rand_ts(),
            ),
            "log",
        )

    for _ in range(n_metrics):
        push_row(
            (
                cur_id,
                "metric",
                service,
                service,
                _payload_json(
                    {
                        "name": rng.choice(["qps", "cpu", "heap"]),
                        "value": float(rng.randint(1, 9999)),
                    }
                ),
                rand_ts(),
            ),
            "metric",
        )

    for _ in range(n_traces):
        push_row(
            (
                cur_id,
                "trace",
                service,
                service,
                _payload_json({"trace_id": f"tr-{cur_id}", "spans": []}),
                rand_ts(),
            ),
            "trace",
        )

    for _ in range(n_deploys):
        push_row(
            (
                cur_id,
                "deploy",
                service,
                service,
                _payload_json(
                    {"version": f"v{rng.randint(0, 9)}.{rng.randint(0, 20)}.0", "actor": "noise-ci"}
                ),
                rand_ts(),
            ),
            "deploy",
        )

    if pending:
        bulk_insert_event_rows(substrate, pending)

    return counts
