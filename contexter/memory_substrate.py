"""In-memory DuckDB substrate for event ingestion and routing."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import datetime
from typing import Any, Protocol

import duckdb

from contexter.events import Event
from contexter.identity_tracker import IdentityTracker

EventHandler = Callable[[Event, str], None]

_DRIFT_KIND: str = "identity.drift"

_EVENTS_DDL = """
CREATE TABLE events (
    event_id      BIGINT PRIMARY KEY,
    kind          VARCHAR NOT NULL,
    canonical_service VARCHAR NOT NULL,
    raw_service   VARCHAR NOT NULL,
    payload       JSON,
    occurred_at   TIMESTAMPTZ NOT NULL
)
"""


class Router(Protocol):
    def __call__(self, event: Event, canonical_service: str) -> None: ...


class MemorySubstrate:
    """In-memory DuckDB store with batched ingest, kind routing, and identity resolution."""

    __slots__ = (
        "_batch_size",
        "_conn",
        "_identity",
        "_next_id",
        "_pending",
        "_routers",
    )

    def __init__(
        self,
        *,
        batch_size: int = 256,
        identity: IdentityTracker | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self._batch_size = batch_size
        self._identity = identity if identity is not None else IdentityTracker()
        self._conn = duckdb.connect(database=":memory:")
        self._conn.execute(_EVENTS_DDL)
        self._conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_events_service_time
            ON events (canonical_service, occurred_at)
            """
        )
        self._next_id = 1
        self._pending: list[tuple[Any, ...]] = []
        self._routers: dict[str, list[EventHandler]] = {}

    @property
    def identity(self) -> IdentityTracker:
        return self._identity

    @property
    def connection(self) -> duckdb.DuckDBPyConnection:
        return self._conn

    def register_router(self, kind: str, handler: EventHandler) -> None:
        """Register a handler invoked for every ingested event of ``kind``."""
        self._routers.setdefault(kind, []).append(handler)

    def ingest(self, event: Event | Mapping[str, Any]) -> int:
        """Normalize, route, buffer, and optionally flush one event. Returns ``event_id``."""
        record = event if isinstance(event, Event) else Event.from_mapping(dict(event))
        canonical = self._resolve_service(record)
        canonical_event = record.with_canonical_service(canonical)

        self._dispatch(canonical_event, canonical)
        event_id = self._enqueue(canonical_event, record.service)
        if len(self._pending) >= self._batch_size:
            self.flush()
        return event_id

    def ingest_many(self, events: Iterable[Event | Mapping[str, Any]]) -> list[int]:
        """Ingest a sequence of events, flushing any remainder at the end."""
        ids: list[int] = []
        for event in events:
            ids.append(self.ingest(event))
        self.flush()
        return ids

    def flush(self) -> int:
        """Write buffered rows to DuckDB. Returns number of rows inserted."""
        if not self._pending:
            return 0
        self._conn.executemany(
            """
            INSERT INTO events (
                event_id, kind, canonical_service, raw_service, payload, occurred_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            self._pending,
        )
        count = len(self._pending)
        self._pending.clear()
        return count

    def query(
        self,
        sql: str,
        parameters: Sequence[Any] | None = None,
    ) -> list[tuple[Any, ...]]:
        if parameters is None:
            cursor = self._conn.execute(sql)
        else:
            cursor = self._conn.execute(sql, parameters)
        return cursor.fetchall()

    def events_for_service(
        self,
        service: str,
        *,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Return stored events for the canonical service (resolves aliases)."""
        if service:
            self._identity.register(service)
            canonical = self._identity.resolve(service)
        else:
            canonical = "unknown"
        if since is not None and until is not None:
            rows = self.query(
                """
                SELECT event_id, kind, canonical_service, raw_service, payload, occurred_at
                FROM events
                WHERE canonical_service = ?
                  AND occurred_at >= ?
                  AND occurred_at <= ?
                ORDER BY event_id
                """,
                [canonical, since, until],
            )
        else:
            rows = self.query(
                """
                SELECT event_id, kind, canonical_service, raw_service, payload, occurred_at
                FROM events
                WHERE canonical_service = ?
                ORDER BY event_id
                """,
                [canonical],
            )
        return [self._row_to_dict(row) for row in rows]

    def events_by_kind(self, kind: str) -> list[dict[str, Any]]:
        rows = self.query(
            """
            SELECT event_id, kind, canonical_service, raw_service, payload, occurred_at
            FROM events
            WHERE kind = ?
            ORDER BY event_id
            """,
            [kind],
        )
        return [self._row_to_dict(row) for row in rows]

    def last_deploy_before(
        self,
        service: str,
        until: datetime,
    ) -> datetime | None:
        """Most recent deploy on the resolved canonical service at or before ``until``."""
        if not service:
            return None
        self._identity.register(service)
        canonical = self._identity.resolve(service)
        rows = self.query(
            """
            SELECT occurred_at
            FROM events
            WHERE canonical_service = ?
              AND kind = 'deploy'
              AND occurred_at <= ?
            ORDER BY occurred_at DESC
            LIMIT 1
            """,
            [canonical, until],
        )
        if not rows:
            return None
        return rows[0][0]

    def close(self) -> None:
        self.flush()
        self._conn.close()

    def __enter__(self) -> MemorySubstrate:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _resolve_service(self, event: Event) -> str:
        if event.kind == _DRIFT_KIND:
            return self._resolve_drift(event)
        self._identity.register(event.service)
        return self._identity.resolve(event.service)

    def _resolve_drift(self, event: Event) -> str:
        payload = event.payload or {}
        old = payload.get("from_", payload.get("old"))
        new = payload.get("to", payload.get("new"))
        if old is None or new is None:
            fallback = event.service or "unknown"
            self._identity.register(fallback)
            return self._identity.resolve(fallback)
        self._identity.union(
            str(old),
            str(new),
            occurred_at=event.occurred_at_utc(),
        )
        return self._identity.resolve(str(new))

    def _dispatch(self, event: Event, canonical: str) -> None:
        for handler in self._routers.get(event.kind, ()):
            handler(event, canonical)
        for handler in self._routers.get("*", ()):
            handler(event, canonical)

    def _enqueue(self, event: Event, raw_service: str) -> int:
        event_id = self._next_id
        self._next_id += 1
        self._pending.append(
            (
                event_id,
                event.kind,
                event.service,
                raw_service,
                event.payload_json(),
                event.occurred_at_utc(),
            )
        )
        return event_id

    @staticmethod
    def _row_to_dict(row: tuple[Any, ...]) -> dict[str, Any]:
        payload = row[4]
        if isinstance(payload, str):
            payload = json.loads(payload)
        return {
            "event_id": row[0],
            "kind": row[1],
            "canonical_service": row[2],
            "raw_service": row[3],
            "payload": payload,
            "occurred_at": row[5],
        }
