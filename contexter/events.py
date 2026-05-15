"""Event types for the memory substrate ingest pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True, slots=True)
class Event:
    """A single ingestible record routed by ``kind`` and tied to a service identity."""

    kind: str
    service: str
    payload: dict[str, Any] | None = None
    occurred_at: datetime | None = None

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> Event:
        occurred = data.get("occurred_at")
        if isinstance(occurred, str):
            try:
                occurred = datetime.fromisoformat(occurred.replace("Z", "+00:00"))
            except ValueError:
                occurred = None
        payload = data.get("payload")
        if payload is not None and not isinstance(payload, dict):
            payload = None
        return cls(
            kind=str(data.get("kind", "unknown")),
            service=str(data.get("service", "unknown")),
            payload=payload,
            occurred_at=occurred,
        )

    def with_canonical_service(self, canonical: str) -> Event:
        if canonical == self.service:
            return self
        return Event(
            kind=self.kind,
            service=canonical,
            payload=self.payload,
            occurred_at=self.occurred_at,
        )

    def occurred_at_utc(self) -> datetime:
        if self.occurred_at is None:
            return datetime.now(timezone.utc)
        if self.occurred_at.tzinfo is None:
            return self.occurred_at.replace(tzinfo=timezone.utc)
        return self.occurred_at.astimezone(timezone.utc)

    def payload_json(self) -> str | None:
        if self.payload is None:
            return None
        return json.dumps(self.payload, separators=(",", ":"), sort_keys=True)
