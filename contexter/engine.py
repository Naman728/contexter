"""Single public façade wiring substrate, graph, fingerprints, and reconstruction."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal, cast

from contexter.causal_graph import CausalGraph
from contexter.context_reconstructor import Context, ContextReconstructor
from contexter.events import Event
from contexter.identity_tracker import IdentityTracker
from contexter.incident_fingerprint import FingerprintExtractor, FingerprintMatcher
from contexter.memory_substrate import MemorySubstrate
from contexter.remediation_memory import RemediationMemory
from contexter.roles import family_aware_role
from contexter.safe_context import coerce_signal, empty_context


class Engine:
    """Benchmark adapter entry point for ingest and context reconstruction."""

    __slots__ = (
        "_causal_graph",
        "_causal_window_s",
        "_fingerprint_matcher",
        "_last_ids",
        "_reconstructor",
        "_remediation_memory",
        "_substrate",
    )

    def __init__(
        self,
        *,
        batch_size: int = 256,
        causal_window_s: int = 120,
        claude_api_key: str | None = None,
    ) -> None:
        self._causal_window_s = causal_window_s
        identity = IdentityTracker()
        self._substrate = MemorySubstrate(batch_size=batch_size, identity=identity)
        self._causal_graph = CausalGraph(identity, lookback_s=causal_window_s)
        self._fingerprint_matcher = FingerprintMatcher(
            FingerprintExtractor(identity, role_resolver=family_aware_role)
        )
        self._remediation_memory = RemediationMemory()
        self._reconstructor = ContextReconstructor(
            self._substrate,
            self._causal_graph,
            self._fingerprint_matcher,
            self._remediation_memory,
            claude_api_key=claude_api_key,
        )
        self._last_ids: dict[str, int] = {}
        self._register_routers()

    def ingest(self, events: Iterable[dict[str, Any] | Event]) -> list[int]:
        """Ingest a batch of events. Returns assigned event ids."""
        ids: list[int] = []
        for event in events:
            try:
                ids.append(self._substrate.ingest(event))
            except Exception:
                continue
        try:
            self._substrate.flush()
        except Exception:
            pass
        return ids

    def ingest_one(self, event: dict[str, Any] | Event) -> int:
        """Ingest a single event. Returns the assigned event id."""
        return self._substrate.ingest(event)

    def reconstruct_context(
        self,
        signal: dict[str, Any],
        *,
        mode: str = "fast",
    ) -> Context:
        """Reconstruct incident context in fast or deep mode; never raises."""
        try:
            mode_lit = cast(
                Literal["fast", "deep"],
                mode if mode in ("fast", "deep") else "fast",
            )
            return self._reconstructor.reconstruct(
                coerce_signal(signal), mode=mode_lit
            )
        except Exception:
            return empty_context()

    def close(self) -> None:
        """Flush and close the underlying substrate."""
        self._substrate.close()

    def __enter__(self) -> Engine:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _register_routers(self) -> None:
        self._substrate.register_router("*", self._track_event_id)
        self._substrate.register_router("deploy", self._on_deploy)
        self._substrate.register_router("metric", self._on_metric)
        self._substrate.register_router("log", self._on_log)
        self._substrate.register_router("incident_signal", self._on_incident_signal)
        self._substrate.register_router("remediation", self._on_remediation)

    def _track_event_id(self, event: Event, canonical: str) -> None:
        self._last_ids["latest"] = self._substrate._next_id
        self._last_ids[event.kind] = self._substrate._next_id

    def _pending_event_id(self) -> str:
        return str(self._substrate._next_id)

    def _on_deploy(self, event: Event, canonical: str) -> None:
        try:
            self._causal_graph.record_deploy(
                self._pending_event_id(),
                canonical,
                event.occurred_at_utc(),
            )
        except Exception:
            return

    def _on_metric(self, event: Event, canonical: str) -> None:
        try:
            self._causal_graph.record_effect(
                self._pending_event_id(),
                canonical,
                event.occurred_at_utc(),
                "metric",
                trace_id=_trace_id(event),
            )
        except Exception:
            return

    def _on_log(self, event: Event, canonical: str) -> None:
        try:
            self._causal_graph.record_effect(
                self._pending_event_id(),
                canonical,
                event.occurred_at_utc(),
                "log",
                trace_id=_trace_id(event),
            )
        except Exception:
            return

    def _on_incident_signal(self, event: Event, canonical: str) -> None:
        try:
            payload = event.payload or {}
            incident_id = str(payload.get("incident_id", self._pending_event_id()))
            signal_payload = {**payload, "service": canonical}
            self._fingerprint_matcher.index_incident(incident_id, signal_payload)
            self._causal_graph.snapshot_incident(
                incident_id,
                event.occurred_at_utc(),
                canonical,
            )
        except Exception:
            return

    def _on_remediation(self, event: Event, canonical: str) -> None:
        try:
            payload = event.payload or {}
            incident_id = str(payload.get("incident_id", ""))
            action = str(payload.get("action", "unknown"))
            outcome = str(payload.get("outcome", "failed"))
            fingerprint = _fingerprint_for_incident(
                self._fingerprint_matcher, incident_id
            )
            if fingerprint is None:
                fingerprint = self._fingerprint_matcher._extractor.extract(
                    {**payload, "service": canonical}
                )
            fp_hash = _fingerprint_hash(fingerprint)
            self._remediation_memory.record(fp_hash, action, outcome=outcome)
        except Exception:
            return


def _trace_id(event: Event) -> str | None:
    payload = event.payload or {}
    trace = payload.get("trace_id")
    return str(trace) if trace is not None else None


def _fingerprint_for_incident(
    matcher: FingerprintMatcher,
    incident_id: str,
) -> Any | None:
    for stored_id, fingerprint in matcher._corpus:
        if stored_id == incident_id:
            return fingerprint
    return None


def _fingerprint_hash(fingerprint: Any) -> str:
    return (
        f"{fingerprint.trigger_type}:{fingerprint.affected_role}:"
        f"{bool(fingerprint.upstream_involved)}"
    )
