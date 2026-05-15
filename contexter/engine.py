"""Single public façade wiring substrate, graph, fingerprints, and reconstruction."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Literal, cast

from contexter.causal_graph import CausalGraph
from contexter.context_reconstructor import Context, ContextReconstructor
from contexter.events import Event
from contexter.identity_tracker import IdentityTracker
from contexter.incident_neighborhood_memory import IncidentNeighborhoodMemory
from contexter.incident_fingerprint import (
    FingerprintExtractor,
    FingerprintMatcher,
    compute_retrieval_features,
    fingerprint_remediation_hash,
)
from contexter.memory_substrate import MemorySubstrate
from contexter.remediation_memory import RemediationMemory
from contexter.retrieval_adaptation import RetrievalAdaptation
from contexter.service_dependency_graph import parse_trace_call_edges
from contexter.roles import family_aware_role
from contexter.safe_context import coerce_signal, empty_context


class Engine:
    """Benchmark adapter entry point for ingest and context reconstruction."""

    __slots__ = (
        "_causal_graph",
        "_causal_window_s",
        "_fingerprint_matcher",
        "_last_ids",
        "_neighborhood_memory",
        "_reconstructor",
        "_remediation_memory",
        "_retrieval_adaptation",
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
        self._remediation_memory = RemediationMemory()
        self._fingerprint_matcher = FingerprintMatcher(
            FingerprintExtractor(identity, role_resolver=family_aware_role),
            remediation_memory=self._remediation_memory,
        )
        self._neighborhood_memory = IncidentNeighborhoodMemory(identity)
        self._retrieval_adaptation = RetrievalAdaptation()
        self._reconstructor = ContextReconstructor(
            self._substrate,
            self._causal_graph,
            self._fingerprint_matcher,
            self._remediation_memory,
            neighborhood_memory=self._neighborhood_memory,
            retrieval_adaptation=self._retrieval_adaptation,
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

    def retrieval_stats(self) -> dict[str, Any]:
        """Online retrieval adaptation metrics (weights, EMA contributions, failure modes)."""
        return self._retrieval_adaptation.stats()

    def retrieval_snapshot(self) -> dict[str, Any]:
        """Serializable retrieval adaptation state (bounded, deterministic round-trip)."""
        return self._retrieval_adaptation.snapshot()

    def restore_retrieval_snapshot(self, data: Mapping[str, Any]) -> None:
        """Restore adaptation from :meth:`retrieval_snapshot`."""
        self._retrieval_adaptation.restore_snapshot(data)

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
        self._substrate.register_router("trace", self._on_trace)
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
                payload=event.payload,
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
                payload=event.payload,
            )
        except Exception:
            return

    def _on_trace(self, event: Event, canonical: str) -> None:
        try:
            payload = event.payload if isinstance(event.payload, dict) else {}
            for caller, callee in parse_trace_call_edges(
                payload,
                self._substrate.identity,
            ):
                self._causal_graph.dependency_graph.add_call_edge(caller, callee)
        except Exception:
            return

    def _on_incident_signal(self, event: Event, canonical: str) -> None:
        try:
            payload = event.payload or {}
            raw_upstream = payload.get(
                "upstream", payload.get("upstream_services", ())
            )
            if not isinstance(raw_upstream, (list, tuple, set, frozenset)):
                raw_upstream = ()
            self._neighborhood_memory.record_incident_affiliation(
                canonical, raw_upstream
            )
            incident_id = str(payload.get("incident_id", self._pending_event_id()))
            signal_payload = {**payload, "service": canonical}
            ts = event.occurred_at_utc()
            self._causal_graph.snapshot_incident(incident_id, ts, canonical)
            fp = self._fingerprint_matcher._extractor.extract(signal_payload)
            edges = self._causal_graph.edges_for_incident(incident_id)
            feat = compute_retrieval_features(
                fp, edges, self._substrate, canonical, ts
            )
            ctx: dict[str, Any] = {}
            if (
                "deploy_window" in signal_payload
                and signal_payload.get("deploy_window") is not None
            ):
                ctx["deploy_window"] = signal_payload.get("deploy_window")
            if signal_payload.get("post_deploy_metric"):
                ctx["post_deploy_metric"] = True
            self._fingerprint_matcher.index(
                incident_id,
                fp,
                ctx,
                retrieval_features=feat,
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
            fp_hash = _fingerprint_hash_for_remediation(
                self._fingerprint_matcher, incident_id, fingerprint
            )
            self._remediation_memory.record(fp_hash, action, outcome=outcome)
        except Exception:
            return


def _fingerprint_hash_for_remediation(matcher: Any, incident_id: str, fingerprint: Any) -> str:
    rf = None
    if incident_id:
        try:
            rf = matcher.retrieval_features_for_incident(str(incident_id))
        except Exception:
            rf = None
    return fingerprint_remediation_hash(fingerprint, rf)


def _trace_id(event: Event) -> str | None:
    payload = event.payload or {}
    trace = payload.get("trace_id")
    return str(trace) if trace is not None else None


def _fingerprint_for_incident(
    matcher: FingerprintMatcher,
    incident_id: str,
) -> Any | None:
    for stored_id, fingerprint, _ctx, _rf in matcher._corpus:
        if stored_id == incident_id:
            return fingerprint
    return None

