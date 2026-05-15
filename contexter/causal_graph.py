"""Deploy-to-effect causal graph with incident snapshots."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from contexter.identity_tracker import IdentityTracker

_DEPLOY_LOOKBACK_S = 120
_MAX_DEPLOYS_PER_SERVICE = 20


@dataclass(frozen=True, slots=True)
class CausalEdge:
    """Directed causal link from a deploy (cause) to a downstream effect."""

    cause_id: str
    effect_id: str
    evidence: list[str]
    confidence: float
    occurred_at: datetime


class CausalGraph:
    """Links deploy events to subsequent effects on the same canonical service."""

    __slots__ = (
        "_deploys",
        "_edge_services",
        "_edges",
        "_identity",
        "_lookback_s",
        "_snapshots",
    )

    def __init__(
        self,
        identity: IdentityTracker | None = None,
        *,
        lookback_s: int = _DEPLOY_LOOKBACK_S,
    ) -> None:
        self._identity = identity if identity is not None else IdentityTracker()
        self._lookback_s = lookback_s
        self._deploys: dict[str, deque[tuple[str, str, datetime]]] = {}
        self._edges: list[CausalEdge] = []
        self._edge_services: dict[tuple[str, str], str] = {}
        self._snapshots: dict[str, list[CausalEdge]] = {}

    def record_deploy(self, event_id: str, service: str, ts: datetime) -> None:
        """Store a deploy event for ``service``, retaining only the last 20."""
        canonical = self._canonical(service)
        history = self._deploys.setdefault(
            canonical,
            deque(maxlen=_MAX_DEPLOYS_PER_SERVICE),
        )
        history.append((event_id, canonical, ts))

    def record_effect(
        self,
        event_id: str,
        service: str,
        ts: datetime,
        kind: str,
        trace_id: str | None = None,
    ) -> list[CausalEdge]:
        """Link recent deploys on ``service`` to this effect. Returns new edges."""
        canonical = self._canonical(service)
        created: list[CausalEdge] = []
        for deploy_id, _deploy_service, deploy_ts in self._deploys.get(canonical, ()):
            if deploy_ts >= ts:
                continue
            delta_s = (ts - deploy_ts).total_seconds()
            if delta_s > self._lookback_s:
                continue
            confidence = max(0.0, min(1.0, 1.0 - (delta_s / self._lookback_s)))
            evidence = [f"kind:{kind}", f"service:{canonical}"]
            if trace_id is not None:
                evidence.append(f"trace_id:{trace_id}")
            edge = CausalEdge(
                cause_id=deploy_id,
                effect_id=event_id,
                evidence=evidence,
                confidence=confidence,
                occurred_at=ts,
            )
            self._edges.append(edge)
            self._edge_services[(deploy_id, event_id)] = canonical
            created.append(edge)
        return created

    def snapshot_incident(
        self,
        incident_id: str,
        ts: datetime,
        service: str,
        window_s: int = 300,
    ) -> None:
        """Capture edges whose effects fall within ``window_s`` seconds before ``ts``."""
        canonical = self._canonical(service)
        window_start = ts - timedelta(seconds=window_s)
        matched: list[CausalEdge] = []
        for edge in self._edges:
            if self._edge_services.get((edge.cause_id, edge.effect_id)) != canonical:
                continue
            if window_start <= edge.occurred_at <= ts:
                matched.append(edge)
        self._snapshots[incident_id] = matched

    def edges_for_incident(self, incident_id: str) -> list[CausalEdge]:
        """Return snapshotted edges sorted by ``occurred_at`` ascending."""
        edges = self._snapshots.get(incident_id, [])
        return sorted(edges, key=lambda edge: edge.occurred_at)

    def _canonical(self, service: str) -> str:
        self._identity.register(service)
        return self._identity.resolve(service)
