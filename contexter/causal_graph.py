"""Deploy-to-effect causal graph with incident snapshots and cross-service propagation."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from contexter.identity_tracker import IdentityTracker
from contexter.service_dependency_graph import ServiceDependencyGraph

_DEPLOY_LOOKBACK_S = 120
_MAX_DEPLOYS_PER_SERVICE = 20
_PROPAGATION_TIME_WINDOW_S = 90
_PROPAGATION_BASE_CONFIDENCE = 0.82
_PROPAGATION_HOP_DECAY = 0.85
_MAX_ANOMALY_HISTORY = 64


@dataclass(frozen=True, slots=True)
class CausalEdge:
    """Directed causal link (deploy→effect or cross-service propagation)."""

    cause_id: str
    effect_id: str
    evidence: list[str]
    confidence: float
    occurred_at: datetime
    cause_service: str = ""
    effect_service: str = ""


class CausalGraph:
    """Links deploy events to effects and correlates anomalies along trace-learned topology."""

    __slots__ = (
        "_anomalies",
        "_deploys",
        "_dependency",
        "_edge_pairs",
        "_edges",
        "_hop_decay",
        "_identity",
        "_lookback_s",
        "_propagation_window_s",
        "_snapshots",
    )

    def __init__(
        self,
        identity: IdentityTracker | None = None,
        *,
        lookback_s: int = _DEPLOY_LOOKBACK_S,
        propagation_window_s: int = _PROPAGATION_TIME_WINDOW_S,
        hop_decay: float = _PROPAGATION_HOP_DECAY,
    ) -> None:
        self._identity = identity if identity is not None else IdentityTracker()
        self._lookback_s = lookback_s
        self._propagation_window_s = propagation_window_s
        self._hop_decay = float(hop_decay)
        self._deploys: dict[str, deque[tuple[str, str, datetime]]] = {}
        self._edges: list[CausalEdge] = []
        self._edge_pairs: set[tuple[str, str]] = set()
        self._snapshots: dict[str, list[CausalEdge]] = {}
        self._dependency = ServiceDependencyGraph(self._identity)
        self._anomalies: dict[str, deque[tuple[datetime, str]]] = {}

    @property
    def dependency_graph(self) -> ServiceDependencyGraph:
        return self._dependency

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
        payload: dict[str, Any] | None = None,
    ) -> list[CausalEdge]:
        """Link recent deploys on ``service`` to this effect; optionally propagate."""
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
                cause_service=canonical,
                effect_service=canonical,
            )
            self._append_edge(edge)
            created.append(edge)

        if self._is_degrading(kind, payload):
            created.extend(
                self._cross_service_propagate(
                    canonical,
                    ts,
                    event_id,
                    kind,
                )
            )
            self._remember_anomaly(canonical, ts, event_id)
        return created

    def record_propagation_edge(
        self,
        *,
        cause_id: str,
        effect_id: str,
        cause_service: str,
        effect_service: str,
        ts: datetime,
        confidence: float,
        hop: int,
    ) -> CausalEdge | None:
        """Public hook for tests; stores a single propagation edge if not duplicate."""
        cs = self._canonical(cause_service)
        es = self._canonical(effect_service)
        evidence = [
            f"propagation:{hop}",
            f"from:{cs}",
            f"to:{es}",
        ]
        edge = CausalEdge(
            cause_id=cause_id,
            effect_id=effect_id,
            evidence=evidence,
            confidence=max(0.0, min(1.0, confidence)),
            occurred_at=ts,
            cause_service=cs,
            effect_service=es,
        )
        if self._append_edge(edge):
            return edge
        return None

    def snapshot_incident(
        self,
        incident_id: str,
        ts: datetime,
        service: str,
        window_s: int = 300,
    ) -> None:
        """Capture deploy edges and propagation paths ending at ``service`` within the window."""
        canonical = self._canonical(service)
        window_start = ts - timedelta(seconds=window_s)
        matched: list[CausalEdge] = []
        seen_pairs: set[tuple[str, str]] = set()
        queue: deque[str] = deque([canonical])
        seen_services: set[str] = {canonical}

        while queue:
            svc = queue.popleft()
            for edge in self._edges:
                if edge.effect_service != svc:
                    continue
                if not (window_start <= edge.occurred_at <= ts):
                    continue
                key = (edge.cause_id, edge.effect_id)
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                matched.append(edge)
                cs = edge.cause_service
                if cs and cs != svc and cs not in seen_services:
                    seen_services.add(cs)
                    queue.append(cs)

        self._snapshots[incident_id] = sorted(
            matched,
            key=lambda edge: edge.occurred_at,
        )

    def edges_for_incident(self, incident_id: str) -> list[CausalEdge]:
        """Return snapshotted edges sorted by ``occurred_at`` ascending."""
        edges = self._snapshots.get(incident_id, [])
        return sorted(edges, key=lambda edge: edge.occurred_at)

    def _append_edge(self, edge: CausalEdge) -> bool:
        key = (edge.cause_id, edge.effect_id)
        if key in self._edge_pairs:
            return False
        self._edge_pairs.add(key)
        self._edges.append(edge)
        return True

    def _canonical(self, service: str) -> str:
        self._identity.register(service)
        return self._identity.resolve(service)

    @staticmethod
    def _is_degrading(kind: str, payload: dict[str, Any] | None) -> bool:
        if payload is None:
            return False
        if kind == "log":
            return str(payload.get("level", "")).lower() == "error"
        if kind == "metric":
            if payload.get("degraded") is True or payload.get("anomaly") is True:
                return True
            name = str(payload.get("name", "")).lower()
            if "latency" in name or "timeout" in name:
                return True
        return False

    def _remember_anomaly(self, service: str, ts: datetime, event_id: str) -> None:
        hist = self._anomalies.setdefault(
            service,
            deque(maxlen=_MAX_ANOMALY_HISTORY),
        )
        hist.append((ts, event_id))

    def _cross_service_propagate(
        self,
        service: str,
        ts: datetime,
        event_id: str,
        kind: str,
    ) -> list[CausalEdge]:
        created: list[CausalEdge] = []
        for neighbor in self._dependency.neighbors(service):
            hist = self._anomalies.get(neighbor)
            if not hist:
                continue
            for nts, nid in hist:
                if nid == event_id:
                    continue
                if abs((ts - nts).total_seconds()) > self._propagation_window_s:
                    continue
                hops = self._dependency.shortest_hops(service, neighbor)
                if hops is None or hops < 1:
                    continue
                conf = _PROPAGATION_BASE_CONFIDENCE * (self._hop_decay**hops)
                time_damp = 1.0 - min(
                    1.0,
                    abs((ts - nts).total_seconds()) / float(self._propagation_window_s or 1),
                )
                conf = max(0.0, min(1.0, conf * (0.55 + 0.45 * time_damp)))
                if nts <= ts:
                    c_id, e_id = nid, event_id
                    c_svc, e_svc = neighbor, service
                    edge_ts = ts
                else:
                    c_id, e_id = event_id, nid
                    c_svc, e_svc = service, neighbor
                    edge_ts = max(ts, nts)
                evidence = [
                    f"propagation:{hops}",
                    f"from:{c_svc}",
                    f"to:{e_svc}",
                    f"pair:{kind}",
                ]
                edge = CausalEdge(
                    cause_id=c_id,
                    effect_id=e_id,
                    evidence=evidence,
                    confidence=conf,
                    occurred_at=edge_ts,
                    cause_service=c_svc,
                    effect_service=e_svc,
                )
                if self._append_edge(edge):
                    created.append(edge)
        return created
