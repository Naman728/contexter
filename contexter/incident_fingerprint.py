"""Topology-independent incident fingerprints and structural matching."""

from __future__ import annotations

import heapq
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from contexter.events import Event
from contexter.identity_tracker import IdentityTracker
from contexter.roles import family_aware_role

RoleResolver = Callable[[str], str]

_DEFAULT_WEIGHTS: tuple[float, float, float] = (0.35, 0.35, 0.30)


@dataclass(frozen=True, slots=True)
class IncidentFingerprint:
    """Structural incident signature independent of concrete topology names."""

    trigger_type: str
    affected_role: str
    upstream_involved: frozenset[str]

    def as_tuple(self) -> tuple[str, str, frozenset[str]]:
        return (self.trigger_type, self.affected_role, self.upstream_involved)


@dataclass(frozen=True, slots=True)
class MatchResult:
    incident_id: str
    fingerprint: IncidentFingerprint
    score: float


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def structural_similarity(
    left: IncidentFingerprint,
    right: IncidentFingerprint,
    *,
    weights: Sequence[float] = _DEFAULT_WEIGHTS,
) -> float:
    """Score in ``[0, 1]`` from trigger, affected role, and upstream overlap."""
    if len(weights) != 3:
        raise ValueError("weights must have exactly three components")
    w_trigger, w_role, w_upstream = weights
    total = w_trigger + w_role + w_upstream
    if total <= 0:
        raise ValueError("weights must sum to a positive value")
    w_trigger, w_role, w_upstream = (
        w_trigger / total,
        w_role / total,
        w_upstream / total,
    )

    trigger_score = 1.0 if left.trigger_type == right.trigger_type else 0.0
    role_score = 1.0 if left.affected_role == right.affected_role else 0.0
    upstream_score = jaccard(left.upstream_involved, right.upstream_involved)

    return (
        w_trigger * trigger_score
        + w_role * role_score
        + w_upstream * upstream_score
    )


class FingerprintExtractor:
    """Build fingerprints by resolving services through ``IdentityTracker``."""

    __slots__ = ("_identity", "_role_resolver")

    def __init__(
        self,
        identity: IdentityTracker | None = None,
        role_resolver: RoleResolver | None = None,
    ) -> None:
        self._identity = identity if identity is not None else IdentityTracker()
        self._role_resolver = role_resolver or family_aware_role

    @property
    def identity(self) -> IdentityTracker:
        return self._identity

    def extract(self, incident: Mapping[str, Any] | Event) -> IncidentFingerprint:
        if isinstance(incident, Event):
            return self.extract_event(incident)
        return self._extract_mapping(incident)

    def extract_event(self, event: Event) -> IncidentFingerprint:
        payload = event.payload or {}
        merged: dict[str, Any] = {
            "trigger_type": payload.get("trigger_type", event.kind),
            "service": event.service,
            "affected_role": payload.get("affected_role"),
            "upstream": payload.get("upstream", payload.get("upstream_services", ())),
            "upstream_roles": payload.get("upstream_roles"),
        }
        return self._extract_mapping(merged)

    def _extract_mapping(self, incident: Mapping[str, Any]) -> IncidentFingerprint:
        trigger_type = str(incident.get("trigger_type", "unknown"))

        affected_role = incident.get("affected_role")
        if affected_role is not None:
            role = self._role_resolver(str(affected_role))
        else:
            service = incident.get("service", incident.get("affected_service", "unknown"))
            role = self._role_for_service(str(service))

        upstream_roles = incident.get("upstream_roles")
        if upstream_roles is not None:
            upstream = _normalize_role_set(upstream_roles, self._role_resolver)
        else:
            raw_upstream = incident.get(
                "upstream",
                incident.get("upstream_services", incident.get("upstream_involved", ())),
            )
            if not isinstance(raw_upstream, (list, tuple, set, frozenset)):
                raw_upstream = ()
            upstream = frozenset(
                self._role_for_service(str(name)) for name in raw_upstream
            )

        upstream = frozenset(r for r in upstream if r != role)
        return IncidentFingerprint(trigger_type, role, upstream)

    def _role_for_service(self, service: str) -> str:
        if not service:
            service = "unknown"
        self._identity.register(service)
        canonical = self._identity.resolve(service)
        return self._role_resolver(canonical)


class FingerprintMatcher:
    """In-memory corpus with structural similarity and top-k retrieval."""

    __slots__ = ("_by_trigger", "_corpus", "_extractor", "_weights")

    def __init__(
        self,
        extractor: FingerprintExtractor | None = None,
        *,
        weights: Sequence[float] = _DEFAULT_WEIGHTS,
    ) -> None:
        self._extractor = extractor or FingerprintExtractor()
        self._weights = tuple(weights)
        self._corpus: list[tuple[str, IncidentFingerprint]] = []
        self._by_trigger: dict[str, list[int]] = defaultdict(list)

    def __len__(self) -> int:
        return len(self._corpus)

    def index(
        self,
        incident_id: str,
        fingerprint: IncidentFingerprint,
    ) -> None:
        idx = len(self._corpus)
        self._corpus.append((incident_id, fingerprint))
        self._by_trigger[fingerprint.trigger_type].append(idx)

    def index_incident(
        self,
        incident_id: str,
        incident: Mapping[str, Any] | Event,
    ) -> IncidentFingerprint:
        fingerprint = self._extractor.extract(incident)
        self.index(incident_id, fingerprint)
        return fingerprint

    def top_k(
        self,
        query: IncidentFingerprint | Mapping[str, Any] | Event,
        k: int = 5,
        *,
        min_score: float = 0.0,
        exclude_incident_id: str | None = None,
    ) -> list[MatchResult]:
        if k < 1:
            return []
        try:
            fingerprint = (
                query
                if isinstance(query, IncidentFingerprint)
                else self._extractor.extract(query)
            )
        except (KeyError, TypeError, ValueError):
            return []
        if not self._corpus:
            return []

        candidate_indices = self._by_trigger.get(fingerprint.trigger_type, [])
        if len(candidate_indices) < len(self._corpus):
            indices = candidate_indices
        else:
            indices = range(len(self._corpus))

        scored: list[tuple[float, str, IncidentFingerprint]] = []
        for idx in indices:
            incident_id, candidate = self._corpus[idx]
            if exclude_incident_id is not None and incident_id == exclude_incident_id:
                continue
            score = structural_similarity(
                fingerprint, candidate, weights=self._weights
            )
            if score >= min_score:
                scored.append((score, incident_id, candidate))

        if len(scored) < k:
            for idx, (incident_id, candidate) in enumerate(self._corpus):
                if idx in candidate_indices:
                    continue
                if exclude_incident_id is not None and incident_id == exclude_incident_id:
                    continue
                score = structural_similarity(
                    fingerprint, candidate, weights=self._weights
                )
                if score >= min_score:
                    scored.append((score, incident_id, candidate))

        best = heapq.nlargest(k, scored, key=lambda row: row[0])
        return [
            MatchResult(incident_id=incident_id, fingerprint=fp, score=score)
            for score, incident_id, fp in best
        ]

    def best_match(
        self,
        query: IncidentFingerprint | Mapping[str, Any] | Event,
        *,
        min_score: float = 0.0,
    ) -> MatchResult | None:
        results = self.top_k(query, k=1, min_score=min_score)
        return results[0] if results else None

    def clear(self) -> None:
        self._corpus.clear()
        self._by_trigger.clear()


def _normalize_role_set(
    values: Iterable[Any],
    role_resolver: RoleResolver,
) -> frozenset[str]:
    return frozenset(role_resolver(str(value)) for value in values)
