"""Historical co-involvement of services on incident signals (topology continuity)."""

from __future__ import annotations

from collections.abc import Iterable

from contexter.identity_tracker import IdentityTracker


class IncidentNeighborhoodMemory:
    """Undirected co-occurrence edges between canonical services from past incidents."""

    __slots__ = ("_identity", "_edges", "_peers")

    def __init__(self, identity: IdentityTracker) -> None:
        self._identity = identity
        self._edges: set[tuple[str, str]] = set()
        self._peers: dict[str, set[str]] = {}

    def record_incident_affiliation(
        self,
        primary_service: str,
        raw_upstream: Iterable[str],
    ) -> None:
        """Record that ``primary_service`` appeared with these upstream names on one signal."""
        if not primary_service:
            return
        self._identity.register(primary_service)
        root = self._identity.resolve(primary_service)
        group: set[str] = {root}
        for raw in raw_upstream:
            name = str(raw).strip()
            if not name:
                continue
            self._identity.register(name)
            group.add(self._identity.resolve(name))
        roots = sorted(group)
        for i, a in enumerate(roots):
            for b in roots[i + 1 :]:
                key = (a, b) if a < b else (b, a)
                self._edges.add(key)
                self._peers.setdefault(a, set()).add(b)
                self._peers.setdefault(b, set()).add(a)

    def pair_seen(self, a: str, b: str) -> bool:
        """True if two distinct canonical services co-occurred on some historical incident."""
        if not a or not b:
            return False
        self._identity.register(a)
        self._identity.register(b)
        ra = self._identity.resolve(a)
        rb = self._identity.resolve(b)
        if ra == rb:
            return False
        key = (ra, rb) if ra < rb else (rb, ra)
        return key in self._edges

    def peers(self, canonical: str) -> frozenset[str]:
        """Historical co-involvement peers of ``canonical`` (canonical names, excluding self)."""
        if not canonical:
            return frozenset()
        self._identity.register(canonical)
        r = self._identity.resolve(canonical)
        return frozenset(self._peers.get(r, ()))

    def peer_jaccard(self, a: str, b: str) -> float:
        """Jaccard overlap of historical peer sets (canonicalized)."""
        self._identity.register(a)
        self._identity.register(b)
        ra = self._identity.resolve(a)
        rb = self._identity.resolve(b)
        if ra == rb:
            return 0.0
        pa = self._peers.get(ra, set()) | {ra}
        pb = self._peers.get(rb, set()) | {rb}
        inter = len(pa & pb)
        union = len(pa | pb)
        if union == 0:
            return 0.0
        return inter / union
