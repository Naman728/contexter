"""Lightweight service call graph learned from trace span relationships."""

from __future__ import annotations

from collections import deque

from typing import Any

from contexter.identity_tracker import IdentityTracker


def parse_trace_call_edges(
    payload: dict[str, Any] | None,
    identity: IdentityTracker,
) -> list[tuple[str, str]]:
    """Return ``(caller, callee)`` pairs: parent span service invokes child span service."""
    if not payload:
        return []
    spans = payload.get("spans")
    if not isinstance(spans, list):
        return []
    by_id: dict[str, dict[str, object]] = {}
    for raw in spans:
        if not isinstance(raw, dict):
            continue
        sid = raw.get("span_id")
        if sid is None:
            continue
        by_id[str(sid)] = raw
    edges: list[tuple[str, str]] = []
    for raw in spans:
        if not isinstance(raw, dict):
            continue
        child_svc = raw.get("service")
        if child_svc is None:
            continue
        pid = raw.get("parent_span_id")
        if pid is None or pid == "":
            continue
        parent = by_id.get(str(pid))
        if not parent:
            continue
        psvc = parent.get("service")
        if psvc is None:
            continue
        caller = str(psvc).strip()
        callee = str(child_svc).strip()
        if not caller or not callee or caller == callee:
            continue
        identity.register(caller)
        identity.register(callee)
        edges.append(
            (identity.resolve(caller), identity.resolve(callee)),
        )
    return edges


class ServiceDependencyGraph:
    """Directed edges *caller -> callee* (caller invokes downstream service)."""

    __slots__ = ("_callees", "_callers", "_identity")

    def __init__(self, identity: IdentityTracker) -> None:
        self._identity = identity
        self._callees: dict[str, set[str]] = {}
        self._callers: dict[str, set[str]] = {}

    def add_call_edge(self, caller: str, callee: str) -> None:
        """Record that *caller* invoked *callee* (canonical names)."""
        if not caller or not callee or caller == callee:
            return
        self._identity.register(caller)
        self._identity.register(callee)
        c = self._identity.resolve(caller)
        d = self._identity.resolve(callee)
        if c == d:
            return
        self._callees.setdefault(c, set()).add(d)
        self._callers.setdefault(d, set()).add(c)

    def neighbors(self, service: str) -> set[str]:
        """Services one hop away in the undirected sense (callers ∪ callees)."""
        self._identity.register(service)
        s = self._identity.resolve(service)
        out: set[str] = set()
        out.update(self._callees.get(s, ()))
        out.update(self._callers.get(s, ()))
        return out

    def shortest_hops(self, a: str, b: str) -> int | None:
        """Shortest undirected hop count between services; ``None`` if unreachable."""
        self._identity.register(a)
        self._identity.register(b)
        sa = self._identity.resolve(a)
        sb = self._identity.resolve(b)
        if sa == sb:
            return 0
        q: deque[tuple[str, int]] = deque([(sa, 0)])
        seen: set[str] = {sa}
        while q:
            cur, depth = q.popleft()
            if depth > 32:
                break
            for nxt in self.neighbors(cur):
                if nxt in seen:
                    continue
                if nxt == sb:
                    return depth + 1
                seen.add(nxt)
                q.append((nxt, depth + 1))
        return None
