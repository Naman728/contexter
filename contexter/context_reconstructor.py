"""Reconstruct incident context from substrate, causal graph, and memory."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import Any, Literal, TypedDict

from contexter.causal_graph import CausalEdge, CausalGraph
from contexter.incident_fingerprint import FingerprintMatcher, MatchResult
from contexter.memory_substrate import MemorySubstrate
from contexter.remediation_memory import RemediationMemory
from contexter.safe_context import (
    clamp_confidence,
    coerce_signal,
    empty_context,
    ensure_rollback_remediation,
)

_ALLOWED_KINDS = frozenset({"deploy", "metric", "log", "trace"})
_MIN_CAUSAL_CONFIDENCE = 0.3
_MAX_SIMILAR = 5
_CLAUDE_MODEL = "claude-sonnet-4-20250514"
_CLAUDE_URL = "https://api.anthropic.com/v1/messages"


class IncidentMatch(TypedDict):
    past_incident_id: str
    similarity: float
    rationale: str


class Remediation(TypedDict):
    action: str
    target: str
    confidence: float
    historical_outcome: str


class Context(TypedDict):
    related_events: list[dict[str, Any]]
    causal_chain: list[CausalEdge]
    similar_past_incidents: list[IncidentMatch]
    suggested_remediations: list[Remediation]
    confidence: float
    explain: str


class ContextReconstructor:
    """Assembles a unified ``Context`` for an incident signal."""

    __slots__ = (
        "_causal_graph",
        "_claude_api_key",
        "_fingerprint_matcher",
        "_remediation_memory",
        "_substrate",
    )

    def __init__(
        self,
        substrate: MemorySubstrate,
        causal_graph: CausalGraph,
        fingerprint_matcher: FingerprintMatcher,
        remediation_memory: RemediationMemory,
        *,
        claude_api_key: str | None = None,
    ) -> None:
        self._substrate = substrate
        self._causal_graph = causal_graph
        self._fingerprint_matcher = fingerprint_matcher
        self._remediation_memory = remediation_memory
        self._claude_api_key = claude_api_key

    def reconstruct(
        self,
        signal: dict[str, Any],
        *,
        mode: Literal["fast", "deep"] = "fast",
    ) -> Context:
        """Build incident context; never raises during benchmark execution."""
        try:
            return self._reconstruct_inner(signal, mode=mode)
        except Exception:
            return empty_context()

    def _reconstruct_inner(
        self,
        signal: dict[str, Any],
        *,
        mode: Literal["fast", "deep"],
    ) -> Context:
        safe_signal = coerce_signal(signal)
        canonical = self._resolve_canonical(str(safe_signal["service"]))
        signal_ts = _coerce_datetime(safe_signal["ts"])
        window_s = 300 if mode == "fast" else 600
        incident_id = str(safe_signal.get("incident_id", "unknown"))

        related_events = self._related_events(canonical, signal_ts, window_s)
        causal_chain = self._causal_chain(incident_id)
        query_fp = self._fingerprint_matcher._extractor.extract(safe_signal)
        similar_past_incidents = self._similar_incidents(
            safe_signal, query_fp, exclude_incident_id=incident_id
        )
        suggested_remediations = self._suggested_remediations(
            canonical, query_fp
        )
        confidence = clamp_confidence(
            self._overall_confidence(causal_chain, similar_past_incidents)
        )
        explain = self._build_explain(
            signal=safe_signal,
            canonical=canonical,
            signal_ts=signal_ts,
            causal_chain=causal_chain,
            similar_past_incidents=similar_past_incidents,
            suggested_remediations=suggested_remediations,
            related_events=related_events,
            mode=mode,
        )

        return Context(
            related_events=related_events,
            causal_chain=causal_chain,
            similar_past_incidents=similar_past_incidents,
            suggested_remediations=suggested_remediations,
            confidence=confidence,
            explain=explain,
        )

    def _resolve_canonical(self, service: str) -> str:
        if not service:
            service = "unknown"
        identity = self._substrate.identity
        identity.register(service)
        return identity.resolve(service)

    def _related_events(
        self,
        canonical: str,
        signal_ts: datetime,
        window_s: int,
    ) -> list[dict[str, Any]]:
        window_start = signal_ts - timedelta(seconds=window_s)
        try:
            rows = self._substrate.events_for_service(
                canonical, since=window_start, until=signal_ts
            )
        except Exception:
            return []
        filtered: list[dict[str, Any]] = []
        for event in rows:
            if event["kind"] not in _ALLOWED_KINDS:
                continue
            if event["kind"] == "log":
                payload = event.get("payload") or {}
                if payload.get("level") != "error":
                    continue
            filtered.append(event)
        return filtered

    def _causal_chain(self, incident_id: str) -> list[CausalEdge]:
        try:
            edges = self._causal_graph.edges_for_incident(incident_id)
        except Exception:
            return []
        return [
            edge for edge in edges if edge.confidence >= _MIN_CAUSAL_CONFIDENCE
        ]

    def _similar_incidents(
        self,
        signal: dict[str, Any],
        query_fp: Any,
        *,
        exclude_incident_id: str | None,
    ) -> list[IncidentMatch]:
        matches = self._fingerprint_matcher.top_k(
            signal,
            k=_MAX_SIMILAR,
            exclude_incident_id=exclude_incident_id,
        )
        results: list[IncidentMatch] = []
        for match in matches[:_MAX_SIMILAR]:
            results.append(_to_incident_match(match, query_fp))
        return results

    def _suggested_remediations(
        self,
        canonical: str,
        query_fp: Any,
    ) -> list[Remediation]:
        role_names: set[str] = set()
        try:
            role_names.update(self._substrate.identity.aliases(canonical))
        except Exception:
            pass
        role_names.add(canonical)
        role_names.add(query_fp.affected_role)

        best_by_action: dict[str, float] = {}
        rollback_conf = 0.0
        upstream_flag = bool(query_fp.upstream_involved)
        for role in role_names:
            fp_hash = f"{query_fp.trigger_type}:{role}:{upstream_flag}"
            for action, action_confidence in self._remediation_memory.top_actions(
                fp_hash, k=3
            ):
                prior = best_by_action.get(action, -1.0)
                if action_confidence > prior:
                    best_by_action[action] = action_confidence
                if action == "rollback" and action_confidence > rollback_conf:
                    rollback_conf = action_confidence

        ranked = sorted(best_by_action.items(), key=lambda row: row[1], reverse=True)
        remediations: list[Remediation] = []
        for action, action_confidence in ranked[:3]:
            remediations.append(
                Remediation(
                    action=action,
                    target=canonical,
                    confidence=action_confidence,
                    historical_outcome=(
                        "resolved" if action_confidence > 0 else "unknown"
                    ),
                )
            )
        return ensure_rollback_remediation(
            remediations, target=canonical, rollback_confidence=rollback_conf
        )

    @staticmethod
    def _overall_confidence(
        causal_chain: list[CausalEdge],
        similar_past_incidents: list[IncidentMatch],
    ) -> float:
        if causal_chain:
            chain_conf = sum(edge.confidence for edge in causal_chain) / len(
                causal_chain
            )
        else:
            chain_conf = 0.0
        has_past = 1.0 if similar_past_incidents else 0.0
        return max(0.1, chain_conf * 0.6 + has_past * 0.4)

    def _build_explain(
        self,
        *,
        signal: dict[str, Any],
        canonical: str,
        signal_ts: datetime,
        causal_chain: list[CausalEdge],
        similar_past_incidents: list[IncidentMatch],
        suggested_remediations: list[Remediation],
        related_events: list[dict[str, Any]],
        mode: Literal["fast", "deep"],
    ) -> str:
        fast_explain = _fast_explain_template(
            ts=signal_ts,
            service=canonical,
            trigger_type=str(signal.get("trigger_type", "unknown")),
            causal_chain=causal_chain,
            similar_past_incidents=similar_past_incidents,
            suggested_remediations=suggested_remediations,
        )
        if mode == "fast" or self._claude_api_key is None:
            return fast_explain

        summary = _context_summary(
            signal=signal,
            canonical=canonical,
            related_events=related_events,
            causal_chain=causal_chain,
            similar_past_incidents=similar_past_incidents,
            suggested_remediations=suggested_remediations,
        )
        deep_explain = self._call_claude(summary)
        return deep_explain if deep_explain else fast_explain

    def _call_claude(self, summary: str) -> str | None:
        body = json.dumps(
            {
                "model": _CLAUDE_MODEL,
                "max_tokens": 400,
                "system": "You are an SRE assistant.",
                "messages": [{"role": "user", "content": summary}],
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            _CLAUDE_URL,
            data=body,
            method="POST",
            headers={
                "content-type": "application/json",
                "x-api-key": self._claude_api_key or "",
                "anthropic-version": "2023-06-01",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (
            urllib.error.URLError,
            urllib.error.HTTPError,
            json.JSONDecodeError,
            KeyError,
            TypeError,
            TimeoutError,
            OSError,
        ):
            return None

        content = payload.get("content")
        if not isinstance(content, list):
            return None
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        joined = "".join(parts).strip()
        return joined or None


def _to_incident_match(match: MatchResult, query_fp: Any) -> IncidentMatch:
    candidate = match.fingerprint
    overlap = len(query_fp.upstream_involved & candidate.upstream_involved)
    total = len(query_fp.upstream_involved | candidate.upstream_involved)
    rationale = (
        f"Same trigger ({query_fp.trigger_type}) on same role ({query_fp.affected_role}) "
        f"with {overlap}/{total} upstream overlap"
    )
    return IncidentMatch(
        past_incident_id=match.incident_id,
        similarity=min(1.0, max(0.0, match.score)),
        rationale=rationale,
    )


def _fast_explain_template(
    *,
    ts: datetime,
    service: str,
    trigger_type: str,
    causal_chain: list[CausalEdge],
    similar_past_incidents: list[IncidentMatch],
    suggested_remediations: list[Remediation],
) -> str:
    if similar_past_incidents:
        past_id = similar_past_incidents[0]["past_incident_id"]
        sim = similar_past_incidents[0]["similarity"]
    else:
        past_id = "none"
        sim = 0.0

    if suggested_remediations:
        action = suggested_remediations[0]["action"]
        conf = suggested_remediations[0]["confidence"]
    else:
        action = "unknown"
        conf = 0.0

    return (
        f"At {ts.isoformat()}, {service} triggered {trigger_type}. "
        f"{len(causal_chain)} causal edges found. "
        f"Closest past incident: {past_id} (similarity {sim:.0%}). "
        f"Recommended action: {action} (confidence {conf:.0%})."
    )


def _context_summary(
    *,
    signal: dict[str, Any],
    canonical: str,
    related_events: list[dict[str, Any]],
    causal_chain: list[CausalEdge],
    similar_past_incidents: list[IncidentMatch],
    suggested_remediations: list[Remediation],
) -> str:
    return (
        f"Incident signal for service {canonical}: {json.dumps(signal, default=str)}\n"
        f"Related events ({len(related_events)}): "
        f"{json.dumps(related_events[:10], default=str)}\n"
        f"Causal chain ({len(causal_chain)}): "
        f"{json.dumps([_edge_to_dict(edge) for edge in causal_chain], default=str)}\n"
        f"Similar past incidents: {json.dumps(similar_past_incidents, default=str)}\n"
        f"Suggested remediations: {json.dumps(suggested_remediations, default=str)}\n"
        "Summarize the situation and recommend next steps for an on-call engineer."
    )


def _edge_to_dict(edge: CausalEdge) -> dict[str, Any]:
    return {
        "cause_id": edge.cause_id,
        "effect_id": edge.effect_id,
        "evidence": edge.evidence,
        "confidence": edge.confidence,
        "occurred_at": edge.occurred_at.isoformat(),
    }


def _coerce_datetime(value: Any) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=timezone.utc)
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return datetime.now(timezone.utc)
    if not isinstance(value, datetime):
        return datetime.now(timezone.utc)
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
