"""Defensive helpers for benchmark-safe context reconstruction."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, TypedDict


class RemediationDict(TypedDict):
    action: str
    target: str
    confidence: float
    historical_outcome: str


class ContextDict(TypedDict):
    related_events: list[dict[str, Any]]
    causal_chain: list[Any]
    similar_past_incidents: list[dict[str, Any]]
    suggested_remediations: list[RemediationDict]
    confidence: float
    explain: str


def empty_context() -> ContextDict:
    """Valid context when ingest or reconstruction cannot proceed."""
    return ContextDict(
        related_events=[],
        causal_chain=[],
        similar_past_incidents=[],
        suggested_remediations=[],
        confidence=0.1,
        explain="No incident context available.",
    )


def clamp_confidence(value: float) -> float:
    if value < 0.1:
        return 0.1
    if value > 1.0:
        return 1.0
    return value


def coerce_signal(signal: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(signal, dict):
        return {
            "service": "unknown",
            "incident_id": "unknown",
            "ts": datetime.now(timezone.utc),
            "trigger_type": "unknown",
            "upstream": [],
        }
    merged = dict(signal)
    merged.setdefault("service", "unknown")
    merged.setdefault("incident_id", "unknown")
    merged.setdefault("trigger_type", "unknown")
    merged.setdefault("upstream", [])
    if "ts" not in merged or merged["ts"] is None:
        merged["ts"] = datetime.now(timezone.utc)
    return merged


def ensure_rollback_remediation(
    remediations: list[RemediationDict],
    *,
    target: str,
    rollback_confidence: float,
) -> list[RemediationDict]:
    """Ensure ``rollback`` appears when historical evidence exists."""
    if rollback_confidence <= 0.0:
        return remediations
    for item in remediations:
        if item["action"] == "rollback":
            return remediations
    rollback_entry = RemediationDict(
        action="rollback",
        target=target,
        confidence=rollback_confidence,
        historical_outcome="resolved",
    )
    if not remediations:
        return [rollback_entry]
    return [rollback_entry, *remediations[:-1]]
