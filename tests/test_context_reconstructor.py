"""Tests for ContextReconstructor."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from contexter.causal_graph import CausalGraph
from contexter.context_reconstructor import Context, ContextReconstructor
from contexter.identity_tracker import IdentityTracker
from contexter.incident_fingerprint import FingerprintExtractor, FingerprintMatcher
from contexter.memory_substrate import MemorySubstrate
from contexter.remediation_memory import RemediationMemory

UTC = timezone.utc
T0 = datetime(2026, 5, 15, 14, 0, 0, tzinfo=UTC)

_REQUIRED_KEYS = frozenset(
    {
        "related_events",
        "causal_chain",
        "similar_past_incidents",
        "suggested_remediations",
        "confidence",
        "explain",
    }
)


def _build_reconstructor(
    identity: IdentityTracker | None = None,
    *,
    claude_api_key: str | None = None,
) -> tuple[ContextReconstructor, MemorySubstrate, CausalGraph, FingerprintMatcher]:
    identity = identity or IdentityTracker()
    substrate = MemorySubstrate(batch_size=1, identity=identity)
    causal = CausalGraph(identity)
    matcher = FingerprintMatcher(FingerprintExtractor(identity))
    remediation = RemediationMemory()
    reconstructor = ContextReconstructor(
        substrate,
        causal,
        matcher,
        remediation,
        claude_api_key=claude_api_key,
    )
    return reconstructor, substrate, causal, matcher


def _base_signal(**overrides: object) -> dict:
    signal = {
        "incident_id": "inc-current",
        "service": "api",
        "ts": T0,
        "trigger_type": "error_rate",
        "upstream": ["auth", "db"],
    }
    signal.update(overrides)
    return signal


class TestFastMode:
    def test_returns_all_required_keys(self) -> None:
        reconstructor, substrate, causal, _matcher = _build_reconstructor()
        substrate.ingest(
            {
                "kind": "metric",
                "service": "api",
                "occurred_at": T0 - timedelta(seconds=30),
                "payload": {"cpu": 0.9},
            }
        )
        causal.record_deploy("d1", "api", T0 - timedelta(seconds=60))
        causal.record_effect("e1", "api", T0 - timedelta(seconds=30), kind="log")
        causal.snapshot_incident("inc-current", T0, "api")

        context = reconstructor.reconstruct(_base_signal(), mode="fast")
        assert set(context.keys()) == _REQUIRED_KEYS

    def test_remediation_target_is_canonical_not_alias(self) -> None:
        identity = IdentityTracker()
        identity.union("api-old", "api-new")
        reconstructor, _substrate, _causal, matcher = _build_reconstructor(identity)
        remediation = reconstructor._remediation_memory

        fp = matcher._extractor.extract(
            {
                "trigger_type": "error_rate",
                "service": "api-old",
                "upstream": [],
            }
        )
        fp_hash = f"{fp.trigger_type}:{fp.affected_role}:{bool(fp.upstream_involved)}"
        remediation.record(fp_hash, "restart", outcome="resolved")

        context = reconstructor.reconstruct(
            _base_signal(service="api-old", incident_id="inc-1", upstream=[]),
            mode="fast",
        )
        assert context["suggested_remediations"]
        assert context["suggested_remediations"][0]["target"] == "api-new"

    def test_confidence_never_below_point_one(self) -> None:
        reconstructor, _, _, _ = _build_reconstructor()
        context = reconstructor.reconstruct(_base_signal(), mode="fast")
        assert context["confidence"] >= 0.1

    def test_explain_non_empty(self) -> None:
        reconstructor, _, _, _ = _build_reconstructor()
        context = reconstructor.reconstruct(_base_signal(), mode="fast")
        assert context["explain"]

    def test_events_outside_window_excluded(self) -> None:
        reconstructor, substrate, causal, _ = _build_reconstructor()
        substrate.ingest(
            {
                "kind": "metric",
                "service": "api",
                "occurred_at": T0 - timedelta(seconds=60),
                "payload": {"cpu": 0.5},
            }
        )
        substrate.ingest(
            {
                "kind": "metric",
                "service": "api",
                "occurred_at": T0 - timedelta(seconds=400),
                "payload": {"cpu": 0.2},
            }
        )
        causal.snapshot_incident("inc-current", T0, "api")

        context = reconstructor.reconstruct(_base_signal(), mode="fast")
        assert len(context["related_events"]) == 1
        assert context["related_events"][0]["payload"]["cpu"] == 0.5

    def test_non_error_logs_excluded(self) -> None:
        reconstructor, substrate, _, _ = _build_reconstructor()
        substrate.ingest(
            {
                "kind": "log",
                "service": "api",
                "occurred_at": T0 - timedelta(seconds=10),
                "payload": {"level": "info", "msg": "ok"},
            }
        )
        substrate.ingest(
            {
                "kind": "log",
                "service": "api",
                "occurred_at": T0 - timedelta(seconds=20),
                "payload": {"level": "error", "msg": "fail"},
            }
        )
        context = reconstructor.reconstruct(_base_signal(), mode="fast")
        assert len(context["related_events"]) == 1
        assert context["related_events"][0]["payload"]["level"] == "error"


class TestRenameAndCausal:
    def test_similar_past_incident_after_drift(self) -> None:
        identity = IdentityTracker()
        reconstructor, substrate, causal, matcher = _build_reconstructor(identity)

        past_signal = {
            "trigger_type": "error_rate",
            "service": "payments-svc",
            "upstream": ["auth", "db"],
        }
        matcher.index_incident("inc-past", past_signal)

        substrate.ingest(
            {
                "kind": "identity.drift",
                "service": "payments-svc",
                "payload": {"old": "payments-svc", "new": "billing-svc"},
            }
        )

        causal.record_deploy("d1", "payments-svc", T0 - timedelta(seconds=90))
        causal.record_effect(
            "e1", "payments-svc", T0 - timedelta(seconds=60), kind="log"
        )
        causal.snapshot_incident("inc-current", T0, "billing-svc")

        context = reconstructor.reconstruct(
            _base_signal(
                service="billing-svc",
                incident_id="inc-current",
                trigger_type="error_rate",
                upstream=["auth", "db"],
            ),
            mode="fast",
        )

        assert context["similar_past_incidents"]
        match = context["similar_past_incidents"][0]
        assert match["past_incident_id"] == "inc-past"
        assert match["similarity"] >= 0.6

    def test_causal_chain_sorted_ascending(self) -> None:
        reconstructor, _, causal, _ = _build_reconstructor()
        causal.record_deploy("d1", "api", T0 - timedelta(seconds=100))
        causal.record_effect("e2", "api", T0 - timedelta(seconds=40), kind="log")
        causal.record_deploy("d2", "api", T0 - timedelta(seconds=80))
        causal.record_effect("e1", "api", T0 - timedelta(seconds=70), kind="log")
        causal.snapshot_incident("inc-current", T0, "api")

        context = reconstructor.reconstruct(_base_signal(), mode="fast")
        chain = context["causal_chain"]
        assert len(chain) >= 2
        for earlier, later in zip(chain, chain[1:]):
            assert earlier.occurred_at <= later.occurred_at

    def test_low_confidence_causal_edges_excluded(self) -> None:
        reconstructor, _, causal, _ = _build_reconstructor()
        causal.record_deploy("d1", "api", T0 - timedelta(seconds=119))
        causal.record_effect("e1", "api", T0, kind="log")
        causal.snapshot_incident("inc-current", T0, "api")

        context = reconstructor.reconstruct(_base_signal(), mode="fast")
        assert context["causal_chain"] == []


class TestDeepMode:
    def test_deep_mode_without_key_falls_back(self) -> None:
        reconstructor, _, _, _ = _build_reconstructor(claude_api_key=None)
        context = reconstructor.reconstruct(_base_signal(), mode="deep")
        assert context["explain"]
        assert "causal edges found" in context["explain"]
