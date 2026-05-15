"""Tests for incident fingerprint extraction and matching."""

from __future__ import annotations

import pytest

from contexter.events import Event
from contexter.identity_tracker import IdentityTracker
from contexter.incident_fingerprint import (
    FingerprintExtractor,
    FingerprintMatcher,
    IncidentFingerprint,
    MatchResult,
    jaccard,
    normalize_trigger,
    partial_upstream_overlap,
    structural_similarity,
)


class TestExtraction:
    def test_topology_independent_after_drift(self) -> None:
        identity = IdentityTracker()
        identity.union("api-v1", "api-v2")
        extractor = FingerprintExtractor(identity)

        baseline = extractor.extract(
            {
                "trigger_type": "error_rate",
                "service": "api-v1",
                "upstream": ["auth-v1", "db-v1"],
            }
        )
        after_rename = extractor.extract(
            {
                "trigger_type": "error_rate",
                "service": "api-v2",
                "upstream": ["auth-v1", "db-v1"],
            }
        )
        assert baseline == after_rename

    def test_explicit_roles_ignore_instance_names(self) -> None:
        extractor = FingerprintExtractor()
        fp = extractor.extract(
            {
                "trigger_type": "latency_spike",
                "affected_role": "gateway",
                "upstream_roles": ["auth", "catalog"],
            }
        )
        assert fp == IncidentFingerprint(
            "latency_spike",
            "gateway",
            frozenset({"auth", "catalog"}),
            "gateway",
        )

    def test_extract_from_event(self) -> None:
        identity = IdentityTracker()
        extractor = FingerprintExtractor(identity)
        event = Event(
            kind="incident",
            service="worker-7",
            payload={
                "trigger_type": "health_check_failed",
                "upstream": ["queue", "cache"],
            },
        )
        fp = extractor.extract_event(event)
        assert fp.trigger_type == "health_check_failed"
        assert fp.affected_role == "worker-7"

    def test_upstream_excludes_affected_role(self) -> None:
        extractor = FingerprintExtractor()
        fp = extractor.extract(
            {
                "trigger_type": "cascade",
                "affected_role": "api",
                "upstream_roles": ["api", "db"],
            }
        )
        assert fp.upstream_involved == frozenset({"db"})


class TestSimilarity:
    def test_identical_fingerprints_score_one(self) -> None:
        fp = IncidentFingerprint("oom", "worker", frozenset({"db"}))
        assert structural_similarity(fp, fp) == pytest.approx(1.0)

    def test_partial_upstream_overlap(self) -> None:
        left = IncidentFingerprint("error_rate", "api", frozenset({"auth", "db"}))
        right = IncidentFingerprint("error_rate", "api", frozenset({"auth", "cache"}))
        # max(Jaccard, partial) boosts upstream vs strict Jaccard-only scoring
        assert structural_similarity(left, right) == pytest.approx(0.85)

    def test_jaccard_empty_sets(self) -> None:
        assert jaccard(frozenset(), frozenset()) == 1.0


class TestTopK:
    def test_top_k_ordered_by_score(self) -> None:
        matcher = FingerprintMatcher()
        matcher.index("a", IncidentFingerprint("error_rate", "api", frozenset({"db"})))
        matcher.index("b", IncidentFingerprint("error_rate", "api", frozenset({"cache"})))
        matcher.index("c", IncidentFingerprint("latency", "api", frozenset({"db"})))

        query = IncidentFingerprint("error_rate", "api", frozenset({"db"}))
        results = matcher.top_k(query, k=2)

        assert len(results) == 2
        assert results[0].incident_id == "a"
        assert results[0].score == pytest.approx(1.0)
        assert results[1].incident_id == "b"
        assert results[0].score >= results[1].score

    def test_min_score_filters_weak_matches(self) -> None:
        matcher = FingerprintMatcher()
        matcher.index("a", IncidentFingerprint("oom", "worker", frozenset()))
        matcher.index("b", IncidentFingerprint("error_rate", "api", frozenset({"db"})))

        query = IncidentFingerprint("error_rate", "api", frozenset({"db"}))
        results = matcher.top_k(query, k=5, min_score=0.9)
        assert [r.incident_id for r in results] == ["b"]

    def test_index_incident_uses_extractor(self) -> None:
        identity = IdentityTracker()
        identity.union("svc-old", "svc-new")
        matcher = FingerprintMatcher(FingerprintExtractor(identity))

        matcher.index_incident(
            "past",
            {"trigger_type": "timeout", "service": "svc-old", "upstream": []},
        )
        matches = matcher.top_k(
            {"trigger_type": "timeout", "service": "svc-new", "upstream": []},
            k=1,
        )
        assert matches[0].incident_id == "past"
        assert matches[0].score == pytest.approx(1.0)

    def test_best_match_none_when_empty(self) -> None:
        matcher = FingerprintMatcher()
        assert matcher.best_match(
            IncidentFingerprint("x", "y", frozenset())
        ) is None


class TestTriggerNormalization:
    def test_latency_family_groups_p99_and_p95(self) -> None:
        assert normalize_trigger("latency_p99_ms") == normalize_trigger("latency_p95_ms")
        assert normalize_trigger("latency_p99_ms") == "latency"

    def test_error_family_groups_rate_and_timeout(self) -> None:
        assert normalize_trigger("5xx_rate") == normalize_trigger("error_rate")
        assert normalize_trigger("timeout_rate") == "error"


class TestRecallEnhancements:
    def test_partial_upstream_overlap_exceeds_jaccard_on_subset(self) -> None:
        a = frozenset({"auth", "db"})
        b = frozenset({"auth", "db", "cache", "cdn", "edge"})
        assert partial_upstream_overlap(a, b) > jaccard(a, b)

    def test_debug_returns_score_breakdown(self) -> None:
        left = IncidentFingerprint("error_rate", "api", frozenset({"auth"}))
        right = IncidentFingerprint("timeout_rate", "api", frozenset({"auth"}))
        bd = structural_similarity(left, right, debug=True)
        assert isinstance(bd, dict)
        assert set(bd.keys()) == {
            "trigger_score",
            "upstream_score",
            "role_score",
            "temporal_score",
            "alias_score",
            "role_family_score",
            "final",
        }
        assert bd["trigger_score"] == 1.0
        assert bd["final"] == pytest.approx(1.0)

    def test_temporal_boost_same_deploy_window(self) -> None:
        left = IncidentFingerprint("error_rate", "api", frozenset({"db"}))
        right = IncidentFingerprint("error_rate", "api", frozenset({"db"}))
        qctx = {"deploy_window": 42}
        cctx = {"deploy_window": 42}
        bd = structural_similarity(
            left, right, query_context=qctx, candidate_context=cctx, debug=True
        )
        assert bd["temporal_score"] == pytest.approx(0.15)
        assert bd["final"] == pytest.approx(1.0)

    def test_top_k_debug_includes_breakdown(self) -> None:
        matcher = FingerprintMatcher()
        matcher.index("p", IncidentFingerprint("error_rate", "api", frozenset({"db"})))
        rows = matcher.top_k(
            IncidentFingerprint("timeout_rate", "api", frozenset({"db"})),
            k=1,
            debug=True,
        )
        assert rows[0].score_breakdown is not None
        assert rows[0].score_breakdown["trigger_score"] == 1.0

    def test_noisy_upstream_still_retrieves_family(self) -> None:
        matcher = FingerprintMatcher()
        matcher.index_incident(
            "past-family",
            {
                "trigger_type": "error_rate",
                "service": "checkout-api",
                "upstream": ["auth", "payments"],
            },
        )
        matcher.index_incident(
            "distractor",
            {
                "trigger_type": "error_rate",
                "service": "unrelated-svc",
                "upstream": ["cache", "cdn"],
            },
        )
        matches = matcher.top_k(
            {
                "trigger_type": "5xx_rate",
                "service": "checkout-api",
                "upstream": ["auth", "payments", "telemetry", "feature-flags", "cdn"],
            },
            k=2,
        )
        assert matches[0].incident_id == "past-family"
        assert matches[0].score >= matches[1].score

    def test_renamed_services_score_high_via_identity(self) -> None:
        identity = IdentityTracker()
        identity.union("orders-api", "fulfillment-api")
        matcher = FingerprintMatcher(FingerprintExtractor(identity))
        matcher.index_incident(
            "past",
            {
                "trigger_type": "latency_p99_ms",
                "service": "orders-api",
                "upstream": ["db-primary"],
            },
        )
        matches = matcher.top_k(
            {
                "trigger_type": "latency_p95_ms",
                "service": "fulfillment-api",
                "upstream": ["db-primary"],
            },
            k=1,
        )
        assert matches[0].incident_id == "past"
        assert matches[0].score >= 0.99

    def test_role_family_boost_when_role_strings_differ(self) -> None:
        left = IncidentFingerprint(
            "error_rate",
            "payments-api",
            frozenset(),
            "payments-api",
        )
        right = IncidentFingerprint(
            "error_rate",
            "payments-svc",
            frozenset(),
            "payments-svc",
        )
        bd = structural_similarity(left, right, debug=True)
        assert bd["role_score"] == 0.0
        assert bd["role_family_score"] == pytest.approx(0.10)
        assert bd["final"] > 0.4

    def test_top_k_mean_latency_stays_low(self) -> None:
        import time

        matcher = FingerprintMatcher()
        for i in range(400):
            matcher.index(
                str(i),
                IncidentFingerprint(
                    "error_rate",
                    "api",
                    frozenset({str(j % 11) for j in range(i % 12)}),
                ),
            )
        q = {
            "trigger_type": "timeout_rate",
            "service": "api",
            "upstream": ["1", "2", "3"],
        }
        t0 = time.perf_counter()
        for _ in range(40):
            matcher.top_k(q, k=5)
        ms = (time.perf_counter() - t0) / 40 * 1000.0
        assert ms < 8.0, f"mean top_k latency {ms:.2f}ms exceeds budget"
