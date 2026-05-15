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
        # trigger + role match (0.7) + jaccard 1/3 * 0.3
        assert structural_similarity(left, right) == pytest.approx(0.7 + 0.1)

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
