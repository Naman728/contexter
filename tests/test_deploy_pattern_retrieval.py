"""Deploy-centric ordered pattern sequences for incident retrieval."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from contexter.engine import Engine
from contexter.incident_fingerprint import (
    FingerprintMatcher,
    RerankContext,
    RetrievalFeatures,
    extract_deploy_pattern_sequence,
    sequence_similarity,
)
from contexter.remediation_memory import RemediationMemory

T0 = datetime(2026, 8, 1, 12, 0, 0, tzinfo=timezone.utc)


class TestSequenceSimilarity:
    def test_prefix_chain_strongly_matches_extended_chain(self) -> None:
        a = ("deploy", "latency", "error")
        b = ("deploy", "latency", "error", "rollback")
        sim = sequence_similarity(a, b)
        assert sim > 0.84
        assert sim < 1.0

    def test_identical_full_rollback_chains_score_one(self) -> None:
        a = ("deploy", "latency", "error", "rollback")
        assert sequence_similarity(a, a) == pytest.approx(1.0)

    def test_unrelated_sequences_score_lower(self) -> None:
        a = ("deploy", "latency", "error")
        b = ("deploy", "remediation", "deploy")
        assert sequence_similarity(a, b) < 0.75

    def test_both_empty_is_one(self) -> None:
        assert sequence_similarity((), ()) == pytest.approx(1.0)

    def test_one_empty_is_zero(self) -> None:
        assert sequence_similarity(("deploy",), ()) == pytest.approx(0.0)


class TestExtractDeployPatternSequence:
    def test_extracts_deploy_latency_error_rollback_order(self) -> None:
        with Engine(batch_size=1) as engine:
            base = T0 - timedelta(minutes=5)
            engine.ingest_one(
                {
                    "kind": "deploy",
                    "service": "api",
                    "occurred_at": base,
                }
            )
            engine.ingest_one(
                {
                    "kind": "metric",
                    "service": "api",
                    "occurred_at": base + timedelta(seconds=30),
                    "payload": {"name": "p99_latency_ms", "degraded": True},
                }
            )
            engine.ingest_one(
                {
                    "kind": "log",
                    "service": "api",
                    "occurred_at": base + timedelta(seconds=60),
                    "payload": {"level": "error", "msg": "boom"},
                }
            )
            engine.ingest_one(
                {
                    "kind": "remediation",
                    "service": "api",
                    "occurred_at": base + timedelta(seconds=90),
                    "payload": {
                        "incident_id": "x",
                        "action": "rollback",
                        "outcome": "resolved",
                    },
                }
            )
            seq = extract_deploy_pattern_sequence(
                engine._substrate,
                "api",
                base + timedelta(seconds=120),
            )
        assert seq == ("deploy", "latency", "error", "rollback")

    def test_collapses_consecutive_duplicate_tokens(self) -> None:
        with Engine(batch_size=1) as engine:
            t = T0
            engine.ingest_one({"kind": "deploy", "service": "svc", "occurred_at": t})
            engine.ingest_one(
                {
                    "kind": "metric",
                    "service": "svc",
                    "occurred_at": t + timedelta(seconds=1),
                    "payload": {"name": "latency_ms", "degraded": True},
                }
            )
            engine.ingest_one(
                {
                    "kind": "metric",
                    "service": "svc",
                    "occurred_at": t + timedelta(seconds=2),
                    "payload": {"name": "latency_ms", "degraded": True},
                }
            )
            seq = extract_deploy_pattern_sequence(
                engine._substrate, "svc", t + timedelta(seconds=10)
            )
        assert seq.count("latency") == 1


class TestRollbackChainsCluster:
    def test_matching_rollback_patterns_rank_above_mismatched(self) -> None:
        matcher = FingerprintMatcher()
        mem = RemediationMemory()

        def rf(pat: tuple[str, ...]) -> RetrievalFeatures:
            return RetrievalFeatures(
                "error",
                "api",
                "api",
                frozenset({"db"}),
                0.9,
                2,
                1,
                "error:api:True",
                pat,
            )

        matcher.index(
            "past-rollback",
            matcher._extractor.extract(
                {
                    "trigger_type": "error_rate",
                    "service": "api",
                    "upstream": ["db"],
                }
            ),
            {},
            retrieval_features=rf(
                ("deploy", "latency", "error", "rollback"),
            ),
        )
        matcher.index(
            "past-other",
            matcher._extractor.extract(
                {
                    "trigger_type": "error_rate",
                    "service": "api",
                    "upstream": ["db"],
                }
            ),
            {},
            retrieval_features=rf(
                ("deploy", "remediation", "deploy"),
            ),
        )

        query = {
            "trigger_type": "error_rate",
            "service": "api",
            "upstream": ["db"],
        }
        qfeat = rf(("deploy", "latency", "error"))
        rctx = RerankContext(query_features=qfeat, remediation_memory=mem)
        hits = matcher.top_k(query, k=2, rerank_context=rctx)
        assert hits[0].incident_id == "past-rollback"
        assert hits[1].incident_id == "past-other"
        assert hits[0].score >= hits[1].score
