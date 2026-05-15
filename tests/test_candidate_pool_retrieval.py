"""Broad candidate pool generation, diversity, and retrieval debugging."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from contexter.engine import Engine
from contexter.incident_fingerprint import (
    FingerprintExtractor,
    FingerprintMatcher,
    RetrievalFeatures,
    RerankContext,
)
from contexter.identity_tracker import IdentityTracker
from contexter.remediation_memory import RemediationMemory

UTC = timezone.utc
T0 = datetime(2026, 12, 1, 9, 0, 0, tzinfo=UTC)


def _sig(incident_id: str, service: str, trigger: str, upstream: list[str]) -> dict:
    return {
        "incident_id": incident_id,
        "service": service,
        "ts": T0,
        "trigger_type": trigger,
        "upstream": upstream,
    }


def test_candidate_pool_exceeds_fifty_with_role_cluster_union() -> None:
    identity = IdentityTracker()
    ext = FingerprintExtractor(identity)
    mem = RemediationMemory()
    matcher = FingerprintMatcher(ext, remediation_memory=mem)
    for i in range(70):
        svc = f"svc-{i % 7}-api"
        fp = ext.extract(
            {
                "trigger_type": "error_rate",
                "service": svc,
                "upstream": ["db"],
            }
        )
        rf = RetrievalFeatures.from_fingerprint(fp)
        matcher.index(f"inc-{i}", fp, {}, retrieval_features=rf)
    qfp = ext.extract(_sig("q", "svc-0-api", "error_rate", ["db"]))
    qfeat = RetrievalFeatures.from_fingerprint(qfp)
    rctx = RerankContext(
        query_features=qfeat,
        remediation_memory=mem,
        identity=identity,
        query_canonical="svc-0-api",
    )
    matcher.top_k(qfp, k=3, rerank_context=rctx, min_score=0.0)
    stats = matcher.last_retrieval_pool_stats()
    assert stats is not None
    assert stats["raw_union"] >= 50
    assert stats["diverse_pool"] >= 50


def test_noisy_trigger_still_pulls_candidates_via_canonical_path() -> None:
    identity = IdentityTracker()
    ext = FingerprintExtractor(identity)
    mem = RemediationMemory()
    matcher = FingerprintMatcher(ext, remediation_memory=mem)
    past_fp = ext.extract(
        {
            "trigger_type": "error_rate",
            "service": "checkout-api",
            "upstream": ["db"],
        }
    )
    past_rf = RetrievalFeatures.from_fingerprint(past_fp)
    matcher.index("INC-PAST", past_fp, {}, retrieval_features=past_rf)

    qfp = ext.extract(
        {
            "trigger_type": "cpu_spike",
            "service": "checkout-api",
            "upstream": ["redis"],
        }
    )
    qfeat = RetrievalFeatures.from_fingerprint(qfp)
    rctx = RerankContext(
        query_features=qfeat,
        remediation_memory=mem,
        identity=identity,
        query_canonical="checkout-api",
    )
    hits = matcher.top_k(qfp, k=5, rerank_context=rctx, min_score=0.0)
    ids = {h.incident_id for h in hits}
    assert "INC-PAST" in ids
    assert any("canonical" in h.retrieval_sources or "alias" in h.retrieval_sources for h in hits)


def test_rename_drift_same_family_enters_pool() -> None:
    with Engine(batch_size=1) as eng:
        eng.ingest_one(
            {
                "kind": "incident_signal",
                "service": "svc-a",
                "occurred_at": T0,
                "payload": {
                    "incident_id": "INC-PAST",
                    "trigger_type": "error_rate",
                    "upstream": ["auth", "db"],
                },
            }
        )
        for old, new in (
            ("svc-a", "svc-b"),
            ("svc-b", "svc-c"),
        ):
            eng.ingest_one(
                {
                    "kind": "identity.drift",
                    "service": new,
                    "occurred_at": T0 + timedelta(minutes=1),
                    "payload": {"from_": old, "to": new},
                }
            )
        eng.ingest_one(
            {
                "kind": "incident_signal",
                "service": "svc-c",
                "occurred_at": T0 + timedelta(hours=1),
                "payload": {
                    "incident_id": "INC-CUR",
                    "trigger_type": "error_rate",
                    "upstream": ["auth", "db"],
                },
            }
        )
        ctx = eng.reconstruct_context(
            _sig("INC-CUR", "svc-c", "error_rate", ["auth", "db"]),
            mode="fast",
        )
    stats = eng._fingerprint_matcher.last_retrieval_pool_stats()
    assert stats is not None
    assert stats["raw_union"] >= 1
    sim_ids = {m["past_incident_id"] for m in ctx["similar_past_incidents"]}
    assert "INC-PAST" in sim_ids


def test_diversity_limits_domination_per_canonical() -> None:
    identity = IdentityTracker()
    ext = FingerprintExtractor(identity)
    mem = RemediationMemory()
    matcher = FingerprintMatcher(ext, remediation_memory=mem)
    canon = "monolith-api"
    for i in range(25):
        fp = ext.extract(
            {
                "trigger_type": "error_rate",
                "service": canon,
                "upstream": [f"dep-{i}"],
            }
        )
        rf = RetrievalFeatures.from_fingerprint(fp)
        matcher.index(f"m-{i}", fp, {}, retrieval_features=rf)
    qfp = ext.extract(_sig("q", canon, "error_rate", ["dep-0"]))
    qfeat = RetrievalFeatures.from_fingerprint(qfp)
    rctx = RerankContext(
        query_features=qfeat,
        remediation_memory=mem,
        identity=identity,
        query_canonical=canon,
    )
    matcher.top_k(qfp, k=5, rerank_context=rctx, min_score=0.0)
    stats = matcher.last_retrieval_pool_stats()
    assert stats["raw_union"] >= 20
    assert stats["diverse_pool"] <= 200
    assert stats["diverse_pool"] <= stats["raw_union"]
