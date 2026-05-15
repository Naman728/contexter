"""Temporal profile extraction and rerank shape similarity."""

from __future__ import annotations

from dataclasses import replace

from contexter.incident_fingerprint import (
    FingerprintMatcher,
    IncidentFingerprint,
    PropagationFingerprint,
    RerankContext,
    RetrievalFeatures,
    TemporalProfile,
    fingerprint_remediation_hash,
    temporal_similarity,
)
from contexter.remediation_memory import RemediationMemory


def test_temporal_similarity_identical_profiles() -> None:
    p = TemporalProfile(120.0, 45.0, 300.0, 20.0)
    assert temporal_similarity(p, p) == 1.0


def test_temporal_similarity_distant_profiles_lower() -> None:
    a = TemporalProfile(60.0, 30.0, 120.0, 15.0)
    b = TemporalProfile(6000.0, 5.0, 10.0, 400.0)
    assert temporal_similarity(a, b) < temporal_similarity(a, a)


def test_similar_temporal_shape_wins_rerank() -> None:
    """Same structure; candidate with matching timing profile ranks above mismatched timing."""
    mem = RemediationMemory()
    matcher = FingerprintMatcher(remediation_memory=mem)
    fp = IncidentFingerprint("error_rate", "api-svc", frozenset({"db"}), "api-svc")

    shape_match = TemporalProfile(90.0, 40.0, 200.0, 25.0)
    shape_noise = TemporalProfile(5.0, 1.0, 20.0, 2.0)

    def rf(tp: TemporalProfile) -> RetrievalFeatures:
        base = RetrievalFeatures.from_fingerprint(fp)
        tmp = replace(
            base,
            deploy_proximity=0.9,
            causal_edge_count=3,
            propagation_edge_count=2,
            deploy_pattern=("deploy", "error"),
            temporal_profile=tp,
        )
        h = fingerprint_remediation_hash(fp, tmp)
        return replace(tmp, remediation_fp_hash=h)

    matcher.index("good", fp, {}, retrieval_features=rf(shape_match))
    matcher.index("bad", fp, {}, retrieval_features=rf(shape_noise))

    qfeat = rf(shape_match)
    rctx = RerankContext(query_features=qfeat, remediation_memory=mem)
    ranked = matcher.top_k(fp, k=2, rerank_context=rctx, two_stage=True)
    assert ranked[0].incident_id == "good"
    assert ranked[0].score >= ranked[1].score
