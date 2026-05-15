"""Negative-evidence penalties in stage-2 reranking (structural contradictions)."""

from __future__ import annotations

from contexter.incident_fingerprint import (
    FingerprintMatcher,
    IncidentFingerprint,
    PropagationFingerprint,
    RerankContext,
    RetrievalFeatures,
    fingerprint_remediation_hash,
    normalize_trigger,
)
from contexter.remediation_memory import RemediationMemory


def _rf(
    *,
    fp: IncidentFingerprint,
    deploy: float,
    causal: int,
    prop_edges: int,
    remed_hash: str,
    deploy_pattern: tuple[str, ...],
    prop_fp: PropagationFingerprint,
) -> RetrievalFeatures:
    return RetrievalFeatures(
        normalize_trigger(fp.trigger_type),
        fp.affected_role,
        fp.canonical_affected,
        fp.upstream_involved,
        deploy_proximity=deploy,
        causal_edge_count=causal,
        propagation_edge_count=prop_edges,
        remediation_fp_hash=remed_hash,
        deploy_pattern=deploy_pattern,
        propagation_fingerprint=prop_fp,
    )


_DEEP_PROP = PropagationFingerprint(
    degradation_order=("redis", "cache", "checkout-api", "frontend", "edge"),
    edge_types=("metric", "metric", "metric", "metric"),
    propagation_hops=(1, 1, 1, 1),
    hop_count=4,
    propagation_depth=4,
)
_SHALLOW_PROP = PropagationFingerprint((), (), (), 0, 0)


def test_deploy_and_upstream_contradictions_rank_below_structural_peer() -> None:
    """Mismatched deploy narrative, proximity, disjoint upstreams, and remed conflict drop the bad row."""
    mem = RemediationMemory()
    pattern = ("deploy", "latency", "error")
    qfp = IncidentFingerprint("latency", "checkout-api", frozenset({"redis"}), "checkout-api")
    qh = fingerprint_remediation_hash(qfp)
    mem.record(qh, "rollback", outcome="resolved")

    qfeat = _rf(
        fp=qfp,
        deploy=0.92,
        causal=5,
        prop_edges=4,
        remed_hash=qh,
        deploy_pattern=pattern,
        prop_fp=_DEEP_PROP,
    )

    matcher = FingerprintMatcher(remediation_memory=mem)

    good_fp = IncidentFingerprint("slow_queries", "checkout-api", frozenset({"redis"}), "checkout-api")
    matcher.index(
        "inc-aligned",
        good_fp,
        {},
        retrieval_features=_rf(
            fp=good_fp,
            deploy=0.91,
            causal=5,
            prop_edges=4,
            remed_hash=qh,
            deploy_pattern=pattern,
            prop_fp=_DEEP_PROP,
        ),
    )

    bad_fp = IncidentFingerprint("slow_queries", "checkout-api", frozenset({"kafka"}), "checkout-api")
    bh = fingerprint_remediation_hash(bad_fp)
    assert bh != qh
    matcher.index(
        "inc-noisy",
        bad_fp,
        {},
        retrieval_features=_rf(
            fp=bad_fp,
            deploy=0.08,
            causal=5,
            prop_edges=4,
            remed_hash=bh,
            deploy_pattern=("latency", "error"),
            prop_fp=_DEEP_PROP,
        ),
    )

    rctx = RerankContext(query_features=qfeat, remediation_memory=mem)
    ranked = matcher.top_k(qfp, k=2, rerank_context=rctx, two_stage=True)
    assert ranked[0].incident_id == "inc-aligned"
    assert ranked[1].incident_id == "inc-noisy"


def test_shallow_candidate_does_not_outrank_deep_propagation_peer() -> None:
    """Query encodes a deep cascade; a shallow fingerprint must not beat a depth-aligned twin."""
    mem = RemediationMemory()
    pattern = ("deploy", "latency", "error")
    qfp = IncidentFingerprint("latency", "checkout-api", frozenset({"redis"}), "checkout-api")
    qh = fingerprint_remediation_hash(qfp)
    mem.record(qh, "rollback", outcome="resolved")

    qfeat = _rf(
        fp=qfp,
        deploy=0.9,
        causal=6,
        prop_edges=5,
        remed_hash=qh,
        deploy_pattern=pattern,
        prop_fp=_DEEP_PROP,
    )

    matcher = FingerprintMatcher(remediation_memory=mem)

    twin = IncidentFingerprint("slow_queries", "checkout-api", frozenset({"redis"}), "checkout-api")

    matcher.index(
        "inc-deep",
        twin,
        {},
        retrieval_features=_rf(
            fp=twin,
            deploy=0.89,
            causal=6,
            prop_edges=5,
            remed_hash=qh,
            deploy_pattern=pattern,
            prop_fp=_DEEP_PROP,
        ),
    )

    matcher.index(
        "inc-shallow",
        twin,
        {},
        retrieval_features=_rf(
            fp=twin,
            deploy=0.89,
            causal=6,
            prop_edges=5,
            remed_hash=qh,
            deploy_pattern=pattern,
            prop_fp=_SHALLOW_PROP,
        ),
    )

    rctx = RerankContext(query_features=qfeat, remediation_memory=mem)
    ranked = matcher.top_k(qfp, k=2, rerank_context=rctx, two_stage=True)
    assert ranked[0].incident_id == "inc-deep"
    assert ranked[1].incident_id == "inc-shallow"
