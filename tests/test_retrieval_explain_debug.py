"""retrieval_explain_debug: zero-cost when off, structured breakdown when on."""

from __future__ import annotations

import io
from contexter.incident_fingerprint import (
    FingerprintMatcher,
    IncidentFingerprint,
    PropagationFingerprint,
    RerankContext,
    RetrievalFeatures,
    fingerprint_remediation_hash,
    normalize_trigger,
    retrieval_explain_debug,
)
from contexter.remediation_memory import RemediationMemory


class _CountingMatcher(FingerprintMatcher):
    def __init__(self) -> None:
        super().__init__()
        self._top_k_calls = 0

    def top_k(self, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        self._top_k_calls += 1
        return super().top_k(*args, **kwargs)


def test_retrieval_explain_debug_disabled_does_not_call_top_k() -> None:
    m = _CountingMatcher()
    q = IncidentFingerprint("latency", "api", frozenset({"db"}), "api")
    assert retrieval_explain_debug(m, q, debug=False) is None
    assert m._top_k_calls == 0


def test_retrieval_explain_debug_legacy_has_sorted_contributions() -> None:
    m = FingerprintMatcher()
    q = IncidentFingerprint("errors", "api", frozenset({"db"}), "api-svc")
    c = IncidentFingerprint("5xx", "api", frozenset({"db"}), "api-svc")
    m.index("hit", c, {})
    rows = retrieval_explain_debug(m, q, k=3, debug=True, print_report=False)
    assert rows is not None
    assert len(rows) == 1
    r = rows[0]
    for key in (
        "incident_id",
        "final_score",
        "trigger_contribution",
        "role_contribution",
        "propagation_contribution",
        "temporal_contribution",
        "topology_contribution",
        "penalties",
        "retrieval_sources",
        "contributions_sorted",
        "extras",
    ):
        assert key in r
    assert r["incident_id"] == "hit"
    assert r["retrieval_sources"] == []
    assert r["contributions_sorted"][0][0]  # non-empty name
    assert isinstance(r["contributions_sorted"][0][1], float)


def test_retrieval_explain_debug_two_stage_includes_sources_and_penalties() -> None:
    mem = RemediationMemory()
    pattern = ("deploy", "latency")
    qfp = IncidentFingerprint("latency", "checkout-api", frozenset({"redis"}), "checkout-api")
    qh = fingerprint_remediation_hash(qfp)
    mem.record(qh, "rollback", outcome="resolved")
    qfeat = RetrievalFeatures(
        normalize_trigger(qfp.trigger_type),
        qfp.affected_role,
        qfp.canonical_affected,
        qfp.upstream_involved,
        deploy_proximity=0.9,
        causal_edge_count=4,
        propagation_edge_count=3,
        remediation_fp_hash=qh,
        deploy_pattern=pattern,
        propagation_fingerprint=PropagationFingerprint(
            degradation_order=("redis", "checkout-api"),
            edge_types=("metric",),
            propagation_hops=(1,),
            hop_count=1,
            propagation_depth=1,
        ),
    )
    matcher = FingerprintMatcher(remediation_memory=mem)
    twin = IncidentFingerprint("slow_queries", "checkout-api", frozenset({"redis"}), "checkout-api")
    matcher.index(
        "inc-a",
        twin,
        {},
        retrieval_features=RetrievalFeatures(
            normalize_trigger(twin.trigger_type),
            twin.affected_role,
            twin.canonical_affected,
            twin.upstream_involved,
            deploy_proximity=0.88,
            causal_edge_count=4,
            propagation_edge_count=3,
            remediation_fp_hash=qh,
            deploy_pattern=pattern,
            propagation_fingerprint=PropagationFingerprint(
                degradation_order=("redis", "checkout-api"),
                edge_types=("metric",),
                propagation_hops=(1,),
                hop_count=1,
                propagation_depth=1,
            ),
        ),
    )
    rctx = RerankContext(query_features=qfeat, remediation_memory=mem)
    buf = io.StringIO()
    rows = retrieval_explain_debug(
        matcher,
        qfp,
        rerank_context=rctx,
        k=2,
        debug=True,
        print_report=True,
        stream=buf,
    )
    assert rows is not None
    assert len(rows) >= 1
    text = buf.getvalue()
    assert "inc-a" in text
    assert "retrieval_sources:" in text
    assert "contributions_sorted" in text
    assert "negative_evidence_multiplier" in rows[0]["penalties"]
