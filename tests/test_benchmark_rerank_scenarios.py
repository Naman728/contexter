"""Benchmark-oriented retrieval scenarios (R@5 / P@5 proxies, no external harness).

Harness targets (informational): recall@5 > 0.60, precision@5 > 0.45.
These tests assert deterministic ranking invariants under hard negatives, deploy overlap,
trigger collisions, cascade depth, and multi-service noise — same code path as production.
"""

from dataclasses import replace

from contexter.incident_fingerprint import (
    FingerprintMatcher,
    IncidentFingerprint,
    PropagationFingerprint,
    RerankContext,
    RetrievalFeatures,
    TemporalProfile,
    fingerprint_remediation_hash,
    normalize_trigger,
)
from contexter.remediation_memory import RemediationMemory


_DEEP = PropagationFingerprint(
    degradation_order=("redis", "api", "edge"),
    edge_types=("metric", "metric"),
    propagation_hops=(1, 1),
    hop_count=2,
    propagation_depth=3,
    role_transitions=("cache>api", "api>gateway"),
    terminal_failure_role="gateway",
    root_degradation_role="cache",
)
_SHALLOW = PropagationFingerprint((), (), (), 0, 0)
_ALT_CASCADE = PropagationFingerprint(
    degradation_order=("worker", "queue", "api"),
    edge_types=("unknown", "metric"),
    propagation_hops=(1, 1),
    hop_count=2,
    propagation_depth=2,
    role_transitions=("worker>queue", "queue>api"),
    terminal_failure_role="api",
    root_degradation_role="worker",
)


def _with_hash(fp: IncidentFingerprint, **kwargs: object) -> RetrievalFeatures:
    """Build ``RetrievalFeatures`` with ``remediation_fp_hash`` aligned to contents."""
    deploy = float(kwargs.get("deploy", 0.0))
    causal = int(kwargs.get("causal", 0))
    prop_edges = int(kwargs.get("prop_edges", 0))
    deploy_pattern = kwargs.get("deploy_pattern", ()) or ()
    prop_fp = kwargs.get("prop_fp", _SHALLOW)
    assert isinstance(prop_fp, PropagationFingerprint)
    tempo = kwargs.get("tempo") or TemporalProfile.missing()
    assert isinstance(tempo, TemporalProfile)
    partial = RetrievalFeatures(
        normalize_trigger(fp.trigger_type),
        fp.affected_role,
        fp.canonical_affected,
        fp.upstream_involved,
        deploy,
        causal,
        prop_edges,
        "",
        tuple(deploy_pattern),
        prop_fp,
        tempo,
    )
    return replace(partial, remediation_fp_hash=fingerprint_remediation_hash(fp, partial))


def _rf(
    *,
    fp: IncidentFingerprint,
    deploy: float,
    causal: int,
    prop_edges: int,
    remed_hash: str,
    deploy_pattern: tuple[str, ...],
    prop_fp: PropagationFingerprint,
    tempo: TemporalProfile | None = None,
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
        temporal_profile=tempo or TemporalProfile.missing(),
    )


def _h(fp: IncidentFingerprint, rf: RetrievalFeatures) -> str:
    return fingerprint_remediation_hash(fp, rf)


def test_hard_negative_same_trigger_wrong_upstream_not_in_top1() -> None:
    """Lexically similar services; wrong upstream / deploy narrative must not win top-1."""
    mem = RemediationMemory()
    qfp = IncidentFingerprint("latency", "checkout-api", frozenset({"redis", "db"}), "checkout-api")
    pattern = ("deploy", "latency", "error")
    qtmp = TemporalProfile(200.0, 30.0, 400.0, 25.0)
    qfeat = _rf(
        fp=qfp,
        deploy=0.92,
        causal=5,
        prop_edges=4,
        remed_hash=_h(
            qfp,
            _rf(
                fp=qfp,
                deploy=0.92,
                causal=5,
                prop_edges=4,
                remed_hash="",
                deploy_pattern=pattern,
                prop_fp=_DEEP,
                tempo=qtmp,
            ),
        ),
        deploy_pattern=pattern,
        prop_fp=_DEEP,
        tempo=qtmp,
    )
    mem.record(qfeat.remediation_fp_hash, "rollback", outcome="resolved")

    matcher = FingerprintMatcher(remediation_memory=mem)
    good = IncidentFingerprint("slow_queries", "checkout-api", frozenset({"redis", "db"}), "checkout-api")
    hard_neg = IncidentFingerprint("slow_queries", "checkout-api", frozenset({"kafka"}), "checkout-api")

    g_rf = _rf(
        fp=good,
        deploy=0.9,
        causal=5,
        prop_edges=4,
        remed_hash=qfeat.remediation_fp_hash,
        deploy_pattern=pattern,
        prop_fp=_DEEP,
        tempo=TemporalProfile(210.0, 28.0, 380.0, 24.0),
    )
    n_rf = _rf(
        fp=hard_neg,
        deploy=0.1,
        causal=5,
        prop_edges=4,
        remed_hash=_h(
            hard_neg,
            _rf(
                fp=hard_neg,
                deploy=0.1,
                causal=5,
                prop_edges=4,
                remed_hash="",
                deploy_pattern=("latency", "error"),
                prop_fp=_DEEP,
                tempo=TemporalProfile(5.0, 1.0, 20.0, 2.0),
            ),
        ),
        deploy_pattern=("latency", "error"),
        prop_fp=_DEEP,
        tempo=TemporalProfile(5.0, 1.0, 20.0, 2.0),
    )

    matcher.index("GOOD", good, {}, retrieval_features=g_rf)
    matcher.index("NEG", hard_neg, {}, retrieval_features=n_rf)

    rctx = RerankContext(query_features=qfeat, remediation_memory=mem)
    ranked = matcher.top_k(qfp, k=5, rerank_context=rctx, two_stage=True)
    ids = [m.incident_id for m in ranked]
    assert "GOOD" in ids[:5]
    assert ids[0] == "GOOD"
    assert ids.index("NEG") > 0


def test_overlapping_deploy_incidents_deploy_aligned_first() -> None:
    """Same trigger/role; first deploy-pattern token overlap should rank deploy path first."""
    mem = RemediationMemory()
    qfp = IncidentFingerprint("errors", "api", frozenset({"db"}), "api")
    p_shared = ("deploy", "errors")
    qfeat = _rf(
        fp=qfp,
        deploy=0.88,
        causal=4,
        prop_edges=3,
        remed_hash=_h(qfp, _rf(fp=qfp, deploy=0.88, causal=4, prop_edges=3, remed_hash="", deploy_pattern=p_shared, prop_fp=_SHALLOW)),
        deploy_pattern=p_shared,
        prop_fp=_SHALLOW,
    )
    matcher = FingerprintMatcher(remediation_memory=mem)

    aligned = IncidentFingerprint("5xx", "api", frozenset({"db"}), "api")
    other = IncidentFingerprint("5xx", "api", frozenset({"db"}), "api")
    matcher.index(
        "aligned",
        aligned,
        {},
        retrieval_features=_rf(
            fp=aligned,
            deploy=0.87,
            causal=4,
            prop_edges=3,
            remed_hash=qfeat.remediation_fp_hash,
            deploy_pattern=p_shared,
            prop_fp=_SHALLOW,
        ),
    )
    matcher.index(
        "other",
        other,
        {},
        retrieval_features=_rf(
            fp=other,
            deploy=0.86,
            causal=4,
            prop_edges=3,
            remed_hash=qfeat.remediation_fp_hash,
            deploy_pattern=("metric", "errors"),
            prop_fp=_SHALLOW,
        ),
    )
    rctx = RerankContext(query_features=qfeat, remediation_memory=mem)
    ranked = matcher.top_k(qfp, k=5, rerank_context=rctx, two_stage=True)
    assert ranked[0].incident_id == "aligned"


def test_same_trigger_different_cascade_topology_prefers_path_match() -> None:
    """Same normalized trigger and role; propagation path similarity breaks ties toward query."""
    mem = RemediationMemory()
    qfp = IncidentFingerprint("errors", "api", frozenset({"db"}), "api")
    qfeat = _rf(
        fp=qfp,
        deploy=0.5,
        causal=4,
        prop_edges=3,
        remed_hash=_h(qfp, _rf(fp=qfp, deploy=0.5, causal=4, prop_edges=3, remed_hash="", deploy_pattern=(), prop_fp=_DEEP)),
        deploy_pattern=(),
        prop_fp=_DEEP,
    )
    matcher = FingerprintMatcher(remediation_memory=mem)
    twin = IncidentFingerprint("timeout", "api", frozenset({"db"}), "api")
    matcher.index(
        "path_match",
        twin,
        {},
        retrieval_features=_rf(
            fp=twin,
            deploy=0.5,
            causal=4,
            prop_edges=3,
            remed_hash=qfeat.remediation_fp_hash,
            deploy_pattern=(),
            prop_fp=_DEEP,
        ),
    )
    matcher.index(
        "path_diff",
        twin,
        {},
        retrieval_features=_rf(
            fp=twin,
            deploy=0.5,
            causal=4,
            prop_edges=3,
            remed_hash=qfeat.remediation_fp_hash,
            deploy_pattern=(),
            prop_fp=_ALT_CASCADE,
        ),
    )
    rctx = RerankContext(query_features=qfeat, remediation_memory=mem)
    ranked = matcher.top_k(qfp, k=5, rerank_context=rctx, two_stage=True)
    assert ranked[0].incident_id == "path_match"
    assert ranked[1].incident_id == "path_diff"


def test_deep_query_shallow_distractor_recall_at5() -> None:
    """Query encodes deep cascade; shallow twin must stay below deep match (P@5 proxy)."""
    mem = RemediationMemory()
    pattern = ("deploy", "latency")
    qfp = IncidentFingerprint("latency", "svc-api", frozenset({"cache"}), "svc-api")
    qh = _h(
        qfp,
        _rf(
            fp=qfp,
            deploy=0.9,
            causal=6,
            prop_edges=5,
            remed_hash="",
            deploy_pattern=pattern,
            prop_fp=_DEEP,
        ),
    )
    mem.record(qh, "rollback", outcome="resolved")
    qfeat = _rf(
        fp=qfp,
        deploy=0.9,
        causal=6,
        prop_edges=5,
        remed_hash=qh,
        deploy_pattern=pattern,
        prop_fp=_DEEP,
    )
    matcher = FingerprintMatcher(remediation_memory=mem)
    twin = IncidentFingerprint("slow_queries", "svc-api", frozenset({"cache"}), "svc-api")
    matcher.index(
        "deep",
        twin,
        {},
        retrieval_features=_rf(
            fp=twin,
            deploy=0.89,
            causal=6,
            prop_edges=5,
            remed_hash=qh,
            deploy_pattern=pattern,
            prop_fp=_DEEP,
        ),
    )
    matcher.index(
        "shallow",
        twin,
        {},
        retrieval_features=_rf(
            fp=twin,
            deploy=0.89,
            causal=6,
            prop_edges=5,
            remed_hash=qh,
            deploy_pattern=pattern,
            prop_fp=_SHALLOW,
        ),
    )
    rctx = RerankContext(query_features=qfeat, remediation_memory=mem)
    ranked = matcher.top_k(qfp, k=5, rerank_context=rctx, two_stage=True)
    ids = [m.incident_id for m in ranked]
    assert ids[:2] == ["deep", "shallow"]
    assert "deep" in ids[:5]


def test_noisy_multi_service_expected_in_top5() -> None:
    """Many *-api distractors share locale; behavioral + structure keep true incident in top-5."""
    mem = RemediationMemory()

    prop_match = PropagationFingerprint(
        degradation_order=("redis", "checkout-api"),
        edge_types=("metric",),
        propagation_hops=(1,),
        hop_count=1,
        propagation_depth=1,
    )
    empty_prop = PropagationFingerprint((), (), (), 0, 0)
    pattern = ("deploy", "latency", "error")

    class _RegionalGraph:
        __slots__ = ()

        def neighbors(self, root: str) -> set[str]:
            if root == "legacy-db":
                return set()
            if root == "checkout-api" or root.endswith("-api"):
                return {"hub", "mesh", "edge", "east", "west"}
            return set()

    graph = _RegionalGraph()
    matcher = FingerprintMatcher(remediation_memory=mem)

    qfp = IncidentFingerprint("p99_latency", "checkout-api", frozenset({"redis"}), "checkout-api")
    qfeat = _rf(
        fp=qfp,
        deploy=0.95,
        causal=4,
        prop_edges=2,
        remed_hash="",
        deploy_pattern=pattern,
        prop_fp=prop_match,
    )
    qfeat = replace(qfeat, remediation_fp_hash=fingerprint_remediation_hash(qfp, qfeat))
    mem.record(qfeat.remediation_fp_hash, "rollback", outcome="resolved")

    true_fp = IncidentFingerprint("slow_queries", "checkout-api", frozenset({"redis"}), "legacy-db")
    matcher.index(
        "inc-true",
        true_fp,
        {},
        retrieval_features=_rf(
            fp=true_fp,
            deploy=0.94,
            causal=4,
            prop_edges=2,
            remed_hash=qfeat.remediation_fp_hash,
            deploy_pattern=pattern,
            prop_fp=prop_match,
        ),
    )
    for i in range(18):
        name = f"noise-{i}-api"
        dfp = IncidentFingerprint("p99_latency", name, frozenset(), name)
        matcher.index(
            f"inc-d{i}",
            dfp,
            {},
            retrieval_features=_with_hash(
                dfp,
                deploy=0.96,
                causal=4,
                prop_edges=2,
                deploy_pattern=("deploy",),
                prop_fp=empty_prop,
            ),
        )

    rctx = RerankContext(
        query_features=qfeat,
        remediation_memory=mem,
        dependency_graph=graph,
        query_canonical="checkout-api",
    )
    ranked = matcher.top_k(qfp, k=5, rerank_context=rctx, two_stage=True)
    ids = [m.incident_id for m in ranked]
    assert "inc-true" in ids
    assert ids.index("inc-true") < 5
