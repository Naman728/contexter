"""Structural IDF reranking: rare patterns beat generic incidents."""

from __future__ import annotations

from contexter.incident_fingerprint import (
    FingerprintMatcher,
    IncidentFingerprint,
    PropagationFingerprint,
    RerankContext,
    RetrievalFeatures,
    normalize_trigger,
    rerank_component_values,
)
from contexter.remediation_memory import RemediationMemory

_EMPTY_PROP = PropagationFingerprint((), (), (), 0, 0)


def _rf(
    *,
    fp: IncidentFingerprint,
    deploy_pattern: tuple[str, ...],
    prop_fp: PropagationFingerprint,
    remed_hash: str = "e:api:True",
) -> RetrievalFeatures:
    return RetrievalFeatures(
        normalize_trigger(fp.trigger_type),
        fp.affected_role,
        fp.canonical_affected,
        fp.upstream_involved,
        deploy_proximity=0.9,
        causal_edge_count=3,
        propagation_edge_count=max(1, prop_fp.hop_count),
        remediation_fp_hash=remed_hash,
        deploy_pattern=deploy_pattern,
        propagation_fingerprint=prop_fp,
    )


def test_rare_deploy_shape_outranks_generic_deploy_incidents() -> None:
    """Many ``(deploy,)`` shapes vs one rare ``(deploy, latency)``; query aligns with rare."""
    mem = RemediationMemory()
    matcher = FingerprintMatcher(remediation_memory=mem)

    rare_pat = ("deploy", "latency", "error")
    generic_pat = ("deploy",)

    rare_fp = IncidentFingerprint("error_rate", "payments-api", frozenset({"db"}), "pay-1")
    matcher.index(
        "rare-1",
        rare_fp,
        {},
        retrieval_features=_rf(fp=rare_fp, deploy_pattern=rare_pat, prop_fp=_EMPTY_PROP),
    )
    for i in range(18):
        gfp = IncidentFingerprint("error_rate", "payments-api", frozenset(), f"svc-{i}")
        matcher.index(
            f"gen-{i}",
            gfp,
            {},
            retrieval_features=_rf(fp=gfp, deploy_pattern=generic_pat, prop_fp=_EMPTY_PROP),
        )

    query_fp = IncidentFingerprint("5xx_rate", "payments-api", frozenset({"db"}), "pay-q")
    qfeat = _rf(fp=query_fp, deploy_pattern=rare_pat, prop_fp=_EMPTY_PROP)
    rctx = RerankContext(query_features=qfeat, remediation_memory=mem)

    ranked = matcher.top_k(query_fp, k=5, rerank_context=rctx, two_stage=True)
    assert ranked[0].incident_id == "rare-1"
    assert ranked[0].score >= ranked[1].score


def test_rare_propagation_depth_outranks_common_shallow_noise() -> None:
    """Many depth-0 cascades vs one deeper propagation depth; query matches deep shape."""
    mem = RemediationMemory()
    matcher = FingerprintMatcher(remediation_memory=mem)

    shallow = PropagationFingerprint((), (), (), 0, 0)
    deep = PropagationFingerprint(
        degradation_order=("redis", "api", "fe"),
        edge_types=("metric", "metric"),
        propagation_hops=(1, 1),
        hop_count=2,
        propagation_depth=4,
    )
    pat = ("deploy", "latency")

    deep_fp = IncidentFingerprint("latency", "checkout-api", frozenset({"cache"}), "c1")
    matcher.index(
        "deep-1",
        deep_fp,
        {},
        retrieval_features=_rf(fp=deep_fp, deploy_pattern=pat, prop_fp=deep),
    )
    for i in range(16):
        sfp = IncidentFingerprint("latency", "checkout-api", frozenset(), f"s{i}")
        matcher.index(
            f"shallow-{i}",
            sfp,
            {},
            retrieval_features=_rf(fp=sfp, deploy_pattern=pat, prop_fp=shallow),
        )

    query_fp = IncidentFingerprint("p99", "checkout-api", frozenset({"cache"}), "c-q")
    qfeat = _rf(fp=query_fp, deploy_pattern=pat, prop_fp=deep)
    rctx = RerankContext(query_features=qfeat, remediation_memory=mem)

    ranked = matcher.top_k(query_fp, k=5, rerank_context=rctx, two_stage=True)
    assert ranked[0].incident_id == "deep-1"
    assert ranked[0].score >= ranked[1].score


def test_matched_trigger_scalar_downweighted_for_dominant_trigger_family() -> None:
    """With corpus IDF, identical coarse triggers contribute less than a raw 1.0 match."""
    mem = RemediationMemory()
    matcher = FingerprintMatcher(remediation_memory=mem)
    pat = ("deploy", "latency")
    prop = _EMPTY_PROP
    for i in range(26):
        fp = IncidentFingerprint("latency", f"svc-{i}", frozenset(), f"svc-{i}")
        matcher.index(str(i), fp, {}, retrieval_features=_rf(fp=fp, deploy_pattern=pat, prop_fp=prop))

    idf = matcher._idf_stats()
    qfp = IncidentFingerprint("latency", "svc-0", frozenset(), "svc-0")
    qfeat = _rf(fp=qfp, deploy_pattern=pat, prop_fp=prop)
    cand_fp = IncidentFingerprint("latency", "svc-1", frozenset(), "svc-1")
    cfeat = _rf(fp=cand_fp, deploy_pattern=pat, prop_fp=prop)
    rctx = RerankContext(query_features=qfeat, remediation_memory=mem)

    raw = rerank_component_values(qfp, qfeat, cand_fp, cfeat, rctx, idf_stats=None)
    adj = rerank_component_values(qfp, qfeat, cand_fp, cfeat, rctx, idf_stats=idf)
    assert raw["trigger"] == 1.0
    assert adj["trigger"] < raw["trigger"]
