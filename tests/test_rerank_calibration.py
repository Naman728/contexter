"""Rerank introspection, behavioral recurrence, and calibration stats."""

from __future__ import annotations

from contexter.incident_fingerprint import (
    FingerprintMatcher,
    IncidentFingerprint,
    PropagationFingerprint,
    RerankContext,
    RetrievalFeatures,
    normalize_trigger,
    rerank_score_breakdown,
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


def test_rerank_decomposition_has_contract_keys() -> None:
    mem = RemediationMemory()
    qfp = IncidentFingerprint("latency", "checkout-api", frozenset({"redis"}), "checkout-api")
    cfp = IncidentFingerprint("latency", "checkout-api", frozenset({"redis"}), "checkout-api")
    qf = RetrievalFeatures.from_fingerprint(qfp)
    cf = RetrievalFeatures.from_fingerprint(cfp)
    ctx = RerankContext(query_features=qf, remediation_memory=mem)
    d = rerank_score_breakdown(qfp, qf, cfp, cf, ctx, incident_id="INC-X")
    for key in (
        "incident_id",
        "total_score",
        "trigger_score",
        "deploy_pattern_score",
        "propagation_score",
        "topology_score",
        "upstream_score",
        "remediation_score",
        "temporal_score",
        "alias_score",
        "role_score",
        "behavioral_recurrence",
        "recurrence_prior",
        "core_linear",
        "idf_trigger",
        "idf_role_cluster",
        "idf_deploy_shape",
        "idf_propagation_depth",
        "rarity_factor",
        "temporal_shape_similarity",
        "negative_evidence_multiplier",
        "generic_feature_multiplier",
        "high_confidence_struct_boost",
        "margin_calibration_delta",
    ):
        assert key in d
    assert d["incident_id"] == "INC-X"


def test_behavioral_recurrence_beats_topology_only_distractors() -> None:
    """One true behavioral recurrence; many *-api services share a fat graph with the query."""
    mem = RemediationMemory()
    remed_h = "latency:checkout-api:True"
    mem.record(remed_h, "rollback", outcome="resolved")

    prop_match = PropagationFingerprint(
        degradation_order=("redis", "checkout-api", "frontend"),
        edge_types=("metric", "metric"),
        propagation_hops=(1, 1),
        hop_count=2,
        propagation_depth=2,
    )
    empty_prop = PropagationFingerprint((), (), (), 0, 0)
    pattern = ("deploy", "latency", "error")

    class _RegionalGraph:
        """Query and *-api distractors share a dense region; legacy-db stays isolated."""

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
        remed_hash=remed_h,
        deploy_pattern=pattern,
        prop_fp=prop_match,
    )

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
            remed_hash=remed_h,
            deploy_pattern=pattern,
            prop_fp=prop_match,
        ),
    )

    for i in range(12):
        name = f"noise-{i}-api"
        dfp = IncidentFingerprint("p99_latency", name, frozenset(), name)
        matcher.index(
            f"inc-d{i}",
            dfp,
            {},
            retrieval_features=_rf(
                fp=dfp,
                deploy=0.96,
                causal=4,
                prop_edges=2,
                remed_hash="latency:checkout-api:False",
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
    assert ranked[0].incident_id == "inc-true"
    for m in ranked[1:]:
        assert m.incident_id.startswith("inc-d")
    calib = matcher.last_ranking_calibration_stats()
    assert calib is not None
    assert "mean_top5_adjacent_spread" in calib
    assert "top1_top2_margin" in calib
    assert "near_tie_count_below_top1" in calib


def test_calibrate_weights_reports_hit_miss_stats() -> None:
    mem = RemediationMemory()
    matcher = FingerprintMatcher(remediation_memory=mem)
    remed_h = "error:api:True"
    mem.record(remed_h, "rollback", outcome="resolved")

    qfp = IncidentFingerprint("errors", "api", frozenset({"db"}), "api-svc")
    qfeat = _rf(
        fp=qfp,
        deploy=0.9,
        causal=3,
        prop_edges=2,
        remed_hash=remed_h,
        deploy_pattern=("deploy", "error"),
        prop_fp=PropagationFingerprint((), (), (), 0, 0),
    )

    good = IncidentFingerprint("5xx", "api", frozenset({"db"}), "api-svc")
    bad = IncidentFingerprint("5xx", "api", frozenset(), "api-svc")
    matcher.index(
        "GOOD",
        good,
        {},
        retrieval_features=_rf(
            fp=good,
            deploy=0.92,
            causal=3,
            prop_edges=2,
            remed_hash=remed_h,
            deploy_pattern=("deploy", "error"),
            prop_fp=PropagationFingerprint((), (), (), 0, 0),
        ),
    )
    matcher.index(
        "BAD",
        bad,
        {},
        retrieval_features=_rf(
            fp=bad,
            deploy=0.05,
            causal=0,
            prop_edges=0,
            remed_hash="error:api:False",
            deploy_pattern=(),
            prop_fp=PropagationFingerprint((), (), (), 0, 0),
        ),
    )

    rctx = RerankContext(query_features=qfeat, remediation_memory=mem)
    runs = [
        {
            "query": qfp,
            "rerank_context": rctx,
            "expected_incident_id": "GOOD",
            "k": 1,
        }
    ]
    rep = matcher.calibrate_weights(runs)
    assert rep["n_top5_hit"] >= 1
    assert "signal_mean_on_hit" in rep
    assert "signal_delta_hit_minus_miss" in rep
    assert "signals_sorted_by_success_correlation" in rep
    assert "false_positive_dominance_counts" in rep
