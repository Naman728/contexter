"""Benchmark rerank failure analyzer (empirical diagnostics, no new retrieval)."""

from __future__ import annotations

import io

from contexter.benchmark_failure_analysis import (
    BenchmarkRunSpec,
    FailureArchetype,
    analyze_benchmark_recall_failures,
    format_rerank_calibration_report,
    print_calibration_report,
)
from contexter.incident_fingerprint import (
    FingerprintMatcher,
    IncidentFingerprint,
    PropagationFingerprint,
    RerankContext,
    RetrievalFeatures,
    effective_rerank_weights,
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


def test_effective_rerank_weights_defaults() -> None:
    mem = RemediationMemory()
    qfp = IncidentFingerprint("latency", "api", frozenset(), "api")
    qf = RetrievalFeatures.from_fingerprint(qfp)
    ctx = RerankContext(query_features=qf, remediation_memory=mem)
    w = effective_rerank_weights(ctx)
    assert abs(sum(w.values()) - 0.79) < 1e-5
    assert "trigger" in w and "temporal" in w


def test_analyzer_flags_not_in_corpus() -> None:
    mem = RemediationMemory()
    matcher = FingerprintMatcher(remediation_memory=mem)
    qfp = IncidentFingerprint("latency", "api", frozenset({"redis"}), "api")
    qfeat = _rf(
        fp=qfp,
        deploy=0.9,
        causal=2,
        prop_edges=2,
        remed_hash="x",
        deploy_pattern=("deploy", "latency"),
        prop_fp=PropagationFingerprint((), (), (), 0, 0),
    )
    matcher.index(
        "ONLY",
        qfp,
        {},
        retrieval_features=qfeat,
    )
    rctx = RerankContext(query_features=qfeat, remediation_memory=mem)
    rep = analyze_benchmark_recall_failures(
        matcher,
        [
            {
                "query": qfp,
                "rerank_context": rctx,
                "expected_incident_id": "NOT_INDEXED",
                "k": 5,
                "label": "ghost",
            }
        ],
    )
    assert rep["n_runs_failures"] == 1
    assert rep["failures_not_in_corpus"] == 1
    assert rep["failures"][0]["not_in_corpus"] is True


def test_benchmark_run_spec_asdict_roundtrip() -> None:
    mem = RemediationMemory()
    qfp = IncidentFingerprint("latency", "api", frozenset(), "api")
    qfeat = RetrievalFeatures.from_fingerprint(qfp)
    rctx = RerankContext(query_features=qfeat, remediation_memory=mem)
    spec = BenchmarkRunSpec(
        query=qfp,
        rerank_context=rctx,
        expected_incident_id="X",
        label="t",
    )
    rep = analyze_benchmark_recall_failures(
        FingerprintMatcher(remediation_memory=mem),
        [spec],
    )
    assert rep["n_runs_failures"] == 1


def test_two_stage_full_ranked_matches_returns_debug_rows() -> None:
    mem = RemediationMemory()
    matcher = FingerprintMatcher(remediation_memory=mem)
    fp = IncidentFingerprint("latency", "api", frozenset(), "api")
    rf = RetrievalFeatures.from_fingerprint(fp)
    matcher.index("a", fp, {}, retrieval_features=rf)
    matcher.index("b", fp, {}, retrieval_features=rf)
    rctx = RerankContext(query_features=rf, remediation_memory=mem)
    rows = matcher.two_stage_full_ranked_matches(fp, rerank_context=rctx)
    assert len(rows) == 2
    assert rows[0].score_breakdown is not None
    assert "rerank_decomposition" in rows[0].score_breakdown


def test_format_and_print_report_smoke() -> None:
    rep = {
        "n_runs_failures": 0,
        "failures_not_in_corpus": 0,
        "failures_not_in_rerank_pool": 0,
        "failures_min_score_gate": 0,
        "failures_not_in_pool": 0,
        "failures_below_min_score_only": 0,
        "margins_top1_minus_true": [],
        "margin_mean": 0.0,
        "margin_p50_proxy": 0.0,
        "components_fp_excess_count_when_higher": {},
        "components_mean_fp_minus_true_when_fp_higher": {},
        "components_mean_true_deficit_when_true_lower": {},
        "components_sorted_fp_over_score": [],
        "components_sorted_true_under_score": [],
        "failure_archetype_counts": {},
        "failure_archetype_top": [],
        "failures": [],
    }
    s = format_rerank_calibration_report(rep)
    assert "Failures" in s
    buf = io.StringIO()
    print_calibration_report(rep, stream=buf)
    assert buf.getvalue()


def test_failure_archetype_enum_values() -> None:
    assert FailureArchetype.DEPLOY_NOISE.value == "deploy_noise_failure"
