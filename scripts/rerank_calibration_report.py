#!/usr/bin/env python3
"""Run empirical rerank failure analysis on a tiny demo corpus (stdout report).

Use from your benchmark harness by calling
``contexter.benchmark_failure_analysis.run_benchmark_calibration_report`` with
your ``FingerprintMatcher`` and labeled ``runs`` (same shape as
``FingerprintMatcher.calibrate_weights``).
"""

from __future__ import annotations

from contexter.benchmark_failure_analysis import BenchmarkRunSpec, run_benchmark_calibration_report
from contexter.incident_fingerprint import (
    FingerprintMatcher,
    IncidentFingerprint,
    PropagationFingerprint,
    RerankContext,
    RetrievalFeatures,
    normalize_trigger,
)
from contexter.remediation_memory import RemediationMemory

_DEEP = PropagationFingerprint(
    degradation_order=("redis", "api"),
    edge_types=("metric", "metric"),
    propagation_hops=(1, 1),
    hop_count=2,
    propagation_depth=2,
)
_SHALLOW = PropagationFingerprint((), (), (), 0, 0)


def _rf(
    fp: IncidentFingerprint,
    *,
    deploy: float,
    remed: str,
    prop: PropagationFingerprint,
    dp: tuple[str, ...] = (),
) -> RetrievalFeatures:
    return RetrievalFeatures(
        normalize_trigger(fp.trigger_type),
        fp.affected_role,
        fp.canonical_affected,
        fp.upstream_involved,
        deploy,
        3,
        2,
        remed,
        dp,
        prop,
    )


def main() -> None:
    mem = RemediationMemory()
    remed = "latency:checkout-api:True"
    mem.record(remed, "rollback", outcome="resolved")
    matcher = FingerprintMatcher(remediation_memory=mem)
    pattern = ("deploy", "latency", "error")

    qfp = IncidentFingerprint("latency", "checkout-api", frozenset({"redis"}), "checkout-api")
    qfeat = _rf(qfp, deploy=0.92, remed=remed, prop=_DEEP, dp=pattern)

    good = IncidentFingerprint("slow_queries", "checkout-api", frozenset({"redis"}), "checkout-api")
    matcher.index(
        "inc-true",
        good,
        {},
        retrieval_features=_rf(good, deploy=0.91, remed=remed, prop=_DEEP, dp=pattern),
    )
    noise = IncidentFingerprint("latency", "noise-api", frozenset(), "noise-api")
    matcher.index(
        "inc-noise",
        noise,
        {},
        retrieval_features=_rf(noise, deploy=0.96, remed="other", prop=_SHALLOW, dp=("deploy",)),
    )

    rctx = RerankContext(query_features=qfeat, remediation_memory=mem, query_canonical="checkout-api")
    runs = [
        BenchmarkRunSpec(
            query=qfp,
            rerank_context=rctx,
            expected_incident_id="inc-true",
            k=1,
            label="demo_expect_rank1",
        ),
        BenchmarkRunSpec(
            query=qfp,
            rerank_context=rctx,
            expected_incident_id="NOT_INDEXED",
            k=5,
            label="demo_missing_expected",
        ),
    ]
    run_benchmark_calibration_report(
        matcher,
        runs,
        title="Demo rerank calibration report (scripts/rerank_calibration_report.py)",
    )


if __name__ == "__main__":
    main()
