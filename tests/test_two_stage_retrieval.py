"""Two-stage fingerprint retrieval: broad recall then structural rerank."""

from __future__ import annotations

import time

from contexter.incident_fingerprint import (
    FingerprintMatcher,
    IncidentFingerprint,
    RerankContext,
    RetrievalFeatures,
    normalize_trigger,
)
from contexter.remediation_memory import RemediationMemory


def _fp(
    trigger: str,
    role: str,
    upstream: frozenset[str],
    canonical: str = "checkout-api",
) -> IncidentFingerprint:
    return IncidentFingerprint(trigger, role, upstream, canonical)


def _feat(
    *,
    deploy: float,
    causal: int,
    prop: int,
    remed_hash: str,
    fp: IncidentFingerprint,
    deploy_pattern: tuple[str, ...] = (),
) -> RetrievalFeatures:
    return RetrievalFeatures(
        normalize_trigger(fp.trigger_type),
        fp.affected_role,
        fp.canonical_affected,
        fp.upstream_involved,
        deploy_proximity=deploy,
        causal_edge_count=causal,
        propagation_edge_count=prop,
        remediation_fp_hash=remed_hash,
        deploy_pattern=deploy_pattern,
    )


class TestTwoStageReranking:
    def test_reranking_raises_same_family_above_noisy_neighbor(self) -> None:
        """Stage-2 uses deploy/causal/prop/remediation shape; family beats trigger-only noise."""
        matcher = FingerprintMatcher()
        mem = RemediationMemory()
        mem.record("error:api:True", "rollback", outcome="resolved")

        family_fp = _fp("error_rate", "api", frozenset({"auth", "db"}))
        noisy_fp = _fp("error_rate", "api", frozenset())

        matcher.index(
            "inc-family",
            family_fp,
            {},
            retrieval_features=_feat(
                deploy=0.92,
                causal=3,
                prop=2,
                remed_hash="error:api:True",
                fp=family_fp,
            ),
        )
        matcher.index(
            "inc-noisy",
            noisy_fp,
            {},
            retrieval_features=_feat(
                deploy=0.05,
                causal=0,
                prop=0,
                remed_hash="error:api:False",
                fp=noisy_fp,
            ),
        )

        query_fp = _fp("5xx_rate", "api", frozenset({"auth", "db", "telemetry"}))
        qfeat = _feat(
            deploy=0.9,
            causal=3,
            prop=2,
            remed_hash="error:api:True",
            fp=query_fp,
        )
        rctx = RerankContext(query_features=qfeat, remediation_memory=mem)

        query = {
            "trigger_type": "5xx_rate",
            "service": "checkout-api",
            "upstream": ["auth", "db", "telemetry"],
        }
        ranked = matcher.top_k(
            query,
            k=5,
            rerank_context=rctx,
            exclude_incident_id="inc-query",
        )
        ids = [m.incident_id for m in ranked]
        assert ids[0] == "inc-family"
        assert ids.index("inc-noisy") > ids.index("inc-family")

    def test_noisy_incident_scores_below_family_on_rerank(self) -> None:
        """Rerank scores separate same-trigger neighbors by ops + remediation shape."""
        matcher = FingerprintMatcher()
        mem = RemediationMemory()
        mem.record("error:api:True", "rollback", outcome="resolved")

        family_fp = _fp("error_rate", "api", frozenset({"auth", "db"}))
        noisy_fp = _fp("error_rate", "api", frozenset())

        matcher.index(
            "inc-family",
            family_fp,
            {},
            retrieval_features=_feat(
                deploy=0.92,
                causal=3,
                prop=2,
                remed_hash="error:api:True",
                fp=family_fp,
            ),
        )
        matcher.index(
            "inc-noisy",
            noisy_fp,
            {},
            retrieval_features=_feat(
                deploy=0.05,
                causal=0,
                prop=0,
                remed_hash="error:api:False",
                fp=noisy_fp,
            ),
        )

        query = {
            "trigger_type": "5xx_rate",
            "service": "checkout-api",
            "upstream": ["auth", "db", "telemetry"],
        }
        qfeat = _feat(
            deploy=0.9,
            causal=3,
            prop=2,
            remed_hash="error:api:True",
            fp=_fp("5xx_rate", "api", frozenset({"auth", "db", "telemetry"})),
        )
        rctx = RerankContext(query_features=qfeat, remediation_memory=mem)
        two = matcher.top_k(query, k=5, rerank_context=rctx)
        assert two[0].incident_id == "inc-family"
        assert two[1].incident_id == "inc-noisy"
        assert two[0].score > two[1].score

    def test_mean_top_k_latency_with_rerank_context(self) -> None:
        matcher = FingerprintMatcher()
        mem = RemediationMemory()
        for i in range(120):
            fp = _fp("error_rate", "api", frozenset({str(j % 7) for j in range(i % 8)}))
            matcher.index(
                str(i),
                fp,
                {},
                retrieval_features=_feat(
                    deploy=0.5,
                    causal=i % 4,
                    prop=i % 3,
                    remed_hash=f"h{i % 5}",
                    fp=fp,
                ),
            )
        q = {"trigger_type": "timeout_rate", "service": "checkout-api", "upstream": ["0", "1"]}
        qfp = matcher._extractor.extract(q)
        qfeat = RetrievalFeatures.from_fingerprint(qfp)
        rctx = RerankContext(query_features=qfeat, remediation_memory=mem)
        t0 = time.perf_counter()
        for _ in range(60):
            matcher.top_k(q, k=5, rerank_context=rctx)
        ms = (time.perf_counter() - t0) / 60 * 1000.0
        assert ms < 10.0, f"mean two-stage top_k {ms:.2f}ms"
