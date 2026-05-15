"""Online adaptive retrieval weights and audit logging."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from contexter.engine import Engine
from contexter.incident_fingerprint import (
    IncidentFingerprint,
    MatchResult,
    RetrievalFeatures,
    RerankContext,
    _EMPTY_PROPAGATION_FP,
    rerank_component_values,
)
from contexter.remediation_memory import RemediationMemory
from contexter.retrieval_adaptation import RetrievalAdaptation, RetrievalWeightState

UTC = timezone.utc
T0 = datetime(2026, 11, 1, 10, 0, 0, tzinfo=UTC)


def _rf(
    *,
    trigger: str = "error",
    role: str = "payments-api",
    canon: str = "payments-api",
    upstream_roles: frozenset[str],
    deploy: float = 0.0,
    edges: int = 0,
    prop: int = 0,
) -> RetrievalFeatures:
    return RetrievalFeatures(
        trigger,
        role,
        canon,
        upstream_roles,
        deploy,
        edges,
        prop,
        f"{trigger}:{role}:True",
        (),
        _EMPTY_PROPAGATION_FP,
    )


class StubMatcher:
    def __init__(self, feats: dict[str, RetrievalFeatures]) -> None:
        self._feats = feats

    def retrieval_features_for_incident(self, incident_id: str) -> RetrievalFeatures | None:
        return self._feats.get(incident_id)


def test_adaptive_weights_evolve_on_success() -> None:
    qfp = IncidentFingerprint(
        "error_rate",
        "checkout-api",
        frozenset({"db", "cache"}),
        "checkout-api",
    )
    qfeat = _rf(
        role="checkout-api",
        canon="checkout-api",
        upstream_roles=frozenset({"db", "cache"}),
    )
    fp_wrong = IncidentFingerprint(
        "error_rate",
        "checkout-api",
        frozenset({"x"}),
        "checkout-api",
    )
    fp_right = IncidentFingerprint(
        "error_rate",
        "checkout-api",
        frozenset({"db", "cache"}),
        "checkout-api",
    )
    matcher = StubMatcher(
        {
            "INC-WRONG": _rf(
                role="checkout-api",
                canon="checkout-api",
                upstream_roles=frozenset({"x"}),
                deploy=0.9,
                prop=0,
            ),
            "INC-RIGHT": _rf(
                role="checkout-api",
                canon="checkout-api",
                upstream_roles=frozenset({"db", "cache"}),
                deploy=0.0,
                prop=2,
            ),
        }
    )
    rem = RemediationMemory()
    rctx = RerankContext(query_features=qfeat, remediation_memory=rem)
    adapt = RetrievalAdaptation(
        weights=RetrievalWeightState(
            w_trigger=0.25,
            w_role=0.15,
            w_upstream=0.10,
            w_propagation=0.20,
            w_temporal=0.15,
        )
    )
    w_upstream_0 = adapt.weight_map()["upstream"]
    matches = [
        MatchResult("INC-WRONG", fp_wrong, 0.72, None),
        MatchResult("INC-RIGHT", fp_right, 0.55, None),
    ]
    for _ in range(8):
        adapt.record_retrieval(
            signal={
                "incident_id": "INC-Q",
                "_retrieval_expected_incident_id": "INC-RIGHT",
            },
            query_fp=qfp,
            query_features=qfeat,
            canonical_service="checkout-api",
            matches=matches,
            matcher=matcher,
            rerank_ctx=rctx,
        )
    w_upstream_1 = adapt.weight_map()["upstream"]
    assert adapt.stats()["n_inversion_weight_updates"] >= 8
    assert adapt.stats()["n_success_weight_updates"] >= 8
    assert w_upstream_1 > w_upstream_0


def test_recall_proxy_improves_similarity_for_expected() -> None:
    """Repeated successful audits with fixed matches nudge weights; best expected score can rise."""
    with Engine(batch_size=1) as eng:
        eng.ingest_one(
            {
                "kind": "incident_signal",
                "service": "svc-a",
                "occurred_at": T0,
                "payload": {
                    "incident_id": "INC-PAST",
                    "trigger_type": "latency",
                    "upstream": ["redis", "db"],
                },
            }
        )
        eng.ingest_one(
            {
                "kind": "incident_signal",
                "service": "svc-a",
                "occurred_at": T0,
                "payload": {
                    "incident_id": "INC-CURR",
                    "trigger_type": "latency",
                    "upstream": ["redis", "db"],
                },
            }
        )
        sig = {
            "incident_id": "INC-CURR",
            "service": "svc-a",
            "ts": T0,
            "trigger_type": "latency",
            "upstream": ["redis", "db"],
            "_retrieval_expected_incident_id": "INC-PAST",
        }
        sims = []
        for _ in range(40):
            ctx = eng.reconstruct_context(sig, mode="fast")
            sims.append(
                max(
                    (float(m["similarity"]) for m in ctx["similar_past_incidents"]),
                    default=0.0,
                )
            )
        assert sims[-1] >= sims[0] - 1e-6
        assert eng.retrieval_stats()["n_audits"] == 40


def test_bounded_weights_no_instability_under_random_feedback() -> None:
    import random

    qfp = IncidentFingerprint("t", "r", frozenset({"u"}), "c")
    qfeat = RetrievalFeatures.from_fingerprint(qfp)
    rem = RemediationMemory()
    rctx = RerankContext(query_features=qfeat, remediation_memory=rem)
    adapt = RetrievalAdaptation()
    rng = random.Random(42)
    for i in range(160):
        fp = IncidentFingerprint(
            f"tr-{i % 3}",
            "r2" if i % 2 else "r",
            frozenset({"a"} if i % 5 else {"u"}),
            "c2",
        )
        rf = RetrievalFeatures.from_fingerprint(fp)
        m = StubMatcher({"X": rf})
        success = rng.random() > 0.55
        sig = {"incident_id": "Q", "_retrieval_expected_incident_id": ("X" if success else "MISS")}
        matches = [MatchResult("X", fp, 0.4 + 0.5 * rng.random(), None)]
        adapt.record_retrieval(
            signal=sig,
            query_fp=qfp,
            query_features=qfeat,
            canonical_service="c",
            matches=matches,
            matcher=m,
            rerank_ctx=rctx,
        )
        w = adapt.weight_map()
        for k in ("trigger", "role", "upstream", "propagation", "temporal"):
            assert 0.06 <= w[k] <= 0.44
        assert abs(sum(w[k] for k in w) - 0.85) < 0.02


def test_audit_log_shape_and_retrieval_stats() -> None:
    adapt = RetrievalAdaptation()
    qfp = IncidentFingerprint("e", "api", frozenset(), "api")
    qfeat = RetrievalFeatures.from_fingerprint(qfp)
    rem = RemediationMemory()
    rctx = RerankContext(query_features=qfeat, remediation_memory=rem)
    m = StubMatcher({})
    adapt.record_retrieval(
        signal={"incident_id": "q1"},
        query_fp=qfp,
        query_features=qfeat,
        canonical_service="api",
        matches=[],
        matcher=m,
        rerank_ctx=rctx,
    )
    audits = adapt.recent_audits(5)
    assert len(audits) == 1
    assert audits[0]["family_hit"] is False
    assert audits[0]["matches"] == []
    st = adapt.stats()
    assert st["n_audits"] == 1
    assert "failure_mode_counts" in st
    assert "recall_trend_mean" in st


def test_retrieval_snapshot_roundtrip() -> None:
    a = RetrievalAdaptation()
    qfp = IncidentFingerprint("e", "api", frozenset({"db"}), "api")
    qfeat = RetrievalFeatures.from_fingerprint(qfp)
    rem = RemediationMemory()
    rctx = RerankContext(query_features=qfeat, remediation_memory=rem)
    m = StubMatcher({})
    a.record_retrieval(
        signal={"incident_id": "q", "_retrieval_expected_incident_id": "X"},
        query_fp=qfp,
        query_features=qfeat,
        canonical_service="api",
        matches=[],
        matcher=m,
        rerank_ctx=rctx,
    )
    snap = a.snapshot()
    b = RetrievalAdaptation()
    b.restore_snapshot(snap)
    assert b.stats()["n_audits"] == a.stats()["n_audits"]
    assert b.weight_map() == a.weight_map()
    assert b.stats()["component_precision_ema"] == a.stats()["component_precision_ema"]


def test_engine_retrieval_snapshot_roundtrip() -> None:
    with Engine(batch_size=1) as eng:
        eng.restore_retrieval_snapshot(eng.retrieval_snapshot())
        assert eng.retrieval_stats()["n_audits"] == 0


def test_rerank_component_values_contract() -> None:
    qfp = IncidentFingerprint("latency", "a", frozenset({"b"}), "s")
    cfp = IncidentFingerprint("latency", "a", frozenset({"b"}), "s")
    qf = RetrievalFeatures.from_fingerprint(qfp)
    cf = RetrievalFeatures.from_fingerprint(cfp)
    rem = RemediationMemory()
    ctx = RerankContext(query_features=qf, remediation_memory=rem)
    v = rerank_component_values(qfp, qf, cfp, cf, ctx)
    assert v["trigger"] == 1.0 and v["upstream"] == 1.0
