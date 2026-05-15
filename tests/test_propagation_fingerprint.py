"""Propagation fingerprints for causal cascade retrieval."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from contexter.causal_graph import CausalEdge
from contexter.incident_fingerprint import (
    IncidentFingerprint,
    _EMPTY_PROPAGATION_FP,
    compute_retrieval_features,
    extract_propagation_fingerprint,
    propagation_similarity,
)

UTC = timezone.utc
T0 = datetime(2026, 7, 1, 12, 0, 0, tzinfo=UTC)


def _prop_edge(
    *,
    cause: str,
    effect: str,
    hop: int,
    kind: str = "metric",
    ts: datetime,
    cid: str = "c1",
    eid: str = "e1",
) -> CausalEdge:
    return CausalEdge(
        cause_id=cid,
        effect_id=eid,
        evidence=[
            f"propagation:{hop}",
            f"from:{cause}",
            f"to:{effect}",
            f"pair:{kind}",
        ],
        confidence=0.8,
        occurred_at=ts,
        cause_service=cause,
        effect_service=effect,
    )


def test_multi_hop_cascades_match_strongly() -> None:
    edges_a = [
        _prop_edge(
            cause="database",
            effect="payments",
            hop=1,
            ts=T0,
            cid="a1",
            eid="a2",
        ),
        _prop_edge(
            cause="payments",
            effect="checkout-api",
            hop=1,
            ts=T0,
            cid="a2",
            eid="a3",
        ),
        _prop_edge(
            cause="checkout-api",
            effect="frontend",
            hop=1,
            ts=T0,
            cid="a3",
            eid="a4",
        ),
    ]
    edges_b = [
        _prop_edge(
            cause="database",
            effect="payments",
            hop=1,
            ts=T0 + timedelta(seconds=2),
            cid="b1",
            eid="b2",
        ),
        _prop_edge(
            cause="payments",
            effect="checkout-api",
            hop=1,
            ts=T0 + timedelta(seconds=3),
            cid="b2",
            eid="b3",
        ),
        _prop_edge(
            cause="checkout-api",
            effect="frontend",
            hop=1,
            ts=T0 + timedelta(seconds=4),
            cid="b3",
            eid="b4",
        ),
    ]
    pa = extract_propagation_fingerprint(edges_a)
    pb = extract_propagation_fingerprint(edges_b)
    assert pa.degradation_order == ("database", "payments", "checkout-api", "frontend")
    assert pa.hop_count == 3
    assert propagation_similarity(pa, pb) >= 0.88


def test_shallow_vs_deep_cascade_scores_low() -> None:
    shallow = extract_propagation_fingerprint(
        [
            _prop_edge(
                cause="cache",
                effect="api",
                hop=1,
                ts=T0,
                cid="s1",
                eid="s2",
            ),
        ]
    )
    deep = extract_propagation_fingerprint(
        [
            _prop_edge(cause="db", effect="pay", hop=2, ts=T0, cid="d1", eid="d2"),
            _prop_edge(cause="pay", effect="co", hop=2, ts=T0, cid="d2", eid="d3"),
            _prop_edge(cause="co", effect="fe", hop=2, ts=T0, cid="d3", eid="d4"),
            _prop_edge(cause="fe", effect="gw", hop=1, ts=T0, cid="d4", eid="d5"),
        ]
    )
    assert shallow.hop_count == 1
    assert deep.hop_count == 4
    sim = propagation_similarity(shallow, deep)
    assert sim < 0.48


def test_unrelated_topologies_score_low() -> None:
    a = extract_propagation_fingerprint(
        [
            _prop_edge(
                cause="redis",
                effect="cart-api",
                hop=1,
                kind="metric",
                ts=T0,
                cid="x1",
                eid="x2",
            ),
            _prop_edge(
                cause="cart-api",
                effect="web",
                hop=1,
                kind="metric",
                ts=T0,
                cid="x2",
                eid="x3",
            ),
        ]
    )
    b = extract_propagation_fingerprint(
        [
            _prop_edge(
                cause="kafka",
                effect="worker",
                hop=1,
                kind="log",
                ts=T0,
                cid="y1",
                eid="y2",
            ),
            _prop_edge(
                cause="worker",
                effect="batch",
                hop=1,
                kind="log",
                ts=T0,
                cid="y2",
                eid="y3",
            ),
        ]
    )
    assert propagation_similarity(a, b) < 0.46


def test_compute_retrieval_features_embeds_propagation_fingerprint() -> None:
    fp = IncidentFingerprint("latency", "frontend", frozenset(), "frontend")
    edges = [
        _prop_edge(cause="db", effect="frontend", hop=2, ts=T0, cid="1", eid="2"),
    ]
    rf = compute_retrieval_features(fp, edges, None, "frontend", T0)
    assert rf.propagation_fingerprint.hop_count == 1
    assert "db" in rf.propagation_fingerprint.degradation_order


def test_empty_snapshots_neutral_similarity() -> None:
    assert propagation_similarity(_EMPTY_PROPAGATION_FP, _EMPTY_PROPAGATION_FP) == 1.0
    one = extract_propagation_fingerprint(
        [_prop_edge(cause="a", effect="b", hop=1, ts=T0, cid="1", eid="2")]
    )
    assert propagation_similarity(one, _EMPTY_PROPAGATION_FP) < 0.2


def test_ordered_service_path_outranks_different_cascade_to_same_frontend() -> None:
    """Same terminal role; ordered db→payments→checkout path should beat cache→api→frontend."""
    gold = extract_propagation_fingerprint(
        [
            _prop_edge(cause="database", effect="payments", hop=1, ts=T0, cid="g1", eid="g2"),
            _prop_edge(cause="payments", effect="checkout-api", hop=1, ts=T0, cid="g2", eid="g3"),
            _prop_edge(cause="checkout-api", effect="frontend", hop=1, ts=T0, cid="g3", eid="g4"),
        ]
    )
    noise = extract_propagation_fingerprint(
        [
            _prop_edge(cause="redis-cache", effect="api-svc", hop=1, ts=T0, cid="n1", eid="n2"),
            _prop_edge(cause="api-svc", effect="frontend", hop=1, ts=T0, cid="n2", eid="n3"),
        ]
    )
    assert propagation_similarity(gold, gold) > propagation_similarity(gold, noise)
