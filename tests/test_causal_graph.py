"""Tests for CausalGraph and CausalEdge."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from contexter.causal_graph import CausalEdge, CausalGraph

UTC = timezone.utc
T0 = datetime(2026, 5, 15, 12, 0, 0, tzinfo=UTC)


class TestRecordEffect:
    def test_deploy_then_effect_within_120s_linear_confidence(self) -> None:
        graph = CausalGraph()
        graph.record_deploy("deploy-1", "api", T0)
        edges = graph.record_effect(
            "effect-1",
            "api",
            T0 + timedelta(seconds=60),
            kind="error_log",
        )

        assert len(edges) == 1
        assert edges[0].cause_id == "deploy-1"
        assert edges[0].effect_id == "effect-1"
        assert edges[0].confidence == pytest.approx(0.5)
        assert "kind:error_log" in edges[0].evidence

    def test_deploy_then_effect_outside_120s_no_edge(self) -> None:
        graph = CausalGraph()
        graph.record_deploy("deploy-1", "api", T0)
        edges = graph.record_effect(
            "effect-1",
            "api",
            T0 + timedelta(seconds=121),
            kind="metric",
        )
        assert edges == []

    def test_confidence_never_negative(self) -> None:
        graph = CausalGraph()
        graph.record_deploy("deploy-1", "api", T0)
        edges = graph.record_effect(
            "effect-1",
            "api",
            T0 + timedelta(seconds=120),
            kind="metric",
        )
        assert len(edges) == 1
        assert edges[0].confidence == 0.0

        edges = graph.record_effect(
            "effect-2",
            "api",
            T0 + timedelta(seconds=200),
            kind="metric",
        )
        assert edges == []

        edges = graph.record_effect(
            "effect-3",
            "api",
            T0 + timedelta(seconds=60),
            kind="metric",
        )
        assert edges[0].confidence == pytest.approx(0.5)
        assert edges[0].confidence >= 0.0

    def test_no_reverse_direction_edge(self) -> None:
        graph = CausalGraph()
        graph.record_deploy("deploy-1", "api", T0 + timedelta(seconds=60))
        edges = graph.record_effect("effect-1", "api", T0, kind="error_log")
        assert edges == []

        graph.record_deploy("deploy-2", "api", T0)
        edges = graph.record_effect("effect-2", "api", T0, kind="error_log")
        assert edges == []


class TestSnapshotAndQuery:
    def test_snapshot_only_captures_window(self) -> None:
        graph = CausalGraph()
        graph.record_deploy("d1", "api", T0)
        graph.record_effect("e1", "api", T0 + timedelta(seconds=30), kind="error_log")
        graph.record_deploy("d2", "api", T0 + timedelta(seconds=200))
        graph.record_effect("e2", "api", T0 + timedelta(seconds=230), kind="metric")

        incident_ts = T0 + timedelta(seconds=240)
        graph.snapshot_incident("inc-1", incident_ts, "api", window_s=300)

        snap = graph.edges_for_incident("inc-1")
        assert len(snap) == 2
        assert {e.effect_id for e in snap} == {"e1", "e2"}

        graph.snapshot_incident("inc-2", T0 + timedelta(seconds=45), "api", window_s=30)
        snap_narrow = graph.edges_for_incident("inc-2")
        assert len(snap_narrow) == 1
        assert snap_narrow[0].effect_id == "e1"

    def test_edges_for_incident_sorted_ascending(self) -> None:
        graph = CausalGraph()
        graph.record_deploy("d1", "api", T0)
        graph.record_effect("e1", "api", T0 + timedelta(seconds=30), kind="error_log")
        graph.record_deploy("d2", "api", T0 + timedelta(seconds=40))
        graph.record_effect("e2", "api", T0 + timedelta(seconds=90), kind="metric")

        graph.snapshot_incident("inc-1", T0 + timedelta(seconds=120), "api")
        ordered = graph.edges_for_incident("inc-1")

        assert ordered[0].effect_id == "e1"
        assert ordered[-1].effect_id == "e2"
        for earlier, later in zip(ordered, ordered[1:]):
            assert earlier.occurred_at <= later.occurred_at

    def test_edges_for_unknown_incident_empty(self) -> None:
        graph = CausalGraph()
        assert graph.edges_for_incident("missing") == []
