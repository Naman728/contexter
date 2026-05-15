"""Multi-hop causal propagation along trace-learned service dependencies."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from contexter.causal_graph import CausalEdge, CausalGraph
from contexter.engine import Engine
from contexter.identity_tracker import IdentityTracker
from contexter.service_dependency_graph import parse_trace_call_edges
from tests.trace_cascade_generators import (
    cascade_timeline,
    checkout_to_database_trace,
    degraded_latency_metric,
)

UTC = timezone.utc
T0 = datetime(2026, 5, 15, 18, 0, 0, tzinfo=UTC)


def _propagation_edges(chain: list[CausalEdge]) -> list[CausalEdge]:
    out: list[CausalEdge] = []
    for edge in chain:
        ev = edge.evidence or ()
        if any(str(x).startswith("propagation:") for x in ev):
            out.append(edge)
    return out


class TestDependencyGraph:
    def test_shortest_hops_along_checkout_chain(self) -> None:
        identity = IdentityTracker()
        graph = CausalGraph(identity)
        dep = graph.dependency_graph
        for caller, callee in (
            ("frontend", "checkout-api"),
            ("checkout-api", "payments"),
            ("payments", "database"),
        ):
            dep.add_call_edge(caller, callee)
        assert dep.shortest_hops("database", "payments") == 1
        assert dep.shortest_hops("database", "frontend") == 3
        assert dep.shortest_hops("frontend", "database") == 3


class TestMultiHopCascade:
    def test_trace_payload_yields_expected_call_edges(self) -> None:
        identity = IdentityTracker()
        trace = checkout_to_database_trace(occurred_at=T0)
        edges = parse_trace_call_edges(trace["payload"], identity)  # type: ignore[arg-type]
        assert ("frontend", "checkout-api") in edges
        assert ("checkout-api", "payments") in edges
        assert ("payments", "database") in edges

    def test_cascade_survives_in_causal_chain_ordered_by_time(self) -> None:
        with Engine(batch_size=1) as engine:
            for ev in cascade_timeline(T0):
                engine.ingest_one(ev)

            ctx = engine.reconstruct_context(
                {
                    "incident_id": "inc-frontend-cascade",
                    "service": "frontend",
                    "ts": T0 + timedelta(seconds=50),
                    "trigger_type": "error_rate",
                    "upstream": ["database"],
                },
                mode="fast",
            )

            prop = _propagation_edges(ctx["causal_chain"])
            assert len(prop) >= 3, f"expected >=3 propagation edges, got {prop!r}"

            ordered = sorted(prop, key=lambda e: e.occurred_at)
            services_forward: list[str] = []
            for e in ordered:
                if not services_forward:
                    services_forward.append(e.cause_service)
                services_forward.append(e.effect_service)

            assert "database" in services_forward
            assert services_forward[-1] == "frontend"

            for e in prop:
                assert e.confidence >= 0.3
                assert any(str(x).startswith("propagation:") for x in e.evidence)

            explain = ctx["explain"].lower()
            assert "propagat" in explain

    def test_neighbor_metric_without_topology_skips_propagation(self) -> None:
        with Engine(batch_size=1) as engine:
            engine.ingest_one(
                degraded_latency_metric("api-a", occurred_at=T0 + timedelta(seconds=1))
            )
            engine.ingest_one(
                degraded_latency_metric("api-b", occurred_at=T0 + timedelta(seconds=5))
            )
            engine.ingest_one(
                {
                    "kind": "incident_signal",
                    "service": "api-b",
                    "occurred_at": T0 + timedelta(seconds=10),
                    "payload": {
                        "incident_id": "inc-iso",
                        "trigger_type": "latency",
                    },
                }
            )
            ctx = engine.reconstruct_context(
                {
                    "incident_id": "inc-iso",
                    "service": "api-b",
                    "ts": T0 + timedelta(seconds=12),
                    "trigger_type": "latency",
                },
                mode="fast",
            )
            assert _propagation_edges(ctx["causal_chain"]) == []
