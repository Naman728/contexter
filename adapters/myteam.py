"""Official P-02 submission adapter: ``adapters.myteam:Engine`` for ``run.py``."""

from __future__ import annotations

from typing import Any, Literal, cast

from adapter import Adapter
from schema import Context, Event, IncidentSignal

from contexter import Engine as ContextEngine


def _bench_event_to_contexter(ev: Event) -> dict[str, Any]:
    """Map harness flat ``Event`` into contexter ``MemorySubstrate`` ingest shape."""
    kind = str(ev.get("kind", "unknown"))
    ts = ev.get("ts")
    service = str(ev.get("service", ev.get("target", "unknown")))

    if kind == "topology":
        change = ev.get("change")
        if change == "rename":
            to_live = str(ev.get("to", ev.get("to_", service)))
            return {
                "kind": "identity.drift",
                "service": to_live,
                "occurred_at": ts,
                "payload": {
                    "from_": str(ev.get("from_", "")),
                    "to": to_live,
                },
            }
        return {
            "kind": "topology",
            "service": service,
            "occurred_at": ts,
            "payload": {
                "change": change,
                "from_": str(ev.get("from_", "")),
                "to": str(ev.get("to", ev.get("to_", ""))),
            },
        }

    if kind == "metric":
        payload = {k: ev[k] for k in ("name", "value", "trace_id") if k in ev}
        return {"kind": "metric", "service": service, "occurred_at": ts, "payload": payload or None}

    if kind == "log":
        payload = {k: ev[k] for k in ("level", "msg", "trace_id") if k in ev}
        return {"kind": "log", "service": service, "occurred_at": ts, "payload": payload or None}

    if kind == "deploy":
        payload = {k: ev[k] for k in ("version", "actor") if k in ev}
        return {"kind": "deploy", "service": service, "occurred_at": ts, "payload": payload or None}

    if kind == "incident_signal":
        return {
            "kind": "incident_signal",
            "service": service,
            "occurred_at": ts,
            "payload": {
                "incident_id": str(ev.get("incident_id", "")),
                "trigger_type": str(
                    ev.get("trigger", ev.get("trigger_type", "unknown"))
                ),
                "upstream": ev.get("upstream", []),
            },
        }

    if kind == "remediation":
        payload = {
            k: ev[k]
            for k in ("incident_id", "action", "outcome", "target", "version")
            if k in ev
        }
        return {"kind": "remediation", "service": service, "occurred_at": ts, "payload": payload}

    return {
        "kind": kind,
        "service": service,
        "occurred_at": ts,
        "payload": {k: v for k, v in ev.items() if k not in ("kind", "service", "ts")},
    }


def _signal_to_engine(sig: IncidentSignal) -> dict[str, Any]:
    """Harness ``trigger`` → contexter ``trigger_type``."""
    return {
        "incident_id": str(sig.get("incident_id", "")),
        "ts": sig.get("ts"),
        "service": str(sig.get("service", "")),
        "trigger_type": str(sig.get("trigger", sig.get("trigger_type", "unknown"))),
        "upstream": list(sig.get("upstream", [])) if isinstance(sig.get("upstream"), list) else [],
    }


def _harness_context(raw: dict[str, Any]) -> Context:
    """Align contexter output with ``schema.Context`` / ``metrics`` expectations."""
    out = dict(raw)
    sim: list[dict[str, Any]] = []
    for m in out.get("similar_past_incidents") or []:
        if not isinstance(m, dict):
            continue
        sim.append(
            {
                "incident_id": str(m.get("past_incident_id", m.get("incident_id", ""))),
                "similarity": float(m.get("similarity", 0.0)),
                "rationale": str(m.get("rationale", "")),
            }
        )
    out["similar_past_incidents"] = cast(list[Any], sim)

    chain: list[dict[str, Any]] = []
    for edge in out.get("causal_chain") or []:
        if hasattr(edge, "cause_id"):
            evl = getattr(edge, "evidence", [])
            evidence = ",".join(evl) if isinstance(evl, list) else str(evl)
            chain.append(
                {
                    "cause_event_id": str(getattr(edge, "cause_id", "")),
                    "effect_event_id": str(getattr(edge, "effect_id", "")),
                    "evidence": evidence,
                    "confidence": float(getattr(edge, "confidence", 0.0)),
                }
            )
        elif isinstance(edge, dict):
            chain.append(edge)
    out["causal_chain"] = cast(list[Any], chain)

    return cast(Context, out)


class Engine(Adapter):
    """Benchmark-facing adapter; delegates to :class:`contexter.engine.Engine`."""

    def __init__(self) -> None:
        self._engine = ContextEngine()

    def ingest(self, events) -> None:
        normalized = [_bench_event_to_contexter(cast(Event, e)) for e in events]
        self._engine.ingest(normalized)

    def reconstruct_context(
        self,
        signal: IncidentSignal,
        mode: Literal["fast", "deep"] = "fast",
    ) -> Context:
        raw = self._engine.reconstruct_context(
            _signal_to_engine(signal),
            mode=mode,
        )
        return _harness_context(dict(raw))

    def close(self) -> None:
        self._engine.close()
