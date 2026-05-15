"""Benchmark harness adapter — one Engine instance per adapter lifecycle."""

from __future__ import annotations

from typing import Any

from contexter import Context, Engine

_engine: Engine | None = None


def _get_engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = Engine(claude_api_key=None)
    return _engine


def reset() -> None:
    """Close and discard the engine (call between benchmark seeds)."""
    global _engine
    if _engine is not None:
        _engine.close()
        _engine = None


def ingest(events) -> None:
    """Ingest events; benchmark harness expects ``None`` return."""
    try:
        _get_engine().ingest(events)
    except Exception:
        pass
    return None


def reconstruct_context(signal: dict[str, Any], mode: str = "fast") -> Context:
    """Return incident context; never raises."""
    try:
        return _get_engine().reconstruct_context(signal, mode=mode)
    except Exception:
        from contexter.safe_context import empty_context

        return empty_context()
