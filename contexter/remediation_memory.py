"""Remediation outcome memory keyed by fingerprint hash and action."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class RemedStats:
    """Aggregate attempt and success counts for a remediation action."""

    attempts: int = 0
    successes: int = 0

    @property
    def confidence(self) -> float:
        """Return ``successes / attempts``, or ``0.0`` when ``attempts`` is zero."""
        if self.attempts == 0:
            return 0.0
        return self.successes / self.attempts


class RemediationMemory:
    """Tracks remediation outcomes per fingerprint hash and action."""

    __slots__ = ("_stats",)

    def __init__(self) -> None:
        self._stats: dict[tuple[str, str], RemedStats] = {}

    def record(
        self,
        fingerprint_hash: str,
        action: str,
        *,
        outcome: str,
    ) -> None:
        """Record one remediation attempt for ``fingerprint_hash`` and ``action``.

        Always increments ``attempts``. Increments ``successes`` only when
        ``outcome`` is ``\"resolved\"``.
        """
        key = (fingerprint_hash, action)
        stats = self._stats.get(key)
        if stats is None:
            stats = RemedStats()
            self._stats[key] = stats
        stats.attempts += 1
        if outcome == "resolved":
            stats.successes += 1

    def confidence(self, fingerprint_hash: str, action: str) -> float:
        """Return confidence for the pair, or ``0.0`` if never recorded."""
        stats = self._stats.get((fingerprint_hash, action))
        if stats is None:
            return 0.0
        return stats.confidence

    def top_actions(
        self,
        fingerprint_hash: str,
        *,
        k: int = 3,
    ) -> list[tuple[str, float]]:
        """Return up to ``k`` ``(action, confidence)`` pairs for ``fingerprint_hash``.

        Results are sorted by confidence descending.
        """
        scored: list[tuple[str, float]] = []
        for (fp_hash, action), stats in self._stats.items():
            if fp_hash != fingerprint_hash:
                continue
            scored.append((action, stats.confidence))
        scored.sort(key=lambda row: row[1], reverse=True)
        return scored[:k]
