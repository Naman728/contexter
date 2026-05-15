"""Remediation outcome memory keyed by fingerprint hash and action."""

from __future__ import annotations

from collections import defaultdict

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

    def top_actions_for_fingerprint_base(
        self,
        base_key: str,
        *,
        k: int = 3,
    ) -> list[tuple[str, float]]:
        """Merge actions across extended hashes that share the same structural ``base_key``."""
        prefix = base_key + ":"
        by_action: dict[str, list[RemedStats]] = defaultdict(list)
        for (fp_hash, action), stats in self._stats.items():
            if fp_hash == base_key or fp_hash.startswith(prefix):
                by_action[action].append(stats)
        scored: list[tuple[str, float]] = []
        for action, lst in by_action.items():
            att = sum(s.attempts for s in lst)
            suc = sum(s.successes for s in lst)
            conf = suc / att if att else 0.0
            scored.append((action, conf))
        scored.sort(key=lambda row: row[1], reverse=True)
        return scored[:k]

    def action_keys_for_hash(self, fingerprint_hash: str) -> frozenset[str]:
        """Distinct remediation actions ever recorded for ``fingerprint_hash``."""
        return frozenset(
            action for (fp_hash, action) in self._stats if fp_hash == fingerprint_hash
        )
