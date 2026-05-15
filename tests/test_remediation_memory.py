"""Tests for RemediationMemory and RemedStats."""

from __future__ import annotations

import pytest

from contexter.remediation_memory import RemedStats, RemediationMemory


class TestRemedStats:
    def test_confidence_zero_when_no_attempts(self) -> None:
        assert RemedStats().confidence == 0.0

    def test_confidence_ratio(self) -> None:
        assert RemedStats(attempts=4, successes=3).confidence == pytest.approx(0.75)


class TestRemediationMemory:
    def test_resolved_outcome_gives_confidence_one(self) -> None:
        memory = RemediationMemory()
        memory.record("fp-a", "restart", outcome="resolved")
        assert memory.confidence("fp-a", "restart") == 1.0

    def test_resolved_and_failed_gives_confidence_half(self) -> None:
        memory = RemediationMemory()
        memory.record("fp-a", "restart", outcome="resolved")
        memory.record("fp-a", "restart", outcome="failed")
        assert memory.confidence("fp-a", "restart") == 0.5

    def test_unknown_pair_returns_zero(self) -> None:
        memory = RemediationMemory()
        assert memory.confidence("missing", "noop") == 0.0

    def test_top_actions_descending_and_respects_k(self) -> None:
        memory = RemediationMemory()
        memory.record("fp-a", "scale", outcome="resolved")
        memory.record("fp-a", "scale", outcome="failed")
        memory.record("fp-a", "restart", outcome="resolved")
        memory.record("fp-a", "rollback", outcome="failed")

        assert memory.top_actions("fp-a", k=2) == [
            ("restart", 1.0),
            ("scale", 0.5),
        ]

    def test_top_actions_for_fingerprint_base_merges_extended_keys(self) -> None:
        memory = RemediationMemory()
        memory.record(
            "error:api:1:cache,db:P0:0:dUfUrUcU:api:api:NA",
            "rollback",
            outcome="resolved",
        )
        memory.record(
            "error:api:1:cache,db:P2_3:1:d0Uf0r0c0:db:api:abc123",
            "rollback",
            outcome="failed",
        )
        merged = memory.top_actions_for_fingerprint_base("error:api:1:cache,db", k=1)
        assert merged[0][0] == "rollback"
        assert merged[0][1] == pytest.approx(1.0 / 2.0)

        memory = RemediationMemory()
        memory.record("fp-a", "restart", outcome="resolved")
        memory.record("fp-b", "restart", outcome="failed")

        assert memory.confidence("fp-a", "restart") == 1.0
        assert memory.confidence("fp-b", "restart") == 0.0
        assert memory.top_actions("fp-b", k=3) == [("restart", 0.0)]
