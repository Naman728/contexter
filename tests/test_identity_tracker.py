"""Tests for IdentityTracker."""

from __future__ import annotations

import pytest

from contexter.identity_tracker import IdentityTracker


class TestResolve:
    def test_unregistered_raises_key_error(self) -> None:
        tracker = IdentityTracker()
        with pytest.raises(KeyError):
            tracker.resolve("missing")

    def test_register_returns_self(self) -> None:
        tracker = IdentityTracker()
        assert tracker.register("a") == "a"
        assert tracker.resolve("a") == "a"

    def test_path_compression_after_chain(self) -> None:
        tracker = IdentityTracker()
        tracker.union("a", "b")
        tracker.union("b", "c")
        assert tracker.resolve("a") == tracker.resolve("c")
        # After resolves, parent should point near root.
        root = tracker.resolve("a")
        assert tracker._parent["a"] == root


class TestUnion:
    def test_union_merges_aliases(self) -> None:
        tracker = IdentityTracker(["svc-v1", "svc-v2"])
        root = tracker.union("svc-v1", "svc-v2")
        assert root == "svc-v2"
        assert tracker.equivalent("svc-v1", "svc-v2")
        assert tracker.aliases("svc-v1") == frozenset({"svc-v1", "svc-v2"})

    def test_union_idempotent(self) -> None:
        tracker = IdentityTracker(["x", "y"])
        r1 = tracker.union("x", "y")
        r2 = tracker.union("x", "y")
        assert r1 == r2
        assert len(list(tracker.groups())) == 1

    def test_transitive_union(self) -> None:
        tracker = IdentityTracker(["a", "b", "c"])
        tracker.union("a", "b")
        tracker.union("b", "c")
        assert tracker.resolve("a") == tracker.resolve("c")
        assert tracker.aliases("a") == frozenset({"a", "b", "c"})

    def test_equal_size_prefers_new_canonical(self) -> None:
        tracker = IdentityTracker(["old", "new"])
        assert tracker.union("old", "new") == "new"
        assert tracker.resolve("old") == "new"

    def test_larger_group_wins_by_size(self) -> None:
        tracker = IdentityTracker()
        tracker.register("hub")
        tracker.register("leaf-a")
        tracker.register("leaf-b")
        tracker.union("leaf-a", "hub")
        tracker.union("leaf-b", "hub")
        # hub's component has size 3; tiny "new" should attach under hub
        tracker.register("tiny")
        root = tracker.union("tiny", "hub")
        assert root == "hub"
        assert tracker.resolve("tiny") == "hub"


class TestAliases:
    def test_aliases_after_multiple_unions(self) -> None:
        tracker = IdentityTracker(["n1", "n2", "n3", "n4"])
        tracker.union("n1", "n2")
        tracker.union("n3", "n4")
        tracker.union("n2", "n3")
        assert tracker.aliases("n1") == frozenset({"n1", "n2", "n3", "n4"})

    def test_groups_iterator(self) -> None:
        tracker = IdentityTracker(["a", "b", "c", "d"])
        tracker.union("a", "b")
        groups = set(tracker.groups())
        assert groups == {frozenset({"a", "b"}), frozenset({"c"}), frozenset({"d"})}


class TestContainer:
    def test_contains_and_len(self) -> None:
        tracker = IdentityTracker(["x"])
        assert "x" in tracker
        assert "y" not in tracker
        assert len(tracker) == 1
        tracker.register("y")
        assert len(tracker) == 2
