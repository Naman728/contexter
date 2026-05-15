"""Union-Find identity tracker for topology drift handling."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Final


class IdentityTracker:
    """Tracks entity identities across renames and merges using disjoint-set union.

    Supports path compression and union by size for amortized near-O(1)
    ``resolve`` and ``union``. When two groups tie in size, the representative
    of ``new`` becomes the canonical root (topology drift semantics).
    """

    __slots__ = ("_parent", "_size", "_aliases")

    def __init__(self, names: Iterable[str] | None = None) -> None:
        self._parent: dict[str, str] = {}
        self._size: dict[str, int] = {}
        self._aliases: dict[str, set[str]] = {}
        if names is not None:
            for name in names:
                self.register(name)

    def register(self, name: str) -> str:
        """Introduce a standalone identity. Returns its canonical representative."""
        self._ensure(name)
        return self._find(name)

    def resolve(self, name: str) -> str:
        """Return the canonical representative for ``name``."""
        return self._find(name)

    def union(self, old: str, new: str) -> str:
        """Merge the identity of ``old`` into ``new``. Returns the canonical root."""
        self._ensure(old)
        self._ensure(new)

        root_old = self._find(old)
        root_new = self._find(new)

        if root_old == root_new:
            return root_old

        if self._size[root_old] < self._size[root_new]:
            self._attach(root_old, root_new)
            return root_new

        if self._size[root_old] > self._size[root_new]:
            self._attach(root_new, root_old)
            return root_old

        # Equal size: prefer ``new`` as canonical (drift: old -> new).
        self._attach(root_old, root_new)
        return root_new

    def aliases(self, name: str) -> frozenset[str]:
        """All names equivalent to ``name`` (including the canonical root)."""
        root = self._find(name)
        return frozenset(self._aliases[root])

    def equivalent(self, a: str, b: str) -> bool:
        """True if ``a`` and ``b`` belong to the same identity group."""
        self._ensure(a)
        self._ensure(b)
        return self._find(a) == self._find(b)

    def canonical(self, name: str) -> str:
        """Alias for :meth:`resolve`."""
        return self.resolve(name)

    def groups(self) -> Iterator[frozenset[str]]:
        """Yield each disjoint identity group."""
        seen: set[str] = set()
        for name in self._parent:
            root = self._find(name)
            if root in seen:
                continue
            seen.add(root)
            yield frozenset(self._aliases[root])

    def __contains__(self, name: str) -> bool:
        return name in self._parent

    def __len__(self) -> int:
        return len(self._parent)

    def _ensure(self, name: str) -> None:
        if name not in self._parent:
            self._parent[name] = name
            self._size[name] = 1
            self._aliases[name] = {name}

    def _find(self, name: str) -> str:
        parent = self._parent[name]
        if parent != name:
            self._parent[name] = self._find(parent)
        return self._parent[name]

    def _attach(self, child_root: str, parent_root: str) -> None:
        self._parent[child_root] = parent_root
        self._size[parent_root] += self._size[child_root]
        self._aliases[parent_root] |= self._aliases.pop(child_root)
