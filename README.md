# Contexter

Production-grade **Union-Find identity tracker** for topology drift: when services, nodes, or resources are renamed or merged, map any alias to a single canonical identity.

## Features

- Path compression and union by size (amortized near **O(α(n))** per operation)
- `resolve(name)` — canonical representative
- `union(old, new)` — merge identities; on tie, `new` wins
- `aliases(name)` — all names in the same equivalence class
- Full type hints and minimal API surface

## Usage

```python
from contexter import IdentityTracker

tracker = IdentityTracker(["pod-abc", "pod-xyz"])

# Drift: old name merged into new
tracker.union("pod-abc", "pod-xyz")

assert tracker.resolve("pod-abc") == "pod-xyz"
assert tracker.aliases("pod-abc") == frozenset({"pod-abc", "pod-xyz"})
```

## Development

```bash
pip install -e ".[dev]"
pytest
```
