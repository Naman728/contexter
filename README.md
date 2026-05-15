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

### P-02 organizer harness (`run.py`)

The benchmark driver expects the official harness directory (files such as `harness.py`, `generator.py`, `schema.py`, `metrics.py`, `adapter.py`). The **contexter** repo root does not include those files, so either set `P02_BENCH` or pass `--bench` to that directory. From this repo root, after `pip install -e .`:

```bash
export P02_BENCH=/absolute/path/to/p02-harness-directory
python run.py --adapter adapters.myteam:Engine --mode fast \
  --seeds 9999 31415 27182 16180 11235 \
  --n-services 20 --days 14 --out report.json
```

Same run without `export` (handy from the contexter directory):

```bash
python run.py --bench /absolute/path/to/p02-harness-directory \
  --adapter adapters.myteam:Engine --mode fast \
  --seeds 9999 31415 27182 16180 11235 \
  --n-services 20 --days 14 --out report.json
```

If that harness tree is merged next to `run.py` so `harness.py` sits in the same directory, `P02_BENCH` is not required. The submission adapter is `adapters/myteam.py` (`Engine` class).
