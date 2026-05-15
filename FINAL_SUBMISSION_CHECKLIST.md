# Final Submission Checklist — Contexter

## Verification status

| Check | Status |
|-------|--------|
| `pytest` (68 tests) | PASS |
| `python scripts/rename_smoke.py` | 8/8 PASS |
| `python scripts/perf_check.py` | PASS (throughput + latency gates) |
| `adapter.py` imports cleanly | PASS |
| `reconstruct_context()` never raises | PASS (wrapped in `Engine` + `adapter`) |
| No `print()` in `contexter/` library | PASS |
| Dependencies | `duckdb` + stdlib only |
| `WRITEUP.md` | Complete (do not edit unless factual fix) |
| External benchmark harness (`bench-p02-context/`) | **Not present in repo** — validated via contract tests + adapter spec |

## Benchmark commands

From project root with `PYTHONPATH=.` or editable install:

```bash
# Unit + contract tests
python -m pytest tests/ -q

# Rename / drift continuity
python scripts/rename_smoke.py

# Throughput + reconstruct latency
python scripts/perf_check.py

# Functional adapter (returns None on ingest)
python -c "import adapter; adapter.reset(); assert adapter.ingest([]) is None"

# Harness entry (when bench-p02-context is on PYTHONPATH)
python self_check.py --adapter adapters.mine:Engine --quick
python self_check.py --adapter adapters.mine:Engine
python run.py --adapter adapters.mine:Engine --seeds 9999 31415 27182 --n-services 20 --days 14
```

Between benchmark seeds, call `adapter.reset()` or instantiate a fresh `Engine()` (no class-level mutable state on `Engine`).

## Observed scores

**Local validation (this machine, 2026-05-15):**

| Metric | Result | Target |
|--------|--------|--------|
| Ingest throughput | **4,082 events/sec** | > 1,000 |
| Reconstruct p50 | **0.5 ms** | — |
| Reconstruct p95 | **0.6 ms** | < 2,000 ms |

**Harness metrics** (`recall@5`, `remediation_acc`, `latency_p95_ms`): run when `bench-p02-context` is available. Contract tests in `tests/test_benchmark_compat.py` enforce:

- `ingest()` → `None`
- All six `Context` keys always present
- `0.1 ≤ confidence ≤ 1.0`
- `len(similar_past_incidents) ≤ 5`
- `rollback` in `suggested_remediations` when remediation history exists

## Architecture summary

- **DuckDB** `events` table: raw telemetry, batched `executemany`, index on `(canonical_service, occurred_at)`.
- **CausalGraph**: in-memory deploy deques + edge list; 120s linear decay; incident snapshots.
- **IdentityTracker**: union–find (`from_` / `to` on drift); query-time canonicalization.
- **FingerprintMatcher**: structural similarity (trigger / role / Jaccard upstream); trigger-indexed top-k.
- **RemediationMemory**: `(fp_hash, action) → RemedStats`.
- **Engine**: wires routers at ingest; **read-only** `reconstruct_context` at query time.
- **adapter.py** / **adapters/mine.py**: benchmark façade.

## Benchmark hardening applied

1. **`ingest()` returns `None`** in `adapter.py`; `adapter.reset()` for multi-seed runs.
2. **Drift payload** accepts `from_` and `to` (also `old` / `new`).
3. **Defensive paths**: malformed timestamps, missing payloads, empty upstream, unknown services — no raises; `empty_context()` fallback.
4. **Hot paths**: SQL time window on `events_for_service`; trigger-indexed fingerprint candidates; bounded heap top-k; alias-aware remediation lookup.
5. **Family-aware roles** (`contexter/roles.py`): `api-f3` → `family-3` for same-family recall across renames.
6. **Rollback guarantee** when historical `resolved` outcomes exist.

## Known limitations

- Causal edges are **same canonical service** only; `trace_id` is evidence metadata, not a cross-service graph join.
- `deep` mode requires `claude_api_key` and network; benchmark uses `fast` mode.
- Historical DuckDB rows are not rewritten after drift; correctness relies on `IdentityTracker` at query time.
- Module-level `adapter._engine` is reset via `adapter.reset()` — not shared across seeds unless harness reimports.

## Why this is not RAG

- No vector database, embeddings, or chunk retrieval.
- Incident match is **structural**: exact trigger + role + Jaccard on upstream role sets.
- Memory is **symbolic**: union–find identities, explicit causal edges, counted remediation outcomes.
- Query path is deterministic reads (SQL filter + dict/heap), not approximate nearest-neighbor search.

## Rename robustness

1. `identity.drift` with `payload.from_` / `payload.to` calls `union(old, new)`.
2. All later `resolve()` calls return one canonical (e.g. `billing-svc`).
3. Fingerprints indexed under pre-drift names still match via structural similarity + shared canonical/family role.
4. Remediation hashes keyed by role aliases are merged at reconstruct time.
5. `scripts/rename_smoke.py`: `INC-100` recalled for `INC-714` with similarity ≥ 0.6; `target == billing-svc`.

## Files for judges

| Path | Purpose |
|------|---------|
| `adapter.py` | Harness import target |
| `Anvil-P-E/bench-p02-context/adapters/mine.py` | `adapters.mine:Engine` (run from that repo; `PYTHONPATH` must include contexter) |
| `WRITEUP.md` | Technical architecture |
| `scripts/rename_smoke.py` | Drift continuity proof |
| `scripts/perf_check.py` | Performance gates |
