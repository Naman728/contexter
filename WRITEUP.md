# Memory Representation

Contexter stores operational memory in three layers with distinct access patterns: an append-oriented event log (DuckDB), a derived causal overlay (Python `CausalGraph`), and a mutable identity map (`IdentityTracker`). A single shared `IdentityTracker` instance is wired into `MemorySubstrate`, `CausalGraph`, `FingerprintExtractor`, and `Engine` so every layer agrees on canonical service names.

## DuckDB events table

Raw observability events land in `MemorySubstrate`, backed by an in-memory DuckDB connection (`duckdb.connect(":memory:")`). The DDL executed at startup is:

```sql
CREATE TABLE events (
    event_id            BIGINT PRIMARY KEY,
    kind                VARCHAR NOT NULL,
    canonical_service   VARCHAR NOT NULL,
    raw_service         VARCHAR NOT NULL,
    payload             JSON,
    occurred_at         TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_events_service_time
ON events (canonical_service, occurred_at);
```

| Column | Purpose |
|--------|---------|
| `event_id` | Monotonic primary key assigned in `_enqueue` before batch flush |
| `kind` | Router discriminator: `deploy`, `metric`, `log`, `trace`, `incident_signal`, `identity.drift`, `remediation`, etc. |
| `canonical_service` | Service name after `IdentityTracker` resolution at ingest |
| `raw_service` | Emitter-supplied name (may be a pre-drift alias) |
| `payload` | JSON document: metrics, log fields, `incident_id`, drift `old`/`new`, remediation `action`/`outcome` |
| `occurred_at` | Event time normalized to UTC in `Event.occurred_at_utc()` |

**Why DuckDB over SQLite.** `reconstruct_context` performs an analytical filter: all rows for one `canonical_service`, then a time window in Python. DuckDB’s columnar engine handles this scan in-process without a server. SQLite is row-oriented; wide JSON payload columns make large time-range scans comparatively expensive. We also rely on native `JSON` and `TIMESTAMPTZ` types without ad-hoc serialization.

**Why DuckDB over pandas.** Ingest would materialize every event as Python objects in a DataFrame. `MemorySubstrate` buffers tuples and flushes with `executemany` into DuckDB—compact storage, filter pushdown on read, no per-event pandas row overhead.

**Batching and ingest performance.** `Engine(batch_size=256)` buffers rows in `_pending` until `len(_pending) >= batch_size` or `ingest_many` calls `flush()`. Each flush issues one `INSERT ... VALUES (?, …)` batch. `scripts/perf_check.py` generates 5,000 events (alternating `deploy`, `metric`, `log`, `trace`, `metric` across 20 services) and selects the best of batch sizes 50, 256, and 512. Latest run: **4,052 events/sec** ingest throughput (target: >1,000).

## In-memory CausalGraph

Causal structure is not queried from SQL. `CausalGraph` maintains:

- `_deploys: dict[str, deque[tuple[str, str, datetime]]]` — per canonical service, the last **20** deploys `(event_id, canonical, ts)` with `deque(maxlen=20)`.
- `_edges: list[CausalEdge]` — append-only directed edges from deploy event id to effect event id.
- `_edge_services: dict[tuple[str, str], str]` — maps `(cause_id, effect_id)` to canonical service for incident snapshots.
- `_snapshots: dict[str, list[CausalEdge]]` — per `incident_id`, edges captured by `snapshot_incident`.

Each `CausalEdge` stores `cause_id`, `effect_id`, `evidence: list[str]`, `confidence: float`, and `occurred_at` (effect timestamp).

**Adjacency model.** The graph is not a general adjacency matrix. It is a **bipartite temporal linkage**: deploy nodes live in per-service deques; effect ingestion scans only that service’s deque and appends to a global edge list. Lookup for an incident is `O(|E|)` filter on `_edges` at snapshot time, then `O(1)` dict read via `_snapshots[incident_id]` at query time.

**Why dict / list / deque instead of NetworkX or a graph DB.** At benchmark scale, |E| per service is bounded by (deploys in 120s window) × (effects). NetworkX adds object overhead and serialization cost for a problem that is “scan ≤20 deploys, append one edge.” A remote graph database would add RPC latency on every `metric`/`log` ingest. Native Python structures match the hot path: one dict key per service, one deque append per deploy, one list append per edge.

**Temporal edge indexing.** Deploy history is time-ordered in each deque. `record_effect` only links deploys where `deploy_ts < effect_ts` and `(effect_ts - deploy_ts) ≤ lookback_s` (default **120**, set by `Engine(causal_window_s=120)`). `snapshot_incident(incident_id, ts, service, window_s=300)` copies edges whose `occurred_at` falls in `[ts - window_s, ts]` for the incident’s canonical service. `edges_for_incident` returns edges sorted ascending by `occurred_at`.

## IdentityTracker

Service renames and merges are modeled as disjoint-set union on string names.

**Data structures.**

- `_parent: dict[str, str]` — union–find parent pointer per name.
- `_size: dict[str, int]` — component size for union-by-size.
- `_aliases: dict[str, set[str]]` — all names in a component, maintained at the root.

**Path compression** (`_find`): during `resolve`, traverse to root and rewrite `parent[name]` to the root, flattening future lookups.

**Union-by-size** (`union`): attach the smaller component under the larger. On equal size, attach `old`’s root under `new`’s root so drift semantics favor the new name.

**Amortized complexity.** With path compression and union by size, `find` and `union` are **O(α(n))** per operation, where α is the inverse Ackermann function—effectively constant for practical name counts.

**Pseudocode (matches `contexter/identity_tracker.py`):**

```
function FIND(name):
    if parent[name] ≠ name:
        parent[name] ← FIND(parent[name])
    return parent[name]

function UNION(old, new):
    ENSURE(old); ENSURE(new)
    r_old ← FIND(old); r_new ← FIND(new)
    if r_old = r_new: return r_old
    if size[r_old] < size[r_new]:
        ATTACH(r_old, r_new); return r_new
    if size[r_old] > size[r_new]:
        ATTACH(r_new, r_old); return r_old
    ATTACH(r_old, r_new); return r_new   # equal size: new wins

function ATTACH(child_root, parent_root):
    parent[child_root] ← parent_root
    size[parent_root] ← size[parent_root] + size[child_root]
    aliases[parent_root] ← aliases[parent_root] ∪ aliases[child_root]
```

# Relationship Synthesis

The `Engine` registers substrate routers that derive relationships at ingest time from raw events.

## Deploy → metric / log linkage

On `kind == "deploy"`, the router calls `causal_graph.record_deploy(event_id, canonical, ts)` using the pending `substrate._next_id` as `event_id`.

On `kind == "metric"` or `"log"`, the router calls `record_effect(event_id, canonical, ts, kind, trace_id=…)`. The graph:

1. Resolves `service` to canonical via `IdentityTracker`.
2. Iterates deploys in `_deploys[canonical]`.
3. Skips deploys with `deploy_ts >= effect_ts` (no reverse causation).
4. Skips deploys where `(effect_ts - deploy_ts).total_seconds() > lookback_s` (**120 seconds** by default).
5. Computes confidence and appends a `CausalEdge`.

**Linear confidence decay** (implemented in `causal_graph.py`):

\[
\text{confidence} = \mathrm{clamp}_{[0,\,1]}\left(1 - \frac{\Delta t}{120}\right)
\]

where \(\Delta t\) is seconds from deploy to effect. Example: 0s → 1.0; 60s → 0.5; 120s → 0.0. `reconstruct_context` drops edges with `confidence < 0.3`.

**Why temporal proximity matters.** A deploy ten minutes before a latency spike is weak evidence; a deploy 30 seconds before is strong evidence. The decay encodes monotonic staleness without a separate ML model.

## Trace correlation

When a metric or log payload includes `trace_id`, `record_effect` adds `trace_id:<value>` to `CausalEdge.evidence`. Causal **edges are created only within the same canonical service** (deploy deque keyed by canonical). The `trace_id` does not, by itself, create cross-service graph edges in the current implementation; it attaches distributed-trace context to a deploy→effect link so operators can correlate an effect with an existing trace. `trace` events are stored in DuckDB and surfaced in `related_events` during reconstruction. Cross-service **dependency** context enters fingerprints via `upstream` / `upstream_roles` (resolved to role sets), not via hostname strings in the fingerprint key.

On `kind == "incident_signal"`, the engine calls `fingerprint_matcher.index_incident(incident_id, signal_payload)` and `causal_graph.snapshot_incident(incident_id, ts, canonical)`.

## structural_similarity

`FingerprintMatcher.top_k` scores the query fingerprint against each indexed incident using `structural_similarity` in `incident_fingerprint.py`. Weights default to **(0.35, 0.35, 0.30)** and are normalized to sum to 1:

\[
S = 0.35 \cdot \mathbb{1}[\text{trigger match}] + 0.35 \cdot \mathbb{1}[\text{role match}] + 0.30 \cdot J(\text{upstream}_L, \text{upstream}_R)
\]

**Jaccard overlap** on upstream role sets:

\[
J(A, B) = \frac{|A \cap B|}{|A \cup B|}
\]

If both sets are empty, \(J = 1.0\). Example: \(A = \{\text{auth}, \text{db}\}\), \(B = \{\text{auth}, \text{cache}\}\) → \(|A \cap B| = 1\), \(|A \cup B| = 3\) → \(J = 1/3\). `top_k` uses `heapq.nlargest` to return the best matches.

## Why service names are excluded from fingerprints

`IncidentFingerprint` contains only `trigger_type`, `affected_role`, and `upstream_involved` (a `frozenset` of roles). `FingerprintExtractor` resolves every service string through `IdentityTracker` before setting `affected_role`; callers may pass explicit `affected_role` / `upstream_roles` to skip raw names entirely.

Raw hostnames (`payments-svc-replica-7`, `billing-svc`) are **lexical** identifiers. Behavioral structure—what failed, what role failed, which upstream roles were involved—is stable across renames. Putting instance names into the fingerprint would fragment memory: the same failure mode would hash differently after every rename.

**Versus naive vector similarity.** Embedding `payments-svc` and `billing-svc` yields high cosine distance despite identical behavior. After `union(payments-svc, billing-svc)`, both names resolve to canonical `billing-svc`; `affected_role` matches and similarity stays high without re-embedding or backfilling vectors. Structural matching is **invariant to topology drift**; lexical embeddings are not.

# Topology Drift Handling

Topology drift—renames, retirements, dependency changes—is a first-class ingest event, not an after-the-fact data repair job.

## Event types

- **Rename:** `identity.drift` with `payload: {"old": "…", "new": "…"}`.
- **Service retirement:** modeled as merge into a successor via `union(old, new)` (successor becomes canonical).
- **Dependency evolution:** upstream sets in incident payloads change over time; Jaccard captures partial overlap between incidents without requiring identical dependency lists.

## Processing `identity.drift`

When `identity.drift(old="payments-svc", new="billing-svc")` is ingested:

1. `MemorySubstrate._resolve_drift` calls `identity.union("payments-svc", "billing-svc")` — **O(α(n))**.
2. Ingest returns canonical **`billing-svc`** (on equal component size, `new` wins).
3. All subsequent `resolve("payments-svc")` and `resolve("billing-svc")` return **`billing-svc`** via path-compressed `find`.

**DuckDB rows are not rewritten.** Historical events retain their ingest-time `canonical_service` and `raw_service`. Continuity is preserved because:

- New ingests resolve names through the tracker before write.
- Queries call `events_for_service("billing-svc")`, which resolves the query and filters on stored canonical.
- `ContextReconstructor._suggested_remediations` iterates `identity.aliases(canonical)` when looking up remediation hashes, so stats recorded under `error_rate:payments-svc:True` remain visible after rename.

**Complexity after repeated merges.** A chain of \(k\) merges on \(n\) names costs \(O(k \cdot \alpha(n))\) total; each `resolve()` remains \(O(\alpha(n))\).

## Concrete rename example

`scripts/rename_smoke.py` exercises:

1. Incidents and deploys under `payments-svc`.
2. Remediation for `INC-100` recorded under the pre-drift fingerprint hash.
3. `identity.drift` to `billing-svc`.
4. New incident `INC-714` on `billing-svc`.
5. `reconstruct_context` finds `INC-100` in `similar_past_incidents` with similarity ≥ 0.6 and recommends `rollback` with `target == "billing-svc"`.

Embedding retrieval would treat `payments-svc` and `billing-svc` as unrelated strings. Union–find declares them the same identity; structural fingerprints align.

# Latency Engineering

## Ingest time (write path)

`Engine.ingest` / `ingest_one` → `MemorySubstrate.ingest` resolves canonical identity, dispatches routers, enqueues rows, and batch-flushes to DuckDB.

| Router trigger | Precomputed work |
|----------------|------------------|
| `deploy` | Push to per-service deploy deque |
| `metric`, `log` | Scan deque, append `CausalEdge`(s), optional `trace_id` in evidence |
| `incident_signal` | Index `IncidentFingerprint`; snapshot causal edges for incident id |
| `remediation` | `RemediationMemory.record(fp_hash, action, outcome)` |
| `identity.drift` | `union(old, new)` inside substrate resolver |

Causal edges, fingerprint corpus entries, and remediation counters are **materialized at ingest**. Query latency does not pay graph-inference cost.

**Complexity (ingest).** Per event: \(O(\alpha(n))\) identity work; per metric/log: \(O(d)\) deploy scan with \(d \le 20\); DuckDB flush amortized over `batch_size` rows.

## Query time (read path)

`reconstruct_context` (via `ContextReconstructor.reconstruct`) performs **reads only**:

1. **Canonicalize** — `identity.resolve(signal["service"])`, \(O(\alpha(n))\).
2. **Related events** — `events_for_service(canonical)`: one indexed DuckDB read on `(canonical_service, occurred_at)`, then Python filter (kinds, 300s fast window, error logs only), \(O(r)\) for \(r\) returned rows.
3. **Causal chain** — `_snapshots[incident_id]` dict lookup + confidence filter, \(O(e)\) edges in snapshot.
4. **Similar incidents** — `top_k` over corpus with `heapq.nlargest`, \(O(m \log k)\) for \(m\) indexed incidents, \(k=5\).
5. **Remediations** — dict lookups in `_stats` for a small set of fingerprint hashes (including alias roles), \(O(1)\) per key.
6. **Explain** — fast template string; deep mode optional HTTP to Claude (not on hot path).

**Why precomputation enables SLOs.** Reconstruct avoids join-heavy inference, graph library overhead, and network I/O in fast mode. Measured on this repository with `scripts/perf_check.py` (5,000-event corpus, 50 fast reconstructs):

| Metric | Measured | Target |
|--------|----------|--------|
| Ingest throughput | **4,052 events/sec** | > 1,000 events/sec |
| Reconstruct p50 | **2.1 ms** | — |
| Reconstruct p95 | **2.4 ms** | < 2,000 ms |

# Memory Evolution

Remediation outcomes accumulate in `RemediationMemory`, separate from the event log and causal overlay.

## RemedStats model

```python
@dataclass(slots=True)
class RemedStats:
    attempts: int = 0
    successes: int = 0

    @property
    def confidence(self) -> float:
        return 0.0 if self.attempts == 0 else self.successes / attempts
```

Storage key: **`(fingerprint_hash, action) → RemedStats`** in `_stats: dict[tuple[str, str], RemedStats]`.

`record(fingerprint_hash, action, outcome)` increments `attempts` always; increments `successes` only when `outcome == "resolved"`.

## fingerprint_hash

Computed in `Engine._on_remediation` and `ContextReconstructor._suggested_remediations`:

```text
fingerprint_hash = f"{trigger_type}:{affected_role}:{bool(upstream_involved)}"
```

Example: `error_rate:api:True`. The boolean captures whether upstream roles were non-empty, not their exact membership (membership is compared in `structural_similarity` via Jaccard). `affected_role` is canonical after identity resolution, so renames that merge into one component share remediation memory when alias-aware lookup runs.

**Topology independence.** The hash does not include `payments-svc-pod-7` or other ephemeral instance ids. Five incidents on the same structural class (`error_rate`, same canonical role, same upstream-non-empty flag) share one stats bucket per action.

## Concrete learning example

Five deploy-caused API incidents, each remediated with `action=rollback`, `outcome=resolved`:

| Event | attempts | successes | confidence |
|-------|----------|-----------|------------|
| After 1st | 1 | 1 | 1.0 |
| After 2nd | 2 | 2 | 1.0 |
| … | … | … | … |
| After 5th | 5 | 5 | **1.0** |

`top_actions(fp_hash, k=3)` returns `("rollback", 1.0)` first. The next `reconstruct_context` for a matching incident surfaces `suggested_remediations[0]` with `historical_outcome="resolved"` and `target` set to the **current** canonical service (e.g. `billing-svc` after drift), not a stale alias.

One failed attempt among successes lowers the ratio: one resolved + one failed → confidence **0.5**. Operational memory improves monotonically with observed outcomes—no embedding retraining, no graph rewrite—only sufficient statistics keyed by structural incident class.
