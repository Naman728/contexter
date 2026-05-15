# Advanced Benchmark Suite for Contexter

This document gives you realistic stress tests and adversarial benchmarks beyond the official harness.

The goal is to test:

* Rename robustness
* Memory persistence
* Similarity retrieval quality
* Remediation learning
* Latency under scale
* Drift handling
* Failure recovery
* Multi-hop incident reconstruction
* Noise resistance
* Temporal reasoning

---

# 1. Massive Multi-Seed Stress Test

This checks:

* stability
* memory leaks
* scaling behavior
* retrieval degradation under load

Run:

```bash
python run.py \
  --adapter adapters.mine:Engine \
  --seeds 1111 2222 3333 4444 5555 6666 7777 8888 9999 \
  --n-services 50 \
  --days 30
```

What to observe:

* Does recall collapse as service count increases?
* Does latency remain stable?
* Does memory usage explode?
* Does remediation_acc stay high?

Good signs:

* latency < 10ms
* recall stable across seeds
* no crashes

---

# 2. Extreme Rename Storm Test

Purpose:
Test whether union-find identity tracking survives aggressive topology drift.

Create:

```text
svc-a -> svc-b
svc-b -> svc-c
svc-c -> svc-d
svc-d -> svc-e
```

Then replay the SAME incident pattern.

Expected:

The engine should still retrieve incidents from svc-a.

This tests:

* path compression
* canonical resolution
* alias continuity
* topology-independent fingerprints

Example test idea:

```python
identity.union("svc-a", "svc-b")
identity.union("svc-b", "svc-c")
identity.union("svc-c", "svc-d")
identity.union("svc-d", "svc-e")
```

Then reconstruct under svc-e.

Expected:

similar_past_incidents contains incidents from svc-a.

---

# 3. Noise Flood Benchmark

Purpose:
See whether retrieval quality survives massive unrelated event noise.

Generate:

* 100k random logs
* random metrics
* unrelated traces
* fake deploys

Then insert ONE true incident family.

Measure:

* recall@5
* latency
* precision

Goal:

Your matcher should still retrieve the correct family despite huge background entropy.

---

# 4. Temporal Drift Benchmark

Purpose:
Test whether causal reconstruction survives large time gaps.

Scenario:

* deploy happens
* 20 minutes later latency spike
* 30 minutes later upstream failure
* remediation 1 hour later

Questions:

* does confidence decay correctly?
* do edges disappear after window expiration?
* does retrieval still work?

This tests:

* temporal windows
* edge decay
* snapshot logic
* stale edge cleanup

---

# 5. False Correlation Benchmark

Purpose:
Test whether your engine invents fake causality.

Scenario:

* deploy occurs
* unrelated metric spike elsewhere
* unrelated service fails

Expected:

No strong causal edge should form.

This is important because weak systems overfit temporal proximity.

---

# 6. Multi-Hop Cascade Benchmark

Purpose:
Test graph reasoning depth.

Scenario:

```text
frontend-api
    ↓
checkout-api
    ↓
payments-svc
    ↓
db-cluster
```

Inject:

* deploy on db-cluster
* latency spike on payments
* errors on checkout
* failures on frontend

Expected:

Causal chain should reconstruct:

```text
db deploy
  → payments latency
  → checkout errors
  → frontend failures
```

This tests:

* chain ordering
* trace linkage
* upstream reasoning
* graph traversal

---

# 7. Remediation Evolution Benchmark

Purpose:
Test whether memory actually learns.

Scenario:
Run same family repeatedly:

```text
rollback → resolved
rollback → resolved
rollback → resolved
restart → failed
restart → failed
```

Expected:

rollback confidence increases toward 1.0.

restart confidence decreases.

This validates:

* reinforcement from outcomes
* memory evolution
* adaptive confidence

---

# 8. Cold Start Benchmark

Purpose:
Measure behavior with NO historical memory.

Expected:

* graceful degradation
* low confidence
* empty similar_past_incidents
* no crashes

This tests:

* robustness
* default handling
* empty-state logic

---

# 9. Long-Term Memory Benchmark

Purpose:
Test whether incidents remain retrievable after massive ingestion.

Scenario:

* ingest millions of events
* old incident appears again after many days

Expected:

Engine still retrieves historical family.

This tests:

* indexing stability
* memory persistence
* retrieval scalability

---

# 10. Latency Saturation Benchmark

Purpose:
Push reconstruct_context extremely hard.

Run:

```python
for _ in range(10000):
    engine.reconstruct_context(signal)
```

Measure:

* p50
* p95
* p99
* max latency

Expected:

Latency remains stable.

Watch for:

* memory leaks
* caching issues
* growing graph traversal cost

---

# 11. Benchmark for Overfitting

Purpose:
Check whether your matcher is accidentally tuned only for the official dataset.

Create completely custom families:

Family A:

```text
cache saturation
```

Family B:

```text
DNS resolution failures
```

Family C:

```text
message queue backlog
```

Expected:

Structural matching still works.

---

# 12. Retrieval Quality Audit

Add temporary debug output:

```python
print({
    "query_incident": signal["incident_id"],
    "returned": matches,
    "expected_family": expected_family,
})
```

Then inspect:

* why matches failed
* whether role weighting is too strong
* whether upstream overlap is too weak
* whether trigger matching is too strict

This is the SINGLE best way to improve recall.

---

# 13. Compare Against Naive Baseline

Create a dumb matcher:

```text
same service name only
```

Then compare:

* rename robustness
* recall
* latency

This helps demonstrate:

why your architecture matters.

---

# 14. Benchmark Targets

Strong submission targets:

| Metric            | Good       |
| ----------------- | ---------- |
| recall@5          | 0.55–0.70  |
| precision@5       | 0.40–0.60  |
| remediation_acc   | 0.90–1.00  |
| latency_p95       | < 10ms     |
| stability         | no crashes |

---

# 15. Most Important Insight

This benchmark is NOT about:

* chatbot intelligence
* LLM prompting
* vector databases
* semantic search

It IS about:

* evolving operational memory
* topology-independent retrieval
* causal reconstruction
* persistent infrastructure intelligence
* incremental graph reasoning

That distinction is the core of the entire problem statement.
