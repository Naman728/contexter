"""Topology-independent incident fingerprints and structural matching."""

from __future__ import annotations

import hashlib
import heapq
import math
import os
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from typing import Any, Literal, TextIO, overload

from contexter.events import Event
from contexter.identity_tracker import IdentityTracker
from contexter.roles import family_aware_role

RoleResolver = Callable[[str], str]

_DEFAULT_WEIGHTS: tuple[float, float, float] = (0.35, 0.35, 0.30)
_ALIAS_BOOST = 0.20
_ROLE_FAMILY_BOOST = 0.10
_TEMPORAL_BOOST = 0.15

_LOCALE_STAGE1_BONUS = 4.0
_DRIFT_CONTINUITY_RERANK_CAP = 0.07
_GRAPH_NEIGHBORHOOD_RERANK_SCALE = 0.06
_HIST_COINV_JACCARD_SCALE = 0.04
_HIST_COINV_PAIR_BONUS = 0.02
_PROPAGATION_FP_RERANK_WEIGHT = 0.142
_ALIAS_RERANK_WEIGHT = 0.018
_TEMPORAL_SHAPE_RERANK_WEIGHT = 0.138
_TEMPORAL_PROFILE_MAX_S = 172_800.0
_NEGATIVE_EVIDENCE_FLOOR = 0.68

_BEHAVIOR_DEPLOY_EXACT = 0.14
_BEHAVIOR_PROP_ORDER_EXACT = 0.12
_BEHAVIOR_TIMING_SHAPE = 0.06
_BEHAVIOR_REMED_ACTIONS_EXACT = 0.05
_BEHAVIOR_CAP = 0.40

_RECURRENCE_PRIOR = 0.10

# IDF rerank: rare structural patterns get higher weight; damped on large corpora.
_IDF_CORPUS_SMALL = 48
_IDF_FLOOR_SMALL_CORPUS = 0.875
_IDF_FLOOR_LARGE_CORPUS = 0.968

_GEN_SUPPRESS_STRENGTH = 0.11
_GEN_MULT_MIN = 0.86
_HIGH_CONF_PROP_T = 0.84
_HIGH_CONF_TEMPO_T = 0.76
_HIGH_CONF_UP_T = 0.72
_HIGH_CONF_ALIAS_STRONG = 0.72
_HIGH_CONF_DRIFT_LINEAGE_MIN = 0.022
_HIGH_STRUCT_BOOST_CAP = 0.075

_MARGIN_DOM_ALPHA = 0.12
_MARGIN_CLUSTER_BETA = 0.095
_RARE_MARGIN_K = 0.028
_GEN_MARGIN_K = 0.032

_MIN_CANDIDATE_POOL = 75
_MAX_CANDIDATE_POOL = 300
_MAX_RAW_CANDIDATE_UNION = 1000
_MAX_PER_CANONICAL_IN_POOL = 5
_MAX_PER_ROLE_CLUSTER_IN_POOL = 8


def normalize_trigger(trigger: str) -> str:
    """Map noisy trigger labels to coarse families for similarity and indexing."""
    t = (trigger or "unknown").strip().lower()
    if not t:
        return "unknown"
    if any(
        x in t
        for x in (
            "latency",
            "p99",
            "p95",
            "p50",
            "p90",
            "slow",
            "duration",
            "slowness",
        )
    ):
        return "latency"
    if any(
        x in t
        for x in (
            "error",
            "5xx",
            "4xx",
            "timeout",
            "fault",
            "exception",
            "oom",
            "panic",
            "crash",
        )
    ):
        return "error"
    if any(x in t for x in ("health", "availability", "ping", "live", "ready", "liveness", "readiness")):
        return "availability"
    if any(x in t for x in ("cpu", "processor", "load", "compute")):
        return "cpu"
    if any(x in t for x in ("memory", "mem", "ram", "resident", "heap")):
        return "memory"
    if any(x in t for x in ("throughput", "qps", "rps", "traffic", "volume", "requests")):
        return "throughput"
    if any(x in t for x in ("disk", "storage", "io", "filesystem", "space")):
        return "disk"
    return t


def infer_role_family(name: str) -> str:
    """Broad operational role bucket from a service or role string."""
    if not name:
        return "unknown"
    n = name.lower()
    if any(x in n for x in ("frontend", "web-ui", "webapp", "fe-", "-fe", "static")):
        return "frontend"
    if any(x in n for x in ("gateway", "edge", "ingress", "lb-", "-lb")):
        return "gateway"
    if any(x in n for x in ("worker", "consumer", "batch", "cron", "job-")):
        return "worker"
    if any(x in n for x in ("queue", "kafka", "sqs", "rabbit", "pubsub", "stream")):
        return "queue"
    if any(x in n for x in ("redis", "memcached", "elasticache", "cache")):
        return "cache"
    if any(
        x in n
        for x in ("db", "database", "postgres", "mysql", "mongo", "sql", "rds", "oltp")
    ):
        return "db"
    if "api" in n or n.endswith("-svc") or "-svc-" in n:
        return "api"
    if n.startswith("family-"):
        return n
    return "other"


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def partial_upstream_overlap(a: frozenset[str], b: frozenset[str]) -> float:
    """Subset-friendly overlap: ``|∩| / max(1, min(|a|,|b|))``."""
    inter = len(a & b)
    return inter / max(1, min(len(a), len(b)))


_MAX_DEPLOY_PATTERN_LEN = 32
_SEQUENCE_RERANK_WEIGHT = 0.235
_RERANK_ROLE_CLUSTER_SOFT = 0.42
_REMED_RERANK_WEIGHT = 0.12


def sequence_similarity(a: Sequence[str], b: Sequence[str]) -> float:
    """Ordered overlap of two token sequences via normalized LCS length.

    ``deploy→latency→error`` aligns strongly with ``deploy→latency→error→rollback``.
    Both empty → ``1.0``; exactly one empty → ``0.0``.
    """
    ta = tuple(a)
    tb = tuple(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    lcs = _lcs_len(ta, tb)
    return (2.0 * lcs) / (len(ta) + len(tb))


def _lcs_len(a: tuple[str, ...], b: tuple[str, ...]) -> int:
    n, m = len(a), len(b)
    dp = [0] * (m + 1)
    for i in range(1, n + 1):
        prev = 0
        for j in range(1, m + 1):
            temp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = temp
    return dp[m]


def _metric_is_latency_signal(payload: dict[str, Any]) -> bool:
    if payload.get("degraded") is True or payload.get("anomaly") is True:
        return True
    name = str(payload.get("name", "")).lower()
    return any(
        x in name
        for x in ("latency", "p99", "p95", "p90", "duration", "timeout", "slow")
    )


def _event_to_deploy_pattern_token(event: Mapping[str, Any]) -> str | None:
    kind = str(event.get("kind", ""))
    payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
    if not isinstance(payload, dict):
        payload = {}
    if kind == "deploy":
        return "deploy"
    if kind == "metric" and _metric_is_latency_signal(payload):
        return "latency"
    if kind == "log" and str(payload.get("level", "")).lower() == "error":
        return "error"
    if kind == "remediation":
        action = str(payload.get("action", "")).lower()
        if action == "rollback":
            return "rollback"
        return "remediation"
    return None


def extract_deploy_pattern_sequence(
    substrate: Any,
    canonical_service: str,
    until_ts: datetime,
    *,
    window_seconds: int = 300,
) -> tuple[str, ...]:
    """Ordered deploy-centric pattern tokens before ``until_ts`` on ``canonical_service``."""
    if not substrate or not canonical_service:
        return ()
    try:
        window_start = until_ts - timedelta(seconds=window_seconds)
        rows = substrate.events_for_service(
            canonical_service, since=window_start, until=until_ts
        )
    except Exception:
        return ()
    rows_sorted = sorted(
        rows,
        key=lambda row: row.get("occurred_at") or until_ts,
    )
    out: list[str] = []
    for row in rows_sorted:
        tok = _event_to_deploy_pattern_token(row)
        if tok is None:
            continue
        if not out or out[-1] != tok:
            out.append(tok)
        if len(out) >= _MAX_DEPLOY_PATTERN_LEN:
            break
    return tuple(out)


def _upstream_score(a: frozenset[str], b: frozenset[str]) -> float:
    j = jaccard(a, b)
    p = partial_upstream_overlap(a, b)
    # Prefer exact matches (high Jaccard) while still rewarding subsets.
    return 0.7 * p + 0.3 * j


def _temporal_boost(
    query_ctx: Mapping[str, Any] | None,
    candidate_ctx: Mapping[str, Any] | None,
) -> float:
    if not query_ctx or not candidate_ctx:
        return 0.0
    qw = query_ctx.get("deploy_window")
    cw = candidate_ctx.get("deploy_window")
    if qw is not None and qw == cw:
        return _TEMPORAL_BOOST
    if bool(query_ctx.get("post_deploy_metric")) and bool(
        candidate_ctx.get("post_deploy_metric")
    ):
        return _TEMPORAL_BOOST
    return 0.0


def _alias_boost(left: IncidentFingerprint, right: IncidentFingerprint) -> float:
    ca = left.canonical_affected
    cb = right.canonical_affected
    if ca and cb and ca == cb:
        return _ALIAS_BOOST
    return 0.0


def _role_family_boost(left: IncidentFingerprint, right: IncidentFingerprint) -> float:
    lf = infer_role_family(left.canonical_affected or left.affected_role)
    rf = infer_role_family(right.canonical_affected or right.affected_role)
    if lf == rf and lf != "unknown" and left.affected_role != right.affected_role:
        return _ROLE_FAMILY_BOOST
    return 0.0


@dataclass(frozen=True, slots=True)
class IncidentFingerprint:
    """Structural incident signature independent of concrete topology names."""

    trigger_type: str
    affected_role: str
    upstream_involved: frozenset[str]
    canonical_affected: str = ""

    def as_tuple(self) -> tuple[str, str, frozenset[str], str]:
        return (
            self.trigger_type,
            self.affected_role,
            self.upstream_involved,
            self.canonical_affected,
        )


@dataclass(frozen=True, slots=True)
class MatchResult:
    incident_id: str
    fingerprint: IncidentFingerprint
    score: float
    score_breakdown: dict[str, float] | None = None
    retrieval_sources: tuple[str, ...] = ()


def _propagation_depth_bucket(depth: int) -> str:
    """Coarse cascade depth for hashing (topology-independent, stable under small drift)."""
    d = max(0, int(depth))
    if d == 0:
        return "P0"
    if d == 1:
        return "P1"
    if d <= 3:
        return "P2_3"
    if d <= 6:
        return "P4_6"
    return "P7p"


def _quantize_dim_bucket(seconds: float, thresholds: tuple[float, ...], prefix: str) -> str:
    """Single-letter prefix + bucket index; ``U`` = unknown (``seconds < 0``)."""
    if seconds < 0.0:
        return f"{prefix}U"
    for i, t in enumerate(thresholds):
        if seconds <= t:
            return f"{prefix}{i}"
    return f"{prefix}{len(thresholds)}"


def _temporal_shape_bucket(t: TemporalProfile) -> str:
    """Compact multi-dimension timing shape token (seconds-quantized, deterministic)."""
    d_thr = (60.0, 300.0, 1800.0, 7200.0, 86400.0, 172800.0)
    f_thr = (1.0, 30.0, 120.0, 600.0, 3600.0, 86400.0)
    r_thr = (60.0, 600.0, 3600.0, 86400.0, 172800.0)
    c_thr = (1.0, 30.0, 120.0, 600.0, 3600.0, 86400.0)
    return "".join(
        (
            _quantize_dim_bucket(t.deploy_to_failure_seconds, d_thr, "d"),
            _quantize_dim_bucket(t.failure_spread_seconds, f_thr, "f"),
            _quantize_dim_bucket(t.remediation_delay_seconds, r_thr, "r"),
            _quantize_dim_bucket(t.cascade_duration_seconds, c_thr, "c"),
        )
    )


def _edge_transition_hash(prop: PropagationFingerprint) -> str:
    """Short deterministic token from ordered role transitions (preferred) or edge types."""
    if prop.role_transitions:
        raw = ">".join(prop.role_transitions)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
    if prop.edge_types:
        return _edge_type_sequence_hash(prop.edge_types)[:12]
    return "NA"


def _remediation_fingerprint_suffix(
    fp: IncidentFingerprint,
    deploy_pattern: tuple[str, ...],
    prop: PropagationFingerprint,
    tempo: TemporalProfile,
) -> str:
    depth_b = _propagation_depth_bucket(prop.propagation_depth)
    deploy_presence = "1" if "deploy" in deploy_pattern else "0"
    t_shape = _temporal_shape_bucket(tempo)
    aff = infer_role_family(fp.canonical_affected or fp.affected_role)
    root_role = prop.root_degradation_role or aff
    term_role = prop.terminal_failure_role or aff
    edge_h = _edge_transition_hash(prop)
    return f"{depth_b}:{deploy_presence}:{t_shape}:{root_role}:{term_role}:{edge_h}"


def _fingerprint_remediation_hash_parts(
    fp: IncidentFingerprint,
    deploy_pattern: tuple[str, ...],
    prop: PropagationFingerprint,
    tempo: TemporalProfile,
) -> str:
    base = fingerprint_remediation_base_key(fp)
    suf = _remediation_fingerprint_suffix(fp, deploy_pattern, prop, tempo)
    return f"{base}:{suf}"


def _upstream_shape_token(upstream: frozenset[str]) -> str:
    """Stable, rename-robust token from upstream role involvement (coarse role families)."""
    if not upstream:
        return "ux0"
    parts = sorted({infer_role_family(str(x)) for x in upstream})
    raw = ",".join(parts)
    if len(raw) <= 40:
        return raw.replace(":", "_")
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


def fingerprint_remediation_base_key(fp: IncidentFingerprint) -> str:
    """Structural prefix shared by all extended remediation hashes for ``fp``."""
    nt = normalize_trigger(fp.trigger_type)
    ux = _upstream_shape_token(fp.upstream_involved)
    return f"{nt}:{fp.affected_role}:{int(bool(fp.upstream_involved))}:{ux}"


def fingerprint_remediation_hash(
    fp: IncidentFingerprint,
    retrieval_features: RetrievalFeatures | None = None,
) -> str:
    """Stable remediation / retrieval hash: structural base plus propagation, deploy, timing.

    Uses normalized trigger family, role, upstream flag, then buckets derived from
    retrieval-time signals (propagation depth, deploy token presence, temporal shape,
    coarse root/terminal roles, and a short transition hash). When ``retrieval_features``
    is omitted, empty propagation / missing temporal / no deploy-pattern tokens apply.
    """
    if retrieval_features is None:
        return _fingerprint_remediation_hash_parts(
            fp, (), _EMPTY_PROPAGATION_FP, TemporalProfile.missing()
        )
    return _fingerprint_remediation_hash_parts(
        fp,
        retrieval_features.deploy_pattern,
        retrieval_features.propagation_fingerprint,
        retrieval_features.temporal_profile,
    )


@dataclass(frozen=True, slots=True)
class PropagationFingerprint:
    """Directed degradation cascade from propagation snapshot edges.

    ``role_transitions`` entries look like ``'db>api'`` (``infer_role_family`` buckets).
    ``edge_type_seq_hash`` is a deterministic SHA-256 prefix of the ordered ``edge_types``.
    ``branching_factor`` is the max fan-out (unique effects) from any single cause service.
    """

    degradation_order: tuple[str, ...]
    edge_types: tuple[str, ...]
    propagation_hops: tuple[int, ...]
    hop_count: int
    propagation_depth: int
    role_transitions: tuple[str, ...] = ()
    edge_type_seq_hash: str = ""
    branching_factor: int = 0
    terminal_failure_role: str = ""
    root_degradation_role: str = ""


_EMPTY_PROPAGATION_FP = PropagationFingerprint((), (), (), 0, 0)


def _parse_propagation_hop(evidence: Sequence[Any]) -> int:
    for raw in evidence:
        s = str(raw)
        if s.startswith("propagation:"):
            try:
                return max(1, int(s.split(":", 1)[1]))
            except (ValueError, IndexError):
                return 1
    return 1


def _parse_propagation_pair_kind(evidence: Sequence[Any]) -> str:
    for raw in evidence:
        s = str(raw)
        if s.startswith("pair:"):
            k = s.split(":", 1)[1].strip().lower()
            return k or "unknown"
    return "unknown"


def _edge_type_sequence_hash(edge_types: tuple[str, ...]) -> str:
    """Deterministic short fingerprint of the ordered edge-type sequence."""
    if not edge_types:
        return ""
    return hashlib.sha256("|".join(edge_types).encode("utf-8")).hexdigest()[:14]


def _role_path_from_services(slim: tuple[str, ...]) -> tuple[str, ...]:
    """Deduped coarse role bucket sequence along the degradation path."""
    out: list[str] = []
    for name in slim:
        r = infer_role_family(name)
        if not out or out[-1] != r:
            out.append(r)
    return tuple(out)


def _normalized_token_edit_similarity(a: tuple[str, ...], b: tuple[str, ...]) -> float:
    """``1 - Levenshtein(a,b) / max(len)`` in ``[0, 1]`` (bounded ``O(|a|·|b|)``)."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    n, m = len(a), len(b)
    if n * m > 4096:
        na, nb = a[:48], b[:48]
        n, m = len(na), len(nb)
        a, b = na, nb
    dp = [list(range(m + 1))]
    for i in range(1, n + 1):
        row = [i] + [0] * m
        ai = a[i - 1]
        prev = dp[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ai == b[j - 1] else 1
            row[j] = min(prev[j] + 1, row[j - 1] + 1, prev[j - 1] + cost)
        dp.append(row)
    dist = dp[n][m]
    return 1.0 - dist / max(n, m, 1)


def extract_propagation_fingerprint(edges: Sequence[Any]) -> PropagationFingerprint:
    """Build a propagation fingerprint from incident snapshot edges (O(|edges|))."""
    props: list[Any] = []
    for edge in edges:
        ev = getattr(edge, "evidence", ()) or ()
        if any(str(x).startswith("propagation:") for x in ev):
            props.append(edge)
    if not props:
        return _EMPTY_PROPAGATION_FP
    props.sort(key=lambda e: getattr(e, "occurred_at", 0))

    ordered_names: list[str] = []
    edge_types: list[str] = []
    hops: list[int] = []
    for e in props:
        ev = tuple(getattr(e, "evidence", ()) or ())
        hops.append(_parse_propagation_hop(ev))
        edge_types.append(_parse_propagation_pair_kind(ev))
        if not ordered_names:
            cs0 = str(getattr(e, "cause_service", "") or "").strip()
            if cs0:
                ordered_names.append(cs0)
        es = str(getattr(e, "effect_service", "") or "").strip()
        if es:
            ordered_names.append(es)

    slim: list[str] = []
    for name in ordered_names:
        if name and (not slim or slim[-1] != name):
            slim.append(name)
    slim_t = tuple(slim)
    hop_count = len(props)
    depth = max(hops) if hops else max(0, len(slim) - 1)

    role_transitions: list[str] = []
    fanout: defaultdict[str, set[str]] = defaultdict(set)
    for e in props:
        cs = str(getattr(e, "cause_service", "") or "").strip()
        es = str(getattr(e, "effect_service", "") or "").strip()
        rc = infer_role_family(cs) if cs else "unknown"
        re_ = infer_role_family(es) if es else "unknown"
        role_transitions.append(f"{rc}>{re_}")
        if cs and es:
            fanout[cs].add(es)
    branching = max((len(v) for v in fanout.values()), default=1)

    et_tuple = tuple(edge_types)
    ehash = _edge_type_sequence_hash(et_tuple)
    root_role = infer_role_family(slim[0]) if slim else ""
    term_role = infer_role_family(slim[-1]) if slim else ""

    return PropagationFingerprint(
        degradation_order=slim_t,
        edge_types=et_tuple,
        propagation_hops=tuple(hops),
        hop_count=hop_count,
        propagation_depth=depth,
        role_transitions=tuple(role_transitions),
        edge_type_seq_hash=ehash,
        branching_factor=int(branching),
        terminal_failure_role=term_role,
        root_degradation_role=root_role,
    )


def propagation_similarity(
    a: PropagationFingerprint,
    b: PropagationFingerprint,
) -> float:
    """Similarity in ``[0, 1]`` with ordered path edit alignment and cascade specificity."""
    if a.hop_count == 0 and b.hop_count == 0:
        return 1.0
    if a.hop_count == 0 or b.hop_count == 0:
        return 0.1

    da, db = a.propagation_depth, b.propagation_depth
    depth_sim = 1.0 - min(1.0, abs(da - db) / max(da, db, 1))
    hop_sim = 1.0 - min(
        1.0,
        abs(a.hop_count - b.hop_count) / max(a.hop_count, b.hop_count, 1),
    )
    path_edit = _normalized_token_edit_similarity(a.degradation_order, b.degradation_order)
    role_path_a = _role_path_from_services(a.degradation_order)
    role_path_b = _role_path_from_services(b.degradation_order)
    role_path_edit = _normalized_token_edit_similarity(role_path_a, role_path_b)
    trans_sim = sequence_similarity(a.role_transitions, b.role_transitions)
    type_sim = sequence_similarity(a.edge_types, b.edge_types)
    topo = jaccard(frozenset(a.degradation_order), frozenset(b.degradation_order))

    br_a = max(1, int(a.branching_factor))
    br_b = max(1, int(b.branching_factor))
    branch_sim = 1.0 - min(1.0, abs(br_a - br_b) / max(br_a, br_b))

    def _role_tag_match(x: str, y: str) -> float:
        if not x or not y:
            return 0.5
        return 1.0 if x == y else 0.0

    root_sim = _role_tag_match(a.root_degradation_role, b.root_degradation_role)
    term_sim = _role_tag_match(a.terminal_failure_role, b.terminal_failure_role)
    hash_sim = (
        1.0
        if a.edge_type_seq_hash
        and b.edge_type_seq_hash
        and a.edge_type_seq_hash == b.edge_type_seq_hash
        else (0.5 if not (a.edge_type_seq_hash or b.edge_type_seq_hash) else 0.0)
    )

    return (
        0.12 * depth_sim
        + 0.08 * hop_sim
        + 0.28 * path_edit
        + 0.16 * role_path_edit
        + 0.12 * trans_sim
        + 0.08 * type_sim
        + 0.06 * topo
        + 0.04 * branch_sim
        + 0.02 * root_sim
        + 0.02 * term_sim
        + 0.02 * hash_sim
    )


@dataclass(frozen=True, slots=True)
class TemporalProfile:
    """Seconds-based timing shape for an incident snapshot (precomputed at index).

    ``-1.0`` means unknown / not applicable for that dimension.
    """

    deploy_to_failure_seconds: float
    failure_spread_seconds: float
    remediation_delay_seconds: float
    cascade_duration_seconds: float

    @staticmethod
    def missing() -> TemporalProfile:
        return TemporalProfile(-1.0, -1.0, -1.0, -1.0)


@dataclass(frozen=True, slots=True)
class RetrievalFeatures:
    """Lightweight features precomputed at ingest for fast reranking."""

    norm_trigger: str
    role: str
    canonical_service: str
    upstream_roles: frozenset[str]
    deploy_proximity: float
    causal_edge_count: int
    propagation_edge_count: int
    remediation_fp_hash: str
    deploy_pattern: tuple[str, ...] = ()
    propagation_fingerprint: PropagationFingerprint = _EMPTY_PROPAGATION_FP
    temporal_profile: TemporalProfile = TemporalProfile.missing()

    @staticmethod
    def from_fingerprint(fingerprint: IncidentFingerprint) -> RetrievalFeatures:
        return RetrievalFeatures(
            normalize_trigger(fingerprint.trigger_type),
            fingerprint.affected_role,
            fingerprint.canonical_affected,
            fingerprint.upstream_involved,
            0.0,
            0,
            0,
            fingerprint_remediation_hash(fingerprint),
            (),
            _EMPTY_PROPAGATION_FP,
            TemporalProfile.missing(),
        )


@dataclass(frozen=True, slots=True)
class RerankContext:
    """Query-side retrieval features plus remediation memory for stage-2 scoring."""

    query_features: RetrievalFeatures
    remediation_memory: Any
    identity: IdentityTracker | None = None
    dependency_graph: Any = None
    neighborhood_memory: Any = None
    query_canonical: str = ""
    rerank_weights: Mapping[str, float] | None = None
    query_signal: Mapping[str, Any] | None = None
    idf_stats: IDFStats | None = None


def _effective_canonical(identity: IdentityTracker | None, stored: str) -> str:
    if not stored:
        return ""
    if identity is None:
        return stored
    identity.register(stored)
    return identity.resolve(stored)


def _retrieval_locale(
    *,
    identity: IdentityTracker | None,
    dependency_graph: Any,
    neighborhood_memory: Any,
    query_canonical: str,
) -> set[str]:
    """Canonical services to treat as one retrieval neighborhood (topology + history)."""
    if not query_canonical:
        return set()
    root = _effective_canonical(identity, query_canonical)
    if not root:
        return set()
    out: set[str] = {root}
    if dependency_graph is not None:
        try:
            out.update(dependency_graph.neighbors(root))
        except Exception:
            pass
    if neighborhood_memory is not None:
        try:
            out.update(neighborhood_memory.peers(root))
        except Exception:
            pass
    return out


def _drift_continuity_rerank(
    identity: IdentityTracker | None,
    qfp: IncidentFingerprint,
    cfp: IncidentFingerprint,
) -> float:
    """Boost when both incidents sit on the same identity chain (possibly different labels)."""
    if identity is None:
        return 0.0
    qn, cn = qfp.canonical_affected, cfp.canonical_affected
    if not qn or not cn:
        return 0.0
    identity.register(qn)
    identity.register(cn)
    rq = identity.resolve(qn)
    if rq != identity.resolve(cn):
        return 0.0
    prof = identity.alias_profile(rq)
    depth = prof.rename_depth
    if qn != cn or depth > 0:
        return min(
            _DRIFT_CONTINUITY_RERANK_CAP,
            0.05 + 0.018 * float(min(depth, 8)),
        )
    return 0.0


def _graph_neighborhood_overlap_rerank(
    dependency_graph: Any,
    identity: IdentityTracker | None,
    query_root: str,
    cand_root: str,
) -> float:
    """Jaccard overlap of dependency-graph neighborhoods (self ∪ 1-hop), canonicalized."""
    if dependency_graph is None or not query_root or not cand_root:
        return 0.0
    qr = _effective_canonical(identity, query_root)
    cr = _effective_canonical(identity, cand_root)
    if not qr or not cr:
        return 0.0
    try:
        nq = frozenset(dependency_graph.neighbors(qr)) | {qr}
        nc = frozenset(dependency_graph.neighbors(cr)) | {cr}
    except Exception:
        return 0.0
    return jaccard(nq, nc) * _GRAPH_NEIGHBORHOOD_RERANK_SCALE


def _historical_neighborhood_rerank(
    neighborhood_memory: Any,
    identity: IdentityTracker | None,
    query_root: str,
    cand_root: str,
) -> float:
    if neighborhood_memory is None or not query_root or not cand_root:
        return 0.0
    qr = _effective_canonical(identity, query_root)
    cr = _effective_canonical(identity, cand_root)
    if not qr or not cr or qr == cr:
        return 0.0
    try:
        pj = float(neighborhood_memory.peer_jaccard(qr, cr))
    except Exception:
        pj = 0.0
    bonus = _HIST_COINV_JACCARD_SCALE * pj
    try:
        if neighborhood_memory.pair_seen(qr, cr):
            bonus += _HIST_COINV_PAIR_BONUS
    except Exception:
        pass
    return min(0.10, bonus)


def _stage1_locale_bonus(
    identity: IdentityTracker | None,
    locale: frozenset[str],
    cfp: IncidentFingerprint,
    cfeat: RetrievalFeatures,
) -> float:
    if not locale or identity is None:
        return 0.0
    stored = cfeat.canonical_service or cfp.canonical_affected
    if not stored:
        return 0.0
    rc = _effective_canonical(identity, stored)
    return _LOCALE_STAGE1_BONUS if rc in locale else 0.0


def _propagation_edge_count(edges: Sequence[Any]) -> int:
    n = 0
    for edge in edges:
        ev = getattr(edge, "evidence", ()) or ()
        if any(str(x).startswith("propagation:") for x in ev):
            n += 1
    return n


def _causal_edge_has_service_kind(evidence: Sequence[Any], kind: str) -> bool:
    prefix = f"kind:{kind}"
    for raw in evidence:
        if str(raw).lower().startswith(prefix.lower()):
            return True
    return False


def _causal_edge_is_failure_signal(edge: Any) -> bool:
    ev = getattr(edge, "evidence", ()) or ()
    return _causal_edge_has_service_kind(ev, "log") or _causal_edge_has_service_kind(ev, "metric")


def extract_temporal_profile(
    edges: Sequence[Any],
    substrate: Any | None,
    canonical_service: str,
    until_ts: datetime,
    *,
    remediation_lookforward_s: int = 7200,
) -> TemporalProfile:
    """Derive timing shape from snapshot edges and substrate (single linear pass over edges)."""
    if not edges:
        return TemporalProfile.missing()

    sorted_edges = sorted(edges, key=lambda e: getattr(e, "occurred_at", until_ts))

    deploy_ts: datetime | None = None
    if substrate is not None and canonical_service:
        try:
            deploy_ts = substrate.last_deploy_before(canonical_service, until_ts)
        except Exception:
            deploy_ts = None

    failure_times: list[datetime] = []
    prop_times: list[datetime] = []
    for e in sorted_edges:
        ev = tuple(getattr(e, "evidence", ()) or ())
        if any(str(x).startswith("propagation:") for x in ev):
            prop_times.append(e.occurred_at)
        if _causal_edge_is_failure_signal(e):
            failure_times.append(e.occurred_at)

    deploy_to_failure = -1.0
    if deploy_ts is not None and failure_times:
        first_fail = min(failure_times)
        try:
            dtf = (first_fail - deploy_ts).total_seconds()
            if dtf >= 0.0:
                deploy_to_failure = min(_TEMPORAL_PROFILE_MAX_S, float(dtf))
        except Exception:
            pass

    failure_spread = -1.0
    if len(failure_times) >= 2:
        failure_spread = min(
            _TEMPORAL_PROFILE_MAX_S,
            float((max(failure_times) - min(failure_times)).total_seconds()),
        )
    elif len(failure_times) == 1:
        failure_spread = 0.0

    cascade_duration = -1.0
    if len(prop_times) >= 2:
        cascade_duration = min(
            _TEMPORAL_PROFILE_MAX_S,
            float((max(prop_times) - min(prop_times)).total_seconds()),
        )
    elif len(prop_times) == 1:
        cascade_duration = 0.0

    remediation_delay = -1.0
    if substrate is not None and canonical_service:
        failure_end = max(failure_times) if failure_times else until_ts
        try:
            fwd = until_ts + timedelta(seconds=remediation_lookforward_s)
            rows = substrate.events_for_service(
                canonical_service, since=failure_end, until=fwd
            )
        except Exception:
            rows = ()
        for row in rows:
            if str(row.get("kind", "")) != "remediation":
                continue
            tsr = row.get("occurred_at")
            if tsr is None:
                continue
            try:
                delta = (tsr - failure_end).total_seconds()
            except Exception:
                continue
            if delta >= 0.0:
                remediation_delay = min(_TEMPORAL_PROFILE_MAX_S, float(delta))
                break

    return TemporalProfile(
        deploy_to_failure,
        failure_spread,
        remediation_delay,
        cascade_duration,
    )


def _temporal_dim_similarity(a: float, b: float, *, log_space: bool) -> float | None:
    if a < 0.0 or b < 0.0:
        return None
    if log_space:
        la = math.log1p(max(a, 0.0))
        lb = math.log1p(max(b, 0.0))
        denom = max(abs(la), abs(lb), 1e-6)
        return 1.0 - min(1.0, abs(la - lb) / denom)
    m = max(a, b, 1.0)
    return 1.0 - min(1.0, abs(a - b) / m)


def temporal_similarity(a: TemporalProfile, b: TemporalProfile) -> float:
    """Weighted similarity in ``[0, 1]`` from normalized timing deltas (unknown dims skipped)."""
    w_deploy = 0.28
    w_spread = 0.22
    w_remed = 0.28
    w_cascade = 0.22
    pairs: list[tuple[float, float]] = []
    s = _temporal_dim_similarity(
        a.deploy_to_failure_seconds, b.deploy_to_failure_seconds, log_space=True
    )
    if s is not None:
        pairs.append((s, w_deploy))
    s = _temporal_dim_similarity(
        a.failure_spread_seconds, b.failure_spread_seconds, log_space=False
    )
    if s is not None:
        pairs.append((s, w_spread))
    s = _temporal_dim_similarity(
        a.remediation_delay_seconds, b.remediation_delay_seconds, log_space=True
    )
    if s is not None:
        pairs.append((s, w_remed))
    s = _temporal_dim_similarity(
        a.cascade_duration_seconds, b.cascade_duration_seconds, log_space=False
    )
    if s is not None:
        pairs.append((s, w_cascade))
    if not pairs:
        return 0.5
    tw = sum(w for _, w in pairs)
    return sum(sim * w for sim, w in pairs) / tw


def compute_retrieval_features(
    fingerprint: IncidentFingerprint,
    edges: Sequence[Any],
    substrate: Any | None,
    canonical_service: str,
    until_ts: datetime,
) -> RetrievalFeatures:
    """Derive deploy proximity, causal/propagation counts, deploy pattern, temporal profile."""
    deploy = 0.0
    if substrate is not None and canonical_service:
        try:
            ld = substrate.last_deploy_before(canonical_service, until_ts)
        except Exception:
            ld = None
        if ld is not None:
            try:
                delta = (until_ts - ld).total_seconds()
            except Exception:
                delta = 0.0
            if delta >= 0.0:
                deploy = max(0.0, 1.0 - min(1.0, delta / 600.0))
    deploy_pattern: tuple[str, ...] = ()
    if substrate is not None and canonical_service:
        try:
            deploy_pattern = extract_deploy_pattern_sequence(
                substrate, canonical_service, until_ts
            )
        except Exception:
            deploy_pattern = ()
    prop_fp = extract_propagation_fingerprint(edges)
    tempo = extract_temporal_profile(edges, substrate, canonical_service, until_ts)
    remed_h = _fingerprint_remediation_hash_parts(
        fingerprint, deploy_pattern, prop_fp, tempo
    )
    return RetrievalFeatures(
        normalize_trigger(fingerprint.trigger_type),
        fingerprint.affected_role,
        fingerprint.canonical_affected,
        fingerprint.upstream_involved,
        deploy,
        len(edges),
        prop_fp.hop_count,
        remed_h,
        deploy_pattern,
        prop_fp,
        tempo,
    )


_DEFAULT_RERANK_WEIGHTS: dict[str, float] = {
    "trigger": 0.33,
    "role": 0.11,
    "upstream": 0.20,
    "propagation": 0.065,
    "temporal": 0.085,
}


def effective_rerank_weights(rctx: RerankContext) -> dict[str, float]:
    """Active linear rerank weights (trigger/role/upstream/propagation/temporal) for diagnostics."""
    w = rctx.rerank_weights
    out = dict(_DEFAULT_RERANK_WEIGHTS)
    if w is None:
        return out
    for k in out:
        if k in w:
            out[k] = float(w[k])  # type: ignore[arg-type]
    return out


def rerank_component_values(
    qfp: IncidentFingerprint,
    qfeat: RetrievalFeatures,
    cfp: IncidentFingerprint,
    cfeat: RetrievalFeatures,
    ctx: RerankContext,
    *,
    idf_stats: IDFStats | None = None,
) -> dict[str, float]:
    """Scalar [0,1] components for adaptive reranking (remed separate, fixed weight).

    When ``idf_stats`` is provided, matched trigger mass is IDF-damped and gated for
    very common trigger families (see :func:`_calibrated_trigger_rerank_value`).
    The linear rerank core (:func:`_weighted_rerank_core`) additionally prevents weighted
    trigger contribution from far exceeding weighted propagation plus upstream.
    """
    trigger = 1.0 if qfeat.norm_trigger == cfeat.norm_trigger else 0.0
    if qfp.affected_role == cfp.affected_role:
        role = 1.0
    elif infer_role_family(qfp.affected_role) == infer_role_family(cfp.affected_role):
        role = _RERANK_ROLE_CLUSTER_SOFT
    else:
        role = 0.0
    upstream = _upstream_score(qfp.upstream_involved, cfp.upstream_involved)

    dq, dc = qfeat.deploy_proximity, cfeat.deploy_proximity
    deploy_sim = 1.0 - min(1.0, abs(dq - dc))
    pq, pc = qfeat.propagation_edge_count, cfeat.propagation_edge_count
    cq, cc = qfeat.causal_edge_count, cfeat.causal_edge_count
    prop_sim = 1.0 - abs(pq - pc) / max(pq, pc, 1)
    causal_sim = 1.0 - abs(cq - cc) / max(cq, cc, 1)
    propagation = 0.5 * prop_sim + 0.5 * causal_sim
    temporal = deploy_sim

    remed = _remediation_similarity(
        ctx.remediation_memory, qfeat.remediation_fp_hash, cfeat.remediation_fp_hash
    )
    if idf_stats is not None:
        trigger = _calibrated_trigger_rerank_value(
            idf_stats, cfeat.norm_trigger, trigger, propagation, upstream
        )
    return {
        "trigger": trigger,
        "role": role,
        "upstream": upstream,
        "propagation": propagation,
        "temporal": temporal,
        "remed": remed,
    }


_TRIGGER_WEIGHTED_EXCESS_BLEND = 0.92


def _weighted_trigger_and_prop_upstream(
    comps: Mapping[str, float],
    wmap: Mapping[str, float],
) -> tuple[float, float]:
    """Linear (propagation+upstream) mass and trigger mass with weighted excess guard."""
    w_t = float(wmap["trigger"])
    w_p = float(wmap["propagation"])
    w_u = float(wmap["upstream"])
    t = float(comps["trigger"])
    p = float(comps["propagation"])
    u = float(comps["upstream"])
    trigger_c = w_t * t
    prop_upstream_c = w_p * p + w_u * u
    if trigger_c > prop_upstream_c + 1e-9:
        trigger_c = prop_upstream_c + _TRIGGER_WEIGHTED_EXCESS_BLEND * (
            trigger_c - prop_upstream_c
        )
    return trigger_c, prop_upstream_c


def _weighted_rerank_core(
    comps: Mapping[str, float],
    wmap: Mapping[str, float] | None,
) -> float:
    m = _DEFAULT_RERANK_WEIGHTS if wmap is None else wmap
    trigger_c, prop_upstream_c = _weighted_trigger_and_prop_upstream(comps, m)
    core = (
        trigger_c
        + float(m["role"]) * float(comps["role"])
        + prop_upstream_c
        + float(m["temporal"]) * float(comps["temporal"])
        + _REMED_RERANK_WEIGHT * float(comps["remed"])
    )
    return core


def _remediation_similarity(
    memory: Any,
    hash_q: str,
    hash_c: str,
) -> float:
    if hash_q == hash_c:
        return 1.0
    try:
        aq = memory.action_keys_for_hash(hash_q)
        ac = memory.action_keys_for_hash(hash_c)
    except Exception:
        return 0.0
    if not aq and not ac:
        return 0.5
    return len(aq & ac) / max(1, len(aq | ac))


@dataclass(slots=True)
class RerankIntrospection:
    """Scalar rerank signals for one query–candidate pair (breakdown + total)."""

    total_score: float
    trigger_score: float
    deploy_pattern_score: float
    propagation_score: float
    topology_score: float
    upstream_score: float
    remediation_score: float
    temporal_score: float
    alias_score: float
    role_score: float
    behavioral_recurrence: float
    recurrence_prior: float
    core_linear: float
    idf_trigger: float
    idf_role_cluster: float
    idf_deploy_shape: float
    idf_propagation_depth: float
    rarity_factor: float
    temporal_shape_similarity: float
    negative_evidence_multiplier: float
    generic_feature_multiplier: float = 1.0
    high_confidence_struct_boost: float = 0.0
    margin_calibration_delta: float = 0.0

    def as_public_dict(self, incident_id: str) -> dict[str, Any]:
        return {
            "incident_id": incident_id,
            "total_score": self.total_score,
            "trigger_score": self.trigger_score,
            "deploy_pattern_score": self.deploy_pattern_score,
            "propagation_score": self.propagation_score,
            "topology_score": self.topology_score,
            "upstream_score": self.upstream_score,
            "remediation_score": self.remediation_score,
            "temporal_score": self.temporal_score,
            "alias_score": self.alias_score,
            "role_score": self.role_score,
            "behavioral_recurrence": self.behavioral_recurrence,
            "recurrence_prior": self.recurrence_prior,
            "core_linear": self.core_linear,
            "idf_trigger": self.idf_trigger,
            "idf_role_cluster": self.idf_role_cluster,
            "idf_deploy_shape": self.idf_deploy_shape,
            "idf_propagation_depth": self.idf_propagation_depth,
            "rarity_factor": self.rarity_factor,
            "temporal_shape_similarity": self.temporal_shape_similarity,
            "negative_evidence_multiplier": self.negative_evidence_multiplier,
            "generic_feature_multiplier": self.generic_feature_multiplier,
            "high_confidence_struct_boost": self.high_confidence_struct_boost,
            "margin_calibration_delta": self.margin_calibration_delta,
        }


def _alias_rerank_component(
    identity: IdentityTracker | None,
    query_canonical: str,
    qfp: IncidentFingerprint,
    cfp: IncidentFingerprint,
    cfeat: RetrievalFeatures,
) -> float:
    qa = (qfp.canonical_affected or "").strip()
    ca = (cfp.canonical_affected or cfeat.canonical_service or "").strip()
    if qa and ca and qa == ca:
        return 1.0
    if identity is None or not query_canonical or not ca:
        return 0.0
    identity.register(query_canonical)
    identity.register(ca)
    if _effective_canonical(identity, query_canonical) == _effective_canonical(identity, ca):
        return 1.0
    return 0.0


def _behavioral_recurrence_bonus(
    qfp: IncidentFingerprint,
    qfeat: RetrievalFeatures,
    cfp: IncidentFingerprint,
    cfeat: RetrievalFeatures,
    ctx: RerankContext,
) -> float:
    b = 0.0
    qdp, cdp = qfeat.deploy_pattern, cfeat.deploy_pattern
    if qdp and cdp and qdp == cdp:
        b += _BEHAVIOR_DEPLOY_EXACT

    qpf, cpf = qfeat.propagation_fingerprint, cfeat.propagation_fingerprint
    if qpf.hop_count > 0 and cpf.hop_count > 0:
        if qpf.edge_types == cpf.edge_types and qpf.degradation_order == cpf.degradation_order:
            b += _BEHAVIOR_PROP_ORDER_EXACT
        qh, ch = qpf.propagation_hops, cpf.propagation_hops
        if qh and ch and qh == ch:
            b += _BEHAVIOR_TIMING_SHAPE

    mem = ctx.remediation_memory
    if mem is not None and qfeat.remediation_fp_hash and qfeat.remediation_fp_hash == cfeat.remediation_fp_hash:
        try:
            aq = mem.action_keys_for_hash(qfeat.remediation_fp_hash)
            ac = mem.action_keys_for_hash(cfeat.remediation_fp_hash)
            if aq and aq == ac:
                b += _BEHAVIOR_REMED_ACTIONS_EXACT
        except Exception:
            pass

    return min(_BEHAVIOR_CAP, b)


def _recurrence_prior(qfeat: RetrievalFeatures, cfeat: RetrievalFeatures) -> float:
    if qfeat.norm_trigger != cfeat.norm_trigger:
        return 0.0
    if not qfeat.remediation_fp_hash or qfeat.remediation_fp_hash != cfeat.remediation_fp_hash:
        return 0.0
    qd = int(qfeat.propagation_fingerprint.propagation_depth)
    cd = int(cfeat.propagation_fingerprint.propagation_depth)
    if qd != cd:
        return 0.0
    qpfx = (
        tuple(qfeat.deploy_pattern[:2])
        if len(qfeat.deploy_pattern) >= 2
        else qfeat.deploy_pattern
    )
    cpfx = (
        tuple(cfeat.deploy_pattern[:2])
        if len(cfeat.deploy_pattern) >= 2
        else cfeat.deploy_pattern
    )
    if qpfx != cpfx:
        return 0.0
    return _RECURRENCE_PRIOR


def _median_float(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    xs = sorted(float(x) for x in values)
    n = len(xs)
    mid = n // 2
    if n % 2:
        return xs[mid]
    return 0.5 * (xs[mid - 1] + xs[mid])


def _generic_feature_multiplier(
    stats: IDFStats | None,
    idf_t: float,
    idf_r: float,
    idf_d: float,
    idf_p: float,
    comps: Mapping[str, float],
    qfeat: RetrievalFeatures,
    cfeat: RetrievalFeatures,
    qfp: IncidentFingerprint,
    cfp: IncidentFingerprint,
) -> float:
    """Down-weight scores driven by corpus-common structure (IDF-style, no new features)."""
    g = 1.0
    total_docs = int(stats.total_docs) if stats is not None else 0
    if stats is not None and total_docs > 0:
        midf = (idf_t + idf_r + idf_d + idf_p) / 4.0
        denom = math.log(max(2, total_docs))
        norm = min(1.0, midf / denom) if denom > 0 else 0.5
        commonness = max(0.0, 1.0 - norm)
        g *= 1.0 - _GEN_SUPPRESS_STRENGTH * min(1.0, commonness * 1.15)
        shape_k = _deploy_pattern_shape_key(cfeat)
        dfc = int(stats.df_deploy_shape.get(shape_k, 0))
        if dfc * 2 >= total_docs and cfeat.deploy_pattern:
            g *= 0.94
        tdf = int(stats.df_trigger.get(cfeat.norm_trigger, 0))
        if tdf * 2 >= max(1, total_docs):
            g *= 0.905
    up = float(comps["upstream"])
    if up < 0.11 and (qfp.upstream_involved or cfp.upstream_involved):
        g *= 0.93
    qd = int(qfeat.propagation_fingerprint.propagation_depth)
    cd = int(cfeat.propagation_fingerprint.propagation_depth)
    if qd >= 2 and cd == 0:
        g *= 0.92
    if stats is not None and total_docs > 0 and idf_r > 0:
        rc = _role_cluster(cfp)
        rfc = int(stats.df_role_cluster.get(rc, 0))
        if rc == "api" and rfc * 2 >= total_docs:
            g *= 0.945
    return max(_GEN_MULT_MIN, g)


def _high_confidence_structural_boost(
    qfp: IncidentFingerprint,
    cfp: IncidentFingerprint,
    qfeat: RetrievalFeatures,
    cfeat: RetrievalFeatures,
    comps: Mapping[str, float],
    prop_fp_sim: float,
    tempo_shape: float,
    alias_s: float,
    drift_lineage: float,
) -> float:
    """Extra separation only when several structural signals align strongly."""
    if prop_fp_sim < _HIGH_CONF_PROP_T:
        return 0.0
    if tempo_shape < _HIGH_CONF_TEMPO_T:
        return 0.0
    if float(comps["upstream"]) < _HIGH_CONF_UP_T:
        return 0.0
    if alias_s < _HIGH_CONF_ALIAS_STRONG and drift_lineage < _HIGH_CONF_DRIFT_LINEAGE_MIN:
        return 0.0
    qpf, cpf = qfeat.propagation_fingerprint, cfeat.propagation_fingerprint
    if not (qpf.root_degradation_role and qpf.root_degradation_role == cpf.root_degradation_role):
        return 0.0
    if not (qpf.terminal_failure_role and qpf.terminal_failure_role == cpf.terminal_failure_role):
        return 0.0
    if qpf.degradation_order and cpf.degradation_order and qpf.degradation_order != cpf.degradation_order:
        if prop_fp_sim < 0.97:
            return 0.0
    return _HIGH_STRUCT_BOOST_CAP


def _topology_mismatch_multiplier(
    ctx: RerankContext,
    qroot: str,
    croot: str,
) -> float:
    """Penalty when dependency graph says query and candidate roots are unrelated."""
    g = ctx.dependency_graph
    if g is None or not qroot or not croot or qroot == croot:
        return 1.0
    try:
        nbr = g.neighbors(qroot)
    except Exception:
        return 1.0
    if croot in nbr:
        return 1.0
    return 0.908


def _known_temporal_dim_count(t: TemporalProfile) -> int:
    n = 0
    for v in (
        t.deploy_to_failure_seconds,
        t.failure_spread_seconds,
        t.remediation_delay_seconds,
        t.cascade_duration_seconds,
    ):
        if v >= 0.0:
            n += 1
    return n


def _negative_evidence_multiplier(
    qfp: IncidentFingerprint,
    qfeat: RetrievalFeatures,
    cfp: IncidentFingerprint,
    cfeat: RetrievalFeatures,
    comps: Mapping[str, float],
    tempo_shape: float,
    ctx: RerankContext,
    qroot: str,
    croot: str,
) -> float:
    """Bounded ``(0, 1]`` multiplier penalizing contradictory query–candidate structure."""
    m = 1.0

    qd = int(qfeat.propagation_fingerprint.propagation_depth)
    cd = int(cfeat.propagation_fingerprint.propagation_depth)
    den = max(qd, cd, 1)
    rel_depth = abs(qd - cd) / float(den)
    if rel_depth >= 0.55:
        m *= 1.0 - 0.118 * min(1.0, (rel_depth - 0.55) / 0.45)

    q_dep = "deploy" in qfeat.deploy_pattern
    c_dep = "deploy" in cfeat.deploy_pattern
    if q_dep != c_dep:
        m *= 0.865

    q_near = qfeat.deploy_proximity >= 0.2
    c_near = cfeat.deploy_proximity >= 0.2
    if q_near != c_near:
        m *= 0.922

    if (
        qd >= 2
        and cd == 0
        and int(qfeat.propagation_fingerprint.hop_count) >= 2
        and int(cfeat.propagation_fingerprint.hop_count) == 0
    ):
        m *= 0.918

    if (
        qfp.upstream_involved
        and cfp.upstream_involved
        and not (qfp.upstream_involved & cfp.upstream_involved)
    ):
        m *= 0.842

    if (
        qfeat.remediation_fp_hash
        and cfeat.remediation_fp_hash
        and qfeat.remediation_fp_hash != cfeat.remediation_fp_hash
        and float(comps["remed"]) < 0.4
    ):
        m *= 0.815

    if (
        tempo_shape < 0.32
        and _known_temporal_dim_count(qfeat.temporal_profile) >= 2
        and _known_temporal_dim_count(cfeat.temporal_profile) >= 2
    ):
        m *= 0.902

    m *= _topology_mismatch_multiplier(ctx, qroot, croot)

    return max(_NEGATIVE_EVIDENCE_FLOOR, m)


def _rerank_introspection(
    qfp: IncidentFingerprint,
    qfeat: RetrievalFeatures,
    cfp: IncidentFingerprint,
    cfeat: RetrievalFeatures,
    ctx: RerankContext,
) -> RerankIntrospection:
    comps = rerank_component_values(qfp, qfeat, cfp, cfeat, ctx, idf_stats=ctx.idf_stats)
    core_linear = _weighted_rerank_core(comps, ctx.rerank_weights)

    deploy_pat = 0.0
    if qfeat.deploy_pattern and cfeat.deploy_pattern:
        deploy_pat = sequence_similarity(qfeat.deploy_pattern, cfeat.deploy_pattern)

    idn = ctx.identity
    qc = ctx.query_canonical
    cname = cfeat.canonical_service or cfp.canonical_affected
    qroot = _effective_canonical(idn, qc) if qc else ""
    croot = _effective_canonical(idn, cname) if cname else ""

    drift = _drift_continuity_rerank(idn, qfp, cfp)
    graph_bonus = _graph_neighborhood_overlap_rerank(
        ctx.dependency_graph, idn, qroot, croot
    )
    hist_bonus = _historical_neighborhood_rerank(
        ctx.neighborhood_memory, idn, qroot, croot
    )
    topology = drift + graph_bonus + hist_bonus

    qpf, cpf = qfeat.propagation_fingerprint, cfeat.propagation_fingerprint
    if qpf.hop_count == 0 and cpf.hop_count == 0:
        prop_fp_sim = 0.0
    else:
        prop_fp_sim = propagation_similarity(qpf, cpf)

    alias_s = _alias_rerank_component(idn, qc, qfp, cfp, cfeat)
    behavioral = _behavioral_recurrence_bonus(qfp, qfeat, cfp, cfeat, ctx)
    rec_prior = _recurrence_prior(qfeat, cfeat)

    stats = ctx.idf_stats
    idf_t = idf_r = idf_d = idf_p = 0.0
    rarity_factor = 1.0
    if stats is not None and stats.total_docs > 0:
        idf_t = _idf_value(stats.total_docs, int(stats.df_trigger.get(cfeat.norm_trigger, 0)))
        idf_r = _idf_value(stats.total_docs, int(stats.df_role_cluster.get(_role_cluster(cfp), 0)))
        idf_d = _idf_value(
            stats.total_docs, int(stats.df_deploy_shape.get(_deploy_pattern_shape_key(cfeat), 0))
        )
        pd = int(cfeat.propagation_fingerprint.propagation_depth)
        idf_p = _idf_value(stats.total_docs, int(stats.df_prop_depth.get(pd, 0)))
        mean_idf = (idf_t + idf_r + idf_d + idf_p) / 4.0
        denom = math.log(max(2, stats.total_docs))
        norm = min(1.0, mean_idf / denom)
        floor = (
            _IDF_FLOOR_LARGE_CORPUS
            if stats.total_docs > _IDF_CORPUS_SMALL
            else _IDF_FLOOR_SMALL_CORPUS
        )
        rarity_factor = floor + (1.0 - floor) * norm

    tempo_shape = temporal_similarity(qfeat.temporal_profile, cfeat.temporal_profile)
    neg_m = _negative_evidence_multiplier(
        qfp, qfeat, cfp, cfeat, comps, tempo_shape, ctx, qroot, croot
    )
    generic_mult = _generic_feature_multiplier(
        stats, idf_t, idf_r, idf_d, idf_p, comps, qfeat, cfeat, qfp, cfp
    )
    hc_boost = _high_confidence_structural_boost(
        qfp,
        cfp,
        qfeat,
        cfeat,
        comps,
        prop_fp_sim,
        tempo_shape,
        alias_s,
        drift,
    )

    pre_beh = (
        core_linear
        + _SEQUENCE_RERANK_WEIGHT * deploy_pat
        + topology
        + _PROPAGATION_FP_RERANK_WEIGHT * prop_fp_sim
        + _ALIAS_RERANK_WEIGHT * alias_s
        + _TEMPORAL_SHAPE_RERANK_WEIGHT * tempo_shape
    )
    pre_beh *= rarity_factor
    pre_beh *= neg_m
    pre_beh *= generic_mult
    pre_beh += hc_boost
    total = min(1.0, pre_beh + behavioral + rec_prior)

    return RerankIntrospection(
        total_score=total,
        trigger_score=float(comps["trigger"]),
        deploy_pattern_score=deploy_pat,
        propagation_score=prop_fp_sim,
        topology_score=topology,
        upstream_score=float(comps["upstream"]),
        remediation_score=float(comps["remed"]),
        temporal_score=float(comps["temporal"]),
        alias_score=alias_s,
        role_score=float(comps["role"]),
        behavioral_recurrence=behavioral,
        recurrence_prior=rec_prior,
        core_linear=core_linear,
        idf_trigger=idf_t,
        idf_role_cluster=idf_r,
        idf_deploy_shape=idf_d,
        idf_propagation_depth=idf_p,
        rarity_factor=rarity_factor,
        temporal_shape_similarity=tempo_shape,
        negative_evidence_multiplier=neg_m,
        generic_feature_multiplier=generic_mult,
        high_confidence_struct_boost=hc_boost,
        margin_calibration_delta=0.0,
    )


def _apply_margin_rank_calibration(
    scores: list[float],
    rarity: list[float],
    gen_mult: list[float],
) -> tuple[list[float], list[float]]:
    """Post-score pass: widen separation using adjacent gaps × rarity / generic slack (O(n))."""
    n = len(scores)
    if n == 0:
        return [], []
    if n == 1:
        return [scores[0]], [0.0]
    gaps = [max(0.0, scores[i] - scores[i + 1]) for i in range(n - 1)]
    med = _median_float(gaps)
    adjusted: list[float] = []
    deltas: list[float] = []
    for i in range(n):
        g_next = gaps[i] if i < n - 1 else gaps[-1]
        rarity_i = float(rarity[i])
        gen_slack = max(0.0, 1.0 - float(gen_mult[i]))
        dom = max(0.0, g_next - med)
        clust = max(0.0, med - g_next)
        delta = (
            _MARGIN_DOM_ALPHA * dom * (rarity_i + _RARE_MARGIN_K)
            - _MARGIN_CLUSTER_BETA * clust * (gen_slack + _GEN_MARGIN_K)
        )
        adjusted.append(scores[i] + delta)
        deltas.append(delta)
    return adjusted, deltas


def _ranking_spread_diagnostics(adjusted_scores: list[float]) -> dict[str, float | int]:
    """Lightweight spread / near-tie stats for the last reranked pool (calibration visibility)."""
    if not adjusted_scores:
        return {
            "mean_top5_adjacent_spread": 0.0,
            "top5_score_range": 0.0,
            "top1_top2_margin": 0.0,
            "near_tie_count_below_top1": 0,
        }
    n5 = min(5, len(adjusted_scores))
    top = adjusted_scores[:n5]
    spreads = [top[i] - top[i + 1] for i in range(len(top) - 1)]
    mean_sp = sum(spreads) / len(spreads) if spreads else 0.0
    t1t2 = top[0] - top[1] if len(top) >= 2 else 0.0
    best = adjusted_scores[0]
    band = 0.012
    near_tie = sum(1 for s in adjusted_scores[1:] if s >= best - band)
    rng = top[0] - top[-1] if len(top) >= 2 else 0.0
    return {
        "mean_top5_adjacent_spread": mean_sp,
        "top5_score_range": rng,
        "top1_top2_margin": t1t2,
        "near_tie_count_below_top1": int(near_tie),
    }


def rerank_score_breakdown(
    qfp: IncidentFingerprint,
    qfeat: RetrievalFeatures,
    cfp: IncidentFingerprint,
    cfeat: RetrievalFeatures,
    ctx: RerankContext,
    *,
    incident_id: str = "",
) -> dict[str, Any]:
    """Public rerank decomposition for one candidate (no side effects)."""
    return _rerank_introspection(qfp, qfeat, cfp, cfeat, ctx).as_public_dict(incident_id)


def _rerank_score(
    qfp: IncidentFingerprint,
    qfeat: RetrievalFeatures,
    cfp: IncidentFingerprint,
    cfeat: RetrievalFeatures,
    ctx: RerankContext,
) -> float:
    return _rerank_introspection(qfp, qfeat, cfp, cfeat, ctx).total_score


def _stage1_recall_score(
    qfp: IncidentFingerprint,
    qfeat: RetrievalFeatures,
    cfp: IncidentFingerprint,
    cfeat: RetrievalFeatures,
    *,
    locale: frozenset[str] | None,
    identity: IdentityTracker | None,
) -> float:
    """Broad recall-oriented score for selecting up to 50 candidates."""
    s = 0.0
    if qfeat.norm_trigger == cfeat.norm_trigger:
        s += 3.0
    if qfp.affected_role == cfp.affected_role:
        s += 1.2
    elif infer_role_family(qfp.affected_role) == infer_role_family(cfp.affected_role):
        s += 0.8
    q_canon_resolved = _effective_canonical(identity, qfeat.canonical_service)
    c_canon_resolved = _effective_canonical(identity, cfeat.canonical_service)
    if q_canon_resolved and q_canon_resolved == c_canon_resolved:
        s += 2.5
    inter = len(qfp.upstream_involved & cfp.upstream_involved)
    if inter > 0:
        s += min(2.4, inter * 0.9)
    elif not qfp.upstream_involved and not cfp.upstream_involved:
        s += 0.8
    if locale is not None:
        s += _stage1_locale_bonus(identity, locale, cfp, cfeat)
    qd = int(qfeat.propagation_fingerprint.propagation_depth)
    cd = int(cfeat.propagation_fingerprint.propagation_depth)
    if min(qd, cd) > 0 and abs(qd - cd) <= 1:
        s += 0.42
    if qfeat.remediation_fp_hash and qfeat.remediation_fp_hash == cfeat.remediation_fp_hash:
        s += 1.2
    qpat, cpat = qfeat.deploy_pattern, cfeat.deploy_pattern
    if qpat and cpat:
        if qpat == cpat:
            s += 1.1
        elif qpat[0] == cpat[0]:
            s += 0.28
    return s


@overload
def structural_similarity(
    left: IncidentFingerprint,
    right: IncidentFingerprint,
    *,
    weights: Sequence[float] = ...,
    query_context: Mapping[str, Any] | None = ...,
    candidate_context: Mapping[str, Any] | None = ...,
    debug: Literal[False] = False,
) -> float: ...


@overload
def structural_similarity(
    left: IncidentFingerprint,
    right: IncidentFingerprint,
    *,
    weights: Sequence[float] = ...,
    query_context: Mapping[str, Any] | None = ...,
    candidate_context: Mapping[str, Any] | None = ...,
    debug: Literal[True],
) -> dict[str, float]: ...


def structural_similarity(
    left: IncidentFingerprint,
    right: IncidentFingerprint,
    *,
    weights: Sequence[float] = _DEFAULT_WEIGHTS,
    query_context: Mapping[str, Any] | None = None,
    candidate_context: Mapping[str, Any] | None = None,
    debug: bool = False,
) -> float | dict[str, float]:
    """Score in ``[0, 1]`` with recall-friendly components; optional score breakdown."""
    bd = _structural_similarity_parts(
        left,
        right,
        weights=weights,
        query_context=query_context,
        candidate_context=candidate_context,
    )
    if debug:
        return bd
    return bd["final"]


def _structural_similarity_parts(
    left: IncidentFingerprint,
    right: IncidentFingerprint,
    *,
    weights: Sequence[float],
    query_context: Mapping[str, Any] | None,
    candidate_context: Mapping[str, Any] | None,
) -> dict[str, float]:
    if len(weights) != 3:
        raise ValueError("weights must have exactly three components")
    w_trigger, w_role, w_upstream = weights
    total = w_trigger + w_role + w_upstream
    if total <= 0:
        raise ValueError("weights must sum to a positive value")
    w_trigger, w_role, w_upstream = (
        w_trigger / total,
        w_role / total,
        w_upstream / total,
    )

    nt_l = normalize_trigger(left.trigger_type)
    nt_r = normalize_trigger(right.trigger_type)
    trigger_score = 1.0 if nt_l == nt_r else 0.0

    role_score = 1.0 if left.affected_role == right.affected_role else 0.0
    upstream_score = _upstream_score(left.upstream_involved, right.upstream_involved)

    base = (
        w_trigger * trigger_score
        + w_role * role_score
        + w_upstream * upstream_score
    )

    temporal_score = _temporal_boost(query_context, candidate_context)
    alias_score = _alias_boost(left, right)
    role_family_score = _role_family_boost(left, right)

    final = base + temporal_score + alias_score + role_family_score
    if final > 1.0:
        final = 1.0
    return {
        "trigger_score": trigger_score,
        "upstream_score": upstream_score,
        "role_score": role_score,
        "temporal_score": temporal_score,
        "alias_score": alias_score,
        "role_family_score": role_family_score,
        "final": final,
    }


class FingerprintExtractor:
    """Build fingerprints by resolving services through ``IdentityTracker``."""

    __slots__ = ("_identity", "_role_resolver")

    def __init__(
        self,
        identity: IdentityTracker | None = None,
        role_resolver: RoleResolver | None = None,
    ) -> None:
        self._identity = identity if identity is not None else IdentityTracker()
        self._role_resolver = role_resolver or family_aware_role

    @property
    def identity(self) -> IdentityTracker:
        return self._identity

    def extract(self, incident: Mapping[str, Any] | Event) -> IncidentFingerprint:
        if isinstance(incident, Event):
            return self.extract_event(incident)
        return self._extract_mapping(incident)

    def extract_event(self, event: Event) -> IncidentFingerprint:
        payload = event.payload or {}
        merged: dict[str, Any] = {
            "trigger_type": payload.get("trigger_type", event.kind),
            "service": event.service,
            "affected_role": payload.get("affected_role"),
            "upstream": payload.get("upstream", payload.get("upstream_services", ())),
            "upstream_roles": payload.get("upstream_roles"),
        }
        return self._extract_mapping(merged)

    def _extract_mapping(self, incident: Mapping[str, Any]) -> IncidentFingerprint:
        trigger_type = str(incident.get("trigger_type", "unknown"))

        affected_role = incident.get("affected_role")
        service_hint = incident.get("service") or incident.get("affected_service")
        if affected_role is not None:
            role = self._role_resolver(str(affected_role))
            if service_hint is None:
                service_hint = str(affected_role)
        else:
            service = incident.get("service", incident.get("affected_service", "unknown"))
            service_hint = service_hint or service
            role = self._role_for_service(str(service))

        upstream_roles = incident.get("upstream_roles")
        if upstream_roles is not None:
            upstream = _normalize_role_set(upstream_roles, self._role_resolver)
        else:
            raw_upstream = incident.get(
                "upstream",
                incident.get("upstream_services", incident.get("upstream_involved", ())),
            )
            if not isinstance(raw_upstream, (list, tuple, set, frozenset)):
                raw_upstream = ()
            upstream = frozenset(
                self._role_for_service(str(name)) for name in raw_upstream
            )

        upstream = frozenset(r for r in upstream if r != role)

        canonical_affected = ""
        if service_hint is not None:
            sh = str(service_hint).strip()
            if sh:
                self._identity.register(sh)
                canonical_affected = self._identity.resolve(sh)

        return IncidentFingerprint(
            trigger_type,
            role,
            upstream,
            canonical_affected,
        )

    def _role_for_service(self, service: str) -> str:
        if not service:
            service = "unknown"
        self._identity.register(service)
        canonical = self._identity.resolve(service)
        return self._role_resolver(canonical)


def _match_context_from_incident(incident: Mapping[str, Any]) -> dict[str, Any]:
    ctx: dict[str, Any] = {}
    if "deploy_window" in incident and incident.get("deploy_window") is not None:
        ctx["deploy_window"] = incident.get("deploy_window")
    if incident.get("post_deploy_metric"):
        ctx["post_deploy_metric"] = True
    return ctx


def _query_context_from_mapping(query: Mapping[str, Any]) -> dict[str, Any]:
    return _match_context_from_incident(query)


def _deploy_proximity_bucket(deploy: float) -> int:
    if deploy <= 0.0:
        return 0
    return min(10, max(0, int(deploy * 11)))


def _role_cluster(fp: IncidentFingerprint) -> str:
    base = infer_role_family(fp.canonical_affected or fp.affected_role)
    if base == "other" and fp.upstream_involved:
        ups = sorted(list(fp.upstream_involved))
        return f"other-ups-{','.join(ups)}"
    return base


def _deploy_pattern_shape_key(rf: RetrievalFeatures) -> tuple[str, ...]:
    """Deploy-pattern shape key aligned with broad-pool ``deploy_pattern_shape`` indexing."""
    dp = rf.deploy_pattern
    if not dp:
        return ()
    if len(dp) >= 2:
        return tuple(dp[:2])
    return (dp[0],)


def _idf_value(total_docs: int, doc_freq: int) -> float:
    """``log(total_docs / (1 + doc_freq))``, clamped to ``>= 0`` (natural log)."""
    if total_docs <= 0:
        return 0.0
    return max(0.0, math.log(total_docs / (1 + doc_freq)))


@dataclass(frozen=True, slots=True)
class IDFStats:
    """Corpus document-frequency snapshot for structural IDF reranking."""

    total_docs: int
    df_trigger: Mapping[str, int]
    df_role_cluster: Mapping[str, int]
    df_deploy_shape: Mapping[tuple[str, ...], int]
    df_prop_depth: Mapping[int, int]


_TRIGGER_FREQ_TOP_FRACTION = 0.20
_TRIGGER_GATE_MIN = 0.6
_TRIGGER_GATE_MAX = 0.8
_TRIGGER_IDF_SUPPRESS_BASE = 0.92
_TRIGGER_IDF_SUPPRESS_SPAN = 0.08
_TRIGGER_GATE_MIN_UNIQUE_TRIGGERS = 20


def _trigger_high_frequency_threshold(df_trigger: Mapping[str, int]) -> int | None:
    """Minimum doc-frequency among the top ``_TRIGGER_FREQ_TOP_FRACTION`` of trigger types."""
    if not df_trigger:
        return None
    pairs = sorted(((int(d), k) for k, d in df_trigger.items()), reverse=True)
    n = len(pairs)
    k = max(1, int(math.ceil(_TRIGGER_FREQ_TOP_FRACTION * n)))
    tier = pairs[:k]
    return min(t[0] for t in tier)


def _calibrated_trigger_rerank_value(
    stats: IDFStats,
    norm_trigger: str,
    trigger_raw: float,
    propagation: float,
    upstream: float,
) -> float:
    """Corpus-aware trigger mass for rerank ``comps`` (matched rows only).

    Applies log IDF damping when the trigger family is very common (``df`` large
    relative to corpus size), and a top-frequency gate (top 20% of trigger
    types by count, when at least ``_TRIGGER_GATE_MIN_UNIQUE_TRIGGERS`` distinct
    families exist) scaling roughly ``_TRIGGER_GATE_MIN``..``_TRIGGER_GATE_MAX``.

    The linear rerank core additionally applies :func:`_weighted_trigger_and_prop_upstream`
    so weighted trigger contribution cannot run far above propagation+upstream.
    """
    if trigger_raw <= 0.0:
        return 0.0
    n = int(stats.total_docs)
    if n <= 0:
        return min(trigger_raw, float(propagation) + float(upstream))

    df = int(stats.df_trigger.get(norm_trigger, 0))
    idf = _idf_value(n, df)
    denom = math.log(max(2, n))
    idf_norm = min(1.0, idf / denom) if denom > 0 else 0.5
    if df * 2 < max(6, n):
        idf_suppress = 1.0
    else:
        idf_suppress = _TRIGGER_IDF_SUPPRESS_BASE + _TRIGGER_IDF_SUPPRESS_SPAN * idf_norm

    gate = 1.0
    if len(stats.df_trigger) >= _TRIGGER_GATE_MIN_UNIQUE_TRIGGERS:
        th = _trigger_high_frequency_threshold(stats.df_trigger)
        if th is not None and df >= th:
            mx = max(int(v) for v in stats.df_trigger.values())
            if mx <= th:
                gate = 0.5 * (_TRIGGER_GATE_MIN + _TRIGGER_GATE_MAX)
            else:
                span = float(mx - th)
                t = (df - th) / span if span > 0 else 1.0
                gate = _TRIGGER_GATE_MAX - (_TRIGGER_GATE_MAX - _TRIGGER_GATE_MIN) * min(1.0, t)

    t_adj = trigger_raw * idf_suppress * gate
    return max(0.0, min(1.0, t_adj))


def _retrieval_recall_debug_enabled(sig: Mapping[str, Any] | None) -> bool:
    if sig and sig.get("_retrieval_recall_debug"):
        return True
    v = os.environ.get("CONTOXTER_RETRIEVAL_DEBUG", "")
    return v in ("1", "true", "yes", "TRUE")


def _rerank_misrank_print_enabled(sig: Mapping[str, Any] | None) -> bool:
    if sig and sig.get("_retrieval_rerank_debug"):
        return True
    return _retrieval_recall_debug_enabled(sig)


def _pool_has_same_family(
    indices: Sequence[int],
    query_fp: IncidentFingerprint,
    corpus: Sequence[tuple[str, IncidentFingerprint, Any, RetrievalFeatures]],
) -> bool:
    qf = _role_cluster(query_fp)
    if qf == "unknown":
        return True
    for idx in indices:
        fp = corpus[idx][1]
        if _role_cluster(fp) == qf:
            return True
    return False


def _top_k_has_same_family(
    query_fp: IncidentFingerprint,
    results: Sequence[MatchResult],
) -> bool:
    qf = _role_cluster(query_fp)
    if qf == "unknown":
        return True
    for m in results:
        if _role_cluster(m.fingerprint) == qf:
            return True
    return False


def _maybe_print_retrieval_recall_debug(
    *,
    query_fp: IncidentFingerprint,
    query_sig: Mapping[str, Any] | None,
    expected_family: str,
    raw_count: int,
    diverse_count: int,
    raw_same_family: bool,
    diverse_same_family: bool,
    ranked_preview: Sequence[tuple[float, str]],
    topk_ok: bool,
    reranked_rows: Sequence[
        tuple[float, str, IncidentFingerprint, RerankIntrospection, tuple[str, ...]]
    ]
    | None = None,
    pool_rank_by_id: Mapping[str, int] | None = None,
    rerank_rank_by_id: Mapping[str, int] | None = None,
) -> None:
    if topk_ok or not _rerank_misrank_print_enabled(query_sig):
        return
    lines = [
        "[contexter retrieval recall]",
        f"  query_trigger={query_fp.trigger_type!r}",
        f"  expected_family={expected_family!r}",
        f"  candidate_raw_union_size={raw_count}",
        f"  candidate_diverse_pool_size={diverse_count}",
        f"  same_family_in_raw_union={raw_same_family}",
        f"  same_family_in_diverse_pool={diverse_same_family}",
        "  top_10_rerank_scores=" + repr(list(ranked_preview[:10])),
    ]
    print("\n".join(lines), flush=True)

    if not (diverse_same_family and not topk_ok and reranked_rows and pool_rank_by_id):
        return

    rrank = rerank_rank_by_id or {}
    lines2 = [
        "[contexter rerank misrank] same_family in pool but not in top-5",
        f"  family_number={expected_family!r}",
        "  top_10_candidates (rerank order):",
    ]
    for i, row in enumerate(reranked_rows[:10], start=1):
        score, iid, _fp, intro, _tags = row
        rr = rrank.get(iid, i)
        pr = pool_rank_by_id.get(iid, -1)
        lines2.append(f"    #{i} id={iid!r} total={score:.4f} pool_rank={pr} rerank_rank={rr}")
        bd = intro.as_public_dict(iid)
        lines2.append(f"      breakdown={bd!r}")
    print("\n".join(lines2), flush=True)


def _diversity_pick_indices(
    sorted_indices: Sequence[int],
    corpus: Sequence[tuple[str, IncidentFingerprint, Any, RetrievalFeatures]],
    *,
    identity: IdentityTracker | None,
    max_pool: int,
    min_pool: int,
    per_canonical: int,
    per_cluster: int,
) -> list[int]:
    """Prefer recall order while capping per canonical identity and role cluster."""

    def canon_of(idx: int) -> str:
        _iid, fp, _ctx, rf = corpus[idx]
        return _effective_canonical(identity, rf.canonical_service or fp.canonical_affected) or (
            fp.canonical_affected or "unknown"
        )

    tiers: tuple[tuple[int, int], ...] = (
        (per_canonical, per_cluster),
        (per_canonical + 2, per_cluster + 2),
        (per_canonical + 6, per_cluster + 6),
        (9999, 9999),
    )
    picked: list[int] = []
    picked_set: set[int] = set()
    canon_used: dict[str, int] = defaultdict(int)
    cluster_used: dict[str, int] = defaultdict(int)

    for max_c, max_f in tiers:
        for idx in sorted_indices:
            if len(picked) >= max_pool:
                return picked[:max_pool]
            if idx in picked_set:
                continue
            c = canon_of(idx)
            cl = _role_cluster(corpus[idx][1])
            if canon_used[c] >= max_c or cluster_used[cl] >= max_f:
                continue
            canon_used[c] += 1
            cluster_used[cl] += 1
            picked.append(idx)
            picked_set.add(idx)
        if len(picked) >= min_pool:
            break
    return picked[:max_pool]


class FingerprintMatcher:
    """In-memory corpus with two-stage retrieval (recall pool → structural rerank)."""

    __slots__ = (
        "_by_canonical",
        "_by_deploy_bucket",
        "_by_deploy_pattern",
        "_by_deploy_prefix",
        "_by_prop_depth",
        "_by_remed_action",
        "_by_remed_hash",
        "_by_role",
        "_by_role_family",
        "_by_trigger",
        "_corpus",
        "_df_deploy_shape",
        "_df_prop_depth",
        "_df_role_cluster",
        "_df_trigger",
        "_extractor",
        "_last_pool_stats",
        "_last_ranking_calib_stats",
        "_remediation_memory",
        "_rf_by_id",
        "_weights",
    )

    def __init__(
        self,
        extractor: FingerprintExtractor | None = None,
        *,
        weights: Sequence[float] = _DEFAULT_WEIGHTS,
        remediation_memory: Any | None = None,
    ) -> None:
        self._extractor = extractor or FingerprintExtractor()
        self._weights = tuple(weights)
        self._remediation_memory = remediation_memory
        self._corpus: list[
            tuple[str, IncidentFingerprint, dict[str, Any], RetrievalFeatures]
        ] = []
        self._by_trigger = defaultdict(list)
        self._by_canonical = defaultdict(list)
        self._by_role = defaultdict(list)
        self._by_role_family = defaultdict(list)
        self._by_prop_depth = defaultdict(list)
        self._by_deploy_pattern = defaultdict(list)
        self._by_deploy_prefix = defaultdict(list)
        self._by_remed_hash = defaultdict(list)
        self._by_deploy_bucket = defaultdict(list)
        self._by_remed_action = defaultdict(list)
        self._rf_by_id: dict[str, RetrievalFeatures] = {}
        self._last_pool_stats: dict[str, int] | None = None
        self._last_ranking_calib_stats: dict[str, float | int] | None = None
        self._df_trigger: defaultdict[str, int] = defaultdict(int)
        self._df_role_cluster: defaultdict[str, int] = defaultdict(int)
        self._df_deploy_shape: defaultdict[tuple[str, ...], int] = defaultdict(int)
        self._df_prop_depth: defaultdict[int, int] = defaultdict(int)

    def __len__(self) -> int:
        return len(self._corpus)

    def retrieval_features_for_incident(self, incident_id: str) -> RetrievalFeatures | None:
        """Return stored :class:`RetrievalFeatures` for a corpus id (for adaptation / audit)."""
        return self._rf_by_id.get(incident_id)

    def last_retrieval_pool_stats(self) -> dict[str, int] | None:
        """Last broad-pool sizes from ``top_k`` (``raw_union``, ``diverse_pool``); ``None`` if none."""
        if self._last_pool_stats is None:
            return None
        return dict(self._last_pool_stats)

    def last_ranking_calibration_stats(self) -> dict[str, float | int] | None:
        """Last stage-2 spread / near-tie diagnostics after margin calibration; ``None`` if none."""
        if self._last_ranking_calib_stats is None:
            return None
        return dict(self._last_ranking_calib_stats)

    def _introspection_for_incident_id(
        self,
        query_fp: IncidentFingerprint,
        qfeat: RetrievalFeatures,
        rctx: RerankContext,
        incident_id: str,
    ) -> RerankIntrospection | None:
        rctx_idf = replace(rctx, idf_stats=self._idf_stats())
        for iid, cfp, _ctx, cfeat in self._corpus:
            if iid == incident_id:
                return _rerank_introspection(query_fp, qfeat, cfp, cfeat, rctx_idf)
        return None

    def _idf_stats(self) -> IDFStats:
        return IDFStats(
            total_docs=len(self._corpus),
            df_trigger=self._df_trigger,
            df_role_cluster=self._df_role_cluster,
            df_deploy_shape=self._df_deploy_shape,
            df_prop_depth=self._df_prop_depth,
        )

    def calibrate_weights(
        self,
        debug_runs: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Lightweight stats: signal means on top-5 hits vs misses, false-positive dominance counts."""
        keys = (
            "trigger_score",
            "deploy_pattern_score",
            "propagation_score",
            "topology_score",
            "upstream_score",
            "remediation_score",
            "temporal_score",
            "alias_score",
            "role_score",
            "behavioral_recurrence",
            "recurrence_prior",
            "core_linear",
            "temporal_shape_similarity",
            "negative_evidence_multiplier",
        )
        n_hit = 0
        n_miss = 0
        sum_hit = {k: 0.0 for k in keys}
        sum_miss = {k: 0.0 for k in keys}
        fp_dom = {k: 0 for k in keys}
        n_ok = 0
        eps = 0.02

        for run in debug_runs:
            rctx = run.get("rerank_context")
            if not isinstance(rctx, RerankContext):
                continue
            expected = str(run.get("expected_incident_id", "") or "")
            if not expected:
                continue
            q = run["query"]
            try:
                qfp = q if isinstance(q, IncidentFingerprint) else self._extractor.extract(q)
            except (KeyError, TypeError, ValueError):
                continue
            qfeat = rctx.query_features
            intro_exp = self._introspection_for_incident_id(qfp, qfeat, rctx, expected)
            if intro_exp is None:
                continue
            de = intro_exp.as_public_dict(expected)
            k = int(run.get("k", 5))
            min_score = float(run.get("min_score", 0.0))
            exc = run.get("exclude_incident_id")
            exc_s = str(exc) if exc is not None else None

            matches = self.top_k(
                qfp,
                k=k,
                min_score=min_score,
                exclude_incident_id=exc_s,
                two_stage=True,
                rerank_context=rctx,
                debug=False,
            )
            n_ok += 1
            hit = any(m.incident_id == expected for m in matches)
            if hit:
                n_hit += 1
                for kk in keys:
                    sum_hit[kk] += float(de[kk])
            else:
                n_miss += 1
                for kk in keys:
                    sum_miss[kk] += float(de[kk])
                if matches and matches[0].incident_id != expected:
                    intro_top = self._introspection_for_incident_id(
                        qfp, qfeat, rctx, matches[0].incident_id
                    )
                    if intro_top is not None:
                        dt = intro_top.as_public_dict(matches[0].incident_id)
                        for kk in keys:
                            if float(dt[kk]) > float(de[kk]) + eps:
                                fp_dom[kk] += 1

        mean_hit = {k: (sum_hit[k] / n_hit) if n_hit else 0.0 for k in keys}
        mean_miss = {k: (sum_miss[k] / n_miss) if n_miss else 0.0 for k in keys}
        delta = {k: mean_hit[k] - mean_miss[k] for k in keys}

        return {
            "n_runs_total": len(debug_runs),
            "n_runs_with_expected_in_corpus": n_ok,
            "n_top5_hit": n_hit,
            "n_top5_miss": n_miss,
            "signal_mean_on_hit": mean_hit,
            "signal_mean_on_miss": mean_miss,
            "signal_delta_hit_minus_miss": delta,
            "signals_sorted_by_success_correlation": sorted(
                delta.items(), key=lambda kv: kv[1], reverse=True
            ),
            "false_positive_dominance_counts": sorted(
                fp_dom.items(), key=lambda kv: kv[1], reverse=True
            ),
        }

    def index(
        self,
        incident_id: str,
        fingerprint: IncidentFingerprint,
        match_context: Mapping[str, Any] | None = None,
        *,
        retrieval_features: RetrievalFeatures | None = None,
    ) -> None:
        idx = len(self._corpus)
        ctx = dict(match_context) if match_context else {}
        rf = retrieval_features or RetrievalFeatures.from_fingerprint(fingerprint)
        self._corpus.append((incident_id, fingerprint, ctx, rf))
        self._rf_by_id[incident_id] = rf
        bucket = normalize_trigger(fingerprint.trigger_type)
        self._by_trigger[bucket].append(idx)

        self._df_trigger[rf.norm_trigger] += 1
        self._df_role_cluster[_role_cluster(fingerprint)] += 1
        self._df_deploy_shape[_deploy_pattern_shape_key(rf)] += 1
        self._df_prop_depth[int(rf.propagation_fingerprint.propagation_depth)] += 1

        idn = self._extractor.identity
        canon_raw = (rf.canonical_service or fingerprint.canonical_affected or "").strip()
        if canon_raw:
            idn.register(canon_raw)
            ck = idn.resolve(canon_raw)
        else:
            ck = "unknown"
        self._by_canonical[ck].append(idx)

        self._by_role[fingerprint.affected_role].append(idx)
        self._by_role_family[_role_cluster(fingerprint)].append(idx)

        d = rf.propagation_fingerprint.propagation_depth
        self._by_prop_depth[int(d)].append(idx)

        if rf.deploy_pattern:
            self._by_deploy_pattern[rf.deploy_pattern].append(idx)
            dp_key = _deploy_pattern_shape_key(rf)
            self._by_deploy_prefix[dp_key].append(idx)

        self._by_remed_hash[rf.remediation_fp_hash].append(idx)
        self._by_deploy_bucket[_deploy_proximity_bucket(rf.deploy_proximity)].append(idx)

        mem = self._remediation_memory
        if mem is not None:
            try:
                for act in mem.action_keys_for_hash(rf.remediation_fp_hash):
                    self._by_remed_action[str(act)].append(idx)
            except Exception:
                pass

    def index_incident(
        self,
        incident_id: str,
        incident: Mapping[str, Any] | Event,
        *,
        retrieval_features: RetrievalFeatures | None = None,
    ) -> IncidentFingerprint:
        fingerprint = self._extractor.extract(incident)
        ctx: dict[str, Any] = {}
        if isinstance(incident, Mapping):
            ctx = _match_context_from_incident(incident)
        self.index(
            incident_id,
            fingerprint,
            ctx,
            retrieval_features=retrieval_features,
        )
        return fingerprint

    def top_k(
        self,
        query: IncidentFingerprint | Mapping[str, Any] | Event,
        k: int = 5,
        *,
        min_score: float = 0.0,
        exclude_incident_id: str | None = None,
        debug: bool = False,
        rerank_context: RerankContext | None = None,
        two_stage: bool = True,
    ) -> list[MatchResult]:
        if k < 1:
            return []
        query_context: dict[str, Any] | None = None
        try:
            fingerprint = (
                query
                if isinstance(query, IncidentFingerprint)
                else self._extractor.extract(query)
            )
            if isinstance(query, Mapping):
                query_context = _query_context_from_mapping(query)
        except (KeyError, TypeError, ValueError):
            return []
        if not self._corpus:
            return []

        if two_stage and rerank_context is not None:
            return self._top_k_two_stage(
                fingerprint,
                query_context,
                k=k,
                min_score=min_score,
                exclude_incident_id=exclude_incident_id,
                debug=debug,
                rerank_context=rerank_context,
            )

        return self._top_k_legacy(
            fingerprint,
            query_context,
            k=k,
            min_score=min_score,
            exclude_incident_id=exclude_incident_id,
            debug=debug,
        )

    def two_stage_full_ranked_matches(
        self,
        query: IncidentFingerprint | Mapping[str, Any] | Event,
        *,
        min_score: float = 0.0,
        exclude_incident_id: str | None = None,
        rerank_context: RerankContext,
    ) -> list[MatchResult]:
        """Full two-stage rerank list (diverse pool, ``min_score`` filtered) with debug decomposition.

        Intended for offline calibration / failure analysis (``k`` is capped by corpus size).
        """
        n = len(self._corpus)
        if n < 1:
            return []
        return self.top_k(
            query,
            k=n,
            min_score=min_score,
            exclude_incident_id=exclude_incident_id,
            debug=True,
            rerank_context=rerank_context,
            two_stage=True,
        )

    def _union_candidate_sources(
        self,
        fingerprint: IncidentFingerprint,
        qfeat: RetrievalFeatures,
        rerank_context: RerankContext,
        exclude_incident_id: str | None,
    ) -> dict[int, set[str]]:
        src: dict[int, set[str]] = defaultdict(set)
        idn = rerank_context.identity
        n = len(self._corpus)

        def touch(indices: Sequence[int], tag: str) -> None:
            for i in indices:
                if not (0 <= i < n):
                    continue
                if exclude_incident_id and self._corpus[i][0] == exclude_incident_id:
                    continue
                src[i].add(tag)

        qc = (rerank_context.query_canonical or "").strip()
        if idn is not None and qc:
            idn.register(qc)
            root = idn.resolve(qc)
            names: set[str] = {root}
            try:
                names |= set(idn.aliases(qc))
            except Exception:
                pass
            for nm in names:
                touch(self._by_canonical.get(nm, []), "canonical" if nm == root else "alias")
        elif qc:
            touch(self._by_canonical.get(qc, []), "canonical")

        touch(self._by_trigger.get(qfeat.norm_trigger, []), "trigger_family")

        touch(self._by_role.get(fingerprint.affected_role, []), "role")

        qcl = _role_cluster(fingerprint)
        touch(self._by_role_family.get(qcl, []), "role_cluster")

        qdepth = int(qfeat.propagation_fingerprint.propagation_depth)
        for delta in (-1, 0, 1):
            touch(self._by_prop_depth.get(qdepth + delta, []), "propagation_depth")

        touch(self._by_deploy_pattern.get(qfeat.deploy_pattern, []), "deploy_pattern")
        if qfeat.deploy_pattern:
            dp_key = _deploy_pattern_shape_key(qfeat)
            touch(self._by_deploy_prefix.get(dp_key, []), "deploy_pattern_shape")

        touch(self._by_remed_hash.get(qfeat.remediation_fp_hash, []), "remediation_hash")

        mem = rerank_context.remediation_memory
        if mem is not None:
            try:
                for act in mem.action_keys_for_hash(qfeat.remediation_fp_hash):
                    touch(self._by_remed_action.get(str(act), []), "remediation_action")
            except Exception:
                pass

        locale_set = _retrieval_locale(
            identity=idn,
            dependency_graph=rerank_context.dependency_graph,
            neighborhood_memory=rerank_context.neighborhood_memory,
            query_canonical=qc,
        )
        for svc in locale_set:
            touch(self._by_canonical.get(svc, []), "neighbor_or_peer")

        qb = _deploy_proximity_bucket(qfeat.deploy_proximity)
        for b in (qb - 1, qb, qb + 1):
            if 0 <= b <= 10:
                touch(self._by_deploy_bucket.get(b, []), "deploy_window")

        if not src and n > 0:
            for j in range(min(n, _MAX_RAW_CANDIDATE_UNION)):
                if exclude_incident_id and self._corpus[j][0] == exclude_incident_id:
                    continue
                src[j].add("fallback_all")

        if len(src) < _MIN_CANDIDATE_POOL:
            locale_f = frozenset(locale_set) if locale_set else None
            ranked_fill: list[tuple[float, int]] = []
            for j in range(n):
                if j in src or (exclude_incident_id and self._corpus[j][0] == exclude_incident_id):
                    continue
                _i, cfp, _c, cfeat = self._corpus[j]
                s1 = _stage1_recall_score(
                    fingerprint,
                    qfeat,
                    cfp,
                    cfeat,
                    locale=locale_f,
                    identity=idn,
                )
                ranked_fill.append((s1, j))
            need = min(_MIN_CANDIDATE_POOL - len(src), max(0, len(ranked_fill)))
            for _s, j in heapq.nlargest(need, ranked_fill, key=lambda t: t[0]):
                if j not in src:
                    src[j].add("broad_fill")

        if len(src) > _MAX_RAW_CANDIDATE_UNION:
            locale_f = frozenset(locale_set) if locale_set else None
            qfam = _role_cluster(fingerprint)
            scored: list[tuple[float, int]] = []
            for j, _tags in src.items():
                _i, cfp, _c, cfeat = self._corpus[j]
                s1 = _stage1_recall_score(
                    fingerprint,
                    qfeat,
                    cfp,
                    cfeat,
                    locale=locale_f,
                    identity=idn,
                )
                boost = 1_000_000.0 if qfam != "unknown" and _role_cluster(cfp) == qfam else 0.0
                scored.append((s1 + boost, j))
            scored.sort(key=lambda t: t[0], reverse=True)
            keep = {j for _s, j in scored[:_MAX_RAW_CANDIDATE_UNION]}
            src = {j: src[j] for j in keep}

        return src

    def _top_k_two_stage(
        self,
        fingerprint: IncidentFingerprint,
        query_context: dict[str, Any] | None,
        *,
        k: int,
        min_score: float,
        exclude_incident_id: str | None,
        debug: bool,
        rerank_context: RerankContext,
    ) -> list[MatchResult]:
        qfeat = rerank_context.query_features
        idn = rerank_context.identity
        rctx_score = replace(rerank_context, idf_stats=self._idf_stats())
        locale_set = _retrieval_locale(
            identity=idn,
            dependency_graph=rerank_context.dependency_graph,
            neighborhood_memory=rerank_context.neighborhood_memory,
            query_canonical=rerank_context.query_canonical,
        )
        locale_for_stage1 = frozenset(locale_set) if locale_set else None

        raw_sources = self._union_candidate_sources(
            fingerprint, qfeat, rerank_context, exclude_incident_id
        )
        raw_count = len(raw_sources)
        raw_keys = list(raw_sources.keys())
        if not raw_keys:
            self._last_pool_stats = {"raw_union": 0, "diverse_pool": 0}
            self._last_ranking_calib_stats = None
            return []

        def stage1_for_idx(idx: int) -> float:
            _i, cfp, _c, cfeat = self._corpus[idx]
            return _stage1_recall_score(
                fingerprint,
                qfeat,
                cfp,
                cfeat,
                locale=locale_for_stage1,
                identity=idn,
            )

        sorted_raw = sorted(raw_keys, key=stage1_for_idx, reverse=True)
        diverse_indices = _diversity_pick_indices(
            sorted_raw,
            self._corpus,
            identity=idn,
            max_pool=_MAX_CANDIDATE_POOL,
            min_pool=_MIN_CANDIDATE_POOL,
            per_canonical=_MAX_PER_CANONICAL_IN_POOL,
            per_cluster=_MAX_PER_ROLE_CLUSTER_IN_POOL,
        )
        diverse_count = len(diverse_indices)
        self._last_pool_stats = {"raw_union": raw_count, "diverse_pool": diverse_count}

        pool_rank_by_id: dict[str, int] = {}
        for pool_pos, idx in enumerate(diverse_indices, start=1):
            pool_incident_id, _cfp, _cctx, _cfeat = self._corpus[idx]
            pool_rank_by_id[pool_incident_id] = pool_pos

        reranked: list[
            tuple[float, str, IncidentFingerprint, RerankIntrospection, dict[str, float] | None, tuple[str, ...]]
        ] = []
        for idx in diverse_indices:
            incident_id, cand_fp, cand_ctx, cand_feat = self._corpus[idx]
            if exclude_incident_id is not None and incident_id == exclude_incident_id:
                continue
            intro = _rerank_introspection(
                fingerprint, qfeat, cand_fp, cand_feat, rctx_score
            )
            r = intro.total_score
            if r < min_score:
                continue
            parts = _structural_similarity_parts(
                fingerprint,
                cand_fp,
                weights=self._weights,
                query_context=query_context,
                candidate_context=cand_ctx,
            )
            if debug:
                seq_sim = sequence_similarity(qfeat.deploy_pattern, cand_feat.deploy_pattern)
                bd = {
                    **parts,
                    "rerank_score": r,
                    "sequence_similarity": seq_sim,
                    "rerank_decomposition": intro.as_public_dict(incident_id),
                }
            else:
                bd = None
            tags = tuple(sorted(raw_sources.get(idx, set())))
            reranked.append((r, incident_id, cand_fp, intro, bd, tags))

        if not reranked:
            self._last_ranking_calib_stats = None
            rerank_rank_by_id = {}
            ranked_rows_for_debug = []
            ranked_preview = []
            results = []
            topk_ok = _top_k_has_same_family(fingerprint, results)
            _maybe_print_retrieval_recall_debug(
                query_fp=fingerprint,
                query_sig=rerank_context.query_signal,
                expected_family=_role_cluster(fingerprint),
                raw_count=raw_count,
                diverse_count=diverse_count,
                raw_same_family=_pool_has_same_family(raw_keys, fingerprint, self._corpus),
                diverse_same_family=_pool_has_same_family(diverse_indices, fingerprint, self._corpus),
                ranked_preview=ranked_preview,
                topk_ok=topk_ok,
                reranked_rows=ranked_rows_for_debug,
                pool_rank_by_id=pool_rank_by_id,
                rerank_rank_by_id=rerank_rank_by_id,
            )
            return results

        reranked.sort(
            key=lambda row: (
                row[0],
                row[3].upstream_score,
                row[3].remediation_score,
                row[3].deploy_pattern_score,
                row[3].temporal_shape_similarity,
                row[3].negative_evidence_multiplier,
                row[3].rarity_factor,
                row[3].idf_deploy_shape,
                row[3].behavioral_recurrence,
                row[3].recurrence_prior,
                row[3].propagation_score,
                row[3].high_confidence_struct_boost,
                row[3].generic_feature_multiplier,
                row[1],
            ),
            reverse=True,
        )
        scores = [row[0] for row in reranked]
        rarity = [row[3].rarity_factor for row in reranked]
        gen_mult = [row[3].generic_feature_multiplier for row in reranked]
        adjusted, _margin_deltas = _apply_margin_rank_calibration(scores, rarity, gen_mult)
        calibrated: list[
            tuple[float, str, IncidentFingerprint, RerankIntrospection, dict[str, float] | None, tuple[str, ...]]
        ] = []
        for row, adj in zip(reranked, adjusted):
            score, incident_id, fp, intro, bd, tags = row
            clip = min(1.0, adj)
            nintro = replace(intro, total_score=clip, margin_calibration_delta=clip - score)
            if bd is not None:
                bd = {
                    **bd,
                    "rerank_score": clip,
                    "rerank_decomposition": nintro.as_public_dict(incident_id),
                }
            calibrated.append((adj, incident_id, fp, nintro, bd, tags))

        calibrated.sort(
            key=lambda row: (
                row[0],
                row[3].upstream_score,
                row[3].remediation_score,
                row[3].deploy_pattern_score,
                row[3].temporal_shape_similarity,
                row[3].negative_evidence_multiplier,
                row[3].rarity_factor,
                row[3].idf_deploy_shape,
                row[3].behavioral_recurrence,
                row[3].recurrence_prior,
                row[3].propagation_score,
                row[3].high_confidence_struct_boost,
                row[3].generic_feature_multiplier,
                row[1],
            ),
            reverse=True,
        )
        self._last_ranking_calib_stats = (
            _ranking_spread_diagnostics([min(1.0, row[0]) for row in calibrated])
            if calibrated
            else None
        )
        rerank_rank_by_id = {row[1]: rank for rank, row in enumerate(calibrated, start=1)}
        ranked_rows_for_debug = [
            (row[3].total_score, row[1], row[2], row[3], row[5]) for row in calibrated
        ]
        ranked_preview = [(row[3].total_score, row[1]) for row in calibrated[:10]]

        best = heapq.nlargest(
            k,
            calibrated,
            key=lambda row: (
                row[0],
                row[3].upstream_score,
                row[3].remediation_score,
                row[3].deploy_pattern_score,
                row[3].temporal_shape_similarity,
                row[3].negative_evidence_multiplier,
                row[3].rarity_factor,
                row[3].idf_deploy_shape,
                row[3].behavioral_recurrence,
                row[3].recurrence_prior,
                row[3].propagation_score,
                row[3].high_confidence_struct_boost,
                row[3].generic_feature_multiplier,
                row[1],
            ),
        )
        results = [
            MatchResult(
                incident_id=incident_id,
                fingerprint=fp,
                score=nintro.total_score,
                score_breakdown=bd,
                retrieval_sources=tags,
            )
            for _adj, incident_id, fp, nintro, bd, tags in best
        ]

        topk_ok = _top_k_has_same_family(fingerprint, results)
        _maybe_print_retrieval_recall_debug(
            query_fp=fingerprint,
            query_sig=rerank_context.query_signal,
            expected_family=_role_cluster(fingerprint),
            raw_count=raw_count,
            diverse_count=diverse_count,
            raw_same_family=_pool_has_same_family(raw_keys, fingerprint, self._corpus),
            diverse_same_family=_pool_has_same_family(diverse_indices, fingerprint, self._corpus),
            ranked_preview=ranked_preview,
            topk_ok=topk_ok,
            reranked_rows=ranked_rows_for_debug,
            pool_rank_by_id=pool_rank_by_id,
            rerank_rank_by_id=rerank_rank_by_id,
        )
        return results

    def _top_k_legacy(
        self,
        fingerprint: IncidentFingerprint,
        query_context: dict[str, Any] | None,
        *,
        k: int,
        min_score: float,
        exclude_incident_id: str | None,
        debug: bool,
    ) -> list[MatchResult]:
        bucket = normalize_trigger(fingerprint.trigger_type)
        candidate_indices = self._by_trigger.get(bucket, [])
        cand_set = set(candidate_indices)
        if len(candidate_indices) >= len(self._corpus):
            candidate_indices = list(range(len(self._corpus)))
            cand_set = set(candidate_indices)

        scored: list[tuple[float, str, IncidentFingerprint, dict[str, float] | None]] = []

        def score_one(
            incident_id: str,
            candidate: IncidentFingerprint,
            cand_ctx: dict[str, Any],
        ) -> None:
            if exclude_incident_id is not None and incident_id == exclude_incident_id:
                return
            parts = _structural_similarity_parts(
                fingerprint,
                candidate,
                weights=self._weights,
                query_context=query_context,
                candidate_context=cand_ctx,
            )
            s = parts["final"]
            if s >= min_score:
                bd = parts if debug else None
                scored.append((s, incident_id, candidate, bd))

        for idx in candidate_indices:
            incident_id, candidate, cand_ctx, _rf = self._corpus[idx]
            score_one(incident_id, candidate, cand_ctx)

        if len(scored) < k:
            for idx, (incident_id, candidate, cand_ctx, _rf) in enumerate(self._corpus):
                if idx in cand_set:
                    continue
                score_one(incident_id, candidate, cand_ctx)

        best = heapq.nlargest(k, scored, key=lambda row: row[0])
        return [
            MatchResult(
                incident_id=incident_id,
                fingerprint=fp,
                score=score,
                score_breakdown=bd,
                retrieval_sources=(),
            )
            for score, incident_id, fp, bd in best
        ]

    def best_match(
        self,
        query: IncidentFingerprint | Mapping[str, Any] | Event,
        *,
        min_score: float = 0.0,
    ) -> MatchResult | None:
        results = self.top_k(query, k=1, min_score=min_score)
        return results[0] if results else None

    def clear(self) -> None:
        self._corpus.clear()
        self._by_trigger.clear()
        self._by_canonical.clear()
        self._by_role.clear()
        self._by_role_family.clear()
        self._by_prop_depth.clear()
        self._by_deploy_pattern.clear()
        self._by_deploy_prefix.clear()
        self._by_remed_hash.clear()
        self._by_deploy_bucket.clear()
        self._by_remed_action.clear()
        self._rf_by_id.clear()
        self._last_pool_stats = None
        self._last_ranking_calib_stats = None
        self._df_trigger.clear()
        self._df_role_cluster.clear()
        self._df_deploy_shape.clear()
        self._df_prop_depth.clear()


def _sorted_contribution_pairs(
    values: Mapping[str, float],
    *,
    by_abs: bool = True,
) -> list[tuple[str, float]]:
    """Sort ``(name, value)`` for diagnostics (largest magnitude first by default)."""
    if by_abs:
        return sorted(values.items(), key=lambda kv: abs(kv[1]), reverse=True)
    return sorted(values.items(), key=lambda kv: kv[1], reverse=True)


def _two_stage_match_explain_row(
    matcher: FingerprintMatcher,
    qfp: IncidentFingerprint,
    qfeat: RetrievalFeatures,
    rctx: RerankContext,
    cand_id: str,
    cfp: IncidentFingerprint,
    cfeat: RetrievalFeatures,
    *,
    final_score: float,
    retrieval_sources: tuple[str, ...],
    fingerprint_breakdown: Mapping[str, Any] | None,
) -> dict[str, Any]:
    rctx_idf = replace(rctx, idf_stats=matcher._idf_stats())
    intro = _rerank_introspection(qfp, qfeat, cfp, cfeat, rctx_idf)
    comps = rerank_component_values(qfp, qfeat, cfp, cfeat, rctx, idf_stats=rctx_idf.idf_stats)
    wmap = _DEFAULT_RERANK_WEIGHTS if rctx.rerank_weights is None else rctx.rerank_weights

    trigger_c, _prop_upstream_linear = _weighted_trigger_and_prop_upstream(comps, wmap)
    role_c = float(wmap["role"]) * float(comps["role"])
    upstream_c = float(wmap["upstream"]) * float(comps["upstream"])
    prop_edges_c = float(wmap["propagation"]) * float(comps["propagation"])
    temporal_deploy_c = float(wmap["temporal"]) * float(comps["temporal"])
    remed_c = _REMED_RERANK_WEIGHT * float(comps["remed"])

    deploy_seq = _SEQUENCE_RERANK_WEIGHT * intro.deploy_pattern_score
    topo = intro.topology_score
    prop_path = _PROPAGATION_FP_RERANK_WEIGHT * intro.propagation_score
    alias_r = _ALIAS_RERANK_WEIGHT * intro.alias_score
    tempo_shape = _TEMPORAL_SHAPE_RERANK_WEIGHT * intro.temporal_shape_similarity

    propagation_total = prop_edges_c + prop_path
    temporal_total = temporal_deploy_c + tempo_shape

    pre_additive = (
        intro.core_linear
        + deploy_seq
        + topo
        + prop_path
        + alias_r
        + tempo_shape
    )
    scale = (
        intro.rarity_factor
        * intro.negative_evidence_multiplier
        * intro.generic_feature_multiplier
    )
    pre_after_scale = pre_additive * scale + intro.high_confidence_struct_boost

    contrib_for_sort: dict[str, float] = {
        "trigger_core": trigger_c,
        "role_core": role_c,
        "upstream_core": upstream_c,
        "propagation_edges_core": prop_edges_c,
        "propagation_path_rerank": prop_path,
        "temporal_deploy_proximity_core": temporal_deploy_c,
        "temporal_shape_rerank": tempo_shape,
        "remediation_core": remed_c,
        "deploy_sequence_rerank": deploy_seq,
        "topology_rerank": topo,
        "alias_rerank": alias_r,
        "behavioral_recurrence": intro.behavioral_recurrence,
        "recurrence_prior": intro.recurrence_prior,
    }

    return {
        "incident_id": cand_id,
        "final_score": final_score,
        "trigger_contribution": trigger_c,
        "role_contribution": role_c,
        "propagation_contribution": propagation_total,
        "temporal_contribution": temporal_total,
        "topology_contribution": topo,
        "penalties": {
            "negative_evidence_multiplier": intro.negative_evidence_multiplier,
            "rarity_factor": intro.rarity_factor,
            "generic_feature_multiplier": intro.generic_feature_multiplier,
            "high_confidence_struct_boost": intro.high_confidence_struct_boost,
            "margin_calibration_delta": intro.margin_calibration_delta,
            "combined_structural_scale": scale,
            "pre_behavioral_linear_unscaled": pre_additive,
            "pre_behavioral_linear_after_scale": pre_after_scale,
        },
        "retrieval_sources": list(retrieval_sources),
        "contributions_sorted": _sorted_contribution_pairs(contrib_for_sort, by_abs=True),
        "extras": {
            "upstream_contribution": upstream_c,
            "remediation_contribution": remed_c,
            "deploy_sequence_rerank": deploy_seq,
            "alias_rerank": alias_r,
            "propagation_edges_core": prop_edges_c,
            "propagation_path_rerank": prop_path,
            "temporal_deploy_proximity_core": temporal_deploy_c,
            "temporal_shape_rerank": tempo_shape,
            "fingerprint_similarity_final": (
                float(fingerprint_breakdown["final"])
                if fingerprint_breakdown and "final" in fingerprint_breakdown
                else None
            ),
        },
    }


def _legacy_match_explain_row(
    weights: Sequence[float],
    cand_id: str,
    *,
    final_score: float,
    parts: Mapping[str, float],
) -> dict[str, Any]:
    w_trigger, w_role, w_upstream = weights
    total = w_trigger + w_role + w_upstream
    if total <= 0:
        w_trigger, w_role, w_upstream = _DEFAULT_WEIGHTS
        total = sum(_DEFAULT_WEIGHTS)
    w_trigger, w_role, w_upstream = (
        w_trigger / total,
        w_role / total,
        w_upstream / total,
    )
    trigger_c = w_trigger * float(parts["trigger_score"])
    role_c = w_role * float(parts["role_score"])
    upstream_c = w_upstream * float(parts["upstream_score"])
    temporal_boost = float(parts["temporal_score"])
    alias_b = float(parts["alias_score"])
    role_fam = float(parts["role_family_score"])
    base = trigger_c + role_c + upstream_c
    contrib_for_sort = {
        "trigger_core": trigger_c,
        "role_core": role_c,
        "upstream_core": upstream_c,
        "temporal_context_boost": temporal_boost,
        "alias_boost": alias_b,
        "role_family_boost": role_fam,
    }
    return {
        "incident_id": cand_id,
        "final_score": final_score,
        "trigger_contribution": trigger_c,
        "role_contribution": role_c,
        "propagation_contribution": 0.0,
        "temporal_contribution": temporal_boost,
        "topology_contribution": 0.0,
        "penalties": {
            "negative_evidence_multiplier": 1.0,
            "rarity_factor": 1.0,
            "combined_structural_scale": 1.0,
            "note": "legacy single-stage fingerprint score (no rerank pool)",
        },
        "retrieval_sources": [],
        "contributions_sorted": _sorted_contribution_pairs(contrib_for_sort, by_abs=True),
        "extras": {
            "upstream_contribution": upstream_c,
            "remediation_contribution": 0.0,
            "fingerprint_base_linear": base,
            "fingerprint_similarity_final": float(parts["final"]),
        },
    }


def _print_retrieval_explain(rows: Sequence[Mapping[str, Any]], *, stream: TextIO) -> None:
    for i, row in enumerate(rows, start=1):
        print(f"--- match rank {i}: {row['incident_id']} ---", file=stream)
        print(f"  final_score: {row['final_score']:.6f}", file=stream)
        print(f"  trigger_contribution: {row['trigger_contribution']:.6f}", file=stream)
        print(f"  role_contribution: {row['role_contribution']:.6f}", file=stream)
        print(f"  propagation_contribution: {row['propagation_contribution']:.6f}", file=stream)
        print(f"  temporal_contribution: {row['temporal_contribution']:.6f}", file=stream)
        print(f"  topology_contribution: {row['topology_contribution']:.6f}", file=stream)
        pen = row["penalties"]
        if isinstance(pen, dict):
            print("  penalties:", file=stream)
            for pk, pv in pen.items():
                if isinstance(pv, (int, float)):
                    print(f"    {pk}: {float(pv):.6f}", file=stream)
                else:
                    print(f"    {pk}: {pv}", file=stream)
        else:
            print(f"  penalties: {pen!r}", file=stream)
        print(f"  retrieval_sources: {row['retrieval_sources']}", file=stream)
        print("  contributions_sorted (by |value|):", file=stream)
        for name, val in row["contributions_sorted"]:
            print(f"    {name}: {val:.6f}", file=stream)
        print(file=stream)


def retrieval_explain_debug(
    matcher: FingerprintMatcher,
    query: IncidentFingerprint | Mapping[str, Any] | Event,
    *,
    rerank_context: RerankContext | None = None,
    k: int = 5,
    min_score: float = 0.0,
    exclude_incident_id: str | None = None,
    two_stage: bool = True,
    debug: bool = False,
    print_report: bool = True,
    stream: TextIO | None = None,
) -> list[dict[str, Any]] | None:
    """Auditor for ``top_k`` ranking: per-hit contributions, penalties, and retrieval tags.

    When ``debug`` is ``False``, returns ``None`` immediately (no ``top_k``, no corpus scans).
    When ``debug`` is ``True``, runs ``top_k(..., debug=True)`` and builds a structured
    breakdown for each hit (two-stage rerank when ``rerank_context`` is set and
    ``two_stage`` is true; otherwise legacy fingerprint scoring).

    Parameters
    ----------
    print_report
        If ``True`` (default), writes a human-readable report to ``stream`` (stdout).
    stream
        Text stream for printing; defaults to ``sys.stdout``.
    """
    if not debug:
        return None

    import sys

    out = stream or sys.stdout
    use_two = bool(two_stage and rerank_context is not None)
    matches = matcher.top_k(
        query,
        k=k,
        min_score=min_score,
        exclude_incident_id=exclude_incident_id,
        debug=True,
        rerank_context=rerank_context,
        two_stage=use_two,
    )

    qfp: IncidentFingerprint
    try:
        qfp = query if isinstance(query, IncidentFingerprint) else matcher._extractor.extract(query)
    except (KeyError, TypeError, ValueError):
        return []

    qfeat = rerank_context.query_features if rerank_context is not None else None

    rows: list[dict[str, Any]] = []
    for m in matches:
        row_by_id: tuple[str, IncidentFingerprint, dict[str, Any], RetrievalFeatures] | None = None
        for tup in matcher._corpus:
            if tup[0] == m.incident_id:
                row_by_id = tup
                break
        if row_by_id is None:
            continue
        _iid, cfp, _cctx, cfeat = row_by_id
        bd = m.score_breakdown or {}
        if use_two and qfeat is not None and rerank_context is not None:
            rows.append(
                _two_stage_match_explain_row(
                    matcher,
                    qfp,
                    qfeat,
                    rerank_context,
                    m.incident_id,
                    cfp,
                    cfeat,
                    final_score=m.score,
                    retrieval_sources=m.retrieval_sources,
                    fingerprint_breakdown=bd if isinstance(bd, dict) else None,
                )
            )
        elif isinstance(bd, dict) and "trigger_score" in bd:
            rows.append(
                _legacy_match_explain_row(
                    matcher._weights,
                    m.incident_id,
                    final_score=m.score,
                    parts=bd,
                )
            )
        else:
            rows.append(
                {
                    "incident_id": m.incident_id,
                    "final_score": m.score,
                    "trigger_contribution": 0.0,
                    "role_contribution": 0.0,
                    "propagation_contribution": 0.0,
                    "temporal_contribution": 0.0,
                    "topology_contribution": 0.0,
                    "penalties": {"note": "no score_breakdown available"},
                    "retrieval_sources": list(m.retrieval_sources),
                    "contributions_sorted": [],
                    "extras": {},
                }
            )

    if print_report and rows:
        _print_retrieval_explain(rows, stream=out)
    return rows


def _normalize_role_set(
    values: Iterable[Any],
    role_resolver: RoleResolver,
) -> frozenset[str]:
    return frozenset(role_resolver(str(value)) for value in values)
