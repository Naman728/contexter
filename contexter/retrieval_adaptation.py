"""Online adaptive retrieval weights and audit logging (no ML dependencies)."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from contexter.incident_fingerprint import (
    IncidentFingerprint,
    MatchResult,
    RetrievalFeatures,
    rerank_component_values,
)

_COMPONENT_KEYS: tuple[str, ...] = (
    "trigger",
    "role",
    "upstream",
    "propagation",
    "temporal",
)
_WEAK_THRESHOLD = 0.38
_SUCCESS_SCORE_FLOOR = 0.22
_EMA_ALPHA = 0.14
_INVERSION_EMA_ALPHA = 0.20
_INVERSION_DELTA_SCALE = 0.09
_PRECISION_EMA_ALPHA = 0.12
_WEIGHT_MIN = 0.07
_WEIGHT_MAX = 0.42
_TARGET_SUM = 0.85
_AUDIT_MAX = 512
_TREND_MAX = 96
_ROLL_WIN_MAX = 48
_MAX_INVERTERS = 8
_SNAPSHOT_VERSION = 1


def _default_weight_map() -> dict[str, float]:
    return {
        "trigger": 0.35,
        "role": 0.08,
        "upstream": 0.22,
        "propagation": 0.12,
        "temporal": 0.08,
    }


def _normalize_clip_weights(raw: Mapping[str, float]) -> dict[str, float]:
    out = {k: float(max(_WEIGHT_MIN, min(_WEIGHT_MAX, raw.get(k, 0.0)))) for k in _COMPONENT_KEYS}
    s = sum(out[k] for k in _COMPONENT_KEYS)
    if s <= 0:
        return _default_weight_map()
    scale = _TARGET_SUM / s
    scaled = {k: out[k] * scale for k in _COMPONENT_KEYS}
    s2 = sum(scaled.values())
    if abs(s2 - _TARGET_SUM) > 1e-6:
        scaled[_COMPONENT_KEYS[0]] += _TARGET_SUM - s2
    return scaled


@dataclass
class RetrievalWeightState:
    """Bounded rerank weights for five adaptive components (remed fixed in rerank)."""

    w_trigger: float = 0.33
    w_role: float = 0.11
    w_upstream: float = 0.20
    w_propagation: float = 0.065
    w_temporal: float = 0.085

    def as_map(self) -> dict[str, float]:
        return _normalize_clip_weights(
            {
                "trigger": self.w_trigger,
                "role": self.w_role,
                "upstream": self.w_upstream,
                "propagation": self.w_propagation,
                "temporal": self.w_temporal,
            }
        )

    @classmethod
    def from_map(cls, m: Mapping[str, float]) -> RetrievalWeightState:
        d = _normalize_clip_weights(m)
        return cls(
            w_trigger=d["trigger"],
            w_role=d["role"],
            w_upstream=d["upstream"],
            w_propagation=d["propagation"],
            w_temporal=d["temporal"],
        )


@dataclass
class RetrievalAdaptation:
    """Audit each reconstruct, analyze failures, tune weights toward successful signals."""

    weights: RetrievalWeightState = field(default_factory=RetrievalWeightState)
    _audit: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=_AUDIT_MAX))
    _recall_trend: deque[float] = field(default_factory=lambda: deque(maxlen=_TREND_MAX))
    _failure_weak: dict[str, int] = field(default_factory=lambda: {k: 0 for k in _COMPONENT_KEYS})
    _contrib_ema: dict[str, float] = field(default_factory=lambda: {k: 0.0 for k in _COMPONENT_KEYS})
    _n_audits: int = 0
    _n_success_updates: int = 0
    _n_inversion_updates: int = 0
    _tp_ema: dict[str, float] = field(default_factory=lambda: {k: 0.5 for k in _COMPONENT_KEYS})
    _fp_ema: dict[str, float] = field(default_factory=lambda: {k: 0.5 for k in _COMPONENT_KEYS})
    _roll_correct_side: dict[str, deque[int]] = field(
        default_factory=lambda: {
            k: deque(maxlen=_ROLL_WIN_MAX) for k in _COMPONENT_KEYS
        }
    )

    def weight_map(self) -> dict[str, float]:
        return self.weights.as_map()

    def record_retrieval(
        self,
        *,
        signal: Mapping[str, Any],
        query_fp: IncidentFingerprint,
        query_features: RetrievalFeatures,
        canonical_service: str,
        matches: Sequence[MatchResult],
        matcher: Any,
        rerank_ctx: Any,
    ) -> None:
        """Log retrieval outcome and nudge weights on success; tally weak dims on failure."""
        self._n_audits += 1
        match_list = list(matches)
        expected = signal.get("_retrieval_expected_incident_id")
        expected_str = str(expected).strip() if expected else ""

        family_hit = _family_hit(query_fp, match_list)
        success = False
        if expected_str:
            success = any(
                m.incident_id == expected_str and m.score >= _SUCCESS_SCORE_FLOOR
                for m in match_list
            )
        else:
            success = family_hit

        self._recall_trend.append(1.0 if success else 0.0)

        max_by_dim: dict[str, float] = {k: 0.0 for k in _COMPONENT_KEYS}
        for m in match_list:
            rf = _features_for_incident(matcher, m.incident_id)
            if rf is None:
                continue
            idf = _matcher_idf_stats(matcher)
            vals = rerank_component_values(
                query_fp, query_features, m.fingerprint, rf, rerank_ctx, idf_stats=idf
            )
            for k in _COMPONENT_KEYS:
                max_by_dim[k] = max(max_by_dim[k], vals[k])

        weak: dict[str, bool] | None = None
        if not success:
            weak = {k: max_by_dim[k] < _WEAK_THRESHOLD for k in _COMPONENT_KEYS}
            for k, is_weak in weak.items():
                if is_weak:
                    self._failure_weak[k] += 1

        audit_entry: dict[str, Any] = {
            "occurred_at": datetime.now(timezone.utc).isoformat(),
            "query_incident_id": str(signal.get("incident_id", "")),
            "canonical_service": canonical_service,
            "trigger_type": query_fp.trigger_type,
            "affected_role": query_fp.affected_role,
            "matches": [
                {
                    "past_incident_id": m.incident_id,
                    "similarity": round(float(m.score), 6),
                    "same_family": _same_family(query_fp, m.fingerprint),
                }
                for m in match_list
            ],
            "family_hit": family_hit,
            "success": success,
            "weak_components": weak,
        }
        if expected_str:
            audit_entry["expected_incident_id"] = expected_str
        self._audit.append(audit_entry)

        wmap = self.weight_map()
        for k in _COMPONENT_KEYS:
            c = wmap[k] * max_by_dim[k]
            prev = self._contrib_ema[k]
            self._contrib_ema[k] = (1.0 - _EMA_ALPHA) * prev + _EMA_ALPHA * c

        cur = self.weight_map()
        if expected_str:
            cur = _apply_inversion_learning(
                self,
                cur,
                expected_str=expected_str,
                query_fp=query_fp,
                query_features=query_features,
                match_list=match_list,
                matcher=matcher,
                rerank_ctx=rerank_ctx,
            )

        if success:
            winner = _pick_success_match(match_list, expected_str, query_fp)
            if winner is not None:
                rf_win = _features_for_incident(matcher, winner.incident_id)
                if rf_win is not None:
                    idf = _matcher_idf_stats(matcher)
                    vals = rerank_component_values(
                        query_fp,
                        query_features,
                        winner.fingerprint,
                        rf_win,
                        rerank_ctx,
                        idf_stats=idf,
                    )
                    ssum = sum(max(vals[k], 1e-6) for k in _COMPONENT_KEYS)
                    target = {
                        k: _TARGET_SUM * max(vals[k], 1e-6) / ssum for k in _COMPONENT_KEYS
                    }
                    new_raw = {
                        k: (1.0 - _EMA_ALPHA) * cur[k] + _EMA_ALPHA * target[k]
                        for k in _COMPONENT_KEYS
                    }
                    self.weights = RetrievalWeightState.from_map(new_raw)
                    self._n_success_updates += 1
                    return

        self.weights = RetrievalWeightState.from_map(cur)

    def recent_audits(self, n: int = 50) -> list[dict[str, Any]]:
        """Last ``n`` audit records (most recent last)."""
        if n <= 0:
            return []
        return list(self._audit)[-n:]

    def stats(self) -> dict[str, Any]:
        """Aggregates for observability and tuning health."""
        rt = list(self._recall_trend)
        recall_mean = sum(rt) / len(rt) if rt else 0.0
        w = self.weight_map()
        prec = {
            k: round(
                self._tp_ema[k] / max(1e-9, self._tp_ema[k] + self._fp_ema[k]),
                5,
            )
            for k in _COMPONENT_KEYS
        }
        roll_mean = {}
        for k in _COMPONENT_KEYS:
            dq = self._roll_correct_side[k]
            roll_mean[k] = round(sum(dq) / len(dq), 5) if dq else 0.0
        return {
            "n_audits": self._n_audits,
            "n_success_weight_updates": self._n_success_updates,
            "n_inversion_weight_updates": self._n_inversion_updates,
            "average_weighted_trigger_contribution": self._contrib_ema["trigger"],
            "average_weighted_upstream_contribution": self._contrib_ema["upstream"],
            "average_weighted_role_contribution": self._contrib_ema["role"],
            "average_weighted_propagation_contribution": self._contrib_ema["propagation"],
            "average_weighted_temporal_contribution": self._contrib_ema["temporal"],
            "recall_trend_mean": round(recall_mean, 5),
            "recall_trend_window": len(rt),
            "current_weights": {k: round(w[k], 5) for k in _COMPONENT_KEYS},
            "failure_mode_counts": dict(self._failure_weak),
            "component_precision_ema": prec,
            "component_tp_ema": {k: round(self._tp_ema[k], 5) for k in _COMPONENT_KEYS},
            "component_fp_ema": {k: round(self._fp_ema[k], 5) for k in _COMPONENT_KEYS},
            "rolling_correct_side_rate": roll_mean,
        }

    def snapshot(self) -> dict[str, Any]:
        """Bounded, JSON-friendly state for persistence (deterministic restore)."""
        return {
            "v": _SNAPSHOT_VERSION,
            "weights": self.weights.as_map(),
            "contrib_ema": {k: float(self._contrib_ema[k]) for k in _COMPONENT_KEYS},
            "failure_weak": {k: int(self._failure_weak[k]) for k in _COMPONENT_KEYS},
            "tp_ema": {k: float(self._tp_ema[k]) for k in _COMPONENT_KEYS},
            "fp_ema": {k: float(self._fp_ema[k]) for k in _COMPONENT_KEYS},
            "n_audits": self._n_audits,
            "n_success_updates": self._n_success_updates,
            "n_inversion_updates": self._n_inversion_updates,
            "recall_trend": [float(x) for x in self._recall_trend],
            "roll_correct_side": {
                k: [int(x) for x in self._roll_correct_side[k]] for k in _COMPONENT_KEYS
            },
        }

    def restore_snapshot(self, data: Mapping[str, Any]) -> None:
        """Restore from :meth:`snapshot` (same process / cold start)."""
        ver = int(data.get("v", 0))
        if ver != _SNAPSHOT_VERSION:
            return
        w = data.get("weights")
        if isinstance(w, Mapping):
            self.weights = RetrievalWeightState.from_map(w)
        ce = data.get("contrib_ema")
        if isinstance(ce, Mapping):
            for k in _COMPONENT_KEYS:
                if k in ce:
                    self._contrib_ema[k] = float(ce[k])
        fw = data.get("failure_weak")
        if isinstance(fw, Mapping):
            for k in _COMPONENT_KEYS:
                if k in fw:
                    self._failure_weak[k] = int(fw[k])
        tp = data.get("tp_ema")
        if isinstance(tp, Mapping):
            for k in _COMPONENT_KEYS:
                if k in tp:
                    self._tp_ema[k] = float(tp[k])
        fp = data.get("fp_ema")
        if isinstance(fp, Mapping):
            for k in _COMPONENT_KEYS:
                if k in fp:
                    self._fp_ema[k] = float(fp[k])
        self._n_audits = int(data.get("n_audits", self._n_audits))
        self._n_success_updates = int(data.get("n_success_updates", self._n_success_updates))
        self._n_inversion_updates = int(
            data.get("n_inversion_updates", self._n_inversion_updates)
        )
        rt = data.get("recall_trend")
        if isinstance(rt, Sequence) and not isinstance(rt, (str, bytes)):
            self._recall_trend.clear()
            for x in rt[-_TREND_MAX:]:
                try:
                    self._recall_trend.append(float(x))
                except (TypeError, ValueError):
                    continue
        rc = data.get("roll_correct_side")
        if isinstance(rc, Mapping):
            for k in _COMPONENT_KEYS:
                seq = rc.get(k)
                if not isinstance(seq, Sequence) or isinstance(seq, (str, bytes)):
                    continue
                dq = self._roll_correct_side[k]
                dq.clear()
                for bit in seq[-_ROLL_WIN_MAX:]:
                    try:
                        b = int(bit)
                        if b in (0, 1):
                            dq.append(b)
                    except (TypeError, ValueError):
                        continue


def _matcher_idf_stats(matcher: Any):
    try:
        fn = getattr(matcher, "_idf_stats", None)
        if callable(fn):
            return fn()
    except Exception:
        return None
    return None


def _vals_for_match(
    matcher: Any,
    query_fp: IncidentFingerprint,
    query_features: RetrievalFeatures,
    m: MatchResult,
    rerank_ctx: Any,
) -> dict[str, float] | None:
    rf = _features_for_incident(matcher, m.incident_id)
    if rf is None:
        return None
    return rerank_component_values(
        query_fp,
        query_features,
        m.fingerprint,
        rf,
        rerank_ctx,
        idf_stats=_matcher_idf_stats(matcher),
    )


def _apply_inversion_learning(
    adapt: RetrievalAdaptation,
    cur: dict[str, float],
    *,
    expected_str: str,
    query_fp: IncidentFingerprint,
    query_features: RetrievalFeatures,
    match_list: list[MatchResult],
    matcher: Any,
    rerank_ctx: Any,
) -> dict[str, float]:
    """When a labeled correct incident ranks below an incorrect one, nudge weights."""
    exp = expected_str.strip()
    if not exp:
        return cur
    idx_ok = -1
    for i, m in enumerate(match_list):
        if m.incident_id == exp:
            idx_ok = i
            break
    if idx_ok <= 0:
        return cur

    v_ok = _vals_for_match(matcher, query_fp, query_features, match_list[idx_ok], rerank_ctx)
    if v_ok is None:
        return cur

    mean_dv: dict[str, float] = {k: 0.0 for k in _COMPONENT_KEYS}
    n_used = 0
    start = max(0, idx_ok - _MAX_INVERTERS)
    for j in range(start, idx_ok):
        mw = match_list[j]
        if mw.incident_id == exp:
            continue
        vw = _vals_for_match(matcher, query_fp, query_features, mw, rerank_ctx)
        if vw is None:
            continue
        for k in _COMPONENT_KEYS:
            mean_dv[k] += float(v_ok[k]) - float(vw[k])
        n_used += 1
    if n_used == 0:
        return cur
    for k in _COMPONENT_KEYS:
        mean_dv[k] /= float(n_used)

    a = _PRECISION_EMA_ALPHA
    for k in _COMPONENT_KEYS:
        vc = float(v_ok[k])
        v_mean_wrong = vc - mean_dv[k]
        gp = 1.0 if vc > v_mean_wrong + 1e-12 else 0.0
        fp = 1.0 if v_mean_wrong > vc + 1e-12 else 0.0
        adapt._tp_ema[k] = (1.0 - a) * adapt._tp_ema[k] + a * gp
        adapt._fp_ema[k] = (1.0 - a) * adapt._fp_ema[k] + a * fp
        adapt._roll_correct_side[k].append(1 if gp >= 1.0 else 0)

    if all(abs(mean_dv[k]) < 1e-12 for k in _COMPONENT_KEYS):
        return cur

    raw = {k: cur[k] + _INVERSION_DELTA_SCALE * mean_dv[k] for k in _COMPONENT_KEYS}
    clipped = _normalize_clip_weights(raw)
    blended = {
        k: (1.0 - _INVERSION_EMA_ALPHA) * cur[k] + _INVERSION_EMA_ALPHA * clipped[k]
        for k in _COMPONENT_KEYS
    }
    adapt._n_inversion_updates += 1
    return blended


def _features_for_incident(matcher: Any, incident_id: str) -> RetrievalFeatures | None:
    try:
        fn = getattr(matcher, "retrieval_features_for_incident", None)
        if callable(fn):
            return fn(incident_id)
    except Exception:
        return None
    return None


def _same_family(q: IncidentFingerprint, c: IncidentFingerprint) -> bool:
    from contexter.incident_fingerprint import infer_role_family

    qf = infer_role_family(q.canonical_affected or q.affected_role)
    cf = infer_role_family(c.canonical_affected or c.affected_role)
    return qf != "unknown" and qf == cf


def _family_hit(query_fp: IncidentFingerprint, matches: Sequence[MatchResult]) -> bool:
    return any(_same_family(query_fp, m.fingerprint) for m in matches)


def _pick_success_match(
    matches: Sequence[MatchResult],
    expected_id: str,
    query_fp: IncidentFingerprint,
) -> MatchResult | None:
    if expected_id:
        for m in matches:
            if m.incident_id == expected_id and m.score >= _SUCCESS_SCORE_FLOOR:
                return m
        return None
    qf = _query_family(query_fp)
    if qf == "unknown":
        return None
    best: MatchResult | None = None
    for m in matches:
        if _query_family(m.fingerprint) != qf:
            continue
        if best is None or m.score > best.score:
            best = m
    return best


def _query_family(fp: IncidentFingerprint) -> str:
    from contexter.incident_fingerprint import infer_role_family

    return infer_role_family(fp.canonical_affected or fp.affected_role)
