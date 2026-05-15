"""Empirical rerank failure analysis for benchmark-style runs (no new retrieval features).

Builds evidence tables from two-stage rerank introspection: per-failure breakdowns,
aggregate FP-vs-TP deltas, margin distributions, and coarse failure archetypes.
"""

from __future__ import annotations

import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from statistics import mean
from typing import Any, TextIO

from contexter.incident_fingerprint import (
    FingerprintMatcher,
    IncidentFingerprint,
    MatchResult,
    RerankContext,
    RetrievalFeatures,
    effective_rerank_weights,
)


class FailureArchetype(str, Enum):
    DEPLOY_NOISE = "deploy_noise_failure"
    TOPOLOGY_MISMATCH = "topology_mismatch"
    TEMPORAL_MISMATCH = "temporal_mismatch"
    PROPAGATION_MISMATCH = "propagation_mismatch"
    GENERIC_TRIGGER_COLLISION = "generic_trigger_collision"
    SHALLOW_DEEP_CONFUSION = "shallow_deep_confusion"
    RENAME_DRIFT_CONFUSION = "rename_drift_confusion"


def _normalize_run(raw: Mapping[str, Any] | BenchmarkRunSpec) -> Mapping[str, Any]:
    if isinstance(raw, BenchmarkRunSpec):
        return {
            "query": raw.query,
            "rerank_context": raw.rerank_context,
            "expected_incident_id": raw.expected_incident_id,
            "k": raw.k,
            "min_score": raw.min_score,
            "exclude_incident_id": raw.exclude_incident_id,
            "label": raw.label,
        }
    return raw


_INTRO_COMPONENT_KEYS: tuple[str, ...] = (
    "trigger_score",
    "role_score",
    "propagation_score",
    "topology_score",
    "upstream_score",
    "temporal_score",
    "temporal_shape_similarity",
    "deploy_pattern_score",
    "remediation_score",
    "rarity_factor",
    "negative_evidence_multiplier",
    "generic_feature_multiplier",
    "high_confidence_struct_boost",
    "margin_calibration_delta",
    "behavioral_recurrence",
    "recurrence_prior",
)


def _decomp(m: MatchResult) -> dict[str, Any] | None:
    bd = m.score_breakdown
    if not isinstance(bd, Mapping):
        return None
    rd = bd.get("rerank_decomposition")
    return rd if isinstance(rd, Mapping) else None


def _intro_scalar_vec(d: Mapping[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for k in _INTRO_COMPONENT_KEYS:
        try:
            out[k] = float(d.get(k, 0.0))
        except (TypeError, ValueError):
            out[k] = 0.0
    out["total_score"] = float(d.get("total_score", 0.0))
    out["penalties_product"] = float(d.get("negative_evidence_multiplier", 1.0)) * float(
        d.get("generic_feature_multiplier", 1.0)
    )
    out["temporal_combined"] = 0.5 * out["temporal_score"] + 0.5 * out["temporal_shape_similarity"]
    return out


def _classify_archetypes(
    qfeat: RetrievalFeatures,
    qfp: IncidentFingerprint,
    intro_true: Mapping[str, Any],
    intro_fp: Mapping[str, Any],
    cfeat_true: RetrievalFeatures,
    cfeat_fp: RetrievalFeatures,
    cand_fp_top: IncidentFingerprint,
) -> list[FailureArchetype]:
    vt = _intro_scalar_vec(intro_true)
    vf = _intro_scalar_vec(intro_fp)
    out: list[FailureArchetype] = []
    eps = 0.08

    if vf["deploy_pattern_score"] > vt["deploy_pattern_score"] + 0.12:
        out.append(FailureArchetype.DEPLOY_NOISE)
    if vf["topology_score"] > vt["topology_score"] + 0.03 and vt["topology_score"] < 0.07:
        out.append(FailureArchetype.TOPOLOGY_MISMATCH)
    if vf["temporal_shape_similarity"] > vt["temporal_shape_similarity"] + 0.15:
        out.append(FailureArchetype.TEMPORAL_MISMATCH)
    if vf["propagation_score"] > vt["propagation_score"] + 0.12:
        out.append(FailureArchetype.PROPAGATION_MISMATCH)

    idf_t = min(float(intro_true.get("idf_trigger", 0.0)), float(intro_fp.get("idf_trigger", 0.0)))
    if qfeat.norm_trigger == cfeat_fp.norm_trigger == cfeat_true.norm_trigger and idf_t < 0.35:
        out.append(FailureArchetype.GENERIC_TRIGGER_COLLISION)

    qd = int(qfeat.propagation_fingerprint.propagation_depth)
    td = int(cfeat_true.propagation_fingerprint.propagation_depth)
    fd = int(cfeat_fp.propagation_fingerprint.propagation_depth)
    if abs(qd - fd) < abs(qd - td) and abs(qd - td) >= 2:
        out.append(FailureArchetype.SHALLOW_DEEP_CONFUSION)

    if vf["alias_score"] > vt["alias_score"] + 0.2 and (
        (cand_fp_top.canonical_affected or "") != (qfp.canonical_affected or "")
    ):
        out.append(FailureArchetype.RENAME_DRIFT_CONFUSION)

    if not out:
        if vf["total_score"] > vt["total_score"] + eps:
            out.append(FailureArchetype.PROPAGATION_MISMATCH)
        else:
            out.append(FailureArchetype.GENERIC_TRIGGER_COLLISION)
    dedup: list[FailureArchetype] = []
    seen: set[str] = set()
    for a in out:
        if a.value not in seen:
            seen.add(a.value)
            dedup.append(a)
    return dedup


@dataclass(slots=True)
class BenchmarkRunSpec:
    """One labeled query (same keys as ``FingerprintMatcher.calibrate_weights``)."""

    query: IncidentFingerprint | Mapping[str, Any]
    rerank_context: RerankContext
    expected_incident_id: str
    k: int = 5
    min_score: float = 0.0
    exclude_incident_id: str | None = None
    label: str = ""


def _extract_query_fp(matcher: FingerprintMatcher, q: Any) -> IncidentFingerprint | None:
    try:
        return q if isinstance(q, IncidentFingerprint) else matcher._extractor.extract(q)
    except (KeyError, TypeError, ValueError):
        return None


def analyze_benchmark_recall_failures(
    matcher: FingerprintMatcher,
    runs: Sequence[Mapping[str, Any] | BenchmarkRunSpec],
    *,
    k: int = 5,
    top_fp_compare: int = 3,
) -> dict[str, Any]:
    """Analyze recall@k failures: ranks, FP-vs-TP introspection deltas, archetypes, aggregates.

    ``runs`` entries are mappings with ``query``, ``rerank_context`` (:class:`RerankContext`),
    ``expected_incident_id``, optional ``k``, ``min_score``, ``exclude_incident_id``, ``label``.
    """
    failures: list[dict[str, Any]] = []
    margins: list[float] = []
    fp_excess_sums: dict[str, float] = {c: 0.0 for c in _INTRO_COMPONENT_KEYS}
    fp_excess_counts: dict[str, int] = {c: 0 for c in _INTRO_COMPONENT_KEYS}
    true_deficit_sums: dict[str, float] = {c: 0.0 for c in _INTRO_COMPONENT_KEYS}
    true_lower_counts: dict[str, int] = {c: 0 for c in _INTRO_COMPONENT_KEYS}
    n_fail = 0
    arche_hist: Counter[str] = Counter()
    not_in_corpus = 0
    not_in_rerank_pool = 0
    min_score_gate_misses = 0

    for raw in runs:
        run = _normalize_run(raw)
        rctx = run.get("rerank_context")
        if not isinstance(rctx, RerankContext):
            continue
        expected = str(run.get("expected_incident_id", "") or "").strip()
        if not expected:
            continue
        q = run.get("query")
        qfp = _extract_query_fp(matcher, q)
        if qfp is None:
            continue
        rk = int(run.get("k", k))
        min_score = float(run.get("min_score", 0.0))
        exc = run.get("exclude_incident_id")
        exc_s = str(exc) if exc is not None else None
        label = str(run.get("label", "") or "")

        topk = matcher.top_k(
            qfp,
            k=rk,
            min_score=min_score,
            exclude_incident_id=exc_s,
            two_stage=True,
            rerank_context=rctx,
            debug=True,
        )
        topk_at_zero = matcher.top_k(
            qfp,
            k=rk,
            min_score=0.0,
            exclude_incident_id=exc_s,
            two_stage=True,
            rerank_context=rctx,
            debug=True,
        )
        hit = any(m.incident_id == expected for m in topk)
        if hit:
            continue

        hit_at_k_min0 = any(m.incident_id == expected for m in topk_at_zero)
        if min_score > 0.0 and hit_at_k_min0 and not any(m.incident_id == expected for m in topk):
            min_score_gate_misses += 1

        n_fail += 1
        full = matcher.two_stage_full_ranked_matches(
            qfp,
            min_score=0.0,
            exclude_incident_id=exc_s,
            rerank_context=rctx,
        )
        rank: int | None = None
        true_row: MatchResult | None = None
        for i, m in enumerate(full, start=1):
            if m.incident_id == expected:
                rank = i
                true_row = m
                break

        intro_exp = matcher._introspection_for_incident_id(qfp, rctx.query_features, rctx, expected)
        cfeat_exp = matcher.retrieval_features_for_incident(expected)

        if intro_exp is None:
            not_in_corpus += 1
        elif rank is None:
            not_in_rerank_pool += 1

        top_fps: list[MatchResult] = [m for m in topk if m.incident_id != expected][:top_fp_compare]
        if not top_fps and full:
            top_fps = [m for m in full if m.incident_id != expected][:top_fp_compare]

        fp_primary = top_fps[0] if top_fps else None
        d_true = None
        if true_row is not None:
            d_true = _decomp(true_row)
        elif intro_exp is not None:
            d_true = intro_exp.as_public_dict(expected)

        d_fp = _decomp(fp_primary) if fp_primary is not None else None

        margin = None
        if d_true is not None:
            ref = topk[0] if topk else (full[0] if full else None)
            if ref is not None:
                top1d = _decomp(ref)
                if top1d is not None:
                    margin = float(top1d.get("total_score", 0.0)) - float(d_true.get("total_score", 0.0))
                    margins.append(margin)

        if d_true is not None and d_fp is not None:
            vt = _intro_scalar_vec(d_true)
            vf = _intro_scalar_vec(d_fp)
            for c in _INTRO_COMPONENT_KEYS:
                diff = vf[c] - vt[c]
                if diff > 0.02:
                    fp_excess_sums[c] += diff
                    fp_excess_counts[c] += 1
                if diff < -0.02:
                    true_deficit_sums[c] += -diff
                    true_lower_counts[c] += 1

        archs: list[FailureArchetype] = []
        if (
            d_true is not None
            and d_fp is not None
            and cfeat_exp is not None
            and fp_primary is not None
        ):
            cfeat_fp = matcher.retrieval_features_for_incident(fp_primary.incident_id)
            if cfeat_fp is not None:
                archs = _classify_archetypes(
                    rctx.query_features,
                    qfp,
                    d_true,
                    d_fp,
                    cfeat_exp,
                    cfeat_fp,
                    fp_primary.fingerprint,
                )
        for a in archs:
            arche_hist[a.value] += 1

        failures.append(
            {
                "label": label,
                "expected_incident_id": expected,
                "recall_at_k": False,
                "k": rk,
                "min_score": min_score,
                "rank_in_full_pool": rank,
                "not_in_corpus": intro_exp is None,
                "not_in_rerank_pool": intro_exp is not None and rank is None,
                "min_score_gate_miss": min_score > 0.0 and hit_at_k_min0,
                "not_in_union_or_corpus": intro_exp is None,
                "below_min_score_only": intro_exp is not None and rank is None,
                "topk_ids": [m.incident_id for m in topk],
                "margin_top1_minus_true": margin,
                "archetypes": [a.value for a in archs],
                "true_decomposition": dict(d_true) if isinstance(d_true, Mapping) else None,
                "top_false_positive_decompositions": [
                    {"incident_id": m.incident_id, "decomposition": _decomp(m)} for m in top_fps
                ],
                "adaptive_weights": effective_rerank_weights(rctx),
            }
        )

    fp_over_score_ranking = sorted(
        (
            (
                c,
                fp_excess_counts[c],
                fp_excess_sums[c] / max(1, fp_excess_counts[c]),
            )
            for c in _INTRO_COMPONENT_KEYS
        ),
        key=lambda t: t[1],
        reverse=True,
    )
    true_under_score_ranking = sorted(
        (
            (
                c,
                true_lower_counts[c],
                true_deficit_sums[c] / max(1, true_lower_counts[c]),
            )
            for c in _INTRO_COMPONENT_KEYS
        ),
        key=lambda t: t[1],
        reverse=True,
    )

    return {
        "n_runs_failures": n_fail,
        "failures_not_in_corpus": not_in_corpus,
        "failures_not_in_rerank_pool": not_in_rerank_pool,
        "failures_min_score_gate": min_score_gate_misses,
        "failures_not_in_pool": not_in_corpus,
        "failures_below_min_score_only": not_in_rerank_pool,
        "margins_top1_minus_true": margins,
        "margin_mean": mean(margins) if margins else 0.0,
        "margin_p50_proxy": sorted(margins)[len(margins) // 2] if margins else 0.0,
        "components_fp_excess_count_when_higher": {
            c: fp_excess_counts[c] for c in _INTRO_COMPONENT_KEYS
        },
        "components_mean_fp_minus_true_when_fp_higher": {
            c: fp_excess_sums[c] / max(1, fp_excess_counts[c]) for c in _INTRO_COMPONENT_KEYS
        },
        "components_mean_true_deficit_when_true_lower": {
            c: true_deficit_sums[c] / max(1, true_lower_counts[c]) for c in _INTRO_COMPONENT_KEYS
        },
        "components_sorted_fp_over_score": fp_over_score_ranking,
        "components_sorted_true_under_score": true_under_score_ranking,
        "failure_archetype_counts": dict(arche_hist.most_common()),
        "failure_archetype_top": arche_hist.most_common(12),
        "failures": failures,
    }


def format_rerank_calibration_report(report: Mapping[str, Any], *, title: str = "Rerank calibration report") -> str:
    lines = [title, "=" * len(title), ""]
    lines.append(f"Failures (recall@k miss): {report.get('n_runs_failures', 0)}")
    lines.append(f"  not in corpus: {report.get('failures_not_in_corpus', report.get('failures_not_in_pool', 0))}")
    lines.append(
        f"  not in rerank pool (union+diversity): {report.get('failures_not_in_rerank_pool', report.get('failures_below_min_score_only', 0))}"
    )
    lines.append(f"  min_score gate only: {report.get('failures_min_score_gate', 0)}")
    lines.append("")
    lines.append(f"Margin (top1_total - true_total) mean: {report.get('margin_mean', 0.0):.4f}")
    lines.append("")
    lines.append("Components most often higher on false positive than true (count, mean excess):")
    for row in report.get("components_sorted_fp_over_score", [])[:10]:
        if isinstance(row, (list, tuple)) and len(row) >= 3:
            c, cnt, avg = row[0], row[1], row[2]
            if cnt:
                lines.append(f"  {c}: n={cnt}  mean_fp_minus_true={avg:.4f}")
    lines.append("")
    lines.append("Largest mean true deficit (when true < FP) — (count, mean deficit):")
    for row in report.get("components_sorted_true_under_score", [])[:10]:
        if isinstance(row, (list, tuple)) and len(row) >= 3:
            c, cnt, avg = row[0], row[1], row[2]
            if cnt:
                lines.append(f"  {c}: n={cnt}  mean_deficit={avg:.4f}")
    lines.append("")
    lines.append("Failure archetypes:")
    for name, cnt in report.get("failure_archetype_top", []):
        lines.append(f"  {name}: {cnt}")
    lines.append("")
    for i, f in enumerate(report.get("failures", []), start=1):
        lines.append("-" * 72)
        lines.append(f"Failure #{i}  expected={f.get('expected_incident_id')!r}  label={f.get('label')!r}")
        lines.append(
            f"  rank_in_full_pool(min_score=0)={f.get('rank_in_full_pool')}  "
            f"not_in_corpus={f.get('not_in_corpus', f.get('not_in_union_or_corpus'))}  "
            f"not_in_rerank_pool={f.get('not_in_rerank_pool', f.get('below_min_score_only'))}  "
            f"min_score_gate={f.get('min_score_gate_miss')}"
        )
        lines.append(f"  top-{f.get('k')} ids: {f.get('topk_ids')}")
        lines.append(f"  margin_top1_minus_true: {f.get('margin_top1_minus_true')}")
        lines.append(f"  archetypes: {f.get('archetypes')}")
        w = f.get("adaptive_weights")
        if isinstance(w, Mapping):
            lines.append(f"  adaptive_weights: {dict(w)}")
        dt = f.get("true_decomposition")
        if isinstance(dt, Mapping):
            lines.append("  true breakdown (rerank_decomposition):")
            lines.append(_format_decomposition_block(dt, indent="    "))
        for block in f.get("top_false_positive_decompositions") or []:
            if not isinstance(block, Mapping):
                continue
            iid = block.get("incident_id")
            dec = block.get("decomposition")
            lines.append(f"  false positive {iid}:")
            if isinstance(dec, Mapping):
                lines.append(_format_decomposition_block(dec, indent="    "))
        lines.append("")
    return "\n".join(lines) + "\n"


def _format_decomposition_block(d: Mapping[str, Any], *, indent: str = "  ") -> str:
    keys = (
        "total_score",
        "trigger_score",
        "role_score",
        "propagation_score",
        "topology_score",
        "upstream_score",
        "temporal_score",
        "temporal_shape_similarity",
        "deploy_pattern_score",
        "remediation_score",
        "rarity_factor",
        "negative_evidence_multiplier",
        "generic_feature_multiplier",
        "high_confidence_struct_boost",
        "margin_calibration_delta",
        "behavioral_recurrence",
        "recurrence_prior",
        "idf_trigger",
        "idf_role_cluster",
        "idf_deploy_shape",
        "idf_propagation_depth",
        "core_linear",
    )
    parts = []
    for key in keys:
        if key in d:
            try:
                parts.append(f"{indent}{key}={float(d[key]):.5f}")
            except (TypeError, ValueError):
                parts.append(f"{indent}{key}={d[key]!r}")
    return "\n".join(parts)


def print_calibration_report(
    report: Mapping[str, Any],
    *,
    title: str = "Rerank calibration report",
    stream: TextIO | None = None,
) -> None:
    text = format_rerank_calibration_report(report, title=title)
    (stream or sys.stdout).write(text)


def run_benchmark_calibration_report(
    matcher: FingerprintMatcher,
    runs: Sequence[Mapping[str, Any] | BenchmarkRunSpec],
    *,
    k: int = 5,
    title: str = "Rerank calibration report",
    stream: TextIO | None = None,
    return_dict: bool = True,
) -> dict[str, Any] | None:
    """Run failure analysis and print a calibration report (for use after benchmark harness)."""
    rep = analyze_benchmark_recall_failures(matcher, runs, k=k)
    print_calibration_report(rep, title=title, stream=stream)
    return rep if return_dict else None


__all__ = (
    "FailureArchetype",
    "BenchmarkRunSpec",
    "analyze_benchmark_recall_failures",
    "format_rerank_calibration_report",
    "print_calibration_report",
    "run_benchmark_calibration_report",
)
