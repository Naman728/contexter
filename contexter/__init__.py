"""Contexter: topology drift and identity resolution."""

from typing import Final

from contexter.causal_graph import CausalEdge, CausalGraph
from contexter.context_reconstructor import Context, ContextReconstructor, IncidentMatch, Remediation
from contexter.engine import Engine
from contexter.events import Event
from contexter.identity_tracker import IdentityTracker
from contexter.incident_fingerprint import (
    FingerprintExtractor,
    FingerprintMatcher,
    IDFStats,
    IncidentFingerprint,
    MatchResult,
    PropagationFingerprint,
    RerankContext,
    RerankIntrospection,
    RetrievalFeatures,
    TemporalProfile,
    compute_retrieval_features,
    extract_deploy_pattern_sequence,
    extract_propagation_fingerprint,
    extract_temporal_profile,
    fingerprint_remediation_hash,
    fingerprint_remediation_base_key,
    infer_role_family,
    normalize_trigger,
    partial_upstream_overlap,
    propagation_similarity,
    rerank_score_breakdown,
    retrieval_explain_debug,
    sequence_similarity,
    structural_similarity,
    temporal_similarity,
    effective_rerank_weights,
)
from contexter.benchmark_failure_analysis import (
    BenchmarkRunSpec,
    FailureArchetype,
    analyze_benchmark_recall_failures,
    format_rerank_calibration_report,
    print_calibration_report,
    run_benchmark_calibration_report,
)
from contexter.memory_substrate import MemorySubstrate
from contexter.remediation_memory import RemediationMemory, RemedStats

__all__: Final = [
    "CausalEdge",
    "CausalGraph",
    "Context",
    "ContextReconstructor",
    "Engine",
    "Event",
    "FingerprintExtractor",
    "FingerprintMatcher",
    "IdentityTracker",
    "IDFStats",
    "infer_role_family",
    "IncidentFingerprint",
    "IncidentMatch",
    "MatchResult",
    "PropagationFingerprint",
    "RerankIntrospection",
    "normalize_trigger",
    "partial_upstream_overlap",
    "propagation_similarity",
    "MemorySubstrate",
    "Remediation",
    "RemediationMemory",
    "RemedStats",
    "RerankContext",
    "rerank_score_breakdown",
    "retrieval_explain_debug",
    "RetrievalFeatures",
    "TemporalProfile",
    "temporal_similarity",
    "compute_retrieval_features",
    "extract_deploy_pattern_sequence",
    "extract_propagation_fingerprint",
    "extract_temporal_profile",
    "fingerprint_remediation_hash",
    "fingerprint_remediation_base_key",
    "sequence_similarity",
    "structural_similarity",
    "effective_rerank_weights",
    "BenchmarkRunSpec",
    "FailureArchetype",
    "analyze_benchmark_recall_failures",
    "format_rerank_calibration_report",
    "print_calibration_report",
    "run_benchmark_calibration_report",
]
