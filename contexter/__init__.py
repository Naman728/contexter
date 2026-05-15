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
    IncidentFingerprint,
    MatchResult,
    structural_similarity,
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
    "IncidentFingerprint",
    "IncidentMatch",
    "MatchResult",
    "MemorySubstrate",
    "Remediation",
    "RemediationMemory",
    "RemedStats",
    "structural_similarity",
]
