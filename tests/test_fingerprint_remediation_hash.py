"""Extended remediation / retrieval fingerprint hashes."""

from __future__ import annotations

from dataclasses import replace

from contexter.incident_fingerprint import (
    IncidentFingerprint,
    PropagationFingerprint,
    RetrievalFeatures,
    TemporalProfile,
    fingerprint_remediation_base_key,
    fingerprint_remediation_hash,
)

_DEEP = PropagationFingerprint(
    degradation_order=("a", "b"),
    edge_types=("metric",),
    propagation_hops=(1,),
    hop_count=1,
    propagation_depth=3,
    role_transitions=("db>api",),
    edge_type_seq_hash="",
    branching_factor=1,
    terminal_failure_role="api",
    root_degradation_role="db",
)


def test_upstream_shape_splits_same_bool_different_roles() -> None:
    a = IncidentFingerprint("latency", "checkout-api", frozenset({"redis"}), "checkout-api")
    b = IncidentFingerprint("latency", "checkout-api", frozenset({"kafka"}), "checkout-api")
    assert fingerprint_remediation_base_key(a) != fingerprint_remediation_base_key(b)
    assert fingerprint_remediation_hash(a) != fingerprint_remediation_hash(b)


def test_propagation_and_temporal_change_extended_suffix() -> None:
    fp = IncidentFingerprint("errors", "api", frozenset({"db"}), "api")
    base = RetrievalFeatures.from_fingerprint(fp)
    shallow = replace(
        base,
        deploy_pattern=("deploy",),
        propagation_fingerprint=_DEEP,
        temporal_profile=TemporalProfile(120.0, 10.0, 300.0, 5.0),
    )
    empty = replace(
        base,
        deploy_pattern=(),
        propagation_fingerprint=PropagationFingerprint((), (), (), 0, 0),
        temporal_profile=TemporalProfile.missing(),
    )
    assert fingerprint_remediation_hash(fp, shallow) != fingerprint_remediation_hash(fp, empty)
