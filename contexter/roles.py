"""Family-aware role normalization for benchmark fingerprint matching."""

from __future__ import annotations

import re

_FAMILY_IN_NAME = re.compile(
    r"(?:^|[\-_.])f(\d+)(?:[\-_.]|$)|family[\-_.]?(\d+)",
    re.IGNORECASE,
)


def extract_family_id(name: str) -> str | None:
    """Return family id string when ``name`` encodes a family number."""
    if not name:
        return None
    match = _FAMILY_IN_NAME.search(name)
    if not match:
        return None
    digits = match.group(1) or match.group(2)
    return digits if digits is not None else None


def family_aware_role(canonical_service: str) -> str:
    """Map a canonical service to ``family-<n>`` when a family number is present."""
    family_id = extract_family_id(canonical_service)
    if family_id is not None:
        return f"family-{family_id}"
    return canonical_service
