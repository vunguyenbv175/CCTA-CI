"""Independent QC_001-D expectations.

This module is deliberately analytic and injection-driven.  It has no import
of the production QC package, production parameter constants, or production
family functions.
"""

from __future__ import annotations

from typing import Final

INDEPENDENT_FLAGS: Final[tuple[str, ...]] = (
    "CARDIAC_MOTION",
    "RESPIRATORY_MOTION",
    "STAIR_STEP",
    "POOR_CONTRAST",
    "EXCESSIVE_NOISE",
    "BLOOMING",
    "BEAM_HARDENING",
    "MISREGISTRATION",
    "INCOMPLETE_ANATOMY",
)

INVALID_EXPECTED_CODES: Final[dict[str, str]] = {
    "unsupported-transfer-syntax": "QC_INPUT_SURFACE_INVALID",
    "malformed-pixel-length": "QC_INPUT_SURFACE_INVALID",
    "multiframe": "QC_INPUT_SURFACE_INVALID",
    "identity-frame-mismatch": "QC_SOURCE_IDENTITY_INVALID",
    "missing-stack-member": "QC_GEOMETRY_INVALID",
    "tilted-nonparallel-stack": "QC_GEOMETRY_INVALID",
    "unknown-scalar-semantics": "QC_SCALAR_DOMAIN_INVALID",
    "absent-required-pair": "QC_REQUIRED_PAIR_ABSENT",
}


def expected_family_outcomes(case: object) -> tuple[str, ...]:
    """Return labels derived only from the independent injection state."""

    expected = {flag: "CLEAR" for flag in INDEPENDENT_FLAGS}
    injection = getattr(case, "injection", None)
    flag = getattr(injection, "flag", None)
    if flag is not None:
        expected[flag] = getattr(injection, "severity")
    return tuple(expected[flag] for flag in INDEPENDENT_FLAGS)


def expected_grade_and_flags(case: object) -> tuple[str, tuple[str, ...]]:
    injection = getattr(case, "injection", None)
    flag = getattr(injection, "flag", None)
    if flag is None:
        return "GOOD", ()
    severity = getattr(injection, "severity")
    return ("ACCEPTABLE", (flag,)) if severity == "MILD" else ("LIMITED", (flag,))


def is_unsafe(case: object) -> bool:
    injection = getattr(case, "injection", None)
    return getattr(injection, "severity", None) == "SEVERE"


__all__ = [
    "INDEPENDENT_FLAGS",
    "INVALID_EXPECTED_CODES",
    "expected_family_outcomes",
    "expected_grade_and_flags",
    "is_unsafe",
]
