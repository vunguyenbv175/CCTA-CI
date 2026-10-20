"""Fail-closed validation errors for the canonical data model.

Every creation-time validation raises a :class:`CanonicalValidationError`
subclass; nothing is ever defaulted or silently repaired (AGENTS.md §3.10,
Master Plan §44).
"""

from __future__ import annotations


class CanonicalValidationError(ValueError):
    """Base class for all fail-closed canonical-model violations."""

    def __init__(self, violations: list[str] | str) -> None:
        self.violations: tuple[str, ...] = (
            (violations,) if isinstance(violations, str) else tuple(violations)
        )
        detail = "; ".join(self.violations)
        super().__init__(f"fail-closed rejection ({len(self.violations)}): {detail}")


class ProvenanceIncompleteError(CanonicalValidationError):
    """A §44 provenance item is missing and not explicitly NotApplicable."""


class ProducerMatrixViolationError(CanonicalValidationError):
    """Artifact kind / producer_type combination is forbidden."""


class RegistrationRequiredError(CanonicalValidationError):
    """Multi-series geometry without an explicit registration transform."""


class GeometryBaseRequiredError(CanonicalValidationError):
    """Geometry-dependent artifact without lineage.geometry_base."""


class ExportGateViolationError(CanonicalValidationError):
    """Export attempted while inputs are not CONFIRMED_FINAL or are STALE."""


class StateTransitionError(CanonicalValidationError):
    """Illegal review/status transition, or non-human CONFIRMED_FINAL."""


class DigestMismatchError(CanonicalValidationError):
    """Self-digest or content digest verification failed."""


class ClosedSchemaError(CanonicalValidationError):
    """Unknown field in a closed-schema record (registry, canonical JSON)."""


class DatasetRoleRejectedError(CanonicalValidationError):
    """Registry dataset-role binding violated a locked-test REJECT rule."""
