"""Immutable input and result DTOs for the deterministic QC engine."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from core.artifacts import DerivedArtifact, GeometryBase
from core.dicom.qc import QCAssessmentArtifact
from core.digest import is_sha256_hex

from .constants import (
    ALGORITHM_VERSION,
    FAMILY_ALGORITHM_IDS,
    FAMILY_METRIC_NAMES,
    FIRST_NINE_FLAGS,
    QCAbstentionCode,
    QCOutcome,
    family_parameter_digest,
)


class QCValidationError(ValueError):
    """Fail-closed QC input, payload, provenance, or registration error."""


class QCInputError(QCValidationError):
    """Typed internal boundary failure carrying a closed abstention code."""

    def __init__(self, code: QCAbstentionCode | str, detail: str = "") -> None:
        try:
            self.code = QCAbstentionCode(code)
        except ValueError:
            self.code = QCAbstentionCode.QC_INPUT_SURFACE_INVALID
        self.detail = detail
        message = self.code.value if not detail else f"{self.code.value}: {detail}"
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class CoverageEnvelope:
    """Owner-supplied study coverage envelope from QC_001_SPEC §3.11."""

    schema_version: str = "ccta.qc.coverage_envelope.v1"
    coordinate_system: str = "DICOM_PATIENT_LPS_MM"
    anchor_world_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    inferior_extent_mm: float = 80.0
    superior_extent_mm: float = 80.0
    terminal_band_mm: float = 16.0
    minimum_body_occupancy: float = 0.20
    boundary_rule: str = "INCLUSIVE_VOXEL_CENTER_V1"

    def __post_init__(self) -> None:
        try:
            anchor = tuple(self.anchor_world_mm)
        except TypeError as exc:
            raise QCValidationError("coverage anchor must be a three-value tuple") from exc
        if len(anchor) != 3:
            raise QCValidationError("coverage anchor must contain exactly three values")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in anchor
        ):
            raise QCValidationError("coverage anchor must contain finite numeric values")
        object.__setattr__(self, "anchor_world_mm", tuple(float(value) for value in anchor))


@dataclass(frozen=True, slots=True)
class QCInput:
    """The only input surface accepted by a normal single-volume QC run.

    The array is expected to be the read-only array returned by the canonical
    volume materializer.  Metadata that is not part of the allowlisted
    ``DicomInstance`` surface is intentionally absent from this DTO.
    """

    volume: Any
    array: Any
    source_manifest_sha256: str
    instances: tuple[Any, ...] = ()
    instance_digests: Mapping[str, str] | tuple[tuple[str, str], ...] = ()
    scalar_domain: str = "STORED"
    intensity_unit: str | None = None
    source_low: float | None = None
    source_high: float | None = None
    transform: Any | None = None
    transform_ref: str | None = None
    transform_sha256: str | None = None
    transform_provenance_ref: str | None = None
    geometry_base: GeometryBase | None = None
    subject_refs: tuple[str, ...] = ()
    pixel_padding_present: bool = False
    modality_lut_present: bool = False
    coverage: CoverageEnvelope | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "instances", tuple(self.instances))
        object.__setattr__(self, "subject_refs", tuple(self.subject_refs))


# The shorter name is useful to callers while retaining the explicit name in
# the contract and in reprs.
QCVolumeInput = QCInput
CanonicalQCInput = QCInput


@dataclass(frozen=True, slots=True)
class FamilyResult:
    """One closed deterministic family result and its typed refusal."""

    flag: str
    metric_value: float | None
    outcome: QCOutcome | str
    abstention_code: QCAbstentionCode | str | None = None
    algorithm_id: str | None = None
    algorithm_version: str = ALGORITHM_VERSION
    metric_name: str | None = None
    parameter_digest: str | None = None

    def __post_init__(self) -> None:
        if self.flag not in FIRST_NINE_FLAGS:
            raise ValueError(f"unknown deterministic QC flag: {self.flag!r}")
        object.__setattr__(self, "outcome", QCOutcome(self.outcome))
        if self.abstention_code is not None:
            object.__setattr__(self, "abstention_code", QCAbstentionCode(self.abstention_code))
        expected_algorithm = FAMILY_ALGORITHM_IDS[self.flag]
        expected_metric = FAMILY_METRIC_NAMES[self.flag]
        object.__setattr__(self, "algorithm_id", self.algorithm_id or expected_algorithm)
        object.__setattr__(self, "metric_name", self.metric_name or expected_metric)
        object.__setattr__(
            self,
            "parameter_digest",
            self.parameter_digest or family_parameter_digest(self.flag),
        )
        if self.algorithm_id != expected_algorithm:
            raise ValueError("family algorithm id is not frozen for this flag")
        if self.algorithm_version != ALGORITHM_VERSION:
            raise ValueError("family algorithm version is closed at 1")
        if not is_sha256_hex(self.parameter_digest):
            raise ValueError("family parameter_digest must be lowercase SHA-256")
        if self.outcome == QCOutcome.ABSTAINED:
            if self.metric_value is not None or self.abstention_code is None:
                raise ValueError("ABSTAINED family results require null metric and a code")
        else:
            if self.metric_value is None or self.abstention_code is not None:
                raise ValueError("non-abstained family results require a metric and no code")
            if isinstance(self.metric_value, bool) or not isinstance(
                self.metric_value, (int, float)
            ):
                raise ValueError("non-abstained family metrics must be numeric")
            metric = float(self.metric_value)
            if not math.isfinite(metric):
                raise ValueError("non-abstained family metrics must be finite")
            object.__setattr__(self, "metric_value", metric)


@dataclass(frozen=True, slots=True)
class QCInputRefusal:
    """Production DTO for an upstream reader/stack/materialization refusal."""

    family: str
    reason_code: QCAbstentionCode | str
    upstream_failure: str
    source_manifest_sha256: str

    def __post_init__(self) -> None:
        if self.family not in FIRST_NINE_FLAGS:
            raise ValueError(f"unknown refusal family: {self.family!r}")
        object.__setattr__(self, "reason_code", QCAbstentionCode(self.reason_code))
        if not isinstance(self.upstream_failure, str) or not self.upstream_failure.strip():
            raise ValueError("upstream_failure is required")
        if not is_sha256_hex(self.source_manifest_sha256):
            raise ValueError("source_manifest_sha256 must be lowercase SHA-256")


@dataclass(frozen=True, slots=True)
class RegisteredPairInput:
    """Closed immutable pair surface required by pair-dependent families.

    The field list is intentionally exact and mirrors QC_001_DET §4.  Both
    arrays are read-only canonical arrays; all resolution is performed from
    the supplied tuple of provenance records and never from global state.
    """

    first_volume: Any
    second_volume: Any
    first_array: Any
    second_array: Any
    first_geometry_base: GeometryBase
    second_geometry_base: GeometryBase
    first_phase_ref: str
    second_phase_ref: str
    registration_artifact: DerivedArtifact
    registration_document: Mapping[str, object]
    provenance_records: tuple[Any, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "provenance_records", tuple(self.provenance_records))
        # Import lazily so pair.py can use this DTO without an import cycle.
        from .pair import validate_registered_pair_input

        validate_registered_pair_input(self)


@dataclass(frozen=True, slots=True)
class QCResultBundle:
    """Both QC projections plus their immutable evidence and provenance."""

    assessment: QCAssessmentArtifact
    artifact: DerivedArtifact
    payload_document: Mapping[str, object]
    family_results: tuple[FamilyResult, ...]
    family_evidence_documents: Mapping[str, Mapping[str, object]]
    parameter_manifest: Mapping[str, object]
    provenance_records: tuple[Any, ...]
    family_result_body: Mapping[str, object]
    refusal: QCInputRefusal | None = None

    @property
    def assessment_artifact(self) -> QCAssessmentArtifact:
        return self.assessment

    @property
    def derived_artifact(self) -> DerivedArtifact:
        return self.artifact

    @property
    def provenance(self) -> tuple[Any, ...]:
        return self.provenance_records

    @property
    def quality_grade(self) -> object:
        return self.assessment.payload.quality_grade

    @property
    def flags(self) -> tuple[str, ...]:
        return self.assessment.payload.flags

    @property
    def artifact_ref(self) -> str:
        return self.assessment.artifact_ref


@dataclass(frozen=True, slots=True)
class QCPolicyDiagnostic:
    """Visible reason why an opaque candidate was not real-QC eligible."""

    code: str
    artifact_ref: str | None = None
    detail: str = ""


@dataclass(frozen=True, slots=True)
class RealQCEligibilityResult:
    """Policy result that preserves the unchanged closed gate result."""

    gate_result: Any
    diagnostics: tuple[QCPolicyDiagnostic, ...] = ()
    eligible_assessments: tuple[QCAssessmentArtifact, ...] = ()

    @property
    def decision(self) -> Any:
        return self.gate_result.decision

    @property
    def reason_code(self) -> Any:
        return self.gate_result.reason_code

    @property
    def reason(self) -> Any:
        return self.gate_result.reason

    @property
    def assessment_refs(self) -> tuple[str, ...]:
        return self.gate_result.assessment_refs

    @property
    def proceed(self) -> bool:
        return self.gate_result.proceed

    @property
    def abstained(self) -> bool:
        return self.gate_result.abstained

    @property
    def diagnostic_codes(self) -> tuple[str, ...]:
        return tuple(item.code for item in self.diagnostics)


__all__ = [
    "CanonicalQCInput",
    "CoverageEnvelope",
    "FamilyResult",
    "QCInput",
    "QCInputError",
    "QCInputRefusal",
    "QCResultBundle",
    "QCPolicyDiagnostic",
    "QCValidationError",
    "QCVolumeInput",
    "RealQCEligibilityResult",
    "RegisteredPairInput",
]
