"""Deterministic Imaging Core 001-A classification and stack admission.

This module contains no model calls and no heuristic fallback.  The visible
decision is obtained from the frozen 4 x 5 x 5 x 2 x 2 boundary table; the
evidence extractors only map canonical DICOM instances to that table's
closed axes.
"""

from __future__ import annotations

import enum
import math
import re
import statistics
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from core.dicom.instance import (
    ADMITTED_TRANSFER_SYNTAXES,
    CARDIAC_SYNCHRONIZATION_TECHNIQUES,
    CT_IMAGE_STORAGE_SOP_CLASS_UID,
    PHOTOMETRIC_INTERPRETATIONS,
    RESCALE_TYPES,
    DicomInstance,
    admitted_bit_depth_tuple,
    stored_padding_bits_are_zero,
    timing_pair_is_valid,
)

ORIENTATION_TOLERANCE: float = 1e-6
PARALLEL_DIRECTION_TOLERANCE: float = 1e-6
IN_PLANE_SPACING_TOLERANCE_MM: float = 1e-6
DUPLICATE_POSITION_TOLERANCE_MM: float = 1e-6
IN_PLANE_DRIFT_TOLERANCE_MM: float = 1e-3
INTER_SLICE_GAP_TOLERANCE_MM: float = 1e-3
# AMENDMENT V1.2 (2026-09-17): relative bound for the uniform projected-gap
# check only — gaps are uniform iff max(gaps)-min(gaps) <= max(1e-6 mm, 1% of
# the median projected gap). The absolute 1e-3 constant above is retained for
# its other use (adjacent-thickness missing-slice check).
INTER_SLICE_GAP_RELATIVE_TOLERANCE: float = 0.01
INTER_SLICE_GAP_ABSOLUTE_FLOOR_MM: float = 1e-6

CARDIAC_KEYWORDS: tuple[str, ...] = (
    "cardiac",
    "heart",
    "coronary",
    "cardio",
    "chest",
    "cca",
    "ccta",
)
NON_CONTRAST_PHRASES: tuple[str, ...] = (
    "non-contrast",
    "noncontrast",
    "without contrast",
    "w/o contrast",
    "unenhanced",
    "non enhanced",
    "non-enhanced",
    "native",
)
EXCLUDED_IMAGE_TYPE_COMPONENTS: tuple[str, ...] = (
    "DERIVED",
    "LOCALIZER",
    "SECONDARY",
    "MPR",
    "MIP",
    "MINIP",
    "3D",
    "VOLUME_RENDERED",
    "VR",
    "THICKSLAB",
    "SLAB",
    "PROJECTION",
    "REFORMATTED",
    "RESAMPLED",
    "DYNAMIC",
    "SPECTRAL",
    "DUAL_ENERGY",
)


class _ValueEnum(enum.StrEnum):
    """String enums compare naturally with the frozen wire vocabulary."""


class ClassificationLabel(_ValueEnum):
    CARDIAC_CT = "CARDIAC_CT"
    CONTRAST = "CONTRAST"
    NON_CONTRAST = "NON_CONTRAST"
    PHASE_BOUND = "PHASE_BOUND"
    UNKNOWN = "UNKNOWN"
    NOT_ANALYZABLE = "NOT_ANALYZABLE"


class ClassificationOutcome(_ValueEnum):
    IMPORTABLE_CLASSIFIED = "IMPORTABLE+CLASSIFIED"
    IMPORTABLE_UNKNOWN = "IMPORTABLE+UNKNOWN"
    NOT_ANALYZABLE = "NOT_ANALYZABLE"


class ClassificationReason(_ValueEnum):
    PROTOCOL_UNSUPPORTED = "PROTOCOL_UNSUPPORTED"
    DERIVED_IMAGE_UNSUPPORTED = "DERIVED_IMAGE_UNSUPPORTED"
    PROTOCOL_EVIDENCE_ABSENT = "PROTOCOL_EVIDENCE_ABSENT"
    CONTRAST_EVIDENCE_ABSENT = "CONTRAST_EVIDENCE_ABSENT"
    CONTRAST_EVIDENCE_INCOMPLETE = "CONTRAST_EVIDENCE_INCOMPLETE"
    CONTRAST_CONFLICT = "CONTRAST_CONFLICT"
    PHASE_EVIDENCE_ABSENT = "PHASE_EVIDENCE_ABSENT"
    PHASE_ACQUISITION_ONLY = "PHASE_ACQUISITION_ONLY"
    PHASE_CONFLICT = "PHASE_CONFLICT"
    PHASE_ECG_AMBIGUOUS = "PHASE_ECG_AMBIGUOUS"
    DICOM_SYNTAX_INVALID = "DICOM_SYNTAX_INVALID"
    UNSUPPORTED_TRANSFER_SYNTAX = "UNSUPPORTED_TRANSFER_SYNTAX"
    GEOMETRY_INVALID = "GEOMETRY_INVALID"
    SERIES_INCOMPLETE = "SERIES_INCOMPLETE"
    SERIES_STACK_INVALID = "SERIES_STACK_INVALID"
    IDENTITY_MISMATCH = "IDENTITY_MISMATCH"
    CLASSIFICATION_COMPLETE = "CLASSIFICATION_COMPLETE"


class ProtocolState(_ValueEnum):
    CARDIAC_KEYWORD_MATCH = "CARDIAC_KEYWORD_MATCH"
    KEYWORD_ABSENT = "KEYWORD_ABSENT"
    PROTOCOL_IDENTITY_INVALID = "PROTOCOL_IDENTITY_INVALID"
    EXCLUDED_IMAGE_TYPE = "EXCLUDED_IMAGE_TYPE"


class ContrastState(_ValueEnum):
    CONTRAST = "CONTRAST"
    NON_CONTRAST = "NON_CONTRAST"
    UNKNOWN_AGENT_ABSENT = "UNKNOWN_AGENT_ABSENT"
    UNKNOWN_UNCORROBORATED_AGENT = "UNKNOWN_UNCORROBORATED_AGENT"
    CONTRADICTORY = "CONTRADICTORY"


class PhaseState(_ValueEnum):
    BOUND = "BOUND"
    UNKNOWN_NO_EVIDENCE = "UNKNOWN_NO_EVIDENCE"
    UNKNOWN_ACQUISITION_ONLY = "UNKNOWN_ACQUISITION_ONLY"
    CONFLICT = "CONFLICT"
    ECG_AMBIGUOUS = "ECG_AMBIGUOUS"


class GeometryState(_ValueEnum):
    VALID = "VALID"
    INVALID = "INVALID"


class SyntaxState(_ValueEnum):
    VALID = "VALID"
    INVALID = "INVALID"


@dataclass(frozen=True, slots=True)
class BoundaryRow:
    """One explicit row of the closed classification oracle."""

    number: int
    protocol: ProtocolState
    contrast: ContrastState
    phase: PhaseState
    geometry: GeometryState
    syntax: SyntaxState
    outcome: ClassificationOutcome
    visible_reason: ClassificationReason


@dataclass(frozen=True, slots=True)
class ClassificationProvenance:
    """PHI-safe evidence metadata for the Rule G classification record."""

    classifier: str
    version: str
    rule_table_version: str
    evidence_fields: tuple[str, ...]
    detail_codes: tuple[str, ...] = ()

    @property
    def algorithm(self) -> str:
        return self.classifier

    @property
    def algorithm_version(self) -> str:
        return self.version

    def as_dict(self) -> dict[str, object]:
        return {
            "classifier": self.classifier,
            "version": self.version,
            "rule_table_version": self.rule_table_version,
            "evidence_fields": self.evidence_fields,
            "detail_codes": self.detail_codes,
        }


@dataclass(frozen=True, slots=True)
class DimensionClassification:
    """One independently stored protocol, contrast, or phase result."""

    value: ClassificationLabel
    reason_code: ClassificationReason
    evidence_fields: tuple[str, ...] = ()

    @property
    def result(self) -> ClassificationLabel:
        return self.value

    @property
    def classification(self) -> ClassificationLabel:
        return self.value

    @property
    def reason(self) -> ClassificationReason:
        return self.reason_code


@dataclass(frozen=True, slots=True)
class ClassificationResult:
    """Complete deterministic classification for one candidate series."""

    protocol: DimensionClassification
    contrast: DimensionClassification
    phase: DimensionClassification
    outcome: ClassificationOutcome
    visible_reason: ClassificationReason
    provenance: ClassificationProvenance
    classifier: str = "rule-table"
    version: str = "1"
    phase_label: str | None = None
    detail_codes: tuple[str, ...] = ()

    @property
    def result(self) -> ClassificationOutcome:
        return self.outcome

    @property
    def reason_code(self) -> ClassificationReason:
        return self.visible_reason

    @property
    def reason(self) -> ClassificationReason:
        return self.visible_reason

    @property
    def importable(self) -> bool:
        return self.outcome != ClassificationOutcome.NOT_ANALYZABLE

    @property
    def is_importable(self) -> bool:
        return self.importable


@dataclass(frozen=True, slots=True)
class StackAdmissionResult:
    """Result of B6 series stack validation."""

    admitted: bool
    instances: tuple[DicomInstance, ...]
    slice_spacing_mm: float | None
    normal: tuple[float, float, float] | None
    reason_code: ClassificationReason | None = None
    detail_code: ClassificationReason | None = None

    @property
    def accepted(self) -> bool:
        return self.admitted

    @property
    def ordered_instances(self) -> tuple[DicomInstance, ...]:
        return self.instances

    @property
    def reason(self) -> ClassificationReason | None:
        return self.reason_code


class StackAdmissionError(ValueError):
    """Typed series-level refusal; no partial stack is returned."""

    def __init__(self, detail_code: ClassificationReason) -> None:
        self.reason_code = ClassificationReason.GEOMETRY_INVALID
        self.detail_code = detail_code
        super().__init__(f"{self.reason_code.value}: {detail_code.value}")


def _boundary_reason(
    protocol: ProtocolState,
    contrast: ContrastState,
    phase: PhaseState,
    geometry: GeometryState,
    syntax: SyntaxState,
) -> tuple[ClassificationOutcome, ClassificationReason]:
    """Apply the spec's fixed visible-reason precedence."""

    if syntax == SyntaxState.INVALID:
        return ClassificationOutcome.NOT_ANALYZABLE, ClassificationReason.DICOM_SYNTAX_INVALID
    if geometry == GeometryState.INVALID:
        return ClassificationOutcome.NOT_ANALYZABLE, ClassificationReason.GEOMETRY_INVALID
    if protocol == ProtocolState.PROTOCOL_IDENTITY_INVALID:
        return ClassificationOutcome.NOT_ANALYZABLE, ClassificationReason.PROTOCOL_UNSUPPORTED
    if protocol == ProtocolState.EXCLUDED_IMAGE_TYPE:
        return ClassificationOutcome.NOT_ANALYZABLE, ClassificationReason.DERIVED_IMAGE_UNSUPPORTED
    if contrast == ContrastState.CONTRADICTORY:
        return ClassificationOutcome.NOT_ANALYZABLE, ClassificationReason.CONTRAST_CONFLICT
    if phase == PhaseState.CONFLICT:
        return ClassificationOutcome.NOT_ANALYZABLE, ClassificationReason.PHASE_CONFLICT
    if phase == PhaseState.ECG_AMBIGUOUS:
        return ClassificationOutcome.NOT_ANALYZABLE, ClassificationReason.PHASE_ECG_AMBIGUOUS

    missing_reason: ClassificationReason | None = None
    if protocol == ProtocolState.KEYWORD_ABSENT:
        missing_reason = ClassificationReason.PROTOCOL_EVIDENCE_ABSENT
    elif contrast == ContrastState.UNKNOWN_AGENT_ABSENT:
        missing_reason = ClassificationReason.CONTRAST_EVIDENCE_ABSENT
    elif contrast == ContrastState.UNKNOWN_UNCORROBORATED_AGENT:
        missing_reason = ClassificationReason.CONTRAST_EVIDENCE_INCOMPLETE
    elif phase == PhaseState.UNKNOWN_NO_EVIDENCE:
        missing_reason = ClassificationReason.PHASE_EVIDENCE_ABSENT
    elif phase == PhaseState.UNKNOWN_ACQUISITION_ONLY:
        missing_reason = ClassificationReason.PHASE_ACQUISITION_ONLY

    if missing_reason is None:
        return (
            ClassificationOutcome.IMPORTABLE_CLASSIFIED,
            ClassificationReason.CLASSIFICATION_COMPLETE,
        )
    return ClassificationOutcome.IMPORTABLE_UNKNOWN, missing_reason


def _make_boundary_grid() -> tuple[BoundaryRow, ...]:
    rows: list[BoundaryRow] = []
    number = 0
    for protocol in ProtocolState:
        for contrast in ContrastState:
            for phase in PhaseState:
                for geometry in GeometryState:
                    for syntax in SyntaxState:
                        number += 1
                        outcome, reason = _boundary_reason(
                            protocol, contrast, phase, geometry, syntax
                        )
                        rows.append(
                            BoundaryRow(
                                number=number,
                                protocol=protocol,
                                contrast=contrast,
                                phase=phase,
                                geometry=geometry,
                                syntax=syntax,
                                outcome=outcome,
                                visible_reason=reason,
                            )
                        )
    if len(rows) != 400:
        raise AssertionError("closed classification boundary must contain 400 rows")
    return tuple(rows)


CLASSIFICATION_BOUNDARY_GRID: tuple[BoundaryRow, ...] = _make_boundary_grid()
BOUNDARY_GRID = CLASSIFICATION_BOUNDARY_GRID
COMPLETE_CLASSIFICATION_ORACLE = CLASSIFICATION_BOUNDARY_GRID


def classification_boundary_grid() -> tuple[BoundaryRow, ...]:
    """Return the immutable 400-row fixture oracle."""

    return CLASSIFICATION_BOUNDARY_GRID


def classify_boundary_states(
    protocol: ProtocolState | str,
    contrast: ContrastState | str,
    phase: PhaseState | str,
    geometry: GeometryState | str,
    syntax: SyntaxState | str,
    *,
    evidence_fields: Iterable[str] = (),
    detail_codes: Iterable[str] = (),
) -> ClassificationResult:
    """Classify a closed boundary tuple exactly as the normative table does."""

    protocol_state = ProtocolState(protocol)
    contrast_state = ContrastState(contrast)
    phase_state = PhaseState(phase)
    geometry_state = GeometryState(geometry)
    syntax_state = SyntaxState(syntax)
    outcome, reason = _boundary_reason(
        protocol_state, contrast_state, phase_state, geometry_state, syntax_state
    )
    protocol_value = (
        ClassificationLabel.CARDIAC_CT
        if protocol_state == ProtocolState.CARDIAC_KEYWORD_MATCH
        else ClassificationLabel.UNKNOWN
        if protocol_state == ProtocolState.KEYWORD_ABSENT
        else ClassificationLabel.NOT_ANALYZABLE
    )
    contrast_value = {
        ContrastState.CONTRAST: ClassificationLabel.CONTRAST,
        ContrastState.NON_CONTRAST: ClassificationLabel.NON_CONTRAST,
        ContrastState.UNKNOWN_AGENT_ABSENT: ClassificationLabel.UNKNOWN,
        ContrastState.UNKNOWN_UNCORROBORATED_AGENT: ClassificationLabel.UNKNOWN,
        ContrastState.CONTRADICTORY: ClassificationLabel.NOT_ANALYZABLE,
    }[contrast_state]
    phase_value = (
        ClassificationLabel.PHASE_BOUND
        if phase_state == PhaseState.BOUND
        else ClassificationLabel.UNKNOWN
        if phase_state in {PhaseState.UNKNOWN_NO_EVIDENCE, PhaseState.UNKNOWN_ACQUISITION_ONLY}
        else ClassificationLabel.NOT_ANALYZABLE
    )
    protocol_reason = {
        ProtocolState.CARDIAC_KEYWORD_MATCH: ClassificationReason.CLASSIFICATION_COMPLETE,
        ProtocolState.KEYWORD_ABSENT: ClassificationReason.PROTOCOL_EVIDENCE_ABSENT,
        ProtocolState.PROTOCOL_IDENTITY_INVALID: ClassificationReason.PROTOCOL_UNSUPPORTED,
        ProtocolState.EXCLUDED_IMAGE_TYPE: ClassificationReason.DERIVED_IMAGE_UNSUPPORTED,
    }[protocol_state]
    contrast_reason = {
        ContrastState.CONTRAST: ClassificationReason.CLASSIFICATION_COMPLETE,
        ContrastState.NON_CONTRAST: ClassificationReason.CLASSIFICATION_COMPLETE,
        ContrastState.UNKNOWN_AGENT_ABSENT: ClassificationReason.CONTRAST_EVIDENCE_ABSENT,
        ContrastState.UNKNOWN_UNCORROBORATED_AGENT: (
            ClassificationReason.CONTRAST_EVIDENCE_INCOMPLETE
        ),
        ContrastState.CONTRADICTORY: ClassificationReason.CONTRAST_CONFLICT,
    }[contrast_state]
    phase_reason = {
        PhaseState.BOUND: ClassificationReason.CLASSIFICATION_COMPLETE,
        PhaseState.UNKNOWN_NO_EVIDENCE: ClassificationReason.PHASE_EVIDENCE_ABSENT,
        PhaseState.UNKNOWN_ACQUISITION_ONLY: ClassificationReason.PHASE_ACQUISITION_ONLY,
        PhaseState.CONFLICT: ClassificationReason.PHASE_CONFLICT,
        PhaseState.ECG_AMBIGUOUS: ClassificationReason.PHASE_ECG_AMBIGUOUS,
    }[phase_state]
    details = tuple(dict.fromkeys(str(code) for code in detail_codes))
    provenance = ClassificationProvenance(
        classifier="rule-table",
        version="1",
        rule_table_version="1",
        evidence_fields=tuple(dict.fromkeys(str(field) for field in evidence_fields)),
        detail_codes=details,
    )
    all_evidence = tuple(dict.fromkeys(str(field) for field in evidence_fields))
    return ClassificationResult(
        protocol=DimensionClassification(protocol_value, protocol_reason, all_evidence),
        contrast=DimensionClassification(contrast_value, contrast_reason, all_evidence),
        phase=DimensionClassification(phase_value, phase_reason, all_evidence),
        outcome=outcome,
        visible_reason=reason,
        provenance=provenance,
        detail_codes=details,
    )


def _norm_text(value: str | None) -> str:
    return value or ""


def _contains_term(value: str, term: str) -> bool:
    # Punctuation remains in the normalized value, while token boundaries
    # ensure e.g. "heartland" does not match "heart".
    return re.search(r"(?<![\w])" + re.escape(term) + r"(?![\w])", value) is not None


def _contains_any(value: str, terms: Iterable[str]) -> bool:
    return any(_contains_term(value, term) for term in terms)


def _protocol_state(instances: Sequence[DicomInstance]) -> tuple[ProtocolState, tuple[str, ...]]:
    fields: list[str] = ["Modality", "SOPClassUID", "ImageType"]
    if any(
        instance.modality != "CT"
        or instance.sop_class_uid != CT_IMAGE_STORAGE_SOP_CLASS_UID
        for instance in instances
    ):
        return ProtocolState.PROTOCOL_IDENTITY_INVALID, tuple(fields)
    if any(
        len(instance.image_type) < 2
        or instance.image_type[0].casefold() != "original"
        or instance.image_type[1].casefold() != "primary"
        or any(
            component.casefold() in {item.casefold() for item in EXCLUDED_IMAGE_TYPE_COMPONENTS}
            for component in instance.image_type
        )
        for instance in instances
    ):
        return ProtocolState.EXCLUDED_IMAGE_TYPE, tuple(fields)
    matched_fields: list[str] = []
    for field_name, values in (
        ("SeriesDescription", (instance.series_description for instance in instances)),
        ("BodyPartExamined", (instance.body_part_examined for instance in instances)),
    ):
        if any(_contains_any(_norm_text(value), CARDIAC_KEYWORDS) for value in values):
            matched_fields.append(field_name)
    if any(instance.protocol_name is not None for instance in instances):
        fields.append("ProtocolName")
    return (
        (ProtocolState.CARDIAC_KEYWORD_MATCH if matched_fields else ProtocolState.KEYWORD_ABSENT),
        tuple(dict.fromkeys((*fields, *matched_fields))),
    )


def _contrast_state(instances: Sequence[DicomInstance]) -> tuple[ContrastState, tuple[str, ...]]:
    agent_values = tuple(instance.contrast_bolus_agent for instance in instances)
    agent_present = any(value is not None and bool(value.strip()) for value in agent_values)
    positive_volume = any(
        value is not None and math.isfinite(value) and value > 0.0
        for value in (instance.contrast_bolus_volume_ml for instance in instances)
    )
    zero_volume = any(
        value is not None and math.isfinite(value) and value == 0.0
        for value in (instance.contrast_bolus_volume_ml for instance in instances)
    )
    timing_evidence = tuple(
        (instance.contrast_bolus_start_time, instance.contrast_bolus_stop_time)
        for instance in instances
        if instance.contrast_bolus_start_time is not None
        or instance.contrast_bolus_stop_time is not None
    )
    timing_present = bool(timing_evidence)
    timing_valid = bool(timing_evidence) and all(
        timing_pair_is_valid(start, stop) for start, stop in timing_evidence
    )
    invalid_timing = timing_present and not timing_valid
    protocol_text = tuple(
        value
        for instance in instances
        for value in (
            instance.series_description,
            instance.body_part_examined,
            instance.protocol_name,
        )
        if value
    )
    explicit_noncontrast = any(
        _contains_any(value, NON_CONTRAST_PHRASES) for value in protocol_text
    )
    fields = ["ContrastBolusAgent"]
    if any(instance.contrast_bolus_volume_ml is not None for instance in instances):
        fields.append("ContrastBolusVolume")
    if any(instance.contrast_bolus_start_time is not None for instance in instances):
        fields.append("ContrastBolusStartTime")
    if any(instance.contrast_bolus_stop_time is not None for instance in instances):
        fields.append("ContrastBolusStopTime")
    if any(instance.contrast_flow_rate is not None for instance in instances):
        fields.append("ContrastFlowRate")
    if any(instance.contrast_flow_duration is not None for instance in instances):
        fields.append("ContrastFlowDuration")
    if explicit_noncontrast:
        fields.extend(("SeriesDescription", "BodyPartExamined", "ProtocolName"))

    if agent_present and (zero_volume or invalid_timing or explicit_noncontrast):
        return ContrastState.CONTRADICTORY, tuple(dict.fromkeys(fields))
    if not agent_present and explicit_noncontrast and not positive_volume and not timing_present:
        return ContrastState.NON_CONTRAST, tuple(dict.fromkeys(fields))
    if agent_present and (positive_volume or timing_valid):
        return ContrastState.CONTRAST, tuple(dict.fromkeys(fields))
    if not agent_present and (positive_volume or timing_present):
        return (
            ContrastState.CONTRADICTORY
            if explicit_noncontrast
            else ContrastState.UNKNOWN_UNCORROBORATED_AGENT,
            tuple(dict.fromkeys(fields)),
        )
    if agent_present:
        return ContrastState.UNKNOWN_UNCORROBORATED_AGENT, tuple(dict.fromkeys(fields))
    return ContrastState.UNKNOWN_AGENT_ABSENT, tuple(dict.fromkeys(fields))


def _phase_state(
    instances: Sequence[DicomInstance],
) -> tuple[PhaseState, tuple[str, ...], str | None]:
    fields = [
        "TemporalPositionIndex",
        "TriggerTime",
        "CardiacSynchronizationTechnique",
        "CardiacSignalSource",
        "AcquisitionNumber",
        "NumberOfTemporalPositions",
    ]
    indices = {
        instance.temporal_position_index
        for instance in instances
        if instance.temporal_position_index is not None
    }
    if len(indices) > 1:
        return PhaseState.CONFLICT, tuple(fields), None
    techniques = {
        instance.cardiac_synchronization_technique
        for instance in instances
        if instance.cardiac_synchronization_technique is not None
    }
    ambiguous_technique = any(
        technique not in CARDIAC_SYNCHRONIZATION_TECHNIQUES for technique in techniques
    )
    if ambiguous_technique:
        return PhaseState.ECG_AMBIGUOUS, tuple(fields), None
    if len(techniques) > 1:
        return PhaseState.CONFLICT, tuple(fields), None
    trigger_values = {
        instance.trigger_time_ms
        for instance in instances
        if instance.trigger_time_ms is not None
    }
    if any(
        value is not None and (not math.isfinite(value) or value < 0.0)
        for value in trigger_values
    ):
        return PhaseState.CONFLICT, tuple(fields), None
    if len(trigger_values) > 1:
        return PhaseState.CONFLICT, tuple(fields), None
    temporal_counts = {
        instance.number_of_temporal_positions
        for instance in instances
        if instance.number_of_temporal_positions is not None
    }
    if len(temporal_counts) > 1:
        return PhaseState.CONFLICT, tuple(fields), None
    if indices:
        index = next(iter(indices))
        return PhaseState.BOUND, tuple(fields), str(index)
    if any(instance.acquisition_number is not None for instance in instances):
        return PhaseState.UNKNOWN_ACQUISITION_ONLY, tuple(fields), None
    return PhaseState.UNKNOWN_NO_EVIDENCE, tuple(fields), None


def _vector_norm(value: tuple[float, float, float]) -> float:
    return math.sqrt(sum(component * component for component in value))


def _cross(
    left: tuple[float, float, float], right: tuple[float, float, float]
) -> tuple[float, float, float]:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _dot(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _subtract(
    left: tuple[float, float, float], right: tuple[float, float, float]
) -> tuple[float, float, float]:
    return tuple(a - b for a, b in zip(left, right))  # type: ignore[return-value]


def _scale(value: tuple[float, float, float], factor: float) -> tuple[float, float, float]:
    return tuple(component * factor for component in value)  # type: ignore[return-value]


def _stack_failure(
    instances: Sequence[DicomInstance], detail: ClassificationReason
) -> StackAdmissionResult:
    return StackAdmissionResult(
        admitted=False,
        instances=(),
        slice_spacing_mm=None,
        normal=None,
        reason_code=ClassificationReason.GEOMETRY_INVALID,
        detail_code=detail,
    )


def validate_stack(
    instances: Sequence[DicomInstance],
    *,
    skipped_instance_could_belong: bool = False,
    skipped: bool | None = None,
) -> StackAdmissionResult:
    """Validate a complete stack under the frozen B6 tolerances.

    AMENDMENT V1.2: the uniform projected-gap check uses
    max(1e-6 mm, 1% of the median projected gap); every other comparison
    stays absolute.
    """

    if skipped is not None:
        skipped_instance_could_belong = skipped
    if not instances:
        return _stack_failure(instances, ClassificationReason.SERIES_INCOMPLETE)
    if skipped_instance_could_belong:
        return _stack_failure(instances, ClassificationReason.SERIES_INCOMPLETE)

    first = instances[0]
    if any(
        not item.study_instance_uid
        or not item.series_instance_uid
        or not item.frame_of_reference_uid
        or not item.sop_instance_uid
        for item in instances
    ):
        return _stack_failure(instances, ClassificationReason.IDENTITY_MISMATCH)
    identities = {
        (item.study_instance_uid, item.series_instance_uid, item.frame_of_reference_uid)
        for item in instances
    }
    if len(identities) != 1:
        return _stack_failure(instances, ClassificationReason.IDENTITY_MISMATCH)
    if len({item.sop_instance_uid for item in instances}) != len(instances):
        return _stack_failure(instances, ClassificationReason.IDENTITY_MISMATCH)
    if any(
        item.rows <= 0
        or item.columns <= 0
        or len(item.image_position_patient) != 3
        or len(item.image_orientation_patient) != 6
        or len(item.pixel_spacing) != 2
        or any(
            not math.isfinite(value)
            for value in (
                *item.image_position_patient,
                *item.image_orientation_patient,
                *item.pixel_spacing,
            )
        )
        or any(value <= 0.0 for value in item.pixel_spacing)
        for item in instances
    ):
        return _stack_failure(instances, ClassificationReason.SERIES_STACK_INVALID)
    if any(
        item.transfer_syntax_uid not in ADMITTED_TRANSFER_SYNTAXES
        or item.photometric_interpretation not in PHOTOMETRIC_INTERPRETATIONS
        or not admitted_bit_depth_tuple(
            item.bits_allocated,
            item.bits_stored,
            item.high_bit,
            item.pixel_representation,
        )
        or not stored_padding_bits_are_zero(
            item.pixel_data,
            bits_stored=item.bits_stored,
            high_bit=item.high_bit,
        )
        or item.pixel_representation not in {0, 1}
        or item.samples_per_pixel != 1
        or item.number_of_frames != 1
        or item.pixel_data_byte_length < 0
        or len(item.pixel_data) != item.pixel_data_byte_length
        or item.pixel_data_byte_length != item.rows * item.columns * 2
        or not math.isfinite(item.rescale_slope)
        or not math.isfinite(item.rescale_intercept)
        or item.rescale_type not in RESCALE_TYPES | {None, ""}
        or (
            item.contrast_bolus_volume_ml is not None
            and item.contrast_bolus_volume_ml < 0.0
        )
        or (
            item.contrast_flow_rate is not None
            and item.contrast_flow_rate < 0.0
        )
        or (
            item.contrast_flow_duration is not None
            and item.contrast_flow_duration < 0.0
        )
        for item in instances
    ):
        return _stack_failure(instances, ClassificationReason.DICOM_SYNTAX_INVALID)
    # AMENDMENT V1.1 stack uniformity (IMAGING_CORE_001_SPEC Part 4.1): every
    # member above already carries an admitted tuple, but a stack MIXING
    # conformant 12-bit and 16-bit members is still a series-level refusal.
    # A stack that would concatenate mixed bit depths is never admitted; the
    # adapter seam (materialize_dicom_stack) enforces the identical rule.
    if len({(item.bits_allocated, item.bits_stored, item.high_bit) for item in instances}) != 1:
        return _stack_failure(instances, ClassificationReason.SERIES_STACK_INVALID)

    def orientation(
        item: DicomInstance,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        return item.image_orientation_patient[:3], item.image_orientation_patient[3:]

    row, column = orientation(first)
    if (
        abs(_vector_norm(row) - 1.0) > ORIENTATION_TOLERANCE
        or abs(_vector_norm(column) - 1.0) > ORIENTATION_TOLERANCE
        or abs(_dot(row, column)) > ORIENTATION_TOLERANCE
    ):
        return _stack_failure(instances, ClassificationReason.SERIES_STACK_INVALID)
    first_normal = _cross(row, column)
    first_normal_norm = _vector_norm(first_normal)
    if abs(first_normal_norm - 1.0) > ORIENTATION_TOLERANCE:
        return _stack_failure(instances, ClassificationReason.SERIES_STACK_INVALID)
    for item in instances:
        current_row, current_column = orientation(item)
        if (
            abs(_vector_norm(current_row) - 1.0) > ORIENTATION_TOLERANCE
            or abs(_vector_norm(current_column) - 1.0) > ORIENTATION_TOLERANCE
            or abs(_dot(current_row, current_column)) > ORIENTATION_TOLERANCE
        ):
            return _stack_failure(instances, ClassificationReason.SERIES_STACK_INVALID)
        current_normal = _cross(current_row, current_column)
        if abs(abs(_dot(first_normal, current_normal)) - 1.0) > PARALLEL_DIRECTION_TOLERANCE:
            return _stack_failure(instances, ClassificationReason.SERIES_STACK_INVALID)
        if any(
            abs(a - b) > PARALLEL_DIRECTION_TOLERANCE
            for a, b in zip(current_row, row)
        ) or any(abs(a - b) > PARALLEL_DIRECTION_TOLERANCE for a, b in zip(current_column, column)):
            return _stack_failure(instances, ClassificationReason.SERIES_STACK_INVALID)
        if any(
            abs(a - b) > IN_PLANE_SPACING_TOLERANCE_MM
            for a, b in zip(item.pixel_spacing, first.pixel_spacing)
        ):
            return _stack_failure(instances, ClassificationReason.SERIES_STACK_INVALID)
        if item.rows != first.rows or item.columns != first.columns:
            return _stack_failure(instances, ClassificationReason.SERIES_STACK_INVALID)
        if item.slice_thickness_mm is not None and (
            not math.isfinite(item.slice_thickness_mm) or item.slice_thickness_mm <= 0.0
        ):
            return _stack_failure(instances, ClassificationReason.SERIES_STACK_INVALID)

    origin = first.image_position_patient

    def projection(item: DicomInstance) -> float:
        return _dot(_subtract(item.image_position_patient, origin), first_normal)

    ordered = tuple(sorted(instances, key=projection))
    projected = tuple(projection(item) for item in ordered)
    gaps = tuple(right - left for left, right in zip(projected, projected[1:]))
    if any(gap <= DUPLICATE_POSITION_TOLERANCE_MM for gap in gaps):
        return _stack_failure(instances, ClassificationReason.SERIES_STACK_INVALID)
    for item, position in zip(ordered, projected):
        delta = _subtract(item.image_position_patient, origin)
        in_plane = _subtract(delta, _scale(first_normal, position))
        if _vector_norm(in_plane) > IN_PLANE_DRIFT_TOLERANCE_MM:
            return _stack_failure(instances, ClassificationReason.SERIES_STACK_INVALID)
    if not gaps:
        slice_spacing = first.slice_thickness_mm
    else:
        median_gap = statistics.median(gaps)
        gap_uniformity_tolerance_mm = max(
            INTER_SLICE_GAP_ABSOLUTE_FLOOR_MM,
            INTER_SLICE_GAP_RELATIVE_TOLERANCE * median_gap,
        )
        if max(gaps) - min(gaps) > gap_uniformity_tolerance_mm:
            return _stack_failure(instances, ClassificationReason.SERIES_STACK_INVALID)
        slice_spacing = gaps[0]
        for index, gap in enumerate(gaps):
            adjacent_thicknesses = tuple(
                thickness
                for thickness in (
                    ordered[index].slice_thickness_mm,
                    ordered[index + 1].slice_thickness_mm,
                )
                if thickness is not None
            )
            if adjacent_thicknesses and gap > (
                max(adjacent_thicknesses) + INTER_SLICE_GAP_TOLERANCE_MM
            ):
                return _stack_failure(instances, ClassificationReason.SERIES_INCOMPLETE)
    if slice_spacing is None or slice_spacing <= 0.0 or not math.isfinite(slice_spacing):
        return _stack_failure(instances, ClassificationReason.SERIES_STACK_INVALID)
    return StackAdmissionResult(
        admitted=True,
        instances=ordered,
        slice_spacing_mm=float(slice_spacing),
        normal=first_normal,
    )


def admit_stack(
    instances: Sequence[DicomInstance],
    *,
    skipped_instance_could_belong: bool = False,
) -> tuple[DicomInstance, ...]:
    """Require admission and return the complete ordered stack only."""

    result = validate_stack(
        instances, skipped_instance_could_belong=skipped_instance_could_belong
    )
    if not result.admitted:
        assert result.detail_code is not None
        raise StackAdmissionError(result.detail_code)
    return result.instances


def _syntax_state(instances: Sequence[DicomInstance]) -> tuple[SyntaxState, tuple[str, ...]]:
    details: list[str] = []
    invalid = False
    for item in instances:
        if item.transfer_syntax_uid not in ADMITTED_TRANSFER_SYNTAXES:
            invalid = True
            details.append("UNSUPPORTED_TRANSFER_SYNTAX")
        if item.pixel_data_byte_length != item.rows * item.columns * 2:
            invalid = True
            details.append("DICOM_SYNTAX_INVALID")
        if not admitted_bit_depth_tuple(
            item.bits_allocated,
            item.bits_stored,
            item.high_bit,
            item.pixel_representation,
        ):
            invalid = True
            details.append("DICOM_SYNTAX_INVALID")
        if not stored_padding_bits_are_zero(
            item.pixel_data,
            bits_stored=item.bits_stored,
            high_bit=item.high_bit,
        ):
            invalid = True
            details.append("DICOM_SYNTAX_INVALID")
        if item.number_of_frames != 1 or item.samples_per_pixel != 1:
            invalid = True
            details.append("DICOM_SYNTAX_INVALID")
        if item.photometric_interpretation not in PHOTOMETRIC_INTERPRETATIONS:
            invalid = True
            details.append("DICOM_SYNTAX_INVALID")
        if item.pixel_representation not in {0, 1}:
            invalid = True
            details.append("DICOM_SYNTAX_INVALID")
        if item.rescale_type not in RESCALE_TYPES | {None, ""}:
            invalid = True
            details.append("DICOM_SYNTAX_INVALID")
        if item.contrast_bolus_volume_ml is not None and item.contrast_bolus_volume_ml < 0.0:
            invalid = True
            details.append("DICOM_SYNTAX_INVALID")
        if item.trigger_time_ms is not None and (
            not math.isfinite(item.trigger_time_ms) or item.trigger_time_ms < 0.0
        ):
            invalid = True
            details.append("DICOM_SYNTAX_INVALID")
        if item.cardiac_signal_source not in {
            None,
            "",
            "ECG",
            "PULSE",
            "OTHER",
            "UNKNOWN",
        }:
            invalid = True
            details.append("DICOM_SYNTAX_INVALID")
    # AMENDMENT V1.1 stack uniformity (Part 4.1): a stack whose ordered members
    # do not share ONE (bits_allocated, bits_stored, high_bit) tuple is a
    # series-level bit-depth refusal — Syntax=INVALID at the classification
    # seam (the adapter seam refuses with SERIES_STACK_INVALID).  The detail
    # code follows the Part 4 mapping: a bit-depth refusal is
    # DICOM_SYNTAX_INVALID, never a new reason code.
    if len({(item.bits_allocated, item.bits_stored, item.high_bit) for item in instances}) > 1:
        invalid = True
        details.append("DICOM_SYNTAX_INVALID")
    return (SyntaxState.INVALID if invalid else SyntaxState.VALID), tuple(dict.fromkeys(details))


def classify_series(
    instances: Sequence[DicomInstance],
    *,
    stack: StackAdmissionResult | None = None,
    skipped_instance_could_belong: bool = False,
) -> ClassificationResult:
    """Classify a series, retaining per-dimension results and safe evidence."""

    if not instances:
        return classify_boundary_states(
            ProtocolState.PROTOCOL_IDENTITY_INVALID,
            ContrastState.UNKNOWN_AGENT_ABSENT,
            PhaseState.UNKNOWN_NO_EVIDENCE,
            GeometryState.INVALID,
            SyntaxState.INVALID,
            evidence_fields=("Modality", "SOPClassUID", "ImageType"),
            detail_codes=(ClassificationReason.SERIES_INCOMPLETE.value,),
        )
    protocol, protocol_fields = _protocol_state(instances)
    contrast, contrast_fields = _contrast_state(instances)
    phase, phase_fields, phase_label = _phase_state(instances)
    syntax, syntax_details = _syntax_state(instances)
    if stack is None:
        stack = validate_stack(
            instances,
            skipped_instance_could_belong=skipped_instance_could_belong,
        )
    geometry = GeometryState.VALID if stack.admitted else GeometryState.INVALID
    geometry_details: tuple[str, ...] = ()
    if stack.detail_code is not None:
        geometry_details = (stack.detail_code.value,)
    result = classify_boundary_states(
        protocol,
        contrast,
        phase,
        geometry,
        syntax,
        evidence_fields=(*protocol_fields, *contrast_fields, *phase_fields),
        detail_codes=(*syntax_details, *geometry_details),
    )
    return ClassificationResult(
        protocol=result.protocol,
        contrast=result.contrast,
        phase=result.phase,
        outcome=result.outcome,
        visible_reason=result.visible_reason,
        provenance=result.provenance,
        classifier=result.classifier,
        version=result.version,
        phase_label=phase_label,
        detail_codes=result.detail_codes,
    )


class RuleTableClassifier:
    """Small dependency-free façade for the frozen rule table."""

    classifier = "rule-table"
    version = "1"

    def classify(
        self,
        instances: Sequence[DicomInstance] | DicomInstance,
        *,
        stack: StackAdmissionResult | None = None,
        skipped_instance_could_belong: bool = False,
    ) -> ClassificationResult:
        sequence = (instances,) if isinstance(instances, DicomInstance) else instances
        return classify_series(
            sequence,
            stack=stack,
            skipped_instance_could_belong=skipped_instance_could_belong,
        )


def classify(
    instances: Sequence[DicomInstance] | DicomInstance,
    *,
    stack: StackAdmissionResult | None = None,
    skipped_instance_could_belong: bool = False,
) -> ClassificationResult:
    return RuleTableClassifier().classify(
        instances,
        stack=stack,
        skipped_instance_could_belong=skipped_instance_could_belong,
    )


__all__ = [
    "BOUNDARY_GRID",
    "CARDIAC_KEYWORDS",
    "CLASSIFICATION_BOUNDARY_GRID",
    "COMPLETE_CLASSIFICATION_ORACLE",
    "ClassificationLabel",
    "ClassificationOutcome",
    "ClassificationProvenance",
    "ClassificationReason",
    "ClassificationResult",
    "ContrastState",
    "DimensionClassification",
    "DUPLICATE_POSITION_TOLERANCE_MM",
    "EXCLUDED_IMAGE_TYPE_COMPONENTS",
    "GeometryState",
    "IN_PLANE_DRIFT_TOLERANCE_MM",
    "IN_PLANE_SPACING_TOLERANCE_MM",
    "INTER_SLICE_GAP_TOLERANCE_MM",
    "INTER_SLICE_GAP_RELATIVE_TOLERANCE",
    "INTER_SLICE_GAP_ABSOLUTE_FLOOR_MM",
    "NON_CONTRAST_PHRASES",
    "ORIENTATION_TOLERANCE",
    "PARALLEL_DIRECTION_TOLERANCE",
    "PhaseState",
    "ProtocolState",
    "RuleTableClassifier",
    "StackAdmissionError",
    "StackAdmissionResult",
    "SyntaxState",
    "admit_stack",
    "classify",
    "classify_boundary_states",
    "classify_series",
    "classification_boundary_grid",
    "validate_stack",
]
