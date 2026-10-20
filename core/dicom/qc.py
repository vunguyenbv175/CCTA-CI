"""The closed QC attachment and consumer-gate seam for Imaging Core 001-A.

This is deliberately a placeholder.  It records no image-quality evidence
and performs no quality computation.  The deterministic constructor is the
only 001 producer and emits the exact evidence-free GOOD payload required by
the specification.
"""

from __future__ import annotations

import enum
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from core.enums import ArtifactStatus, Confidence, ProducerType, QualityGrade
from core.ids import new_ulid

QC_PAYLOAD_SCHEMA = "ccta.payload.qc_assessment.v1"
QC_ANATOMIC_IDENTIFIER_SCHEMA = "ccta.qc.anatomic_identifier.v1"
QC_FLAGS: tuple[str, ...] = (
    "CARDIAC_MOTION",
    "RESPIRATORY_MOTION",
    "STAIR_STEP",
    "POOR_CONTRAST",
    "EXCESSIVE_NOISE",
    "BLOOMING",
    "BEAM_HARDENING",
    "MISREGISTRATION",
    "INCOMPLETE_ANATOMY",
    "MODEL_DISAGREEMENT",
    "ABSTAINED",
)
QC_JUSTIFICATION_CODES: tuple[str, ...] = (
    "PLACEHOLDER_NO_QC_EVIDENCE",
    "RULE_ENGINE_EVIDENCE",
    "HUMAN_EVIDENCE",
    "MODEL_EVIDENCE",
)

_ULID = r"[0123456789ABCDEFGHJKMNPQRSTVWXYZ]{26}"
_SUBJECT_REF = re.compile(
    rf"\A(?P<kind>study_id|series_id|volume_id|mpr_artifact_id):(?P<id>{_ULID})\Z"
)
_PROVENANCE_REF = re.compile(rf"\Aprovenance:{_ULID}\Z")
_QC_ARTIFACT_REF = re.compile(rf"\Aqc_artifact:{_ULID}\Z")
_EVIDENCE_REF = re.compile(
    rf"\A(?:provenance:{_ULID}|source_manifest:[0-9a-f]{{64}}|"
    rf"volume_id:{_ULID}|mpr_artifact_id:{_ULID})\Z"
)
_SUBJECT_KIND_ORDER = {
    "study_id": 0,
    "series_id": 1,
    "volume_id": 2,
    "mpr_artifact_id": 3,
}


class QCScope(enum.StrEnum):
    STUDY = "STUDY"
    VESSEL = "VESSEL"
    REGION = "REGION"


QCProducer = ProducerType


class QCGateDecision(enum.StrEnum):
    PROCEED = "PROCEED"
    ABSTAIN = "ABSTAIN"


class QCReason(enum.StrEnum):
    QC_NOT_ASSESSED = "QC_NOT_ASSESSED"
    QC_LIMITED = "QC_LIMITED"
    QC_NOT_ANALYZABLE = "QC_NOT_ANALYZABLE"
    QC_CONFLICT = "QC_CONFLICT"


@dataclass(frozen=True, slots=True)
class QCAssessmentPayload:
    schema_version: str
    scope: QCScope
    subject_refs: tuple[str, ...]
    quality_grade: QualityGrade
    confidence: Confidence | None
    flags: tuple[str, ...]
    vessel_id: str | None
    region_id: str | None
    producer: ProducerType
    produced_by_ref: str
    supersedes_ref: str | None
    justification_code: str
    evidence_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "scope", QCScope(self.scope))
        object.__setattr__(self, "quality_grade", QualityGrade(self.quality_grade))
        if self.confidence is not None:
            object.__setattr__(self, "confidence", Confidence(self.confidence))
        object.__setattr__(self, "producer", ProducerType(self.producer))
        object.__setattr__(self, "subject_refs", tuple(self.subject_refs))
        object.__setattr__(self, "flags", tuple(self.flags))
        object.__setattr__(self, "evidence_refs", tuple(self.evidence_refs))
        if self.schema_version != QC_PAYLOAD_SCHEMA:
            raise ValueError("QC payload schema is closed")
        if not self.subject_refs or len(set(self.subject_refs)) != len(self.subject_refs):
            raise ValueError("QC subject_refs must be nonempty and unique")
        parsed_refs = tuple(_parse_subject_ref(ref) for ref in self.subject_refs)
        if any(parsed is None for parsed in parsed_refs):
            raise ValueError("QC subject_refs contain an invalid canonical ref")
        canonical_refs = tuple(
            sorted(
                self.subject_refs,
                key=lambda ref: (
                    _SUBJECT_KIND_ORDER[_parse_subject_ref(ref)[0]],  # type: ignore[index]
                    _parse_subject_ref(ref)[1],  # type: ignore[index]
                ),
            )
        )
        if self.subject_refs != canonical_refs:
            raise ValueError("QC subject_refs are not in canonical order")
        if len(set(self.flags)) != len(self.flags) or any(
            flag not in QC_FLAGS for flag in self.flags
        ):
            raise ValueError("QC flags are outside the closed vocabulary")
        if tuple(flag for flag in QC_FLAGS if flag in self.flags) != self.flags:
            raise ValueError("QC flags are not in canonical order")
        if (
            self.confidence == Confidence.NOT_ANALYZABLE
            and self.quality_grade != QualityGrade.NOT_ANALYZABLE
        ):
            raise ValueError("NOT_ANALYZABLE confidence requires NOT_ANALYZABLE grade")
        if self.scope == QCScope.STUDY and (
            self.vessel_id is not None or self.region_id is not None
        ):
            raise ValueError("STUDY QC cannot carry vessel_id or region_id")
        if self.scope == QCScope.STUDY and sum(
            parsed[0] == "study_id" for parsed in parsed_refs if parsed is not None
        ) != 1:
            raise ValueError("STUDY QC requires exactly one study_id subject ref")
        if self.scope in {QCScope.VESSEL, QCScope.REGION}:
            raise ValueError("001 has no admitted vessel or region identifiers")
        if self.producer not in {ProducerType.DETERMINISTIC_ENGINE, ProducerType.HUMAN}:
            raise ValueError("QC producer is outside the closed 001 vocabulary")
        if _PROVENANCE_REF.fullmatch(self.produced_by_ref) is None:
            raise ValueError("produced_by_ref must be provenance:<ULID>")
        if self.supersedes_ref is not None and _QC_ARTIFACT_REF.fullmatch(
            self.supersedes_ref
        ) is None:
            raise ValueError("supersedes_ref must be qc_artifact:<ULID>")
        if self.justification_code not in QC_JUSTIFICATION_CODES:
            raise ValueError("QC justification code is outside the closed vocabulary")
        if self.justification_code == "PLACEHOLDER_NO_QC_EVIDENCE" and (
            self.quality_grade != QualityGrade.GOOD
            or self.producer != ProducerType.DETERMINISTIC_ENGINE
            or self.confidence is not None
            or self.flags
            or self.evidence_refs
        ):
            raise ValueError("placeholder QC must be GOOD, deterministic, and evidence-free")
        if not self.evidence_refs and self.justification_code != "PLACEHOLDER_NO_QC_EVIDENCE":
            raise ValueError("non-placeholder QC requires evidence_refs")
        if len(set(self.evidence_refs)) != len(self.evidence_refs):
            raise ValueError("QC evidence_refs must be unique")
        if any(_EVIDENCE_REF.fullmatch(ref) is None for ref in self.evidence_refs):
            raise ValueError("QC evidence_refs contain an invalid canonical ref")


@dataclass(frozen=True, slots=True)
class QCAssessmentArtifact:
    """A closed seam artifact used by the 001 consumer gate."""

    artifact_ref: str
    payload: QCAssessmentPayload
    producer: ProducerType
    status: str | ArtifactStatus = "ACTIVE"
    kind: str = "QC_ASSESSMENT"
    payload_schema: str = QC_PAYLOAD_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "producer", ProducerType(self.producer))
        status = self.status.value if isinstance(self.status, ArtifactStatus) else self.status
        object.__setattr__(self, "status", status)
        if self.payload.producer != self.producer:
            raise ValueError("QC producer must agree with payload producer")
        if _QC_ARTIFACT_REF.fullmatch(self.artifact_ref) is None:
            raise ValueError("QC artifact ref must be qc_artifact:<ULID>")
        if self.kind != "QC_ASSESSMENT":
            raise ValueError("QC artifact kind is closed")
        if self.payload_schema != QC_PAYLOAD_SCHEMA:
            raise ValueError("QC artifact payload schema is closed")
        if self.status not in {"ACTIVE", "SUPERSEDED", "INVALIDATED", "STALE"}:
            raise ValueError("QC status is outside the closed lifecycle")

    @property
    def quality_grade(self) -> QualityGrade:
        return self.payload.quality_grade

    @property
    def subject_refs(self) -> tuple[str, ...]:
        return self.payload.subject_refs


@dataclass(frozen=True, slots=True)
class QCGateResult:
    decision: QCGateDecision
    reason_code: QCReason | None
    assessment_refs: tuple[str, ...] = ()

    @property
    def proceed(self) -> bool:
        return self.decision == QCGateDecision.PROCEED

    @property
    def abstained(self) -> bool:
        return not self.proceed

    @property
    def reason(self) -> QCReason | None:
        return self.reason_code


def _is_typed_ref(value: str) -> bool:
    return _parse_subject_ref(value) is not None


def _parse_subject_ref(value: object) -> tuple[str, str] | None:
    if not isinstance(value, str):
        return None
    match = _SUBJECT_REF.fullmatch(value)
    return None if match is None else (match.group("kind"), match.group("id"))


def create_qc_assessment(
    *,
    subject_refs: tuple[str, ...],
    provenance_ref: str | None = None,
    scope: QCScope | str = QCScope.STUDY,
    artifact_ref: str | None = None,
) -> QCAssessmentArtifact:
    """Create the evidence-free deterministic 001 QC placeholder."""

    resolved_scope = QCScope(scope)
    resolved_provenance = provenance_ref or f"provenance:{new_ulid()}"
    payload = QCAssessmentPayload(
        schema_version=QC_PAYLOAD_SCHEMA,
        scope=resolved_scope,
        subject_refs=tuple(subject_refs),
        quality_grade=QualityGrade.GOOD,
        confidence=None,
        flags=(),
        vessel_id=None,
        region_id=None,
        producer=ProducerType.DETERMINISTIC_ENGINE,
        produced_by_ref=resolved_provenance,
        supersedes_ref=None,
        justification_code="PLACEHOLDER_NO_QC_EVIDENCE",
        evidence_refs=(),
    )
    return QCAssessmentArtifact(
        artifact_ref=artifact_ref or f"qc_artifact:{new_ulid()}",
        payload=payload,
        producer=ProducerType.DETERMINISTIC_ENGINE,
    )


def _eligible(assessment: QCAssessmentArtifact) -> bool:
    try:
        return (
            assessment.status == "ACTIVE"
            and assessment.kind == "QC_ASSESSMENT"
            and assessment.payload_schema == QC_PAYLOAD_SCHEMA
            and assessment.payload.schema_version == QC_PAYLOAD_SCHEMA
            and assessment.payload.producer == assessment.producer
        )
    except (AttributeError, TypeError, ValueError):
        return False


def _covers(assessment: QCAssessmentArtifact, subject_ref: str) -> bool:
    return subject_ref in assessment.payload.subject_refs


def _lineage_covers(
    assessment: QCAssessmentArtifact,
    subject_ref: str,
    subject_lineage: Mapping[str, object],
) -> bool:
    """Resolve parent coverage from child -> ancestor canonical refs."""

    if _covers(assessment, subject_ref):
        return True
    def refs_from(value: object) -> tuple[str, ...]:
        if isinstance(value, str):
            return (value,)
        if isinstance(value, Mapping):
            refs: list[str] = []
            for key in (
                "ancestor_refs",
                "ancestors",
                "study_ref",
                "study_id",
                "source_series_ref",
                "source_series_refs",
                "series_ref",
                "series_refs",
                "source_volume_ref",
                "source_volume_refs",
                "volume_ref",
                "volume_id",
            ):
                if key in value:
                    refs.extend(refs_from(value[key]))
            return tuple(ref for ref in refs if _parse_subject_ref(ref) is not None)
        if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
            return tuple(ref for item in value for ref in refs_from(item))
        return ()

    def ancestors_for(ref: str) -> tuple[str, ...]:
        value = subject_lineage.get(ref)
        if value is not None:
            return refs_from(value)
        return ()

    ancestors: set[str] = set()
    pending = list(ancestors_for(subject_ref))
    while pending:
        ancestor = pending.pop()
        if ancestor in ancestors or ancestor == subject_ref:
            continue
        ancestors.add(ancestor)
        pending.extend(ancestors_for(ancestor))
    return any(ref in ancestors for ref in assessment.payload.subject_refs)


def qc_consumer_gate(
    consumed_subject_refs: tuple[str, ...],
    assessments: Sequence[QCAssessmentArtifact] = (),
    *,
    consumer: str = "",
    subject_lineage: Mapping[str, object] | None = None,
    lineage: Mapping[str, object] | None = None,
) -> QCGateResult:
    """Resolve closed QC coverage before a consumer may read image pixels."""

    if not isinstance(consumer, str) or not consumer.strip():
        raise ValueError("consumer QC gate requires a consumer identifier")
    if not consumed_subject_refs or len(set(consumed_subject_refs)) != len(consumed_subject_refs):
        raise ValueError("consumer QC gate requires nonempty deduplicated subject_refs")
    eligible = tuple(assessment for assessment in assessments if _eligible(assessment))
    resolved_lineage: dict[str, object] = {}
    if subject_lineage:
        resolved_lineage.update(subject_lineage)
    if lineage:
        resolved_lineage.update(lineage)
    covering: dict[str, list[QCAssessmentArtifact]] = {ref: [] for ref in consumed_subject_refs}
    for subject_ref in consumed_subject_refs:
        covering[subject_ref] = [
            assessment
            for assessment in eligible
            if _lineage_covers(assessment, subject_ref, resolved_lineage)
        ]
        if not covering[subject_ref]:
            return QCGateResult(QCGateDecision.ABSTAIN, QCReason.QC_NOT_ASSESSED)

    all_covering = {
        record.artifact_ref: record
        for records in covering.values()
        for record in records
    }
    superseded_refs = {
        record.payload.supersedes_ref
        for record in all_covering.values()
        if record.payload.supersedes_ref in all_covering
    }
    effective_covering = {
        subject_ref: tuple(
            record
            for record in records
            if record.artifact_ref not in superseded_refs
        )
        for subject_ref, records in covering.items()
    }
    effective = {
        record.artifact_ref: record
        for records in effective_covering.values()
        for record in records
    }
    for records in effective_covering.values():
        if len({record.artifact_ref for record in records}) > 1:
            refs = tuple(sorted(record.artifact_ref for record in records))
            return QCGateResult(QCGateDecision.ABSTAIN, QCReason.QC_CONFLICT, refs)

    unique_effective = effective
    ordered = tuple(sorted(unique_effective))
    if any(
        record.payload.quality_grade == QualityGrade.NOT_ANALYZABLE
        for record in unique_effective.values()
    ):
        return QCGateResult(QCGateDecision.ABSTAIN, QCReason.QC_NOT_ANALYZABLE, ordered)
    if any(
        record.payload.quality_grade == QualityGrade.LIMITED
        for record in unique_effective.values()
    ):
        return QCGateResult(QCGateDecision.ABSTAIN, QCReason.QC_LIMITED, ordered)
    if not all(
        record.payload.quality_grade in {QualityGrade.GOOD, QualityGrade.ACCEPTABLE}
        for record in unique_effective.values()
    ):
        return QCGateResult(QCGateDecision.ABSTAIN, QCReason.QC_NOT_ASSESSED, ordered)
    return QCGateResult(QCGateDecision.PROCEED, None, ordered)


def check_qc_gate(
    consumed_subject_refs: tuple[str, ...],
    assessments: Sequence[QCAssessmentArtifact] = (),
    *,
    consumer: str = "",
    subject_lineage: Mapping[str, object] | None = None,
    lineage: Mapping[str, object] | None = None,
) -> QCGateResult:
    return qc_consumer_gate(
        consumed_subject_refs,
        assessments,
        consumer=consumer,
        subject_lineage=subject_lineage,
        lineage=lineage,
    )


make_qc_assessment = create_qc_assessment


__all__ = [
    "QC_ANATOMIC_IDENTIFIER_SCHEMA",
    "QC_FLAGS",
    "QC_JUSTIFICATION_CODES",
    "QC_PAYLOAD_SCHEMA",
    "QCAssessmentArtifact",
    "QCAssessmentPayload",
    "QCGateDecision",
    "QCGateResult",
    "QCProducer",
    "QCReason",
    "QCScope",
    "check_qc_gate",
    "create_qc_assessment",
    "make_qc_assessment",
    "qc_consumer_gate",
]
