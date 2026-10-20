"""DET-local real-QC eligibility and fail-closed consumer policy."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

from core.artifacts import ArtifactKind, DerivedArtifact
from core.dicom.qc import (
    QCAssessmentArtifact,
    QCAssessmentPayload,
    QCGateDecision,
    QCGateResult,
    QCReason,
    QCScope,
    qc_consumer_gate,
)
from core.enums import ArtifactStatus, Confidence, ProducerType, QualityGrade
from core.provenance import verify_chain

from .constants import PAYLOAD_SCHEMA
from .factory import validate_result_bundle
from .types import (
    QCPolicyDiagnostic,
    QCResultBundle,
    RealQCEligibilityResult,
)


@dataclass(frozen=True, slots=True)
class _Candidate:
    assessment: QCAssessmentArtifact | None
    artifact: DerivedArtifact | None
    bundle: QCResultBundle | None
    raw: object
    ref: str | None
    status: str | None
    relation_valid: bool = True


def _status(value: object) -> str:
    return value.value if isinstance(value, ArtifactStatus) else str(value)


def _diag(code: str, candidate: _Candidate | None = None, detail: str = "") -> QCPolicyDiagnostic:
    return QCPolicyDiagnostic(
        code=code, artifact_ref=None if candidate is None else candidate.ref, detail=detail
    )


def _mapping_payload(candidate: Mapping[str, object]) -> Mapping[str, object] | None:
    payload = candidate.get("payload")
    if isinstance(payload, Mapping):
        return payload
    # A persisted candidate may itself be the thirteen-key payload.
    if "schema_version" in candidate and "quality_grade" in candidate:
        return candidate
    return None


def _candidate_from_raw(raw: object) -> tuple[_Candidate, tuple[QCPolicyDiagnostic, ...]]:
    diagnostics: list[QCPolicyDiagnostic] = []
    if isinstance(raw, QCResultBundle):
        return (
            _Candidate(
                assessment=raw.assessment,
                artifact=raw.artifact,
                bundle=raw,
                raw=raw,
                ref=raw.assessment.artifact_ref,
                status=_status(raw.assessment.status),
            ),
            (),
        )
    if isinstance(raw, QCAssessmentArtifact):
        return (
            _Candidate(
                assessment=raw,
                artifact=None,
                bundle=None,
                raw=raw,
                ref=raw.artifact_ref,
                status=_status(raw.status),
            ),
            (),
        )
    if isinstance(raw, Mapping):
        payload = _mapping_payload(raw)
        ref = raw.get("artifact_ref") if isinstance(raw.get("artifact_ref"), str) else None
        status_value = raw.get("status")
        status = _status(status_value) if status_value is not None else None
        if payload is not None and payload.get("schema_version") != PAYLOAD_SCHEMA:
            schema = payload.get("schema_version")
            code = (
                "QC_SCHEMA_INCOMPATIBLE"
                if schema == "ccta.payload.qc_assessment.v2"
                else "QC_SCHEMA_INVALID"
            )
            return (
                _Candidate(None, None, None, raw, ref, status, relation_valid=False),
                (
                    _diag(
                        code,
                        _Candidate(None, None, None, raw, ref, status),
                        "observed schema is not admitted by this v1 policy",
                    ),
                ),
            )
        if payload is None:
            return (
                _Candidate(None, None, None, raw, ref, status, relation_valid=False),
                (
                    _diag(
                        "QC_SCHEMA_INVALID",
                        _Candidate(None, None, None, raw, ref, status),
                        "candidate payload is not an object",
                    ),
                ),
            )
        try:
            assessment = QCAssessmentArtifact(
                artifact_ref=cast(str, raw.get("artifact_ref", "")),
                payload=QCAssessmentPayload(
                    schema_version=cast(str, payload["schema_version"]),
                    scope=QCScope(cast(str, payload["scope"])),
                    subject_refs=tuple(cast(Sequence[str], payload["subject_refs"])),
                    quality_grade=QualityGrade(cast(str, payload["quality_grade"])),
                    confidence=(
                        None
                        if payload.get("confidence") is None
                        else Confidence(cast(str, payload["confidence"]))
                    ),
                    flags=tuple(cast(Sequence[str], payload["flags"])),
                    vessel_id=cast(str | None, payload["vessel_id"]),
                    region_id=cast(str | None, payload["region_id"]),
                    producer=ProducerType(cast(str, payload["producer"])),
                    produced_by_ref=cast(str, payload["produced_by_ref"]),
                    supersedes_ref=cast(str | None, payload["supersedes_ref"]),
                    justification_code=cast(str, payload["justification_code"]),
                    evidence_refs=tuple(cast(Sequence[str], payload["evidence_refs"])),
                ),
                producer=ProducerType(cast(str, raw.get("producer", payload["producer"]))),
                status=cast(str, raw.get("status", "ACTIVE")),
            )
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            return (
                _Candidate(None, None, None, raw, ref, status, relation_valid=False),
                (
                    _diag(
                        "QC_SCHEMA_INVALID",
                        _Candidate(None, None, None, raw, ref, status),
                        str(exc),
                    ),
                ),
            )
        return (
            _Candidate(
                assessment, None, None, raw, assessment.artifact_ref, _status(assessment.status)
            ),
            tuple(diagnostics),
        )
    return (
        _Candidate(None, None, None, raw, None, None, relation_valid=False),
        (_diag("QC_SCHEMA_INVALID", detail="candidate type is opaque but not a QC record"),),
    )


def _validate_v1_candidate(candidate: _Candidate) -> tuple[QCPolicyDiagnostic, ...]:
    assessment = candidate.assessment
    if assessment is None:
        return (_diag("QC_SCHEMA_INVALID", candidate),)
    if candidate.status != "ACTIVE":
        return (_diag("QC_RECORD_INELIGIBLE", candidate, f"status={candidate.status}"),)
    if assessment.kind != "QC_ASSESSMENT" or assessment.payload_schema != PAYLOAD_SCHEMA:
        return (_diag("QC_SCHEMA_INVALID", candidate, "outer QC kind/schema mismatch"),)
    if assessment.payload.schema_version != PAYLOAD_SCHEMA:
        return (_diag("QC_SCHEMA_INVALID", candidate, "payload schema mismatch"),)
    if (
        assessment.producer != ProducerType.DETERMINISTIC_ENGINE
        or assessment.payload.producer != ProducerType.DETERMINISTIC_ENGINE
    ):
        return (_diag("QC_PRODUCER_INVALID", candidate),)
    if assessment.payload.justification_code == "PLACEHOLDER_NO_QC_EVIDENCE":
        return (
            _diag(
                "QC_LEGACY_PLACEHOLDER",
                candidate,
                "historical evidence-free GOOD is not real-QC coverage",
            ),
        )
    if (
        assessment.payload.justification_code != "RULE_ENGINE_EVIDENCE"
        or not assessment.payload.evidence_refs
    ):
        return (
            _diag(
                "QC_PROVENANCE_INVALID",
                candidate,
                "REAL v1 requires RULE_ENGINE_EVIDENCE and evidence_refs",
            ),
        )
    if candidate.bundle is None or candidate.artifact is None:
        return (
            _diag(
                "QC_PROVENANCE_INVALID",
                candidate,
                "real-QC policy requires the complete derived/provenance bundle",
            ),
        )
    try:
        validate_result_bundle(candidate.bundle)
    except (TypeError, ValueError) as exc:
        return (_diag("QC_PROVENANCE_INVALID", candidate, str(exc)),)
    artifact = candidate.artifact
    if artifact.status != ArtifactStatus.ACTIVE or artifact.kind != ArtifactKind.QC_ASSESSMENT:
        return (
            _diag(
                "QC_RECORD_INELIGIBLE",
                candidate,
                "outer DerivedArtifact is not active QC_ASSESSMENT",
            ),
        )
    if artifact.provenance_ref != assessment.payload.produced_by_ref:
        return (_diag("QC_PROVENANCE_INVALID", candidate, "outer and payload provenance differ"),)
    records = candidate.bundle.provenance_records
    ok, problems = verify_chain(records)
    if not ok:
        return (_diag("QC_PROVENANCE_INVALID", candidate, "; ".join(problems)),)
    return ()


def _refs_from(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Mapping):
        values: list[str] = []
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
                values.extend(_refs_from(value[key]))
        return tuple(values)
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        return tuple(item for part in value for item in _refs_from(part))
    return ()


def _covers(candidate: QCAssessmentArtifact, requested: str, lineage: Mapping[str, object]) -> bool:
    if requested in candidate.payload.subject_refs:
        return True
    found: set[str] = set()
    pending = list(_refs_from(lineage.get(requested)))
    while pending:
        item = pending.pop()
        if item in found or item == requested:
            continue
        found.add(item)
        pending.extend(_refs_from(lineage.get(item)))
    return any(item in found for item in candidate.payload.subject_refs)


def _mapping_covers(
    payload: Mapping[str, object], requested: str, lineage: Mapping[str, object]
) -> bool:
    """Apply the same descendant coverage rule to an opaque candidate.

    Unknown schemas cannot be deserialized into the closed assessment DTO, but
    they must still block a fallback when their declared subject covers the
    requested descendant.  Keeping this resolver local to the policy avoids
    treating an incompatible record as silently absent.
    """

    raw_subjects = payload.get("subject_refs", ())
    if isinstance(raw_subjects, str) or not isinstance(raw_subjects, Sequence):
        return False
    subjects = tuple(item for item in raw_subjects if isinstance(item, str))
    if requested in subjects:
        return True
    found: set[str] = set()
    pending = list(_refs_from(lineage.get(requested)))
    while pending:
        item = pending.pop()
        if item in found or item == requested:
            continue
        found.add(item)
        pending.extend(_refs_from(lineage.get(item)))
    return any(item in found for item in subjects)


def _coverage_key(assessment: QCAssessmentArtifact) -> tuple[object, ...]:
    payload = assessment.payload
    return (payload.scope, payload.subject_refs, payload.vessel_id, payload.region_id)


def _relation_valid(successor: _Candidate, predecessor: _Candidate | None) -> bool:
    if successor.assessment is None or successor.assessment.payload.supersedes_ref is None:
        return True
    if predecessor is None or predecessor.assessment is None:
        return False
    return _coverage_key(successor.assessment) == _coverage_key(predecessor.assessment)


def _select_effective(
    candidates: Sequence[_Candidate],
    consumed: tuple[str, ...],
    lineage: Mapping[str, object],
) -> tuple[tuple[QCAssessmentArtifact, ...], tuple[QCPolicyDiagnostic, ...], tuple[str, ...]]:
    diagnostics: list[QCPolicyDiagnostic] = []
    by_ref = {
        candidate.ref: candidate
        for candidate in candidates
        if candidate.ref is not None and candidate.assessment is not None
    }
    eligible = [
        candidate
        for candidate in candidates
        if candidate.assessment is not None and candidate.status == "ACTIVE"
    ]
    invalid_relation_refs: set[str] = set()
    for candidate in eligible:
        assessment = candidate.assessment
        if assessment is None:
            continue
        predecessor_ref = assessment.payload.supersedes_ref
        if predecessor_ref is not None:
            predecessor = by_ref.get(predecessor_ref)
            if not _relation_valid(candidate, predecessor):
                invalid_relation_refs.add(cast(str, candidate.ref))
                diagnostics.append(
                    _diag(
                        "QC_SUPERSESSION_INVALID",
                        candidate,
                        "predecessor is missing or coverage differs",
                    )
                )
    # A directed successor map is enough to detect cycles in the explicit
    # immutable graph; timestamps and insertion order are intentionally unused.
    for candidate in eligible:
        if candidate.ref in invalid_relation_refs or candidate.assessment is None:
            continue
        seen: set[str] = set()
        current = candidate
        while (
            current.assessment is not None and current.assessment.payload.supersedes_ref is not None
        ):
            ref = current.assessment.payload.supersedes_ref
            if ref in seen:
                invalid_relation_refs.add(cast(str, candidate.ref))
                diagnostics.append(
                    _diag("QC_SUPERSESSION_INVALID", candidate, "supersession cycle")
                )
                break
            seen.add(ref)
            next_candidate = by_ref.get(ref)
            if next_candidate is None or next_candidate.assessment is None:
                break
            current = next_candidate
    effective: list[QCAssessmentArtifact] = []
    missing: list[str] = []
    for requested in consumed:
        covering = [
            candidate
            for candidate in eligible
            if candidate.ref not in invalid_relation_refs
            and candidate.assessment is not None
            and _covers(candidate.assessment, requested, lineage)
        ]
        if not covering:
            missing.append(requested)
            continue
        maxima = [
            candidate
            for candidate in covering
            if not any(
                other is not candidate
                and other.assessment is not None
                and other.assessment.payload.supersedes_ref == candidate.ref
                and _relation_valid(other, candidate)
                for other in covering
            )
        ]
        refs = tuple(sorted(cast(str, item.ref) for item in maxima))
        if len(maxima) > 1:
            diagnostics.append(
                _diag("QC_CONFLICT", detail="incomparable effective maxima: " + ",".join(refs))
            )
        effective.extend(item.assessment for item in maxima if item.assessment is not None)
    unique = tuple(
        {item.artifact_ref: item for item in effective}[ref]
        for ref in sorted({item.artifact_ref for item in effective})
    )
    return unique, tuple(diagnostics), tuple(missing)


class RealQCEligibilityPolicy:
    """DET-local policy wrapper delegating eligible records to qc_consumer_gate."""

    def __init__(self, consumer_schema: str = "v1") -> None:
        if consumer_schema not in {"v1", "v2"}:
            raise ValueError("consumer_schema must be v1 or v2")
        self.consumer_schema = consumer_schema

    def evaluate(
        self,
        consumed_subject_refs: tuple[str, ...],
        candidates: Sequence[object] = (),
        *,
        consumer: str,
        subject_lineage: Mapping[str, object] | None = None,
        lineage: Mapping[str, object] | None = None,
    ) -> RealQCEligibilityResult:
        if not consumed_subject_refs or len(set(consumed_subject_refs)) != len(
            consumed_subject_refs
        ):
            raise ValueError("real-QC policy requires nonempty deduplicated consumed refs")
        resolved_lineage: dict[str, object] = {}
        if subject_lineage:
            resolved_lineage.update(subject_lineage)
        if lineage:
            resolved_lineage.update(lineage)
        parsed: list[_Candidate] = []
        diagnostics: list[QCPolicyDiagnostic] = []
        for raw in candidates:
            candidate, parse_diagnostics = _candidate_from_raw(raw)
            parsed.append(candidate)
            diagnostics.extend(parse_diagnostics)
            if candidate.assessment is not None:
                diagnostics.extend(_validate_v1_candidate(candidate))
        valid = [
            candidate
            for candidate in parsed
            if candidate.assessment is not None and not _validate_v1_candidate(candidate)
        ]
        # A visible incompatible/unknown candidate covering the request blocks
        # fallback to an older schema.  It is deliberately not passed to the
        # historical gate.
        incompatible_cover = False
        for diagnostic in diagnostics:
            if diagnostic.code not in {"QC_SCHEMA_INCOMPATIBLE", "QC_SCHEMA_INVALID"}:
                continue
            diagnosed_candidate = next(
                (item for item in parsed if item.ref == diagnostic.artifact_ref), None
            )
            if diagnosed_candidate is not None and isinstance(diagnosed_candidate.raw, Mapping):
                payload = _mapping_payload(diagnosed_candidate.raw)
                if payload is not None and any(
                    _mapping_covers(payload, requested, resolved_lineage)
                    for requested in consumed_subject_refs
                ):
                    incompatible_cover = True
        effective, selection_diagnostics, missing = _select_effective(
            valid, consumed_subject_refs, resolved_lineage
        )
        diagnostics.extend(selection_diagnostics)
        if incompatible_cover:
            gate = QCGateResult(QCGateDecision.ABSTAIN, QCReason.QC_NOT_ASSESSED, ())
        elif missing:
            gate = QCGateResult(QCGateDecision.ABSTAIN, QCReason.QC_NOT_ASSESSED, ())
        else:
            # This is the unchanged closed gate; the DET-local policy owns
            # eligibility and graph resolution, while grade precedence stays
            # in the historical seam.
            gate = qc_consumer_gate(
                consumed_subject_refs,
                effective,
                consumer=consumer,
                subject_lineage=resolved_lineage,
            )
        return RealQCEligibilityResult(
            gate_result=gate, diagnostics=tuple(diagnostics), eligible_assessments=effective
        )

    gate = evaluate
    check = evaluate


def real_qc_consumer_gate(
    consumed_subject_refs: tuple[str, ...],
    candidates: Sequence[object] = (),
    *,
    consumer: str,
    subject_lineage: Mapping[str, object] | None = None,
    lineage: Mapping[str, object] | None = None,
    consumer_schema: str = "v1",
) -> RealQCEligibilityResult:
    return RealQCEligibilityPolicy(consumer_schema).evaluate(
        consumed_subject_refs,
        candidates,
        consumer=consumer,
        subject_lineage=subject_lineage,
        lineage=lineage,
    )


qc_real_consumer_gate = real_qc_consumer_gate
qc_gate = real_qc_consumer_gate
RealQCPolicy = RealQCEligibilityPolicy


__all__ = [
    "RealQCEligibilityPolicy",
    "RealQCPolicy",
    "qc_gate",
    "qc_real_consumer_gate",
    "real_qc_consumer_gate",
]
