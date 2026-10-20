"""REAL v1 QC artifact factory and result-bundle validation."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

from core.artifacts import (
    ClaimBinding,
    DerivedArtifact,
    DerivedArtifactStore,
    GeometryBase,
    Lineage,
    PayloadRef,
    Producer,
    QualityBlock,
    ReviewBlock,
    create_derived_artifact,
)
from core.dicom.instance import DicomInstance
from core.dicom.qc import (
    QCAssessmentArtifact,
    QCAssessmentPayload,
    QCScope,
)
from core.digest import canonical_digest, is_sha256_hex
from core.enums import (
    ActorType,
    ArtifactKind,
    ArtifactStatus,
    ProducerType,
    ProvenanceEventType,
    QualityGrade,
    ReleaseLane,
    ReviewState,
)
from core.events import utc_now_iso
from core.ids import new_ulid
from core.provenance import (
    Actor,
    BlobRef,
    Environment,
    InputRef,
    NotApplicable,
    ProvenanceRecord,
    create_provenance,
    verify_chain,
)
from core.volume import Volume

from .algorithms import evaluate_all_families, validate_qc_input
from .constants import (
    ALGORITHM_VERSION,
    ASSESSMENT_ALGORITHM_ID,
    FAMILY_ALGORITHM_IDS,
    FAMILY_METRIC_NAMES,
    FIRST_NINE_FLAGS,
    PAYLOAD_SCHEMA,
    QC_FLAGS,
    QC_PREPROCESSING,
    QCAbstentionCode,
    QCOutcome,
    assessment_parameter_digest,
    assessment_parameter_manifest,
    family_parameter_digest,
)
from .pair import validate_registered_pair_input
from .types import (
    CoverageEnvelope,
    FamilyResult,
    QCInput,
    QCInputError,
    QCInputRefusal,
    QCResultBundle,
    QCValidationError,
    RegisteredPairInput,
)

_ULID = re.compile(r"^[0123456789ABCDEFGHJKMNPQRSTVWXYZ]{26}$")
_EVIDENCE_KEYS = frozenset(
    {
        "algorithm_id",
        "algorithm_version",
        "flag",
        "metric_name",
        "metric_value",
        "outcome",
        "abstention_code",
        "parameter_digest",
        "source_manifest_sha256",
    }
)
_DEFAULT_CLAIM = ClaimBinding(
    claim_id="CLM-QC-001",
    claim_version="v0-draft",
    population_ref="QC_001-D:qc-synthetic-v1",
    metric_refs=("QC_001-D", "QC_001-GATE"),
    snapshot_ref=None,
)
_ZERO_COMMIT = "0" * 40
_DEFAULT_SOFTWARE = "ccta-qc-v1"
F2_5_EVIDENCE_WORDING = (
    "standards-conformant synthetic DICOM fixtures; explicitly NOT a real-scanner "
    "coverage claim; real-scanner validation = separate later obligation A1."
)


def _valid_ulid(value: object) -> bool:
    return isinstance(value, str) and _ULID.fullmatch(value) is not None


def _study_subject(study_ref: str) -> str:
    suffix = study_ref.split(":", 1)[1] if study_ref.startswith("study_id:") else study_ref
    if not _valid_ulid(suffix):
        raise QCValidationError("QC v1 study subject must be study_id:<ULID>")
    return f"study_id:{suffix}"


def _volume_subject(volume: Volume) -> str:
    if not _valid_ulid(volume.volume_id):
        raise QCValidationError("QC v1 volume evidence requires volume_id:<ULID>")
    return f"volume_id:{volume.volume_id}"


def _provenance_ref(provenance_id: str) -> str:
    if not _valid_ulid(provenance_id):
        raise QCValidationError("QC real-QC provenance ids must be ULIDs")
    return f"provenance:{provenance_id}"


def _canonical_instance_inputs(context: Any) -> tuple[InputRef, ...]:
    if not context.instances:
        raise QCInputError(
            QCAbstentionCode.QC_SOURCE_IDENTITY_INVALID,
            "normal QC requires ordered source instances",
        )
    values: list[InputRef] = [
        InputRef(
            ref=f"source_manifest:{context.source_manifest_sha256}",
            kind="dataset",
            sha256_at_use=context.source_manifest_sha256,
        )
    ]
    values.extend(
        InputRef(ref=sop, kind="instance", sha256_at_use=digest)
        for sop, digest in context.instance_digests
    )
    values.append(
        InputRef(
            ref=_volume_subject(context.volume),
            kind="volume",
            sha256_at_use=context.volume.blob_ref.sha256,
        )
    )
    frame_digest = canonical_digest(context.frame)
    values.append(
        InputRef(
            ref=f"frame_spec:{frame_digest}",
            kind="artifact",
            sha256_at_use=frame_digest,
        )
    )
    transform = context.source.transform
    if transform is None or not isinstance(context.source.transform_ref, str):
        raise QCInputError(
            QCAbstentionCode.QC_GEOMETRY_INVALID,
            "normal QC requires the supplied versioned transform object and ref",
        )
    transform_digest = canonical_digest(transform)
    if context.source.transform_sha256 != transform_digest:
        raise QCInputError(
            QCAbstentionCode.QC_INPUT_SURFACE_INVALID,
            "versioned transform digest disagrees with supplied transform",
        )
    values.append(
        InputRef(
            ref=context.source.transform_ref,
            kind="artifact",
            sha256_at_use=transform_digest,
        )
    )
    return tuple(values)


def _root_provenance(
    *,
    provenance_id: str,
    subject_ref: str,
    inputs: tuple[InputRef, ...],
    at: str,
    created_by: str,
    code_commit_sha: str,
) -> ProvenanceRecord:
    return create_provenance(
        subject_ref=subject_ref,
        event_type=ProvenanceEventType.DERIVED,
        actor=Actor(actor_type=ActorType.ENGINE, actor_id="qc-supplied-source"),
        inputs=inputs,
        preprocessing="QC_SUPPLIED_CANONICAL_SOURCE_V1",
        parameters_ref=BlobRef(blob_ref=f"qc-source:{subject_ref}", sha256=inputs[0].sha256_at_use),
        human_edit=NotApplicable(value=None, reason="supplied source has no human edit"),
        upstream_refs=NotApplicable(value=None, reason="supplied source is a QC root"),
        environment=Environment(
            software_version=_DEFAULT_SOFTWARE, code_commit_sha=code_commit_sha
        ),
        model_record_ref=None,
        created_by=created_by,
        at=at,
        provenance_id=provenance_id,
    )


def _ensure_upstream_records(
    context: Any,
    supplied: Sequence[ProvenanceRecord],
    *,
    pair: RegisteredPairInput | None,
    at: str,
    created_by: str,
    code_commit_sha: str,
) -> tuple[tuple[ProvenanceRecord, ...], tuple[str, ...]]:
    if any(not isinstance(record, ProvenanceRecord) for record in supplied):
        raise QCInputError(
            QCAbstentionCode.QC_SOURCE_IDENTITY_INVALID,
            "supplied provenance_records must contain ProvenanceRecord objects",
        )
    records = list(supplied)
    by_id = {record.provenance_id: record for record in records}
    if len(by_id) != len(records):
        raise QCInputError(
            QCAbstentionCode.QC_SOURCE_IDENTITY_INVALID,
            "supplied provenance_records contain duplicate ids",
        )
    upstreams: list[str] = []

    if pair is not None:
        validate_registered_pair_input(pair)
        for record in pair.provenance_records:
            existing = by_id.get(record.provenance_id)
            if existing is not None and existing != record:
                raise QCInputError(
                    QCAbstentionCode.QC_REGISTRATION_INVALID,
                    "supplied provenance disagrees with the registered pair snapshot",
                )
            if existing is None:
                records.append(record)
                by_id[record.provenance_id] = record

    def add_root(provenance_id: str, subject_ref: str, digest: str, kind: str, ref: str) -> None:
        if provenance_id in by_id:
            record = by_id[provenance_id]
            if isinstance(record.inputs, NotApplicable):
                raise QCInputError(
                    QCAbstentionCode.QC_SOURCE_IDENTITY_INVALID,
                    f"upstream record {provenance_id!r} has no concrete source input",
                )
            if record.subject_ref != subject_ref or not any(
                item.kind == kind
                and item.ref == ref
                and item.sha256_at_use == digest
                for item in record.inputs
            ):
                raise QCInputError(
                    QCAbstentionCode.QC_SOURCE_IDENTITY_INVALID,
                    f"upstream record {provenance_id!r} does not bind {ref!r}",
                )
            return
        record = _root_provenance(
            provenance_id=provenance_id,
            subject_ref=subject_ref,
            inputs=(InputRef(ref=ref, kind=kind, sha256_at_use=digest),),
            at=at,
            created_by=created_by,
            code_commit_sha=code_commit_sha,
        )
        records.append(record)
        by_id[provenance_id] = record

    volume_provenance = context.volume.provenance_ref
    if not isinstance(volume_provenance, str) or not volume_provenance:
        raise QCInputError(
            QCAbstentionCode.QC_SOURCE_IDENTITY_INVALID, "Volume provenance is missing"
        )
    add_root(
        volume_provenance,
        context.volume.volume_id,
        context.volume.blob_ref.sha256,
        "volume",
        _volume_subject(context.volume),
    )
    upstreams.append(volume_provenance)

    if not isinstance(context.source.transform_provenance_ref, str) or not (
        context.source.transform_provenance_ref
    ):
        raise QCInputError(
            QCAbstentionCode.QC_SOURCE_IDENTITY_INVALID,
            "versioned transform provenance is required",
        )
    frame_provenance = context.source.transform_provenance_ref
    transform_ref = context.source.transform_ref
    transform_digest = canonical_digest(context.source.transform)
    add_root(
        frame_provenance,
        transform_ref,
        transform_digest,
        "artifact",
        transform_ref,
    )
    upstreams.append(frame_provenance)

    if pair is not None:
        if pair.first_volume != context.volume:
            raise QCInputError(
                QCAbstentionCode.QC_REQUIRED_PAIR_INVALID,
                "registered pair first Volume differs from QC input",
            )
        second_provenance = pair.second_volume.provenance_ref
        if not isinstance(second_provenance, str) or not second_provenance:
            raise QCInputError(
                QCAbstentionCode.QC_SOURCE_IDENTITY_INVALID,
                "second Volume provenance is missing",
            )
        if second_provenance not in by_id:
            raise QCInputError(
                QCAbstentionCode.QC_REGISTRATION_INVALID,
                "second volume provenance is unresolved",
            )
        upstreams.insert(1, second_provenance)
        registration_provenance = pair.registration_artifact.provenance_ref
        if not isinstance(registration_provenance, str) or not registration_provenance:
            raise QCInputError(
                QCAbstentionCode.QC_REGISTRATION_INVALID, "registration provenance is missing"
            )
        if registration_provenance not in by_id:
            raise QCInputError(
                QCAbstentionCode.QC_REGISTRATION_INVALID, "registration provenance is unresolved"
            )
        upstreams.append(registration_provenance)
    if len(set(upstreams)) != len(upstreams):
        raise QCInputError(
            QCAbstentionCode.QC_SOURCE_IDENTITY_INVALID,
            "QC upstream provenance refs must be unique",
        )
    if any(ref not in by_id for ref in upstreams):
        raise QCInputError(
            QCAbstentionCode.QC_SOURCE_IDENTITY_INVALID,
            "QC upstream provenance refs are unresolved",
        )
    return tuple(records), tuple(upstreams)


def _input_refs_with_pair(context: Any, pair: RegisteredPairInput | None) -> tuple[InputRef, ...]:
    values = list(_canonical_instance_inputs(context))
    if pair is not None:
        registration = pair.registration_artifact
        values.append(
            InputRef(
                ref=f"registration:{registration.artifact_id}",
                kind="artifact",
                sha256_at_use=registration.payload.blob_sha256,
            )
        )
        second = pair.second_volume
        second_frame_digest = canonical_digest(second.coordinate_frame)
        values.append(
            InputRef(
                ref=_volume_subject(second),
                kind="volume",
                sha256_at_use=second.blob_ref.sha256,
            )
        )
        values.append(
            InputRef(
                ref=f"frame_spec:{second_frame_digest}",
                kind="artifact",
                sha256_at_use=second_frame_digest,
            )
        )
        first_frame_ref = next(
            item.ref
            for item in values
            if item.kind == "artifact" and item.ref.startswith("frame_spec:")
        )
        if values[-1].ref == first_frame_ref:
            values.pop()
    return tuple(values)


def _family_evidence_document(
    result: FamilyResult, source_manifest_sha256: str
) -> dict[str, object]:
    return {
        "algorithm_id": result.algorithm_id,
        "algorithm_version": result.algorithm_version,
        "flag": result.flag,
        "metric_name": result.metric_name,
        "metric_value": result.metric_value,
        "outcome": QCOutcome(result.outcome).value,
        "abstention_code": QCAbstentionCode(result.abstention_code).value
        if result.abstention_code is not None
        else None,
        "parameter_digest": result.parameter_digest,
        "source_manifest_sha256": source_manifest_sha256,
    }


def _flags_and_grade(
    family_results: Sequence[FamilyResult],
) -> tuple[QualityGrade, tuple[str, ...]]:
    if len(family_results) != 9:
        raise QCValidationError("the real QC matrix requires all nine family results")
    expected_order = tuple(item.flag for item in family_results)
    if expected_order != FIRST_NINE_FLAGS:
        raise QCValidationError("family results are not in frozen order")
    abstained = [item for item in family_results if item.outcome == QCOutcome.ABSTAINED]
    if abstained:
        return QualityGrade.NOT_ANALYZABLE, ("ABSTAINED",)
    severe = [item.flag for item in family_results if item.outcome == QCOutcome.SEVERE]
    mild = [item.flag for item in family_results if item.outcome == QCOutcome.MILD]
    non_clear = tuple(flag for flag in FIRST_NINE_FLAGS if flag in severe or flag in mild)
    if severe:
        return QualityGrade.LIMITED, non_clear
    if mild:
        return QualityGrade.ACCEPTABLE, non_clear
    return QualityGrade.GOOD, ()


def validate_grade_flag_matrix(
    quality_grade: QualityGrade | str,
    flags: Sequence[str],
    family_results: Sequence[FamilyResult],
) -> bool:
    """Validate the closed grade↔flag matrix before either view is made."""

    grade = QualityGrade(quality_grade)
    flag_tuple = tuple(flags)
    if len(set(flag_tuple)) != len(flag_tuple):
        raise QCValidationError("QC flags contain duplicates")
    allowed = set(FIRST_NINE_FLAGS) | {"ABSTAINED"}
    if any(flag not in allowed for flag in flag_tuple):
        raise QCValidationError("QC flags contain an unknown or model label")
    if tuple(flag for flag in QC_FLAGS if flag in flag_tuple) != flag_tuple:
        raise QCValidationError("QC flags are not in canonical order")
    expected_grade, expected_flags = _flags_and_grade(family_results)
    if grade != expected_grade or flag_tuple != expected_flags:
        raise QCValidationError("grade/flag matrix contradiction")
    return True


def _payload_document(
    *,
    subject_refs: tuple[str, ...],
    grade: QualityGrade,
    flags: tuple[str, ...],
    produced_by_ref: str,
    supersedes_ref: str | None,
    evidence_refs: tuple[str, ...],
) -> dict[str, object]:
    # This is deliberately the exact thirteen-key v1 document.  Do not add
    # family metrics here; those bind through the evidence provenance records.
    return {
        "schema_version": PAYLOAD_SCHEMA,
        "scope": QCScope.STUDY.value,
        "subject_refs": subject_refs,
        "quality_grade": grade.value,
        "confidence": None,
        "flags": flags,
        "vessel_id": None,
        "region_id": None,
        "producer": ProducerType.DETERMINISTIC_ENGINE.value,
        "produced_by_ref": produced_by_ref,
        "supersedes_ref": supersedes_ref,
        "justification_code": "RULE_ENGINE_EVIDENCE",
        "evidence_refs": evidence_refs,
    }


def _build_bundle(
    *,
    context: Any | None,
    family_results: tuple[FamilyResult, ...],
    refusal: QCInputRefusal | None,
    pair: RegisteredPairInput | None,
    subject_refs: tuple[str, ...],
    source_manifest_sha256: str,
    source_inputs: tuple[InputRef, ...],
    source_volume: Volume | None,
    geometry_base: GeometryBase,
    provenance_records: Sequence[ProvenanceRecord],
    upstream_refs: tuple[str, ...],
    artifact_id: str,
    aggregate_provenance_id: str,
    family_provenance_ids: tuple[str, ...],
    supersedes_ref: str | None,
    claim_binding: ClaimBinding,
    release_lane: ReleaseLane,
    created_by: str,
    created_at: str | None,
    code_commit_sha: str,
    software_version: str,
    artifact_store: DerivedArtifactStore | None,
) -> QCResultBundle:
    if not _valid_ulid(artifact_id) or not _valid_ulid(aggregate_provenance_id):
        raise QCValidationError("QC artifact and aggregate provenance ids must be ULIDs")
    if len(set(family_provenance_ids)) != len(family_provenance_ids) or any(
        not _valid_ulid(item) for item in family_provenance_ids
    ):
        raise QCValidationError("QC family provenance ids must be unique ULIDs")
    if not source_inputs or any(not isinstance(item, InputRef) for item in source_inputs):
        raise QCValidationError("QC provenance inputs must be concrete InputRef objects")
    if len({(item.ref, item.kind) for item in source_inputs}) != len(source_inputs):
        raise QCValidationError("QC provenance inputs must not contain duplicate refs")
    if not upstream_refs or len(set(upstream_refs)) != len(upstream_refs):
        raise QCValidationError("QC provenance upstream_refs must be unique and nonempty")
    if any(not isinstance(item, str) or not item for item in upstream_refs):
        raise QCValidationError("QC provenance upstream_refs must be concrete strings")
    if refusal is None:
        if context is None:
            raise QCValidationError("normal QC requires a validated context")
        grade, flags = _flags_and_grade(family_results)
    else:
        if len(family_results) != 1 or family_results[0].outcome != QCOutcome.ABSTAINED:
            raise QCValidationError("upstream refusal must contain one abstained family")
        grade, flags = QualityGrade.NOT_ANALYZABLE, ("ABSTAINED",)
    validate_grade_flag_matrix(
        grade,
        flags,
        family_results if refusal is None else _expand_refusal_results(family_results, refusal),
    )
    assessment_digest = assessment_parameter_digest()
    parameter_manifest = assessment_parameter_manifest()
    aggregate_id = _provenance_ref(aggregate_provenance_id)
    family_records: list[ProvenanceRecord] = []
    evidence_documents: dict[str, Mapping[str, object]] = {}
    input_tuple = tuple(source_inputs)
    for index, result in enumerate(family_results):
        family_provenance_id = family_provenance_ids[index]
        evidence_document = _family_evidence_document(result, source_manifest_sha256)
        evidence_digest = canonical_digest(evidence_document)
        family_record = create_provenance(
            subject_ref=f"qc_family_evidence:{evidence_digest}",
            event_type=ProvenanceEventType.DERIVED,
            actor=Actor(actor_type=ActorType.ENGINE, actor_id=str(result.algorithm_id)),
            inputs=input_tuple,
            preprocessing=QC_PREPROCESSING,
            parameters_ref=BlobRef(
                blob_ref=f"qc-parameters:{result.parameter_digest}",
                sha256=cast(str, result.parameter_digest),
            ),
            human_edit=NotApplicable(value=None, reason="deterministic QC; no human edit"),
            upstream_refs=upstream_refs,
            environment=Environment(
                software_version=software_version,
                code_commit_sha=code_commit_sha,
                deterministic_seed=None,
            ),
            model_record_ref=None,
            created_by=created_by,
            at=created_at,
            provenance_id=family_provenance_id,
        )
        family_records.append(family_record)
        evidence_documents[_provenance_ref(family_provenance_id)] = evidence_document
    if refusal is None:
        volume_refs: tuple[str, ...] = ()
        if source_volume is not None:
            volume_values = [source_volume]
            if pair is not None and pair.second_volume != source_volume:
                volume_values.append(pair.second_volume)
            volume_refs = tuple(_volume_subject(volume) for volume in volume_values)
        evidence_refs = (
            aggregate_id,
            f"source_manifest:{source_manifest_sha256}",
            *volume_refs,
        ) + tuple(_provenance_ref(item) for item in family_provenance_ids)
    else:
        evidence_refs = (
            aggregate_id,
            f"source_manifest:{source_manifest_sha256}",
            _provenance_ref(family_provenance_ids[0]),
        )
    payload_doc = _payload_document(
        subject_refs=subject_refs,
        grade=grade,
        flags=flags,
        produced_by_ref=aggregate_id,
        supersedes_ref=supersedes_ref,
        evidence_refs=evidence_refs,
    )
    if set(payload_doc) != {
        "schema_version",
        "scope",
        "subject_refs",
        "quality_grade",
        "confidence",
        "flags",
        "vessel_id",
        "region_id",
        "producer",
        "produced_by_ref",
        "supersedes_ref",
        "justification_code",
        "evidence_refs",
    }:
        raise QCValidationError("QC payload key set is not closed")
    aggregate_inputs = tuple(source_inputs)
    aggregate_record = create_provenance(
        subject_ref=artifact_id,
        event_type=ProvenanceEventType.RECOMPUTED
        if supersedes_ref is not None
        else ProvenanceEventType.DERIVED,
        actor=Actor(actor_type=ActorType.ENGINE, actor_id=ASSESSMENT_ALGORITHM_ID),
        inputs=aggregate_inputs,
        preprocessing=QC_PREPROCESSING,
        parameters_ref=BlobRef(
            blob_ref=f"qc-parameters:{assessment_digest}", sha256=assessment_digest
        ),
        human_edit=NotApplicable(value=None, reason="deterministic QC; no human edit"),
        upstream_refs=upstream_refs,
        environment=Environment(
            software_version=software_version,
            code_commit_sha=code_commit_sha,
            deterministic_seed=None,
        ),
        model_record_ref=None,
        created_by=created_by,
        at=created_at,
        provenance_id=aggregate_provenance_id,
    )
    all_records = tuple(provenance_records) + tuple(family_records) + (aggregate_record,)
    # Reject duplicate ids before verify_chain silently overwrites a map key.
    if len({record.provenance_id for record in all_records}) != len(all_records):
        raise QCValidationError("QC provenance ids are not unique")
    chain_ok, chain_problems = verify_chain(all_records)
    if not chain_ok:
        raise QCValidationError("QC provenance chain failed: " + "; ".join(chain_problems))
    producer = Producer(
        producer_type=ProducerType.DETERMINISTIC_ENGINE,
        algorithm=ASSESSMENT_ALGORITHM_ID,
        algorithm_version=ALGORITHM_VERSION,
        parameters_blob_sha256=assessment_digest,
    )
    artifact = create_derived_artifact(
        artifact_id=artifact_id,
        study_ref=source_volume.study_ref if source_volume is not None else subject_refs[0],
        kind=ArtifactKind.QC_ASSESSMENT,
        payload=PayloadRef(
            payload_schema=PAYLOAD_SCHEMA,
            blob_sha256=canonical_digest(payload_doc),
            codec="json",
            inline_summary=(("quality_grade", grade.value),),
        ),
        lineage=Lineage(
            input_volume_refs=()
            if source_volume is None
            else (
                (source_volume.volume_id, pair.second_volume.volume_id)
                if pair is not None
                else (source_volume.volume_id,)
            ),
            input_artifact_refs=() if pair is None else (pair.registration_artifact.artifact_id,),
            input_series_refs=()
            if source_volume is None
            else (
                tuple(source_volume.source_series_refs)
                if pair is None
                else tuple(
                    (*source_volume.source_series_refs, *pair.second_volume.source_series_refs)
                )
            ),
            geometry_base=geometry_base,
            registration_ref=None
            if pair is None
            else f"registration:{pair.registration_artifact.artifact_id}",
        ),
        claim_binding=claim_binding,
        producer=producer,
        review=ReviewBlock(
            review_state=ReviewState.PREDICTED,
            supersedes_ref=supersedes_ref,
        ),
        quality=QualityBlock(quality_grade=grade, confidence=None, flags=flags),
        release_lane=release_lane,
        provenance_ref=aggregate_id,
        created_by=created_by,
        created_at=created_at,
    )
    assessment = QCAssessmentArtifact(
        artifact_ref=f"qc_artifact:{artifact_id}",
        payload=QCAssessmentPayload(
            schema_version=PAYLOAD_SCHEMA,
            scope=QCScope.STUDY,
            subject_refs=subject_refs,
            quality_grade=grade,
            confidence=None,
            flags=flags,
            vessel_id=None,
            region_id=None,
            producer=ProducerType.DETERMINISTIC_ENGINE,
            produced_by_ref=aggregate_id,
            supersedes_ref=supersedes_ref,
            justification_code="RULE_ENGINE_EVIDENCE",
            evidence_refs=evidence_refs,
        ),
        producer=ProducerType.DETERMINISTIC_ENGINE,
        status=ArtifactStatus.ACTIVE,
    )
    bundle = QCResultBundle(
        assessment=assessment,
        artifact=artifact,
        payload_document=payload_doc,
        family_results=family_results,
        family_evidence_documents=evidence_documents,
        parameter_manifest=parameter_manifest,
        provenance_records=all_records,
        family_result_body={
            "families": tuple(
                {
                    "algorithm_id": item.algorithm_id,
                    "algorithm_version": item.algorithm_version,
                    "flag": item.flag,
                    "metric_value": item.metric_value,
                    "outcome": QCOutcome(item.outcome).value,
                    "abstention_code": (
                        QCAbstentionCode(item.abstention_code).value
                        if item.abstention_code is not None
                        else None
                    ),
                }
                for item in family_results
            ),
            "quality_grade": grade.value,
            "flags": flags,
            "parameter_digest": assessment_digest,
        },
        refusal=refusal,
    )
    validate_result_bundle(bundle)
    if artifact_store is not None:
        if supersedes_ref is not None and supersedes_ref.startswith("qc_artifact:"):
            old_id = supersedes_ref.split(":", 1)[1]
            if old_id in artifact_store:
                artifact_store.mark_superseded(old_id, superseded_by=artifact_id)
        artifact_store.add(artifact)
    return bundle


def _expand_refusal_results(
    results: Sequence[FamilyResult], refusal: QCInputRefusal
) -> tuple[FamilyResult, ...]:
    """Supply matrix facts for the one-family refusal representation."""

    if len(results) != 1 or results[0].flag != refusal.family:
        raise QCValidationError("refusal family result does not bind refusal DTO")
    by_flag = {item.flag: item for item in results}
    return tuple(
        by_flag.get(
            flag,
            FamilyResult(
                flag=flag,
                metric_value=None,
                outcome=QCOutcome.ABSTAINED,
                abstention_code=refusal.reason_code,
            ),
        )
        for flag in FIRST_NINE_FLAGS
    )


def create_real_qc_assessment(
    source: QCInput,
    *,
    pair: RegisteredPairInput | None = None,
    coverage: CoverageEnvelope | Mapping[str, object] | None = None,
    provenance_records: Sequence[ProvenanceRecord] = (),
    artifact_store: DerivedArtifactStore | None = None,
    subject_refs: tuple[str, ...] | None = None,
    artifact_id: str | None = None,
    provenance_id: str | None = None,
    family_provenance_ids: tuple[str, ...] | None = None,
    supersedes_ref: str | None = None,
    claim_binding: ClaimBinding = _DEFAULT_CLAIM,
    release_lane: ReleaseLane = ReleaseLane.PERSONAL_RESEARCH,
    created_by: str = "qc_001_det",
    created_at: str | None = None,
    code_commit_sha: str = _ZERO_COMMIT,
    software_version: str = _DEFAULT_SOFTWARE,
    id_factory: Callable[[], str] | None = None,
) -> QCResultBundle:
    """Run the nine families and create the shared-identity REAL v1 bundle."""

    context = validate_qc_input(source)
    if pair is not None:
        validate_registered_pair_input(pair)
    records, upstreams = _ensure_upstream_records(
        context,
        provenance_records,
        pair=pair,
        at=created_at or utc_now_iso(),
        created_by=created_by,
        code_commit_sha=code_commit_sha,
    )
    family_results = evaluate_all_families(source, pair=pair, coverage=coverage)
    mint = id_factory or new_ulid
    resolved_artifact_id = artifact_id or mint()
    resolved_provenance_id = provenance_id or mint()
    resolved_family_ids = family_provenance_ids or tuple(mint() for _ in family_results)
    if len(resolved_family_ids) != len(family_results):
        raise QCValidationError("one provenance id is required per family result")
    subjects = (
        (_study_subject(context.volume.study_ref),)
        if subject_refs is None
        else tuple(subject_refs)
    )
    if not subjects:
        raise QCValidationError("QC subject_refs are required")
    geometry_base = context.geometry_base
    if pair is not None:
        from .pair import pair_geometry_id, pair_geometry_version

        geometry_base = GeometryBase(pair_geometry_id(pair), pair_geometry_version(pair))
    source_inputs = _input_refs_with_pair(context, pair)
    return _build_bundle(
        context=context,
        family_results=family_results,
        refusal=None,
        pair=pair,
        subject_refs=subjects,
        source_manifest_sha256=context.source_manifest_sha256,
        source_inputs=source_inputs,
        source_volume=context.volume,
        geometry_base=geometry_base,
        provenance_records=records,
        upstream_refs=upstreams,
        artifact_id=resolved_artifact_id,
        aggregate_provenance_id=resolved_provenance_id,
        family_provenance_ids=tuple(resolved_family_ids),
        supersedes_ref=supersedes_ref,
        claim_binding=claim_binding,
        release_lane=release_lane,
        created_by=created_by,
        created_at=created_at,
        code_commit_sha=code_commit_sha,
        software_version=software_version,
        artifact_store=artifact_store,
    )


def validate_result_bundle(bundle: QCResultBundle) -> bool:
    """Mechanically validate both views, exact evidence order, and bindings."""

    if not isinstance(bundle, QCResultBundle):
        raise QCValidationError("QC result bundle type is invalid")
    artifact = bundle.artifact
    assessment = bundle.assessment
    if not isinstance(artifact, DerivedArtifact) or not isinstance(
        assessment, QCAssessmentArtifact
    ):
        raise QCValidationError("QC artifact views have invalid types")
    payload = assessment.payload
    payload_keys = {
        "schema_version",
        "scope",
        "subject_refs",
        "quality_grade",
        "confidence",
        "flags",
        "vessel_id",
        "region_id",
        "producer",
        "produced_by_ref",
        "supersedes_ref",
        "justification_code",
        "evidence_refs",
    }
    if assessment.artifact_ref != f"qc_artifact:{artifact.artifact_id}":
        raise QCValidationError("QC views do not share one artifact identity")
    if (
        assessment.status != "ACTIVE"
        or assessment.kind != artifact.kind.value
        or assessment.payload_schema != artifact.payload.payload_schema
    ):
        raise QCValidationError("QC outer view lifecycle/kind/schema is not bound")
    if (
        artifact.kind != ArtifactKind.QC_ASSESSMENT
        or artifact.status != ArtifactStatus.ACTIVE
        or artifact.payload.payload_schema != PAYLOAD_SCHEMA
        or artifact.payload.blob_sha256 != canonical_digest(bundle.payload_document)
    ):
        raise QCValidationError("QC derived artifact kind/schema/status/digest is invalid")
    if (
        not isinstance(bundle.payload_document, Mapping)
        or set(bundle.payload_document) != payload_keys
    ):
        raise QCValidationError("QC payload has extra or missing keys")
    if (
        payload.schema_version != PAYLOAD_SCHEMA
        or payload.scope != QCScope.STUDY
        or len(payload.subject_refs) != 1
        or artifact.study_ref != payload.subject_refs[0]
    ):
        raise QCValidationError("QC study identity/scope is not bound across views")
    expected_payload = {
        "schema_version": payload.schema_version,
        "scope": payload.scope.value,
        "subject_refs": payload.subject_refs,
        "quality_grade": payload.quality_grade.value,
        "confidence": None,
        "flags": payload.flags,
        "vessel_id": None,
        "region_id": None,
        "producer": payload.producer.value,
        "produced_by_ref": payload.produced_by_ref,
        "supersedes_ref": payload.supersedes_ref,
        "justification_code": payload.justification_code,
        "evidence_refs": payload.evidence_refs,
    }
    if canonical_digest(expected_payload) != canonical_digest(bundle.payload_document):
        raise QCValidationError("payload DTO and payload document disagree")
    if (
        payload.produced_by_ref != artifact.provenance_ref
        or payload.producer != ProducerType.DETERMINISTIC_ENGINE
        or assessment.producer != artifact.producer.producer_type
        or artifact.producer.producer_type != ProducerType.DETERMINISTIC_ENGINE
        or payload.justification_code != "RULE_ENGINE_EVIDENCE"
    ):
        raise QCValidationError("QC producer/justification binding is invalid")
    if (
        artifact.quality.quality_grade != payload.quality_grade
        or artifact.quality.confidence != payload.confidence
        or artifact.quality.flags != payload.flags
        or artifact.review.supersedes_ref != payload.supersedes_ref
        or artifact.review.review_state != ReviewState.PREDICTED
    ):
        raise QCValidationError("QC quality/review projections are not bound")
    if (
        artifact.producer.algorithm != ASSESSMENT_ALGORITHM_ID
        or artifact.producer.algorithm_version != ALGORITHM_VERSION
        or artifact.producer.parameters_blob_sha256 != assessment_parameter_digest()
    ):
        raise QCValidationError("QC assessment algorithm identity is invalid")
    if dict(bundle.parameter_manifest) != assessment_parameter_manifest():
        raise QCValidationError("QC assessment parameter manifest is not frozen")

    expected_grade: QualityGrade
    expected_flags: tuple[str, ...]
    if bundle.refusal is not None:
        expected_grade, expected_flags = (
            QualityGrade.NOT_ANALYZABLE,
            ("ABSTAINED",),
        )
        if (
            len(bundle.family_results) != 1
            or bundle.family_results[0].flag != bundle.refusal.family
            or bundle.family_results[0].abstention_code != bundle.refusal.reason_code
            or bundle.family_results[0].outcome != QCOutcome.ABSTAINED
            or payload.quality_grade != QualityGrade.NOT_ANALYZABLE
            or payload.flags != ("ABSTAINED",)
        ):
            raise QCValidationError("QC refusal result is not bound to NOT_ANALYZABLE")
        if (
            artifact.lineage.geometry_base is None
            or artifact.lineage.geometry_base.geometry_id
            != f"qc_refusal_surface:{bundle.refusal.source_manifest_sha256}"
            or artifact.lineage.geometry_base.geometry_version != 1
        ):
            raise QCValidationError("QC refusal sentinel geometry is invalid")
        expected_family_count = 1
        if (
            artifact.lineage.input_volume_refs
            or artifact.lineage.input_artifact_refs
            or artifact.lineage.input_series_refs
            or artifact.lineage.registration_ref is not None
        ):
            raise QCValidationError("QC refusal lineage may not claim image geometry")
        if bundle.refusal.source_manifest_sha256 != payload.evidence_refs[1].split(":", 1)[1]:
            raise QCValidationError("QC refusal source manifest binding is invalid")
    else:
        expected_grade, expected_flags = _flags_and_grade(bundle.family_results)
        if len(bundle.family_results) != len(FIRST_NINE_FLAGS):
            raise QCValidationError("normal QC must contain all nine family results")
        if payload.quality_grade != expected_grade or payload.flags != expected_flags:
            raise QCValidationError("QC grade/flag matrix is inconsistent")
        expected_family_count = len(FIRST_NINE_FLAGS)
        if artifact.lineage.geometry_base is None:
            raise QCValidationError("normal QC must bind a GeometryBase")
        if not artifact.lineage.input_volume_refs:
            raise QCValidationError("normal QC must bind its source volume")
        if len(artifact.lineage.input_volume_refs) not in {1, 2}:
            raise QCValidationError("normal QC has an invalid source-volume count")
        if len(artifact.lineage.input_volume_refs) == 1:
            if (
                artifact.lineage.input_artifact_refs
                or artifact.lineage.registration_ref is not None
            ):
                raise QCValidationError("single-volume QC may not claim registration")
        elif (
            len(artifact.lineage.input_artifact_refs) != 1
            or artifact.lineage.registration_ref
            != f"registration:{artifact.lineage.input_artifact_refs[0]}"
        ):
            raise QCValidationError("paired QC registration lineage is not exact")
    if bundle.refusal is not None and expected_grade != QualityGrade.NOT_ANALYZABLE:
        raise QCValidationError("refusal matrix expansion is inconsistent")

    evidence = payload.evidence_refs
    if not isinstance(evidence, tuple) or len(set(evidence)) != len(evidence):
        raise QCValidationError("QC evidence refs must be a unique tuple")
    aggregate_evidence = f"provenance:{artifact.provenance_ref.removeprefix('provenance:')}"
    if not evidence or evidence[0] != aggregate_evidence:
        raise QCValidationError("aggregate evidence must be first")
    dataset_refs = [item for item in evidence if item.startswith("source_manifest:")]
    if len(dataset_refs) != 1:
        raise QCValidationError("QC evidence requires one source manifest ref")
    source_manifest_sha = dataset_refs[0].split(":", 1)[1]
    if not is_sha256_hex(source_manifest_sha):
        raise QCValidationError("QC evidence source manifest is not a SHA-256")
    if evidence[1] != dataset_refs[0]:
        raise QCValidationError("source manifest evidence must be second")
    family_refs = tuple(ref for ref in evidence if ref.startswith("provenance:"))[1:]
    if len(family_refs) != expected_family_count:
        raise QCValidationError("QC family evidence count is invalid")
    if tuple(ref for ref in family_refs) != tuple(ref for ref in evidence[-expected_family_count:]):
        raise QCValidationError("QC family evidence must be the final ordered refs")
    volume_refs = tuple(ref for ref in evidence if ref.startswith("volume_id:"))
    if bundle.refusal is not None:
        if volume_refs or len(evidence) != 3:
            raise QCValidationError("QC refusal evidence may not claim a volume")
    else:
        expected_volume_refs = tuple(
            f"volume_id:{ref}" for ref in artifact.lineage.input_volume_refs
        )
        if volume_refs != expected_volume_refs:
            raise QCValidationError("QC volume evidence order does not match lineage")
        if len(evidence) != 3 + len(volume_refs) + expected_family_count - 1:
            # The first three terms are aggregate, source, and the first
            # volume; a single-volume run has one volume and nine families.
            raise QCValidationError("QC evidence length is invalid")

    if bundle.refusal is None:
        expected_evidence = (
            aggregate_evidence,
            dataset_refs[0],
            *volume_refs,
            *family_refs,
        )
    else:
        expected_evidence = (aggregate_evidence, dataset_refs[0], *family_refs)
    if evidence != expected_evidence:
        raise QCValidationError("QC evidence refs are not in the frozen canonical order")

    aggregate_id = artifact.provenance_ref.removeprefix("provenance:")
    if not _valid_ulid(aggregate_id):
        raise QCValidationError("aggregate provenance ref is not a ULID")
    if len({record.provenance_id for record in bundle.provenance_records}) != len(
        bundle.provenance_records
    ):
        raise QCValidationError("QC provenance ids are duplicated")
    records = {record.provenance_id: record for record in bundle.provenance_records}
    aggregate = records.get(aggregate_id)
    if aggregate is None:
        raise QCValidationError("aggregate provenance is unresolved")
    if (
        aggregate.actor.actor_type != ActorType.ENGINE
        or aggregate.actor.actor_id != ASSESSMENT_ALGORITHM_ID
        or aggregate.subject_ref != artifact.artifact_id
        or aggregate.event_type not in {ProvenanceEventType.DERIVED, ProvenanceEventType.RECOMPUTED}
        or aggregate.preprocessing != QC_PREPROCESSING
        or aggregate.model_record_ref is not None
        or not isinstance(aggregate.human_edit, NotApplicable)
        or aggregate.human_edit.reason != "deterministic QC; no human edit"
        or isinstance(aggregate.parameters_ref, NotApplicable)
        or aggregate.parameters_ref.sha256 != assessment_parameter_digest()
        or isinstance(aggregate.inputs, NotApplicable)
        or not aggregate.inputs
        or isinstance(aggregate.upstream_refs, NotApplicable)
        or not aggregate.upstream_refs
        or len(set(aggregate.upstream_refs)) != len(aggregate.upstream_refs)
        or any(not isinstance(item, str) or not item for item in aggregate.upstream_refs)
    ):
        raise QCValidationError("aggregate QC provenance branch is invalid")
    manifest_inputs = [item for item in aggregate.inputs if item.kind == "dataset"]
    if (
        len(manifest_inputs) != 1
        or manifest_inputs[0].ref != dataset_refs[0]
        or manifest_inputs[0].sha256_at_use != source_manifest_sha
    ):
        raise QCValidationError("aggregate source-manifest input is not concretely bound")
    if any(not is_sha256_hex(item.sha256_at_use) for item in aggregate.inputs):
        raise QCValidationError("aggregate provenance contains an invalid input digest")
    if not isinstance(aggregate.parameters_ref, BlobRef):
        raise QCValidationError("aggregate QC parameters must be a concrete BlobRef")
    if aggregate.parameters_ref.blob_ref != f"qc-parameters:{assessment_parameter_digest()}":
        raise QCValidationError("aggregate QC parameter blob ref is not content-bound")
    if any(item not in records for item in aggregate.upstream_refs):
        raise QCValidationError("aggregate QC upstream provenance is unresolved")

    if len(bundle.family_evidence_documents) != expected_family_count:
        raise QCValidationError("family evidence document count is invalid")
    for family_ref, family_result in zip(family_refs, bundle.family_results):
        family_id = family_ref.split(":", 1)[1]
        if not _valid_ulid(family_id):
            raise QCValidationError("family evidence ref is not a ULID")
        record = records.get(family_id)
        document = bundle.family_evidence_documents.get(family_ref)
        if record is None or document is None or set(document) != set(_EVIDENCE_KEYS):
            raise QCValidationError("family evidence provenance/document is unresolved")
        if record.subject_ref != f"qc_family_evidence:{canonical_digest(document)}":
            raise QCValidationError("family evidence subject digest is not bound")
        if (
            record.actor.actor_type != ActorType.ENGINE
            or record.actor.actor_id != family_result.algorithm_id
            or record.event_type != ProvenanceEventType.DERIVED
            or record.preprocessing != QC_PREPROCESSING
            or record.model_record_ref is not None
            or not isinstance(record.human_edit, NotApplicable)
            or record.human_edit.reason != "deterministic QC; no human edit"
            or isinstance(record.parameters_ref, NotApplicable)
            or record.parameters_ref.sha256 != family_result.parameter_digest
            or record.parameters_ref.blob_ref
            != f"qc-parameters:{family_result.parameter_digest}"
            or record.inputs != aggregate.inputs
            or record.upstream_refs != aggregate.upstream_refs
            or record.environment.software_version != aggregate.environment.software_version
            or record.environment.code_commit_sha != aggregate.environment.code_commit_sha
        ):
            raise QCValidationError("family evidence provenance binding is invalid")
        expected_document = _family_evidence_document(family_result, source_manifest_sha)
        if dict(document) != expected_document:
            raise QCValidationError("family evidence body does not bind FamilyResult")
        if (
            family_result.algorithm_id != FAMILY_ALGORITHM_IDS[family_result.flag]
            or family_result.algorithm_version != ALGORITHM_VERSION
            or family_result.metric_name != FAMILY_METRIC_NAMES[family_result.flag]
            or family_result.parameter_digest != family_parameter_digest(family_result.flag)
        ):
            raise QCValidationError("family result identity is not frozen")

    expected_family_evidence_keys = set(family_refs)
    if set(bundle.family_evidence_documents) != expected_family_evidence_keys:
        raise QCValidationError("family evidence map has extra or missing refs")
    expected_body = {
        "families": tuple(
            {
                "algorithm_id": item.algorithm_id,
                "algorithm_version": item.algorithm_version,
                "flag": item.flag,
                "metric_value": item.metric_value,
                "outcome": QCOutcome(item.outcome).value,
                "abstention_code": QCAbstentionCode(item.abstention_code).value
                if item.abstention_code is not None
                else None,
            }
            for item in bundle.family_results
        ),
        "quality_grade": payload.quality_grade.value,
        "flags": payload.flags,
        "parameter_digest": assessment_parameter_digest(),
    }
    if dict(bundle.family_result_body) != expected_body:
        raise QCValidationError("content-derived family result body is not exact")
    ok, problems = verify_chain(bundle.provenance_records)
    if not ok:
        raise QCValidationError("QC bundle provenance chain failed: " + "; ".join(problems))
    return True


def create_qc_assessment(*args: Any, **kwargs: Any) -> QCResultBundle:
    """Public production constructor; unlike the closed seam this is REAL."""

    return create_real_qc_assessment(*args, **kwargs)


build_qc_assessment = create_real_qc_assessment
make_qc_assessment = create_real_qc_assessment
assess_qc = create_real_qc_assessment


UPSTREAM_REFUSAL_CODES: dict[str, QCAbstentionCode] = {
    "unsupported transfer syntax": QCAbstentionCode.QC_INPUT_SURFACE_INVALID,
    "unsupported_transfer_syntax": QCAbstentionCode.QC_INPUT_SURFACE_INVALID,
    "malformed pixel length": QCAbstentionCode.QC_INPUT_SURFACE_INVALID,
    "malformed_pixel_length": QCAbstentionCode.QC_INPUT_SURFACE_INVALID,
    "pixel length mismatch": QCAbstentionCode.QC_INPUT_SURFACE_INVALID,
    "pixel_length_mismatch": QCAbstentionCode.QC_INPUT_SURFACE_INVALID,
    "multiframe": QCAbstentionCode.QC_INPUT_SURFACE_INVALID,
    "multi frame": QCAbstentionCode.QC_INPUT_SURFACE_INVALID,
    "number of frames": QCAbstentionCode.QC_INPUT_SURFACE_INVALID,
    "identity or frameofreferenceuid mismatch": QCAbstentionCode.QC_SOURCE_IDENTITY_INVALID,
    "identity or frame of reference uid mismatch": QCAbstentionCode.QC_SOURCE_IDENTITY_INVALID,
    "identity mismatch": QCAbstentionCode.QC_SOURCE_IDENTITY_INVALID,
    "frameofreferenceuid mismatch": QCAbstentionCode.QC_SOURCE_IDENTITY_INVALID,
    "frame of reference uid mismatch": QCAbstentionCode.QC_SOURCE_IDENTITY_INVALID,
    "missing stack member": QCAbstentionCode.QC_GEOMETRY_INVALID,
    "missing slice": QCAbstentionCode.QC_GEOMETRY_INVALID,
    "series incomplete": QCAbstentionCode.QC_GEOMETRY_INVALID,
    "tilted/nonparallel stack": QCAbstentionCode.QC_GEOMETRY_INVALID,
    "tilted nonparallel stack": QCAbstentionCode.QC_GEOMETRY_INVALID,
    "gantry tilt": QCAbstentionCode.QC_GEOMETRY_INVALID,
    "series stack invalid": QCAbstentionCode.QC_GEOMETRY_INVALID,
    "unknown scalar semantics": QCAbstentionCode.QC_SCALAR_DOMAIN_INVALID,
    "scalar domain invalid": QCAbstentionCode.QC_SCALAR_DOMAIN_INVALID,
    "intensity unit mismatch": QCAbstentionCode.QC_SCALAR_DOMAIN_INVALID,
    "absent required pair": QCAbstentionCode.QC_REQUIRED_PAIR_ABSENT,
    "required pair absent": QCAbstentionCode.QC_REQUIRED_PAIR_ABSENT,
}


def _failure_text(failure: object) -> str:
    if isinstance(failure, str):
        return failure.strip().casefold()
    reason = getattr(failure, "reason_code", None)
    detail = getattr(failure, "detail_code", None)
    return (
        " ".join(
            str(item)
            for item in (reason, detail, str(failure), getattr(failure, "args", ()))
            if item is not None
        )
        .strip()
        .casefold()
    )


def map_upstream_failure(failure: object) -> QCAbstentionCode:
    text = _failure_text(failure)
    normalized = text.replace("_", " ").replace("-", " ")
    # Preserve the typed reader/stack/materializer reason where available;
    # this avoids depending on exception prose or filenames.
    typed_detail = str(getattr(failure, "detail_code", "")).casefold().replace("_", " ")
    typed_reason = str(getattr(failure, "reason_code", "")).casefold().replace("_", " ")
    typed = f"{typed_reason} {typed_detail} {normalized}"
    if "unsupported transfer syntax" in typed:
        return QCAbstentionCode.QC_INPUT_SURFACE_INVALID
    if "pixel length mismatch" in typed or "malformed pixel length" in typed:
        return QCAbstentionCode.QC_INPUT_SURFACE_INVALID
    if "multiframe" in typed or "multi frame" in typed or "number of frames" in typed:
        return QCAbstentionCode.QC_INPUT_SURFACE_INVALID
    if "dicom syntax invalid" in typed:
        return QCAbstentionCode.QC_INPUT_SURFACE_INVALID
    if (
        "identity mismatch" in typed
        or "identity frame mismatch" in typed
        or "frame of reference uid mismatch" in typed
    ):
        return QCAbstentionCode.QC_SOURCE_IDENTITY_INVALID
    if "series incomplete" in typed or "missing stack member" in typed:
        return QCAbstentionCode.QC_GEOMETRY_INVALID
    if "stack invalid" in typed or "tilted" in typed or "nonparallel" in typed:
        return QCAbstentionCode.QC_GEOMETRY_INVALID
    if "scalar domain invalid" in typed or "unknown scalar semantics" in typed:
        return QCAbstentionCode.QC_SCALAR_DOMAIN_INVALID
    if "required pair absent" in typed or "absent required pair" in typed:
        return QCAbstentionCode.QC_REQUIRED_PAIR_ABSENT
    for key, code in UPSTREAM_REFUSAL_CODES.items():
        normalized_key = key.replace("_", " ").replace("-", " ")
        if normalized_key in normalized:
            return code
    raise QCValidationError(f"unmapped upstream QC refusal: {failure!r}")


def make_qc_input_refusal(
    *,
    family: str,
    failure: object,
    source_manifest_sha256: str,
) -> QCInputRefusal:
    code = map_upstream_failure(failure)
    return QCInputRefusal(
        family=family,
        reason_code=code,
        upstream_failure=_failure_text(failure),
        source_manifest_sha256=source_manifest_sha256,
    )


def create_qc_input_refusal(
    *,
    failure: object,
    source_manifest_sha256: str,
    study_ref: str,
    family: str = "CARDIAC_MOTION",
    instances: Sequence[Any] = (),
    instance_digests: Mapping[str, str] | tuple[tuple[str, str], ...] = (),
    provenance_records: Sequence[ProvenanceRecord] = (),
    source_provenance_ref: str | None = None,
    artifact_id: str | None = None,
    provenance_id: str | None = None,
    family_provenance_id: str | None = None,
    supersedes_ref: str | None = None,
    claim_binding: ClaimBinding = _DEFAULT_CLAIM,
    release_lane: ReleaseLane = ReleaseLane.PERSONAL_RESEARCH,
    created_by: str = "qc_001_det",
    created_at: str | None = None,
    code_commit_sha: str = _ZERO_COMMIT,
    software_version: str = _DEFAULT_SOFTWARE,
    artifact_store: DerivedArtifactStore | None = None,
    id_factory: Callable[[], str] | None = None,
) -> QCResultBundle:
    """Convert a typed reader/stack/materializer failure into REAL ABSTAINED."""

    if not is_sha256_hex(source_manifest_sha256):
        raise QCValidationError("source manifest must be lowercase SHA-256")
    refusal = make_qc_input_refusal(
        family=family,
        failure=failure,
        source_manifest_sha256=source_manifest_sha256,
    )
    subject = _study_subject(study_ref)
    at = created_at or utc_now_iso()
    ids = id_factory or new_ulid
    resolved_artifact_id = artifact_id or ids()
    resolved_provenance_id = provenance_id or ids()
    resolved_family_id = family_provenance_id or ids()
    if not all(
        _valid_ulid(item)
        for item in (resolved_artifact_id, resolved_provenance_id, resolved_family_id)
    ):
        raise QCValidationError("QC refusal artifact/provenance ids must be ULIDs")
    if not isinstance(instances, Sequence) or len(instances) == 0:
        raise QCValidationError("refusal instances must be a nonempty ordered sequence")
    input_refs: list[InputRef] = [
        InputRef(
            ref=f"source_manifest:{source_manifest_sha256}",
            kind="dataset",
            sha256_at_use=source_manifest_sha256,
        )
    ]
    digest_map = (
        dict(instance_digests) if isinstance(instance_digests, Mapping) else dict(instance_digests)
    )
    for instance in instances:
        if not isinstance(instance, DicomInstance):
            raise QCValidationError("refusal source inputs must be DicomInstance metadata")
        sop = getattr(instance, "sop_instance_uid", None)
        pixel = getattr(instance, "pixel_data", None)
        if not isinstance(sop, str) or not isinstance(pixel, (bytes, bytearray)):
            raise QCValidationError("refusal source inputs must be DicomInstance metadata")
        digest = digest_map.get(sop) or hashlib.sha256(bytes(pixel)).hexdigest()
        if not is_sha256_hex(digest):
            raise QCValidationError("refusal instance digest is invalid")
        input_refs.append(InputRef(ref=sop, kind="instance", sha256_at_use=digest))
    source_ref = source_provenance_ref or f"qc-refusal-source:{source_manifest_sha256}"
    records = list(provenance_records)
    if source_ref not in {record.provenance_id for record in records}:
        records.append(
            _root_provenance(
                provenance_id=source_ref,
                subject_ref=f"source_manifest:{source_manifest_sha256}",
                inputs=(input_refs[0],),
                at=at,
                created_by=created_by,
                code_commit_sha=code_commit_sha,
            )
        )
    result = _build_bundle(
        context=None,
        family_results=(
            FamilyResult(
                flag=family,
                metric_value=None,
                outcome=QCOutcome.ABSTAINED,
                abstention_code=refusal.reason_code,
            ),
        ),
        refusal=refusal,
        pair=None,
        subject_refs=(subject,),
        source_manifest_sha256=source_manifest_sha256,
        source_inputs=tuple(input_refs),
        source_volume=None,
        geometry_base=GeometryBase(
            geometry_id=f"qc_refusal_surface:{source_manifest_sha256}",
            geometry_version=1,
        ),
        provenance_records=tuple(records),
        upstream_refs=(source_ref,),
        artifact_id=resolved_artifact_id,
        aggregate_provenance_id=resolved_provenance_id,
        family_provenance_ids=(resolved_family_id,),
        supersedes_ref=supersedes_ref,
        claim_binding=claim_binding,
        release_lane=release_lane,
        created_by=created_by,
        created_at=created_at,
        code_commit_sha=code_commit_sha,
        software_version=software_version,
        artifact_store=artifact_store,
    )
    return result


__all__ = [
    "F2_5_EVIDENCE_WORDING",
    "UPSTREAM_REFUSAL_CODES",
    "assess_qc",
    "build_qc_assessment",
    "create_qc_assessment",
    "create_qc_input_refusal",
    "create_real_qc_assessment",
    "make_qc_assessment",
    "make_qc_input_refusal",
    "map_upstream_failure",
    "validate_grade_flag_matrix",
    "validate_result_bundle",
]
