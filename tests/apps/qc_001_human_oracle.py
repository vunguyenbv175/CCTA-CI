"""Independent QC_001-HUMAN expectations (test-only; NO ``core.qc`` import).

The oracle derives the expected state, refs, statuses, provenance records,
event list, digests, refusal codes, and the effective assessment of each
Part 4.4 case ONLY from the fixture definitions in
``qc_001_human_fixtures`` and the literals transcribed there.  It never
imports or calls the production QC package; a mechanical architecture test
proves the import boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Final

from core.digest import canonical_digest
from core.enums import ActorType, ProvenanceEventType, QualityGrade
from core.provenance import (
    Actor,
    BlobRef,
    Environment,
    HumanEdit,
    InputRef,
    NotApplicable,
    ProvenanceRecord,
    create_provenance,
)
from tests.apps.qc_001_d_fixtures import ulid
from tests.apps.qc_001_human_fixtures import (
    ACTOR_ID_HUMAN,
    ACTOR_TYPE_VALUE,
    CREATED_BY_VALUE,
    EVENT_ACTOR_ROLE,
    EVENT_ENTITY,
    EVENT_HUMAN_CORRECTION_VALUE,
    EVENT_INITIAL_HUMAN_VALUE,
    EVENT_TYPE_STATUS,
    FIXED_AT,
    GEOMETRY_STEP,
    HUMAN_EVIDENCE_KIND,
    HUMAN_EVIDENCE_SUBJECT_PREFIX,
    HUMAN_JUSTIFICATION_CODE,
    PARAMETERS_NOT_APPLICABLE_REASON,
    PAYLOAD_SCHEMA_VALUE,
    PREPROCESSING_VALUE,
    PRODUCER_VALUE,
    SCOPE_VALUE,
    SOFTWARE_VERSION_VALUE,
    SOURCE_ROOT_PREFIX,
    STALE_REASON_TEMPLATE,
    STATUS_ACTIVE,
    STATUS_STALE,
    STATUS_SUPERSEDED,
    ZERO_SHA,
    HumanCase,
    HumanStep,
    SourceIdentity,
)

#: The event-type literal required for each request shape.
_EVENT_BY_CORRECTION: Final[dict[bool, str]] = {
    False: EVENT_INITIAL_HUMAN_VALUE,
    True: EVENT_HUMAN_CORRECTION_VALUE,
}


def artifact_ref(seed: int) -> str:
    return f"qc_artifact:{ulid(seed)}"


def aggregate_id(seed: int) -> str:
    return ulid(seed + 1)


def human_event_id(seed: int) -> str:
    return ulid(seed + 2)


@dataclass(frozen=True, slots=True)
class EventSpec:
    """One expected append-only StateEvent; ``at`` is only ISO-validated."""

    entity: str
    entity_id: str
    event_type: str
    from_value: str
    to_value: str
    actor_role: str
    evidence_ref: str


@dataclass(frozen=True, slots=True)
class StepExpectation:
    """The expected outcome of one assessment recipe step."""

    step: HumanStep
    correction: bool
    artifact_id: str
    artifact_ref: str
    aggregate_provenance_id: str
    human_event_provenance_id: str
    prior: StepExpectation | None
    grade: QualityGrade
    flags: tuple[str, ...]
    event_type: ProvenanceEventType
    supersedes_ref: str | None


def step_expectation(
    step: HumanStep,
    *,
    prior: StepExpectation | None,
    case: HumanCase,
    index: int,
) -> StepExpectation:
    """Derive the expected outcome of one recipe step from the fixtures."""

    correction = step.supersedes is not None
    seed = step.reuse_seed_of if step.reuse_seed_of is not None else step.seed
    return StepExpectation(
        step=step,
        correction=correction,
        artifact_id=ulid(seed),
        artifact_ref=f"qc_artifact:{ulid(seed)}",
        aggregate_provenance_id=aggregate_id(step.seed),
        human_event_provenance_id=human_event_id(step.seed),
        prior=prior,
        grade=QualityGrade(step.grade),
        flags=tuple(step.flags),
        event_type=ProvenanceEventType(
            step.event_type_value
            if step.event_type_value is not None
            else _EVENT_BY_CORRECTION[correction]
        ),
        supersedes_ref=None if prior is None or not correction else prior.artifact_ref,
    )


def step_identity(step: HumanStep, identity: SourceIdentity) -> SourceIdentity:
    """Resolve the source identity a step binds.

    A descendant step (frozen case 8) binds a REAL descendant canonical
    volume identity — a new volume id and volume provenance ref over the same
    content-addressed source bytes — so the predecessor and the successor
    never bind the same volume member.  Every other step binds the case's
    original canonical volume.
    """

    if not step.descendant:
        return identity
    descendant_volume_id = ulid(step.descendant_volume_id_seed)
    return replace(
        identity,
        volume_id=descendant_volume_id,
        volume_provenance_ref=ulid(step.descendant_volume_provenance_seed),
        geometry_id=f"volume_geometry:{descendant_volume_id}",
    )


def expected_input_refs(identity: SourceIdentity) -> tuple[InputRef, ...]:
    """The ordered source tuple bound by both human records.

    Construction rule transcribed from the benchmark contract: the dataset
    manifest ref, one ``instance`` per ordered SOP ref at use, the volume
    ref, and the FrameSpec/transform artifact entries.
    """

    values: list[InputRef] = [
        InputRef(
            ref=f"source_manifest:{identity.source_manifest_sha256}",
            kind="dataset",
            sha256_at_use=identity.source_manifest_sha256,
        )
    ]
    values.extend(
        InputRef(ref=sop, kind="instance", sha256_at_use=digest)
        for sop, digest in identity.instances
    )
    values.append(
        InputRef(
            ref=f"volume_id:{identity.volume_id}",
            kind="volume",
            sha256_at_use=identity.volume_blob_sha256,
        )
    )
    values.append(
        InputRef(
            ref=f"frame_spec:{identity.frame_digest}",
            kind="artifact",
            sha256_at_use=identity.frame_digest,
        )
    )
    values.append(
        InputRef(
            ref=identity.transform_ref,
            kind="artifact",
            sha256_at_use=identity.transform_sha256,
        )
    )
    return tuple(values)


def source_root_id(identity: SourceIdentity) -> str:
    return f"{SOURCE_ROOT_PREFIX}:{identity.source_manifest_sha256}"


def expected_upstreams_aggregate(
    step: StepExpectation, identity: SourceIdentity
) -> tuple[str, ...]:
    """(source-manifest, volume, FrameSpec/transform, prior assessment)."""

    upstreams = [
        source_root_id(identity),
        identity.volume_provenance_ref,
        identity.transform_provenance_ref,
    ]
    if step.prior is not None and step.correction:
        upstreams.append(step.prior.aggregate_provenance_id)
    return tuple(upstreams)


def expected_upstreams_human_event(
    step: StepExpectation, identity: SourceIdentity
) -> tuple[str, ...]:
    """(prior assessment for corrections, plus source/volume/FrameSpec)."""

    upstreams: list[str] = []
    if step.prior is not None and step.correction:
        upstreams.append(step.prior.aggregate_provenance_id)
    upstreams.extend(
        [
            source_root_id(identity),
            identity.volume_provenance_ref,
            identity.transform_provenance_ref,
        ]
    )
    return tuple(upstreams)


def expected_evidence_document(
    step: StepExpectation, case: HumanCase, identity: SourceIdentity
) -> dict[str, object]:
    """The frozen closed human evidence document."""

    return {
        "evidence_kind": HUMAN_EVIDENCE_KIND,
        "actor_type": ACTOR_TYPE_VALUE,
        "actor_id": ACTOR_ID_HUMAN,
        "editor": ACTOR_ID_HUMAN,
        "justification_code": HUMAN_JUSTIFICATION_CODE,
        "event_type": step.event_type.value,
        "quality_grade": step.grade.value,
        "flags": step.flags,
        "assessment_ref": step.artifact_ref,
        "supersedes_ref": step.supersedes_ref,
        "source_manifest_sha256": identity.source_manifest_sha256,
        "volume_id": identity.volume_id,
        "frame_digest": identity.frame_digest,
        "edit_summary": case.edit_summary,
        "edited_at": FIXED_AT,
    }


@dataclass(frozen=True, slots=True)
class HumanRecordPair:
    """The two expected §44 records plus their evidence document."""

    aggregate: ProvenanceRecord
    human_event: ProvenanceRecord
    evidence_document: dict[str, object]
    evidence_subject_ref: str


def expected_records(
    step: StepExpectation,
    case: HumanCase,
    identity: SourceIdentity,
) -> HumanRecordPair:
    """Rebuild both expected §44 records with the transcribed literals."""

    evidence_document = expected_evidence_document(step, case, identity)
    human_edit = HumanEdit(
        editor=ACTOR_ID_HUMAN,
        edit_summary=str(evidence_document["edit_summary"]),
        at=str(evidence_document["edited_at"]),
    )
    environment = Environment(
        software_version=SOFTWARE_VERSION_VALUE, code_commit_sha=ZERO_SHA
    )
    parameters: NotApplicable[BlobRef] = NotApplicable(
        value=None, reason=PARAMETERS_NOT_APPLICABLE_REASON
    )
    actor = Actor(actor_type=ActorType.HUMAN, actor_id=ACTOR_ID_HUMAN)
    inputs = expected_input_refs(identity)
    aggregate = create_provenance(
        subject_ref=step.artifact_id,
        event_type=step.event_type,
        actor=actor,
        inputs=inputs,
        preprocessing=PREPROCESSING_VALUE,
        parameters_ref=parameters,
        human_edit=human_edit,
        upstream_refs=expected_upstreams_aggregate(step, identity),
        environment=environment,
        model_record_ref=None,
        created_by=CREATED_BY_VALUE,
        at=FIXED_AT,
        provenance_id=step.aggregate_provenance_id,
    )
    evidence_digest = canonical_digest(evidence_document)
    human_event = create_provenance(
        subject_ref=f"{HUMAN_EVIDENCE_SUBJECT_PREFIX}:{evidence_digest}",
        event_type=step.event_type,
        actor=actor,
        inputs=inputs,
        preprocessing=PREPROCESSING_VALUE,
        parameters_ref=parameters,
        human_edit=human_edit,
        upstream_refs=expected_upstreams_human_event(step, identity),
        environment=environment,
        model_record_ref=None,
        created_by=CREATED_BY_VALUE,
        at=FIXED_AT,
        provenance_id=step.human_event_provenance_id,
    )
    return HumanRecordPair(
        aggregate=aggregate,
        human_event=human_event,
        evidence_document=evidence_document,
        evidence_subject_ref=f"{HUMAN_EVIDENCE_SUBJECT_PREFIX}:{evidence_digest}",
    )


def expected_payload_document(
    step: StepExpectation, identity: SourceIdentity
) -> dict[str, object]:
    """The exact thirteen-key v1 payload document."""

    return {
        "schema_version": PAYLOAD_SCHEMA_VALUE,
        "scope": SCOPE_VALUE,
        "subject_refs": (identity.study_ref,),
        "quality_grade": step.grade.value,
        "confidence": None,
        "flags": step.flags,
        "vessel_id": None,
        "region_id": None,
        "producer": PRODUCER_VALUE,
        "produced_by_ref": f"provenance:{step.aggregate_provenance_id}",
        "supersedes_ref": step.supersedes_ref,
        "justification_code": HUMAN_JUSTIFICATION_CODE,
        "evidence_refs": (
            f"provenance:{step.aggregate_provenance_id}",
            f"source_manifest:{identity.source_manifest_sha256}",
            f"volume_id:{identity.volume_id}",
            f"provenance:{step.human_event_provenance_id}",
        ),
    }


def expected_step_events(
    step: StepExpectation, prior: StepExpectation | None, identity: SourceIdentity
) -> tuple[EventSpec, ...]:
    """The exact append-only events one step appends.

    A geometry advance marks the pinned assessment STALE first; a
    correction then supersedes it (from_value ``STALE``).  Every other
    correction supersedes an ``ACTIVE`` predecessor.  Assessment creation
    itself appends no birth event.
    """

    events: list[EventSpec] = []
    advance = step.step.advance_geometry_to
    if advance and prior is not None:
        events.append(
            EventSpec(
                entity=EVENT_ENTITY,
                entity_id=prior.artifact_id,
                event_type=EVENT_TYPE_STATUS,
                from_value=STATUS_ACTIVE,
                to_value=STATUS_STALE,
                actor_role=EVENT_ACTOR_ROLE,
                # The STALE propagation names the PRIOR geometry lineage: the
                # descendant volume's own geometry id is bound by the new
                # record, not by the staleness of the old one.
                evidence_ref=STALE_REASON_TEMPLATE.format(
                    geometry_id=identity.geometry_id
                ),
            )
        )
    if step.correction and prior is not None:
        events.append(
            EventSpec(
                entity=EVENT_ENTITY,
                entity_id=prior.artifact_id,
                event_type=EVENT_TYPE_STATUS,
                from_value=STATUS_STALE if advance else STATUS_ACTIVE,
                to_value=STATUS_SUPERSEDED,
                actor_role=EVENT_ACTOR_ROLE,
                evidence_ref=step.artifact_id,
            )
        )
    return tuple(events)


@dataclass(frozen=True, slots=True)
class ValidCaseExpectation:
    """Everything the benchmark asserts for one valid case."""

    steps: tuple[StepExpectation, ...]
    events: tuple[EventSpec, ...]
    statuses: dict[str, str]
    effective_ref: str


def expected_valid_case(
    case: HumanCase, identity: SourceIdentity
) -> ValidCaseExpectation:
    """Derive the complete expected outcome of one valid case."""

    if not case.valid:
        raise ValueError(f"case {case.case_id} is not a valid case")
    steps: list[StepExpectation] = []
    events: list[EventSpec] = []
    statuses: dict[str, str] = {}
    prior: StepExpectation | None = None
    for step in case.steps:
        if step.producer == GEOMETRY_STEP:
            # A pure geometry advance appends only the STALE propagation of
            # the currently effective assessment; no assessment is created.
            if prior is None:
                raise ValueError("a geometry advance requires a prior assessment")
            events.extend(
                expected_step_events(
                    StepExpectation(
                        step=step,
                        correction=False,
                        artifact_id="",
                        artifact_ref="",
                        aggregate_provenance_id="",
                        human_event_provenance_id="",
                        prior=prior,
                        grade=QualityGrade(step.grade),
                        flags=tuple(step.flags),
                        event_type=ProvenanceEventType(_EVENT_BY_CORRECTION[False]),
                        supersedes_ref=None,
                    ),
                    prior,
                    identity,
                )
            )
            statuses[prior.artifact_id] = STATUS_STALE
            continue
        expectation = step_expectation(step, prior=prior, case=case, index=len(steps))
        # The geometry advance marks the PRIOR (pre-descendant) geometry stale;
        # record derivation uses the identity the step actually binds.
        events.extend(expected_step_events(expectation, prior, identity))
        if expectation.prior is not None and expectation.correction:
            statuses[expectation.prior.artifact_id] = STATUS_SUPERSEDED
        steps.append(expectation)
        prior = expectation
    if not steps:
        raise ValueError("a case must create at least one assessment")
    final = steps[-1]
    statuses[final.artifact_id] = STATUS_ACTIVE
    return ValidCaseExpectation(
        steps=tuple(steps),
        events=tuple(events),
        statuses=statuses,
        effective_ref=final.artifact_ref,
    )


def expected_refusal_code(case: HumanCase) -> str:
    if case.refusal_code is None:
        raise ValueError(f"case {case.case_id} is not a forbidden case")
    return case.refusal_code


def expected_baseline(
    case: HumanCase, identity: SourceIdentity
) -> ValidCaseExpectation:
    """Derive the pre-attempt state of a forbidden case (its baseline)."""

    return expected_valid_case(
        HumanCase(
            ordinal=case.ordinal,
            case_id=case.case_id,
            label=case.label,
            valid=True,
            steps=case.steps,
        ),
        identity,
    )


__all__ = [
    "EventSpec",
    "HumanRecordPair",
    "StepExpectation",
    "ValidCaseExpectation",
    "aggregate_id",
    "artifact_ref",
    "expected_baseline",
    "expected_evidence_document",
    "expected_input_refs",
    "expected_payload_document",
    "expected_records",
    "expected_refusal_code",
    "expected_step_events",
    "expected_upstreams_aggregate",
    "expected_upstreams_human_event",
    "expected_valid_case",
    "human_event_id",
    "source_root_id",
    "step_expectation",
    "step_identity",
]
