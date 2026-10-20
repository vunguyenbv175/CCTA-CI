"""``DerivedArtifact`` - every derived product of the platform (Part B §3.4).

One entity distinguished by ``kind``; creation-time validation enforces the
producer-kind restriction matrix (Rule D), geometry-base binding (H-010),
registration requirement (H-009), claim binding (F1 §2a) and the export
gate (H-011). Review/state changes flow only through append-only events.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from dataclasses import fields as dc_fields

from core.digest import require_sha256_hex
from core.enums import (
    ActorType,
    ArtifactKind,
    ArtifactStatus,
    Confidence,
    ProducerType,
    QualityGrade,
    ReleaseLane,
    ReviewState,
)
from core.errors import (
    CanonicalValidationError,
    ExportGateViolationError,
    GeometryBaseRequiredError,
    RegistrationRequiredError,
    StateTransitionError,
)
from core.events import EventLog, Projection, StateEvent, utc_now_iso, validate_iso8601_utc
from core.ids import new_ulid
from core.provenance import InvalidationDAG

ARTIFACT_SCHEMA = "ccta.canonical.derived_artifact.v1"

#: Producer-kind restriction matrix (Luna fix #5, Master Plan Rule D).
#: LLM is forbidden everywhere except REPORT_DRAFT; MEASUREMENT_SET accepts
#: deterministic engines only; physics results come from PHYSICS_BACKEND.
PRODUCER_MATRIX: dict[ArtifactKind, frozenset[ProducerType]] = {
    ArtifactKind.SEGMENTATION: frozenset({ProducerType.MODEL, ProducerType.HUMAN}),
    ArtifactKind.CENTERLINE: frozenset(
        {ProducerType.DETERMINISTIC_ENGINE, ProducerType.HUMAN}
    ),
    ArtifactKind.MESH: frozenset(
        {
            ProducerType.MODEL,
            ProducerType.DETERMINISTIC_ENGINE,
            ProducerType.HUMAN,
        }
    ),
    ArtifactKind.SURFACE: frozenset(
        {
            ProducerType.MODEL,
            ProducerType.DETERMINISTIC_ENGINE,
            ProducerType.HUMAN,
        }
    ),
    ArtifactKind.LANDMARK_SET: frozenset(
        {
            ProducerType.MODEL,
            ProducerType.DETERMINISTIC_ENGINE,
            ProducerType.HUMAN,
        }
    ),
    ArtifactKind.CORONARY_GRAPH: frozenset(
        {
            ProducerType.MODEL,
            ProducerType.DETERMINISTIC_ENGINE,
        }
    ),
    ArtifactKind.LESION_SET: frozenset(
        {ProducerType.DETERMINISTIC_ENGINE, ProducerType.RULE_ENGINE}
    ),
    ArtifactKind.MEASUREMENT_SET: frozenset({ProducerType.DETERMINISTIC_ENGINE}),
    ArtifactKind.QC_ASSESSMENT: frozenset(
        {
            ProducerType.RULE_ENGINE,
            ProducerType.DETERMINISTIC_ENGINE,
            ProducerType.MODEL,
            ProducerType.HUMAN,
        }
    ),
    ArtifactKind.CLASSIFICATION_CANDIDATE: frozenset(
        {ProducerType.MODEL, ProducerType.HUMAN}
    ),
    ArtifactKind.PHYSIOLOGY_RESULT: frozenset({ProducerType.PHYSICS_BACKEND}),
    ArtifactKind.PHYSIOLOGY_SURROGATE: frozenset({ProducerType.MODEL}),
    ArtifactKind.REPORT_DRAFT: frozenset(
        {ProducerType.LLM, ProducerType.HUMAN, ProducerType.DETERMINISTIC_ENGINE}
    ),
    ArtifactKind.COORDINATE_TRANSFORM: frozenset({ProducerType.DETERMINISTIC_ENGINE}),
    ArtifactKind.REGISTRATION: frozenset({ProducerType.DETERMINISTIC_ENGINE}),
    ArtifactKind.CORRECTION: frozenset({ProducerType.HUMAN}),
    ArtifactKind.EXPORT_PACKAGE: frozenset({ProducerType.DETERMINISTIC_ENGINE}),
    ArtifactKind.MPR_VIEW: frozenset({ProducerType.DETERMINISTIC_ENGINE}),
}

#: Kinds that pin lineage.geometry_base for the invalidation DAG (H-010).
GEOMETRY_DEPENDENT_KINDS: frozenset[ArtifactKind] = frozenset(
    {
        ArtifactKind.SEGMENTATION,
        ArtifactKind.CENTERLINE,
        ArtifactKind.MESH,
        ArtifactKind.SURFACE,
        ArtifactKind.LANDMARK_SET,
        ArtifactKind.CORONARY_GRAPH,
        ArtifactKind.LESION_SET,
        ArtifactKind.MEASUREMENT_SET,
        ArtifactKind.QC_ASSESSMENT,
        ArtifactKind.PHYSIOLOGY_RESULT,
        ArtifactKind.PHYSIOLOGY_SURROGATE,
        ArtifactKind.REPORT_DRAFT,
    }
)

#: Rigid review lifecycle (Master Plan §12, §43).
LEGAL_REVIEW_TRANSITIONS: dict[ReviewState, frozenset[ReviewState]] = {
    ReviewState.PREDICTED: frozenset({ReviewState.ACCEPTED, ReviewState.REJECTED}),
    ReviewState.ACCEPTED: frozenset(
        {ReviewState.CORRECTED, ReviewState.CONFIRMED_FINAL}
    ),
    ReviewState.CORRECTED: frozenset(
        {ReviewState.ACCEPTED, ReviewState.CONFIRMED_FINAL, ReviewState.REJECTED}
    ),
    ReviewState.REJECTED: frozenset(),
    ReviewState.CONFIRMED_FINAL: frozenset(),
}


# ---------------------------------------------------------------------------
# Component blocks
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PayloadRef:
    """Canonical payload schema + content-addressed blob."""

    payload_schema: str
    blob_sha256: str
    codec: str = "json"
    inline_summary: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class GeometryBase:
    """Invalidation-DAG key: pinned upstream geometry id + version."""

    geometry_id: str
    geometry_version: int


@dataclass(frozen=True)
class Lineage:
    """Input DAG of an artifact (P5, §44 upstream dependencies)."""

    input_volume_refs: tuple[str, ...] = ()
    input_artifact_refs: tuple[str, ...] = ()
    input_series_refs: tuple[str, ...] = ()
    geometry_base: GeometryBase | None = None
    registration_ref: str | None = None


@dataclass(frozen=True)
class ClaimBinding:
    """Claim-registry link enforced at creation (F1 §2a, Luna fix #1)."""

    claim_id: str
    claim_version: str
    population_ref: str | None = None
    metric_refs: tuple[str, ...] = ()
    snapshot_ref: str | None = None


@dataclass(frozen=True)
class Producer:
    """Producer block of an artifact."""

    producer_type: ProducerType
    model_record_ref: str | None = None
    algorithm: str | None = None
    algorithm_version: str | None = None
    parameters_blob_sha256: str | None = None
    adapter_ref: str | None = None


@dataclass(frozen=True)
class ReviewTransition:
    """One immutable entry of review.state_history."""

    from_state: ReviewState | None
    to_state: ReviewState
    actor_role: ActorType
    at: str


@dataclass(frozen=True)
class ReviewBlock:
    """Review lifecycle block; transitions only via store events."""

    review_state: ReviewState
    reviewed_by: str | None = None
    reviewed_at: str | None = None
    state_history: tuple[ReviewTransition, ...] = ()
    supersedes_ref: str | None = None


@dataclass(frozen=True)
class QualityBlock:
    """§9/§41 quality signals."""

    quality_grade: QualityGrade | None = None
    confidence: Confidence | None = None
    flags: tuple[str, ...] = ()


@dataclass(frozen=True)
class DerivedArtifact:
    """The universal derived record (Part B §3.4)."""

    artifact_id: str
    study_ref: str
    kind: ArtifactKind
    payload: PayloadRef
    lineage: Lineage
    claim_binding: ClaimBinding
    producer: Producer
    review: ReviewBlock
    quality: QualityBlock
    release_lane: ReleaseLane
    provenance_ref: str
    status: ArtifactStatus = ArtifactStatus.ACTIVE
    invalidated_reason: str | None = None
    stale_reason: str | None = None
    schema_version: str = ARTIFACT_SCHEMA
    created_by: str = ""
    created_at: str = ""
    extra_guard: tuple[()] = field(default=(), repr=False, compare=False)


# ---------------------------------------------------------------------------
# Creation-time validation (fail-closed)
# ---------------------------------------------------------------------------


def create_derived_artifact(
    *,
    study_ref: str,
    kind: ArtifactKind,
    payload: PayloadRef,
    lineage: Lineage,
    claim_binding: ClaimBinding,
    producer: Producer,
    release_lane: ReleaseLane,
    provenance_ref: str,
    quality: QualityBlock | None = None,
    review: ReviewBlock | None = None,
    status: ArtifactStatus = ArtifactStatus.ACTIVE,
    invalidated_reason: str | None = None,
    stale_reason: str | None = None,
    created_by: str = "",
    created_at: str | None = None,
    artifact_id: str | None = None,
) -> DerivedArtifact:
    """Validate every §3.4 rule; raise listing ALL violations."""
    violations: list[str] = []
    stamp = created_at or utc_now_iso()
    try:
        validate_iso8601_utc(stamp, "created_at")
    except ValueError as exc:
        violations.append(str(exc))
    if not study_ref:
        violations.append("study_ref is required")
    if not created_by:
        violations.append("created_by is required")
    if not provenance_ref:
        violations.append("provenance_ref is REQUIRED (P4/H-007)")

    # payload ------------------------------------------------------------
    try:
        require_sha256_hex(payload.blob_sha256, "payload.blob_sha256")
    except ValueError as exc:
        violations.append(str(exc))
    if not (
        payload.payload_schema.startswith("ccta.payload.")
        or payload.payload_schema.startswith("ccta.mpr.view.v1")
        or payload.payload_schema == "ccta.mpr.view.v2"
    ):
        violations.append(
            "payload_schema must be a ccta.payload.* or ccta.mpr.view.v1 "
            "or ccta.mpr.view.v2 URI, "
            f"got {payload.payload_schema!r}"
        )

    # claim binding (F1 §2a) ----------------------------------------------
    if not claim_binding.claim_id.startswith("CLM-"):
        violations.append(f"claim_id must reference the claim registry: {claim_binding.claim_id!r}")
    if not claim_binding.claim_version:
        violations.append("claim_binding.claim_version is required")

    # producer-kind restriction matrix ------------------------------------
    allowed = PRODUCER_MATRIX.get(kind)
    if allowed is None:
        violations.append(f"no producer matrix row for kind {kind}")
    elif producer.producer_type not in allowed:
        violations.append(
            f"producer_type={producer.producer_type.value} not allowed for "
            f"kind={kind.value} (allowed: "
            f"{sorted(p.value for p in allowed)})"
        )
    if producer.producer_type == ProducerType.LLM and kind != ArtifactKind.REPORT_DRAFT:
        violations.append("LLM producers are forbidden outside REPORT_DRAFT (Rule D)")
    if producer.producer_type == ProducerType.MODEL and not producer.model_record_ref:
        violations.append("producer_type=MODEL requires model_record_ref")
    if producer.producer_type in (
        ProducerType.DETERMINISTIC_ENGINE,
        ProducerType.PHYSICS_BACKEND,
    ) and not producer.parameters_blob_sha256:
        violations.append(
            f"producer_type={producer.producer_type.value} requires frozen parameters"
        )
    else:
        try:
            if producer.parameters_blob_sha256 is not None:
                require_sha256_hex(
                    producer.parameters_blob_sha256, "producer.parameters_blob_sha256"
                )
        except ValueError as exc:
            violations.append(str(exc))
    if producer.producer_type == ProducerType.ADAPTER and not producer.adapter_ref:
        violations.append("producer_type=ADAPTER requires adapter_ref")

    # geometry base (H-010) -------------------------------------------------
    if kind in GEOMETRY_DEPENDENT_KINDS and lineage.geometry_base is None:
        raise GeometryBaseRequiredError(
            f"kind={kind.value} depends on geometry and requires lineage.geometry_base"
        )
    if lineage.geometry_base is not None and lineage.geometry_base.geometry_version < 1:
        violations.append("geometry_base.geometry_version starts at 1")

    # registration rule (H-009) ---------------------------------------------
    if len(set(lineage.input_series_refs)) > 1 and not lineage.registration_ref:
        raise RegistrationRequiredError(
            "artifact reads >1 series/phase without lineage.registration_ref (H-009)"
        )

    # measurement determinism (Rule C / §6.4) --------------------------------
    if kind == ArtifactKind.MEASUREMENT_SET:
        if not lineage.input_artifact_refs:
            violations.append(
                "MEASUREMENT_SET must reference its source geometry artifacts"
            )

    # kind-specific lifecycle rules ------------------------------------------
    effective_review = review or ReviewBlock(review_state=ReviewState.PREDICTED)
    if kind == ArtifactKind.CLASSIFICATION_CANDIDATE and (
        effective_review.review_state != ReviewState.PREDICTED
    ):
        violations.append("CLASSIFICATION_CANDIDATE is always born PREDICTED")
    if kind == ArtifactKind.CORRECTION and effective_review.supersedes_ref is None:
        violations.append("CORRECTION requires review.supersedes_ref (H-004)")
    if status == ArtifactStatus.INVALIDATED and not invalidated_reason:
        violations.append("invalidated_reason is required when status=INVALIDATED")
    if status == ArtifactStatus.STALE and not stale_reason:
        violations.append("stale_reason is required when status=STALE")
    if not isinstance(release_lane, ReleaseLane):
        violations.append("release_lane is required (P6)")

    if violations:
        raise CanonicalValidationError(violations)

    return DerivedArtifact(
        artifact_id=artifact_id or new_ulid(),
        study_ref=study_ref,
        kind=kind,
        payload=payload,
        lineage=lineage,
        claim_binding=claim_binding,
        producer=producer,
        review=effective_review,
        quality=quality or QualityBlock(),
        release_lane=release_lane,
        provenance_ref=provenance_ref,
        status=status,
        invalidated_reason=invalidated_reason,
        stale_reason=stale_reason,
        created_by=created_by,
        created_at=stamp,
    )


# ---------------------------------------------------------------------------
# Append-only store with DAG-aware gates
# ---------------------------------------------------------------------------


class DerivedArtifactStore:
    """Registry of artifacts; all state changes ride the shared event log.

    Guarantees:
    - no setter mutates a stored artifact - every transition appends an
      event and swaps in a rebuilt frozen record (I13);
    - ``invalidate`` fans out STALE through the invalidation DAG;
    - export packages can only bind CONFIRMED_FINAL, active inputs.
    """

    def __init__(
        self,
        log: EventLog,
        dag: InvalidationDAG,
    ) -> None:
        self._log = log
        self._dag = dag
        self._artifacts: dict[str, DerivedArtifact] = {}

    # -- read access ---------------------------------------------------------

    def get(self, artifact_id: str) -> DerivedArtifact:
        try:
            return self._artifacts[artifact_id]
        except KeyError:
            raise KeyError(f"unknown artifact {artifact_id}") from None

    def __contains__(self, artifact_id: object) -> bool:
        return isinstance(artifact_id, str) and artifact_id in self._artifacts

    def __len__(self) -> int:
        return len(self._artifacts)

    @property
    def event_log(self) -> EventLog:
        return self._log

    @property
    def invalidation_dag(self) -> InvalidationDAG:
        return self._dag

    # -- write path ------------------------------------------------------------

    def add(self, artifact: DerivedArtifact) -> DerivedArtifact:
        """Register a validated artifact and seed its event-log state."""
        if artifact.artifact_id in self._artifacts:
            raise CanonicalValidationError(
                f"artifact {artifact.artifact_id} already exists (append-only)"
            )
        if artifact.kind == ArtifactKind.EXPORT_PACKAGE:
            self._enforce_export_gate(artifact)
        self._artifacts[artifact.artifact_id] = artifact
        self._log.seed(
            entity="derived_artifact",
            entity_id=artifact.artifact_id,
            initial_fields={
                "review_state": artifact.review.review_state.value,
                "status": artifact.status.value,
                "release_lane": artifact.release_lane.value,
            },
        )
        if artifact.lineage.geometry_base is not None:
            base = artifact.lineage.geometry_base
            try:
                self._dag.register_geometry(base.geometry_id, max(base.geometry_version, 1))
            except ValueError:
                pass  # already tracked at >= this version; binding pins its own target
            self._dag.bind_dependent(
                artifact.artifact_id,
                base.geometry_id,
                base.geometry_version,
            )
        return artifact

    def _enforce_export_gate(self, artifact: DerivedArtifact) -> None:
        """EXPORT_PACKAGE binds only CONFIRMED_FINAL + ACTIVE inputs (H-011)."""
        problems: list[str] = []
        for ref in artifact.lineage.input_artifact_refs:
            try:
                source = self.get(ref)
            except KeyError:
                problems.append(f"export input {ref!r} does not exist")
                continue
            if source.review.review_state != ReviewState.CONFIRMED_FINAL:
                problems.append(
                    f"export input {ref} is {source.review.review_state.value}, "
                    "only CONFIRMED_FINAL may be exported"
                )
            if source.status != ArtifactStatus.ACTIVE:
                problems.append(
                    f"export input {ref} has status {source.status.value}; "
                    "STALE/SUPERSEDED/INVALIDATED data is blocked from export"
                )
        if problems:
            raise ExportGateViolationError(problems)

    # -- event-driven transitions ---------------------------------------------

    def apply_review_transition(
        self,
        artifact_id: str,
        to_state: ReviewState,
        *,
        actor_role: ActorType,
        reviewed_by: str | None = None,
        evidence_ref: str | None = None,
        at: str | None = None,
    ) -> DerivedArtifact:
        """Append a REVIEW event and rebuild the artifact (I13)."""
        artifact = self.get(artifact_id)
        current = artifact.review.review_state
        allowed = LEGAL_REVIEW_TRANSITIONS.get(current, frozenset())
        if to_state not in allowed:
            raise StateTransitionError(
                f"illegal review transition {current.value} -> {to_state.value}"
            )
        if to_state == ReviewState.CONFIRMED_FINAL and actor_role != ActorType.HUMAN:
            raise StateTransitionError(
                "->CONFIRMED_FINAL requires an explicit HUMAN actor (§43)"
            )
        stamp = at or utc_now_iso()
        self._log.append_event(
            StateEvent(
                entity="derived_artifact",
                entity_id=artifact_id,
                event_type="REVIEW",
                from_value=current.value,
                to_value=to_state.value,
                actor_role=actor_role.value,
                at=stamp,
                evidence_ref=evidence_ref,
            )
        )
        history = artifact.review.state_history + (
            ReviewTransition(
                from_state=current, to_state=to_state, actor_role=actor_role, at=stamp
            ),
        )
        updated_review = ReviewBlock(
            review_state=to_state,
            reviewed_by=reviewed_by or artifact.review.reviewed_by,
            reviewed_at=stamp,
            state_history=history,
            supersedes_ref=artifact.review.supersedes_ref,
        )
        kwargs = {f.name: getattr(artifact, f.name) for f in dc_fields(DerivedArtifact)}
        kwargs["review"] = updated_review
        updated = DerivedArtifact(**kwargs)
        self._artifacts[artifact_id] = updated
        return updated

    def mark_superseded(self, artifact_id: str, *, superseded_by: str) -> DerivedArtifact:
        """Mark the predecessor SUPERSEDED when a correction/new run lands."""
        artifact = self.get(artifact_id)
        if artifact.status not in (ArtifactStatus.ACTIVE, ArtifactStatus.STALE):
            raise StateTransitionError(
                f"cannot supersede artifact in status {artifact.status.value}"
            )
        stamp = utc_now_iso()
        self._log.append_event(
            StateEvent(
                entity="derived_artifact",
                entity_id=artifact_id,
                event_type="STATUS",
                from_value=artifact.status.value,
                to_value=ArtifactStatus.SUPERSEDED.value,
                actor_role="ENGINE",
                at=stamp,
                evidence_ref=superseded_by,
            )
        )
        return self._replace(artifact, status=ArtifactStatus.SUPERSEDED)

    # -- invalidation fan-out (H-010) -------------------------------------------

    def register_new_geometry_version(self, geometry_id: str, new_version: int) -> tuple[str, ...]:
        """Advance a geometry; dependents of older versions become STALE."""
        self._dag.register_geometry(geometry_id, new_version)
        return self._apply_stale_propagation(geometry_id)

    def invalidate(self, artifact_id: str, *, reason: str) -> tuple[str, ...]:
        """INVALIDATED an artifact and propagate STALE to its dependents."""
        artifact = self.get(artifact_id)
        if not reason:
            raise CanonicalValidationError("invalidation requires a reason")
        if artifact.status == ArtifactStatus.INVALIDATED:
            return ()
        stamp = utc_now_iso()
        self._log.append_event(
            StateEvent(
                entity="derived_artifact",
                entity_id=artifact_id,
                event_type="STATUS",
                from_value=artifact.status.value,
                to_value=ArtifactStatus.INVALIDATED.value,
                actor_role="ENGINE",
                at=stamp,
                evidence_ref=reason,
            )
        )
        self._replace(artifact, status=ArtifactStatus.INVALIDATED, reason=reason)
        self._dag.mark_invalidated(artifact_id)
        return self._apply_stale_propagation(artifact_id)

    def _apply_stale_propagation(self, geometry_id: str) -> tuple[str, ...]:
        newly_stale: list[str] = []
        for dependent_id in self._dag.stale_dependents(geometry_id):
            if dependent_id not in self._artifacts:
                continue
            dep = self._artifacts[dependent_id]
            if dep.status in (ArtifactStatus.STALE, ArtifactStatus.INVALIDATED):
                continue
            reason = (
                f"upstream geometry {geometry_id} invalidated or advanced beyond the "
                "version this artifact pinned (H-010)"
            )
            stamp = utc_now_iso()
            self._log.append_event(
                StateEvent(
                    entity="derived_artifact",
                    entity_id=dependent_id,
                    event_type="STATUS",
                    from_value=dep.status.value,
                    to_value=ArtifactStatus.STALE.value,
                    actor_role="ENGINE",
                    at=stamp,
                    evidence_ref=reason,
                )
            )
            self._artifacts[dependent_id] = self._replace(
                dep, status=ArtifactStatus.STALE, reason=reason
            )
            newly_stale.append(dependent_id)
        return tuple(newly_stale)

    # -- helpers ---------------------------------------------------------------

    def _replace(
        self,
        artifact: DerivedArtifact,
        *,
        status: ArtifactStatus,
        reason: str | None = None,
    ) -> DerivedArtifact:
        kwargs: dict[str, object] = {
            f.name: getattr(artifact, f.name) for f in dc_fields(DerivedArtifact)
        }
        kwargs["status"] = status
        if status == ArtifactStatus.INVALIDATED and reason:
            kwargs["invalidated_reason"] = reason
        if status == ArtifactStatus.STALE and reason:
            kwargs["stale_reason"] = reason
        updated = DerivedArtifact(**kwargs)  # type: ignore[arg-type]
        self._artifacts[updated.artifact_id] = updated
        return updated

    def project(self, artifact_id: str) -> Projection:
        """Read-only projection replayed from the event log (I13 invariant)."""
        return self._log.project_state(artifact_id, entity="derived_artifact")


def create_export_package(
    store: DerivedArtifactStore,
    *,
    study_ref: str,
    input_artifact_refs: tuple[str, ...],
    payload: PayloadRef,
    claim_binding: ClaimBinding,
    provenance_ref: str,
    release_lane: ReleaseLane,
    created_by: str,
    parameters_blob_sha256: str,
    input_volume_refs: tuple[str, ...] = (),
    input_series_refs: tuple[str, ...] = (),
    quality: QualityBlock | None = None,
) -> DerivedArtifact:
    """Build + register an EXPORT_PACKAGE under the confirmed-only gate.

    The gate itself runs inside :meth:`DerivedArtifactStore.add`; candidate
    or stale inputs make this raise ``ExportGateViolationError`` (H-011).
    """
    candidate = create_derived_artifact(
        study_ref=study_ref,
        kind=ArtifactKind.EXPORT_PACKAGE,
        payload=payload,
        lineage=Lineage(
            input_volume_refs=input_volume_refs,
            input_artifact_refs=input_artifact_refs,
            input_series_refs=input_series_refs,
        ),
        claim_binding=claim_binding,
        producer=Producer(
            producer_type=ProducerType.DETERMINISTIC_ENGINE,
            algorithm="export_packager",
            parameters_blob_sha256=parameters_blob_sha256,
        ),
        release_lane=release_lane,
        provenance_ref=provenance_ref,
        quality=quality,
        created_by=created_by,
    )
    return store.add(candidate)


__all__ = [
    "ARTIFACT_SCHEMA",
    "ClaimBinding",
    "DerivedArtifact",
    "DerivedArtifactStore",
    "GEOMETRY_DEPENDENT_KINDS",
    "GeometryBase",
    "Lineage",
    "PayloadRef",
    "Producer",
    "PRODUCER_MATRIX",
    "QualityBlock",
    "ReviewBlock",
    "ReviewTransition",
    "create_derived_artifact",
    "create_export_package",
]
