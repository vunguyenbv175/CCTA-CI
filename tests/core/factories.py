"""Shared deterministic builders for core tests.

Every factory produces fully valid objects; tests mutate one argument at a
time to prove fail-closed behavior.
"""

from __future__ import annotations

import hashlib
from typing import Any

from core.artifacts import (
    ClaimBinding,
    DerivedArtifactStore,
    GeometryBase,
    Lineage,
    PayloadRef,
    Producer,
    QualityBlock,
    ReviewBlock,
    create_derived_artifact,
)
from core.coordinates import CoordinateTransform, FrameSpec, SpaceKind
from core.enums import (
    ActorType,
    ArtifactKind,
    ArtifactStatus,
    ContrastClass,
    ContrastPresence,
    ExamType,
    Handedness,
    ImportSource,
    PrivacyClass,
    ProducerType,
    ProvenanceEventType,
    QualityGrade,
    ReleaseLane,
    ReviewState,
    SeriesClass,
    Units,
    VolumeKind,
)
from core.errors import CanonicalValidationError
from core.events import EventLog
from core.provenance import (
    Actor,
    BlobRef,
    Environment,
    InputRef,
    InvalidationDAG,
    NotApplicable,
    ProvenanceRecord,
    create_provenance,
)
from core.study import (
    InstanceEntry,
    OriginalDicomRef,
    create_study,
    register_series,
)
from core.volume import VolumeBlobRef, create_volume

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
COMMIT = "0123456789abcdef0123456789abcdef01234567"
AT = "2026-01-15T10:30:00Z"

_STUDY_SEQ = iter(range(1, 10_000))
_SERIES_SEQ = iter(range(1, 10_000))


def sha256_of(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------


def make_actor() -> Actor:
    return Actor(actor_type=ActorType.ENGINE, actor_id="deterministic-measurer")


def make_inputs(count: int = 1) -> tuple[InputRef, ...]:
    return tuple(
        InputRef(
            ref=f"series-{i + 1}", kind="series", sha256_at_use=sha256_of(f"input-{i}")
        )
        for i in range(count)
    )


def _model_path_mint_problems(overrides: dict[str, Any]) -> list[str]:
    """The MODEL-PATH provenance mint markers (QC_001_MODEL_PREP r3-F1).

    The screen keys on the MINTABLE model-path shape, so the shared helper
    itself refuses the record a model-prep-adjacent caller could actually
    obtain. Deliberately NOT a blanket "any model_record_ref" refusal: the
    upstream provenance-domain negatives (a MODEL actor WITHOUT a ref, an
    ENGINE actor WITH a ref) are NOT mintable -- the provenance constructor
    itself fails closed on both -- and must keep falling through to its own
    ``ProvenanceIncompleteError`` so the upstream 44-section tests keep
    their exact semantics.
    """
    problems: list[str] = []
    actor = overrides.get("actor", make_actor())
    if getattr(actor, "actor_type", None) is ActorType.MODEL and overrides.get(
        "model_record_ref"
    ):
        # the ONLY mintable model-actor shape: the provenance constructor
        # itself raises whenever a MODEL actor has no model_record_ref, and
        # an ENGINE actor with a ref violates the xor rule (both fall
        # through to its own validation).
        problems.append("actor_type=MODEL with model_record_ref set")
    subject = overrides.get("subject_ref", "artifact-1")
    if isinstance(subject, str) and subject.startswith("artifact:"):
        # the model-prep path's subject convention (artifact:structure-...);
        # every upstream fixture subject is 'artifact-1'/'study-1'/... (dash).
        problems.append(f"model-path subject {subject!r}")
    return problems


def make_provenance(**overrides: Any):
    """Mint an engine-path ProvenanceRecord -- or REFUSE a model-path mint.

    MODEL-PATH PROVENANCE IS REFUSED BY DEFAULT (QC_001_MODEL_PREP r3-F1):
    this admitted helper itself -- structurally, in this shared module, not
    in any test fixture -- refuses to mint a record carrying the mintable
    model-path shape (an ``ActorType.MODEL`` actor together with a set
    ``model_record_ref``) or a model-path subject (``artifact:...``). The
    refusal is therefore behavioural and fixture-free: any caller, any
    harness, any plain script gets the same refusal.

    A provenance-domain test that genuinely needs to exercise model-path
    provenance construction must pass ``allow_model_path_provenance=True``
    explicitly (the provenance-domain opt-in); model-prep-adjacent code and
    tests NEVER pass it, and a permanent guard in
    ``tests/models/test_model_prep.py`` fails the suite if the opt-in
    appears anywhere in the model-prep path.
    """
    allow_model_path = overrides.pop("allow_model_path_provenance", False)
    if not allow_model_path:
        prohibited = _model_path_mint_problems(overrides)
        if prohibited:
            raise CanonicalValidationError(
                "make_provenance refuses to mint a MODEL-path provenance "
                "record by default (QC_001-MODEL is declared-not-executed; "
                "the #17 r8/r9 invention class; r3-F1): "
                + "; ".join(prohibited)
                + "; a provenance-domain test that genuinely needs this "
                "mint must pass allow_model_path_provenance=True explicitly"
            )
    kwargs: dict[str, Any] = dict(
        subject_ref="artifact-1",
        event_type=ProvenanceEventType.DERIVED,
        actor=make_actor(),
        inputs=make_inputs(),
        preprocessing="pipeline-1@v2",
        parameters_ref=BlobRef(blob_ref="params.json", sha256=sha256_of("params")),
        human_edit=NotApplicable(value=None, reason="fully automatic engine run"),
        upstream_refs=("prov-root",),
        environment=Environment(
            software_version="1.0.0",
            code_commit_sha=COMMIT,
            runtime="cpu",
            deterministic_seed="42",
        ),
        created_by="engine",
        at=AT,
        provenance_id="prov-test-0001",
    )
    kwargs.update(overrides)
    return create_provenance(**kwargs)


def make_na_provenance() -> ProvenanceRecord:
    """Record where every optional §44 item is explicitly NotApplicable."""
    return create_provenance(
        subject_ref="study-1",
        event_type=ProvenanceEventType.IMPORTED,
        actor=make_actor(),
        inputs=NotApplicable(value=None, reason="import job has no image inputs"),
        preprocessing=NotApplicable(value=None, reason="raw passthrough import"),
        parameters_ref=NotApplicable(value=None, reason="no tunable parameters"),
        human_edit=NotApplicable(value=None, reason="no human touched this"),
        upstream_refs=NotApplicable(value=None, reason="root of the DAG"),
        environment=Environment(software_version="1.0.0", code_commit_sha=COMMIT),
        created_by="adapter",
        at=AT,
    )


# ---------------------------------------------------------------------------
# study / series
# ---------------------------------------------------------------------------


def make_study(**overrides: Any):
    n = next(_STUDY_SEQ)
    kwargs: dict[str, Any] = dict(
        dicom_study_instance_uid=f"1.2.840.study.{n}",
        privacy_class=PrivacyClass.SYNTHETIC,
        import_source=ImportSource.LOCAL_FOLDER,
        import_job_ref="prov-import-1",
        source_root_path="/data/raw",
        exam_type=ExamType.CCTA,
        contrast_present=ContrastPresence.PRESENT,
        multiphase=False,
        classifier="rule-engine@1.0",
        dataset_refs=("DS-001",),
        accession_number=None,
        created_by="owner",
        imported_at=AT,
        study_id=f"study-fixed-{n:04d}",
    )
    kwargs.update(overrides)
    return create_study(**kwargs)


def make_instances(count: int = 3) -> tuple[InstanceEntry, ...]:
    entries = []
    for i in range(count):
        sop = f"1.2.840.sop.{i}"
        entries.append(
            InstanceEntry(sop_instance_uid=sop, file_path=f"slices/{i}.dcm", sha256=sha256_of(sop))
        )
    return tuple(entries)


def make_series_manifest(study_multiphase: bool = False, **overrides: Any):
    n = next(_SERIES_SEQ)
    instances = overrides.pop("instances", make_instances())
    ordered_sop_uids = tuple(inst.sop_instance_uid for inst in instances)
    kwargs: dict[str, Any] = dict(
        study_ref="study-fixed-0001",
        study_multiphase=study_multiphase,
        dicom_series_instance_uid=f"1.2.840.series.{n}",
        frame_of_reference_uid="1.2.840.for.1",
        modality="CT",
        series_class=SeriesClass.AXIAL_SOURCE,
        contrast_class=ContrastClass.ARTERIAL,
        classifier="rule-engine@1.0",
        classification_provenance_ref="prov-classify-1",
        instances=instances,
        ordered_sop_uids=ordered_sop_uids,
        original_dicom_ref=OriginalDicomRef(root_path="/data/raw", manifest_sha256=SHA_A),
        acquisition_phase=None,
        created_by="owner",
        created_at=AT,
        series_id=f"series-fixed-{n:04d}",
    )
    kwargs.update(overrides)
    return register_series(**kwargs)


# ---------------------------------------------------------------------------
# coordinates / volume
# ---------------------------------------------------------------------------

OBLIQUE_ORIENTATION = (0.0, 0.6, 0.8, 1.0, 0.0, 0.0)


def make_frame(**overrides: Any) -> FrameSpec:
    kwargs: dict[str, Any] = dict(
        frame_of_reference_uid="1.2.840.for.1",
        image_position_patient=(-125.0, -120.4, 42.0),
        image_orientation_patient=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        spacing_mm=(0.5, 0.45, 0.6),
        dimensions=(512, 512, 300),
    )
    kwargs.update(overrides)
    return FrameSpec(**kwargs)


def make_transform(**overrides: Any) -> CoordinateTransform:
    kwargs: dict[str, Any] = dict(
        transform_id="world-to-mesh",
        from_space=SpaceKind.WORLD,
        to_space=SpaceKind.MESH,
        method="icp-registration-v1",
        matrix=(
            (1.0, 0.0, 0.0, 10.0),
            (0.0, 0.0, 1.0, -2.5),
            (0.0, -1.0, 0.0, 3.25),
            (0.0, 0.0, 0.0, 1.0),
        ),
        frame_of_reference_uid="1.2.840.for.1",
        series_instance_uid="1.2.840.series.1",
        direction_cosines=((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, -1.0, 0.0)),
        translation=(10.0, -2.5, 3.25),
        handedness=Handedness.LPS,
        units=Units.MM,
    )
    kwargs.update(overrides)
    return CoordinateTransform(**kwargs)


def make_blob_ref(**overrides: Any) -> VolumeBlobRef:
    kwargs: dict[str, Any] = dict(
        sha256=SHA_B, codec="zarr.v2", compression="blosc", byte_size=1024
    )
    kwargs.update(overrides)
    return VolumeBlobRef(**kwargs)


def make_volume(**overrides: Any):
    kwargs: dict[str, Any] = dict(
        study_ref="study-fixed-0001",
        source_series_refs=("series-fixed-0001",),
        kind=VolumeKind.CANONICAL,
        coordinate_frame=make_frame(),
        blob_ref=make_blob_ref(),
        provenance_ref="prov-volume-1",
        release_lane=ReleaseLane.PERSONAL_RESEARCH,
    )
    kwargs.update(overrides)
    return create_volume(**kwargs)


# ---------------------------------------------------------------------------
# derived artifacts + store
# ---------------------------------------------------------------------------


def make_payload(schema: str = "ccta.payload.segmentation.v1") -> PayloadRef:
    return PayloadRef(payload_schema=schema, blob_sha256=sha256_of("payload-bytes"))


def make_claim() -> ClaimBinding:
    return ClaimBinding(
        claim_id="CLM-SEG-001",
        claim_version="v3",
        population_ref="pop-ccta-routine",
        metric_refs=("dice>=0.85", "hd95<=3mm"),
        snapshot_ref="snap-2026-01",
    )


def make_producer(**overrides: Any) -> Producer:
    kwargs: dict[str, Any] = dict(
        producer_type=ProducerType.MODEL,
        model_record_ref="coronary-seg@1.2.0",
    )
    kwargs.update(overrides)
    return Producer(**kwargs)


def make_geometry_base(geometry_id: str = "geometry-A", version: int = 1) -> GeometryBase:
    return GeometryBase(geometry_id=geometry_id, geometry_version=version)


def make_lineage(**overrides: Any) -> Lineage:
    kwargs: dict[str, Any] = dict(
        input_volume_refs=("volume-1",),
        input_artifact_refs=(),
        input_series_refs=("series-fixed-0001",),
        geometry_base=make_geometry_base(),
        registration_ref=None,
    )
    kwargs.update(overrides)
    return Lineage(**kwargs)


def make_artifact(kind: ArtifactKind = ArtifactKind.SEGMENTATION, **overrides: Any):
    producer_defaults: dict[ArtifactKind, Producer] = {
        ArtifactKind.SEGMENTATION: make_producer(),
        ArtifactKind.MEASUREMENT_SET: Producer(
            producer_type=ProducerType.DETERMINISTIC_ENGINE,
            algorithm="area-stenosis",
            algorithm_version="1.0.0",
            parameters_blob_sha256=SHA_C,
        ),
        ArtifactKind.PHYSIOLOGY_RESULT: Producer(
            producer_type=ProducerType.PHYSICS_BACKEND,
            algorithm="0d-flow",
            algorithm_version="2.1.0",
            parameters_blob_sha256=SHA_C,
        ),
        ArtifactKind.REPORT_DRAFT: Producer(
            producer_type=ProducerType.LLM,
            model_record_ref="report-llm@0.3.0",
        ),
        ArtifactKind.CLASSIFICATION_CANDIDATE: Producer(
            producer_type=ProducerType.MODEL,
            model_record_ref="cadrads-classifier@1.0.0",
        ),
        ArtifactKind.CENTERLINE: Producer(
            producer_type=ProducerType.DETERMINISTIC_ENGINE,
            algorithm="vessel-tracker",
            algorithm_version="1.4.0",
            parameters_blob_sha256=SHA_C,
        ),
        ArtifactKind.CORRECTION: Producer(
            producer_type=ProducerType.HUMAN,
            algorithm="radiologist-correction",
        ),
        ArtifactKind.EXPORT_PACKAGE: Producer(
            producer_type=ProducerType.DETERMINISTIC_ENGINE,
            algorithm="export_packager",
            parameters_blob_sha256=SHA_C,
        ),
    }
    lineage_overrides = overrides.pop("lineage", None)
    review_overrides = overrides.pop("review", None)
    if kind == ArtifactKind.MEASUREMENT_SET and lineage_overrides is None:
        lineage_overrides = make_lineage(
            input_volume_refs=(),
            input_artifact_refs=("centerline-artifact",),
            input_series_refs=(),
        )
    kwargs: dict[str, Any] = dict(
        study_ref="study-fixed-0001",
        kind=kind,
        payload=overrides.pop("payload", make_payload()),
        lineage=lineage_overrides if lineage_overrides is not None else make_lineage(),
        claim_binding=overrides.pop("claim_binding", make_claim()),
        producer=producer_defaults[kind],
        release_lane=ReleaseLane.PERSONAL_RESEARCH,
        provenance_ref="prov-derived-1",
        quality=QualityBlock(quality_grade=QualityGrade.GOOD),
        review=review_overrides
        if review_overrides is not None
        else ReviewBlock(review_state=ReviewState.PREDICTED),
        status=ArtifactStatus.ACTIVE,
        created_by="engine",
        created_at=AT,
        artifact_id=overrides.pop("artifact_id", "art-1"),
    )
    kwargs.update(overrides)
    return create_derived_artifact(**kwargs)


def make_store() -> tuple[DerivedArtifactStore, EventLog, InvalidationDAG]:
    log = EventLog()
    dag = InvalidationDAG()
    return DerivedArtifactStore(log=log, dag=dag), log, dag
