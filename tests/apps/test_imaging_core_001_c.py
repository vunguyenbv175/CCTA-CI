"""IMAGING_CORE_001_C: end-to-end Phase-0 exit-gate evidence.

These tests deliberately assemble the new 001-C wiring in the test driver so
the already-closed 001-A/001-B product seams remain unchanged.  The driver
still uses those seams directly: ``StudyWorkspaceManager`` performs governed
import/classification, ``CanonicalVolumeLoader`` performs canonical scalar
materialization, ``build_mpr`` produces the persisted view, and
``DerivedStateStoreManager`` owns SAVE/REOPEN.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, replace
from itertools import product
from pathlib import Path
from typing import cast

import pytest

from apps.workstation.cli import main
from apps.workstation.round_trip import (
    CHECK_BLOBS,
    CHECK_COORDS,
    CHECK_MEASURE,
    CHECK_MPR,
    CHECK_PROV,
    CHECK_REPLAY,
    CHECK_VIEWER,
    MANDATED_CHECKS,
    RoundTripVerificationEngine,
    _compute_canonical_measurements,
)
from apps.workstation.session_store import (
    DerivedStateStoreManager,
    RestoredSession,
    SessionSnapshot,
    WorkstationPaths,
    decode_value,
    encode_value,
)
from apps.workstation.volume_loader import (
    CanonicalVolumeLoader,
    build_frame_spec,
    measurement_parameters_digest,
)
from apps.workstation.workspace import (
    CODE_COMMIT_SHA,
    PAYLOAD_TRANSFORM_SCHEMA,
    RELEASE_LANE,
    SOFTWARE_VERSION,
    StudyWorkspaceManager,
)
from core.artifacts import (
    ClaimBinding,
    DerivedArtifact,
    DerivedArtifactStore,
    GeometryBase,
    Lineage,
    PayloadRef,
    Producer,
    create_derived_artifact,
)
from core.coordinates import CoordinateTransform, assert_round_trip, transform_for_frame
from core.dicom.qc import (
    QC_PAYLOAD_SCHEMA,
    QCAssessmentArtifact,
    QCGateDecision,
    QCReason,
    create_qc_assessment,
    qc_consumer_gate,
)
from core.dicom.reader import read_dicom_instance
from core.dicom.stack import validate_stack
from core.digest import canonical_digest
from core.enums import (
    ActorType,
    ArtifactKind,
    ProducerType,
    ProvenanceEventType,
    QualityGrade,
    VolumeKind,
)
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
    verify_chain,
)
from core.volume import Volume, VolumeBlobRef, create_volume
from tests.apps.fixture_factory import VolumeFixtureCase, build_volume_fixture_matrix
from validation.regression.runner import RegressionRunner
from visualization.mpr import MPRBuildResult, build_mpr

_AT = "2026-01-01T00:00:00Z"
_ACTOR = Actor(actor_type=ActorType.ENGINE, actor_id="ccta-imaging-core-001-c")
_ENVIRONMENT = Environment(
    software_version=SOFTWARE_VERSION,
    code_commit_sha=CODE_COMMIT_SHA,
    runtime="python-3.12 (imaging-core-001-c)",
    deterministic_seed="IMAGING_CORE_001_C",
)
_CLAIM_VIZ = ClaimBinding(claim_id="CLM-VIZ-001", claim_version="v0-draft")
_CLAIM_MEAS = ClaimBinding(claim_id="CLM-MEAS-001", claim_version="v0-draft")
_ULID_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
_PLANES = ("AXIAL", "CORONAL", "SAGITTAL")
_METHODS = ("NEAREST", "TRILINEAR")
_REQUIRED_CHECKS = (
    CHECK_BLOBS,
    CHECK_COORDS,
    CHECK_MEASURE,
    CHECK_PROV,
    CHECK_REPLAY,
    CHECK_MPR,
    # VIEWER_001 (milestone #19) extends the mandated set with the seventh
    # check; sessions without viewer documents pass it trivially.
    CHECK_VIEWER,
)

# Eight geometry strata cover orientation × normal × origin.  The final two
# cases exercise the scalar-domain branches without weakening that geometry
# coverage.
E2E_STRATA: tuple[tuple[str, str, str, str], ...] = tuple(
    (orientation, normal, origin, "unsigned")
    for orientation, normal, origin in product(
        ("axis-aligned", "oblique"),
        ("standard", "reversed"),
        ("zero", "nonzero"),
    )
) + (
    ("axis-aligned", "standard", "zero", "signed"),
    ("oblique", "reversed", "nonzero", "rescaled-fractional"),
)


def _ulid(number: int) -> str:
    """Encode a small deterministic counter as a valid Crockford ULID."""
    value = max(number, 0)
    digits: list[str] = []
    for _ in range(26):
        digits.append(_ULID_ALPHABET[value & 31])
        value >>= 5
    return "".join(reversed(digits))


def _id_factory(seed: int) -> Callable[[], str]:
    next_number = seed

    def mint() -> str:
        nonlocal next_number
        result = _ulid(next_number)
        next_number += 1
        return result

    return mint


def _study_ref(study_id: str) -> str:
    return f"study_id:{study_id}"


def _volume_ref(volume_id: str) -> str:
    return f"volume_id:{volume_id}"


def _mpr_ref(artifact_id: str) -> str:
    return f"mpr_artifact_id:{artifact_id}"


def _transform_ref(artifact_id: str) -> str:
    return f"coordinate_transform:{artifact_id}"


def _provenance_ref(provenance_id: str) -> str:
    return f"provenance:{provenance_id}"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _provenance(
    *,
    subject_ref: str,
    provenance_id: str,
    inputs: tuple[InputRef, ...],
    preprocessing: str,
    parameters_ref: BlobRef,
    upstream_refs: tuple[str, ...],
    event_type: ProvenanceEventType = ProvenanceEventType.DERIVED,
) -> ProvenanceRecord:
    return create_provenance(
        subject_ref=subject_ref,
        event_type=event_type,
        actor=_ACTOR,
        inputs=inputs,
        preprocessing=preprocessing,
        parameters_ref=parameters_ref,
        human_edit=NotApplicable(value=None, reason="no human edits in Phase-0 engine run"),
        upstream_refs=upstream_refs,
        environment=_ENVIRONMENT,
        created_by="imaging-core-001-c-test-driver",
        at=_AT,
        provenance_id=provenance_id,
    )


def _transform_artifact(
    *,
    volume: Volume,
    transform: CoordinateTransform,
    transform_artifact_id: str,
    transform_provenance_id: str,
    volume_provenance: ProvenanceRecord,
    store: DerivedArtifactStore,
) -> tuple[DerivedArtifact, ProvenanceRecord, dict[str, object]]:
    document = cast(dict[str, object], encode_value(transform))
    parameters_sha = canonical_digest(encode_value(volume.coordinate_frame))
    provenance = _provenance(
        subject_ref=transform_artifact_id,
        provenance_id=transform_provenance_id,
        inputs=(
            InputRef(
                ref=volume.volume_id,
                kind="volume",
                sha256_at_use=volume.blob_ref.sha256,
            ),
        ),
        preprocessing="voxel-to-world-affine-from-persisted-frame-spec",
        parameters_ref=BlobRef(blob_ref=PAYLOAD_TRANSFORM_SCHEMA, sha256=parameters_sha),
        upstream_refs=(volume_provenance.provenance_id,),
    )
    artifact = create_derived_artifact(
        artifact_id=transform_artifact_id,
        study_ref=volume.study_ref,
        kind=ArtifactKind.COORDINATE_TRANSFORM,
        payload=PayloadRef(
            payload_schema=PAYLOAD_TRANSFORM_SCHEMA,
            blob_sha256=canonical_digest(document),
            codec="json",
            inline_summary=(("transform_id", transform.transform_id),),
        ),
        lineage=Lineage(input_volume_refs=(volume.volume_id,)),
        claim_binding=_CLAIM_VIZ,
        producer=Producer(
            producer_type=ProducerType.DETERMINISTIC_ENGINE,
            algorithm="dicom-frame-transform",
            algorithm_version=SOFTWARE_VERSION,
            parameters_blob_sha256=parameters_sha,
        ),
        release_lane=RELEASE_LANE,
        provenance_ref=provenance.provenance_id,
        created_by="imaging-core-001-c-test-driver",
        created_at=_AT,
    )
    store.add(artifact)
    return artifact, provenance, document


def _measurement_artifact(
    *,
    loader: CanonicalVolumeLoader,
    volume: Volume,
    transform_artifact: DerivedArtifact,
    volume_provenance: ProvenanceRecord,
    transform_provenance: ProvenanceRecord,
    artifact_id: str,
    provenance_id: str,
    store: DerivedArtifactStore,
) -> tuple[DerivedArtifact, ProvenanceRecord, dict[str, object]]:
    values = loader.read_array(volume.blob_ref.sha256)
    document = _compute_canonical_measurements(values, volume.coordinate_frame.dimensions)
    provenance = _provenance(
        subject_ref=artifact_id,
        provenance_id=provenance_id,
        inputs=(
            InputRef(
                ref=volume.volume_id,
                kind="volume",
                sha256_at_use=volume.blob_ref.sha256,
            ),
            InputRef(
                ref=transform_artifact.artifact_id,
                kind="artifact",
                sha256_at_use=transform_artifact.payload.blob_sha256,
            ),
        ),
        preprocessing="canonical-scalar-central-box-statistics",
        parameters_ref=BlobRef(
            blob_ref="measurement:central-box-hu-stats:v1",
            sha256=measurement_parameters_digest(),
        ),
        upstream_refs=(volume_provenance.provenance_id, transform_provenance.provenance_id),
    )
    artifact = create_derived_artifact(
        artifact_id=artifact_id,
        study_ref=volume.study_ref,
        kind=ArtifactKind.MEASUREMENT_SET,
        payload=PayloadRef(
            payload_schema=str(document["schema"]),
            blob_sha256=canonical_digest(document),
            codec="json",
            inline_summary=(("algorithm", str(document["algorithm"])),),
        ),
        lineage=Lineage(
            input_volume_refs=(volume.volume_id,),
            input_artifact_refs=(transform_artifact.artifact_id,),
            geometry_base=GeometryBase(
                geometry_id=transform_artifact.artifact_id,
                geometry_version=1,
            ),
        ),
        claim_binding=_CLAIM_MEAS,
        producer=Producer(
            producer_type=ProducerType.DETERMINISTIC_ENGINE,
            algorithm=str(document["algorithm"]),
            algorithm_version=str(document["params_version"]),
            parameters_blob_sha256=measurement_parameters_digest(),
        ),
        release_lane=RELEASE_LANE,
        provenance_ref=provenance.provenance_id,
        created_by="imaging-core-001-c-test-driver",
        created_at=_AT,
    )
    store.add(artifact)
    return artifact, provenance, document


def _mpr_artifact(
    *,
    volume: Volume,
    transform: CoordinateTransform,
    transform_artifact: DerivedArtifact,
    volume_provenance: ProvenanceRecord,
    transform_provenance: ProvenanceRecord,
    plane: str,
    method: str,
    artifact_id: str,
    provenance_id: str,
    loader: CanonicalVolumeLoader,
    study_instance_uid: str,
    series_instance_uid: str,
    store: DerivedArtifactStore,
) -> tuple[MPRBuildResult, ProvenanceRecord]:
    result = build_mpr(
        volume,
        transform,
        plane=plane,
        origin=volume.coordinate_frame.image_position_patient,
        in_plane_spacing_mm=(1.0, 1.0),
        matrix_size=(2, 3),
        method=method,
        out_of_bounds_fill_value=-1000.0,
        loader=loader,
        transform_artifact_ref=_transform_ref(transform_artifact.artifact_id),
        provenance_ref=_provenance_ref(provenance_id),
        study_instance_uid=study_instance_uid,
        series_instance_uid=series_instance_uid,
        frame_of_reference_uid=volume.coordinate_frame.frame_of_reference_uid,
        release_lane=RELEASE_LANE,
        created_by="imaging-core-001-c-test-driver",
        artifact_id=artifact_id,
    )
    provenance = _provenance(
        subject_ref=result.artifact.artifact_id,
        provenance_id=provenance_id,
        inputs=(
            InputRef(
                ref=volume.volume_id,
                kind="volume",
                sha256_at_use=volume.blob_ref.sha256,
            ),
            InputRef(
                ref=transform_artifact.artifact_id,
                kind="artifact",
                sha256_at_use=transform_artifact.payload.blob_sha256,
            ),
        ),
        preprocessing=f"orthogonal-mpr:{plane}:{method}",
        parameters_ref=BlobRef(
            blob_ref=f"mpr:sampling:{plane}:{method}",
            sha256=str(result.payload["sampling_parameters_sha256"]),
        ),
        upstream_refs=(volume_provenance.provenance_id, transform_provenance.provenance_id),
    )
    store.add(result.artifact)
    return result, provenance


def _qc_document(assessment: QCAssessmentArtifact) -> dict[str, object]:
    payload = assessment.payload
    return {
        "schema": QC_PAYLOAD_SCHEMA,
        "artifact_ref": assessment.artifact_ref,
        "kind": assessment.kind,
        "status": assessment.status,
        "payload": {
            "schema_version": payload.schema_version,
            "scope": payload.scope.value,
            "subject_refs": list(payload.subject_refs),
            "quality_grade": payload.quality_grade.value,
            "confidence": payload.confidence.value if payload.confidence else None,
            "flags": list(payload.flags),
            "vessel_id": payload.vessel_id,
            "region_id": payload.region_id,
            "producer": payload.producer.value,
            "produced_by_ref": payload.produced_by_ref,
            "supersedes_ref": payload.supersedes_ref,
            "justification_code": payload.justification_code,
            "evidence_refs": list(payload.evidence_refs),
        },
    }


@dataclass(frozen=True)
class E2EPipelineResult:
    fixture: VolumeFixtureCase
    paths: WorkstationPaths
    store: DerivedStateStoreManager
    artifact_store: DerivedArtifactStore
    session: RestoredSession
    volume: Volume
    volume_provenance: ProvenanceRecord
    transform: CoordinateTransform
    transform_artifact: DerivedArtifact
    transform_provenance: ProvenanceRecord
    measurement_artifact: DerivedArtifact
    measurement_provenance: ProvenanceRecord
    mpr_results: tuple[MPRBuildResult, ...]
    mpr_provenance: tuple[ProvenanceRecord, ...]
    qc: QCAssessmentArtifact
    qc_provenance: ProvenanceRecord
    artifacts: tuple[DerivedArtifact, ...]
    provenance: tuple[ProvenanceRecord, ...]
    payload_documents: dict[str, dict[str, object]]
    id_factory: Callable[[], str]


def run_001_c_pipeline(
    workspace: Path,
    fixture: VolumeFixtureCase,
    *,
    id_seed: int = 100,
) -> E2EPipelineResult:
    """Run import → SAVE/REOPEN → derive → SAVE/REOPEN for one fixture."""
    source = fixture.paths[0].parent
    ids = _id_factory(id_seed)
    manager = StudyWorkspaceManager(workspace, id_factory=ids)
    imported = manager.import_study(source)
    assert len(imported.series) == 1
    paths = imported.paths
    store = DerivedStateStoreManager(paths)
    imported_snapshot = SessionSnapshot(
        study=imported.study,
        series=tuple(item.series for item in imported.series),
        volumes=(),
        artifacts=(),
        payload_documents={},
        provenance=imported.provenance,
        log=EventLog(),
        skipped_instances=imported.skipped,
    )
    store.save(imported_snapshot)
    reopened_import = store.reopen()

    series = reopened_import.series[0]
    source_root = Path(series.original_dicom_ref.root_path)
    instances = []
    for entry in series.manifest.instances:
        path = source_root / entry.file_path
        assert _sha256(path.read_bytes()) == entry.sha256
        instance = read_dicom_instance(path)
        assert instance.sop_instance_uid == entry.sop_instance_uid
        instances.append(instance)
    stack = validate_stack(tuple(instances))
    assert stack.admitted is True
    assert stack.slice_spacing_mm is not None
    ordered = stack.instances
    shape = (len(ordered), ordered[0].rows, ordered[0].columns)
    frame = build_frame_spec(
        frame_of_reference_uid=series.frame_of_reference_uid,
        image_position_patient=ordered[0].image_position_patient,
        image_orientation_patient=ordered[0].image_orientation_patient,
        pixel_spacing_mm=ordered[0].pixel_spacing,
        slice_spacing_mm=stack.slice_spacing_mm,
        dimensions=shape,
    )
    scalar_domain = "HU" if fixture.pixels == "rescaled-fractional" else "STORED"
    loader = CanonicalVolumeLoader(paths.blobs)
    materialized = loader.materialize_dicom_stack(
        ordered,
        scalar_domain=scalar_domain,
        intensity_unit=scalar_domain,
        frame=frame,
    )
    assert materialized.frame is not None
    volume_provenance_id = ids()
    volume_id = ids()
    volume_provenance = _provenance(
        subject_ref=volume_id,
        provenance_id=volume_provenance_id,
        inputs=(
            InputRef(
                ref=series.series_id,
                kind="series",
                sha256_at_use=series.manifest.composite_series_sha256,
            ),
        ),
        preprocessing="dicom-reader-rule-table-stack-admission-canonical-materialization",
        parameters_ref=BlobRef(
            blob_ref="volume:canonical-scalar-materialization:v1",
            sha256=canonical_digest(materialized.envelope),
        ),
        upstream_refs=(
            next(
                record.provenance_id
                for record in reopened_import.provenance
                if record.event_type == ProvenanceEventType.IMPORTED
            ),
        ),
    )
    volume = create_volume(
        study_ref=reopened_import.study.study_id,
        source_series_refs=(series.series_id,),
        kind=VolumeKind.CANONICAL,
        coordinate_frame=materialized.frame,
        blob_ref=VolumeBlobRef(
            sha256=materialized.blob.sha256,
            codec=str(materialized.envelope["codec"]),
            byte_size=materialized.blob.byte_size,
        ),
        provenance_ref=volume_provenance.provenance_id,
        release_lane=RELEASE_LANE,
        volume_id=volume_id,
    )

    artifact_store = DerivedArtifactStore(reopened_import.log, InvalidationDAG())
    transform = transform_for_frame(
        volume.coordinate_frame,
        version=1,
        series_instance_uid=series.dicom_series_instance_uid,
    )
    assert_round_trip(transform, transform.inverse)
    transform_artifact, transform_provenance, transform_document = _transform_artifact(
        volume=volume,
        transform=transform,
        transform_artifact_id=ids(),
        transform_provenance_id=ids(),
        volume_provenance=volume_provenance,
        store=artifact_store,
    )
    measurement_artifact, measurement_provenance, measurement_document = _measurement_artifact(
        loader=loader,
        volume=volume,
        transform_artifact=transform_artifact,
        volume_provenance=volume_provenance,
        transform_provenance=transform_provenance,
        artifact_id=ids(),
        provenance_id=ids(),
        store=artifact_store,
    )

    mpr_results: list[MPRBuildResult] = []
    mpr_provenance: list[ProvenanceRecord] = []
    for plane, method in product(_PLANES, _METHODS):
        mpr_result, mpr_record = _mpr_artifact(
            volume=volume,
            transform=transform,
            transform_artifact=transform_artifact,
            volume_provenance=volume_provenance,
            transform_provenance=transform_provenance,
            plane=plane,
            method=method,
            artifact_id=ids(),
            provenance_id=ids(),
            loader=loader,
            study_instance_uid=reopened_import.study.dicom_study_instance_uid,
            series_instance_uid=series.dicom_series_instance_uid,
            store=artifact_store,
        )
        mpr_results.append(mpr_result)
        mpr_provenance.append(mpr_record)

    qc_provenance_id = ids()
    qc = create_qc_assessment(
        subject_refs=(_study_ref(reopened_import.study.study_id),),
        provenance_ref=_provenance_ref(qc_provenance_id),
        artifact_ref=f"qc_artifact:{ids()}",
    )
    qc_document = _qc_document(qc)
    qc_provenance = _provenance(
        subject_ref=qc.artifact_ref,
        provenance_id=qc_provenance_id,
        inputs=tuple(
            InputRef(
                ref=result.artifact.artifact_id,
                kind="artifact",
                sha256_at_use=result.artifact.payload.blob_sha256,
            )
            for result in mpr_results
        ),
        preprocessing="placeholder-qc-no-image-quality-computation",
        parameters_ref=BlobRef(
            blob_ref=QC_PAYLOAD_SCHEMA,
            sha256=canonical_digest(qc_document),
        ),
        upstream_refs=tuple(record.provenance_id for record in mpr_provenance),
    )
    mpr_subjects = tuple(_mpr_ref(result.artifact.artifact_id) for result in mpr_results)
    volume_subject = _volume_ref(volume.volume_id)
    study_subject = _study_ref(reopened_import.study.study_id)
    subject_lineage: dict[str, object] = {
        **{subject: {"source_volume_ref": volume_subject} for subject in mpr_subjects},
        volume_subject: {"study_ref": study_subject},
    }
    gate = qc_consumer_gate(
        mpr_subjects,
        assessments=(qc,),
        consumer="mpr",
        subject_lineage=subject_lineage,
    )
    assert gate.decision == QCGateDecision.PROCEED
    assert gate.reason_code is None

    artifacts = (
        transform_artifact,
        measurement_artifact,
        *(result.artifact for result in mpr_results),
    )
    all_provenance = (
        *reopened_import.provenance,
        volume_provenance,
        transform_provenance,
        measurement_provenance,
        *mpr_provenance,
        qc_provenance,
    )
    documents = {
        transform_artifact.artifact_id: transform_document,
        measurement_artifact.artifact_id: measurement_document,
        **{result.artifact.artifact_id: result.payload for result in mpr_results},
        qc.artifact_ref: qc_document,
    }
    final_snapshot = SessionSnapshot(
        study=reopened_import.study,
        series=reopened_import.series,
        volumes=(volume,),
        artifacts=artifacts,
        payload_documents=documents,
        provenance=all_provenance,
        log=reopened_import.log,
        skipped_instances=reopened_import.skipped_instances,
    )
    store.save(final_snapshot)
    reopened = store.reopen()
    pipeline_result = E2EPipelineResult(
        fixture=fixture,
        paths=paths,
        store=store,
        artifact_store=artifact_store,
        session=reopened,
        volume=volume,
        volume_provenance=volume_provenance,
        transform=transform,
        transform_artifact=transform_artifact,
        transform_provenance=transform_provenance,
        measurement_artifact=measurement_artifact,
        measurement_provenance=measurement_provenance,
        mpr_results=tuple(mpr_results),
        mpr_provenance=tuple(mpr_provenance),
        qc=qc,
        qc_provenance=qc_provenance,
        artifacts=artifacts,
        provenance=all_provenance,
        payload_documents=documents,
        id_factory=ids,
    )
    _assert_reopened_identity(pipeline_result)
    return pipeline_result


def _assert_reopened_identity(result: E2EPipelineResult) -> None:
    session = result.session
    assert session.volumes == (result.volume,)
    assert session.artifacts == result.artifacts
    assert session.provenance == result.provenance
    assert session.payload_documents == result.payload_documents
    assert session.volumes[0].blob_ref.sha256 == result.volume.blob_ref.sha256
    loader = CanonicalVolumeLoader(result.paths.blobs)
    assert loader.read_blob(session.volumes[0].blob_ref.sha256) == loader.read_blob(
        result.volume.blob_ref.sha256
    )

    transform_document = session.payload_documents[result.transform_artifact.artifact_id]
    decoded_transform = decode_value(transform_document)
    assert isinstance(decoded_transform, CoordinateTransform)
    assert decoded_transform == result.transform
    assert result.transform.version == 1
    assert result.transform_artifact.lineage.input_volume_refs == (result.volume.volume_id,)

    assert {
        (str(payload["plane"]), str(payload["method"]))
        for payload in (
            session.payload_documents[mpr.artifact.artifact_id] for mpr in result.mpr_results
        )
    } == set(product(_PLANES, _METHODS))
    for mpr in result.mpr_results:
        payload = session.payload_documents[mpr.artifact.artifact_id]
        assert canonical_digest(payload) == mpr.artifact.payload.blob_sha256
        assert canonical_digest(payload) == canonical_digest(
            result.payload_documents[mpr.artifact.artifact_id]
        )
        pixel_ref = cast(dict[str, object], payload["pixel_blob_ref"])
        assert isinstance(pixel_ref["sha256"], str)
        assert session.pixel_blobs[pixel_ref["sha256"]] == mpr.pixels.tobytes(order="C")

    chain_ok, chain_problems = verify_chain(session.provenance)
    assert chain_ok, chain_problems
    by_id = {record.provenance_id: record for record in session.provenance}
    assert by_id[result.qc_provenance.provenance_id].subject_ref == result.qc.artifact_ref
    for artifact in session.artifacts:
        record = by_id[artifact.provenance_ref]
        assert record.subject_ref == artifact.artifact_id
    for volume in session.volumes:
        record = by_id[volume.provenance_ref]
        assert record.subject_ref == volume.volume_id

    report = RoundTripVerificationEngine(result.paths.root).verify(session)
    assert report.verdict == "PASS"
    assert tuple(report.checks) == tuple(_REQUIRED_CHECKS)
    for check_name in _REQUIRED_CHECKS:
        assert report.checks[check_name].passed, report.checks[check_name].details


@pytest.fixture(scope="module")
def volume_fixture_matrix(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[VolumeFixtureCase, ...]:
    return build_volume_fixture_matrix(tmp_path_factory.mktemp("volume-001-fixtures"))


def _fixture_for(
    fixtures: tuple[VolumeFixtureCase, ...],
    stratum: tuple[str, str, str, str],
) -> VolumeFixtureCase:
    matches = tuple(
        fixture
        for fixture in fixtures
        if (
            fixture.orientation,
            fixture.normal,
            fixture.origin,
            fixture.pixels,
        )
        == stratum
        and fixture.spacing == "isotropic"
    )
    assert len(matches) == 1
    return matches[0]


@pytest.mark.parametrize(
    "stratum",
    E2E_STRATA,
    ids=lambda value: "-".join(value),
)
def test_imaging_core_001_c_full_phase0_exit_gate(
    tmp_path: Path,
    volume_fixture_matrix: tuple[VolumeFixtureCase, ...],
    stratum: tuple[str, str, str, str],
) -> None:
    fixture = _fixture_for(volume_fixture_matrix, stratum)
    result = run_001_c_pipeline(tmp_path / "workspace", fixture, id_seed=1000)
    assert result.fixture.fixture_id == fixture.fixture_id
    assert len(result.mpr_results) == 6
    assert result.session.skipped_instances == ()
    assert result.session.study.exam_classification.exam_type.value == "CCTA"
    assert result.qc.quality_grade == QualityGrade.GOOD
    assert result.qc.producer == ProducerType.DETERMINISTIC_ENGINE
    assert result.qc.payload.evidence_refs == ()


def test_imaging_core_001_c_cli_exit_gate(
    tmp_path: Path,
    volume_fixture_matrix: tuple[VolumeFixtureCase, ...],
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture = _fixture_for(
        volume_fixture_matrix,
        ("oblique", "reversed", "nonzero", "rescaled-fractional"),
    )
    result = run_001_c_pipeline(tmp_path / "workspace", fixture, id_seed=1100)
    code = main(["verify", "--workspace", str(result.paths.root), "--no-network-guard"])
    document = json.loads(capsys.readouterr().out)
    assert code == 0
    assert document["verdict"] == "PASS"
    assert all(document["checks"][name]["passed"] for name in _REQUIRED_CHECKS)
    assert document["mandated_checks"] == list(MANDATED_CHECKS)


def test_imaging_core_001_c_provenance_nine_fields_and_dag(
    tmp_path: Path,
    volume_fixture_matrix: tuple[VolumeFixtureCase, ...],
) -> None:
    fixture = _fixture_for(
        volume_fixture_matrix,
        ("axis-aligned", "standard", "zero", "signed"),
    )
    result = run_001_c_pipeline(tmp_path / "workspace", fixture, id_seed=1200)
    session = result.session
    chain_ok, problems = verify_chain(session.provenance)
    assert chain_ok, problems
    assert all(record.record_sha256 for record in session.provenance)

    # §44's nine items are represented by the closed record fields below.
    # Deterministic runs explicitly take the algorithm branch: model ref is
    # None, actor/environment carry algorithm + version, and human edits are
    # an explicit NotApplicable wrapper rather than an omitted value.
    nine_fields = (
        "source_image",
        "preprocessing",
        "model",
        "model_version",
        "algorithm",
        "algorithm_version",
        "parameters",
        "human_edits",
        "upstream_dependencies",
    )
    assert len(nine_fields) == 9
    representatives = (
        next(
            record
            for record in session.provenance
            if record.event_type == ProvenanceEventType.IMPORTED
        ),
        next(
            record
            for record in session.provenance
            if record.event_type == ProvenanceEventType.CLASSIFIED
        ),
        result.volume_provenance,
        result.transform_provenance,
        result.mpr_provenance[0],
        result.qc_provenance,
    )
    for record in representatives:
        assert record.inputs or isinstance(record.inputs, NotApplicable)
        assert record.preprocessing or isinstance(record.preprocessing, NotApplicable)
        assert record.model_record_ref is None
        assert record.actor.actor_id
        assert record.environment.software_version
        assert len(record.environment.code_commit_sha) == 40
        assert record.parameters_ref or isinstance(record.parameters_ref, NotApplicable)
        assert record.human_edit or isinstance(record.human_edit, NotApplicable)
        assert record.upstream_refs or isinstance(record.upstream_refs, NotApplicable)

    by_id = {record.provenance_id: record for record in session.provenance}
    series = session.series[0]
    assert by_id[series.classification.classification_provenance_ref].subject_ref == (
        series.dicom_series_instance_uid
    )
    assert by_id[session.study.import_info.import_job_ref].subject_ref == (
        session.study.dicom_study_instance_uid
    )
    for artifact in session.artifacts:
        assert by_id[artifact.provenance_ref].subject_ref == artifact.artifact_id
    for volume in session.volumes:
        assert by_id[volume.provenance_ref].subject_ref == volume.volume_id

    report = RoundTripVerificationEngine(result.paths.root).verify(session)
    assert report.checks[CHECK_PROV].passed


def test_imaging_core_001_c_qc_consumer_gate_matrix(
    tmp_path: Path,
    volume_fixture_matrix: tuple[VolumeFixtureCase, ...],
) -> None:
    fixture = _fixture_for(
        volume_fixture_matrix,
        ("axis-aligned", "standard", "zero", "unsigned"),
    )
    result = run_001_c_pipeline(tmp_path / "workspace", fixture, id_seed=1300)
    volume_ref = _volume_ref(result.volume.volume_id)
    lineage = {volume_ref: {"study_ref": _study_ref(result.session.study.study_id)}}
    assert (
        qc_consumer_gate((volume_ref,), consumer="measurement", assessments=()).reason_code
        == QCReason.QC_NOT_ASSESSED
    )

    limited = replace(
        result.qc,
        payload=replace(
            result.qc.payload,
            quality_grade=QualityGrade.LIMITED,
            justification_code="RULE_ENGINE_EVIDENCE",
            evidence_refs=(_provenance_ref(result.qc_provenance.provenance_id),),
        ),
    )
    assert (
        qc_consumer_gate(
            (volume_ref,),
            assessments=(limited,),
            consumer="measurement",
            subject_lineage=lineage,
        ).reason_code
        == QCReason.QC_LIMITED
    )
    not_analyzable = replace(
        result.qc,
        payload=replace(
            result.qc.payload,
            quality_grade=QualityGrade.NOT_ANALYZABLE,
            justification_code="RULE_ENGINE_EVIDENCE",
            evidence_refs=(_provenance_ref(result.qc_provenance.provenance_id),),
        ),
    )
    assert (
        qc_consumer_gate(
            (volume_ref,),
            assessments=(not_analyzable,),
            consumer="measurement",
            subject_lineage=lineage,
        ).reason_code
        == QCReason.QC_NOT_ANALYZABLE
    )
    assert (
        qc_consumer_gate(
            (volume_ref,),
            assessments=(result.qc,),
            consumer="measurement",
            subject_lineage=lineage,
        ).decision
        == QCGateDecision.PROCEED
    )


def test_imaging_core_001_c_invalidation_is_append_only_and_recomputes_new_mpr(
    tmp_path: Path,
    volume_fixture_matrix: tuple[VolumeFixtureCase, ...],
) -> None:
    fixture = _fixture_for(
        volume_fixture_matrix,
        ("oblique", "reversed", "nonzero", "unsigned"),
    )
    result = run_001_c_pipeline(tmp_path / "workspace", fixture, id_seed=1400)
    old_mpr = result.mpr_results[0]
    old_mpr_id = old_mpr.artifact.artifact_id
    old_transform_id = result.transform_artifact.artifact_id
    old_payload_digest = old_mpr.artifact.payload.blob_sha256

    stale_ids = result.artifact_store.register_new_geometry_version(old_transform_id, 2)
    assert old_mpr_id in stale_ids
    assert result.artifact_store.get(old_mpr_id).status.value == "STALE"
    assert result.artifact_store.get(old_mpr_id).payload.blob_sha256 == old_payload_digest
    history = result.artifact_store.event_log.history_for(old_mpr_id, entity="derived_artifact")
    assert any(event.to_value == "STALE" for event in history)

    new_transform = transform_for_frame(
        result.volume.coordinate_frame,
        version=2,
        series_instance_uid=result.session.series[0].dicom_series_instance_uid,
    )
    new_transform_artifact, new_transform_provenance, new_transform_document = _transform_artifact(
        volume=result.volume,
        transform=new_transform,
        transform_artifact_id=result.id_factory(),
        transform_provenance_id=result.id_factory(),
        volume_provenance=result.volume_provenance,
        store=result.artifact_store,
    )
    new_mpr, new_mpr_provenance = _mpr_artifact(
        volume=result.volume,
        transform=new_transform,
        transform_artifact=new_transform_artifact,
        volume_provenance=result.volume_provenance,
        transform_provenance=new_transform_provenance,
        plane=str(old_mpr.payload["plane"]),
        method=str(old_mpr.payload["method"]),
        artifact_id=result.id_factory(),
        provenance_id=result.id_factory(),
        loader=CanonicalVolumeLoader(result.paths.blobs),
        study_instance_uid=result.session.study.dicom_study_instance_uid,
        series_instance_uid=result.session.series[0].dicom_series_instance_uid,
        store=result.artifact_store,
    )
    assert new_mpr.artifact.artifact_id != old_mpr_id
    assert new_mpr.artifact.lineage.input_artifact_refs == (new_transform_artifact.artifact_id,)
    assert new_mpr.artifact.lineage.geometry_base == GeometryBase(
        geometry_id=new_transform_artifact.artifact_id,
        geometry_version=2,
    )
    assert new_mpr.artifact.payload.blob_sha256 != old_payload_digest
    assert result.artifact_store.get(old_mpr_id).status.value == "STALE"
    assert result.artifact_store.get(new_mpr.artifact.artifact_id).status.value == "ACTIVE"
    assert result.artifact_store.get(old_mpr_id).payload.blob_sha256 == old_payload_digest
    assert new_transform_document != result.payload_documents[result.transform_artifact.artifact_id]


def _content_digests(result: E2EPipelineResult) -> dict[str, object]:
    return {
        "volume_blob_sha256": result.volume.blob_ref.sha256,
        "transform_payload_sha256": result.transform_artifact.payload.blob_sha256,
        "measurement_payload_sha256": result.measurement_artifact.payload.blob_sha256,
        "mpr": [
            {
                "plane": mpr.payload["plane"],
                "method": mpr.payload["method"],
                "payload_sha256": mpr.artifact.payload.blob_sha256,
                "pixel_blob_sha256": mpr.pixel_blob_sha256,
            }
            for mpr in result.mpr_results
        ],
    }


def test_imaging_core_001_c_golden_regression_is_bit_reproducible(
    tmp_path: Path,
    volume_fixture_matrix: tuple[VolumeFixtureCase, ...],
) -> None:
    fixture = _fixture_for(
        volume_fixture_matrix,
        ("axis-aligned", "standard", "zero", "unsigned"),
    )
    first = run_001_c_pipeline(tmp_path / "first", fixture, id_seed=1500)
    second = run_001_c_pipeline(tmp_path / "second", fixture, id_seed=1500)
    assert _content_digests(first) == _content_digests(second)

    golden_path = (
        Path(__file__).resolve().parents[2]
        / "validation"
        / "regression"
        / "imaging_core_001_c.json"
    )
    runner = RegressionRunner(RoundTripVerificationEngine(first.paths.root))
    golden = runner.load_golden(golden_path)
    comparison = runner.compare(runner.run_fresh(first.paths.root), golden)
    assert comparison.verdict == "PASS", comparison.to_dict()
    expected_digests = golden.get("content_digests")
    assert expected_digests == _content_digests(first)
    assert "path" not in json.dumps(expected_digests).lower()
    assert "timestamp" not in json.dumps(expected_digests).lower()
