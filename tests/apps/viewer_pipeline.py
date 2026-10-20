"""Deterministic governed-session builder for the VIEWER_001 tests.

Builds a real WS-02 session over the synthetic study fixture: import ->
derive -> MPR views (two per plane, distinct plane coordinates) -> HUMAN
acceptance review -> save -> reopen.  All ids are counter-seeded ULIDs so
every artifact digest is stable across runs.  No patient data anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from apps.workstation.session_store import (
    DerivedStateStoreManager,
    RestoredSession,
    SessionSnapshot,
)
from apps.workstation.workspace import StudyWorkspaceManager
from core.artifacts import DerivedArtifact, DerivedArtifactStore, InvalidationDAG
from core.coordinates import CoordinateTransform
from core.digest import canonical_digest
from core.enums import ActorType, ArtifactKind, ProvenanceEventType, ReviewState
from core.events import EventLog
from core.provenance import (
    Actor,
    BlobRef,
    Environment,
    InputRef,
    NotApplicable,
    ProvenanceRecord,
    create_provenance,
)
from tests.apps.fixture_factory import build_synthetic_study
from visualization.mpr import MPRBuildResult, build_mpr

_AT = "2026-09-07T00:00:00Z"
_ENGINE_ACTOR = Actor(actor_type=ActorType.ENGINE, actor_id="ccta-viewer-fixture")
_FIXED_ULID_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"

_PLANES = ("AXIAL", "CORONAL", "SAGITTAL")


def _ulid(number: int) -> str:
    """Deterministic Crockford-base32 ULID shaped id for fixture stability."""

    value = number
    chars = []
    for _shift in range(26):
        chars.append(_FIXED_ULID_ALPHABET[value & 0x1F])
        value >>= 5
    return "".join(reversed(chars))


@dataclass(frozen=True)
class ViewerSessionFixture:
    """Everything the viewer tests need from one governed session build."""

    workspace: Path
    dicom_dir: Path
    store: DerivedStateStoreManager
    session: RestoredSession
    mpr_artifacts: tuple[DerivedArtifact, ...]
    mpr_results: tuple[MPRBuildResult, ...]
    mpr_provenance: tuple[ProvenanceRecord, ...]
    source_manifest_sha256: str
    study_id: str
    qc_bundle: object = None


def build_viewer_session(
    workspace: Path,
    *,
    dicom_dir: Path | None = None,
) -> ViewerSessionFixture:
    """Import -> derive -> MPRs -> accept -> save -> reopen, deterministically."""

    workspace = Path(workspace)
    source = dicom_dir or build_synthetic_study(
        workspace.parent / f"{workspace.name}-dicom", slices_per_phase=3
    )
    # Deterministic ULID sequence for every artifact/provenance id.
    _id_counter = iter(range(1, 1_000_000))
    manager = StudyWorkspaceManager(
        workspace, id_factory=lambda: _ulid(next(_id_counter))
    )
    imported = manager.import_study(Path(source))
    store_manager = DerivedStateStoreManager(imported.paths)
    store_manager.save(
        SessionSnapshot(
            study=imported.study,
            series=tuple(item.series for item in imported.series),
            volumes=(),
            artifacts=(),
            payload_documents={},
            provenance=imported.provenance,
            log=EventLog(),
            skipped_instances=imported.skipped,
        )
    )
    session = store_manager.reopen()
    outcome = manager.derive(session)
    documents = dict(session.payload_documents)
    documents.update(outcome.payload_documents)
    store_manager.save(
        SessionSnapshot(
            study=session.study,
            series=session.series,
            volumes=session.volumes + outcome.volumes,
            artifacts=session.artifacts + outcome.artifacts,
            payload_documents=documents,
            provenance=session.provenance + outcome.provenance,
            log=session.log,
            skipped_instances=session.skipped_instances,
        )
    )
    session = store_manager.reopen()

    volume = session.volumes[0]
    series = session.series[0]
    transform_artifact = next(
        artifact
        for artifact in session.artifacts
        if artifact.kind == ArtifactKind.COORDINATE_TRANSFORM
    )
    from apps.workstation.session_store import decode_value

    transform = decode_value(session.payload_documents[transform_artifact.artifact_id])
    assert isinstance(transform, CoordinateTransform)

    artifact_store = DerivedArtifactStore(session.log, InvalidationDAG())
    loader = manager  # noqa: F841  (manager holds paths; loader built below)
    from apps.workstation.volume_loader import CanonicalVolumeLoader

    blob_loader = CanonicalVolumeLoader(session.paths.blobs)

    mpr_results: list[MPRBuildResult] = []
    mpr_provenance: list[ProvenanceRecord] = []
    accepted_records: list[DerivedArtifact] = []

    # Two distinct in-plane origins per plane (different plane coordinates so
    # the LAYER-G ordering and scrolling are exercised).
    origins: dict[str, tuple[tuple[float, float, float], ...]] = {
        "AXIAL": ((0.0, 0.0, 0.0), (0.0, 0.0, 2.0)),
        "CORONAL": ((0.0, 0.0, 0.0), (0.0, -1.5, 0.0)),
        "SAGITTAL": ((0.0, 0.0, 0.0), (1.5, 0.0, 0.0)),
    }
    for plane in _PLANES:
        for origin in origins[plane]:
            artifact_id = _ulid(next(_id_counter))
            provenance_id = _ulid(next(_id_counter))
            mpr_record = create_provenance(
                subject_ref=artifact_id,
                event_type=ProvenanceEventType.DERIVED,
                actor=_ENGINE_ACTOR,
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
                preprocessing="fixture-mpr-view-nearest",
                parameters_ref=BlobRef(
                    blob_ref="fixture-mpr-parameters",
                    sha256=canonical_digest({"plane": plane, "origin": list(origin)}),
                ),
                human_edit=NotApplicable(value=None, reason="deterministic fixture MPR"),
                upstream_refs=(
                    volume.provenance_ref,
                    transform_artifact.provenance_ref,
                ),
                environment=Environment(
                    software_version="ccta-viewer-fixture",
                    code_commit_sha="0" * 40,
                    runtime="python-3.12 (viewer fixture)",
                    deterministic_seed="VIEWER_001_E2E",
                ),
                created_by="ccta_viewer_fixture",
                at=_AT,
                provenance_id=provenance_id,
            )
            result = build_mpr(
                volume,
                transform,
                plane=plane,
                origin=origin,
                in_plane_spacing_mm=(0.5, 0.5),
                matrix_size=(4, 4),
                method="NEAREST",
                out_of_bounds_fill_value=0.0,
                loader=blob_loader,
                transform_artifact_ref=f"coordinate_transform:{transform_artifact.artifact_id}",
                provenance_ref=f"provenance:{provenance_id}",
                study_instance_uid=session.study.dicom_study_instance_uid,
                series_instance_uid=series.dicom_series_instance_uid,
                frame_of_reference_uid=volume.coordinate_frame.frame_of_reference_uid,
                artifact_id=artifact_id,
                created_by="ccta_viewer_fixture",
            )
            artifact_store.add(result.artifact)
            artifact_store.apply_review_transition(
                artifact_id,
                ReviewState.ACCEPTED,
                actor_role=ActorType.HUMAN,
                reviewed_by="viewer-fixture-acceptor",
                evidence_ref="synthetic accepted MPR for the viewer E2E",
                at=_AT,
            )
            mpr_results.append(result)
            mpr_provenance.append(mpr_record)
            accepted_records.append(artifact_store.get(artifact_id))

    documents.update(
        {result.artifact.artifact_id: result.payload for result in mpr_results}
    )
    merged_provenance: tuple[ProvenanceRecord, ...] = (
        *session.provenance,
        *outcome.provenance,
        *mpr_provenance,
    )
    store_manager.save(
        SessionSnapshot(
            study=session.study,
            series=session.series,
            volumes=session.volumes,
            artifacts=outcome.artifacts + tuple(accepted_records),
            payload_documents=documents,
            provenance=merged_provenance,
            log=session.log,
            skipped_instances=session.skipped_instances,
            pixel_blobs={
                result.pixel_blob_sha256: blob_loader.read_blob(result.pixel_blob_sha256)
                for result in mpr_results
            },
        )
    )
    restored = store_manager.reopen()
    source_manifest = restored.series[0].original_dicom_ref.manifest_sha256
    return ViewerSessionFixture(
        workspace=workspace,
        dicom_dir=Path(source),
        store=store_manager,
        session=restored,
        mpr_artifacts=tuple(
            artifact for artifact in restored.artifacts if artifact.kind == ArtifactKind.MPR_VIEW
        ),
        mpr_results=tuple(mpr_results),
        mpr_provenance=tuple(mpr_provenance),
        source_manifest_sha256=source_manifest,
        study_id=f"study_id:{restored.study.study_id}",
    )


def _read_instance(restored, file_path: str):
    """Re-read one source instance through the governed reader (read-only)."""

    from pathlib import Path as _Path

    from core.dicom import read_dicom_instance

    return read_dicom_instance(
        _Path(restored.series[0].original_dicom_ref.root_path) / file_path
    )


__all__ = ["ViewerSessionFixture", "build_viewer_session"]
