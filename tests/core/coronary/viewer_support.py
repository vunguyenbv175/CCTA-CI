"""Viewer-session fixtures carrying the CORONARY_CENTERLINE_001 slice.

The coronary viewer session is built over the analytic phantom volume: two
accepted MPR frames per plane are minted directly from the phantom (no DICOM
involved), the sealed CENTERLINE artifact is added, and the session is saved
through the real WS-02 session store and reopened.  The engine therefore
renders phantom anatomy with the derived path overlay - the same volume
throughout, no cross-study fixture mismatch.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from apps.workstation.session_store import (
    DerivedStateStoreManager,
    RestoredSession,
    SessionSnapshot,
    encode_value,
)
from core.artifacts import (
    ClaimBinding,
    DerivedArtifact,
    DerivedArtifactStore,
    InvalidationDAG,
    Lineage,
    PayloadRef,
    Producer,
    create_derived_artifact,
)
from core.digest import canonical_digest
from core.enums import (
    ActorType,
    ArtifactKind,
    ProducerType,
    ProvenanceEventType,
    ReleaseLane,
    ReviewState,
)
from core.provenance import (
    Actor,
    BlobRef,
    Environment,
    InputRef,
    NotApplicable,
    ProvenanceRecord,
    create_provenance,
)
from tests.apps.viewer_pipeline import ViewerSessionFixture, build_viewer_session
from tests.core.coronary.support import (
    STUDY_INSTANCE_UID,
    ULIDS,
    SliceFixture,
    build_centerline_for_vessel,
    build_slice_fixture,
    vessel_assignment_ids,
)
from visualization.cpr import cpr_provenance
from visualization.mpr import build_mpr

_ULID_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
_FIXTURE_AT = "2026-09-14T00:00:00Z"

#: In-plane pixel geometry of the phantom MPR frames: 32x32 at 1 mm, placed
#: to cover the phantom helix (world x/y/z span roughly 15..33 mm).
_MPR_MATRIX = (32, 32)
_MPR_SPACING = (1.0, 1.0)
_MPR_ORIGINS: tuple[tuple[str, tuple[float, float, float]], ...] = (
    ("AXIAL", (8.0, 8.0, 20.0)),
    ("AXIAL", (8.0, 8.0, 30.0)),
    ("CORONAL", (8.0, 20.0, 8.0)),
    ("CORONAL", (8.0, 28.0, 8.0)),
    ("SAGITTAL", (20.0, 8.0, 8.0)),
    ("SAGITTAL", (28.0, 8.0, 8.0)),
)


def _ulid_number(number: int) -> str:
    value = number
    chars = []
    for _shift in range(26):
        chars.append(_ULID_ALPHABET[value & 0x1F])
        value >>= 5
    return "".join(reversed(chars))


def _fixture_ulid(offset: int) -> str:
    """Deterministic ULID-shaped ids for the fixture MPR artifacts."""

    return _ulid_number(0x01A9_0000 + offset)


@dataclass(frozen=True)
class CoronaryViewerFixture:
    """Everything the coronary viewer tests need from one governed session."""

    workspace: Path
    base: ViewerSessionFixture
    slice: SliceFixture
    store: DerivedStateStoreManager
    session: RestoredSession
    mpr_artifacts: tuple[DerivedArtifact, ...]
    centerline_id: str
    cpr_id: str
    code_commit_sha: str
    vessel_ids: dict[str, tuple[str, str]] | None = None


def _phantom_mpr_provenance(
    slice_fixture: SliceFixture,
    *,
    mpr_id: str,
    plane: str,
    origin: tuple[float, float, float],
    volume_provenance_ref: str,
    transform_provenance_ref: str,
    code_commit_sha: str,
    provenance_id: str,
) -> ProvenanceRecord:
    return create_provenance(
        subject_ref=mpr_id,
        event_type=ProvenanceEventType.DERIVED,
        actor=Actor(
            actor_type=ActorType.ENGINE, actor_id="ccta-coronary-fixture"
        ),
        inputs=(
            InputRef(
                ref=f"volume_id:{slice_fixture.volume.volume_id}",
                kind="volume",
                sha256_at_use=slice_fixture.volume.blob_ref.sha256,
            ),
        ),
        preprocessing=f"phantom-mpr-view:{plane}-nearest",
        parameters_ref=BlobRef(
            blob_ref="coronary-phantom-mpr-parameters",
            sha256=canonical_digest({"plane": plane, "origin": list(origin)}),
        ),
        human_edit=NotApplicable(value=None, reason="deterministic fixture MPR"),
        upstream_refs=(volume_provenance_ref, transform_provenance_ref),
        environment=Environment(
            software_version="ccta-coronary-fixture",
            code_commit_sha=code_commit_sha,
            runtime="python-3.12 (coronary fixture)",
            deterministic_seed=None,
        ),
        created_by="ccta-coronary-fixture",
        at=_FIXTURE_AT,
        provenance_id=provenance_id,
    )


def build_coronary_viewer_fixture(
    workspace: Path, *, repo_root: Path, code_commit_sha: str
) -> CoronaryViewerFixture:
    """Import scaffolding -> phantom slice -> phantom MPRs -> save/reopen."""

    workspace = Path(workspace)
    base = build_viewer_session(workspace)
    session = base.session
    slice_fixture = build_slice_fixture(
        workspace / "coronary",
        blobs_dir=session.paths.blobs,
        code_commit_sha=code_commit_sha,
        union_extra_vessels=True,
    )
    artifact_store = DerivedArtifactStore(session.log, InvalidationDAG())
    documents = dict(session.payload_documents)
    documents[slice_fixture.centerline.artifact.artifact_id] = dict(
        slice_fixture.centerline.payload
    )
    provenance_records: list[ProvenanceRecord] = [
        slice_fixture.volume_provenance,
        slice_fixture.transform_provenance,
        slice_fixture.centerline_provenance,
    ]
    mpr_artifacts: list[DerivedArtifact] = []
    for index, (plane, origin) in enumerate(_MPR_ORIGINS):
        artifact_id = _fixture_ulid(index)
        provenance_id = _fixture_ulid(100 + index)
        provenance_record = _phantom_mpr_provenance(
            slice_fixture,
            mpr_id=artifact_id,
            plane=plane,
            origin=origin,
            volume_provenance_ref=slice_fixture.volume_provenance.provenance_id,
            transform_provenance_ref=slice_fixture.transform_provenance.provenance_id,
            code_commit_sha=code_commit_sha,
            provenance_id=provenance_id,
        )
        result = build_mpr(
            slice_fixture.volume,
            slice_fixture.transform,
            plane=plane,
            origin=origin,
            in_plane_spacing_mm=_MPR_SPACING,
            matrix_size=_MPR_MATRIX,
            method="NEAREST",
            out_of_bounds_fill_value=0.0,
            loader=slice_fixture.loader,
            transform_artifact_ref=(
                f"coordinate_transform:{ULIDS['transform_artifact']}"
            ),
            provenance_ref=f"provenance:{provenance_id}",
            study_instance_uid=STUDY_INSTANCE_UID,
            series_instance_uid=slice_fixture.transform.series_instance_uid,
            frame_of_reference_uid=slice_fixture.transform.frame_of_reference_uid,
            artifact_id=artifact_id,
            created_by="ccta-coronary-fixture",
        )
        artifact_store.add(result.artifact)
        artifact_store.apply_review_transition(
            artifact_id,
            ReviewState.ACCEPTED,
            actor_role=ActorType.HUMAN,
            reviewed_by="coronary-fixture-acceptor",
            evidence_ref="synthetic phantom MPR for the coronary overlay",
            at=_FIXTURE_AT,
        )
        mpr_artifacts.append(artifact_store.get(artifact_id))
        provenance_records.append(provenance_record)
        documents[artifact_id] = dict(result.payload)

    artifact_store.add(slice_fixture.centerline.artifact)
    # The slice chain's persisted versioned transform: the F5 EDIT re-seed
    # route re-derives through the SAME persisted WORLD->VOXEL binding the
    # sealed centerline records; a session without it must refuse rather than
    # fabricate a transform identity.
    transform_document = encode_value(slice_fixture.transform)
    assert isinstance(transform_document, dict)
    transform_artifact = create_derived_artifact(
        study_ref=slice_fixture.volume.study_ref,
        kind=ArtifactKind.COORDINATE_TRANSFORM,
        payload=PayloadRef(
            payload_schema="ccta.payload.coordinate_transform.v1",
            blob_sha256=canonical_digest(transform_document),
            codec="json",
            inline_summary=(("transform_id", slice_fixture.transform.transform_id),),
        ),
        lineage=Lineage(input_volume_refs=(slice_fixture.volume.volume_id,)),
        claim_binding=ClaimBinding(claim_id="CLM-VIZ-001", claim_version="v0-draft"),
        producer=Producer(
            producer_type=ProducerType.DETERMINISTIC_ENGINE,
            algorithm="dicom-frame-transform",
            algorithm_version="v1",
            parameters_blob_sha256=canonical_digest(encode_value(slice_fixture.frame)),
        ),
        release_lane=ReleaseLane.PERSONAL_RESEARCH,
        provenance_ref=slice_fixture.transform_provenance.provenance_id,
        created_by="ccta-coronary-fixture",
        created_at=_FIXTURE_AT,
        artifact_id=ULIDS["transform_artifact"],
    )
    artifact_store.add(transform_artifact)
    documents[transform_artifact.artifact_id] = transform_document
    # CORONARY_REVIEW_VIEW_V1 (Decision B): the accepted CPR computation
    # persists as its OWN view artifact through the narrowly admitted
    # ccta.cpr.view.v1 seam (never as a fake orthogonal MPR).  The CPR
    # artifact is bound to the centerline geometry (geometry_base), so the
    # EDIT/REJECT STALE fan-out reaches it exactly like the overlay.
    cpr_provenance_id = _fixture_ulid(120)
    cpr_provenance_record = cpr_provenance(
        subject_ref=f"mpr_artifact_id:{ULIDS['cpr_artifact']}",
        volume=slice_fixture.volume,
        centerline_artifact=artifact_store.get(slice_fixture.centerline.artifact.artifact_id),
        sampling_parameters_sha256=str(slice_fixture.cpr.payload["sampling_parameters_sha256"]),
        volume_provenance_ref=slice_fixture.volume_provenance.provenance_id,
        centerline_provenance_ref=slice_fixture.centerline_provenance.provenance_id,
        software_version="ccta-coronary-fixture",
        code_commit_sha=code_commit_sha,
        runtime="python-3.12 (coronary fixture)",
        created_by="ccta-coronary-fixture",
        at=_FIXTURE_AT,
        provenance_id=cpr_provenance_id,
    )
    artifact_store.add(slice_fixture.cpr.artifact)
    documents[slice_fixture.cpr.artifact.artifact_id] = dict(slice_fixture.cpr.payload)
    provenance_records.append(cpr_provenance_record)
    # CORONARY_MULTI_VESSEL_NAV: two extra labeled paths (LAD/LCx;
    # FIXTURE_DERIVED) on the SAME phantom volume, each with its own CPR
    # view bound to its own centerline geometry (independent STALE fan-out).
    # The base RCA path above is untouched (legacy #28 behavior preserved).
    vessel_ids: dict[str, tuple[str, str]] = {
        "VESSEL:RCA": (
            slice_fixture.centerline.artifact.artifact_id,
            slice_fixture.cpr.artifact.artifact_id,
        )
    }
    for _vessel_label in ("VESSEL:LAD", "VESSEL:LCX"):
        _ids = vessel_assignment_ids(_vessel_label)
        _cl, _cl_prov, _cpr, _cpr_prov = build_centerline_for_vessel(
            slice_fixture.volume,
            slice_fixture.transform,
            slice_fixture.frame,
            loader=slice_fixture.loader,
            vessel_label=_vessel_label,
            artifact_ids=_ids,
            code_commit_sha=code_commit_sha,
        )
        artifact_store.add(_cl.artifact)
        artifact_store.apply_review_transition(
            _cl.artifact.artifact_id,
            ReviewState.ACCEPTED,
            actor_role=ActorType.HUMAN,
            reviewed_by="coronary-fixture-acceptor",
            evidence_ref=f"synthetic phantom centerline for {_vessel_label}",
            at=_FIXTURE_AT,
        )
        documents[_cl.artifact.artifact_id] = dict(_cl.payload)
        artifact_store.add(_cpr.artifact)
        documents[_cpr.artifact.artifact_id] = dict(_cpr.payload)
        provenance_records.extend((_cl_prov, _cpr_prov))
        vessel_ids[_vessel_label] = (
            _cl.artifact.artifact_id,
            _cpr.artifact.artifact_id,
        )
    retained = tuple(
        artifact
        for artifact in session.artifacts
        if artifact.kind is not ArtifactKind.MPR_VIEW
    )
    base.store.save(
        SessionSnapshot(
            study=session.study,
            series=session.series,
            volumes=session.volumes + (slice_fixture.volume,),
            artifacts=retained
            + tuple(mpr_artifacts)
            + (
                transform_artifact,
                artifact_store.get(slice_fixture.centerline.artifact.artifact_id),
                artifact_store.get(slice_fixture.cpr.artifact.artifact_id),
            )
            + tuple(
                artifact_store.get(_cl_id)
                for _label, (_cl_id, _cpr_id) in vessel_ids.items()
                if _label != "VESSEL:RCA"
            )
            + tuple(
                artifact_store.get(_cpr_id)
                for _label, (_cl_id, _cpr_id) in vessel_ids.items()
                if _label != "VESSEL:RCA"
            ),
            payload_documents=documents,
            provenance=session.provenance + tuple(provenance_records),
            log=session.log,
            skipped_instances=session.skipped_instances,
        )
    )
    restored = base.store.reopen()
    return CoronaryViewerFixture(
        workspace=workspace,
        base=base,
        slice=slice_fixture,
        store=base.store,
        session=restored,
        mpr_artifacts=tuple(mpr_artifacts),
        centerline_id=slice_fixture.centerline.artifact.artifact_id,
        cpr_id=slice_fixture.cpr.artifact.artifact_id,
        code_commit_sha=code_commit_sha,
        vessel_ids=vessel_ids,
    )


def ambiguous_centerline_session(
    session: RestoredSession,
) -> RestoredSession:
    """The same session with a SECOND ACTIVE unlabeled CENTERLINE (F8 input).

    Duplicates the RCA path's artifact id/payload under a fresh id with the
    label REMOVED (``vessel_label=None``): an unlabeled tie for the legacy
    ``None`` selection that the overlay must refuse as AMBIGUOUS instead of
    silently picking one.  Payload documents travel with the session, so the
    duplicate's document is cloned alongside its artifact.
    """

    from tests.core.coronary.support import ULIDS

    base_id = ULIDS["centerline_artifact"]
    centerlines = [
        artifact
        for artifact in session.artifacts
        if artifact.kind is ArtifactKind.CENTERLINE
        and artifact.artifact_id == base_id
    ]
    if len(centerlines) != 1:
        raise AssertionError(
            "the base coronary session must carry the RCA CENTERLINE artifact"
        )
    duplicate_id = _fixture_ulid(240)
    duplicate = replace(centerlines[0], artifact_id=duplicate_id)
    base_document = dict(session.payload_documents[base_id])
    base_document["vessel_label"] = None
    documents = dict(session.payload_documents)
    documents[duplicate_id] = base_document
    return replace(
        session,
        artifacts=session.artifacts + (duplicate,),
        payload_documents=documents,
    )


def with_ambiguous_centerlines(session: RestoredSession) -> RestoredSession:
    """Back-compat alias of :func:`ambiguous_centerline_session`."""

    return ambiguous_centerline_session(session)


__all__ = [
    "CoronaryViewerFixture",
    "ambiguous_centerline_session",
    "build_coronary_viewer_fixture",
    "with_ambiguous_centerlines",
]
