"""T-6 (core half): the human accept / edit (re-seed) / reject seam.

Rule F: the seam is HUMAN-only and append-only; a rejected path is
invalidated with a STALE fan-out to its dependents; an edit derives NEW
artifacts (never an in-place edit) and advances the geometry version.  The
viewer overlay read surface exposes the resulting ``review_state`` and refuses
when no ACTIVE centerline remains.
"""

from __future__ import annotations

import pytest

from apps.viewer.overlay import (
    REFUSAL_UNAVAILABLE,
    CoronaryOverlayRefusalError,
    build_coronary_overlay_document,
)
from core.artifacts import DerivedArtifact, DerivedArtifactStore, InvalidationDAG
from core.coronary import (
    CoronaryDecision,
    Seed,
    VesselnessParameters,
    apply_coronary_decision,
    build_centerline,
)
from core.enums import ActorType, ArtifactKind, ArtifactStatus, ProducerType, ReviewState
from core.errors import StateTransitionError
from core.events import EventLog
from tests.core.coronary.support import (
    FRAME_OF_REFERENCE_UID,
    STUDY_INSTANCE_UID,
    ULIDS,
    SliceFixture,
    helix_points,
)

_AT = "2026-09-14T00:00:00Z"
_SESSION_ID = "01HZZZZZZZZZZZZZZZZZZZZZ40"
_EMPTY_GEOMETRY: dict[str, list[object]] = {"AXIAL": [], "CORONAL": [], "SAGITTAL": []}
_NEW_CENTERLINE_ID = "01HZZZZZZZZZZZZZZZZZZZZZ13"
_NEW_CENTERLINE_PROVENANCE = "01HZZZZZZZZZZZZZZZZZZZZZ14"
_CORRECTION_ID = "01HZZZZZZZZZZZZZZZZZZZZZ15"
_CORRECTION_PROVENANCE = "01HZZZZZZZZZZZZZZZZZZZZZ16"


def _store(slice_fixture: SliceFixture) -> DerivedArtifactStore:
    store = DerivedArtifactStore(EventLog(), InvalidationDAG())
    store.add(slice_fixture.centerline.artifact)
    store.add(slice_fixture.cpr.artifact)  # bound to the centerline geometry
    return store


def _overlay(
    artifacts: tuple[DerivedArtifact, ...],
    documents: dict[str, dict[str, object]],
) -> dict[str, object]:
    return build_coronary_overlay_document(
        session_id=_SESSION_ID,
        artifacts=artifacts,
        payload_documents=documents,
        geometry=_EMPTY_GEOMETRY,
        visible=True,
    )


def test_decision_vocabulary_is_closed() -> None:
    assert {decision.value for decision in CoronaryDecision} == {"ACCEPT", "EDIT", "REJECT"}


def test_a_non_centerline_subject_is_refused(slice_fixture: SliceFixture) -> None:
    store = _store(slice_fixture)
    with pytest.raises(StateTransitionError):
        apply_coronary_decision(
            store,
            centerline_id=slice_fixture.cpr.artifact.artifact_id,
            decision=CoronaryDecision.ACCEPT,
            actor_role=ActorType.HUMAN,
            reviewed_by="dr-human",
        )


def test_accept_is_human_only(slice_fixture: SliceFixture) -> None:
    store = _store(slice_fixture)
    centerline_id = slice_fixture.centerline.artifact.artifact_id
    with pytest.raises(StateTransitionError):
        apply_coronary_decision(
            store,
            centerline_id=centerline_id,
            decision=CoronaryDecision.ACCEPT,
            actor_role=ActorType.ENGINE,
            reviewed_by="engine",
        )
    assert store.get(centerline_id).review.review_state is ReviewState.PREDICTED


def test_accept_moves_to_accepted_and_the_overlay_exposes_it(
    slice_fixture: SliceFixture,
) -> None:
    store = _store(slice_fixture)
    centerline_id = slice_fixture.centerline.artifact.artifact_id
    outcome = apply_coronary_decision(
        store,
        centerline_id=centerline_id,
        decision=CoronaryDecision.ACCEPT,
        actor_role=ActorType.HUMAN,
        reviewed_by="dr-human",
        evidence_ref="path reviewed on the phantom",
        at=_AT,
    )
    assert outcome.review_state == "ACCEPTED"
    assert outcome.artifact_status == "ACTIVE"
    assert store.get(centerline_id).review.review_state is ReviewState.ACCEPTED
    overlay = _overlay(
        (store.get(centerline_id),),
        {centerline_id: slice_fixture.centerline.payload},
    )
    assert overlay["review_state"] == "ACCEPTED"


def test_reject_invalidates_and_fans_stale(slice_fixture: SliceFixture) -> None:
    store = _store(slice_fixture)
    centerline_id = slice_fixture.centerline.artifact.artifact_id
    cpr_id = slice_fixture.cpr.artifact.artifact_id
    outcome = apply_coronary_decision(
        store,
        centerline_id=centerline_id,
        decision=CoronaryDecision.REJECT,
        actor_role=ActorType.HUMAN,
        reviewed_by="dr-human",
        evidence_ref="path rejected: wrong vessel",
        at=_AT,
    )
    assert outcome.review_state == "REJECTED"
    assert outcome.artifact_status == "INVALIDATED"
    assert store.get(centerline_id).status is ArtifactStatus.INVALIDATED
    assert cpr_id in outcome.staled_artifact_ids
    assert store.get(cpr_id).status is ArtifactStatus.STALE
    with pytest.raises(CoronaryOverlayRefusalError) as excinfo:
        _overlay(
            (store.get(centerline_id),),
            {centerline_id: slice_fixture.centerline.payload},
        )
    assert excinfo.value.code == REFUSAL_UNAVAILABLE


def test_edit_reseeds_into_new_artifacts(slice_fixture: SliceFixture) -> None:
    store = _store(slice_fixture)
    centerline_id = slice_fixture.centerline.artifact.artifact_id
    cpr_id = slice_fixture.cpr.artifact.artifact_id
    helix = helix_points()
    probe = helix[80]
    new_seed = Seed(
        i=int(round(probe[0])),
        j=int(round(probe[1])),
        k=int(round(probe[2])),
        direction=(
            float(helix[100][0] - helix[60][0]),
            float(helix[100][1] - helix[60][1]),
            float(helix[100][2] - helix[60][2]),
        ),
    )

    from core.coronary import CenterlineBuildResult

    rebuilt_holder: list[CenterlineBuildResult] = []

    def rebuild():
        rebuilt = build_centerline(
            slice_fixture.volume,
            slice_fixture.transform,
            slice_fixture.frame,
            loader=slice_fixture.loader,
            seed=new_seed,
            parameters=VesselnessParameters(scales_mm=(1.0, 2.0), threshold=0.09),
            provenance_ref=f"provenance:{_NEW_CENTERLINE_PROVENANCE}",
            transform_artifact_ref=f"coordinate_transform:{ULIDS['transform_artifact']}",
            study_instance_uid=STUDY_INSTANCE_UID,
            frame_of_reference_uid=FRAME_OF_REFERENCE_UID,
            scalar_domain="HU",
            source_array=slice_fixture.raw,
            vessel_label="VESSEL:RCA",
            artifact_id=_NEW_CENTERLINE_ID,
            created_by="ccta-coronary-fixture",
        )
        rebuilt_holder.append(rebuilt)
        return rebuilt

    outcome = apply_coronary_decision(
        store,
        centerline_id=centerline_id,
        decision=CoronaryDecision.EDIT,
        actor_role=ActorType.HUMAN,
        reviewed_by="dr-human",
        evidence_ref="re-seeded at helix[80]",
        at=_AT,
        correction_payload={
            "schema_version": "ccta.payload.coronary_correction.v1",
            "decision": "EDIT",
            "seed_voxel": [new_seed.i, new_seed.j, new_seed.k],
            "seed_direction_world": list(new_seed.unit_direction()),
            "reason": "re-seed the path start",
        },
        rebuild=rebuild,
        correction_provenance_ref=f"provenance:{_CORRECTION_PROVENANCE}",
        correction_study_ref="study_id:01HZZZZZZZZZZZZZZZZZZZZZ00",
        correction_artifact_id=_CORRECTION_ID,
    )
    assert outcome.new_centerline_id == _NEW_CENTERLINE_ID
    correction = store.get(_CORRECTION_ID)
    assert correction.kind is ArtifactKind.CORRECTION
    assert correction.producer.producer_type is ProducerType.HUMAN
    assert store.get(centerline_id).status is ArtifactStatus.SUPERSEDED
    assert store.get(cpr_id).status is ArtifactStatus.STALE
    assert cpr_id in outcome.staled_artifact_ids
    assert store.invalidation_dag.geometry_version(centerline_id) == 2
    successor = store.get(_NEW_CENTERLINE_ID)
    assert successor.status is ArtifactStatus.ACTIVE
    assert successor.review.review_state is ReviewState.PREDICTED
    assert len(rebuilt_holder) == 1
    # The overlay selects the NEW ACTIVE centerline and reports it read-only;
    # the SUPERSEDED predecessor is not a candidate.
    overlay = _overlay(
        (store.get(centerline_id), successor),
        {
            centerline_id: slice_fixture.centerline.payload,
            _NEW_CENTERLINE_ID: dict(rebuilt_holder[0].payload),
        },
    )
    assert overlay["centerline_ref"] == f"centerline_artifact_id:{_NEW_CENTERLINE_ID}"
    assert overlay["review_state"] == "PREDICTED"
