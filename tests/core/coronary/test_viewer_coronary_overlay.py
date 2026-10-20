"""T-6 (viewer half): the coronary overlay read surface + the toggle.

The engine-level tests run against the real governed viewer session (F11
build identity resolved from the executing clean checkout) with the sealed
CENTERLINE artifact restored from the WS-02 session store:

* ``GET /session/:id/coronary`` serves the closed overlay document whose
  projections bind to the current LAYER-G frames;
* a session without an ACTIVE centerline refuses with the typed
  ``CENTERLINE_OVERLAY_UNAVAILABLE`` (HTTP 409);
* the ``SET_OVERLAY_VISIBILITY`` toggle is a closed-enum grammar extension
  (unknown ids / non-boolean values are ``UNKNOWN_PARAM``), persists, and
  survives a session reopen.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pytest

from apps.viewer.build_identity import resolve_build_identity
from apps.viewer.engine import ViewerEngine
from apps.viewer.grammar import (
    ACTION_PARAM_KEYS,
    ACTIONS,
    SessionGrammarView,
    ViewerRejectionError,
    validate_action,
)
from apps.viewer.overlay import (
    CORONARY_OVERLAY_ID,
    REFUSAL_UNAVAILABLE,
    CoronaryOverlayRefusalError,
    validate_coronary_overlay_document,
)
from apps.viewer.server import ViewerServer
from apps.workstation.session_store import RestoredSession
from core.digest import canonical_json
from core.enums import ArtifactKind
from tests.apps.test_viewer_e2e import _apply, _deterministic_ulids
from tests.core.coronary.support import helix_points
from tests.core.coronary.viewer_support import (
    CoronaryViewerFixture,
    build_coronary_viewer_fixture,
)

_AT = "2026-09-14T00:00:00Z"
_TOGGLE_NONCE = "coronary-overlay-toggle-0001"


@dataclass(frozen=True)
class _EngineCase:
    engine: ViewerEngine
    fixture: CoronaryViewerFixture


@pytest.fixture(autouse=True)
def _run_inside_checkout(checkout_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(checkout_root)


@pytest.fixture(scope="module")
def coronary_viewer(
    tmp_path_factory: pytest.TempPathFactory,
) -> CoronaryViewerFixture:
    checkout = Path(__file__).resolve().parents[3]
    code_commit_sha = resolve_build_identity(checkout)
    return build_coronary_viewer_fixture(
        tmp_path_factory.mktemp("coronary-viewer"),
        repo_root=checkout,
        code_commit_sha=code_commit_sha,
    )


@pytest.fixture()
def case(coronary_viewer: CoronaryViewerFixture, checkout_root: Path) -> _EngineCase:
    engine = ViewerEngine(
        coronary_viewer.session,
        id_factory=_deterministic_ulids(),
        now=lambda: _AT,
        repo_root=checkout_root,
    )
    engine.open()
    _select_rca_vessel(engine)
    return _EngineCase(engine=engine, fixture=coronary_viewer)


def _strip_centerline(session: RestoredSession) -> RestoredSession:
    """The same session without its CENTERLINE artifact (F11-clean input)."""

    return RestoredSession(
        paths=session.paths,
        study=session.study,
        series=session.series,
        volumes=session.volumes,
        artifacts=tuple(
            artifact
            for artifact in session.artifacts
            if artifact.kind is not ArtifactKind.CENTERLINE
        ),
        payload_documents=dict(session.payload_documents),
        provenance=session.provenance,
        log=session.log,
        skipped_instances=session.skipped_instances,
        projections=dict(session.projections),
        session_digest=session.session_digest,
        pixel_blobs=dict(session.pixel_blobs),
    )


def test_overlay_document_is_closed_and_projected(case: _EngineCase) -> None:
    document = case.engine.coronary_overlay_document()
    assert validate_coronary_overlay_document(document) == document
    assert document["overlay_id"] == CORONARY_OVERLAY_ID
    assert document["visible"] is True
    assert document["review_state"] == "PREDICTED"
    assert document["vessel_label"] == "VESSEL:RCA"
    payload = case.fixture.slice.centerline.payload
    assert document["point_count"] == len(cast(list[Any], payload["points_world"]))
    assert document["s_mm"] == payload["s_mm"]
    assert document["centerline_ref"] == (
        f"centerline_artifact_id:{case.fixture.centerline_id}"
    )
    accepted_ids = {
        artifact.artifact_id
        for artifact in case.fixture.session.artifacts
        if artifact.kind is ArtifactKind.MPR_VIEW
        and artifact.review.review_state.value in {"ACCEPTED", "CONFIRMED_FINAL"}
    }
    assert len(accepted_ids) == 6
    projections = cast(dict[str, list[dict[str, Any]]], document["plane_projections"])
    for plane in ("AXIAL", "CORONAL", "SAGITTAL"):
        frames = projections[plane]
        assert len(frames) == 2
        coordinates: list[float] = []
        for index, frame in enumerate(frames):
            assert frame["frame_index"] == index
            mpr_id = str(frame["source_mpr_ref"]).split(":", 1)[1]
            assert mpr_id in accepted_ids
            mpr_document = case.fixture.session.payload_documents[mpr_id]
            coordinate = mpr_document["plane_coordinate_mm"]
            assert isinstance(coordinate, (int, float)) and not isinstance(
                coordinate, bool
            )
            coordinates.append(float(coordinate))
            assert len(frame["columns_rows"]) == document["point_count"]
            assert len(frame["out_of_plane_mm"]) == document["point_count"]
        assert coordinates == sorted(coordinates)


def test_overlay_unavailable_without_a_centerline(
    case: _EngineCase, checkout_root: Path
) -> None:
    stripped = _strip_centerline(case.fixture.session)
    engine = ViewerEngine(
        stripped,
        id_factory=_deterministic_ulids(),
        now=lambda: _AT,
        repo_root=checkout_root,
    )
    engine.open()
    with pytest.raises(CoronaryOverlayRefusalError) as excinfo:
        engine.coronary_overlay_document()
    assert excinfo.value.code == REFUSAL_UNAVAILABLE


def test_coronary_route_serves_document_and_typed_refusal(
    case: _EngineCase, checkout_root: Path
) -> None:
    server = ViewerServer()
    server.add_engine(case.engine)
    _host, port = server.start_background()
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/session/{case.engine.session_id()}/coronary",
            timeout=5,
        ) as response:
            assert response.status == 200
            document = json.loads(response.read())
        assert validate_coronary_overlay_document(document) == document
        assert document["session_id"] == case.engine.session_id()

        stripped = _strip_centerline(case.fixture.session)
        engine = ViewerEngine(
            stripped,
            id_factory=_deterministic_ulids(),
            now=lambda: _AT,
            repo_root=checkout_root,
        )
        engine.open()
        server.add_engine(engine)
        try:
            urllib.request.urlopen(
                f"http://127.0.0.1:{port}/session/{engine.session_id()}/coronary",
                timeout=5,
            )
            raise AssertionError("the overlay GET must refuse without a centerline")
        except urllib.error.HTTPError as error:
            assert error.code == 409
            refusal = json.loads(error.read())
        assert refusal["error"]["code"] == REFUSAL_UNAVAILABLE
    finally:
        server.stop()


def test_overlay_toggle_persists_across_reopen(
    case: _EngineCase, coronary_viewer: CoronaryViewerFixture, checkout_root: Path
) -> None:
    engine = case.engine
    session_id = engine.session_id()
    assert engine.state_document()["overlay_visibility"] == {CORONARY_OVERLAY_ID: True}
    assert engine.state_document()["overlay_mode"] == "SLICE_LOCAL"
    state = _apply(
        engine,
        session_id,
        "SET_OVERLAY_VISIBILITY",
        {"overlay_id": CORONARY_OVERLAY_ID, "visible": False, "mode": "SLICE_LOCAL"},
        view={"plane": "AXIAL", "frame_index": 0},
        nonce=_TOGGLE_NONCE,
        parent=_last_select_nonce(engine),
    )
    assert state["overlay_visibility"] == {CORONARY_OVERLAY_ID: False}
    assert state["overlay_mode"] == "SLICE_LOCAL"
    document = engine.coronary_overlay_document()
    assert document["visible"] is False
    assert document["mode"] == "SLICE_LOCAL"
    engine.persist(coronary_viewer.store)

    reopened = coronary_viewer.store.reopen()
    engine2 = ViewerEngine(
        reopened,
        id_factory=_deterministic_ulids(),
        now=lambda: _AT,
        repo_root=checkout_root,
    )
    engine2.open()
    assert engine2.state_document()["overlay_visibility"] == {CORONARY_OVERLAY_ID: False}
    assert engine2.state_document()["overlay_mode"] == "SLICE_LOCAL"


def test_overlay_action_is_a_closed_grammar_extension() -> None:
    assert "SET_OVERLAY_VISIBILITY" in ACTIONS
    assert "SELECT_VESSEL" in ACTIONS
    assert ACTIONS[-1] == "SELECT_VESSEL"
    assert ACTION_PARAM_KEYS["SELECT_VESSEL"] == ("vessel_label",)
    assert ACTION_PARAM_KEYS["SET_OVERLAY_VISIBILITY"] == ("overlay_id", "visible", "mode")
    assert ACTION_PARAM_KEYS["REQUEST_CENTERLINE_DECISION"] == (
        "subject_ref",
        "centerline_ref",
        "decision",
        "seed_voxel",
        "seed_direction_world",
    )
    session = SessionGrammarView(
        session_id="01HZZZZZZZZZZZZZZZZZZZZZ50",
        study_id="study_id:01HZZZZZZZZZZZZZZZZZZZZZ51",
        frame_count_by_plane={"AXIAL": 1, "CORONAL": 1, "SAGITTAL": 1},
        wl_presets_active=(),
        used_nonces=frozenset(),
        accepted_nonces=frozenset(),
    )

    def body(overlay_id: object, visible: object, mode: object = "SLICE_LOCAL") -> bytes:
        return canonical_json(
            {
                "action": "SET_OVERLAY_VISIBILITY",
                "params": {"overlay_id": overlay_id, "visible": visible, "mode": mode},
                "client_nonce": "overlay-visibility-0001",
                "view": {"plane": "AXIAL", "frame_index": 0},
                "lineage": {
                    "session_id": "01HZZZZZZZZZZZZZZZZZZZZZ50",
                    "parent_action_nonce": None,
                },
            }
        ).encode("utf-8")

    with pytest.raises(ViewerRejectionError) as excinfo:
        validate_action(
            body("OTHER", True),
            path_session_id="01HZZZZZZZZZZZZZZZZZZZZZ50",
            session=session,
        )
    assert excinfo.value.code.value == "UNKNOWN_PARAM"
    with pytest.raises(ViewerRejectionError) as excinfo:
        validate_action(
            body(CORONARY_OVERLAY_ID, 1),
            path_session_id="01HZZZZZZZZZZZZZZZZZZZZZ50",
            session=session,
        )
    assert excinfo.value.code.value == "UNKNOWN_PARAM"
    with pytest.raises(ViewerRejectionError) as excinfo:
        validate_action(
            body(CORONARY_OVERLAY_ID, True, "EVERYTHING"),
            path_session_id="01HZZZZZZZZZZZZZZZZZZZZZ50",
            session=session,
        )
    assert excinfo.value.code.value == "UNKNOWN_PARAM"
    parsed = validate_action(
        body(CORONARY_OVERLAY_ID, False),
        path_session_id="01HZZZZZZZZZZZZZZZZZZZZZ50",
        session=session,
    )
    assert parsed.params == {
        "overlay_id": CORONARY_OVERLAY_ID,
        "visible": False,
        "mode": "SLICE_LOCAL",
    }


# ---------------------------------------------------------------------------
# att0002 F5: the REQUEST_CENTERLINE_DECISION end-to-end route
# ---------------------------------------------------------------------------

_DECISION_ACCEPT_NONCE = "centerline-decision-accept-0001"
_DECISION_REJECT_NONCE = "centerline-decision-reject-0001"
_DECISION_EDIT_NONCE = "centerline-decision-edit-0001"
_QC_CORRECTION_NONCE = "coronary-qc-correction-0001"


def _post_action(port: int, session_id: str, body: bytes) -> tuple[int, dict[str, object]]:
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/session/{session_id}/action",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Origin": f"http://127.0.0.1:{port}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def _decision_body(
    *,
    session_id: str,
    study_id: str,
    nonce: str,
    decision: str,
    centerline_ref: str,
    seed_voxel: list[int] | None = None,
    seed_direction_world: list[float] | None = None,
    parent: str | None = None,
) -> bytes:
    return canonical_json(
        {
            "action": "REQUEST_CENTERLINE_DECISION",
            "params": {
                "subject_ref": f"study_id:{study_id}",
                "centerline_ref": centerline_ref,
                "decision": decision,
                "seed_voxel": seed_voxel,
                "seed_direction_world": seed_direction_world,
            },
            "client_nonce": nonce,
            "view": {"plane": "AXIAL", "frame_index": 0},
            "lineage": {"session_id": session_id, "parent_action_nonce": parent},
        }
    ).encode("utf-8")


def _study_id(case: _EngineCase) -> str:
    return str(case.fixture.session.study.study_id)


def test_centerline_decision_accept_routes_through_http(case: _EngineCase) -> None:
    """F5: a real POST reaches ``apply_coronary_decision`` (ACCEPT) and the
    overlay read surface exposes the human ACCEPTED state."""

    engine = case.engine
    server = ViewerServer()
    server.add_engine(engine)
    _host, port = server.start_background()
    try:
        assert case.engine.coronary_overlay_document()["review_state"] == "PREDICTED"
        status, state = _post_action(
            port,
            engine.session_id(),
            _decision_body(
                session_id=engine.session_id(),
                study_id=_study_id(case),
                nonce=_DECISION_ACCEPT_NONCE,
                decision="ACCEPT",
                centerline_ref=f"centerline_artifact_id:{case.fixture.centerline_id}",
                parent=_last_select_nonce(engine),
            ),
        )
        assert status == 200
        assert state["action_count"] == 2
        assert engine.new_qc_artifacts() == ()
        document = engine.coronary_overlay_document()
        assert document["review_state"] == "ACCEPTED"
        assert document["centerline_ref"] == (
            f"centerline_artifact_id:{case.fixture.centerline_id}"
        )
    finally:
        server.stop()


def test_centerline_decision_reject_is_typed_and_keeps_qc_untouched(
    case: _EngineCase, checkout_root: Path
) -> None:
    """F5: REJECT invalidates the path (typed overlay refusal afterwards); the
    SEPARATE QC correction route still accepts its own ACTION unchanged."""

    engine = case.engine
    session_id = engine.session_id()
    server = ViewerServer()
    server.add_engine(engine)
    _host, port = server.start_background()
    try:
        status, _state = _post_action(
            port,
            session_id,
            _decision_body(
                session_id=session_id,
                study_id=_study_id(case),
                nonce=_DECISION_REJECT_NONCE,
                decision="REJECT",
                centerline_ref=f"centerline_artifact_id:{case.fixture.centerline_id}",
                parent=_last_select_nonce(engine),
            ),
        )
        assert status == 200
        with pytest.raises(CoronaryOverlayRefusalError) as excinfo:
            engine.coronary_overlay_document()
        assert excinfo.value.code == REFUSAL_UNAVAILABLE
        # A second decision on the same (now non-ACTIVE) path is a typed
        # refusal, never a silent re-application.
        status, refusal = _post_action(
            port,
            session_id,
            _decision_body(
                session_id=session_id,
                study_id=_study_id(case),
                nonce="centerline-decision-reject-0002",
                decision="ACCEPT",
                centerline_ref=f"centerline_artifact_id:{case.fixture.centerline_id}",
                parent=_DECISION_REJECT_NONCE,
            ),
        )
        assert status == 409
        error = refusal["error"]
        assert isinstance(error, dict)
        assert error["code"] == REFUSAL_UNAVAILABLE
        # QC semantics are unchanged: the QC action still routes into the
        # QC_001-HUMAN seam on this very session.
        qc_body = canonical_json(
            {
                "action": "REQUEST_CORRECTION",
                "params": {
                    "subject_ref": f"study_id:{_study_id(case)}",
                    "proposed_grade": "NOT_ANALYZABLE",
                    "justification_code": "HUMAN_EVIDENCE",
                },
                "client_nonce": _QC_CORRECTION_NONCE,
                "view": {"plane": "AXIAL", "frame_index": 0},
                "lineage": {
                    "session_id": session_id,
                    "parent_action_nonce": _DECISION_REJECT_NONCE,
                },
            }
        ).encode("utf-8")
        status, _state = _post_action(port, session_id, qc_body)
        assert status == 200
        assert len(engine.new_qc_artifacts()) == 1
    finally:
        server.stop()


def test_centerline_decision_edit_reseeds_and_persists(
    case: _EngineCase, coronary_viewer: CoronaryViewerFixture, checkout_root: Path
) -> None:
    """F5: EDIT re-seeds into NEW artifacts (HUMAN CORRECTION + new
    CENTERLINE), the overlay selects the successor, and the snapshot survives
    a WS-02 save/reopen."""

    engine = case.engine
    session_id = engine.session_id()
    old_id = case.fixture.centerline_id
    helix = helix_points()
    probe = helix[80]
    seed_voxel = [
        int(round(float(probe[0]))),
        int(round(float(probe[1]))),
        int(round(float(probe[2]))),
    ]
    seed_direction = [
        float(helix[100][0] - helix[60][0]),
        float(helix[100][1] - helix[60][1]),
        float(helix[100][2] - helix[60][2]),
    ]
    server = ViewerServer()
    server.add_engine(engine)
    _host, port = server.start_background()
    try:
        status, _state = _post_action(
            port,
            session_id,
            _decision_body(
                session_id=session_id,
                study_id=_study_id(case),
                nonce=_DECISION_EDIT_NONCE,
                decision="EDIT",
                centerline_ref=f"centerline_artifact_id:{old_id}",
                seed_voxel=seed_voxel,
                seed_direction_world=seed_direction,
                parent=_last_select_nonce(engine),
            ),
        )
        assert status == 200
        document = engine.coronary_overlay_document()
        new_ref = str(document["centerline_ref"])
        new_id = new_ref.split(":", 1)[1]
        assert new_id != old_id
        assert document["review_state"] == "PREDICTED"
        point_count = document["point_count"]
        assert isinstance(point_count, int) and point_count > 0

        engine.persist(coronary_viewer.store)
    finally:
        server.stop()

    reopened = coronary_viewer.store.reopen()
    by_id = {artifact.artifact_id: artifact for artifact in reopened.artifacts}
    assert new_id in by_id
    assert by_id[new_id].kind is ArtifactKind.CENTERLINE
    assert by_id[new_id].status.value == "ACTIVE"
    assert by_id[old_id].status.value == "SUPERSEDED"
    corrections = [
        artifact
        for artifact in reopened.artifacts
        if artifact.kind is ArtifactKind.CORRECTION
    ]
    assert len(corrections) == 1
    assert corrections[0].producer.producer_type.value == "HUMAN"
    engine2 = ViewerEngine(
        reopened,
        id_factory=_deterministic_ulids(),
        now=lambda: _AT,
        repo_root=checkout_root,
    )
    engine2.open()
    document2 = engine2.coronary_overlay_document()
    assert document2["centerline_ref"] == f"centerline_artifact_id:{new_id}"

_DECISION_EDIT_REPEAT_NONCE_1 = "centerline-decision-edit-repeat-0001"
_DECISION_EDIT_REPEAT_NONCE_2 = "centerline-decision-edit-repeat-0002"

def test_centerline_decision_edit_repeat_targets_the_successor(
    case: _EngineCase, coronary_viewer: CoronaryViewerFixture, checkout_root: Path
) -> None:
    """F5-repeat: a second parent-linked EDIT targets the in-memory successor.

    The successor minted by the first EDIT has its sealed payload only in the
    working documents until persist; the re-seed route must resolve it.  A
    restored-only lookup refused the second EDIT with
    ``CENTERLINE_DECISION_REFUSED`` (HTTP 409).  Both EDITs must be 200 and
    the persisted chain must read old -> successor-1 -> successor-2.
    """

    engine = case.engine
    session_id = engine.session_id()
    old_id = case.fixture.centerline_id
    helix = helix_points()
    first_probe = helix[80]
    first_seed = [
        int(round(float(first_probe[0]))),
        int(round(float(first_probe[1]))),
        int(round(float(first_probe[2]))),
    ]
    first_direction = [
        float(helix[100][0] - helix[60][0]),
        float(helix[100][1] - helix[60][1]),
        float(helix[100][2] - helix[60][2]),
    ]
    second_probe = helix[40]
    second_seed = [
        int(round(float(second_probe[0]))),
        int(round(float(second_probe[1]))),
        int(round(float(second_probe[2]))),
    ]
    second_direction = [
        float(helix[60][0] - helix[20][0]),
        float(helix[60][1] - helix[20][1]),
        float(helix[60][2] - helix[20][2]),
    ]
    server = ViewerServer()
    server.add_engine(engine)
    _host, port = server.start_background()
    try:
        status, state = _post_action(
            port,
            session_id,
            _decision_body(
                session_id=session_id,
                study_id=_study_id(case),
                nonce=_DECISION_EDIT_REPEAT_NONCE_1,
                decision="EDIT",
                centerline_ref=f"centerline_artifact_id:{old_id}",
                seed_voxel=first_seed,
                seed_direction_world=first_direction,
                parent=_last_select_nonce(engine),
            ),
        )
        assert status == 200
        assert state["action_count"] == 2
        document1 = engine.coronary_overlay_document()
        successor1_ref = str(document1["centerline_ref"])
        successor1_id = successor1_ref.split(":", 1)[1]
        assert successor1_id != old_id
        assert document1["review_state"] == "PREDICTED"

        status, state = _post_action(
            port,
            session_id,
            _decision_body(
                session_id=session_id,
                study_id=_study_id(case),
                nonce=_DECISION_EDIT_REPEAT_NONCE_2,
                decision="EDIT",
                centerline_ref=successor1_ref,
                seed_voxel=second_seed,
                seed_direction_world=second_direction,
                parent=_DECISION_EDIT_REPEAT_NONCE_1,
            ),
        )
        assert status == 200
        assert state["action_count"] == 3
        document2 = engine.coronary_overlay_document()
        successor2_ref = str(document2["centerline_ref"])
        successor2_id = successor2_ref.split(":", 1)[1]
        assert successor2_id not in (old_id, successor1_id)
        assert document2["review_state"] == "PREDICTED"
        point_count = document2["point_count"]
        assert isinstance(point_count, int) and point_count > 0

        engine.persist(coronary_viewer.store)
    finally:
        server.stop()

    reopened = coronary_viewer.store.reopen()
    by_id = {artifact.artifact_id: artifact for artifact in reopened.artifacts}
    assert by_id[successor2_id].kind is ArtifactKind.CENTERLINE
    assert by_id[successor2_id].status.value == "ACTIVE"
    assert by_id[successor1_id].status.value == "SUPERSEDED"
    assert by_id[old_id].status.value == "SUPERSEDED"
    corrections = {
        artifact.lineage.input_artifact_refs[0]: artifact
        for artifact in reopened.artifacts
        if artifact.kind is ArtifactKind.CORRECTION
    }
    assert set(corrections) == {old_id, successor1_id}
    assert corrections[old_id].producer.producer_type.value == "HUMAN"
    assert corrections[old_id].review.supersedes_ref == (
        f"centerline_artifact_id:{old_id}"
    )
    assert corrections[successor1_id].producer.producer_type.value == "HUMAN"
    assert corrections[successor1_id].review.supersedes_ref == (
        f"centerline_artifact_id:{successor1_id}"
    )
    engine2 = ViewerEngine(
        reopened,
        id_factory=_deterministic_ulids(),
        now=lambda: _AT,
        repo_root=checkout_root,
    )
    engine2.open()
    document3 = engine2.coronary_overlay_document()
    assert document3["centerline_ref"] == successor2_ref


_RCA_SELECT_NONCES: dict[str, int] = {}
_RCA_SELECT_LAST: dict[str, str] = {}


def _last_select_nonce(engine: object) -> str:
    """The SELECT_VESSEL nonce the case() helper accepted (lineage parent)."""

    from apps.viewer.engine import ViewerEngine as _ViewerEngine

    assert isinstance(engine, _ViewerEngine)
    return _RCA_SELECT_LAST[engine.session_id()]


def _select_rca_vessel(engine: object) -> None:
    """Select the legacy RCA path (multi-vessel fixture migration helper).

    The #28 tests were written against a single-vessel fixture; the
    multi-vessel fixture carries three ACTIVE labeled paths, so the legacy
    single-path assertions must select RCA first (the #28 path).  Nonce and
    lineage chaining are handled per engine session.
    """

    from apps.viewer.engine import ViewerEngine
    from core.digest import canonical_json as _canonical_json

    assert isinstance(engine, ViewerEngine)
    state = engine._require_state()
    accepted = list(state.accepted_nonces)
    parent = accepted[-1] if accepted else None
    count = _RCA_SELECT_NONCES.get(engine.session_id(), 0)
    _RCA_SELECT_NONCES[engine.session_id()] = count + 1
    action = _canonical_json(
        {
            "action": "SELECT_VESSEL",
            "params": {"vessel_label": "VESSEL:RCA"},
            "client_nonce": f"select-rca-legacy-{count:04d}",
            "view": {"plane": "AXIAL", "frame_index": 0},
            "lineage": {
                "session_id": engine.session_id(),
                "parent_action_nonce": parent,
            },
        }
    ).encode("utf-8")
    engine.apply_action_bytes(action, path_session_id=engine.session_id())
    _RCA_SELECT_LAST[engine.session_id()] = f"select-rca-legacy-{count:04d}"

