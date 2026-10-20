"""T2 physician/manual seed workflow, Phases A + B (server side).

Phase A: focused matrix over the dedicated ``POST /session/:id/seed`` route
(NOT a grammar action) as the RESEED-only path — a valid seed builds through
the route; an out-of-volume voxel refuses typed; a zero direction refuses
typed; the request carries no hidden defaults; reselect supersedes with
provenance intact; a fresh session carries no seed; voxel->patient math
matches the persisted transform on the case02 geometry constants.

Phase B: FIRST-BUILD — when no ACTIVE centerline exists (the real T2 case),
the same route builds the INITIAL centerline from the physician seed via the
governed ``build_centerline`` derivation (no STALE fan-out, initial-build
provenance recorded distinctly); explicit refusal coverage for the new branch
(out-of-volume, zero direction, ambiguous source volume, ambiguous existing
paths, derivation abstention on the plain-harness volume).  Synthetic
fixtures only — no real pixels are committed anywhere in this module.
"""

from __future__ import annotations

import json
import math
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

import pytest

from apps.viewer.build_identity import resolve_build_identity
from apps.viewer.engine import SeedRequest, ViewerEngine, ViewerOpenError
from apps.viewer.grammar import ACTIONS, RejectionCode
from apps.viewer.overlay import (
    REFUSAL_AMBIGUOUS,
    REFUSAL_UNAVAILABLE,
    CoronaryOverlayRefusalError,
)
from apps.viewer.server import ViewerServer
from apps.workstation.session_store import RestoredSession
from core.coordinates import (
    FrameSpec,
    VoxelCoordinate,
    apply_matrix,
    assert_round_trip,
    transform_for_frame,
)
from core.coronary import Seed, VesselnessParameters, world_point_of
from core.digest import canonical_json
from core.enums import ActorType, ArtifactKind
from tests.apps.test_viewer_e2e import _deterministic_ulids
from tests.apps.viewer_pipeline import build_viewer_session
from tests.core.coronary.support import helix_points, vessel_seed
from tests.core.coronary.viewer_support import (
    CoronaryViewerFixture,
    build_coronary_viewer_fixture,
    with_ambiguous_centerlines,
)

_AT = "2026-09-17T00:00:00Z"

#: Case02 geometry constants (evaluation report T1_CASE02_EVALUATION_20260917:
#: 256 diagnostic axial locations at 512x512, 0.625 mm slice spacing, HU
#: rescale).  Orientation/origin/in-plane spacing below are synthetic axial
#: representatives — geometry constants only, no PHI and no pixels.
CASE02_SLICES = 256
CASE02_ROWS = 512
CASE02_COLUMNS = 512
CASE02_DIMENSIONS_KJI = (CASE02_SLICES, CASE02_ROWS, CASE02_COLUMNS)
CASE02_SPACING_MM = (0.5, 0.5, 0.625)
CASE02_ORIENTATION = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
CASE02_ORIGIN = (0.0, 0.0, 0.0)
CASE02_FRAME_OF_REFERENCE_UID = "1.2.826.0.1.3680043.8.498.77.940"
CASE02_SERIES_INSTANCE_UID = "1.2.826.0.1.3680043.8.498.77.941"


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
        tmp_path_factory.mktemp("t2-seed-viewer"),
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
    return _EngineCase(engine=engine, fixture=coronary_viewer)


def _select_rca(engine: ViewerEngine, nonce: str) -> None:
    """Select the RCA path first (the multi-vessel fixture needs one)."""

    state = engine.state_document()
    accepted = cast(list[Any], state["accepted_nonces"])
    parent = str(accepted[-1]) if accepted else None
    body = canonical_json(
        {
            "action": "SELECT_VESSEL",
            "params": {"vessel_label": "VESSEL:RCA"},
            "client_nonce": nonce,
            "view": {"plane": "AXIAL", "frame_index": 0},
            "lineage": {
                "session_id": engine.session_id(),
                "parent_action_nonce": parent,
            },
        }
    ).encode("utf-8")
    engine.apply_action_bytes(body, path_session_id=engine.session_id())


def _seed_body(
    *,
    session_id: str,
    voxel: list[int] | None,
    direction: list[float] | None,
    nonce: str,
    parent: str | None,
) -> bytes:
    envelope: dict[str, object] = {
        "client_nonce": nonce,
        "lineage": {"parent_action_nonce": parent, "session_id": session_id},
    }
    # Missing keys are omitted (never defaulted) so the no-default refusal
    # path can be exercised explicitly.
    if voxel is not None:
        envelope["voxel"] = voxel
    if direction is not None:
        envelope["direction"] = direction
    return canonical_json(envelope).encode("utf-8")


def _post_seed(
    port: int,
    session_id: str,
    body: bytes,
) -> tuple[int, dict[str, object]]:
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/session/{session_id}/seed",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Origin": f"http://127.0.0.1:{port}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def _phantom_seed(index: int, ahead: int, behind: int) -> tuple[list[int], list[float]]:
    """One vessel-evidenced seed on the analytic helix (deterministic)."""

    helix = helix_points()
    probe = helix[index]
    voxel = [
        int(round(float(probe[0]))),
        int(round(float(probe[1]))),
        int(round(float(probe[2]))),
    ]
    direction = [
        float(helix[ahead][0] - helix[behind][0]),
        float(helix[ahead][1] - helix[behind][1]),
        float(helix[ahead][2] - helix[behind][2]),
    ]
    return voxel, direction


def _unit(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(component * component for component in vector))
    assert norm > 0.0
    return [component / norm for component in vector]


def _error_code(payload: dict[str, object]) -> str:
    error = cast(dict[str, object], payload["error"])
    return str(error["code"])


# ---------------------------------------------------------------------------
# Route shape: dedicated, not a grammar action
# ---------------------------------------------------------------------------


def test_seed_route_is_dedicated_not_a_grammar_action() -> None:
    """The seed path adds no grammar action (the closed 11 are untouched)."""

    assert tuple(ACTIONS) == (
        "SET_SLICE",
        "SET_PLANE",
        "SET_WL_PRESET",
        "SET_WL_CUSTOM",
        "CLEAR_WL",
        "SET_VIEWPORT_MATRIX",
        "OPEN_QC_PANEL",
        "REQUEST_CORRECTION",
        "SET_OVERLAY_VISIBILITY",
        "REQUEST_CENTERLINE_DECISION",
        "SELECT_VESSEL",
    )
    assert "SEED" not in ACTIONS


def test_seed_route_binds_unknown_sessions() -> None:
    """The dedicated route exists and resolves sessions before parsing."""

    server = ViewerServer()
    _host, port = server.start_background()
    try:
        status, payload = _post_seed(
            port,
            "01HZZZZZZZZZZZZZZZZZZZZZ99",
            _seed_body(
                session_id="01HZZZZZZZZZZZZZZZZZZZZZ99",
                voxel=[1, 1, 1],
                direction=[1.0, 0.0, 0.0],
                nonce="t2-seed-unknown-0001",
                parent=None,
            ),
        )
    finally:
        server.stop()
    assert status == 404
    assert _error_code(payload) == "SESSION_NOT_FOUND"


# ---------------------------------------------------------------------------
# No hidden defaults: a fresh session carries no seed
# ---------------------------------------------------------------------------


def test_fresh_session_has_no_seed(
    tmp_path: Path, checkout_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fresh viewer sessions (no centerline) report no seed, verbatim."""

    monkeypatch.chdir(checkout_root)
    fixture = build_viewer_session(tmp_path / "t2-seed-fresh-ws")
    engine = ViewerEngine(
        fixture.session,
        id_factory=_deterministic_ulids(),
        now=lambda: _AT,
        repo_root=checkout_root,
    )
    engine.open()
    assert engine.current_seed() is None
    with pytest.raises(CoronaryOverlayRefusalError) as excinfo:
        engine.coronary_overlay_document()
    assert excinfo.value.code == REFUSAL_UNAVAILABLE
    state = engine.state_document()
    assert state["accepted_nonces"] == []
    assert state["action_count"] == 0


def test_seed_request_has_no_hidden_defaults(case: _EngineCase) -> None:
    """Omitted voxel/direction are refused, never defaulted in."""

    engine = case.engine
    server = ViewerServer()
    server.add_engine(engine)
    _host, port = server.start_background()
    try:
        _select_rca(engine, "t2-seed-nodefault-select-01")
        parent = "t2-seed-nodefault-select-01"
        voxel, direction = _phantom_seed(80, 100, 60)
        for nonce, bad_voxel, bad_direction in (
            ("t2-seed-nodefault-000001", None, direction),
            ("t2-seed-nodefault-000002", voxel, None),
        ):
            status, payload = _post_seed(
                port,
                engine.session_id(),
                _seed_body(
                    session_id=engine.session_id(),
                    voxel=bad_voxel,
                    direction=bad_direction,
                    nonce=nonce,
                    parent=parent,
                ),
            )
            assert status == 400
            assert _error_code(payload) == "UNKNOWN_PARAM"
        # The refused attempts changed nothing but their consumed nonces.
        assert engine.current_seed() is None or (
            engine.current_seed() or {}).get("centerline_ref") == (
            f"centerline_artifact_id:{case.fixture.centerline_id}"
        )
    finally:
        server.stop()


# ---------------------------------------------------------------------------
# Valid build through the route
# ---------------------------------------------------------------------------


def test_valid_seed_builds_through_the_route(case: _EngineCase) -> None:
    """A physician seed POST builds a NEW centerline through the reseed seam."""

    engine = case.engine
    server = ViewerServer()
    server.add_engine(engine)
    _host, port = server.start_background()
    try:
        _select_rca(engine, "t2-seed-valid-select-0001")
        voxel, direction = _phantom_seed(80, 100, 60)
        status, state = _post_seed(
            port,
            engine.session_id(),
            _seed_body(
                session_id=engine.session_id(),
                voxel=voxel,
                direction=direction,
                nonce="t2-seed-valid-00000001",
                parent="t2-seed-valid-select-0001",
            ),
        )
        assert status == 200
        assert state["action_count"] == 2
        assert cast(list[Any], state["accepted_nonces"])[-1] == "t2-seed-valid-00000001"
        seed = engine.current_seed()
        assert seed is not None
        assert seed["voxel"] == voxel
        assert seed["direction"] == pytest.approx(_unit(direction))
        assert seed["centerline_ref"] != (
            f"centerline_artifact_id:{case.fixture.centerline_id}"
        )
        overlay = engine.coronary_overlay_document()
        assert overlay["centerline_ref"] == seed["centerline_ref"]
        assert overlay["review_state"] == "PREDICTED"
        assert cast(int, overlay["point_count"]) > 0
    finally:
        server.stop()


# ---------------------------------------------------------------------------
# Typed refusals: out-of-volume voxel, zero direction
# ---------------------------------------------------------------------------


def test_out_of_volume_seed_refuses_typed(case: _EngineCase) -> None:
    """A voxel on the exclusive upper bound refuses SEED_OUT_OF_BOUNDS."""

    engine = case.engine
    server = ViewerServer()
    server.add_engine(engine)
    _host, port = server.start_background()
    try:
        _select_rca(engine, "t2-seed-oob-select-00001")
        _voxel, direction = _phantom_seed(80, 100, 60)
        # The phantom volume is 48^3: i == 48 is exactly out of volume.
        status, payload = _post_seed(
            port,
            engine.session_id(),
            _seed_body(
                session_id=engine.session_id(),
                voxel=[48, 24, 24],
                direction=direction,
                nonce="t2-seed-oob-0000000001",
                parent="t2-seed-oob-select-00001",
            ),
        )
        assert status == 409
        assert _error_code(payload) == "SEED_OUT_OF_BOUNDS"
        # Fail-closed: the ACTIVE path is untouched, the nonce is consumed.
        assert engine.coronary_overlay_document()["centerline_ref"] == (
            f"centerline_artifact_id:{case.fixture.centerline_id}"
        )
        status, payload = _post_seed(
            port,
            engine.session_id(),
            _seed_body(
                session_id=engine.session_id(),
                voxel=[48, 24, 24],
                direction=direction,
                nonce="t2-seed-oob-0000000001",
                parent="t2-seed-oob-select-00001",
            ),
        )
        assert status == 400
        assert _error_code(payload) == "NONCE_REUSE"
    finally:
        server.stop()


def test_zero_direction_seed_refuses_typed(case: _EngineCase) -> None:
    """A zero direction hint refuses SEED_DIRECTION_INVALID."""

    engine = case.engine
    server = ViewerServer()
    server.add_engine(engine)
    _host, port = server.start_background()
    try:
        _select_rca(engine, "t2-seed-zero-select-0001")
        voxel, _direction = _phantom_seed(80, 100, 60)
        status, payload = _post_seed(
            port,
            engine.session_id(),
            _seed_body(
                session_id=engine.session_id(),
                voxel=voxel,
                direction=[0.0, 0.0, 0.0],
                nonce="t2-seed-zero-0000000001",
                parent="t2-seed-zero-select-0001",
            ),
        )
        assert status == 409
        assert _error_code(payload) == "SEED_DIRECTION_INVALID"
        assert engine.coronary_overlay_document()["centerline_ref"] == (
            f"centerline_artifact_id:{case.fixture.centerline_id}"
        )
    finally:
        server.stop()


# ---------------------------------------------------------------------------
# Reselect supersedes with provenance intact
# ---------------------------------------------------------------------------


def test_reselect_supersedes_with_provenance_intact(
    coronary_viewer: CoronaryViewerFixture, checkout_root: Path
) -> None:
    """Two physician seeds chain old -> A -> B; HUMAN provenance is intact."""

    recorded: list[dict[str, object]] = []

    class _Recorder:
        def record(self, **kwargs: object) -> None:
            recorded.append(dict(kwargs))

    engine = ViewerEngine(
        coronary_viewer.session,
        id_factory=_deterministic_ulids(),
        now=lambda: _AT,
        repo_root=checkout_root,
        audit_sink=_Recorder(),
    )
    engine.open()
    old_id = coronary_viewer.centerline_id
    server = ViewerServer()
    server.add_engine(engine)
    _host, port = server.start_background()
    try:
        _select_rca(engine, "t2-seed-reselect-select01")
        voxel_a, direction_a = _phantom_seed(80, 100, 60)
        status, _state = _post_seed(
            port,
            engine.session_id(),
            _seed_body(
                session_id=engine.session_id(),
                voxel=voxel_a,
                direction=direction_a,
                nonce="t2-seed-reselect-000001",
                parent="t2-seed-reselect-select01",
            ),
        )
        assert status == 200
        seed_a = engine.current_seed()
        assert seed_a is not None
        id_a = str(seed_a["centerline_ref"]).split(":", 1)[1]
        assert id_a != old_id
        voxel_b, direction_b = _phantom_seed(40, 60, 20)
        status, state = _post_seed(
            port,
            engine.session_id(),
            _seed_body(
                session_id=engine.session_id(),
                voxel=voxel_b,
                direction=direction_b,
                nonce="t2-seed-reselect-000002",
                parent="t2-seed-reselect-000001",
            ),
        )
        assert status == 200
        assert state["action_count"] == 3
        seed_b = engine.current_seed()
        assert seed_b is not None
        id_b = str(seed_b["centerline_ref"]).split(":", 1)[1]
        assert id_b not in (old_id, id_a)
        assert seed_b["voxel"] == voxel_b
        assert seed_b["direction"] == pytest.approx(_unit(direction_b))
        engine.persist(coronary_viewer.store)
    finally:
        server.stop()

    reopened = coronary_viewer.store.reopen()
    by_id = {artifact.artifact_id: artifact for artifact in reopened.artifacts}
    assert by_id[id_b].kind is ArtifactKind.CENTERLINE
    assert by_id[id_b].status.value == "ACTIVE"
    assert by_id[id_a].status.value == "SUPERSEDED"
    assert by_id[old_id].status.value == "SUPERSEDED"
    corrections = [
        artifact
        for artifact in reopened.artifacts
        if artifact.kind is ArtifactKind.CORRECTION
    ]
    assert len(corrections) == 2
    by_predecessor = {
        artifact.lineage.input_artifact_refs[0]: artifact for artifact in corrections
    }
    assert set(by_predecessor) == {old_id, id_a}
    for predecessor, correction in by_predecessor.items():
        assert correction.producer.producer_type.value == "HUMAN"
        assert correction.review.supersedes_ref == (
            f"centerline_artifact_id:{predecessor}"
        )
    # The sealed correction payloads carry the physician seeds verbatim,
    # marked by the physician-selected seed reason (never a silent default).
    documents = reopened.payload_documents
    payload_a = cast(dict[str, Any], documents[by_predecessor[old_id].artifact_id])
    assert payload_a["decision"] == "EDIT"
    assert (
        payload_a["reason"]
        == "viewer seed (physician-selected seed -> NEW centerline)"
    )
    assert payload_a["seed_voxel"] == voxel_a
    assert payload_a["seed_direction_world"] == pytest.approx(_unit(direction_a))
    payload_b = cast(dict[str, Any], documents[by_predecessor[id_a].artifact_id])
    assert payload_b["decision"] == "EDIT"
    assert (
        payload_b["reason"]
        == "viewer seed (physician-selected seed -> NEW centerline)"
    )
    assert payload_b["seed_voxel"] == voxel_b
    assert payload_b["seed_direction_world"] == pytest.approx(_unit(direction_b))
    # Session/nonce lineage: the accepted chain is select -> A -> B, and the
    # HUMAN correction provenance plus the seed evidence refs bind the nonces.
    assert engine.state_document()["accepted_nonces"] == [
        "t2-seed-reselect-select01",
        "t2-seed-reselect-000001",
        "t2-seed-reselect-000002",
    ]
    provenance = list(engine.new_provenance())
    correction_records = [
        record
        for record in provenance
        if record.subject_ref.startswith("correction_artifact_id:")
    ]
    assert len(correction_records) == 2
    for record in correction_records:
        assert record.actor.actor_type is ActorType.HUMAN
        assert record.human_edit is not None
    centerline_records = [
        record
        for record in provenance
        if record.subject_ref
        in (f"centerline_artifact_id:{id_a}", f"centerline_artifact_id:{id_b}")
    ]
    assert len(centerline_records) == 2
    for record in centerline_records:
        assert record.actor.actor_type is ActorType.ENGINE
        assert isinstance(record.inputs, tuple)
        seed_inputs = [entry for entry in record.inputs if entry.kind == "seed"]
        assert len(seed_inputs) == 1
    # Session/nonce lineage through the governed audit channel: one sealed
    # event per accepted seed, bound to the session with the parent chain.
    accepted = [
        event
        for event in recorded
        if event.get("event_type") == "VIEWER_ACTION_ACCEPTED"
        and event.get("to_value") == "SEED"
    ]
    assert [event["entity_id"] for event in accepted] == [
        "t2-seed-reselect-000001",
        "t2-seed-reselect-000002",
    ]
    assert all(event["evidence_ref"] == engine.session_id() for event in accepted)
    seed_payloads = [cast(dict[str, Any], event["payload"]) for event in accepted]
    assert seed_payloads[0]["parent_action_nonce"] == "t2-seed-reselect-select01"
    assert seed_payloads[0]["voxel"] == voxel_a
    assert seed_payloads[1]["parent_action_nonce"] == "t2-seed-reselect-000001"
    assert seed_payloads[1]["voxel"] == voxel_b
    # The successor generation serves on the overlay immediately.
    engine2 = ViewerEngine(
        reopened,
        id_factory=_deterministic_ulids(),
        now=lambda: _AT,
        repo_root=checkout_root,
    )
    engine2.open()
    assert engine2.coronary_overlay_document()["centerline_ref"] == (
        f"centerline_artifact_id:{id_b}"
    )


# ---------------------------------------------------------------------------
# Case02 geometry constants: voxel->patient math through the persisted affine
# ---------------------------------------------------------------------------


def _case02_frame() -> FrameSpec:
    return FrameSpec(
        frame_of_reference_uid=CASE02_FRAME_OF_REFERENCE_UID,
        image_position_patient=CASE02_ORIGIN,
        image_orientation_patient=CASE02_ORIENTATION,
        spacing_mm=CASE02_SPACING_MM,
        dimensions=CASE02_DIMENSIONS_KJI,
    )


def test_case02_voxel_to_patient_matches_the_persisted_transform() -> None:
    """Voxel->patient resolution on the case02 grid uses the admitted affine.

    Dimensions (256 slices x 512 rows x 512 columns) and the 0.625 mm slice
    spacing come from the T1 case02 evaluation; the conversion itself is the
    persisted WORLD<->VOXEL linear part (the same conversion ``Seed``
    documents), checked here against the hand-computed axial affine.
    """

    frame = _case02_frame()
    assert frame.dimensions == (256, 512, 512)
    transform = transform_for_frame(
        frame, series_instance_uid=CASE02_SERIES_INSTANCE_UID
    )
    assert_round_trip(transform, transform.inverse)
    matrix = frame.voxel_to_world_matrix()
    assert matrix == transform.matrix
    sx, sy, sz = CASE02_SPACING_MM
    ox, oy, oz = CASE02_ORIGIN
    for voxel in ((0, 0, 0), (511, 511, 255), (256, 128, 64), (1, 0, 0)):
        i, j, k = voxel
        expected = (ox + i * sx, oy + j * sy, oz + k * sz)
        assert world_point_of(matrix, voxel) == pytest.approx(expected)
        assert apply_matrix(matrix, VoxelCoordinate(i=i, j=j, k=k)) == pytest.approx(
            expected
        )
    # The explicit server-side bounds rule on the case02 grid: 0 <= i <
    # columns, 0 <= j < rows, 0 <= k < slices.
    columns, rows, slices = CASE02_COLUMNS, CASE02_ROWS, CASE02_SLICES

    def in_volume(voxel: tuple[int, int, int]) -> bool:
        i, j, k = voxel
        return 0 <= i < columns and 0 <= j < rows and 0 <= k < slices

    assert in_volume((0, 0, 0))
    assert in_volume((511, 511, 255))
    assert not in_volume((512, 0, 0))
    assert not in_volume((0, 512, 0))
    assert not in_volume((0, 0, 256))
    assert not in_volume((-1, 0, 0))
    # In-bounds case02 voxels are well-formed Seeds; the direction stored in
    # provenance is the unit direction.
    seed = Seed(i=511, j=511, k=255, direction=(0.0, 0.0, 2.0))
    assert seed.voxel == (511, 511, 255)
    assert seed.unit_direction() == pytest.approx((0.0, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Phase B: FIRST-BUILD — the seed route on a session with no ACTIVE centerline
# ---------------------------------------------------------------------------

_ULID_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def _offset_ulids(start: int):
    """Deterministic ULIDs from a non-zero counter start (id-space isolation).

    Same construction as ``tests.apps.test_viewer_e2e._deterministic_ulids``
    with a disjoint counter range, so Phase B persist/reopen assertions can
    target their own artifacts on the shared module store without colliding
    with the Phase A id sequence.
    """

    class _Factory:
        def __init__(self) -> None:
            self._n = start

        def __call__(self, at_ms: int | None = None) -> str:
            del at_ms
            self._n += 1
            value = (0x01A9 << 80) | self._n
            return "".join(
                _ULID_ALPHABET[(value >> (shift * 5)) & 0x1F] for shift in range(25, -1, -1)
            )

    return _Factory()


def _strip_to_volume_only(session: RestoredSession) -> RestoredSession:
    """The session without any centerline-lineage artifacts (the T2 case).

    Drops every CENTERLINE and CORRECTION artifact plus every framed CPR view
    (bound to a removed centerline generation); volumes, transforms, and the
    MPR frame geometry stay untouched.  Provenance history is retained
    (append-only); candidacy is what changes, so the overlay refuses
    UNAVAILABLE and the seed route takes the first-build branch.
    """

    drop: set[str] = set()
    for artifact in session.artifacts:
        if artifact.kind is ArtifactKind.CENTERLINE or artifact.kind is ArtifactKind.CORRECTION:
            drop.add(artifact.artifact_id)
        elif artifact.kind is ArtifactKind.MPR_VIEW:
            document = session.payload_documents.get(artifact.artifact_id)
            if isinstance(document, dict) and document.get("schema_version") == (
                "ccta.cpr.view.v1"
            ):
                drop.add(artifact.artifact_id)
    return replace(
        session,
        artifacts=tuple(
            artifact for artifact in session.artifacts if artifact.artifact_id not in drop
        ),
        payload_documents={
            key: value for key, value in session.payload_documents.items() if key not in drop
        },
    )


def _volume_only_engine(
    session: RestoredSession,
    checkout_root: Path,
    *,
    ulid_start: int,
    sink: Any | None = None,
) -> ViewerEngine:
    engine = ViewerEngine(
        session,
        id_factory=_offset_ulids(ulid_start),
        now=lambda: _AT,
        repo_root=checkout_root,
        audit_sink=sink,
    )
    engine.open()
    return engine


def test_first_build_creates_initial_centerline(
    coronary_viewer: CoronaryViewerFixture, checkout_root: Path
) -> None:
    """A seed on a centerline-free session builds the INITIAL centerline."""

    recorded: list[dict[str, object]] = []

    class _Recorder:
        def record(self, **kwargs: object) -> None:
            recorded.append(dict(kwargs))

    session = _strip_to_volume_only(coronary_viewer.session)
    engine = _volume_only_engine(session, checkout_root, ulid_start=700000, sink=_Recorder())
    assert engine.current_seed() is None
    with pytest.raises(CoronaryOverlayRefusalError) as excinfo:
        engine.coronary_overlay_document()
    assert excinfo.value.code == REFUSAL_UNAVAILABLE
    server = ViewerServer()
    server.add_engine(engine)
    _host, port = server.start_background()
    try:
        voxel, direction = _phantom_seed(40, 60, 20)
        status, state = _post_seed(
            port,
            engine.session_id(),
            _seed_body(
                session_id=engine.session_id(),
                voxel=voxel,
                direction=direction,
                nonce="t2-fb-first-0000000001",
                parent=None,
            ),
        )
    finally:
        server.stop()
    assert status == 200
    # The first-build seed is the FIRST accepted action (parent None chains).
    assert state["action_count"] == 1
    assert state["accepted_nonces"] == ["t2-fb-first-0000000001"]
    seed = engine.current_seed()
    assert seed is not None
    assert seed["voxel"] == voxel
    assert seed["direction"] == pytest.approx(_unit(direction))
    new_id = str(seed["centerline_ref"]).split(":", 1)[1]
    overlay = engine.coronary_overlay_document()
    assert overlay["centerline_ref"] == seed["centerline_ref"]
    assert overlay["review_state"] == "PREDICTED"
    assert cast(int, overlay["point_count"]) > 0
    # Initial-build provenance is distinctly marked (not the reseed runtime),
    # with the physician seed bound as a typed derivation input.
    centerline_records = [
        record
        for record in engine.new_provenance()
        if record.subject_ref == f"centerline_artifact_id:{new_id}"
    ]
    assert len(centerline_records) == 1
    assert centerline_records[0].environment.runtime is not None
    assert "initial build" in centerline_records[0].environment.runtime
    assert centerline_records[0].actor.actor_type is ActorType.ENGINE
    assert isinstance(centerline_records[0].inputs, tuple)
    seed_inputs = [entry for entry in centerline_records[0].inputs if entry.kind == "seed"]
    assert len(seed_inputs) == 1
    # No CORRECTION artifact exists for an initial build (nothing superseded).
    assert [
        record
        for record in engine.new_provenance()
        if record.subject_ref.startswith("correction_artifact_id:")
    ] == []
    engine.persist(coronary_viewer.store)
    reopened = coronary_viewer.store.reopen()
    by_id = {artifact.artifact_id: artifact for artifact in reopened.artifacts}
    assert by_id[new_id].kind is ArtifactKind.CENTERLINE
    assert by_id[new_id].status.value == "ACTIVE"
    payload = cast(dict[str, Any], reopened.payload_documents[new_id])
    assert payload["schema_version"] == "ccta.payload.coronary_centerline.v1"
    assert payload["seed_voxel"] == voxel
    assert payload["seed_direction_world"] == pytest.approx(_unit(direction))
    # The frozen default parameters ran (recorded by digest, never silent);
    # the study ref is the viewer session's own study.
    assert payload["parameters_sha256"] == VesselnessParameters().sha256
    assert payload["vessel_label"] is None
    assert payload["study_instance_uid"] == session.study.dicom_study_instance_uid
    assert [
        artifact
        for artifact in reopened.artifacts
        if artifact.kind is ArtifactKind.CORRECTION
        and artifact.lineage.input_artifact_refs == (new_id,)
    ] == []
    accepted = [
        event
        for event in recorded
        if event.get("event_type") == "VIEWER_ACTION_ACCEPTED" and event.get("to_value") == "SEED"
    ]
    assert [event["entity_id"] for event in accepted] == ["t2-fb-first-0000000001"]
    assert cast(dict[str, Any], accepted[0]["payload"])["voxel"] == voxel


def test_first_build_then_reseed_supersedes(
    coronary_viewer: CoronaryViewerFixture, checkout_root: Path
) -> None:
    """The unmodified EDIT path supersedes a first-built centerline."""

    session = _strip_to_volume_only(coronary_viewer.session)
    engine = _volume_only_engine(session, checkout_root, ulid_start=710000)
    server = ViewerServer()
    server.add_engine(engine)
    _host, port = server.start_background()
    try:
        voxel_a, direction_a = _phantom_seed(40, 60, 20)
        status, _state = _post_seed(
            port,
            engine.session_id(),
            _seed_body(
                session_id=engine.session_id(),
                voxel=voxel_a,
                direction=direction_a,
                nonce="t2-fb-chain-0000000001",
                parent=None,
            ),
        )
        assert status == 200
        seed_a = engine.current_seed()
        assert seed_a is not None
        id_a = str(seed_a["centerline_ref"]).split(":", 1)[1]
        voxel_b, direction_b = _phantom_seed(200, 220, 180)
        status, state = _post_seed(
            port,
            engine.session_id(),
            _seed_body(
                session_id=engine.session_id(),
                voxel=voxel_b,
                direction=direction_b,
                nonce="t2-fb-chain-0000000002",
                parent="t2-fb-chain-0000000001",
            ),
        )
        assert status == 200
        assert state["action_count"] == 2
        seed_b = engine.current_seed()
        assert seed_b is not None
        id_b = str(seed_b["centerline_ref"]).split(":", 1)[1]
        assert id_b != id_a
        engine.persist(coronary_viewer.store)
    finally:
        server.stop()
    reopened = coronary_viewer.store.reopen()
    by_id = {artifact.artifact_id: artifact for artifact in reopened.artifacts}
    assert by_id[id_b].status.value == "ACTIVE"
    assert by_id[id_a].status.value == "SUPERSEDED"
    corrections = [
        artifact
        for artifact in reopened.artifacts
        if artifact.kind is ArtifactKind.CORRECTION
        and artifact.lineage.input_artifact_refs == (id_a,)
    ]
    assert len(corrections) == 1
    assert corrections[0].producer.producer_type.value == "HUMAN"
    payload_b = cast(dict[str, Any], reopened.payload_documents[corrections[0].artifact_id])
    assert payload_b["decision"] == "EDIT"
    assert payload_b["seed_voxel"] == voxel_b
    engine2 = ViewerEngine(
        reopened,
        id_factory=_deterministic_ulids(),
        now=lambda: _AT,
        repo_root=checkout_root,
    )
    engine2.open()
    assert engine2.coronary_overlay_document()["centerline_ref"] == (
        f"centerline_artifact_id:{id_b}"
    )


def test_first_build_carries_selected_vessel_label(
    coronary_viewer: CoronaryViewerFixture, checkout_root: Path
) -> None:
    """A vessel-selected first build labels the INITIAL centerline."""

    session = _strip_to_volume_only(coronary_viewer.session)
    engine = _volume_only_engine(session, checkout_root, ulid_start=720000)
    server = ViewerServer()
    server.add_engine(engine)
    _host, port = server.start_background()
    try:
        _select_rca(engine, "t2-fb-vessel-select-0001")
        probe = vessel_seed("VESSEL:RCA")
        status, _state = _post_seed(
            port,
            engine.session_id(),
            _seed_body(
                session_id=engine.session_id(),
                voxel=[probe.i, probe.j, probe.k],
                direction=list(probe.direction),
                nonce="t2-fb-vessel-000000001",
                parent="t2-fb-vessel-select-0001",
            ),
        )
        assert status == 200
        seed = engine.current_seed()
        assert seed is not None
        new_id = str(seed["centerline_ref"]).split(":", 1)[1]
        engine.persist(coronary_viewer.store)
    finally:
        server.stop()
    reopened = coronary_viewer.store.reopen()
    payload = cast(dict[str, Any], reopened.payload_documents[new_id])
    assert payload["vessel_label"] == "VESSEL:RCA"
    assert (
        engine.coronary_overlay_document(vessel_label="VESSEL:RCA")["centerline_ref"]
        == f"centerline_artifact_id:{new_id}"
    )


def test_first_build_out_of_volume_refuses_typed(
    coronary_viewer: CoronaryViewerFixture, checkout_root: Path
) -> None:
    """A first-build voxel outside every session volume refuses typed."""

    session = _strip_to_volume_only(coronary_viewer.session)
    engine = _volume_only_engine(session, checkout_root, ulid_start=730000)
    server = ViewerServer()
    server.add_engine(engine)
    _host, port = server.start_background()
    try:
        _voxel, direction = _phantom_seed(40, 60, 20)
        # i == 48 is the exclusive upper bound of the 48^3 phantom volume
        # (and far outside the tiny DICOM volume): contained in none.
        status, payload = _post_seed(
            port,
            engine.session_id(),
            _seed_body(
                session_id=engine.session_id(),
                voxel=[48, 24, 24],
                direction=direction,
                nonce="t2-fb-oob-000000000001",
                parent=None,
            ),
        )
        assert status == 409
        assert _error_code(payload) == "SEED_OUT_OF_BOUNDS"
        assert engine.current_seed() is None
        with pytest.raises(CoronaryOverlayRefusalError):
            engine.coronary_overlay_document()
        status, payload = _post_seed(
            port,
            engine.session_id(),
            _seed_body(
                session_id=engine.session_id(),
                voxel=[48, 24, 24],
                direction=direction,
                nonce="t2-fb-oob-000000000001",
                parent=None,
            ),
        )
        assert status == 400
        assert _error_code(payload) == "NONCE_REUSE"
    finally:
        server.stop()


def test_first_build_zero_direction_refuses_typed(
    coronary_viewer: CoronaryViewerFixture, checkout_root: Path
) -> None:
    """A first-build zero direction hint refuses typed, minting nothing."""

    session = _strip_to_volume_only(coronary_viewer.session)
    engine = _volume_only_engine(session, checkout_root, ulid_start=740000)
    server = ViewerServer()
    server.add_engine(engine)
    _host, port = server.start_background()
    try:
        voxel, _direction = _phantom_seed(40, 60, 20)
        status, payload = _post_seed(
            port,
            engine.session_id(),
            _seed_body(
                session_id=engine.session_id(),
                voxel=voxel,
                direction=[0.0, 0.0, 0.0],
                nonce="t2-fb-zero-00000000001",
                parent=None,
            ),
        )
        assert status == 409
        assert _error_code(payload) == "SEED_DIRECTION_INVALID"
        assert engine.current_seed() is None
        assert engine.state_document()["accepted_nonces"] == []
    finally:
        server.stop()


def test_first_build_ambiguous_volume_refuses(
    coronary_viewer: CoronaryViewerFixture, checkout_root: Path
) -> None:
    """A seed contained in several volumes refuses instead of picking one."""

    session = _strip_to_volume_only(coronary_viewer.session)
    engine = _volume_only_engine(session, checkout_root, ulid_start=750000)
    server = ViewerServer()
    server.add_engine(engine)
    _host, port = server.start_background()
    try:
        # (1, 1, 1) sits inside BOTH the tiny DICOM volume and the phantom:
        # the first-build route must not silently choose between them.
        status, payload = _post_seed(
            port,
            engine.session_id(),
            _seed_body(
                session_id=engine.session_id(),
                voxel=[1, 1, 1],
                direction=[1.0, 0.0, 0.0],
                nonce="t2-fb-ambvol-0000000001",
                parent=None,
            ),
        )
        assert status == 409
        assert _error_code(payload) == "CENTERLINE_DECISION_REFUSED"
        assert "2 session volumes" in str(payload["error"])
        assert engine.current_seed() is None
        with pytest.raises(CoronaryOverlayRefusalError) as excinfo:
            engine.coronary_overlay_document()
        assert excinfo.value.code == REFUSAL_UNAVAILABLE
    finally:
        server.stop()


def test_first_build_abstention_refuses_without_fabrication(
    coronary_viewer: CoronaryViewerFixture, checkout_root: Path
) -> None:
    """A seed the derivation cannot trace abstains typed (no fallback path)."""

    session = _strip_to_volume_only(coronary_viewer.session)
    engine = _volume_only_engine(session, checkout_root, ulid_start=760000)
    server = ViewerServer()
    server.add_engine(engine)
    _host, port = server.start_background()
    try:
        # The helix[80] seed carries vessel evidence but does not resolve to
        # a traceable path under the frozen default parameters: the
        # derivation abstains and the route surfaces the refusal verbatim.
        voxel, direction = _phantom_seed(80, 100, 60)
        status, payload = _post_seed(
            port,
            engine.session_id(),
            _seed_body(
                session_id=engine.session_id(),
                voxel=voxel,
                direction=direction,
                nonce="t2-fb-abstain-000000001",
                parent=None,
            ),
        )
        assert status == 409
        assert _error_code(payload) == "CENTERLINE_DECISION_REFUSED"
        assert "NOT_ANALYZABLE" in str(payload["error"])
        assert engine.current_seed() is None
        assert engine.state_document()["accepted_nonces"] == []
    finally:
        server.stop()


def test_seed_over_ambiguous_centerlines_refuses_without_building(
    coronary_viewer: CoronaryViewerFixture, checkout_root: Path
) -> None:
    """First-build never fires over an ambiguous existing path set."""

    engine = ViewerEngine(
        with_ambiguous_centerlines(coronary_viewer.session),
        id_factory=_offset_ulids(770000),
        now=lambda: _AT,
        repo_root=checkout_root,
    )
    engine.open()
    server = ViewerServer()
    server.add_engine(engine)
    _host, port = server.start_background()
    try:
        voxel, direction = _phantom_seed(40, 60, 20)
        status, payload = _post_seed(
            port,
            engine.session_id(),
            _seed_body(
                session_id=engine.session_id(),
                voxel=voxel,
                direction=direction,
                nonce="t2-fb-ambpath-000000001",
                parent=None,
            ),
        )
        assert status == 409
        assert _error_code(payload) == REFUSAL_AMBIGUOUS
        assert engine.state_document()["accepted_nonces"] == []
        with pytest.raises(CoronaryOverlayRefusalError) as excinfo:
            engine.coronary_overlay_document()
        assert excinfo.value.code == REFUSAL_AMBIGUOUS
    finally:
        server.stop()


def test_first_build_on_plain_viewer_volume_refuses_without_fabrication(
    tmp_path: Path, checkout_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The plain-harness volume (no vessel evidence) abstains typed.

    ``build_viewer_session`` carries a single tiny synthetic volume with no
    coronary content: the first-build derivation finds no vessel evidence at
    the seed and refuses instead of fabricating a path.
    """

    monkeypatch.chdir(checkout_root)
    fixture = build_viewer_session(tmp_path / "t2-fb-plain-ws")
    engine = ViewerEngine(
        fixture.session,
        id_factory=_offset_ulids(780000),
        now=lambda: _AT,
        repo_root=checkout_root,
    )
    engine.open()
    server = ViewerServer()
    server.add_engine(engine)
    _host, port = server.start_background()
    try:
        status, payload = _post_seed(
            port,
            engine.session_id(),
            _seed_body(
                session_id=engine.session_id(),
                voxel=[1, 1, 1],
                direction=[1.0, 0.0, 0.0],
                nonce="t2-fb-plain-00000000001",
                parent=None,
            ),
        )
        assert status == 409
        assert _error_code(payload) == "CENTERLINE_DECISION_REFUSED"
        assert "NOT_ANALYZABLE" in str(payload["error"])
        assert engine.current_seed() is None
    finally:
        server.stop()

# ---------------------------------------------------------------------------
# F2_SEED_NONCE_IRREVERSIBILITY: ONE (session_id, client_nonce) APPLIES ONCE
# ---------------------------------------------------------------------------
#
# The re-seed acceptance path used to be the ONLY seed path that never
# re-checked the client nonce: validation (``_validate_seed_request``) and the
# first-build admission (``_admit_seed_first_build``) both refuse a seen
# nonce, but a nonce consumed by ANY other request in the window between
# validation and acceptance still reached the re-seed appends, so the same
# identity could be ACCEPTED TWICE (``used_nonces`` and ``accepted_nonces``
# gain the value twice, ``action_count`` advances twice) and any terminal
# record published for the identity could be overturned (R6-F1 / R6-F2).
#
# The window is created deterministically here through the REAL route: one
# seed is parked at the entry of ``_execute_seed`` (i.e. after validation,
# before anything is applied) while a second, malformed attempt carrying the
# same ``client_nonce`` consumes it fail-closed.  Synthetic phantom session
# only - no patient anatomy and no clinical claim.
#
# Failing against the UNCHANGED engine is the point of these tests: the
# parked seed is accepted a second time (HTTP 200, the nonce twice in the
# ledger, a new centerline applied).


def _nonce_ledger_counts(document: dict[str, object], nonce: str) -> tuple[int, int]:
    """``(used, accepted)`` occurrence counts of one nonce in the served ledger."""

    used = cast(list[Any], document["used_nonces"])
    accepted = cast(list[Any], document["accepted_nonces"])
    return used.count(nonce), accepted.count(nonce)


def _application_snapshot(engine: ViewerEngine) -> dict[str, object]:
    """Every value a second application of one nonce would mutate."""

    return {
        "state": engine.state_document(),
        "overlay": engine.coronary_overlay_document(),
        "seed": engine.current_seed(),
        "artifact_ids": sorted(artifact.artifact_id for artifact in engine._working_artifacts()),
        "payload_ids": sorted(engine._working_payload_documents()),
        "new_coronary_artifacts": sorted(engine._new_coronary_artifacts),
        "new_coronary_documents": sorted(engine._new_coronary_documents),
        "new_provenance": [record.provenance_id for record in engine._new_provenance],
    }


def _park_seed_and_consume_its_nonce(
    engine: ViewerEngine,
    monkeypatch: pytest.MonkeyPatch,
    *,
    port: int,
    nonce: str,
    parent: str,
    voxel: list[int],
    direction: list[float],
) -> dict[str, Any]:
    """Resume one validated seed whose nonce another request consumed.

    The seed is parked at the entry of ``_execute_seed`` - after
    ``_validate_seed_request`` (which requires an UNSEEN nonce) and before
    ANY application - then a malformed attempt carrying the same
    ``client_nonce`` consumes it fail-closed.  Returns the parked seed's real
    HTTP outcome and the consuming attempt's outcome.
    """

    entered = threading.Event()
    release = threading.Event()
    real_execute_seed = ViewerEngine._execute_seed

    def _paused_execute_seed(
        self: ViewerEngine, request: SeedRequest
    ) -> dict[str, object]:
        if request.client_nonce == nonce:
            entered.set()
            assert release.wait(timeout=60), "the parked seed was never released"
        return real_execute_seed(self, request)

    monkeypatch.setattr(ViewerEngine, "_execute_seed", _paused_execute_seed)
    outcome: dict[str, Any] = {}

    def _fire() -> None:
        outcome["seed"] = _post_seed(
            port,
            engine.session_id(),
            _seed_body(
                session_id=engine.session_id(),
                voxel=voxel,
                direction=direction,
                nonce=nonce,
                parent=parent,
            ),
        )

    worker = threading.Thread(target=_fire)
    worker.start()
    try:
        assert entered.wait(timeout=60), "the seed never reached its acceptance path"
        # A different request CONSUMES the same nonce without accepting it
        # (the pre-existing fail-closed accounting for a malformed attempt).
        consuming = canonical_json(
            {
                "client_nonce": nonce,
                "direction": direction,
                "lineage": {"parent_action_nonce": parent, "session_id": engine.session_id()},
                "unexpected_key": 1,
                "voxel": voxel,
            }
        ).encode("utf-8")
        outcome["consuming"] = _post_seed(port, engine.session_id(), consuming)
        # The window is open: the nonce is SEEN, never accepted.
        used, accepted = _nonce_ledger_counts(engine.state_document(), nonce)
        assert used == 1
        assert accepted == 0
    finally:
        release.set()
        worker.join(timeout=120)
    assert not worker.is_alive(), "the parked seed never returned"
    return outcome


def test_reseed_duplicate_nonce_refuses_and_never_accepts_twice(
    case: _EngineCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A seed whose nonce was consumed meanwhile is refused, applied once only."""

    engine = case.engine
    server = ViewerServer()
    server.add_engine(engine)
    _host, port = server.start_background()
    try:
        _select_rca(engine, "t2-dup-select-0000000001")
        baseline = engine.state_document()
        assert baseline["action_count"] == 1
        voxel, direction = _phantom_seed(80, 100, 60)
        outcome = _park_seed_and_consume_its_nonce(
            engine,
            monkeypatch,
            port=port,
            nonce="t2-dup-seed-0000000001",
            parent="t2-dup-select-0000000001",
            voxel=voxel,
            direction=direction,
        )
        # The consuming attempt behaved exactly as before the correction.
        consuming_status, consuming_payload = outcome["consuming"]
        assert consuming_status == 400
        assert _error_code(consuming_payload) == "UNKNOWN_PARAM"
        # The parked seed is now refused with the EXISTING closed vocabulary
        # (NONCE_REUSE) - no new rejection code, no new status word.
        seed_status, seed_payload = outcome["seed"]
        assert seed_status == 400
        assert _error_code(seed_payload) == "NONCE_REUSE"
        # The identity is recorded EXACTLY ONCE: the ledger cannot carry a
        # second entry for a nonce whose only application was refused.
        document = engine.state_document()
        assert _nonce_ledger_counts(document, "t2-dup-seed-0000000001") == (1, 0)
        assert cast(list[Any], document["used_nonces"]) == [
            "t2-dup-select-0000000001",
            "t2-dup-seed-0000000001",
        ]
        # The accepted chain and the action count advance EXACTLY ONCE (for
        # the one real action), never twice for one identity.
        assert cast(list[Any], document["accepted_nonces"]) == ["t2-dup-select-0000000001"]
        assert document["action_count"] == 1
    finally:
        server.stop()


def test_duplicate_nonce_refusal_applies_nothing_beyond_the_refusal(
    coronary_viewer: CoronaryViewerFixture,
    checkout_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The refused duplicate mutates NOTHING: no reseed, no second audit event."""

    recorded: list[dict[str, object]] = []

    class _Recorder:
        def record(self, **kwargs: object) -> None:
            recorded.append(dict(kwargs))

    engine = ViewerEngine(
        coronary_viewer.session,
        id_factory=_offset_ulids(790000),
        now=lambda: _AT,
        repo_root=checkout_root,
        audit_sink=_Recorder(),
    )
    engine.open()
    server = ViewerServer()
    server.add_engine(engine)
    _host, port = server.start_background()
    try:
        _select_rca(engine, "t2-dupapply-select-000000001")
        # The restored session already carries its own seed; a second
        # application would move it to a NEW centerline.
        seed_before = engine.current_seed()
        assert seed_before is not None
        voxel, direction = _phantom_seed(80, 100, 60)
        entered = threading.Event()
        release = threading.Event()
        real_execute_seed = ViewerEngine._execute_seed

        def _paused_execute_seed(
            self: ViewerEngine, request: SeedRequest
        ) -> dict[str, object]:
            if request.client_nonce == "t2-dupapply-seed-0000000001":
                entered.set()
                assert release.wait(timeout=60), "the parked seed was never released"
            return real_execute_seed(self, request)

        monkeypatch.setattr(ViewerEngine, "_execute_seed", _paused_execute_seed)
        outcome: dict[str, Any] = {}

        def _fire() -> None:
            outcome["seed"] = _post_seed(
                port,
                engine.session_id(),
                _seed_body(
                    session_id=engine.session_id(),
                    voxel=voxel,
                    direction=direction,
                    nonce="t2-dupapply-seed-0000000001",
                    parent="t2-dupapply-select-000000001",
                ),
            )

        worker = threading.Thread(target=_fire)
        worker.start()
        try:
            assert entered.wait(timeout=60), "the seed never reached its acceptance path"
            _post_seed(
                port,
                engine.session_id(),
                canonical_json(
                    {
                        "client_nonce": "t2-dupapply-seed-0000000001",
                        "direction": direction,
                        "lineage": {
                            "parent_action_nonce": "t2-dupapply-select-000000001",
                            "session_id": engine.session_id(),
                        },
                        "unexpected_key": 1,
                        "voxel": voxel,
                    }
                ).encode("utf-8"),
            )
            # Snapshot AFTER the consuming attempt, so the only difference the
            # duplicate could make is its own.
            before = _application_snapshot(engine)
            before_records = [dict(event) for event in recorded]
        finally:
            release.set()
            worker.join(timeout=120)
        assert not worker.is_alive(), "the parked seed never returned"
        status, payload = outcome["seed"]
        assert status == 400
        assert _error_code(payload) == "NONCE_REUSE"
        # NOTHING was applied: no re-seed, no new centerline or correction, no
        # new payload document, no new provenance, no served-state movement.
        assert _application_snapshot(engine) == before
        # Explicitly: the served seed never moved to a new derivation.
        assert engine.current_seed() == seed_before
        assert engine.current_seed() is not None
        # Exactly ONE new audit event - the real refusal bookkeeping - and no
        # acceptance event for the refused identity.
        assert len(recorded) == len(before_records) + 1
        event = recorded[-1]
        assert recorded[:-1] == before_records
        assert event["event_type"] == "VIEWER_ACTION_REJECTED"
        assert event["to_value"] == "NONCE_REUSE"
        assert event["payload"] == {"code": "NONCE_REUSE"}
        assert not [
            entry
            for entry in recorded
            if entry.get("event_type") == "VIEWER_ACTION_ACCEPTED"
            and entry.get("entity_id") == "t2-dupapply-seed-0000000001"
        ]
    finally:
        server.stop()


def test_first_seed_paths_are_unchanged_by_the_duplicate_guard(
    coronary_viewer: CoronaryViewerFixture,
    checkout_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An UNSEEN nonce behaves exactly as before on every first-seed path.

    Covers, in one session per branch: a fresh nonce still succeeds (re-seed
    and first-build), ``SEED_OUT_OF_BOUNDS`` still refuses typed with the
    nonce consumed and its replay still ``NONCE_REUSE``, and the
    NON-CONSUMING ``SESSION_OPEN_FAILED`` still leaves the nonce UNUSED so
    the same identity may legitimately be retried (the case the duplicate
    guard must never block).
    """

    # (1) Re-seed on a session that already carries an ACTIVE centerline.
    engine = ViewerEngine(
        coronary_viewer.session,
        id_factory=_offset_ulids(795000),
        now=lambda: _AT,
        repo_root=checkout_root,
    )
    engine.open()
    server = ViewerServer()
    server.add_engine(engine)
    _host, port = server.start_background()
    try:
        _select_rca(engine, "t2-fresh-select-00000000001")
        seed_before = engine.current_seed()
        assert seed_before is not None
        voxel, direction = _phantom_seed(80, 100, 60)
        status, document = _post_seed(
            port,
            engine.session_id(),
            _seed_body(
                session_id=engine.session_id(),
                voxel=voxel,
                direction=direction,
                nonce="t2-fresh-seed-0000000001",
                parent="t2-fresh-select-00000000001",
            ),
        )
        assert status == 200
        assert document["action_count"] == 2
        assert _nonce_ledger_counts(document, "t2-fresh-seed-0000000001") == (1, 1)
        # A fresh nonce really applies: a NEW derivation replaced the ACTIVE
        # one (the guard refuses duplicates only, never a first application).
        applied = engine.current_seed()
        assert applied is not None
        assert applied["centerline_ref"] != seed_before["centerline_ref"]
        # Out-of-volume refusal is untouched: typed, fail-closed, the ACTIVE
        # path stays, and the refused nonce is consumed (replay NONCE_REUSE).
        _voxel, direction_oob = _phantom_seed(80, 100, 60)
        oob_body = _seed_body(
            session_id=engine.session_id(),
            voxel=[48, 24, 24],
            direction=direction_oob,
            nonce="t2-fresh-oob-00000000001",
            parent="t2-fresh-seed-0000000001",
        )
        status, payload = _post_seed(port, engine.session_id(), oob_body)
        assert status == 409
        assert _error_code(payload) == "SEED_OUT_OF_BOUNDS"
        assert _nonce_ledger_counts(engine.state_document(), "t2-fresh-oob-00000000001") == (
            1,
            0,
        )
        active = engine.coronary_overlay_document()["centerline_ref"]
        status, payload = _post_seed(port, engine.session_id(), oob_body)
        assert status == 400
        assert _error_code(payload) == "NONCE_REUSE"
        assert engine.coronary_overlay_document()["centerline_ref"] == active
    finally:
        server.stop()

    # (2) The non-consuming SESSION_OPEN_FAILED: the identity stays UNUSED,
    # so the very same nonce is a legitimate first application afterwards.
    engine = ViewerEngine(
        coronary_viewer.session,
        id_factory=_offset_ulids(796000),
        now=lambda: _AT,
        repo_root=checkout_root,
    )
    engine.open()
    server = ViewerServer()
    server.add_engine(engine)
    _host, port = server.start_background()
    real_persisted_transform = ViewerEngine._persisted_transform

    def _failing_transform(
        self: ViewerEngine, frame: FrameSpec | None = None
    ) -> tuple[Any, str, str]:
        raise ViewerOpenError(
            RejectionCode.SESSION_OPEN_FAILED,
            ["the correction route requires the persisted versioned transform"],
        )

    try:
        _select_rca(engine, "t2-openfail-select-000000001")
        monkeypatch.setattr(ViewerEngine, "_persisted_transform", _failing_transform)
        voxel, direction = _phantom_seed(80, 100, 60)
        body = _seed_body(
            session_id=engine.session_id(),
            voxel=voxel,
            direction=direction,
            nonce="t2-openfail-seed-0000000001",
            parent="t2-openfail-select-000000001",
        )
        status, payload = _post_seed(port, engine.session_id(), body)
        assert status == 409
        assert _error_code(payload) == "SESSION_OPEN_FAILED"
        # Non-consuming: the identity was never accepted and never consumed.
        assert _nonce_ledger_counts(engine.state_document(), "t2-openfail-seed-0000000001") == (
            0,
            0,
        )
        # The duplicate guard must not block this legitimate first
        # application: the SAME nonce succeeds once the session recovers.
        monkeypatch.setattr(
            ViewerEngine, "_persisted_transform", real_persisted_transform
        )
        status, document = _post_seed(port, engine.session_id(), body)
        assert status == 200
        assert _nonce_ledger_counts(document, "t2-openfail-seed-0000000001") == (1, 1)
        assert document["action_count"] == 2
    finally:
        server.stop()

    # (3) First-build on a centerline-free session (the real T2 case).
    first_build = _volume_only_engine(
        _strip_to_volume_only(coronary_viewer.session), checkout_root, ulid_start=797000
    )
    server = ViewerServer()
    server.add_engine(first_build)
    _host, port = server.start_background()
    try:
        voxel, direction = _phantom_seed(40, 60, 20)
        body = _seed_body(
            session_id=first_build.session_id(),
            voxel=voxel,
            direction=direction,
            nonce="t2-fresh-firstbuild-00000001",
            parent=None,
        )
        status, document = _post_seed(port, first_build.session_id(), body)
        assert status == 200
        assert _nonce_ledger_counts(document, "t2-fresh-firstbuild-00000001") == (1, 1)
        assert document["action_count"] == 1
        # A replay of an already-used nonce is still the pre-existing
        # validation-time NONCE_REUSE refusal.
        status, payload = _post_seed(port, first_build.session_id(), body)
        assert status == 400
        assert _error_code(payload) == "NONCE_REUSE"
    finally:
        server.stop()
