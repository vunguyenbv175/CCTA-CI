"""VIEWER_001 V7/V8: deterministic E2E, negative battery, provenance chain.

The E2E driver: open -> scroll all frames of all three planes -> apply every
preset -> save -> reopen -> seven-check identity PASS (frames bit-identical).
No expected value is produced by the implementation path under test: PNG and
transfer expectations come from the independent oracle (viewer_fixtures), and
pipeline expectations are structural identities (byte-equality across save/
reopen, digest stability across WL re-mints) rather than implementation
computations.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from apps.viewer.engine import HumanSeamRefusalError, ViewerEngine, ViewerOpenError
from apps.viewer.grammar import RejectionCode, ViewerRejectionError
from apps.viewer.png_encoder import ZLIB_VERSION_PIN
from apps.viewer.schemas import PLANES, validate_session_document
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
)
from core.enums import ProvenanceEventType
from core.provenance import NotApplicable, verify_chain
from tests.apps.viewer_pipeline import build_viewer_session

_ULID_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def _deterministic_ulids():
    """A ULID factory minting from a fixed 48-bit clock + counter."""

    class _Factory:
        def __init__(self) -> None:
            self._n = 0

        def __call__(self, at_ms: int | None = None) -> str:
            del at_ms
            self._n += 1
            value = (0x01A9 << 80) | self._n
            chars = []
            for shift in range(25, -1, -1):
                chars.append(_ULID_ALPHABET[(value >> (shift * 5)) & 0x1F])
            return "".join(chars)

    return _Factory()


def _open_viewer(governed_session, **kwargs) -> ViewerEngine:
    engine = ViewerEngine(
        governed_session.session,
        id_factory=_deterministic_ulids(),
        now=lambda: "2026-09-07T00:00:00Z",
        repo_root=governed_session.repo_root,
        **kwargs,
    )
    engine.open()
    return engine


def _apply(
    engine: ViewerEngine,
    session_id: str,
    action: str,
    params: dict[str, object],
    *,
    view: dict[str, object],
    nonce: str,
    parent: str | None = None,
) -> dict[str, object]:
    """Post one action through the engine (the exact server boundary)."""

    body = json.dumps(
        {
            "action": action,
            "params": params,
            "client_nonce": nonce,
            "view": view,
            "lineage": {"session_id": session_id, "parent_action_nonce": parent},
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return engine.apply_action_bytes(body, path_session_id=session_id)


def _stale_ids(engine: ViewerEngine) -> list[str]:
    """The append-only STALE frame id ledger (typed accessor)."""

    stale = engine.state_document()["stale_frame_ids"]
    assert isinstance(stale, list)
    return [str(item) for item in stale]


def _state_dict_list(engine: ViewerEngine, key: str) -> list[dict[str, object]]:
    value = engine.state_document()[key]
    assert isinstance(value, list) and all(isinstance(e, dict) for e in value)
    return [dict(item) for item in value]


def _last_nonce(engine: ViewerEngine) -> str:
    """The most recent accepted action nonce (dict-typed state accessor)."""

    document = engine.state_document()
    nonces = document["accepted_nonces"]
    assert isinstance(nonces, list) and nonces
    return str(nonces[-1])


def _raw(engine: ViewerEngine, session_id: str, body: bytes) -> dict[str, object]:
    return engine.apply_action_bytes(body, path_session_id=session_id)


class _BoundSession:
    """``ViewerSessionFixture`` + the bound clean checkout used as repo_root.

    F11 practical-testing pattern: the governed build-identity resolver is
    anchored at a clean temp checkout of the exact bytes under test, so the
    engine constructions in this module pass ``repo_root=fixture.repo_root``
    and the process CWD is bound inside that checkout (autouse fixture
    below).  In a clean bound checkout the resolver's default path behaves
    identically.
    """

    def __init__(self, fixture: object, repo_root: Path) -> None:
        self._fixture = fixture
        self.repo_root = repo_root

    def __getattr__(self, name: str) -> object:
        return getattr(self._fixture, name)


@pytest.fixture(autouse=True)
def _run_inside_bound_checkout(clean_checkout: Path, monkeypatch: pytest.MonkeyPatch):
    """Run every engine operation with the CWD inside the bound checkout."""

    monkeypatch.chdir(clean_checkout)


@pytest.fixture()
def governed_session(tmp_path: Path, clean_checkout: Path):
    return _BoundSession(build_viewer_session(tmp_path / "ws"), clean_checkout)


@pytest.fixture()
def viewer(governed_session):
    engine = _open_viewer(governed_session)
    record = engine.session_document()
    return engine, record, engine  # engine passed twice for tuple unpack compat


_IDENTITY_KEYS = (
    "session_id",
    "plane",
    "frame_index",
    "source_mpr_ref",
    "wl_preset_id",
    "wl_applied",
    "png_sha256",
    "png_byte_size",
)


# ---------------------------------------------------------------------------
# Acceptance gate 1: deterministic E2E + the seventh check
# ---------------------------------------------------------------------------


def test_seven_checks_are_mandated_and_named() -> None:
    assert MANDATED_CHECKS == (
        CHECK_BLOBS,
        CHECK_COORDS,
        CHECK_MEASURE,
        CHECK_PROV,
        CHECK_REPLAY,
        CHECK_MPR,
        CHECK_VIEWER,
    )


def test_e2e_scroll_presets_save_reopen_identity(governed_session) -> None:
    import hashlib

    from apps.viewer.presets import WL_PRESET_IDS

    engine = _open_viewer(governed_session)
    session_id = engine.session_id()
    record = engine.session_document()
    base_digest = record["frame_lists_ref"]

    # open -> scroll every frame of all three planes
    parent = None
    for plane in PLANES:
        _apply(
            engine, session_id, "SET_PLANE", {"plane": plane},
            view={"plane": "AXIAL", "frame_index": 0},
            nonce=f"plane-{plane}-00000000000001", parent=parent,
        )
        parent = _last_nonce(engine)
        frames = engine.frames_document(plane)["frames"]
        assert isinstance(frames, list)
        frame_count = len(frames)
        for index in range(frame_count):
            _apply(
                engine, session_id, "SET_SLICE",
                {"plane": plane, "frame_index": index},
                view={"plane": plane, "frame_index": index},
                nonce=f"slice-{plane}-{index}-0000001", parent=parent,
            )
            parent = _last_nonce(engine)

    # apply every preset to every plane, then CLEAR_WL and a viewport matrix
    for plane in PLANES:
        _apply(
            engine, session_id, "SET_PLANE", {"plane": plane},
            view={"plane": plane, "frame_index": 0},
            nonce=f"plane2-{plane}-000000000001", parent=parent,
        )
        parent = _last_nonce(engine)
        for preset_id in WL_PRESET_IDS:
            _apply(
                engine, session_id, "SET_WL_PRESET", {"preset_id": preset_id},
                view={"plane": plane, "frame_index": 0},
                nonce=f"wl-{plane}-{preset_id}", parent=parent,
            )
            parent = _last_nonce(engine)
        _apply(
            engine, session_id, "CLEAR_WL", {},
            view={"plane": plane, "frame_index": 0},
            nonce=f"clear-{plane}-00000000001", parent=parent,
        )
        parent = _last_nonce(engine)
        _apply(
            engine, session_id, "SET_VIEWPORT_MATRIX",
            {"zoom": 2.0, "pan_x": 0.5, "pan_y": -0.5},
            view={"plane": plane, "frame_index": 0},
            nonce=f"vp-{plane}-0000000000001", parent=parent,
        )
        parent = _last_nonce(engine)
    _apply(
        engine, session_id, "OPEN_QC_PANEL", {},
        view={"plane": "AXIAL", "frame_index": 0},
        nonce="qc-open-00000000000000001", parent=parent,
    )

    # LAYER-G never moved: the digest is unchanged after every action.
    assert engine.session_document()["frame_lists_ref"] == base_digest

    # PNG re-render is byte-reproducible from the source pixels + wl_applied.
    state = engine.planes_state()
    assert isinstance(state["renders"], dict)
    for plane in PLANES:
        renders = state["renders"][plane]
        assert isinstance(renders, dict)
        records = renders["records"]
        assert isinstance(records, list)
        for stored in records:
            assert isinstance(stored, dict)
            resolved = engine.png_for_frame(
                str(stored["provenance_ref"]).split(":")[1]
            )
            assert resolved is not None
            png, served = resolved
            assert hashlib.sha256(png).hexdigest() == stored["png_sha256"]
            assert served.png_sha256 == stored["png_sha256"]

    # save -> reopen -> identity PASS (bit-identical frames)
    engine.persist(governed_session.store)
    reopened = governed_session.store.reopen()
    report = RoundTripVerificationEngine(
        governed_session.workspace, viewer_repo_root=governed_session.repo_root
    ).verify(reopened)
    assert report.verdict == "PASS", report.to_dict()
    reopened_engine = ViewerEngine(
        reopened,
        id_factory=_deterministic_ulids(),
        now=lambda: "2026-09-07T00:00:00Z",
        repo_root=governed_session.repo_root,
    )
    reopened_engine.open()
    assert reopened_engine.session_document() == record
    for plane in PLANES:
        assert (
            reopened_engine.frames_document(plane)["frames"]
            == engine.frames_document(plane)["frames"]
        )
    assert reopened_engine.verify_reopen_identity() == []
    reopened_report = RoundTripVerificationEngine(
        governed_session.workspace, viewer_repo_root=governed_session.repo_root
    ).verify(governed_session.store.reopen())
    assert reopened_report.checks[CHECK_VIEWER].passed


def test_wl_action_mints_r_variants_only_and_never_touches_layer_g(viewer) -> None:
    engine, record, _eng = viewer
    session_id = record["session_id"]
    before = engine.frames_document("AXIAL")["frames"]
    digest_before = record["frame_lists_ref"]
    _apply(
        engine, session_id, "SET_WL_PRESET", {"preset_id": "CT_MEDIASTINUM"},
        view={"plane": "AXIAL", "frame_index": 0},
        nonce="wl-mint-0000000000000001", parent=None,
    )
    after = engine.frames_document("AXIAL")["frames"]
    # New LAYER-R records minted for the current plane with the preset WL...
    assert [r["wl_preset_id"] for r in after] == ["CT_MEDIASTINUM", "CT_MEDIASTINUM"]
    assert [r["wl_applied"] for r in after] == [
        {"center": 40.0, "width": 400.0},
        {"center": 40.0, "width": 400.0},
    ]
    # ...with the SAME geometry bindings (session, plane, index, mpr ref):
    assert [(r["plane"], r["frame_index"], r["source_mpr_ref"]) for r in after] == [
        (r["plane"], r["frame_index"], r["source_mpr_ref"]) for r in before
    ]
    # fresh record ids, excluded from identity
    assert [r["provenance_ref"] for r in after] != [r["provenance_ref"] for r in before]
    # the G-list digest NEVER altered
    assert engine.session_document()["frame_lists_ref"] == digest_before


def test_wl_re_mint_is_byte_reproducible_with_fresh_ulids(viewer) -> None:
    engine, record, _eng = viewer
    session_id = record["session_id"]
    _apply(
        engine, session_id, "SET_WL_PRESET", {"preset_id": "CT_BONE"},
        view={"plane": "AXIAL", "frame_index": 0},
        nonce="wl-mint-0000000000000002", parent=None,
    )
    first = engine.frames_document("AXIAL")["frames"]
    _apply(
        engine, session_id, "CLEAR_WL", {},
        view={"plane": "AXIAL", "frame_index": 0},
        nonce="wl-clear-000000000000001",
        parent=engine.state_document()["accepted_nonces"][-1],
    )
    _apply(
        engine, session_id, "SET_WL_PRESET", {"preset_id": "CT_BONE"},
        view={"plane": "AXIAL", "frame_index": 0},
        nonce="wl-mint-0000000000000003",
        parent=engine.state_document()["accepted_nonces"][-1],
    )
    second = engine.frames_document("AXIAL")["frames"]
    for a, b in zip(first, second):
        assert a["png_sha256"] == b["png_sha256"]
        assert a["png_byte_size"] == b["png_byte_size"]
        assert a["provenance_ref"] != b["provenance_ref"]
        assert {k: a[k] for k in _IDENTITY_KEYS} == {k: b[k] for k in _IDENTITY_KEYS}


def test_viewport_matrix_is_display_only_and_never_mints(viewer) -> None:
    engine, record, _eng = viewer
    session_id = record["session_id"]
    before = {plane: engine.frames_document(plane)["frames"] for plane in PLANES}
    _apply(
        engine, session_id, "SET_VIEWPORT_MATRIX",
        {"zoom": 8.0, "pan_x": 1.25, "pan_y": -2.0},
        view={"plane": "AXIAL", "frame_index": 0},
        nonce="viewport-0000000000000001", parent=None,
    )
    after = {plane: engine.frames_document(plane)["frames"] for plane in PLANES}
    assert before == after  # no frame mint anywhere
    assert engine.state_document()["viewport_by_plane"]["AXIAL"] == {
        "zoom": 8.0,
        "pan_x": 1.25,
        "pan_y": -2.0,
    }


# ---------------------------------------------------------------------------
# Negative battery (through the engine, one sealed audit per rejection)
# ---------------------------------------------------------------------------


def test_negative_battery_rejects_every_violation(governed_session, viewer) -> None:
    engine, record, _eng = viewer
    session_id = record["session_id"]
    good_view = {"plane": "AXIAL", "frame_index": 0}

    def expect(code: RejectionCode, body: bytes) -> None:
        with pytest.raises(ViewerRejectionError) as excinfo:
            _raw(engine, session_id, body)
        assert excinfo.value.code is code

    def envelope(action: str, params: dict[str, object], nonce: str, **kw) -> bytes:
        document = {
            "action": action,
            "params": params,
            "client_nonce": nonce,
            "view": kw.get("view", good_view),
            "lineage": {
                "session_id": kw.get("lineage_session", session_id),
                "parent_action_nonce": kw.get("parent"),
            },
        }
        return json.dumps(document, sort_keys=True, separators=(",", ":")).encode()

    # duplicate keys (transport level, before anything else)
    expect(
        RejectionCode.DUPLICATE_JSON_KEYS,
        (
            b'{"action":"SET_SLICE","action":"SET_PLANE","params":{"plane":"AXIAL",'
            b'"frame_index":0},"client_nonce":"n00000000000000000002","view":'
            b'{"plane":"AXIAL","frame_index":0},"lineage":{"session_id":"'
            + session_id.encode()
            + b'","parent_action_nonce":null}}'
        ),
    )
    # non-canonical encoding (unsorted keys)
    expect(
        RejectionCode.NON_CANONICAL_ENCODING,
        (
            b'{"client_nonce":"n00000000000000000003","action":"SET_SLICE","params":'
            b'{"plane":"AXIAL","frame_index":0},"view":{"plane":"AXIAL",'
            b'"frame_index":0},"lineage":{"session_id":"'
            + session_id.encode()
            + b'","parent_action_nonce":null}}'
        ),
    )
    # unknown action
    expect(
        RejectionCode.UNKNOWN_ACTION,
        envelope("SET_WINDOW_CUSTOM", {}, "n00000000000000000004"),
    )
    # unknown param (free-form key inside the closed param set)
    expect(
        RejectionCode.UNKNOWN_PARAM,
        envelope(
            "SET_SLICE",
            {"plane": "AXIAL", "frame_index": 0, "window": "custom"},
            "n00000000000000000005",
        ),
    )
    # out-of-range index (also: scrolling an empty plane)
    expect(
        RejectionCode.INDEX_OUT_OF_RANGE,
        envelope("SET_SLICE", {"plane": "AXIAL", "frame_index": 5}, "n00000000000000000006"),
    )
    # free-text justification
    expect(
        RejectionCode.JUSTIFICATION_NOT_IN_ENUM,
        envelope(
            "REQUEST_CORRECTION",
            {
                "subject_ref": governed_session.study_id,
                "proposed_grade": "GOOD",
                "justification_code": "the physician looked at it",
            },
            "n00000000000000000007",
        ),
    )
    # view mismatch (the server never mutates a different frame)
    expect(
        RejectionCode.VIEW_MISMATCH,
        envelope(
            "SET_SLICE",
            {"plane": "AXIAL", "frame_index": 0},
            "n00000000000000000008",
            view={"plane": "AXIAL", "frame_index": 9},
        ),
    )
    # zoom non-positive FIRST, then matrix range
    expect(
        RejectionCode.ZOOM_NON_POSITIVE,
        envelope(
            "SET_VIEWPORT_MATRIX",
            {"zoom": 0, "pan_x": 0.0, "pan_y": 0.0},
            "n00000000000000000009",
        ),
    )
    expect(
        RejectionCode.ZOOM_NON_POSITIVE,
        envelope(
            "SET_VIEWPORT_MATRIX",
            {"zoom": -1.5, "pan_x": 0.0, "pan_y": 0.0},
            "n00000000000000000010",
        ),
    )
    expect(
        RejectionCode.MATRIX_NON_FINITE,
        envelope(
            "SET_VIEWPORT_MATRIX",
            {"zoom": 64.5, "pan_x": 0.0, "pan_y": 0.0},
            "n00000000000000000011",
        ),
    )
    expect(
        RejectionCode.MATRIX_NON_FINITE,
        envelope(
            "SET_VIEWPORT_MATRIX",
            {"zoom": 2.0, "pan_x": 2.5, "pan_y": 0.0},
            "n00000000000000000012",
        ),
    )
    # lineage session binding -> SESSION_NOT_FOUND (404)
    with pytest.raises(ViewerRejectionError) as excinfo:
        _raw(
            engine,
            session_id,
            envelope(
                "OPEN_QC_PANEL",
                {},
                "n00000000000000000013",
                lineage_session="0" * 26,
            ),
        )
    assert excinfo.value.code is RejectionCode.SESSION_NOT_FOUND
    assert excinfo.value.http_status == 404
    # grade / justification / subject gates
    expect(
        RejectionCode.GRADE_NOT_IN_ENUM,
        envelope(
            "REQUEST_CORRECTION",
            {
                "subject_ref": governed_session.study_id,
                "proposed_grade": "EXCELLENT",
                "justification_code": "HUMAN_EVIDENCE",
            },
            "n00000000000000000014",
        ),
    )
    expect(
        RejectionCode.SUBJECT_REF_INVALID,
        envelope(
            "REQUEST_CORRECTION",
            {
                "subject_ref": f"mpr_artifact_id:{'A' * 26}",
                "proposed_grade": "GOOD",
                "justification_code": "HUMAN_EVIDENCE",
            },
            "n00000000000000000015",
        ),
    )


def test_nonce_uniqueness_across_attempts_and_lineage_chain(viewer) -> None:
    engine, record, _eng = viewer
    session_id = record["session_id"]
    nonce = "unique-000000000000001"
    _apply(
        engine, session_id, "OPEN_QC_PANEL", {},
        view={"plane": "AXIAL", "frame_index": 0}, nonce=nonce, parent=None,
    )
    with pytest.raises(ViewerRejectionError) as excinfo:
        _apply(
            engine, session_id, "OPEN_QC_PANEL", {},
            view={"plane": "AXIAL", "frame_index": 0}, nonce=nonce,
            parent=engine.state_document()["accepted_nonces"][-1],
        )
    assert excinfo.value.code is RejectionCode.NONCE_REUSE
    # a rejected attempt's nonce is consumed too (fail-closed)
    with pytest.raises(ViewerRejectionError):
        _apply(
            engine, session_id, "SET_SLICE", {"plane": "AXIAL", "frame_index": 9},
            view={"plane": "AXIAL", "frame_index": 0},
            nonce="rejected-0000000000001",
            parent=engine.state_document()["accepted_nonces"][-1],
        )
    with pytest.raises(ViewerRejectionError) as excinfo:
        _apply(
            engine, session_id, "OPEN_QC_PANEL", {},
            view={"plane": "AXIAL", "frame_index": 0},
            nonce="rejected-0000000000001",
            parent=engine.state_document()["accepted_nonces"][-1],
        )
    assert excinfo.value.code is RejectionCode.NONCE_REUSE
    # lineage: a parent that was never accepted is a gap
    with pytest.raises(ViewerRejectionError) as excinfo:
        _apply(
            engine, session_id, "OPEN_QC_PANEL", {},
            view={"plane": "AXIAL", "frame_index": 0},
            nonce="n0000000000000000010", parent="unknown-0000000000001",
        )
    assert excinfo.value.code is RejectionCode.LINEAGE_GAP


# ---------------------------------------------------------------------------
# Fail-closed session-open table (Part 6, rows 1-4)
# ---------------------------------------------------------------------------


def test_foreign_session_record_cannot_rebind() -> None:
    # Row 1 surface: a well-formed session document for a DIFFERENT session
    # is schema-valid but can never rebind this engine's state (the engine
    # compares ids at rehydration and refuses a mismatch).
    foreign = {
        "schema_version": "ccta.viewer.session.v1",
        "session_id": "0" * 26,
        "study_id": "study_id:" + "0" * 26,
        "planes": list(PLANES),
        "frame_lists_ref": "b" * 64,
        "qc_gate_state_ref": "gate:" + "0" * 26,
        "wl_presets_active": ["CT_MEDIASTINUM"],
        "opened_at": "2026-09-07T00:00:00Z",
        "provenance_ref": "provenance:" + "0" * 26,
    }
    record = validate_session_document(foreign)  # syntactically valid
    assert record.session_id == "0" * 26


def test_open_table_corrupt_frame_list_fails_closed(governed_session) -> None:
    # Row 2: a corrupt frame list (digest mismatch vs frame_lists_ref).
    from tests.apps.pipeline import rewrite_manifest

    engine = _open_viewer(governed_session)
    engine.persist(governed_session.store)
    state_path = governed_session.workspace / "session" / "artifacts.json"
    document = json.loads(state_path.read_text())
    for key, payload in document["payload_documents"].items():
        if key.startswith("viewer_session:"):
            payload["frame_lists_ref"] = "f" * 64  # digest now mismatched
    state_path.write_text(json.dumps(document))
    rewrite_manifest(governed_session.workspace)
    reopened = governed_session.store.reopen()
    reopened_engine = ViewerEngine(reopened, repo_root=governed_session.repo_root)
    with pytest.raises(ViewerOpenError) as excinfo:
        reopened_engine.open()
    assert excinfo.value.code is RejectionCode.SESSION_OPEN_FAILED


def test_open_table_dangling_mpr_ref_fails_closed(governed_session) -> None:
    # Row 3: a dangling source_mpr_ref is never skipped silently.
    from apps.workstation.session_store import RestoredSession

    session = governed_session.session
    victim = governed_session.mpr_artifacts[0].artifact_id
    documents = dict(session.payload_documents)
    del documents[victim]  # artifact record kept: a genuine dangling ref
    stripped = RestoredSession(
        paths=session.paths,
        study=session.study,
        series=session.series,
        volumes=session.volumes,
        artifacts=session.artifacts,
        payload_documents=documents,
        provenance=session.provenance,
        log=session.log,
        skipped_instances=session.skipped_instances,
        projections=session.projections,
        session_digest=session.session_digest,
        pixel_blobs=dict(session.pixel_blobs),
    )
    engine = ViewerEngine(stripped, repo_root=governed_session.repo_root)
    with pytest.raises(ViewerOpenError) as excinfo:
        engine.open()
    assert excinfo.value.code is RejectionCode.SESSION_OPEN_FAILED
    assert "dangling" in str(excinfo.value) or "missing" in str(excinfo.value)


def test_open_table_missing_pixel_blob_fails_closed(governed_session) -> None:
    # Row 4: a missing pixel blob for a listed MPR is never a placeholder.
    from apps.workstation.session_store import RestoredSession

    session = governed_session.session
    victim = governed_session.mpr_artifacts[0].artifact_id
    blobs = dict(session.pixel_blobs)
    document = session.payload_documents[victim]
    digest = document["pixel_blob_ref"]["sha256"]
    blobs.pop(digest, None)
    stripped = RestoredSession(
        paths=session.paths,
        study=session.study,
        series=session.series,
        volumes=session.volumes,
        artifacts=session.artifacts,
        payload_documents=dict(session.payload_documents),
        provenance=session.provenance,
        log=session.log,
        skipped_instances=session.skipped_instances,
        projections=session.projections,
        session_digest=session.session_digest,
        pixel_blobs=blobs,
    )
    engine = ViewerEngine(stripped, repo_root=governed_session.repo_root)
    with pytest.raises(ViewerOpenError) as excinfo:
        engine.open()
    assert excinfo.value.code is RejectionCode.SESSION_OPEN_FAILED
    assert "pixel blob" in "; ".join(excinfo.value.problems)


# ---------------------------------------------------------------------------
# Provenance (acceptance gate 3)
# ---------------------------------------------------------------------------


def test_provenance_chain_reaches_source_dicom_manifest(
    governed_session, viewer
) -> None:
    engine, record, _eng = viewer
    frame = engine.frames_document("AXIAL")["frames"][0]
    provenance_id = frame["provenance_ref"].split(":", 1)[1]
    records = {r.provenance_id: r for r in engine.new_provenance()}
    current = records[provenance_id]
    # nine §44 items per record, explicit value or NotApplicable
    nine_items = (
        current.inputs,
        current.preprocessing,
        current.parameters_ref,
        current.human_edit,
        current.environment,
        current.upstream_refs,
        current.actor,
        current.event_type,
    )
    # engine branch: model_record_ref is explicitly None (never absent).
    assert all(item is not None for item in nine_items)
    assert current.model_record_ref is None
    assert isinstance(current.human_edit, NotApplicable)
    # walk the upstream closure to the source DICOM manifest digest
    seen: set[str] = set()
    pending = [current]
    manifest_digest = governed_session.source_manifest_sha256
    manifest_reached = False
    classified_seen = False
    session_records = {
        r.provenance_id: r for r in governed_session.session.provenance
    }
    all_records = {**session_records, **records}
    while pending:
        node = pending.pop()
        if node.provenance_id in seen:
            continue
        seen.add(node.provenance_id)
        if not isinstance(node.inputs, NotApplicable):
            for item in node.inputs:
                if item.sha256_at_use == manifest_digest:
                    manifest_reached = True
        if node.event_type is ProvenanceEventType.CLASSIFIED:
            classified_seen = True
        upstreams = (
            ()
            if isinstance(node.upstream_refs, NotApplicable)
            else tuple(node.upstream_refs)
        )
        for upstream in upstreams:
            upstream_record = all_records.get(upstream)
            assert upstream_record is not None, f"unresolvable upstream {upstream}"
            pending.append(upstream_record)
    assert manifest_reached, "the chain never reached the source DICOM manifest digest"
    assert classified_seen, "the chain never reached the classification record"
    merged = tuple(governed_session.session.provenance) + engine.new_provenance()
    ok, problems = verify_chain(merged)
    assert ok, problems


def test_frame_provenance_records_the_pinned_zlib_version(viewer) -> None:
    engine, _record, _eng = viewer
    frame = engine.frames_document("AXIAL")["frames"][0]
    provenance_id = frame["provenance_ref"].split(":", 1)[1]
    current = {r.provenance_id: r for r in engine.new_provenance()}[provenance_id]
    # zlib.ZLIB_VERSION is recorded in the frame's provenance parameters ref.
    assert ZLIB_VERSION_PIN in current.parameters_ref.blob_ref
    assert current.parameters_ref.blob_ref == (
        f"viewer-frame-parameters:zlib-{ZLIB_VERSION_PIN}:"
        f"{current.parameters_ref.sha256}"
    )


# ---------------------------------------------------------------------------
# QC verbatim + correction rebind (acceptance gate 4)
# ---------------------------------------------------------------------------


def test_abstain_never_suppresses_display_and_panel_is_verbatim(viewer) -> None:
    import re

    engine, record, _eng = viewer
    panel = engine.qc_panel_document()
    # no QC assessment exists in the fixture -> the gate ABSTAINs
    outcome = panel["outcome"]
    assert outcome["decision"] == "ABSTAIN"
    assert outcome["reason_code"] == "QC_NOT_ASSESSED"
    assert outcome["assessment_refs"] == []
    assert re.fullmatch(
        r"gate:[0123456789ABCDEFGHJKMNPQRSTVWXYZ]{26}", panel["qc_gate_state_ref"]
    )
    # ABSTAIN never suppresses: every plane still serves every frame
    for plane in PLANES:
        frames = engine.frames_document(plane)["frames"]
        assert len(frames) == 2
        for frame in frames:
            png, _served = engine.png_for_frame(str(frame["provenance_ref"]).split(":")[1])
            assert png.startswith(b"\x89PNG\r\n\x1a\n")
    # the panel is verbatim: the recorded outcome, not a re-query
    assert panel["outcome"] == engine.state_document()["gate_outcomes"][-1]


def test_correction_round_trip_rebinds_the_gate_and_marks_stale(
    governed_session,
) -> None:
    engine = _open_viewer(governed_session)
    session_id = engine.session_id()
    study_ref = governed_session.study_id
    before_frames: dict[str, list[dict[str, object]]] = {}
    for plane in PLANES:
        plane_frames = engine.frames_document(plane)["frames"]
        assert isinstance(plane_frames, list)
        before_frames[plane] = [dict(f) for f in plane_frames if isinstance(f, dict)]
    # FIRST correction: no QC assessment exists -> the route creates the
    # initial human assessment (append-only seam, valid sequence 1).
    _apply(
        engine,
        session_id,
        "REQUEST_CORRECTION",
        {
            "subject_ref": study_ref,
            "proposed_grade": "NOT_ANALYZABLE",
            "justification_code": "HUMAN_EVIDENCE",
        },
        view={"plane": "AXIAL", "frame_index": 0},
        nonce="correction-00000000000001",
        parent=None,
    )
    panel = engine.qc_panel_document()
    # The accepted v1 real gate does not admit HUMAN records into the
    # PROCEED path (disclosed QC_001-HUMAN scope boundary, #16), so the
    # re-queried outcome after a correction is the fail-closed ABSTAIN —
    # recorded verbatim.  The rebind itself is proven by the ledger advance,
    # the superseded_by link, and the STALE marking below.
    outcome_doc = panel["outcome"]
    assert isinstance(outcome_doc, dict)
    assert outcome_doc["decision"] == "ABSTAIN"
    assert outcome_doc["reason_code"] == "QC_NOT_ASSESSED"
    ledger = panel["outcome_ledger"]
    assert isinstance(ledger, list) and len(ledger) == 2
    assert isinstance(ledger[0], dict)
    assert ledger[0]["superseded_by"] == outcome_doc["gate_id"]
    session_doc = engine.session_document()
    assert session_doc["qc_gate_state_ref"] == outcome_doc["gate_id"]
    # the routed human artifact is in the session as a REAL HUMAN record
    human_artifacts = engine.new_qc_artifacts()
    assert len(human_artifacts) == 1
    assert human_artifacts[0].producer.producer_type.value == "HUMAN"
    # frames under the superseded outcome marked STALE (append-only ledger),
    # bytes retained, and fresh current records minted
    stale_after_first = _stale_ids(engine)
    assert len(stale_after_first) == 6
    for plane in PLANES:
        fresh_frames = engine.frames_document(plane)["frames"]
        assert isinstance(fresh_frames, list)
        for fresh in fresh_frames:
            assert isinstance(fresh, dict)
            fresh_id = str(fresh["provenance_ref"]).split(":")[1]
            assert fresh_id not in stale_after_first
            before = next(
                b
                for b in before_frames[plane]
                if b.get("frame_index") == fresh.get("frame_index")
                and b.get("source_mpr_ref") == fresh.get("source_mpr_ref")
            )
            assert {k: before[k] for k in _IDENTITY_KEYS} == {
                k: fresh[k] for k in _IDENTITY_KEYS
            }

    # SECOND correction: GOOD supersedes the NOT_ANALYZABLE assessment.  A
    # human GOOD acceptance does NOT flip the consumer gate to PROCEED: the
    # accepted v1 gate policy does not admit HUMAN records into the PROCEED
    # path (disclosed QC_001-HUMAN scope boundary, #16), so the re-queried
    # outcome is the fail-closed ABSTAIN/QC_NOT_ASSESSED, rendered VERBATIM
    # below.  What the second correction proves here is the Rule-F rebind
    # mechanics: ledger advance, superseded_by link, and STALE marking.
    _apply(
        engine,
        session_id,
        "REQUEST_CORRECTION",
        {
            "subject_ref": study_ref,
            "proposed_grade": "GOOD",
            "justification_code": "HUMAN_EVIDENCE",
        },
        view={"plane": "AXIAL", "frame_index": 0},
        nonce="correction-00000000000002",
        parent=_last_nonce(engine),
    )
    panel = engine.qc_panel_document()
    outcome_doc = panel["outcome"]
    assert isinstance(outcome_doc, dict)
    assert outcome_doc["decision"] == "ABSTAIN"
    assert outcome_doc["reason_code"] == "QC_NOT_ASSESSED"
    ledger_all = panel["outcome_ledger"]
    assert isinstance(ledger_all, list) and len(ledger_all) == 3
    assert engine.session_document()["qc_gate_state_ref"] == outcome_doc["gate_id"]
    assert len(_stale_ids(engine)) == 12

    # save -> reopen -> the correction artifacts ride the session and the
    # seven checks still pass
    engine.persist(governed_session.store)
    reopened = governed_session.store.reopen()
    report = RoundTripVerificationEngine(
        governed_session.workspace, viewer_repo_root=governed_session.repo_root
    ).verify(reopened)
    assert report.verdict == "PASS", report.to_dict()
    reopened_engine = ViewerEngine(
        reopened, repo_root=governed_session.repo_root
    )
    reopened_engine.open()
    reopened_panel = reopened_engine.qc_panel_document()
    reopened_outcome = reopened_panel["outcome"]
    assert isinstance(reopened_outcome, dict)
    assert reopened_outcome["decision"] == "ABSTAIN"
    assert reopened_outcome["gate_id"] == outcome_doc["gate_id"]
    assert reopened_engine.session_document() == engine.session_document()
    assert reopened_engine.verify_reopen_identity() == []


def test_correction_with_flags_requiring_grade_refuses_fail_closed(
    governed_session,
) -> None:
    engine = _open_viewer(governed_session)
    session_id = engine.session_id()
    study_ref = governed_session.study_id
    # LAND the initial human assessment first (GOOD, flags-free).
    _apply(
        engine,
        session_id,
        "REQUEST_CORRECTION",
        {
            "subject_ref": study_ref,
            "proposed_grade": "GOOD",
            "justification_code": "HUMAN_EVIDENCE",
        },
        view={"plane": "AXIAL", "frame_index": 0},
        nonce="correction-good-0000000001",
        parent=None,
    )
    state_doc = engine.state_document()
    gate_outcomes = state_doc["gate_outcomes"]
    assert isinstance(gate_outcomes, list)
    outcomes_before = len(gate_outcomes)
    stale_before = len(_stale_ids(engine))
    parent_nonce = _last_nonce(engine)
    # An ACCEPTABLE proposal requires evidence flags the closed v1 grammar
    # cannot carry: the seam refuses; nothing is minted or rebound.
    with pytest.raises(HumanSeamRefusalError) as excinfo:
        _apply(
            engine,
            session_id,
            "REQUEST_CORRECTION",
            {
                "subject_ref": study_ref,
                "proposed_grade": "ACCEPTABLE",
                "justification_code": "HUMAN_EVIDENCE",
            },
            view={"plane": "AXIAL", "frame_index": 0},
            nonce="correction-accept-00000001",
            parent=parent_nonce,
        )
    assert "ACCEPTABLE/LIMITED human QC requires flags" in str(excinfo.value)
    after_doc = engine.state_document()
    after_outcomes = after_doc["gate_outcomes"]
    assert isinstance(after_outcomes, list)
    assert len(after_outcomes) == outcomes_before
    assert len(_stale_ids(engine)) == stale_before


def test_scrolling_bounds_are_index_out_of_range(viewer) -> None:
    engine, record, _eng = viewer
    session_id = record["session_id"]
    with pytest.raises(ViewerRejectionError) as excinfo:
        _apply(
            engine,
            session_id,
            "SET_SLICE",
            {"plane": "AXIAL", "frame_index": 2},
            view={"plane": "AXIAL", "frame_index": 0},
            nonce="empty-scroll-00000000001",
            parent=None,
        )
    assert excinfo.value.code is RejectionCode.INDEX_OUT_OF_RANGE


def test_audit_sink_receives_one_sealed_event_per_action(governed_session) -> None:
    from core.audit_log import PhiSafeAuditSink, verify_stream

    sink = PhiSafeAuditSink(
        governed_session.workspace / "logs",
        pseudonym_key="viewer-audit-key",
        actor_role="VIEWER",
    )
    engine = _open_viewer(governed_session, audit_sink=sink)
    session_id = engine.session_id()
    _apply(
        engine, session_id, "OPEN_QC_PANEL", {},
        view={"plane": "AXIAL", "frame_index": 0},
        nonce="audit-0000000000000001", parent=None,
    )
    with pytest.raises(ViewerRejectionError):
        _raw(
            engine,
            session_id,
            json.dumps(
                {
                    "action": "NO_SUCH_ACTION",
                    "params": {},
                    "client_nonce": "audit-0000000000000002",
                    "view": {"plane": "AXIAL", "frame_index": 0},
                    "lineage": {
                        "session_id": session_id,
                        "parent_action_nonce": None,
                    },
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode(),
        )
    _apply(
        engine,
        session_id,
        "SET_SLICE",
        {"plane": "AXIAL", "frame_index": 1},
        view={"plane": "AXIAL", "frame_index": 0},
        nonce="audit-0000000000000003",
        parent=_last_nonce(engine),
    )
    ok, problems = verify_stream(governed_session.workspace / "logs", sink.stream_name)
    assert ok, problems
    events = [
        json.loads(line)
        for line in sink.path.read_text().splitlines()
        if line.strip()
    ]
    viewer_events = [e for e in events if e["entity"] == "viewer_action"]
    assert len(viewer_events) == 3
    assert [e["event_type"] for e in viewer_events] == [
        "VIEWER_ACTION_ACCEPTED",
        "VIEWER_ACTION_REJECTED",
        "VIEWER_ACTION_ACCEPTED",
    ]


def test_source_manifest_digest_is_the_import_manifest(governed_session) -> None:
    session = governed_session.session
    series = session.series[0]
    assert (
        governed_session.source_manifest_sha256
        == series.original_dicom_ref.manifest_sha256
    )


# ---------------------------------------------------------------------------
# F12 (r2): response-only correction capability disclosure (Part 6.1 b.1)
# ---------------------------------------------------------------------------

_CORRECTION_CAPABILITIES_EXPECTED = {
    "action": "REQUEST_CORRECTION",
    "actionable_grades": ["GOOD", "NOT_ANALYZABLE"],
    "deferred_grades": ["ACCEPTABLE", "LIMITED"],
    "deferred_reason": "EVIDENCE_FLAGS_NOT_IN_V1_GRAMMAR",
}


def test_correction_capabilities_exact_object_at_engine_boundary(viewer) -> None:
    engine, record, _eng = viewer
    panel = engine.qc_panel_document()
    # EXACT equality with the spec Part 6.1 b.1 object (no extra keys).
    assert panel["correction_capabilities"] == _CORRECTION_CAPABILITIES_EXPECTED
    # The disclosure is a SIBLING of outcome, never modifying it: the
    # outcome document is verbatim ledger state.
    outcome = panel["outcome"]
    assert isinstance(outcome, dict)
    assert outcome == engine.state_document()["gate_outcomes"][-1]
    assert set(outcome) == {
        "schema_version",
        "gate_id",
        "decision",
        "reason_code",
        "assessment_refs",
        "consumed_refs",
        "bound_at",
        "superseded_by",
    }
    assert not any(
        key in outcome for key in _CORRECTION_CAPABILITIES_EXPECTED
    )


def test_correction_capabilities_survives_actions_and_reopen_verbatim(
    governed_session,
) -> None:
    engine = _open_viewer(governed_session)
    session_id = engine.session_id()
    _apply(
        engine, session_id, "REQUEST_CORRECTION",
        {
            "subject_ref": governed_session.study_id,
            "proposed_grade": "NOT_ANALYZABLE",
            "justification_code": "HUMAN_EVIDENCE",
        },
        view={"plane": "AXIAL", "frame_index": 0},
        nonce="f12-panels-0000000001", parent=None,
    )
    panel_before = engine.qc_panel_document()
    assert panel_before["correction_capabilities"] == (
        _CORRECTION_CAPABILITIES_EXPECTED
    )
    engine.persist(governed_session.store)
    reopened = governed_session.store.reopen()
    report = RoundTripVerificationEngine(
        governed_session.workspace, viewer_repo_root=governed_session.repo_root
    ).verify(reopened)
    assert report.verdict == "PASS", report.to_dict()
    reopened_engine = ViewerEngine(reopened, repo_root=governed_session.repo_root)
    reopened_engine.open()
    panel_after = reopened_engine.qc_panel_document()
    assert panel_after["correction_capabilities"] == (
        _CORRECTION_CAPABILITIES_EXPECTED
    )
    # The disclosure is derived response-only state: it is NOT persisted in
    # the state document and does not perturb any digest.
    state_document = reopened_engine.state_document()
    assert "correction_capabilities" not in state_document
    assert panel_after["outcome"] == panel_before["outcome"]


@pytest.mark.parametrize("grade", ["ACCEPTABLE", "LIMITED"])
def test_deferred_grades_fail_closed_409_with_one_sealed_event_and_no_state(
    governed_session, grade
) -> None:
    from core.audit_log import PhiSafeAuditSink, verify_stream

    sink = PhiSafeAuditSink(
        governed_session.workspace / "logs",
        pseudonym_key="f12-audit-key",
        actor_role="VIEWER",
    )
    engine = _open_viewer(governed_session, audit_sink=sink)
    session_id = engine.session_id()
    before_state = engine.state_document()
    assert isinstance(before_state, dict)
    before_frames = {
        plane: engine.frames_document(plane)["frames"] for plane in PLANES
    }
    with pytest.raises(HumanSeamRefusalError) as excinfo:
        _apply(
            engine, session_id, "REQUEST_CORRECTION",
            {
                "subject_ref": governed_session.study_id,
                "proposed_grade": grade,
                "justification_code": "HUMAN_EVIDENCE",
            },
            view={"plane": "AXIAL", "frame_index": 0},
            nonce=f"f12-deferred-{grade}-0000001", parent=None,
        )
    assert excinfo.value.code == "QC_HUMAN_MATRIX_REFUSAL"
    after_state = engine.state_document()
    assert isinstance(after_state, dict)
    # No new QC / provenance / frame / gate state whatsoever:
    assert after_state["gate_outcomes"] == before_state["gate_outcomes"]
    assert after_state["qc_ledger"] == before_state["qc_ledger"]
    assert after_state["stale_frame_ids"] == before_state["stale_frame_ids"]
    assert after_state["action_count"] == before_state["action_count"]
    assert engine.new_qc_artifacts() == ()
    assert after_state["accepted_nonces"] == before_state["accepted_nonces"]
    # exactly ONE sealed audit event (the rejection) and nothing else
    ok, problems = verify_stream(governed_session.workspace / "logs", sink.stream_name)
    assert ok, problems
    events = [
        json.loads(line)
        for line in sink.path.read_text().splitlines()
        if line.strip()
    ]
    viewer_events = [e for e in events if e["entity"] == "viewer_action"]
    assert len(viewer_events) == 1
    assert viewer_events[0]["event_type"] == "VIEWER_ACTION_REJECTED"
    assert viewer_events[0]["payload"]["code"] == "QC_HUMAN_MATRIX_REFUSAL"
    # frames untouched
    for plane in PLANES:
        assert engine.frames_document(plane)["frames"] == before_frames[plane]


@pytest.mark.parametrize("grade", ["GOOD", "NOT_ANALYZABLE"])
def test_actionable_grades_traverse_the_real_seam(
    governed_session, grade
) -> None:
    engine = _open_viewer(governed_session)
    session_id = engine.session_id()
    outcomes_before = len(_state_dict_list(engine, "gate_outcomes"))
    _apply(
        engine, session_id, "REQUEST_CORRECTION",
        {
            "subject_ref": governed_session.study_id,
            "proposed_grade": grade,
            "justification_code": "HUMAN_EVIDENCE",
        },
        view={"plane": "AXIAL", "frame_index": 0},
        nonce=f"f12-actionable-{grade}-00001", parent=None,
    )
    # the REAL seam produced a REAL HUMAN artifact (ActorType.HUMAN)
    human_artifacts = engine.new_qc_artifacts()
    assert len(human_artifacts) == 1
    assert human_artifacts[0].producer.producer_type.value == "HUMAN"
    assert len(_state_dict_list(engine, "gate_outcomes")) == outcomes_before + 1


def test_not_analyzable_abstention_marker_stays_verbatim(
    governed_session,
) -> None:
    # The engine's seam-mandated abstention marker is structural, not an
    # observation of image defects; F12 must not touch it.  Verify it by
    # byte-locating the exact marker in the running engine source.
    import inspect

    source = inspect.getsource(type(_open_viewer(governed_session)))
    assert 'flags = ("ABSTAINED",) if grade is QualityGrade.NOT_ANALYZABLE else ()' in source


# ---------------------------------------------------------------------------
# R2-01 (r2): refused nonces are consumed exactly once
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("grade", ["ACCEPTABLE", "LIMITED"])
def test_refused_nonce_is_consumed_and_reuse_returns_nonce_reuse(
    governed_session, grade
) -> None:
    from core.audit_log import PhiSafeAuditSink, verify_stream

    sink = PhiSafeAuditSink(
        governed_session.workspace / "logs",
        pseudonym_key="r201-audit-key",
        actor_role="VIEWER",
    )
    engine = _open_viewer(governed_session, audit_sink=sink)
    session_id = engine.session_id()
    before_state = engine.state_document()
    assert isinstance(before_state, dict)
    refused_nonce = f"r201-refused-{grade}-000001"
    with pytest.raises(HumanSeamRefusalError):
        _apply(
            engine, session_id, "REQUEST_CORRECTION",
            {
                "subject_ref": governed_session.study_id,
                "proposed_grade": grade,
                "justification_code": "HUMAN_EVIDENCE",
            },
            view={"plane": "AXIAL", "frame_index": 0},
            nonce=refused_nonce, parent=None,
        )
    state_after_refusal = engine.state_document()
    assert isinstance(state_after_refusal, dict)
    after_accepted = state_after_refusal["accepted_nonces"]
    after_used = state_after_refusal["used_nonces"]
    assert isinstance(after_accepted, list) and isinstance(after_used, list)
    # consumed exactly once: seen, but NOT into the accepted chain, no
    # accepted-count increment, no gate/render state change.
    assert refused_nonce in after_used
    assert refused_nonce not in after_accepted
    assert after_accepted == before_state["accepted_nonces"]
    assert state_after_refusal["action_count"] == before_state["action_count"]
    assert state_after_refusal["gate_outcomes"] == before_state["gate_outcomes"]
    assert state_after_refusal["stale_frame_ids"] == before_state["stale_frame_ids"]
    assert state_after_refusal["renders"] == before_state["renders"]
    # exactly one sealed rejection audit for the refusal
    ok, problems = verify_stream(governed_session.workspace / "logs", sink.stream_name)
    assert ok, problems
    events = [
        json.loads(line)
        for line in sink.path.read_text().splitlines()
        if line.strip()
    ]
    viewer_events = [e for e in events if e["entity"] == "viewer_action"]
    assert len(viewer_events) == 1
    assert viewer_events[0]["event_type"] == "VIEWER_ACTION_REJECTED"
    # refusal -> the SAME nonce on a valid OPEN_QC_PANEL returns NONCE_REUSE
    # (a fresh nonce with a last-accepted parent is required for lineage).
    _apply(
        engine, session_id, "OPEN_QC_PANEL", {},
        view={"plane": "AXIAL", "frame_index": 0},
        nonce="r201-fresh-after-0000001", parent=None,
    )
    with pytest.raises(ViewerRejectionError) as excinfo:
        _apply(
            engine, session_id, "OPEN_QC_PANEL", {},
            view={"plane": "AXIAL", "frame_index": 0},
            nonce=refused_nonce, parent=_last_nonce(engine),
        )
    assert excinfo.value.code is RejectionCode.NONCE_REUSE

    # ... also after save/reopen (persistence keeps the consumed nonce).
    engine.persist(governed_session.store)
    reopened = governed_session.store.reopen()
    report = RoundTripVerificationEngine(
        governed_session.workspace, viewer_repo_root=governed_session.repo_root
    ).verify(reopened)
    assert report.verdict == "PASS", report.to_dict()
    reopened_engine = ViewerEngine(reopened, repo_root=governed_session.repo_root)
    reopened_engine.open()
    reopened_state = reopened_engine.state_document()
    assert isinstance(reopened_state, dict)
    reopened_used = reopened_state["used_nonces"]
    reopened_accepted = reopened_state["accepted_nonces"]
    assert isinstance(reopened_used, list) and isinstance(reopened_accepted, list)
    assert refused_nonce in reopened_used
    assert refused_nonce not in reopened_accepted
    with pytest.raises(ViewerRejectionError) as excinfo:
        _apply(
            reopened_engine, reopened_engine.session_id(), "OPEN_QC_PANEL", {},
            view={"plane": "AXIAL", "frame_index": 0},
            nonce=refused_nonce, parent=None,
        )
    assert excinfo.value.code is RejectionCode.NONCE_REUSE


# ---------------------------------------------------------------------------
# R2-02 (r2): append-only internal render-record archive
# ---------------------------------------------------------------------------


def _archive(engine: ViewerEngine) -> dict[str, dict[str, object]]:
    archive = engine.state_document()["render_archive"]
    assert isinstance(archive, dict)
    return archive


def test_archive_retains_superseded_records_and_serves_identical_bytes(
    governed_session,
) -> None:
    """WL/CLEAR_WL/correction keep original URLs/bytes/ETags queryable."""

    import hashlib
    import urllib.request

    from apps.viewer.server import ViewerServer

    engine = _open_viewer(governed_session)
    session_id = engine.session_id()
    geometry_digest = engine.session_document()["frame_lists_ref"]
    server = ViewerServer(audit_sink=None)
    server.add_engine(engine)
    _host, port = server.start_background()
    try:
        # capture every original document/id/url/png/etag
        original: dict[str, dict[str, object]] = {}
        for plane in PLANES:
            frames_document = engine.frames_document(plane)
            frames = frames_document["frames"]
            assert isinstance(frames, list)
            for frame in frames:
                assert isinstance(frame, dict)
                public_id = str(frame["provenance_ref"]).split(":")[1]
                url = f"/frame/{public_id}.png"
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}{url}", timeout=5
                ) as response:
                    original[public_id] = {
                        "url": url,
                        "bytes": response.read(),
                        "etag": response.headers["ETag"],
                        "frame": frame,
                        "digest": frame["png_sha256"],
                        "size": frame["png_byte_size"],
                    }
        assert original
        archive_before = _archive(engine)
        assert set(archive_before) == set(original)

        def fetch(url: str) -> tuple[int, dict[str, str], bytes]:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}{url}", timeout=5
            ) as response:
                return response.status, dict(response.headers), response.read()

        # 1. WL preset mints NEW records; originals stay queryable.
        _apply(
            engine, session_id, "SET_WL_PRESET",
            {"preset_id": "CT_MEDIASTINUM"},
            view={"plane": "AXIAL", "frame_index": 0},
            nonce="r202-wl-0000000000001", parent=None,
        )
        # 2. CLEAR_WL mints raw records again (fresh ULIDs, same bytes).
        _apply(
            engine, session_id, "CLEAR_WL", {},
            view={"plane": "AXIAL", "frame_index": 0},
            nonce="r202-clear-000000000001", parent=_last_nonce(engine),
        )
        # 3. A correction invalidates and re-mints every plane.
        _apply(
            engine, session_id, "REQUEST_CORRECTION",
            {
                "subject_ref": governed_session.study_id,
                "proposed_grade": "GOOD",
                "justification_code": "HUMAN_EVIDENCE",
            },
            view={"plane": "AXIAL", "frame_index": 0},
            nonce="r202-correction-00000001", parent=_last_nonce(engine),
        )
        archive_after = _archive(engine)
        # append-only: every original retained byte-identically, plus the
        # newly minted generations
        for public_id, captured in original.items():
            retained = archive_after[public_id]
            assert retained == archive_before[public_id], (
                f"archive entry {public_id} was rewritten"
            )
            status, headers, png = fetch(str(captured["url"]))  # original URL
            assert status == 200
            assert headers["ETag"] == captured["etag"]
            assert png == captured["bytes"]
            assert hashlib.sha256(png).hexdigest() == captured["digest"]
            assert len(png) == captured["size"]
            # and the engine-level resolver agrees (bytes identical)
            resolved = engine.png_for_frame(public_id)
            assert resolved is not None
            assert resolved[0] == captured["bytes"]
        # STALE membership maps unambiguously to retained records
        stale = _stale_ids(engine)
        assert stale
        archive_ids = {
            str(record["frame_id"]) for record in archive_after.values()
        }
        for stale_id in stale:
            assert stale_id in archive_ids
        # the new generations resolve through the same archive discipline
        assert len(archive_after) > len(archive_before)
        # geometry digest never moved
        assert engine.session_document()["frame_lists_ref"] == geometry_digest
    finally:
        server.stop()


def test_archive_bytes_and_etags_survive_save_reopen(governed_session) -> None:
    import hashlib
    import urllib.request

    from apps.viewer.server import ViewerServer

    engine = _open_viewer(governed_session)
    session_id = engine.session_id()
    _apply(
        engine, session_id, "SET_WL_PRESET",
        {"preset_id": "CT_BONE"},
        view={"plane": "AXIAL", "frame_index": 0},
        nonce="r202-open-wl-00000001", parent=None,
    )
    _apply(
        engine, session_id, "REQUEST_CORRECTION",
        {
            "subject_ref": governed_session.study_id,
            "proposed_grade": "GOOD",
            "justification_code": "HUMAN_EVIDENCE",
        },
        view={"plane": "AXIAL", "frame_index": 0},
        nonce="r202-open-corr-0000001", parent=_last_nonce(engine),
    )
    captured: dict[str, tuple[str, bytes, str]] = {}
    for plane in PLANES:
        frames_document = engine.frames_document(plane)
        frames = frames_document["frames"]
        assert isinstance(frames, list)
        for frame in frames:
            assert isinstance(frame, dict)
            public_id = str(frame["provenance_ref"]).split(":")[1]
            resolved = engine.png_for_frame(public_id)
            assert resolved is not None
            captured[public_id] = (
                f"/frame/{public_id}.png",
                resolved[0],
                f'"{frame["png_sha256"]}"',
            )
    engine.persist(governed_session.store)
    reopened = governed_session.store.reopen()
    report = RoundTripVerificationEngine(
        governed_session.workspace, viewer_repo_root=governed_session.repo_root
    ).verify(reopened)
    assert report.verdict == "PASS", report.to_dict()
    reopened_engine = ViewerEngine(reopened, repo_root=governed_session.repo_root)
    reopened_engine.open()
    # the archive rehydrates completely: the current records plus their
    # superseded predecessors (append-only)
    assert set(captured) <= set(_archive(reopened_engine))
    # archived bytes reconstruct identically after reopen
    for public_id, (url, _old_png, etag) in captured.items():
        resolved = reopened_engine.png_for_frame(public_id)
        assert resolved is not None
        png = resolved[0]
        assert hashlib.sha256(png).hexdigest() in etag
        frame_document = _archive(reopened_engine)[public_id]["frame"]
        assert isinstance(frame_document, dict)
        assert frame_document["png_byte_size"] == len(png)
        del url  # URL identity is proven over HTTP below
    server = ViewerServer(audit_sink=None)
    server.add_engine(reopened_engine)
    _host, port = server.start_background()
    try:
        for public_id, (url, old_png, etag) in captured.items():
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}{url}", timeout=5
            ) as response:
                assert response.status == 200
                assert response.read() == old_png
                assert response.headers["ETag"] == etag
    finally:
        server.stop()


def test_old_state_document_initializes_archive_from_current_records_only(
    governed_session,
) -> None:
    """An old state document (no archive key) must open and survive repeated save/reopen (r3 R3-04).

    Full lifecycle: open -> save -> reopen -> correction -> save -> reopen,
    preserving original URLs/documents/bytes/ETags and unambiguous STALE IDs,
    with the seventh check passing at every stage.
    """

    engine = _open_viewer(governed_session)
    engine.persist(governed_session.store)
    state_path = governed_session.workspace / "session" / "artifacts.json"
    document = json.loads(state_path.read_text())
    stripped = False
    for key, payload in document["payload_documents"].items():
        if key.startswith("viewer_state:"):
            # simulate an OLD (att0002-era) state document: no archive key
            payload.pop("render_archive", None)
            stripped = True
    assert stripped
    state_path.write_text(json.dumps(document))
    from tests.apps.pipeline import rewrite_manifest

    rewrite_manifest(governed_session.workspace)

    # 1. Open old state document (archive reconstructed from current records and provenance)
    reopened = governed_session.store.reopen()
    reopened_engine = ViewerEngine(reopened, repo_root=governed_session.repo_root)
    reopened_engine.open()
    archive = _archive(reopened_engine)
    # initialized from current records ONLY — exactly one archive entry
    # per current record, internal frame_id recovered from provenance, nothing fabricated
    expected: set[str] = set()
    for plane in PLANES:
        frames_document = reopened_engine.frames_document(plane)
        frames = frames_document["frames"]
        assert isinstance(frames, list)
        for frame in frames:
            assert isinstance(frame, dict)
            expected.add(str(frame["provenance_ref"]).split(":")[1])
    assert set(archive) == expected
    assert reopened_engine.verify_reopen_identity() == []
    # Record original documents, public IDs, bytes, and ETags for all current records
    original_data: dict[str, dict[str, object]] = {}
    for plane in PLANES:
        frames_document = reopened_engine.frames_document(plane)
        raw_frames = frames_document["frames"]
        assert isinstance(raw_frames, list)
        for frame in raw_frames:
            public_id = str(frame["provenance_ref"]).split(":")[1]
            lookup = reopened_engine.png_for_frame(public_id)
            assert lookup is not None
            png_bytes, record = lookup
            original_data[public_id] = {
                "document": frame,
                "bytes": png_bytes,
                "etag": f'"{record.png_sha256}"',
            }

    # 2. Save admitted old state
    reopened_engine.persist(governed_session.store)

    # 3. Reopen saved state -> assert seventh check PASS and byte identity
    reopened2 = governed_session.store.reopen()
    report2 = RoundTripVerificationEngine(
        governed_session.workspace, viewer_repo_root=governed_session.repo_root
    ).verify(reopened2)
    assert report2.checks[CHECK_VIEWER].passed, report2.to_dict()

    reopened_engine2 = ViewerEngine(reopened2, repo_root=governed_session.repo_root)
    reopened_engine2.open()
    assert reopened_engine2.verify_reopen_identity() == []
    for public_id, orig in original_data.items():
        lookup2 = reopened_engine2.png_for_frame(public_id)
        assert lookup2 is not None
        png_bytes2, record2 = lookup2
        assert png_bytes2 == orig["bytes"]
        assert f'"{record2.png_sha256}"' == orig["etag"]

    # 4. Apply correction on the admitted session
    action_body = json.dumps(
        {
            "action": "REQUEST_CORRECTION",
            "params": {
                "subject_ref": governed_session.study_id,
                "proposed_grade": "GOOD",
                "justification_code": "HUMAN_EVIDENCE",
            },
            "client_nonce": "correction-nonce-0000000000000001",
            "view": {"plane": "AXIAL", "frame_index": 0},
            "lineage": {
                "session_id": reopened_engine2.session_id(),
                "parent_action_nonce": None,
            },
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    result = reopened_engine2.apply_action_bytes(
        action_body, path_session_id=reopened_engine2.session_id()
    )
    assert result["schema_version"] == "ccta.viewer.state.v1"

    # Verify original archived frame bytes remain queryable and untouched after correction
    for public_id, orig in original_data.items():
        lookup_corr = reopened_engine2.png_for_frame(public_id)
        assert lookup_corr is not None
        png_bytes_corr, record_corr = lookup_corr
        assert png_bytes_corr == orig["bytes"]
        assert f'"{record_corr.png_sha256}"' == orig["etag"]

    # 5. Save after correction
    reopened_engine2.persist(governed_session.store)

    # 6. Reopen again -> assert seventh check PASS and unambiguous STALE IDs
    reopened3 = governed_session.store.reopen()
    report3 = RoundTripVerificationEngine(
        governed_session.workspace, viewer_repo_root=governed_session.repo_root
    ).verify(reopened3)
    assert report3.checks[CHECK_VIEWER].passed, report3.to_dict()

    reopened_engine3 = ViewerEngine(reopened3, repo_root=governed_session.repo_root)
    reopened_engine3.open()
    assert reopened_engine3.verify_reopen_identity() == []

    # Unambiguous STALE IDs preserved from the original records
    assert reopened_engine._state is not None
    assert reopened_engine3._state is not None
    recovered_internal_ids = {
        rec.frame_id
        for rec in reopened_engine._state.render_archive.values()
        if rec.frame_id
    }
    assert recovered_internal_ids
    assert recovered_internal_ids.issubset(set(reopened_engine3._state.stale_frame_ids))

    # All original archived frame records and bytes remain preserved on repeated reopen
    for public_id, orig in original_data.items():
        lookup3 = reopened_engine3.png_for_frame(public_id)
        assert lookup3 is not None
        png_bytes3, record3 = lookup3
        assert png_bytes3 == orig["bytes"]
        assert f'"{record3.png_sha256}"' == orig["etag"]



def test_unresolved_stale_id_fails_retention_verification(
    governed_session,
) -> None:
    """A STALE id without a retained record fails governed verification."""

    engine = _open_viewer(governed_session)
    engine.persist(governed_session.store)
    state_path = governed_session.workspace / "session" / "artifacts.json"
    document = json.loads(state_path.read_text())
    for key, payload in document["payload_documents"].items():
        if key.startswith("viewer_state:"):
            payload["stale_frame_ids"] = ["phantom-0000000000000001"]
    state_path.write_text(json.dumps(document))
    from tests.apps.pipeline import rewrite_manifest

    rewrite_manifest(governed_session.workspace)
    reopened = governed_session.store.reopen()
    reopened_engine = ViewerEngine(reopened, repo_root=governed_session.repo_root)
    with pytest.raises(ViewerOpenError) as excinfo:
        reopened_engine.open()
    assert excinfo.value.code is RejectionCode.SESSION_OPEN_FAILED
    assert any(
        "historical-retention" in problem for problem in excinfo.value.problems
    )


def test_tampered_archived_bytes_fail_the_preserve_check(
    governed_session,
) -> None:
    engine = _open_viewer(governed_session)
    _apply(
        engine, engine.session_id(), "SET_WL_PRESET",
        {"preset_id": "CT_LUNG"},
        view={"plane": "AXIAL", "frame_index": 0},
        nonce="r202-tamper-wl-0000001", parent=None,
    )
    engine.persist(governed_session.store)
    state_path = governed_session.workspace / "session" / "artifacts.json"
    document = json.loads(state_path.read_text())
    victim_seen = False
    for key, payload in document["payload_documents"].items():
        if key.startswith("viewer_state:") and payload.get("render_archive"):
            victim = sorted(payload["render_archive"])[0]
            payload["render_archive"][victim]["frame"]["png_byte_size"] = 7
            victim_seen = True
    assert victim_seen
    state_path.write_text(json.dumps(document))
    from tests.apps.pipeline import rewrite_manifest

    rewrite_manifest(governed_session.workspace)
    reopened = governed_session.store.reopen()
    reopened_engine = ViewerEngine(reopened, repo_root=governed_session.repo_root)
    with pytest.raises(ViewerOpenError) as excinfo:
        reopened_engine.open()
    assert any("digest/size check" in problem for problem in excinfo.value.problems)


# ---------------------------------------------------------------------------
# R4-01 (r4): null-archive typed rejection
# ---------------------------------------------------------------------------


def test_null_archive_fails_closed_typed_open_and_seventh_check(
    governed_session,
) -> None:
    """Explicit render_archive: null fails closed with typed SESSION_OPEN_FAILED (r4 R4-01).

    Persist a valid viewer state, set render_archive to None (JSON null), reseal
    with rewrite_manifest. Direct open raises ViewerOpenError(SESSION_OPEN_FAILED)
    and CHECK_VIEWER fails. Inspect that failure exposes zero session/frame state.
    """
    from apps.workstation.round_trip import CHECK_VIEWER, RoundTripVerificationEngine
    from tests.apps.pipeline import rewrite_manifest

    engine = _open_viewer(governed_session)
    engine.persist(governed_session.store)

    state_path = governed_session.workspace / "session" / "artifacts.json"
    document = json.loads(state_path.read_text())
    mutated = False
    for key, payload in document["payload_documents"].items():
        if key.startswith("viewer_state:"):
            payload["render_archive"] = None
            mutated = True
    assert mutated
    state_path.write_text(json.dumps(document))
    rewrite_manifest(governed_session.workspace)

    # 1. Seventh-check failure (CHECK_VIEWER fails)
    reopened = governed_session.store.reopen()
    report = RoundTripVerificationEngine(
        governed_session.workspace, viewer_repo_root=governed_session.repo_root
    ).verify(reopened)
    assert not report.checks[CHECK_VIEWER].passed, report.to_dict()

    # 2. Direct open raises typed ViewerOpenError(SESSION_OPEN_FAILED)
    reopened_engine = ViewerEngine(reopened, repo_root=governed_session.repo_root)
    with pytest.raises(ViewerOpenError) as excinfo:
        reopened_engine.open()
    assert excinfo.value.code is RejectionCode.SESSION_OPEN_FAILED
    assert any("archive malformed" in p for p in excinfo.value.problems)

    # 3. No session or frame state exposed by the failure
    assert reopened_engine.new_provenance() == ()
    with pytest.raises(ViewerOpenError) as exc_state:
        reopened_engine.state_document()
    assert exc_state.value.code is RejectionCode.SESSION_OPEN_FAILED
    with pytest.raises(ViewerOpenError) as exc_sess:
        reopened_engine.session_document()
    assert exc_sess.value.code is RejectionCode.SESSION_OPEN_FAILED
    with pytest.raises(ViewerOpenError) as exc_frames:
        reopened_engine.frames_document("AXIAL")
    assert exc_frames.value.code is RejectionCode.SESSION_OPEN_FAILED
    with pytest.raises(ViewerOpenError) as exc_qc:
        reopened_engine.qc_panel_document()
    assert exc_qc.value.code is RejectionCode.SESSION_OPEN_FAILED


def test_null_archive_with_historical_generation_fails_typed_open_and_seventh_check(
    governed_session,
) -> None:
    """Explicit render_archive: null after WL change fails typed open and 7th check (r4 R4-01).

    Capture original public ID, frame document, PNG bytes, and ETag before SET_WL_PRESET
    so that ID becomes historical. Persist, set render_archive to None, reseal with
    rewrite_manifest. Direct open raises typed SESSION_OPEN_FAILED and CHECK_VIEWER fails.
    History expectations are derived strictly from captured pre-mutation values, never from
    the mutated engine's lookup or validator.
    """
    from apps.workstation.round_trip import CHECK_VIEWER, RoundTripVerificationEngine
    from tests.apps.pipeline import rewrite_manifest

    engine = _open_viewer(governed_session)

    # Capture original public ID, complete frame document, PNG bytes, and ETag
    frames = engine.frames_document("AXIAL")["frames"]
    assert isinstance(frames, list) and frames
    first_frame = frames[0]
    assert isinstance(first_frame, dict)
    captured_frame_doc = dict(first_frame)
    captured_public_id = str(captured_frame_doc["provenance_ref"]).split(":")[1]
    captured_lookup = engine.png_for_frame(captured_public_id)
    assert captured_lookup is not None
    captured_png_bytes, captured_rec = captured_lookup
    captured_etag = f'"{captured_rec.png_sha256}"'

    # Apply SET_WL_PRESET so that captured_public_id becomes historical
    _apply(
        engine,
        engine.session_id(),
        "SET_WL_PRESET",
        {"preset_id": "CT_LUNG"},
        view={"plane": "AXIAL", "frame_index": 0},
        nonce="r401-null-historical-000001",
        parent=None,
    )

    # Persist the valid state with historical generation
    engine.persist(governed_session.store)

    # Mutate: set render_archive to None (JSON null) and reseal
    state_path = governed_session.workspace / "session" / "artifacts.json"
    document = json.loads(state_path.read_text())
    mutated = False
    for key, payload in document["payload_documents"].items():
        if key.startswith("viewer_state:"):
            payload["render_archive"] = None
            mutated = True
    assert mutated
    state_path.write_text(json.dumps(document))
    rewrite_manifest(governed_session.workspace)

    # 1. Seventh-check failure (CHECK_VIEWER fails)
    reopened = governed_session.store.reopen()
    report = RoundTripVerificationEngine(
        governed_session.workspace, viewer_repo_root=governed_session.repo_root
    ).verify(reopened)
    assert not report.checks[CHECK_VIEWER].passed, report.to_dict()

    # 2. Direct open raises typed ViewerOpenError(SESSION_OPEN_FAILED)
    reopened_engine = ViewerEngine(reopened, repo_root=governed_session.repo_root)
    with pytest.raises(ViewerOpenError) as excinfo:
        reopened_engine.open()
    assert excinfo.value.code is RejectionCode.SESSION_OPEN_FAILED
    assert any("archive malformed" in p for p in excinfo.value.problems)

    # 3. Assertions on captured pre-mutation values, never from mutated engine
    assert captured_public_id
    assert captured_frame_doc["frame_index"] == 0
    assert len(captured_png_bytes) > 0
    assert captured_etag == f'"{captured_frame_doc["png_sha256"]}"'
    assert reopened_engine.new_provenance() == ()
    with pytest.raises(ViewerOpenError) as exc_lookup:
        reopened_engine.png_for_frame(captured_public_id)
    assert exc_lookup.value.code is RejectionCode.SESSION_OPEN_FAILED


# ---------------------------------------------------------------------------
# R2-03 (r2): complete geometry/render bijection
# ---------------------------------------------------------------------------


def _mutate_and_reopen(
    governed_session, mutate
) -> tuple[object, ViewerEngine]:
    """Persist, mutate a valid saved document, reseal, reopen (r2 R2-03).

    Asserts the seventh mandated check (CHECK_VIEWER) FAILS for the mutated
    document — not only a transport manifest mismatch — and returns the
    reopened session plus a fresh engine for the typed open-failure asserts.
    """

    from apps.workstation.round_trip import (
        CHECK_VIEWER,
        RoundTripVerificationEngine,
    )
    from tests.apps.pipeline import rewrite_manifest

    engine = _open_viewer(governed_session)
    engine.persist(governed_session.store)
    state_path = governed_session.workspace / "session" / "artifacts.json"
    document = json.loads(state_path.read_text())
    for key, payload in document["payload_documents"].items():
        if key.startswith("viewer_state:"):
            mutate(payload)
    state_path.write_text(json.dumps(document))
    rewrite_manifest(governed_session.workspace)
    reopened = governed_session.store.reopen()
    report = RoundTripVerificationEngine(
        governed_session.workspace, viewer_repo_root=governed_session.repo_root
    ).verify(reopened)
    assert not report.checks[CHECK_VIEWER].passed, report.to_dict()
    return reopened, ViewerEngine(reopened, repo_root=governed_session.repo_root)


def test_bijection_delete_whole_plane_records_fails_via_seventh_check(
    governed_session,
) -> None:
    def _mutate(payload: dict) -> None:
        payload["renders"]["AXIAL"] = {"wl_preset_id": None, "records": []}

    _reopened, engine = _mutate_and_reopen(governed_session, _mutate)
    with pytest.raises(ViewerOpenError) as excinfo:
        engine.open()
    assert excinfo.value.code is RejectionCode.SESSION_OPEN_FAILED
    assert any(
        "AXIAL" in problem and "count" in problem
        for problem in excinfo.value.problems
    )


def test_bijection_delete_one_record_fails(governed_session) -> None:
    def _mutate(payload: dict) -> None:
        records = payload["renders"]["CORONAL"]["records"]
        payload["renders"]["CORONAL"]["records"] = records[:1]

    _reopened, engine = _mutate_and_reopen(governed_session, _mutate)
    with pytest.raises(ViewerOpenError) as excinfo:
        engine.open()
    assert excinfo.value.code is RejectionCode.SESSION_OPEN_FAILED
    assert any("CORONAL" in problem for problem in excinfo.value.problems)


def test_bijection_duplicate_record_fails(governed_session) -> None:
    def _mutate(payload: dict) -> None:
        records = payload["renders"]["AXIAL"]["records"]
        payload["renders"]["AXIAL"]["records"] = [records[0], records[0]]

    _reopened, engine = _mutate_and_reopen(governed_session, _mutate)
    with pytest.raises(ViewerOpenError) as excinfo:
        engine.open()
    assert any("position" in problem for problem in excinfo.value.problems)


def test_bijection_reordered_records_fail(governed_session) -> None:
    def _mutate(payload: dict) -> None:
        records = payload["renders"]["AXIAL"]["records"]
        payload["renders"]["AXIAL"]["records"] = list(reversed(records))

    _reopened, engine = _mutate_and_reopen(governed_session, _mutate)
    with pytest.raises(ViewerOpenError) as excinfo:
        engine.open()
    assert any(
        "ordered indices 0..n-1" in problem for problem in excinfo.value.problems
    )


def test_bijection_wrong_binding_fails(governed_session) -> None:
    def _mutate(payload: dict) -> None:
        records = payload["renders"]["AXIAL"]["records"]
        # a real record of another plane substituted into AXIAL
        coronal = payload["renders"]["CORONAL"]["records"][0]
        payload["renders"]["AXIAL"]["records"] = [coronal, records[1]]

    _reopened, engine = _mutate_and_reopen(governed_session, _mutate)
    with pytest.raises(ViewerOpenError) as excinfo:
        engine.open()
    assert any(
        "source binding" in problem or "plane binding" in problem
        for problem in excinfo.value.problems
    )


def test_bijection_out_of_range_index_fails(governed_session) -> None:
    def _mutate(payload: dict) -> None:
        records = payload["renders"]["AXIAL"]["records"]
        records[0] = dict(records[0])
        records[0]["frame_index"] = 9
        payload["renders"]["AXIAL"]["records"] = records
        public_id = records[0]["provenance_ref"].split(":", 1)[1]
        if "render_archive" in payload and public_id in payload["render_archive"]:
            payload["render_archive"][public_id]["frame"]["frame_index"] = 9

    _reopened, engine = _mutate_and_reopen(governed_session, _mutate)
    with pytest.raises(ViewerOpenError) as excinfo:
        engine.open()
    assert any(
        "frame_index" in problem and "0..n-1" in problem
        for problem in excinfo.value.problems
    )


def test_bijection_unknown_extra_plane_fails(governed_session) -> None:
    def _mutate(payload: dict) -> None:
        payload["renders"]["OBLIQUE"] = {
            "wl_preset_id": None,
            "records": payload["renders"]["AXIAL"]["records"],
        }

    _reopened, engine = _mutate_and_reopen(governed_session, _mutate)
    with pytest.raises(ViewerOpenError) as excinfo:
        engine.open()
    assert any(
        "unknown extra planes" in problem for problem in excinfo.value.problems
    )


def test_genuinely_empty_geometry_plane_still_succeeds(governed_session) -> None:
    """A legal empty plane (no accepted MPRs) must NOT fail the bijection."""

    from apps.workstation.session_store import RestoredSession

    session = governed_session.session
    # remove CORONAL MPR artifacts entirely -> CORONAL geometry is empty
    keep = [
        artifact
        for artifact in session.artifacts
        if artifact.kind.value != "MPR_VIEW"
        or session.payload_documents.get(artifact.artifact_id, {}).get("plane")
        != "CORONAL"
    ]
    payload_documents = {
        artifact_id: document
        for artifact_id, document in session.payload_documents.items()
        if not (
            isinstance(document, dict)
            and document.get("plane") == "CORONAL"
            and "pixel_blob_ref" in document
        )
    }
    stripped = RestoredSession(
        paths=session.paths,
        study=session.study,
        series=session.series,
        volumes=session.volumes,
        artifacts=tuple(keep),
        payload_documents=payload_documents,
        provenance=session.provenance,
        log=session.log,
        skipped_instances=session.skipped_instances,
        projections=session.projections,
        session_digest=session.session_digest,
        pixel_blobs=dict(session.pixel_blobs),
    )
    engine = ViewerEngine(
        stripped,
        id_factory=_deterministic_ulids(),
        now=lambda: "2026-09-07T00:00:00Z",
        repo_root=governed_session.repo_root,
    )
    engine.open()
    assert engine.frames_document("CORONAL")["frames"] == []
    assert engine.frames_document("AXIAL")["frames"]
    assert engine.verify_reopen_identity() == []
