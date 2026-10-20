"""VIEWER_001 V1/V4: closed schemas and the 7-action grammar negative battery.

The negative battery pins every one of the 16 typed rejection codes, the
pinned precedence statements (zoom-first, matrix range second, lineage
session binding), duplicate-key rejection, canonical-encoding enforcement,
and the schema key-set closure.  Expected codes come from the independent
oracle's transcribed spec table (``tests.apps.viewer_fixtures``).
"""

from __future__ import annotations

import json

import pytest

from apps.viewer.grammar import (
    ACTIONS,
    RejectionCode,
    SessionGrammarView,
    ViewerRejectionError,
    parse_action_body,
    validate_action,
)
from apps.viewer.schemas import (
    PLANES,
    ViewerSchemaError,
    frame_lists_digest,
    validate_frame_document,
    validate_session_document,
)
from tests.apps.viewer_fixtures import (
    ACTION_SET,
    PRESET_TABLE,
    REJECTION_CODES,
)

ULID = "01ARZ3NDEKTSV4RRFFQ69G5FAV"
SESSION_ID = ULID
STUDY_ID = f"study_id:{ULID}"


def _default_presets() -> tuple[str, ...]:
    return tuple(preset_id for preset_id, _center, _width in PRESET_TABLE)


def _view(
    *,
    counts: dict[str, int] | None = None,
    used: tuple[str, ...] = (),
    accepted: tuple[str, ...] = (),
    presets: tuple[str, ...] = _default_presets(),
) -> SessionGrammarView:
    return SessionGrammarView(
        session_id=SESSION_ID,
        study_id=STUDY_ID,
        frame_count_by_plane=counts or {"AXIAL": 3, "CORONAL": 2, "SAGITTAL": 0},
        wl_presets_active=presets,
        used_nonces=frozenset(used),
        accepted_nonces=frozenset(accepted),
    )


def _body(
    *,
    action: str = "SET_SLICE",
    params: dict[str, object] | None = None,
    client_nonce: str = "abcdefgh12345678",
    view: dict[str, object] | None = None,
    session_id: str = SESSION_ID,
    parent: str | None = None,
) -> bytes:
    document = {
        "action": action,
        "params": params if params is not None else {"plane": "AXIAL", "frame_index": 1},
        "client_nonce": client_nonce,
        "view": view if view is not None else {"plane": "AXIAL", "frame_index": 1},
        "lineage": {"session_id": session_id, "parent_action_nonce": parent},
    }
    return json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")


def test_action_set_is_exactly_the_closed_ten() -> None:
    assert ACTIONS == ACTION_SET
    assert len(ACTIONS) == 11


def test_rejection_vocabulary_is_exactly_sixteen() -> None:
    assert [code.value for code in RejectionCode] == list(REJECTION_CODES)


def test_valid_action_parses() -> None:
    parsed = validate_action(_body(), path_session_id=SESSION_ID, session=_view())
    assert parsed.action == "SET_SLICE"
    assert parsed.view_plane == "AXIAL"
    assert parsed.view_frame_index == 1
    assert parsed.session_id == SESSION_ID
    assert parsed.parent_action_nonce is None


# ---------------------------------------------------------------------------
# Envelope-level rejections
# ---------------------------------------------------------------------------


def test_duplicate_json_keys_rejected_at_any_depth() -> None:
    body = (
        b'{"action":"SET_SLICE","action":"SET_PLANE","params":{},'
        b'"client_nonce":"abcdefgh12345678","view":{"plane":"AXIAL",'
        b'"frame_index":0},"lineage":{"session_id":"'
        + SESSION_ID.encode()
        + b'","parent_action_nonce":null}}'
    )
    with pytest.raises(ViewerRejectionError) as excinfo:
        parse_action_body(body)
    assert excinfo.value.code is RejectionCode.DUPLICATE_JSON_KEYS
    # nested duplicates also fail at parse time
    nested = (
        b'{"action":"SET_SLICE","params":{},"client_nonce":"abcdefgh12345678",'
        b'"view":{"plane":"AXIAL","plane":"CORONAL","frame_index":0},'
        b'"lineage":{"session_id":"'
        + SESSION_ID.encode()
        + b'","parent_action_nonce":null}}'
    )
    with pytest.raises(ViewerRejectionError) as excinfo:
        parse_action_body(nested)
    assert excinfo.value.code is RejectionCode.DUPLICATE_JSON_KEYS


def test_non_canonical_encoding_rejected() -> None:
    # whitespace is not canonical
    with pytest.raises(ViewerRejectionError) as excinfo:
        parse_action_body(b"{\n  \"action\": \"SET_SLICE\",\n \"params\": {}\n}")
    assert excinfo.value.code is RejectionCode.NON_CANONICAL_ENCODING
    # unsorted keys are not canonical (same content, different byte order)
    unsorted = (
        b'{"client_nonce":"abcdefgh12345678","action":"SET_SLICE","params":{},'
        b'"view":{"plane":"AXIAL","frame_index":1},"lineage":{"session_id":"'
        + SESSION_ID.encode()
        + b'","parent_action_nonce":null}}'
    )
    with pytest.raises(ViewerRejectionError) as excinfo:
        parse_action_body(unsorted)
    assert excinfo.value.code is RejectionCode.NON_CANONICAL_ENCODING
    # NaN/Infinity constants are rejected outright
    body = _body()
    for constant in (b"NaN", b"Infinity", b"-Infinity"):
        poisoned = body.replace(b"\"frame_index\":1", b"\"frame_index\":" + constant)
        with pytest.raises(ViewerRejectionError) as excinfo:
            parse_action_body(poisoned)
        assert excinfo.value.code is RejectionCode.NON_CANONICAL_ENCODING


def test_unknown_action_rejected() -> None:
    with pytest.raises(ViewerRejectionError) as excinfo:
        validate_action(
            _body(action="SET_WINDOW_CUSTOM", params={}),
            path_session_id=SESSION_ID,
            session=_view(),
        )
    assert excinfo.value.code is RejectionCode.UNKNOWN_ACTION


def test_unknown_param_rejected_for_extra_or_missing_keys() -> None:
    # an extra param key breaks the closed param set
    with pytest.raises(ViewerRejectionError) as excinfo:
        validate_action(
            _body(action="SET_SLICE", params={"plane": "AXIAL", "frame_index": 1, "extra": 1}),
            path_session_id=SESSION_ID,
            session=_view(),
        )
    assert excinfo.value.code is RejectionCode.UNKNOWN_PARAM
    # a missing param key too
    with pytest.raises(ViewerRejectionError) as excinfo:
        validate_action(
            _body(action="SET_SLICE", params={"plane": "AXIAL"}),
            path_session_id=SESSION_ID,
            session=_view(),
        )
    assert excinfo.value.code is RejectionCode.UNKNOWN_PARAM
    # the exactly-closed CLEAR_WL params pass the param gate
    parsed = validate_action(
        _body(action="CLEAR_WL", params={}),
        path_session_id=SESSION_ID,
        session=_view(),
    )
    assert parsed.action == "CLEAR_WL"


def test_envelope_extra_key_rejected() -> None:
    document = {
        "action": "SET_SLICE",
        "params": {"plane": "AXIAL", "frame_index": 1},
        "client_nonce": "abcdefgh12345678",
        "view": {"plane": "AXIAL", "frame_index": 1},
        "lineage": {"session_id": SESSION_ID, "parent_action_nonce": None},
        "extra": 1,
    }
    body = json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")
    with pytest.raises(ViewerRejectionError) as excinfo:
        validate_action(body, path_session_id=SESSION_ID, session=_view())
    assert excinfo.value.code is RejectionCode.UNKNOWN_PARAM


def test_malformed_client_nonce_rejected_as_unknown_param() -> None:
    for nonce in ("short", "x" * 65, "bad space", "bad/slash", "bad.dot"):
        with pytest.raises(ViewerRejectionError) as excinfo:
            validate_action(
                _body(client_nonce=nonce),
                path_session_id=SESSION_ID,
                session=_view(),
            )
        assert excinfo.value.code is RejectionCode.UNKNOWN_PARAM


def test_nonce_grammar_bounds_are_exact() -> None:
    # exactly 16 and exactly 64 legal chars are accepted at the grammar layer
    for nonce in ("a" * 16, "A-b_9" * 12 + "abcd"):
        assert len(nonce) in (16, 64)
        parsed = validate_action(
            _body(client_nonce=nonce),
            path_session_id=SESSION_ID,
            session=_view(),
        )
        assert parsed.client_nonce == nonce


# ---------------------------------------------------------------------------
# Pinned precedence: zoom first, then matrix range; lineage session binding
# ---------------------------------------------------------------------------


def test_zoom_non_positive_checked_first_before_range_and_view() -> None:
    # zoom = 0 with an out-of-range view and a reused nonce: zoom wins.
    with pytest.raises(ViewerRejectionError) as excinfo:
        validate_action(
            _body(
                action="SET_VIEWPORT_MATRIX",
                params={"zoom": 0, "pan_x": 0.0, "pan_y": 0.0},
                view={"plane": "AXIAL", "frame_index": 99},
                client_nonce="abcdefgh12345678",
            ),
            path_session_id=SESSION_ID,
            session=_view(used=("abcdefgh12345678",)),
        )
    assert excinfo.value.code is RejectionCode.ZOOM_NON_POSITIVE
    with pytest.raises(ViewerRejectionError) as excinfo:
        validate_action(
            _body(
                action="SET_VIEWPORT_MATRIX",
                params={"zoom": -1.5, "pan_x": 0.0, "pan_y": 0.0},
            ),
            path_session_id=SESSION_ID,
            session=_view(),
        )
    assert excinfo.value.code is RejectionCode.ZOOM_NON_POSITIVE


def test_matrix_range_after_zoom_positivity() -> None:
    matrix_cases: tuple[tuple[dict[str, object], str], ...] = (
        ({"zoom": 64.5, "pan_x": 0.0, "pan_y": 0.0}, "range"),
        ({"zoom": 2.0, "pan_x": 2.5, "pan_y": 0.0}, "range"),
        ({"zoom": 2.0, "pan_x": 0.0, "pan_y": -2.1}, "range"),
        ({"zoom": 1.0, "pan_x": 0.0, "pan_y": float("inf")}, "canonical"),
    )
    for params, kind in matrix_cases:
        body = _body(action="SET_VIEWPORT_MATRIX", params=params)
        if kind == "canonical":
            # inf pan encodes as Infinity -> NON_CANONICAL at parse time
            with pytest.raises(ViewerRejectionError) as excinfo:
                validate_action(body, path_session_id=SESSION_ID, session=_view())
            assert excinfo.value.code is RejectionCode.NON_CANONICAL_ENCODING
        else:
            with pytest.raises(ViewerRejectionError) as excinfo:
                validate_action(body, path_session_id=SESSION_ID, session=_view())
            assert excinfo.value.code is RejectionCode.MATRIX_NON_FINITE


def test_valid_matrix_boundaries_accepted() -> None:
    parsed = validate_action(
        _body(
            action="SET_VIEWPORT_MATRIX",
            params={"zoom": 64, "pan_x": 2.0, "pan_y": -2.0},
        ),
        path_session_id=SESSION_ID,
        session=_view(),
    )
    assert parsed.action == "SET_VIEWPORT_MATRIX"


def test_lineage_session_binding_is_session_not_found() -> None:
    with pytest.raises(ViewerRejectionError) as excinfo:
        validate_action(
            _body(session_id="01ARZ3NDEKTSV4RRFFQ69G5FAW"),
            path_session_id=SESSION_ID,
            session=_view(),
        )
    assert excinfo.value.code is RejectionCode.SESSION_NOT_FOUND
    assert excinfo.value.http_status == 404


def test_path_session_binding_is_session_not_found() -> None:
    with pytest.raises(ViewerRejectionError) as excinfo:
        validate_action(_body(), path_session_id="01ARZ3NDEKTSV4RRFFQ69G5FAW", session=_view())
    assert excinfo.value.code is RejectionCode.SESSION_NOT_FOUND


def test_rejection_payload_has_no_free_form_extra_keys() -> None:
    try:
        validate_action(_body(action="NOPE"), path_session_id=SESSION_ID, session=_view())
        raise AssertionError("expected rejection")
    except ViewerRejectionError as rejection:
        payload = rejection.payload
        error = payload["error"]
        assert isinstance(error, dict)
        assert set(payload) == {"error"}
        assert set(error) == {"code", "message"}


# ---------------------------------------------------------------------------
# View / nonce / lineage
# ---------------------------------------------------------------------------


def test_view_mismatch_rejected() -> None:
    with pytest.raises(ViewerRejectionError) as excinfo:
        validate_action(
            _body(view={"plane": "AXIAL", "frame_index": 99}),
            path_session_id=SESSION_ID,
            session=_view(),
        )
    assert excinfo.value.code is RejectionCode.VIEW_MISMATCH
    # empty plane list: any view is a mismatch (explicit empty-state)
    with pytest.raises(ViewerRejectionError) as excinfo:
        validate_action(
            _body(view={"plane": "SAGITTAL", "frame_index": 0}),
            path_session_id=SESSION_ID,
            session=_view(),
        )
    assert excinfo.value.code is RejectionCode.VIEW_MISMATCH


def test_nonce_reuse_rejected_even_after_a_rejected_attempt() -> None:
    with pytest.raises(ViewerRejectionError) as excinfo:
        validate_action(
            _body(),
            path_session_id=SESSION_ID,
            session=_view(used=("abcdefgh12345678",)),
        )
    assert excinfo.value.code is RejectionCode.NONCE_REUSE


def test_lineage_gap_rejected() -> None:
    # first action with a parent
    with pytest.raises(ViewerRejectionError) as excinfo:
        validate_action(
            _body(parent="abcdefgh12345678"), path_session_id=SESSION_ID, session=_view()
        )
    assert excinfo.value.code is RejectionCode.LINEAGE_GAP
    # later action without a parent
    with pytest.raises(ViewerRejectionError) as excinfo:
        validate_action(
            _body(parent=None),
            path_session_id=SESSION_ID,
            session=_view(used=("n0nce00000000001",), accepted=("n0nce00000000001",)),
        )
    assert excinfo.value.code is RejectionCode.LINEAGE_GAP
    # later action whose parent was never accepted
    with pytest.raises(ViewerRejectionError) as excinfo:
        validate_action(
            _body(parent="parent00000000000001"),
            path_session_id=SESSION_ID,
            session=_view(used=("n0nce00000000001",), accepted=("n0nce00000000001",)),
        )
    assert excinfo.value.code is RejectionCode.LINEAGE_GAP


def test_lineage_chain_through_any_accepted_nonce_is_legal() -> None:
    parsed = validate_action(
        _body(parent="n0nce00000000001"),
        path_session_id=SESSION_ID,
        session=_view(used=("n0nce00000000001",), accepted=("n0nce00000000001",)),
    )
    assert parsed.parent_action_nonce == "n0nce00000000001"


# ---------------------------------------------------------------------------
# Action-specific semantic gates
# ---------------------------------------------------------------------------


def test_index_out_of_range_rejected() -> None:
    with pytest.raises(ViewerRejectionError) as excinfo:
        validate_action(
            _body(params={"plane": "AXIAL", "frame_index": 3}),
            path_session_id=SESSION_ID,
            session=_view(),
        )
    assert excinfo.value.code is RejectionCode.INDEX_OUT_OF_RANGE
    # scrolling an empty plane list is INDEX_OUT_OF_RANGE (explicit empty-state)
    with pytest.raises(ViewerRejectionError) as excinfo:
        validate_action(
            _body(params={"plane": "SAGITTAL", "frame_index": 0}),
            path_session_id=SESSION_ID,
            session=_view(),
        )
    assert excinfo.value.code is RejectionCode.INDEX_OUT_OF_RANGE


def test_unknown_preset_rejected() -> None:
    with pytest.raises(ViewerRejectionError) as excinfo:
        validate_action(
            _body(action="SET_WL_PRESET", params={"preset_id": "CT_WINDOW_CUSTOM"}),
            path_session_id=SESSION_ID,
            session=_view(),
        )
    assert excinfo.value.code is RejectionCode.UNKNOWN_PRESET


def test_grade_not_in_enum_rejected() -> None:
    for grade in ("EXCELLENT", "good", "", "PROCEED"):
        with pytest.raises(ViewerRejectionError) as excinfo:
            validate_action(
                _body(
                    action="REQUEST_CORRECTION",
                    params={
                        "subject_ref": STUDY_ID,
                        "proposed_grade": grade,
                        "justification_code": "HUMAN_EVIDENCE",
                    },
                ),
                path_session_id=SESSION_ID,
                session=_view(),
            )
        assert excinfo.value.code is RejectionCode.GRADE_NOT_IN_ENUM


def test_justification_not_in_enum_rejected() -> None:
    for justification in (
        "PLACEHOLDER_NO_QC_EVIDENCE",
        "RULE_ENGINE_EVIDENCE",
        "MODEL_EVIDENCE",
        "free text from the physician",
    ):
        with pytest.raises(ViewerRejectionError) as excinfo:
            validate_action(
                _body(
                    action="REQUEST_CORRECTION",
                    params={
                        "subject_ref": STUDY_ID,
                        "proposed_grade": "GOOD",
                        "justification_code": justification,
                    },
                ),
                path_session_id=SESSION_ID,
                session=_view(),
            )
        assert excinfo.value.code is RejectionCode.JUSTIFICATION_NOT_IN_ENUM


def test_subject_ref_invalid_rejected() -> None:
    for subject in (
        f"mpr_artifact_id:{ULID}",
        "study_id:01ARZ3NDEKTSV4RRFFQ69G5FAW",  # a different study
        f"volume_id:{ULID}",
        "study_id:NOTAULID",
        "free text",
    ):
        with pytest.raises(ViewerRejectionError) as excinfo:
            validate_action(
                _body(
                    action="REQUEST_CORRECTION",
                    params={
                        "subject_ref": subject,
                        "proposed_grade": "GOOD",
                        "justification_code": "HUMAN_EVIDENCE",
                    },
                ),
                path_session_id=SESSION_ID,
                session=_view(),
            )
        assert excinfo.value.code is RejectionCode.SUBJECT_REF_INVALID


def test_valid_correction_grammar_passes() -> None:
    parsed = validate_action(
        _body(
            action="REQUEST_CORRECTION",
            params={
                "subject_ref": STUDY_ID,
                "proposed_grade": "NOT_ANALYZABLE",
                "justification_code": "HUMAN_EVIDENCE",
            },
        ),
        path_session_id=SESSION_ID,
        session=_view(),
    )
    assert parsed.action == "REQUEST_CORRECTION"


def _decision_params(**overrides: object) -> dict[str, object]:
    params: dict[str, object] = {
        "subject_ref": STUDY_ID,
        "centerline_ref": f"centerline_artifact_id:{ULID}",
        "decision": "ACCEPT",
        "seed_voxel": None,
        "seed_direction_world": None,
    }
    params.update(overrides)
    return params


def test_centerline_decision_is_a_closed_grammar_extension() -> None:
    """att0002 F5: the decision action is closed; EDIT carries a typed seed."""

    accept = validate_action(
        _body(action="REQUEST_CENTERLINE_DECISION", params=_decision_params()),
        path_session_id=SESSION_ID,
        session=_view(),
    )
    assert accept.action == "REQUEST_CENTERLINE_DECISION"
    assert accept.params["decision"] == "ACCEPT"
    for decision in ("ACCEPT", "REJECT"):
        parsed = validate_action(
            _body(
                action="REQUEST_CENTERLINE_DECISION",
                params=_decision_params(decision=decision),
            ),
            path_session_id=SESSION_ID,
            session=_view(),
        )
        assert parsed.params["decision"] == decision
    edit = validate_action(
        _body(
            action="REQUEST_CENTERLINE_DECISION",
            params=_decision_params(
                decision="EDIT",
                seed_voxel=[3, 4, 5],
                seed_direction_world=[0.0, 0.0, 1.0],
            ),
        ),
        path_session_id=SESSION_ID,
        session=_view(),
    )
    assert edit.params["seed_voxel"] == [3, 4, 5]


def test_centerline_decision_rejects_open_domains() -> None:
    cases: tuple[tuple[dict[str, object], RejectionCode], ...] = (
        (_decision_params(decision="APPROVE"), RejectionCode.UNKNOWN_PARAM),
        (_decision_params(decision="EDIT"), RejectionCode.UNKNOWN_PARAM),
        (
            _decision_params(
                decision="EDIT",
                seed_voxel=[1, 2],
                seed_direction_world=[0.0, 0.0, 1.0],
            ),
            RejectionCode.UNKNOWN_PARAM,
        ),
        (
            _decision_params(
                decision="EDIT",
                seed_voxel=[1, 2, -3],
                seed_direction_world=[0.0, 0.0, 1.0],
            ),
            RejectionCode.UNKNOWN_PARAM,
        ),
        (
            _decision_params(
                decision="EDIT",
                seed_voxel=[1, 2, 3],
                seed_direction_world=[0.0, 0.0, 0.0],
            ),
            RejectionCode.UNKNOWN_PARAM,
        ),
        (
            _decision_params(decision="ACCEPT", seed_voxel=[1, 2, 3]),
            RejectionCode.UNKNOWN_PARAM,
        ),
        (
            _decision_params(centerline_ref=f"mpr_artifact_id:{ULID}"),
            RejectionCode.UNKNOWN_PARAM,
        ),
        (
            _decision_params(centerline_ref="centerline_artifact_id:NOT-A-ULID"),
            RejectionCode.UNKNOWN_PARAM,
        ),
        (_decision_params(subject_ref="free text"), RejectionCode.SUBJECT_REF_INVALID),
        (
            _decision_params(subject_ref=f"study_id:{'A' * 26}"),
            RejectionCode.SUBJECT_REF_INVALID,
        ),
    )
    for params, expected in cases:
        with pytest.raises(ViewerRejectionError) as excinfo:
            validate_action(
                _body(action="REQUEST_CENTERLINE_DECISION", params=params),
                path_session_id=SESSION_ID,
                session=_view(),
            )
        assert excinfo.value.code is expected, params


# ---------------------------------------------------------------------------
# Closed schemas (V1)
# ---------------------------------------------------------------------------


def _frame_document() -> dict[str, object]:
    return {
        "schema_version": "ccta.viewer.frame.v1",
        "session_id": ULID,
        "plane": "AXIAL",
        "frame_index": 0,
        "source_mpr_ref": f"mpr_artifact_id:{ULID}",
        "wl_preset_id": None,
        "wl_applied": {"center": 815.5, "width": 1627.0},
        "png_sha256": "a" * 64,
        "png_byte_size": 220,
        "provenance_ref": f"provenance:{ULID}",
    }


def _session_document() -> dict[str, object]:
    return {
        "schema_version": "ccta.viewer.session.v1",
        "session_id": ULID,
        "study_id": STUDY_ID,
        "planes": list(PLANES),
        "frame_lists_ref": "b" * 64,
        "qc_gate_state_ref": f"gate:{ULID}",
        "wl_presets_active": [row[0] for row in PRESET_TABLE],
        "opened_at": "2026-09-07T00:00:00Z",
        "provenance_ref": f"provenance:{ULID}",
    }


def test_frame_schema_is_exactly_ten_keys() -> None:
    record = validate_frame_document(_frame_document())
    assert set(record.to_document()) == {
        "schema_version",
        "session_id",
        "plane",
        "frame_index",
        "source_mpr_ref",
        "wl_preset_id",
        "wl_applied",
        "png_sha256",
        "png_byte_size",
        "provenance_ref",
    }
    assert record.identity_tuple == (
        ULID,
        "AXIAL",
        0,
        f"mpr_artifact_id:{ULID}",
        None,
        {"center": 815.5, "width": 1627.0},
    )


def test_frame_schema_rejects_missing_unknown_and_malformed() -> None:
    document = _frame_document()
    with pytest.raises(ViewerSchemaError):
        validate_frame_document({k: v for k, v in document.items() if k != "wl_applied"})
    extra = dict(document)
    extra["extra_key"] = 1
    with pytest.raises(ViewerSchemaError):
        validate_frame_document(extra)
    for mutate in (
        lambda d: d.update({"plane": "OBLIQUE"}),
        lambda d: d.update({"frame_index": -1}),
        lambda d: d.update({"wl_applied": None}),
        lambda d: d.update({"wl_applied": {"center": 40}}),  # int, not 40.0
        lambda d: d.update({"wl_applied": {"center": 40.0}}),  # missing width
        lambda d: d.update({"png_sha256": "A" * 64}),
        lambda d: d.update({"png_byte_size": 0}),
        lambda d: d.update({"source_mpr_ref": f"volume_id:{ULID}"}),
    ):
        broken = dict(document)
        mutate(broken)
        with pytest.raises(ViewerSchemaError):
            validate_frame_document(broken)


def test_session_schema_is_exactly_nine_keys() -> None:
    record = validate_session_document(_session_document())
    assert set(record.to_document()) == {
        "schema_version",
        "session_id",
        "study_id",
        "planes",
        "frame_lists_ref",
        "qc_gate_state_ref",
        "wl_presets_active",
        "opened_at",
        "provenance_ref",
    }
    with pytest.raises(ViewerSchemaError):
        validate_session_document(
            {k: v for k, v in _session_document().items() if k != "frame_lists_ref"}
        )
    extra = dict(_session_document())
    extra["drift"] = True
    with pytest.raises(ViewerSchemaError):
        validate_session_document(extra)
    broken = dict(_session_document())
    broken["planes"] = ["AXIAL", "CORONAL"]
    with pytest.raises(ViewerSchemaError):
        validate_session_document(broken)
    broken = dict(_session_document())
    broken["qc_gate_state_ref"] = f"qc_artifact:{ULID}"
    with pytest.raises(ViewerSchemaError):
        validate_session_document(broken)


def test_wl_applied_int_form_is_rejected_in_digest_inputs() -> None:
    # The biconditional: wl_applied ALWAYS present with canonical floats
    # (40.0 never 40).
    document = _frame_document()
    document["wl_applied"] = {"center": 40, "width": 400}
    with pytest.raises(ViewerSchemaError):
        validate_frame_document(document)


def test_frame_lists_digest_is_order_and_content_sensitive() -> None:
    bindings = {"AXIAL": [(0, "A" * 26), (1, "B" * 26)], "CORONAL": [], "SAGITTAL": []}
    digest = frame_lists_digest(ULID, bindings)
    assert digest == frame_lists_digest(ULID, dict(bindings))
    flipped = {"AXIAL": [(0, "B" * 26), (1, "A" * 26)], "CORONAL": [], "SAGITTAL": []}
    assert digest != frame_lists_digest(ULID, flipped)
    assert digest != frame_lists_digest("0" * 26, bindings)
    # the element projection carries session_id, plane, index, and mpr ref
    from apps.viewer.schemas import geometry_projection  # noqa: PLC0415

    projection = geometry_projection(ULID, bindings)
    assert projection["AXIAL"] == [
        [ULID, "AXIAL", 0, f"mpr_artifact_id:{'A' * 26}"],
        [ULID, "AXIAL", 1, f"mpr_artifact_id:{'B' * 26}"],
    ]
    assert projection["CORONAL"] == [] and projection["SAGITTAL"] == []
