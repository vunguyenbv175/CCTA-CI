"""Interaction grammar (VIEWER_001_SPEC Parts 3 + 3.1; ADAPT of 008B).

All client mutations are action POSTs with the strict envelope ``{"action":
str, "params": object, "client_nonce": str, "view": object, "lineage":
object}`` — no duplicate keys (rejected), canonical encoding only (byte rule
per ``core.digest.canonical_json`` with sort_keys=true, separators=(',',':'),
ensure_ascii=False).

The v1 closed action set is exactly: ``SET_SLICE``, ``SET_PLANE``,
``SET_WL_PRESET``, ``CLEAR_WL``, ``SET_VIEWPORT_MATRIX``, ``OPEN_QC_PANEL``,
``REQUEST_CORRECTION``.  The v2 successor (VIEWER_001_SPEC Part 4.2) adds
exactly ONE action alongside them: ``SET_WL_CUSTOM`` with the closed param
keys ``("center", "width")``.  Its refusal vocabulary is the EXISTING typed
family — both numerics must be finite binary32 values and ``width`` must be
a positive binary32 value; every violation is refused as ``UNKNOWN_PARAM``
(the frozen 16-code vocabulary is unchanged, no new code is minted), and the
server never clamps (Rule 10).

The CORONARY_CENTERLINE_001 addition is exactly ONE further action,
``SET_OVERLAY_VISIBILITY`` (the one overlay-visibility toggle of the
single-path coronary overlay): params ``("overlay_id", "visible")`` with
``overlay_id`` in the closed allowlist ``("CENTERLINE",)`` and a strict
boolean ``visible``.  CORONARY_REVIEW_VIEW_V1 (Decision A) extends this ONE
action with exactly ONE further closed key, ``mode`` in
``("SLICE_LOCAL", "PROJECTION")`` (default ``SLICE_LOCAL``; unknown modes
refuse as ``UNKNOWN_PARAM``).  Every violation is refused as
``UNKNOWN_PARAM`` — the frozen 16-code vocabulary still gains nothing.

The CORONARY_CENTERLINE_001 att0002 correction adds exactly ONE closed
action, ``REQUEST_CENTERLINE_DECISION``: the single-path human seam
(ACCEPT / EDIT re-seed / REJECT of §6/§7) cannot be expressed by
``REQUEST_CORRECTION`` (whose closed params are the QC assessment triples),
so the decision rides its own closed action: params ``("subject_ref",
"centerline_ref", "decision", "seed_voxel", "seed_direction_world")`` with
``decision`` in the closed ``("ACCEPT", "EDIT", "REJECT")`` vocabulary, the
seed fields REQUIRED for EDIT and exactly ``null`` otherwise, and every
violation refused as ``UNKNOWN_PARAM``/``SUBJECT_REF_INVALID`` — the frozen
16-code vocabulary still gains nothing and the existing QC route keeps its
semantics byte-identical.

Typed rejection codes (HTTP 400 unless noted) with the PINNED precedence:

1. ``DUPLICATE_JSON_KEYS``     — parse level (any nesting depth).
2. ``NON_CANONICAL_ENCODING``  — request bytes are not the canonical form.
3. ``SESSION_NOT_FOUND`` (404) — the path session is unknown, or
   ``lineage.session_id`` is not exactly the session id in the request path.
4. ``UNKNOWN_ACTION``          — action outside the closed action set.
5. ``UNKNOWN_PARAM``           — envelope key set, param key set, or view/
   lineage key set not exactly closed; malformed ``client_nonce``; wrong
   value types.
6. ``ZOOM_NON_POSITIVE``       — SET_VIEWPORT_MATRIX: ``zoom <= 0`` or
   non-finite zoom, checked FIRST (spec-pinned).
7. ``MATRIX_NON_FINITE``       — then range violations: ``zoom > 64``, pan
   outside +-2 viewport extents, any non-finite pan (spec-pinned).  The
   server NEVER clamps silently (Rule 10).
8. ``VIEW_MISMATCH``           — the view does not resolve to a valid frame
   of that plane.
9. ``NONCE_REUSE``             — the client_nonce was already seen in this
   session's lifetime (accepted or rejected attempts both consume).
10. ``LINEAGE_GAP``            — parent_action_nonce is not None on the
    first action, or does not reference an accepted action nonce.
11. ``INDEX_OUT_OF_RANGE``     — SET_SLICE frame_index outside the closed
    plane list (also the empty-plane scroll).
12. ``UNKNOWN_PRESET``         — preset_id outside the session's active set.
13. ``GRADE_NOT_IN_ENUM``      — proposed_grade outside the QualityGrade enum.
14. ``JUSTIFICATION_NOT_IN_ENUM`` — justification_code is not exactly
    HUMAN_EVIDENCE (PLACEHOLDER_NO_QC_EVIDENCE, RULE_ENGINE_EVIDENCE and
    MODEL_EVIDENCE are rejected here).
15. ``SUBJECT_REF_INVALID``    — subject_ref is not exactly
    ``study_id:<ULID>`` or does not bind THIS session's study.
16. ``SESSION_OPEN_FAILED`` (409) — the Part 6 session-open table (raised at
    open, surfaced through the same payload shape).

The rejection payload is ``{"error": {"code": <code>, "message": str}}`` with
no free-form extra keys.  Wherever ``wl_applied`` numbers enter a digest
input they serialize via canonical_json float form (``40.0``, never ``40``).
"""

from __future__ import annotations

import enum
import json
import math
import re
import struct
from dataclasses import dataclass
from typing import Final

from apps.viewer.overlay import OVERLAY_IDS, OVERLAY_MODES
from apps.viewer.schemas import PLANES
from core.digest import canonical_json
from core.qc.gate import V2_VESSEL_IDS as VESSEL_LABEL_TOKENS

ACTIONS: Final[tuple[str, ...]] = (
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

#: The closed per-action param key sets (exactly these keys, no extras).
ACTION_PARAM_KEYS: Final[dict[str, tuple[str, ...]]] = {
    "SET_SLICE": ("plane", "frame_index"),
    "SET_PLANE": ("plane",),
    "SET_WL_PRESET": ("preset_id",),
    "SET_WL_CUSTOM": ("center", "width"),
    "CLEAR_WL": (),
    "SET_VIEWPORT_MATRIX": ("zoom", "pan_x", "pan_y"),
    "OPEN_QC_PANEL": (),
    "REQUEST_CORRECTION": ("subject_ref", "proposed_grade", "justification_code"),
    "SET_OVERLAY_VISIBILITY": ("overlay_id", "visible", "mode"),
    "REQUEST_CENTERLINE_DECISION": (
        "subject_ref",
        "centerline_ref",
        "decision",
        "seed_voxel",
        "seed_direction_world",
    ),
    "SELECT_VESSEL": ("vessel_label",),
}

ENVELOPE_KEYS: Final[frozenset[str]] = frozenset(
    {"action", "params", "client_nonce", "view", "lineage"}
)

NONCE_RE: Final[re.Pattern[str]] = re.compile(r"\A[A-Za-z0-9_-]{16,64}\Z")
QUALITY_GRADES: Final[tuple[str, ...]] = (
    "GOOD",
    "ACCEPTABLE",
    "LIMITED",
    "NOT_ANALYZABLE",
)
#: The closed human decision vocabulary of the single-path coronary seam.
CENTERLINE_DECISIONS: Final[tuple[str, ...]] = ("ACCEPT", "EDIT", "REJECT")
JUSTIFICATION_HUMAN_EVIDENCE: Final[str] = "HUMAN_EVIDENCE"
ZOOM_MAX: Final[float] = 64.0
PAN_EXTENT_LIMIT: Final[float] = 2.0
#: The largest finite IEEE-754 binary32 magnitude.  Custom window/level
#: numerics are validated against it so the engine's single binary32 cast
#: (the frozen transfer's only rounding primitive) can never overflow.
FLOAT32_MAX: Final[float] = 3.4028234663852886e38

_STUDY_REF_RE: Final[re.Pattern[str]] = re.compile(r"\Astudy_id:[A-Z0-9]{26}\Z")
_MPR_REF_RE: Final[re.Pattern[str]] = re.compile(r"\Ampr_artifact_id:[A-Z0-9]{26}\Z")
_CENTERLINE_REF_RE: Final[re.Pattern[str]] = re.compile(
    r"\Acenterline_artifact_id:[A-Z0-9]{26}\Z"
)


class RejectionCode(enum.StrEnum):
    """The 16 closed typed rejection codes (Part 3.1)."""

    DUPLICATE_JSON_KEYS = "DUPLICATE_JSON_KEYS"
    NON_CANONICAL_ENCODING = "NON_CANONICAL_ENCODING"
    UNKNOWN_ACTION = "UNKNOWN_ACTION"
    UNKNOWN_PARAM = "UNKNOWN_PARAM"
    VIEW_MISMATCH = "VIEW_MISMATCH"
    NONCE_REUSE = "NONCE_REUSE"
    LINEAGE_GAP = "LINEAGE_GAP"
    INDEX_OUT_OF_RANGE = "INDEX_OUT_OF_RANGE"
    UNKNOWN_PRESET = "UNKNOWN_PRESET"
    ZOOM_NON_POSITIVE = "ZOOM_NON_POSITIVE"
    MATRIX_NON_FINITE = "MATRIX_NON_FINITE"
    GRADE_NOT_IN_ENUM = "GRADE_NOT_IN_ENUM"
    JUSTIFICATION_NOT_IN_ENUM = "JUSTIFICATION_NOT_IN_ENUM"
    SUBJECT_REF_INVALID = "SUBJECT_REF_INVALID"
    SESSION_NOT_FOUND = "SESSION_NOT_FOUND"
    SESSION_OPEN_FAILED = "SESSION_OPEN_FAILED"


_STATUS_BY_CODE: Final[dict[str, int]] = {
    RejectionCode.SESSION_NOT_FOUND.value: 404,
    RejectionCode.SESSION_OPEN_FAILED.value: 409,
}


class ViewerRejectionError(Exception):
    """A typed, fail-closed action rejection (always visible, never silent)."""

    def __init__(self, code: RejectionCode | str, message: str) -> None:
        self.code = RejectionCode(code)
        self.message = message
        self.http_status = _STATUS_BY_CODE.get(self.code.value, 400)
        super().__init__(f"{self.code.value}: {message}")

    @property
    def payload(self) -> dict[str, object]:
        """The exact rejection payload shape — no free-form extra keys."""

        return {"error": {"code": self.code.value, "message": self.message}}


@dataclass(frozen=True, slots=True)
class ParsedAction:
    """A fully validated action request (all grammar gates passed)."""

    action: str
    params: dict[str, object]
    client_nonce: str
    view_plane: str
    view_frame_index: int
    session_id: str
    parent_action_nonce: str | None


def _reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ViewerRejectionError(
                RejectionCode.DUPLICATE_JSON_KEYS,
                f"duplicate JSON object key {key!r} is rejected",
            )
        result[key] = value
    return result


def _parse_constant(value: str) -> object:
    raise ViewerRejectionError(
        RejectionCode.NON_CANONICAL_ENCODING,
        f"non-canonical JSON constant {value!r} is rejected",
    )


def parse_action_body(body: bytes) -> dict[str, object]:
    """Parse the raw POST body: duplicate-key and canonical-byte gates."""

    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ViewerRejectionError(
            RejectionCode.NON_CANONICAL_ENCODING, "request body is not UTF-8"
        ) from exc
    try:
        parsed = json.loads(text, object_pairs_hook=_reject_duplicates,
                            parse_constant=_parse_constant)
    except ViewerRejectionError:
        raise
    except json.JSONDecodeError as exc:
        raise ViewerRejectionError(
            RejectionCode.NON_CANONICAL_ENCODING,
            f"request body is not valid JSON ({exc.msg})",
        ) from None
    if not isinstance(parsed, dict):
        raise ViewerRejectionError(
            RejectionCode.NON_CANONICAL_ENCODING,
            "request body must be a canonical JSON object",
        )
    try:
        canonical = canonical_json(parsed)
    except (TypeError, ValueError) as exc:
        raise ViewerRejectionError(
            RejectionCode.NON_CANONICAL_ENCODING,
            f"request body is not canonically serializable ({exc})",
        ) from None
    if text != canonical:
        raise ViewerRejectionError(
            RejectionCode.NON_CANONICAL_ENCODING,
            "request bytes are not the canonical encoding (sort_keys=true, "
            "separators=(',', ':'), ensure_ascii=false)",
        )
    return parsed


def _require_object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ViewerRejectionError(RejectionCode.UNKNOWN_PARAM, f"{label} must be an object")
    return value


def _require_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ViewerRejectionError(RejectionCode.UNKNOWN_PARAM, f"{label} must be an integer")
    return value


def _require_str(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise ViewerRejectionError(RejectionCode.UNKNOWN_PARAM, f"{label} must be a string")
    return value


def _is_seed_voxel(value: object) -> bool:
    """The closed seed-voxel domain: three non-negative integers."""

    return (
        isinstance(value, list)
        and len(value) == 3
        and all(
            isinstance(entry, int) and not isinstance(entry, bool) and entry >= 0
            for entry in value
        )
    )


def _is_seed_direction(value: object) -> bool:
    """The closed seed-direction domain: three finite numbers, non-zero."""

    if not isinstance(value, list) or len(value) != 3:
        return False
    numbers: list[float] = []
    for entry in value:
        if isinstance(entry, bool) or not isinstance(entry, (int, float)):
            return False
        number = float(entry)
        if not math.isfinite(number):
            return False
        numbers.append(number)
    return sum(number * number for number in numbers) > 0.0


_F32 = struct.Struct("<f")


def _to_binary32(value: float) -> float:
    """One IEEE binary32 cast of an already finite-bounded number."""

    return float(_F32.unpack(_F32.pack(value))[0])


def _require_wl_custom_number(value: object, label: str) -> float:
    """One custom-WL numeric: a finite binary32-range value or UNKNOWN_PARAM.

    ``bool`` is never a number here; a binary64-magnitude beyond the binary32
    maximum is refused (the later cast would overflow to infinity); NaN and
    infinities are refused.  The server NEVER clamps.
    """

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ViewerRejectionError(
            RejectionCode.UNKNOWN_PARAM, f"{label} must be a finite number"
        )
    try:
        number = float(value)
    except OverflowError:
        raise ViewerRejectionError(
            RejectionCode.UNKNOWN_PARAM, f"{label} must be a finite number"
        ) from None
    if not math.isfinite(number) or abs(number) > FLOAT32_MAX:
        raise ViewerRejectionError(
            RejectionCode.UNKNOWN_PARAM,
            f"{label} must be a finite binary32 number",
        )
    return number


def _validate_wl_custom_params(params: dict[str, object]) -> None:
    """The closed SET_WL_CUSTOM value domain (spec Part 4.2).

    Both ``center`` and ``width`` must be finite binary32 values; ``width``
    must additionally stay a POSITIVE binary32 value (a width that underflows
    to binary32 zero is refused).  Every refusal reuses the existing
    ``UNKNOWN_PARAM`` code — the frozen 16-code vocabulary gains nothing.
    """

    _require_wl_custom_number(params.get("center"), "params.center")
    width = _require_wl_custom_number(params.get("width"), "params.width")
    if not _to_binary32(width) > 0.0:
        raise ViewerRejectionError(
            RejectionCode.UNKNOWN_PARAM,
            "params.width must be a positive binary32 value (the server never clamps)",
        )


@dataclass(frozen=True, slots=True)
class SessionGrammarView:
    """The read-only session facts the grammar validates against."""

    session_id: str
    study_id: str
    frame_count_by_plane: dict[str, int]
    wl_presets_active: tuple[str, ...]
    used_nonces: frozenset[str]
    accepted_nonces: frozenset[str]


def validate_action(
    body: bytes,
    *,
    path_session_id: str,
    session: SessionGrammarView,
) -> ParsedAction:
    """Validate one action POST against the full pinned precedence."""

    parsed = parse_action_body(body)

    # 3. Session resolution + lineage.session_id binding (spec-pinned).
    if session.session_id != path_session_id:
        raise ViewerRejectionError(
            RejectionCode.SESSION_NOT_FOUND,
            "no open viewer session for this id",
        )
    lineage = _require_object(parsed.get("lineage"), "lineage")
    if set(lineage) != {"session_id", "parent_action_nonce"}:
        raise ViewerRejectionError(
            RejectionCode.UNKNOWN_PARAM,
            "lineage must be exactly {session_id, parent_action_nonce}",
        )
    if lineage["session_id"] != session.session_id:
        raise ViewerRejectionError(
            RejectionCode.SESSION_NOT_FOUND,
            "lineage.session_id must equal the session id in the request path",
        )

    # 4/5. Closed action + envelope/param key sets.
    action = parsed.get("action")
    if not isinstance(action, str) or action not in ACTIONS:
        raise ViewerRejectionError(
            RejectionCode.UNKNOWN_ACTION,
            "action is outside the closed v1 set",
        )
    if set(parsed) != ENVELOPE_KEYS:
        raise ViewerRejectionError(
            RejectionCode.UNKNOWN_PARAM,
            "envelope must be exactly {action, params, client_nonce, view, lineage}",
        )
    params = _require_object(parsed.get("params"), "params")
    expected_keys = set(ACTION_PARAM_KEYS[action])
    if set(params) != expected_keys:
        raise ViewerRejectionError(
            RejectionCode.UNKNOWN_PARAM,
            f"{action} params must be exactly {sorted(expected_keys)}",
        )
    view = _require_object(parsed.get("view"), "view")
    if set(view) != {"plane", "frame_index"}:
        raise ViewerRejectionError(
            RejectionCode.UNKNOWN_PARAM,
            "view must be exactly {plane, frame_index}",
        )

    # client_nonce grammar (malformed -> UNKNOWN_PARAM; reuse -> NONCE_REUSE).
    client_nonce = _require_str(parsed.get("client_nonce"), "client_nonce")
    if NONCE_RE.fullmatch(client_nonce) is None:
        raise ViewerRejectionError(
            RejectionCode.UNKNOWN_PARAM,
            "client_nonce must be 16-64 characters from [A-Za-z0-9_-]",
        )

    parent = lineage["parent_action_nonce"]
    if parent is not None and (
        not isinstance(parent, str) or NONCE_RE.fullmatch(parent) is None
    ):
        raise ViewerRejectionError(
            RejectionCode.UNKNOWN_PARAM,
            "parent_action_nonce must be a client_nonce or None",
        )

    # 6/7. Matrix value validation with the spec-pinned internal order.
    if action == "SET_VIEWPORT_MATRIX":
        zoom = params.get("zoom")
        if isinstance(zoom, bool) or not isinstance(zoom, (int, float)):
            raise ViewerRejectionError(
                RejectionCode.UNKNOWN_PARAM, "zoom must be a number"
            )
        if not math.isfinite(float(zoom)) or float(zoom) <= 0.0:
            raise ViewerRejectionError(
                RejectionCode.ZOOM_NON_POSITIVE,
                "zoom must be finite and > 0 (the server never clamps)",
            )
        for pan_label in ("pan_x", "pan_y"):
            pan = params.get(pan_label)
            if isinstance(pan, bool) or not isinstance(pan, (int, float)):
                raise ViewerRejectionError(
                    RejectionCode.UNKNOWN_PARAM, f"{pan_label} must be a number"
                )
        zoom_f = float(zoom)
        pan_x_f = float(params["pan_x"])  # type: ignore[arg-type]
        pan_y_f = float(params["pan_y"])  # type: ignore[arg-type]
        non_finite_pan = not math.isfinite(pan_x_f) or not math.isfinite(pan_y_f)
        out_of_range = (
            zoom_f > ZOOM_MAX
            or abs(pan_x_f) > PAN_EXTENT_LIMIT
            or abs(pan_y_f) > PAN_EXTENT_LIMIT
        )
        if non_finite_pan or out_of_range:
            raise ViewerRejectionError(
                RejectionCode.MATRIX_NON_FINITE,
                "zoom must be <= 64 and pan finite within +-2 viewport extents",
            )

    # 8. The view must resolve to a valid frame of that plane.
    view_plane = _require_str(view.get("plane"), "view.plane")
    if view_plane not in PLANES:
        raise ViewerRejectionError(
            RejectionCode.UNKNOWN_PARAM,
            "view.plane is outside the closed plane enum",
        )
    view_index = _require_int(view.get("frame_index"), "view.frame_index")
    if view_index < 0 or view_index >= session.frame_count_by_plane.get(view_plane, 0):
        raise ViewerRejectionError(
            RejectionCode.VIEW_MISMATCH,
            "view does not resolve to a valid frame of that plane",
        )

    # 9. Nonce uniqueness for the session's lifetime.
    if client_nonce in session.used_nonces:
        raise ViewerRejectionError(
            RejectionCode.NONCE_REUSE,
            "client_nonce was already seen in this session's lifetime",
        )

    # 10. Lineage chain.
    if not session.accepted_nonces:
        if parent is not None:
            raise ViewerRejectionError(
                RejectionCode.LINEAGE_GAP,
                "the first action of a session carries parent_action_nonce=None",
            )
    else:
        if parent is None or parent not in session.accepted_nonces:
            raise ViewerRejectionError(
                RejectionCode.LINEAGE_GAP,
                "parent_action_nonce does not reference an accepted action",
            )

    # 11-15. Action-specific semantic gates.
    if action == "SET_SLICE":
        slice_plane = _require_str(params.get("plane"), "params.plane")
        if slice_plane not in PLANES:
            raise ViewerRejectionError(
                RejectionCode.UNKNOWN_PARAM,
                "params.plane is outside the closed plane enum",
            )
        slice_index = _require_int(params.get("frame_index"), "params.frame_index")
        if slice_index < 0 or slice_index >= session.frame_count_by_plane.get(
            slice_plane, 0
        ):
            raise ViewerRejectionError(
                RejectionCode.INDEX_OUT_OF_RANGE,
                "frame_index is outside the closed plane frame list",
            )
    elif action == "SET_PLANE":
        target_plane = _require_str(params.get("plane"), "params.plane")
        if target_plane not in PLANES:
            raise ViewerRejectionError(
                RejectionCode.UNKNOWN_PARAM,
                "params.plane is outside the closed plane enum",
            )
    elif action == "SET_WL_PRESET":
        preset_id = _require_str(params.get("preset_id"), "params.preset_id")
        if preset_id not in session.wl_presets_active:
            raise ViewerRejectionError(
                RejectionCode.UNKNOWN_PRESET,
                "preset_id is outside the session's active preset set",
            )
    elif action == "SET_WL_CUSTOM":
        _validate_wl_custom_params(params)
    elif action == "REQUEST_CORRECTION":
        subject_ref = _require_str(params.get("subject_ref"), "params.subject_ref")
        if (
            _STUDY_REF_RE.fullmatch(subject_ref) is None
            or subject_ref != session.study_id
        ):
            raise ViewerRejectionError(
                RejectionCode.SUBJECT_REF_INVALID,
                "subject_ref must be exactly this session's study_id:<ULID>",
            )
        proposed_grade = _require_str(
            params.get("proposed_grade"), "params.proposed_grade"
        )
        if proposed_grade not in QUALITY_GRADES:
            raise ViewerRejectionError(
                RejectionCode.GRADE_NOT_IN_ENUM,
                "proposed_grade is outside the accepted QualityGrade enum",
            )
        justification_code = _require_str(
            params.get("justification_code"), "params.justification_code"
        )
        if justification_code != JUSTIFICATION_HUMAN_EVIDENCE:
            raise ViewerRejectionError(
                RejectionCode.JUSTIFICATION_NOT_IN_ENUM,
                "justification_code must be exactly HUMAN_EVIDENCE",
            )
    elif action == "SET_OVERLAY_VISIBILITY":
        overlay_id = _require_str(params.get("overlay_id"), "params.overlay_id")
        if overlay_id not in OVERLAY_IDS:
            raise ViewerRejectionError(
                RejectionCode.UNKNOWN_PARAM,
                "params.overlay_id is outside the closed overlay allowlist",
            )
        if not isinstance(params.get("visible"), bool):
            raise ViewerRejectionError(
                RejectionCode.UNKNOWN_PARAM,
                "params.visible must be a boolean",
            )
        overlay_mode = _require_str(params.get("mode"), "params.mode")
        if overlay_mode not in OVERLAY_MODES:
            raise ViewerRejectionError(
                RejectionCode.UNKNOWN_PARAM,
                "params.mode is outside the closed SLICE_LOCAL | PROJECTION vocabulary",
            )
    elif action == "REQUEST_CENTERLINE_DECISION":
        subject_ref = _require_str(params.get("subject_ref"), "params.subject_ref")
        if (
            _STUDY_REF_RE.fullmatch(subject_ref) is None
            or subject_ref != session.study_id
        ):
            raise ViewerRejectionError(
                RejectionCode.SUBJECT_REF_INVALID,
                "subject_ref must be exactly this session's study_id:<ULID>",
            )
        centerline_ref = _require_str(
            params.get("centerline_ref"), "params.centerline_ref"
        )
        if _CENTERLINE_REF_RE.fullmatch(centerline_ref) is None:
            raise ViewerRejectionError(
                RejectionCode.UNKNOWN_PARAM,
                "params.centerline_ref must be exactly centerline_artifact_id:<ULID>",
            )
        decision = _require_str(params.get("decision"), "params.decision")
        if decision not in CENTERLINE_DECISIONS:
            raise ViewerRejectionError(
                RejectionCode.UNKNOWN_PARAM,
                "params.decision is outside the closed ACCEPT/EDIT/REJECT vocabulary",
            )
        seed_voxel = params.get("seed_voxel")
        seed_direction = params.get("seed_direction_world")
        if decision == "EDIT":
            if not _is_seed_voxel(seed_voxel):
                raise ViewerRejectionError(
                    RejectionCode.UNKNOWN_PARAM,
                    "EDIT requires params.seed_voxel as three non-negative integers",
                )
            if not _is_seed_direction(seed_direction):
                raise ViewerRejectionError(
                    RejectionCode.UNKNOWN_PARAM,
                    "EDIT requires params.seed_direction_world as a finite non-zero 3-vector",
                )
        elif seed_voxel is not None or seed_direction is not None:
            raise ViewerRejectionError(
                RejectionCode.UNKNOWN_PARAM,
                "ACCEPT/REJECT must carry seed_voxel=null and seed_direction_world=null",
            )
    elif action == "SELECT_VESSEL":
        vessel_label = _require_str(params.get("vessel_label"), "params.vessel_label")
        if vessel_label not in VESSEL_LABEL_TOKENS:
            raise ViewerRejectionError(
                RejectionCode.UNKNOWN_PARAM,
                "params.vessel_label is outside the frozen VESSEL:* tokens",
            )

    return ParsedAction(
        action=action,
        params=params,
        client_nonce=client_nonce,
        view_plane=view_plane,
        view_frame_index=view_index,
        session_id=str(lineage["session_id"]),
        parent_action_nonce=parent,
    )


__all__ = [
    "ACTIONS",
    "ACTION_PARAM_KEYS",
    "CENTERLINE_DECISIONS",
    "ENVELOPE_KEYS",
    "FLOAT32_MAX",
    "JUSTIFICATION_HUMAN_EVIDENCE",
    "NONCE_RE",
    "ParsedAction",
    "PAN_EXTENT_LIMIT",
    "QUALITY_GRADES",
    "RejectionCode",
    "SessionGrammarView",
    "ViewerRejectionError",
    "ZOOM_MAX",
    "parse_action_body",
    "validate_action",
]
