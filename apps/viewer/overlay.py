"""The coronary centerline overlay document of the viewer (display-only).

CORONARY_CENTERLINE_001, output O-3: the viewer draws the reviewable
centerline path over the frozen MPR stack and exposes exactly ONE overlay
visibility toggle.  This module owns the closed overlay document and its
display projection; it never touches LAYER-G, never mutates an artifact, and
never carries a measurement.

Two honest coordinate statements are kept apart:

* ``path_world_mm`` / ``s_mm`` - the sealed CENTERLINE payload geometry,
  verbatim in DICOM patient (LPS) millimetres.  It is the canonical record of
  what is drawn.
* ``plane_projections`` - the DISPLAY-ONLY mapping of the same points into
  each LAYER-G frame's in-plane pixel indices ``(column, row)`` (plus the
  signed out-of-plane distance in millimetres).  The viewer's frozen client
  deliberately carries no patient geometry (see ``webapp/src/lib/metadata.js``):
  the stack viewports work in the served PNG's index space, and the server is
  the only place that knows the volume affine.  Projecting here - server-side,
  from the validated MPR payload geometry - is therefore the only mapping that
  is not a silent client-side fabrication.  The projection is a parallel
  projection along the frame normal; it is explicitly display-only and never a
  distance/measurement surface (Rule C).

Exactly one ACTIVE CENTERLINE artifact may be present in the session (the
slice is single-vessel, single-path); zero or multiple candidates are typed,
visible refusals - never a silent pick.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any, Final, cast

from apps.viewer.schemas import PLANES, ULID_RE
from core.artifacts import DerivedArtifact
from core.coronary import VESSEL_LABELS, validate_centerline_payload
from core.enums import ArtifactKind, ArtifactStatus, ReviewState

#: The closed overlay allowlist (the frozen enum-allowlist shape).
OVERLAY_IDS: Final[tuple[str, ...]] = ("CENTERLINE",)
#: The single v1 coronary overlay id.
CORONARY_OVERLAY_ID: Final[str] = "CENTERLINE"
#: The closed overlay document schema.
OVERLAY_SCHEMA: Final[str] = "ccta.viewer.coronary_overlay.v1"
#: The frozen display-projection rule recorded in every document.
#: The projection math is byte-identical under both display modes; only the
#: DRAW SUBSET changes (Decision A).
OVERLAY_PROJECTION_RULE: Final[str] = "PARALLEL_IN_PLANE_PIXEL_INDEX_V1"
#: The frozen slice-local slab rule of the normal MPR overlay (Decision A):
#: a centerline point is drawn on the displayed frame IFF
#: ``|out_of_plane_mm| <= max(in_plane_spacing_mm) / 2`` of that frame's own
#: ``in_plane_spacing_mm`` pair (the MPR payload's recorded sampling cell).
OVERLAY_SLAB_RULE: Final[str] = "SLICE_LOCAL_HALF_CELL_OF_FRAME_SPACING_V1"
#: The closed display-mode vocabulary of the overlay read surface (Decision
#: A): the normal slice-local overlay plus an explicitly labeled PROJECTION
#: localizer mode (default OFF on the client).
OVERLAY_MODES: Final[tuple[str, ...]] = ("SLICE_LOCAL", "PROJECTION")
#: The default display mode (the normal overlay is slice-local).
OVERLAY_DEFAULT_MODE: Final[str] = "SLICE_LOCAL"
#: The visible localizer caption of the PROJECTION mode (display-only; the
#: caption text is fixed so a projection can never be confused with
#: same-slice anatomy).
OVERLAY_PROJECTION_CAPTION: Final[str] = "PROJECTION — not same-slice anatomy"
#: The canonical reference prefix of a CENTERLINE artifact.
CENTERLINE_REF_PREFIX: Final[str] = "centerline_artifact_id:"

_ULID_CHARS = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
_CENTERLINE_REF_RE: Final[re.Pattern[str]] = re.compile(
    rf"centerline_artifact_id:[{_ULID_CHARS}]{{26}}\Z"
)
_MPR_REF_RE: Final[re.Pattern[str]] = re.compile(rf"mpr_artifact_id:[{_ULID_CHARS}]{{26}}\Z")

_DOCUMENT_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "session_id",
        "overlay_id",
        "visible",
        "mode",
        "centerline_ref",
        "review_state",
        "vessel_label",
        "point_count",
        "s_mm",
        "path_world_mm",
        "projection_rule",
        "slab_rule",
        "plane_projections",
    }
)

_PROJECTION_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "frame_index",
        "source_mpr_ref",
        "slab_half_thickness_mm",
        "in_slab_runs",
        "columns_rows",
        "out_of_plane_mm",
    }
)

#: The closed typed-refusal vocabulary of the overlay read surface.
REFUSAL_UNAVAILABLE: Final[str] = "CENTERLINE_OVERLAY_UNAVAILABLE"
REFUSAL_AMBIGUOUS: Final[str] = "CENTERLINE_OVERLAY_AMBIGUOUS"
REFUSAL_INVALID: Final[str] = "CENTERLINE_OVERLAY_INVALID"


class CoronaryOverlayRefusalError(ValueError):
    """A typed, fail-closed overlay refusal (HTTP 409, never a fallback)."""

    http_status: Final[int] = 409

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")

    @property
    def payload(self) -> dict[str, object]:
        return {"error": {"code": self.code, "message": self.message}}


def _refuse(code: str, message: str) -> CoronaryOverlayRefusalError:
    return CoronaryOverlayRefusalError(code, message)


def select_active_centerline(
    artifacts: Sequence[DerivedArtifact],
    vessel_label: str | None = None,
    payload_documents: Mapping[str, Mapping[str, object]] | None = None,
) -> DerivedArtifact:
    """The single ACTIVE CENTERLINE of the session, or a typed refusal.

    Status, not review state, decides candidacy: PREDICTED/ACCEPTED/CORRECTED/
    CONFIRMED_FINAL artifacts are all ACTIVE and reviewable; REJECTED is
    ``INVALIDATED`` and SUPERSEDED/STALE predecessors are excluded by status.

    CORONARY_MULTI_VESSEL_NAV (Decision A): an optional ``vessel_label``
    filter narrows candidacy to ACTIVE centerlines whose sealed payload
    carries that frozen ``VESSEL:*`` label.  ``None`` preserves the exact
    #28 single-path behavior (including the exact refusal messages).
    Refusal shapes are unchanged: zero matches refuse UNAVAILABLE, more than
    one refuses AMBIGUOUS.  A non-frozen label refuses INVALID without
    touching candidacy.
    """

    if vessel_label is not None and (
        not isinstance(vessel_label, str) or vessel_label not in VESSEL_LABELS
    ):
        raise _refuse(
            REFUSAL_INVALID,
            f"vessel_label {vessel_label!r} is outside the frozen VESSEL:* tokens",
        )
    active = [
        artifact
        for artifact in artifacts
        if artifact.kind is ArtifactKind.CENTERLINE
        and artifact.status is ArtifactStatus.ACTIVE
    ]
    if vessel_label is not None and payload_documents is not None:
        kept: list[DerivedArtifact] = []
        for artifact in active:
            document = payload_documents.get(artifact.artifact_id)
            if not isinstance(document, Mapping):
                continue
            try:
                label = validate_centerline_payload(document).get("vessel_label")
            except Exception:  # noqa: BLE001 - an invalid payload cannot match
                continue
            if label == vessel_label:
                kept.append(artifact)
        active = kept
    if not active:
        if vessel_label is None:
            raise _refuse(
                REFUSAL_UNAVAILABLE,
                "no ACTIVE CENTERLINE artifact is present in this session",
            )
        raise _refuse(
            REFUSAL_UNAVAILABLE,
            f"no ACTIVE CENTERLINE artifact carries vessel_label {vessel_label!r}",
        )
    if len(active) > 1:
        if vessel_label is None:
            raise _refuse(
                REFUSAL_AMBIGUOUS,
                f"{len(active)} ACTIVE CENTERLINE artifacts are present; "
                "the single-path overlay cannot choose one",
            )
        raise _refuse(
            REFUSAL_AMBIGUOUS,
            f"{len(active)} ACTIVE CENTERLINE artifacts carry "
            f"vessel_label {vessel_label!r}; the overlay cannot choose one",
        )
    return active[0]


def _require_vec3(value: object, label: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise _refuse(REFUSAL_INVALID, f"{label} is not a 3-vector")
    result: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise _refuse(REFUSAL_INVALID, f"{label} must contain finite numbers")
        number = float(item)
        if not math.isfinite(number):
            raise _refuse(REFUSAL_INVALID, f"{label} must contain finite numbers")
        result.append(number)
    return (result[0], result[1], result[2])


def _frame_geometry(
    frame_id: str, mpr_document: Mapping[str, object]
) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
    float,
    float,
    float,
]:
    """The validated in-plane geometry of one LAYER-G MPR frame.

    Returns ``(origin, row_axis, column_axis, row_spacing, column_spacing,
    slab_half_thickness_mm)`` where the slab half-thickness is the frozen
    slice-local rule: ``max(in_plane_spacing_mm) / 2`` of THIS frame's own
    recorded sampling cell (``SLICE_LOCAL_HALF_CELL_OF_FRAME_SPACING_V1``).
    """

    origin = _require_vec3(mpr_document.get("origin"), f"{frame_id}.origin")
    row_axis = _require_vec3(mpr_document.get("row_axis"), f"{frame_id}.row_axis")
    column_axis = _require_vec3(
        mpr_document.get("column_axis"), f"{frame_id}.column_axis"
    )
    spacing = mpr_document.get("in_plane_spacing_mm")
    if not isinstance(spacing, (list, tuple)) or len(spacing) != 2:
        raise _refuse(REFUSAL_INVALID, f"{frame_id}.in_plane_spacing_mm is not a pair")
    row_spacing = float(spacing[0])
    column_spacing = float(spacing[1])
    if not math.isfinite(row_spacing) or row_spacing <= 0.0:
        raise _refuse(REFUSAL_INVALID, f"{frame_id}.row spacing must be positive")
    if not math.isfinite(column_spacing) or column_spacing <= 0.0:
        raise _refuse(REFUSAL_INVALID, f"{frame_id}.column spacing must be positive")
    slab_half_thickness_mm = max(row_spacing, column_spacing) / 2.0
    return (
        origin,
        row_axis,
        column_axis,
        row_spacing,
        column_spacing,
        slab_half_thickness_mm,
    )


def _project_point(
    point: tuple[float, float, float],
    geometry: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
        float,
        float,
        float,
    ],
) -> tuple[tuple[float, float], float]:
    """Project one world point into ``(column, row, out_of_plane_mm)``.

    The projection math is byte-identical to before (parallel projection
    along the frame normal); only the DRAW SUBSET is slab-filtered elsewhere.
    """

    origin, row_axis, column_axis, row_spacing, column_spacing, _slab = geometry
    delta = (
        point[0] - origin[0],
        point[1] - origin[1],
        point[2] - origin[2],
    )
    row_mm = sum(a * b for a, b in zip(delta, row_axis))
    column_mm = sum(a * b for a, b in zip(delta, column_axis))
    normal = (
        row_axis[1] * column_axis[2] - row_axis[2] * column_axis[1],
        row_axis[2] * column_axis[0] - row_axis[0] * column_axis[2],
        row_axis[0] * column_axis[1] - row_axis[1] * column_axis[0],
    )
    out_of_plane_mm = sum(a * b for a, b in zip(delta, normal))
    return ((column_mm / column_spacing, row_mm / row_spacing), out_of_plane_mm)


def build_coronary_overlay_document(
    *,
    session_id: str,
    artifacts: Sequence[DerivedArtifact],
    payload_documents: Mapping[str, Mapping[str, object]],
    geometry: Mapping[str, Sequence[Any]],
    visible: bool,
    overlay_id: str = CORONARY_OVERLAY_ID,
    mode: str = OVERLAY_DEFAULT_MODE,
    vessel_label: str | None = None,
) -> dict[str, object]:
    """Build the closed ``ccta.viewer.coronary_overlay.v1`` document.

    ``geometry`` is the LAYER-G frame list per plane
    (``apps.viewer.frames.GeometryFrame``); its MPR payload documents supply
    the frame affine used for the display-only projection.

    Decision A: the normal overlay is SLICE-LOCAL.  The server computes the
    slab-filtered draw subset per frame (``in_slab_runs``: consecutive
    in-slab runs of point indices, cut at the slab boundary by linear
    interpolation in world mm, display-only); the full projection arrays stay
    in the document for audit.  The client draws exactly the served subset.
    ``mode`` selects the served subset (``SLICE_LOCAL`` default); an unknown
    mode is a typed refusal.  The served ``mode`` is authoritative — a client
    PROJECTION request never changes the stored visibility state.
    """

    if overlay_id not in OVERLAY_IDS:
        raise _refuse(
            REFUSAL_INVALID,
            f"overlay_id {overlay_id!r} is outside the closed allowlist",
        )
    if not isinstance(visible, bool):
        raise _refuse(REFUSAL_INVALID, "visible must be a boolean")
    if not isinstance(mode, str) or mode not in OVERLAY_MODES:
        raise _refuse(
            REFUSAL_INVALID,
            f"mode {mode!r} is outside the closed SLICE_LOCAL | PROJECTION vocabulary",
        )
    artifact = select_active_centerline(
        artifacts, vessel_label=vessel_label, payload_documents=payload_documents
    )
    document = payload_documents.get(artifact.artifact_id)
    if document is None:
        raise _refuse(
            REFUSAL_INVALID,
            f"CENTERLINE {artifact.artifact_id} has no payload document",
        )
    try:
        validated = validate_centerline_payload(document)
    except Exception as exc:  # noqa: BLE001 - a refused document is a typed refusal
        raise _refuse(
            REFUSAL_INVALID,
            f"CENTERLINE {artifact.artifact_id} payload is invalid ({exc})",
        ) from None
    points_value = validated["points_world"]
    arcs_value = validated["s_mm"]
    points: list[tuple[float, float, float]] = []
    for entry in cast(list[list[float]], points_value):
        points.append((float(entry[0]), float(entry[1]), float(entry[2])))
    arcs = [float(value) for value in cast(list[float], arcs_value)]

    projections: dict[str, list[dict[str, object]]] = {}
    for plane in PLANES:
        frames: list[dict[str, object]] = []
        for frame in geometry.get(plane, ()):
            frame_id = str(getattr(frame, "mpr_artifact_id", ""))
            mpr_document = payload_documents.get(frame_id)
            if not isinstance(mpr_document, Mapping):
                raise _refuse(
                    REFUSAL_INVALID,
                    f"LAYER-G frame {frame_id} has no MPR payload document",
                )
            frame_geometry = _frame_geometry(frame_id, mpr_document)
            slab_half = frame_geometry[5]
            columns_rows: list[list[float]] = []
            out_of_plane_mm: list[float] = []
            for world_point in points:
                (column, row), depth = _project_point(world_point, frame_geometry)
                columns_rows.append([column, row])
                out_of_plane_mm.append(depth)
            if mode == "SLICE_LOCAL":
                runs = slab_runs_for_frame(
                    points, out_of_plane_mm, columns_rows, slab_half
                )
            else:
                runs = [
                    {
                        "start_index": 0,
                        "end_index": len(points) - 1,
                        "columns_rows": [list(pair) for pair in columns_rows],
                    }
                ]
            frames.append(
                {
                    "frame_index": int(getattr(frame, "frame_index", len(frames))),
                    "source_mpr_ref": f"mpr_artifact_id:{frame_id}",
                    "slab_half_thickness_mm": slab_half,
                    "in_slab_runs": runs,
                    "columns_rows": columns_rows,
                    "out_of_plane_mm": out_of_plane_mm,
                }
            )
        projections[plane] = frames

    overlay: dict[str, object] = {
        "schema_version": OVERLAY_SCHEMA,
        "session_id": session_id,
        "overlay_id": overlay_id,
        "visible": visible,
        "mode": mode,
        "centerline_ref": f"{CENTERLINE_REF_PREFIX}{artifact.artifact_id}",
        "review_state": artifact.review.review_state.value,
        "vessel_label": validated.get("vessel_label"),
        "point_count": len(points),
        "s_mm": arcs,
        "path_world_mm": [list(point) for point in points],
        "projection_rule": OVERLAY_PROJECTION_RULE,
        "slab_rule": OVERLAY_SLAB_RULE,
        "plane_projections": projections,
    }
    return validate_coronary_overlay_document(overlay)


def slab_runs_for_frame(
    points: Sequence[tuple[float, float, float] | Sequence[float]],
    out_of_plane_mm: Sequence[float],
    columns_rows: Sequence[Sequence[float]],
    slab_half_thickness_mm: float,
) -> list[dict[str, object]]:
    """The slab-filtered draw subset of one frame (Decision A, server-side).

    A point is drawn IFF ``|out_of_plane_mm| <= slab_half_thickness_mm``.
    The result is the list of consecutive in-slab runs; a run whose segment
    straddles the slab boundary is cut at the boundary crossing by linear
    interpolation in world mm (display-only): the crossing's in-plane pixel
    indices are interpolated with the same segment fraction.  Out-of-slab
    runs are not drawn and never connected across the gap.  Deterministic:
    same inputs => same runs, no RNG, no view-state dependence.
    """

    count = len(out_of_plane_mm)
    runs: list[dict[str, object]] = []
    current: list[list[float]] | None = None
    current_start = 0

    def in_slab(depth: float) -> bool:
        return abs(float(depth)) <= float(slab_half_thickness_mm)

    def crossing_fraction(
        before_point: Sequence[float],
        after_point: Sequence[float],
        before_depth: float,
        after_depth: float,
    ) -> float:
        # Linear interpolation of the slab-boundary crossing in world mm:
        # the fraction along (before -> after) where |depth| == slab half.
        # The exiting/entering boundary has the sign of the out-of-slab end.
        limit = float(slab_half_thickness_mm)
        target = limit if float(after_depth) > 0.0 else -limit
        if not in_slab(before_depth):
            target = limit if float(before_depth) > 0.0 else -limit
        numerator = target - float(before_depth)
        denominator = float(after_depth) - float(before_depth)
        if denominator == 0.0:
            return 0.0
        fraction = numerator / denominator
        if fraction < 0.0:
            return 0.0
        if fraction > 1.0:
            return 1.0
        return fraction

    def interpolated_pair(
        before_pair: Sequence[float],
        after_pair: Sequence[float],
        fraction: float,
    ) -> list[float]:
        return [
            float(before_pair[0]) + fraction * (float(after_pair[0]) - float(before_pair[0])),
            float(before_pair[1]) + fraction * (float(after_pair[1]) - float(before_pair[1])),
        ]

    for index in range(count):
        depth = float(out_of_plane_mm[index])
        pair = [float(columns_rows[index][0]), float(columns_rows[index][1])]
        if in_slab(depth):
            if current is None:
                current = []
                current_start = index
                if index > 0:
                    # Entering the slab: cut the straddling segment at the
                    # boundary (interpolated in world mm via the same
                    # segment fraction on the in-plane indices).
                    before_point = points[index - 1]
                    after_point = points[index]
                    fraction = crossing_fraction(
                        before_point, after_point, float(out_of_plane_mm[index - 1]), depth
                    )
                    current.append(
                        interpolated_pair(columns_rows[index - 1], pair, fraction)
                    )
            current.append(pair)
            if index == count - 1:
                assert current is not None
                runs.append(
                    {
                        "start_index": current_start,
                        "end_index": index,
                        "columns_rows": current,
                    }
                )
        else:
            if current is not None:
                # Exiting the slab: cut the straddling segment at the
                # boundary before closing the run.
                before_point = points[index - 1]
                after_point = points[index]
                fraction = crossing_fraction(
                    before_point, after_point, float(out_of_plane_mm[index - 1]), depth
                )
                current.append(
                    interpolated_pair(columns_rows[index - 1], pair, fraction)
                )
                runs.append(
                    {
                        "start_index": current_start,
                        "end_index": index - 1,
                        "columns_rows": current,
                    }
                )
                current = None
    return runs


def validate_coronary_overlay_document(
    document: Mapping[str, object],
) -> dict[str, object]:
    """Validate the closed overlay document exactly (fail-closed)."""

    if not isinstance(document, Mapping):
        raise _refuse(REFUSAL_INVALID, "overlay document must be an object")
    if any(not isinstance(key, str) for key in document):
        raise _refuse(REFUSAL_INVALID, "overlay document keys must be strings")
    keys = set(document)
    missing = sorted(_DOCUMENT_FIELDS - keys)
    unknown = sorted(keys - _DOCUMENT_FIELDS)
    if missing or unknown:
        raise _refuse(
            REFUSAL_INVALID,
            f"overlay document key set is not closed (missing={missing}, unknown={unknown})",
        )
    if document["schema_version"] != OVERLAY_SCHEMA:
        raise _refuse(REFUSAL_INVALID, "overlay schema_version mismatch")
    session_id = document["session_id"]
    if not isinstance(session_id, str) or ULID_RE.fullmatch(session_id) is None:
        raise _refuse(REFUSAL_INVALID, "overlay session_id must be a ULID")
    overlay_id = document["overlay_id"]
    if not isinstance(overlay_id, str) or overlay_id not in OVERLAY_IDS:
        raise _refuse(REFUSAL_INVALID, "overlay_id is outside the closed allowlist")
    if not isinstance(document["visible"], bool):
        raise _refuse(REFUSAL_INVALID, "overlay visible must be a boolean")
    if not isinstance(document["mode"], str) or document["mode"] not in OVERLAY_MODES:
        raise _refuse(
            REFUSAL_INVALID,
            "overlay mode is outside the closed SLICE_LOCAL | PROJECTION vocabulary",
        )
    if (
        not isinstance(document["centerline_ref"], str)
        or _CENTERLINE_REF_RE.fullmatch(document["centerline_ref"]) is None
    ):
        raise _refuse(REFUSAL_INVALID, "centerline_ref has the wrong canonical form")
    review_state = document["review_state"]
    if not isinstance(review_state, str) or review_state not in {
        state.value for state in ReviewState
    }:
        raise _refuse(REFUSAL_INVALID, "review_state is outside the closed enum")
    vessel_label = document["vessel_label"]
    if vessel_label is not None and (
        not isinstance(vessel_label, str) or vessel_label not in VESSEL_LABELS
    ):
        raise _refuse(
            REFUSAL_INVALID, "vessel_label is outside the frozen VESSEL:* tokens"
        )
    point_count = document["point_count"]
    if (
        isinstance(point_count, bool)
        or not isinstance(point_count, int)
        or point_count <= 0
    ):
        raise _refuse(REFUSAL_INVALID, "point_count must be a positive integer")
    s_mm = document["s_mm"]
    path = document["path_world_mm"]
    if not isinstance(s_mm, list) or len(s_mm) != point_count:
        raise _refuse(REFUSAL_INVALID, "s_mm length must agree with point_count")
    if not isinstance(path, list) or len(path) != point_count:
        raise _refuse(REFUSAL_INVALID, "path_world_mm length must agree with point_count")
    previous = -1.0
    for index, value in enumerate(s_mm):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise _refuse(REFUSAL_INVALID, f"s_mm[{index}] must be finite")
        number = float(value)
        if not math.isfinite(number) or number < previous or number < 0.0:
            raise _refuse(REFUSAL_INVALID, "s_mm must be monotonic and non-negative")
        previous = number
    if float(s_mm[0]) != 0.0:
        raise _refuse(REFUSAL_INVALID, "s_mm must start at 0")
    for index, point in enumerate(path):
        if not isinstance(point, list) or len(point) != 3:
            raise _refuse(REFUSAL_INVALID, f"path_world_mm[{index}] must be a 3-vector")
        for axis, value in enumerate(point):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise _refuse(
                    REFUSAL_INVALID,
                    f"path_world_mm[{index}][{axis}] must be finite",
                )
            if not math.isfinite(float(value)):
                raise _refuse(
                    REFUSAL_INVALID,
                    f"path_world_mm[{index}][{axis}] must be finite",
                )
    if document["projection_rule"] != OVERLAY_PROJECTION_RULE:
        raise _refuse(REFUSAL_INVALID, "projection_rule is fixed")
    if document["slab_rule"] != OVERLAY_SLAB_RULE:
        raise _refuse(REFUSAL_INVALID, "slab_rule is fixed")
    projections = document["plane_projections"]
    if not isinstance(projections, Mapping) or set(projections) != set(PLANES):
        raise _refuse(
            REFUSAL_INVALID,
            "plane_projections must carry exactly the closed plane enum",
        )
    for plane in PLANES:
        frames = projections[plane]
        if not isinstance(frames, list):
            raise _refuse(REFUSAL_INVALID, f"plane_projections[{plane}] must be a list")
        for frame in frames:
            if not isinstance(frame, Mapping) or set(frame) != _PROJECTION_FIELDS:
                raise _refuse(
                    REFUSAL_INVALID,
                    f"plane_projections[{plane}] frame is not a closed projection",
                )
            if (
                isinstance(frame["frame_index"], bool)
                or not isinstance(frame["frame_index"], int)
                or frame["frame_index"] < 0
            ):
                raise _refuse(
                    REFUSAL_INVALID,
                    "projection frame_index must be a non-negative integer",
                )
            if (
                not isinstance(frame["source_mpr_ref"], str)
                or _MPR_REF_RE.fullmatch(frame["source_mpr_ref"]) is None
            ):
                raise _refuse(
                    REFUSAL_INVALID, "projection source_mpr_ref has the wrong canonical form"
                )
            slab_half = frame["slab_half_thickness_mm"]
            if (
                isinstance(slab_half, bool)
                or not isinstance(slab_half, (int, float))
                or not math.isfinite(float(slab_half))
                or float(slab_half) <= 0.0
            ):
                raise _refuse(
                    REFUSAL_INVALID,
                    "projection slab_half_thickness_mm must be a positive finite number",
                )
            _validate_slab_runs(frame["in_slab_runs"], point_count)
            columns_rows = frame["columns_rows"]
            depths = frame["out_of_plane_mm"]
            if not isinstance(columns_rows, list) or len(columns_rows) != point_count:
                raise _refuse(
                    REFUSAL_INVALID,
                    "projection columns_rows length must agree with point_count",
                )
            if not isinstance(depths, list) or len(depths) != point_count:
                raise _refuse(
                    REFUSAL_INVALID,
                    "projection out_of_plane_mm length must agree with point_count",
                )
            for index, pair in enumerate(columns_rows):
                if not isinstance(pair, list) or len(pair) != 2:
                    raise _refuse(
                        REFUSAL_INVALID,
                        f"projection columns_rows[{index}] must be a pair",
                    )
                for axis, value in enumerate(pair):
                    if (
                        isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not math.isfinite(float(value))
                    ):
                        raise _refuse(
                            REFUSAL_INVALID,
                            f"projection columns_rows[{index}][{axis}] must be finite",
                        )
            for index, value in enumerate(depths):
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                ):
                    raise _refuse(
                        REFUSAL_INVALID,
                        f"projection out_of_plane_mm[{index}] must be finite",
                    )
    return dict(document)


_RUN_FIELDS: Final[frozenset[str]] = frozenset(
    {"start_index", "end_index", "columns_rows"}
)


def _validate_slab_runs(runs: object, point_count: int) -> None:
    """Validate the server-computed slab draw subset of one frame."""

    if not isinstance(runs, list):
        raise _refuse(REFUSAL_INVALID, "projection in_slab_runs must be a list")
    previous_end = -1
    for position, run in enumerate(runs):
        if not isinstance(run, Mapping) or set(run) != _RUN_FIELDS:
            raise _refuse(
                REFUSAL_INVALID,
                f"projection in_slab_runs[{position}] is not a closed run",
            )
        start = run["start_index"]
        end = run["end_index"]
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(end, bool)
            or not isinstance(end, int)
            or start < 0
            or end < start
            or end >= point_count
        ):
            raise _refuse(
                REFUSAL_INVALID,
                f"projection in_slab_runs[{position}] has an invalid index range",
            )
        if start <= previous_end:
            raise _refuse(
                REFUSAL_INVALID,
                "projection in_slab_runs must be ordered and non-overlapping",
            )
        previous_end = end
        run_pairs = run["columns_rows"]
        if not isinstance(run_pairs, list) or not run_pairs:
            raise _refuse(
                REFUSAL_INVALID,
                f"projection in_slab_runs[{position}] columns_rows must be non-empty",
            )
        span = int(end) - int(start) + 1
        # A run carries exactly its in-slab points plus at most one
        # interpolated boundary cut at each end (span, span+1 or span+2).
        if len(run_pairs) < span or len(run_pairs) > span + 2:
            raise _refuse(
                REFUSAL_INVALID,
                f"projection in_slab_runs[{position}] length disagrees with its range",
            )
        for index, pair in enumerate(run_pairs):
            if not isinstance(pair, list) or len(pair) != 2:
                raise _refuse(
                    REFUSAL_INVALID,
                    f"projection in_slab_runs[{position}] columns_rows[{index}] "
                    "must be a pair",
                )
            for axis, value in enumerate(pair):
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                ):
                    raise _refuse(
                        REFUSAL_INVALID,
                        f"projection in_slab_runs[{position}] columns_rows[{index}][{axis}] "
                        "must be finite",
                    )


__all__ = [
    "CENTERLINE_REF_PREFIX",
    "CORONARY_OVERLAY_ID",
    "CoronaryOverlayRefusalError",
    "OVERLAY_DEFAULT_MODE",
    "OVERLAY_IDS",
    "OVERLAY_MODES",
    "OVERLAY_PROJECTION_CAPTION",
    "OVERLAY_PROJECTION_RULE",
    "OVERLAY_SCHEMA",
    "OVERLAY_SLAB_RULE",
    "REFUSAL_AMBIGUOUS",
    "REFUSAL_INVALID",
    "REFUSAL_UNAVAILABLE",
    "build_coronary_overlay_document",
    "select_active_centerline",
    "slab_runs_for_frame",
    "validate_coronary_overlay_document",
]
