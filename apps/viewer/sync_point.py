"""The synchronized MPR crosshair cursor read surface (ephemeral view aid).

MPR_SYNC_CURSOR: one shared world-space cursor across the AXIAL / CORONAL /
SAGITTAL panels.  The server derives a read-only sync-point document from one
``(plane, frame_index, row, column)`` pick:

* forward (pixel -> world): the persisted MPR frame geometry through the
  owned staged-arithmetic helper
  :func:`visualization.mpr._world_pixel_center` (REUSED AS-IS — the exact
  sampling lattice of ``reslice``; never recomputed here);
* inverse (world -> per-plane pixels): the parallel in-plane projection
  :func:`apps.viewer.overlay._project_point` (REUSED AS-IS) against each
  plane's nearest served frame.

Rule B: the canonical MPR payloads stay the single source of truth; this
module keeps no geometry of its own.  Rule C/D: no measurement, no readout,
no LLM anywhere near this path.  Rule F: the cursor is ephemeral — no stored
state, nothing persisted, nothing to invalidate (a derivation that touches no
runtime state at all).  Rule G: no new provenance — the document reuses the
existing view documents (``source_mpr_ref`` bindings).

Out-of-volume fails explicitly with the typed ``SYNC_POINT_OUT_OF_VOLUME``
refusal (never a partial cursor, never a clamped position).  The closed
``ACTIONS`` grammar is untouched.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Final, cast

from apps.viewer.overlay import OVERLAY_PROJECTION_RULE, _project_point
from apps.viewer.schemas import PLANES, ULID_RE
from visualization.mpr import _world_pixel_center, plane_axes

#: The closed sync-point document schema.
SYNC_POINT_SCHEMA: Final[str] = "ccta.viewer.sync_point.v1"
#: The mandatory document kind marker (a cursor is a view aid, never a finding).
SYNC_CURSOR_KIND: Final[str] = "CURSOR"
#: The mandatory visible caption (fixed so a cursor can never be confused
#: with an anatomical finding or a measurement readout).
SYNC_CURSOR_CAPTION: Final[str] = "CURSOR \u2014 view aid, not a finding"
#: The frozen forward-derivation rule recorded in every document: the owned
#: staged float32 world-coordinate arithmetic of the MPR sampling lattice.
SYNC_FORWARD_RULE: Final[str] = "MPR_WORLD_PIXEL_CENTER_V1"
#: The frozen inverse-projection rule recorded in every document (REUSED
#: AS-IS from the coronary overlay display projection).
SYNC_PROJECTION_RULE: Final[str] = OVERLAY_PROJECTION_RULE

_DOCUMENT_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "session_id",
        "kind",
        "caption",
        "derivation_rule",
        "projection_rule",
        "source",
        "world_mm",
        "planes",
    }
)

_SOURCE_FIELDS: Final[frozenset[str]] = frozenset(
    {"plane", "frame_index", "row", "column"}
)

_PLANE_ENTRY_FIELDS: Final[frozenset[str]] = frozenset(
    {"frame_index", "source_mpr_ref", "row", "column"}
)

#: The closed typed-refusal vocabulary of the sync-point read surface.
REFUSAL_INVALID: Final[str] = "SYNC_POINT_INVALID"
REFUSAL_OUT_OF_VOLUME: Final[str] = "SYNC_POINT_OUT_OF_VOLUME"
REFUSAL_UNAVAILABLE: Final[str] = "SYNC_POINT_UNAVAILABLE"


class SyncPointRefusalError(ValueError):
    """A typed, fail-closed sync-point refusal (HTTP 409, never a fallback)."""

    http_status: Final[int] = 409

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")

    @property
    def payload(self) -> dict[str, object]:
        return {"error": {"code": self.code, "message": self.message}}


def _refuse(code: str, message: str) -> SyncPointRefusalError:
    return SyncPointRefusalError(code, message)


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


def _sync_frame_geometry(
    frame_id: str, mpr_document: Mapping[str, object]
) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
    float,
    float,
    int,
    int,
]:
    """The validated served geometry of one LAYER-G MPR frame.

    Returns ``(origin, row_axis, column_axis, row_spacing, column_spacing,
    rows, columns)`` read response-only from the persisted MPR payload —
    field reads only, never recomputed geometry.
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
    matrix = mpr_document.get("matrix_size")
    if (
        not isinstance(matrix, (list, tuple))
        or len(matrix) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in matrix
        )
    ):
        raise _refuse(REFUSAL_INVALID, f"{frame_id}.matrix_size is not a pair")
    rows = int(cast(Sequence[int], matrix)[0])
    columns = int(cast(Sequence[int], matrix)[1])
    return (origin, row_axis, column_axis, row_spacing, column_spacing, rows, columns)


def _require_pixel(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _refuse(REFUSAL_INVALID, f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise _refuse(REFUSAL_INVALID, f"{label} must be a finite number")
    return number


def _nearest_frame_index(
    plane: str,
    world: tuple[float, float, float],
    frames: Sequence[Any],
) -> int:
    """The LAYER-G frame of *plane* nearest the world point (read-only).

    Nearest by absolute out-of-plane distance
    (``|dot(world, normal) - plane_coordinate_mm|``); ties resolve to the
    lower frame index.  Deterministic: same inputs => same frame.
    """

    (normal, _, _) = plane_axes(plane)
    coordinate = sum(a * b for a, b in zip(world, normal))
    best = 0
    best_distance = math.inf
    for position, frame in enumerate(frames):
        distance = abs(coordinate - float(frame.plane_coordinate_mm))
        if distance < best_distance:
            best_distance = distance
            best = position
    return best


def build_sync_point_document(
    *,
    session_id: str,
    plane: str,
    frame_index: int,
    row: float,
    column: float,
    geometry: Mapping[str, Sequence[Any]],
    payload_documents: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    """Build the closed ``ccta.viewer.sync_point.v1`` document.

    Derived response-only from the session's served LAYER-G geometry plus the
    persisted MPR payloads: the source ``(plane, frame_index, row, column)``
    maps forward to one shared world coordinate, which maps back into the
    nearest frame of every plane.  Touches no runtime state (the cursor is
    ephemeral; reselect simply derives again).  Any out-of-matrix position —
    source or projected — refuses ``SYNC_POINT_OUT_OF_VOLUME`` rather than
    drawing a partial cursor.
    """

    if not isinstance(session_id, str) or ULID_RE.fullmatch(session_id) is None:
        raise _refuse(REFUSAL_INVALID, "sync-point session_id must be a ULID")
    if not isinstance(plane, str) or plane not in PLANES:
        raise _refuse(
            REFUSAL_INVALID,
            f"plane {plane!r} is outside the closed AXIAL | CORONAL | SAGITTAL enum",
        )
    for candidate in PLANES:
        if not geometry.get(candidate, ()):
            raise _refuse(
                REFUSAL_UNAVAILABLE,
                f"plane {candidate} carries no served frame; the cursor cannot "
                "synchronize across all three planes",
            )
    frames = list(geometry.get(plane, ()))
    if (
        isinstance(frame_index, bool)
        or not isinstance(frame_index, int)
        or not 0 <= frame_index < len(frames)
    ):
        raise _refuse(
            REFUSAL_INVALID,
            f"frame_index {frame_index!r} is outside the served {plane} frames",
        )
    source_row = _require_pixel(row, "row")
    source_column = _require_pixel(column, "column")
    source_frame = frames[frame_index]
    source_id = str(source_frame.mpr_artifact_id)
    source_payload = payload_documents.get(source_id)
    if not isinstance(source_payload, Mapping):
        raise _refuse(
            REFUSAL_INVALID, f"LAYER-G frame {source_id} has no MPR payload document"
        )
    source_geometry = _sync_frame_geometry(source_id, source_payload)
    (
        source_origin,
        source_row_axis,
        source_column_axis,
        source_row_spacing,
        source_column_spacing,
        source_rows,
        source_columns,
    ) = source_geometry
    if not 0.0 <= source_row <= float(source_rows - 1):
        raise _refuse(
            REFUSAL_OUT_OF_VOLUME,
            f"row {source_row} is outside the served {plane} frame "
            f"(0..{source_rows - 1})",
        )
    if not 0.0 <= source_column <= float(source_columns - 1):
        raise _refuse(
            REFUSAL_OUT_OF_VOLUME,
            f"column {source_column} is outside the served {plane} frame "
            f"(0..{source_columns - 1})",
        )
    # Forward (REUSED AS-IS): the staged float32 world-coordinate arithmetic
    # of the MPR sampling lattice — the cursor sits exactly on lattice math.
    staged = _world_pixel_center(
        source_origin,
        source_row_axis,
        source_column_axis,
        source_row_spacing,
        source_column_spacing,
        cast(Any, source_row),
        cast(Any, source_column),
    )
    world = (float(staged[0]), float(staged[1]), float(staged[2]))
    if not all(math.isfinite(value) for value in world):
        raise _refuse(REFUSAL_INVALID, "the derived world coordinate is non-finite")
    plane_entries: dict[str, dict[str, object]] = {}
    for candidate in PLANES:
        candidate_frames = list(geometry.get(candidate, ()))
        nearest = _nearest_frame_index(candidate, world, candidate_frames)
        entry_frame = candidate_frames[nearest]
        entry_id = str(entry_frame.mpr_artifact_id)
        entry_payload = payload_documents.get(entry_id)
        if not isinstance(entry_payload, Mapping):
            raise _refuse(
                REFUSAL_INVALID,
                f"LAYER-G frame {entry_id} has no MPR payload document",
            )
        entry_geometry = _sync_frame_geometry(entry_id, entry_payload)
        (
            entry_origin,
            entry_row_axis,
            entry_column_axis,
            entry_row_spacing,
            entry_column_spacing,
            entry_rows,
            entry_columns,
        ) = entry_geometry
        # Inverse (REUSED AS-IS): the parallel in-plane projection of the
        # coronary overlay display math, evaluated on the nearest frame.
        (entry_column, entry_row), _depth = _project_point(
            world,
            (
                entry_origin,
                entry_row_axis,
                entry_column_axis,
                entry_row_spacing,
                entry_column_spacing,
                0.0,
            ),
        )
        if not (
            0.0 <= entry_row <= float(entry_rows - 1)
            and 0.0 <= entry_column <= float(entry_columns - 1)
        ):
            raise _refuse(
                REFUSAL_OUT_OF_VOLUME,
                f"the shared world point projects outside the served {candidate} "
                f"frame (row {entry_row}, column {entry_column}); refusing a "
                "partial cursor",
            )
        plane_entries[candidate] = {
            "frame_index": nearest,
            "source_mpr_ref": f"mpr_artifact_id:{entry_id}",
            "row": float(entry_row),
            "column": float(entry_column),
        }
    document: dict[str, object] = {
        "schema_version": SYNC_POINT_SCHEMA,
        "session_id": session_id,
        "kind": SYNC_CURSOR_KIND,
        "caption": SYNC_CURSOR_CAPTION,
        "derivation_rule": SYNC_FORWARD_RULE,
        "projection_rule": SYNC_PROJECTION_RULE,
        "source": {
            "plane": plane,
            "frame_index": frame_index,
            "row": float(source_row),
            "column": float(source_column),
        },
        "world_mm": [world[0], world[1], world[2]],
        "planes": plane_entries,
    }
    return validate_sync_point_document(document)


def validate_sync_point_document(document: Mapping[str, object]) -> dict[str, object]:
    """Validate the closed sync-point document exactly (fail-closed)."""

    if not isinstance(document, Mapping):
        raise _refuse(REFUSAL_INVALID, "sync-point document must be an object")
    if any(not isinstance(key, str) for key in document):
        raise _refuse(REFUSAL_INVALID, "sync-point document keys must be strings")
    keys = set(document)
    missing = sorted(_DOCUMENT_FIELDS - keys)
    unknown = sorted(keys - _DOCUMENT_FIELDS)
    if missing or unknown:
        raise _refuse(
            REFUSAL_INVALID,
            "sync-point document key set is not closed "
            f"(missing={missing}, unknown={unknown})",
        )
    if document["schema_version"] != SYNC_POINT_SCHEMA:
        raise _refuse(REFUSAL_INVALID, "sync-point schema_version mismatch")
    session_id = document["session_id"]
    if not isinstance(session_id, str) or ULID_RE.fullmatch(session_id) is None:
        raise _refuse(REFUSAL_INVALID, "sync-point session_id must be a ULID")
    if document["kind"] != SYNC_CURSOR_KIND:
        raise _refuse(REFUSAL_INVALID, "sync-point kind must be CURSOR")
    if document["caption"] != SYNC_CURSOR_CAPTION:
        raise _refuse(REFUSAL_INVALID, "sync-point caption is fixed")
    if document["derivation_rule"] != SYNC_FORWARD_RULE:
        raise _refuse(REFUSAL_INVALID, "sync-point derivation_rule is fixed")
    if document["projection_rule"] != SYNC_PROJECTION_RULE:
        raise _refuse(REFUSAL_INVALID, "sync-point projection_rule is fixed")
    source = document["source"]
    if not isinstance(source, Mapping) or set(source) != _SOURCE_FIELDS:
        raise _refuse(REFUSAL_INVALID, "sync-point source is not a closed pick")
    if source["plane"] not in PLANES:
        raise _refuse(REFUSAL_INVALID, "sync-point source plane is outside the enum")
    if (
        isinstance(source["frame_index"], bool)
        or not isinstance(source["frame_index"], int)
        or source["frame_index"] < 0
    ):
        raise _refuse(REFUSAL_INVALID, "sync-point source frame_index is invalid")
    for label in ("row", "column"):
        value = source[label]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise _refuse(REFUSAL_INVALID, f"sync-point source {label} must be finite")
    world = document["world_mm"]
    if not isinstance(world, list) or len(world) != 3:
        raise _refuse(REFUSAL_INVALID, "sync-point world_mm must be a 3-vector")
    for axis, value in enumerate(world):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise _refuse(
                REFUSAL_INVALID, f"sync-point world_mm[{axis}] must be finite"
            )
    planes = document["planes"]
    if not isinstance(planes, Mapping) or set(planes) != set(PLANES):
        raise _refuse(
            REFUSAL_INVALID,
            "sync-point planes must carry exactly the closed plane enum",
        )
    for candidate in PLANES:
        entry = planes[candidate]
        if not isinstance(entry, Mapping) or set(entry) != _PLANE_ENTRY_FIELDS:
            raise _refuse(
                REFUSAL_INVALID,
                f"sync-point planes[{candidate}] is not a closed plane entry",
            )
        if (
            isinstance(entry["frame_index"], bool)
            or not isinstance(entry["frame_index"], int)
            or entry["frame_index"] < 0
        ):
            raise _refuse(
                REFUSAL_INVALID,
                f"sync-point planes[{candidate}] frame_index is invalid",
            )
        ref = entry["source_mpr_ref"]
        if (
            not isinstance(ref, str)
            or not ref.startswith("mpr_artifact_id:")
            or ULID_RE.fullmatch(ref.split(":", 1)[1]) is None
        ):
            raise _refuse(
                REFUSAL_INVALID,
                f"sync-point planes[{candidate}] source_mpr_ref has the wrong form",
            )
        for label in ("row", "column"):
            value = entry[label]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise _refuse(
                    REFUSAL_INVALID,
                    f"sync-point planes[{candidate}] {label} must be finite",
                )
    return dict(document)


__all__ = [
    "REFUSAL_INVALID",
    "REFUSAL_OUT_OF_VOLUME",
    "REFUSAL_UNAVAILABLE",
    "SYNC_CURSOR_CAPTION",
    "SYNC_CURSOR_KIND",
    "SYNC_FORWARD_RULE",
    "SYNC_POINT_SCHEMA",
    "SYNC_PROJECTION_RULE",
    "SyncPointRefusalError",
    "build_sync_point_document",
    "validate_sync_point_document",
]
