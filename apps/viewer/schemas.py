"""Viewer record schemas: ``ccta.viewer.frame.v1`` and ``ccta.viewer.session.v1``.

Implements VIEWER_001_SPEC Parts 1-2 with the TWO-LAYER identity model:

- LAYER G (GEOMETRY frames): the closed, immutable per-session frame list
  built at session open (Part 5.1).  A geometry frame binds
  (session, plane, frame_index) to exactly one accepted ``MPR_VIEW``
  artifact.  Geometry frames are NEVER created, expanded, or reordered by
  any viewer action; ``frame_lists_ref`` covers LAYER-G only.
- LAYER R (RENDER variants): display-only renders of a geometry frame.  A
  render variant = (geometry frame, WL state).  WL changes mint LAYER-R
  variants ONLY — the MPR pixels and the geometry list are untouched.
  ``wl_applied`` is ALWAYS present (never None).  The LAYER-R variant class
  is named by ``wl_preset_id``: a frozen preset id names the preset, ``None``
  is either the raw/identity mapping (Part 4.1) or the v2 custom user window
  (Part 4.2) whose effective values are exactly the recorded ``wl_applied``.

Both serialized records have EXACTLY the spec key sets, validated fail-closed
(every violation listed, nothing defaulted).  The schemas live OUTSIDE
``core/`` (no ``core/enums.py`` change).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Final

from core.digest import canonical_digest

FRAME_SCHEMA: Final[str] = "ccta.viewer.frame.v1"
SESSION_SCHEMA: Final[str] = "ccta.viewer.session.v1"
STATE_SCHEMA: Final[str] = "ccta.viewer.state.v1"
GATE_OUTCOME_SCHEMA: Final[str] = "ccta.viewer.qc_gate_outcome.v1"

ULID_RE: Final[re.Pattern[str]] = re.compile(
    r"\A[0123456789ABCDEFGHJKMNPQRSTVWXYZ]{26}\Z"
)
_MPR_REF_RE: Final[re.Pattern[str]] = re.compile(r"\Ampr_artifact_id:[A-Z0-9]{26}\Z")
_PROVENANCE_REF_RE: Final[re.Pattern[str]] = re.compile(r"\Aprovenance:[A-Z0-9]{26}\Z")
_GATE_REF_RE: Final[re.Pattern[str]] = re.compile(r"\Agate:[A-Z0-9]{26}\Z")
_STUDY_REF_RE: Final[re.Pattern[str]] = re.compile(r"\Astudy_id:[A-Z0-9]{26}\Z")

PLANES: Final[tuple[str, ...]] = ("AXIAL", "CORONAL", "SAGITTAL")

_FRAME_FIELDS: Final[tuple[str, ...]] = (
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
)

_SESSION_FIELDS: Final[tuple[str, ...]] = (
    "schema_version",
    "session_id",
    "study_id",
    "planes",
    "frame_lists_ref",
    "qc_gate_state_ref",
    "wl_presets_active",
    "opened_at",
    "provenance_ref",
)


class ViewerSchemaError(ValueError):
    """A closed viewer record schema was violated (fail-closed)."""


def _require_ulid(value: object, label: str) -> str:
    if not isinstance(value, str) or ULID_RE.fullmatch(value) is None:
        raise ViewerSchemaError(f"{label} must be a 26-char ULID")
    return value


def validate_plane(value: object, label: str = "plane") -> str:
    if not isinstance(value, str) or value not in PLANES:
        raise ViewerSchemaError(f"{label} must be one of AXIAL, CORONAL, SAGITTAL")
    return value


def _require_sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ViewerSchemaError(f"{label} must be a lowercase 64-hex sha256")
    return value


@dataclass(frozen=True, slots=True)
class ViewerFrameRecord:
    """One serialized ``ccta.viewer.frame.v1`` record (binds both layers)."""

    schema_version: str
    session_id: str
    plane: str
    frame_index: int
    source_mpr_ref: str
    wl_preset_id: str | None
    wl_applied: dict[str, float]
    png_sha256: str
    png_byte_size: int
    provenance_ref: str
    frame_id: str = ""

    @property
    def identity_tuple(self) -> tuple[str, str, int, str, str | None, dict[str, float]]:
        """The stable identity tuple of a frame RECORD (Part 5.1).

        Exactly ``(session_id, plane, frame_index, source_mpr_ref,
        wl_preset_id, wl_applied)`` — the LAYER-G part plus the LAYER-R part.
        The frame ULID is record-identifier metadata ONLY and is EXCLUDED.
        A custom-WL frame (``wl_preset_id=None`` with user-applied values)
        is identified by its applied values ONLY — never by a preset id.
        """

        return (
            self.session_id,
            self.plane,
            self.frame_index,
            self.source_mpr_ref,
            self.wl_preset_id,
            dict(self.wl_applied),
        )

    def to_document(self) -> dict[str, object]:
        """The closed 10-key serialized document (Part 1, exactly)."""

        return {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "plane": self.plane,
            "frame_index": self.frame_index,
            "source_mpr_ref": self.source_mpr_ref,
            "wl_preset_id": self.wl_preset_id,
            "wl_applied": dict(self.wl_applied),
            "png_sha256": self.png_sha256,
            "png_byte_size": self.png_byte_size,
            "provenance_ref": self.provenance_ref,
        }


def validate_frame_document(document: object) -> ViewerFrameRecord:
    """Validate the closed ``ccta.viewer.frame.v1`` document, fail-closed."""

    if not isinstance(document, dict):
        raise ViewerSchemaError("viewer frame record must be a JSON object")
    keys = set(document)
    missing = sorted(set(_FRAME_FIELDS) - keys)
    unknown = sorted(keys - set(_FRAME_FIELDS))
    if missing or unknown:
        raise ViewerSchemaError(
            f"viewer frame record key set is not closed (missing={missing}, "
            f"unknown={unknown})"
        )
    if document["schema_version"] != FRAME_SCHEMA:
        raise ViewerSchemaError("frame schema_version must be ccta.viewer.frame.v1")
    session_id = _require_ulid(document["session_id"], "session_id")
    plane = validate_plane(document["plane"])
    frame_index = document["frame_index"]
    if (
        isinstance(frame_index, bool)
        or not isinstance(frame_index, int)
        or frame_index < 0
    ):
        raise ViewerSchemaError("frame_index must be a nonnegative integer")
    source_mpr_ref = document["source_mpr_ref"]
    if not isinstance(source_mpr_ref, str) or not source_mpr_ref.startswith(
        "mpr_artifact_id:"
    ):
        raise ViewerSchemaError(
            "source_mpr_ref must be exactly 'mpr_artifact_id:<ULID>'"
        )
    _require_ulid(source_mpr_ref.split(":", 1)[1], "source_mpr_ref ULID")
    wl_preset_id = document["wl_preset_id"]
    if wl_preset_id is not None and not isinstance(wl_preset_id, str):
        raise ViewerSchemaError("wl_preset_id must be a preset id string or None")
    wl_applied = document["wl_applied"]
    if not isinstance(wl_applied, dict) or set(wl_applied) != {"center", "width"}:
        raise ViewerSchemaError(
            "wl_applied must ALWAYS be present as exactly {center, width}"
        )
    center = wl_applied["center"]
    width = wl_applied["width"]
    for name, value in (("center", center), ("width", width)):
        # Canonical float form: floats, never ints (spec Part 3.1 digest rule).
        if isinstance(value, bool) or not isinstance(value, float):
            raise ViewerSchemaError(
                f"wl_applied.{name} must serialize as a canonical float "
                "(e.g. 40.0, never 40)"
            )
        # v2 custom-WL frames carry user-derived values: a persisted record
        # must never feed a non-finite number into the frozen transfer.
        if not math.isfinite(value):
            raise ViewerSchemaError(f"wl_applied.{name} must be finite")
    png_sha256 = _require_sha256(document["png_sha256"], "png_sha256")
    png_byte_size = document["png_byte_size"]
    if (
        isinstance(png_byte_size, bool)
        or not isinstance(png_byte_size, int)
        or png_byte_size <= 0
    ):
        raise ViewerSchemaError("png_byte_size must be a positive integer")
    provenance_ref = document["provenance_ref"]
    if not isinstance(provenance_ref, str) or not provenance_ref.startswith(
        "provenance:"
    ):
        raise ViewerSchemaError("provenance_ref must be exactly 'provenance:<ULID>'")
    _require_ulid(provenance_ref.split(":", 1)[1], "provenance_ref ULID")
    return ViewerFrameRecord(
        schema_version=FRAME_SCHEMA,
        session_id=session_id,
        plane=plane,
        frame_index=frame_index,
        source_mpr_ref=source_mpr_ref,
        wl_preset_id=wl_preset_id,
        wl_applied={"center": center, "width": width},
        png_sha256=png_sha256,
        png_byte_size=png_byte_size,
        provenance_ref=provenance_ref,
    )


@dataclass(frozen=True, slots=True)
class ViewerSessionRecord:
    """One serialized ``ccta.viewer.session.v1`` record (Part 2, exactly)."""

    schema_version: str
    session_id: str
    study_id: str
    planes: tuple[str, ...]
    frame_lists_ref: str
    qc_gate_state_ref: str
    wl_presets_active: tuple[str, ...]
    opened_at: str
    provenance_ref: str

    def to_document(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "study_id": self.study_id,
            "planes": list(self.planes),
            "frame_lists_ref": self.frame_lists_ref,
            "qc_gate_state_ref": self.qc_gate_state_ref,
            "wl_presets_active": list(self.wl_presets_active),
            "opened_at": self.opened_at,
            "provenance_ref": self.provenance_ref,
        }


def validate_session_document(document: object) -> ViewerSessionRecord:
    """Validate the closed ``ccta.viewer.session.v1`` document, fail-closed."""

    if not isinstance(document, dict):
        raise ViewerSchemaError("viewer session record must be a JSON object")
    keys = set(document)
    missing = sorted(set(_SESSION_FIELDS) - keys)
    unknown = sorted(keys - set(_SESSION_FIELDS))
    if missing or unknown:
        raise ViewerSchemaError(
            f"viewer session record key set is not closed (missing={missing}, "
            f"unknown={unknown})"
        )
    if document["schema_version"] != SESSION_SCHEMA:
        raise ViewerSchemaError("session schema_version must be ccta.viewer.session.v1")
    session_id = _require_ulid(document["session_id"], "session_id")
    study_id = document["study_id"]
    if not isinstance(study_id, str) or not study_id.startswith("study_id:"):
        raise ViewerSchemaError("study_id must be exactly 'study_id:<ULID>'")
    _require_ulid(study_id.split(":", 1)[1], "study_id ULID")
    planes = document["planes"]
    if planes != list(PLANES):
        raise ViewerSchemaError(
            "planes must be exactly the tuple ('AXIAL', 'CORONAL', 'SAGITTAL')"
        )
    frame_lists_ref = _require_sha256(document["frame_lists_ref"], "frame_lists_ref")
    qc_gate_state_ref = document["qc_gate_state_ref"]
    if not isinstance(qc_gate_state_ref, str) or not _GATE_REF_RE.fullmatch(
        qc_gate_state_ref
    ):
        raise ViewerSchemaError("qc_gate_state_ref must be exactly 'gate:<ULID>'")
    wl_presets_active = document["wl_presets_active"]
    if not isinstance(wl_presets_active, list) or any(
        not isinstance(item, str) or not item for item in wl_presets_active
    ):
        raise ViewerSchemaError(
            "wl_presets_active must be a tuple of frozen preset ids"
        )
    opened_at = document["opened_at"]
    if not isinstance(opened_at, str) or not opened_at:
        raise ViewerSchemaError("opened_at is required display metadata")
    provenance_ref = document["provenance_ref"]
    if not isinstance(provenance_ref, str) or not provenance_ref.startswith(
        "provenance:"
    ):
        raise ViewerSchemaError("provenance_ref must be exactly 'provenance:<ULID>'")
    _require_ulid(provenance_ref.split(":", 1)[1], "provenance_ref ULID")
    return ViewerSessionRecord(
        schema_version=SESSION_SCHEMA,
        session_id=session_id,
        study_id=study_id,
        planes=tuple(planes),
        frame_lists_ref=frame_lists_ref,
        qc_gate_state_ref=qc_gate_state_ref,
        wl_presets_active=tuple(wl_presets_active),
        opened_at=opened_at,
        provenance_ref=provenance_ref,
    )


def geometry_projection(
    session_id: str, bindings: dict[str, list[tuple[int, str]]]
) -> dict[str, object]:
    """The LAYER-G element projection per plane (Part 2 frame_lists_ref input).

    For each plane, the ordered ``[session_id, plane, frame_index,
    mpr_artifact_id]`` tuples — geometry bindings only; render variants are
    excluded.  ``frame_lists_ref`` is the canonical digest over this object.
    """

    projection: dict[str, object] = {}
    for plane in PLANES:
        rows: list[list[object]] = []
        for frame_index, mpr_id in bindings.get(plane, []):
            rows.append([session_id, plane, frame_index, f"mpr_artifact_id:{mpr_id}"])
        projection[plane] = rows
    return projection


def frame_lists_digest(
    session_id: str, bindings: dict[str, list[tuple[int, str]]]
) -> str:
    """Digest over the canonical JSON of the ordered LAYER-G tuple lists."""

    return canonical_digest(geometry_projection(session_id, bindings))


__all__ = [
    "FRAME_SCHEMA",
    "GATE_OUTCOME_SCHEMA",
    "PLANES",
    "SESSION_SCHEMA",
    "STATE_SCHEMA",
    "ULID_RE",
    "ViewerFrameRecord",
    "ViewerSchemaError",
    "ViewerSessionRecord",
    "frame_lists_digest",
    "geometry_projection",
    "validate_frame_document",
    "validate_plane",
    "validate_session_document",
]
