"""Deterministic single-path CPR (curved planar reformat) framing.

CORONARY_CENTERLINE_001, output O-2: a straightened view along ONE reviewable
coronary centerline path.  This is a NEW governed framing entry point - it is
NOT a :func:`visualization.mpr.reslice` call (that kernel admits only the
AXIAL/CORONAL/SAGITTAL planes).  Its sampling internals are the MPR kernel's,
reused verbatim (ADAPT, prior-art decision):

* ``_world_to_voxel`` - the staged binary32 WORLD->VOXEL evaluation;
* ``_sample_nearest`` / ``_sample_trilinear`` - the frozen samplers;
* ``_f32`` / ``_f32_mul`` / ``_f32_add`` - the binary32 discipline;
* ``_in_extent`` and ``output_dtype_policy`` - extent fill and dtype choice.

Framing geometry (frozen, deterministic - no RNG, no Frenet flips):

1. The centerline polyline is re-sampled at the uniform row step
   ``s_spacing_mm`` (rows ``i * s_spacing_mm``; a trailing remainder shorter
   than one step is not emitted).  Every row origin is the interpolated
   polyline point at that arc length.  F1: the polyline is the VALIDATED,
   DIGEST-BOUND sealed payload geometry of the CENTERLINE artifact - the live
   in-memory derivation array is never a framing source.
2. The tangent of row 0 is the normalized one-sided difference; interior
   tangents use central differences; the last one is one-sided.  Tangents are
   never allowed to vanish (typed abstention ``CPR_FRAMING_INVALID``).
3. The normal of row 0 is the world axis least aligned with the first
   tangent, projected into the cross-section plane (a deterministic,
   reproducible reference - the smallest ``argmin`` ties break on the lowest
   axis index).  At every later row the previous normal is advanced by the
   MINIMAL (Rodrigues) rotation that carries the previous tangent onto the
   current tangent, then re-orthogonalized: a parallel-transport
   rotation-minimizing frame with no flip ambiguity.
4. ``binormal = tangent x normal`` (right-handed); the cross-section plane is
   spanned by ``(normal, binormal)`` and every row is sampled along the
   **binormal** (the radial axis) at offsets
   ``(column - (radial_count - 1) / 2) * radial_spacing_mm``.

Pixels are indexed ``(s-index, radial-offset)`` exactly as the bound spec
requires, and the mapping back to world millimetres is the stored
``frame_sequence`` (round-trip tested).  The payload schema is
``ccta.cpr.view.v1`` under the same immutable-pixel-blob and codec discipline
as ``ccta.mpr.view.v1/v2``; the artifact is an ``MPR_VIEW`` produced by
``DETERMINISTIC_ENGINE`` (the CPR is a view, not a measurement and never a
SEGMENTATION artifact).

Rule C boundary: pixels and a frame sequence only - no diameter, area,
stenosis, CAD-RADS or FFR value exists anywhere in this module.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from typing import Any, Final, cast

import numpy as np

from apps.workstation.volume_loader import (
    CANONICAL_FLOAT32_CODEC,
    CANONICAL_INT16_CODEC,
    CanonicalVolumeLoader,
)
from core.artifacts import (
    ClaimBinding,
    DerivedArtifact,
    GeometryBase,
    Lineage,
    PayloadRef,
    Producer,
    QualityBlock,
    ReviewBlock,
)
from core.coordinates import (
    CoordinateTransform,
    FrameSpec,
    assert_round_trip,
    invert_matrix,
)
from core.coronary import (
    CORONARY_CLAIM_VERSION,
    CenterlineBuildResult,
    validate_centerline_payload,
)
from core.digest import canonical_digest, is_sha256_hex
from core.enums import (
    ArtifactKind,
    ProducerType,
    ProvenanceEventType,
    ReleaseLane,
    ReviewState,
    SpaceKind,
)
from core.errors import CanonicalValidationError
from core.events import utc_now_iso
from core.ids import new_ulid, validate_ulid
from core.provenance import (
    Actor,
    ActorType,
    BlobRef,
    Environment,
    InputRef,
    NotApplicable,
    ProvenanceRecord,
    create_provenance,
)
from visualization.mpr import (
    MPR_ARITHMETIC_ORDER,
    MPR_ROUNDING_RULE,
    MPR_VOXEL_CENTER_CONVENTION,
    MPRAbstentionError,
    MPRValidationError,
    _f32,
    _f32_add,
    _f32_mul,
    _in_extent,
    _sample_nearest,
    _sample_trilinear,
    _source_index_bounds,
    _source_is_exactly_integral,
    _world_to_voxel,
    output_dtype_policy,
)

#: The closed payload schema of the straightened-view set.
CPR_SCHEMA: Final[str] = "ccta.cpr.view.v1"
#: Claim binding (the accepted visualization claim; no new claim is invented).
CPR_CLAIM_ID: Final[str] = "CLM-VIZ-001"
CPR_CLAIM_VERSION: Final[str] = CORONARY_CLAIM_VERSION
#: Deterministic algorithm identity recorded in the producer block.
CPR_ALGORITHM: Final[str] = "coronary-cpr-numpy"
CPR_ALGORITHM_VERSION: Final[str] = "parallel-transport-rodrigues-resample-v1"
#: Frozen framing rules (recorded in the payload and the sampling digest).
CPR_FRAMING_RULE: Final[str] = "PARALLEL_TRANSPORT_RODRIGUES_MINIMAL_ROTATION_V1"
CPR_CROSS_SECTION_RULE: Final[str] = (
    "RIGHT_HANDED_TANGENT_NORMAL_BINORMAL_RADIAL_BINORMAL_V1"
)
#: Deterministic actor identity of the CPR framing.
CPR_ACTOR_ID: Final[str] = "ccta-cpr"
#: Canonical reference prefix of a CENTERLINE artifact.
_CENTERLINE_REF_PREFIX: Final[str] = "centerline_artifact_id:"

_ULID_CHARS = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
_UID_RE: Final[re.Pattern[str]] = re.compile(r"[0-9]+(?:\.[0-9]+)*\Z")
_CENTERLINE_REF_RE: Final[re.Pattern[str]] = re.compile(
    rf"centerline_artifact_id:[{_ULID_CHARS}]{{26}}\Z"
)
_SOURCE_VOLUME_REF_RE: Final[re.Pattern[str]] = re.compile(
    rf"volume_id:[{_ULID_CHARS}]{{26}}\Z"
)
_TRANSFORM_REF_RE: Final[re.Pattern[str]] = re.compile(
    rf"coordinate_transform:[{_ULID_CHARS}]{{26}}\Z"
)
_PROVENANCE_REF_RE: Final[re.Pattern[str]] = re.compile(
    rf"provenance:[{_ULID_CHARS}]{{26}}\Z"
)
_FRAME_REF_RE: Final[re.Pattern[str]] = re.compile(r"frame_spec:([0-9a-f]{64})\Z")

_PAYLOAD_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "centerline_ref",
        "centerline_payload_sha256",
        "source_frame_spec_ref",
        "source_volume_ref",
        "source_volume_blob_sha256",
        "source_volume_shape",
        "voxel_spacing_mm",
        "out_of_plane_ref",
        "voxel_center_convention",
        "framing_rule",
        "cross_section_rule",
        "method",
        "out_of_bounds_fill_value",
        "arithmetic_order",
        "output_dtype",
        "rounding_rule",
        "scalar_domain",
        "intensity_unit",
        "s_spacing_mm",
        "radial_spacing_mm",
        "radial_count",
        "matrix_size",
        "output_extent_mm",
        "frame_sequence",
        "sampling_parameters_sha256",
        "pixel_blob_ref",
        "study_instance_uid",
        "series_instance_uid",
        "frame_of_reference_uid",
        "provenance_ref",
    }
)

#: Exact per-frame key set of the stored frame sequence.
_FRAME_FIELDS: Final[frozenset[str]] = frozenset(
    {"s_mm", "origin_world_mm", "tangent", "normal", "binormal"}
)

_UNIT_TOLERANCE: Final[float] = 1e-6


class CPRValidationError(MPRValidationError):
    """A closed CPR payload contract was violated (fail-closed)."""


class CPRAbstentionError(MPRAbstentionError):
    """A requested CPR view cannot be produced safely and must abstain."""


#: Typed refusal code of the WS-02 session-store / viewer-panel seam (F6):
#: ``ccta.cpr.view.v1`` is a complete view set but the frozen store contract
#: does not admit it yet, so that integration route abstains explicitly.
CPR_SESSION_INTEGRATION_UNAVAILABLE: Final[str] = "CPR_SESSION_INTEGRATION_UNAVAILABLE"


class CPRSessionIntegrationRefusal(CPRValidationError):
    """Typed abstention of the WS-02 persistence / viewer-panel seam (F6)."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__([code, detail])


@dataclass(frozen=True)
class CPRBuildResult:
    """The complete persisted representation of one straightened view."""

    artifact: DerivedArtifact
    payload: dict[str, object]
    pixels: np.ndarray
    pixel_blob_sha256: str
    pixel_blob_byte_size: int

    @property
    def pixel_blob_ref(self) -> dict[str, object]:
        return cast(dict[str, object], self.payload["pixel_blob_ref"])

    @property
    def matrix_size(self) -> tuple[int, int]:
        rows, columns = cast(list[int], self.payload["matrix_size"])
        return (int(rows), int(columns))


# ---------------------------------------------------------------------------
# Framing geometry (parallel transport, deterministic)
# ---------------------------------------------------------------------------


def resample_frames(
    points_world_mm: np.ndarray,
    s_mm: np.ndarray,
    s_spacing_mm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(s_values, origins, tangents, normals, binormals)``.

    ``origins`` are the polyline points at the uniform arc lengths
    ``i * s_spacing_mm``; all returned arrays are float64 of shape
    ``(row_count, 3)`` (``s_values`` is ``(row_count,)``).  Every degenerate
    input is a typed abstention, never a silently repaired frame.
    """

    points = np.asarray(points_world_mm, dtype=np.float64)
    arcs = np.asarray(s_mm, dtype=np.float64)
    spacing = float(s_spacing_mm)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 2:
        raise CPRAbstentionError("CPR_FRAMING_INVALID", "POLYLINE_SHAPE_INVALID")
    if arcs.ndim != 1 or arcs.shape[0] != points.shape[0]:
        raise CPRAbstentionError("CPR_FRAMING_INVALID", "ARC_LENGTH_SHAPE_INVALID")
    if not math.isfinite(spacing) or spacing <= 0.0:
        raise CPRAbstentionError("CPR_FRAMING_INVALID", "S_SPACING_INVALID")
    if not bool(np.all(np.isfinite(points))) or not bool(np.all(np.isfinite(arcs))):
        raise CPRAbstentionError("CPR_FRAMING_INVALID", "NONFINITE_PATH")
    if float(arcs[-1]) <= 0.0:
        raise CPRAbstentionError("CPR_FRAMING_INVALID", "DEGENERATE_FRAMING_PATH")
    row_count = int(math.floor(float(arcs[-1]) / spacing)) + 1
    s_values = np.arange(row_count, dtype=np.float64) * spacing
    origins = np.empty((row_count, 3), dtype=np.float64)
    for index in range(row_count):
        s = float(s_values[index])
        segment = int(np.searchsorted(arcs, s, side="right") - 1)
        segment = min(max(segment, 0), points.shape[0] - 2)
        span = float(arcs[segment + 1] - arcs[segment])
        fraction = 0.0 if span <= 0.0 else (s - float(arcs[segment])) / span
        origins[index] = points[segment] + fraction * (
            points[segment + 1] - points[segment]
        )

    tangents = np.empty_like(origins)
    if row_count == 1:
        overall = points[-1] - points[0]
        norm = float(np.linalg.norm(overall))
        if norm == 0.0:
            raise CPRAbstentionError("CPR_FRAMING_INVALID", "DEGENERATE_FRAMING_PATH")
        tangents[0] = overall / norm
    else:
        tangents[0] = origins[1] - origins[0]
        for index in range(1, row_count - 1):
            tangents[index] = origins[index + 1] - origins[index - 1]
        tangents[-1] = origins[-1] - origins[-2]
        for index in range(row_count):
            norm = float(np.linalg.norm(tangents[index]))
            if norm == 0.0:
                raise CPRAbstentionError(
                    "CPR_FRAMING_INVALID", "DEGENERATE_TANGENT"
                )
            tangents[index] = tangents[index] / norm

    normals = np.empty_like(origins)
    normals[0] = _reference_normal(tangents[0])
    for index in range(1, row_count):
        rotated = _minimal_rotation(
            normals[index - 1], tangents[index - 1], tangents[index]
        )
        projected = rotated - float(np.dot(rotated, tangents[index])) * tangents[index]
        norm = float(np.linalg.norm(projected))
        if norm == 0.0:
            raise CPRAbstentionError(
                "CPR_FRAMING_INVALID", "PARALLEL_TRANSPORT_DEGENERATE"
            )
        normals[index] = projected / norm
    binormals = np.cross(tangents, normals)
    return (s_values, origins, tangents, normals, binormals)


def _reference_normal(tangent: np.ndarray) -> np.ndarray:
    """The world axis least aligned with the tangent, projected and normalized."""

    axis_index = int(np.argmin(np.abs(tangent)))
    axis = np.zeros(3, dtype=np.float64)
    axis[axis_index] = 1.0
    projected = axis - float(np.dot(axis, tangent)) * tangent
    norm = float(np.linalg.norm(projected))
    if norm == 0.0:
        raise CPRAbstentionError("CPR_FRAMING_INVALID", "REFERENCE_NORMAL_DEGENERATE")
    return projected / norm


def _minimal_rotation(
    vector: np.ndarray, before: np.ndarray, after: np.ndarray
) -> np.ndarray:
    """Rodrigues minimal rotation of ``vector`` from ``before`` onto ``after``.

    The half-angle stable form ``R v = v + (k x v) + [k]_x^2 v / (1 + c)`` is
    used with ``k = before x after`` (not normalized) and ``c = before·after``;
    an antiparallel pair is a typed abstention (it has no unique minimal
    rotation).
    """

    cross = np.cross(before, after)
    cosine = float(np.dot(before, after))
    if cosine <= -1.0 + 1e-12:
        raise CPRAbstentionError(
            "CPR_FRAMING_INVALID", "PARALLEL_TRANSPORT_ANTIPARALLEL"
        )
    return (
        vector
        + np.cross(cross, vector)
        + np.cross(cross, np.cross(cross, vector)) / (1.0 + cosine)
    )


# ---------------------------------------------------------------------------
# Resampling kernel (MPR internals reused verbatim)
# ---------------------------------------------------------------------------


def resample_cross_sections(
    source: np.ndarray,
    world_to_voxel: CoordinateTransform,
    *,
    s_values: np.ndarray,
    origins: np.ndarray,
    binormals: np.ndarray,
    s_spacing_mm: float,
    radial_spacing_mm: float,
    radial_count: int,
    method: str,
    out_of_bounds_fill_value: float,
    source_values_exactly_integral: bool | None = None,
) -> tuple[np.ndarray, str]:
    """Pure-NumPy CPR kernel returning ``(pixels, codec)``.

    This mirrors the :func:`visualization.mpr.reslice` scaffold exactly - the
    destination is fully initialized, every pixel is written exactly once in
    row-major order, and each world sample is generated with the same staged
    binary32 arithmetic.  The only difference is the grid generation: row
    ``r`` has origin ``origins[r]`` and the column direction is the frame's
    ``binormal`` (the radial axis), so the sample at ``(r, c)`` is
    ``origins[r] + ((c - (radial_count - 1) / 2) * radial_spacing_mm) *
    binormals[r]``.
    """

    if not isinstance(source, np.ndarray) or source.ndim != 3:
        raise CPRAbstentionError("CPR_SOURCE_INVALID", "VOLUME_SHAPE_INVALID")
    if not isinstance(world_to_voxel, CoordinateTransform):
        raise CPRAbstentionError("CPR_TRANSFORM_INVALID", "WORLD_TO_VOXEL_REQUIRED")
    if source.dtype.kind not in {"i", "u", "f"}:
        raise CPRAbstentionError("CPR_SOURCE_INVALID", "VOLUME_DTYPE_INVALID")
    if any(int(dimension) <= 0 for dimension in source.shape):
        raise CPRAbstentionError("CPR_SOURCE_INVALID", "VOLUME_SHAPE_INVALID")
    for value in source.flat:
        try:
            numeric_value = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise CPRAbstentionError("CPR_SOURCE_INVALID", "NONFINITE_SOURCE") from exc
        if not math.isfinite(numeric_value):
            raise CPRAbstentionError("CPR_SOURCE_INVALID", "NONFINITE_SOURCE")
        if abs(numeric_value) > float(np.finfo(np.float32).max):
            raise CPRAbstentionError("CPR_SOURCE_INVALID", "FLOAT32_RANGE")
    if (
        world_to_voxel.from_space is not SpaceKind.WORLD
        or world_to_voxel.to_space is not SpaceKind.VOXEL
    ):
        raise CPRAbstentionError("CPR_TRANSFORM_INVALID", "WORLD_TO_VOXEL_REQUIRED")
    for matrix_row in world_to_voxel.matrix:
        for value in matrix_row:
            if not math.isfinite(float(value)):
                raise CPRAbstentionError("CPR_TRANSFORM_INVALID", "NONFINITE_MATRIX")
    if source_values_exactly_integral is not None and not isinstance(
        source_values_exactly_integral, bool
    ):
        raise CPRValidationError("source_values_exactly_integral must be a boolean")
    origins_array = np.asarray(origins, dtype=np.float64)
    binormals_array = np.asarray(binormals, dtype=np.float64)
    s_array = np.asarray(s_values, dtype=np.float64)
    rows = int(origins_array.shape[0])
    if (
        origins_array.shape != (rows, 3)
        or binormals_array.shape != (rows, 3)
        or s_array.shape != (rows,)
    ):
        raise CPRValidationError("frame arrays must be (rows, 3) and (rows,)")
    if (
        isinstance(radial_count, bool)
        or not isinstance(radial_count, int)
        or not 1 <= radial_count <= 2**31 - 1
    ):
        raise CPRValidationError("radial_count must be a positive 32-bit integer")
    if not isinstance(s_spacing_mm, Real) or isinstance(s_spacing_mm, bool):
        raise CPRValidationError("s_spacing_mm must be positive and finite")
    if not isinstance(radial_spacing_mm, Real) or isinstance(radial_spacing_mm, bool):
        raise CPRValidationError("radial_spacing_mm must be positive and finite")
    s_spacing = float(s_spacing_mm)
    radial_spacing = float(radial_spacing_mm)
    if not math.isfinite(s_spacing) or s_spacing <= 0.0:
        raise CPRValidationError("s_spacing_mm must be positive and finite")
    if not math.isfinite(radial_spacing) or radial_spacing <= 0.0:
        raise CPRValidationError("radial_spacing_mm must be positive and finite")
    if not isinstance(method, str) or method not in {"NEAREST", "TRILINEAR"}:
        raise CPRValidationError("method must be NEAREST or TRILINEAR")
    if (
        isinstance(out_of_bounds_fill_value, bool)
        or not isinstance(out_of_bounds_fill_value, Real)
        or not math.isfinite(float(out_of_bounds_fill_value))
    ):
        raise CPRValidationError("out_of_bounds_fill_value must be finite")
    fill = _f32(out_of_bounds_fill_value)
    if not bool(np.all(np.isfinite(origins_array))) or not bool(
        np.all(np.isfinite(binormals_array))
    ):
        raise CPRValidationError("frame arrays must be finite")

    try:
        output = np.full((rows, radial_count), fill, dtype=np.dtype("<f4"), order="C")
    except (MemoryError, OverflowError, ValueError) as exc:
        raise CPRAbstentionError(
            "CPR_OUTPUT_INVALID", "OUTPUT_ALLOCATION_FAILED"
        ) from exc
    dimensions = _source_index_bounds(source)
    center_index = _f32((radial_count - 1) / 2.0)
    spacing_f32 = _f32(radial_spacing)
    for row in range(rows):
        origin = origins_array[row]
        binormal = binormals_array[row]
        for column in range(radial_count):
            column_shift = _f32(_f32(column) - center_index)
            radial_distance = _f32_mul(column_shift, spacing_f32)
            world: list[np.float32] = [
                _f32_add(_f32(origin[axis]), _f32_mul(radial_distance, _f32(binormal[axis])))
                for axis in range(3)
            ]
            q = _world_to_voxel(world_to_voxel.matrix, world)
            if not _in_extent(q, dimensions):
                output[row, column] = fill
            elif method == "NEAREST":
                output[row, column] = _sample_nearest(source, q)
            else:
                output[row, column] = _sample_trilinear(source, q)

    fill_is_exact_int16 = (
        float(out_of_bounds_fill_value) == math.trunc(float(out_of_bounds_fill_value))
        and -32768 <= float(out_of_bounds_fill_value) <= 32767
    )
    selected_dtype = (
        output_dtype_policy(source, output, method=method)
        if source_values_exactly_integral is None
        else output_dtype_policy(
            source,
            output,
            method=method,
            source_values_exactly_integral=source_values_exactly_integral,
        )
        if fill_is_exact_int16
        else "float32"
    )
    if selected_dtype == "int16":
        integer_output = np.asarray(output, dtype=np.dtype("<i2"), order="C")
        integer_output.setflags(write=False)
        return integer_output, CANONICAL_INT16_CODEC
    if not all(math.isfinite(float(value)) for value in output.flat):
        raise CPRAbstentionError("CPR_ARITHMETIC_INVALID", "NONFINITE_OUTPUT")
    output = np.asarray(output, dtype=np.dtype("<f4"), order="C")
    output.setflags(write=False)
    return output, CANONICAL_FLOAT32_CODEC


# ---------------------------------------------------------------------------
# Payload assembly + validation
# ---------------------------------------------------------------------------


def _sampling_parameters(payload: Mapping[str, object]) -> dict[str, object]:
    """The canonical sampling-parameter object of the CPR digest rule."""

    transform_ref = cast(dict[str, object], payload["out_of_plane_ref"])
    document: dict[str, object] = {
        "centerline_ref": payload["centerline_ref"],
        "centerline_payload_sha256": payload["centerline_payload_sha256"],
        "source_frame_spec_ref": payload["source_frame_spec_ref"],
        "source_volume_ref": payload["source_volume_ref"],
        "source_volume_blob_sha256": payload["source_volume_blob_sha256"],
        "source_volume_shape": payload["source_volume_shape"],
        "voxel_spacing_mm": payload["voxel_spacing_mm"],
        "transform_ref": transform_ref["transform_artifact_ref"],
        "transform_version": transform_ref["transform_version"],
        "framing_rule": payload["framing_rule"],
        "cross_section_rule": payload["cross_section_rule"],
        "method": payload["method"],
        "out_of_bounds_fill_value": payload["out_of_bounds_fill_value"],
        "arithmetic_order": payload["arithmetic_order"],
        "output_dtype": payload["output_dtype"],
        "rounding_rule": payload["rounding_rule"],
        "scalar_domain": payload["scalar_domain"],
        "s_spacing_mm": payload["s_spacing_mm"],
        "radial_spacing_mm": payload["radial_spacing_mm"],
        "radial_count": payload["radial_count"],
        "matrix_size": payload["matrix_size"],
        "output_extent_mm": payload["output_extent_mm"],
        "frame_sequence": payload["frame_sequence"],
    }
    return document


def sampling_parameters_sha256(payload: Mapping[str, object]) -> str:
    """Return the canonical digest over a payload's sampling fields."""

    return canonical_digest(_sampling_parameters(payload))


def _require_ref(value: object, pattern: re.Pattern[str], label: str) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise CPRValidationError(f"{label} has the wrong canonical reference form")
    return value


def _require_dicom_uid(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or not 1 <= len(value) <= 64
        or _UID_RE.fullmatch(value) is None
    ):
        raise CPRValidationError(f"{label} must be a DICOM UID")
    return value


def _require_vec(value: object, length: int, label: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise CPRValidationError(f"{label} must have length {length}")
    result: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, Real):
            raise CPRValidationError(f"{label} must contain finite numbers")
        try:
            number = float(item)
        except (TypeError, ValueError, OverflowError) as exc:
            raise CPRValidationError(f"{label} must contain finite numbers") from exc
        if not math.isfinite(number):
            raise CPRValidationError(f"{label} must contain finite numbers")
        result.append(number)
    return tuple(result)


def _require_orthonormal_frame(
    tangent: tuple[float, ...],
    normal: tuple[float, ...],
    binormal: tuple[float, ...],
    label: str,
) -> None:
    for name, vector in (("tangent", tangent), ("normal", normal), ("binormal", binormal)):
        norm = math.sqrt(sum(component * component for component in vector))
        if abs(norm - 1.0) > _UNIT_TOLERANCE:
            raise CPRValidationError(f"{label}.{name} is not a unit vector")
    for left_name, left, right_name, right in (
        ("tangent", tangent, "normal", normal),
        ("tangent", tangent, "binormal", binormal),
        ("normal", normal, "binormal", binormal),
    ):
        if abs(sum(a * b for a, b in zip(left, right))) > _UNIT_TOLERANCE:
            raise CPRValidationError(
                f"{label}.{left_name} and {label}.{right_name} are not orthogonal"
            )
    cross = (
        tangent[1] * normal[2] - tangent[2] * normal[1],
        tangent[2] * normal[0] - tangent[0] * normal[2],
        tangent[0] * normal[1] - tangent[1] * normal[0],
    )
    if max(abs(a - b) for a, b in zip(cross, binormal)) > _UNIT_TOLERANCE:
        raise CPRValidationError(
            f"{label} is not a right-handed (tangent, normal, binormal) frame"
        )


def validate_cpr_payload(payload: Mapping[str, object]) -> dict[str, object]:
    """Validate the closed ``ccta.cpr.view.v1`` payload exactly."""

    if not isinstance(payload, Mapping):
        raise CPRValidationError("CPR payload must be an object")
    if any(not isinstance(key, str) for key in payload):
        raise CPRValidationError("CPR payload keys must be strings")
    keys = set(payload)
    missing = sorted(_PAYLOAD_FIELDS - keys)
    unknown = sorted(keys - _PAYLOAD_FIELDS)
    if missing or unknown:
        problems = []
        if missing:
            problems.append(f"missing fields: {', '.join(missing)}")
        if unknown:
            problems.append(f"unknown fields: {', '.join(unknown)}")
        raise CPRValidationError(problems)
    if payload["schema_version"] != CPR_SCHEMA:
        raise CPRValidationError("schema_version must be ccta.cpr.view.v1")
    _require_ref(payload["centerline_ref"], _CENTERLINE_REF_RE, "centerline_ref")
    if not is_sha256_hex(payload["centerline_payload_sha256"]):
        raise CPRValidationError("centerline_payload_sha256 must be lowercase sha256")
    _require_ref(
        payload["source_frame_spec_ref"], _FRAME_REF_RE, "source_frame_spec_ref"
    )
    _require_ref(payload["source_volume_ref"], _SOURCE_VOLUME_REF_RE, "source_volume_ref")
    if not is_sha256_hex(payload["source_volume_blob_sha256"]):
        raise CPRValidationError("source_volume_blob_sha256 must be lowercase sha256")
    shape = payload["source_volume_shape"]
    if not isinstance(shape, (list, tuple)) or len(shape) != 3:
        raise CPRValidationError("source_volume_shape must be 3-D")
    for axis, value in enumerate(shape):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise CPRValidationError(
                f"source_volume_shape[{axis}] must be a positive integer"
            )
    spacing_mm = _require_vec(payload["voxel_spacing_mm"], 3, "voxel_spacing_mm")
    if any(value <= 0.0 for value in spacing_mm):
        raise CPRValidationError("voxel_spacing_mm must be positive")
    transform = payload["out_of_plane_ref"]
    if isinstance(transform, Mapping) and any(
        not isinstance(key, str) for key in transform
    ):
        raise CPRValidationError("out_of_plane_ref keys must be strings")
    if not isinstance(transform, Mapping) or set(transform) != {
        "transform_artifact_ref",
        "transform_version",
        "direction",
    }:
        raise CPRValidationError("out_of_plane_ref has an unexpected shape")
    _require_ref(
        transform["transform_artifact_ref"],
        _TRANSFORM_REF_RE,
        "transform_artifact_ref",
    )
    version = transform["transform_version"]
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise CPRValidationError("transform_version must be a positive integer")
    if transform["direction"] != "WORLD_TO_VOXEL":
        raise CPRValidationError("out_of_plane_ref direction is fixed")
    if payload["voxel_center_convention"] != MPR_VOXEL_CENTER_CONVENTION:
        raise CPRValidationError("voxel_center_convention is fixed")
    if payload["framing_rule"] != CPR_FRAMING_RULE:
        raise CPRValidationError("framing_rule is fixed")
    if payload["cross_section_rule"] != CPR_CROSS_SECTION_RULE:
        raise CPRValidationError("cross_section_rule is fixed")
    method = payload["method"]
    if not isinstance(method, str) or method not in {"NEAREST", "TRILINEAR"}:
        raise CPRValidationError("method is outside the closed enum")
    fill = payload["out_of_bounds_fill_value"]
    if (
        isinstance(fill, bool)
        or not isinstance(fill, (int, float))
        or not math.isfinite(float(fill))
    ):
        raise CPRValidationError("out_of_bounds_fill_value must be finite")
    fill_number = float(fill)
    if abs(fill_number) > float(np.finfo(np.float32).max):
        raise CPRValidationError("out_of_bounds_fill_value overflows float32")
    if payload["arithmetic_order"] != MPR_ARITHMETIC_ORDER:
        raise CPRValidationError("arithmetic_order is fixed")
    output_dtype = payload["output_dtype"]
    if not isinstance(output_dtype, str) or output_dtype not in {"int16", "float32"}:
        raise CPRValidationError("output_dtype is outside the closed enum")
    if output_dtype == "int16":
        if method != "NEAREST":
            raise CPRValidationError("int16 output is legal only for NEAREST")
        if (
            fill_number != math.trunc(fill_number)
            or fill_number < -32768
            or fill_number > 32767
        ):
            raise CPRValidationError("int16 output cannot represent the fill value")
    if payload["rounding_rule"] != MPR_ROUNDING_RULE:
        raise CPRValidationError("rounding_rule is fixed")
    scalar_domain = payload["scalar_domain"]
    if not isinstance(scalar_domain, str) or scalar_domain not in {
        "STORED",
        "MODALITY_TRANSFORMED",
        "HU",
    }:
        raise CPRValidationError("scalar_domain is outside the closed vocabulary")
    if payload["intensity_unit"] != scalar_domain:
        raise CPRValidationError("intensity_unit must equal scalar_domain")
    s_spacing = payload["s_spacing_mm"]
    if (
        isinstance(s_spacing, bool)
        or not isinstance(s_spacing, Real)
        or not math.isfinite(float(s_spacing))
        or float(s_spacing) <= 0.0
    ):
        raise CPRValidationError("s_spacing_mm must be positive and finite")
    radial_spacing = payload["radial_spacing_mm"]
    if (
        isinstance(radial_spacing, bool)
        or not isinstance(radial_spacing, Real)
        or not math.isfinite(float(radial_spacing))
        or float(radial_spacing) <= 0.0
    ):
        raise CPRValidationError("radial_spacing_mm must be positive and finite")
    radial_count = payload["radial_count"]
    if (
        isinstance(radial_count, bool)
        or not isinstance(radial_count, int)
        or not 1 <= radial_count <= 2**31 - 1
    ):
        raise CPRValidationError("radial_count must be a positive integer")
    matrix_size = payload["matrix_size"]
    if not isinstance(matrix_size, (list, tuple)) or len(matrix_size) != 2:
        raise CPRValidationError("matrix_size must be a two-vector")
    rows, columns = 0, 0
    for axis, value in enumerate(matrix_size):
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 2**31 - 1:
            raise CPRValidationError("matrix_size must contain positive integers")
        if axis == 0:
            rows = value
        else:
            columns = value
    if columns != radial_count:
        raise CPRValidationError("matrix_size columns must equal radial_count")
    extent = _require_vec(payload["output_extent_mm"], 2, "output_extent_mm")
    expected_extent = (rows * float(s_spacing), columns * float(radial_spacing))
    if extent != expected_extent:
        raise CPRValidationError("output_extent_mm is not the exact cell extent")
    frames = payload["frame_sequence"]
    if not isinstance(frames, list) or len(frames) != rows:
        raise CPRValidationError("frame_sequence must carry exactly one frame per row")
    previous_s = -1.0
    for index, frame in enumerate(frames):
        if not isinstance(frame, Mapping) or set(frame) != _FRAME_FIELDS:
            raise CPRValidationError(f"frame_sequence[{index}] is not a closed frame")
        s_value = frame["s_mm"]
        if (
            isinstance(s_value, bool)
            or not isinstance(s_value, (int, float))
            or not math.isfinite(float(s_value))
        ):
            raise CPRValidationError(f"frame_sequence[{index}].s_mm must be finite")
        expected_s = index * float(s_spacing)
        if abs(float(s_value) - expected_s) > 1e-9 * max(1.0, expected_s):
            raise CPRValidationError(
                f"frame_sequence[{index}].s_mm is not the uniform row position"
            )
        if float(s_value) <= previous_s:
            raise CPRValidationError("frame_sequence s_mm order is not strict")
        previous_s = float(s_value)
        origin = _require_vec(
            frame["origin_world_mm"], 3, f"frame_sequence[{index}].origin_world_mm"
        )
        tangent = _require_vec(
            frame["tangent"], 3, f"frame_sequence[{index}].tangent"
        )
        normal = _require_vec(frame["normal"], 3, f"frame_sequence[{index}].normal")
        binormal = _require_vec(
            frame["binormal"], 3, f"frame_sequence[{index}].binormal"
        )
        del origin  # position is carried; no closed-form check is possible
        _require_orthonormal_frame(tangent, normal, binormal, f"frame_sequence[{index}]")
    if not is_sha256_hex(payload["sampling_parameters_sha256"]):
        raise CPRValidationError("sampling_parameters_sha256 must be lowercase sha256")
    if sampling_parameters_sha256(payload) != payload["sampling_parameters_sha256"]:
        raise CPRValidationError(
            "sampling_parameters_sha256 does not bind the sampling fields"
        )
    pixel_ref = payload["pixel_blob_ref"]
    if isinstance(pixel_ref, Mapping) and any(not isinstance(key, str) for key in pixel_ref):
        raise CPRValidationError("pixel_blob_ref keys must be strings")
    if not isinstance(pixel_ref, Mapping) or set(pixel_ref) != {
        "sha256",
        "codec",
        "byte_size",
    }:
        raise CPRValidationError("pixel_blob_ref has an unexpected shape")
    if not is_sha256_hex(pixel_ref["sha256"]):
        raise CPRValidationError("pixel_blob_ref.sha256 must be lowercase sha256")
    expected_codec = "int16-le" if output_dtype == "int16" else "float32-le"
    if pixel_ref["codec"] != expected_codec:
        raise CPRValidationError("pixel_blob_ref codec disagrees with output_dtype")
    byte_size = pixel_ref["byte_size"]
    expected_byte_size = rows * columns * (2 if expected_codec == "int16-le" else 4)
    if (
        isinstance(byte_size, bool)
        or not isinstance(byte_size, int)
        or byte_size != expected_byte_size
        or byte_size <= 0
    ):
        raise CPRValidationError("pixel_blob_ref.byte_size is incorrect")
    for label in ("study_instance_uid", "series_instance_uid", "frame_of_reference_uid"):
        _require_dicom_uid(payload[label], label)
    _require_ref(payload["provenance_ref"], _PROVENANCE_REF_RE, "provenance_ref")
    return dict(payload)


def decode_cpr_pixel_blob(payload: Mapping[str, object], data: bytes) -> np.ndarray:
    """Validate and decode one persisted CPR pixel blob without mutation."""

    validated = validate_cpr_payload(payload)
    if not isinstance(data, bytes):
        raise CPRValidationError("pixel blob must be immutable bytes")
    pixel_ref = cast(dict[str, object], validated["pixel_blob_ref"])
    digest = cast(str, pixel_ref["sha256"])
    if hashlib.sha256(data).hexdigest() != digest:
        raise CPRValidationError("pixel blob content does not match pixel_blob_ref")
    matrix_size = validated["matrix_size"]
    if not isinstance(matrix_size, (list, tuple)) or len(matrix_size) != 2:
        raise CPRValidationError("matrix_size is not a canonical two-vector")
    rows, columns = int(matrix_size[0]), int(matrix_size[1])
    dtype = np.dtype("<i2" if pixel_ref["codec"] == "int16-le" else "<f4")
    expected = rows * columns * dtype.itemsize
    if len(data) != expected or pixel_ref["byte_size"] != expected:
        raise CPRValidationError("pixel blob byte_size does not match matrix_size")
    pixels = np.frombuffer(data, dtype=dtype).copy().reshape((rows, columns))
    if not bool(np.all(np.isfinite(pixels))):
        raise CPRValidationError("pixel blob contains nonfinite values")
    pixels.setflags(write=False)
    return pixels


# ---------------------------------------------------------------------------
# WS-02 persistence seam (F6 abstention RETIRED by CORONARY_REVIEW_VIEW_V1)
# ---------------------------------------------------------------------------


def require_ws02_snapshot_admission() -> None:
    """Retired gate: the WS-02 store now admits ``ccta.cpr.view.v1``.

    CORONARY_REVIEW_VIEW_V1 (Decision B) delivered the governed amendment
    this gate named: ``apps/workstation/session_store.py`` routes an
    ``MPR_VIEW`` whose outer ``payload_schema`` is ``ccta.cpr.view.v1``
    through ``validate_cpr_payload`` + ``decode_cpr_pixel_blob`` with the
    same digest checks as the MPR path (which is byte-identical).  This
    function is RETAINED ONLY so historical callers fail with a visible
    typed refusal pointing at the new seam instead of silently succeeding;
    new code must persist CPR views through ``DerivedStateStoreManager``
    directly.  The retired refusal code
    ``CPR_SESSION_INTEGRATION_UNAVAILABLE`` MUST NOT be raised on any live
    path (no dead refusal path).
    """

    raise CPRSessionIntegrationRefusal(
        CPR_SESSION_INTEGRATION_UNAVAILABLE,
        "retired: persist ccta.cpr.view.v1 through DerivedStateStoreManager "
        "(CORONARY_REVIEW_VIEW_V1 Decision B); this gate no longer guards a live path",
    )


# ---------------------------------------------------------------------------
# Reference normalization (MPR/CENTERLINE precedent)
# ---------------------------------------------------------------------------


def _normalize_source_volume_ref(volume_id: str) -> str:
    if not isinstance(volume_id, str):
        raise CPRValidationError("source Volume id must be a ULID")
    if volume_id.startswith("volume_id:"):
        _require_ref(volume_id, _SOURCE_VOLUME_REF_RE, "source_volume_ref")
        return volume_id
    try:
        validate_ulid(volume_id)
    except (TypeError, ValueError) as exc:
        raise CPRValidationError("source Volume id must be a ULID") from exc
    return f"volume_id:{volume_id}"


def _normalize_transform_ref(transform: CoordinateTransform, value: str | None) -> str:
    ref = transform.transform_id if value is None else value
    if not isinstance(ref, str):
        raise CPRValidationError(
            "transform_artifact_ref is required as coordinate_transform:<ULID>"
        )
    if not ref.startswith("coordinate_transform:"):
        try:
            validate_ulid(ref)
        except (TypeError, ValueError) as exc:
            raise CPRValidationError(
                "transform_artifact_ref is required as coordinate_transform:<ULID>"
            ) from exc
        ref = f"coordinate_transform:{ref}"
    _require_ref(ref, _TRANSFORM_REF_RE, "transform_artifact_ref")
    return ref


def _normalize_provenance_ref(value: str) -> tuple[str, str]:
    if not isinstance(value, str):
        raise CPRValidationError("provenance_ref must be provenance:<ULID>")
    if value.startswith("provenance:"):
        ref = _require_ref(value, _PROVENANCE_REF_RE, "provenance_ref")
        return ref, ref.split(":", 1)[1]
    try:
        validate_ulid(value)
    except (TypeError, ValueError) as exc:
        raise CPRValidationError("provenance_ref must be provenance:<ULID>") from exc
    return f"provenance:{value}", value


def _normalize_transform(transform: CoordinateTransform) -> CoordinateTransform:
    if transform.from_space is SpaceKind.WORLD and transform.to_space is SpaceKind.VOXEL:
        return transform
    if transform.from_space is SpaceKind.VOXEL and transform.to_space is SpaceKind.WORLD:
        return transform.inverse
    raise CPRAbstentionError("CPR_TRANSFORM_INVALID", "WORLD_TO_VOXEL_REQUIRED")


def _validate_transform_for_frame(
    transform: CoordinateTransform, frame: FrameSpec
) -> None:
    try:
        actual_world_to_voxel = _normalize_transform(transform)
        expected_world_to_voxel = invert_matrix(frame.voxel_to_world_matrix())
    except (CanonicalValidationError, TypeError, ValueError, OverflowError) as exc:
        raise CPRAbstentionError("CPR_TRANSFORM_INVALID", "FRAME_AFFINE_INVALID") from exc
    if actual_world_to_voxel.matrix != expected_world_to_voxel:
        raise CPRAbstentionError("CPR_TRANSFORM_INVALID", "FRAME_AFFINE_MISMATCH")


def _normalize_centerline(
    centerline: CenterlineBuildResult,
    *,
    source_volume_ref: str,
) -> tuple[DerivedArtifact, dict[str, object]]:
    """Require the persisted CENTERLINE this CPR view is framed on.

    The sealed payload is re-validated and its digest must equal the
    artifact's payload digest; the centerline must have been derived from the
    SAME source volume (pixel resampling from another volume would be a silent
    cross-volume fallback, Rule 10).  F1: the framing geometry is taken from
    THIS validated sealed document - the live in-memory path carried by the
    build result is never a geometry source, so tampering with it cannot move
    a CPR frame.
    """

    if not isinstance(centerline, CenterlineBuildResult):
        raise CPRValidationError("centerline must be a CenterlineBuildResult")
    artifact = centerline.artifact
    if artifact.kind is not ArtifactKind.CENTERLINE:
        raise CPRValidationError("centerline artifact kind must be CENTERLINE")
    document = validate_centerline_payload(centerline.payload)
    if canonical_digest(document) != artifact.payload.blob_sha256:
        raise CPRValidationError("centerline payload digest disagrees with its artifact")
    if document.get("source_volume_ref") != source_volume_ref:
        raise CPRValidationError(
            "the centerline must be derived from the same source volume"
        )
    return artifact, document


# ---------------------------------------------------------------------------
# Build entry point
# ---------------------------------------------------------------------------


def build_cpr_view(
    source_volume: Any,
    transform: CoordinateTransform,
    *,
    centerline: CenterlineBuildResult,
    s_spacing_mm: float,
    radial_spacing_mm: float,
    radial_count: int,
    method: str,
    out_of_bounds_fill_value: float,
    loader: CanonicalVolumeLoader | None = None,
    source_array: np.ndarray | None = None,
    transform_artifact_ref: str | None = None,
    provenance_ref: str,
    study_instance_uid: str,
    series_instance_uid: str | None = None,
    frame_of_reference_uid: str | None = None,
    claim_version: str = CPR_CLAIM_VERSION,
    release_lane: ReleaseLane | None = None,
    created_by: str = CPR_ACTOR_ID,
    artifact_id: str | None = None,
    provenance_record_ref: str | None = None,
) -> CPRBuildResult:
    """Build and persist ONE validated CPR ``MPR_VIEW`` artifact.

    ``source_volume`` is a canonical :class:`core.volume.Volume`; ``loader``
    decodes the canonical blob and persists the immutable pixel blob.
    ``centerline`` is the sealed CENTERLINE build result this view is framed
    on; ``provenance_ref`` is mandatory and must resolve to the caller's §44
    provenance record (this function never fabricates provenance).  The
    framing parameters (``framing_rule``/``cross_section_rule``) are fixed
    constants, never caller options.
    """

    from core.volume import Volume

    if not isinstance(source_volume, Volume):
        raise CPRValidationError("source_volume must be a canonical Volume")
    if not isinstance(transform, CoordinateTransform):
        raise CPRValidationError("transform must be a persisted CoordinateTransform")
    if not isinstance(s_spacing_mm, Real) or isinstance(s_spacing_mm, bool):
        raise CPRValidationError("s_spacing_mm must be positive and finite")
    if not isinstance(radial_spacing_mm, Real) or isinstance(radial_spacing_mm, bool):
        raise CPRValidationError("radial_spacing_mm must be positive and finite")
    if (
        isinstance(radial_count, bool)
        or not isinstance(radial_count, int)
        or radial_count < 1
    ):
        raise CPRValidationError("radial_count must be a positive integer")
    if loader is None:
        raise CPRValidationError("loader is required to persist the CPR pixel blob")
    if not isinstance(claim_version, str) or not claim_version:
        raise CPRValidationError("claim_version is required")
    if not isinstance(created_by, str) or not created_by:
        raise CPRValidationError("created_by is required")
    artifact_ulid = artifact_id or new_ulid()
    try:
        validate_ulid(artifact_ulid)
    except (TypeError, ValueError) as exc:
        raise CPRValidationError("artifact_id must be a ULID") from exc
    if release_lane is not None and not isinstance(release_lane, ReleaseLane):
        raise CPRValidationError("release_lane must be a ReleaseLane")
    if not isinstance(source_volume.study_ref, str) or not source_volume.study_ref:
        raise CPRValidationError("source_volume.study_ref is required")
    _require_dicom_uid(study_instance_uid, "study_instance_uid")
    if (
        not isinstance(transform.version, int)
        or isinstance(transform.version, bool)
        or transform.version < 1
    ):
        raise CPRValidationError("transform.version must be a positive integer")
    if (
        transform.frame_of_reference_uid
        != source_volume.coordinate_frame.frame_of_reference_uid
    ):
        raise CPRValidationError(
            "transform and source volume FrameOfReferenceUID disagree"
        )
    if (
        series_instance_uid is not None
        and series_instance_uid != transform.series_instance_uid
    ):
        raise CPRValidationError("series_instance_uid disagrees with transform")
    if (
        frame_of_reference_uid is not None
        and frame_of_reference_uid != transform.frame_of_reference_uid
    ):
        raise CPRValidationError("frame_of_reference_uid disagrees with transform")

    expected_source_frame_ref = f"frame_spec:{canonical_digest(source_volume.coordinate_frame)}"
    _require_ref(expected_source_frame_ref, _FRAME_REF_RE, "source_frame_spec_ref")
    transform_ref = _normalize_transform_ref(transform, transform_artifact_ref)
    provenance_payload_ref, provenance_id = _normalize_provenance_ref(provenance_ref)
    if provenance_record_ref is not None:
        _provided_ref, provided_id = _normalize_provenance_ref(provenance_record_ref)
        if provided_id != provenance_id:
            raise CPRValidationError(
                "provenance_record_ref disagrees with provenance_ref"
            )
    series_uid = (
        transform.series_instance_uid if series_instance_uid is None else series_instance_uid
    )
    frame_uid = (
        source_volume.coordinate_frame.frame_of_reference_uid
        if frame_of_reference_uid is None
        else frame_of_reference_uid
    )
    _require_dicom_uid(series_uid, "series_instance_uid")
    _require_dicom_uid(frame_uid, "frame_of_reference_uid")
    source_ref = _normalize_source_volume_ref(source_volume.volume_id)

    envelope = loader.read_envelope(source_volume.blob_ref.sha256)
    scalar_domain = envelope.get("scalar_domain", "STORED")
    intensity_unit = envelope.get("intensity_unit", scalar_domain)
    if not isinstance(scalar_domain, str) or scalar_domain not in {
        "STORED",
        "MODALITY_TRANSFORMED",
        "HU",
    }:
        raise CPRValidationError("source volume scalar_domain is invalid")
    if not isinstance(intensity_unit, str) or intensity_unit != scalar_domain:
        raise CPRValidationError(
            "source volume intensity_unit disagrees with scalar_domain"
        )
    if envelope.get("blob_sha256") != source_volume.blob_ref.sha256:
        raise CPRAbstentionError("CPR_SOURCE_INVALID", "SOURCE_BLOB_REF_MISMATCH")
    if envelope.get("codec") != source_volume.blob_ref.codec:
        raise CPRAbstentionError("CPR_SOURCE_INVALID", "SOURCE_CODEC_MISMATCH")
    envelope_size = envelope.get("byte_size")
    if envelope_size is not None and envelope_size != source_volume.blob_ref.byte_size:
        raise CPRAbstentionError("CPR_SOURCE_INVALID", "SOURCE_BYTE_SIZE_MISMATCH")
    canonical_source = loader.read_array(source_volume.blob_ref.sha256, envelope=envelope)
    if source_array is None:
        source_array = canonical_source
    source = np.asarray(source_array)
    expected_shape = (
        int(source_volume.coordinate_frame.dimensions[0]),
        int(source_volume.coordinate_frame.dimensions[1]),
        int(source_volume.coordinate_frame.dimensions[2]),
    )
    if tuple(int(value) for value in source.shape) != expected_shape:
        raise CPRAbstentionError("CPR_SOURCE_INVALID", "VOLUME_SHAPE_MISMATCH")
    if source.dtype != canonical_source.dtype or source.tobytes(
        order="C"
    ) != canonical_source.tobytes(order="C"):
        raise CPRAbstentionError("CPR_SOURCE_INVALID", "SOURCE_BLOB_MISMATCH")
    source_values_exactly_integral = envelope.get("source_values_exactly_integral")
    if source_values_exactly_integral is None:
        source_values_exactly_integral = _source_is_exactly_integral(canonical_source)
    if not isinstance(source_values_exactly_integral, bool):
        raise CPRAbstentionError("CPR_SOURCE_INVALID", "INTEGRAL_FLAG_INVALID")
    _validate_transform_for_frame(transform, source_volume.coordinate_frame)
    try:
        assert_round_trip(transform, transform.inverse)
    except CanonicalValidationError as exc:
        raise CPRAbstentionError("CPR_TRANSFORM_INVALID", "AFFINE_ROUND_TRIP") from exc
    world_to_voxel = _normalize_transform(transform)

    centerline_artifact, centerline_document = _normalize_centerline(
        centerline, source_volume_ref=source_ref
    )
    # F1: the frame sequence is derived from the VALIDATED, DIGEST-BOUND sealed
    # payload coordinates (never from the live ``centerline.path`` array, which
    # is not sealed).  A tampered in-memory path cannot move a CPR frame.
    points = np.asarray(centerline_document["points_world"], dtype=np.float64)
    arcs = np.asarray(centerline_document["s_mm"], dtype=np.float64)
    s_values, origins, tangents, normals, binormals = resample_frames(
        points, arcs, float(s_spacing_mm)
    )
    rows = int(origins.shape[0])
    columns = int(radial_count)
    pixels, codec = resample_cross_sections(
        source,
        world_to_voxel,
        s_values=s_values,
        origins=origins,
        binormals=binormals,
        s_spacing_mm=float(s_spacing_mm),
        radial_spacing_mm=float(radial_spacing_mm),
        radial_count=columns,
        method=method,
        out_of_bounds_fill_value=out_of_bounds_fill_value,
        source_values_exactly_integral=source_values_exactly_integral,
    )
    pixel_blob = loader.write_blob(pixels.tobytes(order="C"))
    extent = (rows * float(s_spacing_mm), columns * float(radial_spacing_mm))
    payload: dict[str, object] = {
        "schema_version": CPR_SCHEMA,
        "centerline_ref": f"{_CENTERLINE_REF_PREFIX}{centerline_artifact.artifact_id}",
        "centerline_payload_sha256": centerline_artifact.payload.blob_sha256,
        "source_frame_spec_ref": expected_source_frame_ref,
        "source_volume_ref": source_ref,
        "source_volume_blob_sha256": source_volume.blob_ref.sha256,
        "source_volume_shape": [int(value) for value in source.shape],
        "voxel_spacing_mm": [
            float(source_volume.coordinate_frame.spacing_mm[0]),
            float(source_volume.coordinate_frame.spacing_mm[1]),
            float(source_volume.coordinate_frame.spacing_mm[2]),
        ],
        "out_of_plane_ref": {
            "transform_artifact_ref": transform_ref,
            "transform_version": int(transform.version),
            "direction": "WORLD_TO_VOXEL",
        },
        "voxel_center_convention": MPR_VOXEL_CENTER_CONVENTION,
        "framing_rule": CPR_FRAMING_RULE,
        "cross_section_rule": CPR_CROSS_SECTION_RULE,
        "method": method,
        "out_of_bounds_fill_value": float(out_of_bounds_fill_value),
        "arithmetic_order": MPR_ARITHMETIC_ORDER,
        "output_dtype": "int16" if codec == CANONICAL_INT16_CODEC else "float32",
        "rounding_rule": MPR_ROUNDING_RULE,
        "scalar_domain": scalar_domain,
        "intensity_unit": intensity_unit,
        "s_spacing_mm": float(s_spacing_mm),
        "radial_spacing_mm": float(radial_spacing_mm),
        "radial_count": columns,
        "matrix_size": [rows, columns],
        "output_extent_mm": [extent[0], extent[1]],
        "frame_sequence": [
            {
                "s_mm": float(s_values[index]),
                "origin_world_mm": [float(value) for value in origins[index]],
                "tangent": [float(value) for value in tangents[index]],
                "normal": [float(value) for value in normals[index]],
                "binormal": [float(value) for value in binormals[index]],
            }
            for index in range(rows)
        ],
        "sampling_parameters_sha256": "",
        "pixel_blob_ref": {
            "sha256": pixel_blob.sha256,
            "codec": codec,
            "byte_size": pixel_blob.byte_size,
        },
        "study_instance_uid": study_instance_uid,
        "series_instance_uid": series_uid,
        "frame_of_reference_uid": frame_uid,
        "provenance_ref": provenance_payload_ref,
    }
    payload["sampling_parameters_sha256"] = sampling_parameters_sha256(payload)
    validated_payload = validate_cpr_payload(payload)
    centerline_id = centerline_artifact.artifact_id
    artifact = DerivedArtifact(
        study_ref=source_volume.study_ref,
        kind=ArtifactKind.MPR_VIEW,
        payload=PayloadRef(
            payload_schema=CPR_SCHEMA,
            blob_sha256=canonical_digest(validated_payload),
            codec="json",
            inline_summary=(
                ("centerline_ref", f"{_CENTERLINE_REF_PREFIX}{centerline_id}"),
                ("matrix_size", f"{rows}x{columns}"),
            ),
        ),
        lineage=Lineage(
            input_volume_refs=(source_volume.volume_id,),
            input_artifact_refs=(centerline_id,),
            geometry_base=GeometryBase(geometry_id=centerline_id, geometry_version=1),
        ),
        claim_binding=ClaimBinding(claim_id=CPR_CLAIM_ID, claim_version=claim_version),
        producer=Producer(
            producer_type=ProducerType.DETERMINISTIC_ENGINE,
            algorithm=CPR_ALGORITHM,
            algorithm_version=CPR_ALGORITHM_VERSION,
            parameters_blob_sha256=cast(str, validated_payload["sampling_parameters_sha256"]),
        ),
        review=ReviewBlock(review_state=ReviewState.PREDICTED),
        quality=QualityBlock(),
        release_lane=release_lane or source_volume.release_lane,
        provenance_ref=provenance_id,
        created_by=created_by,
        created_at=utc_now_iso(),
        artifact_id=artifact_ulid,
    )
    return CPRBuildResult(
        artifact=artifact,
        payload=validated_payload,
        pixels=pixels,
        pixel_blob_sha256=pixel_blob.sha256,
        pixel_blob_byte_size=pixel_blob.byte_size,
    )


# ---------------------------------------------------------------------------
# §44 provenance
# ---------------------------------------------------------------------------


def cpr_provenance(
    *,
    subject_ref: str,
    volume: Any,
    centerline_artifact: DerivedArtifact,
    sampling_parameters_sha256: str,
    volume_provenance_ref: str,
    centerline_provenance_ref: str,
    software_version: str,
    code_commit_sha: str,
    runtime: str,
    created_by: str,
    at: str | None = None,
    provenance_id: str | None = None,
) -> ProvenanceRecord:
    """The full §44 nine-item provenance record of one CPR view.

    ``subject_ref`` is the CPR ``MPR_VIEW`` artifact reference the record is
    about; the two inputs are the source volume and the CENTERLINE artifact
    the framing was taken on.
    """

    if not isinstance(subject_ref, str) or not subject_ref:
        raise CPRValidationError("subject_ref is required")
    if not is_sha256_hex(sampling_parameters_sha256):
        raise CPRValidationError("sampling_parameters_sha256 must be 64-hex")
    if centerline_artifact.kind is not ArtifactKind.CENTERLINE:
        raise CPRValidationError("centerline_artifact must be a CENTERLINE artifact")
    parameters_ref = BlobRef(
        blob_ref=f"coronary-cpr-parameters:{sampling_parameters_sha256}",
        sha256=canonical_digest(
            {
                "sampling_parameters_sha256": sampling_parameters_sha256,
                "framing_rule": CPR_FRAMING_RULE,
                "cross_section_rule": CPR_CROSS_SECTION_RULE,
            }
        ),
    )
    return create_provenance(
        subject_ref=subject_ref,
        event_type=ProvenanceEventType.DERIVED,
        actor=Actor(actor_type=ActorType.ENGINE, actor_id=CPR_ACTOR_ID),
        inputs=(
            InputRef(
                ref=f"volume_id:{volume.volume_id}",
                kind="volume",
                sha256_at_use=volume.blob_ref.sha256,
            ),
            InputRef(
                ref=f"{_CENTERLINE_REF_PREFIX}{centerline_artifact.artifact_id}",
                kind="artifact",
                sha256_at_use=centerline_artifact.payload.blob_sha256,
            ),
        ),
        preprocessing=(
            "coronary-cpr:parallel-transport-frames+"
            "mpr-internals-resample"
        ),
        parameters_ref=parameters_ref,
        human_edit=NotApplicable(
            value=None,
            reason="deterministic framing; the human seed rides the centerline chain",
        ),
        upstream_refs=(volume_provenance_ref, centerline_provenance_ref),
        environment=Environment(
            software_version=software_version,
            code_commit_sha=code_commit_sha,
            runtime=runtime,
            deterministic_seed=None,
        ),
        created_by=created_by,
        at=at,
        provenance_id=provenance_id,
    )


create_cpr_view = build_cpr_view
build_cpr = build_cpr_view
validate_payload = validate_cpr_payload

__all__ = [
    "CPR_ACTOR_ID",
    "CPR_ALGORITHM",
    "CPR_ALGORITHM_VERSION",
    "CPR_CLAIM_ID",
    "CPR_CLAIM_VERSION",
    "CPR_CROSS_SECTION_RULE",
    "CPR_FRAMING_RULE",
    "CPR_SCHEMA",
    "CPR_SESSION_INTEGRATION_UNAVAILABLE",
    "CPRAbstentionError",
    "CPRBuildResult",
    "CPRSessionIntegrationRefusal",
    "CPRValidationError",
    "build_cpr",
    "build_cpr_view",
    "create_cpr_view",
    "cpr_provenance",
    "decode_cpr_pixel_blob",
    "require_ws02_snapshot_admission",
    "resample_cross_sections",
    "resample_frames",
    "sampling_parameters_sha256",
    "validate_cpr_payload",
    "validate_payload",
]
