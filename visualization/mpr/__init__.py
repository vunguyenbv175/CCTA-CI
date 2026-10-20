"""Deterministic orthogonal MPR views.

The module deliberately has no viewer state.  A view is a canonical
``DerivedArtifact`` plus a closed JSON payload and an immutable pixel blob.
All sampling is performed by :func:`reslice` under the arithmetic contract in
IMAGING_CORE_001_SPEC Part 3.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping, Sequence
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
from core.digest import canonical_digest, is_sha256_hex
from core.enums import (
    ArtifactKind,
    ProducerType,
    ReleaseLane,
    ReviewState,
    SpaceKind,
)
from core.errors import CanonicalValidationError
from core.events import utc_now_iso
from core.ids import new_ulid, validate_ulid

MPR_SCHEMA: Final[str] = "ccta.mpr.view.v2"
#: v1 remains valid forever: an MPR_VIEW artifact persisted before
#: IMAGING_CORE_002_ADMISSION carries no source-bit-depth record and stays
#: verifiable.  Version selection is by the payload's own ``schema_version``,
#: never by inference from the fields present.
MPR_SCHEMA_LEGACY: Final[str] = "ccta.mpr.view.v1"
MPR_ARITHMETIC_ORDER: Final[str] = "FLOAT32_LEFT_ASSOCIATIVE_NO_FMA_V1"
MPR_ROUNDING_RULE: Final[str] = "SOURCE_ROUND_HALF_AWAY_FLOAT32_CAST_RNE_V1"
MPR_VOXEL_CENTER_CONVENTION: Final[str] = "INTEGER_INDEX_IS_VOXEL_CENTER_V1"
MPR_CLAIM_ID: Final[str] = "CLM-VIZ-001"
MPR_CLAIM_VERSION: Final[str] = "v0-draft"

_INT16_MIN = -32768
_INT16_MAX = 32767
_ULID_CHARS = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
_ULID_RE = re.compile(rf"[{_ULID_CHARS}]{{26}}\Z")
_UID_RE = re.compile(r"[0-9]+(?:\.[0-9]+)*\Z")
_SOURCE_VOLUME_REF_RE = re.compile(rf"volume_id:[{_ULID_CHARS}]{{26}}\Z")
_TRANSFORM_REF_RE = re.compile(rf"coordinate_transform:[{_ULID_CHARS}]{{26}}\Z")
_PROVENANCE_REF_RE = re.compile(rf"provenance:[{_ULID_CHARS}]{{26}}\Z")
_FRAME_REF_RE = re.compile(r"frame_spec:([0-9a-f]{64})\Z")

_PAYLOAD_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "plane",
        "origin",
        "row_axis",
        "column_axis",
        "plane_coordinate_mm",
        "in_plane_spacing_mm",
        "matrix_size",
        "output_extent_mm",
        "source_frame_spec_ref",
        "out_of_plane_ref",
        "voxel_center_convention",
        "method",
        "out_of_bounds_fill_value",
        "arithmetic_order",
        "output_dtype",
        "rounding_rule",
        "scalar_domain",
        "intensity_unit",
        "sampling_parameters_sha256",
        "pixel_blob_ref",
        "source_volume_ref",
        "study_instance_uid",
        "series_instance_uid",
        "frame_of_reference_uid",
        "provenance_ref",
    }
)

#: v2 (IMAGING_CORE_002_ADMISSION) adds the additive source-bit-depth record so
#: an MPR frame stays auditable back to the stored representation of its source
#: series.  The v2 field set is exactly the v1 set plus these four fields.
MPR_SOURCE_BIT_DEPTH_FIELDS: Final[frozenset[str]] = frozenset(
    {"bits_allocated", "bits_stored", "high_bit", "pixel_representation"}
)
_PAYLOAD_FIELDS_V2: Final[frozenset[str]] = _PAYLOAD_FIELDS | MPR_SOURCE_BIT_DEPTH_FIELDS

#: The canonical-volume envelope's names for the same four values
#: (``apps.workstation.volume_loader``).  Read-only mirror: MPR records them
#: under the Part 3 payload names above and never renames them.
ENVELOPE_BIT_DEPTH_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "source_bits_allocated",
        "source_bits_stored",
        "source_high_bit",
        "source_pixel_representation",
    }
)

_PLANE_TABLE: Final[dict[str, tuple[tuple[float, float, float], ...]]] = {
    "AXIAL": ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    "CORONAL": ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    "SAGITTAL": ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
}


class MPRValidationError(CanonicalValidationError):
    """A closed MPR payload or sampling contract was violated."""


class MPRAbstentionError(MPRValidationError):
    """A requested view cannot be produced safely and must abstain."""

    def __init__(self, reason_code: str, detail: str | None = None) -> None:
        self.reason_code = reason_code
        self.detail_code = detail
        super().__init__([reason_code] if detail is None else [reason_code, detail])


@dataclass(frozen=True)
class MPRBuildResult:
    """The complete persisted representation of one MPR view."""

    artifact: DerivedArtifact
    payload: dict[str, object]
    pixels: np.ndarray
    pixel_blob_sha256: str
    pixel_blob_byte_size: int

    @property
    def pixel_blob_ref(self) -> dict[str, object]:
        return cast(dict[str, object], self.payload["pixel_blob_ref"])


def plane_axes(plane: str) -> tuple[tuple[float, float, float], ...]:
    """Return ``(normal, row_axis, column_axis)`` for one closed plane."""
    if not isinstance(plane, str):
        raise MPRValidationError(
            f"plane must be AXIAL, CORONAL, or SAGITTAL: {plane!r}"
        )
    try:
        return _PLANE_TABLE[plane.upper()]
    except KeyError as exc:
        raise MPRValidationError(
            f"plane must be AXIAL, CORONAL, or SAGITTAL: {plane!r}"
        ) from exc


def frame_spec_ref(frame: FrameSpec) -> str:
    """Content reference for the exact source FrameSpec JSON."""
    return f"frame_spec:{canonical_digest(frame)}"


def _f32(value: object) -> np.float32:
    """One explicit IEEE binary32 cast (RNE in the admitted NumPy runtime)."""
    try:
        result = np.asarray(cast(Any, value), dtype=np.float32).reshape(())
    except (FloatingPointError, TypeError, ValueError, OverflowError) as exc:
        raise MPRAbstentionError("MPR_ARITHMETIC_INVALID", "FLOAT32_CAST_INVALID") from exc
    if not bool(np.isfinite(result)):
        raise MPRAbstentionError("MPR_ARITHMETIC_INVALID", "NONFINITE_FLOAT32")
    return np.float32(result)


def _f32_mul(left: object, right: object) -> np.float32:
    try:
        product = np.multiply(
            np.float32(cast(Any, left)), np.float32(cast(Any, right)), dtype=np.float32
        )
    except FloatingPointError as exc:
        raise MPRAbstentionError("MPR_ARITHMETIC_INVALID", "NONFINITE_FLOAT32") from exc
    return _f32(product)


def _f32_add(left: object, right: object) -> np.float32:
    try:
        summed = np.add(
            np.float32(cast(Any, left)), np.float32(cast(Any, right)), dtype=np.float32
        )
    except FloatingPointError as exc:
        raise MPRAbstentionError("MPR_ARITHMETIC_INVALID", "NONFINITE_FLOAT32") from exc
    return _f32(summed)


def _round_half_away(value: np.float32) -> int:
    """Round one binary32 coordinate half away from zero."""
    number = float(value)
    if number >= 0.0:
        return int(math.floor(number + 0.5))
    return int(math.ceil(number - 0.5))


def _world_pixel_center(
    origin: Sequence[float],
    row_axis: Sequence[float],
    column_axis: Sequence[float],
    row_spacing: float,
    column_spacing: float,
    row: int,
    column: int,
) -> tuple[np.float32, np.float32, np.float32]:
    """Generate one center with the exact staged world-coordinate arithmetic."""
    row_index = _f32(row)
    column_index = _f32(column)
    row_distance = _f32_mul(row_index, _f32(row_spacing))
    column_distance = _f32_mul(column_index, _f32(column_spacing))
    result: list[np.float32] = []
    for axis in range(3):
        base = _f32(origin[axis])
        row_component = _f32_mul(row_distance, _f32(row_axis[axis]))
        base_plus_row = _f32_add(base, row_component)
        column_component = _f32_mul(column_distance, _f32(column_axis[axis]))
        result.append(_f32_add(base_plus_row, column_component))
    return (result[0], result[1], result[2])


def _world_to_voxel(
    matrix: tuple[tuple[float, ...], ...],
    world: Sequence[np.float32],
) -> tuple[np.float32, np.float32, np.float32]:
    """Evaluate M·[x,y,z,1] left-to-right with materialized float32 terms."""
    result: list[np.float32] = []
    for row in range(3):
        term_x = _f32_mul(_f32(matrix[row][0]), world[0])
        term_y = _f32_mul(_f32(matrix[row][1]), world[1])
        first = _f32_add(term_x, term_y)
        term_z = _f32_mul(_f32(matrix[row][2]), world[2])
        second = _f32_add(first, term_z)
        translation = _f32(matrix[row][3])
        result.append(_f32_add(second, translation))
    return (result[0], result[1], result[2])


def _source_index_bounds(source: np.ndarray) -> tuple[int, int, int]:
    # Canonical source layout is (slice=k, row=j, column=i), while q is (i,j,k).
    return (int(source.shape[2]), int(source.shape[1]), int(source.shape[0]))


def _in_extent(q: Sequence[np.float32], dimensions: Sequence[int]) -> bool:
    return all(
        float(q[index]) >= 0.0 and float(q[index]) <= dimensions[index] - 1
        for index in range(3)
    )


def _sample_nearest(source: np.ndarray, q: Sequence[np.float32]) -> np.float32:
    i = _round_half_away(q[0])
    j = _round_half_away(q[1])
    k = _round_half_away(q[2])
    if (
        i < 0
        or i >= source.shape[2]
        or j < 0
        or j >= source.shape[1]
        or k < 0
        or k >= source.shape[0]
    ):
        raise MPRAbstentionError("MPR_SAMPLE_INVALID", "NEAREST_INDEX_OUT_OF_EXTENT")
    return _f32(source[k, j, i])


def _lerp(a: object, b: object, t: np.float32) -> np.float32:
    one_minus_t = _f32(np.subtract(np.float32(1.0), t, dtype=np.float32))
    left = _f32_mul(_f32(a), one_minus_t)
    right = _f32_mul(_f32(b), t)
    return _f32_add(left, right)


def _sample_trilinear(source: np.ndarray, q: Sequence[np.float32]) -> np.float32:
    dimensions = _source_index_bounds(source)
    lower = [int(math.floor(float(component))) for component in q]
    upper = [min(lower[axis] + 1, dimensions[axis] - 1) for axis in range(3)]
    t = [_f32(np.float32(q[axis]) - np.float32(lower[axis])) for axis in range(3)]

    # i interpolation, then j interpolation, then k interpolation.  No
    # reduction/contraction operation is used anywhere in this path.
    i00 = _lerp(source[lower[2], lower[1], lower[0]], source[lower[2], lower[1], upper[0]], t[0])
    i10 = _lerp(source[lower[2], upper[1], lower[0]], source[lower[2], upper[1], upper[0]], t[0])
    i01 = _lerp(source[upper[2], lower[1], lower[0]], source[upper[2], lower[1], upper[0]], t[0])
    i11 = _lerp(source[upper[2], upper[1], lower[0]], source[upper[2], upper[1], upper[0]], t[0])
    j0 = _lerp(i00, i10, t[1])
    j1 = _lerp(i01, i11, t[1])
    return _lerp(j0, j1, t[2])


def _source_is_exactly_integral(source: np.ndarray) -> bool:
    if source.dtype.kind in {"i", "u"}:
        return True
    if source.dtype.kind != "f":
        return False
    for value in source.flat:
        number = float(value)
        if not math.isfinite(number) or number != math.trunc(number):
            return False
    return True


def _buffer_is_exact_int16(values: np.ndarray) -> bool:
    for value in values.flat:
        number = float(value)
        if (
            not math.isfinite(number)
            or number != math.trunc(number)
            or number < _INT16_MIN
            or number > _INT16_MAX
        ):
            return False
    return True


def output_dtype_policy(
    source: np.ndarray,
    sampled_values: np.ndarray,
    *,
    method: str,
    source_values_exactly_integral: bool | None = None,
) -> str:
    """Return the closed Part 3 output dtype decision for sampled values."""
    if source_values_exactly_integral is None:
        source_integral = _source_is_exactly_integral(source)
    elif isinstance(source_values_exactly_integral, bool):
        source_integral = source_values_exactly_integral
    else:
        raise MPRValidationError("source_values_exactly_integral must be a boolean")
    if (
        method == "NEAREST"
        and source_integral
        and _buffer_is_exact_int16(sampled_values)
    ):
        return "int16"
    return "float32"


def reslice(
    source: np.ndarray,
    world_to_voxel: CoordinateTransform,
    *,
    plane: str,
    origin: tuple[float, float, float],
    in_plane_spacing_mm: tuple[float, float],
    matrix_size: tuple[int, int],
    method: str,
    out_of_bounds_fill_value: float,
    source_values_exactly_integral: bool | None = None,
) -> tuple[np.ndarray, str]:
    """Pure-NumPy MPR kernel returning ``(pixels, codec)``.

    ``source`` is always indexed ``source[k, j, i]``.  The transform must be
    the persisted WORLD→VOXEL transform; a voxel→world transform is rejected
    rather than silently inverted in the arithmetic kernel.
    """
    if not isinstance(source, np.ndarray) or source.ndim != 3:
        raise MPRAbstentionError("MPR_SOURCE_INVALID", "VOLUME_SHAPE_INVALID")
    if not isinstance(world_to_voxel, CoordinateTransform):
        raise MPRAbstentionError("MPR_TRANSFORM_INVALID", "WORLD_TO_VOXEL_REQUIRED")
    if source.dtype.kind not in {"i", "u", "f"}:
        raise MPRAbstentionError("MPR_SOURCE_INVALID", "VOLUME_DTYPE_INVALID")
    if any(int(d) <= 0 for d in source.shape):
        raise MPRAbstentionError("MPR_SOURCE_INVALID", "VOLUME_SHAPE_INVALID")
    if source.dtype.kind in {"i", "u"}:
        # Integers are always finite, and every int/uint width NumPy admits
        # (up to uint64, ~1.8e19) sits below the float32 range ceiling
        # (~3.4e38), so the per-voxel scan below cannot fail: skip it.
        pass
    elif source.dtype.kind == "f" and source.dtype.itemsize <= 8:
        # Exact probe: float16/float32 widen to float64 without rounding and
        # float64 is the identity, so the verdict matches float(value) per
        # voxel.  np.argmax over the C-order ravel finds the same first
        # failing voxel the scalar scan would raise on.
        probe = np.asarray(source, dtype=np.float64)
        out_of_spec = ~np.isfinite(probe) | (
            np.abs(probe) > float(np.finfo(np.float32).max)
        )
        if bool(np.any(out_of_spec)):
            first_bad = int(np.argmax(out_of_spec.ravel()))
            try:
                numeric_value = float(source.flat[first_bad])
            except (TypeError, ValueError, OverflowError) as exc:
                raise MPRAbstentionError(
                    "MPR_SOURCE_INVALID", "NONFINITE_SOURCE"
                ) from exc
            if not math.isfinite(numeric_value):
                raise MPRAbstentionError("MPR_SOURCE_INVALID", "NONFINITE_SOURCE")
            raise MPRAbstentionError("MPR_SOURCE_INVALID", "FLOAT32_RANGE")
    else:
        for value in source.flat:
            try:
                numeric_value = float(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise MPRAbstentionError(
                    "MPR_SOURCE_INVALID", "NONFINITE_SOURCE"
                ) from exc
            if not math.isfinite(numeric_value):
                raise MPRAbstentionError("MPR_SOURCE_INVALID", "NONFINITE_SOURCE")
            if source.dtype.kind in {"i", "u", "f"} and abs(numeric_value) > float(
                np.finfo(np.float32).max
            ):
                raise MPRAbstentionError("MPR_SOURCE_INVALID", "FLOAT32_RANGE")
    if (
        world_to_voxel.from_space is not SpaceKind.WORLD
        or world_to_voxel.to_space is not SpaceKind.VOXEL
    ):
        raise MPRAbstentionError("MPR_TRANSFORM_INVALID", "WORLD_TO_VOXEL_REQUIRED")
    if source_values_exactly_integral is not None and not isinstance(
        source_values_exactly_integral, bool
    ):
        raise MPRValidationError("source_values_exactly_integral must be a boolean")
    if any(
        not math.isfinite(float(value))
        for matrix_row in world_to_voxel.matrix
        for value in matrix_row
    ):
        raise MPRAbstentionError("MPR_TRANSFORM_INVALID", "NONFINITE_MATRIX")
    _normal, row_axis, column_axis = plane_axes(plane)
    if (
        not isinstance(origin, (list, tuple))
        or len(origin) != 3
        or any(
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(float(value))
            for value in origin
        )
    ):
        raise MPRValidationError("origin must be a finite Vec3")
    if (
        not isinstance(in_plane_spacing_mm, (list, tuple))
        or len(in_plane_spacing_mm) != 2
        or any(
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(float(value))
            or float(value) <= 0.0
            for value in in_plane_spacing_mm
        )
    ):
        raise MPRValidationError("in_plane_spacing_mm must be two positive finite values")
    if (
        not isinstance(matrix_size, (list, tuple))
        or len(matrix_size) != 2
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 1 <= value <= 2**31 - 1
            for value in matrix_size
        )
    ):
        raise MPRValidationError("matrix_size must contain positive 32-bit integers")
    rows, columns = int(matrix_size[0]), int(matrix_size[1])
    if not isinstance(method, str) or method not in {"NEAREST", "TRILINEAR"}:
        raise MPRValidationError("method must be NEAREST or TRILINEAR")
    if (
        isinstance(out_of_bounds_fill_value, bool)
        or not isinstance(out_of_bounds_fill_value, Real)
        or not math.isfinite(float(out_of_bounds_fill_value))
    ):
        raise MPRValidationError("out_of_bounds_fill_value must be finite")
    fill = _f32(out_of_bounds_fill_value)

    # Fully initialize the destination; no output or scratch value is ever
    # read before this assignment.  Every pixel is then written exactly once
    # in row-major (r,c) order.
    try:
        output = np.full((rows, columns), fill, dtype=np.dtype("<f4"), order="C")
    except (MemoryError, OverflowError, ValueError) as exc:
        raise MPRAbstentionError(
            "MPR_OUTPUT_INVALID", "OUTPUT_ALLOCATION_FAILED"
        ) from exc
    dimensions = _source_index_bounds(source)

    # Vectorized form of the per-pixel loop.  Each stage below issues the same
    # IEEE binary32 operations, in the same left-associative order, as one
    # scalar iteration through _world_pixel_center, _world_to_voxel, and
    # _sample_nearest/_sample_trilinear: NumPy float32 ufuncs evaluate every
    # element with a single correctly-rounded operation (no FMA, no
    # reduction reordering), so each output element is bit-identical to the
    # scalar loop.  Index and rounding stages run in float64, exactly matching
    # float(np.float32) widening followed by math.floor/math.ceil.
    def _lerp_grids(
        left: np.ndarray, right: np.ndarray, ratio: np.ndarray, one_minus: np.ndarray
    ) -> np.ndarray:
        product_left = np.multiply(left, one_minus, dtype=np.float32)
        product_right = np.multiply(right, ratio, dtype=np.float32)
        return np.add(product_left, product_right, dtype=np.float32)

    # Run the vectorized kernel under the caller's ambient floating-point
    # exception policy (no errstate override): the scalar baseline observed
    # the ambient policy, so the vectorized ufuncs must too — a caller-side
    # under/over/divide/invalid='raise' then raises identically instead of
    # returning a pixel the scalar path would have refused.  A raised
    # FloatingPointError maps to the same closed verdict the scalar
    # _f32/_f32_mul/_f32_add helpers produce.
    try:
        row_scale = _f32(float(in_plane_spacing_mm[0]))
        column_scale = _f32(float(in_plane_spacing_mm[1]))
        row_distances = np.empty(rows, dtype=np.float32)
        for row_index in range(rows):
            row_distances[row_index] = _f32_mul(_f32(row_index), row_scale)
        column_distances = np.empty(columns, dtype=np.float32)
        for column_index in range(columns):
            column_distances[column_index] = _f32_mul(_f32(column_index), column_scale)
        origin_f32 = [_f32(origin[axis]) for axis in range(3)]
        row_axis_f32 = [_f32(row_axis[axis]) for axis in range(3)]
        column_axis_f32 = [_f32(column_axis[axis]) for axis in range(3)]
        matrix_f32 = [
            [_f32(world_to_voxel.matrix[matrix_row][matrix_column]) for matrix_column in range(4)]
            for matrix_row in range(3)
        ]
        world_grids: list[np.ndarray] = []
        for axis in range(3):
            row_component = np.multiply(
                row_distances, row_axis_f32[axis], dtype=np.float32
            )
            base_plus_row = np.add(origin_f32[axis], row_component, dtype=np.float32)
            column_component = np.multiply(
                column_distances, column_axis_f32[axis], dtype=np.float32
            )
            world_grids.append(
                np.add(base_plus_row[:, None], column_component[None, :], dtype=np.float32)
            )
            if not bool(np.all(np.isfinite(world_grids[axis]))):
                raise MPRAbstentionError("MPR_ARITHMETIC_INVALID", "NONFINITE_FLOAT32")
        voxel_grids: list[np.ndarray] = []
        for matrix_row in range(3):
            term_x = np.multiply(matrix_f32[matrix_row][0], world_grids[0], dtype=np.float32)
            term_y = np.multiply(matrix_f32[matrix_row][1], world_grids[1], dtype=np.float32)
            first = np.add(term_x, term_y, dtype=np.float32)
            term_z = np.multiply(matrix_f32[matrix_row][2], world_grids[2], dtype=np.float32)
            second = np.add(first, term_z, dtype=np.float32)
            voxel_grids.append(np.add(second, matrix_f32[matrix_row][3], dtype=np.float32))
            if not bool(np.all(np.isfinite(voxel_grids[matrix_row]))):
                raise MPRAbstentionError("MPR_ARITHMETIC_INVALID", "NONFINITE_FLOAT32")
        inside = np.ones((rows, columns), dtype=bool)
        for axis in range(3):
            coordinates = voxel_grids[axis].astype(np.float64)
            inside &= coordinates >= 0.0
            inside &= coordinates <= float(dimensions[axis] - 1)
        if method == "NEAREST":
            if bool(np.any(inside)):
                nearest_subscripts: list[np.ndarray] = []
                for axis in range(3):
                    magnitudes = voxel_grids[axis][inside].astype(np.float64)
                    nonneg = magnitudes >= 0.0
                    nearest_subscripts.append(
                        np.where(
                            nonneg,
                            np.floor(magnitudes + 0.5),
                            np.ceil(magnitudes - 0.5),
                        ).astype(np.int64)
                    )
                nearest_i, nearest_j, nearest_k = nearest_subscripts
                if (
                    bool(np.any(nearest_i < 0))
                    or bool(np.any(nearest_i >= source.shape[2]))
                    or bool(np.any(nearest_j < 0))
                    or bool(np.any(nearest_j >= source.shape[1]))
                    or bool(np.any(nearest_k < 0))
                    or bool(np.any(nearest_k >= source.shape[0]))
                ):
                    raise MPRAbstentionError(
                        "MPR_SAMPLE_INVALID", "NEAREST_INDEX_OUT_OF_EXTENT"
                    )
                gathered = source[nearest_k, nearest_j, nearest_i]
                output[inside] = np.asarray(gathered, dtype=np.float32)
        elif bool(np.any(inside)):
            fractions: list[np.ndarray] = []
            lowers: list[np.ndarray] = []
            uppers: list[np.ndarray] = []
            for axis in range(3):
                magnitudes = voxel_grids[axis][inside].astype(np.float64)
                lower = np.floor(magnitudes).astype(np.int64)
                lowers.append(lower)
                uppers.append(np.minimum(lower + 1, dimensions[axis] - 1))
                fractions.append(
                    np.subtract(
                        voxel_grids[axis][inside],
                        lower.astype(np.float32),
                        dtype=np.float32,
                    )
                )
            lower_i, lower_j, lower_k = lowers
            upper_i, upper_j, upper_k = uppers
            corner_000 = np.asarray(
                source[lower_k, lower_j, lower_i], dtype=np.float32
            )
            corner_100 = np.asarray(
                source[lower_k, lower_j, upper_i], dtype=np.float32
            )
            corner_010 = np.asarray(
                source[lower_k, upper_j, lower_i], dtype=np.float32
            )
            corner_110 = np.asarray(
                source[lower_k, upper_j, upper_i], dtype=np.float32
            )
            corner_001 = np.asarray(
                source[upper_k, lower_j, lower_i], dtype=np.float32
            )
            corner_101 = np.asarray(
                source[upper_k, lower_j, upper_i], dtype=np.float32
            )
            corner_011 = np.asarray(
                source[upper_k, upper_j, lower_i], dtype=np.float32
            )
            corner_111 = np.asarray(
                source[upper_k, upper_j, upper_i], dtype=np.float32
            )
            one_minus = [
                np.subtract(np.float32(1.0), fractions[axis], dtype=np.float32)
                for axis in range(3)
            ]
            blend_i00 = _lerp_grids(corner_000, corner_100, fractions[0], one_minus[0])
            blend_i10 = _lerp_grids(corner_010, corner_110, fractions[0], one_minus[0])
            blend_i01 = _lerp_grids(corner_001, corner_101, fractions[0], one_minus[0])
            blend_i11 = _lerp_grids(corner_011, corner_111, fractions[0], one_minus[0])
            blend_j0 = _lerp_grids(blend_i00, blend_i10, fractions[1], one_minus[1])
            blend_j1 = _lerp_grids(blend_i01, blend_i11, fractions[1], one_minus[1])
            sampled = _lerp_grids(blend_j0, blend_j1, fractions[2], one_minus[2])
            if not bool(np.all(np.isfinite(sampled))):
                raise MPRAbstentionError("MPR_ARITHMETIC_INVALID", "NONFINITE_FLOAT32")
            output[inside] = sampled
    except FloatingPointError as exc:
        raise MPRAbstentionError("MPR_ARITHMETIC_INVALID", "NONFINITE_FLOAT32") from exc

    # An int16 pixel buffer cannot represent a fractional fill without a
    # silent truncation.  Keep the exact input fill in the predicate even if
    # its binary32 cast happens to land on an integer value.
    fill_is_exact_int16 = (
        float(out_of_bounds_fill_value) == math.trunc(float(out_of_bounds_fill_value))
        and _INT16_MIN <= float(out_of_bounds_fill_value) <= _INT16_MAX
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
    if not bool(np.all(np.isfinite(output))):
        raise MPRAbstentionError("MPR_ARITHMETIC_INVALID", "NONFINITE_OUTPUT")
    output = np.asarray(output, dtype=np.dtype("<f4"), order="C")
    output.setflags(write=False)
    return output, CANONICAL_FLOAT32_CODEC


def _sampling_parameters(
    payload: Mapping[str, object],
) -> dict[str, object]:
    """Canonical parameter object named by Part 3's digest rule."""
    transform_ref = cast(dict[str, object], payload["out_of_plane_ref"])
    geometry = {
        "origin": payload["origin"],
        "row_axis": payload["row_axis"],
        "column_axis": payload["column_axis"],
        "plane_coordinate_mm": payload["plane_coordinate_mm"],
        "in_plane_spacing_mm": payload["in_plane_spacing_mm"],
        "matrix_size": payload["matrix_size"],
        "output_extent_mm": payload["output_extent_mm"],
        "source_frame_spec_ref": payload["source_frame_spec_ref"],
    }
    return {
        "plane": payload["plane"],
        "geometry": geometry,
        "transform_ref": transform_ref["transform_artifact_ref"],
        "transform_version": transform_ref["transform_version"],
        "method": payload["method"],
        "out_of_bounds_fill_value": payload["out_of_bounds_fill_value"],
        "arithmetic_order": payload["arithmetic_order"],
        "output_dtype": payload["output_dtype"],
        "rounding_rule": payload["rounding_rule"],
        "scalar_domain": payload["scalar_domain"],
    }


def sampling_parameters_sha256(payload: Mapping[str, object]) -> str:
    """Return the Part 3 canonical digest for a payload's sampling fields."""
    return canonical_digest(_sampling_parameters(payload))


def _require_ref(
    value: object,
    pattern: re.Pattern[str],
    label: str,
) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise MPRValidationError(f"{label} has the wrong canonical reference form")
    return value


def _require_dicom_uid(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or not 1 <= len(value) <= 64
        or _UID_RE.fullmatch(value) is None
    ):
        raise MPRValidationError(f"{label} must be a DICOM UID")
    return value


def _require_vec(value: object, length: int, label: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise MPRValidationError(f"{label} must have length {length}")
    result: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, Real):
            raise MPRValidationError(f"{label} must contain finite numbers")
        try:
            number = float(item)
        except (TypeError, ValueError, OverflowError) as exc:
            raise MPRValidationError(f"{label} must contain finite numbers") from exc
        if not math.isfinite(number):
            raise MPRValidationError(f"{label} must contain finite numbers")
        result.append(number)
    return tuple(result)


def validate_mpr_payload(payload: Mapping[str, object]) -> dict[str, object]:
    """Validate the closed ``ccta.mpr.view.v1``/``v2`` payload exactly.

    v2 is v1 plus the four additive source-bit-depth fields (IMAGING_CORE_002
    ADMISSION).  Both versions are accepted so a v1 frame artifact stays
    verifiable; version selection is by the payload's own ``schema_version``,
    never by inference.
    """
    if not isinstance(payload, Mapping):
        raise MPRValidationError("MPR payload must be an object")
    if any(not isinstance(key, str) for key in payload):
        raise MPRValidationError("MPR payload keys must be strings")
    declared = payload.get("schema_version")
    if declared == MPR_SCHEMA_LEGACY:
        expected_fields = _PAYLOAD_FIELDS
    elif declared == MPR_SCHEMA:
        expected_fields = _PAYLOAD_FIELDS_V2
    else:
        raise MPRValidationError(
            f"schema_version must be {MPR_SCHEMA} or {MPR_SCHEMA_LEGACY}"
        )
    keys = set(payload)
    missing = sorted(expected_fields - keys)
    unknown = sorted(keys - expected_fields)
    if missing or unknown:
        problems = []
        if missing:
            problems.append(f"missing fields: {', '.join(missing)}")
        if unknown:
            problems.append(f"unknown fields: {', '.join(unknown)}")
        raise MPRValidationError(problems)
    plane = payload["plane"]
    if not isinstance(plane, str) or plane not in _PLANE_TABLE:
        raise MPRValidationError("plane is outside the closed enum")
    normal, expected_row, expected_column = plane_axes(cast(str, plane))
    origin = _require_vec(payload["origin"], 3, "origin")
    row_axis = _require_vec(payload["row_axis"], 3, "row_axis")
    column_axis = _require_vec(payload["column_axis"], 3, "column_axis")
    for actual, expected, label in (
        (row_axis, expected_row, "row_axis"),
        (column_axis, expected_column, "column_axis"),
    ):
        norm = math.sqrt(sum(value * value for value in actual))
        component_error = max(abs(a - b) for a, b in zip(actual, expected))
        if (
            abs(norm - 1.0) > 1e-6
            or component_error > 1e-6
            or actual != expected
        ):
            raise MPRValidationError(f"{label} does not equal the fixed plane table")
    if abs(sum(a * b for a, b in zip(row_axis, column_axis))) > 1e-6:
        raise MPRValidationError("row_axis and column_axis are not orthogonal")
    plane_coordinate = payload["plane_coordinate_mm"]
    if isinstance(plane_coordinate, bool) or not isinstance(plane_coordinate, Real):
        raise MPRValidationError("plane_coordinate_mm must be finite")
    try:
        plane_coordinate_number = float(plane_coordinate)
    except (TypeError, ValueError, OverflowError) as exc:
        raise MPRValidationError("plane_coordinate_mm must be finite") from exc
    plane_error = plane_coordinate_number - sum(a * b for a, b in zip(origin, normal))
    if not math.isfinite(plane_coordinate_number) or abs(plane_error) > 1e-6:
        raise MPRValidationError("plane_coordinate_mm does not match origin")
    spacing = _require_vec(payload["in_plane_spacing_mm"], 2, "in_plane_spacing_mm")
    if any(value <= 0.0 for value in spacing):
        raise MPRValidationError("in_plane_spacing_mm must be positive")
    matrix_size_value = payload["matrix_size"]
    if not isinstance(matrix_size_value, (list, tuple)) or len(matrix_size_value) != 2:
        raise MPRValidationError("matrix_size must be a two-vector")
    matrix_size: tuple[int, int] = (0, 0)
    converted_size: list[int] = []
    for value in matrix_size_value:
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 2**31 - 1:
            raise MPRValidationError("matrix_size must contain positive integers")
        converted_size.append(value)
    matrix_size = (converted_size[0], converted_size[1])
    extent = _require_vec(payload["output_extent_mm"], 2, "output_extent_mm")
    expected_extent = (matrix_size[0] * spacing[0], matrix_size[1] * spacing[1])
    if extent != expected_extent:
        raise MPRValidationError("output_extent_mm is not the exact cell extent")
    _require_ref(payload["source_frame_spec_ref"], _FRAME_REF_RE, "source_frame_spec_ref")
    out_of_plane = payload["out_of_plane_ref"]
    if isinstance(out_of_plane, Mapping) and any(
        not isinstance(key, str) for key in out_of_plane
    ):
        raise MPRValidationError("out_of_plane_ref keys must be strings")
    if not isinstance(out_of_plane, Mapping) or set(out_of_plane) != {
        "transform_artifact_ref",
        "transform_version",
        "direction",
    }:
        raise MPRValidationError("out_of_plane_ref has an unexpected shape")
    _require_ref(
        out_of_plane["transform_artifact_ref"],
        _TRANSFORM_REF_RE,
        "transform_artifact_ref",
    )
    version = out_of_plane["transform_version"]
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise MPRValidationError("transform_version must be a positive integer")
    if out_of_plane["direction"] != "WORLD_TO_VOXEL":
        raise MPRValidationError("out_of_plane_ref direction is fixed")
    if payload["voxel_center_convention"] != MPR_VOXEL_CENTER_CONVENTION:
        raise MPRValidationError("voxel_center_convention is fixed")
    method = payload["method"]
    if not isinstance(method, str) or method not in {"NEAREST", "TRILINEAR"}:
        raise MPRValidationError("method is outside the closed enum")
    fill = payload["out_of_bounds_fill_value"]
    if (
        isinstance(fill, bool)
        or not isinstance(fill, (int, float))
        or not math.isfinite(float(fill))
    ):
        raise MPRValidationError("out_of_bounds_fill_value must be finite")
    try:
        fill_number = float(fill)
    except (TypeError, ValueError, OverflowError) as exc:
        raise MPRValidationError("out_of_bounds_fill_value must be finite") from exc
    if abs(fill_number) > float(np.finfo(np.float32).max):
        raise MPRValidationError("out_of_bounds_fill_value overflows float32")
    if payload["arithmetic_order"] != MPR_ARITHMETIC_ORDER:
        raise MPRValidationError("arithmetic_order is fixed")
    output_dtype = payload["output_dtype"]
    if not isinstance(output_dtype, str) or output_dtype not in {"int16", "float32"}:
        raise MPRValidationError("output_dtype is outside the closed enum")
    if output_dtype == "int16":
        if method != "NEAREST":
            raise MPRValidationError("int16 output is legal only for NEAREST")
        if (
            fill_number != math.trunc(fill_number)
            or fill_number < _INT16_MIN
            or fill_number > _INT16_MAX
        ):
            raise MPRValidationError("int16 output cannot represent the fill value")
    if payload["rounding_rule"] != MPR_ROUNDING_RULE:
        raise MPRValidationError("rounding_rule is fixed")
    scalar_domain = payload["scalar_domain"]
    if not isinstance(scalar_domain, str) or scalar_domain not in {
        "STORED",
        "MODALITY_TRANSFORMED",
        "HU",
    }:
        raise MPRValidationError("scalar_domain is outside the closed vocabulary")
    if payload["intensity_unit"] != scalar_domain:
        raise MPRValidationError("intensity_unit must equal scalar_domain")
    if not is_sha256_hex(payload["sampling_parameters_sha256"]):
        raise MPRValidationError("sampling_parameters_sha256 must be lowercase sha256")
    if sampling_parameters_sha256(payload) != payload["sampling_parameters_sha256"]:
        raise MPRValidationError("sampling_parameters_sha256 does not bind sampling fields")
    pixel_ref = payload["pixel_blob_ref"]
    if isinstance(pixel_ref, Mapping) and any(not isinstance(key, str) for key in pixel_ref):
        raise MPRValidationError("pixel_blob_ref keys must be strings")
    if not isinstance(pixel_ref, Mapping) or set(pixel_ref) != {"sha256", "codec", "byte_size"}:
        raise MPRValidationError("pixel_blob_ref has an unexpected shape")
    if not is_sha256_hex(pixel_ref["sha256"]):
        raise MPRValidationError("pixel_blob_ref.sha256 must be lowercase sha256")
    expected_codec = "int16-le" if output_dtype == "int16" else "float32-le"
    if pixel_ref["codec"] != expected_codec:
        raise MPRValidationError("pixel_blob_ref codec disagrees with output_dtype")
    byte_size = pixel_ref["byte_size"]
    expected_byte_size = matrix_size[0] * matrix_size[1] * (
        2 if expected_codec == "int16-le" else 4
    )
    if (
        isinstance(byte_size, bool)
        or not isinstance(byte_size, int)
        or byte_size != expected_byte_size
        or byte_size <= 0
    ):
        raise MPRValidationError("pixel_blob_ref.byte_size is incorrect")
    _require_ref(payload["source_volume_ref"], _SOURCE_VOLUME_REF_RE, "source_volume_ref")
    for label in ("study_instance_uid", "series_instance_uid", "frame_of_reference_uid"):
        _require_dicom_uid(payload[label], label)
    _require_ref(payload["provenance_ref"], _PROVENANCE_REF_RE, "provenance_ref")
    if declared == MPR_SCHEMA:
        _validate_source_bit_depth(payload)
    return dict(payload)


def _validate_source_bit_depth(payload: Mapping[str, object]) -> None:
    """Validate the closed v2 source-bit-depth record (provenance only).

    The four fields are integers and are NOT reinterpreted as an admission
    decision here: the reader owns admission (IMAGING_CORE_001 Part 4.1).  This
    only guarantees the recorded provenance is well-typed and internally
    consistent, so a v2 frame can always be audited back to the stored
    representation of its source series.
    """
    values: dict[str, int] = {}
    for label in sorted(MPR_SOURCE_BIT_DEPTH_FIELDS):
        value = payload[label]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise MPRValidationError(f"{label} must be a non-negative integer")
        values[label] = int(value)
    if values["bits_allocated"] <= 0:
        raise MPRValidationError("bits_allocated must be positive")
    if values["bits_stored"] <= 0 or values["bits_stored"] > values["bits_allocated"]:
        raise MPRValidationError("bits_stored is outside bits_allocated")
    # Container containment (IMAGING_CORE_002 fix round, R1 F5): HighBit is a
    # bit position INSIDE the 16-bit container, so high_bit <= bits_allocated-1
    # in addition to the bits_stored consistency check above.  A record like
    # bits_allocated=16, bits_stored=12, high_bit=99 cannot describe a real
    # source tuple and must never reach a persisted v2 payload.
    if values["high_bit"] > values["bits_allocated"] - 1:
        raise MPRValidationError("high_bit is outside the bits_allocated container")
    if values["high_bit"] < values["bits_stored"] - 1:
        raise MPRValidationError("high_bit is inconsistent with bits_stored")
    if values["pixel_representation"] not in {0, 1}:
        raise MPRValidationError("pixel_representation is outside the closed enum")


def _envelope_source_bit_depth(envelope: Mapping[str, object]) -> dict[str, int] | None:
    """Return the volume envelope's v2 source-bit-depth record, or None for v1.

    ``None`` means the envelope predates the amendment (no bit-depth fields at
    all).  A PARTIAL record is a defect, not a v1 envelope: it abstains with
    ``SOURCE_BIT_DEPTH_MISSING``.  A malformed value abstains with
    ``SOURCE_BIT_DEPTH_INVALID``.  Both are closed MPR_SOURCE_INVALID details
    following the existing pattern at :364-381/:992-994 — no silent default.
    """
    present = [name for name in sorted(ENVELOPE_BIT_DEPTH_FIELDS) if name in envelope]
    if not present:
        return None
    if len(present) != len(ENVELOPE_BIT_DEPTH_FIELDS):
        raise MPRAbstentionError("MPR_SOURCE_INVALID", "SOURCE_BIT_DEPTH_MISSING")
    record: dict[str, int] = {}
    for name in present:
        value = envelope[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise MPRAbstentionError("MPR_SOURCE_INVALID", "SOURCE_BIT_DEPTH_INVALID")
        record[name] = int(value)
    mapped = {
        "bits_allocated": record["source_bits_allocated"],
        "bits_stored": record["source_bits_stored"],
        "high_bit": record["source_high_bit"],
        "pixel_representation": record["source_pixel_representation"],
    }
    # Container containment (R1 F5): the envelope mirror carries the same
    # bound the v2 payload validator enforces — high_bit must sit inside
    # bits_allocated (15 or 11 for the admitted tuples).  A malformed record
    # abstains here instead of reaching a v2 payload.
    if (
        mapped["bits_allocated"] <= 0
        or mapped["bits_stored"] <= 0
        or mapped["bits_stored"] > mapped["bits_allocated"]
        or mapped["high_bit"] < mapped["bits_stored"] - 1
        or mapped["high_bit"] > mapped["bits_allocated"] - 1
        or mapped["pixel_representation"] not in {0, 1}
    ):
        raise MPRAbstentionError("MPR_SOURCE_INVALID", "SOURCE_BIT_DEPTH_INVALID")
    return mapped


def _normalize_source_volume_ref(volume_id: str) -> str:
    if not isinstance(volume_id, str):
        raise MPRValidationError("source Volume id must be a ULID")
    if volume_id.startswith("volume_id:"):
        _require_ref(volume_id, _SOURCE_VOLUME_REF_RE, "source_volume_ref")
        return volume_id
    try:
        validate_ulid(volume_id)
    except (TypeError, ValueError) as exc:
        raise MPRValidationError("source Volume id must be a ULID") from exc
    return f"volume_id:{volume_id}"


def _normalize_transform_ref(transform: CoordinateTransform, value: str | None) -> str:
    ref = transform.transform_id if value is None else value
    if not isinstance(ref, str):
        raise MPRValidationError(
            "transform_artifact_ref is required as coordinate_transform:<ULID>"
        )
    if not ref.startswith("coordinate_transform:"):
        try:
            validate_ulid(ref)
        except ValueError as exc:
            raise MPRValidationError(
                "transform_artifact_ref is required as coordinate_transform:<ULID>"
            ) from exc
        ref = f"coordinate_transform:{ref}"
    _require_ref(ref, _TRANSFORM_REF_RE, "transform_artifact_ref")
    return ref


def _normalize_provenance_ref(value: str) -> tuple[str, str]:
    if not isinstance(value, str):
        raise MPRValidationError("provenance_ref must be provenance:<ULID>")
    if value.startswith("provenance:"):
        ref = _require_ref(value, _PROVENANCE_REF_RE, "provenance_ref")
        return ref, ref.split(":", 1)[1]
    try:
        validate_ulid(value)
    except (TypeError, ValueError) as exc:
        raise MPRValidationError("provenance_ref must be provenance:<ULID>") from exc
    return f"provenance:{value}", value


def _normalize_transform(
    transform: CoordinateTransform,
) -> CoordinateTransform:
    if transform.from_space is SpaceKind.WORLD and transform.to_space is SpaceKind.VOXEL:
        return transform
    if transform.from_space is SpaceKind.VOXEL and transform.to_space is SpaceKind.WORLD:
        return transform.inverse
    raise MPRAbstentionError("MPR_TRANSFORM_INVALID", "WORLD_TO_VOXEL_REQUIRED")


def _validate_transform_for_frame(
    transform: CoordinateTransform, frame: FrameSpec
) -> None:
    """Require the supplied persisted transform to describe the source grid."""
    try:
        actual_world_to_voxel = _normalize_transform(transform)
        expected_world_to_voxel = invert_matrix(frame.voxel_to_world_matrix())
    except (CanonicalValidationError, TypeError, ValueError, OverflowError) as exc:
        raise MPRAbstentionError("MPR_TRANSFORM_INVALID", "FRAME_AFFINE_INVALID") from exc
    if actual_world_to_voxel.matrix != expected_world_to_voxel:
        raise MPRAbstentionError("MPR_TRANSFORM_INVALID", "FRAME_AFFINE_MISMATCH")


def decode_pixel_blob(payload: Mapping[str, object], data: bytes) -> np.ndarray:
    """Validate and decode one persisted MPR pixel blob without mutation."""
    validated = validate_mpr_payload(payload)
    if not isinstance(data, bytes):
        raise MPRValidationError("pixel blob must be immutable bytes")
    pixel_ref = cast(dict[str, object], validated["pixel_blob_ref"])
    digest = cast(str, pixel_ref["sha256"])
    if hashlib.sha256(data).hexdigest() != digest:
        raise MPRValidationError("pixel blob content does not match pixel_blob_ref")
    matrix_size = validated["matrix_size"]
    if not isinstance(matrix_size, (list, tuple)) or len(matrix_size) != 2:
        raise MPRValidationError("matrix_size is not a canonical two-vector")
    rows, columns = matrix_size
    dtype = np.dtype("<i2" if pixel_ref["codec"] == "int16-le" else "<f4")
    expected = rows * columns * dtype.itemsize
    if len(data) != expected or pixel_ref["byte_size"] != expected:
        raise MPRValidationError("pixel blob byte_size does not match matrix_size")
    pixels = np.frombuffer(data, dtype=dtype).copy().reshape((rows, columns))
    if not bool(np.all(np.isfinite(pixels))):
        raise MPRValidationError("pixel blob contains nonfinite values")
    pixels.setflags(write=False)
    return pixels


def build_mpr(
    source_volume: Any,
    transform: CoordinateTransform,
    *,
    plane: str,
    origin: tuple[float, float, float],
    in_plane_spacing_mm: tuple[float, float],
    matrix_size: tuple[int, int],
    method: str,
    out_of_bounds_fill_value: float,
    loader: CanonicalVolumeLoader | None = None,
    source_array: np.ndarray | None = None,
    source_frame_spec_ref: str | None = None,
    transform_artifact_ref: str | None = None,
    provenance_ref: str,
    study_instance_uid: str,
    series_instance_uid: str | None = None,
    frame_of_reference_uid: str | None = None,
    claim_version: str = MPR_CLAIM_VERSION,
    release_lane: ReleaseLane | None = None,
    created_by: str = "ccta-mpr",
    artifact_id: str | None = None,
    provenance_record_ref: str | None = None,
) -> MPRBuildResult:
    """Build and persist one validated MPR ``DerivedArtifact``.

    ``source_volume`` is a canonical :class:`core.volume.Volume`.  Callers may
    provide ``source_array`` for an already decoded canonical blob; otherwise
    ``loader`` is required and the canonical blob is decoded by its envelope.
    ``provenance_ref`` is mandatory and must resolve to the caller's §44
    provenance record; this function never fabricates provenance.
    """
    from core.volume import Volume

    if not isinstance(source_volume, Volume):
        raise MPRValidationError("source_volume must be a canonical Volume")
    if not isinstance(transform, CoordinateTransform):
        raise MPRValidationError("transform must be a persisted CoordinateTransform")
    if loader is None:
        raise MPRValidationError("loader is required to persist the MPR pixel blob")
    if not isinstance(claim_version, str) or not claim_version:
        raise MPRValidationError("claim_version is required")
    if not isinstance(created_by, str) or not created_by:
        raise MPRValidationError("created_by is required")
    artifact_ulid = artifact_id or new_ulid()
    try:
        validate_ulid(artifact_ulid)
    except (TypeError, ValueError) as exc:
        raise MPRValidationError("artifact_id must be a ULID") from exc
    if release_lane is not None and not isinstance(release_lane, ReleaseLane):
        raise MPRValidationError("release_lane must be a ReleaseLane")
    if not isinstance(source_volume.study_ref, str) or not source_volume.study_ref:
        raise MPRValidationError("source_volume.study_ref is required")
    _require_dicom_uid(study_instance_uid, "study_instance_uid")
    if (
        not isinstance(transform.version, int)
        or isinstance(transform.version, bool)
        or transform.version < 1
    ):
        raise MPRValidationError("transform.version must be a positive integer")
    if (
        transform.frame_of_reference_uid
        != source_volume.coordinate_frame.frame_of_reference_uid
    ):
        raise MPRValidationError("transform and source volume FrameOfReferenceUID disagree")
    if (
        series_instance_uid is not None
        and series_instance_uid != transform.series_instance_uid
    ):
        raise MPRValidationError("series_instance_uid disagrees with transform")
    if (
        frame_of_reference_uid is not None
        and frame_of_reference_uid != transform.frame_of_reference_uid
    ):
        raise MPRValidationError("frame_of_reference_uid disagrees with transform")

    # Resolve every reference and carried identity before running the kernel or
    # writing a content-addressed pixel blob.  A rejected request must not
    # leave a partially constructed MPR behind.
    expected_source_frame_ref = frame_spec_ref(source_volume.coordinate_frame)
    source_frame_ref = (
        expected_source_frame_ref if source_frame_spec_ref is None else source_frame_spec_ref
    )
    _require_ref(source_frame_ref, _FRAME_REF_RE, "source_frame_spec_ref")
    if source_frame_ref != expected_source_frame_ref:
        raise MPRValidationError("source_frame_spec_ref disagrees with source volume FrameSpec")
    transform_ref = _normalize_transform_ref(transform, transform_artifact_ref)
    provenance_payload_ref, provenance_id = _normalize_provenance_ref(provenance_ref)
    if provenance_record_ref is not None:
        _provided_provenance_ref, provided_provenance_id = _normalize_provenance_ref(
            provenance_record_ref
        )
        if provided_provenance_id != provenance_id:
            raise MPRValidationError("provenance_record_ref disagrees with provenance_ref")
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
        raise MPRValidationError("source volume scalar_domain is invalid")
    if not isinstance(intensity_unit, str) or intensity_unit != scalar_domain:
        raise MPRValidationError("source volume intensity_unit disagrees with scalar_domain")
    if envelope.get("blob_sha256") != source_volume.blob_ref.sha256:
        raise MPRAbstentionError("MPR_SOURCE_INVALID", "SOURCE_BLOB_REF_MISMATCH")
    if envelope.get("codec") != source_volume.blob_ref.codec:
        raise MPRAbstentionError("MPR_SOURCE_INVALID", "SOURCE_CODEC_MISMATCH")
    envelope_size = envelope.get("byte_size")
    if envelope_size is not None and envelope_size != source_volume.blob_ref.byte_size:
        raise MPRAbstentionError("MPR_SOURCE_INVALID", "SOURCE_BYTE_SIZE_MISMATCH")
    canonical_source = loader.read_array(source_volume.blob_ref.sha256, envelope=envelope)
    if source_array is None:
        source_array = canonical_source
    source = np.asarray(source_array)
    expected_shape = (
        source_volume.coordinate_frame.dimensions[0],
        source_volume.coordinate_frame.dimensions[1],
        source_volume.coordinate_frame.dimensions[2],
    )
    if tuple(int(value) for value in source.shape) != expected_shape:
        raise MPRAbstentionError("MPR_SOURCE_INVALID", "VOLUME_SHAPE_MISMATCH")
    if source.dtype != canonical_source.dtype or source.tobytes(
        order="C"
    ) != canonical_source.tobytes(order="C"):
        raise MPRAbstentionError("MPR_SOURCE_INVALID", "SOURCE_BLOB_MISMATCH")
    source_values_exactly_integral = envelope.get("source_values_exactly_integral")
    if source_values_exactly_integral is None:
        source_values_exactly_integral = _source_is_exactly_integral(canonical_source)
    if not isinstance(source_values_exactly_integral, bool):
        raise MPRAbstentionError("MPR_SOURCE_INVALID", "INTEGRAL_FLAG_INVALID")
    # v2 source-bit-depth provenance.  A legacy v1 envelope (written before
    # IMAGING_CORE_002_ADMISSION) carries no bit-depth record: the frame is then
    # emitted as v1 rather than abstaining, so an old volume stays viewable.
    # A v2-capable envelope that MISSes or CORRUPTs the record abstains with the
    # closed codes below — provenance is never fabricated and never defaulted.
    bit_depth_record = _envelope_source_bit_depth(envelope)
    schema_version = MPR_SCHEMA if bit_depth_record is not None else MPR_SCHEMA_LEGACY
    _validate_transform_for_frame(transform, source_volume.coordinate_frame)
    try:
        assert_round_trip(transform, transform.inverse)
    except CanonicalValidationError as exc:
        raise MPRAbstentionError("MPR_TRANSFORM_INVALID", "AFFINE_ROUND_TRIP") from exc
    world_to_voxel = _normalize_transform(transform)
    actual_plane = str(plane).upper()
    normal, row_axis, column_axis = plane_axes(actual_plane)
    pixels, codec = reslice(
        source,
        world_to_voxel,
        plane=actual_plane,
        origin=origin,
        in_plane_spacing_mm=in_plane_spacing_mm,
        matrix_size=matrix_size,
        method=method,
        out_of_bounds_fill_value=out_of_bounds_fill_value,
        source_values_exactly_integral=source_values_exactly_integral,
    )
    pixel_blob = loader.write_blob(pixels.tobytes(order="C"))
    rows, columns = int(matrix_size[0]), int(matrix_size[1])
    spacing = (float(in_plane_spacing_mm[0]), float(in_plane_spacing_mm[1]))
    extent = (rows * spacing[0], columns * spacing[1])
    payload: dict[str, object] = {
        "schema_version": schema_version,
        "plane": actual_plane,
        "origin": list(float(value) for value in origin),
        "row_axis": list(row_axis),
        "column_axis": list(column_axis),
        "plane_coordinate_mm": float(
            sum(float(a) * float(b) for a, b in zip(origin, normal))
        ),
        "in_plane_spacing_mm": list(spacing),
        "matrix_size": [rows, columns],
        "output_extent_mm": list(extent),
        "source_frame_spec_ref": source_frame_ref,
        "out_of_plane_ref": {
            "transform_artifact_ref": transform_ref,
            "transform_version": int(transform.version),
            "direction": "WORLD_TO_VOXEL",
        },
        "voxel_center_convention": MPR_VOXEL_CENTER_CONVENTION,
        "method": method,
        "out_of_bounds_fill_value": float(out_of_bounds_fill_value),
        "arithmetic_order": MPR_ARITHMETIC_ORDER,
        "output_dtype": "int16" if codec == CANONICAL_INT16_CODEC else "float32",
        "rounding_rule": MPR_ROUNDING_RULE,
        "scalar_domain": "STORED",
        "intensity_unit": "STORED",
        "sampling_parameters_sha256": "",
        "pixel_blob_ref": {
            "sha256": pixel_blob.sha256,
            "codec": codec,
            "byte_size": pixel_blob.byte_size,
        },
        "source_volume_ref": source_ref,
        "study_instance_uid": study_instance_uid,
        "series_instance_uid": series_uid,
        "frame_of_reference_uid": frame_uid,
        "provenance_ref": provenance_payload_ref,
    }
    payload["scalar_domain"] = scalar_domain
    payload["intensity_unit"] = intensity_unit
    if bit_depth_record is not None:
        payload.update(bit_depth_record)
    payload["sampling_parameters_sha256"] = sampling_parameters_sha256(payload)
    validated_payload = validate_mpr_payload(payload)
    transform_id = transform_ref.split(":", 1)[1]
    transform_lineage_id = transform_id
    artifact = DerivedArtifact(
        artifact_id=artifact_ulid,
        study_ref=source_volume.study_ref,
        kind=ArtifactKind.MPR_VIEW,
        payload=PayloadRef(
            payload_schema=schema_version,
            blob_sha256=canonical_digest(validated_payload),
            codec="json",
            inline_summary=(("plane", actual_plane), ("method", method)),
        ),
        lineage=Lineage(
            input_volume_refs=(source_volume.volume_id,),
            input_artifact_refs=(transform_lineage_id,),
            geometry_base=GeometryBase(
                geometry_id=transform_lineage_id,
                geometry_version=int(transform.version),
            ),
        ),
        claim_binding=ClaimBinding(claim_id=MPR_CLAIM_ID, claim_version=claim_version),
        producer=Producer(
            producer_type=ProducerType.DETERMINISTIC_ENGINE,
            algorithm="orthogonal-mpr-numpy",
            algorithm_version=MPR_ARITHMETIC_ORDER,
            parameters_blob_sha256=cast(str, validated_payload["sampling_parameters_sha256"]),
        ),
        review=ReviewBlock(review_state=ReviewState.PREDICTED),
        quality=QualityBlock(),
        release_lane=release_lane or source_volume.release_lane,
        provenance_ref=provenance_id,
        created_by=created_by,
        created_at=utc_now_iso(),
    )
    return MPRBuildResult(
        artifact=artifact,
        payload=validated_payload,
        pixels=pixels,
        pixel_blob_sha256=pixel_blob.sha256,
        pixel_blob_byte_size=pixel_blob.byte_size,
    )


create_mpr_view = build_mpr
build_mpr_view = build_mpr
create_mpr = build_mpr
reslice_volume = reslice
validate_payload = validate_mpr_payload

__all__ = [
    "MPR_ARITHMETIC_ORDER",
    "MPR_CLAIM_ID",
    "MPR_CLAIM_VERSION",
    "MPR_ROUNDING_RULE",
    "MPR_SCHEMA",
    "MPR_VOXEL_CENTER_CONVENTION",
    "MPRAbstentionError",
    "MPRBuildResult",
    "MPRValidationError",
    "build_mpr",
    "build_mpr_view",
    "create_mpr",
    "create_mpr_view",
    "decode_pixel_blob",
    "frame_spec_ref",
    "output_dtype_policy",
    "plane_axes",
    "reslice",
    "reslice_volume",
    "sampling_parameters_sha256",
    "validate_mpr_payload",
    "validate_payload",
]
