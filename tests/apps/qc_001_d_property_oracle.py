"""Independent QC_001-D property generators and world/scalar oracles.

This module intentionally has no import from the production quality package.
It describes the physical transforms from the accepted property contract and
adapts the real reader/materializer fixture surface for the integration driver.
Expected outcomes remain injection-driven in ``qc_001_d_oracle``; this module
only checks physical equivalence and exact coordinate relations.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, replace
from functools import cache
from typing import Final, cast

import numpy as np

from core.coordinates import FrameSpec, matmul, transform_for_frame
from core.dicom.instance import DicomInstance
from core.digest import canonical_digest
from core.volume import Volume
from tests.apps.qc_001_d_fixtures import (
    CensusCase,
    PreparedQCInput,
    ulid,
)

TRANSLATION: Final[tuple[float, float, float]] = (17.0, -11.0, 5.0)
ZERO_TRANSLATION: Final[tuple[float, float, float]] = (0.0, 0.0, 0.0)
INVERSE_TRANSLATION: Final[tuple[float, float, float]] = (-17.0, 11.0, -5.0)
EXACT_REPLICA_ORIGIN: Final[tuple[float, float, float]] = (8.0, -16.0, 32.0)
EXACT_REPLICA_IN_PLANE_SPACING: Final[tuple[float, float]] = (1.0, 2.0)
EXACT_REPLICA_COVERAGE_XY: Final[tuple[float, float]] = (40.0, 48.0)
EXACT_BINARY64_LIMIT: Final[int] = 2**53
IDENTITY_AFFINE: Final[tuple[float, ...]] = (
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
)
_PROPERTY_DIMENSIONS: Final[tuple[int, int, int]] = (9, 64, 64)
_PROPERTY_UNSIGNED_LEVELS: Final[dict[str, float]] = {
    "central": 30_000.0,
    "outer_high": 50_000.0,
}
_PROPERTY_SIGNED_LEVELS: Final[dict[str, float]] = {
    "central": 10_000.0,
    "outer_high": 22_000.0,
}


@dataclass(frozen=True, slots=True)
class PropertySurface:
    """One independently generated transformed reader/materializer surface."""

    prepared: PreparedQCInput
    name: str
    registration_affine: tuple[float, ...] = IDENTITY_AFFINE
    overlap_lower_lps_mm: tuple[float, float, float] = (
        -10_000.0,
        -10_000.0,
        -10_000.0,
    )
    overlap_upper_lps_mm: tuple[float, float, float] = (
        10_000.0,
        10_000.0,
        10_000.0,
    )


def _cross(
    first: tuple[float, float, float], second: tuple[float, float, float]
) -> tuple[float, float, float]:
    return (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )


def _frame_axes(
    frame: FrameSpec,
) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]:
    row = cast(
        tuple[float, float, float],
        tuple(float(value) for value in frame.image_orientation_patient[:3]),
    )
    column = cast(
        tuple[float, float, float],
        tuple(float(value) for value in frame.image_orientation_patient[3:]),
    )
    normal = _cross(row, column)
    length = sum(value * value for value in normal) ** 0.5
    return row, column, cast(
        tuple[float, float, float], tuple(value / length for value in normal)
    )


def _raw_world_center(frame: FrameSpec, k: int, j: int, i: int) -> tuple[float, float, float]:
    matrix = frame.voxel_to_world_matrix()
    # Keep the arithmetic order used by the source-instance binding explicit.
    return cast(
        tuple[float, float, float],
        tuple(
            (
                (
                    (matrix[axis][3] + matrix[axis][0] * float(i))
                    + matrix[axis][1] * float(j)
                )
                + matrix[axis][2] * float(k)
            )
            for axis in range(3)
        ),
    )


def _world_center(frame: FrameSpec, k: int, j: int, i: int) -> tuple[float, float, float]:
    matrix = frame.voxel_to_world_matrix()
    # Keep the comparator on the raw binary64 affine expression.
    return cast(
        tuple[float, float, float],
        tuple(
            (((matrix[axis][3] + matrix[axis][0] * float(i))
              + matrix[axis][1] * float(j)) + matrix[axis][2] * float(k))
            for axis in range(3)
        ),
    )


def _canonical_world_center(
    frame: FrameSpec,
    k: int,
    j: int,
    i: int,
    translation_normalization_mm: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Evaluate affine terms in a representation-independent order.

    Legal re-indexing changes the storage-axis term order.  The physical
    direction terms are evaluated in one canonical order, while the affine
    translation remains bound to the actual FrameSpec matrix.  A nonzero
    normalization is reserved for the declared global translation transform.
    """

    dimensions, canonical_vectors, signs, order, canonical_translation = (
        _canonical_frame_terms(frame)
    )
    indices = (i, j, k)
    result = [
        canonical_translation[axis] + float(translation_normalization_mm[axis])
        for axis in range(3)
    ]
    for column in order:
        index = indices[column]
        if signs[column] < 0:
            index = dimensions[column] - 1 - index
        for axis in range(3):
            result[axis] = result[axis] + canonical_vectors[column][axis] * float(index)
    return cast(tuple[float, float, float], tuple(result))


@cache
def _canonical_frame_terms(
    frame: FrameSpec,
) -> tuple[
    tuple[int, int, int],
    tuple[tuple[float, float, float], ...],
    tuple[int, int, int],
    tuple[int, int, int],
    tuple[float, float, float],
]:
    """Cache signed axes and an affine-bound canonical translation."""

    matrix = frame.voxel_to_world_matrix()
    dimensions = (
        int(frame.dimensions[2]),
        int(frame.dimensions[1]),
        int(frame.dimensions[0]),
    )
    vectors = tuple(
        tuple(float(matrix[axis][column]) for axis in range(3))
        for column in range(3)
    )
    canonical_vectors: list[tuple[float, float, float]] = []
    signs: list[int] = []
    for vector in vectors:
        first_nonzero = next(value for value in vector if value != 0.0)
        sign = 1 if first_nonzero > 0.0 else -1
        signs.append(sign)
        canonical_vectors.append(
            cast(tuple[float, float, float], tuple(sign * value for value in vector))
        )
    order = tuple(
        sorted(
            range(3),
            key=lambda column: (canonical_vectors[column], column),
        )
    )
    canonical_translation = [float(matrix[axis][3]) for axis in range(3)]
    for column in order:
        if signs[column] < 0:
            for axis in range(3):
                canonical_translation[axis] = (
                    canonical_translation[axis]
                    + vectors[column][axis] * float(dimensions[column] - 1)
                )
    return (
        dimensions,
        cast(tuple[tuple[float, float, float], ...], tuple(canonical_vectors)),
        cast(tuple[int, int, int], tuple(signs)),
        cast(tuple[int, int, int], order),
        cast(tuple[float, float, float], tuple(canonical_translation)),
    )


def world_scalar_multiset(
    frame: FrameSpec,
    array: np.ndarray,
    *,
    canonical: bool = False,
    translation_normalization_mm: tuple[float, float, float] = ZERO_TRANSLATION,
) -> tuple[tuple[float, float, float, float], ...]:
    """Return a sorted ``(world_x, world_y, world_z, scalar)`` multiset."""

    if translation_normalization_mm not in {
        ZERO_TRANSLATION,
        INVERSE_TRANSLATION,
    }:
        raise ValueError("only the declared translation inverse is supported")
    if canonical:
        centers = _canonical_world_centers(frame, translation_normalization_mm)
    else:
        raw_centers = _raw_world_centers(frame)
        centers = cast(
            tuple[tuple[float, float, float], ...],
            tuple(
                tuple(
                    float(value) + float(delta)
                    for value, delta in zip(center, translation_normalization_mm)
                )
                for center in raw_centers
            ),
        )
    values = [(*center, float(value)) for center, value in zip(centers, array.reshape(-1))]
    return tuple(sorted(values))


@cache
def _raw_world_centers(frame: FrameSpec) -> tuple[tuple[float, float, float], ...]:
    """Cache the unrounded affine centers used by the default oracle path."""

    return tuple(
        _world_center(frame, k, j, i)
        for k in range(int(frame.dimensions[0]))
        for j in range(int(frame.dimensions[1]))
        for i in range(int(frame.dimensions[2]))
    )


@cache
def _canonical_world_centers(
    frame: FrameSpec,
    translation_normalization_mm: tuple[float, float, float],
) -> tuple[tuple[float, float, float], ...]:
    """Cache exact canonical centers independently of scalar values."""

    return tuple(
        _canonical_world_center(frame, k, j, i, translation_normalization_mm)
        for k in range(int(frame.dimensions[0]))
        for j in range(int(frame.dimensions[1]))
        for i in range(int(frame.dimensions[2]))
    )


def _paired_array(prepared: PreparedQCInput) -> np.ndarray:
    if prepared.second_array is not None:
        return prepared.second_array
    return prepared.first_array


def _paired_instances(prepared: PreparedQCInput) -> tuple[DicomInstance, ...]:
    if prepared.second_instances is not None:
        return prepared.second_instances
    return prepared.first_instances


def _translated_frame(frame: FrameSpec, vector: tuple[float, float, float]) -> FrameSpec:
    origin = cast(
        tuple[float, float, float],
        tuple(
            float(old) + float(delta)
            for old, delta in zip(frame.image_position_patient, vector)
        ),
    )
    return replace(frame, image_position_patient=origin)


def _rebind_instances(
    instances: tuple[DicomInstance, ...],
    frame: FrameSpec,
    *,
    raw_slices: tuple[bytes, ...] | None = None,
    instance_offset: int = 0,
    rescale_slope: float | None = None,
    rescale_intercept: float | None = None,
    rescale_type: str | None = None,
) -> tuple[DicomInstance, ...]:
    rows = int(frame.dimensions[1])
    columns = int(frame.dimensions[2])
    orientation = frame.image_orientation_patient
    spacing = (float(frame.spacing_mm[1]), float(frame.spacing_mm[0]))
    rebound: list[DicomInstance] = []
    selected = instances[instance_offset : instance_offset + int(frame.dimensions[0])]
    if len(selected) != int(frame.dimensions[0]):
        raise ValueError("transformed FrameSpec slice count does not bind instances")
    for index, instance in enumerate(selected):
        data = instance.pixel_data if raw_slices is None else raw_slices[index]
        rebound.append(
            replace(
                instance,
                image_position_patient=_raw_world_center(frame, index, 0, 0),
                image_orientation_patient=orientation,
                pixel_spacing=spacing,
                rows=rows,
                columns=columns,
                pixel_data=data,
                pixel_data_byte_length=len(data),
                rescale_slope=(
                    instance.rescale_slope
                    if rescale_slope is None
                    else rescale_slope
                ),
                rescale_intercept=(
                    instance.rescale_intercept
                    if rescale_intercept is None
                    else rescale_intercept
                ),
                rescale_type=(
                    instance.rescale_type if rescale_type is None else rescale_type
                ),
            )
        )
    return tuple(rebound)


def _readonly(array: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(array)
    result.setflags(write=False)
    return result


def _property_ring_mask(shape: tuple[int, int]) -> np.ndarray:
    rows, columns = shape
    outer = np.zeros(shape, dtype=bool)
    inner = np.zeros(shape, dtype=bool)
    row_start = math.ceil(0.35 * (rows - 1))
    column_start = math.ceil(0.35 * (columns - 1))
    row_end = math.floor(0.65 * (rows - 1))
    column_end = math.floor(0.65 * (columns - 1))
    outer[row_start : row_end + 1, column_start : column_end + 1] = True
    row_margin = max(2, (row_end - row_start + 1) // 6)
    column_margin = max(2, (column_end - column_start + 1) // 6)
    inner[
        row_start + row_margin : row_end - row_margin + 1,
        column_start + column_margin : column_end - column_margin + 1,
    ] = True
    return outer & ~inner


def _property_central_mask(shape: tuple[int, int, int]) -> np.ndarray:
    slices, rows, columns = shape
    u = np.linspace(-1.0, 1.0, columns)
    v = np.linspace(-1.0, 1.0, rows)
    mask = (np.abs(v)[None, :, None] <= 0.30) & (np.abs(u)[None, None, :] <= 0.30)
    return np.broadcast_to(mask, (slices, rows, columns)).copy()


def _property_base_values(case: CensusCase) -> np.ndarray:
    """Build independent injection pixels for the exact axis-aligned basis."""

    if case.geometry is None:
        raise ValueError("exact property replica requires a valid geometry stratum")
    shape = _PROPERTY_DIMENSIONS
    signed = case.geometry.pixels == "signed"
    central = _property_central_mask(shape)
    rows, columns = shape[1], shape[2]
    u = np.linspace(-1.0, 1.0, columns)
    v = np.linspace(-1.0, 1.0, rows)
    rho = np.sqrt(u[None, :] ** 2 + v[:, None] ** 2)
    injection = case.injection
    if signed:
        smooth = 1500.0 + 7500.0 * np.exp(-(rho * rho) / 0.55)
        levels = _PROPERTY_SIGNED_LEVELS
    else:
        smooth = 5000.0 + 25000.0 * np.exp(-(rho * rho) / 0.55)
        levels = _PROPERTY_UNSIGNED_LEVELS
    smooth[rho <= 0.24] = levels["central"]
    smooth[(rho >= 0.78) & (rho <= 0.85)] = levels["outer_high"]
    values = np.broadcast_to(smooth, shape).copy()
    values[:, _property_ring_mask((rows, columns))] += 500.0

    if injection is None or injection.flag is None:
        return values
    flag = injection.flag
    severity = injection.severity
    assert flag is not None and severity is not None

    if flag == "RESPIRATORY_MOTION":
        # Exact replicas use the integer in-plane basis and positive slice normal.
        band_limits = (
            (0.20, 0.40)
            if severity == "MILD"
            else (0.23, 0.30)
        )
        respiratory_band = (rho >= band_limits[0]) & (rho <= band_limits[1])
        injection_slice = 1 if severity == "MILD" else shape[0] - 2
        amplitude = (4000.0 if severity == "MILD" else 9000.0) if signed else (
            7000.0 if severity == "MILD" else 15000.0
        )
        if signed:
            values[injection_slice, respiratory_band] -= amplitude
        else:
            values[injection_slice, respiratory_band] += amplitude
        return values

    if flag == "STAIR_STEP":
        state_slices = {2, 3} if severity == "MILD" else {1, 3, 5, 7}
        block = np.zeros((rows, columns), dtype=bool)
        block_rows = (20, 22) if signed else (21, 24)
        block[block_rows[0] : block_rows[1], 27:39] = True
        for k in state_slices:
            values[k, block] += 5000.0
        return values

    if flag == "POOR_CONTRAST":
        amplitude = (8000.0 if severity == "MILD" else 12000.0) if signed else (
            22000.0 if severity == "MILD" else 30000.0
        )
        u_ramp = np.broadcast_to(u[None, :], (rows, columns))
        values[:, central[0]] += amplitude * (u_ramp[central[0]] - 0.30)
        return values

    if flag == "EXCESSIVE_NOISE":
        amplitude = (5000.0 if severity == "MILD" else 7000.0) if signed else (
            13_000.0 if severity == "MILD" else 24_000.0
        )
        _grid_k, grid_j, grid_i = np.indices(shape)
        pseudo = ((grid_j * 29 + grid_i * 43) % 11) - 5
        noise = amplitude * pseudo / 5.0
        checkerboard = ((grid_j + grid_i) % 2) == 0
        values[central & checkerboard] += noise[central & checkerboard]
        return values

    if flag == "BLOOMING":
        core = np.zeros((rows, columns), dtype=bool)
        core[30:33, 30:33] = True
        halo = np.zeros((rows, columns), dtype=bool)
        halo[29:34, 29:34] = True
        halo[core] = False
        if signed:
            low_halo, high_core, high_halo = 6000.0, 24000.0, 18000.0
        else:
            low_halo, high_core, high_halo = 35000.0, 65000.0, 56000.0
        high_halo_mask = np.zeros((rows, columns), dtype=bool)
        if severity == "MILD":
            high_halo_mask[29, 31] = True
            high_halo_mask[33, 31] = True
            high_halo_mask[31, 29] = True
            high_halo_mask &= halo
        else:
            high_halo_mask[29, 29:34] = True
            high_halo_mask[30, 29:32] = True
            high_halo_mask &= halo
        for slice_index in range(shape[0]):
            values[slice_index, core] = high_core
            values[slice_index, high_halo_mask] = high_halo
            values[slice_index, halo & ~high_halo_mask] = low_halo
        return values

    if flag == "BEAM_HARDENING":
        high_value = (
            (22000.0 if not signed else (7000.0 if severity == "MILD" else 8000.0))
            if severity == "MILD"
            else (26000.0 if not signed else 8000.0)
        )
        angle = np.arctan2(v[:, None], u[None, :])
        sector = (angle >= -math.pi) & (angle < -math.pi + math.pi / 4.0)
        annulus = (rho >= 0.45) & (rho <= 0.70)
        values[:, annulus & sector] = high_value
        return values

    return values


def _property_pair_values(case: CensusCase, first: np.ndarray) -> np.ndarray:
    """Build the independent paired phase for the exact axis-aligned basis."""

    if case.geometry is None:
        raise ValueError("exact property replica requires a valid geometry stratum")
    result = np.array(first, dtype=np.float64, copy=True)
    injection = case.injection
    if injection is None or injection.flag not in {"CARDIAC_MOTION", "MISREGISTRATION"}:
        return np.ascontiguousarray(result)
    signed = case.geometry.pixels == "signed"
    if injection.flag == "CARDIAC_MOTION":
        shape = (
            int(result.shape[0]),
            int(result.shape[1]),
            int(result.shape[2]),
        )
        central = _property_central_mask(shape)[0]
        factor = 0.65 if injection.severity == "MILD" else 1.0
        reference = (
            10_000.25
            if signed
            else 30_000.25
            if case.geometry.pixels == "rescaled-fractional"
            else 30_000.0
        )
        result[:, central] = reference + factor * (result[:, central] - reference)
        if injection.severity == "SEVERE":
            rows, columns = result.shape[1:]
            u = np.linspace(-1.0, 1.0, columns)
            v = np.linspace(-1.0, 1.0, rows)
            inner = np.sqrt(v[:, None] * v[:, None] + u[None, :] * u[None, :]) < 0.20
            grid_j, grid_i = np.indices((rows, columns))
            checker = np.where((grid_j + grid_i) % 2 == 0, 1.0, -1.0)
            amplitude = 7000.0 if signed else 10000.0
            result[:, inner] += amplitude * checker[inner]
        return np.ascontiguousarray(result)

    # The exact in-plane basis selects a row-aligned edge strip for every scalar
    # representation; the census column-aligned branch is anisotropic-only.
    edge_strip = np.zeros(result.shape[1:], dtype=bool)
    edge_strip[:27 if signed else 30, :] = True
    offset = (
        (2200.0 if injection.severity == "MILD" else 4200.0)
        if signed
        else (4400.0 if injection.severity == "MILD" else 8800.0)
    )
    result[:, edge_strip] += offset
    return np.ascontiguousarray(result)


def _property_replica_array(case: CensusCase, *, second: bool) -> np.ndarray:
    """Encode independent exact-basis injection values in the row format."""

    first = _property_base_values(case)
    values = _property_pair_values(case, first) if second else first
    if case.geometry is None:
        raise ValueError("exact property replica requires a valid geometry stratum")
    representation = case.geometry.pixels
    if representation == "signed":
        return _readonly(np.rint(values).astype("<i2"))
    if representation == "rescaled-fractional":
        raw = np.rint(values - 0.25).astype("<u2")
        decoded = raw.astype(np.float32) + np.float32(0.25)
        return _readonly(np.ascontiguousarray(decoded, dtype=np.float32))
    raw = np.rint(values).astype("<u2")
    return _readonly(np.ascontiguousarray(raw.astype(np.float32)))


def _exact_replica_z_spacing(case: CensusCase) -> float:
    injection = case.injection
    if injection is not None and injection.flag == "INCOMPLETE_ANATOMY":
        return 18.0 if injection.severity == "MILD" else 14.0
    return 20.0


def _exact_replica_frame(
    prepared: PreparedQCInput,
    *,
    z_spacing: float,
) -> FrameSpec:
    source = prepared.first_volume.coordinate_frame
    return FrameSpec(
        frame_of_reference_uid=source.frame_of_reference_uid,
        image_position_patient=EXACT_REPLICA_ORIGIN,
        image_orientation_patient=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        spacing_mm=(
            EXACT_REPLICA_IN_PLANE_SPACING[0],
            EXACT_REPLICA_IN_PLANE_SPACING[1],
            z_spacing,
        ),
        dimensions=cast(
            tuple[int, int, int], tuple(int(value) for value in source.dimensions)
        ),
        coordinate_system=source.coordinate_system,
        handedness=source.handedness,
        units=source.units,
        rescale_slope=source.rescale_slope,
        rescale_intercept=source.rescale_intercept,
        rescale_applied=source.rescale_applied,
    )


def _exact_replica_coverage_anchor(
    frame: FrameSpec,
) -> tuple[float, float, float]:
    z_extent = float(frame.spacing_mm[2]) * float(int(frame.dimensions[0]) - 1)
    return (
        EXACT_REPLICA_COVERAGE_XY[0],
        EXACT_REPLICA_COVERAGE_XY[1],
        float(frame.image_position_patient[2]) + z_extent / 2.0,
    )


def _assert_exact_binary64(value: float, *, label: str) -> None:
    numeric = float(value)
    if not math.isfinite(numeric) or abs(numeric) >= EXACT_BINARY64_LIMIT:
        raise AssertionError(f"{label} is outside the exact binary64 integer range")
    integer = int(numeric)
    if float(integer) != numeric:
        raise AssertionError(f"{label} is not an exactly representable integer binary64")


@cache
def _assert_exact_frame_basis(frame: FrameSpec) -> None:
    """Prove all affine coefficients and voxel centers are exact binary64 integers."""

    matrix = frame.voxel_to_world_matrix()
    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            _assert_exact_binary64(value, label=f"affine[{row_index},{column_index}]")
    k_indices, j_indices, i_indices = np.indices(
        tuple(int(value) for value in frame.dimensions), dtype=np.float64
    )
    for axis in range(3):
        centers = (
            (matrix[axis][3] + matrix[axis][0] * i_indices)
            + matrix[axis][1] * j_indices
        ) + matrix[axis][2] * k_indices
        if not bool(np.all(np.isfinite(centers))):
            raise AssertionError(f"voxel-center axis {axis} is nonfinite")
        if not bool(np.all(np.abs(centers) < EXACT_BINARY64_LIMIT)):
            raise AssertionError(f"voxel-center axis {axis} exceeds exact binary64 range")
        if not bool(np.all(centers == np.trunc(centers))):
            raise AssertionError(f"voxel-center axis {axis} is not integral binary64")


def _assert_exact_replica_frame(frame: FrameSpec, *, label: str) -> None:
    row = tuple(float(value) for value in frame.image_orientation_patient[:3])
    column = tuple(float(value) for value in frame.image_orientation_patient[3:])
    if any(value not in {-1.0, 0.0, 1.0} for value in (*row, *column)):
        raise AssertionError(f"{label} is not axis-aligned")
    if sum(value != 0.0 for value in row) != 1 or sum(value != 0.0 for value in column) != 1:
        raise AssertionError(f"{label} does not contain signed basis directions")
    if next(index for index, value in enumerate(row) if value != 0.0) == next(
        index for index, value in enumerate(column) if value != 0.0
    ):
        raise AssertionError(f"{label} row and column directions are not independent")
    for axis, value in enumerate((*frame.image_position_patient, *frame.spacing_mm)):
        _assert_exact_binary64(value, label=f"{label}[{axis}]")
    _assert_exact_frame_basis(frame)


def _assert_exact_translation_basis(
    frame: FrameSpec,
    *,
    translated_frame: FrameSpec,
    anchor: tuple[float, float, float],
) -> None:
    translated_origin = tuple(
        (float(old) + float(delta))
        for old, delta in zip(frame.image_position_patient, TRANSLATION)
    )
    restored_origin = tuple(
        (float(value) + float(delta))
        for value, delta in zip(translated_origin, INVERSE_TRANSLATION)
    )
    for axis, value in enumerate(translated_origin):
        _assert_exact_binary64(value, label=f"translated-origin[{axis}]")
    for axis, value in enumerate(restored_origin):
        _assert_exact_binary64(value, label=f"restored-origin[{axis}]")
        if value != frame.image_position_patient[axis]:
            raise AssertionError("exact translation removal did not restore the origin")
    dimensions = tuple(int(value) for value in frame.dimensions)
    k_indices, j_indices, i_indices = np.indices(dimensions, dtype=np.float64)
    base_matrix = frame.voxel_to_world_matrix()
    translated_matrix = translated_frame.voxel_to_world_matrix()
    for axis in range(3):
        base_centers = (
            (base_matrix[axis][3] + base_matrix[axis][0] * i_indices)
            + base_matrix[axis][1] * j_indices
        ) + base_matrix[axis][2] * k_indices
        translated_centers = (
            (translated_matrix[axis][3] + translated_matrix[axis][0] * i_indices)
            + translated_matrix[axis][1] * j_indices
        ) + translated_matrix[axis][2] * k_indices
        restored_centers = translated_centers + float(INVERSE_TRANSLATION[axis])
        if not bool(np.all(restored_centers == np.trunc(restored_centers))):
            raise AssertionError(f"restored voxel-center axis {axis} is not exact")
        if not bool(np.array_equal(restored_centers, base_centers)):
            raise AssertionError(f"translation removal changed voxel-center axis {axis}")
    translated_anchor = tuple(
        float(old) + float(delta) for old, delta in zip(anchor, TRANSLATION)
    )
    restored_anchor = tuple(
        float(value) + float(delta)
        for value, delta in zip(translated_anchor, INVERSE_TRANSLATION)
    )
    for axis, value in enumerate((*anchor, *translated_anchor, *restored_anchor)):
        _assert_exact_binary64(value, label=f"translation-anchor[{axis}]")


def assert_exact_surface_relation(
    source: PreparedQCInput,
    transformed: PreparedQCInput,
    *,
    translation_normalization_mm: tuple[float, float, float] = ZERO_TRANSLATION,
    label: str = "surface",
) -> None:
    """Prove raw and canonical world/scalar multisets before QC evaluation."""

    source_members = [
        (source.first_volume.coordinate_frame, source.first_array, "first")
    ]
    transformed_members = [
        (
            transformed.first_volume.coordinate_frame,
            transformed.first_array,
            "first",
        )
    ]
    if source.second_volume is not None and source.second_array is not None:
        if transformed.second_volume is None or transformed.second_array is None:
            raise AssertionError(f"{label} lost its paired volume")
        source_members.append(
            (source.second_volume.coordinate_frame, source.second_array, "second")
        )
        transformed_members.append(
            (
                transformed.second_volume.coordinate_frame,
                transformed.second_array,
                "second",
            )
        )
    for (source_frame, source_array, side), (
        transformed_frame,
        transformed_array,
        transformed_side,
    ) in zip(source_members, transformed_members):
        if side != transformed_side:
            raise AssertionError(f"{label} changed volume ordering")
        expected_raw = world_scalar_multiset(source_frame, source_array)
        actual_raw = world_scalar_multiset(
            transformed_frame,
            transformed_array,
            translation_normalization_mm=translation_normalization_mm,
        )
        if actual_raw != expected_raw:
            raise AssertionError(f"{label} {side} raw world/scalar multiset changed")
        expected_canonical = world_scalar_multiset(
            source_frame, source_array, canonical=True
        )
        actual_canonical = world_scalar_multiset(
            transformed_frame,
            transformed_array,
            canonical=True,
            translation_normalization_mm=translation_normalization_mm,
        )
        if actual_canonical != expected_canonical:
            raise AssertionError(
                f"{label} {side} canonical world/scalar multiset changed"
            )


def _assert_exact_legal_basis(
    base: PreparedQCInput,
    *,
    anchor: tuple[float, float, float],
    variant: int,
) -> None:
    """Prove every legal exact-coordinate surface before production QC runs."""

    hand = handedness_preserving(base, variant=variant + 1)
    reverse = reverse_i_and_k(base, variant=variant + 2)
    translation = translated(base, variant=variant + 3)
    surfaces = (PropertySurface(base, "exact_replica_base"), hand, reverse, translation)
    for surface in surfaces:
        frames = [surface.prepared.first_volume.coordinate_frame]
        if surface.prepared.second_volume is not None:
            frames.append(surface.prepared.second_volume.coordinate_frame)
        for index, frame in enumerate(frames):
            _assert_exact_replica_frame(frame, label=f"legal-frame-{index}")
    assert_exact_surface_relation(
        base, hand.prepared, label="handedness-construction"
    )
    assert_exact_surface_relation(
        base, reverse.prepared, label="reverse-construction"
    )
    assert_exact_surface_relation(
        base,
        translation.prepared,
        translation_normalization_mm=INVERSE_TRANSLATION,
        label="translation-construction",
    )
    _assert_exact_translation_basis(
        base.first_volume.coordinate_frame,
        translated_frame=translation.prepared.first_volume.coordinate_frame,
        anchor=anchor,
    )
    for value in conjugate_translation_affine(IDENTITY_AFFINE, TRANSLATION):
        _assert_exact_binary64(value, label="translation-conjugated-affine")
    for value in translation.overlap_lower_lps_mm + translation.overlap_upper_lps_mm:
        _assert_exact_binary64(value, label="translation-overlap-bound")


def exact_legal_replica(
    row: CensusCase,
    prepared: PreparedQCInput,
    *,
    variant: int,
) -> PropertySurface:
    """Build one exact property-only surface for one valid census row."""

    if not row.valid or row.geometry is None:
        raise ValueError("exact property replica requires a valid census row")
    if prepared.case != row:
        raise ValueError("replica row and prepared case identity do not match")
    z_spacing = _exact_replica_z_spacing(row)
    first_frame = _exact_replica_frame(prepared, z_spacing=z_spacing)
    second_source = prepared.second_volume
    if second_source is None:
        raise ValueError("exact property replica requires the paired volume")
    second_frame = replace(
        first_frame,
        frame_of_reference_uid=second_source.coordinate_frame.frame_of_reference_uid,
        rescale_slope=second_source.coordinate_frame.rescale_slope,
        rescale_intercept=second_source.coordinate_frame.rescale_intercept,
        rescale_applied=second_source.coordinate_frame.rescale_applied,
    )
    first_array = _property_replica_array(row, second=False)
    second_array = _property_replica_array(row, second=True)
    expected_shape = tuple(int(value) for value in first_frame.dimensions)
    if tuple(int(value) for value in first_array.shape) != expected_shape:
        raise ValueError("exact replica first array does not preserve row dimensions")
    if tuple(int(value) for value in second_array.shape) != expected_shape:
        raise ValueError("exact replica second array does not preserve row dimensions")
    anchor = _exact_replica_coverage_anchor(first_frame)
    _assert_exact_replica_frame(first_frame, label="exact-replica-first-frame")
    _assert_exact_replica_frame(second_frame, label="exact-replica-second-frame")
    for value in (*first_frame.image_position_patient, *first_frame.spacing_mm, *anchor):
        _assert_exact_binary64(value, label="exact-replica-basis")
    surface = _rebuild_prepared(
        prepared,
        name="exact_legal_replica",
        first_frame=first_frame,
        first_array=first_array,
        second_frame=second_frame,
        second_array=second_array,
        variant=variant,
        coverage_anchor=anchor,
    )
    _assert_exact_legal_basis(surface.prepared, anchor=anchor, variant=variant)
    return surface


def _stored_slice_bytes(array: np.ndarray, representation: str) -> bytes:
    """Encode one canonical slice using the independent fixture convention."""

    values = np.asarray(array, dtype=np.float64)
    if representation == "signed":
        if not bool(np.all(values == np.trunc(values))):
            raise ValueError("signed property slice is not integral")
        return np.ascontiguousarray(values, dtype="<i2").tobytes(order="C")
    if representation == "rescaled-fractional":
        values = values - 0.25
    if not bool(np.all(values == np.trunc(values))):
        raise ValueError("unsigned property slice is not integral")
    if float(values.min()) < 0.0 or float(values.max()) > 65535.0:
        raise ValueError("property slice is outside uint16")
    return np.ascontiguousarray(values, dtype="<u2").tobytes(order="C")


def _raw_slices(array: np.ndarray, representation: str) -> tuple[bytes, ...]:
    return tuple(
        _stored_slice_bytes(array[index], representation)
        for index in range(int(array.shape[0]))
    )


def _new_volume(
    volume: Volume,
    frame: FrameSpec,
    array: np.ndarray,
    *,
    ordinal: int,
    side: int,
    variant: int,
    transform_ref: str | None,
) -> Volume:
    digest = hashlib.sha256(array.tobytes(order="C")).hexdigest()
    return replace(
        volume,
        volume_id=ulid(8_000_000 + ordinal * 128 + variant * 8 + side),
        coordinate_frame=frame,
        transform_ref=transform_ref,
        blob_ref=replace(
            volume.blob_ref,
            sha256=digest,
            byte_size=int(array.nbytes),
        ),
        provenance_ref=ulid(8_100_000 + ordinal * 128 + variant * 8 + side),
    )


def _rebuild_prepared(
    prepared: PreparedQCInput,
    *,
    name: str,
    first_frame: FrameSpec,
    first_array: np.ndarray,
    second_frame: FrameSpec,
    second_array: np.ndarray,
    variant: int,
    first_raw_slices: tuple[bytes, ...] | None = None,
    second_raw_slices: tuple[bytes, ...] | None = None,
    first_instance_offset: int = 0,
    second_instance_offset: int = 0,
    first_source_instances: tuple[DicomInstance, ...] | None = None,
    second_source_instances: tuple[DicomInstance, ...] | None = None,
    case: CensusCase | None = None,
    coverage_anchor: tuple[float, float, float] | None = None,
    registration_affine: tuple[float, ...] = IDENTITY_AFFINE,
    overlap_lower_lps_mm: tuple[float, float, float] = (
        -10_000.0,
        -10_000.0,
        -10_000.0,
    ),
    overlap_upper_lps_mm: tuple[float, float, float] = (
        10_000.0,
        10_000.0,
        10_000.0,
    ),
) -> PropertySurface:
    first_array = _readonly(first_array)
    second_array = _readonly(second_array)
    first_transform_ref = f"transform:{ulid(8_200_000 + prepared.case.ordinal * 128 + variant)}"
    first_transform = transform_for_frame(
        first_frame,
        series_instance_uid=prepared.first_instances[0].series_instance_uid,
    )
    geometry = transformed_case = prepared.case if case is None else case
    if geometry.geometry is None:
        raise ValueError("property surface requires a valid geometry stratum")
    representation = geometry.geometry.pixels
    first_instances = _rebind_instances(
        prepared.first_instances if first_source_instances is None else first_source_instances,
        first_frame,
        raw_slices=(
            _raw_slices(first_array, representation)
            if first_raw_slices is None
            else first_raw_slices
        ),
        instance_offset=first_instance_offset,
        rescale_slope=first_frame.rescale_slope,
        rescale_intercept=first_frame.rescale_intercept,
        rescale_type="HU" if representation == "rescaled-fractional" else None,
    )
    second_instances = _rebind_instances(
        (prepared.second_instances or ())
        if second_source_instances is None
        else second_source_instances,
        second_frame,
        raw_slices=(
            _raw_slices(second_array, representation)
            if second_raw_slices is None
            else second_raw_slices
        ),
        instance_offset=second_instance_offset,
        rescale_slope=second_frame.rescale_slope,
        rescale_intercept=second_frame.rescale_intercept,
        rescale_type="HU" if representation == "rescaled-fractional" else None,
    )
    first_volume = _new_volume(
        prepared.first_volume,
        first_frame,
        first_array,
        ordinal=prepared.case.ordinal,
        side=0,
        variant=variant,
        transform_ref=first_transform_ref,
    )
    if prepared.second_volume is None:
        raise ValueError("property surface requires the paired fixture volume")
    second_volume = _new_volume(
        prepared.second_volume,
        second_frame,
        second_array,
        ordinal=prepared.case.ordinal,
        side=1,
        variant=variant,
        transform_ref=prepared.second_volume.transform_ref,
    )
    transformed = replace(
        prepared,
        case=transformed_case,
        first_instances=first_instances,
        first_volume=first_volume,
        first_array=first_array,
        first_transform=first_transform,
        first_transform_ref=first_transform_ref,
        first_transform_sha256=canonical_digest(first_transform),
        first_transform_provenance_ref=ulid(
            8_300_000 + prepared.case.ordinal * 128 + variant
        ),
        second_instances=second_instances,
        second_volume=second_volume,
        second_array=second_array,
        second_frame=second_frame,
        coverage_anchor=(
            prepared.coverage_anchor if coverage_anchor is None else coverage_anchor
        ),
    )
    return PropertySurface(
        prepared=replace(transformed, first_transform_sha256=canonical_digest(first_transform)),
        name=name,
        registration_affine=registration_affine,
        overlap_lower_lps_mm=overlap_lower_lps_mm,
        overlap_upper_lps_mm=overlap_upper_lps_mm,
    )


def handedness_preserving(prepared: PreparedQCInput, *, variant: int) -> PropertySurface:
    """Transpose i/j and reverse k with the exact accepted FrameSpec relation."""

    old_frame = prepared.first_volume.coordinate_frame
    row, column, normal = _frame_axes(old_frame)
    old_shape = tuple(int(value) for value in prepared.first_array.shape)
    old_sx, old_sy, old_sz = (float(value) for value in old_frame.spacing_mm)
    new_frame = replace(
        old_frame,
        image_position_patient=cast(
            tuple[float, float, float],
            tuple(
                float(old) + float(delta)
                for old, delta in zip(
                    old_frame.image_position_patient,
                    (
                        normal[axis] * old_sz * (old_shape[0] - 1)
                        for axis in range(3)
                    ),
                )
            ),
        ),
        image_orientation_patient=(*column, *row),
        spacing_mm=(old_sy, old_sx, old_sz),
        dimensions=(old_shape[0], old_shape[2], old_shape[1]),
    )
    second_frame = replace(
        prepared.second_volume.coordinate_frame if prepared.second_volume else old_frame,
        image_position_patient=new_frame.image_position_patient,
        image_orientation_patient=new_frame.image_orientation_patient,
        spacing_mm=new_frame.spacing_mm,
        dimensions=new_frame.dimensions,
    )
    first = np.transpose(prepared.first_array[::-1, :, :], (0, 2, 1))
    second = np.transpose(_paired_array(prepared)[::-1, :, :], (0, 2, 1))
    return _rebuild_prepared(
        prepared,
        name="handedness",
        first_frame=new_frame,
        first_array=first,
        second_frame=second_frame,
        second_array=second,
        variant=variant,
        first_source_instances=prepared.first_instances[::-1],
        second_source_instances=_paired_instances(prepared)[::-1],
    )


@cache
def _reverse_i_k_origin(frame: FrameSpec) -> tuple[float, float, float]:
    """Apply the accepted reverse-i/k origin relation in binary64."""

    row, _column, normal = _frame_axes(frame)
    shape = tuple(int(value) for value in frame.dimensions)
    raw = tuple(
        float(old)
        + row[axis] * float(frame.spacing_mm[0]) * (shape[2] - 1)
        + normal[axis] * float(frame.spacing_mm[2]) * (shape[0] - 1)
        for axis, old in enumerate(frame.image_position_patient)
    )
    return cast(tuple[float, float, float], raw)


def reverse_i_and_k(prepared: PreparedQCInput, *, variant: int) -> PropertySurface:
    """Reverse i and k with row sign reversal and the exact origin update."""

    old_frame = prepared.first_volume.coordinate_frame
    origin = _reverse_i_k_origin(old_frame)
    old_orientation = old_frame.image_orientation_patient
    new_row = (-old_orientation[0], -old_orientation[1], -old_orientation[2])
    new_orientation = (
        new_row[0],
        new_row[1],
        new_row[2],
        old_orientation[3],
        old_orientation[4],
        old_orientation[5],
    )
    new_frame = replace(
        old_frame,
        image_position_patient=origin,
        image_orientation_patient=new_orientation,
    )
    second_old = prepared.second_volume.coordinate_frame if prepared.second_volume else old_frame
    second_frame = replace(
        second_old,
        image_position_patient=origin,
        image_orientation_patient=new_frame.image_orientation_patient,
    )
    first = prepared.first_array[::-1, :, ::-1]
    second_source = _paired_array(prepared)
    second = second_source[::-1, :, ::-1]
    return _rebuild_prepared(
        prepared,
        name="reverse_i_k",
        first_frame=new_frame,
        first_array=first,
        second_frame=second_frame,
        second_array=second,
        variant=variant,
        first_source_instances=prepared.first_instances[::-1],
        second_source_instances=_paired_instances(prepared)[::-1],
    )


def translated(prepared: PreparedQCInput, *, variant: int) -> PropertySurface:
    """Apply the common finite LPS translation to both members and coverage."""

    first_frame = _translated_frame(prepared.first_volume.coordinate_frame, TRANSLATION)
    second_old = prepared.second_volume.coordinate_frame if prepared.second_volume else first_frame
    second_frame = _translated_frame(second_old, TRANSLATION)
    anchor = cast(
        tuple[float, float, float],
        tuple(
            old + delta for old, delta in zip(prepared.coverage_anchor, TRANSLATION)
        ),
    )
    lower = cast(
        tuple[float, float, float],
        tuple(-10_000.0 + delta for delta in TRANSLATION),
    )
    upper = cast(
        tuple[float, float, float],
        tuple(10_000.0 + delta for delta in TRANSLATION),
    )
    return _rebuild_prepared(
        prepared,
        name="translation",
        first_frame=first_frame,
        first_array=prepared.first_array,
        second_frame=second_frame,
        second_array=_paired_array(prepared),
        variant=variant,
        coverage_anchor=anchor,
        registration_affine=conjugate_translation_affine(IDENTITY_AFFINE, TRANSLATION),
        overlap_lower_lps_mm=lower,
        overlap_upper_lps_mm=upper,
    )


def round_trip(prepared: PreparedQCInput, *, variant: int) -> PropertySurface:
    """C-order serialize/decode with unchanged geometry and byte-equal arrays."""

    def decode(array: np.ndarray) -> np.ndarray:
        return _readonly(
            np.frombuffer(array.tobytes(order="C"), dtype=array.dtype)
            .copy()
            .reshape(array.shape)
        )

    return _rebuild_prepared(
        prepared,
        name="c_order_round_trip",
        first_frame=prepared.first_volume.coordinate_frame,
        first_array=decode(prepared.first_array),
        second_frame=prepared.second_volume.coordinate_frame
        if prepared.second_volume
        else prepared.first_volume.coordinate_frame,
        second_array=decode(_paired_array(prepared)),
        variant=variant,
    )


def cropped(
    prepared: PreparedQCInput, *, direction: str, variant: int
) -> PropertySurface:
    """Remove one physical terminal layer while retaining the old envelope."""

    if direction not in {"inferior", "superior"}:
        raise ValueError("crop direction must be inferior or superior")
    old_frame = prepared.first_volume.coordinate_frame
    _row, _column, normal = _frame_axes(old_frame)
    old_shape = tuple(int(value) for value in prepared.first_array.shape)
    if direction == "inferior":
        first_array = prepared.first_array[1:, :, :]
        second_array = _paired_array(prepared)[1:, :, :]
        origin = cast(
            tuple[float, float, float],
            tuple(
                old + normal[axis] * float(old_frame.spacing_mm[2])
                for axis, old in enumerate(old_frame.image_position_patient)
            ),
        )
    else:
        first_array = prepared.first_array[:-1, :, :]
        second_array = _paired_array(prepared)[:-1, :, :]
        origin = old_frame.image_position_patient
    dimensions = (old_shape[0] - 1, old_shape[1], old_shape[2])
    first_frame = replace(old_frame, image_position_patient=origin, dimensions=dimensions)
    second_old = prepared.second_volume.coordinate_frame if prepared.second_volume else old_frame
    second_frame = replace(second_old, image_position_patient=origin, dimensions=dimensions)
    if direction == "inferior":
        first_instances = prepared.first_instances[1:]
        second_instances = (
            _paired_instances(prepared)[1:]
        )
    else:
        first_instances = prepared.first_instances[:-1]
        second_instances = _paired_instances(prepared)[:-1]
    return _rebuild_prepared(
        prepared,
        name=f"crop_{direction}",
        first_frame=first_frame,
        first_array=first_array,
        second_frame=second_frame,
        second_array=second_array,
        variant=variant,
        first_source_instances=first_instances,
        second_source_instances=second_instances,
    )


def _integer_storage(values: np.ndarray) -> tuple[np.ndarray, int, float, float, str]:
    """Select an exact integer representation for one canonical array."""

    values64 = np.asarray(values, dtype=np.float64)
    if bool(np.all(values64 == np.trunc(values64))) and float(values64.min()) >= 0.0:
        if float(values64.max()) <= 65535.0:
            return np.ascontiguousarray(values64, dtype="<u2"), 0, 1.0, 0.0, "HU"
    if bool(np.all(values64 == np.trunc(values64))) and float(values64.min()) >= -32768.0:
        if float(values64.max()) <= 32767.0:
            return np.ascontiguousarray(values64, dtype="<i2"), 1, 1.0, 0.0, "HU"
    intercept = 0.25
    raw64 = values64 - intercept
    if not bool(np.all(raw64 == np.trunc(raw64))):
        raise ValueError("fixture has no exact integer HU representation")
    if float(raw64.min()) < 0.0 or float(raw64.max()) > 65535.0:
        raise ValueError("fixture HU representation is outside uint16")
    raw = np.ascontiguousarray(raw64, dtype="<u2")
    if not bool(np.array_equal(np.asarray(raw, dtype=np.float64) + intercept, values64)):
        raise ValueError("integer HU representation is not exact")
    return raw, 0, 1.0, intercept, "HU"


def hu_representation(prepared: PreparedQCInput, *, variant: int) -> PropertySurface:
    """Materialize an independently selected exact integer+rescale encoding."""

    first_raw, first_rep, slope, intercept, domain = _integer_storage(prepared.first_array)
    second_source = _paired_array(prepared)
    second_raw, second_rep, second_slope, second_intercept, second_domain = _integer_storage(
        second_source
    )
    if (first_rep, slope, intercept, domain) != (
        second_rep,
        second_slope,
        second_intercept,
        second_domain,
    ):
        raise ValueError("paired HU representation selected inconsistent scalar semantics")
    first_output, _codec, _fields = prepared.loader.materialize_array(
        first_raw,
        shape=(int(first_raw.shape[0]), int(first_raw.shape[1]), int(first_raw.shape[2])),
        pixel_representation=first_rep,
        scalar_domain=domain,
        intensity_unit=domain,
        rescale_slope=slope,
        rescale_intercept=intercept,
        rescale_type="HU" if domain == "HU" else None,
    )
    second_output, _codec2, _fields2 = prepared.loader.materialize_array(
        second_raw,
        shape=(int(second_raw.shape[0]), int(second_raw.shape[1]), int(second_raw.shape[2])),
        pixel_representation=second_rep,
        scalar_domain=domain,
        intensity_unit=domain,
        rescale_slope=second_slope,
        rescale_intercept=second_intercept,
        rescale_type="HU" if domain == "HU" else None,
    )
    if not bool(np.array_equal(np.asarray(first_output), prepared.first_array)):
        raise ValueError("HU storage did not reproduce the first canonical array exactly")
    if not bool(np.array_equal(np.asarray(second_output), second_source)):
        raise ValueError("HU storage did not reproduce the second canonical array exactly")
    old_first = prepared.first_volume.coordinate_frame
    old_second = prepared.second_volume.coordinate_frame if prepared.second_volume else old_first
    first_frame = replace(
        old_first,
        rescale_slope=slope,
        rescale_intercept=intercept,
        rescale_applied=domain != "STORED",
    )
    second_frame = replace(
        old_second,
        rescale_slope=second_slope,
        rescale_intercept=second_intercept,
        rescale_applied=second_domain != "STORED",
    )
    raw_first = tuple(first_raw[k].tobytes(order="C") for k in range(first_raw.shape[0]))
    raw_second = tuple(second_raw[k].tobytes(order="C") for k in range(second_raw.shape[0]))
    transformed_case = prepared.case
    if transformed_case.geometry is None:
        raise ValueError("HU property requires a valid geometry stratum")
    transformed_case = replace(
        transformed_case,
        geometry=replace(
            transformed_case.geometry,
            pixels="rescaled-fractional" if domain == "HU" else transformed_case.geometry.pixels,
        ),
    )
    return _rebuild_prepared(
        prepared,
        name="hu_representation",
        first_frame=first_frame,
        first_array=np.asarray(first_output),
        second_frame=second_frame,
        second_array=np.asarray(second_output),
        variant=variant,
        first_raw_slices=raw_first,
        second_raw_slices=raw_second,
        case=transformed_case,
    )


def conjugate_translation_affine(
    affine_row_major: tuple[float, ...], vector: tuple[float, float, float]
) -> tuple[float, ...]:
    """Conjugate an affine with shared 4x4 matrix routines."""

    if len(affine_row_major) != 16:
        raise ValueError("affine must contain sixteen values")
    matrix = cast(
        tuple[tuple[float, float, float, float], ...],
        tuple(
            tuple(affine_row_major[row * 4 + column] for column in range(4))
            for row in range(4)
        ),
    )
    translation = (
        (1.0, 0.0, 0.0, vector[0]),
        (0.0, 1.0, 0.0, vector[1]),
        (0.0, 0.0, 1.0, vector[2]),
        (0.0, 0.0, 0.0, 1.0),
    )
    from core.coordinates import invert_matrix

    conjugated = matmul(translation, matmul(matrix, invert_matrix(translation)))
    return tuple(value for row in conjugated for value in row)


__all__ = [
    "IDENTITY_AFFINE",
    "PropertySurface",
    "TRANSLATION",
    "assert_exact_surface_relation",
    "conjugate_translation_affine",
    "cropped",
    "exact_legal_replica",
    "handedness_preserving",
    "hu_representation",
    "reverse_i_and_k",
    "round_trip",
    "translated",
    "world_scalar_multiset",
]
