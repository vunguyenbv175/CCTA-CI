"""Independent synthetic DICOM inputs for QC_001-D.

This module is deliberately an input generator, not a QC oracle.  It does not
import the production QC package or any of its parameter modules.  The only
shared fixture utility is the existing Part-10 writer; all QC injection
labels, geometry strata, and pixel patterns are local to this benchmark.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Final

import numpy as np

from apps.workstation.volume_loader import CanonicalVolumeLoader
from core.coordinates import CoordinateTransform, FrameSpec, transform_for_frame
from core.dicom.instance import DicomInstance
from core.dicom.reader import read_dicom_instance
from core.dicom.stack import validate_stack
from core.digest import canonical_digest
from core.enums import ReleaseLane, VolumeKind
from core.volume import Volume, VolumeBlobRef, create_volume
from tests.apps.fixture_factory import build_instance

_ULID_ALPHABET: Final[str] = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
_DIMENSIONS: Final[tuple[int, int, int]] = (9, 64, 64)
_TARGET_CENTER: Final[tuple[float, float, float]] = (0.0, 0.0, 0.0)
_SHIFTED_CENTER: Final[tuple[float, float, float]] = (17.0, -11.0, 5.0)
_FRAME_UID_ROOT: Final[str] = "1.2.826.0.1.3680043.8.498.91"
_SYNTHETIC_STUDY_UID: Final[str] = "1.2.826.0.1.3680043.8.498.91.1"
_UNSIGNED_LEVELS: Final[dict[str, float]] = {
    "low": 0.0,
    "central": 20000.0,
    "reference": 10000.0,
    "outer_high": 50000.0,
    "bloom_core": 65000.0,
    "bloom_halo": 45000.0,
}
_SIGNED_LEVELS: Final[dict[str, float]] = {
    "low": -30000.0,
    "central": 10000.0,
    "reference": -5000.0,
    "outer_high": 22000.0,
    "bloom_core": 32760.0,
    "bloom_halo": 31000.0,
}


@dataclass(frozen=True, slots=True)
class GeometryStratum:
    """One of the 48 admitted storage/geometry representations."""

    orientation: str
    spacing: str
    normal: str
    origin: str
    pixels: str


@dataclass(frozen=True, slots=True)
class Injection:
    """Known synthetic injection state used by the independent oracle."""

    flag: str | None
    severity: str | None


@dataclass(frozen=True, slots=True)
class CensusCase:
    """A complete QC_001-D census row."""

    ordinal: int
    fixture_id: str
    geometry: GeometryStratum | None
    injection: Injection | None
    invalid_family: str | None = None
    matched_control: bool = False

    @property
    def valid(self) -> bool:
        return self.invalid_family is None


@dataclass(frozen=True, slots=True)
class PreparedQCInput:
    """Real-reader/materializer output needed by a QC test driver."""

    case: CensusCase
    source_manifest_sha256: str
    first_instances: tuple[DicomInstance, ...]
    first_volume: Volume
    first_array: np.ndarray
    first_transform: CoordinateTransform
    first_transform_ref: str
    first_transform_sha256: str
    first_transform_provenance_ref: str
    second_instances: tuple[DicomInstance, ...] | None
    second_volume: Volume | None
    second_array: np.ndarray | None
    second_frame: FrameSpec | None
    loader: CanonicalVolumeLoader
    coverage_anchor: tuple[float, float, float]


_GEOMETRY_STRATA: tuple[GeometryStratum, ...] = tuple(
    GeometryStratum(*values)
    for values in product(
        ("axis-aligned", "oblique"),
        ("isotropic", "anisotropic"),
        ("standard", "reversed"),
        ("zero", "nonzero"),
        ("unsigned", "signed", "rescaled-fractional"),
    )
)

_FAMILY_NAMES: Final[tuple[str, ...]] = (
    "CARDIAC_MOTION",
    "RESPIRATORY_MOTION",
    "STAIR_STEP",
    "POOR_CONTRAST",
    "EXCESSIVE_NOISE",
    "BLOOMING",
    "BEAM_HARDENING",
    "MISREGISTRATION",
    "INCOMPLETE_ANATOMY",
    "ABSTAINED",
)


def ulid(number: int) -> str:
    """Encode a small integer as a deterministic 26-character ULID."""

    if number < 0:
        raise ValueError("ULID number must be nonnegative")
    value = int(number)
    chars = ["0"] * 26
    for index in range(25, -1, -1):
        chars[index] = _ULID_ALPHABET[value & 31]
        value >>= 5
    if value:
        raise ValueError("ULID number is too large")
    return "".join(chars)


def geometry_strata() -> tuple[GeometryStratum, ...]:
    """Return the closed 2×2×2×2×3 geometry/scalar census grid."""

    return _GEOMETRY_STRATA


def census_cases() -> tuple[CensusCase, ...]:
    """Return exactly 912 valid rows plus 8 invalid/control pairs."""

    rows: list[CensusCase] = []
    ordinal = 0
    for geometry in _GEOMETRY_STRATA:
        rows.append(
            CensusCase(
                ordinal=ordinal,
                fixture_id=f"QC001-D-{ordinal:04d}",
                geometry=geometry,
                injection=Injection(None, None),
            )
        )
        ordinal += 1
        for flag in _FAMILY_NAMES[:-1]:
            for severity in ("MILD", "SEVERE"):
                rows.append(
                    CensusCase(
                        ordinal=ordinal,
                        fixture_id=f"QC001-D-{ordinal:04d}",
                        geometry=geometry,
                        injection=Injection(flag, severity),
                    )
                )
                ordinal += 1
    invalid_names = (
        "unsupported-transfer-syntax",
        "malformed-pixel-length",
        "multiframe",
        "identity-frame-mismatch",
        "missing-stack-member",
        "tilted-nonparallel-stack",
        "unknown-scalar-semantics",
        "absent-required-pair",
    )
    for invalid_name in invalid_names:
        rows.append(
            CensusCase(
                ordinal=ordinal,
                fixture_id=f"QC001-D-{ordinal:04d}-{invalid_name}",
                geometry=None,
                injection=None,
                invalid_family=invalid_name,
            )
        )
        ordinal += 1
        rows.append(
            CensusCase(
                ordinal=ordinal,
                fixture_id=f"QC001-D-{ordinal:04d}-{invalid_name}-control",
                geometry=_GEOMETRY_STRATA[ordinal % len(_GEOMETRY_STRATA)],
                injection=Injection(None, None),
                matched_control=True,
            )
        )
        ordinal += 1
    assert len(rows) == 928
    assert sum(row.valid for row in rows) == 920
    assert sum(row.invalid_family is not None for row in rows) == 8
    return tuple(rows)


def valid_census_cases() -> tuple[CensusCase, ...]:
    return tuple(row for row in census_cases() if row.valid)


def invalid_census_cases() -> tuple[CensusCase, ...]:
    return tuple(row for row in census_cases() if row.invalid_family is not None)


def source_manifest_digest(case: CensusCase) -> str:
    """Return the fixture manifest digest used by the QC source contract."""

    return _manifest_digest(case)


def _manifest_digest(case: CensusCase) -> str:
    descriptor = {
        "fixture_id": case.fixture_id,
        "geometry": None
        if case.geometry is None
        else {
            "orientation": case.geometry.orientation,
            "spacing": case.geometry.spacing,
            "normal": case.geometry.normal,
            "origin": case.geometry.origin,
            "pixels": case.geometry.pixels,
        },
        "injection": None
        if case.injection is None
        else {"flag": case.injection.flag, "severity": case.injection.severity},
        "invalid_family": case.invalid_family,
    }
    encoded = json.dumps(descriptor, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _geometry_definition(
    geometry: GeometryStratum,
    *,
    z_spacing: float = 20.0,
) -> tuple[FrameSpec, tuple[float, float, float]]:
    if geometry.orientation == "axis-aligned":
        row = (1.0, 0.0, 0.0)
        column = (0.0, 1.0, 0.0)
    else:
        row = (math.sqrt(3.0) / 2.0, 0.0, 0.5)
        column = (0.0, 1.0, 0.0)
    if geometry.normal == "reversed":
        row = (-row[0], -row[1], -row[2])
    normal = (
        row[1] * column[2] - row[2] * column[1],
        row[2] * column[0] - row[0] * column[2],
        row[0] * column[1] - row[1] * column[0],
    )
    center = _TARGET_CENTER if geometry.origin == "zero" else _SHIFTED_CENTER
    if geometry.spacing == "isotropic":
        in_plane = (1.0, 1.0)
    else:
        in_plane = (0.7, 1.3)
    row_offset = -23.5 * in_plane[0]
    column_offset = -23.5 * in_plane[1]
    origin = (
        center[0] - normal[0] * z_spacing * 4.0 + row[0] * row_offset + column[0] * column_offset,
        center[1] - normal[1] * z_spacing * 4.0 + row[1] * row_offset + column[1] * column_offset,
        center[2] - normal[2] * z_spacing * 4.0 + row[2] * row_offset + column[2] * column_offset,
    )
    frame_uid = f"{_FRAME_UID_ROOT}.{1 if geometry.normal == 'standard' else 2}"
    frame = FrameSpec(
        frame_of_reference_uid=frame_uid,
        image_position_patient=origin,
        image_orientation_patient=(*row, *column),
        spacing_mm=(in_plane[0], in_plane[1], z_spacing),
        dimensions=_DIMENSIONS,
    )
    return frame, center


def frame_for_case(case: CensusCase) -> tuple[FrameSpec, tuple[float, float, float]]:
    if case.geometry is None:
        raise ValueError("invalid census rows do not have a normal geometry")
    z_spacing = 20.0
    if case.injection is not None and case.injection.flag == "INCOMPLETE_ANATOMY":
        z_spacing = 18.0 if case.injection.severity == "MILD" else 14.0
    return _geometry_definition(case.geometry, z_spacing=z_spacing)


def _ring_mask(
    *,
    shift: int = 0,
    shape: tuple[int, int] = (_DIMENSIONS[1], _DIMENSIONS[2]),
) -> np.ndarray:
    rows, columns = shape
    outer = np.zeros(shape, dtype=bool)
    inner = np.zeros(shape, dtype=bool)
    row_start = math.ceil(0.35 * (rows - 1)) + shift
    column_start = math.ceil(0.35 * (columns - 1)) + shift
    row_end = math.floor(0.65 * (rows - 1)) + shift
    column_end = math.floor(0.65 * (columns - 1)) + shift
    outer[row_start : row_end + 1, column_start : column_end + 1] = True
    row_margin = max(2, (row_end - row_start + 1) // 6)
    column_margin = max(2, (column_end - column_start + 1) // 6)
    inner[
        row_start + row_margin : row_end - row_margin + 1,
        column_start + column_margin : column_end - column_margin + 1,
    ] = True
    return outer & ~inner


def _central_mask(shape: tuple[int, int, int]) -> np.ndarray:
    slices, rows, columns = shape
    u = np.linspace(-1.0, 1.0, columns)
    v = np.linspace(-1.0, 1.0, rows)
    mask = (np.abs(v)[None, :, None] <= 0.30) & (np.abs(u)[None, None, :] <= 0.30)
    return np.broadcast_to(mask, (slices, rows, columns)).copy()


def _reference_mask(shape: tuple[int, int, int]) -> np.ndarray:
    slices, rows, columns = shape
    u = np.linspace(-1.0, 1.0, columns)
    v = np.linspace(-1.0, 1.0, rows)
    radius = np.maximum(np.abs(v)[None, :, None], np.abs(u)[None, None, :])
    mask = (radius >= 0.45) & (radius <= 0.75)
    return np.broadcast_to(mask, (slices, rows, columns)).copy()


def _base_array(case: CensusCase) -> np.ndarray:
    shape = _DIMENSIONS
    central = _central_mask(shape)
    rows, columns = shape[1], shape[2]
    u = np.linspace(-1.0, 1.0, columns)
    v = np.linspace(-1.0, 1.0, rows)
    rho = np.sqrt(u[None, :] ** 2 + v[:, None] ** 2)
    if case.geometry and case.geometry.pixels == "signed":
        # Keep the signed phantom comfortably inside int16 while leaving
        # enough headroom for the high-density blooming profile.  The
        # normalized family metrics remain representation-independent.
        smooth = 1500.0 + 7500.0 * np.exp(-(rho * rho) / 0.55)
    else:
        smooth = 5000.0 + 25000.0 * np.exp(-(rho * rho) / 0.55)
    smooth += 250.0 * u[None, :] + 175.0 * v[:, None]
    levels = (
        _SIGNED_LEVELS
        if case.geometry and case.geometry.pixels == "signed"
        else _UNSIGNED_LEVELS
    )
    smooth[rho <= 0.24] = levels["central"] if levels is _SIGNED_LEVELS else 30000.0
    smooth[(rho >= 0.78) & (rho <= 0.85)] = levels["outer_high"]
    values = np.broadcast_to(smooth, shape).copy()
    values[:, _ring_mask()] += 500.0

    injection = case.injection
    if injection is None or injection.flag is None:
        return values
    flag = injection.flag
    severity = injection.severity
    assert flag is not None and severity is not None

    if flag == "RESPIRATORY_MOTION":
        # A single physical edge-band offset changes the adjacent-slice
        # residual while keeping the homogeneous central support used by the
        # noise family unchanged.  The band intersects the smooth phantom's
        # edge support but is not a repeated edge construction, so the
        # independent stair-step negative remains clean.  The selected slice
        # is inside the supported physical central band for both handedness
        # representations.
        amplitude = 7000.0 if severity == "MILD" else 15000.0
        if case.geometry is not None and case.geometry.pixels == "signed":
            band_limits = (
                (0.20, 0.40)
                if case.geometry.spacing == "anisotropic" and severity == "MILD"
                else (0.23, 0.30)
            )
        else:
            band_limits = (0.20, 0.40) if severity == "MILD" else (0.23, 0.30)
        respiratory_band = (rho >= band_limits[0]) & (rho <= band_limits[1])
        injection_slice = 1 if severity == "MILD" else 7
        # Keep the severe displacement on the physical high-side comparison
        # for the reversed storage stratum as well.  In that representation
        # the superior-positive traversal visits stored slice 1 before the
        # terminal excluded layer; using stored slice 7 would make the
        # modified slice the first member of its only supported comparison,
        # changing the Q.90 edge population rather than the residual answer.
        if case.geometry is not None and case.geometry.normal == "reversed":
            injection_slice = 1
        if levels is _SIGNED_LEVELS:
            amplitude = 4000.0 if severity == "MILD" else 9000.0
            # Keep the displacement on the physical central edge population;
            # broad slice-wide offsets would be a genuine high-frequency
            # contribution to the noise family as well.
            values[injection_slice, respiratory_band] -= amplitude
        else:
            values[injection_slice, respiratory_band] += amplitude
        return values

    if flag == "STAIR_STEP":
        # A compact rectangular boundary is deliberately moved on selected
        # slices.  It is large enough to satisfy the 32-edge support rule but
        # does not perturb the regional or whole-slice statistics used by the
        # other families.  The selected physical pairs are 2/8 (MILD) and
        # 4/8 (SEVERE), satisfying the frozen repeated-fraction predicates.
        state_slices = {2, 3} if severity == "MILD" else {1, 3, 5, 7}
        block = np.zeros((rows, columns), dtype=bool)
        block_rows = (20, 22) if levels is _SIGNED_LEVELS else (21, 24)
        block[block_rows[0] : block_rows[1], 27:39] = True
        for k in range(shape[0]):
            if k in state_slices:
                values[k, block] += 5000.0
        return values

    if flag == "POOR_CONTRAST":
        # A low-frequency in-plane ramp increases the deterministic central
        # MAD without introducing block edges into the noise/stair proxies.
        # The offset keeps the signed representation within its representable
        # range while deliberately making the central/reference CNR poor.
        amplitude = (
            (8000.0 if severity == "MILD" else 12000.0)
            if levels is _SIGNED_LEVELS
            else (22000.0 if severity == "MILD" else 30000.0)
        )
        u_ramp = np.broadcast_to(u[None, :], (rows, columns))
        values[:, central[0]] += amplitude * (u_ramp[central[0]] - 0.30)
        return values

    if flag == "EXCESSIVE_NOISE":
        anisotropic = case.geometry is not None and case.geometry.spacing == "anisotropic"
        amplitude = (
            (
                (5000.0 if severity == "MILD" else 8000.0)
                if anisotropic
                else (5000.0 if severity == "MILD" else 7000.0)
            )
            if levels is _SIGNED_LEVELS
            else (
                (9000.0 if severity == "MILD" else 16500.0)
                if anisotropic
                else (8500.0 if severity == "MILD" else 14500.0)
            )
        )
        _grid_k, grid_j, grid_i = np.indices(shape)
        # The same in-plane noise realization on each slice isolates the
        # multi-scale noise proxy from the adjacent-slice motion/stair tests.
        # Keep the injected high-frequency population on an alternating
        # checkerboard.  The production contrast proxy uses the same central
        # region, so injecting every central voxel would legitimately lower
        # its CNR and contaminate this one-family census state.  Half-support
        # checkerboard noise still exceeds both noise boundaries while its
        # robust central/reference separation remains a clean negative.
        pseudo = ((grid_j * 29 + grid_i * 43) % 11) - 5
        noise = amplitude * pseudo / 5.0
        checkerboard = ((grid_j + grid_i) % 2) == 0
        values[central & checkerboard] += noise[central & checkerboard]
        return values

    if flag == "BLOOMING":
        # Keep the local density profile spatially localized and repeat its
        # spatial support on every slice so BLOOMING does not manufacture a
        # STAIR_STEP transition.
        core_bounds = (30, 33)
        halo_bounds = (29, 34)
        core = np.zeros((rows, columns), dtype=bool)
        core[core_bounds[0] : core_bounds[1], core_bounds[0] : core_bounds[1]] = True
        halo = np.zeros((rows, columns), dtype=bool)
        halo[halo_bounds[0] : halo_bounds[1], halo_bounds[0] : halo_bounds[1]] = True
        halo[core] = False
        if levels is _SIGNED_LEVELS:
            low_halo, high_core, high_halo = 6000.0, 24000.0, 18000.0
        else:
            if severity == "MILD":
                low_halo, high_core, high_halo = 35000.0, 65000.0, 56000.0
            else:
                low_halo, high_core, high_halo = 35000.0, 65000.0, 56000.0
        high_halo_mask = np.zeros((rows, columns), dtype=bool)
        if severity == "MILD":
            selected_halo = np.zeros((rows, columns), dtype=bool)
            selected_halo[29, 31] = True
            selected_halo[33, 31] = True
            selected_halo[31, 29] = True
            high_halo_mask = selected_halo & halo
        else:
            # Eight intermediate-density halo voxels give the severe profile
            # a larger t50/t90 separation while remaining below the global
            # high-density candidate threshold.
            high_halo_mask[29, 29:34] = True
            high_halo_mask[30, 29:32] = True
            high_halo_mask &= halo
        for slice_index in range(values.shape[0]):
            values[slice_index, core] = high_core
            values[slice_index, high_halo_mask] = high_halo
            values[slice_index, halo & ~high_halo_mask] = low_halo
        # The profile is repeated on every physical slice.  Blooming is a
        # spatial high-density family; repeating its profile keeps a legal
        # volume's inter-slice edge population stable for STAIR_STEP.
        return values

    if flag == "BEAM_HARDENING":
        high_value = (
            (22000.0 if levels is _UNSIGNED_LEVELS else (7000.0 if severity == "MILD" else 8000.0))
            if severity == "MILD"
            else (26000.0 if levels is _UNSIGNED_LEVELS else 8000.0)
        )
        angle = np.arctan2(v[:, None], u[None, :])
        sector = (angle >= -math.pi) & (angle < -math.pi + math.pi / 4.0)
        annulus = (rho >= 0.45) & (rho <= 0.70)
        values[:, annulus & sector] = high_value
        return values

    # Pair-only injections are expressed by the second volume builder.
    return values


def first_array_for_case(case: CensusCase) -> np.ndarray:
    """Construct a canonical scalar array before DICOM representation."""

    return np.ascontiguousarray(_base_array(case), dtype=np.float64)


def second_array_for_case(case: CensusCase, first: np.ndarray) -> np.ndarray:
    """Construct the explicit registered phase used by pair families."""

    result = np.array(first, dtype=np.float64, copy=True)
    injection = case.injection
    if injection is None or injection.flag not in {"CARDIAC_MOTION", "MISREGISTRATION"}:
        return np.ascontiguousarray(result)
    if injection.flag == "CARDIAC_MOTION":
        # Rescale the central profile smoothly around its reference level.  It
        # gives the cardiac sharpness/residual composite a known state while
        # keeping the Q.95 edge residual below the independent
        # MISREGISTRATION boundary.  A checkerboard or sparse edge offset
        # couples the two paired families and is intentionally avoided.
        result_shape = (
            int(result.shape[0]),
            int(result.shape[1]),
            int(result.shape[2]),
        )
        central = _central_mask(result_shape)[0]
        if injection.severity == "MILD":
            factor = 0.65
        else:
            factor = 1.0
        reference = (
            10000.25
            if case.geometry and case.geometry.pixels == "signed"
            else 30000.25
            if case.geometry and case.geometry.pixels == "rescaled-fractional"
            else 30000.0
        )
        result[:, central] = reference + factor * (result[:, central] - reference)
        if injection.severity == "SEVERE":
            # Add a high-frequency second-phase sharpness change strictly in
            # the inner support.  The source edge-center residual population
            # remains unchanged, so MISREGISTRATION stays independently
            # negative while CARDIAC_MOTION receives its prescribed
            # sharpness contribution.
            rows, columns = result.shape[1:]
            u = np.linspace(-1.0, 1.0, columns)
            v = np.linspace(-1.0, 1.0, rows)
            inner = np.sqrt(v[:, None] * v[:, None] + u[None, :] * u[None, :]) < 0.20
            grid_j, grid_i = np.indices((rows, columns))
            checker = np.where((grid_j + grid_i) % 2 == 0, 1.0, -1.0)
            amplitude = 7000.0 if case.geometry and case.geometry.pixels == "signed" else 10000.0
            result[:, inner] += amplitude * checker[inner]
        return np.ascontiguousarray(result)
    # A half-plane phase offset supplies a high Q.95 residual population while
    # keeping the cardiac median edge residual below its boundary.  The jump
    # is placed in a source-slice gap, so the sampled second-phase sharpness
    # remains stable for the corresponding spacing representation.
    edge_strip = np.zeros(result.shape[1:], dtype=bool)
    if case.geometry is not None and case.geometry.spacing == "anisotropic":
        edge_strip[:, :30] = True
    elif case.geometry is not None and case.geometry.pixels == "signed":
        edge_strip[:27, :] = True
    else:
        edge_strip[:30, :] = True
    offset = (
        (2200.0 if injection.severity == "MILD" else 4200.0)
        if case.geometry and case.geometry.pixels == "signed"
        else (4400.0 if injection.severity == "MILD" else 8800.0)
    )
    result[:, edge_strip] += offset
    return np.ascontiguousarray(result)


def _representation(
    values: np.ndarray,
    pixels: str,
) -> tuple[np.ndarray, str, str, float, float, str | None]:
    if pixels == "unsigned":
        return np.rint(values).astype("<u2"), "STORED", "STORED", 1.0, 0.0, None
    if pixels == "signed":
        return np.rint(values).astype("<i2"), "STORED", "STORED", 1.0, 0.0, None
    slope = 1.0
    intercept = 0.25
    raw = np.rint((values - intercept) / slope).astype("<u2")
    return raw, "HU", "HU", slope, intercept, "HU"


def _write_stack(
    root: Path,
    *,
    case: CensusCase,
    frame: FrameSpec,
    canonical_values: np.ndarray,
    series_number: int,
    representation: str,
    invalid_family: str | None = None,
) -> tuple[Path, ...]:
    raw, _domain, _unit, slope, intercept, rescale_type = _representation(
        canonical_values, representation
    )
    series_uid = f"{_FRAME_UID_ROOT}.2.{series_number}"
    paths: list[Path] = []
    row, column = frame.image_orientation_patient[:3], frame.image_orientation_patient[3:]
    normal = (
        row[1] * column[2] - row[2] * column[1],
        row[2] * column[0] - row[0] * column[2],
        row[0] * column[1] - row[1] * column[0],
    )
    for k in range(int(canonical_values.shape[0])):
        position = (
            frame.image_position_patient[0] + normal[0] * frame.spacing_mm[2] * k,
            frame.image_position_patient[1] + normal[1] * frame.spacing_mm[2] * k,
            frame.image_position_patient[2] + normal[2] * frame.spacing_mm[2] * k,
        )
        data = raw[k].tobytes(order="C")
        if invalid_family == "malformed-pixel-length" and k == 0:
            data = data[:-2]
        number_of_frames = 2 if invalid_family == "multiframe" else 1
        transfer = (
            # Deflated Explicit VR LE stays outside the admitted set (Implicit
            # VR LE itself is admitted since DICOM_IMPLICIT_VR_LE_ADMISSION_001).
            "1.2.840.10008.1.2.1.99"
            if invalid_family == "unsupported-transfer-syntax"
            else "1.2.840.10008.1.2.1"
        )
        frame_uid = frame.frame_of_reference_uid
        if invalid_family == "identity-frame-mismatch" and k == _DIMENSIONS[0] - 1:
            frame_uid = frame_uid + ".9"
        path = root / f"slice-{k:02d}.dcm"
        build_instance(
            path,
            sop_instance_uid=f"{_FRAME_UID_ROOT}.3.{series_number}.{k + 1}",
            series_instance_uid=series_uid,
            study_instance_uid=_SYNTHETIC_STUDY_UID,
            frame_of_reference_uid=frame_uid,
            position=position,
            rows=_DIMENSIONS[1],
            columns=_DIMENSIONS[2],
            spacing=f"{frame.spacing_mm[1]}\\{frame.spacing_mm[0]}",
            thickness=str(frame.spacing_mm[2]),
            pixel_data=data,
            orientation="\\".join(str(float(value)) for value in frame.image_orientation_patient),
            transfer_syntax_uid=transfer,
            number_of_frames=number_of_frames,
            pixel_representation=1 if representation == "signed" else 0,
            rescale_slope=None if rescale_type is None else slope,
            rescale_intercept=None if rescale_type is None else intercept,
            rescale_type=rescale_type,
        )
        paths.append(path)
    return tuple(paths)


def _read_and_materialize(
    paths: tuple[Path, ...],
    *,
    frame: FrameSpec,
    domain: str,
    loader: CanonicalVolumeLoader,
) -> tuple[tuple[DicomInstance, ...], np.ndarray, FrameSpec, VolumeBlobRef]:
    instances = tuple(read_dicom_instance(path) for path in paths)
    stack = validate_stack(instances)
    if not stack.admitted or stack.slice_spacing_mm is None:
        raise ValueError("synthetic valid stack was not admitted")
    materialized = loader.materialize_dicom_stack(
        stack.instances,
        scalar_domain=domain,
        intensity_unit=domain,
        frame=frame,
    )
    if materialized.frame is None:
        raise ValueError("materializer did not return the FrameSpec")
    return stack.instances, materialized.array, materialized.frame, materialized.volume_blob_ref


def prepare_valid_case(root: Path, case: CensusCase) -> PreparedQCInput:
    """Execute synthetic DICOM -> reader -> stack -> materializer for one row."""

    if not case.valid or case.geometry is None:
        raise ValueError("prepare_valid_case requires a valid census row")
    frame, anchor = frame_for_case(case)
    first_values = first_array_for_case(case)
    second_values = second_array_for_case(case, first_values)
    representation = case.geometry.pixels
    domain = "HU" if representation == "rescaled-fractional" else "STORED"
    case_root = root / case.fixture_id
    first_paths = _write_stack(
        case_root / "first",
        case=case,
        frame=frame,
        canonical_values=first_values,
        series_number=case.ordinal * 2 + 1,
        representation=representation,
    )
    second_paths = _write_stack(
        case_root / "second",
        case=case,
        frame=frame,
        canonical_values=second_values,
        series_number=case.ordinal * 2 + 2,
        representation=representation,
    )
    loader = CanonicalVolumeLoader(root / "blobs" / case.fixture_id)
    first_instances, first_array, materialized_frame, first_blob_ref = _read_and_materialize(
        first_paths, frame=frame, domain=domain, loader=loader
    )
    second_instances, second_array, second_frame, second_blob_ref = _read_and_materialize(
        second_paths, frame=frame, domain=domain, loader=loader
    )
    case_base = 10_000 + case.ordinal * 32
    study_ref = f"study_id:{ulid(case_base)}"
    first_series_ref = f"series_id:{ulid(case_base + 1)}"
    second_series_ref = f"series_id:{ulid(case_base + 2)}"
    first_transform_ref = f"transform:{ulid(case_base + 5)}"
    first_transform_provenance_ref = ulid(case_base + 6)
    first_transform = transform_for_frame(
        materialized_frame, series_instance_uid=first_instances[0].series_instance_uid
    )
    first_volume = create_volume(
        study_ref=study_ref,
        source_series_refs=(first_series_ref,),
        kind=VolumeKind.CANONICAL,
        coordinate_frame=materialized_frame,
        blob_ref=first_blob_ref,
        provenance_ref=ulid(case_base + 3),
        release_lane=ReleaseLane.PERSONAL_RESEARCH,
        transform_ref=first_transform_ref,
        volume_id=ulid(case_base + 4),
    )
    second_volume = create_volume(
        study_ref=study_ref,
        source_series_refs=(second_series_ref,),
        kind=VolumeKind.CANONICAL,
        coordinate_frame=second_frame,
        blob_ref=second_blob_ref,
        provenance_ref=ulid(case_base + 7),
        release_lane=ReleaseLane.PERSONAL_RESEARCH,
        volume_id=ulid(case_base + 8),
    )
    return PreparedQCInput(
        case=case,
        source_manifest_sha256=_manifest_digest(case),
        first_instances=first_instances,
        first_volume=first_volume,
        first_array=first_array,
        first_transform=first_transform,
        first_transform_ref=first_transform_ref,
        first_transform_sha256=canonical_digest(first_transform),
        first_transform_provenance_ref=first_transform_provenance_ref,
        second_instances=second_instances,
        second_volume=second_volume,
        second_array=second_array,
        second_frame=second_frame,
        loader=loader,
        coverage_anchor=anchor,
    )


def build_invalid_stack(
    root: Path,
    case: CensusCase,
) -> tuple[Path, ...]:
    """Write one out-of-envelope DICOM stack for the invalid census rows."""

    if case.invalid_family is None:
        raise ValueError("build_invalid_stack requires an invalid census row")
    invalid = case.invalid_family
    if invalid in {"unsupported-transfer-syntax", "malformed-pixel-length", "multiframe"}:
        frame, _anchor = _geometry_definition(_GEOMETRY_STRATA[0])
        values = first_array_for_case(
            CensusCase(case.ordinal, case.fixture_id, _GEOMETRY_STRATA[0], Injection(None, None))
        )
        return _write_stack(
            root / case.fixture_id / "invalid",
            case=case,
            frame=frame,
            canonical_values=values,
            series_number=20_000 + case.ordinal,
            representation="unsigned",
            invalid_family=invalid,
        )
    if invalid in {
        "identity-frame-mismatch",
        "missing-stack-member",
        "tilted-nonparallel-stack",
        "unknown-scalar-semantics",
    }:
        frame, _anchor = _geometry_definition(_GEOMETRY_STRATA[0])
        values = first_array_for_case(
            CensusCase(case.ordinal, case.fixture_id, _GEOMETRY_STRATA[0], Injection(None, None))
        )
        if invalid == "missing-stack-member":
            # A two-slice stack with 20 mm thickness and a 40 mm gap is
            # rejected as incomplete by the real stack validator.
            values = values[:2]
        if invalid == "tilted-nonparallel-stack":
            path_root = root / case.fixture_id / "invalid"
            first_paths = _write_stack(
                path_root,
                case=case,
                frame=frame,
                canonical_values=values,
                series_number=20_000 + case.ordinal,
                representation="unsigned",
            )
            # Rebuild the final slice with a nonparallel orientation.  The
            # reader/stack path, rather than a post-hoc QC exception, owns the
            # refusal.
            last = first_paths[-1]
            raw = values[-1].astype("<u2").tobytes(order="C")
            build_instance(
                last,
                sop_instance_uid=f"{_FRAME_UID_ROOT}.3.{20_000 + case.ordinal}.9",
                series_instance_uid=f"{_FRAME_UID_ROOT}.2.{20_000 + case.ordinal}",
                study_instance_uid=_SYNTHETIC_STUDY_UID,
                frame_of_reference_uid=frame.frame_of_reference_uid,
                position=(
                    frame.image_position_patient[0]
                    + frame.image_orientation_patient[1] * frame.spacing_mm[2] * 8.0,
                    frame.image_position_patient[1]
                    + frame.image_orientation_patient[1] * frame.spacing_mm[2] * 8.0,
                    frame.image_position_patient[2]
                    + frame.image_orientation_patient[1] * frame.spacing_mm[2] * 8.0,
                ),
                rows=_DIMENSIONS[1],
                columns=_DIMENSIONS[2],
                spacing=f"{frame.spacing_mm[1]}\\{frame.spacing_mm[0]}",
                thickness=str(frame.spacing_mm[2]),
                pixel_data=raw,
                orientation="1\\0\\0\\0.01\\0.99995\\0",
            )
            return first_paths
        path_root = root / case.fixture_id / "invalid"
        paths: list[Path] = []
        count = 2 if invalid == "missing-stack-member" else _DIMENSIONS[0]
        for index in range(count):
            step = 40.0 if invalid == "missing-stack-member" and index else float(index)
            matrix = frame.voxel_to_world_matrix()
            position = (
                frame.image_position_patient[0] + matrix[0][2] * step,
                frame.image_position_patient[1] + matrix[1][2] * step,
                frame.image_position_patient[2] + matrix[2][2] * step,
            )
            path = path_root / f"slice-{index:02d}.dcm"
            build_instance(
                path,
                sop_instance_uid=f"{_FRAME_UID_ROOT}.3.{20_000 + case.ordinal}.{index + 1}",
                series_instance_uid=f"{_FRAME_UID_ROOT}.2.{20_000 + case.ordinal}",
                study_instance_uid=_SYNTHETIC_STUDY_UID,
                frame_of_reference_uid=(
                    frame.frame_of_reference_uid + ".9"
                    if invalid == "identity-frame-mismatch" and index == count - 1
                    else frame.frame_of_reference_uid
                ),
                position=position,
                rows=_DIMENSIONS[1],
                columns=_DIMENSIONS[2],
                spacing=f"{frame.spacing_mm[1]}\\{frame.spacing_mm[0]}",
                thickness=str(frame.spacing_mm[2]),
                pixel_data=values[index].astype("<u2").tobytes(order="C"),
            )
            paths.append(path)
        return tuple(paths)
    raise ValueError(f"invalid DICOM stack family is not file-based: {invalid}")


def source_bound_instances(paths: tuple[Path, ...]) -> tuple[DicomInstance, ...]:
    """Reduce refusing fixture files without applying the production reader policy."""

    import pydicom

    instances: list[DicomInstance] = []
    try:
        for path in paths:
            dataset = pydicom.dcmread(str(path), force=False, defer_size=None)
            pixel_data = getattr(dataset, "PixelData", None)
            if not isinstance(pixel_data, (bytes, bytearray, memoryview)):
                raise ValueError("source fixture has no concrete PixelData bytes")
            file_meta = getattr(dataset, "file_meta", None)
            transfer_syntax = getattr(file_meta, "TransferSyntaxUID", None)
            if transfer_syntax is None:
                raise ValueError("source fixture has no transfer syntax identity")

            def optional_float(name: str) -> float | None:
                value = getattr(dataset, name, None)
                return None if value is None else float(value)

            def optional_text(name: str) -> str | None:
                value = getattr(dataset, name, None)
                return None if value is None else str(value)

            pixel_bytes = bytes(pixel_data)
            slope = optional_float("RescaleSlope")
            intercept = optional_float("RescaleIntercept")
            position = dataset.ImagePositionPatient
            orientation = dataset.ImageOrientationPatient
            spacing = dataset.PixelSpacing
            instances.append(
                DicomInstance(
                    study_instance_uid=str(dataset.StudyInstanceUID),
                    series_instance_uid=str(dataset.SeriesInstanceUID),
                    sop_instance_uid=str(dataset.SOPInstanceUID),
                    frame_of_reference_uid=str(dataset.FrameOfReferenceUID),
                    modality=str(dataset.Modality),
                    sop_class_uid=str(dataset.SOPClassUID),
                    image_type=tuple(str(value) for value in dataset.ImageType),
                    series_description=optional_text("SeriesDescription"),
                    body_part_examined=optional_text("BodyPartExamined"),
                    protocol_name=optional_text("ProtocolName"),
                    image_position_patient=(
                        float(position[0]),
                        float(position[1]),
                        float(position[2]),
                    ),
                    image_orientation_patient=(
                        float(orientation[0]),
                        float(orientation[1]),
                        float(orientation[2]),
                        float(orientation[3]),
                        float(orientation[4]),
                        float(orientation[5]),
                    ),
                    pixel_spacing=(float(spacing[0]), float(spacing[1])),
                    slice_thickness_mm=optional_float("SliceThickness"),
                    rows=int(dataset.Rows),
                    columns=int(dataset.Columns),
                    pixel_representation=int(dataset.PixelRepresentation),
                    bits_allocated=int(dataset.BitsAllocated),
                    bits_stored=int(dataset.BitsStored),
                    high_bit=int(dataset.HighBit),
                    rescale_slope=1.0 if slope is None else slope,
                    rescale_intercept=0.0 if intercept is None else intercept,
                    rescale_type=optional_text("RescaleType"),
                    contrast_bolus_agent=optional_text("ContrastBolusAgent"),
                    contrast_bolus_volume_ml=optional_float("ContrastBolusVolume"),
                    contrast_bolus_start_time=optional_text("ContrastBolusStartTime"),
                    contrast_bolus_stop_time=optional_text("ContrastBolusStopTime"),
                    contrast_flow_rate=optional_float("ContrastFlowRate"),
                    contrast_flow_duration=optional_float("ContrastFlowDuration"),
                    temporal_position_index=(
                        None
                        if getattr(dataset, "TemporalPositionIndex", None) is None
                        else int(dataset.TemporalPositionIndex)
                    ),
                    trigger_time_ms=optional_float("TriggerTime"),
                    cardiac_synchronization_technique=optional_text(
                        "CardiacSynchronizationTechnique"
                    ),
                    cardiac_signal_source=optional_text("CardiacSignalSource"),
                    acquisition_number=(
                        None
                        if getattr(dataset, "AcquisitionNumber", None) is None
                        else int(dataset.AcquisitionNumber)
                    ),
                    number_of_temporal_positions=(
                        None
                        if getattr(dataset, "NumberOfTemporalPositions", None) is None
                        else int(dataset.NumberOfTemporalPositions)
                    ),
                    transfer_syntax_uid=str(transfer_syntax),
                    pixel_data=pixel_bytes,
                    pixel_data_byte_length=len(pixel_bytes),
                    number_of_frames=int(getattr(dataset, "NumberOfFrames", 1)),
                    samples_per_pixel=int(dataset.SamplesPerPixel),
                    photometric_interpretation=str(dataset.PhotometricInterpretation),
                    study_date=optional_text("StudyDate"),
                )
            )
    except Exception as exc:
        raise AssertionError("could not derive source-bound refusal metadata") from exc
    return tuple(instances)


__all__ = [
    "CensusCase",
    "GeometryStratum",
    "Injection",
    "PreparedQCInput",
    "build_invalid_stack",
    "census_cases",
    "first_array_for_case",
    "frame_for_case",
    "geometry_strata",
    "prepare_valid_case",
    "second_array_for_case",
    "source_bound_instances",
    "source_manifest_digest",
    "ulid",
    "valid_census_cases",
]
