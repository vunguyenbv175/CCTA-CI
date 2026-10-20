"""Coordinate systems and transforms (Master Plan §7).

The mandatory chain is ``voxel <-> world <-> mesh <-> centerline``. The
canonical world frame IS DICOM patient space (LPS, mm). Transforms are
versioned artifacts with provenance - never hidden state - and every
transform exposes round-trip verification hooks required by §7.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

from core.enums import CoordinateSystem, Handedness, SpaceKind, Units
from core.errors import CanonicalValidationError

Matrix4 = tuple[tuple[float, float, float, float], ...]
Vec3 = tuple[float, float, float]
Cosines3 = tuple[Vec3, Vec3, Vec3]

_ROUND_TRIP_TOLERANCE_MM = 1e-6
_DESCRIPTOR_TOLERANCE = 1e-6
_TRANSLATION_TOLERANCE = 1e-9


# ---------------------------------------------------------------------------
# Point types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VoxelCoordinate:
    """Integer voxel index within a volume grid."""

    i: int
    j: int
    k: int


@dataclass(frozen=True)
class WorldCoordinate:
    """Millimetre position in canonical world (DICOM patient LPS) space."""

    x: float
    y: float
    z: float

    def __post_init__(self) -> None:
        for name in ("x", "y", "z"):
            value = getattr(self, name)
            if not math.isfinite(value):
                raise ValueError(f"WorldCoordinate.{name} must be finite")


@dataclass(frozen=True)
class MeshCoordinate:
    """Vertex position in a mesh's local frame."""

    u: float
    v: float
    w: float
    vertex_id: int | None = None

    def __post_init__(self) -> None:
        for name in ("u", "v", "w"):
            value = getattr(self, name)
            if not math.isfinite(value):
                raise ValueError(f"MeshCoordinate.{name} must be finite")


@dataclass(frozen=True)
class CenterlineCoordinate:
    """Arc-length parametrized centerline position.

    ``s_mm`` is distance along the vessel polyline from its origin;
    ``offset_mm`` is a signed radial offset in the local cross-section plane.
    """

    s_mm: float
    offset_mm: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.s_mm) or self.s_mm < 0.0:
            raise ValueError("CenterlineCoordinate.s_mm must be finite and >= 0")
        if not math.isfinite(self.offset_mm):
            raise ValueError("CenterlineCoordinate.offset_mm must be finite")


# ---------------------------------------------------------------------------
# Frame specification (§7 mandated descriptors)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FrameSpec:
    """Complete geometric descriptor of a voxel grid (Part B §3.3)."""

    frame_of_reference_uid: str
    image_position_patient: Vec3
    image_orientation_patient: tuple[float, float, float, float, float, float]
    spacing_mm: Vec3
    dimensions: tuple[int, int, int]
    coordinate_system: CoordinateSystem = CoordinateSystem.DICOM_PATIENT
    handedness: Handedness = Handedness.LPS
    units: Units = Units.MM
    rescale_slope: float = 1.0
    rescale_intercept: float = 0.0
    rescale_applied: bool = False

    def __post_init__(self) -> None:
        problems: list[str] = []
        if not self.frame_of_reference_uid:
            problems.append("frame_of_reference_uid is mandatory")
        row = self.image_orientation_patient[:3]
        col = self.image_orientation_patient[3:]
        if len(self.image_orientation_patient) != 6:
            problems.append("image_orientation_patient must be a 6-vector")
        else:
            row_norm = math.sqrt(sum(c * c for c in row))
            col_norm = math.sqrt(sum(c * c for c in col))
            if abs(row_norm - 1.0) > 1e-6 or abs(col_norm - 1.0) > 1e-6:
                problems.append(
                    "image_orientation_patient must hold two unit direction cosines"
                )
        if any(s <= 0 for s in self.spacing_mm):
            problems.append("spacing_mm must be strictly positive")
        if any(d <= 0 for d in self.dimensions):
            problems.append("dimensions must be strictly positive")
        if problems:
            raise CanonicalValidationError(problems)

    def voxel_to_world_matrix(self) -> Matrix4:
        """Affine mapping voxel indices -> world millimetres."""
        ox, oy, oz = self.image_position_patient
        row_cos, col_cos = (
            self.image_orientation_patient[:3],
            self.image_orientation_patient[3:],
        )
        sx, sy, sz = self.spacing_mm
        normal = _cross(row_cos, col_cos)
        return (
            (row_cos[0] * sx, col_cos[0] * sy, normal[0] * sz, ox),
            (row_cos[1] * sx, col_cos[1] * sy, normal[1] * sz, oy),
            (row_cos[2] * sx, col_cos[2] * sy, normal[2] * sz, oz),
            (0.0, 0.0, 0.0, 1.0),
        )


# ---------------------------------------------------------------------------
# Affine algebra (pure stdlib, 4x4 homogeneous matrices)
# ---------------------------------------------------------------------------


def _matrix(rows: list[list[float]]) -> Matrix4:
    return cast(Matrix4, tuple(tuple(row) for row in rows))


def identity_matrix() -> Matrix4:
    return (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )


def matmul(a: Matrix4, b: Matrix4) -> Matrix4:
    rows = [
        [sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)] for i in range(4)
    ]
    return _matrix(rows)


def invert_matrix(m: Matrix4) -> Matrix4:
    """General 4x4 inverse via Gauss-Jordan elimination."""
    aug = [list(row) + [1.0 if i == j else 0.0 for j in range(4)] for i, row in enumerate(m)]
    for col in range(4):
        pivot_row = max(range(col, 4), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot_row][col]) < 1e-12:
            raise ValueError("matrix is singular; cannot build transform pair")
        aug[col], aug[pivot_row] = aug[pivot_row], aug[col]
        pivot = aug[col][col]
        aug[col] = [v / pivot for v in aug[col]]
        for r in range(4):
            if r == col:
                continue
            factor = aug[r][col]
            if factor != 0.0:
                aug[r] = [vr - factor * vc for vr, vc in zip(aug[r], aug[col])]
    return _matrix([row[4:] for row in aug])


def apply_matrix(
    m: Matrix4, point: VoxelCoordinate | WorldCoordinate | MeshCoordinate
) -> Vec3:
    if isinstance(point, VoxelCoordinate):
        px, py, pz = float(point.i), float(point.j), float(point.k)
    elif isinstance(point, WorldCoordinate):
        px, py, pz = point.x, point.y, point.z
    else:
        px, py, pz = point.u, point.v, point.w
    return (
        m[0][0] * px + m[0][1] * py + m[0][2] * pz + m[0][3],
        m[1][0] * px + m[1][1] * py + m[1][2] * pz + m[1][3],
        m[2][0] * px + m[2][1] * py + m[2][2] * pz + m[2][3],
    )


def transform_for_frame(
    frame: FrameSpec,
    *,
    version: int = 1,
    series_instance_uid: str = "",
) -> CoordinateTransform:
    """Canonical voxel->world transform derived from a frame spec (§7).

    ``series_instance_uid`` is a mandatory descriptor (Part A §1.2); the
    source series pins exactly which acquisition the grid came from.
    """
    if not series_instance_uid:
        raise CanonicalValidationError(
            "series_instance_uid is mandatory for coordinate transforms"
        )
    row_cos, col_cos = (
        frame.image_orientation_patient[:3],
        frame.image_orientation_patient[3:],
    )
    normal = _cross(row_cos, col_cos)
    return CoordinateTransform(
        transform_id=f"voxel-to-world:{frame.frame_of_reference_uid}",
        from_space=SpaceKind.VOXEL,
        to_space=SpaceKind.WORLD,
        method="dicom_image_position_orientation_spacing",
        matrix=frame.voxel_to_world_matrix(),
        frame_of_reference_uid=frame.frame_of_reference_uid,
        series_instance_uid=series_instance_uid,
        direction_cosines=(row_cos, col_cos, normal),
        translation=frame.image_position_patient,
        handedness=frame.handedness,
        units=frame.units,
        version=version,
    )


def _cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _transpose_cosines(cosines: Cosines3) -> Cosines3:
    return cast(
        Cosines3,
        tuple(tuple(cosines[r][c] for r in range(3)) for c in range(3)),
    )


# ---------------------------------------------------------------------------
# Transform artifact payload
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CoordinateTransform:
    """A versioned, explicit spatial transform (P7 / Master Plan §7).

    This is the payload bound inside a ``DerivedArtifact(kind=
    COORDINATE_TRANSFORM)``; it must never live as hidden runtime state.

    Mandatory geometric descriptors (Part A §1.2): explicit handedness,
    physical units, orientation direction cosines (3x3), translation vector,
    source ``FrameOfReferenceUID`` and source ``SeriesInstanceUID``. All six
    are required at creation; nothing is defaulted in.
    """

    transform_id: str
    from_space: SpaceKind
    to_space: SpaceKind
    method: str
    matrix: Matrix4
    frame_of_reference_uid: str
    series_instance_uid: str
    direction_cosines: Cosines3
    translation: Vec3
    #: Mandatory geometric descriptors (Part A §1.2) - no defaults, omission
    #: must fail loudly instead of silently becoming LPS/MM.
    handedness: Handedness
    units: Units
    version: int = 1
    round_trip_error_mm: float | None = None

    def __post_init__(self) -> None:
        problems: list[str] = []
        if self.from_space == self.to_space:
            problems.append("from_space and to_space must differ")
        if not self.method:
            problems.append("method is required")
        if not self.frame_of_reference_uid:
            problems.append("frame_of_reference_uid is required")
        if not self.series_instance_uid:
            problems.append("series_instance_uid is required (Part A §1.2)")
        if not isinstance(self.handedness, Handedness):
            problems.append("handedness must be an explicit Handedness value")
        if self.units is not Units.MM:
            problems.append(
                "spatial transforms carry millimetre units (units=Units.MM)"
            )
        if self.version < 1:
            problems.append("version must be >= 1")
        shape_ok = len(self.matrix) == 4 and all(len(r) == 4 for r in self.matrix)
        if not shape_ok:
            problems.append("matrix must be 4x4")
        else:
            try:
                invert_matrix(self.matrix)
            except ValueError as exc:
                problems.append(str(exc))
        _validate_direction_cosines(self.direction_cosines, problems)
        _validate_translation(self.translation, problems)
        if shape_ok and len(self.translation) == 3:
            matrix_translation = (
                self.matrix[0][3],
                self.matrix[1][3],
                self.matrix[2][3],
            )
            mismatched = any(
                not math.isclose(
                    t, m, rel_tol=_TRANSLATION_TOLERANCE, abs_tol=_TRANSLATION_TOLERANCE
                )
                for t, m in zip(self.translation, matrix_translation)
            )
            if mismatched:
                problems.append(
                    "translation vector must match the matrix translation column"
                )
        if problems:
            raise CanonicalValidationError(problems)

    @property
    def inverse(self) -> CoordinateTransform:
        inv_matrix = invert_matrix(self.matrix)
        return CoordinateTransform(
            transform_id=f"{self.transform_id}-inv",
            from_space=self.to_space,
            to_space=self.from_space,
            method=f"inverse({self.method})",
            matrix=inv_matrix,
            frame_of_reference_uid=self.frame_of_reference_uid,
            series_instance_uid=self.series_instance_uid,
            direction_cosines=_transpose_cosines(self.direction_cosines),
            translation=(inv_matrix[0][3], inv_matrix[1][3], inv_matrix[2][3]),
            handedness=self.handedness,
            units=self.units,
            version=self.version,
            round_trip_error_mm=self.round_trip_error_mm,
        )

    def compose_with(self, second: CoordinateTransform) -> CoordinateTransform:
        """Compose ``self`` then ``second``; spaces must connect exactly."""
        if self.to_space != second.from_space:
            raise CanonicalValidationError(
                f"cannot compose {self.from_space}->{self.to_space} "
                f"with {second.from_space}->{second.to_space}"
            )
        product = matmul(second.matrix, self.matrix)
        cosines = cast(
            Cosines3,
            tuple(
                tuple(
                    sum(
                        second.direction_cosines[i][k] * self.direction_cosines[k][j]
                        for k in range(3)
                    )
                    for j in range(3)
                )
                for i in range(3)
            ),
        )
        return CoordinateTransform(
            transform_id=f"{self.transform_id}+{second.transform_id}",
            from_space=self.from_space,
            to_space=second.to_space,
            method=f"compose({self.method},{second.method})",
            matrix=product,
            frame_of_reference_uid=self.frame_of_reference_uid,
            series_instance_uid=self.series_instance_uid,
            direction_cosines=cosines,
            translation=(product[0][3], product[1][3], product[2][3]),
            handedness=self.handedness,
            units=self.units,
            version=max(self.version, second.version),
        )


def _validate_direction_cosines(cosines: Cosines3, problems: list[str]) -> None:
    """Direction cosines (Part A §1.2) must be an orthonormal 3x3 matrix."""
    shape_ok = len(cosines) == 3 and all(len(row) == 3 for row in cosines)
    if not shape_ok:
        problems.append("direction_cosines must be a 3x3 matrix of direction cosines")
        return
    finite = all(math.isfinite(c) for row in cosines for c in row)
    if not finite:
        problems.append("direction_cosines entries must be finite")
        return
    for i, row in enumerate(cosines):
        norm = math.sqrt(sum(c * c for c in row))
        if abs(norm - 1.0) > _DESCRIPTOR_TOLERANCE:
            problems.append(
                f"direction_cosines row {i} is not a unit direction cosine (|{norm:.9f}|)"
            )
    for i, j in ((0, 1), (0, 2), (1, 2)):
        dot = sum(a * b for a, b in zip(cosines[i], cosines[j]))
        if abs(dot) > _DESCRIPTOR_TOLERANCE:
            problems.append(
                f"direction_cosines rows {i} and {j} are not orthogonal (dot={dot:.9f})"
            )


def _validate_translation(translation: Vec3, problems: list[str]) -> None:
    """Translation vector (Part A §1.2) must be a finite 3-vector."""
    if len(translation) != 3 or not all(math.isfinite(c) for c in translation):
        problems.append("translation must be a finite 3-vector")


# ---------------------------------------------------------------------------
# Round-trip test hooks (Master Plan §7 mandate)
# ---------------------------------------------------------------------------


def round_trip_max_error(
    forward: CoordinateTransform,
    backward: CoordinateTransform,
    samples: tuple[Vec3, ...] = ((0.0, 0.0, 0.0), (10.5, -20.25, 33.125)),
) -> float:
    """Maximum Euclidean error of ``backward(forward(p))`` over *samples*.

    Spaces must form a closed loop; this hook backs the automated §7 tests.
    """
    if forward.to_space != backward.from_space or backward.to_space != forward.from_space:
        raise CanonicalValidationError(
            "round-trip requires backward to close the loop opened by forward"
        )
    worst = 0.0
    for sample in samples:
        p = WorldCoordinate(*sample)
        mid = apply_matrix(forward.matrix, p)
        back = apply_matrix(backward.matrix, WorldCoordinate(*mid))
        err = math.dist(sample, back)
        worst = max(worst, err)
    return worst


def assert_round_trip(
    forward: CoordinateTransform,
    backward: CoordinateTransform,
    samples: tuple[Vec3, ...] = ((0.0, 0.0, 0.0), (10.5, -20.25, 33.125)),
    tolerance_mm: float = _ROUND_TRIP_TOLERANCE_MM,
) -> None:
    """Fail closed when the transform pair does not round-trip sub-millimetre.

    Callers should persist the measured error into the COORDINATE_TRANSFORM
    artifact's ``round_trip_error_mm``.
    """
    error = round_trip_max_error(forward, backward, samples)
    if error > tolerance_mm:
        raise CanonicalValidationError(
            f"round-trip error {error:.9f} mm exceeds tolerance {tolerance_mm} mm"
        )


__all__ = [
    "Cosines3",
    "FrameSpec",
    "Matrix4",
    "MeshCoordinate",
    "CenterlineCoordinate",
    "CoordinateTransform",
    "VoxelCoordinate",
    "Vec3",
    "WorldCoordinate",
    "apply_matrix",
    "assert_round_trip",
    "identity_matrix",
    "invert_matrix",
    "matmul",
    "round_trip_max_error",
    "transform_for_frame",
]
