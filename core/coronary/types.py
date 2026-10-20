"""Closed vocabularies and parameter records of the coronary centerline slice.

CORONARY_CENTERLINE_001 (execution notes: the CMR-independent coronary track):
one seeded deterministic centerline path over the canonical HU volume.  This
module owns ONLY the frozen parameter record and the typed refusal vocabulary;
the algorithms live in :mod:`core.coronary.vesselness` and
:mod:`core.coronary.centerline`.

Rule C boundary: nothing in this package computes a measurement.  There is no
diameter, area, stenosis, CAD-RADS or FFR value anywhere - the slice produces
STRUCTURE (a vesselness response, a binary derivation mask, one ordered
polyline) and nothing else.

Rule D boundary: no LLM and no model participates; the vesselness/thinning
arithmetic is pure deterministic NumPy (no RNG anywhere).
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field
from typing import Final

from core.digest import canonical_digest
from core.errors import CanonicalValidationError

#: Payload schema of the CENTERLINE artifact produced by this slice.
CENTERLINE_SCHEMA: Final[str] = "ccta.payload.coronary_centerline.v1"
#: Claim binding (the accepted visualization claim; no new claim is invented).
CORONARY_CLAIM_ID: Final[str] = "CLM-VIZ-001"
CORONARY_CLAIM_VERSION: Final[str] = "v0-draft"
#: Deterministic algorithm identity recorded in the producer block.
CORONARY_ALGORITHM: Final[str] = "coronary-centerline-numpy"
CORONARY_ALGORITHM_VERSION: Final[str] = "frangi-edt-ridge-centered-spine-v1"
#: Voxel ordering of the canonical volume array: (k, j, i) == (slice, row, col).
VOLUME_ORDERING: Final[str] = "(slice,row,column)=(k,j,i)"
#: Codec of the retained derivation-mask evidence blob (one byte per voxel;
#: the mask bytes ride a content-addressed blob, NEVER a SEGMENTATION artifact).
MASK_EVIDENCE_CODEC: Final[str] = "bool-u8"

#: Frozen arithmetic/derivation rule identifiers (recorded in parameters_ref).
VESSELNESS_RULE: Final[str] = "FRANGI_HESSIAN_SIGMA2_MAX_V1"
SMOOTHING_RULE: Final[str] = "SEPARABLE_GAUSSIAN_EDGE_REPLICATE_V1"
HESSIAN_RULE: Final[str] = "CENTRAL_DIFFERENCE_SECOND_ORDER_V1"
THINNING_RULE: Final[str] = "EXACT_EDT_RIDGE_CENTERED_GEODESIC_SPINE_V1"
TRACE_RULE: Final[str] = "GEODESIC_DIAMETER_RIDGE_WEIGHTED_PATH_DIRECTION_HINT_V1"
POLARITY_BRIGHT: Final[str] = "BRIGHT"


class CoronaryRefusalCode(enum.StrEnum):
    """Closed typed-refusal vocabulary of the coronary derivation.

    ``NOT_ANALYZABLE`` is the abstention the Master Plan demands instead of a
    fabricated path.  Non-HU scalar domains are NOT re-implemented here: that
    safety is inherited from the existing F18 viewer gate at frame
    construction (IMAGING_CORE_002_F18_SEMANTIC_GATE), so this vocabulary
    mints no domain refusal.
    """

    NOT_ANALYZABLE = "NOT_ANALYZABLE"
    SEED_OUT_OF_BOUNDS = "SEED_OUT_OF_BOUNDS"
    SEED_DIRECTION_INVALID = "SEED_DIRECTION_INVALID"
    VOLUME_INVALID = "VOLUME_INVALID"
    PARAMETERS_INVALID = "PARAMETERS_INVALID"
    NONFINITE_RESPONSE = "NONFINITE_RESPONSE"


class CoronaryValidationError(CanonicalValidationError):
    """Base class of the coronary module's fail-closed refusals."""

    def __init__(self, code: CoronaryRefusalCode, detail: str = "") -> None:
        self.code = CoronaryRefusalCode(code)
        self.detail = detail
        super().__init__([self.code.value] if not detail else [self.code.value, detail])


def _refuse(code: CoronaryRefusalCode, detail: str = "") -> CoronaryValidationError:
    return CoronaryValidationError(code, detail)


@dataclass(frozen=True)
class VesselnessParameters:
    """Frozen, digest-bound parameters of the vesselness + mask derivation.

    ``scales_mm`` are the Gaussian scales in millimetres (physical length, so
    the same parameters mean the same thing at any voxel spacing).  ``threshold``
    is the closed mask threshold applied to the vesselness response.  The
    Frangi constants ``alpha``/``beta``/``gamma`` follow the classic
    formulation; ``boundary_rule``/``hessian_rule``/``thinning_rule`` pin the
    discrete arithmetic so the response is bit-identical for the same
    volume + parameters (no RNG).

    The digest of :meth:`to_document` is what every artifact of the slice
    records in its ``parameters_ref`` - the mask bytes and the vesselness
    response bytes are carried as separate evidence digests inside that same
    document, never as a SEGMENTATION artifact (producer matrix).
    """

    scales_mm: tuple[float, ...] = (1.0, 2.0, 3.0)
    threshold: float = 0.35
    alpha: float = 0.5
    beta: float = 0.5
    gamma: float = 0.5
    polarity: str = POLARITY_BRIGHT
    thinning_connectivity: int = 26
    boundary_rule: str = SMOOTHING_RULE
    hessian_rule: str = HESSIAN_RULE
    vesselness_rule: str = VESSELNESS_RULE
    thinning_rule: str = THINNING_RULE
    trace_rule: str = TRACE_RULE

    def __post_init__(self) -> None:
        problems: list[str] = []
        if not self.scales_mm:
            problems.append("scales_mm must be non-empty")
        for scale in self.scales_mm:
            if (
                isinstance(scale, bool)
                or not isinstance(scale, (int, float))
                or not math.isfinite(float(scale))
                or float(scale) <= 0.0
            ):
                problems.append(f"scales_mm entries must be positive and finite: {scale!r}")
        if any(float(a) >= float(b) for a, b in zip(self.scales_mm, self.scales_mm[1:])):
            problems.append("scales_mm must be strictly ascending")
        for label, value, low, high in (
            ("threshold", self.threshold, 0.0, float("inf")),
            ("alpha", self.alpha, 0.0, float("inf")),
            ("beta", self.beta, 0.0, float("inf")),
            ("gamma", self.gamma, 0.0, float("inf")),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not (low < float(value) <= high)
            ):
                problems.append(f"{label} must be a finite positive number")
        if self.polarity != POLARITY_BRIGHT:
            problems.append(f"polarity must be {POLARITY_BRIGHT!r}")
        if self.thinning_connectivity != 26:
            problems.append("thinning_connectivity is fixed to 26")
        if problems:
            raise _refuse(CoronaryRefusalCode.PARAMETERS_INVALID, "; ".join(problems))

    def to_document(self) -> dict[str, object]:
        return {
            "schema_version": "ccta.coronary.vesselness_parameters.v1",
            "scales_mm": [float(scale) for scale in self.scales_mm],
            "threshold": float(self.threshold),
            "alpha": float(self.alpha),
            "beta": float(self.beta),
            "gamma": float(self.gamma),
            "polarity": self.polarity,
            "thinning_connectivity": int(self.thinning_connectivity),
            "boundary_rule": self.boundary_rule,
            "hessian_rule": self.hessian_rule,
            "vesselness_rule": self.vesselness_rule,
            "thinning_rule": self.thinning_rule,
            "trace_rule": self.trace_rule,
        }

    @property
    def sha256(self) -> str:
        return canonical_digest(self.to_document())


@dataclass(frozen=True)
class Seed:
    """The one human seed of the slice: a voxel index plus a direction hint.

    ``direction`` is a finite non-zero direction in the canonical WORLD frame
    (DICOM patient LPS, millimetres); the derivation converts it into voxel
    index space through the persisted WORLD->VOXEL linear part.  Both arrive
    as typed API parameters - never free text.
    """

    i: int
    j: int
    k: int
    direction: tuple[float, float, float]

    def __post_init__(self) -> None:
        for name, value in (("i", self.i), ("j", self.j), ("k", self.k)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise _refuse(
                    CoronaryRefusalCode.VOLUME_INVALID,
                    f"seed voxel {name} must be a non-negative integer",
                )
        direction = self.direction
        if (
            not isinstance(direction, tuple)
            or len(direction) != 3
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in direction
            )
        ):
            raise _refuse(
                CoronaryRefusalCode.SEED_DIRECTION_INVALID,
                "seed direction must be a finite 3-vector",
            )
        norm = math.sqrt(sum(float(value) ** 2 for value in direction))
        if norm <= 0.0:
            raise _refuse(
                CoronaryRefusalCode.SEED_DIRECTION_INVALID,
                "seed direction must be non-zero",
            )

    @property
    def voxel(self) -> tuple[int, int, int]:
        return (self.i, self.j, self.k)

    def unit_direction(self) -> tuple[float, float, float]:
        norm = math.sqrt(sum(float(value) ** 2 for value in self.direction))
        return (
            float(self.direction[0]) / norm,
            float(self.direction[1]) / norm,
            float(self.direction[2]) / norm,
        )

    def to_document(self) -> dict[str, object]:
        return {
            "seed_voxel": [self.i, self.j, self.k],
            "seed_direction_world": [float(value) for value in self.unit_direction()],
        }


@dataclass(frozen=True)
class CoronaryDiagnostics:
    """Deterministic, non-measurement diagnostics of one derivation.

    These are counts and digests over the derivation itself (Rule C: no
    diameter, area, stenosis or any physiological quantity).
    """

    response_sha256: str
    mask_sha256: str
    mask_voxel_count: int
    skeleton_voxel_count: int
    voxel_count: int
    extra: tuple[tuple[str, object], ...] = field(default=())

    def to_document(self) -> dict[str, object]:
        document: dict[str, object] = {
            "vesselness_response_sha256": self.response_sha256,
            "vessel_mask_sha256": self.mask_sha256,
            "vessel_mask_voxel_count": int(self.mask_voxel_count),
            "skeleton_voxel_count": int(self.skeleton_voxel_count),
            "voxel_count": int(self.voxel_count),
        }
        for key, value in self.extra:
            document[key] = value
        return document


__all__ = [
    "CENTERLINE_SCHEMA",
    "CORONARY_ALGORITHM",
    "CORONARY_ALGORITHM_VERSION",
    "CORONARY_CLAIM_ID",
    "CORONARY_CLAIM_VERSION",
    "HESSIAN_RULE",
    "POLARITY_BRIGHT",
    "SMOOTHING_RULE",
    "THINNING_RULE",
    "TRACE_RULE",
    "VESSELNESS_RULE",
    "VOLUME_ORDERING",
    "CoronaryDiagnostics",
    "CoronaryRefusalCode",
    "CoronaryValidationError",
    "Seed",
    "VesselnessParameters",
]
