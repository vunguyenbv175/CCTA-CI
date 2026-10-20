"""Pure deterministic QC family algorithms.

This module consumes only the reduced DICOM DTO, a canonical read-only array,
the persisted ``Volume``/``FrameSpec`` surface, and (for two families) the
closed registered-pair DTO.  It does not read DICOM files and does not know
about applications or model outputs.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np

from core.artifacts import GeometryBase
from core.coordinates import CoordinateTransform, FrameSpec, invert_matrix
from core.dicom.instance import (
    ADMITTED_TRANSFER_SYNTAXES,
    DicomInstance,
    admitted_bit_depth_tuple,
    stored_padding_bits_are_zero,
)
from core.digest import canonical_digest, is_sha256_hex
from core.enums import ArtifactStatus, VolumeKind
from core.volume import Volume

from .constants import (
    FAMILY_ALGORITHM_IDS,
    FAMILY_METRIC_NAMES,
    FIRST_NINE_FLAGS,
    QCAbstentionCode,
    QCOutcome,
    classify_high_bad,
    classify_incomplete_anatomy,
    classify_low_bad,
    classify_stair_step,
    family_parameter_digest,
)
from .types import (
    CoverageEnvelope,
    FamilyResult,
    QCInput,
    QCInputError,
    QCValidationError,
    RegisteredPairInput,
)

_EPS = 1e-12
_GEOMETRY_EPS = 1e-12
_PAIR_FLAGS = frozenset({"CARDIAC_MOTION", "MISREGISTRATION"})
_SIX_NEIGHBOURS: tuple[tuple[int, int, int], ...] = (
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
)


@dataclass(frozen=True, slots=True)
class _Context:
    source: QCInput
    volume: Volume
    array: Any
    frame: FrameSpec
    instances: tuple[DicomInstance, ...]
    source_manifest_sha256: str
    instance_digests: tuple[tuple[str, str], ...]
    scalar_domain: str
    intensity_unit: str
    source_low: float
    source_high: float
    transformed_low: float
    transformed_high: float
    geometry_base: GeometryBase
    cache: dict[object, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class _PairSample:
    k: int
    j: int
    i: int
    first: float
    second: float
    second_index: tuple[float, float, float]


def _fail(code: QCAbstentionCode | str, detail: str) -> QCInputError:
    return QCInputError(code, detail)


def _validate_frame(frame: FrameSpec, expected_shape: tuple[int, int, int]) -> None:
    if not isinstance(frame, FrameSpec):
        raise _fail(QCAbstentionCode.QC_GEOMETRY_INVALID, "FrameSpec required")
    if tuple(int(value) for value in frame.dimensions) != expected_shape:
        raise _fail(QCAbstentionCode.QC_GEOMETRY_INVALID, "FrameSpec dimensions mismatch")
    if (
        getattr(frame.coordinate_system, "value", frame.coordinate_system) != "DICOM_PATIENT"
        or getattr(frame.handedness, "value", frame.handedness) != "LPS"
        or getattr(frame.units, "value", frame.units) != "MM"
    ):
        raise _fail(
            QCAbstentionCode.QC_GEOMETRY_INVALID,
            "FrameSpec must use DICOM patient LPS millimetres",
        )
    if len(frame.image_orientation_patient) != 6:
        raise _fail(QCAbstentionCode.QC_GEOMETRY_INVALID, "orientation must have six values")
    values = (
        *frame.image_position_patient,
        *frame.image_orientation_patient,
        *frame.spacing_mm,
    )
    if not all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in values):
        raise _fail(QCAbstentionCode.QC_GEOMETRY_INVALID, "geometry contains nonfinite values")
    row = cast(
        tuple[float, float, float], tuple(float(v) for v in frame.image_orientation_patient[:3])
    )
    col = cast(
        tuple[float, float, float], tuple(float(v) for v in frame.image_orientation_patient[3:])
    )
    row_norm = math.sqrt(sum(value * value for value in row))
    col_norm = math.sqrt(sum(value * value for value in col))
    dot = sum(a * b for a, b in zip(row, col))
    if (
        not math.isclose(row_norm, 1.0, rel_tol=1e-6, abs_tol=1e-6)
        or not math.isclose(col_norm, 1.0, rel_tol=1e-6, abs_tol=1e-6)
        or abs(dot) > 1e-6
        or any(float(value) <= 0.0 for value in frame.spacing_mm)
    ):
        raise _fail(QCAbstentionCode.QC_GEOMETRY_INVALID, "orientation or spacing invalid")
    if (
        isinstance(frame.rescale_applied, bool) is False
        or not math.isfinite(float(frame.rescale_slope))
        or not math.isfinite(float(frame.rescale_intercept))
    ):
        raise _fail(
            QCAbstentionCode.QC_SCALAR_DOMAIN_INVALID, "FrameSpec scalar transform is invalid"
        )
    normal = _cross(row, col)
    normal_norm = math.sqrt(sum(value * value for value in normal))
    if not math.isfinite(normal_norm) or normal_norm <= _EPS:
        raise _fail(QCAbstentionCode.QC_GEOMETRY_INVALID, "slice normal is singular")
    if not frame.frame_of_reference_uid:
        raise _fail(QCAbstentionCode.QC_GEOMETRY_INVALID, "FrameOfReferenceUID is required")
    try:
        matrix = frame.voxel_to_world_matrix()
        inverse = invert_matrix(matrix)
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        raise _fail(QCAbstentionCode.QC_GEOMETRY_INVALID, "FrameSpec transform is invalid") from exc
    if not all(
        math.isfinite(float(item)) for row_values in (*matrix, *inverse) for item in row_values
    ):
        raise _fail(QCAbstentionCode.QC_GEOMETRY_INVALID, "FrameSpec transform is nonfinite")


def _mapping_items(
    value: Mapping[str, str] | tuple[tuple[str, str], ...],
) -> tuple[tuple[str, str], ...]:
    if isinstance(value, Mapping):
        items = tuple(value.items())
    elif isinstance(value, tuple):
        items = value
    else:
        raise _fail(
            QCAbstentionCode.QC_INPUT_SURFACE_INVALID,
            "instance digests must be a mapping or tuple of pairs",
        )
    if any(
        not isinstance(key, str) or not isinstance(item, str)
        for key, item in items
    ):
        raise _fail(
            QCAbstentionCode.QC_SOURCE_IDENTITY_INVALID,
            "instance digest refs and values must be strings",
        )
    return cast(tuple[tuple[str, str], ...], items)


def _validate_transform(
    source: QCInput,
    frame: FrameSpec,
    instances: tuple[DicomInstance, ...],
) -> None:
    """Validate an optional persisted voxel-to-world transform binding.

    ``FrameSpec`` remains the geometric authority.  A supplied transform is
    admitted only when it is the versioned transform derived from that exact
    frame and source series; an arbitrary affine must never become a hidden
    second geometry authority.
    """

    transform = source.transform
    # A FrameSpec by itself is not a persisted transform artifact.  The
    # deterministic engine therefore requires the complete versioned
    # transform triple (object, content ref, and provenance ref) at its
    # boundary; a missing member must not turn into a synthetic fallback.
    if transform is None:
        raise _fail(
            QCAbstentionCode.QC_GEOMETRY_INVALID,
            "a versioned voxel-to-world transform is required",
        )
    if not isinstance(transform, CoordinateTransform):
        raise _fail(
            QCAbstentionCode.QC_GEOMETRY_INVALID,
            "versioned CoordinateTransform is required",
        )
    if not isinstance(source.transform_ref, str) or not source.transform_ref:
        raise _fail(
            QCAbstentionCode.QC_INPUT_SURFACE_INVALID,
            "transform_ref must be a nonempty persisted reference",
        )
    if not isinstance(source.transform_sha256, str) or not is_sha256_hex(
        source.transform_sha256
    ):
        raise _fail(
            QCAbstentionCode.QC_INPUT_SURFACE_INVALID,
            "transform_sha256 must be lowercase SHA-256",
        )
    if not isinstance(source.transform_provenance_ref, str) or not source.transform_provenance_ref:
        raise _fail(
            QCAbstentionCode.QC_SOURCE_IDENTITY_INVALID,
            "transform provenance reference is required",
        )
    calculated_digest = canonical_digest(transform)
    if source.transform_sha256 != calculated_digest:
        raise _fail(
            QCAbstentionCode.QC_INPUT_SURFACE_INVALID,
            "versioned transform digest disagrees with supplied transform",
        )
    if (
        transform.from_space.value != "voxel"
        or transform.to_space.value != "world"
        or not isinstance(transform.version, int)
        or isinstance(transform.version, bool)
        or transform.version < 1
        or transform.frame_of_reference_uid != frame.frame_of_reference_uid
        or transform.units.value != "MM"
        or transform.handedness.value != "LPS"
    ):
        raise _fail(
            QCAbstentionCode.QC_GEOMETRY_INVALID,
            "versioned transform descriptors do not bind the FrameSpec",
        )
    if transform.matrix != frame.voxel_to_world_matrix():
        raise _fail(
            QCAbstentionCode.QC_GEOMETRY_INVALID,
            "versioned transform matrix differs from FrameSpec",
        )
    if tuple(transform.translation) != tuple(frame.image_position_patient):
        raise _fail(
            QCAbstentionCode.QC_GEOMETRY_INVALID,
            "versioned transform translation differs from FrameSpec",
        )
    if instances and any(
        transform.series_instance_uid != instance.series_instance_uid for instance in instances
    ):
        raise _fail(
            QCAbstentionCode.QC_SOURCE_IDENTITY_INVALID,
            "versioned transform series does not bind ordered source instances",
        )


def _endpoint_values(
    source: QCInput,
    instances: tuple[DicomInstance, ...],
    array: Any,
    frame: FrameSpec,
) -> tuple[float, float, float, float]:
    if source.source_low is not None or source.source_high is not None:
        if source.source_low is None or source.source_high is None:
            raise _fail(
                QCAbstentionCode.QC_SCALAR_DOMAIN_INVALID, "both source endpoints are required"
            )
        low = float(source.source_low)
        high = float(source.source_high)
    elif instances:
        first = instances[0]
        bits = int(first.bits_stored)
        if bits < 1 or bits > 32 or int(first.pixel_representation) not in {0, 1}:
            raise _fail(
                QCAbstentionCode.QC_SCALAR_DOMAIN_INVALID, "representable endpoint metadata invalid"
            )
        if int(first.pixel_representation) == 0:
            low, high = 0.0, float((1 << bits) - 1)
        else:
            low, high = float(-(1 << (bits - 1))), float((1 << (bits - 1)) - 1)
    else:
        kind = getattr(array.dtype, "kind", "")
        if kind == "u":
            low, high = 0.0, float(np.iinfo(array.dtype).max)
        elif kind == "i":
            info = np.iinfo(array.dtype)
            low, high = float(info.min), float(info.max)
        else:
            low, high = float(np.nanmin(array)), float(np.nanmax(array))
    if not (math.isfinite(low) and math.isfinite(high) and low < high):
        raise _fail(
            QCAbstentionCode.QC_SCALAR_DOMAIN_INVALID, "representable endpoint range invalid"
        )
    slope = float(frame.rescale_slope)
    intercept = float(frame.rescale_intercept)
    transformed = (low * slope + intercept, high * slope + intercept)
    transformed_low, transformed_high = min(transformed), max(transformed)
    if not (math.isfinite(transformed_low) and math.isfinite(transformed_high)):
        raise _fail(QCAbstentionCode.QC_SCALAR_DOMAIN_INVALID, "rescaled endpoints nonfinite")
    return low, high, transformed_low, transformed_high


def validate_qc_input(source: QCInput) -> _Context:
    """Validate the complete common QC input surface before metric work."""

    if not isinstance(source, QCInput):
        raise _fail(QCAbstentionCode.QC_INPUT_SURFACE_INVALID, "QCInput is required")
    if not isinstance(source.volume, Volume):
        raise _fail(QCAbstentionCode.QC_INPUT_SURFACE_INVALID, "canonical Volume is required")
    volume = source.volume
    if volume.status != ArtifactStatus.ACTIVE:
        raise _fail(QCAbstentionCode.QC_SOURCE_IDENTITY_INVALID, "volume is not ACTIVE")
    if volume.kind != VolumeKind.CANONICAL:
        raise _fail(
            QCAbstentionCode.QC_INPUT_SURFACE_INVALID,
            "QC accepts only the canonical source Volume",
        )
    array = source.array
    if not isinstance(array, np.ndarray):
        raise _fail(QCAbstentionCode.QC_INPUT_SURFACE_INVALID, "canonical NumPy array is required")
    if array.ndim != 3 or not array.flags.c_contiguous:
        raise _fail(QCAbstentionCode.QC_INPUT_SURFACE_INVALID, "array must be C-contiguous 3-D")
    if bool(array.flags.writeable):
        raise _fail(QCAbstentionCode.QC_INPUT_SURFACE_INVALID, "canonical array must be read-only")
    if array.dtype.kind not in {"i", "u", "f"}:
        raise _fail(QCAbstentionCode.QC_INPUT_SURFACE_INVALID, "array scalar dtype is unsupported")
    shape = cast(tuple[int, int, int], tuple(int(value) for value in array.shape))
    if any(value <= 0 for value in shape):
        raise _fail(QCAbstentionCode.QC_INPUT_SURFACE_INVALID, "array dimensions must be positive")
    frame = volume.coordinate_frame
    _validate_frame(frame, shape)
    if not bool(np.all(np.isfinite(array))):
        raise _fail(QCAbstentionCode.QC_SCALAR_DOMAIN_INVALID, "array contains nonfinite values")
    if source.pixel_padding_present or source.modality_lut_present:
        raise _fail(
            QCAbstentionCode.QC_SCALAR_DOMAIN_INVALID, "padding and LUT inputs are unsupported"
        )
    if not isinstance(source.pixel_padding_present, bool) or not isinstance(
        source.modality_lut_present, bool
    ):
        raise _fail(
            QCAbstentionCode.QC_SCALAR_DOMAIN_INVALID,
            "padding and LUT presence flags must be boolean",
        )
    if not is_sha256_hex(source.source_manifest_sha256):
        raise _fail(
            QCAbstentionCode.QC_SOURCE_IDENTITY_INVALID, "source manifest must be lowercase SHA-256"
        )
    if not volume.volume_id or not volume.study_ref or not volume.source_series_refs:
        raise _fail(QCAbstentionCode.QC_SOURCE_IDENTITY_INVALID, "volume identity is incomplete")
    if not source.instances:
        raise _fail(
            QCAbstentionCode.QC_SOURCE_IDENTITY_INVALID,
            "normal QC requires ordered DicomInstance metadata",
        )
    actual_blob_sha = hashlib.sha256(array.tobytes(order="C")).hexdigest()
    if actual_blob_sha != volume.blob_ref.sha256:
        raise _fail(
            QCAbstentionCode.QC_INPUT_SURFACE_INVALID, "array bytes disagree with Volume blob"
        )
    if int(array.nbytes) != int(volume.blob_ref.byte_size):
        raise _fail(
            QCAbstentionCode.QC_INPUT_SURFACE_INVALID,
            "array byte size disagrees with Volume blob",
        )
    domain = source.scalar_domain
    unit = source.intensity_unit if source.intensity_unit is not None else domain
    if (
        not isinstance(domain, str)
        or not isinstance(unit, str)
        or domain not in {"STORED", "MODALITY_TRANSFORMED", "HU"}
        or unit != domain
    ):
        raise _fail(QCAbstentionCode.QC_SCALAR_DOMAIN_INVALID, "scalar domain and unit must agree")
    try:
        instances = tuple(source.instances)
    except (TypeError, ValueError) as exc:
        raise _fail(QCAbstentionCode.QC_INPUT_SURFACE_INVALID, "instances must be a tuple") from exc
    if instances and len(instances) != shape[0]:
        raise _fail(
            QCAbstentionCode.QC_SOURCE_IDENTITY_INVALID,
            "ordered SOP count must equal the canonical slice count",
        )
    seen_sop: set[str] = set()
    first_instance = instances[0]
    if not isinstance(first_instance, DicomInstance):
        raise _fail(
            QCAbstentionCode.QC_INPUT_SURFACE_INVALID,
            "only DicomInstance metadata is admitted",
        )
    expected_scalar_metadata = (
        float(first_instance.rescale_slope),
        float(first_instance.rescale_intercept),
        first_instance.rescale_type,
    )
    for slice_index, instance in enumerate(instances):
        if not isinstance(instance, DicomInstance):
            raise _fail(
                QCAbstentionCode.QC_INPUT_SURFACE_INVALID, "only DicomInstance metadata is admitted"
            )
        if instance.study_instance_uid != instances[0].study_instance_uid:
            raise _fail(QCAbstentionCode.QC_SOURCE_IDENTITY_INVALID, "mixed StudyInstanceUID")
        if instance.series_instance_uid != first_instance.series_instance_uid:
            raise _fail(QCAbstentionCode.QC_SOURCE_IDENTITY_INVALID, "mixed SeriesInstanceUID")
        if instance.frame_of_reference_uid != frame.frame_of_reference_uid:
            raise _fail(QCAbstentionCode.QC_GEOMETRY_INVALID, "FrameOfReferenceUID mismatch")
        if int(instance.rows) != shape[1] or int(instance.columns) != shape[2]:
            raise _fail(
                QCAbstentionCode.QC_GEOMETRY_INVALID, "instance dimensions disagree with Volume"
            )
        if instance.sop_instance_uid in seen_sop:
            raise _fail(QCAbstentionCode.QC_SOURCE_IDENTITY_INVALID, "duplicate SOP instance")
        seen_sop.add(instance.sop_instance_uid)
        if instance.transfer_syntax_uid not in ADMITTED_TRANSFER_SYNTAXES:
            raise _fail(
                QCAbstentionCode.QC_INPUT_SURFACE_INVALID,
                "instance transfer syntax is not in the admitted set",
            )
        if len(instance.pixel_data) != int(instance.rows) * int(instance.columns) * 2:
            raise _fail(
                QCAbstentionCode.QC_INPUT_SURFACE_INVALID, "instance pixel length is invalid"
            )
        if instance.pixel_data_byte_length != len(instance.pixel_data):
            raise _fail(
                QCAbstentionCode.QC_INPUT_SURFACE_INVALID,
                "instance pixel byte-length metadata is invalid",
            )
        if instance.number_of_frames != 1 or instance.samples_per_pixel != 1:
            raise _fail(
                QCAbstentionCode.QC_INPUT_SURFACE_INVALID,
                "only single-frame monochrome source instances are admitted",
            )
        if not admitted_bit_depth_tuple(
            instance.bits_allocated,
            instance.bits_stored,
            instance.high_bit,
            instance.pixel_representation,
        ):
            raise _fail(
                QCAbstentionCode.QC_INPUT_SURFACE_INVALID,
                "source scalar representation is outside the admitted surface",
            )
        if not stored_padding_bits_are_zero(
            instance.pixel_data,
            bits_stored=instance.bits_stored,
            high_bit=instance.high_bit,
        ):
            raise _fail(
                QCAbstentionCode.QC_INPUT_SURFACE_INVALID,
                "source 12-bit stored words carry nonconformant padding bits",
            )
        if instance.pixel_representation not in {0, 1}:
            raise _fail(
                QCAbstentionCode.QC_SCALAR_DOMAIN_INVALID,
                "source pixel representation is invalid",
            )
        if (
            float(instance.rescale_slope),
            float(instance.rescale_intercept),
            instance.rescale_type,
        ) != expected_scalar_metadata:
            raise _fail(
                QCAbstentionCode.QC_SCALAR_DOMAIN_INVALID,
                "source scalar metadata is not uniform across ordered instances",
            )
        expected_position = world_center(frame, slice_index, 0, 0)
        if tuple(float(value) for value in instance.image_position_patient) != expected_position:
            raise _fail(
                QCAbstentionCode.QC_GEOMETRY_INVALID,
                "ordered SOP position does not bind the FrameSpec slice index",
            )
        if instance.image_orientation_patient != frame.image_orientation_patient:
            raise _fail(
                QCAbstentionCode.QC_GEOMETRY_INVALID,
                "source orientation is not uniform with FrameSpec",
            )
        if instance.pixel_spacing != (
            float(frame.spacing_mm[1]),
            float(frame.spacing_mm[0]),
        ):
            raise _fail(
                QCAbstentionCode.QC_GEOMETRY_INVALID,
                "source in-plane spacing is not uniform with FrameSpec",
            )
    provided_digests = _mapping_items(source.instance_digests)
    if len({key for key, _ in provided_digests}) != len(provided_digests):
        raise _fail(
            QCAbstentionCode.QC_SOURCE_IDENTITY_INVALID, "instance digest keys are duplicated"
        )
    digest_map: dict[str, str] = {}
    for key, provided_digest in provided_digests:
        logical_key = key.removeprefix("sop:")
        if logical_key in digest_map:
            raise _fail(
                QCAbstentionCode.QC_SOURCE_IDENTITY_INVALID,
                "instance digest refs contain duplicate SOP aliases",
            )
        digest_map[logical_key] = provided_digest
    accepted_digest_keys = {instance.sop_instance_uid for instance in instances} | {
        f"sop:{instance.sop_instance_uid}" for instance in instances
    }
    if any(key not in accepted_digest_keys for key, _ in provided_digests):
        raise _fail(
            QCAbstentionCode.QC_SOURCE_IDENTITY_INVALID, "instance digest has an unknown SOP ref"
        )
    normalized_digests: list[tuple[str, str]] = []
    for instance in instances:
        instance_digest = digest_map.get(instance.sop_instance_uid)
        if instance_digest is None:
            raise _fail(
                QCAbstentionCode.QC_SOURCE_IDENTITY_INVALID,
                "every ordered SOP ref requires a supplied digest",
            )
        if not is_sha256_hex(instance_digest):
            raise _fail(
                QCAbstentionCode.QC_SOURCE_IDENTITY_INVALID,
                "instance digest is not lowercase SHA-256",
            )
        if instance_digest != hashlib.sha256(instance.pixel_data).hexdigest():
            raise _fail(
                QCAbstentionCode.QC_SOURCE_IDENTITY_INVALID,
                "supplied instance digest does not bind DicomInstance.pixel_data",
            )
        normalized_digests.append((instance.sop_instance_uid, instance_digest))
    if not first_instance.study_instance_uid:
        raise _fail(QCAbstentionCode.QC_SOURCE_IDENTITY_INVALID, "source study identity is invalid")
    if tuple(float(value) for value in first_instance.image_position_patient) != tuple(
        float(value) for value in frame.image_position_patient
    ):
        raise _fail(
            QCAbstentionCode.QC_GEOMETRY_INVALID, "FrameSpec origin differs from first SOP position"
        )
    if tuple(float(value) for value in first_instance.image_orientation_patient) != tuple(
        float(value) for value in frame.image_orientation_patient
    ):
        raise _fail(
            QCAbstentionCode.QC_GEOMETRY_INVALID, "FrameSpec orientation differs from SOP metadata"
        )
    if tuple(float(value) for value in first_instance.pixel_spacing) != (
        float(frame.spacing_mm[1]),
        float(frame.spacing_mm[0]),
    ):
        raise _fail(
            QCAbstentionCode.QC_GEOMETRY_INVALID,
            "FrameSpec in-plane spacing differs from SOP metadata",
        )
    if source.scalar_domain == "STORED" and frame.rescale_applied:
        raise _fail(
            QCAbstentionCode.QC_SCALAR_DOMAIN_INVALID, "STORED input cannot have an applied rescale"
        )
    if source.scalar_domain != "STORED" and not frame.rescale_applied:
        raise _fail(
            QCAbstentionCode.QC_SCALAR_DOMAIN_INVALID,
            "transformed input must record an applied rescale",
        )
    if source.scalar_domain == "STORED" and (
        frame.rescale_slope != 1.0 or frame.rescale_intercept != 0.0
    ):
        raise _fail(
            QCAbstentionCode.QC_SCALAR_DOMAIN_INVALID,
            "STORED input cannot carry a non-identity FrameSpec rescale",
        )
    if source.scalar_domain != "STORED" and (
        float(first_instance.rescale_slope) != float(frame.rescale_slope)
        or float(first_instance.rescale_intercept) != float(frame.rescale_intercept)
    ):
        raise _fail(
            QCAbstentionCode.QC_SCALAR_DOMAIN_INVALID,
            "transformed input rescale does not bind source metadata",
        )
    _validate_transform(source, frame, instances)
    if volume.transform_ref is not None and volume.transform_ref != source.transform_ref:
        raise _fail(
            QCAbstentionCode.QC_SOURCE_IDENTITY_INVALID,
            "QC transform ref does not bind the canonical Volume transform ref",
        )
    low, high, transformed_low, transformed_high = _endpoint_values(source, instances, array, frame)
    base = source.geometry_base or GeometryBase(
        geometry_id=f"volume_geometry:{volume.volume_id}", geometry_version=1
    )
    if (
        not isinstance(base, GeometryBase)
        or isinstance(base.geometry_version, bool)
        or not isinstance(base.geometry_version, int)
        or base.geometry_version < 1
    ):
        raise _fail(QCAbstentionCode.QC_GEOMETRY_INVALID, "geometry version must be >= 1")
    if base.geometry_id != f"volume_geometry:{volume.volume_id}":
        raise _fail(
            QCAbstentionCode.QC_GEOMETRY_INVALID,
            "single-volume geometry base is not bound to Volume",
        )
    return _Context(
        source=source,
        volume=volume,
        array=array,
        frame=frame,
        instances=instances,
        source_manifest_sha256=source.source_manifest_sha256,
        instance_digests=tuple(normalized_digests),
        scalar_domain=domain,
        intensity_unit=unit,
        source_low=low,
        source_high=high,
        transformed_low=transformed_low,
        transformed_high=transformed_high,
        geometry_base=base,
    )


def lower_quantile(values: Sequence[float] | Any, probability: float) -> float:
    """The closed lower order statistic ``S[floor(p*(n-1))]``."""

    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0, 1]")
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0 or not bool(np.all(np.isfinite(array))):
        raise ValueError("quantile requires a nonempty finite population")
    ordered = np.sort(array, kind="stable")
    index = math.floor(probability * (int(ordered.size) - 1))
    return float(ordered[index])


def median(values: Sequence[float] | Any) -> float:
    return lower_quantile(values, 0.50)


def mad(values: Sequence[float] | Any) -> float:
    centre = median(values)
    return median(np.abs(np.asarray(values, dtype=np.float64) - centre))


def robust_range(values: Sequence[float] | Any) -> float:
    result = lower_quantile(values, 0.95) - lower_quantile(values, 0.05)
    if not math.isfinite(result) or result <= 0.0:
        raise _fail(QCAbstentionCode.QC_DYNAMIC_RANGE_INVALID, "robust range is not positive")
    return float(result)


def _cross(
    a: tuple[float, float, float], b: tuple[float, float, float]
) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _superior_positive_normal(
    normal: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Orient a slice normal according to the frozen superior-positive rule."""

    if normal[2] < 0.0:
        return cast(tuple[float, float, float], tuple(-value for value in normal))
    if normal[2] > 0.0:
        return normal
    pivot = max(range(3), key=lambda axis: abs(normal[axis]))
    if normal[pivot] < 0.0:
        return cast(tuple[float, float, float], tuple(-value for value in normal))
    return normal


def frame_axes(
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
    col = cast(
        tuple[float, float, float],
        tuple(float(value) for value in frame.image_orientation_patient[3:]),
    )
    normal = _cross(row, col)
    norm = math.sqrt(sum(value * value for value in normal))
    normalized = cast(tuple[float, float, float], tuple(value / norm for value in normal))
    return row, col, normalized


def world_center(frame: FrameSpec, k: int, j: int, i: int) -> tuple[float, float, float]:
    matrix = frame.voxel_to_world_matrix()
    sx_i, sy_j, sz_k = float(i), float(j), float(k)
    # Keep the written arithmetic order explicit; these are binary64 values.
    return (
        (((matrix[0][3] + matrix[0][0] * sx_i) + matrix[0][1] * sy_j) + matrix[0][2] * sz_k),
        (((matrix[1][3] + matrix[1][0] * sx_i) + matrix[1][1] * sy_j) + matrix[1][2] * sz_k),
        (((matrix[2][3] + matrix[2][0] * sx_i) + matrix[2][1] * sy_j) + matrix[2][2] * sz_k),
    )


def _normalized_axes(shape: tuple[int, int, int]) -> tuple[Any, Any, Any]:
    k, j, i = shape
    u = (
        np.zeros(i, dtype=np.float64)
        if i == 1
        else 2.0 * np.arange(i, dtype=np.float64) / (i - 1) - 1.0
    )
    v = (
        np.zeros(j, dtype=np.float64)
        if j == 1
        else 2.0 * np.arange(j, dtype=np.float64) / (j - 1) - 1.0
    )
    w = (
        np.zeros(k, dtype=np.float64)
        if k == 1
        else 2.0 * np.arange(k, dtype=np.float64) / (k - 1) - 1.0
    )
    return u, v, w


def normalized_mask(
    shape: tuple[int, int, int],
    *,
    u_abs: float,
    v_abs: float,
    w_abs: float,
) -> Any:
    u, v, w = _normalized_axes(shape)
    return (
        (np.abs(w)[:, None, None] <= w_abs)
        & (np.abs(v)[None, :, None] <= v_abs)
        & (np.abs(u)[None, None, :] <= u_abs)
    )


def _central_mask(context: _Context, *, u_abs: float, v_abs: float, w_abs: float) -> Any:
    shape = cast(tuple[int, int, int], tuple(int(value) for value in context.array.shape))
    key = ("mask", shape, float(u_abs), float(v_abs), float(w_abs))
    cached = context.cache.get(key)
    if cached is None:
        cached = normalized_mask(shape, u_abs=u_abs, v_abs=v_abs, w_abs=w_abs)
        context.cache[key] = cached
    return cached


def _edge_gradient(slice_values: Any, spacing: tuple[float, float]) -> Any:
    rows, columns = (int(value) for value in slice_values.shape)
    sx, sy = float(spacing[0]), float(spacing[1])
    result = np.zeros((rows, columns), dtype=np.float64)
    values = np.asarray(slice_values, dtype=np.float64)
    # Associate each physical edge with both of its incident voxel centers.
    # A one-sided assignment would make a legal stored-axis reversal move an
    # edge to a different center and would make the threshold population
    # representation-dependent.  The symmetric max is still the prescribed
    # finite-difference magnitude, while being equivariant under reversal and
    # row/column transposition.
    if columns > 1:
        horizontal = np.abs((values[:, 1:] - values[:, :-1]) / sx)
        result[:, 1:] = np.maximum(result[:, 1:], horizontal)
        result[:, :-1] = np.maximum(result[:, :-1], horizontal)
    if rows > 1:
        vertical = np.abs((values[1:, :] - values[:-1, :]) / sy)
        result[1:, :] = np.maximum(result[1:, :], vertical)
        result[:-1, :] = np.maximum(result[:-1, :], vertical)
    return result


def _edge_gradient_volume(array: Any, spacing: tuple[float, float]) -> Any:
    """Build an in-plane physical-gradient field for a canonical array."""

    shape = _shape3(array)
    result = np.zeros(shape, dtype=np.float64)
    for k in range(shape[0]):
        result[k] = _edge_gradient(array[k], spacing)
    return result


def _cached_edge_gradient_volume(
    context: _Context, array: Any, spacing: tuple[float, float], label: str
) -> Any:
    """Return one exact in-plane gradient field for this evaluation context."""

    key = ("edge-gradient-volume", label, id(array), tuple(float(v) for v in spacing))
    cached = context.cache.get(key)
    if cached is None:
        cached = _edge_gradient_volume(array, spacing)
        context.cache[key] = cached
    return cached


def _three_d_gradient(context: _Context) -> Any:
    key = ("three-d-gradient", id(context.array), tuple(float(v) for v in context.frame.spacing_mm))
    cached = context.cache.get(key)
    if cached is not None:
        return cached
    array = context.array
    k_count, j_count, i_count = (int(value) for value in array.shape)
    sx, sy, sz = (float(value) for value in context.frame.spacing_mm)
    result = np.zeros(array.shape, dtype=np.float64)
    values = np.asarray(array, dtype=np.float64)
    if i_count > 1:
        horizontal = np.abs((values[:, :, 1:] - values[:, :, :-1]) / sx)
        result[:, :, 1:] = np.maximum(result[:, :, 1:], horizontal)
        result[:, :, :-1] = np.maximum(result[:, :, :-1], horizontal)
    if j_count > 1:
        vertical = np.abs((values[:, 1:, :] - values[:, :-1, :]) / sy)
        result[:, 1:, :] = np.maximum(result[:, 1:, :], vertical)
        result[:, :-1, :] = np.maximum(result[:, :-1, :], vertical)
    if k_count > 1:
        depth = np.abs((values[1:, :, :] - values[:-1, :, :]) / sz)
        result[1:, :, :] = np.maximum(result[1:, :, :], depth)
        result[:-1, :, :] = np.maximum(result[:-1, :, :], depth)
    context.cache[key] = result
    return result


def _weighted_max_p95(
    values: Sequence[float], max_weight: float = 0.50, p95_weight: float = 0.50
) -> float:
    if not values:
        raise _fail(QCAbstentionCode.QC_INSUFFICIENT_SUPPORT, "no supported slices")
    return float(max_weight * max(values) + p95_weight * lower_quantile(values, 0.95))


def _shape3(array: Any) -> tuple[int, int, int]:
    shape = tuple(int(value) for value in array.shape)
    if len(shape) != 3:
        raise _fail(QCAbstentionCode.QC_INPUT_SURFACE_INVALID, "a 3-D array is required")
    return cast(tuple[int, int, int], shape)


def _slice_edge_set(
    values: Any,
    mask: Any,
    spacing: tuple[float, float],
) -> set[tuple[int, int]]:
    """Return the native-grid Q.90 edge set inside one slice mask."""

    return _slice_edges_and_threshold(values, mask, spacing)[0]


def _cached_slice_edges(
    context: _Context, slice_index: int, mask: Any
) -> tuple[set[tuple[int, int]], float]:
    """Cache a native slice edge population shared by adjacent comparisons."""

    key = ("slice-edges", id(context.array), int(slice_index), id(mask))
    cached = context.cache.get(key)
    if cached is None:
        cached = _slice_edges_and_threshold(
            context.array[slice_index], mask[slice_index], context.frame.spacing_mm[:2]
        )
        context.cache[key] = cached
    return cast(tuple[set[tuple[int, int]], float], cached)


def _slice_edges_and_threshold(
    values: Any,
    mask: Any,
    spacing: tuple[float, float],
) -> tuple[set[tuple[int, int]], float]:
    """Return a native-grid edge set and the Q.90 used to define it."""

    gradient = _edge_gradient(values, spacing)
    locations = np.asarray(gradient)[np.asarray(mask, dtype=bool)]
    if locations.size == 0:
        raise _fail(QCAbstentionCode.QC_INSUFFICIENT_SUPPORT, "edge mask is empty")
    threshold = lower_quantile(locations, 0.90)
    return (
        {
            (int(j), int(i))
            for j, i in np.argwhere(np.asarray(mask, dtype=bool) & (gradient >= threshold))
        },
        threshold,
    )


def _result(flag: str, metric: float, outcome: QCOutcome) -> FamilyResult:
    if not math.isfinite(float(metric)):
        raise _fail(QCAbstentionCode.QC_INPUT_SURFACE_INVALID, "metric is nonfinite")
    return FamilyResult(
        flag=flag,
        metric_value=float(metric),
        outcome=outcome,
        algorithm_id=FAMILY_ALGORITHM_IDS[flag],
        metric_name=FAMILY_METRIC_NAMES[flag],
        parameter_digest=family_parameter_digest(flag),
    )


def _abstained(flag: str, code: QCAbstentionCode | str) -> FamilyResult:
    return FamilyResult(
        flag=flag,
        metric_value=None,
        outcome=QCOutcome.ABSTAINED,
        abstention_code=code,
        algorithm_id=FAMILY_ALGORITHM_IDS[flag],
        metric_name=FAMILY_METRIC_NAMES[flag],
        parameter_digest=family_parameter_digest(flag),
    )


def _source_range(context: _Context) -> float:
    key = ("source-range", id(context.array))
    cached = context.cache.get(key)
    if cached is None:
        cached = robust_range(context.array)
        context.cache[key] = cached
    return float(cached)


def _pair_samples(
    context: _Context,
    pair: RegisteredPairInput,
    *,
    central_only: bool,
) -> tuple[tuple[_PairSample, ...], int]:
    from .pair import pair_affine_and_overlap

    key = ("pair-samples", id(pair), bool(central_only))
    cached = context.cache.get(key)
    if cached is not None:
        return cast(tuple[tuple[_PairSample, ...], int], cached)

    if pair.first_volume != context.volume:
        raise _fail(
            QCAbstentionCode.QC_REQUIRED_PAIR_INVALID,
            "pair first Volume differs from QC input",
        )
    if (
        hashlib.sha256(pair.first_array.tobytes(order="C")).hexdigest()
        != context.volume.blob_ref.sha256
    ):
        raise _fail(
            QCAbstentionCode.QC_REQUIRED_PAIR_INVALID,
            "pair first array is not the QC volume",
        )
    matrix, lower, upper = pair_affine_and_overlap(pair)
    inverse_second = invert_matrix(pair.second_volume.coordinate_frame.voxel_to_world_matrix())
    shape = tuple(int(value) for value in context.array.shape)
    mask = _central_mask(context, u_abs=0.35, v_abs=0.35, w_abs=0.75)
    identity = (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )
    first_frame = context.frame
    second_frame = pair.second_volume.coordinate_frame
    if (
        matrix == identity
        and first_frame.voxel_to_world_matrix() == second_frame.voxel_to_world_matrix()
        and shape == tuple(int(value) for value in pair.second_array.shape)
        and all(
            lower[axis] <= corner[axis] <= upper[axis]
            for corner in _frame_world_corners(first_frame)
            for axis in range(3)
        )
    ):
        positions = (
            (k, j, i)
            for k in range(shape[0])
            for j in range(shape[1])
            for i in range(shape[2])
            if not central_only or bool(mask[k, j, i])
        )
        fast_samples = tuple(
            _PairSample(
                k,
                j,
                i,
                float(context.array[k, j, i]),
                float(pair.second_array[k, j, i]),
                (float(i), float(j), float(k)),
            )
            for k, j, i in positions
        )
        result = (
            fast_samples,
            int(np.count_nonzero(mask)) if central_only else int(np.prod(shape)),
        )
        context.cache[key] = result
        return result
    total = 0
    samples: list[_PairSample] = []
    for k in range(shape[0]):
        for j in range(shape[1]):
            for i in range(shape[2]):
                if central_only and not bool(mask[k, j, i]):
                    continue
                total += 1
                point = world_center(context.frame, k, j, i)
                if not all(lower[index] <= point[index] <= upper[index] for index in range(3)):
                    continue
                q: list[float] = []
                for row in range(3):
                    value = 0.0
                    for column, coordinate in enumerate((*point, 1.0)):
                        value = value + float(matrix[row][column]) * float(coordinate)
                    q.append(value)
                second_index = _apply_matrix_to_point(inverse_second, (q[0], q[1], q[2]))
                sampled_value = _sample_float32(pair.second_array, second_index)
                if sampled_value is None or not math.isfinite(sampled_value):
                    continue
                first_value = float(context.array[k, j, i])
                samples.append(
                    _PairSample(k, j, i, first_value, float(sampled_value), second_index)
                )
    result = (tuple(samples), total)
    context.cache[key] = result
    return result


def _pair_scale(context: _Context, pair: RegisteredPairInput) -> float:
    """Compute ``S_pair`` without materializing unused full-grid samples."""

    key = ("pair-scale", id(pair))
    cached = context.cache.get(key)
    if cached is not None:
        return float(cached)
    from .pair import pair_affine_and_overlap

    matrix, lower, upper = pair_affine_and_overlap(pair)
    identity = (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )
    first_frame = context.frame
    second_frame = pair.second_volume.coordinate_frame
    shape = tuple(int(value) for value in context.array.shape)
    if (
        matrix == identity
        and first_frame.voxel_to_world_matrix() == second_frame.voxel_to_world_matrix()
        and shape == tuple(int(value) for value in pair.second_array.shape)
        and all(
            lower[axis] <= corner[axis] <= upper[axis]
            for corner in _frame_world_corners(first_frame)
            for axis in range(3)
        )
    ):
        values = np.concatenate(
            (
                np.asarray(context.array, dtype=np.float64).reshape(-1),
                np.asarray(pair.second_array, dtype=np.float64).reshape(-1),
            )
        )
    else:
        samples, _ = _pair_samples(context, pair, central_only=False)
        values = np.asarray(
            [value for item in samples for value in (item.first, item.second)], dtype=np.float64
        )
    result = robust_range(values)
    context.cache[key] = result
    return result


def _frame_world_corners(frame: FrameSpec) -> tuple[tuple[float, float, float], ...]:
    """Return the eight voxel-center world corners for overlap fast-path checks."""

    matrix = frame.voxel_to_world_matrix()
    shape = tuple(int(value) for value in frame.dimensions)
    return tuple(
        (
            matrix[0][3] + matrix[0][0] * i + matrix[0][1] * j + matrix[0][2] * k,
            matrix[1][3] + matrix[1][0] * i + matrix[1][1] * j + matrix[1][2] * k,
            matrix[2][3] + matrix[2][0] * i + matrix[2][1] * j + matrix[2][2] * k,
        )
        for k in (0, shape[0] - 1)
        for j in (0, shape[1] - 1)
        for i in (0, shape[2] - 1)
    )


def _apply_matrix_to_point(
    matrix: Any, point: tuple[float, float, float]
) -> tuple[float, float, float]:
    result: list[float] = []
    for row in range(3):
        value = 0.0
        for column, coordinate in enumerate((*point, 1.0)):
            value = value + float(matrix[row][column]) * float(coordinate)
        result.append(value)
    return cast(tuple[float, float, float], tuple(result))


def _sample_float32(array: Any, index: tuple[float, float, float]) -> float | None:
    """Sample with the admitted float32 left-associative trilinear rule."""

    k_count, j_count, i_count = (int(value) for value in array.shape)
    x, y, z = index
    if not all(math.isfinite(value) for value in index):
        return None
    if x < 0.0 or y < 0.0 or z < 0.0 or x > i_count - 1 or y > j_count - 1 or z > k_count - 1:
        return None
    i0 = min(int(math.floor(x)), i_count - 1)
    j0 = min(int(math.floor(y)), j_count - 1)
    k0 = min(int(math.floor(z)), k_count - 1)
    i1 = min(i0 + 1, i_count - 1)
    j1 = min(j0 + 1, j_count - 1)
    k1 = min(k0 + 1, k_count - 1)
    dx = np.float32(x - i0)
    dy = np.float32(y - j0)
    dz = np.float32(z - k0)

    def f(value: Any) -> Any:
        return np.float32(value)

    c000 = f(array[k0, j0, i0])
    c100 = f(array[k0, j0, i1])
    c010 = f(array[k0, j1, i0])
    c110 = f(array[k0, j1, i1])
    c001 = f(array[k1, j0, i0])
    c101 = f(array[k1, j0, i1])
    c011 = f(array[k1, j1, i0])
    c111 = f(array[k1, j1, i1])
    c00 = f(c000 + f(dx * f(c100 - c000)))
    c10 = f(c010 + f(dx * f(c110 - c010)))
    c01 = f(c001 + f(dx * f(c101 - c001)))
    c11 = f(c011 + f(dx * f(c111 - c011)))
    c0 = f(c00 + f(dy * f(c10 - c00)))
    c1 = f(c01 + f(dy * f(c11 - c01)))
    return float(f(c0 + f(dz * f(c1 - c0))))


def _sample_2d_float32(array: Any, index: tuple[float, float]) -> float | None:
    """Sample one ``(row,column)`` field with the same binary32 lerp rule."""

    y, x = index
    rows, columns = int(array.shape[0]), int(array.shape[1])
    if not all(math.isfinite(value) for value in index):
        return None
    if y < 0.0 or x < 0.0 or y > rows - 1 or x > columns - 1:
        return None
    j0 = min(int(math.floor(y)), rows - 1)
    i0 = min(int(math.floor(x)), columns - 1)
    j1 = min(j0 + 1, rows - 1)
    i1 = min(i0 + 1, columns - 1)
    dx = np.float32(x - i0)
    dy = np.float32(y - j0)

    def f(value: Any) -> Any:
        return np.float32(value)

    c00 = f(array[j0, i0])
    c10 = f(array[j0, i1])
    c01 = f(array[j1, i0])
    c11 = f(array[j1, i1])
    top = f(c00 + f(dx * f(c10 - c00)))
    bottom = f(c01 + f(dx * f(c11 - c01)))
    return float(f(top + f(dy * f(bottom - top))))


def _physical_slice_order(context: _Context) -> tuple[int, ...]:
    """Order slices on a canonical superior-positive physical axis.

    The sign of ``row × column`` is an orientation convention, not a
    physical ordering decision.  Canonicalizing that sign keeps a legal
    reversal of both stored slice order and in-plane handedness from changing
    which member of an adjacent pair supplies the edge population.  Derive
    the order from the signed slice step rather than projecting a world
    centre, so an equivalent representation cannot change the result through
    binary64 addition order.
    """

    key = ("physical-slice-order", id(context.array), context.frame)
    cached = context.cache.get(key)
    if cached is not None:
        return cast(tuple[int, ...], cached)
    shape = _shape3(context.array)
    _, _, normal = frame_axes(context.frame)
    normal = _superior_positive_normal(normal)
    matrix = context.frame.voxel_to_world_matrix()
    slice_step = sum(matrix[axis][2] * normal[axis] for axis in range(3))
    if not math.isfinite(slice_step) or slice_step == 0.0:
        raise _fail(
            QCAbstentionCode.QC_GEOMETRY_INVALID,
            "slice step cannot establish a physical order",
        )
    result = (
        tuple(range(shape[0]))
        if slice_step > 0.0
        else tuple(range(shape[0] - 1, -1, -1))
    )
    context.cache[key] = result
    return result


def _slice_intersection_samples(
    context: _Context,
    first_k: int,
    second_k: int,
    mask: Any,
) -> tuple[tuple[int, int, tuple[float, float, float]], ...]:
    """Map first-slice centers into the second slice's in-plane rectangle."""

    shape = _shape3(context.array)
    # Adjacent slices in one canonical volume necessarily share this
    # FrameSpec.  Their physical in-plane center grids are therefore exactly
    # coincident; spelling out that identity avoids an unnecessary affine
    # inversion/interpolation round trip while preserving the mandated
    # physical-intersection population and C-order traversal.
    del context
    coordinates = np.argwhere(mask[first_k])
    return tuple(
        (int(j), int(i), (float(i), float(j), float(second_k)))
        for j, i in coordinates
        if 0 <= int(j) < shape[1] and 0 <= int(i) < shape[2]
    )


def _slice_edge_intersection(
    context: _Context,
    first_k: int,
    second_k: int,
    mask: Any,
) -> tuple[tuple[int, int, tuple[float, float, float]], ...]:
    """Return common-grid centers carrying first-slice edge support."""

    first_edges = _cached_slice_edges(context, first_k, mask)[0]
    mapped = _slice_intersection_samples(context, first_k, second_k, mask)
    return tuple(item for item in mapped if (item[0], item[1]) in first_edges)


def _validate_pair_for_family(
    context: _Context, pair: RegisteredPairInput | None
) -> RegisteredPairInput:
    if pair is None:
        raise _fail(
            QCAbstentionCode.QC_REQUIRED_PAIR_ABSENT, "explicit registered pair is required"
        )
    key = ("validated-pair", id(pair))
    if key not in context.cache:
        try:
            from .pair import validate_registered_pair_input

            validate_registered_pair_input(pair)
        except QCValidationError as exc:
            raise _fail(QCAbstentionCode.QC_REQUIRED_PAIR_INVALID, str(exc)) from exc
        context.cache[key] = True
    if pair.first_volume != context.volume:
        raise _fail(QCAbstentionCode.QC_REQUIRED_PAIR_INVALID, "pair is not bound to first Volume")
    if (
        hashlib.sha256(pair.first_array.tobytes(order="C")).hexdigest()
        != context.volume.blob_ref.sha256
    ):
        raise _fail(
            QCAbstentionCode.QC_REQUIRED_PAIR_INVALID, "pair first array is not the QC volume"
        )
    return pair


def cardiac_motion(context: _Context, pair: RegisteredPairInput | None = None) -> FamilyResult:
    flag = "CARDIAC_MOTION"
    registered = _validate_pair_for_family(context, pair)
    samples, central_total = _pair_samples(context, registered, central_only=True)
    if central_total <= 0 or len(samples) / central_total < 0.80:
        raise _fail(
            QCAbstentionCode.QC_REGISTRATION_INVALID, "central registered overlap is below 0.80"
        )
    scale = _pair_scale(context, registered)
    by_slice: dict[int, list[_PairSample]] = {}
    for item in samples:
        by_slice.setdefault(item.k, []).append(item)
    slice_scores: list[float] = []
    second_gradient = _cached_edge_gradient_volume(
        context,
        registered.second_array,
        (
            float(registered.second_volume.coordinate_frame.spacing_mm[0]),
            float(registered.second_volume.coordinate_frame.spacing_mm[1]),
        ),
        "pair-second",
    )
    native_gradient_volume = _cached_edge_gradient_volume(
        context, context.array, context.frame.spacing_mm[:2], "source"
    )
    shape = _shape3(context.array)
    central = _central_mask(context, u_abs=0.35, v_abs=0.35, w_abs=0.75)
    for k in range(shape[0]):
        if not bool(np.any(central[k])):
            continue
        gradient = native_gradient_volume[k]
        native_gradients = [
            float(gradient[j, i])
            for j in range(shape[1])
            for i in range(shape[2])
            if bool(central[k, j, i])
        ]
        if not native_gradients:
            raise _fail(QCAbstentionCode.QC_INSUFFICIENT_SUPPORT, "cardiac edge mask is empty")
        threshold = lower_quantile(native_gradients, 0.90)
        edge = [
            item
            for item in by_slice.get(k, ())
            if float(gradient[item.j, item.i]) >= threshold
        ]
        if len(edge) < 32:
            raise _fail(QCAbstentionCode.QC_INSUFFICIENT_SUPPORT, "cardiac edge support below 32")
        residual = median([abs(item.first - item.second) for item in edge]) / scale
        g_a = lower_quantile([float(gradient[item.j, item.i]) for item in edge], 0.95)
        g_b_values: list[float] = []
        for item in edge:
            sampled_gradient = _sample_float32(second_gradient, item.second_index)
            if sampled_gradient is None or not math.isfinite(sampled_gradient):
                raise _fail(
                    QCAbstentionCode.QC_GEOMETRY_INVALID,
                    "registered second-gradient sample is invalid",
                )
            g_b_values.append(sampled_gradient)
        g_b = lower_quantile(g_b_values, 0.95)
        sharpness = 1.0 - min(g_a, g_b) / max(g_a, g_b, _EPS)
        slice_scores.append(0.75 * residual + 0.25 * sharpness)
    score = _weighted_max_p95(slice_scores)
    return _result(flag, score, classify_high_bad(score, 0.08, 0.16))


def respiratory_motion(context: _Context) -> FamilyResult:
    flag = "RESPIRATORY_MOTION"
    shape = _shape3(context.array)
    if shape[0] < 3:
        raise _fail(QCAbstentionCode.QC_INSUFFICIENT_SUPPORT, "at least three slices are required")
    scale = _source_range(context)
    mask = _central_mask(context, u_abs=0.35, v_abs=0.35, w_abs=0.75)
    scores: list[float] = []
    order = _physical_slice_order(context)
    for first_k, second_k in zip(order, order[1:]):
        if not bool(np.any(mask[first_k])) or not bool(np.any(mask[second_k])):
            continue
        edge_samples = _slice_edge_intersection(context, first_k, second_k, mask)
        if len(edge_samples) < 32:
            raise _fail(
                QCAbstentionCode.QC_INSUFFICIENT_SUPPORT,
                "adjacent slice edge support is below 32",
            )
        residuals: list[float] = []
        for j, i, second_index in edge_samples:
            second = _sample_float32(context.array, second_index)
            if second is None or not math.isfinite(second):
                raise _fail(
                    QCAbstentionCode.QC_GEOMETRY_INVALID,
                    "adjacent comparison sample is outside the physical grid",
                )
            residuals.append(abs(float(context.array[first_k, j, i]) - second))
        scores.append(median(residuals) / scale)
    if len(scores) < 2:
        raise _fail(
            QCAbstentionCode.QC_INSUFFICIENT_SUPPORT,
            "fewer than two adjacent supported pairs",
        )
    score = _weighted_max_p95(scores)
    return _result(flag, score, classify_high_bad(score, 0.08, 0.16))


def stair_step(context: _Context) -> FamilyResult:
    flag = "STAIR_STEP"
    shape = _shape3(context.array)
    if shape[0] < 3:
        raise _fail(QCAbstentionCode.QC_INSUFFICIENT_SUPPORT, "at least three slices are required")
    mask = _central_mask(context, u_abs=0.35, v_abs=0.35, w_abs=0.75)
    discontinuities: list[float] = []
    order = _physical_slice_order(context)
    for first_k, second_k in zip(order, order[1:]):
        if not bool(np.any(mask[first_k])) or not bool(np.any(mask[second_k])):
            continue
        first_edges, first_threshold = _cached_slice_edges(context, first_k, mask)
        second_edges, second_threshold = _cached_slice_edges(context, second_k, mask)
        # The comparison is between adjacent slices of this one FrameSpec,
        # so the first slice's physical intersection grid is its native
        # in-plane grid.  Integer-center sampling is exact identity here.
        del first_threshold, second_threshold
        first_edge_set = first_edges
        second_edge_set = second_edges
        if len(first_edge_set) < 32 or len(second_edge_set) < 32:
            raise _fail(
                QCAbstentionCode.QC_INSUFFICIENT_SUPPORT,
                "stair-step comparison edge support is below 32",
            )
        union = first_edge_set | second_edge_set
        intersection = first_edge_set & second_edge_set
        if not union:
            raise _fail(QCAbstentionCode.QC_INSUFFICIENT_SUPPORT, "stair-step edge union is empty")
        discontinuities.append(1.0 - len(intersection) / len(union))
    if len(discontinuities) < 2:
        raise _fail(
            QCAbstentionCode.QC_INSUFFICIENT_SUPPORT,
            "fewer than two adjacent edge pairs",
        )
    score = _weighted_max_p95(discontinuities)
    mild_count = sum(value >= 0.15 for value in discontinuities)
    severe_count = sum(value >= 0.30 for value in discontinuities)
    outcome = classify_stair_step(
        score,
        supported_mild_count=mild_count,
        supported_severe_count=severe_count,
        pair_count=len(discontinuities),
    )
    return _result(flag, score, outcome)


def _clip_fraction(context: _Context, values: Any) -> float:
    lower, upper = (
        (context.source_low, context.source_high)
        if context.scalar_domain == "STORED"
        else (context.transformed_low, context.transformed_high)
    )
    clipped = (values == lower) | (values == upper)
    return float(np.count_nonzero(clipped)) / float(values.size)


def poor_contrast(context: _Context) -> FamilyResult:
    flag = "POOR_CONTRAST"
    mask_c = _central_mask(context, u_abs=0.30, v_abs=0.30, w_abs=0.75)
    shape = cast(tuple[int, int, int], tuple(int(value) for value in context.array.shape))
    u, v, _ = _normalized_axes(shape)
    shell_2d = (np.maximum(np.abs(v)[None, :, None], np.abs(u)[None, None, :]) >= 0.45) & (
        np.maximum(np.abs(v)[None, :, None], np.abs(u)[None, None, :]) <= 0.75
    )
    _, _, w = _normalized_axes(shape)
    mask_r = shell_2d & (np.abs(w)[:, None, None] <= 0.75)
    values_c = context.array[mask_c]
    values_r = context.array[mask_r]
    if values_c.size < 64 or values_r.size < 64:
        raise _fail(QCAbstentionCode.QC_INSUFFICIENT_SUPPORT, "contrast masks require 64 voxels")
    union = context.array[mask_c | mask_r]
    if _clip_fraction(context, union) > 0.01:
        raise _fail(QCAbstentionCode.QC_CLIPPING_UNRESOLVABLE, "regional clipping exceeds 0.01")
    scale = _source_range(context)
    centre = median(values_c)
    reference = median(values_r)
    denominator = math.sqrt(
        mad(values_c) * mad(values_c) + mad(values_r) * mad(values_r) + (1e-12 * scale) ** 2
    )
    if denominator <= 0.0 or not math.isfinite(denominator):
        raise _fail(QCAbstentionCode.QC_DYNAMIC_RANGE_INVALID, "contrast denominator is invalid")
    cnr = (centre - reference) / denominator
    return _result(flag, cnr, classify_low_bad(cnr, 1.0, 2.0))


def _box_mean(array: Any, k: int, j: int, i: int, radius: int) -> float:
    k0, k1 = max(0, k - radius), min(int(array.shape[0]) - 1, k + radius)
    j0, j1 = max(0, j - radius), min(int(array.shape[1]) - 1, j + radius)
    i0, i1 = max(0, i - radius), min(int(array.shape[2]) - 1, i + radius)
    total = 0.0
    count = 0
    for kk in range(k0, k1 + 1):
        for jj in range(j0, j1 + 1):
            for ii in range(i0, i1 + 1):
                total = total + float(array[kk, jj, ii])
                count += 1
    if count == 0:
        raise _fail(QCAbstentionCode.QC_INSUFFICIENT_SUPPORT, "empty box")
    return total / count


def excessive_noise(context: _Context) -> FamilyResult:
    flag = "EXCESSIVE_NOISE"
    mask = _central_mask(context, u_abs=0.30, v_abs=0.30, w_abs=0.75)
    gradient = _three_d_gradient(context)
    gradient_values = gradient[mask]
    if gradient_values.size == 0:
        raise _fail(QCAbstentionCode.QC_INSUFFICIENT_SUPPORT, "noise central mask is empty")
    qg = lower_quantile(gradient_values, 0.50)
    support = [
        (k, j, i)
        for k in range(context.array.shape[0])
        for j in range(context.array.shape[1])
        for i in range(context.array.shape[2])
        if bool(mask[k, j, i]) and float(gradient[k, j, i]) <= qg
    ]
    if len(support) < 512:
        raise _fail(QCAbstentionCode.QC_INSUFFICIENT_SUPPORT, "homogeneous support is below 512")
    scale = _source_range(context)
    residuals: list[float] = []
    for radius in (1, 2):
        squared: list[float] = []
        for k, j, i in support:
            delta = float(context.array[k, j, i]) - _box_mean(context.array, k, j, i, radius)
            squared.append(delta * delta)
        residuals.append(math.sqrt(sum(squared) / len(squared)) / scale)
    variances: list[float] = []
    for k, j, i in support:
        neighbours: list[float] = []
        for kk in range(max(0, k - 1), min(context.array.shape[0] - 1, k + 1) + 1):
            for jj in range(max(0, j - 1), min(context.array.shape[1] - 1, j + 1) + 1):
                for ii in range(max(0, i - 1), min(context.array.shape[2] - 1, i + 1) + 1):
                    neighbours.append(float(context.array[kk, jj, ii]))
        mean_value = sum(neighbours) / len(neighbours)
        variances.append(sum((value - mean_value) ** 2 for value in neighbours) / len(neighbours))
    variance_score = math.sqrt(lower_quantile(variances, 0.95)) / scale
    score = 0.50 * max(residuals) + 0.50 * variance_score
    return _result(flag, score, classify_high_bad(score, 0.08, 0.15))


def _component_seeds(mask: Any) -> tuple[set[tuple[int, int, int]], ...]:
    remaining = {(int(k), int(j), int(i)) for k, j, i in zip(*np.nonzero(mask))}
    components: list[set[tuple[int, int, int]]] = []
    while remaining:
        seed = min(remaining)
        remaining.remove(seed)
        component = {seed}
        pending = [seed]
        while pending:
            k, j, i = pending.pop()
            for dk, dj, di in _SIX_NEIGHBOURS:
                candidate = (k + dk, j + dj, i + di)
                if candidate in remaining:
                    remaining.remove(candidate)
                    component.add(candidate)
                    pending.append(candidate)
        components.append(component)
    return tuple(components)


def _flood_threshold(
    context: _Context,
    seeds: set[tuple[int, int, int]],
    central: Any,
    threshold: float,
) -> set[tuple[int, int, int]]:
    allowed = {
        (int(k), int(j), int(i))
        for k, j, i in zip(*np.nonzero(central & (context.array >= threshold)))
    }
    result = set(seed for seed in seeds if seed in allowed)
    pending = list(result)
    while pending:
        k, j, i = pending.pop()
        for dk, dj, di in _SIX_NEIGHBOURS:
            candidate = (k + dk, j + dj, i + di)
            if candidate in allowed and candidate not in result:
                result.add(candidate)
                pending.append(candidate)
    return result


def blooming(context: _Context) -> FamilyResult:
    flag = "BLOOMING"
    central = _central_mask(context, u_abs=0.30, v_abs=0.30, w_abs=0.75)
    if int(np.count_nonzero(central)) < 64:
        raise _fail(QCAbstentionCode.QC_INSUFFICIENT_SUPPORT, "blooming central mask is below 64")
    all_values = context.array.reshape(-1)
    # The robust range is a prerequisite for the high-density profile, even
    # when the candidate population happens to be empty.  A valid negative
    # is CLEAR; an uncalibrated/flat profile is ABSTAINED.
    robust_range(all_values)
    threshold = max(
        lower_quantile(all_values, 0.99),
        median(all_values) + 5.0 * 1.4826 * mad(all_values),
    )
    candidates = central & (context.array >= threshold)
    if not bool(np.any(candidates)):
        return _result(flag, 0.0, QCOutcome.CLEAR)
    retained = tuple(component for component in _component_seeds(candidates) if len(component) >= 8)
    if not retained:
        raise _fail(
            QCAbstentionCode.QC_INSUFFICIENT_SUPPORT, "all blooming candidates are undersupported"
        )
    scores: list[float] = []
    for component in retained:
        shell = {
            (k + dk, j + dj, i + di)
            for k, j, i in component
            for dk, dj, di in _SIX_NEIGHBOURS
            if 0 <= k + dk < context.array.shape[0]
            and 0 <= j + dj < context.array.shape[1]
            and 0 <= i + di < context.array.shape[2]
            and bool(central[k + dk, j + dj, i + di])
            and (k + dk, j + dj, i + di) not in component
        }
        if not shell:
            raise _fail(
                QCAbstentionCode.QC_INSUFFICIENT_SUPPORT, "blooming exterior shell is empty"
            )
        background = median([float(context.array[index]) for index in shell])
        peak = max(float(context.array[index]) for index in component)
        if peak <= background:
            raise _fail(
                QCAbstentionCode.QC_INSUFFICIENT_SUPPORT, "blooming profile is not positive"
            )
        t50 = background + 0.50 * (peak - background)
        t90 = background + 0.90 * (peak - background)
        a50 = _flood_threshold(context, component, central, t50)
        a90 = _flood_threshold(context, component, central, t90)
        scores.append((len(a50) - len(a90)) / max(len(a90), 1))
    score = _weighted_max_p95(scores)
    return _result(flag, score, classify_high_bad(score, 0.25, 0.50))


def beam_hardening(context: _Context) -> FamilyResult:
    flag = "BEAM_HARDENING"
    shape = cast(tuple[int, int, int], tuple(int(value) for value in context.array.shape))
    u, v, _ = _normalized_axes(shape)
    rho = np.sqrt(v[:, None] * v[:, None] + u[None, :] * u[None, :])
    angle = np.arctan2(v[:, None], u[None, :])
    supported_needed = math.ceil(0.80 * shape[0])
    supported_scores: list[float] = []
    supported_slices = range(shape[0])
    for k in supported_slices:
        slice_values = context.array[k]
        domain = np.isfinite(slice_values) & (rho <= 0.85)
        body_pixels = slice_values[domain]
        if body_pixels.size == 0:
            continue
        q10 = lower_quantile(body_pixels, 0.10)
        q90 = lower_quantile(body_pixels, 0.90)
        body = domain & (slice_values >= q10 + 0.05 * (q90 - q10))
        central = body & (rho <= 0.25)
        annulus = body & (rho >= 0.45) & (rho <= 0.70)
        if np.count_nonzero(central) < 64 or np.count_nonzero(annulus) < 64:
            continue
        sectors: list[Any] = []
        for sector in range(8):
            lo = -math.pi + sector * (2.0 * math.pi / 8.0)
            hi = -math.pi + (sector + 1) * (2.0 * math.pi / 8.0)
            sectors.append(annulus & (angle >= lo) & (angle < hi))
        if any(np.count_nonzero(item) < 8 for item in sectors):
            continue
        clip_population = slice_values[central | annulus]
        if _clip_fraction(context, clip_population) > 0.01:
            raise _fail(
                QCAbstentionCode.QC_CLIPPING_UNRESOLVABLE, "beam-hardening clipping exceeds 0.01"
            )
        # S_k is defined over every finite in-plane D_k value, not over the
        # thresholded body mask or the central/annular subsets.
        scale = lower_quantile(body_pixels, 0.95) - lower_quantile(body_pixels, 0.05)
        if not math.isfinite(scale) or scale <= 0.0:
            raise _fail(QCAbstentionCode.QC_DYNAMIC_RANGE_INVALID, "slice robust range is invalid")
        c = max(0.0, (median(slice_values[annulus]) - median(slice_values[central])) / scale)
        sector_medians = [median(slice_values[item]) for item in sectors]
        asymmetry = (max(sector_medians) - min(sector_medians)) / scale
        supported_scores.append(max(c, asymmetry))
    if len(supported_scores) < supported_needed:
        raise _fail(
            QCAbstentionCode.QC_INSUFFICIENT_SUPPORT, "supported slices are below ceil(0.80*K)"
        )
    score = _weighted_max_p95(supported_scores)
    return _result(flag, score, classify_high_bad(score, 0.05, 0.10))


def misregistration(context: _Context, pair: RegisteredPairInput | None = None) -> FamilyResult:
    flag = "MISREGISTRATION"
    registered = _validate_pair_for_family(context, pair)
    samples, central_total = _pair_samples(context, registered, central_only=True)
    if central_total <= 0 or len(samples) / central_total < 0.80:
        raise _fail(
            QCAbstentionCode.QC_REGISTRATION_INVALID, "central registered overlap is below 0.80"
        )
    scale = _pair_scale(context, registered)
    by_slice: dict[int, list[_PairSample]] = {}
    for item in samples:
        by_slice.setdefault(item.k, []).append(item)
    scores: list[float] = []
    shape = _shape3(context.array)
    central = _central_mask(context, u_abs=0.35, v_abs=0.35, w_abs=0.75)
    native_gradient_volume = _cached_edge_gradient_volume(
        context, context.array, context.frame.spacing_mm[:2], "source"
    )
    for k in range(shape[0]):
        if not bool(np.any(central[k])):
            continue
        gradient = native_gradient_volume[k]
        native_gradients = [
            float(gradient[j, i])
            for j in range(shape[1])
            for i in range(shape[2])
            if bool(central[k, j, i])
        ]
        if not native_gradients:
            raise _fail(QCAbstentionCode.QC_INSUFFICIENT_SUPPORT, "registered edge mask is empty")
        threshold = lower_quantile(native_gradients, 0.90)
        edge = [
            item
            for item in by_slice.get(k, ())
            if float(gradient[item.j, item.i]) >= threshold
        ]
        if len(edge) < 32:
            raise _fail(
                QCAbstentionCode.QC_INSUFFICIENT_SUPPORT, "registered edge support below 32"
            )
        scores.append(
            lower_quantile([abs(item.first - item.second) for item in edge], 0.95) / scale
        )
    if not scores:
        raise _fail(QCAbstentionCode.QC_INSUFFICIENT_SUPPORT, "no registered edge slices")
    score = _weighted_max_p95(scores)
    return _result(flag, score, classify_high_bad(score, 0.10, 0.20))


def _coerce_coverage(value: CoverageEnvelope | Mapping[str, object] | None) -> CoverageEnvelope:
    if isinstance(value, CoverageEnvelope):
        envelope = value
    elif isinstance(value, Mapping):
        allowed = {
            "schema_version",
            "coordinate_system",
            "anchor_world_mm",
            "inferior_extent_mm",
            "superior_extent_mm",
            "terminal_band_mm",
            "minimum_body_occupancy",
            "boundary_rule",
        }
        if set(value) != allowed:
            raise _fail(
                QCAbstentionCode.QC_COVERAGE_ENVELOPE_INVALID,
                "coverage envelope key set is not closed",
            )
        try:
            envelope = CoverageEnvelope(**cast(dict[str, Any], dict(value)))
        except (TypeError, ValueError, OverflowError) as exc:
            raise _fail(
                QCAbstentionCode.QC_COVERAGE_ENVELOPE_INVALID,
                "coverage envelope cannot be constructed",
            ) from exc
    else:
        raise _fail(QCAbstentionCode.QC_COVERAGE_ENVELOPE_INVALID, "coverage envelope is required")
    if (
        envelope.schema_version != "ccta.qc.coverage_envelope.v1"
        or envelope.coordinate_system != "DICOM_PATIENT_LPS_MM"
        or envelope.boundary_rule != "INCLUSIVE_VOXEL_CENTER_V1"
        or len(tuple(envelope.anchor_world_mm)) != 3
        or not all(math.isfinite(float(value)) for value in envelope.anchor_world_mm)
        or any(isinstance(value, bool) for value in envelope.anchor_world_mm)
        or envelope.inferior_extent_mm != 80.0
        or envelope.superior_extent_mm != 80.0
        or envelope.terminal_band_mm != 16.0
        or envelope.minimum_body_occupancy != 0.20
    ):
        raise _fail(
            QCAbstentionCode.QC_COVERAGE_ENVELOPE_INVALID,
            "coverage envelope literals are not frozen",
        )
    return envelope


def incomplete_anatomy(
    context: _Context, coverage: CoverageEnvelope | Mapping[str, object] | None = None
) -> FamilyResult:
    flag = "INCOMPLETE_ANATOMY"
    envelope = _coerce_coverage(coverage if coverage is not None else context.source.coverage)
    _, _, normal = frame_axes(context.frame)
    normal = _superior_positive_normal(normal)
    shape = tuple(int(value) for value in context.array.shape)
    anchor = tuple(float(value) for value in envelope.anchor_world_mm)
    matrix = context.frame.voxel_to_world_matrix()
    origin_projection = sum(
        (float(matrix[axis][3]) - anchor[axis]) * normal[axis] for axis in range(3)
    )
    index_projections = tuple(
        sum(float(matrix[axis][column]) * normal[axis] for axis in range(3))
        for column in range(3)
    )

    def projection(k: int, j: int, i: int) -> float:
        value = origin_projection
        value = value + index_projections[0] * float(i)
        value = value + index_projections[1] * float(j)
        value = value + index_projections[2] * float(k)
        return value

    # The projection of the affine voxel-center grid is affine in (i,j,k),
    # so its extrema are attained at the eight grid corners.  The terminal
    # occupancy still examines every voxel in each selected slice below; this
    # reduction only avoids traversing interior coordinates that cannot change
    # the extent extrema.
    projections = [
        projection(k, j, i)
        for k in (0, shape[0] - 1)
        for j in (0, shape[1] - 1)
        for i in (0, shape[2] - 1)
    ]
    slice_projections = [
        projection(k, shape[1] // 2, shape[2] // 2) for k in range(shape[0])
    ]
    if not projections or not all(math.isfinite(value) for value in projections):
        raise _fail(QCAbstentionCode.QC_GEOMETRY_INVALID, "world coverage projections are invalid")
    actual_inferior = min(projections)
    actual_superior = max(projections)
    inferior_raw = (80.0 + actual_inferior) / 80.0
    superior_raw = (80.0 - actual_superior) / 80.0
    inferior_deficit = 0.0 if math.isclose(inferior_raw, 0.0, abs_tol=_GEOMETRY_EPS) else max(
        0.0, inferior_raw
    )
    superior_deficit = 0.0 if math.isclose(superior_raw, 0.0, abs_tol=_GEOMETRY_EPS) else max(
        0.0, superior_raw
    )
    q10 = lower_quantile(context.array, 0.10)
    q90 = lower_quantile(context.array, 0.90)
    body_cut = q10 + 0.05 * (q90 - q10)
    if not math.isfinite(body_cut):
        raise _fail(QCAbstentionCode.QC_BODY_OCCUPANCY_INDETERMINATE, "body threshold is nonfinite")
    bands = (
        (actual_inferior, actual_inferior + 16.0),
        (actual_superior - 16.0, actual_superior),
    )
    occupancies: list[float] = []
    for low, high in bands:
        indices = [
            k
            for k, projection in enumerate(slice_projections)
            if (projection > low or math.isclose(projection, low, abs_tol=_GEOMETRY_EPS))
            and (projection < high or math.isclose(projection, high, abs_tol=_GEOMETRY_EPS))
        ]
        if not indices:
            raise _fail(
                QCAbstentionCode.QC_BODY_OCCUPANCY_INDETERMINATE, "terminal band contains no slice"
            )
        for k in indices:
            finite_values = context.array[k][np.isfinite(context.array[k])]
            if finite_values.size < 1:
                raise _fail(
                    QCAbstentionCode.QC_BODY_OCCUPANCY_INDETERMINATE,
                    "terminal slice has no finite support",
                )
            occupancies.append(
                float(np.count_nonzero(finite_values >= body_cut)) / float(finite_values.size)
            )
    if not occupancies:
        raise _fail(QCAbstentionCode.QC_BODY_OCCUPANCY_INDETERMINATE, "terminal occupancy is empty")
    occupancy_deficit = max(0.0, (0.20 - min(occupancies)) / 0.20)
    score = max(inferior_deficit, superior_deficit, occupancy_deficit)
    return _result(flag, score, classify_incomplete_anatomy(score))


def evaluate_family(
    flag: str,
    source: QCInput,
    *,
    pair: RegisteredPairInput | None = None,
    coverage: CoverageEnvelope | Mapping[str, object] | None = None,
) -> FamilyResult:
    """Validate and execute one frozen deterministic family."""

    if flag not in FIRST_NINE_FLAGS:
        raise ValueError(f"unknown deterministic QC family: {flag!r}")
    try:
        context = validate_qc_input(source)
        return _evaluate_context_family(flag, context, pair=pair, coverage=coverage)
    except QCInputError as exc:
        return _abstained(flag, exc.code)
    except (AttributeError, IndexError, KeyError, TypeError, ValueError, OverflowError) as exc:
        # A malformed supplied object is an input refusal, never a clean
        # result.  Keep the public result typed even when a caller bypasses
        # the dataclass constructor with an opaque/persisted object.
        del exc
        return _abstained(flag, QCAbstentionCode.QC_INPUT_SURFACE_INVALID)


def _evaluate_context_family(
    flag: str,
    context: _Context,
    *,
    pair: RegisteredPairInput | None,
    coverage: CoverageEnvelope | Mapping[str, object] | None,
) -> FamilyResult:
    """Dispatch one family after the shared input context is validated."""

    if flag == "CARDIAC_MOTION":
        return cardiac_motion(context, pair)
    if flag == "RESPIRATORY_MOTION":
        return respiratory_motion(context)
    if flag == "STAIR_STEP":
        return stair_step(context)
    if flag == "POOR_CONTRAST":
        return poor_contrast(context)
    if flag == "EXCESSIVE_NOISE":
        return excessive_noise(context)
    if flag == "BLOOMING":
        return blooming(context)
    if flag == "BEAM_HARDENING":
        return beam_hardening(context)
    if flag == "MISREGISTRATION":
        return misregistration(context, pair)
    return incomplete_anatomy(context, coverage)


def evaluate_all_families(
    source: QCInput,
    *,
    pair: RegisteredPairInput | None = None,
    coverage: CoverageEnvelope | Mapping[str, object] | None = None,
) -> tuple[FamilyResult, ...]:
    """Run all nine families in their immutable canonical order."""

    try:
        context = validate_qc_input(source)
    except QCInputError as exc:
        return tuple(_abstained(flag, exc.code) for flag in FIRST_NINE_FLAGS)
    except (AttributeError, IndexError, KeyError, TypeError, ValueError, OverflowError):
        return tuple(
            _abstained(flag, QCAbstentionCode.QC_INPUT_SURFACE_INVALID)
            for flag in FIRST_NINE_FLAGS
        )
    results: list[FamilyResult] = []
    for flag in FIRST_NINE_FLAGS:
        try:
            results.append(_evaluate_context_family(flag, context, pair=pair, coverage=coverage))
        except QCInputError as exc:
            results.append(_abstained(flag, exc.code))
        except (AttributeError, IndexError, KeyError, TypeError, ValueError, OverflowError):
            results.append(_abstained(flag, QCAbstentionCode.QC_INPUT_SURFACE_INVALID))
    return tuple(results)


# Explicit family aliases make the frozen family surface easy to discover and
# keep callers from having to dispatch on strings themselves.
compute_cardiac_motion = cardiac_motion
compute_respiratory_motion = respiratory_motion
compute_stair_step = stair_step
compute_poor_contrast = poor_contrast
compute_excessive_noise = excessive_noise
compute_blooming = blooming
compute_beam_hardening = beam_hardening
compute_misregistration = misregistration
compute_incomplete_anatomy = incomplete_anatomy
run_cardiac_motion = cardiac_motion
run_respiratory_motion = respiratory_motion
run_stair_step = stair_step
run_poor_contrast = poor_contrast
run_excessive_noise = excessive_noise
run_blooming = blooming
run_beam_hardening = beam_hardening
run_misregistration = misregistration
run_incomplete_anatomy = incomplete_anatomy


__all__ = [
    "beam_hardening",
    "blooming",
    "cardiac_motion",
    "classify_high_bad",
    "classify_incomplete_anatomy",
    "classify_low_bad",
    "classify_stair_step",
    "compute_beam_hardening",
    "compute_blooming",
    "compute_cardiac_motion",
    "compute_excessive_noise",
    "compute_incomplete_anatomy",
    "compute_misregistration",
    "compute_poor_contrast",
    "compute_respiratory_motion",
    "compute_stair_step",
    "excessive_noise",
    "evaluate_all_families",
    "evaluate_family",
    "frame_axes",
    "incomplete_anatomy",
    "lower_quantile",
    "mad",
    "median",
    "misregistration",
    "normalized_mask",
    "poor_contrast",
    "respiratory_motion",
    "robust_range",
    "run_beam_hardening",
    "run_blooming",
    "run_cardiac_motion",
    "run_excessive_noise",
    "run_incomplete_anatomy",
    "run_misregistration",
    "run_poor_contrast",
    "run_respiratory_motion",
    "run_stair_step",
    "stair_step",
    "validate_qc_input",
    "world_center",
]
