"""Independent VOLUME-001/MPR-001 known-answer checks.

The expected values in this file are generated from analytic index functions
and a small ``struct`` binary32 oracle.  They never call the MPR implementation
to obtain expected pixels.
"""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import replace
from itertools import product
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from apps.workstation.round_trip import CHECK_MPR, RoundTripVerificationEngine
from apps.workstation.session_store import (
    DerivedStateStoreManager,
    SessionSnapshot,
    WorkstationPaths,
)
from apps.workstation.volume_loader import (
    CANONICAL_FLOAT32_CODEC,
    CANONICAL_INT16_CODEC,
    CanonicalVolumeLoader,
    VolumeAbstentionError,
)
from core.coordinates import FrameSpec, assert_round_trip, transform_for_frame
from core.dicom.qc import QCGateDecision, QCReason, create_qc_assessment, qc_consumer_gate
from core.enums import QualityGrade, ReleaseLane, VolumeKind
from core.events import EventLog
from core.ids import new_ulid
from core.volume import Volume, create_volume
from tests.core.factories import make_study
from visualization.mpr import (
    build_mpr,
    sampling_parameters_sha256,
    validate_mpr_payload,
)

VOLUME_STRATA = tuple(
    product(
        ("axis-aligned", "oblique"),
        ("isotropic", "anisotropic"),
        ("standard", "reversed"),
        ("zero", "nonzero"),
        ("unsigned", "signed", "rescaled-fractional"),
    )
)
MPR_STRATA = tuple(
    product(
        VOLUME_STRATA,
        ("AXIAL", "CORONAL", "SAGITTAL"),
        ("NEAREST", "TRILINEAR"),
    )
)


def _f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def _mul(left: float, right: float) -> float:
    return _f32(_f32(left) * _f32(right))


def _add(left: float, right: float) -> float:
    return _f32(_f32(left) + _f32(right))


def _geometry(
    orientation_kind: str,
    spacing_kind: str,
    normal_kind: str,
    origin_kind: str,
) -> FrameSpec:
    row: tuple[float, float, float] = (
        (1.0, 0.0, 0.0)
        if orientation_kind == "axis-aligned"
        else (math.sqrt(3.0) / 2.0, 0.0, 0.5)
    )
    column = (0.0, 1.0, 0.0)
    if normal_kind == "reversed":
        row = cast(tuple[float, float, float], tuple(-value for value in row))
    orientation = row + column
    spacing = (1.0, 1.0, 1.0) if spacing_kind == "isotropic" else (0.7, 1.3, 0.9)
    origin = (0.0, 0.0, 0.0) if origin_kind == "zero" else (11.0, -7.0, 3.5)
    return FrameSpec(
        frame_of_reference_uid="1.2.826.0.1.3680043.8.498.77.9",
        image_position_patient=origin,
        image_orientation_patient=orientation,
        spacing_mm=spacing,
        dimensions=(3, 4, 5),
    )


def _raw_values(pixel_kind: str) -> tuple[np.ndarray, str, str, float, float, str | None]:
    values = np.empty(
        (3, 4, 5), dtype=np.int16 if pixel_kind == "signed" else np.uint16
    )
    for k in range(3):
        for j in range(4):
            for i in range(5):
                values[k, j, i] = (
                    100 + 3 * i + 5 * j + 7 * k
                    if pixel_kind != "signed"
                    else -100 + 3 * i + 5 * j + 7 * k
                )
    if pixel_kind == "rescaled-fractional":
        raw = values.astype(np.uint16)
        return raw, "HU", "HU", 0.5, -20.25, "HU"
    return values, "STORED", "STORED", 1.0, 0.0, None


def _analytic_volume(pixel_kind: str) -> np.ndarray:
    raw, _domain, _unit, slope, intercept, _type = _raw_values(pixel_kind)
    expected = np.empty(raw.shape, dtype=np.float64)
    for k in range(raw.shape[0]):
        for j in range(raw.shape[1]):
            for i in range(raw.shape[2]):
                expected[k, j, i] = float(raw[k, j, i]) * slope + intercept
    return expected


@pytest.mark.parametrize("stratum", VOLUME_STRATA)
def test_volume_001_known_answer_grid(tmp_path: Path, stratum: tuple[str, ...]) -> None:
    orientation, spacing, normal, origin, pixel_kind = stratum
    loader = CanonicalVolumeLoader(tmp_path / "blobs")
    raw, domain, unit, slope, intercept, rescale_type = _raw_values(pixel_kind)
    materialized = loader.materialize_canonical(
        raw,
        scalar_domain=domain,
        intensity_unit=unit,
        rescale_slope=slope,
        rescale_intercept=intercept,
        rescale_type=rescale_type,
    )
    expected = _analytic_volume(pixel_kind)
    if pixel_kind == "rescaled-fractional":
        actual = materialized.array.astype(np.float64)
        metric = np.abs(actual - expected) / np.maximum(np.abs(expected), 1.0)
        assert float(np.max(metric)) <= 1e-6
        assert materialized.envelope["codec"] == CANONICAL_FLOAT32_CODEC
    else:
        assert np.array_equal(materialized.array, expected.astype(np.int16))
        assert materialized.envelope["codec"] == CANONICAL_INT16_CODEC
    assert materialized.envelope["scalar_domain"] == domain
    assert materialized.envelope["intensity_unit"] == unit
    assert (
        hashlib.sha256(loader.read_blob(materialized.blob.sha256)).hexdigest()
        == materialized.blob.sha256
    )
    assert np.array_equal(
        loader.read_array(materialized.blob.sha256), materialized.array
    )

    frame = _geometry(orientation, spacing, normal, origin)
    transform = transform_for_frame(frame, series_instance_uid="1.2.826.0.1.3680043.8.498.77.2.1")
    assert_round_trip(transform, transform.inverse)
    assert frame.voxel_to_world_matrix() == transform.matrix


def test_volume_scalar_negative_fixtures_fail_closed(tmp_path: Path) -> None:
    loader = CanonicalVolumeLoader(tmp_path / "blobs")
    raw = np.ones((1, 1, 1), dtype=np.uint16)
    with pytest.raises(VolumeAbstentionError, match="SCALAR_DOMAIN_INVALID"):
        loader.materialize_canonical(raw, scalar_domain="UNKNOWN")
    with pytest.raises(VolumeAbstentionError, match="MODALITY_LUT_UNSUPPORTED"):
        loader.materialize_canonical(raw, modality_lut_present=True)
    with pytest.raises(VolumeAbstentionError, match="PIXEL_PADDING_UNSUPPORTED"):
        loader.materialize_canonical(raw, pixel_padding_value=0)
    with pytest.raises(VolumeAbstentionError, match="RESCALE_TYPE_UNSUPPORTED"):
        loader.materialize_canonical(raw, scalar_domain="HU", rescale_type="OD")
    with pytest.raises(VolumeAbstentionError, match="SCALAR_OVERFLOW"):
        loader.materialize_canonical(raw, scalar_domain="HU", rescale_slope=1e40)


def test_volume_stack_scalar_metadata_mismatch_is_series_abstention(tmp_path: Path) -> None:
    loader = CanonicalVolumeLoader(tmp_path / "blobs")
    first = SimpleNamespace(
        rows=1,
        columns=1,
        pixel_representation=0,
        rescale_slope=1.0,
        rescale_intercept=0.0,
        rescale_type=None,
        pixel_data=b"\x01\x00",
    )
    second = SimpleNamespace(**{**vars(first), "rescale_slope": 2.0})
    with pytest.raises(VolumeAbstentionError, match="SCALAR_METADATA_MISMATCH"):
        loader.materialize_dicom_stack((first, second))


def _source_volume(
    loader: CanonicalVolumeLoader,
    frame: FrameSpec,
    pixel_kind: str,
) -> tuple[Volume, np.ndarray]:
    raw, domain, unit, slope, intercept, rescale_type = _raw_values(pixel_kind)
    materialized = loader.materialize_canonical(
        raw,
        scalar_domain=domain,
        intensity_unit=unit,
        rescale_slope=slope,
        rescale_intercept=intercept,
        rescale_type=rescale_type,
    )
    volume = create_volume(
        study_ref="study-mpr-001",
        source_series_refs=("series-mpr-001",),
        kind=VolumeKind.CANONICAL,
        coordinate_frame=frame,
        blob_ref=materialized.volume_blob_ref,
        provenance_ref=new_ulid(),
        release_lane=ReleaseLane.PERSONAL_RESEARCH,
        volume_id=new_ulid(),
    )
    return volume, materialized.array


def _plane_table(plane: str) -> tuple[tuple[float, float, float], ...]:
    return {
        "AXIAL": ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        "CORONAL": ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        "SAGITTAL": ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    }[plane]


def _oracle_center(
    origin: tuple[float, float, float],
    row_axis: tuple[float, float, float],
    column_axis: tuple[float, float, float],
    row_spacing: float,
    column_spacing: float,
    row: int,
    column: int,
) -> tuple[float, float, float]:
    row_distance = _mul(float(row), row_spacing)
    column_distance = _mul(float(column), column_spacing)
    result: list[float] = []
    for axis in range(3):
        result.append(
            _add(
                _add(origin[axis], _mul(row_distance, row_axis[axis])),
                _mul(column_distance, column_axis[axis]),
            )
        )
    return result[0], result[1], result[2]


def _oracle_q(
    matrix: tuple[tuple[float, ...], ...], world: tuple[float, float, float]
) -> tuple[float, float, float]:
    values: list[float] = []
    for row in range(3):
        first = _add(_mul(matrix[row][0], world[0]), _mul(matrix[row][1], world[1]))
        second = _add(first, _mul(matrix[row][2], world[2]))
        values.append(_add(second, matrix[row][3]))
    return values[0], values[1], values[2]


def _oracle_lerp(a: float, b: float, t: float) -> float:
    return _add(_mul(a, _f32(1.0 - t)), _mul(b, t))


def _oracle_pixels(
    source: np.ndarray,
    matrix: tuple[tuple[float, ...], ...],
    plane: str,
    origin: tuple[float, float, float],
    in_plane_spacing_mm: tuple[float, float],
    method: str,
    fill: float,
) -> np.ndarray:
    _normal, row_axis, column_axis = _plane_table(plane)
    output = np.empty((2, 3), dtype=np.float32)
    dims = (source.shape[2], source.shape[1], source.shape[0])
    for row in range(2):
        for column in range(3):
            q = _oracle_q(
                matrix,
                _oracle_center(
                    origin,
                    row_axis,
                    column_axis,
                    in_plane_spacing_mm[0],
                    in_plane_spacing_mm[1],
                    row,
                    column,
                ),
            )
            if any(q[axis] < 0.0 or q[axis] > dims[axis] - 1 for axis in range(3)):
                output[row, column] = _f32(fill)
                continue
            if method == "NEAREST":
                indices = tuple(math.floor(float(value) + 0.5) for value in q)
                output[row, column] = _f32(source[indices[2], indices[1], indices[0]])
                continue
            lower = tuple(math.floor(float(value)) for value in q)
            upper = tuple(min(lower[axis] + 1, dims[axis] - 1) for axis in range(3))
            t = tuple(_f32(q[axis] - lower[axis]) for axis in range(3))
            i00 = _oracle_lerp(
                source[lower[2], lower[1], lower[0]],
                source[lower[2], lower[1], upper[0]],
                t[0],
            )
            i10 = _oracle_lerp(
                source[lower[2], upper[1], lower[0]],
                source[lower[2], upper[1], upper[0]],
                t[0],
            )
            i01 = _oracle_lerp(
                source[upper[2], lower[1], lower[0]],
                source[upper[2], lower[1], upper[0]],
                t[0],
            )
            i11 = _oracle_lerp(
                source[upper[2], upper[1], lower[0]],
                source[upper[2], upper[1], upper[0]],
                t[0],
            )
            j0 = _oracle_lerp(i00, i10, t[1])
            j1 = _oracle_lerp(i01, i11, t[1])
            output[row, column] = _oracle_lerp(j0, j1, t[2])
    return output


@pytest.mark.parametrize("stratum", MPR_STRATA)
def test_mpr_001_known_answer_grid(
    tmp_path: Path, stratum: tuple[tuple[str, ...], str, str]
) -> None:
    (orientation, spacing, normal, origin_kind, pixel_kind), plane, method = stratum
    loader = CanonicalVolumeLoader(tmp_path / "blobs")
    frame = _geometry(orientation, spacing, normal, origin_kind)
    volume, source = _source_volume(loader, frame, pixel_kind)
    transform = transform_for_frame(frame, series_instance_uid="1.2.826.0.1.3680043.8.498.77.2.1")
    result = build_mpr(
        volume,
        transform,
        plane=plane,
        origin=frame.image_position_patient,
        in_plane_spacing_mm=(1.0, 1.0),
        matrix_size=(2, 3),
        method=method,
        out_of_bounds_fill_value=-1000.0,
        loader=loader,
        transform_artifact_ref=f"coordinate_transform:{new_ulid()}",
        provenance_ref=f"provenance:{new_ulid()}",
        study_instance_uid="1.2.826.0.1.3680043.8.498.77.1",
        series_instance_uid="1.2.826.0.1.3680043.8.498.77.2.1",
    )
    expected = _oracle_pixels(
        source,
        transform.inverse.matrix,
        plane,
        frame.image_position_patient,
        (1.0, 1.0),
        method,
        -1000.0,
    )
    expected_dtype = (
        "int16"
        if method == "NEAREST" and pixel_kind != "rescaled-fractional"
        else "float32"
    )
    assert result.payload["output_dtype"] == expected_dtype
    if result.payload["output_dtype"] == "int16":
        assert method == "NEAREST"
        assert np.array_equal(result.pixels, expected.astype(np.int16))
    else:
        actual = result.pixels.astype(np.float64)
        metric = np.abs(actual - expected) / np.maximum(np.abs(expected), 1.0)
        assert float(np.max(metric)) <= 1e-6
    assert result.payload["scalar_domain"] == (
        "HU" if pixel_kind == "rescaled-fractional" else "STORED"
    )
    assert validate_mpr_payload(result.payload)["pixel_blob_ref"] == result.pixel_blob_ref
    assert loader.read_blob(result.pixel_blob_sha256) == result.pixels.tobytes(order="C")


def test_mpr_payload_closed_schema_and_dtype_negatives(tmp_path: Path) -> None:
    loader = CanonicalVolumeLoader(tmp_path / "blobs")
    frame = _geometry("axis-aligned", "isotropic", "standard", "zero")
    volume, _source = _source_volume(loader, frame, "unsigned")
    transform = transform_for_frame(frame, series_instance_uid="1.2.3.4")
    result = build_mpr(
        volume,
        transform,
        plane="AXIAL",
        origin=(0.0, 0.0, 0.0),
        in_plane_spacing_mm=(1.0, 1.0),
        matrix_size=(1, 1),
        method="NEAREST",
        out_of_bounds_fill_value=0.0,
        loader=loader,
        transform_artifact_ref=f"coordinate_transform:{new_ulid()}",
        provenance_ref=f"provenance:{new_ulid()}",
        study_instance_uid="1.2.3",
        series_instance_uid="1.2.3.4",
    )
    bad = dict(result.payload)
    bad["unexpected"] = True
    with pytest.raises(Exception, match="unknown fields"):
        validate_mpr_payload(bad)
    bad_dtype = dict(result.payload)
    bad_dtype["method"] = "TRILINEAR"
    bad_dtype["sampling_parameters_sha256"] = sampling_parameters_sha256(bad_dtype)
    with pytest.raises(Exception, match="int16 output is legal only"):
        validate_mpr_payload(bad_dtype)


def test_mpr_session_reopen_preserves_payload_and_pixel_digests(tmp_path: Path) -> None:
    paths = WorkstationPaths.create(tmp_path / "session")
    loader = CanonicalVolumeLoader(paths.blobs)
    frame = _geometry("axis-aligned", "isotropic", "standard", "zero")
    volume, _source = _source_volume(loader, frame, "unsigned")
    transform = transform_for_frame(frame, series_instance_uid="1.2.3.4")
    result = build_mpr(
        volume,
        transform,
        plane="AXIAL",
        origin=(0.0, 0.0, 0.0),
        in_plane_spacing_mm=(1.0, 1.0),
        matrix_size=(2, 3),
        method="NEAREST",
        out_of_bounds_fill_value=-1.0,
        loader=loader,
        transform_artifact_ref=f"coordinate_transform:{new_ulid()}",
        provenance_ref=f"provenance:{new_ulid()}",
        study_instance_uid="1.2.3",
        series_instance_uid="1.2.3.4",
    )
    log = EventLog()
    log.seed(
        entity="derived_artifact",
        entity_id=result.artifact.artifact_id,
        initial_fields={
            "review_state": "PREDICTED",
            "status": "ACTIVE",
            "release_lane": "PERSONAL_RESEARCH",
        },
    )
    study = make_study()
    snapshot = SessionSnapshot(
        study=study,
        series=(),
        volumes=(volume,),
        artifacts=(result.artifact,),
        payload_documents={result.artifact.artifact_id: result.payload},
        provenance=(),
        log=log,
        skipped_instances=(),
    )
    store = DerivedStateStoreManager(paths)
    store.save(snapshot)
    reopened = store.reopen()
    restored = reopened.payload_documents[result.artifact.artifact_id]
    assert hashlib.sha256(
        json_bytes(restored)
    ).hexdigest() == result.artifact.payload.blob_sha256
    assert loader.read_blob(result.pixel_blob_sha256) == (
        paths.blobs
        / "blobs"
        / "sha256"
        / result.pixel_blob_sha256[:2]
        / f"{result.pixel_blob_sha256}.bin"
    ).read_bytes()
    assert RoundTripVerificationEngine(paths.root).verify(reopened).checks[CHECK_MPR].passed


def test_mpr_consumer_qc_seam_is_fail_closed() -> None:
    volume_ref = "volume_id:01ARZ3NDEKTSV4RRFFQ69G5FAW"
    provenance_ref = "provenance:01ARZ3NDEKTSV4RRFFQ69G5FAT"
    assessment = create_qc_assessment(
        subject_refs=("study_id:01ARZ3NDEKTSV4RRFFQ69G5FAV", volume_ref),
        provenance_ref=provenance_ref,
        artifact_ref="qc_artifact:01ARZ3NDEKTSV4RRFFQ69G5FAS",
    )
    assert (
        qc_consumer_gate((volume_ref,), consumer="mpr", assessments=()).reason_code
        == QCReason.QC_NOT_ASSESSED
    )
    assert (
        qc_consumer_gate((volume_ref,), consumer="mpr", assessments=(assessment,)).decision
        == QCGateDecision.PROCEED
    )
    limited = replace(
        assessment,
        payload=replace(
            assessment.payload,
            quality_grade=QualityGrade.LIMITED,
            justification_code="RULE_ENGINE_EVIDENCE",
            evidence_refs=(provenance_ref,),
        ),
    )
    assert (
        qc_consumer_gate((volume_ref,), consumer="mpr", assessments=(limited,)).reason_code
        == QCReason.QC_LIMITED
    )
    not_analyzable = replace(
        assessment,
        payload=replace(
            assessment.payload,
            quality_grade=QualityGrade.NOT_ANALYZABLE,
            justification_code="RULE_ENGINE_EVIDENCE",
            evidence_refs=(provenance_ref,),
        ),
    )
    assert (
        qc_consumer_gate(
            (volume_ref,), consumer="mpr", assessments=(not_analyzable,)
        ).reason_code
        == QCReason.QC_NOT_ANALYZABLE
    )


def json_bytes(value: object) -> bytes:
    import json

    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
