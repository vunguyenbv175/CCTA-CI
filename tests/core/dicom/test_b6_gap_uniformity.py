"""B6 geometry tolerance regression tests (IMAGING_CORE_001_SPEC AMENDMENT V1.2).

Hermetic synthetic fixtures only: small 4x4 images with deterministic noise
pixels and synthetic UIDs. No real patient data is read or referenced.

Covered rule: members are ordered by IPP projected on the slice normal and
projected gaps are uniform iff max(gaps)-min(gaps) <= max(1e-6 mm, 1% of the
median projected gap); the admitted spacing stays gaps[0]. Every other B6
comparison stays absolute.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.dicom import (
    ClassificationReason,
    DicomInstance,
    classify_series,
    read_dicom_instance,
    validate_stack,
)
from tests.apps.fixture_factory import build_instance

_STUDY_UID = "1.2.826.0.1.3680043.8.498.77.100"
_SERIES_UID = "1.2.826.0.1.3680043.8.498.77.101"
_FRAME_UID = "1.2.826.0.1.3680043.8.498.77.109"
_SOP_UID_ROOT = "1.2.826.0.1.3680043.8.498.77.102"

_DEFAULT_ORIENTATION = "1\\0\\0\\0\\1\\0"


def _build_stack(
    tmp_path: Path,
    case: str,
    positions: list[float],
    *,
    thickness: str = "1.0",
    orientations: list[str] | None = None,
    temporal_indices: list[int | None] | None = None,
    pixel_representation: int = 0,
) -> tuple[DicomInstance, ...]:
    """Write one synthetic axial stack and read it back through the adapter."""
    instances: list[DicomInstance] = []
    for index, z in enumerate(positions):
        path = build_instance(
            tmp_path / case / f"slice-{index:04d}.dcm",
            sop_instance_uid=f"{_SOP_UID_ROOT}.{index}",
            series_instance_uid=_SERIES_UID,
            study_instance_uid=_STUDY_UID,
            frame_of_reference_uid=_FRAME_UID,
            position=(0.0, 0.0, z),
            thickness=thickness,
            orientation=(
                orientations[index] if orientations is not None else _DEFAULT_ORIENTATION
            ),
            temporal_position_index=(
                temporal_indices[index] if temporal_indices is not None else None
            ),
            pixel_representation=pixel_representation,
        )
        instances.append(read_dicom_instance(path))
    return tuple(instances)


def test_ideal_uniform_stack_admitted(tmp_path: Path) -> None:
    """Case 1: ideal uniform 0.625-mm stack is admitted with spacing gaps[0]."""
    instances = _build_stack(
        tmp_path,
        "ideal",
        [0.0, 0.625, 1.25, 1.875],
        thickness="0.625",
    )
    result = validate_stack(instances)
    assert result.admitted
    assert result.detail_code is None
    assert result.slice_spacing_mm == pytest.approx(0.625)
    assert [item.image_position_patient[2] for item in result.instances] == [
        0.0,
        0.625,
        1.25,
        1.875,
    ]


def test_ge_quantization_pattern_admitted(tmp_path: Path) -> None:
    """Case 2: 256-slice GE-pattern stack with 0.624/0.625-mm alternating gaps.

    Uses the observed (bits_allocated, bits_stored, high_bit,
    pixel_representation) = (16, 16, 15, 1) tuple. The ~0.001-mm spread
    exceeds the retired absolute 1e-3-mm bound but is within 1% of the
    median projected gap, so the stack is admitted with spacing gaps[0].
    """
    gaps = [0.624 if index % 2 == 0 else 0.625 for index in range(255)]
    positions = [0.0]
    for gap in gaps:
        positions.append(positions[-1] + gap)
    instances = _build_stack(
        tmp_path,
        "ge-pattern",
        positions,
        thickness="0.625",
        pixel_representation=1,
    )
    assert instances[0].bits_allocated == 16
    assert instances[0].bits_stored == 16
    assert instances[0].high_bit == 15
    assert instances[0].pixel_representation == 1
    result = validate_stack(instances)
    assert result.admitted
    assert result.detail_code is None
    assert len(result.instances) == 256
    assert result.slice_spacing_mm == pytest.approx(0.624)


def test_spacing_discontinuity_refused(tmp_path: Path) -> None:
    """Case 3: deliberate ~1.5x nominal-gap discontinuity is SERIES_STACK_INVALID."""
    instances = _build_stack(
        tmp_path,
        "discontinuity",
        [0.0, 0.625, 1.25, 2.1875],
        thickness="0.625",
    )
    result = validate_stack(instances)
    assert not result.admitted
    assert result.reason_code == ClassificationReason.GEOMETRY_INVALID
    assert result.detail_code == ClassificationReason.SERIES_STACK_INVALID


def test_missing_slice_refused(tmp_path: Path) -> None:
    """Case 4: gap exceeding thickness + tolerance is SERIES_INCOMPLETE."""
    instances = _build_stack(tmp_path, "missing", [0.0, 2.0], thickness="1.0")
    result = validate_stack(instances)
    assert not result.admitted
    assert result.reason_code == ClassificationReason.GEOMETRY_INVALID
    assert result.detail_code == ClassificationReason.SERIES_INCOMPLETE


def test_duplicate_position_refused(tmp_path: Path) -> None:
    """Case 5: duplicate projected position is SERIES_STACK_INVALID."""
    instances = _build_stack(tmp_path, "duplicate", [0.0, 0.0], thickness="1.0")
    result = validate_stack(instances)
    assert not result.admitted
    assert result.reason_code == ClassificationReason.GEOMETRY_INVALID
    assert result.detail_code == ClassificationReason.SERIES_STACK_INVALID


def test_tilted_orientation_refused(tmp_path: Path) -> None:
    """Case 6: tilted/inconsistent IOP is SERIES_STACK_INVALID."""
    instances = _build_stack(
        tmp_path,
        "tilt",
        [0.0, 1.0],
        thickness="1.0",
        orientations=[_DEFAULT_ORIENTATION, "1\\0\\0\\0.01\\0.99995\\0"],
    )
    result = validate_stack(instances)
    assert not result.admitted
    assert result.reason_code == ClassificationReason.GEOMETRY_INVALID
    assert result.detail_code == ClassificationReason.SERIES_STACK_INVALID


def test_mixed_phase_duplicate_position_refused(tmp_path: Path) -> None:
    """Case 7: mixed-phase duplicate-position stack is SERIES_STACK_INVALID.

    Differing TemporalPositionIndex values do not rescue a geometric
    duplicate; the classification seam still reports GEOMETRY_INVALID.
    """
    instances = _build_stack(
        tmp_path,
        "mixed-phase-duplicate",
        [0.0, 0.0],
        thickness="1.0",
        temporal_indices=[1, 2],
    )
    result = validate_stack(instances)
    assert not result.admitted
    assert result.reason_code == ClassificationReason.GEOMETRY_INVALID
    assert result.detail_code == ClassificationReason.SERIES_STACK_INVALID
    classified = classify_series(instances)
    assert classified.visible_reason == ClassificationReason.GEOMETRY_INVALID
