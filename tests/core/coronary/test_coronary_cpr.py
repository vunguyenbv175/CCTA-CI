"""T-2/T-3: CPR determinism, round-trips, and the framing contract.

* rebuild twice = byte-identical pixel blob (content-addressed, so the same
  bytes also give the same blob sha);
* world->voxel->world round-trip <= 1e-6 mm for every stored frame origin and
  for reconstructed cross-section samples;
* ``s_mm`` monotonic (centerline payload) and the CPR row grid uniform;
* the frame sequence lies exactly on the sealed centerline polyline;
* a centerline from ANOTHER source volume is refused (no silent cross-volume
  fallback).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from apps.workstation.volume_loader import CanonicalVolumeLoader
from core.coordinates import FrameSpec, invert_matrix, transform_for_frame
from core.enums import ReleaseLane, VolumeKind
from core.volume import VolumeBlobRef, create_volume
from tests.core.coronary.support import (
    FRAME_OF_REFERENCE_UID,
    SERIES_INSTANCE_UID,
    STUDY_INSTANCE_UID,
    ULIDS,
    SliceFixture,
    as_float,
    as_int,
    segment_distances,
)
from visualization.cpr import (
    CPRSessionIntegrationRefusal,
    CPRValidationError,
    build_cpr_view,
    decode_cpr_pixel_blob,
    require_ws02_snapshot_admission,
    validate_cpr_payload,
)
from visualization.mpr import MPRValidationError, validate_mpr_payload

_REBUILD_PROVENANCE_ID = "01HZZZZZZZZZZZZZZZZZZZZZ11"
_REBUILD_ARTIFACT_ID = "01HZZZZZZZZZZZZZZZZZZZZZ12"
_SEALED_PROVENANCE_ID = "01HZZZZZZZZZZZZZZZZZZZZZ17"
_SEALED_ARTIFACT_ID = "01HZZZZZZZZZZZZZZZZZZZZZ18"


def _matrix_apply(matrix: tuple[tuple[float, ...], ...], vector: tuple[float, float, float]):
    return tuple(
        matrix[row][0] * vector[0]
        + matrix[row][1] * vector[1]
        + matrix[row][2] * vector[2]
        + matrix[row][3]
        for row in range(3)
    )


def test_cpr_rebuild_is_byte_identical(slice_fixture: SliceFixture) -> None:
    rebuilt = build_cpr_view(
        slice_fixture.volume,
        slice_fixture.transform,
        centerline=slice_fixture.centerline,
        s_spacing_mm=2.0,
        radial_spacing_mm=1.0,
        radial_count=5,
        method="TRILINEAR",
        out_of_bounds_fill_value=0.0,
        loader=slice_fixture.loader,
        provenance_ref=f"provenance:{_REBUILD_PROVENANCE_ID}",
        transform_artifact_ref=f"coordinate_transform:{ULIDS['transform_artifact']}",
        study_instance_uid=STUDY_INSTANCE_UID,
        series_instance_uid=SERIES_INSTANCE_UID,
        frame_of_reference_uid=FRAME_OF_REFERENCE_UID,
        artifact_id=_REBUILD_ARTIFACT_ID,
        created_by="ccta-cpr-fixture",
    )
    assert rebuilt.pixels.tobytes(order="C") == slice_fixture.cpr.pixels.tobytes(
        order="C"
    )
    assert rebuilt.pixel_blob_sha256 == slice_fixture.cpr.pixel_blob_sha256
    # The sampling digest binds the pixels + framing + sampling parameters; the
    # artifact's provenance ref is the only field that legitimately differs.
    assert (
        rebuilt.payload["sampling_parameters_sha256"]
        == slice_fixture.cpr.payload["sampling_parameters_sha256"]
    )
    assert rebuilt.payload["frame_sequence"] == slice_fixture.cpr.payload["frame_sequence"]


def test_cpr_payload_decodes_and_binds_its_pixels(slice_fixture: SliceFixture) -> None:
    result = slice_fixture.cpr
    assert validate_cpr_payload(result.payload) == result.payload
    decoded = decode_cpr_pixel_blob(
        result.payload, result.pixels.tobytes(order="C")
    )
    assert np.array_equal(decoded, result.pixels)
    assert result.payload["output_dtype"] == "float32"
    assert result.pixel_blob_ref["codec"] == "float32-le"
    assert as_int(result.pixel_blob_ref["byte_size"]) == result.pixels.nbytes
    assert result.matrix_size == (50, 5)


def test_cpr_round_trips_world_to_voxel_to_world(slice_fixture: SliceFixture) -> None:
    frame = slice_fixture.frame
    voxel_to_world = frame.voxel_to_world_matrix()
    world_to_voxel = invert_matrix(voxel_to_world)
    frames = slice_fixture.cpr.payload["frame_sequence"]
    assert isinstance(frames, list) and len(frames) == 50
    radial_count = as_int(slice_fixture.cpr.payload["radial_count"])
    radial_spacing = as_float(slice_fixture.cpr.payload["radial_spacing_mm"])
    rows = (0, len(frames) // 2, len(frames) - 1)
    columns = (0, (radial_count - 1) // 2, radial_count - 1)
    for row in rows:
        stored = frames[row]
        assert isinstance(stored, dict)
        origin = tuple(float(value) for value in stored["origin_world_mm"])
        binormal = tuple(float(value) for value in stored["binormal"])
        for column in columns:
            shift = (column - (radial_count - 1) / 2.0) * radial_spacing
            sample = (
                origin[0] + shift * binormal[0],
                origin[1] + shift * binormal[1],
                origin[2] + shift * binormal[2],
            )
            voxel = _matrix_apply(world_to_voxel, sample)
            back = _matrix_apply(voxel_to_world, (voxel[0], voxel[1], voxel[2]))
            error = max(abs(a - b) for a, b in zip(back, sample))
            assert error <= 1e-6, f"round-trip error {error} > 1e-6 mm"


def test_cpr_frame_sequence_is_uniform_and_monotonic(slice_fixture: SliceFixture) -> None:
    payload = slice_fixture.cpr.payload
    spacing = as_float(payload["s_spacing_mm"])
    frames = payload["frame_sequence"]
    assert isinstance(frames, list)
    previous = -1.0
    for index, frame in enumerate(frames):
        assert isinstance(frame, dict)
        s_value = as_float(frame["s_mm"])
        assert abs(s_value - index * spacing) <= 1e-9
        assert s_value > previous
        previous = s_value
    last = frames[-1]
    assert isinstance(last, dict)
    centerline_arcs = np.asarray(slice_fixture.centerline.payload["s_mm"], dtype=np.float64)
    assert bool(np.all(np.diff(centerline_arcs) >= 0.0))
    assert float(centerline_arcs[0]) == 0.0
    assert as_float(last["s_mm"]) <= float(centerline_arcs[-1])


def test_cpr_frames_lie_on_the_sealed_polyline(slice_fixture: SliceFixture) -> None:
    points = np.asarray(slice_fixture.centerline.payload["points_world"], dtype=np.float64)
    frames = slice_fixture.cpr.payload["frame_sequence"]
    assert isinstance(frames, list)
    origins_list = []
    for frame in frames:
        assert isinstance(frame, dict)
        origins_list.append(frame["origin_world_mm"])
    origins = np.asarray(origins_list, dtype=np.float64)
    distances = segment_distances(origins, points)
    assert float(distances.max()) <= 1e-9


def test_cpr_center_samples_are_inside_the_lumen(slice_fixture: SliceFixture) -> None:
    pixels = slice_fixture.cpr.pixels
    center_column = (pixels.shape[1] - 1) // 2
    bright = int(np.count_nonzero(pixels[:, center_column] == 300.0))
    assert bright >= 40, f"only {bright}/50 center samples hit the lumen (300 HU)"
    assert float(pixels.min()) >= -100.0
    assert float(pixels.max()) <= 300.0


def test_cpr_refuses_a_centerline_from_another_volume(
    tmp_path: Path, slice_fixture: SliceFixture
) -> None:
    loader = CanonicalVolumeLoader(tmp_path / "other-blobs")
    raw = np.full((8, 8, 8), 300, dtype=np.int16)
    materialized = loader.materialize_canonical(raw, scalar_domain="HU", intensity_unit="HU")
    frame = FrameSpec(
        frame_of_reference_uid="1.2.826.0.1.3680043.8.498.77.920",
        image_position_patient=(0.0, 0.0, 0.0),
        image_orientation_patient=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        spacing_mm=(1.0, 1.0, 1.0),
        dimensions=(8, 8, 8),
    )
    other_volume = create_volume(
        study_ref="study_id:01HZZZZZZZZZZZZZZZZZZZZZ30",
        source_series_refs=("series:01HZZZZZZZZZZZZZZZZZZZZZ31",),
        kind=VolumeKind.CANONICAL,
        coordinate_frame=frame,
        blob_ref=VolumeBlobRef(
            sha256=materialized.blob.sha256,
            codec="int16-le",
            byte_size=materialized.blob.byte_size,
        ),
        provenance_ref="provenance:01HZZZZZZZZZZZZZZZZZZZZZ32",
        release_lane=ReleaseLane.PERSONAL_RESEARCH,
        volume_id="01HZZZZZZZZZZZZZZZZZZZZZ33",
    )
    other_transform = transform_for_frame(
        frame, series_instance_uid="1.2.826.0.1.3680043.8.498.77.921"
    )
    with pytest.raises(CPRValidationError, match="same source volume"):
        build_cpr_view(
            other_volume,
            other_transform,
            centerline=slice_fixture.centerline,
            s_spacing_mm=2.0,
            radial_spacing_mm=1.0,
            radial_count=5,
            method="TRILINEAR",
            out_of_bounds_fill_value=0.0,
            loader=loader,
            provenance_ref="provenance:01HZZZZZZZZZZZZZZZZZZZZZ34",
            transform_artifact_ref="coordinate_transform:01HZZZZZZZZZZZZZZZZZZZZZ35",
            study_instance_uid=STUDY_INSTANCE_UID,
            artifact_id="01HZZZZZZZZZZZZZZZZZZZZZ36",
            created_by="ccta-cpr-fixture",
        )


# ---------------------------------------------------------------------------
# att0002 corrections: F1 (sealed-path framing) / F6 (WS-02 abstention)
# ---------------------------------------------------------------------------


def test_cpr_frames_from_the_sealed_payload_not_the_live_path(
    slice_fixture: SliceFixture,
) -> None:
    """F1: shifting the live ``CoronaryPath`` points cannot move a CPR frame.

    The probe replaces only ``path.points_world`` with +10 mm shifted points
    while the sealed CENTERLINE payload digest stays unchanged: the rebuilt
    view must remain byte-identical because framing reads the sealed payload.
    """

    shifted = tuple(
        (float(x) + 10.0, float(y) + 10.0, float(z) + 10.0)
        for x, y, z in slice_fixture.centerline.path.points_world
    )
    tampered = replace(
        slice_fixture.centerline,
        path=replace(slice_fixture.centerline.path, points_world=shifted),
    )
    assert (
        tampered.path.points_world != slice_fixture.centerline.path.points_world
    ), "the tampered live path must actually differ"
    rebuilt = build_cpr_view(
        slice_fixture.volume,
        slice_fixture.transform,
        centerline=tampered,
        s_spacing_mm=2.0,
        radial_spacing_mm=1.0,
        radial_count=5,
        method="TRILINEAR",
        out_of_bounds_fill_value=0.0,
        loader=slice_fixture.loader,
        provenance_ref=f"provenance:{_SEALED_PROVENANCE_ID}",
        transform_artifact_ref=f"coordinate_transform:{ULIDS['transform_artifact']}",
        study_instance_uid=STUDY_INSTANCE_UID,
        series_instance_uid=SERIES_INSTANCE_UID,
        frame_of_reference_uid=FRAME_OF_REFERENCE_UID,
        artifact_id=_SEALED_ARTIFACT_ID,
        created_by="ccta-cpr-fixture",
    )
    assert rebuilt.payload["centerline_payload_sha256"] == (
        slice_fixture.centerline.artifact.payload.blob_sha256
    )
    assert (
        rebuilt.pixels.tobytes(order="C")
        == slice_fixture.cpr.pixels.tobytes(order="C")
    )
    assert rebuilt.payload["frame_sequence"] == slice_fixture.cpr.payload["frame_sequence"]


def test_cpr_ws02_persistence_route_abstains_with_a_typed_refusal(
    slice_fixture: SliceFixture,
) -> None:
    """F6 (RETIRED by CORONARY_REVIEW_VIEW_V1 Decision B): the legacy WS-02
    abstention gate survives ONLY as a retired pointer at the new seam.

    The CPR view is complete and decodable; the governed store amendment now
    admits ``ccta.cpr.view.v1`` through ``DerivedStateStoreManager`` (see
    ``test_cpr_view_persists_through_the_session_store`` in
    ``test_coronary_review_view.py``).  The retired gate still raises its
    typed refusal so historical callers fail visibly, and the MPR validator
    still refuses the CPR schema (the MPR path was not widened by fallback).
    """

    result = slice_fixture.cpr
    assert validate_cpr_payload(result.payload) == result.payload
    assert (
        decode_cpr_pixel_blob(result.payload, result.pixels.tobytes(order="C")).shape
        == result.matrix_size
    )
    with pytest.raises(CPRSessionIntegrationRefusal) as excinfo:
        require_ws02_snapshot_admission()
    assert excinfo.value.code == "CPR_SESSION_INTEGRATION_UNAVAILABLE"
    assert "retired" in excinfo.value.detail
    # The MPR validator still refuses the CPR schema: the store admits CPR
    # ONLY via the dedicated ccta.cpr.view.v1 branch (no schema fallback).
    with pytest.raises(MPRValidationError):
        validate_mpr_payload(result.payload)
