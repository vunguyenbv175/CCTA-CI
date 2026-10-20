"""CanonicalVolumeLoader + deterministic measurement tests (WS02 §B2, Rule C)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from apps.workstation.volume_loader import (
    CanonicalVolumeLoader,
    build_frame_spec,
    compute_measurements,
    measurement_parameters_digest,
)
from core.errors import DigestMismatchError


def test_blob_round_trip_is_content_addressed(tmp_path: Path) -> None:
    loader = CanonicalVolumeLoader(tmp_path)
    payload = b"\x01\x02\x03\x04"
    stored = loader.write_blob(payload)
    assert stored.sha256 == hashlib.sha256(payload).hexdigest()
    assert loader.read_blob(stored.sha256) == payload
    assert loader.blob_path(stored.sha256).exists()


def test_tampered_blob_fails_its_content_address(tmp_path: Path) -> None:
    loader = CanonicalVolumeLoader(tmp_path)
    stored = loader.write_blob(b"original-bytes")
    path = loader.blob_path(stored.sha256)
    path.write_bytes(b"tampered-bytes!")
    with pytest.raises(DigestMismatchError):
        loader.read_blob(stored.sha256)


def test_envelope_round_trip(tmp_path: Path) -> None:
    loader = CanonicalVolumeLoader(tmp_path)
    stored = loader.write_blob(b"data")
    loader.write_envelope(stored.sha256, {"codec": "uint16-le", "shape": [1, 2, 2]})
    envelope = loader.read_envelope(stored.sha256)
    assert envelope["shape"] == [1, 2, 2]
    raw = json.loads(loader.envelope_path(stored.sha256).read_text(encoding="utf-8"))
    assert raw["schema"].startswith("ccta.volume.envelope.")


def test_central_box_measurement_known_answer() -> None:
    # shape (2 slices, 3 rows, 3 cols); sequential voxel values 0..17
    shape = (2, 3, 3)
    values = list(range(18))
    data = b"".join(v.to_bytes(2, "little") for v in values)
    document = compute_measurements(data, shape)
    # ROI lo=(0,1,1) hi=(2,2,2): window is k in {0,1}, j=1, i=1 -> voxels {4, 13}
    assert document["voxel_count"] == 2
    assert document["mean_hu"] == 8.5
    assert document["min_hu"] == 4
    assert document["max_hu"] == 13
    assert document["params_sha256"] == measurement_parameters_digest()


def test_wrong_payload_size_is_rejected() -> None:
    with pytest.raises(ValueError):
        compute_measurements(b"\x00", (2, 2, 2))


def test_build_frame_spec_maps_spacing_in_dicom_order() -> None:
    frame = build_frame_spec(
        frame_of_reference_uid="FOR-1",
        image_position_patient=(1.0, 2.0, 3.0),
        image_orientation_patient=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        pixel_spacing_mm=(0.6, 0.4),
        slice_spacing_mm=0.8,
        dimensions=(4, 5, 6),
    )
    # DICOM PixelSpacing is (row, column); FrameSpec spacing is (col, row, slice)
    assert frame.spacing_mm == (0.4, 0.6, 0.8)
    assert frame.dimensions == (4, 5, 6)
