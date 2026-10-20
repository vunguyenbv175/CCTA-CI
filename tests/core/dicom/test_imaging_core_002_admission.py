"""IMAGING_CORE_002_ADMISSION — independent-oracle tests for the 12-bit arm.

Every expectation here is derived from the DICOM standard's pixel-encoding
semantics (PS3.5 container/content; BitsAllocated is the 16-bit word
container, BitsStored the significant bits, HighBit == BitsStored-1) and from
the frozen rescale arithmetic ``HU = stored x slope + intercept``.  No
expected value is read back from the implementation under test: fixtures are
built by explicit byte construction, and the oracle is written out by hand.

The battery:
  (i)   known-answer  stored 0/1/2048/4095 -> HU -1024/-1023/1024/3071
  (ii)  forbidden-value  any of bits 12-15 set -> DICOM_SYNTAX_INVALID
  (iii) mixed 12/16-bit stack -> series-level refusal
  (iv)  population reproduction (LOCAL ONLY, SKIP when absent)
  (v)   signed-12 survives refused; every other tuple refused as before
"""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from apps.workstation.volume_loader import (
    SOURCE_BIT_DEPTH_FIELDS,
    CanonicalVolumeLoader,
)
from core.coordinates import FrameSpec, transform_for_frame
from core.dicom import (
    DicomAbstentionError,
    DicomInstance,
    read_dicom_instance,
    validate_stack,
)
from core.dicom.instance import (
    admitted_bit_depth_tuple,
    stored_padding_bits_are_zero,
)
from core.enums import ReleaseLane, VolumeKind
from core.events import EventLog
from core.ids import new_ulid
from core.volume import create_volume
from tests.apps.fixture_factory import build_instance
from visualization.mpr import build_mpr

STUDY_UID: str = "1.2.826.0.1.3680043.8.498.77.1"
SERIES_UID: str = "1.2.826.0.1.3680043.8.498.77.2.1"
FRAME_UID: str = "1.2.826.0.1.3680043.8.498.77.9"
#: A fixed ULID-shaped id so a rebuilt MPR is byte-comparable (see
#: :func:`_pinned_mpr_frame`): it pins the identity-bearing refs only.
PINNED_ULID: str = "0123456789ABCDEFGHJKMNPQRSTVWXYZ0"[:26]

#: The admitted 12-bit-stored tuple (IMAGING_CORE_001_SPEC Part 4.1).
TWELVE_BIT: dict[str, int] = {
    "bits_allocated": 16,
    "bits_stored": 12,
    "high_bit": 11,
    "pixel_representation": 0,
}
SIXTEEN_BIT: dict[str, int] = {
    "bits_allocated": 16,
    "bits_stored": 16,
    "high_bit": 15,
    "pixel_representation": 0,
}
SIGNED_TWELVE_BIT: dict[str, int] = {**TWELVE_BIT, "pixel_representation": 1}

#: slope 1 / intercept -1024 / RescaleType HU -- the frozen rescale contract.
SLOPE: float = 1.0
INTERCEPT: float = -1024.0


def _geometry(
    orientation_kind: str,
    spacing_kind: str,
    normal_kind: str,
    origin_kind: str,
) -> FrameSpec:
    """The 001-B analytic grid geometry (independent of the implementation)."""
    import math

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
        frame_of_reference_uid=FRAME_UID,
        image_position_patient=origin,
        image_orientation_patient=orientation,
        spacing_mm=spacing,
        dimensions=(3, 4, 5),
    )


def _analytic_unsigned() -> np.ndarray:
    """The 001-B analytic grid (values 100+3i+5j+7k), built here by hand."""
    values = np.empty((3, 4, 5), dtype=np.uint16)
    for k in range(3):
        for j in range(4):
            for i in range(5):
                values[k, j, i] = 100 + 3 * i + 5 * j + 7 * k
    return values


def _materialize_with(
    loader: CanonicalVolumeLoader,
    values: np.ndarray,
    *,
    depth: dict[str, int] | None = None,
) -> object:
    """Materialize *values* as a STORED canonical volume (record optional)."""
    kwargs: dict[str, object] = (
        {
            "source_bits_allocated": depth["bits_allocated"],
            "source_bits_stored": depth["bits_stored"],
            "source_high_bit": depth["high_bit"],
            "source_pixel_representation": depth["pixel_representation"],
        }
        if depth is not None
        else {}
    )
    return loader.materialize_canonical(
        values,
        scalar_domain="STORED",
        intensity_unit="STORED",
        rescale_slope=1.0,
        rescale_intercept=0.0,
        rescale_type=None,
        **cast(dict[str, Any], kwargs),
    )


def _record_free_volume(
    loader: CanonicalVolumeLoader, frame: FrameSpec, values: np.ndarray
) -> Any:
    """A pre-amendment (record-free) canonical Volume over the analytic grid."""
    materialized = cast(Any, _materialize_with(loader, values))
    return create_volume(
        study_ref="study-ic002-r2",
        source_series_refs=("series-ic002-r2",),
        kind=VolumeKind.CANONICAL,
        coordinate_frame=frame,
        blob_ref=materialized.volume_blob_ref,
        provenance_ref=new_ulid(),
        release_lane=ReleaseLane.PERSONAL_RESEARCH,
        volume_id=new_ulid(),
    )


def _mpr_frame(loader: CanonicalVolumeLoader, volume: Any, frame: FrameSpec) -> Any:
    """One NEAREST MPR frame for *volume* (2x3 matrix, the 001-B shape)."""
    transform = transform_for_frame(frame, series_instance_uid=SERIES_UID)
    return build_mpr(
        volume,
        transform,
        plane="AXIAL",
        origin=frame.image_position_patient,
        in_plane_spacing_mm=(1.0, 1.0),
        matrix_size=(2, 3),
        method="NEAREST",
        out_of_bounds_fill_value=-1.0,
        loader=loader,
        transform_artifact_ref=f"coordinate_transform:{new_ulid()}",
        provenance_ref=f"provenance:{new_ulid()}",
        study_instance_uid=STUDY_UID,
        series_instance_uid=SERIES_UID,
    )


def _pinned_mpr_frame(loader: CanonicalVolumeLoader, volume: Any, frame: FrameSpec) -> Any:
    """A BYTE-STABLE MPR frame: fixed refs, so two calls compare exactly.

    ``_mpr_frame`` mints a fresh ULID per call (as the product does); a
    byte-identity assertion across two builds needs the identity-bearing refs
    pinned instead.  Only the refs differ — geometry, sampling and pixels are
    produced by the same frozen code path.
    """
    transform = transform_for_frame(frame, series_instance_uid=SERIES_UID)
    return build_mpr(
        volume,
        transform,
        plane="AXIAL",
        origin=frame.image_position_patient,
        in_plane_spacing_mm=(1.0, 1.0),
        matrix_size=(2, 3),
        method="NEAREST",
        out_of_bounds_fill_value=-1.0,
        loader=loader,
        transform_artifact_ref=f"coordinate_transform:{PINNED_ULID}",
        provenance_ref=f"provenance:{PINNED_ULID}",
        study_instance_uid=STUDY_UID,
        series_instance_uid=SERIES_UID,
    )


def _source_volume(
    loader: CanonicalVolumeLoader,
    frame: FrameSpec,
    pixel_kind: str,
    *,
    source_bit_depth: dict[str, int] | None = None,
) -> tuple[Any, Any]:
    """One canonical volume over the analytic grid (values 100+3i+5j+7k)."""
    values = np.empty((3, 4, 5), dtype=np.int16 if pixel_kind == "signed" else np.uint16)
    for k in range(3):
        for j in range(4):
            for i in range(5):
                values[k, j, i] = (
                    100 + 3 * i + 5 * j + 7 * k
                    if pixel_kind != "signed"
                    else -100 + 3 * i + 5 * j + 7 * k
                )
    depth_kwargs: dict[str, object] = (
        {
            "source_bits_allocated": source_bit_depth["bits_allocated"],
            "source_bits_stored": source_bit_depth["bits_stored"],
            "source_high_bit": source_bit_depth["high_bit"],
            "source_pixel_representation": source_bit_depth["pixel_representation"],
        }
        if source_bit_depth is not None
        else {}
    )
    materialized = loader.materialize_canonical(
        values,
        scalar_domain="STORED",
        intensity_unit="STORED",
        rescale_slope=1.0,
        rescale_intercept=0.0,
        rescale_type=None,
        **cast(dict[str, Any], depth_kwargs),
    )
    volume = create_volume(
        study_ref="study-ic002-fix",
        source_series_refs=("series-ic002-fix",),
        kind=VolumeKind.CANONICAL,
        coordinate_frame=frame,
        blob_ref=materialized.volume_blob_ref,
        provenance_ref=new_ulid(),
        release_lane=ReleaseLane.PERSONAL_RESEARCH,
        volume_id=new_ulid(),
    )
    return volume, materialized.array


def _sop(index: int) -> str:
    return f"1.2.826.0.1.3680043.8.498.77.3.{index}"


def _mutate(instance: DicomInstance, **changes: Any) -> DicomInstance:
    """Dataclass replace with deliberately malformed field values (F7/F5)."""
    return replace(instance, **cast(dict[str, Any], changes))


def _words(values: tuple[int, ...]) -> bytes:
    """Explicit little-endian uint16 byte construction of one frame."""
    return b"".join(struct.pack("<H", int(value) & 0xFFFF) for value in values)


def _build(
    path: Path,
    *,
    index: int = 1,
    words: tuple[int, ...] = (0, 1, 2048, 4095),
    rows: int = 2,
    columns: int = 2,
    position: tuple[float, float, float] = (0.0, 0.0, 0.0),
    series_uid: str = SERIES_UID,
    **kwargs: object,
) -> Path:
    """Write one 12-bit-stored instance; *kwargs* override any field."""
    merged: dict[str, Any] = {**TWELVE_BIT, **cast(dict[str, Any], kwargs)}
    return build_instance(
        path,
        sop_instance_uid=_sop(index),
        series_instance_uid=series_uid,
        study_instance_uid=STUDY_UID,
        frame_of_reference_uid=FRAME_UID,
        position=position,
        rows=rows,
        columns=columns,
        pixel_data=_words(words),
        rescale_slope=SLOPE,
        rescale_intercept=INTERCEPT,
        rescale_type="HU",
        **merged,
    )


# ---------------------------------------------------------------------------
# (i) known-answer: stored -> HU through the governed pipeline
# ---------------------------------------------------------------------------


def test_known_answer_stored_to_hu_is_independent_oracle(tmp_path: Path) -> None:
    """stored 0/1/2048/4095 -> HU -1024/-1023/1024/3071 (slope 1, int -1024).

    The oracle is the DICOM Modality LUT transform computed by hand; it is not
    read from the implementation.
    """
    stored = (0, 1, 2048, 4095)
    expected_hu = tuple(int(value) * SLOPE + INTERCEPT for value in stored)
    assert expected_hu == (-1024, -1023, 1024, 3071)

    instance = read_dicom_instance(_build(tmp_path / "ka.dcm", words=stored))
    assert (instance.bits_allocated, instance.bits_stored, instance.high_bit) == (16, 12, 11)
    assert instance.pixel_representation == 0

    # Independent decode: the 16-bit container holds the stored value verbatim
    # (bits 12-15 are zero), so the low 12 bits ARE the stored sample.
    decoded = np.frombuffer(instance.pixel_data, dtype="<u2")
    assert tuple(int(value) for value in decoded) == stored
    assert np.array_equal(decoded, np.asarray(stored, dtype=np.uint16))

    # The canonical materialization applies the rescale in the HU domain.
    loader = CanonicalVolumeLoader(tmp_path / "blobs")
    materialized = loader.materialize_canonical(
        instance.pixel_data,
        shape=(1, 2, 2),
        pixel_representation=0,
        scalar_domain="HU",
        intensity_unit="HU",
        rescale_slope=SLOPE,
        rescale_intercept=INTERCEPT,
        rescale_type="HU",
        source_bits_allocated=16,
        source_bits_stored=12,
        source_high_bit=11,
        source_pixel_representation=0,
    )
    hu = np.asarray(materialized.array).reshape(-1)
    assert tuple(int(value) for value in hu) == expected_hu
    # int16-le is legal: every value is exactly integral and in int16 range.
    assert materialized.envelope["codec"] == "int16-le"
    assert materialized.envelope["scalar_domain"] == "HU"
    assert materialized.envelope["source_values_exactly_integral"] is True
    # Lossless round trip: HU + 1024 == stored, per pixel.
    assert np.array_equal(hu.astype(np.int32) + 1024, decoded.astype(np.int32))


def test_known_answer_full_12_bit_range_round_trips_losslessly(tmp_path: Path) -> None:
    """Every representable 12-bit stored value survives stored -> HU -> stored."""
    loader = CanonicalVolumeLoader(tmp_path / "blobs")
    stored = np.arange(0, 4096, dtype=np.uint16)
    materialized = loader.materialize_canonical(
        stored.tobytes(),
        shape=(16, 16, 16),
        pixel_representation=0,
        scalar_domain="HU",
        intensity_unit="HU",
        rescale_slope=SLOPE,
        rescale_intercept=INTERCEPT,
        rescale_type="HU",
        source_bits_allocated=16,
        source_bits_stored=12,
        source_high_bit=11,
        source_pixel_representation=0,
    )
    hu = np.asarray(materialized.array).reshape(-1)
    assert hu.min() == -1024 and hu.max() == 3071
    assert np.array_equal(hu.astype(np.int32) + 1024, stored.astype(np.int32))


def test_12_bit_series_is_admitted_end_to_end(tmp_path: Path) -> None:
    """A conformant 12-bit stack is admitted at the reader and the stack gate."""
    paths = [
        _build(tmp_path / f"s{index}.dcm", index=index, position=(0.0, 0.0, float(index)))
        for index in range(3)
    ]
    instances = tuple(read_dicom_instance(path) for path in paths)
    assert len(instances) == 3
    stack = validate_stack(instances)
    assert stack.admitted


# ---------------------------------------------------------------------------
# (ii) forbidden-value: nonconformant padding abstains, never silently masks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "label",
    ("bit12", "bit13", "bit14", "bit15", "all-padding", "0xFFFF"),
)
def test_nonconformant_padding_bits_abstain(tmp_path: Path, label: str) -> None:
    word = {
        "bit12": 0x1000,
        "bit13": 0x2000,
        "bit14": 0x4000,
        "bit15": 0x8000,
        "all-padding": 0xF000,
        "0xFFFF": 0xFFFF,
    }[label]
    # The offending word is the LAST sample: a partial-scan shortcut would
    # still see it, and an early-exit mask would hide it.
    path = _build(tmp_path / f"pad-{label}.dcm", words=(0, 1, 2, word))
    with pytest.raises(DicomAbstentionError) as error:
        read_dicom_instance(path)
    assert error.value.reason_code == "DICOM_SYNTAX_INVALID"
    assert error.value.detail_code in (None, "DICOM_SYNTAX_INVALID")


def test_nonconformant_padding_abstains_at_the_stack_gate(tmp_path: Path) -> None:
    """The DTO-level stack gate refuses padding independently of the reader.

    The reader already refuses a padded FILE (see the case above).  This case
    builds the DTO directly, so a caller that never went through the reader
    still cannot slip a padded member past the stack gate: a partial move of
    the predicate would be a fail-open defect.
    """
    good = read_dicom_instance(_build(tmp_path / "good.dcm", index=1))
    padded = replace(good, sop_instance_uid=_sop(2), pixel_data=_words((0, 1, 2, 0x1000)))
    # F9 (positive control): a single conformant member IS admitted — asserted
    # directly, never through a vacuous tautology.
    assert validate_stack((good,)).admitted
    assert not validate_stack((good, padded)).admitted
    assert not validate_stack((padded,)).admitted


def test_padding_check_is_a_pure_predicate() -> None:
    assert stored_padding_bits_are_zero(_words((0, 4095, 1)), bits_stored=12, high_bit=11)
    assert not stored_padding_bits_are_zero(_words((4096,)), bits_stored=12, high_bit=11)
    assert not stored_padding_bits_are_zero(_words((0x8000,)), bits_stored=12, high_bit=11)
    # Non-12-bit tuples are not subject to the padding rule.
    assert stored_padding_bits_are_zero(_words((0xFFFF,)), bits_stored=16, high_bit=15)
    # An odd byte length is not conformant.
    assert not stored_padding_bits_are_zero(b"\x00\x01\x02", bits_stored=12, high_bit=11)


# ---------------------------------------------------------------------------
# (iii) mixed 12/16-bit stack -> series-level refusal
# ---------------------------------------------------------------------------


def test_mixed_12_and_16_bit_stack_is_refused_at_series_level(tmp_path: Path) -> None:
    twelve = read_dicom_instance(_build(tmp_path / "twelve.dcm", index=1))
    sixteen = read_dicom_instance(
        _build(tmp_path / "sixteen.dcm", index=2, **cast(dict[str, Any], SIXTEEN_BIT))
    )
    from apps.workstation.volume_loader import VolumeAbstentionError

    loader = CanonicalVolumeLoader(tmp_path / "blobs")
    with pytest.raises(VolumeAbstentionError) as error:
        loader.materialize_dicom_stack((twelve, sixteen))
    assert error.value.reason_code == "SERIES_STACK_INVALID"
    assert error.value.detail_code == "SOURCE_BIT_DEPTH_MISMATCH"


def test_mixed_stack_is_refused_by_classification_and_not_importable(
    tmp_path: Path,
) -> None:
    """F1: the classification seam is atomic with the reader and the adapter.

    Every member above is individually conformant, so per-instance syntax
    cannot refuse; the stack-level uniformity rule (Part 4.1) must:
    validate_stack refuses the stack, and classify_series reports
    NOT_ANALYZABLE with the mapping's DICOM_SYNTAX_INVALID detail — never
    IMPORTABLE+UNKNOWN.
    """
    from core.dicom import ClassificationOutcome, ClassificationReason, classify_series

    twelve = read_dicom_instance(_build(tmp_path / "twelve.dcm", index=1))
    sixteen = read_dicom_instance(
        _build(tmp_path / "sixteen.dcm", index=2, **cast(dict[str, Any], SIXTEEN_BIT))
    )
    for order in ((twelve, sixteen), (sixteen, twelve)):
        stack = validate_stack(order)
        assert not stack.admitted
        assert stack.reason_code == ClassificationReason.GEOMETRY_INVALID
        assert stack.detail_code == ClassificationReason.SERIES_STACK_INVALID
        classified = classify_series(order, stack=stack)
        assert classified.outcome == ClassificationOutcome.NOT_ANALYZABLE
        assert "DICOM_SYNTAX_INVALID" in classified.detail_codes


def test_classification_still_admits_uniform_12_bit_series(tmp_path: Path) -> None:
    """Uniformity, not depth, is the rule: a uniform 12-bit series imports."""
    from core.dicom import ClassificationOutcome, classify_series

    instances = tuple(
        read_dicom_instance(
            _build(tmp_path / f"u{index}.dcm", index=index, position=(0.0, 0.0, float(index)))
        )
        for index in range(2)
    )
    stack = validate_stack(instances)
    assert stack.admitted
    classified = classify_series(instances, stack=stack)
    assert classified.outcome == ClassificationOutcome.IMPORTABLE_UNKNOWN


def test_uniform_stacks_materialize_in_both_depths(tmp_path: Path) -> None:
    """Uniformity, not depth, is the rule: both uniform stacks are accepted."""
    loader = CanonicalVolumeLoader(tmp_path / "blobs")
    for label, depth in (("12", TWELVE_BIT), ("16", SIXTEEN_BIT)):
        stack = tuple(
            read_dicom_instance(
                _build(
                    tmp_path / f"{label}-{index}.dcm",
                    index=index,
                    words=(0, 1, 4095, 4095) if label == "12" else (0, 1, 65535, 65535),
                    **cast(dict[str, Any], depth),
                )
            )
            for index in range(2)
        )
        result = loader.materialize_dicom_stack(stack)
        assert result.envelope["source_bits_stored"] == depth["bits_stored"]
        assert result.envelope["source_high_bit"] == depth["high_bit"]
        assert result.envelope["source_bits_allocated"] == 16
        assert result.envelope["source_pixel_representation"] == 0


def test_adapter_refuses_nonconformant_padding_at_series_level(tmp_path: Path) -> None:
    """The stack adapter re-checks padding: a hand-built DTO cannot bypass it."""
    from apps.workstation.volume_loader import VolumeAbstentionError

    good = read_dicom_instance(_build(tmp_path / "ok.dcm", index=1))
    padded = replace(good, sop_instance_uid=_sop(2), pixel_data=_words((0, 1, 2, 0x8000)))
    loader = CanonicalVolumeLoader(tmp_path / "blobs")
    with pytest.raises(VolumeAbstentionError) as error:
        loader.materialize_dicom_stack((good, padded))
    assert error.value.detail_code == "SOURCE_PADDING_BITS_NONCONFORMANT"


# ---------------------------------------------------------------------------
# (v) boundary matrix: the admitted tuple set is exactly the predicate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("bits_allocated", "bits_stored", "high_bit", "pixel_representation"),
    (
        (16, 16, 15, 0),  # 16-bit arm, unsigned
        (16, 16, 15, 1),  # 16-bit arm, signed
        (16, 12, 11, 0),  # 12-bit arm (the amendment)
    ),
)
def test_predicate_admits_exactly_the_two_arms(
    bits_allocated: int,
    bits_stored: int,
    high_bit: int,
    pixel_representation: int,
) -> None:
    assert admitted_bit_depth_tuple(
        bits_allocated, bits_stored, high_bit, pixel_representation
    )


@pytest.mark.parametrize(
    ("label", "bits_allocated", "bits_stored", "high_bit", "pixel_representation"),
    (
        ("signed-12", 16, 12, 11, 1),
        ("12-with-high-bit-15", 16, 12, 15, 0),
        ("12-with-high-bit-12", 16, 12, 12, 0),
        ("8-bit", 16, 8, 7, 0),
        ("10-bit", 16, 10, 9, 0),
        ("14-bit", 16, 14, 13, 0),
        ("allocated-8", 8, 8, 7, 0),
        ("allocated-32", 32, 32, 31, 0),
        ("16-with-high-bit-11", 16, 16, 11, 0),
        ("zero", 0, 0, 0, 0),
        ("none", None, None, None, None),
    ),
)
def test_predicate_refuses_every_other_tuple(
    label: str,
    bits_allocated: int | None,
    bits_stored: int | None,
    high_bit: int | None,
    pixel_representation: int | None,
) -> None:
    del label
    assert not admitted_bit_depth_tuple(
        bits_allocated, bits_stored, high_bit, pixel_representation
    )


def test_signed_12_bit_file_is_still_refused(tmp_path: Path) -> None:
    path = _build(tmp_path / "signed12.dcm", **cast(dict[str, Any], SIGNED_TWELVE_BIT))
    with pytest.raises(DicomAbstentionError) as error:
        read_dicom_instance(path)
    assert error.value.reason_code == "DICOM_SYNTAX_INVALID"


@pytest.mark.parametrize(
    ("label", "kwargs"),
    (
        ("allocated-8", {"bits_allocated": 8}),
        ("samples-3", {"samples_per_pixel": 3}),
        ("photometric-rgb", {"photometric_interpretation": "RGB"}),
        ("multiframe", {"number_of_frames": 2}),
        ("excluded-image-type", {"image_type": ("ORIGINAL", "PRIMARY", "MPR")}),
        ("non-ct-sop", {"sop_class_uid": "1.2.840.10008.5.1.4.1.1.4"}),
    ),
)
def test_previously_refused_families_stay_refused(
    tmp_path: Path, label: str, kwargs: dict[str, object]
) -> None:
    """The amendment widens bit depth only; every other gate is unchanged."""
    del label
    with pytest.raises(DicomAbstentionError):
        read_dicom_instance(_build(tmp_path / "guard.dcm", **cast(dict[str, Any], kwargs)))


def test_admitted_transfer_syntax_and_image_type_contract_hold(tmp_path: Path) -> None:
    """Implicit VR LE is admitted; DERIVED/SECONDARY/MPR/MIP stays refused.

    DICOM_IMPLICIT_VR_LE_ADMISSION_001 widened the admitted transfer-syntax set
    only; the ImageType boundary below is unchanged from the 12-bit amendment.
    """
    instance = read_dicom_instance(
        _build(tmp_path / "implicit.dcm", transfer_syntax_uid="1.2.840.10008.1.2")
    )
    assert instance.transfer_syntax_uid == "1.2.840.10008.1.2"

    with pytest.raises(DicomAbstentionError) as error:
        read_dicom_instance(
            _build(
                tmp_path / "derived.dcm",
                image_type=("DERIVED", "SECONDARY", "MPR", "MIP"),
            )
        )
    assert error.value.reason_code == "DERIVED_IMAGE_UNSUPPORTED"


# ---------------------------------------------------------------------------
# (iv) population reproduction -- LOCAL ONLY, SKIP when the data is absent
# ---------------------------------------------------------------------------

#: The COCA 102-file set is staged OUTSIDE git on the implement host.  No
#: filename, path component, or PHI value from it is ever recorded: this test
#: reports counts, ranges and digests only.
COCA_ROOT = Path("/home/ubuntu/projects/CCTA/data/external/coca/patient")
COCA_PATIENTS: tuple[str, ...] = ("36", "226", "279")
_SKIP_REASON = (
    "COCA population not staged on this host "
    f"({COCA_ROOT}/{{{','.join(COCA_PATIENTS)}}} absent); "
    "local-only reproduction test"
)


def _coca_files() -> tuple[Path, ...]:
    if not COCA_ROOT.is_dir():
        return ()
    found: list[Path] = []
    for patient in COCA_PATIENTS:
        directory = COCA_ROOT / patient
        if not directory.is_dir():
            continue
        found.extend(sorted(path for path in directory.rglob("*.dcm") if path.is_file()))
    return tuple(found)


@pytest.mark.skipif(not _coca_files(), reason=_SKIP_REASON)
def test_coca_population_reproduces_stored_to_hu_losslessly(tmp_path: Path) -> None:
    """LOCAL ONLY: full lossless stored->HU reproduction + unchanged geometry.

    Counts, ranges and digests only.  No filename, path, or PHI value is
    asserted, printed, or committed.
    """
    files = _coca_files()
    assert files, "skip guard"
    loader = CanonicalVolumeLoader(tmp_path / "blobs")
    counts = {"files": 0, "pixels": 0, "mismatches": 0}
    stored_min: int | None = None
    stored_max: int | None = None
    hu_min: int | None = None
    hu_max: int | None = None
    geometry_digest = hashlib.sha256()
    tuples: set[tuple[int, int, int, int]] = set()

    for path in files:
        instance = read_dicom_instance(path)
        assert isinstance(instance, DicomInstance)
        tuples.add(
            (
                instance.bits_allocated,
                instance.bits_stored,
                instance.high_bit,
                instance.pixel_representation,
            )
        )
        stored = np.frombuffer(instance.pixel_data, dtype="<u2")
        materialized = loader.materialize_canonical(
            instance.pixel_data,
            shape=(1, instance.rows, instance.columns),
            pixel_representation=instance.pixel_representation,
            scalar_domain="HU",
            intensity_unit="HU",
            rescale_slope=instance.rescale_slope,
            rescale_intercept=instance.rescale_intercept,
            rescale_type=instance.rescale_type,
            source_bits_allocated=instance.bits_allocated,
            source_bits_stored=instance.bits_stored,
            source_high_bit=instance.high_bit,
            source_pixel_representation=instance.pixel_representation,
        )
        hu = np.asarray(materialized.array).reshape(-1)
        # Independent oracle: HU + 1024 == stored, per pixel (slope 1).
        mismatches = int(np.count_nonzero(hu.astype(np.int64) + 1024 != stored.astype(np.int64)))
        counts["files"] += 1
        counts["pixels"] += int(stored.size)
        counts["mismatches"] += mismatches
        stored_min = int(stored.min()) if stored_min is None else min(stored_min, int(stored.min()))
        stored_max = int(stored.max()) if stored_max is None else max(stored_max, int(stored.max()))
        hu_min = int(hu.min()) if hu_min is None else min(hu_min, int(hu.min()))
        hu_max = int(hu.max()) if hu_max is None else max(hu_max, int(hu.max()))
        # Geometry binding: rows/columns/spacing/position feed one digest.
        geometry_digest.update(
            struct.pack(
                "<II7d",
                int(instance.rows),
                int(instance.columns),
                float(instance.pixel_spacing[0]),
                float(instance.pixel_spacing[1]),
                float(instance.image_position_patient[0]),
                float(instance.image_position_patient[1]),
                float(instance.image_position_patient[2]),
                float(instance.rescale_slope),
                float(instance.rescale_intercept),
            )
        )

    assert counts["mismatches"] == 0, counts
    # BIND the population claim (R1 F8): the reproduction is exact for the
    # FULL staged population — exactly 102 files, exactly 26,738,688 pixels,
    # and the recorded geometry digest.  A one-file (or any partial) subset
    # FAILS here instead of passing vacuously; a host without the staged
    # data SKIPS (the guard above), never passes falsely.
    assert counts["files"] == 102, counts["files"]
    assert counts["pixels"] == 26_738_688, counts["pixels"]
    assert (
        geometry_digest.hexdigest()
        == "70cb48b7579e9119179b979bdfebbaf9a099a1de8199d43d97f1cf3a0009cb83"
    )
    # The bit-depth tuple is uniform across the reproduction set, and every
    # stored value lies in the 12-bit range (bits 12-15 zero) -- this is the
    # population property the amendment's conformance clause rests on.
    assert tuples == {(16, 12, 11, 0)}, sorted(tuples)
    assert stored_min is not None and stored_min >= 0
    assert stored_max is not None and stored_max <= 4095
    assert hu_min is not None and hu_max is not None
    # Report digests/counts only -- never a filename or a PHI value.
    print(
        f"POPULATION: files={counts['files']} pixels={counts['pixels']} "
        f"stored_range=[{stored_min},{stored_max}] hu_range=[{hu_min},{hu_max}] "
        f"geometry_sha256={geometry_digest.hexdigest()} tuples={len(tuples)}"
    )


def test_population_test_skips_cleanly_when_absent() -> None:
    """The skip condition is disclosed: absence SKIPs, it never passes."""
    if _coca_files():
        pytest.skip("population present; the skip branch is not exercised here")
    assert not COCA_ROOT.is_dir() or not any(
        (COCA_ROOT / patient).is_dir() for patient in COCA_PATIENTS
    )
    assert "COCA population not staged" in _SKIP_REASON


# ---------------------------------------------------------------------------
# R1 fix round: provenance integrity (F3), container containment (F5),
# typed DTO fallback (F7)
# ---------------------------------------------------------------------------


def test_envelope_source_tuple_is_never_overwritten_by_equal_pixels(
    tmp_path: Path,
) -> None:
    """F3: equal pixel bytes from different sources cannot corrupt provenance.

    The envelope is keyed by the pixel blob digest alone, so a 16-bit source
    and a 12-bit source CAN produce byte-identical canonical blobs.  The first
    recorded source tuple is immutable for that content address: a differing
    re-materialization is a typed refusal (fail-closed), and the original
    envelope keeps its own record.
    """
    from apps.workstation.volume_loader import VolumeAbstentionError

    loader = CanonicalVolumeLoader(tmp_path / "blobs")
    raw = _words((0, 1, 2048, 4095))
    kwargs: dict[str, object] = {
        "shape": (1, 2, 2),
        "pixel_representation": 0,
        "scalar_domain": "STORED",
        "intensity_unit": "STORED",
    }
    sixteen = loader.materialize_canonical(
        raw,
        source_bits_allocated=16,
        source_bits_stored=16,
        source_high_bit=15,
        source_pixel_representation=0,
        **cast(dict[str, Any], kwargs),
    )
    with pytest.raises(VolumeAbstentionError) as error:
        loader.materialize_canonical(
            raw,
            source_bits_allocated=16,
            source_bits_stored=12,
            source_high_bit=11,
            source_pixel_representation=0,
            **cast(dict[str, Any], kwargs),
        )
    assert error.value.detail_code == "SOURCE_BIT_DEPTH_MISMATCH"
    # The first record survives byte-for-byte; build_mpr reads THIS record.
    envelope = loader.read_envelope(sixteen.blob.sha256)
    assert envelope["source_bits_stored"] == 16
    assert envelope["source_high_bit"] == 15


def test_record_free_envelope_is_never_claimed_by_a_later_record(
    tmp_path: Path,
) -> None:
    """R2 F3-residual: a record-free envelope is FIRST-WRITER-WINS, not upgraded.

    A pre-amendment (record-free, v1) envelope keeps its deliberate absence of
    provenance for that content address.  A later record-bearing
    materialization of EQUAL pixel bytes is a typed refusal — equal pixels do
    not establish the earlier volume's source representation, and overwriting
    it would silently flip that volume's MPR schema (v1 -> v2) and break its
    saved sessions.  A record-free re-materialization of the same bytes writes
    the identical envelope and stays allowed.
    """
    from apps.workstation.volume_loader import VolumeAbstentionError

    loader = CanonicalVolumeLoader(tmp_path / "blobs")
    raw = _words((0, 1, 2048, 4095))
    base: dict[str, object] = {
        "shape": (1, 2, 2),
        "pixel_representation": 0,
        "scalar_domain": "STORED",
        "intensity_unit": "STORED",
    }
    first = loader.materialize_canonical(raw, **cast(dict[str, Any], base))
    assert all(name not in first.envelope for name in SOURCE_BIT_DEPTH_FIELDS)
    # A later source claiming a record for these bytes is REFUSED, and the
    # record-free envelope survives byte-for-byte.
    with pytest.raises(VolumeAbstentionError) as error:
        loader.materialize_canonical(
            raw,
            source_bits_allocated=16,
            source_bits_stored=12,
            source_high_bit=11,
            source_pixel_representation=0,
            **cast(dict[str, Any], base),
        )
    assert error.value.detail_code == "SOURCE_BIT_DEPTH_MISMATCH"
    kept = loader.read_envelope(first.blob.sha256)
    assert all(name not in kept for name in SOURCE_BIT_DEPTH_FIELDS)
    # A record-free repeat is the same content: allowed, no refusal.
    again = loader.materialize_canonical(raw, **cast(dict[str, Any], base))
    assert again.blob.sha256 == first.blob.sha256
    assert all(name not in again.envelope for name in SOURCE_BIT_DEPTH_FIELDS)


def test_record_free_envelope_collision_is_refused_end_to_end(tmp_path: Path) -> None:
    """R2 F3-residual, product-shaped: the legacy volume keeps its v1 binding.

    A record-free (v1) volume with a saved session and a v1 MPR frame is not
    re-labelled when an equal-pixel source carrying a record is materialized
    into the same blob store: the collision is refused, the old volume still
    rebuilds a v1 frame, and the saved session still reopens.
    """
    from apps.workstation.volume_loader import VolumeAbstentionError

    loader = CanonicalVolumeLoader(tmp_path / "blobs")
    values = _analytic_unsigned()
    frame = _geometry("axis-aligned", "isotropic", "standard", "zero")
    volume = _record_free_volume(loader, frame, values)
    first = _mpr_frame(loader, volume, frame)
    assert first.payload["schema_version"] == "ccta.mpr.view.v1"
    payload_before = dict(first.payload)
    pixels_before = first.pixels.tobytes(order="C")

    with pytest.raises(VolumeAbstentionError) as error:
        _materialize_with(
            loader,
            values,
            depth={"bits_allocated": 16, "bits_stored": 12, "high_bit": 11,
                   "pixel_representation": 0},
        )
    assert error.value.detail_code == "SOURCE_BIT_DEPTH_MISMATCH"

    rebuilt = _mpr_frame(CanonicalVolumeLoader(tmp_path / "blobs"), volume, frame)
    assert rebuilt.payload["schema_version"] == "ccta.mpr.view.v1"
    assert "bits_stored" not in rebuilt.payload
    assert rebuilt.pixels.tobytes(order="C") == pixels_before
    assert rebuilt.payload["pixel_blob_ref"] == first.payload["pixel_blob_ref"]
    assert dict(rebuilt.payload) == payload_before or all(
        key in {"provenance_ref", "out_of_plane_ref", "sampling_parameters_sha256"}
        for key in set(rebuilt.payload) | set(payload_before)
        if rebuilt.payload.get(key) != payload_before.get(key)
    )


def test_v2_record_container_bound_is_enforced(tmp_path: Path) -> None:
    """F5: high_bit outside the 16-bit container can never reach a v2 payload.

    bits_allocated=16, bits_stored=12, high_bit=99 satisfies neither the v2
    payload validator nor its envelope mirror; both abstain with the closed
    SOURCE_BIT_DEPTH_INVALID code.
    """
    from visualization.mpr import (
        MPRAbstentionError,
        MPRValidationError,
        validate_mpr_payload,
    )

    frame = _geometry("axis-aligned", "isotropic", "standard", "zero")

    def _build(loader: CanonicalVolumeLoader, volume: Any) -> Any:
        transform = transform_for_frame(frame, series_instance_uid="1.2.3.4")
        return build_mpr(
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

    def _materialize(loader: CanonicalVolumeLoader, high_bit: int) -> Any:
        return _source_volume(
            loader,
            frame,
            "unsigned",
            source_bit_depth={
                "bits_allocated": 16,
                "bits_stored": 12,
                "high_bit": high_bit,
                "pixel_representation": 0,
            },
        )[0]

    # The envelope mirror path: build_mpr reads the envelope by blob digest,
    # and a record carrying high_bit=99 abstains instead of emitting v2.
    mirror_loader = CanonicalVolumeLoader(tmp_path / "mirror-blobs")
    volume = _materialize(mirror_loader, high_bit=99)
    with pytest.raises(MPRAbstentionError) as error:
        _build(mirror_loader, volume)
    assert error.value.reason_code == "MPR_SOURCE_INVALID"
    assert error.value.detail_code == "SOURCE_BIT_DEPTH_INVALID"

    # The v2 payload path: a well-typed record with an in-container high_bit
    # produces a v2 frame, but the payload validator refuses the mutated
    # record (a persisted payload can never carry high_bit=99).
    payload_loader = CanonicalVolumeLoader(tmp_path / "payload-blobs")
    volume2 = _materialize(payload_loader, high_bit=11)
    payload = dict(_build(payload_loader, volume2).payload)
    assert payload["schema_version"] == "ccta.mpr.view.v2"
    assert payload["high_bit"] == 11
    payload["high_bit"] = 99
    with pytest.raises(MPRValidationError, match="container"):
        validate_mpr_payload(payload)


def test_partial_and_malformed_dto_depth_refuse_typed(tmp_path: Path) -> None:
    """F7: the legacy fallback distinguishes absent, partial, and malformed.

    A duck-typed legacy DTO that omits ALL depth fields keeps a record-free
    envelope (compatibility); a PARTIAL set or a malformed value is a typed
    refusal — never a silent drop into a legacy envelope.
    """
    from apps.workstation.volume_loader import VolumeAbstentionError

    loader = CanonicalVolumeLoader(tmp_path / "blobs")

    def _materialize(instances: Any) -> Any:
        return loader.materialize_dicom_stack(instances)

    good = read_dicom_instance(_build(tmp_path / "good.dcm", index=1))

    # (a) uniform absence stays legal (legacy adapter), envelope record-free.
    legacy = SimpleNamespace(
        rows=good.rows,
        columns=good.columns,
        pixel_representation=good.pixel_representation,
        rescale_slope=good.rescale_slope,
        rescale_intercept=good.rescale_intercept,
        rescale_type=good.rescale_type,
        pixel_data=good.pixel_data,
    )
    result = _materialize((legacy,))
    assert all(name not in result.envelope for name in SOURCE_BIT_DEPTH_FIELDS)

    # (b) a partial set on one member -> SOURCE_BIT_DEPTH_INCOMPLETE.
    partial = _mutate(good, sop_instance_uid=_sop(2), high_bit=None)
    with pytest.raises(VolumeAbstentionError) as error:
        _materialize((good, partial))
    assert error.value.detail_code == "SOURCE_BIT_DEPTH_INCOMPLETE"

    # (b2) a recorded member beside a record-free member is also incomplete.
    with pytest.raises(VolumeAbstentionError) as error:
        _materialize((good, legacy))
    assert error.value.detail_code == "SOURCE_BIT_DEPTH_INCOMPLETE"

    # (c) a malformed value -> SOURCE_BIT_DEPTH_INVALID.
    malformed = _mutate(good, sop_instance_uid=_sop(2), high_bit="11")
    with pytest.raises(VolumeAbstentionError) as error:
        _materialize((good, malformed))
    assert error.value.detail_code == "SOURCE_BIT_DEPTH_INVALID"
    malformed_bool = _mutate(good, sop_instance_uid=_sop(2), bits_stored=True)
    with pytest.raises(VolumeAbstentionError) as error:
        _materialize((good, malformed_bool))
    assert error.value.detail_code == "SOURCE_BIT_DEPTH_INVALID"


def test_partial_dto_depth_refuses_even_as_single_member(tmp_path: Path) -> None:
    """The F7 refusals do not depend on mixing with a recorded member."""
    from apps.workstation.volume_loader import VolumeAbstentionError

    loader = CanonicalVolumeLoader(tmp_path / "blobs")
    good = read_dicom_instance(_build(tmp_path / "good.dcm", index=1))
    with pytest.raises(VolumeAbstentionError, match="SOURCE_BIT_DEPTH_INCOMPLETE"):
        loader.materialize_dicom_stack((_mutate(good, high_bit=None),))
    with pytest.raises(VolumeAbstentionError, match="SOURCE_BIT_DEPTH_INVALID"):
        loader.materialize_dicom_stack((_mutate(good, high_bit="11"),))


# ---------------------------------------------------------------------------
# FOLLOW-UP 01 (IC002-R3-R1) -- cross-version canonical-envelope preservation
# under HU derive: an R2 STORED envelope plus a re-imported HU write over
# BYTE-IDENTICAL pixels (identity rescale) must never overwrite the earlier
# binding.
#
# The compared field set is the FULL scalar binding
# (apps.workstation.volume_loader.SCALAR_BINDING_FIELDS):
#     source_bits_allocated, source_bits_stored, source_high_bit,
#     source_pixel_representation,      <- the R1/R2 source tuple
#     scalar_domain, intensity_unit, rescale_applied,
#     rescale_slope, rescale_intercept  <- what the samples MEAN
# with an ABSENT (slope, intercept) pair normalized to the identity pair
# (1.0, 0.0) per the spec, and a bool never treated as an int.  The refusal
# uses the EXISTING closed typed code SCALAR_DOMAIN_INVALID /
# SOURCE_BIT_DEPTH_MISMATCH -- no new reason codes.
# ---------------------------------------------------------------------------

#: An IDENTITY rescale (stored == HU): the STORED and HU canonical blobs are
#: byte-identical, so they collide on one content address.  This is the R3 case.
IDENTITY_SERIES_UID: str = "1.2.826.0.1.3680043.8.498.77.4.1"
ABSENT_SERIES_UID: str = "1.2.826.0.1.3680043.8.498.77.4.2"
IDENTITY_VALUE: int = 300
IDENTITY_SLICES: int = 3
#: The R3 reproduction's own pixel digest (three 4x5 slices of stored 300).
IDENTITY_DIGEST: str = (
    "2ea8cec48182d9f2763f995c59d6b6626c79935b004e1f56530fdb0eb5698ad3"
)


def _identity_series(dest: Path, *, series_uid: str = IDENTITY_SERIES_UID) -> Path:
    """Three Explicit-VR ORIGINAL/PRIMARY 12-bit CT slices, every sample 300.

    Slope 1 / intercept 0 / RescaleType HU: an IDENTITY rescale, so the STORED
    and HU canonical blobs are byte-identical at one content address.
    """
    dest.mkdir(parents=True, exist_ok=True)
    for index in range(IDENTITY_SLICES):
        build_instance(
            dest / f"identity-{index}.dcm",
            sop_instance_uid=f"1.2.826.0.1.3680043.8.498.78.3.{index + 1}",
            series_instance_uid=series_uid,
            study_instance_uid=STUDY_UID,
            frame_of_reference_uid=FRAME_UID,
            position=(0.0, 0.0, float(index)),
            rows=4,
            columns=5,
            pixel_data=_words((IDENTITY_VALUE,) * 20),
            rescale_slope=1.0,
            rescale_intercept=0.0,
            rescale_type="HU",
            **cast(dict[str, Any], TWELVE_BIT),
        )
    return dest


def _identity_stack(dest: Path) -> tuple[Any, ...]:
    """The admitted DTO stack of :func:`_identity_series`."""
    return tuple(
        read_dicom_instance(path)
        for path in sorted(_identity_series(dest).glob("*.dcm"))
    )


def _fresh_session(imported: Any) -> Any:
    """A re-imported series with NO volume attached yet (the collide input)."""
    return SimpleNamespace(
        study=imported.study,
        series=tuple(item.series for item in imported.series),
        volumes=(),
        artifacts=(),
        payload_documents={},
        provenance=imported.provenance,
        log=EventLog(),
        skipped_instances=imported.skipped,
    )


def _import_snapshot(imported: Any) -> Any:
    """A pre-derive session snapshot for one freshly imported study."""
    from apps.workstation.session_store import SessionSnapshot

    return SessionSnapshot(
        study=imported.study,
        series=tuple(item.series for item in imported.series),
        volumes=(),
        artifacts=(),
        payload_documents={},
        provenance=imported.provenance,
        log=EventLog(),
        skipped_instances=imported.skipped,
    )


def test_identity_rescale_cross_version_collision_is_refused_and_session_survives(
    tmp_path: Path,
) -> None:
    """(a) R2 STORED envelope + HU re-import: typed refusal, old session intact.

    The earlier envelope records STORED/STORED with rescale_applied false and
    the tuple (16,12,11,0); the re-imported series derives HU with
    rescale_applied true over BYTE-IDENTICAL pixels.  The guard compares the
    FULL scalar binding (not the tuple alone), refuses with the existing
    closed typed code, and leaves the earlier session's manifest, session
    payload, MPR payload and pixels byte-identical.
    """
    from apps.workstation.session_store import (
        DerivedStateStoreManager,
        SessionSnapshot,
    )
    from apps.workstation.volume_loader import VolumeAbstentionError
    from apps.workstation.workspace import StudyWorkspaceManager

    source = _identity_series(tmp_path / "source")
    frame = _geometry("axis-aligned", "isotropic", "standard", "zero")

    # 1. The R2-era session: import, materialize STORED (what the R2 derive
    #    path selected -- its only difference from now is that domain), save
    #    the session with an MPR bound to that envelope, reopen it.
    manager = StudyWorkspaceManager(tmp_path / "ws")
    imported = manager.import_study(source)
    loader = CanonicalVolumeLoader(imported.paths.blobs)
    r2 = loader.materialize_dicom_stack(
        _identity_stack(tmp_path / "r2source"),
        scalar_domain="STORED",
        intensity_unit="STORED",
    )
    assert r2.blob.sha256 == IDENTITY_DIGEST, "the R3 digest must be under test"
    assert r2.envelope["scalar_domain"] == "STORED"
    assert r2.envelope["intensity_unit"] == "STORED"
    assert r2.envelope["rescale_applied"] is False
    assert tuple(r2.envelope[name] for name in SOURCE_BIT_DEPTH_FIELDS) == (16, 12, 11, 0)

    volume = create_volume(
        study_ref="study-ic002-fu01-r2",
        source_series_refs=("series-ic002-fu01-r2",),
        kind=VolumeKind.CANONICAL,
        coordinate_frame=frame,
        blob_ref=r2.volume_blob_ref,
        provenance_ref=new_ulid(),
        release_lane=ReleaseLane.PERSONAL_RESEARCH,
        volume_id=new_ulid(),
    )
    mpr = _pinned_mpr_frame(loader, volume, frame)
    assert mpr.payload["schema_version"] == "ccta.mpr.view.v2"
    assert mpr.payload["scalar_domain"] == "STORED"

    log = EventLog()
    log.seed(
        entity="derived_artifact",
        entity_id=mpr.artifact.artifact_id,
        initial_fields={
            "review_state": "PREDICTED",
            "status": "ACTIVE",
            "release_lane": "PERSONAL_RESEARCH",
        },
    )
    store = DerivedStateStoreManager(imported.paths)
    store.save(
        SessionSnapshot(
            study=imported.study,
            series=tuple(item.series for item in imported.series),
            volumes=(volume,),
            artifacts=(mpr.artifact,),
            payload_documents={mpr.artifact.artifact_id: mpr.payload},
            provenance=imported.provenance,
            log=log,
            skipped_instances=imported.skipped,
        )
    )
    saved = store.reopen()
    assert saved.volumes[0].blob_ref.sha256 == IDENTITY_DIGEST

    envelope_before = loader.envelope_path(IDENTITY_DIGEST).read_bytes()
    manifest_before = (imported.paths.session / "manifest.json").read_bytes()
    payload_before = json.dumps(mpr.payload, sort_keys=True, separators=(",", ":")).encode()
    pixels_before = mpr.pixels.tobytes(order="C")

    # 2. Re-import the SAME source into the SAME workspace, then derive the
    #    re-imported series: the product derive path selects HU over identical
    #    pixels at the identical content address.
    reimported = manager.import_study(source)
    with pytest.raises(VolumeAbstentionError) as error:
        manager.derive(_fresh_session(reimported))
    assert error.value.reason_code == "SCALAR_DOMAIN_INVALID"
    assert error.value.detail_code == "SOURCE_BIT_DEPTH_MISMATCH"

    # 3. The earlier envelope, session manifest, MPR payload and pixels are all
    #    byte-identical, and the volume still rebuilds the same STORED frame.
    assert loader.envelope_path(IDENTITY_DIGEST).read_bytes() == envelope_before
    assert (imported.paths.session / "manifest.json").read_bytes() == manifest_before
    survivor = store.reopen()
    assert survivor.volumes[0].blob_ref.sha256 == IDENTITY_DIGEST
    assert survivor.payload_documents[mpr.artifact.artifact_id] == mpr.payload
    assert (
        json.dumps(
            survivor.payload_documents[mpr.artifact.artifact_id],
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        == payload_before
    )
    rebuilt = _pinned_mpr_frame(
        CanonicalVolumeLoader(imported.paths.blobs), volume, frame
    )
    assert rebuilt.payload["schema_version"] == "ccta.mpr.view.v2"
    assert rebuilt.payload["scalar_domain"] == "STORED"
    assert rebuilt.pixels.tobytes(order="C") == pixels_before
    assert (
        json.dumps(rebuilt.payload, sort_keys=True, separators=(",", ":")).encode()
        == payload_before
    )


def test_absent_rescale_same_dimension_collision_is_refused(tmp_path: Path) -> None:
    """(b) An ABSENT rescale collides identically (normalization proven).

    The reader records an absent RescaleSlope/Intercept as the identity pair
    (1.0, 0.0), so the absent spelling and the explicit spelling are the SAME
    binding; what collides is the DOMAIN (STORED vs HU) over equal pixels.
    """
    from apps.workstation.volume_loader import VolumeAbstentionError

    dest = tmp_path / "absent-rescale"
    dest.mkdir(parents=True, exist_ok=True)
    for index in range(IDENTITY_SLICES):
        build_instance(
            dest / f"absent-{index}.dcm",
            sop_instance_uid=f"1.2.826.0.1.3680043.8.498.79.3.{index + 1}",
            series_instance_uid=ABSENT_SERIES_UID,
            study_instance_uid=STUDY_UID,
            frame_of_reference_uid=FRAME_UID,
            position=(0.0, 0.0, float(index)),
            rows=4,
            columns=5,
            pixel_data=_words((IDENTITY_VALUE,) * 20),
            **cast(dict[str, Any], TWELVE_BIT),
        )
    stack = tuple(read_dicom_instance(path) for path in sorted(dest.glob("*.dcm")))
    # The reader's normalization: an absent rescale IS the identity pair.
    assert float(cast(Any, stack[0].rescale_slope)) == 1.0
    assert float(cast(Any, stack[0].rescale_intercept)) == 0.0

    loader = CanonicalVolumeLoader(tmp_path / "blobs")
    stored = loader.materialize_dicom_stack(
        stack, scalar_domain="STORED", intensity_unit="STORED"
    )
    assert stored.blob.sha256 == IDENTITY_DIGEST
    before = loader.envelope_path(stored.blob.sha256).read_bytes()
    with pytest.raises(VolumeAbstentionError) as error:
        loader.materialize_dicom_stack(stack, scalar_domain="HU", intensity_unit="HU")
    assert (error.value.reason_code, error.value.detail_code) == (
        "SCALAR_DOMAIN_INVALID",
        "SOURCE_BIT_DEPTH_MISMATCH",
    )
    assert loader.envelope_path(stored.blob.sha256).read_bytes() == before
    kept = loader.read_envelope(stored.blob.sha256)
    assert kept["scalar_domain"] == "STORED"
    assert kept["rescale_applied"] is False
    assert float(cast(Any, kept["rescale_slope"])) == 1.0
    assert float(cast(Any, kept["rescale_intercept"])) == 0.0


def test_same_domain_idempotent_rematerialization_stays_allowed(tmp_path: Path) -> None:
    """(c) The trap: legitimate same-domain idempotence must NOT be refused.

    HU+HU and STORED+STORED re-materialization of the same bytes/tuple/
    metadata is ALLOWED with the digest unchanged.  Proves the extended
    comparison is not over-broad.
    """
    hu_loader = CanonicalVolumeLoader(tmp_path / "blobs-hu")
    hu_stack = _identity_stack(tmp_path / "idem-hu")
    hu = hu_loader.materialize_dicom_stack(
        hu_stack, scalar_domain="HU", intensity_unit="HU"
    )
    again = hu_loader.materialize_dicom_stack(
        hu_stack, scalar_domain="HU", intensity_unit="HU"
    )
    assert again.blob.sha256 == hu.blob.sha256, "HU+HU idempotence must stay allowed"
    assert again.envelope == hu.envelope

    stored_loader = CanonicalVolumeLoader(tmp_path / "blobs-stored")
    stored_stack = _identity_stack(tmp_path / "idem-stored")
    stored = stored_loader.materialize_dicom_stack(
        stored_stack, scalar_domain="STORED", intensity_unit="STORED"
    )
    stored_again = stored_loader.materialize_dicom_stack(
        stored_stack, scalar_domain="STORED", intensity_unit="STORED"
    )
    assert stored_again.blob.sha256 == stored.blob.sha256, (
        "STORED+STORED idempotence must stay allowed"
    )
    assert stored_again.envelope == stored.envelope


def test_rescale_normalization_rules_are_exact(tmp_path: Path) -> None:
    """(c) The normalization rules themselves, enumerated one by one.

    Absent == identity (1.0, 0.0); a bool is never an int; numeric spellings
    compare as binary64; a present-but-unusable pair is unequal, never guessed.
    """
    normalize = CanonicalVolumeLoader._normalize_rescale_pair
    # Absent (both None) is EXACTLY the identity pair (the R2-era spelling).
    assert normalize(None, None) == (1.0, 0.0)
    # Explicit identity, in either numeric spelling.
    assert normalize(1.0, 0.0) == (1.0, 0.0)
    assert normalize(1, 0) == (1.0, 0.0)
    # A bool is never an int.
    assert normalize(True, 0.0) is None
    assert normalize(1.0, False) is None
    # A present-but-unusable pair is unequal rather than guessed (no coercion).
    assert normalize("1.0", 0.0) is None
    assert normalize(float("nan"), 0.0) is None
    assert normalize(float("inf"), 0.0) is None
    assert normalize(1.0, None) is None
    assert normalize(None, 0.0) is None
    # A non-identity pair stays distinct from the identity pair.
    assert normalize(1.0, -1024.0) != (1.0, 0.0)

    values = np.full((3, 4, 5), IDENTITY_VALUE, dtype=np.uint16)
    loader = CanonicalVolumeLoader(tmp_path / "blobs-norm")
    base: dict[str, object] = {
        "scalar_domain": "HU",
        "intensity_unit": "HU",
        "rescale_type": "HU",
    }
    first = loader.materialize_canonical(
        values, rescale_slope=1.0, rescale_intercept=0.0, **cast(dict[str, Any], base)
    )
    # The int spelling of the same pair is the SAME binding (allowed).
    assert (
        loader.materialize_canonical(
            values, rescale_slope=1, rescale_intercept=0, **cast(dict[str, Any], base)
        ).blob.sha256
        == first.blob.sha256
    )


def test_save_and_reopen_are_green_on_both_sides(tmp_path: Path) -> None:
    """(d) Save/reopen green on BOTH sides of the refused collision.

    New side: a FRESH store derives HU, saves and reopens green (and re-derive
    is a no-op).  Old side: an R2-semantics session saved before the collision
    reopens green afterwards, with its manifest byte-identical.
    """
    from apps.workstation.session_store import (
        DerivedStateStoreManager,
        SessionSnapshot,
    )
    from apps.workstation.volume_loader import VolumeAbstentionError
    from apps.workstation.workspace import StudyWorkspaceManager

    # -- new side: a fresh workspace is unaffected by the fix ---------------
    source = _identity_series(tmp_path / "fresh")
    manager = StudyWorkspaceManager(tmp_path / "ws-fresh")
    imported = manager.import_study(source)
    store = DerivedStateStoreManager(imported.paths)
    store.save(_import_snapshot(imported))
    session = store.reopen()
    outcome = manager.derive(session)
    store.save(
        SessionSnapshot(
            study=session.study,
            series=session.series,
            volumes=session.volumes + outcome.volumes,
            artifacts=session.artifacts + outcome.artifacts,
            payload_documents={**session.payload_documents, **outcome.payload_documents},
            provenance=session.provenance + outcome.provenance,
            log=session.log,
            skipped_instances=session.skipped_instances,
        )
    )
    reopened = store.reopen()
    assert len(reopened.volumes) == 1
    envelope = CanonicalVolumeLoader(imported.paths.blobs).read_envelope(
        reopened.volumes[0].blob_ref.sha256
    )
    assert envelope["scalar_domain"] == "HU"
    assert envelope["intensity_unit"] == "HU"
    assert envelope["rescale_applied"] is True
    assert tuple(envelope[name] for name in SOURCE_BIT_DEPTH_FIELDS) == (16, 12, 11, 0)
    # workspace.py:669 skip path: re-derive produces no new volumes.
    repeat = manager.derive(reopened)
    assert len(repeat.volumes) == 0
    assert len(repeat.already_derived_series) == 1
    # The fresh session still reopens and its HU samples survive.
    assert store.reopen().volumes[0].blob_ref.sha256 == reopened.volumes[0].blob_ref.sha256
    samples = np.asarray(
        CanonicalVolumeLoader(imported.paths.blobs).read_array(
            reopened.volumes[0].blob_ref.sha256, envelope=envelope
        )
    )
    assert int(samples[0, 0, 0]) == IDENTITY_VALUE

    # -- old side: the pre-collision session survives the refusal -----------
    old_source = _identity_series(tmp_path / "old")
    old_manager = StudyWorkspaceManager(tmp_path / "ws-old")
    old_imported = old_manager.import_study(old_source)
    old_loader = CanonicalVolumeLoader(old_imported.paths.blobs)
    # The R2-semantics envelope (STORED), as R2's derive would have written it.
    r2 = old_loader.materialize_dicom_stack(
        tuple(read_dicom_instance(path) for path in sorted(old_source.glob("*.dcm"))),
        scalar_domain="STORED",
        intensity_unit="STORED",
    )
    old_store = DerivedStateStoreManager(old_imported.paths)
    old_store.save(_import_snapshot(old_imported))
    session = old_store.reopen()
    old_volume = create_volume(
        study_ref=old_imported.study.study_id,
        source_series_refs=(session.series[0].series_id,),
        kind=VolumeKind.CANONICAL,
        coordinate_frame=_geometry("axis-aligned", "isotropic", "standard", "zero"),
        blob_ref=r2.volume_blob_ref,
        provenance_ref=new_ulid(),
        release_lane=ReleaseLane.PERSONAL_RESEARCH,
        volume_id=new_ulid(),
    )
    old_store.save(
        SessionSnapshot(
            study=session.study,
            series=session.series,
            volumes=(old_volume,),
            artifacts=session.artifacts,
            payload_documents=session.payload_documents,
            provenance=session.provenance,
            log=session.log,
            skipped_instances=session.skipped_instances,
        )
    )
    manifest_before = (old_imported.paths.session / "manifest.json").read_bytes()
    assert old_store.reopen().volumes[0].blob_ref.sha256 == IDENTITY_DIGEST

    with pytest.raises(VolumeAbstentionError) as error:
        old_manager.derive(_fresh_session(old_imported))
    assert (error.value.reason_code, error.value.detail_code) == (
        "SCALAR_DOMAIN_INVALID",
        "SOURCE_BIT_DEPTH_MISMATCH",
    )
    # The OLD session still reopens green, manifest byte-identical.
    assert (old_imported.paths.session / "manifest.json").read_bytes() == manifest_before
    survivor = old_store.reopen()
    assert survivor.volumes[0].blob_ref.sha256 == IDENTITY_DIGEST
    assert old_loader.read_envelope(IDENTITY_DIGEST)["scalar_domain"] == "STORED"


# ---------------------------------------------------------------------------
# R2 fix round F15: the product derive path selects HU (end-to-end oracle)
# ---------------------------------------------------------------------------

#: The F15 oracle fixture: a physical 300 HU sample stored as 1324 under the
#: owner-representative rescale (slope 1, intercept -1024).  The HU value is
#: computed by hand: 1324 * 1 + (-1024) = 300.  No expectation below is read
#: back from the implementation.
F15_STORED: int = 1324
F15_HU: int = F15_STORED * int(SLOPE) + int(INTERCEPT)  # == 300
#: CT_VESSEL_CONTRAST center 300 / width 800 -> lo = -100, hi = 700.  The
#: 8-bit byte for s=300 is floor(f32(255 * t) + 0.5) with t = (300 - lo) /
#: (hi - lo) = 0.5 -> floor(127.5 + 0.5) = 128.  Hand-computed, exact.
F15_WL_CENTER: float = 300.0
F15_WL_WIDTH: float = 800.0
F15_WL_BYTE_HU: int = 128
#: The same preset applied to the STORED value 1324 clips at the high bound
#: (1324 >= 700) -> byte 255.  This is the defect the oracle must exclude.
F15_WL_BYTE_STORED: int = 255


def _f15_series(dest: Path, *, slices: int = 3, rows: int = 4, columns: int = 5) -> Path:
    """Explicit-VR ORIGINAL/PRIMARY 12-bit CT, every sample F15_STORED."""
    dest.mkdir(parents=True, exist_ok=True)
    for index in range(slices):
        _build(
            dest / f"f15-{index}.dcm",
            index=index + 1,
            rows=rows,
            columns=columns,
            position=(0.0, 0.0, float(index)),
            words=(F15_STORED,) * (rows * columns),
            **cast(dict[str, Any], TWELVE_BIT),
        )
    return dest


def test_product_derive_selects_hu_end_to_end(tmp_path: Path) -> None:
    """F15: import -> derive -> MPR -> viewer/measurement is HU-semantic.

    The oracle is the DICOM Modality LUT transform computed by hand
    (``HU = stored * slope + intercept``) and the frozen VIEWER_001 transfer
    function applied by hand: a physical 300 HU sample is stored as 1324, and
    under ``CT_VESSEL_CONTRAST`` (center 300 / width 800) it must render at
    the mid-window byte 128, not the clipped 255 a STORED-semantic volume
    would produce.  Every expectation is derived independently of the code.
    """
    from apps.viewer.presets import preset_by_id
    from apps.viewer.transfer import window_bounds, window_byte
    from apps.workstation.session_store import (
        DerivedStateStoreManager,
        SessionSnapshot,
    )
    from apps.workstation.volume_loader import CanonicalVolumeLoader
    from apps.workstation.workspace import StudyWorkspaceManager
    from core.events import EventLog

    # The hand oracle must be what the test then asserts (never the reverse).
    assert F15_HU == 300
    assert F15_WL_BYTE_HU == 128
    assert F15_WL_BYTE_STORED == 255

    source = _f15_series(tmp_path / "dicom")
    manager = StudyWorkspaceManager(tmp_path / "ws")
    imported = manager.import_study(source)
    store = DerivedStateStoreManager(imported.paths)
    store.save(
        SessionSnapshot(
            study=imported.study,
            series=tuple(item.series for item in imported.series),
            volumes=(),
            artifacts=(),
            payload_documents={},
            provenance=imported.provenance,
            log=EventLog(),
            skipped_instances=imported.skipped,
        )
    )
    session = store.reopen()
    outcome = manager.derive(session)
    assert len(outcome.volumes) == 1
    volume = outcome.volumes[0]

    loader = CanonicalVolumeLoader(imported.paths.blobs)
    envelope = loader.read_envelope(volume.blob_ref.sha256)
    # (1) the derive path selected HU, and the rescale was applied.
    assert envelope["scalar_domain"] == "HU"
    assert envelope["intensity_unit"] == "HU"
    assert envelope["rescale_applied"] is True
    assert float(cast(Any, envelope["rescale_slope"])) == SLOPE
    assert float(cast(Any, envelope["rescale_intercept"])) == INTERCEPT
    assert envelope["rescale_type"] == "HU"

    # (2) the canonical sample is 300 HU, not the stored 1324.
    array = np.asarray(loader.read_array(volume.blob_ref.sha256, envelope=envelope))
    assert array.shape == (3, 4, 5)
    assert np.all(array == np.asarray(F15_HU, dtype=array.dtype))
    assert int(array[0, 0, 0]) == 300

    # (3) source bit-depth provenance survives the HU selection unchanged.
    assert envelope["source_bits_allocated"] == 16
    assert envelope["source_bits_stored"] == 12
    assert envelope["source_high_bit"] == 11
    assert envelope["source_pixel_representation"] == 0

    # (4) the measurement artifact reports 300 HU (it is named *_hu).
    measurement = next(
        document
        for document in outcome.payload_documents.values()
        if str(document.get("algorithm")) == "central-box-hu-stats"
    )
    assert measurement["mean_hu"] == 300.0
    assert measurement["min_hu"] == 300
    assert measurement["max_hu"] == 300
    # Central box of (3,4,5) by hand: lo = dim//3 = (1,1,1), hi = dim - lo =
    # (2,3,4) -> k in {1}, j in {1,2}, i in {1,2,3} -> 1 * 2 * 3 = 6 voxels.
    assert measurement["voxel_count"] == 6
    assert measurement["roi_lo"] == [1, 1, 1]
    assert measurement["roi_hi"] == [2, 3, 4]

    # (5) the MPR carries the HU domain and the transformed samples.
    frame = volume.coordinate_frame
    transform = transform_for_frame(frame, series_instance_uid=SERIES_UID)
    mpr = build_mpr(
        volume,
        transform,
        plane="AXIAL",
        origin=frame.image_position_patient,
        in_plane_spacing_mm=(1.0, 1.0),
        matrix_size=(2, 3),
        method="NEAREST",
        out_of_bounds_fill_value=0.0,
        loader=loader,
        transform_artifact_ref=f"coordinate_transform:{new_ulid()}",
        provenance_ref=f"provenance:{new_ulid()}",
        study_instance_uid=STUDY_UID,
        series_instance_uid=SERIES_UID,
    )
    assert mpr.payload["scalar_domain"] == "HU"
    assert mpr.payload["intensity_unit"] == "HU"
    pixels = np.asarray(mpr.pixels)
    assert int(pixels[0, 0]) == 300
    assert mpr.payload["bits_allocated"] == 16
    assert mpr.payload["bits_stored"] == 12

    # (6) the viewer's frozen preset acts on 300, not on 1324.
    preset = preset_by_id("CT_VESSEL_CONTRAST")
    assert (preset.center, preset.width) == (F15_WL_CENTER, F15_WL_WIDTH)
    lo, hi = window_bounds(preset.center, preset.width)
    assert (lo, hi) == (-100.0, 700.0)  # center -/+ width/2, hand-computed
    assert window_byte(float(F15_HU), lo, hi) == F15_WL_BYTE_HU
    assert window_byte(float(F15_STORED), lo, hi) == F15_WL_BYTE_STORED
    # The real decoded MPR pixels render at the HU byte.
    assert window_byte(float(pixels[0, 0]), lo, hi) == F15_WL_BYTE_HU
    assert window_byte(float(pixels[0, 0]), lo, hi) != F15_WL_BYTE_STORED

    # (7) geometry and provenance are untouched by the domain selection: the
    # frame is the one built from the SOURCE DICOM, bound here by hand (the
    # F15 fixture's own spacing/orientation/positions and 3 slices of 4x5).
    source_instance = read_dicom_instance(sorted(source.glob("*.dcm"))[0])
    assert tuple(int(v) for v in frame.dimensions) == (3, 4, 5)
    assert tuple(float(v) for v in frame.spacing_mm) == (
        float(source_instance.pixel_spacing[0]),
        float(source_instance.pixel_spacing[1]),
        1.0,  # slice spacing: 1.0 mm between consecutive z positions
    )
    assert tuple(float(v) for v in frame.image_position_patient) == (0.0, 0.0, 0.0)
    assert frame.frame_of_reference_uid == FRAME_UID
    assert tuple(volume.source_series_refs) == (session.series[0].series_id,)


def test_derive_hu_persists_across_save_and_reopen(tmp_path: Path) -> None:
    """F15: the HU selection survives save -> reopen with the session intact."""
    from apps.workstation.round_trip import RoundTripVerificationEngine
    from apps.workstation.session_store import (
        DerivedStateStoreManager,
        SessionSnapshot,
    )
    from apps.workstation.volume_loader import CanonicalVolumeLoader
    from apps.workstation.workspace import StudyWorkspaceManager
    from core.events import EventLog

    source = _f15_series(tmp_path / "dicom")
    manager = StudyWorkspaceManager(tmp_path / "ws")
    imported = manager.import_study(source)
    store = DerivedStateStoreManager(imported.paths)
    store.save(
        SessionSnapshot(
            study=imported.study,
            series=tuple(item.series for item in imported.series),
            volumes=(),
            artifacts=(),
            payload_documents={},
            provenance=imported.provenance,
            log=EventLog(),
            skipped_instances=imported.skipped,
        )
    )
    session = store.reopen()
    outcome = manager.derive(session)
    store.save(
        SessionSnapshot(
            study=session.study,
            series=session.series,
            volumes=session.volumes + outcome.volumes,
            artifacts=session.artifacts + outcome.artifacts,
            payload_documents={**session.payload_documents, **outcome.payload_documents},
            provenance=session.provenance + outcome.provenance,
            log=session.log,
            skipped_instances=session.skipped_instances,
        )
    )
    reopened = store.reopen()
    loader = CanonicalVolumeLoader(imported.paths.blobs)
    envelope = loader.read_envelope(reopened.volumes[0].blob_ref.sha256)
    assert envelope["scalar_domain"] == "HU"
    assert envelope["rescale_applied"] is True
    array = np.asarray(
        loader.read_array(reopened.volumes[0].blob_ref.sha256, envelope=envelope)
    )
    assert int(array[0, 0, 0]) == F15_HU
    assert (
        RoundTripVerificationEngine(imported.paths.root).verify(reopened).verdict == "PASS"
    )


# ---------------------------------------------------------------------------
# R1 fix round (att0002) -- FU01-R1-F2 disclosure regression.
#
# A FRESH multi-series workspace is NOT unconditionally unaffected by the
# full-binding guard.  Two admitted series can canonicalize to the SAME
# content address while carrying DIFFERENT rescale bindings, so the SECOND
# derive into that fresh workspace refuses with the contracted typed code.
# The refusal is SUPPORTED behaviour (it protects the first series'
# provenance); it is asserted here as such, with the first series' workspace
# files proven intact afterwards.  No product code is touched by this test.
# ---------------------------------------------------------------------------

#: Series A: stored 300, slope 1, intercept 0   -> 300 * 1 + 0     = 300 HU.
#: Series B: stored 1324, slope 1, intercept -1024 -> 1324 * 1 - 1024 = 300 HU.
#: Different stored values, different recorded bindings, ONE canonical blob.
F2_SERIES_A_UID: str = "1.2.826.0.1.3680043.8.498.77.6.1"
F2_SERIES_B_UID: str = "1.2.826.0.1.3680043.8.498.77.6.2"
F2_STORED_A: int = 300
F2_STORED_B: int = 1324
F2_INTERCEPT_B: float = -1024.0
#: The shared canonical value, computed by hand from both spellings.
F2_HU_VALUE: int = 300


def _same_digest_series_pair(dest: Path) -> tuple[dict[str, object], ...]:
    """Two 12-bit series: identical canonical HU, DIFFERENT rescale binding.

    Both are Explicit-VR ORIGINAL/PRIMARY unsigned 12-bit CT (three 4x5
    slices, RescaleType HU) and both canonicalize to 300 HU at every sample,
    hence to the identical content digest -- but series A records the slope
    1 / intercept 0 binding and series B records slope 1 / intercept -1024.
    The two spellings are hand-derived: 300*1+0 == 1324*1-1024 == 300.
    """
    specs = (
        (F2_SERIES_A_UID, F2_STORED_A, 0.0, "phase-a", 1),
        (F2_SERIES_B_UID, F2_STORED_B, F2_INTERCEPT_B, "phase-b", 2),
    )
    built: list[dict[str, object]] = []
    for uid, stored, intercept, label, temporal in specs:
        directory = dest / label
        directory.mkdir(parents=True, exist_ok=True)
        for index in range(IDENTITY_SLICES):
            build_instance(
                directory / f"pair-{label}-{index}.dcm",
                sop_instance_uid=f"1.2.826.0.1.3680043.8.498.81.{temporal}.{index + 1}",
                series_instance_uid=uid,
                study_instance_uid=STUDY_UID,
                frame_of_reference_uid=FRAME_UID,
                position=(0.0, 0.0, float(index)),
                rows=4,
                columns=5,
                pixel_data=_words((stored,) * 20),
                rescale_slope=SLOPE,
                rescale_intercept=intercept,
                rescale_type="HU",
                temporal_position_index=temporal,
                **cast(dict[str, Any], TWELVE_BIT),
            )
        built.append({"uid": uid, "dir": directory, "stored": stored, "intercept": intercept})
    return tuple(built)


def _one_series_session(imported: Any, series_uid: str) -> Any:
    """A derive input holding exactly ONE of an imported study's series."""
    from apps.workstation.session_store import SessionSnapshot

    chosen = tuple(
        item.series
        for item in imported.series
        if item.series.dicom_series_instance_uid == series_uid
    )
    assert len(chosen) == 1, "the fixture series must be present exactly once"
    return SessionSnapshot(
        study=imported.study,
        series=chosen,
        volumes=(),
        artifacts=(),
        payload_documents={},
        provenance=imported.provenance,
        log=EventLog(),
        skipped_instances=imported.skipped,
    )


def test_fresh_multi_series_workspace_refuses_same_digest_different_rescale(
    tmp_path: Path,
) -> None:
    """F2 (disclosure): a FRESH two-series workspace can refuse -- asserted.

    Preconditions, each checked independently rather than assumed:

    * the workspace is brand new and held NO blob before the first derive;
    * series A (stored 300, slope 1, intercept 0) and series B (stored 1324,
      slope 1, intercept -1024) both canonicalize to 300 HU at every sample,
      so both claim the SAME content digest IDENTITY_DIGEST;
    * their recorded rescale bindings nevertheless differ (0 vs -1024).

    Supported behaviour under test: the FIRST derive occupies the address and
    the SECOND, same-digest / different-binding derive raises the contracted
    closed-vocabulary refusal SCALAR_DOMAIN_INVALID / SOURCE_BIT_DEPTH_MISMATCH.
    The first series' blob and envelope remain byte-identical and readable
    afterwards; derive returns no outcome, and no session save or automatic
    rollback occurs (see the report's narrowed F3 wording).
    """
    from apps.workstation.session_store import (
        DerivedStateStoreManager,
        SessionSnapshot,
    )
    from apps.workstation.volume_loader import VolumeAbstentionError
    from apps.workstation.workspace import StudyWorkspaceManager

    # The hand oracle: two different stored values, one physical value.
    assert F2_STORED_A * int(SLOPE) + 0 == F2_HU_VALUE
    assert F2_STORED_B * int(SLOPE) + int(F2_INTERCEPT_B) == F2_HU_VALUE

    source = tmp_path / "study"
    pair = _same_digest_series_pair(source)

    manager = StudyWorkspaceManager(tmp_path / "ws-fresh-multiseries")
    imported = manager.import_study(source)
    # A genuinely multi-series import (the fresh-workspace condition).
    assert len(imported.series) == 2
    loader = CanonicalVolumeLoader(imported.paths.blobs)

    # (1) The precondition: BOTH series canonicalize to the same address while
    #     carrying different bindings.  Proved in two SEPARATE blob stores so
    #     that neither materialization can influence the other.
    for offset, entry in enumerate(pair):
        stack = tuple(
            read_dicom_instance(path)
            for path in sorted(cast(Path, entry["dir"]).glob("*.dcm"))
        )
        isolated = CanonicalVolumeLoader(tmp_path / f"proof-{offset}")
        hu = isolated.materialize_dicom_stack(stack, scalar_domain="HU", intensity_unit="HU")
        assert hu.blob.sha256 == IDENTITY_DIGEST, "both series must share the digest"
        assert float(cast(Any, hu.envelope["rescale_intercept"])) == float(
            cast(float, entry["intercept"])
        )
        samples = np.asarray(isolated.read_array(hu.blob.sha256, envelope=hu.envelope))
        assert bool(np.all(samples == np.asarray(F2_HU_VALUE, dtype=samples.dtype)))

    # (2) The workspace is fresh: no blob or envelope exists at the address.
    assert not loader.envelope_path(IDENTITY_DIGEST).exists()
    assert not loader.blob_path(IDENTITY_DIGEST).exists()

    # (3) First derive (series A) occupies the shared address in the FRESH
    #     workspace -- this is the individually-unoccupied case, unaffected.
    store = DerivedStateStoreManager(imported.paths)
    first_session = _one_series_session(imported, F2_SERIES_A_UID)
    first = manager.derive(first_session)
    assert len(first.volumes) == 1
    assert first.volumes[0].blob_ref.sha256 == IDENTITY_DIGEST
    blob_after_first = loader.blob_path(IDENTITY_DIGEST).read_bytes()
    envelope_after_first = loader.envelope_path(IDENTITY_DIGEST).read_bytes()
    occupied = loader.read_envelope(IDENTITY_DIGEST)
    assert occupied["scalar_domain"] == "HU"
    assert occupied["rescale_applied"] is True
    assert float(cast(Any, occupied["rescale_intercept"])) == 0.0
    # Derive persists nothing by itself; the caller saves.
    store.save(
        SessionSnapshot(
            study=first_session.study,
            series=first_session.series,
            volumes=first.volumes,
            artifacts=first.artifacts,
            payload_documents=first.payload_documents,
            provenance=first_session.provenance + first.provenance,
            log=first_session.log,
            skipped_instances=first_session.skipped_instances,
        )
    )
    manifest_after_first = (imported.paths.session / "manifest.json").read_bytes()

    # (4) Second derive (series B): same digest, DIFFERENT binding -> the
    #     contracted typed refusal.  No upgrade or legacy envelope is involved.
    with pytest.raises(VolumeAbstentionError) as error:
        manager.derive(_one_series_session(imported, F2_SERIES_B_UID))
    assert error.value.reason_code == "SCALAR_DOMAIN_INVALID"
    assert error.value.detail_code == "SOURCE_BIT_DEPTH_MISMATCH"

    # (5) The FIRST series' workspace files are intact and still readable: the
    #     refusal happened before replacing the colliding envelope.
    assert loader.blob_path(IDENTITY_DIGEST).read_bytes() == blob_after_first
    assert loader.envelope_path(IDENTITY_DIGEST).read_bytes() == envelope_after_first
    survivor = loader.read_envelope(IDENTITY_DIGEST)
    assert survivor["scalar_domain"] == "HU"
    assert survivor["rescale_applied"] is True
    assert float(cast(Any, survivor["rescale_intercept"])) == 0.0, (
        "the first writer's binding must survive, not be re-labelled"
    )
    assert tuple(survivor[name] for name in SOURCE_BIT_DEPTH_FIELDS) == (16, 12, 11, 0)
    kept = np.asarray(loader.read_array(IDENTITY_DIGEST, envelope=survivor))
    assert kept.shape == (IDENTITY_SLICES, 4, 5)
    assert int(kept[0, 0, 0]) == F2_HU_VALUE

    # (6) User-visible consequence, stated plainly: the refused derive returns
    #     no outcome at all (not a partial one), and the session saved before
    #     it still reopens with series A alone -- the refusal wrote nothing.
    assert (imported.paths.session / "manifest.json").read_bytes() == manifest_after_first
    reopened = store.reopen()
    assert len(reopened.volumes) == 1
    assert reopened.volumes[0].blob_ref.sha256 == IDENTITY_DIGEST


# ---------------------------------------------------------------------------
# FOLLOW-UP 01 (F19, P1) -- voxel SHAPE is a STRUCTURAL binding of a content
# address.
#
# A blob digest addresses raw BYTES, and equal bytes do not determine
# geometry: 4 int16 samples are one byte string whether they are laid out as
# (1,2,2), (2,1,2) or (4,1,1).  Equal bytes + an equal nine-field scalar
# binding + a DIFFERENT shape must therefore refuse with the EXISTING typed
# code, leaving the first shape's envelope byte-identical; equal bytes + equal
# binding + the SAME shape stays allowed (idempotent rematerialization).
#
# The tests below are HERMETIC: they drive CanonicalVolumeLoader directly over
# a tmp_path blob store with hand-built arrays.  No DICOM fixture, no import,
# no workspace, no session, no viewer and no golden artifact is involved, and
# no product behaviour other than the collision guard is exercised.
# ---------------------------------------------------------------------------

#: Four samples: identical bytes under (1,2,2) and under (2,1,2).
F19_SHAPE_A: tuple[int, int, int] = (1, 2, 2)
F19_SHAPE_B: tuple[int, int, int] = (2, 1, 2)
F19_SAMPLE_COUNT: int = 4
#: The same four physical values in both layouts, hand-listed (no arithmetic
#: oracle is needed: the two arrays hold the same four numbers in the same
#: C order, only the shape metadata differs).
F19_SAMPLES: tuple[int, ...] = (300, 300, 300, 300)


def _f19_array(shape: tuple[int, int, int]) -> Any:
    """Four int16 samples (all 300) declared with *shape* geometry.

    Built by hand: one flat sequence of :data:`F19_SAMPLE_COUNT` values
    reshaped to *shape*, so both spellings are byte-identical in C order.
    """
    flat = np.asarray(F19_SAMPLES, dtype=np.uint16)
    assert flat.size == F19_SAMPLE_COUNT
    assert shape[0] * shape[1] * shape[2] == F19_SAMPLE_COUNT
    return flat.reshape(shape)


def _f19_kwargs() -> dict[str, Any]:
    """The identical nine-field scalar binding for both materializations.

    One source (12-bit stored, unsigned), one domain (STORED), one identity
    rescale -- so the ONLY thing that differs between the two writes is the
    geometry.  That isolates shape as the variable under test.
    """
    return {
        "source_bits_allocated": 16,
        "source_bits_stored": 12,
        "source_high_bit": 11,
        "source_pixel_representation": 0,
        "scalar_domain": "STORED",
        "intensity_unit": "STORED",
        "rescale_slope": 1.0,
        "rescale_intercept": 0.0,
        "rescale_type": "HU",
    }


def test_shape_is_a_structural_binding_of_the_content_address(tmp_path: Path) -> None:
    """F19: equal bytes + equal binding + DIFFERENT shape => typed refusal.

    Preconditions, each asserted rather than assumed:

    * the two arrays are byte-identical (they claim ONE content address);
    * their nine-field scalar bindings are equal (isolating shape as the only
      differing dimension);
    * their envelopes' shape metadata genuinely differs.

    Supported behaviour under test: the second write raises the EXISTING
    closed-vocabulary refusal SCALAR_DOMAIN_INVALID / SOURCE_BIT_DEPTH_MISMATCH
    -- no new reason code -- the envelope bytes are unchanged, and the FIRST
    shape survives a reopen (read_array decodes with the first writer's
    geometry, not the incoming one).
    """
    from apps.workstation.volume_loader import VolumeAbstentionError

    loader = CanonicalVolumeLoader(tmp_path / "blobs-f19")
    first_array = _f19_array(F19_SHAPE_A)
    second_array = _f19_array(F19_SHAPE_B)
    kwargs = _f19_kwargs()

    # (1) The collision condition really holds: one content address, one
    #     binding, two geometries.
    assert first_array.tobytes() == second_array.tobytes(), (
        "both layouts must be byte-identical to collide on one digest"
    )
    assert list(first_array.shape) == list(F19_SHAPE_A)
    assert list(second_array.shape) == list(F19_SHAPE_B)
    assert F19_SHAPE_A != F19_SHAPE_B

    # (2) First writer occupies the address with shape A.
    first = loader.materialize_canonical(first_array, **kwargs)
    digest = first.blob.sha256
    assert first.envelope["shape"] == list(F19_SHAPE_A)
    envelope_bytes = loader.envelope_path(digest).read_bytes()
    blob_bytes = loader.blob_path(digest).read_bytes()
    binding_first = CanonicalVolumeLoader._scalar_binding(first.envelope)

    # The nine scalar fields of both sides are equal -- shape is the ONLY
    # difference, so a refusal here can only be structural.
    incoming_fields = dict(first.envelope)
    incoming_fields["shape"] = list(F19_SHAPE_B)
    binding_incoming = CanonicalVolumeLoader._scalar_binding(incoming_fields)
    assert binding_first[:-1] == binding_incoming[:-1], (
        "the nine-field scalar binding must be equal for this test to isolate shape"
    )

    # (3) The second write refuses, typed, with the EXISTING code.
    with pytest.raises(VolumeAbstentionError) as caught:
        loader.materialize_canonical(second_array, **kwargs)
    assert caught.value.reason_code == "SCALAR_DOMAIN_INVALID"
    assert caught.value.detail_code == "SOURCE_BIT_DEPTH_MISMATCH"

    # (4) Nothing was rewritten: envelope and blob are byte-identical.
    assert loader.envelope_path(digest).read_bytes() == envelope_bytes
    assert loader.blob_path(digest).read_bytes() == blob_bytes

    # (5) The FIRST shape survives a reopen -- the envelope still describes
    #     geometry A, and read_array decodes with it.
    survivor = loader.read_envelope(digest)
    assert survivor["shape"] == list(F19_SHAPE_A)
    assert survivor["byte_size"] == int(cast(Any, first.envelope["byte_size"]))
    decoded = np.asarray(loader.read_array(digest))
    assert decoded.shape == F19_SHAPE_A, (
        "the first writer's geometry must survive, not the incoming one"
    )
    assert int(decoded.reshape(-1)[0]) == F19_SAMPLES[0]


def test_same_shape_rematerialization_stays_allowed_after_f19(tmp_path: Path) -> None:
    """F19 (allowed side): equal bytes + equal binding + SAME shape => allowed.

    Idempotent rematerialization must keep working: the guard's shape
    dimension is an EQUALITY test over the structural binding, not a blanket
    refusal of a second write at an occupied address.
    """
    loader = CanonicalVolumeLoader(tmp_path / "blobs-f19-allowed")
    array = _f19_array(F19_SHAPE_A)
    kwargs = _f19_kwargs()

    first = loader.materialize_canonical(array, **kwargs)
    digest = first.blob.sha256
    envelope_bytes = loader.envelope_path(digest).read_bytes()

    # Same everything, same shape: ALLOWED (no refusal), same digest.
    again = loader.materialize_canonical(_f19_array(F19_SHAPE_A), **kwargs)
    assert again.blob.sha256 == digest
    assert again.envelope["shape"] == list(F19_SHAPE_A)
    # Idempotent: the envelope is left exactly as the first writer wrote it.
    assert loader.envelope_path(digest).read_bytes() == envelope_bytes
    decoded = np.asarray(loader.read_array(digest))
    assert decoded.shape == F19_SHAPE_A


def test_shape_binding_is_structural_not_a_raw_list_compare() -> None:
    """F19: shape compares STRUCTURALLY, so spellings are not the variable.

    An equal shape spelled as a list, a tuple or another Sequence compares
    equal; a genuinely different geometry compares unequal; and an
    absent/invalid shape is its own distinct state rather than being coerced
    into a guess (fail-closed, matching the rescale normalization rules).
    """
    shape_binding = CanonicalVolumeLoader._shape_binding

    # Equal geometry, different container spelling -> EQUAL (structural).
    assert shape_binding({"shape": [1, 2, 2]}) == (1, 2, 2)
    assert shape_binding({"shape": (1, 2, 2)}) == (1, 2, 2)
    assert shape_binding({"shape": [1, 2, 2]}) == shape_binding({"shape": (1, 2, 2)})

    # A different geometry -> unequal.
    assert shape_binding({"shape": [1, 2, 2]}) != shape_binding({"shape": [2, 1, 2]})
    assert shape_binding({"shape": [1, 2, 2]}) != shape_binding({"shape": [4, 1, 1]})

    # Absent, malformed and out-of-contract shapes are each their own state:
    # never coerced, never defaulted, never equal to a real geometry.
    absent = shape_binding({})
    assert absent != (1, 2, 2)
    assert shape_binding({"shape": [1, 2]}) != (1, 2, 2)  # not 3D
    assert shape_binding({"shape": [1, 2, 0]}) != (1, 2, 2)  # non-positive
    assert shape_binding({"shape": [1, 2, True]}) != (1, 2, 2)  # a bool is not an int
    assert shape_binding({"shape": "1,2,2"}) != (1, 2, 2)
    assert shape_binding({"shape": [1, 2, 2]}) != absent


# ---------------------------------------------------------------------------
# IC002-R1 F10-rest — direct product persistence, legacy byte identity,
# typed missing-source-depth abstention, and outer/inner schema matching.
# Each case reuses the existing fixtures/helpers above; no fixture data is
# committed and no golden is edited.
# ---------------------------------------------------------------------------


def test_product_import_derive_save_reopen_emits_v2_with_all_fields(tmp_path: Path) -> None:
    """F10(a): the PRODUCT import/derive/save/reopen path emits v2 with all fields.

    The oracle is the DICOM rescale arithmetic computed by hand: stored 1324
    with slope 1 / intercept -1024 is 300 HU (``1324 - 1024 = 300``), so the
    product derive path must canonicalize to 300, persist a v2 envelope with
    the four source-bit-depth fields, and produce a v2 MPR payload carrying the
    same four fields and the HU domain.  The save/reopen round trip keeps the
    envelope and the MPR payload byte-identical.
    """
    from apps.workstation.session_store import DerivedStateStoreManager, SessionSnapshot
    from apps.workstation.workspace import StudyWorkspaceManager
    from core.digest import canonical_digest
    from visualization.mpr import validate_mpr_payload

    # Hand oracle: 1324 - 1024 = 300.
    assert F15_HU == 300
    assert F15_STORED == 1324

    source = _f15_series(tmp_path / "dicom")
    manager = StudyWorkspaceManager(tmp_path / "ws")
    imported = manager.import_study(source)
    assert len(imported.series) == 1

    store = DerivedStateStoreManager(imported.paths)
    store.save(_import_snapshot(imported))
    session = store.reopen()
    outcome = manager.derive(session)
    assert len(outcome.volumes) == 1
    volume = outcome.volumes[0]

    loader = CanonicalVolumeLoader(imported.paths.blobs)
    envelope = loader.read_envelope(volume.blob_ref.sha256)
    # The derive path selected HU and recorded the full v2 source tuple.
    assert envelope["scalar_domain"] == "HU"
    assert envelope["intensity_unit"] == "HU"
    assert envelope["rescale_applied"] is True
    assert float(cast(float, envelope["rescale_slope"])) == SLOPE
    assert float(cast(float, envelope["rescale_intercept"])) == INTERCEPT
    assert tuple(envelope[name] for name in SOURCE_BIT_DEPTH_FIELDS) == (16, 12, 11, 0)

    # Canonical samples are HU (300), not stored (1324).
    array = np.asarray(loader.read_array(volume.blob_ref.sha256, envelope=envelope))
    assert int(array[0, 0, 0]) == F15_HU
    assert bool(np.all(array == np.asarray(F15_HU, dtype=array.dtype)))

    # Build one pinned MPR from the product volume (same frozen code path).
    frame = volume.coordinate_frame
    series_uid = imported.series[0].series.dicom_series_instance_uid
    transform = transform_for_frame(frame, series_instance_uid=series_uid)
    mpr = build_mpr(
        volume,
        transform,
        plane="AXIAL",
        origin=frame.image_position_patient,
        in_plane_spacing_mm=(1.0, 1.0),
        matrix_size=(2, 3),
        method="NEAREST",
        out_of_bounds_fill_value=0.0,
        loader=loader,
        transform_artifact_ref=f"coordinate_transform:{PINNED_ULID}",
        provenance_ref=f"provenance:{PINNED_ULID}",
        study_instance_uid=STUDY_UID,
        series_instance_uid=series_uid,
    )
    payload = mpr.payload
    # v2 with all four additive fields.
    assert payload["schema_version"] == "ccta.mpr.view.v2"
    assert payload["bits_allocated"] == 16
    assert payload["bits_stored"] == 12
    assert payload["high_bit"] == 11
    assert payload["pixel_representation"] == 0
    assert payload["scalar_domain"] == "HU"
    assert payload["intensity_unit"] == "HU"
    # Outer artifact schema matches inner payload version.
    assert mpr.artifact.payload.payload_schema == "ccta.mpr.view.v2"
    assert mpr.artifact.payload.payload_schema == payload["schema_version"]
    assert mpr.artifact.payload.blob_sha256 == canonical_digest(payload)
    assert validate_mpr_payload(payload) == payload
    # MPR pixels are HU (300) as well.
    assert int(np.asarray(mpr.pixels)[0, 0]) == F15_HU

    # Save the derived session and reopen: envelope and payload survive verbatim.
    payload_before = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    merged_docs = {
        **session.payload_documents,
        **outcome.payload_documents,
        mpr.artifact.artifact_id: payload,
    }
    store.save(
        SessionSnapshot(
            study=session.study,
            series=session.series,
            volumes=session.volumes + outcome.volumes,
            artifacts=session.artifacts + outcome.artifacts,
            payload_documents=merged_docs,
            provenance=session.provenance + outcome.provenance,
            log=session.log,
            skipped_instances=session.skipped_instances,
        )
    )
    reopened = store.reopen()
    # The manifest now contains the derived volume (different from before), but it is still valid.
    assert (imported.paths.session / "manifest.json").exists()
    # The reopened volume is the same content address with the same v2 envelope.
    assert reopened.volumes[0].blob_ref.sha256 == volume.blob_ref.sha256
    envelope_after = loader.read_envelope(reopened.volumes[0].blob_ref.sha256)
    assert envelope_after == envelope
    # The reopened payload document is byte-identical.
    reopened_doc = reopened.payload_documents[mpr.artifact.artifact_id]
    assert (
        json.dumps(reopened_doc, sort_keys=True, separators=(",", ":")).encode()
        == payload_before
    )
    # Rebuilding the same MPR after reopen still emits v2 with all fields.
    volume2 = reopened.volumes[0]
    series_uid2 = imported.series[0].series.dicom_series_instance_uid
    transform2 = transform_for_frame(volume2.coordinate_frame, series_instance_uid=series_uid2)
    rebuilt = build_mpr(
        volume2,
        transform2,
        plane="AXIAL",
        origin=volume2.coordinate_frame.image_position_patient,
        in_plane_spacing_mm=(1.0, 1.0),
        matrix_size=(2, 3),
        method="NEAREST",
        out_of_bounds_fill_value=0.0,
        loader=loader,
        transform_artifact_ref=f"coordinate_transform:{PINNED_ULID}",
        provenance_ref=f"provenance:{PINNED_ULID}",
        study_instance_uid=STUDY_UID,
        series_instance_uid=series_uid2,
    )
    assert rebuilt.payload["schema_version"] == "ccta.mpr.view.v2"
    assert rebuilt.payload["bits_allocated"] == 16
    assert (
        json.dumps(rebuilt.payload, sort_keys=True, separators=(",", ":")).encode()
        == payload_before
    )


def test_legacy_v1_frames_are_byte_identical_after_reopen(tmp_path: Path) -> None:
    """F10(b): legacy (record-free, v1) MPR frames survive save/reopen byte-identical.

    A pre-amendment canonical volume carries no source-bit-depth record; its
    MPR is v1 and must stay v1 after the session is saved and reopened, with
    both the envelope and the pixel bytes unchanged.  The pinned MPR helper
    makes the byte comparison exact (only identity-bearing refs differ).
    """
    from apps.workstation.session_store import DerivedStateStoreManager, SessionSnapshot
    from apps.workstation.workspace import StudyWorkspaceManager
    from core.events import EventLog

    # Create a workspace to obtain session/blob paths.
    source = _f15_series(tmp_path / "src")
    manager = StudyWorkspaceManager(tmp_path / "ws")
    imported = manager.import_study(source)
    loader = CanonicalVolumeLoader(imported.paths.blobs)

    values = _analytic_unsigned()
    frame = _geometry("axis-aligned", "isotropic", "standard", "zero")
    volume = _record_free_volume(loader, frame, values)
    # The envelope is deliberately record-free (pre-amendment).
    envelope = loader.read_envelope(volume.blob_ref.sha256)
    assert all(name not in envelope for name in SOURCE_BIT_DEPTH_FIELDS)

    first = _pinned_mpr_frame(loader, volume, frame)
    assert first.payload["schema_version"] == "ccta.mpr.view.v1"
    assert "bits_allocated" not in first.payload
    assert "bits_stored" not in first.payload
    assert "high_bit" not in first.payload
    assert "pixel_representation" not in first.payload
    assert first.artifact.payload.payload_schema == "ccta.mpr.view.v1"
    assert first.artifact.payload.payload_schema == first.payload["schema_version"]

    pixels_before = first.pixels.tobytes(order="C")
    payload_before = json.dumps(first.payload, sort_keys=True, separators=(",", ":")).encode()
    blob_before = loader.blob_path(volume.blob_ref.sha256).read_bytes()
    envelope_before = loader.envelope_path(volume.blob_ref.sha256).read_bytes()

    # Save a session that owns this legacy volume and its v1 MPR.
    store = DerivedStateStoreManager(imported.paths)
    log = EventLog()
    log.seed(
        entity="derived_artifact",
        entity_id=first.artifact.artifact_id,
        initial_fields={
            "review_state": "PREDICTED",
            "status": "ACTIVE",
            "release_lane": "PERSONAL_RESEARCH",
        },
    )
    # The import path has no saved session yet; seed it first so reopen has a base.
    store.save(_import_snapshot(imported))
    session = store.reopen()
    # Overwrite the session with the legacy volume+MPR so the reopen path is exercised for v1.
    store.save(
        SessionSnapshot(
            study=session.study,
            series=session.series,
            volumes=(volume,),
            artifacts=(first.artifact,),
            payload_documents={first.artifact.artifact_id: first.payload},
            provenance=session.provenance,
            log=log,
            skipped_instances=session.skipped_instances,
        )
    )
    reopened = store.reopen()
    assert reopened.volumes[0].blob_ref.sha256 == volume.blob_ref.sha256
    doc = reopened.payload_documents[first.artifact.artifact_id]
    assert doc["schema_version"] == "ccta.mpr.view.v1"
    # Blob and envelope files are byte-identical after save/reopen.
    assert loader.blob_path(volume.blob_ref.sha256).read_bytes() == blob_before
    assert loader.envelope_path(volume.blob_ref.sha256).read_bytes() == envelope_before
    kept = loader.read_envelope(volume.blob_ref.sha256)
    assert all(name not in kept for name in SOURCE_BIT_DEPTH_FIELDS)

    # Rebuilding the same MPR after reopen is byte-identical (pinned refs).
    rebuilt = _pinned_mpr_frame(
        CanonicalVolumeLoader(imported.paths.blobs), reopened.volumes[0], frame
    )
    assert rebuilt.payload["schema_version"] == "ccta.mpr.view.v1"
    assert "bits_allocated" not in rebuilt.payload
    assert rebuilt.pixels.tobytes(order="C") == pixels_before
    rebuilt_bytes = json.dumps(rebuilt.payload, sort_keys=True, separators=(",", ":")).encode()
    assert rebuilt_bytes == payload_before
    reopened_bytes = json.dumps(
        reopened.payload_documents[first.artifact.artifact_id],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    assert reopened_bytes == payload_before
    assert rebuilt.artifact.payload.payload_schema == "ccta.mpr.view.v1"


def test_typed_missing_source_depth_abstention(tmp_path: Path) -> None:
    """F10(c): public product consumers refuse partial/malformed depth records.

    The volume is first produced through the same import -> derive -> save ->
    reopen path as the passing v2 product case.  The persisted envelope is then
    corrupted at the test boundary.  The public loader/MPR consumer observes
    the corruption and reports its typed code; no adapter method or private MPR
    helper is called by this test.
    """
    from apps.workstation.session_store import DerivedStateStoreManager, SessionSnapshot
    from apps.workstation.volume_loader import VolumeAbstentionError
    from apps.workstation.workspace import StudyWorkspaceManager
    from visualization.mpr import MPRAbstentionError

    class _CorruptedEnvelopeLoader:
        """Expose a corrupted envelope to public MPR code without writing it."""

        def __init__(
            self,
            base: CanonicalVolumeLoader,
            corrupted: dict[str, object],
            valid: dict[str, object],
        ) -> None:
            self._base = base
            self._corrupted = corrupted
            self._valid = valid

        def read_envelope(self, digest: str) -> dict[str, object]:
            assert digest == volume.blob_ref.sha256
            return dict(self._corrupted)

        def read_array(self, digest: str, *, envelope: dict[str, object] | None = None) -> Any:
            assert digest == volume.blob_ref.sha256
            # Decode through the real public loader using the untouched record;
            # build_mpr still receives and classifies the corrupted record.
            return self._base.read_array(digest, envelope=self._valid)

        def write_blob(self, data: bytes) -> Any:
            return self._base.write_blob(data)

    source = _f15_series(tmp_path / "dicom")
    manager = StudyWorkspaceManager(tmp_path / "ws")
    imported = manager.import_study(source)
    store = DerivedStateStoreManager(imported.paths)
    store.save(_import_snapshot(imported))
    imported_session = store.reopen()
    outcome = manager.derive(imported_session)
    assert len(outcome.volumes) == 1
    store.save(
        SessionSnapshot(
            study=imported_session.study,
            series=imported_session.series,
            volumes=imported_session.volumes + outcome.volumes,
            artifacts=imported_session.artifacts + outcome.artifacts,
            payload_documents={
                **imported_session.payload_documents,
                **outcome.payload_documents,
            },
            provenance=imported_session.provenance + outcome.provenance,
            log=imported_session.log,
            skipped_instances=imported_session.skipped_instances,
        )
    )
    reopened = store.reopen()
    volume = reopened.volumes[0]
    loader = CanonicalVolumeLoader(reopened.paths.blobs)
    digest = volume.blob_ref.sha256
    original_envelope = loader.read_envelope(digest)
    assert tuple(original_envelope[name] for name in SOURCE_BIT_DEPTH_FIELDS) == (16, 12, 11, 0)
    original_envelope_bytes = loader.envelope_path(digest).read_bytes()
    frame = volume.coordinate_frame
    series_uid = reopened.series[0].dicom_series_instance_uid
    transform = transform_for_frame(frame, series_instance_uid=series_uid)

    def _build_public_mpr(public_loader: Any) -> Any:
        return build_mpr(
            volume,
            transform,
            plane="AXIAL",
            origin=frame.image_position_patient,
            in_plane_spacing_mm=(1.0, 1.0),
            matrix_size=(2, 3),
            method="NEAREST",
            out_of_bounds_fill_value=0.0,
            loader=public_loader,
            transform_artifact_ref=f"coordinate_transform:{PINNED_ULID}",
            provenance_ref=f"provenance:{PINNED_ULID}",
            study_instance_uid=STUDY_UID,
            series_instance_uid=series_uid,
        )

    # The actual persisted-product path rejects malformed envelope records with
    # the loader's typed public codes, and the same envelope file is restored
    # after every case so no attempted write can target a different loader.
    envelope_path = loader.envelope_path(digest)
    for missing_field in SOURCE_BIT_DEPTH_FIELDS:
        corrupted = dict(original_envelope)
        corrupted.pop(missing_field)
        envelope_path.write_text(json.dumps(corrupted), encoding="utf-8")
        try:
            with pytest.raises(VolumeAbstentionError) as error:
                _build_public_mpr(loader)
            assert error.value.detail_code == "SOURCE_BIT_DEPTH_INCOMPLETE"
        finally:
            envelope_path.write_bytes(original_envelope_bytes)
        assert loader.read_envelope(digest) == original_envelope
        assert envelope_path.read_bytes() == original_envelope_bytes

    for bad in (
        {**original_envelope, "source_high_bit": "11"},
        {**original_envelope, "source_bits_stored": True},
    ):
        envelope_path.write_text(json.dumps(bad), encoding="utf-8")
        try:
            with pytest.raises(VolumeAbstentionError) as error:
                _build_public_mpr(loader)
            assert error.value.detail_code == "SOURCE_BIT_DEPTH_INVALID"
        finally:
            envelope_path.write_bytes(original_envelope_bytes)
        assert loader.read_envelope(digest) == original_envelope
        assert envelope_path.read_bytes() == original_envelope_bytes

    # The public MPR consumer has a distinct missing/invalid classification
    # once it is handed the corrupted record.  The wrapper only supplies that
    # record; the public build path and real loader perform the observation.
    for missing_field in SOURCE_BIT_DEPTH_FIELDS:
        corrupted = dict(original_envelope)
        corrupted.pop(missing_field)
        with pytest.raises(MPRAbstentionError) as mpr_error:
            _build_public_mpr(
                _CorruptedEnvelopeLoader(loader, corrupted, original_envelope)
            )
        assert mpr_error.value.reason_code == "MPR_SOURCE_INVALID"
        assert mpr_error.value.detail_code == "SOURCE_BIT_DEPTH_MISSING"
        assert envelope_path.read_bytes() == original_envelope_bytes

    for bad in (
        {**original_envelope, "source_high_bit": "11"},
        {**original_envelope, "source_bits_stored": True},
    ):
        with pytest.raises(MPRAbstentionError) as mpr_error:
            _build_public_mpr(_CorruptedEnvelopeLoader(loader, bad, original_envelope))
        assert mpr_error.value.reason_code == "MPR_SOURCE_INVALID"
        assert mpr_error.value.detail_code == "SOURCE_BIT_DEPTH_INVALID"
        assert envelope_path.read_bytes() == original_envelope_bytes


def test_matching_outer_inner_schemas_for_both_versions(tmp_path: Path) -> None:
    """F10(d): reopened v1/v2 artifacts preserve schema and content digests.

    The v2 case uses the modern product import/derive path.  The v1 case uses
    the existing record-free legacy seeding pattern.  Fresh builds are checked
    as setup, but the acceptance assertions are made against artifacts and
    payload documents returned by ``DerivedStateStoreManager.reopen``.
    """
    from apps.workstation.session_store import DerivedStateStoreManager, SessionSnapshot
    from apps.workstation.workspace import StudyWorkspaceManager
    from core.digest import canonical_digest
    from visualization.mpr import MPR_SCHEMA, MPR_SCHEMA_LEGACY, validate_mpr_payload

    def _assert_reopened_schema(store: Any, artifact_id: str, expected: str) -> dict[str, object]:
        restored = store.reopen()
        artifact = next(item for item in restored.artifacts if item.artifact_id == artifact_id)
        document = restored.payload_documents[artifact_id]
        assert artifact.payload.payload_schema == expected
        assert document["schema_version"] == expected
        assert artifact.payload.payload_schema == document["schema_version"]
        assert artifact.payload.blob_sha256 == canonical_digest(document)
        assert validate_mpr_payload(document) == document
        return document

    frame = _geometry("axis-aligned", "isotropic", "standard", "zero")
    values = _analytic_unsigned()

    # ---- v2: modern import -> derive -> save -> reopen -----------------------
    source = _f15_series(tmp_path / "v2-dicom")
    manager = StudyWorkspaceManager(tmp_path / "v2-ws")
    imported = manager.import_study(source)
    store_v2 = DerivedStateStoreManager(imported.paths)
    store_v2.save(_import_snapshot(imported))
    imported_session = store_v2.reopen()
    outcome = manager.derive(imported_session)
    volume_v2 = outcome.volumes[0]
    loader_v2 = CanonicalVolumeLoader(imported.paths.blobs)
    frame_v2 = volume_v2.coordinate_frame
    series_uid_v2 = imported.series[0].series.dicom_series_instance_uid
    transform_v2 = transform_for_frame(frame_v2, series_instance_uid=series_uid_v2)
    mpr_v2 = build_mpr(
        volume_v2,
        transform_v2,
        plane="AXIAL",
        origin=frame_v2.image_position_patient,
        in_plane_spacing_mm=(1.0, 1.0),
        matrix_size=(2, 3),
        method="NEAREST",
        out_of_bounds_fill_value=0.0,
        loader=loader_v2,
        transform_artifact_ref=f"coordinate_transform:{PINNED_ULID}",
        provenance_ref=f"provenance:{PINNED_ULID}",
        study_instance_uid=STUDY_UID,
        series_instance_uid=series_uid_v2,
    )
    assert mpr_v2.payload["schema_version"] == MPR_SCHEMA == "ccta.mpr.view.v2"
    assert tuple(
        loader_v2.read_envelope(volume_v2.blob_ref.sha256)[name]
        for name in SOURCE_BIT_DEPTH_FIELDS
    ) == (16, 12, 11, 0)
    imported_session.log.seed(
        entity="derived_artifact",
        entity_id=mpr_v2.artifact.artifact_id,
        initial_fields={
            "review_state": mpr_v2.artifact.review.review_state.value,
            "status": mpr_v2.artifact.status.value,
            "release_lane": mpr_v2.artifact.release_lane.value,
        },
    )
    store_v2.save(
        SessionSnapshot(
            study=imported_session.study,
            series=imported_session.series,
            volumes=imported_session.volumes + outcome.volumes,
            artifacts=imported_session.artifacts + outcome.artifacts + (mpr_v2.artifact,),
            payload_documents={
                **imported_session.payload_documents,
                **outcome.payload_documents,
                mpr_v2.artifact.artifact_id: mpr_v2.payload,
            },
            provenance=imported_session.provenance + outcome.provenance,
            log=imported_session.log,
            skipped_instances=imported_session.skipped_instances,
        )
    )
    reopened_v2 = store_v2.reopen()
    document_v2 = _assert_reopened_schema(
        store_v2, mpr_v2.artifact.artifact_id, MPR_SCHEMA
    )
    assert document_v2["bits_allocated"] == 16
    assert document_v2["bits_stored"] == 12
    assert document_v2["high_bit"] == 11
    assert document_v2["pixel_representation"] == 0

    # ---- v1: existing record-free legacy-seeding pattern ---------------------
    manager_v1 = StudyWorkspaceManager(tmp_path / "v1-ws")
    imported_v1 = manager_v1.import_study(_f15_series(tmp_path / "v1-dicom"))
    loader_v1 = CanonicalVolumeLoader(imported_v1.paths.blobs)
    volume_v1 = _record_free_volume(loader_v1, frame, values)
    envelope_v1 = loader_v1.read_envelope(volume_v1.blob_ref.sha256)
    assert all(name not in envelope_v1 for name in SOURCE_BIT_DEPTH_FIELDS)
    mpr_v1 = _pinned_mpr_frame(loader_v1, volume_v1, frame)
    assert mpr_v1.payload["schema_version"] == MPR_SCHEMA_LEGACY == "ccta.mpr.view.v1"

    store_v1 = DerivedStateStoreManager(imported_v1.paths)
    store_v1.save(_import_snapshot(imported_v1))
    seeded_v1 = store_v1.reopen()
    seeded_v1.log.seed(
        entity="derived_artifact",
        entity_id=mpr_v1.artifact.artifact_id,
        initial_fields={
            "review_state": mpr_v1.artifact.review.review_state.value,
            "status": mpr_v1.artifact.status.value,
            "release_lane": mpr_v1.artifact.release_lane.value,
        },
    )
    store_v1.save(
        SessionSnapshot(
            study=seeded_v1.study,
            series=seeded_v1.series,
            volumes=(volume_v1,),
            artifacts=(mpr_v1.artifact,),
            payload_documents={mpr_v1.artifact.artifact_id: mpr_v1.payload},
            provenance=seeded_v1.provenance,
            log=seeded_v1.log,
            skipped_instances=seeded_v1.skipped_instances,
        )
    )
    reopened_v1 = store_v1.reopen()
    document_v1 = _assert_reopened_schema(
        store_v1, mpr_v1.artifact.artifact_id, MPR_SCHEMA_LEGACY
    )
    assert "bits_allocated" not in document_v1
    assert "bits_stored" not in document_v1
    assert "high_bit" not in document_v1
    assert "pixel_representation" not in document_v1
    assert reopened_v2.volumes[0].blob_ref.sha256 == volume_v2.blob_ref.sha256
    assert reopened_v1.volumes[0].blob_ref.sha256 == volume_v1.blob_ref.sha256
