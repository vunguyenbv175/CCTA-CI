"""Seeded synthetic study fixtures — NO real patient data anywhere (§B9 gate).

Emits Explicit-VR Little-Endian Part-10 files containing the deterministic
metadata needed by the Imaging Core DICOM boundary. All UIDs, pixels
(LCG-seeded noise) and geometry are fixed constants, so every derived digest
is stable across runs.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Final, cast

import numpy as np
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import UID

STUDY_UID: Final[str] = "1.2.826.0.1.3680043.8.498.77.1"
SERIES_UID_TEMPLATE: Final[str] = "1.2.826.0.1.3680043.8.498.77.2.{phase}"
SOP_UID_TEMPLATE: Final[str] = "1.2.826.0.1.3680043.8.498.77.3.{phase}.{slice}"
FRAME_UID: Final[str] = "1.2.826.0.1.3680043.8.498.77.9"

TRANSFER_SYNTAX_EXPLICIT_VR_LE: Final[str] = "1.2.840.10008.1.2.1"
DEFAULT_ORIENTATION: Final[str] = "1\\0\\0\\0\\1\\0"
DEFAULT_SPACING: Final[str] = "0.5\\0.5"
DEFAULT_THICKNESS: Final[str] = "1.0"
SLICE_STEP_MM: Final[float] = 1.0

_LONG_FORM_VRS: Final[frozenset[bytes]] = frozenset(
    {b"OB", b"OD", b"OF", b"OL", b"OV", b"OW", b"SQ", b"SV", b"UC", b"UN", b"UR", b"UT", b"UV"}
)


def _pad_even(value: bytes, pad: bytes) -> bytes:
    return value if len(value) % 2 == 0 else value + pad


def _element(group: int, element: int, vr: bytes, value: bytes) -> bytes:
    tag = struct.pack("<HH", group, element)
    if vr in _LONG_FORM_VRS:
        return tag + vr + b"\x00\x00" + struct.pack("<I", len(value)) + value
    return tag + vr + struct.pack("<H", len(value)) + value


def _text_element(group: int, element: int, vr: bytes, text: str) -> bytes:
    pad = b" " if vr != b"UI" else b"\x00"
    return _element(group, element, vr, _pad_even(text.encode("ascii"), pad))


def _us_element(group: int, element: int, value: int) -> bytes:
    return _element(group, element, b"US", struct.pack("<H", value))


def _lcg(seed: int) -> bytes:
    """Deterministic uint16-LE pixel noise for one slice."""
    state = (seed * 2654435761 + 1013904223) & 0x7FFFFFFF
    out = bytearray()
    for _ in range(16):
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        word = (state >> 8) & 0xFFFF
        out += struct.pack("<H", word)
        out += struct.pack("<H", (word ^ 0x00FF) & 0xFFFF)
    return bytes(out)


def build_instance(
    path: Path,
    *,
    sop_instance_uid: str,
    series_instance_uid: str,
    study_instance_uid: str = STUDY_UID,
    frame_of_reference_uid: str = FRAME_UID,
    position: tuple[float, float, float],
    rows: int = 4,
    columns: int = 4,
    modality: str = "CT",
    temporal_position: str | None = None,
    orientation: str = DEFAULT_ORIENTATION,
    spacing: str = DEFAULT_SPACING,
    thickness: str = DEFAULT_THICKNESS,
    pixel_seed: int = 1,
    temporal_position_index: int | None = None,
    image_type: tuple[str, ...] = ("ORIGINAL", "PRIMARY"),
    series_description: str | None = None,
    body_part_examined: str | None = "CHEST",
    protocol_name: str | None = None,
    pixel_representation: int = 0,
    rescale_slope: float | None = None,
    rescale_intercept: float | None = None,
    rescale_type: str | None = None,
    contrast_bolus_agent: str | None = None,
    contrast_bolus_volume: float | None = None,
    contrast_bolus_start_time: str | None = None,
    contrast_bolus_stop_time: str | None = None,
    contrast_flow_rate: float | None = None,
    contrast_flow_duration: float | None = None,
    trigger_time: float | None = None,
    cardiac_synchronization_technique: str | None = None,
    cardiac_signal_source: str | None = None,
    acquisition_number: int | None = None,
    number_of_temporal_positions: int | None = None,
    number_of_frames: int = 1,
    samples_per_pixel: int = 1,
    photometric_interpretation: str = "MONOCHROME2",
    pixel_data: bytes | None = None,
    pixel_data_byte_length_override: int | None = None,
    transfer_syntax_uid: str = TRANSFER_SYNTAX_EXPLICIT_VR_LE,
    sop_class_uid: str = "1.2.840.10008.5.1.4.1.1.2",
    bits_allocated: int = 16,
    bits_stored: int = 16,
    high_bit: int = 15,
    study_date: str | None = None,
) -> Path:
    """Write one deterministic Explicit-VR-LE DICOM instance with pydicom."""
    meta = FileMetaDataset()
    meta.FileMetaInformationVersion = b"\x00\x01"
    meta.MediaStorageSOPClassUID = UID(sop_class_uid)
    meta.MediaStorageSOPInstanceUID = UID(sop_instance_uid)
    meta.TransferSyntaxUID = UID(transfer_syntax_uid)
    meta.ImplementationClassUID = UID("1.2.826.0.1.3680043.8.498.77.99")
    meta.ImplementationVersionName = "CCTA001A"
    dataset = FileDataset(
        str(path), {}, file_meta=meta, preamble=b"\x00" * 128
    )
    dataset.SOPClassUID = sop_class_uid
    dataset.SOPInstanceUID = sop_instance_uid
    dataset.Modality = modality
    dataset.StudyInstanceUID = study_instance_uid
    dataset.SeriesInstanceUID = series_instance_uid
    dataset.FrameOfReferenceUID = frame_of_reference_uid
    dataset.ImageType = list(image_type)
    dataset.SeriesDescription = series_description if series_description is not None else (
        f"cardiac ccta {temporal_position}" if temporal_position else "cardiac ccta"
    )
    if body_part_examined is not None:
        dataset.BodyPartExamined = body_part_examined
    if protocol_name is not None:
        dataset.ProtocolName = protocol_name
    dataset.ImagePositionPatient = list(position)
    dataset.ImageOrientationPatient = [float(item) for item in orientation.split("\\")]
    dataset.PixelSpacing = [float(item) for item in spacing.split("\\")]
    dataset.SliceThickness = float(thickness)
    dataset.Rows = rows
    dataset.Columns = columns
    dataset.BitsAllocated = bits_allocated
    dataset.BitsStored = bits_stored
    dataset.HighBit = high_bit
    dataset.PixelRepresentation = pixel_representation
    dataset.SamplesPerPixel = samples_per_pixel
    dataset.PhotometricInterpretation = photometric_interpretation
    if temporal_position_index is not None:
        dataset.TemporalPositionIndex = temporal_position_index
    if trigger_time is not None:
        dataset.TriggerTime = trigger_time
    if cardiac_synchronization_technique is not None:
        dataset.CardiacSynchronizationTechnique = cardiac_synchronization_technique
    if cardiac_signal_source is not None:
        dataset.CardiacSignalSource = cardiac_signal_source
    if acquisition_number is not None:
        dataset.AcquisitionNumber = acquisition_number
    if number_of_temporal_positions is not None:
        dataset.NumberOfTemporalPositions = number_of_temporal_positions
    if contrast_bolus_agent is not None:
        dataset.ContrastBolusAgent = contrast_bolus_agent
    if contrast_bolus_volume is not None:
        dataset.ContrastBolusVolume = contrast_bolus_volume
    if contrast_bolus_start_time is not None:
        dataset.ContrastBolusStartTime = contrast_bolus_start_time
    if contrast_bolus_stop_time is not None:
        dataset.ContrastBolusStopTime = contrast_bolus_stop_time
    if contrast_flow_rate is not None:
        dataset.ContrastFlowRate = contrast_flow_rate
    if contrast_flow_duration is not None:
        dataset.ContrastFlowDuration = contrast_flow_duration
    if rescale_slope is not None:
        dataset.RescaleSlope = rescale_slope
    if rescale_intercept is not None:
        dataset.RescaleIntercept = rescale_intercept
    if rescale_type is not None:
        dataset.RescaleType = rescale_type
    if study_date is not None:
        dataset.StudyDate = study_date
    expected_bytes = (
        rows * columns * max(bits_allocated // 8, 1) * number_of_frames * samples_per_pixel
    )
    pixels = (
        pixel_data
        if pixel_data is not None
        else (_lcg(pixel_seed) * ((expected_bytes // len(_lcg(pixel_seed))) + 1))[
            :expected_bytes
        ]
    )
    if pixel_data_byte_length_override is not None:
        if pixel_data_byte_length_override < len(pixels):
            pixels = pixels[:pixel_data_byte_length_override]
        else:
            pixels = pixels + b"\x00" * (pixel_data_byte_length_override - len(pixels))
    if number_of_frames != 1:
        dataset.NumberOfFrames = number_of_frames
    dataset.PixelData = pixels
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_as(str(path), write_like_original=False)
    return path


def build_synthetic_study(
    dest: Path,
    *,
    phases: tuple[str, ...] = ("75% R-R",),
    slices_per_phase: int = 3,
    rows: int = 4,
    columns: int = 4,
    include_corrupt_file: bool = False,
    include_foreign_study_file: bool = False,
) -> Path:
    """Build a deterministic multi-phase synthetic study directory."""
    root = Path(dest)
    root.mkdir(parents=True, exist_ok=True)
    for phase_index, label in enumerate(phases, start=1):
        series_uid = SERIES_UID_TEMPLATE.format(phase=phase_index)
        for slice_index in range(slices_per_phase):
            build_instance(
                root / f"phase{phase_index}_slice{slice_index}.dcm",
                sop_instance_uid=SOP_UID_TEMPLATE.format(phase=phase_index, slice=slice_index),
                series_instance_uid=series_uid,
                position=(0.0, 0.0, slice_index * SLICE_STEP_MM),
                rows=rows,
                columns=columns,
                temporal_position=label if len(phases) > 1 else None,
                temporal_position_index=phase_index if len(phases) > 1 else None,
                protocol_name=label if len(phases) > 1 else None,
                contrast_bolus_agent="iohexol" if len(phases) > 1 else None,
                contrast_bolus_volume=50.0 if len(phases) > 1 else None,
                pixel_seed=100 * phase_index + slice_index,
            )
    if include_corrupt_file:
        (root / "broken.dcm").write_bytes(b"not a dicom file at all" * 10)
    if include_foreign_study_file:
        build_instance(
            root / "foreign.dcm",
            sop_instance_uid=SOP_UID_TEMPLATE.format(phase=99, slice=0),
            series_instance_uid=SERIES_UID_TEMPLATE.format(phase=99),
            study_instance_uid="1.2.826.0.1.3680043.8.498.77.555",
            position=(0.0, 0.0, 50.0),
            temporal_position=None,
            pixel_seed=999,
        )
    return root


@dataclass(frozen=True)
class VolumeFixtureCase:
    """Manifest entry for one deterministic VOLUME-001 stratum."""

    fixture_id: str
    orientation: str
    spacing: str
    normal: str
    origin: str
    pixels: str
    paths: tuple[Path, ...]


VOLUME_001_STRATA: tuple[tuple[str, ...], ...] = tuple(
    product(
        ("axis-aligned", "oblique"),
        ("isotropic", "anisotropic"),
        ("standard", "reversed"),
        ("zero", "nonzero"),
        ("unsigned", "signed", "rescaled-fractional"),
    )
)


def build_volume_fixture_matrix(dest: Path) -> tuple[VolumeFixtureCase, ...]:
    """Write the complete deterministic 2×2×2×2×3 VOLUME-001 ingest grid."""

    root = Path(dest)
    cases: list[VolumeFixtureCase] = []
    for ordinal, (
        orientation_kind,
        spacing_kind,
        normal_kind,
        origin_kind,
        pixel_kind,
    ) in enumerate(VOLUME_001_STRATA):
        fixture_id = f"VOLUME-001-{ordinal:02d}"
        if orientation_kind == "axis-aligned":
            orientation = "1\\0\\0\\0\\1\\0"
            normal_vector = (0.0, 0.0, 1.0)
        else:
            orientation = "0.8660254\\0\\0.5\\0\\1\\0"
            normal_vector = (-0.5, 0.0, 0.8660254)
        if normal_kind == "reversed":
            components = orientation.split("\\")
            orientation = "\\".join(
                [str(-float(item)) for item in components[:3]] + components[3:]
            )
            normal_vector = cast(
                tuple[float, float, float], tuple(-item for item in normal_vector)
            )
        spacing = "0.5\\0.5" if spacing_kind == "isotropic" else "0.4\\0.7"
        origin = (0.0, 0.0, 0.0) if origin_kind == "zero" else (11.0, -7.0, 3.5)
        rows, columns = 4, 4
        if pixel_kind == "signed":
            values = (np.arange(rows * columns, dtype=np.int16) - 8).astype("<i2")
            representation = 1
            slope = intercept = rescale_type = None
        elif pixel_kind == "rescaled-fractional":
            values = np.arange(rows * columns, dtype=np.uint16).astype("<u2")
            representation = 0
            slope, intercept, rescale_type = 0.5, -100.25, "HU"
        else:
            values = np.arange(rows * columns, dtype=np.uint16).astype("<u2")
            representation = 0
            slope = intercept = rescale_type = None
        paths: list[Path] = []
        series_uid = f"1.2.826.0.1.3680043.8.498.77.20.{ordinal + 1}"
        for slice_index in range(3):
            position = cast(
                tuple[float, float, float],
                tuple(
                    origin[axis] + normal_vector[axis] * float(slice_index)
                    for axis in range(3)
                ),
            )
            paths.append(
                build_instance(
                    root / fixture_id / f"slice-{slice_index}.dcm",
                    sop_instance_uid=(
                        f"1.2.826.0.1.3680043.8.498.77.30."
                        f"{ordinal + 1}.{slice_index + 1}"
                    ),
                    series_instance_uid=series_uid,
                    position=position,
                    rows=rows,
                    columns=columns,
                    orientation=orientation,
                    spacing=spacing,
                    pixel_representation=representation,
                    rescale_slope=slope,
                    rescale_intercept=intercept,
                    rescale_type=rescale_type,
                    pixel_data=values.tobytes(),
                    pixel_seed=100 + ordinal,
                )
            )
        cases.append(
            VolumeFixtureCase(
                fixture_id=fixture_id,
                orientation=orientation_kind,
                spacing=spacing_kind,
                normal=normal_kind,
                origin=origin_kind,
                pixels=pixel_kind,
                paths=tuple(paths),
            )
        )
    return tuple(cases)


@dataclass(frozen=True)
class ClassificationFixtureCase:
    """One row of the complete Part 4 boundary oracle."""

    row_number: int
    protocol_state: str
    contrast_state: str
    phase_state: str
    geometry_state: str
    syntax_state: str
    expected_outcome: str
    expected_reason: str


@dataclass(frozen=True)
class NegativeFixtureCase:
    """One deterministic reader or series-admission abstention fixture."""

    fixture_id: str
    paths: tuple[Path, ...]
    expected_reason: str
    stage: str


_ORACLE_PROTOCOL_STATES: tuple[str, ...] = (
    "CARDIAC_KEYWORD_MATCH",
    "KEYWORD_ABSENT",
    "PROTOCOL_IDENTITY_INVALID",
    "EXCLUDED_IMAGE_TYPE",
)
_ORACLE_CONTRAST_STATES: tuple[str, ...] = (
    "CONTRAST",
    "NON_CONTRAST",
    "UNKNOWN_AGENT_ABSENT",
    "UNKNOWN_UNCORROBORATED_AGENT",
    "CONTRADICTORY",
)
_ORACLE_PHASE_STATES: tuple[str, ...] = (
    "BOUND",
    "UNKNOWN_NO_EVIDENCE",
    "UNKNOWN_ACQUISITION_ONLY",
    "CONFLICT",
    "ECG_AMBIGUOUS",
)
_ORACLE_GEOMETRY_STATES: tuple[str, ...] = ("VALID", "INVALID")
_ORACLE_SYNTAX_STATES: tuple[str, ...] = ("VALID", "INVALID")


def _oracle_outcome(
    protocol: str, contrast: str, phase: str, geometry: str, syntax: str
) -> tuple[str, str]:
    """Independent literal implementation of the closed Part 4 table."""

    if syntax == "INVALID":
        return "NOT_ANALYZABLE", "DICOM_SYNTAX_INVALID"
    if geometry == "INVALID":
        return "NOT_ANALYZABLE", "GEOMETRY_INVALID"
    if protocol == "PROTOCOL_IDENTITY_INVALID":
        return "NOT_ANALYZABLE", "PROTOCOL_UNSUPPORTED"
    if protocol == "EXCLUDED_IMAGE_TYPE":
        return "NOT_ANALYZABLE", "DERIVED_IMAGE_UNSUPPORTED"
    if contrast == "CONTRADICTORY":
        return "NOT_ANALYZABLE", "CONTRAST_CONFLICT"
    if phase == "CONFLICT":
        return "NOT_ANALYZABLE", "PHASE_CONFLICT"
    if phase == "ECG_AMBIGUOUS":
        return "NOT_ANALYZABLE", "PHASE_ECG_AMBIGUOUS"
    if protocol == "KEYWORD_ABSENT":
        return "IMPORTABLE+UNKNOWN", "PROTOCOL_EVIDENCE_ABSENT"
    if contrast == "UNKNOWN_AGENT_ABSENT":
        return "IMPORTABLE+UNKNOWN", "CONTRAST_EVIDENCE_ABSENT"
    if contrast == "UNKNOWN_UNCORROBORATED_AGENT":
        return "IMPORTABLE+UNKNOWN", "CONTRAST_EVIDENCE_INCOMPLETE"
    if phase == "UNKNOWN_NO_EVIDENCE":
        return "IMPORTABLE+UNKNOWN", "PHASE_EVIDENCE_ABSENT"
    if phase == "UNKNOWN_ACQUISITION_ONLY":
        return "IMPORTABLE+UNKNOWN", "PHASE_ACQUISITION_ONLY"
    return "IMPORTABLE+CLASSIFIED", "CLASSIFICATION_COMPLETE"


def classification_oracle_fixtures() -> tuple[ClassificationFixtureCase, ...]:
    """Return the complete 400-row oracle from an independent literal table."""

    cases: list[ClassificationFixtureCase] = []
    row_number = 0
    for protocol, contrast, phase, geometry, syntax in product(
        _ORACLE_PROTOCOL_STATES,
        _ORACLE_CONTRAST_STATES,
        _ORACLE_PHASE_STATES,
        _ORACLE_GEOMETRY_STATES,
        _ORACLE_SYNTAX_STATES,
    ):
        row_number += 1
        expected_outcome, expected_reason = _oracle_outcome(
            protocol, contrast, phase, geometry, syntax
        )
        cases.append(
            ClassificationFixtureCase(
                row_number=row_number,
                protocol_state=protocol,
                contrast_state=contrast,
                phase_state=phase,
                geometry_state=geometry,
                syntax_state=syntax,
                expected_outcome=expected_outcome,
                expected_reason=expected_reason,
            )
        )
    assert len(cases) == 400
    return tuple(cases)


def build_negative_fixture_matrix(dest: Path) -> tuple[NegativeFixtureCase, ...]:
    """Write one deterministic fixture for every 001-A abstention family."""

    root = Path(dest)
    cases: list[NegativeFixtureCase] = []

    def uid(case_number: int) -> tuple[str, str]:
        return (
            f"1.2.826.0.1.3680043.8.498.77.40.{case_number}",
            f"1.2.826.0.1.3680043.8.498.77.41.{case_number}",
        )

    def one(
        fixture_id: str,
        expected_reason: str,
        *,
        stage: str = "reader",
        **kwargs: object,
    ) -> None:
        case_number = len(cases) + 1
        series_uid, sop_uid = uid(case_number)
        path = build_instance(
            root / fixture_id / "instance.dcm",
            sop_instance_uid=sop_uid,
            series_instance_uid=series_uid,
            position=(0.0, 0.0, 0.0),
            **cast(dict[str, Any], kwargs),
        )
        cases.append(
            NegativeFixtureCase(
                fixture_id=fixture_id,
                paths=(path,),
                expected_reason=expected_reason,
                stage=stage,
            )
        )

    one(
        "unsupported-transfer-syntax",
        "UNSUPPORTED_TRANSFER_SYNTAX",
        # Deflated Explicit VR LE stays outside the admitted set (Implicit VR
        # LE itself is admitted since DICOM_IMPLICIT_VR_LE_ADMISSION_001).
        transfer_syntax_uid="1.2.840.10008.1.2.1.99",
    )
    one(
        "non-ct-sop",
        "PROTOCOL_UNSUPPORTED",
        sop_class_uid="1.2.840.10008.5.1.4.1.1.4",
    )
    one(
        "excluded-image-type",
        "DERIVED_IMAGE_UNSUPPORTED",
        image_type=("ORIGINAL", "PRIMARY", "MPR"),
    )
    one(
        "pixel-byte-length-mismatch",
        "DICOM_SYNTAX_INVALID",
        pixel_data_byte_length_override=2,
    )
    one("multiframe", "DICOM_SYNTAX_INVALID", number_of_frames=2)

    def pair(
        fixture_id: str,
        expected_reason: str,
        *,
        positions: tuple[tuple[float, float, float], ...],
        frame_uids: tuple[str, ...] | None = None,
        orientations: tuple[str, ...] | None = None,
    ) -> None:
        case_number = len(cases) + 1
        series_uid, _ = uid(case_number)
        paths: list[Path] = []
        for index, position in enumerate(positions):
            paths.append(
                build_instance(
                    root / fixture_id / f"slice-{index}.dcm",
                    sop_instance_uid=(
                        f"1.2.826.0.1.3680043.8.498.77.42."
                        f"{case_number}.{index + 1}"
                    ),
                    series_instance_uid=series_uid,
                    position=position,
                    frame_of_reference_uid=(
                        frame_uids[index] if frame_uids is not None else FRAME_UID
                    ),
                    orientation=(
                        orientations[index]
                        if orientations is not None
                        else DEFAULT_ORIENTATION
                    ),
                )
            )
        cases.append(
            NegativeFixtureCase(
                fixture_id=fixture_id,
                paths=tuple(paths),
                expected_reason=expected_reason,
                stage="stack",
            )
        )

    pair(
        "duplicate-positions",
        "SERIES_STACK_INVALID",
        positions=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    )
    pair(
        "gantry-tilt",
        "SERIES_STACK_INVALID",
        positions=((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        orientations=(
            DEFAULT_ORIENTATION,
            "1\\0\\0\\0.01\\0.99995\\0",
        ),
    )
    pair(
        "nonuniform-gaps",
        "SERIES_STACK_INVALID",
        positions=((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 0.0, 3.0)),
    )
    pair(
        "missing-slice",
        "SERIES_INCOMPLETE",
        positions=((0.0, 0.0, 0.0), (0.0, 0.0, 2.0)),
    )
    pair(
        "mixed-frame-identity",
        "IDENTITY_MISMATCH",
        positions=((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        frame_uids=(FRAME_UID, f"{FRAME_UID}.1"),
    )
    return tuple(cases)


def counter_id_factory(seed: int):
    """Deterministic id sequence so golden runs reproduce byte-exactly."""
    from itertools import count

    numbers = count()
    def factory() -> str:
        return f"ID{seed:03d}{next(numbers):05d}"

    return factory


__all__ = [
    "DEFAULT_ORIENTATION",
    "DEFAULT_SPACING",
    "DEFAULT_THICKNESS",
    "FRAME_UID",
    "SERIES_UID_TEMPLATE",
    "SLICE_STEP_MM",
    "STUDY_UID",
    "SOP_UID_TEMPLATE",
    "build_instance",
    "build_negative_fixture_matrix",
    "build_synthetic_study",
    "build_volume_fixture_matrix",
    "classification_oracle_fixtures",
    "counter_id_factory",
    "ClassificationFixtureCase",
    "NegativeFixtureCase",
    "VolumeFixtureCase",
    "VOLUME_001_STRATA",
]
