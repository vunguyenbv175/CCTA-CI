"""Closed DICOM instance boundary for Imaging Core 001-A.

The only object that leaves this module is :class:`DicomInstance`.  In
particular, a pydicom ``Dataset`` is never part of the canonical data model
or an exception payload.  The reader is deliberately narrow: the 001
admission contract accepts the two uncompressed little-endian CT
image-storage transfer syntaxes (Explicit VR LE and Implicit VR LE, via
pydicom's native parser, no transcoding) and one single-frame, monochrome
pixel representation.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, NoReturn

from core.errors import CanonicalValidationError

TRANSFER_SYNTAX_EXPLICIT_VR_LE: Final[str] = "1.2.840.10008.1.2.1"
TRANSFER_SYNTAX_IMPLICIT_VR_LE: Final[str] = "1.2.840.10008.1.2"
#: The single canonical admitted-syntax set (Rule B): every admission point
#: consults this set, never a local copy.  Both members are uncompressed
#: little-endian CT image-storage syntaxes parsed natively by pydicom; the
#: original TransferSyntaxUID is carried through to the canonical DTO.
ADMITTED_TRANSFER_SYNTAXES: Final[frozenset[str]] = frozenset(
    {TRANSFER_SYNTAX_EXPLICIT_VR_LE, TRANSFER_SYNTAX_IMPLICIT_VR_LE}
)
CT_IMAGE_STORAGE_SOP_CLASS_UID: Final[str] = "1.2.840.10008.5.1.4.1.1.2"
PHOTOMETRIC_INTERPRETATIONS: Final[frozenset[str]] = frozenset(
    {"MONOCHROME1", "MONOCHROME2"}
)
RESCALE_TYPES: Final[frozenset[str]] = frozenset({"HU", "NONE"})
CARDIAC_SIGNAL_SOURCES: Final[frozenset[str]] = frozenset(
    {"ECG", "PULSE", "OTHER", "UNKNOWN"}
)
CARDIAC_SYNCHRONIZATION_TECHNIQUES: Final[frozenset[str]] = frozenset(
    {"PROSPECTIVE", "RETROSPECTIVE", "NONE"}
)
MAX_NORMALIZED_TEXT_LENGTH: Final[int] = 256

# --- Admitted pixel bit-depth tuples (IMAGING_CORE_001_SPEC Part 4.1) -------
# BitsAllocated fixes the 16-bit word container; BitsStored fixes the
# significant bits; HighBit is their top position.  Two arms are admitted:
# 16-stored (unchanged) and the standards-correct 12-stored-in-16 unsigned arm
# admitted by amendment V1.1.  Signed 12-bit is deliberately NOT admitted.
ADMITTED_BITS_ALLOCATED: Final[int] = 16
ADMITTED_16_BIT_BITS_STORED: Final[int] = 16
ADMITTED_16_BIT_HIGH_BIT: Final[int] = 15
ADMITTED_12_BIT_BITS_STORED: Final[int] = 12
ADMITTED_12_BIT_HIGH_BIT: Final[int] = 11
ADMITTED_12_BIT_PIXEL_REPRESENTATION: Final[int] = 0
#: Bits 12-15 of the 16-bit container are padding for the 12-bit arm and must
#: all be zero; any set bit is nonconformant padding (never silently masked).
_12_BIT_PADDING_MASK: Final[int] = 0xF000

_UID_PATTERN: Final[re.Pattern[str]] = re.compile(r"[0-9]+(?:\.[0-9]+)*\Z")
_TIME_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\A(?:[01][0-9]|2[0-3])(?:[0-5][0-9](?:[0-5][0-9](?:\.[0-9]{1,6})?)?)?\Z"
)
_DATE_PATTERN: Final[re.Pattern[str]] = re.compile(r"\A[0-9]{8}\Z")


def admitted_bit_depth_tuple(
    bits_allocated: int | None,
    bits_stored: int | None,
    high_bit: int | None,
    pixel_representation: int | None,
) -> bool:
    """Return the closed Syntax=VALID bit-depth predicate (Part 4.1).

    Admit iff ``bits_allocated == 16 AND ((bits_stored == 16 AND high_bit == 15)
    OR (bits_stored == 12 AND high_bit == 11 AND pixel_representation == 0))``.
    Header semantics only: never dataset-, source-, or filename-dependent.
    """
    if bits_allocated != ADMITTED_BITS_ALLOCATED:
        return False
    if (
        bits_stored == ADMITTED_16_BIT_BITS_STORED
        and high_bit == ADMITTED_16_BIT_HIGH_BIT
    ):
        return True
    if (
        bits_stored == ADMITTED_12_BIT_BITS_STORED
        and high_bit == ADMITTED_12_BIT_HIGH_BIT
        and pixel_representation == ADMITTED_12_BIT_PIXEL_REPRESENTATION
    ):
        return True
    return False


def is_12_bit_stored_tuple(
    bits_allocated: int | None,
    bits_stored: int | None,
    high_bit: int | None,
    pixel_representation: int | None,
) -> bool:
    """True only for the admitted 12-bit-stored arm (unsigned, 16-bit container)."""
    return (
        bits_allocated == ADMITTED_BITS_ALLOCATED
        and bits_stored == ADMITTED_12_BIT_BITS_STORED
        and high_bit == ADMITTED_12_BIT_HIGH_BIT
        and pixel_representation == ADMITTED_12_BIT_PIXEL_REPRESENTATION
    )


def stored_padding_bits_are_zero(
    pixel_data: bytes | bytearray | memoryview,
    *,
    bits_stored: int | None,
    high_bit: int | None,
) -> bool:
    """Return True when no unused high bit is set in any stored 16-bit word.

    For the 12-bit arm, bits 12-15 of every stored little-endian word must be
    zero (equivalently every stored value is in ``0..2**12-1``).  For every
    other tuple the check is trivially satisfied, so the caller's tuple
    predicate remains the sole admission authority.  An odd byte length is not
    conformant and therefore reports False.
    """
    if bits_stored != ADMITTED_12_BIT_BITS_STORED or high_bit != ADMITTED_12_BIT_HIGH_BIT:
        return True
    data = bytes(pixel_data)
    if len(data) % 2 != 0:
        return False
    # Each little-endian word's high byte carries bits 8-15; bits 12-15 are its
    # top nibble.  max() over the byte slice runs at C speed and short-circuits
    # nothing, so a single scan covers the whole frame.
    return max(data[1::2], default=0) <= 0x0F


class DicomAbstentionError(CanonicalValidationError):
    """Typed, fail-closed rejection at the DICOM adapter boundary."""

    def __init__(
        self,
        reason_code: str,
        *,
        detail_code: str | None = None,
        study_instance_uid: str | None = None,
        series_instance_uid: str | None = None,
        sop_instance_uid: str | None = None,
    ) -> None:
        self.reason_code = reason_code
        self.detail_code = detail_code
        self.study_instance_uid = study_instance_uid
        self.series_instance_uid = series_instance_uid
        self.sop_instance_uid = sop_instance_uid
        codes = [reason_code]
        if detail_code and detail_code != reason_code:
            codes.append(detail_code)
        # Never include a path, Dataset repr, or free-text DICOM value.
        super().__init__(codes)


class DicomReadError(DicomAbstentionError):
    """Alias-level type for callers that distinguish read from validation."""


class UnsupportedTransferSyntaxError(DicomAbstentionError):
    """A parsed, syntactically valid but non-admitted transfer syntax."""


class DicomSyntaxError(DicomAbstentionError):
    """Malformed or out-of-envelope DICOM syntax."""


DicomValidationError = DicomAbstentionError


def normalize_text(value: object, *, field: str = "text") -> str | None:
    """Normalize one permitted free-text value at the adapter boundary.

    ``None`` and an empty normalized value represent an absent optional DICOM
    value.  A non-string or overlong value is a typed syntax rejection.  The
    original value is never included in the error.
    """

    if value is None:
        return None
    if not isinstance(value, str):
        raise DicomSyntaxError("DICOM_SYNTAX_INVALID")
    normalized = re.sub(r"\s+", " ", value.strip().casefold())
    if len(normalized) > MAX_NORMALIZED_TEXT_LENGTH:
        raise DicomSyntaxError("DICOM_SYNTAX_INVALID")
    return normalized or None


def _optional_normalized_text(value: object, *, field: str) -> str | None:
    return normalize_text(value, field=field)


def _canonical_control_text(value: object, *, allow_explicit_empty: bool = False) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise DicomSyntaxError("DICOM_SYNTAX_INVALID")
    normalized = re.sub(r"\s+", " ", value.strip()).upper()
    if not normalized and not allow_explicit_empty:
        return None
    if len(normalized) > MAX_NORMALIZED_TEXT_LENGTH:
        raise DicomSyntaxError("DICOM_SYNTAX_INVALID")
    return normalized


def _finite_float(value: object, *, allow_none: bool = True) -> float | None:
    if value is None and allow_none:
        return None
    if isinstance(value, bool):
        raise DicomSyntaxError("DICOM_SYNTAX_INVALID")
    try:
        converted = float(str(value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise DicomSyntaxError("DICOM_SYNTAX_INVALID") from exc
    if not math.isfinite(converted):
        raise DicomSyntaxError("DICOM_SYNTAX_INVALID")
    return converted


def _optional_positive_int(
    value: object,
    *,
    allow_zero: bool = False,
    reason_code: str = "DICOM_SYNTAX_INVALID",
) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise DicomSyntaxError("DICOM_SYNTAX_INVALID")
    text = str(value).strip()
    if re.fullmatch(r"\+?[0-9]+", text) is None:
        raise DicomSyntaxError(reason_code)
    try:
        converted = int(text)
    except (TypeError, ValueError, OverflowError) as exc:
        raise DicomSyntaxError(reason_code) from exc
    if converted < (0 if allow_zero else 1):
        raise DicomSyntaxError(reason_code)
    return converted


def _time_seconds(value: str | None) -> float | None:
    """Return a comparable DICOM TM value without retaining extra metadata."""

    if value is None:
        return None
    if not _TIME_PATTERN.fullmatch(value):
        return None
    hour = int(value[0:2])
    minute = int(value[2:4]) if len(value) >= 4 else 0
    second_end = 6 if len(value) >= 6 else 4
    second = int(value[4:second_end]) if len(value) >= 6 else 0
    fraction = 0.0
    if "." in value:
        fraction = float(f"0.{value.split('.', 1)[1]}")
    return hour * 3600.0 + minute * 60.0 + second + fraction


def is_valid_dicom_time(value: str | None) -> bool:
    return value is not None and _time_seconds(value) is not None


def timing_pair_is_valid(start: str | None, stop: str | None) -> bool:
    start_seconds = _time_seconds(start)
    stop_seconds = _time_seconds(stop)
    return (
        start_seconds is not None
        and stop_seconds is not None
        and start_seconds < stop_seconds
    )


@dataclass(frozen=True, slots=True)
class DicomInstance:
    """The closed, allowlist-reduced canonical representation of one slice."""

    study_instance_uid: str
    series_instance_uid: str
    sop_instance_uid: str
    frame_of_reference_uid: str
    modality: str
    sop_class_uid: str
    image_type: tuple[str, ...]
    series_description: str | None
    body_part_examined: str | None
    protocol_name: str | None
    image_position_patient: tuple[float, float, float]
    image_orientation_patient: tuple[float, float, float, float, float, float]
    pixel_spacing: tuple[float, float]
    slice_thickness_mm: float | None
    rows: int
    columns: int
    pixel_representation: int
    bits_allocated: int
    bits_stored: int
    high_bit: int
    rescale_slope: float
    rescale_intercept: float
    rescale_type: str | None
    contrast_bolus_agent: str | None
    contrast_bolus_volume_ml: float | None
    contrast_bolus_start_time: str | None
    contrast_bolus_stop_time: str | None
    contrast_flow_rate: float | None
    contrast_flow_duration: float | None
    temporal_position_index: int | None
    trigger_time_ms: float | None
    cardiac_synchronization_technique: str | None
    cardiac_signal_source: str | None
    acquisition_number: int | None
    number_of_temporal_positions: int | None
    transfer_syntax_uid: str
    pixel_data: bytes
    pixel_data_byte_length: int
    number_of_frames: int
    samples_per_pixel: int
    photometric_interpretation: str
    study_date: str | None

    def __post_init__(self) -> None:
        # Canonicalize only values that are not free text; raw free text is
        # normalized before construction and is never retained here.
        for name in ("series_description", "body_part_examined", "protocol_name"):
            value = normalize_text(getattr(self, name), field=name)
            object.__setattr__(self, name, value)
        for name in ("contrast_bolus_agent",):
            object.__setattr__(
                self, name, _optional_normalized_text(getattr(self, name), field=name)
            )
        for name in ("cardiac_synchronization_technique", "cardiac_signal_source", "rescale_type"):
            value = _canonical_control_text(
                getattr(self, name),
                allow_explicit_empty=name == "cardiac_synchronization_technique",
            )
            object.__setattr__(self, name, value)
        for name in ("photometric_interpretation", "modality"):
            value = _canonical_control_text(getattr(self, name))
            object.__setattr__(self, name, value or "")
        object.__setattr__(
            self,
            "image_type",
            tuple(str(item).strip().upper() for item in self.image_type),
        )
        object.__setattr__(self, "pixel_data", bytes(self.pixel_data))
        object.__setattr__(
            self,
            "contrast_bolus_start_time",
            _clean_time(self.contrast_bolus_start_time),
        )
        object.__setattr__(
            self,
            "contrast_bolus_stop_time",
            _clean_time(self.contrast_bolus_stop_time),
        )
        object.__setattr__(self, "study_date", _clean_date(self.study_date))

    # Names used by the DICOM standard remain available as read-only aliases;
    # they are properties, not additional persisted DTO fields.
    @property
    def slice_thickness(self) -> float | None:
        return self.slice_thickness_mm

    @property
    def contrast_bolus_volume(self) -> float | None:
        return self.contrast_bolus_volume_ml

    @property
    def trigger_time(self) -> float | None:
        return self.trigger_time_ms

    @property
    def temporal_position(self) -> int | None:
        return self.temporal_position_index


def _clean_time(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise DicomSyntaxError("DICOM_SYNTAX_INVALID")
    cleaned = value.strip()
    return cleaned or None


def _clean_date(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise DicomSyntaxError("DICOM_SYNTAX_INVALID")
    cleaned = value.strip()
    return cleaned or None


def _is_uid(value: object) -> bool:
    if not isinstance(value, str) or not 1 <= len(value) <= 64:
        return False
    return _UID_PATTERN.fullmatch(value) is not None


def _single(
    ds: Any,
    keyword: str,
    *,
    required: bool = False,
    reason_code: str = "DICOM_SYNTAX_INVALID",
) -> Any:
    if keyword not in ds:
        if required:
            raise DicomSyntaxError(reason_code)
        return None
    value = getattr(ds, keyword)
    if isinstance(value, (list, tuple)):
        raise DicomSyntaxError(reason_code)
    return value


def _numeric_tuple(
    ds: Any,
    keyword: str,
    *,
    required: bool = False,
    reason_code: str = "DICOM_SYNTAX_INVALID",
) -> tuple[float, ...]:
    if keyword not in ds:
        if required:
            raise DicomSyntaxError(reason_code)
        return ()
    value = getattr(ds, keyword)
    if isinstance(value, str):
        parts = tuple(part.strip() for part in value.split("\\"))
    elif isinstance(value, SequenceABC) and not isinstance(value, (bytes, bytearray)):
        parts = tuple(value)
    else:
        raise DicomSyntaxError(reason_code)
    try:
        converted = tuple(float(item) for item in parts)
    except (TypeError, ValueError, OverflowError) as exc:
        raise DicomSyntaxError(reason_code) from exc
    if any(not math.isfinite(item) for item in converted):
        raise DicomSyntaxError(reason_code)
    return converted


def _text(ds: Any, keyword: str, *, required: bool = False) -> str | None:
    value = _single(ds, keyword, required=required)
    if value is None:
        return None
    if not isinstance(value, str):
        raise DicomSyntaxError("DICOM_SYNTAX_INVALID")
    return value


def _uid(ds: Any, keyword: str, *, required: bool = True) -> str:
    raw = _text(ds, keyword, required=required)
    if raw is None:
        raise DicomSyntaxError("DICOM_SYNTAX_INVALID")
    value = raw.strip().strip("\x00")
    if not _is_uid(value):
        raise DicomSyntaxError("DICOM_SYNTAX_INVALID")
    return value


def _frame_of_reference_uid(ds: Any) -> str:
    """Read the geometry identity with its closed detail-code mapping."""

    raw = _text(ds, "FrameOfReferenceUID")
    if raw is None:
        raise DicomAbstentionError(
            "GEOMETRY_INVALID", detail_code="IDENTITY_MISMATCH"
        )
    value = raw.strip().strip("\x00")
    if not _is_uid(value):
        raise DicomAbstentionError(
            "GEOMETRY_INVALID", detail_code="IDENTITY_MISMATCH"
        )
    return value


def _number(ds: Any, keyword: str, *, required: bool = False) -> float | None:
    value = _single(ds, keyword, required=required)
    return _finite_float(value)


def _integer(ds: Any, keyword: str, *, required: bool = False) -> int | None:
    value = _single(ds, keyword, required=required)
    return _optional_positive_int(value, allow_zero=True)


def _multi_text(ds: Any, keyword: str, *, required: bool = False) -> tuple[str, ...]:
    if keyword not in ds:
        if required:
            raise DicomSyntaxError("DICOM_SYNTAX_INVALID")
        return ()
    value = getattr(ds, keyword)
    if isinstance(value, str):
        values = tuple(part.strip() for part in value.split("\\"))
    elif isinstance(value, SequenceABC) and not isinstance(value, (bytes, bytearray)):
        values = tuple(value)
    else:
        raise DicomSyntaxError("DICOM_SYNTAX_INVALID")
    if not values or any(not isinstance(item, str) or not item.strip() for item in values):
        raise DicomSyntaxError("DICOM_SYNTAX_INVALID")
    return tuple(item.strip().upper() for item in values)


def _candidate_identity(ds: Any) -> tuple[str | None, str | None, str | None]:
    def candidate(keyword: str) -> str | None:
        try:
            value = getattr(ds, keyword, None)
        except Exception:
            return None
        if value is None or not isinstance(value, str):
            return None
        cleaned = value.strip().strip("\x00")
        return cleaned if _is_uid(cleaned) else None

    return (
        candidate("StudyInstanceUID"),
        candidate("SeriesInstanceUID"),
        candidate("SOPInstanceUID"),
    )


def _raise_with_identity(
    reason_code: str,
    *,
    detail_code: str | None = None,
    identity: tuple[str | None, str | None, str | None] = (None, None, None),
) -> NoReturn:
    raise DicomAbstentionError(
        reason_code,
        detail_code=detail_code,
        study_instance_uid=identity[0],
        series_instance_uid=identity[1],
        sop_instance_uid=identity[2],
    )


def read_dicom_instance(path: Path) -> DicomInstance:
    """Read and reduce one admitted DICOM instance with pydicom.

    The function intentionally does not expose the Dataset, and all exception
    text is a closed reason code.  It also reads only ``PixelData`` bytes; it
    never invokes a decoder or materializes a pydicom ``pixel_array``.
    """

    try:
        import pydicom
        from pydicom.errors import InvalidDicomError
    except ImportError as exc:  # pragma: no cover - dependency admission gate
        raise DicomAbstentionError("DICOM_SYNTAX_INVALID") from exc

    try:
        dataset = pydicom.dcmread(str(path), force=False, defer_size=None)
    except InvalidDicomError as exc:
        # "magic" is a closed parser reason retained for the WS-02 legacy
        # fixture assertion; no filename or input text is included.
        raise DicomSyntaxError("DICOM_SYNTAX_INVALID", detail_code="DICOM_MAGIC_INVALID") from exc
    except Exception as exc:
        raise DicomSyntaxError("DICOM_SYNTAX_INVALID") from exc

    identity = _candidate_identity(dataset)
    file_meta = getattr(dataset, "file_meta", None)
    raw_transfer = getattr(file_meta, "TransferSyntaxUID", None) if file_meta is not None else None
    transfer = str(raw_transfer).strip().strip("\x00") if raw_transfer is not None else None
    if transfer is None or not _is_uid(transfer):
        _raise_with_identity("DICOM_SYNTAX_INVALID", identity=identity)
    if transfer not in ADMITTED_TRANSFER_SYNTAXES:
        raise UnsupportedTransferSyntaxError(
            "DICOM_SYNTAX_INVALID",
            detail_code="UNSUPPORTED_TRANSFER_SYNTAX",
            study_instance_uid=identity[0],
            series_instance_uid=identity[1],
            sop_instance_uid=identity[2],
        )

    try:
        study_uid = _uid(dataset, "StudyInstanceUID")
        series_uid = _uid(dataset, "SeriesInstanceUID")
        sop_uid = _uid(dataset, "SOPInstanceUID")
        frame_uid = _frame_of_reference_uid(dataset)
        modality = _text(dataset, "Modality", required=True)
        sop_class_uid = _uid(dataset, "SOPClassUID")
        image_type = _multi_text(dataset, "ImageType", required=True)
    except DicomAbstentionError as exc:
        if (
            exc.study_instance_uid is None
            and exc.series_instance_uid is None
            and exc.sop_instance_uid is None
        ):
            _raise_with_identity(
                exc.reason_code,
                detail_code=exc.detail_code,
                identity=identity,
            )
        raise
    except (AttributeError, KeyError, TypeError, ValueError):
        _raise_with_identity("DICOM_SYNTAX_INVALID", identity=identity)

    if modality is None or not modality.strip():
        _raise_with_identity("DICOM_SYNTAX_INVALID", identity=(study_uid, series_uid, sop_uid))
    if modality.strip() != "CT":
        raise DicomAbstentionError(
            "PROTOCOL_UNSUPPORTED",
            study_instance_uid=study_uid,
            series_instance_uid=series_uid,
            sop_instance_uid=sop_uid,
        )
    if sop_class_uid != CT_IMAGE_STORAGE_SOP_CLASS_UID:
        raise DicomAbstentionError(
            "PROTOCOL_UNSUPPORTED",
            study_instance_uid=study_uid,
            series_instance_uid=series_uid,
            sop_instance_uid=sop_uid,
        )
    image_type_components = set(image_type)
    excluded = {
        "DERIVED",
        "LOCALIZER",
        "SECONDARY",
        "MPR",
        "MIP",
        "MINIP",
        "3D",
        "VOLUME_RENDERED",
        "VR",
        "THICKSLAB",
        "SLAB",
        "PROJECTION",
        "REFORMATTED",
        "RESAMPLED",
        "DYNAMIC",
        "SPECTRAL",
        "DUAL_ENERGY",
    }
    if (
        len(image_type) < 2
        or image_type[0] != "ORIGINAL"
        or image_type[1] != "PRIMARY"
        or image_type_components.intersection(excluded)
    ):
        raise DicomAbstentionError(
            "DERIVED_IMAGE_UNSUPPORTED",
            study_instance_uid=study_uid,
            series_instance_uid=series_uid,
            sop_instance_uid=sop_uid,
        )

    try:
        position = _numeric_tuple(
            dataset,
            "ImagePositionPatient",
            required=True,
            reason_code="GEOMETRY_INVALID",
        )
        orientation = _numeric_tuple(
            dataset,
            "ImageOrientationPatient",
            required=True,
            reason_code="GEOMETRY_INVALID",
        )
        spacing = _numeric_tuple(
            dataset,
            "PixelSpacing",
            required=True,
            reason_code="GEOMETRY_INVALID",
        )
        if len(position) != 3 or len(orientation) != 6 or len(spacing) != 2:
            raise DicomSyntaxError("GEOMETRY_INVALID")
        if any(not math.isfinite(item) for item in (*position, *orientation, *spacing)):
            raise DicomSyntaxError("GEOMETRY_INVALID")
        if any(item <= 0.0 for item in spacing):
            raise DicomSyntaxError("GEOMETRY_INVALID")
        rows = _optional_positive_int(
            _single(dataset, "Rows", required=True, reason_code="GEOMETRY_INVALID"),
            reason_code="GEOMETRY_INVALID",
        )
        columns = _optional_positive_int(
            _single(
                dataset, "Columns", required=True, reason_code="GEOMETRY_INVALID"
            ),
            reason_code="GEOMETRY_INVALID",
        )
        if rows is None or columns is None:
            raise DicomSyntaxError("GEOMETRY_INVALID")
        slice_thickness = _number(dataset, "SliceThickness")

        bits_allocated = _optional_positive_int(
            _single(dataset, "BitsAllocated", required=True), allow_zero=True
        )
        bits_stored = _optional_positive_int(
            _single(dataset, "BitsStored", required=True), allow_zero=True
        )
        high_bit = _optional_positive_int(
            _single(dataset, "HighBit", required=True), allow_zero=True
        )
        pixel_representation = _optional_positive_int(
            _single(dataset, "PixelRepresentation", required=True), allow_zero=True
        )
        samples_per_pixel = _optional_positive_int(
            _single(dataset, "SamplesPerPixel", required=True), allow_zero=True
        )
        number_of_frames_value = _single(dataset, "NumberOfFrames")
        number_of_frames = (
            1
            if number_of_frames_value is None
            else _optional_positive_int(number_of_frames_value, allow_zero=True)
        )
        if number_of_frames is None:
            raise DicomSyntaxError("DICOM_SYNTAX_INVALID")
        photometric = _text(dataset, "PhotometricInterpretation", required=True)
        if photometric is None:
            raise DicomSyntaxError("DICOM_SYNTAX_INVALID")
        photometric = photometric.strip()
        if photometric not in PHOTOMETRIC_INTERPRETATIONS:
            raise DicomAbstentionError(
                "DICOM_SYNTAX_INVALID",
                study_instance_uid=study_uid,
                series_instance_uid=series_uid,
                sop_instance_uid=sop_uid,
            )
        # IMAGING_CORE_001 Part 4.1: the closed bit-depth tuple predicate.
        if not admitted_bit_depth_tuple(
            bits_allocated, bits_stored, high_bit, pixel_representation
        ):
            raise DicomAbstentionError(
                "DICOM_SYNTAX_INVALID",
                study_instance_uid=study_uid,
                series_instance_uid=series_uid,
                sop_instance_uid=sop_uid,
            )
        if pixel_representation not in {0, 1} or samples_per_pixel != 1 or number_of_frames != 1:
            raise DicomAbstentionError(
                "DICOM_SYNTAX_INVALID",
                study_instance_uid=study_uid,
                series_instance_uid=series_uid,
                sop_instance_uid=sop_uid,
            )
        pixel_value = _single(dataset, "PixelData", required=True)
        if not isinstance(pixel_value, (bytes, bytearray, memoryview)):
            raise DicomSyntaxError("DICOM_SYNTAX_INVALID")
        pixel_data = bytes(pixel_value)
        expected_bytes = rows * columns * 2
        if len(pixel_data) != expected_bytes:
            raise DicomAbstentionError(
                "DICOM_SYNTAX_INVALID",
                study_instance_uid=study_uid,
                series_instance_uid=series_uid,
                sop_instance_uid=sop_uid,
            )
        # 12-bit arm conformance: unused bits 12-15 of every stored word must be
        # zero.  Nonconformant padding ABSTAINS; it is never silently masked.
        if not stored_padding_bits_are_zero(
            pixel_data, bits_stored=bits_stored, high_bit=high_bit
        ):
            raise DicomAbstentionError(
                "DICOM_SYNTAX_INVALID",
                study_instance_uid=study_uid,
                series_instance_uid=series_uid,
                sop_instance_uid=sop_uid,
            )

        slope = _number(dataset, "RescaleSlope")
        intercept = _number(dataset, "RescaleIntercept")
        rescale_type = _canonical_control_text(_text(dataset, "RescaleType"))
        if rescale_type is not None and rescale_type not in RESCALE_TYPES:
            raise DicomAbstentionError(
                "DICOM_SYNTAX_INVALID",
                study_instance_uid=study_uid,
                series_instance_uid=series_uid,
                sop_instance_uid=sop_uid,
            )
        contrast_volume = _number(dataset, "ContrastBolusVolume")
        flow_rate = _number(dataset, "ContrastFlowRate")
        flow_duration = _number(dataset, "ContrastFlowDuration")
        if any(
            value is not None and value < 0.0
            for value in (contrast_volume, flow_rate, flow_duration)
        ):
            raise DicomSyntaxError("DICOM_SYNTAX_INVALID")
        trigger_time = _number(dataset, "TriggerTime")
        if trigger_time is not None and trigger_time < 0.0:
            raise DicomSyntaxError("DICOM_SYNTAX_INVALID")
        contrast_start = _clean_time(_text(dataset, "ContrastBolusStartTime"))
        contrast_stop = _clean_time(_text(dataset, "ContrastBolusStopTime"))
        for time_value in (contrast_start, contrast_stop):
            if time_value is not None and not is_valid_dicom_time(time_value):
                raise DicomSyntaxError("DICOM_SYNTAX_INVALID")
        temporal_index = _optional_positive_int(_single(dataset, "TemporalPositionIndex"))
        acquisition_number = _optional_positive_int(_single(dataset, "AcquisitionNumber"))
        number_temporal = _optional_positive_int(
            _single(dataset, "NumberOfTemporalPositions")
        )
        cardiac_technique = _canonical_control_text(
            _text(dataset, "CardiacSynchronizationTechnique"),
            allow_explicit_empty=True,
        )
        cardiac_source = _canonical_control_text(_text(dataset, "CardiacSignalSource"))
        if cardiac_source is not None and cardiac_source not in CARDIAC_SIGNAL_SOURCES:
            raise DicomSyntaxError("DICOM_SYNTAX_INVALID")
        study_date = _clean_date(_text(dataset, "StudyDate"))
        if study_date is not None and _DATE_PATTERN.fullmatch(study_date) is None:
            raise DicomSyntaxError("DICOM_SYNTAX_INVALID")
        series_description = _optional_normalized_text(
            _text(dataset, "SeriesDescription"), field="SeriesDescription"
        )
        body_part = _optional_normalized_text(
            _text(dataset, "BodyPartExamined"), field="BodyPartExamined"
        )
        protocol_name = _optional_normalized_text(
            _text(dataset, "ProtocolName"), field="ProtocolName"
        )
        agent = _optional_normalized_text(
            _text(dataset, "ContrastBolusAgent"), field="ContrastBolusAgent"
        )
    except DicomAbstentionError as exc:
        if (
            exc.study_instance_uid is None
            and exc.series_instance_uid is None
            and exc.sop_instance_uid is None
        ):
            _raise_with_identity(
                exc.reason_code,
                detail_code=exc.detail_code,
                identity=(study_uid, series_uid, sop_uid),
            )
        raise
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
        _raise_with_identity("DICOM_SYNTAX_INVALID", identity=(study_uid, series_uid, sop_uid))

    return DicomInstance(
        study_instance_uid=study_uid,
        series_instance_uid=series_uid,
        sop_instance_uid=sop_uid,
        frame_of_reference_uid=frame_uid,
        modality=modality,
        sop_class_uid=sop_class_uid,
        image_type=image_type,
        series_description=series_description,
        body_part_examined=body_part,
        protocol_name=protocol_name,
        image_position_patient=position,
        image_orientation_patient=orientation,
        pixel_spacing=spacing,
        slice_thickness_mm=slice_thickness,
        rows=rows,
        columns=columns,
        pixel_representation=pixel_representation or 0,
        bits_allocated=bits_allocated or 0,
        bits_stored=bits_stored or 0,
        high_bit=high_bit or 0,
        rescale_slope=1.0 if slope is None else slope,
        rescale_intercept=0.0 if intercept is None else intercept,
        rescale_type=rescale_type,
        contrast_bolus_agent=agent,
        contrast_bolus_volume_ml=contrast_volume,
        contrast_bolus_start_time=contrast_start,
        contrast_bolus_stop_time=contrast_stop,
        contrast_flow_rate=flow_rate,
        contrast_flow_duration=flow_duration,
        temporal_position_index=temporal_index,
        trigger_time_ms=trigger_time,
        cardiac_synchronization_technique=cardiac_technique,
        cardiac_signal_source=cardiac_source,
        acquisition_number=acquisition_number,
        number_of_temporal_positions=number_temporal,
        transfer_syntax_uid=transfer,
        pixel_data=pixel_data,
        pixel_data_byte_length=len(pixel_data),
        number_of_frames=number_of_frames,
        samples_per_pixel=samples_per_pixel or 0,
        photometric_interpretation=photometric,
        study_date=study_date,
    )


__all__ = [
    "ADMITTED_12_BIT_BITS_STORED",
    "ADMITTED_12_BIT_HIGH_BIT",
    "ADMITTED_12_BIT_PIXEL_REPRESENTATION",
    "ADMITTED_16_BIT_BITS_STORED",
    "ADMITTED_16_BIT_HIGH_BIT",
    "ADMITTED_BITS_ALLOCATED",
    "ADMITTED_TRANSFER_SYNTAXES",
    "CARDIAC_SIGNAL_SOURCES",
    "CARDIAC_SYNCHRONIZATION_TECHNIQUES",
    "CT_IMAGE_STORAGE_SOP_CLASS_UID",
    "DicomAbstentionError",
    "DicomInstance",
    "DicomReadError",
    "DicomSyntaxError",
    "DicomValidationError",
    "MAX_NORMALIZED_TEXT_LENGTH",
    "PHOTOMETRIC_INTERPRETATIONS",
    "RESCALE_TYPES",
    "TRANSFER_SYNTAX_EXPLICIT_VR_LE",
    "TRANSFER_SYNTAX_IMPLICIT_VR_LE",
    "UnsupportedTransferSyntaxError",
    "admitted_bit_depth_tuple",
    "is_12_bit_stored_tuple",
    "is_valid_dicom_time",
    "normalize_text",
    "read_dicom_instance",
    "stored_padding_bits_are_zero",
    "timing_pair_is_valid",
]
