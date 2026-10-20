"""VIEWER_001 independent analytic oracle fixtures.

This module is an INDEPENDENT ORACLE: no expected value here is produced by
the implementation path under test.  The frozen arithmetic of the transfer
function is re-derived from the VIEWER_001_SPEC text with explicit struct
casts, and the PNG encoder here is written independently (mirroring the CMR
``services/workstation/frames.py`` byte anchor reproduced from the frozen
parameter list, not imported from the code under test).  The accepted
001-B/001-C discipline: expectations come only from this module or from
closed-form analytic evaluation.
"""

from __future__ import annotations

import hashlib
import math
import struct
import zlib
from typing import Final

import numpy as np

# ---------------------------------------------------------------------------
# Part 4 — the frozen preset table (spec text, transcribed independently)
# ---------------------------------------------------------------------------

PRESET_TABLE: Final[tuple[tuple[str, float, float], ...]] = (
    ("CT_MEDIASTINUM", 40.0, 400.0),
    ("CT_VESSEL_CONTRAST", 300.0, 800.0),
    ("CT_LUNG", -600.0, 1500.0),
    ("CT_BONE", 300.0, 1500.0),
    ("CT_SOFT_TISSUE", 50.0, 350.0),
)

ACTION_SET: Final[tuple[str, ...]] = (
    "SET_SLICE",
    "SET_PLANE",
    "SET_WL_PRESET",
    "SET_WL_CUSTOM",
    "CLEAR_WL",
    "SET_VIEWPORT_MATRIX",
    "OPEN_QC_PANEL",
    "REQUEST_CORRECTION",
    "SET_OVERLAY_VISIBILITY",
    "REQUEST_CENTERLINE_DECISION",
    "SELECT_VESSEL",
)

REJECTION_CODES: Final[tuple[str, ...]] = (
    "DUPLICATE_JSON_KEYS",
    "NON_CANONICAL_ENCODING",
    "UNKNOWN_ACTION",
    "UNKNOWN_PARAM",
    "VIEW_MISMATCH",
    "NONCE_REUSE",
    "LINEAGE_GAP",
    "INDEX_OUT_OF_RANGE",
    "UNKNOWN_PRESET",
    "ZOOM_NON_POSITIVE",
    "MATRIX_NON_FINITE",
    "GRADE_NOT_IN_ENUM",
    "JUSTIFICATION_NOT_IN_ENUM",
    "SUBJECT_REF_INVALID",
    "SESSION_NOT_FOUND",
    "SESSION_OPEN_FAILED",
)

QUALITY_GRADES: Final[tuple[str, ...]] = (
    "GOOD",
    "ACCEPTABLE",
    "LIMITED",
    "NOT_ANALYZABLE",
)


# ---------------------------------------------------------------------------
# Independent binary32 arithmetic (explicit struct casts; NOT the product code)
# ---------------------------------------------------------------------------

def f32(value: float) -> float:
    """One explicit IEEE-754 binary32 cast via struct (independent of numpy)."""

    return struct.unpack("<f", struct.pack("<f", value))[0]


def oracle_window_bounds(center: float, width: float) -> tuple[float, float]:
    """Spec step 1: binary64 evaluation, then ONE binary32 cast of each bound."""

    lo64 = float(center) - float(width) / 2.0
    hi64 = float(center) + float(width) / 2.0
    return (f32(lo64), f32(hi64))


def oracle_identity_window(minimum: float, maximum: float) -> tuple[float, float]:
    """Raw identity window: binary64 center/width, then one binary32 cast."""

    center64 = (float(minimum) + float(maximum)) / 2.0
    width64 = float(maximum) - float(minimum)
    return (f32(center64), f32(width64))


def oracle_window_byte(s: float, lo: float, hi: float) -> int:
    """The frozen transfer function, re-derived independently from spec text.

    1. ordered clamps; 2. binary32 subtract then divide (each rounded to
    binary32 immediately, left-to-right, no FMA); 3. one binary32 multiply;
    4. binary64 add of 0.5 then floor; 5. grayscale replication.
    """

    if s <= lo:
        t = 0.0
    elif s >= hi:
        t = 1.0
    else:
        numerator = f32(f32(s) - f32(lo))
        denominator = f32(f32(hi) - f32(lo))
        t = f32(numerator / denominator)
    u = f32(f32(255.0) * f32(t))
    u_add = float(u) + 0.5
    b = math.floor(u_add)
    if b < 0:
        return 0
    if b > 255:
        return 255
    return int(b)


def oracle_window_bytes(
    pixels: np.ndarray, center: float, width: float
) -> np.ndarray:
    """The full RGB24 canvas, evaluated with :func:`oracle_window_byte`."""

    lo, hi = oracle_window_bounds(center, width)
    rows, columns = pixels.shape
    canvas = np.zeros((rows, columns, 3), dtype=np.uint8)
    for row in range(rows):
        for column in range(columns):
            value = pixels[row, column]
            if isinstance(value, np.generic):
                value = value.item()
            b = oracle_window_byte(float(value), lo, hi)
            canvas[row, column, 0] = b
            canvas[row, column, 1] = b
            canvas[row, column, 2] = b
    return canvas


def oracle_exact_extrema(pixels: np.ndarray) -> tuple[float, float]:
    """EXACT extrema over decoded values (int16 exact; float32 via binary64)."""

    minimum = None
    maximum = None
    for value in pixels.flat:
        number = float(value.item() if isinstance(value, np.generic) else value)
        if minimum is None or number < minimum:
            minimum = number
        if maximum is None or number > maximum:
            maximum = number
    assert minimum is not None and maximum is not None
    return (minimum, maximum)


# ---------------------------------------------------------------------------
# Independent PNG encoder (frozen parameter list; written from the spec text)
# ---------------------------------------------------------------------------

PNG_SIGNATURE: Final[bytes] = b"\x89PNG\r\n\x1a\n"


def oracle_chunk(kind: bytes, data: bytes) -> bytes:
    length = struct.pack(">I", len(data))
    body = kind + data
    crc = struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
    return length + body + crc


def oracle_compress(scanlines: bytes) -> bytes:
    """EXACTLY ONE IDAT: level 6, default wbits/memLevel, Z_DEFAULT_STRATEGY,
    a single Z_FINISH pass — via an independently constructed compressor."""

    compressor = zlib.compressobj(
        6, zlib.DEFLATED, 15, 8, zlib.Z_DEFAULT_STRATEGY
    )
    return compressor.compress(scanlines) + compressor.flush(zlib.Z_FINISH)


def oracle_png(canvas: np.ndarray) -> bytes:
    """The frozen PNG bytes for one RGB24 canvas, independently derived."""

    array = np.ascontiguousarray(canvas)
    height, width, _ = array.shape
    scanlines = b"".join(
        b"\x00" + array[row].tobytes(order="C") for row in range(height)
    )
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return PNG_SIGNATURE + b"".join(
        (
            oracle_chunk(b"IHDR", ihdr),
            oracle_chunk(b"IDAT", oracle_compress(scanlines)),
            oracle_chunk(b"IEND", b""),
        )
    )


def oracle_png_sha256(canvas: np.ndarray) -> str:
    return hashlib.sha256(oracle_png(canvas)).hexdigest()


def oracle_canvas_digest(canvas: np.ndarray) -> str:
    """The semantic pixel digest: SHA-256 over decoded RGB24 bytes."""

    return hashlib.sha256(np.ascontiguousarray(canvas).tobytes(order="C")).hexdigest()


def png_chunk_layout(png: bytes) -> list[tuple[bytes, int]]:
    """Parse (kind, length) pairs — an independent structural check."""

    if not png.startswith(PNG_SIGNATURE):
        raise AssertionError("missing PNG signature")
    position = len(PNG_SIGNATURE)
    layout: list[tuple[bytes, int]] = []
    while position < len(png):
        length = struct.unpack(">I", png[position : position + 4])[0]
        kind = png[position + 4 : position + 8]
        data_end = position + 8 + length
        crc_expected = struct.unpack(
            ">I", png[data_end : data_end + 4]
        )[0]
        if zlib.crc32(kind + png[position + 8 : data_end]) & 0xFFFFFFFF != crc_expected:
            raise AssertionError(f"CRC mismatch in chunk {kind!r}")
        layout.append((kind, length))
        position = data_end + 4
    return layout


# ---------------------------------------------------------------------------
# Canonical-frame helper: a deterministic 2-D scalar field with known extrema
# ---------------------------------------------------------------------------

def sample_pixels(rows: int = 4, columns: int = 4) -> np.ndarray:
    """A deterministic pixel ramp with exact extrema 0..3000 (int16-safe)."""

    values = np.zeros((rows, columns), dtype=np.int16)
    for row in range(rows):
        for column in range(columns):
            index = row * columns + column
            values[row, column] = np.int16((index * 97) % 3001)
    return values


__all__ = [
    "ACTION_SET",
    "PNG_SIGNATURE",
    "PRESET_TABLE",
    "QUALITY_GRADES",
    "REJECTION_CODES",
    "f32",
    "oracle_canvas_digest",
    "oracle_chunk",
    "oracle_exact_extrema",
    "oracle_identity_window",
    "oracle_png",
    "oracle_png_sha256",
    "oracle_window_bounds",
    "oracle_window_byte",
    "oracle_window_bytes",
    "png_chunk_layout",
    "sample_pixels",
]
