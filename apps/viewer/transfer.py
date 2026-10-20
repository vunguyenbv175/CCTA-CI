"""Frozen binary32 window/level transfer function (VIEWER_001_SPEC Part 4.1).

The arithmetic is pinned exactly, in order:

1. Window bounds ``lo = center - width/2`` and ``hi = center + width/2``,
   both evaluated in binary64 and then cast to binary32 ONCE.
2. Ordered clamp branches: ``s <= lo`` -> 0.0; ``s >= hi`` -> 1.0; else
   ``t = (binary32(s) - lo) / (hi - lo)`` — one subtract, one divide, each
   rounded to binary32 immediately (left-to-right; NO FMA).
3. ``u = binary32(255.0) * t`` (single multiply, binary32).
4. ``u_add = binary64(u) + 0.5`` (the add in binary64, then floor) —
   ``b = floor(u_add)`` (round-half-up); defensive clamp to [0, 255].
5. Grayscale replication R = G = B = b.  No gamma, no LUT, no palette.

For raw frames (``CLEAR_WL``) the same function is applied with the identity
window over the frame's own pixel extrema: the endpoints are the EXACT extrema
over the decoded MPR pixel values (int16 exactly; float32 over the binary32
values), with ``center = (min + max) / 2`` and ``width = (max - min)``
evaluated in binary64 and then cast to binary32 once.  Degenerate
``max == min`` needs no extra rule: the ordered clamp branches yield a uniform
byte 0.

All arithmetic is over the declared scalar domain (HU), never over stored
values with a different intercept assumption.  No constructed divergence
exists between the binary32 and binary64 add in step 4; binary64 is the
pinned form.
"""

from __future__ import annotations

import math
from typing import Final

import numpy as np

TRANSFER_FUNCTION_ID: Final[str] = "VIEWER_BINARY32_WL_V1"
#: The single explicit cast used for every binary32 rounding step.
_F32_DTYPE: Final[np.dtype[np.float32]] = np.dtype("<f4")


def to_binary32(value: float | np.floating) -> np.float32:
    """One explicit IEEE binary32 cast (the only rounding primitive here)."""

    result = np.asarray(float(value), dtype=_F32_DTYPE).reshape(())
    return np.float32(result)


def window_bounds(center: float, width: float) -> tuple[float, float]:
    """Step 1: binary64 evaluation, then ONE binary32 cast of each bound."""

    lo64 = float(center) - float(width) / 2.0
    hi64 = float(center) + float(width) / 2.0
    return (float(to_binary32(lo64)), float(to_binary32(hi64)))


def identity_window_bounds(
    minimum: float, maximum: float
) -> tuple[float, float]:
    """Raw-frame center/width over the EXACT decoded-pixel extrema.

    ``center = (min + max) / 2`` and ``width = (max - min)`` are evaluated in
    binary64 and cast to binary32 once (the same rule as step 1 of the
    transfer function).  Returns ``(center, width)`` as the recorded
    ``wl_applied`` values.
    """

    center64 = (float(minimum) + float(maximum)) / 2.0
    width64 = float(maximum) - float(minimum)
    return (float(to_binary32(center64)), float(to_binary32(width64)))


def window_byte(s: float, lo: float, hi: float) -> int:
    """The 8-bit channel byte for scalar *s* under the frozen transfer."""

    # Ordered clamp branches (comparisons are exact: every operand is a
    # binary32 value held exactly in binary64).
    if s <= lo:
        t = 0.0
    elif s >= hi:
        t = 1.0
    else:
        # One binary32 subtract, then one binary32 divide; each rounded to
        # binary32 immediately, left-to-right, no FMA.  The denominator is
        # the binary32 difference of the two binary32 bounds.
        numerator = to_binary32(float(to_binary32(s) - to_binary32(lo)))
        denominator = to_binary32(float(to_binary32(hi) - to_binary32(lo)))
        t = float(to_binary32(float(numerator / denominator)))
    # Step 3: single binary32 multiply.
    u = to_binary32(to_binary32(255.0) * to_binary32(t))
    # Step 4: the add in binary64, then floor (round-half-up); defensive
    # clamp — the closed form already implies [0, 255].
    u_add = float(u) + 0.5
    b = math.floor(u_add)
    if b < 0:
        return 0
    if b > 255:
        return 255
    return b


def window_bytes(pixels: np.ndarray, center: float, width: float) -> np.ndarray:
    """Map a 2-D scalar array through the frozen transfer to RGB24 bytes.

    Returns an ``(rows, columns, 3)`` uint8 array with R = G = B = b
    (grayscale replication).  Inputs are never mutated.
    """

    array = np.asarray(pixels)
    if array.ndim != 2:
        raise ValueError("window_bytes requires a 2-D (rows, columns) scalar array")
    lo, hi = window_bounds(center, width)
    rows, columns = array.shape
    canvas = np.empty((rows, columns, 3), dtype=np.uint8)
    for row in range(rows):
        for column in range(columns):
            value = array[row, column]
            if isinstance(value, np.generic):
                value = value.item()
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("window_bytes requires finite pixel values")
            b = window_byte(value, lo, hi)
            canvas[row, column, 0] = b
            canvas[row, column, 1] = b
            canvas[row, column, 2] = b
    return canvas


def exact_extrema(pixels: np.ndarray) -> tuple[float, float]:
    """The EXACT extrema over decoded MPR pixel values.

    int16 arrays are exact integers; float32 arrays are reduced over their
    binary32 values (each value exact in binary64).  Returns ``(min, max)``.
    """

    array = np.asarray(pixels)
    if array.ndim != 2 or array.size == 0:
        raise ValueError("exact_extrema requires a non-empty 2-D array")
    if array.dtype.kind not in {"i", "u", "f"}:
        raise ValueError("exact_extrema requires an int/uint/float array")
    if array.dtype.kind == "f":
        for value in array.flat:
            if not math.isfinite(float(value)):
                raise ValueError("exact_extrema requires finite pixel values")
    minimum = float(array.min())
    maximum = float(array.max())
    if not math.isfinite(minimum) or not math.isfinite(maximum):
        raise ValueError("exact_extrema requires finite extrema")
    return (minimum, maximum)


__all__ = [
    "TRANSFER_FUNCTION_ID",
    "exact_extrema",
    "identity_window_bounds",
    "to_binary32",
    "window_bounds",
    "window_byte",
    "window_bytes",
]
