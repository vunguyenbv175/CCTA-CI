"""Frozen PNG transport encoder (VIEWER_001_SPEC Part 5).

Byte-exact pin, verified first-hand against the CMR byte anchor
``services/workstation/frames.py::encode_png`` (the normative procedure):

- 8-byte PNG signature ``89 50 4E 47 0D 0A 1A 0A`` precedes IHDR;
- RGB24, 8-bit depth, colour type 2 (truecolor), compression method 0,
  filter method 0, interlace 0; IHDR packed ``>IIBBBBB`` as
  ``(width=columns, height=rows, 8, 2, 0, 0, 0)``;
- scanlines in row-major top-to-bottom order, each prefixed with filter byte
  ``0x00``;
- the complete compressed stream in EXACTLY ONE IDAT chunk (zlib level 6,
  default wbits/memLevel, ``Z_DEFAULT_STRATEGY``, a single ``Z_FINISH``
  pass);
- CRC-32 (IEEE, ``zlib.crc32 & 0xFFFFFFFF``) per chunk over kind+data;
- chunk order IHDR, IDAT, IEND; no ancillary chunks.

DETERMINISM SCOPE (honest boundary): "identical input produces identical
bytes" holds within ONE pinned zlib runtime — it is NOT guaranteed across
zlib versions by level alone.  ``zlib.ZLIB_VERSION`` is therefore recorded in
the freeze bundle AND in every frame record's provenance parameters.  PNG is
a transport/render encoding; the SEMANTIC identity of a frame is its LAYER-G
binding + ``wl_applied`` + the decoded PIXEL digest (the SHA-256 over the
decoded RGB24 array after the transfer and BEFORE PNG encoding).  Cross-
runtime reopen identity is out of contract and is never claimed.
"""

from __future__ import annotations

import struct
import zlib
from typing import Final

import numpy as np

PNG_SIGNATURE: Final[bytes] = b"\x89PNG\r\n\x1a\n"
ZLIB_LEVEL: Final[int] = 6
ZLIB_WBITS: Final[int] = zlib.MAX_WBITS
ZLIB_MEM_LEVEL: Final[int] = zlib.DEF_MEM_LEVEL
ZLIB_STRATEGY: Final[int] = zlib.Z_DEFAULT_STRATEGY
ZLIB_VERSION_PIN: Final[str] = zlib.ZLIB_VERSION
PNG_ENCODER_ID: Final[str] = "PNG_RGB24_FROZEN_V1"


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    """One chunk: length, kind+data, CRC-32 over kind+data."""

    payload = kind + data
    return (
        struct.pack(">I", len(data))
        + payload
        + struct.pack(">I", zlib.crc32(payload) & 0xFFFFFFFF)
    )


def encode_png(canvas: np.ndarray) -> bytes:
    """Encode a contiguous RGB24 canvas with the frozen PNG parameters."""

    array = np.asarray(canvas)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("PNG input must be an HxWx3 uint8 RGB array")
    height, width, _channels = array.shape
    if width < 1 or height < 1:
        raise ValueError("PNG input must have positive width and height")
    rgb = np.ascontiguousarray(array)
    # Row-major top-to-bottom scanlines, each prefixed with filter byte 0x00.
    scanlines = b"".join(
        b"\x00" + rgb[row].tobytes(order="C") for row in range(height)
    )
    # EXACTLY ONE IDAT: explicit frozen parameters, a single Z_FINISH pass.
    compressor = zlib.compressobj(
        level=ZLIB_LEVEL,
        method=zlib.DEFLATED,
        wbits=ZLIB_WBITS,
        memLevel=ZLIB_MEM_LEVEL,
        strategy=ZLIB_STRATEGY,
    )
    compressed = compressor.compress(scanlines) + compressor.flush(zlib.Z_FINISH)
    # IHDR: width=columns, height=rows, 8, 2 (truecolor), 0, 0, 0.
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return PNG_SIGNATURE + b"".join(
        (
            _png_chunk(b"IHDR", ihdr),
            _png_chunk(b"IDAT", compressed),
            _png_chunk(b"IEND", b""),
        )
    )


def pixel_digest(canvas: np.ndarray) -> str:
    """The semantic pixel digest: SHA-256 over the decoded RGB24 bytes.

    Defined over the decoded RGB24 byte array AFTER the Part 4.1 transfer
    function and BEFORE PNG encoding (spec Part 5 DETERMINISM SCOPE; derived
    and checked at reopen, never a stored schema field).
    """

    import hashlib

    array = np.asarray(canvas)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("pixel digest requires an HxWx3 uint8 RGB array")
    return hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()


__all__ = [
    "PNG_ENCODER_ID",
    "PNG_SIGNATURE",
    "ZLIB_LEVEL",
    "ZLIB_MEM_LEVEL",
    "ZLIB_STRATEGY",
    "ZLIB_VERSION_PIN",
    "ZLIB_WBITS",
    "encode_png",
    "pixel_digest",
]
