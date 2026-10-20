"""VIEWER_001 V2/V3: the frozen transfer function and the frozen PNG encoder.

Every expected value comes from the independent analytic oracle in
``tests.apps.viewer_fixtures`` — the frozen arithmetic re-derived from the
spec text with explicit struct casts and an independently written PNG
encoder.  NO expected value is produced by the implementation path under test
(the accepted 001-B/001-C discipline).
"""

from __future__ import annotations

import math
import struct
import zlib

import numpy as np
import pytest

from apps.viewer.png_encoder import (
    PNG_SIGNATURE,
    ZLIB_VERSION_PIN,
    encode_png,
    pixel_digest,
)
from apps.viewer.presets import WL_PRESET_IDS, WL_PRESETS, preset_by_id
from apps.viewer.transfer import (
    TRANSFER_FUNCTION_ID,
    exact_extrema,
    identity_window_bounds,
    window_bounds,
    window_byte,
    window_bytes,
)
from tests.apps.viewer_fixtures import (
    PRESET_TABLE,
    f32,
    oracle_identity_window,
    oracle_png,
    oracle_png_sha256,
    oracle_window_bounds,
    oracle_window_byte,
    oracle_window_bytes,
    png_chunk_layout,
    sample_pixels,
)

# ---------------------------------------------------------------------------
# Part 4 — the frozen preset table
# ---------------------------------------------------------------------------


def test_preset_table_is_the_frozen_part_4_table_verbatim() -> None:
    observed = tuple(
        (preset.preset_id, preset.center, preset.width) for preset in WL_PRESETS
    )
    assert observed == PRESET_TABLE
    assert WL_PRESET_IDS == tuple(row[0] for row in PRESET_TABLE)
    for preset_id, center, width in PRESET_TABLE:
        preset = preset_by_id(preset_id)
        assert (preset.preset_id, preset.center, preset.width) == (preset_id, center, width)
        # Canonical float form: the stored values are floats, never ints.
        assert isinstance(preset.center, float) and isinstance(preset.width, float)


# ---------------------------------------------------------------------------
# Part 4.1 — the frozen binary32 transfer function vs the independent oracle
# ---------------------------------------------------------------------------


def test_window_bounds_match_the_independent_oracle_exactly() -> None:
    for _, center, width in PRESET_TABLE:
        assert window_bounds(center, width) == oracle_window_bounds(center, width)
    # Non-representable-in-binary32 windows still agree bit-for-bit.
    for center, width in ((0.1, 0.3), (1e-7, 3.3e-8), (-1024.0, 0.5), (12345.6, 9.9)):
        assert window_bounds(center, width) == oracle_window_bounds(center, width)
        assert window_bounds(center, width) == (
            f32(center - width / 2.0),
            f32(center + width / 2.0),
        )


def test_identity_window_matches_the_oracle_over_exact_extrema() -> None:
    pixels = sample_pixels()
    minimum, maximum = exact_extrema(pixels)
    o_min, o_max = (float(pixels.min()), float(pixels.max()))
    assert (minimum, maximum) == (o_min, o_max)
    assert identity_window_bounds(minimum, maximum) == oracle_identity_window(
        o_min, o_max
    )


def test_window_byte_matches_the_oracle_over_a_dense_int16_grid() -> None:
    for _, center, width in PRESET_TABLE:
        lo, hi = window_bounds(center, width)
        for raw in range(-1200, 2400):
            s = float(np.int16(raw))
            assert window_byte(s, lo, hi) == oracle_window_byte(s, lo, hi)


def test_window_byte_matches_the_oracle_for_float32_samples() -> None:
    # float32 samples: interior values, boundary ulps, and binary32 extrema.
    samples = [0.0, 0.5, -0.5, 1.5, 2.5, -1024.25, 3071.75, 1e6, -1e6]
    rng = np.random.default_rng(20260907)
    samples.extend(float(v) for v in rng.uniform(-2000, 4000, size=512))
    for _, center, width in PRESET_TABLE:
        lo, hi = window_bounds(center, width)
        for raw in samples:
            s = float(np.float32(raw))
            assert window_byte(s, lo, hi) == oracle_window_byte(s, lo, hi)


def test_transfer_clamps_and_rounds_half_up_per_the_pinned_arithmetic() -> None:
    # The mid-point of a symmetric window maps to 128 (255.5 floor).
    assert window_byte(100.0, 0.0, 200.0) == 128
    # s == lo clamps to 0; s == hi clamps to 255 (ordered branches).
    assert window_byte(0.0, 0.0, 200.0) == 0
    assert window_byte(200.0, 0.0, 200.0) == 255
    # Outside the window clamps, never wraps.
    assert window_byte(-10000.0, 0.0, 200.0) == 0
    assert window_byte(10000.0, 0.0, 200.0) == 255
    # Round-half-up: u = 127.5 -> floor(128.0) = 128; u = 126.5 -> 127.
    lo, hi = 0.0, 200.0
    u_half = f32(f32(255.0) * f32(f32(f32(100.0) - f32(lo)) / f32(f32(hi) - f32(lo))))
    assert u_half == 127.5
    assert window_byte(100.0, lo, hi) == 128


def test_degenerate_identity_window_yields_uniform_zero() -> None:
    pixels = np.full((3, 3), np.int16(-42), dtype=np.int16)
    minimum, maximum = exact_extrema(pixels)
    center, width = identity_window_bounds(minimum, maximum)
    canvas = window_bytes(pixels, center, width)
    assert canvas.shape == (3, 3, 3)
    assert int(canvas.max()) == 0  # ordered clamp branches yield byte 0


def test_window_bytes_rgb_replication_matches_the_oracle_canvas() -> None:
    pixels = sample_pixels()
    for _, center, width in PRESET_TABLE:
        ours = window_bytes(pixels, center, width)
        expected = oracle_window_bytes(pixels, center, width)
        assert np.array_equal(ours, expected)
        assert ours.shape == (pixels.shape[0], pixels.shape[1], 3)
        assert ours.dtype == np.uint8
        # Grayscale replication: R == G == B everywhere.
        assert np.array_equal(ours[..., 0], ours[..., 1])
        assert np.array_equal(ours[..., 1], ours[..., 2])


def test_transfer_function_id_is_stable() -> None:
    assert TRANSFER_FUNCTION_ID == "VIEWER_BINARY32_WL_V1"


def test_no_fma_divergence_on_adversarial_windows() -> None:
    # Windows whose (s-lo)/(hi-lo) could tempt an FMA contraction still match
    # the struct-cast oracle bit-for-bit (left-to-right, no FMA).
    for center in (0.0, 0.1, -0.1, 1e-8, 40.0):
        for width in (1e-8, 0.3, 1024.5):
            lo, hi = window_bounds(center, width)
            for raw in (-1024, -1, 0, 1, 1023, 1024):
                s = float(np.int16(raw))
                assert window_byte(s, lo, hi) == oracle_window_byte(s, lo, hi)


# ---------------------------------------------------------------------------
# Part 5 — the frozen PNG encoder
# ---------------------------------------------------------------------------


def test_png_encoder_is_byte_identical_to_the_independent_oracle() -> None:
    pixels = sample_pixels()
    for _, center, width in PRESET_TABLE:
        canvas = window_bytes(pixels, center, width)
        assert encode_png(canvas) == oracle_png(canvas)


def test_png_structure_signature_chunks_and_crcs_are_frozen() -> None:
    canvas = np.zeros((3, 5, 3), dtype=np.uint8)
    canvas[0, 0] = (255, 255, 255)
    png = encode_png(canvas)
    assert png.startswith(PNG_SIGNATURE)
    layout = png_chunk_layout(png)  # raises on any CRC mismatch
    kinds = [kind for kind, _length in layout]
    assert kinds == [b"IHDR", b"IDAT", b"IEND"]  # exactly these, no ancillary
    ihdr_length = layout[0][1]
    assert ihdr_length == 13
    ihdr = png[16 : 16 + 13]
    assert struct.unpack(">IIBBBBB", ihdr) == (5, 3, 8, 2, 0, 0, 0)


def test_png_filter_bytes_are_zero_and_idat_is_exactly_one_chunk() -> None:
    canvas = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    png = encode_png(canvas)
    layout = png_chunk_layout(png)
    idat_chunks = [item for item in layout if item[0] == b"IDAT"]
    assert len(idat_chunks) == 1
    # Reconstruct the filtered scanlines from the single IDAT and verify the
    # filter bytes are 0x00 per row.
    position = png.index(b"IDAT") - 4
    length = struct.unpack(">I", png[position : position + 4])[0]
    raw = zlib.decompress(png[position + 8 : position + 8 + length])
    stride = 3 * 3
    assert len(raw) == (stride + 1) * 2
    assert raw[0] == 0x00 and raw[stride + 1] == 0x00


def test_png_matches_one_shot_zlib_compress_on_the_pinned_runtime() -> None:
    # The explicit compressobj (level 6, default wbits/memLevel,
    # Z_DEFAULT_STRATEGY, single Z_FINISH) must be byte-identical to the
    # one-shot zlib.compress(scanlines, 6) on this pinned runtime.
    canvas = sample_pixels()
    lo, hi = window_bounds(40.0, 400.0)
    for raw in range(0, 3001, 37):
        _ = window_byte(float(raw), lo, hi)  # exercise the path
    expected_png = oracle_png(oracle_window_bytes(canvas, 40.0, 400.0))
    assert encode_png(window_bytes(canvas, 40.0, 400.0)) == expected_png


def test_zlib_version_pin_is_recorded() -> None:
    assert ZLIB_VERSION_PIN == zlib.ZLIB_VERSION
    assert isinstance(ZLIB_VERSION_PIN, str) and ZLIB_VERSION_PIN


def test_pixel_digest_binds_decoded_rgb24_bytes() -> None:
    canvas = sample_pixels()
    mapped = window_bytes(canvas, 40.0, 400.0)
    digest = pixel_digest(mapped)
    import hashlib

    assert digest == hashlib.sha256(np.ascontiguousarray(mapped).tobytes(order="C")).hexdigest()
    # The semantic digest is over the PRE-encoding RGB24 array (it differs
    # from the transport png_sha256 binding).
    assert digest != oracle_png_sha256(mapped)


def test_encoder_rejects_invalid_canvases() -> None:
    with pytest.raises(ValueError):
        encode_png(np.zeros((2, 2), dtype=np.uint8))
    with pytest.raises(ValueError):
        encode_png(np.zeros((2, 2, 4), dtype=np.uint8))
    with pytest.raises(ValueError):
        encode_png(np.zeros((0, 3, 3), dtype=np.uint8))


def test_extrema_rejects_nonfinite_inputs() -> None:
    with pytest.raises(ValueError):
        exact_extrema(np.array([[1.0, float("nan")]], dtype=np.float32))
    with pytest.raises(ValueError):
        window_bytes(
            np.array([[float("inf")]], dtype=np.float32),
            0.0,
            1.0,
        )


def test_non_finite_guard_is_fail_closed_not_clamped() -> None:
    pixels = np.array([[float("nan")]], dtype=np.float32)
    with pytest.raises(ValueError):
        window_bytes(pixels, 0.0, 1.0)


def test_transfer_math_is_over_the_declared_scalar_domain() -> None:
    # The function maps the DECLARED scalar values; it never re-applies a
    # rescale intercept of its own (that binding belongs to the canonical
    # volume envelope, not to the display transform).
    lo, hi = window_bounds(40.0, 400.0)
    assert (lo, hi) == (-160.0, 240.0)
    assert window_byte(-160.0, lo, hi) == 0
    assert window_byte(240.0, lo, hi) == 255
    assert not math.isinf(lo) and not math.isinf(hi)
