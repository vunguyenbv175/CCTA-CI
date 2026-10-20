"""MPR reslice() floating-point exception parity (Codex review finding).

``reslice`` must preserve the caller's NumPy floating-point exception policy:
under a caller-side ``np.errstate(under='raise')`` the vectorized kernel
raises the same closed ``MPRAbstentionError`` the scalar baseline raises,
instead of returning a pixel.  Under the default policy the returned pixel
is bit-identical to the scalar baseline.
"""

from __future__ import annotations

import numpy as np
import pytest

from apps.workstation.volume_loader import CANONICAL_FLOAT32_CODEC
from core.coordinates import CoordinateTransform, Handedness, SpaceKind, Units
from visualization.mpr import (
    MPRAbstentionError,
    _sample_trilinear,
    _world_pixel_center,
    _world_to_voxel,
    reslice,
)

_IDENTITY_MATRIX = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)

# Exact Codex repro: identity WORLD->VOXEL transform, (2,2,2) float32 source
# filled with 1e-38, AXIAL origin (0.1,0.1,0.1), unit spacing, (1,1)
# TRILINEAR output.
_REPRO_ORIGIN = (0.1, 0.1, 0.1)


def _identity_world_to_voxel() -> CoordinateTransform:
    return CoordinateTransform(
        transform_id="mpr-errstate-parity-w2v",
        from_space=SpaceKind.WORLD,
        to_space=SpaceKind.VOXEL,
        method="mpr-errstate-parity-v1",
        matrix=_IDENTITY_MATRIX,
        frame_of_reference_uid="1.2.840.for.mpr-parity",
        series_instance_uid="1.2.840.series.mpr-parity",
        direction_cosines=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        translation=(0.0, 0.0, 0.0),
        handedness=Handedness.LPS,
        units=Units.MM,
    )


def _repro_source() -> np.ndarray:
    return np.full((2, 2, 2), 1e-38, dtype=np.float32)


def _scalar_baseline_pixel(source: np.ndarray) -> float:
    """One staged scalar pixel: the pre-vectorization baseline semantics."""
    world = _world_pixel_center(_REPRO_ORIGIN, (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), 1.0, 1.0, 0, 0)
    voxel = _world_to_voxel(_IDENTITY_MATRIX, world)
    return float(_sample_trilinear(source, voxel))


def _repro_reslice(source: np.ndarray, transform: CoordinateTransform) -> tuple[np.ndarray, str]:
    return reslice(
        source,
        transform,
        plane="AXIAL",
        origin=_REPRO_ORIGIN,
        in_plane_spacing_mm=(1.0, 1.0),
        matrix_size=(1, 1),
        method="TRILINEAR",
        out_of_bounds_fill_value=-1000.0,
    )


def test_reslice_preserves_caller_under_raise_parity() -> None:
    """Caller under='raise' abstains identically on both paths (finding)."""
    source = _repro_source()
    transform = _identity_world_to_voxel()
    with np.errstate(under="raise"):
        with pytest.raises(MPRAbstentionError) as scalar_exc:
            _scalar_baseline_pixel(source)
    assert (scalar_exc.value.reason_code, scalar_exc.value.detail_code) == (
        "MPR_ARITHMETIC_INVALID",
        "NONFINITE_FLOAT32",
    )
    ambient = np.geterr()
    with np.errstate(under="raise"):
        with pytest.raises(MPRAbstentionError) as vectorized_exc:
            _repro_reslice(source, transform)
    assert (vectorized_exc.value.reason_code, vectorized_exc.value.detail_code) == (
        scalar_exc.value.reason_code,
        scalar_exc.value.detail_code,
    )
    assert np.geterr() == ambient


def test_reslice_default_policy_pixel_matches_scalar_baseline() -> None:
    """Default errstate still returns the scalar-baseline pixel, bit-identical."""
    source = _repro_source()
    transform = _identity_world_to_voxel()
    expected = _scalar_baseline_pixel(source)
    assert np.isfinite(expected)
    pixels, codec = _repro_reslice(source, transform)
    assert codec == CANONICAL_FLOAT32_CODEC
    assert pixels.shape == (1, 1)
    assert pixels.tobytes() == np.asarray([[expected]], dtype=np.dtype("<f4")).tobytes()
