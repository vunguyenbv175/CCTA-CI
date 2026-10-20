"""F30-OPT: halo-blocked smoothing parity (Tier-A exact, no tolerance).

The blocked smoother reuses the repo kernel (:func:`_gaussian_kernel`) and
the repo per-axis convolution (:func:`_convolve_axis`) verbatim on
halo-extended slabs (halo = kernel radius, interior stitched).  Every cell
below asserts BIT-EXACT equality (``np.array_equal``) between the unblocked
and blocked paths, including an explicit boundary-voxel audit band.
"""

from __future__ import annotations

import hashlib
import math
import os
import platform
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from apps.workstation.volume_loader import CanonicalVolumeLoader
from core.coronary import vesselness as vesselness_module
from core.coronary.types import (
    CoronaryRefusalCode,
    CoronaryValidationError,
    VesselnessParameters,
)
from core.coronary.vesselness import (
    _BLOCKED_EIG_SUB_SLICES,
    _BLOCKED_MAX_EXTENDED_SLICES,
    _BLOCKED_MIN_VALID_SLICES,
    _SPACING_INDEX_BY_ARRAY_AXIS,
    _TRIMMED_VALID_ROW,
    _blocked_convolve_axis,
    _frangi_vesselness_unblocked,
    _gaussian_kernel,
    _k_smoothing_radius,
    _nan_propagating_max,
    _ordered_by_magnitude,
    _second_derivatives,
    _slab_bounds,
    _slab_max_s,
    _slab_scaled_derivatives,
    _slab_window,
    _smooth,
    _smooth_blocked,
    _smooth_nblocks,
    _smooth_unblocked,
    _symmetric_eigenvalues,
    frangi_vesselness,
    vessel_mask,
)
from tests.core.coronary.support import (
    analytic_volume,
    phantom_parameters,
    straight_tube_volume,
)

ANISO = (0.427734, 0.427734, 0.625)
ISO = (1.0, 1.0, 1.0)
ANISO_K2 = (1.0, 1.0, 2.0)
SPACINGS = (ANISO, ISO, ANISO_K2)
SIGMAS = (1.0, 2.0, 3.0)
NBLOCKS = (2, 3, 5, 7)

#: Real-data gate (private workspace, never a test fixture): the Case02
#: crop digest below is asserted only when this exact blob is provisioned.
#: A fresh checkout without the workspace skips instead of failing.
_REAL_BLOBS_ROOT = Path("/tmp/opencode/t2_ws/blobs")
_REAL_CROP_DIGEST = "7f17e5594ef84278b6ce84016b559cf22f8cae3c3d7e2245d9148b166820facb"
_REAL_CROP_LOADER = CanonicalVolumeLoader(_REAL_BLOBS_ROOT)
_REAL_CROP_AVAILABLE = _REAL_CROP_LOADER.blob_path(_REAL_CROP_DIGEST).exists()

_CS = 48


def _phantoms_48() -> dict[str, np.ndarray]:
    impulse = np.zeros((_CS, _CS, _CS), dtype=np.float64)
    impulse[_CS // 2, _CS // 2, _CS // 2] = 1.0
    step = np.zeros((_CS, _CS, _CS), dtype=np.float64)
    step[:, :, _CS // 2 :] = 1000.0
    ramp = np.ascontiguousarray(
        np.linspace(0.0, 1000.0, _CS, dtype=np.float64)[:, None, None]
        * np.ones((1, _CS, _CS), dtype=np.float64)
    )
    kk, jj, ii = np.mgrid[0:_CS, 0:_CS, 0:_CS].astype(np.float64)
    sine = np.ascontiguousarray(np.sin(2 * np.pi * (ii / 12.0 + jj / 17.0 + kk / 23.0)) * 500.0)
    rng = np.random.RandomState(20260919)
    noise = np.ascontiguousarray(rng.randn(_CS, _CS, _CS) * 100.0 - 100.0)
    return {
        "impulse": impulse,
        "step": step,
        "ramp": ramp,
        "sine": sine,
        "noise": noise,
    }


def _repo_fixtures_48() -> dict[str, np.ndarray]:
    raw, _, _, _ = analytic_volume()
    return {
        "phantom_raw_int16": raw,
        "tube_i": straight_tube_volume(axis="i", span=(6, 26), center=16, radius_mm=2.0),
        "tube_j": straight_tube_volume(axis="j", span=(6, 26), center=16, radius_mm=2.0),
        "tube_k": straight_tube_volume(axis="k", span=(6, 26), center=16, radius_mm=2.0),
    }


def _shape3(volume: np.ndarray) -> tuple[int, int, int]:
    return (int(volume.shape[0]), int(volume.shape[1]), int(volume.shape[2]))


def _boundary_mask(shape: tuple[int, int, int], radius: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for axis, extent in enumerate(shape):
        take = [slice(None), slice(None), slice(None)]
        take[axis] = slice(0, min(radius, extent))
        mask[tuple(take)] = True
        take[axis] = slice(max(0, extent - radius), extent)
        mask[tuple(take)] = True
    return mask


def _max_radius(spacing: tuple[float, float, float], sigma_mm: float) -> int:
    radii = []
    for axis in range(3):
        sigma_points = sigma_mm / spacing[_SPACING_INDEX_BY_ARRAY_AXIS[axis]]
        radii.append(_gaussian_kernel(sigma_points).shape[0] // 2)
    return int(max(radii))


INPUTS: dict[str, np.ndarray] = {**_repo_fixtures_48(), **_phantoms_48()}


@pytest.mark.parametrize("iname", sorted(INPUTS))
@pytest.mark.parametrize("sigma_mm", SIGMAS)
@pytest.mark.parametrize("spacing", SPACINGS)
@pytest.mark.parametrize("nblocks", NBLOCKS)
def test_blocked_matches_unblocked_exactly(
    iname: str, sigma_mm: float, spacing: tuple[float, float, float], nblocks: int
) -> None:
    volume = INPUTS[iname]
    ref = _smooth_unblocked(volume, spacing, sigma_mm)
    cand = _smooth_blocked(volume, spacing, sigma_mm, nblocks)
    assert cand.dtype == np.float64 == ref.dtype
    assert np.array_equal(ref, cand), (
        f"{iname} sigma={sigma_mm} spacing={spacing} nb={nblocks}: "
        f"max|d|={float(np.abs(ref - cand).max())}"
    )


@pytest.mark.parametrize("iname", sorted(INPUTS))
@pytest.mark.parametrize("sigma_mm", SIGMAS)
@pytest.mark.parametrize("spacing", SPACINGS)
def test_boundary_voxel_band_is_exact(
    iname: str, sigma_mm: float, spacing: tuple[float, float, float]
) -> None:
    """Voxels within max-kernel-radius of any face match bit-exactly."""

    volume = INPUTS[iname]
    radius = _max_radius(spacing, sigma_mm)
    band = _boundary_mask(_shape3(volume), radius)
    assert int(band.sum()) > 0
    for nblocks in (2, 3):
        ref = _smooth_unblocked(volume, spacing, sigma_mm)
        cand = _smooth_blocked(volume, spacing, sigma_mm, nblocks)
        assert np.array_equal(ref[band], cand[band]), (
            f"boundary {iname} sigma={sigma_mm} spacing={spacing} nb={nblocks}"
        )


@pytest.mark.parametrize("nblocks", (2, 3, 5, 8, 64))
def test_block_count_does_not_change_the_result(nblocks: int) -> None:
    """The exact result is independent of the block count (incl. clamping)."""

    volume = _phantoms_48()["noise"]
    ref = _smooth_unblocked(volume, ANISO, 3.0)
    assert np.array_equal(ref, _smooth_blocked(volume, ANISO, 3.0, nblocks))


def test_dispatcher_takes_unblocked_path_below_floor() -> None:
    small = np.zeros((21, 21, 21), dtype=np.float64)
    small[10, 10, 10] = 1.0
    assert _smooth_nblocks(_shape3(small)) == 1
    assert np.array_equal(_smooth(small, ANISO_K2, 1.0), _smooth_unblocked(small, ANISO_K2, 1.0))


def test_dispatcher_takes_blocked_path_above_floor() -> None:
    volume = _repo_fixtures_48()["tube_i"]
    nblocks = _smooth_nblocks(_shape3(volume))
    assert nblocks >= 2
    assert np.array_equal(_smooth(volume, ISO, 2.0), _smooth_blocked(volume, ISO, 2.0, nblocks))
    assert np.array_equal(_smooth(volume, ISO, 2.0), _smooth_unblocked(volume, ISO, 2.0))


def test_blocked_convolve_axis_matches_full_axis() -> None:
    volume = _phantoms_48()["sine"]
    for axis in range(3):
        kernel = _gaussian_kernel(2.0 / ANISO[_SPACING_INDEX_BY_ARRAY_AXIS[axis]])
        from core.coronary.vesselness import _convolve_axis

        assert np.array_equal(
            _convolve_axis(np.asarray(volume, dtype=np.float64), kernel, axis),
            _blocked_convolve_axis(np.asarray(volume, dtype=np.float64), kernel, axis, 3),
        )


#: HOST_PARITY (frozen-profile separation): historical VPS baseline for the
#: Phantom48 response-bytes digest. Preserved verbatim; the VPS host computes
#: exactly these bytes. This literal is DOCUMENTATION, never asserted on a
#: non-VPS host (asserting it unconditionally fails on the PC/Ryzen/WSL2 host,
#: whose libm dispatches the same `np.exp`/`np.arccos` transcendental paths to
#: different last-ulp results).
_VPS_FROZEN_RESPONSE_SHA256 = "4124fb53239302ef25eaf6f20b2817266eca4fb5646ee7e47b449ce1a82ea58b"

#: HOST_PARITY: mechanically-bound pinned attributes of the PC/Ryzen/WSL2 host
#: profile. Each value is compared against the live runtime (kernel cpuinfo,
#: loader confstr, interpreter, numpy); there is no environment override, no
#: calibration/config file, no auto-learn, no fallback. Unknown hosts FAIL
#: CLOSED below — never skip, never pass vacuously.
_HOST_PROFILE_PC_RYZEN_WSL2_CPUINFO_MODEL = "AMD Ryzen 5 5600 6-Core Processor"
_HOST_PROFILE_PC_RYZEN_WSL2_NUMPY = "2.5.2"
_HOST_PROFILE_PC_RYZEN_WSL2_PYTHON = "3.12.15"
_HOST_PROFILE_PC_RYZEN_WSL2_GLIBC = "2.43"
_HOST_PROFILE_PC_RYZEN_WSL2_RESPONSE_SHA256 = (
    "cf890b6330d60a32bd0a7779f14ffc6cd8860bb764c129b031a38f34609fd9f6"
)


def _read_cpuinfo_model() -> str:
    """First `model name` line of /proc/cpuinfo (no env override, no cache)."""

    with open("/proc/cpuinfo", encoding="ascii") as handle:
        for line in handle:
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    raise AssertionError("HOST_PARITY: /proc/cpuinfo has no 'model name' line")


def _read_glibc_version() -> str:
    """glibc release from the runtime loader (no env override, no cache)."""

    _confstr = getattr(os, "confstr", None)
    if _confstr is None:
        raise AssertionError(
            "HOST_PARITY: os.confstr unavailable on this platform - "
            "non-POSIX runtime has no calibration"
        )
    try:
        return str(_confstr("CS_GNU_LIBC_VERSION")).split()[-1]
    except (ValueError, OSError) as exc:
        raise AssertionError(
            f"HOST_PARITY: non-glibc runtime has no calibration ({exc!r})"
        ) from exc


def _phantom48_host_response_digest() -> str:
    """Fail-closed host dispatch: VPS literal documented, PC digest asserted.

    Returns the expected response-bytes digest for THIS host, or raises
    AssertionError on any unknown host (never skip, never pass vacuously).
    """

    cpu_model = _read_cpuinfo_model()
    numpy_version = str(np.__version__)
    python_version = platform.python_version()
    glibc_version = _read_glibc_version()
    if (
        cpu_model == _HOST_PROFILE_PC_RYZEN_WSL2_CPUINFO_MODEL
        and numpy_version == _HOST_PROFILE_PC_RYZEN_WSL2_NUMPY
        and python_version == _HOST_PROFILE_PC_RYZEN_WSL2_PYTHON
        and glibc_version == _HOST_PROFILE_PC_RYZEN_WSL2_GLIBC
    ):
        return _HOST_PROFILE_PC_RYZEN_WSL2_RESPONSE_SHA256
    raise AssertionError(
        "HOST_PARITY: unknown host — no calibrated response digest; "
        f"cpu_model={cpu_model!r} numpy={numpy_version!r} "
        f"python={python_version!r} glibc={glibc_version!r}; "
        f"historical VPS baseline={_VPS_FROZEN_RESPONSE_SHA256}"
    )


def test_fixture_response_digest_is_frozen() -> None:
    """Phantom48 Tier-A digest: dispatcher bytes plus the portable mask count.

    The response-bytes digest is host-calibrated (fail-closed fingerprint;
    VPS baseline preserved verbatim in ``_VPS_FROZEN_RESPONSE_SHA256``); the
    mask count (1071) is portable and asserted unconditionally.
    """

    raw, _, _, _ = analytic_volume()
    parameters = phantom_parameters()
    response = frangi_vesselness(raw, (1.0, 1.0, 1.0), parameters)
    mask = vessel_mask(response, parameters)
    response_sha = hashlib.sha256(
        np.ascontiguousarray(response, dtype=np.float64).tobytes(order="C")
    ).hexdigest()
    assert response_sha == _phantom48_host_response_digest()
    assert int(np.count_nonzero(mask)) == 1071


@pytest.mark.skipif(
    not _REAL_CROP_AVAILABLE,
    reason="real-data-gate-only: provision /tmp/opencode/t2_ws/blobs to run the Case02 crop digest",
)
def test_real_crop_response_digest_is_frozen(tmp_path: Path) -> None:
    """Case02 96^3 crop Tier-A digest (blob read-only, never written)."""

    loader = CanonicalVolumeLoader(Path("/tmp/opencode/t2_ws/blobs"))
    full = np.asarray(loader.read_array(_REAL_CROP_DIGEST))
    crop = np.ascontiguousarray(full[56:152, 323:419, 140:236])
    assert crop.shape == (96, 96, 96)
    parameters = VesselnessParameters(scales_mm=(1.0, 2.0), threshold=0.09)
    response = frangi_vesselness(crop, ANISO, parameters)
    response_sha = hashlib.sha256(
        np.ascontiguousarray(response, dtype=np.float64).tobytes(order="C")
    ).hexdigest()
    assert response_sha == "e722d6daa636e53b143c0595f30e5941a3c1072547c2e5b07a9b61d76e8ebfd0"


# ---------------------------------------------------------------------------
# F30-OPT-C FIX R1 (F3): independent oracles for the slab stages
# ---------------------------------------------------------------------------
#
# Every oracle below is computed from the FROZEN full-volume path
# (``_smooth_unblocked`` + ``_second_derivatives`` + the closed-form
# eigensolve); nothing in this section reads the slab machinery under test.

_Probe = tuple[np.ndarray, tuple[float, float, float], VesselnessParameters]


def _oracle_scaled_derivatives(
    volume: np.ndarray, spacing: tuple[float, float, float], sigma_mm: float
) -> tuple[np.ndarray, ...]:
    """Full-volume smoothed Hessian, ``sigma_mm ** 2`` normalized (six arrays)."""

    smoothed = _smooth_unblocked(volume, spacing, sigma_mm)
    derivatives = _second_derivatives(smoothed, spacing)
    scale = np.float64(sigma_mm) * np.float64(sigma_mm)
    return tuple(np.ascontiguousarray(derivative * scale) for derivative in derivatives)


def _oracle_max_s(scaled: tuple[np.ndarray, ...]) -> float:
    """Full-volume ``max(S)`` of one scale, exactly as ``_frangi_response`` sees it."""

    ordered = _ordered_by_magnitude(_symmetric_eigenvalues(*scaled))
    l1, l2, l3 = ordered[..., 0], ordered[..., 1], ordered[..., 2]
    s = np.sqrt(np.maximum(l1 * l1 + l2 * l2 + l3 * l3, 0.0))
    return float(np.max(s))


def _blocked_max_s(
    volume: np.ndarray,
    spacing: tuple[float, float, float],
    sigma_mm: float,
    *,
    valid: int | None = None,
    halo: int | None = None,
) -> float:
    """The blocked two-level reduction of ``max(S)`` for one scale."""

    resolved_halo = _k_smoothing_radius(sigma_mm, spacing) + 1 if halo is None else halo
    if valid is None:
        valid = max(
            int(_BLOCKED_MIN_VALID_SLICES),
            int(_BLOCKED_MAX_EXTENDED_SLICES) - 2 * resolved_halo,
        )
    best = 0.0
    for start, stop in _slab_bounds(int(volume.shape[0]), valid):
        window, offset = _slab_window(volume, start, stop, resolved_halo)
        scaled = _slab_scaled_derivatives(
            window,
            spacing,
            sigma_mm,
            offset=offset,
            width=stop - start,
        )
        best = _nan_propagating_max(
            best, _slab_max_s(scaled, _TRIMMED_VALID_ROW, stop - start)
        )
    return best


def _impulse_volume(
    shape: tuple[int, int, int],
    positions: tuple[tuple[int, int, int], ...],
    *,
    value: float = 1000.0,
    background: float = -100.0,
) -> np.ndarray:
    volume = np.full(shape, background, dtype=np.float64)
    for k, j, i in positions:
        volume[k, j, i] = value
    return np.ascontiguousarray(volume)


def _probe_cases() -> dict[str, _Probe]:
    """Deterministic F3 probe corpus (bounded sizes; one scale unless stated)."""

    iso = (1.0, 1.0, 1.0)
    aniso_k = (1.0, 1.0, 0.125)
    one = VesselnessParameters(scales_mm=(1.0,), threshold=0.09)
    three = VesselnessParameters(scales_mm=(3.0,), threshold=0.09)
    multi = VesselnessParameters(scales_mm=(1.0, 2.0, 3.0), threshold=0.09)
    cases: dict[str, _Probe] = {}

    # Routing floor: exactly 32767 / 32768 / 32769 voxels.
    for label, shape in (
        ("routing_32767", (7, 31, 151)),
        ("routing_32768", (32, 32, 32)),
        ("routing_32769", (9, 11, 331)),
    ):
        center = (shape[0] // 2, shape[1] // 2, shape[2] // 2)
        cases[label] = (_impulse_volume(shape, (center,)), iso, one)

    # k = 1 / 2 (singleton k axis) with the halo far exceeding the depth.
    cases["k1_large_halo"] = (
        _impulse_volume((1, 256, 128), ((0, 128, 64),), background=0.0),
        aniso_k,
        three,
    )
    cases["k2"] = (
        _impulse_volume((2, 128, 128), ((0, 64, 64), (1, 64, 64))),
        iso,
        one,
    )
    # Singleton j / i.
    cases["singleton_j"] = (
        _impulse_volume((128, 1, 256), ((64, 0, 128),), background=0.0),
        iso,
        one,
    )
    cases["singleton_i"] = (
        _impulse_volume((128, 256, 1), ((64, 128, 0),), background=0.0),
        iso,
        one,
    )

    # Depths around the valid width w (sigma=1, ISO: halo 4, w = 56):
    # w-1, w, w+1, 2w-1, 2w, 2w+1 (the last is a one-row tail slab).
    for depth in (55, 56, 57, 111, 112, 113):
        cases[f"depth_{depth}"] = (
            _impulse_volume((depth, 32, 32), ((depth // 2, 16, 16), (depth - 1, 0, 0))),
            iso,
            one,
        )

    # Asymmetric face / edge / corner / seam impulses.
    cases["asymmetric_impulses"] = (
        _impulse_volume(
            (64, 32, 32),
            ((0, 0, 0), (63, 31, 31), (0, 31, 0), (63, 0, 31), (31, 16, 16), (32, 16, 16)),
        ),
        iso,
        one,
    )
    # Global-S max in a later slab (the first slab is weak).
    cases["max_in_last_slab"] = (
        _impulse_volume((128, 32, 32), ((4, 16, 16), (120, 16, 16))),
        iso,
        one,
    )
    # Multiple scales (per-scale halos and valid widths differ).
    cases["multi_scale"] = (
        _impulse_volume((64, 32, 32), ((32, 16, 16), (0, 0, 0), (63, 31, 31))),
        iso,
        multi,
    )
    # Zero / tiny / overflow finite values.
    cases["zeros"] = (np.zeros((64, 32, 32), dtype=np.float64), iso, one)
    cases["tiny"] = (np.full((64, 32, 32), 1e-300), iso, one)
    overflow = np.ascontiguousarray(np.random.RandomState(20260919).randn(160, 16, 16) * 100.0)
    overflow[0, 0, 0] = 1e160
    cases["overflow_finite"] = (overflow, iso, one)
    axis = np.arange(32, dtype=np.float64)[None, None, :] - 15.5
    axis_j = np.arange(32, dtype=np.float64)[None, :, None] - 15.5
    axis_k = np.arange(32, dtype=np.float64)[:, None, None] - 15.5
    cases["overflow_quadratic"] = (
        np.ascontiguousarray(-1e155 * (axis * axis + axis_j * axis_j + axis_k * axis_k)),
        iso,
        one,
    )
    # dtype / layout variants of one shared 32^3 impulse.
    base = np.full((32, 32, 32), -100, dtype=np.int16)
    base[16, 16, 16] = 1000
    for label, variant in (
        ("dtype_int16", base),
        ("dtype_int32", base.astype(np.int32)),
        ("dtype_float32", base.astype(np.float32)),
        ("dtype_float64", base.astype(np.float64)),
        ("layout_fortran", np.asfortranarray(base)),
        ("layout_reversed", base[::-1]),
    ):
        cases[label] = (variant, iso, one)
    return cases


PROBES = _probe_cases()


def _bytes(array: np.ndarray) -> bytes:
    return np.ascontiguousarray(array, dtype=np.float64).tobytes(order="C")


@pytest.mark.parametrize("iname", sorted(INPUTS))
@pytest.mark.parametrize("sigma_mm", SIGMAS)
@pytest.mark.parametrize("spacing", SPACINGS)
def test_slab_scaled_derivatives_match_full_volume_oracle(
    iname: str, sigma_mm: float, spacing: tuple[float, float, float]
) -> None:
    """All six scaled derivatives of every slab match the full-volume oracle."""

    volume = INPUTS[iname]
    halo = _k_smoothing_radius(sigma_mm, spacing) + 1
    valid = max(int(_BLOCKED_MIN_VALID_SLICES), int(_BLOCKED_MAX_EXTENDED_SLICES) - 2 * halo)
    oracle = _oracle_scaled_derivatives(volume, spacing, sigma_mm)
    assert len(oracle) == 6
    for start, stop in _slab_bounds(int(volume.shape[0]), valid):
        window, offset = _slab_window(volume, start, stop, halo)
        slab = _slab_scaled_derivatives(
            window, spacing, sigma_mm, offset=offset, width=stop - start
        )
        assert len(slab) == 6
        for component, expected in zip(slab, oracle):
            got = component[_TRIMMED_VALID_ROW : _TRIMMED_VALID_ROW + (stop - start)]
            assert np.array_equal(got, expected[start:stop]), (
                f"{iname} sigma={sigma_mm} spacing={spacing} slab=({start},{stop})"
            )


@pytest.mark.parametrize("iname", sorted(INPUTS))
@pytest.mark.parametrize("sigma_mm", SIGMAS)
def test_slab_global_max_s_matches_full_volume_oracle(iname: str, sigma_mm: float) -> None:
    """The two-level blocked max(S) equals the full-volume ``np.max``."""

    volume = INPUTS[iname]
    oracle = _oracle_max_s(_oracle_scaled_derivatives(volume, ANISO, sigma_mm))
    got = _blocked_max_s(volume, ANISO, sigma_mm)
    assert got == oracle or (math.isnan(got) and math.isnan(oracle)), (
        f"{iname} sigma={sigma_mm}: blocked={got!r} oracle={oracle!r}"
    )


@pytest.mark.parametrize("pname", sorted(PROBES))
def test_probe_response_bytes_and_masks_match_reference(pname: str) -> None:
    """Dispatcher bytes, dtype, read-only flag and mask equal the reference."""

    volume, spacing, parameters = PROBES[pname]
    reference = _frangi_vesselness_unblocked(volume, spacing, parameters)
    response = frangi_vesselness(volume, spacing, parameters)
    assert response.dtype == reference.dtype == np.dtype("<f8")
    assert not response.flags.writeable
    assert _bytes(response) == _bytes(reference), pname
    reference_mask = vessel_mask(reference, parameters)
    mask = vessel_mask(response, parameters)
    assert np.array_equal(mask, reference_mask), pname
    assert int(np.count_nonzero(mask)) == int(np.count_nonzero(reference_mask))


def test_routing_sizes_32767_32768_32769_select_the_path_by_voxel_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The dispatcher floor is exact: <32768 unblocked, >=32768 blocked."""

    calls: list[str] = []
    real_blocked = vesselness_module._frangi_vesselness_blocked
    real_unblocked = vesselness_module._frangi_vesselness_unblocked

    def blocked(
        array: np.ndarray,
        spacing: tuple[float, float, float],
        parameters: VesselnessParameters,
    ) -> np.ndarray:
        calls.append("blocked")
        return real_blocked(array, spacing, parameters)

    def unblocked(
        array: np.ndarray,
        spacing: tuple[float, float, float],
        parameters: VesselnessParameters,
    ) -> np.ndarray:
        calls.append("unblocked")
        return real_unblocked(array, spacing, parameters)

    monkeypatch.setattr(vesselness_module, "_frangi_vesselness_blocked", blocked)
    monkeypatch.setattr(vesselness_module, "_frangi_vesselness_unblocked", unblocked)
    parameters = VesselnessParameters(scales_mm=(1.0,), threshold=0.09)
    shapes = ((7, 31, 151), (32, 32, 32), (9, 11, 331))
    assert [int(np.prod(shape)) for shape in shapes] == [32767, 32768, 32769]
    for shape in shapes:
        volume = _impulse_volume(shape, ((shape[0] // 2, shape[1] // 2, shape[2] // 2),))
        response = frangi_vesselness(volume, (1.0, 1.0, 1.0), parameters)
        reference = _frangi_vesselness_unblocked(volume, (1.0, 1.0, 1.0), parameters)
        assert np.array_equal(response, reference)
    assert calls == ["unblocked", "blocked", "blocked"]


def test_global_s_max_in_a_later_slab_matches_reference() -> None:
    """A max(S) located in the last slab still drives the same global ``c``."""

    volume = _impulse_volume((128, 32, 32), ((4, 16, 16), (120, 16, 16)))
    spacing = (1.0, 1.0, 1.0)
    parameters = VesselnessParameters(scales_mm=(1.0,), threshold=0.09)
    reference = _frangi_vesselness_unblocked(volume, spacing, parameters)
    response = frangi_vesselness(volume, spacing, parameters)
    assert np.array_equal(response, reference)
    oracle = _oracle_max_s(_oracle_scaled_derivatives(volume, spacing, 1.0))
    assert _blocked_max_s(volume, spacing, 1.0) == oracle
    assert oracle > 0.0


def test_halo_exceeding_depth_and_row_budget_is_clamped_to_the_volume() -> None:
    """A 73-row halo over a 1-row volume copies one plane, not 147."""

    volume = np.zeros((1, 256, 256), dtype=np.int16)
    halo = _k_smoothing_radius(3.0, (1.0, 1.0, 0.125)) + 1
    assert halo == 73
    assert halo > int(_BLOCKED_MAX_EXTENDED_SLICES)
    window, offset = _slab_window(volume, 0, 1, halo)
    assert window.shape == (1, 256, 256)
    assert offset == 0
    scaled = _slab_scaled_derivatives(window, (1.0, 1.0, 0.125), 3.0, offset=offset, width=1)
    for component in scaled:
        assert component.shape == (3, 256, 256)


def test_slab_window_never_materializes_out_of_volume_rows() -> None:
    """The window is the true-row intersection; interior seams keep their halo."""

    thin = np.zeros((1, 8, 8), dtype=np.float64)
    window, offset = _slab_window(thin, 0, 1, 73)
    assert window.shape == (1, 8, 8)
    assert offset == 0
    deep = np.zeros((64, 8, 8), dtype=np.float64)
    window, offset = _slab_window(deep, 20, 30, 5)
    assert window.shape == (20, 8, 8)
    assert offset == 5
    scaled = _slab_scaled_derivatives(window, (1.0, 1.0, 1.0), 1.0, offset=offset, width=10)
    for component in scaled:
        assert component.shape == (12, 8, 8)


def test_nan_propagating_max_matches_numpy_reduction_semantics() -> None:
    """The two-value helper is ``np.max`` of the pair, NaN included."""

    nan = float("nan")
    assert _nan_propagating_max(1.0, 2.0) == 2.0
    assert _nan_propagating_max(0.0, 3.0) == 3.0
    assert _nan_propagating_max(-1.0, -2.0) == -1.0
    assert math.isnan(_nan_propagating_max(0.0, nan))
    assert math.isnan(_nan_propagating_max(nan, 0.0))
    assert math.isnan(_nan_propagating_max(nan, 5.0))
    assert math.isnan(_nan_propagating_max(5.0, nan))
    for existing, candidate in ((1.0, 2.0), (2.0, 1.0), (-1.0, -2.0), (0.0, -0.0)):
        assert _nan_propagating_max(existing, candidate) == float(
            np.max(np.array([existing, candidate]))
        )


def test_nan_subblock_maximum_is_not_discarded() -> None:
    """A NaN in a LATER sub-block poisons the slab reduction like ``np.max``."""

    step = int(_BLOCKED_EIG_SUB_SLICES)
    width = 2 * step
    derivative = np.zeros((width + 2, 1, 1), dtype=np.float64)
    derivative[_TRIMMED_VALID_ROW + step : _TRIMMED_VALID_ROW + width, 0, 0] = np.nan
    scaled = (derivative.copy(),) * 6
    assert math.isnan(_slab_max_s(scaled, _TRIMMED_VALID_ROW, width))
    ordered = _ordered_by_magnitude(_symmetric_eigenvalues(*scaled))
    l1, l2, l3 = ordered[..., 0], ordered[..., 1], ordered[..., 2]
    s = np.sqrt(np.maximum(l1 * l1 + l2 * l2 + l3 * l3, 0.0))
    assert math.isnan(float(np.max(s[_TRIMMED_VALID_ROW : _TRIMMED_VALID_ROW + width])))
    # A NaN outside the valid centre is not part of the scale-global max.
    outside = np.zeros((width + 2, 1, 1), dtype=np.float64)
    outside[0, 0, 0] = np.nan
    scaled_outside = (outside.copy(),) * 6
    assert _slab_max_s(scaled_outside, _TRIMMED_VALID_ROW, width) == 0.0


def test_f1_overflow_response_bytes_mask_and_partition_independence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The F1 repro: NaN max(S) -> all-zero response on both paths, any width."""

    rng = np.random.RandomState(20260919)
    volume = np.ascontiguousarray(rng.randn(160, 16, 16) * 100.0)
    volume[0, 0, 0] = 1e160
    spacing = (1.0, 1.0, 1.0)
    parameters = VesselnessParameters(scales_mm=(1.0,), threshold=0.09)
    reference = _frangi_vesselness_unblocked(volume, spacing, parameters)
    response = frangi_vesselness(volume, spacing, parameters)
    assert np.array_equal(reference, response)
    assert not bool(np.any(reference))
    assert not bool(np.any(vessel_mask(response, parameters)))
    assert math.isnan(_blocked_max_s(volume, spacing, 1.0))
    for eig_width, budget, min_valid in ((1, 64, 16), (3, 32, 8), (17, 20, 4), (64, 200, 16)):
        monkeypatch.setattr(vesselness_module, "_BLOCKED_EIG_SUB_SLICES", eig_width)
        monkeypatch.setattr(vesselness_module, "_BLOCKED_MAX_EXTENDED_SLICES", budget)
        monkeypatch.setattr(vesselness_module, "_BLOCKED_MIN_VALID_SLICES", min_valid)
        assert np.array_equal(reference, frangi_vesselness(volume, spacing, parameters))


@pytest.mark.parametrize("eig_width", (1, 2, 3, 5, 16, 17, 64))
@pytest.mark.parametrize(
    "budget,min_valid",
    ((64, 16), (32, 8), (96, 12), (20, 4), (200, 16)),
)
def test_partition_independence_of_valid_and_eigen_widths(
    monkeypatch: pytest.MonkeyPatch, eig_width: int, budget: int, min_valid: int
) -> None:
    """Any valid/eigen partition reproduces the same response bytes."""

    volume = _impulse_volume(
        (64, 32, 32), ((0, 0, 0), (32, 16, 16), (63, 31, 31), (17, 3, 29))
    )
    spacing = (1.0, 1.0, 1.0)
    parameters = VesselnessParameters(scales_mm=(1.0, 2.0), threshold=0.09)
    reference = _frangi_vesselness_unblocked(volume, spacing, parameters)
    monkeypatch.setattr(vesselness_module, "_BLOCKED_EIG_SUB_SLICES", eig_width)
    monkeypatch.setattr(vesselness_module, "_BLOCKED_MAX_EXTENDED_SLICES", budget)
    monkeypatch.setattr(vesselness_module, "_BLOCKED_MIN_VALID_SLICES", min_valid)
    assert _bytes(frangi_vesselness(volume, spacing, parameters)) == _bytes(reference)


def test_nonfinite_input_refuses_identically_on_both_paths() -> None:
    """A non-finite sample refuses VOLUME_INVALID before any slab work."""

    parameters = VesselnessParameters(scales_mm=(1.0,), threshold=0.09)
    for shape in ((16, 16, 16), (32, 32, 32)):
        for bad in (np.nan, np.inf, -np.inf):
            volume = np.zeros(shape, dtype=np.float64)
            volume[0, 0, 0] = bad
            with pytest.raises(CoronaryValidationError) as excinfo:
                frangi_vesselness(volume, (1.0, 1.0, 1.0), parameters)
            assert excinfo.value.code is CoronaryRefusalCode.VOLUME_INVALID
            with pytest.raises(CoronaryValidationError) as excinfo:
                _frangi_vesselness_unblocked(volume, (1.0, 1.0, 1.0), parameters)
            assert excinfo.value.code is CoronaryRefusalCode.VOLUME_INVALID


def test_nonfinite_response_refusal_is_shared_by_both_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both final finiteness guards refuse NONFINITE_RESPONSE identically."""

    parameters = VesselnessParameters(scales_mm=(1.0,), threshold=0.09)
    volume = np.zeros((32, 32, 32), dtype=np.float64)

    def poisoned_response_into(
        scaled: tuple[np.ndarray, ...],
        alpha: float,
        beta: float,
        c: float,
        offset: int,
        out: np.ndarray,
    ) -> None:
        out[:] = np.nan

    monkeypatch.setattr(vesselness_module, "_slab_response_into", poisoned_response_into)
    with pytest.raises(CoronaryValidationError) as excinfo:
        vesselness_module._frangi_vesselness_blocked(volume, (1.0, 1.0, 1.0), parameters)
    assert excinfo.value.code is CoronaryRefusalCode.NONFINITE_RESPONSE

    def poisoned_frangi_response(
        eigenvalues: np.ndarray, alpha: float, beta: float, gamma: float
    ) -> np.ndarray:
        return np.full(eigenvalues.shape[:-1], np.nan, dtype=np.float64)

    monkeypatch.setattr(vesselness_module, "_frangi_response", poisoned_frangi_response)
    with pytest.raises(CoronaryValidationError) as excinfo:
        vesselness_module._frangi_vesselness_unblocked(volume, (1.0, 1.0, 1.0), parameters)
    assert excinfo.value.code is CoronaryRefusalCode.NONFINITE_RESPONSE


def test_vessel_mask_refuses_a_nonfinite_response() -> None:
    parameters = VesselnessParameters(scales_mm=(1.0,), threshold=0.09)
    response = np.zeros((4, 4, 4), dtype=np.float64)
    response[0, 0, 0] = np.nan
    with pytest.raises(CoronaryValidationError) as excinfo:
        vessel_mask(response, parameters)
    assert excinfo.value.code is CoronaryRefusalCode.VOLUME_INVALID


_CAPPED_PROBE = textwrap.dedent(
    """
    import hashlib
    import sys

    try:
        import resource
    except ImportError:
        print("CAP_UNENFORCEABLE resource module unavailable")
        raise SystemExit(0)

    cap = int(sys.argv[2])
    try:
        resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
    except (AttributeError, OSError, ValueError) as exc:
        print(f"CAP_UNENFORCEABLE {exc!r}")
        raise SystemExit(0)

    import numpy as np

    sys.path.insert(0, sys.argv[1])

    from core.coronary.types import VesselnessParameters
    from core.coronary.vesselness import _frangi_vesselness_unblocked, frangi_vesselness

    def digest(array):
        contiguous = np.ascontiguousarray(array, dtype=np.float64)
        return hashlib.sha256(contiguous.tobytes(order="C")).hexdigest()

    parameters = VesselnessParameters(scales_mm=(3.0,), threshold=0.09)
    spacing = (1.0, 1.0, 0.125)
    cases = {
        "thin-1x256x256": np.full((1, 256, 256), -100, dtype=np.int16),
        "deep-3x64x64": np.full((3, 64, 64), -100, dtype=np.int16),
    }
    for name, volume in cases.items():
        volume[volume.shape[0] // 2, volume.shape[1] // 2, volume.shape[2] // 2] = 1000
        reference = digest(_frangi_vesselness_unblocked(volume, spacing, parameters))
        blocked = digest(frangi_vesselness(volume, spacing, parameters))
        print(f"DIGESTS {name} {reference} {blocked}")
    """
)


def test_thin_volume_large_halo_completes_under_a_capped_address_space() -> None:
    """F2: 512 MiB address-space cap, thin/deep volumes with a 73-row halo."""

    root = Path(__file__).resolve().parents[3]
    completed = subprocess.run(
        [sys.executable, "-c", _CAPPED_PROBE, str(root), str(512 * 1024 * 1024)],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    if "CAP_UNENFORCEABLE" in completed.stdout:
        pytest.skip(completed.stdout.strip())
    assert completed.returncode == 0, completed.stderr
    lines = [line for line in completed.stdout.splitlines() if line.startswith("DIGESTS")]
    assert len(lines) == 2
    for line in lines:
        _, name, reference, blocked = line.split()
        assert reference == blocked, name
