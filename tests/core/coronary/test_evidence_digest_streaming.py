"""F30-2: bounded-memory response/mask evidence digests preserve identity.

The derivation evidence digests (``response_sha256`` / ``mask_sha256``) hash
the canonical JSON of the full arrays.  :func:`_evidence_array_digest`
serializes in bounded slabs; every cell below locks that the slabbed digest
is BYTE-IDENTICAL to the historical one-shot
``canonical_digest({"dtype": ..., "values": array.tolist()})`` identity -
no version bump, no migration rule.  The frozen literals are the HEAD
(pre-change) evidence digests; any serialization drift fails closed here.
"""

from __future__ import annotations

import os
import platform
from pathlib import Path

import numpy as np
import pytest

from apps.workstation.volume_loader import CanonicalVolumeLoader
from core.coronary import centerline as centerline_module
from core.coronary.centerline import _evidence_array_digest
from core.coronary.types import VesselnessParameters
from core.coronary.vesselness import frangi_vesselness, vessel_mask
from core.digest import canonical_digest
from tests.core.coronary.support import analytic_volume, phantom_parameters

rng = np.random.RandomState(20260919)

_RANDOM_CASES: list[np.ndarray] = [
    rng.randn(5, 7, 11),
    (rng.randn(4, 4, 4) * 1e6).astype(np.float64),
    (rng.randn(3, 5, 2) * 1e-12).astype(np.float64),
    np.ascontiguousarray(rng.randn(8, 8, 8)[::2, :, ::2]),
    np.asfortranarray(rng.randn(4, 6, 5)),
    np.full((2, 2, 2), np.inf),
    np.full((2, 2, 2), -np.inf),
    np.full((2, 3, 2), np.nan),
    np.zeros((1, 2, 3)),
    np.zeros((0, 4, 4)),
    np.zeros((1, 1, 1)),
    (rng.randint(-128, 127, size=(6, 5, 4))).astype(np.int8),
    (rng.randint(0, 2, size=(4, 4, 4))).astype(bool).astype(np.int8),
    np.ascontiguousarray(rng.randint(0, 255, size=(17, 13, 19))).astype(np.int8),
]


def _oneshot(dtype_label: str, array: np.ndarray) -> str:
    return canonical_digest({"dtype": dtype_label, "values": array.tolist()})


@pytest.mark.parametrize("index", range(len(_RANDOM_CASES)))
def test_slabbed_digest_matches_oneshot_identity(index: int) -> None:
    array = _RANDOM_CASES[index]
    label = str(array.dtype)
    assert _evidence_array_digest(label, array) == _oneshot(label, array)


def test_slab_size_does_not_change_the_digest(monkeypatch: pytest.MonkeyPatch) -> None:
    array = rng.randn(9, 11, 7)
    expected = _oneshot("float64", array)
    for chunk in (1, 7, 1 << 10, 1 << 30):
        monkeypatch.setattr(centerline_module, "_EVIDENCE_DIGEST_CHUNK_VOXELS", chunk)
        assert _evidence_array_digest("float64", array) == expected


def test_non_3d_inputs_take_the_verbatim_oneshot_path() -> None:
    for array in (np.float64(0.5), rng.randn(7), rng.randn(3, 4)):
        assayed = np.asarray(array)
        assert _evidence_array_digest("float64", assayed) == _oneshot("float64", assayed)


#: HOST_PARITY (frozen-profile separation): historical VPS baseline for the
#: Phantom48 response-evidence digest. Preserved verbatim; the VPS host
#: computes exactly these bytes. This literal is DOCUMENTATION, never asserted
#: on a non-VPS host (asserting it unconditionally fails on the PC/Ryzen/WSL2
#: host, whose libm dispatches the same `np.exp`/`np.arccos` transcendental
#: paths to different last-ulp results).
_VPS_FROZEN_RESPONSE_EVIDENCE_SHA256 = (
    "795ec57207445655badfb5d7c11c4e8ea34ed131697c38bd1f1380ce67e1e506"
)

#: Portable invariant (asserted unconditionally on every host): the Phantom48
#: mask-evidence digest and mask count are bit-stable across VPS and PC hosts.
_PORTABLE_MASK_EVIDENCE_SHA256 = (
    "53ced9b767870f71c82e97a81c3a1383f1be2853d82f4f7c030c10205ff5c677"
)
_PORTABLE_MASK_COUNT = 1071

#: HOST_PARITY: mechanically-bound pinned attributes of the PC/Ryzen/WSL2 host
#: profile. Each value is compared against the live runtime (kernel cpuinfo,
#: loader confstr, interpreter, numpy); there is no environment override, no
#: calibration/config file, no auto-learn, no fallback. Unknown hosts FAIL
#: CLOSED below — never skip, never pass vacuously.
_HOST_PROFILE_PC_RYZEN_WSL2_CPUINFO_MODEL = "AMD Ryzen 5 5600 6-Core Processor"
_HOST_PROFILE_PC_RYZEN_WSL2_NUMPY = "2.5.2"
_HOST_PROFILE_PC_RYZEN_WSL2_PYTHON = "3.12.15"
_HOST_PROFILE_PC_RYZEN_WSL2_GLIBC = "2.43"
_HOST_PROFILE_PC_RYZEN_WSL2_RESPONSE_EVIDENCE_SHA256 = (
    "947f99f9d0f3c9b6e9c8b31b4dffd7403e6e06b42ff94f3e092e7134b29fad1e"
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


def _phantom48_host_response_evidence_digest() -> str:
    """Fail-closed host dispatch: VPS literal documented, PC digest asserted.

    Returns the expected response-evidence digest for THIS host, or raises
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
        return _HOST_PROFILE_PC_RYZEN_WSL2_RESPONSE_EVIDENCE_SHA256
    raise AssertionError(
        "HOST_PARITY: unknown host — no calibrated response-evidence digest; "
        f"cpu_model={cpu_model!r} numpy={numpy_version!r} "
        f"python={python_version!r} glibc={glibc_version!r}; "
        f"historical VPS baseline={_VPS_FROZEN_RESPONSE_EVIDENCE_SHA256}"
    )


def test_phantom48_evidence_identity_is_frozen() -> None:
    """Phantom48 evidence digests: host-calibrated response, portable mask.

    The response-evidence digest is host-calibrated (fail-closed fingerprint;
    VPS baseline preserved verbatim in ``_VPS_FROZEN_RESPONSE_EVIDENCE_SHA256``);
    the mask-evidence digest and mask count are portable and asserted
    unconditionally.
    """

    raw, _, _, _ = analytic_volume()
    parameters = phantom_parameters()
    response = frangi_vesselness(raw, (1.0, 1.0, 1.0), parameters)
    mask = vessel_mask(response, parameters)
    assert (
        _evidence_array_digest(str(response.dtype), response)
        == _phantom48_host_response_evidence_digest()
    )
    assert (
        _evidence_array_digest("bool", np.asarray(mask, dtype=np.int8))
        == _PORTABLE_MASK_EVIDENCE_SHA256
    )
    assert int(np.count_nonzero(mask)) == _PORTABLE_MASK_COUNT


_REAL_BLOBS_ROOT = Path("/tmp/opencode/t2_ws/blobs")
_REAL_CROP_DIGEST = "7f17e5594ef84278b6ce84016b559cf22f8cae3c3d7e2245d9148b166820facb"
_REAL_CROP_AVAILABLE = CanonicalVolumeLoader(_REAL_BLOBS_ROOT).blob_path(_REAL_CROP_DIGEST).exists()


@pytest.mark.skipif(
    not _REAL_CROP_AVAILABLE,
    reason="real-data-gate-only: provision /tmp/opencode/t2_ws/blobs for Case02",
)
def test_case02_crop_evidence_identity_is_frozen() -> None:
    """Case02 96^3 crop evidence digests equal HEAD values exactly (read-only)."""

    loader = CanonicalVolumeLoader(_REAL_BLOBS_ROOT)
    full = np.asarray(loader.read_array(_REAL_CROP_DIGEST))
    crop = np.ascontiguousarray(full[56:152, 323:419, 140:236])
    assert crop.shape == (96, 96, 96)
    parameters = VesselnessParameters(scales_mm=(1.0, 2.0), threshold=0.09)
    response = frangi_vesselness(crop, (0.427734, 0.427734, 0.625), parameters)
    mask = vessel_mask(response, parameters)
    assert (
        _evidence_array_digest(str(response.dtype), response)
        == "fddd79a5f8666060f151c32da8d588211b63725832e4ba3d076ad15d841156a4"
    )
    assert (
        _evidence_array_digest("bool", np.asarray(mask, dtype=np.int8))
        == "0cd77f8362a639c65486190120aedb1158cb4412343bb24916d8dee8d59447a7"
    )
