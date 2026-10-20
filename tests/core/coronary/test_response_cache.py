"""F30-OPT: seed-independent vesselness response/mask cache.

The response/mask are pure functions of source bytes + spacing + parameters,
so a second seed on the same volume must skip the Frangi recompute.  Every
cell below runs the real derive path (``derive_coronary_path`` /
``build_centerline``); the Frangi-call counter is a monkeypatched wrapper in
the tests only - production code carries no counter.

Patterns mirror ``test_vesselness_blocked_parity.py`` (repo fixtures,
bit-exact ``np.array_equal``, frozen Tier-A digests where applicable).
"""

from __future__ import annotations

import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import core.coronary.centerline as centerline_module
import core.coronary.response_cache as rc
from apps.workstation.volume_loader import (
    CANONICAL_INT16_CODEC,
    CanonicalVolumeLoader,
)
from core.coordinates import transform_for_frame
from core.coronary import (
    ResponseCacheStore,
    VesselnessCacheKey,
    build_centerline,
    cache_key_for,
    cached_frangi_vesselness,
    derive_coronary_path,
    source_content_sha256,
)
from core.coronary.response_cache import CACHE_SCHEMA_VERSION
from core.coronary.types import Seed, VesselnessParameters
from core.coronary.vesselness import RESPONSE_DTYPE, frangi_vesselness, vessel_mask
from core.enums import ReleaseLane, VolumeKind
from core.ids import new_ulid
from core.volume import VolumeBlobRef, create_volume
from tests.core.coronary.support import (
    FRAME_OF_REFERENCE_UID,
    SERIES_INSTANCE_UID,
    STUDY_INSTANCE_UID,
    ULIDS,
    analytic_volume,
    helix_points,
    phantom_frame,
    phantom_parameters,
    phantom_seed,
)

SPACING = (1.0, 1.0, 1.0)

_REF: dict[str, np.ndarray] = {}


def _reference() -> tuple[np.ndarray, np.ndarray]:
    """Direct (uncached) response/mask bytes, computed once per session."""

    if "response" not in _REF:
        raw, _, _, _ = analytic_volume()
        parameters = phantom_parameters()
        response = frangi_vesselness(raw, SPACING, parameters)
        _REF["response"] = response
        _REF["mask"] = vessel_mask(response, parameters)
    return _REF["response"], _REF["mask"]


def _raw() -> np.ndarray:
    raw, _, _, _ = analytic_volume()
    return raw


def _second_seed() -> Seed:
    """A different human seed on the same phantom (probed working)."""

    curve = helix_points()
    probe = curve[300]
    ahead = curve[320]
    behind = curve[280]
    return Seed(
        i=int(round(float(probe[0]))),
        j=int(round(float(probe[1]))),
        k=int(round(float(probe[2]))),
        direction=(
            float(ahead[0] - behind[0]),
            float(ahead[1] - behind[1]),
            float(ahead[2] - behind[2]),
        ),
    )


def _cache_evidence(path: Any) -> tuple[object, object]:
    """The recorded hit/miss evidence (None, None when no cache consulted)."""

    document = path.diagnostics.to_document()
    return document.get("response_cache_hit"), document.get("response_cache_key")


def _array_key(
    raw: np.ndarray, parameters: VesselnessParameters, spacing: tuple[float, float, float] = SPACING
) -> VesselnessCacheKey:
    return cache_key_for(
        source_blob_sha256=source_content_sha256(raw),
        volume=raw,
        spacing_mm=spacing,
        parameters=parameters,
        source_kind="array-content-sha256",
    )


def _counting_wrappers(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, int]:
    """Count Frangi computes through the cached path (tests only)."""

    calls = {"frangi": 0, "mask": 0}
    real_frangi = rc.frangi_vesselness
    real_mask = rc.vessel_mask

    def _frangi(*args: Any, **kwargs: Any) -> np.ndarray:
        calls["frangi"] += 1
        return real_frangi(*args, **kwargs)

    def _mask(*args: Any, **kwargs: Any) -> np.ndarray:
        calls["mask"] += 1
        return real_mask(*args, **kwargs)

    monkeypatch.setattr(rc, "frangi_vesselness", _frangi)
    monkeypatch.setattr(rc, "vessel_mask", _mask)
    return calls


# ---------------------------------------------------------------------------
# Key unit tests
# ---------------------------------------------------------------------------


def test_key_is_deterministic_and_seed_free() -> None:
    raw = _raw()
    parameters = phantom_parameters()
    first = _array_key(raw, parameters)
    second = _array_key(raw, parameters)
    assert first.digest == second.digest
    document = first.to_document()
    assert set(document) == {
        "schema_version",
        "source_volume_blob_sha256",
        "source_digest_kind",
        "source_volume_shape",
        "source_volume_dtype",
        "voxel_spacing_mm",
        "parameters_sha256",
        "algorithm",
        "algorithm_version",
    }
    assert document["source_volume_dtype"] == raw.dtype.str
    assert "seed" not in json.dumps(document)


def test_key_changes_with_every_component() -> None:
    raw = _raw()
    parameters = phantom_parameters()
    base = _array_key(raw, parameters)
    blob = hashlib.sha256(b"other-source").hexdigest()
    variants = [
        cache_key_for(
            source_blob_sha256=blob,
            volume=raw,
            spacing_mm=SPACING,
            parameters=parameters,
            source_kind="array-content-sha256",
        ),
        cache_key_for(
            source_blob_sha256=source_content_sha256(raw),
            volume=raw,
            spacing_mm=(1.0, 1.0, 2.0),
            parameters=parameters,
            source_kind="array-content-sha256",
        ),
        cache_key_for(
            source_blob_sha256=source_content_sha256(raw),
            volume=raw,
            spacing_mm=SPACING,
            parameters=VesselnessParameters(scales_mm=(1.0, 2.0), threshold=0.10),
            source_kind="array-content-sha256",
        ),
        cache_key_for(
            source_blob_sha256=source_content_sha256(raw),
            volume=np.zeros((32, 32, 32), dtype=np.int16),
            spacing_mm=SPACING,
            parameters=parameters,
            source_kind="array-content-sha256",
        ),
        VesselnessCacheKey(
            source_blob_sha256=source_content_sha256(raw),
            source_shape=(48, 48, 48),
            spacing_mm=SPACING,
            parameters_sha256=parameters.sha256,
            source_dtype=raw.dtype.str,
            algorithm_version="frangi-edt-ridge-centered-spine-v2",
            source_kind="array-content-sha256",
        ),
        cache_key_for(
            source_blob_sha256=source_content_sha256(raw),
            volume=raw,
            spacing_mm=SPACING,
            parameters=parameters,
            source_kind="volume-blob-sha256",
        ),
    ]
    digests = {variant.digest for variant in variants}
    assert len(digests) == len(variants)
    assert base.digest not in digests


def test_key_rejects_bad_inputs() -> None:
    raw = _raw()
    parameters = phantom_parameters()
    good_sha = source_content_sha256(raw)
    with pytest.raises(ValueError):
        cache_key_for(
            source_blob_sha256="not-hex",
            volume=raw,
            spacing_mm=SPACING,
            parameters=parameters,
        )
    with pytest.raises(ValueError):
        cache_key_for(
            source_blob_sha256=good_sha,
            volume=raw,
            spacing_mm=(1.0, 1.0, -2.0),
            parameters=parameters,
        )
    with pytest.raises(ValueError):
        cache_key_for(
            source_blob_sha256=good_sha,
            volume=np.zeros((8, 8), dtype=np.int16),
            spacing_mm=SPACING,
            parameters=parameters,
        )
    with pytest.raises(ValueError):
        cache_key_for(
            source_blob_sha256=good_sha,
            volume=raw,
            spacing_mm=SPACING,
            parameters="frozen",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError):
        cache_key_for(
            source_blob_sha256=good_sha,
            volume=raw,
            spacing_mm=SPACING,
            parameters=parameters,
            source_kind="blob-kind-typo",
        )
    with pytest.raises(ValueError):
        source_content_sha256(np.zeros((8, 8), dtype=np.int16))
    with pytest.raises(ValueError):
        cached_frangi_vesselness(
            raw,
            SPACING,
            parameters,
            source_blob_sha256="not-hex",
            store=ResponseCacheStore("/nonexistent-root"),
        )


def test_source_content_digest_stable_and_sensitive() -> None:
    raw = _raw()
    assert source_content_sha256(raw) == source_content_sha256(raw.copy())
    flipped = raw.copy()
    flipped[0, 0, 0] = int(flipped[0, 0, 0]) + 1
    assert source_content_sha256(flipped) != source_content_sha256(raw)
    assert source_content_sha256(raw.astype(np.float64)) != source_content_sha256(raw)


def test_blob_key_binds_source_dtype_including_byte_order(tmp_path: Path) -> None:
    """Byte-identical blobs under different dtypes must never share a key.

    Regression for the blob-key collision: the int16 phantom and its uint16
    view share every byte but feed different samples into the derivation, so
    their fresh masks differ and their blob-based keys must differ as well.
    Byte order is key material too (``<i2`` vs ``>i2``).
    """

    raw = _raw()
    assert raw.dtype.str == "<i2"
    unsigned = raw.view(np.uint16)
    assert unsigned.shape == raw.shape
    assert np.ascontiguousarray(unsigned).tobytes(order="C") == np.ascontiguousarray(raw).tobytes(
        order="C"
    )
    be_same_bytes = (
        np.frombuffer(np.ascontiguousarray(raw).tobytes(order="C"), dtype=np.dtype(">i2"))
        .reshape(raw.shape)
        .copy()
    )
    assert be_same_bytes.dtype.str == ">i2"
    assert np.ascontiguousarray(be_same_bytes).tobytes(order="C") == np.ascontiguousarray(
        raw
    ).tobytes(order="C")
    parameters = phantom_parameters()
    blob = hashlib.sha256(b"phantom-blob").hexdigest()
    key_signed = cache_key_for(
        source_blob_sha256=blob,
        volume=raw,
        spacing_mm=SPACING,
        parameters=parameters,
    )
    key_unsigned = cache_key_for(
        source_blob_sha256=blob,
        volume=unsigned,
        spacing_mm=SPACING,
        parameters=parameters,
    )
    key_be = cache_key_for(
        source_blob_sha256=blob,
        volume=be_same_bytes,
        spacing_mm=SPACING,
        parameters=parameters,
    )
    assert key_signed.to_document()["source_volume_dtype"] == "<i2"
    assert key_unsigned.to_document()["source_volume_dtype"] == "<u2"
    assert key_be.to_document()["source_volume_dtype"] == ">i2"
    assert key_signed.digest != key_unsigned.digest
    assert key_signed.digest != key_be.digest
    assert key_unsigned.digest != key_be.digest
    fresh_signed = frangi_vesselness(raw, SPACING, parameters)
    fresh_unsigned = frangi_vesselness(unsigned, SPACING, parameters)
    mask_signed = vessel_mask(fresh_signed, parameters)
    mask_unsigned = vessel_mask(fresh_unsigned, parameters)
    assert np.count_nonzero(mask_signed != mask_unsigned) > 0
    store = ResponseCacheStore(tmp_path / "cache")
    store.store(key_signed, fresh_signed, mask_signed)
    assert store.lookup(key_unsigned) is None
    assert store.lookup(key_be) is None
    got_unsigned = cached_frangi_vesselness(
        unsigned,
        SPACING,
        parameters,
        source_blob_sha256=blob,
        store=store,
    )
    assert got_unsigned.cache_hit is False
    assert np.array_equal(got_unsigned.response, fresh_unsigned)
    assert np.array_equal(got_unsigned.mask, mask_unsigned)
    got_signed = cached_frangi_vesselness(
        raw,
        SPACING,
        parameters,
        source_blob_sha256=blob,
        store=store,
    )
    assert got_signed.cache_hit is True
    assert np.array_equal(got_signed.response, fresh_signed)
    assert np.array_equal(got_signed.mask, mask_signed)


def test_malformed_spacing_refuses_cold_and_warm(tmp_path: Path) -> None:
    """Cold and warm caches refuse malformed spacing identically (VOLUME_INVALID).

    Regression for the warm-hit bypass: an entry stored under valid spacing
    must never be served to a derivation whose spacing fails the uncached
    path.  Both layers share the single ``_require_spacing`` gate, so the
    refusal code matches exactly.
    """

    from core.coronary import CoronaryRefusalCode, CoronaryValidationError

    raw = _raw()
    frame = phantom_frame()
    seed = phantom_seed()
    parameters = phantom_parameters()
    bad_spacing: Any = ("1", 1.0, 1.0)
    with pytest.raises(CoronaryValidationError) as cold_exc:
        derive_coronary_path(raw, bad_spacing, frame, seed, parameters)
    assert cold_exc.value.code is CoronaryRefusalCode.VOLUME_INVALID
    store = ResponseCacheStore(tmp_path / "cache")
    derive_coronary_path(raw, SPACING, frame, seed, parameters, response_cache=store)
    with pytest.raises(CoronaryValidationError) as warm_exc:
        derive_coronary_path(raw, bad_spacing, frame, seed, parameters, response_cache=store)
    assert warm_exc.value.code is CoronaryRefusalCode.VOLUME_INVALID
    blob = hashlib.sha256(b"spacing-probe").hexdigest()
    direct_store = ResponseCacheStore(tmp_path / "direct-cache")
    cached_frangi_vesselness(raw, SPACING, parameters, source_blob_sha256=blob, store=direct_store)
    with pytest.raises(CoronaryValidationError) as direct_cold:
        cached_frangi_vesselness(
            raw,
            bad_spacing,
            parameters,
            source_blob_sha256=hashlib.sha256(b"spacing-probe-cold").hexdigest(),
            store=ResponseCacheStore(tmp_path / "direct-cache-cold"),
        )
    assert direct_cold.value.code is CoronaryRefusalCode.VOLUME_INVALID
    with pytest.raises(CoronaryValidationError) as direct_warm:
        cached_frangi_vesselness(
            raw, bad_spacing, parameters, source_blob_sha256=blob, store=direct_store
        )
    assert direct_warm.value.code is CoronaryRefusalCode.VOLUME_INVALID


# ---------------------------------------------------------------------------
# Store round-trip
# ---------------------------------------------------------------------------


def test_store_lookup_roundtrip_identical_bytes(tmp_path: Path) -> None:
    store = ResponseCacheStore(tmp_path / "cache")
    response_ref, mask_ref = _reference()
    key = _array_key(_raw(), phantom_parameters())
    assert store.lookup(key) is None
    stored = store.store(key, response_ref, mask_ref)
    hit = store.lookup(key)
    assert hit is not None
    assert np.array_equal(hit.response, response_ref)
    assert np.array_equal(hit.mask, mask_ref)
    assert hit.response.dtype == RESPONSE_DTYPE
    assert hit.mask.dtype == np.dtype(bool)
    assert not hit.response.flags.writeable
    assert not hit.mask.flags.writeable
    assert hit.response_sha256 == stored.response_sha256
    assert hit.mask_sha256 == stored.mask_sha256
    assert (
        hit.response_sha256
        == hashlib.sha256(
            np.ascontiguousarray(response_ref, dtype=np.float64).tobytes(order="C")
        ).hexdigest()
    )


def test_invalidate_drops_entry(tmp_path: Path) -> None:
    store = ResponseCacheStore(tmp_path / "cache")
    response_ref, mask_ref = _reference()
    key = _array_key(_raw(), phantom_parameters())
    store.store(key, response_ref, mask_ref)
    assert store.lookup(key) is not None
    assert store.invalidate(key) is True
    assert store.lookup(key) is None
    assert store.invalidate(key) is False


# ---------------------------------------------------------------------------
# Derive-path wiring
# ---------------------------------------------------------------------------


def test_derive_miss_then_hit_identical_bytes(tmp_path: Path) -> None:
    raw = _raw()
    frame = phantom_frame()
    seed = phantom_seed()
    parameters = phantom_parameters()
    store = ResponseCacheStore(tmp_path / "cache")
    first = derive_coronary_path(raw, SPACING, frame, seed, parameters, response_cache=store)
    second = derive_coronary_path(raw, SPACING, frame, seed, parameters, response_cache=store)
    assert np.array_equal(first.response, second.response)
    assert np.array_equal(first.mask, second.mask)
    assert first.diagnostics.response_sha256 == second.diagnostics.response_sha256
    assert first.diagnostics.mask_sha256 == second.diagnostics.mask_sha256
    first_hit, first_key = _cache_evidence(first)
    second_hit, second_key = _cache_evidence(second)
    assert first_hit is False
    assert second_hit is True
    assert first_key == second_key
    assert first_key == _array_key(raw, parameters).digest


def test_second_seed_skips_recompute(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run derive twice with different seeds: Frangi computes once."""

    calls = _counting_wrappers(monkeypatch)
    raw = _raw()
    frame = phantom_frame()
    parameters = phantom_parameters()
    store = ResponseCacheStore(tmp_path / "cache")
    seed_a = phantom_seed()
    seed_b = _second_seed()
    assert (seed_a.i, seed_a.j, seed_a.k) != (seed_b.i, seed_b.j, seed_b.k)
    path_a = derive_coronary_path(raw, SPACING, frame, seed_a, parameters, response_cache=store)
    assert calls == {"frangi": 1, "mask": 1}
    assert _cache_evidence(path_a)[0] is False
    path_b = derive_coronary_path(raw, SPACING, frame, seed_b, parameters, response_cache=store)
    assert calls == {"frangi": 1, "mask": 1}, "second seed must serve the cache hit"
    assert _cache_evidence(path_b) == (True, _cache_evidence(path_a)[1])
    assert np.array_equal(path_a.response, path_b.response)
    assert np.array_equal(path_a.mask, path_b.mask)
    response_ref, mask_ref = _reference()
    assert np.array_equal(path_b.response, response_ref)
    assert np.array_equal(path_b.mask, mask_ref)


def test_different_params_miss(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _counting_wrappers(monkeypatch)
    raw = _raw()
    frame = phantom_frame()
    seed = phantom_seed()
    store = ResponseCacheStore(tmp_path / "cache")
    derive_coronary_path(raw, SPACING, frame, seed, phantom_parameters(), response_cache=store)
    assert calls["frangi"] == 1
    other = VesselnessParameters(scales_mm=(1.0, 2.0), threshold=0.10)
    path = derive_coronary_path(raw, SPACING, frame, seed, other, response_cache=store)
    assert calls["frangi"] == 2
    assert _cache_evidence(path)[0] is False


def test_different_source_misses(tmp_path: Path) -> None:
    raw = _raw()
    frame = phantom_frame()
    seed = phantom_seed()
    parameters = phantom_parameters()
    store = ResponseCacheStore(tmp_path / "cache")
    derive_coronary_path(raw, SPACING, frame, seed, parameters, response_cache=store)
    altered = raw.copy()
    altered[0, 0, 0] = int(altered[0, 0, 0]) + 1
    path = derive_coronary_path(altered, SPACING, frame, seed, parameters, response_cache=store)
    assert _cache_evidence(path)[0] is False
    assert _cache_evidence(path)[1] != _array_key(raw, parameters).digest


def test_no_cache_leaves_diagnostics_unchanged() -> None:
    raw = _raw()
    path = derive_coronary_path(raw, SPACING, phantom_frame(), phantom_seed(), phantom_parameters())
    assert _cache_evidence(path) == (None, None)


# ---------------------------------------------------------------------------
# Corruption fallback (every mode = miss + recompute, never a failure)
# ---------------------------------------------------------------------------


def _corrupt(store: ResponseCacheStore, key: VesselnessCacheKey, mode: str) -> None:
    response_path, mask_path, manifest_path = store._paths(key.digest)
    if mode == "manifest-garbage":
        manifest_path.write_bytes(b"{not json")
    elif mode == "manifest-wrong-key":
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["cache_key_sha256"] = "0" * 64
        manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    elif mode == "response-truncated":
        data = response_path.read_bytes()
        response_path.write_bytes(data[: len(data) // 2])
    elif mode == "mask-deleted":
        mask_path.unlink()
    elif mode == "shape-mismatch":
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["source_volume_shape"] = [48, 48, 47]
        manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    else:  # pragma: no cover - test bug
        raise AssertionError(f"unknown corruption mode {mode}")


@pytest.mark.parametrize(
    "mode",
    [
        "manifest-garbage",
        "manifest-wrong-key",
        "response-truncated",
        "mask-deleted",
        "shape-mismatch",
    ],
)
def test_corrupt_entry_falls_back(tmp_path: Path, mode: str) -> None:
    raw = _raw()
    frame = phantom_frame()
    seed = phantom_seed()
    parameters = phantom_parameters()
    store = ResponseCacheStore(tmp_path / "cache")
    derive_coronary_path(raw, SPACING, frame, seed, parameters, response_cache=store)
    key = _array_key(raw, parameters)
    assert store.lookup(key) is not None
    _corrupt(store, key, mode)
    assert store.lookup(key) is None
    path = derive_coronary_path(raw, SPACING, frame, seed, parameters, response_cache=store)
    assert _cache_evidence(path)[0] is False
    response_ref, mask_ref = _reference()
    assert np.array_equal(path.response, response_ref)
    assert np.array_equal(path.mask, mask_ref)


class _BrokenStore(ResponseCacheStore):
    """A store whose I/O always fails (cache must never fail the derivation)."""

    def lookup(self, key: VesselnessCacheKey) -> Any:
        raise OSError("simulated cache read failure")

    def store(self, key: VesselnessCacheKey, response: np.ndarray, mask: np.ndarray) -> Any:
        raise OSError("simulated cache write failure")


def test_cache_errors_never_fail_derivation(tmp_path: Path) -> None:
    raw = _raw()
    frame = phantom_frame()
    seed = phantom_seed()
    parameters = phantom_parameters()
    store = _BrokenStore(tmp_path / "cache")
    path = derive_coronary_path(raw, SPACING, frame, seed, parameters, response_cache=store)
    assert _cache_evidence(path)[0] is False
    response_ref, mask_ref = _reference()
    assert np.array_equal(path.response, response_ref)
    assert np.array_equal(path.mask, mask_ref)
    # An unbuildable key is a recorded miss, not a derivation failure.
    broken = derive_coronary_path(
        raw,
        SPACING,
        frame,
        seed,
        parameters,
        response_cache=ResponseCacheStore(tmp_path / "cache2"),
        source_blob_sha256="not-hex",
    )
    assert _cache_evidence(broken) == (False, "")
    assert np.array_equal(broken.response, response_ref)


def test_concurrent_derives_share_cache(tmp_path: Path) -> None:
    raw = _raw()
    frame = phantom_frame()
    seed = phantom_seed()
    parameters = phantom_parameters()
    store = ResponseCacheStore(tmp_path / "cache")
    response_ref, mask_ref = _reference()

    def _derive() -> tuple[np.ndarray, np.ndarray]:
        path = derive_coronary_path(raw, SPACING, frame, seed, parameters, response_cache=store)
        return path.response, path.mask

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: _derive(), range(4)))
    for response, mask in results:
        assert np.array_equal(response, response_ref)
        assert np.array_equal(mask, mask_ref)


def test_concurrent_publish_same_key_succeeds(tmp_path: Path) -> None:
    """Synchronized writers publishing identical bytes must all succeed.

    Concurrent derivations on the same key share the target pathname but
    never a temp pathname: every publication either wins or cleanly loses
    the rename, and no writer falls back on FileNotFoundError.
    """

    store = ResponseCacheStore(tmp_path / "cache")
    response_ref, mask_ref = _reference()
    key = _array_key(_raw(), phantom_parameters())
    errors: list[BaseException] = []
    lock = threading.Lock()

    def _publish() -> None:
        try:
            store.store(key, response_ref, mask_ref)
        except BaseException as exc:
            with lock:
                errors.append(exc)

    for _ in range(25):
        barrier = threading.Barrier(4)

        def _synchronized() -> None:
            barrier.wait(timeout=30)
            _publish()

        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda _: _synchronized(), range(4)))
    assert errors == []
    hit = store.lookup(key)
    assert hit is not None
    assert np.array_equal(hit.response, response_ref)
    assert np.array_equal(hit.mask, mask_ref)


def test_publication_failure_returns_computed_arrays_as_miss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unwritable cache dir costs no recompute: one Frangi call, miss served."""

    rc_calls = _counting_wrappers(monkeypatch)
    real_centerline_frangi = centerline_module.frangi_vesselness
    real_centerline_mask = centerline_module.vessel_mask
    centerline_calls = {"frangi": 0, "mask": 0}

    def _centerline_frangi(*args: Any, **kwargs: Any) -> np.ndarray:
        centerline_calls["frangi"] += 1
        return real_centerline_frangi(*args, **kwargs)

    def _centerline_mask(*args: Any, **kwargs: Any) -> np.ndarray:
        centerline_calls["mask"] += 1
        return real_centerline_mask(*args, **kwargs)

    monkeypatch.setattr(centerline_module, "frangi_vesselness", _centerline_frangi)
    monkeypatch.setattr(centerline_module, "vessel_mask", _centerline_mask)
    # A regular file where the store root must be: lookup misses cleanly
    # and publication raises OSError however the suite is run (no chmod,
    # which root would ignore).
    blocker = tmp_path / "cache-root-blocker"
    blocker.write_bytes(b"not-a-directory")
    store = ResponseCacheStore(blocker / "cache")
    raw = _raw()
    parameters = phantom_parameters()
    path = derive_coronary_path(
        raw, SPACING, phantom_frame(), phantom_seed(), parameters, response_cache=store
    )
    assert rc_calls == {"frangi": 1, "mask": 1}
    assert centerline_calls == {"frangi": 0, "mask": 0}
    assert _cache_evidence(path) == (False, _array_key(raw, parameters).digest)
    response_ref, mask_ref = _reference()
    assert np.array_equal(path.response, response_ref)
    assert np.array_equal(path.mask, mask_ref)


# ---------------------------------------------------------------------------
# Provenance evidence
# ---------------------------------------------------------------------------


def test_provenance_document_marks_hit(tmp_path: Path) -> None:
    raw = _raw()
    parameters = phantom_parameters()
    store = ResponseCacheStore(tmp_path / "cache")
    source_sha = hashlib.sha256(b"phantom-blob").hexdigest()
    miss = cached_frangi_vesselness(
        raw, SPACING, parameters, source_blob_sha256=source_sha, store=store
    )
    assert miss.cache_hit is False
    assert miss.compute_wall_s >= 0.0
    miss_doc = miss.provenance_document()
    assert miss_doc["schema_version"] == CACHE_SCHEMA_VERSION
    assert miss_doc["cache_hit"] is False
    assert miss_doc["cache_key_sha256"] == miss.key_digest
    assert miss_doc["source_volume_blob_sha256"] == source_sha
    hit = cached_frangi_vesselness(
        raw, SPACING, parameters, source_blob_sha256=source_sha, store=store
    )
    assert hit.cache_hit is True
    assert hit.compute_wall_s == 0.0
    hit_doc = hit.provenance_document()
    assert hit_doc["cache_hit"] is True
    assert hit_doc["cache_key_sha256"] == miss.key_digest
    assert hit_doc["response_bytes_sha256"] == miss_doc["response_bytes_sha256"]
    assert hit_doc["mask_bytes_sha256"] == miss_doc["mask_bytes_sha256"]
    assert np.array_equal(hit.response, miss.response)
    assert np.array_equal(hit.mask, miss.mask)


# ---------------------------------------------------------------------------
# Real harness path: build_centerline keys on the canonical blob sha256
# ---------------------------------------------------------------------------


def test_build_centerline_uses_blob_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _counting_wrappers(monkeypatch)
    raw = _raw()
    parameters = phantom_parameters()
    seed = phantom_seed()
    frame = phantom_frame()
    loader = CanonicalVolumeLoader(tmp_path / "blobs")
    materialized = loader.materialize_canonical(raw, scalar_domain="HU", intensity_unit="HU")
    volume = create_volume(
        study_ref=f"study_id:{ULIDS['study']}",
        source_series_refs=(f"series:{ULIDS['series']}",),
        kind=VolumeKind.CANONICAL,
        coordinate_frame=frame,
        blob_ref=VolumeBlobRef(
            sha256=materialized.blob.sha256,
            codec=CANONICAL_INT16_CODEC,
            byte_size=materialized.blob.byte_size,
        ),
        provenance_ref=f"provenance:{ULIDS['volume_provenance']}",
        release_lane=ReleaseLane.PERSONAL_RESEARCH,
        volume_id=ULIDS["volume"],
    )
    transform = transform_for_frame(frame, series_instance_uid=SERIES_INSTANCE_UID)
    store = ResponseCacheStore(tmp_path / "cache")

    def _build(artifact_id: str) -> Any:
        return build_centerline(
            volume,
            transform,
            frame,
            loader=loader,
            seed=seed,
            parameters=parameters,
            provenance_ref=f"provenance:{ULIDS['centerline_provenance']}",
            transform_artifact_ref=f"coordinate_transform:{ULIDS['transform_artifact']}",
            study_instance_uid=STUDY_INSTANCE_UID,
            frame_of_reference_uid=FRAME_OF_REFERENCE_UID,
            scalar_domain="HU",
            source_array=raw,
            artifact_id=artifact_id,
            response_cache=store,
            created_by="ccta-coronary-fixture",
        )

    expected_key = cache_key_for(
        source_blob_sha256=volume.blob_ref.sha256,
        volume=raw,
        spacing_mm=SPACING,
        parameters=parameters,
    ).digest
    first = _build(ULIDS["centerline_artifact"])
    assert calls == {"frangi": 1, "mask": 1}
    assert _cache_evidence(first.path) == (False, expected_key)
    second = _build(new_ulid())
    assert calls == {"frangi": 1, "mask": 1}, "second build must serve the cache hit"
    assert _cache_evidence(second.path) == (True, expected_key)
    assert (
        first.payload["vesselness_response_sha256"] == second.payload["vesselness_response_sha256"]
    )
    assert first.payload["vessel_mask_sha256"] == second.payload["vessel_mask_sha256"]
    assert first.payload["vessel_mask_sha256"] == first.path.diagnostics.mask_sha256
    assert np.array_equal(first.path.response, second.path.response)
    assert np.array_equal(first.path.mask, second.path.mask)
