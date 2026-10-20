"""Seed-independent vesselness response/mask cache (F30-OPT, additive).

The Frangi response and its mask are pure functions of the source bytes, the
physical spacing and the frozen parameters - the human seed only enters
downstream (``core.coronary.centerline.derive_coronary_path`` thresholds and
traces, it never changes the response).  This module therefore caches the
``(response, mask)`` pair under a seed-free key so a second seed on the same
volume pays no vesselness recompute.

Reuse-first record (prior art, all ``INTERNAL_REUSE``; adapted from the
untracked ``scratch/core/coronary/vesselness_cache.py`` spike, which was
never wired into the derive path and is superseded by this module):

* key building through :func:`core.digest.canonical_digest` (the single
  canonical hash of the repo);
* CAS layout through :func:`core.volume.content_blob_path`
  (``blobs/sha256/<2-hex>/<digest>`` + a type suffix, mirroring
  ``apps/workstation/volume_loader.py`` which appends ``.bin``/``.json``);
* atomic publish (``.tmp`` + rename) and verify-on-read (SHA-256 check)
  mirroring ``CanonicalVolumeLoader.write_blob``/``read_blob`` (this module
  re-implements those two operations locally because ``core`` must never
  import from ``apps``);
* algorithm identity from :mod:`core.coronary.types`
  (``CORONARY_ALGORITHM``/``CORONARY_ALGORITHM_VERSION``) so a champion
  change can never silently reuse a stale entry (Rule H).

Explicit invalidation rules (no hidden invalidation anywhere):

1. The key binds the source digest, the source-digest KIND (canonical blob
   address vs. array-content fallback - two namespaces that must never be
   confused), the source shape, the source dtype (canonical ``dtype.str``
   including byte order, so byte-identical blobs under different dtypes
   never share a key), the physical spacing, the parameters
   SHA-256, the algorithm identity and the cache schema version.  Any change
   to any of them is a different key and therefore a natural miss - there is
   no TTL, no LRU and no eviction.
2. Entries are immutable once published; :meth:`ResponseCacheStore.store`
   republishes atomically through a per-writer unique temp name plus
   rename (identical inputs yield identical bytes), so concurrent writers
   are safe (last-writer-wins with identical bytes).
3. A corrupt or unverifiable entry (unreadable manifest, digest mismatch,
   shape/plausibility failure) is NEVER served and NEVER raises into the
   derivation: the lookup reports a miss and the caller recomputes.  Use
   :meth:`ResponseCacheStore.invalidate` to drop such entries explicitly.
4. Provenance records cache hits explicitly: the derive path records
   ``response_cache_hit`` / ``response_cache_key`` in the derivation
   diagnostics, and every :func:`cached_frangi_vesselness` result carries
   :meth:`CachedVesselness.provenance_document` with a ``cache_hit`` flag,
   the key digest and the stored byte digests, for the caller to embed.

NumPy-only; no dependency is added.  No arithmetic, threshold, scale,
gamma, domain or precision value lives here - this module only stores and
serves bytes produced by the single governed path
(:func:`frangi_vesselness` + :func:`vessel_mask`).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

from core.coronary.types import (
    CORONARY_ALGORITHM,
    CORONARY_ALGORITHM_VERSION,
    VesselnessParameters,
)
from core.coronary.vesselness import (
    RESPONSE_DTYPE,
    _require_spacing,
    frangi_vesselness,
    vessel_mask,
)
from core.digest import canonical_digest, is_sha256_hex
from core.volume import content_blob_path

#: Schema of the cache manifest and of the cache-key document.  A schema
#: change is a new key namespace (old entries become natural misses).
CACHE_SCHEMA_VERSION: Final[str] = "ccta.coronary.vesselness_cache.v1"
#: Closed vocabulary of what the key's source digest identifies.  The kind
#: is key material so a canonical blob address can never be confused with
#: an array-content fallback digest.
SOURCE_DIGEST_KINDS: Final[frozenset[str]] = frozenset(
    {"volume-blob-sha256", "array-content-sha256"}
)
#: Upper bound accepted for any single manifest dimension (fail-closed
#: against a forged manifest asking for a gigantic allocation).
_MAX_MANIFEST_DIM: Final[int] = 4096


def source_content_sha256(volume: np.ndarray) -> str:
    """Content digest of a source array (array-content fallback key component).

    Covers the dtype, the shape and the C-order bytes, so two arrays share
    a digest iff they feed byte-identical samples into the derivation.  Used
    when no canonical blob address is available (direct
    :func:`derive_coronary_path` calls); the ``build_centerline`` path keys
    on the canonical ``volume-blob-sha256`` instead.
    """

    array = np.asarray(volume)
    if array.ndim != 3:
        raise ValueError("volume must be a 3-D array (k, j, i)")
    hasher = hashlib.sha256()
    hasher.update(str(array.dtype).encode("ascii"))
    hasher.update(json.dumps([int(v) for v in array.shape]).encode("ascii"))
    hasher.update(np.ascontiguousarray(array).tobytes(order="C"))
    return hasher.hexdigest()


@dataclass(frozen=True)
class VesselnessCacheKey:
    """Seed-free identity of one cached ``(response, mask)`` pair.

    ``source_shape`` is ``(k, j, i)``; ``spacing_mm`` is ``(i, j, k)`` -
    the same conventions as the derivation itself.  No seed field exists by
    construction (seed-independence is structural, not a convention).
    """

    source_blob_sha256: str
    source_shape: tuple[int, int, int]
    spacing_mm: tuple[float, float, float]
    parameters_sha256: str
    source_dtype: str
    algorithm: str = CORONARY_ALGORITHM
    algorithm_version: str = CORONARY_ALGORITHM_VERSION
    source_kind: str = "volume-blob-sha256"

    def __post_init__(self) -> None:
        if not is_sha256_hex(self.source_blob_sha256):
            raise ValueError("source_blob_sha256 must be a 64-hex sha256 digest")
        if not is_sha256_hex(self.parameters_sha256):
            raise ValueError("parameters_sha256 must be a 64-hex sha256 digest")
        if (
            not isinstance(self.source_shape, tuple)
            or len(self.source_shape) != 3
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in self.source_shape
            )
        ):
            raise ValueError("source_shape must be three positive integers (k, j, i)")
        if (
            not isinstance(self.spacing_mm, tuple)
            or len(self.spacing_mm) != 3
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(float(value))
                or float(value) <= 0.0
                for value in self.spacing_mm
            )
        ):
            raise ValueError("spacing_mm must be three positive finite numbers (i, j, k)")
        if self.source_kind not in SOURCE_DIGEST_KINDS:
            raise ValueError(f"source_kind must be one of {sorted(SOURCE_DIGEST_KINDS)}")
        try:
            _parsed_dtype = np.dtype(self.source_dtype)
        except TypeError as exc:
            raise ValueError(
                "source_dtype must be a canonical numpy dtype.str including byte order"
            ) from exc
        if self.source_dtype != _parsed_dtype.str:
            raise ValueError(
                "source_dtype must be canonical numpy dtype.str including byte order (e.g. '<i2')"
            )

    def to_document(self) -> dict[str, object]:
        return {
            "schema_version": CACHE_SCHEMA_VERSION,
            "source_volume_blob_sha256": self.source_blob_sha256,
            "source_digest_kind": self.source_kind,
            "source_volume_shape": [int(value) for value in self.source_shape],
            "source_volume_dtype": self.source_dtype,
            "voxel_spacing_mm": [float(value) for value in self.spacing_mm],
            "parameters_sha256": self.parameters_sha256,
            "algorithm": self.algorithm,
            "algorithm_version": self.algorithm_version,
        }

    @property
    def digest(self) -> str:
        """The CAS key: canonical digest of :meth:`to_document`."""

        return canonical_digest(self.to_document())


@dataclass(frozen=True)
class CachedVesselness:
    """One cache outcome: read-only arrays plus explicit hit evidence."""

    response: np.ndarray
    mask: np.ndarray
    cache_hit: bool
    key_digest: str
    source_blob_sha256: str
    response_bytes_sha256: str
    mask_bytes_sha256: str
    compute_wall_s: float

    def provenance_document(self) -> dict[str, object]:
        """Explicit hit/miss record for the caller to embed in provenance."""

        return {
            "schema_version": CACHE_SCHEMA_VERSION,
            "cache_hit": bool(self.cache_hit),
            "cache_key_sha256": self.key_digest,
            "source_volume_blob_sha256": self.source_blob_sha256,
            "response_bytes_sha256": self.response_bytes_sha256,
            "mask_bytes_sha256": self.mask_bytes_sha256,
            "compute_wall_s": float(self.compute_wall_s),
        }


@dataclass(frozen=True)
class StoredVesselness:
    """Verified bytes of one cache entry (read-only arrays + byte digests)."""

    response: np.ndarray
    mask: np.ndarray
    response_sha256: str
    mask_sha256: str


class ResponseCacheStore:
    """Filesystem CAS for cached ``(response, mask)`` pairs.

    ``root`` is any directory (tests use ``tmp_path``); layout, atomic
    publish and verify-on-read mirror the workspace blob store
    (``CanonicalVolumeLoader.write_blob``/``read_blob`` semantics, without
    importing ``apps`` from ``core``).
    """

    def __init__(self, root: Path | str) -> None:
        self._root = Path(root)

    def _paths(self, key_digest: str) -> tuple[Path, Path, Path]:
        base = self._root / content_blob_path(key_digest)
        return (
            base.with_name(base.name + ".response.bin"),
            base.with_name(base.name + ".mask.bin"),
            base.with_name(base.name + ".json"),
        )

    @staticmethod
    def _write_atomic(path: Path, data: bytes) -> None:
        # Unique temp name per writer: concurrent derivations publishing the
        # same key must never share a temp pathname (shared `.tmp` + rename
        # races lose the rename with FileNotFoundError and alias inodes).
        # Same-directory unique temp + rename keeps the publish atomic with
        # last-writer-wins on identical bytes.
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        try:
            tmp.write_bytes(data)
            tmp.replace(path)
        except Exception:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            raise

    @staticmethod
    def _verify_bytes(path: Path, expected_sha256: str) -> bytes | None:
        try:
            data = path.read_bytes()
        except OSError:
            return None
        if hashlib.sha256(data).hexdigest() != expected_sha256:
            return None
        return data

    def store(
        self, key: VesselnessCacheKey, response: np.ndarray, mask: np.ndarray
    ) -> StoredVesselness:
        """Publish one entry atomically; returns the stored byte digests."""

        response_array = np.asarray(response)
        mask_array = np.asarray(mask)
        if response_array.ndim != 3 or mask_array.shape != response_array.shape:
            raise ValueError("response and mask must be 3-D arrays of equal shape")
        if response_array.dtype != RESPONSE_DTYPE:
            raise ValueError(f"response must be {RESPONSE_DTYPE} (frozen precision)")
        if mask_array.dtype != np.dtype(bool):
            raise ValueError("mask must be a bool array")
        if [int(value) for value in response_array.shape] != [
            int(value) for value in key.source_shape
        ]:
            raise ValueError("response shape disagrees with the cache key shape")

        response_bytes = np.ascontiguousarray(response, dtype=np.float64).tobytes(order="C")
        mask_bytes = np.ascontiguousarray(mask, dtype=np.uint8).tobytes(order="C")
        response_sha256 = hashlib.sha256(response_bytes).hexdigest()
        mask_sha256 = hashlib.sha256(mask_bytes).hexdigest()
        manifest = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "cache_key_sha256": key.digest,
            "key_document": key.to_document(),
            "source_volume_shape": [int(value) for value in key.source_shape],
            "response": {
                "sha256": response_sha256,
                "byte_size": len(response_bytes),
                "dtype": "<f8",
            },
            "mask": {
                "sha256": mask_sha256,
                "byte_size": len(mask_bytes),
                "dtype": "u8-bool",
            },
        }
        response_path, mask_path, manifest_path = self._paths(key.digest)
        self._write_atomic(response_path, response_bytes)
        self._write_atomic(mask_path, mask_bytes)
        self._write_atomic(
            manifest_path,
            (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode("utf-8"),
        )
        response_out = np.ascontiguousarray(response, dtype=np.float64)
        mask_out = np.ascontiguousarray(mask, dtype=bool)
        response_out.setflags(write=False)
        mask_out.setflags(write=False)
        return StoredVesselness(
            response=response_out,
            mask=mask_out,
            response_sha256=response_sha256,
            mask_sha256=mask_sha256,
        )

    def lookup(self, key: VesselnessCacheKey) -> StoredVesselness | None:
        """Return the verified entry, or ``None`` on miss/corruption.

        Corruption is a miss by explicit rule (never served, never raised):
        unreadable manifest, schema/key/shape mismatch, implausible
        dimensions, missing blobs or digest mismatches all yield ``None``.
        """

        response_path, mask_path, manifest_path = self._paths(key.digest)
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return None
        if not isinstance(manifest, dict):
            return None
        if manifest.get("schema_version") != CACHE_SCHEMA_VERSION:
            return None
        if manifest.get("cache_key_sha256") != key.digest:
            return None
        shape = manifest.get("source_volume_shape")
        if (
            not isinstance(shape, list)
            or len(shape) != 3
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in shape
            )
            or any(int(value) > _MAX_MANIFEST_DIM for value in shape)
            or [int(value) for value in shape] != [int(value) for value in key.source_shape]
        ):
            return None
        dims = (int(shape[0]), int(shape[1]), int(shape[2]))
        response_meta = manifest.get("response")
        mask_meta = manifest.get("mask")
        if not isinstance(response_meta, dict) or not isinstance(mask_meta, dict):
            return None
        if response_meta.get("dtype") != "<f8" or mask_meta.get("dtype") != "u8-bool":
            return None
        expected_response_bytes = dims[0] * dims[1] * dims[2] * 8
        expected_mask_bytes = dims[0] * dims[1] * dims[2]
        if (
            response_meta.get("byte_size") != expected_response_bytes
            or mask_meta.get("byte_size") != expected_mask_bytes
        ):
            return None
        response_sha256 = response_meta.get("sha256")
        mask_sha256 = mask_meta.get("sha256")
        if not is_sha256_hex(response_sha256) or not is_sha256_hex(mask_sha256):
            return None
        response_bytes = self._verify_bytes(response_path, str(response_sha256))
        mask_bytes = self._verify_bytes(mask_path, str(mask_sha256))
        if response_bytes is None or mask_bytes is None:
            return None
        if len(response_bytes) != expected_response_bytes or len(mask_bytes) != expected_mask_bytes:
            return None
        response = np.frombuffer(response_bytes, dtype=RESPONSE_DTYPE).reshape(dims)
        mask = np.frombuffer(mask_bytes, dtype=np.uint8).reshape(dims).astype(bool)
        response.setflags(write=False)
        mask.setflags(write=False)
        return StoredVesselness(
            response=np.ascontiguousarray(response),
            mask=np.ascontiguousarray(mask),
            response_sha256=str(response_sha256),
            mask_sha256=str(mask_sha256),
        )

    def invalidate(self, key: VesselnessCacheKey) -> bool:
        """Drop one entry explicitly; returns True iff anything was removed."""

        removed = False
        for path in self._paths(key.digest):
            try:
                path.unlink()
                removed = True
            except OSError:
                continue
        return removed


def cache_key_for(
    *,
    source_blob_sha256: str,
    volume: np.ndarray,
    spacing_mm: tuple[float, float, float],
    parameters: VesselnessParameters,
    source_kind: str = "volume-blob-sha256",
) -> VesselnessCacheKey:
    """Build the seed-free cache key for one derivation input set."""

    array = np.asarray(volume)
    if array.ndim != 3:
        raise ValueError("volume must be a 3-D array (k, j, i)")
    if not isinstance(parameters, VesselnessParameters):
        raise ValueError("parameters must be a VesselnessParameters instance")
    # The SAME spacing gate as the uncached path: a malformed spacing must
    # refuse identically cold and warm, never serve a hit on the coerced
    # floats.  Shared validator (no duplicated logic).
    spacing = _require_spacing(spacing_mm)
    shape = (int(array.shape[0]), int(array.shape[1]), int(array.shape[2]))
    return VesselnessCacheKey(
        source_blob_sha256=source_blob_sha256,
        source_shape=shape,
        spacing_mm=spacing,
        parameters_sha256=parameters.sha256,
        source_dtype=array.dtype.str,
        source_kind=source_kind,
    )


def cached_frangi_vesselness(
    volume: np.ndarray,
    spacing_mm: tuple[float, float, float],
    parameters: VesselnessParameters,
    *,
    source_blob_sha256: str,
    store: ResponseCacheStore,
    source_kind: str = "volume-blob-sha256",
) -> CachedVesselness:
    """Return ``(response, mask)``, computing once and serving CAS hits after.

    A miss runs the single governed path (:func:`frangi_vesselness` +
    :func:`vessel_mask` - no duplicate implementation) and publishes the
    bytes; a hit serves verified bytes without touching vesselness.  The
    ``cache_hit`` flag plus :meth:`CachedVesselness.provenance_document`
    are the explicit hit evidence (in particular, on a hit
    ``compute_wall_s`` is ``0.0`` because nothing was computed).

    A publication (store) failure AFTER the compute is NOT a recompute: the
    already-computed arrays are returned as a recorded miss (``cache_hit``
    is ``False``), with the stored byte digests computed locally over the
    same bytes the store would have published.  Only lookup/computation
    failures leave the caller with nothing to serve.
    """

    if not is_sha256_hex(source_blob_sha256):
        raise ValueError("source_blob_sha256 must be a 64-hex sha256 digest")
    if not isinstance(parameters, VesselnessParameters):
        raise ValueError("parameters must be a VesselnessParameters instance")
    key = cache_key_for(
        source_blob_sha256=source_blob_sha256,
        volume=volume,
        spacing_mm=spacing_mm,
        parameters=parameters,
        source_kind=source_kind,
    )
    hit = store.lookup(key)
    if hit is not None:
        return CachedVesselness(
            response=hit.response,
            mask=hit.mask,
            cache_hit=True,
            key_digest=key.digest,
            source_blob_sha256=source_blob_sha256,
            response_bytes_sha256=hit.response_sha256,
            mask_bytes_sha256=hit.mask_sha256,
            compute_wall_s=0.0,
        )
    started = time.perf_counter()
    response = frangi_vesselness(volume, spacing_mm, parameters)
    mask = vessel_mask(response, parameters)
    wall = time.perf_counter() - started
    try:
        stored = store.store(key, response, mask)
    except OSError:
        # Publication failure after a valid compute: serve the computed
        # arrays as a recorded miss instead of forcing the caller to pay
        # for a second compute.  Digests cover exactly the bytes the store
        # would have published (C-order float64 response, uint8 mask).
        response_sha256 = hashlib.sha256(
            np.ascontiguousarray(response, dtype=np.float64).tobytes(order="C")
        ).hexdigest()
        mask_sha256 = hashlib.sha256(
            np.ascontiguousarray(mask, dtype=np.uint8).tobytes(order="C")
        ).hexdigest()
        response_out = np.ascontiguousarray(response)
        mask_out = np.ascontiguousarray(mask)
        response_out.setflags(write=False)
        mask_out.setflags(write=False)
        return CachedVesselness(
            response=response_out,
            mask=mask_out,
            cache_hit=False,
            key_digest=key.digest,
            source_blob_sha256=source_blob_sha256,
            response_bytes_sha256=response_sha256,
            mask_bytes_sha256=mask_sha256,
            compute_wall_s=wall,
        )
    response_out = np.ascontiguousarray(response)
    mask_out = np.ascontiguousarray(mask)
    response_out.setflags(write=False)
    mask_out.setflags(write=False)
    return CachedVesselness(
        response=response_out,
        mask=mask_out,
        cache_hit=False,
        key_digest=key.digest,
        source_blob_sha256=source_blob_sha256,
        response_bytes_sha256=stored.response_sha256,
        mask_bytes_sha256=stored.mask_sha256,
        compute_wall_s=wall,
    )


__all__ = [
    "CACHE_SCHEMA_VERSION",
    "SOURCE_DIGEST_KINDS",
    "CachedVesselness",
    "ResponseCacheStore",
    "StoredVesselness",
    "VesselnessCacheKey",
    "cache_key_for",
    "cached_frangi_vesselness",
    "source_content_sha256",
]
