"""Canonical ``Volume`` records (Part B §3.3).

Volumes are immutable: resampling/cropping/registration produce a NEW
volume with a new ``kind`` and ``transform_ref`` - never an in-place edit.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.coordinates import FrameSpec
from core.digest import require_sha256_hex
from core.enums import ArtifactStatus, ReleaseLane, VolumeKind
from core.errors import CanonicalValidationError
from core.ids import new_ulid

VOLUME_SCHEMA = "ccta.canonical.volume.v1"


@dataclass(frozen=True)
class VolumeBlobRef:
    """Content-addressed voxel payload binding."""

    sha256: str
    codec: str
    compression: str | None = None
    byte_size: int = 0


@dataclass(frozen=True)
class Volume:
    """Canonical 3D/4D volume consumed by every downstream module (§3.3)."""

    volume_id: str
    study_ref: str
    source_series_refs: tuple[str, ...]
    kind: VolumeKind
    coordinate_frame: FrameSpec
    transform_ref: str | None
    registration_ref: str | None
    blob_ref: VolumeBlobRef
    provenance_ref: str
    release_lane: ReleaseLane
    status: ArtifactStatus = ArtifactStatus.ACTIVE
    schema_version: str = VOLUME_SCHEMA


def create_volume(
    *,
    study_ref: str,
    source_series_refs: tuple[str, ...],
    kind: VolumeKind,
    coordinate_frame: FrameSpec,
    blob_ref: VolumeBlobRef,
    provenance_ref: str,
    release_lane: ReleaseLane,
    transform_ref: str | None = None,
    registration_ref: str | None = None,
    volume_id: str | None = None,
) -> Volume:
    """Validate and freeze a :class:`Volume`.

    Fail-closed rules:
    - ``provenance_ref`` is mandatory (P4/I3, H-007);
    - more than one source series REQUIRES ``registration_ref`` (H-009);
    - every non-CANONICAL volume requires a versioned ``transform_ref`` (I8).
    """
    violations: list[str] = []
    if not study_ref:
        violations.append("study_ref is required")
    if not source_series_refs or any(not r for r in source_series_refs):
        violations.append("source_series_refs must be non-empty")
    if not isinstance(kind, VolumeKind):
        violations.append("kind must be a VolumeKind")
    if len(set(source_series_refs)) != len(source_series_refs):
        violations.append("source_series_refs contains duplicates")
    try:
        require_sha256_hex(blob_ref.sha256, "blob_ref.sha256")
    except ValueError as exc:
        violations.append(str(exc))
    if not blob_ref.codec:
        violations.append("blob_ref.codec is required")
    if blob_ref.byte_size <= 0:
        violations.append("blob_ref.byte_size must be positive")

    if not provenance_ref:
        violations.append("provenance_ref is REQUIRED on every Volume (P4/H-007)")
    if not isinstance(release_lane, ReleaseLane):
        violations.append("release_lane is required at creation (P6)")
    if len(source_series_refs) > 1 and not registration_ref:
        violations.append(
            "registration_ref is REQUIRED when a volume spans >1 series/phase (H-009)"
        )
    if kind != VolumeKind.CANONICAL and not transform_ref:
        violations.append(
            f"transform_ref is REQUIRED for kind={kind.value} (I8: transforms are "
            "versioned artifacts, not hidden state)"
        )

    if violations:
        raise CanonicalValidationError(violations)

    return Volume(
        volume_id=volume_id or new_ulid(),
        study_ref=study_ref,
        source_series_refs=tuple(source_series_refs),
        kind=kind,
        coordinate_frame=coordinate_frame,
        transform_ref=transform_ref,
        registration_ref=registration_ref,
        blob_ref=blob_ref,
        provenance_ref=provenance_ref,
        release_lane=release_lane,
    )


def content_blob_path(digest: str) -> str:
    """Content-addressed storage path ``blobs/sha256/<2-hex>/<digest>`` (I6)."""
    digest = require_sha256_hex(digest, "digest")
    return f"blobs/sha256/{digest[:2]}/{digest}"


__all__ = ["VOLUME_SCHEMA", "Volume", "VolumeBlobRef", "content_blob_path", "create_volume"]
