"""LAYER-G frame-list construction and LAYER-R variant minting (Part 5.1).

LAYER-G construction (per session, per plane): the source set is every
ACCEPTED ``MPR_VIEW`` artifact of that plane whose ``source_volume_ref``
resolves to a volume in the restored WS-02 session.  The ordering key is
``plane_coordinate_mm`` ascending, tie-broken by ``mpr_artifact_id`` ULID
ascending (byte order); ``frame_index`` is 0-based over that order.  A plane
with zero qualifying MPRs has an empty frame list (legal; scrolling reports
``INDEX_OUT_OF_RANGE``; the viewer displays an explicit empty-state, never a
fabricated image).

Frame identity (Part 5.1): the stable identity tuple of a frame RECORD is
exactly ``(session_id, plane, frame_index, source_mpr_ref, wl_preset_id,
wl_applied)``.  Two frame records with equal tuples are the SAME logical
frame.  The frame ULID is record-identifier metadata ONLY and is EXCLUDED
from byte-identity checks.  A WL re-mint is byte-reproducible on the pinned
runtime (pure function of the MPR pixel bytes + the frozen transfer function)
even though the new record carries a fresh ULID, and it NEVER alters
``frame_index`` or the LAYER-G list digest.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from apps.viewer.build_identity import (
    BuildIdentityError,
    repo_root_from_file,
    resolve_build_identity,
    validate_commit_sha,
)
from apps.viewer.png_encoder import encode_png, pixel_digest
from apps.viewer.presets import WLPreset
from apps.viewer.schemas import PLANES, ViewerFrameRecord
from apps.viewer.transfer import (
    TRANSFER_FUNCTION_ID,
    exact_extrema,
    identity_window_bounds,
    window_bytes,
)
from core.artifacts import DerivedArtifact
from core.enums import ArtifactKind, ProvenanceEventType, ReviewState
from core.ids import new_ulid
from core.provenance import (
    Actor,
    ActorType,
    BlobRef,
    Environment,
    InputRef,
    NotApplicable,
    ProvenanceRecord,
    create_provenance,
)
from visualization.mpr import decode_pixel_blob

#: Review states that count as an ACCEPTED artifact for LAYER-G binding.
#: PREDICTED (not yet accepted), CORRECTED (awaiting re-acceptance), and
#: REJECTED are never rendered by the viewer (fail-closed).
ACCEPTED_REVIEW_STATES: Final[frozenset[ReviewState]] = frozenset(
    {ReviewState.ACCEPTED, ReviewState.CONFIRMED_FINAL}
)

#: The ONLY scalar domain the viewer's HU-semantic display/measurement paths
#: may consume (VIEWER_001_SPEC Part 4.1: the WL arithmetic is over HU and
#: never over stored values with a different intercept assumption).
HU_SCALAR_DOMAIN: Final[str] = "HU"

#: The MPR payload's recorded scalar domain (``visualization.mpr`` closed
#: vocabulary).  A payload written before the domain was recorded carries NO
#: such key: that is a distinct, record-free state and it must fail closed.
MPR_SCALAR_DOMAIN_KEY: Final[str] = "scalar_domain"

#: The closed scalar-domain vocabulary of an MPR payload — mirrored from
#: ``visualization.mpr`` (read-only mirror; never widened here).
_MPR_SCALAR_DOMAINS: Final[frozenset[str]] = frozenset(
    {"STORED", "MODALITY_TRANSFORMED", "HU"}
)

_FRAME_PROVENANCE_SUBJECT_PREFIX: Final[str] = "viewer_frame"
_VIEWER_ACTOR: Final[Actor] = Actor(actor_type=ActorType.ENGINE, actor_id="ccta-viewer")
_VIEWER_CREATED_BY: Final[str] = "ccta_viewer"
_ZLIB_VERSION_IN_PARAMETERS: Final[str] = (
    "zlib.ZLIB_VERSION recorded in the frame provenance parameters ref"
)


class FrameListConstructionError(ValueError):
    """A fail-closed session-open construction failure (Part 6 table)."""


class NonHUScalarDomainError(FrameListConstructionError):
    """An accepted MPR's source volume is not in the HU scalar domain (F18).

    The viewer's frozen window/level presets (VIEWER_001_SPEC Part 4) and the
    ``*_hu`` measurement fields are defined over HU; applying them to
    decoded samples whose canonical source volume records a NON-HU scalar
    domain silently reinterprets those samples.  Raised at LAYER-G
    frame-list construction so no such frame ever reaches a render or
    measurement path (fail-closed, never a silent skip).

    This is the ONLY new member of the vocabulary and it is a strict SUBCLASS
    of :class:`FrameListConstructionError` — the existing closed viewer
    vocabulary is extended by specialisation, not by a new top-level type:
    every existing ``except FrameListConstructionError`` site (in particular
    ``ViewerEngine._build_geometry``, which translates it into the existing
    typed ``RejectionCode.SESSION_OPEN_FAILED``, HTTP 409) keeps working
    unchanged and still refuses the open.  Absent or record-free domains are
    their own distinct state and fail closed here too.
    """


@dataclass(frozen=True, slots=True)
class GeometryFrame:
    """One LAYER-G geometry binding."""

    plane: str
    frame_index: int
    mpr_artifact_id: str
    plane_coordinate_mm: float
    source_volume_ref: str


@dataclass(frozen=True, slots=True)
class RenderMint:
    """One minted LAYER-R variant record plus its provenance."""

    record: ViewerFrameRecord
    provenance: ProvenanceRecord
    canvas_digest: str


def _accepted_mpr_artifacts(artifacts: tuple[DerivedArtifact, ...]) -> list[DerivedArtifact]:
    return [
        artifact
        for artifact in artifacts
        if artifact.kind == ArtifactKind.MPR_VIEW
        and artifact.review.review_state in ACCEPTED_REVIEW_STATES
    ]


def _require_hu_scalar_domain(
    *,
    mpr_id: str,
    document: Mapping[str, object],
    volume_scalar_domains: Mapping[str, str] | None,
) -> None:
    """Fail closed unless the MPR's source volume RECORDS the HU domain.

    Keyed on the RECORDED scalar domain of the canonical source volume — the
    value the session actually reopens with (``build_mpr`` copies it from the
    volume envelope into the MPR payload, where it is bound by
    ``sampling_parameters_sha256``).  NEVER a guessed or defaulted domain:
    an absent payload key, a non-string value, a domain outside the closed
    MPR vocabulary, or a payload/envelope disagreement is its own distinct
    refusal state.  The envelope is authoritative where it is supplied
    (``volume_scalar_domains``), and it must agree with the payload it
    produced.
    """

    payload_domain = document.get(MPR_SCALAR_DOMAIN_KEY)
    if not isinstance(payload_domain, str) or isinstance(payload_domain, bool):
        raise NonHUScalarDomainError(
            f"MPR_VIEW {mpr_id}: payload records no usable scalar_domain; "
            "a record-free domain cannot reach an HU-semantic viewer path"
        )
    if payload_domain not in _MPR_SCALAR_DOMAINS:
        raise NonHUScalarDomainError(
            f"MPR_VIEW {mpr_id}: scalar_domain {payload_domain!r} is outside the "
            "closed vocabulary"
        )
    if volume_scalar_domains is not None:
        source = document.get("source_volume_ref")
        volume_id = source.split(":", 1)[1] if isinstance(source, str) and ":" in source else None
        envelope_domain = (
            volume_scalar_domains.get(volume_id)
            if volume_id is not None
            else None
        )
        if envelope_domain is None:
            raise NonHUScalarDomainError(
                f"MPR_VIEW {mpr_id}: source volume {volume_id} records no scalar "
                "domain; a record-free domain cannot reach an HU-semantic viewer path"
            )
        if envelope_domain != payload_domain:
            raise NonHUScalarDomainError(
                f"MPR_VIEW {mpr_id}: payload scalar_domain {payload_domain!r} "
                f"disagrees with the source volume's recorded {envelope_domain!r}"
            )
    if payload_domain != HU_SCALAR_DOMAIN:
        raise NonHUScalarDomainError(
            f"MPR_VIEW {mpr_id}: source volume scalar domain is {payload_domain!r}, "
            f"not {HU_SCALAR_DOMAIN!r}; the HU-semantic presets and *_hu measurement "
            "labels must never be applied to these samples"
        )


def build_geometry_lists(
    session_id: str,
    artifacts: tuple[DerivedArtifact, ...],
    payload_documents: dict[str, dict[str, object]],
    volume_ids: tuple[str, ...],
    volume_scalar_domains: Mapping[str, str] | None = None,
) -> dict[str, list[GeometryFrame]]:
    """Construct the closed LAYER-G frame lists per plane (Part 5.1).

    ``volume_ids`` are the volumes of the restored WS-02 session.  Raises
    :class:`FrameListConstructionError` on any dangling reference, missing
    payload, unreadable pixel blob, or payload/geometry inconsistency —
    never skips a frame silently.

    ``volume_scalar_domains`` (optional) maps ``volume_id -> recorded
    canonical scalar domain`` for the restored session, i.e. the envelopes
    the session actually reopens with.  When supplied, the F18 semantic gate
    is enforced: every accepted MPR_VIEW must record the HU domain and must
    agree with its source volume's envelope; anything else (including a
    record-free domain) is a typed :class:`NonHUScalarDomainError`.  Callers
    that cannot supply the mapping get the payload-side gate only, which is
    still fail-closed on any non-HU or record-free payload.
    """

    volume_set = set(volume_ids)
    candidates: dict[str, list[tuple[float, str]]] = {plane: [] for plane in PLANES}
    accepted = _accepted_mpr_artifacts(artifacts)
    for artifact in accepted:
        document = payload_documents.get(artifact.artifact_id)
        if document is None:
            raise FrameListConstructionError(
                f"dangling MPR_VIEW {artifact.artifact_id}: payload document missing"
            )
        plane = document.get("plane")
        if plane not in PLANES:
            raise FrameListConstructionError(
                f"MPR_VIEW {artifact.artifact_id}: plane {plane!r} outside the closed enum"
            )
        source_volume_ref = document.get("source_volume_ref")
        if not isinstance(source_volume_ref, str) or not source_volume_ref.startswith(
            "volume_id:"
        ):
            raise FrameListConstructionError(
                f"MPR_VIEW {artifact.artifact_id}: source_volume_ref has the wrong form"
            )
        volume_id = source_volume_ref.split(":", 1)[1]
        if volume_id not in volume_set:
            raise FrameListConstructionError(
                f"MPR_VIEW {artifact.artifact_id}: source volume {volume_id} does not "
                "resolve to a volume in the restored session"
            )
        coordinate = document.get("plane_coordinate_mm")
        if isinstance(coordinate, bool) or not isinstance(coordinate, (int, float)):
            raise FrameListConstructionError(
                f"MPR_VIEW {artifact.artifact_id}: plane_coordinate_mm is not numeric"
            )
        _require_hu_scalar_domain(
            mpr_id=artifact.artifact_id,
            document=document,
            volume_scalar_domains=volume_scalar_domains,
        )
        candidates[str(plane)].append((float(coordinate), artifact.artifact_id))

    lists: dict[str, list[GeometryFrame]] = {}
    for plane in PLANES:
        ordered = sorted(candidates[plane], key=lambda item: (item[0], item[1]))
        lists[plane] = [
            GeometryFrame(
                plane=plane,
                frame_index=index,
                mpr_artifact_id=mpr_id,
                plane_coordinate_mm=coordinate,
                source_volume_ref=f"volume_id:{_volume_of(mpr_id, artifacts, payload_documents)}",
            )
            for index, (coordinate, mpr_id) in enumerate(ordered)
        ]
    return lists


def _volume_of(
    mpr_id: str,
    artifacts: tuple[DerivedArtifact, ...],
    payload_documents: dict[str, dict[str, object]],
) -> str:
    document = payload_documents.get(mpr_id)
    if document is None:
        raise FrameListConstructionError(
            f"dangling MPR_VIEW {mpr_id}: payload document missing"
        )
    source = document.get("source_volume_ref")
    if not isinstance(source, str) or ":" not in source:
        raise FrameListConstructionError(
            f"MPR_VIEW {mpr_id}: source_volume_ref has the wrong form"
        )
    return source.split(":", 1)[1]


def _frame_parameter_document(
    *,
    wl_preset_id: str | None,
    center: float,
    width: float,
    canvas_digest: str,
    source_pixel_digest: str,
) -> dict[str, object]:
    from apps.viewer.png_encoder import (
        PNG_ENCODER_ID,
        ZLIB_LEVEL,
        ZLIB_MEM_LEVEL,
        ZLIB_STRATEGY,
        ZLIB_VERSION_PIN,
        ZLIB_WBITS,
    )

    return {
        "transfer_function": TRANSFER_FUNCTION_ID,
        "png_encoder": PNG_ENCODER_ID,
        "zlib_version": ZLIB_VERSION_PIN,
        "zlib_level": ZLIB_LEVEL,
        "zlib_wbits": ZLIB_WBITS,
        "zlib_mem_level": ZLIB_MEM_LEVEL,
        "zlib_strategy": ZLIB_STRATEGY,
        "wl_preset_id": wl_preset_id,
        "wl_applied": {"center": center, "width": width},
        "semantic_pixel_digest": canvas_digest,
        "source_pixel_digest": source_pixel_digest,
    }


def _frame_parameters_blob(
    parameter_document: dict[str, object],
) -> tuple[BlobRef, str]:
    from apps.viewer.png_encoder import ZLIB_VERSION_PIN
    from core.digest import canonical_digest, canonical_json

    document = dict(parameter_document)
    document["zlib_version_note"] = _ZLIB_VERSION_IN_PARAMETERS
    document_json = canonical_json(document)
    sha256 = canonical_digest(document)
    # The pinned runtime is literally visible in the parameters ref: the
    # spec's DETERMINISM SCOPE requires zlib.ZLIB_VERSION in every frame
    # record's provenance parameters.
    return (
        BlobRef(
            blob_ref=f"viewer-frame-parameters:zlib-{ZLIB_VERSION_PIN}:{sha256}",
            sha256=sha256,
        ),
        document_json,
    )


def mint_render_variant(
    *,
    session_id: str,
    geometry: GeometryFrame,
    wl_preset_id: str | None,
    center: float,
    width: float,
    mpr_artifact: DerivedArtifact,
    mpr_payload: dict[str, object],
    pixel_blob: bytes,
    provenance_upstream: tuple[str, ...],
    code_commit_sha: str,
    frame_id: str | None = None,
    at: str | None = None,
    id_factory: object = None,
    repo_root: Path | str | None = None,
) -> RenderMint:
    """Mint ONE LAYER-R variant record for a geometry frame (pure function).

    ``center``/``width`` are the EFFECTIVE window actually applied (for a
    preset the preset's stored values; for the raw mapping the computed
    identity values, already recorded and passed in).  The render is a pure
    function of the MPR pixel bytes + the frozen transfer function: no
    resampling, no geometry recomputation.

    ``code_commit_sha`` is a REQUIRED keyword argument carrying the governed
    implementation identity (r2 F11): it must already be resolved and
    verified by the caller's checkout, and it is re-validated here so a
    standalone mint can never stamp a syntax-only caller string, a fallback
    constant, or forty zeros into frame provenance.  ``repo_root`` pins the
    checkout the identity must exist in (the engine passes its resolved
    root; the default derives it from this module's location).
    """

    del id_factory  # reserved; frame ids come from the caller's factory
    resolved_root = repo_root_from_file()
    if repo_root is not None and Path(repo_root).resolve() != resolved_root:
        raise BuildIdentityError(
            f"supplied repo_root {repo_root} does not match executing code root {resolved_root}"
        )
    current_sha = resolve_build_identity(resolved_root)
    validated_commit_sha = validate_commit_sha(code_commit_sha)
    if validated_commit_sha != current_sha:
        raise BuildIdentityError(
            f"supplied commit identity {code_commit_sha} does not match "
            f"current producing checkout {current_sha}"
        )
    pixels = decode_pixel_blob(mpr_payload, pixel_blob)
    canvas = window_bytes(pixels, center, width)
    canvas_digest = pixel_digest(canvas)
    png_bytes = encode_png(canvas)
    import hashlib

    png_sha256 = hashlib.sha256(png_bytes).hexdigest()
    source_pixel_digest = hashlib.sha256(
        pixels.tobytes(order="C")
    ).hexdigest()

    parameter_document = _frame_parameter_document(
        wl_preset_id=wl_preset_id,
        center=center,
        width=width,
        canvas_digest=canvas_digest,
        source_pixel_digest=source_pixel_digest,
    )
    parameters_ref, _document_json = _frame_parameters_blob(parameter_document)

    mint = new_ulid if frame_id is None else (lambda: frame_id)
    frame_ulid = mint()
    provenance = create_provenance(
        subject_ref=f"{_FRAME_PROVENANCE_SUBJECT_PREFIX}:{frame_ulid}",
        event_type=ProvenanceEventType.DERIVED,
        actor=_VIEWER_ACTOR,
        inputs=(
            InputRef(
                ref=f"mpr_artifact_id:{mpr_artifact.artifact_id}",
                kind="artifact",
                sha256_at_use=mpr_artifact.payload.blob_sha256,
            ),
        ),
        preprocessing="viewer-display-render-no-resampling",
        parameters_ref=parameters_ref,
        human_edit=NotApplicable(
            value=None, reason="deterministic viewer render; no human edit"
        ),
        upstream_refs=provenance_upstream,
        environment=Environment(
            software_version="ccta-viewer-0.1.0",
            code_commit_sha=validated_commit_sha,
            runtime="python-stdlib-viewer",
            deterministic_seed=None,
        ),
        model_record_ref=None,
        created_by=_VIEWER_CREATED_BY,
        at=at,
        provenance_id=None,
    )
    record = ViewerFrameRecord(
        schema_version="ccta.viewer.frame.v1",
        session_id=session_id,
        plane=geometry.plane,
        frame_index=geometry.frame_index,
        source_mpr_ref=f"mpr_artifact_id:{geometry.mpr_artifact_id}",
        wl_preset_id=wl_preset_id,
        wl_applied={"center": float(center), "width": float(width)},
        png_sha256=png_sha256,
        png_byte_size=len(png_bytes),
        provenance_ref=f"provenance:{provenance.provenance_id}",
        frame_id=frame_ulid,
    )
    return RenderMint(
        record=record, provenance=provenance, canvas_digest=canvas_digest
    )


def identity_window_for_pixels(pixels: object) -> tuple[float, float]:
    """The raw/identity window over the decoded pixel extrema (Part 4.1)."""

    minimum, maximum = exact_extrema(pixels)  # type: ignore[arg-type]
    return identity_window_bounds(minimum, maximum)


def preset_window(preset: WLPreset) -> tuple[float, float]:
    """The preset's stored ``(center, width)`` as canonical floats."""

    return (float(preset.center), float(preset.width))


def render_png_for_record(
    mpr_payload: dict[str, object], pixel_blob: bytes, center: float, width: float
) -> bytes:
    """Re-derive the frozen PNG bytes for a record on the pinned runtime."""

    pixels = decode_pixel_blob(mpr_payload, pixel_blob)
    canvas = window_bytes(pixels, center, width)
    return encode_png(canvas)


__all__ = [
    "ACCEPTED_REVIEW_STATES",
    "FrameListConstructionError",
    "GeometryFrame",
    "HU_SCALAR_DOMAIN",
    "NonHUScalarDomainError",
    "RenderMint",
    "build_geometry_lists",
    "identity_window_for_pixels",
    "mint_render_variant",
    "preset_window",
    "render_png_for_record",
]
