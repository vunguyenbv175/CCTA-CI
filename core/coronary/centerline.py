"""The seeded coronary centerline derivation and its governed seams.

The slice (CORONARY_CENTERLINE_001) is exactly:

``canonical HU volume + one human seed (voxel + world direction hint)``
-> Frangi-class vesselness (pure NumPy, parameters-digest-bound)
-> binary derivation mask -> 3-D thinning -> skeleton graph trace
-> ONE ordered polyline -> arc-length parametrization into the existing
:class:`core.coordinates.CenterlineCoordinate` -> a ``CENTERLINE`` artifact
(producer ``DETERMINISTIC_ENGINE``).

Producer-kind discipline (the binding prior-art correction): the vessel mask
is carried ONLY as derivation evidence inside this derivation's parameter
document and payload digests - this module NEVER writes a ``SEGMENTATION``
artifact (``PRODUCER_MATRIX`` admits no deterministic segmentation producer).

Non-HU scalar domains are NOT re-implemented here.  The canonical chain's
non-HU safety is inherited from the existing F18 viewer gate at frame
construction (IMAGING_CORE_002_F18_SEMANTIC_GATE); this module records the
source volume's RECORDED scalar domain as evidence and mints no domain
refusal of its own.

Rule C: structure only - one polyline and its arc lengths.  There is no
diameter, area, stenosis, CAD-RADS or FFR value anywhere in this module.
"""

from __future__ import annotations

import enum
import hashlib
import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Final, cast

import numpy as np

from core.artifacts import (
    ArtifactKind,
    ClaimBinding,
    DerivedArtifact,
    DerivedArtifactStore,
    GeometryBase,
    Lineage,
    PayloadRef,
    Producer,
    QualityBlock,
    ReviewBlock,
    create_derived_artifact,
)
from core.coordinates import (
    CenterlineCoordinate,
    CoordinateTransform,
    FrameSpec,
    Matrix4,
    SpaceKind,
    VoxelCoordinate,
    apply_matrix,
    assert_round_trip,
    invert_matrix,
)
from core.coronary.response_cache import (
    ResponseCacheStore,
    cached_frangi_vesselness,
    source_content_sha256,
)
from core.coronary.thinning import orient_skeleton_path, thinning_skeleton
from core.coronary.types import (
    CENTERLINE_SCHEMA,
    CORONARY_ALGORITHM,
    CORONARY_ALGORITHM_VERSION,
    CORONARY_CLAIM_ID,
    CORONARY_CLAIM_VERSION,
    MASK_EVIDENCE_CODEC,
    VOLUME_ORDERING,
    CoronaryDiagnostics,
    CoronaryRefusalCode,
    Seed,
    VesselnessParameters,
    _refuse,
)
from core.coronary.vesselness import RESPONSE_DTYPE, frangi_vesselness, vessel_mask
from core.digest import canonical_digest, is_sha256_hex
from core.enums import (
    ActorType,
    ProducerType,
    ProvenanceEventType,
    ReleaseLane,
    ReviewState,
)
from core.errors import CanonicalValidationError, StateTransitionError
from core.events import utc_now_iso
from core.ids import new_ulid, validate_ulid
from core.provenance import (
    Actor,
    BlobRef,
    Environment,
    InputRef,
    NotApplicable,
    ProvenanceRecord,
    create_provenance,
)
from core.qc.gate import V2_VESSEL_IDS

#: Canonical reference form of a CENTERLINE artifact.
CENTERLINE_REF_PREFIX: Final[str] = "centerline_artifact_id:"

#: Payload schema of the narrow vessel-assignment record (Decision B).
VESSEL_ASSIGNMENT_SCHEMA: Final[str] = "ccta.payload.vessel_assignment.v1"
#: Closed assignment-source vocabulary (Decision B): who/what assigned the
#: label.  ``ANATOMY_VERIFIED`` is human-authoritative (HUMAN actor required);
#: the derivation never infers anatomy from geometry.
VESSEL_ASSIGNMENT_SOURCES: Final[frozenset[str]] = frozenset(
    {"USER_ASSIGNED", "FIXTURE_DERIVED", "ANATOMY_VERIFIED"}
)

_ASSIGNMENT_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "vessel_label",
        "assignment_source",
        "centerline_ref",
        "centerline_payload_sha256",
        "source_volume_ref",
        "provenance_ref",
    }
)


def validate_vessel_assignment(payload: Mapping[str, object]) -> dict[str, object]:
    """Validate the closed ``ccta.payload.vessel_assignment.v1`` record."""

    if not isinstance(payload, Mapping):
        raise _refuse(CoronaryRefusalCode.VOLUME_INVALID, "assignment must be an object")
    keys = set(payload)
    missing = sorted(_ASSIGNMENT_FIELDS - keys)
    unknown = sorted(keys - _ASSIGNMENT_FIELDS)
    if missing or unknown:
        raise _refuse(
            CoronaryRefusalCode.VOLUME_INVALID,
            f"assignment key set is not closed (missing={missing}, unknown={unknown})",
        )
    if payload["schema_version"] != VESSEL_ASSIGNMENT_SCHEMA:
        raise _refuse(CoronaryRefusalCode.VOLUME_INVALID, "assignment schema_version mismatch")
    label = payload["vessel_label"]
    if not isinstance(label, str) or label not in VESSEL_LABELS:
        raise _refuse(
            CoronaryRefusalCode.VOLUME_INVALID,
            "assignment vessel_label is outside the frozen VESSEL:* tokens",
        )
    source = payload["assignment_source"]
    if not isinstance(source, str) or source not in VESSEL_ASSIGNMENT_SOURCES:
        raise _refuse(
            CoronaryRefusalCode.VOLUME_INVALID,
            "assignment_source is outside the closed USER_ASSIGNED | "
            "FIXTURE_DERIVED | ANATOMY_VERIFIED vocabulary",
        )
    centerline_ref = payload["centerline_ref"]
    if (
        not isinstance(centerline_ref, str)
        or not centerline_ref.startswith(CENTERLINE_REF_PREFIX)
        or len(centerline_ref) != len(CENTERLINE_REF_PREFIX) + 26
    ):
        raise _refuse(
            CoronaryRefusalCode.VOLUME_INVALID,
            "assignment centerline_ref has the wrong canonical form",
        )
    if not is_sha256_hex(payload["centerline_payload_sha256"]):
        raise _refuse(
            CoronaryRefusalCode.VOLUME_INVALID,
            "assignment centerline_payload_sha256 must be 64-hex",
        )
    volume_ref = payload["source_volume_ref"]
    if not isinstance(volume_ref, str) or not volume_ref.startswith("volume_id:"):
        raise _refuse(
            CoronaryRefusalCode.VOLUME_INVALID,
            "assignment source_volume_ref has the wrong canonical form",
        )
    provenance_ref = payload["provenance_ref"]
    if not isinstance(provenance_ref, str) or not provenance_ref.startswith("provenance:"):
        raise _refuse(
            CoronaryRefusalCode.VOLUME_INVALID,
            "assignment provenance_ref has the wrong canonical form",
        )
    return dict(payload)


def vessel_assignment_provenance(
    *,
    subject_ref: str,
    centerline_artifact_id: str,
    centerline_payload_sha256: str,
    volume_id: str,
    volume_blob_sha256: str,
    vessel_label: str,
    assignment_source: str,
    actor_type: ActorType,
    actor_id: str,
    human_reviewer: str | None,
    software_version: str,
    code_commit_sha: str,
    runtime: str,
    created_by: str,
    at: str | None = None,
    provenance_id: str | None = None,
) -> ProvenanceRecord:
    """The full §44 provenance record of one vessel-label assignment.

    The assigning actor is recorded honestly: ENGINE/fixture for
    ``FIXTURE_DERIVED``, HUMAN for ``USER_ASSIGNED``/``ANATOMY_VERIFIED``.
    ``human_reviewer`` names the assigning user for human sources (required)
    and must be ``None`` for fixture sources.
    """

    if vessel_label not in VESSEL_LABELS:
        raise _refuse(
            CoronaryRefusalCode.VOLUME_INVALID,
            "assignment vessel_label is outside the frozen VESSEL:* tokens",
        )
    if assignment_source not in VESSEL_ASSIGNMENT_SOURCES:
        raise _refuse(
            CoronaryRefusalCode.VOLUME_INVALID,
            "assignment_source is outside the closed vocabulary",
        )
    if assignment_source == "FIXTURE_DERIVED":
        if actor_type is not ActorType.ENGINE:
            raise _refuse(
                CoronaryRefusalCode.VOLUME_INVALID,
                "FIXTURE_DERIVED assignments require the ENGINE actor",
            )
        if human_reviewer is not None:
            raise _refuse(
                CoronaryRefusalCode.VOLUME_INVALID,
                "FIXTURE_DERIVED assignments carry no human reviewer",
            )
        human: Any = NotApplicable(
            value=None, reason="fixture-derived label; no human assigned this vessel"
        )
    else:
        if actor_type is not ActorType.HUMAN:
            raise _refuse(
                CoronaryRefusalCode.VOLUME_INVALID,
                "USER_ASSIGNED/ANATOMY_VERIFIED assignments require the HUMAN actor",
            )
        if not isinstance(human_reviewer, str) or not human_reviewer:
            raise _refuse(
                CoronaryRefusalCode.VOLUME_INVALID,
                "human assignments require a non-empty human_reviewer",
            )
        human = human_reviewer
    if not is_sha256_hex(centerline_payload_sha256) or not is_sha256_hex(
        volume_blob_sha256
    ):
        raise _refuse(
            CoronaryRefusalCode.VOLUME_INVALID,
            "assignment digests must be 64-hex",
        )
    return create_provenance(
        subject_ref=subject_ref,
        event_type=ProvenanceEventType.DERIVED,
        actor=Actor(actor_type=actor_type, actor_id=actor_id),
        inputs=(
            InputRef(
                ref=f"centerline_artifact_id:{centerline_artifact_id}",
                kind="artifact",
                sha256_at_use=centerline_payload_sha256,
            ),
            InputRef(
                ref=f"volume_id:{volume_id}",
                kind="volume",
                sha256_at_use=volume_blob_sha256,
            ),
        ),
        preprocessing=f"vessel-assignment:{vessel_label}:{assignment_source}",
        parameters_ref=BlobRef(
            blob_ref=f"coronary-vessel-assignment:{vessel_label}:{assignment_source}",
            sha256=canonical_digest(
                {
                    "vessel_label": vessel_label,
                    "assignment_source": assignment_source,
                    "centerline_payload_sha256": centerline_payload_sha256,
                }
            ),
        ),
        human_edit=human,
        upstream_refs=NotApplicable(
            value=None,
            reason="assignment binds existing records; no new upstream derivation",
        ),
        environment=Environment(
            software_version=software_version,
            code_commit_sha=code_commit_sha,
            runtime=runtime,
            deterministic_seed=None,
        ),
        created_by=created_by,
        at=at,
        provenance_id=provenance_id,
    )
#: Deterministic actor identity of the coronary derivation.
CORONARY_ACTOR_ID: Final[str] = "ccta-coronary"
#: The only vessel-label vocabulary admitted (frozen v2 VESSEL:* tokens).
VESSEL_LABELS: Final[frozenset[str]] = frozenset(V2_VESSEL_IDS)

_PAYLOAD_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "algorithm",
        "algorithm_version",
        "parameters_sha256",
        "scales_mm",
        "threshold",
        "seed_voxel",
        "seed_direction_world",
        "seed_skeleton_voxel",
        "seed_snap_mm",
        "point_count",
        "points_world",
        "s_mm",
        "offset_mm",
        "voxel_path",
        "vesselness_response_sha256",
        "vessel_mask_sha256",
        "vessel_mask_blob_ref",
        "vessel_mask_voxel_count",
        "skeleton_voxel_count",
        "source_volume_ref",
        "source_volume_blob_sha256",
        "source_frame_spec_ref",
        "source_volume_shape",
        "voxel_spacing_mm",
        "volume_ordering",
        "out_of_plane_ref",
        "scalar_domain",
        "intensity_unit",
        "vessel_label",
        "study_instance_uid",
        "frame_of_reference_uid",
        "provenance_ref",
    }
)


@dataclass(frozen=True)
class CoronaryPath:
    """The deterministic structure of one derivation (no measurement)."""

    seed: Seed
    parameters: VesselnessParameters
    response: np.ndarray
    mask: np.ndarray
    skeleton: np.ndarray
    voxel_path: np.ndarray
    points_world: tuple[tuple[float, float, float], ...]
    s_mm: tuple[float, ...]
    coordinates: tuple[CenterlineCoordinate, ...]
    diagnostics: CoronaryDiagnostics
    #: The live skeleton voxel the human seed was projected onto (i, j, k).
    seed_skeleton_voxel: tuple[int, int, int] = (0, 0, 0)
    #: The millimetre distance of that projection (0.0 when the seed survived).
    seed_snap_mm: float = 0.0


@dataclass(frozen=True)
class CenterlineBuildResult:
    """The persisted representation of one CENTERLINE artifact."""

    artifact: DerivedArtifact
    payload: dict[str, object]
    path: CoronaryPath

    @property
    def payload_sha256(self) -> str:
        return canonical_digest(self.payload)

    @property
    def coalesced_voxel_path(self) -> np.ndarray:
        return self.path.voxel_path


def _nearest_skeleton_voxel(
    skeleton: np.ndarray,
    seed_voxel: tuple[int, int, int],
    spacing_mm: tuple[float, float, float],
) -> tuple[tuple[int, int, int], float]:
    """Nearest live skeleton voxel to the seed (deterministic tie-break).

    ``np.argwhere`` walks the skeleton in canonical index order, so the FIRST
    minimum (lowest ``(k, j, i)`` on an exact tie) wins; the distance is
    reported in millimetres through the voxel spacing so the payload records
    exactly how far the human seed was projected.
    """

    coordinates = np.argwhere(skeleton)
    if coordinates.shape[0] == 0:
        raise _refuse(CoronaryRefusalCode.NOT_ANALYZABLE, "the skeleton is empty")
    deltas = coordinates.astype(np.float64) - np.asarray(seed_voxel, dtype=np.float64)
    scaled = deltas * np.asarray(
        [spacing_mm[2], spacing_mm[1], spacing_mm[0]], dtype=np.float64
    )
    distances = np.sqrt(np.sum(scaled * scaled, axis=1))
    index = int(np.argmin(distances))
    k, j, i = (int(coordinates[index][0]), int(coordinates[index][1]), int(coordinates[index][2]))
    return (k, j, i), float(distances[index])


def _diagnostics_extra(
    *,
    voxel_path: np.ndarray,
    skeleton_seed: tuple[int, int, int],
    snap_distance: float,
    response_cache: ResponseCacheStore | None,
    cache_hit: bool,
    cache_key_hex: str,
) -> tuple[tuple[str, object], ...]:
    """Diagnostics evidence, including explicit response-cache hit/miss fields.

    The cache entries are appended only when a cache was consulted, so the
    historical no-cache diagnostics document is unchanged.
    """

    extra: list[tuple[str, object]] = [
        ("point_count", int(voxel_path.shape[0])),
        (
            "seed_skeleton_voxel",
            [int(skeleton_seed[2]), int(skeleton_seed[1]), int(skeleton_seed[0])],
        ),
        ("seed_snap_mm", float(snap_distance)),
    ]
    if response_cache is not None:
        extra.append(("response_cache_hit", bool(cache_hit)))
        extra.append(("response_cache_key", str(cache_key_hex)))
    return tuple(extra)


def _cached_or_fresh_response(
    array: np.ndarray,
    spacing: tuple[float, float, float],
    parameters: VesselnessParameters,
    response_cache: ResponseCacheStore | None,
    source_blob_sha256: str | None,
) -> tuple[np.ndarray, np.ndarray, bool, str]:
    """Serve ``(response, mask)`` from the seed-independent cache or recompute.

    The response/mask are pure functions of the source bytes, spacing and
    parameters, so a cache hit is bit-identical to a recompute by
    construction.  Lookup/computation failures (unreadable/corrupt entry,
    I/O error, unbuildable key) are a recorded miss: the derivation
    recomputes and never fails here.  A publication failure AFTER a valid
    compute is served by :func:`cached_frangi_vesselness` itself, which
    returns the already-computed arrays as a recorded miss - this layer
    never pays for a second compute in that case.  Served arrays are
    additionally shape/dtype-checked against the live input before use
    (mismatch = miss + recompute).
    """

    if response_cache is None:
        response = frangi_vesselness(array, spacing, parameters)
        return response, vessel_mask(response, parameters), False, ""
    try:
        if source_blob_sha256 is None:
            source_digest = source_content_sha256(array)
            source_kind = "array-content-sha256"
        else:
            if not is_sha256_hex(source_blob_sha256):
                raise ValueError("source_blob_sha256 is not a 64-hex sha256 digest")
            source_digest = source_blob_sha256
            source_kind = "volume-blob-sha256"
        cached = cached_frangi_vesselness(
            array,
            spacing,
            parameters,
            source_blob_sha256=source_digest,
            source_kind=source_kind,
            store=response_cache,
        )
    except Exception:
        fresh = frangi_vesselness(array, spacing, parameters)
        return fresh, vessel_mask(fresh, parameters), False, ""
    served_response, served_mask = cached.response, cached.mask
    if (
        served_response.dtype != RESPONSE_DTYPE
        or served_mask.dtype != np.dtype(bool)
        or served_response.shape != array.shape
        or served_mask.shape != array.shape
    ):
        fresh = frangi_vesselness(array, spacing, parameters)
        return fresh, vessel_mask(fresh, parameters), False, cached.key_digest
    return served_response, served_mask, cached.cache_hit, cached.key_digest


#: Maximum voxels materialized as Python scalars in one evidence-digest
#: slab.  The response/mask evidence digests hash the canonical JSON of the
#: full arrays; slab chunking keeps the transient Python-object peak bounded
#: while emitting byte-identical JSON, so evidence identity is preserved
#: exactly (no version bump, no migration rule).
_EVIDENCE_DIGEST_CHUNK_VOXELS: Final[int] = 1 << 20


def _canonical_json_bytes(value: object) -> bytes:
    """Canonical JSON bytes of a small, fully materialized value."""

    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _evidence_array_digest(dtype_label: str, array: np.ndarray) -> str:
    """Bounded-memory SHA-256 over the ``{"dtype", "values"}`` evidence document.

    Byte-identical to
    ``canonical_digest({"dtype": dtype_label, "values": array.tolist()})``:
    with ``sort_keys=True`` the document serializes as
    ``{"dtype":<label>,"values":<nested>}``, and nested-array JSON is strictly
    compositional under the fixed ``(",", ":")`` separators (no whitespace),
    so hashing slab-serialized chunks joined by ``,`` reproduces the one-shot
    bytes exactly while only one slab (at most
    ``_EVIDENCE_DIGEST_CHUNK_VOXELS`` voxels as Python scalars) is
    materialized at a time.  Non-3-D inputs take the one-shot path verbatim
    (identical output by definition); the derivation call sites always pass
    3-D arrays.
    """

    values = np.asarray(array)
    if values.ndim != 3:
        return canonical_digest({"dtype": dtype_label, "values": values.tolist()})
    per_slice = int(values.shape[1]) * int(values.shape[2])
    slab_slices = max(
        1, min(int(values.shape[0]), _EVIDENCE_DIGEST_CHUNK_VOXELS // max(1, per_slice))
    )
    hasher = hashlib.sha256()
    hasher.update(b'{"dtype":')
    hasher.update(_canonical_json_bytes(dtype_label))
    hasher.update(b',"values":[')
    first = True
    for start in range(0, int(values.shape[0]), slab_slices):
        slab_text = json.dumps(
            values[start : start + slab_slices].tolist(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        if not first:
            hasher.update(b",")
        first = False
        hasher.update(slab_text[1:-1].encode("utf-8"))
    hasher.update(b"]}")
    return hasher.hexdigest()


def derive_coronary_path(
    volume: np.ndarray,
    spacing_mm: tuple[float, float, float],
    frame: FrameSpec,
    seed: Seed,
    parameters: VesselnessParameters,
    *,
    response_cache: ResponseCacheStore | None = None,
    source_blob_sha256: str | None = None,
) -> CoronaryPath:
    """Run the full derivation and return the deterministic structure.

    ``response_cache`` is an optional seed-independent response/mask store:
    when supplied, the Frangi compute is served from CAS on a key hit and
    published on a miss (lookup failures fall back to recompute, while a
    publication failure serves the already-computed arrays as a recorded
    miss).  The hit
    evidence (``response_cache_hit`` / ``response_cache_key``) is recorded
    in the path diagnostics - a hit is never silent.  ``source_blob_sha256``
    names the canonical source blob for the key; when omitted, the key uses
    the array-content digest of ``volume``.  Both keyword arguments default
    to the historical behaviour (no cache).
    """

    if not isinstance(frame, FrameSpec):
        raise _refuse(CoronaryRefusalCode.VOLUME_INVALID, "frame must be a FrameSpec")
    array = np.asarray(volume)
    shape = (int(array.shape[0]), int(array.shape[1]), int(array.shape[2]))
    if tuple(int(value) for value in frame.dimensions) != shape:
        raise _refuse(
            CoronaryRefusalCode.VOLUME_INVALID,
            "volume shape disagrees with the FrameSpec dimensions",
        )
    seed_voxel = (seed.k, seed.j, seed.i)
    if not (
        0 <= seed.i < shape[2] and 0 <= seed.j < shape[1] and 0 <= seed.k < shape[0]
    ):
        raise _refuse(
            CoronaryRefusalCode.SEED_OUT_OF_BOUNDS,
            f"seed voxel {(seed.i, seed.j, seed.k)} is outside the volume {shape}",
        )
    response, mask, cache_hit, cache_key_hex = _cached_or_fresh_response(
        array, spacing_mm, parameters, response_cache, source_blob_sha256
    )
    mask_count = int(np.count_nonzero(mask))
    if mask_count == 0:
        raise _refuse(
            CoronaryRefusalCode.NOT_ANALYZABLE,
            "the vesselness response is empty at the frozen threshold",
        )
    if not bool(mask[seed_voxel]):
        raise _refuse(
            CoronaryRefusalCode.NOT_ANALYZABLE,
            "the seed voxel carries no vessel evidence in the derivation mask",
        )
    skeleton = np.zeros(mask.shape, dtype=bool)
    spine = thinning_skeleton(mask, seed_voxel)
    for row in range(spine.shape[0]):
        skeleton[int(spine[row][0]), int(spine[row][1]), int(spine[row][2])] = True
    if not bool(np.any(skeleton)):
        raise _refuse(
            CoronaryRefusalCode.NOT_ANALYZABLE,
            "the derivation skeleton is empty",
        )
    # The human seed is a point in the visible lumen; the derived spine is
    # free to pass beside exactly that voxel.  The seed is therefore projected
    # deterministically onto the derived spine and the projection distance is
    # RECORDED in the payload - a typed, visible behaviour, never a hidden
    # fallback (Rule 10).
    skeleton_seed = seed_voxel
    snap_distance = 0.0
    if not bool(skeleton[seed_voxel]):
        skeleton_seed, snap_distance = _nearest_skeleton_voxel(skeleton, seed_voxel, spacing_mm)
    if not bool(skeleton[skeleton_seed]):
        raise _refuse(CoronaryRefusalCode.NOT_ANALYZABLE, "the skeleton has no live voxel")
    skeleton = np.ascontiguousarray(skeleton)
    skeleton.setflags(write=False)
    world_to_voxel = invert_matrix(frame.voxel_to_world_matrix())
    hint_world = np.asarray(seed.unit_direction(), dtype=np.float64)
    # F2: the trace dot-products the hint against spine voxels in the canonical
    # (k, j, i) voxel order.  The WORLD->VOXEL affine returns components in
    # (i, j, k), so the components are re-ordered EXPLICITLY here - never
    # passed through in the wrong axis order.
    hint_i = (
        float(world_to_voxel[0][0]) * hint_world[0]
        + float(world_to_voxel[0][1]) * hint_world[1]
        + float(world_to_voxel[0][2]) * hint_world[2]
    )
    hint_j = (
        float(world_to_voxel[1][0]) * hint_world[0]
        + float(world_to_voxel[1][1]) * hint_world[1]
        + float(world_to_voxel[1][2]) * hint_world[2]
    )
    hint_k = (
        float(world_to_voxel[2][0]) * hint_world[0]
        + float(world_to_voxel[2][1]) * hint_world[1]
        + float(world_to_voxel[2][2]) * hint_world[2]
    )
    hint_voxel = (hint_k, hint_j, hint_i)
    if math.sqrt(sum(component * component for component in hint_voxel)) == 0.0:
        raise _refuse(
            CoronaryRefusalCode.SEED_DIRECTION_INVALID,
            "the world direction hint collapses to zero in voxel space",
        )
    voxel_path = orient_skeleton_path(spine, skeleton_seed, hint_voxel)
    if voxel_path.shape[0] < 2:
        raise _refuse(
            CoronaryRefusalCode.NOT_ANALYZABLE,
            "the traced skeleton component is a single voxel",
        )
    points: list[tuple[float, float, float]] = []
    matrix = frame.voxel_to_world_matrix()
    for index in range(voxel_path.shape[0]):
        k, j, i = (int(voxel_path[index][0]), int(voxel_path[index][1]), int(voxel_path[index][2]))
        point = np.asarray(
            apply_matrix(matrix, VoxelCoordinate(i=i, j=j, k=k)), dtype=np.float64
        )
        points.append((float(point[0]), float(point[1]), float(point[2])))
    arc: list[float] = [0.0]
    for index in range(1, len(points)):
        previous = points[index - 1]
        current = points[index]
        step = math.dist(previous, current)
        arc.append(arc[-1] + step)
    coordinates = tuple(
        CenterlineCoordinate(s_mm=float(value), offset_mm=0.0) for value in arc
    )
    diagnostics = CoronaryDiagnostics(
        response_sha256=_evidence_array_digest(str(response.dtype), response),
        mask_sha256=_evidence_array_digest("bool", np.asarray(mask, dtype=np.int8)),
        mask_voxel_count=mask_count,
        skeleton_voxel_count=int(np.count_nonzero(skeleton)),
        voxel_count=int(array.size),
        extra=_diagnostics_extra(
            voxel_path=voxel_path,
            skeleton_seed=skeleton_seed,
            snap_distance=snap_distance,
            response_cache=response_cache,
            cache_hit=cache_hit,
            cache_key_hex=cache_key_hex,
        ),
    )
    return CoronaryPath(
        seed=seed,
        parameters=parameters,
        response=response,
        mask=mask,
        skeleton=skeleton,
        voxel_path=np.asarray(voxel_path, dtype=np.int64),
        points_world=tuple(points),
        s_mm=tuple(arc),
        coordinates=coordinates,
        diagnostics=diagnostics,
        seed_skeleton_voxel=(int(skeleton_seed[2]), int(skeleton_seed[1]), int(skeleton_seed[0])),
        seed_snap_mm=float(snap_distance),
    )


def _require_ref(value: object, prefix: str, label: str) -> str:
    if not isinstance(value, str) or not value.startswith(prefix):
        raise _refuse(
            CoronaryRefusalCode.VOLUME_INVALID, f"{label} must be exactly {prefix}<ULID>"
        )
    try:
        validate_ulid(value.split(":", 1)[1])
    except (TypeError, ValueError):
        raise _refuse(
            CoronaryRefusalCode.VOLUME_INVALID, f"{label} must be exactly {prefix}<ULID>"
        ) from None
    return value


def _normalize_volume_ref(volume_id: str) -> str:
    if not isinstance(volume_id, str):
        raise _refuse(CoronaryRefusalCode.VOLUME_INVALID, "source Volume id must be a ULID")
    if volume_id.startswith("volume_id:"):
        return _require_ref(volume_id, "volume_id:", "source_volume_ref")
    try:
        validate_ulid(volume_id)
    except (TypeError, ValueError):
        raise _refuse(
            CoronaryRefusalCode.VOLUME_INVALID, "source Volume id must be a ULID"
        ) from None
    return f"volume_id:{volume_id}"


def _normalize_transform_ref(transform: CoordinateTransform, value: str | None) -> str:
    ref = transform.transform_id if value is None else value
    if not isinstance(ref, str):
        raise _refuse(
            CoronaryRefusalCode.VOLUME_INVALID,
            "transform_artifact_ref is required as coordinate_transform:<ULID>",
        )
    if not ref.startswith("coordinate_transform:"):
        try:
            validate_ulid(ref)
        except (TypeError, ValueError):
            raise _refuse(
                CoronaryRefusalCode.VOLUME_INVALID,
                "transform_artifact_ref is required as coordinate_transform:<ULID>",
            ) from None
        ref = f"coordinate_transform:{ref}"
    return _require_ref(ref, "coordinate_transform:", "transform_artifact_ref")


def _normalize_provenance_ref(value: str) -> tuple[str, str]:
    if not isinstance(value, str):
        raise _refuse(
            CoronaryRefusalCode.VOLUME_INVALID,
            "provenance_ref must be provenance:<ULID>",
        )
    if value.startswith("provenance:"):
        ref = _require_ref(value, "provenance:", "provenance_ref")
        return ref, ref.split(":", 1)[1]
    try:
        validate_ulid(value)
    except (TypeError, ValueError):
        raise _refuse(
            CoronaryRefusalCode.VOLUME_INVALID, "provenance_ref must be provenance:<ULID>"
        ) from None
    return f"provenance:{value}", value


def frame_spec_ref(frame: FrameSpec) -> str:
    """Content reference of the exact source FrameSpec JSON (MPR precedent)."""

    return f"frame_spec:{canonical_digest(frame)}"


def _normalize_transform_for_frame(
    transform: CoordinateTransform, frame: FrameSpec
) -> CoordinateTransform:
    if transform.from_space is SpaceKind.WORLD and transform.to_space is SpaceKind.VOXEL:
        world_to_voxel = transform
    elif transform.from_space is SpaceKind.VOXEL and transform.to_space is SpaceKind.WORLD:
        world_to_voxel = transform.inverse
    else:
        raise _refuse(
            CoronaryRefusalCode.VOLUME_INVALID, "a WORLD<->VOXEL transform is required"
        )
    expected = invert_matrix(frame.voxel_to_world_matrix())
    if world_to_voxel.matrix != expected:
        raise _refuse(
            CoronaryRefusalCode.VOLUME_INVALID,
            "the persisted transform does not describe the source FrameSpec",
        )
    return world_to_voxel


def validate_centerline_payload(payload: Mapping[str, object]) -> dict[str, object]:
    """Validate the closed ``ccta.payload.coronary_centerline.v1`` payload."""

    if not isinstance(payload, Mapping):
        raise _refuse(CoronaryRefusalCode.VOLUME_INVALID, "payload must be an object")
    keys = set(payload)
    missing = sorted(_PAYLOAD_FIELDS - keys)
    unknown = sorted(keys - _PAYLOAD_FIELDS)
    if missing or unknown:
        raise _refuse(
            CoronaryRefusalCode.VOLUME_INVALID,
            f"payload key set is not closed (missing={missing}, unknown={unknown})",
        )
    if payload["schema_version"] != CENTERLINE_SCHEMA:
        raise _refuse(CoronaryRefusalCode.VOLUME_INVALID, "payload schema_version mismatch")
    if payload["algorithm"] != CORONARY_ALGORITHM:
        raise _refuse(CoronaryRefusalCode.VOLUME_INVALID, "payload algorithm mismatch")
    if payload["algorithm_version"] != CORONARY_ALGORITHM_VERSION:
        raise _refuse(CoronaryRefusalCode.VOLUME_INVALID, "payload algorithm_version mismatch")
    if not is_sha256_hex(payload["parameters_sha256"]):
        raise _refuse(CoronaryRefusalCode.VOLUME_INVALID, "parameters_sha256 must be 64-hex")
    points = payload["points_world"]
    arcs = payload["s_mm"]
    offsets = payload["offset_mm"]
    voxels = payload["voxel_path"]
    if not isinstance(points, list) or not points:
        raise _refuse(CoronaryRefusalCode.VOLUME_INVALID, "points_world must be a non-empty list")
    if not isinstance(arcs, list) or len(arcs) != len(points):
        raise _refuse(CoronaryRefusalCode.VOLUME_INVALID, "s_mm length mismatch")
    if not isinstance(offsets, list) or len(offsets) != len(points):
        raise _refuse(CoronaryRefusalCode.VOLUME_INVALID, "offset_mm length mismatch")
    if not isinstance(voxels, list) or len(voxels) != len(points):
        raise _refuse(CoronaryRefusalCode.VOLUME_INVALID, "voxel_path length mismatch")
    point_count = payload["point_count"]
    if (
        isinstance(point_count, bool)
        or not isinstance(point_count, int)
        or point_count != len(points)
    ):
        raise _refuse(CoronaryRefusalCode.VOLUME_INVALID, "point_count mismatch")
    previous = -1.0
    for index, value in enumerate(arcs):
        number = _finite(value, f"s_mm[{index}]")
        if number < 0.0 or number < previous:
            raise _refuse(CoronaryRefusalCode.VOLUME_INVALID, "s_mm must be monotonic")
        previous = number
    if _finite(arcs[0], "s_mm[0]") != 0.0:
        raise _refuse(CoronaryRefusalCode.VOLUME_INVALID, "s_mm must start at 0")
    for index, value in enumerate(offsets):
        if _finite(value, f"offset_mm[{index}]") != 0.0:
            raise _refuse(
                CoronaryRefusalCode.VOLUME_INVALID, "offset_mm is zero on the single path"
            )
    for index, point in enumerate(points):
        if not isinstance(point, list) or len(point) != 3:
            raise _refuse(CoronaryRefusalCode.VOLUME_INVALID, f"points_world[{index}] shape")
        for axis, value in enumerate(point):
            _finite(value, f"points_world[{index}][{axis}]")
    for index, voxel in enumerate(voxels):
        if not isinstance(voxel, list) or len(voxel) != 3:
            raise _refuse(CoronaryRefusalCode.VOLUME_INVALID, f"voxel_path[{index}] shape")
        for axis, value in enumerate(voxel):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise _refuse(
                    CoronaryRefusalCode.VOLUME_INVALID,
                    f"voxel_path[{index}][{axis}] must be a non-negative integer",
                )
    seed_voxel = payload["seed_voxel"]
    if not isinstance(seed_voxel, list) or len(seed_voxel) != 3:
        raise _refuse(CoronaryRefusalCode.VOLUME_INVALID, "seed_voxel shape")
    for axis, value in enumerate(seed_voxel):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise _refuse(CoronaryRefusalCode.VOLUME_INVALID, f"seed_voxel[{axis}]")
    direction = payload["seed_direction_world"]
    if not isinstance(direction, list) or len(direction) != 3:
        raise _refuse(CoronaryRefusalCode.VOLUME_INVALID, "seed_direction_world shape")
    for axis, value in enumerate(direction):
        _finite(value, f"seed_direction_world[{axis}]")
    skeleton_seed = payload["seed_skeleton_voxel"]
    if not isinstance(skeleton_seed, list) or len(skeleton_seed) != 3:
        raise _refuse(CoronaryRefusalCode.VOLUME_INVALID, "seed_skeleton_voxel shape")
    for axis, value in enumerate(skeleton_seed):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise _refuse(CoronaryRefusalCode.VOLUME_INVALID, f"seed_skeleton_voxel[{axis}]")
    if _finite(payload["seed_snap_mm"], "seed_snap_mm") < 0.0:
        raise _refuse(CoronaryRefusalCode.VOLUME_INVALID, "seed_snap_mm must be >= 0")
    scales = payload["scales_mm"]
    if not isinstance(scales, list) or not scales:
        raise _refuse(CoronaryRefusalCode.VOLUME_INVALID, "scales_mm must be a non-empty list")
    for index, value in enumerate(scales):
        if _finite(value, f"scales_mm[{index}]") <= 0.0:
            raise _refuse(CoronaryRefusalCode.VOLUME_INVALID, "scales_mm must be positive")
    if _finite(payload["threshold"], "threshold") <= 0.0:
        raise _refuse(CoronaryRefusalCode.VOLUME_INVALID, "threshold must be positive")
    _require_ref(payload["source_volume_ref"], "volume_id:", "source_volume_ref")
    if not is_sha256_hex(payload["source_volume_blob_sha256"]):
        raise _refuse(CoronaryRefusalCode.VOLUME_INVALID, "source_volume_blob_sha256")
    frame_ref = payload["source_frame_spec_ref"]
    if (
        not isinstance(frame_ref, str)
        or not frame_ref.startswith("frame_spec:")
        or not is_sha256_hex(frame_ref.split(":", 1)[1])
    ):
        raise _refuse(
            CoronaryRefusalCode.VOLUME_INVALID,
            "source_frame_spec_ref must be exactly frame_spec:<64-hex sha256>",
        )
    if not is_sha256_hex(payload["vesselness_response_sha256"]):
        raise _refuse(CoronaryRefusalCode.VOLUME_INVALID, "vesselness_response_sha256")
    if not is_sha256_hex(payload["vessel_mask_sha256"]):
        raise _refuse(CoronaryRefusalCode.VOLUME_INVALID, "vessel_mask_sha256")
    mask_blob = payload["vessel_mask_blob_ref"]
    if not isinstance(mask_blob, Mapping) or set(mask_blob) != {
        "sha256",
        "codec",
        "byte_size",
    }:
        raise _refuse(CoronaryRefusalCode.VOLUME_INVALID, "vessel_mask_blob_ref shape")
    if not is_sha256_hex(mask_blob["sha256"]):
        raise _refuse(CoronaryRefusalCode.VOLUME_INVALID, "vessel_mask_blob_ref.sha256")
    if mask_blob["codec"] != MASK_EVIDENCE_CODEC:
        raise _refuse(CoronaryRefusalCode.VOLUME_INVALID, "vessel_mask_blob_ref.codec")
    for label in (
        "vessel_mask_voxel_count",
        "skeleton_voxel_count",
    ):
        value = payload[label]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise _refuse(CoronaryRefusalCode.VOLUME_INVALID, f"{label} must be non-negative")
    shape = payload["source_volume_shape"]
    if not isinstance(shape, list) or len(shape) != 3:
        raise _refuse(CoronaryRefusalCode.VOLUME_INVALID, "source_volume_shape must be 3-D")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in shape
    ):
        raise _refuse(
            CoronaryRefusalCode.VOLUME_INVALID,
            "source_volume_shape must contain positive integers",
        )
    byte_size = mask_blob["byte_size"]
    if isinstance(byte_size, bool) or not isinstance(byte_size, int):
        raise _refuse(
            CoronaryRefusalCode.VOLUME_INVALID, "vessel_mask_blob_ref.byte_size must be an integer"
        )
    if byte_size != int(shape[0]) * int(shape[1]) * int(shape[2]):
        raise _refuse(
            CoronaryRefusalCode.VOLUME_INVALID,
            "vessel_mask_blob_ref.byte_size must equal the one-byte-per-voxel mask size",
        )
    spacing = payload["voxel_spacing_mm"]
    if not isinstance(spacing, list) or len(spacing) != 3:
        raise _refuse(CoronaryRefusalCode.VOLUME_INVALID, "voxel_spacing_mm must be 3-D")
    for axis, value in enumerate(spacing):
        if _finite(value, f"voxel_spacing_mm[{axis}]") <= 0.0:
            raise _refuse(CoronaryRefusalCode.VOLUME_INVALID, "voxel spacing must be positive")
    if payload["volume_ordering"] != VOLUME_ORDERING:
        raise _refuse(CoronaryRefusalCode.VOLUME_INVALID, "volume_ordering is fixed")
    out_of_plane = payload["out_of_plane_ref"]
    if not isinstance(out_of_plane, Mapping) or set(out_of_plane) != {
        "transform_artifact_ref",
        "transform_version",
        "direction",
    }:
        raise _refuse(CoronaryRefusalCode.VOLUME_INVALID, "out_of_plane_ref shape")
    _require_ref(
        out_of_plane["transform_artifact_ref"],
        "coordinate_transform:",
        "transform_artifact_ref",
    )
    version = out_of_plane["transform_version"]
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise _refuse(CoronaryRefusalCode.VOLUME_INVALID, "transform_version must be positive")
    if out_of_plane["direction"] != "WORLD_TO_VOXEL":
        raise _refuse(CoronaryRefusalCode.VOLUME_INVALID, "out_of_plane_ref direction is fixed")
    domain = payload["scalar_domain"]
    if not isinstance(domain, str) or domain not in {"STORED", "MODALITY_TRANSFORMED", "HU"}:
        raise _refuse(CoronaryRefusalCode.VOLUME_INVALID, "scalar_domain is outside the vocabulary")
    if payload["intensity_unit"] != domain:
        raise _refuse(CoronaryRefusalCode.VOLUME_INVALID, "intensity_unit must equal scalar_domain")
    label_value = payload["vessel_label"]
    if label_value is not None and label_value not in VESSEL_LABELS:
        raise _refuse(
            CoronaryRefusalCode.VOLUME_INVALID, "vessel_label is outside the frozen VESSEL:* tokens"
        )
    for label in ("study_instance_uid", "frame_of_reference_uid"):
        value = payload[label]
        if not isinstance(value, str) or not value:
            raise _refuse(CoronaryRefusalCode.VOLUME_INVALID, f"{label} is required")
    _require_ref(payload["provenance_ref"], "provenance:", "provenance_ref")
    return dict(payload)


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _refuse(CoronaryRefusalCode.VOLUME_INVALID, f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise _refuse(CoronaryRefusalCode.VOLUME_INVALID, f"{label} must be a finite number")
    return number


# ---------------------------------------------------------------------------
# CENTERLINE artifact construction (DETERMINISTIC_ENGINE producer)
# ---------------------------------------------------------------------------


def build_centerline(
    source_volume: Any,
    transform: CoordinateTransform,
    frame: FrameSpec,
    *,
    loader: Any,
    seed: Seed,
    parameters: VesselnessParameters,
    provenance_ref: str,
    study_instance_uid: str,
    frame_of_reference_uid: str,
    scalar_domain: str,
    source_array: np.ndarray | None = None,
    transform_artifact_ref: str | None = None,
    response_cache: ResponseCacheStore | None = None,
    spherical_label: str | None = None,
    vessel_label: str | None = None,
    claim_version: str = CORONARY_CLAIM_VERSION,
    release_lane: ReleaseLane | None = None,
    created_by: str = "ccta-coronary",
    artifact_id: str | None = None,
) -> CenterlineBuildResult:
    """Derive and freeze ONE CENTERLINE artifact plus its evidence payload.

    ``source_volume`` is the canonical :class:`core.volume.Volume`; the caller
    supplies the decoded ``source_array`` (or the canonical blob is decoded
    through ``loader``).  The mask bytes are carried ONLY as the derivation's
    evidence digests inside the payload and the provenance parameters ref -
    this function never writes a SEGMENTATION artifact.

    ``scalar_domain`` is the RECORDED scalar domain of the source volume's
    envelope; it is carried as evidence (never silently defaulted).  Non-HU
    safety is inherited from the existing F18 viewer gate - no domain refusal
    is re-implemented here.

    ``response_cache`` is an optional seed-independent response/mask store,
    keyed here on the canonical source blob SHA-256; see
    :func:`derive_coronary_path`.
    """

    from core.volume import Volume

    if not isinstance(source_volume, Volume):
        raise _refuse(
            CoronaryRefusalCode.VOLUME_INVALID, "source_volume must be a canonical Volume"
        )
    if not isinstance(transform, CoordinateTransform):
        raise _refuse(CoronaryRefusalCode.VOLUME_INVALID, "transform must be a persisted transform")
    if not isinstance(frame, FrameSpec):
        raise _refuse(CoronaryRefusalCode.VOLUME_INVALID, "frame must be a FrameSpec")
    if not isinstance(seed, Seed):
        raise _refuse(CoronaryRefusalCode.VOLUME_INVALID, "seed must be a Seed")
    if not isinstance(parameters, VesselnessParameters):
        raise _refuse(
            CoronaryRefusalCode.PARAMETERS_INVALID, "parameters must be a VesselnessParameters"
        )
    if not isinstance(claim_version, str) or not claim_version:
        raise _refuse(CoronaryRefusalCode.VOLUME_INVALID, "claim_version is required")
    if not isinstance(created_by, str) or not created_by:
        raise _refuse(CoronaryRefusalCode.VOLUME_INVALID, "created_by is required")
    if scalar_domain not in {"STORED", "MODALITY_TRANSFORMED", "HU"}:
        raise _refuse(CoronaryRefusalCode.VOLUME_INVALID, "scalar_domain is outside the vocabulary")
    if vessel_label is not None and vessel_label not in VESSEL_LABELS:
        raise _refuse(
            CoronaryRefusalCode.VOLUME_INVALID, "vessel_label is outside the frozen VESSEL:* tokens"
        )
    del spherical_label  # reserved spelling of the same frozen token set
    artifact_ulid = artifact_id or new_ulid()
    try:
        validate_ulid(artifact_ulid)
    except (TypeError, ValueError):
        raise _refuse(CoronaryRefusalCode.VOLUME_INVALID, "artifact_id must be a ULID") from None
    if release_lane is not None and not isinstance(release_lane, ReleaseLane):
        raise _refuse(CoronaryRefusalCode.VOLUME_INVALID, "release_lane must be a ReleaseLane")
    if not isinstance(source_volume.study_ref, str) or not source_volume.study_ref:
        raise _refuse(CoronaryRefusalCode.VOLUME_INVALID, "source_volume.study_ref is required")
    if not isinstance(study_instance_uid, str) or not study_instance_uid:
        raise _refuse(CoronaryRefusalCode.VOLUME_INVALID, "study_instance_uid is required")
    if frame.frame_of_reference_uid != source_volume.coordinate_frame.frame_of_reference_uid:
        raise _refuse(
            CoronaryRefusalCode.VOLUME_INVALID,
            "frame and source volume FrameOfReferenceUID disagree",
        )
    if frame_of_reference_uid != frame.frame_of_reference_uid:
        raise _refuse(
            CoronaryRefusalCode.VOLUME_INVALID, "frame_of_reference_uid disagrees with the frame"
        )
    _normalize_transform_for_frame(transform, frame)
    try:
        assert_round_trip(transform, transform.inverse)
    except CanonicalValidationError:
        raise _refuse(
            CoronaryRefusalCode.VOLUME_INVALID, "the persisted transform fails its round-trip"
        ) from None

    expected_frame_ref = frame_spec_ref(frame)
    if loader is None:
        raise _refuse(CoronaryRefusalCode.VOLUME_INVALID, "loader is required")
    envelope = loader.read_envelope(source_volume.blob_ref.sha256)
    if envelope.get("blob_sha256") != source_volume.blob_ref.sha256:
        raise _refuse(CoronaryRefusalCode.VOLUME_INVALID, "source blob ref mismatch")
    canonical_source = loader.read_array(source_volume.blob_ref.sha256, envelope=envelope)
    if source_array is None:
        source_array = canonical_source
    source = np.asarray(source_array)
    expected_shape = tuple(int(value) for value in frame.dimensions)
    if tuple(int(value) for value in source.shape) != expected_shape:
        raise _refuse(CoronaryRefusalCode.VOLUME_INVALID, "source array shape mismatch")
    if (
        source.dtype != canonical_source.dtype
        or source.tobytes(order="C") != canonical_source.tobytes(order="C")
    ):
        raise _refuse(CoronaryRefusalCode.VOLUME_INVALID, "source array does not match the blob")

    spacing = (
        float(frame.spacing_mm[0]),
        float(frame.spacing_mm[1]),
        float(frame.spacing_mm[2]),
    )
    path = derive_coronary_path(
        source,
        spacing,
        frame,
        seed,
        parameters,
        response_cache=response_cache,
        source_blob_sha256=source_volume.blob_ref.sha256,
    )
    # F7: the derivation mask is retained as a resolvable, immutable,
    # content-addressed evidence blob (never a SEGMENTATION artifact and never
    # only an in-memory array).  ``loader`` is mandatory, so the bytes always
    # land in the same governed blob store as the canonical volume.
    mask_bytes = np.ascontiguousarray(path.mask, dtype=np.uint8).tobytes(order="C")
    mask_blob = loader.write_blob(mask_bytes)
    transform_ref = _normalize_transform_ref(transform, transform_artifact_ref)
    provenance_payload_ref, provenance_id = _normalize_provenance_ref(provenance_ref)
    source_ref = _normalize_volume_ref(source_volume.volume_id)
    parameters_document = parameters.to_document()
    payload: dict[str, object] = {
        "schema_version": CENTERLINE_SCHEMA,
        "algorithm": CORONARY_ALGORITHM,
        "algorithm_version": CORONARY_ALGORITHM_VERSION,
        "parameters_sha256": parameters.sha256,
        "scales_mm": parameters_document["scales_mm"],
        "threshold": parameters_document["threshold"],
        "seed_voxel": [int(seed.i), int(seed.j), int(seed.k)],
        "seed_direction_world": [float(value) for value in seed.unit_direction()],
        "seed_skeleton_voxel": [
            int(path.seed_skeleton_voxel[0]),
            int(path.seed_skeleton_voxel[1]),
            int(path.seed_skeleton_voxel[2]),
        ],
        "seed_snap_mm": float(path.seed_snap_mm),
        "point_count": int(path.voxel_path.shape[0]),
        "points_world": [[float(value) for value in point] for point in path.points_world],
        "s_mm": [float(value) for value in path.s_mm],
        "offset_mm": [0.0 for _ in path.s_mm],
        "voxel_path": [
            [
                int(path.voxel_path[index][2]),
                int(path.voxel_path[index][1]),
                int(path.voxel_path[index][0]),
            ]
            for index in range(path.voxel_path.shape[0])
        ],
        "vesselness_response_sha256": path.diagnostics.response_sha256,
        "vessel_mask_sha256": path.diagnostics.mask_sha256,
        "vessel_mask_blob_ref": {
            "sha256": mask_blob.sha256,
            "codec": MASK_EVIDENCE_CODEC,
            "byte_size": int(mask_blob.byte_size),
        },
        "vessel_mask_voxel_count": int(path.diagnostics.mask_voxel_count),
        "skeleton_voxel_count": int(path.diagnostics.skeleton_voxel_count),
        "source_volume_ref": source_ref,
        "source_volume_blob_sha256": source_volume.blob_ref.sha256,
        "source_frame_spec_ref": expected_frame_ref,
        "source_volume_shape": [int(value) for value in frame.dimensions],
        "voxel_spacing_mm": [spacing[0], spacing[1], spacing[2]],
        "volume_ordering": VOLUME_ORDERING,
        "out_of_plane_ref": {
            "transform_artifact_ref": transform_ref,
            "transform_version": int(transform.version),
            "direction": "WORLD_TO_VOXEL",
        },
        "scalar_domain": scalar_domain,
        "intensity_unit": scalar_domain,
        "vessel_label": vessel_label,
        "study_instance_uid": study_instance_uid,
        "frame_of_reference_uid": frame_of_reference_uid,
        "provenance_ref": provenance_payload_ref,
    }
    validated = validate_centerline_payload(payload)
    artifact = create_derived_artifact(
        study_ref=source_volume.study_ref,
        kind=ArtifactKind.CENTERLINE,
        payload=PayloadRef(
            payload_schema=CENTERLINE_SCHEMA,
            blob_sha256=canonical_digest(validated),
            codec="json",
            inline_summary=(
                ("point_count", str(path.voxel_path.shape[0])),
                ("vessel_label", str(vessel_label)),
            ),
        ),
        lineage=Lineage(
            input_volume_refs=(source_volume.volume_id,),
            input_artifact_refs=(transform_ref.split(":", 1)[1],),
            geometry_base=GeometryBase(
                geometry_id=transform_ref.split(":", 1)[1],
                geometry_version=int(transform.version),
            ),
        ),
        claim_binding=ClaimBinding(claim_id=CORONARY_CLAIM_ID, claim_version=claim_version),
        producer=Producer(
            producer_type=ProducerType.DETERMINISTIC_ENGINE,
            algorithm=CORONARY_ALGORITHM,
            algorithm_version=CORONARY_ALGORITHM_VERSION,
            parameters_blob_sha256=cast(str, parameters.sha256),
        ),
        review=ReviewBlock(review_state=ReviewState.PREDICTED),
        quality=QualityBlock(),
        release_lane=release_lane or source_volume.release_lane,
        provenance_ref=provenance_id,
        created_by=created_by,
        created_at=utc_now_iso(),
        artifact_id=artifact_ulid,
    )
    return CenterlineBuildResult(artifact=artifact, payload=validated, path=path)


def centerline_provenance(
    *,
    subject_ref: str,
    volume: Any,
    seed: Seed,
    parameters: VesselnessParameters,
    mask_sha256: str,
    mask_blob_sha256: str,
    response_sha256: str,
    transform_artifact_ref: str,
    transform_digest: str,
    volume_provenance_ref: str,
    transform_provenance_ref: str,
    software_version: str,
    code_commit_sha: str,
    runtime: str,
    created_by: str,
    at: str | None = None,
    provenance_id: str | None = None,
) -> ProvenanceRecord:
    """The full §44 nine-item provenance record of one centerline derivation.

    F7 closure: the typed human seed is an explicit derivation INPUT
    (``InputRef`` kind ``seed``, digest-bound), and the derivation mask bytes
    are retained as a resolvable immutable evidence blob whose content address
    binds the ``parameters_ref`` (no SEGMENTATION artifact is written and no
    SEGMENTATION provenance is minted).  ``volume`` is the canonical
    :class:`core.volume.Volume`; ``subject_ref`` is the CENTERLINE artifact
    reference the record is about.
    """

    if not isinstance(subject_ref, str) or not subject_ref:
        raise _refuse(CoronaryRefusalCode.VOLUME_INVALID, "subject_ref is required")
    if not isinstance(seed, Seed):
        raise _refuse(CoronaryRefusalCode.VOLUME_INVALID, "seed must be a Seed")
    seed_sha256 = canonical_digest(seed.to_document())
    if not isinstance(mask_sha256, str) or not is_sha256_hex(mask_sha256):
        raise _refuse(CoronaryRefusalCode.VOLUME_INVALID, "mask_sha256 must be 64-hex")
    if not isinstance(mask_blob_sha256, str) or not is_sha256_hex(mask_blob_sha256):
        raise _refuse(
            CoronaryRefusalCode.VOLUME_INVALID, "mask_blob_sha256 must be 64-hex"
        )
    if not isinstance(response_sha256, str) or not is_sha256_hex(response_sha256):
        raise _refuse(CoronaryRefusalCode.VOLUME_INVALID, "response_sha256 must be 64-hex")
    parameters_ref = BlobRef(
        blob_ref=(
            "coronary-centerline-parameters:"
            f"{parameters.sha256}:seed-{seed_sha256}:mask-{mask_sha256}:"
            f"mask-blob-{mask_blob_sha256}:response-{response_sha256}"
        ),
        sha256=canonical_digest(
            {
                "parameters_sha256": parameters.sha256,
                "seed_sha256": seed_sha256,
                "vessel_mask_sha256": mask_sha256,
                "vessel_mask_blob_sha256": mask_blob_sha256,
                "vesselness_response_sha256": response_sha256,
            }
        ),
    )
    return create_provenance(
        subject_ref=subject_ref,
        event_type=ProvenanceEventType.DERIVED,
        actor=Actor(actor_type=ActorType.ENGINE, actor_id=CORONARY_ACTOR_ID),
        inputs=(
            InputRef(
                ref=f"volume_id:{volume.volume_id}",
                kind="volume",
                sha256_at_use=volume.blob_ref.sha256,
            ),
            InputRef(
                ref=transform_artifact_ref,
                kind="artifact",
                sha256_at_use=transform_digest,
            ),
            InputRef(
                ref=f"seed:{seed_sha256}",
                kind="seed",
                sha256_at_use=seed_sha256,
            ),
        ),
        preprocessing=(
            "coronary-derivation:separable-gaussian-frangi-vesselness+"
            "exact-edt-ridge-centered-geodesic-spine+ridge-weighted-direction-hint-trace"
        ),
        parameters_ref=parameters_ref,
        human_edit=NotApplicable(
            value=None,
            reason=(
                "deterministic derivation; the human seed is bound as a typed "
                "derivation input and recorded in the payload"
            ),
        ),
        upstream_refs=(volume_provenance_ref, transform_provenance_ref),
        environment=Environment(
            software_version=software_version,
            code_commit_sha=code_commit_sha,
            runtime=runtime,
            deterministic_seed=None,
        ),
        created_by=created_by,
        at=at,
        provenance_id=provenance_id,
    )


# ---------------------------------------------------------------------------
# Human accept / edit (re-seed) / reject seam (Rule F)
# ---------------------------------------------------------------------------


class CoronaryDecision(enum.StrEnum):
    """The closed human decision vocabulary of the single-path review seam."""

    ACCEPT = "ACCEPT"
    EDIT = "EDIT"
    REJECT = "REJECT"


@dataclass(frozen=True)
class CoronaryDecisionOutcome:
    """The append-only outcome of one human decision (never an in-place edit)."""

    decision: str
    centerline_id: str
    review_state: str
    artifact_status: str
    new_centerline_id: str | None = None
    correction_artifact_id: str | None = None
    superseded_artifact_id: str | None = None
    staled_artifact_ids: tuple[str, ...] = ()


def _require_human(actor_role: object, reviewed_by: str | None) -> ActorType:
    """The single-path review is a HUMAN-only seam (Rule F)."""

    if actor_role is not ActorType.HUMAN:
        raise StateTransitionError(
            "the centerline review seam is human-only; "
            f"got actor_role={actor_role!r}"
        )
    if reviewed_by is not None and (not isinstance(reviewed_by, str) or not reviewed_by):
        raise StateTransitionError("reviewed_by must be a non-empty string or None")
    return ActorType.HUMAN


def _stale_dependents_of(store: DerivedArtifactStore, centerline_id: str) -> tuple[str, ...]:
    return tuple(
        dependent_id
        for dependent_id in store.invalidation_dag.stale_dependents(centerline_id)
        if dependent_id in store
    )


def apply_coronary_decision(
    store: DerivedArtifactStore,
    *,
    centerline_id: str,
    decision: CoronaryDecision,
    actor_role: ActorType,
    reviewed_by: str | None = None,
    evidence_ref: str | None = None,
    at: str | None = None,
    correction_payload: Mapping[str, object] | None = None,
    rebuild: Callable[[], CenterlineBuildResult] | None = None,
    correction_provenance_ref: str | None = None,
    correction_study_ref: str | None = None,
    correction_claim_version: str = CORONARY_CLAIM_VERSION,
    correction_release_lane: ReleaseLane | None = None,
    correction_created_by: str = CORONARY_ACTOR_ID,
    correction_artifact_id: str | None = None,
    new_geometry_version: int | None = None,
) -> CoronaryDecisionOutcome:
    """Apply ONE human decision to the single centerline path.

    * ``ACCEPT`` - an append-only REVIEW event moves the artifact to the
      human ACCEPTED state; the sealed payload is never touched.
    * ``REJECT`` - the artifact moves to REJECTED and is INVALIDATED, which
      fans STALE out to every dependent bound to this centerline geometry
      (its straightened views); bytes are retained and still queryable.
    * ``EDIT`` - a re-seed recompute: a HUMAN ``CORRECTION`` artifact records
      the new seed, ``rebuild`` derives a NEW centerline (never an in-place
      edit), the predecessor is SUPERSEDED and the geometry version advances
      so the predecessor's dependents go STALE.

    The seam refuses anything but a HUMAN actor (Rule F: human-only final)
    and a CENTERLINE subject; the closed decision vocabulary is enforced by
    :class:`CoronaryDecision`.
    """

    decision = CoronaryDecision(decision)
    _require_human(actor_role, reviewed_by)
    try:
        artifact = store.get(centerline_id)
    except KeyError:
        raise StateTransitionError(f"unknown centerline artifact {centerline_id!r}") from None
    if artifact.kind is not ArtifactKind.CENTERLINE:
        raise StateTransitionError(
            f"artifact {centerline_id!r} is kind {artifact.kind.value}, not CENTERLINE"
        )

    if decision is CoronaryDecision.ACCEPT:
        updated = store.apply_review_transition(
            centerline_id,
            ReviewState.ACCEPTED,
            actor_role=ActorType.HUMAN,
            reviewed_by=reviewed_by,
            evidence_ref=evidence_ref,
            at=at,
        )
        return CoronaryDecisionOutcome(
            decision=decision.value,
            centerline_id=centerline_id,
            review_state=updated.review.review_state.value,
            artifact_status=updated.status.value,
        )

    if decision is CoronaryDecision.REJECT:
        updated = store.apply_review_transition(
            centerline_id,
            ReviewState.REJECTED,
            actor_role=ActorType.HUMAN,
            reviewed_by=reviewed_by,
            evidence_ref=evidence_ref,
            at=at,
        )
        staled = store.invalidate(
            centerline_id,
            reason=evidence_ref or "centerline rejected by human review",
        )
        return CoronaryDecisionOutcome(
            decision=decision.value,
            centerline_id=centerline_id,
            review_state=updated.review.review_state.value,
            artifact_status=store.get(centerline_id).status.value,
            staled_artifact_ids=tuple(staled),
        )

    # EDIT (re-seed recompute) ------------------------------------------------
    if rebuild is None:
        raise StateTransitionError("an EDIT decision requires a rebuild callable")
    if correction_payload is None:
        raise StateTransitionError("an EDIT decision requires the correction payload")
    if correction_provenance_ref is None or correction_study_ref is None:
        raise StateTransitionError(
            "an EDIT decision requires the correction provenance ref and study ref"
        )
    response = rebuild()
    correction = create_derived_artifact(
        study_ref=correction_study_ref,
        kind=ArtifactKind.CORRECTION,
        payload=PayloadRef(
            payload_schema="ccta.payload.coronary_correction.v1",
            blob_sha256=canonical_digest(dict(correction_payload)),
            codec="json",
            inline_summary=(("decision", decision.value),),
        ),
        lineage=Lineage(
            input_artifact_refs=(centerline_id,),
            input_volume_refs=artifact.lineage.input_volume_refs,
            geometry_base=artifact.lineage.geometry_base,
        ),
        claim_binding=ClaimBinding(
            claim_id=CORONARY_CLAIM_ID, claim_version=correction_claim_version
        ),
        producer=Producer(
            producer_type=ProducerType.HUMAN,
            algorithm="human-centerline-reseed",
            algorithm_version=CORONARY_ALGORITHM_VERSION,
        ),
        review=ReviewBlock(
            review_state=ReviewState.PREDICTED,
            supersedes_ref=f"{CENTERLINE_REF_PREFIX}{centerline_id}",
            reviewed_by=reviewed_by,
            reviewed_at=at or utc_now_iso(),
        ),
        quality=QualityBlock(),
        release_lane=correction_release_lane or artifact.release_lane,
        provenance_ref=correction_provenance_ref,
        created_by=correction_created_by,
        created_at=at or utc_now_iso(),
        artifact_id=correction_artifact_id,
    )
    store.add(correction)
    store.add(response.artifact)
    version = new_geometry_version
    if version is None:
        version = store.invalidation_dag.geometry_version(centerline_id) + 1
    if version < 1:
        version = 1
    staled = store.register_new_geometry_version(centerline_id, version)
    successor = store.mark_superseded(
        centerline_id, superseded_by=response.artifact.artifact_id
    )
    return CoronaryDecisionOutcome(
        decision=decision.value,
        centerline_id=centerline_id,
        review_state=successor.review.review_state.value,
        artifact_status=successor.status.value,
        new_centerline_id=response.artifact.artifact_id,
        correction_artifact_id=correction.artifact_id,
        superseded_artifact_id=centerline_id,
        staled_artifact_ids=tuple(staled),
    )


def centerline_ref(artifact_id: str) -> str:
    """The canonical reference form of a CENTERLINE artifact id."""

    return f"{CENTERLINE_REF_PREFIX}{artifact_id}"


def assert_centerline_coordinates(
    path: CoronaryPath,
) -> tuple[CenterlineCoordinate, ...]:
    """Validate the arc-length parametrization of one derivation (T-3 hook).

    ``s_mm`` must be monotonically non-decreasing and the coordinates must be
    the existing :class:`core.coordinates.CenterlineCoordinate` values.
    """

    previous = -1.0
    for coordinate in path.coordinates:
        if not isinstance(coordinate, CenterlineCoordinate):
            raise _refuse(
                CoronaryRefusalCode.VOLUME_INVALID, "coordinates must be CenterlineCoordinate"
            )
        if coordinate.s_mm < previous or coordinate.s_mm < 0.0 or coordinate.offset_mm != 0.0:
            raise _refuse(
                CoronaryRefusalCode.VOLUME_INVALID,
                "the arc-length parametrization is not monotonic",
            )
        previous = coordinate.s_mm
    if len(path.coordinates) != len(path.s_mm):
        raise _refuse(CoronaryRefusalCode.VOLUME_INVALID, "s_mm/coordinate length mismatch")
    return path.coordinates


def world_point_of(
    matrix: Matrix4, voxel: tuple[int, int, int]
) -> tuple[float, float, float]:
    """Voxel index -> world millimetres through one FrameSpec affine (test hook)."""

    point = apply_matrix(matrix, VoxelCoordinate(i=voxel[0], j=voxel[1], k=voxel[2]))
    return (float(point[0]), float(point[1]), float(point[2]))


__all__ = [
    "CENTERLINE_REF_PREFIX",
    "CORONARY_ACTOR_ID",
    "CenterlineBuildResult",
    "CoronaryDecision",
    "CoronaryDecisionOutcome",
    "CoronaryPath",
    "VESSEL_LABELS",
    "apply_coronary_decision",
    "assert_centerline_coordinates",
    "build_centerline",
    "centerline_provenance",
    "centerline_ref",
    "derive_coronary_path",
    "frame_spec_ref",
    "validate_centerline_payload",
    "world_point_of",
]
