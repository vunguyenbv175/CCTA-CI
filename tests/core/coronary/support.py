"""Deterministic fixtures of the CORONARY_CENTERLINE_001 suite.

The synthetic phantom is analytic ground truth: a helix with a focal narrowing
and a side branch, painted as a swept sphere (Table T-1).  Every number the
tests assert against is either hand-computable from this module or measured
polyline-to-polyline (never vertex-based).

No RNG is involved anywhere; the same inputs give the same volume, the same
derivation and the same CPR pixels on every run.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

from apps.workstation.volume_loader import (
    CANONICAL_INT16_CODEC,
    CanonicalVolumeLoader,
)
from core.coordinates import CoordinateTransform, FrameSpec, transform_for_frame
from core.coronary import (
    CenterlineBuildResult,
    Seed,
    VesselnessParameters,
    build_centerline,
    centerline_provenance,
)
from core.digest import canonical_digest
from core.enums import ArtifactKind, ProducerType, ProvenanceEventType, ReleaseLane, VolumeKind
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
from core.volume import Volume, VolumeBlobRef, create_volume
from visualization.cpr import CPRBuildResult, build_cpr_view, cpr_provenance

#: The fixture chain's recorded producing commit (40-hex; the viewer fixture
#: substitutes the checkout's resolved identity where one exists).
FIXED_COMMIT_SHA: Final[str] = "c0ffee" * 6 + "c0ff"

DIM: Final[int] = 48
HELIX_RADIUS_MM: Final[float] = 9.0
HELIX_TURNS: Final[float] = 1.5
HELIX_Z0: Final[float] = 6.0
HELIX_Z1: Final[float] = DIM - 6.0
BASE_RADIUS_MM: Final[float] = 2.0
TIP_RADIUS_MM: Final[float] = 1.5
TAPER_FRACTION: Final[float] = 0.03
NARROW_RADIUS_MM: Final[float] = 1.3
NARROW_INTERVAL: Final[tuple[float, float]] = (0.45, 0.58)
BRANCH_HEAD_RADIUS_MM: Final[float] = 0.8
BRANCH_TAIL_RADIUS_MM: Final[float] = 1.0
EXTENT_MM: Final[float] = 0.5
SCALES_MM: Final[tuple[float, ...]] = (1.0, 2.0)
THRESHOLD: Final[float] = 0.09
VOXEL_MM: Final[float] = 1.0

#: Fixed valid ULIDs (Crockford base32, 26 chars) for deterministic fixtures.
ULIDS: Final[dict[str, str]] = {
    "study": "01HZZZZZZZZZZZZZZZZZZZZZ00",
    "series": "01HZZZZZZZZZZZZZZZZZZZZZ01",
    "volume": "01HZZZZZZZZZZZZZZZZZZZZZ02",
    "transform_artifact": "01HZZZZZZZZZZZZZZZZZZZZZ03",
    "volume_provenance": "01HZZZZZZZZZZZZZZZZZZZZZ04",
    "transform_provenance": "01HZZZZZZZZZZZZZZZZZZZZZ05",
    "centerline_artifact": "01HZZZZZZZZZZZZZZZZZZZZZ06",
    "centerline_provenance": "01HZZZZZZZZZZZZZZZZZZZZZ07",
    "cpr_artifact": "01HZZZZZZZZZZZZZZZZZZZZZ08",
    "cpr_provenance": "01HZZZZZZZZZZZZZZZZZZZZZ09",
    "viewer_session": "01HZZZZZZZZZZZZZZZZZZZZZ10",
}

#: CORONARY_MULTI_VESSEL_NAV: deterministic ULIDs for the two extra fixture
#: vessels (LAD/LCx; the RCA path reuses the base centerline/cpr ULIDs).
VESSEL_EXTRA_ULIDS: Final[dict[str, dict[str, str]]] = {
    "VESSEL:LAD": {
        "centerline_artifact": "01HZZZZZZZZZZZZZZZZZZZZZ30",
        "centerline_provenance": "01HZZZZZZZZZZZZZZZZZZZZZ31",
        "cpr_artifact": "01HZZZZZZZZZZZZZZZZZZZZZ32",
        "cpr_provenance": "01HZZZZZZZZZZZZZZZZZZZZZ33",
    },
    "VESSEL:LCX": {
        "centerline_artifact": "01HZZZZZZZZZZZZZZZZZZZZZ34",
        "centerline_provenance": "01HZZZZZZZZZZZZZZZZZZZZZ35",
        "cpr_artifact": "01HZZZZZZZZZZZZZZZZZZZZZ36",
        "cpr_provenance": "01HZZZZZZZZZZZZZZZZZZZZZ37",
    },
}


def vessel_assignment_ids(vessel_label: str) -> dict[str, str]:
    """Deterministic artifact/provenance ULIDs for one fixture vessel."""

    return dict(VESSEL_EXTRA_ULIDS[vessel_label])


FRAME_OF_REFERENCE_UID: Final[str] = "1.2.826.0.1.3680043.8.498.77.900"
STUDY_INSTANCE_UID: Final[str] = "1.2.826.0.1.3680043.8.498.77.901"
SERIES_INSTANCE_UID: Final[str] = "1.2.826.0.1.3680043.8.498.77.902"


def helix_points(n: int = 600) -> np.ndarray:
    """The analytic helix centreline in ``(i, j, k)`` voxel coordinates."""

    t = np.linspace(0.0, 1.0, n)
    angle = 2.0 * math.pi * HELIX_TURNS * t
    x = 0.5 * DIM + HELIX_RADIUS_MM * np.cos(angle)
    y = 0.5 * DIM + HELIX_RADIUS_MM * np.sin(angle)
    z = HELIX_Z0 + (HELIX_Z1 - HELIX_Z0) * t
    return np.stack([x, y, z], axis=-1)


def branch_points(n: int = 90, length: float = 5.0) -> np.ndarray:
    """The analytic side branch (from helix[300], length 5 mm)."""

    base = helix_points()[300]
    direction = np.array([0.75, 0.6, 0.25])
    direction = direction / np.linalg.norm(direction)
    t = np.linspace(0.0, 1.0, n)
    return base[None, :] + t[:, None] * length * direction[None, :]


#: CORONARY_MULTI_VESSEL_NAV fixtures: three deterministic analytic vessel
#: curves in ``(i, j, k)`` voxel coordinates on the same 48^3 phantom grid.
#: LAD reuses the helix; LCx is the helix mirrored in i (a distinct,
#: non-overlapping path); RCA is a straight diagonal tube in the opposite
#: corner.  All stay inside the volume with margin for the swept-sphere paint.
VESSEL_CURVES: Final[dict[str, str]] = {
    "VESSEL:LAD": "helix",
    "VESSEL:LCX": "mirrored-helix",
    "VESSEL:RCA": "diagonal",
}
VESSEL_SEED_INDICES: Final[dict[str, int]] = {
    "VESSEL:LAD": 40,
    "VESSEL:LCX": 40,
    "VESSEL:RCA": 30,
}


def vessel_points(vessel_label: str, n: int = 600) -> np.ndarray:
    """Analytic centerline voxels of one fixture vessel (deterministic)."""

    if vessel_label == "VESSEL:LAD":
        return helix_points(n=n)
    if vessel_label == "VESSEL:LCX":
        base = helix_points(n=n)
        mirrored = base.copy()
        mirrored[:, 0] = float(DIM) - base[:, 0]
        return np.ascontiguousarray(mirrored)
    if vessel_label == "VESSEL:RCA":
        t = np.linspace(0.0, 1.0, n)
        start = np.array([6.0, 6.0, float(DIM) - 8.0])
        end = np.array([float(DIM) - 10.0, float(DIM) - 6.0, 8.0])
        return np.ascontiguousarray(start[None, :] + t[:, None] * (end - start)[None, :])
    raise ValueError(f"unknown fixture vessel {vessel_label!r}")


def vessel_seed(vessel_label: str) -> Seed:
    """One human seed on the analytic vessel curve, with its tangent hint."""

    curve = vessel_points(vessel_label)
    index = VESSEL_SEED_INDICES[vessel_label]
    probe = curve[index]
    ahead = curve[min(index + 20, len(curve) - 1)]
    behind = curve[max(index - 20, 0)]
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


def radii_profile(n: int) -> np.ndarray:
    """The analytic radius profile: base, focal narrowing, tapered ends."""

    t = np.linspace(0.0, 1.0, n)
    radius = np.full(n, BASE_RADIUS_MM)
    narrow = (t >= NARROW_INTERVAL[0]) & (t <= NARROW_INTERVAL[1])
    radius[narrow] = NARROW_RADIUS_MM
    taper = np.minimum(t / TAPER_FRACTION, (1.0 - t) / TAPER_FRACTION)
    radius = np.minimum(
        radius, TIP_RADIUS_MM + (BASE_RADIUS_MM - TIP_RADIUS_MM) * np.clip(taper, 0.0, 1.0)
    )
    return radius


def extend_curve(
    curve: np.ndarray, radii: np.ndarray, distance: float
) -> tuple[np.ndarray, np.ndarray]:
    """Extend a curve by ``distance`` at each end along its end tangents."""

    start_direction = curve[0] - curve[1]
    start_direction = start_direction / np.linalg.norm(start_direction)
    end_direction = curve[-1] - curve[-2]
    end_direction = end_direction / np.linalg.norm(end_direction)
    extended = np.vstack(
        [curve[0] - start_direction * distance, curve, curve[-1] + end_direction * distance]
    )
    extended_radii = np.concatenate([[radii[0]], radii, [radii[-1]]])
    return extended, extended_radii


def paint_variable(shape: tuple[int, int, int], curve: np.ndarray, radii: np.ndarray) -> np.ndarray:
    """Voxelize a swept sphere of a variable radius in ``(i, j, k)`` space."""

    mask = np.zeros(shape, dtype=bool)
    for point, radius in zip(curve, radii):
        r = int(math.ceil(float(radius)))
        ci, cj, ck = (
            int(round(float(point[0]))),
            int(round(float(point[1]))),
            int(round(float(point[2]))),
        )
        i0, i1 = max(0, ci - r), min(shape[0], ci + r + 1)
        j0, j1 = max(0, cj - r), min(shape[1], cj + r + 1)
        k0, k1 = max(0, ck - r), min(shape[2], ck + r + 1)
        if i0 >= i1 or j0 >= j1 or k0 >= k1:
            continue
        ii, jj, kk = np.meshgrid(
            np.arange(i0, i1), np.arange(j0, j1), np.arange(k0, k1), indexing="ij"
        )
        distance = np.sqrt(
            (ii - point[0]) ** 2 + (jj - point[1]) ** 2 + (kk - point[2]) ** 2
        )
        mask[i0:i1, j0:j1, k0:k1] |= distance <= radius
    return mask


def analytic_volume(
    union_extra_vessels: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(raw, analytic, helix_extended, branch_extended)``.

    ``raw`` is the canonical ``(k, j, i)`` int16 HU volume; ``analytic`` is its
    boolean ground-truth mask in the same ``(k, j, i)`` layout.

    CORONARY_MULTI_VESSEL_NAV: ``union_extra_vessels=True`` additionally paints
    the LCx (mirrored helix) and RCA (diagonal) fixture tubes so all three
    vessel seeds carry vessel evidence.  Default ``False`` preserves the exact
    #28 base bytes (phantom gate).
    """

    helix = helix_points()
    branch = branch_points()
    helix_extended, helix_radii = extend_curve(helix, radii_profile(len(helix)), EXTENT_MM)
    branch_radii = np.linspace(BRANCH_HEAD_RADIUS_MM, BRANCH_TAIL_RADIUS_MM, len(branch))
    branch_extended, branch_radii_extended = extend_curve(branch, branch_radii, EXTENT_MM)
    mask_ik = paint_variable(
        (DIM, DIM, DIM), helix_extended, helix_radii
    ) | paint_variable((DIM, DIM, DIM), branch_extended, branch_radii_extended)
    if union_extra_vessels:
        for _label in ("VESSEL:LCX", "VESSEL:RCA"):
            _curve = vessel_points(_label, n=200)
            _radii = np.full(len(_curve), 1.5)
            mask_ik = mask_ik | paint_variable((DIM, DIM, DIM), _curve, _radii)
    analytic = np.transpose(mask_ik, (2, 1, 0))
    raw = np.full((DIM, DIM, DIM), -100, dtype=np.int16)
    raw[analytic] = 300
    return (raw, np.ascontiguousarray(analytic), helix_extended, branch_extended)


def phantom_frame() -> FrameSpec:
    return FrameSpec(
        frame_of_reference_uid=FRAME_OF_REFERENCE_UID,
        image_position_patient=(0.0, 0.0, 0.0),
        image_orientation_patient=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        spacing_mm=(VOXEL_MM, VOXEL_MM, VOXEL_MM),
        dimensions=(DIM, DIM, DIM),
    )


def phantom_seed() -> Seed:
    """One human seed on the analytic helix, with the analytic tangent hint."""

    helix = helix_points()
    probe = helix[40]
    return Seed(
        i=int(round(probe[0])),
        j=int(round(probe[1])),
        k=int(round(probe[2])),
        direction=(
            float(helix[60][0] - helix[20][0]),
            float(helix[60][1] - helix[20][1]),
            float(helix[60][2] - helix[20][2]),
        ),
    )


def phantom_parameters() -> VesselnessParameters:
    return VesselnessParameters(scales_mm=SCALES_MM, threshold=THRESHOLD)


def straight_tube_volume(
    *,
    axis: str,
    span: tuple[int, int],
    center: int,
    radius_mm: float,
    dimensions: int = 32,
) -> np.ndarray:
    """An int16 HU volume with one bright straight tube (F2/F3 probes).

    ``axis`` is the PHYSICAL axis (``"i"`` = world x, ``"j"`` = world y,
    ``"k"`` = world z) for the identity-orientation frame.  The tube is
    painted as a swept sphere of ``radius_mm`` along that axis between the two
    voxel indices of ``span`` (inclusive); the other two physical axes are
    centred on ``center``.  The returned array is ``(k, j, i)``, -100 HU
    background / +300 HU tube.
    """

    if axis not in {"i", "j", "k"}:
        raise ValueError("axis must be one of i, j, k")
    array_axis = {"i": 2, "j": 1, "k": 0}[axis]
    physical_axis = {"i": 0, "j": 1, "k": 2}[axis]
    volume = np.full((dimensions, dimensions, dimensions), -100, dtype=np.int16)
    grid = np.indices(volume.shape).astype(np.float64)
    low, high = (float(min(span)), float(max(span)))
    along = grid[array_axis]
    axial = np.maximum(np.maximum(low - along, along - high), 0.0)
    distance_squared = axial * axial
    for physical in range(3):
        if physical == physical_axis:
            continue
        distance_squared = distance_squared + (
            grid[2 - physical] - float(center)
        ) ** 2
    volume[np.sqrt(distance_squared) <= float(radius_mm)] = 300
    return volume


def identity_frame(dimensions: int = 32) -> FrameSpec:
    """The identity-orientation (i=x, j=y, k=z) 1 mm frame of F2/F3 probes."""

    return FrameSpec(
        frame_of_reference_uid="1.2.826.0.1.3680043.8.498.77.930",
        image_position_patient=(0.0, 0.0, 0.0),
        image_orientation_patient=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        spacing_mm=(1.0, 1.0, 1.0),
        dimensions=(dimensions, dimensions, dimensions),
    )


def as_float(value: object, label: str = "value") -> float:
    """Coerce one payload number strictly (test helper, never a default)."""

    assert isinstance(value, (int, float)) and not isinstance(value, bool), label
    return float(value)


def as_int(value: object, label: str = "value") -> int:
    """Coerce one payload integer strictly (test helper, never a default)."""

    assert isinstance(value, int) and not isinstance(value, bool), label
    return int(value)


def segment_distances(
    points: np.ndarray, curve: np.ndarray, chunk: int = 1024
) -> np.ndarray:
    """Min point-to-SEGMENT distance from every point to a polyline."""

    starts = curve[:-1][None, :, :]
    ends = curve[1:][None, :, :]
    segment = ends - starts
    length_squared = np.sum(segment * segment, axis=-1)
    length_squared[length_squared == 0.0] = 1.0
    output = np.empty(points.shape[0], dtype=np.float64)
    for start in range(0, points.shape[0], chunk):
        block = points[start : start + chunk][:, None, :]
        offset = block - starts
        projection = np.clip(
            np.sum(offset * segment, axis=-1) / length_squared, 0.0, 1.0
        )
        closest = starts + projection[..., None] * segment
        distance = np.sqrt(np.sum((block - closest) ** 2, axis=-1))
        output[start : start + chunk] = distance.min(axis=1)
    return output


@dataclass(frozen=True)
class SliceFixture:
    """The complete slice chain: volume -> transform -> centerline -> CPR."""

    loader: CanonicalVolumeLoader
    raw: np.ndarray
    analytic: np.ndarray
    helix_extended: np.ndarray
    branch_extended: np.ndarray
    frame: FrameSpec
    transform: CoordinateTransform
    volume: Volume
    seed: Seed
    parameters: VesselnessParameters
    centerline: CenterlineBuildResult
    centerline_provenance: ProvenanceRecord
    volume_provenance: ProvenanceRecord
    transform_provenance: ProvenanceRecord
    cpr: CPRBuildResult
    cpr_provenance: ProvenanceRecord
    code_commit_sha: str


def _root_environment(code_commit_sha: str) -> Environment:
    return Environment(
        software_version="ccta-coronary-fixture",
        code_commit_sha=code_commit_sha,
        runtime="python-3.12 (coronary fixture)",
        deterministic_seed=None,
    )


def build_volume_provenance(
    volume: Volume, *, code_commit_sha: str
) -> ProvenanceRecord:
    return create_provenance(
        subject_ref=f"volume_id:{volume.volume_id}",
        event_type=ProvenanceEventType.IMPORTED,
        actor=Actor(actor_type=ActorType.ENGINE, actor_id="ccta-coronary-fixture"),
        inputs=(
            InputRef(
                ref="synthetic_dicom_series:coronary-phantom",
                kind="series",
                sha256_at_use=volume.blob_ref.sha256,
            ),
        ),
        preprocessing="synthetic phantom materialization (no DICOM; no PHI)",
        parameters_ref=BlobRef(
            blob_ref="coronary-phantom-parameters",
            sha256=canonical_digest(
                {"dimensions": DIM, "spacing_mm": VOXEL_MM, "layout": "(k,j,i)"}
            ),
        ),
        human_edit=NotApplicable(value=None, reason="synthetic analytic phantom"),
        upstream_refs=NotApplicable(
            value=None, reason="root synthetic phantom import (no upstream record)"
        ),
        environment=_root_environment(code_commit_sha),
        created_by="ccta-coronary-fixture",
        at="2026-09-14T00:00:00Z",
        provenance_id=ULIDS["volume_provenance"],
    )


def build_transform_provenance(
    volume: Volume, volume_provenance: ProvenanceRecord, *, code_commit_sha: str
) -> ProvenanceRecord:
    return create_provenance(
        subject_ref=f"coordinate_transform:{ULIDS['transform_artifact']}",
        event_type=ProvenanceEventType.DERIVED,
        actor=Actor(actor_type=ActorType.ENGINE, actor_id="ccta-coronary-fixture"),
        inputs=(
            InputRef(
                ref=f"volume_id:{volume.volume_id}",
                kind="volume",
                sha256_at_use=volume.blob_ref.sha256,
            ),
        ),
        preprocessing="frame-derived voxel<->world affine (transform_for_frame)",
        parameters_ref=BlobRef(
            blob_ref="coronary-phantom-frame-affine",
            sha256=canonical_digest({"frame": str(FRAME_OF_REFERENCE_UID)}),
        ),
        human_edit=NotApplicable(value=None, reason="deterministic frame affine"),
        upstream_refs=(volume_provenance.provenance_id,),
        environment=_root_environment(code_commit_sha),
        created_by="ccta-coronary-fixture",
        at="2026-09-14T00:00:00Z",
        provenance_id=ULIDS["transform_provenance"],
    )


def build_centerline_for_vessel(
    volume: Volume,
    transform: CoordinateTransform,
    frame: FrameSpec,
    *,
    loader: CanonicalVolumeLoader,
    vessel_label: str,
    artifact_ids: dict[str, str],
    study_instance_uid: str = STUDY_INSTANCE_UID,
    frame_of_reference_uid: str = FRAME_OF_REFERENCE_UID,
    code_commit_sha: str = FIXED_COMMIT_SHA,
    at: str = "2026-09-14T00:00:00Z",
) -> tuple[CenterlineBuildResult, ProvenanceRecord, CPRBuildResult, ProvenanceRecord]:
    """Derive one labeled centerline + CPR view on an existing volume.

    CORONARY_MULTI_VESSEL_NAV: the vessel curve/seed come from
    :func:`vessel_points`/:func:`vessel_seed`; the derivation calls are the
    same governed ``build_centerline``/``build_cpr_view`` the base fixture
    uses (no parallel implementation).  All ids come from the caller so the
    viewer fixture owns ULID allocation.
    """

    from core.volume import Volume as _Volume  # noqa: F401  (type use below)

    seed = vessel_seed(vessel_label)
    parameters = phantom_parameters()
    centerline = build_centerline(
        volume,
        transform,
        frame,
        loader=loader,
        seed=seed,
        parameters=parameters,
        provenance_ref=f"provenance:{artifact_ids['centerline_provenance']}",
        transform_artifact_ref=f"coordinate_transform:{ULIDS['transform_artifact']}",
        study_instance_uid=study_instance_uid,
        frame_of_reference_uid=frame_of_reference_uid,
        scalar_domain="HU",
        source_array=None,
        vessel_label=vessel_label,
        artifact_id=artifact_ids["centerline_artifact"],
        created_by="ccta-coronary-fixture",
    )
    volume_provenance_id = ULIDS["volume_provenance"]
    transform_provenance_id = ULIDS["transform_provenance"]
    centerline_provenance_record = centerline_provenance(
        subject_ref=f"centerline_artifact_id:{centerline.artifact.artifact_id}",
        volume=volume,
        seed=seed,
        parameters=parameters,
        mask_sha256=centerline.path.diagnostics.mask_sha256,
        mask_blob_sha256=str(centerline.payload["vessel_mask_blob_ref"]["sha256"]),  # type: ignore[index]
        response_sha256=centerline.path.diagnostics.response_sha256,
        transform_artifact_ref=f"coordinate_transform:{ULIDS['transform_artifact']}",
        transform_digest=canonical_digest(
            {"transform_id": transform.transform_id, "version": transform.version}
        ),
        volume_provenance_ref=volume_provenance_id,
        transform_provenance_ref=transform_provenance_id,
        software_version="ccta-coronary-fixture",
        code_commit_sha=code_commit_sha,
        runtime="python-3.12 (coronary fixture)",
        created_by="ccta-coronary-fixture",
        at=at,
        provenance_id=artifact_ids["centerline_provenance"],
    )
    cpr = build_cpr_view(
        volume,
        transform,
        centerline=centerline,
        s_spacing_mm=2.0,
        radial_spacing_mm=1.0,
        radial_count=5,
        method="TRILINEAR",
        out_of_bounds_fill_value=0.0,
        loader=loader,
        provenance_ref=f"provenance:{artifact_ids['cpr_provenance']}",
        transform_artifact_ref=f"coordinate_transform:{ULIDS['transform_artifact']}",
        study_instance_uid=study_instance_uid,
        series_instance_uid=SERIES_INSTANCE_UID,
        frame_of_reference_uid=frame_of_reference_uid,
        artifact_id=artifact_ids["cpr_artifact"],
        created_by="ccta-cpr-fixture",
    )
    cpr_provenance_record = cpr_provenance(
        subject_ref=f"mpr_artifact_id:{cpr.artifact.artifact_id}",
        volume=volume,
        centerline_artifact=centerline.artifact,
        sampling_parameters_sha256=str(cpr.payload["sampling_parameters_sha256"]),
        volume_provenance_ref=volume_provenance_id,
        centerline_provenance_ref=centerline_provenance_record.provenance_id,
        software_version="ccta-cpr-fixture",
        code_commit_sha=code_commit_sha,
        runtime="python-3.12 (coronary fixture)",
        created_by="ccta-cpr-fixture",
        at=at,
        provenance_id=artifact_ids["cpr_provenance"],
    )
    return (centerline, centerline_provenance_record, cpr, cpr_provenance_record)


def build_slice_fixture(
    workspace: Path,
    *,
    blobs_dir: Path | None = None,
    code_commit_sha: str = FIXED_COMMIT_SHA,
    union_extra_vessels: bool = False,
) -> SliceFixture:
    """Build and derive the whole slice once (deterministic)."""

    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    loader = CanonicalVolumeLoader(blobs_dir if blobs_dir is not None else workspace / "blobs")
    raw, analytic, helix_extended, branch_extended = analytic_volume(
        union_extra_vessels=union_extra_vessels
    )
    materialized = loader.materialize_canonical(
        raw, scalar_domain="HU", intensity_unit="HU"
    )
    frame = phantom_frame()
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
    volume_provenance = build_volume_provenance(volume, code_commit_sha=code_commit_sha)
    transform_provenance = build_transform_provenance(
        volume, volume_provenance, code_commit_sha=code_commit_sha
    )
    transform = transform_for_frame(frame, series_instance_uid=SERIES_INSTANCE_UID)
    seed = phantom_seed()
    parameters = phantom_parameters()
    centerline = build_centerline(
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
        vessel_label="VESSEL:RCA",
        artifact_id=ULIDS["centerline_artifact"],
        created_by="ccta-coronary-fixture",
    )
    centerline_provenance_record = centerline_provenance(
        subject_ref=f"centerline_artifact_id:{centerline.artifact.artifact_id}",
        volume=volume,
        seed=seed,
        parameters=parameters,
        mask_sha256=centerline.path.diagnostics.mask_sha256,
        mask_blob_sha256=str(centerline.payload["vessel_mask_blob_ref"]["sha256"]),  # type: ignore[index]
        response_sha256=centerline.path.diagnostics.response_sha256,
        transform_artifact_ref=f"coordinate_transform:{ULIDS['transform_artifact']}",
        transform_digest=canonical_digest(
            {"transform_id": transform.transform_id, "version": transform.version}
        ),
        volume_provenance_ref=volume_provenance.provenance_id,
        transform_provenance_ref=transform_provenance.provenance_id,
        software_version="ccta-coronary-fixture",
        code_commit_sha=code_commit_sha,
        runtime="python-3.12 (coronary fixture)",
        created_by="ccta-coronary-fixture",
        at="2026-09-14T00:00:00Z",
        provenance_id=ULIDS["centerline_provenance"],
    )
    cpr = build_cpr_view(
        volume,
        transform,
        centerline=centerline,
        s_spacing_mm=2.0,
        radial_spacing_mm=1.0,
        radial_count=5,
        method="TRILINEAR",
        out_of_bounds_fill_value=0.0,
        loader=loader,
        provenance_ref=f"provenance:{ULIDS['cpr_provenance']}",
        transform_artifact_ref=f"coordinate_transform:{ULIDS['transform_artifact']}",
        study_instance_uid=STUDY_INSTANCE_UID,
        series_instance_uid=SERIES_INSTANCE_UID,
        frame_of_reference_uid=FRAME_OF_REFERENCE_UID,
        artifact_id=ULIDS["cpr_artifact"],
        created_by="ccta-cpr-fixture",
    )
    cpr_provenance_record = cpr_provenance(
        subject_ref=f"mpr_artifact_id:{cpr.artifact.artifact_id}",
        volume=volume,
        centerline_artifact=centerline.artifact,
        sampling_parameters_sha256=str(cpr.payload["sampling_parameters_sha256"]),
        volume_provenance_ref=volume_provenance.provenance_id,
        centerline_provenance_ref=centerline_provenance_record.provenance_id,
        software_version="ccta-cpr-fixture",
        code_commit_sha=code_commit_sha,
        runtime="python-3.12 (coronary fixture)",
        created_by="ccta-cpr-fixture",
        at="2026-09-14T00:00:00Z",
        provenance_id=ULIDS["cpr_provenance"],
    )
    assert cpr.artifact.kind is ArtifactKind.MPR_VIEW
    assert centerline.artifact.kind is ArtifactKind.CENTERLINE
    assert cpr.artifact.producer.producer_type is ProducerType.DETERMINISTIC_ENGINE
    return SliceFixture(
        loader=loader,
        raw=raw,
        analytic=analytic,
        helix_extended=helix_extended,
        branch_extended=branch_extended,
        frame=frame,
        transform=transform,
        volume=volume,
        seed=seed,
        parameters=parameters,
        centerline=centerline,
        centerline_provenance=centerline_provenance_record,
        volume_provenance=volume_provenance,
        transform_provenance=transform_provenance,
        cpr=cpr,
        cpr_provenance=cpr_provenance_record,
        code_commit_sha=code_commit_sha,
    )


__all__ = [
    "BRANCH_HEAD_RADIUS_MM",
    "DIM",
    "FIXED_COMMIT_SHA",
    "FRAME_OF_REFERENCE_UID",
    "NARROW_RADIUS_MM",
    "SCALES_MM",
    "SERIES_INSTANCE_UID",
    "STUDY_INSTANCE_UID",
    "SliceFixture",
    "THRESHOLD",
    "ULIDS",
    "VESSEL_EXTRA_ULIDS",
    "VOXEL_MM",
    "analytic_volume",
    "as_float",
    "as_int",
    "branch_points",
    "build_centerline_for_vessel",
    "build_slice_fixture",
    "extend_curve",
    "helix_points",
    "identity_frame",
    "paint_variable",
    "phantom_frame",
    "phantom_parameters",
    "phantom_seed",
    "radii_profile",
    "vessel_assignment_ids",
    "vessel_points",
    "vessel_seed",
    "segment_distances",
    "straight_tube_volume",
]
