"""T-1: the analytic phantom gate (mask Dice + polyline Hausdorff <= voxel).

The phantom is the frozen ``support.analytic_volume``: a helix with a focal
narrowing and a side branch, painted as a swept sphere with a 0.5 mm end
extension.  Distances are POLYLINE-TO-POLYLINE (point-to-segment), never
vertex-based; the voxel is 1.0 mm isotropic.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from apps.workstation.volume_loader import CanonicalVolumeLoader
from core.coordinates import FrameSpec, transform_for_frame
from core.coronary import (
    CoronaryRefusalCode,
    CoronaryValidationError,
    Seed,
    VesselnessParameters,
    build_centerline,
    derive_coronary_path,
    thinning_skeleton,
)
from core.enums import ReleaseLane, VolumeKind
from core.volume import VolumeBlobRef, create_volume
from tests.core.coronary.support import (
    VOXEL_MM,
    SliceFixture,
    as_float,
    identity_frame,
    segment_distances,
    straight_tube_volume,
)

_TUBE_PARAMETERS = VesselnessParameters(scales_mm=(1.0, 2.0), threshold=0.05)


def test_phantom_mask_dice_and_hausdorff_within_one_voxel(
    slice_fixture: SliceFixture,
) -> None:
    """Measured on the frozen phantom: dice 0.9309, p->h 0.9376, h->p 0.8640."""

    mask = slice_fixture.centerline.path.mask
    analytic = slice_fixture.analytic
    assert int(mask.sum()) > 0
    assert int(analytic.sum()) > 0
    intersection = int(np.count_nonzero(mask & analytic))
    dice = 2.0 * intersection / (int(mask.sum()) + int(analytic.sum()))
    assert dice >= 0.90, f"phantom mask dice {dice:.4f} is below 0.90"

    points = np.asarray(slice_fixture.centerline.path.points_world, dtype=np.float64)
    produced_to_analytic = float(
        segment_distances(points, slice_fixture.helix_extended).max()
    )
    analytic_to_produced = float(
        segment_distances(slice_fixture.helix_extended, points).max()
    )
    assert produced_to_analytic <= VOXEL_MM
    assert analytic_to_produced <= VOXEL_MM


def test_centerline_is_one_ordered_polyline(slice_fixture: SliceFixture) -> None:
    """ONE path: ordered world points, monotonic arc lengths from 0."""

    path = slice_fixture.centerline.path
    points = np.asarray(path.points_world, dtype=np.float64)
    arcs = np.asarray(path.s_mm, dtype=np.float64)
    assert points.ndim == 2 and points.shape[1] == 3
    assert arcs.shape[0] == points.shape[0] >= 2
    assert float(arcs[0]) == 0.0
    assert bool(np.all(np.diff(arcs) >= 0.0))
    # The polyline is connected: no jump larger than the voxel diagonal.
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    assert float(steps.max()) <= float(np.sqrt(3.0) * VOXEL_MM) + 1e-9
    # Every recorded arc length matches the travelled polyline distance.
    travelled = np.concatenate([[0.0], np.cumsum(steps)])
    assert bool(np.allclose(arcs, travelled, atol=1e-9))


def test_derivation_is_bit_identical_across_runs(slice_fixture: SliceFixture) -> None:
    """Same volume + seed + parameters = the same response/mask/path (no RNG)."""

    repeated = derive_coronary_path(
        slice_fixture.raw,
        (VOXEL_MM, VOXEL_MM, VOXEL_MM),
        slice_fixture.frame,
        slice_fixture.seed,
        slice_fixture.parameters,
    )
    original = slice_fixture.centerline.path
    assert repeated.diagnostics.response_sha256 == original.diagnostics.response_sha256
    assert repeated.diagnostics.mask_sha256 == original.diagnostics.mask_sha256
    assert repeated.diagnostics.mask_voxel_count == original.diagnostics.mask_voxel_count
    assert repeated.voxel_path.tobytes() == original.voxel_path.tobytes()
    assert repeated.points_world == original.points_world
    assert repeated.s_mm == original.s_mm


def test_seed_projection_is_recorded_in_the_payload(slice_fixture: SliceFixture) -> None:
    """A seed beside the skeleton is projected and the distance is RECORDED."""

    payload = slice_fixture.centerline.payload
    skeleton_voxel = payload["seed_skeleton_voxel"]
    assert isinstance(skeleton_voxel, list) and len(skeleton_voxel) == 3
    snap = as_float(payload["seed_snap_mm"], "seed_snap_mm")
    assert 0.0 <= snap <= float(np.sqrt(3.0) * VOXEL_MM)
    if snap > 0.0:
        # A real projection: the projected voxel is a live skeleton voxel.
        i, j, k = (int(skeleton_voxel[0]), int(skeleton_voxel[1]), int(skeleton_voxel[2]))
        assert bool(slice_fixture.centerline.path.skeleton[k, j, i])


# ---------------------------------------------------------------------------
# att0002 corrections: F2 (direction hint) / F3 (anisotropic spacing) /
# F4 (seed branch preservation or typed abstention)
# ---------------------------------------------------------------------------


def _tube_path(
    axis: str, direction: tuple[float, float, float]
) -> tuple[tuple[float, float, float], ...]:
    volume = straight_tube_volume(axis=axis, span=(6, 26), center=16, radius_mm=2.0)
    seed = Seed(i=16, j=16, k=16, direction=direction)
    path = derive_coronary_path(
        volume, (1.0, 1.0, 1.0), identity_frame(), seed, _TUBE_PARAMETERS
    )
    return path.points_world


def _oriented_axis(axis: str, direction: tuple[float, float, float]) -> tuple[float, float]:
    physical = {"i": 0, "j": 1, "k": 2}[axis]
    points = _tube_path(axis, direction)
    return (points[0][physical], points[-1][physical])


def test_direction_hint_uses_the_i_j_k_axis_order_and_points_ahead() -> None:
    """F2: +X/+Y/+Z orient ahead; the opposite hints orient behind."""

    for axis, positive, negative in (
        ("i", (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)),
        ("j", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0)),
        ("k", (0.0, 0.0, 1.0), (0.0, 0.0, -1.0)),
    ):
        first, last = _oriented_axis(axis, positive)
        assert first < last, f"+{axis} hint must end AHEAD of the seed"
        first, last = _oriented_axis(axis, negative)
        assert first > last, f"-{axis} hint must end BEHIND the seed"


def test_direction_hint_follows_an_oblique_world_affine() -> None:
    """F2: the hint is mapped through the persisted frame, not renamed."""

    frame = FrameSpec(
        frame_of_reference_uid="1.2.826.0.1.3680043.8.498.77.931",
        image_position_patient=(0.0, 0.0, 0.0),
        image_orientation_patient=(0.8, 0.6, 0.0, -0.6, 0.8, 0.0),
        spacing_mm=(1.0, 1.0, 1.0),
        dimensions=(32, 32, 32),
    )
    volume = straight_tube_volume(axis="i", span=(6, 26), center=16, radius_mm=2.0)
    # The world image of the +i axis is the frame's row cosine (0.8, 0.6, 0).
    forward = derive_coronary_path(
        volume,
        (1.0, 1.0, 1.0),
        frame,
        Seed(i=16, j=16, k=16, direction=(0.8, 0.6, 0.0)),
        _TUBE_PARAMETERS,
    )
    backward = derive_coronary_path(
        volume,
        (1.0, 1.0, 1.0),
        frame,
        Seed(i=16, j=16, k=16, direction=(-0.8, -0.6, 0.0)),
        _TUBE_PARAMETERS,
    )
    assert forward.points_world[0][0] < forward.points_world[-1][0]
    assert backward.points_world[0][0] > backward.points_world[-1][0]


def test_gaussian_smoothing_maps_physical_spacing_to_array_axes() -> None:
    """F3: (i,j,k) spacing is applied to the (k,j,i) array axes explicitly.

    For ``spacing_mm=(1, 1, 2)`` and a 1 mm Gaussian, an impulse spreads
    ~1.0 index units along i and j but ~0.5 index units along k; the old
    raw-axis mapping produced (1.0, 1.0, 0.5) in array order instead.
    """

    from core.coronary.vesselness import _smooth

    impulse = np.zeros((21, 21, 21), dtype=np.float64)
    impulse[10, 10, 10] = 1.0
    smoothed = _smooth(impulse, (1.0, 1.0, 2.0), 1.0)
    coordinates = np.arange(21, dtype=np.float64) - 10.0
    sigmas: list[float] = []
    for axis in range(3):
        line = smoothed.sum(
            axis=tuple(other for other in range(3) if other != axis)
        )
        variance = float((line * coordinates * coordinates).sum() / line.sum())
        sigmas.append(math.sqrt(variance))
    assert abs(sigmas[0] - 0.5) < 0.05, f"k sigma {sigmas[0]}"
    assert abs(sigmas[1] - 1.0) < 0.05, f"j sigma {sigmas[1]}"
    assert abs(sigmas[2] - 1.0) < 0.05, f"i sigma {sigmas[2]}"


def test_anisotropic_spacing_keeps_the_derivation_orientable() -> None:
    """F3: an anisotropic grid still yields a hint-oriented single path."""

    frame = FrameSpec(
        frame_of_reference_uid="1.2.826.0.1.3680043.8.498.77.932",
        image_position_patient=(0.0, 0.0, 0.0),
        image_orientation_patient=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        spacing_mm=(1.0, 1.0, 2.0),
        dimensions=(32, 32, 32),
    )
    volume = straight_tube_volume(axis="i", span=(6, 26), center=16, radius_mm=2.0)
    points = derive_coronary_path(
        volume,
        (1.0, 1.0, 2.0),
        frame,
        Seed(i=16, j=16, k=16, direction=(1.0, 0.0, 0.0)),
        _TUBE_PARAMETERS,
    ).points_world
    assert points[0][0] < points[-1][0]


def test_centerline_payload_records_the_physical_i_j_k_spacing(tmp_path: Path) -> None:
    """F3: ``build_centerline`` records the frame's (i, j, k) spacing verbatim
    and the retained mask-evidence blob at exactly one byte per voxel."""

    loader = CanonicalVolumeLoader(tmp_path / "blobs")
    raw = straight_tube_volume(axis="i", span=(8, 24), center=16, radius_mm=2.0)
    materialized = loader.materialize_canonical(raw, scalar_domain="HU", intensity_unit="HU")
    frame = FrameSpec(
        frame_of_reference_uid="1.2.826.0.1.3680043.8.498.77.933",
        image_position_patient=(0.0, 0.0, 0.0),
        image_orientation_patient=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        spacing_mm=(1.0, 1.0, 2.0),
        dimensions=(32, 32, 32),
    )
    volume = create_volume(
        study_ref="study_id:01HZZZZZZZZZZZZZZZZZZZZZ60",
        source_series_refs=("series:01HZZZZZZZZZZZZZZZZZZZZZ61",),
        kind=VolumeKind.CANONICAL,
        coordinate_frame=frame,
        blob_ref=VolumeBlobRef(
            sha256=materialized.blob.sha256,
            codec="int16-le",
            byte_size=materialized.blob.byte_size,
        ),
        provenance_ref="provenance:01HZZZZZZZZZZZZZZZZZZZZZ62",
        release_lane=ReleaseLane.PERSONAL_RESEARCH,
        volume_id="01HZZZZZZZZZZZZZZZZZZZZZ63",
    )
    transform = transform_for_frame(frame, series_instance_uid="1.2.826.0.1.3680043.8.498.77.934")
    result = build_centerline(
        volume,
        transform,
        frame,
        loader=loader,
        seed=Seed(i=16, j=16, k=16, direction=(1.0, 0.0, 0.0)),
        parameters=_TUBE_PARAMETERS,
        provenance_ref="provenance:01HZZZZZZZZZZZZZZZZZZZZZ64",
        transform_artifact_ref="coordinate_transform:01HZZZZZZZZZZZZZZZZZZZZZ65",
        study_instance_uid="1.2.826.0.1.3680043.8.498.77.935",
        frame_of_reference_uid=frame.frame_of_reference_uid,
        scalar_domain="HU",
        source_array=raw,
        artifact_id="01HZZZZZZZZZZZZZZZZZZZZZ66",
        created_by="ccta-coronary-fixture",
    )
    assert result.payload["voxel_spacing_mm"] == [1.0, 1.0, 2.0]
    blob_ref = result.payload["vessel_mask_blob_ref"]
    assert isinstance(blob_ref, dict)
    assert int(blob_ref["byte_size"]) == int(raw.size)
    assert result.path.points_world[0][0] < result.path.points_world[-1][0]


def _t_mask() -> np.ndarray:
    """A connected T: crossbar along i, short branch along +j, seeded at the
    branch tip (the F4 counterexample shape)."""

    mask = np.zeros((24, 24, 24), dtype=bool)
    mask[10:13, 10:13, 4:21] = True  # crossbar along i
    mask[10:13, 12:20, 11:14] = True  # branch along +j
    return mask


def test_seeded_branch_is_preserved_when_the_diameter_would_discard_it() -> None:
    """F4: a T-mask seeded at the branch tip keeps the seed's branch in the
    ONE output path (the geodesic diameter alone runs along the crossbar)."""

    mask = _t_mask()
    seed = (11, 19, 12)  # (k, j, i) branch tip
    spine = np.asarray(thinning_skeleton(mask, seed))
    distances = np.sqrt(((spine - np.asarray(seed)) ** 2).sum(axis=1))
    assert float(distances.min()) <= 1.0, "the seed was dropped from the path"
    assert int((spine[:, 1] >= 14).sum()) >= 3, "the seed branch is missing"
    assert int((spine[:, 2] >= 18).sum()) >= 1, "the crossbar far end is missing"


def test_unresolvable_seed_branch_abstains_with_not_analyzable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """F4 fallback: when even the seed-anchored path cannot carry the seed,
    the derivation abstains (typed NOT_ANALYZABLE), never silently discards."""

    monkeypatch.setattr(
        "core.coronary.thinning._seed_inside_spine_tube", lambda *args, **kwargs: False
    )
    with pytest.raises(CoronaryValidationError) as excinfo:
        thinning_skeleton(_t_mask(), (11, 19, 12))
    assert excinfo.value.code is CoronaryRefusalCode.NOT_ANALYZABLE
