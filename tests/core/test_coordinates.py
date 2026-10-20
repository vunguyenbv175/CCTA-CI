"""Coordinate chain voxel<->world<->mesh<->centerline with round-trip hooks (§7)."""

from __future__ import annotations

import dataclasses
import math

import pytest

from core.coordinates import (
    CenterlineCoordinate,
    CoordinateTransform,
    MeshCoordinate,
    SpaceKind,
    VoxelCoordinate,
    WorldCoordinate,
    apply_matrix,
    assert_round_trip,
    identity_matrix,
    invert_matrix,
    matmul,
    round_trip_max_error,
    transform_for_frame,
)
from core.enums import Handedness, Units
from core.errors import CanonicalValidationError
from tests.core.factories import OBLIQUE_ORIENTATION, make_frame, make_transform


class TestFrameSpec:
    def test_valid_frame_accepted(self):
        frame = make_frame()
        assert frame.handedness.value == "LPS"

    def test_non_unit_orientation_rejected(self):
        with pytest.raises(CanonicalValidationError) as err:
            make_frame(image_orientation_patient=(2.0, 0.0, 0.0, 0.0, 1.0, 0.0))
        assert any("unit direction cosines" in v for v in err.value.violations)

    def test_short_orientation_rejected(self):
        with pytest.raises(CanonicalValidationError):
            make_frame(image_orientation_patient=(1.0, 0.0, 0.0))

    def test_zero_spacing_rejected(self):
        with pytest.raises(CanonicalValidationError):
            make_frame(spacing_mm=(0.5, 0.0, 0.6))

    def test_missing_frame_uid_rejected(self):
        with pytest.raises(CanonicalValidationError):
            make_frame(frame_of_reference_uid="")

    def test_oblique_orientation_supported(self):
        frame = make_frame(
            image_orientation_patient=OBLIQUE_ORIENTATION,
            spacing_mm=(0.35, 0.4, 0.9),
        )
        m = frame.voxel_to_world_matrix()
        # Column norms equal the spacing (orthonormal direction cosines).
        for col, expected in zip(range(3), (0.35, 0.4, 0.9)):
            norm = math.sqrt(sum(m[r][col] ** 2 for r in range(3)))
            assert norm == pytest.approx(expected)


class TestVoxelWorldRoundTrip:
    def test_identity_round_trip_on_oblique_grid(self):
        frame = make_frame(image_orientation_patient=OBLIQUE_ORIENTATION)
        forward = transform_for_frame(frame, series_instance_uid="1.2.840.series.1")
        backward = forward.inverse
        assert_round_trip(forward, backward)

    def test_voxel_center_maps_to_image_position(self):
        frame = make_frame()
        world = apply_matrix(frame.voxel_to_world_matrix(), VoxelCoordinate(0, 0, 0))
        assert world == pytest.approx(frame.image_position_patient)

    def test_anisotropic_spacing_respected(self):
        frame = make_frame()
        world = apply_matrix(frame.voxel_to_world_matrix(), VoxelCoordinate(2, 3, 4))
        assert world[0] == pytest.approx(-125.0 + 2 * 0.5)
        assert world[1] == pytest.approx(-120.4 + 3 * 0.45)
        assert world[2] == pytest.approx(42.0 + 4 * 0.6)

    def test_measured_error_persisted_in_artifact_payload(self):
        frame = make_frame()
        forward = transform_for_frame(frame, series_instance_uid="1.2.840.series.9")
        error = round_trip_max_error(forward, forward.inverse)
        sealed = dataclasses.replace(forward, round_trip_error_mm=error)
        assert sealed.round_trip_error_mm is not None
        assert sealed.round_trip_error_mm < 1e-9


class TestMeshAndCenterlineChain:
    def test_world_mesh_round_trip(self):
        t = make_transform()  # world -> mesh
        assert_round_trip(t, t.inverse)

    def test_full_chain_voxel_world_mesh_composes(self):
        frame = make_frame()
        voxel_to_world = transform_for_frame(frame, series_instance_uid="1.2.840.series.1")
        world_to_mesh = make_transform(series_instance_uid="1.2.840.series.1")
        chained = voxel_to_world.compose_with(world_to_mesh)
        assert chained.from_space == SpaceKind.VOXEL
        assert chained.to_space == SpaceKind.MESH
        inverse_chain = world_to_mesh.inverse.compose_with(voxel_to_world.inverse)
        assert_round_trip(chained, inverse_chain)

    def test_centerline_parametrization_transforms_to_world_and_back(self):
        # Straight-vessel centerline: s_mm along +x from origin; offset in y.
        scale = (
            (0.0, 1.0, 0.0, 10.0),
            (1.0, 0.0, 0.0, -5.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        )
        centerline_to_world = make_transform(
            transform_id="centerline-to-world",
            from_space=SpaceKind.CENTERLINE,
            to_space=SpaceKind.WORLD,
            method="straight-polyline",
            matrix=scale,
            direction_cosines=((0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
            translation=(10.0, -5.0, 0.0),
        )
        assert_round_trip(centerline_to_world, centerline_to_world.inverse)
        point = apply_matrix(scale, WorldCoordinate(12.5, 3.0, 0.0))  # (s, offset)
        assert point[0] == pytest.approx(3.0 + 10.0)  # x = offset + shift
        assert point[1] == pytest.approx(12.5 - 5.0)  # y = s - shift
        _ = CenterlineCoordinate(s_mm=12.5, offset_mm=3.0)

    def test_composition_requires_adjacent_spaces(self):
        frame = make_frame()
        voxel_to_world = transform_for_frame(frame, series_instance_uid="1.2.840.series.1")
        with pytest.raises(CanonicalValidationError):
            voxel_to_world.compose_with(
                transform_for_frame(frame, series_instance_uid="1.2.840.series.1")
            )


class TestTransformValidation:
    def test_same_space_rejected(self):
        with pytest.raises(CanonicalValidationError):
            make_transform(from_space=SpaceKind.WORLD, to_space=SpaceKind.WORLD)

    def test_singular_matrix_rejected(self):
        with pytest.raises(CanonicalValidationError):
            make_transform(
                matrix=(
                    (1.0, 0.0, 0.0, 0.0),
                    (0.0, 1.0, 0.0, 0.0),
                    (0.0, 0.0, 0.0, 0.0),  # rank deficient
                    (0.0, 0.0, 0.0, 0.0),
                )
            )

    def test_bad_shape_rejected(self):
        with pytest.raises(CanonicalValidationError):
            make_transform(matrix=((1.0, 0.0), (0.0, 1.0)))

    def test_version_must_be_positive(self):
        with pytest.raises(CanonicalValidationError):
            make_transform(version=0)

    def test_inverse_property_swaps_spaces(self):
        t = make_transform()
        inv = t.inverse
        assert (inv.from_space, inv.to_space) == (t.to_space, t.from_space)


class TestMandatoryGeometricDescriptors:
    """Part A §1.2: transforms carry explicit handedness, units, direction
    cosines (3x3), translation, FrameOfReferenceUID and SeriesInstanceUID."""

    def test_all_descriptors_present_on_valid_transform(self):
        t = make_transform()
        assert t.handedness == Handedness.LPS
        assert t.units == Units.MM
        assert t.frame_of_reference_uid == "1.2.840.for.1"
        assert t.series_instance_uid == "1.2.840.series.1"
        assert len(t.direction_cosines) == 3
        assert all(len(row) == 3 for row in t.direction_cosines)
        assert len(t.translation) == 3

    def test_missing_series_uid_rejected(self):
        with pytest.raises(CanonicalValidationError) as err:
            make_transform(series_instance_uid="")
        assert any("series_instance_uid" in v for v in err.value.violations)

    def test_transform_for_frame_requires_series_uid(self):
        with pytest.raises(CanonicalValidationError):
            transform_for_frame(make_frame())

    def test_non_unit_direction_cosine_row_rejected(self):
        with pytest.raises(CanonicalValidationError) as err:
            make_transform(direction_cosines=((2.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))
        assert any("unit direction cosine" in v for v in err.value.violations)

    def test_non_orthogonal_direction_cosines_rejected(self):
        cosines = (
            (1.0, 0.0, 0.0),
            (math.cos(0.5), math.sin(0.5), 0.0),
            (0.0, 0.0, 1.0),
        )
        with pytest.raises(CanonicalValidationError) as err:
            make_transform(direction_cosines=cosines)
        assert any("orthogonal" in v for v in err.value.violations)

    def test_wrong_shape_direction_cosines_rejected(self):
        with pytest.raises(CanonicalValidationError) as err:
            make_transform(direction_cosines=((1.0, 0.0), (0.0, 1.0)))
        assert any("3x3" in v for v in err.value.violations)

    def test_translation_matrix_mismatch_rejected(self):
        with pytest.raises(CanonicalValidationError) as err:
            make_transform(translation=(99.0, -2.5, 3.25))
        assert any("translation vector must match" in v for v in err.value.violations)

    def test_nan_translation_rejected(self):
        with pytest.raises(CanonicalValidationError) as err:
            make_transform(
                translation=(float("nan"), -2.5, 3.25),
                matrix=(
                    (1.0, 0.0, 0.0, float("nan")),
                    (0.0, 0.0, 1.0, -2.5),
                    (0.0, -1.0, 0.0, 3.25),
                    (0.0, 0.0, 0.0, 1.0),
                ),
            )
        assert any("translation" in v for v in err.value.violations)

    def test_hu_units_rejected_for_spatial_transform(self):
        with pytest.raises(CanonicalValidationError) as err:
            make_transform(units=Units.HU)
        assert any("millimetre" in v for v in err.value.violations)

    def test_explicit_handedness_required_type(self):
        with pytest.raises(CanonicalValidationError) as err:
            make_transform(handedness="LPS")
        assert any("handedness" in v for v in err.value.violations)

    def test_inverse_carries_descriptors_with_transposed_cosines(self):
        t = make_transform()
        inv = t.inverse
        assert inv.frame_of_reference_uid == t.frame_of_reference_uid
        assert inv.series_instance_uid == t.series_instance_uid
        assert inv.handedness == t.handedness
        assert inv.units == t.units
        expected = tuple(
            tuple(t.direction_cosines[r][c] for r in range(3)) for c in range(3)
        )
        assert inv.direction_cosines == expected

    def test_compose_propagates_descriptors(self):
        frame = make_frame()
        voxel_to_world = transform_for_frame(frame, series_instance_uid="1.2.840.series.2")
        world_to_mesh = make_transform(series_instance_uid="1.2.840.series.2")
        chained = voxel_to_world.compose_with(world_to_mesh)
        assert chained.series_instance_uid == "1.2.840.series.2"
        assert chained.frame_of_reference_uid == frame.frame_of_reference_uid
        assert chained.handedness == Handedness.LPS
        assert chained.units == Units.MM

    def test_transform_for_frame_populates_descriptors_from_frame(self):
        frame = make_frame(image_orientation_patient=OBLIQUE_ORIENTATION)
        t = transform_for_frame(frame, series_instance_uid="1.2.840.series.3")
        row = frame.image_orientation_patient[:3]
        col = frame.image_orientation_patient[3:]
        normal = (
            row[1] * col[2] - row[2] * col[1],
            row[2] * col[0] - row[0] * col[2],
            row[0] * col[1] - row[1] * col[0],
        )
        assert t.direction_cosines == (row, col, normal)
        assert t.translation == frame.image_position_patient


class TestAffineHelpers:
    def test_matmul_identity(self):
        a = make_transform().matrix
        product = matmul(a, identity_matrix())
        assert product == a

    def test_double_inverse_is_identity(self):
        m = invert_matrix(make_transform().matrix)
        twice = matmul(m, invert_matrix(m))
        for row in twice:
            pass
        identity = identity_matrix()
        for i in range(4):
            for j in range(4):
                assert twice[i][j] == pytest.approx(identity[i][j])

    def test_finite_coordinate_enforcement(self):
        with pytest.raises(ValueError):
            WorldCoordinate(float("nan"), 0.0, 0.0)
        with pytest.raises(ValueError):
            MeshCoordinate(0.0, float("inf"), 0.0)
        with pytest.raises(ValueError):
            CenterlineCoordinate(s_mm=-1.0)

    def test_round_trip_space_mismatch_detected(self):
        voxel_to_world = transform_for_frame(make_frame(), series_instance_uid="1.2.840.series.1")
        mesh_transform = make_transform()
        with pytest.raises(CanonicalValidationError):
            round_trip_max_error(voxel_to_world, mesh_transform)

    def test_assert_round_trip_failure_path(self):
        bad_forward = make_transform(method="lossy")
        sloppy_backward = make_transform(
            transform_id="bad-inv",
            from_space=SpaceKind.MESH,
            to_space=SpaceKind.WORLD,
            method="wrong-matrix",
            matrix=(
                (1.0, 0.0, 0.0, 999.0),
                (0.0, 1.0, 0.0, 0.0),
                (0.0, 0.0, 1.0, 0.0),
                (0.0, 0.0, 0.0, 1.0),
            ),
            direction_cosines=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            translation=(999.0, 0.0, 0.0),
        )
        with pytest.raises(CanonicalValidationError):
            assert_round_trip(bad_forward, sloppy_backward)


class TestB8MandatoryDescriptorsNoDefaults:
    """Blocker b8: handedness/units are mandatory with NO silent defaults -
    omitting either must fail loudly (TypeError) instead of becoming LPS/MM."""

    @staticmethod
    def _kwargs_without(*drop: str) -> dict[str, object]:
        base = make_transform()
        return {
            f.name: getattr(base, f.name)
            for f in dataclasses.fields(CoordinateTransform)
            if f.name not in drop
        }

    def test_b8_omitting_handedness_is_type_error(self):
        with pytest.raises(TypeError):
            CoordinateTransform(**self._kwargs_without("handedness"))  # type: ignore[arg-type]

    def test_b8_omitting_units_is_type_error(self):
        with pytest.raises(TypeError):
            CoordinateTransform(**self._kwargs_without("units"))  # type: ignore[arg-type]

    def test_b8_descriptor_fields_carry_no_dataclass_default(self):
        defaults = {
            f.name: f.default
            for f in dataclasses.fields(CoordinateTransform)
            if f.name in ("handedness", "units")
        }
        assert defaults["handedness"] is dataclasses.MISSING
        assert defaults["units"] is dataclasses.MISSING

    def test_b8_explicit_descriptors_still_build_valid_transforms(self):
        t = make_transform(handedness=Handedness.LPS, units=Units.MM)
        assert t.handedness == Handedness.LPS
        assert t.units == Units.MM
        assert_round_trip(t, t.inverse)

    def test_b8_transform_for_frame_propagates_explicit_descriptors(self):
        # The frame's mandatory descriptors flow explicitly into the derived
        # transform - no silent LPS/MM substitution anywhere in the chain.
        frame = make_frame()
        t = transform_for_frame(frame, series_instance_uid="1.2.840.series.1")
        assert t.handedness is frame.handedness
        assert t.units is frame.units
