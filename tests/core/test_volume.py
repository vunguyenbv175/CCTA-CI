"""Volume creation rules (Part B §3.3, H-009, I3, I8)."""

from __future__ import annotations

import pytest

from core.enums import VolumeKind
from core.errors import CanonicalValidationError
from core.volume import content_blob_path
from tests.core.factories import make_blob_ref, make_volume


class TestVolumeCreation:
    def test_canonical_volume_without_transform_ok(self):
        volume = make_volume()
        assert volume.kind == VolumeKind.CANONICAL
        assert volume.transform_ref is None
        assert volume.registration_ref is None

    @pytest.mark.parametrize("kind", [VolumeKind.RESAMPLED, VolumeKind.CROPPED])
    def test_non_canonical_requires_transform_ref(self, kind: VolumeKind):
        with pytest.raises(CanonicalValidationError) as err:
            make_volume(kind=kind)
        assert any("transform_ref" in v for v in err.value.violations)

    def test_non_canonical_with_transform_ref_accepted(self):
        volume = make_volume(
            kind=VolumeKind.RESAMPLED,
            transform_ref="transform-artifact-1",
        )
        assert volume.transform_ref == "transform-artifact-1"

    def test_multi_series_requires_registration(self):
        with pytest.raises(CanonicalValidationError) as err:
            make_volume(source_series_refs=("series-a", "series-b"))
        assert any("registration_ref" in v for v in err.value.violations)

    def test_multi_series_with_registration_accepted(self):
        volume = make_volume(
            source_series_refs=("series-a", "series-b"),
            registration_ref="reg-artifact-1",
            transform_ref="transform-artifact-2",
        )
        assert volume.registration_ref == "reg-artifact-1"

    def test_provenance_required(self):
        with pytest.raises(CanonicalValidationError):
            make_volume(provenance_ref="")

    def test_duplicate_series_refs_rejected(self):
        with pytest.raises(CanonicalValidationError) as err:
            make_volume(source_series_refs=("s", "s"))
        assert any("duplicates" in v for v in err.value.violations)

    def test_empty_study_ref_rejected(self):
        with pytest.raises(CanonicalValidationError):
            make_volume(study_ref="")

    @pytest.mark.parametrize(
        "blob_overrides",
        [
            {"sha256": "short"},
            {"codec": ""},
            {"byte_size": 0},
        ],
    )
    def test_blob_binding_validated(self, blob_overrides: dict):
        base: dict[str, object] = {"sha256": "b" * 64, "codec": "zarr.v2", "byte_size": 10}
        base.update(blob_overrides)
        with pytest.raises(CanonicalValidationError):
            make_volume(blob_ref=make_blob_ref(**base))

    def test_frame_validation_surfaces_through_factory(self):
        from tests.core.factories import make_frame

        with pytest.raises(CanonicalValidationError):
            make_volume(coordinate_frame=make_frame(dimensions=(0, 10, 10)))

    def test_volumes_are_immutable(self):
        volume = make_volume()
        with pytest.raises(AttributeError):  # frozen dataclass: no setter path (I13)
            volume.status = "SUPERSEDED"


def test_content_blob_path_layout():
    digest = "ab" + "0" * 62
    assert content_blob_path(digest) == f"blobs/sha256/ab/{digest}"
    with pytest.raises(ValueError):
        content_blob_path("zz")
