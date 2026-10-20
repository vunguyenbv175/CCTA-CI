"""T-4: typed refusals with exact codes.

Every refusal is fail-closed and typed (never a fabricated path): empty
vesselness and a seed without vessel evidence abstain with ``NOT_ANALYZABLE``,
a seed outside the volume refuses with ``SEED_OUT_OF_BOUNDS``, and the non-HU
scalar domain defers to the EXISTING F18 viewer gate
(``apps.viewer.frames.NonHUScalarDomainError``) - the coronary module mints no
domain refusal of its own.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from apps.viewer.frames import NonHUScalarDomainError, build_geometry_lists
from apps.workstation.volume_loader import (
    CANONICAL_INT16_CODEC,
    CanonicalVolumeLoader,
)
from core.artifacts import DerivedArtifactStore, InvalidationDAG
from core.coordinates import FrameSpec, transform_for_frame
from core.coronary import (
    CoronaryRefusalCode,
    CoronaryValidationError,
    Seed,
    VesselnessParameters,
    derive_coronary_path,
)
from core.enums import ActorType, ReleaseLane, ReviewState, VolumeKind
from core.events import EventLog
from core.volume import VolumeBlobRef, create_volume
from tests.core.coronary.support import VOXEL_MM, SliceFixture
from visualization.mpr import build_mpr

_SECOND_AT = "2026-09-14T00:00:00Z"


def test_empty_vesselness_abstains_not_analyzable(slice_fixture: SliceFixture) -> None:
    constant = np.full_like(slice_fixture.raw, -100)
    seed = Seed(i=24, j=24, k=24, direction=(1.0, 0.0, 0.0))
    with pytest.raises(CoronaryValidationError) as excinfo:
        derive_coronary_path(
            constant,
            (VOXEL_MM, VOXEL_MM, VOXEL_MM),
            slice_fixture.frame,
            seed,
            slice_fixture.parameters,
        )
    assert excinfo.value.code is CoronaryRefusalCode.NOT_ANALYZABLE
    assert "empty" in excinfo.value.detail


def test_seed_without_vessel_evidence_abstains(slice_fixture: SliceFixture) -> None:
    seed = Seed(i=2, j=2, k=2, direction=(1.0, 0.0, 0.0))
    with pytest.raises(CoronaryValidationError) as excinfo:
        derive_coronary_path(
            slice_fixture.raw,
            (VOXEL_MM, VOXEL_MM, VOXEL_MM),
            slice_fixture.frame,
            seed,
            slice_fixture.parameters,
        )
    assert excinfo.value.code is CoronaryRefusalCode.NOT_ANALYZABLE
    assert "no vessel evidence" in excinfo.value.detail


def test_seed_outside_the_volume_refuses(slice_fixture: SliceFixture) -> None:
    seed = Seed(i=1000, j=24, k=24, direction=(1.0, 0.0, 0.0))
    with pytest.raises(CoronaryValidationError) as excinfo:
        derive_coronary_path(
            slice_fixture.raw,
            (VOXEL_MM, VOXEL_MM, VOXEL_MM),
            slice_fixture.frame,
            seed,
            slice_fixture.parameters,
        )
    assert excinfo.value.code is CoronaryRefusalCode.SEED_OUT_OF_BOUNDS


def test_zero_direction_seed_refuses_at_construction() -> None:
    with pytest.raises(CoronaryValidationError) as excinfo:
        Seed(i=1, j=1, k=1, direction=(0.0, 0.0, 0.0))
    assert excinfo.value.code is CoronaryRefusalCode.SEED_DIRECTION_INVALID


def test_parameters_are_closed(slice_fixture: SliceFixture) -> None:
    with pytest.raises(CoronaryValidationError) as excinfo:
        VesselnessParameters(scales_mm=(), threshold=0.1)
    assert excinfo.value.code is CoronaryRefusalCode.PARAMETERS_INVALID


def test_coronary_vocabulary_mints_no_domain_refusal() -> None:
    """The closed vocabulary defers non-HU to the existing F18 gate."""

    assert {code.value for code in CoronaryRefusalCode} == {
        "NOT_ANALYZABLE",
        "SEED_OUT_OF_BOUNDS",
        "SEED_DIRECTION_INVALID",
        "VOLUME_INVALID",
        "PARAMETERS_INVALID",
        "NONFINITE_RESPONSE",
    }


# ---------------------------------------------------------------------------
# F18: the non-HU refusal is the EXISTING viewer gate
# ---------------------------------------------------------------------------


def _small_frame(dimensions: tuple[int, int, int] = (8, 8, 8)) -> FrameSpec:
    return FrameSpec(
        frame_of_reference_uid="1.2.826.0.1.3680043.8.498.77.910",
        image_position_patient=(0.0, 0.0, 0.0),
        image_orientation_patient=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        spacing_mm=(1.0, 1.0, 1.0),
        dimensions=dimensions,
    )


def _domain_session(tmp_path: Path, *, scalar_domain: str):
    """Build one canonical volume + one accepted MPR frame of that domain."""

    loader = CanonicalVolumeLoader(tmp_path / f"blobs-{scalar_domain}")
    raw = (
        np.full((8, 8, 8), 300, dtype=np.int16)
        if scalar_domain == "HU"
        else np.full((8, 8, 8), 1324, dtype=np.int16)
    )
    codec = CANONICAL_INT16_CODEC
    materialized = loader.materialize_canonical(
        raw, scalar_domain=scalar_domain, intensity_unit=scalar_domain
    )
    frame = _small_frame()
    volume = create_volume(
        study_ref="study_id:01HZZZZZZZZZZZZZZZZZZZZZ20",
        source_series_refs=("series:01HZZZZZZZZZZZZZZZZZZZZZ21",),
        kind=VolumeKind.CANONICAL,
        coordinate_frame=frame,
        blob_ref=VolumeBlobRef(
            sha256=materialized.blob.sha256,
            codec=codec,
            byte_size=materialized.blob.byte_size,
        ),
        provenance_ref="provenance:01HZZZZZZZZZZZZZZZZZZZZZ22",
        release_lane=ReleaseLane.PERSONAL_RESEARCH,
        volume_id="01HZZZZZZZZZZZZZZZZZZZZZ23",
    )
    transform = transform_for_frame(
        frame, series_instance_uid="1.2.826.0.1.3680043.8.498.77.911"
    )
    result = build_mpr(
        volume,
        transform,
        plane="AXIAL",
        origin=(0.0, 0.0, 0.0),
        in_plane_spacing_mm=(1.0, 1.0),
        matrix_size=(4, 4),
        method="NEAREST",
        out_of_bounds_fill_value=0.0,
        loader=loader,
        transform_artifact_ref="coordinate_transform:01HZZZZZZZZZZZZZZZZZZZZZ24",
        provenance_ref="provenance:01HZZZZZZZZZZZZZZZZZZZZZ25",
        study_instance_uid="1.2.826.0.1.3680043.8.498.77.912",
        source_array=raw,
        artifact_id="01HZZZZZZZZZZZZZZZZZZZZZ26",
        created_by="ccta-coronary-fixture",
    )
    store = DerivedArtifactStore(EventLog(), InvalidationDAG())
    store.add(result.artifact)
    accepted = store.apply_review_transition(
        result.artifact.artifact_id,
        ReviewState.ACCEPTED,
        actor_role=ActorType.HUMAN,
        reviewed_by="coronary-fixture-acceptor",
        evidence_ref="F18 domain probe",
        at=_SECOND_AT,
    )
    return volume, accepted, result.payload


def test_non_hu_domain_refuses_via_the_existing_f18_gate(tmp_path: Path) -> None:
    volume, artifact, payload = _domain_session(tmp_path, scalar_domain="STORED")
    with pytest.raises(NonHUScalarDomainError):
        build_geometry_lists(
            session_id="01HZZZZZZZZZZZZZZZZZZZZZ27",
            artifacts=(artifact,),
            payload_documents={artifact.artifact_id: payload},
            volume_ids=(volume.volume_id,),
            volume_scalar_domains={volume.volume_id: "STORED"},
        )


def test_hu_domain_passes_the_same_gate(tmp_path: Path) -> None:
    volume, artifact, payload = _domain_session(tmp_path, scalar_domain="HU")
    lists = build_geometry_lists(
        session_id="01HZZZZZZZZZZZZZZZZZZZZZ28",
        artifacts=(artifact,),
        payload_documents={artifact.artifact_id: payload},
        volume_ids=(volume.volume_id,),
        volume_scalar_domains={volume.volume_id: "HU"},
    )
    assert len(lists["AXIAL"]) == 1
    assert lists["AXIAL"][0].mpr_artifact_id == artifact.artifact_id
