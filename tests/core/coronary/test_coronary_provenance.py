"""T-5: full §44 provenance over the slice graph (``verify_chain`` green).

The vessel mask is derivation EVIDENCE: its bytes are carried as digests in
the centerline payload and in the §44 parameters blob - no SEGMENTATION
artifact is ever written (the producer matrix admits no deterministic
segmentation producer).
"""

from __future__ import annotations

import numpy as np

from core.coronary import (
    CORONARY_ALGORITHM,
    CORONARY_ALGORITHM_VERSION,
    MASK_EVIDENCE_CODEC,
    Seed,
    centerline_provenance,
)
from core.digest import canonical_digest
from core.enums import ArtifactKind, ProducerType
from core.provenance import BlobRef, verify_chain
from tests.core.coronary.support import ULIDS, SliceFixture, helix_points
from visualization.cpr import CPR_ALGORITHM, CPR_ALGORITHM_VERSION


def test_verify_chain_is_green_over_the_slice_graph(slice_fixture: SliceFixture) -> None:
    records = (
        slice_fixture.volume_provenance,
        slice_fixture.transform_provenance,
        slice_fixture.centerline_provenance,
        slice_fixture.cpr_provenance,
    )
    ok, problems = verify_chain(records)
    assert ok, problems
    assert problems == []


def test_every_record_carries_the_nine_items(slice_fixture: SliceFixture) -> None:
    for record in (
        slice_fixture.volume_provenance,
        slice_fixture.transform_provenance,
        slice_fixture.centerline_provenance,
        slice_fixture.cpr_provenance,
    ):
        assert record.subject_ref
        assert record.preprocessing
        assert record.parameters_ref is not None
        assert record.human_edit is not None
        assert record.environment.software_version
        assert record.environment.code_commit_sha == slice_fixture.code_commit_sha
        assert record.created_by
        assert record.at


def test_centerline_artifact_is_deterministic_engine_and_geometry_bound(
    slice_fixture: SliceFixture,
) -> None:
    artifact = slice_fixture.centerline.artifact
    assert artifact.kind is ArtifactKind.CENTERLINE
    assert artifact.producer.producer_type is ProducerType.DETERMINISTIC_ENGINE
    assert artifact.producer.algorithm == CORONARY_ALGORITHM
    assert artifact.producer.algorithm_version == CORONARY_ALGORITHM_VERSION
    geometry = artifact.lineage.geometry_base
    assert geometry is not None
    assert geometry.geometry_id == ULIDS["transform_artifact"]
    assert geometry.geometry_version == 1
    assert artifact.lineage.input_volume_refs == (slice_fixture.volume.volume_id,)
    assert artifact.lineage.input_artifact_refs == (ULIDS["transform_artifact"],)


def test_cpr_view_is_an_mpr_view_framed_on_the_centerline(
    slice_fixture: SliceFixture,
) -> None:
    cpr = slice_fixture.cpr
    assert cpr.artifact.kind is ArtifactKind.MPR_VIEW
    assert cpr.artifact.producer.producer_type is ProducerType.DETERMINISTIC_ENGINE
    assert cpr.artifact.producer.algorithm == CPR_ALGORITHM
    assert cpr.artifact.producer.algorithm_version == CPR_ALGORITHM_VERSION
    assert cpr.artifact.payload.payload_schema == "ccta.cpr.view.v1"
    centerline_id = slice_fixture.centerline.artifact.artifact_id
    geometry = cpr.artifact.lineage.geometry_base
    assert geometry is not None
    assert geometry.geometry_id == centerline_id
    assert geometry.geometry_version == 1
    assert cpr.artifact.lineage.input_artifact_refs == (centerline_id,)
    assert cpr.artifact.lineage.input_volume_refs == (slice_fixture.volume.volume_id,)
    payload = cpr.payload
    assert payload["centerline_ref"] == f"centerline_artifact_id:{centerline_id}"
    assert (
        payload["centerline_payload_sha256"]
        == slice_fixture.centerline.artifact.payload.blob_sha256
    )


def test_mask_bytes_never_become_a_segmentation_artifact(slice_fixture: SliceFixture) -> None:
    payload = slice_fixture.centerline.payload
    diagnostics = slice_fixture.centerline.path.diagnostics
    assert payload["vessel_mask_sha256"] == diagnostics.mask_sha256
    assert payload["vesselness_response_sha256"] == diagnostics.response_sha256
    assert payload["vessel_mask_voxel_count"] == diagnostics.mask_voxel_count
    parameters_ref = slice_fixture.centerline_provenance.parameters_ref
    assert isinstance(parameters_ref, BlobRef)
    assert parameters_ref.blob_ref.startswith("coronary-centerline-parameters:")
    for artifact in (slice_fixture.centerline.artifact, slice_fixture.cpr.artifact):
        assert artifact.kind is not ArtifactKind.SEGMENTATION


def test_single_path_label_uses_the_frozen_vessel_tokens(slice_fixture: SliceFixture) -> None:
    from core.coronary import VESSEL_LABELS

    label = slice_fixture.centerline.payload["vessel_label"]
    assert label == "VESSEL:RCA"
    assert label in VESSEL_LABELS


# ---------------------------------------------------------------------------
# att0002 corrections: F7 (seed input + retained mask-evidence bytes)
# ---------------------------------------------------------------------------


def test_seed_is_bound_as_a_derivation_input(slice_fixture: SliceFixture) -> None:
    """F7: the typed human seed is an explicit digest-bound INPUT of the
    derivation, so two different seeds can never share identical provenance."""

    seed_sha256 = canonical_digest(slice_fixture.seed.to_document())
    seed_refs = [
        ref
        for ref in slice_fixture.centerline_provenance.inputs  # type: ignore[union-attr]
        if ref.kind == "seed"
    ]
    assert len(seed_refs) == 1
    assert seed_refs[0].ref == f"seed:{seed_sha256}"
    assert seed_refs[0].sha256_at_use == seed_sha256
    assert seed_sha256 in slice_fixture.centerline_provenance.parameters_ref.blob_ref  # type: ignore[union-attr]

    helix = helix_points()
    probe = helix[80]
    other_seed = Seed(
        i=int(round(probe[0])),
        j=int(round(probe[1])),
        k=int(round(probe[2])),
        direction=(
            float(helix[100][0] - helix[60][0]),
            float(helix[100][1] - helix[60][1]),
            float(helix[100][2] - helix[60][2]),
        ),
    )
    other_record = centerline_provenance(
        subject_ref=f"centerline_artifact_id:{ULIDS['centerline_artifact']}",
        volume=slice_fixture.volume,
        seed=other_seed,
        parameters=slice_fixture.parameters,
        mask_sha256=slice_fixture.centerline.path.diagnostics.mask_sha256,
        mask_blob_sha256=str(
            slice_fixture.centerline.payload["vessel_mask_blob_ref"]["sha256"]  # type: ignore[index]
        ),
        response_sha256=slice_fixture.centerline.path.diagnostics.response_sha256,
        transform_artifact_ref=f"coordinate_transform:{ULIDS['transform_artifact']}",
        transform_digest=canonical_digest(
            {
                "transform_id": slice_fixture.transform.transform_id,
                "version": slice_fixture.transform.version,
            }
        ),
        volume_provenance_ref=slice_fixture.volume_provenance.provenance_id,
        transform_provenance_ref=slice_fixture.transform_provenance.provenance_id,
        software_version="ccta-coronary-fixture",
        code_commit_sha=slice_fixture.code_commit_sha,
        runtime="python-3.12 (coronary fixture)",
        created_by="ccta-coronary-fixture",
        at="2026-09-14T00:00:00Z",
        provenance_id="01HZZZZZZZZZZZZZZZZZZZZZ19",
    )
    assert other_record.inputs != slice_fixture.centerline_provenance.inputs
    assert other_record.parameters_ref != slice_fixture.centerline_provenance.parameters_ref


def test_mask_evidence_bytes_are_retained_and_resolvable(
    slice_fixture: SliceFixture,
) -> None:
    """F7: the derivation mask bytes are retained as a resolvable immutable
    content-addressed blob (never only digests, never a SEGMENTATION write)."""

    payload = slice_fixture.centerline.payload
    blob_ref = payload["vessel_mask_blob_ref"]
    assert isinstance(blob_ref, dict)
    assert blob_ref["codec"] == MASK_EVIDENCE_CODEC
    data = slice_fixture.loader.read_blob(str(blob_ref["sha256"]))
    assert len(data) == int(blob_ref["byte_size"])
    decoded = np.frombuffer(data, dtype=np.uint8).reshape(
        slice_fixture.centerline.path.mask.shape
    )
    assert np.array_equal(decoded.astype(bool), slice_fixture.centerline.path.mask)
    assert int(blob_ref["byte_size"]) == int(slice_fixture.centerline.path.mask.size)
    assert (
        str(blob_ref["sha256"])
        in slice_fixture.centerline_provenance.parameters_ref.blob_ref  # type: ignore[union-attr]
    )
    assert slice_fixture.centerline.artifact.kind is not ArtifactKind.SEGMENTATION
    assert slice_fixture.cpr.artifact.kind is not ArtifactKind.SEGMENTATION
