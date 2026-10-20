"""The 001-C reviewer follow-up for MPR producer legality."""

from __future__ import annotations

import pytest

from core.artifacts import ClaimBinding, Lineage, PayloadRef, Producer, create_derived_artifact
from core.enums import ArtifactKind, ProducerType, ReleaseLane
from core.errors import CanonicalValidationError


def test_forged_non_deterministic_mpr_view_is_rejected_at_creation() -> None:
    with pytest.raises(CanonicalValidationError, match="MPR_VIEW"):
        create_derived_artifact(
            study_ref="study-001-c",
            kind=ArtifactKind.MPR_VIEW,
            payload=PayloadRef(
                payload_schema="ccta.mpr.view.v1",
                blob_sha256="a" * 64,
            ),
            lineage=Lineage(),
            claim_binding=ClaimBinding(claim_id="CLM-VIZ-001", claim_version="v0-draft"),
            producer=Producer(
                producer_type=ProducerType.MODEL,
                model_record_ref="forged-mpr-model@1.0.0",
            ),
            release_lane=ReleaseLane.PERSONAL_RESEARCH,
            provenance_ref="provenance-forged-mpr",
            created_by="forged-test",
            created_at="2026-01-01T00:00:00Z",
            artifact_id="forged-mpr-view",
        )


def test_deterministic_mpr_view_v2_is_accepted_at_creation() -> None:
    artifact = create_derived_artifact(
        study_ref="study-001-c",
        kind=ArtifactKind.MPR_VIEW,
        payload=PayloadRef(
            payload_schema="ccta.mpr.view.v2",
            blob_sha256="a" * 64,
        ),
        lineage=Lineage(),
        claim_binding=ClaimBinding(claim_id="CLM-VIZ-001", claim_version="v0-draft"),
        producer=Producer(
            producer_type=ProducerType.DETERMINISTIC_ENGINE,
            algorithm="mpr-engine",
            algorithm_version="2.0",
            parameters_blob_sha256="b" * 64,
        ),
        release_lane=ReleaseLane.PERSONAL_RESEARCH,
        provenance_ref="provenance-mpr-v2",
        created_by="deterministic-test",
        created_at="2026-01-01T00:00:00Z",
        artifact_id="mpr-v2-deterministic",
    )

    assert artifact.kind is ArtifactKind.MPR_VIEW
    assert artifact.payload.payload_schema == "ccta.mpr.view.v2"
    assert artifact.producer.producer_type is ProducerType.DETERMINISTIC_ENGINE


def test_forged_non_deterministic_mpr_view_v2_is_rejected_at_creation() -> None:
    with pytest.raises(CanonicalValidationError, match="MPR_VIEW"):
        create_derived_artifact(
            study_ref="study-001-c",
            kind=ArtifactKind.MPR_VIEW,
            payload=PayloadRef(
                payload_schema="ccta.mpr.view.v2",
                blob_sha256="a" * 64,
            ),
            lineage=Lineage(),
            claim_binding=ClaimBinding(claim_id="CLM-VIZ-001", claim_version="v0-draft"),
            producer=Producer(
                producer_type=ProducerType.MODEL,
                model_record_ref="forged-mpr-model@1.0.0",
            ),
            release_lane=ReleaseLane.PERSONAL_RESEARCH,
            provenance_ref="provenance-forged-mpr-v2",
            created_by="forged-test-v2",
            created_at="2026-01-01T00:00:00Z",
            artifact_id="forged-mpr-view-v2",
        )
