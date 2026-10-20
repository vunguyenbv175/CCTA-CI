"""QC_001-D test-driver adapters and an independent fixture oracle.

The fixture module is intentionally the only source of injected answers.  This
support module merely turns its reader/materializer output into the production
DTOs used by the integration tests.
"""

from __future__ import annotations

import hashlib
from typing import Final

import numpy as np

from core.artifacts import DerivedArtifactStore, GeometryBase
from core.digest import canonical_digest
from core.qc import (
    CoverageEnvelope,
    QCInput,
    QCResultBundle,
    RegisteredPairInput,
    create_real_qc_assessment,
    create_registered_pair_input,
)
from tests.apps.qc_001_d_fixtures import PreparedQCInput, ulid

FIXED_AT: Final[str] = "2026-01-01T00:00:00Z"
ZERO_SHA1: Final[str] = "0" * 40


def _readonly_c_array(array: np.ndarray) -> np.ndarray:
    result = np.array(array, copy=True, order="C")
    result.setflags(write=False)
    return result


def qc_input(prepared: PreparedQCInput) -> QCInput:
    """Build a QCInput from the canonical reader/materializer result."""

    pixels = prepared.case.geometry
    assert pixels is not None
    is_rescaled = pixels.pixels == "rescaled-fractional"
    return QCInput(
        volume=prepared.first_volume,
        array=prepared.first_array,
        source_manifest_sha256=prepared.source_manifest_sha256,
        instances=prepared.first_instances,
        instance_digests={
            instance.sop_instance_uid: hashlib.sha256(instance.pixel_data).hexdigest()
            for instance in prepared.first_instances
        },
        scalar_domain="HU" if is_rescaled else "STORED",
        intensity_unit="HU" if is_rescaled else "STORED",
        transform=prepared.first_transform,
        transform_ref=prepared.first_transform_ref,
        transform_sha256=prepared.first_transform_sha256,
        transform_provenance_ref=prepared.first_transform_provenance_ref,
        geometry_base=GeometryBase(f"volume_geometry:{prepared.first_volume.volume_id}", 1),
        coverage=CoverageEnvelope(anchor_world_mm=prepared.coverage_anchor),
    )


def registered_pair(
    prepared: PreparedQCInput,
    *,
    affine: tuple[float, ...] = (
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ),
    overlap_lower: tuple[float, float, float] = (-10_000.0, -10_000.0, -10_000.0),
    overlap_upper: tuple[float, float, float] = (10_000.0, 10_000.0, 10_000.0),
    registration_version: int = 1,
    first_geometry_version: int = 1,
    second_geometry_version: int = 1,
    registration_id_seed: int | None = None,
) -> RegisteredPairInput:
    """Build the explicit registration for a prepared two-phase case."""

    first = _readonly_c_array(prepared.first_array)
    second_array = prepared.second_array
    second_volume = prepared.second_volume
    if second_array is None or second_volume is None:
        raise ValueError("registered pair requires a prepared second volume")
    second = _readonly_c_array(second_array)
    ordinal = prepared.case.ordinal
    registration_seed = (
        2_000_000 + ordinal * 2
        if registration_id_seed is None
        else registration_id_seed
    )
    return create_registered_pair_input(
        first_volume=prepared.first_volume,
        second_volume=second_volume,
        first_array=first,
        second_array=second,
        first_phase_ref="phase:BASE",
        second_phase_ref="phase:FOLLOW",
        affine_4x4_row_major=affine,
        overlap_lower_lps_mm=overlap_lower,
        overlap_upper_lps_mm=overlap_upper,
        registration_version=registration_version,
        first_geometry_version=first_geometry_version,
        second_geometry_version=second_geometry_version,
        registration_artifact_id=ulid(registration_seed),
        registration_provenance_id=ulid(registration_seed + 1),
        created_at=FIXED_AT,
        code_commit_sha=ZERO_SHA1,
    ).pair


def real_bundle(
    prepared: PreparedQCInput,
    *,
    id_seed: int = 3_000_000,
    pair: RegisteredPairInput | None = None,
    supersedes_ref: str | None = None,
    artifact_store: DerivedArtifactStore | None = None,
) -> QCResultBundle:
    """Create one deterministic bundle with stable IDs for byte comparisons."""

    source = qc_input(prepared)
    selected_pair = registered_pair(prepared) if pair is None else pair
    family_ids = tuple(ulid(id_seed + 2 + index) for index in range(9))
    return create_real_qc_assessment(
        source,
        pair=selected_pair,
        coverage=source.coverage,
        provenance_records=selected_pair.provenance_records,
        artifact_id=ulid(id_seed),
        provenance_id=ulid(id_seed + 1),
        family_provenance_ids=family_ids,
        supersedes_ref=supersedes_ref,
        artifact_store=artifact_store,
        created_at=FIXED_AT,
        code_commit_sha=ZERO_SHA1,
    )


def bundle_digest_view(bundle: QCResultBundle) -> tuple[str, str, str, str]:
    """Return the exact content-derived identity/evidence digest tuple."""

    evidence_digest = canonical_digest(
        {
            ref: bundle.family_evidence_documents[ref]
            for ref in sorted(bundle.family_evidence_documents)
        }
    )
    return (
        canonical_digest(bundle.payload_document),
        canonical_digest(bundle.parameter_manifest),
        evidence_digest,
        canonical_digest(bundle.family_result_body),
    )


__all__ = [
    "FIXED_AT",
    "ZERO_SHA1",
    "bundle_digest_view",
    "qc_input",
    "real_bundle",
    "registered_pair",
]
