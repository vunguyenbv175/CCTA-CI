"""QC_001-D artifact, provenance, supersession, and real-consumer policy tests."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from core.artifacts import DerivedArtifactStore
from core.dicom.qc import create_qc_assessment as create_legacy_placeholder
from core.dicom.qc import qc_consumer_gate
from core.digest import canonical_digest
from core.enums import ArtifactStatus, QualityGrade
from core.events import EventLog
from core.provenance import InvalidationDAG, NotApplicable
from core.qc import (
    F2_5_EVIDENCE_WORDING,
    PairGeometryTracker,
    QCValidationError,
    create_qc_input_refusal,
    real_qc_consumer_gate,
    validate_grade_flag_matrix,
    validate_result_bundle,
)
from tests.apps.qc_001_d_fixtures import CensusCase, census_cases, prepare_valid_case, ulid
from tests.apps.qc_001_d_property_oracle import handedness_preserving
from tests.apps.qc_001_d_support import (
    FIXED_AT,
    bundle_digest_view,
    real_bundle,
    registered_pair,
)


def _clean_case() -> CensusCase:
    return next(
        case
        for case in census_cases()
        if case.valid
        and case.injection is not None
        and case.injection.flag is None
        and case.geometry is not None
        and case.geometry.pixels == "unsigned"
    )


def test_real_factory_emits_one_identity_two_views_and_complete_evidence() -> None:
    case = _clean_case()
    with TemporaryDirectory(prefix="qc-factory-") as directory:
        bundle = real_bundle(prepare_valid_case(Path(directory), case))
    assert validate_result_bundle(bundle)
    assert F2_5_EVIDENCE_WORDING == (
        "standards-conformant synthetic DICOM fixtures; explicitly NOT a real-scanner "
        "coverage claim; real-scanner validation = separate later obligation A1."
    )
    assert bundle.assessment.artifact_ref == f"qc_artifact:{bundle.artifact.artifact_id}"
    assert bundle.artifact.kind.value == "QC_ASSESSMENT"
    assert bundle.artifact.producer.producer_type.value == "DETERMINISTIC_ENGINE"
    assert bundle.assessment.payload.justification_code == "RULE_ENGINE_EVIDENCE"
    assert len(bundle.payload_document) == 13
    assert bundle.assessment.payload.evidence_refs[0] == bundle.artifact.provenance_ref
    assert bundle.assessment.payload.evidence_refs[1].startswith("source_manifest:")
    assert len(bundle.family_evidence_documents) == 9
    assert all(
        not isinstance(record.inputs, NotApplicable)
        and not isinstance(record.upstream_refs, NotApplicable)
        and not isinstance(record.parameters_ref, NotApplicable)
        for record in bundle.provenance_records
        if record.provenance_id == bundle.artifact.provenance_ref.removeprefix("provenance:")
        or record.provenance_id
        in {ref.split(":", 1)[1] for ref in bundle.assessment.payload.evidence_refs[4:]}
    )


def test_repeatability_is_byte_identical_with_fixed_identity_inputs() -> None:
    case = _clean_case()
    with TemporaryDirectory(prefix="qc-repeat-") as left, TemporaryDirectory(
        prefix="qc-repeat-"
    ) as right:
        first = real_bundle(prepare_valid_case(Path(left), case), id_seed=3_500_000)
        second = real_bundle(prepare_valid_case(Path(right), case), id_seed=3_500_000)
    assert bundle_digest_view(first) == bundle_digest_view(second)
    assert first.assessment == second.assessment
    assert first.artifact == second.artifact
    assert first.family_results == second.family_results
    assert first.family_evidence_documents == second.family_evidence_documents
    assert first.provenance_records == second.provenance_records


def test_grade_flag_matrix_rejects_contradictions_before_artifact_creation() -> None:
    case = _clean_case()
    with TemporaryDirectory(prefix="qc-matrix-") as directory:
        bundle = real_bundle(prepare_valid_case(Path(directory), case))
    with pytest.raises(QCValidationError):
        validate_grade_flag_matrix(QualityGrade.GOOD, ("CARDIAC_MOTION",), bundle.family_results)
    with pytest.raises(QCValidationError):
        validate_grade_flag_matrix(QualityGrade.NOT_ANALYZABLE, (), bundle.family_results)
    with pytest.raises(QCValidationError):
        validate_grade_flag_matrix(
            QualityGrade.GOOD,
            ("CARDIAC_MOTION", "CARDIAC_MOTION"),
            bundle.family_results,
        )
    altered = dict(bundle.payload_document)
    altered["flags"] = ("ABSTAINED",)
    with pytest.raises(QCValidationError):
        validate_result_bundle(replace(bundle, payload_document=altered))


def test_recomputation_is_new_and_old_payload_stays_immutable() -> None:
    case = _clean_case()
    with TemporaryDirectory(prefix="qc-recompute-") as directory:
        prepared = prepare_valid_case(Path(directory), case)
        store = DerivedArtifactStore(EventLog(), InvalidationDAG())
        old = real_bundle(prepared, id_seed=3_600_000, artifact_store=store)
        old_payload = dict(old.payload_document)
        new = real_bundle(
            prepared,
            id_seed=3_600_100,
            supersedes_ref=old.assessment.artifact_ref,
            artifact_store=store,
        )
        assert old.payload_document == old_payload
        assert old.artifact.artifact_id != new.artifact.artifact_id
        assert old.assessment.artifact_ref != new.assessment.artifact_ref
        assert new.assessment.payload.supersedes_ref == old.assessment.artifact_ref
        assert bundle_digest_view(old)[0] != bundle_digest_view(new)[0]
        assert store.get(old.artifact.artifact_id).status == ArtifactStatus.SUPERSEDED
        assert store.get(new.artifact.artifact_id).status == ArtifactStatus.ACTIVE


def test_refusal_requires_nonempty_ordered_source_instances() -> None:
    with pytest.raises(QCValidationError, match="nonempty ordered sequence"):
        create_qc_input_refusal(
            failure="absent required pair",
            source_manifest_sha256="0" * 64,
            study_ref=f"study_id:{ulid(3_900_000)}",
        )
    with pytest.raises(QCValidationError, match="nonempty ordered sequence"):
        create_qc_input_refusal(
            failure="absent required pair",
            source_manifest_sha256="0" * 64,
            study_ref=f"study_id:{ulid(3_900_001)}",
            instances=object(),  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("component", ("first", "second", "registration"))
def test_h010_pair_tracker_independently_recomputes_each_component(
    component: str,
) -> None:
    case = _clean_case()
    with TemporaryDirectory(prefix="qc-h010-") as directory:
        prepared = prepare_valid_case(Path(directory), case)
        store = DerivedArtifactStore(EventLog(), InvalidationDAG())
        pair = registered_pair(prepared, registration_id_seed=4_000_000)
        old = real_bundle(prepared, pair=pair, id_seed=3_700_000, artifact_store=store)
        old_payload = dict(old.payload_document)
        old_payload_digest = canonical_digest(old_payload)
        old_artifact_payload = old.artifact.payload
        tracker = PairGeometryTracker(store, pair)
        assert old.artifact.lineage.geometry_base is not None
        assert tracker.composite_geometry_id == old.artifact.lineage.geometry_base.geometry_id
        assert tracker.composite_version == old.artifact.lineage.geometry_base.geometry_version

        bumpers = {
            "first": tracker.bump_first,
            "second": tracker.bump_second,
            "registration": tracker.bump_registration,
        }
        stale = bumpers[component]()
        assert old.artifact.artifact_id in stale
        component_versions = {
            "first": (2, 1, 1),
            "second": (1, 2, 1),
            "registration": (1, 1, 2),
        }[component]
        expected_version = sum(component_versions) - 2
        assert tracker.composite_version == expected_version
        assert tracker.composite_version > 1
        stored_old = store.get(old.artifact.artifact_id)
        assert stored_old.status == ArtifactStatus.STALE
        assert stored_old.payload == old_artifact_payload
        assert old.payload_document == old_payload
        assert canonical_digest(old.payload_document) == old_payload_digest

        versions = {
            "first": {"first_geometry_version": 2},
            "second": {"second_geometry_version": 2},
            "registration": {"registration_version": 2},
        }[component]
        new_pair = registered_pair(
            prepared,
            registration_id_seed=4_000_100,
            first_geometry_version=versions.get("first_geometry_version", 1),
            second_geometry_version=versions.get("second_geometry_version", 1),
            registration_version=versions.get("registration_version", 1),
        )
        new = real_bundle(
            prepared,
            pair=new_pair,
            id_seed=3_700_100,
            supersedes_ref=old.assessment.artifact_ref,
            artifact_store=store,
        )
        assert validate_result_bundle(new)
        assert new.artifact.artifact_id != old.artifact.artifact_id
        assert new.artifact.provenance_ref != old.artifact.provenance_ref
        assert new.assessment.payload.supersedes_ref == old.assessment.artifact_ref
        assert new.artifact.review.supersedes_ref == old.assessment.artifact_ref
        assert new.assessment.payload.produced_by_ref != old.assessment.payload.produced_by_ref
        assert new.artifact.lineage != old.artifact.lineage
        assert new.artifact.payload.blob_sha256 != old.artifact.payload.blob_sha256
        assert bundle_digest_view(new)[0] != bundle_digest_view(old)[0]
        assert new.family_result_body == old.family_result_body
        assert bundle_digest_view(new)[1] == bundle_digest_view(old)[1]
        old_family_ids = {
            ref.split(":", 1)[1]
            for ref in old.assessment.payload.evidence_refs
            if ref.startswith("provenance:")
        }
        new_family_ids = {
            ref.split(":", 1)[1]
            for ref in new.assessment.payload.evidence_refs
            if ref.startswith("provenance:")
        }
        assert old_family_ids.isdisjoint(new_family_ids)
        new_aggregate = next(
            record
            for record in new.provenance_records
            if record.provenance_id == new.artifact.provenance_ref.removeprefix("provenance:")
        )
        assert new_aggregate.provenance_id != old.artifact.provenance_ref.removeprefix(
            "provenance:"
        )
        assert new_aggregate.event_type.value == "RECOMPUTED"
        assert new.artifact.lineage.geometry_base is not None
        assert new.artifact.lineage.geometry_base.geometry_version == tracker.composite_version
        assert store.get(old.artifact.artifact_id).status == ArtifactStatus.SUPERSEDED
        assert store.get(new.artifact.artifact_id).status == ArtifactStatus.ACTIVE
        assert store.get(old.artifact.artifact_id).payload == old_artifact_payload


def test_f6_recompute_under_physical_equivalent_geometry_compares_content_only() -> None:
    case = _clean_case()
    with TemporaryDirectory(prefix="qc-f6-equivalent-") as directory:
        prepared = prepare_valid_case(Path(directory), case)
        store = DerivedArtifactStore(EventLog(), InvalidationDAG())
        old_pair = registered_pair(prepared, registration_id_seed=4_100_000)
        old = real_bundle(
            prepared,
            pair=old_pair,
            id_seed=3_900_000,
            artifact_store=store,
        )
        transformed = handedness_preserving(prepared, variant=60_000).prepared
        new_pair = registered_pair(transformed, registration_id_seed=4_100_100)
        new = real_bundle(
            transformed,
            pair=new_pair,
            id_seed=3_900_100,
            supersedes_ref=old.assessment.artifact_ref,
            artifact_store=store,
        )

    assert validate_result_bundle(new)
    assert new.artifact.artifact_id != old.artifact.artifact_id
    assert new.assessment.artifact_ref != old.assessment.artifact_ref
    assert new.artifact.provenance_ref != old.artifact.provenance_ref
    assert new.assessment.payload.produced_by_ref != old.assessment.payload.produced_by_ref
    assert new.assessment.payload.supersedes_ref == old.assessment.artifact_ref
    assert new.artifact.review.supersedes_ref == old.assessment.artifact_ref
    assert new.artifact.lineage != old.artifact.lineage
    assert new.artifact.payload.blob_sha256 != old.artifact.payload.blob_sha256
    assert bundle_digest_view(new)[0] != bundle_digest_view(old)[0]
    assert new.family_result_body == old.family_result_body
    assert canonical_digest(new.parameter_manifest) == canonical_digest(old.parameter_manifest)
    assert new.provenance_records != old.provenance_records
    assert store.get(old.artifact.artifact_id).status == ArtifactStatus.SUPERSEDED
    assert store.get(new.artifact.artifact_id).status == ArtifactStatus.ACTIVE


def test_real_policy_rejects_placeholder_unknown_schema_and_stale_records() -> None:
    case = _clean_case()
    study = f"study_id:{ulid(7_000_000)}"
    placeholder = create_legacy_placeholder(subject_refs=(study,))
    legacy = qc_consumer_gate((study,), (placeholder,), consumer="legacy-fixture")
    assert legacy.proceed
    real_placeholder = real_qc_consumer_gate((study,), (placeholder,), consumer="real-qc")
    assert real_placeholder.abstained
    assert real_placeholder.reason_code.value == "QC_NOT_ASSESSED"
    assert "QC_LEGACY_PLACEHOLDER" in real_placeholder.diagnostic_codes

    unknown = {
        "artifact_ref": f"qc_artifact:{ulid(7_000_001)}",
        "status": "ACTIVE",
        "payload": {
            "schema_version": "ccta.payload.qc_assessment.v2",
            "subject_refs": (study,),
            "quality_grade": "GOOD",
        },
    }
    unknown_result = real_qc_consumer_gate((study,), (unknown,), consumer="real-qc")
    assert unknown_result.abstained
    assert unknown_result.reason_code.value == "QC_NOT_ASSESSED"
    assert "QC_SCHEMA_INCOMPATIBLE" in unknown_result.diagnostic_codes

    with TemporaryDirectory(prefix="qc-policy-") as directory:
        bundle = real_bundle(prepare_valid_case(Path(directory), case))
    stale_assessment = replace(bundle.assessment, status="STALE")
    stale_result = real_qc_consumer_gate(
        (bundle.assessment.payload.subject_refs[0],),
        (replace(bundle, assessment=stale_assessment),),
        consumer="real-qc",
    )
    assert stale_result.abstained
    assert stale_result.reason_code.value == "QC_NOT_ASSESSED"
    assert "QC_RECORD_INELIGIBLE" in stale_result.diagnostic_codes


def test_policy_selects_explicit_successor_and_reports_incomparable_conflict() -> None:
    case = _clean_case()
    with TemporaryDirectory(prefix="qc-policy-chain-") as directory:
        prepared = prepare_valid_case(Path(directory), case)
        old = real_bundle(prepared, id_seed=3_800_000)
        successor = real_bundle(
            prepared,
            id_seed=3_800_100,
            supersedes_ref=old.assessment.artifact_ref,
        )
        study = successor.assessment.payload.subject_refs[0]
        selected = real_qc_consumer_gate(
            (study,),
            (successor, old),
            consumer="real-qc",
        )
        assert selected.proceed
        assert selected.assessment_refs == (successor.assessment.artifact_ref,)

        branch = real_bundle(prepared, id_seed=3_800_200)
        conflict = real_qc_consumer_gate((study,), (old, branch), consumer="real-qc")
        assert conflict.abstained
        assert conflict.reason_code.value == "QC_CONFLICT"
        assert "QC_CONFLICT" in conflict.diagnostic_codes


def test_f2_5_wording_and_claim_boundary_are_not_silently_rewritten() -> None:
    assert "real-scanner validation = separate later obligation A1." in F2_5_EVIDENCE_WORDING
    assert "clinical" not in F2_5_EVIDENCE_WORDING.casefold()
    assert FIXED_AT.endswith("Z")
