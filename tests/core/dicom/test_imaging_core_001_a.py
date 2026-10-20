"""Imaging Core 001-A boundary, adapter, stack, and QC seam tests."""

from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path
from typing import Any, cast

import pytest

from core.dicom import (
    CLASSIFICATION_BOUNDARY_GRID,
    ClassificationOutcome,
    ClassificationReason,
    ContrastState,
    DicomAbstentionError,
    DicomInstance,
    GeometryState,
    PhaseState,
    ProtocolState,
    QCGateDecision,
    QCReason,
    SyntaxState,
    classify_boundary_states,
    classify_series,
    create_qc_assessment,
    qc_consumer_gate,
    read_dicom_instance,
    validate_stack,
)
from core.enums import QualityGrade
from tests.apps.fixture_factory import (
    build_instance,
    build_negative_fixture_matrix,
    build_volume_fixture_matrix,
    classification_oracle_fixtures,
)


def _instance(tmp_path: Path, **kwargs: object) -> DicomInstance:
    path = build_instance(
        tmp_path / "instance.dcm",
        sop_instance_uid="1.2.826.0.1.3680043.8.498.77.3.1.1",
        series_instance_uid="1.2.826.0.1.3680043.8.498.77.2.1",
        position=(0.0, 0.0, 0.0),
        **cast(dict[str, Any], kwargs),
    )
    return read_dicom_instance(path)


def test_dicom_instance_is_frozen_and_allowlist_reduced(tmp_path: Path) -> None:
    instance = _instance(
        tmp_path,
        series_description="  CARDIAC   secret-description  ",
        protocol_name="  Protocol   Name  ",
    )
    assert isinstance(instance, DicomInstance)
    assert instance.series_description == "cardiac secret-description"
    assert instance.protocol_name == "protocol name"
    assert "secret-description" in repr(instance)
    assert not hasattr(instance, "PatientName")
    assert {field.name for field in fields(instance)} == {
        "study_instance_uid",
        "series_instance_uid",
        "sop_instance_uid",
        "frame_of_reference_uid",
        "modality",
        "sop_class_uid",
        "image_type",
        "series_description",
        "body_part_examined",
        "protocol_name",
        "image_position_patient",
        "image_orientation_patient",
        "pixel_spacing",
        "slice_thickness_mm",
        "rows",
        "columns",
        "pixel_representation",
        "bits_allocated",
        "bits_stored",
        "high_bit",
        "rescale_slope",
        "rescale_intercept",
        "rescale_type",
        "contrast_bolus_agent",
        "contrast_bolus_volume_ml",
        "contrast_bolus_start_time",
        "contrast_bolus_stop_time",
        "contrast_flow_rate",
        "contrast_flow_duration",
        "temporal_position_index",
        "trigger_time_ms",
        "cardiac_synchronization_technique",
        "cardiac_signal_source",
        "acquisition_number",
        "number_of_temporal_positions",
        "transfer_syntax_uid",
        "pixel_data",
        "pixel_data_byte_length",
        "number_of_frames",
        "samples_per_pixel",
        "photometric_interpretation",
        "study_date",
    }
    with pytest.raises((AttributeError, TypeError)):
        instance.rows = 5  # type: ignore[misc]


@pytest.mark.parametrize(
    ("name", "kwargs", "expected"),
    (
        ("syntax", {"bits_allocated": 8}, "DICOM_SYNTAX_INVALID"),
        ("samples", {"samples_per_pixel": 3}, "DICOM_SYNTAX_INVALID"),
        ("photometric", {"photometric_interpretation": "RGB"}, "DICOM_SYNTAX_INVALID"),
        ("frames", {"number_of_frames": 2}, "DICOM_SYNTAX_INVALID"),
        (
            "excluded",
            {"image_type": ("ORIGINAL", "PRIMARY", "MPR")},
            "DERIVED_IMAGE_UNSUPPORTED",
        ),
        (
            "sop",
            {"sop_class_uid": "1.2.840.10008.5.1.4.1.1.4"},
            "PROTOCOL_UNSUPPORTED",
        ),
    ),
)
def test_reader_abstains_on_each_closed_instance_failure(
    tmp_path: Path,
    name: str,
    kwargs: dict[str, object],
    expected: str,
) -> None:
    del name
    with pytest.raises(DicomAbstentionError) as error:
        _instance(tmp_path, **kwargs)
    assert error.value.reason_code == expected
    assert "RGB" not in str(error.value)


def test_reader_rejects_byte_length_and_non_admitted_syntax(tmp_path: Path) -> None:
    with pytest.raises(DicomAbstentionError) as error:
        _instance(tmp_path, pixel_data_byte_length_override=2)
    assert error.value.reason_code == "DICOM_SYNTAX_INVALID"

    # Implicit VR LE is admitted since DICOM_IMPLICIT_VR_LE_ADMISSION_001;
    # Deflated Explicit VR LE remains outside the admitted syntax set.
    with pytest.raises(DicomAbstentionError) as error:
        _instance(tmp_path, transfer_syntax_uid="1.2.840.10008.1.2.1.99")
    assert error.value.reason_code == "DICOM_SYNTAX_INVALID"
    assert error.value.detail_code == "UNSUPPORTED_TRANSFER_SYNTAX"


def test_boundary_oracle_has_exact_frozen_shape_and_precedence() -> None:
    assert len(CLASSIFICATION_BOUNDARY_GRID) == 400
    assert CLASSIFICATION_BOUNDARY_GRID[0].number == 1
    assert CLASSIFICATION_BOUNDARY_GRID[-1].number == 400
    for fixture, row in zip(classification_oracle_fixtures(), CLASSIFICATION_BOUNDARY_GRID):
        assert fixture.row_number == row.number
        expected = classify_boundary_states(
            fixture.protocol_state,
            fixture.contrast_state,
            fixture.phase_state,
            fixture.geometry_state,
            fixture.syntax_state,
        )
        assert expected.outcome.value == fixture.expected_outcome
        assert expected.visible_reason.value == fixture.expected_reason
        assert expected.outcome == row.outcome
        assert expected.visible_reason == row.visible_reason
    syntax_first = classify_boundary_states(
        ProtocolState.PROTOCOL_IDENTITY_INVALID,
        ContrastState.CONTRADICTORY,
        PhaseState.CONFLICT,
        GeometryState.INVALID,
        SyntaxState.INVALID,
    )
    assert syntax_first.outcome == ClassificationOutcome.NOT_ANALYZABLE
    assert syntax_first.visible_reason == ClassificationReason.DICOM_SYNTAX_INVALID


def test_classification_retains_per_dimension_and_never_uses_temporal_as_contrast(
    tmp_path: Path,
) -> None:
    instance = _instance(
        tmp_path,
        temporal_position_index=1,
        series_description="cardiac",
    )
    result = classify_series((instance,))
    assert result.protocol.value == "CARDIAC_CT"
    assert result.contrast.value == "UNKNOWN"
    assert result.phase.value == "PHASE_BOUND"
    assert result.visible_reason == ClassificationReason.CONTRAST_EVIDENCE_ABSENT
    assert result.provenance.classifier == "rule-table"
    assert result.provenance.rule_table_version == "1"
    assert "TemporalPositionIndex" in result.provenance.evidence_fields


@pytest.mark.parametrize(
    (
        "case",
        "kwargs",
        "expected_outcome",
        "expected_contrast",
        "expected_contrast_reason",
        "expected_visible_reason",
    ),
    (
        (
            "agent-without-corroboration",
            {"contrast_bolus_agent": "iohexol"},
            ClassificationOutcome.IMPORTABLE_UNKNOWN,
            "UNKNOWN",
            ClassificationReason.CONTRAST_EVIDENCE_INCOMPLETE,
            ClassificationReason.CONTRAST_EVIDENCE_INCOMPLETE,
        ),
        (
            "agent-with-positive-volume",
            {"contrast_bolus_agent": "iohexol", "contrast_bolus_volume": 50.0},
            ClassificationOutcome.IMPORTABLE_UNKNOWN,
            "CONTRAST",
            ClassificationReason.CLASSIFICATION_COMPLETE,
            ClassificationReason.PHASE_EVIDENCE_ABSENT,
        ),
        (
            "agent-with-valid-timing",
            {
                "contrast_bolus_agent": "iohexol",
                "contrast_bolus_start_time": "101000",
                "contrast_bolus_stop_time": "101500",
            },
            ClassificationOutcome.IMPORTABLE_UNKNOWN,
            "CONTRAST",
            ClassificationReason.CLASSIFICATION_COMPLETE,
            ClassificationReason.PHASE_EVIDENCE_ABSENT,
        ),
        (
            "agent-with-zero-volume",
            {"contrast_bolus_agent": "iohexol", "contrast_bolus_volume": 0.0},
            ClassificationOutcome.NOT_ANALYZABLE,
            "NOT_ANALYZABLE",
            ClassificationReason.CONTRAST_CONFLICT,
            ClassificationReason.CONTRAST_CONFLICT,
        ),
        (
            "positive-volume-without-agent",
            {"contrast_bolus_volume": 50.0},
            ClassificationOutcome.IMPORTABLE_UNKNOWN,
            "UNKNOWN",
            ClassificationReason.CONTRAST_EVIDENCE_INCOMPLETE,
            ClassificationReason.CONTRAST_EVIDENCE_INCOMPLETE,
        ),
    ),
)
def test_contrast_evidence_rules_use_real_pydicom_fixtures(
    tmp_path: Path,
    case: str,
    kwargs: dict[str, object],
    expected_outcome: ClassificationOutcome,
    expected_contrast: str,
    expected_contrast_reason: ClassificationReason,
    expected_visible_reason: ClassificationReason,
) -> None:
    del case
    instance = _instance(tmp_path, **kwargs)
    result = classify_series((instance,))

    assert result.outcome == expected_outcome
    assert result.contrast.value == expected_contrast
    assert result.contrast.reason_code == expected_contrast_reason
    assert result.visible_reason == expected_visible_reason


def test_classification_positive_noncontrast_and_conflict(tmp_path: Path) -> None:
    noncontrast = _instance(
        tmp_path,
        series_description="cardiac native non-contrast",
        body_part_examined=None,
    )
    result = classify_series((noncontrast,))
    assert result.contrast.value == "NON_CONTRAST"
    assert result.outcome == ClassificationOutcome.IMPORTABLE_UNKNOWN
    assert result.visible_reason == ClassificationReason.PHASE_EVIDENCE_ABSENT

    conflict = replace(
        noncontrast,
        contrast_bolus_agent="iohexol",
        contrast_bolus_volume_ml=0.0,
    )
    conflict_result = classify_series((conflict,))
    assert conflict_result.outcome == ClassificationOutcome.NOT_ANALYZABLE
    assert conflict_result.visible_reason == ClassificationReason.CONTRAST_CONFLICT


def test_phase_conflicts_are_not_silently_reduced_to_unknown(tmp_path: Path) -> None:
    first = _instance(tmp_path, trigger_time=10.0)
    second = replace(
        first,
        sop_instance_uid="1.2.826.0.1.3680043.8.498.77.3.1.2",
        trigger_time_ms=20.0,
        image_position_patient=(0.0, 0.0, 1.0),
    )
    trigger_conflict = classify_series((first, second))
    assert trigger_conflict.phase.value == "NOT_ANALYZABLE"
    assert trigger_conflict.visible_reason == ClassificationReason.PHASE_CONFLICT

    first_technique = replace(
        first,
        trigger_time_ms=None,
        cardiac_synchronization_technique="PROSPECTIVE",
    )
    second_technique = replace(
        first,
        sop_instance_uid="1.2.826.0.1.3680043.8.498.77.3.1.2",
        trigger_time_ms=None,
        cardiac_synchronization_technique="RETROSPECTIVE",
        image_position_patient=(0.0, 0.0, 1.0),
    )
    technique_conflict = classify_series((first_technique, second_technique))
    assert technique_conflict.phase.value == "NOT_ANALYZABLE"
    assert technique_conflict.visible_reason == ClassificationReason.PHASE_CONFLICT


def test_stack_validator_refuses_duplicate_drift_gap_and_missing(tmp_path: Path) -> None:
    first = _instance(tmp_path)
    second_path = build_instance(
        tmp_path / "second.dcm",
        sop_instance_uid="1.2.826.0.1.3680043.8.498.77.3.1.2",
        series_instance_uid=first.series_instance_uid,
        position=(0.0, 0.0, 1.0),
    )
    second = read_dicom_instance(second_path)
    assert validate_stack((first, second)).admitted
    drift = validate_stack(
        (first, replace(second, image_position_patient=(0.1, 0.0, 1.0)))
    )
    duplicate = validate_stack(
        (first, replace(second, image_position_patient=(0.0, 0.0, 0.0)))
    )
    missing = validate_stack(
        (first, replace(second, image_position_patient=(0.0, 0.0, 2.0)))
    )
    assert drift.detail_code == ClassificationReason.SERIES_STACK_INVALID
    assert duplicate.detail_code == ClassificationReason.SERIES_STACK_INVALID
    assert missing.detail_code == ClassificationReason.SERIES_INCOMPLETE
    assert not validate_stack((first,), skipped_instance_could_belong=True).admitted


def test_volume_fixture_factory_covers_all_48_strata(tmp_path: Path) -> None:
    cases = build_volume_fixture_matrix(tmp_path / "volume")
    assert len(cases) == 48
    assert all(len(case.paths) == 3 for case in cases)
    assert len({case.fixture_id for case in cases}) == 48
    for case in cases:
        instances = tuple(read_dicom_instance(path) for path in case.paths)
        stack = validate_stack(instances)
        assert stack.admitted, case.fixture_id
        result = classify_series(instances, stack=stack)
        assert result.protocol.value == "CARDIAC_CT"


def test_negative_fixture_matrix_covers_reader_and_stack_abstentions(
    tmp_path: Path,
) -> None:
    cases = build_negative_fixture_matrix(tmp_path / "negative")
    assert len(cases) == 10
    for case in cases:
        if case.stage == "reader":
            with pytest.raises(DicomAbstentionError) as error:
                read_dicom_instance(case.paths[0])
            observed = error.value.detail_code or error.value.reason_code
            assert observed == case.expected_reason, case.fixture_id
        else:
            instances = tuple(read_dicom_instance(path) for path in case.paths)
            stack = validate_stack(instances)
            assert not stack.admitted, case.fixture_id
            assert stack.detail_code is not None
            assert stack.detail_code.value == case.expected_reason, case.fixture_id


def test_qc_placeholder_and_consumer_abstentions() -> None:
    study_ref = "study_id:01ARZ3NDEKTSV4RRFFQ69G5FAV"
    volume_ref = "volume_id:01ARZ3NDEKTSV4RRFFQ69G5FAW"
    provenance_ref = "provenance:01ARZ3NDEKTSV4RRFFQ69G5FAT"
    artifact_ref = "qc_artifact:01ARZ3NDEKTSV4RRFFQ69G5FAS"
    missing = qc_consumer_gate((volume_ref,), consumer="mpr")
    assert missing.decision == QCGateDecision.ABSTAIN
    assert missing.reason_code == QCReason.QC_NOT_ASSESSED
    placeholder = create_qc_assessment(
        subject_refs=(study_ref, volume_ref),
        provenance_ref=provenance_ref,
        artifact_ref=artifact_ref,
    )
    assert placeholder.payload.quality_grade.value == "GOOD"
    assert placeholder.payload.flags == ()
    assert placeholder.payload.evidence_refs == ()
    assert placeholder.payload.schema_version == "ccta.payload.qc_assessment.v1"
    proceed = qc_consumer_gate((volume_ref,), (placeholder,), consumer="mpr")
    assert proceed.decision == QCGateDecision.PROCEED

    limited = replace(
        placeholder,
        payload=replace(
            placeholder.payload,
            quality_grade=QualityGrade.LIMITED,
            justification_code="RULE_ENGINE_EVIDENCE",
            evidence_refs=(provenance_ref,),
        ),
    )
    assert (
        qc_consumer_gate((volume_ref,), (limited,), consumer="mpr").reason_code
        == QCReason.QC_LIMITED
    )
    not_analyzable = replace(
        placeholder,
        payload=replace(
            placeholder.payload,
            quality_grade=QualityGrade.NOT_ANALYZABLE,
            justification_code="RULE_ENGINE_EVIDENCE",
            evidence_refs=(provenance_ref,),
        ),
    )
    assert (
        qc_consumer_gate(
            (volume_ref,), (not_analyzable,), consumer="mpr"
        ).reason_code
        == QCReason.QC_NOT_ANALYZABLE
    )


def test_qc_gate_resolves_lineage_and_supersession() -> None:
    study_ref = "study_id:01BX5ZZKBKACTAV9WEVGEMMVRZ"
    series_ref = "series_id:01BX5ZZKBKACTAV9WEVGEMMVS0"
    volume_ref = "volume_id:01BX5ZZKBKACTAV9WEVGEMMVS1"
    old_ref = "qc_artifact:01BX5ZZKBKACTAV9WEVGEMMVS2"
    new_ref = "qc_artifact:01BX5ZZKBKACTAV9WEVGEMMVS3"
    old = create_qc_assessment(
        subject_refs=(study_ref,),
        provenance_ref="provenance:01BX5ZZKBKACTAV9WEVGEMMVS4",
        artifact_ref=old_ref,
    )
    newer_payload = replace(
        old.payload,
        produced_by_ref="provenance:01BX5ZZKBKACTAV9WEVGEMMVS5",
        supersedes_ref=old_ref,
        justification_code="RULE_ENGINE_EVIDENCE",
        evidence_refs=("provenance:01BX5ZZKBKACTAV9WEVGEMMVS5",),
    )
    newer = replace(old, artifact_ref=new_ref, payload=newer_payload)
    result = qc_consumer_gate(
        (volume_ref,),
        (old, newer),
        consumer="volume-consumer",
        subject_lineage={
            volume_ref: {"series_ref": series_ref},
            series_ref: {"study_ref": study_ref},
        },
    )
    assert result.decision == QCGateDecision.PROCEED
    assert result.assessment_refs == (new_ref,)
