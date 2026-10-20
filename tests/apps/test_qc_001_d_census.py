"""QC_001-D real-reader census and policy evidence (N=928)."""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

import pytest

from apps.workstation.volume_loader import CanonicalVolumeLoader, VolumeAbstentionError
from core.dicom.instance import DicomAbstentionError, DicomInstance
from core.dicom.reader import read_dicom_instance
from core.dicom.stack import validate_stack
from core.enums import QualityGrade
from core.qc import (
    QCAbstentionCode,
    QCOutcome,
    create_qc_input_refusal,
    evaluate_family,
    real_qc_consumer_gate,
    validate_result_bundle,
)
from core.qc.types import FamilyResult
from tests.apps.qc_001_d_fixtures import (
    CensusCase,
    build_invalid_stack,
    census_cases,
    prepare_valid_case,
    source_bound_instances,
    source_manifest_digest,
    ulid,
)
from tests.apps.qc_001_d_oracle import (
    INDEPENDENT_FLAGS,
    INVALID_EXPECTED_CODES,
    expected_family_outcomes,
    expected_grade_and_flags,
    is_unsafe,
)
from tests.apps.qc_001_d_support import qc_input, real_bundle


def _instance_digests(instances: tuple[DicomInstance, ...]) -> dict[str, str]:
    return {
        instance.sop_instance_uid: hashlib.sha256(instance.pixel_data).hexdigest()
        for instance in instances
    }


def _outcome_value(result: FamilyResult) -> str:
    return result.outcome.value if isinstance(result.outcome, QCOutcome) else str(result.outcome)


def _upstream_failure(
    root: Path, case: CensusCase
) -> tuple[object, tuple[DicomInstance, ...]]:
    """Execute the real reader/stack/materializer boundary for one invalid row."""

    def provenance_instances() -> tuple[DicomInstance, ...]:
        control = census_cases()[case.ordinal + 1]
        return prepare_valid_case(root, control).first_instances

    if case.invalid_family == "absent-required-pair":
        return "absent required pair", provenance_instances()
    paths = build_invalid_stack(root, case)
    instances: list[DicomInstance] = []
    try:
        for path in paths:
            instances.append(read_dicom_instance(path))
    except DicomAbstentionError as exc:
        if instances:
            return exc, tuple(instances)
        if case.invalid_family not in {
            "unsupported-transfer-syntax",
            "malformed-pixel-length",
            "multiframe",
        }:
            raise AssertionError(
                "unexpected pre-parse refusal without source-bound metadata"
            ) from exc
        return exc, source_bound_instances(paths)
    stack = validate_stack(tuple(instances))
    if not stack.admitted:
        return stack.detail_code, tuple(instances)
    if case.invalid_family == "unknown-scalar-semantics":
        try:
            CanonicalVolumeLoader(root / "unknown-blobs").materialize_dicom_stack(
                tuple(instances), scalar_domain="UNKNOWN", intensity_unit="UNKNOWN"
            )
        except VolumeAbstentionError as exc:
            return exc, tuple(instances)
    raise AssertionError("invalid census fixture was admitted without a typed failure")


def test_independent_oracle_has_no_production_qc_import() -> None:
    source = Path(__file__).with_name("qc_001_d_oracle.py").read_text(encoding="utf-8")
    assert "core.qc" not in source
    assert "core/qc" not in source


def test_qc_001_d_census_exactly_928_real_reader_cases(tmp_path: Path) -> None:
    rows = census_cases()
    assert len(rows) == 928
    assert sum(row.valid for row in rows) == 920
    assert sum(row.invalid_family is not None for row in rows) == 8
    assert len(INDEPENDENT_FLAGS) == 9

    exact_matches = 0
    unsafe_cases = 0
    unsafe_proceeds = 0
    provenance_complete = 0
    abstention_reasons: list[str] = []
    per_flag = {
        flag: Counter({"TP": 0, "FN": 0, "TN": 0, "FP": 0})
        for flag in INDEPENDENT_FLAGS
    }
    abstained_counts = Counter({"TP": 0, "FN": 0, "TN": 0, "FP": 0})

    for row in rows:
        if row.valid:
            prepared = prepare_valid_case(tmp_path, row)
            bundle = real_bundle(prepared, id_seed=3_000_000 + row.ordinal * 16)
            assert validate_result_bundle(bundle)
            result_labels = tuple(_outcome_value(result) for result in bundle.family_results)
            assert result_labels == expected_family_outcomes(row)
            expected_grade, expected_flags = expected_grade_and_flags(row)
            assert bundle.assessment.payload.quality_grade.value == expected_grade
            assert bundle.flags == expected_flags
            gate = real_qc_consumer_gate(
                (bundle.assessment.payload.subject_refs[0],),
                (bundle,),
                consumer="qc_001_d_census",
            )
            if is_unsafe(row):
                unsafe_cases += 1
                assert gate.abstained
                assert gate.reason_code.value == "QC_LIMITED"
                unsafe_proceeds += int(gate.proceed)
            else:
                assert gate.proceed
            assert len(bundle.provenance_records) >= 14
            assert len(bundle.family_evidence_documents) == 9
            assert bundle.assessment.artifact_ref == f"qc_artifact:{bundle.artifact.artifact_id}"
            assert bundle.artifact.provenance_ref == bundle.assessment.payload.produced_by_ref
            records = {record.provenance_id: record for record in bundle.provenance_records}
            for reference in bundle.assessment.payload.evidence_refs:
                if reference.startswith("provenance:"):
                    assert reference.split(":", 1)[1] in records
            provenance_complete += 1
            exact_matches += 1

            actual_injected = row.injection is not None and row.injection.flag is not None
            for result, flag in zip(bundle.family_results, INDEPENDENT_FLAGS):
                actual_positive = _outcome_value(result) in {"MILD", "SEVERE"}
                if actual_injected and row.injection.flag == flag:  # type: ignore[union-attr]
                    per_flag[flag]["TP" if actual_positive else "FN"] += 1
                else:
                    per_flag[flag]["FP" if actual_positive else "TN"] += 1
            has_abstention = any(
                result.outcome == QCOutcome.ABSTAINED for result in bundle.family_results
            )
            # ABSTAINED is the fail-closed upstream/refusal endpoint, so every
            # admitted valid row is a negative regardless of which named
            # family was injected.
            abstained_counts["FP"] += int(actual_injected and has_abstention)
            abstained_counts["TN"] += int(not has_abstention)
            continue

        assert row.invalid_family is not None
        failure: object
        instances: tuple[DicomInstance, ...]
        if row.invalid_family == "absent-required-pair":
            control = rows[row.ordinal + 1]
            prepared_control = prepare_valid_case(tmp_path, control)
            observed = evaluate_family("CARDIAC_MOTION", qc_input(prepared_control))
            assert observed.outcome == QCOutcome.ABSTAINED
            assert observed.abstention_code == QCAbstentionCode.QC_REQUIRED_PAIR_ABSENT
        failure, instances = _upstream_failure(tmp_path, row)
        expected_code = INVALID_EXPECTED_CODES[row.invalid_family]
        refusal = create_qc_input_refusal(
            failure=failure,
            source_manifest_sha256=source_manifest_digest(row),
            study_ref=f"study_id:{ulid(6_000_000 + row.ordinal)}",
            family="CARDIAC_MOTION",
            instances=instances,
            instance_digests=_instance_digests(instances),
            artifact_id=ulid(6_100_000 + row.ordinal * 4),
            provenance_id=ulid(6_100_001 + row.ordinal * 4),
            family_provenance_id=ulid(6_100_002 + row.ordinal * 4),
            created_at="2026-01-01T00:00:00Z",
            code_commit_sha="0" * 40,
        )
        assert refusal.quality_grade == QualityGrade.NOT_ANALYZABLE
        assert refusal.flags == ("ABSTAINED",)
        assert refusal.refusal is not None
        assert QCAbstentionCode(refusal.refusal.reason_code).value == expected_code
        assert validate_result_bundle(refusal)
        gate = real_qc_consumer_gate(
            (refusal.assessment.payload.subject_refs[0],),
            (refusal,),
            consumer="qc_001_d_invalid",
        )
        assert gate.abstained
        assert gate.reason_code.value == "QC_NOT_ANALYZABLE"
        abstention_reasons.append(QCAbstentionCode(refusal.refusal.reason_code).value)
        for flag in INDEPENDENT_FLAGS:
            per_flag[flag]["TN"] += 1
        provenance_complete += 1
        exact_matches += 1
        unsafe_cases += 1
        unsafe_proceeds += int(gate.proceed)
        abstained_counts["TP"] += 1

    assert exact_matches == 928
    assert unsafe_cases == 440
    assert unsafe_proceeds == 0
    assert provenance_complete == 928
    assert Counter(abstention_reasons) == Counter(INVALID_EXPECTED_CODES.values())
    for flag in INDEPENDENT_FLAGS:
        assert per_flag[flag]["TP"] == 96
        assert per_flag[flag]["FN"] == 0
        assert per_flag[flag]["TN"] == 832
        assert per_flag[flag]["FP"] == 0
    assert abstained_counts["TP"] == 8
    assert abstained_counts["FN"] == 0
    assert abstained_counts["TN"] == 920
    assert abstained_counts["FP"] == 0


@pytest.mark.parametrize("invalid_name", tuple(INVALID_EXPECTED_CODES))
def test_invalid_failure_mapping_is_closed(invalid_name: str) -> None:
    from core.qc import map_upstream_failure

    assert map_upstream_failure(invalid_name) == QCAbstentionCode(
        INVALID_EXPECTED_CODES[invalid_name]
    )
