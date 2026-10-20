"""§2.3 shared enums - exact membership, ArtifactStatus includes STALE."""

from __future__ import annotations

from enum import Enum

import pytest

from core import enums


def test_artifact_status_includes_stale():
    members = {s.value for s in enums.ArtifactStatus}
    assert members == {"ACTIVE", "SUPERSEDED", "INVALIDATED", "STALE"}


@pytest.mark.parametrize(
    ("enum_type", "expected"),
    [
        (enums.ReleaseLane, {"PERSONAL_RESEARCH", "CLINICAL_CANDIDATE", "CLINICAL_RELEASE"}),
        (enums.QualityGrade, {"GOOD", "ACCEPTABLE", "LIMITED", "NOT_ANALYZABLE"}),
        (enums.Confidence, {"HIGH", "MEDIUM", "LOW", "UNRELIABLE", "NOT_ANALYZABLE"}),
        (
            enums.ModelStatus,
            {"EXPERIMENTAL", "CHALLENGER", "CHAMPION", "DEPRECATED", "BLOCKED"},
        ),
        (
            enums.ReviewState,
            {"PREDICTED", "ACCEPTED", "CORRECTED", "REJECTED", "CONFIRMED_FINAL"},
        ),
        (
            enums.ArtifactKind,
            {
                "SEGMENTATION",
                "CENTERLINE",
                "MESH",
                "SURFACE",
                "LANDMARK_SET",
                "CORONARY_GRAPH",
                "LESION_SET",
                "MEASUREMENT_SET",
                "QC_ASSESSMENT",
                "CLASSIFICATION_CANDIDATE",
                "PHYSIOLOGY_RESULT",
                "PHYSIOLOGY_SURROGATE",
                "REPORT_DRAFT",
                "COORDINATE_TRANSFORM",
                "REGISTRATION",
                "CORRECTION",
                "EXPORT_PACKAGE",
                "MPR_VIEW",
            },
        ),
        (
            enums.ProducerType,
            {
                "MODEL",
                "DETERMINISTIC_ENGINE",
                "RULE_ENGINE",
                "PHYSICS_BACKEND",
                "LLM",
                "HUMAN",
                "ADAPTER",
            },
        ),
    ],
)
def test_enum_memberships(enum_type: type[Enum], expected: set[str]) -> None:
    assert {member.value for member in enum_type} == expected


def test_coordinate_world_is_dicom_patient_lps_mm():
    assert enums.CoordinateSystem.DICOM_PATIENT.value == "DICOM_PATIENT"
    assert enums.Handedness.LPS.value == "LPS"
    assert {u.value for u in enums.Units} == {"MM", "HU"}
