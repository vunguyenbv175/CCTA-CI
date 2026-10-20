"""Pure deterministic-family, classifier, and fail-closed boundary tests."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from math import nextafter
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest

from core.qc import (
    CoverageEnvelope,
    QCAbstentionCode,
    QCOutcome,
    classify_high_bad,
    classify_incomplete_anatomy,
    classify_low_bad,
    classify_stair_step,
    evaluate_family,
    lower_quantile,
    median,
    robust_range,
)
from tests.apps.qc_001_d_fixtures import census_cases, prepare_valid_case
from tests.apps.qc_001_d_support import qc_input, registered_pair


def test_quantile_is_the_lower_order_statistic() -> None:
    values = (4.0, 1.0, 9.0, 2.0, 8.0)
    assert lower_quantile(values, 0.0) == 1.0
    assert lower_quantile(values, 0.50) == 4.0
    assert lower_quantile(values, 0.95) == 8.0
    assert median(values) == 4.0
    assert robust_range(values) == 7.0


def test_exact_108_ulp_boundary_population_uses_production_classifiers() -> None:
    high_bad = {
        "CARDIAC_MOTION": (0.08, 0.16),
        "RESPIRATORY_MOTION": (0.08, 0.16),
        "EXCESSIVE_NOISE": (0.08, 0.15),
        "BLOOMING": (0.25, 0.50),
        "BEAM_HARDENING": (0.05, 0.10),
        "MISREGISTRATION": (0.10, 0.20),
    }
    population: list[tuple[str, str, int, str, QCOutcome]] = []
    for domain in ("STORED", "MODALITY_TRANSFORMED", "HU"):
        for flag, thresholds in high_bad.items():
            for index, threshold in enumerate(thresholds):
                lower = {
                    0: QCOutcome.CLEAR,
                    1: QCOutcome.MILD,
                }[index]
                upper = QCOutcome.MILD if index == 0 else QCOutcome.SEVERE
                for side, value, expected in (
                    ("below", nextafter(threshold, -float("inf")), lower),
                    ("exact", threshold, upper),
                ):
                    assert classify_high_bad(value, *thresholds) == expected
                    population.append((domain, flag, index, side, expected))
        for flag, thresholds in (("POOR_CONTRAST", (2.0, 1.0)),):
            for index, threshold in enumerate(thresholds):
                expected_below = QCOutcome.MILD if index == 0 else QCOutcome.SEVERE
                expected_exact = QCOutcome.CLEAR if index == 0 else QCOutcome.MILD
                for side, value, expected in (
                    ("below", nextafter(threshold, -float("inf")), expected_below),
                    ("exact", threshold, expected_exact),
                ):
                    assert classify_low_bad(value, thresholds[1], thresholds[0]) == expected
                    population.append((domain, flag, index, side, expected))
        stair_values = (
            (0, nextafter(0.15, -float("inf")), QCOutcome.CLEAR),
            (0, 0.15, QCOutcome.MILD),
            (1, nextafter(0.30, -float("inf")), QCOutcome.MILD),
            (1, 0.30, QCOutcome.SEVERE),
        )
        for index, value, expected in stair_values:
            assert classify_stair_step(
                value,
                supported_mild_count=2 if index == 0 else 8,
                supported_severe_count=0 if index == 0 else 4,
                pair_count=8,
            ) == expected
            population.append((domain, "STAIR_STEP", index, "below/exact", expected))
        for index, value, expected in (
            (0, 0.0, QCOutcome.CLEAR),
            (0, nextafter(0.0, float("inf")), QCOutcome.MILD),
            (1, nextafter(0.25, -float("inf")), QCOutcome.MILD),
            (1, 0.25, QCOutcome.SEVERE),
        ):
            assert classify_incomplete_anatomy(value) == expected
            population.append((domain, "INCOMPLETE_ANATOMY", index, "below/exact", expected))
    assert len(population) == 108


def test_common_input_failures_are_typed_abstentions() -> None:
    case = next(
        case
        for case in census_cases()
        if case.valid and case.injection is not None and case.injection.flag is None
    )
    with TemporaryDirectory(prefix="qc-unit-input-") as directory:
        prepared = prepare_valid_case(Path(directory), case)
        source = qc_input(prepared)

        missing_pair = evaluate_family("CARDIAC_MOTION", source)
        assert missing_pair.outcome == QCOutcome.ABSTAINED
        assert missing_pair.abstention_code == QCAbstentionCode.QC_REQUIRED_PAIR_ABSENT

        invalid_domain = evaluate_family(
            "POOR_CONTRAST", replace(source, scalar_domain="UNKNOWN", intensity_unit="UNKNOWN")
        )
        assert invalid_domain.outcome == QCOutcome.ABSTAINED
        assert invalid_domain.abstention_code == QCAbstentionCode.QC_SCALAR_DOMAIN_INVALID

        writable = np.array(source.array, copy=True, order="C")
        writable_volume = replace(
            source.volume,
            blob_ref=replace(
                source.volume.blob_ref,
                sha256=hashlib.sha256(writable.tobytes(order="C")).hexdigest(),
            ),
        )
        writable_input = replace(source, volume=writable_volume, array=writable)
        writable_result = evaluate_family("POOR_CONTRAST", writable_input)
        assert writable_result.outcome == QCOutcome.ABSTAINED
        assert writable_result.abstention_code == QCAbstentionCode.QC_INPUT_SURFACE_INVALID

        nonfinite = np.array(source.array, dtype=np.float64, copy=True, order="C")
        nonfinite[0, 0, 0] = np.nan
        nonfinite.setflags(write=False)
        nonfinite_volume = replace(
            source.volume,
            blob_ref=replace(
                source.volume.blob_ref,
                sha256=hashlib.sha256(nonfinite.tobytes(order="C")).hexdigest(),
                byte_size=nonfinite.nbytes,
            ),
        )
        nonfinite_input = replace(source, volume=nonfinite_volume, array=nonfinite)
        nonfinite_result = evaluate_family("POOR_CONTRAST", nonfinite_input)
        assert nonfinite_result.outcome == QCOutcome.ABSTAINED
        assert nonfinite_result.abstention_code == QCAbstentionCode.QC_SCALAR_DOMAIN_INVALID

        missing_coverage = evaluate_family("INCOMPLETE_ANATOMY", replace(source, coverage=None))
        assert missing_coverage.outcome == QCOutcome.ABSTAINED
        assert missing_coverage.abstention_code == QCAbstentionCode.QC_COVERAGE_ENVELOPE_INVALID


def test_pair_dependent_families_use_the_explicit_registered_pair() -> None:
    case = next(
        case
        for case in census_cases()
        if case.valid
        and case.injection is not None
        and case.injection.flag == "MISREGISTRATION"
        and case.injection.severity == "SEVERE"
        and case.geometry is not None
        and case.geometry.pixels == "unsigned"
    )
    with TemporaryDirectory(prefix="qc-unit-pair-") as directory:
        prepared = prepare_valid_case(Path(directory), case)
        source = qc_input(prepared)
        pair = registered_pair(prepared)
        result = evaluate_family("MISREGISTRATION", source, pair=pair)
        assert result.outcome == QCOutcome.SEVERE
        assert result.metric_value is not None and result.metric_value >= 0.20


@pytest.mark.parametrize(
    ("flag", "severity"),
    tuple(
        (flag, severity)
        for flag in (
            "CARDIAC_MOTION",
            "RESPIRATORY_MOTION",
            "STAIR_STEP",
            "POOR_CONTRAST",
            "EXCESSIVE_NOISE",
            "BLOOMING",
            "BEAM_HARDENING",
            "MISREGISTRATION",
            "INCOMPLETE_ANATOMY",
        )
        for severity in ("MILD", "SEVERE")
    ),
)
def test_each_injected_family_reaches_its_named_severity(flag: str, severity: str) -> None:
    case = next(
        case
        for case in census_cases()
        if case.valid
        and case.injection is not None
        and case.injection.flag == flag
        and case.injection.severity == severity
        and case.geometry is not None
        and case.geometry.pixels == "unsigned"
    )
    with TemporaryDirectory(prefix="qc-unit-family-") as directory:
        prepared = prepare_valid_case(Path(directory), case)
        source = qc_input(prepared)
        pair = registered_pair(prepared) if flag in {"CARDIAC_MOTION", "MISREGISTRATION"} else None
        result = evaluate_family(
            flag,
            source,
            pair=pair,
            coverage=CoverageEnvelope(anchor_world_mm=prepared.coverage_anchor),
        )
        assert isinstance(result.outcome, QCOutcome)
        assert result.outcome.value == severity
