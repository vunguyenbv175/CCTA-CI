"""QC_001-D property, metamorphic, and fail-closed evidence.

The expected labels for the census remain in ``qc_001_d_oracle``.  This file
uses the separate physical property generator and only compares production
results after each generator has sealed its relation and population.
"""

from __future__ import annotations

import ast
import hashlib
import math
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest

from apps.workstation.volume_loader import CanonicalVolumeLoader, VolumeAbstentionError
from core.coordinates import FrameSpec
from core.dicom.qc import QCAssessmentPayload
from core.enums import QualityGrade
from core.qc import (
    ABSTAINED_FLAG,
    QCInput,
    QCOutcome,
    create_qc_input_refusal,
    evaluate_all_families,
    evaluate_family,
    real_qc_consumer_gate,
)
from core.qc.types import FamilyResult, QCResultBundle
from tests.apps.qc_001_d_fixtures import (
    CensusCase,
    PreparedQCInput,
    census_cases,
    prepare_valid_case,
    source_manifest_digest,
    ulid,
)
from tests.apps.qc_001_d_oracle import (
    expected_family_outcomes,
    expected_grade_and_flags,
)
from tests.apps.qc_001_d_property_oracle import (
    IDENTITY_AFFINE,
    INVERSE_TRANSLATION,
    TRANSLATION,
    PropertySurface,
    assert_exact_surface_relation,
    conjugate_translation_affine,
    cropped,
    exact_legal_replica,
    handedness_preserving,
    hu_representation,
    reverse_i_and_k,
    round_trip,
    translated,
    world_scalar_multiset,
)
from tests.apps.qc_001_d_support import bundle_digest_view, qc_input, real_bundle, registered_pair


@dataclass
class _PropertyCorpus:
    """Lazy real-reader corpus shared by all property tests in this module."""

    rows: tuple[CensusCase, ...]
    root: Path
    prepared: dict[str, PreparedQCInput]
    baselines: dict[str, QCResultBundle]
    replicas: dict[str, PropertySurface]
    evaluations: dict[str, tuple[FamilyResult, ...]]


@pytest.fixture(scope="module")
def property_corpus(tmp_path_factory: pytest.TempPathFactory) -> _PropertyCorpus:
    root = tmp_path_factory.mktemp("qc-001-d-properties")
    return _PropertyCorpus(census_cases(), root, {}, {}, {}, {})


def _property_rows(corpus: _PropertyCorpus) -> tuple[CensusCase, ...]:
    rows = tuple(row for row in corpus.rows if row.valid and not row.matched_control)
    assert len(rows) == 912
    return rows


def _prepared(corpus: _PropertyCorpus, row: CensusCase) -> PreparedQCInput:
    cached = corpus.prepared.get(row.fixture_id)
    if cached is None:
        cached = prepare_valid_case(corpus.root, row)
        corpus.prepared[row.fixture_id] = cached
    return cached


def _replica(corpus: _PropertyCorpus, row: CensusCase) -> PropertySurface:
    cached = corpus.replicas.get(row.fixture_id)
    if cached is None:
        cached = exact_legal_replica(
            row,
            _prepared(corpus, row),
            variant=60_000 + row.ordinal,
        )
        corpus.replicas[row.fixture_id] = cached
    return cached


def _refusal(corpus: _PropertyCorpus, row: CensusCase, seed: int) -> QCResultBundle:
    if row.invalid_family is None:
        raise AssertionError("refusal helper requires an invalid row")
    control = corpus.rows[row.ordinal + 1]
    prepared = _prepared(corpus, control)
    instances = prepared.first_instances
    return create_qc_input_refusal(
        failure=row.invalid_family,
        source_manifest_sha256=source_manifest_digest(row),
        study_ref=f"study_id:{ulid(6_000_000 + row.ordinal)}",
        family="CARDIAC_MOTION",
        instances=instances,
        instance_digests={
            instance.sop_instance_uid: hashlib.sha256(instance.pixel_data).hexdigest()
            for instance in instances
        },
        artifact_id=ulid(seed),
        provenance_id=ulid(seed + 1),
        family_provenance_id=ulid(seed + 2),
        created_at="2026-01-01T00:00:00Z",
        code_commit_sha="0" * 40,
    )


def _make_baseline(corpus: _PropertyCorpus, row: CensusCase) -> QCResultBundle:
    if row.valid:
        return real_bundle(
            _prepared(corpus, row),
            id_seed=3_000_000 + row.ordinal * 16,
        )
    return _refusal(corpus, row, 6_100_000 + row.ordinal * 4)


def _baseline(corpus: _PropertyCorpus, row: CensusCase) -> QCResultBundle:
    cached = corpus.baselines.get(row.fixture_id)
    if cached is None:
        cached = _make_baseline(corpus, row)
        corpus.baselines[row.fixture_id] = cached
    return cached


def _evaluate_surface(surface: PropertySurface) -> tuple[FamilyResult, ...]:
    source = qc_input(surface.prepared)
    pair = registered_pair(
        surface.prepared,
        affine=surface.registration_affine,
        overlap_lower=surface.overlap_lower_lps_mm,
        overlap_upper=surface.overlap_upper_lps_mm,
    )
    return evaluate_all_families(source, pair=pair, coverage=source.coverage)


def _evaluate_prepared(prepared: PreparedQCInput) -> tuple[FamilyResult, ...]:
    return _evaluate_surface(PropertySurface(prepared=prepared, name="base"))


def _evaluate_cached(
    corpus: _PropertyCorpus,
    key: str,
    prepared: PreparedQCInput,
) -> tuple[FamilyResult, ...]:
    cached = corpus.evaluations.get(key)
    if cached is None:
        cached = _evaluate_prepared(prepared)
        corpus.evaluations[key] = cached
    return cached


def _replica_baseline(
    corpus: _PropertyCorpus, row: CensusCase
) -> tuple[FamilyResult, ...]:
    """Seal the independent answer before evaluating the replica with QC."""

    key = f"replica:{row.fixture_id}"
    cached = corpus.evaluations.get(key)
    if cached is None:
        expected_outcomes = expected_family_outcomes(row)
        expected_grade, expected_flags = expected_grade_and_flags(row)
        surface = _replica(corpus, row)
        cached = _evaluate_surface(surface)
        _assert_replica_known_answer(
            cached,
            expected_outcomes=expected_outcomes,
            expected_grade=expected_grade,
            expected_flags=expected_flags,
        )
        corpus.evaluations[key] = cached
    return cached


def _outcomes(results: tuple[FamilyResult, ...]) -> tuple[str, ...]:
    return tuple(getattr(result.outcome, "value", str(result.outcome)) for result in results)


def _grade(results: tuple[FamilyResult, ...]) -> str:
    outcomes = _outcomes(results)
    if ABSTAINED_FLAG in outcomes or "ABSTAINED" in outcomes:
        return QualityGrade.NOT_ANALYZABLE.value
    if "SEVERE" in outcomes:
        return QualityGrade.LIMITED.value
    if "MILD" in outcomes:
        return QualityGrade.ACCEPTABLE.value
    return QualityGrade.GOOD.value


def _assert_same_physical_result(
    expected: tuple[FamilyResult, ...], observed: tuple[FamilyResult, ...]
) -> None:
    assert _outcomes(observed) == _outcomes(expected)
    assert _grade(observed) == _grade(expected)
    assert tuple(result.flag for result in observed) == tuple(result.flag for result in expected)
    assert tuple(result.metric_value is None for result in observed) == tuple(
        result.metric_value is None for result in expected
    )
    for result in observed:
        if result.metric_value is not None:
            assert math.isfinite(float(result.metric_value))
    assert tuple(result.metric_value for result in observed) == tuple(
        result.metric_value for result in expected
    )
    assert tuple(result.abstention_code for result in observed) == tuple(
        result.abstention_code for result in expected
    )
    assert observed == expected


def _assert_replica_known_answer(
    observed: tuple[FamilyResult, ...],
    *,
    expected_outcomes: tuple[str, ...],
    expected_grade: str,
    expected_flags: tuple[str, ...],
) -> None:
    assert _outcomes(observed) == expected_outcomes
    assert _grade(observed) == expected_grade
    assert tuple(
        result.flag
        for result in observed
        if getattr(result.outcome, "value", str(result.outcome)) != "CLEAR"
    ) == expected_flags


def test_property_oracles_have_no_production_quality_imports() -> None:
    """Mechanically audit both independent oracle source files."""

    root = Path(__file__).parent
    paths = (root / "qc_001_d_oracle.py", root / "qc_001_d_property_oracle.py")
    forbidden_names = {
        "evaluate_family",
        "evaluate_all_families",
        "FAMILY_LITERALS",
        "FAMILY_PARAMETER_SCHEMA",
        "FAMILY_ALGORITHM_IDS",
    }
    for path in paths:
        source = path.read_text(encoding="utf-8")
        assert "core.qc" not in source
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
                assert all(alias.name not in forbidden_names for alias in node.names)
            else:
                continue
            assert all(
                module != "core.qc" and not module.startswith("core.qc.")
                for module in modules
            )


def test_one_um_origin_shift_is_unequal_in_raw_and_canonical_paths() -> None:
    frame = FrameSpec(
        frame_of_reference_uid="1.2.826.0.1.3680043.8.498.91.9",
        image_position_patient=(10.0, 20.0, 30.0),
        image_orientation_patient=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        spacing_mm=(1.0, 1.0, 1.0),
        dimensions=(2, 2, 2),
    )
    shifted = replace(
        frame,
        image_position_patient=(
            float(frame.image_position_patient[0]) + 0.001,
            float(frame.image_position_patient[1]) + 0.001,
            float(frame.image_position_patient[2]) + 0.001,
        ),
    )
    array = np.arange(8, dtype=np.float64).reshape((2, 2, 2))

    assert world_scalar_multiset(frame, array) != world_scalar_multiset(shifted, array)
    assert world_scalar_multiset(
        frame, array, canonical=True
    ) != world_scalar_multiset(shifted, array, canonical=True)


def test_one_tenth_picometer_origin_shift_is_unequal_in_raw_and_canonical_paths(
    property_corpus: _PropertyCorpus,
) -> None:
    frame = FrameSpec(
        frame_of_reference_uid="1.2.826.0.1.3680043.8.498.91.9",
        image_position_patient=(10.0, 20.0, 30.0),
        image_orientation_patient=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        spacing_mm=(1.0, 1.0, 1.0),
        dimensions=(2, 2, 2),
    )
    shifted = replace(
        frame,
        image_position_patient=(
            float(frame.image_position_patient[0]) + 1e-13,
            float(frame.image_position_patient[1]) + 1e-13,
            float(frame.image_position_patient[2]) + 1e-13,
        ),
    )
    array = np.arange(8, dtype=np.float64).reshape((2, 2, 2))

    assert world_scalar_multiset(frame, array) != world_scalar_multiset(shifted, array)
    assert world_scalar_multiset(
        frame, array, canonical=True
    ) != world_scalar_multiset(shifted, array, canonical=True)

    row = _property_rows(property_corpus)[0]
    replica = _replica(property_corpus, row).prepared
    replica_frame = replica.first_volume.coordinate_frame
    replica_shifted = replace(
        replica_frame,
        image_position_patient=(
            float(replica_frame.image_position_patient[0]) + 1e-13,
            float(replica_frame.image_position_patient[1]) + 1e-13,
            float(replica_frame.image_position_patient[2]) + 1e-13,
        ),
    )
    assert world_scalar_multiset(
        replica_frame, replica.first_array
    ) != world_scalar_multiset(replica_shifted, replica.first_array)
    assert world_scalar_multiset(
        replica_frame, replica.first_array, canonical=True
    ) != world_scalar_multiset(
        replica_shifted, replica.first_array, canonical=True
    )


def test_repeatability_is_exact_for_all_928_census_manifests(
    property_corpus: _PropertyCorpus,
) -> None:
    executions = 0
    for row in property_corpus.rows:
        first = _baseline(property_corpus, row)
        if row.valid:
            second = real_bundle(
                _prepared(property_corpus, row),
                id_seed=3_000_000 + row.ordinal * 16,
            )
        else:
            second = _refusal(property_corpus, row, 6_100_000 + row.ordinal * 4)
        assert bundle_digest_view(first) == bundle_digest_view(second)
        assert first.assessment == second.assessment
        assert first.artifact == second.artifact
        assert first.payload_document == second.payload_document
        assert first.parameter_manifest == second.parameter_manifest
        assert first.family_results == second.family_results
        assert first.family_evidence_documents == second.family_evidence_documents
        assert first.family_result_body == second.family_result_body
        assert first.provenance_records == second.provenance_records
        executions += 2
    assert len(property_corpus.rows) == 928
    assert executions == 1_856


def test_handedness_preserving_invariance_is_912_pairs_1824_executions(
    property_corpus: _PropertyCorpus,
) -> None:
    pairs = 0
    executions = 0
    for row in _property_rows(property_corpus):
        replica = _replica(property_corpus, row).prepared
        base = _replica_baseline(property_corpus, row)
        surface = handedness_preserving(replica, variant=10_000 + row.ordinal)
        transformed = surface.prepared
        assert_exact_surface_relation(
            replica, transformed, label=f"handedness:{row.fixture_id}"
        )
        observed = _evaluate_surface(surface)
        _assert_same_physical_result(base, observed)
        pairs += 1
        executions += 2
    assert pairs == 912
    assert executions == 1_824


def test_four_physical_equivalences_have_fixed_order_and_3648_executions(
    property_corpus: _PropertyCorpus,
) -> None:
    transforms = (
        handedness_preserving,
        reverse_i_and_k,
        translated,
        round_trip,
    )
    assert tuple(function.__name__ for function in transforms) == (
        "handedness_preserving",
        "reverse_i_and_k",
        "translated",
        "round_trip",
    )
    executions = 0
    for row in _property_rows(property_corpus):
        prepared = _prepared(property_corpus, row)
        replica = _replica(property_corpus, row).prepared
        replica_base = _replica_baseline(property_corpus, row)
        original_base = _evaluate_cached(
            property_corpus, f"original:{row.fixture_id}", prepared
        )
        for index, function in enumerate(transforms):
            source = prepared if function is round_trip else replica
            base = original_base if function is round_trip else replica_base
            surface = function(source, variant=20_000 + row.ordinal * 8 + index)
            transformed = surface.prepared
            if function in {handedness_preserving, reverse_i_and_k}:
                assert_exact_surface_relation(
                    source,
                    transformed,
                    label=f"{function.__name__}:{row.fixture_id}",
                )
            elif function is translated:
                assert surface.registration_affine == conjugate_translation_affine(
                    IDENTITY_AFFINE, TRANSLATION
                )
                assert surface.prepared.coverage_anchor == tuple(
                    float(old) + float(delta)
                    for old, delta in zip(source.coverage_anchor, TRANSLATION)
                )
                assert_exact_surface_relation(
                    source,
                    transformed,
                    translation_normalization_mm=INVERSE_TRANSLATION,
                    label=f"translated:{row.fixture_id}",
                )
            else:
                assert np.array_equal(transformed.first_array, source.first_array)
                assert_exact_surface_relation(
                    source,
                    transformed,
                    label=f"round_trip:{row.fixture_id}",
                )
            observed = _evaluate_surface(surface)
            _assert_same_physical_result(base, observed)
            executions += 1
    assert executions == 912 * 4
    assert executions == 3_648


def test_hu_representation_is_exact_for_all_912_valid_property_cases(
    property_corpus: _PropertyCorpus,
) -> None:
    executions = 0
    for row in _property_rows(property_corpus):
        prepared = _prepared(property_corpus, row)
        surface = hu_representation(prepared, variant=30_000 + row.ordinal)
        transformed = surface.prepared
        assert transformed.case.geometry is not None
        assert transformed.case.geometry.pixels == "rescaled-fractional"
        assert np.array_equal(transformed.first_array, prepared.first_array)
        assert transformed.second_array is not None
        assert prepared.second_array is not None
        assert np.array_equal(transformed.second_array, prepared.second_array)
        assert transformed.first_volume.coordinate_frame.rescale_applied is True
        assert transformed.second_volume is not None
        assert transformed.second_volume.coordinate_frame.rescale_applied is True
        assert all(instance.rescale_type == "HU" for instance in transformed.first_instances)
        assert all(instance.rescale_type == "HU" for instance in transformed.second_instances or ())
        assert_exact_surface_relation(
            prepared,
            transformed,
            label=f"hu_representation:{row.fixture_id}",
        )
        expected = _evaluate_prepared(prepared)
        observed = _evaluate_surface(surface)
        _assert_same_physical_result(expected, observed)
        executions += 1
    assert executions == 912


def test_crop_monotonicity_is_48_strata_times_two_directions(
    property_corpus: _PropertyCorpus,
) -> None:
    clean_rows = tuple(
        row
        for row in _property_rows(property_corpus)
        if row.injection is not None and row.injection.flag is None
    )
    assert len(clean_rows) == 48
    order = {"CLEAR": 0, "MILD": 1, "SEVERE": 2, "ABSTAINED": 3}
    executions = 0
    violations: list[tuple[str, str]] = []
    for row in clean_rows:
        prepared = _prepared(property_corpus, row)
        source = qc_input(prepared)
        baseline = evaluate_family(
            "INCOMPLETE_ANATOMY", source, coverage=source.coverage
        )
        for index, direction in enumerate(("inferior", "superior")):
            surface = cropped(
                prepared,
                direction=direction,
                variant=40_000 + row.ordinal * 2 + index,
            )
            transformed = surface.prepared
            observed_source = qc_input(transformed)
            observed = evaluate_family(
                "INCOMPLETE_ANATOMY",
                observed_source,
                coverage=source.coverage,
            )
            assert transformed.first_array.shape[0] == prepared.first_array.shape[0] - 1
            assert len(transformed.first_instances) == transformed.first_array.shape[0]
            if direction == "inferior":
                assert np.array_equal(transformed.first_array, prepared.first_array[1:])
            else:
                assert np.array_equal(transformed.first_array, prepared.first_array[:-1])
            baseline_label = getattr(baseline.outcome, "value", str(baseline.outcome))
            observed_label = getattr(observed.outcome, "value", str(observed.outcome))
            if order[observed_label] < order[baseline_label]:
                violations.append((row.fixture_id, direction))
            executions += 1
    assert executions == 48 * 2
    assert executions == 96
    assert violations == []


def test_padding_refusals_are_exactly_three_domains_by_three_cases() -> None:
    domains = ("STORED", "MODALITY_TRANSFORMED", "HU")
    padding_cases = (
        {"pixel_padding_present": True},
        {"pixel_padding_value": 0},
        {"pixel_padding_range_limit": 2},
    )
    refusals = 0
    for domain in domains:
        for padding in padding_cases:
            with pytest.raises(VolumeAbstentionError) as raised:
                CanonicalVolumeLoader.materialize_array(
                    np.array([[[1]]], dtype="<u2"),
                    shape=(1, 1, 1),
                    pixel_representation=0,
                    scalar_domain=domain,
                    intensity_unit=domain,
                    **padding,
                )
            assert raised.value.reason_code == "PIXEL_PADDING_UNSUPPORTED"
            refusals += 1
    assert refusals == 3 * 3
    assert refusals == 9
    with pytest.raises(TypeError):
        QCInput(
            volume=object(),
            array=object(),
            source_manifest_sha256="0" * 64,
            mask=object(),  # type: ignore[call-arg]
        )


def test_fail_closed_mutation_population_is_exactly_14(
    property_corpus: _PropertyCorpus,
) -> None:
    mutations = 0
    for row in property_corpus.rows:
        if row.invalid_family is None:
            continue
        refusal = _baseline(property_corpus, row)
        assert refusal.quality_grade == QualityGrade.NOT_ANALYZABLE
        gate = real_qc_consumer_gate(
            (refusal.assessment.payload.subject_refs[0],),
            (refusal,),
            consumer="qc_001_d_fail_closed",
        )
        assert gate.abstained
        mutations += 1
    clean = next(
        row
        for row in _property_rows(property_corpus)
        if row.injection is not None and row.injection.flag is None
    )
    source = qc_input(_prepared(property_corpus, clean))

    def with_array(array: np.ndarray) -> QCInput:
        digest = hashlib.sha256(array.tobytes(order="C")).hexdigest()
        volume = replace(
            source.volume,
            blob_ref=replace(source.volume.blob_ref, sha256=digest, byte_size=int(array.nbytes)),
        )
        return replace(source, volume=volume, array=array)

    nonfinite_values: list[np.ndarray] = []
    for value in (float("nan"), float("inf"), float("-inf")):
        array = np.array(source.array, dtype=np.float64, copy=True, order="C")
        array[0, 0, 0] = value
        array.setflags(write=False)
        nonfinite_values.append(array)
    for array in nonfinite_values:
        results = evaluate_all_families(with_array(array), coverage=source.coverage)
        assert all(result.outcome == QCOutcome.ABSTAINED for result in results)
        mutations += 1

    unsupported = replace(source, scalar_domain="UNSUPPORTED", intensity_unit="UNSUPPORTED")
    results = evaluate_all_families(unsupported, coverage=source.coverage)
    assert all(result.outcome == QCOutcome.ABSTAINED for result in results)
    mutations += 1

    short = np.array(source.array[:-1], copy=True, order="C")
    short.setflags(write=False)
    results = evaluate_all_families(with_array(short), coverage=source.coverage)
    assert all(result.outcome == QCOutcome.ABSTAINED for result in results)
    mutations += 1

    invalid_frame = replace(
        source.volume.coordinate_frame,
        dimensions=(8, source.array.shape[1], source.array.shape[2]),
    )
    invalid_geometry = replace(
        source,
        volume=replace(source.volume, coordinate_frame=invalid_frame),
    )
    results = evaluate_all_families(invalid_geometry, coverage=source.coverage)
    assert all(result.outcome == QCOutcome.ABSTAINED for result in results)
    mutations += 1

    assert mutations == 8 + 6
    assert mutations == 14


def test_all_928_payloads_reject_three_closed_vocabulary_mutations(
    property_corpus: _PropertyCorpus,
) -> None:
    mutations = 0

    def reject(payload: QCAssessmentPayload, flags: tuple[str, ...]) -> None:
        with pytest.raises((TypeError, ValueError)):
            replace(payload, flags=flags)

    for row in property_corpus.rows:
        payload = _baseline(property_corpus, row).assessment.payload
        order_flags: tuple[str, ...]
        if payload.flags == (ABSTAINED_FLAG,):
            order_flags = (ABSTAINED_FLAG, "CARDIAC_MOTION")
        elif not payload.flags:
            order_flags = (ABSTAINED_FLAG, "CARDIAC_MOTION")
        else:
            order_flags = (ABSTAINED_FLAG, *payload.flags)
        reject(payload, order_flags)
        duplicate_flags = (
            (ABSTAINED_FLAG, ABSTAINED_FLAG)
            if not payload.flags
            else (*payload.flags, payload.flags[-1])
        )
        reject(payload, duplicate_flags)
        reject(payload, (*payload.flags, "UNKNOWN_QC_FLAG"))
        mutations += 3
    assert len(property_corpus.rows) == 928
    assert mutations == 928 * 3
    assert mutations == 2_784


def test_metamorphic_generation_has_fixed_order_and_3648_executions(
    property_corpus: _PropertyCorpus,
) -> None:
    transforms = (
        handedness_preserving,
        translated,
        hu_representation,
        round_trip,
    )
    assert tuple(function.__name__ for function in transforms) == (
        "handedness_preserving",
        "translated",
        "hu_representation",
        "round_trip",
    )
    executions = 0
    for row in _property_rows(property_corpus):
        prepared = _prepared(property_corpus, row)
        replica = _replica(property_corpus, row).prepared
        replica_base = _replica_baseline(property_corpus, row)
        original_base = _evaluate_cached(
            property_corpus, f"original:{row.fixture_id}", prepared
        )
        for index, function in enumerate(transforms):
            source = prepared if function in {hu_representation, round_trip} else replica
            base = original_base if source is prepared else replica_base
            surface = function(source, variant=50_000 + row.ordinal * 8 + index)
            if function is handedness_preserving:
                assert_exact_surface_relation(
                    source,
                    surface.prepared,
                    label=f"handedness:{row.fixture_id}",
                )
            elif function is translated:
                assert_exact_surface_relation(
                    source,
                    surface.prepared,
                    translation_normalization_mm=INVERSE_TRANSLATION,
                    label=f"translated:{row.fixture_id}",
                )
            elif function is hu_representation:
                assert np.array_equal(surface.prepared.first_array, source.first_array)
                assert_exact_surface_relation(
                    source,
                    surface.prepared,
                    label=f"hu_representation:{row.fixture_id}",
                )
            elif function is round_trip:
                assert_exact_surface_relation(
                    source,
                    surface.prepared,
                    label=f"round_trip:{row.fixture_id}",
                )
            observed = _evaluate_surface(surface)
            _assert_same_physical_result(base, observed)
            executions += 1
    assert executions == 912 * 4
    assert executions == 3_648
