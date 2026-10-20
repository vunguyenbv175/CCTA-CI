"""Frozen deterministic QC vocabulary, profiles, and classification rules.

The values in this module are the implementation of QC_001_SPEC §3.  They
are deliberately data, rather than scattered literals in the individual
family functions, so a parameter digest is a complete description of the
engine that produced a result.
"""

from __future__ import annotations

import enum
import math
from typing import Final, cast

from core.dicom.qc import QC_FLAGS as CLOSED_QC_FLAGS
from core.digest import canonical_digest


class QCOutcome(enum.StrEnum):
    """One deterministic family outcome."""

    CLEAR = "CLEAR"
    MILD = "MILD"
    SEVERE = "SEVERE"
    ABSTAINED = "ABSTAINED"


class QCAbstentionCode(enum.StrEnum):
    """Closed typed-refusal vocabulary from QC_001_SPEC §3.1."""

    QC_INPUT_SURFACE_INVALID = "QC_INPUT_SURFACE_INVALID"
    QC_SCALAR_DOMAIN_INVALID = "QC_SCALAR_DOMAIN_INVALID"
    QC_GEOMETRY_INVALID = "QC_GEOMETRY_INVALID"
    QC_SOURCE_IDENTITY_INVALID = "QC_SOURCE_IDENTITY_INVALID"
    QC_REQUIRED_PAIR_ABSENT = "QC_REQUIRED_PAIR_ABSENT"
    QC_REQUIRED_PAIR_INVALID = "QC_REQUIRED_PAIR_INVALID"
    QC_REGISTRATION_INVALID = "QC_REGISTRATION_INVALID"
    QC_INSUFFICIENT_SUPPORT = "QC_INSUFFICIENT_SUPPORT"
    QC_DYNAMIC_RANGE_INVALID = "QC_DYNAMIC_RANGE_INVALID"
    QC_CLIPPING_UNRESOLVABLE = "QC_CLIPPING_UNRESOLVABLE"
    QC_COVERAGE_ENVELOPE_INVALID = "QC_COVERAGE_ENVELOPE_INVALID"
    QC_BODY_OCCUPANCY_INDETERMINATE = "QC_BODY_OCCUPANCY_INDETERMINATE"


FIRST_NINE_FLAGS: Final[tuple[str, ...]] = (
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
MODEL_DISAGREEMENT_FLAG: Final[str] = "MODEL_DISAGREEMENT"
ABSTAINED_FLAG: Final[str] = "ABSTAINED"
QC_FLAGS: Final[tuple[str, ...]] = tuple(CLOSED_QC_FLAGS)
assert QC_FLAGS == FIRST_NINE_FLAGS + (MODEL_DISAGREEMENT_FLAG, ABSTAINED_FLAG)

ALGORITHM_VERSION: Final[str] = "1"
ASSESSMENT_ALGORITHM_ID: Final[str] = "qc-assessment-v1"
ASSESSMENT_PARAMETER_SCHEMA: Final[str] = "ccta.qc.assessment_parameters.v1"
FAMILY_PARAMETER_SCHEMA: Final[str] = "ccta.qc.parameter_profile.v1"
PAYLOAD_SCHEMA: Final[str] = "ccta.payload.qc_assessment.v1"
REGISTRATION_PAYLOAD_SCHEMA: Final[str] = "ccta.payload.qc_registration.v1"
QC_PREPROCESSING: Final[str] = "QC_INPUT_NORMALIZATION_V1"
ARITHMETIC_RULE: Final[str] = "IEEE754_BINARY64_RNE_NO_FMA_V1"
QUANTILE_RULE: Final[str] = "LOWER_ORDER_STATISTIC_FLOOR_P_N_MINUS_1_V1"

FAMILY_ALGORITHM_IDS: Final[dict[str, str]] = {
    flag: f"qc-{flag.lower().replace('_', '-')}-v1" for flag in FIRST_NINE_FLAGS
}
FAMILY_METRIC_NAMES: Final[dict[str, str]] = {
    "CARDIAC_MOTION": "cardiac_pair_edge_residual_score",
    "RESPIRATORY_MOTION": "respiratory_slice_residual_score",
    "STAIR_STEP": "stair_edge_jaccard_discontinuity",
    "POOR_CONTRAST": "regional_contrast_to_noise",
    "EXCESSIVE_NOISE": "multiscale_residual_noise_score",
    "BLOOMING": "high_density_halo_ratio",
    "BEAM_HARDENING": "slice_cupping_streak_score",
    "MISREGISTRATION": "registered_pair_edge_residual_score",
    "INCOMPLETE_ANATOMY": "coverage_extent_occupancy_deficit",
}

# These objects are intentionally written in the same closed shape as the
# specification.  ``canonical_digest`` sorts object keys when binding them;
# preserving the literal values here makes the manifest inspectable as well.
FAMILY_LITERALS: Final[dict[str, dict[str, object]]] = {
    "CARDIAC_MOTION": {
        "central_u_abs": 0.35,
        "central_v_abs": 0.35,
        "central_w_abs": 0.75,
        "edge_gradient_percentile": 0.90,
        "edge_gradient_statistic": "P95",
        "edge_min_voxels": 32,
        "residual_statistic": "MEDIAN",
        "residual_normalization": "S_PAIR",
        "residual_weight": 0.75,
        "sharpness_weight": 0.25,
        "sharpness_ratio_epsilon": 1e-12,
        "slice_max_weight": 0.50,
        "slice_p95_weight": 0.50,
        "mild_threshold": 0.08,
        "severe_threshold": 0.16,
        "minimum_overlap": 0.80,
        "pair_grid": "FIRST_VOLUME_GRID_PHYSICAL_OVERLAP_V1",
        "pair_resample_arithmetic": "FLOAT32_LEFT_ASSOCIATIVE_NO_FMA_V1",
        "pair_overlap_population": "CENTRAL_MASK_C",
    },
    "RESPIRATORY_MOTION": {
        "central_u_abs": 0.35,
        "central_v_abs": 0.35,
        "central_w_abs": 0.75,
        "edge_gradient_percentile": 0.90,
        "edge_gradient_statistic": "P95",
        "edge_min_voxels": 32,
        "residual_statistic": "MEDIAN",
        "residual_normalization": "S",
        "slice_max_weight": 0.50,
        "slice_p95_weight": 0.50,
        "mild_threshold": 0.08,
        "severe_threshold": 0.16,
        "minimum_adjacent_pairs": 2,
        "comparison_grid": "PHYSICAL_INTERSECTION_FIRST_SLICE_V1",
        "comparison_arithmetic": "FLOAT32_LEFT_ASSOCIATIVE_NO_FMA_V1",
        "comparison_edge_population": "FIRST_SLICE_EDGE_INTERSECTION",
    },
    "STAIR_STEP": {
        "central_u_abs": 0.35,
        "central_v_abs": 0.35,
        "central_w_abs": 0.75,
        "edge_gradient_percentile": 0.90,
        "edge_min_voxels": 32,
        "edge_set_operation": "JACCARD",
        "connectivity": 6,
        "slice_max_weight": 0.50,
        "slice_p95_weight": 0.50,
        "mild_threshold": 0.15,
        "severe_threshold": 0.30,
        "mild_repeated_fraction": 0.25,
        "severe_repeated_fraction": 0.50,
        "repeated_count_rounding": "CEIL",
        "minimum_adjacent_pairs": 2,
        "comparison_grid": "PHYSICAL_INTERSECTION_FIRST_SLICE_V1",
        "comparison_arithmetic": "FLOAT32_LEFT_ASSOCIATIVE_NO_FMA_V1",
    },
    "POOR_CONTRAST": {
        "central_u_abs_max": 0.30,
        "central_v_abs_max": 0.30,
        "central_w_abs_max": 0.75,
        "reference_radius_min": 0.45,
        "reference_radius_max": 0.75,
        "reference_radius_definition": "MAX_ABS_UV",
        "mask_min_voxels": 64,
        "clip_fraction_max": 0.01,
        "clip_endpoint_rule": "SOURCE_REPRESENTABLE_ENDPOINTS_AFFINE_RESCALE_V1",
        "cnr_denominator": "SQRT_MAD_SUM_SQUARES_PLUS_NORMALIZED_EPSILON",
        "epsilon": 1e-12,
        "cnr_mild_boundary": 2.0,
        "cnr_severe_boundary": 1.0,
    },
    "EXCESSIVE_NOISE": {
        "central_u_abs": 0.30,
        "central_v_abs": 0.30,
        "central_w_abs": 0.75,
        "homogeneous_gradient_percentile": 0.50,
        "homogeneous_gradient_statistic": "Q_FLOOR",
        "homogeneous_min_voxels": 512,
        "box_radii": [1, 2],
        "box_boundary": "IN_BOUNDS_ACTUAL_COUNT",
        "local_variance_radius": 1,
        "local_variance_statistic": "SQRT_Q95",
        "residual_weight": 0.50,
        "variance_weight": 0.50,
        "mild_threshold": 0.08,
        "severe_threshold": 0.15,
    },
    "BLOOMING": {
        "central_u_abs": 0.30,
        "central_v_abs": 0.30,
        "central_w_abs": 0.75,
        "central_min_voxels": 64,
        "candidate_population": "WHOLE_VOLUME_FINITE",
        "candidate_percentile": 0.99,
        "candidate_zscore": 5.0,
        "candidate_zscore_scale": 1.4826,
        "candidate_threshold_operator": "MAX",
        "dynamic_range": "Q95_MINUS_Q05",
        "connectivity": 6,
        "candidate_min_voxels": 8,
        "profile_level_low": 0.50,
        "profile_level_high": 0.90,
        "shell_radius": 1,
        "shell_definition": "EXTERIOR_6_NEIGHBOR_C",
        "component_max_weight": 0.50,
        "component_p95_weight": 0.50,
        "mild_threshold": 0.25,
        "severe_threshold": 0.50,
    },
    "BEAM_HARDENING": {
        "body_low_percentile": 0.10,
        "body_high_percentile": 0.90,
        "body_range_fraction": 0.05,
        "body_population": "IN_PLANE_RHO_LE_0.85",
        "body_radius_max": 0.85,
        "central_radius_max": 0.25,
        "annulus_radius_min": 0.45,
        "annulus_radius_max": 0.70,
        "central_min_voxels": 64,
        "annulus_min_voxels": 64,
        "sector_min_voxels": 8,
        "sector_count": 8,
        "slice_range_low_percentile": 0.05,
        "slice_range_high_percentile": 0.95,
        "slice_range_population": "ALL_FINITE_IN_PLANE_VALUES",
        "supported_slice_fraction": 0.80,
        "supported_count_rounding": "CEIL",
        "clip_fraction_max": 0.01,
        "clip_population": "BODY_MASK_CENTER_PLUS_ANNULUS",
        "slice_max_weight": 0.50,
        "slice_p95_weight": 0.50,
        "mild_threshold": 0.05,
        "severe_threshold": 0.10,
    },
    "MISREGISTRATION": {
        "central_u_abs": 0.35,
        "central_v_abs": 0.35,
        "central_w_abs": 0.75,
        "edge_gradient_percentile": 0.90,
        "edge_min_voxels": 32,
        "residual_percentile": 0.95,
        "residual_statistic": "Q_FLOOR",
        "residual_normalization": "S_PAIR",
        "slice_max_weight": 0.50,
        "slice_p95_weight": 0.50,
        "minimum_overlap": 0.80,
        "mild_threshold": 0.10,
        "severe_threshold": 0.20,
        "pair_grid": "FIRST_VOLUME_GRID_PHYSICAL_OVERLAP_V1",
        "pair_resample_arithmetic": "FLOAT32_LEFT_ASSOCIATIVE_NO_FMA_V1",
        "pair_overlap_population": "CENTRAL_MASK_C",
    },
    "INCOMPLETE_ANATOMY": {
        "inferior_extent_mm": 80.0,
        "superior_extent_mm": 80.0,
        "terminal_band_mm": 16.0,
        "minimum_body_occupancy": 0.20,
        "body_low_percentile": 0.10,
        "body_high_percentile": 0.90,
        "body_range_fraction": 0.05,
        "body_population": "WHOLE_VOLUME_FINITE",
        "minimum_terminal_slice_support": 1,
        "mild_deficit_boundary": 0.25,
        "severe_deficit_boundary": 0.25,
        "boundary_rule": "INCLUSIVE_VOXEL_CENTER_V1",
        "terminal_band_rule": "INCLUSIVE_ACTUAL_EXTREME_BOUNDS_V1",
        "orientation": "SUPERIOR_POSITIVE",
    },
}


def family_parameter_object(flag: str) -> dict[str, object]:
    """Return the complete closed parameter object for one family."""

    if flag not in FIRST_NINE_FLAGS:
        raise ValueError(f"not a deterministic QC family: {flag!r}")
    return {
        "schema_version": FAMILY_PARAMETER_SCHEMA,
        "algorithm_id": FAMILY_ALGORITHM_IDS[flag],
        "algorithm_version": ALGORITHM_VERSION,
        "parameters": {
            "common_arithmetic": ARITHMETIC_RULE,
            "common_quantile": QUANTILE_RULE,
            "family_literals": dict(FAMILY_LITERALS[flag]),
        },
    }


def family_parameter_digest(flag: str) -> str:
    """Content digest of one family parameter object."""

    return canonical_digest(family_parameter_object(flag))


def assessment_parameter_manifest() -> dict[str, object]:
    """Build the nine-family assessment manifest in frozen family order."""

    return {
        "schema_version": ASSESSMENT_PARAMETER_SCHEMA,
        "algorithm_id": ASSESSMENT_ALGORITHM_ID,
        "algorithm_version": ALGORITHM_VERSION,
        "arithmetic": ARITHMETIC_RULE,
        "flag_parameters": [
            {
                "algorithm_id": FAMILY_ALGORITHM_IDS[flag],
                "algorithm_version": ALGORITHM_VERSION,
                "flag": flag,
                "parameter_digest": family_parameter_digest(flag),
                "parameters": dict(
                    cast(dict[str, object], family_parameter_object(flag)["parameters"])
                ),
            }
            for flag in FIRST_NINE_FLAGS
        ],
    }


def assessment_parameter_digest() -> str:
    return canonical_digest(assessment_parameter_manifest())


def _finite(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def classify_high_bad(
    metric: float,
    mild_threshold: float,
    severe_threshold: float,
) -> QCOutcome:
    """Classify a finite high-is-bad metric using the frozen inequalities."""

    if not _finite(metric):
        raise ValueError("QC metric must be finite")
    if not _finite(mild_threshold) or not _finite(severe_threshold):
        raise ValueError("QC thresholds must be finite")
    if float(mild_threshold) >= float(severe_threshold):
        raise ValueError("high-bad thresholds must be ordered mild < severe")
    value = float(metric)
    if value < float(mild_threshold):
        return QCOutcome.CLEAR
    if value < float(severe_threshold):
        return QCOutcome.MILD
    return QCOutcome.SEVERE


def classify_low_bad(
    metric: float,
    severe_threshold: float,
    mild_threshold: float,
) -> QCOutcome:
    """Classify a finite low-is-bad metric using the frozen inequalities."""

    if not _finite(metric):
        raise ValueError("QC metric must be finite")
    if not _finite(severe_threshold) or not _finite(mild_threshold):
        raise ValueError("QC thresholds must be finite")
    if float(severe_threshold) >= float(mild_threshold):
        raise ValueError("low-bad thresholds must be ordered severe < mild")
    value = float(metric)
    if value >= float(mild_threshold):
        return QCOutcome.CLEAR
    if value >= float(severe_threshold):
        return QCOutcome.MILD
    return QCOutcome.SEVERE


def classify_stair_step(
    score: float,
    *,
    supported_mild_count: int,
    supported_severe_count: int,
    pair_count: int,
) -> QCOutcome:
    """Apply the STAIR_STEP score and repeated-fraction predicates."""

    if not _finite(score) or pair_count < 1:
        raise ValueError("STAIR_STEP classifier requires a finite score and pairs")
    if supported_mild_count < 0 or supported_severe_count < 0:
        raise ValueError("repeated counts cannot be negative")
    mild_needed = math.ceil(0.25 * pair_count)
    severe_needed = math.ceil(0.50 * pair_count)
    if score < 0.15 or supported_mild_count < mild_needed:
        return QCOutcome.CLEAR
    if score >= 0.30 and supported_severe_count >= severe_needed:
        return QCOutcome.SEVERE
    return QCOutcome.MILD


def classify_incomplete_anatomy(score: float) -> QCOutcome:
    """Classify the non-negative coverage deficit metric."""

    if not _finite(score) or float(score) < 0.0:
        raise ValueError("coverage deficit must be a finite non-negative metric")
    if score == 0.0:
        return QCOutcome.CLEAR
    if score < 0.25:
        return QCOutcome.MILD
    return QCOutcome.SEVERE


__all__ = [
    "ABSTAINED_FLAG",
    "ALGORITHM_VERSION",
    "ARITHMETIC_RULE",
    "ASSESSMENT_ALGORITHM_ID",
    "ASSESSMENT_PARAMETER_SCHEMA",
    "FAMILY_ALGORITHM_IDS",
    "FAMILY_LITERALS",
    "FAMILY_METRIC_NAMES",
    "FAMILY_PARAMETER_SCHEMA",
    "FIRST_NINE_FLAGS",
    "MODEL_DISAGREEMENT_FLAG",
    "PAYLOAD_SCHEMA",
    "QCAbstentionCode",
    "QC_FLAGS",
    "QCOutcome",
    "QC_PREPROCESSING",
    "QUANTILE_RULE",
    "REGISTRATION_PAYLOAD_SCHEMA",
    "assessment_parameter_digest",
    "assessment_parameter_manifest",
    "classify_high_bad",
    "classify_incomplete_anatomy",
    "classify_low_bad",
    "classify_stair_step",
    "family_parameter_digest",
    "family_parameter_object",
]
