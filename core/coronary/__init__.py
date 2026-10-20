"""Coronary centerline derivation (CORONARY_CENTERLINE_001).

Placement decision (worker-owned per the slice contract): the deterministic
coronary derivation lives under ``core/coronary/`` because every input is a
canonical core object (``core.volume.Volume``, ``FrameSpec``,
``CoordinateTransform``) and the output rides the canonical artifact model
(``DerivedArtifact`` of kind ``CENTERLINE``).  The CPR framing entry point
lives beside the MPR kernel under ``visualization/cpr/`` because its sampling
internals are the MPR kernel's, and the viewer overlay lives in
``apps/viewer/`` as an additive, grammar-preserving ADAPT.

Boundaries this package holds:

* Rule C - structure only (one polyline + arc lengths); no measurement.
* Rule D - no LLM, no model, no RNG; pure deterministic NumPy.
* Producer matrix - the vessel mask is derivation EVIDENCE (parameters_ref +
  payload digests); no ``SEGMENTATION`` artifact is ever written here.
* F18 - non-HU safety is inherited from the existing viewer gate; this
  package mints no domain refusal and never re-implements it.

The public surface is re-exported below so downstream modules import from
``core.coronary`` only.
"""

from __future__ import annotations

from core.coronary.centerline import (
    CENTERLINE_REF_PREFIX,
    CORONARY_ACTOR_ID,
    VESSEL_ASSIGNMENT_SCHEMA,
    VESSEL_ASSIGNMENT_SOURCES,
    VESSEL_LABELS,
    CenterlineBuildResult,
    CoronaryDecision,
    CoronaryDecisionOutcome,
    CoronaryPath,
    apply_coronary_decision,
    assert_centerline_coordinates,
    build_centerline,
    centerline_provenance,
    centerline_ref,
    derive_coronary_path,
    frame_spec_ref,
    validate_centerline_payload,
    validate_vessel_assignment,
    vessel_assignment_provenance,
    world_point_of,
)
from core.coronary.response_cache import (
    CACHE_SCHEMA_VERSION,
    SOURCE_DIGEST_KINDS,
    CachedVesselness,
    ResponseCacheStore,
    StoredVesselness,
    VesselnessCacheKey,
    cache_key_for,
    cached_frangi_vesselness,
    source_content_sha256,
)
from core.coronary.thinning import (
    exact_edt,
    orient_skeleton_path,
    thinning_skeleton,
    trace_skeleton,
)
from core.coronary.types import (
    CENTERLINE_SCHEMA,
    CORONARY_ALGORITHM,
    CORONARY_ALGORITHM_VERSION,
    CORONARY_CLAIM_ID,
    CORONARY_CLAIM_VERSION,
    MASK_EVIDENCE_CODEC,
    VOLUME_ORDERING,
    CoronaryDiagnostics,
    CoronaryRefusalCode,
    CoronaryValidationError,
    Seed,
    VesselnessParameters,
)
from core.coronary.vesselness import RESPONSE_DTYPE, frangi_vesselness, vessel_mask

__all__ = [
    "CACHE_SCHEMA_VERSION",
    "CENTERLINE_REF_PREFIX",
    "CENTERLINE_SCHEMA",
    "CORONARY_ACTOR_ID",
    "CORONARY_ALGORITHM",
    "CORONARY_ALGORITHM_VERSION",
    "CORONARY_CLAIM_ID",
    "CORONARY_CLAIM_VERSION",
    "MASK_EVIDENCE_CODEC",
    "RESPONSE_DTYPE",
    "SOURCE_DIGEST_KINDS",
    "VESSEL_ASSIGNMENT_SCHEMA",
    "VESSEL_ASSIGNMENT_SOURCES",
    "VESSEL_LABELS",
    "VOLUME_ORDERING",
    "CachedVesselness",
    "CenterlineBuildResult",
    "CoronaryDecision",
    "CoronaryDecisionOutcome",
    "CoronaryDiagnostics",
    "CoronaryPath",
    "CoronaryRefusalCode",
    "CoronaryValidationError",
    "ResponseCacheStore",
    "Seed",
    "StoredVesselness",
    "VesselnessParameters",
    "VesselnessCacheKey",
    "apply_coronary_decision",
    "assert_centerline_coordinates",
    "build_centerline",
    "cache_key_for",
    "cached_frangi_vesselness",
    "centerline_provenance",
    "centerline_ref",
    "derive_coronary_path",
    "exact_edt",
    "frame_spec_ref",
    "frangi_vesselness",
    "orient_skeleton_path",
    "source_content_sha256",
    "thinning_skeleton",
    "trace_skeleton",
    "validate_centerline_payload",
    "validate_vessel_assignment",
    "vessel_assignment_provenance",
    "vessel_mask",
    "world_point_of",
]
