"""The real-QC consumer gate (QC_001_GATE).

A consumer calls this gate BEFORE reading input pixels or invoking analysis.
The gate resolves effective real-QC coverage for every canonical ref the
consumer declares, selects the single canonical reason by the closed
precedence, and returns exactly ``PROCEED`` or ``ABSTAIN``.

Design boundaries (contract QC_001_GATE):

* The canonical reason set stays the closed four values of ``QCReason``.
  ``QC_SCHEMA_INCOMPATIBLE``, ``QC_SCHEMA_INVALID`` and
  ``QC_LEGACY_PLACEHOLDER`` are diagnostic EVENT codes carried on the result;
  they are NOT enum members and never widen ``QCReason``.
* This module CALLS the accepted upstream machinery and never modifies it:
  the #16 study-level selector
  (``core.qc.human.select_effective_assessment``, deferred binding — see
  :func:`_select_effective_assessment`) and the DET-local eligibility policy
  (``core.qc.policy.RealQCEligibilityPolicy``, which owns producer identity,
  producer/payload agreement, outer kind/schema binding, grade/flag matrix,
  ``validate_result_bundle`` and provenance-chain verification).  Selection
  ownership is gate-scoped: :func:`_select_effective_for_ref` adds only
  per-consumed-ref iteration around the accepted study-level selector.
* Coverage is mechanical.  Timestamps, file names, insertion order,
  confidence and the most favorable grade are never used to select coverage.
* v1 identifier admission is EMPTY; a v2 vessel/region token is reserved only
  and must match a v2 assessment's scope and identifier exactly.
* Fail-closed everywhere: any candidate that cannot be proven eligible is
  refused with a visible diagnostic, never silently dropped and never passed.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, cast

from core.dicom.qc import (
    QCAssessmentArtifact,
    QCGateDecision,
    QCGateResult,
    QCReason,
    QCScope,
)
from core.enums import QualityGrade
from core.provenance import NotApplicable, ProvenanceRecord, verify_chain

from .constants import assessment_parameter_digest, family_parameter_digest
from .types import QCPolicyDiagnostic, QCValidationError

#: The canonical subject-ref grammar of the closed v1 seam.
_CANONICAL_REF: Final[re.Pattern[str]] = re.compile(
    r"\A(study_id|series_id|volume_id|mpr_artifact_id):"
    r"[0123456789ABCDEFGHJKMNPQRSTVWXYZ]{26}\Z"
)

#: The finite v2 vessel/region token grammar.  The frozen v2 tokens carry the
#: prefixed form (``VESSEL:LAD``, ``REGION:HEART``), so the grammar admits the
#: colon between the prefix and the token; it still admits exactly one finite
#: token — no sets, no wildcards, no whitespace.
_V2_TOKEN: Final[re.Pattern[str]] = re.compile(r"\A(?:VESSEL|REGION):[A-Z0-9_]{1,32}\Z")


#: Regexes for the v2 grammars the gate validates itself.
_PROVENANCE_REF: Final[re.Pattern[str]] = re.compile(
    r"\Aprovenance:[0123456789ABCDEFGHJKMNPQRSTVWXYZ]{26}\Z"
)
_SHA256_HEX: Final[re.Pattern[str]] = re.compile(r"\A[0-9a-f]{64}\Z")

#: The canonical v1 evidence-ref forms (spec 51, inherited by v2 at 166/183):
#: provenance, source_manifest, volume_id, and mpr_artifact_id refs.  Garbage
#: strings are not evidence refs; valid volume/MPR refs ARE evidence refs.
_EVIDENCE_REF: Final[re.Pattern[str]] = re.compile(
    r"\A(?:source_manifest:[0-9a-f]{64}"
    r"|provenance:[0123456789ABCDEFGHJKMNPQRSTVWXYZ]{26}"
    r"|volume_id:[0123456789ABCDEFGHJKMNPQRSTVWXYZ]{26}"
    r"|mpr_artifact_id:[0123456789ABCDEFGHJKMNPQRSTVWXYZ]{26})\Z"
)

#: The canonical outer QC artifact-ref grammar (core/dicom/qc.py:187).
_QC_ARTIFACT_REF: Final[re.Pattern[str]] = re.compile(
    r"\Aqc_artifact:[0123456789ABCDEFGHJKMNPQRSTVWXYZ]{26}\Z"
)


def _canonical_subject_order(refs: Sequence[str]) -> tuple[str, ...]:
    """The canonical subject-ref order (kind rank, then id) — spec 173."""

    return tuple(
        sorted(
            refs,
            key=lambda ref: (_SUBJECT_KIND_ORDER[ref.split(":", 1)[0]], ref.split(":", 1)[1]),
        )
    )


def _select_effective_assessment(
    study_ref: str, candidates: Sequence[object]
) -> QCAssessmentArtifact | None:
    """CALL the accepted #16 study-level selector (never modified).

    The import is deferred because ``core.qc.human`` derives its admission
    whitelists by importing every ``core.qc`` submodule, including this one
    (``core/qc/human.py:331-358``).  A module-level import would therefore be
    circular.  Deferring the binding keeps the accepted selector authoritative
    while leaving ``human.py`` byte-identical.
    """

    from .human import select_effective_assessment

    return cast(
        "QCAssessmentArtifact | None", select_effective_assessment(study_ref, candidates)
    )


def _validate_v1_candidate(
    consumed_refs: tuple[str, ...],
    candidates: Sequence[object],
    *,
    consumer: str,
    lineage: Mapping[str, object],
) -> tuple[tuple[QCAssessmentArtifact, ...], tuple[QCPolicyDiagnostic, ...]]:
    """CALL the accepted DET-local eligibility policy for one candidate.

    The policy (``core/qc/policy.py:169-230``) owns producer identity,
    producer/payload agreement, outer kind/schema binding, grade/flag matrix
    validation, ``validate_result_bundle``, and provenance-chain verification.
    The gate CALLS it through a ``RealQCEligibilityPolicy`` instance; it never
    duplicates or relaxes those rules.

    The accepted policy conflates per-candidate validation with its own
    study-level selection, while spec 2.2 assigns SELECTION to the gate-owned
    adapter.  The gate therefore uses the policy for what it owns —
    per-candidate validation — and keeps graph resolution local: a candidate
    is validated when the policy raised no *validation-level* diagnostic
    against it.  Relation-level codes (``QC_SUPERSESSION_INVALID``,
    ``QC_CONFLICT``) are selection outcomes and do not make a candidate
    ineligible; the gate's own graph rules decide those, which is what lets a
    valid supersession keep its predecessor available for chain resolution.

    Any candidate the policy cannot prove valid is refused with its visible
    diagnostic — never a silent pass.
    """

    from .policy import RealQCEligibilityPolicy

    policy = RealQCEligibilityPolicy(consumer_schema="v1")
    try:
        result = policy.evaluate(
            consumed_refs,
            candidates,
            consumer=consumer,
            subject_lineage=lineage,
        )
    except (TypeError, ValueError, AttributeError, QCValidationError) as exc:
        # A structurally forged candidate (e.g. a string where the policy
        # expects an enum) fails closed: nothing is eligible and the fault
        # is visible.
        return (), (
            _diagnostic(
                DIAG_SCHEMA_INVALID, detail=f"candidate cannot be validated: {exc}"
            ),
        )
    # A candidate is valid when the policy raised no VALIDATION-level
    # diagnostic naming it.  Relation-level codes describe selection outcomes
    # the gate's own graph rules own (see the docstring above).
    diagnosed: set[str] = set()
    for diagnostic in result.diagnostics:
        if diagnostic.code in _RELATION_LEVEL_CODES:
            continue
        if diagnostic.artifact_ref is None:
            # A candidate-less diagnostic indicts the whole candidate set.
            return (), tuple(result.diagnostics)
        diagnosed.add(diagnostic.artifact_ref)
    validated: list[QCAssessmentArtifact] = []
    for candidate in candidates:
        assessment = _assessment_of(candidate)
        if assessment is None:
            continue
        if assessment.artifact_ref in diagnosed:
            continue
        validated.append(assessment)
    return tuple(validated), tuple(result.diagnostics)


#: The exact schema capability a consumer may declare (spec 2.1).
CONSUMER_SCHEMAS: Final[tuple[str, ...]] = ("v1", "v2")

#: The observed schema of a v2 assessment record.
V2_PAYLOAD_SCHEMA: Final[str] = "ccta.payload.qc_assessment.v2"

#: The closed v1 payload schema (mirrors ``core.dicom.qc.QC_PAYLOAD_SCHEMA``).
V1_PAYLOAD_SCHEMA: Final[str] = "ccta.payload.qc_assessment.v1"

#: The exact reserved v2 anatomic-identifier schema (spec §1.3).
V2_IDENTIFIER_SCHEMA: Final[str] = "ccta.qc.anatomic_identifier.v2"

#: The exact fifteen admitted v2 payload keys (spec §1.3).  Missing, duplicate,
#: unknown, and extra keys are all rejected.
V2_REQUIRED_KEYS: Final[tuple[str, ...]] = (
    "schema_version",
    "scope",
    "subject_refs",
    "quality_grade",
    "confidence",
    "flags",
    "vessel_id",
    "region_id",
    "producer",
    "produced_by_ref",
    "supersedes_ref",
    "justification_code",
    "evidence_refs",
    "anatomic_identifier_schema",
    "flag_evidence",
)

#: The exact FlagEvidence keys (spec 210).  No unknown keys.
FLAG_EVIDENCE_KEYS: Final[tuple[str, ...]] = (
    "flag",
    "outcome",
    "metric_name",
    "metric_value",
    "threshold_profile",
    "metric_unit",
    "algorithm_id",
    "algorithm_version",
    "parameter_digest",
    "evidence_refs",
    "abstention_code",
)

#: The ten frozen flags a v2 ``flag_evidence`` tuple must carry, in frozen
#: order, with ``MODEL_DISAGREEMENT`` in position ten (spec 227).
FIRST_TEN_FLAGS: Final[tuple[str, ...]] = (
    "CARDIAC_MOTION",
    "RESPIRATORY_MOTION",
    "STAIR_STEP",
    "POOR_CONTRAST",
    "EXCESSIVE_NOISE",
    "BLOOMING",
    "BEAM_HARDENING",
    "MISREGISTRATION",
    "INCOMPLETE_ANATOMY",
    "MODEL_DISAGREEMENT",
)

#: The frozen finite v2 vessel identifier set (spec 190).  Prefixed form
#: only; a bare ``LAD`` is not a token and an arbitrary ``BANANA`` is not a
#: token.  Adding or reinterpreting a token requires a further schema and
#: acceptance decision.
V2_VESSEL_IDS: Final[frozenset[str]] = frozenset(
    {
        "VESSEL:LM",
        "VESSEL:LAD",
        "VESSEL:LCX",
        "VESSEL:RCA",
        "VESSEL:D1",
        "VESSEL:D2",
        "VESSEL:OM1",
        "VESSEL:OM2",
        "VESSEL:PDA",
        "VESSEL:PLB",
    }
)

#: The frozen finite v2 region identifier set (spec 197).
V2_REGION_IDS: Final[frozenset[str]] = frozenset(
    {
        "REGION:HEART",
        "REGION:CORONARY_ORIGIN",
        "REGION:CARDIAC_BASE",
        "REGION:CARDIAC_APEX",
        "REGION:MEDIASTINUM",
    }
)

#: The finite admission set a v2-aware request may carry (spec 303/305): the
#: union of the two frozen sets.  A token is admitted by MEMBERSHIP, not by
#: grammar — ``VESSEL:BANANA`` is not admitted (spec 187/203).
V2_REQUEST_TOKENS: Final[frozenset[str]] = V2_VESSEL_IDS | V2_REGION_IDS

#: The closed confidence vocabulary (spec 175).  ``NOT_ANALYZABLE`` confidence
#: pairs with the ``NOT_ANALYZABLE`` grade and is not a gate override.
V2_CONFIDENCE_VALUES: Final[frozenset[str]] = frozenset(
    {"HIGH", "MEDIUM", "LOW", "UNRELIABLE", "NOT_ANALYZABLE"}
)

#: The closed typed-refusal vocabulary for ``abstention_code`` (spec 224 with
#: ``QCAbstentionCode``, spec 3.1).  An unknown code is a schema error, never
#: an accepted abstention.
V2_ABSTENTION_CODES: Final[frozenset[str]] = frozenset(
    {
        "QC_INPUT_SURFACE_INVALID",
        "QC_SCALAR_DOMAIN_INVALID",
        "QC_GEOMETRY_INVALID",
        "QC_SOURCE_IDENTITY_INVALID",
        "QC_REQUIRED_PAIR_ABSENT",
        "QC_REQUIRED_PAIR_INVALID",
        "QC_REGISTRATION_INVALID",
        "QC_INSUFFICIENT_SUPPORT",
        "QC_DYNAMIC_RANGE_INVALID",
        "QC_CLIPPING_UNRESOLVABLE",
        "QC_COVERAGE_ENVELOPE_INVALID",
        "QC_BODY_OCCUPANCY_INDETERMINATE",
    }
)

#: The eleven-value flag vocabulary in canonical order (spec 176): the first
#: ten frozen flags plus the assessment-level ``ABSTAINED`` representation.
QC_FLAGS: Final[tuple[str, ...]] = FIRST_TEN_FLAGS + ("ABSTAINED",)

#: The canonical subject-ref kind order (spec 173 via the v1 rule).
_SUBJECT_KIND_ORDER: Final[dict[str, int]] = {
    "study_id": 0,
    "series_id": 1,
    "volume_id": 2,
    "mpr_artifact_id": 3,
}

#: The closed metric map used by ``FlagEvidence`` (spec 245-258): the metric
#: name is fixed by family and cannot be free text.
V2_METRIC_NAMES: Final[dict[str, str]] = {
    "CARDIAC_MOTION": "cardiac_pair_edge_residual_score",
    "RESPIRATORY_MOTION": "respiratory_slice_residual_score",
    "STAIR_STEP": "stair_edge_jaccard_discontinuity",
    "POOR_CONTRAST": "regional_contrast_to_noise",
    "EXCESSIVE_NOISE": "multiscale_residual_noise_score",
    "BLOOMING": "high_density_halo_ratio",
    "BEAM_HARDENING": "slice_cupping_streak_score",
    "MISREGISTRATION": "registered_pair_edge_residual_score",
    "INCOMPLETE_ANATOMY": "coverage_extent_occupancy_deficit",
    "MODEL_DISAGREEMENT": "model_structure_disagreement",
}

#: The metric unit is fixed by the metric map (spec 245).
V2_METRIC_UNITS: Final[dict[str, str]] = {
    "CARDIAC_MOTION": "DIMENSIONLESS",
    "RESPIRATORY_MOTION": "DIMENSIONLESS",
    "STAIR_STEP": "DIMENSIONLESS",
    "POOR_CONTRAST": "DIMENSIONLESS",
    "EXCESSIVE_NOISE": "DIMENSIONLESS",
    "BLOOMING": "DIMENSIONLESS",
    "BEAM_HARDENING": "DIMENSIONLESS",
    "MISREGISTRATION": "DIMENSIONLESS",
    "INCOMPLETE_ANATOMY": "DIMENSIONLESS",
    "MODEL_DISAGREEMENT": "DIMENSIONLESS",
}

#: The ten exact family ids (spec 215); ``algorithm_id`` must equal
#: ``threshold_profile`` and the family id.
V2_FAMILY_IDS: Final[dict[str, str]] = {
    flag: f"qc-{flag.lower().replace('_', '-')}-v1" for flag in FIRST_TEN_FLAGS
}

#: R8-F3 (spec 226/229-230, 953-955): position ten of the frozen v2
#: flag_evidence order is RESERVED for the later accepted model contract.
#: QC_001 owns NO model-comparator parameter object — the gate neither
#: invents one nor resolves one — and QC_001-MODEL is declared-not-executed
#: (spec 953-955), so a v2 record's position-ten ``parameter_digest`` can
#: never resolve here.  Every parseable v2 record is therefore fail-closed
#: INELIGIBLE with this visible diagnostic (a diagnostic EVENT code, never a
#: new ``QCReason`` member); the structural v2 diagnostics coexist with it.
_RESERVED_POSITION_TEN_DETAIL: Final[str] = (
    "flag_evidence position ten (MODEL_DISAGREEMENT) is reserved for the later "
    "accepted model contract (spec 226/229-230): no accepted "
    "qc-model-disagreement-v1 parameter object exists in QC_001 and "
    "QC_001-MODEL is declared-not-executed (spec 953-955), so the position-ten "
    "parameter_digest can never resolve and every parseable v2 record is "
    "fail-closed ineligible"
)

#: Diagnostic EVENT codes.  Visible on the gate result; never ``QCReason``
#: members and never a fallback to a broader reason.
DIAG_SCHEMA_INCOMPATIBLE: Final[str] = "QC_SCHEMA_INCOMPATIBLE"
DIAG_SCHEMA_INVALID: Final[str] = "QC_SCHEMA_INVALID"
DIAG_LEGACY_PLACEHOLDER: Final[str] = "QC_LEGACY_PLACEHOLDER"
DIAG_PROVENANCE_INVALID: Final[str] = "QC_PROVENANCE_INVALID"
DIAG_PRODUCER_INVALID: Final[str] = "QC_PRODUCER_INVALID"
DIAG_RECORD_INELIGIBLE: Final[str] = "QC_RECORD_INELIGIBLE"
DIAG_SUPERSESSION_INVALID: Final[str] = "QC_SUPERSESSION_INVALID"
DIAG_CONFLICT: Final[str] = "QC_CONFLICT"

#: The closed ineligible lifecycle statuses (spec 2.1).
INELIGIBLE_STATUSES: Final[frozenset[str]] = frozenset(
    {"SUPERSEDED", "INVALIDATED", "STALE"}
)

#: The one historical, evidence-free justification code.
PLACEHOLDER_JUSTIFICATION: Final[str] = "PLACEHOLDER_NO_QC_EVIDENCE"

#: Diagnostic codes that describe a SELECTION outcome (owned by the gate's
#: own graph rules per spec 2.2) rather than a per-candidate validation
#: failure (owned by the accepted DET-local policy).
_RELATION_LEVEL_CODES: Final[frozenset[str]] = frozenset(
    {"QC_SUPERSESSION_INVALID", "QC_CONFLICT"}
)

#: Lineage keys that carry canonical ancestor refs (spec 2.1 coverage table).
_LINEAGE_REF_KEYS: Final[tuple[str, ...]] = (
    "ancestor_refs",
    "ancestors",
    "study_ref",
    "study_id",
    "source_series_ref",
    "source_series_refs",
    "series_ref",
    "series_refs",
    "source_volume_ref",
    "source_volume_refs",
    "volume_ref",
    "volume_id",
)


# ---------------------------------------------------------------------------
# Request and result surfaces
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class QCConsumerRequest:
    """The exact consumer gate request (spec 2.1).

    There is no bypass argument.  Every consumed ref must match the canonical
    ref grammar; a v1 request admits exactly the four v1 subject kinds, and a
    v2-aware request may carry one finite vessel/region token that must match
    the assessment scope and token.
    """

    consumed_refs: tuple[str, ...]
    consumer: str
    consumer_schema: str = "v1"
    requested_identifier: str | None = None
    subject_lineage: Mapping[str, object] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        object.__setattr__(self, "consumed_refs", tuple(self.consumed_refs))
        object.__setattr__(self, "consumer", str(self.consumer))
        object.__setattr__(self, "consumer_schema", str(self.consumer_schema))
        if not self.consumed_refs:
            raise ValueError("the consumer gate requires a nonempty consumed-ref tuple")
        if len(set(self.consumed_refs)) != len(self.consumed_refs):
            raise ValueError("the consumer gate requires deduplicated consumed refs")
        # F4: every consumed ref must match the canonical ref grammar.
        for ref in self.consumed_refs:
            if _CANONICAL_REF.fullmatch(ref) is None:
                raise ValueError(
                    "the consumer gate requires canonical consumed refs "
                    f"(study_id|series_id|volume_id|mpr_artifact_id:<ULID>): {ref!r}"
                )
        if not self.consumer.strip():
            raise ValueError("the consumer gate requires a consumer identifier")
        if self.consumer_schema not in CONSUMER_SCHEMAS:
            raise ValueError("the consumer schema capability must be v1 or v2")
        # v1 identifier admission is EMPTY (spec Part 7.1); a token is only
        # meaningful for a v2-aware request.
        if self.requested_identifier is not None:
            if self.consumer_schema != "v2":
                raise ValueError("v1 identifier admission is empty; a token requires v2")
            token = str(self.requested_identifier)
            if not token.strip():
                raise ValueError("the requested identifier token must be nonempty")
            # R3-F1: the token is a MEMBER of the frozen finite sets — grammar
            # alone is insufficient.  VESSEL:BANANA and REGION:FOO match the
            # prefixed shape but are not admitted tokens (spec 187/203,
            # 303/305; the grammar check stays as the first, cheaper gate).
            if _V2_TOKEN.fullmatch(token) is None:
                raise ValueError(
                    "the requested identifier must be one finite token "
                    f"(no sets, wildcards, or separators): {token!r}"
                )
            if token not in V2_REQUEST_TOKENS:
                raise ValueError(
                    "the requested identifier must be one of the frozen finite "
                    f"vessel or region tokens: {token!r}"
                )
        if self.subject_lineage is None:
            object.__setattr__(self, "subject_lineage", {})
        else:
            object.__setattr__(self, "subject_lineage", dict(self.subject_lineage))


@dataclass(frozen=True, slots=True)
class QCConsumerGateResult:
    """The canonical gate result plus its visible diagnostic channel."""

    decision: QCGateDecision
    reason_code: QCReason | None
    assessment_refs: tuple[str, ...]
    consumer: str
    consumed_refs: tuple[str, ...]
    diagnostics: tuple[QCPolicyDiagnostic, ...] = ()
    effective_assessments: tuple[QCAssessmentArtifact, ...] = ()

    @property
    def proceed(self) -> bool:
        return self.decision == QCGateDecision.PROCEED

    @property
    def abstained(self) -> bool:
        return self.decision == QCGateDecision.ABSTAIN

    @property
    def reason(self) -> QCReason | None:
        return self.reason_code

    @property
    def diagnostic_codes(self) -> tuple[str, ...]:
        return tuple(item.code for item in self.diagnostics)

    def gate_result(self) -> QCGateResult:
        """The closed ``QCGateResult`` this decision carries."""

        return QCGateResult(self.decision, self.reason_code, self.assessment_refs)


# ---------------------------------------------------------------------------
# Coverage (mechanical; no timestamps, names, order, or confidence)
# ---------------------------------------------------------------------------


def _refs_from(value: object) -> tuple[str, ...]:
    """Extract canonical ancestor refs from one lineage entry."""

    if isinstance(value, str):
        return (value,)
    if isinstance(value, Mapping):
        values: list[str] = []
        for key in _LINEAGE_REF_KEYS:
            if key in value:
                values.extend(_refs_from(value[key]))
        return tuple(values)
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        return tuple(item for part in value for item in _refs_from(part))
    return ()


def descendant_refs(
    requested: str, lineage: Mapping[str, object]
) -> tuple[str, ...]:
    """The canonical ancestor closure of ``requested`` (excluding itself)."""

    found: set[str] = set()
    pending = list(_refs_from(lineage.get(requested)))
    while pending:
        item = pending.pop()
        if item in found or item == requested:
            continue
        found.add(item)
        pending.extend(_refs_from(lineage.get(item)))
    return tuple(sorted(found))


def covers(
    assessment: QCAssessmentArtifact, requested: str, lineage: Mapping[str, object]
) -> bool:
    """True when the assessment mechanically covers ``requested`` (spec 2.1)."""

    subjects = tuple(assessment.payload.subject_refs)
    if requested in subjects:
        return True
    # A STUDY/SERIES/VOLUME subject covers every requested descendant whose
    # lineage resolves to it; an MPR subject covers only itself.
    return any(ref in subjects for ref in descendant_refs(requested, lineage))


def _identifier_matches(assessment: QCAssessmentArtifact, token: str | None) -> bool:
    """Validate a finite v2 vessel/region token against the scope and token.

    v1 identifier admission is EMPTY: a STUDY-scoped v1 record can never
    satisfy a vessel/region token.  For a v2 record the token must match the
    assessment's scope and its exact admitted identifier.
    """

    payload = assessment.payload
    if token is None:
        return True
    if payload.scope == QCScope.STUDY:
        return False
    if payload.scope == QCScope.VESSEL:
        return payload.vessel_id == token
    if payload.scope == QCScope.REGION:
        return payload.region_id == token
    return False


# ---------------------------------------------------------------------------
# Candidate views of any admitted schema
# ---------------------------------------------------------------------------


def _coverage_key(assessment: QCAssessmentArtifact) -> tuple[object, ...]:
    payload = assessment.payload
    return (payload.scope, payload.subject_refs, payload.vessel_id, payload.region_id)


def _status_of(assessment: QCAssessmentArtifact) -> str:
    status = assessment.status
    return status.value if hasattr(status, "value") else str(status)


def _schema_of(candidate: object) -> str | None:
    """Best-effort observed schema of a candidate record."""

    payload = getattr(candidate, "payload", None)
    if payload is not None:
        schema = getattr(payload, "schema_version", None)
        if isinstance(schema, str):
            return schema
    if isinstance(candidate, Mapping):
        inner = candidate.get("payload")
        if isinstance(inner, Mapping):
            schema = inner.get("schema_version")
            if isinstance(schema, str):
                return schema
        outer = candidate.get("schema_version")
        if isinstance(outer, str):
            return outer
    return None


def _assessment_of(candidate: object) -> QCAssessmentArtifact | None:
    """The assessment view of one candidate record."""

    if isinstance(candidate, QCAssessmentArtifact):
        return candidate
    assessment = getattr(candidate, "assessment", None)
    if isinstance(assessment, QCAssessmentArtifact):
        return assessment
    return None


def _records_of(candidate: object) -> tuple[object, ...] | None:
    """Provenance records of a candidate, when it exposes them.

    Both shapes are read: a closed bundle exposes an attribute, and a v2
    mapping carries the key.  Without the Mapping branch a mapping could never
    bind ``produced_by_ref``, which is the R3-F3 defect.
    """

    records = getattr(candidate, "provenance_records", None)
    if records is None and isinstance(candidate, Mapping):
        records = candidate.get("provenance_records")
    if isinstance(records, tuple):
        return records
    if isinstance(records, Sequence) and not isinstance(records, (bytes, bytearray, str)):
        return tuple(records)
    return None


def _candidate_lineage(candidate: object) -> Mapping[str, object] | None:
    """REMOVED candidate lineage accessor (R5-F1).

    The frozen v1/v2 payload contains NO candidate lineage field; subject
    resolution must use the REQUEST's caller-supplied ``subject_lineage``.
    Reading a candidate-declared mapping let a forged record vouch for its
    own subjects (the R5-F1 defect), so this accessor now always returns
    ``None``; it is kept only so stale call sites fail loudly in review.
    """

    del candidate
    return None


def _producer_record(
    candidate: object, produced_by: str
) -> object | None:
    """The PRODUCER record of the chain, looked up by ``produced_by_ref``.

    R4-F4: ``produced_by_ref`` names the one provenance record that PRODUCED
    the payload — not merely any member of the chain (spec 48/180).
    """

    records = _records_of(candidate) or ()
    for record in records:
        if str(getattr(record, "provenance_id", "")) == produced_by.split(":", 1)[1]:
            return record
    return None


def _subjects_of(candidate: object) -> tuple[str, ...]:
    """Best-effort subject refs of a candidate of any admitted schema."""

    payload = getattr(candidate, "payload", None)
    subjects = getattr(payload, "subject_refs", None) if payload is not None else None
    if subjects is None and isinstance(candidate, Mapping):
        inner = candidate.get("payload")
        if isinstance(inner, Mapping):
            subjects = inner.get("subject_refs")
    if isinstance(subjects, str):
        return (subjects,)
    if isinstance(subjects, Sequence) and not isinstance(subjects, (bytes, bytearray, str)):
        return tuple(item for item in subjects if isinstance(item, str))
    return ()


def _opaque_covers(
    candidate: object, requested: str, lineage: Mapping[str, object]
) -> bool:
    """Coverage for a record that cannot enter the closed v1 DTO.

    An unknown- or v2-schema record must still block a fallback when its
    declared subject covers the requested descendant; treating it as absent
    would be a silent downgrade.
    """

    subjects = _subjects_of(candidate)
    if requested in subjects:
        return True
    return any(ref in subjects for ref in descendant_refs(requested, lineage))


def _diagnostic(
    code: str, artifact_ref: str | None = None, detail: str = ""
) -> QCPolicyDiagnostic:
    return QCPolicyDiagnostic(code=code, artifact_ref=artifact_ref, detail=detail)


# ---------------------------------------------------------------------------
# Eligibility: CALL the accepted DET-local policy (F2) plus schema handling
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _V2Record:
    """A validated v2 assessment record (spec §1.3 reserved v2 schema).

    A v2 record cannot enter the closed v1 DTO: ``QCAssessmentPayload``
    rejects ``VESSEL``/``REGION`` scope outright (v1 identifier admission is
    EMPTY), so carrying a v2 record in the v1 DTO would coerce it.  The gate
    therefore keeps its own normalized view, built only after the EXACT
    fifteen-key v2 table (including ``anatomic_identifier_schema`` and
    ``flag_evidence``) validates and the outer eligibility rules pass.
    """

    artifact_ref: str
    schema_version: str
    scope: str
    subject_refs: tuple[str, ...]
    quality_grade: QualityGrade
    vessel_id: str | None
    region_id: str | None
    supersedes_ref: str | None
    #: The REAL outer status of the mapping (never hardcoded): a mapping whose
    #: outer record is STALE/SUPERSEDED/INVALIDATED is ineligible (R2-F2).
    status: str
    payload: Mapping[str, object]
    raw: object


def _v2_record(candidate: object) -> _V2Record | None:
    """Build the normalized v2 view after the exact v2 table validates.

    The outer status, kind, and producer of the mapping are read here so the
    outer eligibility rules apply to a mapping exactly as they do to a v1 DTO
    (R2-F2): a mapping cannot escape them by not being a closed DTO.
    """

    if isinstance(candidate, _V2Record):
        return candidate
    if not isinstance(candidate, Mapping):
        return None
    raw_ref = candidate.get("artifact_ref")
    payload = candidate.get("payload")
    if not isinstance(raw_ref, str) or not isinstance(payload, Mapping):
        return None
    # R4-F5: the outer ref must match the canonical qc_artifact grammar
    # (core/dicom/qc.py:187) — any string is not a ref.
    if _QC_ARTIFACT_REF.fullmatch(raw_ref) is None:
        return None
    # R4-F5: absent outer status, kind, and payload schema are INVALID — no
    # valid defaults are substituted (fail-open was the R4-F5 defect).
    if "status" not in candidate or "kind" not in candidate or (
        "payload_schema" not in candidate
    ):
        return None
    outer_status = candidate.get("status")
    if not isinstance(outer_status, str):
        return None
    outer_kind = candidate.get("kind")
    if not isinstance(outer_kind, str) or outer_kind != "QC_ASSESSMENT":
        # The outer kind/schema binding is an eligibility rule (spec 320);
        # a mapping with a wrong outer kind is not a QC assessment record.
        return None
    outer_payload_schema = candidate.get("payload_schema")
    if outer_payload_schema != V2_PAYLOAD_SCHEMA:
        # A mapping whose outer schema is missing or disagrees with the v2
        # payload is not a selectable v2 record (R3-F3/R4-F5).
        return None
    outer_producer = candidate.get("producer")
    payload_producer = payload.get("producer")
    if outer_producer != payload_producer:
        # Producer and payload must agree (spec 320); a forged outer producer
        # (e.g. MODEL) is refused exactly as it is for a v1 DTO.
        return None
    # R7-F4: the outer envelope of a deterministic mapping carries the real
    # Producer block (spec 112-115): algorithm "qc-assessment-v1", version "1",
    # and parameters_blob_sha256 = the assessment-manifest digest.  A mapping
    # without them is not selectable.
    if payload_producer == "DETERMINISTIC_ENGINE":
        outer_algorithm = candidate.get("algorithm")
        outer_version = candidate.get("algorithm_version")
        outer_parameters = candidate.get("parameters_blob_sha256")
        if (
            outer_algorithm != "qc-assessment-v1"
            or outer_version != "1"
            or not isinstance(outer_parameters, str)
            or _SHA256_HEX.fullmatch(outer_parameters) is None
        ):
            return None
    return _V2Record(
        artifact_ref=raw_ref,
        schema_version=str(payload.get("schema_version")),
        scope=str(payload.get("scope")),
        subject_refs=tuple(
            item for item in payload.get("subject_refs", ()) if isinstance(item, str)
        ),
        quality_grade=QualityGrade(str(payload.get("quality_grade"))),
        vessel_id=cast("str | None", payload.get("vessel_id")),
        region_id=cast("str | None", payload.get("region_id")),
        supersedes_ref=cast("str | None", payload.get("supersedes_ref")),
        status=outer_status,
        payload=dict(payload),
        raw=candidate,
    )


@dataclass(frozen=True, slots=True)
class _View:
    """The schema-neutral view the gate selects on."""

    ref: str
    subjects: tuple[str, ...]
    scope: str
    vessel_id: str | None
    region_id: str | None
    supersedes_ref: str | None
    grade: QualityGrade
    raw: object

    @property
    def coverage_key(self) -> tuple[object, ...]:
        return (self.scope, self.subjects, self.vessel_id, self.region_id)


def _view_of(candidate: object) -> _View | None:
    """The schema-neutral view of one admitted candidate."""

    if isinstance(candidate, _V2Record):
        return _View(
            ref=candidate.artifact_ref,
            subjects=candidate.subject_refs,
            scope=candidate.scope,
            vessel_id=candidate.vessel_id,
            region_id=candidate.region_id,
            supersedes_ref=candidate.supersedes_ref,
            grade=candidate.quality_grade,
            raw=candidate.raw,
        )
    assessment = _assessment_of(candidate)
    if assessment is None:
        return None
    payload = assessment.payload
    return _View(
        ref=assessment.artifact_ref,
        subjects=tuple(payload.subject_refs),
        scope=payload.scope.value,
        vessel_id=payload.vessel_id,
        region_id=payload.region_id,
        supersedes_ref=payload.supersedes_ref,
        grade=payload.quality_grade,
        raw=candidate,
    )


def _view_covers(view: _View, requested: str, lineage: Mapping[str, object]) -> bool:
    """Mechanical coverage for either schema (no timestamps or ordering)."""

    if requested in view.subjects:
        return True
    return any(ref in view.subjects for ref in descendant_refs(requested, lineage))


def _view_identifier_matches(view: _View, token: str | None) -> bool:
    """The finite v2 token must match the scope and identifier exactly."""

    if token is None:
        return True
    if view.scope == QCScope.STUDY.value:
        return False
    if view.scope == QCScope.VESSEL.value:
        return view.vessel_id == token
    if view.scope == QCScope.REGION.value:
        return view.region_id == token
    return False


def _eligibility(
    candidates: Sequence[object],
    *,
    consumer_schema: str,
    consumed_refs: tuple[str, ...],
    consumer: str,
    lineage: Mapping[str, object],
) -> tuple[
    tuple[QCAssessmentArtifact, ...],
    tuple[QCPolicyDiagnostic, ...],
    tuple[object, ...],
]:  # noqa: E501
    """Resolve eligibility by CALLING the accepted policy (never duplicating).

    The complete capability/schema matrix (spec 2.3) is applied:

    * v1 consumer + v1 record: the policy validates producer identity,
      producer/payload agreement, outer kind/schema binding, grade/flag
      matrix, the complete bundle, and the provenance chain.
    * v1 consumer + v2 record: explicit ``QC_SCHEMA_INCOMPATIBLE`` with the
      consumer capability, observed schema, required schema, and ref — never
      a silent discard of the newer record.
    * v2 consumer + v1 record: normal shared-field selection (v1 semantics
      unchanged) — the policy validates it as a v1 record.
    * v2 consumer + v2 record: the structural v2 table is validated on the
      record itself (all fifteen keys present and well-formed, the closed flag
      vocabulary, the canonical subject order, and the finite identifier
      binding) and every clause emits its own diagnostic, but the record is
      then fail-closed INELIGIBLE: position ten of the frozen flag_evidence
      order is reserved (spec 226-233) and no accepted model parameter object
      exists in QC_001, so a v2 record can never be selected here.
    * unknown/malformed schema at either consumer: ``QC_SCHEMA_INVALID`` or
      ``QC_SCHEMA_INCOMPATIBLE``; the record is never coerced to v1.
    """

    diagnostics: list[QCPolicyDiagnostic] = []
    eligible: list[QCAssessmentArtifact] = []
    opaque_cover: list[object] = []
    v1_candidates: list[object] = []

    for candidate in candidates:
        schema = _schema_of(candidate)
        assessment = _assessment_of(candidate)
        ref = None if assessment is None else assessment.artifact_ref

        # --- unknown / malformed schema (either consumer) ---------------
        if schema is not None and schema not in (V1_PAYLOAD_SCHEMA, V2_PAYLOAD_SCHEMA):
            diagnostics.append(
                _diagnostic(DIAG_SCHEMA_INVALID, ref, f"unknown schema: {schema}")
            )
            if any(_opaque_covers(candidate, r, lineage) for r in consumed_refs):
                opaque_cover.append(candidate)
            continue

        # --- v2 record --------------------------------------------------
        if schema == V2_PAYLOAD_SCHEMA:
            # R2-F2: the candidate's artifact ref must be resolved from the
            # mapping itself, never left None — the mandatory v1-consumer
            # incompatibility diagnostic has to carry it (spec 370).
            mapping_ref = ref
            if mapping_ref is None and isinstance(candidate, Mapping):
                raw_ref = candidate.get("artifact_ref")
                if isinstance(raw_ref, str):
                    mapping_ref = raw_ref
            if consumer_schema == "v1":
                # v1 consumer + v2 record: explicit incompatibility, never a
                # silent downgrade to an older v1 record.  The diagnostic
                # carries the consumer capability, observed schema, required
                # schema, and artifact ref (spec 370, instructions 195).
                diagnostics.append(
                    _diagnostic(
                        DIAG_SCHEMA_INCOMPATIBLE,
                        mapping_ref,
                        "consumer capability v1; observed schema "
                        f"{V2_PAYLOAD_SCHEMA}; required schema {V1_PAYLOAD_SCHEMA}; "
                        f"artifact ref {mapping_ref}",
                    )
                )
                if any(_opaque_covers(candidate, r, lineage) for r in consumed_refs):
                    opaque_cover.append(candidate)
                continue
            # v2-aware consumer: the structural v2 table is validated on the
            # record (R2-F1), so every v2 clause keeps its own specific
            # diagnostic, but the record CANNOT become eligible: position ten
            # of the frozen flag_evidence order is reserved (spec 226-233) and
            # no accepted qc-model-disagreement-v1 parameter object exists in
            # QC_001 (QC_001-MODEL is declared-not-executed, spec 953-955), so
            # a v2 record's position-ten parameter_digest can NEVER resolve
            # here.  Every parseable v2 record is fail-closed INELIGIBLE with
            # a visible reserved diagnostic (R8-F3; the diagnostic EVENT code
            # is the existing closed vocabulary, never a new QCReason member).
            problems = _validate_v2_record(candidate, lineage)
            problems = problems + _validate_v2_assessment_level(candidate)
            problems = problems + (
                _diagnostic(
                    DIAG_SCHEMA_INVALID, mapping_ref, _RESERVED_POSITION_TEN_DETAIL
                ),
            )
            diagnostics.extend(problems)
            if any(_opaque_covers(candidate, r, lineage) for r in consumed_refs):
                opaque_cover.append(candidate)
            continue

        # --- v1 record at either consumer -------------------------------
        v1_candidates.append(candidate)

    # The accepted DET-local policy owns every v1 eligibility rule.  It is
    # CALLED once over the whole v1 candidate set (it validates each candidate
    # and resolves its own explicit-ref-graph selection), never modified.
    if v1_candidates:
        eligible_now, policy_diagnostics = _validate_v1_candidate(
            consumed_refs,
            v1_candidates,
            consumer=consumer,
            lineage=lineage,
        )
        diagnostics.extend(policy_diagnostics)
        eligible.extend(eligible_now)

    return tuple(eligible), tuple(diagnostics), tuple(opaque_cover)


def _validate_v2_assessment_level(
    candidate: object,
) -> tuple[QCPolicyDiagnostic, ...]:
    """The assessment-level v2 relationships (spec 224) and the G0-G5 matrix.

    With no ``ABSTAINED`` FlagEvidence entry, ``flags`` is exactly the
    canonical-order tuple of entries whose outcome is ``MILD`` or ``SEVERE``
    (empty when all entries are ``CLEAR``); with any required entry
    ``ABSTAINED``, ``flags`` is exactly ``("ABSTAINED",)`` and
    ``quality_grade`` is ``NOT_ANALYZABLE``.

    The grade/outcome matrix (spec 262-276) is then enforced in full:

    * **G0** — ``GOOD`` with ``flags=()``; all families ``CLEAR``;
    * **G1** — ``ACCEPTABLE`` with every flag ``MILD`` and no ``SEVERE``;
    * **G2** — ``LIMITED`` with at least one ``SEVERE``;
    * **G3** — ``NOT_ANALYZABLE`` with ``flags=("ABSTAINED",)``;
    * **G4** — any ``MODEL_DISAGREEMENT`` occurrence is rejected in QC_001;
    * **G5** — every other combination is a payload error (spec 276: the
      validator does not guess a grade, drop a flag, or downgrade to GOOD).
    """

    problems: list[QCPolicyDiagnostic] = []
    ref: str | None = None
    if isinstance(candidate, Mapping):
        raw_ref = candidate.get("artifact_ref")
        if isinstance(raw_ref, str):
            ref = raw_ref
    payload = getattr(candidate, "payload", None)
    if payload is None and isinstance(candidate, Mapping):
        payload = candidate.get("payload")
    if not isinstance(payload, Mapping):
        return ()
    entries = payload.get("flag_evidence")
    flags = payload.get("flags")
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        return ()
    outcomes = tuple(
        entry.get("outcome")
        for entry in entries
        if isinstance(entry, Mapping) and "outcome" in entry
    )
    if len(outcomes) != len(FIRST_TEN_FLAGS):
        # The per-entry validator already reported the malformed block; the
        # assessment-level relation cannot be evaluated on it.
        return ()
    any_abstained = any(outcome == "ABSTAINED" for outcome in outcomes)
    if any_abstained:
        if flags != ("ABSTAINED",):
            problems.append(
                _diagnostic(
                    DIAG_SCHEMA_INVALID,
                    ref,
                    f"v2 flags must be exactly ('ABSTAINED',) when any required "
                    f"entry is ABSTAINED; got {flags!r}",
                )
            )
        if str(payload.get("quality_grade")) != "NOT_ANALYZABLE":
            problems.append(
                _diagnostic(
                    DIAG_SCHEMA_INVALID,
                    ref,
                    "v2 quality_grade must be NOT_ANALYZABLE when any required "
                    f"entry is ABSTAINED; got {payload.get('quality_grade')!r}",
                )
            )
    else:
        derived = tuple(
            flag
            for flag, outcome in zip(FIRST_TEN_FLAGS, outcomes)
            if outcome in ("MILD", "SEVERE")
        )
        if flags != derived:
            problems.append(
                _diagnostic(
                    DIAG_SCHEMA_INVALID,
                    ref,
                    f"v2 flags must be exactly the canonical-order MILD/SEVERE "
                    f"tuple {derived}; got {flags!r}",
                )
            )
    problems.extend(_validate_v2_grade_matrix(payload, outcomes, ref))
    return tuple(problems)


def _validate_v2_grade_matrix(
    payload: Mapping[str, object],
    outcomes: tuple[object, ...],
    ref: str | None,
) -> tuple[QCPolicyDiagnostic, ...]:
    """The authoritative G0-G5 grade/outcome matrix (spec 262-276, R3-F2).

    ``P9`` is a nonempty subset of the first nine flags; ``MODEL_DISAGREEMENT``
    is position ten and is reserved, so it is never part of a P9 flag tuple.
    """

    problems: list[QCPolicyDiagnostic] = []
    grade = str(payload.get("quality_grade"))
    flags = payload.get("flags")
    resolved_flags = tuple(flags) if isinstance(flags, Sequence) else ()
    p9_flags = tuple(
        flag for flag, outcome in zip(FIRST_TEN_FLAGS, outcomes) if outcome in ("MILD", "SEVERE")
    )
    severities = {
        flag: outcome
        for flag, outcome in zip(FIRST_TEN_FLAGS, outcomes)
        if outcome in ("MILD", "SEVERE")
    }

    # G4: any non-clear MODEL_DISAGREEMENT is rejected before artifact
    # creation; the per-entry validator reports the reserved entry itself.
    disagreement_outcome = outcomes[len(FIRST_TEN_FLAGS) - 1]
    if disagreement_outcome != "CLEAR":
        problems.append(
            _diagnostic(
                DIAG_SCHEMA_INVALID,
                ref,
                f"G4: MODEL_DISAGREEMENT must be CLEAR; got {disagreement_outcome!r}. "
                "A non-clear entry requires the separate accepted model/structure "
                "contract and is rejected in QC_001.",
            )
        )

    if any(outcome == "ABSTAINED" for outcome in outcomes):
        # G3: the refusal row. No severity flag may accompany a refusal.
        if grade != "NOT_ANALYZABLE" or resolved_flags != ("ABSTAINED",):
            problems.append(
                _diagnostic(
                    DIAG_SCHEMA_INVALID,
                    ref,
                    f"G3: a refusal requires grade=NOT_ANALYZABLE and "
                    f"flags=('ABSTAINED',); got grade={grade!r}, flags={resolved_flags!r}",
                )
            )
        return tuple(problems)

    any_severe = any(outcome == "SEVERE" for outcome in outcomes)
    any_mild = any(outcome == "MILD" for outcome in outcomes)

    if not any_severe and not any_mild:
        # G0: all CLEAR.
        if grade != "GOOD" or resolved_flags != ():
            problems.append(
                _diagnostic(
                    DIAG_SCHEMA_INVALID,
                    ref,
                    f"G0: all-CLEAR evidence requires grade=GOOD and flags=(); "
                    f"got grade={grade!r}, flags={resolved_flags!r}",
                )
            )
    elif any_severe:
        # G2: at least one SEVERE.
        if grade != "LIMITED":
            problems.append(
                _diagnostic(
                    DIAG_SCHEMA_INVALID,
                    ref,
                    f"G2: a SEVERE outcome requires grade=LIMITED; got {grade!r}",
                )
            )
        if not isinstance(flags, Sequence) or not resolved_flags:
            problems.append(
                _diagnostic(
                    DIAG_SCHEMA_INVALID,
                    ref,
                    f"G2: a SEVERE outcome requires a nonempty P9 flag tuple; "
                    f"got {resolved_flags!r}",
                )
            )
        elif not any(severities.get(flag) == "SEVERE" for flag in resolved_flags):
            problems.append(
                _diagnostic(
                    DIAG_SCHEMA_INVALID,
                    ref,
                    f"G2: at least one flag in {resolved_flags!r} must be SEVERE",
                )
            )
        elif any(flag == "MODEL_DISAGREEMENT" for flag in resolved_flags):
            problems.append(
                _diagnostic(
                    DIAG_SCHEMA_INVALID,
                    ref,
                    "G2: MODEL_DISAGREEMENT is absent from a QC_001 flag tuple",
                )
            )
    else:
        # G1: at least one MILD and no SEVERE.
        if grade != "ACCEPTABLE":
            problems.append(
                _diagnostic(
                    DIAG_SCHEMA_INVALID,
                    ref,
                    f"G1: MILD-only evidence requires grade=ACCEPTABLE; got {grade!r}",
                )
            )
        if not p9_flags:
            problems.append(
                _diagnostic(
                    DIAG_SCHEMA_INVALID,
                    ref,
                    "G1: MILD evidence requires a nonempty P9 flag tuple",
                )
            )
        elif any(severities.get(flag) != "MILD" for flag in resolved_flags):
            problems.append(
                _diagnostic(
                    DIAG_SCHEMA_INVALID,
                    ref,
                    f"G1: every flag in {resolved_flags!r} must be MILD",
                )
            )
    return tuple(problems)


def _validate_v2_record(
    candidate: object, lineage: Mapping[str, object] | None = None
) -> tuple[QCPolicyDiagnostic, ...]:
    """Validate the EXACT frozen v2 table (spec §1.3, R2-F1).

    The accepted key set is exactly fifteen keys:
    ``schema_version, scope, subject_refs, quality_grade, confidence, flags,
    vessel_id, region_id, producer, produced_by_ref, supersedes_ref,
    justification_code, evidence_refs, anatomic_identifier_schema,
    flag_evidence``.  Missing, duplicate, unknown, and extra keys are all
    rejected.  Identifiers must be the frozen prefixed tokens (``VESSEL:LAD``
    style); a bare ``LAD`` or an arbitrary ``BANANA`` is invalid.  The
    ``FlagEvidence`` relationships of spec 210-226 are validated in full.

    R5-F1: ``lineage`` is the REQUEST's caller-supplied ``subject_lineage``.
    The frozen payload has no candidate lineage field, so subject resolution
    never reads a candidate-declared mapping.
    """

    problems: list[QCPolicyDiagnostic] = []
    assessment = _assessment_of(candidate)
    ref = None if assessment is None else assessment.artifact_ref
    payload = getattr(candidate, "payload", None)
    if payload is None and isinstance(candidate, Mapping):
        payload = candidate.get("payload")

    # --- outer eligibility (R2-F2): a mapping goes through the same rules ---
    # R4-F5: absent outer status, kind, and payload_schema are INVALID — no
    # valid defaults are substituted, matching the accepted DTO (qc.py:187).
    if isinstance(candidate, Mapping):
        if "status" not in candidate:
            problems.append(
                _diagnostic(
                    DIAG_SCHEMA_INVALID, ref, "outer status is required (no default)"
                )
            )
        if "kind" not in candidate:
            problems.append(
                _diagnostic(
                    DIAG_SCHEMA_INVALID, ref, "outer kind is required (no default)"
                )
            )
        if "payload_schema" not in candidate:
            problems.append(
                _diagnostic(
                    DIAG_SCHEMA_INVALID,
                    ref,
                    "outer payload_schema is required (no default)",
                )
            )
        if "artifact_ref" in candidate and (
            not isinstance(candidate.get("artifact_ref"), str)
            or _QC_ARTIFACT_REF.fullmatch(str(candidate["artifact_ref"])) is None
        ):
            problems.append(
                _diagnostic(
                    DIAG_SCHEMA_INVALID,
                    ref,
                    f"artifact_ref must be qc_artifact:<ULID>: "
                    f"{candidate.get('artifact_ref')!r}",
                )
            )
        outer_status = candidate.get("status", "ACTIVE")
        outer_status = (
            outer_status.value if hasattr(outer_status, "value") else outer_status
        )
        if outer_status in INELIGIBLE_STATUSES:
            problems.append(
                _diagnostic(
                    DIAG_RECORD_INELIGIBLE, ref, f"status={outer_status}"
                )
            )
        elif outer_status != "ACTIVE":
            problems.append(
                _diagnostic(DIAG_RECORD_INELIGIBLE, ref, f"status={outer_status}")
            )
        outer_kind = candidate.get("kind", "QC_ASSESSMENT")
        if outer_kind != "QC_ASSESSMENT":
            problems.append(
                _diagnostic(
                    DIAG_SCHEMA_INVALID, ref, f"outer kind is closed: {outer_kind!r}"
                )
            )
        outer_producer = candidate.get("producer")
        if outer_producer != payload.get("producer") if isinstance(payload, Mapping) else True:
            problems.append(
                _diagnostic(
                    DIAG_PRODUCER_INVALID,
                    ref,
                    f"outer producer {outer_producer!r} disagrees with payload "
                    f"producer {payload.get('producer')!r}"
                    if isinstance(payload, Mapping)
                    else "outer producer cannot be compared without a payload",
                )
            )
        # R7-F4: the outer envelope of a deterministic mapping carries the real
        # Producer block (spec 112-115): algorithm "qc-assessment-v1", version
        # "1", and parameters_blob_sha256 = the assessment-manifest digest.
        if isinstance(payload, Mapping) and payload.get("producer") == (
            "DETERMINISTIC_ENGINE"
        ):
            outer_algorithm = candidate.get("algorithm")
            outer_version = candidate.get("algorithm_version")
            outer_parameters = candidate.get("parameters_blob_sha256")
            if "algorithm" not in candidate or outer_algorithm != "qc-assessment-v1":
                problems.append(
                    _diagnostic(
                        DIAG_PRODUCER_INVALID,
                        ref,
                        "the outer Producer must declare algorithm "
                        f"'qc-assessment-v1' (spec 113); got {outer_algorithm!r}",
                    )
                )
            if "algorithm_version" not in candidate or outer_version != "1":
                problems.append(
                    _diagnostic(
                        DIAG_PRODUCER_INVALID,
                        ref,
                        "the outer Producer must declare algorithm_version '1' "
                        f"(spec 114); got {outer_version!r}",
                    )
                )
            if (
                "parameters_blob_sha256" not in candidate
                or not isinstance(outer_parameters, str)
                or _SHA256_HEX.fullmatch(outer_parameters) is None
            ):
                problems.append(
                    _diagnostic(
                        DIAG_PRODUCER_INVALID,
                        ref,
                        "the outer Producer must declare a lowercase-Sha256Hex "
                        "parameters_blob_sha256 (spec 115); got "
                        f"{outer_parameters!r}",
                    )
                )
            else:
                # Spec 104/122/139/141: that digest is the assessment-manifest
                # digest, bound by the PRODUCING record's parameters_ref — the
                # EXACT §44 record named by produced_by_ref (R9-F1, spec 48) —
                # and it must be exactly BlobRef("qc-parameters:<digest>",
                # <digest>) for the REAL seam manifest's digest.
                manifest_digest = _assessment_manifest_digest_of(candidate)
                if manifest_digest != outer_parameters:
                    problems.append(
                        _diagnostic(
                            DIAG_PROVENANCE_INVALID,
                            ref,
                            "the outer parameters_blob_sha256 must equal the "
                            "assessment-manifest digest bound by the producer "
                            "record named by produced_by_ref — a digest carried "
                            "by any OTHER chain record cannot bind it "
                            f"(BlobRef('qc-parameters:<digest>', <digest>), spec "
                            f"48/104/122/139/141); outer={outer_parameters!r}, "
                            f"bound={manifest_digest!r}",
                        )
                    )
        # R3-F3 + R4-F5: the outer payload_schema is REQUIRED (no default) and
        # must AGREE with the v2 payload.  The old rule accepted either v1 or
        # v2 and defaulted the missing key, so an outer-v1/payload-v2 record
        # or a schema-less record proceeded (spec 180/320).
        if "payload_schema" in candidate:
            outer_payload_schema = candidate.get("payload_schema")
            if outer_payload_schema not in (V1_PAYLOAD_SCHEMA, V2_PAYLOAD_SCHEMA):
                problems.append(
                    _diagnostic(
                        DIAG_SCHEMA_INVALID,
                        ref,
                        f"outer payload schema is closed: {outer_payload_schema!r}",
                    )
                )
            elif outer_payload_schema != V2_PAYLOAD_SCHEMA:
                problems.append(
                    _diagnostic(
                        DIAG_SCHEMA_INVALID,
                        ref,
                        f"outer payload schema {outer_payload_schema!r} disagrees with the "
                        f"v2 payload schema {V2_PAYLOAD_SCHEMA!r}",
                    )
                )
        # R3-F3: provenance verification is MANDATORY — a v2 record with no
        # chain cannot bind produced_by_ref and is refused (spec 180/320).
        records = _records_of(candidate)
        if records is None or not records:
            problems.append(
                _diagnostic(
                    DIAG_PROVENANCE_INVALID,
                    ref,
                    "a v2 record must supply its provenance chain; "
                    "produced_by_ref cannot bind without it",
                )
            )
        else:
            ok, chain_problems = verify_chain(cast("Sequence[ProvenanceRecord]", records))
            if not ok:
                problems.append(
                    _diagnostic(
                        DIAG_PROVENANCE_INVALID,
                        ref,
                        "; ".join(str(item) for item in chain_problems),
                    )
                )

    if not isinstance(payload, Mapping):
        problems.append(
            _diagnostic(DIAG_SCHEMA_INVALID, ref, "a v2 record must carry its payload table")
        )
        return tuple(problems)

    # --- exact key set (missing AND extra keys rejected) ----------------
    missing = [key for key in V2_REQUIRED_KEYS if key not in payload]
    if missing:
        problems.append(
            _diagnostic(
                DIAG_SCHEMA_INVALID, ref, f"v2 table is missing keys: {sorted(missing)}"
            )
        )
    extra = [key for key in payload if key not in V2_REQUIRED_KEYS]
    if extra:
        problems.append(
            _diagnostic(
                DIAG_SCHEMA_INVALID, ref, f"v2 table carries unknown keys: {sorted(extra)}"
            )
        )
    if missing or extra:
        return tuple(problems)

    # --- schema literal -------------------------------------------------
    if payload.get("schema_version") != V2_PAYLOAD_SCHEMA:
        problems.append(
            _diagnostic(DIAG_SCHEMA_INVALID, ref, "v2 schema_version disagrees")
        )
    if payload.get("anatomic_identifier_schema") != V2_IDENTIFIER_SCHEMA:
        problems.append(
            _diagnostic(
                DIAG_SCHEMA_INVALID,
                ref,
                "anatomic_identifier_schema must be "
                f"{V2_IDENTIFIER_SCHEMA!r}, got {payload.get('anatomic_identifier_schema')!r}",
            )
        )

    # --- scope and the finite prefixed identifier sets -------------------
    scope = payload.get("scope")
    if scope not in ("STUDY", "VESSEL", "REGION"):
        problems.append(
            _diagnostic(DIAG_SCHEMA_INVALID, ref, f"v2 scope is closed: {scope!r}")
        )
    vessel_id = payload.get("vessel_id")
    region_id = payload.get("region_id")
    if scope == "STUDY":
        if vessel_id is not None or region_id is not None:
            problems.append(
                _diagnostic(
                    DIAG_SCHEMA_INVALID, ref, "v2 STUDY scope cannot carry identifiers"
                )
            )
    elif scope == "VESSEL":
        if region_id is not None:
            problems.append(
                _diagnostic(DIAG_SCHEMA_INVALID, ref, "v2 VESSEL scope has region_id")
            )
        if vessel_id not in V2_VESSEL_IDS:
            problems.append(
                _diagnostic(
                    DIAG_SCHEMA_INVALID,
                    ref,
                    f"vessel_id must be one of the frozen V2_VESSEL_IDS "
                    f"(prefixed form, e.g. 'VESSEL:LAD'); got {vessel_id!r}",
                )
            )
    elif scope == "REGION":
        if vessel_id is not None:
            problems.append(
                _diagnostic(DIAG_SCHEMA_INVALID, ref, "v2 REGION scope has vessel_id")
            )
        if region_id not in V2_REGION_IDS:
            problems.append(
                _diagnostic(
                    DIAG_SCHEMA_INVALID,
                    ref,
                    f"region_id must be one of the frozen V2_REGION_IDS; got {region_id!r}",
                )
            )

    # --- closed vocabularies --------------------------------------------
    # R3-F2: subject_refs is nonempty, unique, canonically ordered, and obeys
    # the one-study rule (spec 173).
    subjects = payload.get("subject_refs")
    if not isinstance(subjects, Sequence) or isinstance(subjects, (str, bytes)):
        problems.append(
            _diagnostic(DIAG_SCHEMA_INVALID, ref, "v2 subject_refs must be a tuple")
        )
    else:
        resolved_subjects: list[str] = []
        for subject in subjects:
            if not isinstance(subject, str) or _CANONICAL_REF.fullmatch(subject) is None:
                problems.append(
                    _diagnostic(
                        DIAG_SCHEMA_INVALID,
                        ref,
                        f"v2 subject ref is not canonical: {subject!r}",
                    )
                )
            else:
                resolved_subjects.append(subject)
        if not resolved_subjects:
            problems.append(
                _diagnostic(
                    DIAG_SCHEMA_INVALID, ref, "v2 subject_refs must be nonempty"
                )
            )
        if len(set(resolved_subjects)) != len(resolved_subjects):
            problems.append(
                _diagnostic(
                    DIAG_SCHEMA_INVALID,
                    ref,
                    f"v2 subject_refs must be unique: {resolved_subjects!r}",
                )
            )
        if tuple(resolved_subjects) != _canonical_subject_order(resolved_subjects):
            problems.append(
                _diagnostic(
                    DIAG_SCHEMA_INVALID,
                    ref,
                    f"v2 subject_refs are not in canonical order: {resolved_subjects!r}",
                )
            )
        # The one-study rule: a v2 record carries at most one study subject.
        studies = [item for item in resolved_subjects if item.startswith("study_id:")]
        if len(studies) > 1:
            problems.append(
                _diagnostic(
                    DIAG_SCHEMA_INVALID,
                    ref,
                    f"v2 subject_refs carry more than one study: {studies!r}",
                )
            )
        if scope == "STUDY" and len(studies) != 1:
            problems.append(
                _diagnostic(
                    DIAG_SCHEMA_INVALID,
                    ref,
                    "v2 STUDY scope requires exactly one study_id subject ref",
                )
            )
        # R4-F3 + R5-F1: cross-study resolution (spec 41, 173, 203).  A v2
        # VESSEL or REGION record carries exactly one `study_id:<ULID>` subject
        # AND one identifier token (spec 203-206), and every subject must
        # RESOLVE to that one study — a volume of another study cannot ride
        # inside a record whose study subject differs.  R5-F1: the lineage is
        # the REQUEST's caller-supplied mapping; a candidate cannot vouch for
        # its own subjects because the frozen payload has no lineage field.
        if lineage is None:
            lineage = {}
        subject_studies: set[str] = set()
        unresolved: list[str] = []
        for subject in resolved_subjects:
            if subject.startswith("study_id:"):
                subject_studies.add(subject)
                continue
            if not isinstance(lineage, Mapping):
                unresolved.append(subject)
                continue
            ancestors = descendant_refs(subject, lineage)
            subject_studies.update(
                item for item in ancestors if item.startswith("study_id:")
            )
            if not any(item.startswith("study_id:") for item in ancestors):
                unresolved.append(subject)
        if len(subject_studies) > 1:
            problems.append(
                _diagnostic(
                    DIAG_SCHEMA_INVALID,
                    ref,
                    f"v2 subject_refs resolve to more than one study: "
                    f"{sorted(subject_studies)!r}",
                )
            )
        if unresolved:
            problems.append(
                _diagnostic(
                    DIAG_SCHEMA_INVALID,
                    ref,
                    f"v2 subject refs do not resolve to a study: {unresolved!r}",
                )
            )
        if scope in ("VESSEL", "REGION"):
            # spec 203: exactly one study subject plus exactly one token.
            studies = [
                item for item in resolved_subjects if item.startswith("study_id:")
            ]
            if len(studies) != 1:
                problems.append(
                    _diagnostic(
                        DIAG_SCHEMA_INVALID,
                        ref,
                        f"a v2 {scope} record carries exactly one study_id "
                        f"subject (spec 203); got {studies!r}",
                    )
                )
            elif len(subject_studies) != 1:
                problems.append(
                    _diagnostic(
                        DIAG_SCHEMA_INVALID,
                        ref,
                        f"every subject of a v2 {scope} record must resolve to "
                        f"its one study (spec 41/173); resolved "
                        f"{sorted(subject_studies)!r}",
                    )
                )

    # R3-F2: confidence is validated against the closed set (spec 175).
    confidence_value = payload.get("confidence")
    if confidence_value is not None and confidence_value not in V2_CONFIDENCE_VALUES:
        problems.append(
            _diagnostic(
                DIAG_SCHEMA_INVALID,
                ref,
                f"v2 confidence is closed: {confidence_value!r}",
            )
        )
    elif (
        confidence_value == "NOT_ANALYZABLE"
        and str(payload.get("quality_grade")) != "NOT_ANALYZABLE"
    ):
        problems.append(
            _diagnostic(
                DIAG_SCHEMA_INVALID,
                ref,
                "NOT_ANALYZABLE confidence requires the NOT_ANALYZABLE grade",
            )
        )

    # R3-F2: flags is the eleven-value vocabulary in canonical order (spec 176).
    payload_flags = payload.get("flags")
    if isinstance(payload_flags, Sequence) and not isinstance(payload_flags, (str, bytes)):
        resolved_flags = tuple(payload_flags)
        if len(set(resolved_flags)) != len(resolved_flags) or any(
            flag not in QC_FLAGS for flag in resolved_flags
        ):
            problems.append(
                _diagnostic(
                    DIAG_SCHEMA_INVALID,
                    ref,
                    f"v2 flags are outside the closed vocabulary: {resolved_flags!r}",
                )
            )
        elif tuple(flag for flag in QC_FLAGS if flag in resolved_flags) != resolved_flags:
            problems.append(
                _diagnostic(
                    DIAG_SCHEMA_INVALID,
                    ref,
                    f"v2 flags are not in canonical order: {resolved_flags!r}",
                )
            )
    grade_value = payload.get("quality_grade")
    try:
        grade = QualityGrade(str(grade_value))
    except ValueError:
        problems.append(
            _diagnostic(DIAG_SCHEMA_INVALID, ref, f"v2 quality_grade is closed: {grade_value!r}")
        )
        grade = None
    if payload.get("producer") not in ("DETERMINISTIC_ENGINE", "HUMAN"):
        problems.append(
            _diagnostic(
                DIAG_PRODUCER_INVALID,
                ref,
                f"v2 producer must be DETERMINISTIC_ENGINE or HUMAN; "
                f"{payload.get('producer')!r} requires a later accepted model contract",
            )
        )
    # R8-F2 sweep (spec 49, inherited at 166): supersedes_ref is None for the
    # first record or exactly qc_artifact:<ULID> for the prior record.  A
    # malformed ref is a payload error at validation time, not merely an
    # unresolvable relation at selection time.
    supersedes_value = payload.get("supersedes_ref")
    if supersedes_value is not None and (
        not isinstance(supersedes_value, str)
        or _QC_ARTIFACT_REF.fullmatch(supersedes_value) is None
    ):
        problems.append(
            _diagnostic(
                DIAG_SCHEMA_INVALID,
                ref,
                f"v2 supersedes_ref must be None or qc_artifact:<ULID>; "
                f"got {supersedes_value!r}",
            )
        )
    # R3-F3 + R4-F4: produced_by_ref is syntax-checked AND bound to the
    # PRODUCER record of the supplied chain — not merely any member (spec 48:
    # "the immutable provenance record producing this payload"; 180).
    produced_by = payload.get("produced_by_ref")
    producer_value = payload.get("producer")
    if not isinstance(produced_by, str) or _PROVENANCE_REF.fullmatch(produced_by) is None:
        problems.append(
            _diagnostic(DIAG_PROVENANCE_INVALID, ref, "v2 produced_by_ref is malformed")
        )
    else:
        records = _records_of(candidate)
        producer_record = _producer_record(candidate, produced_by)
        if producer_record is None:
            problems.append(
                _diagnostic(
                    DIAG_PROVENANCE_INVALID,
                    ref,
                    f"v2 produced_by_ref {produced_by!r} does not name the producer "
                    "record of the supplied provenance chain",
                )
            )
        else:
            # R5-F3: the producer record must be ABOUT this candidate — its
            # ``subject_ref`` must equal the candidate's own artifact id
            # (spec 48: "the immutable provenance record producing this
            # payload", inherited at 166/180).  A chain record about a
            # DIFFERENT artifact does not authorize this candidate.
            actor_id = str(getattr(getattr(producer_record, "actor", None), "actor_id", ""))
            artifact_id = (
                str(candidate.get("artifact_ref")).split(":", 1)[1]
                if isinstance(candidate, Mapping)
                and isinstance(candidate.get("artifact_ref"), str)
                and ":" in str(candidate.get("artifact_ref"))
                else None
            )
            subject_ref = str(getattr(producer_record, "subject_ref", ""))
            subject_matches = (
                artifact_id is not None
                and subject_ref.split(":")[-1] == artifact_id
            )
            candidate_ref = (
                str(candidate.get("artifact_ref"))
                if isinstance(candidate, Mapping)
                else str(getattr(candidate, "artifact_ref", ref))
            )
            if not subject_matches:
                problems.append(
                    _diagnostic(
                        DIAG_PROVENANCE_INVALID,
                        ref,
                        f"v2 produced_by_ref {produced_by!r} names a record about "
                        f"{subject_ref!r}, not about this candidate "
                        f"({candidate_ref!r})",
                    )
                )
            # R5-F4 + R6-F1/R6-F3: the producer record's ACTOR semantics must
            # match the payload producer (spec 116-121, 147-150, inherited at
            # 166).  NotApplicable detection uses isinstance against the real
            # seam type, not a string comparison of the type name.
            actor_type = str(getattr(getattr(producer_record, "actor", None), "actor_type", ""))
            actor_type = actor_type.rsplit(".", 1)[-1]
            human_edit = getattr(producer_record, "human_edit", None)
            is_not_applicable = isinstance(human_edit, NotApplicable)
            human_edit_reason = (
                str(getattr(human_edit, "reason", "")) if is_not_applicable else ""
            )
            parameters_ref = getattr(producer_record, "parameters_ref", None)
            parameters_not_applicable = isinstance(parameters_ref, NotApplicable)
            parameters_reason = (
                str(getattr(parameters_ref, "reason", ""))
                if parameters_not_applicable
                else ""
            )
            has_parameters = (
                parameters_ref is not None
                and not parameters_not_applicable
                and bool(getattr(parameters_ref, "sha256", ""))
            )
            preprocessing = str(getattr(producer_record, "preprocessing", ""))
            event_type_value = str(getattr(producer_record, "event_type", ""))
            event_type_value = event_type_value.rsplit(".", 1)[-1]
            if producer_value == "DETERMINISTIC_ENGINE":
                # Spec 116-121: ActorType.ENGINE (exactly), actor_id
                # qc-assessment-v1, event_type in DERIVED|RECOMPUTED|REVIEWED,
                # the preprocessing literal QC_INPUT_NORMALIZATION_V1, and a
                # nonempty parameters_ref (R6-F1).
                if actor_type != "ENGINE":
                    problems.append(
                        _diagnostic(
                            DIAG_PROVENANCE_INVALID,
                            ref,
                            f"v2 produced_by_ref {produced_by!r} requires "
                            f"ActorType.ENGINE; got {actor_type!r}",
                        )
                    )
                if actor_id != "qc-assessment-v1" or not has_parameters:
                    problems.append(
                        _diagnostic(
                            DIAG_PROVENANCE_INVALID,
                            ref,
                            f"v2 produced_by_ref {produced_by!r} is not the deterministic "
                            f"assessment producer record (actor_id={actor_id!r}, "
                            f"parameters_ref={has_parameters})",
                        )
                    )
                if event_type_value not in ("DERIVED", "RECOMPUTED", "REVIEWED"):
                    problems.append(
                        _diagnostic(
                            DIAG_PROVENANCE_INVALID,
                            ref,
                            f"v2 produced_by_ref {produced_by!r} event_type must be "
                            f"DERIVED, RECOMPUTED, or REVIEWED; got "
                            f"{event_type_value!r}",
                        )
                    )
                if preprocessing != "QC_INPUT_NORMALIZATION_V1":
                    problems.append(
                        _diagnostic(
                            DIAG_PROVENANCE_INVALID,
                            ref,
                            f"v2 produced_by_ref {produced_by!r} preprocessing must be "
                            f"the canonical literal QC_INPUT_NORMALIZATION_V1; got "
                            f"{preprocessing!r}",
                        )
                    )
                if not is_not_applicable:
                    problems.append(
                        _diagnostic(
                            DIAG_PROVENANCE_INVALID,
                            ref,
                            f"v2 produced_by_ref {produced_by!r} human_edit must be an "
                            "explicit NotApplicable value (spec 118); got an actual "
                            "HumanEdit",
                        )
                    )
                # R7-F3: source InputRefs (spec 105/119) and resolvable
                # upstream_refs (spec 121) are MANDATORY for the deterministic
                # producer record — NotApplicable for either fails closed.
                producer_inputs = getattr(producer_record, "inputs", None)
                inputs_not_applicable = isinstance(producer_inputs, NotApplicable)
                inputs_empty = (
                    inputs_not_applicable
                    or not list(getattr(producer_inputs, "__iter__", lambda: ())())
                )
                if inputs_not_applicable or inputs_empty:
                    problems.append(
                        _diagnostic(
                            DIAG_PROVENANCE_INVALID,
                            ref,
                            f"v2 produced_by_ref {produced_by!r} requires source "
                            "InputRefs (spec 105/119); got "
                            + (
                                "NotApplicable"
                                if inputs_not_applicable
                                else "an empty inputs tuple"
                            ),
                        )
                    )
                upstream_refs = getattr(producer_record, "upstream_refs", None)
                upstream_not_applicable = isinstance(upstream_refs, NotApplicable)
                upstream_items = (
                    []
                    if upstream_not_applicable
                    else list(getattr(upstream_refs, "__iter__", lambda: ())())
                )
                if upstream_not_applicable or not upstream_items:
                    problems.append(
                        _diagnostic(
                            DIAG_PROVENANCE_INVALID,
                            ref,
                            f"v2 produced_by_ref {produced_by!r} requires resolvable "
                            "upstream_refs (spec 121); got "
                            + (
                                "NotApplicable"
                                if upstream_not_applicable
                                else "an empty upstream tuple"
                            ),
                        )
                    )
            elif producer_value == "HUMAN":
                # Spec 147-150: a HUMAN record's §44 entry has ActorType.HUMAN,
                # an ACTUAL HumanEdit (not NotApplicable), and
                # parameters_ref=NotApplicable with the EXACT frozen reason
                # (R6-F3).
                if actor_type != "HUMAN":
                    problems.append(
                        _diagnostic(
                            DIAG_PROVENANCE_INVALID,
                            ref,
                            f"v2 produced_by_ref {produced_by!r} requires "
                            f"ActorType.HUMAN; got {actor_type!r}",
                        )
                    )
                if is_not_applicable:
                    problems.append(
                        _diagnostic(
                            DIAG_PROVENANCE_INVALID,
                            ref,
                            f"v2 produced_by_ref {produced_by!r} must carry an actual "
                            f"HumanEdit; got NotApplicable({human_edit_reason!r})",
                        )
                    )
                if has_parameters:
                    problems.append(
                        _diagnostic(
                            DIAG_PROVENANCE_INVALID,
                            ref,
                            f"v2 produced_by_ref {produced_by!r} must carry "
                            "parameters_ref=NotApplicable; got a concrete parameter "
                            "blob",
                        )
                    )
                if not parameters_not_applicable or parameters_reason != (
                    "HUMAN_REVIEW_HAS_NO_DETERMINISTIC_PARAMETER_BLOB"
                ):
                    problems.append(
                        _diagnostic(
                            DIAG_PROVENANCE_INVALID,
                            ref,
                            f"v2 produced_by_ref {produced_by!r} parameters_ref reason "
                            "must be exactly "
                            f"'HUMAN_REVIEW_HAS_NO_DETERMINISTIC_PARAMETER_BLOB'; got "
                            f"{parameters_reason!r}",
                        )
                    )
    justification = payload.get("justification_code")
    if justification not in ("RULE_ENGINE_EVIDENCE", "HUMAN_EVIDENCE"):
        problems.append(
            _diagnostic(
                DIAG_SCHEMA_INVALID,
                ref,
                f"v2 justification code is closed: {justification!r} "
                "(PLACEHOLDER_NO_QC_EVIDENCE is invalid in v2)",
            )
        )
    # R4-F2: the v1 producer/justification binding is INHERITED (spec 99/100,
    # 147; inheritance at 166): a deterministic producer pairs only with
    # RULE_ENGINE_EVIDENCE and confidence=None; a HUMAN producer pairs only
    # with HUMAN_EVIDENCE.  Deterministic + HIGH confidence is the v1
    # "no uncalibrated confidence for a gate decision" rule.
    producer_value = payload.get("producer")
    if producer_value == "DETERMINISTIC_ENGINE" and justification not in (
        None,
        "RULE_ENGINE_EVIDENCE",
    ):
        problems.append(
            _diagnostic(
                DIAG_PRODUCER_INVALID,
                ref,
                f"a DETERMINISTIC_ENGINE record requires RULE_ENGINE_EVIDENCE; "
                f"got justification_code={justification!r}",
            )
        )
    if producer_value == "DETERMINISTIC_ENGINE" and payload.get("confidence") is not None:
        problems.append(
            _diagnostic(
                DIAG_SCHEMA_INVALID,
                ref,
                f"a deterministic record carries confidence=None (spec 100/147); "
                f"got {payload.get('confidence')!r}",
            )
        )
    if producer_value == "HUMAN" and justification not in (None, "HUMAN_EVIDENCE"):
        problems.append(
            _diagnostic(
                DIAG_PRODUCER_INVALID,
                ref,
                f"a HUMAN record requires HUMAN_EVIDENCE; got "
                f"justification_code={justification!r}",
            )
        )
    # R3-F2: evidence_refs is a nonempty tuple of CANONICAL v1 evidence refs
    # and is unique (spec 183).  The frozen v1 evidence forms are the
    # ``source_manifest:<Sha256Hex>`` and ``provenance:<ULID>`` shapes.
    evidence = payload.get("evidence_refs")
    if (
        not isinstance(evidence, Sequence)
        or isinstance(evidence, (str, bytes))
        or not evidence
    ):
        problems.append(
            _diagnostic(DIAG_SCHEMA_INVALID, ref, "v2 record requires evidence_refs")
        )
    else:
        resolved_evidence = tuple(evidence)
        if any(
            not isinstance(item, str) or _EVIDENCE_REF.fullmatch(item) is None
            for item in resolved_evidence
        ):
            problems.append(
                _diagnostic(
                    DIAG_SCHEMA_INVALID,
                    ref,
                    f"v2 evidence ref is not canonical: {resolved_evidence!r}",
                )
            )
        if len(set(resolved_evidence)) != len(resolved_evidence):
            problems.append(
                _diagnostic(
                    DIAG_SCHEMA_INVALID,
                    ref,
                    f"v2 evidence_refs must be unique: {resolved_evidence!r}",
                )
            )
        # R5-F4 + R6-F2: the assessment-evidence clause (spec 101, inherited at
        # 166).  A real DETERMINISTIC assessment's evidence contains ITS OWN
        # assessment provenance ref — the exact produced_by_ref, not any
        # provenance-prefixed string — plus at least one source-manifest or
        # canonical-volume/MPR ref.
        if producer_value == "DETERMINISTIC_ENGINE":
            has_own_provenance_evidence = isinstance(produced_by, str) and (
                produced_by in resolved_evidence
            )
            has_source_evidence = any(
                isinstance(item, str)
                and (
                    item.startswith("source_manifest:")
                    or item.startswith("volume_id:")
                    or item.startswith("mpr_artifact_id:")
                )
                for item in resolved_evidence
            )
            if not has_own_provenance_evidence or not has_source_evidence:
                problems.append(
                    _diagnostic(
                        DIAG_SCHEMA_INVALID,
                        ref,
                        "a deterministic v2 record's evidence_refs must contain its "
                        "OWN assessment provenance ref (produced_by_ref) plus at "
                        "least one source-manifest or canonical-volume/MPR ref "
                        f"(spec 101); produced_by_ref={produced_by!r}, "
                        f"evidence={resolved_evidence!r}",
                    )
                )
        elif producer_value == "HUMAN":
            # R7-F1: a HUMAN record's evidence contains a nonempty HUMAN
            # evidence ref (spec 147/148, inherited at 166).  A provenance ref
            # of the record's own review is the human evidence anchor; a bare
            # source-manifest ref is NOT human evidence.  Spec 148 requires a
            # provenance ref naming the human review record, so the evidence
            # must include the record's OWN produced_by_ref.
            has_human_evidence = isinstance(produced_by, str) and (
                produced_by in resolved_evidence
            )
            if not has_human_evidence:
                problems.append(
                    _diagnostic(
                        DIAG_SCHEMA_INVALID,
                        ref,
                        "a HUMAN v2 record's evidence_refs must contain its OWN "
                        "human-review provenance ref (produced_by_ref) — a bare "
                        "source-manifest ref is not human evidence (spec 147/148); "
                        f"produced_by_ref={produced_by!r}, "
                        f"evidence={resolved_evidence!r}",
                    )
                )

    # --- flag_evidence: the complete FlagEvidence relationships ---------
    problems.extend(
        _validate_v2_flag_evidence(
            payload.get("flag_evidence"), grade, ref, candidate, payload.get("producer")
        )
    )
    return tuple(problems)


def _assessment_manifest_digest_of(candidate: object) -> str | None:
    """The assessment-manifest digest bound by the chain's producing record.

    R7-F4 (spec 104/122/141): the producer record's ``parameters_ref`` must be
    exactly ``BlobRef("qc-parameters:<assessment-digest>", <assessment-digest>)``;
    anything else is unbound and returns ``None``.

    R9-F1 (spec 48/139/141): the binding record is EXACTLY the one named by
    the payload's ``produced_by_ref`` — the immutable provenance record that
    produced this payload.  A foreign ``qc-assessment-v1`` record elsewhere in
    the chain, even one carrying the real seam manifest digest, is NOT this
    candidate's producer and can never satisfy the manifest binding; an
    unresolvable or malformed ``produced_by_ref`` is likewise unbound
    (fail-closed).  Pre-att0010 this helper scanned the WHOLE chain for any
    ``qc-assessment-v1`` record carrying the real digest, which let a foreign
    record lend its digest to a candidate whose actual producer record was
    bound to a fake manifest.
    """

    payload = getattr(candidate, "payload", None)
    if payload is None and isinstance(candidate, Mapping):
        payload = candidate.get("payload")
    produced_by = (
        payload.get("produced_by_ref") if isinstance(payload, Mapping) else None
    )
    if (
        not isinstance(produced_by, str)
        or _PROVENANCE_REF.fullmatch(produced_by) is None
    ):
        return None
    record = _producer_record(candidate, produced_by)
    if record is None:
        return None
    if str(getattr(getattr(record, "actor", None), "actor_id", "")) != (
        "qc-assessment-v1"
    ):
        return None
    parameters_ref = getattr(record, "parameters_ref", None)
    if isinstance(parameters_ref, NotApplicable):
        return None
    sha = getattr(parameters_ref, "sha256", None)
    blob_ref = getattr(parameters_ref, "blob_ref", None)
    if not (isinstance(sha, str) and sha) or not isinstance(blob_ref, str):
        return None
    expected = assessment_parameter_digest()
    # Spec 141: the ref is EXACTLY BlobRef("qc-parameters:<assessment-
    # digest>", <assessment-digest>) — the real seam digest, not merely a
    # self-consistent pair (R7-F2/R7-F4) and not a digest carried by some
    # OTHER chain record (R9-F1).
    if blob_ref != f"qc-parameters:{expected}" or sha != expected:
        return None
    return sha


def _bound_parameter_digests(candidate: object) -> frozenset[str] | None:
    """The parameter digests bound in the candidate's provenance chain (R4-F4).

    Returns ``None`` when no chain is exposed (the caller then applies only
    the syntax rule); otherwise the set of every ``parameters_ref.sha256`` the
    chain actually carries.  A ``parameter_digest`` outside this set does not
    resolve to a bound parameter object (spec 222).  R5-F2: set membership is
    necessary but NOT sufficient — the per-flag check is family-specific.
    """

    records = _records_of(candidate)
    if records is None:
        return None
    digests: set[str] = set()
    for record in records:
        parameters_ref = getattr(record, "parameters_ref", None)
        sha = getattr(parameters_ref, "sha256", None)
        if isinstance(sha, str) and sha:
            digests.add(sha)
        blob_sha = getattr(parameters_ref, "blob_ref", None)
        if isinstance(blob_sha, str) and blob_sha.startswith("qc-parameters:"):
            digests.add(blob_sha.split(":", 1)[1])
    return frozenset(digests)


def _validate_v2_flag_evidence(
    entries: object,
    grade: QualityGrade | None,
    ref: str | None,
    candidate: object,
    producer_value: object,
) -> tuple[QCPolicyDiagnostic, ...]:
    """Validate the exact ``FlagEvidence`` block (spec 210-226)."""

    problems: list[QCPolicyDiagnostic] = []
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        problems.append(
            _diagnostic(
                DIAG_SCHEMA_INVALID, ref, "v2 flag_evidence must be a tuple of entries"
            )
        )
        return tuple(problems)
    # Exactly one entry per each of the first ten flags, in frozen order.
    expected_flags = FIRST_TEN_FLAGS
    if len(entries) != len(expected_flags):
        problems.append(
            _diagnostic(
                DIAG_SCHEMA_INVALID,
                ref,
                f"v2 flag_evidence must hold exactly {len(expected_flags)} entries "
                f"(one per frozen flag, in order); got {len(entries)}",
            )
        )
        return tuple(problems)
    for index, entry in enumerate(entries):
        flag = expected_flags[index]
        if not isinstance(entry, Mapping):
            problems.append(
                _diagnostic(
                    DIAG_SCHEMA_INVALID, ref, f"flag_evidence[{index}] is not an object"
                )
            )
            continue
        missing = [key for key in FLAG_EVIDENCE_KEYS if key not in entry]
        if missing:
            problems.append(
                _diagnostic(
                    DIAG_SCHEMA_INVALID,
                    ref,
                    f"flag_evidence[{index}] ({flag}) is missing keys: {sorted(missing)}",
                )
            )
            continue
        extra = [key for key in entry if key not in FLAG_EVIDENCE_KEYS]
        if extra:
            problems.append(
                _diagnostic(
                    DIAG_SCHEMA_INVALID,
                    ref,
                    f"flag_evidence[{index}] ({flag}) carries unknown keys: {sorted(extra)}",
                )
            )
            continue
        if entry.get("flag") != flag:
            problems.append(
                _diagnostic(
                    DIAG_SCHEMA_INVALID,
                    ref,
                    f"flag_evidence[{index}] must be flag {flag!r} in frozen order; "
                    f"got {entry.get('flag')!r}",
                )
            )
            continue
        # Outcome is closed; ABSTAINED pairs with metric_value=None and a
        # non-null abstention_code, every other outcome with a finite
        # metric_value and a null abstention_code (spec 224-226).
        outcome = entry.get("outcome")
        if outcome not in ("CLEAR", "MILD", "SEVERE", "ABSTAINED"):
            problems.append(
                _diagnostic(
                    DIAG_SCHEMA_INVALID,
                    ref,
                    f"flag_evidence[{index}] outcome is closed: {outcome!r}",
                )
            )
            continue
        metric_name = entry.get("metric_name")
        if metric_name != V2_METRIC_NAMES[flag]:
            problems.append(
                _diagnostic(
                    DIAG_SCHEMA_INVALID,
                    ref,
                    f"flag_evidence[{index}] ({flag}) metric_name must be "
                    f"{V2_METRIC_NAMES[flag]!r}; got {metric_name!r}",
                )
            )
        threshold = entry.get("threshold_profile")
        if threshold != V2_FAMILY_IDS[flag]:
            problems.append(
                _diagnostic(
                    DIAG_SCHEMA_INVALID,
                    ref,
                    f"flag_evidence[{index}] ({flag}) threshold_profile must be "
                    f"{V2_FAMILY_IDS[flag]!r}; got {threshold!r}",
                )
            )
        unit = entry.get("metric_unit")
        if unit != V2_METRIC_UNITS[flag]:
            problems.append(
                _diagnostic(
                    DIAG_SCHEMA_INVALID,
                    ref,
                    f"flag_evidence[{index}] ({flag}) metric_unit must be "
                    f"{V2_METRIC_UNITS[flag]!r}; got {unit!r}",
                )
            )
        algorithm_id = entry.get("algorithm_id")
        if algorithm_id != V2_FAMILY_IDS[flag]:
            problems.append(
                _diagnostic(
                    DIAG_SCHEMA_INVALID,
                    ref,
                    f"flag_evidence[{index}] ({flag}) algorithm_id must equal "
                    f"{V2_FAMILY_IDS[flag]!r}; got {algorithm_id!r}",
                )
            )
        if entry.get("algorithm_version") != "1":
            problems.append(
                _diagnostic(
                    DIAG_SCHEMA_INVALID,
                    ref,
                    f"flag_evidence[{index}] algorithm_version must be '1'; "
                    f"got {entry.get('algorithm_version')!r}",
                )
            )
        parameter_digest = entry.get("parameter_digest")
        if (
            not isinstance(parameter_digest, str)
            or _SHA256_HEX.fullmatch(parameter_digest) is None
        ):
            problems.append(
                _diagnostic(
                    DIAG_SCHEMA_INVALID,
                    ref,
                    f"flag_evidence[{index}] parameter_digest must be a lowercase "
                    f"Sha256Hex; got {parameter_digest!r}",
                )
            )
        else:
            # R4-F4 + R5-F2 (spec 218-222): the digest must RESOLVE to THAT
            # flag's family parameter object.  The digest is verified against
            # the FROZEN family parameter object digest computed from the seam
            # itself — an arbitrary self-consistent digest/blob-ref pair
            # declared by a fake provenance record does NOT name the actual
            # family parameter object and is rejected.  R8-F3: there is no
            # tenth entry to resolve in QC_001 — MODEL_DISAGREEMENT is
            # reserved (spec 226/229-230), no accepted model parameter object
            # exists, and every parseable v2 record is fail-closed ineligible
            # with the reserved-position diagnostic (see _eligibility), so the
            # per-family digest check covers only the nine deterministic
            # families.
            if flag != "MODEL_DISAGREEMENT" and producer_value == "DETERMINISTIC_ENGINE":
                family_actor = V2_FAMILY_IDS[flag]
                frozen_digest = family_parameter_digest(flag)
                if parameter_digest != frozen_digest:
                        problems.append(
                            _diagnostic(
                                DIAG_SCHEMA_INVALID,
                                ref,
                                f"flag_evidence[{index}] ({flag}) parameter_digest does "
                                f"not name the actual {family_actor} family parameter "
                                f"object (its frozen content digest is "
                                f"{frozen_digest!r}); got {parameter_digest!r}",
                            )
                        )
        # R3-F2: the per-flag evidence refs are CANONICAL and unique (spec 223).
        flag_evidence_refs = entry.get("evidence_refs")
        if (
            not isinstance(flag_evidence_refs, Sequence)
            or isinstance(flag_evidence_refs, (str, bytes))
            or not flag_evidence_refs
        ):
            problems.append(
                _diagnostic(
                    DIAG_SCHEMA_INVALID,
                    ref,
                    f"flag_evidence[{index}] ({flag}) requires evidence_refs",
                )
            )
        else:
            resolved_flag_refs = tuple(flag_evidence_refs)
            if any(
                not isinstance(item, str) or _EVIDENCE_REF.fullmatch(item) is None
                for item in resolved_flag_refs
            ):
                problems.append(
                    _diagnostic(
                        DIAG_SCHEMA_INVALID,
                        ref,
                        f"flag_evidence[{index}] ({flag}) evidence ref is not "
                        f"canonical: {resolved_flag_refs!r}",
                    )
                )
            if len(set(resolved_flag_refs)) != len(resolved_flag_refs):
                problems.append(
                    _diagnostic(
                        DIAG_SCHEMA_INVALID,
                        ref,
                        f"flag_evidence[{index}] ({flag}) evidence_refs must be unique",
                    )
                )
        # R3-F2: MODEL_DISAGREEMENT is reserved and never non-clear on a
        # deterministic producer (spec 222-226, 240-241, matrix G4).
        if flag == "MODEL_DISAGREEMENT" and outcome != "CLEAR":
            problems.append(
                _diagnostic(
                    DIAG_SCHEMA_INVALID,
                    ref,
                    "MODEL_DISAGREEMENT must be CLEAR: a non-clear entry requires the "
                    f"separate accepted model/structure contract; got {outcome!r}",
                )
            )
        metric_value = entry.get("metric_value")
        abstention_code = entry.get("abstention_code")
        if outcome == "ABSTAINED":
            if metric_value is not None:
                problems.append(
                    _diagnostic(
                        DIAG_SCHEMA_INVALID,
                        ref,
                        f"flag_evidence[{index}] ({flag}) ABSTAINED requires "
                        "metric_value=None",
                    )
                )
            # R3-F2: the abstention code is a member of the closed vocabulary;
            # an unknown code is a schema error, never an accepted abstention.
            if abstention_code is None:
                problems.append(
                    _diagnostic(
                        DIAG_SCHEMA_INVALID,
                        ref,
                        f"flag_evidence[{index}] ({flag}) ABSTAINED requires a "
                        "non-null abstention_code",
                    )
                )
            elif abstention_code not in V2_ABSTENTION_CODES:
                problems.append(
                    _diagnostic(
                        DIAG_SCHEMA_INVALID,
                        ref,
                        f"flag_evidence[{index}] ({flag}) abstention_code is closed: "
                        f"{abstention_code!r}",
                    )
                )
        else:
            if not isinstance(metric_value, (int, float)) or isinstance(metric_value, bool):
                problems.append(
                    _diagnostic(
                        DIAG_SCHEMA_INVALID,
                        ref,
                        f"flag_evidence[{index}] ({flag}) outcome {outcome!r} requires "
                        "a finite metric_value",
                    )
                )
            elif metric_value != metric_value or metric_value in (
                float("inf"),
                float("-inf"),
            ):
                problems.append(
                    _diagnostic(
                        DIAG_SCHEMA_INVALID,
                        ref,
                        f"flag_evidence[{index}] ({flag}) metric_value must be finite",
                    )
                )
            if abstention_code is not None:
                problems.append(
                    _diagnostic(
                        DIAG_SCHEMA_INVALID,
                        ref,
                        f"flag_evidence[{index}] ({flag}) non-ABSTAINED requires "
                        "abstention_code=None",
                    )
                )
    # The assessment-level relationships (spec 224): with no ABSTAINED entry,
    # flags is exactly the canonical-order tuple of MILD/SEVERE entries (empty
    # when all CLEAR); with any ABSTAINED entry, flags is exactly
    # ("ABSTAINED",) and the grade is NOT_ANALYZABLE.  These are checked by
    # the caller against the whole record; here the per-entry rules are done.
    return tuple(problems)


# ---------------------------------------------------------------------------
# Gate-scoped selection adapter (spec 2.2)
# ---------------------------------------------------------------------------


def _select_effective_for_ref(
    requested: str,
    covering: Sequence[_View],
    *,
    lineage: Mapping[str, object],
) -> tuple[tuple[_View, ...], tuple[QCPolicyDiagnostic, ...]]:
    """The effective maxima covering one consumed ref, or a visible conflict.

    Gate-scoped adapter over the accepted #16 selector: the explicit
    ``supersedes_ref`` graph resolves per consumed ref, and two distinct
    maxima are incomparable overlap (``QC_CONFLICT``).  A consumer never
    chooses a winner, and no timestamp or file name ever substitutes for a
    ref.

    F3: an unresolvable ancestor anywhere in the chain invalidates the
    CANDIDATE RELATION — the successor is ineligible (visible ineligibility),
    never effective.  With no other valid coverage the outcome is
    ``QC_NOT_ASSESSED``.
    """

    diagnostics: list[QCPolicyDiagnostic] = []
    by_ref: dict[str, _View] = {view.ref: view for view in covering}

    # Check 1-4: same coverage keys, exact ref, chain-valid and acyclic, and
    # at most one direct predecessor per record.
    invalid: set[str] = set()
    for view in covering:
        predecessor_ref = view.supersedes_ref
        if predecessor_ref is None:
            continue
        predecessor = by_ref.get(predecessor_ref)
        # Check 1 + 2: the predecessor is the exact named ref with equal
        # coverage keys; no timestamp or creation order substitutes.
        if predecessor is None or predecessor.coverage_key != view.coverage_key:
            invalid.add(view.ref)
            diagnostics.append(
                _diagnostic(
                    DIAG_SUPERSESSION_INVALID,
                    view.ref,
                    "predecessor is missing or coverage differs",
                )
            )
            continue
        # Check 3: the chain is acyclic AND every prior record resolves.  A
        # missing ancestor invalidates the candidate relation (F3): the walk
        # must reach a record with no supersedes_ref; breaking early on an
        # unresolvable ref leaves the successor ineligible, never effective.
        seen: set[str] = set()
        current_ref: str | None = predecessor_ref
        chain_broken = False
        while current_ref is not None:
            if current_ref in seen:
                invalid.add(view.ref)
                diagnostics.append(
                    _diagnostic(
                        DIAG_SUPERSESSION_INVALID, view.ref, "supersession cycle"
                    )
                )
                chain_broken = True
                break
            seen.add(current_ref)
            next_view = by_ref.get(current_ref)
            if next_view is None:
                # F3: an unresolvable ancestor invalidates the relation.
                invalid.add(view.ref)
                diagnostics.append(
                    _diagnostic(
                        DIAG_SUPERSESSION_INVALID,
                        view.ref,
                        f"supersession chain is unresolvable at {current_ref}",
                    )
                )
                chain_broken = True
                break
            current_ref = next_view.supersedes_ref
        if chain_broken:
            continue

    # Check 4: two active successors of one predecessor are distinct branches
    # until one explicitly supersedes the other; both therefore stay maxima.
    valid = [view for view in covering if view.ref not in invalid]
    superseded = {view.supersedes_ref for view in valid}
    maxima = [view for view in valid if view.ref not in superseded]
    if not maxima:
        return (), tuple(diagnostics)

    refs = tuple(sorted(view.ref for view in maxima))
    # Two distinct effective maxima covering the same requested ref are
    # incomparable overlap: QC_CONFLICT, and never a winner.
    if len(maxima) > 1:
        diagnostics.append(
            _diagnostic(
                DIAG_CONFLICT,
                refs[0],
                "incomparable effective maxima: " + ",".join(refs),
            )
        )
        return tuple(maxima), tuple(diagnostics)

    single = maxima[0]

    # The accepted #16 study-level selector is CALLED, never modified.  It
    # independently confirms the explicit-ref-graph resolution and raises on
    # incomparable maxima; the gate translates that into QC_CONFLICT.  Only
    # candidates that carry a closed v1 assessment view can be handed to it:
    # a v2 record must never be coerced into the v1 DTO.
    study_candidates = [
        view.raw for view in covering if _assessment_of(view.raw) is not None
    ]
    if study_candidates and single.raw in study_candidates:
        study_subjects = single.subjects
        study_ref = study_subjects[0] if study_subjects else requested
        try:
            selected = _select_effective_assessment(study_ref, study_candidates)
        except QCValidationError as exc:
            detail = str(exc)
            if "incomparable effective maxima" in detail:
                diagnostics.append(_diagnostic(DIAG_CONFLICT, refs[0], detail))
                return tuple(maxima), tuple(diagnostics)
            diagnostics.append(_diagnostic(DIAG_SUPERSESSION_INVALID, refs[0], detail))
            return (), tuple(diagnostics)
        except ValueError:
            # Any other selection problem is visible ineligibility, never a pass.
            diagnostics.append(
                _diagnostic(
                    DIAG_SUPERSESSION_INVALID, refs[0], "selection is not resolvable"
                )
            )
            return (), tuple(diagnostics)
        if selected is not None and selected.artifact_ref != single.ref:
            # The accepted selector resolved a different record; follow it only
            # when it is itself one of this ref's active maxima.
            for view in valid:
                if view.ref == selected.artifact_ref:
                    return (view,), tuple(diagnostics)
    return (single,), tuple(diagnostics)


# ---------------------------------------------------------------------------
# Reason precedence (spec 2.4)
# ---------------------------------------------------------------------------


def _effective_views(views: Sequence[_View]) -> tuple[QCAssessmentArtifact, ...]:
    """The effective records as closed v1 assessment views.

    A validated v2 record is surfaced by its refs and grade on the gate
    result, but it is never coerced into the closed v1 DTO, so it does not
    appear in this tuple.  The refs it contributes remain visible in
    ``assessment_refs``.
    """

    values: list[QCAssessmentArtifact] = []
    for view in views:
        assessment = _assessment_of(view.raw)
        if assessment is not None:
            values.append(assessment)
    return tuple(values)


def _reason_for_grades(grades: Sequence[QualityGrade]) -> QCReason | None:
    """Precedence 3-5 over the effective grades actually observed."""

    if any(grade == QualityGrade.NOT_ANALYZABLE for grade in grades):
        return QCReason.QC_NOT_ANALYZABLE
    if any(grade == QualityGrade.LIMITED for grade in grades):
        return QCReason.QC_LIMITED
    if all(grade in {QualityGrade.GOOD, QualityGrade.ACCEPTABLE} for grade in grades):
        return None
    return QCReason.QC_NOT_ASSESSED


def qc_consumer_gate_request(
    request: QCConsumerRequest,
    candidates: Sequence[object] = (),
) -> QCConsumerGateResult:
    """Evaluate the real-QC consumer gate for one request (spec 2.1-2.4).

    The precedence is evaluated only after every requested ref has been
    collected, exactly as specified:

    1. any consumed ref lacks eligible effective coverage;
    2. any consumed ref has two or more incomparable effective maxima;
    3. effective coverage contains a ``NOT_ANALYZABLE`` grade;
    4. effective coverage contains a ``LIMITED`` grade;
    5. every effective grade is ``GOOD`` or ``ACCEPTABLE``.

    Confidence never changes this order.  After an ``ABSTAIN`` the caller must
    read zero pixels, create zero analysis artifacts, and return this result
    unchanged.
    """

    diagnostics: list[QCPolicyDiagnostic] = []
    lineage = request.subject_lineage
    token = request.requested_identifier

    # Eligibility: CALL the accepted policy (v1 records) and the structural
    # v2 validator (v2 records at a v2-aware consumer).  Every refusal is
    # visible; nothing is silently dropped.  No v2 record can become eligible
    # in QC_001 (reserved position ten, spec 226-233), so the selection pool
    # below is exactly the policy-validated v1 assessments.
    (
        eligible_assessments,
        eligibility_diagnostics,
        opaque_cover,
    ) = _eligibility(
        candidates,
        consumer_schema=request.consumer_schema,
        consumed_refs=request.consumed_refs,
        consumer=request.consumer,
        lineage=lineage,
    )
    diagnostics.extend(eligibility_diagnostics)

    # The schema-neutral selection pool: the accepted v1 assessments produced
    # by the DET policy (a v2 record is never coerced into the closed v1 DTO).
    pool: list[_View] = []
    for assessment in eligible_assessments:
        view = _view_of(assessment)
        if view is not None:
            pool.append(view)

    # Per consumed ref: collect effective maxima in canonical artifact order.
    effective: dict[str, _View] = {}
    missing: list[str] = []
    conflict_refs: tuple[str, ...] = ()
    for requested in request.consumed_refs:
        covering: list[_View] = []
        for view in pool:
            if _view_covers(view, requested, lineage) and _view_identifier_matches(
                view, token
            ):
                covering.append(view)
        if opaque_cover:
            # An incompatible/unknown record covering the request blocks the
            # fallback to an older compatible record; no silent downgrade.
            covering = []
        if not covering:
            missing.append(requested)
            continue
        maxima, selection_diagnostics = _select_effective_for_ref(
            requested, covering, lineage=lineage
        )
        diagnostics.extend(selection_diagnostics)
        if not maxima:
            missing.append(requested)
            continue
        if len(maxima) > 1:
            conflict_refs = tuple(sorted(view.ref for view in maxima))
            for view in maxima:
                effective[view.ref] = view
            continue
        single_view = maxima[0]
        effective[single_view.ref] = single_view

    ordered_refs = tuple(sorted(effective))
    ordered = tuple(effective[ref] for ref in ordered_refs)

    # Priority 1: any consumed ref lacks eligible effective coverage.
    if missing or opaque_cover:
        return QCConsumerGateResult(
            decision=QCGateDecision.ABSTAIN,
            reason_code=QCReason.QC_NOT_ASSESSED,
            assessment_refs=(),
            consumer=request.consumer,
            consumed_refs=request.consumed_refs,
            diagnostics=tuple(diagnostics),
        )

    # Priority 2: incomparable effective maxima on any consumed ref.
    if conflict_refs:
        return QCConsumerGateResult(
            decision=QCGateDecision.ABSTAIN,
            reason_code=QCReason.QC_CONFLICT,
            assessment_refs=conflict_refs,
            consumer=request.consumer,
            consumed_refs=request.consumed_refs,
            diagnostics=tuple(diagnostics),
            effective_assessments=_effective_views(ordered),
        )

    # Priorities 3-5: the worst applicable grade across the effective set.
    grades: list[QualityGrade] = [view.grade for view in ordered]
    reason = _reason_for_grades(grades)
    if reason is None:
        return QCConsumerGateResult(
            decision=QCGateDecision.PROCEED,
            reason_code=None,
            assessment_refs=ordered_refs,
            consumer=request.consumer,
            consumed_refs=request.consumed_refs,
            diagnostics=tuple(diagnostics),
            effective_assessments=_effective_views(ordered),
        )
    return QCConsumerGateResult(
        decision=QCGateDecision.ABSTAIN,
        reason_code=reason,
        assessment_refs=ordered_refs,
        consumer=request.consumer,
        consumed_refs=request.consumed_refs,
        diagnostics=tuple(diagnostics),
        effective_assessments=_effective_views(ordered),
    )


def real_qc_consumer_gate(
    consumed_subject_refs: tuple[str, ...],
    candidates: Sequence[object] = (),
    *,
    consumer: str,
    consumer_schema: str = "v1",
    requested_identifier: str | None = None,
    subject_lineage: Mapping[str, object] | None = None,
    lineage: Mapping[str, object] | None = None,
) -> QCConsumerGateResult:
    """Convenience entry point mirroring the accepted DET-local signature."""

    resolved: dict[str, object] = {}
    if subject_lineage:
        resolved.update(subject_lineage)
    if lineage:
        resolved.update(lineage)
    return qc_consumer_gate_request(
        QCConsumerRequest(
            consumed_refs=tuple(consumed_subject_refs),
            consumer=consumer,
            consumer_schema=consumer_schema,
            requested_identifier=requested_identifier,
            subject_lineage=resolved,
        ),
        candidates,
    )


#: The QC_001_GATE public surface.
__all__ = [
    "CONSUMER_SCHEMAS",
    "DIAG_CONFLICT",
    "DIAG_LEGACY_PLACEHOLDER",
    "DIAG_PRODUCER_INVALID",
    "DIAG_PROVENANCE_INVALID",
    "DIAG_RECORD_INELIGIBLE",
    "DIAG_SCHEMA_INCOMPATIBLE",
    "DIAG_SCHEMA_INVALID",
    "DIAG_SUPERSESSION_INVALID",
    "INELIGIBLE_STATUSES",
    "PLACEHOLDER_JUSTIFICATION",
    "QCConsumerGateResult",
    "QCConsumerRequest",
    "V1_PAYLOAD_SCHEMA",
    "V2_PAYLOAD_SCHEMA",
    "covers",
    "descendant_refs",
    "qc_consumer_gate_request",
    "real_qc_consumer_gate",
]
