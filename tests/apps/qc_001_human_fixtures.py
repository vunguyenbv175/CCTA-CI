"""QC_001-HUMAN synthetic workflow/state-transition fixtures (spec Part 4.4).

This module is the authoritative source of the frozen human literals that the
independent oracle imports, the sixteen census rows transcribed from the
accepted Part 4.4 table, and the per-case lineage recipes.  Like
``qc_001_d_fixtures`` it is deliberately an input generator: it never
imports the production QC package, so the oracle stays independent of the
implementation under test.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Final

from core.digest import canonical_digest
from tests.apps.qc_001_d_fixtures import (
    CensusCase,
    PreparedQCInput,
    census_cases,
)

# ---------------------------------------------------------------------------
# Frozen human literals (QC_001_SPEC §1.2 / Part 4.4; authoritative for the
# independent oracle — the production module transcribes the same values and
# the benchmark asserts byte-for-byte agreement).
# ---------------------------------------------------------------------------

ACTOR_ID_HUMAN: Final[str] = "qc-human-correction-v1"
ACTOR_TYPE_VALUE: Final[str] = "HUMAN"
EVENT_INITIAL_HUMAN_VALUE: Final[str] = "REVIEWED"
EVENT_HUMAN_CORRECTION_VALUE: Final[str] = "CORRECTED"
HUMAN_EVIDENCE_KIND: Final[str] = "HUMAN_EVIDENCE"
HUMAN_JUSTIFICATION_CODE: Final[str] = "HUMAN_EVIDENCE"
PRODUCER_VALUE: Final[str] = "HUMAN"
PARAMETERS_NOT_APPLICABLE_REASON: Final[str] = (
    "HUMAN_REVIEW_HAS_NO_DETERMINISTIC_PARAMETER_BLOB"
)
PAYLOAD_KEY_COUNT: Final[int] = 13
PAYLOAD_SCHEMA_VALUE: Final[str] = "ccta.payload.qc_assessment.v1"
SCOPE_VALUE: Final[str] = "STUDY"
PREPROCESSING_VALUE: Final[str] = "QC_INPUT_NORMALIZATION_V1"
SOFTWARE_VERSION_VALUE: Final[str] = "ccta-qc-v1"
CREATED_BY_VALUE: Final[str] = "qc_001_human"
ZERO_SHA: Final[str] = "0" * 40
FIXED_AT: Final[str] = "2026-01-01T00:00:00Z"
SOURCE_ROOT_PREFIX: Final[str] = "qc-source-manifest"
HUMAN_EVIDENCE_SUBJECT_PREFIX: Final[str] = "qc_human_evidence"
STALE_REASON_TEMPLATE: Final[str] = (
    "upstream geometry {geometry_id} invalidated or advanced beyond the "
    "version this artifact pinned (H-010)"
)
DEFAULT_EDIT_SUMMARY: Final[str] = "human review confirms the study-level assessment"

#: Deterministic id offsets of a case-8 descendant canonical volume.  The
#: descendant carries a NEW canonical volume identity (volume id and volume
#: provenance ref) so the predecessor and the successor never bind the same
#: volume member, while the source bytes stay content-identical.
DESCENDANT_VOLUME_SEED_OFFSET: Final[int] = 50
DESCENDANT_PROVENANCE_SEED_OFFSET: Final[int] = 51

#: Real state-event literals frozen by the store emission site.
EVENT_ENTITY: Final[str] = "derived_artifact"
EVENT_TYPE_STATUS: Final[str] = "STATUS"
EVENT_ACTOR_ROLE: Final[str] = "ENGINE"
STATUS_SUPERSEDED: Final[str] = "SUPERSEDED"
STATUS_STALE: Final[str] = "STALE"
STATUS_ACTIVE: Final[str] = "ACTIVE"

# ---------------------------------------------------------------------------
# Frozen refusal table (cases 9-16 plus the additional 16a-16c probes).
# ---------------------------------------------------------------------------

REFUSAL_CODE_OVERWRITE: Final[str] = "QC_HUMAN_OVERWRITE_REJECTED"
REFUSAL_CODE_SUPERSESSION: Final[str] = "QC_SUPERSESSION_INVALID"
REFUSAL_CODE_EVIDENCE: Final[str] = "QC_HUMAN_EVIDENCE_INVALID"
REFUSAL_CODE_PROVENANCE: Final[str] = "QC_PROVENANCE_UNVERIFIED"
REFUSAL_CODE_ACTOR: Final[str] = "QC_HUMAN_ACTOR_INVALID"
REFUSAL_CODE_EVENT: Final[str] = "QC_HUMAN_EVENT_INVALID"

# ---------------------------------------------------------------------------
# Frozen endpoint raw counts and rates (Part 4.4; no imputation, no sampling).
# ---------------------------------------------------------------------------

VALID_N: Final[int] = 8
FORBIDDEN_N: Final[int] = 8
CENSUS_N: Final[int] = 16
VALID_TRANSITIONS_EXACT: Final[int] = 8
VALID_EVIDENCE_COMPLETE: Final[int] = 8
OLD_BYTES_DENOMINATOR: Final[int] = 7
OLD_BYTES_UNCHANGED: Final[int] = 7
EFFECTIVE_SELECTED_EXACT: Final[int] = 8
OVERWRITES_ACCEPTED: Final[int] = 0
MISSING_SUPERSEDES_ACCEPTED: Final[int] = 0
STALE_DEPENDENT_ACCEPTED: Final[int] = 0
CANDIDATE_PASSES_ACCEPTED: Final[int] = 0
POST_ABSTAIN_CALLS: Final[int] = 0
INVALID_ACTOR_ACCEPTED: Final[int] = 0
INVALID_EDITOR_ACCEPTED: Final[int] = 0
INVALID_EVENT_ACCEPTED: Final[int] = 0

# ---------------------------------------------------------------------------
# Source identity of one prepared canonical input (oracle-visible only).
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SourceIdentity:
    """Every digest/ref the oracle needs to rebuild the expected records."""

    study_ref: str
    source_manifest_sha256: str
    volume_id: str
    volume_blob_sha256: str
    volume_provenance_ref: str
    source_series_refs: tuple[str, ...]
    instances: tuple[tuple[str, str], ...]
    frame_digest: str
    transform_ref: str
    transform_sha256: str
    transform_provenance_ref: str
    geometry_id: str


def source_identity(prepared: PreparedQCInput) -> SourceIdentity:
    """Derive the oracle-visible identity without touching the QC package."""

    volume = prepared.first_volume
    frame = volume.coordinate_frame
    return SourceIdentity(
        study_ref=volume.study_ref,
        source_manifest_sha256=prepared.source_manifest_sha256,
        volume_id=volume.volume_id,
        volume_blob_sha256=volume.blob_ref.sha256,
        volume_provenance_ref=str(volume.provenance_ref),
        source_series_refs=tuple(volume.source_series_refs),
        instances=tuple(
            (instance.sop_instance_uid, hashlib.sha256(instance.pixel_data).hexdigest())
            for instance in prepared.first_instances
        ),
        frame_digest=canonical_digest(frame),
        transform_ref=prepared.first_transform_ref,
        transform_sha256=prepared.first_transform_sha256,
        transform_provenance_ref=prepared.first_transform_provenance_ref,
        geometry_id=f"volume_geometry:{volume.volume_id}",
    )


# ---------------------------------------------------------------------------
# Lineage recipes for the sixteen census cases and the extra probes.
# ---------------------------------------------------------------------------

HUMAN_STEP: Final[str] = "HUMAN"
DET_STEP: Final[str] = "ENGINE_DET"
REFUSAL_STEP: Final[str] = "ENGINE_REFUSAL"
GEOMETRY_STEP: Final[str] = "GEOMETRY_ADVANCE"


@dataclass(frozen=True, slots=True)
class HumanStep:
    """One recipe step executed through a real production constructor."""

    producer: str
    grade: str
    flags: tuple[str, ...]
    seed: int
    supersedes: int | None = None
    geometry_version: int = 1
    advance_geometry_to: int = 0
    reuse_seed_of: int | None = None
    fault: str = ""
    event_type_value: str | None = None
    descendant: bool = False

    @property
    def descendant_volume_id_seed(self) -> int:
        """Deterministic id seed of this step's descendant canonical volume."""

        return self.seed + DESCENDANT_VOLUME_SEED_OFFSET

    @property
    def descendant_volume_provenance_seed(self) -> int:
        """Deterministic provenance seed of the descendant canonical volume."""

        return self.seed + DESCENDANT_PROVENANCE_SEED_OFFSET


@dataclass(frozen=True, slots=True)
class HumanCase:
    """One of the sixteen Part 4.4 census rows (or an extra probe)."""

    ordinal: int
    case_id: str
    label: str
    valid: bool
    steps: tuple[HumanStep, ...]
    refusal_code: str | None = None
    attempt: HumanStep | None = None
    attempt_kind: str = "correction"
    second_study: bool = False
    edit_summary: str = DEFAULT_EDIT_SUMMARY


def _seed(ordinal: int, slot: int) -> int:
    return 8_000_000 + ordinal * 100 + slot


_SECOND_STUDY_SEED: Final[int] = 9_500_000


def second_study_seed() -> int:
    return _SECOND_STUDY_SEED


def _valid(ordinal: int, case_id: str, label: str, steps: tuple[HumanStep, ...]) -> HumanCase:
    return HumanCase(
        ordinal=ordinal, case_id=case_id, label=label, valid=True, steps=steps
    )


def _forbidden(
    ordinal: int,
    case_id: str,
    label: str,
    steps: tuple[HumanStep, ...],
    *,
    refusal_code: str,
    attempt: HumanStep,
    attempt_kind: str = "correction",
    second_study: bool = False,
) -> HumanCase:
    return HumanCase(
        ordinal=ordinal,
        case_id=case_id,
        label=label,
        valid=False,
        steps=steps,
        refusal_code=refusal_code,
        attempt=attempt,
        attempt_kind=attempt_kind,
        second_study=second_study,
    )


def human_census_cases() -> tuple[HumanCase, ...]:
    """Return exactly the sixteen Part 4.4 rows (eight valid, eight forbidden)."""

    valid: tuple[HumanCase, ...] = (
        _valid(
            1,
            "QC001-H-01",
            "initial human assessment (no predecessor)",
            (HumanStep(HUMAN_STEP, "GOOD", (), _seed(1, 0)),),
        ),
        _valid(
            2,
            "QC001-H-02",
            "deterministic -> human correction",
            (
                HumanStep(DET_STEP, "GOOD", (), _seed(2, 0)),
                HumanStep(
                    HUMAN_STEP,
                    "ACCEPTABLE",
                    ("CARDIAC_MOTION",),
                    _seed(2, 10),
                    supersedes=0,
                ),
            ),
        ),
        _valid(
            3,
            "QC001-H-03",
            "human -> human correction (chained)",
            (
                HumanStep(HUMAN_STEP, "GOOD", (), _seed(3, 0)),
                HumanStep(
                    HUMAN_STEP,
                    "LIMITED",
                    ("POOR_CONTRAST",),
                    _seed(3, 10),
                    supersedes=0,
                ),
                HumanStep(HUMAN_STEP, "GOOD", (), _seed(3, 20), supersedes=1),
            ),
        ),
        _valid(
            4,
            "QC001-H-04",
            "correction that keeps GOOD",
            (
                HumanStep(HUMAN_STEP, "GOOD", (), _seed(4, 0)),
                HumanStep(HUMAN_STEP, "GOOD", (), _seed(4, 10), supersedes=0),
            ),
        ),
        _valid(
            5,
            "QC001-H-05",
            "correction GOOD -> LIMITED",
            (
                HumanStep(HUMAN_STEP, "GOOD", (), _seed(5, 0)),
                HumanStep(
                    HUMAN_STEP,
                    "LIMITED",
                    ("EXCESSIVE_NOISE",),
                    _seed(5, 10),
                    supersedes=0,
                ),
            ),
        ),
        _valid(
            6,
            "QC001-H-06",
            "correction LIMITED -> GOOD",
            (
                HumanStep(HUMAN_STEP, "LIMITED", ("BLOOMING",), _seed(6, 0)),
                HumanStep(HUMAN_STEP, "GOOD", (), _seed(6, 10), supersedes=0),
            ),
        ),
        _valid(
            7,
            "QC001-H-07",
            "correction resolving prior ABSTAINED",
            (
                HumanStep(REFUSAL_STEP, "NOT_ANALYZABLE", ("ABSTAINED",), _seed(7, 0)),
                HumanStep(HUMAN_STEP, "GOOD", (), _seed(7, 10), supersedes=0),
            ),
        ),
        _valid(
            8,
            "QC001-H-08",
            "correction with a descendant volume ref",
            (
                HumanStep(HUMAN_STEP, "GOOD", (), _seed(8, 0), geometry_version=1),
                HumanStep(
                    HUMAN_STEP,
                    "GOOD",
                    (),
                    _seed(8, 10),
                    supersedes=0,
                    geometry_version=1,
                    advance_geometry_to=2,
                    # A REAL descendant volume ref: the correction binds a new
                    # canonical volume identity (the geometry advance makes the
                    # predecessor STALE first).
                    descendant=True,
                ),
            ),
        ),
    )
    forbidden: tuple[HumanCase, ...] = (
        _forbidden(
            9,
            "QC001-H-09",
            "overwrite of an existing assessment (mutate in place)",
            (HumanStep(HUMAN_STEP, "GOOD", (), _seed(9, 0)),),
            refusal_code=REFUSAL_CODE_OVERWRITE,
            attempt=HumanStep(
                HUMAN_STEP, "GOOD", (), _seed(9, 10), reuse_seed_of=0
            ),
            attempt_kind="initial",
        ),
        _forbidden(
            10,
            "QC001-H-10",
            "correction with no supersedes_ref",
            (HumanStep(HUMAN_STEP, "GOOD", (), _seed(10, 0)),),
            refusal_code=REFUSAL_CODE_SUPERSESSION,
            attempt=HumanStep(HUMAN_STEP, "GOOD", (), _seed(10, 10)),
        ),
        _forbidden(
            11,
            "QC001-H-11",
            "wrong predecessor (not the effective assessment)",
            (
                HumanStep(HUMAN_STEP, "GOOD", (), _seed(11, 0)),
                HumanStep(
                    HUMAN_STEP, "LIMITED", ("POOR_CONTRAST",), _seed(11, 10), supersedes=0
                ),
            ),
            refusal_code=REFUSAL_CODE_SUPERSESSION,
            attempt=HumanStep(
                HUMAN_STEP, "GOOD", (), _seed(11, 20), supersedes=0
            ),
        ),
        _forbidden(
            12,
            "QC001-H-12",
            "cycle (A supersedes B and B supersedes A)",
            (
                HumanStep(HUMAN_STEP, "GOOD", (), _seed(12, 0)),
                HumanStep(HUMAN_STEP, "GOOD", (), _seed(12, 10), supersedes=0),
            ),
            refusal_code=REFUSAL_CODE_SUPERSESSION,
            attempt=HumanStep(
                HUMAN_STEP, "GOOD", (), _seed(12, 20), supersedes=1, reuse_seed_of=0
            ),
        ),
        _forbidden(
            13,
            "QC001-H-13",
            "cross-study predecessor",
            (HumanStep(HUMAN_STEP, "GOOD", (), _seed(13, 0)),),
            refusal_code=REFUSAL_CODE_SUPERSESSION,
            attempt=HumanStep(
                HUMAN_STEP, "GOOD", (), _seed(13, 20), supersedes=0
            ),
            second_study=True,
        ),
        _forbidden(
            14,
            "QC001-H-14",
            "invalid evidence (missing/ill-typed HUMAN_EVIDENCE)",
            (HumanStep(HUMAN_STEP, "GOOD", (), _seed(14, 0)),),
            refusal_code=REFUSAL_CODE_EVIDENCE,
            attempt=HumanStep(
                HUMAN_STEP,
                "GOOD",
                (),
                _seed(14, 10),
                supersedes=0,
                fault="evidence_kind",
            ),
        ),
        _forbidden(
            15,
            "QC001-H-15",
            "stale (superseded) predecessor used as effective",
            (
                HumanStep(HUMAN_STEP, "GOOD", (), _seed(15, 0), geometry_version=1),
                HumanStep(
                    GEOMETRY_STEP,
                    "GOOD",
                    (),
                    _seed(15, 10),
                    geometry_version=1,
                    advance_geometry_to=2,
                ),
            ),
            refusal_code=REFUSAL_CODE_SUPERSESSION,
            attempt=HumanStep(
                HUMAN_STEP, "GOOD", (), _seed(15, 20), supersedes=0, geometry_version=1
            ),
        ),
        _forbidden(
            16,
            "QC001-H-16",
            "unverified provenance (verify_chain fails)",
            (HumanStep(HUMAN_STEP, "GOOD", (), _seed(16, 0)),),
            refusal_code=REFUSAL_CODE_PROVENANCE,
            attempt=HumanStep(
                HUMAN_STEP,
                "GOOD",
                (),
                _seed(16, 10),
                supersedes=0,
                fault="tampered_provenance",
            ),
        ),
    )
    rows = valid + forbidden
    assert len(valid) == VALID_N
    assert len(forbidden) == FORBIDDEN_N
    assert len(rows) == CENSUS_N
    return rows


def valid_human_cases() -> tuple[HumanCase, ...]:
    return tuple(row for row in human_census_cases() if row.valid)


def forbidden_human_cases() -> tuple[HumanCase, ...]:
    return tuple(row for row in human_census_cases() if not row.valid)


def human_probe_cases() -> tuple[HumanCase, ...]:
    """The additional 16a-16c negative probes (they do not change N=16)."""

    base = 8_100_000
    return (
        _forbidden(
            21,
            "QC001-H-16a",
            "invalid actor (not ActorType.HUMAN / wrong actor id)",
            (HumanStep(HUMAN_STEP, "GOOD", (), base),),
            refusal_code=REFUSAL_CODE_ACTOR,
            attempt=HumanStep(
                HUMAN_STEP, "GOOD", (), base + 10, supersedes=0, fault="actor_type"
            ),
        ),
        _forbidden(
            22,
            "QC001-H-16b",
            "invalid editor (HumanEdit editor is not the frozen identity)",
            (HumanStep(HUMAN_STEP, "GOOD", (), base + 100),),
            refusal_code=REFUSAL_CODE_ACTOR,
            attempt=HumanStep(
                HUMAN_STEP, "GOOD", (), base + 110, supersedes=0, fault="editor"
            ),
        ),
        _forbidden(
            23,
            "QC001-H-16c",
            "invalid event type (not the enum member required for the case)",
            (HumanStep(HUMAN_STEP, "GOOD", (), base + 200),),
            refusal_code=REFUSAL_CODE_EVENT,
            attempt=HumanStep(
                HUMAN_STEP,
                "GOOD",
                (),
                base + 210,
                supersedes=0,
                fault="event_type",
                event_type_value=EVENT_INITIAL_HUMAN_VALUE,
            ),
        ),
    )


def clean_census_row(index: int = 0) -> CensusCase:
    """Return the index-th clean unsigned census row used as the source."""

    rows = [
        case
        for case in census_cases()
        if case.valid
        and case.injection is not None
        and case.injection.flag is None
        and case.geometry is not None
        and case.geometry.pixels == "unsigned"
    ]
    return rows[index]


def second_study_census_row() -> CensusCase:
    """A distinct clean row providing the second study of case 13."""

    return clean_census_row(1)


__all__ = [
    "ACTOR_ID_HUMAN",
    "ACTOR_TYPE_VALUE",
    "CANDIDATE_PASSES_ACCEPTED",
    "CENSUS_N",
    "CREATED_BY_VALUE",
    "DEFAULT_EDIT_SUMMARY",
    "DET_STEP",
    "EFFECTIVE_SELECTED_EXACT",
    "EVENT_ACTOR_ROLE",
    "EVENT_ENTITY",
    "EVENT_INITIAL_HUMAN_VALUE",
    "EVENT_HUMAN_CORRECTION_VALUE",
    "EVENT_TYPE_STATUS",
    "FIXED_AT",
    "FORBIDDEN_N",
    "GEOMETRY_STEP",
    "HUMAN_EVIDENCE_KIND",
    "HUMAN_EVIDENCE_SUBJECT_PREFIX",
    "HUMAN_JUSTIFICATION_CODE",
    "HUMAN_STEP",
    "HumanCase",
    "HumanStep",
    "INVALID_ACTOR_ACCEPTED",
    "INVALID_EDITOR_ACCEPTED",
    "INVALID_EVENT_ACCEPTED",
    "MISSING_SUPERSEDES_ACCEPTED",
    "OLD_BYTES_DENOMINATOR",
    "OLD_BYTES_UNCHANGED",
    "OVERWRITES_ACCEPTED",
    "PARAMETERS_NOT_APPLICABLE_REASON",
    "PAYLOAD_KEY_COUNT",
    "PAYLOAD_SCHEMA_VALUE",
    "POST_ABSTAIN_CALLS",
    "PREPROCESSING_VALUE",
    "PRODUCER_VALUE",
    "REFUSAL_CODE_ACTOR",
    "REFUSAL_CODE_EVIDENCE",
    "REFUSAL_CODE_EVENT",
    "REFUSAL_CODE_OVERWRITE",
    "REFUSAL_CODE_PROVENANCE",
    "REFUSAL_CODE_SUPERSESSION",
    "REFUSAL_STEP",
    "SCOPE_VALUE",
    "SOFTWARE_VERSION_VALUE",
    "STALE_REASON_TEMPLATE",
    "STATUS_ACTIVE",
    "STATUS_STALE",
    "STATUS_SUPERSEDED",
    "STALE_DEPENDENT_ACCEPTED",
    "SourceIdentity",
    "VALID_EVIDENCE_COMPLETE",
    "VALID_N",
    "VALID_TRANSITIONS_EXACT",
    "clean_census_row",
    "forbidden_human_cases",
    "human_census_cases",
    "human_probe_cases",
    "second_study_census_row",
    "second_study_seed",
    "source_identity",
    "valid_human_cases",
]
