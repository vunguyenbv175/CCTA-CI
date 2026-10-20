"""Independent census definitions for the QC_001_GATE consumer benchmark.

This module is deliberately an input/recipe generator.  It defines the frozen
N=32 census (eight scenarios crossed with four lineage shapes) and the
deterministic identity arithmetic both the driver and the independent oracle
reuse.  It never imports ``core.qc.gate``, so the oracle stays independent of
the implementation under test.

Every census record is built by the REAL production constructors in the test
driver (``create_real_qc_assessment`` / ``create_qc_input_refusal`` /
``create_qc_assessment``) from REAL synthetic DICOM read through the REAL
reader and materializer.  No fixture short-circuit and no hand-built
assessment exists anywhere in this envelope.

Scope discipline: these are standards-conformant synthetic DICOM fixtures over
the admitted canonical volume geometry.  They are workflow/state-transition
material for the consumer gate only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

# ---------------------------------------------------------------------------
# Frozen census counts (spec 4.5)
# ---------------------------------------------------------------------------

#: Eight scenarios crossed with four lineage shapes; no more, no less.
SCENARIO_N: Final[int] = 8
LINEAGE_N: Final[int] = 4
CENSUS_N: Final[int] = SCENARIO_N * LINEAGE_N  # 32

#: The four lineage shapes, verbatim from the spec.
LINEAGE_SHAPES: Final[tuple[str, ...]] = ("L1", "L2", "L3", "L4")
LINEAGE_LABELS: Final[dict[str, str]] = {
    "L1": "STUDY",
    "L2": "SERIES -> VOLUME",
    "L3": "VOLUME -> MPR",
    "L4": "STUDY -> VOLUME + MPR",
}

#: The eight scenarios in census order.
SCENARIO_IDS: Final[tuple[str, ...]] = (
    "GATE-1",
    "GATE-2",
    "GATE-3",
    "GATE-4",
    "GATE-5",
    "GATE-6",
    "GATE-7",
    "GATE-8",
)

# ---------------------------------------------------------------------------
# Frozen acceptance endpoints (spec 4.5)
# ---------------------------------------------------------------------------

#: ABSTAIN cells that must never become a consumer pass.
UNSAFE_PROCEED_TARGET: Final[int] = 0
#: Every cell must match its expected decision/reason/refs exactly.
GATE_REASON_EXACT_MATCH_TARGET: Final[int] = CENSUS_N
#: Every ABSTAIN cell must produce zero downstream consumer calls.
POST_ABSTAIN_CONSUMER_CALL_TARGET: Final[int] = 0

#: The number of ABSTAIN cells in the census (all scenarios except GATE-1
#: and GATE-7, which are the two PROCEED families), per lineage shape.
ABSTAIN_SCENARIOS: Final[tuple[str, ...]] = (
    "GATE-2",
    "GATE-3",
    "GATE-4",
    "GATE-5",
    "GATE-6",
    "GATE-8",
)
PROCEED_SCENARIOS: Final[tuple[str, ...]] = ("GATE-1", "GATE-7")
ABSTAIN_CELL_N: Final[int] = len(ABSTAIN_SCENARIOS) * LINEAGE_N  # 24
PROCEED_CELL_N: Final[int] = len(PROCEED_SCENARIOS) * LINEAGE_N  # 8

# ---------------------------------------------------------------------------
# Frozen literals mirrored from the closed seams (never imported here)
# ---------------------------------------------------------------------------

PAYLOAD_SCHEMA_VALUE: Final[str] = "ccta.payload.qc_assessment.v1"
V2_PAYLOAD_SCHEMA_VALUE: Final[str] = "ccta.payload.qc_assessment.v2"
UNKNOWN_PAYLOAD_SCHEMA_VALUE: Final[str] = "ccta.payload.qc_assessment.v9"
PLACEHOLDER_JUSTIFICATION_VALUE: Final[str] = "PLACEHOLDER_NO_QC_EVIDENCE"
RULE_ENGINE_JUSTIFICATION_VALUE: Final[str] = "RULE_ENGINE_EVIDENCE"
PRODUCER_VALUE: Final[str] = "DETERMINISTIC_ENGINE"
SCOPE_VALUE: Final[str] = "STUDY"
FIXED_AT: Final[str] = "2026-01-01T00:00:00Z"
ZERO_SHA1: Final[str] = "0" * 40
ZERO_SHA256: Final[str] = "0" * 64

#: The closed ineligible lifecycle statuses exercised by GATE-8.
INELIGIBLE_STATUS_VALUES: Final[tuple[str, ...]] = (
    "STALE",
    "SUPERSEDED",
    "INVALIDATED",
)

#: Ref-kind prefixes of the closed canonical subject-ref grammar.
REF_KIND_STUDY: Final[str] = "study_id"
REF_KIND_SERIES: Final[str] = "series_id"
REF_KIND_VOLUME: Final[str] = "volume_id"
REF_KIND_MPR: Final[str] = "mpr_artifact_id"

#: The canonical order used for every attached-ref assertion.  The subject
#: kinds are ordered study, series, volume, MPR; artifact refs are ordered by
#: their canonical ``qc_artifact:<ULID>`` string.
SUBJECT_KIND_ORDER: Final[dict[str, int]] = {
    REF_KIND_STUDY: 0,
    REF_KIND_SERIES: 1,
    REF_KIND_VOLUME: 2,
    REF_KIND_MPR: 3,
}

# ---------------------------------------------------------------------------
# Deterministic identity arithmetic (shared by driver and oracle)
# ---------------------------------------------------------------------------


def cell_seed(ordinal: int) -> int:
    """The deterministic ID block base of one census cell."""

    return 6_000_000 + ordinal * 1_000


def artifact_id(seed: int) -> str:
    """The artifact ULID of a record built on ``seed``."""

    return ulid(seed)


def provenance_id(seed: int) -> str:
    """The aggregate provenance ULID of a record built on ``seed``."""

    return ulid(seed + 1)


def family_provenance_id(seed: int, index: int) -> str:
    """The provenance ULID of family ``index`` of a record built on ``seed``."""

    return ulid(seed + 10 + index)


def family_provenance_ids(seed: int) -> tuple[str, ...]:
    """The nine frozen family provenance ULIDs, in family order."""

    return tuple(family_provenance_id(seed, index) for index in range(9))


def artifact_ref(seed: int) -> str:
    """The canonical ``qc_artifact:<ULID>`` ref of a record built on ``seed``."""

    return f"qc_artifact:{artifact_id(seed)}"


def placeholder_artifact_ref(seed: int) -> str:
    """The artifact ref of the historical placeholder built on ``seed``."""

    return f"qc_artifact:{ulid(seed)}"


def opaque_artifact_ref(seed: int) -> str:
    """The artifact ref of an opaque (v2/unknown-schema) record."""

    return f"qc_artifact:{ulid(seed)}"


# ---------------------------------------------------------------------------
# Census cell definition
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GateCell:
    """One of the thirty-two QC_001_GATE census cells.

    ``scenario`` is one of the eight GATE families and ``shape`` is one of the
    four lineage shapes.  The driver builds every referenced record through the
    real production constructors; this object carries only the recipe.
    """

    ordinal: int
    cell_id: str
    scenario: str
    shape: str
    label: str
    #: Expected canonical decision (``PROCEED`` or ``ABSTAIN``).
    decision: str
    #: Expected canonical reason, or ``None`` for a PROCEED cell.
    reason: str | None
    #: Whether the cell must attach effective refs (PROCEED and the
    #: conflict/not-analyzable/limited ABSTAIN families do; the
    #: not-assessed family attaches none).
    attaches_refs: bool
    #: The GATE-1 positive subcase grade (GOOD / ACCEPTABLE); the two
    #: subcases live inside GATE-1 and are not a ninth scenario.
    subcase: str | None = None
    #: The GATE-8 subcase variant.
    variant: str | None = None
    #: Whether the cell is an ABSTAIN cell (drives the zero-call barrier).
    abstains: bool = True

    @property
    def scenario_number(self) -> int:
        return int(self.scenario.split("-")[1])


def _cell(
    ordinal: int,
    scenario: str,
    shape: str,
    *,
    decision: str,
    reason: str | None,
    attaches_refs: bool,
    label: str,
    subcase: str | None = None,
    variant: str | None = None,
) -> GateCell:
    return GateCell(
        ordinal=ordinal,
        cell_id=f"{scenario}-{shape}",
        scenario=scenario,
        shape=shape,
        label=f"{label} [{LINEAGE_LABELS[shape]}]",
        decision=decision,
        reason=reason,
        attaches_refs=attaches_refs,
        subcase=subcase,
        variant=variant,
        abstains=decision == "ABSTAIN",
    )


def gate_census_cases() -> tuple[GateCell, ...]:
    """Return exactly the thirty-two Part 4.5 census cells (8 scenarios x 4)."""

    rows: list[GateCell] = []
    ordinal = 0
    for shape in LINEAGE_SHAPES:
        # GATE-1 ordinary good/acceptable.  The two positive subcases
        # (GOOD and ACCEPTABLE) live inside this one scenario family and are
        # exercised as subcases of the cell; they are not a ninth scenario.
        ordinal += 1
        rows.append(
            _cell(
                ordinal,
                "GATE-1",
                shape,
                decision="PROCEED",
                reason=None,
                attaches_refs=True,
                label="ordinary good/acceptable coverage",
            )
        )
        # GATE-2 missing coverage.
        ordinal += 1
        rows.append(
            _cell(
                ordinal,
                "GATE-2",
                shape,
                decision="ABSTAIN",
                reason="QC_NOT_ASSESSED",
                attaches_refs=False,
                label="no eligible QC assessment",
            )
        )
        # GATE-3 legacy placeholder.
        ordinal += 1
        rows.append(
            _cell(
                ordinal,
                "GATE-3",
                shape,
                decision="ABSTAIN",
                reason="QC_NOT_ASSESSED",
                attaches_refs=False,
                label="legacy PLACEHOLDER_NO_QC_EVIDENCE",
            )
        )
        # GATE-4 limited effective coverage.
        ordinal += 1
        rows.append(
            _cell(
                ordinal,
                "GATE-4",
                shape,
                decision="ABSTAIN",
                reason="QC_LIMITED",
                attaches_refs=True,
                label="valid real v1 LIMITED coverage",
            )
        )
        # GATE-5 not-analyzable / unsupported end-to-end.
        ordinal += 1
        rows.append(
            _cell(
                ordinal,
                "GATE-5",
                shape,
                decision="ABSTAIN",
                reason="QC_NOT_ANALYZABLE",
                attaches_refs=True,
                label="unsupported input refusal",
            )
        )
        # GATE-6 incomparable conflict.
        ordinal += 1
        rows.append(
            _cell(
                ordinal,
                "GATE-6",
                shape,
                decision="ABSTAIN",
                reason="QC_CONFLICT",
                attaches_refs=True,
                label="incomparable overlap",
            )
        )
        # GATE-7 valid supersession.
        ordinal += 1
        rows.append(
            _cell(
                ordinal,
                "GATE-7",
                shape,
                decision="PROCEED",
                reason=None,
                attaches_refs=True,
                label="valid explicit supersession",
            )
        )
        # GATE-8 invalid/stale/unknown-schema/provenance.  Its six subcases
        # are exercised inside this one scenario cell (they all yield the
        # same canonical ABSTAIN/QC_NOT_ASSESSED), so the census stays
        # 8 scenarios x 4 shapes = 32 exactly as the spec fixes it.
        ordinal += 1
        rows.append(
            _cell(
                ordinal,
                "GATE-8",
                shape,
                decision="ABSTAIN",
                reason="QC_NOT_ASSESSED",
                attaches_refs=False,
                label="invalid/stale/unknown-schema/provenance",
            )
        )

    assert len(rows) == CENSUS_N, f"the census must hold exactly {CENSUS_N} cells"
    return tuple(rows)


def gate_census_cells(scenario: str) -> tuple[GateCell, ...]:
    """All four lineage-shape cells of one scenario."""

    return tuple(cell for cell in gate_census_cases() if cell.scenario == scenario)


def abstain_cells() -> tuple[GateCell, ...]:
    """The twenty-four ABSTAIN cells (the zero-call barrier population)."""

    return tuple(cell for cell in gate_census_cases() if cell.abstains)


def proceed_cells() -> tuple[GateCell, ...]:
    """The eight PROCEED cells."""

    return tuple(cell for cell in gate_census_cases() if not cell.abstains)


def gate8_cells() -> tuple[GateCell, ...]:
    """The four GATE-8 cells (one per lineage shape)."""

    return tuple(cell for cell in gate_census_cases() if cell.scenario == "GATE-8")


#: The six GATE-8 subcases.  They live INSIDE the single GATE-8 scenario cell
#: (every one yields the canonical ``ABSTAIN``/``QC_NOT_ASSESSED``), and each
#: is asserted separately so the required diagnostic is visible.
GATE8_SUBCASES: Final[tuple[str, ...]] = (
    "v2-at-v1",
    "unknown-schema",
    "malformed-provenance",
    "stale-only",
    "invalidated-only",
    "malformed-supersession",
)

#: The diagnostic EVENT code each GATE-8 subcase must surface.  These are
#: event codes, NOT members of the closed ``QCReason`` enum.
GATE8_SUBCASE_DIAGNOSTICS: Final[dict[str, str]] = {
    "v2-at-v1": "QC_SCHEMA_INCOMPATIBLE",
    "unknown-schema": "QC_SCHEMA_INVALID",
    "malformed-provenance": "QC_PROVENANCE_INVALID",
    "stale-only": "QC_RECORD_INELIGIBLE",
    "invalidated-only": "QC_RECORD_INELIGIBLE",
    "malformed-supersession": "QC_SUPERSESSION_INVALID",
}

#: The GATE-8 subcases whose diagnostic proves provenance invalidity.
GATE8_PROVENANCE_SUBCASES: Final[tuple[str, ...]] = ("malformed-provenance",)


def gate8_subcases() -> tuple[str, ...]:
    """The six GATE-8 subcases exercised inside each GATE-8 cell."""

    return GATE8_SUBCASES


#: The two positive GATE-1 subcases.  "ordinary GOOD/ACCEPTABLE means two
#: positive subcases in one scenario family; it is not a ninth scenario."
GATE1_SUBCASES: Final[tuple[str, ...]] = ("GOOD", "ACCEPTABLE")


def gate1_subcases() -> tuple[str, ...]:
    """The two positive subcases exercised inside each GATE-1 cell."""

    return GATE1_SUBCASES


# ---------------------------------------------------------------------------
# The five pre-GATE integration-checkpoint classes (spec 4.5)
# ---------------------------------------------------------------------------

#: The five end-to-end classes the checkpoint must pass through the REAL
#: reader/producer -> persistence -> effective selector -> gate -> consumer.
CHECKPOINT_CLASSES: Final[tuple[str, ...]] = (
    "legacy-placeholder",
    "unknown-schema",
    "conflicting-or-superseded",
    "malformed-provenance",
    "unsupported-inputs",
)


def ulid(number: int) -> str:
    """Encode a small integer as a deterministic 26-character ULID.

    Mirrors the QC_001-D fixture helper so IDs stay stable across the
    driver, the oracle, and the report.
    """

    alphabet = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
    if number < 0:
        raise ValueError("ULID number must be nonnegative")
    value = int(number)
    chars = ["0"] * 26
    for index in range(25, -1, -1):
        chars[index] = alphabet[value & 31]
        value >>= 5
    if value:
        raise ValueError("ULID number is too large")
    return "".join(chars)


__all__ = [
    "ABSTAIN_CELL_N",
    "ABSTAIN_SCENARIOS",
    "CHECKPOINT_CLASSES",
    "CENSUS_N",
    "FIXED_AT",
    "GATE1_SUBCASES",
    "GATE8_PROVENANCE_SUBCASES",
    "GATE8_SUBCASE_DIAGNOSTICS",
    "GATE8_SUBCASES",
    "GATE_REASON_EXACT_MATCH_TARGET",
    "INELIGIBLE_STATUS_VALUES",
    "LINEAGE_LABELS",
    "LINEAGE_N",
    "LINEAGE_SHAPES",
    "PAYLOAD_SCHEMA_VALUE",
    "PLACEHOLDER_JUSTIFICATION_VALUE",
    "POST_ABSTAIN_CONSUMER_CALL_TARGET",
    "PROCEED_CELL_N",
    "PROCEED_SCENARIOS",
    "PRODUCER_VALUE",
    "REF_KIND_MPR",
    "REF_KIND_SERIES",
    "REF_KIND_STUDY",
    "REF_KIND_VOLUME",
    "RULE_ENGINE_JUSTIFICATION_VALUE",
    "SCENARIO_IDS",
    "SCENARIO_N",
    "SCOPE_VALUE",
    "SUBJECT_KIND_ORDER",
    "UNKNOWN_PAYLOAD_SCHEMA_VALUE",
    "UNSAFE_PROCEED_TARGET",
    "V2_PAYLOAD_SCHEMA_VALUE",
    "ZERO_SHA1",
    "ZERO_SHA256",
    "GateCell",
    "abstain_cells",
    "artifact_id",
    "artifact_ref",
    "cell_seed",
    "family_provenance_id",
    "family_provenance_ids",
    "gate8_cells",
    "gate1_subcases",
    "gate8_subcases",
    "gate_census_cases",
    "gate_census_cells",
    "opaque_artifact_ref",
    "placeholder_artifact_ref",
    "proceed_cells",
    "provenance_id",
    "ulid",
]
