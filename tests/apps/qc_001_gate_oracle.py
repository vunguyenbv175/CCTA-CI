"""Independent QC_001_GATE expectations (test-only; NO ``core.qc.gate`` import).

The oracle derives the expected decision, reason, effective-ref attachment
set, and required visible diagnostics for each of the thirty-two census cells
ONLY from the fixture definitions in ``qc_001_gate_fixtures`` and the literals
transcribed there.  It never imports or calls the module under test; the
census proves that import boundary mechanically (AST scan + line scan).

It does import the closed gate vocabulary (``QCGateDecision``, ``QCReason``)
from ``core.dicom.qc`` because that is the frozen closed seam the census
asserts against, not the implementation under test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from core.dicom.qc import QCGateDecision, QCReason
from tests.apps.qc_001_gate_fixtures import (
    ABSTAIN_CELL_N,
    CENSUS_N,
    GATE1_SUBCASES,
    GATE8_SUBCASE_DIAGNOSTICS,
    GATE8_SUBCASES,
    LINEAGE_N,
    SCENARIO_IDS,
    SCENARIO_N,
    GateCell,
    artifact_ref,
    cell_seed,
)

# ---------------------------------------------------------------------------
# The canonical reason set (closed four; spec 2.4)
# ---------------------------------------------------------------------------

#: The closed ``QCReason`` members.  Diagnostic codes are NOT members and are
#: never added to this set.
CLOSED_REASONS: Final[frozenset[str]] = frozenset(
    {
        QCReason.QC_NOT_ASSESSED.value,
        QCReason.QC_LIMITED.value,
        QCReason.QC_NOT_ANALYZABLE.value,
        QCReason.QC_CONFLICT.value,
    }
)

#: Diagnostic EVENT codes (never enum members).
DIAG_SCHEMA_INCOMPATIBLE: Final[str] = "QC_SCHEMA_INCOMPATIBLE"
DIAG_SCHEMA_INVALID: Final[str] = "QC_SCHEMA_INVALID"
DIAG_LEGACY_PLACEHOLDER: Final[str] = "QC_LEGACY_PLACEHOLDER"
DIAG_PROVENANCE_INVALID: Final[str] = "QC_PROVENANCE_INVALID"
DIAG_RECORD_INELIGIBLE: Final[str] = "QC_RECORD_INELIGIBLE"
DIAG_SUPERSESSION_INVALID: Final[str] = "QC_SUPERSESSION_INVALID"
DIAG_CONFLICT: Final[str] = "QC_CONFLICT"

#: The canonical decisions.
DECISION_PROCEED: Final[str] = QCGateDecision.PROCEED.value
DECISION_ABSTAIN: Final[str] = QCGateDecision.ABSTAIN.value

# ---------------------------------------------------------------------------
# Per-scenario expectations (transcribed from spec 4.5)
# ---------------------------------------------------------------------------

#: Expected (decision, reason) per scenario.  ``reason`` is ``None`` exactly
#: for a PROCEED cell.
SCENARIO_EXPECTATIONS: Final[dict[str, tuple[str, str | None]]] = {
    "GATE-1": (DECISION_PROCEED, None),
    "GATE-2": (DECISION_ABSTAIN, QCReason.QC_NOT_ASSESSED.value),
    "GATE-3": (DECISION_ABSTAIN, QCReason.QC_NOT_ASSESSED.value),
    "GATE-4": (DECISION_ABSTAIN, QCReason.QC_LIMITED.value),
    "GATE-5": (DECISION_ABSTAIN, QCReason.QC_NOT_ANALYZABLE.value),
    "GATE-6": (DECISION_ABSTAIN, QCReason.QC_CONFLICT.value),
    "GATE-7": (DECISION_PROCEED, None),
    "GATE-8": (DECISION_ABSTAIN, QCReason.QC_NOT_ASSESSED.value),
}

#: Scenarios whose ABSTAIN output still attaches the effective refs in
#: canonical order (conflict, not-analyzable, limited).  A not-assessed
#: ABSTAIN attaches nothing: there is no eligible effective coverage.
SCENARIO_ATTACHES_REFS: Final[dict[str, bool]] = {
    "GATE-1": True,
    "GATE-2": False,
    "GATE-3": False,
    "GATE-4": True,
    "GATE-5": True,
    "GATE-6": True,
    "GATE-7": True,
    "GATE-8": False,
}

#: The diagnostic each scenario must surface.  ``None`` means no diagnostic
#: is required (an ordinary PROCEED or a plain missing-coverage ABSTAIN).
SCENARIO_DIAGNOSTICS: Final[dict[str, str | None]] = {
    "GATE-1": None,
    "GATE-2": None,
    "GATE-3": DIAG_LEGACY_PLACEHOLDER,
    "GATE-4": None,
    "GATE-5": None,
    "GATE-6": DIAG_CONFLICT,
    "GATE-7": None,
    "GATE-8": None,
}

#: The number of effective refs a cell attaches, per scenario.  A conflict
#: cell surfaces BOTH incomparable maxima; every other attaching cell
#: attaches the single effective maximum (one record covers every consumed
#: ref of the shape).
SCENARIO_EFFECTIVE_REF_COUNT: Final[dict[str, int]] = {
    "GATE-1": 1,
    "GATE-2": 0,
    "GATE-3": 0,
    "GATE-4": 1,
    "GATE-5": 1,
    "GATE-6": 2,
    "GATE-7": 1,
    "GATE-8": 0,
}

#: The grade each GATE-1 subcase must carry.
GATE1_SUBCASE_GRADES: Final[dict[str, str]] = {
    "GOOD": "GOOD",
    "ACCEPTABLE": "ACCEPTABLE",
}

#: F5: which candidate record of each cell supplies the effective coverage,
#: expressed as the seed OFFSET from the cell's own ID block
#: (``cell_seed(cell.ordinal)``).  The oracle freezes the exact expected ref
#: SET (not a count) and the census asserts equality to it.
#:
#: +0     -> the cell's own primary record
#: +200   -> the cell's second record (conflict partner / successor)
#: None   -> no record supplies coverage
EFFECTIVE_SEED_OFFSETS: Final[dict[str, tuple[int | None, ...]]] = {
    "GATE-1": (0,),
    "GATE-2": (),
    "GATE-3": (),
    "GATE-4": (0,),
    "GATE-5": (0,),
    # Incomparable overlap: BOTH maxima are surfaced in canonical order.
    "GATE-6": (0, 200),
    # A valid supersession: the SUCCESSOR is attached, the predecessor is not.
    "GATE-7": (200,),
    "GATE-8": (),
}


def expected_effective_refs(cell: GateCell) -> tuple[str, ...]:
    """The EXACT expected effective-ref set, in canonical artifact-ref order.

    Derived only from the fixture ID arithmetic; the census compares the
    gate's ``assessment_refs`` to this tuple with plain equality, so a wrong
    ref, a missing ref, an extra ref, or a wrong order all fail.
    """

    seed = cell_seed(cell.ordinal)
    refs = tuple(
        artifact_ref(seed + offset)
        for offset in EFFECTIVE_SEED_OFFSETS[cell.scenario]
        if offset is not None
    )
    return tuple(sorted(refs))


@dataclass(frozen=True, slots=True)
class CellExpectation:
    """Everything the census asserts for one of the thirty-two cells."""

    cell: GateCell
    decision: str
    reason: str | None
    attaches_refs: bool
    effective_ref_count: int
    required_diagnostic: str | None
    #: Per-subcase required diagnostics (GATE-8 subcase battery).
    subcase_diagnostics: tuple[tuple[str, str], ...]

    @property
    def abstains(self) -> bool:
        return self.decision == DECISION_ABSTAIN

    @property
    def cell_id(self) -> str:
        return cell_id(self.cell)


def cell_id(cell: GateCell) -> str:
    """The stable census node id of one cell."""

    return f"{cell.scenario}-{cell.shape}"


def expected_decision(cell: GateCell) -> str:
    """The exact canonical decision of one cell."""

    decision, _ = SCENARIO_EXPECTATIONS[cell.scenario]
    return decision


def expected_reason(cell: GateCell) -> str | None:
    """The exact canonical reason of one cell (``None`` on PROCEED)."""

    _, reason = SCENARIO_EXPECTATIONS[cell.scenario]
    return reason


def expected_effective_ref_count(cell: GateCell) -> int:
    """How many effective refs the cell must attach, in canonical order."""

    return SCENARIO_EFFECTIVE_REF_COUNT[cell.scenario]


def expected_diagnostic(cell: GateCell) -> str | None:
    """The diagnostic the cell must surface, if any."""

    return SCENARIO_DIAGNOSTICS[cell.scenario]


def cell_expectation(cell: GateCell) -> CellExpectation:
    """Derive the complete expectation of one census cell from the fixtures."""

    subcase_diagnostics: tuple[tuple[str, str], ...] = ()
    if cell.scenario == "GATE-8":
        subcase_diagnostics = tuple(
            (subcase, GATE8_SUBCASE_DIAGNOSTICS[subcase]) for subcase in GATE8_SUBCASES
        )
    return CellExpectation(
        cell=cell,
        decision=expected_decision(cell),
        reason=expected_reason(cell),
        attaches_refs=SCENARIO_ATTACHES_REFS[cell.scenario],
        effective_ref_count=expected_effective_ref_count(cell),
        required_diagnostic=expected_diagnostic(cell),
        subcase_diagnostics=subcase_diagnostics,
    )


def gate1_subcase_grades() -> tuple[tuple[str, str], ...]:
    """The two positive GATE-1 subcases and their grades."""

    return tuple((subcase, GATE1_SUBCASE_GRADES[subcase]) for subcase in GATE1_SUBCASES)


def gate8_subcase_diagnostics() -> tuple[tuple[str, str], ...]:
    """The six GATE-8 subcases and the diagnostic each must surface."""

    return tuple(
        (subcase, GATE8_SUBCASE_DIAGNOSTICS[subcase]) for subcase in GATE8_SUBCASES
    )


# ---------------------------------------------------------------------------
# Frozen rate expectations (spec 4.5)
# ---------------------------------------------------------------------------

#: An ABSTAIN cell must never become a consumer pass.
EXPECTED_UNSAFE_PROCEED_RATE: Final[float] = 0.0
#: Every cell must match its expected decision/reason/refs exactly.
EXPECTED_REASON_EXACT_MATCH_RATE: Final[float] = 1.0
#: Every ABSTAIN cell must perform zero downstream consumer calls.
EXPECTED_POST_ABSTAIN_CALL_COUNT: Final[int] = 0

#: The denominators the frozen rates are computed over.
ABSTAIN_DENOMINATOR: Final[int] = ABSTAIN_CELL_N
CENSUS_DENOMINATOR: Final[int] = CENSUS_N


def expected_census_shape() -> tuple[tuple[str, object], ...]:
    """The census-shape facts the architecture test asserts literally."""

    return (
        ("census_n", CENSUS_N),
        ("scenario_n", SCENARIO_N),
        ("lineage_n", LINEAGE_N),
        ("scenario_ids", SCENARIO_IDS),
        ("abstain_cell_n", ABSTAIN_CELL_N),
    )


__all__ = [
    "ABSTAIN_DENOMINATOR",
    "CENSUS_DENOMINATOR",
    "CLOSED_REASONS",
    "DECISION_ABSTAIN",
    "DECISION_PROCEED",
    "DIAG_CONFLICT",
    "DIAG_LEGACY_PLACEHOLDER",
    "DIAG_PROVENANCE_INVALID",
    "DIAG_RECORD_INELIGIBLE",
    "DIAG_SCHEMA_INCOMPATIBLE",
    "DIAG_SCHEMA_INVALID",
    "DIAG_SUPERSESSION_INVALID",
    "EXPECTED_POST_ABSTAIN_CALL_COUNT",
    "EXPECTED_REASON_EXACT_MATCH_RATE",
    "EXPECTED_UNSAFE_PROCEED_RATE",
    "GATE1_SUBCASE_GRADES",
    "SCENARIO_ATTACHES_REFS",
    "SCENARIO_DIAGNOSTICS",
    "EFFECTIVE_SEED_OFFSETS",
    "SCENARIO_EFFECTIVE_REF_COUNT",
    "SCENARIO_EXPECTATIONS",
    "CellExpectation",
    "cell_expectation",
    "cell_id",
    "expected_census_shape",
    "expected_decision",
    "expected_diagnostic",
    "expected_effective_ref_count",
    "expected_effective_refs",
    "expected_reason",
    "gate1_subcase_grades",
    "gate8_subcase_diagnostics",
]
