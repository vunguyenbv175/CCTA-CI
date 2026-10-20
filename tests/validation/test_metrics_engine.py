"""§48 workflow metrics engine tests (WS02 §B9)."""

from __future__ import annotations

import pytest

from core.artifacts import DerivedArtifactStore, QualityBlock, ReviewBlock
from core.enums import ArtifactKind, Confidence, QualityGrade, ReviewState
from core.events import EventLog, StateEvent
from core.provenance import InvalidationDAG
from tests.core.factories import make_artifact
from validation.metrics.engine import STRATA_FIELDS, StratumFields, WorkflowMetricsEngine

T0 = "2026-01-15T10:00:00Z"
T1 = "2026-01-15T10:05:00Z"
T2 = "2026-01-15T10:20:00Z"


def _artifact_with_history(
    log: EventLog,
    *,
    artifact_id: str,
    transitions: list[tuple[str, str, str]],
    grade: QualityGrade | None = None,
    confidence: Confidence | None = None,
):
    """Register an artifact and replay REVIEW transitions into the journal."""
    artifact = make_artifact(
        ArtifactKind.MEASUREMENT_SET,
        artifact_id=artifact_id,
        review=ReviewBlock(review_state=ReviewState.PREDICTED),
        quality=QualityBlock(quality_grade=grade, confidence=confidence),
        created_at=T0,
    )
    store = DerivedArtifactStore(log, InvalidationDAG())
    seeded = store.add(artifact)
    current = "PREDICTED"
    for to_state, actor_role, stamp in transitions:
        log.append_event(
            StateEvent(
                entity="derived_artifact",
                entity_id=artifact_id,
                event_type="REVIEW",
                from_value=current,
                to_value=to_state,
                actor_role=actor_role,
                at=stamp,
            )
        )
        current = to_state
    return seeded


def test_metrics_from_review_transitions() -> None:
    log = EventLog()
    good = _artifact_with_history(
        log, artifact_id="art-good", transitions=[("ACCEPTED", "HUMAN", T1)]
    )
    corrected = _artifact_with_history(
        log,
        artifact_id="art-corrected",
        transitions=[("CORRECTED", "HUMAN", T1), ("ACCEPTED", "HUMAN", T2)],
    )
    abstained = _artifact_with_history(
        log,
        artifact_id="art-abstained",
        transitions=[("REJECTED", "HUMAN", T1)],
        grade=QualityGrade.NOT_ANALYZABLE,
        confidence=Confidence.UNRELIABLE,
    )

    engine = WorkflowMetricsEngine(
        log, [good, corrected, abstained], verification_verdict="PASS"
    )
    metrics = engine.compute().to_dict()

    assert metrics["artifact_count"] == 3
    assert metrics["correction_frequency"] == pytest.approx(1 / 3)
    assert metrics["abstention_rate"] == pytest.approx(1 / 3)
    assert metrics["reopen_recompute_reproducibility"] == 1.0
    # time-to-first-reviewable: art-good accepted at T1 (+300s), corrected at T2 (+1200s)
    assert metrics["time_to_first_reviewable_result_seconds"] == 300.0
    # human override (correction or rejection) happened on 2 of 3 artifacts
    assert metrics["radiologist_override_rate"] == pytest.approx(2 / 3)


def test_stratified_fields_are_structural_stubs() -> None:
    document = StratumFields().to_dict()
    assert tuple(document) == STRATA_FIELDS
    assert all(value is None for value in document.values())

    metrics = WorkflowMetricsEngine(EventLog(), []).compute()
    stratified = metrics.to_dict()["stratified"]
    assert isinstance(stratified, dict)
    assert stratified["populated"] is False


def test_reproducibility_tracks_exit_gate_verdict() -> None:
    failing = WorkflowMetricsEngine(EventLog(), [], verification_verdict="FAIL")
    assert failing.compute().reopen_recompute_reproducibility == 0.0
    without_report = WorkflowMetricsEngine(EventLog(), [])
    assert without_report.compute().reopen_recompute_reproducibility is None


def test_empty_session_metrics_are_zero_safe() -> None:
    metrics = WorkflowMetricsEngine(EventLog(), []).compute().to_dict()
    assert metrics["correction_frequency"] == 0.0
    assert metrics["abstention_rate"] == 0.0
    assert metrics["time_to_first_reviewable_result_seconds"] is None


def test_timestamps_used_by_engine_are_iso_utc() -> None:
    from validation.metrics.engine import _parse_iso

    assert _parse_iso(T0).isoformat().endswith("+00:00")
