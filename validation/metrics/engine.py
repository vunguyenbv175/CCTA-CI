"""Workflow metric vocabulary from EventLog replays (Master Plan §48, §B9).

Metrics are a MEASUREMENT vocabulary only: they feed validation evidence and
never hold promotion authority (Rule H stays in ``models.registry``).

Stratified reporting hooks (§49) are stubbed STRUCTURALLY: the field set is
fixed now so later modules can populate it without schema breakage; Phase 0
leaves them unset.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Final

from core.artifacts import DerivedArtifact
from core.enums import Confidence, QualityGrade, ReviewState
from core.events import EventLog

STRATA_SCHEMA: Final[str] = "ccta.validation.strata.v1"

#: §49 stratification fields — structural stubs (Part B9).
STRATA_FIELDS: Final[tuple[str, ...]] = (
    "scanner_vendor",
    "scanner_generation",
    "reconstruction_kernel",
    "slice_thickness_mm",
    "heart_rate_bpm",
    "motion_grade",
    "calcium_grade",
    "contrast_quality",
    "anatomy_region",
    "vessel_size_mm",
    "lesion_type",
    "protocol",
)


@dataclass(frozen=True)
class StratumFields:
    """Fixed §49 stratum slots; None until a later module populates them."""

    scanner_vendor: str | None = None
    scanner_generation: str | None = None
    reconstruction_kernel: str | None = None
    slice_thickness_mm: float | None = None
    heart_rate_bpm: float | None = None
    motion_grade: str | None = None
    calcium_grade: str | None = None
    contrast_quality: str | None = None
    anatomy_region: str | None = None
    vessel_size_mm: float | None = None
    lesion_type: str | None = None
    protocol: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in STRATA_FIELDS}


@dataclass(frozen=True)
class WorkflowMetrics:
    """§48 personal-workflow metric snapshot."""

    artifact_count: int
    reviewed_artifact_count: int
    time_to_first_reviewable_result_seconds: float | None
    correction_frequency: float
    correction_magnitude_samples: tuple[float, ...]
    radiologist_override_rate: float | None
    abstention_rate: float
    reopen_recompute_reproducibility: float | None
    stratified: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "abstention_rate": self.abstention_rate,
            "artifact_count": self.artifact_count,
            "correction_frequency": self.correction_frequency,
            "correction_magnitude_samples": list(self.correction_magnitude_samples),
            "radiologist_override_rate": self.radiologist_override_rate,
            "reopen_recompute_reproducibility": self.reopen_recompute_reproducibility,
            "reviewed_artifact_count": self.reviewed_artifact_count,
            "schema": "ccta.validation.workflow_metrics.v1",
            "stratified": dict(self.stratified),
            "time_to_first_reviewable_result_seconds": (
                self.time_to_first_reviewable_result_seconds
            ),
        }


def _parse_iso(value: str) -> datetime:
    text = value[:-1] + "+00:00" if value.endswith("Z") else value
    return datetime.fromisoformat(text)


def _is_not_analyzable(artifact: DerivedArtifact) -> bool:
    grade = artifact.quality.quality_grade
    confidence = artifact.quality.confidence
    return grade is QualityGrade.NOT_ANALYZABLE or confidence is Confidence.NOT_ANALYZABLE


def stratify(
    items: Sequence[object], selector: Callable[[object], StratumFields | None]
) -> dict[str, object]:
    """Structural stratification skeleton (§49): groups + empty-by-default."""
    buckets: dict[tuple[object, ...], int] = {}
    for item in items:
        fields_obj = selector(item)
        if fields_obj is None:
            continue
        key = tuple(fields_obj.to_dict().values())
        if any(part is not None for part in key):
            buckets[key] = buckets.get(key, 0) + 1
    return {
        "schema": STRATA_SCHEMA,
        "fields": list(STRATA_FIELDS),
        "populated": bool(buckets),
        "buckets": {"|".join(str(p) for p in key): count for key, count in sorted(buckets.items())},
    }


class WorkflowMetricsEngine:
    """Computes §48 metrics from a replayed journal + restored artifacts."""

    def __init__(
        self,
        log: EventLog,
        artifacts: Sequence[DerivedArtifact],
        *,
        verification_verdict: str | None = None,
    ) -> None:
        self._log = log
        self._artifacts = tuple(artifacts)
        self._verdict = verification_verdict

    def compute(self) -> WorkflowMetrics:
        total = len(self._artifacts)
        reviewable_times: list[float] = []
        corrected = 0
        overridden = 0
        reviewed = 0
        abstentions = 0

        for artifact in self._artifacts:
            history = self._log.history_for(artifact.artifact_id, entity="derived_artifact")
            created = _parse_iso(artifact.created_at) if artifact.created_at else None
            has_human_override = False
            reached_reviewable = False
            for event in history:
                if event.event_type != "REVIEW":
                    continue
                to_state = event.to_value
                if to_state == ReviewState.CORRECTED.value:
                    corrected += 1
                if (
                    to_state in (ReviewState.CORRECTED.value, ReviewState.REJECTED.value)
                    and event.actor_role == "HUMAN"
                ):
                    has_human_override = True
                if to_state in (ReviewState.ACCEPTED.value, ReviewState.CONFIRMED_FINAL.value):
                    if not reached_reviewable and created is not None:
                        reviewable_times.append((_parse_iso(event.at) - created).total_seconds())
                    reached_reviewable = True
            if reached_reviewable or history:
                reviewed += 1
            if has_human_override:
                overridden += 1
            if _is_not_analyzable(artifact):
                abstentions += 1

        reproducibility: float | None = None
        if self._verdict is not None:
            reproducibility = 1.0 if self._verdict == "PASS" else 0.0

        return WorkflowMetrics(
            artifact_count=total,
            reviewed_artifact_count=reviewed,
            time_to_first_reviewable_result_seconds=(
                min(reviewable_times) if reviewable_times else None
            ),
            correction_frequency=corrected / total if total else 0.0,
            # Magnitude needs payload-level deltas; structurally stubbed (§48).
            correction_magnitude_samples=(),
            radiologist_override_rate=overridden / total if total else None,
            abstention_rate=abstentions / total if total else 0.0,
            reopen_recompute_reproducibility=reproducibility,
            stratified=stratify(self._artifacts, lambda _a: StratumFields()),
        )


__all__ = [
    "STRATA_FIELDS",
    "STRATA_SCHEMA",
    "StratumFields",
    "WorkflowMetrics",
    "WorkflowMetricsEngine",
    "stratify",
]
