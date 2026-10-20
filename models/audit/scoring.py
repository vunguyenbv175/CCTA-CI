"""Risk-adjusted scoring and deterministic ranking (WS01 deliverable #2).

Aggregation formula (documented, deterministic):

    total = Σ ( dimension_score * weight )

where ``weight`` is drawn from the active weights-version map.
Default ``v1`` uses uniform weights (1.0 per dimension) so ranking
reflects raw clinical+operational fitness; future versions may
risk-weight security/license dimensions higher without changing
record schema. ``weights_version`` is stored on each record so
ranking is reproducible.

Higher total = better candidate. Tie-breaks are deterministic:
candidate_name lexicographically, then version lexicographically.

Weights map is versioned; only known versions are accepted.
"""

from __future__ import annotations

from collections.abc import Iterable

from .record import DIMENSION_NAMES, TechnologyAuditRecord

WEIGHTS_VERSION = "v1"

#: Weights per version. ``v1`` uniform — change is a version bump.
DIMENSION_WEIGHTS: dict[str, dict[str, float]] = {
    "v1": {name: 1.0 for name in DIMENSION_NAMES},
    # Example future risk-adjusted version (not active by default):
    # v2 would weight security/license higher:
    # "v2": {
    #     "clinical_technical_evidence": 1.0,
    #     "license_redistribution": 1.5,
    #     "reproducibility_auditability": 1.0,
    #     "security_dependency_health": 1.5,
    #     "local_offline_operability": 1.2,
    #     "maintainability": 1.0,
    #     "integration_cost": 1.0,
    # },
}

__all__ = [
    "WEIGHTS_VERSION",
    "DIMENSION_WEIGHTS",
    "audit_score",
    "rank_audits",
    "ranked_with_scores",
]


def _weights_for(version: str) -> dict[str, float]:
    if version not in DIMENSION_WEIGHTS:
        raise ValueError(f"unknown weights_version {version!r}; known: {sorted(DIMENSION_WEIGHTS)}")
    return DIMENSION_WEIGHTS[version]


def audit_score(
    record: TechnologyAuditRecord,
    *,
    weights_version: str | None = None,
) -> float:
    """Compute risk-adjusted total for *record*.

    Uses ``record.weights_version`` unless an explicit ``weights_version``
    override is given (for comparative analysis). Formula is documented
    at module top: weighted sum of 0–5 dimension scores.
    """
    wv = weights_version or record.weights_version
    weights = _weights_for(wv)
    total = 0.0
    for name in DIMENSION_NAMES:
        dim = getattr(record, name)
        # DimensionScore is int subclass, so int(dim) is score
        total += float(int(dim)) * float(weights[name])
    return total


def ranked_with_scores(
    records: Iterable[TechnologyAuditRecord],
    *,
    weights_version: str | None = None,
) -> list[tuple[TechnologyAuditRecord, float]]:
    """Return ``(record, score)`` pairs ranked descending.

    Deterministic tie-breaks: ``( -score, candidate_name, version )``.
    """
    scored = [
        (rec, audit_score(rec, weights_version=weights_version)) for rec in records
    ]
    scored.sort(key=lambda pair: (-pair[1], pair[0].candidate_name, pair[0].version))
    return scored


def rank_audits(
    records: Iterable[TechnologyAuditRecord],
    *,
    weights_version: str | None = None,
) -> list[TechnologyAuditRecord]:
    """Deterministic ranking — higher total first, name then version tie-break."""
    return [rec for rec, _ in ranked_with_scores(records, weights_version=weights_version)]
