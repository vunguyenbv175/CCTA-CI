"""models.audit — technology-audit workflow (Master Plan §3, §76-9).

Public surface re-exports the closed-schema record, ledger, scoring
helper and seed builder so callers can ``from models.audit import ...``.

All persistence and hashing goes through :mod:`core.digest` and
:mod:`core.events`; no new runtime dependencies are introduced.
"""

from __future__ import annotations

from .ledger import LEDGER_SCHEMA, AuditLedger
from .record import (
    AUDIT_SCHEMA,
    REUSE_PREFERENCE_ORDER,
    TRANSIENT_AUDIT_FIELDS,
    AuditStatus,
    DimensionScore,
    ProvenanceRef,
    ReuseDecision,
    StatusHistoryEntry,
    TechnologyAuditRecord,
    compute_audit_digest,
    load_audit_record,
    record_from_dict,
    record_to_dict,
    validate_audit_record,
    verify_audit_digest,
)
from .scoring import (
    DIMENSION_WEIGHTS,
    WEIGHTS_VERSION,
    audit_score,
    rank_audits,
    ranked_with_scores,
)
from .seeds import (
    EMBEDDED_REF,
    SEED_RECORDS,
    build_seed_ledger,
    get_seed_ledger,
)

__all__ = [
    # record
    "AUDIT_SCHEMA",
    "TRANSIENT_AUDIT_FIELDS",
    "AuditStatus",
    "DimensionScore",
    "ProvenanceRef",
    "ReuseDecision",
    "REUSE_PREFERENCE_ORDER",
    "StatusHistoryEntry",
    "TechnologyAuditRecord",
    "compute_audit_digest",
    "load_audit_record",
    "record_from_dict",
    "record_to_dict",
    "validate_audit_record",
    "verify_audit_digest",
    # ledger
    "AuditLedger",
    "LEDGER_SCHEMA",
    # scoring
    "DIMENSION_WEIGHTS",
    "WEIGHTS_VERSION",
    "audit_score",
    "rank_audits",
    "ranked_with_scores",
    # seeds
    "EMBEDDED_REF",
    "SEED_RECORDS",
    "build_seed_ledger",
    "get_seed_ledger",
]
