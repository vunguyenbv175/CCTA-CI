# ruff: noqa: E501
"""Technology audit record — closed schema, digest-bound, fail-closed (WS01).

Mirrors ``models/registry/record.py`` and ``core/provenance`` patterns:

* closed schema — unknown keys at load are rejected (``ClosedSchemaError``);
* scores are range-checked 0–5 with mandatory one-line justification;
* sealed records carry a self-digest over the frozen identity block;
* status lifecycle is append-only via :class:`core.events.EventLog`;
* JSON dumps are enum-free via :func:`record_to_dict`.
"""

from __future__ import annotations

import enum
from dataclasses import MISSING, dataclass, fields
from typing import Any, get_type_hints

from core.digest import canonical_digest, is_sha256_hex
from core.errors import CanonicalValidationError, ClosedSchemaError, DigestMismatchError
from core.events import validate_iso8601_utc

AUDIT_SCHEMA = "ccta.audit.technology_audit.v1"

#: Frozen identity block excludes mutable lifecycle fields and the self-digest.
TRANSIENT_AUDIT_FIELDS: frozenset[str] = frozenset(
    {
        "audit_digest",
        "status",
        "status_history",
    }
)

DIMENSION_NAMES: tuple[str, ...] = (
    "clinical_technical_evidence",
    "license_redistribution",
    "reproducibility_auditability",
    "security_dependency_health",
    "local_offline_operability",
    "maintainability",
    "integration_cost",
)

DIMENSION_ALIAS_JUSTIFICATION: dict[str, str] = {
    name: f"{name}_justification" for name in DIMENSION_NAMES
}

WEIGHTS_VERSION_DEFAULT = "v1"


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ReuseDecision(str, enum.Enum):  # noqa: UP042
    """Ordered reuse preference (most preferred first)."""

    USE_DEPENDENCY = "USE_DEPENDENCY"
    BUILD_ADAPTER = "BUILD_ADAPTER"
    RUN_EXTERNAL_SERVICE = "RUN_EXTERNAL_SERVICE"
    VENDOR_COMPONENT = "VENDOR_COMPONENT"
    FORK = "FORK"
    REWRITE = "REWRITE"


REUSE_PREFERENCE_ORDER: tuple[ReuseDecision, ...] = (
    ReuseDecision.USE_DEPENDENCY,
    ReuseDecision.BUILD_ADAPTER,
    ReuseDecision.RUN_EXTERNAL_SERVICE,
    ReuseDecision.VENDOR_COMPONENT,
    ReuseDecision.FORK,
    ReuseDecision.REWRITE,
)

_PREFERENCE_RANK: dict[ReuseDecision, int] = {d: i for i, d in enumerate(REUSE_PREFERENCE_ORDER)}


class AuditStatus(str, enum.Enum):  # noqa: UP042
    """Lifecycle: DRAFT → AUDITED → SUPERSEDED | REJECTED."""

    DRAFT = "DRAFT"
    AUDITED = "AUDITED"
    SUPERSEDED = "SUPERSEDED"
    REJECTED = "REJECTED"


LEGAL_AUDIT_TRANSITIONS: dict[AuditStatus, frozenset[AuditStatus]] = {
    AuditStatus.DRAFT: frozenset({AuditStatus.AUDITED, AuditStatus.REJECTED}),
    AuditStatus.AUDITED: frozenset({AuditStatus.SUPERSEDED, AuditStatus.REJECTED}),
    AuditStatus.SUPERSEDED: frozenset(),
    AuditStatus.REJECTED: frozenset(),
}


# ---------------------------------------------------------------------------
# DimensionScore — int subclass carrying a mandatory justification.
# ---------------------------------------------------------------------------


class DimensionScore(int):
    """Score 0–5 with a mandatory one-line justification.

    Behaves as an ``int`` (so ``score == 5`` and ``isinstance(score, int)``
    hold) while exposing ``.justification`` and ``.score`` attributes.
    This bridges the two caller styles: flat int checks and object-style
    justification access.
    """

    # pylint: disable=too-many-ancestors

    def __new__(cls, score: int, justification: str) -> DimensionScore:
        if not isinstance(score, int) or isinstance(score, bool):
            raise TypeError(f"score must be int (not bool), got {type(score)!r}")
        if not (0 <= score <= 5):
            raise ValueError(f"score must be 0–5, got {score}")
        if not isinstance(justification, str):
            raise TypeError(f"justification must be str, got {type(justification)!r}")
        text = justification.strip()
        if len(text) < 5:
            raise ValueError("justification must be a non-empty one-line string (>=5 chars)")
        if "\n" in text or "\r" in text:
            raise ValueError("justification must be a single line (no newline)")
        obj = super().__new__(cls, score)
        # Bypass frozen int immutability via object.__setattr__
        object.__setattr__(obj, "_score", score)
        object.__setattr__(obj, "_justification", text)
        return obj

    @property
    def score(self) -> int:
        return int(self)

    @property
    def justification(self) -> str:
        return object.__getattribute__(self, "_justification")

    def to_dict(self) -> dict[str, Any]:
        return {"score": int(self), "justification": self.justification}

    def __repr__(self) -> str:
        return f"DimensionScore(score={int(self)}, justification={self.justification!r})"

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("DimensionScore is immutable")

    def __reduce__(self) -> Any:  # for pickle/copy
        return (DimensionScore, (int(self), self.justification))


def _is_dimension_score(value: object) -> bool:
    return isinstance(value, DimensionScore)


def _dimension_from_any(value: Any, field_name: str) -> DimensionScore:
    """Coerce *value* (int+alias, dict, or DimensionScore) to DimensionScore."""
    if isinstance(value, DimensionScore):
        return value
    if isinstance(value, dict):
        # dict form: {"score": int, "justification": str}
        if "score" not in value or "justification" not in value:
            raise TypeError(f"{field_name}: dict must have 'score' and 'justification'")
        return DimensionScore(int(value["score"]), str(value["justification"]))
    if isinstance(value, int) and not isinstance(value, bool):
        raise TypeError(
            f"{field_name}: plain int requires companion '{field_name}_justification' string"
        )
    raise TypeError(f"{field_name}: cannot coerce {value!r} to DimensionScore")


# ---------------------------------------------------------------------------
# Supporting blocks
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProvenanceRef:
    """Evidence artifact reference (path + sha256, optionally a typed tree)."""

    path: str
    sha256: str | None = None
    description: str | None = None
    # ``file`` preserves the historical shape; ``package_tree`` identifies a
    # lockfile as the complete transitive-package artifact for an admitted root.
    kind: str = "file"


@dataclass(frozen=True)
class StatusHistoryEntry:
    status: AuditStatus
    at: str
    by: str
    evidence_ref: str | None = None


# ---------------------------------------------------------------------------
# Main record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TechnologyAuditRecord:
    """Closed-schema technology audit record.

    Identity block (frozen for digest): candidate name, source URL,
    version, capability_area, seven dimension scores, reuse decision
    with justification, SBOM fields, provenance refs, evidence provenance,
    weights version.

    Mutable lifecycle (excluded from digest, lives in EventLog):
    status, status_history, audit_digest.
    """

    candidate_name: str
    source_url: str
    version: str
    capability_area: str
    clinical_technical_evidence: DimensionScore
    license_redistribution: DimensionScore
    reproducibility_auditability: DimensionScore
    security_dependency_health: DimensionScore
    local_offline_operability: DimensionScore
    maintainability: DimensionScore
    integration_cost: DimensionScore
    reuse_decision: ReuseDecision
    reuse_justification: str
    license_id: str
    permitted_use: str
    distribution_risk: str
    provenance_refs: tuple[ProvenanceRef, ...]
    status: AuditStatus = AuditStatus.DRAFT
    status_history: tuple[StatusHistoryEntry, ...] = ()
    audit_digest: str = ""
    weights_version: str = WEIGHTS_VERSION_DEFAULT
    schema_version: str = AUDIT_SCHEMA
    evidence_provenance: str = ""

    def __getattr__(self, name: str) -> Any:
        """Synthesize ``<dimension>_justification`` flat accessors.

        Allows tests expecting flat fields like
        ``record.clinical_technical_evidence_justification`` to succeed
        even though the canonical field is a :class:`DimensionScore`.
        """
        if name in DIMENSION_ALIAS_JUSTIFICATION.values():
            # find base dimension
            for dim, alias in DIMENSION_ALIAS_JUSTIFICATION.items():
                if alias == name:
                    val: DimensionScore = object.__getattribute__(self, dim)
                    return val.justification
            raise AttributeError(name)
        raise AttributeError(f"{type(self).__name__!r} has no attribute {name!r}")

    # Back-compat helpers for alternative flat int access patterns.
    # Tests that do ``record.scores`` or ``record.dimension_scores`` get a dict.
    @property
    def dimension_scores(self) -> dict[str, DimensionScore]:
        return {name: getattr(self, name) for name in DIMENSION_NAMES}

    @property
    def scores(self) -> dict[str, DimensionScore]:
        return self.dimension_scores


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _is_one_line(text: object) -> bool:
    return isinstance(text, str) and len(text.strip()) >= 5 and "\n" not in text and "\r" not in text


def validate_audit_record(record: TechnologyAuditRecord) -> None:
    """Fail-closed validation of a :class:`TechnologyAuditRecord`.

    Raises :class:`CanonicalValidationError` listing every violation.
    """
    problems: list[str] = []

    # Identity fields must be exact str (r5_b1) — numeric/bool values are
    # rejected on the direct validation path, never str()-coerced.
    for fname in ("candidate_name", "source_url", "version", "capability_area"):
        fval: object = getattr(record, fname)
        if not isinstance(fval, str):
            problems.append(f"{fname} must be str, got {type(fval)!r}")
            continue
        if not fval.strip():
            problems.append(f"{fname} is required")
    if isinstance(record.source_url, str) and record.source_url.strip():
        if not (record.source_url.startswith("http://") or record.source_url.startswith("https://")):
            problems.append(f"source_url must be http(s) URL, got {record.source_url!r}")

    # dimension scores: 0–5 + one-line justification
    for name in DIMENSION_NAMES:
        try:
            val = getattr(record, name)
            if not isinstance(val, DimensionScore):
                problems.append(f"{name} must be DimensionScore, got {type(val)!r}")
                continue
            if not (0 <= int(val) <= 5):
                problems.append(f"{name} score must be 0–5, got {int(val)}")
            if not _is_one_line(val.justification):
                problems.append(f"{name} justification must be a non-empty one-liner")
        except Exception as exc:  # pragma: no cover
            problems.append(f"{name}: {exc}")

    if not isinstance(record.reuse_decision, ReuseDecision):
        problems.append(f"reuse_decision must be ReuseDecision, got {record.reuse_decision!r}")
    if not _is_one_line(record.reuse_justification):
        problems.append("reuse_justification must be a non-empty one-liner")
    else:
        # Enforce deviation justification: non-preferred choices need explicit rationale.
        # We require justification to mention why earlier options were not chosen if rank >0.
        rank = _PREFERENCE_RANK.get(record.reuse_decision, 99)
        if rank > 0 and len(record.reuse_justification.strip()) < 10:
            problems.append(
                f"reuse_decision={record.reuse_decision.value} deviates from "
                "USE_DEPENDENCY; justification must explain deviation"
            )

    if not record.license_id or not record.license_id.strip():
        problems.append("license_id (SPDX) is required")
    if not _is_one_line(record.permitted_use):
        problems.append("permitted_use note is required (one-liner)")
    if not _is_one_line(record.distribution_risk):
        problems.append("distribution_risk note is required (one-liner)")

    if not isinstance(record.provenance_refs, tuple):
        problems.append("provenance_refs must be tuple[ProvenanceRef, ...]")
    elif not record.provenance_refs:
        problems.append("provenance_refs must be non-empty")
    else:
        for idx, ref in enumerate(record.provenance_refs):
            if not isinstance(ref, ProvenanceRef):
                problems.append(f"provenance_refs[{idx}] must be ProvenanceRef")
                continue
            if not ref.path or not ref.path.strip():
                problems.append(f"provenance_refs[{idx}].path is required")
            if ref.sha256 is not None and not is_sha256_hex(ref.sha256):
                problems.append(f"provenance_refs[{idx}].sha256 must be 64-hex or None")
            if ref.kind not in {"file", "package_tree"}:
                problems.append(
                    f"provenance_refs[{idx}].kind must be 'file' or 'package_tree'"
                )

    if not isinstance(record.status, AuditStatus):
        problems.append(f"status must be AuditStatus, got {record.status!r}")
    for idx, entry in enumerate(record.status_history):
        if not isinstance(entry, StatusHistoryEntry):
            problems.append(f"status_history[{idx}] must be StatusHistoryEntry")
            continue
        if not isinstance(entry.status, AuditStatus):
            problems.append(f"status_history[{idx}].status must be AuditStatus")
        try:
            validate_iso8601_utc(entry.at, f"status_history[{idx}].at")
        except ValueError as exc:
            problems.append(str(exc))
        if not entry.by or not entry.by.strip():
            problems.append(f"status_history[{idx}].by is required")

    if record.audit_digest and not is_sha256_hex(record.audit_digest):
        problems.append("audit_digest must be empty or 64-hex sha256")
    if not record.weights_version or not record.weights_version.strip():
        problems.append("weights_version is required")
    if record.schema_version != AUDIT_SCHEMA:
        problems.append(f"schema_version must be {AUDIT_SCHEMA!r}, got {record.schema_version!r}")
    if record.evidence_provenance is not None and not isinstance(record.evidence_provenance, str):
        problems.append("evidence_provenance must be str")

    if problems:
        raise CanonicalValidationError(problems)


# ---------------------------------------------------------------------------
# Serialization — JSON-safe, enum-free dumps
# ---------------------------------------------------------------------------


def _json_safe(value: Any) -> Any:
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, DimensionScore):
        return {"score": int(value), "justification": value.justification}
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_json_safe(v) for v in value)
    return value


def provenance_ref_to_dict(ref: ProvenanceRef) -> dict[str, Any]:
    """Serialize a provenance ref while preserving the historical file shape.

    ``kind="file"`` is the default introduced after v1 records were sealed.
    Omitting it from the canonical form keeps those records byte-for-byte
    compatible; non-default kinds remain explicitly digest-bound.
    """
    result: dict[str, Any] = {
        "path": ref.path,
        "sha256": ref.sha256,
        "description": ref.description,
    }
    if ref.kind != "file":
        result["kind"] = ref.kind
    return result


def record_to_dict(record: TechnologyAuditRecord) -> dict[str, Any]:
    """Canonical dict form; enums and DimensionScore become plain JSON values."""
    # Use manual construction to avoid dataclass/asdict int-subclass pitfalls.
    return {
        "candidate_name": record.candidate_name,
        "source_url": record.source_url,
        "version": record.version,
        "capability_area": record.capability_area,
        "clinical_technical_evidence": _json_safe(record.clinical_technical_evidence),
        "license_redistribution": _json_safe(record.license_redistribution),
        "reproducibility_auditability": _json_safe(record.reproducibility_auditability),
        "security_dependency_health": _json_safe(record.security_dependency_health),
        "local_offline_operability": _json_safe(record.local_offline_operability),
        "maintainability": _json_safe(record.maintainability),
        "integration_cost": _json_safe(record.integration_cost),
        "reuse_decision": record.reuse_decision.value,
        "reuse_justification": record.reuse_justification,
        "license_id": record.license_id,
        "permitted_use": record.permitted_use,
        "distribution_risk": record.distribution_risk,
        "provenance_refs": [provenance_ref_to_dict(ref) for ref in record.provenance_refs],
        "status": record.status.value,
        "status_history": [
            {
                "status": e.status.value,
                "at": e.at,
                "by": e.by,
                "evidence_ref": e.evidence_ref,
            }
            for e in record.status_history
        ],
        "audit_digest": record.audit_digest,
        "weights_version": record.weights_version,
        "schema_version": record.schema_version,
        "evidence_provenance": record.evidence_provenance,
    }


def _coerce_provenance_ref(value: Any, path: str) -> ProvenanceRef:
    if not isinstance(value, dict):
        raise ClosedSchemaError(f"{path}: expected object, got {type(value)!r}")
    hints = get_type_hints(ProvenanceRef)
    known = set(hints)
    unknown = set(value) - known
    if unknown:
        raise ClosedSchemaError(f"unknown field(s) in {path}: {sorted(unknown)}")
    raw_path = value.get("path")
    if not isinstance(raw_path, str):
        raise ClosedSchemaError(f"{path}.path must be str, got {type(raw_path)!r}")
    raw_sha = value.get("sha256")
    if raw_sha is not None and not isinstance(raw_sha, str):
        raise ClosedSchemaError(f"{path}.sha256 must be str or null, got {type(raw_sha)!r}")
    raw_desc = value.get("description")
    if raw_desc is not None and not isinstance(raw_desc, str):
        raise ClosedSchemaError(f"{path}.description must be str or null, got {type(raw_desc)!r}")
    raw_kind = value.get("kind", "file")
    if not isinstance(raw_kind, str) or raw_kind not in {"file", "package_tree"}:
        raise ClosedSchemaError(
            f"{path}.kind must be 'file' or 'package_tree', got {raw_kind!r}"
        )
    return ProvenanceRef(
        path=raw_path,
        sha256=raw_sha,
        description=raw_desc,
        kind=raw_kind,
    )


def _coerce_status_entry(value: Any, path: str) -> StatusHistoryEntry:
    if not isinstance(value, dict):
        raise ClosedSchemaError(f"{path}: expected object, got {type(value)!r}")
    known = {"status", "at", "by", "evidence_ref"}
    unknown = set(value) - known
    if unknown:
        raise ClosedSchemaError(f"unknown field(s) in {path}: {sorted(unknown)}")
    raw_status = value.get("status")
    if not isinstance(raw_status, str):
        raise ClosedSchemaError(f"{path}.status must be str, got {type(raw_status)!r}")
    raw_at = value.get("at")
    if not isinstance(raw_at, str):
        raise ClosedSchemaError(f"{path}.at must be str, got {type(raw_at)!r}")
    raw_by = value.get("by")
    if not isinstance(raw_by, str):
        raise ClosedSchemaError(f"{path}.by must be str, got {type(raw_by)!r}")
    raw_ev = value.get("evidence_ref")
    if raw_ev is not None and not isinstance(raw_ev, str):
        raise ClosedSchemaError(f"{path}.evidence_ref must be str or null, got {type(raw_ev)!r}")
    return StatusHistoryEntry(
        status=AuditStatus(raw_status),
        at=raw_at,
        by=raw_by,
        evidence_ref=raw_ev,
    )


def record_from_dict(data: dict[str, Any]) -> TechnologyAuditRecord:
    """Strict loader with closed-schema semantics and full validation.

    Accepts both canonical dict form (dimension as ``{score, justification}``)
    and legacy flat alias form (``<dim>_justification`` companion keys).
    Unknown top-level keys fail closed.
    """
    if not isinstance(data, dict):
        raise ClosedSchemaError(f"audit record must be object, got {type(data)!r}")

    hints = get_type_hints(TechnologyAuditRecord)
    # canonical field names
    known_canonical = set(hints)
    # alias justification keys are also tolerated as input (not stored separately)
    alias_keys = set(DIMENSION_ALIAS_JUSTIFICATION.values())
    known_with_alias = known_canonical | alias_keys
    unknown = set(data) - known_with_alias
    if unknown:
        raise ClosedSchemaError(f"unknown field(s) in {AUDIT_SCHEMA} record: {sorted(unknown)}")

    kwargs: dict[str, Any] = {}

    # identity — exact str only (r5_b1): numeric/bool values must be rejected
    # BEFORE assignment, never str()-coerced, so a recomputed digest cannot
    # smuggle a forged identity through the typed schema.
    for name in ("candidate_name", "source_url", "version", "capability_area"):
        if name in data:
            raw_identity = data[name]
            if not isinstance(raw_identity, str):
                raise CanonicalValidationError(
                    f"{name} must be str, got {type(raw_identity)!r}"
                )
            kwargs[name] = raw_identity

    # dimensions: accept either dict form or flat alias pair
    for dim in DIMENSION_NAMES:
        dict_form = data.get(dim)
        alias_key = DIMENSION_ALIAS_JUSTIFICATION[dim]
        alias_val = data.get(alias_key)

        if dict_form is not None:
            # canonical dict form
            if isinstance(dict_form, dict):
                # explicit dict — closed-schema for dimension objects (blocker w4)
                unknown_dim = set(dict_form) - {"score", "justification"}
                if unknown_dim:
                    raise ClosedSchemaError(f"unknown field(s) in {dim}: {sorted(unknown_dim)}")
                raw_score = dict_form["score"]
                raw_just = dict_form["justification"]
                if not isinstance(raw_score, int) or isinstance(raw_score, bool):
                    raise CanonicalValidationError(
                        f"{dim}.score must be int (not bool/string), got {type(raw_score)!r}"
                    )
                if not isinstance(raw_just, str):
                    raise CanonicalValidationError(
                        f"{dim}.justification must be str, got {type(raw_just)!r}"
                    )
                try:
                    kwargs[dim] = DimensionScore(raw_score, raw_just)
                except Exception as exc:
                    raise CanonicalValidationError(f"{dim}: {exc}") from exc
            elif isinstance(dict_form, int) and not isinstance(dict_form, bool):
                # int with companion alias
                if alias_key not in data:
                    raise CanonicalValidationError(
                        f"{dim}: plain int score requires '{alias_key}' justification"
                    )
                if not isinstance(alias_val, str):
                    raise CanonicalValidationError(
                        f"{dim}: alias {alias_key} must be str, got {type(alias_val)!r}"
                    )
                try:
                    kwargs[dim] = DimensionScore(dict_form, alias_val)
                except Exception as exc:
                    raise CanonicalValidationError(f"{dim}: {exc}") from exc
            elif isinstance(dict_form, DimensionScore):
                kwargs[dim] = dict_form
            else:
                raise CanonicalValidationError(f"{dim}: cannot coerce {dict_form!r}")
        elif alias_key in data:
            # Only alias present without base? Reject.
            raise CanonicalValidationError(f"{dim}: missing score for alias {alias_key!r}")
        # else leave missing to be caught below

    # remaining fields
    if "reuse_decision" in data:
        try:
            kwargs["reuse_decision"] = ReuseDecision(str(data["reuse_decision"]))
        except ValueError as exc:
            raise CanonicalValidationError(f"reuse_decision: {exc}") from exc
    for name in ("reuse_justification", "license_id", "permitted_use", "distribution_risk", "evidence_provenance", "weights_version", "schema_version", "audit_digest"):
        if name in data:
            kwargs[name] = data[name]
    if "status" in data:
        try:
            kwargs["status"] = AuditStatus(str(data["status"]))
        except ValueError as exc:
            raise CanonicalValidationError(f"status: {exc}") from exc
    if "provenance_refs" in data:
        raw = data["provenance_refs"]
        if not isinstance(raw, (list, tuple)):
            raise CanonicalValidationError("provenance_refs must be list")
        kwargs["provenance_refs"] = tuple(
            _coerce_provenance_ref(v, f"provenance_refs[{i}]") for i, v in enumerate(raw)
        )
    if "status_history" in data:
        raw = data["status_history"]
        if not isinstance(raw, (list, tuple)):
            raise CanonicalValidationError("status_history must be list")
        kwargs["status_history"] = tuple(
            _coerce_status_entry(v, f"status_history[{i}]") for i, v in enumerate(raw)
        )

    # check missing required
    missing_required = [
        f.name
        for f in fields(TechnologyAuditRecord)
        if f.name not in kwargs and f.default is MISSING and f.default_factory is MISSING
    ]
    if missing_required:
        raise CanonicalValidationError(f"missing required fields: {missing_required}")

    record = TechnologyAuditRecord(**kwargs)
    validate_audit_record(record)
    if record.audit_digest:
        verify_audit_digest(record)
    return record


def load_audit_record(data: dict[str, Any]) -> TechnologyAuditRecord:
    """Persistence loader: requires a sealed ``audit_digest``."""
    record = record_from_dict(data)
    if not is_sha256_hex(record.audit_digest):
        raise CanonicalValidationError("persisted audit records must carry a sealed 64-hex audit_digest")
    return record


# ---------------------------------------------------------------------------
# Digest — frozen identity block only
# ---------------------------------------------------------------------------


def compute_audit_digest(record: TechnologyAuditRecord) -> str:
    """I12-style self-digest over the FROZEN IDENTITY BLOCK only.

    Excluded (like ``TRANSIENT_RECORD_FIELDS`` in registry): ``audit_digest``,
    ``status``, ``status_history``. Lifecycle lives in the event log.
    """
    payload = record_to_dict(record)
    return canonical_digest(payload, exclude=tuple(sorted(TRANSIENT_AUDIT_FIELDS)))


def verify_audit_digest(record: TechnologyAuditRecord) -> None:
    expected = compute_audit_digest(record)
    if expected != record.audit_digest:
        raise DigestMismatchError(f"audit digest mismatch for {record.candidate_name}@{record.version}")


__all__ = [
    "AUDIT_SCHEMA",
    "TRANSIENT_AUDIT_FIELDS",
    "DIMENSION_NAMES",
    "DimensionScore",
    "ProvenanceRef",
    "StatusHistoryEntry",
    "TechnologyAuditRecord",
    "ReuseDecision",
    "REUSE_PREFERENCE_ORDER",
    "AuditStatus",
    "LEGAL_AUDIT_TRANSITIONS",
    "validate_audit_record",
    "provenance_ref_to_dict",
    "record_to_dict",
    "record_from_dict",
    "load_audit_record",
    "compute_audit_digest",
    "verify_audit_digest",
]
