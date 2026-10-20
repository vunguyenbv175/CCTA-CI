"""§44 provenance records and the invalidation DAG (Part B §3.5, H-007/H-010).

A ``ProvenanceRecord`` is valid only when ALL NINE Master Plan §44 items have
an explicit value OR an explicit :class:`NotApplicable` wrapper with a
reason. Anything missing fails closed at creation - nothing is defaulted.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from core.digest import (
    DIGEST_EXCLUSIONS,
    canonical_digest,
    is_sha256_hex,
    require_sha256_hex,
)
from core.enums import ActorType, ProvenanceEventType
from core.errors import ProvenanceIncompleteError
from core.events import utc_now_iso, validate_iso8601_utc
from core.ids import new_ulid

PROVENANCE_SCHEMA = "ccta.canonical.provenance_record.v1"

_HEX40_DIGITS = frozenset("0123456789abcdef")




# ---------------------------------------------------------------------------
# Explicit NotApplicable wrappers (Luna R2 fix #5)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NotApplicable[T]:
    """Explicit "not applicable" marker carrying the reason.

    Optional §44 fields accept either a real value OR this wrapper. Bare
    ``None``, empty strings or empty collections are rejected - an omitted
    item must never be silently inferred (fail-closed).
    """

    value: T | None
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.reason, str) or len(self.reason.strip()) < 4:
            raise ProvenanceIncompleteError(
                f"NotApplicable requires a meaningful reason, got {self.reason!r}"
            )


@dataclass(frozen=True)
class BlobRef:
    """Content-addressed parameter/payload reference."""

    blob_ref: str
    sha256: str


@dataclass(frozen=True)
class HumanEdit:
    """Human-edit record (§44 "human edits")."""

    editor: str
    edit_summary: str
    at: str


@dataclass(frozen=True)
class InputRef:
    """One consumed input bound at use time (source images, artifacts...)."""

    ref: str
    kind: str  # series | instance | volume | artifact | dataset
    sha256_at_use: str


@dataclass(frozen=True)
class Environment:
    """Execution environment of the recorded event (no PHI allowed)."""

    software_version: str
    code_commit_sha: str  # 40-hex git SHA
    runtime: str | None = None
    deterministic_seed: str | None = None


@dataclass(frozen=True)
class Actor:
    """Who performed the recorded action."""

    actor_type: ActorType
    actor_id: str


@dataclass(frozen=True)
class ProvenanceRecord:
    """Append-only provenance record (Part B §3.5).

    Create exclusively through :meth:`create`; construction validates the
    nine §44 items and seals ``record_sha256``.
    """

    provenance_id: str
    subject_ref: str
    event_type: ProvenanceEventType
    actor: Actor
    inputs: tuple[InputRef, ...] | NotApplicable[tuple[InputRef, ...]]
    preprocessing: str | NotApplicable[str]
    parameters_ref: BlobRef | NotApplicable[BlobRef]
    human_edit: HumanEdit | NotApplicable[HumanEdit]
    upstream_refs: tuple[str, ...] | NotApplicable[tuple[str, ...]]
    environment: Environment
    model_record_ref: str | None
    schema_version: str = PROVENANCE_SCHEMA
    created_by: str = ""
    at: str = ""
    record_sha256: str = ""

    @property
    def created_at(self) -> str:
        return self.at


# ---------------------------------------------------------------------------
# Nine §44 items - fail-closed creation
# ---------------------------------------------------------------------------


def create_provenance(
    *,
    subject_ref: str,
    event_type: ProvenanceEventType,
    actor: Actor,
    inputs: tuple[InputRef, ...] | NotApplicable[tuple[InputRef, ...]],
    preprocessing: str | NotApplicable[str],
    parameters_ref: BlobRef | NotApplicable[BlobRef],
    human_edit: HumanEdit | NotApplicable[HumanEdit],
    upstream_refs: tuple[str, ...] | NotApplicable[tuple[str, ...]],
    environment: Environment,
    model_record_ref: str | None = None,
    created_by: str = "",
    at: str | None = None,
    provenance_id: str | None = None,
) -> ProvenanceRecord:
    """Validate the nine §44 items and seal the record digest.

    Raises :class:`ProvenanceIncompleteError` listing EVERY violated item -
    never just the first - so callers can repair records in one pass.
    """
    violations: list[str] = []

    stamp = at or utc_now_iso()
    try:
        validate_iso8601_utc(stamp, "at")
    except ValueError as exc:
        violations.append(str(exc))

    if not subject_ref:
        violations.append("subject_ref is required")
    if not created_by:
        violations.append("created_by is required")
    if not isinstance(event_type, ProvenanceEventType):
        violations.append("event_type must be a ProvenanceEventType")

    # -- item 1: source image refs -----------------------------------------
    _check_inputs(inputs, violations)

    # -- item 2: preprocessing spec ----------------------------------------
    if preprocessing is None or (
        not isinstance(preprocessing, NotApplicable) and not str(preprocessing).strip()
    ):
        violations.append("§44 item 'preprocessing' missing (value or NotApplicable)")

    # -- items 3+4+5: model XOR algorithm branch ---------------------------
    is_model_actor = actor.actor_type == ActorType.MODEL
    if is_model_actor:
        if not model_record_ref:
            violations.append(
                "actor_type=MODEL requires model_record_ref (model id+version)"
            )
    else:
        if not actor.actor_id.strip():
            violations.append("algorithm branch: actor.actor_id (algorithm name) required")
        if model_record_ref:
            violations.append("algorithm branch set but model_record_ref also populated")

    # -- item 5b: algorithm version via code commit ------------------------
    env_sha = environment.code_commit_sha
    if not (isinstance(env_sha, str) and len(env_sha) == 40 and set(env_sha) <= _HEX40_DIGITS):
        violations.append("environment.code_commit_sha must be a 40-hex git SHA")
    if not environment.software_version:
        violations.append("environment.software_version is required")

    # -- item 6: parameters digest ------------------------------------------
    if parameters_ref is None:
        violations.append("§44 item 'parameters' missing (value or NotApplicable)")
    elif not isinstance(parameters_ref, NotApplicable):
        try:
            require_sha256_hex(parameters_ref.sha256, "parameters_ref.sha256")
        except ValueError as exc:
            violations.append(str(exc))
        if not parameters_ref.blob_ref:
            violations.append("parameters_ref.blob_ref is required")

    # -- item 7: human edits -------------------------------------------------
    if human_edit is None:
        violations.append("§44 item 'human edits' missing (value or NotApplicable)")
    elif not isinstance(human_edit, NotApplicable):
        if not human_edit.editor or not human_edit.edit_summary:
            violations.append("human_edit needs editor and edit_summary")
        else:
            try:
                validate_iso8601_utc(human_edit.at, "human_edit.at")
            except ValueError as exc:
                violations.append(str(exc))

    # -- item 8: upstream dependency DAG ------------------------------------
    if isinstance(upstream_refs, NotApplicable):
        pass
    elif not upstream_refs:
        violations.append(
            "§44 item 'upstream dependencies' missing "
            "(upstream_refs non-empty or NotApplicable)"
        )
    elif any(not ref for ref in upstream_refs):
        violations.append("upstream_refs contains empty references")

    if violations:
        raise ProvenanceIncompleteError(violations)

    seed = ProvenanceRecord(
        provenance_id=provenance_id or new_ulid(),
        subject_ref=subject_ref,
        event_type=event_type,
        actor=actor,
        inputs=inputs,
        preprocessing=preprocessing,
        parameters_ref=parameters_ref,
        human_edit=human_edit,
        upstream_refs=upstream_refs,
        environment=environment,
        model_record_ref=model_record_ref,
        created_by=created_by,
        at=stamp,
    )
    digest = canonical_digest(seed, tuple(DIGEST_EXCLUSIONS[PROVENANCE_SCHEMA]))
    return ProvenanceRecord(
        provenance_id=seed.provenance_id,
        subject_ref=subject_ref,
        event_type=event_type,
        actor=actor,
        inputs=inputs,
        preprocessing=preprocessing,
        parameters_ref=parameters_ref,
        human_edit=human_edit,
        upstream_refs=upstream_refs,
        environment=environment,
        model_record_ref=model_record_ref,
        created_by=created_by,
        at=stamp,
        record_sha256=digest,
    )


def _check_inputs(
    inputs: tuple[InputRef, ...] | NotApplicable[tuple[InputRef, ...]] | None,
    violations: list[str],
) -> None:
    """Item 1 + half of item 8: input refs bound with digests."""
    if inputs is None:
        violations.append("§44 item 'source image' missing (inputs or NotApplicable)")
        return
    if isinstance(inputs, NotApplicable):
        return
    if not inputs:
        violations.append("§44 item 'source image' missing (inputs or NotApplicable)")
        return
    #: ``seed`` is the typed human seed of a governed derivation
    #: (CORONARY_CENTERLINE_001 §5: ``InputRef`` = volume + seed).
    known_kinds = {"series", "instance", "volume", "artifact", "dataset", "seed"}
    for idx, ref in enumerate(inputs):
        if not ref.ref:
            violations.append(f"inputs[{idx}].ref is empty")
        if ref.kind not in known_kinds:
            violations.append(f"inputs[{idx}].kind unknown: {ref.kind!r}")
        if not is_sha256_hex(ref.sha256_at_use):
            violations.append(f"inputs[{idx}].sha256_at_use must be a 64-hex digest")


# ---------------------------------------------------------------------------
# Digest verification + chain audit
# ---------------------------------------------------------------------------


def verify_self_digest(record: ProvenanceRecord) -> bool:
    """Recompute the sealed digest; False on any tamper of included fields."""
    if not is_sha256_hex(record.record_sha256):
        return False
    expected = canonical_digest(record, tuple(DIGEST_EXCLUSIONS[PROVENANCE_SCHEMA]))
    return expected == record.record_sha256


def verify_chain(
    records: Iterable[ProvenanceRecord],
) -> tuple[bool, list[str]]:
    """Verify every record's digest and upstream resolvability.

    Returns ``(ok, problems)``; problems list record ids failing checks.
    """
    by_id = {rec.provenance_id: rec for rec in records}
    problems: list[str] = []
    for rid, rec in by_id.items():
        if not verify_self_digest(rec):
            problems.append(f"{rid}: self-digest mismatch (tampered or corrupted)")
        ups = (
            ()
            if isinstance(rec.upstream_refs, NotApplicable)
            else tuple(rec.upstream_refs)
        )
        for up in ups:
            if up not in by_id:
                problems.append(f"{rid}: unresolvable upstream ref {up!r}")
    return (not problems, problems)


# ---------------------------------------------------------------------------
# Invalidation DAG (H-010, Luna fix #4, I7)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GeometryBinding:
    """Dependent artifact pinned to a geometry id + version."""

    dependent_id: str
    geometry_id: str
    target_geometry_version: int


class InvalidationDAG:
    """Tracks geometry-version dependencies and propagates STALE downstream.

    Any modification of an upstream geometry (new ``geometry_version``) marks
    all transitive dependents STALE via append-only STATUS events; invalidated
    data is never deleted (I7).
    """

    def __init__(self) -> None:
        self._geometry_versions: dict[str, int] = {}
        self._bindings: dict[str, GeometryBinding] = {}
        self._invalidated: dict[str, bool] = {}

    def register_geometry(self, geometry_id: str, geometry_version: int) -> None:
        if geometry_version < 1:
            raise ValueError("geometry_version starts at 1")
        current = self._geometry_versions.get(geometry_id, 0)
        if geometry_version <= current:
            raise ValueError(
                f"geometry {geometry_id} version must increase beyond {current}"
            )
        self._geometry_versions[geometry_id] = geometry_version

    def mark_invalidated(self, geometry_id: str) -> None:
        self._invalidated[geometry_id] = True

    def geometry_version(self, geometry_id: str) -> int:
        """Current registered version of *geometry_id* (0 when unregistered).

        Additive read-only accessor: callers that need the next deterministic
        geometry version (e.g. a governed recompute that must advance the
        version exactly once) read it here instead of inspecting private
        state.  Reading never mutates the DAG.
        """

        return int(self._geometry_versions.get(geometry_id, 0))

    def bind_dependent(
        self, dependent_id: str, geometry_id: str, target_geometry_version: int
    ) -> None:
        if dependent_id in self._bindings:
            raise ValueError(f"dependent {dependent_id} already bound")
        if geometry_id not in self._geometry_versions:
            raise KeyError(f"unknown geometry {geometry_id}")
        self._bindings[dependent_id] = GeometryBinding(
            dependent_id=dependent_id,
            geometry_id=geometry_id,
            target_geometry_version=target_geometry_version,
        )

    def stale_dependents(self, geometry_id: str) -> tuple[str, ...]:
        """Transitive closure of dependents made stale by *geometry_id*.

        A dependent is stale when its bound geometry was invalidated or has
        advanced beyond the pinned version; staleness then propagates through
        chains of geometry->dependent bindings.
        """
        result: set[str] = set()
        for binding in self._bindings.values():
            if binding.geometry_id == geometry_id and self._directly_stale(binding):
                result.add(binding.dependent_id)
        changed = True
        while changed:
            changed = False
            for binding in self._bindings.values():
                if (
                    binding.geometry_id in result
                    and binding.dependent_id not in result
                    and binding.dependent_id != geometry_id
                ):
                    result.add(binding.dependent_id)
                    changed = True
        return tuple(sorted(result))

    def _directly_stale(self, binding: GeometryBinding) -> bool:
        return self._invalidated.get(binding.geometry_id, False) or (
            self._geometry_versions.get(binding.geometry_id, 0)
            > binding.target_geometry_version
        )

    def is_stale(self, dependent_id: str) -> bool:
        """True iff *dependent_id* is directly or transitively stale."""
        binding = self._bindings.get(dependent_id)
        if binding is None:
            return False
        if self._directly_stale(binding):
            return True
        return dependent_id in self.stale_dependents(binding.geometry_id)


__all__ = [
    "Actor",
    "BlobRef",
    "Environment",
    "GeometryBinding",
    "HumanEdit",
    "InputRef",
    "InvalidationDAG",
    "NotApplicable",
    "PROVENANCE_SCHEMA",
    "ProvenanceRecord",
    "create_provenance",
    "verify_chain",
    "verify_self_digest",
]
