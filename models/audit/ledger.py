# ruff: noqa: E501
"""Audit ledger — append-only, capability-projected, JSON round-trippable.

Mirrors ``models/registry`` patterns:
* frozen records are sealed with :func:`compute_audit_digest`;
* mutable status lives only in the shared :class:`core.events.EventLog`;
* JSON dumps are enum-free and ``json.dumps``-serializable.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from dataclasses import fields as dc_fields
from types import MappingProxyType

from core.errors import (
    CanonicalValidationError,
    ClosedSchemaError,
    DigestMismatchError,
    StateTransitionError,
)
from core.events import EventLog, Projection, StateEvent, event_to_dict, utc_now_iso

from .record import (
    LEGAL_AUDIT_TRANSITIONS,
    AuditStatus,
    StatusHistoryEntry,
    TechnologyAuditRecord,
    compute_audit_digest,
    load_audit_record,
    record_to_dict,
    validate_audit_record,
)
from .scoring import rank_audits

LEDGER_SCHEMA = "ccta.audit.ledger.v1"


def _audit_ref(candidate_name: str, version: str) -> str:
    """Canonical ``candidate@version`` reference; mirrors registry ``model_ref``."""
    return f"{candidate_name}@{version}"


def _norm_key(name: str) -> str:
    import re

    # Normalize to alphanumerics only, lowercased — so "3D Slicer",
    # "3d-slicer", "3d_slicer" and "nnU-Net"/"nnunet" all map together.
    return re.sub(r"[^a-z0-9]", "", name.strip().lower())


@dataclass(frozen=True)
class LedgerEntry:
    """View helper for ranked listing."""

    record: TechnologyAuditRecord
    score: float


def _freeze_json_value(value: object) -> object:
    """Recursively convert *value* into immutable equivalents (r7 b3-b).

    Mappings become ``MappingProxyType`` over fresh dicts, sequences become
    tuples, sets become frozensets — at EVERY nesting level — so no mutable
    container inside a surfaced event payload is ever shared with callers.
    JSON-derived scalars (str/int/float/bool/None) are immutable as-is.
    """
    if isinstance(value, Mapping):
        return MappingProxyType({k: _freeze_json_value(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json_value(v) for v in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_json_value(v) for v in value)
    return value


def _frozen_event_copy(evt: StateEvent) -> StateEvent:
    """Defensively copied, deeply immutable clone of *evt* (r6/r7 b3).

    Every scalar field of :class:`StateEvent` is already frozen; payload
    mappings can be mutated in place at arbitrary nesting depth when backed
    by plain dicts. Public surfacing therefore rebuilds the event with its
    payload deep-frozen via :func:`_freeze_json_value`, so external callers
    can neither mutate shared state at any depth nor observe later internal
    changes, and ``json.dumps`` stays exact via
    :func:`core.events.event_to_dict` (which converts mappings to dicts).
    """
    return StateEvent(
        entity=evt.entity,
        entity_id=evt.entity_id,
        event_type=evt.event_type,
        from_value=evt.from_value,
        to_value=evt.to_value,
        actor_role=evt.actor_role,
        at=evt.at,
        evidence_ref=evt.evidence_ref,
        payload=_freeze_json_value(dict(evt.payload)),  # type: ignore[arg-type]
    )


class EventLogView:
    """Immutable point-in-time snapshot facade over an :class:`EventLog`.

    Exposes history/projection reads (``events``, ``history_for``,
    ``project_state``) and deliberately does NOT expose ``seed`` /
    ``append_event``, so external callers cannot mutate the ledger's
    lifecycle state behind its back.

    Instances are built only by :func:`_event_log_view`, which copies the
    log's content into defensive immutable data at build time. The view
    holds NO reference to the mutable log through ANY reachable path — no
    attribute (public, private, or name-mangled), no method closure cell,
    no function defaults — so introspection of instance or class metadata
    cannot reach ``append_event``/``seed`` (r6 b3-a / r7 b3-a). Projection
    replay runs on a TRANSIENT throwaway log rebuilt from the snapshot
    copies; results are returned strictly by value. A view never changes
    after construction; later ledger mutations are observed only by
    re-acquiring ``ledger.event_log`` for a fresh snapshot.
    """

    __slots__ = ("_events", "_initials")

    #: Immutable event copies captured at snapshot time.
    _events: tuple[StateEvent, ...]
    #: Birth records (per ``(entity, entity_id)``), read-only copies.
    _initials: Mapping[tuple[str, str], Mapping[str, str]]

    def __init__(
        self,
        events: tuple[StateEvent, ...],
        initials: Mapping[tuple[str, str], Mapping[str, str]],
    ) -> None:
        self._events = events
        self._initials = initials

    @property
    def events(self) -> tuple[StateEvent, ...]:
        """All events known at snapshot time (immutable copies)."""
        return self._events

    def history_for(
        self, entity_id: str, *, entity: str | None = None
    ) -> tuple[StateEvent, ...]:
        """Ordered immutable events for *entity_id* at snapshot time."""

        def _matches(event: StateEvent) -> bool:
            if event.entity_id != entity_id:
                return False
            return entity is None or event.entity == entity

        return tuple(e for e in self._events if _matches(e))

    def project_state(
        self, entity_id: str, *, entity: str | None = None
    ) -> Projection:
        """Replayed read-only projection for *entity_id*.

        The replay runs on a transient EventLog assembled purely from the
        snapshot copies; that temporary object is method-local and only the
        value-result :class:`Projection` escapes (r7 b3-a).
        """
        replay = EventLog()
        for (ent, ent_id), initial_fields in self._initials.items():
            replay.seed(entity=ent, entity_id=ent_id, initial_fields=dict(initial_fields))
        for evt in self._events:
            replay.append_event(evt)
        return replay.project_state(entity_id, entity=entity)

    def __len__(self) -> int:
        return len(self._events)

    def __iter__(self) -> Iterator[StateEvent]:
        return iter(self._events)


def _event_log_view(log: EventLog) -> EventLogView:
    """Build an immutable snapshot view of *log*'s current content (r7 b3-a).

    *log* is consumed here as a transient argument only: its events and
    birth records are copied into deeply immutable data, after which the
    view shares nothing with it — no closure cells, no defaults, no
    attributes reference the live object.
    """
    initials: dict[tuple[str, str], Mapping[str, str]] = {}
    for (entity, entity_id), initial_fields in log._initial.items():  # noqa: SLF001
        initials[(entity, entity_id)] = MappingProxyType(dict(initial_fields))
    return EventLogView(
        events=tuple(_frozen_event_copy(e) for e in log.events),
        initials=MappingProxyType(initials),
    )


def _snapshot_event_log(src: EventLog) -> EventLog:
    """Copy *src* into a fresh, internally-owned :class:`EventLog` (r5_b3).

    The ledger must not retain a caller-owned mutable identity for its
    lifecycle state, so constructor log injection is snapshotted: birth
    records are re-seeded and events are replayed in order through the
    public append path (which re-validates optimistic concurrency).
    """
    dst = EventLog()
    # Birth records have no public accessor on EventLog; copy them verbatim
    # so projections inside the owned copy replay identically.
    for (entity, entity_id), initial_fields in src._initial.items():  # noqa: SLF001
        dst.seed(entity=entity, entity_id=entity_id, initial_fields=dict(initial_fields))
    for evt in src.events:
        # r6 b3: replay immutable copies so the ledger never shares a
        # caller-owned StateEvent (or its mutable payload dict) either.
        dst.append_event(_frozen_event_copy(evt))
    return dst


class AuditLedger:
    """Projection store + deterministic state machine over the event log.

    * ``add`` seals and seeds a DRAFT record;
    * ``transition`` appends a STATUS event and rebuilds the frozen record;
    * ``project_status`` replays the log (read-only projection);
    * ``ranked`` delegates to :func:`scoring.rank_audits` with deterministic
      tie-breaks (name then version).
    """

    def __init__(self, *, log: EventLog | None = None) -> None:
        # r5_b3: snapshot the injected log into an internally-owned EventLog so
        # the caller retains no mutable alias into this ledger's state.
        self._log: EventLog = _snapshot_event_log(log) if log is not None else EventLog()
        self._records: dict[str, TechnologyAuditRecord] = {}
        # normalized lookup for case-insensitive access
        self._norm_index: dict[tuple[str, str], str] = {}

    # -- read access -----------------------------------------------------------

    def __len__(self) -> int:
        return len(self._records)

    def __contains__(self, ref: object) -> bool:
        return isinstance(ref, str) and ref in self._records

    def __iter__(self) -> Iterator[TechnologyAuditRecord]:
        return iter(self._records.values())

    @property
    def event_log(self) -> EventLogView:
        """Point-in-time read-only snapshot of the event log (r5_b3/r6/r7 b3).

        Only history/projection reads are exposed — ``seed`` and
        ``append_event`` are unreachable, the raw log object is reachable
        through no attribute, closure cell, or function default on the view
        (snapshots are pure copied data), and every surfaced event is a
        deeply immutable defensive copy — so external callers can never
        desynchronize the projection from the stored records. A snapshot is
        built fresh on every access, so it always reflects the ledger's
        current state; previously obtained views stay frozen at their
        capture point and never change.
        """
        return _event_log_view(self._log)

    def all_records(self) -> list[TechnologyAuditRecord]:
        return list(self._records.values())

    def get(self, candidate_name: str, version: str) -> TechnologyAuditRecord:
        """Lookup by ``(candidate_name, version)`` — case-insensitive fallback."""
        ref = _audit_ref(candidate_name, version)
        if ref in self._records:
            return self._records[ref]
        # case-insensitive search
        norm = (_norm_key(candidate_name), version.strip())
        matched = self._norm_index.get(norm)
        if matched is not None and matched in self._records:
            return self._records[matched]
        # also try normalized version? keep exact version match
        # second fallback: scan
        for key, rec in self._records.items():
            if _norm_key(rec.candidate_name) == _norm_key(candidate_name) and rec.version == version:
                return rec
        raise KeyError(f"unknown audit {candidate_name!r}@{version!r}")

    # Back-compat alias
    def lookup(self, candidate_name: str, version: str) -> TechnologyAuditRecord:
        return self.get(candidate_name, version)

    def project_status(self, candidate_name: str, version: str) -> AuditStatus:
        """Replay the event log for ``candidate@version``; mirrors registry."""
        ref = _audit_ref(candidate_name, version)
        # try exact, then normalized fallback to find real ref
        if ref not in self._records:
            # resolve via get to handle case-insensitivity
            rec = self.get(candidate_name, version)
            ref = _audit_ref(rec.candidate_name, rec.version)
        projection = self._log.project_state(ref, entity="technology_audit")
        raw = projection.get("status", AuditStatus.DRAFT.value)
        return AuditStatus(str(raw))

    def by_capability(self, capability_area: str) -> list[TechnologyAuditRecord]:
        """Project audits for one capability area (unsorted)."""
        norm = capability_area.strip().lower()
        return [r for r in self._records.values() if r.capability_area.strip().lower() == norm]

    def ranked(
        self,
        capability_area: str | None = None,
        *,
        weights_version: str | None = None,
    ) -> list[TechnologyAuditRecord]:
        """Risk-adjusted ranking with deterministic tie-breaks.

        If *capability_area* is given, only that slice is ranked.
        """
        pool = self.by_capability(capability_area) if capability_area else self.all_records()
        return rank_audits(pool, weights_version=weights_version)

    def latest_status_projection(self, candidate_name: str, version: str) -> dict[str, str]:
        """Read-only projection dict for ``candidate@version``."""
        rec = self.get(candidate_name, version)
        ref = _audit_ref(rec.candidate_name, rec.version)
        proj = self._log.project_state(ref, entity="technology_audit")
        return proj.to_dict()

    # -- write path ------------------------------------------------------------

    def add(self, record: TechnologyAuditRecord) -> TechnologyAuditRecord:
        """Freeze, seal, and register a new DRAFT audit.

        Validates fully, computes ``audit_digest`` if empty, seeds the event
        log with initial ``status``. Duplicate ``(candidate, version)`` is
        rejected — corrections are new versions or supersession events.
        """
        validate_audit_record(record)
        ref = _audit_ref(record.candidate_name, record.version)
        if ref in self._records:
            raise CanonicalValidationError(f"{ref} already audited; corrections are new versions")
        # also check normalized duplicate to avoid case-violations
        norm = (_norm_key(record.candidate_name), record.version)
        if norm in self._norm_index:
            raise CanonicalValidationError(
                f"{ref} collides (case-insensitive) with {self._norm_index[norm]!r}"
            )

        # Enforce DRAFT-first (blocker w3) — new audits must start DRAFT with empty history
        if record.status != AuditStatus.DRAFT:
            raise CanonicalValidationError(f"new audits must start DRAFT, got {record.status.value}")
        if record.status_history:
            raise CanonicalValidationError("new audits must have empty status_history")

        # Seal if not already sealed; otherwise verify digest (blocker w2)
        sealed = record
        if not record.audit_digest:
            digest = compute_audit_digest(record)
            kwargs = {f.name: getattr(record, f.name) for f in dc_fields(TechnologyAuditRecord)}
            kwargs["audit_digest"] = digest
            sealed = TechnologyAuditRecord(**kwargs)
        else:
            # Verify any provided non-empty digest is correct — reject 64-zero or tampered
            expected = compute_audit_digest(record)
            if expected != record.audit_digest:
                raise DigestMismatchError(f"audit digest mismatch for {record.candidate_name}@{record.version}")

        self._records[ref] = sealed
        self._norm_index[norm] = ref
        self._log.seed(
            entity="technology_audit",
            entity_id=ref,
            initial_fields={"status": sealed.status.value},
        )
        return sealed

    def transition(
        self,
        candidate_name: str,
        version: str,
        to_status: AuditStatus,
        *,
        by: str | None = None,
        evidence_ref: str | None = None,
        at: str | None = None,
        actor_role: str | None = None,
    ) -> TechnologyAuditRecord:
        """Append a STATUS event and rebuild the record (I13).

        Validates the ``LEGAL_AUDIT_TRANSITIONS`` graph; terminal states
        never convert. Unifies ``actor_role`` and ``by`` to one canonical actor name (r4_b2).

        Public arguments are validated with exact types BEFORE any record
        lookup or event append (r5_b2), so an invalid call has ZERO side
        effects; defaults are selected with explicit ``is None`` checks and a
        blank effective actor is rejected fail-closed.
        """
        # -- r5_b2: exact-type validation, no side effects ---------------------
        if not isinstance(candidate_name, str) or not candidate_name.strip():
            raise CanonicalValidationError(
                f"candidate_name must be a non-empty str, got {candidate_name!r}"
            )
        if not isinstance(version, str) or not version.strip():
            raise CanonicalValidationError(
                f"version must be a non-empty str, got {version!r}"
            )
        if not isinstance(to_status, AuditStatus):
            raise CanonicalValidationError(
                f"to_status must be AuditStatus, got {to_status!r}"
            )
        for arg_name, arg_value in (
            ("by", by),
            ("actor_role", actor_role),
            ("evidence_ref", evidence_ref),
            ("at", at),
        ):
            if arg_value is not None and not isinstance(arg_value, str):
                raise CanonicalValidationError(
                    f"{arg_name} must be str or None, got {type(arg_value)!r}"
                )
        if at is not None and not at.strip():
            raise CanonicalValidationError("at must be a non-blank ISO-8601 string when provided")
        # Defaults chosen via explicit `is None` checks — truthiness would
        # silently swallow False/0/"" values (r5_b2).
        stamp = utc_now_iso() if at is None else at
        effective_actor = actor_role if actor_role is not None else by
        if effective_actor is None:
            effective_actor = "auditor"
        if not effective_actor.strip():
            raise CanonicalValidationError(
                f"effective actor must be a non-blank str, got {effective_actor!r}"
            )

        # resolve real record for case-insensitivity
        rec = self.get(candidate_name, version)
        ref = _audit_ref(rec.candidate_name, rec.version)
        current = self.project_status(rec.candidate_name, rec.version)
        allowed = LEGAL_AUDIT_TRANSITIONS.get(current, frozenset())
        if to_status not in allowed:
            raise StateTransitionError(
                f"illegal audit transition {current.value} -> {to_status.value} for {ref}"
            )
        self._log.append_event(
            StateEvent(
                entity="technology_audit",
                entity_id=ref,
                event_type="STATUS",
                from_value=current.value,
                to_value=to_status.value,
                actor_role=effective_actor,
                at=stamp,
                evidence_ref=evidence_ref,
            )
        )
        # rebuild frozen record with new status + history (by aligns with actor_role)
        history = rec.status_history + (
            StatusHistoryEntry(status=to_status, at=stamp, by=effective_actor, evidence_ref=evidence_ref),
        )
        kwargs: dict[str, object] = {f.name: getattr(rec, f.name) for f in dc_fields(TechnologyAuditRecord)}
        kwargs["status"] = to_status
        kwargs["status_history"] = history
        updated = TechnologyAuditRecord(**kwargs)  # type: ignore[arg-type]
        self._records[ref] = updated
        return updated

    def audit(
        self,
        candidate_name: str,
        version: str,
        *,
        by: str = "auditor",
        evidence_ref: str | None = None,
        at: str | None = None,
    ) -> TechnologyAuditRecord:
        """Convenience: DRAFT → AUDITED."""
        return self.transition(candidate_name, version, AuditStatus.AUDITED, by=by, evidence_ref=evidence_ref, at=at)

    def supersede(
        self,
        candidate_name: str,
        version: str,
        *,
        by: str = "auditor",
        evidence_ref: str | None = None,
        at: str | None = None,
    ) -> TechnologyAuditRecord:
        """Convenience: AUDITED → SUPERSEDED."""
        return self.transition(candidate_name, version, AuditStatus.SUPERSEDED, by=by, evidence_ref=evidence_ref, at=at)

    def reject(
        self,
        candidate_name: str,
        version: str,
        *,
        by: str = "auditor",
        evidence_ref: str | None = None,
        at: str | None = None,
    ) -> TechnologyAuditRecord:
        """Convenience: DRAFT/AUDITED → REJECTED."""
        rec = self.get(candidate_name, version)
        cur = self.project_status(rec.candidate_name, rec.version)
        if cur == AuditStatus.DRAFT or cur == AuditStatus.AUDITED:
            return self.transition(candidate_name, version, AuditStatus.REJECTED, by=by, evidence_ref=evidence_ref, at=at)
        raise StateTransitionError(f"cannot reject from {cur.value}")

    # -- JSON round-trip -------------------------------------------------------

    def to_dict(self) -> dict[str, object]:
        """JSON-safe dict — records sorted for deterministic digest."""
        # sort records for deterministic JSON
        sorted_records = sorted(self._records.values(), key=lambda r: (r.candidate_name, r.version))
        return {
            "schema_version": LEDGER_SCHEMA,
            "records": [record_to_dict(r) for r in sorted_records],
            "events": [event_to_dict(e) for e in self._log.events],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> AuditLedger:
        """Strict loader with closed-schema semantics."""
        if not isinstance(data, dict):
            raise ClosedSchemaError(f"ledger must be object, got {type(data)!r}")
        known = {"schema_version", "records", "events"}
        unknown = set(data) - known
        if unknown:
            raise ClosedSchemaError(f"unknown field(s) in {LEDGER_SCHEMA}: {sorted(unknown)}")
        if data.get("schema_version") != LEDGER_SCHEMA:
            raise CanonicalValidationError(f"ledger schema_version must be {LEDGER_SCHEMA!r}")

        raw_records = data.get("records")
        if not isinstance(raw_records, list):
            raise CanonicalValidationError("ledger.records must be list")
        ledger = cls()
        for idx, raw in enumerate(raw_records):
            if not isinstance(raw, dict):
                raise ClosedSchemaError(f"records[{idx}] must be object")
            # use load_audit_record to enforce sealed digest
            rec = load_audit_record(raw)
            ref = _audit_ref(rec.candidate_name, rec.version)
            norm = (_norm_key(rec.candidate_name), rec.version)
            # Fail closed on exact or normalized candidate/version duplicates (r4_b3)
            if ref in ledger._records:
                raise CanonicalValidationError(f"duplicate record {ref!r} in ledger load")
            if norm in ledger._norm_index:
                raise CanonicalValidationError(
                    f"normalized duplicate {ref!r} conflicts with {ledger._norm_index[norm]!r} in ledger load"
                )
            ledger._records[ref] = rec
            ledger._norm_index[norm] = ref

        # seed event log with initial status for each record (so project_state works
        # even before replaying events). This mirrors registry's seed logic.
        for ref, rec in ledger._records.items():
            # If history is non-empty the initial seed was DRAFT and events replay
            # to the final status; if history empty the seed was the record's own
            # status (e.g. direct AUDITED without transition). Detect and seed
            # accordingly so projection matches.
            initial = AuditStatus.DRAFT.value if rec.status_history else rec.status.value
            ledger._log.seed(
                entity="technology_audit",
                entity_id=ref,
                initial_fields={"status": initial},
            )

        raw_events = data.get("events", [])
        if not isinstance(raw_events, list):
            raise CanonicalValidationError("ledger.events must be list")
        # replay events in order — closed-schema for event keys (blocker w4)
        known_event_keys = {
            "entity",
            "entity_id",
            "event_type",
            "from_value",
            "to_value",
            "actor_role",
            "at",
            "evidence_ref",
            "payload",
        }
        for idx, raw in enumerate(raw_events):
            if not isinstance(raw, dict):
                raise ClosedSchemaError(f"events[{idx}] must be object")
            unknown_evt = set(raw) - known_event_keys
            if unknown_evt:
                raise ClosedSchemaError(f"unknown field(s) in events[{idx}]: {sorted(unknown_evt)}")
            # strict type checks for event fields (blocker r3_b3 — evidence_ref=123 must fail)
            for field_name in ("entity", "entity_id", "event_type", "to_value", "actor_role", "at"):
                val = raw.get(field_name)
                if not isinstance(val, str):
                    raise CanonicalValidationError(f"events[{idx}].{field_name} must be str, got {type(val)!r}")
            for field_name in ("from_value", "evidence_ref"):
                val = raw.get(field_name)
                if val is not None and not isinstance(val, str):
                    raise CanonicalValidationError(f"events[{idx}].{field_name} must be str or null, got {type(val)!r}")
            payload_val = raw.get("payload", {})
            if not isinstance(payload_val, dict):
                raise CanonicalValidationError(f"events[{idx}].payload must be object, got {type(payload_val)!r}")
            # reconstruct StateEvent (closed schema via StateEvent.__post_init__)
            evt = StateEvent(
                entity=raw["entity"],
                entity_id=raw["entity_id"],
                event_type=raw["event_type"],
                from_value=raw.get("from_value"),
                to_value=raw["to_value"],
                actor_role=raw["actor_role"],
                at=raw["at"],
                evidence_ref=raw.get("evidence_ref"),
                payload=dict(payload_val),
            )
            ledger._log.append_event(evt)
            # also update the frozen record's status/history to stay in sync
            # (record already has latest status after replay; we just ensure
            # ledger's _records reflects projected status if needed).
            # For strictness, we verify that projected status matches record's field.
            # If mismatch, we update record to projected.

        # After replay, verify each record's status/history matches projected events — fail closed (blocker w3 + w7)
        for ref, rec in list(ledger._records.items()):
            try:
                proj = ledger._log.project_state(ref, entity="technology_audit")
                proj_status = AuditStatus(str(proj.get("status", rec.status.value)))
                if proj_status != rec.status:
                    raise CanonicalValidationError(
                        f"ledger status mismatch for {ref}: stored {rec.status.value} != projected {proj_status.value} — events must drive status"
                    )
                # Non-DRAFT status must be reached via events/history — reject direct AUDITED with empty history
                if rec.status != AuditStatus.DRAFT and not rec.status_history:
                    raise CanonicalValidationError(
                        f"{ref}: non-DRAFT status {rec.status.value} requires non-empty status_history and events"
                    )
                # Verify history length matches events and contents align (w7)
                hist_len = len(rec.status_history)
                events_for_ref = [e for e in ledger._log.events if e.entity_id == ref and e.entity == "technology_audit"]
                if hist_len != len(events_for_ref):
                    raise CanonicalValidationError(
                        f"ledger history mismatch for {ref}: status_history len {hist_len} != events {len(events_for_ref)}"
                    )
                for idx, (hist, ev) in enumerate(zip(rec.status_history, events_for_ref)):
                    # event_type must be STATUS and entity must match
                    if ev.event_type != "STATUS":
                        raise CanonicalValidationError(f"invalid event_type {ev.event_type!r} for {ref} (expected STATUS)")
                    if ev.entity != "technology_audit":
                        raise CanonicalValidationError(f"invalid entity {ev.entity!r} for {ref}")
                    if ev.entity_id != ref:
                        raise CanonicalValidationError(f"event entity_id mismatch {ev.entity_id!r} != {ref!r}")
                    if ev.from_value == ev.to_value:
                        raise CanonicalValidationError(f"NOOP event for {ref} at index {idx}: from==to {ev.from_value!r}")
                    if hist.status.value != ev.to_value:
                        raise CanonicalValidationError(
                            f"history[{idx}] status {hist.status.value!r} != event to_value {ev.to_value!r} for {ref}"
                        )
                    if hist.at != ev.at:
                        raise CanonicalValidationError(f"history[{idx}] at mismatch for {ref}")
                    if hist.by != ev.actor_role:
                        raise CanonicalValidationError(f"history[{idx}] by {hist.by!r} != actor_role {ev.actor_role!r} for {ref}")
                    if hist.evidence_ref != ev.evidence_ref:
                        raise CanonicalValidationError(f"history[{idx}] evidence_ref mismatch for {ref}")
                    expected_from = rec.status_history[idx - 1].status.value if idx > 0 else AuditStatus.DRAFT.value
                    if ev.from_value != expected_from:
                        raise CanonicalValidationError(
                            f"event[{idx}] from_value {ev.from_value!r} != expected {expected_from!r} for {ref}"
                        )
                    # Legal transition check
                    try:
                        from_st = AuditStatus(ev.from_value) if ev.from_value else None
                        to_st = AuditStatus(ev.to_value)
                    except ValueError as exc:
                        raise CanonicalValidationError(f"invalid status in event for {ref}: {exc}") from exc
                    if from_st is not None and to_st not in LEGAL_AUDIT_TRANSITIONS.get(from_st, frozenset()):
                        raise CanonicalValidationError(
                            f"illegal transition {ev.from_value!r} -> {ev.to_value!r} for {ref}"
                        )
            except KeyError as exc:
                raise CanonicalValidationError(f"missing projection for {ref}") from exc

        return ledger

    @classmethod
    def from_json(cls, text: str) -> AuditLedger:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise CanonicalValidationError(f"ledger JSON invalid: {exc}") from exc
        return cls.from_dict(data)


__all__ = [
    "LEDGER_SCHEMA",
    "AuditLedger",
    "EventLogView",
    "_audit_ref",
]
