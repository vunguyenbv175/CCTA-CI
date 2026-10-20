"""Append-only state event log (I13, Luna fix #8).

"Mutable-looking" fields (``review_state``, ``status``, ``release_lane``,
promotion metadata) have NO setters. Every change is an appended event
``{entity, entity_id, from, to, actor_role, at, evidence_ref}``; the current
state is a read-only projection replayed from the event log. The storage API
exposes only :meth:`EventLog.append_event` and :meth:`EventLog.project_state`.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from dataclasses import fields as dc_fields
from datetime import UTC, datetime
from enum import Enum
from types import MappingProxyType
from typing import cast

from core.errors import StateTransitionError


def utc_now_iso() -> str:
    """Current time as ISO-8601 UTC with explicit ``Z`` suffix."""
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def validate_iso8601_utc(value: object, field_name: str) -> str:
    """Return *value* if it parses as an ISO-8601 timestamp, else raise."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be an ISO-8601 timestamp string")
    text = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"{field_name} is not ISO-8601: {value!r}") from exc
    return value


@dataclass(frozen=True)
class StateEvent:
    """One immutable state-transition event."""

    entity: str
    entity_id: str
    event_type: str
    from_value: str | None
    to_value: str
    actor_role: str
    at: str
    evidence_ref: str | None = None
    payload: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        missing = [
            fname
            for fname in ("entity", "entity_id", "event_type", "to_value", "actor_role", "at")
            if not getattr(self, fname)
        ]
        if missing:
            raise ValueError(f"StateEvent missing fields: {missing}")
        validate_iso8601_utc(self.at, "at")


class Projection(dict[str, str]):
    """Read-only current-state view of one entity.

    Dict-backed (r5 attempt 3, blocker b11): ``project_state()`` sanitizes
    every value to a JSON-ready form (enums become their string ``.value``
    recursively), so ``json.dumps(project_state(...))`` serializes the
    projection natively without TypeError.
    """

    def __init__(self, entity_id: str, fields: Mapping[str, str]) -> None:
        super().__init__(fields)
        self._entity_id = entity_id

    @property
    def entity_id(self) -> str:
        return self._entity_id

    @property
    def fields(self) -> Mapping[str, str]:
        """Read-only mapping view; mutation attempts raise TypeError."""
        return MappingProxyType(self)

    def to_dict(self) -> dict[str, str]:
        """Plain-dict view; every value is a JSON-ready value (b11)."""
        return dict(self)

    def to_json(self) -> str:
        """Canonical JSON serialization of this projection."""
        return json.dumps(self, sort_keys=True)


def event_to_dict(event: StateEvent) -> dict[str, object]:
    """JSON-ready dict of *event*; enum-free by construction so
    ``json.dumps`` always round-trips event payloads exactly."""
    return {
        f.name: _json_safe_value(getattr(event, f.name)) for f in dc_fields(event)
    }


def _json_safe_value(value: object) -> object:
    """Enum instances become their plain string values; mappings become dicts."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {key: _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_json_safe_value(item) for item in value)
    return value


class EventLog:
    """Append-only event store with deterministic projection replay.

    ``append_event`` fails closed unless the event's ``from_value`` matches
    the currently projected value (optimistic-concurrency guard), so no two
    writers can race past stale state.
    """

    #: Maps event_type -> projected-field name for the standard vocabulary.
    EVENT_FIELD: Mapping[str, str] = {
        "STATUS": "status",
        "REVIEW": "review_state",
        "LANE_PROMOTED": "release_lane",
        "REGISTERED": "registry_status",
        "INTAKE_PASSED": "registry_status",
        "PROMOTED": "registry_status",
        "DEPRECATED": "registry_status",
        "BLOCKED": "registry_status",
        "ROLLED_BACK": "registry_status",
    }

    def __init__(self) -> None:
        self._events: list[StateEvent] = []
        self._initial: dict[tuple[str, str], dict[str, str]] = {}

    def seed(self, entity: str, entity_id: str, initial_fields: Mapping[str, str]) -> None:
        """Register the creation-time state of an entity (its birth record)."""
        key = (entity, entity_id)
        if key in self._initial:
            raise StateTransitionError(
                f"{entity}/{entity_id} already seeded into the event log"
            )
        self._initial[key] = dict(initial_fields)

    @property
    def events(self) -> tuple[StateEvent, ...]:
        """All appended events, in append order (read-only view)."""
        return tuple(self._events)

    def initial_states(
        self,
    ) -> tuple[tuple[tuple[str, str], Mapping[str, str]], ...]:
        """Read-only view of the seeded birth records (additive WS-02 export).

        WS-02 session persistence must be able to serialize the full journal
        (seeds + events) without reaching into private state. This accessor
        exposes the seeds and changes no behavior; projections remain the only
        sanctioned way to read current state.
        """
        return tuple(
            ((entity, entity_id), MappingProxyType(dict(fields_map)))
            for (entity, entity_id), fields_map in self._initial.items()
        )

    def append_event(self, event: StateEvent) -> StateEvent:
        """Append *event* after verifying its ``from_value`` projection.

        Raises :class:`StateTransitionError` when the entity was never
        seeded, or when another writer already moved the projected field
        away from ``event.from_value``.
        """
        key = (event.entity, event.entity_id)
        if key not in self._initial and not any(
            (e.entity, e.entity_id) == key for e in self._events
        ):
            raise StateTransitionError(
                f"{event.entity}/{event.entity_id} unknown to the event log"
            )
        field_name = self.EVENT_FIELD.get(event.event_type)
        if field_name is not None:
            current = self.project_state(event.entity_id, entity=event.entity).get(field_name)
            if current != event.from_value:
                raise StateTransitionError(
                    f"{event.entity}/{event.entity_id} {field_name}: expected "
                    f"from={event.from_value!r}, actual={current!r}"
                )
        self._events.append(event)
        return event

    def project_state(
        self, entity_id: str, *, entity: str | None = None
    ) -> Projection:
        """Replay all events for *entity_id* into a read-only projection."""
        key_filter: tuple[str, str] | None = None
        if entity is not None:
            key_filter = (entity, entity_id)
        relevant: list[StateEvent] = []
        initial: dict[str, str] = {}
        for (ent, eid), fields_map in self._initial.items():
            if eid == entity_id and (key_filter is None or key_filter == (ent, eid)):
                initial.update(fields_map)
        for event in self._events:
            if event.entity_id != entity_id:
                continue
            if key_filter is not None and (event.entity, event.entity_id) != key_filter:
                continue
            relevant.append(event)
            field_name = self.EVENT_FIELD.get(event.event_type)
            if field_name is not None:
                initial[field_name] = event.to_value
        if not initial and not relevant:
            raise KeyError(f"no state known for {entity!r}/{entity_id}")
        # R5 b11: emit ONLY JSON-ready values - enum-valued seeds/events are
        # converted to their string .value recursively through dicts/lists,
        # so projections serialize with json.dumps without TypeError.
        safe_fields = cast(
            Mapping[str, str], _json_safe_value(dict(initial))
        )
        return Projection(entity_id=entity_id, fields=safe_fields)

    def history_for(self, entity_id: str, *, entity: str | None = None) -> tuple[StateEvent, ...]:
        """Ordered event history for *entity_id* (immutable projection feed)."""

        def _matches(event: StateEvent) -> bool:
            if event.entity_id != entity_id:
                return False
            return entity is None or event.entity == entity

        return tuple(e for e in self._events if _matches(e))

    def __iter__(self) -> Iterator[StateEvent]:  # pragma: no cover - convenience
        return iter(tuple(self._events))


__all__ = [
    "EventLog",
    "Projection",
    "StateEvent",
    "event_to_dict",
    "utc_now_iso",
    "validate_iso8601_utc",
]
