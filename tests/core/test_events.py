"""I13 append-only event log: append_event / project_state, no setters."""

from __future__ import annotations

import dataclasses
import json

import pytest

from core.enums import ModelStatus, ReleaseLane, ReviewState
from core.errors import StateTransitionError
from core.events import EventLog, Projection, StateEvent, event_to_dict


def make_log() -> EventLog:
    log = EventLog()
    log.seed("derived_artifact", "art-1", {"status": "ACTIVE", "review_state": "PREDICTED"})
    return log


def test_append_and_project_replay():
    log = make_log()
    log.append_event(
        StateEvent(
            entity="derived_artifact",
            entity_id="art-1",
            event_type="REVIEW",
            from_value="PREDICTED",
            to_value="ACCEPTED",
            actor_role="HUMAN",
            at="2026-01-15T10:30:00Z",
        )
    )
    projection = log.project_state("art-1")
    assert projection.get("review_state") == "ACCEPTED"
    assert projection.get("status") == "ACTIVE"


def test_projection_is_read_only_view():
    log = make_log()
    projection = log.project_state("art-1")
    with pytest.raises(TypeError):
        projection.fields["status"] = "HACKED"  # type: ignore[index]


def test_from_state_mismatch_fails_closed():
    log = make_log()
    log.append_event(
        StateEvent(
            entity="derived_artifact",
            entity_id="art-1",
            event_type="STATUS",
            from_value="ACTIVE",
            to_value="INVALIDATED",
            actor_role="ENGINE",
            at="2026-01-15T10:31:00Z",
        )
    )
    with pytest.raises(StateTransitionError):
        log.append_event(
            StateEvent(
                entity="derived_artifact",
                entity_id="art-1",
                event_type="STATUS",
                from_value="ACTIVE",  # stale view - another writer moved it
                to_value="SUPERSEDED",
                actor_role="ENGINE",
                at="2026-01-15T10:32:00Z",
            )
        )


def test_unknown_entity_rejected():
    log = EventLog()
    with pytest.raises(StateTransitionError):
        log.append_event(
            StateEvent(
                entity="model_registry",
                entity_id="ghost@1.0.0",
                event_type="PROMOTED",
                from_value="CHALLENGER",
                to_value="CHAMPION",
                actor_role="registry-cli",
                at="2026-01-15T10:30:00Z",
            )
        )


def test_double_seeding_rejected():
    log = make_log()
    with pytest.raises(StateTransitionError):
        log.seed("derived_artifact", "art-1", {"status": "ACTIVE"})


def test_events_are_immutable_records():
    log = make_log()
    event = StateEvent(
        entity="derived_artifact",
        entity_id="art-1",
        event_type="STATUS",
        from_value="ACTIVE",
        to_value="STALE",
        actor_role="ENGINE",
        at="2026-01-15T10:30:00Z",
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        event.to_value = "HACKED"  # type: ignore[misc]
    log.append_event(event)
    assert log.events == (event,)


def test_event_and_projection_json_round_trip():
    """Luna R3 fix #3: events and projections serialize as plain JSON."""
    log = make_log()
    event = StateEvent(
        entity="derived_artifact",
        entity_id="art-1",
        event_type="REVIEW",
        from_value="PREDICTED",
        to_value="ACCEPTED",
        actor_role="HUMAN",
        at="2026-01-15T10:30:00Z",
        evidence_ref="review.md",
        payload={"note": "ok"},
    )
    log.append_event(event)
    dumped = json.dumps(event_to_dict(event))
    restored = json.loads(dumped)
    assert restored["to_value"] == "ACCEPTED"
    assert restored["payload"] == {"note": "ok"}

    projection = log.project_state("art-1")
    assert isinstance(projection, Projection)
    projection_json = json.dumps(projection.to_dict())
    assert json.loads(projection_json) == {
        "status": "ACTIVE",
        "review_state": "ACCEPTED",
    }


class TestB11ProjectionJsonSafety:
    """Blocker b11: project_state()/Projection emit ONLY JSON-ready values -
    enum-valued seeds and events become their string .value recursively."""

    def test_b11_enum_seeded_projection_dumps_cleanly(self):
        log = EventLog()
        log.seed(
            "model_registry",
            "seg@1.0.0",
            {"status": ModelStatus.EXPERIMENTAL},  # type: ignore[dict-item]
        )
        projection = log.project_state("seg@1.0.0")
        assert isinstance(projection.get("status"), str)
        dumped = json.dumps(projection.to_dict())
        assert json.loads(dumped) == {"status": "EXPERIMENTAL"}

    def test_b11_enum_event_to_value_sanitized_in_projection(self):
        log = make_log()
        log.append_event(
            StateEvent(
                entity="derived_artifact",
                entity_id="art-1",
                event_type="REVIEW",
                from_value="PREDICTED",
                to_value=ReviewState.ACCEPTED,  # type: ignore[arg-type]
                actor_role="HUMAN",
                at="2026-01-15T10:30:00Z",
            )
        )
        projection = log.project_state("art-1")
        dumped = json.dumps(projection.to_dict())  # must not raise TypeError
        assert json.loads(dumped)["review_state"] == "ACCEPTED"

    def test_b11_nested_enum_values_sanitized_recursively(self):
        log = EventLog()
        log.seed(
            "model_registry",
            "ens@1.0.0",
            {
                "meta": {"status": ModelStatus.CHALLENGER},  # type: ignore[dict-item]
                "lanes": [ReleaseLane.PERSONAL_RESEARCH],  # type: ignore[dict-item]
            },
        )
        projection = log.project_state("ens@1.0.0")
        parsed = json.loads(json.dumps(projection.to_dict()))
        assert parsed["meta"]["status"] == "CHALLENGER"
        assert parsed["lanes"] == ["PERSONAL_RESEARCH"]

    def test_b11_parse_back_equals_re_serialized_projection(self):
        log = make_log()
        log.append_event(
            StateEvent(
                entity="derived_artifact",
                entity_id="art-1",
                event_type="STATUS",
                from_value="ACTIVE",
                to_value="SUPERSEDED",
                actor_role="ENGINE",
                at="2026-01-15T10:31:00Z",
            )
        )
        first = log.project_state("art-1")
        replayed = log.project_state("art-1")
        assert json.loads(first.to_json()) == json.loads(replayed.to_json())
        assert json.loads(first.to_json()) == first.to_dict()
        assert json.loads(first.to_json()) == {
            "status": "SUPERSEDED",
            "review_state": "PREDICTED",
        }

    def test_b11_json_dumps_of_projection_object_directly_succeeds(self):
        # Literal blocker probe: the Projection returned by project_state()
        # must be JSON-serializable as-is (dict-backed, enum-free values).
        projection = make_log().project_state("art-1")
        dumped = json.dumps(projection)
        parsed = json.loads(dumped)
        assert parsed == {"status": "ACTIVE", "review_state": "PREDICTED"}
        reprojected = make_log().project_state("art-1")
        assert json.loads(json.dumps(reprojected)) == parsed

    def test_b11_projection_fields_remain_read_only(self):
        projection = make_log().project_state("art-1")
        with pytest.raises(TypeError):
            projection.fields["status"] = "HACKED"  # type: ignore[index]
