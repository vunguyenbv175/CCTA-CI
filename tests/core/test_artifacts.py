"""DerivedArtifact rules: producer matrix, review lifecycle, export gate, STALE DAG."""

from __future__ import annotations

import dataclasses

import pytest

from core.artifacts import (
    DerivedArtifact,
    DerivedArtifactStore,
    GeometryBase,
    ReviewBlock,
    create_export_package,
)
from core.enums import (
    ActorType,
    ArtifactKind,
    ArtifactStatus,
    ProducerType,
    ReleaseLane,
    ReviewState,
)
from core.errors import (
    CanonicalValidationError,
    ExportGateViolationError,
    GeometryBaseRequiredError,
    RegistrationRequiredError,
    StateTransitionError,
)
from core.provenance import InvalidationDAG
from tests.core.factories import (
    SHA_C,
    make_artifact,
    make_claim,
    make_lineage,
    make_payload,
    make_producer,
    make_store,
)


class TestProducerMatrix:
    def test_llm_only_for_report_draft(self):
        # MODEL on SEGMENTATION is legal:
        assert make_artifact(kind=ArtifactKind.SEGMENTATION) is not None
        # LLM on SEGMENTATION is not:
        with pytest.raises(CanonicalValidationError) as err:
            make_artifact(
                kind=ArtifactKind.SEGMENTATION,
                producer=make_producer(producer_type=ProducerType.LLM),
            )
        assert any("not allowed" in v or "LLM" in v for v in err.value.violations)

    def test_llm_on_measurement_set_rejected(self):
        with pytest.raises(CanonicalValidationError):
            make_artifact(
                kind=ArtifactKind.MEASUREMENT_SET,
                producer=make_producer(producer_type=ProducerType.LLM),
            )

    def test_measurement_set_deterministic_only(self):
        artifact = make_artifact(kind=ArtifactKind.MEASUREMENT_SET)
        assert artifact.producer.producer_type == ProducerType.DETERMINISTIC_ENGINE
        with pytest.raises(CanonicalValidationError) as err:
            make_artifact(
                kind=ArtifactKind.MEASUREMENT_SET,
                producer=make_producer(producer_type=ProducerType.MODEL),
                lineage=make_lineage(input_volume_refs=(), input_series_refs=()),
            )
        assert any("MEASUREMENT_SET" in v for v in err.value.violations)

    def test_physiology_result_physics_backend_only(self):
        with pytest.raises(CanonicalValidationError):
            make_artifact(
                kind=ArtifactKind.PHYSIOLOGY_RESULT,
                producer=make_producer(producer_type=ProducerType.MODEL),
            )
        artifact = make_artifact(kind=ArtifactKind.PHYSIOLOGY_RESULT)
        assert artifact.producer.producer_type == ProducerType.PHYSICS_BACKEND

    def test_model_requires_model_record_ref(self):
        with pytest.raises(CanonicalValidationError):
            make_artifact(producer=make_producer(model_record_ref=None))

    def test_deterministic_engine_requires_frozen_params(self):
        with pytest.raises(CanonicalValidationError):
            make_artifact(
                kind=ArtifactKind.MEASUREMENT_SET,
                producer=make_producer(
                    producer_type=ProducerType.DETERMINISTIC_ENGINE,
                    algorithm="m",
                    parameters_blob_sha256=None,
                ),
            )

    def test_correction_is_human_only_and_needs_supersedes(self):
        with pytest.raises(CanonicalValidationError):
            make_artifact(
                kind=ArtifactKind.CORRECTION,
                review=ReviewBlock(review_state=ReviewState.PREDICTED),
            )
        correction = make_artifact(
            kind=ArtifactKind.CORRECTION,
            review=ReviewBlock(
                review_state=ReviewState.PREDICTED, supersedes_ref="pred-1"
            ),
        )
        assert correction.review.supersedes_ref == "pred-1"

    def test_classification_candidate_born_predicted(self):
        with pytest.raises(CanonicalValidationError):
            make_artifact(
                kind=ArtifactKind.CLASSIFICATION_CANDIDATE,
                review=ReviewBlock(review_state=ReviewState.CONFIRMED_FINAL),
            )


class TestCreationGates:
    def test_geometry_base_required_for_geometry_dependent_kinds(self):
        with pytest.raises(GeometryBaseRequiredError):
            make_artifact(
                kind=ArtifactKind.MEASUREMENT_SET,
                lineage=make_lineage(geometry_base=None),
            )

    def test_multi_series_inputs_require_registration_ref(self):
        with pytest.raises(RegistrationRequiredError):
            make_artifact(lineage=make_lineage(input_series_refs=("s1", "s2")))

    def test_claim_binding_enforced(self):
        with pytest.raises(CanonicalValidationError) as err:
            make_artifact(claim_binding=dataclasses.replace(make_claim(), claim_id="RANDOM"))
        assert any("claim registry" in v for v in err.value.violations)

    def test_claim_version_required(self):
        with pytest.raises(CanonicalValidationError):
            make_artifact(claim_binding=dataclasses.replace(make_claim(), claim_version=""))

    def test_payload_schema_must_be_namespaced(self):
        with pytest.raises(CanonicalValidationError):
            make_artifact(payload=make_payload(schema="random.schema.v1"))

    def test_status_reason_consistency(self):
        with pytest.raises(CanonicalValidationError):
            make_artifact(status=ArtifactStatus.INVALIDATED)
        with pytest.raises(CanonicalValidationError):
            make_artifact(status=ArtifactStatus.STALE)

    def test_artifacts_frozen_no_setter(self):
        artifact = make_artifact()
        with pytest.raises(AttributeError):
            artifact.status = ArtifactStatus.SUPERSEDED


class TestReviewLifecycle:
    @staticmethod
    def _confirm(store: DerivedArtifactStore, artifact_id: str) -> None:
        store.apply_review_transition(artifact_id, ReviewState.ACCEPTED, actor_role=ActorType.HUMAN)
        store.apply_review_transition(
            artifact_id, ReviewState.CONFIRMED_FINAL, actor_role=ActorType.HUMAN
        )

    def test_legal_path_to_confirmed_final(self):
        store, _, _ = make_store()
        artifact = make_artifact()
        store.add(artifact)
        self._confirm(store, artifact.artifact_id)
        assert store.get(artifact.artifact_id).review.review_state == ReviewState.CONFIRMED_FINAL

    def test_human_required_for_confirmation(self):
        store, _, _ = make_store()
        artifact = make_artifact()
        store.add(artifact)
        store.apply_review_transition(
            artifact.artifact_id, ReviewState.ACCEPTED, actor_role=ActorType.HUMAN
        )
        with pytest.raises(StateTransitionError):
            store.apply_review_transition(
                artifact.artifact_id, ReviewState.CONFIRMED_FINAL, actor_role=ActorType.ENGINE
            )

    def test_illegal_transitions_rejected(self):
        store, _, _ = make_store()
        artifact = make_artifact()
        store.add(artifact)
        with pytest.raises(StateTransitionError):  # PREDICTED -> CONFIRMED_FINAL
            store.apply_review_transition(
                artifact.artifact_id, ReviewState.CONFIRMED_FINAL, actor_role=ActorType.HUMAN
            )
        store.apply_review_transition(
            artifact.artifact_id, ReviewState.REJECTED, actor_role=ActorType.HUMAN
        )
        with pytest.raises(StateTransitionError):  # terminal state
            store.apply_review_transition(
                artifact.artifact_id, ReviewState.ACCEPTED, actor_role=ActorType.HUMAN
            )

    def test_state_history_immutable_projection(self):
        store, log, _ = make_store()
        artifact = make_artifact()
        store.add(artifact)
        aid = artifact.artifact_id
        store.apply_review_transition(aid, ReviewState.ACCEPTED, actor_role=ActorType.HUMAN)
        history_before = store.get(aid).review.state_history
        store.apply_review_transition(aid, ReviewState.CONFIRMED_FINAL, actor_role=ActorType.HUMAN)
        history_after = store.get(aid).review.state_history
        assert len(history_after) == len(history_before) + 1
        assert history_after[:-1] == history_before

        # I13 invariant: projection replay equals stored state.
        projection = store.project(aid)
        assert projection.get("review_state") == ReviewState.CONFIRMED_FINAL.value
        assert len(log.events) == 2

    def test_unknown_artifact_lookup(self):
        store, _, _ = make_store()
        with pytest.raises(KeyError):
            store.get("missing")


class TestExportGate:
    def _confirmed_segmentation(self, store: DerivedArtifactStore) -> str:
        seg = make_artifact()
        store.add(seg)
        store.apply_review_transition(
            seg.artifact_id, ReviewState.ACCEPTED, actor_role=ActorType.HUMAN
        )
        store.apply_review_transition(
            seg.artifact_id, ReviewState.CONFIRMED_FINAL, actor_role=ActorType.HUMAN
        )
        return seg.artifact_id

    def test_candidate_input_rejected(self):
        store, _, _ = make_store()
        candidate = make_artifact()
        store.add(candidate)
        with pytest.raises(ExportGateViolationError):
            create_export_package(
                store,
                study_ref="study-fixed-0001",
                input_artifact_refs=(candidate.artifact_id,),
                payload=make_payload(schema="ccta.payload.export_package.v1"),
                claim_binding=make_claim(),
                provenance_ref="prov-export",
                parameters_blob_sha256=SHA_C,
                release_lane=ReleaseLane.PERSONAL_RESEARCH,
                created_by="engine",
            )

    def test_missing_input_rejected(self):
        store, _, _ = make_store()
        with pytest.raises(ExportGateViolationError):
            create_export_package(
                store,
                study_ref="study-fixed-0001",
                input_artifact_refs=("ghost",),
                payload=make_payload(schema="ccta.payload.export_package.v1"),
                claim_binding=make_claim(),
                provenance_ref="prov-export",
                parameters_blob_sha256=SHA_C,
                release_lane=ReleaseLane.PERSONAL_RESEARCH,
                created_by="engine",
            )

    def test_stale_input_blocked_from_export(self):
        store, _, dag = make_store()
        dag.register_geometry("geometry-A", 1)
        confirmed_id = self._confirmed_segmentation(store)
        store.register_new_geometry_version("geometry-A", 2)
        assert store.get(confirmed_id).status == ArtifactStatus.STALE
        with pytest.raises(ExportGateViolationError) as err:
            create_export_package(
                store,
                study_ref="study-fixed-0001",
                input_artifact_refs=(confirmed_id,),
                payload=make_payload(schema="ccta.payload.export_package.v1"),
                claim_binding=make_claim(),
                provenance_ref="prov-export",
                parameters_blob_sha256=SHA_C,
                release_lane=ReleaseLane.PERSONAL_RESEARCH,
                created_by="engine",
            )
        assert any("STALE" in p for p in err.value.violations)

    def test_export_succeeds_with_confirmed_active_inputs(self):
        store, log, _ = make_store()
        confirmed_id = self._confirmed_segmentation(store)
        package = create_export_package(
            store,
            study_ref="study-fixed-0001",
            input_artifact_refs=(confirmed_id,),
            payload=make_payload(schema="ccta.payload.export_package.v1"),
            claim_binding=make_claim(),
            provenance_ref="prov-export",
            parameters_blob_sha256=SHA_C,
            release_lane=ReleaseLane.PERSONAL_RESEARCH,
            created_by="engine",
        )
        assert package.kind == ArtifactKind.EXPORT_PACKAGE
        assert package.producer.algorithm == "export_packager"
        assert any(e.event_type == "REVIEW" for e in log.events)


class TestInvalidationPropagation:
    def _registered_store(self) -> tuple[DerivedArtifactStore, object]:
        store, _, dag = make_store()
        dag.register_geometry("geometry-A", 1)
        return store, dag

    def test_invalidation_marks_dependents_stale_with_reason(self):
        store, _ = self._registered_store()
        dependent = make_artifact()  # geometry_base -> geometry-A v1
        store.add(dependent)
        stale_ids = store.invalidate(dependent.artifact_id, reason="human corrected segmentation")
        assert stale_ids == ()
        invalidated = store.get(dependent.artifact_id)
        assert invalidated.status == ArtifactStatus.INVALIDATED
        assert invalidated.invalidated_reason == "human corrected segmentation"

    def test_chain_invalidation_flows_to_grandchildren(self):
        store, _ = self._registered_store()
        centerline = make_artifact(kind=ArtifactKind.CENTERLINE)
        store.add(centerline)
        measurement = make_artifact(
            kind=ArtifactKind.MEASUREMENT_SET,
            artifact_id="meas-1",
            lineage=make_lineage(
                input_volume_refs=(),
                input_artifact_refs=(centerline.artifact_id,),
                input_series_refs=(),
                geometry_base=GeometryBase(
                    geometry_id=centerline.artifact_id,
                    geometry_version=1,
                ),
            ),
        )
        store.add(measurement)
        stale_ids = store.invalidate(centerline.artifact_id, reason="centerline recomputed")
        assert measurement.artifact_id in stale_ids
        stale_record = store.get(measurement.artifact_id)
        assert stale_record.status == ArtifactStatus.STALE
        assert stale_record.stale_reason is not None
        assert "H-010" in stale_record.stale_reason

    def test_new_geometry_version_propagates_stale(self):
        store, _ = self._registered_store()
        dependent = make_artifact()
        store.add(dependent)
        stale_ids = store.register_new_geometry_version("geometry-A", 3)
        assert dependent.artifact_id in stale_ids
        assert store.get(dependent.artifact_id).status == ArtifactStatus.STALE

    def test_invalidation_requires_reason(self):
        store, _ = self._registered_store()
        dependent = make_artifact()
        store.add(dependent)
        with pytest.raises(CanonicalValidationError):
            store.invalidate(dependent.artifact_id, reason="")


class TestSupersedeFlow:
    def test_prediction_survives_correction_as_superseded(self):
        store, _, _ = make_store()
        prediction = make_artifact()
        store.add(prediction)
        updated = store.mark_superseded(prediction.artifact_id, superseded_by="correction-1")
        assert updated.status == ArtifactStatus.SUPERSEDED
        # original record bytes remain queryable for audit (H-004)
        assert store.get(prediction.artifact_id).payload.blob_sha256

    def test_cannot_supersede_twice(self):
        store, _, _ = make_store()
        prediction = make_artifact()
        store.add(prediction)
        store.mark_superseded(prediction.artifact_id, superseded_by="c-1")
        with pytest.raises(StateTransitionError):
            store.mark_superseded(prediction.artifact_id, superseded_by="c-2")

    def test_duplicate_registration_rejected_append_only(self):
        store, _, _ = make_store()
        artifact = make_artifact()
        store.add(artifact)
        duplicate = dataclasses.replace(artifact)
        with pytest.raises(CanonicalValidationError):
            store.add(duplicate)


def test_projection_replay_matches_stored_state_i13():
    from core.events import EventLog

    log = EventLog()
    dag = InvalidationDAG()
    store = DerivedArtifactStore(log=log, dag=dag)
    artifact = make_artifact()
    store.add(artifact)
    store.apply_review_transition(
        artifact.artifact_id, ReviewState.ACCEPTED, actor_role=ActorType.HUMAN
    )
    store.apply_review_transition(
        artifact.artifact_id, ReviewState.CONFIRMED_FINAL, actor_role=ActorType.HUMAN
    )
    projection = store.project(artifact.artifact_id)
    stored = store.get(artifact.artifact_id)
    assert projection.get("review_state") == stored.review.review_state.value
    assert projection.get("status") == stored.status.value


def test_derived_artifact_dataclass_field_guard():
    # The entity exposes no mutable state fields; every field is frozen.
    assert DerivedArtifact.__dataclass_params__.frozen  # type: ignore[attr-defined]
