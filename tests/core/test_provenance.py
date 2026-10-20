"""ProvenanceRecord: nine §44 items fail-closed, NotApplicable, DAG (H-007/H-010)."""

from __future__ import annotations

import dataclasses

import pytest

from core.digest import canonical_json
from core.enums import ActorType, ProvenanceEventType
from core.errors import (
    ProvenanceIncompleteError,
)
from core.provenance import (
    Actor,
    Environment,
    InputRef,
    InvalidationDAG,
    NotApplicable,
    verify_chain,
    verify_self_digest,
)
from tests.core.factories import (
    make_actor,
    make_na_provenance,
    make_provenance,
)


class TestNineItems:
    def test_complete_record_accepted_and_sealed(self):
        record = make_provenance()
        assert len(record.record_sha256) == 64
        assert verify_self_digest(record)

    def test_all_not_applicable_variant_accepted(self):
        record = make_na_provenance()
        assert verify_self_digest(record)

    def test_missing_source_image_inputs_rejected(self):
        with pytest.raises(ProvenanceIncompleteError) as err:
            make_provenance(inputs=())
        assert any("source image" in v for v in err.value.violations)

    def test_missing_preprocessing_rejected(self):
        with pytest.raises(ProvenanceIncompleteError) as err:
            make_provenance(preprocessing=None)
        assert any("preprocessing" in v for v in err.value.violations)

    def test_empty_preprocessing_string_is_missing(self):
        with pytest.raises(ProvenanceIncompleteError):
            make_provenance(preprocessing="   ")

    def test_model_branch_requires_model_record_ref(self):
        with pytest.raises(ProvenanceIncompleteError) as err:
            make_provenance(
                actor=Actor(actor_type=ActorType.MODEL, actor_id="seg-model"),
                model_record_ref=None,
            )
        assert any("model_record_ref" in v for v in err.value.violations)

    def test_algorithm_branch_requires_actor_id(self):
        with pytest.raises(ProvenanceIncompleteError):
            make_provenance(actor=Actor(actor_type=ActorType.ENGINE, actor_id="  "))

    def test_exactly_one_branch_xor(self):
        with pytest.raises(ProvenanceIncompleteError) as err:
            make_provenance(
                actor=make_actor(),
                model_record_ref="coronary-seg@1.0.0",  # engine + model ref: both branches
            )
        assert any(
            "model_record_ref also populated" in v for v in err.value.violations
        )

    def test_commit_sha_must_be_40_hex(self):
        with pytest.raises(ProvenanceIncompleteError) as err:
            make_provenance(
                environment=Environment(
                    software_version="1.0.0",
                    code_commit_sha="deadbeef",
                )
            )
        assert any("40-hex" in v for v in err.value.violations)

    def test_parameters_require_value_or_na(self):
        with pytest.raises(ProvenanceIncompleteError) as err:
            make_provenance(parameters_ref=None)
        assert any("parameters" in v for v in err.value.violations)

    def test_human_edit_requires_value_or_na(self):
        with pytest.raises(ProvenanceIncompleteError) as err:
            make_provenance(human_edit=None)
        assert any("human edits" in v for v in err.value.violations)

    def test_upstream_refs_required_for_non_root(self):
        with pytest.raises(ProvenanceIncompleteError) as err:
            make_provenance(upstream_refs=())
        assert any("upstream dependencies" in v for v in err.value.violations)

    def test_all_violations_listed_at_once(self):
        with pytest.raises(ProvenanceIncompleteError) as err:
            make_provenance(
                inputs=None,
                preprocessing=None,
                parameters_ref=None,
                human_edit=None,
                upstream_refs=(),
                environment=Environment(software_version="", code_commit_sha="bad"),
                subject_ref="",
            )
        assert len(err.value.violations) >= 6


class TestNotApplicableWrapper:
    def test_carries_value_and_reason(self) -> None:
        na: NotApplicable[None] = NotApplicable(value=None, reason="no parameters were used")
        assert na.reason == "no parameters were used"
        with pytest.raises(ProvenanceIncompleteError):
            NotApplicable[None](value=None, reason="")  # empty reason fails closed


class TestDigestAndChain:
    def test_self_digest_deterministic(self):
        a, b = make_provenance(), make_provenance()
        assert a.record_sha256 == b.record_sha256

    def test_tamper_detection_on_included_field(self):
        record = make_provenance()
        tampered = dataclasses.replace(record, subject_ref="evil")
        assert not verify_self_digest(tampered)

    def test_digest_independent_of_sealed_field_itself(self):
        from core.digest import DIGEST_EXCLUSIONS

        record = make_provenance()
        payload = dataclasses.asdict(record)
        exclusions = tuple(DIGEST_EXCLUSIONS["ccta.canonical.provenance_record.v1"])
        blob = canonical_json(payload, exclusions)
        import hashlib

        assert hashlib.sha256(blob.encode("utf-8")).hexdigest() == record.record_sha256

    def test_input_digests_bound_at_use_time(self):
        with pytest.raises(ProvenanceIncompleteError):
            make_provenance(inputs=(InputRef(ref="s1", kind="series", sha256_at_use="x"),))

    def test_unknown_input_kind_rejected(self):
        with pytest.raises(ProvenanceIncompleteError):
            make_provenance(
                inputs=(
                    InputRef(ref="s1", kind="holodeck", sha256_at_use="a" * 64),
                )
            )

    def test_chain_verification_detects_unresolvable_upstream(self):
        root = make_provenance(provenance_id="prov-root")
        child = make_provenance(provenance_id="prov-child", upstream_refs=("prov-root",))
        ok, problems = verify_chain([child, root])
        assert ok and not problems

        orphan = make_provenance(provenance_id="prov-orphan", upstream_refs=("missing",))
        ok2, problems2 = verify_chain([orphan])
        assert not ok2 and any("unresolvable" in p for p in problems2)


class TestInvalidationDAG:
    def test_geometry_versions_monotonic(self):
        dag = InvalidationDAG()
        dag.register_geometry("geo-1", 1)
        dag.register_geometry("geo-1", 2)
        with pytest.raises(ValueError):
            dag.register_geometry("geo-1", 2)
        with pytest.raises(ValueError):
            dag.register_geometry("geo-1", 1)

    def test_binding_requires_known_geometry(self):
        dag = InvalidationDAG()
        with pytest.raises(KeyError):
            dag.bind_dependent("dep-1", "ghost", 1)

    def test_double_binding_rejected(self):
        dag = InvalidationDAG()
        dag.register_geometry("geo-1", 1)
        dag.bind_dependent("dep-1", "geo-1", 1)
        with pytest.raises(ValueError):
            dag.bind_dependent("dep-1", "geo-1", 1)

    def test_version_advance_marks_dependents_stale(self):
        dag = InvalidationDAG()
        dag.register_geometry("geo-1", 1)
        dag.bind_dependent("measurement-1", "geo-1", 1)
        assert not dag.is_stale("measurement-1")
        dag.register_geometry("geo-1", 2)
        assert dag.is_stale("measurement-1")
        assert dag.stale_dependents("geo-1") == ("measurement-1",)

    def test_invalidation_propagates_transitively(self):
        dag = InvalidationDAG()
        dag.register_geometry("seg", 1)
        dag.register_geometry("centerline", 1)
        dag.bind_dependent("centerline", "seg", 1)  # centerline depends on seg
        dag.bind_dependent("stenosis", "centerline", 1)  # chain
        dag.mark_invalidated("seg")
        assert set(dag.stale_dependents("seg")) >= {"centerline", "stenosis"}

    def test_unbound_artifact_never_stale(self):
        dag = InvalidationDAG()
        dag.register_geometry("geo-1", 5)
        assert not dag.is_stale("unrelated")


def test_event_types_cover_the_append_only_vocabulary():
    expected = {
        "IMPORTED",
        "DISCOVERED",
        "CLASSIFIED",
        "DERIVED",
        "RECOMPUTED",
        "REVIEWED",
        "CORRECTED",
        "INVALIDATED",
        "SUPERSEDED",
        "EXPORTED",
        "LANE_PROMOTED",
    }
    assert {e.value for e in ProvenanceEventType} == expected
