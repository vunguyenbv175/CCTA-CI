"""Registry record schema: closed schema, digests, dataset-role REJECT rules."""

from __future__ import annotations

import dataclasses
import json
from typing import Any

import pytest

from core.enums import ModelStatus, ReleaseLane
from core.errors import CanonicalValidationError, ClosedSchemaError, DigestMismatchError
from models.registry.record import (
    AcceptanceRef,
    ChampionOf,
    DatasetBinding,
    EnsembleBlock,
    OwnerGateV2,
    PromotionBlock,
    TrainingSnapshot,
    compute_release_digest,
    load_registry_record,
    record_from_dict,
    record_to_dict,
    validate_dataset_role_rules,
    validate_promotion_block,
    validate_registry_record,
    verify_release_digest,
)
from tests.models.factories import (
    AT,
    make_benchmark_ref,
    make_record,
    make_training_ref,
    sha,
)


class TestB12SourceClassRequired:
    """Blocker b12 (Luna exact-SHA review @004eaaa): DatasetBinding.source_class
    is a safety-relevant F4 vocabulary field - it must carry NO silent
    default, so omitting it fails closed like b8 handedness/units."""

    def test_b12_source_class_has_no_dataclass_default(self):
        defaults = {f.name: f.default for f in dataclasses.fields(DatasetBinding)}
        assert defaults["source_class"] is dataclasses.MISSING

    def test_b12_omitting_source_class_fails_closed(self):
        base = make_training_ref()
        kwargs = {
            f.name: getattr(base, f.name)
            for f in dataclasses.fields(DatasetBinding)
            if f.name != "source_class"
        }
        with pytest.raises(TypeError):
            DatasetBinding(**kwargs)

    def test_b12_explicit_source_class_builds_and_full_record_validates(self):
        benchmark = make_benchmark_ref()
        assert benchmark.source_class == "local_real_world"
        training = make_training_ref()
        assert training.source_class == "public"
        validate_dataset_role_rules(make_record())


class TestB13LineageSnapshotSourceClass:
    """Blocker b13 (Luna exact-SHA review @8328cba): lineage training
    snapshots must carry an explicit F4 source_class so local_real_world
    provenance cannot hide in lineage (R0b/R2 now enforceable there)."""

    @staticmethod
    def _snapshot_kwargs_without_source_class(**overrides: Any) -> dict[str, Any]:
        base = make_record().lineage.training_snapshot
        assert base is not None
        kwargs: dict[str, Any] = {
            f.name: getattr(base, f.name)
            for f in dataclasses.fields(TrainingSnapshot)
            if f.name != "source_class"
        }
        kwargs.update(overrides)
        return kwargs

    def _record_with_snapshot(self, snap: TrainingSnapshot) -> Any:
        record = make_record()
        return dataclasses.replace(
            record,
            lineage=dataclasses.replace(record.lineage, training_snapshot=snap),
        )

    def test_b13_omitting_source_class_on_training_snapshot_fails(self):
        with pytest.raises(TypeError):
            TrainingSnapshot(**self._snapshot_kwargs_without_source_class())

    def test_b13_local_real_world_training_snapshot_rejected(self):
        snap = TrainingSnapshot(
            **self._snapshot_kwargs_without_source_class(source_class="local_real_world")
        )
        with pytest.raises(CanonicalValidationError) as err:
            validate_dataset_role_rules(self._record_with_snapshot(snap))
        assert any("R2" in v and "lineage" in v for v in err.value.violations)

    def test_b13_unknown_source_class_training_snapshot_rejected(self):
        snap = TrainingSnapshot(
            **self._snapshot_kwargs_without_source_class(source_class="scraped")
        )
        with pytest.raises(CanonicalValidationError) as err:
            validate_dataset_role_rules(self._record_with_snapshot(snap))
        assert any("R0b" in v and "lineage" in v for v in err.value.violations)

    def test_b13_public_source_class_training_snapshot_still_valid(self):
        record = make_record()
        assert record.lineage.training_snapshot is not None
        assert record.lineage.training_snapshot.source_class == "public"
        validate_dataset_role_rules(record)


def seal(record: Any) -> Any:
    return dataclasses.replace(record, release_digest=compute_release_digest(record))


class TestClosedSchema:
    def test_unknown_top_level_field_rejected(self):
        payload = record_to_dict(make_record())
        payload["prompt_injection"] = "yes"
        with pytest.raises(ClosedSchemaError):
            record_from_dict(payload)

    def test_unknown_nested_field_rejected(self):
        payload = record_to_dict(make_record())
        payload["weights"]["extra"] = 1
        with pytest.raises(ClosedSchemaError) as err:
            record_from_dict(payload)
        assert "'extra'" in str(err.value) and "weights" in str(err.value)

    def test_round_trip_preserves_digest(self):
        record = make_record()
        restored = record_from_dict(record_to_dict(record))
        assert compute_release_digest(restored) == compute_release_digest(record)
        assert restored == record

    def test_missing_required_field_rejected(self):
        payload = record_to_dict(make_record())
        del payload["model_id"]
        with pytest.raises(CanonicalValidationError):
            record_from_dict(payload)


class TestReleaseDigest:
    def test_deterministic(self):
        a, b = make_record(), make_record()
        assert compute_release_digest(a) == compute_release_digest(b)

    def test_self_reference_excluded_i12(self):
        seeded = dataclasses.replace(make_record(), release_digest="stale-digest")
        digest = compute_release_digest(seeded)
        assert digest == compute_release_digest(make_record())
        sealed = dataclasses.replace(seeded, release_digest=digest)
        verify_release_digest(sealed)

    def test_tamper_detection(self):
        record = make_record()
        sealed = dataclasses.replace(record, release_digest=compute_release_digest(record))
        verify_release_digest(sealed)
        tampered = dataclasses.replace(
            sealed, weights=dataclasses.replace(sealed.weights, sha256="f" * 64)
        )
        with pytest.raises(DigestMismatchError):
            verify_release_digest(tampered)


class TestDatasetRoleRejectRules:
    def _record_with(self, **policy_overrides: Any):
        record = make_record()

        if "training_data_refs" in policy_overrides:
            policy = dataclasses.replace(
                record.data_policy,
                training_data_refs=policy_overrides["training_data_refs"],
            )
            return dataclasses.replace(record, data_policy=policy)
        if "benchmark_ref" in policy_overrides:
            validation = dataclasses.replace(
                record.validation,
                benchmark_ref=policy_overrides["benchmark_ref"],
            )
            return dataclasses.replace(record, validation=validation)
        if "training_snapshot" in policy_overrides:
            lineage = dataclasses.replace(
                record.lineage,
                training_snapshot=policy_overrides["training_snapshot"],
            )
            return dataclasses.replace(record, lineage=lineage)
        return record

    def test_r1_locked_test_split_cannot_train(self):
        record = self._record_with(
            training_data_refs=(make_training_ref(split_role="locked_test"),)
        )
        with pytest.raises(CanonicalValidationError) as err:
            validate_dataset_role_rules(record)
        assert any("R1" in v for v in err.value.violations)

    def test_r2_local_real_world_training_rejected(self):
        record = self._record_with(
            training_data_refs=(make_training_ref(source_class="local_real_world"),)
        )
        with pytest.raises(CanonicalValidationError) as err:
            validate_dataset_role_rules(record)
        assert any("R2" in v for v in err.value.violations)

    @pytest.mark.parametrize(
        ("field_name", "value"),
        [
            ("split_role", "train"),
            ("locked", False),
            ("patient_disjoint_asserted", False),
        ],
    )
    def test_r3_benchmark_binding_requirements(self, field_name: str, value: object):
        bad = dataclasses.replace(make_benchmark_ref(), **{field_name: value})  # type: ignore[arg-type]
        record = self._record_with(benchmark_ref=bad)
        with pytest.raises(CanonicalValidationError) as err:
            validate_dataset_role_rules(record)
        assert any("R3" in v for v in err.value.violations)

    def test_r3_benchmark_mandatory(self):
        record = self._record_with(benchmark_ref=None)
        with pytest.raises(CanonicalValidationError) as err:
            validate_dataset_role_rules(record)
        assert any("R3" in v for v in err.value.violations)

    def test_r4_same_snapshot_conflicting_roles(self):
        refs = (
            make_training_ref(dataset_id="ds-x", snapshot_version="s1"),
            make_training_ref(dataset_id="ds-x", snapshot_version="s1", split_role="val"),
        )
        record = self._record_with(training_data_refs=refs)
        with pytest.raises(CanonicalValidationError) as err:
            validate_dataset_role_rules(record)
        assert any("R4" in v for v in err.value.violations)

    def test_r5_training_snapshot_needs_patient_disjointness_assertion(self):
        snap = TrainingSnapshot(
            dataset_id="ds-train",
            snapshot_version="2026-01",
            split_role="train",
            locked=True,
            checksum_sha256="a" * 64,
            source_class="public",
            patient_disjoint_asserted=False,
        )
        record = self._record_with(training_snapshot=snap)
        with pytest.raises(CanonicalValidationError) as err:
            validate_dataset_role_rules(record)
        assert any("R5" in v for v in err.value.violations)

    def test_r0_unknown_split_role_rejected(self):
        record = self._record_with(
            training_data_refs=(make_training_ref(split_role="mystery"),)
        )
        with pytest.raises(CanonicalValidationError) as err:
            validate_dataset_role_rules(record)
        assert any("R0a" in v for v in err.value.violations)

    def test_r0_unknown_source_class_rejected(self):
        record = self._record_with(
            training_data_refs=(make_training_ref(source_class="scraped"),)
        )
        with pytest.raises(CanonicalValidationError) as err:
            validate_dataset_role_rules(record)
        assert any("R0b" in v for v in err.value.violations)

    @pytest.mark.parametrize(
        ("second_role",),
        [("val",), ("calibration",), ("locked_test",)],
    )
    def test_r4_any_conflicting_roles_on_one_snapshot_rejected(self, second_role: str):
        refs = (
            make_training_ref(dataset_id="ds-x", snapshot_version="s1"),
            make_training_ref(dataset_id="ds-x", snapshot_version="s1", split_role=second_role),
        )
        record = self._record_with(training_data_refs=refs)
        with pytest.raises(CanonicalValidationError) as err:
            validate_dataset_role_rules(record)
        assert any("R4" in v and "conflicting roles" in v for v in err.value.violations)

    def test_r6_training_ref_requires_patient_disjointness_assertion(self):
        record = self._record_with(
            training_data_refs=(make_training_ref(patient_disjoint_asserted=False),)
        )
        with pytest.raises(CanonicalValidationError) as err:
            validate_dataset_role_rules(record)
        assert any("R6" in v for v in err.value.violations)

    def test_compliant_record_passes_all_rules(self):
        validate_dataset_role_rules(make_record())


class TestB9BenchmarkAndLineageRoleCoverage:
    """Blocker b9: role-conflict checks span the UNION of training refs,
    validation.benchmark_ref and lineage.training_snapshot (fail closed)."""

    @staticmethod
    def _record() -> Any:
        return make_record()

    def _with_benchmark(self, benchmark: Any) -> Any:
        record = self._record()
        return dataclasses.replace(
            record,
            validation=dataclasses.replace(record.validation, benchmark_ref=benchmark),
        )

    def _with_training_refs(self, refs: tuple[Any, ...]) -> Any:
        record = self._record()
        return dataclasses.replace(
            record,
            data_policy=dataclasses.replace(record.data_policy, training_data_refs=refs),
        )

    def _with_lineage_snapshot(self, snapshot: TrainingSnapshot) -> Any:
        record = self._record()
        return dataclasses.replace(
            record,
            lineage=dataclasses.replace(record.lineage, training_snapshot=snapshot),
        )

    def test_b9_benchmark_unknown_source_class_rejected(self):
        bad = make_benchmark_ref(source_class="scraped")
        with pytest.raises(CanonicalValidationError) as err:
            validate_dataset_role_rules(self._with_benchmark(bad))
        assert any("R0b" in v and "benchmark" in v for v in err.value.violations)

    def test_b9_benchmark_unknown_split_role_rejected(self):
        bad = make_benchmark_ref(split_role="mystery")
        with pytest.raises(CanonicalValidationError) as err:
            validate_dataset_role_rules(self._with_benchmark(bad))
        assert any("R0a" in v and "benchmark" in v for v in err.value.violations)

    def test_b9_benchmark_non_locked_test_split_role_rejected(self):
        bad = make_benchmark_ref(split_role="train")
        with pytest.raises(CanonicalValidationError) as err:
            validate_dataset_role_rules(self._with_benchmark(bad))
        assert any("R3a" in v for v in err.value.violations)

    def test_b9_same_snapshot_as_locked_test_benchmark_and_training_ref_rejected(self):
        # The locked-test snapshot is also referenced as a training ref.
        refs = (
            make_training_ref(dataset_id="ds-locked-test", snapshot_version="2026-01"),
        )
        record = self._with_benchmark(make_benchmark_ref())
        record = dataclasses.replace(
            record,
            data_policy=dataclasses.replace(record.data_policy, training_data_refs=refs),
        )
        with pytest.raises(CanonicalValidationError) as err:
            validate_dataset_role_rules(record)
        assert any("R4" in v and "conflicting roles" in v for v in err.value.violations)
        assert any("locked_test" in v and "train" in v
                   for v in err.value.violations if "R4" in v)

    def test_b9_lineage_snapshot_unknown_split_role_rejected(self):
        snap = TrainingSnapshot(
            dataset_id="ds-train",
            snapshot_version="2026-01",
            split_role="mystery",
            locked=True,
            checksum_sha256="a" * 64,
            source_class="public",
            patient_disjoint_asserted=True,
        )
        with pytest.raises(CanonicalValidationError) as err:
            validate_dataset_role_rules(self._with_lineage_snapshot(snap))
        assert any("R0a" in v and "lineage" in v for v in err.value.violations)

    def test_b9_lineage_snapshot_locked_test_split_role_rejected(self):
        snap = TrainingSnapshot(
            dataset_id="ds-train",
            snapshot_version="2026-01",
            split_role="locked_test",
            locked=True,
            checksum_sha256="a" * 64,
            source_class="public",
            patient_disjoint_asserted=True,
        )
        with pytest.raises(CanonicalValidationError) as err:
            validate_dataset_role_rules(self._with_lineage_snapshot(snap))
        assert any("R1" in v and "lineage" in v for v in err.value.violations)

    def test_b9_lineage_snapshot_conflicting_with_training_ref_rejected(self):
        snap = TrainingSnapshot(
            dataset_id="ds-train",  # same snapshot as the default training ref...
            snapshot_version="2026-01",
            split_role="calibration",  # ...but under a different role -> R4.
            locked=True,
            checksum_sha256="a" * 64,
            source_class="public",
            patient_disjoint_asserted=True,
        )
        with pytest.raises(CanonicalValidationError) as err:
            validate_dataset_role_rules(self._with_lineage_snapshot(snap))
        assert any("R4" in v for v in err.value.violations)

    def test_b9_identical_roles_across_surfaces_do_not_conflict(self):
        # Lineage snapshot repeating the training ref's role is consistent.
        record = self._record()
        assert record.data_policy.training_data_refs[0].split_role == "train"
        assert record.lineage.training_snapshot is not None
        assert record.lineage.training_snapshot.split_role == "train"
        validate_dataset_role_rules(record)

    @staticmethod
    def _binding_kwargs_without(source_class: bool, **overrides: Any) -> dict[str, Any]:
        base = make_training_ref()
        kwargs: dict[str, Any] = {
            f.name: getattr(base, f.name) for f in dataclasses.fields(DatasetBinding)
        }
        if not source_class:
            del kwargs["source_class"]
        kwargs.update(overrides)
        return kwargs

    def test_b9_training_ref_omitting_source_class_is_type_error(self):
        # r5 attempt 3: no silent "public" default - omission fails closed.
        with pytest.raises(TypeError):
            DatasetBinding(**self._binding_kwargs_without(source_class=False))

    def test_b9_benchmark_ref_omitting_source_class_is_type_error(self):
        with pytest.raises(TypeError):
            DatasetBinding(
                **self._binding_kwargs_without(
                    source_class=False,
                    split_role="locked_test",
                )
            )


class TestStructuralValidation:
    def test_non_kebab_model_id_rejected(self):
        bad = make_record(model_id="Coronary Seg!")
        with pytest.raises(CanonicalValidationError) as err:
            validate_registry_record(bad)
        assert any("kebab" in v for v in err.value.violations)

    def test_champion_requires_promotion_block(self):
        record = dataclasses.replace(
            make_record(),
            status=ModelStatus.CHAMPION,
            champion_of=ChampionOf(
                task="coronary_segmentation", lane=ReleaseLane.PERSONAL_RESEARCH
            ),
            promotion=None,
        )
        with pytest.raises(CanonicalValidationError) as err:
            validate_registry_record(record)
        assert any("CHAMPION" in v for v in err.value.violations)

    def test_ensemble_requires_block_and_members(self):
        no_block = make_record(kind="ensemble")
        with pytest.raises(CanonicalValidationError) as err:
            validate_registry_record(no_block)
        assert any("ensemble block" in v for v in err.value.violations)

        empty = make_ensemble_members_case(EnsembleBlock(members=(), combination_strategy="x"))
        with pytest.raises(CanonicalValidationError) as err:
            validate_registry_record(empty)
        assert any("members" in v for v in err.value.violations)

    def test_owner_gate_v2_completeness_enforced(self):
        weak_gate = OwnerGateV2(
            required=True,
            status="pending",  # not approved
            approver="owner",
            approved_at=AT,
            evidence_path="reports/x.md",
            evidence_sha256="a" * 64,
        )
        promotion = PromotionBlock(
            bundle_path="/tmp/bundle.json",
            bundle_sha256="b" * 64,
            contract_sha256="c" * 64,
            permit_sha256="d" * 64,
            envelope_sha256="e" * 64,
            accepted_sha="f" * 40,
            rollback_target="coronary-seg@1.1.0",
            owner_gate=weak_gate,
        )
        record = dataclasses.replace(
            make_record(),
            status=ModelStatus.CHAMPION,
            champion_of=ChampionOf(
                task="coronary_segmentation", lane=ReleaseLane.PERSONAL_RESEARCH
            ),
            promotion=promotion,
        )
        with pytest.raises(CanonicalValidationError) as err:
            validate_registry_record(record)
        assert any("owner_gate" in v for v in err.value.violations)


class TestLoadValidation:
    """Luna R3 fix #2: the loading path validates exactly like registration."""

    def test_tampered_sealed_record_fails_closed(self):
        payload = record_to_dict(seal(make_record()))
        assert isinstance(payload["status"], str)  # enum-free dump
        payload["weights"]["sha256"] = "f" * 64
        with pytest.raises(CanonicalValidationError):
            record_from_dict(payload)

    def test_incomplete_record_rejected_at_load(self):
        payload = record_to_dict(make_record())
        del payload["validation"]["benchmark_ref"]
        with pytest.raises(CanonicalValidationError) as err:
            record_from_dict(payload)
        assert any("benchmark_ref" in v for v in err.value.violations)

    def test_incomplete_record_missing_benchmark_none_rejected(self):
        payload = record_to_dict(make_record())
        payload["validation"]["benchmark_ref"] = None
        with pytest.raises(CanonicalValidationError) as err:
            record_from_dict(payload)
        assert any("R3" in v for v in err.value.violations)

    def test_malformed_digest_rejected_at_load(self):
        payload = record_to_dict(make_record())
        payload["source"]["commit"] = "nothex"
        with pytest.raises(CanonicalValidationError):
            record_from_dict(payload)

    def test_sealed_record_loads_equal(self):
        sealed = seal(make_record())
        restored = record_from_dict(record_to_dict(sealed))
        assert restored == sealed

    def test_persistence_loader_requires_sealed_digest(self):
        with pytest.raises(CanonicalValidationError) as err:
            load_registry_record(record_to_dict(make_record()))  # digest == ""
        assert any("sealed" in v for v in err.value.violations)

    def test_persistence_loader_accepts_sealed_record(self):
        sealed = seal(make_record())
        assert load_registry_record(record_to_dict(sealed)) == sealed


class TestB10SealedRecordDigestValidation:
    """Blocker b10: malformed digests on sealed records REJECT at load -
    validation is orthogonal to release_digest coverage (acceptance_ref
    stays transient/I12 but is still structurally validated)."""

    @staticmethod
    def _sealed_payload() -> dict[str, Any]:
        return record_to_dict(seal(make_record()))

    def _make_acceptance_ref(self) -> AcceptanceRef:
        return AcceptanceRef(
            contract_sha256="c" * 64,
            permit_sha256="d" * 64,
            envelope_sha256="e" * 64,
            bundle_sha256="b" * 64,
            accepted_sha="f" * 40,
        )

    def test_b10_performance_sha256_not_64hex_rejected_at_load(self):
        payload = self._sealed_payload()
        payload["validation"]["performance_sha256"] = "x"
        with pytest.raises(CanonicalValidationError) as err:
            record_from_dict(payload)
        assert any("performance_sha256" in v for v in err.value.violations)

    def test_b10_malformed_performance_sha_rejected_by_direct_validation(self):
        record = make_record()
        bad = dataclasses.replace(
            record,
            validation=dataclasses.replace(record.validation, performance_sha256="x"),
        )
        with pytest.raises(CanonicalValidationError) as err:
            validate_registry_record(bad)
        assert any("performance_sha256" in v for v in err.value.violations)

    def test_b10_benchmark_checksum_not_64hex_rejected_at_load(self):
        payload = self._sealed_payload()
        payload["validation"]["benchmark_ref"]["checksum_sha256"] = "x" * 63
        with pytest.raises(CanonicalValidationError) as err:
            record_from_dict(payload)
        assert any(
            "benchmark_ref.checksum_sha256" in v for v in err.value.violations
        )

    def test_b10_lineage_snapshot_checksum_not_64hex_rejected_at_load(self):
        payload = self._sealed_payload()
        payload["lineage"]["training_snapshot"]["checksum_sha256"] = "x"
        with pytest.raises(CanonicalValidationError) as err:
            record_from_dict(payload)
        assert any("training_snapshot.checksum_sha256" in v for v in err.value.violations)

    @pytest.mark.parametrize(
        ("section", "key"),
        [
            ("validation.performance_sha256", None),
            ("validation.benchmark_ref", "checksum_sha256"),
            ("lineage.training_snapshot", "checksum_sha256"),
        ],
    )
    def test_b10_empty_digest_strings_rejected_at_load(
        self, section: str, key: str | None
    ):
        # The classic silent omission: "" must not pass as a digest.
        payload = self._sealed_payload()
        target: dict[str, Any] = payload
        for part in section.split(".")[:-1]:
            target = target[part]
        leaf = section.split(".")[-1]
        if key is None:
            target[leaf] = ""
        else:
            target[leaf][key] = ""
        with pytest.raises(CanonicalValidationError):
            record_from_dict(payload)

    @pytest.mark.parametrize(
        "field_name",
        ["contract_sha256", "permit_sha256", "envelope_sha256", "bundle_sha256"],
    )
    def test_b10_acceptance_ref_digests_rejected_at_load(self, field_name: str):
        payload = self._sealed_payload()
        payload["acceptance_ref"] = dataclasses.asdict(self._make_acceptance_ref())
        payload["acceptance_ref"][field_name] = "not-a-digest"
        with pytest.raises(CanonicalValidationError) as err:
            record_from_dict(payload)
        assert any(f"acceptance_ref.{field_name}" in v for v in err.value.violations)

    def test_b10_acceptance_ref_accepted_sha_not_git_sha_rejected_at_load(self):
        payload = self._sealed_payload()
        payload["acceptance_ref"] = dataclasses.asdict(self._make_acceptance_ref())
        payload["acceptance_ref"]["accepted_sha"] = "z" * 40
        with pytest.raises(CanonicalValidationError) as err:
            record_from_dict(payload)
        assert any("accepted_sha" in v for v in err.value.violations)

    def test_b10_garbage_acceptance_ref_rejected_even_though_digest_excluded(self):
        # acceptance_ref sits in TRANSIENT_RECORD_FIELDS (I12), so sealing is
        # unaffected - the rejection must come from structural validation.
        record = dataclasses.replace(
            make_record(),
            acceptance_ref=AcceptanceRef(
                contract_sha256="garbage",
                permit_sha256="garbage",
                envelope_sha256="garbage",
                bundle_sha256="garbage",
                accepted_sha="nope",
            ),
        )
        sealed = seal(record)  # digest computable: acceptance_ref excluded
        verify_release_digest(sealed)  # ...and it verifies.
        with pytest.raises(CanonicalValidationError) as err:
            record_from_dict(record_to_dict(sealed))
        assert any(v.startswith("acceptance_ref.") for v in err.value.violations)

    def test_b10_well_formed_acceptance_ref_still_loads(self):
        record = dataclasses.replace(
            make_record(), acceptance_ref=self._make_acceptance_ref()
        )
        sealed = seal(record)
        restored = record_from_dict(record_to_dict(sealed))
        assert restored == sealed

    def test_b10_well_formed_digest_paths_all_load(self):
        # Positive control covering every path b10 validates: performance
        # digest, benchmark + lineage snapshot checksums, acceptance chain.
        payload = record_to_dict(
            seal(dataclasses.replace(make_record(), acceptance_ref=self._make_acceptance_ref()))
        )
        assert isinstance(payload["validation"]["performance_sha256"], str)
        assert len(payload["validation"]["benchmark_ref"]["checksum_sha256"]) == 64
        assert len(payload["lineage"]["training_snapshot"]["checksum_sha256"]) == 64
        restored = record_from_dict(payload)
        assert restored.validation.performance_sha256 == sha("perf-report")
        assert restored.acceptance_ref is not None
        assert restored.acceptance_ref.accepted_sha == "f" * 40


class TestJsonSerialization:
    """Luna R3 fix #3: dumps never see a raw Enum; reload is lossless."""

    def test_json_dumps_of_projected_state_succeeds_and_reloads_equal(self):
        sealed = seal(make_record(status=ModelStatus.CHALLENGER))
        dumped = json.dumps(record_to_dict(sealed))
        reloaded = record_from_dict(json.loads(dumped))
        assert reloaded == sealed
        assert reloaded.status == ModelStatus.CHALLENGER
        assert reloaded.intended_use.release_lane == ReleaseLane.PERSONAL_RESEARCH

    def test_no_enum_objects_in_dump(self):
        def _walk(node: object) -> None:
            if isinstance(node, dict):
                for value in node.values():
                    _walk(value)
            elif isinstance(node, (list, tuple)):
                for item in node:
                    _walk(item)
            else:
                assert not isinstance(node, ModelStatus)
                assert not isinstance(node, ReleaseLane)

        _walk(record_to_dict(make_record()))


class TestPromotionBlockValidation:
    """Luna R3 fix #6: structured owner gate + CSI-001 chain digests."""

    def _promotion(self, **overrides: Any) -> PromotionBlock:
        kwargs: dict[str, Any] = dict(
            bundle_path="/tmp/bundle.json",
            bundle_sha256="b" * 64,
            contract_sha256="c" * 64,
            permit_sha256="d" * 64,
            envelope_sha256="e" * 64,
            accepted_sha="f" * 40,
            rollback_target="coronary-seg@1.1.0",
            owner_gate=OwnerGateV2(
                required=True,
                status="approved",
                approver="owner",
                approved_at=AT,
                evidence_path="reports/owner-gate.md",
                evidence_sha256="a" * 64,
            ),
        )
        kwargs.update(overrides)
        return PromotionBlock(**kwargs)

    def test_valid_block_passes(self):
        validate_promotion_block(self._promotion())

    @pytest.mark.parametrize(
        ("field_name", "value"),
        [
            ("required", False),
            ("status", "pending"),
            ("approver", ""),
            ("approved_at", ""),
            ("evidence_path", ""),
            ("evidence_sha256", "xyz"),
        ],
    )
    def test_missing_or_malformed_gate_rejected(self, field_name: str, value: object):
        gate = dataclasses.replace(self._promotion().owner_gate, **{field_name: value})  # type: ignore[arg-type]
        with pytest.raises(CanonicalValidationError) as err:
            validate_promotion_block(self._promotion(owner_gate=gate))
        assert any(field_name in v for v in err.value.violations)

    @pytest.mark.parametrize(
        "field_name",
        ["bundle_sha256", "contract_sha256", "permit_sha256", "envelope_sha256"],
    )
    def test_malformed_chain_digest_rejected(self, field_name: str):
        with pytest.raises(CanonicalValidationError) as err:
            validate_promotion_block(self._promotion(**{field_name: "0" * 40}))
        assert any(field_name in v for v in err.value.violations)

    def test_bad_accepted_sha_rejected(self):
        with pytest.raises(CanonicalValidationError) as err:
            validate_promotion_block(self._promotion(accepted_sha="z" * 40))
        assert any("accepted_sha" in v for v in err.value.violations)

    def test_missing_rollback_target_rejected(self):
        with pytest.raises(CanonicalValidationError) as err:
            validate_promotion_block(self._promotion(rollback_target=""))
        assert any("rollback_target" in v for v in err.value.violations)


def make_ensemble_members_case(ensemble: EnsembleBlock):
    return make_record(
        model_id="seg-ensemble",
        kind="ensemble",
        ensemble=ensemble,
    )
