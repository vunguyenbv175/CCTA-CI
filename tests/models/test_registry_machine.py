"""Champion/challenger machine: intake, promotion chain, BLOCKED, ensembles."""

from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import replace
from pathlib import Path

import pytest

from core.enums import ReleaseLane
from core.errors import CanonicalValidationError, DigestMismatchError, StateTransitionError
from core.events import EventLog
from models.registry import BundleAcceptor, ModelRegistry, model_ref
from models.registry.record import (
    ChampionOf,
    ModelStatus,
    OwnerGateV2,
    PromotionBlock,
    RegistryRecord,
    verify_release_digest,
)
from tests.models.factories import (
    AT,
    WEIGHTS_BYTES,
    make_ensemble_record,
    make_record,
    weights_sha,
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_bundle(
    tmp_path: Path,
    *,
    task_id: str = "coronary_segmentation",
    review_tree: list[str] | None = None,
) -> tuple[str, str]:
    """Write a plausible acceptance bundle; returns (path, digest)."""
    tree = review_tree if review_tree is not None else [
        weights_sha(),
        "preproc-pipeline@2.0",
    ]
    doc = {"task_id": task_id, "shas": {"review_target_sha": tree}}
    target = tmp_path / f"bundle-{abs(hash(doc['task_id'])) % 100000}-{len(tree)}.json"
    target.write_text(json.dumps(doc), encoding="utf-8")
    return str(target), sha256_bytes(target.read_bytes())


def make_promotion(tmp_path: Path, **overrides: object) -> PromotionBlock:
    path, digest = write_bundle(tmp_path)
    kwargs: dict[str, object] = dict(
        bundle_path=path,
        bundle_sha256=digest,
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
    return PromotionBlock(**kwargs)  # type: ignore[arg-type]


def accepting_bundle(_: str, __: str, ___: str, ____: str) -> bool:
    return True


def make_registry(acceptor: BundleAcceptor | None = accepting_bundle):
    log = EventLog()
    return ModelRegistry(log=log, accept_bundle=acceptor)


def make_challenger(
    acceptor: BundleAcceptor | None = accepting_bundle,
    version: str = "1.2.0",
) -> tuple[ModelRegistry, str]:
    """Registered + intake-passed challenger ready for promotion."""
    registry = ModelRegistry(log=EventLog(), accept_bundle=acceptor)
    record = make_record(model_version=version)
    registry.register(record)
    ref = model_ref(record.model_id, version)
    registry.pass_intake(
        ref,
        weights_bytes=WEIGHTS_BYTES,
        claim_statuses={"CLM-SEG-001": "threshold_pending"},
    )
    return registry, ref


class TestRegistrationAndIntake:
    def test_register_seals_experimental_with_digest(self):
        registry = make_registry()
        sealed = registry.register(make_record())
        ref = model_ref("coronary-seg", "1.2.0")
        assert sealed.release_digest
        assert registry.project_status(ref) == ModelStatus.EXPERIMENTAL
        assert len(sealed.status_history) == 0

    def test_duplicate_registration_rejected(self):
        registry = make_registry()
        registry.register(make_record())
        with pytest.raises(CanonicalValidationError):
            registry.register(make_record())

    def test_registration_with_promotion_preseeded_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            promotion = make_promotion(Path(td))
            record = dataclasses_replace_promoted(make_record(), promotion)
            with pytest.raises(CanonicalValidationError):
                make_registry().register(record)

    def test_intake_happy_path(self):
        registry = make_registry()
        registry.register(make_record())
        ref = model_ref("coronary-seg", "1.2.0")
        updated = registry.pass_intake(
            ref, weights_bytes=WEIGHTS_BYTES, claim_statuses={"CLM-SEG-001": "threshold_pending"}
        )
        assert updated.status == ModelStatus.CHALLENGER
        assert registry.project_status(ref) == ModelStatus.CHALLENGER

    def test_intake_weight_checksum_enforced(self):
        registry = make_registry()
        registry.register(make_record())
        with pytest.raises(CanonicalValidationError) as err:
            registry.pass_intake(
                model_ref("coronary-seg", "1.2.0"),
                weights_bytes=b"tampered",
                claim_statuses={"CLM-SEG-001": "threshold_pending"},
            )
        assert any("weights checksum" in v for v in err.value.violations)

    def test_intake_claim_floor_enforced(self):
        registry = make_registry()
        registry.register(make_record())
        with pytest.raises(CanonicalValidationError):
            registry.pass_intake(
                model_ref("coronary-seg", "1.2.0"),
                weights_bytes=WEIGHTS_BYTES,
                claim_statuses={"CLM-SEG-001": "draft"},
            )

    def test_intake_unknown_claim_fails_closed(self):
        registry = make_registry()
        registry.register(make_record())
        with pytest.raises(CanonicalValidationError):
            registry.pass_intake(
                model_ref("coronary-seg", "1.2.0"),
                weights_bytes=WEIGHTS_BYTES,
                claim_statuses={},
            )

    def test_intake_clinical_lane_requires_local_data_policy(self):
        import dataclasses

        from core.enums import ReleaseLane

        base = make_record()
        intended = dataclasses.replace(
            base.intended_use, release_lane=ReleaseLane.CLINICAL_CANDIDATE
        )
        policy = dataclasses.replace(base.data_policy, remote_processing=True)
        record = dataclasses.replace(base, intended_use=intended, data_policy=policy)
        registry = make_registry()
        registry.register(record)
        with pytest.raises(CanonicalValidationError) as err:
            registry.pass_intake(
                model_ref("coronary-seg", "1.2.0"),
                weights_bytes=WEIGHTS_BYTES,
                claim_statuses={"CLM-SEG-001": "threshold_pending"},
            )
        assert any("data policy" in v for v in err.value.violations)

    def test_intake_license_review_required(self):
        import dataclasses

        base = make_record()
        license_block = dataclasses.replace(base.license, review_ref="")
        record = dataclasses.replace(base, license=license_block)
        registry = make_registry()
        registry.register(record)
        with pytest.raises(CanonicalValidationError):
            registry.pass_intake(
                model_ref("coronary-seg", "1.2.0"),
                weights_bytes=WEIGHTS_BYTES,
                claim_statuses={"CLM-SEG-001": "threshold_pending"},
            )


class TestPromotionChain:
    def _challenger(self, version: str = "1.2.0") -> tuple[ModelRegistry, str]:
        return make_challenger(version=version)

    def test_promotion_happy_path_binds_full_chain(self, tmp_path):
        registry, ref = self._challenger()
        promoted = registry.promote(ref, make_promotion(tmp_path))
        assert promoted.status == ModelStatus.CHAMPION
        assert promoted.acceptance_ref is not None
        assert promoted.acceptance_ref.bundle_sha256
        assert promoted.champion_of is not None
        assert promoted.champion_of.task == "coronary_segmentation"
        assert registry.project_status(ref) == ModelStatus.CHAMPION
        assert registry.champion_of("coronary_segmentation", __import__(
            "core.enums", fromlist=["ReleaseLane"]
        ).ReleaseLane.PERSONAL_RESEARCH) == ref
        # rollback target recorded
        assert promoted.promotion is not None
        assert promoted.promotion.rollback_target == "coronary-seg@1.1.0"

    def test_step1_bundle_bytes_digest_mismatch(self, tmp_path):
        registry, ref = self._challenger()
        promotion = make_promotion(tmp_path)
        tampered = dataclasses_replace_promotion(promotion, bundle_sha256="0" * 64)
        with pytest.raises(DigestMismatchError):
            registry.promote(ref, tampered)

    def test_step2_governance_accept_must_pass(self, tmp_path):
        def rejecting_acceptor(*_: object) -> bool:
            return False

        registry, ref = make_challenger(acceptor=rejecting_acceptor)
        with pytest.raises(StateTransitionError):
            registry.promote(ref, make_promotion(tmp_path))

    def test_step2_requires_injected_acceptor(self, tmp_path):
        registry, ref = make_challenger(acceptor=None)
        with pytest.raises(CanonicalValidationError):
            registry.promote(ref, make_promotion(tmp_path))

    def test_step3_task_mismatch_rejected(self, tmp_path):
        registry, ref = self._challenger()
        promotion = make_promotion(tmp_path)
        # rewrite bundle to a different task while keeping recorded digest valid
        doc = json.loads(Path(promotion.bundle_path).read_text(encoding="utf-8"))
        doc["task_id"] = "calcium_scoring"
        path = Path(promotion.bundle_path)
        path.write_text(json.dumps(doc), encoding="utf-8")
        rebound = dataclasses_replace_promotion(
            promotion,
            bundle_sha256=sha256_bytes(path.read_bytes()),
        )
        with pytest.raises(StateTransitionError) as err:
            registry.promote(ref, rebound)
        assert "step 3" in str(err.value)

    def test_step4_artifacts_must_be_bound(self, tmp_path):
        registry, ref = self._challenger()
        promotion = make_promotion(tmp_path)
        doc = json.loads(Path(promotion.bundle_path).read_text(encoding="utf-8"))
        doc["shas"]["review_target_sha"] = ["unrelated-sha"]
        path = Path(promotion.bundle_path)
        path.write_text(json.dumps(doc), encoding="utf-8")
        rebound = dataclasses_replace_promotion(
            promotion, bundle_sha256=sha256_bytes(path.read_bytes())
        )
        with pytest.raises(StateTransitionError) as err:
            registry.promote(ref, rebound)
        assert "step 4" in str(err.value)

    def test_experimental_cannot_skip_to_champion(self, tmp_path):
        registry = make_registry()
        registry.register(make_record())
        with pytest.raises(StateTransitionError):
            registry.promote(
                model_ref("coronary-seg", "1.2.0"), make_promotion(tmp_path)
            )


class TestChampionUniquenessAndDeprecation:
    def test_new_champion_auto_deprecates_previous(self, tmp_path):
        from core.enums import ReleaseLane

        registry, first_ref = make_challenger(version="1.0.0")
        registry.promote(first_ref, make_promotion(tmp_path))

        challenger = make_record(model_version="2.0.0")
        registry.register(challenger)
        second_ref = model_ref("coronary-seg", "2.0.0")
        registry.pass_intake(
            second_ref,
            weights_bytes=WEIGHTS_BYTES,
            claim_statuses={"CLM-SEG-001": "threshold_pending"},
        )
        registry.promote(second_ref, make_promotion(tmp_path))

        lane_key = ("coronary_segmentation", ReleaseLane.PERSONAL_RESEARCH)
        assert registry.champion_of(*lane_key) == second_ref
        assert registry.project_status(first_ref) == ModelStatus.DEPRECATED
        deprecated_events = [
            e for e in registry.event_log.events if e.event_type == "DEPRECATED"
        ]
        assert any(e.payload.get("superseded_by") == second_ref for e in deprecated_events)

    def test_deprecated_cannot_be_repromoted_directly(self, tmp_path):
        registry, first_ref = make_challenger(version="1.0.0")
        registry.promote(first_ref, make_promotion(tmp_path))
        registry.deprecate(first_ref, reason="replaced")
        with pytest.raises(StateTransitionError):
            registry.promote(first_ref, make_promotion(tmp_path))


class TestBlockedFirstClass:
    def test_blocked_challenger_never_converts(self, tmp_path):
        registry, ref = make_challenger()
        registry.block(ref, reason="contamination found")
        assert registry.project_status(ref) == ModelStatus.BLOCKED
        with pytest.raises(StateTransitionError):
            registry.promote(ref, make_promotion(tmp_path))

    def test_blocking_requires_reason(self):
        registry, ref = make_challenger()
        with pytest.raises(CanonicalValidationError):
            registry.block(ref, reason="")


class TestEnsembleFailClosed:
    def test_member_deprecation_blocks_ensemble(self, tmp_path):
        registry = make_registry()
        member = make_record()
        ensemble = make_ensemble_record(members=(model_ref("coronary-seg", "1.2.0"),))
        registry.register(member)
        registry.register(ensemble)
        ens_ref = model_ref("seg-ensemble", "1.0.0")

        registry.deprecate(model_ref("coronary-seg", "1.2.0"), reason="weights pulled")
        assert registry.project_status(ens_ref) == ModelStatus.BLOCKED

    def test_member_blocked_blocks_ensemble(self):
        registry = make_registry()
        member = make_record()
        ensemble = make_ensemble_record(members=(model_ref("coronary-seg", "1.2.0"),))
        registry.register(member)
        registry.register(ensemble)
        registry.block(model_ref("coronary-seg", "1.2.0"), reason="failed QC audit")
        assert (
            registry.project_status(model_ref("seg-ensemble", "1.0.0")) == ModelStatus.BLOCKED
        )


# ---------------------------------------------------------------------------
# small local helpers (dataclasses.replace wrappers for typing clarity)
# ---------------------------------------------------------------------------


def dataclasses_replace_promoted(
    record: RegistryRecord, promotion: PromotionBlock
) -> RegistryRecord:
    return replace(
        record,
        status=ModelStatus.CHAMPION,
        champion_of=ChampionOf(task="coronary_segmentation", lane=ReleaseLane.PERSONAL_RESEARCH),
        promotion=promotion,
    )


def dataclasses_replace_promotion(promotion: PromotionBlock, **changes: object) -> PromotionBlock:
    return replace(promotion, **changes)  # type: ignore[arg-type]


def test_os_level_bundle_tamper_detected(tmp_path):
    """Belt-and-braces: bundle edited after digest recording fails step 1."""
    registry, ref = make_challenger()
    promotion = make_promotion(tmp_path)
    path = Path(promotion.bundle_path)
    original = path.read_text(encoding="utf-8")
    path.write_text(original + " ", encoding="utf-8")  # os-level edit post-digest
    with pytest.raises(DigestMismatchError):
        registry.promote(ref, promotion)


class TestReleaseDigestStability:
    """Luna R3 fix #4 (I12/I13): transitions never invalidate release_digest."""

    def test_intake_transition_keeps_release_digest(self):
        registry = make_registry()
        sealed = registry.register(make_record())
        ref = model_ref("coronary-seg", "1.2.0")
        challenger = registry.pass_intake(
            ref,
            weights_bytes=WEIGHTS_BYTES,
            claim_statuses={"CLM-SEG-001": "threshold_pending"},
        )
        assert challenger.status_history  # history moved
        assert challenger.release_digest == sealed.release_digest
        verify_release_digest(challenger)  # does not raise

    def test_full_lifecycle_keeps_release_digest(self, tmp_path):
        registry, ref = make_challenger()
        sealed = registry.get(ref)
        registry.promote(ref, make_promotion(tmp_path))
        assert registry.get(ref).release_digest == sealed.release_digest
        verify_release_digest(registry.get(ref))

    def test_deprecation_and_block_keep_release_digest(self):
        registry, ref = make_challenger()
        sealed = registry.get(ref)
        registry.block(ref, reason="contamination")
        verify_release_digest(registry.get(ref))
        assert registry.get(ref).release_digest == sealed.release_digest

        registry2, ref2 = make_challenger(version="9.9.9")
        sealed2 = registry2.get(ref2)
        registry2.deprecate(ref2, reason="obsolete")
        verify_release_digest(registry2.get(ref2))
        assert registry2.get(ref2).release_digest == sealed2.release_digest

    def test_identity_mutation_changes_release_digest(self):
        from models.registry.record import compute_release_digest

        registry, ref = make_challenger()
        frozen = registry.get(ref)
        original = frozen.release_digest
        tampered_identity = replace(
            frozen,
            input_contract=replace(frozen.input_contract, preprocessing="preproc-pipeline@3.0"),
        )
        assert compute_release_digest(tampered_identity) != original


class TestTerminalStateReEntry:
    """Luna R3 fix #5: DEPRECATED/BLOCKED can never re-enter CHALLENGER."""

    def test_deprecated_cannot_reenter_challenger_via_intake(self):
        registry = make_registry()
        registry.register(make_record())
        ref = model_ref("coronary-seg", "1.2.0")
        registry.deprecate(ref, reason="superseded design")
        with pytest.raises(StateTransitionError) as err:
            registry.pass_intake(
                ref,
                weights_bytes=WEIGHTS_BYTES,
                claim_statuses={"CLM-SEG-001": "threshold_pending"},
            )
        assert "re-enter" in str(err.value)
        assert registry.project_status(ref) == ModelStatus.DEPRECATED

    def test_blocked_cannot_reenter_challenger_via_intake(self):
        registry = make_registry()
        registry.register(make_record())
        ref = model_ref("coronary-seg", "1.2.0")
        registry.block(ref, reason="failed QC audit")
        with pytest.raises(StateTransitionError) as err:
            registry.pass_intake(
                ref,
                weights_bytes=WEIGHTS_BYTES,
                claim_statuses={"CLM-SEG-001": "threshold_pending"},
            )
        assert "re-enter" in str(err.value)
        assert registry.project_status(ref) == ModelStatus.BLOCKED


class TestPromotionEventValidation:
    """Luna R3 fix #6: gate + chain digests validated BEFORE anything applies."""

    def _challenger_with_bad_promotion(self, tmp_path: Path, **overrides: object):
        registry, ref = make_challenger()
        kwargs: dict[str, object] = dict(
            bundle_path=str(tmp_path / "bundle.json"),
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
        return registry, ref, PromotionBlock(**kwargs)  # type: ignore[arg-type]

    def _assert_nothing_applied(self, registry: ModelRegistry, ref: str) -> None:
        assert registry.project_status(ref) == ModelStatus.CHALLENGER
        last_type = registry.event_log.history_for(
            ref, entity="model_registry"
        )[-1].event_type
        assert last_type == "INTAKE_PASSED"

    def test_unapproved_gate_rejected_before_applying(self, tmp_path):
        bad_gate = OwnerGateV2(
            required=True,
            status="pending",
            approver="owner",
            approved_at=AT,
            evidence_path="reports/owner-gate.md",
            evidence_sha256="a" * 64,
        )
        registry, ref, promotion = self._challenger_with_bad_promotion(
            tmp_path, owner_gate=bad_gate
        )
        with pytest.raises(CanonicalValidationError):
            registry.promote(ref, promotion)
        self._assert_nothing_applied(registry, ref)

    def test_optional_gate_rejected(self, tmp_path):
        optional_gate = OwnerGateV2(
            required=False,
            status="approved",
            approver="owner",
            approved_at=AT,
            evidence_path="reports/owner-gate.md",
            evidence_sha256="a" * 64,
        )
        registry, ref, promotion = self._challenger_with_bad_promotion(
            tmp_path, owner_gate=optional_gate
        )
        with pytest.raises(CanonicalValidationError) as err:
            registry.promote(ref, promotion)
        assert any("required must be true" in v for v in err.value.violations)
        self._assert_nothing_applied(registry, ref)

    def test_gate_evidence_without_digest_rejected(self, tmp_path):
        unevidenced = OwnerGateV2(
            required=True,
            status="approved",
            approver="owner",
            approved_at=AT,
            evidence_path="reports/owner-gate.md",
            evidence_sha256="deadbeef",
        )
        registry, ref, promotion = self._challenger_with_bad_promotion(
            tmp_path, owner_gate=unevidenced
        )
        with pytest.raises(CanonicalValidationError):
            registry.promote(ref, promotion)
        self._assert_nothing_applied(registry, ref)

    @pytest.mark.parametrize(
        ("field_name", "value"),
        [
            ("contract_sha256", "short"),
            ("permit_sha256", "g" * 64),
            ("envelope_sha256", "0" * 40),
            ("accepted_sha", "f" * 64),  # sha256 length is not a git SHA
        ],
    )
    def test_malformed_chain_binding_rejected(
        self, tmp_path, field_name: str, value: str
    ):
        registry, ref, promotion = self._challenger_with_bad_promotion(
            tmp_path, **{field_name: value}
        )
        with pytest.raises(CanonicalValidationError) as err:
            registry.promote(ref, promotion)
        assert any(field_name in v for v in err.value.violations)
        self._assert_nothing_applied(registry, ref)

    def test_gate_validation_precedes_bundle_io(self, tmp_path):
        # bundle_path points nowhere; step-0 gate failure must fire first.
        missing_gate = OwnerGateV2(
            required=True,
            status="approved",
            approver="",  # missing approver
            approved_at=AT,
            evidence_path="reports/owner-gate.md",
            evidence_sha256="a" * 64,
        )
        registry, ref, promotion = self._challenger_with_bad_promotion(
            tmp_path, owner_gate=missing_gate
        )
        with pytest.raises(CanonicalValidationError) as err:
            registry.promote(ref, promotion)
        assert any("approver" in v for v in err.value.violations)
        self._assert_nothing_applied(registry, ref)
