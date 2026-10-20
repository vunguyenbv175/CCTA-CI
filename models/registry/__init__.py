"""Champion/challenger registry machine (WS00_DESIGN.md Part C §4–§7).

Status lives in an append-only event log; the frozen release record is never
mutated (I13). Promotion re-verifies the full CSI-001 acceptance chain in the
mandated order, fails closed on any gap, and enforces at most one CHAMPION
per ``(task, lane)``. BLOCKED is first-class and never converts.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import fields as dc_fields
from pathlib import Path

from core.enums import ModelStatus, ReleaseLane
from core.errors import (
    CanonicalValidationError,
    DigestMismatchError,
    StateTransitionError,
)
from core.events import EventLog, StateEvent, utc_now_iso

from .record import (
    AcceptanceRef,
    ChampionOf,
    PromotionBlock,
    RegistryRecord,
    StatusHistoryEntry,
    compute_release_digest,
    intake_binding_presence_problems,
    validate_promotion_block,
    validate_registry_record,
)

#: Callable that re-runs ``python -m orchestration.governance accept`` for a
#: bundle with its pinned contract/permit/envelope digests. The product code
#: must not import orchestration directly (boundary discipline), so the
#: acceptor is injected by the caller.
BundleAcceptor = Callable[[str, str, str, str], bool]

_CLAIM_STATUS_FLOOR = "threshold_pending"
_CLAIM_STATUS_ORDER = ("draft", "threshold_pending", "evidenced", "locked")

#: Statuses that can never re-enter CHALLENGER through intake (Luna R3 fix #5).
_TERMINAL_FOR_INTAKE = frozenset({ModelStatus.DEPRECATED, ModelStatus.BLOCKED})


def model_ref(model_id: str, version: str) -> str:
    """Canonical ``model_id@version`` reference."""
    return f"{model_id}@{version}"


class ModelRegistry:
    """Registry store + deterministic state machine over the event log."""

    def __init__(self, log: EventLog, accept_bundle: BundleAcceptor | None = None) -> None:
        self._log = log
        self._accept_bundle = accept_bundle
        self._records: dict[str, RegistryRecord] = {}
        self._champions: dict[tuple[str, ReleaseLane], str] = {}
        self._ensembles: list[str] = []

    # -- read access -----------------------------------------------------------

    def get(self, ref: str) -> RegistryRecord:
        try:
            return self._records[ref]
        except KeyError:
            raise KeyError(f"unknown registry record {ref!r}") from None

    def __contains__(self, ref: object) -> bool:
        return isinstance(ref, str) and ref in self._records

    @property
    def event_log(self) -> EventLog:
        return self._log

    def champion_of(self, task: str, lane: ReleaseLane) -> str | None:
        return self._champions.get((task, lane))

    def project_status(self, ref: str) -> ModelStatus:
        projection = self._log.project_state(ref, entity="model_registry")
        return ModelStatus(projection.get("registry_status", "EXPERIMENTAL"))

    # -- registration ------------------------------------------------------------

    def register(self, record: RegistryRecord) -> RegistryRecord:
        """Freeze + register a new EXPERIMENTAL release (closed schema)."""
        validate_registry_record(record)
        ref = model_ref(record.model_id, record.model_version)
        if ref in self._records:
            raise CanonicalValidationError(
                f"{ref} already registered; corrections are new versions"
            )
        if record.status != ModelStatus.EXPERIMENTAL or record.promotion is not None:
            raise CanonicalValidationError(
                "new registrations start EXPERIMENTAL with no promotion block"
            )
        sealed = _with(
            record,
            status=ModelStatus.EXPERIMENTAL,
            release_digest=compute_release_digest(record),
        )
        self._records[ref] = sealed
        self._log.seed(
            entity="model_registry",
            entity_id=ref,
            initial_fields={"registry_status": ModelStatus.EXPERIMENTAL.value},
        )
        if record.ensemble is not None:
            self._ensembles.append(ref)
        return sealed

    # -- intake: EXPERIMENTAL -> CHALLENGER ---------------------------------------

    def pass_intake(
        self,
        ref: str,
        *,
        weights_bytes: bytes,
        claim_statuses: Mapping[str, str],
    ) -> RegistryRecord:
        """Deterministic intake checks; no verdict needed (Part C §4)."""
        record = self.get(ref)
        current = self.project_status(ref)
        # Luna R3 fix #5: DEPRECATED and BLOCKED are terminal for intake;
        # neither may re-enter CHALLENGER (fail-closed).
        if current in _TERMINAL_FOR_INTAKE:
            raise StateTransitionError(
                f"{current.value} is first-class and can never re-enter "
                f"CHALLENGER via intake ({ref})"
            )
        problems: list[str] = []
        actual_weights = hashlib.sha256(weights_bytes).hexdigest()
        if actual_weights != record.weights.sha256:
            problems.append(
                f"weights checksum mismatch on disk: {actual_weights} != {record.weights.sha256}"
            )
        if not record.license.review_ref:
            problems.append("license review record missing (§52)")
        clinical_lane = record.intended_use.release_lane != ReleaseLane.PERSONAL_RESEARCH
        if clinical_lane and (
            record.data_policy.remote_processing
            or record.data_policy.network_required
            or record.data_policy.logs_sensitive_data
        ):
            problems.append("data policy inconsistent with a clinical-intent lane")
        for claim_id in record.validation.claim_ids:
            status = claim_statuses.get(claim_id)
            if status is None or status not in _CLAIM_STATUS_ORDER:
                problems.append(f"claim {claim_id} unknown to the claim registry")
            elif _CLAIM_STATUS_ORDER.index(status) < _CLAIM_STATUS_ORDER.index(
                _CLAIM_STATUS_FLOOR
            ):
                problems.append(
                    f"claim {claim_id} below {_CLAIM_STATUS_FLOOR} (got {status})"
                )
        # QC_001_MODEL_PREP D1 (additive problem, §9.1): a missing/empty
        # adapter or output-schema binding is an intake-blocking problem.
        # Presence only — no adapter id grammar, no schema format rule; the
        # record schema itself stays closed (survey §B-D1).
        problems.extend(intake_binding_presence_problems(record))
        if problems:
            raise CanonicalValidationError(problems)

        self._append_transition(
            ref,
            event_type="INTAKE_PASSED",
            from_status=current,
            to_status=ModelStatus.CHALLENGER,
            evidence_ref="intake checks passed",
        )
        updated = _with(self._records[ref], status=ModelStatus.CHALLENGER)
        updated = _with_history(
            updated, StatusHistoryEntry(status="CHALLENGER", at=_now(), by="registry-cli")
        )
        self._records[ref] = updated
        return updated

    # -- promotion -----------------------------------------------------------------

    def promote(
        self,
        ref: str,
        promotion: object,
        *,
        expected_task_id: str | None = None,
        at: str | None = None,
        by: str = "registry-cli",
    ) -> RegistryRecord:
        """Verify the CSI-001 chain then append PROMOTED/DEPRECATED events.

        Verification order is mandated (Luna fix #9) and each step fails
        closed:
          0. the structured owner gate and all CSI-001 chain digests are
             well-formed (Luna R3 fix #6) - rejected BEFORE anything applies;
          1. bytes(bundle_path) digest == bundle_sha256;
          2. governance acceptor re-run returns PASS;
          3. bundle milestone/task matches the release being promoted;
          4. bundle review_target_sha tree binds weights.sha256 and the
             preprocessing pipeline id@version.
        """
        if not isinstance(promotion, PromotionBlock):
            raise CanonicalValidationError("promotion must be a PromotionBlock")
        validate_promotion_block(promotion)
        record = self.get(ref)
        current = self.project_status(ref)
        if current == ModelStatus.BLOCKED:
            raise StateTransitionError("BLOCKED is first-class and never converts")
        if current != ModelStatus.CHALLENGER:
            raise StateTransitionError(
                f"only CHALLENGER may be promoted (current: {current.value})"
            )

        bundle_bytes = _read_bundle_bytes(promotion.bundle_path)
        actual_digest = hashlib.sha256(bundle_bytes).hexdigest()
        if actual_digest != promotion.bundle_sha256:
            raise DigestMismatchError(
                f"promotion step 1 failed: bundle bytes digest {actual_digest} "
                f"!= {promotion.bundle_sha256}"
            )
        if self._accept_bundle is None:
            raise CanonicalValidationError(
                "promotion step 2 requires an injected governance acceptor "
                "(product code does not import orchestration)"
            )
        if not self._accept_bundle(
            promotion.bundle_path,
            promotion.contract_sha256,
            promotion.permit_sha256,
            promotion.envelope_sha256,
        ):
            raise StateTransitionError(
                "promotion step 2 failed: governance accept did not PASS"
            )

        try:
            bundle_doc = json.loads(bundle_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CanonicalValidationError(f"bundle is not valid JSON: {exc}") from exc
        bundle_task = str(bundle_doc.get("task_id") or bundle_doc.get("milestone") or "")
        expected = expected_task_id or record.task
        if bundle_task != expected:
            raise StateTransitionError(
                f"promotion step 3 failed: bundle task {bundle_task!r} != {expected!r}"
            )

        shas = bundle_doc.get("shas") or {}
        tree = shas.get("review_target_sha") or []
        if isinstance(tree, str):
            tree = [tree]
        bound_items = {
            record.weights.sha256,
            record.input_contract.preprocessing,
        }
        missing = sorted(item for item in bound_items if item not in set(tree))
        if missing:
            raise StateTransitionError(
                f"promotion step 4 failed: review_target_sha does not bind {missing}"
            )

        lane = record.intended_use.release_lane
        key = (record.task, lane)
        previous_champion = self._champions.get(key)
        if previous_champion is not None:
            self.deprecate(
                previous_champion,
                superseded_by=ref,
                reason=f"challenged and replaced by {ref}",
                at=at,
                by=by,
            )

        stamp = at or _now()
        self._log.append_event(
            StateEvent(
                entity="model_registry",
                entity_id=ref,
                event_type="PROMOTED",
                from_value=ModelStatus.CHALLENGER.value,
                to_value=ModelStatus.CHAMPION.value,
                actor_role=by,
                at=stamp,
                evidence_ref=promotion.bundle_path,
                payload={
                    "bundle_sha256": promotion.bundle_sha256,
                    "contract_sha256": promotion.contract_sha256,
                    "permit_sha256": promotion.permit_sha256,
                    "envelope_sha256": promotion.envelope_sha256,
                    "accepted_sha": promotion.accepted_sha,
                    "rollback_target": promotion.rollback_target,
                },
            )
        )
        promoted = _with(
            record,
            status=ModelStatus.CHAMPION,
            champion_of=ChampionOf(task=record.task, lane=lane),
            promotion=promotion,
            acceptance_ref=AcceptanceRef(
                contract_sha256=promotion.contract_sha256,
                permit_sha256=promotion.permit_sha256,
                envelope_sha256=promotion.envelope_sha256,
                bundle_sha256=promotion.bundle_sha256,
                accepted_sha=promotion.accepted_sha,
            ),
        )
        promoted = _with_history(
            promoted,
            StatusHistoryEntry(
                status="CHAMPION", at=stamp, by=by, evidence_ref=promotion.bundle_path
            ),
        )
        self._records[ref] = promoted
        self._champions[key] = ref
        self._check_ensembles_for(ref)
        return promoted

    # -- first-class negative states ----------------------------------------------

    def deprecate(
        self,
        ref: str,
        *,
        superseded_by: str | None = None,
        reason: str = "",
        at: str | None = None,
        by: str = "registry-cli",
    ) -> RegistryRecord:
        """Append DEPRECATED; deprecated records can never be re-promoted."""
        record = self.get(ref)
        current = self.project_status(ref)
        if current == ModelStatus.DEPRECATED:
            return record
        stamp = at or _now()
        self._log.append_event(
            StateEvent(
                entity="model_registry",
                entity_id=ref,
                event_type="DEPRECATED",
                from_value=current.value,
                to_value=ModelStatus.DEPRECATED.value,
                actor_role=by,
                at=stamp,
                evidence_ref=superseded_by or reason or None,
                payload={"superseded_by": superseded_by} if superseded_by else {},
            )
        )
        updated = _with(record, status=ModelStatus.DEPRECATED)
        if self._champions.get((record.task, record.intended_use.release_lane)) == ref:
            del self._champions[(record.task, record.intended_use.release_lane)]
        updated = _with_history(
            updated,
            StatusHistoryEntry(status="DEPRECATED", at=stamp, by=by, evidence_ref=reason or None),
        )
        self._records[ref] = updated
        self._check_ensembles_for(ref)
        return updated

    def block(self, ref: str, *, reason: str, by: str = "registry-cli") -> RegistryRecord:
        """BLOCKED is first-class; a blocked champion's lane serves rollback."""
        record = self.get(ref)
        current = self.project_status(ref)
        if not reason:
            raise CanonicalValidationError("blocking requires an explicit reason")
        stamp = _now()
        self._log.append_event(
            StateEvent(
                entity="model_registry",
                entity_id=ref,
                event_type="BLOCKED",
                from_value=current.value,
                to_value=ModelStatus.BLOCKED.value,
                actor_role=by,
                at=stamp,
                evidence_ref=reason,
            )
        )
        updated = _with(record, status=ModelStatus.BLOCKED)
        if self._champions.get((record.task, record.intended_use.release_lane)) == ref:
            del self._champions[(record.task, record.intended_use.release_lane)]
        updated = _with_history(
            updated, StatusHistoryEntry(status="BLOCKED", at=stamp, by=by, evidence_ref=reason)
        )
        self._records[ref] = updated
        self._check_ensembles_for(ref)
        return updated

    # -- ensemble fail-closed rule ---------------------------------------------------

    def _check_ensembles_for(self, changed_ref: str) -> None:
        """Member DEPRECATED/BLOCKED => ensemble auto-BLOCKED (no substitution)."""
        changed_status = self.project_status(changed_ref)
        if changed_status not in (ModelStatus.DEPRECATED, ModelStatus.BLOCKED):
            return
        for ens_ref in list(self._ensembles):
            if ens_ref not in self._records:
                continue
            ens = self._records[ens_ref]
            if ens.ensemble is None or changed_ref not in ens.ensemble.members:
                continue
            if self.project_status(ens_ref) == ModelStatus.BLOCKED:
                continue
            reason = (
                f"ensemble member {changed_ref} became {changed_status.value}; "
                "no silent member substitution is allowed"
            )
            stamp = _now()
            self._log.append_event(
                StateEvent(
                    entity="model_registry",
                    entity_id=ens_ref,
                    event_type="BLOCKED",
                    from_value=self.project_status(ens_ref).value,
                    to_value=ModelStatus.BLOCKED.value,
                    actor_role="ENGINE",
                    at=stamp,
                    evidence_ref=reason,
                )
            )
            blocked = _with(ens, status=ModelStatus.BLOCKED)
            blocked = _with_history(
                blocked,
                StatusHistoryEntry(status="BLOCKED", at=stamp, by="ENGINE", evidence_ref=reason),
            )
            self._records[ens_ref] = blocked

    # -- helpers ----------------------------------------------------------------------

    def _append_transition(
        self,
        ref: str,
        *,
        event_type: str,
        from_status: ModelStatus,
        to_status: ModelStatus,
        evidence_ref: str,
    ) -> None:
        self._log.append_event(
            StateEvent(
                entity="model_registry",
                entity_id=ref,
                event_type=event_type,
                from_value=from_status.value,
                to_value=to_status.value,
                actor_role="registry-cli",
                at=_now(),
                evidence_ref=evidence_ref,
            )
        )


def _read_bundle_bytes(path: str) -> bytes:
    try:
        return Path(path).read_bytes()
    except OSError as exc:
        raise CanonicalValidationError(f"cannot read bundle {path!r}: {exc}") from exc


def _now() -> str:
    return utc_now_iso()


def _with(record: RegistryRecord, **changes: object) -> RegistryRecord:
    kwargs = {f.name: getattr(record, f.name) for f in dc_fields(RegistryRecord)}
    kwargs.update(changes)
    return RegistryRecord(**kwargs)


def _with_history(record: RegistryRecord, entry: StatusHistoryEntry) -> RegistryRecord:
    return _with(record, status_history=(*record.status_history, entry))


__all__ = [
    "BundleAcceptor",
    "ModelRegistry",
    "intake_binding_presence_problems",
    "model_ref",
]
