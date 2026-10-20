"""Registry record schema ``ccta.model_registry.v1`` (WS00_DESIGN.md Part C §2).

Closed schema: unknown fields are rejected at load (CSI-001 AC-001 analogy).
Every model release is a frozen artifact; ``release_digest`` is computed via
the shared canonical hashing rule with the self-reference excluded (I12).
"""

from __future__ import annotations

import enum
import types
from dataclasses import MISSING, asdict, dataclass, fields
from typing import Any, Union, get_args, get_origin, get_type_hints

from core.digest import canonical_digest, is_sha256_hex, require_sha256_hex
from core.enums import ModelStatus, ReleaseLane
from core.errors import CanonicalValidationError, ClosedSchemaError, DigestMismatchError

REGISTRY_SCHEMA = "ccta.model_registry.v1"

#: Split roles defined by the F4 dataset registry (§46/F4 §2) - not by F5.
SPLIT_ROLES: frozenset[str] = frozenset({"train", "val", "calibration", "locked_test"})

#: Source classes of dataset snapshots; ``local_real_world`` may never train
#: models bound to clinical claims (F5 §2a).
SOURCE_CLASSES: frozenset[str] = frozenset(
    {"public", "synthetic", "local_real_world"}
)

KEBAB_OK = frozenset("abcdefghijklmnopqrstuvwxyz0123456789-")

#: Mutable-lifecycle fields (I13): they live exclusively in the event log /
#: history projections and NEVER influence ``release_digest`` (I12 frozen
#: identity block only: identity + weights + preprocessing + benchmark refs).
TRANSIENT_RECORD_FIELDS: frozenset[str] = frozenset(
    {
        "release_digest",
        "status",
        "status_history",
        "champion_of",
        "promotion",
        "acceptance_ref",
    }
)


# ---------------------------------------------------------------------------
# Typed dataset snapshot bindings (Luna fix #7 + R3 fix)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetBinding:
    """Typed reference into the F4 dataset registry."""

    dataset_id: str
    snapshot_version: str
    split_role: str
    locked: bool
    checksum_sha256: str
    #: Mandatory (r5 attempt 3): a silent "public" default would let an
    #: omitted provenance class vacuously pass the R0b vocabulary check.
    source_class: str
    patient_disjoint_asserted: bool = False


# ---------------------------------------------------------------------------
# Record blocks
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceBlock:
    repository: str
    commit: str  # 40-hex


@dataclass(frozen=True)
class WeightsBlock:
    location: str
    sha256: str
    format: str
    adapter: str


@dataclass(frozen=True)
class LicenseBlock:
    code: str
    weights: str
    commercial_allowed: bool
    review_ref: str


@dataclass(frozen=True)
class DataPolicyBlock:
    logs_sensitive_data: bool
    remote_processing: bool = False
    network_required: bool = False
    training_data_refs: tuple[DatasetBinding, ...] = ()


@dataclass(frozen=True)
class InputContract:
    modality: str
    protocol: str
    preprocessing: str  # pipeline id@version - change forces version bump (P9)


@dataclass(frozen=True)
class OutputContract:
    schema: str


@dataclass(frozen=True)
class IntendedUseBlock:
    release_lane: ReleaseLane
    target_user: str
    limitations: tuple[str, ...] = ()


@dataclass(frozen=True)
class RuntimeBlock:
    local_supported: bool
    network_required: bool
    hardware: str


@dataclass(frozen=True)
class ValidationBlock:
    claim_ids: tuple[str, ...]
    claim_version_at_registration: str
    benchmark_ref: DatasetBinding | None
    performance_path: str
    performance_sha256: str


@dataclass(frozen=True)
class RiskBlock:
    known_failure_modes: tuple[str, ...] = ()
    required_qc: tuple[str, ...] = ()


@dataclass(frozen=True)
class TrainingSnapshot:
    """``lineage.training_snapshot`` typed binding (Luna R2 fix #6b)."""

    dataset_id: str
    snapshot_version: str
    split_role: str
    locked: bool
    checksum_sha256: str
    #: Mandatory (b13, Luna review @8328cba): without an explicit F4 source
    #: class a local_real_world snapshot could hide in lineage un-checked.
    source_class: str
    patient_disjoint_asserted: bool


@dataclass(frozen=True)
class LineageBlock:
    base_model: str | None = None
    training_snapshot: TrainingSnapshot | None = None


@dataclass(frozen=True)
class ChampionOf:
    task: str
    lane: ReleaseLane


@dataclass(frozen=True)
class OwnerGateV2:
    """Structured v2 owner gate - bare path+sha is NOT approval evidence."""

    required: bool
    status: str
    approver: str
    approved_at: str
    evidence_path: str
    evidence_sha256: str


@dataclass(frozen=True)
class PromotionBlock:
    """Digest-bound acceptance evidence (Luna fix #9 verification order)."""

    bundle_path: str
    bundle_sha256: str
    contract_sha256: str
    permit_sha256: str
    envelope_sha256: str
    accepted_sha: str
    rollback_target: str
    owner_gate: OwnerGateV2


@dataclass(frozen=True)
class AcceptanceRef:
    """Full CSI-001 chain binding (Luna R2 fix #8)."""

    contract_sha256: str
    permit_sha256: str
    envelope_sha256: str
    bundle_sha256: str
    accepted_sha: str


@dataclass(frozen=True)
class EnsembleBlock:
    members: tuple[str, ...]
    combination_strategy: str


@dataclass(frozen=True)
class StatusHistoryEntry:
    status: str
    at: str
    by: str
    evidence_ref: str | None = None


@dataclass(frozen=True)
class RegistryRecord:
    """One frozen model release record (Part C §2)."""

    model_id: str
    model_version: str
    task: str
    kind: str  # model | ensemble
    source: SourceBlock
    weights: WeightsBlock
    license: LicenseBlock
    data_policy: DataPolicyBlock
    input_contract: InputContract
    output_contract: OutputContract
    intended_use: IntendedUseBlock
    runtime: RuntimeBlock
    validation: ValidationBlock
    risk: RiskBlock
    registered_by: str
    registered_at: str
    lineage: LineageBlock = LineageBlock()
    status: ModelStatus = ModelStatus.EXPERIMENTAL
    champion_of: ChampionOf | None = None
    promotion: PromotionBlock | None = None
    ensemble: EnsembleBlock | None = None
    acceptance_ref: AcceptanceRef | None = None
    release_digest: str = ""
    status_history: tuple[StatusHistoryEntry, ...] = ()
    schema_version: str = REGISTRY_SCHEMA


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _is_hex40_git_sha(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and set(value) <= set("0123456789abcdef")
    )


def validate_dataset_role_rules(record: RegistryRecord) -> None:
    """Fail-closed REJECT rules for dataset roles (Part C §2 data_policy).

    R5 fix: the role ledger spans the UNION of training refs, the locked
    benchmark ref and the lineage training snapshot - any snapshot carrying
    more than one role across those surfaces rejects (R4).
    """
    violations: list[str] = []
    training = record.data_policy.training_data_refs
    snapshots_seen: dict[tuple[str, str], set[str]] = {}

    def _note(dataset_id: str, snapshot_version: str, role: str) -> None:
        snapshots_seen.setdefault((dataset_id, snapshot_version), set()).add(role)

    for ref in training:
        _note(ref.dataset_id, ref.snapshot_version, ref.split_role)
        # R0a/R0b: roles and source classes come from the F4 vocabulary only.
        if ref.split_role not in SPLIT_ROLES:
            violations.append(
                f"REJECT R0a: training ref {ref.dataset_id}@{ref.snapshot_version} "
                f"has unknown split_role {ref.split_role!r}"
            )
        if ref.source_class not in SOURCE_CLASSES:
            violations.append(
                f"REJECT R0b: training ref {ref.dataset_id}@{ref.snapshot_version} "
                f"has unknown source_class {ref.source_class!r}"
            )
        # R1: a locked-test split can never be used for training.
        if ref.split_role == "locked_test":
            violations.append(
                f"REJECT R1: training ref {ref.dataset_id}@{ref.snapshot_version} "
                "uses split_role=locked_test"
            )
        # R2 (F5 §2a): local real-world data must not train governed models.
        if ref.source_class == "local_real_world":
            violations.append(
                f"REJECT R2: training ref {ref.dataset_id}@{ref.snapshot_version} "
                "has source_class=local_real_world"
            )
        if not is_sha256_hex(ref.checksum_sha256):
            violations.append(f"training ref {ref.dataset_id}: checksum must be 64-hex")
        # R6: patient-disjointness must be asserted for every training snapshot.
        if not ref.patient_disjoint_asserted:
            violations.append(
                f"REJECT R6: training ref {ref.dataset_id}@{ref.snapshot_version} "
                "lacks a patient_disjointness assertion"
            )

    benchmark = record.validation.benchmark_ref
    # R3: benchmark binding MUST be locked_test + locked + patient-disjoint.
    if benchmark is None:
        violations.append("REJECT R3: validation.benchmark_ref is mandatory")
    else:
        # R5 fix: the benchmark participates in vocabulary checks, checksum
        # well-formedness and cross-surface conflict detection like any other
        # dataset reference.
        _note(benchmark.dataset_id, benchmark.snapshot_version, benchmark.split_role)
        if benchmark.split_role not in SPLIT_ROLES:
            violations.append(
                f"REJECT R0a: benchmark ref {benchmark.dataset_id}@"
                f"{benchmark.snapshot_version} has unknown split_role "
                f"{benchmark.split_role!r}"
            )
        if benchmark.source_class not in SOURCE_CLASSES:
            violations.append(
                f"REJECT R0b: benchmark ref {benchmark.dataset_id}@"
                f"{benchmark.snapshot_version} has unknown source_class "
                f"{benchmark.source_class!r}"
            )
        if not is_sha256_hex(benchmark.checksum_sha256):
            violations.append(
                f"benchmark ref {benchmark.dataset_id}: checksum must be 64-hex"
            )
        if benchmark.split_role != "locked_test":
            violations.append(
                f"REJECT R3a: benchmark split_role={benchmark.split_role!r}, "
                "must be locked_test"
            )
        if not benchmark.locked:
            violations.append("REJECT R3b: benchmark snapshot must be locked=true")
        if not benchmark.patient_disjoint_asserted:
            violations.append(
                "REJECT R3c: patient-disjointness must be asserted in the F4 record"
            )

    # R5: lineage training snapshots obey the same fail-closed rules.
    snap = record.lineage.training_snapshot
    if snap is not None:
        _note(snap.dataset_id, snap.snapshot_version, snap.split_role)
        if snap.split_role not in SPLIT_ROLES:
            violations.append(
                f"REJECT R0a: lineage.training_snapshot {snap.dataset_id}@"
                f"{snap.snapshot_version} has unknown split_role {snap.split_role!r}"
            )
        if snap.split_role == "locked_test":
            violations.append(
                f"REJECT R1: lineage.training_snapshot {snap.dataset_id}@"
                f"{snap.snapshot_version} uses split_role=locked_test"
            )
        # b13: lineage snapshots carry an explicit F4 source class, so the
        # R2 ban (never local_real_world) is enforceable here too.
        if snap.source_class not in SOURCE_CLASSES:
            violations.append(
                f"REJECT R0b: lineage.training_snapshot {snap.dataset_id}@"
                f"{snap.snapshot_version} has unknown source_class "
                f"{snap.source_class!r}"
            )
        if snap.source_class == "local_real_world":
            violations.append(
                f"REJECT R2: lineage.training_snapshot {snap.dataset_id}@"
                f"{snap.snapshot_version} has source_class=local_real_world"
            )
        if not is_sha256_hex(snap.checksum_sha256):
            violations.append(
                f"lineage.training_snapshot {snap.dataset_id}: checksum must be 64-hex"
            )
        if not snap.patient_disjoint_asserted:
            violations.append(
                "REJECT R5: lineage.training_snapshot lacks patient_disjoint_asserted"
            )
        if not snap.locked:
            violations.append("REJECT R5b: lineage.training_snapshot must be locked")

    # R4: one snapshot may carry exactly ONE role; any conflict rejects.
    for (_ds, _snap), roles in snapshots_seen.items():
        if len(roles) > 1:
            violations.append(
                f"REJECT R4: snapshot {_ds}@{_snap} referenced with conflicting "
                f"roles {sorted(roles)}"
            )

    if violations:
        raise CanonicalValidationError(violations)


def validate_promotion_block(promotion: PromotionBlock) -> None:
    """Fail-closed validation of a CSI-001 promotion payload.

    The structured v2 ``owner_gate`` must be a real approval (required,
    approved, evidenced by path + 64-hex digest), every chain digest must be
    well-formed, and ``accepted_sha`` must be a 40-hex git SHA.
    """
    problems: list[str] = []
    gate = promotion.owner_gate
    if not isinstance(gate, OwnerGateV2):
        problems.append("promotion.owner_gate must be a structured OwnerGateV2 object")
    else:
        if not gate.required:
            problems.append("owner_gate.required must be true")
        if gate.status != "approved":
            problems.append(
                f"owner_gate.status must be 'approved', got {gate.status!r}"
            )
        if not gate.approver:
            problems.append("owner_gate.approver is required")
        if not gate.approved_at:
            problems.append("owner_gate.approved_at is required")
        if not gate.evidence_path:
            problems.append("owner_gate.evidence_path is required")
        if not is_sha256_hex(gate.evidence_sha256):
            problems.append("owner_gate.evidence_sha256 must be a 64-hex sha256 digest")
    for name in ("bundle_sha256", "contract_sha256", "permit_sha256", "envelope_sha256"):
        value = getattr(promotion, name)
        if not is_sha256_hex(value):
            problems.append(f"promotion.{name} must be a 64-hex sha256 digest")
    accepted = promotion.accepted_sha
    if not _is_hex40_git_sha(accepted):
        problems.append("promotion.accepted_sha must be a 40-hex git SHA")
    if not promotion.rollback_target:
        problems.append("promotion.rollback_target is required")
    if problems:
        raise CanonicalValidationError(problems)


def validate_registry_record(record: RegistryRecord) -> None:
    """Structural validation of a frozen record (closed schema semantics)."""
    problems: list[str] = []
    bad_id = set(record.model_id) - KEBAB_OK
    if not record.model_id or bad_id:
        problems.append(f"model_id must be kebab-case slug, got {record.model_id!r}")
    if not record.model_version:
        problems.append("model_version is required")
    if not record.task:
        problems.append("task is required")
    if record.kind not in ("model", "ensemble"):
        problems.append(f"kind must be 'model' or 'ensemble', got {record.kind!r}")
    if record.kind == "ensemble" and record.ensemble is None:
        problems.append("kind=ensemble requires an ensemble block")
    try:
        require_sha256_hex(record.weights.sha256, "weights.sha256")
    except ValueError as exc:
        problems.append(str(exc))
    commit = record.source.commit
    if not _is_hex40_git_sha(commit):
        problems.append("source.commit must be a 40-hex git SHA")
    if record.status == ModelStatus.CHAMPION and (
        record.champion_of is None or record.promotion is None
    ):
        problems.append(
            "status=CHAMPION requires champion_of and a digest-bound promotion block"
        )
    if record.promotion is not None:
        try:
            validate_promotion_block(record.promotion)
        except CanonicalValidationError as exc:
            problems.extend(exc.violations)
    if not record.intended_use.release_lane:
        problems.append("intended_use.release_lane is required")
    ens = record.ensemble
    if ens is not None:
        if not ens.members:
            problems.append("ensemble.members must be non-empty")
        if not ens.combination_strategy:
            problems.append("ensemble.combination_strategy is required")

    # R5 fix (sealed-record validation gaps): digest well-formedness is
    # enforced on EVERY load regardless of digest coverage. acceptance_ref
    # stays excluded from release_digest (I12) - validation is orthogonal.
    if not is_sha256_hex(record.validation.performance_sha256):
        problems.append("validation.performance_sha256 must be a 64-hex sha256 digest")
    bench = record.validation.benchmark_ref
    if bench is not None and not is_sha256_hex(bench.checksum_sha256):
        problems.append(
            "validation.benchmark_ref.checksum_sha256 must be a 64-hex sha256 digest"
        )
    lineage_snap = record.lineage.training_snapshot
    if lineage_snap is not None and not is_sha256_hex(lineage_snap.checksum_sha256):
        problems.append(
            "lineage.training_snapshot.checksum_sha256 must be a 64-hex sha256 digest"
        )
    acceptance = record.acceptance_ref
    if acceptance is not None:
        for name in (
            "contract_sha256",
            "permit_sha256",
            "envelope_sha256",
            "bundle_sha256",
        ):
            if not is_sha256_hex(getattr(acceptance, name)):
                problems.append(
                    f"acceptance_ref.{name} must be a 64-hex sha256 digest"
                )
        if not _is_hex40_git_sha(acceptance.accepted_sha):
            problems.append("acceptance_ref.accepted_sha must be a 40-hex git SHA")

    try:
        validate_dataset_role_rules(record)
    except CanonicalValidationError as exc:
        problems.extend(exc.violations)

    if problems:
        raise CanonicalValidationError(problems)


def intake_binding_presence_problems(record: RegistryRecord) -> list[str]:
    """Intake-side PRESENCE check for binding fields (QC_001_MODEL_PREP D1).

    Returns the problem strings for a missing/empty ``weights.adapter`` or
    ``output_contract.schema`` so the governed intake path
    (``ModelRegistry.pass_intake``) can collect them together with ALL other
    intake problems and raise once, fail-closed, with a visible reason naming
    the field. Additive per instructions §9.1: no existing validator, problem
    string, or schema field is changed.

    PRESENCE ONLY: no adapter id grammar, no ``id@version`` format rule, no
    structural parse — any such rule touches record semantics and is a
    separate owner decision (survey §B-D1 note 1).
    """
    problems: list[str] = []
    adapter = record.weights.adapter
    if adapter is None or not adapter.strip():
        problems.append("weights.adapter binding missing or empty")
    schema = record.output_contract.schema
    if schema is None or not schema.strip():
        problems.append("output_contract.schema binding missing or empty")
    return problems


# ---------------------------------------------------------------------------
# Serialization (canonical dict <-> record), unknown fields fail closed
# ---------------------------------------------------------------------------


def _json_safe(value: Any) -> Any:
    """Recursively convert enums to their string values (JSON round-trip safe)."""
    if isinstance(value, enum.Enum):
        return _json_safe(value.value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_json_safe(item) for item in value)
    return value


def record_to_dict(record: RegistryRecord) -> dict[str, Any]:
    """Canonical dict form; every enum becomes its plain string value so the
    dump is directly ``json.dumps``-serializable (Luna R3 fix #3)."""
    return _json_safe(asdict(record))


def record_from_dict(data: dict[str, Any]) -> RegistryRecord:
    """Strict loader with closed-schema semantics AND full registration-grade
    validation (Luna R3 fix #2): unknown keys rejected, required fields
    present, digests well-formed; a carried ``release_digest`` must verify."""
    hints = get_type_hints(RegistryRecord)
    known = set(hints)
    unknown = set(data) - known
    if unknown:
        raise ClosedSchemaError(
            f"unknown field(s) in {REGISTRY_SCHEMA} record: {sorted(unknown)}"
        )
    kwargs: dict[str, Any] = {}
    for name, hint in hints.items():
        if name not in data:
            continue
        try:
            kwargs[name] = _coerce(hint, data[name], name)
        except TypeError as exc:
            raise CanonicalValidationError(f"{name}: {exc}") from exc
    missing_required = [
        f.name
        for f in fields(RegistryRecord)
        if f.name not in kwargs
        and f.default is MISSING
        and f.default_factory is MISSING
    ]
    if missing_required:
        raise CanonicalValidationError(f"missing required fields: {missing_required}")
    record = RegistryRecord(**kwargs)
    validate_registry_record(record)
    if record.release_digest:
        verify_release_digest(record)
    return record


def load_registry_record(data: dict[str, Any]) -> RegistryRecord:
    """Persistence-path loader: everything :func:`record_from_dict` enforces,
    plus a mandatory sealed ``release_digest`` (frozen records are always
    sealed at registration)."""
    record = record_from_dict(data)
    if not is_sha256_hex(record.release_digest):
        raise CanonicalValidationError(
            "persisted registry records must carry a sealed 64-hex release_digest"
        )
    return record


def _coerce(hint: Any, value: Any, path: str) -> Any:
    origin = get_origin(hint)
    args = get_args(hint)

    if isinstance(hint, type) and issubclass(hint, enum.Enum):
        return hint(value)
    if origin in (tuple, list):
        inner = args[0] if args else Any
        items = [_coerce(inner, item, f"{path}[{i}]") for i, item in enumerate(value)]
        return tuple(items) if origin is tuple else items
    if origin is Union or origin is types.UnionType:
        if value is None:
            return None
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _coerce(non_none[0], value, path)
        raise ClosedSchemaError(f"{path}: unsupported union {args!r}")
    if isinstance(hint, type) and is_dataclass_type(hint):
        if not isinstance(value, dict):
            raise ClosedSchemaError(f"{path}: expected object, got {type(value)!r}")
        sub_hints = get_type_hints(hint)
        unknown = set(value) - set(sub_hints)
        if unknown:
            raise ClosedSchemaError(f"unknown field(s) in {path}: {sorted(unknown)}")
        return hint(
            **{k: _coerce(sub_hints[k], v, f"{path}.{k}") for k, v in value.items()}
        )
    return value


def is_dataclass_type(obj: Any) -> bool:
    from dataclasses import is_dataclass

    return is_dataclass(obj) and isinstance(obj, type)


def compute_release_digest(record: RegistryRecord) -> str:
    """I12 self-digest over the FROZEN IDENTITY BLOCK only.

    Excluded (Luna R3 fix #4, per I12/I13): ``release_digest`` itself and all
    mutable-lifecycle fields (``status``, ``status_history``, ``champion_of``,
    ``promotion``, ``acceptance_ref``). Transitions live exclusively in the
    event log / history projections, so a status change can never invalidate
    the digest of a frozen release.
    """
    payload = record_to_dict(record)
    return canonical_digest(payload, exclude=tuple(sorted(TRANSIENT_RECORD_FIELDS)))


def verify_release_digest(record: RegistryRecord) -> None:
    expected = compute_release_digest(record)
    if expected != record.release_digest:
        raise DigestMismatchError(
            f"release digest mismatch for {record.model_id}@{record.model_version}"
        )


__all__ = [
    "AcceptanceRef",
    "ChampionOf",
    "DataPolicyBlock",
    "DatasetBinding",
    "EnsembleBlock",
    "InputContract",
    "IntendedUseBlock",
    "LicenseBlock",
    "LineageBlock",
    "OwnerGateV2",
    "PromotionBlock",
    "REGISTRY_SCHEMA",
    "RegistryRecord",
    "RiskBlock",
    "RuntimeBlock",
    "SPLIT_ROLES",
    "SOURCE_CLASSES",
    "SourceBlock",
    "StatusHistoryEntry",
    "TRANSIENT_RECORD_FIELDS",
    "TrainingSnapshot",
    "ValidationBlock",
    "WeightsBlock",
    "compute_release_digest",
    "intake_binding_presence_problems",
    "load_registry_record",
    "record_from_dict",
    "record_to_dict",
    "validate_dataset_role_rules",
    "validate_promotion_block",
    "validate_registry_record",
    "verify_release_digest",
]
