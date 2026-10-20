"""Append-only human assessment and correction workflow (QC_001_SPEC Part 4.4).

This module is the minimum production surface required by the accepted
``QC_001-HUMAN`` benchmark on top of the accepted QC_001_DET engine.  It adds
two entry points and never changes the deterministic family algorithms:

- :func:`create_initial_human_assessment` — the NO-predecessor case.  The
  emitted thirteen-key v1 payload still carries ``supersedes_ref`` as a PRESENT
  key with the NULL value; no supersession event is produced.
- :func:`create_human_correction` — the correction case.  It requires the
  prior EFFECTIVE assessment, emits ``supersedes_ref`` as a non-null ref to it,
  and appends the store's real SUPERSEDED ``StateEvent`` for the predecessor.

Frozen human literals (single actor identity, REVIEWED/CORRECTED event types,
``HUMAN_EVIDENCE`` justification, ``parameters_ref=NotApplicable`` with the
reason ``HUMAN_REVIEW_HAS_NO_DETERMINISTIC_PARAMETER_BLOB``) are the only
admitted forms.  ``HumanEdit`` stays OUTSIDE the frozen payload document and
binds only through the §44 provenance records.

Refusal precedence (frozen, fail-closed): supersession-shape faults
(``QC_SUPERSESSION_INVALID``) are reported before evidence faults
(``QC_HUMAN_EVIDENCE_INVALID``), before provenance faults
(``QC_PROVENANCE_UNVERIFIED``), before actor faults
(``QC_HUMAN_ACTOR_INVALID``), before event-type faults
(``QC_HUMAN_EVENT_INVALID``).  The append-only identity guard
(``QC_HUMAN_OVERWRITE_REJECTED``) runs last, immediately before the single
write, mirroring the store's own append-only admission check; a reused
ancestor identity inside a correction is therefore reported as the shape
fault (cycle) first, which is the only ordering consistent with the frozen
refusal table.  No refused request ever writes an artifact, an event, a
status change, or a payload byte.

Add-side admission preflight (review r2 finding 3, hardened r3 finding 2 and
r4 finding 1): before the first visible mutation the constructor proves that
EVERY admission structure ``DerivedArtifactStore.add`` consults would admit
the successor — the artifact map, the append-only event-log seed, AND the
invalidation DAG.  The DAG rehearsal mirrors ``add``'s exact
``register_geometry`` + ``bind_dependent`` sequence on a deep copy of the
store's DAG, read through the store's public ``invalidation_dag`` property
(a pure read: the real DAG is never mutated by the preflight).  Because
properties and injected objects are only trustworthy when they are the
canonical ones, the constructor first runs a TRUST GATE: the store's runtime
type must be exactly ``DerivedArtifactStore`` with the class-owned
``invalidation_dag`` descriptor, and the injected log and DAG objects must
be exactly ``EventLog`` and ``InvalidationDAG`` — a subclass, an accessor
override, or a lying injected object refuses ``QC_PROVENANCE_UNVERIFIED``
before any mutation (no caller-supplied DAG argument exists either).
"""

from __future__ import annotations

import copy
import enum
import pkgutil
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, is_dataclass
from dataclasses import fields as dataclass_fields
from importlib import import_module
from typing import Any, cast

from core.artifacts import (
    ArtifactKind,
    ArtifactStatus,
    ClaimBinding,
    DerivedArtifact,
    DerivedArtifactStore,
    Lineage,
    PayloadRef,
    Producer,
    QualityBlock,
    ReviewBlock,
    create_derived_artifact,
)
from core.dicom.qc import QCAssessmentArtifact, QCAssessmentPayload, QCScope
from core.digest import canonical_digest
from core.enums import (
    ActorType,
    ProducerType,
    ProvenanceEventType,
    QualityGrade,
    ReleaseLane,
    ReviewState,
)
from core.events import EventLog, utc_now_iso, validate_iso8601_utc
from core.ids import new_ulid
from core.provenance import (
    Actor,
    BlobRef,
    Environment,
    HumanEdit,
    InputRef,
    InvalidationDAG,
    NotApplicable,
    ProvenanceRecord,
    create_provenance,
    verify_chain,
)

from .algorithms import validate_qc_input
from .constants import ABSTAINED_FLAG, FIRST_NINE_FLAGS, PAYLOAD_SCHEMA, QC_PREPROCESSING
from .factory import (
    _DEFAULT_CLAIM,
    _DEFAULT_SOFTWARE,
    _ZERO_COMMIT,
    _canonical_instance_inputs,
    _root_provenance,
    _study_subject,
    _valid_ulid,
)
from .types import QCInput, QCValidationError

#: The single frozen human actor identity (never an ENGINE id, never per-case).
ACTOR_ID_HUMAN: str = "qc-human-correction-v1"
#: Evidence kind bound through the payload ``evidence_refs`` mechanism.
HUMAN_EVIDENCE_KIND: str = "HUMAN_EVIDENCE"
#: The only admitted justification code for a HUMAN assessment (spec §1.2).
HUMAN_JUSTIFICATION_CODE: str = "HUMAN_EVIDENCE"
#: Event type for the no-predecessor initial human assessment.
EVENT_INITIAL_HUMAN = ProvenanceEventType.REVIEWED
#: Event type for every human correction of a prior assessment.
EVENT_HUMAN_CORRECTION = ProvenanceEventType.CORRECTED
#: The frozen v1 payload key count; the key set never grows.
PAYLOAD_KEY_COUNT: int = 13
#: Subject prefix of the append-only HUMAN_EVIDENCE provenance record.
HUMAN_EVIDENCE_SUBJECT_PREFIX: str = "qc_human_evidence"
#: Deterministic provenance id of the source-manifest root record.
SOURCE_ROOT_PREFIX: str = "qc-source-manifest"
#: Default ``created_by`` stamp of the human workflow surface.
HUMAN_CREATED_BY: str = "qc_001_human"
#: The frozen §44 parameters binding for every human record.
PARAMETERS_NOT_APPLICABLE: NotApplicable[BlobRef] = NotApplicable(
    value=None, reason="HUMAN_REVIEW_HAS_NO_DETERMINISTIC_PARAMETER_BLOB"
)
_HUMAN_FLAGS: tuple[str, ...] = FIRST_NINE_FLAGS + (ABSTAINED_FLAG,)


class QCHumanRefusalCode(enum.StrEnum):
    """Closed typed-refusal vocabulary of the human workflow (Part 4.4)."""

    QC_SUPERSESSION_INVALID = "QC_SUPERSESSION_INVALID"
    QC_HUMAN_EVIDENCE_INVALID = "QC_HUMAN_EVIDENCE_INVALID"
    QC_PROVENANCE_UNVERIFIED = "QC_PROVENANCE_UNVERIFIED"
    QC_HUMAN_ACTOR_INVALID = "QC_HUMAN_ACTOR_INVALID"
    QC_HUMAN_EVENT_INVALID = "QC_HUMAN_EVENT_INVALID"
    QC_HUMAN_OVERWRITE_REJECTED = "QC_HUMAN_OVERWRITE_REJECTED"


class QCHumanRefusalError(QCValidationError):
    """Fail-closed typed refusal; the code is asserted exactly by the benchmark."""

    def __init__(self, code: QCHumanRefusalCode, detail: str = "") -> None:
        self.code = QCHumanRefusalCode(code)
        super().__init__(f"{code.value}: {detail}" if detail else code.value)


def _refuse(code: QCHumanRefusalCode, detail: str) -> QCHumanRefusalError:
    return QCHumanRefusalError(code, detail)


@dataclass(frozen=True, slots=True)
class HumanEvidence:
    """The request-bound human evidence content of one assessment.

    Only the frozen values are admitted; the benchmark supplies wrong values
    through this DTO to prove the constructor fails closed.
    """

    edit_summary: str
    edited_at: str
    kind: str = HUMAN_EVIDENCE_KIND
    actor_type: ActorType = ActorType.HUMAN
    actor_id: str = ACTOR_ID_HUMAN
    editor: str = ACTOR_ID_HUMAN


@dataclass(frozen=True, slots=True)
class QCHumanAssessmentResult:
    """Both QC views plus the append-only human provenance of one assessment."""

    assessment: QCAssessmentArtifact
    artifact: DerivedArtifact
    payload_document: Mapping[str, object]
    human_evidence_document: Mapping[str, object]
    human_edit: HumanEdit
    provenance_records: tuple[ProvenanceRecord, ...]
    aggregate_provenance_id: str
    human_event_provenance_id: str
    supersedes_ref: str | None
    event_type: ProvenanceEventType

    @property
    def artifact_ref(self) -> str:
        return self.assessment.artifact_ref


def _validate_human_matrix(grade: QualityGrade, flags: tuple[str, ...]) -> None:
    """Enforce the closed grade<->flag matrix for a human judgment."""

    if len(set(flags)) != len(flags):
        raise QCValidationError("human QC flags contain duplicates")
    if any(flag not in _HUMAN_FLAGS for flag in flags):
        raise QCValidationError("human QC flags contain an unknown or model label")
    if tuple(flag for flag in _HUMAN_FLAGS if flag in flags) != flags:
        raise QCValidationError("human QC flags are not in canonical order")
    if grade == QualityGrade.NOT_ANALYZABLE:
        if flags != (ABSTAINED_FLAG,):
            raise QCValidationError("NOT_ANALYZABLE human QC carries exactly ABSTAINED")
    elif ABSTAINED_FLAG in flags:
        raise QCValidationError("ABSTAINED flag requires a NOT_ANALYZABLE grade")
    elif grade == QualityGrade.GOOD:
        if flags:
            raise QCValidationError("GOOD human QC carries no flags")
    elif not flags:
        raise QCValidationError("ACCEPTABLE/LIMITED human QC requires flags")


def _artifact_ref(artifact_id: str) -> str:
    return f"qc_artifact:{artifact_id}"


@dataclass(frozen=True, slots=True)
class _GraphView:
    """One assessment's supersession-graph membership (no payload needed)."""

    ref: str
    study: str
    supersedes_ref: str | None
    status: str
    artifact_id: str


def _effective_tips(views: Sequence[_GraphView]) -> tuple[_GraphView, ...]:
    """Return the assessment tips: records no other record explicitly supersedes.

    This is the single graph resolution used by both the selector and the
    constructor's effective-predecessor proof.  Refs only: no timestamp, no
    file order, no status guessing.
    """

    superseded = {view.supersedes_ref for view in views if view.supersedes_ref is not None}
    return tuple(view for view in views if view.ref not in superseded)


def _preflight_supersession(
    artifact_store: DerivedArtifactStore, predecessor_id: str
) -> None:
    """Verify every store precondition of the supersession before it is emitted.

    The append-only event log is the only remaining failure source after the
    identity preflight: a malformed or externally seeded log can make
    ``mark_superseded`` raise.  Checking the projected status and the log's
    knowledge of the predecessor here keeps that residual failure BEFORE the
    transition is visible, so a refused or failed request leaves the
    predecessor in its prior status with zero appended events.
    """

    predecessor = artifact_store.get(predecessor_id)
    if predecessor.status not in (ArtifactStatus.ACTIVE, ArtifactStatus.STALE):
        raise _refuse(
            QCHumanRefusalCode.QC_SUPERSESSION_INVALID,
            "the predecessor left a supersedable status before the write",
        )
    if not _event_log_knows(artifact_store, predecessor_id):
        raise _refuse(
            QCHumanRefusalCode.QC_PROVENANCE_UNVERIFIED,
            "the predecessor is unknown to the append-only event log",
        )
    projected = artifact_store.project(predecessor_id).get("status")
    if projected != predecessor.status.value:
        raise _refuse(
            QCHumanRefusalCode.QC_PROVENANCE_UNVERIFIED,
            "the predecessor's projected status disagrees with its record",
        )


def _event_log_knows(artifact_store: DerivedArtifactStore, artifact_id: str) -> bool:
    """True when the append-only event log already seeds this artifact identity."""

    return any(
        entity == "derived_artifact" and entity_id == artifact_id
        for (entity, entity_id), _fields in artifact_store.event_log.initial_states()
    )


#: The canonical instance-attribute key sets of the frozen admission classes,
#: read from a freshly constructed instance of each (never hard-coded, so a
#: legitimate upstream field addition cannot silently break the guard while a
#: caller-injected attribute key can never be canonical), together with the
#: canonical VALUE SHAPE of every field: the concrete type the class
#: constructor itself creates for that field (review r7 finding 1 — a
#: non-callable dict-SUBCLASS value under a canonical name carries
#: caller-defined container semantics that the preflight reads but the
#: rehearsal must also trust).  Derived at import from the same fresh
#: instances, so nothing is hard-coded.
def _canonical_surface(instance: object) -> tuple[frozenset[str], dict[str, type]]:
    """Read a fresh canonical instance's field names and value shapes."""

    fields = vars(instance)
    return (
        frozenset(fields),
        {name: type(value) for name, value in fields.items()},
    )


_CANONICAL_STORE_FIELDS, _CANONICAL_STORE_VALUE_SHAPES = _canonical_surface(
    DerivedArtifactStore(EventLog(), InvalidationDAG())
)
_CANONICAL_EVENT_LOG_FIELDS, _CANONICAL_EVENT_LOG_VALUE_SHAPES = _canonical_surface(
    EventLog()
)
_CANONICAL_DAG_FIELDS, _CANONICAL_DAG_VALUE_SHAPES = _canonical_surface(
    InvalidationDAG()
)

#: RECURSIVE PURE-DATA WHITELIST for the admission substrate (review r9
#: finding 1).  The att0008 guard constrains the DIRECT types of the
#: canonical field values but nothing INSIDE those containers; a tuple
#: SUBCLASS key inside the concrete builtin ``_initial`` dict can retain the
#: real tuple's hash/equality (so ``EventLog.seed`` membership sees the
#: duplicate) while overriding ``__iter__`` (so ``initial_states()``'s
#: unpacking sees a decoy) — a stable preflight/write asymmetry the r9
#: reviewer reproduced as a full partial write.  The validator below therefore
#: accepts ONLY pure data across the FULL reachable substrate value tree:
#: exact builtin scalars, exact builtin containers whose every member is
#: recursively valid, frozen dataclasses whose class is defined in the core
#: admission modules (derived at import — never hard-coded), and enum members
#: of core-derived enum classes.  No subclass of anything is accepted
#: anywhere in the tree.
_CANONICAL_SCALARS: frozenset[type] = frozenset({str, int, float, bool, bytes})

_SUBSTRATE_PACKAGES: tuple[str, ...] = (
    "core.artifacts",
    "core.events",
    "core.provenance",
    "core.enums",
    "core.qc",
)


def _core_substrate_module_names() -> tuple[str, ...]:
    """The core admission-module universe, derived at import.

    The substrate root modules above (never hard-coded per class) plus,
    one level deep, the submodules of any root that is a package (e.g.
    ``core.provenance`` and ``core.qc``), discovered via ``pkgutil``.
    Importing a substrate module is safe: they are the modules the guard's
    own imports already load transitively.
    """

    names: list[str] = []
    for root in _SUBSTRATE_PACKAGES:
        module = import_module(root)
        names.append(root)
        for module_info in pkgutil.iter_modules(
            getattr(module, "__path__", ()), prefix=f"{root}."
        ):
            names.append(module_info.name)
    return tuple(dict.fromkeys(names))


def _derive_substrate_whitelists() -> tuple[frozenset[type], frozenset[type]]:
    """Derive the frozen-dataclass and enum class whitelists at import.

    A class qualifies only when it is DEFINED in a core substrate module
    (its ``__module__`` is the scanned module), is a genuine dataclass, and
    is FROZEN (``frozen=True``); non-frozen or non-dataclass core classes
    are excluded, as is anything defined outside the core substrate
    modules.  Enum classes must be defined in a substrate module and
    subclass :class:`enum.Enum`.  Because these whitelists are read-only
    snapshots of immutable class objects, class-level patching of the
    underlying classes remains the documented standing boundary (it is
    runtime modification of the program, not a caller-object mutation).
    """

    frozen_classes: set[type] = set()
    enum_classes: set[type] = set()
    for module_name in _core_substrate_module_names():
        module = import_module(module_name)
        for value in vars(module).values():
            if not isinstance(value, type) or value.__module__ != module_name:
                continue
            if is_dataclass(value):
                params = getattr(value, "__dataclass_params__", None)
                if params is not None and params.frozen:
                    frozen_classes.add(value)
            elif issubclass(value, enum.Enum):
                enum_classes.add(value)
    return frozenset(frozen_classes), frozenset(enum_classes)


_CANONICAL_FROZEN_DATACLASSES, _CANONICAL_ENUM_CLASSES = (
    _derive_substrate_whitelists()
)

#: Generous recursion cap: the deepest real substrate tree measured is five
#: levels (artifact map → artifact → payload → producer → …).  Anything past
#: this bound is refused rather than followed, fail-closed.
_MAX_PURE_DATA_DEPTH = 24


def _assert_untouched_instance_surface(
    label: str,
    obj: object,
    canonical: frozenset[str],
    shapes: dict[str, type],
) -> None:
    """Prove *obj*'s instance surface and field VALUES are untouched.

    Python resolves an instance attribute ahead of the class function, so an
    attribute whose name is NOT a canonical field necessarily shadows a real
    method or smuggles a hook (``__deepcopy__``, ``__class__``, …), and a
    callable VALUE under a canonical name would be equally non-canonical.
    Both cases let the preflight and the inherited ``add`` observe different
    behavior from the same object (review r5 finding 1), so any divergence
    refuses ``QC_PROVENANCE_UNVERIFIED`` before the object is used.

    The attribute mapping itself must first be the CONCRETE BUILTIN ``dict``
    (review r6 / att0007): Python accepts ``obj.__dict__ = <mapping>`` for
    any dict SUBCLASS, and such a subclass can override ``__iter__``/``keys``
    so that ``set(vars(obj))`` reports the canonical key set while the real
    underlying storage still holds a shadowed method that attribute lookup
    returns — defeating the key-set proof entirely.  Requiring
    ``type(vars(obj)) is dict`` closes that vector: with the concrete
    builtin, ``vars(obj)`` iterates the SAME storage that instance attribute
    lookup and the inherited ``add`` read, so the key set cannot lie and the
    exact-key-set plus non-callable checks below regain full proof strength.

    Finally every canonical field VALUE must be the CONCRETE BUILTIN TYPE the
    class constructor itself creates (review r7 finding 1): a non-callable
    ``dict`` SUBCLASS under a canonical name can define caller-controlled
    container semantics (an ``items()`` that hides a pre-seeded successor
    identity while ordinary membership still sees it), so the preflight
    (``initial_states()`` reads ``.items()``) and the live admission checks
    (``seed()`` reads ``key in self._initial``) disagree about the SAME
    storage and the write can fail late — after supersession and insertion —
    a stable no-concurrency partial write the r7 reviewer reproduced through
    the real constructor.  Restricting the DIRECT field values to the
    concrete builtins the constructor creates removes every
    caller-definable container semantic at that level; the contents INSIDE
    those builtins are then closed separately by the recursive pure-data
    validator (``_assert_pure_data_substrate``, review r9 finding 1), which
    proves no node of the reachable substrate value tree can redefine
    hash, equality, unpacking, or iteration between the preflight and the
    write.
    """

    fields = vars(obj)
    if type(fields) is not dict:
        raise _refuse(
            QCHumanRefusalCode.QC_PROVENANCE_UNVERIFIED,
            f"the {label} instance attribute mapping is not the concrete "
            "builtin dict, so its attribute surface cannot be verified",
        )
    if set(fields) != canonical:
        extra = sorted(set(fields) - canonical)
        missing = sorted(canonical - set(fields))
        raise _refuse(
            QCHumanRefusalCode.QC_PROVENANCE_UNVERIFIED,
            f"the {label} instance attribute set is not canonical"
            + (f"; unexpected: {extra}" if extra else "")
            + (f"; missing: {missing}" if missing else ""),
        )
    for name in sorted(fields):
        value = fields[name]
        if callable(value):
            raise _refuse(
                QCHumanRefusalCode.QC_PROVENANCE_UNVERIFIED,
                f"the {label} instance attribute {name!r} is callable and "
                "therefore not canonical",
            )
        if type(value) is not shapes[name]:
            raise _refuse(
                QCHumanRefusalCode.QC_PROVENANCE_UNVERIFIED,
                f"the {label} instance attribute {name!r} is "
                f"{type(value).__name__}, not the canonical "
                f"{shapes[name].__name__} the class constructor creates",
            )


def _assert_pure_data_substrate(
    label: str, root: object, _visiting: frozenset[int] | None = None
) -> None:
    """Recursively prove *root* is PURE DATA (review r9 finding 1).

    Every node in the reachable value tree of the admission substrate must
    be one of:

    - ``None``;
    - an EXACT builtin scalar (``str``, ``int``, ``float``, ``bool``,
      ``bytes`` — ``type(x) is <builtin>``, no subclass anywhere);
    - an EXACT builtin container (``dict``, ``list``, ``tuple``,
      ``frozenset``) whose every key/member and value is recursively
      valid;
    - an instance of a frozen dataclass DEFINED in a core admission module
      (whitelist derived at import; exact type, no subclass) whose per-
      instance attribute mapping is the CONCRETE BUILTIN ``dict`` (review
      r10 / att0010 — with the builtin mapping, the field-set iteration
      reads real storage and cannot present a false field set, so no
      hidden per-instance ``__deepcopy__``/``__class__`` hook can exist on
      an accepted dataclass node), whose field set matches
      ``dataclasses.fields`` of its class exactly and whose every field
      value is recursively valid; or
    - a member of a core-derived enum class (exact class; members are
      singletons, so there is no substitution surface).

    Everything else is refused ``QC_PROVENANCE_UNVERIFIED``: a
    caller-controlled node with redefinable data-model semantics (a tuple
    subclass key that keeps real hash/equality while unpacking as a decoy,
    a str-subclass inner key or value, a foreign object inside the artifact
    map, ...) can make the preflight and the live admission checks disagree
    about the same storage, which is exactly the stable pre-call
    partial-write shape the r9 reviewer reproduced.  With the whole tree
    pure data, ``initial_states()``, ``seed()``, ``bind_dependent()``, and
    the inherited ``add()`` all read the same storage through the same
    builtin semantics — the preflight and the write cannot disagree.

    Recursion: pure-data trees are finite and acyclic in practice; cycles
    are still guarded by an id-based ``_visiting`` set (a node already on
    the current path is SKIPPED, not followed), and the walk is capped at
    ``_MAX_PURE_DATA_DEPTH`` (a generous bound far above the measured real
    tree) — past the cap the guard refuses rather than descends,
    fail-closed.
    """

    if _visiting is None:
        _visiting = frozenset()
    if len(_visiting) > _MAX_PURE_DATA_DEPTH:
        raise _refuse(
            QCHumanRefusalCode.QC_PROVENANCE_UNVERIFIED,
            f"the {label} value tree exceeds the pure-data depth bound; "
            "the admission substrate cannot be verified",
        )
    node_type = type(root)
    if root is None or node_type in _CANONICAL_SCALARS:
        return
    if node_type in (dict, list, tuple, frozenset):
        if id(root) in _visiting:
            return
        visiting = _visiting | {id(root)}
        if node_type is dict:
            container = cast("dict[Any, Any]", root)
            for key, value in list(container.items()):
                _assert_pure_data_substrate(label, key, visiting)
                _assert_pure_data_substrate(label, value, visiting)
        else:
            for member in list(cast("Iterable[Any]", root)):
                _assert_pure_data_substrate(label, member, visiting)
        return
    if node_type in _CANONICAL_ENUM_CLASSES:
        return
    if node_type in _CANONICAL_FROZEN_DATACLASSES:
        if id(root) in _visiting:
            return
        visiting = _visiting | {id(root)}
        # The per-instance attribute mapping must be the CONCRETE BUILTIN
        # dict (review r10 / att0010).  Python accepts
        # ``obj.__dict__ = <mapping>`` even for frozen dataclasses (via
        # ``object.__setattr__``), and a dict SUBCLASS can override
        # ``__iter__``/``keys`` to present exactly the declared field set
        # while its real storage hides a per-instance hook such as
        # ``__deepcopy__`` — which ``copy.deepcopy`` consults (the
        # manager-confirmed r10 probe: the rehearsal deep copy returned a
        # decoy object).  With the concrete builtin, the field-set
        # iteration below reads REAL storage and cannot lie: any hidden
        # extra key (including ``__deepcopy__``/``__class__`` hooks)
        # surfaces in the comparison and refuses, and a missing key
        # refuses too.
        node_fields = vars(root)
        if type(node_fields) is not dict:
            raise _refuse(
                QCHumanRefusalCode.QC_PROVENANCE_UNVERIFIED,
                f"the {label} contains a frozen dataclass whose instance "
                "attribute mapping is not the concrete builtin dict, so "
                "its attribute surface cannot be verified",
            )
        declared = {
            field.name for field in dataclass_fields(cast("type[Any]", node_type))
        }
        actual = {name for name in node_fields}
        if actual != declared:
            raise _refuse(
                QCHumanRefusalCode.QC_PROVENANCE_UNVERIFIED,
                f"the {label} contains a frozen dataclass whose instance "
                "attribute set diverges from its declared fields",
            )
        for name in sorted(declared):
            _assert_pure_data_substrate(label, getattr(root, name), visiting)
        return
    raise _refuse(
        QCHumanRefusalCode.QC_PROVENANCE_UNVERIFIED,
        f"the {label} value tree contains a "
        f"{node_type.__module__}.{node_type.__qualname__} node, which is "
        "not pure admission data",
    )


def _assert_canonical_admission_surface(
    artifact_store: DerivedArtifactStore,
) -> None:
    """Prove the admission substrate is exactly the frozen canonical classes.

    The zero-partial proof of the final write is derived from the CANONICAL
    admission surface: the frozen ``DerivedArtifactStore.add`` consults its
    private artifact map, private event log, and private invalidation DAG,
    and the preflights read those through the base class's own ``event_log``
    and ``invalidation_dag`` properties.  Any of those surfaces can LIE:

    - a ``DerivedArtifactStore`` SUBCLASS can override ``invalidation_dag``
      to advertise a clean decoy DAG while the inherited ``add`` continues to
      mutate the real private ``_dag`` (review r4 finding 1 — the reviewer
      reproduced the full partial write through such a subtype);
    - a subclass can equally override ``event_log`` so the preflight's
      ``initial_states()`` read hides a pre-seeded successor identity that
      the inherited ``add`` then re-seeds (raising AFTER the artifact was
      already inserted — the same non-atomic partial write through the log
      seam);
    - the ``log`` and ``dag`` OBJECTS injected at construction can themselves
      be lying subclasses whose ``initial_states``/``seed`` or
      ``register_geometry``/``bind_dependent`` are asymmetric between the
      rehearsal copy and the live object;
    - even an EXACT canonical-type instance is mutable: an INSTANCE
      ATTRIBUTE can shadow a method (or smuggle ``__deepcopy__``), so
      ``type(obj) is EventLog`` still says nothing about which callable the
      preflight will actually invoke (review r5 finding 1 — the reviewer
      reproduced the partial write by shadowing ``initial_states`` and by an
      asymmetric ``bind_dependent`` on exact canonical instances);
    - the attribute MAPPING itself can be substituted: Python accepts
      ``obj.__dict__ = <dict subclass>``, and a subclass overriding
      ``__iter__``/``keys`` makes ``set(vars(obj))`` report the canonical key
      set while the real storage still holds the shadowing attribute that
      lookup returns (review r6 / att0007 — the manager confirmed the probe),
      which is why the surface guard first requires the concrete builtin
      ``dict``;
    - a canonical field VALUE inside an otherwise-canonical mapping can be a
      caller-controlled NON-CALLABLE container with caller-defined semantics:
      a ``dict`` subclass under ``_initial`` whose ``items()`` hides a
      pre-seeded successor identity while ordinary membership still sees it
      makes the preflight (``initial_states()`` reads ``.items()``) lie while
      the live ``seed()`` (membership) sees the truth — the write then fails
      LATE, after supersession and insertion (review r7 finding 1 — the
      reviewer reproduced the partial write through the real constructor),
      which is why every field value must also be the concrete builtin type
      the constructor creates;
    - the CONTENTS of those concrete builtin containers are still
      unconstrained by direct type checks: a tuple SUBCLASS key inside the
      builtin ``_initial`` dict can retain the real tuple's hash/equality
      (so ``seed()`` membership sees the duplicate) while overriding
      ``__iter__`` (so ``initial_states()``'s unpacking sees a decoy), a
      stable preflight/write asymmetry the r9 reviewer reproduced as a full
      partial write — which is why the guard ALSO recursively proves the
      whole reachable substrate value tree is PURE DATA.

    The store is therefore admitted to the write ONLY when its runtime type
    is EXACTLY ``DerivedArtifactStore``, its ``invalidation_dag`` accessor is
    the class-owned base descriptor, the injected log and DAG objects are
    EXACTLY ``EventLog`` and ``InvalidationDAG``, and every one of those
    three objects carries a CONCRETE BUILTIN ``dict`` attribute mapping whose
    key set is EXACTLY the canonical field set with non-callable values of
    the CONCRETE BUILTIN TYPE the class constructor creates (derived at
    import from fresh instances), and every node of the reachable substrate
    value tree is PURE DATA (exact builtin scalars/containers, import-derived
    frozen dataclasses with exact field sets, core-derived enum members — no
    subclass accepted anywhere in the tree) — so no subclass, accessor
    override, injected object, instance-method shadow, smuggled
    ``__deepcopy__``, substituted ``__dict__`` mapping, substituted canonical
    field value, or redefinable-semantics node inside the containers can be
    anywhere in the trust chain.  Refusal code:
    ``QC_PROVENANCE_UNVERIFIED`` — the fault is a verification failure of
    the write's substrate (the same frozen family as the event-log/DAG
    collision refusals), not a supersession-shape fault; the request's own
    supersession shape is untouched by it.  Precedence: the gate runs BEFORE
    the store is read at all, so a request against a non-canonical store
    reports this substrate fault; the frozen shape > evidence > provenance
    ranking is unaffected because a canonical store never reaches the gate.
    """

    if type(artifact_store) is not DerivedArtifactStore:
        raise _refuse(
            QCHumanRefusalCode.QC_PROVENANCE_UNVERIFIED,
            "the artifact store is a subclass; the canonical admission "
            "surface cannot be verified",
        )
    if type(artifact_store).invalidation_dag is not (
        DerivedArtifactStore.invalidation_dag
    ):
        raise _refuse(
            QCHumanRefusalCode.QC_PROVENANCE_UNVERIFIED,
            "the store's invalidation_dag accessor is not the class-owned "
            "base property",
        )
    log = artifact_store.event_log
    if type(log) is not EventLog:
        raise _refuse(
            QCHumanRefusalCode.QC_PROVENANCE_UNVERIFIED,
            "the append-only event log is not the canonical EventLog type",
        )
    dag = artifact_store.invalidation_dag
    if type(dag) is not InvalidationDAG:
        raise _refuse(
            QCHumanRefusalCode.QC_PROVENANCE_UNVERIFIED,
            "the invalidation DAG is not the canonical InvalidationDAG type",
        )
    # Instance-surface and VALUE-SHAPE clauses (review r5 finding 1, extended
    # for the r6 __dict__-substitution vector and the r7 canonical
    # field-value-substitution vector).  An exact canonical TYPE still says
    # nothing about the METHOD SURFACE of that instance: Python resolves an
    # instance attribute ahead of the class function, so a caller can shadow
    # `initial_states` / `bind_dependent` / `register_geometry` / `seed`, or
    # smuggle `__deepcopy__`, on an exact EventLog or InvalidationDAG instance
    # whose type is unchanged — and can hide such a shadow from `vars()` by
    # substituting a lying dict subclass for `__dict__`.  Even with the
    # outer mapping the CONCRETE BUILTIN dict and the keys exactly canonical,
    # a canonical field VALUE can still be a non-callable dict subclass with
    # caller-defined container semantics (an items() that hides a pre-seeded
    # identity while membership sees it), so the preflight and the live
    # admission checks disagree.  Requiring the CONCRETE BUILTIN dict for the
    # attribute mapping makes `vars()` read the same real storage that
    # attribute lookup and `add()` use, and requiring every field value to be
    # the CONCRETE BUILTIN TYPE the class constructor creates (derived at
    # import from fresh instances) makes iteration/membership/items()/keys()
    # non-redefinable across the whole reachable value tree: the preflight
    # and the inherited `add()` provably invoke the SAME real class methods
    # on the SAME live storage with the SAME concrete semantics.
    for label, obj, fields_, shapes_ in (
        (
            "artifact store",
            artifact_store,
            _CANONICAL_STORE_FIELDS,
            _CANONICAL_STORE_VALUE_SHAPES,
        ),
        (
            "event log",
            log,
            _CANONICAL_EVENT_LOG_FIELDS,
            _CANONICAL_EVENT_LOG_VALUE_SHAPES,
        ),
        (
            "invalidation DAG",
            dag,
            _CANONICAL_DAG_FIELDS,
            _CANONICAL_DAG_VALUE_SHAPES,
        ),
    ):
        _assert_untouched_instance_surface(label, obj, fields_, shapes_)
    # Recursive PURE-DATA validation of the full reachable substrate value
    # tree (review r9 finding 1).  Every key, member, and value inside the
    # canonical containers — the store's artifact map, the log's event list
    # and seed map, the DAG's version/bind/invalidation maps — must be pure
    # data: exact builtin scalars/containers, import-derived frozen
    # dataclasses with exact field sets, and core-derived enum members.  No
    # subclass is accepted anywhere in the tree, so no node can redefine
    # hash, equality, unpacking, or iteration between the preflight and the
    # write.  The walk starts at each guarded object's own instance dict
    # (already proven concrete and canonical by the surface clauses above);
    # the store's `_log`/`_dag` values ARE the guarded companion objects, so
    # they are skipped here and validated as their own roots below.
    for label, obj, companion_ids in (
        ("artifact store", artifact_store, (id(log), id(dag))),
        ("event log", log, ()),
        ("invalidation DAG", dag, ()),
    ):
        for name, value in vars(obj).items():
            if id(value) in companion_ids:
                continue
            _assert_pure_data_substrate(f"{label} field {name!r}", value)


def _preflight_dag_admission(
    artifact_store: DerivedArtifactStore, artifact: DerivedArtifact
) -> None:
    """Prove the invalidation DAG would admit the successor BEFORE any write.

    ``DerivedArtifactStore.add`` binds the successor id in the DAG only AFTER
    inserting the artifact and seeding the append-only log, so a pre-existing
    binding of the successor id would otherwise surface as a late failure
    that has already mutated the store.  The exact admission sequence ``add``
    performs — ``register_geometry`` (ValueError swallowed) followed by
    ``bind_dependent`` — is therefore rehearsed on a deep copy of the STORE'S
    OWN DAG, read through the public ``invalidation_dag`` property (review r3
    finding 2: a caller-supplied DAG cannot be authenticated, so the store
    itself is the only admissible proof source).  This is a pure read of the
    real DAG: a rehearsed duplicate-bind refusal raises the frozen
    ``QC_PROVENANCE_UNVERIFIED`` while nothing at all has been mutated, and a
    passing rehearsal proves the real ``add`` cannot fail at its own bind.
    """

    base = artifact.lineage.geometry_base
    if base is None:
        return  # add() skips the DAG entirely for a geometry-less artifact
    rehearsal = copy.deepcopy(artifact_store.invalidation_dag)
    try:
        rehearsal.register_geometry(base.geometry_id, max(base.geometry_version, 1))
    except ValueError:
        pass  # already tracked at >= this version; add() swallows the same error
    try:
        rehearsal.bind_dependent(
            artifact.artifact_id, base.geometry_id, base.geometry_version
        )
    except ValueError:
        raise _refuse(
            QCHumanRefusalCode.QC_PROVENANCE_UNVERIFIED,
            "the successor identity is already bound in the invalidation DAG",
        ) from None


def _store_views(
    artifact_store: DerivedArtifactStore, study_ref: str
) -> tuple[_GraphView, ...]:
    """Enumerate one study's assessments from the store's append-only seeds.

    ``EventLog.initial_states`` exposes every artifact the store has ever
    admitted, so the constructor can resolve the study's supersession graph
    through the public append-only surface without mutating anything.
    """

    views: list[_GraphView] = []
    for (entity, entity_id), _fields in artifact_store.event_log.initial_states():
        if entity != "derived_artifact" or entity_id not in artifact_store:
            continue
        artifact = artifact_store.get(entity_id)
        if artifact.kind != ArtifactKind.QC_ASSESSMENT or artifact.study_ref != study_ref:
            continue
        views.append(
            _GraphView(
                ref=_artifact_ref(entity_id),
                study=artifact.study_ref,
                supersedes_ref=artifact.review.supersedes_ref,
                status=artifact.status.value,
                artifact_id=entity_id,
            )
        )
    return tuple(views)


def _validate_supersession_shape(
    *,
    artifact_store: DerivedArtifactStore,
    supersedes_ref: str | None,
    new_artifact_id: str,
    study_ref: str,
    request_geometry_id: str,
    request_geometry_version: int,
) -> str:
    """Resolve and validate the predecessor; return its artifact id.

    Every failure here is a supersession-shape fault and refuses with
    ``QC_SUPERSESSION_INVALID`` (cases 10/11/12/13/15 of the frozen table).
    """

    if not isinstance(supersedes_ref, str) or not supersedes_ref.startswith(
        "qc_artifact:"
    ):
        raise _refuse(
            QCHumanRefusalCode.QC_SUPERSESSION_INVALID,
            "a correction requires a concrete qc_artifact predecessor ref",
        )
    old_id = supersedes_ref.split(":", 1)[1]
    if not _valid_ulid(old_id):
        raise _refuse(
            QCHumanRefusalCode.QC_SUPERSESSION_INVALID,
            "the predecessor ref must be qc_artifact:<ULID>",
        )
    try:
        predecessor = artifact_store.get(old_id)
    except KeyError:
        raise _refuse(
            QCHumanRefusalCode.QC_SUPERSESSION_INVALID,
            "the declared predecessor is not a resolvable assessment",
        ) from None
    if predecessor.kind != ArtifactKind.QC_ASSESSMENT:
        raise _refuse(
            QCHumanRefusalCode.QC_SUPERSESSION_INVALID,
            "the declared predecessor is not a QC assessment",
        )
    if predecessor.study_ref != study_ref:
        raise _refuse(
            QCHumanRefusalCode.QC_SUPERSESSION_INVALID,
            "the declared predecessor belongs to a different study",
        )
    # Cycle: walking the explicit supersedes chain from the predecessor must
    # never reach the new assessment identity or revisit any ref.
    new_ref = _artifact_ref(new_artifact_id)
    visited: set[str] = {new_ref}
    cursor = old_id
    while True:
        if cursor in visited:
            raise _refuse(
                QCHumanRefusalCode.QC_SUPERSESSION_INVALID,
                "the supersession chain contains a cycle",
            )
        visited.add(cursor)
        current = artifact_store.get(cursor)
        upstream = current.review.supersedes_ref
        if upstream is None:
            break
        if upstream == new_ref or upstream in visited:
            raise _refuse(
                QCHumanRefusalCode.QC_SUPERSESSION_INVALID,
                "the supersession chain contains a cycle",
            )
        next_id = upstream.split(":", 1)[1]
        if next_id not in artifact_store:
            break
        cursor = next_id
    if predecessor.status not in (ArtifactStatus.ACTIVE, ArtifactStatus.STALE):
        raise _refuse(
            QCHumanRefusalCode.QC_SUPERSESSION_INVALID,
            "a superseded or invalidated predecessor is not the effective assessment",
        )
    # The declared predecessor must be the UNIQUE EFFECTIVE assessment of the
    # study: resolve the study's supersession graph from the store's
    # append-only seeds and require exactly one tip, and require that tip to
    # be the declared predecessor.  This refuses a correction naming one of
    # two incomparable branches and never leaves a study without a single
    # effective assessment (frozen case 11 semantics).
    tips = _effective_tips(_store_views(artifact_store, study_ref))
    if len(tips) != 1 or tips[0].artifact_id != old_id:
        raise _refuse(
            QCHumanRefusalCode.QC_SUPERSESSION_INVALID,
            "the declared predecessor is not the unique effective assessment "
            "of the study",
        )
    prior_base = predecessor.lineage.geometry_base
    if prior_base is not None:
        if predecessor.status == ArtifactStatus.STALE:
            # The only legal way to supersede a stale dependent is the
            # descendant recompute path (valid case 8): either a strictly
            # newer version of the same geometry lineage, or a new canonical
            # (descendant) volume identity.  Re-pinning the stale geometry is
            # refused as frozen case 15.
            if (
                request_geometry_id == prior_base.geometry_id
                and request_geometry_version <= prior_base.geometry_version
            ):
                raise _refuse(
                    QCHumanRefusalCode.QC_SUPERSESSION_INVALID,
                    "a stale predecessor may be superseded only through the "
                    "descendant recompute path",
                )
        elif (
            request_geometry_id == prior_base.geometry_id
            and request_geometry_version < prior_base.geometry_version
        ):
            raise _refuse(
                QCHumanRefusalCode.QC_SUPERSESSION_INVALID,
                "a correction may not pin an older geometry than its predecessor",
            )
    return old_id


def _validate_human_evidence(human_evidence: HumanEvidence | Any) -> None:
    """Evidence-structure faults refuse with QC_HUMAN_EVIDENCE_INVALID."""

    if not isinstance(human_evidence, HumanEvidence):
        raise _refuse(
            QCHumanRefusalCode.QC_HUMAN_EVIDENCE_INVALID,
            "a human assessment requires a HumanEvidence object",
        )
    if human_evidence.kind != HUMAN_EVIDENCE_KIND:
        raise _refuse(
            QCHumanRefusalCode.QC_HUMAN_EVIDENCE_INVALID,
            "the human evidence kind must be HUMAN_EVIDENCE",
        )
    if not isinstance(human_evidence.edit_summary, str) or not (
        human_evidence.edit_summary.strip()
    ):
        raise _refuse(
            QCHumanRefusalCode.QC_HUMAN_EVIDENCE_INVALID,
            "the human evidence edit summary is required",
        )
    try:
        validate_iso8601_utc(human_evidence.edited_at, "edited_at")
    except ValueError:
        raise _refuse(
            QCHumanRefusalCode.QC_HUMAN_EVIDENCE_INVALID,
            "the human evidence timestamp must be ISO-8601 UTC",
        ) from None


def _validate_human_request_bindings(
    evidence: HumanEvidence, event_type: object, correction: bool
) -> None:
    """Typed pre-assembly checks for actor/editor and event-type SHAPE.

    These run before any ``create_provenance`` call so a malformed request can
    never surface as a generic ``ProvenanceIncompleteError``: an absent or
    empty editor must carry ``QC_HUMAN_ACTOR_INVALID`` (probe 16b) and a
    free-form/non-member event value must carry ``QC_HUMAN_EVENT_INVALID``
    (probe 16c).  The IDENTITY checks (actor type/id/editor equal to the
    frozen values, event member required by the case) remain after the
    provenance chain check, which preserves the frozen precedence
    (shape > evidence > provenance > actor > event) for co-occurring
    semantic faults.
    """

    if not isinstance(evidence.actor_type, ActorType):
        raise _refuse(
            QCHumanRefusalCode.QC_HUMAN_ACTOR_INVALID,
            "the actor type must be an ActorType member",
        )
    if not isinstance(evidence.actor_id, str) or not evidence.actor_id.strip():
        raise _refuse(
            QCHumanRefusalCode.QC_HUMAN_ACTOR_INVALID,
            "the human actor id is required",
        )
    if not isinstance(evidence.editor, str) or not evidence.editor.strip():
        raise _refuse(
            QCHumanRefusalCode.QC_HUMAN_ACTOR_INVALID,
            "the human editor identity is required",
        )
    if not isinstance(event_type, ProvenanceEventType):
        raise _refuse(
            QCHumanRefusalCode.QC_HUMAN_EVENT_INVALID,
            "the event type must be a real ProvenanceEventType member",
        )
    assert isinstance(event_type, ProvenanceEventType)
    assert isinstance(correction, bool)


def _validate_human_actor(evidence: HumanEvidence) -> None:
    """Actor/editor identity faults refuse with QC_HUMAN_ACTOR_INVALID."""

    if evidence.actor_type != ActorType.HUMAN:
        raise _refuse(
            QCHumanRefusalCode.QC_HUMAN_ACTOR_INVALID,
            "a human assessment requires ActorType.HUMAN",
        )
    if evidence.actor_id != ACTOR_ID_HUMAN:
        raise _refuse(
            QCHumanRefusalCode.QC_HUMAN_ACTOR_INVALID,
            "the human actor id is not the frozen human identity",
        )
    if evidence.editor != ACTOR_ID_HUMAN:
        raise _refuse(
            QCHumanRefusalCode.QC_HUMAN_ACTOR_INVALID,
            "the human editor identity is not the frozen human identity",
        )


def _root(
    *,
    provenance_id: str,
    subject_ref: str,
    digest: str,
    kind: str,
    ref: str,
    at: str,
    created_by: str,
    code_commit_sha: str,
) -> ProvenanceRecord:
    return _root_provenance(
        provenance_id=provenance_id,
        subject_ref=subject_ref,
        inputs=(InputRef(ref=ref, kind=kind, sha256_at_use=digest),),
        at=at,
        created_by=created_by,
        code_commit_sha=code_commit_sha,
    )


def _ensure_human_roots(
    context: Any,
    supplied: Sequence[ProvenanceRecord],
    *,
    at: str,
    created_by: str,
    code_commit_sha: str,
) -> tuple[tuple[ProvenanceRecord, ...], str, str, str]:
    """Ensure the source-manifest, volume, and FrameSpec/transform roots.

    The returned ids are, in order, the source-manifest provenance id, the
    volume provenance id, and the FrameSpec/transform provenance id — the
    exact upstream members frozen by the benchmark contract.
    """

    if any(not isinstance(record, ProvenanceRecord) for record in supplied):
        raise QCValidationError("supplied provenance_records must be ProvenanceRecords")
    records: list[ProvenanceRecord] = list(supplied)
    by_id = {record.provenance_id: record for record in records}
    if len(by_id) != len(records):
        raise QCValidationError("supplied provenance_records contain duplicate ids")

    def ensure(provenance_id: str, subject_ref: str, digest: str, kind: str, ref: str) -> None:
        existing = by_id.get(provenance_id)
        if existing is not None:
            if isinstance(existing.inputs, NotApplicable):
                raise QCValidationError(
                    f"upstream record {provenance_id!r} has no concrete source input"
                )
            if existing.subject_ref != subject_ref or not any(
                item.kind == kind and item.ref == ref and item.sha256_at_use == digest
                for item in existing.inputs
            ):
                raise QCValidationError(
                    f"upstream record {provenance_id!r} does not bind {ref!r}"
                )
            return
        record = _root(
            provenance_id=provenance_id,
            subject_ref=subject_ref,
            digest=digest,
            kind=kind,
            ref=ref,
            at=at,
            created_by=created_by,
            code_commit_sha=code_commit_sha,
        )
        records.append(record)
        by_id[provenance_id] = record

    source_manifest_sha = context.source_manifest_sha256
    source_root_id = f"{SOURCE_ROOT_PREFIX}:{source_manifest_sha}"
    ensure(
        source_root_id,
        f"source_manifest:{source_manifest_sha}",
        source_manifest_sha,
        "dataset",
        f"source_manifest:{source_manifest_sha}",
    )
    volume = context.volume
    volume_provenance = volume.provenance_ref
    if not isinstance(volume_provenance, str) or not volume_provenance:
        raise QCValidationError("volume provenance is missing")
    ensure(
        volume_provenance,
        volume.volume_id,
        volume.blob_ref.sha256,
        "volume",
        f"volume_id:{volume.volume_id}",
    )
    frame_provenance = context.source.transform_provenance_ref
    transform_ref = context.source.transform_ref
    if not isinstance(frame_provenance, str) or not frame_provenance:
        raise QCValidationError("versioned transform provenance is required")
    if not isinstance(transform_ref, str) or not transform_ref:
        raise QCValidationError("versioned transform ref is required")
    transform_digest = canonical_digest(context.source.transform)
    ensure(frame_provenance, transform_ref, transform_digest, "artifact", transform_ref)
    return tuple(records), source_root_id, volume_provenance, frame_provenance


def _evidence_document(
    *,
    evidence: HumanEvidence,
    grade: QualityGrade,
    flags: tuple[str, ...],
    artifact_ref: str,
    supersedes_ref: str | None,
    context: Any,
    event_type: ProvenanceEventType,
) -> dict[str, object]:
    """The frozen closed human evidence document."""

    return {
        "evidence_kind": evidence.kind,
        "actor_type": evidence.actor_type.value,
        "actor_id": evidence.actor_id,
        "editor": evidence.editor,
        "justification_code": HUMAN_JUSTIFICATION_CODE,
        "event_type": event_type.value,
        "quality_grade": grade.value,
        "flags": flags,
        "assessment_ref": artifact_ref,
        "supersedes_ref": supersedes_ref,
        "source_manifest_sha256": context.source_manifest_sha256,
        "volume_id": context.volume.volume_id,
        "frame_digest": canonical_digest(context.frame),
        "edit_summary": evidence.edit_summary,
        "edited_at": evidence.edited_at,
    }


def _create_human_assessment(
    source: QCInput,
    *,
    correction: bool,
    supersedes_ref: str | None,
    quality_grade: QualityGrade | str,
    flags: Sequence[str],
    human_evidence: HumanEvidence | Any,
    provenance_records: Sequence[ProvenanceRecord],
    artifact_store: DerivedArtifactStore | None,
    subject_refs: tuple[str, ...] | None,
    artifact_id: str | None,
    aggregate_provenance_id: str | None,
    human_event_provenance_id: str | None,
    event_type: ProvenanceEventType | None,
    claim_binding: ClaimBinding,
    release_lane: ReleaseLane,
    created_by: str,
    created_at: str | None,
    code_commit_sha: str,
    software_version: str,
    id_factory: Any,
) -> QCHumanAssessmentResult:
    # Validate the same canonical input surface the deterministic engine uses.
    context = validate_qc_input(source)
    grade = QualityGrade(quality_grade)
    flag_tuple = tuple(flags)
    _validate_human_matrix(grade, flag_tuple)
    # Source identity: the assessment subject is exactly the canonical source
    # volume's study.  A caller-supplied foreign study is refused so the
    # payload, artifact, volume, source manifest, transform, and HUMAN
    # evidence can never drift apart (Part 4.4 same-source-identity rule).
    canonical_subject = _study_subject(context.volume.study_ref)
    subjects = (canonical_subject,) if subject_refs is None else tuple(subject_refs)
    if not subjects:
        raise QCValidationError("human QC subject_refs are required")
    if subjects != (canonical_subject,):
        raise QCValidationError(
            "human QC subject_refs must bind the canonical source volume's study "
            f"(expected {canonical_subject!r}, got {subjects!r})"
        )

    mint = id_factory or new_ulid
    resolved_artifact_id = artifact_id or mint()
    resolved_aggregate_id = aggregate_provenance_id or mint()
    resolved_human_event_id = human_event_provenance_id or mint()
    if not _valid_ulid(resolved_artifact_id) or not _valid_ulid(resolved_aggregate_id):
        raise QCValidationError("human artifact and aggregate ids must be ULIDs")
    if not _valid_ulid(resolved_human_event_id):
        raise QCValidationError("human event provenance id must be a ULID")
    new_ref = _artifact_ref(resolved_artifact_id)
    stamp = created_at or utc_now_iso()
    # Trust gate (review r4 finding 1): the admission substrate must be the
    # frozen canonical classes BEFORE the constructor reads the store at all.
    # A store SUBCLASS, an accessor override, or a lying injected log/DAG
    # object must never be consulted for a supersession-shape or admission
    # decision, because the inherited ``add`` would then write against a
    # different substrate than the one the preflights proved (the r4 reviewer
    # reproduced the full partial write through such a subtype).  This runs
    # before step 1 so the frozen precedence still ranks genuine shape faults
    # above the substrate fault for a canonical store; a non-canonical store
    # simply never reaches the shape path.
    if artifact_store is not None:
        _assert_canonical_admission_surface(artifact_store)
    # Default ONLY on a genuine absence: every other non-ProvenanceEventType
    # value — including every falsy value ("", False, 0) — must reach the
    # typed event-shape guard below and refuse with QC_HUMAN_EVENT_INVALID
    # (probe 16c semantics), never be silently replaced by the valid default
    # (review r2 finding 1).
    if event_type is None:
        resolved_event_type: ProvenanceEventType = (
            EVENT_HUMAN_CORRECTION if correction else EVENT_INITIAL_HUMAN
        )
    else:
        resolved_event_type = event_type

    # 1. Supersession-shape faults (frozen precedence: first).
    predecessor_id: str | None = None
    prior_aggregate_id: str | None = None
    if correction:
        if artifact_store is None:
            raise _refuse(
                QCHumanRefusalCode.QC_SUPERSESSION_INVALID,
                "a correction requires the artifact store to resolve its predecessor",
            )
        predecessor_id = _validate_supersession_shape(
            artifact_store=artifact_store,
            supersedes_ref=supersedes_ref,
            new_artifact_id=resolved_artifact_id,
            study_ref=subjects[0],
            request_geometry_id=context.geometry_base.geometry_id,
            request_geometry_version=context.geometry_base.geometry_version,
        )
        prior_aggregate_id = artifact_store.get(
            predecessor_id
        ).provenance_ref.removeprefix("provenance:")
        if not _valid_ulid(prior_aggregate_id):
            raise _refuse(
                QCHumanRefusalCode.QC_SUPERSESSION_INVALID,
                "the predecessor provenance ref must be a ULID",
            )
    elif supersedes_ref is not None:
        raise _refuse(
            QCHumanRefusalCode.QC_SUPERSESSION_INVALID,
            "an initial human assessment carries no predecessor relation",
        )

    # 2. Evidence faults.
    _validate_human_evidence(human_evidence)
    evidence = cast(HumanEvidence, human_evidence)

    # 2b. Actor/editor and event-type SHAPE, before provenance assembly, so a
    # malformed request can never surface as a generic provenance error.
    _validate_human_request_bindings(evidence, resolved_event_type, correction)

    # 3. Provenance faults: assemble the full append-only record set and run
    # the real verify_chain before anything is emitted.
    root_records, source_root_id, volume_root_id, frame_root_id = _ensure_human_roots(
        context,
        provenance_records,
        at=stamp,
        created_by=created_by,
        code_commit_sha=code_commit_sha,
    )
    # The ordered prior-assessment provenance branch must be the STORED
    # predecessor's own aggregate record, not a substitution: a caller-supplied
    # record reusing the predecessor's provenance id is accepted only when its
    # subject binds the predecessor artifact AND its event/actor are exactly
    # the identities the stored predecessor's immutable relation implies —
    # the EVENT is derived from the stored predecessor's own ReviewBlock.
    # supersedes_ref (None = initial: REVIEWED for HUMAN, DERIVED for the
    # deterministic engine; set = correction: CORRECTED for HUMAN, RECOMPUTED
    # for the engine), and the actor is exactly the stored producer's
    # identity (the frozen human actor, or the stored engine producer id).
    # Membership in a producer-family event set is NOT enough: a re-sealed
    # record with the same-family alternative event is equally forged and
    # refuses (review r3 finding 1, completing r2 finding 2).
    if prior_aggregate_id is not None and predecessor_id is not None:
        stored_prior = next(
            (
                record
                for record in root_records
                if record.provenance_id == prior_aggregate_id
            ),
            None,
        )
        if stored_prior is not None:
            prior_artifact = artifact_store.get(predecessor_id) if (
                artifact_store is not None
            ) else None
            if (
                stored_prior.subject_ref != predecessor_id
                or not isinstance(stored_prior.event_type, ProvenanceEventType)
                or prior_artifact is None
            ):
                raise _refuse(
                    QCHumanRefusalCode.QC_PROVENANCE_UNVERIFIED,
                    "the supplied prior provenance record does not bind the "
                    "stored predecessor artifact",
                )
            prior_producer = prior_artifact.producer
            # The stored predecessor's own relation: supersedes_ref None means
            # an initial assessment, set means a correction/recompute.
            prior_is_correction = prior_artifact.review.supersedes_ref is not None
            if prior_producer.producer_type == ProducerType.HUMAN:
                required_prior_event = (
                    ProvenanceEventType.CORRECTED
                    if prior_is_correction
                    else ProvenanceEventType.REVIEWED
                )
                required_prior_actor = Actor(
                    actor_type=ActorType.HUMAN, actor_id=ACTOR_ID_HUMAN
                )
                actor_detail = "the stored human predecessor"
            elif prior_producer.producer_type == ProducerType.DETERMINISTIC_ENGINE:
                engine_actor_id = prior_producer.algorithm
                if not isinstance(engine_actor_id, str) or not engine_actor_id:
                    raise _refuse(
                        QCHumanRefusalCode.QC_PROVENANCE_UNVERIFIED,
                        "the stored deterministic predecessor carries no "
                        "engine producer identity",
                    )
                required_prior_actor = Actor(
                    actor_type=ActorType.ENGINE, actor_id=engine_actor_id
                )
                actor_detail = "the stored engine predecessor"
                # A deterministic initial assessment carries DERIVED, a
                # deterministic recompute/correction carries RECOMPUTED —
                # mirroring the engine emission site's own event choice.
                required_prior_event = (
                    ProvenanceEventType.RECOMPUTED
                    if prior_is_correction
                    else ProvenanceEventType.DERIVED
                )
            else:
                raise _refuse(
                    QCHumanRefusalCode.QC_PROVENANCE_UNVERIFIED,
                    "the stored predecessor is not a human or deterministic "
                    "QC assessment producer",
                )
            if stored_prior.event_type != required_prior_event or (
                stored_prior.actor != required_prior_actor
            ):
                raise _refuse(
                    QCHumanRefusalCode.QC_PROVENANCE_UNVERIFIED,
                    "the supplied prior provenance event/actor is inconsistent "
                    f"with {actor_detail}: exactly the "
                    f"{required_prior_event.value} event is required",
                )
    input_tuple = _canonical_instance_inputs(context)
    human_edit = HumanEdit(
        editor=evidence.editor,
        edit_summary=evidence.edit_summary,
        at=evidence.edited_at,
    )
    aggregate_upstreams = (source_root_id, volume_root_id, frame_root_id) + (
        (prior_aggregate_id,) if prior_aggregate_id is not None else ()
    )
    human_event_upstreams = (
        (prior_aggregate_id,) if prior_aggregate_id is not None else ()
    ) + (source_root_id, volume_root_id, frame_root_id)
    environment = Environment(
        software_version=software_version, code_commit_sha=code_commit_sha
    )
    aggregate_record = create_provenance(
        subject_ref=resolved_artifact_id,
        event_type=resolved_event_type,
        actor=Actor(actor_type=evidence.actor_type, actor_id=evidence.actor_id),
        inputs=input_tuple,
        preprocessing=QC_PREPROCESSING,
        parameters_ref=PARAMETERS_NOT_APPLICABLE,
        human_edit=human_edit,
        upstream_refs=aggregate_upstreams,
        environment=environment,
        model_record_ref=None,
        created_by=created_by,
        at=stamp,
        provenance_id=resolved_aggregate_id,
    )
    evidence_document = _evidence_document(
        evidence=evidence,
        grade=grade,
        flags=flag_tuple,
        artifact_ref=new_ref,
        supersedes_ref=supersedes_ref if correction else None,
        context=context,
        event_type=resolved_event_type,
    )
    evidence_digest = canonical_digest(evidence_document)
    human_event_record = create_provenance(
        subject_ref=f"{HUMAN_EVIDENCE_SUBJECT_PREFIX}:{evidence_digest}",
        event_type=resolved_event_type,
        actor=Actor(actor_type=evidence.actor_type, actor_id=evidence.actor_id),
        inputs=input_tuple,
        preprocessing=QC_PREPROCESSING,
        parameters_ref=PARAMETERS_NOT_APPLICABLE,
        human_edit=human_edit,
        upstream_refs=human_event_upstreams,
        environment=environment,
        model_record_ref=None,
        created_by=created_by,
        at=stamp,
        provenance_id=resolved_human_event_id,
    )
    all_records = tuple(root_records) + (aggregate_record, human_event_record)
    if len({record.provenance_id for record in all_records}) != len(all_records):
        raise QCValidationError("human provenance ids are not unique")
    chain_ok, chain_problems = verify_chain(all_records)
    if not chain_ok:
        raise _refuse(
            QCHumanRefusalCode.QC_PROVENANCE_UNVERIFIED,
            "the human provenance chain failed: " + "; ".join(chain_problems),
        )

    # 4. Actor/editor validity faults.
    _validate_human_actor(evidence)

    # 5. Event-type validity faults.
    required_event = EVENT_HUMAN_CORRECTION if correction else EVENT_INITIAL_HUMAN
    if resolved_event_type != required_event:
        raise _refuse(
            QCHumanRefusalCode.QC_HUMAN_EVENT_INVALID,
            f"this case requires the {required_event.value} event type",
        )

    # 6. Append-only identity guard AND complete admission preflight, both
    # before the single write.  The successor identity must be free in the
    # store map AND unseeded in the append-only event log, so the later
    # ``artifact_store.add`` can never fail after a predecessor has already
    # been marked SUPERSEDED (zero-partial-application on a failed write).
    if artifact_store is not None:
        if resolved_artifact_id in artifact_store or _event_log_knows(
            artifact_store, resolved_artifact_id
        ):
            raise _refuse(
                QCHumanRefusalCode.QC_HUMAN_OVERWRITE_REJECTED,
                "an assessment artifact with this identity already exists",
            )

    aggregate_ref = f"provenance:{resolved_aggregate_id}"
    human_event_ref = f"provenance:{resolved_human_event_id}"
    evidence_refs = (
        aggregate_ref,
        f"source_manifest:{context.source_manifest_sha256}",
        f"volume_id:{context.volume.volume_id}",
        human_event_ref,
    )
    payload_document: dict[str, object] = {
        "schema_version": PAYLOAD_SCHEMA,
        "scope": QCScope.STUDY.value,
        "subject_refs": subjects,
        "quality_grade": grade.value,
        "confidence": None,
        "flags": flag_tuple,
        "vessel_id": None,
        "region_id": None,
        "producer": ProducerType.HUMAN.value,
        "produced_by_ref": aggregate_ref,
        "supersedes_ref": supersedes_ref if correction else None,
        "justification_code": HUMAN_JUSTIFICATION_CODE,
        "evidence_refs": evidence_refs,
    }
    if len(payload_document) != PAYLOAD_KEY_COUNT:
        raise QCValidationError("the human payload key set is not closed")
    payload = QCAssessmentPayload(
        schema_version=PAYLOAD_SCHEMA,
        scope=QCScope.STUDY,
        subject_refs=subjects,
        quality_grade=grade,
        confidence=None,
        flags=flag_tuple,
        vessel_id=None,
        region_id=None,
        producer=ProducerType.HUMAN,
        produced_by_ref=aggregate_ref,
        supersedes_ref=supersedes_ref if correction else None,
        justification_code=HUMAN_JUSTIFICATION_CODE,
        evidence_refs=evidence_refs,
    )
    assessment = QCAssessmentArtifact(
        artifact_ref=new_ref,
        payload=payload,
        producer=ProducerType.HUMAN,
        status=ArtifactStatus.ACTIVE,
    )
    artifact = create_derived_artifact(
        artifact_id=resolved_artifact_id,
        study_ref=subjects[0],
        kind=ArtifactKind.QC_ASSESSMENT,
        payload=PayloadRef(
            payload_schema=PAYLOAD_SCHEMA,
            blob_sha256=canonical_digest(payload_document),
            codec="json",
            inline_summary=(("quality_grade", grade.value),),
        ),
        lineage=Lineage(
            input_volume_refs=(context.volume.volume_id,),
            input_series_refs=tuple(context.volume.source_series_refs),
            geometry_base=context.geometry_base,
        ),
        claim_binding=claim_binding,
        producer=Producer(producer_type=ProducerType.HUMAN),
        review=ReviewBlock(
            review_state=ReviewState.PREDICTED, supersedes_ref=payload.supersedes_ref
        ),
        quality=QualityBlock(quality_grade=grade, confidence=None, flags=flag_tuple),
        release_lane=release_lane,
        provenance_ref=aggregate_ref,
        created_by=created_by,
        created_at=stamp,
    )
    result = QCHumanAssessmentResult(
        assessment=assessment,
        artifact=artifact,
        payload_document=payload_document,
        human_evidence_document=evidence_document,
        human_edit=human_edit,
        provenance_records=all_records,
        aggregate_provenance_id=resolved_aggregate_id,
        human_event_provenance_id=resolved_human_event_id,
        supersedes_ref=payload.supersedes_ref,
        event_type=resolved_event_type,
    )
    validate_human_assessment(result)
    if artifact_store is not None:
        # The trust gate (`_assert_canonical_admission_surface`) already ran
        # before the first store read, so the preflights below read a substrate
        # the inherited ``add`` will write against.  Preflight every remaining
        # store precondition BEFORE any state transition: the predecessor must
        # still be in a supersedable status, its projected status must match
        # the transition's from_value, and the append-only log must still know
        # it.  The DAG rehearsal then proves the successor would pass EVERY
        # add-side admission structure — artifact map, event-log seed, and
        # invalidation-DAG bind — on the STORE'S OWN DAG, so the store write
        # below cannot fail after the predecessor has already been marked
        # SUPERSEDED (review r2 finding 3, hardened by r3 finding 2 and r4
        # finding 1: zero-partial-application on a failed write, with no
        # caller-supplied DAG argument and no substitutable accessor).
        if predecessor_id is not None:
            _preflight_supersession(artifact_store, predecessor_id)
        _preflight_dag_admission(artifact_store, artifact)
        # The real store emission site appends the exact SUPERSEDED event.
        if predecessor_id is not None:
            artifact_store.mark_superseded(
                predecessor_id, superseded_by=resolved_artifact_id
            )
        artifact_store.add(artifact)
    return result


def create_initial_human_assessment(
    source: QCInput,
    *,
    quality_grade: QualityGrade | str,
    flags: Sequence[str],
    human_evidence: HumanEvidence,
    provenance_records: Sequence[ProvenanceRecord] = (),
    artifact_store: DerivedArtifactStore | None = None,
    subject_refs: tuple[str, ...] | None = None,
    artifact_id: str | None = None,
    aggregate_provenance_id: str | None = None,
    human_event_provenance_id: str | None = None,
    event_type: ProvenanceEventType | None = None,
    claim_binding: ClaimBinding = _DEFAULT_CLAIM,
    release_lane: ReleaseLane = ReleaseLane.PERSONAL_RESEARCH,
    created_by: str = HUMAN_CREATED_BY,
    created_at: str | None = None,
    code_commit_sha: str = _ZERO_COMMIT,
    software_version: str = _DEFAULT_SOFTWARE,
    id_factory: Any = None,
) -> QCHumanAssessmentResult:
    """Create the NO-predecessor human assessment (valid sequence 1).

    The payload carries ``supersedes_ref`` as a present key with the NULL
    value; no supersession event is produced.
    """

    return _create_human_assessment(
        source,
        correction=False,
        supersedes_ref=None,
        quality_grade=quality_grade,
        flags=flags,
        human_evidence=human_evidence,
        provenance_records=provenance_records,
        artifact_store=artifact_store,
        subject_refs=subject_refs,
        artifact_id=artifact_id,
        aggregate_provenance_id=aggregate_provenance_id,
        human_event_provenance_id=human_event_provenance_id,
        event_type=event_type,
        claim_binding=claim_binding,
        release_lane=release_lane,
        created_by=created_by,
        created_at=created_at,
        code_commit_sha=code_commit_sha,
        software_version=software_version,
        id_factory=id_factory,
    )


def create_human_correction(
    source: QCInput,
    *,
    supersedes_ref: str | None,
    quality_grade: QualityGrade | str,
    flags: Sequence[str],
    human_evidence: HumanEvidence | Any,
    provenance_records: Sequence[ProvenanceRecord] = (),
    artifact_store: DerivedArtifactStore | None = None,
    subject_refs: tuple[str, ...] | None = None,
    artifact_id: str | None = None,
    aggregate_provenance_id: str | None = None,
    human_event_provenance_id: str | None = None,
    event_type: ProvenanceEventType | None = None,
    claim_binding: ClaimBinding = _DEFAULT_CLAIM,
    release_lane: ReleaseLane = ReleaseLane.PERSONAL_RESEARCH,
    created_by: str = HUMAN_CREATED_BY,
    created_at: str | None = None,
    code_commit_sha: str = _ZERO_COMMIT,
    software_version: str = _DEFAULT_SOFTWARE,
    id_factory: Any = None,
) -> QCHumanAssessmentResult:
    """Correct the prior effective assessment (append-only, valid cases 2-8).

    Requires a concrete ``qc_artifact:`` predecessor that is the study's
    current effective assessment; emits the non-null ``supersedes_ref`` and
    appends the store's real SUPERSEDED ``StateEvent`` for the predecessor.
    """

    return _create_human_assessment(
        source,
        correction=True,
        supersedes_ref=supersedes_ref,
        quality_grade=quality_grade,
        flags=flags,
        human_evidence=human_evidence,
        provenance_records=provenance_records,
        artifact_store=artifact_store,
        subject_refs=subject_refs,
        artifact_id=artifact_id,
        aggregate_provenance_id=aggregate_provenance_id,
        human_event_provenance_id=human_event_provenance_id,
        event_type=event_type,
        claim_binding=claim_binding,
        release_lane=release_lane,
        created_by=created_by,
        created_at=created_at,
        code_commit_sha=code_commit_sha,
        software_version=software_version,
        id_factory=id_factory,
    )


def validate_human_assessment(result: QCHumanAssessmentResult) -> bool:
    """Mechanically validate the human views, evidence, and §44 records."""

    if not isinstance(result, QCHumanAssessmentResult):
        raise QCValidationError("the human result type is invalid")
    payload = result.assessment.payload
    document = result.payload_document
    if not isinstance(document, Mapping) or len(document) != PAYLOAD_KEY_COUNT:
        raise QCValidationError("the human payload key set is not closed")
    expected_document = {
        "schema_version": payload.schema_version,
        "scope": payload.scope.value,
        "subject_refs": payload.subject_refs,
        "quality_grade": payload.quality_grade.value,
        "confidence": None,
        "flags": payload.flags,
        "vessel_id": None,
        "region_id": None,
        "producer": payload.producer.value,
        "produced_by_ref": payload.produced_by_ref,
        "supersedes_ref": payload.supersedes_ref,
        "justification_code": payload.justification_code,
        "evidence_refs": payload.evidence_refs,
    }
    if dict(document) != expected_document:
        raise QCValidationError("the human payload DTO and document disagree")
    if payload.producer != ProducerType.HUMAN or result.assessment.producer != ProducerType.HUMAN:
        raise QCValidationError("the human producer binding is invalid")
    if payload.justification_code != HUMAN_JUSTIFICATION_CODE:
        raise QCValidationError("the human justification code is invalid")
    if result.assessment.artifact_ref != f"qc_artifact:{result.artifact.artifact_id}":
        raise QCValidationError("the human views do not share one artifact identity")
    artifact = result.artifact
    if (
        artifact.kind != ArtifactKind.QC_ASSESSMENT
        or artifact.status != ArtifactStatus.ACTIVE
        or artifact.payload.payload_schema != PAYLOAD_SCHEMA
        or artifact.payload.blob_sha256 != canonical_digest(document)
        or artifact.provenance_ref != payload.produced_by_ref
        or artifact.producer.producer_type != ProducerType.HUMAN
        or artifact.quality.quality_grade != payload.quality_grade
        or artifact.quality.flags != payload.flags
        or artifact.review.supersedes_ref != payload.supersedes_ref
        or artifact.review.review_state != ReviewState.PREDICTED
    ):
        raise QCValidationError("the human derived-artifact projection is invalid")
    if artifact.lineage.geometry_base is None or not artifact.lineage.input_volume_refs:
        raise QCValidationError("the human artifact must bind its source geometry")
    evidence_refs = payload.evidence_refs
    if len(evidence_refs) != 4 or len(set(evidence_refs)) != len(evidence_refs):
        raise QCValidationError("the human evidence ref tuple is not the frozen shape")
    if (
        evidence_refs[0] != payload.produced_by_ref
        or not evidence_refs[1].startswith("source_manifest:")
        or not evidence_refs[2].startswith("volume_id:")
        or evidence_refs[3] != f"provenance:{result.human_event_provenance_id}"
    ):
        raise QCValidationError("the human evidence refs are not in the frozen order")

    records = {record.provenance_id: record for record in result.provenance_records}
    if len(records) != len(result.provenance_records):
        raise QCValidationError("the human provenance ids are duplicated")
    aggregate = records.get(result.aggregate_provenance_id)
    human_event = records.get(result.human_event_provenance_id)
    if aggregate is None or human_event is None:
        raise QCValidationError("the human provenance records are unresolved")
    evidence_digest = canonical_digest(result.human_evidence_document)
    if human_event.subject_ref != f"{HUMAN_EVIDENCE_SUBJECT_PREFIX}:{evidence_digest}":
        raise QCValidationError("the human evidence subject digest is not bound")
    for record, upstreams in (
        (aggregate, record_upstreams(aggregate)),
        (human_event, record_upstreams(human_event)),
    ):
        if (
            record.actor != Actor(actor_type=ActorType.HUMAN, actor_id=ACTOR_ID_HUMAN)
            or record.event_type != result.event_type
            or record.preprocessing != QC_PREPROCESSING
            or record.model_record_ref is not None
            or record.parameters_ref != PARAMETERS_NOT_APPLICABLE
            or record.human_edit != result.human_edit
            or isinstance(record.inputs, NotApplicable)
            or not record.inputs
            or not upstreams
            or len(set(upstreams)) != len(upstreams)
            or any(item not in records for item in upstreams)
        ):
            raise QCValidationError("the human §44 provenance branch is invalid")
    if aggregate.subject_ref != result.artifact.artifact_id:
        raise QCValidationError("the aggregate record does not bind the artifact identity")
    if aggregate.inputs != human_event.inputs:
        raise QCValidationError("both human records must bind the same ordered inputs")
    ok, problems = verify_chain(result.provenance_records)
    if not ok:
        raise QCValidationError("the human provenance chain failed: " + "; ".join(problems))
    return True


def record_upstreams(record: ProvenanceRecord) -> tuple[str, ...]:
    """Return the concrete upstream refs of a record (empty for NotApplicable)."""

    if isinstance(record.upstream_refs, NotApplicable):
        return ()
    return tuple(record.upstream_refs)


@dataclass(frozen=True, slots=True)
class _SelectionView:
    ref: str
    study: str
    supersedes_ref: str | None
    status: str
    assessment: QCAssessmentArtifact


def _selection_view(raw: object) -> _SelectionView | None:
    assessment: QCAssessmentArtifact | None
    if isinstance(raw, QCHumanAssessmentResult):
        assessment = raw.assessment
    elif hasattr(raw, "assessment") and isinstance(
        getattr(raw, "assessment"), QCAssessmentArtifact
    ):
        assessment = cast(QCAssessmentArtifact, getattr(raw, "assessment"))
    elif isinstance(raw, QCAssessmentArtifact):
        assessment = raw
    else:
        return None
    if not assessment.payload.subject_refs:
        return None
    return _SelectionView(
        ref=assessment.artifact_ref,
        study=assessment.payload.subject_refs[0],
        supersedes_ref=assessment.payload.supersedes_ref,
        status=str(assessment.status),
        assessment=assessment,
    )


def select_effective_assessment(
    study_ref: str, candidates: Sequence[object]
) -> QCAssessmentArtifact | None:
    """Resolve the effective assessment from the explicit ref graph only.

    Only explicitly superseded records are excluded; incomparable active
    maxima are a conflict (never a timestamp or file-order winner).
    """

    views = [view for view in map(_selection_view, candidates) if view is not None]
    active = [view for view in views if view.status == "ACTIVE" and view.study == study_ref]
    if not active:
        return None
    by_ref = {view.ref: view for view in active}
    superseded = {
        view.supersedes_ref for view in active if view.supersedes_ref in by_ref
    }
    maxima = [view for view in active if view.ref not in superseded]
    if not maxima:
        return None
    if len(maxima) > 1:
        refs = ",".join(sorted(view.ref for view in maxima))
        raise QCValidationError(f"incomparable effective maxima: {refs}")
    return maxima[0].assessment


__all__ = [
    "ACTOR_ID_HUMAN",
    "EVENT_HUMAN_CORRECTION",
    "EVENT_INITIAL_HUMAN",
    "HUMAN_CREATED_BY",
    "HUMAN_EVIDENCE_KIND",
    "HUMAN_EVIDENCE_SUBJECT_PREFIX",
    "HUMAN_JUSTIFICATION_CODE",
    "HumanEvidence",
    "PARAMETERS_NOT_APPLICABLE",
    "PAYLOAD_KEY_COUNT",
    "QCHumanAssessmentResult",
    "QCHumanRefusalCode",
    "QCHumanRefusalError",
    "SOURCE_ROOT_PREFIX",
    "create_human_correction",
    "create_initial_human_assessment",
    "record_upstreams",
    "select_effective_assessment",
    "validate_human_assessment",
]
