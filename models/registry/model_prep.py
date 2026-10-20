"""QC_001-MODEL declaration layer (QC_001_MODEL_PREP D2/D3/D4, milestone #18).

DECLARED-NOT-EXECUTED. QC_001_SPEC Part 4.6 declares the ``QC_001-MODEL``
interface without executing it: no model dependency, model artifact, model
producer, model threshold result, or model claim is part of the closed QC_001
build. Everything in this module is therefore inert by construction (D5):
frozen, slotted dataclasses, pure validation functions, and closed-vocabulary
references only. No network I/O, no new dependency, no executable model path.
This module mints NO provenance record of any kind: provenance reachability is
expressed by digest-bound reference only, and a permanent mechanical guard in
``tests/models/test_model_prep.py`` plus the STRUCTURAL refusal inside the
shared ``tests.core.factories.make_provenance`` helper keep the model path
unable to mint one (r3-F1).

What this module is NOT:
- NOT a ``qc-model-disagreement-v1`` parameter object. That family id stays a
  reserved gate-side STRING with no resolvable parameter object (r8-F3); the
  schema string below is pinned to ``ccta.qc.model_prep.pair_binding.v1``
  (instructions §9.3).
- NOT a provenance record, an artifact, a producer, or a claim: nothing here
  is created, registered, promoted, or recorded by QC_001.
- NOT an authority (instructions §10.4): the pair binding carries digest-bound
  references only — no lifecycle, approval, admissibility, invalidation, or
  "current member" semantics. A reproducible identity is not an admissible,
  current, clinically meaningful result: hashes do not prove approval,
  topology preservation, registry eligibility, or complete downstream
  invalidation, and a faithfully hashed lossy projection is still lossy
  (§10.6).

Fail-closed AT THE VALUE LEVEL (r2 root cause: closed at att0005, re-opened
by r3, re-opened by r4, re-opened AGAIN by r5, and re-opened once more by
r6): att0004 validated in ``__post_init__`` only; att0005 moved validation to
every point of use but derived the self-seal from PUBLIC fields; att0007
moved the seal into a private slot with the authoring surface still
reachable (R4-F1); att0008 guarded the slot with a write-once descriptor
whose OWN retained raw member descriptor (``_member``) was a plain
attribute-chain re-seal/delete handle (R5-F1); att0009 moved the seal bytes
into a module-private vault — and the r6 review showed the vault and the
ever-sealed table were BOTH reachable as a plain closure-cell chain from the
exported ``_SEAL_MACHINERY``, so deleting the id from both and calling the
public ``write_once`` re-blessed a one-field patch of an existing sealed
member (R6-F1).

THE SIX-ROUND LESSON: every stored-checkpoint mechanism left a
caller-reachable handle to the STORED STATE, and each review found it. The
pattern to break is not the handle — it is the STORAGE. att0010 therefore
DELETES the storage instead of guarding it:

- **There is no seal storage of any kind.** No slot, no vault, no table, no
  ``id()`` key, no weakref, no closure state, no ``_SEAL_MACHINERY``.
  ``binding_seal`` is a COMPUTED read-only PROPERTY:
  :func:`_derive_binding_seal` is evaluated at EVERY read over the component's
  CURRENT payload, so there is no cached value to re-bless and no table to
  clear. With nothing stored, the r6 "patch + re-bless" route has no
  blessing to obtain: a patched member simply BECOMES a different value with
  a different (self-consistent) seal and a MOVED ``declaration_sha256``.
- **Defense-in-depth payload write lock.** Each component's payload names
  carry a data descriptor (:class:`_PayloadLock`) installed over the slot's
  member descriptor. The guard stores NO descriptor and no table — only the
  field name; reads and writes go through the module-private
  ``_SLOT_READERS`` registry, and ``__set__``/``__delete__`` perform a
  CALLING-FRAME identity check against the module's own ``__init__`` /
  ``__setstate__`` code objects (captured as plain data at import) before
  they will write; every other caller — ``object.__setattr__`` included —
  is refused with the visible §10.2 reason. Deletion is refused
  unconditionally. The guard is DEFENSE-IN-DEPTH: the architecture's
  correctness does NOT rest on it (see the next bullet and the report's
  bounded-claims section).
- **The core guarantee is the construction-only value flow.** The module
  provides NO sanctioned patch route: payloads are written only by a
  component ``__init__`` (complete restatement) or by a ``__setstate__``
  that accepts ONLY construction-shaped serialized state onto an EMPTY
  target and validates it FULLY before touching a single slot
  (buffer-then-commit — R6-F2). §10.2's "never patch one field" is enforced
  as: no patch route exists.
- **The honest boundary.** In-place payload mutation is not detectable
  without stored state, and six rounds proved every stored state is
  reachable; with no stored seal a patched value also cannot be re-blessed,
  so the result is a NEW IDENTITY — indistinguishable from a fresh
  construction with those values — which the construction path validates
  exactly as far as OFFLINE DERIVATION ALLOWS (the standing att0007
  mutual-consistency residual, unchanged). Per §10.4/§10.6: reproducible
  identity is not admissibility — hashes do not prove approval, current
  membership, or clinical meaning.
- ``__getstate__`` runs the full use-time validation before emitting the
  payload with its (derived) seal, so serialization cannot launder a value
  the validators refuse;
- a caller who assembles a COMPLETE fresh object (all payload fields, through
  the sanctioned construction route) has produced a value value-equal to a
  sanctioned fresh construction — no old digests can be retained, no partial
  patch can survive — the designed offline boundary.

The att0005 layers remain in force on top: exact-type field checks on every
public field, closed types (``__init_subclass__`` refuses subclassing), the
import-time frozen snapshot of the pinned §9.2 mapping for the D3 refusal,
and the digest-time/use-time revalidators (:func:`pair_binding_problems`,
:func:`refusal_problems`).

Round 7 (r7) closed the two single-call routes the r7 review recorded OUTSIDE
that flow: (R7-F1) every component ``__init__`` and the pair's now FIRST check
the instance is EMPTY through :func:`_construction_reentry_problems` — which
routes every populated decision through :func:`_empty_target_problems`, the
ONE shared empty-target predicate (the same module attribute the R6-F2
in-suite mutation probes patch) — and refuses re-entry with the pinned §10.2
complete-fresh-only message, atomically; (R7-F2) the module's own write
helpers :func:`_write_slot`/:func:`_commit_state` are no longer plain patch
routes: the same predicate refuses populated targets and
:func:`_relay_frame_authorized`/:func:`_construction_frame_authorized`
refuse every caller outside the module's sanctioned write relays. The
module-wide slot-writer audit (report §0.0-r7.4) names every slot-touching
surface with its disposition; the disclosed residual remains the
``_SLOT_READERS`` multi-hop chase (now the only write surface past the
gates) plus the identity-not-admissibility boundary (§10.4/§10.6).
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Mapping
from dataclasses import InitVar, dataclass
from types import MappingProxyType
from typing import Final

from core.coordinates import FrameSpec
from core.digest import canonical_digest, is_sha256_hex
from core.enums import QualityGrade
from core.errors import CanonicalValidationError
from core.qc.constants import QCAbstentionCode, QCOutcome

#: The reserved family id (r8-F3). It stays a gate-side STRING with no
#: resolvable parameter object; the schema string below must never
#: equal or embed it — rejected explicitly in the declaration's
#: ``__post_init__`` (r1-F2) and asserted by permanent probes.
RESERVED_FAMILY_ID: Final[str] = "qc-model-disagreement-v1"

#: Pinned schema string for the D2 pair-binding declaration (§9.3). It must
#: never collide with, or reuse, the reserved family id ``qc-model-
#: disagreement-v1``; the non-collision is asserted by a permanent probe.
PAIR_BINDING_SCHEMA: Final[str] = "ccta.qc.model_prep.pair_binding.v1"

#: Name excluded from canonical hashing. Since att0007 the construction seal
#: is NOT a dataclass field, so no seal can appear in the canonical JSON at
#: all; this exclusion is kept as a belt-and-braces invariant so any future
#: field with this name stays out of the self-referential digest surface.
_SEAL_FIELD: Final[str] = "binding_seal"

#: The PRIVATE name of each component's construction seal (r3-F2). Not a
#: dataclass field, not an ``__init__`` parameter, never hashed. Since
#: att0010 it is NOT STORED AT ALL: ``binding_seal`` is a computed property
#: (:func:`_derive_binding_seal` over the current payload, evaluated at every
#: read). The name is retained only so the historical probes and the
#: "no stored seal" invariants stay pinnable: ``_binding_seal`` must never
#: appear in a component's ``__slots__``, in ``__dict__``, or in any table.
_SEAL_SLOT: Final[str] = "_binding_seal"

#: The visible reason a binding value fails use-time/digest-time seal
#: verification (r3-F2): the payload no longer matches the seal derived over
#: it, so the value was modified after construction or the seal was forged.
#: Only a COMPLETE fresh construction of the binding is sanctioned. att0010:
#: the seal is COMPUTED at every read, so this fires only for a value whose
#: payload was written by some route the module did not sanction.
_STALE_SEAL_PROBLEM: Final[str] = (
    "carries a stale self-seal: the payload no longer matches the seal "
    "derived at construction — the value was modified after construction or "
    "the seal was forged; only a COMPLETE fresh construction of the binding "
    "(every component restated together) is sanctioned (§10.2: never patch "
    "one field)"
)

#: The visible reason a PAYFIELD write or delete is refused (R6-F1
#: defense-in-depth): payload fields are construction-only. No caller outside
#: the module's own ``__init__``/``__setstate__`` frames may write one —
#: ``object.__setattr__`` included — so a one-field patch of an existing
#: binding has no sanctioned route at all (§10.2).
_PAYLOAD_WRITE_PROBLEM: Final[str] = (
    "payload fields are construction-only (§10.2: never patch one field): "
    "a binding payload may be written only by the component's own "
    "construction or by a complete construction-shaped deserialization onto "
    "an empty target — an in-place patch of one field is refused, and a "
    "member is replaced only by swapping its COMPLETE binding"
)

#: The visible reason a payload field DELETE is refused (R6-F1): there is no
#: partial removal either — a field is never unbound, in any frame.
_PAYLOAD_DELETE_PROBLEM: Final[str] = (
    "payload fields cannot be deleted (§10.2: never patch one field): a "
    "binding is restated COMPLETE or not at all, so no payload field may be "
    "unbound — not by the construction route, not by deserialization, and "
    "not by any caller"
)

#: The visible reason a deserialization attempt is refused because the state
#: is not deserialization data (R4-F1 requirement A/D): the serialized state
#: must be CONSTRUCTION-SHAPED — the complete payload fields PLUS the seal
#: the state was serialized with. A caller-supplied bare payload tuple is a
#: partial re-bless, not deserialization, and is refused.
_NOT_DESERIALIZATION_PROBLEM: Final[str] = (
    "deserialization state must be CONSTRUCTION-SHAPED (the complete "
    "payload fields plus the seal the state was serialized with); a "
    "caller-supplied payload without its seal is not deserialization data "
    "and cannot re-bless an existing value (R4-F1, §10.2)"
)

#: The visible reason a deserialization attempt is refused because the
#: target is NOT EMPTY (R6-F2 atomicity requirement): restoring state onto a
#: populated object is an in-place mutation, which §10.2 forbids.
_POPULATED_TARGET_PROBLEM: Final[str] = (
    "deserialization target is already populated: state may be restored "
    "only onto an EMPTY (freshly allocated) target — restoring onto an "
    "existing value is an in-place mutation and is refused (§10.2); "
    "validation therefore runs BEFORE any slot is written, so a refused "
    "__setstate__ leaves the target byte-identical (R6-F2)"
)


# ---------------------------------------------------------------------------
# value-level validation primitives (r2: exact types, no normalisation)
# ---------------------------------------------------------------------------


def _present(value: object) -> bool:
    """EXACT non-empty ``str``: a ``str`` SUBCLASS may carry attributes out-
    side the canonical digest (r2-F2), so subclasses are refused outright."""
    return type(value) is str and bool(value.strip())


def _canonical_sha256(value: object) -> bool:
    """EXACT lowercase 64-hex ``str`` (subclass carriers refused, r2-F2)."""
    return type(value) is str and is_sha256_hex(value)


def _field_values(obj: object, names: tuple[str, ...]) -> dict[str, object]:
    """Read the declared fields of *obj*; a missing field is recorded as
    ``None`` so partially-forged objects fail validation with a named
    problem instead of an ``AttributeError``."""
    values: dict[str, object] = {}
    for name in names:
        try:
            values[name] = getattr(obj, name)
        except AttributeError:
            values[name] = None
    return values


# ---------------------------------------------------------------------------
# the CONSTRUCTION-ONLY private seal (r3-F2/F3 single derivation; R4-F1/R5-F1/
# R6-F1: the six-round lesson — DO NOT STORE THE SEAL). att0010 computes the
# seal at every read and deletes every storage apparatus (no slot, no vault,
# no table, no id(), no weakref, no closure state).
# ---------------------------------------------------------------------------

#: Domain separation for the seal derivation (r3-F2). The seal is NOT the
#: documented payload digest: it is derived over a private, domain-separated
#: material, so recomputing the PUBLIC payload digest and offering it to the
#: seal surface does not reproduce the construction seal.
_BINDING_SEAL_DOMAIN: Final[str] = "ccta.qc.model_prep.private-construction-seal.v1"


def _derive_binding_seal(
    label: str, payload_fields: tuple[str, ...], values: Mapping[str, object]
) -> str:
    """The SINGLE private seal derivation (r3-F2) — COMPUTED, never stored.

    Deterministic, domain-separated, and deliberately NOT the publicly
    documented payload digest: a caller who recomputes ``canonical_digest``
    over the payload fields obtains a DIFFERENT value.

    att0010 (R6-F1): this function is now the SEAL ITSELF, evaluated at every
    read of :attr:`binding_seal`. There is nothing to install, nothing to
    re-bless, and no table to clear: for a given payload there is exactly one
    derivable seal, and a payload that was modified in place simply derives
    its OWN (different) seal — which is why the value-level guarantee moved to
    the construction-only value flow (no sanctioned patch route) and to the
    payload write lock below.
    """
    return canonical_digest(
        {
            "seal_domain": _BINDING_SEAL_DOMAIN,
            "component": label,
            "payload": {name: values.get(name) for name in payload_fields},
        }
    )


def component_binding_seal(
    component: object,
    label: str,
    payload_fields: tuple[str, ...],
) -> str:
    """Derive *component*'s seal over its CURRENT payload (att0010).

    The single read path behind the :attr:`binding_seal` property. It reads
    each payload field through the public attribute surface, so a value whose
    payload was written by any route derives the seal that matches WHAT IT
    NOW IS — there is no stored seal to contradict it and therefore no
    "stale seal" state to re-bless (R6-F1).
    """
    return _derive_binding_seal(
        label, payload_fields, {name: getattr(component, name) for name in payload_fields}
    )


# ---------------------------------------------------------------------------
# R6-F1 defense-in-depth: the payload write lock
# ---------------------------------------------------------------------------

#: Per-``(type, name)`` SURVIVING member descriptor, used ONLY to read/write
#: the slot storage underneath the payload lock (see :class:`_PayloadLock`).
#: A module attribute is plain data and is therefore CALLER-REACHABLE — that
#: is disclosed, not waived: a caller who reaches this mapping can write
#: through the member descriptor it holds, bypassing the frame gate. That is a
#: deliberate MULTI-HOP chase (module attribute -> dict -> descriptor ->
#: ``__set__``), versus the r6 route's zero hops after the tables. The guard
#: is DEFENSE-IN-DEPTH: the architecture's correctness does NOT rest on it,
#: because there is no stored seal to re-bless (see :func:`_derive_binding_seal`).
_SLOT_READERS: Final[dict[tuple[type, str], object]] = {}


def _read_slot(obj: object, name: str) -> object:
    """Read the payload slot *name* of *obj* through its member descriptor.

    Types WITHOUT a payload lock (the non-sealed declared types, e.g.
    :class:`PairBindingDeclaration`) fall back to ordinary attribute lookup,
    so the empty-target check below sees their real slot state.
    """
    descriptor = _SLOT_READERS.get((type(obj), name))
    if descriptor is None:
        return object.__getattribute__(obj, name)
    return descriptor.__get__(obj)  # type: ignore[attr-defined]


#: The visible reason a CONSTRUCTOR re-entry is refused on an ALREADY-
#: POPULATED target (R7-F1): construction is complete-fresh ONLY — an
#: existing binding can never be re-constructed in place, no matter how the
#: incoming values compare to the current payload (§10.2: never patch one
#: field; re-derive via a COMPLETE fresh construction). The r7 prompt pins
#: this exact sentence; the enforcement point is the empty-target check
#: itself (the frame gate passing is EXPECTED on this route: the caller
#: legitimately IS calling ``__init__``).
_CONSTRUCTION_REENTRY_PROBLEM: Final[str] = (
    "construction is complete-fresh only; an existing binding cannot be "
    "re-constructed in place (§10.2: never patch one field; re-derive via "
    "a COMPLETE fresh construction)"
)


def _payload_field_names(cls: type) -> tuple[str, ...]:
    """The payload slot names registered for *cls* (R7-F2).

    Derived from the ``_SLOT_READERS`` registry keys (insertion order =
    payload field order), so the whole-target emptiness gate in
    :func:`_commit_state` and the per-slot gate in :func:`_write_slot` read
    exactly the slots the payload lock protects.
    """
    return tuple(name for locked_cls, name in _SLOT_READERS if locked_cls is cls)


def _construction_reentry_problems(
    instance: object, names: tuple[str, ...]
) -> list[str]:
    """The CONSTRUCTION-side empty-target refusal (R7-F1), phrased with the
    §10.2 complete-fresh-only reason.

    THE shared empty-target predicate: every populated decision routes
    through :func:`_empty_target_problems` (the module attribute the in-suite
    R6-F2 mutation probes patch) — one definition of "empty", used by
    ``__init__``, ``__setstate__``, :func:`_commit_state` and
    :func:`_write_slot` alike. Reads go through the same slot readers the
    payload lock uses (:func:`_read_slot` via :func:`_slot_is_populated`).
    """
    populated = [
        name for name in names if _empty_target_problems(instance, (name,))
    ]
    if not populated:
        return []
    return [
        f"{type(instance).__name__} {_CONSTRUCTION_REENTRY_PROBLEM}: "
        f"already populated field(s) {sorted(populated)}"
    ]


def _relay_frame_authorized(caller: object) -> bool:
    """R7-F2 gate of :func:`_write_slot` (module attribute — a PATCHABLE
    seam, exactly like :func:`_empty_target_problems`, so the in-suite
    mutation probes can prove the gate load-bearing).

    True iff the immediate caller frame of ``_write_slot`` is one of the
    module's TWO sanctioned write relays:

    - :func:`_commit_state` — which itself admits only the module's own
      ``__init__``/``__setstate__`` code objects
      (:data:`_AUTHORIZED_PAYLOAD_FRAMES`), and
    - :class:`_PayloadLock.__set__` — which itself admits only the module's
      own construction frames via its calling-frame identity check.

    Every other caller — a plain single-call ``mp._write_slot(...)`` from
    anywhere — is refused with the visible §10.2 reason.
    """
    code = getattr(caller, "f_code", None)
    return code is not None and code in (
        _commit_state.__code__,
        _PayloadLock.__set__.__code__,
    )


def _construction_frame_authorized(caller: object) -> bool:
    """R7-F2 gate of :func:`_commit_state` (module attribute — a PATCHABLE
    seam, exactly like :func:`_empty_target_problems`).

    True iff the immediate caller frame of ``_commit_state`` is one of the
    module's own ``__init__``/``__setstate__`` code objects
    (:data:`_AUTHORIZED_PAYLOAD_FRAMES`, read at call time so the pair's
    frames are included). Every other caller — a plain single-call
    ``mp._commit_state(...)`` from anywhere — is refused with the visible
    §10.2 reason.
    """
    code = getattr(caller, "f_code", None)
    return code is not None and code in _AUTHORIZED_PAYLOAD_FRAMES


def _write_slot(obj: object, name: str, value: object) -> None:
    """Write the payload slot *name* of *obj* through its member descriptor.

    R7-F2: as a plain module attribute this helper IS the underlying write
    path (it resolves the RAW member descriptor through ``_SLOT_READERS``,
    one hop past the frame-gated :class:`_PayloadLock`), so it carries BOTH
    r7 gates on every call:

    1. the shared EMPTY-TARGET predicate
       (:func:`_construction_reentry_problems`, which routes through
       :func:`_empty_target_problems` — the one definition of "empty" the
       in-suite mutation probes patch) — a populated slot is never written;
    2. the RELAY FRAME check (:func:`_relay_frame_authorized`) — only the
       module's two sanctioned write relays may call this helper at all.

    The descriptor written through is the RAW member descriptor recorded in
    ``_SLOT_READERS`` — it does NOT re-enter ``_PayloadLock.__set__`` (no
    self-gate recursion).
    """
    descriptor = _SLOT_READERS.get((type(obj), name))
    if descriptor is None:
        raise CanonicalValidationError(
            [
                f"{type(obj).__name__}.{name} cannot be written: "
                f"{type(obj).__name__} is not a payload-locked component type"
            ]
        )
    reentry = _construction_reentry_problems(obj, (name,))
    if reentry:
        raise CanonicalValidationError(reentry)
    if not _relay_frame_authorized(sys._getframe(1)):
        raise CanonicalValidationError(
            [f"{type(obj).__name__}.{name} {_PAYLOAD_WRITE_PROBLEM}"]
        )
    descriptor.__set__(obj, value)  # type: ignore[attr-defined]


class _PayloadLock:
    """The frame-gated payload write lock (R6-F1, DEFENSE-IN-DEPTH).

    Installed via ``type.__setattr__`` OVER each payload name's original
    member descriptor. The guard instance stores NO descriptor, no table, and
    no reference to any object: ONLY the field name (a plain ``str``). Every
    read/write therefore resolves its descriptor through the module-private
    ``_SLOT_READERS`` registry at call time — so capturing the guard yields
    no write handle at all (the r5/r6 defect pattern: a wrapper that keeps
    the raw descriptor IS the handle).

    ``__set__``/``__delete__`` perform a CALLING-FRAME identity check: the
    caller's ``f_code`` must be one of the module's own ``__init__`` or
    ``__setstate__`` code objects (captured as plain data at import; code
    objects are identity-comparable and are not rebindable per call). Every
    other caller — ``object.__setattr__`` included, which routes through data
    descriptors — is refused with the visible §10.2 reason. Deletion is
    refused unconditionally, in every frame.
    """

    __slots__ = ("_name",)

    #: The payload field name this guard is installed over — the guard's ONLY
    #: state (a plain ``str``; no descriptor, no table, no object reference).
    _name: str
    #: The authorized frames, set at install time (module-own code objects).
    _codes: frozenset[object] = frozenset()

    def __init__(self, name: str) -> None:
        object.__setattr__(self, "_name", name)

    def _authorized(self) -> bool:
        frame = sys._getframe(2)
        return frame is not None and frame.f_code in type(self)._codes

    def __get__(self, obj: object, owner: object = None) -> object:
        if obj is None:
            return self
        return _read_slot(obj, self._name)

    def __set__(self, obj: object, value: object) -> None:
        name = self._name
        if not self._authorized():
            raise CanonicalValidationError(
                [f"{type(obj).__name__}.{name} {_PAYLOAD_WRITE_PROBLEM}"]
            )
        _write_slot(obj, name, value)

    def __delete__(self, obj: object) -> None:
        raise CanonicalValidationError(
            [f"{type(obj).__name__}.{self._name} {_PAYLOAD_DELETE_PROBLEM}"]
        )


def _install_payload_locks(cls: type, names: tuple[str, ...]) -> None:
    """Install a :class:`_PayloadLock` over every payload slot of *cls*.

    The ORIGINAL member descriptor is read once from the class dict, recorded
    in ``_SLOT_READERS`` (the module-private, read/write-through registry the
    guard consults), and then REPLACED in the class dict by the guard — so no
    caller-reachable object holds the raw descriptor as an attribute of the
    guard, and the guard instance itself carries no state but the name.
    """
    for name in names:
        original = cls.__dict__.get(name)
        _SLOT_READERS[(cls, name)] = original
        type.__setattr__(cls, name, _PayloadLock(name))


def _empty_target_problems(instance: object, names: tuple[str, ...]) -> list[str]:
    """Non-empty iff *instance* already carries any of *names* (R6-F2).

    A deserialization target must be an EMPTY (freshly allocated) instance:
    a populated slot means the object is an EXISTING value, and restoring
    state onto it is an in-place mutation (§10.2). This is checked BEFORE any
    slot is written.
    """
    populated = [name for name in names if _slot_is_populated(instance, name)]
    if not populated:
        return []
    return [
        f"{type(instance).__name__} {_POPULATED_TARGET_PROBLEM}: "
        f"already populated field(s) {sorted(populated)}"
    ]


def _commit_state(instance: object, values: Mapping[str, object]) -> None:
    """Assign every payload slot of *instance* (the COMMIT step, R6-F2).

    Called only after the incoming state has been validated IN FULL. R7-F2:
    as a plain module attribute this helper would be a single-call patch
    route onto an existing value, so it carries BOTH r7 gates:

    1. the shared EMPTY-TARGET predicate (:func:`_construction_reentry_problems`
       over the type's registered payload slots — the same
       :func:`_empty_target_problems` definition ``__init__`` and
       ``__setstate__`` use) — a populated component is never committed onto,
       no matter which frame asks;
    2. the FRAME check — the immediate caller's code object must be one of
       the module's own ``__init__``/``__setstate__`` code objects
       (:data:`_AUTHORIZED_PAYLOAD_FRAMES`, read at call time so the pair's
       frames are included).

    Each write then goes through :func:`_write_slot` (which re-asserts the
    per-slot empty-target predicate and its own relay frame gate).
    """
    reentry = _construction_reentry_problems(
        instance, _payload_field_names(type(instance))
    )
    if reentry:
        raise CanonicalValidationError(reentry)
    if not _construction_frame_authorized(sys._getframe(1)):
        raise CanonicalValidationError(
            [f"{type(instance).__name__} {_PAYLOAD_WRITE_PROBLEM}"]
        )
    for name, value in values.items():
        _write_slot(instance, name, value)


def _slot_is_populated(instance: object, name: str) -> bool:
    """True iff *instance*'s payload slot *name* already carries a value.

    A fresh allocation has every slot EMPTY; a populated slot means the
    object is an EXISTING value (R4-F1/R6-F2).
    """
    try:
        _read_slot(instance, name)
    except AttributeError:
        return False
    return True


# ---------------------------------------------------------------------------
# construction / deserialization validation (shared by all three components)
# ---------------------------------------------------------------------------


def _seal_problems(
    label: str, values: Mapping[str, object], payload_fields: tuple[str, ...]
) -> list[str]:
    """USE-TIME/DIGEST-TIME seal verification (r3-F2/r3-F3; att0010).

    The seal in *values* (emitted by a serializer or supplied by a caller)
    must be an exact plain-``str`` 64-hex value equal to the private
    derivation over the payload it accompanies. att0010: the seal on a live
    component is COMPUTED, so a component that reached the validators through
    a sanctioned route always matches; a state whose seal does not match the
    payload it claims was not serialized from a valid value.
    """
    seal = values.get(_SEAL_FIELD)
    if type(seal) is not str or not is_sha256_hex(seal):
        return [
            f"{label} binding seal must be an exact plain-str 64-hex "
            "construction seal derived only by the module's private "
            "derivation: sentinel, empty, forged, and str-subclass carrier "
            "values are refused outright — there is no fresh-on-empty "
            "(r3-F2/r3-F3)"
        ]
    if seal != _derive_binding_seal(label, payload_fields, values):
        return [f"{label} {_STALE_SEAL_PROBLEM}"]
    return []


def _construction_shaped_state(
    type_name: str,
    state: object,
    payload_fields: tuple[str, ...],
    label: str,
    field_problems: Callable[[Mapping[str, object]], list[str]],
) -> dict[str, object]:
    """Validate a CONSTRUCTION-SHAPED deserialization state (R4-F1 A/C/D;
    R6-F2: pure validation — it writes NOTHING).

    Deserialization data is a COMPLETE serialized value: every payload
    field PLUS the seal the state was serialized with. The state's OWN seal
    is verified against the private derivation over the state's own payload —
    a state whose seal does not match the payload it claims was not produced
    by ``__getstate__`` of a valid object and is refused fail-closed. A
    caller-supplied BARE payload (no seal) is exactly the R4-F1 partial
    re-bless and is refused.

    Returns the payload mapping on success. The caller commits it only after
    EVERY check has passed (buffer-then-commit, R6-F2).
    """
    expected_len = len(payload_fields) + 1
    payload: dict[str, object]
    seal: object
    if isinstance(state, Mapping):
        if set(state.keys()) != set(payload_fields) | {_SEAL_FIELD}:
            raise CanonicalValidationError(
                [
                    f"{type_name} {_NOT_DESERIALIZATION_PROBLEM}: a state "
                    f"mapping must carry exactly the declared fields "
                    f"{list(payload_fields)} plus the seal; got "
                    f"{sorted(map(str, state))}"
                ]
            )
        payload = {name: state[name] for name in payload_fields}
        seal = state[_SEAL_FIELD]
    elif isinstance(state, (list, tuple)) and len(state) == expected_len:
        payload = dict(zip(payload_fields, state))
        seal = state[-1]
    else:
        raise CanonicalValidationError(
            [
                f"{type_name} {_NOT_DESERIALIZATION_PROBLEM}: got "
                f"{type(state).__name__}"
                + (f" of length {len(state)}" if hasattr(state, "__len__") else "")
            ]
        )
    # payload field problems first (each named, r2 visibility contract):
    problems = field_problems(payload)
    if problems:
        raise CanonicalValidationError(problems)
    # then the state's OWN seal against the private derivation over the
    # payload it claims (R4-F1: a mismatched seal was not serialized from
    # a valid sealed value), through the SINGLE use-time seal verifier
    # (:func:`_seal_problems`) so there is exactly one seal check in the
    # module — named problems first, then the seal.
    seal_problems = _seal_problems(
        label, {**payload, _SEAL_FIELD: seal}, payload_fields
    )
    if seal_problems:
        raise CanonicalValidationError(
            [
                f"{type_name} {_NOT_DESERIALIZATION_PROBLEM}: the state's "
                "own seal does not match the private derivation over the "
                "payload it claims — the state was not serialized from a "
                "valid sealed value"
                + (f" ({'; '.join(seal_problems)})" if seal_problems else "")
            ]
        )
    return payload




# ---------------------------------------------------------------------------
# D2 — pair-binding declaration object (instructions §9.3 + §10)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, init=False)
class StructureMemberBinding:
    """One complete member binding of a declared structure pair (§10.1-§10.3).

    Identity (``structure_ref``) and payload (``structure_sha256``) are BOTH
    carried — neither alone (§10.1): two different refs may carry equal
    payloads, and one ref may be re-bound. The model-side metadata (release,
    weights digest, adapter, output schema) and the structure's provenance
    are made UNAMBIGUOUSLY reachable by digest-bound reference — the binding
    does not re-state them as a second authority (§10.3): the registry
    release digest seals the frozen identity block that contains
    ``weights.sha256``, ``weights.adapter`` and ``output_contract.schema``.

    Fail-closed AT THE VALUE LEVEL (r1-F2 construction, r2-F2/F4 carriers,
    r3-F2 seal, R4-F1/R5-F1/R6-F1 construction-only authoring): slot-closed
    (no attribute outside the canonical digest can be attached to an exact
    instance; subclasses are refused outright); every public field is
    exact-type checked; the construction seal is COMPUTED at every read
    (:func:`_derive_binding_seal` over the current payload — no slot, no
    vault, no table, nothing stored, so nothing to re-bless, R6-F1); each
    payload field carries a frame-gated write lock
    (:class:`_PayloadLock`, DEFENSE-IN-DEPTH) that refuses every caller
    outside this module's ``__init__``/``__setstate__`` frames; and
    ``__setstate__`` accepts only construction-shaped serialized state,
    validates it IN FULL before touching a slot, and commits onto an EMPTY
    target only (R6-F2). ``dataclasses.replace`` on this type raises (the
    ``_complete_fresh_construction_only`` tripwire), so the ONLY derivation
    route is the complete binding constructed fresh (§10.2) — the fail-closed
    branch, because offline mutual consistency between ``structure_ref`` and
    ``provenance_sha256`` is not derivable in this reference-only layer.
    """

    structure_ref: str
    structure_sha256: str
    model_ref: str
    release_sha256: str
    provenance_ref: str
    provenance_sha256: str
    #: r3-F2 replace()-refusal tripwire: an ``InitVar`` WITHOUT a default is
    #: invisible to ``fields()``/``asdict``/the canonical digest, but
    #: ``dataclasses.replace()`` raises ``ValueError`` unless it is explicitly
    #: passed — and the custom ``__init__`` below accepts no such parameter,
    #: so passing it raises ``TypeError``. Components are therefore derived
    #: ONLY by complete fresh construction, never by patching (§10.2).
    _complete_fresh_construction_only: InitVar[bool]

    __slots__ = (
        "structure_ref",
        "structure_sha256",
        "model_ref",
        "release_sha256",
        "provenance_ref",
        "provenance_sha256",
        # R6-F1: NO seal slot (the seal is COMPUTED, never stored) and NO
        # __weakref__ (nothing holds a weak reference to this value any
        # more: there is no vault, no table and no id-keyed state to sweep).
        # Every slot above is therefore a payload slot, and the payload is
        # the entire instance surface.
    )

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError(
            "subclassing StructureMemberBinding is refused: declaration values "
            "are closed types; a subclass would regain __dict__ and could skip "
            "validation (r2-F2/r2-F3)"
        )

    def __init__(
        self,
        *,
        structure_ref: str,
        structure_sha256: str,
        model_ref: str,
        release_sha256: str,
        provenance_ref: str,
        provenance_sha256: str,
    ) -> None:
        # R7-F1: construction is COMPLETE-FRESH ONLY — refuse re-entry on an
        # already-populated instance (bound or unbound ``__init__`` alike)
        # BEFORE any other work, so an existing binding can never be
        # re-constructed in place (§10.2). The frame gate passing is
        # EXPECTED here (the caller legitimately IS calling ``__init__``);
        # the empty-target check is the enforcement point. No slot has been
        # touched at this point, so the refusal is atomic by construction.
        reentry = _construction_reentry_problems(self, _MEMBER_PAYLOAD_FIELDS)
        if reentry:
            raise CanonicalValidationError(reentry)
        # COMPLETE restatement only (§10.2). The payload write lock admits
        # this frame (an authorized __init__ code object) and refuses every
        # other caller; the seal is not stored, so there is nothing else to
        # write — it is derived on every read of `binding_seal`.
        values: dict[str, object] = {
            "structure_ref": structure_ref,
            "structure_sha256": structure_sha256,
            "model_ref": model_ref,
            "release_sha256": release_sha256,
            "provenance_ref": provenance_ref,
            "provenance_sha256": provenance_sha256,
        }
        problems = _member_field_problems(values)
        if problems:
            raise CanonicalValidationError(problems)
        _commit_state(self, values)

    @property
    def binding_seal(self) -> str:
        """The COMPUTED construction seal — read-only, never stored (R6-F1).

        Evaluated at EVERY read as :func:`_derive_binding_seal` over this
        component's CURRENT payload. There is no cached value, no slot, no
        vault, and no table, so there is no seal state to clear, forge or
        re-bless: a payload that was modified in place simply derives the
        seal that matches what it now is (and thereby moves
        ``declaration_sha256`` — a NEW identity, not a re-blessed one, per
        §10.4/§10.6). It is never an ``__init__`` parameter (so
        ``replace()`` cannot carry or reset it) and never enters
        ``repr``/equality/hashing/``asdict``/the canonical digest.
        """
        return component_binding_seal(
            self, "structure member binding", _MEMBER_PAYLOAD_FIELDS
        )

    def __getstate__(self) -> tuple[object, ...]:
        """Seal-carrying, tamper-refusing serialization (r3-F2/R4-F1).

        The full use-time validation runs BEFORE any state is emitted, so a
        value the validators refuse cannot be laundered through pickle/copy.
        The state is the CONSTRUCTION-SHAPED form: the payload fields plus
        the seal derived over them (``__setstate__`` re-verifies that seal
        against the private derivation over the payload before restoring
        anything).
        """
        problems = _member_use_problems(self)
        if problems:
            raise CanonicalValidationError(problems)
        return tuple(getattr(self, name) for name in _MEMBER_PAYLOAD_FIELDS) + (
            self.binding_seal,
        )

    def __setstate__(self, state: object) -> None:
        """DESERIALIZATION-ONLY state restore — never a resealer (R4-F1 A/C;
        R6-F2: BUFFER-THEN-COMMIT, so a refused call changes NOTHING).

        Accepts ONLY construction-shaped state (the complete payload fields
        PLUS the seal the state was serialized with) and verifies that seal
        against the private derivation over the state's own payload. EVERY
        check — shape, every field problem, and the EMPTY-TARGET requirement
        — runs BEFORE a single slot is written, so a refused ``__setstate__``
        leaves the target 100% unchanged. An existing (populated) component
        can therefore never be mutated through deserialization: restoring is
        for freshly allocated targets only, and a component swap is a
        COMPLETE fresh construction (§10.2).
        """
        values = _construction_shaped_state(
            type(self).__name__,
            state,
            _MEMBER_PAYLOAD_FIELDS,
            "structure member binding",
            _member_field_problems,
        )
        problems = _empty_target_problems(self, _MEMBER_PAYLOAD_FIELDS)
        if problems:
            raise CanonicalValidationError(problems)
        _commit_state(self, values)


_MEMBER_PAYLOAD_FIELDS: Final[tuple[str, ...]] = (
    "structure_ref",
    "structure_sha256",
    "model_ref",
    "release_sha256",
    "provenance_ref",
    "provenance_sha256",
)
#: The read surface the validators hash/verify: the payload fields plus the
#: COMPUTED seal (att0010: ``binding_seal`` is a property, not a field, so
#: it is read through the public attribute surface).
_MEMBER_READ_FIELDS: Final[tuple[str, ...]] = _MEMBER_PAYLOAD_FIELDS + (_SEAL_FIELD,)


def _presence_problem(label: str, field: str, value: object) -> str:
    """The visible reason for a refused ref field (r2-F2): a ``str`` SUBCLASS
    carries attributes outside the canonical digest, so it is named as its
    own defect instead of being lumped in with missing/empty values."""
    if isinstance(value, str) and type(value) is not str:
        return (
            f"{label} {field} must be exactly a plain str: str subclasses "
            "carry attributes outside the canonical digest and are refused "
            "(r2-F2)"
        )
    return f"{label} {field} missing or empty"


def _digest_problem(label: str, field: str, value: object) -> str:
    if isinstance(value, str) and type(value) is not str:
        return (
            f"{label} {field} must be exactly a plain str 64-hex sha256 "
            "digest: str subclasses carry attributes outside the canonical "
            "digest and are refused (r2-F2)"
        )
    return f"{label} {field} must be a 64-hex sha256 digest"


def _member_field_problems(values: Mapping[str, object]) -> list[str]:
    problems: list[str] = []
    for name in ("structure_ref", "model_ref", "provenance_ref"):
        if not _present(values.get(name)):
            problems.append(_presence_problem("structure member binding", name, values.get(name)))
    for name in ("structure_sha256", "release_sha256", "provenance_sha256"):
        if not _canonical_sha256(values.get(name)):
            problems.append(_digest_problem("structure member binding", name, values.get(name)))
    return problems


def _member_use_problems(member: object) -> list[str]:
    """Point-of-use validation of a claimed StructureMemberBinding (r2-F2,
    r3-F2/F3): exact type, exact payload fields, and the privately derived
    construction seal verified over the CURRENT payload. att0010: the seal is
    COMPUTED, so this verifies that the value's payload is one the private
    derivation agrees with — a value whose payload was written by any route
    derives its own seal and therefore passes, which is exactly why the
    value-level guarantee is the construction-only payload flow (no
    sanctioned patch route) plus the frame-gated payload lock.
    """
    if type(member) is not StructureMemberBinding:
        return [
            "structure member binding must be exactly StructureMemberBinding "
            "(subclass carriers are refused)"
        ]
    values = _field_values(member, _MEMBER_READ_FIELDS)
    problems = _member_field_problems(values)
    if problems:
        return problems
    return _seal_problems("structure member binding", values, _MEMBER_PAYLOAD_FIELDS)


@dataclass(frozen=True, init=False)
class SourceVolumeBinding:
    """The SHARED source identity of the declared pair (Part 4.6).

    One canonical source volume ref + the ``FrameSpec`` digest under the
    ACCEPTED rule (see :func:`frame_spec_digest` — the same canonical rule
    the accepted QC pair documents use) + the source-manifest digest that
    keys the future independent pair oracle.

    Fail-closed AT THE VALUE LEVEL (r1-F2 construction, r2-F2/F4 carriers,
    r3-F2 seal, R4-F1/R5-F1/R6-F1 construction-only authoring): slot-closed;
    exact-type field checks; the PRIVATE construction seal is COMPUTED at every
    read (nothing stored — no slot, no vault, no table; nothing to re-bless,
    R6-F1); every payload field carries the frame-gated write lock
    (:class:`_PayloadLock`, defense-in-depth); no sentinel and no
    ``replace()``; and ``__setstate__`` accepts only construction-shaped
    serialized state, validates it IN FULL before touching a slot, and commits
    onto an EMPTY target only (R6-F2).
    """

    volume_ref: str
    frame_spec_sha256: str
    source_manifest_sha256: str
    #: r3-F2 replace()-refusal tripwire (see StructureMemberBinding).
    _complete_fresh_construction_only: InitVar[bool]

    __slots__ = (
        "volume_ref",
        "frame_spec_sha256",
        "source_manifest_sha256",
        # R6-F1: NO seal slot and NO __weakref__ (see
        # StructureMemberBinding.__slots__).
    )

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError(
            "subclassing SourceVolumeBinding is refused: declaration values "
            "are closed types; a subclass would regain __dict__ and could skip "
            "validation (r2-F2/r2-F3)"
        )

    def __init__(
        self,
        *,
        volume_ref: str,
        frame_spec_sha256: str,
        source_manifest_sha256: str,
    ) -> None:
        # R7-F1: construction is COMPLETE-FRESH ONLY — see
        # StructureMemberBinding.__init__ (§10.2; atomic: nothing written yet).
        reentry = _construction_reentry_problems(self, _SOURCE_PAYLOAD_FIELDS)
        if reentry:
            raise CanonicalValidationError(reentry)
        values: dict[str, object] = {
            "volume_ref": volume_ref,
            "frame_spec_sha256": frame_spec_sha256,
            "source_manifest_sha256": source_manifest_sha256,
        }
        problems = _source_field_problems(values)
        if problems:
            raise CanonicalValidationError(problems)
        _commit_state(self, values)

    @property
    def binding_seal(self) -> str:
        """The COMPUTED construction seal — read-only, never stored (R6-F1;
        see StructureMemberBinding.binding_seal)."""
        return component_binding_seal(
            self, "source volume binding", _SOURCE_PAYLOAD_FIELDS
        )

    def __getstate__(self) -> tuple[object, ...]:
        problems = _source_use_problems(self)
        if problems:
            raise CanonicalValidationError(problems)
        return tuple(getattr(self, name) for name in _SOURCE_PAYLOAD_FIELDS) + (
            self.binding_seal,
        )

    def __setstate__(self, state: object) -> None:
        values = _construction_shaped_state(
            type(self).__name__,
            state,
            _SOURCE_PAYLOAD_FIELDS,
            "source volume binding",
            _source_field_problems,
        )
        problems = _empty_target_problems(self, _SOURCE_PAYLOAD_FIELDS)
        if problems:
            raise CanonicalValidationError(problems)
        _commit_state(self, values)


_SOURCE_PAYLOAD_FIELDS: Final[tuple[str, ...]] = (
    "volume_ref",
    "frame_spec_sha256",
    "source_manifest_sha256",
)
_SOURCE_READ_FIELDS: Final[tuple[str, ...]] = _SOURCE_PAYLOAD_FIELDS + (_SEAL_FIELD,)


def _source_field_problems(values: Mapping[str, object]) -> list[str]:
    problems: list[str] = []
    if not _present(values.get("volume_ref")):
        problems.append(
            _presence_problem("source volume binding", "volume_ref", values.get("volume_ref"))
        )
    for name in ("frame_spec_sha256", "source_manifest_sha256"):
        if not _canonical_sha256(values.get(name)):
            problems.append(_digest_problem("source volume binding", name, values.get(name)))
    return problems


def _source_use_problems(source: object) -> list[str]:
    if type(source) is not SourceVolumeBinding:
        return [
            "source volume binding must be exactly SourceVolumeBinding "
            "(subclass carriers are refused)"
        ]
    values = _field_values(source, _SOURCE_READ_FIELDS)
    problems = _source_field_problems(values)
    if problems:
        return problems
    return _seal_problems("source volume binding", values, _SOURCE_PAYLOAD_FIELDS)


@dataclass(frozen=True, init=False)
class ComparatorBinding:
    """Comparator identity as content-addressed values (declared, never run).

    ``parameters_sha256`` content-binds the comparator parameters WITHOUT
    embedding them: a digest reference is not a parameter object, and no
    ``qc-model-disagreement-v1`` parameter object exists or may be created
    (r8-F3).

    Fail-closed AT THE VALUE LEVEL (r1-F2 construction, r2-F2/F4 carriers,
    r3-F2 seal, R4-F1/R5-F1/R6-F1 construction-only authoring): slot-closed;
    exact-type field checks; the PRIVATE construction seal is COMPUTED at every
    read (nothing stored — no slot, no vault, no table; nothing to re-bless,
    R6-F1); every payload field carries the frame-gated write lock
    (:class:`_PayloadLock`, defense-in-depth); no sentinel and no
    ``replace()``; and ``__setstate__`` accepts only construction-shaped
    serialized state, validates it IN FULL before touching a slot, and commits
    onto an EMPTY target only (R6-F2).
    """

    algorithm_id: str
    algorithm_version: str
    parameters_sha256: str
    #: r3-F2 replace()-refusal tripwire (see StructureMemberBinding).
    _complete_fresh_construction_only: InitVar[bool]

    __slots__ = (
        "algorithm_id",
        "algorithm_version",
        "parameters_sha256",
        # R6-F1: NO seal slot and NO __weakref__ (see
        # StructureMemberBinding.__slots__).
    )

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError(
            "subclassing ComparatorBinding is refused: declaration values "
            "are closed types; a subclass would regain __dict__ and could skip "
            "validation (r2-F2/r2-F3)"
        )

    def __init__(
        self,
        *,
        algorithm_id: str,
        algorithm_version: str,
        parameters_sha256: str,
    ) -> None:
        # R7-F1: construction is COMPLETE-FRESH ONLY — see
        # StructureMemberBinding.__init__ (§10.2; atomic: nothing written yet).
        reentry = _construction_reentry_problems(self, _COMPARATOR_PAYLOAD_FIELDS)
        if reentry:
            raise CanonicalValidationError(reentry)
        values: dict[str, object] = {
            "algorithm_id": algorithm_id,
            "algorithm_version": algorithm_version,
            "parameters_sha256": parameters_sha256,
        }
        problems = _comparator_field_problems(values)
        if problems:
            raise CanonicalValidationError(problems)
        _commit_state(self, values)

    @property
    def binding_seal(self) -> str:
        """The COMPUTED construction seal — read-only, never stored (R6-F1;
        see StructureMemberBinding.binding_seal)."""
        return component_binding_seal(
            self, "comparator binding", _COMPARATOR_PAYLOAD_FIELDS
        )

    def __getstate__(self) -> tuple[object, ...]:
        problems = _comparator_use_problems(self)
        if problems:
            raise CanonicalValidationError(problems)
        return tuple(getattr(self, name) for name in _COMPARATOR_PAYLOAD_FIELDS) + (
            self.binding_seal,
        )

    def __setstate__(self, state: object) -> None:
        values = _construction_shaped_state(
            type(self).__name__,
            state,
            _COMPARATOR_PAYLOAD_FIELDS,
            "comparator binding",
            _comparator_field_problems,
        )
        problems = _empty_target_problems(self, _COMPARATOR_PAYLOAD_FIELDS)
        if problems:
            raise CanonicalValidationError(problems)
        _commit_state(self, values)


_COMPARATOR_PAYLOAD_FIELDS: Final[tuple[str, ...]] = (
    "algorithm_id",
    "algorithm_version",
    "parameters_sha256",
)
_COMPARATOR_READ_FIELDS: Final[tuple[str, ...]] = (
    _COMPARATOR_PAYLOAD_FIELDS + (_SEAL_FIELD,)
)


def _comparator_field_problems(values: Mapping[str, object]) -> list[str]:
    problems: list[str] = []
    if not _present(values.get("algorithm_id")):
        problems.append(
            _presence_problem("comparator binding", "algorithm_id", values.get("algorithm_id"))
        )
    if not _present(values.get("algorithm_version")):
        problems.append(
            _presence_problem(
                "comparator binding", "algorithm_version", values.get("algorithm_version")
            )
        )
    if not _canonical_sha256(values.get("parameters_sha256")):
        problems.append(
            _digest_problem(
                "comparator binding", "parameters_sha256", values.get("parameters_sha256")
            )
        )
    return problems


def _comparator_use_problems(comparator: object) -> list[str]:
    if type(comparator) is not ComparatorBinding:
        return [
            "comparator binding must be exactly ComparatorBinding "
            "(subclass carriers are refused)"
        ]
    values = _field_values(comparator, _COMPARATOR_READ_FIELDS)
    problems = _comparator_field_problems(values)
    if problems:
        return problems
    return _seal_problems("comparator binding", values, _COMPARATOR_PAYLOAD_FIELDS)


# ---------------------------------------------------------------------------
# R6-F1: the payload write lock is installed and BOUND to this module's own
# construction/deserialization frames, after all three component types are
# fully defined (the code objects are captured as plain data at import).
# ---------------------------------------------------------------------------

#: The ONLY calling frames allowed to write a payload slot (R6-F1): this
#: module's own component ``__init__`` and ``__setstate__`` code objects.
#: Code objects are identity-comparable and are NOT rebindable per call, so
#: an alias, a rebind, or a wrapper function does not help: a foreign frame
#: (``object.__setattr__`` from a test, a REPL, or another module) is
#: refused with the visible §10.2 reason. This is DEFENSE-IN-DEPTH — see the
#: module docstring: the value-level guarantee does not rest on it, because
#: the seal is computed and cannot be re-blessed.
_AUTHORIZED_PAYLOAD_FRAMES: frozenset[object] = frozenset(
    {
        StructureMemberBinding.__init__.__code__,
        StructureMemberBinding.__setstate__.__code__,
        SourceVolumeBinding.__init__.__code__,
        SourceVolumeBinding.__setstate__.__code__,
        ComparatorBinding.__init__.__code__,
        ComparatorBinding.__setstate__.__code__,
    }
)
_PayloadLock._codes = _AUTHORIZED_PAYLOAD_FRAMES

#: Install the lock over every payload slot of every component (R6-F1). Each
#: original member descriptor is recorded in ``_SLOT_READERS`` and then
#: REPLACED in the class dict, so no guard instance carries it.
_install_payload_locks(StructureMemberBinding, _MEMBER_PAYLOAD_FIELDS)
_install_payload_locks(SourceVolumeBinding, _SOURCE_PAYLOAD_FIELDS)
_install_payload_locks(ComparatorBinding, _COMPARATOR_PAYLOAD_FIELDS)

# The pair-level guard anchor: the in-place-mutation refusal must ship with
# the module (a probe asserts the marker below exists in this source).


@dataclass(frozen=True, slots=True, init=False)
class PairBindingDeclaration:
    """The D2 declaration object (instructions §9.3 + §10).

    Declaration layer ONLY: a frozen value a future paired-structure
    comparison contract will consume. NOT a gate parameter object, NOT a
    provenance record, NOT an artifact. It binds BOTH structure refs
    symmetrically (equal shape, equal rank, no "current best" resolution, no
    second authority), keyed by digests. It carries NO lifecycle, approval,
    admissibility, or invalidation authority (§10.4): a later blocked or
    deprecated member leaves this historical binding intact while the pinned
    D3 refusal prevents admissible current execution. Members may only be
    replaced by swapping the COMPLETE binding — the swapped-in member is
    constructed FRESH (every component restated together) and the pair is
    re-derived with a new :attr:`declaration_sha256`; there is no partial
    member update and no member patching (§10.2, r3-F2: member patching is
    structurally refused at the component type).

    Fail-closed AT THE VALUE LEVEL (r1-F2 construction + r2-F2/F4 + R4-F1
    in-place-mutation refusal): slot-closed; validated at EVERY
    construction path, at DIGEST time (:attr:`declaration_sha256`
    validates before hashing) and at USE time
    (:func:`pair_binding_problems`): the schema string is pinned (the
    RESERVED family id is rejected explicitly), every field is exact-type
    checked, each member must be exactly the declared binding type
    (privately sealed, unmodified), and the pair must bind two DISTINCT
    structure identities (Part 4.6: two explicitly identified structure
    artifacts on the same canonical source volume). ``__setstate__``
    restores serialized state onto a FRESH target only, validating the state
    IN FULL before any slot is written (R6-F2 buffer-then-commit), so a
    component can never be swapped in place through deserialization (R4-F1
    requirement f) and a refused restore leaves the target unchanged.
    """

    first: StructureMemberBinding
    second: StructureMemberBinding
    source: SourceVolumeBinding
    comparator: ComparatorBinding
    schema_version: str = PAIR_BINDING_SCHEMA

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError(
            "subclassing PairBindingDeclaration is refused: declaration values "
            "are closed types; a subclass would regain __dict__ and could skip "
            "validation (r2-F2/r2-F3)"
        )

    def __init__(
        self,
        *,
        first: StructureMemberBinding,
        second: StructureMemberBinding,
        source: SourceVolumeBinding,
        comparator: ComparatorBinding,
        schema_version: str = PAIR_BINDING_SCHEMA,
    ) -> None:
        """R7-F1: construction is COMPLETE-FRESH ONLY — the constructor
        FIRST refuses an ALREADY-POPULATED target (bound or unbound
        ``__init__`` alike) via the shared empty-target predicate
        (:func:`_construction_reentry_problems`, whose decisions route
        through :func:`_empty_target_problems` — the same definition of
        "empty" every other construction/deserialization gate uses). The
        refusal fires BEFORE any field is written, so it is atomic by
        construction (an existing declaration is never touched). The frame
        gate passing on this route is EXPECTED (the caller legitimately IS
        calling ``__init__``); the empty-target check is the enforcement
        point (§10.2: never patch one field; re-derive via a COMPLETE fresh
        construction).

        Validation stays in :meth:`__post_init__` (the r2 construction
        discipline: called after the fields are written, exactly like the
        previously generated dataclass ``__init__``) so the in-suite r2
        mutation probe that disables it keeps demonstrating its defect class.
        """
        reentry = _construction_reentry_problems(self, _PAIR_FIELDS)
        if reentry:
            raise CanonicalValidationError(reentry)
        for name, value in (
            ("first", first),
            ("second", second),
            ("source", source),
            ("comparator", comparator),
            ("schema_version", schema_version),
        ):
            object.__setattr__(self, name, value)
        self.__post_init__()

    def __post_init__(self) -> None:
        problems = _pair_problems(self)
        if problems:
            raise CanonicalValidationError(problems)

    def __getstate__(self) -> dict[str, object]:
        """Declared-fields state (exactly the ``__getstate__`` form)."""
        return {name: object.__getattribute__(self, name) for name in _PAIR_FIELDS}

    def __setstate__(self, state: object) -> None:
        """BUFFER-THEN-COMMIT restore (R6-F2): validate EVERYTHING first.

        The incoming state is normalized, then fully validated (every field
        problem, the pinned schema, the sealed components, distinct
        identities), and the EMPTY-TARGET requirement is checked — BEFORE a
        single slot is written. A refused ``__setstate__`` therefore leaves
        the target 100% unchanged (no half-applied payload), which is the
        non-atomicity r6 recorded on the component path (R6-F2).
        """
        values = _deserialized_state(type(self).__name__, state, _PAIR_FIELDS)
        problems = _pair_field_problems(values)
        populated = _empty_target_problems(self, _PAIR_FIELDS)
        if populated:
            problems = problems + [
                f"{type(self).__name__} deserialization refused on a "
                "populated declaration (R4-F1): an existing pair binding "
                "cannot be mutated in place through __setstate__ — a "
                "component swap is a COMPLETE fresh declaration with a new "
                "declaration_sha256 (§10.2)"
            ]
        if problems:
            raise CanonicalValidationError(problems)
        for name in _PAIR_FIELDS:
            object.__setattr__(self, name, values[name])

    @property
    def declaration_sha256(self) -> str:
        """Content-address of the declaration itself (shared canonical rule).

        DIGEST-TIME enforcement (r2-F2/F4, r3-F2): the declaration is fully
        re-validated — exact types, sealed unmodified members/source/
        comparator (each private seal verified over its CURRENT payload),
        pinned schema, distinct identities — BEFORE any digest is computed.
        A value tampered after construction (slot write, forged seal) cannot
        obtain a canonical seal. Since att0007 the construction seals are not
        dataclass fields, so no seal can appear in the canonical JSON at all;
        the exclusion below documents that invariant permanently.
        """
        problems = _pair_problems(self)
        if problems:
            raise CanonicalValidationError(problems)
        return canonical_digest(self, (_SEAL_FIELD,))


_PAIR_FIELDS: Final[tuple[str, ...]] = (
    "first",
    "second",
    "source",
    "comparator",
    "schema_version",
)

_COMPONENT_FIELD_SPECS: Final[tuple[tuple[str, type, Callable[[object], list[str]]], ...]] = (
    ("first", StructureMemberBinding, _member_use_problems),
    ("second", StructureMemberBinding, _member_use_problems),
    ("source", SourceVolumeBinding, _source_use_problems),
    ("comparator", ComparatorBinding, _comparator_use_problems),
)

#: R6-F1: the PAIR's own fields get the SAME payload lock, and the pair's
#: ``__init__``/``__setstate__`` frames are authorized to write them — so an
#: in-place component swap or schema rewrite on an existing declaration is
#: refused at WRITE time, exactly like a component's payload (§10.2).
_AUTHORIZED_PAYLOAD_FRAMES = _AUTHORIZED_PAYLOAD_FRAMES | frozenset(
    {
        PairBindingDeclaration.__init__.__code__,
        PairBindingDeclaration.__setstate__.__code__,
    }
)
_PayloadLock._codes = _AUTHORIZED_PAYLOAD_FRAMES
_install_payload_locks(PairBindingDeclaration, _PAIR_FIELDS)


def _schema_problems(schema_version: object) -> list[str]:
    problems: list[str] = []
    if type(schema_version) is not str:
        problems.append(
            "pair binding schema_version must be exactly str (subclass "
            "carriers are refused)"
        )
    if schema_version == RESERVED_FAMILY_ID:
        problems.append(
            "pair binding schema_version reuses the RESERVED family id "
            f"{RESERVED_FAMILY_ID!r} (r8-F3): it stays a gate-side string "
            "with no resolvable parameter object and is never reused here"
        )
    elif schema_version != PAIR_BINDING_SCHEMA:
        problems.append(
            "pair binding schema_version must be the pinned "
            f"{PAIR_BINDING_SCHEMA!r}"
        )
    return problems


def _pair_field_problems(values: Mapping[str, object]) -> list[str]:
    problems = _schema_problems(values.get("schema_version"))
    for name, component_type, validator in _COMPONENT_FIELD_SPECS:
        component = values.get(name)
        if type(component) is not component_type:
            problems.append(f"pair binding {name} must be a {component_type.__name__}")
            continue
        problems.extend(validator(component))
    first = values.get("first")
    second = values.get("second")
    if (
        type(first) is StructureMemberBinding
        and type(second) is StructureMemberBinding
        and first.structure_ref == second.structure_ref
    ):
        problems.append(
            "pair binding first and second must bind two DISTINCT structure "
            "identities (Part 4.6: two explicitly identified structure "
            "artifacts on the same canonical source volume); a degenerate "
            "same-identity pair is not a binding and is no swap target"
        )
    return problems


def _pair_problems(pair: object) -> list[str]:
    """Point-of-use validation of a claimed PairBindingDeclaration (r2-F2)."""
    if type(pair) is not PairBindingDeclaration:
        return [
            "pair binding declaration must be exactly PairBindingDeclaration "
            "(subclass carriers are refused)"
        ]
    return _pair_field_problems(_field_values(pair, _PAIR_FIELDS))


def _deserialized_state(
    type_name: str, state: object, names: tuple[str, ...]
) -> dict[str, object]:
    """Normalize a ``__setstate__`` payload to a field-value mapping.

    FAIL-CLOSED (r2-F2): the dataclass-generated ``__setstate__`` silently
    ignored wrong-shaped state (a tuple was applied as a single slot value);
    this override accepts ONLY the exact declared field sequence (the
    ``__getstate__`` form) or an exact field-name mapping — anything else
    raises. Used by the non-sealed declared types (:class:`PairBindingDeclaration`,
    :class:`ModelPrepRefusal`, :class:`ModelPrepCensusDeclaration`); the
    SEALED components use :func:`_construction_shaped_state` (payload +
    the seal the state was serialized with — R4-F1).
    """
    if isinstance(state, Mapping):
        if set(state.keys()) != set(names):
            raise CanonicalValidationError(
                [
                    f"{type_name} deserialization state must carry exactly the "
                    f"declared fields {list(names)}; got {sorted(map(str, state))}"
                ]
            )
        return dict(state)
    if isinstance(state, (list, tuple)) and len(state) == len(names):
        return dict(zip(names, state))
    raise CanonicalValidationError(
        [
            f"{type_name} deserialization state must be the declared "
            f"{len(names)}-field sequence or field mapping; got "
            f"{type(state).__name__}"
        ]
    )


def pair_binding_problems(declaration: object) -> list[str]:
    """USE-TIME validation of a claimed pair-binding declaration (r2-F2/F4,
    r3-F2/F3).

    The value-level contract every consumer must apply before trusting a
    declaration: exact type, pinned schema, exact sealed unmodified
    components (each private construction seal verified over its current
    payload), distinct identities. Returns every problem; empty list means
    the value is a complete, sealed, construction-consistent binding.
    """
    return _pair_problems(declaration)


def create_pair_binding_declaration(
    *,
    first: StructureMemberBinding,
    second: StructureMemberBinding,
    source: SourceVolumeBinding,
    comparator: ComparatorBinding,
) -> PairBindingDeclaration:
    """Supported construction entry point for a pair-binding declaration.

    Validation is fail-closed AT THE VALUE LEVEL (r1-F2 construction, r2-F2/F4,
    r3-F2): it lives in :meth:`PairBindingDeclaration.__post_init__` (the same
    checks that run at digest time and use time), so this factory and direct
    construction are equally refusing — every violated field is named in one
    :class:`CanonicalValidationError`. Computes nothing, runs nothing,
    records nothing.
    """
    return PairBindingDeclaration(
        first=first, second=second, source=source, comparator=comparator
    )


def frame_spec_digest(frame: FrameSpec) -> str:
    """The single ACCEPTED ``FrameSpec`` digest rule: ``canonical_digest(frame)``.

    This is the same rule the accepted QC pair documents apply (the private
    ``core.qc.pair`` frame-digest helper). It is re-exposed here — the rule,
    not a fork — so declaration-side callers cannot invent a second FrameSpec
    digest convention (survey §A5). Rule identity is asserted by a permanent
    probe. The VALUE is validated at the point of use (r2-F2): exactly a
    ``FrameSpec``, not a subclass carrier.
    """
    if type(frame) is not FrameSpec:
        raise CanonicalValidationError(
            "frame_spec_digest requires exactly a core.coordinates FrameSpec "
            "(subclass carriers are refused)"
        )
    return canonical_digest(frame)


# ---------------------------------------------------------------------------
# D3 — pinned fail-closed typed refusals (instructions §9.2 — BINDING)
# ---------------------------------------------------------------------------


#: Verbatim Part 4.6 safety case -> EXISTING closed ``QCAbstentionCode``.
#: The enum is CLOSED: no new code, no rename, no default refusal, no
#: substitution. Outcome for every case is ``QCOutcome.ABSTAINED`` +
#: ``QualityGrade.NOT_ANALYZABLE``.
SAFETY_CASE_ABSTENTION_CODE: Final[Mapping[str, QCAbstentionCode]] = MappingProxyType(
    {
        "Missing model record": QCAbstentionCode.QC_REQUIRED_PAIR_ABSENT,
        "Missing/blocked adapter binding": QCAbstentionCode.QC_REQUIRED_PAIR_INVALID,
        "Missing structure (either side)": QCAbstentionCode.QC_REQUIRED_PAIR_ABSENT,
        "Blocked or deprecated registry member": QCAbstentionCode.QC_SOURCE_IDENTITY_INVALID,
        "Unresolvable upstream provenance": QCAbstentionCode.QC_REGISTRATION_INVALID,
    }
)

#: Import-time FROZEN snapshot of the pinned §9.2 mapping (r2-F3): the public
#: name above stays exported for inspection, but the TYPE validates against
#: this private reference, so rebinding/corrupting the module attribute can
#: never make a refusal express an unknown case or a re-pinned pairing.
_PINNED_CASE_CODE_MAPPING: Final[Mapping[str, QCAbstentionCode]] = (
    SAFETY_CASE_ABSTENTION_CODE
)

#: The five pinned codes as a frozen closed set, computed from the pinned
#: mapping at import. The refusal type refuses any code outside this set —
#: including a code a corrupted (monkeypatched) mapping might supply — so the
#: TYPE cannot express a refusal outside the pinned §9.2 vocabulary (r1-F3).
_PINNED_D3_CODES: Final[frozenset[QCAbstentionCode]] = frozenset(
    _PINNED_CASE_CODE_MAPPING.values()
)

_REFUSAL_FIELDS: Final[tuple[str, ...]] = (
    "safety_case",
    "abstention_code",
    "outcome",
    "quality_grade",
)


@dataclass(frozen=True, slots=True)
class ModelPrepRefusal:
    """A typed D3 refusal for one verbatim Part 4.6 safety case.

    Fail-closed AT THE VALUE LEVEL (r1-F3 construction + r2-F3):
    ``__post_init__`` pins the outcome to ``QCOutcome.ABSTAINED`` and the
    grade to ``QualityGrade.NOT_ANALYZABLE`` (any other value — including a
    non-member — raises), and pins ``abstention_code`` to the pinned §9.2
    pairing and the five EXISTING closed codes via the import-time frozen
    snapshot (any other value, any non-member, and any unknown case — even
    under a corrupted module mapping — raises). The type carries NO
    member/structure field and is slot-closed: member substitution is
    structurally inexpressible. Subclassing is REFUSED (a subclass could
    override ``__post_init__`` and regain ``__dict__``), and
    ``__setstate__`` re-runs the full validation on deserialization state, so
    ``object.__new__`` + state assignment cannot forge a proceed-shaped
    refusal. Consumers re-check ANY refusal at the point of use with
    :func:`refusal_problems`.
    """

    safety_case: str
    abstention_code: QCAbstentionCode
    outcome: QCOutcome = QCOutcome.ABSTAINED
    quality_grade: QualityGrade = QualityGrade.NOT_ANALYZABLE

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError(
            "subclassing ModelPrepRefusal is refused: a subclass could "
            "override __post_init__, regain __dict__, and express a "
            "proceed-shaped refusal (r2-F3)"
        )

    def __post_init__(self) -> None:
        problems = _refusal_field_problems(_field_values(self, _REFUSAL_FIELDS))
        if problems:
            raise CanonicalValidationError(problems)

    def __setstate__(self, state: object) -> None:
        values = _deserialized_state(type(self).__name__, state, _REFUSAL_FIELDS)
        problems = _refusal_field_problems(values)
        if problems:
            raise CanonicalValidationError(problems)
        for name in _REFUSAL_FIELDS:
            object.__setattr__(self, name, values[name])


def _refusal_field_problems(values: Mapping[str, object]) -> list[str]:
    problems: list[str] = []
    case = values.get("safety_case")
    code = values.get("abstention_code")
    outcome = values.get("outcome")
    grade = values.get("quality_grade")
    if not _present(case):
        problems.append("refusal safety_case missing or empty")
    if type(code) is not QCAbstentionCode:
        problems.append(
            "refusal abstention_code must be an EXISTING QCAbstentionCode "
            "member (closed vocabulary: no new code, no default refusal)"
        )
    if type(outcome) is not QCOutcome:
        problems.append("refusal outcome must be an EXISTING QCOutcome member")
    elif outcome != QCOutcome.ABSTAINED:
        problems.append(
            "refusal outcome is pinned to QCOutcome.ABSTAINED; a refusal "
            "cannot be expressed as a proceed (CLEAR/MILD/SEVERE refused)"
        )
    if type(grade) is not QualityGrade:
        problems.append("refusal quality_grade must be an EXISTING QualityGrade member")
    elif grade != QualityGrade.NOT_ANALYZABLE:
        problems.append(
            "refusal quality_grade is pinned to QualityGrade.NOT_ANALYZABLE"
        )
    if type(case) is str and case in _PINNED_CASE_CODE_MAPPING:
        if type(code) is QCAbstentionCode:
            if code not in _PINNED_D3_CODES:
                problems.append(
                    f"refusal abstention_code {code!r} is outside the "
                    "five pinned §9.2 codes; MODEL_PREP refuses any code outside the "
                    "pinned closed mapping"
                )
            elif code != _PINNED_CASE_CODE_MAPPING[case]:
                problems.append(
                    f"refusal abstention_code for {case!r} must be the "
                    f"pinned {_PINNED_CASE_CODE_MAPPING[case]!r}"
                )
    elif type(case) is str:
        problems.append(
            f"unknown MODEL_PREP safety case {case!r}; the pinned "
            "closed mapping admits no default refusal and never substitutes "
            "a member"
        )
    return problems


def refusal_problems(refusal: object) -> list[str]:
    """USE-TIME validation of a claimed ModelPrepRefusal (r2-F3).

    The value-level contract every consumer must apply before trusting a
    refusal: exact type, outcome pinned to ``ABSTAINED``, grade pinned to
    ``NOT_ANALYZABLE``, the abstention code pinned to the import-time §9.2
    snapshot, no unknown case. Returns every problem; empty list means the
    value cannot express a proceed or carry a substitute member.
    """
    if type(refusal) is not ModelPrepRefusal:
        return [
            "MODEL_PREP refusal must be exactly ModelPrepRefusal "
            "(subclass carriers are refused)"
        ]
    return _refusal_field_problems(_field_values(refusal, _REFUSAL_FIELDS))


def refusal_for_case(safety_case: str) -> ModelPrepRefusal:
    """Resolve a verbatim safety case to its pinned refusal — fail-closed.

    An unknown case raises; it is NEVER mapped to a closest-available code
    and NEVER produces a substitute member (no silent fallbacks). The
    resulting refusal is construction-validated against the import-time
    frozen snapshot, so a corrupted module mapping cannot re-pin a pairing
    or admit an unknown case (r2-F3).
    """
    try:
        code = SAFETY_CASE_ABSTENTION_CODE[safety_case]
    except KeyError:
        raise CanonicalValidationError(
            f"unknown MODEL_PREP safety case {safety_case!r}; the pinned "
            "closed mapping admits no default refusal and never substitutes "
            "a member"
        ) from None
    return ModelPrepRefusal(safety_case=safety_case, abstention_code=code)


# ---------------------------------------------------------------------------
# D4 — the N=96 DECLARED census shape (Part 4.6 row; declared-not-executed)
# ---------------------------------------------------------------------------

#: G = 48 geometry/scalar strata — the accepted QC_001-D census strata
#: (2 orientations × 2 spacings × 2 normals × 2 origins × 3 pixel classes).
#: The enumeration itself is proven against the accepted QC_001-D fixture
#: strata by a permanent probe; this module pins only the shape.
MODEL_PREP_GEOMETRY_STRATA_COUNT: Final[int] = 48

#: M = {AGREE, DISAGREE} = 2 (Part 4.6: G=48 × M={AGREE, DISAGREE}=2).
MODEL_PREP_MEMBER_ROLES: Final[tuple[str, ...]] = ("AGREE", "DISAGREE")

#: N = G × M = 96, declared. NO model is run, NO metric is computed, NO
#: result is recorded by QC_001 (D4).
MODEL_PREP_DECLARED_N: Final[int] = 96

#: Declared primary endpoints — the future locked endpoints (Part 4.6).
MODEL_PREP_PRIMARY_ENDPOINTS: Final[tuple[str, ...]] = (
    "model_disagreement_flag_exact_match_rate=1.0",
    "deterministic_comparator_exactness=1.0",
)

#: Declared-only thresholds (Part 4.6, verbatim formulas). STRINGS ONLY:
#: they are never parsed, never evaluated; no TP/TN/FP/FN count exists
#: anywhere in QC_001.
MODEL_PREP_DECLARED_ONLY_THRESHOLDS: Final[tuple[str, ...]] = (
    "sensitivity_MODEL_DISAGREEMENT=TP/(TP+FN)=1.0",
    "specificity_MODEL_DISAGREEMENT=TN/(TN+FP)=1.0",
)

_CENSUS_FIELDS: Final[tuple[str, ...]] = (
    "geometry_strata_count",
    "member_roles",
    "declared_n",
    "primary_endpoints",
    "declared_only_thresholds",
    "executed",
)


def _census_exact_str_elements(
    field: str, values: Mapping[str, object]
) -> list[str]:
    """Every element of a census collection must be exactly a plain ``str``
    (r3-F3): an element-level ``str`` subclass carrier can carry authority
    state without changing collection equality."""
    value = values.get(field)
    if type(value) is not tuple:
        return []
    return [
        f"census {field} elements must be exactly plain str (str-subclass "
        "element carriers are refused, r3-F3)"
        for element in value
        if type(element) is not str
    ]


def _census_field_problems(values: Mapping[str, object]) -> list[str]:
    problems: list[str] = []
    strata = values.get("geometry_strata_count")
    if type(strata) is not int:
        problems.append(
            "census geometry_strata_count must be exactly a plain int "
            "(equality-compatible int subclass carriers are refused, r3-F3)"
        )
    elif strata != MODEL_PREP_GEOMETRY_STRATA_COUNT:
        problems.append(
            "geometry_strata_count must be "
            f"{MODEL_PREP_GEOMETRY_STRATA_COUNT} (G, QC_001_SPEC Part 4.6)"
        )
    roles = values.get("member_roles")
    if type(roles) is not tuple:
        problems.append(
            "census member_roles must be exactly a plain tuple "
            "(equality-compatible tuple subclass carriers are refused, r3-F3)"
        )
    else:
        problems.extend(_census_exact_str_elements("member_roles", values))
        if roles != MODEL_PREP_MEMBER_ROLES:
            problems.append(
                f"member_roles must be {MODEL_PREP_MEMBER_ROLES} (M, Part 4.6)"
            )
    declared_n = values.get("declared_n")
    if type(declared_n) is not int:
        problems.append(
            "census declared_n must be exactly a plain int "
            "(equality-compatible int subclass carriers are refused, r3-F3)"
        )
    elif declared_n != MODEL_PREP_DECLARED_N:
        problems.append(
            f"declared_n must be {MODEL_PREP_DECLARED_N} (N = G × M, Part 4.6)"
        )
    if (
        type(strata) is int
        and type(roles) is tuple
        and type(declared_n) is int
        and declared_n != strata * len(roles)
    ):
        problems.append(
            "declared_n must equal geometry_strata_count × len(member_roles)"
        )
    endpoints = values.get("primary_endpoints")
    if type(endpoints) is not tuple:
        problems.append(
            "census primary_endpoints must be exactly a plain tuple "
            "(equality-compatible tuple subclass carriers are refused, r3-F3)"
        )
    else:
        problems.extend(_census_exact_str_elements("primary_endpoints", values))
        if endpoints != MODEL_PREP_PRIMARY_ENDPOINTS:
            problems.append("primary_endpoints must be the verbatim Part 4.6 endpoints")
    thresholds = values.get("declared_only_thresholds")
    if type(thresholds) is not tuple:
        problems.append(
            "census declared_only_thresholds must be exactly a plain tuple "
            "(equality-compatible tuple subclass carriers are refused, r3-F3)"
        )
    else:
        problems.extend(_census_exact_str_elements("declared_only_thresholds", values))
        if thresholds != MODEL_PREP_DECLARED_ONLY_THRESHOLDS:
            problems.append(
                "declared_only_thresholds must be the verbatim Part 4.6 "
                "declared-only thresholds"
            )
    executed = values.get("executed")
    if type(executed) is not bool or executed is not False:
        problems.append(
            "the QC_001-MODEL census is declared-not-executed; executed must "
            "be exactly False (bool-equal carriers such as 0/1 and bool "
            "subclass values are refused, r3-F3)"
        )
    return problems


@dataclass(frozen=True, slots=True)
class ModelPrepCensusDeclaration:
    """The D4 declared census shape — a declaration, not an executed benchmark.

    Fail-closed against drift: any count, role set, endpoint or threshold
    that does not match the pinned Part 4.6 row is refused, and anything
    other than exactly ``False`` for ``executed`` is refused outright (the
    census is declared, never executed, in QC_001). Every field is
    EXACT-type checked (r3-F3): exactly ``int`` counts, exactly ``tuple``
    collections of exactly plain-``str`` elements, ``executed`` exactly
    ``False`` — no equality-compatible subclass carrier can ride a field.
    ``__setstate__`` re-runs the same validation (r2 hardening):
    deserialization cannot bypass the pins.
    """

    geometry_strata_count: int
    member_roles: tuple[str, ...]
    declared_n: int
    primary_endpoints: tuple[str, ...]
    declared_only_thresholds: tuple[str, ...]
    executed: bool = False

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError(
            "subclassing ModelPrepCensusDeclaration is refused: declaration "
            "values are closed types; a subclass could skip validation "
            "(r2-F2/r2-F3 hardening)"
        )

    def __post_init__(self) -> None:
        problems = _census_field_problems(_field_values(self, _CENSUS_FIELDS))
        if problems:
            raise CanonicalValidationError(problems)

    def __setstate__(self, state: object) -> None:
        values = _deserialized_state(type(self).__name__, state, _CENSUS_FIELDS)
        problems = _census_field_problems(values)
        if problems:
            raise CanonicalValidationError(problems)
        for name in _CENSUS_FIELDS:
            object.__setattr__(self, name, values[name])


#: The pinned census declaration (constructed here, so drift fails at import).
MODEL_PREP_CENSUS: Final[ModelPrepCensusDeclaration] = ModelPrepCensusDeclaration(
    geometry_strata_count=MODEL_PREP_GEOMETRY_STRATA_COUNT,
    member_roles=MODEL_PREP_MEMBER_ROLES,
    declared_n=MODEL_PREP_DECLARED_N,
    primary_endpoints=MODEL_PREP_PRIMARY_ENDPOINTS,
    declared_only_thresholds=MODEL_PREP_DECLARED_ONLY_THRESHOLDS,
)


__all__ = [
    "PAIR_BINDING_SCHEMA",
    "RESERVED_FAMILY_ID",
    "ComparatorBinding",
    "MODEL_PREP_CENSUS",
    "MODEL_PREP_DECLARED_N",
    "MODEL_PREP_DECLARED_ONLY_THRESHOLDS",
    "MODEL_PREP_GEOMETRY_STRATA_COUNT",
    "MODEL_PREP_MEMBER_ROLES",
    "MODEL_PREP_PRIMARY_ENDPOINTS",
    "ModelPrepCensusDeclaration",
    "ModelPrepRefusal",
    "PairBindingDeclaration",
    "SAFETY_CASE_ABSTENTION_CODE",
    "SourceVolumeBinding",
    "StructureMemberBinding",
    "create_pair_binding_declaration",
    "frame_spec_digest",
    "pair_binding_problems",
    "refusal_for_case",
    "refusal_problems",
]
