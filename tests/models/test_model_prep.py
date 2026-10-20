"""QC_001_MODEL_PREP probes (milestone #18, att0010 correction round 6): D1
intake binding presence check, D2 pair-binding declaration (fail-closed BY
CONSTRUCTION), D3 pinned safety refusals (fail-closed BY CONSTRUCTION), the
D5 offline/inert boundary, the r3 correction probes (R3-F1..R3-F3, reworked
to the att0008 semantics where the r4 review falsified them), and the r4
correction probes (R4-F1: the construction-only seal authoring surface).

PROVENANCE DISCIPLINE (r1-F1, binding; r3-F1 structural — r4-confirmed): the
MODEL_PREP path mints NO provenance record of any kind. QC_001-MODEL is
declared-not-executed — no model dependency, model artifact, model producer,
model threshold result, or model claim is part of the closed QC_001
implementation — and a synthetic model-path provenance record is exactly the
invention class milestone #17's r8/r9 rejected. The prohibition is enforced
at THREE independent layers, the first of which is STRUCTURAL and lives in
the shared helper itself (r3-F1: the admitted helper ``tests.core.factories.
make_provenance`` REFUSES the mintable model-path shape by default — a
provenance-domain caller must pass ``allow_model_path_provenance=True``
explicitly, which nothing in the model-prep path ever does):

1. STRUCTURAL (fixture-free, in ``tests/core/factories.py``): the helper
   refuses by VALUE before any record exists — any caller, any harness, any
   plain script gets the same refusal.
2. MECHANICAL (this file): the r1-F1 AST construction guard + static grep
   over the MODEL_PREP code/test paths, the opt-in screen (no call in the
   guarded paths may pass the provenance-domain opt-in), and the guard
   self-checks (anchors + the shared helper's refusal ships with the guard).
3. VALUE-LEVEL: ``_model_path_record_problems`` asserts the RESULT on every
   record the suite actually produces or binds.

Every claimed check carries a NEGATIVE (violated -> refused with the
specific reason visible) and a CONTROL (restored -> reason absent). Enforcement
is mutation-verified: the in-process patches below are permanent.

R4-F1/R5-F1/R6-F1 (value-level construction-only authoring): the r4 review
reproduced that the r3 private-slot seal was still authorable; att0008
guarded it with a write-once slot guard, whose OWN retained raw member
descriptor (``_seal_slot_guard._member``) the r5 review then used as a
no-introspection re-seal/delete handle (R5-F1); att0009 moved the seal bytes
into a module-private vault, and the r6 review reached BOTH seal tables
through a plain closure-cell chain from the exported ``_SEAL_MACHINERY`` —
delete-both + the public ``write_once`` re-blessed a one-field patch of an
existing sealed member (R6-F1).

**THE SIX-ROUND LESSON:** every stored-checkpoint mechanism left a
caller-reachable handle to the STORED STATE, and each review found it. att0010
therefore DELETES THE STORAGE instead of guarding it:

- ``binding_seal`` is a COMPUTED read-only property (:func:`_derive_binding_seal`
  over the component's CURRENT payload, evaluated at every read). No slot, no
  vault, no table, no ``id()`` key, no weakref, no closure state, and no
  ``_SEAL_MACHINERY`` at all — so there is NO SEAL TO RE-BLESS (R6-F1).
- Each payload field carries a frame-gated write lock (``_PayloadLock``) that
  refuses every caller outside this module's ``__init__``/``__setstate__``
  frames — ``object.__setattr__`` included — with the visible §10.2 reason.
  It is DEFENSE-IN-DEPTH: the guard stores NO raw descriptor (capturing it
  yields no handle — the r5/r6 defect pattern), and the architecture's
  correctness does NOT rest on it, because there is no stored seal.
- ``__setstate__`` is BUFFER-THEN-COMMIT (R6-F2): it validates the incoming
  state FULLY and checks the EMPTY-TARGET requirement BEFORE writing a single
  slot, so a refused restore leaves the target 100% unchanged.

The honest boundary (see the report): in-place payload mutation is not
detectable without stored state, and six rounds proved every stored state is
reachable; with no stored seal a patched value cannot be re-blessed, so the
result is a NEW IDENTITY — indistinguishable from a fresh construction with
those values — which the construction path validates exactly as far as
OFFLINE DERIVATION ALLOWS (the standing att0007 mutual-consistency residual).
Per §10.4/§10.6: reproducible identity is not admissibility.
"""


from __future__ import annotations

import ast
import copy
import dataclasses
import pickle
import subprocess
import sys
from dataclasses import fields as dc_fields
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any

import pytest

import models.registry as registry_module
from core.digest import canonical_digest, canonical_json, is_sha256_hex
from core.enums import ActorType, ModelStatus, QualityGrade
from core.errors import CanonicalValidationError, StateTransitionError
from core.events import EventLog
from core.provenance import ProvenanceRecord, verify_chain, verify_self_digest
from core.qc.constants import QCAbstentionCode, QCOutcome
from models.registry import ModelRegistry, intake_binding_presence_problems, model_prep, model_ref
from models.registry.model_prep import (
    _COMPARATOR_PAYLOAD_FIELDS,
    _MEMBER_PAYLOAD_FIELDS,
    _SLOT_READERS,
    _SOURCE_PAYLOAD_FIELDS,
    MODEL_PREP_CENSUS,
    MODEL_PREP_DECLARED_N,
    MODEL_PREP_DECLARED_ONLY_THRESHOLDS,
    MODEL_PREP_GEOMETRY_STRATA_COUNT,
    MODEL_PREP_MEMBER_ROLES,
    MODEL_PREP_PRIMARY_ENDPOINTS,
    PAIR_BINDING_SCHEMA,
    RESERVED_FAMILY_ID,
    SAFETY_CASE_ABSTENTION_CODE,
    ComparatorBinding,
    ModelPrepCensusDeclaration,
    ModelPrepRefusal,
    PairBindingDeclaration,
    SourceVolumeBinding,
    StructureMemberBinding,
    _derive_binding_seal,
    _member_field_problems,
    _member_use_problems,
    create_pair_binding_declaration,
    frame_spec_digest,
    pair_binding_problems,
    refusal_for_case,
    refusal_problems,
)
from models.registry.record import compute_release_digest
from tests.core.factories import make_actor, make_na_provenance, make_provenance
from tests.models.factories import WEIGHTS_BYTES, make_record, sha

# ---------------------------------------------------------------------------
# R1-F1 — the model path mints NO provenance record (mechanical guard)
# ---------------------------------------------------------------------------

#: The provenance-construction API the guard polices. Assembled from two
#: literals so this guard's own source stays clean for the mechanical grep it
#: pins (a guard must never trip on itself).
_MINTING_SYMBOL = "create_" "provenance"

#: The provenance record type (constructed instances are as prohibited as
#: minted ones). Same self-clean assembly.
_RECORD_TYPE = "Provenance" "Record"

#: Structural anchors of the guard itself: if the guard machinery is deleted
#: or renamed away, these disappear from the source and the self-check probe
#: fails — the guard cannot be quietly removed.
_GUARD_ANCHORS = (
    "_MINTING_SYMBOL",
    "_RECORD_TYPE",
    "_provenance_constructing_imports",
    "_call_count",
    "TestR1F1NoModelPathProvenanceInvention",
    # r3-F1 anchors: the structural refusal must ship WITH the guard (it
    # lives in the shared helper, so its presence is anchored there too).
    "_assert_no_model_path_opt_in",
    "TestR3F1StructuralHelperRefusal",
    "_model_path_record_problems",
)

#: The provenance-domain opt-in parameter of the shared helper (assembled from
#: literals so this guard's own source stays clean for the value screens that
#: key on it): NOTHING in the model-prep path may ever pass it — a caller who
#: passes it is exactly a caller minting model-path provenance.
_OPT_IN_KWARG = "allow_model_path_" "provenance"

#: The model-path provenance marker kwarg (same self-clean assembly): passing
#: a model_record_ref to the shared helper is a model-path mint route.
_MODEL_REF_KWARG = "model_" "record_ref"

#: The MODEL_PREP code/test path files the AST guard covers.
_MODEL_PREP_PATH_FILES = (
    Path(model_prep.__file__),
    Path(__file__),
    Path(__file__).parents[1] / "apps" / "test_qc_001_model_prep_census.py",
)


def _provenance_constructing_imports(tree: ast.AST) -> set[str]:
    """Import aliases of the provenance-construction API within one AST.

    Only CONSTRUCTION symbols count: ``verify_self_digest``/``verify_chain``
    are verification APIs and are deliberately callable from these probes
    (they exercise reused, already-existing accepted records).
    """
    aliases: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "core.provenance":
            for alias in node.names:
                if alias.name in {_MINTING_SYMBOL, _RECORD_TYPE}:
                    aliases.add(alias.asname or alias.name)
    return aliases


def _call_count(tree: ast.AST, name: str) -> int:
    """Number of call sites of *name* (plain or attribute) in one AST."""
    return sum(
        1
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Name) and node.func.id == name)
            or (isinstance(node.func, ast.Attribute) and node.func.attr == name)
        )
    )


def _assert_no_construction(tree: ast.AST, path: Path) -> None:
    """Fail with the visible reason if *tree* constructs a provenance record."""
    for alias in sorted(_provenance_constructing_imports(tree)):
        assert _call_count(tree, alias) == 0, (
            f"{path}: '{alias}' (a provenance-construction API) is imported "
            "and/or called in the MODEL_PREP path; the model path mints NO "
            "provenance record (declared-not-executed; the #17 r8/r9 "
            "invention class)"
        )
    assert _call_count(tree, _MINTING_SYMBOL) == 0, (
        f"{path}: a bare qualified construction call is refused"
    )
    assert _call_count(tree, _RECORD_TYPE) == 0, (
        f"{path}: direct provenance-record construction is refused"
    )


def _opt_in_keyword_count(tree: ast.AST, kwarg: str) -> int:
    """Number of call sites passing keyword *kwarg* (any callee) in one AST."""
    return sum(
        1
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        for keyword in node.keywords
        if keyword.arg == kwarg
    )


def _assert_no_model_path_opt_in(tree: ast.AST, path: Path) -> None:
    """r3-F1 dormant-route screen: no call in *tree* passes the
    provenance-domain opt-in.

    The shared helper refuses the mintable model-path shape BY DEFAULT; the
    ONLY way to make it mint is the explicit opt-in — so a call that passes
    the opt-in IS a model-path mint route, dormant or live. This screen
    fires on the SOURCE (no execution needed), which is what detects a
    DORMANT helper-mediated mint planted in the guarded paths.
    """
    assert _opt_in_keyword_count(tree, _OPT_IN_KWARG) == 0, (
        f"{path}: a call passes {_OPT_IN_KWARG!r} — the provenance-domain "
        "opt-in that unmutes the model-path provenance mint; nothing in the "
        "MODEL_PREP path may pass it (the model path mints NO provenance "
        "record; the shared helper refuses the mint by default)"
    )


def _planted_mint_value_problems(tree: ast.AST, path: Path) -> list[str]:
    """The value-level screen for PLANTED mint sources (r3-F1 mutation).

    Unlike :func:`_assert_no_model_path_opt_in` (which runs over the guarded
    paths, where the suite's own fixture-free negative legitimately calls the
    shared helper with model-path VALUES in order to assert the refusal), this
    screen is applied to FOREIGN/planted sources: any call passing the
    opt-in, a ``model_record_ref`` keyword, or a ``subject_ref`` string
    literal with the model-path ``artifact:`` prefix is a model-path mint
    route, dormant or live.
    """
    problems: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for keyword in node.keywords:
            if keyword.arg == _OPT_IN_KWARG:
                problems.append(
                    f"{path}: a call passes the provenance-domain opt-in "
                    f"{_OPT_IN_KWARG!r} (a model-path provenance mint route)"
                )
            if keyword.arg == _MODEL_REF_KWARG:
                problems.append(
                    f"{path}: a call passes {_MODEL_REF_KWARG!r} (the "
                    "model-path provenance marker)"
                )
            if (
                keyword.arg == "subject_ref"
                and isinstance(keyword.value, ast.Constant)
                and isinstance(keyword.value.value, str)
                and keyword.value.value.startswith("artifact:")
            ):
                problems.append(
                    f"{path}: a call passes the model-path subject "
                    f"{keyword.value.value!r}"
                )
    return problems


class TestR1F1NoModelPathProvenanceInvention:
    """The mechanical guard: no MODEL_PREP code/test path mints a record.

    Milestone #17's r8/r9 rejected the invented model-path provenance class;
    r1-F1 caught this suite reintroducing it because the prohibition was
    prose. It is now enforced by permanent, self-checking probes.
    """

    def test_guard_self_check_the_mechanism_cannot_go_quiet(self):
        # The guard must be load-bearing: if its detection machinery were
        # removed from this file, these anchors would disappear and this
        # probe would fail (a guard that can be silently deleted is prose).
        source = Path(__file__).read_text(encoding="utf-8")
        for anchor in _GUARD_ANCHORS:
            assert anchor in source, anchor

    def test_no_provenance_construction_anywhere_in_the_model_prep_path(self):
        # NEGATIVE: no file in the MODEL_PREP code/test path constructs a
        # provenance record — not once, not under any import alias.
        for path in _MODEL_PREP_PATH_FILES:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            _assert_no_construction(tree, path)

    def test_no_model_path_opt_in_anywhere_in_the_model_prep_path(self):
        # r3-F1 NEGATIVE (dormant route): no call in the guarded paths passes
        # the provenance-domain opt-in — the only way to unmute the shared
        # helper's default model-path refusal. A dormant helper-mediated
        # mint planted in these files is detected by THIS screen, source-level,
        # without executing it.
        for path in _MODEL_PREP_PATH_FILES:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            _assert_no_model_path_opt_in(tree, path)

    def test_the_structural_refusal_ships_with_the_guard(self):
        # r3-F1: the behavioural enforcement lives in the SHARED helper
        # (tests/core/factories.py), so it cannot be deleted from this
        # suite's own files without leaving a detectable hole: the shared
        # module must still carry the refusal (marker + screen) and the
        # opt-in parameter. If that machinery is removed, this probe fails.
        import tests.core.factories as factories_module

        source = Path(factories_module.__file__).read_text(encoding="utf-8")
        for anchor in (
            "_model_path_mint_problems",
            "allow_model_path_" "provenance",
            "refuses to mint a MODEL-path provenance",
        ):
            assert anchor in source, (
                f"the shared helper's structural model-path refusal is "
                f"missing its marker {anchor!r}; the refusal must ship with "
                "the helper itself (r3-F1: fixture-free, not test-local)"
            )
        assert not hasattr(factories_module, "_MINTING_SYMBOL")  # self-clean

    def test_product_registry_layer_never_imports_core_provenance(self):
        # The product module (models/registry/**) does not even import from
        # core.provenance — the declaration layer carries digest-bound
        # REFERENCES only; nothing is minted, sealed, or recorded by QC_001.
        pkg_dir = Path(registry_module.__file__).parent
        for py_file in sorted(pkg_dir.glob("*.py")):
            tree = ast.parse(py_file.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    assert not module.startswith("core.provenance"), (
                        f"{py_file}:{node.lineno}: product code imports {module}; "
                        "the MODEL_PREP declaration layer carries digest-bound "
                        "references only"
                    )

    def test_static_grep_is_clean_and_the_detector_fires_on_a_regrowth(self, tmp_path):
        # The dispatch-mandated mechanical grep over the MODEL_PREP paths —
        # run here so it cannot rot; exit code 1 means ZERO matches. The
        # patterns are assembled from the split-literal guard constants so
        # the grep cannot hit this probe's own source.
        result = subprocess.run(
            [
                "grep",
                "-rn",
                "--include=*.py",
                "-e",
                _MINTING_SYMBOL,
                "-e",
                _RECORD_TYPE + "(",
                "models/registry",
                "tests/models/test_model_prep.py",
                "tests/apps/test_qc_001_model_prep_census.py",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 1, f"grep hits:\n{result.stdout}"

        # The detector is demonstrated LIVE on a planted regression planted
        # OUTSIDE the guarded files — the exact r1-F1 pattern (a model-path
        # record mint) — without re-infecting the suite:
        planted = tmp_path / "planted_regression.py"
        planted.write_text(
            f"from core.provenance import {_MINTING_SYMBOL}\n"
            "def regrow():\n"
            f"    return {_MINTING_SYMBOL}(subject_ref='artifact:structure-a')\n",
            encoding="utf-8",
        )
        tree = ast.parse(planted.read_text(encoding="utf-8"))
        aliases = _provenance_constructing_imports(tree)
        assert aliases == {_MINTING_SYMBOL}
        with pytest.raises(AssertionError, match="mints NO"):
            _assert_no_construction(tree, planted)

    def test_this_suite_uses_only_the_shared_non_model_path_producers(self):
        # The ONLY provenance-producing names this suite may reference are the
        # shared non-model-path fixture producers (the repo's own accepted
        # machinery), and they must actually be imported so the digest-bound
        # reference probes stay reachable.
        tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "core.provenance":
                imported.update(alias.name for alias in node.names)
        # verification APIs only — construction APIs are refused above:
        assert imported <= {"verify_self_digest", "verify_chain", _RECORD_TYPE}, imported
        assert "verify_self_digest" in imported  # the reuse probes are live


# ---------------------------------------------------------------------------
# D1 — intake binding presence check (§9.1)
# ---------------------------------------------------------------------------


def make_registry() -> ModelRegistry:
    return ModelRegistry(log=EventLog(), accept_bundle=None)


def register(record=make_record) -> tuple[ModelRegistry, str]:
    """Register one factory record; returns (registry, canonical ref)."""
    registry = make_registry()
    record = record() if callable(record) else record
    registry.register(record)
    return registry, model_ref(record.model_id, record.model_version)


def intake_kwargs(claim_overrides: dict[str, str] | None = None) -> dict[str, Any]:
    return {
        "weights_bytes": WEIGHTS_BYTES,
        "claim_statuses": {"CLM-SEG-001": "threshold_pending", **(claim_overrides or {})},
    }


class TestD1IntakeBindingPresence:
    def _registered_broken(self, **weights_or_schema):
        record = make_record()
        weights = replace(record.weights, **weights_or_schema.get("weights", {}))
        output_contract = replace(
            record.output_contract, **weights_or_schema.get("output_contract", {})
        )
        record = replace(record, weights=weights, output_contract=output_contract)
        registry = make_registry()
        registry.register(record)
        return registry, model_ref(record.model_id, record.model_version)

    # -- negatives: the pinned problem string naming the field is visible ----

    @pytest.mark.parametrize("adapter", ["", "   ", None])
    def test_missing_or_empty_adapter_refused(self, adapter):
        registry, ref = self._registered_broken(weights={"adapter": adapter})
        with pytest.raises(CanonicalValidationError) as excinfo:
            registry.pass_intake(ref, **intake_kwargs())
        assert "weights.adapter binding missing or empty" in excinfo.value.violations

    @pytest.mark.parametrize("schema", ["", "   ", None])
    def test_missing_or_empty_output_schema_refused(self, schema):
        registry, ref = self._registered_broken(output_contract={"schema": schema})
        with pytest.raises(CanonicalValidationError) as excinfo:
            registry.pass_intake(ref, **intake_kwargs())
        assert (
            "output_contract.schema binding missing or empty" in excinfo.value.violations
        )

    # -- control: restored bindings -> the pinned problems are absent --------

    def test_control_restored_bindings_pass_intake(self):
        registry, ref = register()
        assert intake_binding_presence_problems(make_record()) == []
        updated = registry.pass_intake(ref, **intake_kwargs())
        assert updated.status == ModelStatus.CHALLENGER

    def test_control_restored_via_standalone_precheck(self):
        record = make_record()
        assert intake_binding_presence_problems(record) == []

    # -- composition: collects ALL problems; existing strings unchanged ------

    def test_binding_problem_composes_with_existing_problems(self):
        registry, ref = self._registered_broken(weights={"adapter": ""})
        with pytest.raises(CanonicalValidationError) as excinfo:
            registry.pass_intake(ref, **intake_kwargs({"CLM-SEG-001": "draft"}))
        violations = excinfo.value.violations
        assert "weights.adapter binding missing or empty" in violations
        # pre-existing problem strings keep their exact meaning (§9.1):
        assert "claim CLM-SEG-001 below threshold_pending (got draft)" in violations

    # -- reused existing carriers (probe, unchanged semantics) ---------------

    def test_existing_carrier_terminal_for_intake_unchanged(self):
        registry, ref = register()
        registry.deprecate(ref, reason="test")
        with pytest.raises(StateTransitionError):
            registry.pass_intake(ref, **intake_kwargs())

    def test_existing_carrier_weights_mismatch_unchanged(self):
        registry, ref = register()
        with pytest.raises(CanonicalValidationError) as excinfo:
            registry.pass_intake(
                ref, weights_bytes=b"other-bytes", claim_statuses=intake_kwargs()["claim_statuses"]
            )
        assert any(
            v.startswith("weights checksum mismatch on disk:")
            for v in excinfo.value.violations
        )

    # -- mutation verification: enforcement removed -> probe FAILS ----------

    def test_d1_presence_check_is_load_bearing(self, monkeypatch):
        registry, ref = self._registered_broken(weights={"adapter": ""})
        monkeypatch.setattr(
            registry_module, "intake_binding_presence_problems", lambda record: []
        )
        # enforcement removed -> the negative probe's refusal is GONE:
        updated = registry.pass_intake(ref, **intake_kwargs())
        assert updated.status == ModelStatus.CHALLENGER
        monkeypatch.undo()
        # restored -> the same input is refused again:
        registry2, ref2 = self._registered_broken(weights={"adapter": ""})
        with pytest.raises(CanonicalValidationError) as excinfo:
            registry2.pass_intake(ref2, **intake_kwargs())
        assert "weights.adapter binding missing or empty" in excinfo.value.violations


# ---------------------------------------------------------------------------
# D2 — pair-binding declaration object (§9.3 + §10) — fail-closed BY
# CONSTRUCTION: direct construction is exactly as refusing as the factory
# ---------------------------------------------------------------------------


def _slot_writer(cls: type, name: str) -> Any:
    """The member descriptor for *cls*'s payload slot *name* — the DISCLOSED
    `_SLOT_READERS` residual route (mutation probes ONLY).

    att0010: the payload lock deliberately keeps NO raw descriptor (that was
    the r5/r6 defect pattern), so the only route to the underlying slot
    writer is the module-private ``_SLOT_READERS`` mapping: a plain module
    attribute -> dict -> member descriptor -> ``__set__``. That is a
    DELIBERATE MULTI-HOP CHASE, and its honest tier is disclosed in the
    module docstring and the report (it is not debugger-tier and not
    introspection: it is a plain chain). The NEGATIVE tests prove no
    one-hop route writes a payload field; this helper is used ONLY by the
    mutation probes that must disable a guard to prove it is load-bearing.
    """
    return _SLOT_READERS[(cls, name)]


def _write_payload_raw(obj: object, name: str, value: object) -> None:
    """Write *obj*'s payload slot *name* through the disclosed residual
    (mutation probes ONLY; see :func:`_slot_writer`)."""
    _slot_writer(type(obj), name).__set__(obj, value)


def _member_use_problems_fresh(obj: object) -> bool:
    """True iff *obj* still has an EMPTY payload slot (i.e. a refused
    ``__setstate__`` wrote nothing — R6-F2 buffer-then-commit)."""
    return any(
        not _slot_is_populated(obj, name) for name in _MEMBER_PAYLOAD_FIELDS
    )


def _slot_is_populated(obj: object, name: str) -> bool:
    """True iff *obj*'s payload slot *name* carries a value."""
    try:
        _slot_writer(type(obj), name).__get__(obj)
    except AttributeError:
        return False
    return True


def _current_frame_code() -> object:
    """The code object of THIS helper's caller's frame — used by the payload
    lock mutation probe to authorize a foreign frame (mutation ONLY)."""
    return sys._getframe(1).f_code


def member(structure_ref: str = "artifact:structure-a", digest: str | None = None):
    return StructureMemberBinding(
        structure_ref=structure_ref,
        structure_sha256=digest if digest is not None else sha(structure_ref),
        model_ref="coronary-seg@1.2.0",
        release_sha256=sha("release-a"),
        provenance_ref=f"provenance:{structure_ref}",
        provenance_sha256=sha(f"prov-{structure_ref}"),
    )


def source() -> SourceVolumeBinding:
    return SourceVolumeBinding(
        volume_ref="volume_id:01MODELPREPVOLUME00000000000",
        frame_spec_sha256=sha("frame"),
        source_manifest_sha256=sha("manifest"),
    )


def comparator() -> ComparatorBinding:
    return ComparatorBinding(
        algorithm_id="structure-comparator",
        algorithm_version="1",
        parameters_sha256=sha("params"),
    )


def declaration(**overrides) -> PairBindingDeclaration:
    kwargs = dict(
        first=member("artifact:structure-a"),
        second=member("artifact:structure-b"),
        source=source(),
        comparator=comparator(),
    )
    kwargs.update(overrides)
    return create_pair_binding_declaration(**kwargs)


class TestD2PairBindingDeclaration:
    def test_schema_string_is_pinned(self):
        assert PAIR_BINDING_SCHEMA == "ccta.qc.model_prep.pair_binding.v1"

    def test_schema_string_is_not_the_reserved_family_id(self):
        # r8-F3: the reserved family id must stay a gate-side STRING with no
        # resolvable parameter object; the declaration neither reuses it nor
        # carries it anywhere.
        serialized = repr(dataclasses.asdict(declaration()))
        assert RESERVED_FAMILY_ID not in serialized

    def test_valid_declaration_seals_content_address(self):
        d = declaration()
        assert is_sha256_hex(d.declaration_sha256)
        # r3-F2: the construction seal is NOT a dataclass field, so no seal
        # can appear in the canonical JSON at all — the canonical digest
        # surface is exactly the declared fields. The historical exclusion
        # rule (att0005) is kept in the property for belt-and-braces and is
        # a no-op now that no field carries a seal (asserted below).
        assert d.declaration_sha256 == canonical_digest(d, ("binding_seal",))
        assert d.declaration_sha256 == canonical_digest(d)
        for component in (d.first, d.second, d.source, d.comparator):
            assert "binding_seal" not in canonical_json(component)
            assert "_binding_seal" not in canonical_json(component)

    def test_negative_missing_refs_refused_with_named_field(self):
        # r1-F2 form: the COMPONENT refuses at construction; the field is
        # named in the visible reason.
        with pytest.raises(CanonicalValidationError) as excinfo:
            member(structure_ref="")
        assert (
            "structure member binding structure_ref missing or empty"
            in excinfo.value.violations
        )

    @pytest.mark.parametrize(
        ("broken", "label"),
        [
            (lambda m: replace(m, structure_sha256="nope"), "structure_sha256"),
            (lambda m: replace(m, release_sha256="nope"), "release_sha256"),
            (lambda m: replace(m, provenance_sha256="nope"), "provenance_sha256"),
        ],
    )
    def test_negative_bad_digests_refused_with_named_field(self, broken, label):
        # r3-F2: dataclasses.replace on a component is STRUCTURALLY refused
        # (the only member route is complete fresh construction), so the
        # invalid digests are planted by a COMPLETE FRESH CONSTRUCTION — the
        # component validation fires at construction with the field named:
        with pytest.raises(CanonicalValidationError) as excinfo:
            StructureMemberBinding(
                structure_ref="artifact:structure-a",
                structure_sha256=(
                    "nope" if label == "structure_sha256" else sha("artifact:structure-a")
                ),
                model_ref="coronary-seg@1.2.0",
                release_sha256=(
                    "nope" if label == "release_sha256" else sha("release-a")
                ),
                provenance_ref="provenance:artifact:structure-a",
                provenance_sha256=(
                    "nope" if label == "provenance_sha256"
                    else sha("prov-artifact:structure-a")
                ),
            )
        assert (
            f"structure member binding {label} must be a 64-hex sha256 digest"
            in excinfo.value.violations
        )

    def test_negative_bad_source_and_comparator_digests_refused(self):
        with pytest.raises(CanonicalValidationError) as excinfo:
            SourceVolumeBinding(
                volume_ref="volume_id:01MODELPREPVOLUME00000000000",
                frame_spec_sha256="nope",
                source_manifest_sha256=sha("manifest"),
            )
        assert (
            "source volume binding frame_spec_sha256 must be a 64-hex sha256 digest"
            in excinfo.value.violations
        )
        with pytest.raises(CanonicalValidationError) as excinfo:
            ComparatorBinding(
                algorithm_id="structure-comparator",
                algorithm_version="1",
                parameters_sha256="nope",
            )
        assert (
            "comparator binding parameters_sha256 must be a 64-hex sha256 digest"
            in excinfo.value.violations
        )

    def test_control_restored_fields_accepted(self):
        d = declaration()
        assert d.schema_version == PAIR_BINDING_SCHEMA

    def test_identity_and_payload_are_both_carried_neither_alone(self):
        # §10.1: two DIFFERENT refs may carry EQUAL payloads; the binding
        # carries both facts and neither alone.
        d = declaration(
            first=member("artifact:structure-a"),
            second=member("artifact:structure-b", digest=sha("artifact:structure-a")),
        )
        assert d.first.structure_ref != d.second.structure_ref
        assert d.first.structure_sha256 == d.second.structure_sha256

    def test_swap_is_complete_binding_swap_with_new_content_address(self):
        # §10.2: replacement = a new complete frozen value; no partial update.
        d = declaration()
        swapped = replace(d, first=member("artifact:structure-c"))
        assert swapped.declaration_sha256 != d.declaration_sha256
        assert d.first.structure_ref == "artifact:structure-a"  # original intact
        with pytest.raises(dataclasses.FrozenInstanceError):
            d.first = member()  # type: ignore[misc]
        with pytest.raises(dataclasses.FrozenInstanceError):
            d.comparator.algorithm_id = "other"  # type: ignore[misc]

    def test_no_partial_member_update_or_current_resolution_api(self):
        # §10.2/§10.4: no partial update, no "latest member" resolution, no
        # lifecycle/approval/invalidation authority on the module surface.
        banned = ("current", "resolve", "approve", "invalidate", "substitute", "promote")
        public = {name for name in dir(model_prep) if not name.startswith("_")}
        assert not any(b in name for b in banned for name in public)


class TestR1F2DeclarationFailClosedByConstruction:
    """r1-F2: the exported dataclasses are not bypassable.

    Direct construction is EXACTLY as refusing as the supported factory —
    validation lives in ``__post_init__`` — and the types are slot-closed, so
    no instance can carry anything outside the canonical digest (§10.4).
    """

    # -- (a) the reserved family id / any unpinned schema is REJECTED --------

    @pytest.mark.parametrize(
        "schema",
        ["qc-model-disagreement-v1", "ccta.qc.model_prep.pair_binding.v2", "", "   "],
    )
    def test_negative_reserved_or_unpinned_schema_refused_on_direct_construction(
        self, schema
    ):
        # DIRECT construction (the bypass r1 demonstrated) — no factory call
        # anywhere in this probe:
        with pytest.raises(CanonicalValidationError) as excinfo:
            PairBindingDeclaration(
                first=member("artifact:structure-a"),
                second=member("artifact:structure-b"),
                source=source(),
                comparator=comparator(),
                schema_version=schema,
            )
        if schema == RESERVED_FAMILY_ID:
            assert "RESERVED family id" in excinfo.value.violations[0]
        else:
            assert "schema_version must be the pinned" in excinfo.value.violations[0]

    def test_control_supported_schema_accepted_on_direct_construction(self):
        # CONTROL: the pinned schema constructs identically either way.
        direct = PairBindingDeclaration(
            first=member("artifact:structure-a"),
            second=member("artifact:structure-b"),
            source=source(),
            comparator=comparator(),
        )
        via_factory = declaration()
        assert direct == via_factory
        assert direct.declaration_sha256 == via_factory.declaration_sha256

    # -- (b) invalid bindings are rejected at construction -------------------

    @pytest.mark.parametrize(
        ("broken", "label"),
        [
            (lambda: StructureMemberBinding(
                structure_ref="artifact:structure-a",
                structure_sha256="nope",
                model_ref="coronary-seg@1.2.0",
                release_sha256=sha("release-a"),
                provenance_ref="provenance:artifact:structure-a",
                provenance_sha256=sha("prov-artifact:structure-a"),
            ), "structure_sha256"),
            (lambda: StructureMemberBinding(
                structure_ref="artifact:structure-a",
                structure_sha256=sha("artifact:structure-a"),
                model_ref="coronary-seg@1.2.0",
                release_sha256="",
                provenance_ref="provenance:artifact:structure-a",
                provenance_sha256=sha("prov-artifact:structure-a"),
            ), "release_sha256"),
            (lambda: StructureMemberBinding(
                structure_ref="artifact:structure-a",
                structure_sha256=sha("artifact:structure-a"),
                model_ref="   ",
                release_sha256=sha("release-a"),
                provenance_ref="provenance:artifact:structure-a",
                provenance_sha256=sha("prov-artifact:structure-a"),
            ), "model_ref"),
            (lambda: StructureMemberBinding(
                structure_ref="artifact:structure-a",
                structure_sha256=sha("artifact:structure-a"),
                model_ref="coronary-seg@1.2.0",
                release_sha256=sha("release-a"),
                provenance_ref=None,  # type: ignore[arg-type]
                provenance_sha256=sha("prov-artifact:structure-a"),
            ), "provenance_ref"),
            (
                lambda: SourceVolumeBinding(
                    volume_ref="volume_id:01MODELPREPVOLUME00000000000",
                    frame_spec_sha256=sha("frame"),
                    source_manifest_sha256="deadbeef",
                ),
                "source_manifest_sha256",
            ),
            (lambda: SourceVolumeBinding(
                volume_ref="",
                frame_spec_sha256=sha("frame"),
                source_manifest_sha256=sha("manifest"),
            ), "volume_ref"),
            (
                lambda: ComparatorBinding(
                    algorithm_id="structure-comparator",
                    algorithm_version="1",
                    parameters_sha256="0" * 63,
                ),
                "parameters_sha256",
            ),
            (lambda: ComparatorBinding(
                algorithm_id=None,  # type: ignore[arg-type]
                algorithm_version="1",
                parameters_sha256=sha("params"),
            ), "algorithm_id"),
        ],
    )
    def test_negative_invalid_component_refused_at_construction(self, broken, label):
        with pytest.raises(CanonicalValidationError) as excinfo:
            broken()  # the COMPONENT type itself refuses — no declaration needed
        assert label in excinfo.value.violations[0]

    def test_negative_non_binding_member_type_refused(self):
        # A declaration cannot be assembled from non-binding stand-ins:
        with pytest.raises(CanonicalValidationError) as excinfo:
            PairBindingDeclaration(
                first="artifact:structure-a",  # type: ignore[arg-type]
                second=member("artifact:structure-b"),
                source=source(),
                comparator=comparator(),
            )
        assert "pair binding first must be a StructureMemberBinding" in (
            excinfo.value.violations[0]
        )
        with pytest.raises(CanonicalValidationError) as excinfo:
            PairBindingDeclaration(
                first=member("artifact:structure-a"),
                second=member("artifact:structure-b"),
                source=object(),  # type: ignore[arg-type]
                comparator=comparator(),
            )
        assert "pair binding source must be a SourceVolumeBinding" in (
            excinfo.value.violations[0]
        )

    def test_negative_degenerate_same_identity_pair_refused(self):
        # Part 4.6: TWO explicitly identified structure artifacts on the same
        # canonical source volume. A same-identity pair is not a binding and
        # is exactly the seam a deferred/partial patch would fake a swap
        # through — refused at construction.
        with pytest.raises(CanonicalValidationError) as excinfo:
            declaration(second=member("artifact:structure-a"))
        assert "two DISTINCT structure identities" in excinfo.value.violations[0]
        with pytest.raises(CanonicalValidationError):
            replace(declaration(), second=declaration().first)

    # -- (c) a partial member patch cannot hide inside the declaration -------

    def test_negative_partial_member_patch_refused_at_construction(self):
        # §10.2 + r3-F2: there is no partial member update, and the patch
        # route itself is now STRUCTURALLY refused — dataclasses.replace on
        # a component raises before any value exists, so a patched member
        # cannot come into existence and cannot be swapped into any
        # declaration. The invalid-value side is separately refused at the
        # component construction (the bad-digest probes above).
        d = declaration()
        with pytest.raises(ValueError, match="_complete_fresh_construction_only"):
            replace(d.first, release_sha256="nope")  # type: ignore[call-arg]
        # and a valid-looking one-field patch is refused equally:
        with pytest.raises(ValueError, match="_complete_fresh_construction_only"):
            replace(d.first, release_sha256=sha("release-c"))  # type: ignore[call-arg]
        assert pair_binding_problems(d) == []

    def test_a_valid_member_change_is_a_new_complete_binding_with_a_new_address(self):
        # §10.2, CONTROL side (r2-F4 rework): the ONLY replacement is the
        # swap of a COMPLETE binding — constructed FRESH (all six fields
        # restated), never patched out of an existing sealed value.
        d = declaration()
        fresh = StructureMemberBinding(
            structure_ref="artifact:structure-a",
            structure_sha256=sha("artifact:structure-a"),
            model_ref="coronary-seg@1.2.0",
            release_sha256=sha("release-c"),
            provenance_ref="provenance:artifact:structure-a",
            provenance_sha256=sha("prov-artifact:structure-a"),
        )
        swapped = replace(d, first=fresh)
        assert swapped.first.release_sha256 == sha("release-c")
        assert swapped.declaration_sha256 != d.declaration_sha256
        assert d.first.release_sha256 == sha("release-a")  # original intact
        assert pair_binding_problems(swapped) == []

    def test_negative_frozen_instance_patch_refused(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            declaration().first = member()  # type: ignore[misc]

    # -- (d) NO attribute outside the canonical digest (slot closure) --------

    @staticmethod
    def _instance_of(cls: type) -> object:
        if cls is PairBindingDeclaration:
            return declaration()
        if cls is StructureMemberBinding:
            return member()
        if cls is SourceVolumeBinding:
            return source()
        if cls is ComparatorBinding:
            return comparator()
        if cls is ModelPrepRefusal:
            return refusal_for_case("Missing model record")
        if cls is ModelPrepCensusDeclaration:
            return MODEL_PREP_CENSUS
        raise AssertionError(f"uncovered class {cls}")

    @staticmethod
    def _refuse_attribute_write(instance: object, attr: str) -> None:
        """Assert an attribute write is structurally refused on a slotted type.

        Plain ``setattr`` raises ``TypeError`` (no ``__dict__``, unknown
        slot); the ``object.__setattr__`` bypass raises ``AttributeError``.
        Both rejections must hold.
        """
        with pytest.raises((TypeError, AttributeError)):
            setattr(instance, attr, True)
        with pytest.raises(AttributeError):
            object.__setattr__(instance, attr, True)

    @pytest.mark.parametrize(
        "cls",
        [
            PairBindingDeclaration,
            StructureMemberBinding,
            SourceVolumeBinding,
            ComparatorBinding,
            ModelPrepRefusal,
            ModelPrepCensusDeclaration,
        ],
    )
    def test_negative_injected_attribute_refused_outside_the_digest(self, cls):
        # §10.4: the binding must stay state-free — no approval, invalidation
        # or "current" state may ride on an instance OUTSIDE the canonical
        # digest. Slots close the plain write AND the object.__setattr__
        # bypass (no __dict__ exists to fall back to).
        instance = self._instance_of(cls)
        assert not hasattr(instance, "__dict__"), cls.__name__
        for attr in ("approved", "invalidated", "current_member", "status"):
            self._refuse_attribute_write(instance, attr)

    def test_control_slot_closure_matches_the_declared_field_set(self):
        # CONTROL: the slot closure is exactly the declared field set — the
        # canonical digest surface and the attribute surface coincide —
        # PLUS, for the sealed components, the ONE private construction-seal
        # slot, which is deliberately NOT a field (r3-F2: not in replace()/
        # repr/compare/asdict/the digest surface) and whose only writer is
        # the private derivation (verified by the seal probes).
        for cls in (
            PairBindingDeclaration,
            ModelPrepRefusal,
            ModelPrepCensusDeclaration,
        ):
            assert set(getattr(cls, "__slots__")) == {
                f.name for f in dc_fields(cls)
            }, cls.__name__
        for component_cls in (
            StructureMemberBinding,
            SourceVolumeBinding,
            ComparatorBinding,
        ):
            slots = set(getattr(component_cls, "__slots__"))
            field_names = {f.name for f in dc_fields(component_cls)}
            # R6-F1: the slot closure is EXACTLY the payload field set —
            # no seal slot (the seal is COMPUTED, never stored) and no
            # __weakref__ (nothing holds a weak reference to a component
            # any more: there is no vault, no table, and no id-keyed state
            # to sweep). Every slot is therefore a payload slot, and the
            # payload is the entire instance surface.
            assert slots == field_names, component_cls.__name__
            assert "__weakref__" not in slots, component_cls.__name__
            # the seal name is NOT a declared field: dataclasses.replace
            # cannot carry it and the canonical digest cannot see it.
            assert "_binding_seal" not in field_names
            assert "binding_seal" not in field_names

    def test_mutation_construction_pins_are_load_bearing(self, monkeypatch):
        # MUTATION (permanent, in-suite): construction-time validation removed
        # -> the reserved-schema / invalid-member bypass REAPPEARS (probes
        # FAIL); restored -> every input is refused again (probes PASS).
        # r3-F2: the components validate in their custom __init__ (routed
        # through _seal_into), so the mutation disables THAT; the invalid
        # member is built by complete fresh construction (the only route).
        # R5-F1: the bypassing writer writes through the machinery's
        # first-only vault closure (the only write path there is) — the
        # mutated-away layer is the _seal_into validation, not the vault.

        def bypassing_component_init(self, **values):
            # R6-F1: the ONLY route that can write a payload slot at all is
            # the module's own construction frame (the payload lock's frame
            # gate), so the mutation disables the VALIDATION inside the
            # sanctioned route: it writes the raw caller values unvalidated.
            for name in values:
                _write_payload_raw(self, name, values[name])

        monkeypatch.setattr(PairBindingDeclaration, "__post_init__", lambda self: None)
        monkeypatch.setattr(
            StructureMemberBinding, "__init__", bypassing_component_init
        )
        invalid_member = StructureMemberBinding(
            structure_ref="artifact:structure-a",
            structure_sha256="nope",
            model_ref="coronary-seg@1.2.0",
            release_sha256=sha("release-a"),
            provenance_ref="provenance:artifact:structure-a",
            provenance_sha256=sha("prov-artifact:structure-a"),
        )
        assert invalid_member.structure_sha256 == "nope"  # the defect class
        bypassed = PairBindingDeclaration(
            first=invalid_member,
            second=member("artifact:structure-b"),
            source=source(),
            comparator=comparator(),
            schema_version=RESERVED_FAMILY_ID,
        )
        assert bypassed.schema_version == RESERVED_FAMILY_ID  # the defect class
        monkeypatch.undo()
        with pytest.raises(CanonicalValidationError):
            StructureMemberBinding(
                structure_ref="artifact:structure-a",
                structure_sha256="nope",
                model_ref="coronary-seg@1.2.0",
                release_sha256=sha("release-a"),
                provenance_ref="provenance:artifact:structure-a",
                provenance_sha256=sha("prov-artifact:structure-a"),
            )
        with pytest.raises(CanonicalValidationError):
            PairBindingDeclaration(
                first=member("artifact:structure-a"),
                second=member("artifact:structure-b"),
                source=source(),
                comparator=comparator(),
                schema_version=RESERVED_FAMILY_ID,
            )

    # -- digest-bound reachability (§10.3; r1-F1 rework: NO minted record) ----

    def test_digest_bound_reachability_through_a_real_registry_record(self):
        # §10.3: model release, weights digest, adapter, output schema are
        # unambiguously reachable via the digest-bound release reference —
        # the binding does not re-state them.
        record = make_record()
        registry = make_registry()
        sealed = registry.register(record)
        ref = model_ref(record.model_id, record.model_version)
        m = StructureMemberBinding(
            structure_ref="artifact:structure-a",
            structure_sha256=sha("artifact:structure-a"),
            model_ref=ref,
            release_sha256=sealed.release_digest,
            provenance_ref="provenance:prov-a",
            provenance_sha256=sha("prov-a"),
        )
        assert compute_release_digest(record) == m.release_sha256
        # reached through the reference, not restated in the binding:
        assert record.weights.sha256 == sha(WEIGHTS_BYTES)
        assert record.weights.adapter == "models/adapters/segmentation.py@abc123"
        assert record.output_contract.schema == "ccta.payload.segmentation.v1"
        assert not hasattr(m, "weights_sha256") and not hasattr(m, "adapter_ref")
        # §10.2: a patched adapter cannot hide behind the old binding — the
        # sealed release digest moves:
        tampered = replace(record, weights=replace(record.weights, adapter="other"))
        assert compute_release_digest(tampered) != m.release_sha256

    def test_provenance_reachability_is_a_digest_bound_reference_not_a_minted_record(self):
        # §10.3 (r1-F1 form): the binding's provenance reachability is the
        # SEALED DIGEST AS A VALUE. No record is minted for the model path;
        # an ALREADY EXISTING accepted provenance record from a NON-model
        # path (the shared tests.core.factories engine-path record — the
        # repo's own machinery) is bound by its sealed digest.
        accepted = make_provenance()
        assert isinstance(accepted, ProvenanceRecord)
        assert verify_self_digest(accepted)
        assert is_sha256_hex(accepted.record_sha256)
        bound = StructureMemberBinding(
            structure_ref="artifact:structure-a",
            structure_sha256=sha("artifact:structure-a"),
            model_ref="coronary-seg@1.2.0",
            release_sha256=sha("release-a"),
            provenance_ref=f"provenance:{accepted.provenance_id}",
            provenance_sha256=accepted.record_sha256,  # digest-bound reference
        )
        assert bound.provenance_sha256 == accepted.record_sha256
        assert not hasattr(bound, "record") and not hasattr(bound, "provenance_record")
        # tampering the REFERENCED record breaks its seal while the frozen
        # reference keeps carrying the ORIGINAL seal — the stale seal no
        # longer matches the tampered content, which is exactly what
        # verify_self_digest detects (the reference supplies the evidence;
        # it is not a copy that would re-seal itself):
        tampered = replace(accepted, subject_ref="artifact-tampered")
        assert not verify_self_digest(tampered)
        assert tampered.record_sha256 == bound.provenance_sha256  # the stale seal
        assert verify_self_digest(accepted)  # the accepted record is untouched

    def test_control_reused_accepted_records_stay_chain_valid(self):
        # CONTROL: the reused accepted records (non-model path) verify both
        # their self digests and their chain — the reuse path is sound.
        root = make_na_provenance()  # the accepted root-of-DAG fixture record
        derived = make_provenance(upstream_refs=(root.provenance_id,))
        assert verify_self_digest(root) and verify_self_digest(derived)
        ok, problems = verify_chain((root, derived))
        assert ok, problems


class TestD2DigestClosure:
    def test_binding_holds_references_not_metadata_copies(self):
        # §10.3: references supply the evidence; every leaf is a string.
        def leaves(value):
            if isinstance(value, dict):
                for item in value.values():
                    yield from leaves(item)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    yield from leaves(item)
            else:
                yield value

        d = declaration()
        assert all(isinstance(leaf, str) for leaf in leaves(dataclasses.asdict(d)))

    def test_frame_digest_rule_is_the_accepted_pair_rule(self):
        from core.coordinates import FrameSpec
        from core.qc.pair import _frame_digest

        frame = FrameSpec(
            frame_of_reference_uid="1.2.826.0.1.3680043.8.498.91.1",
            image_position_patient=(0.0, 0.0, 0.0),
            image_orientation_patient=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            spacing_mm=(1.0, 1.0, 1.0),
            dimensions=(4, 4, 4),
        )
        assert frame_spec_digest(frame) == _frame_digest(frame) == canonical_digest(frame)

    def test_no_lifecycle_or_authority_fields(self):
        # §10.4: the binding carries NO lifecycle, approval, or invalidation
        # authority; it is state-free and reference-only.
        banned = {
            "status",
            "approval",
            "approved_by",
            "invalidated",
            "invalidation",
            "current",
            "latest",
            "champion",
            "admissible",
            "substitute",
            "replacement",
            "superseded",
        }
        for cls in (
            PairBindingDeclaration,
            StructureMemberBinding,
            SourceVolumeBinding,
            ComparatorBinding,
        ):
            names = {f.name for f in dc_fields(cls)}
            assert names.isdisjoint(banned), (cls.__name__, names & banned)


# ---------------------------------------------------------------------------
# D3 — pinned fail-closed typed refusals (§9.2) — fail-closed BY CONSTRUCTION
# ---------------------------------------------------------------------------

PINNED_D3_MAPPING = {
    "Missing model record": QCAbstentionCode.QC_REQUIRED_PAIR_ABSENT,
    "Missing/blocked adapter binding": QCAbstentionCode.QC_REQUIRED_PAIR_INVALID,
    "Missing structure (either side)": QCAbstentionCode.QC_REQUIRED_PAIR_ABSENT,
    "Blocked or deprecated registry member": QCAbstentionCode.QC_SOURCE_IDENTITY_INVALID,
    "Unresolvable upstream provenance": QCAbstentionCode.QC_REGISTRATION_INVALID,
}


class TestD3PinnedSafetyRefusals:
    def test_mapping_is_exactly_the_pinned_9_2_table(self):
        assert dict(SAFETY_CASE_ABSTENTION_CODE) == PINNED_D3_MAPPING

    @pytest.mark.parametrize("case", list(PINNED_D3_MAPPING))
    def test_every_case_refuses_abstained_not_analyzable(self, case):
        refusal = refusal_for_case(case)
        assert refusal.abstention_code == PINNED_D3_MAPPING[case]
        assert refusal.outcome == QCOutcome.ABSTAINED
        assert refusal.quality_grade == QualityGrade.NOT_ANALYZABLE

    def test_codes_are_existing_closed_vocabulary_members(self):
        closed = set(QCAbstentionCode)
        assert set(SAFETY_CASE_ABSTENTION_CODE.values()) <= closed

    def test_unknown_case_fails_closed_without_default(self):
        with pytest.raises(CanonicalValidationError) as excinfo:
            refusal_for_case("closest available structure")
        message = str(excinfo.value)
        assert "closest available structure" in message
        assert "no default refusal" in message
        assert "never substitutes" in message

    def test_refusal_type_cannot_express_member_substitution(self):
        assert {f.name for f in dc_fields(ModelPrepRefusal)} == {
            "safety_case",
            "abstention_code",
            "outcome",
            "quality_grade",
        }

    def test_d3_mapping_is_load_bearing(self, monkeypatch):
        # MUTATION (permanent, in-suite; r2-F3 semantics): the refusal type
        # validates against the IMPORT-TIME frozen snapshot of the pinned
        # §9.2 mapping, so a corrupted module mapping can no longer admit a
        # re-pinned pairing — the corruption is REFUSED (fail-closed), and a
        # dropped case still fails closed at lookup.
        swapped = dict(SAFETY_CASE_ABSTENTION_CODE)
        swapped["Missing model record"] = QCAbstentionCode.QC_REGISTRATION_INVALID
        monkeypatch.setattr(model_prep, "SAFETY_CASE_ABSTENTION_CODE", swapped)
        with pytest.raises(CanonicalValidationError):
            refusal_for_case("Missing model record")  # re-pinning refused
        # enforcement removed (the snapshot pin itself patched out) -> the
        # re-pinned pairing would flow (the defect class becomes expressible;
        # this is what the snapshot enforcement prevents):
        monkeypatch.setattr(
            model_prep, "_PINNED_CASE_CODE_MAPPING", MappingProxyType(swapped)
        )
        wrong = refusal_for_case("Missing model record")
        assert wrong.abstention_code == QCAbstentionCode.QC_REGISTRATION_INVALID
        # restored -> the pinned pairing is enforced again:
        monkeypatch.undo()
        assert refusal_for_case("Missing model record").abstention_code == (
            QCAbstentionCode.QC_REQUIRED_PAIR_ABSENT
        )
        # dropped case -> the lookup fails closed:
        dropped = dict(SAFETY_CASE_ABSTENTION_CODE)
        del dropped["Unresolvable upstream provenance"]
        monkeypatch.setattr(model_prep, "SAFETY_CASE_ABSTENTION_CODE", dropped)
        with pytest.raises(CanonicalValidationError):
            refusal_for_case("Unresolvable upstream provenance")
        monkeypatch.undo()
        assert refusal_for_case(
            "Unresolvable upstream provenance"
        ).abstention_code == QCAbstentionCode.QC_REGISTRATION_INVALID


class TestR1F3RefusalFailClosedByConstruction:
    """r1-F3: a refusal must be impossible to express as a proceed.

    Direct construction pins outcome/grade (CLEAR/GOOD raise), pins the
    abstention code to the pinned §9.2 pairing and the five-code closed set,
    and the slot-closed type carries NO member field, so member substitution
    is structurally inexpressible.
    """

    # -- CLEAR/GOOD and every other proceed shape are REJECTED ---------------

    @pytest.mark.parametrize("outcome", [o for o in QCOutcome if o != QCOutcome.ABSTAINED])
    def test_negative_non_abstained_outcome_refused_on_direct_construction(self, outcome):
        # DIRECT construction (the bypass r1 demonstrated) — CLEAR/GOOD (and
        # MILD/SEVERE) raise; a refusal cannot be expressed as a proceed.
        with pytest.raises(CanonicalValidationError) as excinfo:
            ModelPrepRefusal(
                safety_case="Missing model record",
                abstention_code=QCAbstentionCode.QC_REQUIRED_PAIR_ABSENT,
                outcome=outcome,
            )
        assert "cannot be expressed as a proceed" in excinfo.value.violations[0]

    @pytest.mark.parametrize(
        "grade", [g for g in QualityGrade if g != QualityGrade.NOT_ANALYZABLE]
    )
    def test_negative_non_not_analyzable_grade_refused_on_direct_construction(self, grade):
        with pytest.raises(CanonicalValidationError) as excinfo:
            ModelPrepRefusal(
                safety_case="Missing model record",
                abstention_code=QCAbstentionCode.QC_REQUIRED_PAIR_ABSENT,
                quality_grade=grade,
            )
        assert "quality_grade is pinned" in excinfo.value.violations[0]

    def test_negative_replace_cannot_regrade_or_proceed(self):
        refusal = refusal_for_case("Missing model record")
        with pytest.raises(CanonicalValidationError):
            replace(refusal, outcome=QCOutcome.CLEAR)
        with pytest.raises(CanonicalValidationError):
            replace(refusal, quality_grade=QualityGrade.GOOD)

    def test_control_pinned_refusal_constructs_directly_and_matches_the_factory(self):
        direct = ModelPrepRefusal(
            safety_case="Missing model record",
            abstention_code=QCAbstentionCode.QC_REQUIRED_PAIR_ABSENT,
        )
        assert direct == refusal_for_case("Missing model record")
        assert direct.outcome == QCOutcome.ABSTAINED
        assert direct.quality_grade == QualityGrade.NOT_ANALYZABLE

    # -- the code stays inside the pinned §9.2 vocabulary --------------------

    def test_negative_mismatched_pinned_pairing_refused(self):
        with pytest.raises(CanonicalValidationError) as excinfo:
            ModelPrepRefusal(
                safety_case="Missing model record",
                abstention_code=QCAbstentionCode.QC_REGISTRATION_INVALID,
            )
        assert "must be the pinned" in excinfo.value.violations[0]

    def test_negative_code_outside_the_pinned_closed_set_refused(self, monkeypatch):
        # Even a corrupted module mapping cannot make the TYPE express a
        # refusal outside the five pinned codes: the closed set is computed
        # from the pinned mapping at import and re-checked at construction.
        corrupted = dict(SAFETY_CASE_ABSTENTION_CODE)
        corrupted["Missing model record"] = QCAbstentionCode.QC_GEOMETRY_INVALID
        monkeypatch.setattr(model_prep, "SAFETY_CASE_ABSTENTION_CODE", corrupted)
        with pytest.raises(CanonicalValidationError) as excinfo:
            refusal_for_case("Missing model record")
        assert "outside the five pinned" in excinfo.value.violations[0]

    def test_negative_unknown_case_refused_on_direct_construction(self):
        # Direct construction is exactly as refusing as the factory: no
        # closest-available mapping, no default refusal, no substitution.
        with pytest.raises(CanonicalValidationError) as excinfo:
            ModelPrepRefusal(
                safety_case="closest available structure",
                abstention_code=QCAbstentionCode.QC_REQUIRED_PAIR_ABSENT,
            )
        assert "unknown MODEL_PREP safety case" in excinfo.value.violations[0]
        assert "no default refusal" in excinfo.value.violations[0]

    def test_negative_non_member_code_refused(self):
        with pytest.raises(CanonicalValidationError) as excinfo:
            ModelPrepRefusal(
                safety_case="Missing model record",
                abstention_code="QC_REQUIRED_PAIR_ABSENT",  # type: ignore[arg-type]
            )
        assert "EXISTING QCAbstentionCode member" in excinfo.value.violations[0]

    # -- member substitution stays structurally inexpressible ----------------

    def test_negative_substitute_member_cannot_be_carried_or_attached(self):
        refusal = refusal_for_case("Missing model record")
        # no field can carry a substitute member:
        assert {f.name for f in dc_fields(ModelPrepRefusal)} == {
            "safety_case",
            "abstention_code",
            "outcome",
            "quality_grade",
        }
        # and no attribute can be attached (slot closure):
        with pytest.raises((TypeError, AttributeError)):
            refusal.substitute_member = member("artifact:structure-x")  # type: ignore[attr-defined]
        with pytest.raises(AttributeError):
            object.__setattr__(refusal, "substitute_member", member("artifact:structure-x"))

    def test_mutation_outcome_pinning_is_load_bearing(self, monkeypatch):
        # MUTATION (permanent, in-suite): the outcome/grade pinning removed ->
        # a CLEAR/GOOD "refusal" becomes constructible (the bypass reappears,
        # probes FAIL); restored -> refused again (probes PASS).
        monkeypatch.setattr(ModelPrepRefusal, "__post_init__", lambda self: None)
        bypassed = ModelPrepRefusal(
            safety_case="Missing model record",
            abstention_code=QCAbstentionCode.QC_REQUIRED_PAIR_ABSENT,
            outcome=QCOutcome.CLEAR,
            quality_grade=QualityGrade.GOOD,
        )
        assert bypassed.outcome == QCOutcome.CLEAR  # the defect class
        monkeypatch.undo()
        with pytest.raises(CanonicalValidationError):
            ModelPrepRefusal(
                safety_case="Missing model record",
                abstention_code=QCAbstentionCode.QC_REQUIRED_PAIR_ABSENT,
                outcome=QCOutcome.CLEAR,
            )


# ---------------------------------------------------------------------------
# D4 quick shape checks (full enumeration proof in tests/apps census file)
# ---------------------------------------------------------------------------


class TestD4DeclaredCensusShape:
    def test_census_declaration_matches_part_4_6_row(self):
        assert MODEL_PREP_CENSUS.geometry_strata_count == 48
        assert MODEL_PREP_MEMBER_ROLES == ("AGREE", "DISAGREE")
        assert MODEL_PREP_DECLARED_N == 96
        assert MODEL_PREP_CENSUS.executed is False


class TestR3F3CensusExactTypeClosure:
    """R3-F3: the census fields are EXACT-type checked — equality-compatible
    subclass carriers cannot ride a count, role/endpoint/threshold
    collection, or the ``executed`` flag.

    The r3 review reproduced the att0005 gap: an ``int`` subclass with
    ``approved`` attached passed ``geometry_strata_count != 48`` (equality)
    and a ``tuple`` subclass passed the collection checks; ``executed=0``
    passed because ``0 == False`` is true in Python. Every carrier is now
    refused by exact type, and ``executed`` must be exactly ``False``.
    """

    def _pinned_kwargs(self) -> dict[str, Any]:
        return {
            "geometry_strata_count": MODEL_PREP_GEOMETRY_STRATA_COUNT,
            "member_roles": MODEL_PREP_MEMBER_ROLES,
            "declared_n": MODEL_PREP_DECLARED_N,
            "primary_endpoints": MODEL_PREP_PRIMARY_ENDPOINTS,
            "declared_only_thresholds": MODEL_PREP_DECLARED_ONLY_THRESHOLDS,
        }

    def test_negative_int_subclass_count_carrier_refused(self):
        class Count(int):
            approved = True

        with pytest.raises(CanonicalValidationError) as excinfo:
            ModelPrepCensusDeclaration(
                **{**self._pinned_kwargs(), "geometry_strata_count": Count(48)}
            )
        assert any(
            "geometry_strata_count must be exactly a plain int" in p
            for p in excinfo.value.violations
        )
        with pytest.raises(CanonicalValidationError) as excinfo:
            ModelPrepCensusDeclaration(
                **{**self._pinned_kwargs(), "declared_n": Count(96)}
            )
        assert any(
            "declared_n must be exactly a plain int" in p
            for p in excinfo.value.violations
        )

    def test_negative_tuple_subclass_roles_carrier_refused(self):
        class Roles(tuple):
            approved = True

        with pytest.raises(CanonicalValidationError) as excinfo:
            ModelPrepCensusDeclaration(
                **{**self._pinned_kwargs(), "member_roles": Roles(("AGREE", "DISAGREE"))}
            )
        assert any(
            "member_roles must be exactly a plain tuple" in p
            for p in excinfo.value.violations
        )
        for field_name, pinned in (
            ("primary_endpoints", MODEL_PREP_PRIMARY_ENDPOINTS),
            ("declared_only_thresholds", MODEL_PREP_DECLARED_ONLY_THRESHOLDS),
        ):
            with pytest.raises(CanonicalValidationError) as excinfo:
                ModelPrepCensusDeclaration(
                    **{
                        **self._pinned_kwargs(),
                        field_name: Roles(pinned),
                    }
                )
            assert any(
                f"{field_name} must be exactly a plain tuple" in p
                for p in excinfo.value.violations
            )

    def test_negative_str_subclass_element_carrier_refused(self):
        class Element(str):
            approved = True

        with pytest.raises(CanonicalValidationError) as excinfo:
            ModelPrepCensusDeclaration(
                **{
                    **self._pinned_kwargs(),
                    "member_roles": (Element("AGREE"), "DISAGREE"),
                }
            )
        assert any(
            "member_roles elements must be exactly plain str" in p
            for p in excinfo.value.violations
        )

    def test_negative_executed_zero_and_bool_subclass_carriers_refused(self):
        # r3 evidence: executed=0 passed at att0005 because 0 == False.
        with pytest.raises(CanonicalValidationError) as excinfo:
            ModelPrepCensusDeclaration(**{**self._pinned_kwargs(), "executed": 0})
        assert any(
            v.startswith("the QC_001-MODEL census is declared-not-executed")
            for v in excinfo.value.violations
        )
        with pytest.raises(CanonicalValidationError):
            ModelPrepCensusDeclaration(**{**self._pinned_kwargs(), "executed": 1})

        class TruthyBool(int):
            def __bool__(self) -> bool:  # pragma: no cover - never called
                return True

        with pytest.raises(CanonicalValidationError):
            ModelPrepCensusDeclaration(
                **{**self._pinned_kwargs(), "executed": TruthyBool(0)}
            )

    def test_control_exact_types_construct_and_validate(self):
        # CONTROL: the pinned exact types construct and the use-time value
        # screen accepts the module census; executed is exactly False.
        census = ModelPrepCensusDeclaration(**self._pinned_kwargs())
        assert type(census.geometry_strata_count) is int
        assert type(census.member_roles) is tuple
        assert type(census.executed) is bool and census.executed is False
        assert MODEL_PREP_CENSUS == census

    def test_mutation_exact_type_census_checks_are_load_bearing(self, monkeypatch):
        # MUTATION (permanent, in-suite; r3-F3): the exact-type census checks
        # removed -> the equality-compatible carriers are ACCEPTED again
        # (the defect class reappears); restored -> refused.
        import models.registry.model_prep as mp_module

        class Count(int):
            approved = True

        def relaxed(values):  # the att0005 semantics (equality only)
            problems = []
            if values.get("geometry_strata_count") != MODEL_PREP_GEOMETRY_STRATA_COUNT:
                problems.append("geometry_strata_count must be 48")
            if values.get("executed"):
                problems.append("the QC_001-MODEL census is declared-not-executed")
            return problems

        monkeypatch.setattr(mp_module, "_census_field_problems", relaxed)
        bypassed = ModelPrepCensusDeclaration(
            **{**self._pinned_kwargs(), "geometry_strata_count": Count(48), "executed": 0}
        )
        assert bypassed.geometry_strata_count == 48  # the defect class
        assert bypassed.executed == 0  # the defect class (0 is not False)
        monkeypatch.undo()
        with pytest.raises(CanonicalValidationError):
            ModelPrepCensusDeclaration(
                **{**self._pinned_kwargs(), "geometry_strata_count": Count(48)}
            )
        with pytest.raises(CanonicalValidationError):
            ModelPrepCensusDeclaration(**{**self._pinned_kwargs(), "executed": 0})

    def test_endpoints_are_declared_strings_never_metrics(self):
        for text in (
            *MODEL_PREP_PRIMARY_ENDPOINTS,
            *MODEL_PREP_DECLARED_ONLY_THRESHOLDS,
        ):
            assert isinstance(text, str)


# ---------------------------------------------------------------------------
# D5 — offline-experimental / inert-by-construction boundary
# ---------------------------------------------------------------------------

MODEL_PREP_IMPORT_ALLOWLIST = frozenset(
    {
        "__future__",
        "collections.abc",
        "dataclasses",
        # R6-F1: sys is used ONLY for sys._getframe() in the payload write
        # lock's calling-frame identity check (no inspect, no introspection
        # of private machinery: it compares a code object against the
        # module's own captured code objects). It is stdlib, pure (no
        # file/process/network surface), adds no dependency, and is
        # exercised by the D5 no-exec/no-file probe below.
        "sys",
        "types",
        "typing",
        "core.coordinates",
        "core.digest",
        "core.enums",
        "core.errors",
        "core.qc.constants",
    }
)


def imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                modules.add(node.module)
            elif node.level > 0 and node.module:
                modules.add(f".{node.module}")
    return modules


class TestD5OfflineExperimentalBoundary:
    def test_declaration_module_import_allowlist(self):
        # stdlib + already-admitted core modules ONLY: no network, no
        # process surface, no new dependency can enter unnoticed.
        modules = imported_modules(Path(model_prep.__file__))
        unexpected = modules - MODEL_PREP_IMPORT_ALLOWLIST
        assert not unexpected, f"unpinned imports: {sorted(unexpected)}"

    def test_no_exec_eval_or_file_surface_in_declaration_module(self):
        tree = ast.parse(Path(model_prep.__file__).read_text(encoding="utf-8"))
        banned = {"open", "eval", "exec", "compile", "__import__", "system"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name):
                    assert func.id not in banned, func.id
                if isinstance(func, ast.Attribute):
                    assert func.attr not in banned, func.attr

    def test_models_registry_stays_import_clean_of_orchestration(self):
        # BundleAcceptor injection precedent: product code never imports
        # orchestration (boundary discipline).
        pkg_dir = Path(registry_module.__file__).parent
        for py_file in sorted(pkg_dir.glob("*.py")):
            modules = imported_modules(py_file)
            assert not any(
                m == "orchestration" or m.startswith("orchestration.") for m in modules
            ), py_file

    def test_declaration_construction_is_side_effect_free(self, tmp_path):
        declaration()
        refusal_for_case("Missing model record")
        assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# r2 — value/object-level validation (correction round 2: R2-F1..R2-F4)
# ---------------------------------------------------------------------------


def _model_path_record_problems(record: object) -> list[str]:
    """The VALUE-LEVEL model-path provenance screen (r2-F1, r3-F1 layer 3).

    Any provenance record produced or reachable in the model-prep path must
    fail this screen — regardless of which name, helper, or route produced
    it. Layer 1 (r3-F1) is the STRUCTURAL refusal inside the shared helper
    itself; layer 2 is the mechanical AST/grep guard; THIS is the
    result-level assertion applied to every record the path actually
    produces or binds.
    """
    if not isinstance(record, ProvenanceRecord):
        return ["model-path provenance screen: not a ProvenanceRecord"]
    problems: list[str] = []
    if record.actor.actor_type is ActorType.MODEL:
        problems.append(
            "model-path provenance: actor_type is MODEL (declared-not-"
            "executed; the #17 r8/r9 invention class)"
        )
    if getattr(record, "model_record_ref", None):
        problems.append("model-path provenance: model_record_ref is set")
    if str(record.subject_ref).startswith("artifact:"):
        problems.append("model-path provenance: model-path subject_ref")
    return problems


class TestR3F1StructuralHelperRefusal:
    """R3-F1: the model-path provenance mint is refused STRUCTURALLY, in the
    shared helper itself — not by a pytest autouse fixture.

    The r3 review reproduced the att0005 guard's failure: the behavioural
    screen existed only as a monkeypatch installed by an autouse fixture in
    this test module, so a fixture-free call to the admitted helper
    ``tests.core.factories.make_provenance`` still minted the prohibited
    record, and a planted alternate helper passed the name-based AST
    detector. The correction is STRUCTURAL (in ``tests/core/factories.py``):
    the helper itself refuses the mintable model-path shape by default
    (``ActorType.MODEL`` + a set ``model_record_ref``, or an ``artifact:``
    subject) unless the caller explicitly passes the provenance-domain
    opt-in ``allow_model_path_provenance=True`` — which NOTHING in the
    model-prep path ever passes (the opt-in screen below fails the suite if
    it appears). There is NO autouse fixture anywhere in this module: every
    negative below runs fixture-free and holds with plain Python too.
    """

    def test_negative_the_reviewer_reproducing_helper_mint_is_refused(self):
        # EXACT reviewer construction (r3 evidence): arbitrary overrides
        # through the admitted helper, NO fixture, NO construction import
        # anywhere in this file — refused BY THE HELPER before any record
        # exists. Re-run as a plain script (no pytest machinery) in the
        # att0007 verification battery (§7.10 of the report).
        with pytest.raises(
            CanonicalValidationError, match="refuses to mint a MODEL-path provenance"
        ):
            make_provenance(
                subject_ref="artifact:structure-a",
                actor=replace(
                    make_actor(),
                    actor_type=ActorType.MODEL,
                    actor_id="model-prep-probe",
                ),
                model_record_ref="coronary-seg@1.2.0",
            )

    def test_negative_each_mintable_model_path_shape_is_refused_independently(self):
        # fail-closed on EACH mintable model-path shape, independently:
        model_actor = replace(
            make_actor(), actor_type=ActorType.MODEL, actor_id="model-prep-probe"
        )
        with pytest.raises(
            CanonicalValidationError, match="actor_type=MODEL with model_record_ref set"
        ):
            make_provenance(
                actor=model_actor, model_record_ref="coronary-seg@1.2.0"
            )
        with pytest.raises(CanonicalValidationError, match="model-path subject"):
            make_provenance(subject_ref="artifact:structure-a")

    def test_control_engine_path_helper_mints_still_work(self):
        # CONTROL: the SAME helper with NON-model values mints as before and
        # the reuse probes stay reachable — the screen keys on the record's
        # VALUES (actor type / model ref / subject), not on the symbol.
        accepted = make_provenance()
        assert isinstance(accepted, ProvenanceRecord)
        assert verify_self_digest(accepted)
        assert accepted.actor.actor_type is not ActorType.MODEL
        assert accepted.model_record_ref is None
        assert _model_path_record_problems(accepted) == []
        root = make_na_provenance()
        assert _model_path_record_problems(root) == []
        derived = make_provenance(upstream_refs=(root.provenance_id,))
        ok_chain, problems = verify_chain((root, derived))
        assert ok_chain, problems

    def test_control_upstream_provenance_domain_negatives_still_fall_through(self):
        # CONTROL (upstream parity): the helper's screen refuses only the
        # MINTABLE model-path shape. The upstream provenance-domain
        # negatives are NOT mintable — the provenance constructor itself
        # fails closed on both — so they keep falling through to its own
        # ProvenanceIncompleteError with the §44 message unchanged. This is
        # what keeps every upstream consumer of the shared factory green
        # (tests/core/test_provenance.py pins both of these exact behaviors).
        from core.errors import ProvenanceIncompleteError

        with pytest.raises(ProvenanceIncompleteError) as excinfo:
            make_provenance(
                actor=replace(make_actor(), actor_type=ActorType.MODEL, actor_id="seg-model"),
                model_record_ref=None,
            )
        assert any("model_record_ref" in v for v in excinfo.value.violations)
        with pytest.raises(ProvenanceIncompleteError) as excinfo:
            make_provenance(
                actor=make_actor(),
                model_record_ref="coronary-seg@1.0.0",
            )
        assert any(
            "model_record_ref also populated" in v for v in excinfo.value.violations
        )

    def test_mutation_the_structural_refusal_is_load_bearing(self, monkeypatch):
        # MUTATION (permanent, in-suite; r3-F1): the helper's structural
        # refusal removed -> the reviewer's exact construction MINTS again
        # (the defect class reappears, in memory, discarded) and the minted
        # record's VALUES are what the value screen flags; restored ->
        # refused again — fixture-free either way (there is no fixture).
        import tests.core.factories as factories_module

        monkeypatch.setattr(
            factories_module, "_model_path_mint_problems", lambda overrides: []
        )
        minted = make_provenance(
            subject_ref="artifact:structure-a",
            actor=replace(
                make_actor(), actor_type=ActorType.MODEL, actor_id="model-prep-probe"
            ),
            model_record_ref="coronary-seg@1.2.0",
        )
        assert minted.actor.actor_type is ActorType.MODEL  # the defect class
        problems = _model_path_record_problems(minted)
        assert problems, "the value screen must flag the minted record's VALUES"
        assert any("actor_type is MODEL" in p for p in problems)
        monkeypatch.undo()
        with pytest.raises(
            CanonicalValidationError, match="refuses to mint a MODEL-path provenance"
        ):
            make_provenance(
                subject_ref="artifact:structure-a",
                actor=replace(make_actor(), actor_type=ActorType.MODEL),
                model_record_ref="coronary-seg@1.2.0",
            )

    def test_mutation_a_planted_working_minting_helper_is_detected(self, tmp_path):
        # MUTATION (r3-F1 requirement (b)): plant a WORKING minting helper —
        # one that actually mints the prohibited record (it must pass the
        # provenance-domain opt-in to get past the structural refusal; that
        # opt-in is the ONLY unmute route) — and prove the guard fires:
        #   (dormant) the opt-in screen and the value screen fire on the
        #             planted SOURCE, without executing it;
        #   (live)    executing it mints, and the value screen flags the
        #             minted record's VALUES.
        opt_in = _OPT_IN_KWARG
        planted = tmp_path / "planted_working_minting_helper.py"
        planted.write_text(
            "import dataclasses\n"
            "from core.enums import ActorType\n"
            "from tests.core.factories import make_actor, make_provenance\n"
            "\n"
            "def dormant_model_path_minter():\n"
            "    return make_provenance(\n"
            f"        {opt_in}=True,\n"
            "        subject_ref='artifact:structure-a',\n"
            "        actor=dataclasses.replace(\n"
            "            make_actor(),\n"
            "            actor_type=ActorType.MODEL,\n"
            "            actor_id='planted-mint',\n"
            "        ),\n"
            "        model_record_ref='coronary-seg@1.2.0',\n"
            "    )\n",
            encoding="utf-8",
        )
        # (dormant) the guard fires on the source, no execution needed:
        tree = ast.parse(planted.read_text(encoding="utf-8"))
        with pytest.raises(AssertionError, match="provenance-domain opt-in"):
            _assert_no_model_path_opt_in(tree, planted)
        value_problems = _planted_mint_value_problems(tree, planted)
        assert value_problems, "the value screen must fire on the planted source"
        assert any(_MODEL_REF_KWARG in p for p in value_problems)
        assert any("model-path subject" in p for p in value_problems)
        # the r2-era name-based detector alone would NOT have fired here
        # (the planted helper has no construction symbol) — that is exactly
        # the r3-F1 defect this probe pins as closed by the new screens:
        _assert_no_construction(tree, planted)  # passes: no construction symbol
        # (live) executing the planted helper mints, and the value screen
        # flags the minted record's VALUES:
        import importlib.util

        spec = importlib.util.spec_from_file_location("planted_working_minting_helper", planted)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        minter = getattr(module, "dormant_model_path_minter")
        minted = minter()
        problems = _model_path_record_problems(minted)
        assert problems, "the value screen must flag the planted mint"
        assert any("actor_type is MODEL" in p for p in problems)

    def test_the_guard_cannot_go_quiet(self):
        # the structural refusal is part of the permanent guard machinery:
        # the anchors (this class, the opt-in screen, the value screen) and
        # the shared helper's refusal markers are all self-checked (see
        # test_guard_self_check_the_mechanism_cannot_go_quiet and
        # test_the_structural_refusal_ships_with_the_guard).
        source = Path(__file__).read_text(encoding="utf-8")
        for anchor in (
            "TestR3F1StructuralHelperRefusal",
            "_model_path_record_problems",
            "_planted_mint_value_problems",
        ):
            assert anchor in source, anchor



class TestR2F2ValueLevelAuthorityClosure:
    """R2-F2: authority state outside the canonical digest is impossible
    REGARDLESS of carrier class or construction route.

    Reviewer reproductions, all refused now:
    (a) a ``str`` subclass carrying ``approved`` survives into
        ``pair.first.provenance_ref``;
    (b) a ``StructureMemberBinding`` subclass regains ``__dict__`` and is
        accepted by the pair constructor;
    (c) ``object.__new__`` + the generated ``__setstate__`` mints members
        (and a RESERVED-schema pair) that ``__post_init__`` never saw.
    """

    # -- (a) str subclass carriers refused at the exact-type check ----------

    def test_negative_str_subclass_ref_carrier_refused(self):
        class Ref(str):
            pass

        carrier = Ref("provenance:prov-a")
        carrier.approved = True  # type: ignore[attr-defined]
        # r3-F2: dataclasses.replace on a component is STRUCTURALLY refused
        # (the only member route is complete fresh construction), so the
        # carrier is planted by a COMPLETE FRESH CONSTRUCTION and must be
        # refused there by the exact-type field check:
        with pytest.raises(CanonicalValidationError) as excinfo:
            StructureMemberBinding(
                structure_ref="artifact:structure-a",
                structure_sha256=sha("artifact:structure-a"),
                model_ref="coronary-seg@1.2.0",
                release_sha256=sha("release-a"),
                provenance_ref=carrier,
                provenance_sha256=sha("prov-artifact:structure-a"),
            )
        assert "must be exactly a plain str" in excinfo.value.violations[0]

    def test_negative_str_subclass_identity_carrier_refused(self):
        class Ref(str):
            pass

        with pytest.raises(CanonicalValidationError) as excinfo:
            StructureMemberBinding(
                structure_ref=Ref("artifact:structure-a"),
                structure_sha256=sha("artifact:structure-a"),
                model_ref="coronary-seg@1.2.0",
                release_sha256=sha("release-a"),
                provenance_ref="provenance:artifact:structure-a",
                provenance_sha256=sha("prov-artifact:structure-a"),
            )
        assert "must be exactly a plain str" in excinfo.value.violations[0]

    def test_control_plain_str_fields_accepted(self):
        m = member()
        assert type(m.provenance_ref) is str
        assert type(m.structure_ref) is str

    # -- (b) dataclass subclass carriers refused (closed types) --------------

    def test_negative_member_subclass_refused(self):
        with pytest.raises(TypeError, match="subclassing StructureMemberBinding"):

            class Smuggler(StructureMemberBinding):
                pass

    def test_negative_source_and_comparator_subclasses_refused(self):
        with pytest.raises(TypeError, match="subclassing SourceVolumeBinding"):

            class S(SourceVolumeBinding):
                pass

        with pytest.raises(TypeError, match="subclassing ComparatorBinding"):

            class C(ComparatorBinding):
                pass

    def test_negative_pair_subclass_refused(self):
        with pytest.raises(TypeError, match="subclassing PairBindingDeclaration"):

            class P(PairBindingDeclaration):
                pass

    # -- (c) __setstate__ is sealed: deserialization re-validates ------------

    def test_negative_setstate_bad_digest_refused(self):
        # R4-F1: deserialization state must be CONSTRUCTION-SHAPED (payload
        # + the serialized seal). A state whose PAYLOAD carries an invalid
        # digest is refused — and a bare payload (no seal) is refused by
        # shape before the payload is even examined.
        forged = StructureMemberBinding.__new__(StructureMemberBinding)
        with pytest.raises(CanonicalValidationError) as excinfo:
            forged.__setstate__(
                ["x", "nope", "coronary-seg@1.2.0", sha("release-a"), "p:x", sha("p")]
            )
        assert "must be CONSTRUCTION-SHAPED" in excinfo.value.violations[0]
        # a construction-shaped state whose payload carries a bad digest is
        # refused with the field named:
        with pytest.raises(CanonicalValidationError) as excinfo:
            forged.__setstate__(
                ["x", "nope", "coronary-seg@1.2.0", sha("release-a"), "p:x", sha("p"),
                 "f" * 64]
            )
        assert "structure_sha256 must be a 64-hex sha256 digest" in (
            excinfo.value.violations[0]
        )

    def test_negative_setstate_reserved_schema_pair_refused(self):
        forged_pair = PairBindingDeclaration.__new__(PairBindingDeclaration)
        with pytest.raises(CanonicalValidationError) as excinfo:
            forged_pair.__setstate__(
                [
                    member("artifact:structure-a"),
                    member("artifact:structure-b"),
                    source(),
                    comparator(),
                    RESERVED_FAMILY_ID,
                ]
            )
        assert "RESERVED family id" in excinfo.value.violations[0]

    def test_negative_setstate_wrong_shaped_state_refused(self):
        # the generated __setstate__ silently accepted ANY state shape; the
        # sealed one accepts only the exact declared field sequence/mapping.
        forged = StructureMemberBinding.__new__(StructureMemberBinding)
        with pytest.raises(CanonicalValidationError, match="deserialization state"):
            forged.__setstate__(("only-one-field",))

    def test_negative_setstate_stale_seal_refused(self):
        # R4-F1: the serialized state must carry the seal it was serialized
        # with, and that seal must match the private derivation over the
        # state's OWN payload. A state carrying a stale/wrong seal — the
        # reviewer's partial re-bless shape — is refused outright, and a
        # state carrying a NON-SEAL extra element is refused by shape:
        forged = StructureMemberBinding.__new__(StructureMemberBinding)
        with pytest.raises(CanonicalValidationError, match="CONSTRUCTION-SHAPED"):
            forged.__setstate__(
                [
                    "x",
                    sha("x"),
                    "coronary-seg@1.2.0",
                    sha("release-a"),
                    "p:x",
                    sha("p"),
                    sha("not-the-payload"),
                ]
            )
        # the exact R4-F1 shape — the CURRENT (old) seal carried with a
        # PATCHED payload — is refused because the seal no longer matches
        # the payload it claims. R6-F1: the patch itself is now refused at
        # WRITE time by the payload lock (no patch route exists at all), so
        # the state is assembled directly to exercise the seal check:
        m = member()
        patched_payload = {
            name: getattr(m, name) for name in _MEMBER_PAYLOAD_FIELDS
        }
        patched_payload["structure_ref"] = "artifact:replacement"
        patched_state = tuple(
            patched_payload[name] for name in _MEMBER_PAYLOAD_FIELDS
        ) + (m.binding_seal,)
        forged2 = StructureMemberBinding.__new__(StructureMemberBinding)
        with pytest.raises(CanonicalValidationError, match="own seal does not match"):
            forged2.__setstate__(patched_state)

    def test_control_pickle_round_trip_of_valid_values(self):
        d = declaration()
        assert pickle.loads(pickle.dumps(d)) == d
        r = refusal_for_case("Missing model record")
        assert pickle.loads(pickle.dumps(r)) == r
        assert pickle.loads(pickle.dumps(MODEL_PREP_CENSUS)) == MODEL_PREP_CENSUS
        assert pickle.loads(pickle.dumps(member())) == member()
        # r3-F2/R4-F1: the member state is CONSTRUCTION-SHAPED (payload +
        # the serialized seal), the seal is verified against the state's
        # own payload at load, and a pickled copy of an equal payload
        # carries an EQUAL seal (deterministic private derivation):
        m = member()
        reloaded = pickle.loads(pickle.dumps(m))
        assert reloaded.binding_seal == m.binding_seal
        # the serialized state carries the seal the value was serialized
        # with — a bare payload tuple is NOT the state form anymore:
        state = m.__getstate__()
        assert len(state) == len(_MEMBER_PAYLOAD_FIELDS) + 1
        assert is_sha256_hex(state[-1])
        fresh = StructureMemberBinding.__new__(StructureMemberBinding)
        fresh.__setstate__(state)
        assert fresh == m and fresh.binding_seal == m.binding_seal

    def test_negative_unpickling_a_slot_tampered_member_refused(self):
        # r3-F2 anti-laundering: a member tampered after construction cannot
        # be laundered through serialization — __getstate__ runs the FULL
        # use-time validation (including the seal check) BEFORE emitting any
        # state, so the tamper is refused at DUMP time.
        blob = pickle.dumps(member())
        patched = pickle.loads(blob)
        # R6-F1: the tamper itself is refused at WRITE time by the payload
        # lock (a one-field patch has no route), and the value is therefore
        # still a valid, untampered, serializable value:
        with pytest.raises(CanonicalValidationError, match="construction-only"):
            object.__setattr__(patched, "structure_sha256", sha("TAMPERED"))
        assert pickle.loads(pickle.dumps(patched)) == patched
        # ...and the laundering route it closed is demonstrated by a value
        # whose state was forged BEFORE it was ever a value: a state whose
        # seal does not match the payload it claims is refused at LOAD:
        m = member()
        forged_state = tuple(getattr(m, n) for n in _MEMBER_PAYLOAD_FIELDS) + (
            sha("not-the-payload"),
        )
        forged = StructureMemberBinding.__new__(StructureMemberBinding)
        with pytest.raises(CanonicalValidationError, match="own seal does not match"):
            forged.__setstate__(forged_state)

    # -- authority state outside the digest: caught at digest + use time -----

    def test_negative_slot_tamper_cannot_obtain_a_canonical_seal(self):
        # R6-F1: the tamper is refused at WRITE time by the payload lock —
        # the pair stays valid and its digest is unchanged, i.e. there is no
        # "tampered" state to obtain a canonical seal from at all:
        stealth = declaration()
        original_digest = stealth.declaration_sha256
        with pytest.raises(CanonicalValidationError, match="construction-only"):
            object.__setattr__(stealth.first, "structure_sha256", sha("EVASIVE"))
        assert stealth.declaration_sha256 == original_digest
        assert pair_binding_problems(stealth) == []
        # the mutation probe below proves the lock is what refuses it.

    def test_negative_post_init_cannot_re_pin_the_schema(self):
        # R6-F1: the pair's own fields now carry the payload lock too, so an
        # in-place schema re-pin is refused at WRITE time — the declaration
        # can never be made to carry the RESERVED family id at all:
        d = declaration()
        with pytest.raises(CanonicalValidationError, match="never patch one field"):
            object.__setattr__(d, "schema_version", RESERVED_FAMILY_ID)
        assert d.schema_version == PAIR_BINDING_SCHEMA
        assert pair_binding_problems(d) == []
        # and the construction route still refuses the reserved id directly:
        with pytest.raises(CanonicalValidationError, match="RESERVED family id"):
            PairBindingDeclaration(
                first=d.first,
                second=d.second,
                source=d.source,
                comparator=d.comparator,
                schema_version=RESERVED_FAMILY_ID,
            )

    def test_negative_subclass_carrier_refused_at_use_time(self):
        # even a manually forged object that passes isinstance() cannot pass
        # the value-level screen: exact type + seal are required.
        probe = SimpleNamespace(
            structure_ref="artifact:structure-a",
            structure_sha256=sha("artifact:structure-a"),
            model_ref="coronary-seg@1.2.0",
            release_sha256=sha("release-a"),
            provenance_ref="p:a",
            provenance_sha256=sha("p"),
            binding_seal=None,
        )
        problems = pair_binding_problems(
            SimpleNamespace(first=probe, second=member("artifact:structure-b"),
                            source=source(), comparator=comparator(),
                            schema_version=PAIR_BINDING_SCHEMA)
        )
        assert any("must be exactly" in p for p in problems)

    def test_mutation_exact_type_enforcement_is_load_bearing(self, monkeypatch):
        # MUTATION (permanent, in-suite): the exact-type checks relaxed back
        # to isinstance-based acceptance -> the r2-F2 str-subclass carrier is
        # ACCEPTED again on a fresh construction (the defect class reappears);
        # restored -> refused with the exact-type reason.
        import models.registry.model_prep as mp_module

        class Ref(str):
            pass

        monkeypatch.setattr(
            mp_module, "_present", lambda v: isinstance(v, str) and bool(v.strip())
        )
        monkeypatch.setattr(
            mp_module,
            "_canonical_sha256",
            lambda v: isinstance(v, str) and is_sha256_hex(v),
        )
        carrier_member = StructureMemberBinding(
            structure_ref=Ref("artifact:structure-a"),
            structure_sha256=sha("artifact:structure-a"),
            model_ref="coronary-seg@1.2.0",
            release_sha256=sha("release-a"),
            provenance_ref="provenance:artifact:structure-a",
            provenance_sha256=sha("prov-artifact:structure-a"),
        )
        assert type(carrier_member.structure_ref) is Ref  # the defect class
        monkeypatch.undo()
        with pytest.raises(CanonicalValidationError, match="exactly a plain str"):
            StructureMemberBinding(
                structure_ref=Ref("artifact:structure-a"),
                structure_sha256=sha("artifact:structure-a"),
                model_ref="coronary-seg@1.2.0",
                release_sha256=sha("release-a"),
                provenance_ref="provenance:artifact:structure-a",
                provenance_sha256=sha("prov-artifact:structure-a"),
            )


class TestR2F3RefusalValidatedAtUse:
    """R2-F3: a refusal must be unable to express proceed through ANY route.

    Reviewer reproductions, all refused now:
    (a) ``object.__new__`` + the generated ``__setstate__`` produces
        CLEAR/GOOD that survives pickle;
    (b) a subclass overriding ``__post_init__`` constructs CLEAR/GOOD and
        accepts an attached member via ``object.__setattr__``;
    (c) a corrupted module mapping admits an unknown case.
    """

    def test_negative_new_plus_setstate_proceed_refused(self):
        forged = ModelPrepRefusal.__new__(ModelPrepRefusal)
        with pytest.raises(CanonicalValidationError, match="cannot be expressed as a proceed"):
            forged.__setstate__(
                [
                    "Missing model record",
                    QCAbstentionCode.QC_REQUIRED_PAIR_ABSENT,
                    QCOutcome.CLEAR,
                    QualityGrade.GOOD,
                ]
            )

    def test_negative_proceed_shaped_refusal_cannot_resurrect_through_pickle(self):
        # forge by writing slots directly (bypasses __init__ AND __setstate__),
        # then attempt to resurrect it: __setstate__ re-validates the state.
        forged = ModelPrepRefusal.__new__(ModelPrepRefusal)
        object.__setattr__(forged, "safety_case", "Missing model record")
        object.__setattr__(forged, "abstention_code", QCAbstentionCode.QC_REQUIRED_PAIR_ABSENT)
        object.__setattr__(forged, "outcome", QCOutcome.CLEAR)
        object.__setattr__(forged, "quality_grade", QualityGrade.GOOD)
        with pytest.raises(CanonicalValidationError, match="cannot be expressed as a proceed"):
            pickle.loads(pickle.dumps(forged))

    def test_negative_refusal_subclass_refused(self):
        with pytest.raises(TypeError, match="subclassing ModelPrepRefusal"):

            class PseudoRefusal(ModelPrepRefusal):
                def __post_init__(self) -> None:
                    pass

    def test_negative_corrupted_module_mapping_admits_nothing(self, monkeypatch):
        # the type validates against the IMPORT-TIME frozen snapshot, not the
        # mutable module binding: an unknown case stays refused even when the
        # module attribute is corrupted.
        monkeypatch.setattr(
            model_prep,
            "SAFETY_CASE_ABSTENTION_CODE",
            MappingProxyType(
                {"Some unknown case": QCAbstentionCode.QC_GEOMETRY_INVALID}
            ),
        )
        with pytest.raises(CanonicalValidationError, match="unknown MODEL_PREP safety case"):
            ModelPrepRefusal(
                safety_case="Some unknown case",
                abstention_code=QCAbstentionCode.QC_GEOMETRY_INVALID,
            )
        with pytest.raises(CanonicalValidationError, match="unknown MODEL_PREP safety case"):
            refusal_for_case("Some unknown case")
        monkeypatch.undo()
        assert refusal_for_case("Missing model record").abstention_code == (
            QCAbstentionCode.QC_REQUIRED_PAIR_ABSENT
        )

    def test_use_time_validator_refuses_proceed_shaped_values(self):
        # the point-of-USE contract: any consumer can re-check any refusal
        # and a proceed-shaped value is refused by value, not by route.
        assert refusal_problems(refusal_for_case("Missing model record")) == []
        probe = SimpleNamespace(
            safety_case="Missing model record",
            abstention_code=QCAbstentionCode.QC_REQUIRED_PAIR_ABSENT,
            outcome=QCOutcome.CLEAR,
            quality_grade=QualityGrade.GOOD,
        )
        problems = refusal_problems(probe)
        assert any("must be exactly ModelPrepRefusal" in p for p in problems)
        # slot closure on an exact instance (unchanged, re-asserted):
        refusal = refusal_for_case("Missing model record")
        with pytest.raises(AttributeError):
            object.__setattr__(refusal, "substitute_member", member("artifact:structure-x"))

    def test_mutation_setstate_sealing_is_load_bearing(self, monkeypatch):
        # MUTATION (permanent, in-suite): the sealed __setstate__ removed
        # (the dataclass-generated silent assigner restored: state applied
        # to slots with NO validation) -> the r2-F3 forged CLEAR/GOOD
        # refusal is constructible through __new__ + __setstate__;
        # restored -> refused.
        import models.registry.model_prep as mp_module

        monkeypatch.setattr(
            mp_module, "_deserialized_state", lambda n, s, names: dict(zip(names, s))
        )
        monkeypatch.setattr(
            mp_module, "_refusal_field_problems", lambda values: []
        )
        forged = ModelPrepRefusal.__new__(ModelPrepRefusal)
        forged.__setstate__(
            [
                "Missing model record",
                QCAbstentionCode.QC_REQUIRED_PAIR_ABSENT,
                QCOutcome.CLEAR,
                QualityGrade.GOOD,
            ]
        )
        assert forged.outcome == QCOutcome.CLEAR  # the defect class
        monkeypatch.undo()
        forged2 = ModelPrepRefusal.__new__(ModelPrepRefusal)
        with pytest.raises(CanonicalValidationError, match="cannot be expressed as a proceed"):
            forged2.__setstate__(
                [
                    "Missing model record",
                    QCAbstentionCode.QC_REQUIRED_PAIR_ABSENT,
                    QCOutcome.CLEAR,
                    QualityGrade.GOOD,
                ]
            )

    def test_control_valid_refusal_validates_at_use(self):
        for case in SAFETY_CASE_ABSTENTION_CODE:
            assert refusal_problems(refusal_for_case(case)) == []


class TestR3F2PrivateConstructionSeal:
    """R3-F2/R3-F4 lineage: the seal is NOT caller-authorable, and the only
    member derivation route is the COMPLETE fresh construction (§10.2).

    The r3 review reproduced the att0005 defects fixture-free: resetting the
    public seal field to its construction sentinel (``binding_seal=''``)
    resealed and accepted a one-field ``replace`` patch; recomputing the
    documented public payload digest after an ``object.__setattr__`` slot
    patch also passed. The seal is therefore now a PRIVATE slot (single
    writer, single derivation, no sentinel semantics, excluded from
    replace()/repr/compare/asdict/the declaration digest), and
    ``dataclasses.replace`` on a component is STRUCTURALLY refused — the
    FAIL-CLOSED branch, because full offline mutual consistency between
    ``structure_ref`` and ``provenance_sha256`` is NOT derivable in this
    reference-only layer (the digests name records the declaration does not
    resolve).
    """

    def test_negative_the_reviewer_seal_reset_construction_is_refused(self):
        # EXACT reviewer construction:
        # replace(pair.first, structure_ref=..., binding_seal='') — the seal
        # is not an __init__ parameter at all: replace cannot carry it
        # (unknown kwarg) and cannot even reach the tripwire past it.
        d = declaration()
        with pytest.raises((TypeError, ValueError)):
            replace(d.first, structure_ref="artifact:replacement", binding_seal="")  # type: ignore[call-arg]
        # and the enclosing swap never happens — the member route is refused
        # at the inner step, with the original pair intact:
        assert pair_binding_problems(d) == []
        assert d.first.structure_ref == "artifact:structure-a"

    def test_negative_one_field_member_replace_is_structurally_refused(self):
        # r3-F2 fail-closed branch: the ONE-FIELD patch route (new identity,
        # old evidence bindings — the chimera) is refused BEFORE any value
        # exists: dataclasses.replace on a component raises.
        d = declaration()
        with pytest.raises(ValueError, match="_complete_fresh_construction_only"):
            replace(d.first, structure_ref="artifact:replacement")  # type: ignore[call-arg]
        for component in (d.source, d.comparator):
            with pytest.raises(ValueError, match="_complete_fresh_construction_only"):
                replace(component)  # type: ignore[call-arg]
        # a caller who tries to satisfy the tripwire explicitly hits the
        # custom __init__, which has no such parameter:
        with pytest.raises(TypeError):
            replace(d.first, structure_ref="artifact:replacement",
                    _complete_fresh_construction_only=True)
        # the exact r2-F4 construction is refused at the inner step:
        with pytest.raises((TypeError, ValueError)):
            replace(d, first=replace(d.first, structure_ref="artifact:replacement"))  # type: ignore[call-arg]
        assert pair_binding_problems(d) == []
        assert d.first.structure_ref == "artifact:structure-a"  # original intact

    def test_negative_seal_recomputation_bypass_is_refused(self):
        # EXACT reviewer construction (r3 evidence): patch ONE payload slot
        # through object.__setattr__, then install the recomputed PUBLIC
        # payload digest as the seal — the att0005 seal was exactly the
        # public payload digest, so this passed. att0010 closes it TWICE:
        # (1) the patch itself is refused at WRITE time by the payload lock
        # (there is no patch route at all); (2) the PUBLIC payload digest is
        # not the seal even if it were installed — the seal is COMPUTED over
        # a domain-separated private material, so the public digest never
        # validates as a seal.
        d = declaration()
        m = d.first
        original_seal = m.binding_seal
        # (1) the patch is refused at write time — nothing lands:
        with pytest.raises(CanonicalValidationError, match="construction-only"):
            object.__setattr__(m, "structure_ref", "artifact:replacement")
        assert m.structure_ref == "artifact:structure-a"
        assert m.binding_seal == original_seal
        assert pair_binding_problems(d) == []
        # (2) the public payload digest is NOT the seal: a state carrying it
        # in the seal position is refused at load, and the private
        # derivation is domain-separated from it.
        public_payload_digest = canonical_digest(
            {name: getattr(m, name) for name in _MEMBER_PAYLOAD_FIELDS}
        )
        assert public_payload_digest != m.binding_seal
        forged = StructureMemberBinding.__new__(StructureMemberBinding)
        with pytest.raises(CanonicalValidationError, match="own seal does not match"):
            forged.__setstate__(
                tuple(getattr(m, n) for n in _MEMBER_PAYLOAD_FIELDS)
                + (public_payload_digest,)
            )
        # and there is no writable surface for the seal name at all:
        with pytest.raises((AttributeError, TypeError)):
            object.__setattr__(m, "_binding_seal", public_payload_digest)
        assert m.binding_seal == original_seal

    def test_negative_sentinel_semantics_refused_entirely(self):
        # r3-F2: no "fresh on empty" exists. A missing, empty, or None seal
        # is REFUSED — a forged member with no seal cannot validate.
        # R5-F1: the sentinel values can no longer even be INSTALLED (the
        # seal name has no writable surface), so every probe is a fresh
        # unsealed target and the USE-TIME validation is what refuses.
        from models.registry.model_prep import _member_use_problems

        # R6-F1: a payload can be written ONLY through the sanctioned
        # construction/deserialization frames, so a "forged" target is built
        # by the disclosed multi-hop residual (a plain module attribute ->
        # dict -> member descriptor -> __set__) — the SAME route the
        # mutation probes use. It is a raw slot fill with NO seal, and the
        # point of this probe is what the value layer does with it.
        forged = StructureMemberBinding.__new__(StructureMemberBinding)
        payload = {
            "structure_ref": "artifact:structure-a",
            "structure_sha256": sha("artifact:structure-a"),
            "model_ref": "coronary-seg@1.2.0",
            "release_sha256": sha("release-a"),
            "provenance_ref": "provenance:artifact:structure-a",
            "provenance_sha256": sha("prov-artifact:structure-a"),
        }
        for name, value in payload.items():
            _write_payload_raw(forged, name, value)
        # att0010: the seal is COMPUTED, so an EMPTY payload slot is what
        # makes a target "not a value" — a complete payload now derives its
        # own seal. There is therefore no sentinel to install and no
        # "fresh on empty": the sentinel is refused at the STATE level
        # (a serialized state whose seal element is a sentinel):
        payload_values = tuple(payload[n] for n in _MEMBER_PAYLOAD_FIELDS)
        for sentinel in ("", None, "fresh", "0" * 64):
            probe = StructureMemberBinding.__new__(StructureMemberBinding)
            with pytest.raises(CanonicalValidationError):
                probe.__setstate__(payload_values + (sentinel,))
            # and the target is still EMPTY (buffer-then-commit: a refused
            # __setstate__ writes nothing — R6-F2):
            assert _member_use_problems_fresh(probe), sentinel
        # an INCOMPLETE payload (one slot never written) is refused at use
        # time — the missing field is named, and no sentinel rescues it:
        partial = StructureMemberBinding.__new__(StructureMemberBinding)
        for name in list(payload)[:-1]:
            _write_payload_raw(partial, name, payload[name])
        problems = _member_use_problems(partial)
        assert problems and any("provenance_sha256" in p for p in problems)
        # exact-shape str-subclass carrier (r3-F3): a carrier in the seal
        # element of a state is refused by the exact-plain-str check.
        expected = _derive_binding_seal(
            "structure member binding", _MEMBER_PAYLOAD_FIELDS, payload
        )

        class SealCarrier(str):
            pass

        carrier = SealCarrier(expected)
        carrier.approved = True  # type: ignore[attr-defined]
        probe = StructureMemberBinding.__new__(StructureMemberBinding)
        with pytest.raises(CanonicalValidationError, match="own seal does not match"):
            probe.__setstate__(
                tuple(payload[n] for n in _MEMBER_PAYLOAD_FIELDS) + (carrier,)
            )
        # CONTROL: the correct exact-str seal validates — and it is
        # COMPUTED, not installed (a complete fresh construction derives the
        # same value over the same payload):
        real = member("artifact:structure-a", digest=payload["structure_sha256"])
        assert _member_use_problems(real) == []
        assert real.binding_seal == expected  # the derivation is deterministic
        assert type(real.binding_seal) is str

    def test_control_the_only_member_route_is_complete_fresh_construction(self):
        # CONTROL: the sanctioned route — every component restated together
        # in one complete fresh construction — constructs, seals, digests
        # and validates; the swap is a pair-level replace with the fresh
        # member, which moves the content address and leaves the original
        # intact (§10.2).
        d = declaration()
        fresh = StructureMemberBinding(
            structure_ref="artifact:structure-c",
            structure_sha256=sha("artifact:structure-c"),
            model_ref="coronary-seg@1.2.0",
            release_sha256=sha("release-c"),
            provenance_ref="provenance:artifact:structure-c",
            provenance_sha256=sha("prov-artifact:structure-c"),
        )
        swapped = replace(d, first=fresh)
        assert swapped.declaration_sha256 != d.declaration_sha256
        assert d.first.structure_ref == "artifact:structure-a"  # original intact
        assert pair_binding_problems(swapped) == []
        # the seal is caller-INSPECTABLE but not caller-AUTHORABLE:
        assert type(fresh.binding_seal) is str and is_sha256_hex(fresh.binding_seal)
        assert fresh.binding_seal != canonical_digest(
            {name: getattr(fresh, name) for name in (
                "structure_ref", "structure_sha256", "model_ref",
                "release_sha256", "provenance_ref", "provenance_sha256",
            )}
        )
        # the seal is excluded from replace()/repr/compare/asdict/digest:
        assert {f.name for f in dc_fields(StructureMemberBinding)} == {
            "structure_ref", "structure_sha256", "model_ref",
            "release_sha256", "provenance_ref", "provenance_sha256",
        }
        assert "_binding_seal" not in repr(fresh)
        assert "_binding_seal" not in dataclasses.asdict(fresh)
        assert fresh == StructureMemberBinding(
            structure_ref="artifact:structure-c",
            structure_sha256=sha("artifact:structure-c"),
            model_ref="coronary-seg@1.2.0",
            release_sha256=sha("release-c"),
            provenance_ref="provenance:artifact:structure-c",
            provenance_sha256=sha("prov-artifact:structure-c"),
        )
        assert "binding_seal" not in canonical_json(swapped)

    def test_control_declaration_digest_excludes_the_seal_by_construction(self):
        # r3-F3: the seal cannot smuggle authority because it never enters
        # the canonical digest surface at all (it is not a field, and it
        # is not even instance state since R5-F1). Rewriting a member's
        # seal is structurally IMPOSSIBLE (no writable surface), and the
        # canonical JSON of the pair carries no seal key anywhere.
        d = declaration()
        m = d.first
        original_digest = d.declaration_sha256
        # att0010: the seal is COMPUTED, so it cannot be rewritten at all —
        # the seal NAME has no write surface (it is not a slot, not a
        # field, and not instance state), and the property has no setter:
        with pytest.raises((AttributeError, TypeError)):
            object.__setattr__(m, "_binding_seal", "f" * 64)
        with pytest.raises(AttributeError):
            setattr(m, "binding_seal", "f" * 64)
        # the seal is therefore a pure function of the payload, the digest
        # still equals itself (the seal was never inside it), and the
        # canonical JSON carries no seal key anywhere:
        assert m.binding_seal == d.first.binding_seal
        assert m.binding_seal == _derive_binding_seal(
            "structure member binding",
            _MEMBER_PAYLOAD_FIELDS,
            {name: getattr(m, name) for name in _MEMBER_PAYLOAD_FIELDS},
        )
        assert d.declaration_sha256 == original_digest
        assert "binding_seal" not in canonical_json(d)

    def test_mutation_seal_verification_is_load_bearing(self, monkeypatch):
        # MUTATION (permanent, in-suite; r3-F2): the use-time/digest-time
        # seal verification removed -> a state carrying a forged seal is
        # ACCEPTED (the defect class reappears, in memory, discarded);
        # restored -> refused. att0010: the seal is COMPUTED, so the
        # verification that is load-bearing is the one applied to a
        # SERIALIZED state (a state whose seal element does not match the
        # payload it claims was never produced by __getstate__).
        import models.registry.model_prep as mp_module

        m = member()
        public_payload_digest = canonical_digest(
            {name: getattr(m, name) for name in _MEMBER_PAYLOAD_FIELDS}
        )
        payload_values = tuple(
            getattr(m, name) for name in _MEMBER_PAYLOAD_FIELDS
        )
        monkeypatch.setattr(mp_module, "_seal_problems", lambda *a, **k: [])
        try:
            # with verification gone, the forged-seal state is accepted:
            target = StructureMemberBinding.__new__(StructureMemberBinding)
            target.__setstate__(payload_values + (public_payload_digest,))
            assert target.binding_seal == m.binding_seal  # the defect class
        finally:
            monkeypatch.undo()
        # restored: the SAME forged state is refused, and the target is left
        # EMPTY (buffer-then-commit: a refused __setstate__ writes nothing):
        target2 = StructureMemberBinding.__new__(StructureMemberBinding)
        with pytest.raises(CanonicalValidationError, match="own seal does not match"):
            target2.__setstate__(payload_values + (public_payload_digest,))
        assert _member_use_problems_fresh(target2)
        # and the seal name still has no write surface:
        with pytest.raises((AttributeError, TypeError)):
            object.__setattr__(m, "_binding_seal", public_payload_digest)
        assert _member_use_problems(m) == []

    def test_mutation_the_private_derivation_is_load_bearing(self, monkeypatch):
        # MUTATION (r3-F2): the private, domain-separated derivation
        # replaced by the PUBLIC payload digest -> the seal becomes publicly
        # reproducible again (the att0005 defect class: recomputing the
        # documented payload digest defeats tamper detection); restored ->
        # the recomputation attack is refused.
        import models.registry.model_prep as mp_module

        def public_digest(label, payload_fields, values):
            return canonical_digest(
                {name: values.get(name) for name in payload_fields}
            )

        m = member()
        monkeypatch.setattr(mp_module, "_derive_binding_seal", public_digest)
        try:
            # the defect class: the computed seal is now exactly the
            # documented PUBLIC payload digest, so a caller can reproduce
            # it without the private domain separation:
            assert m.binding_seal == canonical_digest(
                {name: getattr(m, name) for name in _MEMBER_PAYLOAD_FIELDS}
            )
        finally:
            monkeypatch.undo()
        # restored: the private derivation is domain-separated, so the
        # public payload digest is NOT the seal and a state carrying it is
        # refused:
        public_payload_digest = canonical_digest(
            {name: getattr(m, name) for name in _MEMBER_PAYLOAD_FIELDS}
        )
        assert public_payload_digest != m.binding_seal
        target = StructureMemberBinding.__new__(StructureMemberBinding)
        with pytest.raises(CanonicalValidationError, match="own seal does not match"):
            target.__setstate__(
                tuple(getattr(m, n) for n in _MEMBER_PAYLOAD_FIELDS)
                + (public_payload_digest,)
            )

    def test_mutation_the_replace_tripwire_is_load_bearing(self, monkeypatch):
        # MUTATION (r3-F2): the structural replace-refusal removed -> a
        # one-field member patch becomes derivable again (the att0005/r2-F4
        # chimera: new identity wearing old evidence bindings); restored ->
        # refused structurally.
        # simulate removing the tripwire from the dataclass field table:
        fields_map = dict(StructureMemberBinding.__dataclass_fields__)
        tripwire = fields_map.pop("_complete_fresh_construction_only")
        monkeypatch.setattr(
            StructureMemberBinding, "__dataclass_fields__", fields_map
        )
        try:
            m = member()
            rebuilt = replace(m, structure_ref="artifact:replacement")
            assert rebuilt.structure_ref == "artifact:replacement"  # defect class
        finally:
            fields_map["_complete_fresh_construction_only"] = tripwire
            monkeypatch.undo()
        with pytest.raises(ValueError, match="_complete_fresh_construction_only"):
            replace(member(), structure_ref="artifact:replacement")


# ---------------------------------------------------------------------------
# R4 — CONSTRUCTION-ONLY seal authoring (correction round 4: R4-F1/R4-F2)
# ---------------------------------------------------------------------------


class TestR4F1ConstructionOnlySealAuthoring:
    """R4-F1: the seal AUTHORING SURFACE is construction-only.

    The r4 review reproduced, fixture-free, that the r3 mechanism kept
    moving WHERE the seal lives while the authoring surface stayed
    reachable: the public ``__setstate__`` accepted caller-supplied payload
    state and wrote a fresh seal over it (a one-field patch kept the old
    digests and passed everything), and the module attributes
    ``_seal_into``/``_derive_binding_seal`` resealed existing objects
    directly. att0008 guarded the slot with a write-once descriptor; the
    r5 review then reproduced that the guard's retained RAW member
    descriptor (``_seal_slot_guard._member``) was itself a no-introspection
    re-seal and delete handle (R5-F1); att0009 moved the seal bytes into a
    module-private vault that r6 reached by a plain closure-cell chain
    (R6-F1). The att0010 fix DELETES THE STORAGE:

    - the seal is COMPUTED (``binding_seal`` is a property evaluating the
      private derivation at every read) — there is no slot, no vault, no
      table, no ``id()`` key and no ``_SEAL_MACHINERY``, so there is NO
      SEAL TO RE-BLESS and no ``_seal_into`` to call;
    - each payload field carries a frame-gated write lock: the ONE-FIELD
      patch at the heart of every r4/r5/r6 construction is refused at
      WRITE time (``object.__setattr__`` included), because only this
      module's ``__init__``/``__setstate__`` frames may write a payload;
    - ``__setstate__`` accepts ONLY construction-shaped state (payload +
      the seal the state was serialized with, verified against the
      private derivation over the state's OWN payload), validates it IN
      FULL and checks the EMPTY-TARGET requirement BEFORE writing (R6-F2
      buffer-then-commit), so it is deserialization-only and never a
      resealer;
    - ``PairBindingDeclaration.__setstate__`` refuses populated targets.

    Every negative below is the reviewer's EXACT construction. Every
    control proves a legitimate path still works. The mutation probes
    prove each guard is load-bearing (disabled -> the reviewer's
    construction re-succeeds; restored -> refused).
    """

    # -- (a) the reviewer's primary construction ----------------------------

    def test_negative_r4_route_1_setstate_reseal_of_a_patched_member(self):
        # EXACT r4 construction: valid member -> one-field object.__setattr__
        # -> member.__setstate__(tuple of current payload values).
        # att0010: the construction is now refused at its FIRST step — the
        # one-field patch has no route at all (§10.2) — so there is no
        # patched payload left to re-bless.
        d = declaration()
        m = d.first
        before = {name: getattr(m, name) for name in _MEMBER_PAYLOAD_FIELDS}
        with pytest.raises(CanonicalValidationError, match="construction-only"):
            object.__setattr__(m, "structure_ref", "artifact:replacement")
        # nothing moved: the payload is byte-identical and the pair is valid
        assert {name: getattr(m, name) for name in _MEMBER_PAYLOAD_FIELDS} == before
        assert pair_binding_problems(d) == []
        # the remaining r4 variants are refused on their own merits:
        state = tuple(getattr(m, name) for name in _MEMBER_PAYLOAD_FIELDS)
        # (a) a BARE payload tuple is not deserialization data:
        with pytest.raises(CanonicalValidationError, match="CONSTRUCTION-SHAPED"):
            m.__setstate__(state)
        # (b) payload + the CURRENT seal, on this populated target: refused
        # because the target is not empty (never an in-place mutation) —
        # and it stays refused with a seal freshly derived over any
        # caller-chosen payload:
        st = state + (m.binding_seal,)
        with pytest.raises(CanonicalValidationError, match="already populated"):
            m.__setstate__(st)
        vals = dict(before)
        vals["structure_ref"] = "artifact:replacement"
        derived = _derive_binding_seal(
            "structure member binding", _MEMBER_PAYLOAD_FIELDS, vals
        )
        with pytest.raises(CanonicalValidationError, match="already populated"):
            m.__setstate__(
                tuple(vals[n] for n in _MEMBER_PAYLOAD_FIELDS) + (derived,)
            )
        # and the target is UNCHANGED by every refusal (R6-F2):
        assert {name: getattr(m, name) for name in _MEMBER_PAYLOAD_FIELDS} == before
        # (c) the SAME derived state onto an EMPTY target is a complete fresh
        # construction (a second spelling of it — no old digests retained);
        # it is not a re-bless of the existing member:
        fresh = StructureMemberBinding.__new__(StructureMemberBinding)
        fresh.__setstate__(
            tuple(vals[n] for n in _MEMBER_PAYLOAD_FIELDS) + (derived,)
        )
        assert fresh.structure_ref == "artifact:replacement"
        assert fresh is not m
        assert d.first is m

    def test_negative_r4_route_2_direct_seal_into_on_existing_objects(self):
        # EXACT r4 construction: mp._seal_into(existing_object, ...) —
        # att0010: there is NO seal writer left to call (the seal is
        # computed, not stored), so the module attribute does not exist and
        # the payload-write route it used is refused for all three
        # component types.
        import models.registry.model_prep as mp_module

        assert not hasattr(mp_module, "_seal_into")
        assert not hasattr(mp_module, "_SEAL_MACHINERY")
        assert not hasattr(mp_module, "_SealVault")
        assert not hasattr(mp_module, "_is_unsealed")
        d = declaration()
        for obj, field in (
            (d.first, "structure_ref"),
            (d.source, "volume_ref"),
            (d.comparator, "algorithm_id"),
        ):
            with pytest.raises(CanonicalValidationError, match="construction-only"):
                object.__setattr__(obj, field, "artifact:replacement")
        # a fresh object from a FOREIGN frame (this test) is refused too —
        # the payload lock admits only the module's own frames:
        blank = StructureMemberBinding.__new__(StructureMemberBinding)
        with pytest.raises(CanonicalValidationError, match="construction-only"):
            object.__setattr__(blank, "structure_ref", "artifact:replacement")

    def test_negative_r4_route_3_derive_plus_manual_seal_write(self):
        # EXACT r4 construction: mp._derive_binding_seal(...) + manual
        # _binding_seal write. att0010: the seal is COMPUTED, so (i) the
        # one-field patch it was meant to bless is refused at WRITE time,
        # and (ii) the seal name has NO writable surface at all — it is not
        # a slot, not a field, and not instance state — so the derived value
        # cannot be installed by any spelling.
        m = member()
        original_seal = m.binding_seal
        with pytest.raises(CanonicalValidationError, match="construction-only"):
            object.__setattr__(m, "structure_ref", "artifact:replacement")
        vals = {name: getattr(m, name) for name in _MEMBER_PAYLOAD_FIELDS}
        derived = _derive_binding_seal(
            "structure member binding", _MEMBER_PAYLOAD_FIELDS, vals
        )
        # every install spelling is refused (no write surface anywhere):
        with pytest.raises(AttributeError):
            object.__setattr__(m, "_binding_seal", derived)
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(m, "_binding_seal", derived)
        with pytest.raises(AttributeError):
            setattr(m, "binding_seal", derived)  # the property has no setter
        # delattr routes refused (FrozenInstanceError IS an AttributeError):
        with pytest.raises(AttributeError):
            delattr(m, "_binding_seal")
        with pytest.raises(AttributeError):
            object.__delattr__(m, "_binding_seal")
        # the seal is what it computes to — nothing was installed:
        assert m.binding_seal == original_seal == derived
        assert _member_use_problems(m) == []

    def test_negative_r4_route_4_bare_payload_setstate_refused(self):
        # (d) __setstate__ with a caller-supplied bare payload tuple (no
        # seal) — on a fresh target AND on an existing member — is refused:
        m = member()
        bare = tuple(getattr(m, name) for name in _MEMBER_PAYLOAD_FIELDS)
        fresh = StructureMemberBinding.__new__(StructureMemberBinding)
        with pytest.raises(CanonicalValidationError, match="CONSTRUCTION-SHAPED"):
            fresh.__setstate__(bare)
        with pytest.raises(CanonicalValidationError, match="CONSTRUCTION-SHAPED"):
            m.__setstate__(bare)
        # a mapping without the seal element is refused by shape:
        with pytest.raises(CanonicalValidationError, match="CONSTRUCTION-SHAPED"):
            fresh.__setstate__(
                {name: getattr(m, name) for name in _MEMBER_PAYLOAD_FIELDS}
            )
        # a state whose seal does not match its payload is refused:
        with pytest.raises(CanonicalValidationError, match="own seal does not match"):
            fresh.__setstate__(bare + ("f" * 64,))

    def test_negative_r4_route_5_serialization_cannot_launder_a_reseal(self):
        # (e) serialization must not launder a value the validators refuse.
        # att0010: an in-place patch has NO route, so the laundering probe
        # covers the only remaining shape — a value whose payload was filled
        # by the disclosed multi-hop residual and is therefore NOT a value
        # the sanctioned routes can produce (an incomplete payload):
        # __getstate__ runs the full use-time validation BEFORE emitting.
        m = member()
        blob = pickle.dumps(m)
        patched = pickle.loads(blob)
        with pytest.raises(CanonicalValidationError, match="construction-only"):
            object.__setattr__(patched, "structure_sha256", sha("TAMPERED"))
        assert pickle.dumps(patched) == blob  # untampered -> serializes clean
        # a target filled by the residual with an INCOMPLETE payload is
        # refused at dump time (the missing field is named):
        partial = StructureMemberBinding.__new__(StructureMemberBinding)
        for name in list(_MEMBER_PAYLOAD_FIELDS)[:-1]:
            _write_payload_raw(partial, name, getattr(m, name))
        with pytest.raises(CanonicalValidationError, match="provenance_sha256"):
            pickle.dumps(partial)
        with pytest.raises(CanonicalValidationError, match="provenance_sha256"):
            copy.copy(partial)
        with pytest.raises(CanonicalValidationError, match="provenance_sha256"):
            copy.deepcopy(partial)
        # and a FORGED blob (state whose seal does not match its payload)
        # is refused at LOAD time — pickle's reconstruction calls the
        # same validated __setstate__:
        state = tuple(getattr(member(), name) for name in _MEMBER_PAYLOAD_FIELDS)
        forged_blob = pickle.dumps(
            tuple(list(state) + [sha("not-the-payload")]),
            protocol=pickle.HIGHEST_PROTOCOL,
        )
        forged = pickle.loads(forged_blob)  # plain tuple: loads fine
        target = StructureMemberBinding.__new__(StructureMemberBinding)
        with pytest.raises(CanonicalValidationError, match="own seal does not match"):
            target.__setstate__(forged)

    def test_negative_r4_route_6_source_comparator_and_pair_routes(self):
        # (f) the same routes on SourceVolumeBinding.volume_ref and
        # ComparatorBinding.algorithm_id -> REFUSED.
        d = declaration()
        s = d.source
        # att0010: the one-field patch is refused at WRITE time on both
        # components, and the bare-payload __setstate__ it fed is refused
        # on its own merits (not deserialization data):
        with pytest.raises(CanonicalValidationError, match="construction-only"):
            object.__setattr__(s, "volume_ref", "volume_id:REPLACED000000000000000")
        with pytest.raises(CanonicalValidationError, match="construction-only"):
            object.__setattr__(c_ := d.comparator, "algorithm_id", "other-comparator")
        svals = tuple(getattr(s, name) for name in _SOURCE_PAYLOAD_FIELDS)
        with pytest.raises(CanonicalValidationError, match="CONSTRUCTION-SHAPED"):
            s.__setstate__(svals)
        c = c_
        cvals = tuple(getattr(c, name) for name in _COMPARATOR_PAYLOAD_FIELDS)
        with pytest.raises(CanonicalValidationError, match="CONSTRUCTION-SHAPED"):
            c.__setstate__(cvals)
        # PairBindingDeclaration one-component __setstate__ in-place
        # mutation -> REFUSED (populated target). NOTE: this probe uses a
        # FRESH pair so the only problem visible is the populated-target
        # refusal (a pair whose members were tampered above would fail
        # component validation first — the layers compose).
        d2 = declaration()
        pair_state = [
            member("artifact:structure-c"),
            d2.second,
            d2.source,
            d2.comparator,
            d2.schema_version,
        ]
        with pytest.raises(
            CanonicalValidationError, match="populated declaration"
        ):
            d2.__setstate__(pair_state)
        assert d2.first.structure_ref == "artifact:structure-a"  # unchanged

    # -- controls: legitimate paths still work ------------------------------

    def test_control_r4_pickle_round_trips_all_protocols(self):
        m = member()
        s = source()
        c = comparator()
        d = declaration()
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            rm = pickle.loads(pickle.dumps(m, protocol=proto))
            assert rm == m and rm.binding_seal == m.binding_seal
            rs = pickle.loads(pickle.dumps(s, protocol=proto))
            assert rs == s and rs.binding_seal == s.binding_seal
            rc = pickle.loads(pickle.dumps(c, protocol=proto))
            assert rc == c and rc.binding_seal == c.binding_seal
            rd = pickle.loads(pickle.dumps(d, protocol=proto))
            assert rd == d and rd.declaration_sha256 == d.declaration_sha256

    def test_control_r4_copy_deepcopy_and_fresh_target_restore(self):
        m = member()
        assert copy.copy(m) == m
        assert copy.deepcopy(m) == m
        # fresh-target deserialization (sequence form and mapping form):
        state = m.__getstate__()
        fresh = StructureMemberBinding.__new__(StructureMemberBinding)
        fresh.__setstate__(state)
        assert fresh == m and fresh.binding_seal == m.binding_seal
        mapping = dict(zip(_MEMBER_PAYLOAD_FIELDS, state[:-1]))
        mapping["binding_seal"] = state[-1]
        fresh2 = StructureMemberBinding.__new__(StructureMemberBinding)
        fresh2.__setstate__(mapping)
        assert fresh2 == m and fresh2.binding_seal == m.binding_seal

    def test_control_r4_the_sanctioned_complete_swap_still_works(self):
        d = declaration()
        fresh = StructureMemberBinding(
            structure_ref="artifact:structure-c",
            structure_sha256=sha("artifact:structure-c"),
            model_ref="coronary-seg@1.2.0",
            release_sha256=sha("release-c"),
            provenance_ref="provenance:artifact:structure-c",
            provenance_sha256=sha("prov-artifact:structure-c"),
        )
        swapped = replace(d, first=fresh)
        assert swapped.declaration_sha256 != d.declaration_sha256
        assert d.first.structure_ref == "artifact:structure-a"
        assert pair_binding_problems(swapped) == []
        # att0010: the seal is COMPUTED, so there is NO seal entry in any
        # class dict at all and no guard object of any kind — the payload
        # lock ships on every PAYLOAD name instead, and it stores no
        # descriptor (capturing it yields no write handle):
        for cls in (StructureMemberBinding, SourceVolumeBinding, ComparatorBinding):
            assert not hasattr(cls, "_seal_slot_guard"), cls.__name__
            assert "_binding_seal" not in cls.__dict__, cls.__name__
            assert "_binding_seal" not in cls.__slots__, cls.__name__
            assert type(cls.__dict__["binding_seal"]) is property, cls.__name__
            assert cls.__dict__["binding_seal"].fset is None, cls.__name__
            for name in cls.__slots__:
                guard = cls.__dict__[name]
                assert type(guard).__name__ == "_PayloadLock", (cls.__name__, name)
                # the guard carries NO descriptor and NO state but the name:
                assert not hasattr(guard, "__dict__")
                assert guard._name == name

    # -- mutations: each new guard is load-bearing --------------------------

    def test_mutation_r4_the_payload_write_lock_is_load_bearing(self, monkeypatch):
        # MUTATION (R6-F1): the payload lock's FRAME GATE removed -> the
        # plainest patch route of all (``object.__setattr__`` on one payload
        # field of an existing binding) RE-OPENS — the exact r4/r5/r6 first
        # step; restored -> refused. R7-F2 layering: the write path now
        # carries TWO further independent gates (the r7 relay-frame gate in
        # ``_write_slot`` and the shared empty-target predicate), so the
        # defect class lands only when ALL of them are neutered together —
        # no single neutered gate re-opens a one-call patch route.
        import models.registry.model_prep as mp_module

        d = declaration()
        m = d.first
        before = {name: getattr(m, name) for name in _MEMBER_PAYLOAD_FIELDS}
        original_digest = d.declaration_sha256

        # the r7 gates ALONE keep the route closed even with the lock's frame
        # gate neutered (defense-in-depth, R7-F2):
        monkeypatch.setattr(
            mp_module._PayloadLock,
            "_codes",
            frozenset({_current_frame_code()}),
        )
        with pytest.raises(CanonicalValidationError):
            object.__setattr__(m, "structure_ref", "artifact:replacement")
        assert {name: getattr(m, name) for name in _MEMBER_PAYLOAD_FIELDS} == before

        # ALL gates neutered -> the one-field patch LANDS (the defect class):
        monkeypatch.setattr(
            mp_module, "_relay_frame_authorized", lambda caller: True
        )
        monkeypatch.setattr(
            mp_module, "_empty_target_problems", lambda instance, names: []
        )
        try:
            object.__setattr__(m, "structure_ref", "artifact:replacement")
            assert m.structure_ref == "artifact:replacement"  # the defect class
            # the pair still validates (the seal is COMPUTED, so the patched
            # member simply derives a new seal) and its digest MOVES — a new
            # identity, not a re-blessed one (§10.4/§10.6):
            assert pair_binding_problems(d) == []
            assert d.declaration_sha256 != original_digest
        finally:
            monkeypatch.undo()
        # restored: the SAME plain patch is refused at WRITE time, and the
        # value that was patched in the mutation branch is left alone:
        m2 = member()
        with pytest.raises(CanonicalValidationError, match="construction-only"):
            object.__setattr__(m2, "structure_ref", "artifact:replacement")
        assert m2.structure_ref == "artifact:structure-a"
        # and the still-tampered member from the mutation branch is now the
        # one value no sanctioned route produced — it validates as a fresh
        # identity would (the att0010 honest boundary), while the ORIGINAL
        # pair is untouched by every subsequent refusal:
        assert d.first is m
        assert {name: getattr(m, name) for name in _MEMBER_PAYLOAD_FIELDS} != before

    def test_mutation_r4_the_setstate_shape_guard_is_load_bearing(self, monkeypatch):
        # MUTATION (R4-F1 A/C/D): the construction-shaped-state requirement
        # removed (the att0007 bare-payload __setstate__ semantics restored)
        # -> a caller-supplied bare payload becomes a value — the reviewer's
        # EXACT route-1 shape; restored -> refused.

        m = member()
        bare = tuple(getattr(m, name) for name in _MEMBER_PAYLOAD_FIELDS)

        def old_style_setstate(self, state_):
            values = dict(zip(_MEMBER_PAYLOAD_FIELDS, state_))
            for name in _MEMBER_PAYLOAD_FIELDS:
                _write_payload_raw(self, name, values[name])

        real_setstate = StructureMemberBinding.__setstate__
        StructureMemberBinding.__setstate__ = old_style_setstate  # type: ignore[method-assign, unused-ignore]
        try:
            target = StructureMemberBinding.__new__(StructureMemberBinding)
            target.__setstate__(bare)  # the defect class: no seal required
            assert target == m
        finally:
            StructureMemberBinding.__setstate__ = real_setstate  # type: ignore[method-assign, unused-ignore]
        # restored: the bare payload is refused by shape on BOTH a fresh
        # target and an existing member, and the target is left EMPTY:
        target2 = StructureMemberBinding.__new__(StructureMemberBinding)
        with pytest.raises(CanonicalValidationError, match="CONSTRUCTION-SHAPED"):
            target2.__setstate__(bare)
        assert _member_use_problems_fresh(target2)
        with pytest.raises(CanonicalValidationError, match="CONSTRUCTION-SHAPED"):
            m.__setstate__(bare)

    def test_mutation_r4_the_empty_target_requirement_is_load_bearing(self):
        # MUTATION (R6-F2): the EMPTY-TARGET requirement removed -> a
        # populated component can be mutated in place through
        # ``__setstate__`` (an in-place component rewrite, §10.2); restored
        # -> refused, with the target left byte-identical.
        import models.registry.model_prep as mp_module

        m = member()
        other = member("artifact:structure-z")
        state = other.__getstate__()
        real = mp_module._empty_target_problems
        mp_module._empty_target_problems = lambda instance, names: []
        try:
            m.__setstate__(state)
            assert m.structure_ref == "artifact:structure-z"  # the defect class
        finally:
            mp_module._empty_target_problems = real
        # (``m`` was mutated by the demonstration above and is discarded —
        # the assertions below use a fresh value.)
        m2 = member()
        before2 = {name: getattr(m2, name) for name in _MEMBER_PAYLOAD_FIELDS}
        with pytest.raises(CanonicalValidationError, match="already populated"):
            m2.__setstate__(state)
        # R6-F2: the refusal left the target 100% unchanged:
        assert {name: getattr(m2, name) for name in _MEMBER_PAYLOAD_FIELDS} == before2
        assert pair_binding_problems(declaration()) == []

    def test_mutation_r4_the_pair_populated_target_guard_is_load_bearing(self):
        # MUTATION: the pair __setstate__ populated-target refusal removed
        # -> the reviewer's route-7 (one-component in-place swap) LANDS;
        # restored -> refused.
        import models.registry.model_prep as mp_module

        d = declaration()
        pair_state = [
            member("artifact:structure-c"),
            d.second,
            d.source,
            d.comparator,
            d.schema_version,
        ]
        real = mp_module._empty_target_problems
        mp_module._empty_target_problems = lambda instance, names: []
        try:
            d.__setstate__(pair_state)
            assert d.first.structure_ref == "artifact:structure-c"  # defect class
        finally:
            mp_module._empty_target_problems = real
        d2 = declaration()
        st2 = [member("artifact:structure-c"), d2.second, d2.source,
               d2.comparator, d2.schema_version]
        with pytest.raises(CanonicalValidationError, match="populated declaration"):
            d2.__setstate__(st2)
        assert d2.first.structure_ref == "artifact:structure-a"  # unchanged

    def test_the_r4_guards_cannot_go_quiet(self):
        # the R4/R5 guard machinery is anchored in the module source: if the
        # vault, the machinery factory, the construction-only writer checks,
        # or the construction-shaped state validator were deleted, these
        # markers disappear and the in-suite mutation probes above also lose
        # their patch targets (which would fail loudly).
        source = Path(model_prep.__file__).read_text(encoding="utf-8")
        for anchor in (
            "_PayloadLock",
            "_install_payload_locks",
            "_AUTHORIZED_PAYLOAD_FRAMES",
            "_SLOT_READERS",
            "_construction_shaped_state",
            "_empty_target_problems",
            "_commit_state",
            "_NOT_DESERIALIZATION_PROBLEM",
            "_PAYLOAD_WRITE_PROBLEM",
            "_PAYLOAD_DELETE_PROBLEM",
            "_POPULATED_TARGET_PROBLEM",
            # R7-F1/R7-F2: the complete-fresh-only construction gate and the
            # helper gates must ship with the module (they cannot go quiet):
            "_CONSTRUCTION_REENTRY_PROBLEM",
            "_construction_reentry_problems",
            "_relay_frame_authorized",
            "_construction_frame_authorized",
            "_payload_field_names",
        ):
            assert anchor in source, anchor
        # R6-F1: the STORAGE apparatus is GONE — not renamed, not guarded.
        # (The module docstring names the deleted machinery when it narrates
        # the six-round history, so the screen runs over the CODE only.)
        code = source.split('"""', 2)[-1]
        for deleted in (
            "class _SealVault",
            "class _SealMachinery",
            "_SEAL_MACHINERY",
            "_make_seal_machinery",
            "_install_seal_machinery",
            "_AUTHORIZED_SEAL_CALLERS",
            "_seal_into",
            "_is_unsealed",
            "_fresh_unsealed",
            "_seal_slot_guard",
            "_SEAL_DESCRIPTOR_TYPE_NAME",
        ):
            assert deleted not in code, f"the deleted name {deleted!r} is BACK"
        # ...and the module exports none of it:
        for deleted_attr in (
            "_SEAL_MACHINERY",
            "_SealVault",
            "_SealMachinery",
            "_is_unsealed",
            "_seal_into",
            "_fresh_unsealed",
        ):
            assert not hasattr(model_prep, deleted_attr), deleted_attr
        # the payload lock ships on every payload slot of every component,
        # and the seal is a COMPUTED property (never a slot, never stored):
        for cls in (StructureMemberBinding, SourceVolumeBinding, ComparatorBinding):
            assert not hasattr(cls, "_seal_slot_guard"), cls.__name__
            assert "_binding_seal" not in cls.__dict__, cls.__name__
            assert "_binding_seal" not in cls.__slots__, cls.__name__
            assert "__weakref__" not in cls.__slots__, cls.__name__
            entry = cls.__dict__["binding_seal"]
            assert type(entry) is property and entry.fset is None, cls.__name__
            for name in cls.__slots__:
                guard = cls.__dict__[name]
                assert type(guard).__name__ == "_PayloadLock", (cls.__name__, name)
                assert not hasattr(guard, "__dict__")  # no state carrier

# ---------------------------------------------------------------------------
# R6-F1/R6-F2 — the seal storage is DELETED (correction round 6): the
# delete-both re-bless route is structurally ABSENT, and a refused
# __setstate__ is ATOMIC.
# ---------------------------------------------------------------------------

class TestR6F1NoSealStorageAtAll:
    """R6-F1: the r6 delete-both construction is STRUCTURALLY ABSENT.

    The r6 review reproduced, at the exact att0009-r5 target, that
    ``mp._SEAL_MACHINERY[cls]`` is a plain module attribute whose
    closure-cell chain exposes BOTH seal tables as ordinary attribute
    accesses; deleting the object's id from both and calling the public
    ``write_once`` re-sealed a one-field patch of an existing sealed member
    (old digests retained, empty problems, freshly minted digest, copy/
    deepcopy/pickle 0-5 survival). This attempt was opened by re-running
    that EXACT construction at the dispatch HEAD and confirming it still
    ADMITTED (see the report).

    att0010's fix is not a better guard on the storage — it DELETES THE
    STORAGE:

    - ``binding_seal`` is a COMPUTED property (the private derivation
      evaluated at every read). There is no slot, no vault, no table, no
      ``id()`` key, no weakref and no ``_SEAL_MACHINERY``, so the
      delete-both construction has NOTHING to delete and ``write_once`` DOES
      NOT EXIST (asserted as an AttributeError, not merely noted).
    - the payload fields carry a frame-gated write lock; the one-field patch
      that every r4/r5/r6 construction begins with is refused at WRITE time.
    - R6-F2: ``__setstate__`` validates the incoming state FULLY and checks
      the EMPTY-TARGET requirement BEFORE writing a slot, so a refused
      restore leaves the target 100% unchanged.

    Each negative below is the r6 record's (or the r5 record's) EXACT
    construction. Each control proves a sanctioned path still works.
    """

    # -- (a) the r6 construction is structurally ABSENT ----------------------

    def test_negative_r6_the_seal_machinery_does_not_exist(self):
        # THE r6 ROUTE'S FIRST HOP IS GONE: `hasattr(mp, '_SEAL_MACHINERY')`
        # is False, so `mp._SEAL_MACHINERY[cls]` raises AttributeError and
        # the closure-cell chain to BOTH tables cannot even be started.
        import models.registry.model_prep as mp

        assert not hasattr(mp, "_SEAL_MACHINERY")
        with pytest.raises(AttributeError, match="_SEAL_MACHINERY"):
            mp._SEAL_MACHINERY[StructureMemberBinding]  # type: ignore[attr-defined]
        # ...and every other name the r6/r5 routes needed is gone too:
        for name in (
            "_SealVault",
            "_SealMachinery",
            "_SealRO",
            "_make_seal_machinery",
            "_install_seal_machinery",
            "_is_unsealed",
            "_seal_into",
            "_fresh_unsealed",
            "_state_seal",
            "_seal_deleted_problem",
            "write_once",
        ):
            assert not hasattr(mp, name), name
            with pytest.raises(AttributeError):
                getattr(mp, name)
        # the deleted names are not in __all__ either:
        assert "_SEAL_MACHINERY" not in mp.__all__

    @pytest.mark.parametrize(
        ("component", "field", "payload_fields", "label"),
        [
            ("first", "structure_ref", _MEMBER_PAYLOAD_FIELDS,
             "structure member binding"),
            ("source", "volume_ref", _SOURCE_PAYLOAD_FIELDS,
             "source volume binding"),
            ("comparator", "algorithm_id", _COMPARATOR_PAYLOAD_FIELDS,
             "comparator binding"),
        ],
    )
    def test_negative_r6_delete_both_construction_is_absent(
        self, component, field, payload_fields, label
    ):
        # The r6 construction, step by step, on all three components:
        #   mach = mp._SEAL_MACHINERY[cls]        -> AttributeError (absent)
        #   ever/vault = <closure cells>          -> cannot be reached
        #   del ever[oid]; del vault._table[oid]  -> nothing to delete
        #   mach.write_once(obj, derived)         -> does not exist
        import models.registry.model_prep as mp

        d = declaration()
        obj = getattr(d, component)
        before = {name: getattr(obj, name) for name in payload_fields}
        with pytest.raises(AttributeError, match="_SEAL_MACHINERY"):
            mp._SEAL_MACHINERY[type(obj)]  # type: ignore[attr-defined]
        # the ONE-FIELD patch the route needs is refused at write time:
        with pytest.raises(CanonicalValidationError, match="construction-only"):
            object.__setattr__(obj, field, "artifact:EVIL")
        # so the payload never moves, the pair stays valid, and the digest
        # is unchanged (there is nothing to re-bless):
        assert {name: getattr(obj, name) for name in payload_fields} == before
        assert pair_binding_problems(d) == []
        original_digest = d.declaration_sha256
        assert d.declaration_sha256 == original_digest
        # the seal is COMPUTED, so even a caller who derives it gets no
        # installable object:
        derived = _derive_binding_seal(
            label, payload_fields, {n: getattr(obj, n) for n in payload_fields}
        )
        assert obj.binding_seal == derived
        assert not hasattr(mp, "write_once")

    def test_negative_r6_no_module_or_class_attribute_holds_seal_state(self):
        # No module attribute, class attribute, or instance dict carries ANY
        # seal state: no vault, no table, no id-keyed mapping, no weakref.
        import models.registry.model_prep as mp

        for name in dir(mp):
            value = getattr(mp, name)
            assert type(value).__name__ not in (
                "_SealVault",
                "_SealMachinery",
                "_SealRO",
            ), name
        for cls in (StructureMemberBinding, SourceVolumeBinding, ComparatorBinding):
            for name in dir(cls):
                value = getattr(cls, name, None)
                assert type(value).__name__ not in (
                    "_SealVault",
                    "_SealMachinery",
                    "_SealRO",
                ), (cls.__name__, name)
            # the payload lock holds NO descriptor and NO state but the name:
            for name in cls.__slots__:
                guard = cls.__dict__[name]
                assert type(guard).__name__ == "_PayloadLock"
                assert not hasattr(guard, "__dict__")
                assert guard._name == name
        # and no instance carries a __dict__ at all (slotted, frozen):
        m = member()
        assert not hasattr(m, "__dict__")

    # -- (b) the plain patch route is REFUSED (negatives b) ------------------

    def test_negative_plain_object_setattr_is_refused_on_every_payload_field(self):
        # NEGATIVE (b): plain `object.__setattr__(member, 'structure_ref',
        # 'EVIL')` is REFUSED by the frame gate with the §10.2 message —
        # for EVERY payload field of EVERY component — and so is delattr.
        d = declaration()
        for obj, fields in (
            (d.first, _MEMBER_PAYLOAD_FIELDS),
            (d.second, _MEMBER_PAYLOAD_FIELDS),
            (d.source, _SOURCE_PAYLOAD_FIELDS),
            (d.comparator, _COMPARATOR_PAYLOAD_FIELDS),
        ):
            for name in fields:
                before = getattr(obj, name)
                with pytest.raises(
                    CanonicalValidationError, match="never patch one field"
                ):
                    object.__setattr__(obj, name, "EVIL")
                with pytest.raises(
                    CanonicalValidationError, match="cannot be deleted"
                ):
                    object.__delattr__(obj, name)
                # plain spellings are refused by the frozen dataclass / the
                # property-less name (FrozenInstanceError is an AttributeError):
                with pytest.raises(AttributeError):
                    setattr(obj, name, "EVIL")
                with pytest.raises(AttributeError):
                    delattr(obj, name)
                assert getattr(obj, name) == before
        assert pair_binding_problems(d) == []

    def test_negative_the_frame_gate_refuses_every_foreign_frame(self):
        # The gate compares the CALLER's code object against the module's own
        # __init__/__setstate__ code objects, so a wrapper, an alias, or a
        # helper defined anywhere else is refused:
        import models.registry.model_prep as mp

        m = member()

        def helper_write():
            object.__setattr__(m, "structure_ref", "artifact:EVIL")

        with pytest.raises(CanonicalValidationError, match="construction-only"):
            helper_write()
        with pytest.raises(CanonicalValidationError, match="construction-only"):
            object.__setattr__(m, _MEMBER_PAYLOAD_FIELDS[0], "EVIL")
        # even a direct call to the descriptor's __set__ from a foreign frame
        # (the shape the r5 route used) is refused:
        guard = type(m).__dict__["structure_ref"]
        with pytest.raises(CanonicalValidationError, match="construction-only"):
            guard.__set__(m, "artifact:EVIL")
        # and the authorized set is exactly the module's own frames:
        assert mp._AUTHORIZED_PAYLOAD_FRAMES == frozenset(
            {
                StructureMemberBinding.__init__.__code__,
                StructureMemberBinding.__setstate__.__code__,
                SourceVolumeBinding.__init__.__code__,
                SourceVolumeBinding.__setstate__.__code__,
                ComparatorBinding.__init__.__code__,
                ComparatorBinding.__setstate__.__code__,
                PairBindingDeclaration.__init__.__code__,
                PairBindingDeclaration.__setstate__.__code__,
            }
        )
        assert m.structure_ref == "artifact:structure-a"

    # -- (c) refused __setstate__ is ATOMIC (R6-F2) --------------------------

    @pytest.mark.parametrize(
        ("factory", "payload_fields", "bad_state"),
        [
            ("member", _MEMBER_PAYLOAD_FIELDS, "shape"),
            ("source", _SOURCE_PAYLOAD_FIELDS, "shape"),
            ("comparator", _COMPARATOR_PAYLOAD_FIELDS, "shape"),
        ],
    )
    def test_negative_refused_setstate_leaves_the_target_byte_identical(
        self, factory, payload_fields, bad_state
    ):
        # NEGATIVE (c) / R6-F2: a refused __setstate__ on a POPULATED target
        # must leave the payload byte-identical before/after. (The r6 record
        # showed the att0009 refusal still mutated the payload slots first.)
        cls = {
            "member": StructureMemberBinding,
            "source": SourceVolumeBinding,
            "comparator": ComparatorBinding,
        }[factory]
        value = {"member": member, "source": source, "comparator": comparator}[
            factory
        ]()
        target: Any = object.__new__(cls)
        target.__setstate__(value.__getstate__())  # populate legitimately
        before = {name: getattr(target, name) for name in payload_fields}
        before_seal = target.binding_seal
        # every refusal shape, each leaving the target untouched:
        refusals = (
            # bare payload (not deserialization data):
            tuple(getattr(value, name) for name in payload_fields),
            # payload + a wrong seal:
            tuple(getattr(value, name) for name in payload_fields) + ("f" * 64,),
            # payload + a sentinel:
            tuple(getattr(value, name) for name in payload_fields) + ("",),
            # a valid state, but onto a populated target:
            value.__getstate__(),
            # wrong shape entirely:
            ("nope",),
            # a mapping missing the seal:
            {name: getattr(value, name) for name in payload_fields},
        )
        for state in refusals:
            with pytest.raises(CanonicalValidationError):
                target.__setstate__(state)
            assert {name: getattr(target, name) for name in payload_fields} == before
            assert target.binding_seal == before_seal

    def test_negative_a_populated_target_is_refused_with_the_populated_reason(self):
        # R6-F2: the refusal names the populated-target reason, and the
        # check runs BEFORE any slot is written.
        m = member()
        with pytest.raises(CanonicalValidationError, match="already populated"):
            m.__setstate__(member("artifact:structure-z").__getstate__())
        assert m.structure_ref == "artifact:structure-a"

    # -- (d) r4/r5/r2 constructions stay refused -----------------------------

    def test_control_r6_all_r5_r4_and_r2_constructions_stay_refused(self):
        # The r5 EXACT route (type(obj)._seal_slot_guard._member.__set__) is
        # dead on all three components: the guard object does not exist.
        for cls in (StructureMemberBinding, SourceVolumeBinding, ComparatorBinding):
            assert not hasattr(cls, "_seal_slot_guard"), cls.__name__
        d = declaration()
        for obj, field in (
            (d.first, "structure_ref"),
            (d.source, "volume_ref"),
            (d.comparator, "algorithm_id"),
        ):
            with pytest.raises(AttributeError, match="_seal_slot_guard"):
                type(obj)._seal_slot_guard._member.__set__(  # type: ignore[union-attr]
                    obj, "f" * 64
                )
            with pytest.raises(AttributeError, match="_seal_slot_guard"):
                type(obj)._seal_slot_guard._member.__delete__(obj)  # type: ignore[union-attr]
            with pytest.raises(CanonicalValidationError, match="construction-only"):
                object.__setattr__(obj, field, "artifact:replacement")
        # r4 route 1/2/3 shapes and the r2 constructions:
        m = member()
        bare = tuple(getattr(m, name) for name in _MEMBER_PAYLOAD_FIELDS)
        with pytest.raises(CanonicalValidationError, match="CONSTRUCTION-SHAPED"):
            m.__setstate__(bare)
        with pytest.raises(AttributeError):
            object.__setattr__(m, "_binding_seal", "f" * 64)
        with pytest.raises((TypeError, ValueError)):
            replace(m, structure_ref="artifact:replacement")
        # the pair keeps validating throughout:
        assert pair_binding_problems(d) == []
        assert d.declaration_sha256

    # -- controls ------------------------------------------------------------

    def test_control_r6_transports_and_fresh_construction_live(self):
        d = declaration()
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            rd = pickle.loads(pickle.dumps(d, protocol=proto))
            assert rd == d and rd.declaration_sha256 == d.declaration_sha256
        assert copy.copy(d) == d and copy.deepcopy(d) == d
        for component in (d.first, d.second, d.source, d.comparator):
            assert pickle.loads(pickle.dumps(component)) == component
            assert copy.deepcopy(component) == component
        # fresh construction and the sanctioned complete swap still work:
        fresh = member("artifact:structure-c")
        swapped = replace(d, first=fresh)
        assert swapped.declaration_sha256 != d.declaration_sha256
        assert pair_binding_problems(swapped) == []
        assert d.first.structure_ref == "artifact:structure-a"
        # factory/direct equality and repeat determinism:
        assert create_pair_binding_declaration(
            first=d.first, second=d.second, source=d.source, comparator=d.comparator
        ) == d
        assert d.declaration_sha256 == d.declaration_sha256

    def test_control_r6_digest_continuity_is_preserved_byte_for_byte(self):
        # Digest continuity: identical payloads -> byte-identical declaration
        # digests. The pinned r5/r6 baseline (the exact reviewer fixture) and
        # the r5 record's three component digests must all reproduce.
        def r6_member(ref):
            return StructureMemberBinding(
                structure_ref=ref,
                structure_sha256="1" * 64,
                model_ref="model@1",
                release_sha256="2" * 64,
                provenance_ref="provenance:a",
                provenance_sha256="3" * 64,
            )

        def r6_pair():
            return PairBindingDeclaration(
                first=r6_member("artifact:a"),
                second=r6_member("artifact:b"),
                source=SourceVolumeBinding(
                    volume_ref="volume:a",
                    frame_spec_sha256="4" * 64,
                    source_manifest_sha256="5" * 64,
                ),
                comparator=ComparatorBinding(
                    algorithm_id="compare",
                    algorithm_version="1",
                    parameters_sha256="6" * 64,
                ),
            )

        assert r6_pair().declaration_sha256 == (
            "6e69b5f43e7876b271c401e72b976392b68c3b52e22539added6020f5bb517ad"
        )
        # the three component digests (one-field-patched values, i.e. the
        # identities the r5 route produced — now reachable ONLY by a complete
        # fresh construction with those values):
        for attr, field, expected in (
            ("first", "structure_ref",
             "69c5ea72187d7fd691c9b2eeec4ef8430024f6b0108f3d633cf6b6ebd6f18409"),
            ("source", "volume_ref",
             "baf3b93b34e20c639ab64fbd1fae962d3197bb0922fc6ca267de3a254d148abe"),
            ("comparator", "algorithm_id",
             "5404c8fc3032ccb672bc5246c229e98507399a08fc62721db748ad4843c8574f"),
        ):
            p = r6_pair()
            obj = getattr(p, attr)
            values = {
                name: getattr(obj, name)
                for name in (type(obj).__slots__ if attr != "first" else _MEMBER_PAYLOAD_FIELDS)
            }
            values[field] = "artifact:replacement"
            # a complete fresh construction with the patched value:
            cls = type(obj)
            moved = cls(**values)
            swapped = replace(p, **{attr: moved})
            assert swapped.declaration_sha256 == expected, (attr, field)
        # and identical payloads stay byte-identical across transports:
        p2 = r6_pair()
        assert pickle.loads(pickle.dumps(p2)).declaration_sha256 == (
            p2.declaration_sha256
        )

    # -- mutations: each guard is load-bearing -------------------------------

    def test_mutation_r6_the_frame_gate_is_load_bearing(self, monkeypatch):
        # MUTATION (i): the frame gate removed from the payload descriptor ->
        # the plain patch route RE-OPENS (assert); restored -> refused.
        # R7-F2 layering: the write path now carries TWO further independent
        # gates (the r7 relay-frame gate in ``_write_slot`` and the shared
        # empty-target predicate), so the defect class lands only when ALL
        # of them are neutered together.
        import models.registry.model_prep as mp

        m = member()
        monkeypatch.setattr(
            mp._PayloadLock, "_codes", frozenset({_current_frame_code()})
        )
        with pytest.raises(CanonicalValidationError):
            object.__setattr__(m, "structure_ref", "artifact:EVIL")
        assert m.structure_ref == "artifact:structure-a"  # r7 gates held
        monkeypatch.setattr(
            mp, "_relay_frame_authorized", lambda caller: True
        )
        monkeypatch.setattr(
            mp, "_empty_target_problems", lambda instance, names: []
        )
        try:
            object.__setattr__(m, "structure_ref", "artifact:EVIL")
            assert m.structure_ref == "artifact:EVIL"  # the route re-opened
        finally:
            monkeypatch.undo()
        # restored -> the SAME plain patch is refused:
        m2 = member()
        with pytest.raises(CanonicalValidationError, match="construction-only"):
            object.__setattr__(m2, "structure_ref", "artifact:EVIL")
        assert m2.structure_ref == "artifact:structure-a"

    def test_mutation_r6_the_buffer_then_commit_is_load_bearing(self, monkeypatch):
        # MUTATION (ii): __setstate__ made to assign BEFORE validating -> the
        # non-atomicity RE-APPEARS (a refused restore leaves a half-applied
        # payload); restored -> the refusal is atomic.
        import models.registry.model_prep as mp

        def assigning_setstate(self, state):
            # the att0009 shape: commit first, validate after.
            if isinstance(state, (list, tuple)) and len(state) == len(
                _MEMBER_PAYLOAD_FIELDS
            ) + 1:
                for name, value in zip(_MEMBER_PAYLOAD_FIELDS, state):
                    _write_payload_raw(self, name, value)
            mp._construction_shaped_state(
                type(self).__name__,
                state,
                _MEMBER_PAYLOAD_FIELDS,
                "structure member binding",
                _member_field_problems,
            )

        # a target carrying a VALID payload, then a refused state:
        target = StructureMemberBinding.__new__(StructureMemberBinding)
        target.__setstate__(member("artifact:good").__getstate__())
        before = {n: getattr(target, n) for n in _MEMBER_PAYLOAD_FIELDS}
        bad = tuple(getattr(member(), n) for n in _MEMBER_PAYLOAD_FIELDS) + (
            "f" * 64,
        )
        monkeypatch.setattr(
            StructureMemberBinding, "__setstate__", assigning_setstate
        )
        try:
            with pytest.raises(CanonicalValidationError):
                target.__setstate__(bad)
            # the defect class: the payload was HALF-APPLIED by the refusal
            assert {n: getattr(target, n) for n in _MEMBER_PAYLOAD_FIELDS} != before
        finally:
            monkeypatch.undo()
        # restored -> the refusal is ATOMIC (buffer-then-commit):
        target2 = StructureMemberBinding.__new__(StructureMemberBinding)
        target2.__setstate__(member("artifact:good").__getstate__())
        before2 = {n: getattr(target2, n) for n in _MEMBER_PAYLOAD_FIELDS}
        with pytest.raises(CanonicalValidationError):
            target2.__setstate__(bad)
        assert {n: getattr(target2, n) for n in _MEMBER_PAYLOAD_FIELDS} == before2

    def test_the_r6_guards_cannot_go_quiet(self):
        # The R6 guard machinery is anchored in the module source: if the
        # payload lock, the frame registry, the buffer-then-commit
        # validators, or the empty-target check were deleted, these markers
        # disappear and the mutation probes above lose their patch targets.
        source = Path(model_prep.__file__).read_text(encoding="utf-8")
        for anchor in (
            "class _PayloadLock",
            "_install_payload_locks",
            "_AUTHORIZED_PAYLOAD_FRAMES",
            "_SLOT_READERS",
            "_empty_target_problems",
            "_commit_state",
            "_construction_shaped_state",
            "component_binding_seal",
            "_PAYLOAD_WRITE_PROBLEM",
            "_PAYLOAD_DELETE_PROBLEM",
            "_POPULATED_TARGET_PROBLEM",
            # R7-F1/R7-F2: the complete-fresh-only construction gate and the
            # helper gates must ship with the module (they cannot go quiet):
            "_CONSTRUCTION_REENTRY_PROBLEM",
            "_construction_reentry_problems",
            "_relay_frame_authorized",
            "_construction_frame_authorized",
            "_payload_field_names",
        ):
            assert anchor in source, anchor
        # and there is still no seal storage of any kind in the CODE:
        code = source.split('"""', 2)[-1]
        for deleted in (
            "class _SealVault",
            "class _SealMachinery",
            "_SEAL_MACHINERY",
            "_seal_into",
            "write_once",
            "_is_unsealed",
        ):
            assert deleted not in code, f"the deleted name {deleted!r} is BACK"


class TestR7ConstructionFreshOnly:
    """R7-F1/R7-F2 closure: construction is COMPLETE-FRESH ONLY (§10.2).

    R7-F1: every component ``__init__`` (and the pair's) FIRST checks the
    instance is EMPTY via the SHARED empty-target predicate — bound and
    unbound ``__init__`` re-entry on an already-populated binding is refused
    with the §10.2 complete-fresh-only reason, atomically (the refusal runs
    before any slot is written, so the payload stays byte-identical).

    R7-F2: the module's own write helpers (``mp._write_slot`` /
    ``mp._commit_state``) are no longer plain single-call patch routes: the
    same shared empty-target predicate refuses populated targets, and a
    frame gate refuses every caller outside the module's two sanctioned
    write relays (:func:`_commit_state`, :class:`_PayloadLock.__set__`).
    """

    # -- R7-F1 negatives ----------------------------------------------------

    def test_negative_r7_f1_bound_and_unbound_init_reentry_refused(self):
        # the reviewer's EXACT r7 construction: o.__init__(**patched_vals)
        # (bound) and StructureMemberBinding.__init__(o, **vals) (unbound)
        # on an ALREADY-POPULATED member — both refused with the §10.2
        # message, atomically (payload byte-identical after the refusal).
        d = declaration()
        o = d.first
        before = {n: getattr(o, n) for n in _MEMBER_PAYLOAD_FIELDS}
        digest_before = d.declaration_sha256
        vals = {n: getattr(o, n) for n in _MEMBER_PAYLOAD_FIELDS}
        vals["structure_ref"] = "artifact:replacement"
        with pytest.raises(
            CanonicalValidationError, match="construction is complete-fresh only"
        ):
            o.__init__(**vals)  # type: ignore[misc]
        with pytest.raises(
            CanonicalValidationError, match="re-constructed in place"
        ):
            StructureMemberBinding.__init__(o, **vals)
        assert {n: getattr(o, n) for n in _MEMBER_PAYLOAD_FIELDS} == before
        assert d.declaration_sha256 == digest_before
        assert pair_binding_problems(d) == []

    def test_negative_r7_f1_source_and_comparator_init_reentry_refused(self):
        # the same empty-target gate guards EVERY sealed component __init__.
        d = declaration()
        s_vals = {n: getattr(d.source, n) for n in _SOURCE_PAYLOAD_FIELDS}
        s_vals["volume_ref"] = "artifact:replacement"
        with pytest.raises(CanonicalValidationError, match="complete-fresh only"):
            SourceVolumeBinding.__init__(d.source, **s_vals)
        c_vals = {
            n: getattr(d.comparator, n) for n in _COMPARATOR_PAYLOAD_FIELDS
        }
        c_vals["algorithm_id"] = "artifact:replacement"
        with pytest.raises(CanonicalValidationError, match="complete-fresh only"):
            ComparatorBinding.__init__(d.comparator, **c_vals)
        assert pair_binding_problems(d) == []

    def test_negative_r7_f1_pair_init_reentry_refused(self):
        # the pair's __init__ refuses re-entry on an existing declaration
        # too — the swapped-in complete components can never be written in
        # place; a swap is a COMPLETE FRESH declaration (§10.2).
        d = declaration()
        fresh = member("artifact:structure-c")
        vals = {
            "first": fresh,
            "second": d.second,
            "source": d.source,
            "comparator": d.comparator,
            "schema_version": d.schema_version,
        }
        digest_before = d.declaration_sha256
        with pytest.raises(
            CanonicalValidationError, match="construction is complete-fresh only"
        ):
            d.__init__(**vals)  # type: ignore[misc]
        with pytest.raises(CanonicalValidationError, match="complete-fresh only"):
            PairBindingDeclaration.__init__(d, **vals)
        assert d.first is not fresh
        assert d.declaration_sha256 == digest_before
        assert pair_binding_problems(d) == []

    # -- R7-F2 negatives ----------------------------------------------------

    def test_negative_r7_f2_write_slot_and_commit_state_refuse_populated(self):
        # (b) the names KEPT but gated: a single plain call on an existing
        # bound member is refused by the shared empty-target predicate —
        # the same outcome R7-F2 recorded, now closed. Assert which fix was
        # chosen: the names EXIST and REFUSE.
        d = declaration()
        o = d.first
        before = {n: getattr(o, n) for n in _MEMBER_PAYLOAD_FIELDS}
        digest_before = d.declaration_sha256
        with pytest.raises(CanonicalValidationError, match="complete-fresh only"):
            model_prep._write_slot(o, "structure_ref", "artifact:replacement")
        with pytest.raises(CanonicalValidationError, match="complete-fresh only"):
            model_prep._commit_state(
                o, {n: getattr(o, n) for n in _MEMBER_PAYLOAD_FIELDS}
            )
        assert {n: getattr(o, n) for n in _MEMBER_PAYLOAD_FIELDS} == before
        assert d.declaration_sha256 == digest_before
        assert pair_binding_problems(d) == []

    def test_negative_r7_f2_helpers_refuse_fresh_targets_outside_construction(
        self, monkeypatch
    ):
        # the FRAME gate: even on an EMPTY (freshly allocated) target, the
        # helpers refuse every caller outside the module's own sanctioned
        # write relays — no unvalidated value can be written by a plain call.
        forged = StructureMemberBinding.__new__(StructureMemberBinding)
        with pytest.raises(CanonicalValidationError, match="construction-only"):
            model_prep._write_slot(forged, "structure_ref", "artifact:EVIL")
        with pytest.raises(CanonicalValidationError, match="construction-only"):
            model_prep._commit_state(
                forged,
                {name: "artifact:EVIL" for name in _MEMBER_PAYLOAD_FIELDS},
            )
        assert not any(
            _slot_is_populated(forged, name) for name in _MEMBER_PAYLOAD_FIELDS
        )

    # -- r7 controls (the sanctioned flows stay alive) -----------------------

    def test_control_r7_fresh_construction_transports_and_digest_continuity(self):
        # fresh construction, construction-shaped __setstate__ onto a fresh
        # target, the sanctioned pair swap, transports, and the pinned digest
        # continuity baseline all still work (nothing else regressed).
        d = declaration()
        assert pair_binding_problems(d) == []
        fresh = StructureMemberBinding.__new__(StructureMemberBinding)
        fresh.__setstate__(member("artifact:structure-z").__getstate__())
        assert fresh.structure_ref == "artifact:structure-z"
        assert _member_use_problems(fresh) == []
        swapped = replace(d, first=member("artifact:structure-c"))
        assert swapped.declaration_sha256 != d.declaration_sha256
        assert pair_binding_problems(swapped) == []
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            restored = pickle.loads(pickle.dumps(d, protocol=protocol))
            assert restored == d
        assert copy.deepcopy(d) == d
        assert copy.copy(d) == d
        # the r6 pinned baseline is byte-identical (digest continuity):
        def r6_member(ref):
            return StructureMemberBinding(
                structure_ref=ref,
                structure_sha256="1" * 64,
                model_ref="model@1",
                release_sha256="2" * 64,
                provenance_ref="provenance:a",
                provenance_sha256="3" * 64,
            )

        def r6_pair():
            return PairBindingDeclaration(
                first=r6_member("artifact:a"),
                second=r6_member("artifact:b"),
                source=SourceVolumeBinding(
                    volume_ref="volume:a",
                    frame_spec_sha256="4" * 64,
                    source_manifest_sha256="5" * 64,
                ),
                comparator=ComparatorBinding(
                    algorithm_id="compare",
                    algorithm_version="1",
                    parameters_sha256="6" * 64,
                ),
            )

        assert r6_pair().declaration_sha256 == (
            "6e69b5f43e7876b271c401e72b976392b68c3b52e22539added6020f5bb517ad"
        )

    # -- r7 mutations: every new gate is load-bearing ------------------------

    def test_mutation_r7_f1_the_shared_empty_target_check_is_load_bearing(
        self, monkeypatch
    ):
        # MUTATION (R7-F1): the shared empty-target predicate removed ->
        # constructor re-entry RE-ADMITS (old evidence digests retained, the
        # declaration digest moves, pickle survives); restored -> refused.
        import models.registry.model_prep as mp_module

        d = declaration()
        o = d.first
        original_digest = d.declaration_sha256
        vals = {n: getattr(o, n) for n in _MEMBER_PAYLOAD_FIELDS}
        vals["structure_ref"] = "artifact:replacement"

        monkeypatch.setattr(
            mp_module, "_empty_target_problems", lambda instance, names: []
        )
        try:
            o.__init__(**vals)  # type: ignore[misc]
            assert o.structure_ref == "artifact:replacement"  # defect class
            assert pair_binding_problems(d) == []
            assert d.declaration_sha256 != original_digest
            # the PAIR re-entry re-admits through the SAME removed check:
            d2 = declaration()
            pair_vals = {
                "first": member("artifact:structure-c"),
                "second": d2.second,
                "source": d2.source,
                "comparator": d2.comparator,
                "schema_version": d2.schema_version,
            }
            d2.__init__(**pair_vals)  # type: ignore[misc]
            assert d2.first.structure_ref == "artifact:structure-c"
        finally:
            monkeypatch.undo()
        # restored -> the SAME routes are refused again, atomically:
        d3 = declaration()
        before3 = {n: getattr(o, n) for n in _MEMBER_PAYLOAD_FIELDS}
        with pytest.raises(CanonicalValidationError, match="complete-fresh only"):
            o.__init__(**vals)  # type: ignore[misc]
        with pytest.raises(CanonicalValidationError, match="complete-fresh only"):
            d3.__init__(**pair_vals)  # type: ignore[misc]
        assert {n: getattr(o, n) for n in _MEMBER_PAYLOAD_FIELDS} == before3
        assert pair_binding_problems(d3) == []

    def test_mutation_r7_f2_the_helper_gates_are_load_bearing(self, monkeypatch):
        # MUTATION (R7-F2): the helper guards removed (the r7 relay frame
        # gate AND the construction-frame gate AND the shared emptiness
        # predicate) -> mp._write_slot / mp._commit_state re-OPEN as plain
        # single-call patch routes; restored -> refused.
        import models.registry.model_prep as mp_module

        d = declaration()
        o = d.first
        original_digest = d.declaration_sha256
        values = {n: getattr(o, n) for n in _MEMBER_PAYLOAD_FIELDS}
        values["structure_ref"] = "artifact:replacement"

        monkeypatch.setattr(
            mp_module, "_relay_frame_authorized", lambda caller: True
        )
        monkeypatch.setattr(
            mp_module, "_construction_frame_authorized", lambda caller: True
        )
        monkeypatch.setattr(
            mp_module, "_empty_target_problems", lambda instance, names: []
        )
        try:
            mp_module._write_slot(o, "structure_ref", "artifact:replacement")
            assert o.structure_ref == "artifact:replacement"
            mp_module._commit_state(o, values)  # plain call patches again
            assert o.structure_ref == "artifact:replacement"
            assert d.declaration_sha256 != original_digest
        finally:
            monkeypatch.undo()
        d2 = declaration()
        o2 = d2.first
        before2 = {n: getattr(o2, n) for n in _MEMBER_PAYLOAD_FIELDS}
        with pytest.raises(CanonicalValidationError, match="complete-fresh only"):
            mp_module._write_slot(o2, "structure_ref", "artifact:replacement")
        with pytest.raises(CanonicalValidationError, match="complete-fresh only"):
            mp_module._commit_state(o2, values)
        assert {n: getattr(o2, n) for n in _MEMBER_PAYLOAD_FIELDS} == before2

    def test_mutation_r7_the_layered_gates_hold_singly(self, monkeypatch):
        # the r7 tier standard: NO SINGLE neutered gate re-opens a one-call
        # patch route — the emptiness predicate and the frame gates are
        # INDEPENDENT layers; each alone keeps every route closed.
        import models.registry.model_prep as mp_module

        d = declaration()
        o = d.first
        before = {n: getattr(o, n) for n in _MEMBER_PAYLOAD_FIELDS}
        values = {n: getattr(o, n) for n in _MEMBER_PAYLOAD_FIELDS}
        values["structure_ref"] = "artifact:replacement"

        # (1) only the relay frame gate removed -> the emptiness gate holds:
        monkeypatch.setattr(
            mp_module, "_relay_frame_authorized", lambda caller: True
        )
        with pytest.raises(CanonicalValidationError, match="complete-fresh only"):
            mp_module._write_slot(o, "structure_ref", "artifact:replacement")
        monkeypatch.undo()
        # (2) only the emptiness predicate removed -> the frame gates hold:
        monkeypatch.setattr(
            mp_module, "_empty_target_problems", lambda instance, names: []
        )
        with pytest.raises(CanonicalValidationError, match="construction-only"):
            mp_module._write_slot(o, "structure_ref", "artifact:replacement")
        with pytest.raises(CanonicalValidationError, match="construction-only"):
            mp_module._commit_state(o, values)
        monkeypatch.undo()
        # (3) only the commit-frame gate removed -> the emptiness gate holds:
        monkeypatch.setattr(
            mp_module, "_construction_frame_authorized", lambda caller: True
        )
        with pytest.raises(CanonicalValidationError, match="complete-fresh only"):
            mp_module._commit_state(o, values)
        monkeypatch.undo()
        assert {n: getattr(o, n) for n in _MEMBER_PAYLOAD_FIELDS} == before
        assert pair_binding_problems(d) == []
