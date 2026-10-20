# ruff: noqa: E501
"""WS01 audit workflow — closed schema, scoring, ledger, seeds, digest.

Covers the five deliverables from the WS01 contract; every public
behaviour has pytest coverage so the high-risk implementation lane
fails closed on regressions.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
from collections.abc import Mapping
from typing import Any, cast

import pytest

from core.errors import (
    CanonicalValidationError,
    ClosedSchemaError,
    DigestMismatchError,
    StateTransitionError,
)
from core.events import EventLog, StateEvent, utc_now_iso
from models.audit import (
    AuditLedger,
    AuditStatus,
    DimensionScore,
    ProvenanceRef,
    ReuseDecision,
    TechnologyAuditRecord,
    audit_score,
    build_seed_ledger,
    compute_audit_digest,
    rank_audits,
    record_from_dict,
    record_to_dict,
)
from models.audit.record import (
    DIMENSION_NAMES,
    LEGAL_AUDIT_TRANSITIONS,
    validate_audit_record,
)
from models.audit.scoring import WEIGHTS_VERSION

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _ds(score: int, text: str) -> DimensionScore:
    return DimensionScore(score, text)


def _make_record(**overrides) -> TechnologyAuditRecord:
    base = {
        "candidate_name": "test-lib",
        "source_url": "https://github.com/example/test-lib",
        "version": "1.0.0",
        "capability_area": "test-area",
        "clinical_technical_evidence": _ds(4, "clinical evidence one-liner"),
        "license_redistribution": _ds(4, "license good one-liner"),
        "reproducibility_auditability": _ds(4, "repro good one-liner"),
        "security_dependency_health": _ds(4, "security good one-liner"),
        "local_offline_operability": _ds(5, "offline good one-liner"),
        "maintainability": _ds(4, "maintain good one-liner"),
        "integration_cost": _ds(3, "integration medium one-liner"),
        "reuse_decision": ReuseDecision.USE_DEPENDENCY,
        "reuse_justification": "direct dependency, lowest cost one-liner",
        "license_id": "MIT",
        "permitted_use": "permitted for personal-research with MIT attribution",
        "distribution_risk": "minimal risk, MIT one-liner",
        "provenance_refs": (ProvenanceRef(path="orchestration/logs/web_sol_ws01_tech_audit.md.last.md", sha256="a" * 64),),
        "evidence_provenance": "web_sol",
    }
    base.update(overrides)
    return TechnologyAuditRecord(**base)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 1 — closed dataclass schema + seven dimensions
# ---------------------------------------------------------------------------


class TestClosedSchema:
    def test_unknown_top_level_rejected(self):
        payload = record_to_dict(_make_record())
        payload["prompt_injection"] = "yes"
        with pytest.raises(ClosedSchemaError):
            record_from_dict(payload)

    def test_unknown_nested_provenance_rejected(self):
        payload = record_to_dict(_make_record())
        payload["provenance_refs"][0]["extra"] = 1
        with pytest.raises(ClosedSchemaError):
            record_from_dict(payload)

    def test_round_trip_preserves_all_fields(self):
        rec = _make_record()
        restored = record_from_dict(record_to_dict(rec))
        assert restored == rec
        # JSON dumps never sees a raw Enum
        dumped = json.dumps(record_to_dict(rec))
        assert json.loads(dumped) == record_to_dict(rec)

    def test_missing_required_rejected(self):
        payload = record_to_dict(_make_record())
        del payload["candidate_name"]
        with pytest.raises(CanonicalValidationError):
            record_from_dict(payload)

    def test_all_seven_dimensions_present(self):
        rec = _make_record()
        for name in DIMENSION_NAMES:
            val = getattr(rec, name)
            assert isinstance(val, DimensionScore)
            assert 0 <= int(val) <= 5
            assert isinstance(val.justification, str) and len(val.justification) >= 5

    def test_flat_justification_alias_synthesized(self):
        rec = _make_record()
        # __getattr__ synthesizes <dim>_justification
        assert rec.clinical_technical_evidence_justification == rec.clinical_technical_evidence.justification
        assert rec.license_redistribution_justification == rec.license_redistribution.justification

    def test_dimension_scores_dict_helper(self):
        rec = _make_record()
        assert isinstance(rec.dimension_scores, dict)
        assert set(rec.dimension_scores) == set(DIMENSION_NAMES)


class TestDimensionValidation:
    def test_score_out_of_range_rejected(self):
        with pytest.raises((ValueError, CanonicalValidationError)):
            rec = _make_record(clinical_technical_evidence=_ds(10, "too high one-liner"))
            # DimensionScore itself raises ValueError before validate
            validate_audit_record(rec)

    def test_justification_empty_rejected(self):
        with pytest.raises((ValueError, CanonicalValidationError)):
            bad = DimensionScore(3, "")
            _make_record(clinical_technical_evidence=bad)

    def test_justification_newline_rejected(self):
        with pytest.raises(ValueError):
            DimensionScore(3, "bad\nnewline")

    def test_flat_alias_int_plus_justification_accepted(self):
        payload = record_to_dict(_make_record())
        # replace one dimension with flat int+alias form
        payload["clinical_technical_evidence"] = 5
        payload["clinical_technical_evidence_justification"] = "flat alias one-liner is accepted"
        rec = record_from_dict(payload)
        assert int(rec.clinical_technical_evidence) == 5
        assert rec.clinical_technical_evidence.justification == "flat alias one-liner is accepted"

    def test_score_via_int_subclass_behaviour(self):
        ds = _ds(5, "int subclass justification one-liner")
        assert isinstance(ds, int)
        assert ds == 5
        assert int(ds) == 5
        assert ds.score == 5


class TestReuseDecision:
    def test_all_six_values_present(self):
        vals = {d.value for d in ReuseDecision}
        assert vals == {"USE_DEPENDENCY", "BUILD_ADAPTER", "RUN_EXTERNAL_SERVICE", "VENDOR_COMPONENT", "FORK", "REWRITE"}

    def test_ordered_preference_deviation_needs_justification(self):
        # BUILD_ADAPTER with too-short justification should be rejected
        with pytest.raises(CanonicalValidationError):
            rec = _make_record(reuse_decision=ReuseDecision.BUILD_ADAPTER, reuse_justification="short")
            validate_audit_record(rec)

    def test_use_dependency_with_good_justification_passes(self):
        rec = _make_record(reuse_decision=ReuseDecision.USE_DEPENDENCY, reuse_justification="direct pip dep, lowest cost one-liner")
        validate_audit_record(rec)


class TestSbomAndProvenance:
    def test_license_id_required(self):
        with pytest.raises(CanonicalValidationError):
            validate_audit_record(_make_record(license_id=""))

    def test_permitted_use_required(self):
        with pytest.raises(CanonicalValidationError):
            validate_audit_record(_make_record(permitted_use=""))

    def test_distribution_risk_required(self):
        with pytest.raises(CanonicalValidationError):
            validate_audit_record(_make_record(distribution_risk=""))

    def test_provenance_refs_non_empty(self):
        with pytest.raises(CanonicalValidationError):
            validate_audit_record(_make_record(provenance_refs=()))

    def test_provenance_sha_must_be_hex(self):
        with pytest.raises(CanonicalValidationError):
            bad = ProvenanceRef(path="a/b", sha256="nothex")
            validate_audit_record(_make_record(provenance_refs=(bad,)))


# ---------------------------------------------------------------------------
# 2 — scoring helper: risk-adjusted ranking, weights-version, tie-breaks
# ---------------------------------------------------------------------------


class TestScoring:
    def test_higher_total_is_better(self):
        low = _make_record(candidate_name="a-lib", version="1.0.0", clinical_technical_evidence=_ds(1, "low evidence one-liner"))
        high = _make_record(candidate_name="b-lib", version="1.0.0", clinical_technical_evidence=_ds(5, "high evidence one-liner"))
        assert audit_score(high) > audit_score(low)

    def test_weights_version_recorded_on_record(self):
        rec = _make_record()
        assert rec.weights_version == WEIGHTS_VERSION

    def test_unknown_weights_version_rejected(self):
        rec = _make_record()
        with pytest.raises(ValueError):
            audit_score(rec, weights_version="unknown_v9")

    def test_deterministic_tie_break_name_then_version(self):
        # Same total -> name lexicographically, then version
        r1 = _make_record(candidate_name="alpha", version="1.0.0")
        r2 = _make_record(candidate_name="alpha", version="2.0.0")
        r3 = _make_record(candidate_name="beta", version="1.0.0")
        # Force equal scores by using identical dimensions
        ranked = rank_audits([r3, r2, r1])
        assert ranked == [r1, r2, r3]

    def test_weighted_sum_formula_documented(self):
        # v1 uniform weights: total == sum(scores)
        rec = _make_record(
            clinical_technical_evidence=_ds(5, "evidence one-liner"),
            license_redistribution=_ds(4, "license one-liner"),
            reproducibility_auditability=_ds(3, "repro one-liner"),
            security_dependency_health=_ds(2, "security one-liner"),
            local_offline_operability=_ds(1, "offline one-liner"),
            maintainability=_ds(5, "maintain one-liner"),
            integration_cost=_ds(4, "integration one-liner"),
        )
        expected = 5 + 4 + 3 + 2 + 1 + 5 + 4
        assert audit_score(rec) == expected


# ---------------------------------------------------------------------------
# 3 — AuditLedger: capability projection, lookup, latest-status, JSON
# ---------------------------------------------------------------------------


class TestAuditLedger:
    def test_add_and_lookup(self):
        ledger = AuditLedger()
        rec = _make_record(candidate_name="my-lib", version="2.0.0")
        ledger.add(rec)
        assert ledger.get("my-lib", "2.0.0").candidate_name == "my-lib"
        assert ledger.lookup("my-lib", "2.0.0") == ledger.get("my-lib", "2.0.0")

    def test_lookup_case_insensitive(self):
        ledger = AuditLedger()
        ledger.add(_make_record(candidate_name="HighDicom", version="0.28.1"))
        assert ledger.get("highdicom", "0.28.1").candidate_name == "HighDicom"
        assert ledger.get("HIGHDICOM", "0.28.1").candidate_name == "HighDicom"

    def test_lookup_hyphen_space_equivalence(self):
        ledger = AuditLedger()
        ledger.add(_make_record(candidate_name="3D Slicer", version="5.12.3"))
        assert ledger.get("3d-slicer", "5.12.3").candidate_name == "3D Slicer"
        assert ledger.get("3d slicer", "5.12.3").candidate_name == "3D Slicer"

    def test_by_capability(self):
        ledger = AuditLedger()
        ledger.add(_make_record(candidate_name="lib-a", version="1.0.0", capability_area="dicom-import"))
        ledger.add(_make_record(candidate_name="lib-b", version="1.0.0", capability_area="volume-processing"))
        assert len(ledger.by_capability("dicom-import")) == 1
        assert ledger.by_capability("dicom-import")[0].candidate_name == "lib-a"

    def test_status_lifecycle_via_event_log(self):
        ledger = AuditLedger()
        rec = _make_record(candidate_name="lifecycle-lib", version="1.0.0")
        ledger.add(rec)
        assert ledger.project_status("lifecycle-lib", "1.0.0") == AuditStatus.DRAFT
        ledger.audit("lifecycle-lib", "1.0.0", by="auditor", evidence_ref="evidence")
        assert ledger.project_status("lifecycle-lib", "1.0.0") == AuditStatus.AUDITED
        # AUDITED -> SUPERSEDED is legal
        ledger.supersede("lifecycle-lib", "1.0.0", by="auditor")
        assert ledger.project_status("lifecycle-lib", "1.0.0") == AuditStatus.SUPERSEDED
        # terminal state cannot transition
        with pytest.raises(StateTransitionError):
            ledger.audit("lifecycle-lib", "1.0.0")

    def test_illegal_transition_rejected(self):
        ledger = AuditLedger()
        ledger.add(_make_record(candidate_name="bad-transition", version="1.0.0"))
        # DRAFT -> SUPERSEDED is illegal (must go via AUDITED)
        with pytest.raises(StateTransitionError):
            ledger.supersede("bad-transition", "1.0.0")

    def test_latest_status_projection_is_read_only(self):
        ledger = AuditLedger()
        ledger.add(_make_record(candidate_name="proj-lib", version="1.0.0"))
        proj = ledger.latest_status_projection("proj-lib", "1.0.0")
        assert proj["status"] == AuditStatus.DRAFT.value
        assert isinstance(proj, dict)

    def test_ranked_delegates_to_scoring(self):
        ledger = AuditLedger()
        low = _make_record(candidate_name="low-lib", version="1.0.0", clinical_technical_evidence=_ds(1, "low evidence one-liner"))
        high = _make_record(candidate_name="high-lib", version="1.0.0", clinical_technical_evidence=_ds(5, "high evidence one-liner"))
        ledger.add(low)
        ledger.add(high)
        ranked = ledger.ranked()
        assert ranked[0].candidate_name == "high-lib"

    def test_json_round_trip(self):
        ledger = AuditLedger()
        rec = _make_record(candidate_name="json-lib", version="1.0.0")
        ledger.add(rec)
        ledger.audit("json-lib", "1.0.0", by="auditor")
        dumped = ledger.to_json()
        restored = AuditLedger.from_json(dumped)
        assert len(restored) == 1
        assert restored.get("json-lib", "1.0.0").candidate_name == "json-lib"
        assert restored.project_status("json-lib", "1.0.0") == AuditStatus.AUDITED
        # second round-trip
        assert AuditLedger.from_dict(restored.to_dict()).to_dict() == restored.to_dict()

    def test_ledger_closed_schema_unknown_key_rejected(self):
        ledger = AuditLedger()
        ledger.add(_make_record(candidate_name="closed-lib", version="1.0.0"))
        data = ledger.to_dict()
        data["unknown"] = 1
        with pytest.raises(ClosedSchemaError):
            AuditLedger.from_dict(data)


# ---------------------------------------------------------------------------
# 4 — fail-closed validation: digest-bound sealed records
# ---------------------------------------------------------------------------


class TestDigestSeal:
    def test_sealed_record_carries_digest(self):
        ledger = AuditLedger()
        rec = _make_record(candidate_name="sealed-lib", version="1.0.0")
        sealed = ledger.add(rec)
        assert len(sealed.audit_digest) == 64
        assert compute_audit_digest(sealed) == sealed.audit_digest

    def test_tampered_frozen_block_fails_load(self):
        ledger = AuditLedger()
        sealed = ledger.add(_make_record(candidate_name="tamper-lib", version="1.0.0"))
        payload = record_to_dict(sealed)
        payload["version"] = "9.9.9"
        with pytest.raises(DigestMismatchError):
            record_from_dict(payload)

    def test_tampered_justification_fails_load(self):
        ledger = AuditLedger()
        sealed = ledger.add(_make_record(candidate_name="tamper-just", version="1.0.0"))
        payload = record_to_dict(sealed)
        payload["clinical_technical_evidence"]["justification"] = "evil tamper"
        with pytest.raises(DigestMismatchError):
            record_from_dict(payload)

    def test_status_is_transient_excluded_from_digest(self):
        ledger = AuditLedger()
        sealed = ledger.add(_make_record(candidate_name="transient-lib", version="1.0.0"))
        payload = record_to_dict(sealed)
        # Changing status should NOT invalidate the digest (transient)
        payload["status"] = AuditStatus.AUDITED.value
        # Should load successfully because digest excludes status
        rec2 = record_from_dict(payload)
        assert rec2.status == AuditStatus.AUDITED
        assert compute_audit_digest(rec2) == sealed.audit_digest

    def test_unknown_key_rejected_at_load(self):
        payload = record_to_dict(_make_record())
        payload["extra_field"] = "evil"
        with pytest.raises(ClosedSchemaError):
            record_from_dict(payload)

    def test_score_range_checked_at_load(self):
        payload = record_to_dict(_make_record())
        payload["clinical_technical_evidence"]["score"] = 10
        with pytest.raises(CanonicalValidationError):
            record_from_dict(payload)


# ---------------------------------------------------------------------------
# 5 — seed ledger
# ---------------------------------------------------------------------------


class TestSeedLedger:
    def test_seed_ledger_has_all_required_candidates(self):
        ledger = build_seed_ledger()
        # At least the 11 mandated plus extras =13
        assert len(ledger) >= 11
        required = [
            ("pydicom", "3.0.2"),
            ("HighDicom", "0.28.1"),
            ("dcm4che", "5.35.1"),
            ("pydicom-seg", "0.4.1"),
            ("SimpleITK", "2.5.5"),
            ("ITK", "5.4.6"),
            ("nibabel", "5.4.2"),
            ("MONAI", "1.6.0"),
            ("nnU-Net", "2.8.1"),
            ("VMTK", "1.4.0"),
            ("3D Slicer", "5.12.3"),
        ]
        for name, ver in required:
            rec = ledger.get(name, ver)
            assert rec.candidate_name is not None
            assert rec.version == ver

        # Extensions also present
        for name, ver in [("SlicerRT", "2.7.0"), ("SlicerHeart", "1.5.0")]:
            assert ledger.get(name, ver).candidate_name == name

    def test_seed_licenses_grounded(self):
        ledger = build_seed_ledger()
        checks = {
            "pydicom": "MIT",
            "HighDicom": "MIT",
            "SimpleITK": "Apache-2.0",
            "ITK": "Apache-2.0",
            "MONAI": "Apache-2.0",
            "VMTK": "BSD-3-Clause",
            "SlicerHeart": "BSD-3-Clause",
        }
        for name, expected_license in checks.items():
            # find seed with that name (version lookup via ledger)
            recs = [r for r in ledger.all_records() if r.candidate_name == name]
            assert recs, f"missing {name}"
            assert recs[0].license_id == expected_license, f"{name} license mismatch"

        # dcm4che tri-license
        dcm = ledger.get("dcm4che", "5.35.1")
        assert "MPL" in dcm.license_id and "GPL" in dcm.license_id
        # 3D Slicer custom
        slicer = ledger.get("3D Slicer", "5.12.3")
        assert "Slicer" in slicer.license_id or "LicenseRef" in slicer.license_id

    def test_seed_reuse_decisions_match_advisory(self):
        ledger = build_seed_ledger()
        assert ledger.get("pydicom", "3.0.2").reuse_decision == ReuseDecision.USE_DEPENDENCY
        assert ledger.get("HighDicom", "0.28.1").reuse_decision == ReuseDecision.USE_DEPENDENCY
        assert ledger.get("dcm4che", "5.35.1").reuse_decision == ReuseDecision.BUILD_ADAPTER
        assert ledger.get("VMTK", "1.4.0").reuse_decision == ReuseDecision.BUILD_ADAPTER
        assert ledger.get("3D Slicer", "5.12.3").reuse_decision == ReuseDecision.BUILD_ADAPTER

    def test_seed_status_is_audited(self):
        ledger = build_seed_ledger()
        for rec in ledger.all_records():
            assert ledger.project_status(rec.candidate_name, rec.version) == AuditStatus.AUDITED

    def test_seed_provenance_points_at_advisory(self):
        ledger = build_seed_ledger()
        for rec in ledger.all_records():
            paths = [p.path for p in rec.provenance_refs]
            # w9: seeds use ONLY committed embedded evidence, not the ignored advisory file
            assert any("embedded_advisory_evidence" in p for p in paths), f"{rec.candidate_name} missing embedded provenance"
            assert rec.evidence_provenance, "evidence_provenance must be non-empty and never invent citations"

    def test_seed_json_round_trip(self):
        ledger = build_seed_ledger()
        data = ledger.to_dict()
        restored = AuditLedger.from_dict(data)
        assert len(restored) == len(ledger)
        for rec in ledger.all_records():
            r2 = restored.get(rec.candidate_name, rec.version)
            assert r2.audit_digest == rec.audit_digest

    def test_seed_scores_are_one_liners(self):
        ledger = build_seed_ledger()
        for rec in ledger.all_records():
            for dim in DIMENSION_NAMES:
                val: DimensionScore = getattr(rec, dim)
                assert isinstance(val.justification, str)
                assert "\n" not in val.justification

    def test_seed_weights_version_is_v1(self):
        ledger = build_seed_ledger()
        for rec in ledger.all_records():
            assert rec.weights_version == WEIGHTS_VERSION

    def test_legal_transitions_graph_sanity(self):
        # Ensure DRAFT cannot go directly to SUPERSEDED
        assert AuditStatus.SUPERSEDED not in LEGAL_AUDIT_TRANSITIONS[AuditStatus.DRAFT]
        assert AuditStatus.AUDITED in LEGAL_AUDIT_TRANSITIONS[AuditStatus.DRAFT]
        assert AuditStatus.SUPERSEDED in LEGAL_AUDIT_TRANSITIONS[AuditStatus.AUDITED]
        # terminal states have no outgoing
        assert not LEGAL_AUDIT_TRANSITIONS[AuditStatus.SUPERSEDED]
        assert not LEGAL_AUDIT_TRANSITIONS[AuditStatus.REJECTED]


def test_dimension_score_immutability():
    ds = _ds(3, "immutable one-liner")
    with pytest.raises(AttributeError):
        ds.justification = "evil"  # type: ignore[misc]


def test_frozen_record_immutability():
    rec = _make_record()
    with pytest.raises(dataclasses.FrozenInstanceError):
        rec.candidate_name = "evil"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# w1..w6 — Luna Round-1 blocker regression tests (exact-SHA review on 93c5c4f)
# ---------------------------------------------------------------------------


def test_w1_write_scope_pyproject_untouched():
    """w1: pyproject.toml must not be touched by audit implementation."""
    # This test documents the scope rule; it passes if the file on disk
    # does not contain the worker-added per-file-ignores block.
    import pathlib

    text = pathlib.Path("pyproject.toml").read_text(encoding="utf-8")
    assert "test_audit_workflow.py" not in text, "pyproject.toml must not contain per-file ignores for tests (manager-owned)"


def test_w2_unverified_digest_rejected():
    """w2: ledger.add must verify a provided non-empty digest — 64 zeros must be rejected."""
    ledger = AuditLedger()
    rec = _make_record(candidate_name="w2-lib", version="1.0.0")
    # Craft a record carrying a bogus but hex-valid digest
    bogus = dataclasses.replace(rec, audit_digest="0" * 64)
    with pytest.raises(DigestMismatchError):
        ledger.add(bogus)


def test_w3_add_rejects_non_draft_and_from_dict_fails_on_projection_mismatch():
    """w3: add enforces DRAFT-only; from_dict fails closed on projection/status disagreement."""
    ledger = AuditLedger()
    audited_rec = _make_record(candidate_name="w3-lib", version="1.0.0", status=AuditStatus.AUDITED)
    with pytest.raises(CanonicalValidationError):
        ledger.add(audited_rec)
    # Now test from_dict silently rewriting — construct a ledger dict where stored status != projected
    ledger2 = AuditLedger()
    ledger2.add(_make_record(candidate_name="w3-ledger", version="1.0.0"))
    ledger2.audit("w3-ledger", "1.0.0", by="auditor")
    data = ledger2.to_dict()
    # Tamper stored record status to DRAFT while keeping AUDITED event
    records = cast("list[dict[str, Any]]", data["records"])
    for r in records:
        if r["candidate_name"] == "w3-ledger":
            r["status"] = AuditStatus.DRAFT.value
    with pytest.raises(CanonicalValidationError):
        AuditLedger.from_dict(data)


def test_w4_closed_schema_dimension_and_event():
    """w4: unknown keys inside dimension objects and ledger events are rejected."""
    payload = record_to_dict(_make_record(candidate_name="w4-lib", version="1.0.0"))
    payload["clinical_technical_evidence"]["evil"] = 1
    with pytest.raises(ClosedSchemaError):
        record_from_dict(payload)
    # Event unknown key
    ledger = AuditLedger()
    ledger.add(_make_record(candidate_name="w4-ledger", version="1.0.0"))
    ledger.audit("w4-ledger", "1.0.0", by="auditor")
    data = ledger.to_dict()
    data["events"][0]["unknown_key"] = "evil"  # type: ignore[index]
    with pytest.raises(ClosedSchemaError):
        AuditLedger.from_dict(data)


def test_w5_seed_provenance_has_real_digest():
    """w5: every seed audit carries a real 64-hex digest (embedded self-contained evidence)."""
    from models.audit.seeds import EMBEDDED_REF, build_seed_ledger

    ledger = build_seed_ledger()
    for rec in ledger.all_records():
        # At least one provenance ref must be the embedded self-contained one with real digest
        assert any(p.sha256 is not None and len(p.sha256) == 64 for p in rec.provenance_refs), f"{rec.candidate_name} missing verifiable digest"
        # No provenance ref may have sha None in seeds (all must be verifiable)
        for p in rec.provenance_refs:
            assert p.sha256 is not None, f"{rec.candidate_name} has provenance without sha"
            assert len(p.sha256) == 64
    # The embedded ref itself is the self-contained evidence
    assert EMBEDDED_REF.sha256 is not None and len(EMBEDDED_REF.sha256) == 64


def test_w6_build_seed_ledger_does_not_swallow_errors():
    """w6: build_seed_ledger must not silently swallow audit transition failures."""
    from models.audit import AuditStatus as AS  # local alias
    from models.audit.seeds import SEED_RECORDS

    # If a seed record were already AUDITED, adding it would now fail (w3) — ensure that without try/except the error propagates.
    # Construct a ledger and try to add a non-DRAFT seed directly — should raise, not be swallowed.
    ledger = AuditLedger()
    bad = dataclasses.replace(SEED_RECORDS[0], status=AS.AUDITED)
    with pytest.raises(CanonicalValidationError):
        ledger.add(bad)
    # Also ensure the normal build still succeeds (no silent fallback to DRAFT)
    from models.audit.seeds import build_seed_ledger as bsl

    ledger_ok = bsl()
    for rec in ledger_ok.all_records():
        assert ledger_ok.project_status(rec.candidate_name, rec.version) == AuditStatus.AUDITED


def test_w7_history_integrity_tampered_history_and_noop():
    """w7: load must validate event_type and history vs events; tampered history / NOOP must fail."""
    ledger = AuditLedger()
    ledger.add(_make_record(candidate_name="w7-lib", version="1.0.0"))
    ledger.audit("w7-lib", "1.0.0", by="auditor", evidence_ref="evidence")
    data = ledger.to_dict()
    # Tamper history: change first history entry's at
    records = cast("list[dict[str, Any]]", data["records"])
    for r in records:
        if r["candidate_name"] == "w7-lib":
            r["status_history"][0]["at"] = "2099-01-01T00:00:00Z"
    with pytest.raises(CanonicalValidationError):
        AuditLedger.from_dict(data)
    # Forge NOOP event: from==to
    ledger2 = AuditLedger()
    ledger2.add(_make_record(candidate_name="w7-noop", version="1.0.0"))
    ledger2.audit("w7-noop", "1.0.0", by="auditor")
    data2 = ledger2.to_dict()
    # Add a NOOP event DRAFT->DRAFT (illegal)
    events2 = cast("list[dict[str, Any]]", data2["events"])
    events2.append(
        {
            "entity": "technology_audit",
            "entity_id": "w7-noop@1.0.0",
            "event_type": "STATUS",
            "from_value": "AUDITED",
            "to_value": "AUDITED",
            "actor_role": "auditor",
            "at": "2026-08-25T10:00:00Z",
            "evidence_ref": None,
            "payload": {},
        }
    )
    # Also need to add matching history entry to pass len check, but from==to should still fail
    records2 = cast("list[dict[str, Any]]", data2["records"])
    for r in records2:
        if r["candidate_name"] == "w7-noop":
            history2 = cast("list[dict[str, Any]]", r["status_history"])
            history2.append(
                {"status": "AUDITED", "at": "2026-08-25T10:00:00Z", "by": "auditor", "evidence_ref": None}
            )
    with pytest.raises(CanonicalValidationError):
        AuditLedger.from_dict(data2)
    # Invalid event_type
    ledger3 = AuditLedger()
    ledger3.add(_make_record(candidate_name="w7-badtype", version="1.0.0"))
    ledger3.audit("w7-badtype", "1.0.0", by="auditor")
    data3 = ledger3.to_dict()
    data3["events"][0]["event_type"] = "BOGUS"  # type: ignore[index]
    with pytest.raises(CanonicalValidationError):
        AuditLedger.from_dict(data3)


def test_w8_malformed_types_rejected():
    """w8: DimensionScore(True, ...) and JSON score='3' / justification=12345 must be rejected."""
    with pytest.raises(TypeError):
        DimensionScore(True, "bool should be rejected one-liner")
    payload = record_to_dict(_make_record(candidate_name="w8-lib", version="1.0.0"))
    payload["clinical_technical_evidence"]["score"] = "3"
    with pytest.raises(CanonicalValidationError):
        record_from_dict(payload)
    payload2 = record_to_dict(_make_record(candidate_name="w8-lib2", version="1.0.0"))
    payload2["clinical_technical_evidence"]["justification"] = 12345
    with pytest.raises(CanonicalValidationError):
        record_from_dict(payload2)


def test_w9_seed_digests_use_only_embedded_evidence():
    """w9: seeds must use ONLY committed embedded evidence, no advisory file refs."""
    from models.audit.seeds import EMBEDDED_REF, SEED_RECORDS

    for rec in SEED_RECORDS:
        for p in rec.provenance_refs:
            assert "web_sol_ws01_tech_audit" not in p.path, f"{rec.candidate_name} still references advisory file {p.path}"
            assert p.sha256 is not None and len(p.sha256) == 64
        # At least one ref must be the embedded one
        assert any(p.path == EMBEDDED_REF.path for p in rec.provenance_refs), f"{rec.candidate_name} missing embedded ref"
    # Ledger must also use embedded evidence_ref for audit transitions
    ledger = build_seed_ledger()
    for ev in ledger.event_log.events:
        assert ev.evidence_ref is None or "web_sol_ws01_tech_audit" not in ev.evidence_ref, "audit evidence_ref must not be advisory file"
        if ev.evidence_ref is not None:
            assert "embedded_advisory_evidence" in ev.evidence_ref


# r2_b2/b3/b4 and r3_b1/b3 — Round-2 and Round-3 blocker prefixes required by exact-SHA reviews


def test_r2_b2_history_integrity():
    """r2_b2: history must match events exactly — tampered at/NOOP/event_type must fail."""
    test_w7_history_integrity_tampered_history_and_noop()


def test_r2_b3_type_coercion():
    """r2_b3: bool scores and string/int coercion must be rejected."""
    test_w8_malformed_types_rejected()


def test_r2_b4_seed_self_contained():
    """r2_b4: seeds must be fully self-contained with embedded digests."""
    test_w9_seed_digests_use_only_embedded_evidence()


def test_r3_b1_cwd_independent_hashing():
    """r3_b1: audit_digest must be independent of CWD — no filesystem hashing."""
    from models.audit.seeds import _EMBEDDED_ADVISORY_EVIDENCE, _EMBEDDED_ADVISORY_SHA, EMBEDDED_REF
    from models.audit.seeds import build_seed_ledger as bsl2

    # Digests must be over embedded constant strings only, not files
    assert _EMBEDDED_ADVISORY_SHA == __import__("hashlib").sha256(_EMBEDDED_ADVISORY_EVIDENCE.encode("utf-8")).hexdigest()
    ledger1 = bsl2()
    digests1 = {f"{r.candidate_name}@{r.version}": r.audit_digest for r in ledger1.all_records()}
    ledger2 = bsl2()
    digests2 = {f"{r.candidate_name}@{r.version}": r.audit_digest for r in ledger2.all_records()}
    assert digests1 == digests2, "digests must be deterministic and CWD-independent (no file hashing)"
    # No seed may reference the ignored advisory file path
    for rec in ledger1.all_records():
        for p in rec.provenance_refs:
            assert "orchestration/logs" not in p.path, "seed provenance must not reference ignored advisory file path"
            assert p.sha256 is not None
    # Embedded ref must be present
    assert any(p.path == EMBEDDED_REF.path for r in ledger1.all_records() for p in r.provenance_refs)


def test_r3_b3_evidence_ref_strict_types():
    """r3_b3: evidence_ref must be strictly str or None — int 123 must fail on both event and history."""
    payload = record_to_dict(_make_record(candidate_name="r3-b3-lib", version="1.0.0"))
    # History entry with int evidence_ref
    payload["status_history"] = [{"status": "AUDITED", "at": "2026-08-25T10:00:00Z", "by": "auditor", "evidence_ref": 123}]
    payload["status"] = "AUDITED"
    # Need a matching event to pass len check, but history type check should fail first
    with pytest.raises((ClosedSchemaError, CanonicalValidationError)):
        record_from_dict(payload)
    # Event int evidence_ref via ledger
    ledger = AuditLedger()
    ledger.add(_make_record(candidate_name="r3-b3-ledger", version="1.0.0"))
    ledger.audit("r3-b3-ledger", "1.0.0", by="auditor", evidence_ref="ok")
    data = ledger.to_dict()
    data["events"][0]["evidence_ref"] = 123  # type: ignore[index]
    # Also need to keep history consistent len, but int evidence_ref should still fail on event type check
    with pytest.raises((ClosedSchemaError, CanonicalValidationError)):
        AuditLedger.from_dict(data)


# ---------------------------------------------------------------------------
# r4_b1 / r4_b2 / r4_b3 — Luna Round-4 blocker tests (review on 52d5a3a)
# ---------------------------------------------------------------------------


def test_r4_b1_honest_provenance_digest_labeling():
    """r4_b1: no provenance ref may claim the digest of a file that was not hashed."""
    from models.audit.seeds import SEED_RECORDS

    for rec in SEED_RECORDS:
        for p in rec.provenance_refs:
            # Must NOT claim docs/phase-0/WS00_DESIGN.md with an embedded summary digest
            if "WS00_DESIGN.md" in p.path:
                pytest.fail(f"{rec.candidate_name} claims WS00_DESIGN.md file path but digest is embedded text")
            # Must be honestly labeled as embedded text
            assert p.path.startswith("models/audit/seeds.py#embedded_"), f"dishonest provenance path: {p.path}"
            assert p.sha256 is not None and len(p.sha256) == 64


def test_r4_b2_actor_role_by_unification_and_round_trip():
    """r4_b2: actor_role and by are unified; transition with explicit actor_role round-trips cleanly."""
    ledger = AuditLedger()
    ledger.add(_make_record(candidate_name="r4-b2-lib", version="1.0.0"))
    # Transition with explicit actor_role (different from default "auditor")
    ledger.transition("r4-b2-lib", "1.0.0", AuditStatus.AUDITED, actor_role="security-reviewer", evidence_ref="sec-audit")
    # Verify the in-memory record has by matching actor_role
    rec = ledger.get("r4-b2-lib", "1.0.0")
    assert rec.status_history[0].by == "security-reviewer"
    ev = ledger.event_log.events[0]
    assert ev.actor_role == "security-reviewer"
    assert rec.status_history[0].by == ev.actor_role
    # JSON round-trip must succeed and produce identical ledger
    dumped = ledger.to_json()
    reloaded = AuditLedger.from_json(dumped)
    assert len(reloaded) == 1
    reloaded_rec = reloaded.get("r4-b2-lib", "1.0.0")
    assert reloaded_rec.status_history[0].by == "security-reviewer"
    assert reloaded.project_status("r4-b2-lib", "1.0.0") == AuditStatus.AUDITED


def test_r4_b3_duplicate_records_rejected_in_from_dict():
    """r4_b3: exact AND normalized candidate/version duplicates must be REJECTED during from_dict."""
    ledger = AuditLedger()
    rec1 = _make_record(candidate_name="HighDicom", version="0.28.1")
    ledger.add(rec1)
    data = ledger.to_dict()
    # Exact duplicate
    exact_dup_data = {
        "schema_version": data["schema_version"],
        "records": [data["records"][0], data["records"][0]],  # type: ignore[index]
        "events": data["events"],
    }
    with pytest.raises(CanonicalValidationError) as err:
        AuditLedger.from_dict(exact_dup_data)
    assert any("duplicate record" in v for v in err.value.violations)

    # Normalized duplicate (e.g. "highdicom" vs "HighDicom" at same version)
    rec_norm = _make_record(candidate_name="highdicom", version="0.28.1")
    # need sealed digest for second record
    from models.audit.record import compute_audit_digest
    digest = compute_audit_digest(rec_norm)
    norm_dict = record_to_dict(dataclasses.replace(rec_norm, audit_digest=digest))
    norm_dup_data = {
        "schema_version": data["schema_version"],
        "records": [data["records"][0], norm_dict],  # type: ignore[index]
        "events": data["events"],
    }
    with pytest.raises(CanonicalValidationError) as err2:
        AuditLedger.from_dict(norm_dup_data)
    assert any("normalized duplicate" in v for v in err2.value.violations)


# ---------------------------------------------------------------------------
# r5_b1 / r5_b2 / r5_b3 — Terra xhigh Round-5 blockers (review on 9ec8c09)
# ---------------------------------------------------------------------------


def test_r5_b1_identity_type_coercion_rejected():
    """r5_b1: numeric/bool identity fields must be rejected at load even when
    the attacker recomputes a VALID digest over the coerced form — proving
    rejection comes from typed-schema validation, not DigestMismatchError."""
    base = _make_record(candidate_name="r5-b1-lib", version="1.0.0")
    for field in ("candidate_name", "source_url", "version", "capability_area"):
        for bad in (12345, True):
            payload = record_to_dict(base)
            payload[field] = bad
            # Recomputed valid digest over the str()-coerced identity so a
            # digest check cannot mask the type violation.
            coerced_identity = str(bad)
            if field == "candidate_name":
                coerced = dataclasses.replace(base, candidate_name=coerced_identity)
            elif field == "source_url":
                coerced = dataclasses.replace(base, source_url=coerced_identity)
            elif field == "version":
                coerced = dataclasses.replace(base, version=coerced_identity)
            else:
                coerced = dataclasses.replace(base, capability_area=coerced_identity)
            payload["audit_digest"] = compute_audit_digest(coerced)
            with pytest.raises(CanonicalValidationError) as err:
                record_from_dict(payload)
            assert not isinstance(err.value, DigestMismatchError), (
                f"{field}={bad!r} must fail typed-schema validation, not digest check"
            )
            assert any("must be str" in v for v in err.value.violations)


def test_r5_b1_validate_direct_rejects_non_str_identity():
    """r5_b1: the direct validate_audit_record path must also reject non-str
    identity fields with CanonicalValidationError (no AttributeError leaks)."""
    for field, bad in (
        ("candidate_name", 12345),
        ("source_url", True),
        ("version", 7.0),
        ("capability_area", 0),
    ):
        rec = dataclasses.replace(_make_record(), **{field: bad})  # type: ignore[arg-type]
        with pytest.raises(CanonicalValidationError) as err:
            validate_audit_record(rec)
        assert any(f"{field} must be str" in v for v in err.value.violations)


def test_r5_b2_transition_argument_types_zero_side_effects():
    """r5_b2: non-str/falsey transition arguments and blank effective actors
    raise CanonicalValidationError BEFORE lookup/append — zero side effects."""
    ledger = AuditLedger()
    ledger.add(_make_record(candidate_name="r5-b2-lib", version="1.0.0"))
    baseline_json = ledger.to_json()
    probes: list[dict[str, object]] = [
        {"by": False},  # falsey default would silently swallow
        {"actor_role": 0},
        {"evidence_ref": 123},
        {"at": 0},
        {"by": ""},  # blank effective actor
        {"actor_role": "   "},  # whitespace-only effective actor
    ]
    to_status: AuditStatus = AuditStatus.AUDITED
    for kw in probes:
        with pytest.raises(CanonicalValidationError):
            ledger.transition("r5-b2-lib", "1.0.0", to_status, **kw)  # type: ignore[arg-type]
        cur = ledger.get("r5-b2-lib", "1.0.0")
        assert cur.status == AuditStatus.DRAFT, f"probe {kw} mutated status"
        assert cur.status_history == (), f"probe {kw} mutated history"
        assert len(ledger.event_log) == 0, f"probe {kw} appended an event"
        assert ledger.to_json() == baseline_json, f"probe {kw} changed JSON"
    # plain str masquerading as AuditStatus must be rejected (exact type)
    with pytest.raises(CanonicalValidationError):
        ledger.transition("r5-b2-lib", "1.0.0", "AUDITED")  # type: ignore[arg-type]
    # non-str candidate_name/version
    with pytest.raises(CanonicalValidationError):
        ledger.transition(123, "1.0.0", AuditStatus.AUDITED)  # type: ignore[arg-type]
    with pytest.raises(CanonicalValidationError):
        ledger.transition("r5-b2-lib", None, AuditStatus.AUDITED)  # type: ignore[arg-type]
    assert ledger.to_json() == baseline_json


def test_r5_b2_explicit_none_and_defaults_still_work():
    """r5_b2: explicit `is None` defaulting keeps legitimate calls working."""
    ledger = AuditLedger()
    ledger.add(_make_record(candidate_name="r5-b2-ok", version="1.0.0"))
    # No by/actor_role at all -> defaults to "auditor"
    ledger.transition("r5-b2-ok", "1.0.0", AuditStatus.AUDITED)
    assert ledger.get("r5-b2-ok", "1.0.0").status_history[0].by == "auditor"
    # Explicit None by -> still defaults cleanly
    ledger.transition(
        "r5-b2-ok",
        "1.0.0",
        AuditStatus.SUPERSEDED,
        by=None,
        evidence_ref=None,
        at=None,
        actor_role=None,
    )
    rec = ledger.get("r5-b2-ok", "1.0.0")
    assert rec.status == AuditStatus.SUPERSEDED
    assert rec.status_history[1].by == "auditor"


def test_r5_b3_event_log_view_is_read_only():
    """r5_b3: the public event_log view exposes reads only — no append_event/seed."""
    from models.audit.ledger import EventLogView

    ledger = AuditLedger()
    ledger.add(_make_record(candidate_name="r5-b3-view", version="1.0.0"))
    view = ledger.event_log
    assert isinstance(view, EventLogView)
    # read access works
    assert len(view) == len(view.events)
    proj = view.project_state("r5-b3-view@1.0.0", entity="technology_audit")
    assert proj["status"] == AuditStatus.DRAFT.value
    # mutation surface is gone
    assert not hasattr(view, "append_event")
    assert not hasattr(view, "seed")
    with pytest.raises(AttributeError):
        getattr(view, "append_event")
    with pytest.raises(AttributeError):
        getattr(view, "seed")


def test_r5_b3_constructor_log_injection_no_external_alias():
    """r5_b3: injecting a caller-owned EventLog must snapshot it; external
    appends to the original must not mutate the ledger or its JSON."""
    ext_log = EventLog()
    ledger = AuditLedger(log=ext_log)
    ledger.add(_make_record(candidate_name="alias-lib", version="1.0.0"))
    ledger.audit("alias-lib", "1.0.0", by="auditor")
    json_before = ledger.to_json()
    # Attacker mutates ONLY the caller-owned log (seed + forged event)
    ext_log.seed(
        entity="technology_audit",
        entity_id="alias-lib@1.0.0",
        initial_fields={"status": "DRAFT"},
    )
    ext_log.append_event(
        StateEvent(
            entity="technology_audit",
            entity_id="alias-lib@1.0.0",
            event_type="STATUS",
            from_value="DRAFT",
            to_value="SUPERSEDED",
            actor_role="attacker",
            at=utc_now_iso(),
            evidence_ref=None,
        )
    )
    assert len(ext_log.events) == 1, "external mutation must succeed on the caller-owned log"
    # Ledger state untouched: status projection and stored record stay AUDITED
    assert ledger.project_status("alias-lib", "1.0.0") == AuditStatus.AUDITED
    rec = ledger.get("alias-lib", "1.0.0")
    assert rec.status == AuditStatus.AUDITED
    assert len(rec.status_history) == 1
    assert ledger.to_json() == json_before
    # JSON round-trip stays unchanged after external mutation attempts
    reloaded = AuditLedger.from_json(json_before)
    assert reloaded.to_dict() == ledger.to_dict()
    assert reloaded.project_status("alias-lib", "1.0.0") == AuditStatus.AUDITED


# ---------------------------------------------------------------------------
# r6_b1 / r6_b2 — Terra xhigh Round-6 residual blocker (review on 69244c6)
# ---------------------------------------------------------------------------


def test_r6_b1_event_log_view_hides_raw_log_no_external_append_path():
    """r6_b1: the view must not expose the raw mutable EventLog under ANY
    attribute (``_log`` included), so ``ledger.event_log._log.append_event``
    cannot desynchronize projection vs stored record; after every failed
    external-append attempt the ledger stays fully consistent."""
    ledger = AuditLedger()
    ledger.add(_make_record(candidate_name="r6-b1-view", version="1.0.0"))
    view = ledger.event_log
    # The raw log attribute is gone entirely — no _log, no dict to inject it into.
    assert not hasattr(view, "_log")
    with pytest.raises(AttributeError):
        getattr(view, "_log")  # noqa: B009
    with pytest.raises(AttributeError):
        view._log  # type: ignore[attr-defined]  # noqa: B018
    assert not hasattr(view, "__dict__"), "slots-only view must not admit attribute injection"
    # No reachable attribute on the view exposes a mutation surface
    for attr_name in dir(view):
        obj = getattr(view, attr_name)
        assert not hasattr(obj, "append_event"), f"view.{attr_name} leaks append_event"
        assert not hasattr(obj, "seed"), f"view.{attr_name} leaks seed"
    # Baseline consistency: DRAFT everywhere, no events yet
    baseline_json = ledger.to_json()
    ref = "r6-b1-view@1.0.0"
    proj = view.project_state(ref, entity="technology_audit")
    assert proj["status"] == AuditStatus.DRAFT.value
    assert len(view) == 0 and list(view) == []
    # A legitimate transition is observed on a FRESH snapshot (the earlier
    # point-in-time view stays frozen at its capture point)...
    updated = ledger.audit("r6-b1-view", "1.0.0", evidence_ref="ev-r6b1")
    assert updated.status == AuditStatus.AUDITED
    assert view.project_state(ref, entity="technology_audit")["status"] == AuditStatus.DRAFT.value
    assert len(view.events) == 0, "captured snapshots never change"
    view2 = ledger.event_log
    assert view2 is not view
    proj2 = view2.project_state(ref, entity="technology_audit")
    assert proj2["status"] == AuditStatus.AUDITED.value
    hist = view2.history_for(ref, entity="technology_audit")
    assert len(hist) == 1 and hist[0].to_value == "AUDITED" and hist[0].actor_role == "auditor"
    stored = ledger.get("r6-b1-view", "1.0.0")
    assert stored.status == AuditStatus.AUDITED and len(stored.status_history) == 1
    # ...and projection/record/history stay mutually consistent (no forged append)
    assert len(view2.events) == len(stored.status_history)
    assert view2.events[0].from_value == AuditStatus.DRAFT.value
    assert ledger.to_json() != baseline_json
    reloaded = AuditLedger.from_json(ledger.to_json())
    assert reloaded.project_status("r6-b1-view", "1.0.0") == AuditStatus.AUDITED


def test_r6_b2_surfaced_event_payloads_immutable_round_trip_stable():
    """r6_b2: event data surfaced through the view is immutable/defensively
    copied — mutating a returned payload (or any attacker-held copy) raises
    or diverges locally and can never change to_json()/from_json()."""
    ledger = AuditLedger()
    ledger.add(_make_record(candidate_name="r6-b2-pay", version="1.0.0"))
    ledger.transition("r6-b2-pay", "1.0.0", AuditStatus.AUDITED, evidence_ref="ev-r6b2")
    # Give the STATUS event a non-empty payload via the canonical JSON path,
    # then reload — payload content is serialized, so tampering WOULD matter.
    data = ledger.to_dict()
    assert data["events"], "expected at least one event"
    data["events"][0]["payload"] = {"rationale": "audit committee minutes"}  # type: ignore[index]
    loaded = AuditLedger.from_dict(data)
    json_before = loaded.to_json()
    assert "audit committee minutes" in json_before

    surfaced = loaded.event_log.events[0]
    # Payload writes are blocked outright (MappingProxyType over a copy).
    with pytest.raises(TypeError):
        surfaced.payload["rationale"] = "tampered"  # type: ignore[index]
    # Frozen dataclass field rebinding stays blocked too.
    with pytest.raises(dataclasses.FrozenInstanceError):
        surfaced.actor_role = "attacker"  # type: ignore[misc]
    # An attacker-held copy diverges only locally.
    held_copy = dict(surfaced.payload)
    held_copy["rationale"] = "tampered-copy"
    assert surfaced.payload["rationale"] == "audit committee minutes"

    # history_for and iteration surface equally protected copies
    hist = loaded.event_log.history_for("r6-b2-pay@1.0.0", entity="technology_audit")
    assert hist[0].payload["rationale"] == "audit committee minutes"
    with pytest.raises(TypeError):
        hist[0].payload["rationale"] = "tampered"  # type: ignore[index]
    for ev in loaded.event_log:
        with pytest.raises(TypeError):
            ev.payload["rationale"] = "tampered"  # type: ignore[index]
    # Even empty payloads of normally-built ledgers reject writes
    plain_evt = ledger.event_log.events[0]
    with pytest.raises(TypeError):
        plain_evt.payload["injected"] = "x"  # type: ignore[index]

    # The returned Projection is already a per-call replay copy: mutating it
    # cannot reach the ledger either.
    proj_mut = loaded.event_log.project_state("r6-b2-pay@1.0.0", entity="technology_audit")
    proj_mut["status"] = "REJECTED"
    assert loaded.project_status("r6-b2-pay", "1.0.0") == AuditStatus.AUDITED

    # Round-trip byte-stable and reload-equal despite all mutation attempts
    assert loaded.to_json() == json_before
    reloaded = AuditLedger.from_json(json_before)
    assert reloaded.to_dict() == loaded.to_dict()
    assert reloaded.get("r6-b2-pay", "1.0.0").status == AuditStatus.AUDITED


# ---------------------------------------------------------------------------
# r7_b3-a / r7_b3-b — Terra xhigh Round-7 blockers (review on 4166a9d)
# ---------------------------------------------------------------------------


def test_r7_b3a_raw_log_unreachable_via_introspection():
    """r7_b3-a: the raw mutable EventLog must be unreachable from the public
    view through ANY metadata path — instance/class attributes, method
    ``__closure__`` cells, ``__defaults__``/``__kwdefaults__`` — so an
    external append via any discovered reference is impossible."""
    import types as _types

    ledger = AuditLedger()
    ledger.add(_make_record(candidate_name="r7-b3a-view", version="1.0.0"))
    ledger.audit("r7-b3a-view", "1.0.0", evidence_ref="ev-r7b3a")
    view = ledger.event_log
    raw_log = ledger._log  # noqa: SLF001 — test-side handle for identity probes
    baseline_json = ledger.to_json()

    # The exact reproduction vector from the review must be dead:
    assert type(view).history_for.__closure__ is None, "method closure still holds the log"
    events_property = cast("property", inspect.getattr_static(type(view), "events"))
    events_getter = cast("_types.FunctionType | None", events_property.fget)
    assert events_getter is not None and events_getter.__closure__ is None, "property getter closure still holds the log"
    # No instance __dict__ to stash anything in; no _log attribute either.
    assert not hasattr(view, "__dict__")
    assert not hasattr(view, "_log")
    with pytest.raises(AttributeError):
        getattr(view, "_log")  # noqa: B009

    surfaces: list[tuple[str, object]] = []
    for slot_name in getattr(type(view), "__slots__", ()):
        surfaces.append((f"instance.{slot_name}", getattr(view, slot_name)))
    for name, member in vars(type(view)).items():
        target: object = member
        if isinstance(member, property):
            target = member.fget
        elif isinstance(member, (classmethod, staticmethod)):
            target = member.__func__
        surfaces.append((f"class.{name}", target))
        if isinstance(target, _types.FunctionType):
            surfaces.append((f"{name}.__closure__", target.__closure__))
            surfaces.append((f"{name}.__defaults__", target.__defaults__))
            surfaces.append((f"{name}.__kwdefaults__", target.__kwdefaults__))

    def _assert_clean(label: str, obj: object) -> None:
        if obj is None:
            return
        if isinstance(obj, _types.CellType):
            inner = obj.cell_contents
            assert inner is not raw_log, f"{label} holds the raw log"
            assert not hasattr(inner, "append_event"), f"{label} leaks append_event"
            assert not hasattr(inner, "seed"), f"{label} leaks seed"
            return
        assert obj is not raw_log, f"{label} IS the raw log"
        assert not hasattr(obj, "append_event"), f"{label} exposes append_event"
        assert not hasattr(obj, "seed"), f"{label} exposes seed"

    for label, obj in surfaces:
        if isinstance(obj, tuple):
            for idx, item in enumerate(obj):
                _assert_clean(f"{label}[{idx}]", item)
        else:
            _assert_clean(label, obj)

    # Snapshot content remains correct and consistent; nothing was perturbed.
    ref = "r7-b3a-view@1.0.0"
    assert len(view.events) == 1
    proj = view.project_state(ref, entity="technology_audit")
    assert proj["status"] == AuditStatus.AUDITED.value
    stored = ledger.get("r7-b3a-view", "1.0.0")
    assert stored.status == AuditStatus.AUDITED and len(stored.status_history) == 1
    assert len(view.events) == len(stored.status_history)
    assert ledger.to_json() == baseline_json
    assert AuditLedger.from_json(baseline_json).to_dict() == ledger.to_dict()


def test_r7_b3b_deep_frozen_nested_payloads_round_trip_stable():
    """r7_b3-b: payload structures surfaced by the view are deeply immutable —
    mutation at ANY nesting depth raises, held copies diverge locally only,
    and to_json()/from_json() stay byte-stable with the internal log untouched."""
    ledger = AuditLedger()
    ledger.add(_make_record(candidate_name="r7-b3b-pay", version="1.0.0"))
    ledger.transition("r7-b3b-pay", "1.0.0", AuditStatus.AUDITED, evidence_ref="ev-r7b3b")
    data = ledger.to_dict()
    assert data["events"], "expected at least one event"
    data["events"][0]["payload"] = {  # type: ignore[index]
        "rationale": "committee minutes",
        "nested": {"reason": "original", "deeper": {"leaf": "keep"}},
        "chain": [{"step": 1}],
    }
    loaded = AuditLedger.from_dict(data)
    json_before = loaded.to_json()
    assert "committee minutes" in json_before and '"reason"' in json_before

    evt = loaded.event_log.events[0]
    # Depth-2 and depth-3 dict mutations are blocked...
    with pytest.raises(TypeError):
        evt.payload["nested"]["reason"] = "tampered"  # type: ignore[index]
    with pytest.raises(TypeError):
        evt.payload["nested"]["deeper"]["leaf"] = "tampered"  # type: ignore[index]
    # ...as is mutation through a list element.
    with pytest.raises(TypeError):
        evt.payload["chain"][0]["step"] = 99  # type: ignore[index]
    with pytest.raises(dataclasses.FrozenInstanceError):
        evt.actor_role = "attacker"  # type: ignore[misc]

    # Attacker-held copies diverge locally only; inner values stay protected.
    payload_view: Mapping[str, Mapping[str, object]] = cast(
        "Mapping[str, Mapping[str, object]]", evt.payload
    )
    held: dict[str, object] = dict(cast("Mapping[str, object]", evt.payload))
    held["nested"] = {"reason": "forged"}
    assert payload_view["nested"]["reason"] == "original"
    inner_held: dict[str, object] = dict(payload_view["nested"])
    deeper_held = cast("dict[str, object]", inner_held["deeper"])
    with pytest.raises(TypeError):
        deeper_held["leaf"] = "forged"
    assert cast("dict[str, object]", payload_view["nested"]["deeper"])["leaf"] == "keep"

    # A fully serialized copy can be mutated freely — it shares nothing.
    independent = json.loads(json.dumps(loaded.to_dict()))
    independent["events"][0]["payload"]["nested"]["reason"] = "forged"
    assert loaded.to_json() == json_before

    # history_for and iteration surface equally deep-frozen copies.
    hist = loaded.event_log.history_for("r7-b3b-pay@1.0.0", entity="technology_audit")
    hist_payload: Mapping[str, Mapping[str, object]] = cast(
        "Mapping[str, Mapping[str, object]]", hist[0].payload
    )
    assert hist_payload["nested"]["reason"] == "original"
    hist_deeper = cast("dict[str, object]", hist_payload["nested"]["deeper"])
    with pytest.raises(TypeError):
        hist_deeper["leaf"] = "tampered"
    for ev in loaded.event_log:
        with pytest.raises(TypeError):
            ev.payload["nested"]["reason"] = "tampered"  # type: ignore[index]

    # Internal log untouched; byte-stable JSON round-trip.
    assert loaded.to_json() == json_before
    reloaded = AuditLedger.from_json(json_before)
    assert reloaded.to_json() == json_before
    assert reloaded.get("r7-b3b-pay", "1.0.0").status == AuditStatus.AUDITED
