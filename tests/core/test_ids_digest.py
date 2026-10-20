"""I12 canonical hashing + ULID identity scheme (Part B §2.2)."""

from __future__ import annotations

import pytest

from core.digest import (
    DIGEST_EXCLUSIONS,
    canonical_digest,
    canonical_json,
    is_sha256_hex,
)
from core.ids import new_ulid, ulid_timestamp_ms, validate_ulid


class TestCanonicalDigest:
    def test_deterministic_across_two_serializations(self):
        record_a = {"b": 1, "a": {"z": [1, 2], "y": "x"}}
        record_b = {"a": {"y": "x", "z": [1, 2]}, "b": 1}
        assert canonical_digest(record_a) == canonical_digest(record_b)

    def test_tampered_excluded_field_leaves_digest_unchanged(self):
        base = {"record_sha256": "old", "payload": "v1"}
        tampered = {"record_sha256": "EVIL", "payload": "v1"}
        exclude = ("record_sha256",)
        assert canonical_digest(base, exclude) == canonical_digest(tampered, exclude)

    def test_tampered_included_field_changes_digest(self):
        base = {"record_sha256": "old", "payload": "v1"}
        tampered = {"record_sha256": "old", "payload": "TAMPERED"}
        exclude = ("record_sha256",)
        assert canonical_digest(base, exclude) != canonical_digest(tampered, exclude)

    def test_exclusion_applies_at_every_nesting_level(self):
        doc = {"outer": {"release_digest": "self"}, "release_digest": "self"}
        assert "release_digest" not in canonical_json(doc, exclude=("release_digest",))

    def test_provenance_schema_has_self_reference_exclusion(self):
        assert DIGEST_EXCLUSIONS["ccta.canonical.provenance_record.v1"] == frozenset(
            {"record_sha256"}
        )

    def test_non_serializable_rejected(self):
        with pytest.raises(TypeError):
            canonical_digest({"f": object()})


class TestShaHelpers:
    def test_is_sha256_hex(self):
        assert is_sha256_hex("0" * 64)
        assert not is_sha256_hex("g" * 64)
        assert not is_sha256_hex("0" * 63)


class TestUlids:
    def test_format_and_validation(self):
        value = new_ulid(at_ms=1_700_000_000_000)
        assert len(value) == 26
        validate_ulid(value)
        with pytest.raises(ValueError):
            validate_ulid("not-a-ulid")

    def test_time_sortable(self):
        earlier = new_ulid(at_ms=1_000)
        later = new_ulid(at_ms=2_000)
        assert earlier < later
        assert ulid_timestamp_ms(earlier) == 1_000

    def test_uniqueness_without_clock_argument(self):
        assert len({new_ulid() for _ in range(500)}) == 500

    def test_out_of_range_timestamp_rejected(self):
        with pytest.raises(ValueError):
            new_ulid(at_ms=-1)
