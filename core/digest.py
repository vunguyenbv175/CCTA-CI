"""Canonical hashing (I12, Luna fix #8).

Every self-digest (``record_sha256``, ``release_digest``) is computed over
the canonical JSON of a record AFTER removing self-referential digest fields
(fixed exclusion list per schema version). All modules hash through this one
function; no module implements its own hashing.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from enum import Enum
from typing import Any

#: Self-referential digest fields excluded from canonical hashing, per
#: schema version. Extending this table is a schema-version bump.
DIGEST_EXCLUSIONS: dict[str, frozenset[str]] = {
    "ccta.canonical.provenance_record.v1": frozenset({"record_sha256"}),
    "ccta.canonical.study.v1": frozenset(),
    "ccta.canonical.series.v1": frozenset(),
    "ccta.canonical.volume.v1": frozenset(),
    "ccta.canonical.derived_artifact.v1": frozenset(),
    # I12/I13 (Luna R3 fix #4): the release digest covers ONLY the frozen
    # identity block; mutable-lifecycle fields live in the event log and are
    # excluded so status transitions can never invalidate a sealed release.
    "ccta.model_registry.v1": frozenset(
        {
            "release_digest",
            "status",
            "status_history",
            "champion_of",
            "promotion",
            "acceptance_ref",
        }
    ),
}

_HEX_DIGITS = frozenset("0123456789abcdef")


def canonical_json(value: Any, exclude: tuple[str, ...] | list[str] = ()) -> str:
    """Serialize *value* as canonical JSON (sorted keys, tight separators).

    Keys named in *exclude* are dropped at every nesting level before
    serialization so self-referential digests cannot influence themselves.
    """
    pruned = _prune(value, frozenset(exclude))
    return json.dumps(pruned, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical_digest(value: Any, exclude: tuple[str, ...] | list[str] = ()) -> str:
    """SHA-256 hex digest of :func:`canonical_json` of *value*."""
    return hashlib.sha256(canonical_json(value, exclude).encode("utf-8")).hexdigest()


def is_sha256_hex(value: object) -> bool:
    """True iff *value* is a lowercase 64-hex string."""
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(c in _HEX_DIGITS for c in value)
    )


def require_sha256_hex(value: object, field: str) -> str:
    """Return *value* if it is a 64-hex digest, else raise ``ValueError``."""
    if not is_sha256_hex(value):
        raise ValueError(f"{field} must be a 64-hex sha256 digest, got {value!r}")
    return str(value)


def _prune(value: Any, exclude: frozenset[str]) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _prune(item, exclude)
            for key, item in value.items()
            if key not in exclude
        }
    if isinstance(value, (list, tuple)):
        return [_prune(item, exclude) for item in value]
    if isinstance(value, Enum):
        return _prune(value.value, exclude)
    if is_dataclass(value) and not isinstance(value, type):
        return _prune({f.name: getattr(value, f.name) for f in fields(value)}, exclude)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"not canonically serializable: {type(value)!r}")


__all__ = [
    "DIGEST_EXCLUSIONS",
    "canonical_digest",
    "canonical_json",
    "is_sha256_hex",
    "require_sha256_hex",
]
