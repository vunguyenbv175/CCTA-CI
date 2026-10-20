"""Time-sortable identifiers (Part B §2.2).

ULID-shaped IDs: 48-bit millisecond timestamp + 80-bit randomness, Crockford
base32, 26 uppercase characters. Pure stdlib; no central issuer. The final
choice between ULID and UUIDv7 belongs to ADR §59; both are time-sortable,
so swapping later does not change any schema.
"""

from __future__ import annotations

import os
import time

_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
_ENCODE_LEN = 26
_TIME_CHARS = 10
_RANDOM_CHARS = 16


def new_ulid(at_ms: int | None = None) -> str:
    """Return a new ULID string.

    ``at_ms`` allows deterministic IDs in tests; it must be non-negative and
    fit in 48 bits.
    """
    ts = time.time_ns() // 1_000_000 if at_ms is None else at_ms
    if ts < 0 or ts >= 1 << 48:
        raise ValueError(f"timestamp out of ULID range: {ts}")
    randomness = int.from_bytes(os.urandom(10), "big")
    value = (ts << 80) | randomness
    chars: list[str] = []
    for shift in range(_ENCODE_LEN - 1, -1, -1):
        chars.append(_CROCKFORD[(value >> (shift * 5)) & 0x1F])
    return "".join(chars)


def ulid_timestamp_ms(ulid: str) -> int:
    """Extract the 48-bit millisecond timestamp embedded in a ULID."""
    _validate_ulid(ulid)
    value = 0
    for char in ulid[:_TIME_CHARS]:
        value = (value << 5) | _CROCKFORD.index(char)
    return value


def validate_ulid(ulid: str) -> None:
    """Raise ``ValueError`` unless *ulid* is a well-formed 26-char ULID."""
    _validate_ulid(ulid)


def _validate_ulid(ulid: str) -> None:
    if len(ulid) != _ENCODE_LEN or any(c not in _CROCKFORD for c in ulid):
        raise ValueError(f"malformed ULID: {ulid!r}")


__all__ = ["new_ulid", "ulid_timestamp_ms", "validate_ulid"]
