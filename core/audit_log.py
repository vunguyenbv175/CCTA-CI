"""PHI-safe audit + diagnostic logging (WS02_DESIGN.md Part B §B8).

Two append-only streams share one mechanical core:

- ``operational``: import/save/reopen/export/egress-block/correction events;
- ``diagnostic``: exceptions with stack traces (paths and identifiers scrubbed).

Guarantees (pre-research §2.3 "MUST ADD" list):

1. PHI redaction BEFORE persistence: direct identifiers become keyed-SHA-256
   pseudonyms (HMAC-SHA256 under a workspace-held key); absolute file paths
   are replaced by pseudonymous tokens; raw pixel data is never accepted by
   this API at all.
2. Append-only rotating JSONL files with a SHA-256 chain seal per record
   (``seal = sha256(prev_seal || canonical_json(record))``) — tamper-evident,
   verifiable with :func:`verify_stream`.
3. NO second state machine: event records reuse the ``core.events.StateEvent``
   field vocabulary (entity/entity_id/event_type/from_value/to_value/
   actor_role/at/evidence_ref) but this module implements no transitions and
   no projections; clinical entity state stays exclusively in ``EventLog``.

Compliance targets: Master Plan §6.8 line 479, §50B line 1773, F4 §3.5.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Final

from core.digest import canonical_json
from core.events import utc_now_iso

OPERATIONAL_STREAM: Final[str] = "operational"
DIAGNOSTIC_STREAM: Final[str] = "diagnostic"
GENESIS_SEAL: Final[str] = "0" * 64
DEFAULT_MAX_BYTES: Final[int] = 1_048_576

#: Dict keys whose values are direct identifiers (Master Plan §50B, F4 §3.5).
#: Comparison is case-folded on the raw key text.
DIRECT_IDENTIFIER_KEYS: Final[frozenset[str]] = frozenset(
    {
        "patientname",
        "patientid",
        "patientidentifier",
        "birthdate",
        "patientbirthdate",
        "accessionnumber",
        "studyid",
        "ssn",
        "personname",
    }
)

_PSEUDONYM_PREFIX: Final[str] = "psn-"
_DEFAULT_PSN_LENGTH: Final[int] = 24
_MAX_SCRUBBED_TEXT: Final[int] = 4000

_WINDOWS_PATH_RE: Final[re.Pattern[str]] = re.compile(r"[A-Za-z]:[\\/][^\s\"'<>|*\]\),]+")
_UNC_PATH_RE: Final[re.Pattern[str]] = re.compile(r"\\\\[^\\/\s:\"'<>|]+(?:[\\/][^\\/\s:\"'<>|]+)*")
_POSIX_PATH_RE: Final[re.Pattern[str]] = re.compile(r"(?<![\w~.\-])/(?:[\w.@+\-]+/)+[\w.@+\-]*")


def pseudonymize(value: str, key: str | bytes, *, length: int = _DEFAULT_PSN_LENGTH) -> str:
    """Stable keyed pseudonym (HMAC-SHA256) for a direct identifier."""
    key_bytes = key.encode("utf-8") if isinstance(key, str) else key
    if not key_bytes:
        raise ValueError("pseudonym key must be non-empty")
    if length < 8 or length > 64:
        raise ValueError("pseudonym length must be within [8, 64]")
    digest = hmac.new(key_bytes, value.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{_PSEUDONYM_PREFIX}{digest[:length]}"


def redact_absolute_paths(text: str, key: str | bytes) -> str:
    """Replace absolute filesystem paths with pseudonymous tokens."""
    def _sub(match: re.Match[str]) -> str:
        return f"<redacted-path:{pseudonymize(match.group(0), key)}>"

    text = _UNC_PATH_RE.sub(_sub, text)
    text = _WINDOWS_PATH_RE.sub(_sub, text)
    return _POSIX_PATH_RE.sub(_sub, text)


class PhiSafeAuditSink:
    """Append-only, chain-sealed, PHI-redacted audit sink (Part B §B8)."""

    def __init__(
        self,
        directory: Path,
        *,
        pseudonym_key: str | bytes,
        stream: str = OPERATIONAL_STREAM,
        max_bytes: int = DEFAULT_MAX_BYTES,
        actor_role: str = "WORKSTATION",
    ) -> None:
        key_bytes = (
            pseudonym_key.encode("utf-8") if isinstance(pseudonym_key, str) else pseudonym_key
        )
        if not key_bytes:
            raise ValueError("pseudonym_key must be non-empty")
        if max_bytes < 512:
            raise ValueError("max_bytes must be >= 512 to hold at least one sealed record")
        self._directory = Path(directory)
        self._directory.mkdir(parents=True, exist_ok=True)
        self._key: bytes = key_bytes
        self._stream = stream
        self._actor_role = actor_role
        self._max_bytes = max_bytes
        self._registered: list[tuple[str, str]] = []
        self._writer = _SealedJsonlWriter(
            self._directory / f"{stream}-audit.jsonl", max_bytes=max_bytes
        )
        self._diagnostic_writer: _SealedJsonlWriter | None = None

    @property
    def stream_name(self) -> str:
        return self._stream

    @property
    def path(self) -> Path:
        return self._writer.path

    # -- registration of known sensitive literals ----------------------------

    def register_identifier(self, raw: str, label: str) -> None:
        """Scrub *raw* from free text, replacing it with a labelled pseudonym."""
        if not raw:
            raise ValueError("cannot register an empty identifier")
        if not label or not label.replace("_", "").isalnum():
            raise ValueError("identifier label must be alphanumeric/underscore")
        self._registered.append((raw, label))

    # -- operational audit -----------------------------------------------------

    def record(
        self,
        *,
        entity: str,
        entity_id: str,
        event_type: str,
        to_value: str,
        from_value: str | None = None,
        evidence_ref: str | None = None,
        payload: Mapping[str, object] | None = None,
        actor_role: str | None = None,
        at: str | None = None,
    ) -> str:
        """Append one operational audit event; returns its chain seal."""
        body: dict[str, object] = {
            "at": at or utc_now_iso(),
            "entity": entity,
            "entity_id": self._scrub_text(entity_id),
            "event_type": event_type,
            "from_value": from_value,
            "to_value": to_value,
            "actor_role": actor_role or self._actor_role,
            "evidence_ref": self._scrub_text(evidence_ref) if evidence_ref else None,
            "payload": self.sanitize_payload(payload or {}),
        }
        return self._writer.append(body)

    def record_egress_block(self, *, event: str, destination: str) -> str:
        """Operational record for one blocked network egress attempt (§B7)."""
        host, separator, port_text = destination.rpartition(":")
        payload: dict[str, object]
        if separator and port_text.isdigit():
            payload = {
                "audit_event": event,
                "destination_host_pseudonym": pseudonymize(host, self._key),
                "destination_port": int(port_text),
            }
        else:
            payload = {
                "audit_event": event,
                "destination_host_pseudonym": pseudonymize(destination, self._key),
                "destination_port": None,
            }
        return self.record(
            entity="network_guard",
            entity_id=f"egress:{event}",
            event_type="EGRESS_BLOCKED",
            to_value="BLOCKED",
            payload=payload,
            actor_role="NETWORK_GUARD",
        )

    # -- diagnostic/error stream -------------------------------------------------

    def record_exception(self, exc: BaseException, *, context: str = "") -> str:
        """Append a sanitized diagnostic record for *exc* (§50B line 1773)."""
        if self._diagnostic_writer is None:
            self._diagnostic_writer = _SealedJsonlWriter(
                self._directory / f"{DIAGNOSTIC_STREAM}-audit.jsonl", max_bytes=self._max_bytes
            )
        tb = "".join(_format_exception(exc))
        body: dict[str, object] = {
            "at": utc_now_iso(),
            "entity": "diagnostics",
            "entity_id": self._scrub_text(context or exc.__class__.__name__),
            "event_type": "EXCEPTION",
            "from_value": None,
            "to_value": exc.__class__.__name__,
            "actor_role": self._actor_role,
            "evidence_ref": None,
            "payload": {
                "message": self._scrub_text(str(exc), limit=2000),
                "traceback": self._scrub_text(tb),
            },
        }
        return self._diagnostic_writer.append(body)

    # -- redaction -----------------------------------------------------------------

    def sanitize_payload(self, payload: Mapping[str, object]) -> dict[str, object]:
        """Deep PHI redaction of a payload (identifiers, paths, literals)."""
        sanitized = _sanitize_value(payload, self._key, self._registered)
        assert isinstance(sanitized, dict)
        return sanitized

    def _scrub_text(self, text: str, *, limit: int = _MAX_SCRUBBED_TEXT) -> str:
        scrubbed = redact_absolute_paths(text, self._key)
        for raw, label in sorted(self._registered, key=lambda item: -len(item[0])):
            token = f"<{label}:{pseudonymize(raw, self._key)}>"
            scrubbed = scrubbed.replace(raw, token)
        return scrubbed[:limit]

    # -- readback / verification ------------------------------------------------------

    def iter_records(self) -> list[dict[str, object]]:
        """All sealed records of THIS sink's stream, in append order."""
        return list(iter_stream_records(self._directory, self._stream))

    def verify(self) -> tuple[bool, list[str]]:
        """Verify the chain seal of this sink's stream."""
        return verify_stream(self._directory, self._stream)


# ---------------------------------------------------------------------------
# Sanitization internals
# ---------------------------------------------------------------------------


def _sanitize_value(
    value: object, key: bytes, registered: Sequence[tuple[str, str]]
) -> object:
    if isinstance(value, Mapping):
        out: dict[str, object] = {}
        for raw_key, item in value.items():
            text_key = str(raw_key)
            if text_key.casefold() in DIRECT_IDENTIFIER_KEYS:
                out[text_key] = pseudonymize(str(item), key)
            else:
                out[text_key] = _sanitize_value(item, key, registered)
        return out
    if isinstance(value, (list, tuple)):
        return [_sanitize_value(item, key, registered) for item in value]
    if isinstance(value, str):
        scrubbed = redact_absolute_paths(value, key)
        for raw, label in sorted(registered, key=lambda item: -len(item[0])):
            scrubbed = scrubbed.replace(raw, f"<{label}:{pseudonymize(raw, key)}>")
        return scrubbed
    return value


def _format_exception(exc: BaseException) -> list[str]:
    import traceback

    return traceback.format_exception(type(exc), exc, exc.__traceback__)


# ---------------------------------------------------------------------------
# Sealed JSONL storage (shared by both streams)
# ---------------------------------------------------------------------------

_SEAL_FIELD: Final[str] = "seal"


class _SealedJsonlWriter:
    """Append-only JSONL writer with per-record SHA-256 chain seals."""

    def __init__(self, path: Path, *, max_bytes: int) -> None:
        self.path = Path(path)
        self._max_bytes = max_bytes
        self._seq = 0
        self._last_seal = GENESIS_SEAL
        self._resume()

    def _segments(self) -> list[Path]:
        return _segments_of(self.path)

    def _iter_existing_lines(self) -> Iterator[str]:
        for segment in (*self._segments(), self.path):
            if not segment.exists():
                continue
            for line in segment.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    yield line

    def _resume(self) -> None:
        """Continue an existing chain so separate sink instances interoperate."""
        last_line: str | None = None
        for line in self._iter_existing_lines():
            last_line = line
        if last_line is None:
            return
        try:
            record = json.loads(last_line)
            self._seq = int(record["seq"]) + 1
            self._last_seal = str(record[_SEAL_FIELD])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            # An unparseable tail leaves the chain unverifiable; fail closed by
            # refusing to extend it (caller sees the error on the next append).
            raise ValueError(f"cannot resume sealed stream {self.path}: corrupt tail") from None

    def _rotate_if_needed(self, incoming_size: int) -> None:
        if not self.path.exists():
            return
        if self.path.stat().st_size + incoming_size <= self._max_bytes:
            return
        next_index = len(self._segments()) + 1
        self.path.rename(self.path.with_name(f"{self.path.name}.{next_index}"))

    def append(self, body: Mapping[str, object]) -> str:
        record: dict[str, object] = {
            "seq": self._seq,
            "prev_seal": self._last_seal,
            **body,
        }
        seal_input = self._last_seal + canonical_json(record)
        seal = hashlib.sha256(seal_input.encode("utf-8")).hexdigest()
        record[_SEAL_FIELD] = seal
        line = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        self._rotate_if_needed(len(line) + 1)
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(line + "\n")
        self._seq += 1
        self._last_seal = seal
        return seal


def _segments_of(base: Path) -> list[Path]:
    """Rotated segments of *base*, oldest first (``name.jsonl.N``, N ascending)."""
    rotated: list[tuple[int, Path]] = []
    for candidate in base.parent.glob(base.name + ".*"):
        suffix = candidate.name[len(base.name) + 1 :]
        if suffix.isdigit():
            rotated.append((int(suffix), candidate))
    return [p for _, p in sorted(rotated)]


def iter_stream_records(directory: Path, stream: str) -> Iterator[dict[str, object]]:
    """Yield every sealed record of *stream* in chronological append order."""
    base = Path(directory) / f"{stream}-audit.jsonl"
    for segment in (*_segments_of(base), base):
        if not segment.exists():
            continue
        for line in segment.read_text(encoding="utf-8").splitlines():
            if line.strip():
                yield dict(json.loads(line))


def verify_stream(directory: Path, stream: str) -> tuple[bool, list[str]]:
    """Recompute the full chain of *stream*; ``(ok, problems)``."""
    problems: list[str] = []
    previous = GENESIS_SEAL
    expected_seq = 0
    count = 0
    for record in iter_stream_records(directory, stream):
        where = f"{stream}[{count}]"
        seq = record.get("seq")
        if seq != expected_seq:
            problems.append(f"{where}: seq {seq!r} != expected {expected_seq}")
        prev = record.get("prev_seal")
        if prev != previous:
            problems.append(f"{where}: prev_seal mismatch (chain break)")
        seal = record.get(_SEAL_FIELD)
        body = {k: v for k, v in record.items() if k != _SEAL_FIELD}
        expected = hashlib.sha256(
            (previous + canonical_json(body)).encode("utf-8")
        ).hexdigest()
        if seal != expected:
            problems.append(f"{where}: seal verification failed (tampered or corrupted)")
        previous = str(seal) if isinstance(seal, str) else previous
        expected_seq += 1
        count += 1
    return (not problems, problems)


__all__ = [
    "DEFAULT_MAX_BYTES",
    "DIAGNOSTIC_STREAM",
    "DIRECT_IDENTIFIER_KEYS",
    "GENESIS_SEAL",
    "OPERATIONAL_STREAM",
    "PhiSafeAuditSink",
    "iter_stream_records",
    "pseudonymize",
    "redact_absolute_paths",
    "verify_stream",
]
