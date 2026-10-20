"""PluginRegistryMount — plugin architecture baseline (WS02_DESIGN.md §B6).

A plugin is a directory carrying a CLOSED-SCHEMA ``plugin.toml`` manifest
(id, version, kind, entrypoint, capabilities, claims, network_required) plus
its Python module. Mounting validates the manifest BEFORE any code runs:

- unknown manifest keys are rejected (closed schema, CSI-001 analogy);
- ``network_required`` must exist and be ``false`` in Phase 0;
- the declared producer kind is checked against ``core.artifacts.
  PRODUCER_MATRIX`` for every declared capability — an LLM plugin may only
  declare REPORT_DRAFT (Rule D);
- every claim id must resolve to a locked row of
  ``docs/phase-minus-1/CLAIM_REGISTRY.md`` (F1 binding).

A non-conforming plugin is REJECTED WITH REASONS and never loaded. Accepted
plugins are registered into the shared EventLog using the standard registry
vocabulary (seed + REGISTERED) — no second state machine (Rule E intake).
"""

from __future__ import annotations

import hashlib
import re
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Final

from core.artifacts import PRODUCER_MATRIX
from core.enums import ArtifactKind, ProducerType
from core.errors import CanonicalValidationError
from core.events import EventLog

MANIFEST_NAME: Final[str] = "plugin.toml"
PLUGIN_MANIFEST_SCHEMA: Final[str] = "ccta.plugin.manifest.v1"

ALLOWED_MANIFEST_KEYS: Final[frozenset[str]] = frozenset(
    {
        "id",
        "version",
        "kind",
        "entrypoint",
        "capabilities",
        "claims",
        "network_required",
        # optional, but must carry the exact manifest schema URI when present
        "schema",
    }
)

#: Locked claim rows of docs/phase-minus-1/CLAIM_REGISTRY.md (Phase -1).
KNOWN_CLAIMS: Final[dict[str, str]] = {
    "CLM-QC-001": "v0-draft",
    "CLM-VIZ-001": "v0-draft",
    "CLM-SEG-001": "v0-draft",
    "CLM-MEAS-001": "v0-draft",
    "CLM-REP-001": "v0-draft",
    "CLM-EXP-001": "v0-draft",
}

_KEBAB_RE: Final[re.Pattern[str]] = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
_ENTRYPOINT_RE: Final[re.Pattern[str]] = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_.]*:[A-Za-z_][A-Za-z0-9_]*$"
)


class PluginRejectedError(CanonicalValidationError):
    """Manifest or code failed mount validation; the plugin was NOT loaded."""


@dataclass(frozen=True)
class MountedPlugin:
    """A validated, loaded plugin registration."""

    plugin_id: str
    version: str
    kind: ProducerType
    entrypoint: str
    capabilities: tuple[ArtifactKind, ...]
    claims: tuple[str, ...]
    manifest_sha256: str


@dataclass(frozen=True)
class PluginRejection:
    """One rejected plugin with its fail-closed reasons."""

    directory: Path
    reasons: tuple[str, ...]


class _PluginImportScope:
    """Temporarily put a plugin dir on sys.path while importing its module."""

    def __init__(self, plugin_dir: Path) -> None:
        self._dir = plugin_dir

    def __enter__(self) -> None:
        self._added = str(self._dir) not in sys.path
        if self._added:
            sys.path.insert(0, str(self._dir))

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._added:
            try:
                sys.path.remove(str(self._dir))
            except ValueError:
                pass
        return None


class PluginRegistryMount:
    """Validates and mounts plugins; rejects everything else with reasons."""

    def __init__(self, log: EventLog | None = None) -> None:
        self._log = log
        self.mounted: dict[str, MountedPlugin] = {}
        self.rejections: list[PluginRejection] = []

    # -- public API ---------------------------------------------------------------

    def mount(self, plugin_dir: Path) -> MountedPlugin:
        """Validate *plugin_dir* and load it; raises PluginRejectedError."""
        reasons: list[str] = []
        directory = Path(plugin_dir)
        manifest_path = directory / MANIFEST_NAME
        if not manifest_path.is_file():
            raise PluginRejectedError([f"{MANIFEST_NAME} not found in {directory.name}"])
        raw_bytes = manifest_path.read_bytes()
        manifest_sha256 = hashlib.sha256(raw_bytes).hexdigest()
        try:
            manifest = tomllib.loads(raw_bytes.decode("utf-8"))
        except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
            raise PluginRejectedError([f"{MANIFEST_NAME}: unreadable TOML ({exc})"]) from None

        validated = _validate_manifest(manifest, reasons)
        if reasons:
            raise PluginRejectedError(reasons)
        assert validated is not None

        plugin_id, version, kind, entrypoint, capabilities, claims = validated
        callable_obj = _load_entrypoint(directory, entrypoint, reasons)
        if callable_obj is None:
            raise PluginRejectedError(reasons)

        mounted = MountedPlugin(
            plugin_id=plugin_id,
            version=version,
            kind=kind,
            entrypoint=entrypoint,
            capabilities=capabilities,
            claims=claims,
            manifest_sha256=manifest_sha256,
        )
        self.mounted[plugin_id] = mounted
        if self._log is not None and mounted.plugin_id not in {
            entity_id for (_entity, entity_id), _fields in self._log.initial_states()
        }:
            self._log.seed(
                entity="plugin",
                entity_id=plugin_id,
                initial_fields={"registry_status": "EXPERIMENTAL"},
            )
            from core.events import StateEvent, utc_now_iso

            self._log.append_event(
                StateEvent(
                    entity="plugin",
                    entity_id=plugin_id,
                    event_type="REGISTERED",
                    from_value="EXPERIMENTAL",
                    to_value="REGISTERED",
                    actor_role="plugin-mount",
                    at=utc_now_iso(),
                    evidence_ref=f"{entrypoint}@{version}",
                )
            )
        return mounted

    def mount_all(self, plugins_root: Path) -> tuple[MountedPlugin, ...]:
        """Mount every subdirectory; rejections are recorded, never raised en masse."""
        mounted: list[MountedPlugin] = []
        for candidate in sorted(Path(plugins_root).iterdir()):
            if not candidate.is_dir():
                continue
            try:
                mounted.append(self.mount(candidate))
            except PluginRejectedError as exc:
                self.rejections.append(
                    PluginRejection(directory=candidate, reasons=exc.violations)
                )
        return tuple(mounted)


# ---------------------------------------------------------------------------
# manifest validation (closed schema, PRODUCER_MATRIX, claims)
# ---------------------------------------------------------------------------


def _validate_manifest(
    manifest: dict[str, object], reasons: list[str]
) -> tuple[
    str, str, ProducerType, str, tuple[ArtifactKind, ...], tuple[str, ...]
] | None:
    unknown = sorted(set(manifest) - ALLOWED_MANIFEST_KEYS)
    if unknown:
        reasons.append(f"unknown manifest key(s): {unknown}")
    schema_value = manifest.get("schema")
    if "schema" in manifest and schema_value != PLUGIN_MANIFEST_SCHEMA:
        reasons.append(
            f"manifest schema must be {PLUGIN_MANIFEST_SCHEMA!r}, got {schema_value!r}"
        )
    for key in ("id", "version", "kind", "entrypoint"):
        value = manifest.get(key)
        if not isinstance(value, str) or not value.strip():
            reasons.append(f"{key} must be a non-empty string")

    plugin_id = manifest.get("id")
    if isinstance(plugin_id, str) and not _KEBAB_RE.fullmatch(plugin_id):
        reasons.append(f"id must be a kebab-case slug, got {plugin_id!r}")

    network_required = manifest.get("network_required")
    if "network_required" not in manifest:
        reasons.append("network_required is MANDATORY in Phase 0")
    elif network_required is not False:
        reasons.append(
            "network_required must be exactly false in Phase 0 "
            f"(got {network_required!r})"
        )

    kind_value = manifest.get("kind")
    kind: ProducerType | None = None
    if isinstance(kind_value, str):
        try:
            kind = ProducerType(kind_value)
        except ValueError:
            reasons.append(f"unknown producer kind {kind_value!r}")
    elif kind_value is not None:
        reasons.append("kind must be a string")

    raw_caps = manifest.get("capabilities")
    capabilities: list[ArtifactKind] = []
    if not isinstance(raw_caps, list) or not raw_caps:
        reasons.append("capabilities must be a non-empty array of artifact kinds")
    else:
        for item in raw_caps:
            try:
                capability = ArtifactKind(str(item))
            except ValueError:
                reasons.append(f"unknown capability kind {item!r}")
                continue
            capabilities.append(capability)

    raw_claims = manifest.get("claims")
    claims: list[str] = []
    if not isinstance(raw_claims, list) or not raw_claims:
        reasons.append("claims must be a non-empty array of claim ids (F1 binding)")
    else:
        for item in raw_claims:
            claim_id = str(item)
            if claim_id not in KNOWN_CLAIMS:
                reasons.append(f"claim {claim_id!r} is not in the CLAIM_REGISTRY")
            else:
                claims.append(claim_id)

    entrypoint_value = manifest.get("entrypoint")
    entrypoint: str | None = None
    if isinstance(entrypoint_value, str):
        if not _ENTRYPOINT_RE.fullmatch(entrypoint_value):
            reasons.append(
                f"entrypoint must be 'module:callable', got {entrypoint_value!r}"
            )
        else:
            entrypoint = entrypoint_value

    if kind is not None and capabilities:
        allowed_kinds = [
            capability
            for capability in capabilities
            if kind in PRODUCER_MATRIX.get(capability, frozenset())
        ]
        for capability in capabilities:
            allowed = PRODUCER_MATRIX.get(capability, frozenset())
            if kind not in allowed:
                reasons.append(
                    f"PRODUCER_MATRIX violation: kind={kind.value} may not produce "
                    f"{capability.value} (allowed: "
                    f"{sorted(p.value for p in allowed)})"
                )
        if kind == ProducerType.LLM:
            forbidden = [c for c in capabilities if c != ArtifactKind.REPORT_DRAFT]
            if forbidden:
                reasons.append(
                    "Rule D: LLM plugins may only produce REPORT_DRAFT "
                    f"(forbidden here: {[c.value for c in forbidden]})"
                )
        elif not allowed_kinds and not reasons:
            reasons.append("no declared capability matches the producer matrix row")

    if reasons:
        return None
    assert plugin_id and isinstance(plugin_id, str)
    assert isinstance(manifest.get("version"), str)
    assert kind is not None and entrypoint is not None
    return (
        plugin_id,
        str(manifest["version"]),
        kind,
        entrypoint,
        tuple(dict.fromkeys(capabilities)),
        tuple(dict.fromkeys(claims)),
    )


def _load_entrypoint(directory: Path, entrypoint: str, reasons: list[str]) -> object | None:
    module_name, _, attribute = entrypoint.partition(":")
    import importlib

    try:
        with _PluginImportScope(directory):
            module = importlib.import_module(module_name)
        return getattr(module, attribute)
    except Exception as exc:  # noqa: BLE001 - any failure means rejection
        reasons.append(f"entrypoint {entrypoint!r} failed to load ({exc.__class__.__name__})")
        return None


__all__ = [
    "ALLOWED_MANIFEST_KEYS",
    "KNOWN_CLAIMS",
    "MANIFEST_NAME",
    "MountedPlugin",
    "PLUGIN_MANIFEST_SCHEMA",
    "PluginRejection",
    "PluginRegistryMount",
    "PluginRejectedError",
]
