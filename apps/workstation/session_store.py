"""DerivedStateStoreManager: total-or-fail session persistence (§B3).

Persists/restores a complete session — study, series manifests, volume and
artifact records, payload documents, EventLog journal (seeds + events),
provenance records — into ``<workspace>/session/`` plus the content-addressed
blob tree.

Restore is TOTAL-OR-FAIL: every file listed in ``session/manifest.json`` is
digest-verified BEFORE anything is decoded, undeclared extra files are a
failure too, and any mismatch raises :class:`SessionIntegrityError` listing
every problem — partial state is never returned (no silent fallbacks).

The tagged-JSON encoder below is deliberately allowlist-closed: only frozen
canonical dataclasses/enums of ``core.*`` can be rehydrated, so a tampered
session document cannot smuggle arbitrary class construction.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, fields, is_dataclass, replace
from pathlib import Path
from typing import Any, Final

from core.artifacts import (
    ClaimBinding,
    DerivedArtifact,
    GeometryBase,
    Lineage,
    PayloadRef,
    Producer,
    QualityBlock,
    ReviewBlock,
    ReviewTransition,
)
from core.coordinates import CoordinateTransform, FrameSpec
from core.digest import canonical_digest
from core.enums import (
    ActorType,
    ArtifactKind,
    ArtifactStatus,
    Confidence,
    ContrastClass,
    ContrastPresence,
    CoordinateSystem,
    ExamType,
    Handedness,
    ImportSource,
    PrivacyClass,
    ProducerType,
    ProvenanceEventType,
    QualityGrade,
    ReleaseLane,
    ReviewState,
    SeriesClass,
    SpaceKind,
    Units,
    VolumeKind,
)
from core.errors import CanonicalValidationError, ClosedSchemaError
from core.events import EventLog, StateEvent, event_to_dict, utc_now_iso
from core.provenance import (
    Actor,
    BlobRef,
    Environment,
    HumanEdit,
    InputRef,
    NotApplicable,
    ProvenanceRecord,
)
from core.study import (
    AcquisitionPhase,
    AnnotationEvent,
    ExamClassification,
    ImportInfo,
    InstanceEntry,
    InstanceManifest,
    OriginalDicomRef,
    Series,
    SeriesClassification,
    Study,
)
from core.volume import Volume, VolumeBlobRef

MANIFEST_SCHEMA: Final[str] = "ccta.session.manifest.v1"
MANIFEST_NAME: Final[str] = "manifest.json"


# ---------------------------------------------------------------------------
# Workspace layout + typed abstention record (shared by all §B modules)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WorkstationPaths:
    """Canonical directory layout under an explicit workspace root (§B1)."""

    root: Path
    session: Path
    blobs: Path
    logs: Path
    exports: Path

    @classmethod
    def create(cls, root: Path) -> WorkstationPaths:
        resolved = Path(root)
        paths = cls(
            root=resolved,
            session=resolved / "session",
            blobs=resolved / "blobs",
            logs=resolved / "logs",
            exports=resolved / "exports",
        )
        for directory in (paths.session, paths.blobs, paths.logs, paths.exports):
            directory.mkdir(parents=True, exist_ok=True)
        return paths


@dataclass(frozen=True)
class SkippedInstance:
    """One input file refused at import, with its reason (no silent skips)."""

    path: str
    reason: str


class SessionMissingError(CanonicalValidationError):
    """No saved session manifest exists in the workspace."""


class SessionIntegrityError(CanonicalValidationError):
    """Saved session failed total-or-fail digest verification."""


# ---------------------------------------------------------------------------
# Tagged canonical JSON codec (allowlist-closed)
# ---------------------------------------------------------------------------

_TAG_DATACLASS: Final[str] = "__d__"
_TAG_ENUM: Final[str] = "__e__"
_TAG_TUPLE: Final[str] = "__t__"
_TAG_NOT_APPLICABLE: Final[str] = "__na__"

_CLASS_ALLOWLIST: Final[dict[str, type]] = {
    f"{cls.__module__}.{cls.__qualname__}": cls
    for cls in (
        # core.study
        ImportInfo,
        ExamClassification,
        AnnotationEvent,
        Study,
        AcquisitionPhase,
        InstanceEntry,
        InstanceManifest,
        SeriesClassification,
        OriginalDicomRef,
        Series,
        # core.coordinates
        FrameSpec,
        CoordinateTransform,
        # core.volume
        VolumeBlobRef,
        Volume,
        # core.artifacts
        PayloadRef,
        GeometryBase,
        Lineage,
        ClaimBinding,
        Producer,
        ReviewTransition,
        ReviewBlock,
        QualityBlock,
        DerivedArtifact,
        # core.provenance
        Actor,
        BlobRef,
        HumanEdit,
        InputRef,
        Environment,
        ProvenanceRecord,
        # workstation
        SkippedInstance,
    )
}

_ENUM_ALLOWLIST: Final[dict[str, type]] = {
    f"{cls.__module__}.{cls.__qualname__}": cls
    for cls in (
        PrivacyClass,
        ImportSource,
        ExamType,
        ContrastPresence,
        SeriesClass,
        ContrastClass,
        VolumeKind,
        ArtifactKind,
        ArtifactStatus,
        ProducerType,
        ActorType,
        ProvenanceEventType,
        ReleaseLane,
        ReviewState,
        QualityGrade,
        Confidence,
        CoordinateSystem,
        Handedness,
        Units,
        SpaceKind,
    )
}


def encode_value(value: object) -> object:
    """Encode a canonical object graph into JSON-ready tagged structures."""
    import enum

    if isinstance(value, NotApplicable):
        return {
            _TAG_NOT_APPLICABLE: {
                "reason": value.reason,
                "value": encode_value(value.value),
            }
        }
    if isinstance(value, enum.Enum):
        qualname = f"{type(value).__module__}.{type(value).__qualname__}"
        if qualname not in _ENUM_ALLOWLIST:
            raise ClosedSchemaError(f"enum not allowlisted for session encoding: {qualname}")
        return {_TAG_ENUM: f"{qualname}:{value.value}"}
    if isinstance(value, dict):
        return {str(key): encode_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return {_TAG_TUPLE: [encode_value(item) for item in value]}
    if is_dataclass(value) and not isinstance(value, type):
        qualname = f"{type(value).__module__}.{type(value).__qualname__}"
        if qualname not in _CLASS_ALLOWLIST:
            raise ClosedSchemaError(
                f"type not allowlisted for session encoding: {qualname}"
            )
        return {
            _TAG_DATACLASS: qualname,
            "f": {f.name: encode_value(getattr(value, f.name)) for f in fields(value)},
        }
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"not session-serializable: {type(value)!r}")


def decode_value(encoded: object) -> object:
    """Rehydrate encoded structures; unknown tags/classes fail closed."""
    if isinstance(encoded, dict):
        if _TAG_TUPLE in encoded:
            items = encoded[_TAG_TUPLE]
            if not isinstance(items, list):
                raise ClosedSchemaError("malformed tuple tag")
            return tuple(decode_value(item) for item in items)
        if _TAG_NOT_APPLICABLE in encoded:
            body = encoded[_TAG_NOT_APPLICABLE]
            if not isinstance(body, dict) or "reason" not in body:
                raise ClosedSchemaError("malformed NotApplicable encoding")
            return NotApplicable(
                value=decode_value(body.get("value")), reason=str(body["reason"])
            )
        if _TAG_ENUM in encoded:
            qualified = encoded[_TAG_ENUM]
            if not isinstance(qualified, str) or ":" not in qualified:
                raise ClosedSchemaError("malformed enum encoding")
            qualname, _, raw_value = qualified.rpartition(":")
            cls = _ENUM_ALLOWLIST.get(qualname)
            if cls is None:
                raise ClosedSchemaError(f"unknown enum in session document: {qualname!r}")
            return cls(raw_value)
        if _TAG_DATACLASS in encoded:
            qualname = encoded[_TAG_DATACLASS]
            cls = _CLASS_ALLOWLIST.get(qualname)
            if cls is None or not isinstance(cls, type):
                raise ClosedSchemaError(f"unknown class in session document: {qualname!r}")
            field_map = encoded["f"]
            if not isinstance(field_map, dict):
                raise ClosedSchemaError("malformed dataclass encoding")
            kwargs = {name: decode_value(item) for name, item in field_map.items()}
            return cls(**kwargs)
        return {key: decode_value(item) for key, item in encoded.items()}
    if isinstance(encoded, list):
        return [decode_value(item) for item in encoded]
    return encoded


def _require_document(encoded: object, where: str) -> dict[str, object]:
    if not isinstance(encoded, dict):
        raise SessionIntegrityError([f"{where}: expected a JSON object"])
    return encoded


def _decode_typed[T](encoded: object, target: type[T], where: str) -> T:
    decoded = decode_value(encoded)
    if not isinstance(decoded, target):
        raise SessionIntegrityError(
            [f"{where}: unexpected decoded type {type(decoded)!r}"]
        )
    return decoded


# ---------------------------------------------------------------------------
# Snapshot + restored session records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SessionSnapshot:
    """Everything that constitutes one persisted workstation session."""

    study: Study
    series: tuple[Series, ...]
    volumes: tuple[Volume, ...]
    artifacts: tuple[DerivedArtifact, ...]
    payload_documents: dict[str, dict[str, object]]
    provenance: tuple[ProvenanceRecord, ...]
    log: EventLog
    skipped_instances: tuple[SkippedInstance, ...]
    # MPR pixel payloads are normally written by CanonicalVolumeLoader before
    # save.  This optional handoff lets a caller provide a newly produced
    # content-addressed blob atomically with the snapshot as well.
    pixel_blobs: dict[str, bytes] = field(default_factory=dict)


@dataclass(frozen=True)
class RestoredSession:
    """A fully verified, rehydrated session (reopen output)."""

    paths: WorkstationPaths
    study: Study
    series: tuple[Series, ...]
    volumes: tuple[Volume, ...]
    artifacts: tuple[DerivedArtifact, ...]
    payload_documents: dict[str, dict[str, object]]
    provenance: tuple[ProvenanceRecord, ...]
    log: EventLog
    skipped_instances: tuple[SkippedInstance, ...]
    projections: dict[str, dict[str, str]]
    session_digest: str
    pixel_blobs: dict[str, bytes] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# DerivedStateStoreManager
# ---------------------------------------------------------------------------


def _write_json(path: Path, document: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(document, sort_keys=True, indent=2, ensure_ascii=False)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    tmp.replace(path)


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


class DerivedStateStoreManager:
    """Save/reopen of complete sessions with total-or-fail integrity (§B3)."""

    def __init__(self, paths: WorkstationPaths) -> None:
        self._paths = paths

    # -- save --------------------------------------------------------------------

    def save(self, snapshot: SessionSnapshot) -> str:
        """Persist the snapshot; returns the content-derived session_digest."""
        session_dir = self._paths.session
        if snapshot.pixel_blobs:
            from apps.workstation.volume_loader import CanonicalVolumeLoader

            blob_loader = CanonicalVolumeLoader(self._paths.blobs)
            for expected_digest, data in snapshot.pixel_blobs.items():
                if not isinstance(data, bytes):
                    raise CanonicalValidationError(
                        f"pixel blob {expected_digest}: payload must be bytes"
                    )
                stored = blob_loader.write_blob(data)
                if stored.sha256 != expected_digest:
                    raise CanonicalValidationError(
                        f"pixel blob key {expected_digest} does not match content "
                        f"{stored.sha256}"
                    )
        self._validate_mpr_content(snapshot.artifacts, snapshot.payload_documents)
        documents: dict[str, object] = {
            "study.json": {
                "schema": "ccta.session.study.v1",
                "study": encode_value(snapshot.study),
            },
            "series.json": {
                "schema": "ccta.session.series.v1",
                "series": [encode_value(item) for item in snapshot.series],
            },
            "volumes.json": {
                "schema": "ccta.session.volumes.v1",
                "volumes": [encode_value(item) for item in snapshot.volumes],
            },
            "artifacts.json": {
                "schema": "ccta.session.artifacts.v1",
                "artifacts": [encode_value(item) for item in snapshot.artifacts],
                # Payload documents are ALREADY JSON-ready tagged structures;
                # they must persist byte-stable so their sealed digests
                # (artifact.payload.blob_sha256) verify on reopen.
                "payload_documents": {
                    str(key): dict(document)
                    for key, document in snapshot.payload_documents.items()
                },
            },
            "provenance.json": {
                "schema": "ccta.session.provenance.v1",
                "records": [encode_value(record) for record in snapshot.provenance],
            },
            "journal_seeds.json": {
                "schema": "ccta.session.journal_seeds.v1",
                "seeds": [
                    [entity, entity_id, dict(initial)]
                    for (entity, entity_id), initial in snapshot.log.initial_states()
                ],
            },
            "journal_events.json": {
                "schema": "ccta.session.journal_events.v1",
                "events": [event_to_dict(event) for event in snapshot.log.events],
            },
            "projections.json": {
                "schema": "ccta.session.projections.v1",
                "projections": self._capture_projections(snapshot.log),
            },
            "skipped.json": {
                "schema": "ccta.session.skipped.v1",
                "skipped": [encode_value(item) for item in snapshot.skipped_instances],
            },
        }
        for name, document in documents.items():
            _write_json(session_dir / name, document)

        files: dict[str, dict[str, object]] = {}
        for base in (self._paths.blobs, session_dir):
            for file_path in sorted(base.rglob("*")):
                if not file_path.is_file() or file_path.suffix == ".tmp":
                    continue
                relative = file_path.relative_to(self._paths.root).as_posix()
                if relative == f"session/{MANIFEST_NAME}":
                    continue
                files[relative] = {
                    "sha256": _file_sha256(file_path),
                    "bytes": file_path.stat().st_size,
                }
        undeclared = sorted(
            f"session/{name}" for name in documents if f"session/{name}" not in files
        )
        if undeclared:
            raise CanonicalValidationError(f"session files failed to persist: {undeclared}")
        session_digest = canonical_digest({"files": {k: v["sha256"] for k, v in files.items()}})
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "saved_at": utc_now_iso(),
            "session_digest": session_digest,
            "files": dict(sorted(files.items())),
        }
        _write_json(session_dir / MANIFEST_NAME, manifest)
        return session_digest

    def _capture_projections(self, log: EventLog) -> dict[str, dict[str, object]]:
        projections: dict[str, dict[str, object]] = {}
        for (entity, entity_id), _initial in log.initial_states():
            projection = log.project_state(entity_id, entity=entity)
            projections[entity_id] = {
                "entity": entity,
                "fields": projection.to_dict(),
            }
        return projections

    # -- reopen ---------------------------------------------------------------------

    def reopen(self) -> RestoredSession:
        """Total-or-fail restore; any mismatch aborts before decoding."""
        problems: list[str] = []
        manifest_path = self._paths.session / MANIFEST_NAME
        if not manifest_path.exists():
            raise SessionMissingError(["no saved session manifest in workspace"])
        try:
            manifest = _require_document(_read_json(manifest_path), MANIFEST_NAME)
        except json.JSONDecodeError as exc:
            raise SessionIntegrityError([f"manifest unreadable: {exc}"]) from None
        entries = manifest.get("files")
        if not isinstance(entries, dict):
            raise SessionIntegrityError(["manifest has no files map"])

        declared_files: dict[str, dict[str, object]] = {}
        for relative, meta in entries.items():
            if not isinstance(meta, dict):
                problems.append(f"{relative}: malformed manifest entry")
                continue
            declared_files[str(relative)] = meta

        actual: set[str] = set()
        for base in (self._paths.blobs, self._paths.session):
            for file_path in sorted(base.rglob("*")):
                if not file_path.is_file() or file_path.suffix == ".tmp":
                    continue
                relative = file_path.relative_to(self._paths.root).as_posix()
                if relative == f"session/{MANIFEST_NAME}":
                    continue
                actual.add(relative)

        for relative in sorted(set(declared_files) | actual):
            full = self._paths.root / relative
            meta = declared_files.get(relative)
            if meta is None:
                problems.append(f"{relative}: undeclared extra file on disk")
                continue
            if not full.exists():
                problems.append(f"{relative}: listed in manifest but missing")
                continue
            raw_size = meta.get("bytes", -1)
            size = raw_size if isinstance(raw_size, int) else -1
            expected_sha = str(meta.get("sha256", ""))
            if full.stat().st_size != size:
                problems.append(f"{relative}: size {full.stat().st_size} != manifest {size}")
                continue
            actual_sha = _file_sha256(full)
            if actual_sha != expected_sha:
                problems.append(f"{relative}: sha256 mismatch (tampered or corrupted)")
                continue

        recomputed = canonical_digest(
            {
                "files": {
                    relative: str(meta.get("sha256"))
                    for relative, meta in sorted(declared_files.items())
                    if isinstance(meta, dict) and "sha256" in meta
                }
            }
        )
        recorded_digest = str(manifest.get("session_digest", ""))
        if recorded_digest and recomputed != recorded_digest:
            problems.append("session_digest does not match the manifest contents")
        if problems:
            raise SessionIntegrityError(problems)

        session = self._decode_all()
        return replace(session, session_digest=recorded_digest or recomputed)

    # -- decode helpers ---------------------------------------------------------------

    def _load_document(self, name: str) -> dict[str, object]:
        return _require_document(_read_json(self._paths.session / name), name)

    def _decode_all(self) -> RestoredSession:
        study_doc = self._load_document("study.json")
        series_doc = self._load_document("series.json")
        volumes_doc = self._load_document("volumes.json")
        artifacts_doc = self._load_document("artifacts.json")
        provenance_doc = self._load_document("provenance.json")
        seeds_doc = self._load_document("journal_seeds.json")
        events_doc = self._load_document("journal_events.json")
        projections_doc = self._load_document("projections.json")
        skipped_doc = self._load_document("skipped.json")

        study = _decode_typed(study_doc["study"], Study, "study.json")

        series_list = series_doc.get("series")
        series = tuple(
            _decode_typed(item, Series, "series.json")
            for item in (_ensure_list(series_list, "series.json:series"))
        )
        volume_items = _ensure_list(volumes_doc.get("volumes"), "volumes.json:volumes")
        volumes = tuple(
            _decode_typed(item, Volume, "volumes.json") for item in volume_items
        )
        artifact_items = _ensure_list(
            artifacts_doc.get("artifacts"), "artifacts.json:artifacts"
        )
        artifacts = tuple(
            _decode_typed(item, DerivedArtifact, "artifacts.json")
            for item in artifact_items
        )
        raw_documents = artifacts_doc.get("payload_documents")
        payload_documents: dict[str, dict[str, object]] = {}
        if isinstance(raw_documents, dict):
            for key, document in raw_documents.items():
                if isinstance(document, dict):
                    payload_documents[str(key)] = document
        prov_items = _ensure_list(provenance_doc.get("records"), "provenance.json:records")
        provenance = tuple(
            _decode_typed(item, ProvenanceRecord, "provenance.json")
            for item in prov_items
        )

        log = EventLog()
        seed_rows = _ensure_list(seeds_doc.get("seeds"), "journal_seeds.json:seeds")
        for row in seed_rows:
            if not isinstance(row, list) or len(row) != 3:
                raise SessionIntegrityError(["journal_seeds.json: malformed seed row"])
            entity, entity_id, initial = row
            if (
                not isinstance(entity, str)
                or not isinstance(entity_id, str)
                or not isinstance(initial, dict)
            ):
                raise SessionIntegrityError(["journal_seeds.json: malformed seed types"])
            log.seed(entity, entity_id, {str(k): str(v) for k, v in initial.items()})
        event_rows = _ensure_list(events_doc.get("events"), "journal_events.json:events")
        replayed_events: list[StateEvent] = []
        for row in event_rows:
            if not isinstance(row, dict):
                raise SessionIntegrityError(["journal_events.json: malformed event row"])
            event = StateEvent(**row)
            log.append_event(event)
            replayed_events.append(event)

        saved_projections = projections_doc.get("projections")
        projections: dict[str, dict[str, str]] = {}
        if isinstance(saved_projections, dict):
            for entity_id, wrapped in saved_projections.items():
                if isinstance(wrapped, dict) and isinstance(wrapped.get("fields"), dict):
                    projections[str(entity_id)] = {
                        str(k): str(v) for k, v in wrapped["fields"].items()
                    }
        self._verify_replay(log, projections, artifacts)
        mpr_pixel_blobs = self._validate_mpr_content(artifacts, payload_documents)

        skipped_rows = _ensure_list(skipped_doc.get("skipped"), "skipped.json:skipped")
        skipped = tuple(
            _decode_typed(item, SkippedInstance, "skipped.json") for item in skipped_rows
        )

        return RestoredSession(
            paths=self._paths,
            study=study,
            series=series,
            volumes=volumes,
            artifacts=artifacts,
            payload_documents=payload_documents,
            provenance=provenance,
            log=log,
            skipped_instances=skipped,
            projections=projections,
            session_digest="",
            pixel_blobs=mpr_pixel_blobs,
        )

    def _validate_mpr_content(
        self,
        artifacts: tuple[DerivedArtifact, ...],
        payload_documents: dict[str, dict[str, object]],
    ) -> dict[str, bytes]:
        """Verify MPR payload closure and its immutable pixel blob binding.

        CORONARY_REVIEW_VIEW_V1 (narrow governed admission, Decision B): this
        method additionally routes ``ccta.cpr.view.v1`` ``MPR_VIEW`` artifacts
        through ``validate_cpr_payload`` + ``decode_cpr_pixel_blob`` with the
        same digest checks (outer ``payload_schema`` agreement, sealed-payload
        digest, pixel-blob content/size).  Schema selection is BY THE
        ARTIFACT'S OWN ``payload_schema`` (never by inference from the fields
        present); a CPR payload routed through the MPR validator (or vice
        versa) is a mismatch, never a fallback.  The orthogonal-MPR path
        (``ccta.mpr.view.v1``/``v2``) is byte-identical to before.
        """
        if not any(artifact.kind == ArtifactKind.MPR_VIEW for artifact in artifacts):
            return {}
        from apps.workstation.volume_loader import CanonicalVolumeLoader
        from visualization.mpr import validate_mpr_payload

        loader = CanonicalVolumeLoader(self._paths.blobs)
        problems: list[str] = []
        pixel_blobs: dict[str, bytes] = {}
        for artifact in artifacts:
            if artifact.kind != ArtifactKind.MPR_VIEW:
                continue
            document = payload_documents.get(artifact.artifact_id)
            if document is None:
                problems.append(f"MPR_VIEW {artifact.artifact_id}: payload document missing")
                # Restored base fall-through (F3): a missing document still
                # flows into the closed validator below, which records the
                # second typed problem (invalid payload) — exactly like base.
                document = {}
            schema = artifact.payload.payload_schema
            if schema == "ccta.cpr.view.v1":
                self._validate_one_cpr_view(artifact, document, loader, problems, pixel_blobs)
                continue
            try:
                validated = validate_mpr_payload(document)
            except Exception as exc:  # noqa: BLE001 - session integrity is total-or-fail
                problems.append(f"MPR_VIEW {artifact.artifact_id}: invalid payload ({exc})")
                continue
            # v1 and v2 frames both persist: the outer PayloadRef schema must
            # equal the payload's own schema_version, whichever version it is.
            if artifact.payload.payload_schema != validated.get("schema_version"):
                problems.append(
                    f"MPR_VIEW {artifact.artifact_id}: outer payload schema disagrees"
                )
            if canonical_digest(validated) != artifact.payload.blob_sha256:
                problems.append(f"MPR_VIEW {artifact.artifact_id}: payload digest mismatch")
            pixel_ref = validated["pixel_blob_ref"]
            if not isinstance(pixel_ref, dict):
                problems.append(f"MPR_VIEW {artifact.artifact_id}: pixel_blob_ref is not an object")
                continue
            digest = pixel_ref.get("sha256")
            byte_size = pixel_ref.get("byte_size")
            if not isinstance(digest, str):
                problems.append(f"MPR_VIEW {artifact.artifact_id}: pixel digest missing")
                continue
            try:
                data = loader.read_blob(digest)
                from visualization.mpr import decode_pixel_blob

                decode_pixel_blob(validated, data)
            except Exception as exc:  # noqa: BLE001
                problems.append(f"MPR_VIEW {artifact.artifact_id}: pixel blob unreadable ({exc})")
                continue
            if not isinstance(byte_size, int) or len(data) != byte_size:
                problems.append(
                    f"MPR_VIEW {artifact.artifact_id}: pixel blob size {len(data)} != {byte_size}"
                )
            else:
                pixel_blobs[digest] = data
        if problems:
            raise CanonicalValidationError(problems)
        return pixel_blobs

    def _validate_one_cpr_view(
        self,
        artifact: DerivedArtifact,
        document: dict[str, object],
        loader: Any,
        problems: list[str],
        pixel_blobs: dict[str, bytes],
    ) -> None:
        """Validate ONE ``ccta.cpr.view.v1`` view artifact (Decision B seam).

        Mirrors the MPR branch: closed payload validation, outer-schema
        agreement, sealed-payload digest, then pixel-blob content/size checks.
        Only an ``MPR_VIEW`` whose outer ``payload_schema`` is exactly
        ``ccta.cpr.view.v1`` reaches here; anything else stays on the MPR
        path above (no fallback, no ``plane`` fabrication).
        """
        from visualization.cpr import decode_cpr_pixel_blob, validate_cpr_payload

        try:
            validated = validate_cpr_payload(document)
        except Exception as exc:  # noqa: BLE001 - session integrity is total-or-fail
            problems.append(
                f"MPR_VIEW {artifact.artifact_id}: invalid CPR payload ({exc})"
            )
            return
        if artifact.payload.payload_schema != validated.get("schema_version"):
            problems.append(
                f"MPR_VIEW {artifact.artifact_id}: outer payload schema disagrees"
            )
        if canonical_digest(validated) != artifact.payload.blob_sha256:
            problems.append(f"MPR_VIEW {artifact.artifact_id}: payload digest mismatch")
        pixel_ref = validated["pixel_blob_ref"]
        if not isinstance(pixel_ref, dict):
            problems.append(
                f"MPR_VIEW {artifact.artifact_id}: pixel_blob_ref is not an object"
            )
            return
        digest = pixel_ref.get("sha256")
        byte_size = pixel_ref.get("byte_size")
        if not isinstance(digest, str):
            problems.append(f"MPR_VIEW {artifact.artifact_id}: pixel digest missing")
            return
        try:
            data = loader.read_blob(digest)
            decode_cpr_pixel_blob(validated, data)
        except Exception as exc:  # noqa: BLE001
            problems.append(
                f"MPR_VIEW {artifact.artifact_id}: pixel blob unreadable ({exc})"
            )
            return
        if not isinstance(byte_size, int) or len(data) != byte_size:
            problems.append(
                f"MPR_VIEW {artifact.artifact_id}: pixel blob size {len(data)} != {byte_size}"
            )
        else:
            pixel_blobs[digest] = data
    def _verify_replay(
        self,
        log: EventLog,
        saved_projections: dict[str, dict[str, str]],
        artifacts: tuple[DerivedArtifact, ...],
    ) -> None:
        problems: list[str] = []
        for entity_id, fields_map in saved_projections.items():
            try:
                live = log.project_state(entity_id)
            except KeyError:
                problems.append(f"projection {entity_id}: unknown to replayed journal")
                continue
            if live.to_dict() != fields_map:
                problems.append(f"projection {entity_id}: replay diverges from saved state")
        for artifact in artifacts:
            try:
                live = log.project_state(artifact.artifact_id, entity="derived_artifact")
            except KeyError:
                problems.append(f"artifact {artifact.artifact_id}: absent from journal")
                continue
            expected = {
                "review_state": artifact.review.review_state.value,
                "status": artifact.status.value,
                "release_lane": artifact.release_lane.value,
            }
            if {k: live.get(k) for k in expected} != expected:
                problems.append(f"artifact {artifact.artifact_id}: journal state diverges")
        if problems:
            raise SessionIntegrityError(problems)


def _ensure_list(value: object, where: str) -> list[object]:
    if not isinstance(value, list):
        raise SessionIntegrityError([f"{where}: expected a JSON array"])
    return value


__all__ = [
    "DerivedStateStoreManager",
    "MANIFEST_NAME",
    "MANIFEST_SCHEMA",
    "RestoredSession",
    "SessionIntegrityError",
    "SessionMissingError",
    "SessionSnapshot",
    "SkippedInstance",
    "WorkstationPaths",
    "decode_value",
    "encode_value",
]

