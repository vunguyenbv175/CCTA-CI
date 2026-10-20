"""StudyWorkspaceManager: local study lifecycle shell (§B1).

Phase 0 scope is deliberately headless (pre-research §3.1): import a DICOM
directory read-only, discover/classify series through the frozen WS-00
factories, materialize the canonical volume, derive deterministic artifacts,
and hand everything to the session store. No GUI, no network, no writes into
the DICOM source tree (Rule A).

The bundled reader parses ONLY a whitelisted subset of Explicit VR Little
Endian Part-10 files — exactly what the synthetic Phase-0 fixture emits.
Anything else fails closed as ``NOT_ANALYZABLE`` abstention instead of being
fabricated (AGENTS.md §3.10). Deep tag parsing remains a deferred pydicom
adapter behind this seam (Part C1).
"""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Final

from apps.workstation.round_trip import _compute_canonical_measurements
from apps.workstation.session_store import (
    RestoredSession,
    SkippedInstance,
    WorkstationPaths,
    encode_value,
)
from apps.workstation.volume_loader import (
    CANONICAL_VOLUME_CODECS,
    CanonicalVolumeLoader,
    build_frame_spec,
    compute_measurements,
    measurement_parameters_digest,
)
from core.artifacts import (
    ClaimBinding,
    DerivedArtifact,
    DerivedArtifactStore,
    GeometryBase,
    Lineage,
    PayloadRef,
    Producer,
    create_derived_artifact,
)
from core.coordinates import transform_for_frame
from core.dicom import (
    ADMITTED_TRANSFER_SYNTAXES,
    DicomAbstentionError,
    DicomInstance,
    ReadOnlySourceStore,
    SourceManifest,
    classify_series,
    read_dicom_instance,
    validate_stack,
    verify_file_digest,
)
from core.digest import canonical_digest
from core.enums import (
    ActorType,
    ArtifactKind,
    ContrastClass,
    ContrastPresence,
    ExamType,
    ImportSource,
    PrivacyClass,
    ProducerType,
    ProvenanceEventType,
    ReleaseLane,
    SeriesClass,
    VolumeKind,
)
from core.errors import CanonicalValidationError
from core.ids import new_ulid
from core.provenance import (
    Actor,
    BlobRef,
    Environment,
    InputRef,
    InvalidationDAG,
    NotApplicable,
    ProvenanceRecord,
    create_provenance,
)
from core.study import (
    AcquisitionPhase,
    InstanceEntry,
    OriginalDicomRef,
    Series,
    Study,
    create_study,
    register_series,
)
from core.volume import Volume, VolumeBlobRef, create_volume

SOFTWARE_VERSION: Final[str] = "0.1.0"
#: §44 environment binding: the WS-02 base commit this code line derives from.
CODE_COMMIT_SHA: Final[str] = "f5bea6f01411147315ebf59b053ada43e6632570"
READER_CLASSIFIER: Final[str] = "rule-table"
CLASSIFIER_VERSION: Final[str] = "1"
RELEASE_LANE: Final[ReleaseLane] = ReleaseLane.PERSONAL_RESEARCH

CLAIM_VIZ = ClaimBinding(claim_id="CLM-VIZ-001", claim_version="v0-draft")
CLAIM_MEAS = ClaimBinding(claim_id="CLM-MEAS-001", claim_version="v0-draft")

PAYLOAD_TRANSFORM_SCHEMA: Final[str] = "ccta.payload.coordinate_transform.v1"

_ENGINE_ACTOR: Final[Actor] = Actor(actor_type=ActorType.ENGINE, actor_id="ccta-workstation")


class NotAnalyzableError(CanonicalValidationError):
    """Typed abstention: input quality could not be established (§3.10)."""


# ---------------------------------------------------------------------------
# Minimal Explicit-VR-LE reader (whitelist-only extraction)
# ---------------------------------------------------------------------------

_TAG_SOP_INSTANCE_UID: Final[int] = 0x00080018
_TAG_MODALITY: Final[int] = 0x00080060
_TAG_STUDY_UID: Final[int] = 0x0020000D
_TAG_SERIES_UID: Final[int] = 0x0020000E
_TAG_FRAME_UID: Final[int] = 0x00200052
_TAG_POSITION: Final[int] = 0x00200032
_TAG_ORIENTATION: Final[int] = 0x00200037
_TAG_TEMPORAL: Final[int] = 0x00200100
_TAG_THICKNESS: Final[int] = 0x00180050
_TAG_PIXEL_SPACING: Final[int] = 0x00280030
_TAG_ROWS: Final[int] = 0x00280010
_TAG_COLUMNS: Final[int] = 0x00280011
_TAG_PIXEL_DATA: Final[int] = 0x7FE00010

_WHITELISTED_TAGS: Final[frozenset[int]] = frozenset(
    {
        _TAG_SOP_INSTANCE_UID,
        _TAG_MODALITY,
        _TAG_STUDY_UID,
        _TAG_SERIES_UID,
        _TAG_FRAME_UID,
        _TAG_POSITION,
        _TAG_ORIENTATION,
        _TAG_TEMPORAL,
        _TAG_THICKNESS,
        _TAG_PIXEL_SPACING,
        _TAG_ROWS,
        _TAG_COLUMNS,
        _TAG_PIXEL_DATA,
    }
)

_LONG_FORM_VRS: Final[frozenset[bytes]] = frozenset(
    {b"OB", b"OD", b"OF", b"OL", b"OV", b"OW", b"SQ", b"SV", b"UC", b"UN", b"UR", b"UT", b"UV"}
)


class UnsupportedDicomError(ValueError):
    """File cannot be parsed under the Phase-0 whitelist (fail-closed)."""


@dataclass(frozen=True)
class DicomLite:
    """Whitelisted view of one DICOM instance."""

    sop_instance_uid: str
    study_instance_uid: str
    series_instance_uid: str
    frame_of_reference_uid: str
    modality: str
    image_position_patient: tuple[float, float, float]
    image_orientation_patient: tuple[float, float, float, float, float, float]
    pixel_spacing: tuple[float, float]
    slice_thickness_mm: float | None
    rows: int
    columns: int
    temporal_position: str | None
    pixel_data: bytes


def read_dicom_lite(path: Path) -> DicomLite:
    """Parse one Explicit-VR-LE file, extracting ONLY whitelisted tags.

    Direct identifiers (patient name/id/birth date/accession) are not even
    decoded — PHI minimization starts at the parser boundary (P9).
    """
    data = path.read_bytes()
    if len(data) < 132 or data[128:132] != b"DICM":
        raise UnsupportedDicomError(f"{path.name}: missing DICOM part-10 magic")
    offset = 132
    transfer_syntax: str | None = None
    values: dict[int, bytes] = {}
    while offset < len(data):
        if offset + 8 > len(data):
            raise UnsupportedDicomError(f"{path.name}: truncated element header")
        tag_group, tag_element = struct.unpack_from("<HH", data, offset)
        vr_bytes = data[offset + 4 : offset + 6]
        if len(vr_bytes) != 2 or not vr_bytes.isalpha():
            raise UnsupportedDicomError(f"{path.name}: non-explicit-VR element at {offset}")
        if vr_bytes in _LONG_FORM_VRS:
            length = struct.unpack_from("<I", data, offset + 8)[0]
            header_len = 12
        else:
            length = struct.unpack_from("<H", data, offset + 6)[0]
            header_len = 8
        value_start = offset + header_len
        if length == 0xFFFFFFFF:
            raise UnsupportedDicomError(
                f"{path.name}: undefined-length elements are outside the Phase-0 whitelist"
            )
        if value_start + length > len(data):
            raise UnsupportedDicomError(f"{path.name}: truncated element value")
        tag = (tag_group << 16) | tag_element
        if tag_group == 0x0002:
            if tag == 0x00020010:
                transfer_syntax = (
                    data[value_start : value_start + length]
                    .decode("ascii")
                    .strip(" \x00")
                )
        elif tag in _WHITELISTED_TAGS:
            values[tag] = data[value_start : value_start + length]
        offset = value_start + length

    if transfer_syntax is None:
        raise UnsupportedDicomError(f"{path.name}: missing file-meta transfer syntax")
    if transfer_syntax not in ADMITTED_TRANSFER_SYNTAXES:
        raise UnsupportedDicomError(
            f"{path.name}: transfer syntax {transfer_syntax!r} unsupported in Phase 0"
        )

    def text(tag: int) -> str:
        return values[tag].decode("ascii").strip().strip("\x00").strip()

    def decimal_string(tag: int) -> tuple[float, ...]:
        return tuple(float(part) for part in text(tag).split("\\") if part.strip())

    try:
        position_raw = decimal_string(_TAG_POSITION)
        orientation_raw = decimal_string(_TAG_ORIENTATION)
        spacing_raw = decimal_string(_TAG_PIXEL_SPACING)
        rows = struct.unpack("<H", values[_TAG_ROWS])[0]
        columns = struct.unpack("<H", values[_TAG_COLUMNS])[0]
        sop_uid = text(_TAG_SOP_INSTANCE_UID)
        study_uid = text(_TAG_STUDY_UID)
        series_uid = text(_TAG_SERIES_UID)
        frame_uid = text(_TAG_FRAME_UID)
        modality = text(_TAG_MODALITY)
    except KeyError as exc:
        raise UnsupportedDicomError(f"{path.name}: missing required tag {exc}") from None

    if len(position_raw) != 3 or len(orientation_raw) != 6 or len(spacing_raw) != 2:
        raise UnsupportedDicomError(f"{path.name}: malformed geometry descriptors")

    thickness: float | None = None
    if _TAG_THICKNESS in values:
        parsed_thickness = decimal_string(_TAG_THICKNESS)
        if parsed_thickness:
            thickness = parsed_thickness[0]

    return DicomLite(
        sop_instance_uid=sop_uid,
        study_instance_uid=study_uid,
        series_instance_uid=series_uid,
        frame_of_reference_uid=frame_uid,
        modality=modality,
        rows=rows,
        columns=columns,
        image_position_patient=position_raw,
        image_orientation_patient=orientation_raw,
        pixel_spacing=spacing_raw,
        slice_thickness_mm=thickness,
        temporal_position=text(_TAG_TEMPORAL) if _TAG_TEMPORAL in values else None,
        pixel_data=values.get(_TAG_PIXEL_DATA, b""),
    )


def _slice_normal(
    orientation: tuple[float, float, float, float, float, float],
) -> tuple[float, float, float]:
    row, col = orientation[:3], orientation[3:]
    return (
        row[1] * col[2] - row[2] * col[1],
        row[2] * col[0] - row[0] * col[2],
        row[0] * col[1] - row[1] * col[0],
    )


# ---------------------------------------------------------------------------
# Workspace layout
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SeriesImport:
    """A discovered + classified series bound to its read-only source store."""

    series: Series
    store: ReadOnlySourceStore
    phase_label: str | None


@dataclass(frozen=True)
class ImportedStudy:
    """Result of a successful import (abstention raises instead)."""

    study: Study
    series: tuple[SeriesImport, ...]
    provenance: tuple[ProvenanceRecord, ...]
    skipped: tuple[SkippedInstance, ...]
    paths: WorkstationPaths


@dataclass(frozen=True)
class DeriveOutcome:
    """New canonical records produced by one derive run."""

    volumes: tuple[Volume, ...]
    artifacts: tuple[DerivedArtifact, ...]
    provenance: tuple[ProvenanceRecord, ...]
    already_derived_series: tuple[str, ...]
    payload_documents: dict[str, dict[str, object]]


def _environment(actor_note: str) -> Environment:
    return Environment(
        software_version=SOFTWARE_VERSION,
        code_commit_sha=CODE_COMMIT_SHA,
        runtime=f"python-stdlib ({actor_note})",
        deterministic_seed=None,
    )


def _na(reason: str) -> NotApplicable[Any]:
    """Explicit NotApplicable marker (§44) for optional provenance items."""
    return NotApplicable(value=None, reason=reason)


def _static_params_digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _document(value: object) -> dict[str, object]:
    encoded = encode_value(value)
    if not isinstance(encoded, dict):
        raise CanonicalValidationError("expected a document encoding of a dataclass")
    return encoded


def _phase_display_label(instance: DicomInstance) -> str | None:
    """Render the legacy fixture's normalized R-R label without inference."""

    if instance.temporal_position_index is None:
        return None
    protocol_name = instance.protocol_name
    if protocol_name and protocol_name.endswith("% r-r"):
        prefix = protocol_name[:-5]
        if prefix.isdigit():
            return f"{prefix}% R-R"
    return str(instance.temporal_position_index)


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class StudyWorkspaceManager:
    """Local study workspace lifecycle: import -> derive (§B1, §B2)."""

    def __init__(
        self,
        workspace_root: Path,
        *,
        id_factory: Callable[[], str] | None = None,
        actor: str = "workstation-cli",
    ) -> None:
        self._root = Path(workspace_root)
        self._id_factory = id_factory or new_ulid
        self._actor = actor
        self.paths: WorkstationPaths | None = None

    def initialize(self) -> WorkstationPaths:
        self.paths = WorkstationPaths.create(self._root)
        return self.paths

    # -- import ------------------------------------------------------------------

    def import_study(
        self,
        dicom_dir: Path,
        *,
        privacy_class: PrivacyClass = PrivacyClass.SYNTHETIC,
    ) -> ImportedStudy:
        """Walk *dicom_dir* READ-ONLY and build the frozen canonical records."""
        paths = self.initialize()
        source_root = Path(dicom_dir).resolve()
        if not source_root.is_dir():
            raise NotAnalyzableError(["dicom_dir is not a directory"])

        skipped: list[SkippedInstance] = []
        skipped_series_uids: set[str] = set()
        parsed: list[tuple[str, DicomInstance]] = []
        for file_path in sorted(source_root.rglob("*")):
            if not file_path.is_file():
                continue
            relative = file_path.relative_to(source_root).as_posix()
            try:
                parsed.append((relative, read_dicom_instance(file_path)))
            except DicomAbstentionError as exc:
                if exc.series_instance_uid is not None:
                    skipped_series_uids.add(exc.series_instance_uid)
                detail = exc.detail_code or exc.reason_code
                # The legacy test fixture checks that malformed input is
                # visibly identified as a magic failure.  The remainder is
                # the closed reason vocabulary; no raw Dataset/path text is
                # persisted in the skip record.
                reason = (
                    "magic: DICOM_SYNTAX_INVALID"
                    if detail == "DICOM_MAGIC_INVALID"
                    else str(detail)
                )
                skipped.append(SkippedInstance(path=relative, reason=reason))

        if not parsed:
            raise NotAnalyzableError(
                ["no analyzable DICOM instance found"] + [s.reason for s in skipped[:5]]
            )

        study_uids = {lite.study_instance_uid for _, lite in parsed}
        if len(study_uids) > 1:
            raise NotAnalyzableError(
                [f"directory mixes {len(study_uids)} StudyInstanceUIDs; refusing to guess"]
            )
        study_uid = study_uids.pop()

        grouped: dict[str, list[tuple[str, DicomInstance]]] = {}
        for relative, lite in parsed:
            grouped.setdefault(lite.series_instance_uid, []).append((relative, lite))

        provenance_records: list[ProvenanceRecord] = []
        imports: list[SeriesImport] = []
        contrast_seen = ContrastPresence.UNKNOWN
        contrast_absent_seen = False

        for series_uid in sorted(grouped):
            members = grouped[series_uid]
            member_instances = tuple(item for _, item in members)
            stack = validate_stack(
                member_instances,
                skipped_instance_could_belong=series_uid in skipped_series_uids,
            )
            ordered_instances = stack.instances if stack.admitted else member_instances
            ordered_members = [
                (next(relative for relative, item in members if item is instance), instance)
                for instance in ordered_instances
            ]
            entries = tuple(
                InstanceEntry(
                    sop_instance_uid=instance.sop_instance_uid,
                    file_path=relative,
                    sha256=hashlib.sha256(
                        (source_root / relative).read_bytes()
                    ).hexdigest(),
                )
                for relative, instance in ordered_members
            )

            manifest = SourceManifest(
                root_path=str(source_root),
                entries=tuple((e.sop_instance_uid, e.sha256) for e in entries),
            )
            classified = classify_series(ordered_instances, stack=stack)
            classified_provenance = create_provenance(
                subject_ref=series_uid,
                event_type=ProvenanceEventType.CLASSIFIED,
                actor=_ENGINE_ACTOR,
                inputs=(
                    InputRef(
                        ref=series_uid,
                        kind="series",
                        sha256_at_use=manifest.manifest_sha256,
                    ),
                ),
                preprocessing=(
                    "pydicom-3.0.2-boundary; rule-table-v1; evidence="
                    + ",".join(classified.provenance.evidence_fields)
                ),
                parameters_ref=BlobRef(
                    blob_ref=f"classifier:{READER_CLASSIFIER}:v{CLASSIFIER_VERSION}",
                    sha256=_static_params_digest(
                        f"classifier:{READER_CLASSIFIER}:v{CLASSIFIER_VERSION}"
                    ),
                ),
                human_edit=_na("fully automatic classification"),
                upstream_refs=_na("root classification record"),
                environment=_environment("classifier"),
                created_by=self._actor,
            )
            provenance_records.append(classified_provenance)
            if not stack.admitted:
                detail = stack.detail_code.value if stack.detail_code else "SERIES_STACK_INVALID"
                skipped.append(SkippedInstance(path=series_uid, reason=detail))
                continue
            store = ReadOnlySourceStore(manifest=manifest)
            first = ordered_members[0][1]
            if not classified.importable:
                skipped.append(
                    SkippedInstance(path=series_uid, reason=classified.visible_reason.value)
                )
                continue
            phase_label = next(
                (
                    _phase_display_label(instance)
                    for _, instance in ordered_members
                    if _phase_display_label(instance) is not None
                ),
                classified.phase_label,
            )
            contrast_class = {
                "CONTRAST": ContrastClass.ARTERIAL,
                "NON_CONTRAST": ContrastClass.NON_CONTRAST,
            }.get(classified.contrast.value, ContrastClass.UNKNOWN)
            if classified.contrast.value == "CONTRAST":
                contrast_seen = ContrastPresence.PRESENT
            elif classified.contrast.value == "NON_CONTRAST":
                contrast_absent_seen = True

            thickness_values = [
                instance.slice_thickness_mm
                for _, instance in ordered_members
                if instance.slice_thickness_mm
            ]
            series_record = register_series(
                study_ref="pending-import",
                study_multiphase=False,
                dicom_series_instance_uid=series_uid,
                frame_of_reference_uid=first.frame_of_reference_uid,
                modality=first.modality,
                series_class=SeriesClass.AXIAL_SOURCE,
                contrast_class=contrast_class,
                classifier=READER_CLASSIFIER,
                classification_provenance_ref=classified_provenance.provenance_id,
                instances=entries,
                ordered_sop_uids=tuple(e.sop_instance_uid for e in entries),
                original_dicom_ref=OriginalDicomRef(
                    root_path=str(source_root),
                    manifest_sha256=manifest.manifest_sha256,
                ),
                acquisition_phase=(
                    AcquisitionPhase(
                        label=phase_label,
                        temporal_position=(
                            float(first.temporal_position_index)
                            if first.temporal_position_index is not None
                            else None
                        ),
                    )
                    if phase_label
                    else None
                ),
                slice_thickness_mm=(
                    float(median(thickness_values)) if thickness_values else None
                ),
                created_by=self._actor,
                series_id=self._id_factory(),
            )
            imports.append(
                SeriesImport(series=series_record, store=store, phase_label=phase_label)
            )

        if not imports:
            raise NotAnalyzableError(
                ["every discovered series was refused"] + [s.reason for s in skipped[:5]]
            )

        multiphase = len(imports) > 1
        if multiphase and any(item.phase_label is None for item in imports):
            missing = [
                item.series.dicom_series_instance_uid
                for item in imports
                if item.phase_label is None
            ]
            raise NotAnalyzableError(
                [
                    f"multi-phase study lacks acquisition_phase for series {uid}"
                    for uid in missing
                ]
            )

        imported = create_provenance(
            subject_ref=study_uid,
            event_type=ProvenanceEventType.IMPORTED,
            actor=_ENGINE_ACTOR,
            inputs=tuple(
                InputRef(
                    ref=item.series.dicom_series_instance_uid,
                    kind="series",
                    sha256_at_use=item.series.manifest.composite_series_sha256,
                )
                for item in imports
            ),
            preprocessing="phase0-read-only-walk",
            parameters_ref=BlobRef(
                blob_ref="import:phase0-read-only-walk",
                sha256=_static_params_digest("import:phase0-read-only-walk"),
            ),
            human_edit=_na("automatic import"),
            upstream_refs=tuple(p.provenance_id for p in provenance_records),
            environment=_environment("importer"),
            created_by=self._actor,
        )
        provenance_records.append(imported)

        modalities = {item.series.modality for item in imports}
        exam_type = ExamType.CCTA if modalities == {"CT"} else ExamType.UNKNOWN
        if contrast_seen == ContrastPresence.UNKNOWN and contrast_absent_seen:
            contrast_seen = ContrastPresence.ABSENT
        study = create_study(
            dicom_study_instance_uid=study_uid,
            privacy_class=privacy_class,
            import_source=ImportSource.LOCAL_FOLDER,
            import_job_ref=imported.provenance_id,
            source_root_path=str(source_root),
            exam_type=exam_type,
            contrast_present=contrast_seen,
            multiphase=multiphase,
            classifier=READER_CLASSIFIER,
            created_by=self._actor,
            study_id=self._id_factory(),
        )
        imports = [
            SeriesImport(
                series=_reparent_series(item.series, study.study_id, multiphase),
                store=item.store,
                phase_label=item.phase_label,
            )
            for item in imports
        ]
        return ImportedStudy(
            study=study,
            series=tuple(imports),
            provenance=tuple(provenance_records),
            skipped=tuple(skipped),
            paths=paths,
        )

    # -- derive ---------------------------------------------------------------------

    def derive(self, restored: RestoredSession) -> DeriveOutcome:
        """Materialize canonical volume + deterministic artifacts per series."""
        assert self.paths is not None, "call initialize() before derive()"
        loader = CanonicalVolumeLoader(self.paths.blobs)
        store = DerivedArtifactStore(restored.log, InvalidationDAG())
        known_series = {
            ref for volume in restored.volumes for ref in volume.source_series_refs
        }
        new_volumes: list[Volume] = []
        new_artifacts: list[DerivedArtifact] = []
        new_prov: list[ProvenanceRecord] = []
        new_documents: dict[str, dict[str, object]] = {}
        already: list[str] = []

        imported_prov = _single_provenance(restored.provenance, ProvenanceEventType.IMPORTED)

        for series in restored.series:
            if series.series_id in known_series:
                already.append(series.series_id)
                continue

            volume, vol_prov = self._materialize_volume(loader, series, imported_prov)
            new_volumes.append(volume)
            new_prov.append(vol_prov)

            transform = transform_for_frame(
                volume.coordinate_frame,
                series_instance_uid=series.dicom_series_instance_uid,
            )
            transform_doc = _document(transform)
            transform_artifact, transform_doc, transform_prov = self._build_artifact(
                store,
                restored.study.study_id,
                kind=ArtifactKind.COORDINATE_TRANSFORM,
                payload_schema=PAYLOAD_TRANSFORM_SCHEMA,
                payload_document=transform_doc,
                inline_summary=(("transform_id", transform.transform_id),),
                lineage=Lineage(input_volume_refs=(volume.volume_id,)),
                claim_binding=CLAIM_VIZ,
                algorithm="dicom-frame-transform",
                parameters_sha256=canonical_digest(_document(volume.coordinate_frame)),
                inputs=(
                    InputRef(
                        ref=volume.volume_id,
                        kind="volume",
                        sha256_at_use=volume.blob_ref.sha256,
                    ),
                ),
                preprocessing="voxel-to-world-affine-from-frame-spec",
                upstream_refs=(vol_prov.provenance_id,),
            )
            # R1 F2: the derive path now materializes through the canonical
            # adapter, so the blob may carry either a canonical codec (decode
            # per its envelope) or — for a record-free legacy envelope from a
            # pre-amendment workspace — the raw uint16-le bytes.  The frozen
            # central-box ROI rule is byte-identical for both.
            envelope = loader.read_envelope(volume.blob_ref.sha256)
            if envelope.get("codec") in CANONICAL_VOLUME_CODECS:
                voxels = loader.read_array(
                    volume.blob_ref.sha256, envelope=envelope
                )
                measurement_doc = _compute_canonical_measurements(
                    voxels, volume.coordinate_frame.dimensions
                )
            else:
                voxels = loader.read_blob(volume.blob_ref.sha256)
                measurement_doc = compute_measurements(
                    voxels, volume.coordinate_frame.dimensions
                )
            measurement_artifact, measurement_doc, measurement_prov = self._build_artifact(
                store,
                restored.study.study_id,
                kind=ArtifactKind.MEASUREMENT_SET,
                payload_schema=str(measurement_doc["schema"]),
                payload_document=measurement_doc,
                inline_summary=(("algorithm", str(measurement_doc["algorithm"])),),
                lineage=Lineage(
                    input_volume_refs=(volume.volume_id,),
                    input_artifact_refs=(transform_artifact.artifact_id,),
                    geometry_base=GeometryBase(
                        geometry_id=transform_artifact.artifact_id, geometry_version=1
                    ),
                ),
                claim_binding=CLAIM_MEAS,
                algorithm=str(measurement_doc["algorithm"]),
                parameters_sha256=measurement_parameters_digest(),
                inputs=(
                    InputRef(
                        ref=transform_artifact.artifact_id,
                        kind="artifact",
                        sha256_at_use=transform_artifact.payload.blob_sha256,
                    ),
                    InputRef(
                        ref=volume.volume_id,
                        kind="volume",
                        sha256_at_use=volume.blob_ref.sha256,
                    ),
                ),
                preprocessing="central-box-hu-stats",
                upstream_refs=(vol_prov.provenance_id,),
            )
            new_artifacts.append(transform_artifact)
            new_artifacts.append(measurement_artifact)
            new_documents[transform_artifact.artifact_id] = transform_doc
            new_documents[measurement_artifact.artifact_id] = measurement_doc
            new_prov.append(transform_prov)
            new_prov.append(measurement_prov)

        return DeriveOutcome(
            volumes=tuple(new_volumes),
            artifacts=tuple(new_artifacts),
            provenance=tuple(new_prov),
            already_derived_series=tuple(already),
            payload_documents=new_documents,
        )

    def _build_artifact(
        self,
        store: DerivedArtifactStore,
        study_id: str,
        *,
        kind: ArtifactKind,
        payload_schema: str,
        payload_document: dict[str, object],
        inline_summary: tuple[tuple[str, str], ...],
        lineage: Lineage,
        claim_binding: ClaimBinding,
        algorithm: str,
        parameters_sha256: str,
        inputs: tuple[InputRef, ...],
        preprocessing: str,
        upstream_refs: tuple[str, ...],
    ) -> tuple[DerivedArtifact, dict[str, object], ProvenanceRecord]:
        """Create provenance + artifact with pre-minted identity (no repair)."""
        artifact_id = self._id_factory()
        provenance = create_provenance(
            subject_ref=artifact_id,
            event_type=ProvenanceEventType.DERIVED,
            actor=_ENGINE_ACTOR,
            inputs=inputs,
            preprocessing=preprocessing,
            parameters_ref=BlobRef(
                blob_ref=payload_schema, sha256=parameters_sha256
            ),
            human_edit=_na("deterministic engine"),
            upstream_refs=upstream_refs,
            environment=_environment(algorithm),
            created_by=self._actor,
        )
        artifact = create_derived_artifact(
            study_ref=study_id,
            kind=kind,
            payload=PayloadRef(
                payload_schema=payload_schema,
                blob_sha256=canonical_digest(payload_document),
                codec="json",
                inline_summary=inline_summary,
            ),
            lineage=lineage,
            claim_binding=claim_binding,
            producer=Producer(
                producer_type=ProducerType.DETERMINISTIC_ENGINE,
                algorithm=algorithm,
                algorithm_version=SOFTWARE_VERSION,
                parameters_blob_sha256=parameters_sha256,
            ),
            release_lane=RELEASE_LANE,
            provenance_ref=provenance.provenance_id,
            created_by=self._actor,
            artifact_id=artifact_id,
        )
        store.add(artifact)
        return artifact, payload_document, provenance

    # -- volume materialization ---------------------------------------------------------

    def _materialize_volume(
        self,
        loader: CanonicalVolumeLoader,
        series: Series,
        imported_prov: ProvenanceRecord,
    ) -> tuple[Volume, ProvenanceRecord]:
        assert self.paths is not None
        source_root = Path(series.original_dicom_ref.root_path)
        instances: list[DicomInstance] = []
        for entry in series.manifest.instances:
            full = source_root / entry.file_path
            verify_file_digest(full, entry.sha256)
            instance = read_dicom_instance(full)
            if instance.sop_instance_uid != entry.sop_instance_uid:
                raise CanonicalValidationError(
                    f"source drift: {entry.file_path} no longer carries "
                    f"{entry.sop_instance_uid}"
                )
            instances.append(instance)

        stack = validate_stack(tuple(instances))
        if not stack.admitted or stack.slice_spacing_mm is None:
            detail = stack.detail_code.value if stack.detail_code else "SERIES_STACK_INVALID"
            raise NotAnalyzableError(
                [f"series {series.series_id}: {detail}"]
            )
        ordered_instances = stack.instances
        rows, columns = ordered_instances[0].rows, ordered_instances[0].columns
        slice_spacing = stack.slice_spacing_mm

        shape = (len(ordered_instances), rows, columns)
        frame = build_frame_spec(
            frame_of_reference_uid=series.frame_of_reference_uid,
            image_position_patient=ordered_instances[0].image_position_patient,
            image_orientation_patient=ordered_instances[0].image_orientation_patient,
            pixel_spacing_mm=ordered_instances[0].pixel_spacing,
            slice_spacing_mm=slice_spacing,
            dimensions=shape,
        )
        # R1 F2: derivation routes through the stack adapter — the single
        # authority for stack bit-depth uniformity, per-file 12-bit padding
        # conformance, the canonical codec decision, and the source-bit-depth
        # provenance record.  This path never hand-writes a legacy envelope.
        #
        # R2 F15: the canonical domain for this CT derive path is HU, selected
        # HERE (the one product derive selection point), not left at the
        # adapter's STORED default.  Raw stored samples are not the product's
        # scalar contract: the measurement artifact writes mean/min/max_hu
        # under algorithm ``central-box-hu-stats``, the MPR payload carries the
        # volume's scalar_domain, and the viewer's frozen presets are defined
        # in HU and applied directly to decoded MPR pixels (VIEWER_001_SPEC
        # Part 4.1: the WL arithmetic is over HU and never over stored values
        # with a different intercept assumption).  Deriving STORED made all
        # three HU-semantic while the samples were raw.
        #
        # Safe by construction: (a) an absent rescale is recorded by the reader
        # as slope 1 / intercept 0, so non-rescaled fixtures are numerically
        # unchanged; (b) the adapter fails closed before this call on any
        # incompatible series — MODALITY_LUT (MODALITY_LUT_UNSUPPORTED),
        # padding (PIXEL_PADDING_UNSUPPORTED), an unsupported RescaleType
        # (RESCALE_TYPE_UNSUPPORTED; the reader admits only HU/NONE), and
        # non-conformant padding bits — so no such series ever reaches derive;
        # (c) the accepted-domain table already lists HU, and the codec
        # decision (int16 vs float32) is unchanged by the domain selection.
        materialized = loader.materialize_dicom_stack(
            ordered_instances, scalar_domain="HU", intensity_unit="HU"
        )
        blob = materialized.blob
        envelope = materialized.envelope
        codec = str(envelope["codec"])

        volume_id = self._id_factory()
        vol_prov = create_provenance(
            subject_ref=volume_id,
            event_type=ProvenanceEventType.DERIVED,
            actor=_ENGINE_ACTOR,
            inputs=(
                InputRef(
                    ref=series.dicom_series_instance_uid,
                    kind="series",
                    sha256_at_use=series.manifest.composite_series_sha256,
                ),
            ),
            preprocessing="phase0-slice-sort-and-concatenate",
            parameters_ref=BlobRef(
                blob_ref="volume:phase0-slice-sort-and-concatenate",
                sha256=_static_params_digest("volume:phase0-slice-sort-and-concatenate"),
            ),
            human_edit=_na("deterministic assembly"),
            upstream_refs=(imported_prov.provenance_id,),
            environment=_environment("volume-loader"),
            created_by=self._actor,
        )
        volume = create_volume(
            study_ref=series.study_ref,
            source_series_refs=(series.series_id,),
            kind=VolumeKind.CANONICAL,
            coordinate_frame=frame,
            blob_ref=VolumeBlobRef(
                sha256=blob.sha256, codec=codec, byte_size=blob.byte_size
            ),
            provenance_ref=vol_prov.provenance_id,
            release_lane=RELEASE_LANE,
            volume_id=volume_id,
        )
        return volume, vol_prov


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _reparent_series(series: Series, study_id: str, multiphase: bool) -> Series:
    """Re-freeze a series against the final study id + multiphase flag."""
    return register_series(
        study_ref=study_id,
        study_multiphase=multiphase,
        dicom_series_instance_uid=series.dicom_series_instance_uid,
        frame_of_reference_uid=series.frame_of_reference_uid,
        modality=series.modality,
        series_class=series.classification.series_class,
        contrast_class=series.classification.contrast_class,
        classifier=series.classification.classifier,
        classification_provenance_ref=series.classification.classification_provenance_ref,
        instances=series.manifest.instances,
        ordered_sop_uids=series.manifest.ordered_sop_uids,
        original_dicom_ref=series.original_dicom_ref,
        acquisition_phase=series.manifest.acquisition_phase,
        series_number=series.series_number,
        description=series.description,
        reconstruction_kernel=series.classification.reconstruction_kernel,
        slice_thickness_mm=series.classification.slice_thickness_mm,
        created_by=series.created_by,
        series_id=series.series_id,
    )


def _single_provenance(
    records: tuple[ProvenanceRecord, ...], event_type: ProvenanceEventType
) -> ProvenanceRecord:
    matches = [record for record in records if record.event_type == event_type]
    if len(matches) != 1:
        raise CanonicalValidationError(
            f"expected exactly one {event_type.value} provenance record, found {len(matches)}"
        )
    return matches[0]


__all__ = [
    "CODE_COMMIT_SHA",
    "CLASSIFIER_VERSION",
    "PAYLOAD_TRANSFORM_SCHEMA",
    "READER_CLASSIFIER",
    "SOFTWARE_VERSION",
    "DeriveOutcome",
    "DicomLite",
    "ImportedStudy",
    "NotAnalyzableError",
    "SeriesImport",
    "SkippedInstance",
    "StudyWorkspaceManager",
    "UnsupportedDicomError",
    "WorkstationPaths",
    "read_dicom_lite",
]
