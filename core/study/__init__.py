"""``Study`` and ``Series`` containers (Part B §3.1–3.2).

Both records freeze at import time. Workflow state and annotations live in
append-only event records, never as mutable fields on the Study record.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.digest import canonical_digest, require_sha256_hex
from core.enums import (
    ContrastClass,
    ContrastPresence,
    ExamType,
    ImportSource,
    PrivacyClass,
    ProvenanceEventType,
    SeriesClass,
)
from core.errors import CanonicalValidationError
from core.events import utc_now_iso, validate_iso8601_utc
from core.ids import new_ulid

STUDY_SCHEMA = "ccta.canonical.study.v1"
SERIES_SCHEMA = "ccta.canonical.series.v1"


@dataclass(frozen=True)
class ImportInfo:
    """Import block of a Study record (Part B §3.1)."""

    imported_at: str
    import_source: ImportSource
    import_job_ref: str  # ref<ProvenanceRecord>
    source_root_path: str


@dataclass(frozen=True)
class ExamClassification:
    """Exam classification recorded once at import, immutable afterwards."""

    exam_type: ExamType
    contrast_present: ContrastPresence
    multiphase: bool
    classifier: str


@dataclass(frozen=True)
class AnnotationEvent:
    """Append-only study annotation (never mutates the Study record)."""

    annotation_id: str
    text: str
    at: str
    by: str


@dataclass(frozen=True)
class Study:
    """Root container of one read case (Part B §3.1)."""

    study_id: str
    dicom_study_instance_uid: str
    privacy_class: PrivacyClass
    accession_number: str | None  # SENSITIVE - sealed, never exported/logged
    import_info: ImportInfo
    exam_classification: ExamClassification
    dataset_refs: tuple[str, ...]
    notes: tuple[AnnotationEvent, ...] = ()
    schema_version: str = STUDY_SCHEMA
    created_by: str = ""

    @property
    def created_at(self) -> str:
        """Creation timestamp == import timestamp (frozen at import)."""
        return self.import_info.imported_at

    def with_annotation(self, annotation: AnnotationEvent) -> Study:
        """Return a NEW record carrying an additional annotation event.

        The original record stays untouched (P3 append-only discipline);
        callers re-register the new version through the store.
        """
        _validate_annotation(annotation)
        return Study(
            study_id=self.study_id,
            dicom_study_instance_uid=self.dicom_study_instance_uid,
            privacy_class=self.privacy_class,
            accession_number=self.accession_number,
            import_info=self.import_info,
            exam_classification=self.exam_classification,
            dataset_refs=self.dataset_refs,
            notes=(*self.notes, annotation),
            schema_version=self.schema_version,
            created_by=self.created_by,
        )


@dataclass(frozen=True)
class AcquisitionPhase:
    """Phase identity of a series manifest (Luna fix #2)."""

    label: str  # e.g. "75% R-R", "ED", "ES"
    temporal_position: float | None = None


@dataclass(frozen=True)
class InstanceEntry:
    """One DICOM instance bound into the immutable series manifest."""

    sop_instance_uid: str
    file_path: str
    sha256: str


@dataclass(frozen=True)
class InstanceManifest:
    """Immutable manifest recorded at import (Part B §3.2)."""

    instance_count: int
    ordered_sop_uids: tuple[str, ...]
    composite_series_sha256: str
    instances: tuple[InstanceEntry, ...]
    acquisition_phase: AcquisitionPhase | None


def compute_composite_series_sha256(
    ordered_sop_uids: tuple[str, ...], instance_sha256s: tuple[str, ...]
) -> str:
    """Deterministic composite digest over sorted SOP uids + per-file digests.

    The sort normalizes away slice-order disagreements between writers while
    ``ordered_sop_uids`` preserves the true slice sequence.
    """
    payload = {
        "sorted_sop_uids": sorted(ordered_sop_uids),
        "instance_sha256s_sorted": sorted(instance_sha256s),
    }
    return canonical_digest(payload)


@dataclass(frozen=True)
class SeriesClassification:
    """Series classification block (Master Plan §8)."""

    series_class: SeriesClass
    contrast_class: ContrastClass
    classifier: str
    phase_label: str | None = None
    reconstruction_kernel: str | None = None
    slice_thickness_mm: float | None = None
    classification_provenance_ref: str = ""  # ref<ProvenanceRecord>, required


@dataclass(frozen=True)
class OriginalDicomRef:
    """READ-ONLY contract pointing at untouched original bytes (§6.1)."""

    root_path: str
    manifest_sha256: str


@dataclass(frozen=True)
class Series:
    """One DICOM series discovered + classified within a Study (§3.2)."""

    series_id: str
    study_ref: str
    dicom_series_instance_uid: str
    frame_of_reference_uid: str
    modality: str
    classification: SeriesClassification
    manifest: InstanceManifest
    original_dicom_ref: OriginalDicomRef
    series_number: int | None = None
    description: str | None = None  # SENSITIVE - may contain PHI
    schema_version: str = SERIES_SCHEMA
    created_at: str = ""
    created_by: str = ""
    provenance_event_type: ProvenanceEventType = ProvenanceEventType.DISCOVERED


# ---------------------------------------------------------------------------
# Creation-time validation (fail-closed factories)
# ---------------------------------------------------------------------------


def create_study(
    *,
    dicom_study_instance_uid: str,
    privacy_class: PrivacyClass,
    import_source: ImportSource,
    import_job_ref: str,
    source_root_path: str,
    exam_type: ExamType,
    contrast_present: ContrastPresence,
    multiphase: bool,
    classifier: str,
    dataset_refs: tuple[str, ...] = (),
    accession_number: str | None = None,
    created_by: str,
    imported_at: str | None = None,
    study_id: str | None = None,
) -> Study:
    """Validate and freeze a :class:`Study` record at import time."""
    violations: list[str] = []
    if not dicom_study_instance_uid:
        violations.append("dicom_study_instance_uid must be non-empty")
    if not isinstance(privacy_class, PrivacyClass):
        violations.append("privacy_class must be a PrivacyClass")
    if not import_source or not source_root_path:
        violations.append("import_source and source_root_path are required")
    if not import_job_ref:
        violations.append("import_job_ref (provenance) is required for import")
    if not classifier:
        violations.append("exam_classification.classifier is required")
    if not created_by:
        violations.append("created_by actor is required")
    stamp = imported_at or utc_now_iso()
    try:
        validate_iso8601_utc(stamp, "imported_at")
    except ValueError as exc:
        violations.append(str(exc))
    if violations:
        raise CanonicalValidationError(violations)
    return Study(
        study_id=study_id or new_ulid(),
        dicom_study_instance_uid=dicom_study_instance_uid,
        privacy_class=privacy_class,
        accession_number=accession_number,
        import_info=ImportInfo(
            imported_at=stamp,
            import_source=import_source,
            import_job_ref=import_job_ref,
            source_root_path=source_root_path,
        ),
        exam_classification=ExamClassification(
            exam_type=exam_type,
            contrast_present=contrast_present,
            multiphase=multiphase,
            classifier=classifier,
        ),
        dataset_refs=tuple(dataset_refs),
        notes=(),
        created_by=created_by,
    )


def register_series(
    *,
    study_ref: str,
    study_multiphase: bool,
    dicom_series_instance_uid: str,
    frame_of_reference_uid: str,
    modality: str,
    series_class: SeriesClass,
    contrast_class: ContrastClass,
    classifier: str,
    classification_provenance_ref: str,
    instances: tuple[InstanceEntry, ...],
    ordered_sop_uids: tuple[str, ...],
    original_dicom_ref: OriginalDicomRef,
    acquisition_phase: AcquisitionPhase | None = None,
    series_number: int | None = None,
    description: str | None = None,
    reconstruction_kernel: str | None = None,
    slice_thickness_mm: float | None = None,
    created_by: str,
    created_at: str | None = None,
    series_id: str | None = None,
) -> Series:
    """Validate and freeze a :class:`Series` record (manifest sealed here)."""
    violations: list[str] = []
    stamp = created_at or utc_now_iso()
    try:
        validate_iso8601_utc(stamp, "created_at")
    except ValueError as exc:
        violations.append(str(exc))
    if not study_ref:
        violations.append("study_ref is required (Series cannot exist outside a Study)")
    if not dicom_series_instance_uid:
        violations.append("dicom_series_instance_uid must be non-empty")
    if not frame_of_reference_uid:
        violations.append("frame_of_reference_uid is mandatory for coordinates/export")
    if not modality:
        violations.append("modality is required")
    if not classifier:
        violations.append("classification.classifier is required")
    if not classification_provenance_ref:
        violations.append("classification.classification_provenance_ref is required")
    if not created_by:
        violations.append("created_by actor is required")
    if not original_dicom_ref.root_path:
        violations.append("original_dicom_ref.root_path is required")
    else:
        try:
            require_sha256_hex(
                original_dicom_ref.manifest_sha256, "original_dicom_ref.manifest_sha256"
            )
        except ValueError as exc:
            violations.append(str(exc))

    if not instances:
        violations.append("instance manifest requires at least one instance")
    elif tuple(inst.sop_instance_uid for inst in instances) != tuple(ordered_sop_uids):
        violations.append("instances must match ordered_sop_uids exactly, in order")
    if any(not uid for uid in ordered_sop_uids):
        violations.append("ordered_sop_uids contains empty entries")
    if len(set(ordered_sop_uids)) != len(ordered_sop_uids):
        violations.append("ordered_sop_uids contains duplicates")
    for idx, inst in enumerate(instances):
        try:
            require_sha256_hex(inst.sha256, f"instances[{idx}].sha256")
        except ValueError as exc:
            violations.append(str(exc))
        if not inst.file_path:
            violations.append(f"instances[{idx}].file_path is empty")

    composite = ""
    manifest_viable = (
        instances
        and len(set(ordered_sop_uids)) == len(ordered_sop_uids)
        and tuple(inst.sop_instance_uid for inst in instances) == tuple(ordered_sop_uids)
        and all(inst.sha256 and inst.file_path for inst in instances)
        and all(_sha_ok(inst.sha256) for inst in instances)
    )
    if manifest_viable:
        composite = compute_composite_series_sha256(
            ordered_sop_uids, tuple(inst.sha256 for inst in instances)
        )
    elif not violations:
        violations.append("composite digest cannot be computed over invalid manifest")

    if study_multiphase and acquisition_phase is None:
        violations.append(
            "acquisition_phase is MANDATORY for series of a multi-phase study (H-009)"
        )

    if violations:
        raise CanonicalValidationError(violations)

    return Series(
        series_id=series_id or new_ulid(),
        study_ref=study_ref,
        dicom_series_instance_uid=dicom_series_instance_uid,
        frame_of_reference_uid=frame_of_reference_uid,
        modality=modality,
        classification=SeriesClassification(
            series_class=series_class,
            contrast_class=contrast_class,
            classifier=classifier,
            phase_label=acquisition_phase.label if acquisition_phase else None,
            reconstruction_kernel=reconstruction_kernel,
            slice_thickness_mm=slice_thickness_mm,
            classification_provenance_ref=classification_provenance_ref,
        ),
        manifest=InstanceManifest(
            instance_count=len(instances),
            ordered_sop_uids=tuple(ordered_sop_uids),
            composite_series_sha256=composite,
            instances=tuple(instances),
            acquisition_phase=acquisition_phase,
        ),
        original_dicom_ref=original_dicom_ref,
        series_number=series_number,
        description=description,
        created_at=stamp,
        created_by=created_by,
    )


def _sha_ok(value: str) -> bool:
    try:
        require_sha256_hex(value, "sha256")
    except ValueError:
        return False
    return True


def _validate_annotation(annotation: AnnotationEvent) -> None:
    problems: list[str] = []
    if not annotation.annotation_id or not annotation.text or not annotation.by:
        problems.append("annotation needs id, text and author")
    try:
        validate_iso8601_utc(annotation.at, "annotation.at")
    except ValueError as exc:
        problems.append(str(exc))
    if problems:
        raise CanonicalValidationError(problems)


__all__ = [
    "AcquisitionPhase",
    "AnnotationEvent",
    "ExamClassification",
    "ImportInfo",
    "InstanceEntry",
    "InstanceManifest",
    "OriginalDicomRef",
    "SERIES_SCHEMA",
    "STUDY_SCHEMA",
    "Series",
    "SeriesClassification",
    "Study",
    "compute_composite_series_sha256",
    "create_study",
    "register_series",
]
