"""Original DICOM store contract - READ ONLY (Master Plan §6.1 / I1).

Original bytes and tags are never modified; de-identification creates a new
export artifact. This module provides the manifest verification + the
read-only binding every import must produce. It deliberately contains no
write path.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from core.dicom.classification import (
    BOUNDARY_GRID,
    CARDIAC_KEYWORDS,
    CLASSIFICATION_BOUNDARY_GRID,
    COMPLETE_CLASSIFICATION_ORACLE,
    DUPLICATE_POSITION_TOLERANCE_MM,
    EXCLUDED_IMAGE_TYPE_COMPONENTS,
    IN_PLANE_DRIFT_TOLERANCE_MM,
    IN_PLANE_SPACING_TOLERANCE_MM,
    INTER_SLICE_GAP_TOLERANCE_MM,
    NON_CONTRAST_PHRASES,
    ORIENTATION_TOLERANCE,
    PARALLEL_DIRECTION_TOLERANCE,
    ClassificationLabel,
    ClassificationOutcome,
    ClassificationProvenance,
    ClassificationReason,
    ClassificationResult,
    ContrastState,
    DimensionClassification,
    GeometryState,
    PhaseState,
    ProtocolState,
    RuleTableClassifier,
    StackAdmissionError,
    StackAdmissionResult,
    SyntaxState,
    admit_stack,
    classification_boundary_grid,
    classify,
    classify_boundary_states,
    classify_series,
    validate_stack,
)
from core.dicom.instance import (
    ADMITTED_TRANSFER_SYNTAXES,
    CARDIAC_SIGNAL_SOURCES,
    CARDIAC_SYNCHRONIZATION_TECHNIQUES,
    CT_IMAGE_STORAGE_SOP_CLASS_UID,
    MAX_NORMALIZED_TEXT_LENGTH,
    PHOTOMETRIC_INTERPRETATIONS,
    RESCALE_TYPES,
    TRANSFER_SYNTAX_EXPLICIT_VR_LE,
    TRANSFER_SYNTAX_IMPLICIT_VR_LE,
    DicomAbstentionError,
    DicomInstance,
    DicomReadError,
    DicomSyntaxError,
    DicomValidationError,
    UnsupportedTransferSyntaxError,
    is_valid_dicom_time,
    normalize_text,
    read_dicom_instance,
    timing_pair_is_valid,
)
from core.dicom.qc import (
    QC_ANATOMIC_IDENTIFIER_SCHEMA,
    QC_FLAGS,
    QC_JUSTIFICATION_CODES,
    QC_PAYLOAD_SCHEMA,
    QCAssessmentArtifact,
    QCAssessmentPayload,
    QCGateDecision,
    QCGateResult,
    QCProducer,
    QCReason,
    QCScope,
    check_qc_gate,
    create_qc_assessment,
    make_qc_assessment,
    qc_consumer_gate,
)
from core.digest import require_sha256_hex
from core.errors import DigestMismatchError


@dataclass(frozen=True)
class SourceManifest:
    """Frozen evidence about the untouched originals of one series."""

    root_path: str
    entries: tuple[tuple[str, str], ...]  # ordered (sop_instance_uid, file sha256)

    def __post_init__(self) -> None:
        if not self.root_path:
            raise ValueError("root_path is required")
        if not self.entries:
            raise ValueError("manifest needs at least one entry")

    @property
    def manifest_sha256(self) -> str:
        """Digest over sorted SOP uids + per-file digests (identity seal)."""
        payload = {
            "sorted_sop_uids": sorted(uid for uid, _ in self.entries),
            "instance_sha256s_sorted": sorted(sha for _, sha in self.entries),
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()


def verify_file_digest(path: str | Path, expected_sha256: str) -> None:
    """Fail closed unless the file at *path* hashes to *expected_sha256*."""
    require_sha256_hex(expected_sha256, "expected_sha256")
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise DigestMismatchError(
            f"original bytes tampered or moved: {path} has {digest}, expected {expected_sha256}"
        )


@dataclass(frozen=True)
class ReadOnlySourceStore:
    """Read-only handle onto original DICOM files (§6.1).

    Only verification and byte access are exposed; there is no API capable
    of writing into ``source/dicom/``. The dataclass is frozen, so even the
    handle itself cannot be repointed at different bytes.
    """

    manifest: SourceManifest

    def open_bytes(self, sop_instance_uid: str, relative_path: str) -> bytes:
        """Return original bytes after verifying their digest."""
        for uid, sha in self.manifest.entries:
            if uid == sop_instance_uid:
                full = Path(self.manifest.root_path) / relative_path
                data = full.read_bytes()
                actual = hashlib.sha256(data).hexdigest()
                if actual != sha:
                    raise DigestMismatchError(
                        f"original bytes for {sop_instance_uid} do not match manifest"
                    )
                return data
        raise KeyError(f"sop_instance_uid not in manifest: {sop_instance_uid}")


__all__ = [
    "ADMITTED_TRANSFER_SYNTAXES",
    "BOUNDARY_GRID",
    "CARDIAC_KEYWORDS",
    "CARDIAC_SIGNAL_SOURCES",
    "CARDIAC_SYNCHRONIZATION_TECHNIQUES",
    "CLASSIFICATION_BOUNDARY_GRID",
    "COMPLETE_CLASSIFICATION_ORACLE",
    "CT_IMAGE_STORAGE_SOP_CLASS_UID",
    "DicomAbstentionError",
    "DicomInstance",
    "DicomReadError",
    "DicomSyntaxError",
    "DicomValidationError",
    "EXCLUDED_IMAGE_TYPE_COMPONENTS",
    "IN_PLANE_DRIFT_TOLERANCE_MM",
    "IN_PLANE_SPACING_TOLERANCE_MM",
    "INTER_SLICE_GAP_TOLERANCE_MM",
    "MAX_NORMALIZED_TEXT_LENGTH",
    "NON_CONTRAST_PHRASES",
    "ORIENTATION_TOLERANCE",
    "PARALLEL_DIRECTION_TOLERANCE",
    "PHOTOMETRIC_INTERPRETATIONS",
    "QC_ANATOMIC_IDENTIFIER_SCHEMA",
    "QC_FLAGS",
    "QC_JUSTIFICATION_CODES",
    "QC_PAYLOAD_SCHEMA",
    "QCAssessmentArtifact",
    "QCAssessmentPayload",
    "QCGateDecision",
    "QCGateResult",
    "QCProducer",
    "QCReason",
    "QCScope",
    "RESCALE_TYPES",
    "RuleTableClassifier",
    "StackAdmissionError",
    "StackAdmissionResult",
    "SyntaxState",
    "TRANSFER_SYNTAX_EXPLICIT_VR_LE",
    "TRANSFER_SYNTAX_IMPLICIT_VR_LE",
    "UnsupportedTransferSyntaxError",
    "ClassificationLabel",
    "ClassificationOutcome",
    "ClassificationProvenance",
    "ClassificationReason",
    "ClassificationResult",
    "ContrastState",
    "DimensionClassification",
    "DUPLICATE_POSITION_TOLERANCE_MM",
    "GeometryState",
    "PhaseState",
    "ProtocolState",
    "ReadOnlySourceStore",
    "SourceManifest",
    "admit_stack",
    "check_qc_gate",
    "classify",
    "classify_boundary_states",
    "classify_series",
    "classification_boundary_grid",
    "create_qc_assessment",
    "is_valid_dicom_time",
    "make_qc_assessment",
    "normalize_text",
    "qc_consumer_gate",
    "read_dicom_instance",
    "timing_pair_is_valid",
    "validate_stack",
    "verify_file_digest",
]
