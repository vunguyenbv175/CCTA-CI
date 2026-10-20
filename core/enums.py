"""Canonical enums (WS00_DESIGN.md Part B §2.3 and entity schemas).

These are the only enum definitions allowed in the platform; no module may
define its own copy (Master Plan §6.2, Rule B).
"""

from __future__ import annotations

import enum


class ReleaseLane(enum.Enum):
    """Release lane axis (Master Plan §6.7). Independent of ModelStatus."""

    PERSONAL_RESEARCH = "PERSONAL_RESEARCH"
    CLINICAL_CANDIDATE = "CLINICAL_CANDIDATE"
    CLINICAL_RELEASE = "CLINICAL_RELEASE"


class QualityGrade(enum.Enum):
    """Study/vessel/region quality grade (Master Plan §9)."""

    GOOD = "GOOD"
    ACCEPTABLE = "ACCEPTABLE"
    LIMITED = "LIMITED"
    NOT_ANALYZABLE = "NOT_ANALYZABLE"


class Confidence(enum.Enum):
    """Module-defined confidence scale (Master Plan §41)."""

    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    UNRELIABLE = "UNRELIABLE"
    NOT_ANALYZABLE = "NOT_ANALYZABLE"


class ModelStatus(enum.Enum):
    """Registry status axis (Master Plan §38)."""

    EXPERIMENTAL = "EXPERIMENTAL"
    CHALLENGER = "CHALLENGER"
    CHAMPION = "CHAMPION"
    DEPRECATED = "DEPRECATED"
    BLOCKED = "BLOCKED"


class ReviewState(enum.Enum):
    """Artifact review lifecycle (Master Plan §12, §43).

    Transitions happen only through append-only events; ``CONFIRMED_FINAL``
    requires an explicit human actor.
    """

    PREDICTED = "PREDICTED"
    ACCEPTED = "ACCEPTED"
    CORRECTED = "CORRECTED"
    REJECTED = "REJECTED"
    CONFIRMED_FINAL = "CONFIRMED_FINAL"


class ArtifactStatus(enum.Enum):
    """Artifact lifecycle status (Master Plan §43, H-010). ``STALE`` marks
    dependents of invalidated geometry until recomputation."""

    ACTIVE = "ACTIVE"
    SUPERSEDED = "SUPERSEDED"
    INVALIDATED = "INVALIDATED"
    STALE = "STALE"


class PrivacyClass(enum.Enum):
    """Study privacy class (Part B §3.1, P9 PHI minimization)."""

    IDENTIFIABLE = "IDENTIFIABLE"
    DE_IDENTIFIED = "DE_IDENTIFIED"
    SYNTHETIC = "SYNTHETIC"
    PUBLIC_DATASET = "PUBLIC_DATASET"


class ImportSource(enum.Enum):
    """Where the original DICOM bytes came from (Part B §3.1)."""

    LOCAL_FOLDER = "LOCAL_FOLDER"
    MEDIA = "MEDIA"
    PACS = "PACS"
    DATASET_SNAPSHOT = "DATASET_SNAPSHOT"


class ExamType(enum.Enum):
    """Exam classification recorded at import (Part B §3.1)."""

    CCTA = "CCTA"
    CALCIUM_SCORE = "CALCIUM_SCORE"
    CCTA_PLUS_CACS = "CCTA_PLUS_CACS"
    OTHER = "OTHER"
    UNKNOWN = "UNKNOWN"


class ContrastPresence(enum.Enum):
    """Tri-state contrast marker: ``bool | UNKNOWN`` in the design."""

    PRESENT = "PRESENT"
    ABSENT = "ABSENT"
    UNKNOWN = "UNKNOWN"


class SeriesClass(enum.Enum):
    """Series classification (Master Plan §8)."""

    AXIAL_SOURCE = "AXIAL_SOURCE"
    MPR_REFORMAT = "MPR_REFORMAT"
    VR_DERIVED = "VR_DERIVED"
    LOCALIZER = "LOCALIZER"
    SEG = "SEG"
    SR = "SR"
    PARAMETRIC_MAP = "PARAMETRIC_MAP"
    OTHER = "OTHER"


class ContrastClass(enum.Enum):
    """Series contrast phase classification (Part B §3.2)."""

    NON_CONTRAST = "NON_CONTRAST"
    ARTERIAL = "ARTERIAL"
    VENOUS = "VENOUS"
    DELAYED = "DELAYED"
    UNKNOWN = "UNKNOWN"


class VolumeKind(enum.Enum):
    """Canonical volume kind (Part B §3.3). Variants never mutate in place."""

    CANONICAL = "CANONICAL"
    RESAMPLED = "RESAMPLED"
    CROPPED = "CROPPED"
    MASKED = "MASKED"
    REGISTERED = "REGISTERED"


class ArtifactKind(enum.Enum):
    """Derived artifact kinds (Part B §3.4).

    Domain objects (lesion sets, reports...) live inside payload schemas;
    ``PHYSIOLOGY_SURROGATE`` is the dedicated kind for AI FFR surrogates so
    they can never be confused with deterministic physics results.
    """

    SEGMENTATION = "SEGMENTATION"
    CENTERLINE = "CENTERLINE"
    MESH = "MESH"
    SURFACE = "SURFACE"
    LANDMARK_SET = "LANDMARK_SET"
    CORONARY_GRAPH = "CORONARY_GRAPH"
    LESION_SET = "LESION_SET"
    MEASUREMENT_SET = "MEASUREMENT_SET"
    QC_ASSESSMENT = "QC_ASSESSMENT"
    CLASSIFICATION_CANDIDATE = "CLASSIFICATION_CANDIDATE"
    PHYSIOLOGY_RESULT = "PHYSIOLOGY_RESULT"
    PHYSIOLOGY_SURROGATE = "PHYSIOLOGY_SURROGATE"
    REPORT_DRAFT = "REPORT_DRAFT"
    COORDINATE_TRANSFORM = "COORDINATE_TRANSFORM"
    REGISTRATION = "REGISTRATION"
    CORRECTION = "CORRECTION"
    EXPORT_PACKAGE = "EXPORT_PACKAGE"
    MPR_VIEW = "MPR_VIEW"


class ProducerType(enum.Enum):
    """Who produced an artifact (Part B §3.4 producer block)."""

    MODEL = "MODEL"
    DETERMINISTIC_ENGINE = "DETERMINISTIC_ENGINE"
    RULE_ENGINE = "RULE_ENGINE"
    PHYSICS_BACKEND = "PHYSICS_BACKEND"
    LLM = "LLM"
    HUMAN = "HUMAN"
    ADAPTER = "ADAPTER"


class ActorType(enum.Enum):
    """Provenance actor types (Part B §3.5)."""

    HUMAN = "HUMAN"
    MODEL = "MODEL"
    ENGINE = "ENGINE"
    ADAPTER = "ADAPTER"
    RULE_ENGINE = "RULE_ENGINE"
    LLM = "LLM"


class ProvenanceEventType(enum.Enum):
    """Append-only provenance event types (Part B §3.5)."""

    IMPORTED = "IMPORTED"
    DISCOVERED = "DISCOVERED"
    CLASSIFIED = "CLASSIFIED"
    DERIVED = "DERIVED"
    RECOMPUTED = "RECOMPUTED"
    REVIEWED = "REVIEWED"
    CORRECTED = "CORRECTED"
    INVALIDATED = "INVALIDATED"
    SUPERSEDED = "SUPERSEDED"
    EXPORTED = "EXPORTED"
    LANE_PROMOTED = "LANE_PROMOTED"


class CoordinateSystem(enum.Enum):
    """Canonical world coordinate system (Master Plan §7).

    The canonical world frame IS the DICOM patient coordinate system; no
    external framework coordinate system may leak into canonical records.
    """

    DICOM_PATIENT = "DICOM_PATIENT"


class Handedness(enum.Enum):
    """Frame handedness label (Master Plan §7)."""

    LPS = "LPS"


class Units(enum.Enum):
    """Spatial/intensity units (Master Plan §7)."""

    MM = "MM"
    HU = "HU"


class SpaceKind(enum.StrEnum):
    """Coordinate spaces in the mandatory transform chain (§7):

    voxel <-> world <-> mesh <-> centerline.
    """

    VOXEL = "voxel"
    WORLD = "world"
    MESH = "mesh"
    CENTERLINE = "centerline"
