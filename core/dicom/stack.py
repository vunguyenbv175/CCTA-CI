"""Public series-stack admission seam for Imaging Core B6."""

from core.dicom.classification import (
    DUPLICATE_POSITION_TOLERANCE_MM,
    IN_PLANE_DRIFT_TOLERANCE_MM,
    IN_PLANE_SPACING_TOLERANCE_MM,
    INTER_SLICE_GAP_TOLERANCE_MM,
    ORIENTATION_TOLERANCE,
    PARALLEL_DIRECTION_TOLERANCE,
    StackAdmissionError,
    StackAdmissionResult,
    admit_stack,
    validate_stack,
)

__all__ = [
    "DUPLICATE_POSITION_TOLERANCE_MM",
    "IN_PLANE_DRIFT_TOLERANCE_MM",
    "IN_PLANE_SPACING_TOLERANCE_MM",
    "INTER_SLICE_GAP_TOLERANCE_MM",
    "ORIENTATION_TOLERANCE",
    "PARALLEL_DIRECTION_TOLERANCE",
    "StackAdmissionError",
    "StackAdmissionResult",
    "admit_stack",
    "validate_stack",
]
