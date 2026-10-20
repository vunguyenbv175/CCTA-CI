"""CCTA Studio Phase-0 workstation shell (WS02_DESIGN.md Part B).

CLI entry point: ``python -m apps.workstation.cli`` (kept out of this
package's eager imports so ``-m`` execution does not double-import).
"""

from apps.workstation.backup import WorkspaceBackupManager
from apps.workstation.round_trip import RoundTripVerificationEngine, VerificationReport
from apps.workstation.session_store import (
    DerivedStateStoreManager,
    RestoredSession,
    SessionMissingError,
)
from apps.workstation.volume_loader import CanonicalVolumeLoader, compute_measurements
from apps.workstation.workspace import (
    NotAnalyzableError,
    StudyWorkspaceManager,
    WorkstationPaths,
)

__all__ = [
    "CanonicalVolumeLoader",
    "DerivedStateStoreManager",
    "NotAnalyzableError",
    "RestoredSession",
    "RoundTripVerificationEngine",
    "SessionMissingError",
    "StudyWorkspaceManager",
    "VerificationReport",
    "WorkspaceBackupManager",
    "WorkstationPaths",
    "compute_measurements",
]
