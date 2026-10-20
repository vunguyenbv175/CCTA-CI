"""Shared full-pipeline builder for workstation tests (import→derive→save)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from apps.workstation.session_store import (
    DerivedStateStoreManager,
    RestoredSession,
    SessionSnapshot,
)
from apps.workstation.workspace import StudyWorkspaceManager
from core.events import EventLog
from tests.apps.fixture_factory import build_synthetic_study, counter_id_factory


def run_full_pipeline(
    workspace: Path,
    *,
    dicom_dir: Path | None = None,
    id_seed: int = 7,
) -> RestoredSession:
    """Import → derive → save → reopen on a seeded synthetic study."""
    source = dicom_dir or build_synthetic_study(
        workspace.parent / f"{workspace.name}-dicom", slices_per_phase=3
    )
    factory: Callable[[], str] = counter_id_factory(id_seed)
    manager = StudyWorkspaceManager(workspace, id_factory=factory)
    imported = manager.import_study(Path(source))
    store = DerivedStateStoreManager(imported.paths)
    store.save(
        SessionSnapshot(
            study=imported.study,
            series=tuple(item.series for item in imported.series),
            volumes=(),
            artifacts=(),
            payload_documents={},
            provenance=imported.provenance,
            log=EventLog(),
            skipped_instances=imported.skipped,
        )
    )
    session = store.reopen()
    outcome = manager.derive(session)
    documents = dict(session.payload_documents)
    documents.update(outcome.payload_documents)
    store.save(
        SessionSnapshot(
            study=session.study,
            series=session.series,
            volumes=session.volumes + outcome.volumes,
            artifacts=session.artifacts + outcome.artifacts,
            payload_documents=documents,
            provenance=session.provenance + outcome.provenance,
            log=session.log,
            skipped_instances=session.skipped_instances,
        )
    )
    return store.reopen()


def rewrite_manifest(workspace: Path) -> None:
    """Recompute manifest digests after a test tampers with session files."""
    import hashlib
    import json

    from apps.workstation.session_store import MANIFEST_NAME
    from core.digest import canonical_digest

    root = Path(workspace)
    files: dict[str, dict[str, object]] = {}
    for base in (root / "blobs", root / "session"):
        for file_path in sorted(base.rglob("*")):
            if not file_path.is_file() or file_path.suffix in {".tmp"}:
                continue
            relative = file_path.relative_to(root).as_posix()
            if relative == f"session/{MANIFEST_NAME}":
                continue
            files[relative] = {
                "sha256": hashlib.sha256(file_path.read_bytes()).hexdigest(),
                "bytes": file_path.stat().st_size,
            }
    digest = canonical_digest({"files": {k: v["sha256"] for k, v in files.items()}})
    document = {
        "schema": "ccta.session.manifest.v1",
        "saved_at": "2026-01-01T00:00:00Z",
        "session_digest": digest,
        "files": dict(sorted(files.items())),
    }
    (root / "session" / MANIFEST_NAME).write_text(
        json.dumps(document, sort_keys=True, indent=2), encoding="utf-8", newline="\n"
    )


__all__ = ["rewrite_manifest", "run_full_pipeline"]
